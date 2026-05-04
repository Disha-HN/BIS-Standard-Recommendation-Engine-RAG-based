"""
inference.py — Judge Entry Point for BIS Standards Recommendation Engine

This is the critical evaluation script. Judges run:
    python inference.py --input hidden_private_dataset.json --output team_results.json

The script:
  1. Loads all models and indexes ONCE before the query loop.
  2. Processes each query through the full RAG pipeline.
  3. Writes strict JSON output: [{"id", "retrieved_standards", "latency_seconds"}, ...]

Any crash here = 0 points on automated metrics. All external calls are
wrapped in try/except with graceful fallback.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Resolve project root so this script works when called from any directory ──
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Default paths (relative to project root) ──────────────────────────────────
CHUNKS_PATH = str(PROJECT_ROOT / "data" / "chunks.json")
BM25_PATH = str(PROJECT_ROOT / "indexes" / "bm25.pkl")
FAISS_PATH = str(PROJECT_ROOT / "indexes" / "faiss.index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum seconds between LLM calls to stay under Groq free-tier rate limit.
# Free tier: 6000 TPM. Each call ~1200 tokens → max 5 calls/min.
# With 3s read timeout, each call takes at most 3s → 5 calls in 15s = fine.
# Override via LLM_REQUEST_INTERVAL env var for paid tiers.
_env_interval = os.environ.get("LLM_REQUEST_INTERVAL", "")
try:
    LLM_REQUEST_INTERVAL = float(_env_interval) if _env_interval else 1.5
except ValueError:
    logger.warning(f"Invalid LLM_REQUEST_INTERVAL value {_env_interval!r}; using default 1.5s")
    LLM_REQUEST_INTERVAL = 1.5

# Maximum allowed query length (characters) — enforced in both CLI and UI paths
MAX_QUERY_LENGTH = 2000


def _dedupe_and_clean(
    raw_codes: List[str],
    canonicalize_fn,
    normalize_fn,
    is_valid_fn,
    max_results: int = 5,
) -> List[str]:
    """
    Canonicalize, validate, and deduplicate a list of IS codes.

    - Drops garbage codes (e.g. 'is 100', empty strings)
    - Normalizes format to 'IS NNN: YYYY'
    - Removes duplicates that differ only in spacing/case
    - Caps output at max_results

    Args:
        raw_codes: Raw IS code strings from LLM or retriever.
        canonicalize_fn: Function to normalize display format.
        normalize_fn: Function to normalize for comparison.
        is_valid_fn: Function to validate a code is a real IS standard.
        max_results: Maximum number of codes to return.

    Returns:
        Cleaned, deduplicated list of canonical IS code strings.
    """
    seen: set = set()
    result: List[str] = []
    for code in raw_codes:
        if not code or not is_valid_fn(code):
            if code:
                logger.warning(f"Dropping invalid std_code: {code!r}")
            continue
        norm = normalize_fn(code)
        if norm in seen:
            logger.debug(f"Dropping duplicate std_code: {code!r}")
            continue
        seen.add(norm)
        result.append(canonicalize_fn(code))
        if len(result) == max_results:
            break
    return result


def load_input(input_path: str) -> List[Dict[str, Any]]:
    """
    Load the input JSON file.

    Args:
        input_path: Path to input JSON (list of {id, query} dicts).

    Returns:
        List of query dicts.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} queries from {input_path}")
    return data


def run_inference(
    queries: List[Dict[str, Any]],
    retriever,
    groq_api_key: str,
    gemini_api_key: str,
) -> List[Dict[str, Any]]:
    """
    Run the full RAG pipeline over all queries.

    Args:
        queries: List of {id, query} dicts.
        retriever: Loaded HybridRetriever instance.
        groq_api_key: Groq API key string (empty string = no LLM).
        gemini_api_key: Gemini API key string (empty string = no LLM).

    Returns:
        List of {id, retrieved_standards, latency_seconds} dicts.
        retrieved_standards are deduplicated, canonicalized, and capped at 5.
    """
    from src.generator import generate, canonicalize_std_code, normalize_std_code, is_valid_std_code

    results = []
    total = len(queries)
    last_llm_call_time: float = 0.0  # track last LLM call for rate-limit throttling

    for i, item in enumerate(queries, start=1):
        query_id = item.get("id", f"q{i:03d}")
        query_text = item.get("query", "").strip()

        logger.info(f"[{i}/{total}] Processing: {query_id}")

        start_time = time.time()
        retrieved_standards: List[str] = []

        try:
            if not query_text:
                logger.warning(f"Empty query for {query_id}; skipping.")
            else:
                # Enforce max query length
                if len(query_text) > MAX_QUERY_LENGTH:
                    logger.warning(
                        f"Query {query_id} exceeds max length "
                        f"({len(query_text)} > {MAX_QUERY_LENGTH} chars); truncating."
                    )
                    query_text = query_text[:MAX_QUERY_LENGTH]

                # Retrieve top-5 chunks
                chunks = retriever.retrieve(query_text)

                # Throttle LLM calls to avoid 429 rate-limit errors
                if groq_api_key or gemini_api_key:
                    elapsed_since_last = time.time() - last_llm_call_time
                    if elapsed_since_last < LLM_REQUEST_INTERVAL:
                        time.sleep(LLM_REQUEST_INTERVAL - elapsed_since_last)

                # Generate recommendations (LLM or fallback).
                # Hard latency guard: skip LLM if retrieval already took > 1s,
                # keeping total per-query time well under the 5s target.
                retrieval_time = time.time() - start_time
                use_llm = (groq_api_key or gemini_api_key) and retrieval_time < 1.0

                recommendations = generate(
                    query_text,
                    chunks,
                    groq_api_key=groq_api_key if use_llm else "",
                    gemini_api_key=gemini_api_key if use_llm else "",
                )
                if use_llm:
                    last_llm_call_time = time.time()

                # Extract IS codes — from LLM output if available, else from chunk metadata
                if recommendations:
                    raw_codes = [r["std_code"] for r in recommendations if r.get("std_code")]
                else:
                    # Pure retrieval fallback: use chunk metadata codes
                    raw_codes = [c["std_code"] for c in chunks if c.get("std_code")]

                retrieved_standards = _dedupe_and_clean(
                    raw_codes, canonicalize_std_code, normalize_std_code, is_valid_std_code
                )

        except Exception as e:
            logger.error(f"Error processing {query_id}: {e}")
            # Best-effort fallback: try retrieval-only
            try:
                chunks = retriever.retrieve(query_text)
                raw_codes = [c["std_code"] for c in chunks if c.get("std_code")]
                retrieved_standards = _dedupe_and_clean(
                    raw_codes, canonicalize_std_code, normalize_std_code, is_valid_std_code
                )
            except Exception as e2:
                logger.error(f"Fallback also failed for {query_id}: {e2}")
                retrieved_standards = []

        latency = time.time() - start_time

        results.append(
            {
                "id": query_id,
                "retrieved_standards": retrieved_standards,  # already capped at 5
                "latency_seconds": round(latency, 4),
            }
        )

        logger.info(
            f"  → {len(retrieved_standards)} standards in {latency:.2f}s: "
            f"{retrieved_standards[:3]}"
        )

    return results


def save_output(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write results to JSON file in strict schema.

    Schema: [{"id": str, "retrieved_standards": [str], "latency_seconds": float}]

    Args:
        results: List of result dicts.
        output_path: Path to write output JSON.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine — Inference Script"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with queries",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON results",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Skip LLM enrichment and use retrieval-only mode (faster, no API key needed)",
    )
    args = parser.parse_args()

    # ── Validate input file ────────────────────────────────────────────────────
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # ── Load API keys from environment ─────────────────────────────────────────
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    if args.no_llm:
        groq_api_key = ""
        gemini_api_key = ""
        logger.info("--no-llm flag set: running in retrieval-only mode.")
    elif not groq_api_key and not gemini_api_key:
        logger.warning(
            "No LLM API keys found (GROQ_API_KEY / GEMINI_API_KEY). "
            "Will use retrieval-only fallback. Pass --no-llm to suppress this warning."
        )

    # ── Load models and indexes ONCE ──────────────────────────────────────────
    logger.info("Loading embedding model (one-time)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

    logger.info("Loading retriever indexes (one-time)...")
    try:
        from src.retriever import HybridRetriever
        retriever = HybridRetriever(CHUNKS_PATH, BM25_PATH, FAISS_PATH, model)
    except FileNotFoundError as e:
        logger.error(
            f"Index file not found: {e}\n"
            "Run 'python src/indexer.py' to build indexes first."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        sys.exit(1)

    # ── Load queries ───────────────────────────────────────────────────────────
    try:
        queries = load_input(args.input)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)

    # ── Run inference ──────────────────────────────────────────────────────────
    results = run_inference(queries, retriever, groq_api_key, gemini_api_key)

    # ── Save output ────────────────────────────────────────────────────────────
    try:
        save_output(results, args.output)
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_latency = sum(r["latency_seconds"] for r in results)
    avg_latency = total_latency / len(results) if results else 0
    logger.info(
        f"\nInference complete: {len(results)} queries, "
        f"avg latency = {avg_latency:.2f}s"
    )


if __name__ == "__main__":
    main()
