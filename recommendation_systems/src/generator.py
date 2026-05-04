"""
generator.py — LLM Prompt Construction and Response Parsing

Assembles retrieved chunks into a structured prompt, calls the LLM
(Groq llama-3.1-8b-instant with Gemini fallback), and parses the
structured JSON response. Anti-hallucination is enforced at the
architectural level: the LLM may only reference standards present in
the provided context.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a BIS compliance expert helping Indian MSEs identify applicable standards. "
    "Return only standards explicitly present in the provided context excerpts. "
    "Never invent or guess IS codes."
)

USER_PROMPT_TEMPLATE = """Product description: {query}

Relevant BIS standard excerpts:
{context}

Return a JSON array. Each element must have exactly these keys:
  "std_code": the IS code exactly as shown in the excerpt (e.g. "IS 269: 1989"),
  "title": the standard title,
  "rationale": 1-2 sentences explaining why this standard applies to the product.

Include 3-5 standards. Only use codes from the excerpts above. Output ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# IS code helpers
# ---------------------------------------------------------------------------

def canonicalize_std_code(code: str) -> str:
    """
    Normalize an IS code to a canonical display format: 'IS 269: 1989'.

    Handles variants like:
      'IS 269 : 1989', 'IS269:1989', 'IS 2185 (PART 1) : 1979',
      'IS 1489 (PART1) : 1991', 'IS 8112:1989'
    """
    code = str(code or "").strip()
    # Collapse internal whitespace
    code = re.sub(r"\s+", " ", code)
    # Normalize "(PART N)" / "(PARTN)" → "(Part N)"
    code = re.sub(
        r"\(\s*[Pp][Aa][Rr][Tt]\s*(\d+)\s*\)",
        lambda m: f"(Part {m.group(1)})",
        code,
    )
    # Normalize spaces around colon before year: " : 1989" → ": 1989"
    code = re.sub(r"\s*:\s*(\d{4})", r": \1", code)
    # Ensure space between "IS" and digits: "IS269" → "IS 269"
    code = re.sub(r"^IS(\d)", r"IS \1", code, flags=re.IGNORECASE)
    # Ensure "IS" prefix is uppercase
    code = re.sub(r"^is\b", "IS", code, flags=re.IGNORECASE)
    return code.strip()


def normalize_std_code(code: str) -> str:
    """Strip all non-alphanumeric chars and lowercase — used for comparison only."""
    return re.sub(r"[^a-z0-9]", "", str(code or "").lower().strip())


# A valid IS code must start with "IS" (optionally space) + 2-5 digits + contain a 4-digit year
# Accepts: "IS 269 : 1989", "IS 2185 (Part 1): 1979", "IS 8112:1989"
# Rejects: "is 100" (no year), "IS 99" (too few digits, no year)
_VALID_IS_CODE_RE = re.compile(r"^IS\s*\d{2,5}.*\b\d{4}\b", re.IGNORECASE)


def is_valid_std_code(code: str) -> bool:
    """Return True only if the code looks like a real IS standard with a year."""
    return bool(_VALID_IS_CODE_RE.match(str(code or "").strip()))


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Long chunks are truncated using a head+tail strategy: keep the first 300
    and last 200 characters so both the standard's opening definition and any
    closing scope/application notes are preserved.

    Args:
        chunks: List of chunk dicts from the retriever.

    Returns:
        Formatted context string.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        std_code = chunk.get("std_code", "Unknown")
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        if len(text) > 600:
            # Head + tail truncation: preserve opening definition and closing scope
            text = text[:300] + " … " + text[-200:]
        lines.append(f"[{i}] Standard: {std_code} — {title}\n{text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_array(raw_response: str) -> Optional[str]:
    """
    Extract a JSON array from the LLM output, tolerating markdown fences and noise.

    Uses bracket-balanced extraction instead of a greedy regex so that nested
    objects inside the array don't cause the match to over-extend or under-extend.
    """
    if not raw_response:
        return None

    response = raw_response.strip()

    # Remove markdown code fences if present
    response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\s*```$", "", response, flags=re.IGNORECASE)
    response = response.strip()

    # Fast path: entire response is already a clean array
    if response.startswith("[") and response.endswith("]"):
        return response

    # Bracket-balanced extraction: find the first '[' and walk forward
    # counting brackets until they balance, then return that slice.
    start = response.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(response[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return response[start : i + 1]

    # Brackets never balanced — return whatever we found from '[' onward as a
    # best-effort attempt (the caller will handle JSON parse errors)
    logger.warning("JSON array brackets never balanced; attempting best-effort extraction")
    return response[start:]


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def call_groq(query: str, context: str, api_key: str) -> Optional[str]:
    """
    Call Groq API (llama-3.1-8b-instant).

    Uses max_retries=0 so rate-limit 429s fail fast and we fall back to
    retrieval-only rather than waiting 10+ seconds for a retry.
    Total request timeout is capped at 8s (connect=3s, read=7s) so a slow
    API response never pushes per-query latency above the 5s target when
    combined with fast retrieval.
    """
    try:
        import httpx
        from groq import Groq  # type: ignore

        # Use a full httpx.Timeout: connect=2s, read=3s — fail fast on slow API
        # so per-query latency stays under 5s even on Groq free tier
        timeout = httpx.Timeout(connect=2.0, read=3.0, write=2.0, pool=2.0)
        http_client = httpx.Client(timeout=timeout)
        client = Groq(api_key=api_key, http_client=http_client, max_retries=0)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(query=query, context=context),
                },
            ],
            max_tokens=400,   # Reduced from 512 — rationale text is short; saves ~0.5s
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return None


def call_gemini(query: str, context: str, api_key: str) -> Optional[str]:
    """Fallback: Call Google Gemini 1.5 Flash API."""
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(query=query, context=context)}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=0.1,
            ),
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(
    raw_response: str, retrieved_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parse LLM JSON response and validate IS codes against retrieved chunks.

    Anti-hallucination guard: any IS code not present in the retrieved
    chunks is silently dropped from the output.

    Args:
        raw_response: Raw string from LLM.
        retrieved_chunks: The chunks passed to the LLM as context.

    Returns:
        List of validated standard dicts with std_code, title, rationale.
        IS codes are canonicalized to 'IS NNN: YYYY' format.
    """
    json_text = extract_json_array(raw_response)
    if not json_text:
        logger.warning("No JSON array found in LLM response")
        return []

    # Clean common trailing commas from LLM output
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return []

    # Build set of allowed IS codes from retrieved chunks (normalized for comparison)
    allowed_codes = {
        normalize_std_code(chunk.get("std_code", ""))
        for chunk in retrieved_chunks
        if chunk.get("std_code")
    }

    validated = []
    seen_normalized: set = set()  # deduplication

    for item in parsed:
        if not isinstance(item, dict):
            continue
        raw_code = str(item.get("std_code", "")).strip()

        # Reject garbage / non-IS codes
        if not is_valid_std_code(raw_code):
            logger.warning(f"Dropped invalid code: {raw_code!r}")
            continue

        normalized = normalize_std_code(raw_code)

        # Anti-hallucination: must be in retrieved context
        if normalized not in allowed_codes:
            logger.debug(f"Dropped hallucinated code: {raw_code}")
            continue

        # Deduplication: skip if we've already added this code
        if normalized in seen_normalized:
            logger.debug(f"Dropped duplicate code: {raw_code}")
            continue

        seen_normalized.add(normalized)
        validated.append(
            {
                "std_code": canonicalize_std_code(raw_code),
                "title": str(item.get("title", "")).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )

    return validated


# ---------------------------------------------------------------------------
# Retrieval-only baseline (primary path, LLM enriches rationale on top)
# ---------------------------------------------------------------------------

def build_retrieval_results(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build result list directly from retrieval metadata.

    This is the primary path — IS codes come from the index (no hallucination
    possible). The LLM is called separately to enrich the rationale text.

    Args:
        retrieved_chunks: Top-N chunks from the retriever.

    Returns:
        List of standard dicts with canonicalized std_code, title, and
        template rationale (replaced by LLM rationale if available).
    """
    seen_normalized: set = set()
    results = []

    for chunk in retrieved_chunks:
        raw_code = chunk.get("std_code", "")

        # Skip invalid or empty codes
        if not raw_code or not is_valid_std_code(raw_code):
            logger.warning(f"Skipping chunk with invalid std_code: {raw_code!r}")
            continue

        normalized = normalize_std_code(raw_code)

        # Deduplicate: same standard may appear in multiple sub-chunks
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        canonical = canonicalize_std_code(raw_code)
        results.append(
            {
                "std_code": canonical,
                "title": chunk.get("title", ""),
                "rationale": (
                    f"This standard ({canonical}) covers "
                    f"{chunk.get('category', 'building materials')} and is relevant "
                    "based on keyword and semantic similarity to the product description."
                ),
            }
        )

        if len(results) == 5:
            break

    return results


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def generate(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate standard recommendations for a query using retrieved context.

    IS codes are ALWAYS taken from retrieved chunk metadata (anti-hallucination,
    fast, deterministic). The LLM is called only to generate rationale text.
    If the LLM fails or is unavailable, a template rationale is used instead.

    Args:
        query: Product description string.
        retrieved_chunks: Top-N chunks from the retriever.
        groq_api_key: Groq API key (or read from GROQ_API_KEY env var).
        gemini_api_key: Gemini API key (or read from GEMINI_API_KEY env var).

    Returns:
        List of up to 5 deduplicated, validated standard recommendation dicts.
    """
    if not retrieved_chunks:
        return []

    # ── Step 1: Build results from retrieval metadata (always fast, no LLM) ──
    retrieval_results = build_retrieval_results(retrieved_chunks)

    # ── Step 2: Try LLM for rationale enrichment only ─────────────────────────
    context = build_context(retrieved_chunks)
    llm_rationale_map: Dict[str, str] = {}

    groq_key = groq_api_key if groq_api_key is not None else os.environ.get("GROQ_API_KEY", "")
    gemini_key = gemini_api_key if gemini_api_key is not None else os.environ.get("GEMINI_API_KEY", "")

    # If caller explicitly passed empty string, respect it (don't fall back to env var)
    if groq_api_key == "":
        groq_key = ""
    if gemini_api_key == "":
        gemini_key = ""

    raw = None
    llm_used = "none"
    try:
        if groq_key:
            raw = call_groq(query, context, groq_key)
            llm_used = "groq" if raw else "none"
        if raw is None and gemini_key:
            raw = call_gemini(query, context, gemini_key)
            llm_used = "gemini" if raw else "none"

        if raw:
            parsed = parse_llm_response(raw, retrieved_chunks)
            for item in parsed:
                code_norm = normalize_std_code(item["std_code"])
                llm_rationale_map[code_norm] = item["rationale"]
    except Exception as e:
        logger.warning(f"LLM enrichment failed, using fallback: {e}")
        llm_used = "fallback"

    # ── Step 3: Merge LLM rationale into retrieval results ────────────────────
    for result in retrieval_results:
        code_norm = normalize_std_code(result["std_code"])
        if code_norm in llm_rationale_map:
            result["rationale"] = llm_rationale_map[code_norm]

    logger.info(f"Generated {len(retrieval_results)} recommendations (LLM: {llm_used})")
    return retrieval_results
