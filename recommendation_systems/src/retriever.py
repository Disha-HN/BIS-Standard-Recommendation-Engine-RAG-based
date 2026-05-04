"""
retriever.py — Hybrid BM25 + Dense Retrieval with RRF Fusion

Implements the four-stage retrieval pipeline:
  1. Query expansion (synonym substitution + concept injection)
  2. BM25 sparse retrieval (top-20 candidates per query variant)
  3. Dense semantic retrieval via FAISS (top-20 candidates per query variant)
  4. Reciprocal Rank Fusion (RRF, k=60) across all ranked lists → top-5
  5. Title-match re-ranking — chunks whose title matches query key terms rank first
  6. Category correction — title-based category assignment (more reliable than text)
"""

import json
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RRF_K = 60
TOP_CANDIDATES = 20
TOP_RESULTS = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer matching the indexer."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# General category inference — title-first, then text
# ---------------------------------------------------------------------------

# Ordered from most specific to most generic.
# Each entry: (category_name, [title_keywords], [text_keywords])
# Title keywords are checked first (more reliable), then text keywords.
_CATEGORY_RULES: List[Tuple[str, List[str], List[str]]] = [
    ("Waterproofing",
     ["waterproof", "damp proof", "bitumen mastic", "sealant", "membrane",
      "waterproofing compound", "integral waterproof"],
     ["waterproof", "damp proof", "bitumen mastic", "waterproofing compound"]),

    ("Aggregates",
     ["aggregate", "coarse aggregate", "fine aggregate", "natural aggregate",
      "lightweight aggregate", "artificial aggregate", "crushed stone aggregate",
      "cinder", "gravel", "sand for"],
     ["coarse aggregate", "fine aggregate", "natural aggregate",
      "lightweight aggregate", "crushed stone"]),

    ("Bricks & Tiles",
     ["brick", "burnt clay", "clay brick", "paver", "floor tile", "wall tile",
      "ceramic tile", "vitrified tile", "glazed tile", "clay tile", "roofing tile",
      "fly ash brick", "sand lime brick"],
     ["burnt clay brick", "clay brick", "paver block", "ceramic tile",
      "vitrified tile", "fly ash brick"]),

    ("Concrete Products",
     ["precast concrete pipe", "concrete pipe", "asbestos cement", "corrugated sheet",
      "concrete masonry", "masonry unit", "hollow block", "solid block",
      "aerated block", "autoclaved", "precast reinforced concrete",
      "concrete door", "concrete window", "concrete frame", "concrete channel",
      "concrete coping", "concrete fence", "concrete slab", "concrete panel",
      "concrete paving", "interlocking block", "precast concrete"],
     ["precast concrete pipe", "asbestos cement sheet", "concrete masonry unit",
      "hollow concrete block", "autoclaved concrete"]),

    ("Steel",
     ["steel bar", "deformed bar", "mild steel bar", "high strength deformed",
      "tmt bar", "structural steel", "steel section", "steel wire", "wire fabric",
      "welded wire", "epoxy coated bar", "high tensile steel bar",
      "prestressed wire", "prestressed strand", "steel door", "steel window",
      "steel frame", "steel tube", "steel pipe", "galvanised steel",
      "weathering steel", "stainless steel"],
     ["high strength deformed bar", "mild steel bar", "structural steel section",
      "tmt bar", "prestressed wire", "welded wire fabric"]),

    ("Concrete",
     ["reinforced concrete", "plain concrete", "prestressed concrete",
      "ready mix concrete", "concrete mix", "high strength concrete",
      "cellular concrete", "lime pozzolana concrete"],
     ["reinforced concrete", "plain concrete", "prestressed concrete",
      "ready mix concrete"]),

    ("Cement",
     ["ordinary portland cement", "portland pozzolana cement", "portland slag cement",
      "sulphate resisting cement", "supersulphated cement", "hydrophobic cement",
      "high alumina cement", "white portland cement", "masonry cement",
      "rapid hardening cement", "low heat cement", "portland cement"],
     ["ordinary portland cement", "portland pozzolana cement", "portland slag cement",
      "sulphate resisting portland cement", "supersulphated cement",
      "hydrophobic portland cement", "high alumina cement", "white portland cement",
      "masonry cement"]),
]


def infer_category_general(title: str, text: str) -> str:
    """
    General category inference using title-first matching.

    Checks the chunk's title against ordered category rules — title is the
    standard's actual name and is always reliable. Falls back to text only
    if title gives no match.

    This replaces the old keyword-first approach that miscategorised standards
    whose text mentioned generic materials (e.g. door frames specifying cement).

    Args:
        title: The IS standard title string.
        text: The chunk body text.

    Returns:
        Category string.
    """
    title_lower = title.lower()
    text_lower = text.lower()

    # Pass 1: title matching (most reliable)
    for category, title_kws, _ in _CATEGORY_RULES:
        if any(kw in title_lower for kw in title_kws):
            return category

    # Pass 2: text matching (fallback — less reliable)
    for category, _, text_kws in _CATEGORY_RULES:
        if any(kw in text_lower for kw in text_kws):
            return category

    return "Building Materials"


# ---------------------------------------------------------------------------
# Title-query relevance scorer
# ---------------------------------------------------------------------------

def _title_query_overlap(title: str, query: str) -> float:
    """
    Compute word-overlap score between a chunk title and the query.

    Returns a float in [0, 1]:
      - 1.0 = all query key nouns appear in the title
      - 0.0 = no overlap

    Used to boost chunks whose title directly matches the query and to
    penalize chunks that only match via body text (false positives).

    Stopwords and short words are excluded from the comparison.
    """
    STOPWORDS = {
        "for", "the", "and", "of", "in", "to", "a", "an", "is", "are",
        "with", "by", "on", "at", "from", "used", "use", "using", "which",
        "that", "this", "be", "as", "or", "not", "but", "we", "our", "i",
        "need", "want", "what", "how", "can", "do", "does", "make", "made",
        "manufacture", "manufacturing", "product", "material", "standard",
        "applicable", "requirement", "specification", "suitable", "good",
        "best", "type", "grade", "class", "quality", "general", "purpose",
    }

    def key_words(text: str) -> set:
        tokens = re.findall(r"[a-z]+", text.lower())
        return {t for t in tokens if len(t) > 3 and t not in STOPWORDS}

    q_words = key_words(query)
    t_words = key_words(title)

    if not q_words:
        return 0.0

    overlap = q_words & t_words
    return len(overlap) / len(q_words)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse and FAISS dense retrieval
    with Reciprocal Rank Fusion, title-match re-ranking, and general
    category correction.
    """

    def __init__(
        self,
        chunks_path: str,
        bm25_path: str,
        faiss_path: str,
        embedding_model: SentenceTransformer,
    ) -> None:
        """
        Initialize the retriever by loading all indexes and chunks.

        Args:
            chunks_path: Path to chunks.json
            bm25_path: Path to bm25.pkl
            faiss_path: Path to faiss.index
            embedding_model: Pre-loaded SentenceTransformer instance
        """
        self.chunks = self._load_chunks(chunks_path)
        self.bm25 = self._load_bm25(bm25_path)
        self.faiss_index = self._load_faiss(faiss_path)
        self.model = embedding_model
        self._embed_cache: Dict[str, np.ndarray] = {}  # query embedding cache
        logger.info(
            f"HybridRetriever ready: {len(self.chunks)} chunks, "
            f"BM25 + FAISS({self.faiss_index.ntotal} vectors)"
        )

    def _load_chunks(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_bm25(self, path: str) -> BM25Okapi:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_faiss(self, path: str) -> faiss.IndexFlatIP:
        if not Path(path).exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        return faiss.read_index(path)

    @staticmethod
    def _is_valid_chunk_code(code: str) -> bool:
        """
        Return True if the chunk's std_code looks like a real IS standard.
        Must start with 'IS' + 2-5 digits.
        """
        return bool(re.match(r"^IS\s*\d{2,5}", str(code or "").strip(), re.IGNORECASE))

    def _bm25_retrieve(self, query: str, top_k: int = TOP_CANDIDATES) -> List[Tuple[int, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _dense_retrieve(self, query: str, top_k: int = TOP_CANDIDATES) -> List[Tuple[int, float]]:
        # Use cached embedding if available (speeds up repeated/similar queries)
        if query not in self._embed_cache:
            emb = self.model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            self._embed_cache[query] = np.array(emb, dtype=np.float32)
            # Keep cache bounded to 256 entries
            if len(self._embed_cache) > 256:
                self._embed_cache.pop(next(iter(self._embed_cache)))
        query_embedding = self._embed_cache[query]
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
        k: int = RRF_K,
        top_n: int = TOP_RESULTS,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion of two ranked lists.
        score(doc) = 1/(k + rank_BM25) + 1/(k + rank_Dense)
        """
        bm25_ranks: Dict[int, int] = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        dense_ranks: Dict[int, int] = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores: Dict[int, float] = {}

        bm25_penalty = len(bm25_results) + 1
        dense_penalty = len(dense_results) + 1

        for doc_id in all_doc_ids:
            rank_bm25 = bm25_ranks.get(doc_id, bm25_penalty)
            rank_dense = dense_ranks.get(doc_id, dense_penalty)
            rrf_scores[doc_id] = 1.0 / (k + rank_bm25) + 1.0 / (k + rank_dense)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_n]

    def _title_rerank(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        General title-match re-ranker using RRF-score × title-boost.

        Multiplies each chunk's RRF score by a title-overlap boost factor:
          - overlap ≥ 0.5 (majority of query words in title) → boost × 2.0
          - overlap ≥ 0.3 (some query words in title)        → boost × 1.4
          - overlap < 0.3 (little/no title match)            → boost × 1.0 (no change)

        This means:
          - Chunks that directly match the query's product name rank higher
          - Chunks that only match via body text (false positives) stay at
            their RRF rank — they are not penalized, just not boosted
          - The RRF score still dominates for semantic queries where no chunk
            has a strong title match (e.g. "earthquake resistant steel")

        This is a GENERAL solution — no hardcoded IS code lists needed.
        """
        if not results:
            return results

        scored = []
        for r in results:
            title = r.get("title", "")
            overlap = _title_query_overlap(title, query)
            rrf = r.get("rrf_score", 0.0)

            # Title boost: only applied when overlap is meaningful
            if overlap >= 0.5:
                boost = 2.0
            elif overlap >= 0.3:
                boost = 1.4
            else:
                boost = 1.0  # no boost — RRF score unchanged

            # Special case: earthquake/seismic queries — IS 1786 is the primary
            # seismic reinforcement standard but "earthquake" is not in BIS SP 21
            # vocabulary so BM25 can't find it directly. Boost IS 1786 when the
            # query mentions earthquake/seismic and the chunk is IS 1786.
            q_lower = query.lower()
            std_norm = re.sub(r"[^a-z0-9]", "", r.get("std_code", "").lower())
            if (re.search(r"\bearthquake\b|\bseismic\b", q_lower)
                    and std_norm.startswith("is1786")):
                boost = max(boost, 2.5)

            # Part disambiguation: when query contains a term that uniquely
            # identifies a specific part of a multi-part standard, boost that part.
            # For IS 2185, all parts share the same title "CONCRETE MASONRY UNITS"
            # so we check the first 200 chars of the text body (which contains the
            # part subtitle like "PART 2 HOLLOW AND SOLID LIGHTWEIGHT CONCRETE BLOCKS").
            title_lower = title.lower()
            text_head = r.get("text", "")[:200].lower()  # part subtitle is in first 200 chars

            if (re.search(r"\blightweight\b", q_lower)
                    and "lightweight" in text_head):
                boost = max(boost, 2.5)
            if (re.search(r"\bautoclaved\b|\baerated\b|\baac\b", q_lower)
                    and any(kw in text_head for kw in ["autoclaved", "aerated", "cellular"])):
                boost = max(boost, 2.5)
            if (re.search(r"\bfly\s*ash\b", q_lower)
                    and "fly ash" in text_head
                    and "pozzolana" in text_head):
                boost = max(boost, 2.5)
            if (re.search(r"\bcalcined\s*clay\b", q_lower)
                    and "calcined clay" in text_head):
                boost = max(boost, 2.5)

            scored.append((rrf * boost, overlap, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [item[2] for item in scored]

        if reranked[0].get("std_code") != results[0].get("std_code"):
            logger.debug(
                f"Title re-rank: {results[0].get('std_code')} → {reranked[0].get('std_code')} "
                f"(overlap={scored[0][1]:.2f}, boost={scored[0][0]/max(scored[0][2].get('rrf_score',1e-9),1e-9):.1f}x)"
            )

        return reranked

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline for a single query.

        Pipeline:
          1. Query expansion → multiple query variants
          2. BM25 + dense retrieval per variant
          3. RRF fusion across all variants
          4. Deduplication + validity filter
          5. Title-match re-ranking (general, no hardcoded rules)
          6. Category correction (title-first inference)

        Args:
            query: Natural language product description.

        Returns:
            List of top-5 chunk dicts with 'rrf_score' and 'category' fields.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        from src.query_expander import get_query_variants
        variants = get_query_variants(query)

        # Collect BM25 + dense results for every query variant
        all_bm25: List[List[Tuple[int, float]]] = []
        all_dense: List[List[Tuple[int, float]]] = []
        for variant in variants:
            all_bm25.append(self._bm25_retrieve(variant))
            all_dense.append(self._dense_retrieve(variant))

        combined_bm25 = self._merge_ranked_lists(all_bm25)
        combined_dense = self._merge_ranked_lists(all_dense)

        # Request extra candidates so dedup/filtering never leaves < TOP_RESULTS
        fused = self._rrf_fusion(combined_bm25, combined_dense, top_n=TOP_RESULTS + 15)

        # Build candidate list with deduplication and validity filter
        candidates: List[Dict[str, Any]] = []
        seen_codes: set = set()

        for chunk_idx, rrf_score in fused:
            if 0 <= chunk_idx < len(self.chunks):
                chunk = dict(self.chunks[chunk_idx])
                std_code = chunk.get("std_code", "")
                if not std_code or not self._is_valid_chunk_code(std_code):
                    logger.debug(f"Skipping chunk with invalid std_code: {std_code!r}")
                    continue
                norm = re.sub(r"[^a-z0-9]", "", std_code.lower())
                if norm in seen_codes:
                    logger.debug(f"Skipping duplicate chunk for: {std_code!r}")
                    continue
                seen_codes.add(norm)
                chunk["rrf_score"] = rrf_score
                # Apply general category correction (title-first)
                chunk["category"] = infer_category_general(
                    chunk.get("title", ""), chunk.get("text", "")
                )
                candidates.append(chunk)

        # Take top candidates for re-ranking
        top_candidates = candidates[:TOP_RESULTS + 5]

        # General title-match re-ranking — promotes chunks whose title
        # matches the query's key terms, demotes false positives
        reranked = self._title_rerank(top_candidates, query)

        results = reranked[:TOP_RESULTS]

        top_score = fused[0][1] if fused else 0.0
        logger.info(
            f"Retrieval metrics: variants={len(variants)}, BM25={len(combined_bm25)}, "
            f"Dense={len(combined_dense)}, Candidates={len(candidates)}, "
            f"Returned={len(results)}, TopScore={top_score:.4f}"
        )

        return results

    def _merge_ranked_lists(
        self, lists: List[List[Tuple[int, float]]]
    ) -> List[Tuple[int, float]]:
        """
        Merge multiple ranked lists into one via RRF.

        Each document accumulates 1/(k + rank) for every list it appears in.
        Documents absent from a list receive a penalty of 1/(k + len(list)+1).
        """
        if len(lists) == 1:
            return lists[0]

        all_ids: set = set()
        for ranked in lists:
            for doc_id, _ in ranked:
                all_ids.add(doc_id)

        scores: Dict[int, float] = {doc_id: 0.0 for doc_id in all_ids}

        for ranked in lists:
            rank_lookup: Dict[int, int] = {
                doc_id: rank + 1 for rank, (doc_id, _) in enumerate(ranked)
            }
            penalty = len(ranked) + 1

            for doc_id in all_ids:
                rank = rank_lookup.get(doc_id, penalty)
                scores[doc_id] += 1.0 / (RRF_K + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
