"""
retriever.py — Hybrid BM25 + Dense Retrieval with RRF Fusion

Implements the four-stage retrieval pipeline:
  1. Query expansion (synonym substitution + concept injection)
  2. BM25 sparse retrieval (top-20 candidates per query variant)
  3. Dense semantic retrieval via FAISS (top-20 candidates per query variant)
  4. Reciprocal Rank Fusion (RRF, k=60) across all ranked lists → top-5
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


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse and FAISS dense retrieval
    with Reciprocal Rank Fusion.
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
        Must start with 'IS' + 2-5 digits and contain a 4-digit year.
        Rejects ingestion artifacts like 'is 100' (no year).
        """
        return bool(re.match(r"^IS\s*\d{2,5}.*\b\d{4}\b", str(code or "").strip(), re.IGNORECASE))

    def _bm25_retrieve(self, query: str, top_k: int = TOP_CANDIDATES) -> List[Tuple[int, float]]:
        """
        BM25 retrieval.

        Returns:
            List of (chunk_index, score) tuples sorted by score descending.
        """
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _dense_retrieve(self, query: str, top_k: int = TOP_CANDIDATES) -> List[Tuple[int, float]]:
        """
        Dense semantic retrieval via FAISS.

        Returns:
            List of (chunk_index, score) tuples sorted by score descending.
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)
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
        A document missing from one list is penalised with rank = len(that_list) + 1,
        which is one position beyond the worst actual rank in that list.

        Args:
            bm25_results: Ranked list from BM25 (chunk_idx, score).
            dense_results: Ranked list from dense retrieval (chunk_idx, score).
            k: RRF constant (default 60).
            top_n: Number of results to return.

        Returns:
            Top-n (chunk_idx, rrf_score) tuples sorted by RRF score descending.
        """
        bm25_ranks: Dict[int, int] = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        dense_ranks: Dict[int, int] = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores: Dict[int, float] = {}

        # Penalty rank for a doc missing from one list: one position beyond the list length
        bm25_penalty = len(bm25_results) + 1
        dense_penalty = len(dense_results) + 1

        for doc_id in all_doc_ids:
            rank_bm25 = bm25_ranks.get(doc_id, bm25_penalty)
            rank_dense = dense_ranks.get(doc_id, dense_penalty)
            rrf_scores[doc_id] = 1.0 / (k + rank_bm25) + 1.0 / (k + rank_dense)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_n]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline for a single query.

        Runs retrieval on both the original query and an expanded variant
        (via query_expander), then RRF-merges all ranked lists together.
        This bridges the gap between user language and BIS SP 21 vocabulary.

        Args:
            query: Natural language product description.

        Returns:
            List of top-5 chunk dicts with added 'rrf_score' field.
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

        # Flatten: merge all ranked lists via RRF
        # Each list gets its own RRF contribution; a doc appearing in multiple
        # lists accumulates score from each, naturally boosting consensus hits.
        combined_bm25 = self._merge_ranked_lists(all_bm25)
        combined_dense = self._merge_ranked_lists(all_dense)

        # Final RRF fusion of merged BM25 vs merged dense
        fused = self._rrf_fusion(combined_bm25, combined_dense, top_n=TOP_RESULTS + 5)

        results = []
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
                results.append(chunk)
                if len(results) == TOP_RESULTS:
                    break

        top_score = fused[0][1] if fused else 0.0
        logger.info(
            f"Retrieval metrics: variants={len(variants)}, BM25={len(combined_bm25)}, "
            f"Dense={len(combined_dense)}, Fused={len(results)}, TopScore={top_score:.4f}"
        )

        return results

    def _merge_ranked_lists(
        self, lists: List[List[Tuple[int, float]]]
    ) -> List[Tuple[int, float]]:
        """
        Merge multiple ranked lists into one via RRF, then re-rank by combined score.

        Used to combine results from multiple query variants before the final
        BM25-vs-dense RRF fusion step.

        Args:
            lists: List of ranked (chunk_idx, score) lists.

        Returns:
            Single merged ranked list sorted by combined RRF score descending.
        """
        if len(lists) == 1:
            return lists[0]

        scores: Dict[int, float] = {}
        penalty = TOP_CANDIDATES + 1
        for ranked in lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            # Docs not in this list get a small penalty contribution
            all_ids = set(scores.keys())
            for doc_id in all_ids:
                if doc_id not in {d for d, _ in ranked}:
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + penalty)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
