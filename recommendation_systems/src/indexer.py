"""
indexer.py — Build and Serialize BM25 + FAISS Indexes

Loads chunks from chunks.json, builds a BM25 sparse index and a FAISS
dense index using sentence-transformers embeddings, then serializes both
to disk for fast loading at inference time.
"""

import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Build a BM25 index over chunk texts.

    Args:
        chunks: List of chunk dicts with 'text' field.

    Returns:
        Fitted BM25Okapi instance.
    """
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    logger.info(f"Built BM25 index over {len(corpus)} documents")
    return bm25


def build_faiss_index(
    chunks: List[Dict[str, Any]], model: SentenceTransformer
) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner product / cosine similarity) index.

    Args:
        chunks: List of chunk dicts with 'text' field.
        model: Loaded SentenceTransformer model.

    Returns:
        Populated FAISS index.
    """
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Encoding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def save_indexes(
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    bm25_path: str,
    faiss_path: str,
) -> None:
    """Serialize BM25 and FAISS indexes to disk."""
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index to {bm25_path}")

    faiss.write_index(faiss_index, faiss_path)
    logger.info(f"Saved FAISS index to {faiss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes from chunks")
    parser.add_argument(
        "--chunks", type=str, default="data/chunks.json", help="Path to chunks JSON"
    )
    parser.add_argument(
        "--bm25-out", type=str, default="indexes/bm25.pkl", help="Output path for BM25 pickle"
    )
    parser.add_argument(
        "--faiss-out", type=str, default="indexes/faiss.index", help="Output path for FAISS index"
    )
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)

    # Build BM25
    bm25 = build_bm25_index(chunks)

    # Build FAISS
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    faiss_index = build_faiss_index(chunks, model)

    # Save both
    save_indexes(bm25, faiss_index, args.bm25_out, args.faiss_out)
    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
