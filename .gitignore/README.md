# BIS Standards Recommendation Engine

AI-powered RAG pipeline that maps product descriptions to applicable Bureau of Indian Standards (BIS) — compressing weeks of manual compliance research into seconds.

Built for the **BIS × SS Hackathon 2026** | Track: AI / RAG — Building Materials

---

## Architecture

```
BIS SP 21 PDF
     │
     ▼
PDF Parser (pdfplumber + regex)
     │
     ▼
Smart Chunker (standard-aware splits)
     │
     ├──► BM25 Index (rank-bm25)
     └──► FAISS Dense Index (all-MiniLM-L6-v2)
                │
                ▼
         RRF Fusion (k=60) → Top-5 chunks
                │
                ▼
         LLM (Groq llama-3.1-8b-instant)
                │
                ▼
         Output: IS code · title · rationale
```

---

## Setup

### 0. Recommended Python version

Use Python 3.11 or 3.12 for the most reliable wheel compatibility on Windows. Python 3.13 can work, but may require reinstalling NumPy/PyTorch with matching platform wheels.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build indexes

Place the BIS SP 21 PDF in the project root (or specify its path), then run:

```bash
# Step 1: Parse PDF into chunks
python src/ingestion.py --pdf path/to/BIS_SP21.pdf --output data/chunks.json

# Step 2: Build BM25 + FAISS indexes
python src/indexer.py --chunks data/chunks.json --bm25-out indexes/bm25.pkl --faiss-out indexes/faiss.index
```

---

## Inference (Judge Command)

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

**Output schema (strict):**
```json
[
  {
    "id": "q001",
    "retrieved_standards": ["IS 269: 1989", "IS 455: 1989"],
    "latency_seconds": 1.24
  }
]
```

---

## Evaluation

The organizer's `eval_script.py` reads `expected_standards` from the results file.
On the **judge's machine**, the private test set already contains `expected_standards`,
so `inference.py` output is scored directly.

For **local validation** against the public test set, merge first:

```bash
# Merge expected standards into predictions for local scoring
python scripts/merge_for_eval.py \
    --predictions data/public_results.json \
    --ground-truth public_test_set.json \
    --output data/public_results_eval.json

# Run eval
python eval_script.py --results data/public_results_eval.json
```

**Public test set results:**
| Metric | Score | Target |
|--------|-------|--------|
| Hit Rate @3 | 100.00% | >80% ✅ |
| MRR @5 | 0.9500 | >0.70 ✅ |
| Avg Latency | 2.83s | <5.0s ✅ |

---

## Web UI

```bash
python src/app.py
```

Opens at `http://localhost:7860` (or auto-selected port if 7860 is busy)

**Quick test:**
```bash
# In the UI, try one of the example queries:
# e.g., "We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard applies?"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 7860 already in use | App auto-falls back to available port; check startup logs |
| "Pipeline not loaded" error | Run `python src/indexer.py` to build indexes |
| No API keys (LLM unavailable) | System uses BM25-only fallback; results still high-quality |
| Slow embedding model load | First query takes 5-10s to load model; subsequent queries fast |
| Windows NumPy/PyTorch issues | Use Python 3.11 or 3.12 (3.13 can need wheel reinstall) |

---

## API Keys


Set environment variables before running inference:

```bash
export GROQ_API_KEY=your_groq_key_here
export GEMINI_API_KEY=your_gemini_key_here   # optional fallback
```

Get a free Groq API key at: https://console.groq.com

If no API key is set, the system falls back to retrieval-only mode (no LLM generation).

---

## Repository Structure

```
bis-standards-engine/
├── src/
│   ├── ingestion.py        # PDF → structured chunks
│   ├── indexer.py          # Build + serialize BM25 and FAISS indexes
│   ├── retriever.py        # Hybrid BM25 + dense + RRF fusion
│   ├── generator.py        # LLM prompt construction + response parsing
│   └── app.py              # Gradio UI (runs standalone)
├── data/
│   ├── chunks.json         # Serialized chunk store (generated)
│   └── public_results.json # Results from public test set (generated)
├── indexes/
│   ├── bm25.pkl            # Serialized BM25 index (generated)
│   └── faiss.index         # Serialized FAISS index (generated)
├── inference.py            # Judge entry point (CRITICAL)
├── eval_script.py          # Provided by organizers (do not modify)
├── requirements.txt        # All dependencies, pinned versions
├── presentation.pdf        # 8-slide deck
└── README.md
```

---

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Hit Rate @3 | > 80% | ≥1 correct standard in top-3 results |
| MRR @5 | > 0.70 | Mean Reciprocal Rank in top-5 |
| Avg Latency | < 5.0s | Per-query response time |

---

## Technology Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| PDF parsing | pdfplumber | Best layout extraction for BIS documents |
| BM25 index | rank-bm25 | Exact keyword matching for IS codes |
| Embeddings | all-MiniLM-L6-v2 | Fast (80MB), strong semantic quality |
| Vector store | faiss-cpu | No server required, consumer hardware |
| Result fusion | Custom RRF | Pure Python, no extra dependency |
| LLM | Groq llama-3.1-8b-instant | Free API, <1s inference |
| LLM fallback | Gemini 1.5 Flash | If Groq quota exceeded |
| Web UI | Gradio | Fast to build, clean default styling |
