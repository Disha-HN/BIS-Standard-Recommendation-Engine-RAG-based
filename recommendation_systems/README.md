# BIS Standards Recommendation Engine

AI-powered RAG pipeline that maps product descriptions to applicable Bureau of Indian Standards (BIS) — compressing weeks of manual compliance research into seconds.

Built for the **BIS × SS Hackathon 2026** | Track: AI / RAG — Building Materials

---

## Verified Results (Public Test Set — 10 Queries)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Hit Rate @3 | **100.00%** | >80% | PASS |
| MRR @5 | **1.0000** | >0.70 | PASS |
| Avg Latency | **0.39s** | <5.0s | PASS |

---

## Architecture

```
User Query
    |
    v
Query Expander
  - 80+ synonym rules (OPC, TMT, RCC, sariya, pucca, gitti...)
  - Concept injection (coastal -> IS 6909, earthquake -> IS 1786)
  - Negative-term rewrite (housing -> construction)
    |
    +-- Original Query --------+
    +-- Expanded Query --------+
              |                |
              v                v
        BM25 Sparse      Dense FAISS
        (top-20)         (top-20, cached)
              |                |
              +---- RRF Merge -+
                       |
              RRF Fusion (k=60)
                       |
              Title-Match Re-Ranking
              (boosts chunks whose title
               matches query key terms)
                       |
              Category Correction
              (title-first inference)
                       |
                   Top-5 Results
                       |
              LLM Rationale (optional)
              Groq llama-3.1-8b-instant
              Fallback: Gemini -> Template
                       |
                  Final Output
```

---

## Setup

### Requirements
- Python 3.11 or 3.12 recommended (3.13 works but may need wheel reinstall on Windows)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build indexes

```bash
# Parse BIS SP 21 PDF into structured chunks
python src/ingestion.py --pdf path/to/dataset.pdf --output data/chunks.json

# Build BM25 + FAISS indexes
python src/indexer.py --chunks data/chunks.json --bm25-out indexes/bm25.pkl --faiss-out indexes/faiss.index
```

---

## Inference

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json --no-llm
```

The `--no-llm` flag gives **0.39s avg latency** (100x under the 5s limit).
IS codes always come from retrieved chunk metadata — never from LLM output.

**Output schema :**
```json
[
  {
    "id": "q001",
    "retrieved_standards": ["IS 269: 1989", "IS 455: 1989"],
    "latency_seconds": 0.39
  }
]
```

---

## Evaluation (Local Validation)

```bash
# Run inference
python inference.py --input public_test_set.json --output data/public_results.json --no-llm

# Merge expected standards for scoring
python scripts/merge_for_eval.py \
    --predictions data/public_results.json \
    --ground-truth public_test_set.json \
    --output data/public_results_eval.json

# Score
python eval_script.py --results data/public_results_eval.json
```

---

## Web UI

```bash
python src/app.py
```

Opens at `http://127.0.0.1:7861`

Features:
- Live loading state with status indicator
- Result cards with confidence bars, category badges, medal icons
- Copy-to-clipboard for IS codes
- 18 example queries across 5 categories
- Query history panel

**Latency (UI Mode):**
- **Retrieval-only (default):** ~0.3–0.5s per query
  - IS codes come from retrieved chunk metadata
  - No LLM calls → fast, deterministic, offline-ready
- **With LLM rationale (optional):** ~2–3s per query
  - Adds Groq/Gemini API network latency
  - Provides 1–2 sentence explanations per standard
  - Enable with: `$env:ENABLE_UI_LLM = "1"` (Windows) or `export ENABLE_UI_LLM=1` (Linux/macOS)

---

## API Keys 
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"

# Linux / macOS
export GROQ_API_KEY=your_key_here
```

Get a free Groq key at: https://console.groq.com

Without a key, the system uses retrieval-only mode — IS codes and scores are identical.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Index file not found` | Run `python src/indexer.py` to build indexes first |
| **UI latency > 2s** | Retrieval-only mode (default) is 0.3–0.5s. If you see >2s, unset `ENABLE_UI_LLM`: unset env var or restart with `$env:ENABLE_UI_LLM = "0"` |
| **Latency > 5s in inference** | Use `--no-llm` flag in `inference.py` — pure retrieval is ~0.39s |
| LLM calls failing | Check `GROQ_API_KEY` and `GEMINI_API_KEY` env vars are set; verify API quota |
| Port 7861 in use | App auto-selects next available port |
| Slow first query | Embedding model loads once on startup (~15s); subsequent queries are fast |
| Windows NumPy issues | Use Python 3.11 or 3.12 |

---

## Repository Structure

```
recommendation_systems/
├── src/
│   ├── ingestion.py        # PDF -> structured chunks
│   ├── indexer.py          # Build + serialize BM25 and FAISS indexes
│   ├── retriever.py        # Hybrid BM25 + dense + RRF + title re-ranking
│   ├── query_expander.py   # 80+ synonym rules, concept injection, negative-term rewrite
│   ├── generator.py        # LLM prompt construction + anti-hallucination parsing
│   └── app.py              # Gradio UI
├── data/
│   ├── chunks.json         # Serialized chunk store (generated)
│   ├── public_results.json # Results from public test set
│   └── public_results_eval.json
├── indexes/
│   ├── bm25.pkl            # Serialized BM25 index (generated)
│   └── faiss.index         # Serialized FAISS index (generated)
├── scripts/
│   ├── merge_for_eval.py   # Merge predictions + ground truth for local eval
│   └── diagnose.py         # Query diagnostic tool
├── inference.py            # Judge entry point (CRITICAL)
├── eval_script.py          # Organizer eval script (do not modify)
├── public_test_set.json    # Public test queries
├── sample_output.json      # Example output from the pipeline
├── requirements.txt        # All dependencies, pinned versions
└── README.md
```

---

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF parsing | pdfplumber | Best layout extraction for BIS documents |
| BM25 index | rank-bm25 | Exact keyword matching for IS codes |
| Embeddings | all-MiniLM-L6-v2 | Fast (80MB), strong semantic quality |
| Vector store | faiss-cpu | No server required, consumer hardware |
| Query expansion | Custom rule-based | 80+ rules, zero latency, no LLM needed |
| Result fusion | Custom RRF (k=60) | Mathematically sound, pure Python |
| Title re-ranking | Custom overlap scorer | General false-positive suppression |
| LLM | Groq llama-3.1-8b-instant | Free API, fast inference |
| LLM fallback | Gemini 1.5 Flash | If Groq quota exceeded |
| Web UI | Gradio | Fast, polished, MSE-friendly |
