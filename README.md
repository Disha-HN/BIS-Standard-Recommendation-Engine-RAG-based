# BIS Standards Recommendation Engine

AI-powered RAG pipeline that maps product descriptions to applicable Bureau of Indian Standards (BIS).

Built for the **BIS × SS Hackathon 2026** | Track: AI / RAG — Building Materials

See [`recommendation_systems/README.md`](recommendation_systems/README.md) for full documentation.

## Quick Start

```bash
cd recommendation_systems
pip install -r requirements.txt

# Build indexes (one-time)
python src/ingestion.py --pdf path/to/dataset.pdf --output data/chunks.json
python src/indexer.py

# Run inference (judge command)
python inference.py --input hidden_private_dataset.json --output team_results.json --no-llm

# Run UI
python src/app.py
```

## Results

| Metric | Score | Target |
|--------|-------|--------|
| Hit Rate @3 | **100%** | >80% |
| MRR @5 | **1.0000** | >0.70 |
| Avg Latency | **0.39s** | <5.0s |
