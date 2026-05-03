# BIS Standards Recommendation Engine — Full Technical Report

**Team:** BIS × SS Hackathon 2026 | Track: AI / RAG — Building Materials  
**Date:** May 3, 2026  
**System:** Hybrid RAG Pipeline (BM25 + FAISS + Query Expansion + Groq LLM)

---

## 1. System Architecture

```
User Query
    │
    ▼
Query Expander (synonym map + concept injection + rewrite rules)
    │
    ├── Original Query ──────────────────────────────────┐
    └── Expanded Query (BIS vocabulary terms + IS codes) ┘
              │                                          │
              ▼                                          ▼
        BM25 Sparse                              Dense FAISS
        Retrieval (top-20)                       Retrieval (top-20)
              │                                          │
              └──────────── RRF Merge ───────────────────┘
                                │
                         RRF Fusion (k=60)
                         top-10 candidates
                                │
                    Validity Filter (IS code + year check)
                    Deduplication (normalized code)
                                │
                           Top-5 Results
                                │
                    LLM Rationale (Groq llama-3.1-8b-instant)
                    Fallback: Gemini → Template rationale
                                │
                           Final Output
```

**Key components:**

| Component | Technology | Purpose |
|---|---|---|
| PDF Ingestion | pdfplumber + regex | Parse BIS SP 21 into structured chunks |
| Sparse Index | rank-bm25 (BM25Okapi) | Keyword matching on IS codes & material names |
| Dense Index | FAISS IndexFlatIP + all-MiniLM-L6-v2 | Semantic similarity retrieval |
| Fusion | Reciprocal Rank Fusion (k=60) | Combine BM25 + dense ranked lists |
| Query Expansion | Rule-based synonym map (50+ rules) | Bridge user language → BIS vocabulary |
| LLM | Groq llama-3.1-8b-instant | Generate rationale text |
| Fallback | Gemini 1.5 Flash → Template | Graceful degradation if LLM unavailable |
| UI | Gradio 6.x | Web interface for MSEs |

---

## 2. Official Evaluation Results (Public Test Set — 10 Queries)

```
==================================================
     BIS HACKATHON EVALUATION RESULTS
==================================================
Total Queries Evaluated : 10
Hit Rate @3             : 100.00%  (Target: >80%)   ✅ PASS
MRR @5                  : 0.9500   (Target: >0.70)  ✅ PASS
Avg Latency             : 0.05s    (Target: <5.0s)  ✅ PASS
==================================================
Overall: ✅ ALL TARGETS MET
==================================================
```

All three official targets exceeded. Latency is 100× under the 5s limit.

---

## 3. Extended Test Results — 17 Queries Across 6 Difficulty Levels

### 3.1 Basic Queries (Easy — Pipeline Correctness)

---

**BASIC-01** — `"cement for building construction"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 6452: 1989 | High Alumina Cement for Structural Use | ⚠️ Too specialized for generic query |
| 2 | IS 8043: 1991 | Hydrophobic Portland Cement | ✅ Relevant |
| 3 | IS 269: 1989 | Ordinary Portland Cement, 33 Grade | ✅✅ Best answer |
| 4 | IS 8112: 1989 | 53 Grade OPC | ✅ Relevant |
| 5 | IS 12440: 1988 | Precast Concrete Stone Masonry Blocks | ⚠️ Marginal |

**Score: 4/5 relevant. Hit@3: ✅**  
*Note: IS 269 should ideally be rank 1 for a generic cement query. Query expansion adds "IS 269" hint but dense retrieval still ranks High Alumina first due to semantic similarity to "structural".*

---

**BASIC-02** — `"steel bars for reinforcement"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 432 (Part 1): 1982 | Mild Steel and Medium Tensile Steel Bars | ✅✅ Perfect |
| 2 | IS 1786: 1985 | Plain High Strength Deformed Steel Bars | ✅✅ Perfect |
| 3 | IS 1599: 1985 | Method for Bend Test (Steel) | ✅ Testing standard |
| 4 | IS 2090: 1983 | High Tensile Steel Bars | ✅ Relevant |
| 5 | IS 13620: 1993 | Fusion Bonded Epoxy Coated Bars | ✅ Relevant |

**Score: 5/5 relevant. Hit@3: ✅✅ Excellent.**

---

**BASIC-03** — `"aggregates used in concrete"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 383: 1970 | Coarse and Fine Aggregates from Natural Sources | ✅✅ Perfect |
| 2 | IS 9142: 1979 | Artificial Lightweight Aggregates | ✅ Relevant |
| 3 | IS 6579: 1981 | Coarse Aggregate for Water Bound Macadam | ✅ Relevant |
| 4 | IS 2686: 1977 | Cinder as Fine Aggregates | ✅ Relevant |
| 5 | IS 5871: 1987 | Bitumen Mastic | ⚠️ Marginal |

**Score: 4/5 relevant. Hit@3: ✅✅ Excellent.**

---

### 3.2 Medium Complexity (Semantic Understanding)

---

**MED-01** — `"high strength concrete for bridges"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 2090: 1983 | High Tensile Steel Bars Used in Prestressed Concrete | ✅✅ Perfect |
| 2 | IS 1786: 1985 | Plain High Strength Deformed Steel Bars | ✅ Relevant |
| 3 | IS 6523: 1983 | Precast Reinforced Concrete Door Frames | ❌ Irrelevant |
| 4 | IS 784: 2001 | Prestressed Concrete Pipes | ✅ Relevant |
| 5 | IS 6909: 1990 | Supersulphated Cement | ⚠️ Marginal |

**Score: 3-4/5 relevant. Hit@3: ✅**  
*Note: IS 6523 (door frames) at rank 3 is a false positive from "concrete" keyword overlap.*

---

**MED-02** — `"corrosion resistant steel for coastal areas"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 11587: 1986 | Structural Weather Resistance Steels | ✅✅ Perfect |
| 2 | IS 13620: 1993 | Fusion Bonded Epoxy Coated Reinforcement Bars | ✅✅ Perfect |
| 3 | IS 811: 1987 | Cold Formed Light Gauge Structural Steel | ✅ Relevant |
| 4 | IS 10019: 1981 | Mild Steel Stays and Fasteners | ❌ Irrelevant |
| 5 | IS 3311: 1979 | Waste Plug and Accessories for Sinks | ❌ Irrelevant |

**Score: 3/5 relevant. Hit@3: ✅✅ Top 2 are perfect.**

---

**MED-03** — `"cement suitable for marine environment"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 455: 1989 | Portland Slag Cement | ✅✅ Excellent for marine |
| 2 | IS 6909: 1990 | Supersulphated Cement | ✅✅ Excellent for marine |
| 3 | IS 6523: 1983 | Precast Reinforced Concrete Door Frames | ❌ Irrelevant |
| 4 | IS 8043: 1991 | Hydrophobic Portland Cement | ✅ Relevant |
| 5 | IS 10388: 1982 | Corrugated Coir, Woodwool Sheets | ❌ Irrelevant |

**Score: 3/5 relevant. Hit@3: ✅**  
*Note: IS 6523 keeps appearing as a false positive — its chunk text contains "concrete" heavily.*

---

### 3.3 Real-World Messy Inputs (Natural Language Understanding)

---

**REAL-01** — `"I manufacture cement blocks used near sea areas, need durability"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 2116: 1980 | Sand for Masonry Mortars | ✅ Relevant for block manufacturing |
| 2 | IS 2185 (Part 3): 1984 | Concrete Masonry Units (Autoclaved) | ✅✅ Perfect |
| 3 | IS 12440: 1988 | Precast Concrete Stone Masonry Blocks | ✅✅ Perfect |
| 4 | IS 10360: 1982 | Lime-Pozzolana Concrete | ✅ Relevant |
| 5 | IS 3115: 1992 | Lime Based Blocks | ✅ Relevant |

**Score: 5/5 relevant. Hit@3: ✅✅ Excellent natural language handling.**

---

**REAL-02** — `"steel rods used in construction that should not rust easily"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 13620: 1993 | Fusion Bonded Epoxy Coated Reinforcement Bars | ✅✅ Perfect — anti-corrosion |
| 2 | IS 1786: 1985 | Plain High Strength Deformed Steel Bars | ✅✅ Primary rebar standard |
| 3 | IS 432 (Part 1): 1982 | Mild Steel and Medium Tensile Steel Bars | ✅✅ Core rebar standard |
| 4 | IS 2090: 1983 | High Tensile Steel Bars | ✅ Relevant |
| 5 | IS 1599: 1985 | Method for Bend Test (Steel) | ✅ Testing standard |

**Score: 5/5 relevant. Hit@3: ✅✅ Excellent — query rewrite fixed welding rod false positives.**

---

**REAL-03** — `"materials for road construction with heavy load resistance"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 3308: 1981 | Wood Wool Building Slabs | ❌ Irrelevant |
| 2 | IS 5317: 2002 | Bitumen Mastic for Bridge Decks | ✅ Relevant |
| 3 | IS 215: 1995 | Road Tar | ✅✅ Perfect |
| 4 | IS 736: 1986 | Wrought Aluminium and Aluminium Alloy | ❌ Irrelevant |
| 5 | IS 875 (Part 1): 1987 | Code of Practice for Design Loads | ⚠️ Structural loads |

**Score: 2/5 relevant. Hit@3: ⚠️ Partial (Road Tar at rank 3).**  
*Root cause: "road construction" + "heavy load" has limited vocabulary in BIS SP 21. "Wood Wool" ranks first due to dense similarity to "construction materials".*

---

### 3.4 Ambiguous Queries (Overfitting Test)

---

**AMB-01** — `"eco friendly construction materials"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 3629: 1986 | Structural Timber in Buildings | ✅✅ Eco-friendly material |
| 2 | IS 399: 1963 | Classification of Commercial Timbers | ✅✅ Eco-friendly material |
| 3 | IS 14201: 1994 | Precast Reinforced Concrete Channel | ❌ Irrelevant |
| 4 | IS 1795: 1982 | Pillar Taps for Water Supply | ❌ Irrelevant |
| 5 | IS 771 (Part 2): 1985 | Glazed Fire-Clay Sanitary | ❌ Irrelevant |

**Score: 2/5 relevant. Hit@3: ⚠️ Partial.**  
*Root cause: "eco-friendly" is not a BIS SP 21 category. Query expansion maps it to timber/pozzolana/fly ash, which surfaces timber standards correctly at ranks 1-2, but ranks 3-5 are noise.*

---

**AMB-02** — `"low cost housing materials"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 9197: 1979 | Epoxy Resin, Hardeners and Epoxy Adhesives | ❌ Irrelevant |
| 2 | IS 1823: 1980 | Floor Door Stoppers | ❌ Irrelevant |
| 3 | IS 3115: 1992 | Lime Based Blocks | ✅ Relevant for low-cost housing |
| 4 | IS 4762: 1984 | Worm Drive Clamps | ❌ Irrelevant |
| 5 | IS 4992: 1975 | Door Handles for Mortice Locks | ❌ Irrelevant |

**Score: 1/5 relevant. Hit@3: ❌ Fail.**  
*Root cause: "housing" in BIS SP 21 is associated with door hardware (door housing, lock housing). The dense model maps "housing materials" → door/fitting standards. This is a fundamental vocabulary mismatch — BIS SP 21 does not use "housing" to mean residential construction.*

---

**AMB-03** — `"fire resistant building materials"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 9742: 1993 | Sprayed Mineral Wool Thermal Insulation | ✅✅ Fire/thermal resistant |
| 2 | IS 8272: 1984 | Gypsum Plaster | ✅✅ Fire resistant material |
| 3 | IS 771 (Part 1): 1979 | Glazed Fire-Clay Sanitary Appliances | ⚠️ Fire-clay but sanitary |
| 4 | IS 5509: 2000 | Fire Retardant Plywood | ✅✅ Perfect |
| 5 | IS 4832 (Part 2): 1969 | Chemical Resistant Mortars | ✅ Relevant |

**Score: 4/5 relevant. Hit@3: ✅✅ Excellent.**

---

### 3.5 Hard Queries (Deep Concept Mapping)

---

**HARD-01** — `"cement that can resist sulfate attack in underground structures"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 6909: 1990 | Supersulphated Cement | ✅✅ Designed for aggressive soils |
| 2 | IS 455: 1989 | Portland Slag Cement | ✅✅ Sulfate resistant |
| 3 | IS 12330: 1988 | Sulphate Resisting Portland Cement | ✅✅ Exact match |
| 4 | IS 6598: 1972 | Cellular Concrete for Thermal Insulation | ❌ Irrelevant |
| 5 | IS 9743: 1990 | Thermal Insulation Finishing Cements | ❌ Irrelevant |

**Score: 3/5 relevant. Hit@3: ✅✅ Top 3 are all perfect — exactly the right standards.**

---

**HARD-02** — `"steel reinforcement for earthquake resistant structures"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 432 (Part 1): 1982 | Mild Steel and Medium Tensile Steel Bars | ✅ Relevant |
| 2 | IS 1599: 1985 | Method for Bend Test (Steel) | ✅ Testing standard |
| 3 | IS 12440: 1988 | Precast Concrete Stone Masonry Blocks | ❌ Irrelevant |
| 4 | IS 1786: 1985 | Plain High Strength Deformed Steel Bars | ✅✅ Best answer for seismic |
| 5 | IS 11587: 1986 | Structural Weather Resistance Steels | ✅ Relevant |

**Score: 4/5 relevant. Hit@3: ✅**  
*Note: IS 1786 (the primary seismic reinforcement standard) is at rank 4 instead of rank 1. "Earthquake" has no direct vocabulary in BIS SP 21 — the system correctly retrieves it but cannot rank it first.*

---

**HARD-03** — `"concrete mix suitable for high humidity and coastal exposure"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 6598: 1972 | Cellular Concrete for Thermal Insulation | ❌ Wrong — thermal, not coastal |
| 2 | IS 8043: 1991 | Hydrophobic Portland Cement | ✅✅ Designed for humid/coastal |
| 3 | IS 6909: 1990 | Supersulphated Cement | ✅✅ Excellent for coastal |
| 4 | IS 6523: 1983 | Precast Reinforced Concrete Door Frames | ❌ Irrelevant |
| 5 | IS 455: 1989 | Portland Slag Cement | ✅✅ Excellent for coastal |

**Score: 3/5 relevant. Hit@3: ✅**  
*Note: IS 6598 (Cellular Concrete) at rank 1 is a false positive — "concrete" + "humidity" matches its chunk text. The correct answers are at ranks 2, 3, 5.*

---

### 3.6 Multi-Intent Queries (Ranking & Prioritization)

---

**MULTI-01** — `"high durability and corrosion resistant concrete for bridges in coastal regions"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 6909: 1990 | Supersulphated Cement | ✅✅ Coastal durability |
| 2 | IS 1834: 1984 | Hot Applied Sealing Compounds | ❌ Sealants, not concrete |
| 3 | IS 11587: 1986 | Structural Weather Resistance Steels | ✅ Corrosion resistant |
| 4 | IS 4832 (Part 2): 1969 | Chemical Resistant Mortars | ✅ Relevant |
| 5 | IS 6452: 1989 | High Alumina Cement for Structural Use | ✅✅ High durability |

**Score: 4/5 relevant. Hit@3: ✅**  
*Note: IS 1834 (sealing compounds) at rank 2 is a false positive from "bridge" keyword.*

---

**MULTI-02** — `"low cost but high strength cement for rural construction"`

| Rank | IS Code | Title | Verdict |
|---|---|---|---|
| 1 | IS 6452: 1989 | High Alumina Cement for Structural Use | ⚠️ High strength but expensive |
| 2 | IS 12330: 1988 | Sulphate Resisting Portland Cement | ✅ Relevant |
| 3 | IS 2645: 2003 | Integral Cement Waterproofing Compound | ⚠️ Additive, not cement type |
| 4 | IS 8112: 1989 | 53 Grade OPC | ✅✅ High strength, widely available |
| 5 | IS 8043: 1991 | Hydrophobic Portland Cement | ✅ Relevant |

**Score: 3/5 relevant. Hit@3: ✅**  
*Note: "Low cost" is not a BIS SP 21 concept — the system cannot distinguish expensive vs affordable standards. IS 269 (33 Grade OPC — the cheapest, most common cement) is missing from results.*

---

## 4. Summary Scorecard

| ID | Query | Relevant/5 | Hit@3 | Grade |
|---|---|---|---|---|
| BASIC-01 | cement for building construction | 4/5 | ✅ | B+ |
| BASIC-02 | steel bars for reinforcement | 5/5 | ✅✅ | **A+** |
| BASIC-03 | aggregates used in concrete | 4/5 | ✅✅ | **A** |
| MED-01 | high strength concrete for bridges | 3/5 | ✅ | B |
| MED-02 | corrosion resistant steel for coastal | 3/5 | ✅✅ | B+ |
| MED-03 | cement suitable for marine environment | 3/5 | ✅ | B |
| REAL-01 | cement blocks near sea, need durability | 5/5 | ✅✅ | **A+** |
| REAL-02 | steel rods not rust (messy NL) | 5/5 | ✅✅ | **A+** |
| REAL-03 | road construction heavy load | 2/5 | ⚠️ | D |
| AMB-01 | eco friendly construction materials | 2/5 | ⚠️ | C |
| AMB-02 | low cost housing materials | 1/5 | ❌ | **F** |
| AMB-03 | fire resistant building materials | 4/5 | ✅✅ | **A** |
| HARD-01 | sulfate resistant cement underground | 3/5 | ✅✅ | **A** |
| HARD-02 | earthquake resistant steel | 4/5 | ✅ | B+ |
| HARD-03 | coastal concrete high humidity | 3/5 | ✅ | B |
| MULTI-01 | coastal bridge corrosion concrete | 4/5 | ✅ | B+ |
| MULTI-02 | low cost high strength cement rural | 3/5 | ✅ | B |

**Overall: 14/17 Hit@3 (82%) | Avg Relevant: 3.4/5**

---

## 5. Bugs Fixed During Development

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | 🔴 | F-string syntax error in retriever logging | Extracted value to variable before f-string |
| 2 | 🔴 | RRF penalty rank hardcoded to `TOP_CANDIDATES=20` | Changed to `len(list) + 1` (correct penalty) |
| 3 | 🔴 | Non-greedy regex `*?` truncated LLM JSON arrays | Changed to greedy `*` to capture full array |
| 4 | 🔴 | IS code format inconsistency (`"IS 269 : 1989"` vs `"IS 269: 1989"`) | Added `canonicalize_std_code()` function |
| 5 | 🔴 | Duplicate standards in output (same code, different spacing) | Deduplication at retriever + generator level |
| 6 | 🔴 | `"is 100"` garbage code from ingestion in results | `is_valid_std_code()` requires 4-digit year |
| 7 | 🔴 | Relevance bars showing 0% in UI | Fixed `chunk_map` lookup (normalized key matching) + relative score normalization |
| 8 | 🟡 | `sample_output.json` didn't match actual pipeline output | Regenerated from actual pipeline |
| 9 | 🟡 | `theme`/`css` passed to `launch()` instead of `gr.Blocks()` | Moved to `launch()` (Gradio 6 API) |
| 10 | 🟡 | Rate-limit timer updated even in no-LLM mode | Only update timer when API key present |
| 11 | 🟠 | No API key setup documentation | Created `.env.example` |
| 12 | 🟠 | No explicit retrieval-only mode | Added `--no-llm` flag to `inference.py` |
| 13 | 🟠 | `data/public_results.json` missing from repo | Regenerated and committed |
| 14 | 🔵 | `bm25_fallback()` misleading name | Renamed to `build_retrieval_results()` |
| 15 | 🔵 | `groq==0.9.0` in requirements vs `1.2.0` installed | Pinned to `groq==1.2.0` |
| 16 | 🔵 | Stale RRF docstring after penalty rank fix | Updated docstring |
| 17 | 🔵 | IS header regex too restrictive in ingestion | Fixed pattern to use `\S` instead of `[A-Z]` |

---

## 6. Improvements Made

### Query Expansion (`src/query_expander.py`)
A new module with 50+ synonym rules, concept injections, and query rewrite logic:

- **Synonym substitution** — maps user terms to BIS vocabulary  
  e.g. `"rust"` → `"corrosion resistant epoxy coated weathering"`
- **Concept injection** — appends IS code hints for known concept combinations  
  e.g. `"sulfate"` → `"IS 12330 sulphate resisting IS 6909 supersulphated IS 455 slag"`
- **Query rewrite** — replaces misleading terms before BM25 sees them  
  e.g. `"steel rods ... construction"` → `"steel bars reinforcement ... construction"` (prevents welding rod false positives)
- **Multi-query fusion** — runs retrieval on both original and expanded query, RRF-merges results

### Retriever Improvements (`src/retriever.py`)
- Validity filter: rejects chunks with no 4-digit year in std_code
- Deduplication: same standard in multiple sub-chunks → keep highest-ranked only
- Oversized candidate pool: requests `TOP_RESULTS + 5` from fusion so filtering never leaves fewer than 5 results

### Generator Improvements (`src/generator.py`)
- `canonicalize_std_code()` — normalizes all IS code variants to `"IS NNN: YYYY"` format
- `is_valid_std_code()` — validates codes have IS prefix + digits + 4-digit year
- Deduplication in both `parse_llm_response()` and `build_retrieval_results()`

---

## 7. Known Limitations

| Limitation | Root Cause | Impact |
|---|---|---|
| "low cost housing" → door hardware | "housing" in BIS SP 21 = door/lock housing, not residential | AMB-02 fails completely |
| "road construction" → wood wool slabs | BIS SP 21 road standards are sparse; "construction" is generic | REAL-03 weak |
| "earthquake resistant" → IS 1786 at rank 4 | "earthquake/seismic" not in BIS SP 21 vocabulary | HARD-02 ranking suboptimal |
| IS 6523 (door frames) false positive | Chunk text is dense with "concrete" and "reinforced" | Appears in MED-01, MED-03, HARD-03 |
| IS 6598 (cellular concrete) false positive | "concrete" + "humidity" matches thermal insulation chunk | HARD-03 rank 1 wrong |
| "low cost" concept unmappable | BIS SP 21 has no cost classification | MULTI-02 rank 1 wrong |

---

## 8. Files Changed / Created

| File | Change |
|---|---|
| `src/generator.py` | Full rewrite: canonicalization, validation, deduplication, renamed `build_retrieval_results` |
| `src/retriever.py` | Added validity filter, deduplication, multi-query fusion, `_merge_ranked_lists` |
| `src/query_expander.py` | **New** — 50+ synonym rules, concept injections, query rewrite logic |
| `src/app.py` | Fixed chunk_map lookup, relative confidence scoring, Gradio 6 theme/css fix |
| `inference.py` | Added `--no-llm` flag, `_dedupe_and_clean()`, fixed rate-limit timer |
| `eval_script.py` | Added ✅/❌ pass/fail indicators and overall summary |
| `requirements.txt` | Pinned `groq==1.2.0`, `torch==2.1.2`, `transformers==4.34.1` |
| `.env.example` | **New** — API key setup instructions for Windows and Linux |
| `data/public_results.json` | **New** — regenerated from actual pipeline |
| `data/public_results_eval.json` | Regenerated with correct merged format |
| `scripts/diagnose.py` | **New** — diagnostic script for query result analysis |
| `scripts/merge_for_eval.py` | No change |

---

## 9. How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Set API Keys (optional — for LLM rationale)
```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"
```

### Run Web UI
```bash
python src/app.py
# Opens at http://127.0.0.1:7860
```

### Run Inference (Judge Entry Point)
```bash
# With LLM rationale
python inference.py --input hidden_private_dataset.json --output team_results.json

# Retrieval-only (no API key needed, faster)
python inference.py --input hidden_private_dataset.json --output team_results.json --no-llm
```

### Evaluate Results
```bash
python scripts/merge_for_eval.py \
  --predictions team_results.json \
  --ground-truth public_test_set.json \
  --output merged_eval.json

python eval_script.py --results merged_eval.json
```

### Rebuild Indexes (if PDF changes)
```bash
python src/ingestion.py --pdf dataset.pdf --output data/chunks.json
python src/indexer.py --chunks data/chunks.json
```

---

*Report generated: May 3, 2026*
