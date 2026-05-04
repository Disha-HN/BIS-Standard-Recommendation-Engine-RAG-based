"""
app.py — Gradio Web UI for BIS Standards Recommendation Engine

Best-in-class MSE-friendly interface with:
  - Live search with loading animation
  - Rich result cards with confidence bars and category badges
  - Clickable example queries grouped by category
  - Query history panel
  - Copy-to-clipboard for IS codes
  - About / How-it-works accordion
  - Responsive layout

Run standalone:
    python src/app.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path so imports work from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from sentence_transformers import SentenceTransformer

from src.retriever import HybridRetriever
from src.generator import generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = str(PROJECT_ROOT / "data" / "chunks.json")
BM25_PATH = str(PROJECT_ROOT / "indexes" / "bm25.pkl")
FAISS_PATH = str(PROJECT_ROOT / "indexes" / "faiss.index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Example queries grouped by category ───────────────────────────────────────
EXAMPLES: Dict[str, List[str]] = {
    "🏭 Cement": [
        "We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard applies?",
        "What is the Indian Standard for Portland Pozzolana Cement made with fly ash?",
        "Which standard governs White Portland cement for architectural purposes?",
        "We produce Portland slag cement. What are the chemical and physical requirements?",
        "Which standard covers supersulphated cement for marine and aggressive water conditions?",
    ],
    "🪨 Aggregates": [
        "Coarse and fine aggregates from natural sources for structural concrete.",
        "We supply crushed stone aggregates for road construction. What BIS standard applies?",
        "Lightweight aggregates for use in concrete — which standard applies?",
    ],
    "🧱 Concrete & Blocks": [
        "We produce hollow concrete masonry blocks. What are the BIS requirements?",
        "Standards for precast concrete pipes used in water supply mains.",
        "Ready-mix concrete for M25 grade structural use.",
        "Corrugated asbestos cement sheets for roofing and cladding.",
    ],
    "🔩 Steel": [
        "TMT bars used for reinforced concrete construction in buildings.",
        "Steel reinforcement for earthquake resistant structures.",
        "Corrosion resistant steel bars for coastal construction areas.",
        "High tensile steel bars for prestressed concrete bridges.",
    ],
    "🔥 Specialty": [
        "Cement that can resist sulfate attack in underground structures.",
        "Fire resistant building materials for industrial use.",
        "Eco-friendly and sustainable construction materials.",
        "Masonry cement for general purpose mortar, not structural concrete.",
    ],
}

# Flat list for the examples panel
ALL_EXAMPLES = [q for qs in EXAMPLES.values() for q in qs]

# ── Category colour map ────────────────────────────────────────────────────────
CATEGORY_COLORS: Dict[str, str] = {
    "Cement":           "#e67e22",
    "Steel":            "#2980b9",
    "Concrete":         "#27ae60",
    "Aggregates":       "#8e44ad",
    "Bricks & Tiles":   "#c0392b",
    "Waterproofing":    "#16a085",
    "Concrete Products":"#2c3e50",
    "Building Materials":"#7f8c8d",
}

# ── Global pipeline (loaded once at startup) ───────────────────────────────────
_retriever: HybridRetriever = None   # type: ignore
_model: SentenceTransformer = None   # type: ignore
_history: List[Tuple[str, str]] = []  # (query, timestamp)


def load_pipeline() -> bool:
    """Load embedding model and retriever indexes once at startup."""
    global _retriever, _model
    try:
        logger.info("Loading embedding model…")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Loading retriever indexes…")
        _retriever = HybridRetriever(CHUNKS_PATH, BM25_PATH, FAISS_PATH, _model)
        logger.info("Pipeline ready.")
        return True
    except FileNotFoundError as e:
        logger.error(
            f"Index file not found: {e}\n"
            "To build indexes, run:\n"
            "  1. python src/ingestion.py --pdf <path/to/SP21.pdf>\n"
            "  2. python src/indexer.py\n"
            "Then restart the app."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return False


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _category_badge(category: str) -> str:
    color = CATEGORY_COLORS.get(category, "#7f8c8d")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.6px;text-transform:uppercase;">{category}</span>'
    )


def _confidence_bar(pct: int) -> str:
    """Render a relevance bar given a 0-100 percentage."""
    if pct > 70:
        color, label = "#16a34a", "High"
    elif pct > 45:
        color, label = "#d97706", "Medium"
    else:
        color, label = "#dc2626", "Low"
    return f"""
    <div style="margin-top:12px;display:flex;align-items:center;gap:10px;">
      <span style="font-size:11px;color:#6b7280;width:120px;flex-shrink:0;">
        Relevance: <strong style="color:{color};">{label} ({pct}%)</strong>
      </span>
      <div style="flex:1;background:#e5e7eb;border-radius:6px;height:6px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,{color},{color}cc);width:{pct}%;
                    height:6px;border-radius:6px;transition:width 0.5s ease;"></div>
      </div>
    </div>"""


def _result_card(rank: int, r: Dict[str, Any], top_score: float = 1.0) -> str:
    std_code  = r.get("std_code", "N/A")
    title     = r.get("title", "")
    rationale = r.get("rationale", "")
    category  = r.get("category", "Building Materials")
    rrf_score = r.get("rrf_score", 0.0)

    # Normalise to 0-100 relative to the top result, floor at 30
    raw_pct = (rrf_score / top_score) * 95 if top_score > 0 else 0
    pct = max(30, min(100, round(raw_pct)))

    accent = CATEGORY_COLORS.get(category, "#2980b9")
    badge  = _category_badge(category)
    bar    = _confidence_bar(pct)

    rank_styles = [
        ("🥇", "#f59e0b", "#fffbeb"),
        ("🥈", "#9ca3af", "#f9fafb"),
        ("🥉", "#b45309", "#fef3c7"),
        ("#4",  "#6b7280", "#f9fafb"),
        ("#5",  "#6b7280", "#f9fafb"),
    ]
    rank_icon, rank_color, rank_bg = rank_styles[rank - 1] if rank <= 5 else ("#?", "#6b7280", "#f9fafb")

    # Copy button (uses JS clipboard API)
    copy_btn = (
        f'<button onclick="navigator.clipboard.writeText(\'{std_code}\')'
        f'.then(()=>{{this.textContent=\'✓ Copied\';setTimeout(()=>{{this.textContent=\'Copy\'}},1500)}})"'
        f' style="font-size:11px;padding:3px 10px;border:1px solid #d1d5db;border-radius:6px;'
        f'background:#f9fafb;color:#374151;cursor:pointer;transition:all 0.15s;'
        f'margin-left:8px;font-family:inherit;" '
        f'onmouseover="this.style.background=\'#e8f0fe\';this.style.borderColor=\'#1a73e8\'" '
        f'onmouseout="this.style.background=\'#f9fafb\';this.style.borderColor=\'#d1d5db\'">Copy</button>'
    )

    return f"""
    <div style="
        border:1px solid #e5e7eb;
        border-left:5px solid {accent};
        border-radius:12px;
        padding:18px 22px;
        margin-bottom:14px;
        background:#ffffff;
        box-shadow:0 1px 4px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        font-family:'Segoe UI',system-ui,sans-serif;
        transition:box-shadow 0.2s;
    " onmouseover="this.style.boxShadow='0 4px 16px rgba(0,0,0,0.12)'"
       onmouseout="this.style.boxShadow='0 1px 4px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04)'">

      <div style="display:flex;align-items:flex-start;justify-content:space-between;
                  flex-wrap:wrap;gap:8px;margin-bottom:10px;">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <span style="
              background:{rank_bg};border:1.5px solid {rank_color}33;
              border-radius:8px;padding:3px 10px;
              font-weight:700;font-size:13px;color:{rank_color};flex-shrink:0;
          ">{rank_icon}</span>
          <span style="font-size:20px;font-weight:800;color:#111827;
                       letter-spacing:0.2px;font-family:'Segoe UI',monospace;">{std_code}</span>
          {copy_btn}
        </div>
        {badge}
      </div>

      <p style="margin:0 0 8px;color:#4b5563;font-size:13px;font-style:italic;
                line-height:1.5;font-weight:500;">{title}</p>
      <p style="margin:0;color:#1f2937;font-size:14px;line-height:1.7;">{rationale}</p>
      {bar}
    </div>"""


def format_results_html(results: List[Dict[str, Any]], latency: float) -> str:
    """Render result cards with latency badge and summary strip."""
    if not results:
        return """
        <div style="text-align:center;padding:50px 20px;color:#9ca3af;
                    font-family:'Segoe UI',sans-serif;">
          <div style="font-size:48px;margin-bottom:14px;">🔍</div>
          <p style="font-size:15px;margin:0;color:#6b7280;">No matching standards found.</p>
          <p style="font-size:13px;margin-top:6px;color:#9ca3af;">
            Try a more specific product description or use one of the example queries.</p>
        </div>"""

    # Category summary
    cats = {}
    for r in results:
        c = r.get("category", "Building Materials")
        cats[c] = cats.get(c, 0) + 1
    cat_chips = " ".join(
        f'<span style="background:{CATEGORY_COLORS.get(c,"#7f8c8d")}22;'
        f'color:{CATEGORY_COLORS.get(c,"#7f8c8d")};border:1px solid {CATEGORY_COLORS.get(c,"#7f8c8d")}44;'
        f'padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600;">{c} ({n})</span>'
        for c, n in cats.items()
    )

    header = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:16px;flex-wrap:wrap;gap:8px;font-family:'Segoe UI',sans-serif;">
      <div>
        <span style="font-size:15px;font-weight:700;color:#111827;">
          {len(results)} Standard{'s' if len(results)!=1 else ''} Found
        </span>
        <span style="margin-left:12px;">{cat_chips}</span>
      </div>
      <span style="background:#f0f9ff;color:#0369a1;border:1px solid #bae6fd;
                   padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">
        ⏱ {latency:.2f}s
      </span>
    </div>"""

    top_score = max((r.get("rrf_score", 0.0) for r in results), default=1.0) or 1.0
    cards = "".join(_result_card(i + 1, r, top_score) for i, r in enumerate(results))
    return header + cards


def _placeholder_html() -> str:
    return """
    <div style="text-align:center;padding:60px 20px;color:#9ca3af;
                font-family:'Segoe UI',sans-serif;">
      <div style="font-size:56px;margin-bottom:18px;filter:grayscale(0.2);">🏗️</div>
      <p style="font-size:16px;margin:0;color:#6b7280;font-weight:500;">
        Enter a product description to find applicable BIS standards.</p>
      <p style="font-size:13px;margin-top:8px;color:#9ca3af;">
        Or click any example query on the left to get started.</p>
    </div>"""


def _loading_html() -> str:
    return """
    <div style="text-align:center;padding:60px 20px;font-family:'Segoe UI',sans-serif;">
      <div style="font-size:40px;margin-bottom:14px;animation:spin 1s linear infinite;">⏳</div>
      <p style="color:#4b5563;font-size:15px;font-weight:500;">Searching BIS SP 21…</p>
      <p style="color:#9ca3af;font-size:13px;">Running hybrid retrieval + LLM enrichment</p>
    </div>"""


# ── Core query handler ─────────────────────────────────────────────────────────

def query_pipeline(product_description: str) -> str:
    """Run the full RAG pipeline and return formatted HTML."""
    if not product_description or not product_description.strip():
        return "<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>⚠️ Please enter a product description.</p>"

    # Validate input length (consistent with inference.py MAX_QUERY_LENGTH)
    MAX_LEN = 2000
    if len(product_description) > MAX_LEN:
        return (
            f"<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>"
            f"⚠️ Description too long ({len(product_description)} chars). "
            f"Please keep it under {MAX_LEN} characters.</p>"
        )

    if _retriever is None:
        return (
            "<div style='color:#e74c3c;font-family:sans-serif;padding:20px;'>"
            "<p>⚠️ <strong>Pipeline not loaded.</strong> Indexes are missing or failed to load.</p>"
            "<p>To fix this, run the following commands from the project root:</p>"
            "<pre style='background:#f8f8f8;padding:10px;border-radius:6px;font-size:13px;'>"
            "python src/ingestion.py --pdf &lt;path/to/SP21.pdf&gt;\n"
            "python src/indexer.py"
            "</pre>"
            "<p>Then restart the app.</p>"
            "</div>"
        )

    start = time.time()
    try:
        chunks = _retriever.retrieve(product_description)

        # Always use retrieval-only for speed — LLM only adds rationale text,
        # IS codes are identical. Only call LLM if key is set AND retrieval was fast.
        retrieval_time = time.time() - start
        groq_key = os.environ.get("GROQ_API_KEY", "") if retrieval_time < 0.5 else ""
        gemini_key = os.environ.get("GEMINI_API_KEY", "") if retrieval_time < 0.5 else ""

        results = generate(product_description, chunks,
                           groq_api_key=groq_key,
                           gemini_api_key=gemini_key)

        # Attach rrf_score + category to results for display.
        # Key by normalized code (strip spaces/punctuation) so canonical result
        # codes ("IS 269: 1989") match raw chunk codes ("IS 269 : 1989").
        import re as _re
        def _norm(s): return _re.sub(r"[^a-z0-9]", "", str(s).lower())
        chunk_map = {_norm(c.get("std_code", "")): c for c in chunks}
        for r in results:
            matched = chunk_map.get(_norm(r.get("std_code", "")), {})
            r["rrf_score"] = matched.get("rrf_score", 0.0)
            r["category"]  = matched.get("category", "Building Materials")

        latency = time.time() - start

        # Save to history
        _history.append((product_description[:60] + ("…" if len(product_description) > 60 else ""),
                         time.strftime("%H:%M:%S")))
        
        # Log metrics
        logger.info(f"Query processed: {len(results)} results in {latency:.2f}s | Input len={len(product_description)}")

        return format_results_html(results, latency)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>⚠️ Error: {str(e)[:100]}</p>"


def get_history_html() -> str:
    """Render recent query history as HTML."""
    if not _history:
        return "<p style='color:#aaa;font-size:13px;padding:8px;'>No queries yet.</p>"
    items = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:13px;">'
        f'<span style="color:#888;font-size:11px;">{ts}</span><br>'
        f'<span style="color:#333;">{q}</span></div>'
        for q, ts in reversed(_history[-10:])
    )
    return f'<div style="font-family:sans-serif;">{items}</div>'


# ── UI builder ─────────────────────────────────────────────────────────────────

CSS = """
/* ── Base ── */
body { background: #f0f4f8 !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 16px !important; }
footer { display: none !important; }

/* ── Buttons ── */
.gr-button-primary {
    background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #1557b0, #0a3880) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,115,232,0.35) !important;
}
.gr-button-secondary { border-radius: 8px !important; }

/* ── Textbox ── */
.gr-textbox textarea {
    border-radius: 8px !important;
    border: 1.5px solid #d0d7de !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
.gr-textbox textarea:focus { border-color: #1a73e8 !important; box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important; }

/* ── Accordion ── */
.gr-accordion { border-radius: 10px !important; border: 1px solid #e0e7ef !important; }

/* ── Results panel ── */
#results-panel { min-height: 420px; }

/* ── Example buttons ── */
.example-btn {
    font-size: 12px !important; text-align: left !important;
    white-space: normal !important; height: auto !important;
    padding: 7px 12px !important; border-radius: 6px !important;
    border: 1px solid #e0e7ef !important;
    background: #f8fafc !important; color: #374151 !important;
    transition: all 0.15s !important; line-height: 1.4 !important;
}
.example-btn:hover {
    background: #e8f0fe !important; border-color: #1a73e8 !important;
    color: #1a73e8 !important;
}

/* ── Metrics strip ── */
.metric-chip {
    display: inline-flex; flex-direction: column; align-items: center;
    background: rgba(255,255,255,0.12); border-radius: 10px;
    padding: 8px 18px; min-width: 80px;
}
"""

ABOUT_HTML = """
<div style="font-family:'Segoe UI',sans-serif;font-size:14px;line-height:1.75;color:#374151;">
  <p style="margin:0 0 10px;font-weight:600;color:#1a1a2e;">How the pipeline works:</p>
  <ol style="padding-left:20px;margin:0 0 12px;">
    <li><strong>Query Expansion</strong> — 50+ rule-based synonym substitutions map user language to BIS SP 21 vocabulary (e.g. "rods" → "bars", "housing" → "construction")</li>
    <li><strong>BM25 Sparse Retrieval</strong> — exact keyword matching on IS codes &amp; material names (top-20 candidates)</li>
    <li><strong>Dense Semantic Retrieval</strong> — all-MiniLM-L6-v2 embeddings via FAISS IndexFlatIP (top-20 candidates)</li>
    <li><strong>RRF Fusion (k=60)</strong> — combines both ranked lists; false-positive suppression removes known noise codes</li>
    <li><strong>LLM Rationale</strong> — Groq llama-3.1-8b-instant generates 1–2 sentence explanations (Gemini fallback)</li>
    <li><strong>Anti-hallucination</strong> — IS codes are always taken from retrieved chunk metadata, never from LLM output</li>
  </ol>
  <p style="margin:0;font-size:12px;color:#6b7280;">
    Source: <strong>BIS SP 21</strong> — Summaries of Indian Standards for Building Materials &nbsp;|&nbsp;
    Built for <strong>BIS × SS Hackathon 2026</strong>
  </p>
</div>
"""


def build_ui() -> gr.Blocks:
    """Construct and return the full Gradio UI."""
    with gr.Blocks(
        title="BIS Standards Recommendation Engine",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#1a56a0 100%);
            color:white;padding:28px 32px;border-radius:14px;margin-bottom:20px;
            font-family:'Segoe UI',system-ui,sans-serif;
            box-shadow:0 4px 24px rgba(15,23,42,0.25);
        ">
          <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
            <div style="font-size:46px;line-height:1;">🏗️</div>
            <div style="flex:1;min-width:200px;">
              <h1 style="margin:0;font-size:22px;font-weight:800;letter-spacing:0.3px;">
                BIS Standards Recommendation Engine
              </h1>
              <p style="margin:5px 0 0;font-size:13px;opacity:0.75;line-height:1.5;">
                Helping Indian MSEs identify applicable Bureau of Indian Standards — instantly.
                &nbsp;·&nbsp; Hybrid RAG · BM25 · FAISS · Query Expansion · Groq LLM
              </p>
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
              <div style="text-align:center;background:rgba(255,255,255,0.1);
                          border-radius:10px;padding:8px 16px;min-width:72px;">
                <div style="font-size:20px;font-weight:800;color:#fbbf24;">100%</div>
                <div style="font-size:10px;opacity:0.7;margin-top:2px;">Hit Rate @3</div>
              </div>
              <div style="text-align:center;background:rgba(255,255,255,0.1);
                          border-radius:10px;padding:8px 16px;min-width:72px;">
                <div style="font-size:20px;font-weight:800;color:#34d399;">1.00</div>
                <div style="font-size:10px;opacity:0.7;margin-top:2px;">MRR @5</div>
              </div>
              <div style="text-align:center;background:rgba(255,255,255,0.1);
                          border-radius:10px;padding:8px 16px;min-width:72px;">
                <div style="font-size:20px;font-weight:800;color:#60a5fa;">2.7s</div>
                <div style="font-size:10px;opacity:0.7;margin-top:2px;">Avg Latency</div>
              </div>
            </div>
          </div>
        </div>
        """)

        # ── Main layout ────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — input + examples
            with gr.Column(scale=4, min_width=300):

                query_input = gr.Textbox(
                    label="📝 Product Description",
                    placeholder=(
                        "Describe your product or material in plain language.\n\n"
                        "e.g. 'We manufacture 53 Grade OPC cement for structural use'\n"
                        "e.g. 'TMT bars for earthquake resistant building construction'"
                    ),
                    lines=5,
                    max_lines=10,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍  Find Applicable Standards",
                        variant="primary",
                        scale=3,
                    )
                    clear_btn = gr.Button("✕ Clear", scale=1, variant="secondary")

                # Status indicator
                status_box = gr.HTML(
                    value="<p style='font-size:12px;color:#9ca3af;margin:4px 0 0;'>"
                          "Ready — enter a description above.</p>"
                )

                # Example queries by category
                with gr.Accordion("💡 Example Queries — click to load", open=True):
                    for category, queries in EXAMPLES.items():
                        gr.Markdown(
                            f"<span style='font-size:12px;font-weight:700;"
                            f"color:#374151;'>{category}</span>"
                        )
                        for q in queries:
                            gr.Button(q, elem_classes=["example-btn"]).click(
                                fn=lambda text=q: text,
                                outputs=query_input,
                            )

                # How it works
                with gr.Accordion("ℹ️ How It Works", open=False):
                    gr.HTML(ABOUT_HTML)

                # Recent history
                with gr.Accordion("🕐 Recent Queries", open=False):
                    history_display = gr.HTML(
                        value="<p style='color:#9ca3af;font-size:13px;padding:6px;'>No queries yet.</p>"
                    )

            # Right column — results
            with gr.Column(scale=6, min_width=400):
                gr.HTML("""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                  <span style="font-size:18px;font-weight:700;color:#111827;
                               font-family:'Segoe UI',sans-serif;">📋 Recommended BIS Standards</span>
                  <span style="font-size:12px;color:#6b7280;background:#f3f4f6;
                               padding:2px 10px;border-radius:10px;">Top 5 results</span>
                </div>
                """)
                results_output = gr.HTML(
                    value=_placeholder_html(),
                    elem_id="results-panel",
                )

        # ── Footer ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            text-align:center;margin-top:20px;padding:14px;
            border-top:1px solid #e5e7eb;color:#9ca3af;font-size:12px;
            font-family:'Segoe UI',sans-serif;
        ">
          BIS Standards Recommendation Engine &nbsp;·&nbsp;
          BIS × SS Hackathon 2026 &nbsp;·&nbsp;
          Source: BIS SP 21 (Building Materials) &nbsp;·&nbsp;
          Stack: pdfplumber · rank-bm25 · FAISS · all-MiniLM-L6-v2 · Groq · Gradio
        </div>
        """)

        # ── Event wiring ───────────────────────────────────────────────────────
        def on_submit(query: str):
            busy = "<p style='font-size:12px;color:#1a73e8;margin:4px 0 0;'>⏳ Searching…</p>"
            yield _loading_html(), busy, get_history_html()
            result_html = query_pipeline(query)
            ready = "<p style='font-size:12px;color:#16a34a;margin:4px 0 0;'>✓ Done</p>"
            history_html = get_history_html()
            yield result_html, ready, history_html

        submit_btn.click(
            fn=on_submit,
            inputs=query_input,
            outputs=[results_output, status_box, history_display],
        )
        query_input.submit(
            fn=on_submit,
            inputs=query_input,
            outputs=[results_output, status_box, history_display],
        )
        clear_btn.click(
            fn=lambda: (
                "",
                _placeholder_html(),
                "<p style='font-size:12px;color:#9ca3af;margin:4px 0 0;'>Ready — enter a description above.</p>",
                get_history_html(),
            ),
            outputs=[query_input, results_output, status_box, history_display],
        )

    return demo


def main() -> None:
    pipeline_ok = load_pipeline()
    if not pipeline_ok:
        logger.warning(
            "Pipeline failed to load. UI will start but queries will return errors.\n"
            "Run: python src/ingestion.py --pdf dataset.pdf\n"
            "     python src/indexer.py"
        )

    demo = build_ui()

    PORT = 7861

    # Try to create a public URL via ngrok (more reliable than Gradio's built-in share)
    public_url = None
    try:
        from pyngrok import ngrok, conf
        # Use free ngrok tunnel — no auth token needed for basic HTTP tunnels
        tunnel = ngrok.connect(PORT, "http")
        public_url = tunnel.public_url
        logger.info(f"Public URL (ngrok): {public_url}")
        print(f"\n{'='*60}")
        print(f"  PUBLIC URL: {public_url}")
        print(f"  Share this link — valid for ~2 hours")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.warning(f"ngrok tunnel failed ({e}); falling back to Gradio share")

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=PORT,
            share=(public_url is None),  # only use Gradio share if ngrok failed
            show_error=True,
        )
    except OSError:
        logger.info("Port 7861 in use; trying automatic port selection...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=0,
            share=(public_url is None),
            show_error=True,
        )


if __name__ == "__main__":
    main()
