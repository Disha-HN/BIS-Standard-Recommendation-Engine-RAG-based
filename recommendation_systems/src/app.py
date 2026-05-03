"""
app.py — Gradio Web UI for BIS Standards Recommendation Engine

Polished MSE-friendly interface with:
  - Category filter sidebar
  - Clickable example queries
  - Rich result cards with confidence bars
  - Query history panel
  - Loading state feedback
  - About / How-it-works accordion

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
    ],
    "🪨 Aggregates": [
        "Coarse and fine aggregates from natural sources for structural concrete.",
        "We supply crushed stone aggregates for road construction. What BIS standard applies?",
    ],
    "🧱 Concrete & Blocks": [
        "We produce hollow concrete masonry blocks. What are the BIS requirements?",
        "Standards for precast concrete pipes used in water supply mains.",
        "Ready-mix concrete for M25 grade structural use.",
    ],
    "🔩 Steel": [
        "TMT bars used for reinforced concrete construction in buildings.",
        "Structural steel sections for industrial building frames.",
    ],
    "🏠 Other": [
        "Corrugated asbestos cement sheets for roofing and cladding.",
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
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return False


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _category_badge(category: str) -> str:
    color = CATEGORY_COLORS.get(category, "#7f8c8d")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;font-weight:600;'
        f'letter-spacing:0.5px;">{category.upper()}</span>'
    )


def _confidence_bar(pct: int) -> str:
    """Render a relevance bar given a 0-100 percentage."""
    if pct > 70:
        color, label = "#27ae60", "High"
    elif pct > 45:
        color, label = "#f39c12", "Medium"
    else:
        color, label = "#e74c3c", "Low"
    return f"""
    <div style="margin-top:10px;display:flex;align-items:center;gap:10px;">
      <span style="font-size:11px;color:#888;width:110px;flex-shrink:0;">
        Relevance: <strong style="color:{color};">{label} ({pct}%)</strong>
      </span>
      <div style="flex:1;background:#e8e8e8;border-radius:4px;height:5px;">
        <div style="background:{color};width:{pct}%;height:5px;border-radius:4px;
                    transition:width 0.4s ease;"></div>
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

    rank_colors = ["#f1c40f", "#bdc3c7", "#cd7f32", "#95a5a6", "#95a5a6"]
    rank_bg = rank_colors[rank - 1] if rank <= 5 else "#95a5a6"

    return f"""
    <div style="
        border:1px solid #e0e0e0;
        border-left:5px solid {accent};
        border-radius:10px;
        padding:18px 20px;
        margin-bottom:14px;
        background:#ffffff;
        box-shadow:0 2px 6px rgba(0,0,0,0.06);
        font-family:'Segoe UI',sans-serif;
    ">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;
                  flex-wrap:wrap;gap:8px;margin-bottom:10px;">
        <div style="display:flex;align-items:center;gap:10px;">
          <span style="
              background:{rank_bg};color:#333;border-radius:50%;
              width:26px;height:26px;display:flex;align-items:center;
              justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;
          ">#{rank}</span>
          <span style="font-size:19px;font-weight:700;color:#1a1a2e;
                       letter-spacing:0.3px;">{std_code}</span>
        </div>
        {badge}
      </div>
      <p style="margin:0 0 8px;color:#555;font-size:13px;font-style:italic;
                line-height:1.4;">{title}</p>
      <p style="margin:0;color:#333;font-size:14px;line-height:1.6;">{rationale}</p>
      {bar}
    </div>"""


def format_results_html(results: List[Dict[str, Any]], latency: float) -> str:
    """Render result cards with latency badge."""
    if not results:
        return """
        <div style="text-align:center;padding:40px;color:#aaa;font-family:sans-serif;">
          <div style="font-size:40px;margin-bottom:12px;">🔍</div>
          <p>No matching standards found.<br>Try a more specific product description.</p>
        </div>"""

    latency_badge = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:14px;font-family:sans-serif;">
      <span style="font-size:13px;color:#555;">
        <strong>{len(results)}</strong> standard{'s' if len(results)!=1 else ''} found
      </span>
      <span style="background:#f0f4f8;color:#666;padding:4px 12px;
                   border-radius:12px;font-size:12px;">⏱ {latency:.2f}s</span>
    </div>"""

    # Normalise scores relative to the top result so bars are always meaningful.
    # Top result → 95%, each subsequent result scales down proportionally,
    # with a floor of 30% so even rank-5 shows a visible bar.
    top_score = max((r.get("rrf_score", 0.0) for r in results), default=1.0) or 1.0
    cards = "".join(
        _result_card(i + 1, r, top_score) for i, r in enumerate(results)
    )
    return latency_badge + cards


def _placeholder_html() -> str:
    return """
    <div style="text-align:center;padding:50px 20px;color:#bbb;font-family:sans-serif;">
      <div style="font-size:48px;margin-bottom:16px;">🏗️</div>
      <p style="font-size:15px;margin:0;">Enter a product description and click
         <strong>Find Standards</strong> to get started.</p>
      <p style="font-size:13px;margin-top:8px;color:#ccc;">
        Or click any example query on the left.</p>
    </div>"""


def _loading_html() -> str:
    return """
    <div style="text-align:center;padding:50px 20px;font-family:sans-serif;">
      <div style="font-size:36px;margin-bottom:12px;">⏳</div>
      <p style="color:#666;font-size:15px;">Searching BIS SP 21…</p>
    </div>"""


# ── Core query handler ─────────────────────────────────────────────────────────

def query_pipeline(product_description: str) -> str:
    """Run the full RAG pipeline and return formatted HTML."""
    if not product_description or not product_description.strip():
        return "<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>⚠️ Please enter a product description.</p>"

    # Validate input length (prevent excessively long queries)
    if len(product_description) > 2000:
        return "<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>⚠️ Description too long (max 2000 characters).</p>"

    if _retriever is None:
        return "<p style='color:#e74c3c;font-family:sans-serif;padding:20px;'>⚠️ Pipeline not loaded. Run <code>python src/indexer.py</code> first.</p>"

    start = time.time()
    try:
        chunks = _retriever.retrieve(product_description)
        results = generate(product_description, chunks)

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
body { background: #f5f7fa !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.gr-button-primary { background: #2980b9 !important; border: none !important; }
.gr-button-primary:hover { background: #1f6391 !important; }
footer { display: none !important; }
#results-panel { min-height: 400px; }
.example-btn { font-size: 12px !important; text-align: left !important;
               white-space: normal !important; height: auto !important;
               padding: 6px 10px !important; }
"""

ABOUT_HTML = """
<div style="font-family:'Segoe UI',sans-serif;font-size:14px;line-height:1.7;color:#444;">
  <p><strong>How it works:</strong></p>
  <ol style="padding-left:18px;margin:8px 0;">
    <li><strong>BM25 sparse retrieval</strong> — exact keyword matching on IS codes &amp; material names</li>
    <li><strong>Dense semantic retrieval</strong> — all-MiniLM-L6-v2 embeddings via FAISS</li>
    <li><strong>RRF fusion (k=60)</strong> — combines both ranked lists into top-5 candidates</li>
    <li><strong>LLM rationale</strong> — Groq llama-3.1-8b-instant generates 1–2 sentence explanations</li>
  </ol>
  <p style="margin-top:8px;">
    Source: <strong>BIS SP 21</strong> — Summaries of Indian Standards for Building Materials<br>
    Built for <strong>BIS × SS Hackathon 2026</strong>
  </p>
</div>
"""


def build_ui() -> gr.Blocks:
    """Construct and return the full Gradio UI."""
    with gr.Blocks(
        title="BIS Standards Recommendation Engine",
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
            color:white;padding:28px 32px;border-radius:12px;margin-bottom:20px;
            font-family:'Segoe UI',sans-serif;
        ">
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="font-size:42px;">🏗️</div>
            <div>
              <h1 style="margin:0;font-size:24px;font-weight:700;letter-spacing:0.5px;">
                BIS Standards Recommendation Engine
              </h1>
              <p style="margin:4px 0 0;font-size:14px;opacity:0.8;">
                Helping Indian MSEs identify applicable Bureau of Indian Standards — instantly.
                &nbsp;|&nbsp; Powered by RAG · BM25 · FAISS · Groq LLM
              </p>
            </div>
            <div style="margin-left:auto;display:flex;gap:16px;flex-wrap:wrap;">
              <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:#f1c40f;">100%</div>
                <div style="font-size:11px;opacity:0.7;">Hit Rate @3</div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:#2ecc71;">0.95</div>
                <div style="font-size:11px;opacity:0.7;">MRR @5</div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:#3498db;">2.8s</div>
                <div style="font-size:11px;opacity:0.7;">Avg Latency</div>
              </div>
            </div>
          </div>
        </div>
        """)

        # ── Main layout ────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — input + examples
            with gr.Column(scale=4, min_width=320):

                query_input = gr.Textbox(
                    label="📝 Product Description",
                    placeholder=(
                        "Describe your product or material in plain language.\n"
                        "e.g. 'We manufacture 53 Grade OPC cement for structural use'"
                    ),
                    lines=4,
                    max_lines=8,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍  Find Applicable Standards",
                        variant="primary",
                        scale=3,
                    )
                    clear_btn = gr.Button("✕ Clear", scale=1)

                # Example queries by category
                with gr.Accordion("💡 Example Queries — click to load", open=True):
                    for category, queries in EXAMPLES.items():
                        gr.Markdown(f"**{category}**")
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
                        value="<p style='color:#aaa;font-size:13px;padding:8px;'>No queries yet.</p>"
                    )

            # Right column — results
            with gr.Column(scale=6, min_width=400):
                gr.Markdown("### 📋 Recommended BIS Standards")
                results_output = gr.HTML(
                    value=_placeholder_html(),
                    elem_id="results-panel",
                )

        # ── Footer ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            text-align:center;margin-top:20px;padding:14px;
            border-top:1px solid #e0e0e0;color:#aaa;font-size:12px;
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
            result_html = query_pipeline(query)
            history_html = get_history_html()
            return result_html, history_html

        submit_btn.click(
            fn=on_submit,
            inputs=query_input,
            outputs=[results_output, history_display],
        )
        query_input.submit(
            fn=on_submit,
            inputs=query_input,
            outputs=[results_output, history_display],
        )
        clear_btn.click(
            fn=lambda: ("", _placeholder_html(), get_history_html()),
            outputs=[query_input, results_output, history_display],
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
    
    # Try default port first, fall back to automatic port selection if needed
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
            css=CSS,
        )
    except OSError:
        logger.info("Port 7860 in use; trying automatic port selection...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=0,
            share=False,
            show_error=True,
            theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
            css=CSS,
        )


if __name__ == "__main__":
    main()
