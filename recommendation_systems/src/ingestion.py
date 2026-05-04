"""
ingestion.py — BIS SP 21 PDF Ingestion & Chunking

Parses the BIS SP 21 PDF, detects IS standard boundaries using regex,
and produces structured chunks with metadata for indexing.
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Regex to detect IS standard headers (e.g., "IS 269 : 1989" or "IS 2185 (Part 1) : 1979")
# Matches IS code followed by optional year, then a colon/dash separator and title text
IS_HEADER_PATTERN = re.compile(
    r"IS\s+\d{2,5}(?:\s*\(Part\s*\d+\))?(?:\s*[-:/]\s*\d{4})?\s*[:\-]\s*\S",
    re.IGNORECASE,
)

# Stricter pattern to extract the IS code itself
IS_CODE_PATTERN = re.compile(
    r"(IS\s+\d{2,5}(?:\s*\(Part\s*\d+\))?(?:\s*(?:[-:/]\s*)?\d{4})?)",
    re.IGNORECASE,
)

MAX_TOKENS = 800  # approximate word count limit per chunk
MIN_TOKENS = 100


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 words per token for English technical text."""
    return int(len(text.split()) * 1.3)


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_is_code(text: str) -> str:
    """Extract a clean IS code string from a line of text."""
    match = IS_CODE_PATTERN.search(text)
    if match:
        code = match.group(1)
        # Normalize spacing
        code = re.sub(r"\s+", " ", code).strip()
        # Remove trailing punctuation
        code = code.rstrip(".:,-")
        return code
    return ""


def infer_category(text: str) -> str:
    """
    Infer building material sub-category from chunk text.

    Order matters — more specific product categories are checked first
    so that standards which mention generic materials (e.g. door frames
    specifying 'use OPC cement') are not miscategorised as Cement/Steel.
    """
    text_lower = text.lower()

    # ── Specific product categories first ─────────────────────────────────────
    # Pipes, precast products, blocks, masonry units
    if any(kw in text_lower for kw in ["pipe", "precast concrete pipe", "drainage pipe",
                                        "sewer pipe", "water main"]):
        return "Concrete Products"
    # Door/window frames, partition panels — must come before cement/steel/concrete
    if any(kw in text_lower for kw in ["door frame", "window frame", "door and window",
                                        "partition", "door shutter", "ventilator frame"]):
        return "Concrete Products"
    # Asbestos / fibre cement sheets
    if any(kw in text_lower for kw in ["asbestos cement", "corrugated sheet", "roofing sheet",
                                        "fibre cement", "cladding sheet"]):
        return "Concrete Products"
    # Blocks and masonry units
    if any(kw in text_lower for kw in ["masonry unit", "masonry block", "concrete block",
                                        "hollow block", "solid block", "aerated block",
                                        "autoclaved block"]):
        return "Concrete Products"
    # Bricks and tiles
    if any(kw in text_lower for kw in ["brick", "clay brick", "burnt clay", "paver",
                                        "floor tile", "wall tile", "ceramic tile"]):
        return "Bricks & Tiles"
    # Waterproofing
    if any(kw in text_lower for kw in ["waterproof", "damp proof", "sealant", "bitumen",
                                        "membrane", "waterproofing compound"]):
        return "Waterproofing"
    # Aggregates
    if any(kw in text_lower for kw in ["aggregate", "coarse aggregate", "fine aggregate",
                                        "gravel", "crushed stone", "natural sand"]):
        return "Aggregates"

    # ── Generic material categories ────────────────────────────────────────────
    # Steel — check before concrete because reinforced concrete chunks mention both
    if any(kw in text_lower for kw in ["tmt bar", "deformed bar", "mild steel bar",
                                        "structural steel", "wire rod", "wire fabric",
                                        "epoxy coated bar", "high tensile steel",
                                        "prestressed wire", "welded wire"]):
        return "Steel"
    # Concrete — generic
    if any(kw in text_lower for kw in ["reinforced concrete", "plain concrete", "prestressed concrete",
                                        "ready mix", "concrete mix"]):
        return "Concrete"
    # Cement — only if no more specific category matched above
    if any(kw in text_lower for kw in ["portland cement", "pozzolana cement", "slag cement",
                                        "sulphate resisting cement", "supersulphated cement",
                                        "hydrophobic cement", "high alumina cement",
                                        "white cement", "masonry cement"]):
        return "Cement"
    # Broad fallbacks
    if "cement" in text_lower and "concrete" not in text_lower:
        return "Cement"
    if any(kw in text_lower for kw in ["steel", "reinforcement", "bar", "rod"]):
        return "Steel"
    if "concrete" in text_lower:
        return "Concrete"

    return "Building Materials"


def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parse BIS SP 21 PDF and return a list of structured chunks.

    Each chunk corresponds to one IS standard entry (or a sub-chunk if too long).

    Args:
        pdf_path: Path to the BIS SP 21 PDF file.

    Returns:
        List of chunk dicts with keys: std_code, title, category, page_range, text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Opening PDF: {pdf_path}")
    chunks: List[Dict[str, Any]] = []

    current_std_code: str = ""
    current_title: str = ""
    current_text_lines: List[str] = []
    current_start_page: int = 1

    def flush_chunk(end_page: int) -> None:
        """Save the accumulated text as one or more chunks."""
        nonlocal current_std_code, current_title, current_text_lines, current_start_page

        if not current_text_lines or not current_std_code:
            return

        full_text = clean_text("\n".join(current_text_lines))
        if estimate_tokens(full_text) < MIN_TOKENS:
            return

        category = infer_category(full_text)
        base_meta = {
            "std_code": current_std_code,
            "title": current_title,
            "category": category,
            "subcategory": category,
            "page": f"{current_start_page}-{end_page}",
        }

        # Split oversized chunks at paragraph boundaries
        if estimate_tokens(full_text) > MAX_TOKENS:
            paragraphs = full_text.split("\n\n")
            sub_chunk_lines: List[str] = []
            sub_idx = 0
            for para in paragraphs:
                sub_chunk_lines.append(para)
                combined = "\n\n".join(sub_chunk_lines)
                if estimate_tokens(combined) >= MAX_TOKENS:
                    chunk_text = combined.strip()
                    if estimate_tokens(chunk_text) >= MIN_TOKENS:
                        meta = dict(base_meta)
                        meta["sub_chunk"] = sub_idx
                        chunks.append({**meta, "text": chunk_text})
                        sub_idx += 1
                    sub_chunk_lines = []
            # Flush remaining
            if sub_chunk_lines:
                chunk_text = "\n\n".join(sub_chunk_lines).strip()
                if estimate_tokens(chunk_text) >= MIN_TOKENS:
                    meta = dict(base_meta)
                    meta["sub_chunk"] = sub_idx
                    chunks.append({**meta, "text": chunk_text})
        else:
            chunks.append({**base_meta, "text": full_text})

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Total pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect new IS standard boundary
                if IS_HEADER_PATTERN.search(line):
                    candidate_code = extract_is_code(line)
                    if candidate_code and candidate_code != current_std_code:
                        # Flush previous standard
                        flush_chunk(page_num - 1 if page_num > 1 else 1)

                        # Start new standard
                        current_std_code = candidate_code
                        # Title is the rest of the line after the IS code
                        title_match = re.sub(IS_CODE_PATTERN, "", line).strip()
                        title_match = re.sub(r"^[\s:.\-]+", "", title_match).strip()
                        current_title = title_match if title_match else current_std_code
                        current_text_lines = [line]
                        current_start_page = page_num
                        continue

                current_text_lines.append(line)

        # Flush the last standard
        flush_chunk(total_pages)

    logger.info(f"Extracted {len(chunks)} chunks from PDF")
    return chunks


def save_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Serialize chunks to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BIS SP 21 PDF into structured chunks")
    parser.add_argument("--pdf", type=str, required=True, help="Path to BIS SP 21 PDF")
    parser.add_argument(
        "--output", type=str, default="data/chunks.json", help="Output path for chunks JSON"
    )
    args = parser.parse_args()

    chunks = parse_pdf(args.pdf)
    save_chunks(chunks, args.output)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
