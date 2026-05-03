"""
query_expander.py — Domain-Aware Query Expansion for BIS SP 21

Bridges the gap between user language and BIS SP 21 vocabulary.

Strategy:
  1. Synonym substitution  — maps colloquial terms to BIS terminology
  2. Concept injection     — appends domain keywords for abstract concepts
  3. Multi-query fusion    — runs retrieval on original + expanded query,
                             then RRF-merges both ranked lists

This is purely rule-based (no LLM, no extra latency) and targets the
specific vocabulary gaps observed in BIS SP 21.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synonym map: user term → BIS SP 21 term(s)
# Keys are lowercase regex patterns; values are replacement/append strings.
# Order matters — more specific patterns first.
# ---------------------------------------------------------------------------
SYNONYM_MAP: List[Tuple[str, str]] = [
    # Steel / reinforcement — most specific first
    (r"\bsteel rods?\b",                "steel bars reinforcement deformed high strength IS 432 IS 1786"),
    (r"\brebar\b",                      "high strength deformed steel bars reinforcement IS 1786"),
    (r"\btmt\b",                        "high strength deformed steel bars TMT IS 1786"),
    (r"\bearthquake resistant\b",       "seismic high strength deformed steel bars ductile IS 1786"),
    (r"\bseismic\b",                    "high strength deformed steel bars ductile reinforcement IS 1786"),
    (r"\bshould not rust\b",            "corrosion resistant epoxy coated fusion bonded steel bars IS 13620"),
    (r"\bnot rust\b",                   "corrosion resistant epoxy coated fusion bonded steel bars IS 13620"),
    (r"\brust\b",                       "corrosion resistant epoxy coated weathering steel bars"),
    (r"\bcorrosion resistant steel\b",  "corrosion resistant epoxy coated weathering structural steel IS 11587 IS 13620"),
    (r"\bcorrosion resistant\b",        "corrosion resistant epoxy coated weathering steel"),
    (r"\bweathering steel\b",           "structural weather resistance steel IS 11587"),
    (r"\bcoastal steel\b",              "corrosion resistant epoxy coated weathering steel IS 11587"),

    # Cement / aggressive environments
    (r"\bsulfate attack\b",             "sulphate resisting supersulphated cement IS 12330 IS 6909"),
    (r"\bsulphate attack\b",            "sulphate resisting supersulphated cement IS 12330 IS 6909"),
    (r"\bunderground\b",                "sulphate resisting supersulphated aggressive soil IS 12330 IS 6909"),
    (r"\bmarine\b",                     "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bcoastal\b",                    "supersulphated portland slag sulphate resisting hydrophobic IS 6909 IS 8043"),
    (r"\bhumid\b",                      "hydrophobic portland cement moisture IS 8043"),
    (r"\bhigh humidity\b",              "hydrophobic portland cement moisture coastal IS 8043"),
    (r"\bsea area\b",                   "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bnear sea\b",                   "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bsea\b",                        "supersulphated portland slag sulphate resisting coastal"),
    (r"\bsaline\b",                     "supersulphated sulphate resisting portland slag"),
    (r"\blow cost cement\b",            "ordinary portland cement 33 grade masonry IS 269 IS 3466"),
    (r"\bcheap cement\b",               "ordinary portland cement 33 grade masonry IS 269"),
    (r"\bhigh strength cement\b",       "53 grade ordinary portland cement high alumina IS 8112"),
    (r"\beco.?friendly\b",              "timber bamboo fly ash pozzolana slag natural IS 399 IS 3629 IS 1489"),
    (r"\bgreen building\b",             "fly ash pozzolana slag timber bamboo IS 1489"),
    (r"\bsustainable\b",                "fly ash pozzolana slag timber bamboo IS 1489"),

    # Concrete
    (r"\bhigh strength concrete\b",     "high tensile steel bars prestressed concrete IS 2090 IS 784"),
    (r"\bprestressed\b",                "high tensile steel bars prestressed concrete IS 2090 IS 784"),
    (r"\bbridge\b",                     "high tensile prestressed concrete structural IS 2090 IS 784"),
    (r"\bcoastal concrete\b",           "sulphate resisting hydrophobic portland slag cement IS 12330 IS 8043"),
    (r"\bwaterproof concrete\b",        "integral cement waterproofing admixture IS 2645"),
    (r"\bdurable concrete\b",           "sulphate resisting portland slag high alumina cement IS 455 IS 6452"),

    # Aggregates / road
    (r"\broad construction\b",          "coarse aggregate bitumen road tar macadam IS 215 IS 6579 IS 383"),
    (r"\bheavy load\b",                 "coarse aggregate road bitumen macadam structural IS 383 IS 6579"),
    (r"\bpavement\b",                   "coarse aggregate bitumen road tar IS 215 IS 6579"),

    # Housing / low cost
    (r"\blow.?cost housing\b",          "concrete masonry blocks lime sand bricks precast IS 2185 IS 3115"),
    (r"\baffordable housing\b",         "concrete masonry blocks lime sand bricks precast IS 2185"),
    (r"\brural construction\b",         "ordinary portland cement masonry blocks lime bricks IS 269 IS 2185"),
    (r"\bhousing material\b",           "concrete masonry blocks bricks lime precast IS 2185 IS 3115"),

    # Fire resistance
    (r"\bfire resistant\b",             "fire clay refractory insulation mineral wool gypsum IS 9742 IS 8272 IS 5509"),
    (r"\bfire proof\b",                 "fire clay refractory insulation mineral wool IS 9742"),
    (r"\bfire protection\b",            "fire clay refractory insulation mineral wool IS 9742"),

    # Generic construction
    (r"\bbuilding construction\b",      "ordinary portland cement concrete masonry IS 269 IS 8112"),
    (r"\bconstruction material\b",      "cement concrete steel aggregate masonry IS 269 IS 383"),
    (r"\bgeneral purpose\b",            "ordinary portland cement 33 grade masonry IS 269 IS 3466"),
    (r"\bdurability\b",                 "sulphate resisting portland slag supersulphated IS 455 IS 6909"),
    (r"\bstrength\b",                   "high strength deformed steel bars portland cement IS 1786 IS 8112"),
]

# ---------------------------------------------------------------------------
# Concept injection: if these phrases appear, append extra BIS keywords
# ---------------------------------------------------------------------------
CONCEPT_INJECTIONS: List[Tuple[str, str]] = [
    (r"\bcement\b.*\bcoastal\b",          "IS 6909 supersulphated IS 12330 sulphate resisting IS 455 slag"),
    (r"\bcoastal\b.*\bcement\b",          "IS 6909 supersulphated IS 12330 sulphate resisting IS 455 slag"),
    (r"\bsteel\b.*\breinforcement\b",     "IS 1786 deformed bars IS 432 mild steel IS 2090 high tensile"),
    (r"\bsteel\b.*\bearthquake\b",        "IS 1786 high strength deformed bars ductile seismic"),
    (r"\bsteel\b.*\brust\b",              "IS 13620 fusion bonded epoxy coated IS 1786 deformed bars IS 432"),
    (r"\bsteel\b.*\bconstruction\b",      "IS 432 mild steel IS 1786 deformed bars IS 2090 high tensile"),
    (r"\brods?\b.*\bconstruction\b",      "IS 432 mild steel bars IS 1786 deformed bars reinforcement"),
    (r"\bconstruction\b.*\brods?\b",      "IS 432 mild steel bars IS 1786 deformed bars reinforcement"),
    (r"\bfire\b.*\bresist\b",             "IS 9742 mineral wool IS 8272 gypsum IS 4832 chemical resistant IS 5509 fire retardant"),
    (r"\bsulfate\b|\bsulphate\b",         "IS 12330 sulphate resisting IS 6909 supersulphated IS 455 slag"),
    (r"\bbridge\b",                       "IS 2090 high tensile IS 1786 deformed bars prestressed IS 784"),
    (r"\broad\b",                         "IS 215 road tar IS 6579 coarse aggregate IS 383 aggregate"),
    (r"\beco.?friendly\b",                "IS 399 timber IS 3629 structural timber IS 1489 pozzolana fly ash"),
    (r"\blow.?cost\b",                    "IS 269 ordinary portland IS 3466 masonry cement IS 2185 concrete blocks"),
]


def expand_query(query: str) -> str:
    """
    Expand a user query by substituting synonyms and injecting domain concepts.

    For queries where user language is far from BIS vocabulary (e.g. "rods" vs
    "bars"), the original query is rewritten rather than just appended to, so
    BM25 doesn't get anchored on the wrong term.

    Args:
        query: Raw user query string.

    Returns:
        Expanded query string with BIS-vocabulary terms appended/substituted.
    """
    q = query.strip()
    q_lower = q.lower()
    additions: List[str] = []

    # ── Special case: rewrite queries where user language badly misleads BM25 ──
    # "steel rods ... construction ... rust/corrosion" → rewrite to bar terminology
    if (re.search(r"\bsteel rods?\b", q_lower)
            and re.search(r"\bconstruction\b|\breinforcement\b|\bbuilding\b", q_lower)):
        q = re.sub(r"\bsteel rods?\b", "steel bars reinforcement", q, flags=re.IGNORECASE)
        q_lower = q.lower()
        additions.append("deformed high strength IS 432 IS 1786")

    # Step 1: synonym substitution — collect expansion terms
    for pattern, replacement in SYNONYM_MAP:
        if re.search(pattern, q_lower):
            additions.append(replacement)
            logger.debug(f"Synonym match [{pattern}] → appending: {replacement}")

    # Step 2: concept injection — append IS code hints for known concept combos
    for pattern, injection in CONCEPT_INJECTIONS:
        if re.search(pattern, q_lower):
            additions.append(injection)
            logger.debug(f"Concept injection [{pattern}] → appending: {injection}")

    if additions:
        expanded = q + " " + " ".join(additions)
        logger.info(f"Query expanded: {len(query)} → {len(expanded)} chars")
        return expanded

    return q


def get_query_variants(query: str) -> List[str]:
    """
    Return query variants for multi-query retrieval.

    - If expansion rewrites the query (e.g. "rods" → "bars"), returns only
      the rewritten version to avoid the original misleading BM25.
    - If expansion only appends terms, returns [original, expanded] so both
      contribute to RRF fusion.
    - If no expansion applies, returns [original].

    Args:
        query: Raw user query string.

    Returns:
        List of query strings to retrieve against.
    """
    q = query.strip()
    q_lower = q.lower()

    # Check if a rewrite rule applies (these change the query, not just append)
    rewrite_applied = bool(
        re.search(r"\bsteel rods?\b", q_lower)
        and re.search(r"\bconstruction\b|\breinforcement\b|\bbuilding\b", q_lower)
    )

    expanded = expand_query(query)

    if expanded == query:
        return [query]

    if rewrite_applied:
        # Only use the rewritten+expanded version — original would mislead BM25
        return [expanded]

    # Append-only expansion: run both for broader coverage
    return [query, expanded]
