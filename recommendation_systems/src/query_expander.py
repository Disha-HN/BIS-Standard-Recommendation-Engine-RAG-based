"""
query_expander.py — Domain-Aware Query Expansion for BIS SP 21

Bridges the gap between user language and BIS SP 21 vocabulary.

Strategy:
  1. Synonym substitution  — maps colloquial terms to BIS terminology
  2. Concept injection     — appends domain keywords for abstract concepts
  3. Negative-term rewrite — strips terms that mislead BM25 into wrong domains
  4. Multi-query fusion    — runs retrieval on original + expanded query,
                             then RRF-merges both ranked lists

This is purely rule-based (no LLM, no extra latency) and targets the
specific vocabulary gaps observed in BIS SP 21.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negative-term blocklist: these user terms have NO meaning in BIS SP 21 and
# actively mislead BM25/dense retrieval.  They are stripped from the query
# before retrieval so the remaining terms can match correctly.
# ---------------------------------------------------------------------------
NEGATIVE_TERMS: List[str] = [
    # "housing" in BIS SP 21 = door/lock housing hardware, NOT residential housing.
    # Stripping it prevents "low cost housing" → door-stopper false positives.
    r"\bhousing\b",
    # "cellular" alone maps to cellular concrete (thermal insulation), not general concrete.
    # Only strip when NOT preceded by "cellular concrete" as an intentional query.
]

# ---------------------------------------------------------------------------
# Synonym map: user term → BIS SP 21 term(s)
# Keys are lowercase regex patterns; values are replacement/append strings.
# Order matters — more specific patterns first.
# ---------------------------------------------------------------------------
SYNONYM_MAP: List[Tuple[str, str]] = [
    # Precast concrete door/window frames → IS 6523
    (r"\bdoor\s*frame\b|\bwindow\s*frame\b",
                                        "precast reinforced concrete door window frames IS 6523"),
    (r"\bprecast\b.*\bdoor\b|\bdoor\b.*\bprecast\b",
                                        "precast reinforced concrete door window frames IS 6523"),

    # Concrete masonry blocks — Part disambiguation
    # "lightweight" is the key word that distinguishes Part 2 from Part 1.
    # Dense retrieval ranks Part 1 higher because both chunks share most text.
    # Injecting "lightweight" + IS 2185 Part 2 forces BM25 to boost the right chunk.
    (r"\blightweight\b.*\bblock\b|\bblock\b.*\blightweight\b",
                                        "lightweight concrete masonry blocks IS 2185 Part 2 hollow solid"),
    (r"\blightweight concrete\b",       "lightweight concrete masonry blocks IS 2185 Part 2 hollow solid"),
    (r"\baac\b",                        "autoclaved aerated cellular concrete blocks IS 2185 Part 3"),
    (r"\baerated\b.*\bblock\b|\bblock\b.*\baerated\b",
                                        "autoclaved aerated cellular concrete blocks IS 2185 Part 3"),
    (r"\bautoclaved\b",                 "autoclaved cellular aerated concrete blocks IS 2185 Part 3"),

    # Steel / reinforcement — most specific first
    (r"\bsteel rods?\b",                "steel bars reinforcement deformed high strength IS 432 IS 1786"),
    (r"\biron rods?\b",                 "steel bars reinforcement deformed high strength IS 432 IS 1786"),
    (r"\biron bars?\b",                 "steel bars reinforcement deformed high strength IS 432 IS 1786"),
    (r"\brebar\b",                      "high strength deformed steel bars reinforcement IS 1786"),
    (r"\btmt\b",                        "high strength deformed steel bars TMT IS 1786 IS 432"),
    # Earthquake / seismic — IS 1786 is the primary standard, must rank #1
    # IS 1599 (bend test) is noise — suppress it via false-positive table
    (r"\bearthquake.?resistant\b",      "IS 1786 high strength deformed steel bars seismic ductile IS 432"),
    (r"\bearthquake.?proof\b",          "IS 1786 high strength deformed steel bars seismic ductile IS 432"),
    (r"\bearthquake\b",                 "IS 1786 high strength deformed steel bars seismic ductile IS 432"),
    (r"\bseismic\b",                    "IS 1786 high strength deformed steel bars ductile reinforcement IS 432"),
    (r"\bshould not rust\b",            "corrosion resistant epoxy coated fusion bonded steel bars IS 13620"),
    (r"\bnot rust\b",                   "corrosion resistant epoxy coated fusion bonded steel bars IS 13620"),
    (r"\brust\b",                       "corrosion resistant epoxy coated weathering steel bars"),
    (r"\bcorrosion resistant steel\b",  "corrosion resistant epoxy coated weathering structural steel IS 11587 IS 13620"),
    (r"\bcorrosion resistant\b",        "corrosion resistant epoxy coated weathering steel IS 11587 IS 13620"),
    (r"\bweathering steel\b",           "structural weather resistance steel IS 11587"),
    (r"\bcoastal steel\b",              "corrosion resistant epoxy coated weathering steel IS 11587"),

    # Cement abbreviations / grade names
    (r"\bopc\b",                        "ordinary portland cement IS 269 IS 8112 IS 8041"),
    (r"\bopc\s*33\b",                   "ordinary portland cement 33 grade IS 269"),
    (r"\bopc\s*43\b",                   "ordinary portland cement 43 grade IS 8015"),
    (r"\bopc\s*53\b",                   "ordinary portland cement 53 grade IS 8112"),
    (r"\b53\s*grade\b",                 "53 grade ordinary portland cement IS 8112"),
    (r"\b43\s*grade\b",                 "43 grade ordinary portland cement IS 8015"),
    (r"\b33\s*grade\b",                 "33 grade ordinary portland cement IS 269"),
    (r"\bppc\b",                        "portland pozzolana cement fly ash IS 1489"),
    (r"\bpsc\b",                        "portland slag cement IS 455"),
    (r"\bggbs\b",                       "ground granulated blast furnace slag portland slag cement IS 455"),
    (r"\brcc\b",                        "reinforced cement concrete IS 8112 IS 1786 IS 432 high strength"),
    (r"\bpcc\b",                        "plain cement concrete IS 269 IS 383 aggregate"),
    (r"\bm\s*(?:20|25|30|35|40)\b",     "concrete mix grade IS 8112 IS 383 aggregate cement"),
    # Low heat cement — for dams, mass concrete
    (r"\blow.?heat\b",                  "low heat portland cement IS 12600 dam mass concrete"),
    (r"\bdam\b",                        "low heat portland cement IS 12600 mass concrete"),
    # Structural steel sections
    (r"\bismb\b|\bismc\b|\bisht\b|\bstructural steel section\b",
                                        "structural steel sections IS 2062 IS 808 rolled beams channels"),
    (r"\bstructural steel\b",           "structural steel IS 2062 IS 808 rolled sections beams"),
    # Galvanised / coated wire
    (r"\bgalvanised\b|\bgalvanized\b",  "galvanised steel wire IS 280 IS 4826 zinc coated"),
    # Burnt clay bricks
    (r"\bburnt clay brick\b|\bclay brick\b|\bcommon brick\b",
                                        "burnt clay building bricks IS 1077 common burnt clay"),
    (r"\bfirst class brick\b|\bsecond class brick\b",
                                        "burnt clay building bricks IS 1077 common burnt clay"),
    # Paving / interlocking blocks
    (r"\bpaving block\b|\binterlocking block\b|\bpaver\b",
                                        "concrete paving blocks IS 15658 interlocking paver"),
    # RCC pipes / drainage pipes
    (r"\brcc pipe\b|\bconcrete pipe\b|\bdrainage pipe\b|\bsewer pipe\b",
                                        "precast concrete pipes IS 458 reinforced drainage sewer"),
    # Asbestos cement sheets — IS 459 is corrugated, IS 1626 is flat/building
    (r"\basbestos cement sheet\b|\basbestos sheet\b",
                                        "asbestos cement sheets IS 459 corrugated roofing IS 1626 building"),
    (r"\broofing sheet\b",              "asbestos cement corrugated sheets IS 459 roofing cladding"),
    (r"\bindustrial shed\b|\bfactory roof\b|\bwarehouse roof\b",
                                        "asbestos cement corrugated sheets IS 459 IS 1626 roofing"),
    # Fine aggregate / river sand
    (r"\bfine aggregate\b",             "fine aggregate natural sources IS 383 sand concrete"),
    (r"\briver sand\b",                 "fine aggregate natural sources IS 383 sand concrete"),
    # Ceramic / vitrified tiles
    (r"\bceramic tile\b|\bvitrified tile\b|\bfloor tile\b|\bwall tile\b",
                                        "ceramic tiles IS 13630 IS 777 glazed floor wall"),
    (r"\btile\b.*\bfloor\b|\bfloor\b.*\btile\b",
                                        "ceramic tiles IS 13630 IS 777 glazed floor"),
    # Hindi / Indian colloquial terms
    (r"\bsariya\b|\bsaria\b",           "steel bars reinforcement deformed IS 1786 IS 432 high strength"),
    (r"\bpucca\b",                      "ordinary portland cement IS 269 IS 8112 permanent construction"),
    (r"\bgitti\b|\bstone chips\b|\bstone grit\b",
                                        "coarse aggregate crushed stone IS 383 concrete"),
    (r"\bbalu\b|\bballu\b",             "fine aggregate sand IS 383 concrete"),
    (r"\breta\b|\bbadarpur\b",          "fine aggregate sand IS 383 IS 1542 plaster"),
    (r"\bkachcha\b",                    "ordinary portland cement IS 269 masonry"),
    # Structural steel — ISMB/ISMC sections need IS 2062 (material) + IS 808 (dimensions)
    (r"\bbeam\b.*\bsteel\b|\bsteel\b.*\bbeam\b|\bchannel\b.*\bsteel\b",
                                        "structural steel IS 2062 IS 808 rolled sections ISMB ISMC"),
    (r"\bangle\b.*\bsteel\b|\bsteel\b.*\bangle\b",
                                        "structural steel IS 2062 IS 808 rolled sections angles"),
    # Galvanised wire — IS 280 (plain) + IS 4826 (coated)
    (r"\bgalvanised wire\b|\bgalvanized wire\b|\bzinc coated wire\b",
                                        "galvanised steel wire IS 280 IS 4826 zinc coated fencing"),
    # Burnt clay bricks — IS 1077 is the main standard
    (r"\bbrick\b.*\bclay\b|\bclay\b.*\bbrick\b|\bburnt brick\b|\bfired brick\b",
                                        "burnt clay building bricks IS 1077 common burnt clay"),
    (r"\bbuilding brick\b|\bwall brick\b",
                                        "burnt clay building bricks IS 1077 common burnt clay"),
    # Concrete paving / interlocking blocks — IS 15658
    (r"\bpaving\b|\binterlocking\b|\bpaver block\b",
                                        "concrete paving blocks IS 15658 interlocking precast"),
    # Asbestos corrugated sheets — IS 459 specifically
    (r"\bcorrugated\b.*\basbestos\b|\basbestos\b.*\bcorrugated\b",
                                        "corrugated asbestos cement sheets IS 459 roofing"),
    (r"\basbestos\b.*\broof\b|\broof\b.*\basbestos\b",
                                        "asbestos cement corrugated sheets IS 459 roofing cladding"),
    # Ceramic / floor tiles — IS 13630 (ceramic) + IS 777 (glazed)
    (r"\bceramic\b|\bvitrified\b|\bglazed tile\b",
                                        "ceramic tiles IS 13630 IS 777 glazed floor wall vitrified"),
    (r"\bfloor tile\b|\bwall tile\b|\btile\b.*\bfloor\b",
                                        "ceramic tiles IS 13630 IS 777 glazed floor wall"),
    # Pucca house → permanent construction → OPC cement
    (r"\bpucca\b|\bpakka\b",            "ordinary portland cement IS 269 IS 8112 permanent masonry construction"),
    # Sariya → steel reinforcement bars
    (r"\bsariya\b|\bsaria\b|\blohiya\b",
                                        "high strength deformed steel bars IS 1786 IS 432 reinforcement"),

    # Cement / aggressive environments
    (r"\bsulfate attack\b",             "sulphate resisting supersulphated cement IS 12330 IS 6909"),
    (r"\bsulphate attack\b",            "sulphate resisting supersulphated cement IS 12330 IS 6909"),
    (r"\bunderground\b",                "sulphate resisting supersulphated aggressive soil IS 12330 IS 6909"),
    (r"\bmarine\b",                     "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    # "coastal" / "salty" / "ocean" / "sea" — all map to sulphate-resistant cements
    # "salty" and "ocean" were missing — IS 2645 (waterproofing) was ranking first
    (r"\bcoastal\b",                    "supersulphated portland slag sulphate resisting hydrophobic IS 6909 IS 8043 IS 455"),
    (r"\bsalt(?:y|water)?\b",           "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455 IS 12330"),
    (r"\bocean\b",                      "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455 IS 12330"),
    (r"\bseawater\b",                   "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bhumid\b",                      "hydrophobic portland cement moisture IS 8043"),
    (r"\bhigh humidity\b",              "hydrophobic portland cement moisture coastal IS 8043"),
    (r"\bsea area\b",                   "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bnear sea\b",                   "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bsea\b",                        "supersulphated portland slag sulphate resisting coastal IS 6909 IS 455"),
    (r"\bsaline\b",                     "supersulphated sulphate resisting portland slag IS 6909 IS 455"),
    (r"\blow cost cement\b",            "ordinary portland cement 33 grade masonry IS 269 IS 3466"),
    (r"\bcheap cement\b",               "ordinary portland cement 33 grade masonry IS 269"),
    (r"\bhigh strength cement\b",       "53 grade ordinary portland cement IS 8112"),
    (r"\beco.?friendly\b",              "timber bamboo fly ash pozzolana slag natural IS 399 IS 3629 IS 1489"),
    (r"\bgreen building\b",             "fly ash pozzolana slag timber bamboo IS 1489"),
    (r"\bsustainable\b",                "fly ash pozzolana slag timber bamboo IS 1489"),

    # Concrete — specific types
    (r"\bhigh strength concrete\b",     "high tensile steel bars prestressed concrete IS 2090 IS 784"),
    (r"\bprestressed\b",                "high tensile steel bars prestressed concrete IS 2090 IS 784"),
    (r"\bbridge\b",                     "high tensile prestressed concrete structural IS 2090 IS 784"),
    (r"\bcoastal concrete\b",           "sulphate resisting hydrophobic portland slag cement IS 12330 IS 8043"),
    (r"\bwaterproof concrete\b",        "integral cement waterproofing admixture IS 2645"),
    (r"\bdurable concrete\b",           "sulphate resisting portland slag high alumina cement IS 455 IS 6452"),

    # Aggregates / road — road tar IS 215 is the primary road standard
    (r"\broad construction\b",          "road tar IS 215 coarse aggregate IS 383 IS 6579 macadam"),
    (r"\broad.?building\b",             "road tar IS 215 coarse aggregate IS 383 IS 6579"),
    (r"\bheavy\s+(?:traffic|load)\b.*\broad\b|\broad\b.*\bheavy\s+(?:traffic|load)\b",
                                        "road tar IS 215 coarse aggregate IS 383 IS 6579 macadam"),
    (r"\bheavy load\b",                 "coarse aggregate structural IS 383 IS 6579"),
    (r"\bpavement\b",                   "coarse aggregate road tar IS 215 IS 6579"),
    (r"\bbitumen\b",                    "road tar bitumen macadam IS 215 IS 5317"),
    (r"\bmacadam\b",                    "road tar bitumen coarse aggregate IS 215 IS 6579 IS 383"),

    # Low-cost / affordable construction — map to masonry/blocks, NOT housing hardware
    (r"\blow.?cost\s+(?:construction|building|material|cement|concrete|block)\b",
                                        "ordinary portland cement masonry blocks lime bricks IS 269 IS 2185 IS 3115 IS 3466"),
    (r"\baffordable\s+(?:construction|building|material|cement|concrete|block)\b",
                                        "concrete masonry blocks lime sand bricks precast IS 2185 IS 3115"),
    (r"\brural construction\b",         "ordinary portland cement masonry blocks lime bricks IS 269 IS 2185"),
    (r"\bhousing material\b",           "concrete masonry blocks bricks lime precast IS 2185 IS 3115"),
    (r"\bresidential construction\b",   "ordinary portland cement masonry blocks lime bricks IS 269 IS 2185"),
    # "cement blocks" / "concrete blocks" near sea → masonry units + sulphate cement
    (r"\bcement\s+blocks?\b|\bconcrete\s+blocks?\b|\bmasonry\s+blocks?\b",
                                        "concrete masonry blocks IS 2185 hollow solid precast IS 12440"),
    (r"\bmanufacture\b.*\bblocks?\b|\bblocks?\b.*\bmanufacture\b",
                                        "concrete masonry blocks IS 2185 hollow solid precast IS 12440"),
    (r"\bblocks?\b.*(?:\bsea\b|\bcoastal\b|\bmarine\b|\bdurabilit\b)"
     r"|(?:\bsea\b|\bcoastal\b|\bmarine\b).*\bblocks?\b",
                                        "concrete masonry blocks IS 2185 IS 12440 precast IS 455 IS 6909"),

    # Fire resistance
    (r"\bfire.?resistant\b",            "fire clay refractory insulation mineral wool gypsum IS 9742 IS 8272 IS 5509"),
    (r"\bfire.?proof\b",                "fire clay refractory insulation mineral wool IS 9742"),
    (r"\bfire.?protection\b",           "fire clay refractory insulation mineral wool IS 9742"),
    (r"\bfire retardant\b",             "fire retardant plywood mineral wool IS 5509 IS 9742"),

    # Generic construction — be careful not to over-expand
    (r"\bbuilding construction\b",      "ordinary portland cement concrete masonry IS 269 IS 8112"),
    (r"\bconstruction material\b",      "cement concrete steel aggregate masonry IS 269 IS 383"),
    (r"\bgeneral purpose\b",            "ordinary portland cement 33 grade masonry IS 269 IS 3466"),
    (r"\bdurability\b",                 "sulphate resisting portland slag supersulphated IS 455 IS 6909"),
    # "strength" alone is too generic — only expand when combined with cement/concrete context
    (r"\bhigh\s+strength\s+cement\b",   "53 grade ordinary portland cement IS 8112"),
    (r"\bhigh\s+strength\s+concrete\b", "high tensile steel bars prestressed concrete IS 2090 IS 784"),
]

# ---------------------------------------------------------------------------
# Concept injection: if these phrases appear, append extra BIS keywords.
# These fire AFTER synonym substitution on the (possibly rewritten) query.
# ---------------------------------------------------------------------------
CONCEPT_INJECTIONS: List[Tuple[str, str]] = [
    (r"\bcement\b.*\bcoastal\b",          "IS 6909 supersulphated IS 12330 sulphate resisting IS 455 slag"),
    (r"\bcoastal\b.*\bcement\b",          "IS 6909 supersulphated IS 12330 sulphate resisting IS 455 slag"),
    # Lightweight blocks → always inject Part 2 code
    (r"\blightweight\b.*\bblock\b|\bblock\b.*\blightweight\b",
                                          "IS 2185 Part 2 lightweight hollow solid concrete masonry blocks"),
    (r"\blightweight concrete\b",         "IS 2185 Part 2 lightweight hollow solid concrete masonry blocks"),
    # AAC / autoclaved aerated → Part 3
    (r"\baac\b|\bautoclaved\b|\baerated\b.*\bblock\b|\bblock\b.*\baerated\b",
                                          "IS 2185 Part 3 autoclaved cellular aerated concrete blocks"),
    (r"\bsteel\b.*\breinforcement\b",     "IS 1786 deformed bars IS 432 mild steel IS 2090 high tensile"),
    # Earthquake / seismic — IS 1786 must dominate; put it first in injection
    (r"\bsteel\b.*\bearthquake\b|\bearthquake\b.*\bsteel\b",
                                          "IS 1786 high strength deformed bars ductile seismic IS 432"),
    (r"\bearthquake\b|\bseismic\b",       "IS 1786 high strength deformed bars ductile IS 432 mild steel reinforcement"),
    (r"\bsteel\b.*\brust\b",              "IS 13620 fusion bonded epoxy coated IS 1786 deformed bars IS 432"),
    (r"\bsteel\b.*\bconstruction\b",      "IS 432 mild steel IS 1786 deformed bars IS 2090 high tensile"),
    (r"\brods?\b.*\bconstruction\b",      "IS 432 mild steel bars IS 1786 deformed bars reinforcement"),
    (r"\bconstruction\b.*\brods?\b",      "IS 432 mild steel bars IS 1786 deformed bars reinforcement"),
    (r"\bfire\b.*\bresist\b",             "IS 9742 mineral wool IS 8272 gypsum IS 4832 chemical resistant IS 5509 fire retardant"),
    (r"\bsulfate\b|\bsulphate\b",         "IS 12330 sulphate resisting IS 6909 supersulphated IS 455 slag"),
    (r"\bbridge\b",                       "IS 2090 high tensile IS 1786 deformed bars prestressed IS 784"),
    (r"\broad\b",                         "IS 215 road tar IS 6579 coarse aggregate IS 383 aggregate"),
    (r"\beco.?friendly\b",                "IS 399 timber IS 3629 structural timber IS 1489 pozzolana fly ash"),
    # "low cost" alone → masonry/blocks, not hardware
    (r"\blow.?cost\b",                    "IS 269 ordinary portland IS 3466 masonry cement IS 2185 concrete blocks IS 3115 lime"),
    # Humidity / moisture → hydrophobic cement
    (r"\bhumid\b|\bmoisture\b|\bwet\b",   "IS 8043 hydrophobic portland cement moisture resistant"),
    # Coastal / marine / salty / ocean → slag + supersulphated (NOT waterproofing)
    (r"\bcoastal\b|\bmarine\b|\bocean\b|\bsalt",
                                          "IS 455 portland slag IS 6909 supersulphated IS 12330 sulphate resisting"),
    # Block + sea/coastal → inject both block AND cement standards
    (r"\bblocks?\b.*(?:\bsea\b|\bcoastal\b|\bmarine\b|\bdurabilit\b)"
     r"|(?:\bsea\b|\bcoastal\b|\bmarine\b).*\bblocks?\b",
                                          "IS 2185 concrete masonry blocks IS 12440 precast IS 455 IS 6909 sulphate"),
    # RCC / reinforced concrete → high-strength cement + steel
    (r"\brcc\b|\breinforced\s+cement\s+concrete\b|\breinforced\s+concrete\b",
                                          "IS 8112 53 grade ordinary portland cement IS 1786 IS 432 reinforcement"),
]

# False-positive suppression removed � handled by general title re-ranker in retriever.py


def _strip_negative_terms(query: str) -> str:
    """
    Remove terms that have no useful meaning in BIS SP 21 and actively mislead
    retrieval.  Only strips when the term is not part of a meaningful compound.

    Args:
        query: Query string (already lowercased for matching, but original case preserved).

    Returns:
        Query with negative terms removed.
    """
    q = query
    q_lower = q.lower()

    # Strip "housing" ONLY when it appears as a standalone noun meaning residential
    # (i.e. NOT preceded by "door", "lock", "valve", "bearing" — those are legitimate)
    if re.search(r"\bhousing\b", q_lower) and not re.search(
        r"\b(?:door|lock|valve|bearing|pump|gear)\s+housing\b", q_lower
    ):
        q = re.sub(r"\bhousing\b", "construction", q, flags=re.IGNORECASE)
        logger.debug("Replaced 'housing' → 'construction' to avoid door-hardware false positives")

    return q


def expand_query(query: str) -> str:
    """
    Expand a user query by substituting synonyms and injecting domain concepts.

    For queries where user language is far from BIS vocabulary (e.g. "rods" vs
    "bars", "housing" vs "construction"), the original query is rewritten rather
    than just appended to, so BM25 doesn't get anchored on the wrong term.

    Args:
        query: Raw user query string.

    Returns:
        Expanded query string with BIS-vocabulary terms appended/substituted.
    """
    q = query.strip()

    # ── Step 0: strip negative terms that mislead retrieval ───────────────────
    q = _strip_negative_terms(q)
    q_lower = q.lower()
    additions: List[str] = []

    # ── Special case: rewrite queries where user language badly misleads BM25 ──
    # "steel rods" in any structural/construction context → rewrite to bar terminology
    if re.search(r"\bsteel rods?\b", q_lower) and re.search(
        r"\bconstruction\b|\breinforcement\b|\bbuilding\b|\bbridge\b"
        r"|\broad\b|\bindustrial\b|\bstructur\b|\bframe\b|\bfoundation\b"
        r"|\bconcrete\b|\bcolumn\b|\bbeam\b|\bslab\b|\bpile\b",
        q_lower,
    ):
        q = re.sub(r"\bsteel rods?\b", "steel bars reinforcement", q, flags=re.IGNORECASE)
        q_lower = q.lower()
        additions.append("deformed high strength IS 432 IS 1786")

    # "rods" alone (without "steel") in a structural context → also rewrite
    elif re.search(r"\brods?\b", q_lower) and re.search(
        r"\bconstruction\b|\breinforcement\b|\bbuilding\b|\bbridge\b"
        r"|\bconcrete\b|\bstructur\b|\bcolumn\b|\bbeam\b|\bslab\b",
        q_lower,
    ):
        q = re.sub(r"\brods?\b", "bars reinforcement", q, flags=re.IGNORECASE)
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
        # Cap total expansion to ~100 tokens to avoid diluting BM25 / dense retrieval.
        # Deduplicate individual tokens across all addition strings first.
        seen_tokens: set = set()
        unique_tokens: List[str] = []
        for phrase in additions:
            for tok in phrase.split():
                if tok not in seen_tokens:
                    seen_tokens.add(tok)
                    unique_tokens.append(tok)
                if len(unique_tokens) >= 100:
                    break
            if len(unique_tokens) >= 100:
                break
        expansion_str = " ".join(unique_tokens)
        expanded = q + " " + expansion_str
        logger.info(
            f"Query expanded: {len(query)} → {len(expanded)} chars "
            f"({len(unique_tokens)} expansion tokens)"
        )
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
        and re.search(
            r"\bconstruction\b|\breinforcement\b|\bbuilding\b|\bbridge\b"
            r"|\broad\b|\bindustrial\b|\bstructur\b|\bframe\b|\bfoundation\b"
            r"|\bconcrete\b|\bcolumn\b|\bbeam\b|\bslab\b|\bpile\b",
            q_lower,
        )
    ) or bool(
        re.search(r"\brods?\b", q_lower)
        and not re.search(r"\bsteel rods?\b", q_lower)
        and re.search(
            r"\bconstruction\b|\breinforcement\b|\bbuilding\b|\bbridge\b"
            r"|\bconcrete\b|\bstructur\b|\bcolumn\b|\bbeam\b|\bslab\b",
            q_lower,
        )
    ) or bool(
        # "housing" stripped → rewrite applied
        re.search(r"\bhousing\b", q_lower)
        and not re.search(r"\b(?:door|lock|valve|bearing|pump|gear)\s+housing\b", q_lower)
    )

    expanded = expand_query(query)

    if expanded == query:
        return [query]

    if rewrite_applied:
        # Only use the rewritten+expanded version — original would mislead BM25
        return [expanded]

    # Append-only expansion: run both for broader coverage
    return [query, expanded]
