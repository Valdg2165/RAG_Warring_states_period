"""
SPARQL Expansion for the Warring States KG.

What is SPARQL expansion?
--------------------------
We already have a local RDF graph. "Expansion" means we use SPARQL queries —
either against our OWN graph (rule-based inference) or against WIKIDATA — to
discover new triples that logically follow from what we already know, and add
them back in. Two strategies:

  A) LOCAL INFERENCE RULES (SPARQL on our own graph)
     These are cheap and deterministic. Examples:
       - Inverse:    A conquered B  →  B wasConqueredBy A
       - Inverse:    A studentOf B  →  B hasStudent A
       - Symmetric:  A coOccursWith B  →  B coOccursWith A (already implicit)
       - Transitive: A studentOf B, B studentOf C  →  A intellectualDescendantOf C
       - Chain:      A servedIn S, S attacked T  →  A foughtFor S against T

  B) WIKIDATA 1-HOP EXPANSION (SPARQL on external endpoint)
     For each entity already in our KG, fetch their immediate Wikidata
     neighbours that we haven't pulled yet: colleagues, employers, works,
     battles, movements, etc.

Goal: push total triples from ~28k toward 50k–100k for KGE training.

Input:  kg_artifacts/warring_states_expanded.ttl
Output: kg_artifacts/warring_states_final.ttl
        kg_artifacts/kb_stats_final.txt
"""

import re
import time
import requests
from collections import Counter
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


def _parse_year(date_str: str) -> str:
    """Extract year from Wikidata ISO 8601 date string.
    BC dates: '-YYYY-MM-DD...' → take [:5] to get '-YYYY' (not [:4] which loses last digit).
    AD dates: 'YYYY-MM-DD...' → take [:4].
    """
    if not date_str:
        return ""
    if date_str.startswith("-"):
        return date_str[:5]
    return date_str[:4]

# ── Namespaces ─────────────────────────────────────────────────────────────────
WS   = Namespace("http://warring-states.kg/ontology#")
WSI  = Namespace("http://warring-states.kg/instance/")
PROV = Namespace("http://www.w3.org/ns/prov#")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "WarringStatesKG/1.0 (educational project)"}

INPUT_FILE  = "kg_artifacts/warring_states_expanded.ttl"
OUTPUT_FILE = "kg_artifacts/warring_states_final.ttl"
STATS_FILE  = "kg_artifacts/kb_stats_final.txt"

# ── Inverse property definitions ───────────────────────────────────────────────
# (forward_property, inverse_property, inverse_label)
INVERSE_PAIRS = [
    (WS.conquered,      WS.wasConqueredBy,      "was conquered by"),
    (WS.attacked,       WS.wasAttackedBy,        "was attacked by"),
    (WS.defeated,       WS.wasDefeatedBy,        "was defeated by"),
    (WS.invaded,        WS.wasInvadedBy,         "was invaded by"),
    (WS.annexed,        WS.wasAnnexedBy,         "was annexed by"),
    (WS.captured,       WS.wasCapturedBy,        "was captured by"),
    (WS.studentOf,      WS.hasStudent,           "has student"),
    (WS.influencedBy,   WS.influenced,           "influenced"),
    (WS.authored,       WS.authoredBy,           "authored by"),
    (WS.hasCapital,     WS.isCapitalOf,          "is capital of"),
    (WS.bornIn,         WS.birthplaceOf,         "birthplace of"),
    (WS.diedIn,         WS.deathplaceOf,         "deathplace of"),
    (WS.hasParticipant, WS.participatedIn,       "participated in"),
    (WS.wonBy,          WS.won,                  "won"),
    (WS.servedIn,       WS.hadServant,           "had servant"),
    (WS.headOfState,    WS.wasHeadOfStateOf,     "was head of state of"),
    (WS.follows,        WS.followedBy,           "followed by"),
    (WS.criticized,     WS.wasCriticizedBy,      "was criticized by"),
    (WS.ruled,          WS.wasRuledBy,           "was ruled by"),
    (WS.reformed,       WS.wasReformedBy,        "was reformed by"),
    (WS.sent,           WS.wasSentBy,            "was sent by"),
    (WS.memberOf,       WS.hasMember,            "has member"),
    (WS.partOf,         WS.hasPart,              "has part"),
    (WS.movement,       WS.hasAdherent,          "has adherent"),
    (WS.replaces,       WS.replacedBy,           "replaced by"),
    (WS.unified,        WS.wasUnifiedBy,         "was unified by"),
    (WS.overthrew,      WS.wasOverthrownBy,      "was overthrown by"),
    (WS.pressured,      WS.wasPressuredBy,       "was pressured by"),
    (WS.convinced,      WS.wasConvincedBy,       "was convinced by"),
    (WS.defended,       WS.wasDefendedBy,        "was defended by"),
    (WS.appointed,      WS.wasAppointedBy,       "was appointed by"),
    (WS.installed,      WS.wasInstalledBy,       "was installed by"),
    (WS.enfeoffed,      WS.wasEnfeoffedBy,       "was enfeoffed by"),
    (WS.warned,         WS.wasWarnedBy,          "was warned by"),
    (WS.heldPosition,   WS.positionHeldBy,       "position held by"),
]

# ── Transitivity / chain rules (SPARQL on local graph) ────────────────────────
# Each rule is a (rule_name, sparql_template, new_predicate)
# The SPARQL SELECT must return ?s and ?o
CHAIN_RULES = [
    # intellectual lineage: if A studentOf B and B studentOf C → A intellectualDescendantOf C
    (
        "intellectual_lineage",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:studentOf ?mid .
          ?mid ws:studentOf ?o .
          FILTER(?s != ?o && ?s != ?mid)
        }
        """,
        WS.intellectualDescendantOf,
        "intellectual descendant of",
    ),
    # if A influenced B and B influenced C → A remotelyInfluenced C
    (
        "transitive_influence",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:influenced ?mid .
          ?mid ws:influenced ?o .
          FILTER(?s != ?o && ?s != ?mid)
        }
        """,
        WS.remotelyInfluenced,
        "remotely influenced",
    ),
    # if Person servedIn State and State attacked State2 → Person foughtAgainst State2
    (
        "person_fought_against",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:servedIn ?state .
          ?state ws:attacked ?o .
          FILTER(?s != ?o)
        }
        """,
        WS.foughtAgainst,
        "fought against",
    ),
    # if Person servedIn State and State conquered State2 → Person conqueredFor State2
    (
        "person_conquered_for",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:servedIn ?state .
          ?state ws:conquered ?o .
          FILTER(?s != ?o)
        }
        """,
        WS.participatedInConquestOf,
        "participated in conquest of",
    ),
    # if Person A and B both servedIn same State → they were colleagues
    (
        "colleagues",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:servedIn ?state .
          ?o ws:servedIn ?state .
          ?s a ws:Person .
          ?o a ws:Person .
          FILTER(?s != ?o && STR(?s) < STR(?o))
        }
        """,
        WS.colleague,
        "colleague of",
    ),
    # if A studentOf B and B memberOf School → A also affiliated with that School
    (
        "school_affiliation",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?s ws:studentOf ?teacher .
          ?teacher ws:memberOf ?o .
          FILTER(?s != ?o)
        }
        """,
        WS.affiliatedWith,
        "affiliated with",
    ),
    # if Battle hasParticipant State and State headOfState Person → Person foughtIn Battle
    (
        "leader_fought_in",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?o ws:hasParticipant ?state .
          ?state ws:headOfState ?s .
          ?o a ws:Battle .
        }
        """,
        WS.ledIn,
        "led forces in",
    ),
    # coOccursWith is symmetric — generate reverse direction
    (
        "cooccurs_symmetric",
        """
        SELECT DISTINCT ?s ?o WHERE {
          ?o ws:coOccursWith ?s .
          FILTER NOT EXISTS { ?s ws:coOccursWith ?o }
          FILTER(?s != ?o)
        }
        """,
        WS.coOccursWith,
        "co-occurs with",
    ),
]


# ── Wikidata 1-hop expansion ───────────────────────────────────────────────────

def sparql_wd(query: str) -> list[dict]:
    try:
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    except Exception as e:
        print(f"    [WD error] {e}")
        return []


def uri_safe(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def ensure(g: Graph, label: str, cls=None) -> URIRef:
    uri = WSI[uri_safe(label)]
    g.add((uri, RDFS.label, Literal(label)))
    if cls:
        g.add((uri, RDF.type, cls))
    return uri


PROV_WD = URIRef("https://www.wikidata.org/")


def wd_add(g: Graph, s_lbl: str, pred: URIRef, o_lbl: str,
           s_cls=None, o_cls=None) -> bool:
    s = ensure(g, s_lbl, s_cls)
    o = ensure(g, o_lbl, o_cls)
    t = (s, pred, o)
    if t not in g:
        g.add(t)
        g.add((s, PROV.wasDerivedFrom, PROV_WD))
        return True
    return False


def wikidata_expand_philosophers(g: Graph) -> int:
    """
    For each philosopher born 600–200 BC, fetch:
    students, teachers, influenced, movement, employer.
    """
    n = 0
    query = """
    SELECT DISTINCT ?p ?pLabel ?rel ?relLabel ?target ?targetLabel WHERE {
      ?p wdt:P31 wd:Q5 .
      ?p wdt:P569 ?birth .
      FILTER(?birth > "-620-01-01"^^xsd:dateTime &&
             ?birth < "-180-01-01"^^xsd:dateTime)
      VALUES ?rel {
        wdt:P1066   # student of
        wdt:P802    # student
        wdt:P737    # influenced by
        wdt:P135    # movement
        wdt:P106    # occupation
        wdt:P108    # employer
        wdt:P27     # country of citizenship
      }
      ?p ?rel ?target .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      BIND(STRAFTER(STR(?rel), "http://www.wikidata.org/prop/direct/") AS ?relLabel)
    }
    LIMIT 500
    """
    rows = sparql_wd(query)
    WD_PROP_MAP = {
        "P1066": WS.studentOf,
        "P802":  WS.hasStudent,
        "P737":  WS.influencedBy,
        "P135":  WS.movement,
        "P106":  WS.heldPosition,
        "P108":  WS.servedIn,
        "P27":   WS.nationOf,
    }
    for row in rows:
        p_lbl   = row.get("pLabel",      {}).get("value", "")
        rel_id  = row.get("relLabel",    {}).get("value", "")
        t_lbl   = row.get("targetLabel", {}).get("value", "")
        if not p_lbl or not t_lbl or p_lbl == t_lbl:
            continue
        pred = WD_PROP_MAP.get(rel_id)
        if pred:
            n += wd_add(g, p_lbl, pred, t_lbl, s_cls=WS.Person)
    print(f"  philosophers: +{n}")
    return n


def wikidata_expand_battles(g: Graph) -> int:
    """Fetch all battles with Chinese participants, 600–200 BC."""
    n = 0
    query = """
    SELECT DISTINCT ?battle ?battleLabel ?participant ?participantLabel
                    ?winner ?winnerLabel ?date ?location ?locationLabel WHERE {
      ?battle wdt:P31/wdt:P279* wd:Q178561 .
      ?battle wdt:P585 ?date .
      FILTER(?date > "-700-01-01"^^xsd:dateTime &&
             ?date < "-200-01-01"^^xsd:dateTime)
      OPTIONAL { ?battle wdt:P710 ?participant . }
      OPTIONAL { ?battle wdt:P1111 ?winner . }
      OPTIONAL { ?battle wdt:P276 ?location . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 300
    """
    rows = sparql_wd(query)
    for row in rows:
        b   = row.get("battleLabel",      {}).get("value", "")
        p   = row.get("participantLabel", {}).get("value", "")
        w   = row.get("winnerLabel",      {}).get("value", "")
        loc = row.get("locationLabel",    {}).get("value", "")
        dt  = _parse_year(row.get("date", {}).get("value", ""))
        if b and p:
            n += wd_add(g, b, WS.hasParticipant, p, s_cls=WS.Battle)
        if b and w:
            n += wd_add(g, b, WS.wonBy, w, s_cls=WS.Battle)
        if b and loc:
            n += wd_add(g, b, WS.locatedIn, loc, s_cls=WS.Battle, o_cls=WS.Place)
        if b and dt and dt.lstrip("-").isdigit():
            uri = ensure(g, b, WS.Battle)
            lit = Literal(dt, datatype=XSD.string)
            if (uri, WS.year, lit) not in g:
                g.add((uri, WS.year, lit))
                n += 1
    print(f"  battles: +{n}")
    return n


def wikidata_expand_chinese_states(g: Graph) -> int:
    """For known Chinese states, fetch rulers, battles, capitals, successors."""
    n = 0
    # States during Spring & Autumn / Warring States ~ 770–221 BC
    query = """
    SELECT DISTINCT ?state ?stateLabel ?rel ?relLabel ?val ?valLabel WHERE {
      ?state wdt:P31/wdt:P279* wd:Q3624078 .
      ?state wdt:P361 wd:Q185063 .
      VALUES ?rel {
        wdt:P6   wdt:P35  wdt:P36  wdt:P17
        wdt:P571 wdt:P576 wdt:P1365 wdt:P1366
        wdt:P710 wdt:P607
      }
      ?state ?rel ?val .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      BIND(STRAFTER(STR(?rel),"http://www.wikidata.org/prop/direct/") AS ?relLabel)
    }
    LIMIT 400
    """
    WD_STATE_MAP = {
        "P6":    WS.headOfGovernment,
        "P35":   WS.headOfState,
        "P36":   WS.hasCapital,
        "P17":   WS.locatedIn,
        "P1365": WS.replaces,
        "P1366": WS.replacedBy,
        "P710":  WS.hasParticipant,
        "P607":  WS.foughtIn,
    }
    rows = sparql_wd(query)
    for row in rows:
        s_lbl  = row.get("stateLabel", {}).get("value", "")
        rel_id = row.get("relLabel",   {}).get("value", "")
        v_lbl  = row.get("valLabel",   {}).get("value", "")
        if not s_lbl or not v_lbl or s_lbl == v_lbl:
            continue
        pred = WD_STATE_MAP.get(rel_id)
        if pred:
            n += wd_add(g, s_lbl, pred, v_lbl, s_cls=WS.State)
    print(f"  states: +{n}")
    return n


def wikidata_expand_works(g: Graph) -> int:
    """Fetch all texts / books written during or about the Warring States period."""
    n = 0
    query = """
    SELECT DISTINCT ?work ?workLabel ?author ?authorLabel ?genre ?genreLabel WHERE {
      { ?work wdt:P571 ?date .
        FILTER(?date > "-600-01-01"^^xsd:dateTime && ?date < "-200-01-01"^^xsd:dateTime) }
      UNION
      { ?work wdt:P921/wdt:P361 wd:Q185063 . }
      ?work wdt:P31/wdt:P279* wd:Q571 .
      OPTIONAL { ?work wdt:P50 ?author . }
      OPTIONAL { ?work wdt:P136 ?genre . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 200
    """
    rows = sparql_wd(query)
    for row in rows:
        w_lbl = row.get("workLabel",   {}).get("value", "")
        a_lbl = row.get("authorLabel", {}).get("value", "")
        g_lbl = row.get("genreLabel",  {}).get("value", "")
        if w_lbl and a_lbl:
            n += wd_add(g, a_lbl, WS.authored, w_lbl, s_cls=WS.Person)
        if w_lbl and g_lbl:
            n += wd_add(g, w_lbl, WS.hasGenre, g_lbl)
    print(f"  literary works: +{n}")
    return n


# ── Part A: local SPARQL inference ────────────────────────────────────────────

def apply_inverse_rules(g: Graph) -> int:
    """Generate all inverse-property triples from INVERSE_PAIRS."""
    # First register all inverse properties in the ontology
    for fwd, inv, label in INVERSE_PAIRS:
        g.add((inv, RDF.type, OWL.ObjectProperty))
        g.add((inv, RDFS.label, Literal(label)))
        g.add((fwd, OWL.inverseOf, inv))

    n = 0
    PROV_INFER = URIRef("http://warring-states.kg/inference#inverseRule")
    for fwd, inv, _ in INVERSE_PAIRS:
        for s, _, o in g.triples((None, fwd, None)):
            if not isinstance(o, URIRef):   # skip literals
                continue
            t = (o, inv, s)
            if t not in g:
                g.add(t)
                g.add((o, PROV.wasDerivedFrom, PROV_INFER))
                n += 1
    print(f"  inverse triples: +{n}")
    return n


def apply_chain_rules(g: Graph) -> int:
    """Run each SPARQL chain rule against our local graph."""
    n = 0
    PROV_CHAIN = URIRef("http://warring-states.kg/inference#chainRule")
    for rule_name, sparql_body, new_pred, label in CHAIN_RULES:
        g.add((new_pred, RDF.type, OWL.ObjectProperty))
        g.add((new_pred, RDFS.label, Literal(label)))
        # Inject our prefix into the query
        query = (
            "PREFIX ws:  <http://warring-states.kg/ontology#>\n"
            "PREFIX wsi: <http://warring-states.kg/instance/>\n"
            + sparql_body
        )
        before = n
        for row in g.query(query):
            s, o = row.s, row.o
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue
            t = (s, new_pred, o)
            if t not in g:
                g.add(t)
                g.add((s, PROV.wasDerivedFrom, PROV_CHAIN))
                n += 1
        added = n - before
        if added:
            print(f"    {rule_name:35s} +{added}")
    print(f"  chain rule triples: +{n}")
    return n


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    g = Graph()
    g.parse(INPUT_FILE, format="turtle")
    g.bind("ws", WS); g.bind("wsi", WSI); g.bind("prov", PROV)
    before = len(g)
    print(f"Loaded KG: {before:,} triples\n")

    total_new = 0

    # ── Part A: Local inference ───────────────────────────────────────────────
    print("=" * 55)
    print("PART A — Local SPARQL Inference Rules")
    print("=" * 55)

    print("\n[A1] Generating inverse property triples...")
    total_new += apply_inverse_rules(g)

    print("\n[A2] Applying chain / composition rules...")
    total_new += apply_chain_rules(g)

    print(f"\nAfter local inference: {len(g):,} triples (+{len(g)-before:,})")

    # ── Part B: Wikidata 1-hop expansion ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("PART B — Wikidata 1-Hop Expansion")
    print("=" * 55)

    wikidata_n = 0
    print("\n[B1] Expanding philosophers (600–180 BC)...")
    wikidata_n += wikidata_expand_philosophers(g)
    time.sleep(1)

    print("\n[B2] Expanding battles (700–200 BC)...")
    wikidata_n += wikidata_expand_battles(g)
    time.sleep(1)

    print("\n[B3] Expanding Chinese states (Warring States period)...")
    wikidata_n += wikidata_expand_chinese_states(g)
    time.sleep(1)

    print("\n[B4] Expanding literary works of the period...")
    wikidata_n += wikidata_expand_works(g)

    print(f"\nWikidata expansion added: +{wikidata_n}")

    # ── Final stats ───────────────────────────────────────────────────────────
    after = len(g)
    print("\n" + "=" * 55)
    print("FINAL STATS")
    print("=" * 55)
    print(f"  Before expansion : {before:,}")
    print(f"  After expansion  : {after:,}")
    print(f"  Total added      : {after - before:,}")

    type_counts = Counter(
        str(o).split("#")[-1]
        for _, p, o in g.triples((None, RDF.type, None))
        if "ontology" in str(o)
    )
    print("\nInstance types:")
    for t, c in type_counts.most_common():
        print(f"  {t:35s} {c}")

    skip = {"wasDerivedFrom", "label", "type", "ObjectProperty",
            "inverseOf", "Class", "Ontology"}
    pred_counts = Counter(
        str(p).split("#")[-1] for _, p, _ in g
        if str(p).split("#")[-1] not in skip
    )
    print("\nTop predicates:")
    for p, c in pred_counts.most_common(20):
        print(f"  {p:35s} {c:,}")

    # Save
    g.serialize(destination=OUTPUT_FILE, format="turtle")
    print(f"\nFinal KG saved -> {OUTPUT_FILE}")

    # Update stats file
    lines = [
        f"=== Warring States KG — Final Statistics ===",
        f"",
        f"Total triples      : {after:,}",
        f"Unique subjects    : {len(set(s for s, _, _ in g)):,}",
        f"Added by expansion : {after - before:,}",
        f"  - Inverse rules  : (via INVERSE_PAIRS)",
        f"  - Chain rules    : (SPARQL on local graph)",
        f"  - Wikidata 1-hop : {wikidata_n}",
        f"",
        f"Instance types:",
    ] + [f"  {t:35s} {c}" for t, c in type_counts.most_common()] + [
        f"",
        f"Top predicates:",
    ] + [f"  {p:35s} {c:,}" for p, c in pred_counts.most_common(20)]

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Stats saved -> {STATS_FILE}")


if __name__ == "__main__":
    run()
