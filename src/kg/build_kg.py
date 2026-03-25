"""
RDF Knowledge Graph builder for the Warring States period.

Input:  data/extracted_knowledge.csv   (NER entities)
        data/relation_triples.csv       (dep + co-occurrence triples)

Output: kg_artifacts/ontology.ttl       (OWL ontology — classes & properties)
        kg_artifacts/warring_states.ttl (populated KG instances)
        kg_artifacts/kb_stats.txt       (statistics)
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ── Namespaces ─────────────────────────────────────────────────────────────────
WS   = Namespace("http://warring-states.kg/ontology#")
WSI  = Namespace("http://warring-states.kg/instance/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# ── I/O paths ─────────────────────────────────────────────────────────────────
ENTITY_FILE  = "data/extracted_knowledge.csv"
TRIPLES_FILE = "data/relation_triples.csv"
ONTO_FILE    = "kg_artifacts/ontology.ttl"
KG_FILE      = "kg_artifacts/warring_states.ttl"
STATS_FILE   = "kg_artifacts/kb_stats.txt"


# ── 1.  Curated entity → class mapping ────────────────────────────────────────
# Hand-coded for the most important entities; the rest are classified
# by their NER label (PERSON → ws:Person, GPE/LOC → ws:Place, etc.)

KNOWN_STATES = {
    "Qin", "Chu", "Wei", "Zhao", "Han", "Yan", "Qi",
    "Jin", "Song", "Yue", "Wu", "Zhou", "Shu", "Zheng",
    "Lu", "Wey", "Cai", "Zeng", "Ba", "Shu",
}

KNOWN_PERSONS = {
    "Shang Yang", "Sun Tzu", "Confucius", "Mencius", "Han Fei", "Li Si",
    "Qin Shi Huang", "Wu Qi", "Sima Qian", "Xunzi", "Laozi", "Zhuangzi",
    "Mozi", "Gongsun Yang", "Lian Po", "Wang Jian", "Li Mu", "Bai Qi",
    "Ying Zheng", "Dong Zhongshu", "Sima Tan", "Jing Ke", "Shen Buhai",
    "Shen Dao", "Hui Shi", "Fan Sui", "Su Qin", "Zhang Yi", "Guan Zhong",
    "Sun Bin", "Pang Juan",
}

KNOWN_BATTLES = {
    "Battle of Changping", "Battle of Yique", "Battle of Maling",
    "Battle of Guiling", "Battle of Cannae",
}

KNOWN_SCHOOLS = {
    "Legalism", "Confucianism", "Taoism", "Mohism",
    "Hundred Schools of Thought", "Daoism",
}

KNOWN_DYNASTIES = {
    "Zhou dynasty", "Qin dynasty", "Han dynasty",
    "Shang dynasty", "Western Zhou", "Eastern Zhou",
}

# ── 2.  Predicate → OWL property mapping ──────────────────────────────────────
PREDICATE_MAP = {
    # military
    "attack":    "attacked",
    "conquer":   "conquered",
    "defeat":    "defeated",
    "invade":    "invaded",
    "capture":   "captured",
    "unify":     "unified",
    "unite":     "unified",
    "overthrow": "overthrew",
    "annex":     "annexed",
    "depose":    "deposed",
    "extinguish":"annexed",
    "force":     "pressured",
    "lure":      "luredInto",
    # political
    "rule":      "ruled",
    "enfeoff":   "enfeoffed",
    "appoint":   "appointed",
    "replace":   "replaced",
    "instal":    "installed",
    "declare":   "declared",
    # intellectual
    "criticize": "criticized",
    "criticise": "criticized",
    "study":     "studied",
    "reform":    "reformed",
    "include":   "included",
    "blame":     "blamed",
    "recall":    "referenced",
    "describe":  "described",
    "characterize":"characterized",
    "compare":   "compared",
    "argue":     "argued",
    "present":   "referenced",
    "name":      "referenced",
    "list":      "referenced",
    "warn":      "warned",
    "associate": "associatedWith",
    "resemble":  "resembles",
    "define":    "defined",
    "mention":   "mentioned",
    "claim":     "claimed",
    "depict":    "depicted",
    "understate":"referenced",
    "attest":    "attested",
    "praise":    "praised",
    "identify":  "identified",
    # service
    "serve":     "servedIn",
    "send":      "sent",
    "defend":    "defended",
    "meet":      "met",
    "approach":  "approached",
    "summon":    "summoned",
    "invite":    "invited",
    "offer":     "offered",
    "convince":  "convinced",
    # generic
    "co_occurs_with": "coOccursWith",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def uri_safe(name: str) -> str:
    """Turn a label into a URI-safe string."""
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def instance_uri(name: str) -> URIRef:
    return WSI[uri_safe(name)]


def classify_entity(name: str, ner_label: str) -> URIRef | None:
    """Return the OWL class for an entity."""
    if name in KNOWN_STATES:    return WS.State
    if name in KNOWN_PERSONS:   return WS.Person
    if name in KNOWN_BATTLES:   return WS.Battle
    if name in KNOWN_SCHOOLS:   return WS.PhilosophicalSchool
    if name in KNOWN_DYNASTIES: return WS.Dynasty
    # Fall back on NER label
    mapping = {
        "PERSON": WS.Person,
        "GPE":    WS.Place,
        "LOC":    WS.Place,
        "ORG":    WS.Organization,
        "NORP":   WS.CulturalGroup,
        "EVENT":  WS.Event,
    }
    return mapping.get(ner_label)


# ── 3. Build ontology graph ────────────────────────────────────────────────────

def build_ontology() -> Graph:
    g = Graph()
    g.bind("ws",  WS)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs",RDFS)

    onto = WSI["ontology"]
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal("Warring States China Knowledge Graph Ontology")))

    # ── Classes
    classes = {
        WS.State:              "A Chinese state during the Warring States period",
        WS.Person:             "A historical person (ruler, general, philosopher, etc.)",
        WS.Battle:             "A military engagement",
        WS.PhilosophicalSchool:"A school of thought (Confucianism, Legalism, etc.)",
        WS.Dynasty:            "A Chinese dynasty",
        WS.Place:              "A geographical place",
        WS.Organization:       "An organisation (excluding states)",
        WS.CulturalGroup:      "An ethnic or cultural group",
        WS.Event:              "A historical event",
    }
    for cls, comment in classes.items():
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(cls.split("#")[-1])))
        g.add((cls, RDFS.comment, Literal(comment)))

    # ── Object Properties
    obj_props = {
        # military
        WS.attacked:      ("State",  "State",  "attacked"),
        WS.conquered:     ("State",  "State",  "conquered"),
        WS.defeated:      ("State",  "State",  "defeated"),
        WS.invaded:       ("State",  "State",  "invaded"),
        WS.captured:      ("State",  "Place",  "captured"),
        WS.unified:       ("Person", "Place",  "unified"),
        WS.overthrew:     ("State",  "State",  "overthrew"),
        WS.annexed:       ("State",  "State",  "annexed"),
        WS.pressured:     ("State",  "State",  "pressured"),
        WS.luredInto:     ("State",  "State",  "lured into"),
        # political
        WS.ruled:         ("Person", "State",  "ruled"),
        WS.enfeoffed:     ("Person", "Person", "enfeoffed"),
        WS.appointed:     ("Person", "Person", "appointed"),
        WS.replaced:      ("State",  "Person", "replaced"),
        WS.installed:     ("Person", "Person", "installed"),
        WS.deposed:       ("Person", "Person", "deposed"),
        WS.servedIn:      ("Person", "State",  "served in"),
        WS.sent:          ("Person", "Person", "sent"),
        WS.defended:      ("Person", "Place",  "defended"),
        WS.met:           ("Person", "Person", "met"),
        WS.summoned:      ("Person", "Person", "summoned"),
        WS.invited:       ("Person", "Person", "invited"),
        WS.convinced:     ("Person", "Person", "convinced"),
        WS.warned:        ("Person", "Person", "warned"),
        WS.reformed:      ("Person", "State",  "reformed"),
        # intellectual
        WS.criticized:    ("Person", "Person", "criticized"),
        WS.studied:       ("Person", "Person", "studied"),
        WS.referenced:    ("Person", "Person", "referenced"),
        WS.described:     ("Person", "Person", "described"),
        WS.compared:      ("Person", "Person", "compared"),
        WS.associatedWith:("Person", "Person", "associated with"),
        WS.resembles:     ("Person", "Person", "resembles"),
        WS.praised:       ("Person", "Person", "praised"),
        WS.influenced:    ("Person", "Person", "influenced"),
        # generic
        WS.coOccursWith:  ("Person", "Person", "co-occurs with"),
    }
    for prop, (dom, rng, label) in obj_props.items():
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.label, Literal(label)))

    return g


# ── 4. Build instance graph ────────────────────────────────────────────────────

def build_instance_graph() -> Graph:
    g = Graph()
    g.bind("ws",  WS)
    g.bind("wsi", WSI)
    g.bind("rdf", RDF)
    g.bind("rdfs",RDFS)
    g.bind("prov",PROV)

    # Track all added entities
    added: dict[str, URIRef] = {}   # name → URI

    def ensure_entity(name: str, ner_label: str = ""):
        if name in added:
            return added[name]
        uri = instance_uri(name)
        cls = classify_entity(name, ner_label)
        if cls:
            g.add((uri, RDF.type, cls))
        g.add((uri, RDFS.label, Literal(name)))
        added[name] = uri
        return uri

    # ── Load & add entities from NER
    with open(ENTITY_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row["entity"].strip()
            label = row["label"]
            src   = row["source_url"]
            if not name or len(name) < 2:
                continue
            uri = ensure_entity(name, label)
            g.add((uri, PROV.wasDerivedFrom, URIRef(src)))

    # ── Load & convert triples
    skipped = 0
    added_triples = 0
    with open(TRIPLES_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj_name = row["subject"].strip()
            pred_raw  = row["predicate"].strip()
            obj_name  = row["object"].strip()
            src       = row["source_url"]

            # Skip very noisy objects
            if len(obj_name) < 2 or len(obj_name) > 80:
                skipped += 1
                continue
            # Skip pure-number objects
            if re.fullmatch(r"[\d\s\-–,\.BCbc]+", obj_name):
                skipped += 1
                continue
            # Skip publisher/academic ORGs that sneak in as objects
            noise_orgs = {
                "Cambridge University Press", "Oxford University Press",
                "Routledge", "Brill", "Springer", "Princeton University Press",
                "Columbia University Press", "Penguin", "SUNY Press",
                "Project Gutenberg", "the Wayback Machine",
            }
            if subj_name in noise_orgs or obj_name in noise_orgs:
                skipped += 1
                continue

            ws_pred = PREDICATE_MAP.get(pred_raw)
            if ws_pred is None:
                skipped += 1
                continue

            s_uri = ensure_entity(subj_name)
            o_uri = ensure_entity(obj_name)
            prop  = WS[ws_pred]

            g.add((s_uri, prop, o_uri))
            g.add((s_uri, PROV.wasDerivedFrom, URIRef(src)))
            added_triples += 1

    print(f"  Triples added : {added_triples}")
    print(f"  Triples skipped: {skipped}")
    print(f"  Unique entities: {len(added)}")
    return g, added


# ── 5. Main ────────────────────────────────────────────────────────────────────

def run():
    Path("kg_artifacts").mkdir(exist_ok=True)

    print("Building ontology...")
    onto_g = build_ontology()
    onto_g.serialize(destination=ONTO_FILE, format="turtle")
    print(f"  Ontology saved -> {ONTO_FILE}  ({len(onto_g)} triples)")

    print("\nBuilding instance graph...")
    inst_g, entities = build_instance_graph()
    inst_g.serialize(destination=KG_FILE, format="turtle")
    print(f"  KG saved       -> {KG_FILE}  ({len(inst_g)} triples)")

    # ── Stats
    from collections import Counter
    type_counts = Counter()
    for _, _, o in inst_g.triples((None, RDF.type, None)):
        type_counts[str(o).split("#")[-1]] += 1

    pred_counts = Counter()
    for s, p, o in inst_g:
        if p != RDF.type and p != RDFS.label and "prov" not in str(p):
            pred_counts[str(p).split("#")[-1]] += 1

    stats = []
    stats.append(f"Total RDF triples : {len(inst_g)}")
    stats.append(f"Unique entities   : {len(entities)}")
    stats.append("")
    stats.append("Instance type breakdown:")
    for t, c in type_counts.most_common():
        stats.append(f"  {t:25s} {c}")
    stats.append("")
    stats.append("Top predicates (excluding rdf:type/rdfs:label/prov):")
    for p, c in pred_counts.most_common(15):
        stats.append(f"  {p:25s} {c}")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(stats))

    print()
    for line in stats:
        print(" ", line)


if __name__ == "__main__":
    run()
