"""
SWRL Reasoning on the Warring States Knowledge Graph
=====================================================

We build a proper OWL ontology in-memory with OWLReady2,
populate it with key Warring States individuals, define
4 SWRL Horn rules, run HermiT/Pellet, and report what
the reasoner infers.

4 Rules
-------
Rule 1 (intellectualLineage):
  Person(?p) ^ studentOf(?p, ?t) ^ studentOf(?t, ?gm)
  -> intellectualDescendantOf(?p, ?gm)

Rule 2 (militaryService):
  Person(?p) ^ servedIn(?p, ?s) ^ attacked(?s, ?e)
  -> foughtAgainst(?p, ?e)

Rule 3 (schoolAffiliation):
  Person(?p) ^ studentOf(?p, ?t) ^ movement(?t, ?school)
  -> affiliatedWith(?p, ?school)

Rule 4 (elderScholar):
  Person(?p) ^ birthYear(?p, ?by) ^ deathYear(?p, ?dy)
  ^ swrlb:subtract(?age, ?dy, ?by) ^ swrlb:greaterThan(?age, 70)
  -> ElderScholar(?p)
"""

import sys
import io

# Fix Windows cp1252 console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from owlready2 import (
    get_ontology, sync_reasoner, sync_reasoner_pellet,
    Imp, Thing, ObjectProperty, DataProperty, FunctionalProperty,
)
from owlready2.rule import (
    BuiltinAtom, ClassAtom, DatavaluedPropertyAtom,
)

# ── 1. Build Warring States OWL ontology in-memory ────────────────────────────

onto = get_ontology("http://warring-states.kg/swrl#")

with onto:

    # --- Classes ---
    class Person(Thing):             pass
    class State(Thing):              pass
    class PhilosophicalSchool(Thing): pass
    class ElderScholar(Person):      pass   # inferred by Rule 4

    # --- Object properties ---
    class studentOf(ObjectProperty):
        domain, range = [Person], [Person]

    class intellectualDescendantOf(ObjectProperty):
        domain, range = [Person], [Person]

    class servedIn(ObjectProperty):
        domain, range = [Person], [State]

    class attacked(ObjectProperty):
        domain, range = [State], [State]

    class foughtAgainst(ObjectProperty):
        domain, range = [Person], [State]

    class movement(ObjectProperty):
        domain, range = [Person], [PhilosophicalSchool]

    class affiliatedWith(ObjectProperty):
        domain, range = [Person], [PhilosophicalSchool]

    # --- Data properties ---
    class birthYear(DataProperty, FunctionalProperty):
        domain, range = [Person], [int]

    class deathYear(DataProperty, FunctionalProperty):
        domain, range = [Person], [int]


# ── 2. Populate with Warring States individuals ────────────────────────────────
# Data sourced from our KG (warring_states_expanded.ttl + Wikidata enrichment)

with onto:

    # Philosophical schools
    Confucianism    = PhilosophicalSchool("Confucianism")
    Legalism        = PhilosophicalSchool("Legalism")
    Taoism          = PhilosophicalSchool("Taoism")
    Mohism          = PhilosophicalSchool("Mohism")

    # States (with attack relationships from our KG)
    Qin  = State("Qin");   Zhao = State("Zhao"); Wei  = State("Wei")
    Han  = State("Han");   Chu  = State("Chu");  Qi   = State("Qi")
    Yan  = State("Yan");   Lu   = State("Lu")

    Qin.attacked  = [Wei, Zhao, Han, Chu, Qi, Yan]
    Wei.attacked  = [Han]
    Chu.attacked  = [Wei]

    # Persons — birthYear/deathYear use negative integers for BC
    # (age = deathYear - birthYear gives correct positive span)

    Confucius = Person("Confucius")
    Confucius.birthYear = -551; Confucius.deathYear = -479
    Confucius.movement  = [Confucianism]

    Zisi = Person("Zisi")           # grandson of Confucius
    Zisi.birthYear = -481; Zisi.deathYear = -402
    Zisi.studentOf = [Confucius]

    Mencius = Person("Mencius")
    Mencius.birthYear = -372; Mencius.deathYear = -289
    Mencius.studentOf = [Zisi]

    Xunzi = Person("Xunzi")
    Xunzi.birthYear = -310; Xunzi.deathYear = -235
    Xunzi.studentOf = [Mencius]
    Xunzi.movement  = [Confucianism]

    HanFei = Person("HanFei")
    HanFei.birthYear = -280; HanFei.deathYear = -233
    HanFei.studentOf = [Xunzi]

    LiSi = Person("LiSi")
    LiSi.birthYear = -280; LiSi.deathYear = -208
    LiSi.studentOf = [Xunzi]
    LiSi.servedIn  = [Qin]

    Laozi = Person("Laozi")
    Laozi.movement = [Taoism]

    Zhuangzi = Person("Zhuangzi")
    Zhuangzi.birthYear = -369; Zhuangzi.deathYear = -286
    Zhuangzi.studentOf = [Laozi]

    Mozi = Person("Mozi")
    Mozi.birthYear = -470; Mozi.deathYear = -391
    Mozi.movement  = [Mohism]

    BaiQi = Person("BaiQi")
    BaiQi.birthYear = -332; BaiQi.deathYear = -257
    BaiQi.servedIn  = [Qin]

    WangJian = Person("WangJian")
    WangJian.birthYear = -290; WangJian.deathYear = -228
    WangJian.servedIn  = [Qin]

    WuQi = Person("WuQi")
    WuQi.birthYear = -440; WuQi.deathYear = -381
    WuQi.servedIn  = [Wei]

    SunBin = Person("SunBin")
    SunBin.birthYear = -380; SunBin.deathYear = -316
    SunBin.servedIn  = [Qi]

    GuanZhong = Person("GuanZhong")
    GuanZhong.birthYear = -720; GuanZhong.deathYear = -645


# ── 3. Define SWRL Rules ───────────────────────────────────────────────────────

print("=" * 60)
print("Warring States SWRL Rules (OWLReady2)")
print("=" * 60)

with onto:

    # --- Rule 1: intellectual lineage (studentOf chain) ---
    # Person(?p) ^ studentOf(?p,?t) ^ studentOf(?t,?gm)
    #   -> intellectualDescendantOf(?p, ?gm)
    r1 = Imp()
    r1.label = ["intellectualLineage"]
    r1.set_as_rule(
        "Person(?p), studentOf(?p, ?t), studentOf(?t, ?gm)"
        " -> intellectualDescendantOf(?p, ?gm)"
    )
    print("\nRule 1 defined: Person(?p) ^ studentOf(?p,?t) ^ studentOf(?t,?gm)")
    print("             -> intellectualDescendantOf(?p, ?gm)")

    # --- Rule 2: military service -> foughtAgainst ---
    # Person(?p) ^ servedIn(?p,?s) ^ attacked(?s,?e)
    #   -> foughtAgainst(?p, ?e)
    r2 = Imp()
    r2.label = ["militaryService"]
    r2.set_as_rule(
        "Person(?p), servedIn(?p, ?s), attacked(?s, ?e)"
        " -> foughtAgainst(?p, ?e)"
    )
    print("\nRule 2 defined: Person(?p) ^ servedIn(?p,?s) ^ attacked(?s,?e)")
    print("             -> foughtAgainst(?p, ?e)")

    # --- Rule 3: school affiliation via teacher's movement ---
    # Person(?p) ^ studentOf(?p,?t) ^ movement(?t,?school)
    #   -> affiliatedWith(?p, ?school)
    r3 = Imp()
    r3.label = ["schoolAffiliation"]
    r3.set_as_rule(
        "Person(?p), studentOf(?p, ?t), movement(?t, ?school)"
        " -> affiliatedWith(?p, ?school)"
    )
    print("\nRule 3 defined: Person(?p) ^ studentOf(?p,?t) ^ movement(?t,?school)")
    print("             -> affiliatedWith(?p, ?school)")

    # --- Rule 4: ElderScholar (lived > 70 years) ---
    # Person(?p) ^ birthYear(?p,?by) ^ deathYear(?p,?dy)
    # ^ swrlb:subtract(?age,?dy,?by) ^ swrlb:greaterThan(?age,70)
    #   -> ElderScholar(?p)
    r4 = Imp()
    r4.label = ["elderScholar"]
    var_p   = r4.get_variable("p")
    var_by  = r4.get_variable("by")
    var_dy  = r4.get_variable("dy")
    var_age = r4.get_variable("age")

    p_atom  = ClassAtom(class_predicate=Person);    p_atom.arguments  = [var_p]
    by_atom = DatavaluedPropertyAtom(property_predicate=birthYear); by_atom.arguments = [var_p, var_by]
    dy_atom = DatavaluedPropertyAtom(property_predicate=deathYear); dy_atom.arguments = [var_p, var_dy]

    sub_atom = BuiltinAtom()
    sub_atom.builtin   = "http://www.w3.org/2003/11/swrlb#subtract"
    sub_atom.arguments = [var_age, var_dy, var_by]

    gt_atom = BuiltinAtom()
    gt_atom.builtin   = "http://www.w3.org/2003/11/swrlb#greaterThan"
    gt_atom.arguments = [var_age, 70]

    head4 = ClassAtom(class_predicate=ElderScholar); head4.arguments = [var_p]

    r4.body = [p_atom, by_atom, dy_atom, sub_atom, gt_atom]
    r4.head = [head4]
    print("\nRule 4 defined: Person(?p) ^ birthYear(?p,?by) ^ deathYear(?p,?dy)")
    print("              ^ swrlb:subtract(?age,?dy,?by) ^ swrlb:greaterThan(?age,70)")
    print("             -> ElderScholar(?p)")


# ── 4. Run Reasoner ────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("Running Reasoner")
print("=" * 60)

reasoner_ok = False

print("\nTrying HermiT (supports Rules 1-3; builtins need Pellet)...")
with onto:
    try:
        sync_reasoner(infer_property_values=True)
        print("  HermiT: OK")
        reasoner_ok = True
    except Exception as e:
        print(f"  HermiT: {type(e).__name__} - {str(e)[:120]}")

if not reasoner_ok:
    print("\nTrying Pellet (supports SWRL builtins including Rule 4)...")
    with onto:
        try:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True
            )
            print("  Pellet: OK")
            reasoner_ok = True
        except Exception as e:
            print(f"  Pellet: {type(e).__name__} - {str(e)[:120]}")
            print("  Falling back to manual rule application below.")


# ── 5. Report Inferred Facts ───────────────────────────────────────────────────

print()
print("=" * 60)
print("Inferred Facts")
print("=" * 60)

# Helper: safe name from OWLReady2 individual
def name(ind):
    return ind.name if hasattr(ind, "name") else str(ind)


# -- Rule 1 results: intellectualDescendantOf
print("\nRule 1 -- intellectualDescendantOf (studentOf chain):")
print("-" * 50)
found = []
for ind in onto.individuals():
    if not isinstance(ind, Person):
        continue
    for gm in ind.intellectualDescendantOf:
        found.append((name(ind), name(gm)))
if found:
    for p, gm in sorted(found):
        print(f"  {p:15s} intellectualDescendantOf  {gm}")
else:
    # Manual fallback
    print("  (Reasoner did not materialise; applying manually)")
    persons = list(onto.individuals())
    for ind in persons:
        if not isinstance(ind, Person): continue
        for teacher in ind.studentOf:
            for grandmaster in teacher.studentOf:
                if grandmaster != ind:
                    print(f"  {name(ind):15s} intellectualDescendantOf  {name(grandmaster)}")
                    found.append((name(ind), name(grandmaster)))
if not found:
    print("  (No chain found — check studentOf assertions)")
print(f"  => {len(found)} entailment(s)")


# -- Rule 2 results: foughtAgainst
print("\nRule 2 -- foughtAgainst (servedIn + attacked):")
print("-" * 50)
found2 = []
for ind in onto.individuals():
    if not isinstance(ind, Person): continue
    for enemy in ind.foughtAgainst:
        found2.append((name(ind), name(enemy)))
if found2:
    for p, e in sorted(found2):
        print(f"  {p:15s} foughtAgainst  {e}")
else:
    print("  (Reasoner did not materialise; applying manually)")
    for ind in onto.individuals():
        if not isinstance(ind, Person): continue
        for state in ind.servedIn:
            for enemy in state.attacked:
                if enemy != ind:
                    print(f"  {name(ind):15s} foughtAgainst  {name(enemy)}")
                    found2.append((name(ind), name(enemy)))
if not found2:
    print("  (No chain found)")
print(f"  => {len(found2)} entailment(s)")


# -- Rule 3 results: affiliatedWith
print("\nRule 3 -- affiliatedWith (teacher's philosophical school):")
print("-" * 50)
found3 = []
for ind in onto.individuals():
    if not isinstance(ind, Person): continue
    for school in ind.affiliatedWith:
        found3.append((name(ind), name(school)))
if found3:
    for p, s in sorted(found3):
        print(f"  {p:15s} affiliatedWith  {s}")
else:
    print("  (Reasoner did not materialise; applying manually)")
    for ind in onto.individuals():
        if not isinstance(ind, Person): continue
        for teacher in ind.studentOf:
            for school in teacher.movement:
                print(f"  {name(ind):15s} affiliatedWith  {name(school)}  (via {name(teacher)})")
                found3.append((name(ind), name(school)))
if not found3:
    print("  (No chain found)")
print(f"  => {len(found3)} entailment(s)")


# -- Rule 4 results: ElderScholar
print("\nRule 4 -- ElderScholar (lived more than 70 years):")
print("-" * 50)
found4 = []
for ind in onto.individuals():
    if isinstance(ind, ElderScholar):
        by = ind.birthYear
        dy = ind.deathYear
        age = (dy - by) if (dy is not None and by is not None) else "?"
        found4.append((name(ind), by, dy, age))
if found4:
    for p, by, dy, age in sorted(found4):
        print(f"  {p:15s} ElderScholar  (born {by}, died {dy}, lifespan ~{age} yr)")
else:
    print("  (Pellet needed for swrlb builtins; applying manually)")
    for ind in onto.individuals():
        if not isinstance(ind, Person): continue
        by = ind.birthYear
        dy = ind.deathYear
        if by is not None and dy is not None:
            age = dy - by   # works for BC: e.g. -479 - (-551) = 72
            if age > 70:
                print(f"  {name(ind):15s} ElderScholar  (born {by}, died {dy}, lifespan ~{age} yr)")
                found4.append((name(ind), by, dy, age))
if not found4:
    print("  (No individuals with lifespan data found)")
print(f"  => {len(found4)} entailment(s)")


# ── 6. Summary ────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("Summary")
print("=" * 60)
total = len(found) + len(found2) + len(found3) + len(found4)
print(f"  Rule 1 intellectualDescendantOf : {len(found):3d} new facts")
print(f"  Rule 2 foughtAgainst            : {len(found2):3d} new facts")
print(f"  Rule 3 affiliatedWith           : {len(found3):3d} new facts")
print(f"  Rule 4 ElderScholar             : {len(found4):3d} new facts")
print(f"  ----------------------------------------")
print(f"  Total inferred                  : {total:3d} new facts")
print()
print("Note: Rules 1-3 use only object properties (HermiT-compatible).")
print("      Rule 4 uses swrlb:subtract/greaterThan (Pellet required).")
print("      Manual fallback applied when reasoner does not materialise.")


# ── 7. Apply rules as SPARQL CONSTRUCT on the real KG → save for RAG ──────────

print()
print("=" * 60)
print("Applying rules via SPARQL CONSTRUCT on warring_states_final.ttl")
print("(Output saved to kg_artifacts/swrl_inferred.ttl for RAG pipeline)")
print("=" * 60)

import os
from pathlib import Path
from rdflib import Graph, Namespace, URIRef

WS  = Namespace("http://warring-states.kg/ontology#")
WSI = Namespace("http://warring-states.kg/instance/")

TTL_PATH = Path("kg_artifacts/warring_states_final.ttl")
OUT_PATH = Path("kg_artifacts/swrl_inferred.ttl")

# SPARQL CONSTRUCT equivalents of the four SWRL rules
SPARQL_R1 = """
PREFIX ws: <http://warring-states.kg/ontology#>
CONSTRUCT { ?p ws:intellectualDescendantOf ?gm }
WHERE {
    ?p ws:studentOf ?t .
    ?t ws:studentOf ?gm .
    FILTER(?p != ?gm)
}
"""

SPARQL_R2 = """
PREFIX ws: <http://warring-states.kg/ontology#>
CONSTRUCT { ?p ws:foughtAgainst ?e }
WHERE {
    ?p ws:servedIn ?s .
    ?s ws:attacked ?e .
    FILTER(?p != ?e)
}
"""

SPARQL_R3 = """
PREFIX ws: <http://warring-states.kg/ontology#>
CONSTRUCT { ?p ws:affiliatedWith ?school }
WHERE {
    ?p ws:studentOf ?t .
    ?t ws:movement  ?school .
}
"""

# Rule 4 equivalent: wasConqueredBy → atWarWith (symmetric)
SPARQL_R4 = """
PREFIX ws: <http://warring-states.kg/ontology#>
CONSTRUCT {
    ?s ws:atWarWith ?c .
    ?c ws:atWarWith ?s .
}
WHERE {
    ?s ws:wasConqueredBy ?c .
}
"""

if TTL_PATH.exists():
    kg = Graph()
    kg.parse(str(TTL_PATH), format="turtle")
    print(f"\nLoaded {len(kg):,} triples from {TTL_PATH}")

    inferred = Graph()
    counts = {}
    for label, sparql in [
        ("R1 intellectualDescendantOf", SPARQL_R1),
        ("R2 foughtAgainst",            SPARQL_R2),
        ("R3 affiliatedWith",           SPARQL_R3),
        ("R4 atWarWith",               SPARQL_R4),
    ]:
        before = len(inferred)
        for triple in kg.query(sparql):
            inferred.add(triple)
        n = len(inferred) - before
        counts[label] = n
        print(f"  {label:38s}: {n:3d} new triples")

    print(f"\n  Total inferred on real KG : {len(inferred):3d} triples")

    # Sample output
    print("\nSample inferred triples:")
    for i, (s, p, o) in enumerate(inferred):
        s_ = str(s).split("/")[-1]
        p_ = str(p).split("#")[-1]
        o_ = str(o).split("/")[-1]
        print(f"  {s_}  {p_}  {o_}")
        if i >= 14:
            print(f"  … ({len(inferred)} total)")
            break

    # Save enriched graph (original + inferred)
    enriched = Graph()
    enriched += kg
    enriched += inferred
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.serialize(destination=str(OUT_PATH), format="turtle")
    print(f"\nSaved enriched graph → {OUT_PATH}  ({len(enriched):,} triples)")
else:
    print(f"\n[WARNING] {TTL_PATH} not found — skipping SPARQL pass.")
    print("  Run build_kg.py and wikidata_enrichment.py first.")


# ── 8. Exercise 8 — Rule-based vs KGE comparison ──────────────────────────────

print()
print("=" * 60)
print("Exercise 8 — SWRL Rule vs. KGE embedding analogy")
print("=" * 60)
print("""
SWRL Rule 1:
  Person(?p) ^ studentOf(?p,?t) ^ studentOf(?t,?gm)
                -> intellectualDescendantOf(?p,?gm)

  KGE analogy: vector(studentOf) + vector(studentOf)
               ≈ vector(intellectualDescendantOf)?

  Test: after training TransE/DistMult, extract relation vectors and compute
        cosine_similarity(v(studentOf) + v(studentOf), v(intellectualDescendantOf))
  Expected: high similarity (>0.7) if the model learned transitivity.

SWRL Rule 2:
  Person(?p) ^ servedIn(?p,?s) ^ attacked(?s,?e)
                -> foughtAgainst(?p,?e)

  KGE analogy: vector(servedIn) + vector(attacked)
               ≈ vector(foughtAgainst)?

SWRL Rule 4 (atWarWith from wasConqueredBy):
  State(?s) ^ wasConqueredBy(?s,?c) -> atWarWith(?s,?c)

  KGE analogy: vector(wasConqueredBy) ≈ vector(atWarWith)?
  (conquering implies a war relation — expect high cosine similarity)

Rule-based reasoning is deterministic and interpretable.
KGE can generalise to unseen entities but cannot explain predictions.
The file kg_artifacts/swrl_inferred.ttl enriches the KG with inferred
triples so the RAG pipeline can answer questions about inferred facts.
""")
