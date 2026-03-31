import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import requests
from rdflib import Graph


OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"
TTL_PATH      = Path("kg_artifacts/swrl_inferred.ttl")

MAX_PREDICATES = 60
MAX_CLASSES    = 20
SAMPLE_TRIPLES = 25
MAX_REPAIR_ATTEMPTS = 2
MAX_RESULT_ROWS     = 20

WS_ONTO = "http://warring-states.kg/ontology#"
WS_INST = "http://warring-states.kg/instance/"


EVAL_QUESTIONS = [
    {
        "question": "Who were the students of Confucius?",
        "expected_keyword": "student",
        "reference": "Zisi, Tantai Mieming, and others",
    },
    {
        "question": "Which states were at war with the State of Qin?",
        "expected_keyword": "Zhao",
        "reference": "Zhao, Wei, Han, Chu, Yan, Qi",
    },
    {
        "question": "Who did Aristotle influence?",
        "expected_keyword": "Alexander",
        "reference": "Alexander the Great, Strato of Lampsacus",
    },
    {
        "question": "Who is an intellectual descendant of Confucius?",
        "expected_keyword": "Zisi",
        "reference": "Zisi, Mencius (via Zisi studentOf chain)",
    },
    {
        "question": "Which people held a position as writer?",
        "expected_keyword": "Sun Tzu",
        "reference": "Sun Tzu, Li Si, Sima Qian, Mencius, Laozi, Confucius",
    },
    {
        "question": "Who was born in the State of Zhao?",
        "expected_keyword": "Xunzi",
        "reference": "Xunzi (bornIn Zhao)",
    },
    {
        "question": "Which philosophers influenced Epicharmus of Kos?",
        "expected_keyword": "Pythagoras",
        "reference": "Pythagoras, Xenophanes (from influencedBy triples)",
    },
]


def ask_llm(prompt: str, model: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"[ERROR] {e}"



def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g



def build_schema_summary(g: Graph) -> str:
    ns_lines = []
    for prefix, ns in g.namespace_manager.namespaces():
        ns_lines.append(f"PREFIX {prefix}: <{ns}>")

    class_q = """
        SELECT DISTINCT ?cls (COUNT(?s) AS ?cnt) WHERE {
            ?s a ?cls .
            FILTER(STRSTARTS(STR(?cls), "http://warring-states.kg/ontology#"))
        } GROUP BY ?cls ORDER BY DESC(?cnt)
    """
    classes = []
    for row in g.query(class_q):
        short = str(row.cls).split("#")[-1]
        classes.append(f"  ws:{short}  ({row.cnt} instances)")

    # Predicates
    pred_q = f"""
        SELECT DISTINCT ?p (COUNT(*) AS ?cnt) WHERE {{
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), "{WS_ONTO}"))
        }} GROUP BY ?p ORDER BY DESC(?cnt) LIMIT {MAX_PREDICATES}
    """
    preds = []
    for row in g.query(pred_q):
        short = str(row.p).split("#")[-1]
        preds.append(f"  ws:{short}  ({row.cnt})")

    sample_q = f"""
        SELECT ?s ?p ?o WHERE {{
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), "{WS_ONTO}"))
            FILTER(!STRSTARTS(STR(?p), "{WS_ONTO}coOccursWith"))
            FILTER(!STRSTARTS(STR(?p), "{WS_ONTO}wasDerivedFrom"))
            FILTER(isIRI(?o))
        }} LIMIT {SAMPLE_TRIPLES}
    """
    samples = []
    for row in g.query(sample_q):
        s = str(row.s).split("/")[-1]
        p = str(row.p).split("#")[-1]
        o = str(row.o).split("/")[-1]
        samples.append(f"  {s}  ws:{p}  {o}")

    summary = f"""# Warring States Knowledge Graph — Schema Summary

## Namespace prefixes
PREFIX ws:  <{WS_ONTO}>
PREFIX wsi: <{WS_INST}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

## Classes (use with: ?x a ws:ClassName)
{chr(10).join(classes[:MAX_CLASSES])}

## Predicates (use with: ?s ws:predicateName ?o)
{chr(10).join(preds)}

## Sample triples (entity names as local URI parts, e.g. wsi:Confucius or just Confucius as rdfs:label)
{chr(10).join(samples)}

## Important notes for SPARQL generation
- Entity IRIs look like: <http://warring-states.kg/instance/Confucius>
  OR use rdfs:label matching: ?x rdfs:label "Confucius"@en
- Predicate IRIs: <http://warring-states.kg/ontology#studentOf>
- Always add LIMIT 20 to avoid huge results
"""
    return summary.strip()



BASELINE_PROMPT = """Answer the following question about the Warring States period of ancient China.
Give a direct, factual answer in 2-4 sentences.

Question: {question}
"""

def answer_baseline(question: str, model: str) -> str:
    return ask_llm(BASELINE_PROMPT.format(question=question), model)



SPARQL_SYSTEM = """You generate SPARQL queries for a Warring States China RDF graph.

STRICT RULES — follow exactly:
1. Copy these PREFIX lines verbatim at the top of every query:
   PREFIX ws:   <http://warring-states.kg/ontology#>
   PREFIX wsi:  <http://warring-states.kg/instance/>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
2. NEVER use any other prefix (no wdt:, wd:, owl:, xsd:, w:, wsf:, wsd:).
3. To find an entity by name use: FILTER(CONTAINS(LCASE(STR(?x)), "name"))
4. To find its properties use: ?x ws:predicateName ?y
5. No trailing semicolons after the closing brace.
6. Always add LIMIT 20.
7. Return ONLY a ```sparql ... ``` block. No text outside it.

EXAMPLES:

Question: Who are the students of Confucius?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?studentName WHERE {
  ?teacher rdfs:label ?tl .
  FILTER(CONTAINS(LCASE(STR(?tl)), "confucius"))
  ?student ws:studentOf ?teacher .
  ?student rdfs:label ?studentName .
} LIMIT 20
```

Question: Which states were at war with Qin?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX wsi:  <http://warring-states.kg/instance/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?stateName WHERE {
  ?qin rdfs:label ?ql .
  FILTER(CONTAINS(LCASE(STR(?ql)), "qin"))
  ?state ws:atWarWith ?qin .
  ?state rdfs:label ?stateName .
} LIMIT 20
```

Question: Who did Aristotle influence?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?name WHERE {
  ?aristotle rdfs:label ?al .
  FILTER(CONTAINS(LCASE(STR(?al)), "aristotle"))
  ?aristotle ws:influenced ?person .
  ?person rdfs:label ?name .
} LIMIT 20
```

Question: Who is an intellectual descendant of Confucius?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?descendantName WHERE {
  ?teacher rdfs:label ?tl .
  FILTER(CONTAINS(LCASE(STR(?tl)), "confucius"))
  ?person ws:intellectualDescendantOf ?teacher .
  ?person rdfs:label ?descendantName .
} LIMIT 20
```

Question: Which people held a position as writer?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?personName WHERE {
  ?person ws:heldPosition ?pos .
  ?pos rdfs:label ?pl .
  FILTER(CONTAINS(LCASE(STR(?pl)), "writer"))
  ?person rdfs:label ?personName .
} LIMIT 20
```

Question: Who was born in the State of Zhao?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?personName WHERE {
  {
    ?person ws:bornIn ?place .
    FILTER(CONTAINS(LCASE(STR(?place)), "zhao"))
  } UNION {
    ?place ws:birthplaceOf ?person .
    FILTER(CONTAINS(LCASE(STR(?place)), "zhao"))
  }
  ?person rdfs:label ?personName .
} LIMIT 20
```

Question: Which philosophers influenced Epicharmus of Kos?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?philosopherName WHERE {
  ?epicharmus rdfs:label ?el .
  FILTER(CONTAINS(LCASE(STR(?el)), "epicharmus"))
  ?epicharmus ws:influencedBy ?philosopher .
  ?philosopher rdfs:label ?philosopherName .
} LIMIT 20
```

Question: What did Laozi author?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?bookName WHERE {
  ?person rdfs:label ?pl .
  FILTER(CONTAINS(LCASE(STR(?pl)), "laozi"))
  ?person ws:authored ?book .
  ?book rdfs:label ?bookName .
} LIMIT 20
```

Question: What states took part in the Warring States period?
```sparql
PREFIX ws:   <http://warring-states.kg/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?stateName WHERE {
  ?state a ws:State .
  ?state rdfs:label ?stateName .
} LIMIT 20
```
"""

def make_sparql_prompt(schema: str, question: str) -> str:
    return f"""{SPARQL_SYSTEM}

Available predicates from this graph:
{_extract_pred_list(schema)}

Now generate a query for:
Question: {question}
```sparql"""


def _extract_pred_list(schema: str) -> str:
    """Pull just the predicate lines from the schema summary."""
    lines = [l for l in schema.splitlines() if l.strip().startswith("ws:")]
    return "\n".join(lines[:30]) if lines else "(see schema above)"


REPAIR_SYSTEM = """The SPARQL query below failed. Fix it.

STRICT RULES:
1. Use ONLY these prefixes:
   PREFIX ws:   <http://warring-states.kg/ontology#>
   PREFIX wsi:  <http://warring-states.kg/instance/>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
2. No other prefixes. No trailing semicolons.
3. Use FILTER(CONTAINS(LCASE(STR(?x)), "name")) to match entities.
4. Return ONLY a ```sparql ... ``` block.
"""

def make_repair_prompt(schema: str, question: str, bad_query: str, error: str) -> str:
    return f"""{REPAIR_SYSTEM}

QUESTION: {question}

FAILED SPARQL:
{bad_query}

ERROR: {error}

Write the corrected query:
```sparql"""


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

_PREFIX_FIXES = [
    (re.compile(r"\bwdfs:"),          "rdfs:"),  # wdfs: → rdfs:
    (re.compile(r"\bw:label\b"),      "rdfs:label"),
    (re.compile(r"\bw:[ds]?label\b"), "rdfs:label"),
    (re.compile(r"\bwdt:"),           "ws:"),
    (re.compile(r"\bwsf:"),           "ws:"),
    (re.compile(r"\bwsd:"),           "ws:"),
    (re.compile(r"\bw:"),             "ws:"),    # bare w: → ws:
    (re.compile(r"\bwd:"),            "ws:"),    # wd: → ws:
    # Drop SERVICE wikibase / bd: lines entirely
    (re.compile(r"SERVICE\s+wikibase:[^\n]+\n?"),  ""),
    (re.compile(r"\bbd:\S+[^\n]*\n?"),             ""),
    (re.compile(r"\bwikibase:\S+"),                ""),
]

def _clean_sparql(sparql: str) -> str:
    sparql = re.split(r"</", sparql)[0]
    last_brace = sparql.rfind("}")
    if last_brace != -1:
        tail = sparql[last_brace:]
        limit_m = re.search(r"\}\s*(LIMIT\s+\d+)?", tail)
        sparql = sparql[:last_brace] + (limit_m.group(0) if limit_m else "}")
    sparql = re.sub(r"\?\s+([A-Za-z])", r"?\1", sparql)
    for pat, repl in _PREFIX_FIXES:
        sparql = pat.sub(repl, sparql)
    return sparql.strip()


def extract_sparql(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    raw = re.sub(r"```\s*$", "", raw)
    raw = re.sub(r"^```\w*\s*", "", raw)
    return _clean_sparql(raw)


def run_sparql(g: Graph, query: str):
    results = g.query(query)
    vars_   = [str(v) for v in results.vars]
    rows    = [tuple(str(c) for c in row) for row in results]
    return vars_, rows



SYNTHESIS_PROMPT = """Turn these facts into one sentence. Do NOT say "I cannot" or "context". Just state the facts.

Facts: {results}
Question: {question}

One sentence (start with the main subject, not with "I"):"""

_REFUSAL_PHRASES = [
    "cannot answer", "i cannot", "context does not", "not provided",
    "not mentioned", "passage does not", "i don't", "does not specify",
    "does not provide", "not in the context",
]

def _fmt_values(values: list) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) <= 5:
        return ", ".join(values[:-1]) + " and " + values[-1]
    shown = values[:4]
    return ", ".join(shown) + f", and {len(values) - 4} others"


def synthesize_answer(question: str, vars_: list, rows: list, model: str) -> str:
    if not rows:
        return "(no results from knowledge graph)"
    values = []
    for row in rows[:20]:
        for c in row:
            if c.startswith("http"):
                c = c.split("/")[-1].split("#")[-1]
            if re.match(r"^Q\d+$", c):
                continue
            values.append(c)
    values = list(dict.fromkeys(values))  
    if not values:
        return "(results contained only unresolved identifiers)"

    result_str = _fmt_values(values)
    prompt = SYNTHESIS_PROMPT.format(question=question, results=result_str)
    nl = ask_llm(prompt, model)

    if any(p in nl.lower() for p in _REFUSAL_PHRASES):
        nl = f"{result_str}."
    return nl



def answer_rag(question: str, g: Graph, schema: str, model: str) -> dict:
    sparql = extract_sparql(ask_llm(make_sparql_prompt(schema, question), model))
    attempt = 1
    repaired = False
    last_error = None

    while attempt <= 1 + MAX_REPAIR_ATTEMPTS:
        try:
            vars_, rows = run_sparql(g, sparql)
            if rows:
                nl = synthesize_answer(question, vars_, rows, model)
                return {
                    "sparql": sparql, "vars": vars_, "rows": rows,
                    "nl_answer": nl,
                    "repaired": repaired, "error": None, "attempts": attempt,
                }
            error_hint = "Query executed but returned 0 results. Try relaxing filters or use CONTAINS/LCASE for label matching."
            if attempt > MAX_REPAIR_ATTEMPTS:
                return {
                    "sparql": sparql, "vars": vars_, "rows": [],
                    "nl_answer": "(no results from knowledge graph)",
                    "repaired": repaired, "error": error_hint, "attempts": attempt,
                }
        except Exception as exc:
            last_error = str(exc)
            if attempt > MAX_REPAIR_ATTEMPTS:
                return {
                    "sparql": sparql, "vars": [], "rows": [],
                    "nl_answer": "(query failed)",
                    "repaired": repaired, "error": last_error, "attempts": attempt,
                }
            error_hint = last_error

        sparql   = extract_sparql(ask_llm(make_repair_prompt(schema, question, sparql, error_hint), model))
        repaired = True
        attempt += 1

    return {"sparql": sparql, "vars": [], "rows": [], "nl_answer": "(query failed)",
            "repaired": True, "error": last_error, "attempts": attempt}



def fmt_rows(vars_: list, rows: list) -> str:
    if not rows:
        return "  (no results)"
    header = " | ".join(f"{v:25s}" for v in vars_)
    sep    = "-" * len(header)
    lines  = [header, sep]
    for row in rows[:MAX_RESULT_ROWS]:
        cells = []
        for c in row:
            if c.startswith("http"):
                c = c.split("/")[-1].split("#")[-1]
            cells.append(f"{c:25s}")
        lines.append(" | ".join(cells))
    if len(rows) > MAX_RESULT_ROWS:
        lines.append(f"  … ({len(rows)} total, showing {MAX_RESULT_ROWS})")
    return "\n".join(lines)


def print_comparison(question: str, baseline: str, rag: dict):
    w = 70
    print("\n" + "=" * w)
    print(f"Q: {question}")
    print("=" * w)

    print("\n[Baseline — LLM only, no KG]")
    print(textwrap.fill(baseline, width=w, initial_indent="  ", subsequent_indent="  "))

    print(f"\n[RAG — KG-grounded answer (attempt {rag['attempts']}, repaired={rag['repaired']})]")
    if rag["error"] and not rag["rows"]:
        print(f"  [FAIL] {rag['error']}")
        print(f"  Failed SPARQL:\n{textwrap.indent(rag['sparql'], '    ')}")
    else:
        print(f"\n  Answer: {rag.get('nl_answer', '(see rows below)')}")
        print(f"\n  Evidence ({len(rag['rows'])} rows from KG):")
        print(textwrap.indent(fmt_rows(rag["vars"], rag["rows"]), "    "))
        print(f"\n  SPARQL used:\n{textwrap.indent(rag['sparql'], '    ')}")



def run_evaluation(g: Graph, schema: str, model: str) -> list:
    print("\n" + "=" * 70)
    print(f"EVALUATION SUITE  ({len(EVAL_QUESTIONS)} questions)  model={model}")
    print("=" * 70)

    records = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {q['question']}")
        baseline = answer_baseline(q["question"], model)
        rag      = answer_rag(q["question"], g, schema, model)
        print_comparison(q["question"], baseline, rag)

        rag_answer = rag.get("nl_answer", "(no results)")
        raw_rows_str = fmt_rows(rag["vars"], rag["rows"]) if rag["rows"] else ""
        kw = q["expected_keyword"].lower()
        correct_rag      = (kw in rag_answer.lower() or kw in raw_rows_str.lower()) if rag["rows"] else False
        correct_baseline = kw in baseline.lower()

        records.append({
            "question":         q["question"],
            "reference":        q["reference"],
            "baseline_answer":  baseline[:200],
            "baseline_correct": correct_baseline,
            "rag_nl_answer":    rag_answer[:400],
            "rag_correct":      correct_rag,
            "rag_rows":         len(rag["rows"]),
            "repaired":         rag["repaired"],
            "attempts":         rag["attempts"],
            "sparql":           rag["sparql"],
        })

    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 70)
    hdr = f"{'#':>2}  {'Question':<45}  {'Baseline':>8}  {'RAG':>5}  {'Rows':>5}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(records, 1):
        b = "correct" if r["baseline_correct"] else "wrong"
        ra = "correct" if r["rag_correct"] else ("0 rows" if r["rag_rows"] == 0 else "wrong")
        print(f"{i:>2}  {r['question'][:45]:<45}  {b:>8}  {ra:>5}  {r['rag_rows']:>5}")

    baseline_score = sum(r["baseline_correct"] for r in records)
    rag_score      = sum(r["rag_correct"] for r in records)
    print(f"\nBaseline correct: {baseline_score}/{len(records)}")
    print(f"RAG correct     : {rag_score}/{len(records)}")

    return records



def interactive_loop(g: Graph, schema: str, model: str):
    print("\n" + "=" * 70)
    print("Warring States KG — RAG Demo")
    print(f"Graph: {len(g):,} triples  |  Model: {model}")
    print("Commands: 'eval' = run evaluation suite | 'quit' = exit")
    print("=" * 70)

    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            continue
        if q.lower() == "quit":
            print("Bye.")
            break
        if q.lower() == "eval":
            records = run_evaluation(g, schema, model)
            out = Path("data/rag_evaluation.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nEvaluation saved to {out}")
            continue

        baseline = answer_baseline(q, model)
        rag      = answer_rag(q, g, schema, model)
        print_comparison(q, baseline, rag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",  default=str(TTL_PATH), help="Path to Turtle KG file")
    ap.add_argument("--model",  default=DEFAULT_MODEL,  help="Ollama model name")
    ap.add_argument("--eval",   action="store_true",    help="Run evaluation suite and exit")
    args = ap.parse_args()

    print(f"Loading KG from {args.graph} …")
    g = load_graph(Path(args.graph))
    print(f"  {len(g):,} triples loaded.")

    print("Building schema summary …")
    schema = build_schema_summary(g)

    if args.eval:
        records = run_evaluation(g, schema, args.model)
        out = Path("data/rag_evaluation.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nEvaluation saved to {out}")
    else:
        interactive_loop(g, schema, args.model)


if __name__ == "__main__":
    main()