"""
rag_kge.py  —  Embedding-based RAG using KGE (DistMult) + Ollama
================================================================

Architecture
------------
1. Load trained DistMult entity embeddings (data/kge/results/model_distmult.pt)
2. Load RDF knowledge graph (kg_artifacts/swrl_inferred.ttl)
3. For each question:
   a. Entity detection  : match question words against entity short names
   b. KGE retrieval     : cosine similarity on DistMult embeddings → top-k related entities
   c. Triple expansion  : fetch 1-hop triples from RDF for every retrieved entity
   d. Context assembly  : serialise triples as readable "S --[p]--> O" lines
   e. Answer synthesis  : Ollama generates a grounded answer from the context

Usage
-----
Interactive CLI:
    python src/rag/rag_kge.py

Automated evaluation only:
    python src/rag/rag_kge.py --eval

Choose Ollama model:
    python src/rag/rag_kge.py --model mistral

Use as a library (called from app.py):
    from src.rag.rag_kge import load_resources, answer_question
    resources = load_resources()
    result = answer_question("Who were the students of Confucius?", resources)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix Anaconda OpenMP conflict on Windows

import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import requests
import torch
from rdflib import Graph, URIRef

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"

TTL_PATH     = Path("kg_artifacts/warring_states_final.ttl")
KGE_DIR      = Path("data/kge")
RESULTS_DIR  = KGE_DIR / "results"
MODEL_FILE   = RESULTS_DIR / "model_distmult.pt"

WS_ONTO = "http://warring-states.kg/ontology#"
WS_INST = "http://warring-states.kg/instance/"

TOP_K_SIMILAR       = 3    # extra entities retrieved via embedding similarity
MAX_TRIPLES_PER_ENT = 10   # triples fetched per entity (raised to capture more relations)
MAX_CONTEXT_TRIPLES = 15  # hard cap on context size

# Predicates to skip (too noisy for context)
NOISE_PREDS = {"coOccursWith", "wasDerivedFrom", "type",
               "inverseOf", "sameAs", "label", "comment"}

# ── Evaluation questions (same as rag_sparql for fair comparison) ─────────────

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

# ── Step 1: Load entity / relation mappings ────────────────────────────────────

def _read_mapping(path: Path):
    """Return (id2short, id2full, short2ids) from entity2id.txt or relation2id.txt."""
    id2short  = {}
    id2full   = {}
    short2ids = {}   # short_name_lower → list[int]  (several entities may share a short)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        idx   = int(parts[0])
        short = parts[1]
        full  = parts[2].strip() if len(parts) > 2 else short
        id2short[idx]  = short
        id2full[idx]   = full
        key = short.lower().replace("_", " ")
        short2ids.setdefault(key, []).append(idx)
    return id2short, id2full, short2ids


def load_mappings(kge_dir: Path = KGE_DIR):
    ent_id2short, ent_id2full, ent_short2ids = _read_mapping(kge_dir / "entity2id.txt")
    rel_id2short, rel_id2full, rel_short2ids = _read_mapping(kge_dir / "relation2id.txt")
    return ent_id2short, ent_id2full, ent_short2ids, rel_id2short


# ── Step 2: Load DistMult embeddings ─────────────────────────────────────────

def load_embeddings(model_path: Path = MODEL_FILE):
    """Load entity and relation weight matrices from the saved PyTorch checkpoint."""
    state   = torch.load(model_path, map_location="cpu")
    ent_emb = state["ent_emb.weight"].numpy()   # (n_ent, dim)
    rel_emb = state["rel_emb.weight"].numpy()   # (n_rel, dim)
    return ent_emb, rel_emb


# ── Step 3: Load RDF graph ────────────────────────────────────────────────────

def load_graph(ttl_path: Path = TTL_PATH) -> Graph:
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    return g


# ── Combined resource loader (used by app.py) ─────────────────────────────────

def load_resources(ttl_path: Path = TTL_PATH,
                   kge_dir:  Path = KGE_DIR,
                   model_path: Path = MODEL_FILE) -> dict:
    """Load everything once; return a dict passed to answer_question()."""
    print("Loading RDF graph …")
    g = load_graph(ttl_path)
    print(f"  {len(g):,} triples loaded.")

    print("Loading KGE mappings …")
    ent_id2short, ent_id2full, ent_short2ids, rel_id2short = load_mappings(kge_dir)

    print("Loading DistMult embeddings …")
    ent_emb, rel_emb = load_embeddings(model_path)

    # Precompute normalised embeddings for fast cosine search
    norms     = np.linalg.norm(ent_emb, axis=1, keepdims=True)
    ent_emb_n = ent_emb / (norms + 1e-8)

    print("Resources ready.")
    return {
        "graph":        g,
        "ent_id2short": ent_id2short,
        "ent_id2full":  ent_id2full,
        "ent_short2ids": ent_short2ids,
        "rel_id2short": rel_id2short,
        "ent_emb":      ent_emb,
        "ent_emb_n":    ent_emb_n,
    }


# ── Step 4a: Entity detection (string matching) ───────────────────────────────

def detect_entities(question: str, ent_short2ids: dict, ent_id2full: dict) -> list:
    """
    Return [(entity_id, short_name), ...] for entities whose name appears
    in the question (case-insensitive, underscore-normalised).

    When several entity IDs share the same short name, the one whose full URI
    lives in the instance namespace (WS_INST) is preferred — those entities
    carry the actual data triples (students, relations, etc.).
    """
    q_norm = question.lower().replace("_", " ")
    matches = []
    for key, ids in ent_short2ids.items():
        if len(key) < 3:
            continue
        if key.startswith("-") or re.match(r"^q\d+$", key):
            continue
        if key in q_norm:
            for eid in ids:
                matches.append((eid, key))

    # Group by short name, then pick the best ID for each name:
    # instance URI > any other URI (inference / ontology namespaces are noise here)
    from collections import defaultdict
    by_key: dict = defaultdict(list)
    for eid, key in matches:
        by_key[key].append(eid)

    best: dict = {}  # eid → key
    for key, eids in by_key.items():
        if len(eids) == 1:
            best[eids[0]] = key
        else:
            inst_ids = [e for e in eids if ent_id2full.get(e, "").startswith(WS_INST)]
            chosen = inst_ids if inst_ids else eids
            for e in chosen:
                best[e] = key

    # Sort: longer match (more specific) first
    result = sorted(best.items(), key=lambda x: -len(x[1]))

    # Remove sub-matches: if "shang" is a substring of a longer match "shang yang",
    # drop "shang" so it doesn't pull in the Shang dynasty as a noise entity.
    long_keys = {key for _, key in result}
    filtered = [
        (eid, key) for eid, key in result
        if not any(key != lk and key in lk for lk in long_keys)
    ]

    return filtered[:12]


# ── Step 4b: KGE similarity retrieval ────────────────────────────────────────

def find_similar_entities(entity_ids: list,
                          ent_emb_n: np.ndarray,
                          top_k: int = TOP_K_SIMILAR) -> list:
    """
    Average the embeddings of the detected entities, then find the
    cosine-nearest neighbours across all entities.
    Returns [(entity_id, similarity_score), ...]
    """
    if not entity_ids:
        return []
    seed_ids   = [eid for eid, _ in entity_ids]
    query      = np.mean(ent_emb_n[seed_ids], axis=0)
    query     /= np.linalg.norm(query) + 1e-8
    scores     = ent_emb_n @ query                    # (n_ent,)
    top_ids    = np.argsort(scores)[::-1]
    seed_set   = set(seed_ids)
    similar    = [(int(i), float(scores[i]))
                  for i in top_ids if i not in seed_set]
    return similar[:top_k]


# ── Step 4c: Triple retrieval ─────────────────────────────────────────────────

def _shorten(uri: str) -> str:
    """Convert a full URI to a readable local name."""
    if uri.startswith("http"):
        return uri.split("/")[-1].split("#")[-1]
    return uri


def _fmt_triple(s: str, p: str, o: str):
    """Format one triple as a readable sentence; return None for noisy predicates."""
    p_short = _shorten(p)
    if p_short in NOISE_PREDS:
        return None
    s_ = _shorten(s).replace("_", " ")
    o_ = _shorten(o).replace("_", " ")
    return f"{s_} | {p_short} | {o_}"


def get_entity_triples(g: Graph, full_uri: str,
                       max_triples: int = MAX_TRIPLES_PER_ENT) -> list:
    """Retrieve outgoing + incoming triples for one entity URI."""
    uri    = URIRef(full_uri)
    lines  = []
    for s, p, o in g.triples((uri, None, None)):
        line = _fmt_triple(str(s), str(p), str(o))
        if line:
            lines.append(line)
        if len(lines) >= max_triples:
            break
    for s, p, o in g.triples((None, None, uri)):
        line = _fmt_triple(str(s), str(p), str(o))
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= max_triples:
            break
    return lines


def build_context(g: Graph,
                  detected: list,
                  similar:  list,
                  ent_id2full: dict,
                  max_triples: int = MAX_CONTEXT_TRIPLES) -> str:
    """
    Fetch triples for detected entities first (higher priority),
    then for similar entities until the cap is reached.
    """
    seen    = set()
    context = []
    all_ids = [(eid, "seed") for eid, _ in detected] + \
              [(eid, "sim")  for eid, _ in similar]

    for eid, _ in all_ids:
        full_uri = ent_id2full.get(eid, "")
        if not full_uri.startswith("http"):
            continue
        for line in get_entity_triples(g, full_uri):
            if line not in seen:
                seen.add(line)
                context.append(line)
            if len(context) >= max_triples:
                return "\n".join(context)

    return "\n".join(context)


# ── Step 5: Ollama call ───────────────────────────────────────────────────────

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
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── Baseline: LLM without KG ─────────────────────────────────────────────────

_BASELINE_PROMPT = """\
Answer the following question about the Warring States period of ancient China.
Give a direct, factual answer in 2-4 sentences.

Question: {question}
"""

def answer_baseline(question: str, model: str) -> str:
    """Ask the LLM directly, with no knowledge graph context."""
    return ask_llm(_BASELINE_PROMPT.format(question=question), model)


# ── Step 6: Answer synthesis ──────────────────────────────────────────────────

_ANSWER_PROMPT = """\
You are an expert on the Warring States period of ancient China.
Use ONLY the facts listed below to answer the question.

Each fact is written as:   Subject | predicate | Value

RULES:
- Read every fact line carefully. Extract the value directly from the matching predicate.
- List EVERY entity that is relevant to the question. Do not skip any.
- Answer in 2-5 sentences. Start directly with the answer, no preamble.
- If the facts truly contain nothing relevant, say "Not found in knowledge graph."

FACTS FROM KNOWLEDGE GRAPH:
{context}

QUESTION: {question}

ANSWER:"""

_NO_CONTEXT_PROMPT = """\
Answer the following question about the Warring States period of ancient China.
Give a direct, factual answer in 2-4 sentences.

Question: {question}
"""

def synthesize_answer(question: str, context: str, model: str) -> str:
    if not context.strip():
        # No matching triples in the KG — answer from LLM alone, with a note
        ans = ask_llm(_NO_CONTEXT_PROMPT.format(question=question), model)
        return ans + "\n\n⚠️ (no matching entities found in the knowledge graph — answer from LLM only)"
    prompt = _ANSWER_PROMPT.format(context=context, question=question)
    return ask_llm(prompt, model)


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def answer_question(question: str, resources: dict, model: str = DEFAULT_MODEL) -> dict:
    """
    Public API used by both the CLI and app.py.

    Returns a dict with:
      answer           : str  — natural-language answer
      detected_entities: list — entity names matched in the question
      similar_entities : list — KGE-retrieved neighbours
      context_triples  : int  — number of triples in context
      context          : str  — full context passed to the LLM
    """
    g            = resources["graph"]
    ent_id2short = resources["ent_id2short"]
    ent_id2full  = resources["ent_id2full"]
    ent_short2ids = resources["ent_short2ids"]
    ent_emb_n    = resources["ent_emb_n"]

    # 1. Entity detection
    detected = detect_entities(question, ent_short2ids, ent_id2full)

    # 2. KGE similarity
    similar = find_similar_entities(detected, ent_emb_n, TOP_K_SIMILAR)

    # 3. Build context from triples
    context = build_context(g, detected, similar, ent_id2full)

    # 4. Generate answer
    answer = synthesize_answer(question, context, model)

    return {
        "answer":            answer,
        "detected_entities": [name for _, name in detected],
        "similar_entities":  [ent_id2short.get(eid, str(eid)) for eid, _ in similar],
        "context_triples":   len(context.splitlines()) if context else 0,
        "context":           context,
    }


# ── Evaluation suite ──────────────────────────────────────────────────────────

def run_evaluation(resources: dict, model: str = DEFAULT_MODEL) -> list:
    print("\n" + "=" * 70)
    print(f"KGE-RAG EVALUATION  ({len(EVAL_QUESTIONS)} questions)  model={model}")
    print("=" * 70)

    records = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {q['question']}")
        result = answer_question(q["question"], resources, model)
        kw  = q["expected_keyword"].lower()
        # Normalise underscores so "Sun_Tzu" matches keyword "sun tzu"
        ans_norm = result["answer"].lower().replace("_", " ")
        ctx_norm = result["context"].lower().replace("_", " ")
        correct = (kw in ans_norm or kw in ctx_norm)

        print(f"  Detected : {result['detected_entities']}")
        print(f"  Similar  : {result['similar_entities'][:4]}")
        print(f"  Triples  : {result['context_triples']}")
        print(f"  Answer   : {textwrap.fill(result['answer'], 66)}")
        print(f"  Correct  : {correct}  (keyword: '{kw}')")

        records.append({
            "question":          q["question"],
            "reference":         q["reference"],
            "answer":            result["answer"],
            "correct":           correct,
            "detected_entities": result["detected_entities"],
            "similar_entities":  result["similar_entities"],
            "context_triples":   result["context_triples"],
        })

    # Summary
    score = sum(r["correct"] for r in records)
    print("\n" + "=" * 70)
    print(f"Score: {score}/{len(records)}")
    print("=" * 70)
    for i, r in enumerate(records, 1):
        tag = "OK" if r["correct"] else "FAIL"
        print(f"  [{tag}] {r['question'][:55]:<55}  triples={r['context_triples']}")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def interactive_loop(resources: dict, model: str):
    print("\n" + "=" * 70)
    print("Warring States KG — KGE-RAG Demo")
    print(f"Graph: {len(resources['graph']):,} triples  |  Model: {model}")
    print("Commands: 'eval' = run evaluation | 'quit' = exit")
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
            records = run_evaluation(resources, model)
            out = Path("data/kge_rag_evaluation.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nEvaluation saved to {out}")
            continue

        result = answer_question(q, resources, model)
        w = 70
        print("\n" + "=" * w)
        print(f"Detected entities : {result['detected_entities']}")
        print(f"Similar (KGE)     : {result['similar_entities'][:4]}")
        print(f"Context triples   : {result['context_triples']}")
        print(f"\nAnswer: {textwrap.fill(result['answer'], w)}")


def main():
    ap = argparse.ArgumentParser(description="KGE-based RAG with Ollama")
    ap.add_argument("--graph",   default=str(TTL_PATH),   help="Turtle KG file")
    ap.add_argument("--kge-dir", default=str(KGE_DIR),    help="KGE data directory")
    ap.add_argument("--model",   default=DEFAULT_MODEL,    help="Ollama model name")
    ap.add_argument("--eval",    action="store_true",      help="Run evaluation and exit")
    args = ap.parse_args()

    resources = load_resources(
        ttl_path=Path(args.graph),
        kge_dir=Path(args.kge_dir),
    )

    if args.eval:
        records = run_evaluation(resources, args.model)
        out = Path("data/kge_rag_evaluation.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nEvaluation saved to {out}")
    else:
        interactive_loop(resources, args.model)


if __name__ == "__main__":
    main()
