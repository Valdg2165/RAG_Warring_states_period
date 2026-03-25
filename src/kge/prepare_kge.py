"""
prepare_kge.py  —  Prepare the Warring States KG for Knowledge Graph Embedding.

Reads  : kg_artifacts/swrl_inferred.ttl
Writes : data/kge/train.txt   (80 %)
         data/kge/valid.txt   (10 %)
         data/kge/test.txt    (10 %)
         data/kge/entity2id.txt
         data/kge/relation2id.txt
         data/kge/stats.txt

Each split line: <head_id> \\t <relation_id> \\t <tail_id>

Usage:
    python src/kge/prepare_kge.py [--input kg_artifacts/swrl_inferred.ttl]
                                  [--output-dir data/kge]
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Relations to EXCLUDE (too noisy / uninformative for embedding)
EXCLUDE_RELATIONS = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2002/07/owl#sameAs",
}

# coOccursWith is kept (it is the most frequent relation and gives graph connectivity)

SEED = 42


def shorten(uri: str) -> str:
    """Return the local name of a URI."""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def load_triples(ttl_path: Path):
    """Load all (subject, predicate, object) URI triples from a Turtle file."""
    from rdflib import Graph, URIRef

    print(f"Loading {ttl_path} …")
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    print(f"  Raw triples in graph : {len(g):,}")

    triples = []
    for s, p, o in g:
        # Keep only URI→URI triples (skip literals and blank nodes)
        if not isinstance(s, URIRef) or not isinstance(p, URIRef) or not isinstance(o, URIRef):
            continue
        if str(p) in EXCLUDE_RELATIONS:
            continue
        triples.append((str(s), str(p), str(o)))

    print(f"  URI-only triples kept: {len(triples):,}")
    return triples


def make_ids(triples):
    """Build entity→id and relation→id mappings."""
    entities = sorted({s for s, p, o in triples} | {o for s, p, o in triples})
    relations = sorted({p for s, p, o in triples})
    entity2id  = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def split(triples, entity2id, seed=SEED):
    """
    80 / 10 / 10 split.
    Guarantee: every entity that appears in valid/test also appears in train.
    """
    random.seed(seed)
    shuffled = list(triples)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test  = max(1, n // 10)
    n_valid = max(1, n // 10)
    n_train = n - n_test - n_valid

    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test  = shuffled[n_train + n_valid:]

    # Ensure coverage: any entity ONLY in valid/test → move triple to train
    train_ents = {s for s, p, o in train} | {o for s, p, o in train}

    def fix(split_set, name):
        keep, move = [], []
        for t in split_set:
            s, p, o = t
            if s in train_ents and o in train_ents:
                keep.append(t)
            else:
                move.append(t)
        if move:
            print(f"  Moved {len(move)} triples from {name} -> train (entity coverage)")
        return keep, move

    valid_keep, valid_move = fix(valid, "valid")
    test_keep,  test_move  = fix(test,  "test")
    train = train + valid_move + test_move

    return train, valid_keep, test_keep


def write_split(triples, entity2id, relation2id, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{entity2id[s]}\t{relation2id[p]}\t{entity2id[o]}\n")
    print(f"  Wrote {path}  ({len(triples):,} triples)")


def write_mapping(mapping: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for uri, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{shorten(uri)}\t{uri}\n")


def print_stats(triples, entity2id, relation2id, train, valid, test, out_dir: Path):
    """Print and save dataset statistics."""
    rel_counter = Counter(p for s, p, o in triples)
    lines = [
        "=== KGE Dataset Statistics ===",
        f"Total triples        : {len(triples):,}",
        f"  Train              : {len(train):,}",
        f"  Valid              : {len(valid):,}",
        f"  Test               : {len(test):,}",
        f"Unique entities      : {len(entity2id):,}",
        f"Unique relations     : {len(relation2id):,}",
        "",
        "Top-20 relations by frequency:",
    ]
    for rel, cnt in rel_counter.most_common(20):
        lines.append(f"  {shorten(rel):40s}  {cnt:6,}")

    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "stats.txt").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      default="kg_artifacts/swrl_inferred.ttl")
    ap.add_argument("--output-dir", default="data/kge")
    args = ap.parse_args()

    ttl_path = Path(args.input)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    triples = load_triples(ttl_path)

    # Remove duplicates
    triples = list(dict.fromkeys(triples))
    print(f"  After dedup         : {len(triples):,}")

    # 2. Build mappings
    entity2id, relation2id = make_ids(triples)
    print(f"  Entities            : {len(entity2id):,}")
    print(f"  Relations           : {len(relation2id):,}")

    # 3. Split
    train, valid, test = split(triples, entity2id)
    print(f"  Train / Valid / Test: {len(train):,} / {len(valid):,} / {len(test):,}")

    # 4. Write splits
    write_split(train, entity2id, relation2id, out_dir / "train.txt")
    write_split(valid, entity2id, relation2id, out_dir / "valid.txt")
    write_split(test,  entity2id, relation2id, out_dir / "test.txt")

    # 5. Write mappings
    write_mapping(entity2id,   out_dir / "entity2id.txt")
    write_mapping(relation2id, out_dir / "relation2id.txt")

    # 6. Stats
    print_stats(triples, entity2id, relation2id, train, valid, test, out_dir)

    print(f"\nDone. KGE data ready in: {out_dir}/")


if __name__ == "__main__":
    main()
