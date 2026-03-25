"""
Cleaning + Named Entity Recognition (NER) for Warring States KB.

Input:  data/crawler_output.jsonl
Output: data/extracted_knowledge.csv   (entities + relations)
        data/cleaned_texts.jsonl        (cleaned sentences)

Steps:
  1. Clean text  – strip noise lines, normalise whitespace
  2. NER         – extract PERSON, ORG, GPE, LOC, DATE, EVENT, NORP
  3. Relations   – co-occurrence in same sentence + dependency-based triples
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict

import spacy

# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_FILE   = "data/crawler_output.jsonl"
ENTITY_FILE  = "data/extracted_knowledge.csv"
CLEANED_FILE = "data/cleaned_texts.jsonl"
TRIPLES_FILE = "data/relation_triples.csv"

# Entity types to keep
KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "NORP"}

# Noise patterns to strip from Wikipedia text
NOISE_PATTERNS = [
    r"^\s*\[edit\]\s*$",
    r"^\s*See also\s*$",
    r"^\s*References\s*$",
    r"^\s*External links\s*$",
    r"^\s*Notes\s*$",
    r"^\s*Further reading\s*$",
    r"^\s*\d+\s*$",              # bare page numbers
    r"^\s*\^.*$",                # citation markers
    r"ISBN\s[\d\-X]+",
]

# Dependency relations that signal a predicate between subject and object
SUBJ_DEPS  = {"nsubj", "nsubjpass"}
OBJ_DEPS   = {"dobj", "pobj", "attr", "nmod"}

# Role/occupation keywords to capture as heldPosition triples
ROLE_KEYWORDS = {
    "chancellor", "minister", "general", "philosopher", "strategist",
    "statesman", "politician", "king", "duke", "lord", "emperor",
    "adviser", "advisor", "reformer", "scholar", "writer", "poet",
    "military commander", "prime minister", "court official",
    "military strategist", "political philosopher",
}

# Patterns: "[Name] was a/an <role>" or "[Name], <role> of <Place>"
_ROLE_WAS   = re.compile(
    r"([A-Z][a-zA-Z\s\-]+?)\s+was\s+(?:a|an)\s+([\w\s]+?)(?:\s+of\b|\s+in\b|[,\.])",
    re.UNICODE,
)
_ROLE_APPOS = re.compile(
    r"([A-Z][a-zA-Z\s\-]+?),\s+([\w\s]+?)\s+of\s+([A-Z][a-zA-Z\s]+?)(?:[,\.])",
    re.UNICODE,
)

MODEL = "en_core_web_trf"   # swap to "en_core_web_sm" if memory is tight

# ── 1. Text cleaning ──────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        # Drop noise lines
        if any(re.match(p, line) for p in NOISE_PATTERNS):
            continue
        # Drop very short lines (likely headers / stray bullets)
        stripped = line.strip()
        if len(stripped) < 20:
            continue
        cleaned.append(stripped)
    text = " ".join(cleaned)
    # Collapse multiple spaces / dashes
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"–{2,}", "—", text)
    return text.strip()


# ── 2. NER ─────────────────────────────────────────────────────────────────────

def extract_entities(doc, source_url: str, source_title: str) -> list[dict]:
    rows = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in KEEP_LABELS:
            continue
        text = ent.text.strip()
        # Skip single characters, numbers only, very long strings (noise)
        if len(text) < 2 or re.fullmatch(r"[\d\s\-–,\.]+", text) or len(text) > 80:
            continue
        key = (text.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "entity":     text,
            "label":      ent.label_,
            "source_url": source_url,
            "source_title": source_title,
        })
    return rows


# ── 3. Relation extraction (dependency + co-occurrence) ────────────────────────

def extract_relations_dep(doc, source_url: str) -> list[dict]:
    """
    For each sentence, find entity pairs linked by a verb via dependency parsing.
    Pattern: SUBJ_entity —[verb]→ OBJ_entity
    """
    triples = []
    for sent in doc.sents:
        ents_in_sent = [e for e in sent.ents if e.label_ in KEEP_LABELS]
        if len(ents_in_sent) < 2:
            continue

        ent_tokens = {}
        for ent in ents_in_sent:
            for tok in ent:
                ent_tokens[tok.i] = ent

        for token in sent:
            if token.pos_ != "VERB":
                continue
            subj_ent = obj_ent = None
            for child in token.children:
                if child.dep_ in SUBJ_DEPS and child.i in ent_tokens:
                    subj_ent = ent_tokens[child.i]
                if child.dep_ in OBJ_DEPS and child.i in ent_tokens:
                    obj_ent = ent_tokens[child.i]
            if subj_ent and obj_ent and subj_ent != obj_ent:
                triples.append({
                    "subject":    subj_ent.text.strip(),
                    "predicate":  token.lemma_,
                    "object":     obj_ent.text.strip(),
                    "source_url": source_url,
                })
    return triples


def extract_roles(text: str, source_url: str) -> list[dict]:
    """
    Extract heldPosition triples from the lead paragraph.
    Captures patterns like:
      "Shang Yang was a chancellor of Qin"   → (Shang Yang, heldPosition, chancellor)
      "Sun Tzu, military strategist of Wu"   → (Sun Tzu, heldPosition, military strategist)
    Only looks at the first 1000 characters (lead paragraph) to stay precise.
    """
    lead = text[:1000]
    triples = []
    seen = set()

    for m in _ROLE_WAS.finditer(lead):
        name = m.group(1).strip()
        role = m.group(2).strip().lower()
        if any(kw in role for kw in ROLE_KEYWORDS) and len(name) > 2:
            key = (name.lower(), role)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "subject":    name,
                    "predicate":  "heldPosition",
                    "object":     role,
                    "source_url": source_url,
                })

    for m in _ROLE_APPOS.finditer(lead):
        name = m.group(1).strip()
        role = m.group(2).strip().lower()
        if any(kw in role for kw in ROLE_KEYWORDS) and len(name) > 2:
            key = (name.lower(), role)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "subject":    name,
                    "predicate":  "heldPosition",
                    "object":     role,
                    "source_url": source_url,
                })

    return triples


def extract_cooccurrence(doc, source_url: str) -> list[dict]:
    """
    Simpler fallback: two named entities in the same sentence → co_occurs_with.
    Only emit pairs where both are PERSON / ORG / GPE (skip DATE/LOC noise).
    """
    COOC_LABELS = {"PERSON", "ORG", "GPE", "NORP"}
    triples = []
    seen = set()
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in COOC_LABELS]
        if len(ents) < 2:
            continue
        for i, a in enumerate(ents):
            for b in ents[i+1:]:
                key = tuple(sorted([a.text.lower(), b.text.lower()]))
                if key in seen:
                    continue
                seen.add(key)
                triples.append({
                    "subject":    a.text.strip(),
                    "predicate":  "co_occurs_with",
                    "object":     b.text.strip(),
                    "source_url": source_url,
                })
    return triples


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print(f"Loading spaCy model: {MODEL}")
    nlp = spacy.load(MODEL)
    # Increase max length for large Wikipedia articles
    nlp.max_length = 2_000_000

    pages = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    print(f"Processing {len(pages)} pages...\n")

    all_entities = []
    all_triples  = []
    cleaned_docs = []

    for i, page in enumerate(pages):
        title = page["title"]
        url   = page["url"]
        print(f"[{i+1}/{len(pages)}] {title}")

        clean = clean_text(page["text"])
        doc   = nlp(clean)

        entities     = extract_entities(doc, url, title)
        dep_triples  = extract_relations_dep(doc, url)
        cooc_triples = extract_cooccurrence(doc, url)
        role_triples = extract_roles(clean, url)

        all_entities.extend(entities)
        all_triples.extend(dep_triples)
        all_triples.extend(cooc_triples)
        all_triples.extend(role_triples)

        cleaned_docs.append({"url": url, "title": title, "text": clean})

        print(f"  {len(entities)} entities, "
              f"{len(dep_triples)} dep-triples, "
              f"{len(cooc_triples)} co-occ triples, "
              f"{len(role_triples)} role triples")

    # ── Save entities ──────────────────────────────────────────────────────────
    Path(ENTITY_FILE).parent.mkdir(exist_ok=True)
    with open(ENTITY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "label", "source_title", "source_url"])
        writer.writeheader()
        writer.writerows(all_entities)
    print(f"\nEntities saved -> {ENTITY_FILE}  ({len(all_entities)} rows)")

    # ── Save triples ───────────────────────────────────────────────────────────
    with open(TRIPLES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "source_url"])
        writer.writeheader()
        writer.writerows(all_triples)
    print(f"Triples saved  -> {TRIPLES_FILE}  ({len(all_triples)} rows)")

    # ── Save cleaned texts ─────────────────────────────────────────────────────
    with open(CLEANED_FILE, "w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Cleaned texts  -> {CLEANED_FILE}")

    # ── Quick stats ────────────────────────────────────────────────────────────
    from collections import Counter
    label_counts = Counter(e["label"] for e in all_entities)
    print("\nEntity label breakdown:")
    for label, count in label_counts.most_common():
        print(f"  {label:10s} {count}")

    pred_counts = Counter(t["predicate"] for t in all_triples)
    print("\nTop predicates:")
    for pred, count in pred_counts.most_common(10):
        print(f"  {pred:25s} {count}")


if __name__ == "__main__":
    run()
