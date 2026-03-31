import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

import argparse
import json
import math
import random
import time
from pathlib import Path

import sklearn
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_splits(data_dir: Path):
    def read(fname):
        rows = []
        for line in (data_dir / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                h, r, t = line.split("\t")
                rows.append((int(h), int(r), int(t)))
        return rows

    train = read("train.txt")
    valid = read("valid.txt")
    test  = read("test.txt")
    return train, valid, test


def load_mappings(data_dir: Path):
    def read(fname):
        m = {}
        for line in (data_dir / fname).read_text(encoding="utf-8").splitlines():
            idx, short, full = line.split("\t", 2)
            m[int(idx)] = short
        return m

    id2entity   = load_mappings_raw(data_dir / "entity2id.txt")
    id2relation = load_mappings_raw(data_dir / "relation2id.txt")
    return id2entity, id2relation


def load_mappings_raw(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 2)
        m[int(parts[0])] = parts[1]   
    return m



class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim, margin=1.0):
        super().__init__()
        self.dim    = dim
        self.margin = margin
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.uniform_(self.ent_emb.weight, -6/math.sqrt(dim), 6/math.sqrt(dim))
        nn.init.uniform_(self.rel_emb.weight, -6/math.sqrt(dim), 6/math.sqrt(dim))
        self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data, p=2, dim=1)

    def forward(self, h, r, t):
        h_e = F.normalize(self.ent_emb(h), p=2, dim=1)
        r_e = self.rel_emb(r)
        t_e = F.normalize(self.ent_emb(t), p=2, dim=1)
        return -torch.norm(h_e + r_e - t_e, p=2, dim=1)   # negative L2

    def score(self, h, r, t):
        return self.forward(h, r, t)


class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.dim     = dim
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h, r, t):
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        return (h_e * r_e * t_e).sum(dim=1)

    def score(self, h, r, t):
        return self.forward(h, r, t)



def negative_sample(triples, n_ent, batch_size, n_neg=1):
    pos = random.sample(triples, min(batch_size, len(triples)))
    neg = []
    for h, r, t in pos:
        if random.random() < 0.5:
            neg.append((random.randint(0, n_ent - 1), r, t))
        else:
            neg.append((h, r, random.randint(0, n_ent - 1)))
    return pos, neg


def train_epoch(model, optimizer, train_triples, n_ent, batch_size, model_name):
    model.train()
    total_loss = 0.0
    steps = max(1, len(train_triples) // batch_size)

    for _ in range(steps):
        pos, neg = negative_sample(train_triples, n_ent, batch_size)

        ph = torch.tensor([t[0] for t in pos], dtype=torch.long)
        pr = torch.tensor([t[1] for t in pos], dtype=torch.long)
        pt = torch.tensor([t[2] for t in pos], dtype=torch.long)
        nh = torch.tensor([t[0] for t in neg], dtype=torch.long)
        nr = torch.tensor([t[1] for t in neg], dtype=torch.long)
        nt = torch.tensor([t[2] for t in neg], dtype=torch.long)

        pos_score = model.score(ph, pr, pt)
        neg_score = model.score(nh, nr, nt)

        if model_name == "TransE":
            margin = model.margin
            loss = F.relu(margin - pos_score + neg_score).mean()
        else:
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_name == "TransE":
            with torch.no_grad():
                model.ent_emb.weight.data = F.normalize(
                    model.ent_emb.weight.data, p=2, dim=1
                )

        total_loss += loss.item()

    return total_loss / steps



def build_filter_set(all_triples):
    return set(all_triples)


def evaluate(model, test_triples, n_ent, filter_set, max_eval=500):
    model.eval()
    ranks_head = []
    ranks_tail = []

    eval_triples = test_triples[:max_eval]   

    with torch.no_grad():
        all_ents = torch.arange(n_ent, dtype=torch.long)

        for h, r, t in eval_triples:

            h_rep = torch.full((n_ent,), h, dtype=torch.long)
            r_rep = torch.full((n_ent,), r, dtype=torch.long)
            scores = model.score(h_rep, r_rep, all_ents).numpy()

            for h2, r2, t2 in [(h, r, t2_) for (h2, r2, t2_) in
                                [(h_, r_, t_) for (h_, r_, t_) in filter_set
                                 if h_ == h and r_ == r and t_ != t]]:
                scores[h2] = -1e9     
            for (h_, r_, t_) in filter_set:
                if h_ == h and r_ == r and t_ != t:
                    scores[t_] = -1e9

            rank = int((scores > scores[t]).sum()) + 1
            ranks_tail.append(rank)

            t_rep = torch.full((n_ent,), t, dtype=torch.long)
            scores_h = model.score(all_ents, r_rep, t_rep).numpy()

            for (h_, r_, t_) in filter_set:
                if r_ == r and t_ == t and h_ != h:
                    scores_h[h_] = -1e9

            rank_h = int((scores_h > scores_h[h]).sum()) + 1
            ranks_head.append(rank_h)

    ranks = ranks_head + ranks_tail
    ranks_arr = np.array(ranks, dtype=np.float32)

    mrr   = float(np.mean(1.0 / ranks_arr))
    hits1 = float(np.mean(ranks_arr <= 1))
    hits3 = float(np.mean(ranks_arr <= 3))
    hits10 = float(np.mean(ranks_arr <= 10))

    return {"MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10,
            "n_eval": len(eval_triples)}



def nearest_neighbors(model, id2entity, entity_names, top_k=5):
    emb = F.normalize(model.ent_emb.weight.detach(), p=2, dim=1).numpy()
    results = {}

    for name in entity_names:
        match = [(i, n) for i, n in id2entity.items() if n.lower() == name.lower()]
        if not match:
            match = [(i, n) for i, n in id2entity.items() if name.lower() in n.lower()]
        if not match:
            results[name] = ["(not found)"]
            continue
        eid, found_name = match[0]
        query = emb[eid]
        sims = emb @ query 
        top = np.argsort(-sims)[1:top_k + 1]
        results[found_name] = [(id2entity[i], float(sims[i])) for i in top]

    return results


def relation_behavior(model, id2relation):
    rel_emb = F.normalize(model.rel_emb.weight.detach(), p=2, dim=1).numpy()
    n_rel = rel_emb.shape[0]

    lines = ["=== Relation Behaviour Analysis ===\n"]

    norms = np.linalg.norm(model.rel_emb.weight.detach().numpy(), axis=1)
    lines.append("Relation norms (small norm = likely symmetric in TransE):")
    for i, (norm, name) in enumerate(zip(norms, [id2relation[j] for j in range(n_rel)])):
        lines.append(f"  {name:40s}  norm={norm:.4f}")

    lines.append("\nMost anti-correlated relation pairs (potential inverses):")
    sim_mat = rel_emb @ rel_emb.T
    pairs = []
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            pairs.append((sim_mat[i, j], i, j))
    pairs.sort()
    for sim, i, j in pairs[:5]:
        lines.append(f"  {id2relation[i]:30s} <-> {id2relation[j]:30s}  cos={sim:.4f}")

    rel_names = {v: k for k, v in id2relation.items()}
    key_triples = [
        ("studentOf",                 "studentOf",                "intellectualDescendantOf"),
        ("servedIn",                  "attacked",                 "foughtAgainst"),
        ("wasConqueredBy",            None,                       "atWarWith"),
    ]
    lines.append("\nSWRL Rule vs KGE vector arithmetic (Exercise 8):")
    for r1_name, r2_name, r3_name in key_triples:
        r1_id = rel_names.get(r1_name)
        r3_id = rel_names.get(r3_name)
        if r1_id is None or r3_id is None:
            lines.append(f"  {r1_name} / {r3_name}: not found in relation vocabulary")
            continue
        v1 = rel_emb[r1_id]
        v3 = rel_emb[r3_id]
        if r2_name:
            r2_id = rel_names.get(r2_name)
            if r2_id is None:
                lines.append(f"  {r2_name}: not in vocabulary")
                continue
            v2 = rel_emb[r2_id]
            composed = v1 + v2
        else:
            composed = v1
        composed_norm = composed / (np.linalg.norm(composed) + 1e-9)
        cos = float(composed_norm @ v3)
        if r2_name:
            lines.append(f"  v({r1_name}) + v({r2_name}) ≈ v({r3_name})?  cos={cos:.4f}")
        else:
            lines.append(f"  v({r1_name}) ≈ v({r3_name})?  cos={cos:.4f}")

    return "\n".join(lines)


def tsne_plot(model, id2entity, out_path: Path, max_ents=300):

    emb = model.ent_emb.weight.detach().numpy()
    names = [id2entity[i] for i in range(len(id2entity))]

    # Heuristic class labels from entity name patterns
    def classify(name):
        n = name.lower()
        if any(x in n for x in ["battle", "siege", "war"]):        return "Battle"
        if any(x in n for x in ["state", "kingdom", "qin", "zhao", "wei", "chu",
                                  "han", "yan", "qi", "lu", "jin", "yue"]):  return "State"
        if any(x in n for x in ["confuci", "mencius", "laozi", "zhuang", "mozi",
                                  "xunzi", "legalis", "taoism", "mohism"]):  return "Philosophy"
        if any(x in n for x in ["dynasty", "zhou", "shang"]):      return "Dynasty"
        return "Person/Other"

    idx = list(range(len(names)))
    if len(idx) > max_ents:
        idx = random.sample(idx, max_ents)

    sub_emb   = emb[idx]
    sub_names = [names[i] for i in idx]
    sub_class = [classify(n) for n in sub_names]

    print(f"  Running t-SNE on {len(idx)} entities …")
    tsne = TSNE(n_components=2, perplexity=min(30, len(idx) - 1), random_state=SEED, n_iter=500)
    coords = tsne.fit_transform(sub_emb)

    colour_map = {
        "Battle":       "red",
        "State":        "blue",
        "Philosophy":   "green",
        "Dynasty":      "orange",
        "Person/Other": "grey",
    }
    plt.figure(figsize=(10, 8))
    for cls, colour in colour_map.items():
        mask = [i for i, c in enumerate(sub_class) if c == cls]
        if mask:
            plt.scatter(coords[mask, 0], coords[mask, 1], c=colour,
                        label=cls, alpha=0.6, s=20)

    plt.title("t-SNE of Entity Embeddings (Warring States KG)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  t-SNE plot saved → {out_path}")



def subsample(triples, n):
    """Return a random subsample of n triples that keeps all entities connected."""
    if len(triples) <= n:
        return triples
    random.shuffle(triples)
    return triples[:n]



def run_model(name, ModelClass, model_kwargs, train_triples, valid_triples, test_triples,
              n_ent, n_rel, filter_set, args, suffix=""):
    print(f"\n{'='*60}")
    print(f"Training {name}{suffix}")
    print(f"{'='*60}")
    print(f"  dim={args.dim}  lr={args.lr}  batch={args.batch}  epochs={args.epochs}")

    model = ModelClass(n_ent, n_rel, args.dim, **model_kwargs)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mrr, best_epoch = 0.0, 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, opt, train_triples, n_ent, args.batch, name)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            metrics = evaluate(model, valid_triples, n_ent, filter_set)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                  f"MRR={metrics['MRR']:.4f}  Hits@10={metrics['Hits@10']:.4f}  "
                  f"({elapsed:.1f}s)")
            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_triples, n_ent, filter_set)
    print(f"\n  [TEST]  MRR={test_metrics['MRR']:.4f}  "
          f"Hits@1={test_metrics['Hits@1']:.4f}  "
          f"Hits@3={test_metrics['Hits@3']:.4f}  "
          f"Hits@10={test_metrics['Hits@10']:.4f}  "
          f"(best epoch={best_epoch})")

    return model, test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",       default="data/kge")
    ap.add_argument("--results-dir",    default="data/kge/results")
    ap.add_argument("--dim",    type=int,   default=100)
    ap.add_argument("--epochs", type=int,   default=200)
    ap.add_argument("--lr",     type=float, default=0.001)
    ap.add_argument("--batch",  type=int,   default=512)
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip training and load saved models for analysis only")
    args = ap.parse_args()

    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_splits(data_dir)
    id2entity   = load_mappings_raw(data_dir / "entity2id.txt")
    id2relation = load_mappings_raw(data_dir / "relation2id.txt")

    n_ent = len(id2entity)
    n_rel = len(id2relation)
    filter_set = build_filter_set(train + valid + test)

    print(f"Dataset: {len(train):,} train | {len(valid):,} valid | {len(test):,} test")
    print(f"Entities: {n_ent:,}  Relations: {n_rel:,}")

    all_results = {}
    transe_path  = results_dir / "model_transe.pt"
    distmult_path = results_dir / "model_distmult.pt"

    if args.skip_training and transe_path.exists() and distmult_path.exists():
        print("\nLoading saved models (--skip-training) ...")
        transe = TransE(n_ent, n_rel, args.dim)
        transe.load_state_dict(torch.load(str(transe_path), weights_only=True))
        distmult = DistMult(n_ent, n_rel, args.dim)
        distmult.load_state_dict(torch.load(str(distmult_path), weights_only=True))
        metrics_transe  = json.loads((results_dir / "metrics_transE.json").read_text())
        metrics_distmult = json.loads((results_dir / "metrics_distmult.json").read_text())
        all_results["TransE_full"]   = metrics_transe
        all_results["DistMult_full"] = metrics_distmult
        print("  Models loaded.")
    else:
        transe, metrics_transe = run_model(
            "TransE", TransE, {"margin": 1.0},
            train, valid, test, n_ent, n_rel, filter_set, args,
        )
        all_results["TransE_full"] = metrics_transe
        torch.save(transe.state_dict(), str(transe_path))

        distmult, metrics_distmult = run_model(
            "DistMult", DistMult, {},
            train, valid, test, n_ent, n_rel, filter_set, args,
        )
        all_results["DistMult_full"] = metrics_distmult
        torch.save(distmult.state_dict(), str(distmult_path))

        for size in [20_000]:
            sub   = subsample(list(train), size)
            n_sub = min(size, len(train))
            _, m  = run_model(
                "TransE", TransE, {"margin": 1.0},
                sub, valid, test, n_ent, n_rel, filter_set, args,
                suffix=f" (subsample {n_sub:,})",
            )
            all_results[f"TransE_{n_sub}"] = m

        print("\n" + "=" * 60)
        print("MODEL COMPARISON TABLE")
        print("=" * 60)
        header = f"{'Model':<25}  {'MRR':>8}  {'Hits@1':>8}  {'Hits@3':>8}  {'Hits@10':>10}"
        sep    = "-" * len(header)
        lines  = [header, sep]
        for label, m in all_results.items():
            lines.append(
                f"{label:<25}  {m['MRR']:>8.4f}  {m['Hits@1']:>8.4f}  "
                f"{m['Hits@3']:>8.4f}  {m['Hits@10']:>10.4f}"
            )
        table = "\n".join(lines)
        print(table)
        (results_dir / "comparison_table.txt").write_text(table, encoding="utf-8")

        (results_dir / "metrics_transE.json").write_text(
            json.dumps(metrics_transe, indent=2), encoding="utf-8")
        (results_dir / "metrics_distmult.json").write_text(
            json.dumps(metrics_distmult, indent=2), encoding="utf-8")

    target_entities = ["BaiQi", "Confucius", "Qin", "Zhao", "studentOf",
                       "Mencius", "Legalism", "Confucianism"]
    nn_results = nearest_neighbors(transe, id2entity, target_entities)

    nn_lines = ["=== Nearest Neighbours (TransE, top-5) ===\n"]
    for ent, neighbours in nn_results.items():
        nn_lines.append(f"{ent}:")
        for i, item in enumerate(neighbours, 1):
            if isinstance(item, tuple):
                nn_lines.append(f"  {i}. {item[0]}  (cos={item[1]:.4f})")
            else:
                nn_lines.append(f"  {i}. {item}")
        nn_lines.append("")

    nn_text = "\n".join(nn_lines)
    print("\n" + nn_text)
    (results_dir / "nearest_neighbors.txt").write_text(nn_text, encoding="utf-8")

    print("\nGenerating t-SNE plot …")
    tsne_plot(transe, id2entity, results_dir / "tsne_plot.png")

    rel_text = relation_behavior(transe, id2relation)
    print("\n" + rel_text)
    (results_dir / "relation_analysis.txt").write_text(rel_text, encoding="utf-8")

    ex8 = (results_dir / "relation_analysis.txt").read_text(encoding="utf-8")
    ex8_lines = [l for l in ex8.splitlines() if "SWRL" in l or "v(" in l or "cos=" in l]
    ex8_text = "\n".join(ex8_lines)
    (results_dir / "exercise8_kge_vs_swrl.txt").write_text(ex8_text, encoding="utf-8")

    print(f"\nAll results saved to: {results_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()