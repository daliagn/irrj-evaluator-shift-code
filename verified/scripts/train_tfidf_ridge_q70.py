#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import csv
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

K = 5
MAX_REL = 5.0

def gain(rel):
    return (2 ** float(rel)) - 1.0

def stable_jitter(seed, query_id, story_id, method):
    text = f"{seed}|{query_id}|{story_id}|{method}"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)

def ranked_indices(scores, rows, seed, method):
    return sorted(
        range(len(scores)),
        key=lambda i: (
            float(scores[i]),
            stable_jitter(seed, rows[i]["query_id"], rows[i]["story_id"], method)
        ),
        reverse=True
    )

def dcg_at_k(rels, k=5):
    return sum(gain(rel) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(labels, scores, rows, seed, method, k=5):
    order = ranked_indices(scores, rows, seed, method)
    ranked = [labels[i] for i in order]
    ideal = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg_at_k(ranked, k) / idcg

def err_at_k(labels, scores, rows, seed, method, k=5, max_rel=5.0):
    order = ranked_indices(scores, rows, seed, method)
    err = 0.0
    p_continue = 1.0
    denom = 2 ** max_rel

    for rank, idx in enumerate(order[:k], start=1):
        rel = float(labels[idx])
        R = ((2 ** rel) - 1.0) / denom
        R = max(0.0, min(1.0, R))
        err += p_continue * R / rank
        p_continue *= (1.0 - R)

    return err

def average_ranks(values):
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0

    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1

        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank

        i = j + 1

    return ranks

def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return None

    mx = sum(xs) / n
    my = sum(ys) / n

    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)

    if vx == 0 or vy == 0:
        return None

    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))

def kendall_tau_b(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])

            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt(
        (concordant + discordant + ties_x) *
        (concordant + discordant + ties_y)
    )

    return None if denom == 0 else (concordant - discordant) / denom

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def make_text(row):
    return (
        f"Age: {row.get('age', '')}\n"
        f"Moral: {row.get('moral', '')}\n"
        f"Culture: {row.get('culture', '')}\n"
        f"Story:\n{row.get('story', '')}"
    )

def evaluate_rows(rows, preds, seed, method):
    grouped = defaultdict(list)

    for row, pred in zip(rows, preds):
        grouped[row["query_id"]].append((row, float(pred)))

    per_query = []

    for qid, items in grouped.items():
        q_rows = [x[0] for x in items]
        scores = [x[1] for x in items]
        labels = [float(r["yA_expected"]) for r in q_rows]

        per_query.append({
            "query_id": qid,
            "method": method,
            "pool_size": len(q_rows),
            "ndcg@5": ndcg_at_k(labels, scores, q_rows, seed, method, K),
            "err@5": err_at_k(labels, scores, q_rows, seed, method, K, MAX_REL),
            "spearman": spearman(labels, scores),
            "kendall_tau_b": kendall_tau_b(labels, scores),
        })

    return per_query

def parse_seeds(seed_arg):
    if seed_arg == "all":
        return list(range(10))
    return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="storyrank_clean/data/splits_q70")
    ap.add_argument("--out_dir", default="storyrank_clean/verified/reports/tfidf_ridge_q70")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = "tfidf_ridge_yB_expected"
    all_results = []

    for seed in parse_seeds(args.seeds):
        seed_dir = Path(args.splits_dir) / f"seed_{seed}"

        train_rows = load_jsonl(seed_dir / "train.jsonl")
        test_rows = load_jsonl(seed_dir / "test.jsonl")

        print(f"\nSeed {seed}")
        print("train records:", len(train_rows))
        print("test records:", len(test_rows))

        train_texts = [make_text(r) for r in train_rows]
        test_texts = [make_text(r) for r in test_rows]

        y_train = [float(r["yB_expected"]) for r in train_rows]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=100000,
            sublinear_tf=True,
            stop_words="english"
        )

        print("Vectorizing...")
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        print("X_train:", X_train.shape)
        print("Training Ridge...")
        model = Ridge(alpha=args.alpha, solver="lsqr")
        model.fit(X_train, y_train)

        print("Predicting...")
        preds = model.predict(X_test)

        seed_results = evaluate_rows(test_rows, preds, seed, method)

        for r in seed_results:
            r["seed"] = seed
            r["alpha"] = args.alpha

        all_results.extend(seed_results)

        print(
            f"seed {seed}: "
            f"nDCG@5={mean([r['ndcg@5'] for r in seed_results]):.4f} "
            f"ERR@5={mean([r['err@5'] for r in seed_results]):.4f} "
            f"rho={mean([r['spearman'] for r in seed_results]):.4f} "
            f"tau={mean([r['kendall_tau_b'] for r in seed_results]):.4f}"
        )

    per_query_path = out_dir / f"{method}_per_query_q70.csv"
    with open(per_query_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "query_id",
                "method",
                "alpha",
                "pool_size",
                "ndcg@5",
                "err@5",
                "spearman",
                "kendall_tau_b",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    by_seed = defaultdict(list)
    for r in all_results:
        by_seed[r["seed"]].append(r)

    seed_summaries = []

    for seed, rows in sorted(by_seed.items()):
        seed_summaries.append({
            "seed": seed,
            "method": method,
            "alpha": args.alpha,
            "ndcg@5": mean([r["ndcg@5"] for r in rows]),
            "err@5": mean([r["err@5"] for r in rows]),
            "spearman": mean([r["spearman"] for r in rows]),
            "kendall_tau_b": mean([r["kendall_tau_b"] for r in rows]),
        })

    summary = {
        "method": method,
        "training_label": "yB_expected",
        "evaluation_label": "yA_expected",
        "alpha": args.alpha,
        "ndcg@5_mean": mean([s["ndcg@5"] for s in seed_summaries]),
        "ndcg@5_std": std([s["ndcg@5"] for s in seed_summaries]),
        "err@5_mean": mean([s["err@5"] for s in seed_summaries]),
        "err@5_std": std([s["err@5"] for s in seed_summaries]),
        "spearman_mean": mean([s["spearman"] for s in seed_summaries]),
        "spearman_std": std([s["spearman"] for s in seed_summaries]),
        "kendall_tau_b_mean": mean([s["kendall_tau_b"] for s in seed_summaries]),
        "kendall_tau_b_std": std([s["kendall_tau_b"] for s in seed_summaries]),
        "seed_summaries": seed_summaries,
    }

    summary_path = out_dir / f"{method}_summary_q70.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(per_query_path)
    print(summary_path)

    print("\nSummary:")
    print(
        f"{method:24s} "
        f"nDCG@5={summary['ndcg@5_mean']:.4f}±{summary['ndcg@5_std']:.4f} "
        f"ERR@5={summary['err@5_mean']:.4f}±{summary['err@5_std']:.4f} "
        f"rho={summary['spearman_mean']:.4f}±{summary['spearman_std']:.4f} "
        f"tau={summary['kendall_tau_b_mean']:.4f}±{summary['kendall_tau_b_std']:.4f}"
    )

if __name__ == "__main__":
    main()
