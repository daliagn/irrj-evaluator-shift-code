#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

SPLITS_DIR = Path("storyrank_clean/data/splits_q70")
OUT_DIR = Path("storyrank_clean/verified/reports/tfidf_ridge_label_ablation_q70")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5
MAX_REL = 5.0

def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def make_text(row):
    return (
        f"Age: {row.get('age', '')}\n"
        f"Moral: {row.get('moral', '')}\n"
        f"Culture: {row.get('culture', '')}\n"
        f"Story:\n{row.get('story', '')}"
    )

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

def evaluate(test_rows, preds, seed, method):
    grouped = defaultdict(list)

    for row, pred in zip(test_rows, preds):
        grouped[row["query_id"]].append((row, float(pred)))

    per_query = []

    for qid, items in grouped.items():
        rows = [x[0] for x in items]
        scores = [x[1] for x in items]
        labels = [float(r["yA_expected"]) for r in rows]

        per_query.append({
            "seed": seed,
            "query_id": qid,
            "method": method,
            "pool_size": len(rows),
            "ndcg@5": ndcg_at_k(labels, scores, rows, seed, method, K),
            "err@5": err_at_k(labels, scores, rows, seed, method, K, MAX_REL),
            "spearman": spearman(labels, scores),
            "kendall_tau_b": kendall_tau_b(labels, scores),
        })

    return per_query

def summarise(all_results):
    summary = []

    methods = sorted(set(r["method"] for r in all_results))

    for method in methods:
        rows_m = [r for r in all_results if r["method"] == method]
        by_seed = defaultdict(list)

        for r in rows_m:
            by_seed[r["seed"]].append(r)

        seed_summaries = []

        for seed, rows in sorted(by_seed.items()):
            seed_summaries.append({
                "seed": seed,
                "method": method,
                "ndcg@5": mean([r["ndcg@5"] for r in rows]),
                "err@5": mean([r["err@5"] for r in rows]),
                "spearman": mean([r["spearman"] for r in rows]),
                "kendall_tau_b": mean([r["kendall_tau_b"] for r in rows]),
            })

        summary.append({
            "method": method,
            "training_label": "yB" if "hard" in method else "yB_expected",
            "evaluation_label": "yA_expected",
            "ndcg@5_mean": mean([s["ndcg@5"] for s in seed_summaries]),
            "ndcg@5_std": std([s["ndcg@5"] for s in seed_summaries]),
            "err@5_mean": mean([s["err@5"] for s in seed_summaries]),
            "err@5_std": std([s["err@5"] for s in seed_summaries]),
            "spearman_mean": mean([s["spearman"] for s in seed_summaries]),
            "spearman_std": std([s["spearman"] for s in seed_summaries]),
            "kendall_tau_b_mean": mean([s["kendall_tau_b"] for s in seed_summaries]),
            "kendall_tau_b_std": std([s["kendall_tau_b"] for s in seed_summaries]),
            "seed_summaries": seed_summaries,
        })

    return summary

all_results = []

for seed in range(10):
    seed_dir = SPLITS_DIR / f"seed_{seed}"

    train_rows = load_jsonl(seed_dir / "train.jsonl")
    test_rows = load_jsonl(seed_dir / "test.jsonl")

    print(f"\nSeed {seed}")
    print("train records:", len(train_rows))
    print("test records:", len(test_rows))

    train_texts = [make_text(r) for r in train_rows]
    test_texts = [make_text(r) for r in test_rows]

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

    experiments = [
        ("tfidf_ridge_yB_hard", [float(r["yB"]) for r in train_rows]),
        ("tfidf_ridge_yB_expected", [float(r["yB_expected"]) for r in train_rows]),
    ]

    for method, y_train in experiments:
        print("Training:", method)

        model = Ridge(alpha=1.0, solver="lsqr")
        model.fit(X_train, y_train)

        print("Predicting:", method)
        preds = model.predict(X_test)

        seed_results = evaluate(test_rows, preds, seed, method)
        all_results.extend(seed_results)

        print(
            f"{method}: "
            f"nDCG@5={mean([r['ndcg@5'] for r in seed_results]):.4f} "
            f"ERR@5={mean([r['err@5'] for r in seed_results]):.4f} "
            f"rho={mean([r['spearman'] for r in seed_results]):.4f} "
            f"tau={mean([r['kendall_tau_b'] for r in seed_results]):.4f}"
        )

per_query_path = OUT_DIR / "tfidf_ridge_label_ablation_per_query_q70.csv"
with open(per_query_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "query_id",
            "method",
            "pool_size",
            "ndcg@5",
            "err@5",
            "spearman",
            "kendall_tau_b",
        ],
    )
    writer.writeheader()
    writer.writerows(all_results)

summary = summarise(all_results)

summary_path = OUT_DIR / "tfidf_ridge_label_ablation_summary_q70.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSaved:")
print(per_query_path)
print(summary_path)

print("\nSummary:")
for s in summary:
    print(
        f"{s['method']:24s} "
        f"nDCG@5={s['ndcg@5_mean']:.4f}±{s['ndcg@5_std']:.4f} "
        f"ERR@5={s['err@5_mean']:.4f}±{s['err@5_std']:.4f} "
        f"rho={s['spearman_mean']:.4f}±{s['spearman_std']:.4f} "
        f"tau={s['kendall_tau_b_mean']:.4f}±{s['kendall_tau_b_std']:.4f}"
    )
