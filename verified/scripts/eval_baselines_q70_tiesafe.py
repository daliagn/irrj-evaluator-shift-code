#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import csv
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

SPLITS_DIR = Path("storyrank_clean/data/splits_q70")
OUT_DIR = Path("storyrank_clean/verified/reports/baselines_q70_tiesafe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5
MAX_REL = 5.0

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

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

def bm25_scores(rows):
    docs = [tokenize(r.get("story", "")) for r in rows]
    n_docs = len(docs)

    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    avgdl = sum(len(doc) for doc in docs) / max(1, n_docs)
    k1 = 1.2
    b = 0.75

    first = rows[0]
    query_text = (
        f"{first.get('age', '')} "
        f"{first.get('moral', '')} "
        f"{first.get('culture', '')} "
        "cultural relevance children story"
    )
    q_terms = tokenize(query_text)

    scores = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0

        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (tf[term] * (k1 + 1)) / denom

        scores.append(score)
    return scores

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def evaluate_method(grouped, method, seed):
    per_query = []

    for qid, rows in grouped.items():
        labels = [float(r["yA_expected"]) for r in rows]

        if method == "random":
            scores = [
                stable_jitter(seed, r["query_id"], r["story_id"], method)
                for r in rows
            ]
        elif method == "bm25":
            scores = bm25_scores(rows)
        elif method == "length":
            scores = [len(tokenize(r.get("story", ""))) for r in rows]
        elif method == "bscore_hard_yB":
            scores = [float(r["yB"]) for r in rows]
        elif method == "bscore_expected_yB":
            scores = [float(r["yB_expected"]) for r in rows]
        elif method == "A_oracle_sanity":
            scores = [float(r["yA_expected"]) for r in rows]
        else:
            raise ValueError(method)

        per_query.append({
            "query_id": qid,
            "method": method,
            "ndcg@5": ndcg_at_k(labels, scores, rows, seed, method, K),
            "err@5": err_at_k(labels, scores, rows, seed, method, K, MAX_REL),
            "spearman": spearman(labels, scores),
            "kendall_tau_b": kendall_tau_b(labels, scores),
            "pool_size": len(rows),
        })

    return per_query

methods = [
    "random",
    "bm25",
    "length",
    "bscore_hard_yB",
    "bscore_expected_yB",
    "A_oracle_sanity",
]

all_rows = []

for seed in range(10):
    test_path = SPLITS_DIR / f"seed_{seed}" / "test.jsonl"
    rows = [json.loads(line) for line in open(test_path, encoding="utf-8")]

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["query_id"]].append(r)

    print(f"Seed {seed}: test records={len(rows)}, queries={len(grouped)}")

    for method in methods:
        results = evaluate_method(grouped, method, seed)
        for r in results:
            r["seed"] = seed
        all_rows.extend(results)

per_query_path = OUT_DIR / "baseline_per_query_q70_tiesafe.csv"
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
    writer.writerows(all_rows)

summary = []

for method in methods:
    method_rows = [r for r in all_rows if r["method"] == method]
    by_seed = defaultdict(list)

    for r in method_rows:
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

summary_path = OUT_DIR / "baseline_summary_q70_tiesafe.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSaved:")
print(per_query_path)
print(summary_path)

print("\nSummary:")
for s in summary:
    print(
        f"{s['method']:20s} "
        f"nDCG@5={s['ndcg@5_mean']:.4f}±{s['ndcg@5_std']:.4f} "
        f"ERR@5={s['err@5_mean']:.4f}±{s['err@5_std']:.4f} "
        f"rho={s['spearman_mean']:.4f}±{s['spearman_std']:.4f} "
        f"tau={s['kendall_tau_b_mean']:.4f}±{s['kendall_tau_b_std']:.4f}"
    )
