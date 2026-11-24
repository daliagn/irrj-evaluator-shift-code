#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

METHOD = "roberta_large_pointwise_yB_expected"
ROOT = Path("storyrank_clean/verified/reports")
OUT = ROOT / "roberta_large_pointwise_q70"
OUT.mkdir(parents=True, exist_ok=True)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def to_float(x):
    if x in ["", "None", None]:
        return None
    return float(x)

all_per_query = []
all_scores = []
all_history = {}

for seed in range(10):
    d = ROOT / f"roberta_large_pointwise_q70_seed_{seed}"

    pq = d / f"{METHOD}_per_query_q70.csv"
    sc = d / f"{METHOD}_item_scores_q70.csv"
    sm = d / f"{METHOD}_summary_q70.json"

    if not pq.exists():
        raise FileNotFoundError(f"Missing per-query file for seed {seed}: {pq}")
    if not sc.exists():
        raise FileNotFoundError(f"Missing item-score file for seed {seed}: {sc}")
    if not sm.exists():
        raise FileNotFoundError(f"Missing summary file for seed {seed}: {sm}")

    with open(pq, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if len(rows) != 70:
            raise ValueError(f"Seed {seed}: expected 70 test-query rows, got {len(rows)}")
        all_per_query.extend(rows)

    with open(sc, encoding="utf-8") as f:
        all_scores.extend(list(csv.DictReader(f)))

    with open(sm, encoding="utf-8") as f:
        s = json.load(f)
        all_history[str(seed)] = s.get("training_history", {}).get(str(seed), [])

if len(all_per_query) != 700:
    raise ValueError(f"Expected 700 total query-seed rows, got {len(all_per_query)}")

per_query_out = OUT / f"{METHOD}_per_query_q70.csv"
with open(per_query_out, "w", encoding="utf-8", newline="") as f:
    fieldnames = list(all_per_query[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_per_query)

scores_out = OUT / f"{METHOD}_item_scores_q70.csv"
with open(scores_out, "w", encoding="utf-8", newline="") as f:
    fieldnames = list(all_scores[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_scores)

by_seed = defaultdict(list)
for r in all_per_query:
    by_seed[int(r["seed"])].append(r)

seed_summaries = []
for seed, rows in sorted(by_seed.items()):
    seed_summaries.append({
        "seed": seed,
        "method": METHOD,
        "ndcg@5": mean([to_float(r["ndcg@5"]) for r in rows]),
        "err@5": mean([to_float(r["err@5"]) for r in rows]),
        "spearman": mean([to_float(r["spearman"]) for r in rows]),
        "kendall_tau_b": mean([to_float(r["kendall_tau_b"]) for r in rows]),
    })

summary = {
    "method": METHOD,
    "model_path": "models/roberta-large",
    "training_label": "yB_expected",
    "validation_label": "yB_expected",
    "test_evaluation_label": "yA_expected",
    "selection": "best epoch by validation nDCG@5 using yB_expected",
    "n_query_seed_rows": len(all_per_query),
    "n_seeds": 10,
    "ndcg@5_mean": mean([s["ndcg@5"] for s in seed_summaries]),
    "ndcg@5_std": std([s["ndcg@5"] for s in seed_summaries]),
    "err@5_mean": mean([s["err@5"] for s in seed_summaries]),
    "err@5_std": std([s["err@5"] for s in seed_summaries]),
    "spearman_mean": mean([s["spearman"] for s in seed_summaries]),
    "spearman_std": std([s["spearman"] for s in seed_summaries]),
    "kendall_tau_b_mean": mean([s["kendall_tau_b"] for s in seed_summaries]),
    "kendall_tau_b_std": std([s["kendall_tau_b"] for s in seed_summaries]),
    "seed_summaries": seed_summaries,
    "training_history": all_history,
}

summary_out = OUT / f"{METHOD}_summary_q70.json"
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Saved:")
print(per_query_out)
print(scores_out)
print(summary_out)

print("\nSummary:")
print(
    f"{METHOD} "
    f"nDCG@5={summary['ndcg@5_mean']:.4f}±{summary['ndcg@5_std']:.4f} "
    f"ERR@5={summary['err@5_mean']:.4f}±{summary['err@5_std']:.4f} "
    f"rho={summary['spearman_mean']:.4f}±{summary['spearman_std']:.4f} "
    f"tau={summary['kendall_tau_b_mean']:.4f}±{summary['kendall_tau_b_std']:.4f}"
)
