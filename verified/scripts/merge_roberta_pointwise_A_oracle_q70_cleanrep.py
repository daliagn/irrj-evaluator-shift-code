#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("storyrank_clean/verified/reports")
OUT = BASE / "roberta_large_pointwise_A_oracle_q70_cleanrep"
OUT.mkdir(parents=True, exist_ok=True)

METHOD = "roberta_large_pointwise_yA_expected_oracle"
PER_QUERY_NAME = f"{METHOD}_per_query_q70.csv"
ITEM_NAME = f"{METHOD}_item_scores_q70.csv"
SUMMARY_NAME = f"{METHOD}_summary_q70.json"

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")

per_query_parts = []
item_parts = []
seed_summaries = []
source_dirs = {}

for seed in range(10):
    d = BASE / f"roberta_large_pointwise_A_oracle_q70_cleanrep_seed_{seed}"
    source_dirs[str(seed)] = str(d)

    pq_path = d / PER_QUERY_NAME
    item_path = d / ITEM_NAME
    summary_path = d / SUMMARY_NAME

    if not pq_path.exists():
        raise FileNotFoundError(pq_path)
    if not item_path.exists():
        raise FileNotFoundError(item_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    pq = pd.read_csv(pq_path)
    pq["seed"] = seed
    per_query_parts.append(pq)

    it = pd.read_csv(item_path)
    it["seed"] = seed
    item_parts.append(it)

    ndcg_col = pick_col(pq, ["ndcg@5", "ndcg_at_5", "ndcg5"])
    err_col = pick_col(pq, ["err@5", "err_at_5", "err5"])
    spearman_col = pick_col(pq, ["spearman", "spearman_rho", "rho"])
    kendall_col = pick_col(pq, ["kendall_tau_b", "kendall", "tau_b"])

    seed_summaries.append({
        "seed": seed,
        "method": METHOD,
        "ndcg@5": float(pq[ndcg_col].mean()),
        "err@5": float(pq[err_col].mean()),
        "spearman": float(pq[spearman_col].mean()),
        "kendall_tau_b": float(pq[kendall_col].mean())
    })

per_query = pd.concat(per_query_parts, ignore_index=True)
item_scores = pd.concat(item_parts, ignore_index=True)

per_query.to_csv(OUT / PER_QUERY_NAME, index=False)
item_scores.to_csv(OUT / ITEM_NAME, index=False)

summary = {
    "method": METHOD,
    "training_label": "yA_expected",
    "validation_label": "yA_expected",
    "evaluation_label": "yA_expected",
    "note": "Leaky A-to-A oracle / upper-bound sanity check. Not part of the main strict Judge-B-to-Judge-A protocol.",
    "source_dirs": source_dirs,
    "n_seeds": 10,
    "n_query_seed_pairs": int(len(per_query)),
    "seed_summaries": seed_summaries,
}

for metric in ["ndcg@5", "err@5", "spearman", "kendall_tau_b"]:
    vals = np.array([s[metric] for s in seed_summaries], dtype=float)
    key = metric.replace("@", "").replace("_tau_b", "_tau_b")
    if metric == "ndcg@5":
        summary["ndcg@5_mean"] = float(vals.mean())
        summary["ndcg@5_std"] = float(vals.std(ddof=0))
    elif metric == "err@5":
        summary["err@5_mean"] = float(vals.mean())
        summary["err@5_std"] = float(vals.std(ddof=0))
    elif metric == "spearman":
        summary["spearman_mean"] = float(vals.mean())
        summary["spearman_std"] = float(vals.std(ddof=0))
    elif metric == "kendall_tau_b":
        summary["kendall_tau_b_mean"] = float(vals.mean())
        summary["kendall_tau_b_std"] = float(vals.std(ddof=0))

with open(OUT / SUMMARY_NAME, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved:")
print(OUT / PER_QUERY_NAME)
print(OUT / ITEM_NAME)
print(OUT / SUMMARY_NAME)
print()
print(
    f"{METHOD} "
    f"nDCG@5={summary['ndcg@5_mean']:.4f}±{summary['ndcg@5_std']:.4f} "
    f"ERR@5={summary['err@5_mean']:.4f}±{summary['err@5_std']:.4f} "
    f"rho={summary['spearman_mean']:.4f}±{summary['spearman_std']:.4f} "
    f"tau={summary['kendall_tau_b_mean']:.4f}±{summary['kendall_tau_b_std']:.4f}"
)
