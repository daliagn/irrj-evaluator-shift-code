#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
from pathlib import Path

ROOT = Path("storyrank_clean/verified/reports/roberta_large_classifier_weighted_gamma_q70")
OUT = ROOT / "merged"
OUT.mkdir(parents=True, exist_ok=True)

TAUS = [2, 3, 4]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

all_seed_rows = []
all_query_rows = []
all_score_rows = []

for tau in TAUS:
    for seed in range(10):
        method = f"roberta_large_classifier_weighted_gamma_tau{tau}"
        d = ROOT / f"tau_{tau}" / f"seed_{seed}"

        sm = d / f"{method}_summary_seed_{seed}.json"
        pq = d / f"{method}_ranking_per_query_seed_{seed}.csv"
        sc = d / f"{method}_item_scores_seed_{seed}.csv"

        if not sm.exists():
            raise FileNotFoundError(f"Missing summary: {sm}")
        if not pq.exists():
            raise FileNotFoundError(f"Missing per-query: {pq}")
        if not sc.exists():
            raise FileNotFoundError(f"Missing item scores: {sc}")

        summary = json.load(open(sm, encoding="utf-8"))

        test_sel = summary["test_selected_gamma_metrics"]
        test_def = summary["test_default_gamma_0_5_metrics"]
        ap = summary["always_positive"]
        rank = summary["ranking_as_classifier_probability"]

        all_seed_rows.append({
            "tau": tau,
            "seed": seed,
            "method": method,
            "pos_weight": summary["pos_weight"],
            "train_positive_rate": summary["train_positive_rate"],
            "selected_gamma": summary["selected_gamma"],

            "positive_rate": test_sel["positive_rate"],
            "predicted_positive_rate": test_sel["predicted_positive_rate"],
            "precision": test_sel["precision"],
            "recall": test_sel["recall"],
            "f1": test_sel["f1"],
            "accuracy": test_sel["accuracy"],
            "auroc": test_sel["auroc"],

            "default_0_5_predicted_positive_rate": test_def["predicted_positive_rate"],
            "default_0_5_precision": test_def["precision"],
            "default_0_5_recall": test_def["recall"],
            "default_0_5_f1": test_def["f1"],
            "default_0_5_auroc": test_def["auroc"],

            "always_positive_precision": ap["precision"],
            "always_positive_recall": ap["recall"],
            "always_positive_f1": ap["f1"],
            "always_positive_predicted_positive_rate": ap["predicted_positive_rate"],

            "classifier_as_ranker_ndcg@5": rank["ndcg@5"],
            "classifier_as_ranker_err@5": rank["err@5"],
        })

        qrows = list(csv.DictReader(open(pq, encoding="utf-8")))
        if len(qrows) != 70:
            raise ValueError(f"tau {tau} seed {seed}: expected 70 per-query rows, got {len(qrows)}")
        all_query_rows.extend(qrows)

        all_score_rows.extend(list(csv.DictReader(open(sc, encoding="utf-8"))))

if len(all_seed_rows) != 30:
    raise ValueError(f"Expected 30 seed rows, got {len(all_seed_rows)}")

if len(all_query_rows) != 2100:
    raise ValueError(f"Expected 2100 per-query rows, got {len(all_query_rows)}")

per_seed_out = OUT / "roberta_large_classifier_weighted_gamma_per_seed_q70.csv"
with open(per_seed_out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_seed_rows[0].keys()))
    w.writeheader()
    w.writerows(all_seed_rows)

per_query_out = OUT / "roberta_large_classifier_weighted_gamma_as_ranker_per_query_q70.csv"
with open(per_query_out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_query_rows[0].keys()))
    w.writeheader()
    w.writerows(all_query_rows)

scores_out = OUT / "roberta_large_classifier_weighted_gamma_item_scores_q70.csv"
with open(scores_out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_score_rows[0].keys()))
    w.writeheader()
    w.writerows(all_score_rows)

summary_rows = []
metrics = [
    "pos_weight",
    "train_positive_rate",
    "selected_gamma",
    "positive_rate",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "auroc",
    "default_0_5_predicted_positive_rate",
    "default_0_5_precision",
    "default_0_5_recall",
    "default_0_5_f1",
    "default_0_5_auroc",
    "always_positive_precision",
    "always_positive_recall",
    "always_positive_f1",
    "always_positive_predicted_positive_rate",
    "classifier_as_ranker_ndcg@5",
    "classifier_as_ranker_err@5",
]

for tau in TAUS:
    rows = [r for r in all_seed_rows if int(r["tau"]) == tau]
    item = {
        "method": f"roberta_large_classifier_weighted_gamma_tau{tau}",
        "tau": tau,
        "n_seeds": len(rows),
    }

    for m in metrics:
        vals = [float(r[m]) for r in rows if r[m] not in ["", "None", None]]
        item[m + "_mean"] = mean(vals)
        item[m + "_std"] = std(vals)

    summary_rows.append(item)

summary_out = OUT / "roberta_large_classifier_weighted_gamma_summary_q70.json"
json.dump(summary_rows, open(summary_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("Saved:")
print(per_seed_out)
print(per_query_out)
print(scores_out)
print(summary_out)

print("\nSummary:")
for r in summary_rows:
    print(
        f"{r['method']} "
        f"gamma={r['selected_gamma_mean']:.4f}±{r['selected_gamma_std']:.4f} "
        f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f} "
        f"P={r['precision_mean']:.4f}±{r['precision_std']:.4f} "
        f"R={r['recall_mean']:.4f}±{r['recall_std']:.4f} "
        f"AUROC={r['auroc_mean']:.4f}±{r['auroc_std']:.4f} "
        f"rank_nDCG@5={r['classifier_as_ranker_ndcg@5_mean']:.4f}±{r['classifier_as_ranker_ndcg@5_std']:.4f} "
        f"rank_ERR@5={r['classifier_as_ranker_err@5_mean']:.4f}±{r['classifier_as_ranker_err@5_std']:.4f}"
    )
