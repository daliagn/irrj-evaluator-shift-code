#!/usr/bin/env python3
"""Summarise archived original-rubric rankings across ten query-level splits."""

import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "Ridge hard": (
        "verified/reports/tfidf_ridge_label_ablation_q70/tfidf_ridge_label_ablation_per_query_q70.csv",
        "tfidf_ridge_yB_hard",
    ),
    "Ridge expected": (
        "verified/reports/tfidf_ridge_label_ablation_q70/tfidf_ridge_label_ablation_per_query_q70.csv",
        "tfidf_ridge_yB_expected",
    ),
    "Pairwise hard": (
        "verified/reports/pairwise_tfidf_sgd_hard_q70/pairwise_tfidf_sgd_yB_hard_per_query_q70.csv",
        "pairwise_tfidf_sgd_yB_hard",
    ),
    "Pairwise expected": (
        "verified/reports/pairwise_tfidf_sgd_q70/pairwise_tfidf_sgd_yB_expected_per_query_q70.csv",
        "pairwise_tfidf_sgd_yB_expected",
    ),
    "RoBERTa hard": (
        "verified/reports/roberta_large_pointwise_hard_q70/roberta_large_pointwise_yB_hard_per_query_q70.csv",
        "roberta_large_pointwise_yB_hard",
    ),
    "RoBERTa expected": (
        "verified/reports/roberta_large_pointwise_q70/roberta_large_pointwise_yB_expected_per_query_q70.csv",
        "roberta_large_pointwise_yB_expected",
    ),
}


def load_scores(path, method):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = [row for row in csv.DictReader(handle) if row["method"] == method]

    query_seeds = {(int(row["seed"]), row["query_id"]) for row in records}
    seeds = {int(row["seed"]) for row in records}
    if len(records) != 700 or len(query_seeds) != 700 or seeds != set(range(10)):
        raise ValueError(f"Unexpected query-level records for {method}: {path}")
    return records


def main():
    print(f"{'Model':20s} {'nDCG@5':>9s} {'ERR@5':>9s}")
    for name, (filename, method) in RESULTS.items():
        rows = load_scores(ROOT / filename, method)
        ndcg = mean(float(row["ndcg@5"]) for row in rows)
        err = mean(float(row["err@5"]) for row in rows)
        print(f"{name:20s} {ndcg:9.4f} {err:9.4f}")


if __name__ == "__main__":
    main()
