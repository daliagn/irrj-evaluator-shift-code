#!/usr/bin/env python3
"""Summarise the archived per-query results used in the main paper tables.

No model inference or training is performed. The reported dispersion is the
population standard deviation of the ten per-seed query means (ddof=0).
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "verified" / "reports"

FILES = {
    "baseline": REPORTS / "baselines_q70_tiesafe/baseline_per_query_q70_tiesafe.csv",
    "ridge": REPORTS / "tfidf_ridge_label_ablation_q70/tfidf_ridge_label_ablation_per_query_q70.csv",
    "pairwise_hard": REPORTS / "pairwise_tfidf_sgd_hard_q70/pairwise_tfidf_sgd_yB_hard_per_query_q70.csv",
    "pairwise_expected": REPORTS / "pairwise_tfidf_sgd_q70/pairwise_tfidf_sgd_yB_expected_per_query_q70.csv",
    "roberta_hard": REPORTS / "roberta_large_pointwise_hard_q70/roberta_large_pointwise_yB_hard_per_query_q70.csv",
    "roberta_expected": REPORTS / "roberta_large_pointwise_q70/roberta_large_pointwise_yB_expected_per_query_q70.csv",
}

METHODS = [
    ("Random", "baseline", "random"),
    ("BM25", "baseline", "bm25"),
    ("Length", "baseline", "length"),
    ("Ridge hard", "ridge", "tfidf_ridge_yB_hard"),
    ("Ridge expected", "ridge", "tfidf_ridge_yB_expected"),
    ("Pairwise hard", "pairwise_hard", "pairwise_tfidf_sgd_yB_hard"),
    ("Pairwise expected", "pairwise_expected", "pairwise_tfidf_sgd_yB_expected"),
    ("RoBERTa hard", "roberta_hard", "roberta_large_pointwise_yB_hard"),
    ("RoBERTa expected", "roberta_expected", "roberta_large_pointwise_yB_expected"),
    ("B-score reference", "baseline", "bscore_expected_yB"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def format_metric(rows: list[dict[str, str]], method: str, field: str) -> str:
    seed_values = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            seed_values[int(row["seed"])].append(float(row[field]))
    if set(seed_values) != set(range(10)) or any(len(v) != 70 for v in seed_values.values()):
        raise ValueError(f"Unexpected seed/query coverage for {method}, {field}")
    means = [statistics.fmean(seed_values[s]) for s in range(10)]
    return f"{statistics.fmean(means):.4f} ± {statistics.pstdev(means):.4f}"


def main() -> None:
    datasets = {key: read_rows(path) for key, path in FILES.items()}
    print("Table 2: B-to-A ranking results")
    print("Standard deviations are across ten seed means (population SD).\n")
    print(f"{'Method':<22} {'nDCG@5':<20} {'ERR@5':<20}")
    for label, dataset, method in METHODS:
        rows = datasets[dataset]
        print(f"{label:<22} {format_metric(rows, method, 'ndcg@5'):<20} "
              f"{format_metric(rows, method, 'err@5'):<20}")

    rows = read_rows(ROOT / "outputs/irrj_r2_grade_granularity_v2/granularity_summary.csv")
    print("\nTable 4 / Figure 1: mean nDCG@5 by target resolution")
    print(f"{'Levels':<13} {'Pairwise':<12} {'Ridge':<12}")
    for level in ("2", "3", "5", "10", "20", "50", "100", "continuous"):
        scores = {}
        for row in rows:
            if row["levels"].lower() == level:
                scores[row["model"].lower()] = float(row["mean_ndcg@5"])
        if set(scores) != {"pairwise", "ridge"}:
            raise ValueError(f"Missing grade-resolution rows for {level}")
        print(f"{level:<13} {scores['pairwise']:<12.4f} {scores['ridge']:<12.4f}")


if __name__ == "__main__":
    main()
