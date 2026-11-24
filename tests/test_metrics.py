"""Small regression checks for the ranking metrics used in the paper."""

import ast
import hashlib
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "eval_baselines_q70_tiesafe.py",
    "train_tfidf_ridge_q70.py",
    "train_tfidf_ridge_label_ablation_q70.py",
    "train_pairwise_tfidf_sgd_q70.py",
    "train_pairwise_tfidf_sgd_label_ablation_q70.py",
    "train_roberta_pointwise_q70.py",
    "train_roberta_pointwise_hard_q70.py",
    "train_roberta_pointwise_A_oracle_q70.py",
    "train_roberta_classifier_weighted_gamma_q70.py",
)
FUNCTIONS = {
    "gain", "stable_jitter", "ranked_indices", "dcg_at_k", "ndcg_at_k",
    "err_at_k", "average_ranks", "pearson", "spearman", "kendall_tau_b",
}


def metric_functions(script):
    path = ROOT / "verified" / "scripts" / script
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    names = {node.name for node in selected}
    required = {"gain", "stable_jitter", "ranked_indices", "dcg_at_k",
                "ndcg_at_k", "err_at_k"}
    if not required.issubset(names):
        raise AssertionError(f"Missing ranking metrics in {script}")
    namespace = {"math": math, "hashlib": hashlib, "MAX_REL": 5.0}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RankingMetricsTest(unittest.TestCase):
    def test_metrics_and_ties(self):
        rows = [{"query_id": "q1", "story_id": f"s{i}"} for i in range(5)]
        labels = [5, 4, 3, 2, 1]
        scores = [5, 4, 3, 2, 1]
        for script in SCRIPTS:
            with self.subTest(script=script):
                fn = metric_functions(script)
                self.assertAlmostEqual(fn["ndcg_at_k"](labels, scores, rows, 7, "m"), 1.0)
                self.assertEqual(fn["ndcg_at_k"]([0]*5, scores, rows, 7, "m"), 0.0)
                self.assertEqual(fn["err_at_k"]([0]*5, scores, rows, 7, "m"), 0.0)

                expected_err = 0.0
                continuation = 1.0
                for position, relevance in enumerate(labels, start=1):
                    satisfaction = (2**relevance - 1) / 32
                    expected_err += continuation * satisfaction / position
                    continuation *= (1 - satisfaction)
                self.assertAlmostEqual(fn["err_at_k"](labels, scores, rows, 7, "m"), expected_err)

                tied = [1.0]*5
                ranked = [rows[i]["story_id"] for i in
                          fn["ranked_indices"](tied, rows, 7, "m")]
                order = [3, 1, 4, 0, 2]
                permuted_rows = [rows[i] for i in order]
                ranked_permuted = [permuted_rows[i]["story_id"] for i in
                                   fn["ranked_indices"](tied, permuted_rows, 7, "m")]
                self.assertEqual(ranked, ranked_permuted)

                if "spearman" in fn:
                    self.assertAlmostEqual(fn["spearman"]([1,2,3], [3,2,1]), -1.0)
                    self.assertAlmostEqual(fn["spearman"]([1,1,2], [2,2,1]), -1.0)
                    self.assertIsNone(fn["spearman"]([2,2,2], [1,2,3]))
                if "kendall_tau_b" in fn:
                    self.assertAlmostEqual(fn["kendall_tau_b"]([1,2,3], [3,2,1]), -1.0)
                    self.assertAlmostEqual(fn["kendall_tau_b"]([1,1,2], [1,2,3]),
                                           2 / math.sqrt(6))
                    self.assertIsNone(fn["kendall_tau_b"]([1,1,1], [1,2,3]))


if __name__ == "__main__":
    unittest.main()
