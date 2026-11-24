#!/usr/bin/env python3
"""Judge entropy and confidence across the study corpus.

Read the frozen seed-0 train/val/test partitions once (not all repeated seeds).
All 46,953 distinct stories / 343 queries enter ONCE. The audit never modifies
frozen data, never loads a ranker, and needs no GPU. This is DESCRIPTIVE judge
analysis; it does not establish calibration/accuracy or explain ranker causally.

Run from the repository root (requires the private seed-0 split files):
  python scripts/irrj_r1_2_judge_entropy.py --root . --check-only
  python scripts/irrj_r1_2_judge_entropy.py --root . --bootstrap 20000
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FROZEN = "frozen/v1_generation_judging_46953/splits_q70/seed_0"
PROBS = {"A": "culture_score_A_probs", "B": "culture_score_B_probs"}
EXPECTED = {"A": "yA_expected", "B": "yB_expected"}
GRADES = (1, 2, 3, 4, 5)
EXPECTED_STORIES = 46953
EXPECTED_QUERIES = 343


def arg_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/irrj_r1_2_judge_entropy"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20261022)
    p.add_argument("--expected-score-tolerance", type=float, default=0.02,
                   help="Fail if P-weighted score and frozen expected label differ > this")
    return p.parse_args()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def write_csv(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def try_decode(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(v)
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Cannot parse probability string: {v[:120]!r}") from e
    return v


def parse_probs(v, location):
    v = try_decode(v)
    if isinstance(v, (list, tuple)) and len(v) == 5:
        raw = list(v)
    elif isinstance(v, dict):
        if all(str(k) in v for k in GRADES):
            raw = [v[str(k)] for k in GRADES]
        elif all(k in v for k in GRADES):
            raw = [v[k] for k in GRADES]
        elif all(f"score_{k}" in v for k in GRADES):
            raw = [v[f"score_{k}"] for k in GRADES]
        elif "probs" in v:
            return parse_probs(v["probs"], location)
        else:
            raise ValueError(f"Cannot identify grades 1..5 at {location}; dict keys={list(v)[:20]}")
    else:
        raise ValueError(f"Expected a five-probability list/dict at {location}; got {type(v).__name__}: {str(v)[:120]}")
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Non-numeric probability at {location}: {raw}") from e
    if any(not math.isfinite(x) or x < -1e-9 for x in values):
        raise ValueError(f"Invalid probability at {location}: {values}")
    total = sum(values)
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"Probabilities not normalized at {location}: sum={total}, vals={values}")
    return tuple(max(x, 0.0) / total for x in values)


def entropy_bits(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def per_query_ties(items, judge):
    """Within-query hard grade pairs; distinguishable under expected scores (1e-4 rounded)."""
    n = len(items)
    all_pairs = math.comb(n, 2)
    by_grade = defaultdict(list)
    for r in items:
        by_grade[r[f"hard_{judge}"]].append(round(r[f"recomputed_{judge}"], 4))
    hard_tied = 0
    same_continuous = 0
    for vals in by_grade.values():
        hard_tied += math.comb(len(vals), 2)
        counts = Counter(vals)
        same_continuous += sum(math.comb(k, 2) for k in counts.values())
    return all_pairs, hard_tied, hard_tied - same_continuous


def mean(nums):
    return sum(nums) / len(nums)


def load_and_validate(root, tolerance):
    base = root / FROZEN
    source_files = [base / f"{part}.jsonl" for part in ("train", "val", "test")]
    for fp in source_files:
        if not fp.is_file():
            raise FileNotFoundError(f"Frozen IRRJ original split not found: {fp}")
    partitions, stories, queries = {}, {}, {}
    for part, fp in zip(("train", "val", "test"), source_files):
        pqueries = set()
        with fp.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for k in ("query_id", "story_id", "story", *PROBS.values(), *EXPECTED.values()):
                    if k not in row:
                        raise KeyError(f"{fp}:{lineno}: missing {k}; keys={list(row)[:40]}")
                qid, sid = str(row["query_id"]), str(row["story_id"])
                key = (qid, sid)
                if key in stories:
                    raise ValueError(f"Repeated (query_id, story_id) across frozen partitions: {key}")
                rec = {"query_id": qid, "story_id": sid, "partition": part}
                for j in ("A", "B"):
                    pp = parse_probs(row[PROBS[j]], f"{fp.name}:{lineno}, judge={j}")
                    rec[f"p_{j}"] = pp
                    rec[f"entropy_{j}"] = entropy_bits(pp)
                    rec[f"confidence_{j}"] = max(pp)
                    rec[f"hard_{j}"] = pp.index(max(pp)) + 1
                    rec[f"recomputed_{j}"] = sum(k * val for k, val in zip(GRADES, pp))
                    rec[f"frozen_expected_{j}"] = float(row[EXPECTED[j]])
                    rec[f"expected_error_{j}"] = abs(rec[f"recomputed_{j}"] - rec[f"frozen_expected_{j}"])
                    if rec[f"expected_error_{j}"] > tolerance:
                        raise ValueError(
                            f"Expected score mismatch {key}, judge {j}: "
                            f"from probabilities={rec[f'recomputed_{j}']:.6f}, "
                            f"frozen={rec[f'frozen_expected_{j}']:.6f}. "
                            "STOP: check original rubric/score-probability keys, do not mix labels."
                        )
                rec["hard_disagreement"] = int(rec["hard_A"] != rec["hard_B"])
                rec["abs_expected_difference"] = abs(rec["frozen_expected_A"] - rec["frozen_expected_B"])
                stories[key] = rec
                pqueries.add(qid)
                if qid in queries and queries[qid] != part:
                    raise ValueError(f"Query appears in multiple partitions: {qid}")
                queries[qid] = part
        partitions[part] = {"n_queries": len(pqueries), "n_stories": sum(r["partition"] == part for r in stories.values())}
    expected_parts = {"train": 238, "val": 35, "test": 70}
    if any(partitions[p]["n_queries"] != expected_parts[p] for p in expected_parts):
        raise ValueError(f"Wrong query-level partition sizes: {partitions}")
    if len(stories) != EXPECTED_STORIES or len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"Unexpected dataset count: stories={len(stories)} (expected {EXPECTED_STORIES}), "
                         f"queries={len(queries)} (expected {EXPECTED_QUERIES}).")
    records = list(stories.values())
    return records, partitions, {str(p.relative_to(root)): digest(p) for p in source_files}


def run(opts):
    root = opts.root.resolve()
    records, parts, hashes = load_and_validate(root, opts.expected_score_tolerance)
    by_query = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)
    max_errors = {j: max(r[f"expected_error_{j}"] for r in records) for j in ("A", "B")}
    print(f"PASS: original frozen corpus; {len(records)} unique stories, {len(by_query)} unique queries")
    print(f"Partitions: {parts}")
    print("Max |recomputed expectation - frozen score|: " + ", ".join(f"{j}={max_errors[j]:.8f}" for j in ("A", "B")))
    print("Distribution fields: " + ", ".join(f"{j}={PROBS[j]}" for j in ("A", "B")))
    if opts.check_only:
        print("CHECK-ONLY: no outputs written, no training or GPU required")
        return

    out = opts.output_dir if opts.output_dir.is_absolute() else root / opts.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing analysis directory: {out}")
    out.mkdir(parents=True)

    query_rows = []
    for qid, items in sorted(by_query.items()):
        row = {"query_id": qid, "n_stories": len(items), "partition": items[0]["partition"]}
        for j in ("A", "B"):
            row[f"entropy_{j}_mean"] = mean([r[f"entropy_{j}"] for r in items])
            row[f"confidence_{j}_mean"] = mean([r[f"confidence_{j}"] for r in items])
            row[f"expected_minus_hard_abs_{j}_mean"] = mean(
                [abs(r[f"recomputed_{j}"] - r[f"hard_{j}"]) for r in items]
            )
            row[f"high_conf_{j}_share_ge_0_9"] = mean([r[f"confidence_{j}"] >= 0.9 for r in items])
            row[f"ambiguous_{j}_share_lt_0_6"] = mean([r[f"confidence_{j}"] < 0.6 for r in items])
            pairs, tied, resolved = per_query_ties(items, j)
            row[f"pairs_{j}"] = pairs
            row[f"hard_tied_pairs_{j}"] = tied
            row[f"hard_ties_resolved_by_expected_{j}"] = resolved
            row[f"hard_tie_pair_share_{j}"] = tied / pairs if pairs else 0.0
            row[f"resolved_hard_tie_share_{j}"] = resolved / tied if tied else 0.0
        row["entropy_A_minus_B"] = row["entropy_A_mean"] - row["entropy_B_mean"]
        row["confidence_A_minus_B"] = row["confidence_A_mean"] - row["confidence_B_mean"]
        row["judge_hard_disagreement_share"] = mean([r["hard_disagreement"] for r in items])
        row["judge_expected_abs_diff_mean"] = mean([r["abs_expected_difference"] for r in items])
        query_rows.append(row)

    nqueries = len(query_rows)
    query_fields = list(query_rows[0])
    write_csv(out / "per_query_entropy.csv", query_rows, query_fields)

    judge_rows = []
    for j in ("A", "B"):
        judge_stories = records
        hvals = [r[f"entropy_{j}"] for r in judge_stories]
        conf = [r[f"confidence_{j}"] for r in judge_stories]
        hard = [r[f"hard_{j}"] for r in judge_stories]
        total_pairs = sum(q[f"pairs_{j}"] for q in query_rows)
        tied_pairs = sum(q[f"hard_tied_pairs_{j}"] for q in query_rows)
        resolved = sum(q[f"hard_ties_resolved_by_expected_{j}"] for q in query_rows)
        row = {
            "judge": j, "n_stories": len(records), "n_queries": nqueries,
            "mean_entropy_bits_story_weighted": mean(hvals),
            "mean_entropy_bits_query_weighted": mean([r[f"entropy_{j}_mean"] for r in query_rows]),
            "mean_normalized_entropy_story_weighted": mean(hvals) / math.log2(5),
            "mean_max_probability_story_weighted": mean(conf),
            "mean_max_probability_query_weighted": mean([r[f"confidence_{j}_mean"] for r in query_rows]),
            "median_entropy_bits_story_weighted": sorted(hvals)[len(hvals)//2],
            "share_confidence_at_least_0_9": mean([x >= 0.9 for x in conf]),
            "share_confidence_below_0_6": mean([x < 0.6 for x in conf]),
            "mean_abs_expected_minus_hard_story_weighted": mean([abs(r[f"recomputed_{j}"] - r[f"hard_{j}"]) for r in records]),
            "within_query_hard_tied_pair_share": tied_pairs / total_pairs,
            "within_query_hard_ties_resolved_by_expected_share": resolved / tied_pairs if tied_pairs else 0.0,
            "within_query_candidate_pairs": total_pairs,
            "within_query_hard_tied_pairs": tied_pairs,
            "within_query_ties_resolved_by_expected": resolved,
            "max_expected_score_consistency_error": max_errors[j],
        }
        for k in GRADES:
            row[f"hard_share_grade_{k}"] = hard.count(k) / len(hard)
            row[f"mean_probability_grade_{k}"] = mean([r[f"p_{j}"][k-1] for r in records])
        judge_rows.append(row)
    write_csv(out / "judge_entropy_summary.csv", judge_rows, list(judge_rows[0]))

    # Confidence stratification: descriptive only; four equal-sized entropy bins per judge.
    bins = []
    for j in ("A", "B"):
        sorted_records = sorted(records, key=lambda r: (r[f"entropy_{j}"], r["query_id"], r["story_id"]))
        n = len(sorted_records)
        for idx in range(4):
            group = sorted_records[(idx * n) // 4 : ((idx + 1) * n) // 4]
            bins.append({"judge": j, "entropy_quartile": idx + 1,
                         "n_stories": len(group),
                         "entropy_min_bits": group[0][f"entropy_{j}"],
                         "entropy_max_bits": group[-1][f"entropy_{j}"],
                         "mean_entropy_bits": mean([r[f"entropy_{j}"] for r in group]),
                         "mean_max_probability": mean([r[f"confidence_{j}"] for r in group]),
                         "mean_abs_expected_minus_hard": mean([abs(r[f"recomputed_{j}"] - r[f"hard_{j}"]) for r in group])})
    write_csv(out / "entropy_quartiles.csv", bins, list(bins[0]))

    # Paired cluster bootstrap at the unique query level. Resampling is independent of
    # the ten reused resplit test sets because this analysis uses seed-0 corpus once.
    cluster_rows = []
    for what in ("entropy", "confidence"):
        values = [r[f"{what}_A_minus_B"] for r in query_rows]
        point = mean(values)
        if opts.bootstrap:
            import numpy as np
            rng = np.random.default_rng(opts.bootstrap_seed + (0 if what == "entropy" else 1))
            arr = np.asarray(values, dtype=np.float64)
            vals = []
            chunk = 1000
            for start in range(0, opts.bootstrap, chunk):
                batch = min(chunk, opts.bootstrap - start)
                draw = rng.integers(0, nqueries, size=(batch, nqueries))
                vals.extend(arr[draw].mean(axis=1).tolist())
            low, high = np.quantile(vals, [0.025, 0.975]).tolist()
        else:
            low = high = ""
        cluster_rows.append({"comparison": f"Judge_A_minus_Judge_B_mean_{what}",
                             "unit": "bits" if what == "entropy" else "max probability",
                             "estimate_query_weighted": point,
                             "cluster_bootstrap_ci95_low": low,
                             "cluster_bootstrap_ci95_high": high,
                             "n_unique_queries": nqueries,
                             "bootstrap_draws": opts.bootstrap})
    write_csv(out / "judge_pair_cluster_ci.csv", cluster_rows, list(cluster_rows[0]))

    manifest = {
        "study": "Judge-distribution diagnostics",
        "source": "frozen IRRJ seed-0 train/val/test union",
        "n_unique_stories": len(records), "n_unique_queries": nqueries,
        "partitions": parts, "input_sha256": hashes,
        "distribution_keys": PROBS, "frozen_expected_keys": EXPECTED,
        "max_expectation_recomputation_error": max_errors,
        "expected_score_tolerance": opts.expected_score_tolerance,
        "entropy_base": 2, "entropy_max_bits": math.log2(5),
        "hard_label_definition": "argmax over grade probabilities 1-5",
        "resolved_tie_definition": "Within-query pair has same hard grade, but expected scores differ after rounding to 4 decimal places",
        "ci_method": "paired bootstrap over 343 unique queries; query-weighted means",
        "bootstrap_draws": opts.bootstrap, "bootstrap_seed": opts.bootstrap_seed,
        "limitations": ["No human ground-truth quality/accuracy or confidence calibration claim.",
                        "Descriptive entropy does not identify causal reasons for downstream ranking gains.",
                        "Per-story descriptive summaries are candidate-weighted; paired CIs use equally weighted queries."],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# Judge-distribution diagnostics", "",
              f"Single frozen 343-query corpus, {len(records):,} unique stories; no resplit duplicates.",
              "Entropy is Shannon entropy in bits over the five rubric score probabilities.", ""]
    for r in judge_rows:
        report += [f"Judge {r['judge']}: mean entropy={r['mean_entropy_bits_story_weighted']:.4f} bits; "
                   f"mean confidence={r['mean_max_probability_story_weighted']:.4f}; "
                   f"hard-tied within-query candidate pairs={r['within_query_hard_tied_pair_share']:.3%}; "
                   f"ties distinguishable by expected scores={r['within_query_hard_ties_resolved_by_expected_share']:.3%}."]
    report += ["", "Query-clustered A-minus-B comparisons:"]
    for r in cluster_rows:
        report.append(f"- {r['comparison']}: {r['estimate_query_weighted']:+.4f}; "
                      f"95% CI [{r['cluster_bootstrap_ci95_low']}, {r['cluster_bootstrap_ci95_high']}]")
    report += ["", "Interpretation: descriptive uncertainty of LLM score-token distributions; "
                   "not a confidence calibration against human judgments or proof of the cause "
                   "of expected-vs-hard ranking results."]
    (out / "summary_readme.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    opts = arg_parse()
    if opts.bootstrap < 0 or opts.expected_score_tolerance <= 0:
        raise SystemExit("--bootstrap must be >=0 and --expected-score-tolerance must be >0")
    run(opts)
