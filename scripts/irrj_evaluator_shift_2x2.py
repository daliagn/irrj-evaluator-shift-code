#!/usr/bin/env python3
"""Inference-only original-rubric IRRJ evaluator-transfer 2x2 audit.

Run from the repository root.  Does not retrain, retune, or write into frozen
assets.  Uses 20 previously trained Phase-8 RoBERTa-large 512 checkpoints.

--check-only: standard-library-only integrity check; safe on login node.
Full inference: run in an existing GPU compute allocation, not on login node.

Training on Judge A is a *diagnostic condition*, distinct from the main IRRJ
B-supervised/A-evaluated deployment scenario.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import filecmp
import hashlib
import importlib.util
import json
from pathlib import Path

SPLITS = "frozen/v1_generation_judging_46953/splits_q70"
WORK_SPLITS = "data/splits_q70"
CHECKPOINT_BASE = "outputs/phase8_bias_inheritance"
VERIFIED = "verified/scripts/train_roberta_pointwise_q70.py"
LABELS = {"A": "yA_expected", "B": "yB_expected"}
COMMON_METHOD = "irrj_original_rubric_common_tiebreak_v1"
METRICS = ("ndcg@5", "err@5", "spearman", "kendall_tau_b")
SEED_IDS = range(10)


def args_parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--check-only", action="store_true", help="No model loading/inference; safe on login node")
    ap.add_argument("--seeds", default="all", help="all or comma-separated seeds, e.g. 0,1")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/irrj_revision_evaluator_2x2"))
    ap.add_argument("--model-path", type=Path, default=None, help="Override pretrained roberta-large directory")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=20000, help="Unique-query cluster bootstrap draws")
    ap.add_argument("--bootstrap-seed", type=int, default=20261021)
    return ap.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def select_seeds(value):
    if value == "all":
        return list(SEED_IDS)
    out = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not out or len(set(out)) != len(out) or any(s not in SEED_IDS for s in out):
        raise ValueError("--seeds requires unique values between 0 and 9")
    return out


def paths_for(root, judge, seed):
    base = root / CHECKPOINT_BASE / f"roberta_large_pointwise_y{judge}_expected" / f"seed_{seed}"
    return base / "run_config.json", base / "model/roberta_regressor.pt", base / "model"


def verify_inputs(root, seeds):
    if not (root / VERIFIED).is_file():
        raise FileNotFoundError(root / VERIFIED)
    detail = []
    for seed in seeds:
        base = root / SPLITS / f"seed_{seed}"
        work = root / WORK_SPLITS / f"seed_{seed}"
        for part in ("train", "val", "test"):
            fp, wp = base / f"{part}.jsonl", work / f"{part}.jsonl"
            if not (fp.is_file() and wp.is_file()):
                raise FileNotFoundError(f"Missing IRRJ {part} split: {fp} / {wp}")
            if not filecmp.cmp(fp, wp, shallow=False):
                raise RuntimeError(f"Frozen/current split mismatch: seed {seed}, {part}")
        split_keys = {}
        split_sizes = {}
        test_rows = None
        for part in ("train", "val", "test"):
            rows = read_jsonl(base / f"{part}.jsonl")
            split_keys[part] = {str(r["query_id"]) for r in rows}
            split_sizes[part] = len(rows)
            if part == "test":
                test_rows = rows
        if [len(split_keys[p]) for p in ("train", "val", "test")] != [238, 35, 70]:
            raise ValueError(f"Unexpected query counts at seed {seed}: {split_keys}")
        if any(split_keys[a] & split_keys[b] for a,b in (("train","val"),("train","test"),("val","test"))):
            raise ValueError(f"Query overlap across splits for seed {seed}")
        assert test_rows is not None
        test_ids = [(str(r["query_id"]), str(r["story_id"])) for r in test_rows]
        if len(set(test_ids)) != len(test_ids):
            raise ValueError(f"Duplicate test story/query IDs for seed {seed}")
        for row in test_rows:
            if not all(k in row for k in ("story", "story_id", "query_id", *LABELS.values())):
                raise ValueError(f"Incomplete original-rubric labels at seed {seed}")
            for key in LABELS.values():
                float(row[key])
        for judge in ("A", "B"):
            config_path, checkpoint, token_dir = paths_for(root, judge, seed)
            if not config_path.is_file() or not checkpoint.is_file() or not token_dir.is_dir():
                raise FileNotFoundError(f"Missing {judge} seed {seed} assets: {config_path}, {checkpoint}")
            conf = read_json(config_path)
            if int(conf.get("seed", -1)) != seed or conf.get("training_label") != LABELS[judge]:
                raise ValueError(f"Phase-8 config incompatible: Judge {judge}, seed {seed}")
            if "512" not in str(conf.get("max_length")):
                raise ValueError(f"Unexpected Phase-8 max_length for {judge} seed {seed}: {conf.get('max_length')}")
            detail.append({
                "train_judge": judge, "seed": seed, "training_label": conf["training_label"],
                "n_test_items": len(test_rows), "n_test_queries": len(split_keys["test"]),
                "max_length": conf["max_length"], "run_config": str(config_path),
                "checkpoint": str(checkpoint), "test_sha256": digest(base / "test.jsonl"),
            })
        print(f"CHECK seed {seed}: split hashes match, no overlap, 70 test queries, A+B checkpoints present", flush=True)
    return detail


def verified_module(root):
    import sys
    script = root / VERIFIED
    spec = importlib.util.spec_from_file_location("irrj_verified_metrics_and_model", script)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {script}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    for name in ("StoryDataset", "Collator", "RobertaRegressor", "predict", "evaluate"):
        if not hasattr(m, name):
            raise AttributeError(f"Missing expected function {name} in {script}")
    return m


def load_checkpoint(torch, path):
    # Load only trusted local checkpoints.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Unexpected checkpoint payload: {path}")
    return obj


def evaluate_arm_seed(root, judge, seed, verified, opts):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    split = root / SPLITS / f"seed_{seed}/test.jsonl"
    rows = read_jsonl(split)
    conf_path, checkpoint_path, tokenizer_path = paths_for(root, judge, seed)
    conf = read_json(conf_path)
    payload = load_checkpoint(torch, checkpoint_path)
    validations = {
        "seed": int(payload.get("seed", -1)) == seed,
        "training_label": payload.get("training_label") == LABELS[judge],
        "best_epoch": payload.get("best_epoch") == conf.get("best_epoch"),
        "best_val_ndcg": abs(float(payload.get("best_val_ndcg", -10)) - float(conf.get("best_val_ndcg", -20))) < 1e-10,
        "max_length": int(payload.get("max_length", -1)) == int(conf.get("max_length", -2)),
    }
    if not all(validations.values()):
        raise ValueError(f"Checkpoint provenance mismatch {judge}/seed{seed}: {validations}")
    model_path = opts.model_path or Path(str(payload.get("model_path", "")))
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"roberta-large base model unavailable: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
    # StoryDataset uses this label during data loading; supervision is frozen.
    ds = verified.StoryDataset(rows, tokenizer, int(payload["max_length"]), LABELS[judge])
    loader = DataLoader(ds, batch_size=opts.eval_batch_size, shuffle=False,
                        collate_fn=verified.Collator(tokenizer), num_workers=opts.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and not opts.cpu else "cpu")
    if device.type == "cpu" and not opts.cpu:
        raise RuntimeError("GPU not available. Run on a GPU compute node, not login node.")
    model = verified.RobertaRegressor(str(model_path), dropout=float(payload["dropout"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    scores = verified.predict(model, loader, device)
    if len(scores) != len(rows) or any(s is None for s in scores):
        raise RuntimeError(f"Incomplete model prediction for {judge} seed {seed}")
    item_scores = [
        {"training_judge": judge, "seed": seed, "query_id": r["query_id"],
         "story_id": r["story_id"], "score": float(score),
         "yA_expected": float(r["yA_expected"]), "yB_expected": float(r["yB_expected"])}
        for r, score in zip(rows, scores)
    ]
    query_rows = []
    for eval_judge, label in LABELS.items():
        result = verified.evaluate(rows, scores, seed, COMMON_METHOD, label_key=label)
        if len(result) != 70:
            raise RuntimeError(f"Expected 70 per-query rows, got {len(result)} for {judge}->{eval_judge} seed {seed}")
        for line in result:
            query_rows.append({"training_judge": judge, "evaluation_judge": eval_judge,
                               "seed": seed, "query_id": str(line["query_id"]),
                               "pool_size": line.get("pool_size"),
                               **{metric: float(line[metric]) for metric in METRICS}})
    meta = {"training_judge": judge, "seed": seed, "training_label": LABELS[judge],
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
            "run_config_sha256": digest(conf_path), "original_test_sha256": digest(split),
            "model_path": str(model_path), "max_length": int(payload["max_length"]),
            "best_epoch": payload["best_epoch"], "best_val_ndcg": payload["best_val_ndcg"],
            "checkpoint_checks": validations, "common_tiebreak_method": COMMON_METHOD}
    del model, loader, ds, tokenizer, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return query_rows, item_scores, meta


def summarize(query_rows, n_bootstrap, random_seed):
    import numpy as np
    # The statistical contrasts are evaluated *within the same evaluation judge*
    # and paired by (seed, query_id). Distinct seed-query rows are not independent.
    by_cell = defaultdict(list)
    by_pair = {}
    for row in query_rows:
        cell = (row["training_judge"], row["evaluation_judge"])
        key = (row["seed"], str(row["query_id"]), row["evaluation_judge"], row["training_judge"])
        if key in by_pair:
            raise ValueError(f"Duplicate row {key}")
        by_pair[key] = row
        by_cell[cell].append(row)
    matrix = []
    for training in ("A", "B"):
        for evaluator in ("A", "B"):
            rows = by_cell[(training, evaluator)]
            if not rows:
                continue
            matrix.append({"training_judge": training, "evaluation_judge": evaluator,
                           "n_query_seed": len(rows), "n_unique_queries": len({r["query_id"] for r in rows}),
                           **{m: float(np.mean([r[m] for r in rows])) for m in METRICS}})
    contrasts = []
    paired_rows = []
    rng = np.random.default_rng(random_seed)
    for evaluator, preferred, competitor in (("A", "A", "B"), ("B", "B", "A")):
        first = {(k[0],k[1]):r for k,r in by_pair.items() if k[2] == evaluator and k[3] == preferred}
        second = {(k[0],k[1]):r for k,r in by_pair.items() if k[2] == evaluator and k[3] == competitor}
        if first.keys() != second.keys():
            raise ValueError(f"Unpaired source models under Judge {evaluator}")
        query_ids = sorted({q for _,q in first})
        qindex = {q: i for i, q in enumerate(query_ids)}
        for metric in METRICS:
            total = np.zeros(len(query_ids))
            count = np.zeros(len(query_ids))
            for key in sorted(first):
                delta = float(first[key][metric] - second[key][metric])
                idx = qindex[key[1]]
                total[idx] += delta
                count[idx] += 1
                paired_rows.append({"evaluation_judge": evaluator, "preferred_train": preferred,
                                    "other_train": competitor, "seed": key[0], "query_id": key[1],
                                    "metric": metric, "delta": delta})
            if n_bootstrap:
                sampled = rng.integers(0, len(query_ids), size=(n_bootstrap, len(query_ids)))
                boot_means = total[sampled].sum(axis=1) / count[sampled].sum(axis=1)
                low, high = np.quantile(boot_means, [0.025, 0.975])
            else:
                low, high = float("nan"), float("nan")
            contrasts.append({"eval_judge": evaluator,
                              "contrast": f"train_{preferred}_minus_train_{competitor}",
                              "metric": metric,
                              "n_query_seed": int(count.sum()),
                              "n_unique_queries": len(query_ids),
                              "mean_delta": float(total.sum()/count.sum()),
                              "cluster_bootstrap_ci95_low": float(low),
                              "cluster_bootstrap_ci95_high": float(high),
                              "bootstrap_draws": n_bootstrap})
    return matrix, contrasts, paired_rows


def main():
    opts = args_parse()
    root = opts.root.resolve()
    seeds = select_seeds(opts.seeds)
    info = verify_inputs(root, seeds)
    print(f"PASS: {2*len(seeds)} saved model configurations; frozen IRRJ splits verified", flush=True)
    if opts.check_only:
        print("CHECK-ONLY: no GPU inference; no results have been calculated.")
        return
    if opts.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")
    verified = verified_module(root)
    out = opts.output_dir if opts.output_dir.is_absolute() else root / opts.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {out}")
    out.mkdir(parents=True)
    query_rows, item_rows, provenance = [], [], []
    for seed in seeds:
        for judge in ("A", "B"):
            print(f"INFERENCE: train {judge}, seed {seed}", flush=True)
            q, i, p = evaluate_arm_seed(root, judge, seed, verified, opts)
            query_rows.extend(q)
            item_rows.extend(i)
            provenance.append(p)
    matrix, contrasts, paired = summarize(query_rows, opts.bootstrap, opts.bootstrap_seed)
    write_csv(out / "matrix_mean.csv", matrix, ["training_judge", "evaluation_judge", "n_query_seed", "n_unique_queries", *METRICS])
    write_csv(out / "per_query.csv", query_rows,
              ["training_judge", "evaluation_judge", "seed", "query_id", "pool_size", *METRICS])
    write_csv(out / "item_scores.csv", item_rows,
              ["training_judge", "seed", "query_id", "story_id", "score", "yA_expected", "yB_expected"])
    write_csv(out / "paired_deltas.csv", paired,
              ["evaluation_judge", "preferred_train", "other_train", "seed", "query_id", "metric", "delta"])
    write_csv(out / "contrasts_cluster_ci.csv", contrasts,
              ["eval_judge", "contrast", "metric", "n_query_seed", "n_unique_queries",
               "mean_delta", "cluster_bootstrap_ci95_low", "cluster_bootstrap_ci95_high", "bootstrap_draws"])
    (out / "run_manifest.json").write_text(json.dumps({
        "inference_only": True, "new_training": False, "checkpoint_selection_changed": False,
        "original_irrj_rubric": True, "reference_splits": SPLITS, "seeds": seeds,
        "verified_script": VERIFIED, "verified_script_sha256": digest(root / VERIFIED),
        "common_tiebreak_method": COMMON_METHOD, "mean_estimator": "mean over 70 test queries in each of the 10 repeated query splits",
        "ci_method": "paired cluster bootstrap resampling unique query_id and retaining cross-seed repeats within cluster; descriptive sensitivity only",
        "warning": "Repeated train/test resplits share queries and training data. Cluster intervals do not fully characterise model-training uncertainty. A-trained arms are diagnostic, not the main B-to-A protocol.",
        "preflight": info, "provenance": provenance,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n2x2 nDCG@5 (original IRRJ labels):")
    for r in matrix:
        print(f"  train {r['training_judge']} -> eval {r['evaluation_judge']}: {r['ndcg@5']:.4f}")
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
