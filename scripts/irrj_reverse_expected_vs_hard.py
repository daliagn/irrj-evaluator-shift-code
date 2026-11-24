#!/usr/bin/env python3
"""IRRJ original-rubric reciprocal A->B expected-vs-hard diagnostic.

Reuses saved Phase-8 A-hard RoBERTa512 checkpoints and previously computed
A-expected held-out predictions from irrj_evaluator_shift_2x2.py.
Performs INFERENCE ONLY, no retraining or checkpoint selection.

Usage from the repository root:
  python scripts/irrj_reverse_expected_vs_hard.py --check-only --seeds all
  python scripts/irrj_reverse_expected_vs_hard.py --seeds all --bootstrap 20000
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

SOURCE = "scripts/irrj_evaluator_shift_2x2.py"
HARD_TREE = "outputs/phase8_bias_inheritance/roberta_large_pointwise_yA_hard"
HARD_SPLITS = "data/splits_q70_argmax_hard"
EXPECTED_RESULTS = "outputs/irrj_evaluator_2x2_full/per_query.csv"
HARD_LABEL = "yA"
JUDGES = {"A": "yA_expected", "B": "yB_expected"}
METRICS = ("ndcg@5", "err@5", "spearman", "kendall_tau_b")
METHOD = "irrj_original_rubric_common_tiebreak_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--seeds", default="all")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--expected-results", type=Path, default=Path(EXPECTED_RESULTS))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/irrj_reverse_expected_vs_hard"))
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20261022)
    return p.parse_args()


def load_original(root):
    import sys
    path = root / SOURCE
    if not path.is_file():
        raise FileNotFoundError(f"Existing evaluator script not found: {path}")
    spec = importlib.util.spec_from_file_location("irrj_original_2x2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fullpath(root, path):
    return path if path.is_absolute() else root / path


def rows_by_key(records):
    out = {}
    for r in records:
        key = (str(r["query_id"]), str(r["story_id"]))
        if key in out:
            raise ValueError(f"Duplicate story key: {key}")
        out[key] = r
    return out


def verify_seed(root, seed, original):
    root_split = root / original.SPLITS / f"seed_{seed}"
    hard_split = root / HARD_SPLITS / f"seed_{seed}"
    for part in ("train", "val", "test"):
        original_path = root_split / f"{part}.jsonl"
        hard_path = hard_split / f"{part}.jsonl"
        if not (original_path.is_file() and hard_path.is_file()):
            raise FileNotFoundError(f"Missing {part} split in {root_split} or {hard_split}")
        base = rows_by_key(original.read_jsonl(original_path))
        alt = rows_by_key(original.read_jsonl(hard_path))
        if set(base) != set(alt):
            raise ValueError(f"Hard/expected {part} story membership mismatch at seed {seed}")
        for key in base:
            if str(base[key].get("story")) != str(alt[key].get("story")):
                raise ValueError(f"Hard/expected story text mismatch at seed {seed}, {key}")
            for field in ("age", "moral", "culture"):
                if base[key].get(field) != alt[key].get(field):
                    raise ValueError(f"Hard/expected query field {field} mismatch at seed {seed}, {key}")
    cfgpath = root / HARD_TREE / f"seed_{seed}/run_config.json"
    ckptpath = root / HARD_TREE / f"seed_{seed}/model/roberta_regressor.pt"
    tokenizer_dir = ckptpath.parent
    if not (cfgpath.is_file() and ckptpath.is_file() and tokenizer_dir.is_dir()):
        raise FileNotFoundError(f"Missing A-hard assets at seed {seed}: {cfgpath}, {ckptpath}")
    cfg = original.read_json(cfgpath)
    if int(cfg.get("seed", -1)) != seed or cfg.get("training_label") != HARD_LABEL:
        raise ValueError(f"A-hard config mismatch at seed {seed}: {cfg}")
    if int(cfg.get("max_length", -1)) != 512:
        raise ValueError(f"A-hard max_length is not 512 at seed {seed}")
    expected_cfg = original.read_json(root / original.CHECKPOINT_BASE / f"roberta_large_pointwise_yA_expected/seed_{seed}/run_config.json")
    for key in ("epochs", "batch_size", "eval_batch_size", "grad_accum_steps", "learning_rate", "weight_decay", "warmup_ratio", "max_length", "dropout", "max_grad_norm"):
        if cfg.get(key) != expected_cfg.get(key):
            raise ValueError(f"A-hard and A-expected configs differ on {key}, seed {seed}: {cfg.get(key)} vs {expected_cfg.get(key)}")
    return {"seed":seed,"hard_checkpoint":str(ckptpath),"hard_checkpoint_sha256": original.digest(ckptpath),
            "hard_config":str(cfgpath),"hard_config_sha256":original.digest(cfgpath),
            "original_test_sha256":original.digest(root_split / "test.jsonl"),
            "hard_test_sha256":original.digest(hard_split / "test.jsonl"),
            "hard_training_label":cfg["training_label"], "max_length":cfg["max_length"]}


def read_expected(path, seeds):
    if not path.is_file():
        raise FileNotFoundError(f"A-expected inference results not found: {path}")
    ref = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["training_judge"] != "A" or int(row["seed"]) not in seeds:
                continue
            if row["evaluation_judge"] not in JUDGES:
                continue
            key = (row["evaluation_judge"], int(row["seed"]), str(row["query_id"]))
            if key in ref:
                raise ValueError(f"Duplicate A-expected query result {key}")
            ref[key] = row
    expected_count = 2 * 70 * len(seeds)
    if len(ref) != expected_count:
        raise ValueError(f"A-expected reference has {len(ref)} query rows, expected {expected_count}")
    return ref


def infer_hard_seed(root, seed, original, verified, opts):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    test = original.read_jsonl(root / original.SPLITS / f"seed_{seed}/test.jsonl")
    ckptpath = root / HARD_TREE / f"seed_{seed}/model/roberta_regressor.pt"
    cfg = original.read_json(root / HARD_TREE / f"seed_{seed}/run_config.json")
    payload = original.load_checkpoint(torch, ckptpath)
    checks = {
        "seed":int(payload.get("seed", -1)) == seed,
        "training_label": payload.get("training_label") == HARD_LABEL,
        "best_epoch":payload.get("best_epoch") == cfg.get("best_epoch"),
        "best_val_ndcg":abs(float(payload.get("best_val_ndcg",-10))-float(cfg.get("best_val_ndcg",-20))) < 1e-10,
        "max_length":int(payload.get("max_length",-1)) == 512,
    }
    if not all(checks.values()):
        raise ValueError(f"A-hard checkpoint mismatch, seed {seed}: {checks}")
    model_path = opts.model_path or Path(str(payload.get("model_path", "")))
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(ckptpath.parent), local_files_only=True, use_fast=True)
    dataset = verified.StoryDataset(test, tokenizer, 512, HARD_LABEL)
    loader = DataLoader(dataset, batch_size=opts.eval_batch_size, shuffle=False,
                        collate_fn=verified.Collator(tokenizer), num_workers=opts.num_workers)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for inference")
    device = torch.device("cuda")
    model = verified.RobertaRegressor(str(model_path), dropout=float(payload["dropout"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    scores = verified.predict(model, loader, device)
    if len(scores) != len(test) or any(v is None for v in scores):
        raise RuntimeError(f"Incomplete inference on A-hard seed {seed}")
    query = []
    for judge, label in JUDGES.items():
        rows = verified.evaluate(test, scores, seed, METHOD, label_key=label)
        if len(rows) != 70:
            raise RuntimeError(f"Expected 70 results on A-hard seed {seed}, eval {judge}, got {len(rows)}")
        for r in rows:
            query.append({"training_judge":"A", "supervision":"hard", "evaluation_judge":judge,
                         "seed":seed,"query_id":str(r["query_id"]),"pool_size":r.get("pool_size"),
                         **{m:float(r[m]) for m in METRICS}})
    item = [{"seed":seed,"query_id":str(r["query_id"]),"story_id":str(r["story_id"]),
             "hard_score":float(s),"yA_expected":float(r["yA_expected"]),
             "yB_expected":float(r["yB_expected"])} for r,s in zip(test,scores)]
    del model, loader, dataset, tokenizer, payload
    torch.cuda.empty_cache()
    return query, item


def compare(ref, hard, nboot, seed):
    import numpy as np
    hard_map = {}
    for row in hard:
        k = (row["evaluation_judge"],int(row["seed"]),str(row["query_id"]))
        if k in hard_map:
            raise ValueError(f"Duplicate A-hard row {k}")
        hard_map[k] = row
    if set(hard_map) != set(ref):
        raise ValueError(f"A-expected/A-hard query set mismatch: {len(ref)} vs {len(hard_map)}")
    rng = np.random.default_rng(seed)
    summaries, paired = [], []
    for judge in JUDGES:
        keys = sorted(k for k in ref if k[0] == judge)
        clusters = sorted({k[2] for k in keys})
        qi = {q:i for i,q in enumerate(clusters)}
        for metric in METRICS:
            numerator = np.zeros(len(clusters), dtype=float)
            denominator = np.zeros(len(clusters), dtype=int)
            ev, hv = [], []
            for key in keys:
                e = float(ref[key][metric]); h = float(hard_map[key][metric])
                d = e - h
                ev.append(e); hv.append(h)
                j=qi[key[2]]; numerator[j] += d; denominator[j] += 1
                paired.append({"evaluation_judge":judge, "seed":key[1],"query_id":key[2],
                               "metric":metric,"expected":e,"hard":h,"expected_minus_hard":d})
            if nboot:
                # Batch the resampling so memory stays bounded on shared nodes.
                samples=[]
                for start in range(0, nboot, 2000):
                    take = min(2000,nboot-start)
                    indices = rng.integers(0,len(clusters),size=(take,len(clusters)))
                    samples.append(numerator[indices].sum(axis=1)/denominator[indices].sum(axis=1))
                low, high = [float(v) for v in np.quantile(np.concatenate(samples),[0.025,0.975])]
            else:
                low=high=float("nan")
            summaries.append({"training_judge":"A", "evaluation_judge":judge,
                              "metric":metric,"n_query_seed":len(keys),"n_unique_queries":len(clusters),
                              "expected_mean":float(np.mean(ev)), "hard_mean":float(np.mean(hv)),
                              "expected_minus_hard":float(numerator.sum()/denominator.sum()),
                              "cluster_bootstrap_ci95_low":low, "cluster_bootstrap_ci95_high":high,
                              "bootstrap_draws":nboot})
    return summaries, paired


def main():
    opts = parse_args()
    root = opts.root.resolve()
    original = load_original(root)
    seeds = original.select_seeds(opts.seeds)
    if opts.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    # Preflight original expected checkpoints and canonical IRRJ splits.
    original.verify_inputs(root, seeds)
    provenance = []
    for seed in seeds:
        p = verify_seed(root, seed, original)
        provenance.append(p)
        print(f"CHECK A-hard seed {seed}: identical query/story membership across original and hard splits, checkpoint found", flush=True)
    refpath = fullpath(root, opts.expected_results)
    ref = read_expected(refpath, seeds)
    print(f"PASS: {len(seeds)} A-hard checkpoints; {len(ref)} A-expected reference query scores", flush=True)
    if opts.check_only:
        print("CHECK-ONLY: no GPU inference, no results produced", flush=True)
        return
    out = fullpath(root, opts.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output dir: {out}")
    verified = original.verified_module(root)
    out.mkdir(parents=True)
    hard, items = [], []
    for seed in seeds:
        print(f"INFERENCE A-hard seed {seed}", flush=True)
        q, i = infer_hard_seed(root,seed,original,verified,opts)
        hard.extend(q); items.extend(i)
    summaries, paired = compare(ref,hard,opts.bootstrap,opts.bootstrap_seed)
    original.write_csv(out/"summary_expected_vs_hard.csv",summaries,
                       ["training_judge","evaluation_judge","metric","n_query_seed","n_unique_queries",
                        "expected_mean","hard_mean","expected_minus_hard","cluster_bootstrap_ci95_low",
                        "cluster_bootstrap_ci95_high","bootstrap_draws"])
    original.write_csv(out/"per_query_hard.csv",hard,
                       ["training_judge","supervision","evaluation_judge","seed","query_id","pool_size",*METRICS])
    original.write_csv(out/"paired_deltas.csv",paired,
                       ["evaluation_judge","seed","query_id","metric","expected","hard","expected_minus_hard"])
    original.write_csv(out/"item_scores_hard.csv",items,
                       ["seed","query_id","story_id","hard_score","yA_expected","yB_expected"])
    (out/"run_manifest.json").write_text(json.dumps({
        "evaluation":"IRRJ original rubric, A-trained RoBERTa512 expected vs hard",
        "inference_only":True,"new_training":False,"checkpoint_selection_changed":False,
        "reused_A_expected_reference":str(refpath),
        "reference_sha256":original.digest(refpath),"script":__file__,
        "baseline_evaluator_script_sha256":original.digest(root / SOURCE),
        "metric_script_sha256":original.digest(root / original.VERIFIED),
        "unique_query_cluster_bootstrap":True,"bootstrap_draws":opts.bootstrap,
        "interpretation":"Descriptive uncertainty under repeated overlapping query splits; no independent resplit assumption."
        ,"preflight":provenance},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("\nA-expected minus A-hard (original IRRJ labels):")
    for r in summaries:
        if r["metric"]=="ndcg@5":
            print(f"  evaluation {r['evaluation_judge']}: expected={r['expected_mean']:.4f}, hard={r['hard_mean']:.4f}, "
                  f"delta={r['expected_minus_hard']:+.4f}, 95% CI [{r['cluster_bootstrap_ci95_low']:+.4f},{r['cluster_bootstrap_ci95_high']:+.4f}]")
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
