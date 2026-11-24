#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np

def load_qids(base_path: str):
    obj = json.load(open(base_path, "r", encoding="utf-8"))

    # Support both layouts:
    # 1) train_qids/dev_qids/test_qids
    if all(k in obj for k in ["train_qids", "dev_qids", "test_qids"]):
        return obj, obj["train_qids"], obj["dev_qids"], obj["test_qids"], "qids"

    # 2) train/dev/test (lists)
    if all(k in obj for k in ["train", "dev", "test"]):
        return obj, obj["train"], obj["dev"], obj["test"], "plain"

    raise ValueError(
        f"Base split format not recognized. Top-level keys: {list(obj.keys())[:50]}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_split", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    base_obj, train_q, dev_q, test_q, layout = load_qids(args.base_split)

    n_train, n_dev, n_test = len(train_q), len(dev_q), len(test_q)
    all_qids = sorted(set(train_q + dev_q + test_q))

    # Query IDs must be unique across the input partitions.
    if len(all_qids) != (n_train + n_dev + n_test):
        raise ValueError("Duplicate query IDs in the input partitions.")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seed values")
    paths = [os.path.join(args.out_dir, f"split_seed{s}.json") for s in seeds]
    if any(os.path.exists(path) for path in paths):
        raise FileExistsError("Refusing to overwrite an existing split file")
    os.makedirs(args.out_dir, exist_ok=True)

    for s in seeds:
        rng = np.random.RandomState(s)
        perm = rng.permutation(all_qids).tolist()

        out_train = perm[:n_train]
        out_dev   = perm[n_train:n_train + n_dev]
        out_test  = perm[n_train + n_dev:n_train + n_dev + n_test]

        out = {
            "seed": s,
            "label": base_obj.get("label", None),
            "n_train_qids": len(out_train),
            "n_dev_qids": len(out_dev),
            "n_test_qids": len(out_test),
        }

        if layout == "qids":
            out["train_qids"] = out_train
            out["dev_qids"]   = out_dev
            out["test_qids"]  = out_test
        else:
            out["train"] = out_train
            out["dev"]   = out_dev
            out["test"]  = out_test

        out_path = os.path.join(args.out_dir, f"split_seed{s}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(seeds)} splits to: {args.out_dir}")

if __name__ == "__main__":
    main()
