#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = obj.get("story_id")
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"Missing or invalid story_id in {path}:{line_number}")
            if sid in data:
                raise ValueError(f"Duplicate story_id {sid!r} in {path}:{line_number}")
            data[sid] = obj
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_a", required=True, help="Judge A scores jsonl")
    ap.add_argument("--judge_b", required=True, help="Judge B scores jsonl")
    ap.add_argument("--out", required=True, help="Merged output jsonl")
    args = ap.parse_args()

    A = load_jsonl(args.judge_a)
    B = load_jsonl(args.judge_b)

    common_ids = sorted(set(A) & set(B))
    for sid in common_ids:
        for field in ("query_id", "culture"):
            left, right = A[sid].get(field), B[sid].get(field)
            if left is not None and right is not None and left != right:
                raise ValueError(f"Conflicting {field} for story_id {sid!r}")

    out_path = Path(args.out)
    if out_path.resolve() in (Path(args.judge_a).resolve(), Path(args.judge_b).resolve()):
        raise ValueError("Output path must differ from both input files")
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")

    with out_path.open("w", encoding="utf-8") as out:
        for sid in common_ids:
            a = A[sid]
            b = B[sid]

            merged = {
                "story_id": sid,
                "query_id": a.get("query_id") or b.get("query_id"),
                "culture": a.get("culture") or b.get("culture"),
                "culture_score_A": a.get("culture_score_A"),
                "culture_score_B": b.get("culture_score_B"),
                "status_A": a.get("status"),
                "status_B": b.get("status")
            }
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"[DONE] merged {len(common_ids)} stories")
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

