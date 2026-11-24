#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

def get_qid(rec):
    if rec.get("query_id"):
        return str(rec["query_id"])
    age = rec.get("age")
    moral = rec.get("moral")
    culture = rec.get("culture")
    return f"age={age}|moral={moral}|culture={culture}"

def parse_qid(qid):
    parts = {}
    for item in qid.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            parts[k] = v
    return parts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    counts = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            counts[get_qid(rec)] += 1

    plan = []
    for qid, current in sorted(counts.items()):
        deficit = max(0, args.target - current)
        # generate more than deficit because some outputs will be duplicates or invalid
        to_generate = int(deficit * args.multiplier)
        if deficit > 0:
            parts = parse_qid(qid)
            plan.append({
                "query_id": qid,
                "age": parts.get("age"),
                "moral": parts.get("moral"),
                "culture": parts.get("culture"),
                "current_unique": current,
                "target_unique": args.target,
                "deficit": deficit,
                "suggested_generate": max(to_generate, 10)
            })

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("query_id,age,moral,culture,current_unique,target_unique,deficit,suggested_generate\n")
        for r in plan:
            f.write(
                f'"{r["query_id"]}",{r["age"]},{r["moral"]},{r["culture"]},'
                f'{r["current_unique"]},{r["target_unique"]},{r["deficit"]},{r["suggested_generate"]}\n'
            )

    print("Queries needing top-up:", len(plan))
    print("Total suggested new generations:", sum(r["suggested_generate"] for r in plan))
    print("Saved:", out_json)
    print("Saved:", out_csv)
    print("Weakest 20:")
    for r in sorted(plan, key=lambda x: x["current_unique"])[:20]:
        print(r)

if __name__ == "__main__":
    main()
