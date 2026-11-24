#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from collections import Counter

TEXT_KEYS = ["story", "story_text", "text", "generated_text", "response", "output"]

def get_text(rec):
    for k in TEXT_KEYS:
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return ""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def get_qid(rec):
    if rec.get("query_id"):
        return str(rec["query_id"])

    age = rec.get("age") or rec.get("target_age")
    moral = rec.get("moral") or rec.get("moral_value")
    culture = rec.get("culture") or rec.get("cultural_context")

    if age is not None and moral is not None and culture is not None:
        return f"age={age}|moral={moral}|culture={culture}"

    return "UNKNOWN_QUERY"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scope", choices=["global", "within_query"], default="within_query")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    total = 0
    kept = 0
    dropped = 0

    before_q = Counter()
    after_q = Counter()
    dropped_q = Counter()

    duplicate_examples = []

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            rec = json.loads(line)

            qid = get_qid(rec)
            text = get_text(rec)
            norm = normalize_text(text)

            before_q[qid] += 1

            if args.scope == "within_query":
                key = (qid, norm)
            else:
                key = norm

            if not norm:
                dropped += 1
                dropped_q[qid] += 1
                continue

            if key in seen:
                dropped += 1
                dropped_q[qid] += 1

                if len(duplicate_examples) < 20:
                    duplicate_examples.append({
                        "query_id": qid,
                        "text_preview": text[:300]
                    })
                continue

            seen.add(key)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            after_q[qid] += 1

    pool_sizes = list(after_q.values())

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "scope": args.scope,
        "total_records": total,
        "kept_records": kept,
        "dropped_records": dropped,
        "num_queries_before": len(before_q),
        "num_queries_after": len(after_q),
        "pool_after": {
            "min": min(pool_sizes) if pool_sizes else None,
            "max": max(pool_sizes) if pool_sizes else None,
            "avg": sum(pool_sizes) / len(pool_sizes) if pool_sizes else None,
            "queries_lt_10": sum(1 for x in pool_sizes if x < 10),
            "queries_lt_20": sum(1 for x in pool_sizes if x < 20),
            "queries_lt_30": sum(1 for x in pool_sizes if x < 30),
            "queries_lt_50": sum(1 for x in pool_sizes if x < 50),
        },
        "weakest_queries_after": sorted(after_q.items(), key=lambda x: x[1])[:30],
        "strongest_queries_after": sorted(after_q.items(), key=lambda x: x[1], reverse=True)[:30],
        "duplicate_examples": duplicate_examples,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Input:", input_path)
    print("Output:", output_path)
    print("Scope:", args.scope)
    print("Total:", total)
    print("Kept:", kept)
    print("Dropped:", dropped)
    print("Report:", report_path)

if __name__ == "__main__":
    main()
