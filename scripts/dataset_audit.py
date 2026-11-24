#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

TEXT_KEYS = ["story", "text", "generated_text", "response", "output", "story_text"]
ID_KEYS = ["story_id", "id", "record_id"]
QUERY_KEYS = ["query_id", "qid"]

def load_jsonl(path):
    records = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad += 1
    return records, bad

def get_first(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return ""

def get_text(rec):
    return get_first(rec, TEXT_KEYS)

def get_id(rec):
    return get_first(rec, ID_KEYS)

def get_qid(rec):
    qid = get_first(rec, QUERY_KEYS)
    if qid:
        return qid

    age = rec.get("age") or rec.get("target_age")
    moral = rec.get("moral") or rec.get("moral_value")
    culture = rec.get("culture") or rec.get("cultural_context")

    if age is not None and moral is not None and culture is not None:
        return f"age={age}|moral={moral}|culture={culture}"

    return ""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def word_count(text):
    return len(text.split())

def audit_file(path):
    records, bad_json = load_jsonl(path)

    ids = []
    qids = []
    norm_texts = []
    lengths = []
    key_counter = Counter()

    missing_id = 0
    missing_qid = 0
    missing_text = 0

    for rec in records:
        key_counter.update(rec.keys())

        sid = get_id(rec)
        qid = get_qid(rec)
        text = get_text(rec)

        if not sid:
            missing_id += 1
        if not qid:
            missing_qid += 1
        if not text:
            missing_text += 1

        ids.append(sid)
        qids.append(qid)
        norm_texts.append(normalize_text(text))
        lengths.append(word_count(text))

    id_counts = Counter([x for x in ids if x])
    qid_counts = Counter([x for x in qids if x])
    text_counts = Counter([x for x in norm_texts if x])

    pool_sizes = list(qid_counts.values())

    report = {
        "file": str(path),
        "num_records": len(records),
        "bad_json_lines": bad_json,
        "missing_id": missing_id,
        "missing_qid": missing_qid,
        "missing_text": missing_text,
        "unique_ids": len(id_counts),
        "duplicate_id_groups": sum(1 for _, c in id_counts.items() if c > 1),
        "duplicate_id_extra_records": sum(c - 1 for _, c in id_counts.items() if c > 1),
        "unique_normalized_texts": len(text_counts),
        "exact_duplicate_text_groups": sum(1 for _, c in text_counts.items() if c > 1),
        "exact_duplicate_extra_records": sum(c - 1 for _, c in text_counts.items() if c > 1),
        "length_checks": {
            "lt_30_words": sum(1 for x in lengths if x < 30),
            "lt_50_words": sum(1 for x in lengths if x < 50),
            "lt_100_words": sum(1 for x in lengths if x < 100),
        },
        "pool_summary": {
            "num_queries": len(pool_sizes),
            "min_pool_size": min(pool_sizes) if pool_sizes else None,
            "max_pool_size": max(pool_sizes) if pool_sizes else None,
            "avg_pool_size": (sum(pool_sizes) / len(pool_sizes)) if pool_sizes else None,
            "queries_lt_50": sum(1 for x in pool_sizes if x < 50),
            "queries_lt_80": sum(1 for x in pool_sizes if x < 80),
            "queries_lt_100": sum(1 for x in pool_sizes if x < 100),
        },
        "weakest_queries": sorted(qid_counts.items(), key=lambda x: x[1])[:20],
        "strongest_queries": sorted(qid_counts.items(), key=lambda x: x[1], reverse=True)[:20],
        "top_duplicate_texts": [
            {"count": c, "text_preview": txt[:250]}
            for txt, c in text_counts.most_common(10)
            if c > 1
        ],
        "top_keys": key_counter.most_common(50),
    }

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports = []
    for file_name in args.files:
        path = Path(file_name)
        if not path.exists():
            reports.append({"file": str(path), "error": "file does not exist"})
        else:
            reports.append(audit_file(path))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print(f"Saved audit report to: {out}")

    for r in reports:
        print("=" * 80)
        print("FILE:", r.get("file"))
        if "error" in r:
            print("ERROR:", r["error"])
            continue
        print("records:", r["num_records"])
        print("bad_json_lines:", r["bad_json_lines"])
        print("missing_id:", r["missing_id"])
        print("missing_qid:", r["missing_qid"])
        print("missing_text:", r["missing_text"])
        print("unique_ids:", r["unique_ids"])
        print("duplicate_id_extra_records:", r["duplicate_id_extra_records"])
        print("unique_normalized_texts:", r["unique_normalized_texts"])
        print("exact_duplicate_extra_records:", r["exact_duplicate_extra_records"])
        print("length_checks:", r["length_checks"])
        print("pool_summary:", r["pool_summary"])

if __name__ == "__main__":
    main()
