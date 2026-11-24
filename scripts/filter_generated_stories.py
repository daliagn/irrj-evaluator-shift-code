#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter

TEXT_KEYS = ["story", "story_text", "text"]

def get_text(rec):
    for k in TEXT_KEYS:
        if k in rec and rec[k]:
            return str(rec[k])
    return ""

def word_count(text):
    return len(text.split())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_words", type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output)
    rep = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    examples = []

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue

            counts["total"] += 1
            rec = json.loads(line)
            text = get_text(rec).strip()
            wc = word_count(text)

            reason = None

            if not text:
                reason = "empty_text"
            elif wc < args.min_words:
                reason = "too_short"
            elif "moral:" not in text.lower():
                reason = "missing_moral"

            if reason:
                counts[f"dropped_{reason}"] += 1
                counts["dropped_total"] += 1
                if len(examples) < 30:
                    examples.append({
                        "reason": reason,
                        "query_id": rec.get("query_id"),
                        "word_count": wc,
                        "preview": text[:300]
                    })
                continue

            counts["kept"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "input": args.input,
        "output": args.output,
        "min_words": args.min_words,
        "counts": dict(counts),
        "dropped_examples": examples
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["counts"], indent=2))

if __name__ == "__main__":
    main()
