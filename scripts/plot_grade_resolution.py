#!/usr/bin/env python3
"""Draw the data underlying Figure 1 from archived grade-resolution results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs/irrj_r2_grade_granularity_v2/granularity_summary.csv"
DEFAULT_OUT = ROOT / "local_outputs/figure1_grade_resolution.png"
LEVELS = ["2", "3", "5", "10", "20", "50", "100", "continuous"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to plot Figure 1: python -m pip install matplotlib") from exc

    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_model = {(r["model"].lower(), r["levels"].lower()): float(r["mean_ndcg@5"])
                for r in rows}
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = list(range(len(LEVELS)))
    for model, label, marker in (("pairwise", "Pairwise TF-IDF SGD", "o"),
                                 ("ridge", "TF-IDF Ridge", "s")):
        y = [by_model[(model, level)] for level in LEVELS]
        ax.plot(x, y, marker=marker, linewidth=1.7, label=label)
    ax.set_xticks(x, ["Continuous" if x == "continuous" else x for x in LEVELS])
    ax.set_xlabel("Nominal training-target levels")
    ax.set_ylabel("Judge A nDCG@5")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
