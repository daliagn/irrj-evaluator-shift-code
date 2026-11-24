#!/usr/bin/env python3
"""Query-clustered significance tests (Table 3) from archived per-query results.

- Original main B-supervised/A-evaluated q70 results only (Pairwise, RoBERTa-256).
- No training, GPU, web access, or modification to frozen/verified assets.
- Matched (seed, query_id), repeated test-query IDs clustered in resampling.
"""
from __future__ import annotations
import argparse, csv, math, sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("NumPy is required: python -m pip install numpy")

SOURCE_CHOICES = {
    "pairwise_expected": [
        "frozen/v1_generation_judging_46953/verified_pairwise_tfidf_sgd_yB_expected_per_query_q70.csv",
        "verified/reports/pairwise_tfidf_sgd_q70/pairwise_tfidf_sgd_yB_expected_per_query_q70.csv"],
    "pairwise_hard": [
        "frozen/v1_generation_judging_46953/verified_pairwise_tfidf_sgd_yB_hard_per_query_q70.csv",
        "verified/reports/pairwise_tfidf_sgd_hard_q70/pairwise_tfidf_sgd_yB_hard_per_query_q70.csv"],
    "roberta_expected": [
        "frozen/v1_generation_judging_46953/verified_roberta_large_pointwise_yB_expected_per_query_q70.csv",
        "verified/reports/roberta_large_pointwise_q70/roberta_large_pointwise_yB_expected_per_query_q70.csv"],
    "roberta_hard": [
        "frozen/v1_generation_judging_46953/verified_roberta_large_pointwise_yB_hard_per_query_q70.csv",
        "verified/reports/roberta_large_pointwise_hard_q70/roberta_large_pointwise_yB_hard_per_query_q70.csv"]
}
METRICS = ("ndcg@5", "err@5", "spearman", "kendall_tau_b")


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames or []
        need={"seed", "query_id", *METRICS}
        if not need.issubset(cols):
            raise ValueError(f"Bad columns in {path}: missing {sorted(need-set(cols))}; seen {cols}")
        out={}
        for row in reader:
            k=(int(row["seed"]),str(row["query_id"]))
            if k in out: raise ValueError(f"Duplicate row {k} in {path}")
            values={m:float(row[m]) for m in METRICS}
            if any(not math.isfinite(v) for v in values.values()):
                raise ValueError(f"Nonfinite metric for {k} in {path}")
            out[k]=values
    if len(out)!=700 or len({q for _,q in out})!=315:
        raise ValueError(f"Expected 700 query-seed pairs and 315 unique queries in {path}; got {len(out)} and {len({q for _,q in out})}")
    return out


def write_csv(path, rows):
    if not rows: raise ValueError("No rows for "+str(path))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def mean(values):return float(np.mean(values))


def percentile(values,p):return float(np.quantile(values,p))


def query_clusters(diff_by_pair):
    """Sorted unique query IDs; sum of differences per cluster and cluster sizes."""
    groups=defaultdict(list)
    for (seed,q),d in diff_by_pair.items():groups[q].append(float(d))
    ids=sorted(groups)
    return (np.array([sum(groups[q]) for q in ids],dtype=float),
            np.array([len(groups[q]) for q in ids],dtype=float))


def cluster_resample_ci(sums,counts,draws,seed):
    rng=np.random.default_rng(seed)
    n=len(sums)
    samples=[]
    for start in range(0,draws,500):
        batch=min(500,draws-start)
        inds=rng.integers(0,n,size=(batch,n))
        samples.append(sums[inds].sum(axis=1)/counts[inds].sum(axis=1))
    v=np.concatenate(samples)
    return percentile(v,.025), percentile(v,.975)


def cluster_signflip_p(sums,counts,draws,seed):
    rng=np.random.default_rng(seed)
    observed=abs(sums.sum()/counts.sum())
    n=len(sums); extreme=0
    for start in range(0,draws,500):
        batch=min(500,draws-start)
        signs=rng.choice([-1.,1.],size=(batch,n))
        null=np.abs(signs@sums/counts.sum())
        extreme+=int(np.sum(null>=observed-1e-15))
    return (extreme+1)/(draws+1)


def holm_bonferroni(pvalues):
    m=len(pvalues); indices=sorted(range(m),key=lambda i:pvalues[i]);out=[0.]*m;max_adjusted=0.
    for rank,i in enumerate(indices):
        max_adjusted=max(max_adjusted,(m-rank)*pvalues[i])
        out[i]=min(1.,max_adjusted)
    return out



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--output-dir",type=Path,default=Path("local_outputs/significance"))
    ap.add_argument("--draws",type=int,default=20000)
    ap.add_argument("--check-only",action="store_true")
    a=ap.parse_args()
    if a.draws<1000 and not a.check_only:ap.error("Use at least 1000 draws")
    root=a.root.resolve()
    resolved={}
    for key,options in SOURCE_CHOICES.items():
        choice=next((root/x for x in options if (root/x).is_file()),None)
        if choice is None:raise FileNotFoundError(f"Missing {key}: {options}")
        resolved[key]=choice
    results={k:load_rows(p) for k,p in resolved.items()}
    keys=set(results["pairwise_expected"])
    if any(set(v)!=keys for v in results.values()):
        raise ValueError("Unmatched test queries between main-paper methods")
    print("PASS: all four original B-to-A results matched on 700 query-seed pairs, 315 unique queries")
    for k,v in resolved.items():print("SOURCE",k,":",v.relative_to(root))
    if a.check_only:
        print("CHECK-ONLY: no outputs written, no training or GPU")
        return
    out=a.output_dir if a.output_dir.is_absolute() else root/a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    contrasts=[
        ("Pairwise Expected - Pairwise Hard","pairwise_expected","pairwise_hard",METRICS),
        ("RoBERTa256 Expected - RoBERTa256 Hard","roberta_expected","roberta_hard",METRICS),
        ("Pairwise Expected - RoBERTa256 Expected","pairwise_expected","roberta_expected",("ndcg@5","err@5")),
    ]
    detail=[]; significance=[]
    for contrast_index,(title,left,right,metric_list) in enumerate(contrasts):
        for metric_index,metric in enumerate(metric_list):
            pairs={k:results[left][k][metric]-results[right][k][metric] for k in sorted(keys)}
            diffs=np.fromiter(pairs.values(),dtype=float)
            sums,counts=query_clusters(pairs)
            qlow,qhigh=cluster_resample_ci(sums,counts,a.draws,12345+contrast_index*100+metric_index)
            p=cluster_signflip_p(sums,counts,a.draws,97812+contrast_index*100+metric_index)
            eps=1e-12
            pos=int(np.sum(diffs>eps));neg=int(np.sum(diffs < -eps));ties=len(diffs)-pos-neg
            significance.append({"contrast":title,"metric":metric,"n_query_seed":700,"n_unique_queries":315,
              "mean_difference":mean(diffs),"cluster_ci95_low":qlow,"cluster_ci95_high":qhigh,
              "cluster_signflip_p_raw":p,"holm_p_10_comparisons":"", "wins_left":pos,"wins_right":neg,"ties":ties,
              "median_delta":float(np.median(diffs)),"p10_delta":percentile(diffs,.1),"p90_delta":percentile(diffs,.9),
              "mean_gain_when_left_wins":mean(diffs[diffs>eps]) if pos else 0.,
              "mean_loss_when_right_wins":mean(diffs[diffs < -eps]) if neg else 0.})
            if title.startswith("Pairwise Expected - RoBERTa256"):
                for (seed,q),delta in pairs.items():
                    detail.append({"seed":seed,"query_id":q,"metric":metric,"pairwise_expected":results[left][seed,q][metric],
                       "roberta256_expected":results[right][seed,q][metric],"pairwise_minus_roberta":delta})
    adj=holm_bonferroni([r["cluster_signflip_p_raw"] for r in significance])
    for row,p in zip(significance,adj):row["holm_p_10_comparisons"]=p
    write_csv(out/"r2_9_cluster_significance_holm.csv",significance)
    write_csv(out/"r2_10_pairwise_vs_roberta_query_deltas.csv",detail)
    cmp={row["metric"]:row for row in significance if row["contrast"]=="Pairwise Expected - RoBERTa256 Expected"}
    with (out/"summary_readme.md").open("w",encoding="utf-8") as f:
        f.write("# Query-clustered significance tests\n\n")
        f.write("Original B-to-A, ten repeated 238/35/70 query splits. 700 query-seed units are only 315 distinct queries.\n")
        f.write("No retraining or new judge labels. Resampling flips signs at the unique query-ID level.\n\n")
        for m in ("ndcg@5","err@5"):
            r=cmp[m]
            f.write(f"## Pairwise Expected minus RoBERTa256 Expected: {m}\n")
            f.write(f"Mean difference {r['mean_difference']:+.6f}; cluster 95% CI [{r['cluster_ci95_low']:+.6f}, {r['cluster_ci95_high']:+.6f}].\n")
            f.write(f"Pairwise wins {r['wins_left']}/700, RoBERTa wins {r['wins_right']}/700, ties {r['ties']}. Median {r['median_delta']:+.6f}; 10th and 90th percentiles [{r['p10_delta']:+.6f}, {r['p90_delta']:+.6f}].\n")
            f.write(f"When Pairwise wins mean margin {r['mean_gain_when_left_wins']:+.6f}; when RoBERTa wins mean margin {r['mean_loss_when_right_wins']:+.6f}.\n")
            f.write(f"Cluster signflip p {r['cluster_signflip_p_raw']:.6g}; Holm-adjusted p across ten comparisons {r['holm_p_10_comparisons']:.6g}.\n\n")
        f.write("## Reporting constraints\n")
        f.write("Quantifies which query-seed test cases differ, NOT the changed top-1/top-5 story IDs. That requires matched candidate-level score exports for both methods, which this run does not assume exist.\n")
        f.write("The ten comparisons include 2 supervision contrasts x 4 metrics, plus Pairwise-vs-RoBERTa on two top-k metrics. Holm correction is exploratory familywise control, not validation of all modeling assumptions.\n")
        f.write("Cluster bootstrap and cluster sign-flip account for repeated held-out query IDs but not shared training-set dependencies or judge subjectivity.\n")
    print((out/"summary_readme.md").read_text(encoding="utf-8"))
    print("OUTPUT:",out)

if __name__=="__main__":main()
