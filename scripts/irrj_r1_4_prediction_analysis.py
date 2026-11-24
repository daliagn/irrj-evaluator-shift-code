#!/usr/bin/env python3
"""Query outcomes and saved-prediction analysis; no model inference.

Primary: B-trained, A-evaluated hard-vs-expected results from original IRRJ
RoBERTa-256 and pairwise SGD (per-query nDCG/ERR and query-cluster CIs).
Supplemental: A-trained RoBERTa-512 expected-vs-hard saved item predictions,
actual selected top-1/top-5 changes, per-query outcomes, and entropy strata.
Original frozen labels only; does not modify inputs or rerun models.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("ndcg@5", "err@5", "spearman", "kendall_tau_b")
PRIMARY_OPTIONS = {
    "pairwise_expected": [
        "frozen/v1_generation_judging_46953/verified_pairwise_tfidf_sgd_yB_expected_per_query_q70.csv",
        "verified/reports/pairwise_tfidf_sgd_q70/pairwise_tfidf_sgd_yB_expected_per_query_q70.csv",
    ],
    "pairwise_hard": [
        "frozen/v1_generation_judging_46953/verified_pairwise_tfidf_sgd_yB_hard_per_query_q70.csv",
        "verified/reports/pairwise_tfidf_sgd_hard_q70/pairwise_tfidf_sgd_yB_hard_per_query_q70.csv",
    ],
    "roberta256_expected": [
        "frozen/v1_generation_judging_46953/verified_roberta_large_pointwise_yB_expected_per_query_q70.csv",
        "verified/reports/roberta_large_pointwise_q70/roberta_large_pointwise_yB_expected_per_query_q70.csv",
    ],
    "roberta256_hard": [
        "frozen/v1_generation_judging_46953/verified_roberta_large_pointwise_yB_hard_per_query_q70.csv",
        "verified/reports/roberta_large_pointwise_hard_q70/roberta_large_pointwise_yB_hard_per_query_q70.csv",
    ],
}

DEFAULT_PATHS = {
    "expected_item": "outputs/irrj_evaluator_2x2_full/item_scores.csv",
    "expected_query": "outputs/irrj_evaluator_2x2_full/per_query.csv",
    "hard_item": "outputs/irrj_reverse_expected_vs_hard_full/item_scores_hard.csv",
    "hard_query": "outputs/irrj_reverse_expected_vs_hard_full/per_query_hard.csv",
    "entropy_query": "outputs/irrj_r1_2_judge_entropy/per_query_entropy.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/irrj_r1_4_prediction_analysis"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20261022)
    for name in PRIMARY_OPTIONS:
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       help="Override auto-located original IRRJ B-to-A per-query CSV")
    for name, default in DEFAULT_PATHS.items():
        p.add_argument("--"+name.replace("_", "-"), type=Path, default=Path(default))
    return p.parse_args()


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required saved-results file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, rows):
    if not rows:
        raise ValueError(f"No data for output {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def k_query(row):
    return int(row["seed"]), str(row["query_id"])


def k_item(row):
    return int(row["seed"]), str(row["query_id"]), str(row["story_id"])


def item_index(rows, expected):
    scores = {}
    pools = defaultdict(dict)
    for r in rows:
        if expected and r["training_judge"] != "A":
            continue
        key = k_item(r)
        if key in scores:
            raise ValueError(f"Duplicate story prediction: {key}, expected={expected}")
        score = float(r["score"] if expected else r["hard_score"])
        yA, yB = float(r["yA_expected"]), float(r["yB_expected"])
        if not all(math.isfinite(x) for x in (score,yA,yB)):
            raise ValueError(f"Nonfinite item score: {key}")
        scores[key] = (score,yA,yB)
        pools[key[:2]][key[2]] = (score,yA,yB)
    return scores, pools


def query_index(rows, expected):
    out = {}
    for r in rows:
        if r.get("training_judge") != "A":
            continue
        if r["evaluation_judge"] not in ("A", "B"):
            raise ValueError(f"Unexpected evaluation judge: {r['evaluation_judge']}")
        key = (r["evaluation_judge"], *k_query(r))
        if key in out:
            raise ValueError(f"Duplicate query evaluation: {key}")
        out[key] = {m: float(r[m]) for m in METRICS}
        if not all(math.isfinite(v) for v in out[key].values()):
            raise ValueError(f"Nonfinite query metrics: {key}")
    return out


def entropy_index(rows):
    out = {}
    fields = ("entropy_A_mean", "entropy_B_mean", "hard_tie_pair_share_A",
              "hard_tie_pair_share_B", "judge_hard_disagreement_share",
              "judge_expected_abs_diff_mean")
    for r in rows:
        q = str(r["query_id"])
        if q in out:
            raise ValueError(f"Duplicate unique query entropy: {q}")
        out[q] = {f:float(r[f]) for f in fields}
    return out


def rank(p):
    # Descriptive top-k extraction. Stable story-ID tiebreak; published per-query
    # nDCG/ERR are read from the verified evaluation, not recomputed here.
    return sorted(p, key=lambda sid: (-p[sid][0],sid))


def mean(vals):
    return statistics.fmean(vals)


def descriptive_stats(rows, judge):
    rs = [r for r in rows if r["evaluation_judge"] == judge]
    deltas = [r["delta_ndcg@5"] for r in rs]
    winners = [x for x in deltas if x > 1e-12]
    losers = [x for x in deltas if x < -1e-12]
    keys = {r["query_id"] for r in rs}
    if len(rs) != 700 or len(keys) != 315:
        raise ValueError(f"Unexpected query counts in judge {judge}: {len(rs)} rows, {len(keys)} unique")
    return {
        "evaluation_judge": judge,
        "n_query_seed": len(rs),
        "n_unique_queries": len(keys),
        "expected_wins_ndcg_count": len(winners),
        "hard_wins_ndcg_count": len(losers),
        "equal_ndcg_count": len(deltas) - len(winners) - len(losers),
        "expected_wins_ndcg_fraction": len(winners)/len(rs),
        "hard_wins_ndcg_fraction": len(losers)/len(rs),
        "mean_delta_ndcg":mean(deltas),
        "mean_delta_err":mean([r["delta_err@5"] for r in rs]),
        "changed_top1_fraction":mean([r["top1_changed"] for r in rs]),
        "mean_top5_overlap_fraction":mean([r["top5_overlap_fraction"] for r in rs]),
        "mean_abs_top1_relevance_difference":mean([abs(r["top1_relevance_delta"]) for r in rs]),
        "mean_top1_relevance_difference":mean([r["top1_relevance_delta"] for r in rs]),
        "changed_top1_and_expected_win_fraction":mean([r["top1_changed"] and r["delta_ndcg@5"]>1e-12 for r in rs]),
    }


def clustered_cis(rows, fields, draws, seed):
    if not draws:
        return {f:("","") for f in fields}
    import numpy as np
    ids = sorted({r["query_id"] for r in rows})
    id_to_index = {q:i for i,q in enumerate(ids)}
    numerator = {f:np.zeros(len(ids)) for f in fields}
    denom = np.zeros(len(ids))
    for r in rows:
        i = id_to_index[r["query_id"]]
        denom[i] += 1
        for f in fields:
            numerator[f][i] += float(r[f])
    rng = np.random.default_rng(seed)
    draws_out = {f:[] for f in fields}
    for start in range(0,draws,1000):
        n = min(1000,draws-start)
        indices = rng.integers(0,len(ids),size=(n,len(ids)))
        weights = denom[indices].sum(axis=1)
        for f in fields:
            vals = numerator[f][indices].sum(axis=1) / weights
            draws_out[f].append(vals)
    return {f:tuple(float(x) for x in np.quantile(np.concatenate(draws_out[f]),[0.025,0.975])) for f in fields}




def locate_primary_sources(root, opts):
    resolved = {}
    for name, choices in PRIMARY_OPTIONS.items():
        explicit = getattr(opts, name)
        if explicit is not None:
            fp = explicit if explicit.is_absolute() else root / explicit
            if not fp.is_file():
                raise FileNotFoundError(f"Primary source supplied for {name} does not exist: {fp}")
        else:
            matches = [root / item for item in choices if (root / item).is_file()]
            if not matches:
                raise FileNotFoundError(f"Missing main-paper {name} per-query file. Searched: {choices}. "
                                        f"Use --{name.replace('_','-')} to pass an exact path.")
            fp = matches[0]
        resolved[name] = fp.resolve()
    return resolved


def primary_index(rows, source):
    out = {}
    for r in rows:
        try:
            key = k_query(r)
            vals = {m: float(r[m]) for m in METRICS}
        except KeyError as exc:
            raise ValueError(f"Unexpected per-query columns in {source}: {sorted(r)}") from exc
        if key in out:
            raise ValueError(f"Duplicate (seed,query_id) row in {source}: {key}")
        if not all(math.isfinite(v) for v in vals.values()):
            raise ValueError(f"Nonfinite metric in {source}: {key}")
        out[key] = vals
    if len(out)!=700 or len({q for _,q in out})!=315:
        raise ValueError(f"Wrong main-paper q70 observation count in {source}: "
                         f"{len(out)} query-seed rows, {len({q for _,q in out})} unique queries")
    return out


def primary_outcomes(primary_data, bootstrap, bootstrap_seed):
    # Compare exactly matched query-seed rows rather than pretending the repeated
    # splits are independent. Effect-size inference uses unique-query clusters.
    detailed, summaries = [], []
    for name in ("pairwise", "roberta256"):
        exp = primary_data[name+"_expected"]
        hard = primary_data[name+"_hard"]
        if set(exp)!=set(hard):
            raise ValueError(f"Original {name} per-query keys do not match across hard/expected")
        rows=[]
        for seed,qid in sorted(exp):
            er,hr=exp[(seed,qid)],hard[(seed,qid)]
            delta={f"delta_{m}":er[m]-hr[m] for m in METRICS}
            row={"method":name,"seed":seed,"query_id":qid,
                 "expected_ndcg@5":er["ndcg@5"],"hard_ndcg@5":hr["ndcg@5"],
                 "expected_err@5":er["err@5"],"hard_err@5":hr["err@5"],**delta}
            rows.append(row)
        detailed.extend(rows)
        for metric in METRICS:
            field=f"delta_{metric}"
            vals=[r[field] for r in rows]
            cis=clustered_cis(rows,(field,),bootstrap,bootstrap_seed)[field]
            summaries.append({"method":name,"metric":metric,"n_query_seed":700,"n_unique_queries":315,
                              "expected_mean":mean([exp[k][metric] for k in exp]),
                              "hard_mean":mean([hard[k][metric] for k in hard]),
                              "mean_delta":mean(vals),
                              "expected_win_count":sum(v>1e-12 for v in vals),
                              "hard_win_count":sum(v < -1e-12 for v in vals),
                              "ties_count":sum(abs(v)<=1e-12 for v in vals),
                              "cluster_ci95_low":cis[0],"cluster_ci95_high":cis[1],
                              "bootstrap_draws":bootstrap})
    return detailed,summaries

def main():
    opts = parse_args()
    if opts.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative")
    root = opts.root.resolve()
    paths = {name:(p if p.is_absolute() else root/p).resolve() for name,p in
             ((n,getattr(opts,n)) for n in DEFAULT_PATHS)}
    data = {n:load(p) for n,p in paths.items()}
    primary_paths=locate_primary_sources(root, opts)
    primary={n:primary_index(load(fp),fp) for n,fp in primary_paths.items()}
    if len({tuple(sorted(v)) for v in primary.values()}) != 1:
        raise ValueError("Original main-paper result sets are not aligned on identical query/seed pairs")
    published_ndcg = {
        "pairwise_expected":0.7293,"pairwise_hard":0.6909,
        "roberta256_expected":0.7200,"roberta256_hard":0.6911,
    }
    for name, expected_mean in published_ndcg.items():
        observed=mean([v["ndcg@5"] for v in primary[name].values()])
        if abs(observed-expected_mean)>0.003:
            raise ValueError(f"Main-paper source {name} gives nDCG@5 {observed:.5f}, "
                             f"not near published {expected_mean:.4f}. Check file and score rubric.")
    eitems, epools = item_index(data["expected_item"],True)
    hitems, hpools = item_index(data["hard_item"],False)
    if set(eitems) != set(hitems):
        missing_e = list(set(hitems)-set(eitems))[:3]
        missing_h = list(set(eitems)-set(hitems))[:3]
        raise ValueError(f"Expected/hard story mismatch, missing_expected={missing_e}, missing_hard={missing_h}")
    for key in eitems:
        if any(abs(x-y)>2e-6 for x,y in zip(eitems[key][1:],hitems[key][1:])):
            raise ValueError(f"Original rubric evaluator labels differ for {key}")
    eq = query_index(data["expected_query"],True)
    hq = query_index(data["hard_query"],False)
    if set(eq) != set(hq):
        raise ValueError(f"Expected/hard per-query evaluation keys mismatch: expected={len(eq)}, hard={len(hq)}")
    qe = entropy_index(data["entropy_query"])
    if len(qe) != 343:
        raise ValueError(f"Expected 343 frozen query-entropy records; got {len(qe)}")
    if len(epools) != 700 or len({q for _,q in epools}) != 315:
        raise ValueError(f"Wrong number of query-seed pools: {len(epools)}")
    if set(epools) != set(hpools):
        raise ValueError("Mismatch in expected/hard candidate-pool membership")
    for judge in ("A","B"):
        if {(s,q) for ev,s,q in eq if ev==judge} != set(epools):
            raise ValueError(f"Missing {judge} evaluation result for some pools")
    if any(len(pool)<5 for pool in epools.values()):
        raise ValueError("Fewer than 5 candidates in a test pool")
    if not {q for _,q in epools}.issubset(qe):
        raise ValueError("Not all held-out queries are in original-rubric entropy output")
    seeds = Counter(seed for seed,_ in epools)
    if dict(seeds) != {s:70 for s in range(10)}:
        raise ValueError(f"Unexpected per-seed query counts: {dict(seeds)}")
    print(f"PASS: Main IRRJ B-to-A per-query results: 700 matched query-seed observations for each of Pairwise and RoBERTa256, Expected and Hard")
    for name,p in primary_paths.items():
        print(f"PRIMARY {name}: {p.relative_to(root) if p.is_relative_to(root) else p}")
    print(f"PASS: {len(epools)} matched RoBERTa512 test query pools, 315 unique queries; {len(eitems)} paired A-trained story predictions")
    print("PASS: 10 seeds, original Judge A/B relevance labels match, per-query evaluation keys aligned")
    print(f"PASS: original uncertainty rows cover {len(qe)} unique queries; no model inference required")
    if opts.check_only:
        print("CHECK-ONLY: no output directory created")
        return
    out = opts.output_dir if opts.output_dir.is_absolute() else root / opts.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    per_pool={}
    for (seed,qid),pe in sorted(epools.items()):
        ph = hpools[(seed,qid)]
        if pe.keys()!=ph.keys():
            raise ValueError(f"Story pool mismatch at {seed},{qid}")
        re, rh = rank(pe), rank(ph)
        a, b = re[0], rh[0]
        top5e, top5h = set(re[:5]), set(rh[:5])
        poolbase = {
            "seed":seed,"query_id":qid,"n_candidates":len(pe),
            "expected_top1_story_id":a,"hard_top1_story_id":b,
            "top1_changed":int(a!=b),
            "top5_shared_count":len(top5e & top5h),
            "top5_overlap_fraction":len(top5e & top5h)/5,
            "A_expected_top1_label_A":pe[a][1],
            "A_hard_top1_label_A":ph[b][1],
            "A_expected_top1_label_B":pe[a][2],
            "A_hard_top1_label_B":ph[b][2],
            **qe[qid],
        }
        per_pool[(seed,qid)] = poolbase
    rows=[]
    for judge in ("A","B"):
        for (seed,qid),base in sorted(per_pool.items()):
            delta = {f"delta_{m}":eq[(judge,seed,qid)][m] - hq[(judge,seed,qid)][m] for m in METRICS}
            lab = judge
            row = {
                "evaluation_judge":judge, **base,
                "expected_ndcg@5":eq[(judge,seed,qid)]["ndcg@5"],
                "hard_ndcg@5":hq[(judge,seed,qid)]["ndcg@5"],
                **delta,
                "top1_relevance_delta":base[f"A_expected_top1_label_{lab}"]-base[f"A_hard_top1_label_{lab}"],
            }
            rows.append(row)
    stats = [descriptive_stats(rows,j) for j in ("A","B")]
    main_rows,main_summaries=primary_outcomes(primary,opts.bootstrap,opts.bootstrap_seed+100)
    out.mkdir(parents=True)
    write(out/"main_BtoA_per_query_outcomes.csv",main_rows)
    write(out/"main_BtoA_summary.csv",main_summaries)
    write(out/"prediction_changes_per_query.csv",rows)
    write(out/"prediction_change_summary.csv",stats)
    ci_rows=[]
    for judge in ("A","B"):
        subset=[r for r in rows if r["evaluation_judge"]==judge]
        for r in subset:
            r["expected_win_indicator"]=int(r["delta_ndcg@5"]>1e-12)
            r["hard_win_indicator"]=int(r["delta_ndcg@5"]< -1e-12)
        fields=("top1_changed","top5_overlap_fraction","expected_win_indicator","hard_win_indicator","delta_ndcg@5","top1_relevance_delta")
        cis=clustered_cis(subset,fields,opts.bootstrap,opts.bootstrap_seed+(0 if judge=="A" else 1))
        for field in fields:
            lo,hi=cis[field]
            ci_rows.append({"evaluation_judge":judge,"measure":field,
                            "mean_observed":mean([r[field] for r in subset]),
                            "cluster_ci95_low":lo,"cluster_ci95_high":hi,
                            "n_unique_query_clusters":315,"bootstrap_draws":opts.bootstrap})
    write(out/"prediction_cluster_ci.csv",ci_rows)
    # Stratification on original judge uncertainty is descriptive, with fixed quartiles
    # determined at the unique-query level; it is not a causal explanation.
    strata=[]
    for judge in ("A","B"):
        qids=sorted({r["query_id"] for r in rows})
        ranked=sorted(qids,key=lambda q:(qe[q][f"entropy_{judge}_mean"],q))
        bins={qid:(i*4)//len(ranked)+1 for i,qid in enumerate(ranked)}
        for quartile in range(1,5):
            subset=[r for r in rows if r["evaluation_judge"]==judge and bins[r["query_id"]]==quartile]
            strata.append({"evaluation_judge":judge,"evaluation_judge_entropy_quartile":quartile,
                           "n_unique_queries":len({r["query_id"] for r in subset}),
                           "n_query_seed":len(subset),
                           "mean_eval_judge_entropy_bits":mean([r[f"entropy_{judge}_mean"] for r in subset]),
                           "mean_delta_ndcg@5":mean([r["delta_ndcg@5"] for r in subset]),
                           "expected_win_fraction":mean([r["delta_ndcg@5"]>1e-12 for r in subset]),
                           "top1_changed_fraction":mean([r["top1_changed"] for r in subset]),
                           "mean_top5_overlap_fraction":mean([r["top5_overlap_fraction"] for r in subset])})
    write(out/"entropy_strata_descriptive.csv",strata)
    # Show examples for transparent qualitative review, not as a cherry-picked proof.
    examples=[]
    for judge in ("A","B"):
        subset=sorted((r for r in rows if r["evaluation_judge"]==judge),
                      key=lambda r:(r["delta_ndcg@5"], r["seed"],r["query_id"]))
        for tag,chosen in (("hard_favoured",subset[:10]),("expected_favoured",subset[-10:][::-1])):
            for r in chosen:
                examples.append({"evaluation_judge":judge,"example_type":tag,"seed":r["seed"],"query_id":r["query_id"],
                                 "expected_top1_story_id":r["expected_top1_story_id"],
                                 "hard_top1_story_id":r["hard_top1_story_id"],
                                 "top1_changed":r["top1_changed"],"top5_shared_count":r["top5_shared_count"],
                                 "expected_ndcg@5":r["expected_ndcg@5"],"hard_ndcg@5":r["hard_ndcg@5"],
                                 "delta_ndcg@5":r["delta_ndcg@5"],
                                 "top1_relevance_delta":r["top1_relevance_delta"]})
    write(out/"extreme_query_examples.csv",examples)
    manifest={
        "analysis":"Saved prediction outcomes: A-expected vs A-hard RoBERTa512",
        "no_training":True,"no_gpu_inference":True,
        "original_rubric_only":True,"experiment_scope":"A-trained models evaluated using A and B, not the B-trained main-paper model",
        "source_paths_sha256":{n:{"path":str(p),"sha256":digest(p)} for n,p in {**paths,**primary_paths}.items()},
        "n_query_seed_pools":len(epools),"n_unique_test_queries":315,"n_story_predictions":len(eitems),
        "ranking_tiebreak":"(-saved_model_score, story_id) for descriptive top1/top5 selection; original verified nDCG/ERR loaded unmodified",
        "metric_units":"per-query metrics on original 1-5 expected scores; 10 repeated overlapping query splits",
        "bootstrap":"315 distinct query_id clusters reused across splits; 20,000 draws unless overridden",
        "limitations":["These are observational diagnostics, not causal mechanism tests.",
                       "Hard ties in relevance labels do not ensure better realized selection.",
                       "Findings are limited to the two evaluators studied.",
                       "A-trained models here are 512-token supplements, not original 256-token paper models."]}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report=["# Query outcomes and prediction-level analysis","",
            "## Primary B-trained / A-evaluated results (original main-paper settings)",
            "Query-level comparisons of Pairwise TF-IDF SGD and RoBERTa256 Expected vs Hard on identical test queries.",
            "No main-paper candidate-level predictions are assumed to be available. Top-1 and top-5 changes below come only from the supplemental A-trained RoBERTa512 models.",""]
    for r in main_summaries:
        if r["metric"]=="ndcg@5":
            report.append(f"{r['method']}: Expected wins {r['expected_win_count']}/700, Hard wins {r['hard_win_count']}/700, equal {r['ties_count']}; "
                          f"mean nDCG@5 difference {r['mean_delta']:+.5f}; 95% clustered CI [{r['cluster_ci95_low']:+.5f}, {r['cluster_ci95_high']:+.5f}].")
    report += ["","## Supplemental A-trained RoBERTa512 saved story predictions","",
            "A-trained expected versus A-trained hard models on identical saved test pools.",
            "Top-1 and top-5 differences use saved model scores; nDCG@5/ERR@5 come from verified per-query evaluations.","",
            "Warning: the following are diagnostic associations, not evidence of a causal mechanism.",""]
    for r in stats:
        judge=r["evaluation_judge"]
        report.extend([f"## Evaluation Judge {judge}",
                       f"Expected beats Hard on nDCG@5: {r['expected_wins_ndcg_count']}/{r['n_query_seed']} query-seed cases; "
                       f"Hard beats Expected: {r['hard_wins_ndcg_count']}; equal: {r['equal_ndcg_count']}.",
                       f"Mean nDCG@5 delta (Expected-Hard): {r['mean_delta_ndcg']:+.5f}; mean ERR@5 delta {r['mean_delta_err']:+.5f}.",
                       f"Top-1 story changed: {r['changed_top1_fraction']:.1%}; mean top-5 overlap: {r['mean_top5_overlap_fraction']:.1%}.",
                       f"Mean relevance difference at selected top-1 (Expected minus Hard): {r['mean_top1_relevance_difference']:+.4f}.",""])
    report.append("See prediction_cluster_ci.csv for 95% query-cluster intervals; entropy_strata_descriptive.csv for exploratory strata.")
    report.append("Entropy strata are descriptive and do not establish a causal explanation; repeated test units are not independent.")
    (out/"summary_readme.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))
    print(f"\nResults written to: {out}")


if __name__=="__main__":
    main()
