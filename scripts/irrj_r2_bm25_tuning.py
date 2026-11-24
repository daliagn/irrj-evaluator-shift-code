#!/usr/bin/env python3
"""Validation-tuned query-local BM25 without evaluator leakage.

Use the *exact* original BM25 tokenization, local IDF formula, query construction,
and stable-jitter ranking. Select (k1,b) by Judge B mean nDCG@5 on 35 validation
queries separately for each seed. Only then evaluate default and tuned BM25 on
70 held-out test queries with Judge A. Does not modify the original corpus.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

K1_GRID = (0.2, 0.6, 1.0, 1.2, 1.5, 2.0, 3.0)
B_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT = (1.2, 0.75)
METHOD = 'bm25'  # Match the original stable-jitter tie-break exactly.


def tokenize(text):
    return re.findall(r'[a-zA-Z0-9]+', (text or '').lower())


def stable_jitter(seed, query_id, story_id):
    h = hashlib.md5(f'{seed}|{query_id}|{story_id}|{METHOD}'.encode('utf-8')).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def load_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def verify_seed(root, seed):
    directory = root / 'data/splits_q70' / f'seed_{seed}'
    sets = {}
    for part, required in [('train', 238), ('val', 35), ('test', 70)]:
        p = directory / f'{part}.jsonl'
        if part == 'val' and not p.is_file():
            p = directory / 'dev.jsonl'
        if not p.is_file():
            raise FileNotFoundError(str(p))
        rows = load_jsonl(p)
        qids = {str(r['query_id']) for r in rows}
        if len(qids) != required:
            raise ValueError(f'seed={seed} {part}: expected {required} unique queries; got {len(qids)}')
        sets[part] = (rows, qids)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if sets[a][1] & sets[b][1]:
            raise ValueError(f'seed {seed}: query leakage between {a} and {b}')
    for part in ['val', 'test']:
        for row in sets[part][0]:
            keys = ('age', 'moral', 'culture', 'query_id', 'story_id', 'story', 'yB_expected', 'yA_expected')
            if any(k not in row for k in keys):
                raise ValueError(f'seed={seed}: missing {part} field(s)')
    return sets


def as_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r['query_id'])].append(r)
    return groups


class QueryBM25:
    def __init__(self, rows):
        self.rows = rows
        self.docs = [tokenize(r.get('story', '')) for r in rows]
        self.n_docs = len(self.docs)
        self.df = Counter()
        self.tfs = []
        for doc in self.docs:
            self.df.update(set(doc))
            self.tfs.append(Counter(doc))
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.lengths) / max(1, self.n_docs)
        first = rows[0]
        query_text = (f"{first.get('age', '')} {first.get('moral', '')} "
                      f"{first.get('culture', '')} cultural relevance children story")
        # Retain original duplicate query tokens; original sums each occurrence.
        self.q_terms = tokenize(query_text)
        self.idfs = {
            t: math.log(1 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for t in set(self.q_terms)
        }

    def scores(self, k1, b):
        result = []
        for tf, dl in zip(self.tfs, self.lengths):
            score = 0.0
            for term in self.q_terms:
                if term not in tf:
                    continue
                denom = tf[term] + k1 * (1 - b + b * dl / max(self.avgdl, 1e-9))
                score += self.idfs[term] * (tf[term] * (k1 + 1)) / denom
            result.append(score)
        return result


def ranking_metrics(rows, scores, seed, label):
    order = sorted(range(len(scores)), key=lambda i: (
        float(scores[i]), stable_jitter(seed, rows[i]['query_id'], rows[i]['story_id'])),
        reverse=True)
    rel = [float(r[label]) for r in rows]
    idcg = sum((2 ** r - 1) / math.log2(rank+2)
               for rank, r in enumerate(sorted(rel, reverse=True)[:5]))
    dcg = sum((2 ** rel[i] - 1) / math.log2(rank+2)
              for rank, i in enumerate(order[:5]))
    ndcg = 0.0 if idcg == 0 else dcg / idcg
    err, remaining = 0.0, 1.0
    for rank, idx in enumerate(order[:5], start=1):
        R = max(0.0, min(1.0, ((2 ** rel[idx]) - 1) / (2 ** 5)))
        err += remaining * R / rank
        remaining *= 1 - R
    return ndcg, err


def evaluate_groups(qgroups, params, seed, judge_field):
    out = []
    k1, b = params
    for qid, rows in qgroups.items():
        scores = QueryBM25(rows).scores(k1, b)
        ndcg, err = ranking_metrics(rows, scores, seed, judge_field)
        out.append((qid, len(rows), ndcg, err))
    return out


def tune_params(val_groups, seed):
    # Precompute unsupervised candidate-pool text statistics once per validation query.
    prepared = [(qid, QueryBM25(rows)) for qid, rows in val_groups.items()]
    grid = []
    for k1 in K1_GRID:
        for b in B_GRID:
            values = []
            for qid, info in prepared:
                score = info.scores(k1, b)
                ndcg, err = ranking_metrics(info.rows, score, seed, 'yB_expected')
                values.append((ndcg, err))
            grid.append({'seed': seed, 'k1': k1, 'b': b,
                         'validation_B_ndcg@5': float(np.mean([x[0] for x in values])),
                         'validation_B_err@5': float(np.mean([x[1] for x in values]))})
    # Deterministic tie-breaking: closest to default, then lower parameters.
    ranked = sorted(grid, key=lambda x: (-x['validation_B_ndcg@5'],
                    abs(x['k1']-DEFAULT[0])+abs(x['b']-DEFAULT[1]), x['k1'], x['b']))
    return (ranked[0]['k1'], ranked[0]['b']), grid


def read_bm25_reference(root, seed):
    opts = [
        root / 'verified/reports/baselines_q70_tiesafe/baseline_per_query_q70_tiesafe.csv',
        root / 'frozen/v1_generation_judging_46953/verified_baseline_per_query_q70_tiesafe.csv',
    ]
    for p in opts:
        if not p.is_file():
            continue
        with p.open(newline='', encoding='utf-8') as f:
            data = [r for r in csv.DictReader(f)
                    if int(r['seed']) == seed and r.get('method','').lower() == 'bm25']
        if len(data) == 70:
            return p, {r['query_id']: float(r['ndcg@5']) for r in data}
    return None, None


def write_csv(path, rows):
    if not rows:
        raise ValueError('No data to write')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def train_seed(root, out, seed):
    partitions = verify_seed(root, seed)
    val_groups = as_groups(partitions['val'][0])
    test_groups = as_groups(partitions['test'][0])
    params, grid = tune_params(val_groups, seed)
    default_val = next(x for x in grid if (x['k1'],x['b'])==DEFAULT)
    tuned_val = next(x for x in grid if (x['k1'],x['b'])==params)
    print(f'SEED {seed}: best k1={params[0]} b={params[1]}; '
          f'B validation nDCG@5={tuned_val["validation_B_ndcg@5"]:.6f}; '
          f'default={default_val["validation_B_ndcg@5"]:.6f}', flush=True)
    write_csv(out / f'bm25_grid_seed{seed}.csv', grid)
    prepared_test = [(qid, QueryBM25(rows)) for qid,rows in test_groups.items()]
    results = []
    for name, p in [('default',DEFAULT), ('tuned',params)]:
        for qid,info in prepared_test:
            scores = info.scores(*p)
            ndcg, err = ranking_metrics(info.rows, scores, seed, 'yA_expected')
            results.append({'seed':seed, 'query_id':qid, 'arm':name,
                            'k1':p[0], 'b':p[1], 'pool_size':len(info.rows),
                            'ndcg@5':ndcg, 'err@5':err})
    if len(results)!=140:
        raise AssertionError('Expected 140 metrics across two arms')
    fixed = {r['query_id']:r['ndcg@5'] for r in results if r['arm']=='default'}
    ref_path, ref = read_bm25_reference(root, seed)
    audit = {'reference_path':str(ref_path) if ref_path else None,
             'reference_available':ref is not None}
    if ref is not None:
        if set(ref)!=set(fixed):
            raise ValueError(f'SEED {seed}: reference query IDs mismatch')
        diffs = [fixed[q]-ref[q] for q in fixed]
        audit.update({'max_absolute_delta':float(np.max(np.abs(diffs))),
                      'mean_absolute_delta':float(np.mean(np.abs(diffs))),
                      'mean_signed_delta':float(np.mean(diffs))})
        print(f'SEED {seed}: original BM25 default max abs delta '
              f'{audit["max_absolute_delta"]:.9g}', flush=True)
        if audit['max_absolute_delta'] > 0.001:
            raise AssertionError('Original BM25 reproduction failed: do not use tuning results')
    else:
        print(f'SEED {seed}: original BM25 reference CSV unavailable; aggregate must check mean.', flush=True)
    write_csv(out / f'bm25_tuning_seed{seed}.csv', results)
    manifest = {'seed':seed, 'validation_queries':len(val_groups),
                'test_queries':len(test_groups), 'grid_size':len(grid),
                'selection_metric':'Judge B validation nDCG@5',
                'test_metric':'Judge A test nDCG@5 and ERR@5',
                'default_params':list(DEFAULT), 'selected_params':list(params),
                'B_validation_ndcg@5_default':default_val['validation_B_ndcg@5'],
                'B_validation_ndcg@5_tuned':tuned_val['validation_B_ndcg@5'],
                'A_test_ndcg@5_default':float(np.mean([r['ndcg@5'] for r in results if r['arm']=='default'])),
                'A_test_ndcg@5_tuned':float(np.mean([r['ndcg@5'] for r in results if r['arm']=='tuned'])),
                'A_test_err@5_default':float(np.mean([r['err@5'] for r in results if r['arm']=='default'])),
                'A_test_err@5_tuned':float(np.mean([r['err@5'] for r in results if r['arm']=='tuned'])),
                'reference_audit':audit}
    (out / f'bm25_tuning_seed{seed}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def cluster_ci(rows, metric, draws, rng):
    bas = {(int(r['seed']),r['query_id']):float(r[metric]) for r in rows if r['arm']=='default'}
    tuned = {(int(r['seed']),r['query_id']):float(r[metric]) for r in rows if r['arm']=='tuned'}
    if set(bas)!=set(tuned):
        raise ValueError('Different held-out query sets for tuned/default')
    keys = sorted(bas)
    diffs = np.array([tuned[k]-bas[k] for k in keys])
    qids = sorted({q for _,q in keys})
    buckets = defaultdict(list)
    for (_,q),d in zip(keys,diffs):
        buckets[q].append(d)
    sums = np.asarray([sum(buckets[q]) for q in qids]); counts = np.asarray([len(buckets[q]) for q in qids])
    idx = rng.integers(0, len(qids), size=(draws,len(qids)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo,hi = np.quantile(boots,[0.025,0.975])
    signs = rng.choice([-1.0,1.0], size=(draws,len(qids)))
    rand = abs(np.mean(diffs))
    signflips = np.abs((signs*sums).sum(axis=1)/len(diffs))
    p = (1+int((signflips>=rand).sum()))/(draws+1)
    return float(np.mean(diffs)), float(lo), float(hi), float(p), int((diffs>0).sum()), int((diffs<0).sum()), int((diffs==0).sum())


def aggregate(root, out, draws):
    files = [out/f'bm25_tuning_seed{i}.csv' for i in range(10)]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError('Missing seed files: '+', '.join(missing))
    rows = []
    manifests = []
    for i,f in enumerate(files):
        with f.open(newline='', encoding='utf-8') as h:
            sub = list(csv.DictReader(h))
        if len(sub)!=140:
            raise ValueError(f'{f.name}: expected 140 rows')
        rows.extend(sub)
        manifests.append(json.loads((out/f'bm25_tuning_seed{i}_manifest.json').read_text(encoding='utf-8')))
    if len({r['query_id'] for r in rows})!=315:
        raise ValueError('Expected 315 unique test queries across ten seeds')
    result = {}
    for arm in ('default','tuned'):
        sub = [r for r in rows if r['arm']==arm]
        if len(sub)!=700:
            raise ValueError('Expected 700 observations per arm')
        for metric in ('ndcg@5','err@5'):
            result[f'{arm}_{metric}'] = float(np.mean([float(r[metric]) for r in sub]))
    rng = np.random.default_rng(20261024)
    for metric in ('ndcg@5','err@5'):
        d,lo,hi,p,w,l,t = cluster_ci(rows,metric,draws,rng)
        result[f'delta_tuned_minus_default_{metric}'] = d
        result[f'delta_CI95_{metric}'] = [lo,hi]
        result[f'cluster_signflip_p_{metric}'] = p
        result[f'tuned_wins_{metric}'] = w
        result[f'default_wins_{metric}'] = l
        result[f'ties_{metric}'] = t
    combos = Counter((m['selected_params'][0],m['selected_params'][1]) for m in manifests)
    result['selected_params_by_seed'] = {str(i):m['selected_params'] for i,m in enumerate(manifests)}
    result['parameter_selection_counts'] = {str(k):v for k,v in sorted(combos.items())}
    result['n_query_seed'] = 700
    result['n_unique_test_queries'] = 315
    result['validation_selection']='Judge B only; one (k1,b) pair per seed'
    result['grid']={'k1':list(K1_GRID),'b':list(B_GRID)}
    audits = [m['reference_audit'] for m in manifests if m['reference_audit']['reference_available']]
    # Also independently check an original baseline CSV found at aggregation time.
    if len(audits)<10:
        for seed in range(10):
            if manifests[seed]['reference_audit']['reference_available']:
                continue
            ref_path, ref = read_bm25_reference(root, seed)
            if ref is None:
                continue
            observed={r['query_id']:float(r['ndcg@5']) for r in rows
                      if int(r['seed'])==seed and r['arm']=='default'}
            if set(ref)!=set(observed):
                raise ValueError(f'Seed {seed} reference query mismatch at aggregation')
            diffs=[observed[q]-ref[q] for q in observed]
            maxdiff=float(np.max(np.abs(diffs)))
            if maxdiff>0.001:
                raise AssertionError(f'Seed {seed} original BM25 audit failed: max delta {maxdiff}')
            audits.append({'max_absolute_delta':maxdiff, 'reference_path':str(ref_path)})
    result['original_reference_checked_seeds'] = len(audits)
    if audits:
        result['original_reference_max_absolute_delta']=max(a['max_absolute_delta'] for a in audits)
    else:
        if abs(result['default_ndcg@5']-0.5915) > 0.015:
            raise ValueError('BM25 default differs materially from paper baseline; inspect before use')
    (out/'bm25_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text = [
        '# Validation-tuned BM25',
        'Equation matches original source: local query-pool IDF log(1+(N-df+0.5)/(df+0.5));',
        'TF saturation (tf*(k1+1))/(tf+k1*(1-b+b*len/avglen)); repeated query terms retained.',
        'Original baseline k1=1.2 b=0.75. Tuned using Judge B validation nDCG@5, 35 queries/seed.',
        'Test metrics use only Judge A held-out graded relevance; no A used for selection.',
        f'Grid: k1={list(K1_GRID)}; b={list(B_GRID)}; 10 query resplits; 700 cases /315 unique queries.',
        f'Original CSV reference validated on {len(audits)} seeds.',
        '',
        f'Default Judge A nDCG@5 = {result["default_ndcg@5"]:.6f}',
        f'Tuned Judge A nDCG@5 = {result["tuned_ndcg@5"]:.6f}',
        f'Delta = {result["delta_tuned_minus_default_ndcg@5"]:+.6f}',
        f'95% query-cluster CI = {result["delta_CI95_ndcg@5"]}',
        f'Cluster-signflip p = {result["cluster_signflip_p_ndcg@5"]:.6f}',
        f'Default Judge A ERR@5 = {result["default_err@5"]:.6f}',
        f'Tuned Judge A ERR@5 = {result["tuned_err@5"]:.6f}',
        f'Delta = {result["delta_tuned_minus_default_err@5"]:+.6f}',
        f'95% query-cluster CI = {result["delta_CI95_err@5"]}',
        'Tuning parameters by seed: '+str(result['selected_params_by_seed']),
    ]
    (out/'summary_readme.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with zipfile.ZipFile(out/'r2_bm25_share.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out/'summary_readme.md','summary_readme.md')
        z.write(out/'bm25_summary.json','bm25_summary.json')
        for i in range(10):
            z.write(out/f'bm25_tuning_seed{i}_manifest.json', f'manifests/bm25_tuning_seed{i}_manifest.json')
        for i in range(10):
            z.write(out/f'bm25_grid_seed{i}.csv', f'validation_grids/bm25_grid_seed{i}.csv')
    print((out/'summary_readme.md').read_text(), flush=True)
    print('SHARE:', out/'r2_bm25_share.zip', flush=True)


def selftest():
    # Check vector-free implementation against the verbatim original BM25 formula.
    rows = [
        {'age':'9','moral':'kindness','culture':'South Asian','query_id':'a','story_id':str(i),
         'story':s,'yB_expected':'3.0','yA_expected':'3.2'}
        for i,s in enumerate(['Kindness story about friends, cultural kindness.',
                              'A story story about kindness and children in South Asian cities',
                              'Completely unrelated sentence'])]
    info=QueryBM25(rows)
    for k,b in [(1.2,.75),(.2,0),(3.0,1.0)]:
        docs=[tokenize(r.get('story','')) for r in rows]
        n=len(docs)
        df=Counter()
        for d in docs:
            for t in set(d):df[t]+=1
        avgdl=sum(len(d) for d in docs)/max(1,n)
        first=rows[0]
        q=tokenize(f"{first.get('age','')} {first.get('moral','')} {first.get('culture','')} cultural relevance children story")
        expect=[]
        for d in docs:
            tf=Counter(d);dl=len(d);score=0
            for term in q:
                if term not in tf:continue
                idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
                den=tf[term]+k*(1-b+b*dl/max(avgdl,1e-9))
                score+=idf*(tf[term]*(k+1))/den
            expect.append(score)
        got=info.scores(k,b)
        assert all(abs(a-b)<1e-12 for a,b in zip(expect,got))
    print('PASS selftest: original BM25 formula and stable tokenization matched')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--seeds',default='0',help='0..9, comma-separated, or all')
    ap.add_argument('--output-dir',default='outputs/irrj_r2_bm25_tuning')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--aggregate',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=20000)
    args=ap.parse_args()
    selftest()
    if args.selftest:return
    root=Path(args.root).resolve()
    out=Path(args.output_dir)
    if not out.is_absolute():out=root/out
    if args.aggregate:
        aggregate(root,out,args.bootstrap)
        return
    seeds=list(range(10)) if args.seeds=='all' else [int(s) for s in args.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or not seeds or any(s<0 or s>9 for s in seeds):
        raise ValueError('Provide unique seed indices in 0..9')
    for seed in seeds:
        v=verify_seed(root,seed)
        print(f'PASS seed={seed}: train/val/test query counts 238/35/70; '
              f'validation rows {len(v["val"][0])}; test rows {len(v["test"][0])}')
    if args.check_only:
        print('CHECK-ONLY: no tuning, no writing, no GPU')
        return
    out.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        train_seed(root,out,seed)

if __name__=='__main__':main()
