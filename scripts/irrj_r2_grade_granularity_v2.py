#!/usr/bin/env python3
# Grade-resolution study (Table 4, Figure 1).
# Reproduces the original q70 TF-IDF Pairwise SGD and Ridge procedures.
# No re-judging; Judge A labels are for held-out reporting only.
import argparse, csv, hashlib, json, math, os, random, sys, warnings, zipfile, platform
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = (2, 3, 5, 10, 20, 50, 100)
METRICS = ('ndcg@5', 'err@5')
ORIGINAL = {
    'pairwise': 'frozen/v1_generation_judging_46953/verified_pairwise_tfidf_sgd_yB_expected_per_query_q70.csv',
    'ridge': 'verified/reports/tfidf_ridge_q70/tfidf_ridge_yB_expected_per_query_q70.csv'
}


def jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def text_for(row):
    return (f"Age: {row.get('age', '')}\nMoral: {row.get('moral', '')}\n"
            f"Culture: {row.get('culture', '')}\nStory:\n{row.get('story', '')}")


def stable_jitter(seed, qid, sid, method):
    h = hashlib.md5(f'{seed}|{qid}|{sid}|{method}'.encode('utf-8')).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def eval_metrics(rows, scores, seed, jitter_method):
    grouped = defaultdict(list)
    for rec, score in zip(rows, scores):
        grouped[str(rec['query_id'])].append((rec, float(score)))
    results = []
    for qid, items in grouped.items():
        # Match exact original stable-jitter tie handling.
        order = sorted(range(len(items)),
                       key=lambda i: (items[i][1], stable_jitter(seed, qid, str(items[i][0]['story_id']), jitter_method)),
                       reverse=True)
        rel = [float(row['yA_expected']) for row, _ in items]
        ideal = sorted(rel, reverse=True)[:5]
        discount = [math.log2(i + 2) for i in range(5)]
        idcg = sum((2.0 ** r - 1.0) / d for r, d in zip(ideal, discount))
        dcg = sum((2.0 ** rel[i] - 1.0) / discount[k] for k, i in enumerate(order[:5]))
        ndcg = 0.0 if idcg == 0.0 else dcg / idcg
        err, p_continue = 0.0, 1.0
        for rank, i in enumerate(order[:5], 1):
            R = max(0.0, min(1.0, ((2.0 ** rel[i]) - 1.0) / (2.0 ** 5)))
            err += p_continue * R / rank
            p_continue *= 1.0 - R
        results.append({'seed': seed, 'query_id': qid, 'pool_size': len(items),
                        'ndcg@5': ndcg, 'err@5': err})
    return results


def quantize_expected(values, levels):
    # Fixed rubric range [1,5], no test-derived boundaries.
    # Midpoint/nearest-grid quantisation; grid includes endpoints.
    # Preserve source-training dtypes: Pairwise uses float32, Ridge float64.
    values = np.asarray(values)
    if levels is None:
        return values.copy()
    return (1.0 + (4.0 / (levels - 1)) * np.floor((np.clip(values, 1, 5) - 1.0)
                     * (levels - 1) / 4.0 + 0.5)).astype(values.dtype, copy=False)


def pairs_for_query(indices, labels, rng, max_pairs=1000, min_delta=0.05):
    valid = []
    # Intentionally mirrors the original 2026 code, including its min_delta rule.
    for ai in range(len(indices)):
        i = indices[ai]
        for bj in range(ai + 1, len(indices)):
            j = indices[bj]
            diff = labels[i] - labels[j]
            if diff >= min_delta:
                valid.append((i, j))
            elif diff <= -min_delta:
                valid.append((j, i))
    rng.shuffle(valid)
    if len(valid) > max_pairs:
        valid = valid[:max_pairs]
    return valid


def fit_pairwise(X_train, rows, labels, seed):
    from scipy.sparse import vstack
    from sklearn.linear_model import SGDClassifier
    grouped = defaultdict(list)
    for i, rec in enumerate(rows):
        grouped[rec['query_id']].append(i)
    rng = random.Random(seed)
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5,
                          learning_rate='optimal', random_state=seed)
    first_fit, total_pairs = True, 0
    for epoch in range(1, 4):
        qids = list(grouped.keys())
        rng.shuffle(qids)
        for qid in qids:
            pairs = pairs_for_query(grouped[qid], labels, rng)
            total_pairs += len(pairs)
            if not pairs:
                continue
            feats, targets = [], []
            def flush():
                nonlocal first_fit, feats, targets
                if not feats:
                    return
                Xb = vstack(feats).tocsr()
                yb = np.asarray(targets, dtype=np.int64)
                if first_fit:
                    model.partial_fit(Xb, yb, classes=np.asarray([0, 1], dtype=np.int64))
                    first_fit = False
                else:
                    model.partial_fit(Xb, yb)
                feats, targets = [], []
            for hi, lo in pairs:
                feats.append(X_train[hi] - X_train[lo]); targets.append(1)
                feats.append(X_train[lo] - X_train[hi]); targets.append(0)
                if len(feats) >= 4096:
                    flush()
            flush()
    if first_fit:
        raise ValueError('No valid training pairs; check labels/granularity.')
    return model, total_pairs


def read_reference(path, seed, metric):
    if not path.is_file():
        return None
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if int(r['seed']) == seed]
    return {r['query_id']: float(r[metric]) for r in rows}


def verify_split(root, seed):
    d = root / 'data' / 'splits_q70' / f'seed_{seed}'
    if not d.is_dir():
        raise FileNotFoundError(f'Split directory not found: {d}')
    sets = {}
    for part in ('train', 'val', 'test'):
        f = d / f'{part}.jsonl'
        if not f.is_file() and part == 'val':
            f = d / 'dev.jsonl'
        if not f.is_file():
            raise FileNotFoundError(f'Split file missing: {f}')
        rows = jsonl(f)
        qids = {str(x['query_id']) for x in rows}
        sets[part] = (rows, qids)
    expect = {'train': 238, 'val': 35, 'test': 70}
    for key, n in expect.items():
        if len(sets[key][1]) != n:
            raise ValueError(f'Seed {seed}: {key} expected {n} queries, got {len(sets[key][1])}')
    if any(sets[a][1] & sets[b][1] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError(f'Seed {seed}: query ID leakage across partitions')
    for name in ('train', 'test'):
        for row in sets[name][0]:
            if any(k not in row for k in ('yB_expected', 'yA_expected', 'story', 'story_id')):
                raise ValueError(f'Seed {seed}: missing required field in {name} JSONL')
            yb = float(row['yB_expected']); ya = float(row['yA_expected'])
            if not (1.0 - 1e-5 <= yb <= 5.0 + 1e-5 and 1.0 - 1e-5 <= ya <= 5.0 + 1e-5):
                raise ValueError(f'Seed {seed}: score outside [1,5]')
    return sets


def parse_seeds(s):
    vals = list(range(10)) if s == 'all' else [int(x) for x in s.split(',')]
    if not vals or len(vals) != len(set(vals)) or any(x not in range(10) for x in vals):
        raise ValueError('Seeds must be unique integers in 0..9')
    return vals


def train_seed(root, out, seed, methods):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    partitions = verify_split(root, seed)
    train_rows = partitions['train'][0]
    test_rows = partitions['test'][0]
    print(f'Seed {seed}: vectorizing {len(train_rows)} train, {len(test_rows)} test stories', flush=True)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                          max_df=0.95, max_features=100000, sublinear_tf=True,
                          stop_words='english')
    Xtrain = vec.fit_transform([text_for(x) for x in train_rows]);
    Xtest = vec.transform([text_for(x) for x in test_rows]);
    yB64 = np.asarray([float(x['yB_expected']) for x in train_rows], dtype=np.float64)
    yB32 = yB64.astype(np.float32)
    import scipy, sklearn
    manifest = {'seed': seed, 'n_train': len(train_rows), 'n_test': len(test_rows),
                'tfidf_features': int(Xtrain.shape[1]), 'models': {},
                'software': {'python': platform.python_version(),
                             'numpy': np.__version__, 'scipy': scipy.__version__,
                             'sklearn': sklearn.__version__},
                'reference_policy': 'Compare score-quantisation arms to retrained continuous control in the same seed and environment; report drift from frozen historical reference.'}
    all_rows = []
    for method in methods:
        # Source-code parity: historical Pairwise uses float32 and Ridge float64.
        y_source = yB32 if method == 'pairwise' else yB64
        for level in (None,) + LEVELS:
            label = 'continuous' if level is None else str(level)
            y = quantize_expected(y_source, level)
            unique = int(len(np.unique(y)))
            print(f'  {method} level={label}: distinct training values={unique}', flush=True)
            if method == 'pairwise':
                model, n_pairs = fit_pairwise(Xtrain, train_rows, y, seed)
                pred = model.decision_function(Xtest)
                jitter_method = 'pairwise_tfidf_sgd_yB_expected'
            else:
                model = Ridge(alpha=1.0, solver='lsqr')
                model.fit(Xtrain, y)
                pred = model.predict(Xtest)
                n_pairs = None
                jitter_method = 'tfidf_ridge_yB_expected'
            met = eval_metrics(test_rows, pred, seed, jitter_method)
            for row in met:
                row.update({'model': method, 'levels': label,
                            'unique_targets_train': unique})
            if len(met) != 70:
                raise AssertionError(f'Expected 70 query metrics, got {len(met)}')
            all_rows.extend(met)
            manifest['models'][f'{method}:{label}'] = {
                'mean_ndcg@5': float(np.mean([r['ndcg@5'] for r in met])),
                'mean_err@5': float(np.mean([r['err@5'] for r in met])),
                'unique_target_values': unique,
                'pairwise_pairs_over_3_epochs': n_pairs,
            }
            # Compare the retrained continuous arm with the archived reference.
            if level is None:
                original = root / ORIGINAL[method]
                for metric in METRICS:
                    ref = read_reference(original, seed, metric)
                    if ref is not None:
                        got = {r['query_id']: r[metric] for r in met}
                        if set(got) != set(ref):
                            raise AssertionError(f'Original {method} seed {seed} query IDs differ')
                        qids = sorted(ref)
                        delta = np.asarray([got[q] - ref[q] for q in qids], dtype=np.float64)
                        diag = {'frozen_mean': float(np.mean([ref[q] for q in qids])),
                                'rerun_mean': float(np.mean([got[q] for q in qids])),
                                'mean_signed_delta': float(np.mean(delta)),
                                'mean_absolute_delta': float(np.mean(np.abs(delta))),
                                'max_absolute_delta': float(np.max(np.abs(delta))),
                                'count_abs_delta_gt_0_002': int(np.sum(np.abs(delta) > 0.002)),
                                'matches_exactly': bool(np.all(delta == 0))}
                        manifest['models'][f'{method}:{label}'][f'frozen_reference_audit_{metric}'] = diag
                        print(f'  REFERENCE AUDIT seed={seed} method={method} metric={metric}: '
                              f'mean_frozen={diag["frozen_mean"]:.6f} '
                              f'mean_rerun={diag["rerun_mean"]:.6f} '
                              f'mean_signed_delta={diag["mean_signed_delta"]:+.6f} '
                              f'mean_abs_delta={diag["mean_absolute_delta"]:.6f} '
                              f'max_abs_delta={diag["max_absolute_delta"]:.6f}', flush=True)
                        if abs(diag['mean_signed_delta']) > 0.02 or diag['mean_absolute_delta'] > 0.03:
                            raise AssertionError(f'Seed {seed} {method} {metric}: unusually large difference '
                                                 'from historical reference; STOP for investigation.')
                    else:
                        warnings.warn(f'Original {method} source absent: {original}; continuous arm could not be reference-verified')
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f'granularity_seed{seed}.csv'
    cols = ['seed', 'query_id', 'model', 'levels', 'unique_targets_train',
            'pool_size', 'ndcg@5', 'err@5']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)
    (out / f'granularity_seed{seed}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'PASS seed {seed}: {len(all_rows)} rows, saved {csv_path}', flush=True)


def cluster_ci(rows, model, level, metric, draws=20000, rng=None):
    sub = [x for x in rows if x['model'] == model]
    base = {(int(x['seed']), x['query_id']): float(x[metric]) for x in sub if x['levels'] == 'continuous'}
    arm = {(int(x['seed']), x['query_id']): float(x[metric]) for x in sub if x['levels'] == level}
    if set(arm) != set(base):
        raise ValueError(f'Mismatched query sets for {model} level {level}')
    keys = sorted(base)
    d = np.array([arm[k] - base[k] for k in keys])
    byq = defaultdict(list)
    for key, val in zip(keys, d): byq[key[1]].append(val)
    qs = sorted(byq)
    sums = np.array([sum(byq[q]) for q in qs]); counts = np.array([len(byq[q]) for q in qs])
    rng = rng if rng is not None else np.random.default_rng(2026)
    draws_out = []
    for _ in range(draws):
        idx = rng.integers(0, len(qs), size=len(qs))
        draws_out.append(float(sums[idx].sum() / counts[idx].sum()))
    low, high = np.quantile(draws_out, [0.025, 0.975])
    return float(d.mean()), float(low), float(high), len(d), len(qs)


def aggregate(out, draws):
    required = [out / f'granularity_seed{i}.csv' for i in range(10)]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Cannot aggregate: missing 10-seed outputs: ' + ', '.join(missing))
    rows = []
    for p in required:
        with open(p, encoding='utf-8', newline='') as f:
            data = list(csv.DictReader(f))
        if len(data) != 1120:
            raise ValueError(f'{p}: expected 1120 rows (70 queries x 8 targets x 2 models), got {len(data)}')
        rows.extend(data)
    methods = ('pairwise', 'ridge')
    levels = ('2', '3', '5', '10', '20', '50', '100', 'continuous')
    results = []
    rng = np.random.default_rng(20261023)
    for method in methods:
        for level in levels:
            target = [r for r in rows if r['model'] == method and r['levels'] == level]
            if len(target) != 700 or len({r['query_id'] for r in target}) != 315:
                raise ValueError(f'{method} {level}: expected 700 observations /315 unique queries')
            rec = {'model': method, 'levels': level, 'n_query_seed': 700,
                   'n_unique_queries': 315, 'mean_used_target_values': float(np.mean([int(r['unique_targets_train']) for r in target]))}
            for metric in METRICS:
                rec['mean_' + metric] = float(np.mean([float(r[metric]) for r in target]))
                if level == 'continuous':
                    rec['delta_vs_continuous_' + metric] = 0.0
                    rec['delta_CI95_low_' + metric] = 0.0
                    rec['delta_CI95_high_' + metric] = 0.0
                else:
                    d, lo, hi, n, q = cluster_ci(rows, method, level, metric, draws, rng)
                    rec['delta_vs_continuous_' + metric] = d
                    rec['delta_CI95_low_' + metric] = lo
                    rec['delta_CI95_high_' + metric] = hi
            results.append(rec)
    path = out / 'granularity_summary.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xs = list(LEVELS)
        for method in methods:
            pairs = {r['levels']: r for r in results if r['model'] == method}
            ys = [pairs[str(x)]['mean_ndcg@5'] for x in xs]
            ax.plot(xs, ys, marker='o', label=f'{method.capitalize()} quantized')
            ax.axhline(pairs['continuous']['mean_ndcg@5'], linestyle='--', alpha=.6,
                       label=f'{method.capitalize()} continuous')
        ax.set_xscale('log')
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel('Number of quantized Judge B expected-score levels')
        ax.set_ylabel('Judge A nDCG@5 (held-out queries)')
        ax.set_title('Supervision granularity on the original corpus')
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out / 'r2_grade_granularity_ndcg5.png', dpi=220)
        plt.close(fig)
    except ImportError:
        print('WARN: matplotlib unavailable, skipping PNG figure')
    summary_text = [
        '# Grade-resolution study: 2--100 levels of Judge B expected-score supervision',
        'Only the B expected-score training targets are quantized to a fixed uniform grid on [1,5].',
        'The Judge A test relevance labels stay continuous and unchanged.',
        'The five-level quantized expected score is NOT the original argmax hard grade.',
        'Pairwise retains its original 0.05 minimum preference gap; this affects how many pairs survive quantization.',
        'Each model is retrained from scratch with original q70 splits and TF-IDF hyperparameters.',
        'All confidence intervals cluster by the 315 unique test queries; training randomness is conditional.',
        '',
        'model, levels, mean ndcg@5, delta vs continuous, 95% clustered CI of delta'
    ]
    for r in results:
        summary_text.append(f"{r['model']}, {r['levels']}, {r['mean_ndcg@5']:.6f}, {r['delta_vs_continuous_ndcg@5']:+.6f}, [{r['delta_CI95_low_ndcg@5']:+.6f}, {r['delta_CI95_high_ndcg@5']:+.6f}]")
    (out / 'summary_readme.md').write_text('\n'.join(summary_text) + '\n', encoding='utf-8')
    zip_path = out / 'r2_granularity_share.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('summary_readme.md', 'granularity_summary.csv', 'r2_grade_granularity_ndcg5.png'):
            if (out / name).is_file(): z.write(out / name, name)
        for i in range(10):
            z.write(out / f'granularity_seed{i}_manifest.json', f'manifests/granularity_seed{i}_manifest.json')
    print((out / 'summary_readme.md').read_text(), flush=True)
    print('SHARE:', zip_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--seeds', default='0', help='Comma-separated seed(s), or all')
    ap.add_argument('--output-dir', default='outputs/irrj_r2_grade_granularity')
    ap.add_argument('--methods', default='pairwise,ridge')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=20000)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute(): output = root / output
    if args.aggregate:
        if args.bootstrap < 100: raise ValueError('--bootstrap must be at least 100')
        aggregate(output, args.bootstrap)
        return
    seeds = parse_seeds(args.seeds)
    methods = [s.strip() for s in args.methods.split(',')]
    if set(methods) - {'pairwise', 'ridge'} or not methods or len(methods) != len(set(methods)):
        raise ValueError('Choose unique methods from pairwise,ridge')
    for s in seeds:
        parts = verify_split(root, s)
        print(f"PASS seed {s}: {len(parts['train'][0])} train stories, {len(parts['val'][0])} val stories, "
              f"{len(parts['test'][0])} test stories; query leakage check passed", flush=True)
    if args.check_only:
        print(f"CHECK-ONLY: {len(seeds)} seed(s); {len(methods)} models x 8 grade resolutions; no training, outputs, or GPU")
        for method in methods:
            p = root / ORIGINAL[method]
            print('REFERENCE', method, str(p.relative_to(root)), 'FOUND' if p.is_file() else 'ABSENT')
        return
    for s in seeds:
        train_seed(root, output, s, methods)


if __name__ == '__main__':
    main()
