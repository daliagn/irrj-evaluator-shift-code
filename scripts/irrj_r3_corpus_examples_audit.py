#!/usr/bin/env python3
"""Corpus and example diagnostics (CPU-only, read-only).

Uses original q70 seed_0 split JSONLs. Produces real example story excerpts,
query factor values, and within-query Judge A/B expected-score variation.
No model calls or data uploads. Does not claim human relevance ground truth.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from zipfile import ZipFile, ZIP_DEFLATED

PARTS = ('train','val','test')
EXPECTED = {'train':238, 'val':35, 'test':70}
EXAMPLE_TRIPLES = (
    ('7','kindness','Arab'),
    ('6','patience','East Asian'),
    ('8','honesty','Global / Neutral'),
)
NEEDED = ('query_id','story_id','age','moral','culture','story','yA_expected','yB_expected')

def make_key(row):
    age=str(row['age'])
    if age.endswith('.0'): age=age[:-2]
    return (age, str(row['moral']), str(row['culture']))

def write_csv(path, rows, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--output-dir',default='outputs/irrj_r3_corpus_audit')
    ap.add_argument('--fixture',action='store_true',help='skip expected corpus-size checks; for testing only')
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    src=root/'data'/'splits_q70'/'seed_0'
    output=root/args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    seen_sids=set(); qdata=defaultdict(lambda:{'n':0,'A_sum':0.,'B_sum':0.,'A_min':float('inf'),'A_max':float('-inf'),'B_min':float('inf'),'B_max':float('-inf'),'A_ge4':0,'B_ge4':0})
    factors={'age':Counter(),'moral':Counter(),'culture':Counter()}
    partitions={}; q_trip={}; samples={}
    n=0
    for part in PARTS:
        path=src/f'{part}.jsonl'
        if not path.exists() and part=='val':path=src/'dev.jsonl'
        if not path.exists():raise FileNotFoundError(path)
        part_q=set()
        with path.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                if not line.strip():continue
                x=json.loads(line)
                missing=[k for k in NEEDED if k not in x]
                if missing:raise ValueError(f'{path}:{i}: missing {missing}')
                sid=str(x['story_id']); qid=str(x['query_id']); key=make_key(x)
                if sid in seen_sids:raise ValueError(f'Duplicate story_id {sid}')
                seen_sids.add(sid)
                if qid in q_trip and q_trip[qid]!=key:
                    raise ValueError(f'Inconsistent attributes for {qid}')
                q_trip[qid]=key;part_q.add(qid)
                if not str(x['story']).strip():raise ValueError('Empty story')
                a,b=float(x['yA_expected']),float(x['yB_expected'])
                if not (1-1e-5<=a<=5+1e-5 and 1-1e-5<=b<=5+1e-5):
                    raise ValueError(f'Invalid judge score at {qid}')
                d=qdata[qid];d['n']+=1;d['A_sum']+=a;d['B_sum']+=b
                d['A_min']=min(d['A_min'],a);d['A_max']=max(d['A_max'],a)
                d['B_min']=min(d['B_min'],b);d['B_max']=max(d['B_max'],b)
                d['A_ge4']+=int(a>=4);d['B_ge4']+=int(b>=4)
                for k in factors:factors[k][key[('age','moral','culture').index(k)]]+=1
                if key in EXAMPLE_TRIPLES:
                    token=hashlib.sha256(f'IRRJ_R3_EXAMPLE|{qid}|{sid}'.encode()).hexdigest()
                    if key not in samples or token<samples[key]['hash']:
                        samples[key]={'hash':token,'query_id':qid,'story_id':sid,'age':key[0],
                          'moral':key[1],'culture':key[2],'story':str(x['story']),
                          'judge_a_expected':a,'judge_b_expected':b}
                n+=1
        partitions[part]=part_q
    if any(partitions[a]&partitions[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Query partition leakage')
    if not args.fixture:
        if n!=46953 or len(qdata)!=343:raise ValueError(f'Unexpected data size n={n}, queries={len(qdata)}')
        if {p:len(q) for p,q in partitions.items()}!=EXPECTED:raise ValueError('Unexpected split sizes')
        if min(d['n'] for d in qdata.values())!=101 or max(d['n'] for d in qdata.values())!=166:
            raise ValueError('Unexpected pool sizes')
        if len({tuple(q_trip[q]) for q in qdata})!=343:raise ValueError('Missing factor combinations')
        if any(len(factors[k])!=7 for k in factors):raise ValueError('Unexpected factor values')
    qr=[]
    for qid,d in sorted(qdata.items()):
        age,moral,culture=q_trip[qid];m=d['n']
        qr.append({'query_id':qid,'age':age,'moral':moral,'culture':culture,'n_stories':m,
          'judge_a_expected_mean':round(d['A_sum']/m,6),
          'judge_b_expected_mean':round(d['B_sum']/m,6),
          'judge_a_expected_range':round(d['A_max']-d['A_min'],6),
          'judge_b_expected_range':round(d['B_max']-d['B_min'],6),
          'judge_a_expected_ge4_share':round(d['A_ge4']/m,6),
          'judge_b_expected_ge4_share':round(d['B_ge4']/m,6)})
    write_csv(output/'r3_query_score_variation.csv',qr,list(qr[0]))
    factor_rows=[{'factor':factor,'value':v,'stories':c} for factor,counts in factors.items() for v,c in sorted(counts.items())]
    write_csv(output/'r3_factor_values.csv',factor_rows,['factor','value','stories'])
    examples=[]
    full=[]
    for triple in EXAMPLE_TRIPLES:
        if triple not in samples:continue
        ex=samples[triple]; story=ex['story']
        examples.append({k:v for k,v in ex.items() if k not in ('story','hash')} | {'story_excerpt':story[:750]})
        full.append(f"QUERY {ex['query_id']}\nStory ID: {ex['story_id']}\nJudge A expected: {ex['judge_a_expected']:.5f}\nJudge B expected: {ex['judge_b_expected']:.5f}\n\n{story}\n\n{'='*72}\n")
    if examples:write_csv(output/'r3_real_examples_excerpts.csv',examples,list(examples[0]))
    (output/'r3_real_examples_full_LOCAL_REVIEW.txt').write_text('\n'.join(full),encoding='utf-8')
    summary={'warning':'LLM-assigned expected grades are not human ground truth or age/moral appropriateness ratings',
       'n_stories':n,'n_queries':len(qdata),'queries_by_partition':{p:len(q) for p,q in partitions.items()},
       'factor_values':{k:sorted(factors[k].keys(),key=lambda x:int(x) if k=='age' else x) for k in factors},
       'mean_judge_a_expected_by_story':sum(d['A_sum'] for d in qdata.values())/n,
       'mean_judge_b_expected_by_story':sum(d['B_sum'] for d in qdata.values())/n,
       'fraction_stories_with_judge_a_expected_at_least_4':sum(d['A_ge4'] for d in qdata.values())/n,
       'fraction_stories_with_judge_b_expected_at_least_4':sum(d['B_ge4'] for d in qdata.values())/n,
       'mean_within_query_judge_a_expected_range':mean(d['A_max']-d['A_min'] for d in qdata.values()),
       'mean_within_query_judge_b_expected_range':mean(d['B_max']-d['B_min'] for d in qdata.values()),
       'example_count':len(examples),'scope':'IRRJ cultural-relevance judgements'}
    (output/'r3_corpus_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'r3_readme.txt').write_text('Corpus diagnostics from the frozen seed-0 partitions.\nOutputs summarise the query factors, judge-score variation, and selected generated-story examples.\nScores are model-derived judgements, not human assessments.\n',encoding='utf-8')
    included=['r3_factor_values.csv','r3_query_score_variation.csv','r3_real_examples_excerpts.csv','r3_corpus_summary.json','r3_readme.txt']
    zpath=output/'r3_corpus_share.zip'
    with ZipFile(zpath,'w',ZIP_DEFLATED) as z:
        for name in included:
            p=output/name
            if p.exists():z.write(p,arcname=name)
    print(f'PASS n={n}, queries={len(qdata)}, stories per query {min(d["n"] for d in qdata.values())}-{max(d["n"] for d in qdata.values())}')
    for k in factors:print(f'{k}: {summary["factor_values"][k]}')
    print(f'Example excerpts: {len(examples)}')
    print(f'LOCAL FULL STORY EXAMPLES: {output/"r3_real_examples_full_LOCAL_REVIEW.txt"}')
    print(f'SHARE: {zpath}')

if __name__=='__main__':main()
