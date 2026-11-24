# Learning Under Evaluator Shift in Generate–Select Pipelines

Research code and archived results for *Learning Under Evaluator Shift in Generate–Select Pipelines: Expected-Score Supervision, Ranking, and Classification Diagnostics* by Nahhas, Cai, Razzak and Jameel (*Information Retrieval Research*, 2026).

## Study

We examine learning to rank generated children's stories under evaluator shift. The study uses 46,953 stories across 343 structured queries. Judge B (Llama-3.1-8B-Instruct) provides training and validation labels, while Judge A (Yi-1.5-9B-Chat) scores the held-out test queries. Both judges use the same five-level cultural-relevance rubric.

We compare hard grades with expected-score supervision for TF-IDF Ridge, Pairwise TF-IDF SGD and RoBERTa-large ranking models. The repository includes the research scripts, query partitions and archived results used in the paper.

## Recompute the reported results

Use Python 3.10 or later and run the following commands from the repository root.

**Ranking results and target resolution (Tables 2 and 4)**

```bash
python scripts/paper_tables.py
python scripts/summarise_results.py
```

These scripts calculate the published summaries from the archived per-query CSV files. The reported standard deviations are population standard deviations across the ten seed-level means (`ddof=0`).

**Grade-resolution analysis (Figure 1)**

```bash
python -m pip install matplotlib
python scripts/plot_grade_resolution.py
```

The figure is saved to `local_outputs/figure1_grade_resolution.png`.

**Query-clustered significance tests (Table 3)**

```bash
python -m pip install numpy
python scripts/irrj_r2_phase1_audit.py --root . --output-dir local_outputs/significance
```

This analysis uses 20,000 bootstrap resamples and Holm correction for ten planned contrasts. Generated files are written under `local_outputs/`, leaving the published results unchanged.

These commands inspect or analyse saved results. They do not train models or rerun story-level evaluation.

**Metric and input-integrity checks**

```bash
python -m unittest discover -s tests -v
```

The tests check the ranking metrics and tie-breaking logic in the supplied source, plus basic judge-output integrity. They do not require model weights or story texts.

## Files and manuscript correspondence

| Paper result | Repository files |
| --- | --- |
| Table 1: corpus and split statistics | `verified/reports/final_cleaning_provenance_46953/` |
| Table 2: baselines and learned rankers | `verified/reports/baselines_q70_tiesafe/` and the Ridge, Pairwise and RoBERTa per-query results in `verified/reports/` |
| Table 3: statistical comparisons | `outputs/irrj_r2_phase1_audit/r2_9_cluster_significance_holm.csv` |
| Table 4 and Figure 1: grade resolution | `outputs/irrj_r2_grade_granularity_v2/granularity_summary.csv` |
| Table 5 and Table 9 (Appendix B): classification | `verified/reports/roberta_large_classifier_weighted_gamma_q70/merged/` |
| Tables 6–8: evaluator-direction diagnostics | `verified/reports/roberta_large_pointwise_A_oracle_q70_cleanrep/`, `outputs/irrj_evaluator_2x2_full/` and `outputs/irrj_reverse_expected_vs_hard_full/` |
| Table 12: judge-distribution diagnostics | `outputs/irrj_r1_2_judge_entropy/` |
| Appendix D: BM25 tuning | `outputs/irrj_r2_bm25_tuning/bm25_summary.json` |
| Appendix E: selection changes and confidence intervals | `outputs/irrj_r1_4_prediction_analysis/prediction_change_summary.csv` and `outputs/irrj_r1_2_judge_entropy/judge_pair_cluster_ci.csv` |
| Query-level partitions | `frozen/v1_generation_judging_46953/splits_q70/seed_*/query_split.json` |

The archived summary table is at `frozen/v1_generation_judging_46953/final_tables_q70/verified_results_table_q70.csv`. The A-to-A condition in the paper is a same-evaluator reference, not a theoretical upper bound. The `A_oracle_sanity` rows in the baseline file (nDCG@5 = 1.0) are a pipeline check and are not reported in the paper.

## Code and inputs

The `verified/scripts/` directory contains ranking and classification implementations. The `scripts/` directory contains generation, judging, filtering, analysis and result-reporting code. Generation configurations are in `configs/generation/`; `configs/judge_settings.example.json` illustrates the required judge settings without local model paths.

The corpus and model-derived story-level scores are not included. Training requires labelled `train.jsonl`, `val.jsonl` and `test.jsonl` files for each query-level split. The paper uses ten partitions with 238 training, 35 validation and 70 test queries per seed. The supplied `query_split.json` files define the partitions used for the published experiments.

The following command illustrates how to run the Pairwise TF-IDF SGD training script with the required private data:

```bash
python -m pip install -r requirements-sparse.txt
python verified/scripts/train_pairwise_tfidf_sgd_q70.py \
  --splits_dir /path/to/splits_q70 \
  --out_dir local_outputs/pairwise_expected \
  --seeds 0
```

The specified dependency versions support a sparse-model environment; they are not a verified lockfile for all experiments. RoBERTa training requires PyTorch, Transformers, pretrained model files and the labelled corpus. For the main RoBERTa experiments (Table 11), set `--max_length 256 --batch_size 8 --eval_batch_size 16 --grad_accum_steps 2 --fp16` and use a separate `--out_dir` for each training condition.

The published query partitions cannot be reconstructed by rerunning `scripts/make_10_query_splits.py` against the available files. The repository also does not contain the full procedure for constructing expected-score training records from judge outputs. Scripts requiring private inputs have not all been exercised in this release; some use fixed input/output paths and should be run only in an isolated working copy, not against the frozen evidence directories.

## Availability and limitations

This release supports examination of the source code and recomputation of selected tables, figures and statistical results from archived evidence. It does not contain the 46,953 story texts, complete judge annotations, story-level predictions, vectorisers or trained checkpoints. As a result, independent story-level evaluation and complete retraining from the public files are not supported.

The relevance labels come from language-model judges rather than human assessors. The reported results do not establish cultural authenticity or children's overall story suitability.

## Licence and citation

The research source code, configurations and documentation are provided under the [MIT Licence](LICENSE), Copyright (c) 2026 Dalia G. Nahhas. The archived CSV and JSON results are made available for inspection but are not covered by the software licence. Third-party models and libraries retain their own licensing terms.

Please cite the associated paper when using this repository. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
