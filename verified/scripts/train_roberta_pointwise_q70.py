#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoBERTa-large pointwise ranker for q70.

Protocol:
- Model: local models/roberta-large
- Training label: yB_expected
- Validation label: yB_expected
- Final test evaluation label: yA_expected
- Splits: query-level q70, 10 seeds
- Ranking is always within each query.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


K = 5
MAX_REL = 5.0
METHOD = "roberta_large_pointwise_yB_expected"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def make_text(row):
    return (
        f"Age: {row.get('age', '')}\n"
        f"Moral: {row.get('moral', '')}\n"
        f"Culture: {row.get('culture', '')}\n"
        f"Story:\n{row.get('story', '')}"
    )


def gain(rel):
    return (2 ** float(rel)) - 1.0


def stable_jitter(seed, query_id, story_id, method):
    text = f"{seed}|{query_id}|{story_id}|{method}"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def ranked_indices(scores, rows, seed, method):
    return sorted(
        range(len(scores)),
        key=lambda i: (
            float(scores[i]),
            stable_jitter(seed, rows[i]["query_id"], rows[i]["story_id"], method),
        ),
        reverse=True,
    )


def dcg_at_k(rels, k=5):
    return sum(gain(rel) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(labels, scores, rows, seed, method, k=5):
    order = ranked_indices(scores, rows, seed, method)
    ranked = [labels[i] for i in order]
    ideal = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg_at_k(ranked, k) / idcg


def err_at_k(labels, scores, rows, seed, method, k=5, max_rel=5.0):
    order = ranked_indices(scores, rows, seed, method)
    err = 0.0
    p_continue = 1.0
    denom = 2 ** max_rel

    for rank, idx in enumerate(order[:k], start=1):
        rel = float(labels[idx])
        R = ((2 ** rel) - 1.0) / denom
        R = max(0.0, min(1.0, R))
        err += p_continue * R / rank
        p_continue *= (1.0 - R)

    return err


def average_ranks(values):
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def kendall_tau_b(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return None if denom == 0 else (concordant - discordant) / denom


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def evaluate(rows, scores, seed, method, label_key):
    grouped = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["query_id"]].append((row, float(score)))

    per_query = []
    for qid, items in grouped.items():
        q_rows = [x[0] for x in items]
        preds = [x[1] for x in items]
        labels = [float(r[label_key]) for r in q_rows]
        per_query.append({
            "seed": seed,
            "query_id": qid,
            "method": method,
            "pool_size": len(q_rows),
            "label_key": label_key,
            "ndcg@5": ndcg_at_k(labels, preds, q_rows, seed, method, K),
            "err@5": err_at_k(labels, preds, q_rows, seed, method, K, MAX_REL),
            "spearman": spearman(labels, preds),
            "kendall_tau_b": kendall_tau_b(labels, preds),
        })
    return per_query


class StoryDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, label_key):
        self.rows = rows
        self.texts = [make_text(r) for r in rows]
        self.labels = [float(r[label_key]) for r in rows]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = self.labels[idx]
        enc["idx"] = idx
        return enc


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        labels = torch.tensor([x.pop("labels") for x in batch], dtype=torch.float32)
        idxs = torch.tensor([x.pop("idx") for x in batch], dtype=torch.long)
        enc = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        enc["labels"] = labels
        enc["idx"] = idxs
        return enc


class RobertaRegressor(torch.nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.regressor = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        score = self.regressor(self.dropout(cls)).squeeze(-1)
        return score


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_scores = []
    all_indices = []
    for batch in loader:
        idx = batch.pop("idx")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = model(**batch)
        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_indices.extend(idx.numpy().tolist())

    ordered = [None] * len(all_scores)
    for i, s in zip(all_indices, all_scores):
        ordered[i] = float(s)
    return ordered


def train_one_seed(seed, args):
    set_seed(seed)

    seed_dir = Path(args.splits_dir) / f"seed_{seed}"
    train_rows = load_jsonl(seed_dir / "train.jsonl")
    val_rows = load_jsonl(seed_dir / "val.jsonl")
    test_rows = load_jsonl(seed_dir / "test.jsonl")

    print(f"\nSeed {seed}")
    print("train records:", len(train_rows))
    print("val records:", len(val_rows))
    print("test records:", len(test_rows))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=True,
    )

    collator = Collator(tokenizer)
    train_ds = StoryDataset(train_rows, tokenizer, args.max_length, "yB_expected")
    val_ds = StoryDataset(val_rows, tokenizer, args.max_length, "yB_expected")
    test_ds = StoryDataset(test_rows, tokenizer, args.max_length, "yA_expected")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    model = RobertaRegressor(args.model_path, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))
    loss_fn = torch.nn.MSELoss()

    best_val = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            idx = batch.pop("idx")
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                preds = model(**batch)
                loss = loss_fn(preds, labels)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()) * args.grad_accum_steps)

            if step % args.log_every == 0:
                print(f"seed {seed} epoch {epoch} step {step}/{len(train_loader)} loss={np.mean(losses[-args.log_every:]):.4f}")

        val_scores = predict(model, val_loader, device)
        val_results = evaluate(val_rows, val_scores, seed, METHOD + "_val_yB", label_key="yB_expected")
        val_ndcg = mean([r["ndcg@5"] for r in val_results])
        val_err = mean([r["err@5"] for r in val_results])

        history.append({
            "epoch": epoch,
            "train_loss": mean(losses),
            "val_yB_ndcg@5": val_ndcg,
            "val_yB_err@5": val_err,
        })

        print(f"seed {seed} epoch {epoch}: train_loss={mean(losses):.4f} val_yB_nDCG@5={val_ndcg:.4f} val_yB_ERR@5={val_err:.4f}")

        if best_val is None or val_ndcg > best_val:
            best_val = val_ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_scores = predict(model, val_loader, device)
    test_scores = predict(model, test_loader, device)

    test_results = evaluate(test_rows, test_scores, seed, METHOD, label_key="yA_expected")

    score_records = []
    for split, rows, scores in [("val", val_rows, val_scores), ("test", test_rows, test_scores)]:
        for r, s in zip(rows, scores):
            score_records.append({
                "seed": seed,
                "split": split,
                "method": METHOD,
                "query_id": r.get("query_id"),
                "story_id": r.get("story_id"),
                "age": r.get("age"),
                "moral": r.get("moral"),
                "culture": r.get("culture"),
                "score": float(s),
                "yA_expected": float(r["yA_expected"]),
                "yB_expected": float(r["yB_expected"]),
            })

    print(
        f"seed {seed}: "
        f"nDCG@5={mean([r['ndcg@5'] for r in test_results]):.4f} "
        f"ERR@5={mean([r['err@5'] for r in test_results]):.4f} "
        f"rho={mean([r['spearman'] for r in test_results]):.4f} "
        f"tau={mean([r['kendall_tau_b'] for r in test_results]):.4f}"
    )

    return test_results, score_records, history


def parse_seeds(seed_arg):
    if seed_arg == "all":
        return list(range(10))
    return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="storyrank_clean/data/splits_q70")
    ap.add_argument("--model_path", default="models/roberta-large")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    all_results = []
    all_scores = []
    all_history = {}

    for seed in parse_seeds(args.seeds):
        seed_results, score_records, history = train_one_seed(seed, args)
        all_results.extend(seed_results)
        all_scores.extend(score_records)
        all_history[str(seed)] = history

    per_query_path = out_dir / f"{METHOD}_per_query_q70.csv"
    with open(per_query_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "query_id", "method", "pool_size", "label_key", "ndcg@5", "err@5", "spearman", "kendall_tau_b"],
        )
        writer.writeheader()
        writer.writerows(all_results)

    score_path = out_dir / f"{METHOD}_item_scores_q70.csv"
    with open(score_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "split", "method", "query_id", "story_id", "age", "moral", "culture", "score", "yA_expected", "yB_expected"],
        )
        writer.writeheader()
        writer.writerows(all_scores)

    by_seed = defaultdict(list)
    for r in all_results:
        by_seed[int(r["seed"])].append(r)

    seed_summaries = []
    for seed, rows in sorted(by_seed.items()):
        seed_summaries.append({
            "seed": seed,
            "method": METHOD,
            "ndcg@5": mean([r["ndcg@5"] for r in rows]),
            "err@5": mean([r["err@5"] for r in rows]),
            "spearman": mean([r["spearman"] for r in rows]),
            "kendall_tau_b": mean([r["kendall_tau_b"] for r in rows]),
        })

    summary = {
        "method": METHOD,
        "model_path": args.model_path,
        "training_label": "yB_expected",
        "validation_label": "yB_expected",
        "test_evaluation_label": "yA_expected",
        "selection": "best epoch by validation nDCG@5 using yB_expected",
        "epochs": args.epochs,
        "lr": args.lr,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "score_export": str(score_path),
        "per_query": str(per_query_path),
        "ndcg@5_mean": mean([s["ndcg@5"] for s in seed_summaries]),
        "ndcg@5_std": std([s["ndcg@5"] for s in seed_summaries]),
        "err@5_mean": mean([s["err@5"] for s in seed_summaries]),
        "err@5_std": std([s["err@5"] for s in seed_summaries]),
        "spearman_mean": mean([s["spearman"] for s in seed_summaries]),
        "spearman_std": std([s["spearman"] for s in seed_summaries]),
        "kendall_tau_b_mean": mean([s["kendall_tau_b"] for s in seed_summaries]),
        "kendall_tau_b_std": std([s["kendall_tau_b"] for s in seed_summaries]),
        "seed_summaries": seed_summaries,
        "training_history": all_history,
    }

    summary_path = out_dir / f"{METHOD}_summary_q70.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(per_query_path)
    print(summary_path)
    print(score_path)

    print("\nSummary:")
    print(
        f"{METHOD:35s} "
        f"nDCG@5={summary['ndcg@5_mean']:.4f}±{summary['ndcg@5_std']:.4f} "
        f"ERR@5={summary['err@5_mean']:.4f}±{summary['err@5_std']:.4f} "
        f"rho={summary['spearman_mean']:.4f}±{summary['spearman_std']:.4f} "
        f"tau={summary['kendall_tau_b_mean']:.4f}±{summary['kendall_tau_b_std']:.4f}"
    )


if __name__ == "__main__":
    main()
