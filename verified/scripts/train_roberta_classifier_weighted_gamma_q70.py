#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

K = 5
MAX_REL = 5.0
BASE_METHOD = "roberta_large_classifier_weighted_gamma"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def make_text(r):
    return (
        f"Age: {r.get('age','')}\n"
        f"Moral: {r.get('moral','')}\n"
        f"Culture: {r.get('culture','')}\n"
        f"Story:\n{r.get('story','')}"
    )

def stable_jitter(seed, qid, sid, method):
    h = hashlib.md5(f"{seed}|{qid}|{sid}|{method}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)

def ranked_indices(scores, rows, seed, method):
    return sorted(
        range(len(scores)),
        key=lambda i: (
            float(scores[i]),
            stable_jitter(seed, rows[i]["query_id"], rows[i]["story_id"], method)
        ),
        reverse=True
    )

def gain(rel):
    return (2 ** float(rel)) - 1.0

def dcg_at_k(rels, k=5):
    return sum(gain(rel) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(labels, scores, rows, seed, method, k=5):
    order = ranked_indices(scores, rows, seed, method)
    ranked = [labels[i] for i in order]
    ideal = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg_at_k(ranked, k) / idcg

def err_at_k(labels, scores, rows, seed, method, k=5):
    order = ranked_indices(scores, rows, seed, method)
    err = 0.0
    p_continue = 1.0
    denom = 2 ** MAX_REL
    for rank, idx in enumerate(order[:k], start=1):
        rel = float(labels[idx])
        R = max(0.0, min(1.0, ((2 ** rel) - 1.0) / denom))
        err += p_continue * R / rank
        p_continue *= (1.0 - R)
    return err

def binary_metrics(y_true, probs, gamma=0.5):
    y_pred = [1 if p >= gamma else 0 for p in probs]

    tp = sum(1 for y,p in zip(y_true,y_pred) if y == 1 and p == 1)
    fp = sum(1 for y,p in zip(y_true,y_pred) if y == 0 and p == 1)
    fn = sum(1 for y,p in zip(y_true,y_pred) if y == 1 and p == 0)
    tn = sum(1 for y,p in zip(y_true,y_pred) if y == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    pos = sum(y_true)
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        auroc = None
    else:
        pairs = sorted((p, i, y) for i, (p, y) in enumerate(zip(probs, y_true)))
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            avg = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[pairs[k][1]] = avg
            i = j + 1
        sum_pos_ranks = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
        auroc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)

    return {
        "n": len(y_true),
        "positive_rate": pos / len(y_true) if y_true else 0.0,
        "predicted_positive_rate": sum(y_pred) / len(y_pred) if y_pred else 0.0,
        "gamma": gamma,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

def always_positive_metrics(y_true):
    probs = [1.0] * len(y_true)
    m = binary_metrics(y_true, probs, gamma=0.5)
    m["auroc"] = None
    m["note"] = "AUROC undefined for constant always-positive predictor."
    return m

def choose_gamma_by_val_f1(y_true, probs):
    candidates = sorted(set([round(float(p), 6) for p in probs] + [0.5]))
    best = None

    for g in candidates:
        m = binary_metrics(y_true, probs, gamma=g)
        key = (m["f1"], m["precision"], m["recall"], -abs(g - 0.5))
        if best is None or key > best["key"]:
            best = {"gamma": g, "metrics": m, "key": key}

    return best["gamma"], best["metrics"]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def evaluate_ranking(rows, probs, seed, method):
    grouped = defaultdict(list)
    for r,p in zip(rows, probs):
        grouped[r["query_id"]].append((r,p))

    out = []
    for qid, items in grouped.items():
        qrows = [x[0] for x in items]
        ps = [x[1] for x in items]
        labels = [float(r["yA_expected"]) for r in qrows]
        out.append({
            "seed": seed,
            "query_id": qid,
            "method": method,
            "pool_size": len(qrows),
            "ndcg@5": ndcg_at_k(labels, ps, qrows, seed, method),
            "err@5": err_at_k(labels, ps, qrows, seed, method),
        })
    return out

class ClsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, tau, label_source):
        self.rows = rows
        self.texts = [make_text(r) for r in rows]
        self.labels = [1.0 if float(r[label_source]) >= tau else 0.0 for r in rows]
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
            return_tensors=None
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

class RobertaClassifier(torch.nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls)).squeeze(-1)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, idxs_all = [], []

    for batch in loader:
        idx = batch.pop("idx")
        labels = batch.pop("labels")
        batch = {k:v.to(device) for k,v in batch.items()}
        logits = model(**batch)
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(p)
        idxs_all.extend(idx.numpy().tolist())

    ordered = [None] * len(probs)
    for i, p in zip(idxs_all, probs):
        ordered[i] = float(p)
    return ordered

def train_one(seed, tau, args):
    set_seed(seed)
    method = f"{BASE_METHOD}_tau{int(tau)}"

    seed_dir = Path(args.splits_dir) / f"seed_{seed}"
    train_rows = load_jsonl(seed_dir / "train.jsonl")
    val_rows = load_jsonl(seed_dir / "val.jsonl")
    test_rows = load_jsonl(seed_dir / "test.jsonl")

    print(f"Seed {seed} tau {tau}")
    print("train", len(train_rows), "val", len(val_rows), "test", len(test_rows))

    train_labels = [1 if float(r["yB_expected"]) >= tau else 0 for r in train_rows]
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    pos_weight_value = (neg / pos) if pos > 0 else 1.0

    print("train positives", pos, "train negatives", neg, "pos_weight", pos_weight_value)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    collator = Collator(tokenizer)

    train_ds = ClsDataset(train_rows, tokenizer, args.max_length, tau, "yB_expected")
    val_ds = ClsDataset(val_rows, tokenizer, args.max_length, tau, "yB_expected")
    test_ds = ClsDataset(test_rows, tokenizer, args.max_length, tau, "yA_expected")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device", device)

    model = RobertaClassifier(args.model_path, args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))

    best_val = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            idx = batch.pop("idx")
            labels = batch.pop("labels").to(device)
            batch = {k:v.to(device) for k,v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                logits = model(**batch)
                loss = loss_fn(logits, labels) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()) * args.grad_accum_steps)

            if step % args.log_every == 0:
                print(f"seed {seed} tau {tau} epoch {epoch} step {step}/{len(train_loader)} loss={np.mean(losses[-args.log_every:]):.4f}")

        val_probs = predict_probs(model, val_loader, device)
        val_true = [1 if float(r["yB_expected"]) >= tau else 0 for r in val_rows]
        val_default = binary_metrics(val_true, val_probs, gamma=0.5)
        val_gamma, val_best = choose_gamma_by_val_f1(val_true, val_probs)

        val_score = val_best["f1"]

        history.append({
            "epoch": epoch,
            "train_loss": mean(losses),
            "val_default_gamma_0_5": val_default,
            "val_selected_gamma": val_gamma,
            "val_selected_gamma_metrics": val_best,
        })

        print(
            f"seed {seed} tau {tau} epoch {epoch}: "
            f"loss={mean(losses):.4f} "
            f"val_f1@0.5={val_default['f1']:.4f} "
            f"val_best_gamma={val_gamma:.6f} "
            f"val_best_f1={val_best['f1']:.4f}"
        )

        if best_val is None or val_score > best_val:
            best_val = val_score
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = predict_probs(model, val_loader, device)
    val_true = [1 if float(r["yB_expected"]) >= tau else 0 for r in val_rows]
    selected_gamma, val_selected_metrics = choose_gamma_by_val_f1(val_true, val_probs)

    test_probs = predict_probs(model, test_loader, device)
    test_true = [1 if float(r["yA_expected"]) >= tau else 0 for r in test_rows]

    test_selected = binary_metrics(test_true, test_probs, gamma=selected_gamma)
    test_default = binary_metrics(test_true, test_probs, gamma=0.5)
    ap = always_positive_metrics(test_true)

    ranking_rows = evaluate_ranking(test_rows, test_probs, seed, method)

    score_rows = []
    for split, rows, probs, label_source in [
        ("val", val_rows, val_probs, "yB_expected"),
        ("test", test_rows, test_probs, "yA_expected")
    ]:
        for r,p in zip(rows, probs):
            score_rows.append({
                "seed": seed,
                "split": split,
                "tau": tau,
                "method": method,
                "query_id": r["query_id"],
                "story_id": r["story_id"],
                "prob_positive": p,
                "selected_gamma": selected_gamma,
                "yA_expected": float(r["yA_expected"]),
                "yB_expected": float(r["yB_expected"]),
                "positive_label_source": label_source,
                "positive": 1 if float(r[label_source]) >= tau else 0,
            })

    print(
        f"seed {seed} tau {tau}: "
        f"gamma={selected_gamma:.6f} "
        f"test_F1={test_selected['f1']:.4f} "
        f"P={test_selected['precision']:.4f} "
        f"R={test_selected['recall']:.4f} "
        f"AUROC={test_selected['auroc']}"
    )

    return {
        "seed": seed,
        "tau": tau,
        "method": method,
        "pos_weight": pos_weight_value,
        "train_positive_rate": pos / len(train_labels),
        "selected_gamma": selected_gamma,
        "val_selected_gamma_metrics": val_selected_metrics,
        "test_selected_gamma_metrics": test_selected,
        "test_default_gamma_0_5_metrics": test_default,
        "always_positive": ap,
        "ranking_as_classifier_probability": {
            "ndcg@5": mean([r["ndcg@5"] for r in ranking_rows]),
            "err@5": mean([r["err@5"] for r in ranking_rows]),
        },
        "history": history,
    }, ranking_rows, score_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="storyrank_clean/data/splits_q70")
    ap.add_argument("--model_path", default="models/roberta-large")
    ap.add_argument("--out_dir", default="storyrank_clean/verified/reports/roberta_large_classifier_weighted_gamma_q70")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / f"tau_{int(args.tau)}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, ranking_rows, score_rows = train_one(args.seed, args.tau, args)

    method = f"{BASE_METHOD}_tau{int(args.tau)}"

    with open(out_dir / f"{method}_summary_seed_{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"{method}_ranking_per_query_seed_{args.seed}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
        w.writeheader()
        w.writerows(ranking_rows)

    with open(out_dir / f"{method}_item_scores_seed_{args.seed}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)

    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
