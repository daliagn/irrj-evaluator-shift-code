#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_messages(story, culture):
    system = (
        "You are an expert evaluator of children's stories. "
        "You must evaluate only cultural relevance."
    )
    user = (
        f"Target culture: {culture}\n\n"
        "Score the story's cultural relevance using this scale:\n"
        "1 = not culturally relevant\n"
        "2 = weak cultural relevance\n"
        "3 = moderate cultural relevance\n"
        "4 = strong cultural relevance\n"
        "5 = very strong cultural relevance\n\n"
        "Return only one digit from 1 to 5. Do not explain.\n\n"
        f"Story:\n{story}\n\n"
        "Score:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def build_plain_prompt(story, culture):
    return (
        "You are an expert evaluator of children's stories.\n"
        "Evaluate only cultural relevance.\n\n"
        f"Target culture: {culture}\n\n"
        "Scale:\n"
        "1 = not culturally relevant\n"
        "2 = weak cultural relevance\n"
        "3 = moderate cultural relevance\n"
        "4 = strong cultural relevance\n"
        "5 = very strong cultural relevance\n\n"
        "Return only one digit from 1 to 5. Do not explain.\n\n"
        f"Story:\n{story}\n\n"
        "Score:"
    )

def make_prompt(tok, story, culture):
    messages = build_messages(story, culture)
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return build_plain_prompt(story, culture)

def get_digit_token_ids(tok):
    cand = {}
    for d in ["1", "2", "3", "4", "5"]:
        ids = set()
        for s in [d, " " + d, "\n" + d]:
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            enc = tok.encode(d, add_special_tokens=False)
            if enc:
                ids.add(enc[-1])
        cand[int(d)] = sorted(ids)
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True)
    ap.add_argument("--stories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", required=True, choices=["A", "B", "C"])
    ap.add_argument("--num_parts", type=int, default=1)
    ap.add_argument("--part_id", type=int, default=0)
    ap.add_argument("--max_story_chars", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_cfg(args.settings)
    j = cfg["judges"][args.judge]

    tok = AutoTokenizer.from_pretrained(
        j["model_path"],
        local_files_only=True,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        j["model_path"],
        device_map="auto",
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True
    ).eval()

    digit_ids = get_digit_token_ids(tok)

    seen = wrote = 0
    with open(args.stories, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as o:
        for line in f:
            if (seen % args.num_parts) != args.part_id:
                seen += 1
                continue
            seen += 1

            obj = json.loads(line)
            story = (obj.get("story", "") or "")[:args.max_story_chars]
            culture = obj.get("culture", "Unknown") or "Unknown"

            prompt = make_prompt(tok, story, culture)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

            digit_scores = {}
            for score, ids in digit_ids.items():
                vals = logits[ids]
                digit_scores[score] = float(vals.max().detach().cpu())

            pred = max(digit_scores, key=digit_scores.get)
            probs_tensor = F.softmax(torch.tensor([digit_scores[i] for i in [1,2,3,4,5]]), dim=0)
            probs = {str(i): float(probs_tensor[i-1]) for i in [1,2,3,4,5]}

            o.write(json.dumps({
                "story_id": obj.get("story_id"),
                "query_id": obj.get("query_id"),
                f"culture_score_{args.judge}": int(pred),
                f"culture_score_{args.judge}_probs": probs
            }, ensure_ascii=False) + "\n")

            wrote += 1
            if wrote % 200 == 0:
                print(f"[Judge {args.judge}] part={args.part_id}/{args.num_parts} wrote={wrote}", flush=True)

if __name__ == "__main__":
    main()
