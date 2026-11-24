#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import argparse
import re
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_query_id(age, moral, culture):
    return f"age={age}|moral={moral}|culture={culture}"


def build_prompt(age, culture, moral, variant_id):
    return (
        "You are a children's storyteller.\n"
        "Write ONE original children's story.\n\n"
        "Rules:\n"
        "1) Clear beginning, middle, and ending.\n"
        "2) The story must be suitable for the target age.\n"
        "3) The moral value must be shown through the characters' actions, not only stated at the end.\n"
        "4) The cultural background should shape the setting, behaviour, conflict, or resolution.\n"
        "5) Avoid generic repeated plots such as losing a drawing book, finding an injured animal, "
        "waiting impatiently for food, or helping in a market unless you make the plot clearly original.\n"
        "6) Use different character names, setting details, problem, and resolution.\n"
        "7) End with: Moral: <1-2 sentences>.\n"
        "8) After the moral, write EXACTLY: <END>\n"
        "9) Do NOT include analysis, headings, notes, or extra commentary.\n"
        "10) Output ONLY the story text.\n\n"
        f"- Target age: {age}\n"
        f"- Cultural background: {culture}\n"
        f"- Moral value: {moral}\n"
        f"- Variation ID: {variant_id}\n\n"
        "Story:\n"
    )


def hard_stop_cleanup(text):
    text = text.replace("\r", "").strip()

    ei = text.find("<END>")
    if ei != -1:
        text = text[:ei]

    text = re.split(
        r'(\*\*END\*\*|END\b|STOP\b|The end\b|Human:|Assistant:|User:|System:)',
        text,
        flags=re.IGNORECASE
    )[0]

    low = text.lower()
    mi = low.rfind("moral:")
    if mi != -1:
        lines = text[mi:].splitlines()
        text = text[:mi] + "\n".join(lines[:3])

    return text.strip()


def make_chat_input(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--limit_total", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--slurm_array_task_id", type=int, default=None)
    parser.add_argument("--slurm_array_task_count", type=int, default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)

    input_plan = cfg["input_plan"]
    output_raw = cfg["output_raw"]
    generation_round = cfg.get("generation_round", "topup_round")
    generator_model_name = cfg.get("generator_model", os.path.basename(args.model_path.rstrip("/")))

    max_new_tokens = int(cfg.get("max_new_tokens", 700))
    temperature = float(cfg.get("temperature", 0.9))
    top_p = float(cfg.get("top_p", 0.95))

    set_seed(args.seed)

    plan = load_json(input_plan)

    # Optional: split plan across SLURM array tasks
    if args.slurm_array_task_id is not None and args.slurm_array_task_count is not None:
        task_id = int(args.slurm_array_task_id)
        task_count = int(args.slurm_array_task_count)
        plan = [row for i, row in enumerate(plan) if i % task_count == task_id]
        output_raw = output_raw.replace(".jsonl", f"_array{task_id:03d}of{task_count:03d}.jsonl")

    if args.limit_queries is not None:
        plan = plan[:args.limit_queries]

    prompts = []
    metas = []

    for row in plan:
        age = row["age"]
        moral = row["moral"]
        culture = row["culture"]
        query_id = row["query_id"]
        n_gen = int(row["suggested_generate"])

        for j in range(n_gen):
            variant_id = f"{generation_round}_{query_id}_sample{j}_seed{args.seed}"
            prompts.append(build_prompt(age, culture, moral, variant_id))
            metas.append({
                "age": age,
                "moral": moral,
                "culture": culture,
                "query_id": query_id,
                "topup_sample_index": j,
                "suggested_generate_for_query": n_gen,
                "current_unique_before_topup": row.get("current_unique"),
                "target_unique": row.get("target_unique"),
                "deficit_before_topup": row.get("deficit"),
            })

    if args.limit_total is not None:
        prompts = prompts[:args.limit_total]
        metas = metas[:args.limit_total]

    ensure_dir(os.path.dirname(output_raw))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] model_path={args.model_path}", flush=True)
    print(f"[INFO] input_plan={input_plan}", flush=True)
    print(f"[INFO] output_raw={output_raw}", flush=True)
    print(f"[INFO] num_queries={len(plan)}", flush=True)
    print(f"[INFO] total_prompts={len(prompts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    do_sample = temperature > 0.0
    start = time.time()

    with open(output_raw, "w", encoding="utf-8") as fout:
        written = 0

        for p_batch, m_batch in zip(batched(prompts, args.batch_size), batched(metas, args.batch_size)):
            chat_inputs = [make_chat_input(tokenizer, p) for p in p_batch]

            inputs = tokenizer(
                chat_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=5,
                )

            input_len = inputs["input_ids"].shape[1]
            gen_only = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for text, meta in zip(decoded, m_batch):
                written += 1
                story = hard_stop_cleanup(text)

                story_id = (
                    f"{generation_round}__"
                    f"{meta['query_id'].replace('|', '__').replace('=', '-')}"
                    f"__sample-{meta['topup_sample_index']:05d}"
                    f"__seed-{args.seed}"
                    f"__n-{written:06d}"
                )

                rec = {
                    "story_id": story_id,
                    "age": meta["age"],
                    "culture": meta["culture"],
                    "moral": meta["moral"],
                    "query_id": meta["query_id"],
                    "story": story,
                    "generator_model": generator_model_name,
                    "generation_round": generation_round,
                    "topup_sample_index": meta["topup_sample_index"],
                    "current_unique_before_topup": meta["current_unique_before_topup"],
                    "target_unique": meta["target_unique"],
                    "deficit_before_topup": meta["deficit_before_topup"],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if written == 1 or written % 50 == 0:
                    print(f"[PROGRESS] {written}/{len(prompts)} | {int(time.time() - start)}s", flush=True)

    print(f"[DONE] wrote {written} records to {output_raw}", flush=True)


if __name__ == "__main__":
    main()
