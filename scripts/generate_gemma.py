#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import argparse
from datetime import datetime
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Helpers
# =========================

def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# =========================
# Prompt (Gemma)
# =========================

def build_prompt(age, culture, moral) -> str:
    return (
        "You are a children's storyteller.\n"
        "Write ONE complete children's story.\n"
        "Rules:\n"
        "1) Clear beginning, middle, and ending.\n"
        "2) End with: Moral: <1-2 sentences>.\n"
        "3) After the moral, write EXACTLY: <END>\n"
        "4) Do NOT include any analysis, notes, headings, or extra commentary.\n"
        "5) Output ONLY the story text.\n\n"
        f"- Target age: {age}\n"
        f"- Cultural background: {culture}\n"
        f"- Moral value: {moral}\n\n"
        "Story:\n"
    )

def hard_stop_cleanup(text: str) -> str:
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

def make_chat_input(tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_stories", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    gen_cfg = cfg["generation"]
    paths = cfg["output_paths"]

    model_path = args.model_path or cfg["model_paths"]["generator"]
    base_out_path = args.out_path or paths["stories"]

    n = int(args.num_stories) if args.num_stories else int(gen_cfg["num_stories"])
    batch_size = int(gen_cfg.get("batch_size", 2))

    seed = int(args.seed) if args.seed else int(cfg.get("evaluation", {}).get("seed", 42))
    set_seed(seed)

    ensure_dir(os.path.dirname(base_out_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(base_out_path)
    out_path = f"{base}_{ts}{ext or '.jsonl'}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] model={model_path}", flush=True)
    print(f"[INFO] output={out_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    max_new = int(gen_cfg.get("max_new_tokens", 350))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.9))
    log_every = int(gen_cfg.get("log_every", 50))
    do_sample = temperature > 0.0

    ages = gen_cfg["ages"]
    cultures = gen_cfg["cultures"]
    morals = gen_cfg["moral_values"]

    prompts, meta = [], []
    for _ in range(n):
        age = random.choice(ages)
        culture = random.choice(cultures)
        moral = random.choice(morals)
        prompts.append(build_prompt(age, culture, moral))
        meta.append((age, culture, moral))

    start = time.time()
    story_id = 1

    with open(out_path, "w", encoding="utf-8") as fout:
        for p_batch, m_batch in zip(batched(prompts, batch_size), batched(meta, batch_size)):

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
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                )

            input_len = inputs["input_ids"].shape[1]
            gen_only = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for text, (age, culture, moral) in zip(decoded, m_batch):
                story = hard_stop_cleanup(text)

                fout.write(json.dumps({
                    "story_id": story_id,
                    "age": age,
                    "culture": culture,
                    "moral": moral,
                    "story": story,
                    "generator_model": os.path.basename(model_path.rstrip("/")),
                }, ensure_ascii=False) + "\n")

                if story_id == 1 or story_id % log_every == 0:
                    print(f"[PROGRESS] {story_id}/{n} | {int(time.time()-start)}s", flush=True)

                story_id += 1

    print(f"[DONE] {story_id-1} stories written", flush=True)
    print(f"[DONE] file: {out_path}", flush=True)

if __name__ == "__main__":
    main()
