#!/usr/bin/env python3
"""
Data augmentation for PCL training: balance classes by downsampling majority (0)
and generating similar-in-class text for minority classes (1-4) using a
Hugging Face AutoModelForCausalLM. Writes validation unchanged, then kept
training rows, then streamed generated rows to cleaned_augmented.tsv.
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dev_par_ids(dev_path: str) -> List[int]:
    """Load list of par_ids from dev CSV (par_id column)."""
    dev_ids = []
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_ids.append(int(row["par_id"]))
    return dev_ids


def load_cleaned_data(data_path: str) -> Dict[int, Tuple[str, int]]:
    """Load cleaned.tsv: par_id -> (text, label_int)."""
    data = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            par_id = int(parts[0])
            text = parts[1]
            label = int(parts[2])
            data[par_id] = (text, label)
    return data


def build_class_to_samples(
    all_data: Dict[int, Tuple[str, int]],
    dev_ids: List[int],
) -> Dict[int, List[Tuple[int, str, int]]]:
    """
    Build class_id (0-4) -> list of (par_id, text, label) for training rows only.
    """
    exclude = set(dev_ids)
    class_to_samples: Dict[int, List[Tuple[int, str, int]]] = {
        c: [] for c in range(5)
    }
    for par_id, (text, label) in all_data.items():
        if par_id in exclude:
            continue
        if label not in (0, 1, 2, 3, 4):
            print(f"ERROR: invalid label {label} for par_id {par_id}")
            continue
        class_to_samples[label].append((par_id, text, label))
    return class_to_samples


def build_generation_prompt(seed_text: str, class_label: int) -> str:
    """Build prompt for generating a new paragraph with the same PCL level."""
    pcl_desc = (
        "0 = no PCL, 1 = slight, 2 = moderate, 3 = strong, 4 = highest PCL."
    )
    return (
        "Generate a new short paragraph that is very similar in meaning and style "
        "to the following example, and that has the same level of patronising and "
        "condescending language (PCL) towards vulnerable communities. "
        f"PCL scale: {pcl_desc} "
        f"The example below is class {class_label}. Your output must be the same class. "
        "Output only the new paragraph, nothing else (no explanation, no class label).\n\n"
        f"Example (class {class_label}):\n{seed_text}\n\n"
        "New paragraph:"
    )


def tokenize_batch_with_chat_template(tokenizer, prompts: List[str], max_length: int, device):
    """Tokenize prompts with chat template; return right-padded batch (input_ids, attention_mask)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    list_of_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_ids = ids["input_ids"]
        prompt_ids = [int(x) for x in prompt_ids]
        list_of_ids.append(prompt_ids)
    max_len_batch = max(len(ids) for ids in list_of_ids)
    padded_ids = []
    for ids in list_of_ids:
        pad_len = max_len_batch - len(ids)
        padded = ids + [pad_id] * pad_len
        padded_ids.append(torch.tensor(padded, dtype=torch.long))
    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask


MARKERS = ("new paragraph:", "\n\n")

def _has_marker(text: str) -> bool:
    """Return True if any of the markers appear in text."""
    if not text:
        print(f"ERROR: No text provided")
        return False
    return any(m in text.lower() for m in MARKERS)


def _generate_batch_once(
    prompts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    max_prompt_length: int,
) -> List[str]:
    """One model call; returns list of raw decoded strings (same length as prompts)."""
    if not prompts:
        print("ERROR: No prompts provided")
        return []
    input_ids, attention_mask = tokenize_batch_with_chat_template(
        tokenizer, prompts, max_prompt_length, device
    )
    prompt_lengths = attention_mask.sum(dim=1).cpu().tolist()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    results: List[str] = []
    for i in range(out.size(0)):
        start = prompt_lengths[i]
        new_ids = out[i, start:].tolist()
        if not new_ids:
            results.append("")
            continue
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


def generate_batch(
    prompts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_prompt_length: int = 2048,
) -> List[str]:
    """
    Generate one new paragraph per prompt. If a marker is found in the output,
    retry generation once for that item; if still present after retry, return "".
    Returns a list of decoded new text only (same length as prompts). Empty/invalid become "".
    """
    if not prompts:
        print("ERROR: No prompts provided")
        return []
    results_list = _generate_batch_once(
        prompts, model, tokenizer, device, max_new_tokens, temperature, max_prompt_length
    )
    results: List[str] = [""] * len(prompts)
    retry_list: List[Tuple[int, str]] = []  # (index, prompt)
    for i, result in enumerate(results_list):
        result = result.strip()
        if _has_marker(result):
            print(f"ERROR: Found marker in generated text (will retry once): {result[:80]}...")
            retry_list.append((i, prompts[i]))
        else:
            results[i] = result
    if retry_list:
        retry_indices = [t[0] for t in retry_list]
        retry_prompts = [t[1] for t in retry_list]
        retry_results = _generate_batch_once(
            retry_prompts, model, tokenizer, device, max_new_tokens, temperature, max_prompt_length
        )
        for j, idx in enumerate(retry_indices):
            result = retry_results[j]
            result = result.strip()

            if _has_marker(result):
                print(f"WARNING: Retry still had marker or empty; giving up for prompt index {idx}")
                # results[idx] stays ""
            else:
                results[idx] = result
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Augment PCL training data: balance classes and generate similar text for minority classes"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/cleaned.tsv",
        help="Path to cleaned TSV (par_id, text, label)",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="dev_semeval_parids-labels.csv",
        help="Path to dev CSV listing par_ids in validation set",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/cleaned_augmented.tsv",
        help="Output path for augmented TSV",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale target per class: (total_train/5)*alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens per generated paragraph",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Max prompt length (truncation)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (number of samples per model.generate call)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load data and dev split
    all_data = load_cleaned_data(args.data_path)
    dev_ids = load_dev_par_ids(args.dev_path)
    print(f"Loaded {len(all_data)} rows from {args.data_path}, {len(dev_ids)} dev par_ids")

    class_to_samples = build_class_to_samples(all_data, dev_ids)
    total_train = sum(len(v) for v in class_to_samples.values())
    for c in range(5):
        print(f"  Class {c}: {len(class_to_samples[c])} training samples")
    print(f"Total training samples: {total_train}")

    target_per_class = max(1, int((total_train / 5) * args.alpha))
    print(f"Target per class (alpha={args.alpha}): {target_per_class}")

    # 2. Downsample majority classes; plan generation for minority
    generation_plan: List[Tuple[str, int, int]] = []  # (seed_text, label, num_to_generate)
    for c in range(5):
        samples = class_to_samples[c]
        n = len(samples)
        if n >= target_per_class:
            # Should just be class 0
            class_to_samples[c] = random.sample(samples, target_per_class)
        else:
            need = target_per_class - n
            if n == 0:
                print(f"ERROR: Class {c} has 0 training samples; skipping generation for this class.")
                continue
            per_seed = need // n
            remainder = need % n
            idx = 0
            for par_id, text, label in samples:
                k = per_seed + (1 if idx < remainder else 0)
                if k > 0:
                    generation_plan.append((text, label, k))
                idx += 1

    # Validation and training rows to write (before generated)
    val_rows: List[Tuple[int, str, int]] = []
    for par_id in dev_ids:
        if par_id in all_data:
            text, label = all_data[par_id]
            val_rows.append((par_id, text, label))
    train_rows: List[Tuple[int, str, int]] = []
    for c in range(5):
        for par_id, text, label in class_to_samples[c]:
            train_rows.append((par_id, text, label))

    max_par_id = max(all_data.keys()) if all_data else 0
    next_par_id = max_par_id + 1

    # 3. Load model and tokenizer for generation
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # 4. Open output and write val + train, then stream generated
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    total_generated = sum(k for _, _, k in generation_plan)
    print(f"Writing validation ({len(val_rows)}), then train ({len(train_rows)}), then {total_generated} generated rows to {args.output_path} (batch_size={args.batch_size})")

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        def write_row(par_id: int, text: str, label: int) -> None:
            # Escape tabs in text so we keep 3 columns
            text_escaped = text.replace("\t", " ").replace("\n", " ")
            out_f.write(f"{par_id}\t{text_escaped}\t{label}\n")

        for par_id, text, label in val_rows:
            write_row(par_id, text, label)
        for par_id, text, label in train_rows:
            write_row(par_id, text, label)

        skipped = 0
        for seed_text, label, k in tqdm(generation_plan, desc="Generating"):
            prompt = build_generation_prompt(seed_text, label)
            prompts_batch = [prompt] * k
            for start in range(0, k, args.batch_size):
                batch_prompts = prompts_batch[start : start + args.batch_size]
                new_texts = generate_batch(
                    batch_prompts,
                    model,
                    tokenizer,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_prompt_length=args.max_prompt_length,
                )
                for new_text in new_texts:
                    if not new_text or not new_text.strip():
                        skipped += 1
                        continue
                    write_row(next_par_id, new_text, label)
                    next_par_id += 1
        out_f.flush()

    if skipped:
        print(f"Skipped {skipped} empty/invalid model outputs.")
    print(f"Done. Wrote {args.output_path} (max par_id used: {next_par_id - 1})")


if __name__ == "__main__":
    main()
