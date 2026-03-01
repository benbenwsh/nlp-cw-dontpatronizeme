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


def tokenize_with_chat_template(
    tokenizer, prompt: str, max_length: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrap prompt in chat template; return input_ids, attention_mask on device."""
    messages = [{"role": "user", "content": prompt}]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if not isinstance(tokenized, torch.Tensor):
        tokenized = torch.tensor(tokenized, dtype=torch.long)
    if tokenized.dim() == 1:
        tokenized = tokenized.unsqueeze(0)
    input_ids = tokenized.to(device)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    attention_mask = (input_ids != pad_id).long()
    if attention_mask.sum() == 0:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def generate_one(
    prompt: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_prompt_length: int = 2048,
) -> str:
    """
    Generate one new paragraph using the model. Returns the decoded new text only.
    On empty or invalid output, returns empty string (caller may skip or retry).
    """
    input_ids, attention_mask = tokenize_with_chat_template(
        tokenizer, prompt, max_prompt_length, device
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    # out is of shape (1, L) where L is the length of the generated text

    new_ids = out[0, input_ids.shape[1] :].tolist()
    if not new_ids:
        return ""
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if (input_ids.shape[1] == max_new_tokens):
        print(f"WARNING: Generated text is the same length as the max_new_tokens: {text}")

    # Heuristic: if model output contains "New paragraph:" or similar, take after it
    for marker in ("New paragraph:", "new paragraph:", "\n\n"):
        if marker in text:
            print(f"WARNING: Found marker {marker} in text: {text}")
            text = text.split(marker, 1)[-1].strip()
    return text


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
    print(f"Writing validation ({len(val_rows)}), then train ({len(train_rows)}), then {total_generated} generated rows to {args.output_path}")

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
            for _ in range(k):
                new_text = generate_one(
                    prompt,
                    model,
                    tokenizer,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_prompt_length=args.max_prompt_length,
                )
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
