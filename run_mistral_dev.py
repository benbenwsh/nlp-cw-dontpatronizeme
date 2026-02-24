#!/usr/bin/env python3
"""
Run Mistral 7B few-shot classification on the validation set.
Reads cleaned.tsv and dev par_ids, outputs dev.txt (binary 0/1 per line) and dev_results.txt (metrics + incorrect par_ids).
"""

import argparse
import csv
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


def class_04_to_binary(c: int) -> int:
    """Map class 0-4 to binary: 0-1 -> 0, 2-4 -> 1."""
    if c in (0, 1):
        return 0
    return 1


class ConstrainedDigitLogitsProcessor(LogitsProcessor):
    """Restrict next token to single-token digits 0, 1, 2, 3, 4 only."""

    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, list(self.allowed_token_ids)] = False
        scores.masked_fill_(mask, float("-inf"))
        return scores


def get_digit_token_ids(tokenizer):
    """Return token ids for '0','1','2','3','4' (try with and without leading space)."""
    digits = ["0", "1", "2", "3", "4"]
    ids = set()
    for d in digits:
        tid = tokenizer.convert_tokens_to_ids(d)
        if tid != tokenizer.unk_token_id:
            ids.add(tid)
        tid_space = tokenizer.convert_tokens_to_ids(" " + d)
        if tid_space != tokenizer.unk_token_id:
            ids.add(tid_space)
    return list(ids)


def load_dev_par_ids(dev_path: str) -> list:
    """Load set of par_ids from dev CSV (first column)."""
    dev_ids = []
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_ids.append(int(row["par_id"]))
    return dev_ids


def load_cleaned_data(data_path: str):
    """
    Load cleaned.tsv: par_id (col 0), text (col 1), label (col 2).
    Return dict par_id -> (text, label_int).
    """
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


def load_few_shot_examples(pcl_path: str):
    """
    Load par_id 1, 2, 3 from dontpatronizeme_pcl.tsv.
    Format: par_id, @@id, keyword, code, text (col 4), label (col 5). 0-based.
    Return list of (text, class_label) for 3 examples. Class is 0-4; PCL file has 0/1, use as-is.
    """
    examples = []
    with open(pcl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Data rows for par_id 1,2,3 are at 1-based lines 5,6,7 -> 0-based indices 4,5,6
    for i in (4, 5, 6):
        if i >= len(lines):
            break
        parts = lines[i].strip().split("\t")
        if len(parts) >= 6:
            text = parts[4]
            label = int(parts[5])
            examples.append((text, label))
    return examples


def build_prompt(few_shot: list, text: str) -> str:
    """Build few-shot prompt: task + 3 examples + current text, ending with 'Class:'."""
    prompt = (
        "Classify the following text into a single class 0, 1, 2, 3, or 4 "
        "(0 = no PCL, 4 = strongest PCL). Reply with only one digit.\n\n"
    )
    for ex_text, ex_class in few_shot:
        prompt += f"Text: {ex_text}\nClass: {ex_class}\n\n"
    prompt += f"Text: {text}\nClass:"
    return prompt


def tokenize_with_chat_template(tokenizer, prompt: str, max_length: int, device=None):
    """Wrap prompt in chat template and return input_ids, attention_mask."""
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
    input_ids = tokenized
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = (input_ids != pad_id).long()
    if attention_mask.sum() == 0:
        attention_mask = torch.ones_like(input_ids)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description="Mistral 7B few-shot classification on dev set")
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
        "--pcl_path",
        type=str,
        default="dontpatronizeme_pcl.tsv",
        help="Path to PCL TSV for few-shot examples (par_id 1,2,3)",
    )
    parser.add_argument(
        "--output_dev",
        type=str,
        default="dev.txt",
        help="Output path for binary predictions (one 0 or 1 per line)",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default="dev_results.txt",
        help="Output path for metrics and incorrect par_ids",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model name",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dev par_ids
    dev_ids = load_dev_par_ids(args.dev_path)
    print(f"Loaded {len(dev_ids)} dev par_ids from {args.dev_path}")

    # Load cleaned data and filter to dev, sorted by par_id
    all_data = load_cleaned_data(args.data_path)
    validation_list = []
    for par_id in dev_ids:
        if par_id in all_data:
            text, label = all_data[par_id]
            validation_list.append((par_id, text, label))
    print(f"Validation samples (in cleaned): {len(validation_list)}")

    # Gold binary labels (from cleaned col3: 0-4 -> 0/1)
    gold_binary = [class_04_to_binary(label) for (_, _, label) in validation_list]
    par_ids_ordered = [p for (p, _, _) in validation_list]

    # Few-shot examples
    few_shot = load_few_shot_examples(args.pcl_path)
    if len(few_shot) != 3:
        print(f"Warning: expected 3 few-shot examples, got {len(few_shot)}")
    print(f"Few-shot examples: {len(few_shot)}")

    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    digit_token_ids = get_digit_token_ids(tokenizer)
    if len(digit_token_ids) < 5:
        print("Warning: could not resolve all 5 digit token ids, using", digit_token_ids)
    logits_processor = ConstrainedDigitLogitsProcessor(digit_token_ids)

    # Generate predictions
    predictions_04 = []
    device = next(model.parameters()).device
    max_len = getattr(model.config, "max_position_embeddings", 32768) - 8
    for par_id, text, _ in tqdm(validation_list, desc="Generating"):
        prompt = build_prompt(few_shot, text)
        try:
            input_ids, attention_mask = tokenize_with_chat_template(
                tokenizer, prompt, max_length=max_len, device=device
            )
        except Exception:
            # Fallback if no chat template
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                logits_processor=[logits_processor],
            )

        new_token_id = out[0, -1].item()
        decoded = tokenizer.decode([new_token_id]).strip()
        # Parse digit: allow "0"-"4" or single digit from decoded string
        pred_04 = 0
        for char in decoded:
            if char in "01234":
                pred_04 = int(char)
                break
        predictions_04.append(pred_04)

    # Map to binary
    pred_binary = [class_04_to_binary(p) for p in predictions_04]

    # Write dev.txt (one line per validation sample, 0 or 1)
    with open(args.output_dev, "w", encoding="utf-8") as f:
        for b in pred_binary:
            f.write(f"{b}\n")
    print(f"Wrote {args.output_dev} ({len(pred_binary)} lines)")

    # Metrics (binary: positive class = 1)
    tp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 1)
    accuracy = (tp + tn) / len(gold_binary) if gold_binary else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    incorrect_par_ids = [par_ids_ordered[i] for i in range(len(par_ids_ordered)) if pred_binary[i] != gold_binary[i]]

    # Write results file
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        f.write("Binary classification metrics (positive class = 1 = PCL)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write("\n")
        f.write(f"Incorrect examples (par_id): {len(incorrect_par_ids)}\n")
        f.write("-" * 50 + "\n")
        for pid in incorrect_par_ids:
            f.write(f"{pid}\n")
    print(f"Wrote {args.output_metrics}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Incorrect: {len(incorrect_par_ids)} samples")


if __name__ == "__main__":
    main()
