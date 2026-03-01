#!/usr/bin/env python3
"""
LoRA fine-tune Mistral 7B on PCL data (excluding dev set), save PEFT adapter,
then run validation on dev set. Supports --few_shot (default: no few-shot).
"""

import os
import time
import argparse
import csv
import random
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.generation.logits_process import LogitsProcessor
from peft import LoraConfig, get_peft_model, PeftModel

# --- Helpers (self-contained, same behaviour as run_mistral_dev) ---

INSTRUCTION = (
    "You will be given a text from a paragraph. You need to identify whether "
    "the paragraph contains language that is patronizing or condescending "
    "towards vulnerable communities. This type of language is also known as "
    "PCL (Patronizing and Condescending Language). "
    "Classify the following text into a single class 0, 1, 2, 3, or 4 "
    "(0 = no PCL, 4 = strongest PCL). Reply with only one digit. "
)


def class_04_to_binary(c: int) -> int:
    """Map class 0-4 to binary: 0-1 -> 0, 2-4 -> 1."""
    if c in (0, 1):
        return 0
    elif c in (2, 3, 4):
        return 1
    else:
        print(f"ERROR: invalid class {c}")
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
    """Return token ids for '0','1','2','3','4' (with and without leading space)."""
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


def load_dev_par_ids(dev_path: str) -> List[int]:
    """Load list of par_ids from dev CSV (par_id column)."""
    dev_ids = []
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_ids.append(int(row["par_id"]))
    return dev_ids


def load_pcl_train(data_path: str, exclude_par_ids: List[int]) -> List[Tuple[str, int]]:
    """
    Load training data from cleaned.tsv. Return list of (text, label) for rows
    whose par_id is not in exclude_par_ids.
    """
    exclude = set(exclude_par_ids)
    all_data = load_cleaned_data(data_path)
    examples = []
    for par_id, (text, label) in all_data.items():
        if par_id in exclude:
            continue
        if label not in (0, 1, 2, 3, 4):
            continue
        examples.append((text, label))
    return examples


def load_cleaned_data(data_path: str) -> dict:
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


def load_few_shot_examples(data_path: str) -> List[Tuple[str, int]]:
    """Load fixed few-shot examples from cleaned.tsv (same line indices as before)."""
    examples = []
    FEW_SHOT_EXAMPLE_INDICES = (0, 8, 32, 33, 117)
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in FEW_SHOT_EXAMPLE_INDICES:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                print(f"ERROR: invalid line {line}")
                continue
            text = parts[1]
            label = int(parts[2])
            examples.append((text, label))
    return examples


def build_prompt(few_shot: List[Tuple[str, int]], text: str) -> str:
    """Build prompt: instruction + optional few-shot examples + current text, ending with 'Class:'."""
    prompt = INSTRUCTION
    for ex_text, ex_class in few_shot:
        prompt += f"Text: {ex_text}\nClass: {ex_class}\n\n"
    prompt += f"Text: {text}\nClass:"
    return prompt


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
    
    # Adding padding
    max_len_batch = max(len(ids) for ids in list_of_ids)
    padded_ids = []
    for ids in list_of_ids:
        pad_len = max_len_batch - len(ids)
        padded = ids + [pad_id] * pad_len
        padded_ids.append(torch.tensor(padded, dtype=torch.long))
    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask


# --- Training dataset: prompt + label, labels mask prompt with -100 ---


class PCLDataCollator:
    """Pad batch to longest sequence; pad labels with -100 so loss ignores them."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {}
        batch["input_ids"] = []
        batch["attention_mask"] = []
        batch["labels"] = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            # Right-pad so causal LM sees real tokens first
            input_ids = f["input_ids"].tolist() + [self.pad_id] * pad_len
            attention_mask = f["attention_mask"].tolist() + [0] * pad_len
            labels = f["labels"].tolist() + [-100] * pad_len
            batch["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            batch["attention_mask"].append(torch.tensor(attention_mask, dtype=torch.long))
            batch["labels"].append(torch.tensor(labels, dtype=torch.long))
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])
        return batch


def get_label_token_id(tokenizer, label: int) -> int:
    """Get single token id for class 0-4 (prefer space+digit if available)."""
    for s in (" " + str(label), str(label)):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid != tokenizer.unk_token_id:
            print("ERROR: could not find label token id for", label)
            return tid
    return tokenizer.convert_tokens_to_ids(str(label))


class PCLDataset(Dataset):
    """Dataset of (input_ids, attention_mask, labels) for causal LM; labels are -100 on prompt."""

    def __init__(
        self,
        examples: List[Tuple[str, int]], # List of (text, label) tuples
        tokenizer,
        max_length: int,
        few_shot: List[Tuple[str, int]],
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot = few_shot
        self.examples = examples
        self.label_token_ids = [get_label_token_id(tokenizer, i) for i in range(5)]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        prompt_text = build_prompt(self.few_shot, text)
        # Tokenize prompt (user message + generation prompt, no answer yet)
        messages = [{"role": "user", "content": prompt_text}]
        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_length - 2, # -2 for the answer token
        )
        # Handle BatchEncoding (newer transformers) or list
        prompt_ids = out["input_ids"]
        prompt_ids = list(prompt_ids)

        answer_token_id = self.label_token_ids[label]
        input_ids = prompt_ids + [answer_token_id]
        if len(input_ids) > self.max_length:
            print("ERROR: input_ids length is greater than max_length")
            input_ids = input_ids[: self.max_length]
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + [answer_token_id]
        if len(labels) > self.max_length:
            print("ERROR: labels length is greater than max_length")
            labels = labels[: self.max_length]
        # No padding: return at actual length; collator will pad per batch
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _compute_eval_metrics(eval_preds, tokenizer) -> dict:
    """Compute binary F1 and accuracy (0-1 vs 2-4) from Trainer eval. predictions = logits (N, L, V), label_ids = (N, L) with -100 on prompt/pad."""
    predictions, label_ids = eval_preds.predictions, eval_preds.label_ids
    print(f"Predictions: {predictions}")
    print(f"Label IDs: {label_ids}")
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.array(predictions)
    labels = np.array(label_ids)
    # Logits at last position: (N, L, V) -> (N, V) (N=batch size, L=sequence length, V=vocabulary size)
    if preds.ndim == 3:
        logits_last = preds[:, -1, :]
    else:
        logits_last = preds
    N = logits_last.shape[0]
    label_token_ids = [get_label_token_id(tokenizer, k) for k in range(5)]
    token_id_to_class = {tid: k for k, tid in enumerate(label_token_ids)}
    # Predicted class: argmax over the 5 digit token logits
    preds_04 = []
    for i in range(N):
        scores = [logits_last[i, tid] for tid in label_token_ids]
        preds_04.append(int(np.argmax(scores)))
    # Gold: last non -100 in each row
    gold_04 = []
    for i in range(N):
        valid = np.where(labels[i] != -100)[0]
        gold_token = int(labels[i, valid[-1]]) if len(valid) > 0 else -100
        gold_04.append(token_id_to_class.get(gold_token, 0))
    pred_binary = [class_04_to_binary(p) for p in preds_04]
    gold_binary = [class_04_to_binary(int(g)) for g in gold_04]
    tp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(gold_binary) if gold_binary else 0.0
    return {"eval_f1": f1, "eval_accuracy": accuracy}


def _plot_train_eval_loss(trainer, save_dir: str) -> None:
    """Extract train loss, eval loss, eval F1, and eval accuracy per epoch from trainer.state.log_history; save four plots and metrics_per_epoch.txt."""
    train_loss_by_epoch = defaultdict(list)
    eval_epochs, eval_losses, eval_f1s, eval_accuracies = [], [], [], []
    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss_by_epoch[int(entry["epoch"])].append(entry["loss"])
        if "eval_loss" in entry:
            eval_epochs.append(int(entry["epoch"]))
            eval_losses.append(entry["eval_loss"])
            eval_f1s.append(entry.get("eval_f1", float("nan")))
            eval_accuracies.append(entry.get("eval_accuracy", float("nan")))
    epochs = sorted(train_loss_by_epoch.keys())
    train_losses = [sum(train_loss_by_epoch[e]) / len(train_loss_by_epoch[e]) for e in epochs]
    if not epochs:
        return
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, "metrics_per_epoch.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Epoch\tTrain loss\tEval loss\tEval F1\tEval accuracy\n")
        for i, e in enumerate(epochs):
            train_loss = train_losses[i] if i < len(train_losses) else float("nan")
            eval_idx = eval_epochs.index(e) if e in eval_epochs else None
            eval_loss = eval_losses[eval_idx] if eval_idx is not None else float("nan")
            eval_f1 = eval_f1s[eval_idx] if eval_idx is not None and eval_idx < len(eval_f1s) else float("nan")
            eval_acc = eval_accuracies[eval_idx] if eval_idx is not None and eval_idx < len(eval_accuracies) else float("nan")
            f.write(f"{e}\t{train_loss:.4f}\t{eval_loss:.4f}\t{eval_f1:.4f}\t{eval_acc:.4f}\n")
    print(f"Saved {metrics_path}")

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, marker="o", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss vs epoch")
    ax.set_xticks(epochs)
    fig.savefig(os.path.join(save_dir, "train_loss_vs_epoch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(save_dir, 'train_loss_vs_epoch.png')}")
    if eval_epochs:
        fig2, ax2 = plt.subplots()
        ax2.plot(eval_epochs, eval_losses, marker="o", linestyle="-", color="C1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation loss vs epoch")
        ax2.set_xticks(eval_epochs)
        fig2.savefig(os.path.join(save_dir, "eval_loss_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {os.path.join(save_dir, 'eval_loss_vs_epoch.png')}")
    if eval_epochs and not all(f != f for f in eval_f1s):
        fig3, ax3 = plt.subplots()
        ax3.plot(eval_epochs, eval_f1s, marker="o", linestyle="-", color="C2")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1")
        ax3.set_title("Validation F1 vs epoch")
        ax3.set_xticks(eval_epochs)
        fig3.savefig(os.path.join(save_dir, "eval_f1_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved {os.path.join(save_dir, 'eval_f1_vs_epoch.png')}")
    if eval_epochs and not all(a != a for a in eval_accuracies):
        fig4, ax4 = plt.subplots()
        ax4.plot(eval_epochs, eval_accuracies, marker="o", linestyle="-", color="C3")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Validation accuracy vs epoch")
        ax4.set_xticks(eval_epochs)
        fig4.savefig(os.path.join(save_dir, "eval_accuracy_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved {os.path.join(save_dir, 'eval_accuracy_vs_epoch.png')}")


def train_lora(args, tokenizer, train_examples, few_shot):
    """Load base model, apply LoRA, train, save adapter."""
    print("Loading base model for training...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("Printing trainable parameters...")
    model.print_trainable_parameters()

    max_length = args.max_length
    train_dataset = PCLDataset(train_examples, tokenizer, max_length, few_shot)
    # Eval dataset from dev set (same as ordinal)
    dev_ids = load_dev_par_ids(args.dev_path)
    all_data = load_cleaned_data(args.data_path)
    eval_examples = [(all_data[par_id][0], all_data[par_id][1]) for par_id in dev_ids if par_id in all_data]
    eval_dataset = PCLDataset(eval_examples, tokenizer, max_length, few_shot)
    print(f"Eval examples (dev): {len(eval_dataset)}")

    data_collator = PCLDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    def compute_metrics(eval_preds):
        return _compute_eval_metrics(eval_preds, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.adapter_save_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )
    print("Training with batch size:", args.train_batch_size, "number of epochs:", args.num_epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    time_start = time.time()
    trainer.train()
    time_end = time.time()
    print(f"Training time: {time_end - time_start} seconds")

    _plot_train_eval_loss(trainer, args.adapter_save_path)

    model.save_pretrained(args.adapter_save_path)
    tokenizer.save_pretrained(args.adapter_save_path)
    print(f"Saved adapter and tokenizer to {args.adapter_save_path}")
    return model


def run_validation(args, tokenizer, few_shot):
    """Load base + adapter, run validation, write dev.txt, dev_04.txt, dev_results.txt."""
    print("Loading base model and adapter for validation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_save_path)
    model.eval()

    dev_ids = load_dev_par_ids(args.dev_path)
    all_data = load_cleaned_data(args.data_path)
    validation_list = []
    for par_id in dev_ids:
        if par_id in all_data:
            text, label = all_data[par_id]
            validation_list.append((par_id, text, label))
        else:
            print(f"WARNING: par_id {par_id} not in cleaned data")
    print(f"Validation samples: {len(validation_list)}")

    gold_binary = [class_04_to_binary(l) for (_, _, l) in validation_list]
    par_ids_ordered = [p for (p, _, _) in validation_list]

    digit_token_ids = get_digit_token_ids(tokenizer)
    logits_processor = ConstrainedDigitLogitsProcessor(digit_token_ids)
    device = next(model.parameters()).device
    batch_size = max(1, args.batch_size)

    predictions_04 = []
    for start in tqdm(range(0, len(validation_list), batch_size), desc="Generating"):
        batch_items = validation_list[start : start + batch_size]
        batch_prompts = [build_prompt(few_shot, text) for _, text, _ in batch_items]
        input_ids, attention_mask = tokenize_batch_with_chat_template(
            tokenizer, batch_prompts, max_length=args.max_length, device=device
        )
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                logits_processor=[logits_processor],
            )

        for i in range(out.size(0)):
            new_token_id = out[i, -1].item()
            decoded = tokenizer.decode([new_token_id]).strip()
            pred_04 = 0
            for char in decoded:
                if char in "01234":
                    pred_04 = int(char)
                    break
                else:
                    print(f"Error: invalid character {char} in decoded string {decoded}")
            predictions_04.append(pred_04)

    pred_binary = [class_04_to_binary(p) for p in predictions_04]

    with open(args.output_dev, "w", encoding="utf-8") as f:
        for b in pred_binary:
            f.write(f"{b}\n")
    print(f"Wrote {args.output_dev}")

    with open(args.output_dev_04, "w", encoding="utf-8") as f:
        for p in predictions_04:
            f.write(f"{p}\n")
    print(f"Wrote {args.output_dev_04}")

    tp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred_binary, gold_binary) if p == 0 and g == 1)
    accuracy = (tp + tn) / len(gold_binary) if gold_binary else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    incorrect_par_ids = [
        par_ids_ordered[i] for i in range(len(par_ids_ordered)) if pred_binary[i] != gold_binary[i]
    ]

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


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Mistral on PCL (excl. dev), save adapter, run validation"
    )
    parser.add_argument("--dev_path", type=str, default="dev_semeval_parids-labels.csv")
    parser.add_argument("--data_path", type=str, default="output/cleaned.tsv")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter_save_path", type=str, default="output/mistral_7b_pcl_lora")
    parser.add_argument("--output_dev", type=str, default="dev.txt")
    parser.add_argument("--output_dev_04", type=str, default="dev_04.txt")
    parser.add_argument("--output_metrics", type=str, default="dev_results.txt")
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Use few-shot examples in training and validation (default: no few-shot)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length for training (truncation only; no padding to this length)",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    # TODO: could maybe tune this to see which one is faster
    parser.add_argument("--train_batch_size", type=int, default=2) # 4 leads to OOM
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for validation generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load saved adapter and run validation only",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    dev_ids = load_dev_par_ids(args.dev_path)
    print(f"Loaded {len(dev_ids)} dev par_ids from {args.dev_path}")

    few_shot = load_few_shot_examples(args.data_path) if args.few_shot else []
    if args.few_shot:
        print(f"Using {len(few_shot)} few-shot examples")
    else:
        print("Using no few-shot (instruction + single text only)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.eval_only:
        print("Eval-only mode: skipping training, running validation with saved adapter.")
        run_validation(args, tokenizer, few_shot)
    else:
        train_examples = load_pcl_train(args.data_path, dev_ids)
        print(f"Training examples (PCL minus dev): {len(train_examples)}")
        # Phase 1 & 2: Train LoRA and save
        train_lora(args, tokenizer, train_examples, few_shot)
        # Phase 3: Validation
        run_validation(args, tokenizer, few_shot)


if __name__ == "__main__":
    main()
