#!/usr/bin/env python3
"""
Train only an ordinal regression head on frozen Mistral-7B-Instruct-v0.2 for PCL
classification (5 ordered classes 0-4). No LoRA; LM head is not used. Uses last-token
hidden state and CORAL ordinal regression (coral-pytorch).
"""

import argparse
import csv
import glob
import os
import random
import re
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label

# --- Data loading and prompt helpers (same as run_mistral_lora.py) ---

INSTRUCTION = (
    "You will be given a text from a paragraph. You need to identify whether "
    "the paragraph contains language that is patronizing or condescending "
    "towards vulnerable communities. This type of language is also known as "
    "PCL (Patronizing and Condescending Language). "
    "Classify the following text into a single class 0, 1, 2, 3, or 4 "
    "(0 = no PCL, 4 = strongest PCL). Reply with only one digit. "
)

NUM_CLASSES = 5


def class_04_to_binary(c: int) -> int:
    """Map class 0-4 to binary: 0-1 -> 0, 2-4 -> 1."""
    if c in (0, 1):
        return 0
    elif c in (2, 3, 4):
        return 1
    else:
        print(f"ERROR: invalid class {c}")
        return 1


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
    """Load fixed few-shot examples from cleaned.tsv."""
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
            max_length=max_length, # My use case rarely exceeds even the default 2048 args.max_length
        ) # of type BatchEncoding, similar to dictionary
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


# --- Dataset: prompt only, no answer token; label is 0-4 ---


class OrdinalDataCollator:
    """Pad batch to longest sequence; only input_ids and attention_mask (no LM labels)."""

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
            input_ids = f["input_ids"].tolist() + [self.pad_id] * pad_len
            attention_mask = f["attention_mask"].tolist() + [0] * pad_len
            batch["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            batch["attention_mask"].append(torch.tensor(attention_mask, dtype=torch.long))
            batch["labels"].append(f["labels"])
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])
        return batch


class PCLOrdinalDataset(Dataset):
    """Dataset of (input_ids, attention_mask, label) for ordinal head; prompt only, no answer token."""

    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer,
        max_length: int, # max length of the prompt
        few_shot: List[Tuple[str, int]],
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot = few_shot
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        prompt_text = build_prompt(self.few_shot, text)
        messages = [{"role": "user", "content": prompt_text}]
        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_length,
        )

        prompt_ids = out["input_ids"]
        prompt_ids = [int(x) for x in prompt_ids]

        # No padding because only considering one example
        attention_mask = [1] * len(prompt_ids)
        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# --- Model: frozen Mistral + ordinal head (CORAL) ---


class MistralOrdinalModel(nn.Module):
    """
    Frozen Mistral backbone + trainable CORAL ordinal head.
    Uses last-token hidden state from the transformer (no LM head).
    """

    def __init__(self, model_name: str, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.backbone.requires_grad_(False) # Freeze GPT backbone
        hidden_size = self.backbone.config.hidden_size # Feature dimension (4096 for Mistral-7B-v0.2)
        self.ordinal_head = CoralLayer(size_in=hidden_size, num_classes=num_classes)
        # Place head on same device as backbone output (last layer) when using device_map="auto"
        if torch.cuda.is_available():
            last_layer_device = next(self.backbone.model.layers[-1].parameters()).device
            self.ordinal_head.to(last_layer_device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        last_hidden = outputs.last_hidden_state # last token's hidden state
        batch_size = last_hidden.size(0) # First dimension is the batch size
        last_pos = attention_mask.sum(dim=1) - 1
        device = last_hidden.device
        last_token_hidden = last_hidden[torch.arange(batch_size, device=device), last_pos, :]
        if last_token_hidden.dtype == torch.bfloat16:
            last_token_hidden = last_token_hidden.float()
        logits = self.ordinal_head(last_token_hidden) # (batch, num_classes-1)

        if labels is not None:
            levels = levels_from_labelbatch(labels, num_classes=self.num_classes, dtype=logits.dtype).to(logits.device)
            loss = coral_loss(logits, levels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def compute_metrics_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL logits to class predictions 0-4. logits shape (batch, num_classes-1)."""
    probas = torch.sigmoid(logits)
    return proba_to_label(probas)


def _get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to the latest checkpoint in output_dir (checkpoint-XXXX), or None if none found."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    def step(path: str) -> int:
        m = re.search(r"checkpoint-(\d+)$", path.rstrip(os.sep))
        return int(m.group(1)) if m else 0
    return max(checkpoints, key=step)


def _compute_eval_metrics(eval_preds) -> dict:
    """Compute binary F1 and accuracy (0-1 vs 2-4) from Trainer eval predictions. predictions = logits (N, 4), label_ids = (N,) 0-4."""
    predictions, label_ids = eval_preds.predictions, eval_preds.label_ids
    logits = torch.tensor(predictions, dtype=torch.float32)
    preds_04 = compute_metrics_from_logits(logits)
    pred_binary = [class_04_to_binary(p.item()) for p in preds_04]
    gold_binary = [class_04_to_binary(int(l)) for l in label_ids]
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
    """Extract train loss, eval loss, eval F1, and eval accuracy per epoch from trainer.state.log_history; save four plots."""
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

    # Write per-epoch metrics to a txt file
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
    if eval_epochs and not all(f != f for f in eval_f1s):  # at least one non-NaN F1
        fig3, ax3 = plt.subplots()
        ax3.plot(eval_epochs, eval_f1s, marker="o", linestyle="-", color="C2")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1")
        ax3.set_title("Validation F1 vs epoch")
        ax3.set_xticks(eval_epochs)
        fig3.savefig(os.path.join(save_dir, "eval_f1_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved {os.path.join(save_dir, 'eval_f1_vs_epoch.png')}")
    if eval_epochs and not all(a != a for a in eval_accuracies):  # at least one non-NaN accuracy
        fig4, ax4 = plt.subplots()
        ax4.plot(eval_epochs, eval_accuracies, marker="o", linestyle="-", color="C3")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Validation accuracy vs epoch")
        ax4.set_xticks(eval_epochs)
        fig4.savefig(os.path.join(save_dir, "eval_accuracy_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved {os.path.join(save_dir, 'eval_accuracy_vs_epoch.png')}")


def train_ordinal(args, tokenizer, train_examples, few_shot):
    """Load frozen Mistral + ordinal head, train head only, save head and tokenizer."""
    print("Loading base model (frozen) and ordinal head...")
    model = MistralOrdinalModel(args.model_name, num_classes=NUM_CLASSES)
    model.train()
    n_trainable = sum(p.numel() for p in model.ordinal_head.parameters() if p.requires_grad)
    print(f"Trainable parameters (ordinal head only): {n_trainable}")

    train_dataset = PCLOrdinalDataset(train_examples, tokenizer, args.max_length, few_shot)
    dev_ids = load_dev_par_ids(args.dev_path)
    all_data = load_cleaned_data(args.data_path)
    eval_examples = [all_data[par_id] for par_id in dev_ids]
    eval_dataset = PCLOrdinalDataset(eval_examples, tokenizer, args.max_length, few_shot)
    print(f"Eval examples (dev): {len(eval_dataset)}")

    data_collator = OrdinalDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.head_save_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, # max no of checkpoints to keep
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_eval_metrics,
    )
    resume_from = None
    if args.resume:
        resume_from = _get_latest_checkpoint(args.head_save_path)
        if resume_from:
            print(f"Resuming training from {resume_from}")
        else:
            print(f"WARNING: --resume set but no checkpoint found in {args.head_save_path}; training from scratch")
    time_start = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    time_end = time.time()
    print(f"Training time: {time_end - time_start:.1f} seconds")

    _plot_train_eval_loss(trainer, args.head_save_path)

    os.makedirs(args.head_save_path, exist_ok=True)
    torch.save(model.ordinal_head.state_dict(), os.path.join(args.head_save_path, "ordinal_head.pt"))
    tokenizer.save_pretrained(args.head_save_path)
    print(f"Saved ordinal head and tokenizer to {args.head_save_path}")
    return model


def run_validation(args, tokenizer, few_shot):
    """Load frozen Mistral + trained ordinal head, run on dev set, write predictions and metrics."""
    print("Loading base model and ordinal head for validation...")
    model = MistralOrdinalModel(args.model_name, num_classes=NUM_CLASSES)
    head_path = os.path.join(args.head_save_path, "ordinal_head.pt")
    model.ordinal_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    if torch.cuda.is_available():
        last_layer_device = next(model.backbone.model.layers[-1].parameters()).device
        model.ordinal_head.to(last_layer_device)
    model.eval()

    dev_ids = load_dev_par_ids(args.dev_path)
    all_data = load_cleaned_data(args.data_path)
    validation_list = []
    for par_id in dev_ids:
        if par_id in all_data:
            text, label = all_data[par_id]
            validation_list.append((par_id, text, label))
        else:
            print(f"ERROR: par_id {par_id} not in cleaned data")
    print(f"Validation samples: {len(validation_list)}")

    gold_binary = [class_04_to_binary(l) for (_, _, l) in validation_list]
    par_ids_ordered = [p for (p, _, _) in validation_list]
    device = next(model.parameters()).device
    batch_size = max(1, args.batch_size)

    predictions_04 = []
    for start in tqdm(range(0, len(validation_list), batch_size), desc="Validating"):
        batch_items = validation_list[start : start + batch_size]
        batch_prompts = [build_prompt(few_shot, text) for _, text, _ in batch_items]
        input_ids, attention_mask = tokenize_batch_with_chat_template(
            tokenizer, batch_prompts, max_length=args.max_length, device=device
        )
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        preds = compute_metrics_from_logits(logits)
        for i in range(preds.size(0)):
            predictions_04.append(preds[i].item())

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
    gold_04 = [l for (_, _, l) in validation_list]
    mae = sum(abs(p - g) for p, g in zip(predictions_04, gold_04)) / len(gold_04) if gold_04 else 0.0
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
        f.write(f"MAE (0-4): {mae:.4f}\n")
        f.write("\n")
        f.write(f"Incorrect examples (par_id): {len(incorrect_par_ids)}\n")
        f.write("-" * 50 + "\n")
        for pid in incorrect_par_ids:
            f.write(f"{pid}\n")
    print(f"Wrote {args.output_metrics}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MAE: {mae:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ordinal regression head on frozen Mistral for PCL (0-4), then validate"
    )
    parser.add_argument("--dev_path", type=str, default="dev_semeval_parids-labels.csv")
    parser.add_argument("--data_path", type=str, default="output/cleaned.tsv")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--head_save_path", type=str, default="output/mistral_7b_ordinal_head")
    parser.add_argument("--output_dev", type=str, default="dev.txt")
    parser.add_argument("--output_dev_04", type=str, default="dev_04.txt")
    parser.add_argument("--output_metrics", type=str, default="dev_results.txt")
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Use few-shot examples in prompt (default: no few-shot)",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length (truncation)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load saved ordinal head and run validation only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in --head_save_path (use with --num_epochs for total epochs)",
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
        print("Eval-only mode: skipping training, running validation with saved head.")
        run_validation(args, tokenizer, few_shot)
    else:
        train_examples = load_pcl_train(args.data_path, dev_ids)
        print(f"Training examples (PCL minus dev): {len(train_examples)}")
        train_ordinal(args, tokenizer, train_examples, few_shot)
        run_validation(args, tokenizer, few_shot)


if __name__ == "__main__":
    main()
