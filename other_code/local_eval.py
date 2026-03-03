#!/usr/bin/env python3
"""
Local evaluation: build confusion matrices for binary dev predictions.
Uses dev_semeval_parids-labels.csv and output/cleaned.tsv for gold labels (0-4 -> binary),
and two dev.txt files (one prediction 0/1 per line in dev order).
"""

import argparse
import csv
import os


def class_04_to_binary(c: int) -> int:
    """Map class 0-4 to binary: 0-1 -> 0, 2-4 -> 1."""
    if c in (0, 1):
        return 0
    elif c in (2, 3, 4):
        return 1
    return 1


def load_dev_par_ids(dev_path: str) -> list[int]:
    """Load list of par_ids from dev CSV (par_id column), in order."""
    dev_ids = []
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_ids.append(int(row["par_id"]))
    return dev_ids


def load_cleaned_data(data_path: str) -> dict[int, tuple[str, int]]:
    """Load cleaned.tsv: par_id -> (text, label_int). Label is 0-4."""
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


def load_binary_predictions(txt_path: str) -> list[int]:
    """Load one int 0 or 1 per line from dev.txt."""
    preds = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            preds.append(int(line))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Confusion matrices for two dev prediction files.")
    parser.add_argument("--dev_csv", type=str, default="dev_semeval_parids-labels.csv", help="Dev par_id order CSV")
    parser.add_argument("--cleaned_tsv", type=str, default="output/cleaned.tsv", help="Cleaned TSV (par_id, text, label 0-4)")
    parser.add_argument(
        "--pred_files",
        type=str,
        nargs="+",
        default=[
            "dev/mistral_7b_pcl_lora_6_epochs/dev.txt",
            "dev/4_12_36_37_121/dev.txt",
        ],
        help="Paths to dev.txt (one 0/1 per line)",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save confusion matrix plots")
    args = parser.parse_args()

    # Gold binary in dev order
    dev_ids = load_dev_par_ids(args.dev_csv)
    all_data = load_cleaned_data(args.cleaned_tsv)
    gold_binary = []
    for pid in dev_ids:
        if pid in all_data:
            _, label_04 = all_data[pid]
            gold_binary.append(class_04_to_binary(label_04))
        else:
            gold_binary.append(-1)  # skip later

    # Filter to valid indices (gold >= 0)
    valid_idx = [i for i, g in enumerate(gold_binary) if g >= 0]
    y_true = [gold_binary[i] for i in valid_idx]

    def make_cm(y_true: list[int], y_pred: list[int]):
        """2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
        tn = fp = fn = tp = 0
        for t, p in zip(y_true, y_pred):
            if t == 0 and p == 0:
                tn += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
            else:
                tp += 1
        return [[tn, fp], [fn, tp]], (tn, fp, fn, tp)

    try:
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        sk_cm = None

    for pred_path in args.pred_files:
        name = os.path.basename(os.path.dirname(pred_path)) or os.path.splitext(os.path.basename(pred_path))[0] or "predictions"
        # Sanitize for filename (no path separators)
        name = name.replace(os.sep, "_").replace("/", "_")
        preds = load_binary_predictions(pred_path)
        if len(preds) != len(gold_binary):
            print(f"Skip {pred_path}: got {len(preds)} predictions, expected {len(gold_binary)}")
            continue
        y_pred = [preds[i] for i in valid_idx]
        cm, (tn, fp, fn, tp) = make_cm(y_true, y_pred)
        if sk_cm is not None:
            cm_sk = sk_cm(y_true, y_pred)
            cm = [[int(cm_sk[0, 0]), int(cm_sk[0, 1])], [int(cm_sk[1, 0]), int(cm_sk[1, 1])]]
            tn, fp, fn, tp = cm_sk.ravel()
        print(f"\n--- {pred_path} ({name}) ---")
        print("Confusion matrix (rows=true, cols=pred): [[TN, FP], [FN, TP]]")
        print(cm)
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        print(f"Accuracy: {acc:.4f}")

        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            continue

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix: {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, label="Count")
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"confusion_matrix_{name}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
