#!/usr/bin/env python3
"""
Error analysis for mistral_7b_pcl_lora_6_epochs dev predictions.
Part 1: Errors by keyword (binary 0/1 wrong).
Part 2: Errors by gold class 0-4 (count + percentage), two bar charts.
Part 3: Length distribution (correct vs incorrect).
Uses dev_semeval_parids-labels.csv, output/cleaned.tsv, dev_results.txt, dev_04.txt;
for Part 1 only, dontpatronizeme_pcl.tsv for keyword.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


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


def load_incorrect_par_ids(results_path: str) -> list[int]:
    """Parse dev_results.txt: find 'Incorrect examples (par_id):', then read par_ids."""
    par_ids = []
    with open(results_path, "r", encoding="utf-8") as f:
        found = False
        for line in f:
            line = line.strip()
            if "Incorrect examples (par_id):" in line:
                found = True
                continue
            if not found:
                continue
            if not line or line.startswith("-"):
                continue
            try:
                par_ids.append(int(line))
            except ValueError:
                break
    return par_ids


def load_par_id_to_keyword(pcl_tsv_path: str) -> dict[int, str]:
    """Load par_id -> keyword from dontpatronizeme_pcl.tsv. Skip first 4 disclaimer lines; 6 columns, keyword is 3rd (index 2)."""
    data = {}
    with open(pcl_tsv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            par_id = int(parts[0])
            keyword = parts[2]
            data[par_id] = keyword
    return data


def load_predictions_04(txt_path: str) -> list[int]:
    """Load one int 0-4 per line from dev_04.txt."""
    preds = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            preds.append(int(line))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Error analysis for mistral_7b_pcl_lora_6_epochs dev.")
    parser.add_argument("--dev_csv", type=str, default="dev_semeval_parids-labels.csv", help="Dev par_id order CSV")
    parser.add_argument("--cleaned_tsv", type=str, default="output/cleaned.tsv", help="Cleaned TSV (par_id, text, label 0-4)")
    parser.add_argument("--dev_dir", type=str, default="dev/mistral_7b_pcl_lora_6_epochs", help="Dir with dev.txt, dev_04.txt, dev_results.txt")
    parser.add_argument("--pcl_tsv", type=str, default="dontpatronizeme_pcl.tsv", help="Raw PCL TSV for keyword (Part 1 only)")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to write stats and charts")
    args = parser.parse_args()

    dev_results_path = os.path.join(args.dev_dir, "dev_results.txt")
    dev_04_path = os.path.join(args.dev_dir, "dev_04.txt")

    # Load dev order and cleaned data
    dev_ids = load_dev_par_ids(args.dev_csv)
    all_data = load_cleaned_data(args.cleaned_tsv)
    incorrect_par_ids = set(load_incorrect_par_ids(dev_results_path))

    # Gold 0-4 in dev order (same order as dev_04.txt)
    gold_04 = []
    for pid in dev_ids:
        if pid in all_data:
            _, label = all_data[pid]
            gold_04.append(label)
        else:
            gold_04.append(-1)  # missing

    pred_04 = load_predictions_04(dev_04_path)
    if len(pred_04) != len(gold_04):
        raise ValueError(f"dev_04.txt has {len(pred_04)} lines but dev has {len(gold_04)} rows")

    # --- Part 1: Errors by keyword ---
    par_id_to_keyword = load_par_id_to_keyword(args.pcl_tsv)
    keyword_counts = Counter()  # incorrect count per keyword
    keyword_totals = Counter()   # total dev count per keyword (for percentage)
    for pid in dev_ids:
        if pid in par_id_to_keyword:
            keyword_totals[par_id_to_keyword[pid]] += 1
    for pid in incorrect_par_ids:
        if pid in par_id_to_keyword:
            keyword_counts[par_id_to_keyword[pid]] += 1
        else:
            keyword_counts["(unknown)"] += 1
    keyword_pct_wrong = {
        kw: (100.0 * keyword_counts[kw] / keyword_totals[kw]) if keyword_totals[kw] else 0.0
        for kw in set(keyword_counts) | set(keyword_totals)
    }

    # --- Part 2: Errors by gold class 0-4 ---
    errors_by_gold = defaultdict(int)
    total_by_gold = defaultdict(int)
    for i, (g, p) in enumerate(zip(gold_04, pred_04)):
        if g < 0:
            continue
        total_by_gold[g] += 1
        if g != p:
            errors_by_gold[g] += 1

    # Percentages
    pct_wrong = {}
    for c in range(5):
        tot = total_by_gold.get(c, 0)
        err = errors_by_gold.get(c, 0)
        pct_wrong[c] = (100.0 * err / tot) if tot else 0.0

    # --- Part 3: Lengths (word count) ---
    lengths_correct = []
    lengths_incorrect = []
    for pid in dev_ids:
        if pid not in all_data:
            continue
        text, _ = all_data[pid]
        nwords = len(text.split())
        if pid in incorrect_par_ids:
            lengths_incorrect.append(nwords)
        else:
            lengths_correct.append(nwords)

    # --- Write stats file ---
    stats_path = os.path.join(args.output_dir, "error_analysis_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Error analysis (mistral_7b_pcl_lora_6_epochs dev)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Part 1: Errors by keyword (binary 0/1 wrong)\n")
        f.write("-" * 40 + "\n")
        for kw, count in keyword_counts.most_common():
            tot = keyword_totals.get(kw, 0)
            pct = keyword_pct_wrong.get(kw, 0.0)
            f.write(f"  {kw}: incorrect={count}, total={tot}, pct_wrong={pct:.2f}%\n")
        f.write("\n")

        f.write("Part 2: Errors by gold class (0-4)\n")
        f.write("-" * 40 + "\n")
        for c in range(5):
            tot = total_by_gold.get(c, 0)
            err = errors_by_gold.get(c, 0)
            pct = pct_wrong[c]
            f.write(f"  Class {c}: incorrect={err}, total={tot}, pct_wrong={pct:.2f}%\n")
        f.write("\n")

        f.write("Part 3: Length (word count) summary\n")
        f.write("-" * 40 + "\n")
        if lengths_incorrect:
            f.write(f"  Incorrect: n={len(lengths_incorrect)}, mean={sum(lengths_incorrect)/len(lengths_incorrect):.1f}\n")
        if lengths_correct:
            f.write(f"  Correct:   n={len(lengths_correct)}, mean={sum(lengths_correct)/len(lengths_correct):.1f}\n")
    print(f"Wrote {stats_path}")

    # --- Charts (matplotlib) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping charts.")
        return

    # Part 1: keyword bar chart (count)
    if keyword_counts:
        keywords = [k for k, _ in keyword_counts.most_common()]
        counts = [keyword_counts[k] for k in keywords]
        fig, ax = plt.subplots()
        ax.bar(keywords, counts)
        ax.set_xlabel("Keyword")
        ax.set_ylabel("Count of incorrect examples")
        ax.set_title("Part 1: Errors by keyword (binary wrong)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(args.output_dir, "error_analysis_keyword_bars.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote {out}")

    # Part 1: keyword bar chart (percentage) — only keywords with a dev total
    if keyword_pct_wrong:
        keywords_pct = [k for k, _ in keyword_counts.most_common() if keyword_totals.get(k, 0) > 0]
        if keywords_pct:
            pcts_kw = [keyword_pct_wrong[k] for k in keywords_pct]
            fig, ax = plt.subplots()
            ax.bar(keywords_pct, pcts_kw)
            ax.set_xlabel("Keyword")
            ax.set_ylabel("% of that keyword incorrect")
            ax.set_title("Part 1: Percentage wrong by keyword")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out = os.path.join(args.output_dir, "error_analysis_keyword_pct_bars.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Wrote {out}")

    # Part 2: bar chart 1 (count)
    classes = list(range(5))
    err_counts = [errors_by_gold.get(c, 0) for c in classes]
    fig, ax = plt.subplots()
    ax.bar(classes, err_counts)
    ax.set_xlabel("Gold class")
    ax.set_ylabel("Count of incorrect examples")
    ax.set_title("Part 2: Count of errors by gold class (0-4)")
    ax.set_xticks(classes)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "error_analysis_class_count_bars.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    # Part 2: bar chart 2 (percentage)
    pcts = [pct_wrong[c] for c in classes]
    fig, ax = plt.subplots()
    ax.bar(classes, pcts)
    ax.set_xlabel("Gold class")
    ax.set_ylabel("% of that class incorrect")
    ax.set_title("Part 2: Percentage wrong by gold class (0-4)")
    ax.set_xticks(classes)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "error_analysis_class_pct_bars.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    # Part 3: length distribution (shared bin range for comparability)
    all_lengths = lengths_correct + lengths_incorrect
    bins = min(40, max(10, len(set(all_lengths)))) if all_lengths else 20
    fig, ax = plt.subplots()
    if lengths_incorrect:
        ax.hist(lengths_incorrect, bins=bins, alpha=0.6, label="Incorrect", density=True)
    if lengths_correct:
        ax.hist(lengths_correct, bins=bins, alpha=0.6, label="Correct", density=True)
    ax.set_xlabel("Text length (word count)")
    ax.set_ylabel("Density")
    ax.set_title("Part 3: Length distribution (correct vs incorrect)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(args.output_dir, "error_analysis_length_dist.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()
