#!/usr/bin/env python3
"""
Noise removal for dontpatronizeme_pcl.tsv: clean text in column index 4
(remove outer single quotes when both ends have odd runs of ", then collapse "" runs,
replace ' ' with ", then write a new TSV with only
par_id, text, label (no disclaimer).
"""
import argparse
import re


def remove_outer_single_quotes_if_odd_ends(s: str) -> str:
    """If both ends have an odd number of consecutive double quotes, remove one \" from each end."""
    if len(s) < 2 or s[0] != '"' or s[-1] != '"':
        return s
    start_run = 0
    for c in s:
        if c != '"':
            break
        start_run += 1
    end_run = 0
    for c in reversed(s):
        if c != '"':
            break
        end_run += 1
    if start_run % 2 == 1 and end_run % 2 == 1:
        return s[1:-1]
    return s


def collapse_double_quotes(s: str) -> str:
    """Replace runs of double-quote characters: even run 2k -> k quotes, odd 2k+1 -> k+1 quotes."""
    def replace_run(match: re.Match) -> str:
        n = len(match.group(0))
        k = (n + 1) // 2  # even: n/2, odd: (n+1)/2
        return '"' * k
    return re.sub(r'"+', replace_run, s)


def replace_single_quote_space(s: str) -> str:
    """Replace the three-char sequence single-quote space single-quote with one double-quote."""
    return s.replace("' '", '"')


def clean_text(text: str) -> str:
    """Apply noise-removal steps in order: remove outer single quotes when both ends odd, then collapse quotes and replace ' ', then remove outer wrapping pair."""
    t = remove_outer_single_quotes_if_odd_ends(text)
    t = collapse_double_quotes(t)
    t = replace_single_quote_space(t)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove noise from dontpatronizeme_pcl.tsv and write par_id, text, label to a new TSV."
    )
    parser.add_argument("--input", help="Path to input TSV (dontpatronizeme_pcl.tsv)")
    parser.add_argument("-o", "--output", required=True, help="Path to output TSV")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the column header row (par_id, text, label)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip first 4 lines (disclaimer)
    data_lines = lines[4:]

    output_rows = []
    for line in data_lines:
        row = line.rstrip("\n").split("\t")
        if len(row) != 6:
            print("ERROR: Expected 6 columns, got %d" % len(row))
            continue
        par_id, _, _, _, text, label = row[0], row[1], row[2], row[3], row[4], row[5]
        cleaned_text = clean_text(text)
        output_rows.append((par_id, cleaned_text, label))

    with open(args.output, "w", encoding="utf-8") as f:
        if not args.no_header:
            f.write("par_id\ttext\tlabel\n")
        for par_id, text, label in output_rows:
            f.write("\t".join([par_id, text, label]) + "\n")


if __name__ == "__main__":
    main()
