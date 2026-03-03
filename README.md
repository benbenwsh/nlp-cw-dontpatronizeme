# NLP Coursework: Don't Patronize Me (PCL)

Code and models for Patronizing and Condescending Language (PCL) detection (SemEval 2022): 5-way classification (0–4) with binary evaluation (0–1 → no PCL, 2–4 → PCL).

---

## Prediction outputs: `dev.txt` and `test.txt`

In the repository root: `dev.txt` and `test.txt` are binary predictions (one 0 or 1 per line, same order as dev/test paragraphs). Produced by validation in `run_gpt_lora.py`; used for local eval and submission.

---

## Repository structure

| Path | Description |
|------|-------------|
| `BestModel/` | Best LoRA adapter + script to train/evaluate it |
| `raw_data/` | SemEval TSVs, train/dev CSVs, cleaned/augmented PCL data |
| `other_code/` | Zero-shot, ordinal, augmentation, noise removal, local_eval, error_analysis |
| `graphs/` | Confusion matrices, error-analysis plots |

---

## BestModel

Model: `BestModel/mistral_7b_pcl_lora_6_epochs/` — PEFT LoRA adapter for `mistralai/Mistral-7B-Instruct-v0.2` (adapter weights, adapter_config.json, tokenizer files). Load with `PeftModel.from_pretrained(base_model, "BestModel/mistral_7b_pcl_lora_6_epochs")`.

Script: `BestModel/run_gpt_lora.py` — Trains Mistral-7B with LoRA on PCL (cleaned TSV, dev set excluded from train); optional `--few_shot`. Saves adapter to `--adapter_save_path`, runs validation and writes `dev.txt` (binary), `dev_04.txt`, `dev_results.txt`. Use `--eval_only` to skip training and only run validation with a saved adapter.

---

## Quick start

```bash
pip install -r requirements.txt
```

Eval-only (writes `dev.txt` in current dir):

```bash
python BestModel/run_gpt_lora.py --eval_only \
  --data_path raw_data/cleaned_augmented_0.6.tsv \
  --dev_path raw_data/dev_semeval_parids-labels.csv \
  --adapter_save_path BestModel/mistral_7b_pcl_lora_6_epochs --output_dev dev.txt
```

Train new adapter:

```bash
python BestModel/run_gpt_lora.py --data_path raw_data/cleaned_augmented_0.6.tsv \
  --dev_path raw_data/dev_semeval_parids-labels.csv \
  --adapter_save_path output/my_lora_run --num_epochs 6
```
