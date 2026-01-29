# SFT Ensembles: Subset SFT + LoRA Merge (Turkish)

This repository contains a small experimental study on **SFT ensembles** using the same Turkish SFT dataset with:
- **Random subsets**: R1, R2 (2000 samples each)
- **Category-disjoint subsets**: T1, T2 (2000 samples each, disjoint category sets)
- **Full training split**: FULL

Each run performs **SFT with QLoRA + LoRA**. After training, two merged adapters are produced by **simple parameter-space averaging**:
- `MERGE_R = avg(R1, R2)`
- `MERGE_T = avg(T1, T2)`

Evaluation is reported on a **500-sample test set** using:
- `eval_loss` (lower is better)
- `perplexity (ppl)` (lower is better)

## Model and Dataset
- Base model: `ytu-ce-cosmos/tr-Qwen2.5-0.5B-SFT-v1`
- Dataset: `merve/tr-h4-norobots` (fields: `prompt`, `message`, `category`)

## Files
- `sft_ensembles_clean.ipynb` — main notebook (training, merging, evaluation, figures)
- `results_summary.csv` — final test metrics (eval_loss, ppl) per run
- `figures/` — report-ready plots (pipeline diagram, eval_loss bar, training loss curves)

## How to Run (Google Colab recommended)
1. Open the notebook in Colab.
2. Install dependencies:
   ```bash
   pip install -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub pandas numpy matplotlib
   ```

## Hugging Face token

If you need authentication, add a Colab secret named 'HF_TOKEN'.
Run the notebook cells in order.

## Key settings (inside the notebook)

`SMOKE_TEST`: True for quick sanity checks, False for full runs

`TRAIN_N`: subset size per run (set to >= 2000 for the project requirement)

`TEST_N`: evaluation size (set to >= 500 for the project requirement)

## Acknowledgements / References

Dataset: https://huggingface.co/datasets/merve/tr-h4-norobots
Base model: https://huggingface.co/ytu-ce-cosmos/tr-Qwen2.5-0.5B-SFT-v1
