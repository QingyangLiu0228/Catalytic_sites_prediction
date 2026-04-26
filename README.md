# Catalytic Site Prediction — Set A (3D CNN)

Per-residue binary catalytic site prediction from AlphaFold/PDB structures and
ESM-2 3B embeddings. Residues are classified as "catalytic" vs "non-catalytic"
using a small 3D CNN over a 20 Å voxel box centered on each residue, where
each occupied voxel carries the 2560-dim ESM-2 embedding of the neighboring
residue whose Cα falls into it.

## Directory structure

```
Catalytic_site_prediction_setA_3dcnn/
├── configs/
│   ├── default.yaml        # production hyperparameters
│   └── debug.yaml          # fast end-to-end test (~minutes)
├── data.py                 # feature extraction, Dataset, 1:k_neg sampling
├── cnn3d.py                # CNN3D model + ResidualBlock3D (reserved)
├── train.py                # DDP-enabled training entry point
├── evaluate.py             # per-batch validation generator + log plotting
├── utils.py                # YAML config loader, dict<->Namespace helpers
├── submit_train.slurm      # SLURM script for 8-GPU RTX8000 training
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

The project runs inside the `squidly` conda environment on BioHPC. If you
need to recreate it elsewhere:

```bash
conda create -n squidly python=3.9
conda activate squidly
pip install -r requirements.txt
```

## Quick start (debug run)

Small dataset, single seed, single GPU — should finish in a few minutes.
Use this whenever you change code and want to check nothing is broken before
launching the real job.

```bash
cd /data/data3/conglab/s441865/code/Catalytic_site_prediction_setA_3dcnn
python train.py --config configs/debug.yaml
```

Outputs land in `Results/run_seed42/` (config snapshot, training log, best
model).

## Full training (8-GPU DDP on RTX8000)

```bash
sbatch submit_train.slurm
```

This launches `torchrun --nproc_per_node=8 train.py --config configs/default.yaml`
under SLURM. One seed trains at a time using all 8 GPUs; seeds themselves run
sequentially.

Monitor with:
```bash
squeue -u $USER
tail -f logs/cat3dcnn_<jobid>.out
```

## Config files

Only the YAML file controls hyperparameters; there is no command-line
override. To run a different experiment, create a new file under `configs/`
that overrides only the fields you need. `utils.load_config` auto-merges any
non-default config on top of `default.yaml`.

Example: `configs/lr_sweep_4e3.yaml`
```yaml
lr: 4.0e-3
num_epochs: 30
```

## Output format

Each seed produces a `Results/run_seed<seed>/` directory with:

| File (pattern) | Content |
|------|---------|
| `config_YYYYMMDD_HHMMSS.yaml` | The exact config this run used (after merging). Same time stem on all run artifacts. |
| `train_log_YYYYMMDD_HHMMSS.jsonl` | One line per **training** batch (rank 0 in DDP). Per-batch loss, class metrics, `probs`/`preds`/`labels`/`sample_ids` (`pdb\|chain\|resnum`). |
| `val_log_YYYYMMDD_HHMMSS.jsonl` | One line per **validation** batch. Same style; `phase: val`. `evaluate.py` plots use this file. |
| `best_model_YYYYMMDD_HHMMSS.pt` | State dict of the epoch where the best **batch-AUPRC** was observed, plus the metrics at that batch. |
| `training_YYYYMMDD_HHMMSS.log` | Warnings/errors from the `logging` module. |

After all seeds finish, `Results/summary.json` contains per-seed best metrics
and mean/std across seeds.

To regenerate the validation-curve plot (use the `val_log_*.jsonl` path for your run):
```bash
python evaluate.py --jsonl Results/run_seed42/val_log_20260422_234500.jsonl
```

## Known limitations

These are design choices made during the initial port — documented so future
you knows what to audit before writing the paper.

1. **Batch-level AUPRC/AUROC are not globally comparable.** `auprc` / `auroc` in
   `val_log` are computed on each batch of ~64 samples independently, not on
   the full validation set. Batch-mean AUPRC is noisy (~6 positive samples
   per batch at `k_neg=10`) and **cannot be directly compared** to CLEAN,
   Squidly, or EasIFA numbers, which are global. Use these per-batch
   numbers as training diagnostics, not as final reported metrics.

2. **Validation is 1:10 balanced.** Both train and val are downsampled to a
   1:10 positive:negative ratio. Real catalytic-site prevalence is closer
   to 1–5%, so the AUPRC you see here is inflated relative to a real-world
   deployment scenario. A separate original-distribution evaluation step is
   recommended before publishing.

3. **Checkpoint selection uses best batch-AUPRC.** `best_model_<stamp>.pt` is saved
   at the epoch where some validation batch hit the highest AUPRC. Because
   batch-AUPRC is noisy, the saved checkpoint may not be the truly-best
   epoch. Consider replacing the checkpointing criterion with a global
   per-epoch metric before the final training runs.

4. **PDB length check includes HETATM.** `compare_pdb_esm_lengths` counts
   waters and ligands when comparing PDB chain length against ESM embedding
   length. Some proteins may be filtered out (or let through) in ways that
   do not reflect the standard-residue count. Kept for parity with the
   original script.

5. **Threshold defaults to 0.9.** Precision / recall / F1 columns in the
   log will often be 0 because the model rarely exceeds 0.9 probability.
   AUPRC and AUROC are threshold-free and remain meaningful.

6. **DDP scaling requires lr/epoch retuning.** With 8 GPUs, effective batch
   size = `train_batch_size * 8 = 256`. The single-GPU learning rate
   (`lr=1e-3`) and epoch count (`num_epochs=5`) are likely too small. Try
   `lr ≈ 2–4e-3` and `num_epochs ≈ 20–40` for the first real multi-GPU run.

## Contact

Lab: Cong Lab, UT Southwestern
User: s441865
