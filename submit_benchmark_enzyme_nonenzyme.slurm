#!/bin/bash
# ============================================================================
# SLURM submission: SINGLE-GPU multi-seed enzyme vs non-enzyme benchmark
# ============================================================================
# Mirrors the GNN repo's submit_benchmark_enzyme_nonenzyme.slurm but for the
# 3D-CNN (Catalytic_sites_setA_3dcnn). Runs benchmark_enzyme_nonenzyme.py
# against every results/run_seed*/ checkpoint (or the seeds listed in the
# config), scores enzyme + non-enzyme PDB sets PDB-by-PDB, and writes:
#
#   <results_dir>/run_seed<seed>/bench_enzyme_<stamp>.csv
#   <results_dir>/run_seed<seed>/bench_nonenzyme_<stamp>.csv
#   <results_dir>/run_seed<seed>/bench_enzyme_nonenzyme_metrics_<stamp>.json
#   <results_dir>/run_seed<seed>/bench_enzyme_nonenzyme_threshold_table_<stamp>.txt
#   <results_dir>/run_seed<seed>/bench_enzyme_nonenzyme_curves_<stamp>.png
#   <results_dir>/benchmark_enzyme_nonenzyme_summary.json
#   <results_dir>/benchmark_enzyme_nonenzyme_overlay.png
#   <results_dir>/config_benchmark_<stamp>.yaml
#
# Usage:
#   mkdir -p /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs
#   cd      /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn
#   sbatch  submit_benchmark_enzyme_nonenzyme.slurm                      # full benchmark
#   sbatch  submit_benchmark_enzyme_nonenzyme.slurm configs/benchmark_debug.yaml  # smoke test
#
# Optional: space-separated explicit best_model paths (overrides YAML / run_dirs):
#   sbatch --export=CONFIG=configs/benchmark.yaml,CHECKPOINTS="/path/a.pt /path/b.pt" ...
# ============================================================================

#SBATCH --account=local
#SBATCH --job-name=cat3dcnn_bench
#SBATCH --partition=RTX8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.out
#SBATCH --error=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.err

# Notes:
#   * --time=0 is rejected on many clusters; use a finite walltime.
#   * hgd-aug reports Gres=(null) on GPU nodes; do NOT use --gres=gpu:...
#     The RTX8000 partition still routes jobs to the RTX node; export
#     CUDA_VISIBLE_DEVICES below to pin a single GPU.
#   * Do not use `set -u`: conda's libblas_mkl_activate.sh trips it.

set -eo pipefail
PROJECT_ROOT="/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn"
mkdir -p "${PROJECT_ROOT}/logs"

# ---- Config / run-dir overrides ------------------------------------------
# Priority: CONFIG / RUN_DIRS env vars (--export=...) > positional arg.
CONFIG="${CONFIG:-${1:-configs/benchmark.yaml}}"

echo "================================================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
echo "Benchmark config: ${CONFIG}"
if [ -n "${RUN_DIRS}" ]; then
    echo "RUN_DIRS override: ${RUN_DIRS}"
fi
echo "================================================================"

# ---- Conda env (squidly, same as test.py) --------------------------------
CONDA_SH="/home2/s441865/anaconda3/etc/profile.d/conda.sh"
SQUIDLY_ENV="/home2/s441865/anaconda3/envs/squidly"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda not found at ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${SQUIDLY_ENV}" ]]; then
  echo "ERROR: conda env not found at ${SQUIDLY_ENV}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${SQUIDLY_ENV}"

echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Launch SINGLE-GPU benchmark (no DDP, no torchrun)
# ---------------------------------------------------------------------------
CMD="python -u benchmark_enzyme_nonenzyme.py --config ${CONFIG}"
if [ -n "${CHECKPOINTS}" ]; then
  CMD="${CMD} --checkpoints ${CHECKPOINTS}"
  echo "CHECKPOINTS override: ${CHECKPOINTS}"
fi
if [ -n "${RUN_DIRS}" ]; then
    CMD="${CMD} --run_dirs ${RUN_DIRS}"
fi

echo "Running: ${CMD}"
export PYTHONUNBUFFERED=1
${CMD}

echo "================================================================"
echo "Job finished: $(date)"
echo "================================================================"
