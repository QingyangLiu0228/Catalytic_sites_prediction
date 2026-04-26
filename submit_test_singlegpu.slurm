#!/bin/bash
# ============================================================================
# SLURM submission script: SINGLE-GPU TESTING on RTX8000
# ============================================================================
# Runs test.py to evaluate the 3 setA-trained checkpoints (seed 42 / 123 /
# 2026) on setB_test, in both balanced (1:k_neg) and full modes. Results
# (per-batch jsonl logs, test_summary.json, PR-curve PNGs) are written
# under /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/results/test/
#
# Usage:
#   mkdir -p /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs
#   cd      /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn
#   sbatch  submit_test_singlegpu.slurm                       # full test
#   sbatch  submit_test_singlegpu.slurm configs/test_debug.yaml  # quick smoke test
#
# Slurm .out / .err: .../Catalytic_sites_setA_3dcnn/logs/cat3dcnn_test_<jobid>.{out,err}
#
# IMPORTANT: run `sbatch` from a LOGIN node (e.g. hgd-aug), NOT from a
# compute node. Many sites return "Access/permission denied" when submitting
# from compute nodes.
# ============================================================================
# Notes (carried over from submit_train_singlegpu.slurm):
#   - --time=0 is rejected on many clusters; use a finite walltime below.
#   - This cluster (hgd-aug) reports Gres=(null) on GPU nodes; do NOT use
#     --gres=gpu:... or sbatch fails. The RTX8000 partition still routes
#     jobs to the RTX node; use CUDA_VISIBLE_DEVICES below if needed.

#SBATCH --account=local
#SBATCH --job-name=cat3dcnn_test
#SBATCH --partition=RTX8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Log files always under the project (not $PWD when you sbatch from ~):
#SBATCH --output=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.out
#SBATCH --error=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.err

# Parent dir must exist before submit (or create once):
#   mkdir -p /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs

# Do not use "set -u" (nounset): conda's etc/conda/activate.d/libblas_mkl_activate.sh
# touches MKL_INTERFACE_LAYER and fails with "unbound variable" under -u.
set -eo pipefail
PROJECT_ROOT="/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn"
mkdir -p "${PROJECT_ROOT}/logs"

# Allow overriding the test config from the sbatch CLI:
#   sbatch submit_test_singlegpu.slurm configs/test_debug.yaml
TEST_CONFIG="${1:-configs/test.yaml}"

echo "================================================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
echo "Test config: ${TEST_CONFIG}"
echo "================================================================"

# Conda: same anaconda3 as in ~/.bashrc. Env: squidly (not GNN).
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
# Launch SINGLE-GPU testing (no torchrun; test.py is single-GPU only)
# ---------------------------------------------------------------------------
python -u test.py --config "${TEST_CONFIG}"

echo "================================================================"
echo "Job finished: $(date)"
echo "================================================================"
