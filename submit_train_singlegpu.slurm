#!/bin/bash
# ============================================================================
# SLURM submission script: SINGLE-GPU training on RTX8000 (debug / safe)
# ============================================================================
# Usage:
#   mkdir -p /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs
#   cd /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn
#   sbatch submit_train_singlegpu.slurm
#
# Slurm .out / .err:  .../Catalytic_sites_setA_3dcnn/logs/cat3dcnn_1gpu_<jobid>.{out,err}
#
# IMPORTANT: run `sbatch` from a LOGIN node (e.g. hgd-aug), NOT from a
# compute node (compute-1-0 / compute-1-1). Many sites return
# "Access/permission denied" when submitting from compute nodes.
# ============================================================================
# Note: --time=0 is rejected on many clusters ("Access/permission denied").
#
# This cluster (hgd-aug) reports Gres=(null) on GPU nodes; do NOT use
# --gres=gpu:... or sbatch fails with "Requested node configuration is not
# available". The RTX8000 partition still routes jobs to the RTX node; use
# CUDA_VISIBLE_DEVICES below if needed.

#SBATCH --account=local
#SBATCH --job-name=cat3dcnn_1gpu
#SBATCH --partition=RTX8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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

echo "================================================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
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
# Launch SINGLE-GPU training (NO torchrun, plain python)
# ---------------------------------------------------------------------------
python -u train.py --config configs/single_gpu.yaml

echo "================================================================"
echo "Job finished: $(date)"
echo "================================================================"