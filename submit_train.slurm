#!/bin/bash
# ============================================================================
# SLURM submission script: 8-GPU DDP training on RTX8000
# ============================================================================
# Usage:
#   cd /data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn
#   sbatch submit_train.slurm
#
# After submission:
#   squeue -u $USER                # see queued / running jobs
#   tail -f logs/cat3dcnn_*.out    # live-follow stdout
# ============================================================================

#SBATCH --account=local
#SBATCH --job-name=cat3dcnn
#SBATCH --partition=RTX8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# NOTE: This cluster reports Gres=(null); do NOT add --gres=gpu:N or the job
# will fail with "Requested node configuration is not available".
# Set CUDA_VISIBLE_DEVICES below to select GPUs instead.
#SBATCH --cpus-per-task=32          # 8 GPUs x 4 workers = 32 dataloader procs
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.out
#SBATCH --error=/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Do not use "set -u": conda activate scripts reference unbound vars under -u.
set -eo pipefail

PROJECT_ROOT="/data/data3/conglab/s441865/code/Catalytic_sites_setA_3dcnn"
mkdir -p "${PROJECT_ROOT}/logs"

echo "================================================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

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

# ---------------------------------------------------------------------------
# Move to project directory
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Launch DDP training with torchrun
# ---------------------------------------------------------------------------
# --nproc_per_node must equal the number of GPUs in CUDA_VISIBLE_DEVICES.
# --standalone is for single-node multi-GPU (no rendezvous server needed).
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
echo "Launching torchrun with ${N_GPUS} processes..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${N_GPUS}" \
    train.py \
    --config configs/default.yaml

echo "================================================================"
echo "Job finished: $(date)"
echo "================================================================"
