#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=02:00:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Attempt #${SLURM_RESTART_COUNT:-0}"

# These environment variables are used by torch.distributed and should ideally be set
# before running the python script, or at the very beginning of the python script.
# Master address is the hostname of the first node in the job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS


srun bash -c \
    "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    uv run \
    python train.py --hps ffhq256 \
        --restore_path $SCRATCH/data/vdvae_weights/ffhq256-iter-1700000-model.th \
        --restore_ema_path $SCRATCH/data/vdvae_weights/ffhq256-iter-1700000-model-ema.th \
        --restore_log_path $SCRATCH/data/vdvae_weights/ffhq256-iter-1700000-log.jsonl \
        --restore_optimizer_path $SCRATCH/data/vdvae_weights/ffhq256-iter-1700000-opt.th \
        --test_eval --data_root=$SCRATCH/data"
