#!/bin/bash
#SBATCH --job-name=robust-pgd-train
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --nodes=1                   # All 8 GPUs on one node
#SBATCH --ntasks=1                  # Single launcher task
#SBATCH --gres=gpu:8                # Request 8 GPUs
#SBATCH --cpus-per-task=48          # CPU threads for the single launcher
#SBATCH --time=8:00:00
#SBATCH --mem=120G                  # (8 GPUs need more CPU RAM)
#SBATCH --partition=dubis-gpu-rw
#SBATCH --qos=redwood-freecycle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uu1520797@utah.edu
#SBATCH --account=dubis

# Load your environment
module load cuda
# Activate your Python environment
source /uufs/chpc.utah.edu/common/HIPAA/proj_ODAR/Ananya/nnUnet/nnunet_env_312/bin/activate

# Move to the project directory
cd /uufs/chpc.utah.edu/common/HIPAA/proj_ODAR/Ananya/Pytorch-Adversarial-Training-CIFAR-master

# Optional: ensure logs folder exists
mkdir -p ./slurm_logs

# LAUNCH TRAINING ON 8 GPUs (override default port to avoid collisions)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
srun --ntasks=1 python -m torch.distributed.run --nproc_per_node=8 --master_port=${MASTER_PORT} basic_training_with_robust_dataset.py
