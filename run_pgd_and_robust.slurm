#!/bin/bash
#SBATCH --job-name=robust-pgd-train
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --nodes=1                   # All 8 GPUs on one node
#SBATCH --ntasks=8                  # 1 task per GPU
#SBATCH --gres=gpu:8                # Request 8 GPUs
#SBATCH --cpus-per-task=6           # CPU threads per GPU
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

# LAUNCH TRAINING ON 8 GPUs
srun python -m torch.distributed.run --nproc_per_node=8 pgd_adversarial_training_with_robust_dataset.py
