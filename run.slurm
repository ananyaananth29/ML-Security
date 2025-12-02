#!/bin/bash
#SBATCH --job-name=robust-pgd-train
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Single process for evaluation
#SBATCH --gres=gpu:1                # Only one GPU needed for eval
#SBATCH --cpus-per-task=6           # Dataloader workers etc.
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

# Launch evaluation (no distributed launch needed)
srun python evaluate_transfer_robustness.py
