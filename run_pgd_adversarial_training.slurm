#!/bin/bash
#SBATCH --job-name=pgd-adv-train
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --nodes=1                   # all GPUs on one node
#SBATCH --ntasks=1                  # single launcher task
#SBATCH --gres=gpu:8                # request 8 GPUs
#SBATCH --cpus-per-task=48          # CPU threads for the launcher
#SBATCH --time=8:00:00
#SBATCH --mem=120G
#SBATCH --partition=dubis-gpu-rw
#SBATCH --qos=redwood-freecycle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uu1520797@utah.edu
#SBATCH --account=dubis

# Load environment
module load cuda
source /uufs/chpc.utah.edu/common/HIPAA/proj_ODAR/Ananya/nnUnet/nnunet_env_312/bin/activate

cd /uufs/chpc.utah.edu/common/HIPAA/proj_ODAR/Ananya/Pytorch-Adversarial-Training-CIFAR-master
mkdir -p ./slurm_logs

# Choose a per-job port to avoid collisions
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

# Launch DDP training across 8 GPUs
srun --ntasks=1 python -m torch.distributed.run --nproc_per_node=8 --master_port=${MASTER_PORT} pgd_adversarial_training.py
