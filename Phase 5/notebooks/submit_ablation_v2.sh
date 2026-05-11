#!/bin/bash
#SBATCH --job-name=ablation_v2
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00          # 4 XGBoost fits, no search: under 5 min
#SBATCH --output=ablation_v2_%j.out
#SBATCH --error=ablation_v2_%j.err

echo "Job $SLURM_JOB_ID running on $SLURMD_NODENAME"
echo "Started at: $(date)"
nvidia-smi
echo "----------------------------------------"

cd $SLURM_SUBMIT_DIR
echo "Working dir: $(pwd)"
echo "----------------------------------------"

python -u ablation_study_v2_fixed_split.py

echo "----------------------------------------"
echo "Finished at: $(date)"
