#!/bin/bash
#SBATCH --job-name=phase5_clean
#SBATCH --partition=normal
#SBATCH --gres=gpu:1                    # 1 full A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00                 # generous headroom; expect ~2.5h
#SBATCH --output=phase5_%j.out
#SBATCH --error=phase5_%j.err

# ---------- Environment info -------------------------------------------------
echo "Job $SLURM_JOB_ID running on $SLURMD_NODENAME"
echo "Started at: $(date)"
nvidia-smi
echo "----------------------------------------"

# Packages installed via pip in JupyterHub live in ~/.local/ and are
# automatically picked up by the system Python on HAICORE.

cd $SLURM_SUBMIT_DIR
echo "Working dir: $(pwd)"
ls -la
echo "----------------------------------------"

python -u phase_5_optimization_clean.py

echo "----------------------------------------"
echo "Finished at: $(date)"
