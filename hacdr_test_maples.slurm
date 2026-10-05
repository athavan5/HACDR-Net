#!/bin/bash
#SBATCH --job-name=hacdr_test_maples   # Specify the job name
#SBATCH --output=/scratch/st-ipor-1/bc2020ag/HACDR-Net/test_logs/%x-%A.out     # Specify the output file
#SBATCH --time=02:00:00          # Request 2 hours of runtime
#SBATCH --account=st-ipor-1-gpu     # Specify your allocation code
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4               # 4 CPU cores for dataloading etc.
#SBATCH --mem=64G                       # 64 GB RAM (tune as needed)
#SBATCH --error=/scratch/st-ipor-1/bc2020ag/HACDR-Net/test_logs/%x-%A.err   # Specify the error file
#SBATCH --mail-user=bc2020ag@student.ubc.ca # Email address for job notifications
#SBATCH --mail-type=FAIL,END          # Receive email notifications for all job events

# Move to the directory you ran `sbatch` from
cd /scratch/st-ipor-1/bc2020ag/HACDR-Net/

# Make sure logs directory exists
mkdir -p test_logs

# Activate conda environment
#module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacdr-env

export PYTHONPATH=$PWD:$PYTHONPATH

# Add your commands here
# Run your model on GPU via srun
python tools/test.py HACDRNet_DDHANet/HACDR_idrid.py work_dirs/hacdr_idrid_ORIGINAL/iter_122000.pth --eval mDice

# Deactivate environment
conda deactivate