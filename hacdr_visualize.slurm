#!/bin/bash
#SBATCH --job-name=hacdr_visualize   # Specify the job name
#SBATCH --output=/scratch/st-ipor-1/bc2020ag/HACDR-Net/train_logs/%x-%A.out     # Specify the output file
#SBATCH --time=2:00:00          # Request 48 hours of runtime
#SBATCH --account=st-ipor-1-gpu     # Specify your allocation code
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4               # 4 CPU cores for dataloading etc.
#SBATCH --mem=16G                       # 64 GB RAM (tune as needed)
#SBATCH --error=/scratch/st-ipor-1/bc2020ag/HACDR-Net/train_logs/%x-%A.err   # Specify the error file
#SBATCH --mail-user=bc2020ag@student.ubc.ca # Email address for job notifications
#SBATCH --mail-type=FAIL,END          # Receive email notifications for all job events

# Move to the directory you ran `sbatch` from
cd /scratch/st-ipor-1/bc2020ag/HACDR-Net/

# Make sure logs directory exists
mkdir -p train_logs

# Activate conda environment
#module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacdr-env

export PYTHONPATH=$PWD:$PYTHONPATH

# Add your commands here
# Run your model on GPU via srun
python generate_hacdr_predictions.py \
  --config HACDRNet_DDHANet/HACDR_1440x960_idrid_maples_eyepacs.py \
  --checkpoint work_dirs/hacdr_combined_3_datasets/iter_106000.pth \
  --img_dir data/eyepacs_1440x960/splits/test/images \
  --out_dir visualized_masks/val-combined-test-eyepacs \
  --device cuda:0 \
  --save_color 

# Deactivate environment
conda deactivate