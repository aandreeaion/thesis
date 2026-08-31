#!/bin/bash
#SBATCH --job-name=modernbert_webis
#SBATCH --output=jobs/modernbert_webis_%j.out
#SBATCH --error=jobs/modernbert_webis_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "Job started on:"
hostname
date

echo "Working directory:"
pwd

echo "Activating environment..."
source .venv/bin/activate

echo "Python:"
which python
python --version

echo "Checking packages and GPU..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo "Starting fine-tuning..."
python scripts/finetune_modernBERT.py

echo "Job finished on:"
date
