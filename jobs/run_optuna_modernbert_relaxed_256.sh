#!/bin/bash
#SBATCH --job-name=modernbert_optuna_relaxed_256
#SBATCH --output=jobs/modernbert_optuna_relaxed_256_%j.out
#SBATCH --error=jobs/modernbert_optuna_relaxed_256_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "Job started on:"
hostname
date

echo "Working directory:"
pwd

echo "Loading modules..."
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Setting Hugging Face cache directories to scratch..."
mkdir -p /scratch/$USER/.cache/huggingface/datasets
mkdir -p /scratch/$USER/.cache/huggingface/hub

export HF_HOME="/scratch/$USER/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/hub"

echo "Python:"
which python
python --version

echo "Checking packages and GPU..."
python -c "import torch; import optuna; print('Torch:', torch.__version__); print('Optuna:', optuna.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo "Starting Optuna relaxed tuning with max_length=256..."
python scripts/optuna_modernBERT_relaxed_256.py

echo "Job finished on:"
date

deactivate
