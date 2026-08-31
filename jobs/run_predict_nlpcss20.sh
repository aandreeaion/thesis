#!/bin/bash
#SBATCH --job-name=predict_nlpcss20
#SBATCH --output=outputs/predict_nlpcss20_logs/predict_nlpcss20_%j.out
#SBATCH --error=outputs/predict_nlpcss20_logs/predict_nlpcss20_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

cd /home6/s4429621/thesis-modernBERT

mkdir -p outputs/predict_nlpcss20_logs

source .venv/bin/activate

python scripts/predict_nlpcss-20.py
