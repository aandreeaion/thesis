#!/bin/bash
#SBATCH --job-name=predict_allsides_same_events
#SBATCH --output=outputs/predict_allsides_same_events_logs/predict_allsides_same_events_%j.out
#SBATCH --error=outputs/predict_allsides_same_events_logs/predict_allsides_same_events_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

cd /home6/s4429621/thesis-modernBERT

mkdir -p outputs/predict_allsides_same_events_logs

source .venv/bin/activate

python scripts/predict_allsides_same_event.py