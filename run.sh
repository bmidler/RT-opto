#!/bin/bash
#==============================================================================
# RT-opto CNN-GRU Video Classifier — SLURM submission script
#
# Usage:
#   sbatch run.sh                              # default: full pipeline
#   sbatch --export=MODE=eval_only run.sh      # evaluation only
#==============================================================================

#SBATCH --job-name=RT-opto-Classifier
#SBATCH --partition=witten
#SBATCH --gpus=1
#SBATCH -c 128
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/out-%j.txt
#SBATCH --error=logs/error-%j.txt

mkdir -p logs

module load anacondapy/2023.07-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate general

MODE=${MODE:-full}

echo "============================================"
echo "Job $SLURM_JOB_ID  —  mode=$MODE"
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date
echo "============================================"

# Print GPU info
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') \
            for i in range(torch.cuda.device_count())]"

echo ""
echo ">>> Running Full Pipeline (Train -> Evaluate -> Benchmark) ..."
python3 run.py \
    --labels states_per_session.pkl \
    --video_root ../Data/Defeat-Cohorts \
    --output_dir output-4/ \
    --batch_size ${BATCH_SIZE:-64}

echo ""
echo "Done."
date
