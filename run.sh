#!/bin/bash
#==============================================================================
# RT-opto CNN-GRU Video Classifier — SLURM submission script (multi-GPU)
#
# Requests 4 GPUs and auto-detects whether they land on one node or many,
# then launches torchrun with the correct topology.
#
# Usage:
#   sbatch run.sh                              # default: full pipeline
#   sbatch --export=MODE=eval_only run.sh      # evaluation only
#==============================================================================

#SBATCH --job-name=RT-opto-Classifier
#SBATCH --partition=witten
#SBATCH --nodes=1-4
#SBATCH --gpus=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -c 128
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/out-%j.txt
#SBATCH --error=logs/error-%j.txt

set -euo pipefail
mkdir -p logs

module load anacondapy/2023.07-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate general

MODE=${MODE:-full}

# ---------------------------------------------------------------------------
# Discover topology
# ---------------------------------------------------------------------------
NNODES=${SLURM_NNODES:-1}
# Total GPUs granted to the job
TOTAL_GPUS=${SLURM_NTASKS:-4}
GPUS_PER_NODE=$(( TOTAL_GPUS / NNODES ))

# Pick a free port on the master node for the rendezvous endpoint
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

echo "============================================"
echo "Job $SLURM_JOB_ID  —  mode=$MODE"
echo "Nodes: $NNODES   GPUs total: $TOTAL_GPUS   GPUs/node: $GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Host list: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
date
echo "============================================"

# Print GPU info from each node
srun --ntasks-per-node=1 bash -c \
  "echo \"[\$(hostname)] GPUs: \$CUDA_VISIBLE_DEVICES\"; \
   python3 -c \"import torch; \
     [print(f'  [\$(hostname)] GPU {i}: {torch.cuda.get_device_name(i)}') \
      for i in range(torch.cuda.device_count())]\""

echo ""
echo ">>> Launching distributed training (torchrun) ..."

# srun launches one torchrun per node; torchrun spawns GPUS_PER_NODE workers.
srun --ntasks="$NNODES" --ntasks-per-node=1 \
  torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    run.py \
      --labels states_per_session.pkl \
      --video_root ../Data/Defeat-Cohorts \
      --output_dir output-3/ \
      --batch_size ${BATCH_SIZE:-64}

echo ""
echo "Done."
date
