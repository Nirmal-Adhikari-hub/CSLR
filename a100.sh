#!/bin/bash

# Description: Launch DDP training and log all output to a timestamped logfile

NUM_AVAILABLE=$(nvidia-smi -L | wc -l)
NUM_GPUS=${NUM_GPUS:-$NUM_AVAILABLE}
CONFIG=phoenix2014
WORK_DIR=./work_dir/a100/${CONFIG}/

# find the checkpoint to resume
if [ -f "${WORK_DIR}/_best_model.pt" ]; then
  CKPT="${WORK_DIR}/_best_model.pt"
else
  CKPT=$(ls -t ${WORK_DIR}/*_epoch*_model.pt 2>/dev/null | head -1)
fi
echo "⏩ Resuming from checkpoint: $CKPT"


# Set log filename with timestamp
LOGDIR=./logs
mkdir -p $LOGDIR
LOGFILE="${LOGDIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "Logging output to: $LOGFILE"

# ─── Set a free rendezvous port ────────────────────────────────────────────
export MASTER_PORT=${MASTER_PORT:-29510}   # change 29510 if that’s taken
# If needed, also set MASTER_ADDR to your hostname or IP:
# export MASTER_ADDR=${MASTER_ADDR:-localhost}

# Run training and log output
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} main.py \
    --dataset ${CONFIG} \
    --loss-weights Slow=0.25 Fast=0.25 \
    --work-dir ${WORK_DIR} \
    --batch-size 2 \
    --load-checkpoints $CKPT \
    2>&1 | tee $LOGFILE
