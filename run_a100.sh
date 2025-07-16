#!/bin/bash

# Description: Launch DDP training and log all output to a timestamped logfile

NUM_GPUS=2
CONFIG=phoenix2014
WORK_DIR=./work_dir/${CONFIG}/

# Set log filename with timestamp
LOGDIR=./logs
mkdir -p $LOGDIR
LOGFILE="${LOGDIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "Logging output to: $LOGFILE"

# Run training and log output
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} main.py \
    --dataset ${CONFIG} \
    --loss-weights Slow=0.25 Fast=0.25 \
    --work-dir ${WORK_DIR} \
    2>&1 | tee $LOGFILE
