#!/bin/bash

declare -A node2rank

master_ip="gpu06"
master_port=25000

node2rank["gpu06"]="0"
node2rank["gpu07"]="1"
node2rank["gpu04"]="2"
node2rank["gpu05"]="3"

# export CUDA_LAUNCH_BLOCKING=1
torchrun \
    --nproc_per_node 4 \
    --nnodes ${#node2rank[@]} \
    --node-rank ${node2rank[`hostname`]} \
    --master-addr ${master_ip} \
    --master-port ${master_port} \
        main.py \
        --dataset-root ~/datasets/EmmaGNN/ \
        --dataset-name ogbn-papers100M \
        --model sage \
        --hidden-dim 96 \
        --num-layers 3 \
        --dropout 0.5 \
        --splits 10 \
        --emma \
        --use-fp16