export OMP_NUM_THREADS=24
torchrun \
    --nproc_per_node 4 \
    --standalone main.py \
    --dataset-root ~/datasets/EmmaGNN \
    --dataset-name ogbn-products \
    --epochs 1000 \
    --dropout 0.3 \
    --num-layers 3 \
    --hidden-dim 256 \
    --model sage \
    --lr 0.001 \
    --emma