#!/bin/bash
DATASET_DIR=/home/a/localData/tempt
CHECKPOINT_FILE=/home/a/localData/densenet40cp/model.ckpt-16643
CUDA_VISIBLE_DEVICES=1 python eval_densenet.py \	
	--dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_name=cifar10 \
    --dataset_split_name=test \
    --model_name=densenet_40