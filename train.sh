#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_densenet.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
	--dataset_dir=${DATASET_DIR} \
	--model_name=densenet_40
