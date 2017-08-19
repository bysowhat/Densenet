#!/bin/bash
DATASET_DIR=/home/a/localData/tempt
TRAIN_DIR=/home/a/localData/cifarck/
CUDA_VISIBLE_DEVICES=0 python train_densenet.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --preprocessing_name=cifarnet \
	--dataset_dir=${DATASET_DIR} \
	--model_name=densenet_40 \
	--save_summaries_secs=600 \
    --save_interval_secs=100 \
    --optimizer=adam \
    --learning_rate=0.1 \
    --batch_size=1 \
    --num_clones=1 \
    --num_readers=1 \
    --num_classes=10 \
