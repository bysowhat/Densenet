# DenseNet tensorflow-slim
This repository contains the slim(tensorflow) implementation for the paper [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993).There are two types of `Densely Connected Convolutional Networks`  are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers



Citation:

     @inproceedings{huang2017densely,
          title={Densely connected convolutional networks},
          author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2017}
      }


## Datasets

The current version supports CIFAR-10 datasets. In order to be used for training a DENSENET model, the former need to be converted to TF-Records using the `download_and_convert_data.py` script(https://github.com/tensorflow/models/tree/master/slim).
The evaluation on Imagenet will be added soon.

## Evaluation on CIFAR-10

The present TensorFlow implementation of Densenet models have the following performances:

| Model | Dataset  | error |
|--------|:---------:|:------:|
| [DenseNet(k = 12)]| CIFAR-10 |  6.92 
| [DenseNet-BC(k = 12)] | CIFAR-10 | 5.63

The evaluation metrics should be reproducible by running the following command:
```bash
EVAL_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/DENSENET_iter_.ckpt
python eval_ssd_network.py \
	--dataset_dir=${DATASET_DIR} \
	--checkpoint_path=${CHECKPOINT_FILE} \
	--dataset_name=cifar10 \
	--dataset_split_name=test \
	--model_name=densenet_40
```

## Training

The script `train_densenet.py` is in charged of training the network. Similarly to TF-Slim models, one can pass numerous options to the training process (dataset, optimiser, hyper-parameters, model, ...). In particular, it is possible to provide a checkpoint file which can be use as starting point in order to fine-tune a network.

For instance, one can train a DenseNet (L=40, k=12) on CIFAR-10  as following:
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/
python train_ssd_network.py \
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
    --batch_size=64 \
    --num_clones=4 \
    --num_classes=10 \
    --weight_decay=0.0001 \
    --log_every_n_steps=100 \
    --learning_rate_decay_type=densenetForCifar
```
Note that in addition to the training script flags, one may also want to experiment with data augmentation parameters (random cropping, resolution, ...) in `preprocessing/cifarnet_preprocessing.py`.

The `num_clones` parameter can be set to change the number of model clones to deploy.
In our experiment environment (cudnn v5.1, CUDA 8.0, four K40m GPU), the code runs with speed 1.5iters/s when batch size is set to be 64. The hyperparameters are identical to the original [torch implementation] (https://github.com/liuzhuang13/DenseNet).


## Dependencies:

+ Python 3
+ TensorFlow >= 1.0


## Questions?

Please let me know if you have any questions!
