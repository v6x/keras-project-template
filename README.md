# Keras Project Template
Inspired from https://github.com/Ahmkel/Keras-Project-Template

[CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) is implemented as a sample model. 

## Features
- [tensorpack](https://github.com/tensorpack/tensorpack) powered dataloader
- supports save/load models and optimizers
- supports multi-gpu training
- backups source code and config used to train
- stops training if encounter nan loss
- collages losses to a single graph on tensorboard
- telegram notification on train start/end

## Requirements
python >= 3.6

## Install required packages
```shell
pip install -r requirements.txt
```

## Download dataset
```shell
sh ./download_dataset.sh horse2zebra
```

## Setup config
edit configs/cyclegan.json file

## To train
 ```shell
python train.py -c configs/cyclegan.json
```

## Visualize via Tensorboard
```shell
tensorboard --logdir /path/to/experiment/dir/tensorboard/
```

## Setup telegram notification
Add telegram token and chat_id to environment variables as TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
