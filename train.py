import os
import random
import shutil
from json import JSONDecodeError

import numpy as np
import tensorflow as tf
from keras import backend as K

from data_loader.cyclegan_dataloader import CycleGanDataLoader
from model_trainer_builder import build_model_and_trainer
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs


def get_data_loader(config):
    data_loader_type = config.dataset.data_loader.type
    if data_loader_type == 'cyclegan':
        return CycleGanDataLoader(config)
    else:
        raise ValueError(f"unknown data loader type {data_loader_type}")


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = None
    config_filepath = None
    try:
        args = get_args()
        config_filepath = args.config
        config = process_config(args.config)
    except JSONDecodeError as e:
        print(f"invalid config file: {e}")
        exit(0)
    except Exception as e:
        print(f"missing or invalid arguments: {e}")
        exit(0)

    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.predicted_dir])
    # copy source files
    source_dir = os.path.join(config.callbacks.experiment_dir, 'source')
    if not os.path.exists(source_dir):
        shutil.copytree(os.path.abspath(os.path.curdir), source_dir,
                        ignore=lambda src, names: {'datasets', '__pycache__', '.git', 'results', 'venv'})
    # copy the config file
    shutil.copyfile(config_filepath, os.path.join(config.callbacks.experiment_dir, os.path.split(args.config)[-1]))

    # create tensorflow session and set as keras backed
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.trainer.gpu_memory_fraction
    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)

    print('Create the data generator.')
    data_loader = get_data_loader(config)

    # build model and trainer
    combined_model, trainer = build_model_and_trainer(config, data_loader)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    main()
