import json
import os

from dotmap import DotMap


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    exp_name = config.exp.name
    exp_parent_dir = config.exp.parent_dir
    experiments_dir = config.exp.experiments_dir
    config.callbacks.experiment_dir = os.path.join(exp_parent_dir, experiments_dir, exp_name)
    config.callbacks.tensorboard_log_dir = os.path.join(exp_parent_dir, experiments_dir, exp_name, "tensorboard/")
    config.callbacks.checkpoint_dir = os.path.join(exp_parent_dir, experiments_dir, exp_name, "checkpoint/")
    config.callbacks.predicted_dir = os.path.join(exp_parent_dir, experiments_dir, exp_name, "predicted/")
    return config
