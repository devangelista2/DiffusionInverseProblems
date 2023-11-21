import argparse

import torch
import yaml


def load_config(config_path):
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # load config file
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device used: {device}.")
    config.device = device

    return config
