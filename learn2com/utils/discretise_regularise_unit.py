import os
import sys
import math
import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from functools import partial

import torch

from learn2com.utils.load_ymal import load_yaml


def get_discretise_regularise_unit(config):
    if config is None:
        raise ValueError(
            "Configuration file could not be loaded. Please check the path and format."
        )

    if config["type"] == "Normal":
        scale = config["sigma"]

    partial_func = partial(discretise_regularise_unit, scale=scale)
    return partial_func


def discretise_regularise_unit(x, scale, training):
    if torch.isnan(x).any():
        return x
    if training:
        sample = x + torch.randn_like(x) * scale
        res = torch.sigmoid(sample)
    else:
        res = (x > 0).float()
    return res


if __name__ == "__main__":
    dru = get_discretise_regularise_unit()
    x = torch.tensor([0.5, 1.0, 1.5])
    result = dru(x)
    print("Discretised and regularised output:", result)
