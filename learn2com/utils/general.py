import os
import sys
from pathlib import Path

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from nn_models.learn2com.utils.load_ymal import load_yaml


def get_learn2com_config_path():
    current_dir = Path(__file__).parent.parent
    config_path = current_dir / "configs" / "learn2com_config.yaml"
    return config_path


def get_learn2com_config():
    path = get_learn2com_config_path()
    return load_yaml(path)
