import os
import torch
import importlib
from typing import List

def get_model_name_from_path(model_path):
    """
    Extract model name from model path.
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].endswith('.pth') or model_paths[-1].endswith('.pt'):
        model_name = model_paths[-1].split('.')[0]
    else:
        model_name = model_paths[-1]
    return model_name 