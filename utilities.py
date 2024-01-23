import torch
import warnings
from pandarallel import pandarallel
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor
from natsort import natsorted
import seaborn as sns
from PIL import Image

from tqdm.notebook import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style("whitegrid")

pandarallel.initialize()

warnings.filterwarnings('ignore')


class ClearCache:
    """
    A context manager for clearing the CUDA cache before and after a block of code execution.

    This context manager helps to manage the GPU memory usage by clearing the CUDA cache before the
    code block is entered and after it exits. It can be useful to free up memory on the GPU during
    long-running processes or when memory management is crucial.
    """

    def __enter__(self):
        """
        Clears the CUDA cache before entering the code block.
        """
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clears the CUDA cache after exiting the code block.

        Args:
            exc_type (type): The type of the exception raised, if any.
            exc_val (Exception): The exception instance raised, if any.
            exc_tb (traceback): The traceback information related to the exception, if any.
        """
        torch.cuda.empty_cache()


def images_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move images to device.

    Args:
        tensor (torch.Tensor): tensor with images.
        device (torch.device): device to move the tensor.

    Returns:
        tensor (torch.tensor): tensor with images in the device.

    References:
        https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
        https://medium.com/@snk.nitin/how-to-solve-cuda-out-of-memory-error-850bb247cfb2
    """
    if device == torch.device('cuda:0'):
        try:
            return tensor.to(device)
        except RuntimeError as re:
            print(re)
            torch.cuda.empty_cache()
            return tensor.to(device)
    else:
        return tensor.to(device)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load checkpoint from directory to device

    References:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/utils.py
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, scheduler, epoch, loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, **kwargs):
    """Save checkpoint in memory

    References:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    fixed_params = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    dict_to_save = {**kwargs, **fixed_params}
    # print(f'Saving keys: {dict_to_save.keys()}')
    torch.save(dict_to_save, path)


def nn_params(model):
    """Get the number of parameters of a model

    References:
        https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(df: pd.DataFrame) -> dict:
    df_numeric = df.select_dtypes(include=np.number)
    df_categorical = df.select_dtypes(include=['object', 'category'])
    df_other = df.select_dtypes(exclude=[np.number, 'object', 'category'])

    return {'numeric': df_numeric,
            'categorical': df_categorical,
            'other': df_other}
