import numpy as np
import torch
from torch.utils import data as Data
from typing import Sequence, List, Union

def image2kspace(x):
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.fft2(x)
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x

def kspace2image(x):
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.ifft2(x)
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x

def pseudo2real(x):
    """
    :param x: [..., C=2, H, W]
    :return: [..., H, W]
    """
    return (x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2) ** 0.5


def complex2pseudo(x):
    """
    :param x: [..., H, W] Complex
    :return: [...., C=2, H, W]
    """
    if isinstance(x, np.ndarray):
        return np.stack([x.real, x.imag], axis=-3)
    elif isinstance(x, torch.Tensor):
        return torch.stack([x.real, x.imag], dim=-3)
    else:
        raise RuntimeError("Unsupported type.")

def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    return x[..., 0, :, :] + x[..., 1, :, :] * 1j

# def arbitrary_dataset_split(dataset: Data.Dataset,
#                             indices_list: Sequence[Sequence[int]]
#                             ) -> List[torch.utils.data.Subset]:
#     return [Data.Subset(dataset, indices) for indices in indices_list]