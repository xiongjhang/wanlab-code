import glob
import os
import collections
from typing import Any
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

def create_padded_indexes(indexes, halo_shape):
    return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape))

def traverse_h5_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results

def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def calculate_stats(img: np.array, skip: bool = False) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    if not skip:
        pmin, pmax, mean, std = np.percentile(img, 1), np.percentile(img, 99.6), np.mean(img), np.std(img)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std
    }

def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode='reflect')

def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]

def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]