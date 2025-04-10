import os
import sys
import pickle
from pathlib import Path
from typing import Sequence, Iterable, List, Optional, Type, Union, Tuple
from functools import partial
import logging
import importlib
import shutil

import h5py
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import skimage
from skimage.color import label2rgb
import scipy
import torch
import torch.optim as optim


__class = ['global variable', 'util func', 'format conversion', 'nomalization', 
        'logger', 'tensorboard',
        'module', 'trainer', 'evaluation']

# ===============
# Global Variable
# ===============

_EPSILON = 1e-10
PathType = Union[str, Path, Iterable[str], Iterable[Path]]

# =========
# Util Func
# =========

def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

# ============
# Nomalization
# ============

def image_process(image: np.ndarray, mode: str) -> np.ndarray:
    
    if mode == 'simple_norm':
        image = image / 255

    elif mode == 'maxmin_norm':
        image -= np.min(image)
        image /= np.max(image)
    
    elif mode == 'stand':
        image = (image-image.mean()) / image.std()
        
    return image.astype(np.float32)

# =================
# Format Conversion
# =================

def nrrd_to_tif(nrrd_path: PathType, tif_path: PathType):
    """Turn `.nrrd` to `.tif`

    Args:
        nrrd_path (PathType): _description_
        tif_path (PathType): _description_
    """

    nrrd_image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(nrrd_image, tif_path)

    def tif_to_nrrd(tif_path: PathType, nrrd_path: PathType):
        """Turn `.tif` to `.nrrd`

    Args:
        tif_path (PathType): _description_
        nrrd_path (PathType): _description_
    """

    tif_image = sitk.ReadImage(tif_path)
    sitk.WriteImage(tif_image, nrrd_path)

def read_nrrd(nrrd_path: PathType):
    image = sitk.ReadImage(nrrd_path)
    image = sitk.GetArrayFromImage(image)
    return image

# ===============
# Data Conversion
# ===============

def mask_to_coordinate(
    matrix: np.ndarray, probability: float = 0.5
) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    NOTE - plt.scatter uses the x, y system. Therefore any plots
    must be inverted by assigning x=c, y=r!

    Args:
        matrix: Matrix representation of spot coordinates.
        image_size: Default image size the grid was layed on.
        probability: Cutoff value to round model prediction probability.

    Returns:
        Array of r, c coordinates with the shape (n, 2).
    """
    if not matrix.ndim == 2:
        raise ValueError("Matrix must have a shape of (r, c).")
    if not matrix.shape[0] == matrix.shape[1] and not matrix.shape[0] >= 1:
        raise ValueError("Matrix must have equal length >= 1 of r, c.")
    assert np.max(matrix) <= 1 and np.max(matrix) >= 0, 'Matrix must be prediction probability'

    # Turn prob matrix into binary matrix (0-1)
    binary_matrix = (matrix > probability).astype(int)

    # Label connected regions
    labeled_array = skimage.measure.label(binary_matrix)

    # Compute the centorid coordinates of each conneted regions
    properties = skimage.measure.regionprops(labeled_array)
    centers = [prop.centroid for prop in properties]
    coords = np.array(centers)

    return coords

def coordinate_to_mask(
    coords: np.ndarray, image_size: int = 128, n: int = 1, sigma: float = None, size_c: int = None
) -> np.ndarray:
    """Return np.ndarray of shape (n, n): r, c format.

    Args:
        coords: List of coordinates in r, c format with shape (n, 2).
        image_size: Size of the image from which List of coordinates are extracted.
        n: Size of the neighborhood to set to 1 or apply Gaussian filter.
        sigma: Standard deviation for Gaussian kernel. If None, no Gaussian filter is applied.
        size_c: If empty, assumes a squared image. Else the length of the r axis.

    Returns:
        The prediction matrix as numpy array of shape (n, n): r, c format.
    """
    nrow = ncol = image_size
    if size_c is not None:
        ncol = size_c

    prediction_matrix = np.zeros((nrow, ncol))

    for r, c in coords:
        # Consider bonuder
        r_min = max(0, r - n)
        r_max = min(nrow, r + n + 1)
        c_min = max(0, c - n)
        c_max = min(ncol, c + n + 1)

        # Assign values along pre人iction matrix 
        if sigma is None:
            prediction_matrix[r_min:r_max, c_min:c_max] = 255
        else:
            y, x = np.ogrid[-n:n+1, -n:n+1]
            gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_kernel /= gaussian_kernel.sum()  
            prediction_matrix[r_min:r_max, c_min:c_max] += gaussian_kernel[:r_max-r_min, :c_max-c_min]

    return prediction_matrix

# =====
# Loger
# =====

loggers = {}

def get_logger(name, level=logging.INFO):

    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger

# ===========
# Tensorboard
# ===========

class TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, skip_last_target=False, log_channelwise=False):
        self.skip_last_target = skip_last_target
        self.log_channelwise = log_channelwise

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor

        Returns:
            list[(str, np.ndarray)]: list of tuples of the form (tag, img)
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self._process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def _process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                if self.log_channelwise and name == 'predictions':
                    tag_template = '{}/batch_{}/channel_{}/slice_{}'
                    for channel_idx in range(batch.shape[1]):
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                        img = batch[batch_idx, channel_idx, slice_idx, ...]
                        tagged_images.append((tag, self._normalize_img(img)))
                else:
                    tag = tag_template.format(name, batch_idx, slice_idx)
                    if name in ['predictions', 'targets']:
                        # for single channel predictions, just log the image
                        if batch.shape[1] == 1:
                            img = batch[batch_idx, :, slice_idx, ...]
                            tagged_images.append((tag, self._normalize_img(img)))
                        else:
                            # predictions are probabilities so convert to label image
                            img = batch[batch_idx].argmax(axis=0)
                            # take the middle slice
                            img = img[slice_idx, ...]
                            # convert to label image
                            img = label2rgb(img)
                            img = img.transpose(2, 0, 1)
                            tagged_images.append((tag, img))
                    else:
                        # handle input images
                        if batch.shape[1] in [1, 3]:
                            # if single channel or RGB image, log directly
                            img = batch[batch_idx, :, slice_idx, ...]
                            tagged_images.append((tag, self._normalize_img(img)))
                        else:
                            # log channelwise
                            tag_template = '{}/batch_{}/channel_{}/slice_{}'
                            for channel_idx in range(batch.shape[1]):
                                tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                                img = batch[batch_idx, channel_idx, slice_idx, ...]
                                tagged_images.append((tag, self._normalize_img(img)))

        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                # this is target segmentation so convert to label image
                lbl = label2rgb(img)
                lbl = lbl.transpose(2, 0, 1)
                tagged_images.append((tag, lbl))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return TensorboardFormatter()
    return TensorboardFormatter(**formatter_config)

# ======
# Module
# ======

def get_number_of_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')

# =======
# Trainer
# =======

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)

def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    optim_name = optimizer_config.get('name', 'Adam')
    # common optimizer settings
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)

    # grab optimizer specific settings and init
    # optimizer
    if optim_name == 'Adadelta':
        rho = optimizer_config.get('rho', 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho,
                                   weight_decay=weight_decay)
    elif optim_name == 'Adagrad':
        lr_decay = optimizer_config.get('lr_decay', 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif optim_name == 'AdamW':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'SparseAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == 'Adamax':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)
    elif optim_name == 'ASGD':
        lambd = optimizer_config.get('lambd', 0.0001)
        alpha = optimizer_config.get('alpha', 0.75)
        t0 = optimizer_config.get('t0', 1e6)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, lambd=lambd,
                                 alpha=alpha, t0=t0, weight_decay=weight_decay)
    elif optim_name == 'LBFGS':
        max_iter = optimizer_config.get('max_iter', 20)
        max_eval = optimizer_config.get('max_eval', None)
        tolerance_grad = optimizer_config.get('tolerance_grad', 1e-7)
        tolerance_change = optimizer_config.get('tolerance_change', 1e-9)
        history_size = optimizer_config.get('history_size', 100)
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter,
                                max_eval=max_eval, tolerance_grad=tolerance_grad,
                                tolerance_change=tolerance_change, history_size=history_size)
    elif optim_name == 'NAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        momentum_decay = optimizer_config.get('momentum_decay', 4e-3)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, betas=betas,
                                momentum_decay=momentum_decay,
                                weight_decay=weight_decay)
    elif optim_name == 'RAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'RMSprop':
        alpha = optimizer_config.get('alpha', 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha,
                                  weight_decay=weight_decay)
    elif optim_name == 'Rprop':
        momentum = optimizer_config.get('momentum', 0)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'SGD':
        momentum = optimizer_config.get('momentum', 0)
        dampening = optimizer_config.get('dampening', 0)
        nesterov = optimizer_config.get('nesterov', False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              dampening=dampening, nesterov=nesterov,
                              weight_decay=weight_decay)
    else:  # Adam is default
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas,
                               weight_decay=weight_decay)

    return optimizer

def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


# =========
# Evaluation
# =========

def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

def offset_euclidean(offset: List[tuple]) -> np.ndarray:
    """Calculates the euclidean distance based on row_column_offsets per coordinate."""
    return np.sqrt(np.sum(np.square(np.array(offset)), axis=-1))

def _get_offsets(
    pred: np.ndarray, true: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> List[tuple]:
    """Return a list of (r, c) offsets for all assigned coordinates.

    Args:
        pred: List of all predicted coordinates.
        true: List of all ground truth coordinates.
        rows: Rows of the assigned coordinates (along "true"-axis).
        cols: Columns of the assigned coordinates (along "pred"-axis).
    """
    return [
        (true[r][0] - pred[c][0], true[r][1] - pred[c][1]) for r, c in zip(rows, cols)
    ]

def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float = None
) -> Tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.

    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.

    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.

    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Handle zero-sized matrices (occurs if true or pred has no items)
    if matrix.size == 0:
        return [], []

    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None and cutoff != 0:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    if cutoff is None:
        return list(row), list(col)

    # As scipy will still assign all columns to rows
    # We here remove assigned values falling below the cutoff
    nrow = []
    ncol = []
    for r, c in zip(row, col):
        if matrix[r, c] <= cutoff:
            nrow.append(r)
            ncol.append(c)
    return nrow, ncol