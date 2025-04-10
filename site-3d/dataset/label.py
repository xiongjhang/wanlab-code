import pathlib
import os
from typing import Literal

import numpy as np
import tifffile as tiff
import SimpleITK as sitk


from siteflow.siteflow.utils.util import mask_to_coordinate, coordinate_to_mask, PathType

def read_nrrd(path: PathType):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def write_nrrd(img: np.ndarray, path: PathType):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(path)


def get_site_num(mask_stack: np.ndarray) -> Literal['0', '1', 'multi']:
    """Return site num in total cell seqence

    Args:
        mask_stack (np.ndarray): binary value mask stack

    Return:
        0 - no site
        1 - 1 site
        multi - 2 or more sites    
    """
    pass

# def ()


if __name__ == "__main__":

    src_dir = ''
    tgt_dir = ''
