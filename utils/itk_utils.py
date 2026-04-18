"""
itk_utils.py
============
Shared SimpleITK I/O utilities for dataset conversion scripts.
"""

import SimpleITK as sitk
import numpy as np

def load_medical_image(path: str, dtype=np.float32) -> tuple:
    """
    Read a medical image (e.g. NIfTI, NRRD).
    Returns (sitk_image, numpy_array).
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(dtype)
    return img, arr

def get_voxel_sizes(sitk_image: sitk.Image) -> tuple:
    """
    Return voxel sizes as (z, y, x) in mm.
    SimpleITK native order is (x, y, z), so we flip it.
    """
    return tuple(sitk_image.GetSpacing()[::-1])
