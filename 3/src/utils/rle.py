"""
Encode binary masks to COCO-compatible RLE.
"""

import numpy as np
from pycocotools import mask as mask_utils


def encode_binary_mask(mask: np.ndarray) -> dict:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle
