"""
Global configuration for Cascade Mask RCNN instance-segmentation project.
"""

import pathlib

# DATA_ROOT = pathlib.Path("/workspace/src/cv/3/dataset")
DATA_ROOT = pathlib.Path("/content/dataset")

NUM_CLASSES = 4
CLASS_NAME_MAP = {i: f"class{i}" for i in range(1, NUM_CLASSES + 1)}

VAL_RATIO = 0.2
RANDOM_SEED = 42

# CASCADE_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
CASCADE_YAML = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"

OUTPUT_DIR = pathlib.Path("./output")
