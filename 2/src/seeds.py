"""
Reproducibility helper.

set_seed(seed, deterministic=False)
-----------------------------------
* deterministic=False → only seeds RNGs (fast)
* deterministic=True  → additionally forces cuDNN/CuBLAS deterministic
                        algorithms (slow ‑ but bit‑exact)
"""

from __future__ import annotations
import os, random
import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / cuBLAS reproducibility switches
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
