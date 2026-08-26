import os
import random

import numpy as np
import torch


def enable_determinism():
    """Make fits bitwise reproducible under a fixed seed.

    Without all of this, two runs of identical code differed by up to 0.27 test
    MAE for the modded model -- larger than any effect being measured (see
    FINDINGS.md section 1.3). Call before the first CUDA op: the cuBLAS
    workspace variable is read when the CUDA context is created.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
