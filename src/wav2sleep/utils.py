import logging
import random
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_rank_zero():
    """Check if the current process is rank zero in a Pytorch distributed environment."""
    if not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def rank_zero_only(method):
    """Execute function only on the rank zero process.

    Logs the stack trace.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if is_rank_zero():
            return method(self, *args, **kwargs)
        else:
            logger.debug(f'Skipping {method} on non-rank-zero process.')
            return

    return wrapper


def fix_seeds(seed: int = 42) -> None:
    """Fix all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f'All seeds fixed to {seed}.')
