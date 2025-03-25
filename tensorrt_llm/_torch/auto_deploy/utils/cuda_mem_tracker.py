import gc
from contextlib import contextmanager

import torch

from .logger import ad_logger


@contextmanager
def cuda_memory_tracker(logger=ad_logger):
    """
    Context manager to track CUDA memory allocation differences.

    Logs a warning if there is an increase in memory allocation after the
    code block, which might indicate a potential memory leak.
    """
    mem_before = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated()
        leaked = mem_after - mem_before
        if leaked > 0:
            logger.warning(f"Potential memory leak detected, leaked memory: {leaked} bytes")
