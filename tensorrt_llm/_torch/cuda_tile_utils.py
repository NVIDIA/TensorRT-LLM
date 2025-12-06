import functools
import platform

from ..logger import logger

IS_CUDA_TILE_AVAILABLE = False


@functools.lru_cache()
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def ceil_div(a, b):
    return (a + b - 1) // b


if platform.system() != "Windows":
    try:
        import cuda.tile
        IS_CUDA_TILE_AVAILABLE = True
    except ImportError:
        logger.warning("cuda.tile is not available, certain kernels will not be available")
