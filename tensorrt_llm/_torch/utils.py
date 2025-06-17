import contextlib
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.quantization.utils import fp4_utils

is_torch_compiling_flag = False

aux_stream_name_list = ['Attention', 'MoeShared', 'MoeChunkingOverlap']
AuxStreamType = Enum(
    'AuxStreamType',
    aux_stream_name_list,
)
EventType = Enum(
    'EventType',
    ['Main', *aux_stream_name_list],
    start=0,
)


def set_torch_compiling(enable: bool):
    global is_torch_compiling_flag
    is_torch_compiling_flag = enable


def is_torch_compiling() -> bool:
    global is_torch_compiling_flag
    return is_torch_compiling_flag


_global_attrs = threading.local()


def get_global_attrs():
    return _global_attrs


_model_extra_attrs = threading.local()


def get_model_extra_attrs():
    return getattr(_model_extra_attrs, 'attrs', None)


@contextlib.contextmanager
def model_extra_attrs(attrs: Dict):
    old_attrs = getattr(_model_extra_attrs, 'attrs', None)
    _model_extra_attrs.attrs = attrs
    try:
        yield
    finally:
        _model_extra_attrs.attrs = old_attrs


def with_model_extra_attrs(get_attrs):

    def decorator(func):

        def wrapper(self, *args, **kwargs):
            with model_extra_attrs(get_attrs(self)):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def make_weak_ref(x):

    if isinstance(x, torch.Tensor):
        return convert_to_torch_tensor(
            TensorWrapper(x.data_ptr(), x.dtype, x.shape)) if x.is_cuda else x
    elif isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    elif isinstance(x, list):
        return [make_weak_ref(i) for i in x]
    elif isinstance(x, dict):
        return {k: make_weak_ref(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return x
    else:
        raise TypeError(f"Invalid type {type(x)} to make weak ref")


@dataclass
class Fp4QuantizedTensor:
    fp4_tensor: torch.Tensor
    scaling_factor: torch.Tensor

    @property
    def shape(self):
        return self.fp4_tensor.shape


_disable_fp4_allgather = os.getenv("TLLM_DISABLE_FP4_ALLGATHER", "0") == "1"


def disable_fp4_allgather():
    return _disable_fp4_allgather


def compute_swizzled_sf_shape(row: int, col: int):
    padded_row = pad_up(row, 128)
    padded_col = pad_up(col, 4)
    return padded_row, padded_col


def swizzle_sf(sf: torch.Tensor,
               row: int,
               col: int,
               scaling_vector_size: int = 16):
    """Swizzle FP4 scaling factors using C++ torch op implementation"""
    if row is not None and col is not None:
        sf_cols = ceil_div(col, scaling_vector_size)
        sf = sf.view(-1, row, sf_cols)
    return torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(sf)


def unswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    """Unswizzle scaling factors using C++ torch op implementation"""
    sf_cols = (col + scaling_vector_size - 1) // scaling_vector_size
    # Reshape to expected input shape for C++ op
    sf = sf.view(row, sf_cols)

    return torch.ops.tensorrt_llm.nvfp4_block_scale_interleave_reverse(sf)


def reswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    """Reswizzle scaling factors for multiple partitions using C++ ops"""
    factor = scaling_vector_size * 4
    num_m_tiles = ceil_div(row, 128)
    num_k_tiles = ceil_div(col, factor)
    partition_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
    num_partitions = sf.numel() // partition_size

    # Unswizzle each partition
    sf_reshaped = sf.view(num_partitions, num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(2, 4)
    sf_unswizzle = sf_unswizzle.reshape(num_partitions, num_m_tiles * 32 * 4,
                                        num_k_tiles * 4)

    # Concatenate partitions and re-swizzle for the new dimensions
    total_rows = num_partitions * row
    sf_cols = ceil_div(col, scaling_vector_size)
    sf_concatenated = sf_unswizzle[:, :row].reshape(total_rows, sf_cols)

    return torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(sf_concatenated)


def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    return 1 << (x - 1).bit_length()


def last_positive_power_of_2(x: int) -> int:
    next = next_positive_power_of_2(x)
    if next == x:
        return next

    return next // 2


def nearest_in_buckets(x: int, buckets: List[int]) -> int:
    return min(max(next_positive_power_of_2(x), buckets[0]), buckets[-1])


def get_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = next_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2

    return tuple(num_token_buckets)


def get_last_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = last_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2
    return tuple(num_token_buckets)


def fp4_scale_infer_shape(input_shapes: List[List[int]]):
    """Calculate the dimensions of the fp4 scale tensor.
    """
    out_shape, scale_shape = fp4_utils.get_fp4_shape(input_shapes[0],
                                                     sf_vec_size=16)
    return scale_shape * 2


_enable_piecewise_cuda_graph = True


def set_piecewise_cuda_graph_flag(enable: bool):
    global _enable_piecewise_cuda_graph
    _enable_piecewise_cuda_graph = enable


def get_piecewise_cuda_graph_flag() -> bool:
    global _enable_piecewise_cuda_graph
    return _enable_piecewise_cuda_graph
