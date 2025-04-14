import contextlib
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

from .pipeline_interface import PipelineInterface

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
    elif isinstance(x, PipelineInterface):
        return tuple(make_weak_ref(tensor) for tensor in x)
    else:
        raise TypeError(f"Invalid type {type(x)} to make weak ref")


@dataclass
class Fp4QuantizedTensor:
    fp4_tensor: torch.Tensor
    scaling_factor: torch.Tensor


_disable_fp4_allgather = os.getenv("TLLM_DISABLE_FP4_ALLGATHER", "0") == "1"


def disable_fp4_allgather():
    return _disable_fp4_allgather


def swizzle_sf(sf: torch.Tensor,
               row: int,
               col: int,
               scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_full = torch.zeros(num_m_tiles * 32 * 4,
                          num_k_tiles * 4,
                          dtype=sf.dtype,
                          device=sf.device)
    sf_full[:row, :(col //
                    scaling_vector_size)] = sf[:row, :(col //
                                                       scaling_vector_size)]
    sf_full_reshaped = sf_full.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_full_swizzle = sf_full_reshaped.transpose(1, 3)
    sf_swizzle = sf_full_swizzle.reshape(-1)
    return sf_swizzle


def unswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, :(col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


def reswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    partition_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
    num_partitions = sf.numel() // partition_size
    sf_reshaped = sf.view(num_partitions, num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(2, 4)
    sf_unswizzle = sf_unswizzle.reshape(num_partitions, num_m_tiles * 32 * 4,
                                        num_k_tiles * 4)
    total_rows = num_partitions * row
    num_m_tiles_out = (total_rows + 128 - 1) // 128
    sf_out = torch.zeros(
        num_m_tiles_out,
        4,
        32,
        num_k_tiles,
        4,
        dtype=sf.dtype,
        device=sf.device,
    )
    sf_out_reshaped = sf_out.view(num_m_tiles_out * 32 * 4, num_k_tiles * 4)
    sf_out_reshaped[:total_rows] = sf_unswizzle[:, :row].reshape(total_rows, -1)
    sf_out_swizzle = sf_out.transpose(1, 3).reshape(-1)
    return sf_out_swizzle


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

    return num_token_buckets


def get_last_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = last_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2
    return num_token_buckets
