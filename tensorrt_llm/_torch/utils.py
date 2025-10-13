import contextlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.quantization.utils import fp4_utils

is_torch_compiling_flag = False

aux_stream_name_list = [
    'Attention',
    'MoeShared',
    'MoeChunkingOverlap',
    'MoeBalancer',
]
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
            TensorWrapper(x.data_ptr(), x.dtype, x.shape,
                          x.stride())) if x.is_cuda else x
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
    is_sf_swizzled: bool = True

    @property
    def shape(self):
        return self.fp4_tensor.shape


def compute_swizzled_sf_shape(row: int, col: int):
    padded_row = pad_up(row, 128)
    padded_col = pad_up(col, 4)
    return padded_row, padded_col


def swizzle_sf(sf: torch.Tensor,
               rows: int,
               cols: int,
               scaling_vector_size: int = 16):
    """Swizzle FP4 scaling factors using C++ torch op implementation
    Args:
        sf: [b, rows, cols_sf] or [rows, cols_sf]. The original unswizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols_sf: ceil_div(cols, scaling_vector_size) where cols is the number of columns of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        [b * pad_up(rows, 128) * pad_up(cols_sf, 4), ] 1D swizzled scaling factors, possibly with rows and cols padded.
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    sf = sf.view(-1, rows, sf_cols)
    return torch.ops.trtllm.block_scale_interleave(sf)


def unswizzle_sf(sf: torch.Tensor,
                 rows: int,
                 cols: int,
                 scaling_vector_size: int = 16):
    """Swizzle FP4 scaling factors using C++ torch op implementation
    Args:
        sf: The (padded and) swizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols: cols of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        2D unswizzled scaling factors
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    sf = sf.view(-1, rows, sf_cols)
    return torch.ops.trtllm.block_scale_interleave_reverse(sf).view(-1, sf_cols)


@torch.library.custom_op("trtllm::reswizzle_sf", mutates_args=())
def reswizzle_sf(sf: torch.Tensor,
                 rows: int,
                 cols: int,
                 scaling_vector_size: int = 16) -> torch.Tensor:
    """Reswizzle FP4 scaling factors using C++ torch op implementation.
       It unswizzles the scaling factors in each partition first, then concatenates them together, and finally swizzles them back.
    Args:
        sf: The (padded and) swizzled scaling factors.
        rows: rows of the original unquantized tensor
        cols: cols of the original unquantized tensor
        scaling_vector_size: the size of the scaling vector
    Returns:
        1D reswizzled scaling factors
    """
    sf_cols = ceil_div(cols, scaling_vector_size)
    padded_rows, padded_sf_cols = compute_swizzled_sf_shape(rows, sf_cols)
    padded_cols = padded_sf_cols * scaling_vector_size

    assert sf.numel() % (padded_rows * padded_sf_cols) == 0
    num_partitions = sf.numel() // (padded_rows * padded_sf_cols)

    sf_reshaped = sf.view(num_partitions, padded_rows, padded_sf_cols)

    # Unswizzle each partition
    sf_unswizzled = unswizzle_sf(sf_reshaped, padded_rows, padded_cols,
                                 scaling_vector_size)

    # Brings the unswizzled scaling factors in each partition together
    total_rows = num_partitions * rows
    sf_unswizzled = sf_unswizzled.view(num_partitions, padded_rows,
                                       padded_sf_cols)
    sf_concatenated = sf_unswizzled[:, :rows, :sf_cols].contiguous(
    )  # TODO: This will incur a elementwise kernel
    sf_concatenated = sf_concatenated.view(total_rows, sf_cols)

    # Finally swizzle the concatenated scaling factors
    return swizzle_sf(sf_concatenated, total_rows, cols, scaling_vector_size)


@torch.library.register_fake("trtllm::reswizzle_sf")
def _(sf, rows, cols, scaling_vector_size=16):
    sf_cols = ceil_div(cols, scaling_vector_size)
    padded_rows, padded_sf_cols = compute_swizzled_sf_shape(rows, sf_cols)
    num_partitions = sf.numel() // (padded_rows * padded_sf_cols)
    total_rows = num_partitions * rows
    sz = pad_up(total_rows, 128) * pad_up(cols, 4)
    return sf.new_empty(sz)


def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    # Following code is equivalent to 1 << (x - 1).bit_length()
    # But this impl does not contain bit_length() so can be used by torch compile.
    # It can correctly handle 64bit number which should be enough for now.
    n = x - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


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

    return tuple(num_token_buckets[::-1])


def get_last_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = last_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2
    return tuple(num_token_buckets[::-1])


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


@contextlib.contextmanager
def piecewise_cuda_graph(enable: bool):
    prev_enable = get_piecewise_cuda_graph_flag()
    set_piecewise_cuda_graph_flag(enable)
    try:
        yield
    finally:
        set_piecewise_cuda_graph_flag(prev_enable)


def set_per_request_piecewise_cuda_graph_flag(enable: bool):
    _global_attrs.per_request_piecewise_cuda_graph_flag = enable


def get_per_request_piecewise_cuda_graph_flag() -> bool:
    return getattr(_global_attrs, 'per_request_piecewise_cuda_graph_flag', True)


def create_lm_head_tp_mapping(mapping: Mapping, token_count: int) -> Mapping:
    # We use heuristic to determine the lm_head_tp_size
    # Since token_count=256 will hit the boundary of math-bound problem
    # We use 256 // token_count to determine the lm_head_tp_size
    lm_head_tp_size_raw = 256 // token_count
    lm_head_tp_size = nearest_in_buckets(lm_head_tp_size_raw,
                                         [1, mapping.gpus_per_node])
    assert mapping.tp_size % lm_head_tp_size == 0
    lm_head_pp_size = mapping.pp_size * mapping.tp_size // lm_head_tp_size

    return Mapping(
        world_size=lm_head_tp_size * lm_head_pp_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=lm_head_tp_size,
        pp_size=lm_head_pp_size,
        enable_attention_dp=mapping.enable_attention_dp,
        enable_lm_head_tp_in_adp=mapping.enable_lm_head_tp_in_adp,
    )


def get_device_uuid(device_idx: int) -> str:
    """Get the UUID of a CUDA device using torch cuda api"""

    property = torch.cuda.get_device_properties(device_idx)
    uuid = "GPU-" + str(property.uuid)
    return uuid
