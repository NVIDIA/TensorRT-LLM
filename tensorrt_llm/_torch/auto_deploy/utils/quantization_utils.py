# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from fnmatch import fnmatch
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ..custom_ops.quantization.quant import FP4_GLOBAL_SCALE_MAX, FP8_MAX
from .logger import ad_logger
from .node_utils import (
    extract_weight_name,
    get_quantization_params_from_linear_node,
    is_bmm_op,
    is_linear_op,
    is_op,
    modelopt_dynamic_block_quantize_op,
    modelopt_quantize_op,
)

try:
    from tensorrt_llm.quantization.utils.fp4_utils import float4_sf_dtype
except ImportError:
    float4_sf_dtype = None


FLOAT8_DTYPES = tuple(
    dtype
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, dtype_name, None)) is not None
)


def ensure_tma_col_major(t: torch.Tensor) -> torch.Tensor:
    """Re-apply TMA-aligned column-major layout to a torch.int scale tensor.

    torch.cat always produces contiguous (row-major) output, which violates
    DeepGEMM's stride(-2) == 1 requirement. This function re-creates the
    column-major layout.

    Only affects Blackwell + DeepGEMM: post_load_hook converts scales to
    UE8M0 (torch.int) col-major only in that configuration. On other GPUs
    or without DeepGEMM, scales remain torch.float and this is a no-op.
    """
    if t.dtype != torch.int:
        return t  # Not UE8M0
    # Both stride(-2) and stride(-1) must match the col-major TMA-aligned
    # layout DeepGEMM expects. When size(-1) == 1, a row-major contiguous
    # tensor has stride(-2) == 1 too, so checking stride(-2) alone would
    # incorrectly short-circuit and leave stride(-1) un-aligned.
    expected_inner = ((t.size(-2) + 3) // 4) * 4
    if t.stride(-2) == 1 and t.stride(-1) == expected_inner:
        return t  # Already column-major and TMA-aligned

    remove_dim = False
    if t.dim() == 2:
        t = t.unsqueeze(0)
        remove_dim = True

    b, mn, k = t.shape
    # TMA alignment: 16 bytes / 4 bytes per int32 = 4 elements
    aligned_mn = ((mn + 3) // 4) * 4

    # Create column-major buffer via transpose trick (same as get_col_major_tma_aligned_packed_tensor)
    col_major = torch.transpose(
        torch.empty((b, k, aligned_mn), device=t.device, dtype=torch.int), 1, 2
    )
    col_major[:, :mn, :] = t
    result = col_major[:, :mn, :]

    return result.squeeze(0) if remove_dim else result


def modelopt_fp4_scale_to_cutlass_fp4_scale(modelopt_scale: torch.Tensor) -> torch.Tensor:
    """Converts the modelopt FP4 per-block weight scale to the cutlass format (padded and swizzled)."""
    m, n = modelopt_scale.shape
    pad_m = (128 - m % 128) % 128
    pad_n = (4 - n % 4) % 4
    padded_tensor = F.pad(modelopt_scale, (0, pad_n, 0, pad_m))

    padded_m, padded_n = padded_tensor.shape
    padded_tensor = padded_tensor.view(padded_m // 128, 4, 32, padded_n // 4, 4)
    return padded_tensor.permute(0, 3, 2, 1, 4).reshape(-1).view(torch.uint8)


def cutlass_fp4_scale_to_modelopt_fp4_scale(
    cutlass_scale: torch.Tensor, weight_shape: Tuple
) -> torch.Tensor:
    """Converts the cutlass FP4 (padded and swizzled) per-block weight scale to the cutlass format."""
    m, n = weight_shape
    n = n // 16

    padded_tensor = cutlass_scale.reshape((m + 127) // 128, (n + 3) // 4, 32, 4, 4).permute(
        0, 3, 2, 1, 4
    )
    return padded_tensor.reshape(padded_tensor.size(0) * 128, padded_tensor.size(3) * 4)[
        :m, :n
    ].view(torch.float8_e4m3fn)


def fp4_global_scale(input: torch.Tensor) -> torch.Tensor:
    """Computes the FP4 per-tensor global scale of the input."""
    return FP4_GLOBAL_SCALE_MAX / torch.max(torch.abs(input).to(torch.float))


def fp8_scale(input: torch.Tensor) -> torch.Tensor:
    """Computes the FP8 per-tensor scale of the input."""
    return torch.max(torch.abs(input).to(torch.float)) / FP8_MAX


def is_quantized_graph(gm: GraphModule):
    """Check if the graph is quantized by modelopt."""
    for n in gm.graph.nodes:
        if is_linear_op(n):
            input_params, weight_params, output_params = get_quantization_params_from_linear_node(n)
            if input_params or weight_params or output_params:
                return True

    return False


def is_quantized_op(node: Node):
    return (
        True
        if modelopt_quantize_op is not None
        and modelopt_dynamic_block_quantize_op is not None
        and is_op(node, [modelopt_quantize_op, modelopt_dynamic_block_quantize_op])
        else False
    )


def remove_output_quantizers(gm: GraphModule):
    """Remove output quatnizer if any from the graph."""
    for n in gm.graph.nodes:
        if is_linear_op(n) and len(n.users) == 1:
            user = list(n.users.keys())[0]
            if is_quantized_op(user):
                # skip the output quantizer
                user.replace_all_uses_with(n)


def get_quantization_from_linear_node(node: torch.fx.node.Node):
    """Get quantization format(str) from quantization parameters."""
    input_params, weight_params, _ = get_quantization_params_from_linear_node(node)

    if input_params and weight_params:
        if input_params.is_fp8_e4m3() and weight_params.is_fp8_e4m3():
            return "FP8"
        elif input_params.is_fp4_e2m1() and weight_params.is_fp4_e2m1():
            return "NVFP4"
        else:
            ad_logger.info("Found unsupported quantized nodes. Performance will be sub-optimal.")

    return ""


def _pattern_matches(modname: str, pattern: str) -> bool:
    """Check if an exclude pattern matches the module name.

    Keep behavior aligned with upstream: evaluate exclude entries via fnmatch.
    This preserves exact module-path excludes (for example:
    ``model.layers.0.self_attn.q_a_proj``) and wildcard entries.
    """
    return fnmatch(modname, pattern)


def should_skip_quantization(
    node_or_name: Union[Node, str],
    excluded_patterns: list[str],
) -> bool:
    """Check if a node or parameter name should be skipped based on excluded patterns.

    Supports both glob patterns (e.g., "*gate*") and simple substring patterns
    (e.g., "gate" matches "model.layers.0.block_sparse_moe.gate").
    """
    if isinstance(node_or_name, str):
        modname, _, _ = node_or_name.rpartition(".")
    else:
        if not (is_linear_op(node_or_name) or is_bmm_op(node_or_name)):
            return True
        weight_name = extract_weight_name(node_or_name)
        # extract_weight_name can return False when weight node is not found (e.g. after
        # PR 10718 get_weight_node uses forward mapping; some graph shapes may have no mapping).
        if weight_name is False or not isinstance(weight_name, str):
            return True
        modname = weight_name.rpartition(".")[0]

    return any(_pattern_matches(modname, pattern) for pattern in excluded_patterns)


def _extract_modname(node_or_name: Union[Node, str]) -> Optional[str]:
    """Extract the module name from a graph node or parameter name string.

    Returns None if the module name cannot be determined.
    """
    if isinstance(node_or_name, str):
        modname, _, _ = node_or_name.rpartition(".")
        return modname

    if not (is_linear_op(node_or_name) or is_bmm_op(node_or_name)):
        return None
    weight_name = extract_weight_name(node_or_name)
    if weight_name is False or not isinstance(weight_name, str):
        return None
    return weight_name.rpartition(".")[0]


def should_skip_mixed_precision_quantization(
    node_or_name: Union[Node, str],
    algo_name: str,
    quantized_layers: Dict[str, Dict],
) -> bool:
    """For MIXED_PRECISION configs, check whether this node's per-layer algo matches.

    Returns True (skip) if the layer is absent from quantized_layers or its
    per-layer quant_algo doesn't match ``algo_name``.
    """
    modname = _extract_modname(node_or_name)
    if modname is None:
        return True

    layer_info = quantized_layers.get(modname)
    if layer_info is None:
        return True

    layer_algo = layer_info.get("quant_algo", "").upper()
    if layer_algo != algo_name.upper():
        return True

    return False


def is_mixed_precision_config(qcfg: Dict) -> bool:
    """Return True if the quantization config uses MIXED_PRECISION."""
    return qcfg.get("quant_algo", "").upper() == "MIXED_PRECISION"


def mixed_precision_has_algo(qcfg: Dict, algo_name: str) -> bool:
    """Return True if the MIXED_PRECISION config contains any layer with the given algo."""
    for layer_info in qcfg.get("quantized_layers", {}).values():
        if layer_info.get("quant_algo", "").upper() == algo_name.upper():
            return True
    return False


def extract_scales_from_node(node: Node, scale_names: list[str]) -> Dict[str, Optional[Node]]:
    """
    Extracts scale tensors from node.args/kwargs using a fixed list of expected scale names.
    """
    scales = {}
    args = list(node.args)

    # Try kwargs first
    for i, name in enumerate(scale_names):
        scales[name] = node.kwargs.get(name, None)

    # Fallback to positional args (starting after input, weight, bias)
    for i, name in enumerate(scale_names):
        if scales[name] is None and len(args) > 3 + i:
            scales[name] = args[3 + i]

    return scales


def unpack_uint8_to_int4_weight_2d(
    packed_weight: torch.Tensor, weights_scaling_factor: torch.Tensor
) -> torch.Tensor:
    """
    Reverse of `modelopt.torch.export.quant_utils.pack_int4_in_uint8` for the 2D case.
    Args:
      packed_weight: (out_dim//2, in_dim), uint8
      weights_scaling_factor: (out_dim, in_dim//block_size)  [used for shape/block inference]
    Returns:
      int8 weights in [-8,7], shape (out_dim, in_dim)
    """
    assert packed_weight.dim() == 2
    assert packed_weight.dtype == torch.uint8

    out_half, in_dim = packed_weight.shape
    out_dim = out_half * 2
    block_size = in_dim // weights_scaling_factor.shape[-1]
    assert in_dim % block_size == 0

    # inverse of: reshaped = int8_tensor.T.reshape(in_dim, out_dim//2, 2)
    pw = packed_weight.T.contiguous()  # (in_dim, out_dim//2)

    low = (pw & 0x0F).to(torch.int16)
    high = ((pw >> 4) & 0x0F).to(torch.int16)

    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)

    rebuilt = torch.stack([low, high], dim=-1)  # (in_dim, out_dim//2, 2)
    int8_T = rebuilt.reshape(in_dim, out_dim)  # (in_dim, out_dim)
    int8_W = int8_T.T.contiguous()  # (out_dim, in_dim)
    return int8_W


# copied from modelopt.torch.export.quant_utils.pack_int4_in_uint8
def pack_int4_in_uint8(weight, weights_scaling_factor):
    """Packs the INT4 weights into uint8 tensor."""
    out_dim = weight.shape[-2]
    assert out_dim % 2 == 0, f"Cannot pack weight. Out dimension {out_dim} is not an even number."
    in_dim = weight.shape[-1]
    block_size = weight.shape[-1] // weights_scaling_factor.shape[-1]
    int8_tensor = (
        (weight / weights_scaling_factor[..., :, torch.arange(in_dim) // block_size])
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
    )
    # -- Handle the MoE (3D) case vs. the 2D case --
    if int8_tensor.dim() == 3:
        transpose = int8_tensor.permute(0, 2, 1)
        transpose = transpose.reshape(-1, in_dim, out_dim // 2, 2)
        val0 = transpose[..., 0] & 0x0F
        val1 = transpose[..., 1] & 0x0F
        packed_byte = val0 | (val1 << 4)
        return packed_byte.permute(0, 2, 1).contiguous().view(torch.uint8)
    else:
        # 2D weights: shape typically (out_dim, in_dim)
        reshaped = int8_tensor.T.reshape(in_dim, out_dim // 2, 2)
        val0 = reshaped[..., 0] & 0x0F
        val1 = reshaped[..., 1] & 0x0F
        packed_byte = val0 | (val1 << 4)
        return packed_byte.T.contiguous().view(torch.uint8)
