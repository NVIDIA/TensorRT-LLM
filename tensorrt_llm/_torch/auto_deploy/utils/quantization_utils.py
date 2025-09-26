from fnmatch import fnmatch
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ..custom_ops.quant import FP4_GLOBAL_SCALE_MAX, FP8_MAX
from .logger import ad_logger
from .node_utils import (
    extract_param_names_from_lin_node,
    get_quantization_params_from_linear_node,
    is_bmm_op,
    is_linear_op,
    is_op,
    modelopt_dynamic_block_quantize_op,
    modelopt_quantize_op,
)

try:
    from ....quantization.utils.fp4_utils import float4_sf_dtype
except ImportError:
    float4_sf_dtype = None


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
            print(input_params, weight_params)

    return ""


def should_skip_quantization(
    node_or_name: Union[Node, str],
    excluded_patterns: list[str],
) -> bool:
    """Check if a node or parameter name should be skipped based on excluded patterns."""
    if isinstance(node_or_name, str):
        modname, _, _ = node_or_name.rpartition(".")
    else:
        if not (is_linear_op(node_or_name) or is_bmm_op(node_or_name)):
            return True
        param_name, _ = extract_param_names_from_lin_node(node_or_name)
        modname, _, _ = param_name.rpartition(".")

    return any(fnmatch(modname, pattern) for pattern in excluded_patterns)


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
