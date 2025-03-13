from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ..custom_ops.quant import FP4_GLOBAL_SCALE_MAX, FP8_MAX, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
from .logger import ad_logger
from .node_utils import (
    get_quantization_params_from_linear_node,
    is_linear_op,
    is_op,
    modelopt_dynamic_block_quantize_op,
    modelopt_quantize_op,
)

try:
    from tensorrt_llm._torch.quantization.utils import float4_sf_dtype
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


class QuantizationImpl:
    """An abstracted static class for node quantization."""

    @staticmethod
    def create(quant_type_or_node: Union[str, Node]):
        """Returns the QuantizationImpl based on quantization type or quantized linear node."""
        if isinstance(quant_type_or_node, str):
            quantization_impl_map = {
                "": None,
                "FP8": FP8QuantizationImpl,
                "NVFP4": FP4QuantizationImpl,
            }
            return quantization_impl_map[quant_type_or_node]

        for q in [FP4QuantizationImpl, FP8QuantizationImpl]:
            if is_op(quant_type_or_node, q.target_op()):
                return q
        return None

    @staticmethod
    def target_op():
        """Returns the target quantization ops."""
        raise NotImplementedError("Abstract Interface")

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        """Returns the quantized weight from the original unquantized weight."""
        raise NotImplementedError("Abstract Interface")

    @staticmethod
    def scale_names() -> List[str]:
        """Returns the list of names of the scales for this quantization."""
        return []

    @staticmethod
    def default_scales(original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        """Returns a dict of the default scale values for this quantization."""
        return {}

    @staticmethod
    def load_hook(state_dict, prefix, *args, weight_name: str):
        """Load hook for state_dict quantization pre-processing."""
        pass

    @staticmethod
    def convert_amax_hook(state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        pass

    @staticmethod
    def shard_scales(dim, rank, world_size, **kwargs) -> Dict[str, torch.Tensor]:
        """Returns a dict of sharded quantization scales."""
        return {}

    @staticmethod
    def shard_load_hook(
        state_dict,
        prefix,
        *args,
        weight_name: str,
        weight_shape: Tuple,
        dim: int,
        rank: int,
        world_size: int,
    ):
        """Load hook for state_dict quantized sharding pre-processing.

        This load_hook handles the sharding of the quantization scales.
        """
        pass

    @staticmethod
    def fuse_linear_weights(weights, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class FP8QuantizationImpl(QuantizationImpl):
    @staticmethod
    def target_op():
        return torch.ops.quant.fp8_linear

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(
            original_weight, dtype=torch.float8_e4m3fn, device=original_weight.device
        )

    @staticmethod
    def scale_names() -> List[str]:
        return ["input_scale", "weight_scale"]

    @staticmethod
    def default_scales(original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        return {"input_scale": torch.tensor(1.0), "weight_scale": torch.tensor(1.0)}

    @staticmethod
    def load_hook(state_dict, prefix, *args, weight_name):
        if weight_name in state_dict:
            weight = state_dict[weight_name]
            if weight.dtype != torch.float8_e4m3fn:
                scale = fp8_scale(state_dict[weight_name])
                state_dict[weight_name] = (state_dict[weight_name] / scale).to(torch.float8_e4m3fn)
                state_dict[weight_name + "_scale"] = scale

    @staticmethod
    def convert_amax_hook(state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        if amax_name in state_dict:
            amax = state_dict[amax_name]
            scale = amax / FP8_MAX
            state_dict[scale_name] = scale

    @staticmethod
    def fuse_linear_weights(
        weights, weight_scale, input_scale
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not all(s == input_scale[0] for s in input_scale):
            raise NotImplementedError(f"Cannot fuse due to mismatched input_scale {input_scale}")

        # Handle quantized weights with weight_scale.
        # First we upcast to FP32 precision and then downcast back to the original precision (FP8)
        assert weights[0].dtype == torch.float8_e4m3fn, "Only support FP8 quantized weights fusion."
        fused_fp32_weights = torch.cat(
            [t.to(torch.float) * s for t, s in zip(weights, weight_scale)], dim=0
        )
        new_weight_scale = torch.max(torch.stack(weight_scale))
        fused_fp8_weights = (fused_fp32_weights / new_weight_scale).to(weights[0].dtype)

        return fused_fp8_weights, {
            "weight_scale": new_weight_scale,
            "input_scale": input_scale[0].clone(),
        }


def _shard_fp4_weight_scale(weight_scale, sharded_uint8_weight_shape, dim, rank, world_size):
    assert weight_scale.dim() == 1
    weight_shape_original = list(sharded_uint8_weight_shape)
    weight_shape_original[dim] = weight_shape_original[dim] * world_size
    weight_shape_original[-1] *= 2
    modelopt_weight_scale = cutlass_fp4_scale_to_modelopt_fp4_scale(
        weight_scale, tuple(weight_shape_original)
    )
    return modelopt_fp4_scale_to_cutlass_fp4_scale(
        modelopt_weight_scale.tensor_split(world_size, dim=dim)[rank]
    )


class FP4QuantizationImpl(QuantizationImpl):
    @staticmethod
    def target_op():
        return torch.ops.quant.fp4_linear

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        m, n = original_weight.shape
        return torch.empty((m, n // 2), dtype=torch.uint8, device=original_weight.device)

    @staticmethod
    def scale_names() -> List[str]:
        return ["input_scale", "weight_scale", "alpha"]

    @staticmethod
    def default_scales(original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        m, n = original_weight_shape
        # scaling factors m is padded along 128 and n is padded along 4.
        # check cpp/tensorrt_llm/plugins/fp4GemmPlugin/fp4GemmPlugin.cpp for more details.
        n = n // TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        padded_m = (m + 127) // 128 * 128
        padded_n = (n + 3) // 4 * 4
        # definition of scales
        # input_scale: FP4_GLOBAL_SCALE_MAX / input_amax
        # weight_scale_2: FP4_GLOBAL_SCALE_MAX / weight_amax
        # alpha: 1 / (input_scale * weight_scale_2)
        return {
            "input_scale": torch.tensor(1.0 / 6.0),
            "weight_scale": torch.empty((padded_m * padded_n), dtype=torch.uint8),
            "alpha": torch.tensor(1.0 / 6.0),
        }

    @staticmethod
    def load_hook(state_dict, prefix, *args, weight_name):
        if weight_name in state_dict:
            input_scale_name = weight_name.rsplit(".", 1)[0] + ".input_scale"
            alpha_name = weight_name.rsplit(".", 1)[0] + ".alpha"
            weight = state_dict[weight_name]
            # ModelOpt quantized graph path
            if weight.dtype != torch.uint8:
                assert input_scale_name in state_dict
                # Unquantized weight
                amax_name = weight_name + "_quantizer._amax"
                if amax_name in state_dict:
                    weight_scale_2 = FP4_GLOBAL_SCALE_MAX / state_dict[amax_name].to(torch.float)
                else:
                    weight_scale_2 = fp4_global_scale(weight)
                weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(
                    weight.to("cuda"),
                    weight_scale_2.to("cuda"),
                    TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
                    False,
                )
                state_dict[weight_name] = weight_fp4
                state_dict[weight_name + "_scale"] = weight_scale
                state_dict[weight_name + "_scale_2"] = weight_scale_2
                state_dict[alpha_name] = 1 / (weight_scale_2 * state_dict[input_scale_name])
            # Unified HF ckpt path
            else:
                if (
                    weight_name + "_scale_2" in state_dict
                    and weight_name + "_scale" in state_dict
                    and input_scale_name in state_dict
                    and float4_sf_dtype
                ):
                    state_dict[alpha_name] = (
                        state_dict[weight_name + "_scale_2"] * state_dict[input_scale_name]
                    )
                    state_dict[input_scale_name] = 1 / state_dict[input_scale_name]
                    weight_scale = state_dict[weight_name + "_scale"].view(float4_sf_dtype)
                    ori_shape = weight_scale.shape
                    state_dict[weight_name + "_scale"] = (
                        torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                            weight_scale.view(torch.uint8).cpu().contiguous()
                        )
                        .reshape(ori_shape)
                        .view(float4_sf_dtype)
                        .reshape(-1)
                    )

    def convert_amax_hook(state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        if amax_name in state_dict:
            amax = state_dict[amax_name]
            scale = ((448 * 6) / amax).float()
            state_dict[scale_name] = scale

    @staticmethod
    def shard_scales(dim, rank, world_size, weight_scale, alpha, input_scale, weight_shape):
        result = {}
        result["alpha"] = alpha
        result["input_scale"] = input_scale
        result["weight_scale"] = _shard_fp4_weight_scale(
            weight_scale, weight_shape, dim, rank, world_size
        )

        return result

    @staticmethod
    def shard_load_hook(
        state_dict, prefix, *args, weight_name, weight_shape, dim, rank, world_size
    ):
        if weight_name + "_scale" in state_dict:
            weight_scale = state_dict[weight_name + "_scale"]
            state_dict[weight_name + "_scale"] = _shard_fp4_weight_scale(
                weight_scale, weight_shape, dim, rank, world_size
            )

    @staticmethod
    def fuse_linear_weights(
        weights, weight_scale, alpha, input_scale
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not all(s == input_scale[0] for s in input_scale):
            raise NotImplementedError(f"Cannot fuse due to mismatched input_scale {input_scale}")

        if not all(s == alpha[0] for s in alpha):
            raise NotImplementedError(f"Cannot fuse due to mismatched alpha {alpha}")

        fused_weights = torch.cat(weights, dim=0)
        fused_weight_scale = torch.cat(weight_scale, dim=0)

        return fused_weights, {
            "weight_scale": fused_weight_scale,
            "alpha": alpha[0],
            "input_scale": input_scale[0].clone(),
        }


def is_quantized_graph(gm: GraphModule):
    """Check if the graph is quantized by modelopt."""
    for n in gm.graph.nodes:
        if is_linear_op(n, include_quantization=False):
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
        if is_linear_op(n, include_quantization=False) and len(n.users) == 1:
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
