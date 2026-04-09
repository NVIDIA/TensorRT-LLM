# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.utils import ActivationType

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ...utils.quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    fp4_global_scale,
    is_mixed_precision_config,
    mixed_precision_has_algo,
    should_skip_mixed_precision_quantization,
    should_skip_quantization,
)
from ..interface import SharedConfig, TransformInfo, TransformRegistry
from .quantization import (
    FP8LinearQuantizationFromConfig,
    NVFP4LinearQuantizationFromConfig,
    Quantization,
)

try:
    from .....quantization.utils.fp4_utils import float4_sf_dtype as _float4_sf_dtype
    from ...custom_ops.quantization.quant import (
        FP4_GLOBAL_SCALE_MAX,
        TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
    )
except ImportError:
    FP4_GLOBAL_SCALE_MAX = None
    TRTLLM_NVFP4_SCALING_VECTOR_SIZE = None
    _float4_sf_dtype = None


def _quantize_moe_node(
    gm: GraphModule,
    node: Node,
    quant_impl: Quantization,
    quantized_op: Callable[..., Node],
):
    """
    Replace a torch.ops.auto_deploy.torch_moe node with its quantized version,
    quantizing each expert weight list and registering scales + hooks.
    Automatically handles different scale configurations per quantization type.
    """
    w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)

    scale_keys = quant_impl.scale_names()

    def quantize_param_list(weight_names: List[str]) -> Tuple[List[Node], List[List[Node]]]:
        new_attrs = []
        scale_nodes_group = []
        for name in weight_names:
            orig_weight = gm.get_parameter(name)
            new_weight = quant_impl.quantize_weight(orig_weight)

            # Replace parameter in submodule
            modname, _, attrname = name.rpartition(".")
            submod = gm.get_submodule(modname)
            setattr(submod, attrname, nn.Parameter(new_weight, requires_grad=False))

            # Register new scale buffers
            for scale_name, scale_val in quant_impl.default_scales(orig_weight.shape).items():
                submod.register_buffer(scale_name, scale_val)

            # Register load hook
            gm._register_load_state_dict_pre_hook(partial(quant_impl.load_hook, weight_name=name))

            # Create get_attr nodes for new param and each scale
            with gm.graph.inserting_before(node):
                new_weight_attr = gm.graph.get_attr(name)
                new_attrs.append(new_weight_attr)
                scales = [gm.graph.get_attr(modname + "." + s) for s in scale_keys]
                scale_nodes_group.append(scales)

        return new_attrs, scale_nodes_group

    # Quantize all three expert weights
    w1_attrs, w1_scales = quantize_param_list(w1_names)
    w2_attrs, w2_scales = quantize_param_list(w2_names)
    w3_attrs, w3_scales = quantize_param_list(w3_names)

    # Collect scale tensors per scale type across w1, w2, w3
    def collect_scales(index: int) -> Tuple[List[Node], List[Node], List[Node]]:
        return (
            [s[index] for s in w1_scales],
            [s[index] for s in w2_scales],
            [s[index] for s in w3_scales],
        )

    # Prepare args
    args = [
        node.args[0],  # x
        node.args[1],  # selected_experts
        node.args[2],  # routing_weights
        w1_attrs,
        w2_attrs,
        w3_attrs,
    ]

    for idx in range(len(scale_keys)):
        s1, s2, s3 = collect_scales(idx)
        args.extend([s1, s2, s3])

    # Extract is_gated_mlp and act_fn from the original node
    # These can be in args[6:] or in kwargs
    is_gated_mlp = True  # default
    act_fn = ActivationType.Silu  # default

    if len(node.args) > 6:
        is_gated_mlp = node.args[6]
    elif "is_gated_mlp" in node.kwargs:
        is_gated_mlp = node.kwargs["is_gated_mlp"]

    if len(node.args) > 7:
        act_fn = node.args[7]
    elif "act_fn" in node.kwargs:
        act_fn = node.kwargs["act_fn"]

    # Prepare kwargs for the quantized op
    kwargs = {
        "is_gated_mlp": is_gated_mlp,
        "act_fn": act_fn,
    }

    # Replace the current node with the quantized version
    with gm.graph.inserting_after(node):
        new_node = gm.graph.call_function(
            quantized_op,
            args=tuple(args),
            kwargs=kwargs,
        )
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)


# TODO(Fridah-nv): robust handling similar to `extract_param_names_from_lin_node` or expand it
def _extract_moe_weight_param_lists(moe_node: Node) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a torch.ops.moe.torch_moe node in gm.graph, extract three lists of
    the parameter names for w1_weight, w2_weight, and w3_weight.

    Returns:
      (w1_names, w2_names, w3_names), each a list of strings like 'layer.expert_0.w1.weight'
    """
    # args layout: (x, selected_experts, routing_weights, w1_list, w2_list, w3_list)
    try:
        w1_list, w2_list, w3_list = moe_node.args[3:6]
    except ValueError:
        raise RuntimeError(
            f"Expected moe_node.args to have at least 6 entries, got {len(moe_node.args)}"
        )

    def _unwrap_list(arg) -> List[str]:
        if not isinstance(arg, (list, tuple)):
            raise TypeError(f"Expected a Python list/tuple of get_attr Nodes, got {type(arg)}")
        names: List[str] = []
        for elt in arg:
            if not isinstance(elt, Node) or elt.op != "get_attr":
                raise RuntimeError(f"Expected each list element to be a get_attr Node, got {elt}")
            names.append(elt.target)
        return names

    w1_names = _unwrap_list(w1_list)
    w2_names = _unwrap_list(w2_list)
    w3_names = _unwrap_list(w3_list)

    return w1_names, w2_names, w3_names


@TransformRegistry.register("quantize_fp8_moe")
class QuantizeFP8MOE(FP8LinearQuantizationFromConfig):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    quantized version using the quant_algo from quant_config.
    """

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_fp8_moe

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        is_mixed = is_mixed_precision_config(qcfg)
        if is_mixed:
            if not mixed_precision_has_algo(qcfg, self.algo_name):
                return gm, TransformInfo(
                    skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
                )
            quantized_layers = qcfg.get("quantized_layers", {})
        elif (
            qcfg.get("quant_algo", "").upper() != self.algo_name
            and qcfg.get("quant_method", "").upper() != self.algo_name
        ):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("exclude_modules", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            all_weight_names = w1_names + w2_names + w3_names

            if any(should_skip_quantization(n, excluded_patterns) for n in all_weight_names):
                continue

            if is_mixed and any(
                should_skip_mixed_precision_quantization(n, self.algo_name, quantized_layers)
                for n in all_weight_names
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info


@TransformRegistry.register("quantize_nvfp4_moe")
class QuantizeNVFP4MOE(NVFP4LinearQuantizationFromConfig):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    quantized version using the quant_algo from quant_config.

    Unlike the linear NVFP4 path (which defers scale processing to FuseNVFP4Linear),
    the MoE path has no post-load fusion pass, so quantize_weight computes the full
    kernel-ready scales during the transform (weight_scale swizzled uint8, alpha, inv_input_scale).
    The load_hook handles reloading from float or pre-quantized checkpoints.
    """

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_nvfp4_moe

    def default_scales(self, original_weight_shape):
        if TRTLLM_NVFP4_SCALING_VECTOR_SIZE is None:
            return super().default_scales(original_weight_shape)
        # Fallback: flat uint8 placeholder matching kernel-ready (swizzled) format.
        # Shape must match what _swizzle_nvfp4_scale produces and what the checkpoint stores.
        m, n = original_weight_shape
        n_blocks = n // TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        padded_m = math.ceil(m / 128) * 128
        padded_n = math.ceil(n_blocks / 4) * 4
        return {
            "input_scale": torch.tensor(1.0 / 6.0),
            "weight_scale": torch.zeros(padded_m, padded_n, dtype=torch.uint8),
            "weight_scale_2": torch.tensor(1.0 / 6.0),
        }

    def load_hook(self, state_dict, prefix, *args, weight_name):
        """Full-processing load hook for NVFP4 MoE experts.

        Handles two cases:
        1. Non-quantized checkpoint (weight dtype != uint8): quantize weight and compute
           raw FP8 scales, then process into kernel-ready format.
        2. Pre-quantized checkpoint (weight dtype == uint8): process raw FP8 scales
           (if present) into kernel-ready format.

        MoE experts register scales with names "input_scale", "weight_scale",
        "weight_scale_2" directly on the expert submodule (not using the attrname
        prefix from _scale_buffer_key), so this override uses the correct key names.
        """
        if weight_name not in state_dict:
            return

        modname = weight_name.rsplit(".", 1)[0]
        input_scale_key = modname + ".input_scale"
        weight_scale_key = modname + ".weight_scale"
        ws2_key = modname + ".weight_scale_2"

        weight = state_dict[weight_name]

        # Case 1: non-quantized checkpoint — quantize weight and store raw scales
        if weight.dtype != torch.uint8:
            if FP4_GLOBAL_SCALE_MAX is None or TRTLLM_NVFP4_SCALING_VECTOR_SIZE is None:
                return
            amax_key = weight_name + "_quantizer._amax"
            if amax_key in state_dict:
                ws2_global = FP4_GLOBAL_SCALE_MAX / state_dict[amax_key].to(torch.float)
            else:
                ws2_global = fp4_global_scale(weight)
            weight_fp4, weight_scale_cutlass = torch.ops.trtllm.fp4_quantize(
                weight.to("cuda"),
                ws2_global.to("cuda"),
                TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
                False,
            )
            state_dict[weight_name] = weight_fp4
            m, k = weight.shape
            # Store raw FP8 per-block weight scale (2D, float8_e4m3fn)
            state_dict[weight_scale_key] = cutlass_fp4_scale_to_modelopt_fp4_scale(
                weight_scale_cutlass, (m, k)
            )
            # Store raw weight_scale_2 = 1/ws2_global
            state_dict[ws2_key] = 1 / torch.clamp(ws2_global, min=1e-30)

        # Case 1 & 2: process weight_scale into kernel-ready uint8 2D format.
        # fuse_nvfp4_moe expects per-expert weight_scale as uint8 2D [padded_M, padded_N]
        # so it can stack them into 3D [num_experts, padded_M, padded_N].
        if weight_scale_key not in state_dict:
            return
        raw_ws = state_dict[weight_scale_key]

        if raw_ws.dtype == torch.uint8:
            # Already in uint8 kernel-ready format.
            if raw_ws.ndim == 2:
                # Already 2D — correct format, nothing to do.
                return
            # Flat 1D uint8 from checkpoint: reshape to 2D [padded_M, padded_N].
            m_w, k_half = weight.shape  # weight is uint8 FP4-packed, shape (M, K//2)
            k_blocks = (k_half * 2) // TRTLLM_NVFP4_SCALING_VECTOR_SIZE
            padded_m = math.ceil(m_w / 128) * 128
            padded_n = math.ceil(k_blocks / 4) * 4
            state_dict[weight_scale_key] = raw_ws.reshape(padded_m, padded_n)
            return

        # FP8 weight_scale: swizzle to uint8 2D for fuse_nvfp4_moe compatibility.
        if raw_ws.dtype != torch.float8_e4m3fn:
            return
        if _float4_sf_dtype is None:
            return

        from .fuse_quant import _swizzle_nvfp4_scale

        # Use default 1/6 if input_scale not in source state_dict (e.g. fresh float checkpoint)
        raw_is = state_dict.get(input_scale_key, torch.tensor(1.0 / 6.0)).float()
        if ws2_key not in state_dict:
            return
        raw_ws2 = state_dict[ws2_key].float()

        # Handle flat 1D FP8: reshape to 2D (M, K_blocks) before swizzling
        if raw_ws.ndim == 1:
            m_w, k_half = weight.shape  # weight is uint8-packed, shape (M, K//2)
            k_blocks = (k_half * 2) // TRTLLM_NVFP4_SCALING_VECTOR_SIZE
            raw_ws = raw_ws.reshape(m_w, k_blocks).view(torch.float8_e4m3fn)

        # Swizzle weight_scale: FP8 2D → uint8 2D [padded_M, padded_N] (NOT flattened)
        state_dict[weight_scale_key] = _swizzle_nvfp4_scale(raw_ws, _float4_sf_dtype, flatten=False)
        # Compute alpha = raw_ws2 * raw_is
        state_dict[ws2_key] = torch.clamp(raw_ws2 * raw_is, min=1e-30)
        # Invert input_scale
        state_dict[input_scale_key] = 1.0 / torch.clamp(raw_is, min=1e-30)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        is_mixed = is_mixed_precision_config(qcfg)
        if is_mixed:
            if not mixed_precision_has_algo(qcfg, self.algo_name):
                return gm, TransformInfo(
                    skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
                )
            quantized_layers = qcfg.get("quantized_layers", {})
        elif qcfg.get("quant_algo", "").upper() != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("exclude_modules", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            all_weight_names = w1_names + w2_names + w3_names

            if any(should_skip_quantization(n, excluded_patterns) for n in all_weight_names):
                continue

            if is_mixed and any(
                should_skip_mixed_precision_quantization(n, self.algo_name, quantized_layers)
                for n in all_weight_names
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info


@TransformRegistry.register("quantize_finegrained_fp8_moe")
class QuantizeFineGrainedFP8MOE(Quantization):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    FineGrainedFP8 quantized version.

    This transform handles FineGrained FP8 quantization config format:
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate", "lm_head"]
        }
    """

    algo_name = "fp8"

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_finegrained_fp8_moe

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)

    def scale_names(self) -> List[str]:
        return ["weight_scale_inv"]

    def default_scales(self, original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        # Default block size is 128x128 for FineGrained FP8
        N, K = original_weight_shape
        block_n, block_k = 128, 128
        scale_shape = (math.ceil(N / block_n), math.ceil(K / block_k))
        return {"weight_scale_inv": torch.ones(scale_shape, dtype=torch.bfloat16)}

    def build_custom_args_for_linear(self, scales: Dict[str, "Node"]) -> Tuple:
        return ([scales["weight_scale_inv"]],)

    def load_hook(self, state_dict, prefix, *args, weight_name: str):
        """Load hook to handle HF FineGrainedFP8 checkpoint format."""
        if weight_name not in state_dict:
            return

        weight = state_dict[weight_name]
        if weight.dtype == torch.float8_e4m3fn:
            scale_inv_name = weight_name + "_scale_inv"
            if scale_inv_name in state_dict:
                mod_prefix = weight_name.rsplit(".", 1)[0]
                state_dict[mod_prefix + ".weight_scale_inv"] = state_dict[scale_inv_name]

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Gate by quant_method in quant_config (HF style)
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        quant_method = str(qcfg.get("quant_method", "")).lower()
        if quant_method != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        if qcfg.get("weight_block_size") is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("modules_to_not_convert", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            # Check experts are allowed (no excludes)
            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            if any(
                should_skip_quantization(n, excluded_patterns)
                for n in (w1_names + w2_names + w3_names)
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info
