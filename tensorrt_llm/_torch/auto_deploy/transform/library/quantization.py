# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import Sequence
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...custom_ops.quantization.quant import (
    FP4_GLOBAL_SCALE_MAX,
    FP8_MAX,
    TRTLLM_NVFP4_COLUMN_SIZE,
    TRTLLM_NVFP4_ROW_SIZE,
    TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
    is_column_major,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    WeightBiasInfoCache,
    extract_op_args,
    extract_weight_name,
    extract_weight_nodes,
    get_quantization_params_from_linear_node,
    is_bmm_op,
    is_linear_op,
    is_op,
)
from ...utils.quantization_utils import (
    fp4_global_scale,
    fp8_scale,
    get_quantization_from_linear_node,
    is_mixed_precision_config,
    is_quantized_graph,
    is_quantized_op,
    mixed_precision_has_algo,
    remove_output_quantizers,
    should_skip_mixed_precision_quantization,
    should_skip_quantization,
)
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

try:
    from tensorrt_llm.quantization.utils.fp4_utils import float4_sf_dtype
except ImportError:
    float4_sf_dtype = None


def _is_view_or_reshape_node(node: Node) -> bool:
    if not isinstance(node, Node):
        return False
    if node.op == "call_method":
        return node.target in {"view", "reshape"}
    return is_op(
        node,
        [
            torch.ops.aten.view,
            torch.ops.aten.reshape,
            torch.ops.auto_deploy.view,
        ],
    )


def _view_base_and_shape(node: Node) -> Tuple[object, Tuple[object, ...]]:
    if node.op == "call_method":
        base = node.args[0]
        shape_args = node.args[1:]
    elif is_op(node, torch.ops.auto_deploy.view):
        base = node.args[0]
        shape_args = (node.args[1],)
    else:
        base = node.args[0]
        shape_args = node.args[1:]
    if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
        shape_args = tuple(shape_args[0])
    return base, tuple(shape_args)


def _shape_from_view_or_meta(node: Node) -> Tuple[object, ...] | None:
    if _is_view_or_reshape_node(node):
        _, view_shape = _view_base_and_shape(node)
        return view_shape
    meta_val = node.meta.get("val") if isinstance(node, Node) else None
    if meta_val is not None and hasattr(meta_val, "shape"):
        return tuple(meta_val.shape)
    return None


def _is_static_dim_value(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _extract_layer_type_hint(node: Node, default: str = "unknown") -> str:
    if not isinstance(node, Node):
        return default
    if node.op == "call_function":
        try:
            [layer_type] = extract_op_args(node, "layer_type")
        except RuntimeError:
            layer_type = None
        if layer_type is not None:
            return layer_type
    return node.kwargs.get("layer_type", default)


class Quantization(BaseTransform):
    """Abstract base for config-driven quantization of a single algorithm/op-kind.

    Subclasses MUST implement:
      - algo_name: str                          # e.g., "FP8" or "NVFP4"
      - target_op(self) -> Callable
      - quantize_weight(self, w: Tensor) -> Tensor
      - scale_names(self) -> List[str]
      - default_scales(self, shape: Tuple) -> Dict[str, Tensor]
      - build_custom_args_for_linear(self, scales: Dict[str, Node]) -> Tuple
      - _apply(self, gm, cm, factory, shared_config) -> (gm, TransformInfo)

    Optional (define only if needed):
      - load_hook(self, state_dict, prefix, *args, weight_name: str)
      - post_load_hook(self, module, incompatible_keys, weight_name: str)
      - convert_amax_hook(self, state_dict, prefix, *args, scale_name: str, amax_name: str)
    """

    algo_name: str = None  # override in subclasses

    # Algorithm API
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
    def post_load_hook(state_dict, prefix, *args, weight_name: str):
        """Load hook for state_dict quantization post-processing."""
        pass

    @staticmethod
    def convert_amax_hook(state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        pass

    @staticmethod
    def build_custom_args_for_linear(  # renamed to reflect args
        scale_getattrs: Dict[str, Node],
    ) -> Tuple[object, ...]:
        return ()

    # Transform logic for ModelOPT linear layers
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with WeightBiasInfoCache():
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

            excluded = qcfg.get("exclude_modules", [])
            cnt = 0
            for n in gm.graph.nodes:
                if not is_linear_op(n):
                    continue
                if should_skip_quantization(n, excluded):
                    continue
                if is_mixed and should_skip_mixed_precision_quantization(
                    n, self.algo_name, quantized_layers
                ):
                    continue
                self._insert_quantized_linear(gm, n, is_quantized_graph=False)
                cnt += 1

            return gm, TransformInfo(
                skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=(cnt == 0)
            )

    def _insert_quantized_linear(
        self,
        gm: GraphModule,
        node: Node,
        is_quantized_graph: bool = False,
        load_hook_kwargs: Optional[Dict[str, object]] = None,
        custom_kwargs: Optional[Dict[str, object]] = None,
    ):
        """Replaces the matmul node with a new custom quantized linear node.

        The state_dict is also updated to contain the sharded weights.
        """
        weight_nodes = extract_weight_nodes(node)
        if len(weight_nodes.weights) == 0:
            raise ValueError(f"Linear node {node.name} has no weight")
        lin_weight = weight_nodes.weights[0]

        new_param = nn.Parameter(self.quantize_weight(lin_weight.tensor), requires_grad=False)
        modname, _, attrname = lin_weight.node_key.rpartition(".")

        setattr(lin_weight.submod, attrname, new_param)

        # check modelopt quantizers from graph
        if is_quantized_graph:
            input_params, weight_params, output_params = get_quantization_params_from_linear_node(
                node
            )
            # redirect to input and weight
            node.args = (input_params.input_node, weight_params.input_node, *node.args[2:])

            # redirect output to skip output quantizer if any
            user = list(node.users.keys())[0]
            if len(node.users) == 1 and is_quantized_op(user):
                user.replace_all_uses_with(node)

            # when loading the state_dict, we need to convert input amax to input scale
            input_scale_name = self.scale_names()[0]
            scale_name = modname + "." + input_scale_name
            amax_name = input_params.amax.target
            hook = partial(
                self.convert_amax_hook,
                scale_name=scale_name,
                amax_name=amax_name,
            )
            gm._register_load_state_dict_pre_hook(hook)
            # Note: canonicalize_graph() will remove input/weight/output quantizer

        for scale_name, scale in self.default_scales(lin_weight.tensor.shape).items():
            lin_weight.submod.register_buffer(scale_name, scale)

        gm._register_load_state_dict_pre_hook(
            partial(
                self.load_hook,
                weight_name=lin_weight.node_key,
                **(load_hook_kwargs or {}),
            )
        )
        post_load_hook = getattr(type(self), "post_load_hook", None)
        if post_load_hook is not None and post_load_hook is not Quantization.post_load_hook:
            gm.register_load_state_dict_post_hook(
                partial(self.post_load_hook, weight_name=lin_weight.node_key)
            )

        with gm.graph.inserting_before(node):
            scales = {}
            for scale_name in self.scale_names():
                scales[scale_name] = gm.graph.create_node("get_attr", modname + "." + scale_name)

        custom_args = self.build_custom_args_for_linear(scales)

        # Extract sharding hints by name so we don't depend on positional layout.
        [tp_mode, output_sizes, tp_min_local_shape, layer_type] = extract_op_args(
            node, "tp_mode", "output_sizes", "tp_min_local_shape", "layer_type"
        )
        [inp, weight, bias] = extract_op_args(node, "input", "weight", "bias")
        node.target = self.target_op()
        node.args = (inp, weight, bias, *custom_args)
        node.kwargs = {
            **node.kwargs,
            "tp_mode": tp_mode,
            "output_sizes": output_sizes,
            "tp_min_local_shape": tp_min_local_shape,
            "layer_type": layer_type,
            **(custom_kwargs or {}),
        }

    def _insert_quantized_bmm(
        self,
        gm: GraphModule,
        node: Node,
        is_quantized_graph: bool = False,
    ) -> bool:
        """Replace a bmm op with its quantized equivalent and wire scales/state_dict hooks.

        Returns:
            True if quantization was applied; False if skipped (e.g., unknown shape).
        """
        weight_node = node.args[1]

        # Weight is a parameter
        if weight_node.op == "get_attr":
            # Handle parameter tensor
            param_name = weight_node.target
            original_weight = gm.get_parameter(param_name)
            weight_shape = original_weight.shape

            # Quantize the weight
            new_param = nn.Parameter(self.quantize_weight(original_weight), requires_grad=False)

            # Update the parameter in the model
            modname, _, attrname = param_name.rpartition(".")
            submod = gm.get_submodule(modname)
            setattr(submod, attrname, new_param)

            # Register load state dict hook
            hook = partial(self.load_hook, weight_name=param_name)
            gm._register_load_state_dict_pre_hook(hook)
            post_load_hook = getattr(type(self), "post_load_hook", None)
            if post_load_hook is not None and post_load_hook is not Quantization.post_load_hook:
                hook = partial(self.post_load_hook, weight_name=param_name)
                gm.register_load_state_dict_post_hook(hook)

            # Setup scale names and target module for parameter case
            def get_scale_name(scale_name):
                return attrname + "_" + scale_name

            scale_target_module = submod
            scale_name_prefix = f"{modname}."

        # Weight is a dynamic tensor
        elif hasattr(weight_node, "meta") and "val" in weight_node.meta:
            weight_shape = weight_node.meta["val"].shape

            # Create a unique identifier for this dynamic weight node
            node_id = f"bmm_dynamic_{id(node)}"

            # Setup scale names and target module for dynamic case
            def get_scale_name(scale_name):
                return f"{node_id}_{scale_name}"

            scale_target_module = gm  # Register in root module
            scale_name_prefix = ""

        else:
            # If we can't determine the shape, skip quantization
            return False

        # Common logic for both parameter and dynamic tensor cases
        # Register scales in the target module
        for scale_name, scale in self.default_scales(weight_shape).items():
            scale_buffer_name = get_scale_name(scale_name)
            scale_target_module.register_buffer(scale_buffer_name, scale)

        # Change node target to quantized bmm op
        node.target = self.target_op()

        # Insert scale nodes
        with gm.graph.inserting_before(node):
            scales = {}
            for scale_name in self.scale_names():
                scale_buffer_name = get_scale_name(scale_name)
                scales[scale_name] = gm.graph.create_node(
                    "get_attr", f"{scale_name_prefix}{scale_buffer_name}"
                )

        # Update node arguments and kwargs
        scale_values = [scales[scale_name] for scale_name in self.scale_names()]
        node.args = (*node.args, *scale_values)
        return True


@TransformRegistry.register("quantize_fp8_linear_from_config")
class FP8LinearQuantizationFromConfig(Quantization):
    algo_name = "FP8"

    def target_op(self):
        return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale"]

    def default_scales(self, _shape: Tuple) -> Dict[str, torch.Tensor]:
        return {"input_scale": torch.tensor(1.0), "weight_scale": torch.tensor(1.0)}

    def build_custom_args_for_linear(self, scales: Dict[str, Node]) -> Tuple:
        # (input_scale(list), weight_scale(list), input_zp(list), weight_zp(list))
        return ([scales["input_scale"]], [scales["weight_scale"]], [], [])

    def load_hook(self, state_dict, prefix, *args, weight_name):
        prefix = prefix or ""
        weight_key = prefix + weight_name
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            if weight.dtype != torch.float8_e4m3fn:
                scale = fp8_scale(state_dict[weight_key])
                state_dict[weight_key] = (state_dict[weight_key] / scale).to(torch.float8_e4m3fn)
                state_dict[weight_key + "_scale"] = scale
            else:
                mod_prefix = prefix + weight_name.rsplit(".", 1)[0]
                activation_scale_name = mod_prefix + ".activation_scale"
                weight_scale_inv_name = weight_key + "_scale_inv"
                if activation_scale_name in state_dict:
                    state_dict[mod_prefix + ".input_scale"] = state_dict.pop(activation_scale_name)
                if weight_scale_inv_name in state_dict:
                    state_dict[mod_prefix + ".weight_scale"] = state_dict.pop(weight_scale_inv_name)

    def convert_amax_hook(self, state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        if amax_name in state_dict:
            amax = state_dict[amax_name]
            scale = amax / FP8_MAX
            state_dict[scale_name] = scale

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        # Skip if the config specifies block-wise (fine-grained) FP8 quantization via
        # weight_block_size; those should be handled by FineGrainedFP8LinearQuantization.
        if qcfg and qcfg.get("weight_block_size"):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        return super()._apply(gm, cm, factory, shared_config)


@TransformRegistry.register("quantize_nvfp4_linear_from_config")
class NVFP4LinearQuantizationFromConfig(Quantization):
    algo_name = "NVFP4"

    def target_op(self):
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        m, n = w.shape
        return torch.empty((m, n // 2), dtype=torch.uint8, device=w.device)

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale", "alpha"]

    def _pad_m_n(self, m: int, n: int) -> Tuple[int, int]:
        """Pad m and n to be divisible by 128 and 4 respectively.
        Check cpp/tensorrt_llm/plugins/fp4GemmPlugin/fp4GemmPlugin.cpp for more details.
        """
        padded_m = math.ceil(m / TRTLLM_NVFP4_ROW_SIZE) * TRTLLM_NVFP4_ROW_SIZE
        padded_n = math.ceil(n / TRTLLM_NVFP4_COLUMN_SIZE) * TRTLLM_NVFP4_COLUMN_SIZE
        return padded_m, padded_n

    def default_scales(self, original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        m, n = original_weight_shape
        n = n // TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        padded_m, padded_n = self._pad_m_n(m, n)
        # definition of scales
        # input_scale: FP4_GLOBAL_SCALE_MAX / input_amax
        # weight_scale_2: FP4_GLOBAL_SCALE_MAX / weight_amax
        # alpha: 1 / (input_scale * weight_scale_2)
        return {
            "input_scale": torch.tensor(1.0 / 6.0),
            "weight_scale": torch.empty((padded_m, padded_n), dtype=torch.uint8),
            # "weight_scale": torch.empty((m, n), dtype=torch.uint8),
            # "weight_scale": torch.empty(padded_m * padded_n, dtype=torch.float8_e4m3fn),
            # "weight_scale": torch.empty(padded_m * padded_n, dtype=torch.uint8),
            "alpha": torch.tensor(1.0 / 6.0),
        }

    def build_custom_args_for_linear(self, scales: Dict[str, Node]) -> Tuple:
        # weight_scale list is (cutlass_vec, alpha)
        return ([scales["input_scale"]], [scales["weight_scale"], scales["alpha"]], [], [])

    def load_hook(self, state_dict, prefix, *args, weight_name):
        # Prepend prefix so the hook works when the GraphModule is a submodule
        # of the model on which load_state_dict is called (e.g., VLM models
        # where the text model lives at model.language_model.*).
        weight_name = prefix + weight_name
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
                state_dict[alpha_name] = 1 / torch.clamp(
                    weight_scale_2 * state_dict[input_scale_name], min=1e-30
                )
            # Unified HF ckpt path
            else:
                if (
                    weight_name + "_scale_2" in state_dict
                    and weight_name + "_scale" in state_dict
                    and input_scale_name in state_dict
                    and float4_sf_dtype
                ):
                    alpha = state_dict[weight_name + "_scale_2"] * state_dict[input_scale_name]
                    alpha = torch.clamp(alpha, min=1e-30)
                    state_dict[alpha_name] = alpha
                    input_scale = torch.clamp(state_dict[input_scale_name], min=1e-30)
                    state_dict[input_scale_name] = 1 / input_scale
                    weight_scale = state_dict[weight_name + "_scale"].view(float4_sf_dtype)
                    # Round the weight block scale factors to 128x4 and then swizzle.
                    weight_scale_swizzled = torch.ops.trtllm.block_scale_interleave(
                        weight_scale.view(torch.uint8).cpu().contiguous()
                    ).view(float4_sf_dtype)

                    m, n = weight_scale.shape
                    # scaling factors m is padded along 128 and n is padded along 4.
                    # check cpp/tensorrt_llm/plugins/fp4GemmPlugin/fp4GemmPlugin.cpp for more details.
                    padded_m, padded_n = self._pad_m_n(m, n)
                    swizzled_shape = (padded_m, padded_n)

                    state_dict[weight_name + "_scale"] = weight_scale_swizzled.reshape(
                        swizzled_shape
                    )

    def convert_amax_hook(self, state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        if amax_name in state_dict:
            amax = state_dict[amax_name]
            scale = ((448 * 6) / amax).float()
            state_dict[scale_name] = scale


@TransformRegistry.register("quantize_int4_linear_from_config")
class INT4LinearQuantizationFromConfig(Quantization):
    """Config-based INT4 (AWQ) for the unified ModelOpt checkpoints."""

    algo_name = "W4A16_AWQ"

    @staticmethod
    def target_op():
        return torch.ops.auto_deploy.torch_fake_quant_int4_linear.default

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        N, K = original_weight.shape
        return torch.empty((N // 2, K), dtype=torch.uint8, device=original_weight.device)

    @staticmethod
    def scale_names() -> List[str]:
        return ["pre_quant_scale", "weight_scale"]

    @staticmethod
    def default_scales(original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        N, K = original_weight_shape
        BLOCK = 128
        assert K % BLOCK == 0, "K must be divisible by 128 for INT4 block quant."
        return {
            "pre_quant_scale": torch.ones(K, dtype=torch.float32),
            "weight_scale": torch.empty((N, K // BLOCK), dtype=torch.float32),
        }

    @staticmethod
    def build_custom_args_for_linear(scales: Dict[str, Node]) -> Tuple[object, ...]:
        return ([scales["pre_quant_scale"]], [scales["weight_scale"]], [], [])

    @staticmethod
    def load_hook(state_dict, prefix, *args, weight_name: str):
        """
        Unified ckpt passthrough:
          - weight: keep packed uint8 (N//2, K)
          - pre_quant_scale buffer: (K,) or ones(K) if missing
          - weight_scale buffer: (N, K//128) float32 (no reshape, no *7 here)
        """
        if weight_name not in state_dict:
            return
        BLOCK = 128

        mod_prefix, _, _ = weight_name.rpartition(".")
        pre_qs_ckpt = f"{mod_prefix}.pre_quant_scale"  # may be absent
        wscale_ckpt = f"{mod_prefix}.weight_scale"  # required

        pre_qs_buf = f"{mod_prefix}.pre_quant_scale"
        wscale_buf = f"{mod_prefix}.weight_scale"

        w_packed = state_dict[weight_name]
        if w_packed.dtype != torch.uint8:
            return

        assert wscale_ckpt in state_dict, f"Missing {wscale_ckpt}"
        wscale_mat = state_dict[wscale_ckpt]  # (N, K//128) float32

        N_half, K = w_packed.shape
        N = N_half * 2
        assert K % BLOCK == 0
        assert wscale_mat.shape == (N, K // BLOCK), (
            f"weight_scale shape {wscale_mat.shape} != {(N, K // BLOCK)}"
        )

        # pre_quant_scale: use if present else ones(K)
        if pre_qs_ckpt in state_dict:
            pre_qs_val = state_dict[pre_qs_ckpt].to(torch.float32)
            if pre_qs_val.dim() == 0:
                pre_qs_val = pre_qs_val.expand(K).clone()
            else:
                assert pre_qs_val.numel() == K, (
                    f"{pre_qs_ckpt} has {pre_qs_val.numel()} elems, expected {K}"
                )
        else:
            pre_qs_val = torch.ones(K, dtype=torch.float32)

        state_dict[weight_name] = w_packed  # (N//2, K) uint8
        state_dict[pre_qs_buf] = pre_qs_val  # (K,) float32
        state_dict[wscale_buf] = wscale_mat.to(torch.float32)  # (N, K//128)


@TransformRegistry.register("quantize_fp8_bmm_from_config")
class FP8BMMQuantizationFromConfig(Quantization):
    algo_name = "FP8"

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_fp8_bmm.default

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)

    def scale_names(self) -> List[str]:
        return ["input_scale", "weight_scale"]

    def default_scales(self, _shape: Tuple) -> Dict[str, torch.Tensor]:
        return {"input_scale": torch.tensor(1.0), "weight_scale": torch.tensor(1.0)}

    def load_hook(self, state_dict, prefix, *args, weight_name):
        """Pre-hook: Only handle quantization."""
        if weight_name in state_dict:
            weight = state_dict[weight_name]

            # If weight is not already quantized (not float8)
            if weight.dtype != torch.float8_e4m3fn:
                # Compute weight scale
                weight_scale = fp8_scale(weight)
                weight = (weight / weight_scale).to(torch.float8_e4m3fn)
                state_dict[weight_name + "_scale"] = weight_scale
                state_dict[weight_name] = weight

    def post_load_hook(self, module, incompatible_keys, weight_name):
        """Post-hook: Handle column-major conversion after parameter is loaded."""
        # Navigate to the actual parameter
        *path, attr_name = weight_name.split(".")
        target_module = module
        for p in path:
            target_module = getattr(target_module, p)

        if hasattr(target_module, attr_name):
            param = getattr(target_module, attr_name)
            if isinstance(param, torch.nn.Parameter):
                # Convert to column-major format
                if not is_column_major(param):
                    with torch.no_grad():
                        # Create column-major version
                        param_cm = param.transpose(-2, -1).contiguous().transpose(-2, -1)
                        # Replace the parameter
                        setattr(
                            target_module,
                            attr_name,
                            torch.nn.Parameter(param_cm, requires_grad=param.requires_grad),
                        )

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

        if is_mixed_precision_config(qcfg):
            ad_logger.warning(
                "FP8 BMM quantization does not support MIXED_PRECISION checkpoints, skipping."
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        if qcfg.get("quant_algo", "").upper() != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded = qcfg.get("exclude_modules", [])
        cnt = 0

        with WeightBiasInfoCache():
            for n in gm.graph.nodes:
                if not is_bmm_op(n):
                    continue
                if should_skip_quantization(n, excluded):
                    continue
                if self._insert_quantized_bmm(gm, n, is_quantized_graph=False):
                    cnt += 1

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=True
        )


@TransformRegistry.register("quantize_fp8_from_graph")
class FP8QuantizationFromGraph(FP8LinearQuantizationFromConfig):
    """Fuse ModelOpt-quantized FP8 linears into fused ops."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not is_quantized_graph(gm):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        cnt = 0
        for n in gm.graph.nodes:
            if is_linear_op(n):
                algo_n = get_quantization_from_linear_node(n)
                if (algo_n or "").upper() != "FP8":
                    continue
                self._insert_quantized_linear(gm, n, is_quantized_graph=True)
                cnt += 1

        remove_output_quantizers(gm)
        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=True
        )


@TransformRegistry.register("quantize_nvfp4_from_graph")
class NVFP4QuantizationFromGraph(NVFP4LinearQuantizationFromConfig):
    """Fuse ModelOpt-quantized NVFP4 linears into fused ops."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not is_quantized_graph(gm):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        cnt = 0
        for n in gm.graph.nodes:
            if is_linear_op(n):
                algo_n = get_quantization_from_linear_node(n)
                if (algo_n or "").upper() != "NVFP4":
                    continue
                self._insert_quantized_linear(gm, n, is_quantized_graph=True)
                cnt += 1

        # if cnt > 0:
        remove_output_quantizers(gm)
        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=True
        )


@TransformRegistry.register("quantize_int4_gptq_linear_from_config")
class INT4GPTQLinearQuantizationFromConfig(Quantization):
    """Config-based INT4 GPTQ quantization for GPTQ-quantized checkpoints.

    GPTQ uses:
      - qweight: [K/8, N] int32 (8 packed int4 values per int32)
      - qzeros: [G, N/8] int32 (packed zero points)
      - scales: [G, N] float (per-group scales)
    """

    algo_name = "GPTQ"

    @staticmethod
    def target_op():
        return torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear.default

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        """Returns placeholder qweight tensor [K/8, N] int32."""
        N, K = original_weight.shape
        assert K % 8 == 0, "K must be divisible by 8 for GPTQ int4 packing."
        return torch.empty((K // 8, N), dtype=torch.int32, device=original_weight.device)

    @staticmethod
    def scale_names() -> List[str]:
        return ["scales", "qzeros"]

    @staticmethod
    def default_scales(original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        """Returns placeholder tensors for GPTQ scales and qzeros."""
        N, K = original_weight_shape
        BLOCK = 128  # GPTQ group size
        assert K % BLOCK == 0, "K must be divisible by 128 for GPTQ block quant."
        assert N % 8 == 0, "N must be divisible by 8 for GPTQ qzeros packing."
        G = K // BLOCK
        return {
            "scales": torch.empty((G, N), dtype=torch.float16),
            "qzeros": torch.empty((G, N // 8), dtype=torch.int32),
        }

    @staticmethod
    def build_custom_args_for_linear(scales: Dict[str, Node]) -> Tuple[object, ...]:
        """Build args for torch_fake_quant_int4_gptq_linear:
        (input, weight, bias, input_scale, weight_scale, input_zp, weight_zp)
        -> input_scale=[], weight_scale=[scales], input_zp=[], weight_zp=[qzeros]
        """
        return ([], [scales["scales"]], [], [scales["qzeros"]])

    @staticmethod
    def load_hook(state_dict, prefix, *args, weight_name: str):
        """
        Load hook for GPTQ checkpoints:
          - qweight: keep as [K/8, N] int32
          - scales: [G, N] float16
          - qzeros: [G, N/8] int32

        GPTQ checkpoint uses naming convention:
          - {prefix}qweight
          - {prefix}scales
          - {prefix}qzeros
        """

        mod_prefix, _, _ = weight_name.rpartition(".")

        qweight_ckpt = f"{mod_prefix}.qweight"
        scales_ckpt = f"{mod_prefix}.scales"
        qzeros_ckpt = f"{mod_prefix}.qzeros"

        if qweight_ckpt not in state_dict:
            return

        qweight = state_dict[qweight_ckpt]
        if qweight.dtype != torch.int32:
            return

        K_packed, N = qweight.shape  # [K/8, N]
        K = K_packed * 8

        assert scales_ckpt in state_dict, f"Missing {scales_ckpt}"
        scales = state_dict[scales_ckpt]  # [G, N]
        G = scales.shape[0]

        assert qzeros_ckpt in state_dict, f"Missing {qzeros_ckpt}"
        qzeros = state_dict[qzeros_ckpt]  # [G, N/8]

        # Validate GPTQ weight layout
        assert K % G == 0, f"K ({K}) must be divisible by G ({G})"
        assert scales.shape == (G, N), f"scales shape {scales.shape} != {(G, N)}"
        assert qzeros.shape == (G, N // 8), f"qzeros shape {qzeros.shape} != {(G, N // 8)}"

        # Map to our buffer names
        state_dict[weight_name] = qweight  # [K/8, N] int32
        state_dict[f"{mod_prefix}.scales"] = scales.to(torch.float16)  # [G, N]
        # GPTQ v1 format stores (zero_point - 1); convert to v2 by adding 0x11111111
        # See: gptqmodel.utils.model.convert_gptq_v1_to_v2_format_module
        qzeros_v2 = qzeros + 0x11111111
        state_dict[f"{mod_prefix}.qzeros"] = qzeros_v2  # [G, N/8] int32 (v2 format)
        # Remove the original qweight key to avoid "unexpected key" warnings
        del state_dict[qweight_ckpt]


@TransformRegistry.register("quantize_finegrained_fp8_linear_from_config")
class FineGrainedFP8LinearQuantization(Quantization):
    """Quantization transform for FineGrainedFP8 (block-wise FP8) models.

    This transform replaces linear ops with the FineGrainedFP8 quantized op.
    The FineGrained FP8 format uses per-block weight scales (weight_scale_inv) and
    dynamic input quantization.

    Config format (from HF config.json):
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"]
        }
    """

    algo_name = "fp8"

    def _post_init(self):
        self._weight_block_size = (128, 128)
        self._runtime_scale_name = "weight_scale_inv"
        self._default_scales_layout = None
        self._input_scale_fmt = ""

    def target_op(self):
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default

    def grouped_target_op(self):
        return torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)

    def scale_names(self) -> List[str]:
        return [self._runtime_scale_name]

    def default_scales(self, original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        if self._default_scales_layout is not None:
            return self._default_scales_layout.default_scales(original_weight_shape)

        N, K = original_weight_shape
        block_n, block_k = self._weight_block_size
        # Use ceil to handle dimensions smaller than or not divisible by block size
        # (e.g. after TP sharding or small projection weights).
        scale_shape = (math.ceil(N / block_n), math.ceil(K / block_k))
        return {self._runtime_scale_name: torch.ones(scale_shape, dtype=torch.float32)}

    def build_custom_args_for_linear(self, scales: Dict[str, Node]) -> Tuple:
        return ([], [scales[self.scale_names()[0]]], [], [])

    def _extract_grouped_linear_match(
        self,
        node: Node,
        checkpoint_layout: object,
        excluded: List[str],
    ) -> Tuple[Node, Node, object, Node, str, int] | None:
        if not is_op(node, torch.ops.auto_deploy.torch_grouped_linear):
            return None

        input_node, weight_view, bias_node = extract_op_args(node, "input", "weight", "bias")
        if not isinstance(input_node, Node) or not isinstance(weight_view, Node):
            return None
        if bias_node is not None and not isinstance(bias_node, Node):
            return None
        if not _is_view_or_reshape_node(weight_view):
            return None

        weight_base, weight_shape = _view_base_and_shape(weight_view)
        if not isinstance(weight_base, Node) or weight_base.op != "get_attr":
            return None
        if not isinstance(weight_base.target, str):
            return None
        weight_name = weight_base.target
        if not checkpoint_layout.is_weight_targeted(weight_name, excluded):
            return None

        input_shape = _shape_from_view_or_meta(input_node)
        if input_shape is None or len(input_shape) != 4 or len(weight_shape) != 3:
            return None
        num_groups = weight_shape[0]
        rank = weight_shape[1]
        if not _is_static_dim_value(num_groups) or not _is_static_dim_value(rank):
            return None
        if _is_static_dim_value(input_shape[2]) and input_shape[2] != num_groups:
            return None

        return node, input_node, bias_node, weight_base, weight_name, rank

    def _insert_grouped_quantized_linear(
        self,
        gm: GraphModule,
        source_node: Node,
        input_node: Node,
        bias_node: object,
        weight_node: Node,
        weight_name: str,
        rank: int,
        load_hook_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        original_weight = gm.get_parameter(weight_name)
        new_param = nn.Parameter(self.quantize_weight(original_weight), requires_grad=False)
        modname, _, attrname = weight_name.rpartition(".")
        submod = gm.get_submodule(modname)
        setattr(submod, attrname, new_param)
        weight_node.meta["val"] = new_param.detach()

        for scale_name, scale in self.default_scales(original_weight.shape).items():
            if scale_name in submod._buffers:
                submod._buffers[scale_name] = scale
            else:
                submod.register_buffer(scale_name, scale)

        gm._register_load_state_dict_pre_hook(
            partial(
                self.load_hook,
                weight_name=weight_name,
                **(load_hook_kwargs or {}),
            )
        )

        scale_name = self.scale_names()[0]
        scale_target = f"{modname}.{scale_name}" if modname else scale_name
        layer_type = _extract_layer_type_hint(source_node, _extract_layer_type_hint(input_node))

        with gm.graph.inserting_before(source_node):
            scale_node = gm.graph.create_node("get_attr", scale_target)
            scale_node.meta["val"] = getattr(submod, scale_name).detach()
            grouped_node = gm.graph.call_function(
                self.grouped_target_op(),
                args=(input_node, weight_node, bias_node, [], [scale_node], [], []),
                kwargs={
                    "tp_mode": "colwise",
                    "output_sizes": None,
                    "tp_min_local_shape": rank,
                    "layer_type": layer_type,
                    "input_scale_fmt": self._input_scale_fmt,
                },
            )

        source_node.replace_all_uses_with(grouped_node)
        gm.graph.erase_node(source_node)

    @staticmethod
    def _get_finegrained_fp8_layout(qcfg: Dict):
        checkpoint_layout = qcfg.get("checkpoint_layout")
        if checkpoint_layout is None:
            return None
        return getattr(checkpoint_layout, "finegrained_fp8", None)

    @staticmethod
    def _normalize_weight_block_size(block_size: object) -> Tuple[int, int]:
        if (
            isinstance(block_size, Sequence)
            and not isinstance(block_size, str)
            and len(block_size) == 2
        ):
            return int(block_size[0]), int(block_size[1])
        raise ValueError(f"FineGrained FP8 weight_block_size must have two dims, got {block_size}")

    @staticmethod
    def _resolve_weight_block_size(qcfg: Dict, checkpoint_layout: object) -> Tuple[int, int]:
        if checkpoint_layout is not None:
            return FineGrainedFP8LinearQuantization._normalize_weight_block_size(
                checkpoint_layout.weight_block_size
            )
        if qcfg.get("weight_block_size") is not None:
            return FineGrainedFP8LinearQuantization._normalize_weight_block_size(
                qcfg["weight_block_size"]
            )
        return 128, 128

    @staticmethod
    def _resolve_input_scale_fmt(qcfg: Dict, checkpoint_layout: object) -> str:
        scale_fmt = getattr(checkpoint_layout, "scale_fmt", None)
        if scale_fmt is None:
            scale_fmt = qcfg.get("scale_fmt")
        if scale_fmt is None:
            return ""
        return str(scale_fmt).lower()

    @staticmethod
    def _add_prefix(prefix: str, name: str) -> str:
        if prefix and not name.startswith(prefix):
            return prefix + name
        return name

    def load_hook(self, state_dict, prefix, *args, weight_name: str, checkpoint_layout=None):
        """Load hook to handle FineGrainedFP8 checkpoint format.

        FineGrained FP8 checkpoints store:
        - weight: float8_e4m3fn tensor
        - weight_scale_inv: per-block scale tensor
        """
        prefix = prefix or ""
        weight_key = prefix + weight_name
        if weight_key not in state_dict and weight_name in state_dict:
            weight_key = weight_name
        if weight_key not in state_dict:
            return

        weight = state_dict[weight_key]
        if weight.dtype != torch.float8_e4m3fn:
            return

        active_prefix = prefix if weight_key == prefix + weight_name else ""
        mod_prefix = weight_key.rsplit(".", 1)[0]
        if checkpoint_layout is None:
            scale_inv_name = weight_key + "_scale_inv"
            if scale_inv_name in state_dict:
                # Rename to match our buffer name.
                state_dict[mod_prefix + ".weight_scale_inv"] = state_dict[scale_inv_name]
            return

        layout_weight_name = weight_name
        if weight_key != weight_name and not checkpoint_layout.is_weight_targeted(
            layout_weight_name
        ):
            layout_weight_name = weight_key

        source_name = checkpoint_layout.scale_name_for_weight(layout_weight_name)
        target_name = checkpoint_layout.runtime_scale_name_for_weight(layout_weight_name)
        scale_key = self._add_prefix(active_prefix, source_name)
        if scale_key not in state_dict and source_name in state_dict:
            scale_key = source_name
        if scale_key not in state_dict:
            return

        target_key = self._add_prefix(active_prefix, target_name)
        scale = state_dict[scale_key]
        checkpoint_layout.validate_scale_shape(
            weight_key,
            weight.shape,
            scale_key,
            scale.shape,
        )
        decoded_scale = checkpoint_layout.decode_scale(scale)
        if scale_key != target_key:
            state_dict.pop(scale_key)
        state_dict[target_key] = decoded_scale

    # NOTE: post_load_hook intentionally inherited as None from the base
    # `Quantization`. UE8M0 conversion + TMA col-major layout for DeepGEMM is
    # done atomically inside `_dispatch_trtllm_finegrained_fp8_to_deepgemm`
    # (transform/library/fuse_quant.py) for every node we actually swap to
    # `trtllm_fp8_deepgemm`. Doing it there guarantees the graph never carries
    # a UE8M0 scale paired with a raw-FP32-scale op, which previously caused
    # NaN whenever dispatch failed to swap (deepgemm op missing in build,
    # fuse_finegrained_fp8_linear disabled, partial pipeline, etc.).

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

        if is_mixed_precision_config(qcfg):
            ad_logger.warning(
                "FineGrained FP8 quantization does not support MIXED_PRECISION checkpoints, "
                "skipping."
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        checkpoint_layout = self._get_finegrained_fp8_layout(qcfg)
        quant_method = str(qcfg.get("quant_method", "")).lower()
        if quant_method != self.algo_name and checkpoint_layout is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        if qcfg.get("weight_block_size") is None and checkpoint_layout is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        self._weight_block_size = self._resolve_weight_block_size(qcfg, checkpoint_layout)
        self._default_scales_layout = checkpoint_layout
        self._runtime_scale_name = (
            checkpoint_layout.runtime_scale_name
            if checkpoint_layout is not None
            else "weight_scale_inv"
        )
        self._input_scale_fmt = self._resolve_input_scale_fmt(qcfg, checkpoint_layout)
        if checkpoint_layout is not None:
            excluded = list(
                dict.fromkeys(
                    [
                        *(qcfg.get("exclude_modules") or []),
                        *(qcfg.get("modules_to_not_convert") or []),
                    ]
                )
            )
        else:
            excluded = qcfg.get("modules_to_not_convert") or []

        cnt = 0
        with WeightBiasInfoCache():
            for n in gm.graph.nodes:
                if not is_linear_op(n):
                    continue
                load_hook_kwargs = None
                if checkpoint_layout is not None:
                    weight_name = extract_weight_name(n)
                    if not isinstance(weight_name, str):
                        continue
                    if not checkpoint_layout.is_weight_targeted(weight_name, excluded):
                        continue
                    load_hook_kwargs = {"checkpoint_layout": checkpoint_layout}
                else:
                    if should_skip_quantization(n, excluded):
                        continue
                self._insert_quantized_linear(
                    gm,
                    n,
                    is_quantized_graph=False,
                    load_hook_kwargs=load_hook_kwargs,
                    custom_kwargs={"input_scale_fmt": self._input_scale_fmt},
                )
                cnt += 1

            if checkpoint_layout is not None:
                load_hook_kwargs = {"checkpoint_layout": checkpoint_layout}
                for n in list(gm.graph.nodes):
                    grouped_match = self._extract_grouped_linear_match(
                        n, checkpoint_layout, excluded
                    )
                    if grouped_match is None:
                        continue
                    source_node, input_node, bias_node, weight_node, weight_name, rank = (
                        grouped_match
                    )
                    self._insert_grouped_quantized_linear(
                        gm,
                        source_node,
                        input_node,
                        bias_node,
                        weight_node,
                        weight_name,
                        rank,
                        load_hook_kwargs=load_hook_kwargs,
                    )
                    cnt += 1

        if cnt:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=(cnt == 0)
        )
