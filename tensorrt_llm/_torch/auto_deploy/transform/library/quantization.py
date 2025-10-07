from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...custom_ops.quant import (
    FP4_GLOBAL_SCALE_MAX,
    FP8_MAX,
    TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
    is_column_major,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import (
    extract_param_names_from_lin_node,
    get_quantization_params_from_linear_node,
    is_bmm_op,
    is_linear_op,
)
from ...utils.quantization_utils import (
    fp4_global_scale,
    fp8_scale,
    get_quantization_from_linear_node,
    is_quantized_graph,
    is_quantized_op,
    remove_output_quantizers,
    should_skip_quantization,
)
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

try:
    from .....quantization.utils.fp4_utils import float4_sf_dtype
except ImportError:
    float4_sf_dtype = None


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
        qcfg = factory.get_quant_config()
        if not qcfg or qcfg.get("quant_algo", "").upper() != self.algo_name:
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
            self._insert_quantized_linear(gm, n, is_quantized_graph=False)
            cnt += 1

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=True
        )

    def _insert_quantized_linear(
        self,
        gm: GraphModule,
        node: Node,
        is_quantized_graph: bool = False,
    ):
        """Replaces the matmul node with a new custom quantized linear node.

        The state_dict is also updated to contain the sharded weights.
        """
        param_name, _ = extract_param_names_from_lin_node(node)
        original_weight = gm.get_parameter(param_name)
        new_param = nn.Parameter(self.quantize_weight(original_weight), requires_grad=False)
        modname, _, attrname = param_name.rpartition(".")

        submod = gm.get_submodule(modname)
        setattr(submod, attrname, new_param)

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
            gm._register_load_state_dict_pre_hook(
                partial(
                    self.convert_amax_hook,
                    scale_name=modname + "." + input_scale_name,
                    amax_name=input_params.amax.target,
                )
            )
            # Note: canonicalize_graph() will remove input/weight/output quantizer

        for scale_name, scale in self.default_scales(original_weight.shape).items():
            submod.register_buffer(scale_name, scale)

        gm._register_load_state_dict_pre_hook(partial(self.load_hook, weight_name=param_name))

        with gm.graph.inserting_before(node):
            scales = {}
            for scale_name in self.scale_names():
                scales[scale_name] = gm.graph.create_node("get_attr", modname + "." + scale_name)

        custom_args = self.build_custom_args_for_linear(scales)

        node.target = self.target_op()
        node.args = (*node.args, *custom_args)

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
            gm._register_load_state_dict_pre_hook(partial(self.load_hook, weight_name=param_name))
            if self.post_load_hook:
                gm.register_load_state_dict_post_hook(
                    partial(self.post_load_hook, weight_name=param_name)
                )

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
        if weight_name in state_dict:
            weight = state_dict[weight_name]
            if weight.dtype != torch.float8_e4m3fn:
                scale = fp8_scale(state_dict[weight_name])
                state_dict[weight_name] = (state_dict[weight_name] / scale).to(torch.float8_e4m3fn)
                state_dict[weight_name + "_scale"] = scale

    def convert_amax_hook(self, state_dict, prefix, *args, scale_name: str, amax_name: str):
        """Convert amax from modelopt quantized graph to scales."""
        if amax_name in state_dict:
            amax = state_dict[amax_name]
            scale = amax / FP8_MAX
            state_dict[scale_name] = scale


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

    def default_scales(self, original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
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

    def build_custom_args_for_linear(self, scales: Dict[str, Node]) -> Tuple:
        # weight_scale list is (cutlass_vec, alpha)
        return ([scales["input_scale"]], [scales["weight_scale"], scales["alpha"]], [], [])

    def load_hook(self, state_dict, prefix, *args, weight_name):
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
                        torch.ops.trtllm.block_scale_interleave(
                            weight_scale.view(torch.uint8).cpu().contiguous()
                        )
                        .reshape(ori_shape)
                        .view(float4_sf_dtype)
                        .reshape(-1)
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
        if not qcfg or qcfg.get("quant_algo", "").upper() != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded = qcfg.get("exclude_modules", [])
        cnt = 0
        for n in gm.graph.nodes:
            if not is_bmm_op(n):
                continue
            if should_skip_quantization(n, excluded):
                continue
            if self._insert_quantized_bmm(gm, n, is_quantized_graph=False):
                cnt += 1

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=True
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
            skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=True
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

        remove_output_quantizers(gm)
        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=False, has_valid_shapes=True
        )
