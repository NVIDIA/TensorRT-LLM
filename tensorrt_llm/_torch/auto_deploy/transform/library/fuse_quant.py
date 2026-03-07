import math
from typing import Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


# with bias=None
def _fp8_ref_pattern_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
        x,
        w_fp8,
        None,
        input_scale=[input_scale],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


# with bias!=None
def _fp8_ref_pattern_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
        x,
        w_fp8,
        bias,
        input_scale=[input_scale],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


# NVFP4: with bias=None
def _fp4_ref_pattern_1(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        None,
        input_scale=[input_scale],
        weight_scale=[weight_scale, weight_scale_2],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_1(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        w_fp4,
        bias=None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=weight_scale_2,
    )


# with bias!=None
def _fp4_ref_pattern_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        bias,
        input_scale=[input_scale],
        weight_scale=[weight_scale, weight_scale_2],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        w_fp4,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=weight_scale_2,
    )


def _register_quant_fp8_linear_patterns(patterns: ADPatternMatcherPass, op) -> None:
    """
    Register FP8 linear patterns with robust dummy args and minimal ignores.
    """

    # Define replacement functions that use the provided op
    def _fp8_ref_repl_1(
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        return op(
            x,
            w_fp8,
            None,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )

    def _fp8_ref_repl_2(
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        return op(
            x,
            w_fp8,
            bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )

    # FP8 dummy tensors
    x_fp8 = torch.randn(3, 16, device="meta", dtype=torch.float16)
    w_fp8 = torch.randn(32, 16, device="meta", dtype=torch.float16)
    bias32 = torch.randn(32, device="meta", dtype=torch.float32)
    one = torch.tensor(1.0, device="meta", dtype=torch.float32)

    # no-bias variant
    dummy_args_fp8 = [
        x_fp8,
        w_fp8,
        one,
        torch.tensor(0.5, device="meta", dtype=torch.float32),
    ]
    register_ad_pattern(
        search_fn=_fp8_ref_pattern_1,
        replace_fn=_fp8_ref_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fp8,
    )

    # bias variant
    dummy_args_fp8_2 = [
        x_fp8,
        w_fp8,
        bias32,
        one,
        torch.tensor(0.5, device="meta", dtype=torch.float32),
    ]
    register_ad_pattern(
        search_fn=_fp8_ref_pattern_2,
        replace_fn=_fp8_ref_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fp8_2,
    )


def _register_quant_fp4_linear_patterns(patterns: ADPatternMatcherPass) -> None:
    """
    Register FP4 linear patterns with robust dummy args and minimal ignores.
    """
    # FP4 shape params
    N = 32
    K_packed = 32  # weight is packed by 2 FP4 per byte
    K_eff = 2 * K_packed

    # FP4 dummy tensors
    x_fp4 = torch.randn(3, K_eff, device="meta", dtype=torch.float16)
    w_fp4 = torch.randint(0, 255, (N, K_packed), device="meta", dtype=torch.uint8)

    s_in2 = torch.tensor(0.01, device="meta", dtype=torch.float32)
    ws2 = torch.tensor(1.2345, device="meta", dtype=torch.float32)

    cutlass_len = N * (K_eff // 16)  # 32 * (64/16) = 128
    cutlass_vec = torch.randint(0, 255, (cutlass_len,), device="meta", dtype=torch.uint8)

    # no-bias variant
    dummy_args_fp4_1 = [
        x_fp4,
        w_fp4,
        s_in2,
        cutlass_vec,
        ws2,
    ]
    register_ad_pattern(
        search_fn=_fp4_ref_pattern_1,
        replace_fn=_fp4_ref_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fp4_1,
    )

    # bias variant
    dummy_args_fp4_2 = [
        x_fp4,
        w_fp4,
        torch.randn(N, device="meta", dtype=torch.float16),  # bias
        s_in2,
        cutlass_vec,
        ws2,
    ]
    register_ad_pattern(
        search_fn=_fp4_ref_pattern_2,
        replace_fn=_fp4_ref_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fp4_2,
    )


class FuseFP8LinearConfig(TransformConfig):
    """Configuration for FP8 linear fusion transform."""

    backend: str = Field(
        default="torch",
        description="Backend to use for FP8 linear computation (default: 'torch').",
    )


@TransformRegistry.register("fuse_fp8_linear")
class FuseFP8Linear(BaseTransform):
    """Matches and replaces FP8 fake quantized linear ops with fused torch backend ops."""

    config: FuseFP8LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseFP8LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() not in ["torch", "trtllm"]:
            raise ValueError(f"Unsupported FP8 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        op = (
            torch.ops.auto_deploy.trtllm_quant_fp8_linear
            if self.config.backend.lower() == "trtllm"
            else torch.ops.auto_deploy.torch_quant_fp8_linear
        )

        _register_quant_fp8_linear_patterns(patterns, op)
        cnt = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info


def _collect_nvfp4_scale_keys(gm: GraphModule):
    """Collect (input_scale, weight_scale, weight_scale_2) buffer keys from fused nvfp4 nodes."""
    scale_keys = []
    for node in gm.graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear):
            continue
        scale_map = {}
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            t = inp.target
            if t.endswith(".weight_scale_2"):
                scale_map["ws2"] = t
            elif t.endswith(".weight_scale"):
                scale_map["ws"] = t
            elif t.endswith(".input_scale"):
                scale_map["is"] = t
        if len(scale_map) == 3:
            scale_keys.append((scale_map["is"], scale_map["ws"], scale_map["ws2"]))
    return scale_keys


def _process_nvfp4_scales_inplace(gm: GraphModule, scale_keys):
    """Pre-process raw NVFP4 scale buffers in-place into the format expected by the kernel.

    Converts:
      - weight_scale_2 * input_scale -> alpha (clamped)
      - input_scale -> 1 / input_scale (clamped then inverted)
      - weight_scale -> CUTLASS-swizzled flat uint8 via block_scale_interleave
    """
    try:
        from .....quantization.utils.fp4_utils import float4_sf_dtype
    except ImportError:
        float4_sf_dtype = None

    if not float4_sf_dtype:
        return

    for is_key, ws_key, ws2_key in scale_keys:
        is_mod_path, _, is_attr = is_key.rpartition(".")
        ws_mod_path, _, ws_attr = ws_key.rpartition(".")
        ws2_mod_path, _, ws2_attr = ws2_key.rpartition(".")

        is_submod = gm.get_submodule(is_mod_path)
        ws_submod = gm.get_submodule(ws_mod_path)
        ws2_submod = gm.get_submodule(ws2_mod_path)

        raw_is = getattr(is_submod, is_attr)
        raw_ws = getattr(ws_submod, ws_attr)
        raw_ws2 = getattr(ws2_submod, ws2_attr)
        device = raw_ws.device

        alpha = torch.clamp(raw_ws2 * raw_is, min=1e-30)

        inv_input_scale = 1 / torch.clamp(raw_is, min=1e-30)

        weight_scale = raw_ws.view(float4_sf_dtype)
        weight_scale_swizzled = torch.ops.trtllm.block_scale_interleave(
            weight_scale.view(torch.uint8).cpu().contiguous()
        ).view(float4_sf_dtype)

        m, n = weight_scale.shape
        padded_m = math.ceil(m / 128) * 128
        padded_n = math.ceil(n / 4) * 4
        swizzled_shape = (padded_m, padded_n)

        new_ws = (
            weight_scale_swizzled.reshape(swizzled_shape).view(torch.uint8).reshape(-1).to(device)
        )

        is_submod.register_buffer(is_attr, inv_input_scale)
        ws_submod.register_buffer(ws_attr, new_ws)
        ws2_submod.register_buffer(ws2_attr, alpha)


class FuseNVFP4LinearConfig(TransformConfig):
    """Configuration for NVFP4 linear fusion transform."""

    backend: str = Field(
        default="trtllm",
        description="Backend to use for NVFP4 linear computation (default: 'trtllm').",
    )


@TransformRegistry.register("fuse_nvfp4_linear")
class FuseNVFP4Linear(BaseTransform):
    """Matches and replaces NVFP4 fake quantized linear ops with fused TensorRT-LLM ops."""

    config: FuseNVFP4LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNVFP4LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() != "trtllm":
            raise ValueError(f"Unsupported NVFP4 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        _register_quant_fp4_linear_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        if cnt > 0:
            scale_keys = _collect_nvfp4_scale_keys(gm)
            if scale_keys:
                _process_nvfp4_scales_inplace(gm, scale_keys)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
