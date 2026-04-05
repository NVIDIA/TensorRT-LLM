import math
from typing import Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
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

    # FP4 dummy tensors (weight_scale is FP8 per-block 2D)
    x_fp4 = torch.randn(3, K_eff, device="meta", dtype=torch.float16)
    w_fp4 = torch.randint(0, 255, (N, K_packed), device="meta", dtype=torch.uint8)

    s_in2 = torch.tensor(0.01, device="meta", dtype=torch.float32)
    ws2 = torch.tensor(1.2345, device="meta", dtype=torch.float32)

    # Per-block FP8 weight scale: 2D [N, K_eff//16]
    weight_scale_fp8 = torch.empty((N, K_eff // 16), device="meta", dtype=torch.float8_e4m3fn)

    # no-bias variant
    dummy_args_fp4_1 = [
        x_fp4,
        w_fp4,
        s_in2,
        weight_scale_fp8,
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
        weight_scale_fp8,
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


def _swizzle_nvfp4_scale(
    raw_ws: torch.Tensor, float4_sf_dtype, flatten: bool = True
) -> torch.Tensor:
    """Swizzle a 2D FP8 per-block weight scale into CUTLASS-ready uint8.

    Args:
        raw_ws: Per-block weight scale tensor, dtype=float8_e4m3fn, shape (M, N/16).
        float4_sf_dtype: The float4 scale factor dtype (from fp4_utils).
        flatten: If True (default), return flat 1D uint8. If False, return 2D uint8
            [padded_M, padded_N] suitable for MoE stacking into 3D.

    Returns:
        uint8 tensor in CUTLASS-swizzled layout, on the same device as raw_ws.
        Shape is flat 1D if flatten=True, else 2D [padded_M, padded_N].
    """
    device = raw_ws.device
    weight_scale = raw_ws.view(float4_sf_dtype)
    weight_scale_swizzled = torch.ops.trtllm.block_scale_interleave(
        weight_scale.view(torch.uint8).cpu().contiguous()
    ).view(float4_sf_dtype)
    m, n = weight_scale.shape
    padded_m = math.ceil(m / 128) * 128
    padded_n = math.ceil(n / 4) * 4
    swizzled_shape = (padded_m, padded_n)
    result = weight_scale_swizzled.reshape(swizzled_shape).view(torch.uint8).to(device)
    return result.reshape(-1) if flatten else result


def _collect_nvfp4_scale_keys(gm: GraphModule):
    """Collect unique (input_scale, weight_scale, weight_scale_2) buffer keys from fused nvfp4 nodes.

    Deduplicates by target path so the same quantized module referenced from multiple nodes
    is only processed once by _process_nvfp4_scales_inplace (which mutates buffers).
    """
    scale_keys = []
    seen = set()
    for node in gm.graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear):
            continue
        scale_map = {}
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            t = inp.target
            attr = t.rsplit(".", 1)[-1]
            if attr == "weight_scale_2":
                scale_map["ws2"] = t
            elif attr == "weight_scale":
                scale_map["ws"] = t
            elif attr == "input_scale":
                scale_map["is"] = t
        if len(scale_map) == 3:
            key = (scale_map["is"], scale_map["ws"], scale_map["ws2"])
            if key not in seen:
                seen.add(key)
                scale_keys.append(key)
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
        ad_logger.warning(
            "Could not import float4_sf_dtype from fp4_utils; skipping NVFP4 scale pre-processing."
        )
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

        alpha = torch.clamp(raw_ws2 * raw_is, min=1e-30)

        inv_input_scale = 1 / torch.clamp(raw_is, min=1e-30)

        new_ws = _swizzle_nvfp4_scale(raw_ws, float4_sf_dtype)

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


# ============================================================================
# FineGrained FP8 Linear Patterns (for MiniMax M2, DeepSeek, etc.)
# ============================================================================


# FineGrained FP8: with bias=None
def _finegrained_fp8_pattern_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x,
        w_fp8,
        None,
        input_scale=[],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


def _finegrained_fp8_repl_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.trtllm_finegrained_fp8_linear(
        x,
        w_fp8,
        None,
        weight_scale,
    )


# FineGrained FP8: with bias!=None
def _finegrained_fp8_pattern_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x,
        w_fp8,
        bias,
        input_scale=[],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


def _finegrained_fp8_repl_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.trtllm_finegrained_fp8_linear(
        x,
        w_fp8,
        bias,
        weight_scale,
    )


def _register_finegrained_fp8_linear_patterns(patterns: ADPatternMatcherPass) -> None:
    """
    Register FineGrained FP8 linear patterns.

    FineGrained FP8 uses block-wise weight quantization with per-block scales.
    The replacement uses TRT-LLM's optimized fp8_block_scaling_gemm kernel.
    """
    # FineGrained FP8 dummy tensors
    # weight shape: [N, K], weight_scale shape: [N/128, K/128]
    N, K = 256, 256  # Must be multiples of 128 for block quantization
    x_fg_fp8 = torch.randn(3, K, device="meta", dtype=torch.bfloat16)
    w_fg_fp8 = torch.randn(N, K, device="meta", dtype=torch.float8_e4m3fn)
    bias_fg = torch.randn(N, device="meta", dtype=torch.bfloat16)
    # Per-block weight scale: [N/128, K/128]
    weight_scale_fg = torch.randn(N // 128, K // 128, device="meta", dtype=torch.float32)

    # no-bias variant
    dummy_args_fg_fp8_1 = [
        x_fg_fp8,
        w_fg_fp8,
        weight_scale_fg,
    ]
    register_ad_pattern(
        search_fn=_finegrained_fp8_pattern_1,
        replace_fn=_finegrained_fp8_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fg_fp8_1,
    )

    # bias variant
    dummy_args_fg_fp8_2 = [
        x_fg_fp8,
        w_fg_fp8,
        bias_fg,
        weight_scale_fg,
    ]
    register_ad_pattern(
        search_fn=_finegrained_fp8_pattern_2,
        replace_fn=_finegrained_fp8_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fg_fp8_2,
    )


class FuseFineGrainedFP8LinearConfig(TransformConfig):
    """Configuration for FineGrained FP8 linear fusion transform."""

    backend: str = Field(
        default="trtllm",
        description="Backend to use for FineGrained FP8 linear computation (default: 'trtllm').",
    )


@TransformRegistry.register("fuse_finegrained_fp8_linear")
class FuseFineGrainedFP8Linear(BaseTransform):
    """Matches and replaces FineGrained FP8 fake quantized linear ops with TRT-LLM ops.

    This transform replaces torch_fake_quant_finegrained_fp8_linear (which uses HuggingFace's
    triton kernel) with trtllm_finegrained_fp8_linear (which uses TRT-LLM's optimized
    fp8_block_scaling_gemm kernel).

    Used for models like MiniMax M2 and DeepSeek that use HuggingFace's FineGrained FP8
    quantization format with 128x128 block sizes.
    """

    config: FuseFineGrainedFP8LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseFineGrainedFP8LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() != "trtllm":
            raise ValueError(f"Unsupported FineGrained FP8 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        _register_finegrained_fp8_linear_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
