from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

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
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        None,
        input_scale=[input_scale],
        weight_scale=[weight_scale, alpha],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_1(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        w_fp4,
        bias=None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=alpha,
    )


# with bias!=None
def _fp4_ref_pattern_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        bias,
        input_scale=[input_scale],
        weight_scale=[weight_scale, alpha],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        w_fp4,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=alpha,
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
    alpha = torch.tensor(1.2345, device="meta", dtype=torch.float32)

    cutlass_len = N * (K_eff // 16)  # 32 * (64/16) = 128
    cutlass_vec = torch.randint(0, 255, (cutlass_len,), device="meta", dtype=torch.uint8)

    # no-bias variant
    dummy_args_fp4_1 = [
        x_fp4,
        w_fp4,
        s_in2,
        cutlass_vec,
        alpha,
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
        alpha,
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


# Positional index of weight_scale in trtllm_finegrained_fp8_linear signature:
#   (input, weight, bias, weight_scale, tp_mode=..., ...)
_TRTLLM_FG_FP8_WEIGHT_SCALE_ARG = 3


def _resolve_attr_tensor(gm: GraphModule, attr_node: Node) -> Optional[torch.Tensor]:
    """Resolve a get_attr node's target to the live tensor on `gm`, or None.

    The `weight_scale` arg may be either a registered buffer (common) or a
    parameter. We fall back to a plain getattr walk to remain tolerant of both.
    """
    if not isinstance(attr_node, Node) or attr_node.op != "get_attr":
        return None
    target = attr_node.target
    if not isinstance(target, str):
        return None
    try:
        return gm.get_buffer(target)
    except AttributeError:
        pass
    try:
        return gm.get_parameter(target)
    except AttributeError:
        pass
    obj = gm
    for name in target.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.Tensor) else None


def _dispatch_trtllm_finegrained_fp8_to_deepgemm(gm: GraphModule) -> int:
    """Compile-time dispatch: route UE8M0 weight_scale to the DeepGEMM op.

    After `fuse_finegrained_fp8_linear`'s pattern matcher has produced
    `trtllm_finegrained_fp8_linear` nodes, this pass prefers the dedicated
    `trtllm_fp8_deepgemm` op whenever the weight_scale buffer has been
    converted to UE8M0 packed int by FineGrainedFP8LinearQuantization's
    post_load_hook (which only fires on SM100f).

    Returns the number of rewritten nodes.
    """
    src_op = torch.ops.auto_deploy.trtllm_finegrained_fp8_linear
    dst_op = getattr(torch.ops.auto_deploy, "trtllm_fp8_deepgemm", None)
    if dst_op is None:
        return 0

    num_rewrites = 0
    for node in gm.graph.nodes:
        if not is_op(node, src_op):
            continue

        if len(node.args) <= _TRTLLM_FG_FP8_WEIGHT_SCALE_ARG:
            continue
        weight_scale_arg = node.args[_TRTLLM_FG_FP8_WEIGHT_SCALE_ARG]

        scale_tensor = _resolve_attr_tensor(gm, weight_scale_arg)
        if scale_tensor is None or scale_tensor.dtype != torch.int:
            continue

        # Signatures match positionally; swap the call target in place.
        node.target = dst_op.default
        num_rewrites += 1

    return num_rewrites


@TransformRegistry.register("fuse_finegrained_fp8_linear")
class FuseFineGrainedFP8Linear(BaseTransform):
    """Matches and replaces FineGrained FP8 fake quantized linear ops with TRT-LLM ops.

    Two-stage pipeline:
      1. Pattern matcher rewrites ``torch_fake_quant_finegrained_fp8_linear``
         (HuggingFace triton kernel) to ``trtllm_finegrained_fp8_linear``
         (TRT-LLM ``fp8_block_scaling_gemm`` with FP32 per-block scales).
      2. A compile-time dispatch pass further rewrites any nodes whose
         ``weight_scale`` buffer is UE8M0 packed int (produced by
         ``FineGrainedFP8LinearQuantization.post_load_hook`` on SM100f) to
         the dedicated ``trtllm_fp8_deepgemm`` op. Keeping the SM100f/UE8M0
         path in a separate op avoids per-call hardware / dtype branching
         inside the runtime op.

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

        # Compile-time dispatch to the UE8M0 fast-path op. Counts toward
        # num_matches so downstream graph invariants get re-checked.
        cnt += _dispatch_trtllm_finegrained_fp8_to_deepgemm(gm)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
