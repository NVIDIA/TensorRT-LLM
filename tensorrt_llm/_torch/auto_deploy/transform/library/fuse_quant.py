from typing import Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
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


def _fp8_ref_repl_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_fp8_linear(
        x,
        w_fp8,
        None,
        input_scale=input_scale,
        weight_scale=weight_scale,
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


def _fp8_ref_repl_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_fp8_linear(
        x,
        w_fp8,
        bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
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


def _register_quant_fp8_linear_patterns(patterns: ADPatternMatcherPass) -> None:
    """
    Register FP8 linear patterns with robust dummy args and minimal ignores.
    """
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
        if self.config.backend.lower() != "torch":
            raise ValueError(f"Unsupported FP8 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        _register_quant_fp8_linear_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=False,
            has_valid_shapes=False,
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
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info
