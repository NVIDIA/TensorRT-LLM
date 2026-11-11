"""Public blockwise FP8/FP6 policies; compiled kernels are checked on request."""

from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.policy import QuantizationPolicy

__all__ = ["build_fp6_policy", "build_fp8_policy"]


def build_fp8_policy() -> QuantizationPolicy:
    """Build a blockwise FP8 quantization policy."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.blockwise.unit import _kernels, build_policy  # noqa: PLC0415

    _kernels()
    return build_policy(fp6=False)


def build_fp6_policy() -> QuantizationPolicy:
    """Build a blockwise FP6 quantization policy."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.blockwise.unit import _kernels, build_policy  # noqa: PLC0415

    _kernels()
    return build_policy(fp6=True)
