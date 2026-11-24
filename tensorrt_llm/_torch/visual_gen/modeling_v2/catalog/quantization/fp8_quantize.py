# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.visual_gen.runtime.tuning import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)


def _reference_scale(amax, dtype):
    scale = (amax / amax.new_tensor(448.0)).clamp_min(1.0 / (448.0 * 512))
    return scale.to(dtype)


def _reference_codes(x, scale):
    # The aligned BF16 kernel rounds the FP32 reciprocal before multiplying.
    if x.dtype == torch.bfloat16 and x.data_ptr() % 16 == 0:
        scaled = x.float() * scale.float().reciprocal()
    else:
        scaled = x.float() / scale.float()
    return scaled.clamp(-448, 448).to(torch.float8_e4m3fn)


class NativeFp8PerTensorQuantize(OpWrapper):
    """Quantize contiguous activations to E4M3, returning an input-dtype scale of rank x.ndim."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell("Four-dimensional attention input", dict(shape=(1, 3, 4, 16))),)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)

    def reference(self, x):
        scale = _reference_scale(x.float().abs().amax(), x.dtype)
        return _reference_codes(x, scale), scale.reshape([1] * x.ndim)

    def is_valid(self, x):
        assert x.is_contiguous() and x.ndim >= 2 and x.numel() > 0
        assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)


class NativeStaticFp8PerTensorQuantize(OpWrapper):
    """Quantize to E4M3 using the supplied scalar FP32 dequantization scale."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell("Straddling the saturated FP8 range", dict(shape=(3, 32))),)

    def __call__(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        values, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)
        return values

    def reference(self, x, scale):
        return _reference_codes(x, scale)

    def is_valid(self, x, scale):
        native_fp8_per_tensor_quantize.is_valid(x)
        assert scale.dtype == torch.float32 and scale.numel() == 1 and scale.device == x.device


native_fp8_per_tensor_quantize = NativeFp8PerTensorQuantize()
native_static_fp8_per_tensor_quantize = NativeStaticFp8PerTensorQuantize()


@lru_cache(maxsize=1)
def _te_quantizer():
    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
    except ImportError:
        return None
    return Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda")


class QuantizeE4M3PerTensorRunner(TunableRunner):
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0, -2, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
            ),
        ),
        tune_max_num_tokens=8192,
    )

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return ["trtllm", "te"] if _te_quantizer() is not None else ["trtllm"]

    def forward(self, inputs, tactic="trtllm"):
        (x,) = inputs
        if tactic == "te":
            quantizer = _te_quantizer()
            if quantizer is None:
                raise RuntimeError("Transformer Engine FP8 quantization is unavailable.")
            fp8 = quantizer.quantize(x)
            return fp8._data.view(torch.float8_e4m3fn), fp8._scale_inv.to(x.dtype).reshape(
                [1] * x.ndim
            )
        return native_fp8_per_tensor_quantize(x)


@torch.library.custom_op(
    "trtllm::visual_gen_v2_quantize_e4m3_per_tensor", mutates_args=(), device_types="cuda"
)
def _tuned_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    runner = QuantizeE4M3PerTensorRunner()
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::quantize_e4m3_per_tensor", [runner], runner.tuning_config, [x]
    )
    return runner(inputs=[x], tactic=tactic)


@_tuned_quantize.register_fake
def _(x):
    return x.new_empty(x.shape, dtype=torch.float8_e4m3fn), x.new_empty([1] * x.ndim)


class TunedFp8PerTensorQuantize(NativeFp8PerTensorQuantize):
    """Select native or Transformer Engine quantization for the input shape."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _tuned_quantize(x)


quantize_e4m3_per_tensor = TunedFp8PerTensorQuantize()
