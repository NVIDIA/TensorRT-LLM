# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static quant recipes must be refused against unquantized checkpoints.

With ``dynamic_weight_quant=False`` the loader expects pre-quantized weights
plus their scale tensors from the checkpoint. Loading a high-precision
checkpoint through a static recipe silently corrupts the weights (scales stay
at their default or uninitialized values), so the loader must raise instead.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# Pure-CPU test (stub Linear, no device): runs in the CPU lane (l0_cpu.yml),
# which selects with `-m cpu_only`; GPU stages deselect it via `not cpu_only`.
pytestmark = pytest.mark.cpu_only


class _StubLinear:
    """Records load_weights calls; quant_algo resolves via the loader's global config."""

    quant_config = None

    def __init__(self):
        self.loaded = None

    def load_weights(self, weight_dicts):
        self.loaded = weight_dicts


def _make_loader(quant_algo, dynamic_weight_quant=False, exclude_modules=None):
    model_config = DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=quant_algo, exclude_modules=exclude_modules),
        dynamic_weight_quant=dynamic_weight_quant,
    )
    return DynamicLinearWeightLoader(model_config)


def _bf16_weights():
    return {"weight": torch.zeros(8, 16, dtype=torch.bfloat16)}


class TestStaticQuantGuard:
    @pytest.mark.parametrize(
        "quant_algo",
        [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4],
    )
    def test_static_recipe_vs_unquantized_checkpoint_raises(self, quant_algo):
        loader = _make_loader(quant_algo)
        module = _StubLinear()
        with pytest.raises(ValueError, match="appears to be unquantized"):
            loader.load_linear_weights(module, "blocks.0.attn1.to_q", [_bf16_weights()])
        assert module.loaded is None

    @pytest.mark.parametrize(
        "quant_algo",
        [QuantAlgo.W4A16_AWQ, QuantAlgo.W4A8_AWQ, QuantAlgo.W8A8_SQ_PER_CHANNEL],
    )
    def test_unregistered_static_algo_fails_closed(self, quant_algo):
        """Algos accepted by config parsing but absent from _STATIC_SCALE_KEYS
        (their VisualGen checkpoint layout is unverified) must also refuse a
        high-precision weight instead of silently skipping the check."""
        loader = _make_loader(quant_algo)
        module = _StubLinear()
        with pytest.raises(ValueError, match="fails closed for unverified algos"):
            loader.load_linear_weights(module, "blocks.0.attn1.to_q", [_bf16_weights()])
        assert module.loaded is None

    def test_static_fp8_checkpoint_with_scales_loads(self):
        loader = _make_loader(QuantAlgo.FP8)
        module = _StubLinear()
        weight_dict = {
            "weight": torch.zeros(8, 16, dtype=torch.float8_e4m3fn),
            "weight_scale": torch.ones(1, dtype=torch.float32),
            "input_scale": torch.ones(1, dtype=torch.float32),
        }
        loader.load_linear_weights(module, "blocks.0.attn1.to_q", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_static_nvfp4_checkpoint_with_scales_loads(self):
        loader = _make_loader(QuantAlgo.NVFP4)
        module = _StubLinear()
        weight_dict = {
            "weight": torch.zeros(8, 8, dtype=torch.uint8),
            "weight_scale": torch.zeros(8, 1, dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.ones(1, dtype=torch.float32),
        }
        loader.load_linear_weights(module, "blocks.0.attn1.to_q", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_excluded_module_keeps_high_precision_weights(self):
        loader = _make_loader(QuantAlgo.FP8, exclude_modules=["proj_out"])
        module = _StubLinear()
        weight_dict = _bf16_weights()
        loader.load_linear_weights(module, "proj_out", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_unquantized_recipe_is_unaffected(self):
        loader = _make_loader(None)
        module = _StubLinear()
        weight_dict = _bf16_weights()
        loader.load_linear_weights(module, "blocks.0.attn1.to_q", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_dynamic_recipe_skips_the_guard(self):
        loader = _make_loader(QuantAlgo.FP8, dynamic_weight_quant=True)
        # The guard is a no-op for dynamic recipes; quantization happens later
        # in _maybe_dynamic_quantize (GPU path, not exercised here).
        loader._check_static_quant_scales(_bf16_weights(), QuantAlgo.FP8, "blocks.0.attn1.to_q")
