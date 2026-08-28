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

    def test_unquantizable_module_keeps_high_precision_weights(self) -> None:
        """A module that was never built quantized must not be refused.

        ``quant_algo`` falls back to the global recipe for any module without
        its own ``quant_config``, and that includes modules which cannot be
        quantized: ``Embedding`` reaches this loader because it subclasses
        ``LMHead`` -> ``Linear``, but its ``__init__`` never exposes
        ``quant_config``, so its buffer stays high precision. ModelOpt does not
        list it in ``ignore`` either, since only ``Linear`` targets were ever
        candidates. Nothing was built quantized, so nothing can be corrupted.
        """
        loader = _make_loader(QuantAlgo.FP8)
        module = _StubLinear()
        module.weight = torch.zeros(8, 16, dtype=torch.bfloat16)
        weight_dict = _bf16_weights()
        loader.load_linear_weights(module, "language_model.embed_tokens", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_quantized_destination_still_refuses_unquantized_checkpoint(self) -> None:
        """The destination check must not weaken the guard it sits in front of.

        A module built for FP8 holds a float8 buffer, so a high-precision
        checkpoint weight is still the silent-corruption case and still raises.
        """
        loader = _make_loader(QuantAlgo.FP8)
        module = _StubLinear()
        module.weight = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="appears to be unquantized"):
            loader.load_linear_weights(module, "blocks.0.attn1.to_q", [_bf16_weights()])
        assert module.loaded is None

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
