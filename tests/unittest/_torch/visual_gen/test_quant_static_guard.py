# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static quant recipes must be refused against unquantized checkpoints.

With ``dynamic_weight_quant=False`` the loader expects pre-quantized weights
plus their scale tensors from the checkpoint. Loading a high-precision
checkpoint through a static recipe silently corrupts the weights (scales stay
at their default or uninitialized values), so the loader must raise instead.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod, W4A16NVFP4LinearMethod
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import DiffusionQuantConfig

# Pure-CPU test (stub Linear, no device): runs in the CPU lane (l0_cpu.yml),
# which selects with `-m cpu_only`; GPU stages deselect it via `not cpu_only`.
pytestmark = pytest.mark.cpu_only


class _StubLinear:
    """Records load_weights calls; quant_algo resolves via the loader's global config."""

    quant_config = None

    def __init__(self, quant_algo=None, force_dynamic_quantization=False):
        self.loaded = None
        self.quant_method = NVFP4LinearMethod() if quant_algo == QuantAlgo.NVFP4 else None
        self.force_dynamic_quantization = force_dynamic_quantization

    def load_weights(self, weight_dicts):
        self.loaded = weight_dicts


def _make_loader(
    quant_algo, dynamic_weight_quant=False, exclude_modules=None, force_dynamic_quantization=False
):
    model_config = DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=quant_algo, exclude_modules=exclude_modules),
        dynamic_weight_quant=dynamic_weight_quant,
        force_dynamic_quantization=force_dynamic_quantization,
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
        module = _StubLinear(QuantAlgo.NVFP4)
        weight_dict = {
            "weight": torch.zeros(8, 8, dtype=torch.uint8),
            "weight_scale": torch.zeros(8, 1, dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.ones(1, dtype=torch.float32),
            "input_scale": torch.ones(1, dtype=torch.float32),
        }
        loader.load_linear_weights(module, "blocks.0.attn1.to_q", [weight_dict])
        assert module.loaded == [weight_dict]

    def test_excluded_module_keeps_high_precision_weights(self):
        loader = _make_loader(QuantAlgo.FP8, exclude_modules=["proj_out"])
        module = _StubLinear()
        weight_dict = _bf16_weights()
        loader.load_linear_weights(module, "proj_out", [weight_dict])
        assert module.loaded == [weight_dict]

    @pytest.mark.parametrize("quant_algo", [QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
    @pytest.mark.parametrize("quantized", [False, True])
    def test_module_quant_config_controls_loading_despite_checkpoint_alias(
        self, monkeypatch, quant_algo, quantized
    ):
        """A final module recipe takes precedence over an excluded checkpoint short name."""
        loader = _make_loader(quant_algo, dynamic_weight_quant=True, exclude_modules=["proj_out"])
        module = _StubLinear(quant_algo if quantized else None, force_dynamic_quantization=True)
        module.quant_config = QuantConfig(
            quant_algo=quant_algo if quantized else None, group_size=128
        )
        source = torch.randn(128, 256, dtype=torch.bfloat16)
        monkeypatch.setattr(torch.Tensor, "cuda", lambda value: value)
        if quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            quantizer_name = "quantize_fp8_blockwise"
            tensors = (source.to(torch.float8_e4m3fn), torch.ones(1, 2))
        else:
            quantizer_name = "quantize_nvfp4"
            tensors = (
                torch.zeros(128, 128, dtype=torch.uint8),
                torch.ones(128, 16),
                torch.ones(()),
            )
        quantizer = Mock(return_value=tensors)
        with patch(
            f"tensorrt_llm._torch.visual_gen.quantization.loader.{quantizer_name}", quantizer
        ):
            loader.load_linear_weights(module, "proj_out", [{"weight": source}])
        if quantized:
            quantizer.assert_called_once()
            assert module.loaded[0]["weight"] is tensors[0]
            assert module.loaded[0]["weight_scale"] is tensors[1]
        else:
            quantizer.assert_not_called()
            assert module.loaded[0]["weight"] is source
            assert "weight_scale" not in module.loaded[0]

    def test_quantized_module_alias_cannot_bypass_missing_scales(self):
        loader = _make_loader(QuantAlgo.FP8_BLOCK_SCALES, exclude_modules=["proj_out"])
        module = _StubLinear()
        module.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
        module.weight = torch.empty(8, 16, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="appears to be unquantized"):
            loader.load_linear_weights(module, "proj_out", [_bf16_weights()])
        assert module.loaded is None

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


class TestStaticNVFP4ActivationScale:
    @pytest.fixture
    def quantizer(self, monkeypatch):
        monkeypatch.setattr(torch.Tensor, "cuda", lambda value: value)
        quantizer = Mock(
            return_value=(
                torch.zeros(8, 8, dtype=torch.uint8),
                torch.ones(8, 1, dtype=torch.float8_e4m3fn),
                torch.ones(1),
            )
        )
        monkeypatch.setattr(
            "tensorrt_llm._torch.visual_gen.quantization.loader.quantize_nvfp4", quantizer
        )
        return quantizer

    @pytest.mark.parametrize("dynamic_weight_quant", [False, True])
    @pytest.mark.parametrize(
        "input_scale",
        [
            None,
            torch.tensor(0.0),
            torch.tensor(-1.0),
            torch.tensor(float("nan")),
            torch.tensor(float("inf")),
            torch.empty(0),
            torch.ones(2),
        ],
        ids=["missing", "zero", "negative", "nan", "inf", "empty", "non-scalar"],
    )
    def test_missing_or_invalid_scale_raises_before_quantization(
        self, dynamic_weight_quant, input_scale, quantizer
    ):
        loader = _make_loader(QuantAlgo.NVFP4, dynamic_weight_quant=dynamic_weight_quant)
        module = _StubLinear(QuantAlgo.NVFP4)
        weight_dict = (
            _bf16_weights()
            if dynamic_weight_quant
            else {
                "weight": torch.zeros(8, 8, dtype=torch.uint8),
                "weight_scale": torch.ones(8, 1, dtype=torch.float8_e4m3fn),
                "weight_scale_2": torch.ones(1),
            }
        )
        if input_scale is not None:
            weight_dict["input_scale"] = input_scale
        with pytest.raises(ValueError, match="blocks.0.attn1.to_q.*calibrated 'input_scale'"):
            loader.load_linear_weights(module, "blocks.0.attn1.to_q", [weight_dict])
        quantizer.assert_not_called()
        assert module.loaded is None
        assert not module.force_dynamic_quantization

    def test_load_time_quantization_preserves_calibrated_scale(self, quantizer):
        loader = _make_loader(QuantAlgo.NVFP4, dynamic_weight_quant=True)
        module = _StubLinear(QuantAlgo.NVFP4)
        input_scale = torch.tensor([0.125])
        loader.load_linear_weights(
            module, "blocks.0.attn1.to_q", [{**_bf16_weights(), "input_scale": input_scale}]
        )
        quantizer.assert_called_once()
        assert module.loaded[0]["input_scale"] is input_scale
        assert not module.force_dynamic_quantization
        assert not loader.model_config.force_dynamic_quantization

    def test_step_exclusions_still_require_calibrated_scale(self, quantizer):
        loader = DynamicLinearWeightLoader(
            DiffusionModelConfig(
                quant_config=DiffusionQuantConfig(quant_algo=QuantAlgo.NVFP4, ignored_steps=[0]),
                dynamic_weight_quant=True,
                force_dynamic_quantization=False,
            )
        )
        module = _StubLinear(QuantAlgo.NVFP4)
        with pytest.raises(ValueError, match="calibrated 'input_scale'"):
            loader.load_linear_weights(module, "blocks.0.attn1.to_q", [_bf16_weights()])
        quantizer.assert_not_called()
        assert module.loaded is None
        assert not module.force_dynamic_quantization

    def test_fused_weights_require_scale_for_every_entry(self, quantizer):
        loader = _make_loader(QuantAlgo.NVFP4, dynamic_weight_quant=True)
        module = _StubLinear(QuantAlgo.NVFP4)
        weight_dicts = [
            {**_bf16_weights(), "input_scale": torch.ones(1)},
            _bf16_weights(),
            {**_bf16_weights(), "input_scale": torch.ones(1)},
        ]
        with pytest.raises(ValueError, match="blocks.0.attn1.qkv_proj.*calibrated 'input_scale'"):
            loader.load_linear_weights(module, "blocks.0.attn1.qkv_proj", weight_dicts)
        quantizer.assert_not_called()
        assert module.loaded is None

    @pytest.mark.parametrize("activation_mode", ["dynamic", "weight_only"])
    def test_uncalibrated_dynamic_or_weight_only_path_loads(self, activation_mode, quantizer):
        dynamic = activation_mode == "dynamic"
        loader = _make_loader(
            QuantAlgo.NVFP4, dynamic_weight_quant=True, force_dynamic_quantization=dynamic
        )
        module = _StubLinear(QuantAlgo.NVFP4, force_dynamic_quantization=dynamic)
        if activation_mode == "weight_only":
            module.quant_method = W4A16NVFP4LinearMethod()
        loader.load_linear_weights(module, "blocks.0.attn1.to_q", [_bf16_weights()])
        quantizer.assert_called_once()
        assert "input_scale" not in module.loaded[0]
        assert module.force_dynamic_quantization == dynamic

    def test_excluded_nvfp4_layer_needs_no_activation_scale(self, quantizer):
        loader = _make_loader(
            QuantAlgo.NVFP4, dynamic_weight_quant=True, exclude_modules=["proj_out"]
        )
        module = _StubLinear()
        weight_dict = _bf16_weights()
        loader.load_linear_weights(module, "proj_out", [weight_dict])
        quantizer.assert_not_called()
        assert module.loaded == [weight_dict]
