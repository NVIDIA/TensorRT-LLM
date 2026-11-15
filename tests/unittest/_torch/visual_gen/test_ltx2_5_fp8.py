# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""FP8 Linear fidelity (T2) and stage reload against independent quantization (T1)."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

__extra_import_path__ = ["~/tests/unittest/_torch/visual_gen"]

from ltx2_5_quant_reference import decode_ue8m0, decode_w, decode_x

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.ltx2_5.fp8_lora import FP8StageLoRAWeights
from tensorrt_llm._torch.visual_gen.models.ltx2_5.loading import (
    load_transformer_weights_strict,
    validate_transformer_weights,
)
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_independent_fp8_decode_uses_distinct_row_and_block_scale_grids():
    """L1: analytic unit payload checks distinguish activation and weight axes."""
    payload = torch.ones(256, 256, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], device="cuda")
    weight = decode_w(payload, weight_scale)
    assert torch.equal(weight[:128, :128], torch.ones_like(weight[:128, :128]))
    assert torch.equal(weight[128:, 128:], torch.full_like(weight[128:, 128:], 8.0))
    assert weight[0, 128] == 2 and weight[128, 0] == 4
    activation_scale = torch.stack(
        (torch.arange(1, 257, device="cuda"), torch.arange(257, 513, device="cuda"))
    ).float()
    activation = decode_x(payload, activation_scale)
    assert activation[0, 0] == 1 and activation[255, 127] == 256
    assert activation[0, 128] == 257 and activation[255, 255] == 512


def test_independent_ue8m0_decode_known_exponent_bytes():
    """L1: known little-endian packed exponent word, with a strided scale view."""
    packed = torch.tensor([[0x7E81807F, 0], [0, 0x7E81807F]], dtype=torch.int32, device="cuda").t()
    actual = decode_ue8m0(packed, 4)
    expected = torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda")
    assert torch.equal(actual[0], expected)
    with pytest.raises(ValueError, match="Nonfinite"):
        decode_ue8m0(torch.tensor([[255]], dtype=torch.int32, device="cuda"), 1)


def _config():
    return DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128),
        dynamic_weight_quant=True,
        mapping=Mapping(),
    )


def _linear(raw_scales=False):
    return Linear(
        512,
        512,
        bias=True,
        dtype=torch.bfloat16,
        mapping=Mapping(),
        use_cute_dsl_blockscaling_mm=raw_scales,
        quant_config=_config().quant_config,
    ).cuda()


class _Model(torch.nn.Module):
    def __init__(self, raw_scales=False):
        super().__init__()
        self.proj = _linear(raw_scales)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, device="cuda"))

    @torch.no_grad()
    def load_weights(self, weights):
        self.scale_shift_table.copy_(weights["scale_shift_table"].bfloat16())
        loader = DynamicLinearWeightLoader(_config())
        loader.load_linear_weights(
            self.proj, "proj", [{"weight": weights["proj.weight"], "bias": weights["proj.bias"]}]
        )


@pytest.fixture(params=[False, True], ids=["ue8m0", "fp32-scales"])
def experiment(tmp_path, request):
    torch.manual_seed(42)
    source = {
        "proj.weight": torch.randn(512, 512, dtype=torch.bfloat16) * 0.02,
        "proj.bias": torch.randn(512, dtype=torch.bfloat16) * 0.01,
        "scale_shift_table": torch.tensor([0.33333334, 1.001]),
    }
    a = torch.randn(16, 512, dtype=torch.bfloat16) * 0.1
    b = torch.randn(512, 16, dtype=torch.bfloat16) * 0.1
    path = tmp_path / "adapter.safetensors"
    save_file({"proj.lora_A.weight": a, "proj.lora_B.weight": b}, str(path))
    model = _Model(request.param)
    load_transformer_weights_strict(model, source)
    assert torch.equal(model.scale_shift_table.cpu(), source["scale_shift_table"])
    bank = FP8StageLoRAWeights(model, source, str(path), 1.0)
    model.proj.post_load_weights()
    merged = (b.cuda() * 1.0) @ a.cuda()
    merged.add_(source["proj.weight"].cuda())
    return model, bank, source, merged


def _errors(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    delta = a - b
    result = {
        "max_abs": delta.abs().max().item(),
        "normalized_rmse": (delta.norm() / b.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "p99_abs": torch.quantile(delta.abs(), 0.99).item(),
        "max_channel_mean_shift": (actual.float() - expected.float()).mean(0).abs().max().item(),
    }
    print(result)
    assert torch.isfinite(actual).all(), result
    return result


def test_fp8_stage_reload_matches_independent_merged_weight(experiment):
    """T1: stage-selected Linear output versus a separately quantized BF16 merge."""
    model, bank, source, merged = experiment
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    base = model.proj(x)
    bank.select(True)
    conditioned = model.proj(x)
    oracle = _linear(model.proj.use_cute_dsl_blockscaling_mm)
    q, s = quantize_fp8_blockwise(merged)
    oracle.load_weights([{"weight": q, "weight_scale": s, "bias": source["proj.bias"]}])
    oracle.post_load_weights()
    error = _errors(conditioned, oracle(x))
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error
    assert bank.quantized_target_count == 1
    assert model.proj.weight.dtype == torch.float8_e4m3fn
    assert not torch.equal(conditioned, base), "LoRA must change the output"
    bank.select(False)
    error = _errors(model.proj(x), base)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error
    model.proj(torch.randn(128, 512, device="cuda", dtype=torch.bfloat16))
    bank.select(True)
    error = _errors(model.proj(x), conditioned)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error


def test_fp8_linear_fidelity_against_bf16_oracle(experiment):
    """T2: actual FP8 GEMM, same BF16 weights/input, no attention quantization."""
    model, bank, source, merged = experiment
    bank.select(True)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, merged, source["proj.bias"].cuda())
    error = _errors(model.proj(x), expected)
    # Calibrated from the B200 FP8 GEMM probe (~0.046 relative L2, 0.9989 cosine).
    assert error["normalized_rmse"] <= 0.08 and error["cosine"] >= 0.996, error


def test_fp8_refinement_projection_accepts_channel_major_latents(experiment):
    """T1: actual refinement layout through the preprocessor versus contiguous input."""
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import (
        _LTX25ArgsPreprocessor,
    )

    model, bank, source, merged = experiment
    latent = torch.randn(1, 512, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    assert not latent.is_contiguous()
    preprocessor = SimpleNamespace(
        patchify_proj=model.proj,
        keyframes_embedding_provider=None,
        _prepare_timestep=lambda *args: (torch.zeros(1), torch.zeros(1)),
    )
    modality = SimpleNamespace(
        latent=latent, timesteps=torch.ones(1), keyframes_mask=None, enabled=True
    )
    expected = model.proj(latent.contiguous())
    seen = []
    handle = model.proj.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].is_contiguous())
    )
    try:
        actual = _LTX25ArgsPreprocessor.prepare(preprocessor, modality, None, None, (None, None)).x
    finally:
        handle.remove()
    assert seen == [True]
    assert not latent.is_contiguous()
    error = _errors(actual, expected)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error


def test_partial_fp8_switch_can_restore_base(experiment, monkeypatch):
    model, bank, source, merged = experiment
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    base = model.proj(x)
    load = model.proj.load_weights

    def fail(weights):
        raise RuntimeError("injected reload failure")

    monkeypatch.setattr(model.proj, "load_weights", fail)
    with pytest.raises(RuntimeError, match="injected"):
        bank.select(True)
    assert bank._stage_one is None
    monkeypatch.setattr(model.proj, "load_weights", load)
    bank.select(False)
    error = _errors(model.proj(x), base)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error


def test_derived_scales_do_not_hide_missing_or_extra_checkpoint_keys(experiment):
    model, bank, source, merged = experiment
    validate_transformer_weights(model, source)
    missing = {k: v for k, v in source.items() if k != "scale_shift_table"}
    with pytest.raises(ValueError, match="scale_shift_table"):
        validate_transformer_weights(model, missing)
    with pytest.raises(ValueError, match="unexpected"):
        validate_transformer_weights(model, {**source, "proj.weight_scale": torch.ones(4, 4)})


def test_fp8_stage_bank_restores_excluded_bf16_lora_targets(experiment, tmp_path):
    """T1: excluded targets retain BF16 fusion and exact source values on reload."""
    model, _, source, merged = experiment
    model.proj = Linear(
        512, 512, bias=True, dtype=torch.bfloat16, mapping=Mapping(), quant_config=QuantConfig()
    ).cuda()
    load_transformer_weights_strict(model, source)
    bank = FP8StageLoRAWeights(model, source, str(tmp_path / "adapter.safetensors"), 1.0)
    model.proj.post_load_weights()
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    base = model.proj(x)
    bank.select(True)
    assert bank.quantized_target_count == 0
    weight = model.proj.weight
    weight = weight[0].t() if weight.ndim == 3 else weight
    torch.testing.assert_close(weight, merged, rtol=0, atol=0)
    expected = torch.nn.functional.linear(x, merged, source["proj.bias"].cuda())
    error = _errors(model.proj(x), expected)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error
    bank.select(False)
    error = _errors(model.proj(x), base)
    assert error["normalized_rmse"] <= 0.01 and error["cosine"] >= 0.9999, error


def test_stage_bank_rejects_already_transformed_fp8_scales(experiment, tmp_path):
    model, bank, source, merged = experiment
    assert model.proj._weights_transformed
    with pytest.raises(ValueError, match="pre-transform"):
        FP8StageLoRAWeights(model, source, str(tmp_path / "adapter.safetensors"), 1.0)
