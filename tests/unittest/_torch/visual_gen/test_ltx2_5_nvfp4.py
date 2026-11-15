# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""L1 independent FP4 decoding and L2 dynamic stage-bank/state correctness."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

__extra_import_path__ = ["~/tests/unittest/_torch/visual_gen"]

from ltx2_5_quant_reference import decode_e4m3, decode_operand, unswizzle_scales

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.ltx2_5.loading import (
    load_transformer_weights_strict,
    validate_transformer_weights,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_5.nvfp4_lora import NVFP4StageLoRAWeights
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _config():
    return DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        dynamic_weight_quant=True,
        force_dynamic_quantization=True,
        mapping=Mapping(),
    )


def _linear(bias=False):
    with torch.device("cuda"):
        return Linear(
            512,
            512,
            bias=bias,
            dtype=torch.bfloat16,
            mapping=Mapping(),
            quant_config=_config().quant_config,
            force_dynamic_quantization=True,
        )


def test_fp4_known_nibbles_and_scale_extremes():
    payload = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], device="cuda", dtype=torch.uint8
    )
    expected = torch.tensor(
        [[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]], device="cuda"
    )
    for code, factor in ((0x00, 0), (0x01, 2**-9), (0x38, 1), (0x7E, 448)):
        sf = torch.tensor([[code]], device="cuda", dtype=torch.uint8)
        assert torch.equal(decode_operand(payload, sf, swizzled=False), expected * factor)
    assert torch.isnan(decode_e4m3(torch.tensor([0x7F], device="cuda", dtype=torch.uint8))).all()


@pytest.mark.parametrize("rows,columns", [(1, 256), (40, 4096), (128, 256), (257, 128)])
def test_independent_scale_offsets_invert_native_interleave(rows, columns):
    scales = (
        torch.arange(rows * (columns // 16), device="cuda")
        .remainder(127)
        .to(torch.uint8)
        .reshape(rows, -1)
    )
    actual = torch.ops.trtllm.block_scale_interleave(scales)
    assert torch.equal(unswizzle_scales(actual, rows, columns), scales)


@pytest.mark.parametrize("kind", ["zeros", "tiny", "mixed-zero-block", "signed-extremes"])
def test_native_quantizer_boundaries(kind):
    weight = torch.zeros(128, 512, device="cuda", dtype=torch.bfloat16)
    if kind == "tiny":
        weight.fill_(2**-20)
    elif kind == "mixed-zero-block":
        weight[:, 256:] = 1
    elif kind == "signed-extremes":
        weight[:, :256] = 2**20
        weight[:, 256:] = -(2**20)
    packed, sf, gs = quantize_nvfp4(weight)
    actual = decode_operand(packed, sf, swizzled=False) * gs
    assert torch.isfinite(actual).all(), (kind, gs, sf)
    torch.testing.assert_close(actual, weight.float(), rtol=0.15, atol=0)
    if kind == "signed-extremes":
        assert (packed.bitwise_and(7) == 7).all()
        assert (packed.bitwise_right_shift(4).bitwise_and(7) == 7).all()


@pytest.mark.parametrize("zero_weight", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
def test_dynamic_native_linear_zero_operand_is_finite_zero(zero_weight, with_bias):
    layer = _linear(bias=with_bias)
    weight = torch.full((512, 512), 0 if zero_weight else 1, dtype=torch.bfloat16, device="cuda")
    q, sf, gs = quantize_nvfp4(weight)
    state = {"weight": q, "weight_scale": sf, "weight_scale_2": gs}
    if with_bias:
        state["bias"] = torch.linspace(-0.2, 0.2, 512, device="cuda").bfloat16()
    layer.load_weights([state])
    layer.post_load_weights()
    x = torch.full((3, 512), 1 if zero_weight else 0, dtype=torch.bfloat16, device="cuda")
    output = layer(x)
    expected = state["bias"].expand_as(output) if with_bias else torch.zeros_like(output)
    assert torch.isfinite(output).all() and torch.equal(output, expected)


def test_nvfp4_refinement_accepts_channel_major_latents():
    """T1: the production preprocessor makes rows contiguous for the FP4 quantizer."""
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import (
        _LTX25ArgsPreprocessor,
    )

    layer = _linear()
    weight = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scale, global_scale = quantize_nvfp4(weight)
    with torch.inference_mode():
        layer.load_weights([{"weight": q, "weight_scale": scale, "weight_scale_2": global_scale}])
        layer.post_load_weights()
        latent = torch.randn(1, 512, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        expected = layer(latent.contiguous())
        seen = []
        handle = layer.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].is_contiguous())
        )
        preprocessor = SimpleNamespace(
            patchify_proj=layer,
            keyframes_embedding_provider=None,
            _prepare_timestep=lambda *args: (
                torch.zeros(1, device="cuda"),
                torch.zeros(1, device="cuda"),
            ),
        )
        modality = SimpleNamespace(
            latent=latent, timesteps=torch.ones(1, device="cuda"), keyframes_mask=None, enabled=True
        )
        try:
            actual = _LTX25ArgsPreprocessor.prepare(
                preprocessor, modality, None, None, (None, None)
            ).x
        finally:
            handle.remove()
        assert seen == [True] and not latent.is_contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequence", [40, 128])
def test_nvfp4_video_attention_against_same_input_dense_backend(sequence, monkeypatch):
    """T2: native NVFP4-QK/FP8-V; preexisting FlashInfer SM100 cosine bar >0.98."""
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    torch.manual_seed(2)
    dense = FlashInferAttention(num_heads=32, head_dim=128, dtype=torch.bfloat16)
    quantized = FlashInferAttention(
        num_heads=32,
        head_dim=128,
        dtype=torch.bfloat16,
        quant_attention_config=QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="fp8"),
    )
    calls = []
    original = quantized._run_blockscaled_sm10x

    def observe(*args, **kwargs):
        calls.append(kwargs["qk_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(quantized, "_run_blockscaled_sm10x", observe)
    q, k, v = [
        torch.randn(1, sequence, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        for _ in range(3)
    ]
    actual, expected = quantized.forward(q, k, v), dense.forward(q, k, v)
    a, b = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(
        {
            "sequence": sequence,
            "cosine": cosine,
            "relative_l2": float((a - b).norm() / b.norm()),
            "max_abs": float((a - b).abs().max()),
        }
    )
    assert calls == ["nvfp4"] and torch.isfinite(actual).all() and cosine > 0.98


def test_nvfp4_stage_banks_dynamic_activation_and_original_f32(tmp_path, monkeypatch):
    """T1: stage switches/cache invalidation and dynamic alpha; T2 math tested separately."""
    torch.manual_seed(42)
    config = _config()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _linear()
            self.scale_shift_table = torch.nn.Parameter(torch.empty(2, device="cuda"))

        def load_weights(self, weights):
            self.scale_shift_table.copy_(weights["scale_shift_table"].bfloat16())
            DynamicLinearWeightLoader(config).load_linear_weights(
                self.proj, "proj", [{"weight": weights["proj.weight"]}]
            )

    source = {
        "proj.weight": torch.randn(512, 512, dtype=torch.bfloat16) * 0.02,
        "scale_shift_table": torch.tensor([0.33333334, 1.001]),
    }
    a, b = (
        torch.randn(16, 512, dtype=torch.bfloat16) * 0.1,
        torch.randn(512, 16, dtype=torch.bfloat16) * 0.1,
    )
    path = tmp_path / "adapter.safetensors"
    save_file({"proj.lora_A.weight": a, "proj.lora_B.weight": b}, str(path))
    model = Model()
    with torch.inference_mode():
        load_transformer_weights_strict(model, source)
        bank = NVFP4StageLoRAWeights(model, source, str(path), 1)
        model.proj.post_load_weights()
        x = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
        outputs = {}
        alphas = []
        original = model.proj.quant_method._input_prepare

        def observe(layer, value):
            result = original(layer, value)
            alphas.append(result[2].clone())
            return result

        monkeypatch.setattr(model.proj.quant_method, "_input_prepare", observe)
        for stage in (True, False, True, False):
            model.proj._output_slice_cache = object()
            bank.select(stage)
            assert model.proj._output_slice_cache is None
            output = model.proj(x)
            if stage in outputs:
                assert torch.equal(output, outputs[stage])
            else:
                outputs[stage] = output.clone()
            assert torch.equal(model.scale_shift_table, source["scale_shift_table"].cuda())
        assert not torch.equal(outputs[True], outputs[False])
        model.proj(x * 2)
        assert torch.equal(alphas[-1], alphas[-2] * 2)
        assert bank.quantized_target_count == 1 and model.proj.weight.dtype == torch.uint8
        with pytest.raises(ValueError, match="mismatch"):
            validate_transformer_weights(model, {**source, "proj.arbitrary_scale": torch.ones(1)})
        with pytest.raises(ValueError, match="mismatch"):
            validate_transformer_weights(model, {"proj.weight": source["proj.weight"]})
        original_load = model.proj.load_weights

        def fail_load(*args, **kwargs):
            raise RuntimeError("injected bank reload failure")

        monkeypatch.setattr(model.proj, "load_weights", fail_load)
        with pytest.raises(RuntimeError, match="injected"):
            bank.select(True)
        assert bank._stage_one is None
        monkeypatch.setattr(model.proj, "load_weights", original_load)
        bank.select(False)
        assert torch.equal(model.proj(x), outputs[False])


def test_nvfp4_bank_preserves_explicit_bf16_lora_target(tmp_path):
    """T1: an explicitly excluded projection still gets exact BF16 stage weights."""
    model = torch.nn.Module()
    model.proj = Linear(
        128, 128, bias=False, dtype=torch.bfloat16, mapping=Mapping(), quant_config=QuantConfig()
    ).cuda()
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    a, b = torch.randn(8, 128, dtype=torch.bfloat16), torch.randn(128, 8, dtype=torch.bfloat16)
    path = tmp_path / "adapter.safetensors"
    save_file({"proj.lora_A.weight": a, "proj.lora_B.weight": b}, str(path))
    with torch.inference_mode():
        model.proj.load_weights([{"weight": weight}])
        model.proj.post_load_weights()
        bank = NVFP4StageLoRAWeights(model, {"proj.weight": weight}, str(path), 1)
        merged = b.cuda() @ a.cuda()
        merged.add_(weight.cuda())
        for stage in (True, False, True):
            bank.select(stage)
            actual = model.proj.weight
            if actual.ndim == 3:
                actual = actual[0].t()
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual, merged if stage else weight.cuda())
        assert bank.quantized_target_count == 0
