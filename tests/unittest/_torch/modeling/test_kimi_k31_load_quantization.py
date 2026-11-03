# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3.1 opt-in loading, numerical conversion, and checkpoint shard behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.kimi_k31_load_quantization import (
    dequantize_mxfp4_weight,
    quantize_attention_weight,
    quantize_expert_weight,
)
from tensorrt_llm._torch.models.modeling_kimi_k3_vl import (
    KimiK3ForConditionalGeneration,
    KimiK31ForConditionalGeneration,
)
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _config(*, kda=True, native=False):
    text = SimpleNamespace(
        num_hidden_layers=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        linear_attn_config={"kda_layers": [1] if kda else []},
        quantization_config={"format": "mxfp4-pack-quantized"},
    )
    return ModelConfig(
        pretrained_config=SimpleNamespace(model_type="kimi_k31", text_config=text),
        moe_backend="TRTLLM",
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict={
            "model.layers.0.block_sparse_moe.experts": QuantConfig(
                quant_algo=QuantAlgo.NVFP4, group_size=16
            )
        }
        if native
        else None,
    )


class _Vision(nn.Module):
    def load_weights(self, weights):
        pass


@pytest.mark.cpu_only
@pytest.mark.parametrize("ssm_dtype", [None, torch.float32])
@pytest.mark.parametrize(
    "setting,native,converts",
    [(None, False, False), ("0", False, False), ("1", False, True), ("1", True, False)],
)
def test_k31_construction_switch(monkeypatch, setting, native, converts, ssm_dtype):
    if setting is None:
        monkeypatch.delenv("TLLM_K31_LOAD_QUANTIZE", raising=False)
    else:
        monkeypatch.setenv("TLLM_K31_LOAD_QUANTIZE", setting)
    seen = []

    def initialize(model, config):
        nn.Module.__init__(model)
        seen.append(config)
        model.llm = nn.Identity()

    monkeypatch.setattr(KimiK3ForConditionalGeneration, "__init__", initialize)
    config = _config(native=native)
    original = config.quant_config
    original.mamba_ssm_cache_dtype = ssm_dtype
    # Configuration consumers may populate these caches before model construction.
    _ = original.quant_mode, original.layer_quant_mode
    KimiK31ForConditionalGeneration(config)
    assert seen == [config]
    assert config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert config.quant_config.mamba_ssm_cache_dtype == ssm_dtype
    if converts:
        expected = QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=QuantAlgo.FP8
        )
        assert config.quant_config.layer_quant_mode == expected.layer_quant_mode
        assert (
            config.quant_config_dict["model.layers.0.self_attn.q_proj"].quant_algo
            == QuantAlgo.FP8_BLOCK_SCALES
        )
        assert (
            config.quant_config_dict["model.layers.0.block_sparse_moe.experts"].quant_algo
            == QuantAlgo.NVFP4
        )
    else:
        assert config.quant_config is original


@pytest.mark.cpu_only
@pytest.mark.parametrize("setting", ["true", "yes", "2", ""])
def test_k31_rejects_invalid_switch(monkeypatch, setting):
    monkeypatch.setenv("TLLM_K31_LOAD_QUANTIZE", setting)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        KimiK31ForConditionalGeneration(_config())


@pytest.mark.cpu_only
def test_mxfp4_dequantization_preserves_decoded_values() -> None:
    # Cover every FP4 bit pattern, including a wide range of E8M0 scales.
    packed = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    exponent = torch.arange(16, dtype=torch.uint8).reshape(8, 2) + 112
    exponent[0, 0] = 100
    actual = dequantize_mxfp4_weight(packed, exponent)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])

    def unpack(values: torch.Tensor) -> torch.Tensor:
        indices = torch.stack((values & 15, values >> 4), dim=-1).long().flatten(-2)
        return lut[indices]

    expected = unpack(packed) * torch.exp2(exponent.float() - 127).repeat_interleave(32, 1)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "exponents,message", [([127, 255], "NaN"), ([127, 254], "not representable in BF16")]
)
def test_mxfp4_dequantization_rejects_nonfinite_weights(exponents: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dequantize_mxfp4_weight(
            torch.full((1, 32), 0x77, dtype=torch.uint8),
            torch.tensor([exponents], dtype=torch.uint8),
        )


@pytest.mark.skipif(not torch.cuda.is_available() or not is_sm_100f(), reason="requires Blackwell")
@pytest.mark.parametrize("zero", [False, True])
def test_expert_nvfp4_quantization_round_trip(zero: bool) -> None:
    weight = torch.randn(128, 256, generator=torch.Generator().manual_seed(19)).bfloat16().cuda()
    if zero:
        weight.zero_()
    codes, scales, global_scale = quantize_expert_weight(weight)
    recovered = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        codes.cpu(), scales.cpu().flatten(), global_scale.cpu(), 16, 1, False
    )
    assert scales.shape == (128, 16)
    assert scales.dtype == torch.uint8  # Native quantizer returns E4M3 bytes.
    assert torch.isfinite(recovered).all()
    if zero:
        assert torch.count_nonzero(recovered) == 0
    else:
        error = torch.linalg.vector_norm(
            recovered - weight.cpu().float()
        ) / torch.linalg.vector_norm(weight.cpu().float())
        assert error < 0.11


@pytest.mark.cpu_only
def test_attention_conversion_handles_partial_and_zero_blocks():
    weight = torch.randn(129, 257, generator=torch.Generator().manual_seed(17)).bfloat16()
    weight[:128, :128] = 0
    codes, scales = quantize_attention_weight(weight)
    recovered = (
        codes.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:129, :257]
    )
    assert codes.shape == weight.shape
    assert scales.shape == (2, 3)
    assert torch.isfinite(recovered).all()
    assert torch.count_nonzero(recovered[:128, :128]) == 0
    error = torch.linalg.vector_norm(recovered - weight.float()) / torch.linalg.vector_norm(
        weight.float()
    )
    assert error < 0.04


@pytest.mark.skipif(not torch.cuda.is_available() or not is_sm_100f(), reason="requires Blackwell")
@pytest.mark.parametrize("kda", [True, False])
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1)])
def test_k31_wrapper_loads_bf16_attention_into_fp8(monkeypatch, kda, tp, rank):
    # Reuse the small attention fixture; the public VLM and text loaders run intact.
    from _torch.modeling.test_kimi_linear_checkpoint import _checkpoint, _model

    source = _model(kda=kda, quantized=())
    checkpoint = _checkpoint(source)
    checkpoint = {key: value.cpu() for key, value in checkpoint.items()}
    del source

    def initialize(model, config):
        nn.Module.__init__(model)
        model.llm = _model(kda=kda, tp_size=tp, tp_rank=rank)
        model.mm_encoder = _Vision()

    monkeypatch.setattr(KimiK3ForConditionalGeneration, "__init__", initialize)
    monkeypatch.setenv("TLLM_K31_LOAD_QUANTIZE", "1")
    model = KimiK31ForConditionalGeneration(_config(kda=kda))
    model.load_weights({"language_model." + key: value for key, value in checkpoint.items()})

    # Load an independently prepared native-FP8 checkpoint through the same text model.
    reference = _model(kda=kda, tp_size=tp, tp_rank=rank)
    converted = dict(checkpoint)
    for key, weight in checkpoint.items():
        if ".self_attn." in key and key.endswith(".weight") and weight.ndim == 2:
            codes, scales = quantize_attention_weight(weight.cuda())
            converted[key], converted[key + "_scale"] = codes.cpu(), scales.cpu()
    reference.load_weights(converted)
    for name, value in model.llm.named_parameters():
        expected = dict(reference.named_parameters())[name]
        torch.testing.assert_close(value.float(), expected.float(), rtol=0, atol=0)
    for key, value in checkpoint.items():
        if value.is_floating_point():
            assert value.dtype != torch.float8_e4m3fn


@pytest.mark.skipif(not torch.cuda.is_available() or not is_sm_100f(), reason="requires Blackwell")
@pytest.mark.parametrize("layout", ["dict", "one_file", "split_files"])
@pytest.mark.parametrize("moe_backend", ["TRTLLM", "CUTLASS"])
def test_k31_streamed_experts_match_native_nvfp4_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layout: str, moe_backend: str
) -> None:
    from _torch.modeling.test_kimi_linear_checkpoint import _checkpoint, _model
    from _torch.moe.test_kimi_k3_situ_moe import (
        _TP_EXPERTS,
        _TP_HIDDEN,
        _TP_INTERMEDIATE,
        _load_nvfp4_bank_for,
        _make_nvfp4_moe,
        _make_packed_expert_bank,
        _make_test_gate,
        _quantize_expert_to_nvfp4,
    )
    from safetensors.torch import save_file

    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict

    gate = _make_test_gate()
    moe = _make_nvfp4_moe(gate, moe_backend=moe_backend)
    bank = _make_packed_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    # Different gate/up ranges exercise the shared global scale used by CUTLASS.
    bank[0]["w3_sf"] -= 2
    source = _model(kda=True, quantized=())
    checkpoint = {
        "language_model." + key: value.cpu() for key, value in _checkpoint(source).items()
    }
    del source
    expert_weights = {}
    native_bank = []
    for expert, tensors in enumerate(bank):
        bf16_weights = []
        for projection in ("w1", "w2", "w3"):
            stem = f"language_model.model.layers.0.block_sparse_moe.experts.{expert}.{projection}"
            expert_weights[stem + ".weight_packed"] = tensors[projection]
            expert_weights[stem + ".weight_scale"] = tensors[projection + "_sf"]
            # Decode with the independent reference op and use the native
            # checkpoint quantizer rather than the load-time adapter.
            decoded = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
                tensors[projection].cpu(),
                tensors[projection + "_sf"].cpu().flatten(),
                None,
                32,
                0,
                False,
            )
            bf16_weights.append(decoded.bfloat16().cuda())
        native_bank.append(_quantize_expert_to_nvfp4(*bf16_weights, input_scale=1.0))
    checkpoint.update(expert_weights)
    weights = ConsumableWeightsDict(checkpoint)
    if layout != "dict":
        shards = {"experts.safetensors": expert_weights}
        if layout == "split_files":
            shards = {
                f"{projection}.safetensors": {
                    k: v for k, v in expert_weights.items() if f".{projection}." in k
                }
                for projection in ("w1", "w2", "w3")
            }
        index = {}
        for filename, values in shards.items():
            save_file(values, tmp_path / filename)
            index.update({key: filename for key in values})
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
        weights.checkpoint_dir = str(tmp_path)

    def initialize(model, config):
        nn.Module.__init__(model)
        model.llm = _model(kda=True)
        layer = model.llm.model.layers[0]
        layer.is_moe = True
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.routed_experts = moe
        layer.block_sparse_moe.local_expert_ids = list(moe.backend.initial_local_expert_ids)
        # KimiK3MoERuntime without an expert placement: each local slot loads its own checkpoint
        # expert (ckpt_expert_ids == local_expert_ids).
        layer.block_sparse_moe.ckpt_expert_ids = tuple(layer.block_sparse_moe.local_expert_ids)
        model.mm_encoder = _Vision()

    monkeypatch.setattr(KimiK3ForConditionalGeneration, "__init__", initialize)
    monkeypatch.setenv("TLLM_K31_LOAD_QUANTIZE", "1")
    model = KimiK31ForConditionalGeneration(_config())
    model.load_weights(weights)
    moe.post_load_weights()
    reference = _make_nvfp4_moe(gate, moe_backend=moe_backend)
    _load_nvfp4_bank_for(reference, native_bank, moe_backend)
    torch.manual_seed(43)
    hidden = torch.randn(16, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = gate.compute_logits(hidden)
    expected = reference.forward(hidden, logits, all_rank_num_tokens=None)
    actual = moe.forward(hidden, logits, all_rank_num_tokens=None)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
