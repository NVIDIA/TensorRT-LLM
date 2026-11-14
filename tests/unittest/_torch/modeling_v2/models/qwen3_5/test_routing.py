# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B-NVFP4 target routing, driven by synthetic configs on CPU."""

import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    KvCacheConfig,
    LoraConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM120 = (12, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = "models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.modeling.ModelingV2Qwen3827BNvfp4Sm120Tp1"
# The engine settings the target serves: an fp8-e4m3 KV pool and the quantized
# recurrent-state pool (int16 rows in an int8-typed pool).
_SERVED_KV = dict(dtype="fp8", mamba_ssm_cache_dtype="int8")


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm120(monkeypatch):
    """Route as if this were an RTX PRO 6000 Blackwell, wherever the test runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM120)


def _llm_args(mode="auto", kv=None, **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=KvCacheConfig(**(_SERVED_KV if kv is None else kv)),
        **kwargs,
    )


def _qwen38_config(**text_overrides):
    """The shape Qwen3.8-27B-NVFP4's config.json declares."""
    text = dict(
        num_hidden_layers=64,
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        vocab_size=248320,
        torch_dtype=torch.bfloat16,
    )
    text.update(text_overrides)
    return PretrainedConfig(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        text_config=PretrainedConfig(**text),
    )


# The checkpoint's per-module assignment (hf_quant_config.json quantized_layers).
_RECIPE = {
    "model.language_model.layers.3.self_attn.q_proj": QuantAlgo.FP8,
    "model.language_model.layers.0.linear_attn.in_proj_qkv": QuantAlgo.FP8,
    "model.language_model.layers.0.mlp.gate_proj": QuantAlgo.NVFP4,
    "lm_head": QuantAlgo.NVFP4,
}
# Both algorithms present, assigned to the wrong modules.
_SWAPPED_RECIPE = {
    "model.language_model.layers.3.self_attn.q_proj": QuantAlgo.NVFP4,
    "model.language_model.layers.0.linear_attn.in_proj_qkv": QuantAlgo.NVFP4,
    "model.language_model.layers.0.mlp.gate_proj": QuantAlgo.FP8,
    "lm_head": QuantAlgo.FP8,
}


def _model_config(pretrained_config, spec_config=None, recipe=True, **mapping_kwargs):
    """The MIXED_PRECISION recipe: FP8 attention / Gated-DeltaNet projections, NVFP4 MLP."""
    if recipe is True:
        layer_algos = _RECIPE
    elif recipe is False:
        layer_algos = {"model.language_model.layers.0.mlp.gate_proj": QuantAlgo.NVFP4}
    else:
        layer_algos = recipe
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs) if mapping_kwargs else Mapping(),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={k: QuantConfig(quant_algo=v) for k, v in layer_algos.items()},
    )


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_qwen38_27b_tp1_matches(mode):
    config = _model_config(_qwen38_config())
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TARGET


@pytest.mark.parametrize(
    "config, spec_config, mapping_kwargs, missed",
    [
        (_qwen38_config(num_hidden_layers=48), None, {}, r"text shape \(48,"),
        (_qwen38_config(linear_value_head_dim=64), None, {}, r"text shape \(64, .*, 64, 4, 4,"),
        (_qwen38_config(linear_conv_kernel_dim=3), None, {}, r"text shape \(64, .*, 128, 3, 4,"),
        (_qwen38_config(), None, dict(world_size=2, tp_size=2), "2 ranks"),
        (_qwen38_config(), MTPDecodingConfig(max_draft_len=3), {}, "speculative decoding MTP"),
    ],
    ids=["other-shape", "linear-head-64", "conv-3", "tp2", "mtp"],
)
def test_near_misses_do_not_match(config, spec_config, mapping_kwargs, missed):
    model_config = _model_config(config, spec_config=spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(model_config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(model_config, _llm_args("require"))


@pytest.mark.parametrize("recipe", [False, _SWAPPED_RECIPE], ids=["nvfp4-only", "swapped"])
def test_other_recipe_does_not_match(recipe):
    config = _model_config(_qwen38_config(), recipe=recipe)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="not the FP8/NVFP4 MIXED_PRECISION recipe"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_other_layer_pattern_does_not_match():
    types = ["linear_attention"] * 63 + ["full_attention"]
    config = _model_config(_qwen38_config(layer_types=types))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="layer_types"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_on_sm103_does_not_match(monkeypatch):
    """The target's kernels are built for sm120; a GB300 keeps the built-in model."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_qwen38_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "kv, kwargs, missed",
    [
        (dict(dtype="auto", mamba_ssm_cache_dtype="int8"), {}, "KV cache"),
        (dict(dtype="fp8", mamba_ssm_cache_dtype="bfloat16"), {}, "recurrent-state cache"),
        (_SERVED_KV, dict(prefill_cuda_graph_backend="breakable"), "prefill_cuda_graph_backend"),
        (_SERVED_KV, dict(attn_backend="FLASHINFER"), "attn_backend='FLASHINFER'"),
        (_SERVED_KV, dict(lora_config=LoraConfig()), "LoRA"),
        (
            _SERVED_KV,
            dict(cache_transceiver_config=CacheTransceiverConfig(backend="NIXL")),
            "disaggregated serving",
        ),
        (
            dict(_SERVED_KV, use_kv_cache_manager_v2=False),
            {},
            "use_kv_cache_manager_v2=False",
        ),
    ],
    ids=[
        "bf16-kv",
        "bf16-state",
        "breakable-prefill-graph",
        "flashinfer-attention",
        "lora",
        "disaggregated",
        "kv-manager-v1",
    ],
)
def test_engine_choices_the_target_refuses(kv, kwargs, missed):
    config = _model_config(_qwen38_config())
    assert modeling_v2_resolve(config, _llm_args("auto", kv=kv, **kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", kv=kv, **kwargs))


def test_off_never_imports_or_claims(monkeypatch):
    module = "tensorrt_llm._torch._experimental.modeling_v2." + _TARGET.rsplit(".", 1)[0]
    monkeypatch.delitem(sys.modules, module, raising=False)
    config = _model_config(_qwen38_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    assert module not in sys.modules


@pytest.mark.parametrize(
    "data, refused",
    [
        (None, False),
        ({"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 4)}}, False),
        ({"image": {"pixel_values": torch.zeros(4, 8)}}, True),
        ({"video": {"pixel_values_videos": torch.zeros(4, 8)}}, True),
        ({"multimodal_embedding": torch.zeros(4, 8)}, True),
    ],
    ids=["text", "text-with-mrope", "image", "video", "embedding"],
)
def test_admission_refuses_only_media_requests(data, refused):
    """Media fails at admission as that request's own error, not inside a shared step."""
    config = _model_config(_qwen38_config())
    target = modeling_v2_resolve(config, _llm_args("require"))
    request = SimpleNamespace(py_multimodal_data=data)
    if refused:
        with pytest.raises(ValueError, match="text only"):
            target.validate_request(None, request)
    else:
        target.validate_request(None, request)
