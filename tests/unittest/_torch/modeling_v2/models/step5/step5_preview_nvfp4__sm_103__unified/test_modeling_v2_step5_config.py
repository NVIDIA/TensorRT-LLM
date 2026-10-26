# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the engine reads a Step-5 checkpoint and where its config routes.

Three things, all on the CPU:

* the published ``config.json`` loads through ``load_pretrained_config`` and
  ``ModelConfig.from_pretrained`` with no ``trust_remote_code``, exposing every text field
  the forward reads, and the ModelOpt NVFP4 exclusions leave only the routed experts of
  layers 3-87 quantized (needs the checkpoint: ``STEP5_PREVIEW_CHECKPOINT``, else
  ``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``; skipped without one);
* the ``is_step5`` family predicate;
* the routing table, on synthetic configs of the checkpoint's shape: which deployments
  ``route`` claims and which it declines, and why.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.configs import Step5Config, Step5TextConfig
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_step5_nvfp4_expert_layers,
    get_step5_text_config,
    is_step5,
    load_pretrained_config,
    step5_routed_experts_module,
)
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.step5."
    "step5_preview_nvfp4__sm_103__unified.modeling"
)
_TARGET_CLASS = "ModelingV2Step5PreviewNvfp4Sm103Unified"

_DECODER_LAYERS = 92
_MTP_LAYERS = 3
_FULL_LAYERS = tuple(range(3, _DECODER_LAYERS, 4))
_MOE_LAYERS = tuple(range(3, 91))
_NVFP4_EXPERT_LAYERS = tuple(range(3, 88))


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else root / "Step-5-Preview-NVFP4"


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = _checkpoint_dir()
    if path is None or not (path / "config.json").is_file():
        pytest.skip(
            "no Step-5-Preview-NVFP4 checkpoint: set STEP5_PREVIEW_CHECKPOINT or stage "
            "Step-5-Preview-NVFP4 under LLM_MODELS_ROOT"
        )
    return path


@pytest.fixture(scope="module")
def loaded(checkpoint) -> ModelConfig:
    return ModelConfig.from_pretrained(str(checkpoint), trust_remote_code=False)


# --- the published config.json --------------------------------------------------------


def test_the_published_config_loads_without_remote_code(checkpoint):
    config = load_pretrained_config(str(checkpoint), trust_remote_code=False)
    assert isinstance(config, Step5Config)
    assert config.architectures == ["Step5ForConditionalGeneration"]
    assert config.text_config.architectures == ["Step5ForCausalLM"]
    assert config.torch_dtype == torch.bfloat16
    assert config.text_config.torch_dtype == torch.bfloat16
    assert (config.image_token_id, config.image_token_len) == (128001, 169)
    assert config.understand_projector_stride == 2
    vision = config.vision_config
    assert (vision.layers, vision.width, vision.heads, vision.patch_size, vision.image_size) == (
        47,
        1536,
        16,
        14,
        728,
    )
    assert vision.hidden_act == "quick_gelu"


def test_text_fields_the_forward_reads(checkpoint):
    text = load_pretrained_config(str(checkpoint)).text_config
    assert isinstance(text, Step5TextConfig)
    assert (text.num_hidden_layers, text.hidden_size, text.vocab_size) == (92, 4096, 128896)
    assert (text.num_attention_heads, text.num_attention_groups, text.head_dim) == (64, 4, 192)
    assert text.num_key_value_heads == 4
    assert text.intermediate_size == 13824
    assert text.rms_norm_eps == 1e-5 and text.norm_dtype == "float32"
    assert text.fp32_residual_connection is True
    assert text.use_head_wise_attn_gate is True
    assert text.eos_token_id == [1, 2]

    # Layer kinds: the 92 decoder layers alone, the 3 MTP layers kept aside.
    assert len(text.layer_types) == _DECODER_LAYERS
    assert len(text.mtp_layer_types) == _MTP_LAYERS
    full = tuple(i for i, kind in enumerate(text.layer_types) if kind == "full_attention")
    assert full == _FULL_LAYERS
    assert text.layer_types.count("sliding_attention") == 69
    assert text.sliding_window == 512
    assert [text.attention_window_for(i) for i in (0, 1, 2, 3)] == [512, 512, 512, None]
    assert [text.is_sparse_layer(i) for i in (0, 3, 91)] == [False, True, True]

    # Per-layer RoPE: theta 1e4 and full rotation on sliding layers; theta 1e7, a third of
    # head_dim rotated and llama3 scaling on full layers.
    assert len(text.rope_theta) == _DECODER_LAYERS + _MTP_LAYERS
    assert {text.rope_theta_for(i) for i in range(_DECODER_LAYERS) if i not in _FULL_LAYERS} == {
        1e4
    }
    assert {text.rope_theta_for(i) for i in _FULL_LAYERS} == {1e7}
    assert [text.rotary_dim_for(i) for i in (0, 3)] == [192, 64]
    assert text.rope_scaling["rope_type"] == "llama3"
    assert text.rope_scaling["factor"] == 1.0
    assert text.yarn_only_types == ["full_attention"]
    assert text.rope_scaling_for(0) is None
    assert text.rope_scaling_for(3) == text.rope_scaling

    sparse = text.sparse_config
    assert sparse["topk"] == 512 and sparse["region_block_size"] == 8
    assert sparse["proxy_dim"] == 256 and sparse["sparse_indexer_rope_dim"] == 32
    assert sparse["sparse_indexer_num_heads"] == 16 and sparse["num_provider_groups"] == 4
    assert sparse["attention_impl"] == "sparse_gqa"

    # MLP: dense SwiGLU on layers 0-2 and 91, routed experts on 3-90.
    assert tuple(text.moe_layer_list) == _MOE_LAYERS
    assert [text.is_moe_layer(i) for i in (0, 2, 3, 90, 91)] == [False, False, True, True, False]
    assert (text.moe_num_experts, text.moe_top_k, text.num_experts_per_tok) == (352, 8, 8)
    assert (text.moe_intermediate_size, text.share_expert_dim) == (1536, 1536)
    assert text.moe_router_scaling_factor == 3.0
    assert text.norm_expert_weight is True
    assert text.use_moe_router_bias is True
    assert text.need_fp32_gate is True
    assert {text.swiglu_limit_for(i) for i in range(_DECODER_LAYERS)} == {7.0}
    assert {text.swiglu_limit_shared_for(i) for i in range(_DECODER_LAYERS)} == {7.0}
    assert text.num_nextn_predict_layers == _MTP_LAYERS


def test_the_engine_sees_92_layers_4_kv_heads_and_no_speculation(loaded):
    """The aliases the engine reads off the composite, and the MTP layers left unbuilt."""
    pretrained = loaded.pretrained_config
    assert loaded.spec_config is None
    assert loaded.get_num_attention_layers() == _DECODER_LAYERS
    assert pretrained.num_hidden_layers == _DECODER_LAYERS
    assert pretrained.num_key_value_heads == 4
    assert pretrained.head_dim == 192
    assert pretrained.vocab_size == 128896
    assert pretrained.torch_dtype == torch.bfloat16


def test_nvfp4_covers_the_routed_experts_of_layers_3_to_87_only(checkpoint, loaded):
    quant = loaded.quant_config
    assert quant.quant_algo == QuantAlgo.NVFP4
    assert quant.group_size == 16
    assert quant.kv_cache_quant_algo is None
    assert get_step5_nvfp4_expert_layers(loaded.pretrained_config, quant) == _NVFP4_EXPERT_LAYERS

    # The exclusions agree with the tensors: exactly those layers carry expert scales.
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    scaled = {
        int(name.split(".")[2])
        for name in index["weight_map"]
        if name.endswith(".moe.gate_proj.weight_scale")
    }
    assert tuple(sorted(scaled)) == _NVFP4_EXPERT_LAYERS

    excluded = quant.is_module_excluded_from_quantization
    for layer in range(_DECODER_LAYERS):
        assert excluded(f"model.layers.{layer}.self_attn"), layer
        if layer in _MOE_LAYERS:
            assert excluded(step5_routed_experts_module(layer)) == (
                layer not in _NVFP4_EXPERT_LAYERS
            ), layer
            assert excluded(f"model.layers.{layer}.moe.gate"), layer
            assert excluded(f"model.layers.{layer}.share_expert"), layer
        else:
            assert excluded(f"model.layers.{layer}.mlp"), layer
    for name in (
        "lm_head",
        "model.embed_tokens",
        "model.norm",
        "vision_model",
        "vit_large_projector",
    ):
        assert excluded(name), name


def test_no_lm_head_quantization(loaded):
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

    assert DecoderModelForCausalLM._resolve_lm_head_quant_config(loaded) is None


def test_auto_config_resolves_without_remote_code(checkpoint):
    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=False)
    assert isinstance(config, Step5Config)


def test_the_config_serializes_in_the_checkpoint_shape(checkpoint):
    text = load_pretrained_config(str(checkpoint)).text_config
    published = json.loads((checkpoint / "config.json").read_text())["text_config"]
    serialized = text.to_dict()
    assert serialized["layer_types"] == published["layer_types"]
    assert serialized["rope_scaling"] == published["rope_scaling"]
    assert serialized["rope_theta"] == published["rope_theta"]
    assert Step5TextConfig(**serialized).layer_types == text.layer_types


# --- the family predicate --------------------------------------------------------------


def _step5_config(**text_overrides) -> Step5Config:
    """A composite of the checkpoint's shape, as the loader builds it."""
    total = _DECODER_LAYERS + _MTP_LAYERS
    kinds = ["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(total)]
    text = dict(
        architectures=["Step5ForCausalLM"],
        layer_types=kinds,
        rope_theta=[1e7 if kind == "full_attention" else 1e4 for kind in kinds],
        partial_rotary_factors=[1 / 3 if kind == "full_attention" else 1.0 for kind in kinds],
        rope_scaling={"rope_type": "llama3", "factor": 1.0},
        yarn_only_types=["full_attention"],
        moe_layer_list=list(_MOE_LAYERS),
        swiglu_limits=[7.0] * total,
        swiglu_limits_shared=[7.0] * total,
        num_nextn_predict_layers=_MTP_LAYERS,
        sparse_config=dict(
            enabled=True,
            proxy_dim=256,
            sparse_indexer_rope_dim=32,
            sparse_indexer_num_heads=16,
            sparse_indexer_num_k_heads=1,
            num_provider_groups=4,
            topk=512,
            region_block_size=8,
            compression_method="csa_block_compress",
            attention_impl="sparse_gqa",
            apply_to_layer_types=["full_attention"],
        ),
        torch_dtype="bfloat16",
    )
    text.update(text_overrides)
    config = Step5Config(
        architectures=["Step5ForConditionalGeneration"],
        text_config=text,
        vision_config=dict(image_size=728, patch_size=14, width=1536, layers=47, heads=16),
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    config.torch_dtype = torch.bfloat16
    return config


def test_is_step5():
    composite = _step5_config()
    assert is_step5(composite)
    assert is_step5(composite.text_config)
    assert is_step5(PretrainedConfig(model_type="step5_vl"))
    assert get_step5_text_config(composite) is composite.text_config
    assert get_step5_text_config(composite.text_config) is composite.text_config
    assert not is_step5(PretrainedConfig(architectures=["Qwen4ExpForConditionalGeneration"]))
    assert not is_step5(PretrainedConfig(architectures=["Step3p7ForConditionalGeneration"]))


def test_per_layer_lists_must_cover_the_decoder():
    with pytest.raises(ValueError, match="layer_types"):
        Step5TextConfig(num_hidden_layers=92, layer_types=["sliding_attention"] * 91)
    with pytest.raises(ValueError, match="rope_theta"):
        Step5TextConfig(num_hidden_layers=92, rope_theta=[1e4] * 91)


# --- the routing table -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if on a GB300, wherever the test runs; one NVLink domain."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *a, **k: True))


_MEMBERS = {
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
}


def _model_config(pretrained=None, spec_config=None, weights=QuantAlgo.NVFP4, **mapping_kwargs):
    return ModelConfig(
        pretrained_config=_step5_config() if pretrained is None else pretrained,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=weights, kv_cache_quant_algo=None),
    )


def _llm_args(mode: str, kv_dtype: str = "auto") -> TorchLlmArgs:
    return TorchLlmArgs(
        model="/tmp/dummy_model", modeling_v2=mode, kv_cache_config=KvCacheConfig(dtype=kv_dtype)
    )


def _declined(config, match: str, kv_dtype: str = "auto") -> None:
    """``auto`` falls back and ``require`` names the criterion that missed."""
    assert modeling_v2_resolve(config, _llm_args("auto", kv_dtype)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", kv_dtype))


@pytest.mark.parametrize("kv_dtype", ["auto", "fp8", "nvfp4"])
@pytest.mark.parametrize("member", sorted(_MEMBERS))
def test_the_target_claims_its_members(member, kv_dtype):
    config = _model_config(**_MEMBERS[member])
    target = modeling_v2_resolve(config, _llm_args("require", kv_dtype))
    assert (target.__module__, target.__name__) == (_TARGET, _TARGET_CLASS)


def test_the_resolved_class_is_the_one_the_loader_builds():
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM

    config = _model_config(**_MEMBERS["tep4"])
    target = modeling_v2_resolve(config, _llm_args("require"))
    config = ModelConfig(
        pretrained_config=config.pretrained_config,
        mapping=config.mapping,
        quant_config=config.quant_config,
        modeling_v2_target=target,
    )
    assert AutoModelForCausalLM._resolve_class(config) is target


@pytest.mark.parametrize("capability", [(10, 0), (9, 0), (12, 0)])
def test_other_gpus_are_declined(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    _declined(
        _model_config(**_MEMBERS["tep4"]), match=re.escape(f"compute capability {capability}")
    )


@pytest.mark.parametrize(
    "text_overrides",
    [
        dict(num_hidden_layers=61),
        dict(hidden_size=7168),
        dict(moe_num_experts=256),
        dict(num_attention_groups=8),
        dict(sliding_window=1024),
    ],
    ids=["layers", "hidden", "experts", "kv_groups", "window"],
)
def test_other_text_shapes_are_declined(text_overrides):
    config = _model_config(_step5_config(**text_overrides), **_MEMBERS["tep4"])
    _declined(config, match="text shape")


def test_another_sparse_attention_is_declined():
    sparse = dict(_step5_config().text_config.sparse_config, topk=2048)
    config = _model_config(_step5_config(sparse_config=sparse), **_MEMBERS["tep4"])
    _declined(config, match="sparse shape")


def test_another_vision_tower_is_declined():
    pretrained = _step5_config()
    pretrained.vision_config.layers = 24
    _declined(_model_config(pretrained, **_MEMBERS["tep4"]), match="vision shape")


def test_a_text_only_config_is_declined():
    text = _step5_config().text_config
    text.architectures = ["Step5ForConditionalGeneration"]
    _declined(_model_config(text, **_MEMBERS["tep4"]), match="text_config/vision_config")


@pytest.mark.parametrize("weights", [None, QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES])
def test_other_weight_formats_are_declined(weights):
    _declined(_model_config(weights=weights, **_MEMBERS["tep4"]), match="weights")


@pytest.mark.parametrize("kv_dtype", ["fp8_ds_mla"])
def test_other_kv_caches_are_declined(kv_dtype):
    _declined(_model_config(**_MEMBERS["tep4"]), match="KV cache", kv_dtype=kv_dtype)


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(world_size=1, tp_size=1),
        dict(world_size=2, tp_size=2),
        dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
        dict(world_size=4, tp_size=4, enable_attention_dp=True),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8, gpus_per_node=4),
        dict(
            world_size=8,
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            enable_attention_dp=True,
            gpus_per_node=4,
        ),
    ],
    ids=["tp1", "tp2", "tep2", "tp4-moetp2-ep2", "dep4-moetp4", "tep8", "dep8"],
)
def test_other_layouts_are_declined(mapping_kwargs):
    _declined(_model_config(**mapping_kwargs), match="parallel layout")


def test_pipeline_parallelism_is_declined():
    config = _model_config(world_size=4, tp_size=2, pp_size=2)
    _declined(config, match="pipeline or context")


def test_an_lm_head_split_under_attention_dp_is_declined():
    config = _model_config(**_MEMBERS["dep4"], enable_lm_head_tp_in_adp=True)
    _declined(config, match="enable_lm_head_tp_in_adp")


@pytest.mark.parametrize(
    "spec_config",
    [
        MTPDecodingConfig(max_draft_len=3),
        NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
    ],
    ids=["mtp", "ngram"],
)
def test_speculative_decoding_is_declined(spec_config):
    """The checkpoint's MTP layers are not built: no draft path routes here."""
    config = _model_config(spec_config=spec_config, **_MEMBERS["tep4"])
    _declined(config, match="speculative decoding")


# --- the routed target -------------------------------------------------------------------


def test_the_target_constructs_on_the_text_config():
    config = _model_config(**_MEMBERS["tep4"])
    target = modeling_v2_resolve(config, _llm_args("require"))
    model_config = _model_config(world_size=1, tp_size=1)
    composite = model_config.pretrained_config
    with torch.device("meta"):
        model = target(model_config)
    assert model.lm_head.weight.shape[-1] == 4096
    # The shell adopts the decoder's config, which the engine sizes the pools from; the
    # core keeps the composite.
    assert model_config.pretrained_config is composite.text_config
    assert model.model.composite_config is composite
