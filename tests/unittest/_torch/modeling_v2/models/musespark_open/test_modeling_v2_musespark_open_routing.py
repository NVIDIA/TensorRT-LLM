# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MuseSparkOpenOmniForCausalLM deployment lands, driven through ``modeling_v2_resolve``.

Synthetic configs, as in test_modeling_v2_routing.py: routing reads the flat
config's shape and layer pattern, the mapping and the compute capability
(patched here, so these run on any machine).

The one target claims Muse-Spark-Open-816B-A42B on sm103, attention
tensor-parallel and the experts expert-parallel over the same ranks, at every
tensor-parallel size and whatever the GPUs per node (launcher state: 1, 2, 4 or
8 visible devices, or the tensor-parallel size under Ray). The speculative-decoding
config never changes the verdict: which fast paths run is the target's own
decision at construction (its compositions carry their topology). Every
other value of a criterion declines -- ``auto`` falls back to the built-in
implementation and ``require`` raises naming every fact routing read.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    modeling_v2_resolve,
)
from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    DFlashDecodingConfig,
    DraftTargetDecodingConfig,
    DSparkDecodingConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_ARCH = "MuseSparkOpenOmniForCausalLM"
_TARGET = (
    "models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.modeling."
    "ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep"
)
_NUM_LAYERS = 62
_NUM_VISION_LAYERS = 50

_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "moe_topk",
    "expert_hidden_size",
    "intermediate_size",
    "vocab_size",
    "sliding_window",
    "every_n_layers_nope",
)
_SHAPE = (62, 8192, 128, 16, 64, 256, 8, 4096, 4096, 202048, 2048, 4)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were an sm_103 device, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _target_config(**overrides) -> MuseSparkOpenConfig:
    """The flat config the engine builds for Muse-Spark-Open-816B-A42B.

    The class defaults are that checkpoint's shape; ``overrides`` go through the
    constructor, so derived fields (expert width, layer pattern) follow them.
    """
    config = MuseSparkOpenConfig(**overrides)
    config.architectures = [_ARCH]
    return config


def _open_format_config_dict() -> dict:
    """The published checkpoint's config.json, reduced to what the open-format translation reads."""
    layer_types, layer_rope_theta = [], []
    for i in range(_NUM_LAYERS):
        nope = (_NUM_LAYERS - i - 1) % 4 == 0
        layer_types.append("full_attention" if nope else "sliding_attention")
        layer_rope_theta.append(0 if nope else 500000.0)
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "audio_token_id": 200038,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_spark_text",
            "vocab_size": 202048,
            "hidden_size": 8192,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "sliding_window": 2048,
            "mlp_layer_types": ["sparse"] * _NUM_LAYERS,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "intermediate_size": 16384,
            "moe_hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "routed_scaling_factor": 1.0,
            "router_score_division_eps": 1e-15,
            "final_logit_softcapping": 20.0,
            "output_multiplier": 0.17677669529663687,
            "qk_scale_factor": 5.4730064863875,
            "residual_gate_temperature": 0.3,
            "tie_word_embeddings": False,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "pad_token_id": 200018,
        },
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_size": 1536,
            "num_hidden_layers": _NUM_VISION_LAYERS,
            "num_attention_heads": 16,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
            "layer_types": [
                "full_attention"
                if (i + 1) % 4 == 0 or i == _NUM_VISION_LAYERS - 1
                else "window_attention"
                for i in range(_NUM_VISION_LAYERS)
            ],
        },
        "audio_config": {
            "model_type": "muse_spark_audio",
            "patch_sizes": [16, 6, 5, 4],
            "hidden_sizes": [256, 512, 1024, 2048],
            "depths": [2, 2, 2, 2],
            "num_hidden_layers": 4,
            "conv_kernel_size": 7,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "out_hidden_size": 8192,
            "rms_norm_eps": 1e-6,
        },
        "quantization_config": {"quant_algo": "NVFP4", "group_size": 16},
    }


def _shape_words(config) -> str:
    """The config shape a NoTarget message names for ``config``."""
    return re.escape(f"config shape {tuple(getattr(config, f, None) for f in _SHAPE_FIELDS)}")


def _llm_args(mode: str = "auto") -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs) -> ModelConfig:
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
    )


# The served mapping: tensor_parallel_size 8, moe_expert_parallel_size 8, on two
# four-GPU trays.
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, gpus_per_node=4)


def _tep(tp: int, gpus_per_node: int = 4) -> dict:
    return dict(world_size=tp, tp_size=tp, moe_ep_size=tp, gpus_per_node=gpus_per_node)


# GPUs per node as the launcher sets it: 1, 2, 4 or 8 visible devices, and under
# Ray the tensor-parallel size itself.
_GPUS_PER_NODE = {"gpn1": 1, "gpn2": 2, "gpn4": 4, "gpn8": 8, "ray": None}


def _dflash(max_draft_len: int = 7) -> DFlashDecodingConfig:
    return DFlashDecodingConfig(max_draft_len=max_draft_len)


# Every speculative-decoding config routing must give the same verdict for.
_SPECULATION = {
    "none": lambda: None,
    "dflash3": lambda: _dflash(3),
    "dflash7": lambda: _dflash(7),
    "dflash8": lambda: _dflash(8),
    "dflash7-trtllm-attention": lambda: DFlashDecodingConfig(
        max_draft_len=7, attention_backend="TRTLLM"
    ),
    "dspark": lambda: DSparkDecodingConfig(max_draft_len=7),
    "suffix-automaton": lambda: SADecodingConfig(max_draft_len=4),
    "ngram": lambda: NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
    "draft-target": lambda: DraftTargetDecodingConfig(
        max_draft_len=3, speculative_model=_DUMMY_MODEL
    ),
    "eagle3": lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model=_DUMMY_MODEL),
    "mtp": lambda: MTPDecodingConfig(max_draft_len=3),
}


def test_the_flat_defaults_are_the_816b_shape():
    """The fixture builds the shape route claims, read the way route reads it."""
    assert tuple(getattr(_target_config(), f) for f in _SHAPE_FIELDS) == _SHAPE


def test_tep8_is_expert_parallel_without_attention_dp():
    """The served mapping is the layout the target was written for."""
    mapping = Mapping(**_TEP8)
    assert (
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.enable_attention_dp,
        mapping.gpus_per_node,
        mapping.is_multi_node(),
    ) == (1, 8, False, 4, True)


@pytest.mark.parametrize("mode", [None, "off"], ids=["default", "off"])
def test_off_resolves_nothing(mode):
    args = TorchLlmArgs(model=_DUMMY_MODEL) if mode is None else _llm_args(mode)
    config = _model_config(_target_config(), _dflash(), **_TEP8)
    assert modeling_v2_resolve(config, args) is None


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("gpus", list(_GPUS_PER_NODE))
@pytest.mark.parametrize("speculation", list(_SPECULATION), ids=list(_SPECULATION))
@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16], ids=lambda tp: f"tep{tp}")
def test_816b_tep_matches_at_every_size_speculation_and_node_size(mode, gpus, speculation, tp):
    gpus_per_node = _GPUS_PER_NODE[gpus] or tp
    config = _model_config(_target_config(), _SPECULATION[speculation](), **_tep(tp, gpus_per_node))
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TARGET


class _Unreadable:
    """A speculative config whose every attribute read fails."""

    def __getattr__(self, name):
        raise AssertionError(f"routing read spec_config.{name}")


def test_route_never_reads_the_speculative_config():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open import routing

    target = routing.route(_target_config(), Mapping(**_TEP8), _SM103, _Unreadable())
    assert _qualified(target) == _TARGET
    with pytest.raises(NoTarget):
        routing.route(_target_config(), Mapping(**_TEP8), (10, 0), _Unreadable())


def test_the_published_checkpoint_config_reaches_the_router_key(tmp_path):
    """The checkpoint publishes another architecture name; the config loader's
    open-format branch writes the router key and the flat fields route reads."""
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    (tmp_path / "config.json").write_text(json.dumps(_open_format_config_dict()))
    pretrained_config = load_pretrained_config(str(tmp_path))

    assert pretrained_config.architectures == [_ARCH]
    config = _model_config(pretrained_config, _dflash(), **_TEP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


def test_the_target_keeps_the_builtin_class_contracts():
    """The engine asks these of the class before any instance exists."""
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
        MuseSparkOpenOmniForConditionalGeneration,
    )

    config = _model_config(_target_config(), _dflash(), **_TEP8)
    target = modeling_v2_resolve(config, _llm_args("require"))
    builtin = MuseSparkOpenOmniForConditionalGeneration
    assert issubclass(target, builtin)
    for hook in (
        "get_preferred_kv_cache_manager_version",
        "requires_kv_cache_manager_v2",
        "get_preferred_transceiver_runtime",
    ):
        assert getattr(target, hook)(None) == getattr(builtin, hook)(None), hook


# Each case changes one fact of the claimed deployment: (config overrides, mapping,
# compute capability, what the NoTarget message must name).
_FOREIGN = {
    "tp8-moe-tp8": (
        {},
        dict(world_size=8, tp_size=8, gpus_per_node=4),
        _SM103,
        "moe_ep=1 moe_tp=8",
    ),
    "tp8-ep4-moe-tp2": (
        {},
        dict(world_size=8, tp_size=8, moe_ep_size=4, moe_tp_size=2, gpus_per_node=4),
        _SM103,
        "moe_ep=4 moe_tp=2",
    ),
    "attention-dp": (
        {},
        dict(_TEP8, enable_attention_dp=True),
        _SM103,
        "attention_dp=True",
    ),
    "pp2": (
        {},
        dict(world_size=16, tp_size=8, pp_size=2, moe_ep_size=8, gpus_per_node=4),
        _SM103,
        "ws=16 tp=8 pp=2",
    ),
    "sm100": ({}, _TEP8, (10, 0), r"compute capability \(10, 0\)"),
    "sm90": ({}, _TEP8, (9, 0), r"compute capability \(9, 0\)"),
    "fewer-layers": (dict(num_hidden_layers=58), _TEP8, _SM103, None),
    "narrower": (dict(hidden_size=6144), _TEP8, _SM103, None),
    "fewer-heads": (dict(num_attention_heads=64, num_key_value_heads=8), _TEP8, _SM103, None),
    "fewer-experts": (dict(moe={"num_experts": 128}), _TEP8, _SM103, None),
    "full-width-experts": (dict(moe={"compression_factor": 1}), _TEP8, _SM103, None),
    "other-window": (dict(global_attn_cfg=[4096, 4096, 4096, 0]), _TEP8, _SM103, None),
}

# Every fact routing reads, as the NoTarget message names it.
_FACTS = (
    "compute capability",
    "config shape",
    "ws=",
    "tp=",
    "pp=",
    "cp=",
    "moe_ep=",
    "moe_tp=",
    "attention_dp=",
)


@pytest.mark.parametrize("speculation", list(_SPECULATION), ids=list(_SPECULATION))
@pytest.mark.parametrize("case", list(_FOREIGN), ids=list(_FOREIGN))
def test_a_foreign_deployment_declines_whatever_the_speculation(monkeypatch, case, speculation):
    overrides, mapping_kwargs, capability, missed = _FOREIGN[case]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    pretrained_config = _target_config(**overrides)
    config = _model_config(pretrained_config, _SPECULATION[speculation](), **mapping_kwargs)

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(
        ValueError, match="no target fits 'MuseSparkOpenOmniForCausalLM'"
    ) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert re.search(missed or _shape_words(pretrained_config), message)
    for fact in _FACTS:
        assert fact in message, fact
    assert "speculative" not in message
    assert "GPUs per node" not in message


def test_a_config_without_the_flat_fields_declines():
    """Another assembly under the same architecture name reads as None, not as an error."""
    from transformers import PretrainedConfig

    pretrained_config = PretrainedConfig(architectures=[_ARCH])
    config = _model_config(pretrained_config, None, **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=re.escape(f"config shape {(None,) * len(_SHAPE)}")):
        modeling_v2_resolve(config, _llm_args("require"))


class _Layout(SimpleNamespace):
    """A mapping as routing reads it; reading the GPUs per node fails."""

    @property
    def gpus_per_node(self):
        raise AssertionError("routing read mapping.gpus_per_node")


def _layout(**overrides) -> _Layout:
    """A mapping as routing reads it: the claimed tep8 layout with ``overrides``."""
    fields = dict(
        world_size=8,
        tp_size=8,
        pp_size=1,
        cp_size=1,
        moe_ep_size=8,
        moe_tp_size=1,
        enable_attention_dp=False,
    )
    fields.update(overrides)
    return _Layout(**fields)


@pytest.mark.parametrize(
    "overrides, claimed",
    [
        (dict(), True),
        (dict(world_size=4, tp_size=4, moe_ep_size=4), True),
        (dict(world_size=1, tp_size=1, moe_ep_size=1), True),
        (dict(world_size=16, tp_size=8, cp_size=2), False),
        (dict(world_size=16, tp_size=8, pp_size=2), False),
        (dict(world_size=16), False),
        (dict(moe_ep_size=4), False),
        (dict(moe_ep_size=4, moe_tp_size=2), False),
        (dict(enable_attention_dp=True), False),
    ],
    ids=["tep8", "tep4", "tep1", "cp2", "pp2", "ws16", "ep4", "ep4-tp2", "adp"],
)
def test_route_claims_tep_at_any_size_only(overrides, claimed):
    """Each layout fact on its own, read off the mapping attributes route reads."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open import routing

    layout = _layout(**overrides)
    for spec_config in [factory() for factory in _SPECULATION.values()] + [_Unreadable()]:
        if claimed:
            assert (
                _qualified(routing.route(_target_config(), layout, _SM103, spec_config)) == _TARGET
            )
        else:
            with pytest.raises(NoTarget) as excinfo:
                routing.route(_target_config(), layout, _SM103, spec_config)
            for fact in _FACTS:
                assert fact in str(excinfo.value), fact
