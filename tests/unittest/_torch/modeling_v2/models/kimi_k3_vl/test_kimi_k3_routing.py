# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routing (``models/kimi_k3_vl/routing.py``), driven by synthetic configs on CPU.

No checkpoint weights: ``route`` reads the config shape, the quantization, the mapping, the SM version, the LLM API
arguments, the environment and, with a standalone DSpark drafter, the drafter checkpoint's ``config.json``. Each case
checks that a configuration the targets do not serve resolves to Modeling V1 under ``auto`` and names the requirement
under ``require``, and that ``route`` and the targets apply the same checks (``drafter_config``, ``environment``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    modeling_v2_resolve,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config,
    environment,
)
from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.llm_args import (
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeLoadBalancerConfig,
    NGramDecodingConfig,
    SADecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
# A speculative mode no target implements.
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)
_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True, params=[(10, 0), (10, 3)], ids=["sm100", "sm103"])
def _on_sm10x(request, monkeypatch):
    """Route as if this were each Blackwell GPU, wherever the test actually runs, on GB200 / GB300 trays of one
    NVLink domain. The cases that need a GB200 pin sm 10.0 (``_on_sm100``)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: request.param)
    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *a, **k: True))
    return request.param


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it."""
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    """The loaded config: the Kimi K3 checkpoint declares no KV-cache quantization (a bf16 pool)."""
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec_config,
        quant_config=QuantConfig(kv_cache_quant_algo=None),
    )


_SM100 = (10, 0)
_TP16_MOETP4EP4 = dict(world_size=16, tp_size=16, moe_tp_size=4, moe_ep_size=4)
_TP16_MOETP16EP1 = dict(world_size=16, tp_size=16, moe_tp_size=16, moe_ep_size=1)
_K3_TARGET = (
    "models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling."
    "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4"
)
_K3_TARGET_B = (
    "models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling."
    "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1"
)

# The quantization the MXFP4 checkpoint declares in its config.json, inside ``text_config``.
_K3_CHECKPOINT_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "group_size": 32,
                "num_bits": 4,
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "kv_cache_scheme": None,
}

_K3_NVFP4_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "input_activations": {
                "group_size": 16,
                "num_bits": 4,
                "strategy": "tensor_group",
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "group_size": 16,
                "num_bits": 4,
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "ignore": ["re:.*lm_head.*"],
    "kv_cache_scheme": None,
}


def _k3_config(text_as_dict=False, quantization=_K3_CHECKPOINT_QUANTIZATION, **text_overrides):
    """The shape Kimi K3's own config.json declares: a vision-language wrapper whose language model is
    ``text_config``, with the quantization ``KimiK3Config`` surfaces from it."""
    text = dict(
        model_type="kimi_linear",
        num_hidden_layers=93,
        hidden_size=7168,
        num_experts=896,
        routed_expert_hidden_size=3584,
    )
    text.update(text_overrides)
    fields = dict(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
        text_config=text if text_as_dict else PretrainedConfig(**text),
    )
    if quantization is not None:
        fields["quantization_config"] = quantization
    return PretrainedConfig(**fields)


def _k3_model_config(pretrained_config, quant_algo=QuantAlgo.W4A16_MXFP4, **mapping_kwargs):
    """The loaded config, with the quantization the engine resolves from the MXFP4 checkpoint and the routed-expert
    backend it resolves AUTO to for Kimi K3."""
    config = _model_config(pretrained_config, **mapping_kwargs)
    return replace(config, quant_config=QuantConfig(quant_algo=quant_algo), moe_backend="TRTLLM")


def _k3_llm_args(mode="auto", **kv_cache_overrides) -> TorchLlmArgs:
    """The KV cache settings the K3 targets need, which the deployment has to set."""
    kv_cache = dict(tokens_per_block=64, enable_block_reuse=False, use_kv_cache_manager_v2=True)
    kv_cache.update(kv_cache_overrides)
    return _llm_args(mode, kv_cache_config=KvCacheConfig(**kv_cache))


@pytest.fixture
def _on_sm100(monkeypatch):
    """Route as if this were a GB200."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def test_kimi_k3_config_surfaces_the_text_config_quantization():
    config = KimiK3Config(
        text_config=dict(quantization_config=_K3_CHECKPOINT_QUANTIZATION),
        architectures=["KimiK3ForConditionalGeneration"],
    )
    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        config.quantization_config, "TRTLLM"
    )
    assert layer_quant_config is None
    assert quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
    assert quant_config.group_size == 32


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("text_as_dict", [False, True], ids=["text-config", "text-dict"])
@pytest.mark.parametrize(
    "mapping_kwargs, target",
    [(_TP16_MOETP4EP4, _K3_TARGET), (_TP16_MOETP16EP1, _K3_TARGET_B)],
    ids=["tp16_moetp4ep4", "tp16_moetp16ep1"],
)
def test_kimi_k3_targets_match(mode, text_as_dict, mapping_kwargs, target):
    """The tp16_moetp16ep1 target is the experts split 16 ways by tensor, set explicitly."""
    config = _k3_model_config(_k3_config(text_as_dict=text_as_dict), **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args(mode))) == target


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "quantization",
    [_K3_NVFP4_QUANTIZATION, None],
    ids=["nvfp4", "unquantized"],
)
def test_kimi_k3_other_quantizations_do_not_match(quantization):
    """An NVFP4 declaration and an unquantized checkpoint have the MXFP4 checkpoint's shape; the quantization is
    what keeps them out of a target whose loader reads packed MXFP4 experts."""
    config = _k3_model_config(_k3_config(quantization=quantization), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match="quantization"):
        modeling_v2_resolve(config, _k3_llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "text_overrides, mapping_kwargs, missed",
    [
        # a Kimi-family checkpoint of another depth
        (dict(num_hidden_layers=61), _TP16_MOETP4EP4, "shape"),
        # the expert split left unset: the mapping resolves to moe_tp 16 x moe_ep 1, but the built-in model runs
        # that default as expert parallelism over the 16 ranks, a layout no target serves
        (dict(), dict(world_size=16, tp_size=16), "split_set=False"),
        # experts split 16 ways by expert, set explicitly
        (dict(), dict(world_size=16, tp_size=16, moe_tp_size=1, moe_ep_size=16), "moe_ep=16"),
        # attention data parallelism splits the requests, not the heads
        (dict(), dict(_TP16_MOETP4EP4, enable_attention_dp=True), "attention_dp=True"),
        (dict(), dict(_TP16_MOETP16EP1, enable_attention_dp=True), "attention_dp=True"),
        # pipeline parallelism over the 16 ranks
        (
            dict(),
            dict(world_size=16, tp_size=8, pp_size=2, moe_tp_size=8, moe_ep_size=1),
            "pp=2",
        ),
        # one tray instead of four
        (dict(), dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4), "ws=4"),
    ],
)
def test_kimi_k3_near_misses_do_not_match(text_overrides, mapping_kwargs, missed):
    config = _k3_model_config(_k3_config(**text_overrides), **mapping_kwargs)
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _k3_llm_args("require"))


def _k3_config_in(dtype):
    """The K3 checkpoint config declaring ``dtype``."""
    config = _k3_config()
    config.torch_dtype = dtype
    return config


def _k3_with(mapping_kwargs=_TP16_MOETP4EP4, quant_algo=QuantAlgo.W4A16_MXFP4, **fields):
    """The K3 model config of ``mapping_kwargs`` with ``fields`` replaced."""
    return replace(_k3_model_config(_k3_config(), quant_algo, **mapping_kwargs), **fields)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "config, llm_args, missed",
    [
        # the tp16_moetp16ep1 target decodes without speculation
        (
            _k3_with(_TP16_MOETP16EP1, spec_config=SADecodingConfig(max_draft_len=2)),
            {},
            "decodes without speculation",
        ),
        # tp16_moetp4ep4 speculates with SA or a standalone DSpark drafter only
        (_k3_with(spec_config=_NGRAM), {}, "supports SA and DSpark"),
        (_k3_with(spec_config=DFlashDecodingConfig(max_draft_len=7)), {}, "supports SA and DSpark"),
        # compute dtype other than bf16
        (
            _k3_with(pretrained_config=_k3_config_in(torch.float16)),
            {},
            "dtype torch.float16",
        ),
        # an fp8 KV cache
        (_k3_with(), dict(dtype="fp8"), "KV cache FP8"),
        # a resolved quantization other than the MXFP4 checkpoint's
        (_k3_with(quant_algo=QuantAlgo.NVFP4), {}, "resolved quantization"),
        # per-layer quantization declarations
        (_k3_with(quant_config_dict={"lm_head": QuantConfig()}), {}, "1 per-layer"),
        # an all-reduce strategy other than AUTO or MNNVL
        (_k3_with(allreduce_strategy=AllReduceStrategy.NCCL), {}, "allreduce_strategy NCCL"),
        # the KV cache block size and reuse the deployment did not set
        (_k3_with(), dict(tokens_per_block=32), "tokens_per_block 32"),
        (_k3_with(), dict(enable_block_reuse=True), "enable_block_reuse true"),
        (_k3_with(), dict(use_kv_cache_manager_v2=False), "use_kv_cache_manager_v2 False"),
        (_k3_with(), dict(mamba_ssm_cache_dtype="bfloat16"), "mamba_ssm_cache_dtype bfloat16"),
        # a routed-expert backend the targets' MoE runtime does not run
        (_k3_with(moe_backend="DEEPGEMM"), {}, "moe_config.backend DEEPGEMM"),
        (_k3_with(moe_backend="CUTLASS"), {}, "moe_config.backend CUTLASS"),
        # EPLB
        (_k3_with(moe_load_balancer=MoeLoadBalancerConfig()), {}, "load_balancer"),
    ],
    ids=[
        "b-speculation",
        "a-ngram",
        "a-dflash",
        "fp16",
        "fp8-kv",
        "nvfp4-resolved",
        "per-layer-quant",
        "nccl-allreduce",
        "tokens-per-block-32",
        "block-reuse",
        "kv-manager-v1",
        "bf16-state",
        "deepgemm-moe",
        "cutlass-moe",
        "eplb",
    ],
)
def test_kimi_k3_unmet_settings_do_not_match(config, llm_args, missed):
    """Settings the targets check when built or on their first step; route checks them first so ``auto`` falls
    back instead of failing."""
    assert modeling_v2_resolve(config, _k3_llm_args("auto", **llm_args)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _k3_llm_args("require", **llm_args))


# RadixArk-Kimi-K3-DSpark's config.json, one layer: a Qwen3-shaped GQA drafter whose heads and MLP split over TP16
# (6 query heads and 1 KV head per rank), with the DSpark heads. The hidden size is the heads' width, which other
# backbones' config classes check.
_K3_GQA_DRAFTER = dict(
    architectures=["DFlashDraftModel"],
    model_type="qwen3",
    hidden_size=6144,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=96,
    num_key_value_heads=16,
    head_dim=64,
    vocab_size=1024,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    rope_theta=10000.0,
    max_position_embeddings=4096,
    attention_bias=False,
    use_sliding_window=False,
    dflash_config={"block_size": 8, "target_layer_ids": [1, 46, 91], "mask_token_id": 1000},
    markov_rank=4,
    markov_head_type="vanilla",
)

# Inferact-Kimi-K3-DSpark's config.json at small widths: an MLA drafter (``k3_dspark``) with a YaRN RoPE.
_K3_MLA_DRAFTER = dict(
    architectures=["K3DSparkModel"],
    model_type="k3_dspark",
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=64,
    num_key_value_heads=64,
    q_lora_rank=64,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    vocab_size=1024,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
    rope_scaling={
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "mscale_all_dim": 1.0,
    },
    markov_rank=4,
    markov_head_type="vanilla",
)


def _k3_drafter_checkpoint(path, config=None, weight_key="model.layers.0.weight"):
    """A drafter checkpoint directory at ``path``: its weight index (``weight_key``; an ``mtp.<n>.`` key marks the
    draft embedded in the target) and, unless ``config`` is None, its ``config.json``."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: "model-00001.safetensors"}})
    )
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _k3_dspark(path, **spec_fields):
    """The tp16_moetp4ep4 model config with DSpark on the drafter checkpoint at ``path``."""
    return _k3_with(
        spec_config=DSparkDecodingConfig(max_draft_len=7, speculative_model=path, **spec_fields)
    )


def _k3_declines(config, missed):
    """``config`` routes to no target: Modeling V1 under ``auto``, ``missed`` in the error under ``require``."""
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _k3_llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
def test_kimi_k3_dspark_embedded_draft_does_not_match(tmp_path):
    """DSpark's draft is embedded in the target when the drafter checkpoint's weight index holds ``mtp.<n>.`` keys;
    tp16_moetp4ep4 builds its own drafter from a standalone checkpoint only."""
    path = _k3_drafter_checkpoint(tmp_path, _K3_GQA_DRAFTER, weight_key="mtp.0.layers.0.weight")
    _k3_declines(_k3_dspark(path), "draft embedded in the target")


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "drafter", [_K3_GQA_DRAFTER, _K3_MLA_DRAFTER], ids=["gqa drafter", "mla drafter"]
)
def test_kimi_k3_dspark_standalone_drafters_match(tmp_path, drafter):
    """A standalone drafter checkpoint either of the target's drafters builds: the GQA drafter (RadixArk) or the MLA
    drafter (Inferact)."""
    config = _k3_dspark(_k3_drafter_checkpoint(tmp_path, drafter))
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args("require"))) == _K3_TARGET


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "overrides, missed",
    [
        (dict(model_type="llama"), "backbone is 'llama', not a Qwen3-shaped GQA backbone"),
        (dict(use_sliding_window=True, sliding_window=128), "sliding-window layers"),
        (
            dict(dflash_config={**_K3_GQA_DRAFTER["dflash_config"], "use_swa": True}),
            "sliding windows or DFlash 2 features",
        ),
        (
            dict(dflash_config={**_K3_GQA_DRAFTER["dflash_config"], "causal": True}),
            "the drafter is causal",
        ),
        (dict(attention_bias=True), "projections have biases"),
        (dict(hidden_act="gelu"), "MLP activation is gelu"),
        (dict(markov_head_type="gated"), "Markov head is 'gated', not vanilla"),
        (
            dict(intermediate_size=520),
            r"intermediate_size \(520\) does not split evenly over TP 16",
        ),
        (
            dict(num_key_value_heads=8),
            r"num_key_value_heads \(8\) does not split evenly over TP 16",
        ),
        (
            dict(dflash_config={**_K3_GQA_DRAFTER["dflash_config"], "output_multiplier": -1.0}),
            "output_multiplier must be positive and finite",
        ),
    ],
    ids=[
        "llama backbone",
        "sliding window",
        "dflash2 swa",
        "causal",
        "bias",
        "gelu",
        "gated markov head",
        "intermediate not over tp16",
        "kv heads not over tp16",
        "negative logit multiplier",
    ],
)
def test_kimi_k3_dspark_drafters_the_target_does_not_build_do_not_match(
    tmp_path, overrides, missed
):
    """A standalone drafter checkpoint the target's drafters raise ``NoTarget`` for is declined in ``route``, read
    from the checkpoint's config.json as the engine loads it."""
    path = _k3_drafter_checkpoint(tmp_path, {**_K3_GQA_DRAFTER, **overrides})
    _k3_declines(_k3_dspark(path), "does not build this drafter: .*" + missed)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize("missing", ["config", "directory"])
def test_kimi_k3_dspark_unreadable_drafter_does_not_match(tmp_path, missing):
    """A drafter config that cannot be loaded is declined, so under ``auto`` Modeling V1 reports the load error."""
    if missing == "config":
        path = _k3_drafter_checkpoint(tmp_path)
    else:
        path = str(tmp_path / "absent")
    _k3_declines(_k3_dspark(path), "drafter config at .* cannot be loaded")


# Each check of drafter_config a drafter constructor raises for, the drafter config it is probed with, the attention
# backend, and what the constructor raises. The MLA drafter's later checks run after its block-decode backend
# resolves, so the CuTe DSL probe is patched to pass.
_SHARED_CHECKS = [
    ("gqa_declines", _K3_GQA_DRAFTER, "AUTO", NoTarget),
    ("gqa_tp_declines", _K3_GQA_DRAFTER, "AUTO", NoTarget),
    ("gqa_unknown_attention_backend", _K3_GQA_DRAFTER, "AUTO", ValueError),
    ("gqa_attention_backend_declines", _K3_GQA_DRAFTER, "AUTO", NoTarget),
    ("mla_declines", _K3_MLA_DRAFTER, "AUTO", NoTarget),
    ("mla_cp_declines", _K3_MLA_DRAFTER, "AUTO", NotImplementedError),
    ("mla_tp_declines", _K3_MLA_DRAFTER, "AUTO", ValueError),
    ("mla_unknown_attention_backend", _K3_MLA_DRAFTER, "AUTO", ValueError),
    ("mla_markov_head_declines", _K3_MLA_DRAFTER, "AUTO", ValueError),
    ("mla_causal_declines", _K3_MLA_DRAFTER, "AUTO", ValueError),
]


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "predicate, drafter_fields, backend, raised",
    _SHARED_CHECKS,
    ids=[check[0] for check in _SHARED_CHECKS],
)
def test_kimi_k3_route_and_the_drafters_apply_the_same_predicates(
    tmp_path, monkeypatch, predicate, drafter_fields, backend, raised
):
    """``route`` and the drafter constructors read the drafter config through the same functions of the family's
    ``drafter_config``: one patched to refuse is seen by both, and the constructor raises what it raises for it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        drafter,
        drafter_mla,
    )

    monkeypatch.setattr(drafter_config, predicate, lambda *args: "the probe declines")
    monkeypatch.setattr(drafter_mla._mla_ops, "mla_decode_unavailability_reason", lambda: None)
    path = _k3_drafter_checkpoint(tmp_path, drafter_fields)
    _k3_declines(_k3_dspark(path), "does not build this drafter: the probe declines")

    pretrained_config = drafter_config.load_pretrained_config(path, trust_remote_code=True)
    drafter_cls = (
        drafter_mla.K3MLADSparkDrafter
        if drafter_config.is_mla(pretrained_config)
        else drafter.K3DSparkDrafter
    )
    with pytest.raises(raised, match="the probe declines"):
        drafter_cls(
            ModelConfig(pretrained_config=pretrained_config), dflash_attention_backend=backend
        )


@pytest.mark.usefixtures("_on_sm100")
def test_kimi_k3_route_and_the_mla_drafter_read_the_same_rope(tmp_path, monkeypatch):
    """The MLA drafter's RoPE check: ``route`` and the drafter's ``rope_params`` read the RoPE settings through
    ``drafter_config``'s functions."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        drafter_mla,
    )

    path = _k3_drafter_checkpoint(tmp_path, _K3_MLA_DRAFTER)
    monkeypatch.setattr(drafter_config, "mla_rope_type", lambda scaling: "linear")
    _k3_declines(_k3_dspark(path), "rope_type='linear'; only 'yarn' is supported")
    pretrained_config = drafter_config.load_pretrained_config(path, trust_remote_code=True)
    with pytest.raises(ValueError, match="rope_type='linear'; only 'yarn' is supported"):
        drafter_mla.rope_params(pretrained_config, None)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "overrides, missed",
    [
        (dict(num_attention_heads=72), r"num_attention_heads=72, not divisible by tp_size 16"),
        (dict(intermediate_size=520), r"intermediate_size=520, not divisible by tp_size 16"),
        (dict(markov_head_type="gated"), "markov_head_type='gated'; only 'vanilla' is supported"),
        (dict(causal=True), "sets causal=true; the block decode is non-causal"),
        (
            dict(rope_scaling={"type": "linear", "factor": 2.0}),
            "rope_type='linear'; only 'yarn' is supported",
        ),
        (
            dict(rope_scaling={"type": "yarn", "original_max_position_embeddings": 4096}),
            "YaRN RoPE declares no 'factor'",
        ),
        (
            dict(dflash_config={"final_logit_softcapping": -2.0}),
            "final_logit_softcapping must be non-negative and finite",
        ),
    ],
    ids=[
        "heads not over tp16",
        "mlp not over tp16",
        "gated markov head",
        "causal",
        "linear rope",
        "yarn no factor",
        "negative softcap",
    ],
)
def test_kimi_k3_dspark_mla_drafters_the_target_does_not_build_do_not_match(
    tmp_path, overrides, missed
):
    """A standalone MLA drafter checkpoint the target's MLA drafter refuses at construction is declined in
    ``route``."""
    path = _k3_drafter_checkpoint(tmp_path, {**_K3_MLA_DRAFTER, **overrides})
    _k3_declines(_k3_dspark(path), "does not build this drafter: .*" + missed)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "drafter, probe, backend, missed",
    [
        (
            _K3_GQA_DRAFTER,
            None,
            "CUTEDSL",
            r"K3DSparkDrafter attention backend must be one of \['VANILLA', 'TRTLLM', 'FA4'\], got 'CUTEDSL'",
        ),
        (
            _K3_GQA_DRAFTER,
            "trtllm_gen_unavailability_reason",
            "TRTLLM",
            "attention_backend='TRTLLM' was requested but it is unavailable for K3DSparkDrafter: no probe",
        ),
        (
            _K3_MLA_DRAFTER,
            "mla_paged_decode_unavailability_reason",
            "TRTLLM",
            "attention_backend='TRTLLM' was requested but it is unavailable for K3MLADSparkDrafter: no probe",
        ),
    ],
    ids=["gqa cutedsl", "gqa trtllm unavailable", "mla trtllm unavailable"],
)
def test_kimi_k3_dspark_attention_backends_the_drafters_refuse_do_not_match(
    tmp_path, monkeypatch, drafter, probe, backend, missed
):
    """An attention backend the drafter refuses at construction (an unknown name, or an explicit backend this build
    cannot run) is declined in ``route``."""
    if probe is not None:
        monkeypatch.setattr(drafter_config, probe, lambda *args: "no probe")
    path = _k3_drafter_checkpoint(tmp_path, drafter)
    _k3_declines(_k3_dspark(path, attention_backend=backend), missed)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "drafter, backend",
    [
        (_K3_GQA_DRAFTER, "AUTO"),
        (_K3_GQA_DRAFTER, "VANILLA"),
        (_K3_MLA_DRAFTER, "AUTO"),
        (_K3_MLA_DRAFTER, "CUTEDSL"),
        (_K3_MLA_DRAFTER, "VANILLA"),
    ],
    ids=["gqa auto", "gqa vanilla", "mla auto", "mla cutedsl", "mla vanilla"],
)
def test_kimi_k3_dspark_backends_route_does_not_probe_match(
    tmp_path, monkeypatch, drafter, backend
):
    """Where TRTLLM is not asked for explicitly, ``route`` does not decline on the TRTLLM probes (AUTO falls back to
    VANILLA in the GQA drafter); the CuTe DSL decode and flash-attn are probed only by the constructors."""
    monkeypatch.setattr(
        drafter_config, "trtllm_gen_unavailability_reason", lambda *args: "no probe"
    )
    monkeypatch.setattr(
        drafter_config, "mla_paged_decode_unavailability_reason", lambda: "no probe"
    )
    path = _k3_drafter_checkpoint(tmp_path, drafter)
    config = _k3_dspark(path, attention_backend=backend)
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args("require"))) == _K3_TARGET


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "field",
    ["enable_return_routed_experts", "use_cute_dsl_bf16_gemm", "use_cute_dsl_bf16_bmm"],
)
def test_kimi_k3_unsupported_llm_args_do_not_match(field):
    """Features of the built-in implementation the targets do not carry. The field is set on the built arguments:
    their validator ties the bf16 CuTe DSL switches to the GPU of the machine that runs this test."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    for mode, expected in (("auto", None), ("require", ValueError)):
        llm_args = _k3_llm_args(mode)
        object.__setattr__(llm_args, field, True)
        if expected is None:
            assert modeling_v2_resolve(config, llm_args) is None
        else:
            with pytest.raises(ValueError, match=f"{field} true"):
                modeling_v2_resolve(config, llm_args)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "name, value",
    [
        ("ENABLE_PERFECT_ROUTER", "1"),
        ("EXPERT_STATISTIC_ITER_RANGE", "0-10"),
        ("TRTLLM_LOW_M_GEMM_BACKEND", "auto"),
        ("TLLM_KDA_ENABLE_OPT_PREFILL", "0"),
        ("TLLM_KDA_ENABLE_INDEXED_STATE_POOL", "0"),
        ("FLA_CONV_BACKEND", "cuda"),
    ],
)
def test_kimi_k3_unread_environment_switches_do_not_match(monkeypatch, name, value):
    """Environment switches that move the built-in implementation off the path the targets run."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    monkeypatch.setenv(name, value)
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match=f"environment {name}={value}"):
        modeling_v2_resolve(config, _k3_llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "name, value",
    [("ENABLE_PERFECT_ROUTER", "0"), ("TLLM_KDA_ENABLE_OPT_PREFILL", "1")],
)
def test_kimi_k3_environment_switches_at_their_defaults_match(monkeypatch, name, value):
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    monkeypatch.setenv(name, value)
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args("require"))) == _K3_TARGET


@pytest.mark.usefixtures("_on_sm100")
def test_kimi_k3_default_kv_cache_settings_do_not_match():
    """Without the K3 settings the KV cache config defaults to 32-token blocks with reuse on: no target."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="tokens_per_block 32.*enable_block_reuse true"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "spec_config",
    [SADecodingConfig(max_draft_len=2), None],
    ids=["sa", "none"],
)
def test_kimi_k3_tp16_moetp4ep4_speculation_matches(spec_config):
    config = _k3_with(spec_config=spec_config)
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args("require"))) == _K3_TARGET


def test_kimi_k3_on_another_sm_does_not_match(monkeypatch):
    """Routed as a GB300 (sm 10.3), the K3 targets, sm 10.0 only, match nothing."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _k3_llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "name, value, missed",
    [
        (environment.ATTN_RES_TOPOLOGY, "fused", "KIMI_K3_ATTN_RES_TOPOLOGY must be one of"),
        (
            environment.FUSED_ATTN_RES_MAX_TOKENS,
            "many",
            "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS must be a positive integer, got 'many'",
        ),
        (
            environment.FUSED_ATTN_RES_MAX_TOKENS,
            "0",
            "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS must be >= 1, got 0",
        ),
        (
            environment.MLA_MAX_POSITIONS,
            "1e6",
            "KIMI_K3_MLA_MAX_POSITIONS must be an integer, got '1e6'",
        ),
    ],
    ids=["topology", "max-tokens-text", "max-tokens-zero", "mla-positions"],
)
def test_kimi_k3_environment_values_the_targets_reject_do_not_match(
    monkeypatch, name, value, missed
):
    """A value of a variable the targets read that they would reject when imported or built."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    monkeypatch.setenv(name, value)
    assert modeling_v2_resolve(config, _k3_llm_args("auto")) is None
    with pytest.raises(ValueError, match="environment: " + missed):
        modeling_v2_resolve(config, _k3_llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "values",
    [
        {environment.ATTN_RES_TOPOLOGY: "1", environment.MLA_MAX_POSITIONS: "1048576"},
        {environment.ATTN_RES_TOPOLOGY: "persistent", environment.FUSED_ATTN_RES_MAX_TOKENS: "8"},
        {environment.ATTN_RES_TOPOLOGY: "per_token"},
        {
            environment.AUX_ATTN_RES_STREAM: "0",
            environment.FUSED_ATTN_RES: "0",
            environment.DISABLE_MIN_LATENCY_LATENT_PROJ: "1",
            environment.DISABLE_FUSED_ROUTE_QUANT: "1",
            environment.ROUTER_BF16: "0",
        },
    ],
    ids=["perf-sanity", "persistent", "per-token", "switches"],
)
def test_kimi_k3_environment_values_the_targets_take_match(monkeypatch, values):
    """Values the targets read and accept, including the perf-sanity deployments' settings."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert _qualified(modeling_v2_resolve(config, _k3_llm_args("require"))) == _K3_TARGET


def test_kimi_k3_environment_readers_reject_what_route_declines():
    """``environment.declines`` names a variable exactly when its reader raises."""
    for name, value in (
        (environment.ATTN_RES_TOPOLOGY, "fused"),
        (environment.FUSED_ATTN_RES_MAX_TOKENS, "-3"),
        (environment.MLA_MAX_POSITIONS, "x"),
    ):
        env = {name: value}
        assert len(environment.declines(env)) == 1
        with pytest.raises(ValueError):
            environment.attn_res_topology(env)
            environment.fused_attn_res_max_tokens("per_token", env)
            environment.mla_max_positions(1, env)
    assert environment.declines({}) == []
    assert environment.attn_res_topology({environment.ATTN_RES_TOPOLOGY: "1"}) == "split"
    assert environment.fused_attn_res_max_tokens("split", {}) == 32
    assert environment.fused_attn_res_max_tokens("per_token", {}) == 1
    assert environment.mla_max_positions(7, {}) == 7


_ROUTE_IN_A_FRESH_PROCESS = textwrap.dedent(
    """
    import json
    import sys

    from transformers import PretrainedConfig

    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import routing
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    quantization, text, drafter_path, family = json.loads(sys.argv[1])
    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["KimiK3ForConditionalGeneration"],
            model_type="kimi_k3",
            text_config=PretrainedConfig(**text),
            quantization_config=quantization,
        ),
        mapping=Mapping(world_size=16, tp_size=16, moe_tp_size=4, moe_ep_size=4),
        spec_config=DSparkDecodingConfig(max_draft_len=7, speculative_model=drafter_path),
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4),
        moe_backend="TRTLLM",
    )
    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        modeling_v2="require",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=64, enable_block_reuse=False, use_kv_cache_manager_v2=True
        ),
    )
    try:
        routing.route(config, llm_args, (10, 0))
    except NoTarget as error:
        assert "does not build this drafter" in str(error), error
    else:
        raise AssertionError("route did not decline")
    targets = sorted(name for name in sys.modules if name.startswith(family + ".kimi_k3_mxfp4"))
    print(json.dumps(targets))
    """
)


def test_kimi_k3_route_declines_without_importing_a_target(tmp_path):
    """``route`` declines a standalone DSpark drafter the target does not build (it reads the drafter's config.json
    through the family's ``drafter_config``) without importing either target package, whose ``__init__.py`` loads
    the family's native library. Checked in a fresh interpreter, where no other test has imported a target."""
    path = _k3_drafter_checkpoint(tmp_path, {**_K3_GQA_DRAFTER, "model_type": "llama"})
    text = dict(
        model_type="kimi_linear",
        num_hidden_layers=93,
        hidden_size=7168,
        num_experts=896,
        routed_expert_hidden_size=3584,
    )
    argument = json.dumps([_K3_CHECKPOINT_QUANTIZATION, text, path, _FAMILY])
    result = subprocess.run(
        [sys.executable, "-c", _ROUTE_IN_A_FRESH_PROCESS, argument],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
