# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing of the ModelOpt NVFP4 export and of Rubin (sm_107), on synthetic configs.

The five sm_107 targets (the specialized ``kimi_k31_pro_nvfp4__sm_107__{cep,dep16,helix8,tep8}`` and
the default ``kimi_k31_pro_nvfp4__sm_10x__default``) are stand-ins here, so this exercises the
decision in ``routing.py``, not the model. The ModelOpt configs are shaped as the export's
``config.json`` and ``hf_quant_config.json``; the per-layer recipes go through the engine's ModelOpt
parser, as the loader reads them. ``test_routing.py`` keeps covering the general target and
``test_routing_default_sm107.py`` the default target. The server role routing records reaches the
config the model is built from through the model loader's own config path.
"""

import ast
import copy
import json
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
import yaml
from transformers import PretrainedConfig

from tensorrt_llm import _utils as trtllm_utils
from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    engine_kv_cache_quant_algo,
    modeling_v2_resolve,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31 import routing
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import is_kimi_linear, resolve_auto_ssm_cache_dtype
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MoeLoadBalancerConfig,
    MoePerLayerDynamicLoadBalancerConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_CEP_TARGET = ("kimi_k31_pro_nvfp4__sm_107__cep", "ModelingV2KimiK31ProNvfp4Sm107Cep")
_SPECIALIZED = {
    "cep8": _CEP_TARGET,
    "cep16": _CEP_TARGET,
    "dep16": ("kimi_k31_pro_nvfp4__sm_107__dep16", "ModelingV2KimiK31ProNvfp4Sm107Dep16"),
    # A context server over 12 ranks runs the dep16 target.
    "dep12": ("kimi_k31_pro_nvfp4__sm_107__dep16", "ModelingV2KimiK31ProNvfp4Sm107Dep16"),
    # So does a context server over 8 ranks.
    "dep8": ("kimi_k31_pro_nvfp4__sm_107__dep16", "ModelingV2KimiK31ProNvfp4Sm107Dep16"),
    "helix8": ("kimi_k31_pro_nvfp4__sm_107__helix8", "ModelingV2KimiK31ProNvfp4Sm107Helix8"),
    "tep8": ("kimi_k31_pro_nvfp4__sm_107__tep8", "ModelingV2KimiK31ProNvfp4Sm107Tep8"),
}
_DEFAULT = ("kimi_k31_pro_nvfp4__sm_10x__default", "ModelingV2KimiK31ProNvfp4Sm10xDefault")
_GENERAL = "ModelingV2KimiK31ProMxfp4Sm10xDefault"
# The real role reader; the autouse fixture stands a fixed role in for it.
_DISAGG_ROLE = routing.disagg_role

_DEP16 = dict(world_size=16, tp_size=16, enable_attention_dp=True, moe_ep_size=16)
_HELIX8 = dict(
    world_size=8, tp_size=1, cp_size=8, cp_config={"cp_type": "HELIX"}, moe_tp_size=2, moe_ep_size=4
)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1)


def _ulysses(n, **moe):
    """TP 1 and Ulysses context parallelism over ``n`` ranks; MoE EP ``n`` unless ``moe`` says."""
    moe = moe or dict(moe_ep_size=n, moe_tp_size=1)
    return dict(world_size=n, tp_size=1, cp_size=n, cp_config={"cp_type": "ULYSSES"}, **moe)


_CEP8 = _ulysses(8)
_CEP16 = _ulysses(16)
_DEP12 = dict(world_size=12, tp_size=12, enable_attention_dp=True, moe_ep_size=12)
_DEP8 = dict(world_size=8, tp_size=8, enable_attention_dp=True, moe_ep_size=8)
_LAYOUTS = {
    "cep8": _CEP8,
    "cep16": _CEP16,
    "dep16": _DEP16,
    "helix8": _HELIX8,
    "tep8": _TEP8,
    "dep12": _DEP12,
    "dep8": _DEP8,
}
# The layouts the configurable default target also takes (its attention DP runs powers of two).
_DEFAULT_LAYOUTS = ("dep16", "helix8", "tep8", "dep8")

# --- the ModelOpt NVFP4 export ---
# 8 layers, numbered as linear_attn_config numbers them (from 1): MLA (DSA) on 4 and 8, KDA on
# the rest; layer 0's MLP is dense, the others are MoE.

_TEXT = dict(
    num_hidden_layers=8,
    hidden_size=7168,
    num_experts=896,
    num_experts_per_token=16,
    num_attention_heads=96,
    q_lora_rank=1536,
    kv_lora_rank=512,
    vocab_size=163840,
    moe_intermediate_size=3072,
    routed_expert_hidden_size=3584,
    num_shared_experts=2,
    first_k_dense_replace=1,
    linear_attn_config=dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7]),
    dsa_config=dict(dsa_layers=[4, 8], index_topk=2048, index_num_heads=32, index_head_dim=128),
)
# The export's exclusions, as its hf_quant_config.json lists them.
_EXCLUDE = [
    "*block_sparse_moe.gate*",
    "*block_sparse_moe.routed_expert_up_proj*",
    "*mlp.routed_expert_up_proj*",
    "*block_sparse_moe.routed_expert_down_proj*",
    "*mlp.routed_expert_down_proj*",
    "*block_sparse_moe.routed_expert_norm*",
    "*mlp.routed_expert_norm*",
    "*self_attn.q_conv1d*",
    "*self_attn.k_conv1d*",
    "*self_attn.v_conv1d*",
    "*self_attn.f_b_proj*",
    "*self_attn.o_norm*",
    "*self_attn.q_a_layernorm*",
    "*self_attn.kv_a_layernorm*",
    "*_res_proj*",
    "*_res_norm*",
    "*layernorm*",
    "language_model.model.layers.0.mlp.*",
    "layers.0.mlp.*",
    "model.layers.0.mlp.*",
    "language_model.lm_head",
    "lm_head",
    "vision_tower*",
    "mm_projector*",
]
_PRODUCER = {"name": "modelopt", "version": "0.49.0"}


def _quantized_layers(num_layers=8, mla_layers=(4, 8), shared=True) -> dict:
    """Per-layer recipes under the module spellings the export lists (three prefixes, both MoE
    block names). ``mla_layers`` numbers layers from 1; ``shared`` False leaves the shared experts
    out (BF16)."""
    kda = [i for i in range(num_layers) if i + 1 not in mla_layers]
    layers = {}
    for prefix in ("language_model.model.", "model.", ""):
        for i in range(1, num_layers):
            for block in ("block_sparse_moe", "mlp"):
                layers[f"{prefix}layers.{i}.{block}.experts"] = {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                }
                for proj in ("gate_proj", "up_proj", "gate_up_proj", "down_proj") if shared else ():
                    layers[f"{prefix}layers.{i}.{block}.shared_experts.{proj}"] = {
                        "quant_algo": "MXFP8"
                    }
        for i in kda:
            for proj in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "b_proj", "f_a_proj"):
                layers[f"{prefix}layers.{i}.self_attn.{proj}"] = {"quant_algo": "MXFP8"}
            layers[f"{prefix}layers.{i}.self_attn.fused_qkvg_proj"] = {"quant_algo": "MXFP8"}
            layers[f"{prefix}layers.{i}.self_attn.in_proj_qkvgfab"] = {"quant_algo": "MXFP8"}
    return layers


def _inline_quantization_config(layers: dict) -> dict:
    """config.json's quantization_config: the flat ModelOpt form."""
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "producer": _PRODUCER,
        "group_size": 16,
        "ignore": list(_EXCLUDE),
        "quantized_layers": layers,
    }


def _hf_quant_config(layers: dict) -> dict:
    return {
        "producer": _PRODUCER,
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": list(_EXCLUDE),
            "quantized_layers": layers,
        },
    }


def _modelopt_config(
    tmp_path, mapping_kwargs, spec_config=None, shared=True, attn_backend="TRTLLM"
) -> ModelConfig:
    """The export's config, its recipes read from hf_quant_config.json as the loader reads them."""
    layers = _quantized_layers(shared=shared)
    path = tmp_path / "hf_quant_config.json"
    path.write_text(json.dumps(_hf_quant_config(layers)))
    quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
        str(path), str(tmp_path), "TRTLLM"
    )
    pretrained = PretrainedConfig(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=PretrainedConfig(**_TEXT),
        quantization_config=_inline_quantization_config(layers),
    )
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=quant_config,
        quant_config_dict=layer_quant_config,
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
        attn_backend=attn_backend,
    )


def _dflash2(tmp_path, max_draft_len, **fields) -> DFlashDecodingConfig:
    """One-engine DFlash with a DFlash2 MLA drafter checkpoint of block ``max_draft_len + 1``;
    ``fields`` are further config fields. The drafter history's storage is left unset, as the
    deployments leave it."""
    path = tmp_path / f"dflash2_k{max_draft_len}"
    path.mkdir(exist_ok=True)
    dflash_config = dict(
        block_size=max_draft_len + 1,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=32,
        mask_token_id=163824,
        target_layer_ids=[7, 23, 51, 67, 83, 91],
        attention_mode="mla",
    )
    (path / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["DFlash2DraftModel"], model_type="qwen3", dflash_config=dflash_config
            )
        )
    )
    fields.setdefault("context_kv_dtype", "fp8")
    config = DFlashDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=str(path), **fields
    )
    config.resolve_from_checkpoint()
    return config


def _transceiver() -> CacheTransceiverConfig:
    return CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")


def _llm_args(
    mode="require",
    backend="CUTEDSL_FC12",
    kv_dtype="fp8",
    ssm_dtype=None,
    indexer_k_dtype=None,
    disaggregated=True,
    kv_cache=None,
    moe=None,
    **kwargs,
) -> TorchLlmArgs:
    """``indexer_k_dtype`` names the index-key dtype in the arguments' sparse attention config;
    ``ssm_dtype`` the KDA state dtype; ``kv_cache`` adds kv_cache_config fields and ``moe``
    moe_config fields."""
    sparse = {} if indexer_k_dtype is None else dict(indexer_k_dtype=indexer_k_dtype)
    cache = dict(kv_cache or {})
    if ssm_dtype is not None:
        cache["mamba_ssm_cache_dtype"] = ssm_dtype
    if disaggregated:
        kwargs.setdefault("cache_transceiver_config", _transceiver())
    kwargs.setdefault("disable_mm_encoder", True)
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=KvCacheConfig(dtype=kv_dtype, **cache),
        sparse_attention_config=DeepSeekSparseAttentionConfig(**sparse),
        moe_config=MoeConfig(backend=backend, **(moe or {})),
        **kwargs,
    )


# A helix8 generation server: CUDA graphs for 1, 2, 4, 8 and 16 requests with padding, and the
# overlap scheduler on.
_HELIX8_GRAPHS = dict(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True)
_HELIX8_GEN = dict(cuda_graph_config=_HELIX8_GRAPHS, disable_overlap_scheduler=False)


def _on_gpu(monkeypatch, capability):
    """Report a GPU of ``capability``, also on a host without one: routing reads the capability,
    the sparse attention config's checks read ``get_sm_version()``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(trtllm_utils, "get_sm_version", lambda: capability[0] * 10 + capability[1])


def _stand_in_targets(monkeypatch):
    """Stand-ins for the four sm_107 targets: importing one needs its GPU kernels."""
    for directory, name in (*_SPECIALIZED.values(), _DEFAULT):
        module = types.ModuleType(f"{_FAMILY}.{directory}.modeling")
        setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture(autouse=True)
def rubin(monkeypatch):
    """Route as on Rubin, with stand-ins for the sm_107 targets, on a generation server. The
    default target takes no value here (``test_routing_default_sm107.py`` routes to it)."""
    _on_gpu(monkeypatch, (10, 7))
    _stand_in_targets(monkeypatch)
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: "generation")


def _as_role(monkeypatch, role):
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: role)


def _resolves(config, **args) -> str:
    """The target class name both modes resolve to (they must agree)."""
    names = {
        modeling_v2_resolve(config, _llm_args(mode, **args)).__name__
        for mode in ("auto", "require")
    }
    assert len(names) == 1, names
    return names.pop()


def _refused(config, named, **args) -> None:
    """``auto`` falls back (None), ``require`` raises naming ``named``."""
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match=re.escape(named)):
        modeling_v2_resolve(config, _llm_args("require", **args))


# The tested deployments: layout, role, MoE backend, draft length, arguments.
_DEPLOYMENTS = {
    "cep8-context-k7": ("cep8", "context", "CUTEDSL", 7, {}),
    "cep16-context-k7": ("cep16", "context", "CUTEDSL", 7, {}),
    # The cep context server's other MoE backend: MegaMoE.
    "cep8-context-megamoe-k5": ("cep8", "context", "MEGAMOE_CUTEDSL", 5, {}),
    "cep16-context-megamoe-k7": ("cep16", "context", "MEGAMOE_CUTEDSL", 7, {}),
    "dep16-context-k7": ("dep16", "context", "CUTEDSL", 7, {}),
    "dep16-context-k5": ("dep16", "context", "CUTEDSL", 5, {}),
    "dep12-context-k7": ("dep12", "context", "CUTEDSL", 7, {}),
    "dep12-context-k5": ("dep12", "context", "CUTEDSL", 5, {}),
    "dep8-context-k7": ("dep8", "context", "CUTEDSL", 7, {}),
    "dep8-context-k5": ("dep8", "context", "CUTEDSL", 5, {}),
    "dep16-generation-b12k7": (
        "dep16",
        "generation",
        "MEGAMOE_CUTEDSL",
        7,
        dict(max_batch_size=12),
    ),
    "dep16-generation-b16k5": (
        "dep16",
        "generation",
        "MEGAMOE_CUTEDSL",
        5,
        dict(max_batch_size=16),
    ),
    "helix8-generation-b16k5": (
        "helix8",
        "generation",
        "CUTEDSL_FC12",
        5,
        dict(max_batch_size=16, **_HELIX8_GEN),
    ),
    "helix8-generation-b8k7": (
        "helix8",
        "generation",
        "CUTEDSL_FC12",
        7,
        dict(max_batch_size=8, **_HELIX8_GEN),
    ),
    "tep8-context-k7": ("tep8", "context", "CUTEDSL", 7, {}),
    "tep8-generation-b1k7": ("tep8", "generation", "CUTEDSL_FC12", 7, dict(max_batch_size=1)),
}


@pytest.mark.parametrize("case", list(_DEPLOYMENTS.values()), ids=list(_DEPLOYMENTS))
@pytest.mark.parametrize("ssm_dtype", [None, "float32"], ids=["state-auto", "state-float32"])
def test_the_recipe_lands_on_its_specialized_target(monkeypatch, tmp_path, case, ssm_dtype):
    """The export with an FP8 KV cache, FP8 index keys, an FP32 KDA state ('auto' resolves to it
    on a Kimi model) and DFlash2, at each tested layout, role and draft length."""
    layout, role, backend, draft_len, args = case
    _as_role(monkeypatch, role)
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, draft_len))
    assert engine_kv_cache_quant_algo(config, _llm_args(backend=backend, **args)) is QuantAlgo.FP8
    name = _resolves(config, backend=backend, ssm_dtype=ssm_dtype, **args)
    assert name == _SPECIALIZED[layout][1]
    assert config.extra_attrs[routing.DISAGG_ROLE_ATTR] == role


@pytest.mark.parametrize(
    "layout, backend",
    [("dep16", "MEGAMOE_CUTEDSL"), ("tep8", "CUTEDSL_FC12")],
    ids=["dep16", "tep8"],
)
def test_an_aggregated_server_takes_the_generation_role(monkeypatch, tmp_path, layout, backend):
    """Without a cache transceiver a server is aggregated, and runs the generation constants."""
    monkeypatch.setattr(routing, "disagg_role", _DISAGG_ROLE)
    monkeypatch.delenv(routing.DISAGG_ROLE_ENV, raising=False)
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, 7))
    assert _resolves(config, backend=backend, disaggregated=False) == _SPECIALIZED[layout][1]
    assert config.extra_attrs[routing.DISAGG_ROLE_ATTR] == "generation"


@pytest.mark.parametrize(
    "mapping, layout",
    [(_TEP8, "tep8"), (_CEP8, "cep8"), (_CEP16, "cep16")],
    ids=["tp8", "ulysses-cp8", "ulysses-cp16"],
)
@pytest.mark.parametrize("variable", [None, "1"], ids=["unset", "old-variable-set"])
def test_the_mapping_alone_tells_cep_from_tep8(monkeypatch, tmp_path, mapping, layout, variable):
    """A TP 8 context server is tep8 and a Ulysses CP N one is cepN, whatever the environment
    says: the retired ``TLLM_K31_CEP`` switch is not read."""
    _as_role(monkeypatch, "context")
    if variable is None:
        monkeypatch.delenv("TLLM_K31_CEP", raising=False)
    else:
        monkeypatch.setenv("TLLM_K31_CEP", variable)
    args = dict(backend="CUTEDSL", env_overrides={"TLLM_K31_CEP": variable or "0"})
    config = _modelopt_config(tmp_path, mapping, _dflash2(tmp_path, 7))
    assert _resolves(config, **args) == _SPECIALIZED[layout][1]
    assert config.extra_attrs[routing.DISAGG_ROLE_ATTR] == "context"
    assert not hasattr(routing, "CEP_ENV") and not hasattr(routing, "cep_requested")


def test_an_aggregated_dep12_server_is_refused(monkeypatch, tmp_path):
    """Without a cache transceiver a dep12 server is aggregated and takes the generation role,
    which dep12 does not serve; no default target takes it either."""
    monkeypatch.setattr(routing, "disagg_role", _DISAGG_ROLE)
    monkeypatch.delenv(routing.DISAGG_ROLE_ENV, raising=False)
    config = _modelopt_config(tmp_path, _DEP12, _dflash2(tmp_path, 7))
    _refused(
        config,
        "dep12 serves context servers only, not a generation server",
        backend="CUTEDSL",
        disaggregated=False,
    )
    assert routing.DISAGG_ROLE_ATTR not in config.extra_attrs


# Each specialized server (layout and role), as one of the tested deployments.
_SERVERS = {
    "cep8-context": "cep8-context-k7",
    "cep16-context": "cep16-context-k7",
    "dep16-context": "dep16-context-k7",
    "dep16-generation": "dep16-generation-b12k7",
    "dep12-context": "dep12-context-k7",
    "dep8-context": "dep8-context-k7",
    "helix8-generation": "helix8-generation-b16k5",
    "tep8-context": "tep8-context-k7",
    "tep8-generation": "tep8-generation-b1k7",
}


def _drafting(monkeypatch, tmp_path, server, draft_len):
    """(model config, arguments) of ``server`` in its role, drafting ``draft_len`` tokens."""
    layout, role, backend, _, args = _DEPLOYMENTS[_SERVERS[server]]
    _as_role(monkeypatch, role)
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, draft_len))
    return config, dict(backend=backend, **args)


@pytest.mark.parametrize("draft_len", [3, 4, 5, 6, 7], ids=lambda k: f"k{k}")
@pytest.mark.parametrize("server", list(_SERVERS))
def test_every_server_takes_draft_lengths_3_to_7(monkeypatch, tmp_path, server, draft_len):
    """Every layout and role takes each draft length from 3 to 7, one per server; a context server
    runs the one of the generation server it feeds."""
    config, args = _drafting(monkeypatch, tmp_path, server, draft_len)
    layout, role = server.split("-")
    assert _resolves(config, **args) == _SPECIALIZED[layout][1]
    assert config.extra_attrs[routing.DISAGG_ROLE_ATTR] == role


@pytest.mark.parametrize("draft_len", [1, 2, 8, 9], ids=lambda k: f"k{k}")
@pytest.mark.parametrize("server", list(_SERVERS))
def test_every_server_refuses_other_draft_lengths(monkeypatch, tmp_path, server, draft_len):
    """The reason names the draft lengths the server takes and, above 7, why a longer draft does
    not fit: the KDA verify kernels and one indexer-logits atom hold at most 7 drafts."""
    config, args = _drafting(monkeypatch, tmp_path, server, draft_len)
    layout, role = server.split("-")
    named = f"max_draft_len {draft_len}: a {layout} {role} server runs one of (3, 4, 5, 6, 7)"
    if (
        draft_len <= 2
        and args["backend"] in ("CUTEDSL", "MEGAMOE_CUTEDSL")
        and layout in _DEFAULT_LAYOUTS
    ):
        assert modeling_v2_resolve(config, _llm_args(**args)).__name__ == _DEFAULT[1]
    else:
        assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
        with pytest.raises(ValueError, match=re.escape(named)):
            modeling_v2_resolve(config, _llm_args("require", **args))


def _deployment(tmp_path, name):
    """(model config, arguments) of a tested deployment."""
    layout, _, backend, draft_len, args = _DEPLOYMENTS[name]
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, draft_len))
    return config, dict(backend=backend, **args)


# Refusals: (deployment, model config changes, argument changes, role) -> what the reason names.
_REFUSED = {
    "bf16-kv": (
        "tep8-generation-b1k7",
        {},
        dict(kv_dtype="auto"),
        None,
        "kv_cache_config.dtype 'auto'",
    ),
    "nvfp4-kv": (
        "tep8-generation-b1k7",
        {},
        dict(kv_dtype="nvfp4"),
        None,
        "kv_cache_config.dtype 'nvfp4'",
    ),
    "fp4-index-keys": (
        "tep8-generation-b1k7",
        {},
        dict(indexer_k_dtype="fp4"),
        None,
        "indexer_k_dtype 'fp4'",
    ),
    "bf16-state": (
        "tep8-generation-b1k7",
        {},
        dict(ssm_dtype="bfloat16"),
        None,
        "mamba_ssm_cache_dtype 'bfloat16'",
    ),
    "no-speculation": (
        "tep8-generation-b1k7",
        dict(spec=None),
        {},
        None,
        "no speculative decoding",
    ),
    "dspark": (
        "tep8-generation-b1k7",
        dict(spec="dspark"),
        {},
        None,
        "DSpark speculative decoding",
    ),
    "flashinfer-attention": (
        "tep8-generation-b1k7",
        dict(attn_backend="FLASHINFER"),
        {},
        None,
        "attention backend FLASHINFER",
    ),
    "bf16-shared-experts": (
        "tep8-generation-b1k7",
        dict(shared=False),
        {},
        None,
        "module(s) the NVFP4 export quantizes are BF16 here",
    ),
    "helix8-context": (
        "helix8-generation-b16k5",
        {},
        {},
        "context",
        "helix8 serves generation servers only, not a context server",
    ),
    "cep16-generation": (
        "cep16-context-k7",
        {},
        {},
        "generation",
        "cep16 serves context servers only, not a generation server",
    ),
    "cep16-context-fc12": (
        "cep16-context-k7",
        {},
        dict(backend="CUTEDSL_FC12"),
        "context",
        "a cep16 context server runs CUTEDSL or MEGAMOE_CUTEDSL",
    ),
    "cep16-moe-tp16": (
        "cep16-context-k7",
        dict(mapping=_ulysses(16, moe_ep_size=1, moe_tp_size=16)),
        {},
        None,
        "the specialized targets run cep8 and cep16",
    ),
    # The cep layouts before Ulysses CP: TP N without attention DP, MoE EP N.
    "ring-cp8": (
        "cep8-context-k7",
        dict(mapping=dict(_CEP8, cp_config={"cp_type": "RING"}, moe_ep_size=1, moe_tp_size=8)),
        {},
        None,
        "helix=False ulysses=False",
    ),
    "cep8-generation": (
        "cep8-context-k7",
        {},
        {},
        "generation",
        "cep8 serves context servers only, not a generation server",
    ),
    "cep4": (
        "cep16-context-k7",
        dict(mapping=_ulysses(4)),
        {},
        None,
        "cep4: a rank would hold 224 routed experts",
    ),
    "cep12": (
        "cep16-context-k7",
        dict(mapping=_ulysses(12)),
        {},
        None,
        "cep12: 12 ranks do not split the 96 KDA heads and the 896 experts",
    ),
    "cep32": (
        "cep16-context-k7",
        dict(mapping=_ulysses(32)),
        {},
        None,
        "cep32: no sm_107 run of a context-parallel server over 32 ranks yet",
    ),
    "dep12-generation": (
        "dep12-context-k7",
        {},
        {},
        "generation",
        "dep12 serves context servers only, not a generation server",
    ),
    # Unlike dep16's, no fallback: the configurable default does not take attention DP 12.
    "dep12-context-auto-moe": (
        "dep12-context-k7",
        {},
        dict(backend="AUTO"),
        None,
        "a dep12 context server runs CUTEDSL",
    ),
    "dep12-context-megamoe": (
        "dep12-context-k7",
        {},
        dict(backend="MEGAMOE_CUTEDSL"),
        None,
        "a dep12 context server runs CUTEDSL",
    ),
    "dep12-lm-head-tp-in-adp": (
        "dep12-context-k7",
        {},
        dict(enable_lm_head_tp_in_adp=True),
        None,
        "no enable_lm_head_tp_in_adp",
    ),
    "tp12-without-attention-dp": (
        "dep12-context-k7",
        dict(mapping=dict(world_size=12, tp_size=12, moe_ep_size=12)),
        {},
        None,
        "world_size=12 tp=12 pp=1 cp=1 helix=False ulysses=False attention_dp=False",
    ),
    "dep12-moe-tp2": (
        "dep12-context-k7",
        dict(mapping=dict(_DEP12, moe_tp_size=2, moe_ep_size=6)),
        {},
        None,
        "moe_tp=2 moe_ep=6",
    ),
    "dep24": (
        "dep12-context-k7",
        dict(mapping=dict(world_size=24, tp_size=24, enable_attention_dp=True, moe_ep_size=24)),
        {},
        None,
        "world_size=24",
    ),
    # A dep12 deployment the specialized targets refuse for any other reason has no target either:
    # no default target takes attention DP over 12 ranks.
    "dep12-nvfp4-kv": (
        "dep12-context-k7",
        {},
        dict(kv_dtype="nvfp4"),
        None,
        "kv_cache_config.dtype 'nvfp4'",
    ),
    "dep12-bf16-state": (
        "dep12-context-k7",
        {},
        dict(ssm_dtype="bfloat16"),
        None,
        "mamba_ssm_cache_dtype 'bfloat16'",
    ),
    "dep12-fp4-index-keys": (
        "dep12-context-k7",
        {},
        dict(indexer_k_dtype="fp4"),
        None,
        "indexer_k_dtype 'fp4'",
    ),
    "dep12-no-speculation": (
        "dep12-context-k7",
        dict(spec=None),
        {},
        None,
        "no speculative decoding",
    ),
    "dep12-moe-load-balancer": (
        "dep12-context-k7",
        {},
        dict(moe=dict(load_balancer=MoeLoadBalancerConfig(num_slots=896))),
        None,
        "no MoE load balancer",
    ),
    "helix8-moe-ep8": (
        "helix8-generation-b16k5",
        dict(mapping=dict(_HELIX8, moe_tp_size=1, moe_ep_size=8)),
        {},
        None,
        "moe_tp=1 moe_ep=8",
    ),
    "tp4": (
        "tep8-generation-b1k7",
        dict(mapping=dict(world_size=4, tp_size=4, moe_ep_size=4)),
        {},
        None,
        "world_size=4",
    ),
    "tep8-moe-tp8": (
        "tep8-generation-b1k7",
        dict(mapping=dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)),
        {},
        None,
        "moe_tp=8 moe_ep=1",
    ),
    "tep8-generation-cutedsl": (
        "tep8-generation-b1k7",
        {},
        dict(backend="CUTEDSL"),
        None,
        "a tep8 generation server runs CUTEDSL_FC12",
    ),
    "dep16-context-auto-moe": (
        "dep16-context-k7",
        {},
        dict(backend="AUTO"),
        "context",
        "a dep16 context server runs CUTEDSL",
    ),
    "draft-len-schedule": (
        "tep8-generation-b1k7",
        dict(spec_changes=dict(draft_len_schedule={4: 7, 8: 3})),
        {},
        None,
        "no draft_len_schedule",
    ),
    "max-concurrency": (
        "tep8-generation-b1k7",
        dict(spec_fields=dict(max_concurrency=8)),
        {},
        None,
        "no draft_len_schedule or max_concurrency",
    ),
    # Not a value the config validates; set past it, routing still names it.
    "bf16-history": (
        "tep8-generation-b1k7",
        dict(spec_unchecked=dict(context_kv_dtype="bf16")),
        {},
        None,
        "speculative_config.context_kv_dtype 'bf16'",
    ),
    "bf16-history-context": (
        "dep16-context-k5",
        dict(spec_fields=dict(context_kv_only=True), spec_unchecked=dict(context_kv_dtype="bf16")),
        {},
        "context",
        "speculative_config.context_kv_dtype 'bf16'",
    ),
    "history-only-tep8-generation": (
        "tep8-generation-b1k7",
        dict(spec_fields=dict(context_kv_only=True)),
        {},
        None,
        "context_kv_only on a tep8 generation server",
    ),
    "history-only-helix8": (
        "helix8-generation-b16k5",
        dict(spec_fields=dict(context_kv_only=True)),
        {},
        None,
        "context_kv_only on a helix8 generation server",
    ),
    "acceptance-rate-threshold": (
        "tep8-generation-b1k7",
        dict(spec_changes=dict(acceptance_rate_threshold=0.3, acceptance_rate_window_size=8)),
        {},
        None,
        "no acceptance_rate_threshold",
    ),
    "torch-compile": (
        "tep8-generation-b1k7",
        {},
        dict(torch_compile_config=TorchCompileConfig()),
        None,
        "no torch_compile_config",
    ),
    "piecewise-prefill": (
        "tep8-context-k7",
        {},
        dict(prefill_cuda_graph_backend="piecewise"),
        "context",
        "prefill_cuda_graph_backend 'piecewise'",
    ),
    "breakable-prefill": (
        "tep8-context-k7",
        {},
        dict(prefill_cuda_graph_backend="breakable"),
        "context",
        "prefill_cuda_graph_backend 'breakable'",
    ),
    "moe-load-balancer": (
        "tep8-generation-b1k7",
        {},
        dict(moe=dict(load_balancer=MoeLoadBalancerConfig(num_slots=896))),
        None,
        "no MoE load balancer",
    ),
    "kv-cache-manager-v1": (
        "tep8-generation-b1k7",
        {},
        dict(kv_cache=dict(use_kv_cache_manager_v2=False)),
        None,
        "keep the index keys and the drafter history in KV-V2",
    ),
    "beam-width-2": (
        "tep8-generation-b1k7",
        {},
        dict(max_beam_width=2),
        None,
        "beam width 1, got 2",
    ),
    "kv-pool-rebalance": (
        "tep8-generation-b1k7",
        {},
        dict(kv_cache=dict(enable_kv_pool_rebalance=True)),
        None,
        "without the KV pool rebalance",
    ),
    "tokens-per-block-32": (
        "tep8-generation-b1k7",
        {},
        dict(kv_cache=dict(tokens_per_block=32)),
        None,
        "tokens_per_block 32",
    ),
    "ucx-transceiver": (
        "tep8-generation-b1k7",
        {},
        dict(cache_transceiver_config=CacheTransceiverConfig(backend="UCX")),
        None,
        "Python NIXL transceiver only",
    ),
    "lm-head-tp-in-adp": (
        "dep16-generation-b12k7",
        {},
        dict(enable_lm_head_tp_in_adp=True),
        None,
        "no enable_lm_head_tp_in_adp",
    ),
    "helix8-block-reuse": (
        "helix8-generation-b16k5",
        {},
        dict(kv_cache=dict(enable_block_reuse=True)),
        None,
        "a helix8 generation server runs without block reuse",
    ),
    "helix8-aggregated": (
        "helix8-generation-b16k5",
        {},
        dict(disaggregated=False),
        None,
        "helix8 serves a disaggregated generation server",
    ),
}


@pytest.mark.parametrize("case", list(_REFUSED.values()), ids=list(_REFUSED))
def test_specialized_refusals(monkeypatch, tmp_path, case):
    name, config_changes, args_changes, role, named = case
    layout, deployment_role, backend, draft_len, args = _DEPLOYMENTS[name]
    _as_role(monkeypatch, role or deployment_role)
    config_changes = dict(config_changes)
    spec = config_changes.pop("spec", "dflash2")
    spec_config = None
    if spec == "dflash2":
        spec_config = _dflash2(
            tmp_path,
            config_changes.pop("draft_len", draft_len),
            **config_changes.pop("spec_fields", {}),
        )
        for field, value in config_changes.pop("spec_changes", {}).items():
            setattr(spec_config, field, value)
        for field, value in config_changes.pop("spec_unchecked", {}).items():
            object.__setattr__(spec_config, field, value)
    elif spec == "dspark":
        spec_config = DSparkDecodingConfig(
            max_draft_len=draft_len, speculative_model=str(tmp_path / "no_drafter")
        )
    mapping = config_changes.pop("mapping", _LAYOUTS[layout])
    config = _modelopt_config(tmp_path, mapping, spec_config, **config_changes)
    arguments = {**dict(backend=backend, **args), **args_changes}
    if arguments["backend"] in ("CUTEDSL", "AUTO") and named.startswith(
        ("a tep8 generation", "a dep16 context")
    ):
        assert modeling_v2_resolve(config, _llm_args(**arguments)).__name__ == _DEFAULT[1]
    else:
        _refused(config, named, **arguments)
    assert routing.DISAGG_ROLE_ATTR not in config.extra_attrs


@pytest.mark.parametrize(
    "name", ["dep16-generation-b12k7", "tep8-generation-b1k7", "tep8-context-k7"]
)
@pytest.mark.parametrize("disable_overlap_scheduler", [False, True], ids=["overlap", "no-overlap"])
@pytest.mark.parametrize("graphs", [None, [1, 2, 4, 8]], ids=["eager", "graphs"])
def test_dep16_and_tep8_take_overlap_and_cuda_graphs(
    monkeypatch, tmp_path, name, disable_overlap_scheduler, graphs
):
    """DEP16 and TEP8 servers run with or without the overlap scheduler and CUDA graphs."""
    _as_role(monkeypatch, _DEPLOYMENTS[name][1])
    config, args = _deployment(tmp_path, name)
    args.update(
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=None if graphs is None else dict(batch_sizes=graphs),
    )
    assert _resolves(config, **args) == _SPECIALIZED[_DEPLOYMENTS[name][0]][1]


@pytest.mark.parametrize(
    "name",
    [
        "cep8-context-k7",
        "cep16-context-k7",
        "dep16-context-k5",
        "dep16-context-k7",
        "dep12-context-k7",
        "dep8-context-k7",
        "tep8-context-k7",
    ],
)
def test_a_context_server_may_keep_just_the_drafter_history(monkeypatch, tmp_path, name):
    """``context_kv_only`` on a context server: its target builds only the drafter history."""
    layout, role, backend, draft_len, args = _DEPLOYMENTS[name]
    _as_role(monkeypatch, role)
    spec_config = _dflash2(tmp_path, draft_len, context_kv_only=True)
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], spec_config)
    assert _resolves(config, backend=backend, **args) == _SPECIALIZED[layout][1]
    assert config.extra_attrs[routing.DISAGG_ROLE_ATTR] == "context"


# --- image input: only the dep16 target's context servers encode (routing._VISION_CELLS) ---

#: The context servers that take images: the dep16 target on dep16, dep12 and dep8.
_VISION_SERVERS = ("dep16-context-k7", "dep16-context-k5", "dep12-context-k7", "dep8-context-k7")
_TEXT_ONLY_REFUSAL = (
    "the specialized targets serve text-only deployments outside the dep16, dep12 and dep8 "
    "context servers"
)


@pytest.mark.parametrize("history_only", [False, True], ids=["whole-drafter", "history-only"])
@pytest.mark.parametrize("name", _VISION_SERVERS)
def test_a_dep16_context_server_takes_images(monkeypatch, tmp_path, name, history_only):
    """Without ``disable_mm_encoder`` a dep16, dep12 or dep8 context server keeps the dep16
    target, which builds the vision tower, with the records of its text-only twin; also with
    ``context_kv_only`` (a context server that keeps just the drafter history)."""
    layout, role, backend, draft_len, args = _DEPLOYMENTS[name]
    _as_role(monkeypatch, role)
    records = []
    for disable_mm_encoder in (False, True):
        spec_config = _dflash2(tmp_path, draft_len, context_kv_only=history_only)
        config = _modelopt_config(tmp_path, _LAYOUTS[layout], spec_config)
        arguments = dict(backend=backend, disable_mm_encoder=disable_mm_encoder, **args)
        assert _resolves(config, **arguments) == "ModelingV2KimiK31ProNvfp4Sm107Dep16"
        records.append(dict(config.extra_attrs))
    assert records[0] == records[1]
    assert records[0][routing.DISAGG_ROLE_ATTR] == "context"
    assert routing.LEVER_ENV_ATTR in records[0]


def test_a_dep16_context_server_with_images_still_needs_fp8_history(monkeypatch, tmp_path):
    """The FP8 drafter history rule holds on the servers that take images: with ``auto`` the
    deployment goes to the configurable default target."""
    layout, role, backend, draft_len, args = _DEPLOYMENTS["dep16-context-k7"]
    _as_role(monkeypatch, role)
    spec_config = _dflash2(tmp_path, draft_len, context_kv_dtype="auto")
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], spec_config)
    arguments = dict(backend=backend, disable_mm_encoder=False, **args)
    assert _resolves(config, **arguments) == _DEFAULT[1]
    assert routing.DISAGG_ROLE_ATTR not in config.extra_attrs


# Every other specialized server with images: deployment -> why the configurable default target
# does not take it either, or None where it does.
_TEXT_ONLY_SERVERS = {
    "dep16-generation": ("dep16-generation-b12k7", None),
    "tep8-context": ("tep8-context-k7", None),
    "tep8-generation": ("tep8-generation-b1k7", "the target's routed experts require MoE backend"),
    "helix8-generation": (
        "helix8-generation-b16k5",
        "Kimi K3.1 Helix requires block reuse disabled",
    ),
    "cep8-context": ("cep8-context-k7", "Kimi K3.1 Helix requires TP1 without attention DP"),
    "cep16-context": ("cep16-context-k7", "Kimi K3.1 Helix requires TP1 without attention DP"),
    "cep8-context-megamoe": (
        "cep8-context-megamoe-k5",
        "Kimi K3.1 Helix requires TP1 without attention DP",
    ),
}


@pytest.mark.parametrize("case", list(_TEXT_ONLY_SERVERS.values()), ids=list(_TEXT_ONLY_SERVERS))
def test_every_other_server_with_images_leaves_the_specialized_targets(monkeypatch, tmp_path, case):
    """Without ``disable_mm_encoder`` every other specialized server goes to the configurable
    default target, or has no target where the default refuses it too (a cep server: the default
    takes no Ulysses CP)."""
    name, default_refusal = case
    layout, role, backend, draft_len, args = _DEPLOYMENTS[name]
    _as_role(monkeypatch, role)
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, draft_len))
    arguments = dict(backend=backend, disable_mm_encoder=False, **args)
    if default_refusal is None:
        assert _resolves(config, **arguments) == _DEFAULT[1]
    else:
        named = f"{_TEXT_ONLY_REFUSAL}; configurable default: {default_refusal}"
        _refused(config, named, **arguments)
    assert routing.DISAGG_ROLE_ATTR not in config.extra_attrs


@pytest.mark.parametrize("server", list(_SERVERS))
def test_a_text_only_server_never_reads_the_vision_cells(monkeypatch, tmp_path, server):
    """With ``disable_mm_encoder`` (every text-only deployment) routing does not read
    ``_VISION_CELLS``: the target and its records (the server role and the lever variables) are
    the same with the set empty (the rule before the dep16 context servers took images), as
    shipped, and holding every specialized cell."""
    layout, role, backend, draft_len, args = _DEPLOYMENTS[_SERVERS[server]]
    _as_role(monkeypatch, role)
    outcomes = []
    for cells in (frozenset(), routing._VISION_CELLS, frozenset(routing._SPECIALIZED)):
        monkeypatch.setattr(routing, "_VISION_CELLS", cells)
        config = _modelopt_config(tmp_path, _LAYOUTS[layout], _dflash2(tmp_path, draft_len))
        name = _resolves(config, backend=backend, disable_mm_encoder=True, **args)
        outcomes.append((name, dict(config.extra_attrs)))
    assert outcomes[0] == outcomes[1] == outcomes[2]
    name, records = outcomes[0]
    assert name == _SPECIALIZED[layout][1]
    assert records[routing.DISAGG_ROLE_ATTR] == role
    assert routing.LEVER_ENV_ATTR in records


# Per-layer dynamic EPLB with 8 helper slots per rank at cep8 and cep16.
_PER_LAYER_DYNAMIC = {
    "cep8-context-k7": dict(
        mode="per_layer_dynamic", num_slots=960, helper_slot_mode="shared_banks"
    ),
    "cep16-context-k7": dict(mode="per_layer_dynamic", num_slots=1024, auxiliary_sms=8),
    "dep16-context-k7": dict(
        mode="per_layer_dynamic", num_slots=1024, helper_slot_mode="shared_banks"
    ),
}


@pytest.mark.parametrize("name", list(_PER_LAYER_DYNAMIC))
def test_a_context_server_takes_per_layer_dynamic_eplb(monkeypatch, tmp_path, name):
    """The cep and dep16 context targets take per-layer dynamic EPLB (helper slots on every EP
    rank); the arguments keep their CuteDSL backend and drop the decode CUDA graphs."""
    config, args = _deployment(tmp_path, name)
    _as_role(monkeypatch, "context")
    balancer = _PER_LAYER_DYNAMIC[name]
    llm_args = _llm_args(moe=dict(load_balancer=balancer), **args)
    assert isinstance(llm_args.moe_config.load_balancer, MoePerLayerDynamicLoadBalancerConfig)
    assert llm_args.moe_config.backend == "CUTEDSL"
    assert llm_args.cuda_graph_config is None
    layout = _DEPLOYMENTS[name][0]
    assert _resolves(config, moe=dict(load_balancer=balancer), **args) == _SPECIALIZED[layout][1]


def test_a_cep_megamoe_server_takes_no_load_balancer(monkeypatch, tmp_path):
    """Per-layer dynamic EPLB serves the CuteDSL experts behind the all-to-all: the arguments refuse
    it beside MEGAMOE_CUTEDSL, and a static placement is the target's own option
    (``kimi_k31_expert_placement``), not a load balancer."""
    config, args = _deployment(tmp_path, "cep16-context-megamoe-k7")
    _as_role(monkeypatch, "context")
    balancer = _PER_LAYER_DYNAMIC["cep16-context-k7"]
    try:
        llm_args = _llm_args(moe=dict(load_balancer=balancer), **args)
    except ValueError:
        return
    text = getattr(config.pretrained_config, "text_config", None) or config.pretrained_config
    layout, why = routing._specialized_decision(config, llm_args, text)
    assert layout is None and "no MoE load balancer" in why, why


@pytest.mark.parametrize(
    "name, num_slots, named",
    [
        ("cep8-context-k7", 896, "num_slots=896"),
        ("cep8-context-k7", 900, "split evenly over the 8 EP ranks"),
        ("cep16-context-k7", 1000, "split evenly over the 16 EP ranks"),
    ],
    ids=["no-helpers", "uneven-cep8", "uneven-cep16"],
)
def test_a_cep_context_server_refuses_eplb_without_even_helpers(
    monkeypatch, tmp_path, name, num_slots, named
):
    config, args = _deployment(tmp_path, name)
    _as_role(monkeypatch, "context")
    balancer = dict(mode="per_layer_dynamic", num_slots=num_slots)
    _refused(config, named, moe=dict(load_balancer=balancer), **args)


@pytest.mark.parametrize(
    "name, balancer",
    [
        ("tep8-context-k7", dict(mode="per_layer_dynamic", num_slots=960)),
        ("cep8-context-k7", dict(num_slots=896)),
    ],
    ids=["tep8-context-dynamic", "cep8-static"],
)
def test_other_servers_and_static_placements_take_no_load_balancer(
    monkeypatch, tmp_path, name, balancer
):
    """tep8 takes no load balancer, and no specialized server a static placement. The generation
    servers' backends (MEGAMOE_CUTEDSL, CUTEDSL_FC12) are refused by the arguments already."""
    config, args = _deployment(tmp_path, name)
    _as_role(monkeypatch, _DEPLOYMENTS[name][1])
    _refused(config, "no MoE load balancer", moe=dict(load_balancer=balancer), **args)


@pytest.mark.parametrize("name", list(_DEPLOYMENTS))
def test_the_drafter_history_is_the_targets_fp8(monkeypatch, tmp_path, name):
    """The targets keep the drafter history in FP8 by their own constant. A config that names
    ``context_kv_dtype: fp8`` routes as one that leaves it ``auto`` (the recipe test's), so a
    context and a generation server agree."""
    layout, role, backend, draft_len, args = _DEPLOYMENTS[name]
    _as_role(monkeypatch, role)
    spec_config = _dflash2(tmp_path, draft_len, context_kv_dtype="fp8")
    config = _modelopt_config(tmp_path, _LAYOUTS[layout], spec_config)
    assert _resolves(config, backend=backend, **args) == _SPECIALIZED[layout][1]


@pytest.mark.parametrize("disable_overlap_scheduler", [False, True], ids=["overlap", "no-overlap"])
@pytest.mark.parametrize(
    "graphs",
    [None, _HELIX8_GRAPHS, dict(batch_sizes=list(range(1, 17)), enable_padding=True)],
    ids=["eager", "graphs-1-2-4-8-16", "graphs-1-to-16"],
)
def test_helix8_takes_cuda_graphs_and_the_overlap_scheduler(
    tmp_path, disable_overlap_scheduler, graphs
):
    """helix8 runs with or without CUDA graphs and the overlap scheduler."""
    config, args = _deployment(tmp_path, "helix8-generation-b16k5")
    args.update(disable_overlap_scheduler=disable_overlap_scheduler, cuda_graph_config=graphs)
    assert _resolves(config, **args) == _SPECIALIZED["helix8"][1]


@pytest.mark.parametrize(
    "mapping_kwargs", [_LAYOUTS[n] for n in _DEFAULT_LAYOUTS], ids=list(_DEFAULT_LAYOUTS)
)
def test_the_export_routes_to_the_blackwell_target_on_sm103(monkeypatch, tmp_path, mapping_kwargs):
    """The ModelOpt export selects the SM100/SM103 target on GB300."""
    _on_gpu(monkeypatch, (10, 3))
    config = _modelopt_config(tmp_path, mapping_kwargs, _dflash2(tmp_path, 7))
    target = modeling_v2_resolve(
        config, _llm_args("require", backend="AUTO", kv_cache={"enable_block_reuse": False})
    )
    assert target.__name__ == "ModelingV2KimiK31ProNvfp4Sm10xDefault"


@pytest.mark.parametrize("mapping_kwargs", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM"])
def test_the_general_target_never_takes_the_export_on_sm107(tmp_path, mapping_kwargs, backend):
    """With the general target's own arguments the export still gets a specialized target or a
    NoTarget, never the general target."""
    config = _modelopt_config(tmp_path, mapping_kwargs, _dflash2(tmp_path, 7))
    for mode in ("auto", "require"):
        try:
            target = modeling_v2_resolve(config, _llm_args(mode, backend=backend))
        except ValueError:  # NoTarget under "require"
            continue
        assert target is None or target.__name__ != _GENERAL


def test_the_same_shape_with_mxfp4_experts_routes_to_the_general_target_on_sm103(monkeypatch):
    """Control: with Moonshot's MXFP4 experts the same text config routes on sm_103, so the
    refusals above are about the checkpoint."""
    _on_gpu(monkeypatch, (10, 3))
    config = _mxfp4_config(**_TEP8)
    args = _llm_args(backend="AUTO", disaggregated=False)
    # The general target, or the sm_103 perf target of the layout.
    assert modeling_v2_resolve(config, args).__name__.startswith("ModelingV2KimiK31ProMxfp4")


def test_the_fixture_parses_as_the_export_does(tmp_path):
    config = _modelopt_config(tmp_path, _TEP8)
    assert config.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert config.quant_config.kv_cache_quant_algo is None
    experts = config.quant_config_dict["language_model.model.layers.3.block_sparse_moe.experts"]
    assert (experts.quant_algo, experts.group_size) == (QuantAlgo.NVFP4, 16)
    shared = config.quant_config_dict["model.layers.3.mlp.shared_experts.down_proj"]
    assert shared.quant_algo != QuantAlgo.NVFP4


# --- the server role ---


@pytest.mark.parametrize(
    "overrides, environ, role",
    [
        ({"TRTLLM_DISAGG_ROLE": "context"}, None, "context"),
        ({"TRTLLM_DISAGG_ROLE": "generation"}, "context", "generation"),
        ({"TRTLLM_DISAGG_ROLE": ""}, "context", "generation"),
        (None, "context", "context"),
        (None, None, "generation"),
    ],
    ids=["overrides", "overrides-first", "cleared", "environment", "aggregated"],
)
def test_disagg_role_reads_the_overrides_first(monkeypatch, overrides, environ, role):
    """``trtllm-serve --server_role`` writes the role to the environment and the overrides."""
    if environ is None:
        monkeypatch.delenv(routing.DISAGG_ROLE_ENV, raising=False)
    else:
        monkeypatch.setenv(routing.DISAGG_ROLE_ENV, environ)
    args = _llm_args(env_overrides=overrides) if overrides is not None else _llm_args()
    assert _DISAGG_ROLE(args) == role
    assert _DISAGG_ROLE({"env_overrides": overrides}) == role
    if overrides is None:
        assert _DISAGG_ROLE() == role


def test_the_role_reaches_routing_from_the_overrides(monkeypatch, tmp_path):
    """A server's overrides pick its row of the table: helix8 serves generation only."""
    monkeypatch.setattr(routing, "disagg_role", _DISAGG_ROLE)
    monkeypatch.delenv(routing.DISAGG_ROLE_ENV, raising=False)
    config = _modelopt_config(tmp_path, _HELIX8, _dflash2(tmp_path, 5))
    context = dict(env_overrides={routing.DISAGG_ROLE_ENV: "context"}, **_HELIX8_GEN)
    _refused(config, "not a context server", backend="CUTEDSL_FC12", **context)
    assert _resolves(config, backend="CUTEDSL_FC12", **_HELIX8_GEN) == _SPECIALIZED["helix8"][1]


# --- both loads of a server YAML ---
# The engine routes the config loaded before the target's model defaults apply (the arguments as
# written), then the config the model is built from.

#: The MLA layers of the checkpoint (1-indexed), each with a DSA indexer; KDA on the rest.
_CHECKPOINT_MLA_LAYERS = [*range(4, 93, 4), 93]


def _checkpoint_dir(tmp_path) -> str:
    """A directory with the export's config.json and hf_quant_config.json, 93 layers."""
    kda_layers = [i for i in range(1, 94) if i not in _CHECKPOINT_MLA_LAYERS]
    layers = _quantized_layers(93, _CHECKPOINT_MLA_LAYERS)
    text = dict(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        hidden_size=7168,
        num_hidden_layers=93,
        num_attention_heads=96,
        num_key_value_heads=96,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=2,
        first_k_dense_replace=1,
        moe_intermediate_size=3072,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vocab_size=163840,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            full_attn_layers=_CHECKPOINT_MLA_LAYERS,
            kda_layers=kda_layers,
            head_dim=128,
            num_heads=96,
        ),
        dsa_config=dict(
            dsa_layers=_CHECKPOINT_MLA_LAYERS,
            index_topk=2048,
            index_head_dim=128,
            index_num_heads=32,
            index_rope_dim=0,
            index_k_norm_type="layernorm",
            indexer_use_fp8=False,
            index_topk_freq=1,
            index_skip_topk_offset=1,
        ),
    )
    config = dict(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=text,
        vision_config=dict(vt_hidden_size=1024, vt_num_attention_heads=12, vt_num_hidden_layers=27),
        quantization_config=_inline_quantization_config(layers),
    )
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    (path / "hf_quant_config.json").write_text(json.dumps(_hf_quant_config(layers)))
    return str(path)


#: What decides the target in a server YAML of the recipe, as ``trtllm-serve --config`` reads it,
#: but its speculative_config (``_SERVER_DRAFT_LEN``): validating a DFlash speculative_config
#: against a local checkpoint imports the engine's drafter ops, which a CPU host lacks.
_SERVER_DRAFT_LEN = {
    "tep8-generation": 7,
    "helix8-generation": 5,
    "dep16-context": 5,
    "dep12-context": 5,
    "dep8-context": 5,
}
_SERVER_YAML = {
    "tep8-generation": """\
        tensor_parallel_size: 8
        moe_expert_parallel_size: 8
        enable_attention_dp: false
        max_batch_size: 1
        modeling_v2: require
        disable_mm_encoder: true
        kv_cache_config:
          dtype: fp8
          mamba_ssm_cache_dtype: float32
        moe_config:
          backend: CUTEDSL_FC12
        sparse_attention_config:
          algorithm: dsa
          indexer_k_dtype: fp8
        cache_transceiver_config:
          backend: NIXL
          transceiver_runtime: PYTHON
        """,
    "helix8-generation": """\
        tensor_parallel_size: 1
        context_parallel_size: 8
        moe_tensor_parallel_size: 2
        moe_expert_parallel_size: 4
        cp_config:
          cp_type: HELIX
          tokens_per_block: 64
        enable_attention_dp: false
        max_batch_size: 16
        modeling_v2: require
        disable_mm_encoder: true
        cuda_graph_config:
          batch_sizes: [1, 2, 4, 8, 16]
          enable_padding: true
        disable_overlap_scheduler: false
        kv_cache_config:
          dtype: fp8
          mamba_ssm_cache_dtype: float32
          enable_block_reuse: false
          tokens_per_block: 64
        moe_config:
          backend: CUTEDSL_FC12
        sparse_attention_config:
          algorithm: dsa
          indexer_k_dtype: fp8
        cache_transceiver_config:
          backend: NIXL
          transceiver_runtime: PYTHON
        """,
    "dep16-context": """\
        tensor_parallel_size: 16
        moe_expert_parallel_size: 16
        enable_attention_dp: true
        max_batch_size: 32
        modeling_v2: require
        disable_mm_encoder: true
        cuda_graph_config: null
        disable_overlap_scheduler: true
        kv_cache_config:
          dtype: fp8
          mamba_ssm_cache_dtype: float32
          enable_block_reuse: true
        moe_config:
          backend: CUTEDSL
        sparse_attention_config:
          algorithm: dsa
          indexer_k_dtype: fp8
        cache_transceiver_config:
          backend: NIXL
          transceiver_runtime: PYTHON
        """,
}


# The dep12 context server: dep16's over 12 ranks.
_SERVER_YAML["dep12-context"] = (
    _SERVER_YAML["dep16-context"]
    .replace("tensor_parallel_size: 16", "tensor_parallel_size: 12")
    .replace("moe_expert_parallel_size: 16", "moe_expert_parallel_size: 12")
)
# The dep8 context server: dep16's over 8 ranks.
_SERVER_YAML["dep8-context"] = (
    _SERVER_YAML["dep16-context"]
    .replace("tensor_parallel_size: 16", "tensor_parallel_size: 8")
    .replace("moe_expert_parallel_size: 16", "moe_expert_parallel_size: 8")
)
assert "tensor_parallel_size: 8" in _SERVER_YAML["dep8-context"]


def _loaded_configs(checkpoint, args, spec_config):
    """The two configs the engine routes on: the one ``load_config_and_apply_defaults`` loads,
    without a sparse attention config, and the one the model is built from, with the arguments'."""
    kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        mm_encoder_only=args.mm_encoder_only,
        mapping=args.parallel_config.to_mapping(),
        spec_config=spec_config,
    )
    first = ModelConfig.from_pretrained(checkpoint, **kwargs)
    built = ModelConfig.from_pretrained(
        checkpoint, sparse_attention_config=args.sparse_attention_config, **kwargs
    )
    return first, built


@pytest.mark.parametrize("server", list(_SERVER_YAML))
def test_a_server_yaml_routes_both_loads(monkeypatch, tmp_path, server):
    layout, role = server.split("-")
    _as_role(monkeypatch, role)
    checkpoint = _checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **yaml.safe_load(textwrap.dedent(_SERVER_YAML[server])))
    spec_config = _dflash2(tmp_path, _SERVER_DRAFT_LEN[server])
    for config in _loaded_configs(checkpoint, args, spec_config):
        assert config.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert engine_kv_cache_quant_algo(config, args) is QuantAlgo.FP8
        assert routing._specialized_layout(config.mapping) == layout
        assert modeling_v2_resolve(config, args).__name__ == _SPECIALIZED[layout][1]


#: The DFlash2 MLA drafter's geometry, as its config.json gives it.
_DRAFT_GEOMETRY = {
    "architectures": ["DFlash2DraftModel"],
    "block_size": 8,
    "dflash_config": {"attention_mode": "mla", "block_size": 8},
    "dtype": "bfloat16",
    "head_dim": 64,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "num_attention_heads": 96,
    "num_hidden_layers": 5,
    "num_key_value_heads": 96,
    "qk_rope_head_dim": 64,
    "sliding_window": 1024,
}


def test_the_export_declares_no_per_slot_drafter_context(tmp_path):
    """The specialized targets keep the drafter history in the unified draft cache (KV pool
    pages): the family declares 0 bytes per slot for the export, and the general target's ring
    for Moonshot's MXFP4 checkpoint."""
    from tensorrt_llm._torch._experimental.modeling_v2 import modeling_v2_dflash_ctx_bytes_per_slot

    def args(model_dir):
        return TorchLlmArgs.model_construct(
            model=model_dir,
            modeling_v2="require",
            speculative_config=_dflash2(tmp_path, 7),
            max_seq_len=1048576,
        )

    export = _checkpoint_dir(tmp_path)
    assert modeling_v2_dflash_ctx_bytes_per_slot(args(export), _DRAFT_GEOMETRY) == 0
    mxfp4 = tmp_path / "mxfp4"
    mxfp4.mkdir()
    (mxfp4 / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["KimiK31ForConditionalGeneration"],
                quantization_config={"format": "mxfp4-pack-quantized"},
            )
        )
    )
    ring = modeling_v2_dflash_ctx_bytes_per_slot(args(str(mxfp4)), _DRAFT_GEOMETRY)
    assert isinstance(ring, int) and ring > 0


def test_an_auto_kda_state_resolves_to_fp32_on_the_export(tmp_path):
    """``mamba_ssm_cache_dtype: auto``, which routing accepts: on the K3.1 config the engine
    resolves it to an FP32 KDA state whatever model dtype it would fall back on."""
    checkpoint = _checkpoint_dir(tmp_path)
    options = yaml.safe_load(textwrap.dedent(_SERVER_YAML["tep8-generation"]))
    options["kv_cache_config"].pop("mamba_ssm_cache_dtype")
    args = TorchLlmArgs(model=checkpoint, **options)
    assert args.kv_cache_config.mamba_ssm_cache_dtype == "auto"
    loaded = _loaded_configs(checkpoint, args, _dflash2(tmp_path, 7))
    configs = [config.pretrained_config for config in loaded]
    assert [is_kimi_linear(config) for config in configs] == [True] * 2
    pytest.importorskip("fla")  # the KDA cache manager's package imports fla.modules at load time
    for config in configs:
        assert resolve_auto_ssm_cache_dtype(config, torch.bfloat16) is torch.float32


# --- the role reaches the target ---
# Routing records it in the config's extra_attrs; the model loader carries them across the
# replace that sets the target, to the config the model is built from.

#: A TEP8 context server, in front of the TEP8 generation server above.
_TEP8_CONTEXT_YAML = """\
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    enable_attention_dp: false
    max_batch_size: 32
    max_num_tokens: 16384
    enable_chunked_prefill: true
    modeling_v2: require
    disable_mm_encoder: true
    cuda_graph_config: null
    disable_overlap_scheduler: true
    kv_cache_config:
      dtype: fp8
      mamba_ssm_cache_dtype: float32
      enable_block_reuse: true
      tokens_per_block: 64
    moe_config:
      backend: CUTEDSL
    sparse_attention_config:
      algorithm: dsa
      indexer_k_dtype: fp8
    cache_transceiver_config:
      backend: NIXL
      transceiver_runtime: PYTHON
    """

#: Servers as ``trtllm-serve`` starts them: (server YAML, changes to it (None removes a key), draft
#: length, the ``--server_role`` given (None: none), the role the target builds). The aggregated
#: TEP8 server is the generation server without a cache transceiver, with chunked prefill.
_ROLE_SERVERS = {
    "tep8-context": (_TEP8_CONTEXT_YAML, {}, 7, "context", "context"),
    "tep8-generation": (_SERVER_YAML["tep8-generation"], {}, 7, None, "generation"),
    "tep8-aggregated": (
        _SERVER_YAML["tep8-generation"],
        dict(cache_transceiver_config=None, max_num_tokens=2048, enable_chunked_prefill=True),
        7,
        None,
        "generation",
    ),
    "dep16-context": (_SERVER_YAML["dep16-context"], {}, 5, "context", "context"),
    "dep12-context": (_SERVER_YAML["dep12-context"], {}, 5, "context", "context"),
    "dep8-context": (_SERVER_YAML["dep8-context"], {}, 5, "context", "context"),
    "helix8-generation": (_SERVER_YAML["helix8-generation"], {}, 5, None, "generation"),
}


def _model_loader():
    """The engine's model loader module; skips on a CPU host whose stand-ins cannot import it."""
    try:
        from tensorrt_llm._torch.pyexecutor import model_loader
    except Exception as exc:  # noqa: BLE001 - the engine runtime is not importable on this host
        pytest.skip(f"the model loader does not import here: {type(exc).__name__}: {exc}")
    return model_loader


class _HfConfigOnly:
    """A checkpoint loader that reads the config as the HF loader does, and nothing else."""

    def load_config(self, checkpoint_dir, **kwargs):
        from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader

        return HfConfigLoader().load(checkpoint_dir, **kwargs)


def _target_role_constants(layout) -> dict:
    """``_ROLES``, ``_ROLE_ATTR``, ``_KEY_ATTR`` and ``_LEVER_ENV_ATTR`` of a specialized target, or
    of the default target (``layout`` "default"), read off its source: importing a target needs its
    GPU kernels."""
    directory = _DEFAULT[0] if layout == "default" else _SPECIALIZED[layout][0]
    path = Path(routing.__file__).parent / directory / "modeling.py"
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("_ROLES", "_ROLE_ATTR", "_KEY_ATTR", "_LEVER_ENV_ATTR"):
                found[name] = ast.literal_eval(node.value)
    return found


@pytest.mark.parametrize("server", list(_ROLE_SERVERS))
def test_the_role_reaches_the_target_through_the_loader(monkeypatch, tmp_path, server):
    """``ModelLoader._load_and_validate_config`` on the export directory, with the role passed as
    ``trtllm-serve --server_role`` passes it: the config it returns, and the copy ``load`` builds
    the model from, carry the target and, under the key the target reads, a role it serves.

    The KDA state check, which imports the KDA package (``fla``), is stood in; it gets the config
    the loader returns."""
    model_loader = _model_loader()
    state_checked = []
    monkeypatch.setattr(
        model_loader,
        "validate_and_set_mamba_ssm_cache_dtype",
        lambda config, *args: state_checked.append(config),
    )
    text, changes, draft_len, server_role, role = _ROLE_SERVERS[server]
    layout = server.split("-")[0]
    monkeypatch.setattr(routing, "disagg_role", _DISAGG_ROLE)
    monkeypatch.delenv(routing.DISAGG_ROLE_ENV, raising=False)
    options = yaml.safe_load(textwrap.dedent(text))
    for key, value in changes.items():
        if value is None:
            options.pop(key)
        else:
            options[key] = value
    if server_role is not None:
        # commands/serve.py: the environment and the overrides the workers apply.
        monkeypatch.setenv(routing.DISAGG_ROLE_ENV, server_role)
        options["env_overrides"] = {routing.DISAGG_ROLE_ENV: server_role}
    checkpoint = _checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **options)
    loader = object.__new__(model_loader.ModelLoader)
    loader.llm_args = args
    loader.mapping = args.parallel_config.to_mapping()
    loader.spec_config = _dflash2(tmp_path, draft_len)
    loader.sparse_attention_config = args.sparse_attention_config
    loader.max_num_tokens = args.max_num_tokens
    loader.max_seq_len = args.max_seq_len
    loader.lora_config = None
    config = loader._load_and_validate_config(checkpoint, _HfConfigOnly())
    assert len(state_checked) == 1 and state_checked[0] is config
    assert routing._specialized_layout(config.mapping) == layout
    assert config.modeling_v2_target.__name__ == _SPECIALIZED[layout][1]
    assert config.extra_attrs["kv_cache_dtype"] == "fp8"  # the loader's own keys stay
    built = copy.deepcopy(config)  # ModelLoader.load builds the model from a deep copy
    constants = _target_role_constants(layout)
    assert constants["_ROLE_ATTR"] == routing.DISAGG_ROLE_ATTR
    assert built.extra_attrs[constants["_ROLE_ATTR"]] == role
    assert role in constants["_ROLES"]


@pytest.mark.parametrize("layout", sorted(_SPECIALIZED))
def test_the_targets_read_the_keys_routing_writes(layout):
    """The ``extra_attrs`` keys an sm_107 target reads its role, its lever variables and (the
    default target) its settings key under are the keys routing records them under."""
    constants = _target_role_constants(layout)
    assert constants["_ROLE_ATTR"] == routing.DISAGG_ROLE_ATTR
    assert constants["_LEVER_ENV_ATTR"] == routing.LEVER_ENV_ATTR


# --- Moonshot's MXFP4 checkpoint on sm_107: the general target's recipe, and no other ---


def _mxfp4_config(spec_config=None, indexer_k_dtype="fp8", **mapping_kwargs) -> ModelConfig:
    text = dict(_TEXT, model_type="kimi_linear")
    # The layer layout the MXFP4 targets build: attention-residual blocks, the latent MoE's norm and SiTu
    # betas, and a position-free DSA indexer.
    text.update(attn_res_block_size=4, latent_moe_use_norm=True, activation_situ_linear_beta=25.0)
    text["dsa_config"] = dict(_TEXT["dsa_config"], index_rope_dim=0)
    return ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["KimiK31ForConditionalGeneration"],
            model_type="kimi_k31",
            text_config=PretrainedConfig(**text),
            quantization_config={"format": "mxfp4-pack-quantized"},
        ),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(),
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype=indexer_k_dtype),
    )


@pytest.mark.parametrize(
    "mapping", [_TEP8, dict(world_size=16, tp_size=16, enable_attention_dp=True)]
)
@pytest.mark.parametrize("spec", ["none", "dflash2"])
@pytest.mark.parametrize("kv_dtype", ["fp8", "auto"])
def test_the_mxfp4_checkpoint_keeps_the_general_target_on_sm107(tmp_path, mapping, spec, kv_dtype):
    """FP8 index keys, an FP8 or BF16 KV cache and DFlash2 or no speculative decoding."""
    spec_config = _dflash2(tmp_path, 4, context_kv_dtype="auto") if spec == "dflash2" else None
    config = _mxfp4_config(spec_config, **mapping)
    args = _llm_args(kv_dtype=kv_dtype, backend="AUTO", disaggregated=False)
    assert modeling_v2_resolve(config, args).__name__ == _GENERAL


# Settings of an MXFP4 deployment the general target refuses on sm_107 as on sm_103 -> (model
# config changes, LLM argument changes, the refusal).
_GENERAL_REFUSALS = {
    "fp4-index-keys": (
        dict(indexer_k_dtype="fp4"),
        dict(indexer_k_dtype="fp4"),
        "indexer_k_dtype 'fp4'",
    ),
    "fp4-index-keys-in-the-arguments": (
        {},
        dict(indexer_k_dtype="fp4"),
        "indexer_k_dtype 'fp4'",
    ),
    "nvfp4-kv-cache": ({}, dict(kv_dtype="nvfp4"), "kv_cache_config.dtype='nvfp4'"),
    "dspark": (
        dict(spec="dspark"),
        {},
        "the Kimi K3.1 target runs DFlash2 speculative decoding",
    ),
}


@pytest.mark.parametrize("capability", [(10, 7), (10, 3)], ids=["sm107", "sm103"])
@pytest.mark.parametrize("mapping", [_CEP8, _CEP16], ids=["ulysses-cp8", "ulysses-cp16"])
def test_the_general_target_refuses_ulysses_cp(monkeypatch, capability, mapping):
    """Only the sm_107 cep target runs Ulysses CP; the MXFP4 checkpoint's general target says so."""
    _on_gpu(monkeypatch, capability)
    config = _mxfp4_config(**mapping)
    _refused(
        config,
        "context parallelism ULYSSES: the Kimi K3.1 target runs Helix CP",
        backend="AUTO",
        disaggregated=False,
    )


def test_only_the_cep_target_declares_ulysses_cp():
    """The engine refuses a Ulysses CP mapping for a model class without ``supports_cp_ulysses``
    (``check_cp_ulysses_support``); of the family's targets only cep declares it."""
    from tensorrt_llm._torch.models.modeling_auto import check_cp_ulysses_support

    family = Path(routing.__file__).resolve().parent
    declaring = set()
    for path in family.glob("*/modeling.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    targets = (
                        [stmt.target]
                        if isinstance(stmt, ast.AnnAssign)
                        else getattr(stmt, "targets", [])
                    )
                    names = {t.id for t in targets if isinstance(t, ast.Name)}
                    if "supports_cp_ulysses" in names and ast.literal_eval(stmt.value) is True:
                        declaring.add((path.parent.name, node.name))
    assert declaring == {_CEP_TARGET}

    ulysses = Mapping(**_ulysses(8))
    check_cp_ulysses_support(type("Cep", (), {"supports_cp_ulysses": True}), ulysses)
    check_cp_ulysses_support(type("Tep8", (), {}), Mapping(**_TEP8))
    with pytest.raises(ValueError, match="does not support Ulysses context parallelism"):
        check_cp_ulysses_support(type("Tep8", (), {}), ulysses)
    with pytest.raises(ValueError, match="does not support attention DP"):
        check_cp_ulysses_support(
            type("Cep", (), {"supports_cp_ulysses": True}),
            Mapping(**_ulysses(8), enable_attention_dp=True),
        )


@pytest.mark.parametrize("capability", [(10, 7), (10, 3)], ids=["sm107", "sm103"])
@pytest.mark.parametrize("case", list(_GENERAL_REFUSALS))
def test_the_general_target_refuses_what_it_does_not_run(monkeypatch, tmp_path, capability, case):
    _on_gpu(monkeypatch, capability)
    config_changes, args_changes, refusal = _GENERAL_REFUSALS[case]
    config_changes = dict(config_changes)
    spec_config = None
    if config_changes.pop("spec", None) == "dspark":
        spec_config = DSparkDecodingConfig(
            max_draft_len=5, speculative_model=str(tmp_path / "no_drafter")
        )
    config = _mxfp4_config(spec_config, **config_changes, **_TEP8)
    _refused(config, refusal, backend="AUTO", disaggregated=False, **args_changes)


def test_claims_suite_reads_the_specialized_targets():
    """The claims suite finds a routed target by its one-line import in routing.py; each
    specialized target is one, and so is the default target."""
    source = Path(routing.__file__).read_text()
    routed = dict(re.findall(r"^\s*from \.(\w+)\.modeling import (\w+)$", source, re.MULTILINE))
    for directory, name in (*_SPECIALIZED.values(), _DEFAULT):
        assert routed.get(directory) == name


def test_routing_records_the_lever_variables_the_target_checks(monkeypatch, tmp_path):
    """Routing hands the target the lever variables (by prefix) and the variables read in
    C++ or by the KV cache estimate, the overrides over the environment; nothing else."""
    monkeypatch.setenv("TRTLLM_K3_FC12_SM_RESERVE", "16")
    monkeypatch.setenv("TLLM_ROUTING_DISABLE_COOP_LAUNCH", "1")
    monkeypatch.setenv("MEGAMOE_COMBINE_FORMAT", "bf16")
    monkeypatch.setenv("TLLM_SOMETHING_ELSE", "1")
    config, args = _deployment(tmp_path, "tep8-generation-b1k7")
    args["env_overrides"] = {"TRTLLM_K3_FC12_SM_RESERVE": "8"}
    assert _resolves(config, **args) == _SPECIALIZED["tep8"][1]
    recorded = config.extra_attrs[routing.LEVER_ENV_ATTR]
    assert recorded["TRTLLM_K3_FC12_SM_RESERVE"] == "8"
    assert recorded["TLLM_ROUTING_DISABLE_COOP_LAUNCH"] == "1"
    assert recorded["MEGAMOE_COMBINE_FORMAT"] == "bf16"
    assert "TLLM_SOMETHING_ELSE" not in recorded


# --- dep12: the dep16 target on a context server over 12 ranks ---


@pytest.mark.parametrize(
    "mapping_kwargs, layout",
    [
        (_DEP12, "dep12"),
        (_DEP16, "dep16"),
        (dict(world_size=12, tp_size=12, moe_ep_size=12), None),  # TP12, no attention DP
        (dict(_DEP12, moe_tp_size=2, moe_ep_size=6), None),
        (_DEP8, "dep8"),
        (dict(_DEP8, moe_tp_size=2, moe_ep_size=4), None),
        (dict(world_size=8, tp_size=8, moe_ep_size=8), "tep8"),  # TP8, no attention DP
        (dict(world_size=24, tp_size=24, enable_attention_dp=True, moe_ep_size=24), None),
    ],
    ids=["dep12", "dep16", "tp12", "dep12-moe-tp2", "dep8", "dep8-moe-tp2", "tep8", "dep24"],
)
def test_the_specialized_layouts_take_dep12_and_dep8_and_no_other_new_size(mapping_kwargs, layout):
    assert routing._specialized_layout(Mapping(**mapping_kwargs)) == layout


@pytest.mark.parametrize("layout", ["dep12", "dep8"])
def test_dep12_and_dep8_serve_the_context_role_only(layout):
    """dep12 and dep8 are one row each of the role table: a CuteDSL context server at draft
    lengths 3 to 7."""
    assert {role for name, role in routing._SPECIALIZED if name == layout} == {"context"}
    assert routing._SPECIALIZED[(layout, "context")] == routing._SPECIALIZED[("dep16", "context")]


def test_the_configurable_default_refuses_dep12(monkeypatch, tmp_path):
    """On sm_103 the export runs only on the configurable default, which runs attention DP over
    powers of two: dep12 has no target there (the arguments the default takes at dep16)."""
    _on_gpu(monkeypatch, (10, 3))
    config = _modelopt_config(tmp_path, _DEP12, _dflash2(tmp_path, 7))
    _refused(
        config,
        "attention DP over 12 ranks",
        backend="AUTO",
        kv_cache={"enable_block_reuse": False},
    )
