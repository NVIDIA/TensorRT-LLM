# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 target routing, driven by synthetic configs on CPU."""

import json
import os
import tempfile

import pytest
import torch
from transformers import PretrainedConfig

import tensorrt_llm._utils
from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_NVFP4_TARGET = (
    "models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling."
    "ModelingV2KimiK31ProNvfp4Sm10xDefault"
)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1)


def _qualified(target) -> str:
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


def _on_gpu(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    # Config validators read the SM through get_sm_version, which is fixed to -1 in a process
    # that started without a GPU (the CPU stage).
    monkeypatch.setattr(
        tensorrt_llm._utils, "get_sm_version", lambda: capability[0] * 10 + capability[1]
    )


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    _on_gpu(monkeypatch, (10, 3))


def _k31_config(text_overrides=None, quant_format="mxfp4-pack-quantized"):
    """The shape Kimi K3.1 Pro's own config.json declares (layer count of a truncated copy)."""
    text = dict(
        num_hidden_layers=8,
        first_k_dense_replace=0,
        linear_attn_config=dict(kda_layers=[1, 2, 3, 5, 6, 7], full_attn_layers=[4, 8]),
        hidden_size=7168,
        num_experts=896,
        num_experts_per_token=16,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        vocab_size=163840,
        dsa_config=dict(dsa_layers=[4, 8], index_topk=2048, index_num_heads=32, index_head_dim=128),
    )
    text.update(text_overrides or {})
    return PretrainedConfig(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=PretrainedConfig(**text),
        quantization_config={"format": quant_format},
    )


def _llm_args(mode="auto", kv_dtype="fp8", sparse=True, kv=None, **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=KvCacheConfig(dtype=kv_dtype, **(kv or {})),
        sparse_attention_config=DeepSeekSparseAttentionConfig() if sparse else None,
        **kwargs,
    )


def _model_config(
    pretrained_config,
    spec_config=None,
    sparse=True,
    kv_algo=QuantAlgo.FP8,
    indexer_k_dtype="fp8",
    quant_config=None,
    quant_config_dict=None,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=quant_config or QuantConfig(kv_cache_quant_algo=kv_algo),
        quant_config_dict=quant_config_dict,
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype=indexer_k_dtype)
        if sparse
        else None,
    )


def _dflash_drafter(attention_mode="mla", max_draft_len=4) -> DFlashDecodingConfig:
    """DFlash over a drafter directory holding a DFlash2 config.json of ``attention_mode``."""
    path = tempfile.mkdtemp(prefix="k31_dflash2_")
    dflash_config = dict(
        block_size=5,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=32,
        mask_token_id=163824,
        target_layer_ids=[7, 23, 51, 67, 83, 91],
    )
    if attention_mode is not None:
        dflash_config["attention_mode"] = attention_mode
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(
            dict(
                architectures=["DFlash2DraftModel"], model_type="qwen3", dflash_config=dflash_config
            ),
            f,
        )
    config = DFlashDecodingConfig(max_draft_len=max_draft_len, speculative_model=path)
    config.resolve_from_checkpoint()
    return config


_DFLASH2 = _dflash_drafter()
_DFLASH2_W8 = _dflash_drafter(max_draft_len=7)


_LAYER_KINDS = dict(linear_attn_config=dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7]))


def _nvfp4_model_config(text=None, **kwargs):
    return _model_config(
        _k31_config(text or _LAYER_KINDS, quant_format="modelopt"),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={
            "language_model.model.layers.1.block_sparse_moe.experts": QuantConfig(
                quant_algo=QuantAlgo.NVFP4, group_size=16
            )
        },
        **kwargs,
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_k31_nvfp4_kv_route(monkeypatch, capability):
    _on_gpu(monkeypatch, capability)
    config = _nvfp4_model_config(kv_algo=QuantAlgo.NVFP4, **_TEP8)
    args = _llm_args("require", kv_dtype="nvfp4")
    assert _qualified(modeling_v2_resolve(config, args)) == _NVFP4_TARGET


def test_k31_nvfp4_requires_sparse_mla():
    kinds = dict(linear_attn_config=dict(full_attn_layers=[2, 4, 8], kda_layers=[1, 3, 5, 6, 7]))
    config = _nvfp4_model_config(kinds, kv_algo=QuantAlgo.NVFP4, **_TEP8)
    with pytest.raises(ValueError, match="DSA on every MLA layer"):
        modeling_v2_resolve(config, _llm_args("require", kv_dtype="nvfp4"))


@pytest.mark.parametrize(
    "override, error",
    [
        ("3,7", None),
        ("", None),
        ("0", "only MLA layers"),
        ("8", "outside"),
        ("3,", "comma-separated"),
        ("last", "integer"),
    ],
)
def test_k31_fp8_override_validation(monkeypatch, override, error):
    monkeypatch.setenv("TLLM_KIMI_K3_FP8_KV_LAYERS", override)
    config = _nvfp4_model_config(kv_algo=QuantAlgo.NVFP4, **_TEP8)
    args = _llm_args("require", kv_dtype="nvfp4")
    if error is None:
        assert _qualified(modeling_v2_resolve(config, args)) == _NVFP4_TARGET
    else:
        with pytest.raises(ValueError, match=error):
            modeling_v2_resolve(config, args)


def test_k31_fp8_override_requires_nvfp4(monkeypatch):
    monkeypatch.setenv("TLLM_KIMI_K3_FP8_KV_LAYERS", "3,7")
    config = _nvfp4_model_config(**_TEP8)
    with pytest.raises(ValueError, match="requires NVFP4 cache storage"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("cp", [2, 4, 8])
@pytest.mark.parametrize("spec_config", [None, _DFLASH2, _DFLASH2_W8])
@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_k31_helix_routes_with_managed_cache(cp, spec_config, cuda_graph, overlap):
    config = _nvfp4_model_config(
        spec_config=spec_config,
        world_size=cp,
        tp_size=1,
        cp_size=cp,
        cp_config={"cp_type": "HELIX"},
        moe_ep_size=cp,
        moe_tp_size=1,
    )
    args = _llm_args(
        mode="require",
        cuda_graph_config=CudaGraphConfig(max_batch_size=8) if cuda_graph else None,
        disable_overlap_scheduler=not overlap,
        kv={"enable_block_reuse": False},
    )
    assert _qualified(modeling_v2_resolve(config, args)) == _NVFP4_TARGET


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("main", ["fp8", "nvfp4"])
@pytest.mark.parametrize("indexer", ["fp8", "fp4"])
@pytest.mark.parametrize("state", ["float32", "bfloat16"])
@pytest.mark.parametrize("fallback", ["", None])
def test_nvfp4_checkpoint_uses_one_sm10x_target(
    monkeypatch, capability, main, indexer, state, fallback
):
    _on_gpu(monkeypatch, capability)
    if fallback is None:
        monkeypatch.delenv("TLLM_KIMI_K3_FP8_KV_LAYERS", raising=False)
    else:
        monkeypatch.setenv("TLLM_KIMI_K3_FP8_KV_LAYERS", fallback)
    quant = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
    recipes = {
        "language_model.model.layers.1.block_sparse_moe.experts": QuantConfig(
            quant_algo=QuantAlgo.NVFP4, group_size=16
        ),
        "language_model.model.layers.0.self_attn.q_proj": QuantConfig(
            quant_algo=QuantAlgo.MXFP8, group_size=32
        ),
    }
    config = _model_config(
        _k31_config(_LAYER_KINDS, quant_format="modelopt"),
        quant_config=quant,
        quant_config_dict=recipes,
        indexer_k_dtype=indexer,
        **_TEP8,
    )
    args = _llm_args("require", kv_dtype=main, kv={"mamba_ssm_cache_dtype": state})
    target = modeling_v2_resolve(config, args)
    assert _qualified(target) == (
        "models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling."
        "ModelingV2KimiK31ProNvfp4Sm10xDefault"
    )


@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL"])
@pytest.mark.parametrize(
    "main,indexer,state",
    [
        (main, indexer, state)
        for main in ("fp8", "nvfp4")
        for indexer in ("fp8", "fp4")
        for state in ("float32", "bfloat16")
    ],
)
def test_nvfp4_moe_backends_route_with_independent_caches(backend, main, indexer, state):
    config = _nvfp4_model_config(indexer_k_dtype=indexer, **_TEP8)
    args = _llm_args(
        "require",
        kv_dtype=main,
        kv={"mamba_ssm_cache_dtype": state},
        moe_config=MoeConfig(backend=backend),
    )
    args.sparse_attention_config.indexer_k_dtype = indexer
    assert _qualified(modeling_v2_resolve(config, args)) == _NVFP4_TARGET


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (10, 2)])
def test_nvfp4_blackwell_target_rejects_other_architectures(monkeypatch, capability):
    _on_gpu(monkeypatch, capability)
    with pytest.raises(ValueError, match="sm_107"):
        modeling_v2_resolve(_nvfp4_model_config(**_TEP8), _llm_args("require"))
