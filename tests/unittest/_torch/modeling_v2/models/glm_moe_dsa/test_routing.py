# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 target routing, driven by synthetic configs on CPU."""

import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2 import _router_index
from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(_router_index, "_RESOLVED_MODES", _router_index._RESOLVED_MODES.copy())
    # Routing selects a class; these synthetic CPU configs do not construct it.
    module = ModuleType(
        "tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa."
        "glm_5_2_nvfp4__sm_107__low_latency.modeling"
    )
    module.ModelingV2Glm52Nvfp4Sm107LowLatency = type(
        "ModelingV2Glm52Nvfp4Sm107LowLatency", (), {"__module__": module.__name__}
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it.

    The switch is an LLM API argument, so the tests set one too -- that is the
    surface under test.
    """
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(
    pretrained_config, spec_config=None, kv_cache_quant_algo=QuantAlgo.FP8, **mapping_kwargs
):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec_config,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo),
    )


_GLM52_TARGET = (
    "models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__low_latency.modeling."
    "ModelingV2Glm52Nvfp4Sm107LowLatency"
)
_TP16 = dict(world_size=16, tp_size=16, moe_ep_size=1, moe_tp_size=16)
_TEP4_TARGET = (
    "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.modeling.ModelingV2Glm52Nvfp4Sm103Tep4"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)


@pytest.fixture
def _on_sm107(monkeypatch):
    """Route as if this were a Rubin GPU."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def _glm52_config(**overrides):
    """The shape GLM-5.2-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        hidden_size=6144,
        n_routed_experts=256,
        q_lora_rank=2048,
        index_n_heads=32,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


@pytest.mark.parametrize("tp", [4, 8, 16])
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config",
    [None, MTPDecodingConfig(max_draft_len=7), MTPDecodingConfig(max_draft_len=3)],
    ids=["no-spec", "mtp7", "mtp3"],
)
def test_glm52_supported_tp_matches(_on_sm107, spec_config, mode, tp):
    """Any draft length: the draft loop replays the one MTP module per step."""
    config = _model_config(
        _glm52_config(),
        spec_config=spec_config,
        world_size=tp,
        tp_size=tp,
        moe_ep_size=1,
        moe_tp_size=tp,
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GLM52_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, spec_config, missed",
    [
        (dict(num_hidden_layers=92), _TP16, None, r"shape \(92, 6144, 256, 2048\)"),
        (dict(index_n_heads=64), _TP16, None, "index_n_heads=64"),
        (dict(), dict(_TP16, enable_attention_dp=True), None, "attention_dp=True"),
        (dict(), dict(_TP16, moe_ep_size=16, moe_tp_size=1), None, "moe_ep=16"),
        (dict(), dict(world_size=2, tp_size=2, moe_tp_size=2), None, "world_size=2"),
        (dict(), _TP16, MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True), "MTP"),
        (dict(), _TP16, _NGRAM, "NGRAM"),
    ],
    ids=["other-shape", "heads", "attention-dp", "expert-parallel", "tp2", "vanilla-mtp", "ngram"],
)
def test_glm52_near_misses_do_not_match(
    _on_sm107, config_kwargs, mapping_kwargs, spec_config, missed
):
    config = _model_config(
        _glm52_config(**config_kwargs), spec_config=spec_config, **mapping_kwargs
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_glm52_on_sm103_does_not_match():
    """The target's kernels are built for sm107; a GB300 falls back."""
    config = _model_config(_glm52_config(), **_TP16)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("checkpoint_algo", [None, QuantAlgo.NVFP4])
def test_glm52_fp8_kv_cache_override_matches(_on_sm107, mode, checkpoint_algo):
    config = _model_config(_glm52_config(), kv_cache_quant_algo=checkpoint_algo, **_TP16)
    args = _llm_args(mode, kv_cache_config=KvCacheConfig(dtype="fp8"))
    assert _qualified(modeling_v2_resolve(config, args)) == _GLM52_TARGET


@pytest.mark.parametrize(
    "checkpoint_algo, dtype",
    [
        (None, "auto"),
        (QuantAlgo.NVFP4, "auto"),
        (QuantAlgo.FP8, "nvfp4"),
        (QuantAlgo.FP8, "fp8_ds_mla"),
    ],
)
def test_glm52_unsupported_kv_cache_does_not_match(_on_sm107, checkpoint_algo, dtype):
    config = _model_config(_glm52_config(), kv_cache_quant_algo=checkpoint_algo, **_TP16)
    args = dict(kv_cache_config=KvCacheConfig(dtype=dtype))
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


_GLM52_HIGH_THROUGHPUT_TARGET = (
    "models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__high_throughput.modeling."
    "ModelingV2Glm52Nvfp4Sm107HighThroughput"
)
_DEP16 = dict(world_size=16, tp_size=16, moe_ep_size=16, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("draft_len", [None, 5, 7])
@pytest.mark.parametrize("world_size", [4, 16])
def test_glm52_high_throughput_matches(_on_sm107, monkeypatch, mode, draft_len, world_size):
    # Exercise the real routing predicates on CPU without requiring the target's
    # native library. Native loading and execution are covered by GPU tests.
    _stub_target(monkeypatch, _GLM52_HIGH_THROUGHPUT_TARGET)
    config = _model_config(
        _glm52_config(index_n_heads=32),
        spec_config=MTPDecodingConfig(max_draft_len=draft_len) if draft_len is not None else None,
        **dict(_DEP16, world_size=world_size, tp_size=world_size, moe_ep_size=world_size),
    )
    config.moe_backend = "MEGAMOE_CUTEDSL"
    args = _llm_args(mode, kv_cache_config=KvCacheConfig(dtype="nvfp4"))
    assert _qualified(modeling_v2_resolve(config, args)) == _GLM52_HIGH_THROUGHPUT_TARGET


@pytest.mark.parametrize(
    "mapping_overrides, kv_dtype, backend, index_n_heads, missed",
    [
        (dict(enable_attention_dp=False), "nvfp4", "MEGAMOE_CUTEDSL", 32, "attention_dp=False"),
        (dict(moe_ep_size=1, moe_tp_size=16), "nvfp4", "MEGAMOE_CUTEDSL", 32, r"moe_ep=1\b"),
        (
            dict(world_size=8, tp_size=8, moe_ep_size=8),
            "nvfp4",
            "MEGAMOE_CUTEDSL",
            32,
            "world_size=8",
        ),
        (dict(), "fp8", "MEGAMOE_CUTEDSL", 32, r"KV cache .*FP8"),
        (dict(), "nvfp4", "CUTLASS", 32, "moe_backend=CUTLASS"),
        (dict(), "nvfp4", "MEGAMOE_CUTEDSL", 64, "index_n_heads=64"),
    ],
    ids=[
        "no-attention-dp",
        "no-expert-parallel",
        "dep8",
        "fp8-kv",
        "other-backend",
        "other-index-heads",
    ],
)
def test_glm52_high_throughput_near_misses(
    _on_sm107,
    mapping_overrides: dict,
    kv_dtype: str,
    backend: str,
    index_n_heads: int,
    missed: str,
) -> None:
    config = _model_config(
        _glm52_config(index_n_heads=index_n_heads), **dict(_DEP16, **mapping_overrides)
    )
    config.moe_backend = backend
    kwargs = dict(kv_cache_config=KvCacheConfig(dtype=kv_dtype))
    assert modeling_v2_resolve(config, _llm_args("auto", **kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **kwargs))


def test_glm52_high_throughput_declines_sm103() -> None:
    config = _model_config(_glm52_config(index_n_heads=32), **_DEP16)
    config.moe_backend = "MEGAMOE_CUTEDSL"
    args = _llm_args("auto", kv_cache_config=KvCacheConfig(dtype="nvfp4"))
    assert modeling_v2_resolve(config, args) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(
            config, _llm_args("require", kv_cache_config=KvCacheConfig(dtype="nvfp4"))
        )


_TEP4_DENSE_PROJECTIONS = [
    f"model.layers.{layer_idx}.mlp.{projection}"
    for layer_idx in range(3)
    for projection in ("gate_proj", "up_proj", "down_proj")
]
_TEP4_SHARED_MODULES = [
    f"model.layers.{layer_idx}.mlp.shared_experts" for layer_idx in range(3, 78)
]


def _load_tep4_quantization(
    tmp_path: Path,
    exclude_modules: list[str],
    linear_overrides: dict[str, str] | None,
    kv_cache_quant_algo: str | None = "FP8",
) -> tuple[QuantConfig, dict[str, QuantConfig] | None]:
    """Parse synthetic checkpoint metadata with the production ModelOpt loader."""
    quant_path = tmp_path / "hf_quant_config.json"
    quantization: dict[str, object] = {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": kv_cache_quant_algo,
        "group_size": 16,
        "exclude_modules": exclude_modules,
    }
    if linear_overrides is None:
        quantization["exclude_modules"] = [
            "lm_head",
            "model.embed_tokens",
            "model.layers.*.self_attn",
            "model.layers.*.mlp.gate",
            "model.layers.78",
            *exclude_modules,
        ]
    else:
        quantization["quant_algo"] = "MIXED_PRECISION"
        quantized_layers = {
            f"model.layers.{layer_idx}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
            for layer_idx in range(3, 78)
        }
        quantized_layers.update(
            {
                name: {"quant_algo": algorithm, "group_size": 16}
                for name, algorithm in linear_overrides.items()
            }
        )
        quantization["quantized_layers"] = quantized_layers
    quant_path.write_text(
        json.dumps({"producer": {"name": "modelopt"}, "quantization": quantization})
    )
    return ModelConfig.load_modelopt_quant_config(
        quant_path, tmp_path, "TRTLLM", architecture="GlmMoeDsaForCausalLM"
    )


@pytest.mark.parametrize(
    "exclude_modules,linear_overrides,mtp_enabled,missed",
    [
        pytest.param(
            ["model.layers.[012].mlp", *_TEP4_SHARED_MODULES],
            None,
            True,
            None,
            id="unquantized-dense-and-shared",
        ),
        pytest.param(
            [
                "model.layers.0*",
                "model.layers.1.*",
                "model.layers.2.*",
                *[f"{name}*" for name in _TEP4_SHARED_MODULES],
                "model.layers.78*",
            ],
            None,
            True,
            None,
            id="checkpoint-shaped-wildcards",
        ),
        pytest.param(
            [
                *[
                    name
                    for name in _TEP4_DENSE_PROJECTIONS
                    if name != "model.layers.0.mlp.gate_proj"
                ],
                *_TEP4_SHARED_MODULES,
            ],
            None,
            True,
            "model.layers.0.mlp.gate_proj",
            id="quantized-dense",
        ),
        pytest.param(
            [
                *[
                    name
                    for name in _TEP4_DENSE_PROJECTIONS
                    if name != "model.layers.2.mlp.down_proj"
                ],
                *_TEP4_SHARED_MODULES,
            ],
            None,
            True,
            "model.layers.2.mlp.down_proj",
            id="quantized-last-dense-down",
        ),
        pytest.param([], {}, True, None, id="mixed-routed-only"),
        pytest.param(
            [],
            {"model.layers.2.mlp.down_proj": "NVFP4"},
            True,
            "model.layers.2.mlp.down_proj",
            id="mixed-quantized-dense",
        ),
        pytest.param(
            [],
            {"model.layers.0.mlp.gate_up_proj": "NVFP4"},
            True,
            "model.layers.0.mlp.",
            id="mixed-fused-dense",
        ),
        pytest.param(
            [],
            {"model.layers.3.mlp.shared_experts.down_proj": "NVFP4"},
            True,
            "model.layers.3.mlp.shared_experts.down_proj",
            id="mixed-quantized-shared",
        ),
        pytest.param(
            [
                *_TEP4_DENSE_PROJECTIONS,
                *_TEP4_SHARED_MODULES[1:],
                "model.layers.3.mlp.shared_experts.gate_proj",
                "model.layers.3.mlp.shared_experts.up_proj",
            ],
            None,
            True,
            "model.layers.3.mlp.shared_experts.down_proj",
            id="quantized-shared-down",
        ),
        pytest.param(
            [],
            {"model.layers.78.mlp.shared_experts.down_proj": "NVFP4"},
            True,
            "model.layers.78.mlp.shared_experts.down_proj",
            id="mixed-quantized-mtp-shared",
        ),
        pytest.param(
            [],
            {"model.layers.78.mlp.shared_experts.down_proj": "NVFP4"},
            False,
            None,
            id="inactive-mtp-shared",
        ),
        pytest.param(
            [],
            {"model.layers.2.mlp.down_proj": "NO_QUANT"},
            True,
            None,
            id="explicit-no-quant",
        ),
        pytest.param(
            [
                "model.layers.[012].mlp.gate_up_proj",
                "model.layers.[012].mlp.down_proj",
                *_TEP4_SHARED_MODULES,
            ],
            None,
            True,
            None,
            id="fused-exclusions",
        ),
        pytest.param(
            [
                "model.layers.[012].mlp.gate_proj",
                "model.layers.[012].mlp.up_proj",
                "model.layers.[012].mlp.down_proj",
                *_TEP4_SHARED_MODULES,
            ],
            None,
            True,
            None,
            id="split-exclusions",
        ),
        pytest.param(
            ["model.layers.0.mlp.gate_proj"],
            {"model.layers.0.mlp.up_proj": "NVFP4"},
            True,
            "model.layers.0.mlp.up_proj",
            id="packed-sibling-of-excluded-projection",
        ),
        pytest.param(
            ["model.layers.0.mlp.gate_up_proj"],
            {"model.layers.0.mlp.up_proj": "NVFP4"},
            True,
            "model.layers.0.mlp.up_proj",
            id="packed-projection-behind-fused-exclusion",
        ),
    ],
)
def test_tep4_uses_effective_checkpoint_quantization(
    tmp_path: Path,
    exclude_modules: list[str],
    linear_overrides: dict[str, str] | None,
    mtp_enabled: bool,
    missed: str | None,
) -> None:
    quant_config, quant_config_dict = _load_tep4_quantization(
        tmp_path, exclude_modules, linear_overrides
    )
    if linear_overrides is None:
        assert quant_config.quant_algo == QuantAlgo.NVFP4
        for layer_idx in (10, 20):
            routed_projection = f"model.layers.{layer_idx}.mlp.experts.0.up_proj"
            assert not quant_config.is_module_excluded_from_quantization(routed_projection)
        for unpacked in (
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.3.mlp.gate",
            "model.layers.78.mlp.experts.0.up_proj",
            "lm_head",
        ):
            assert quant_config.is_module_excluded_from_quantization(unpacked)
    model_config = ModelConfig(
        pretrained_config=_glm52_config(
            first_k_dense_replace=3,
            index_topk_freq=4,
            index_skip_topk_offset=3,
            indexer_types=[
                "full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared"
                for layer_idx in range(78)
            ],
        ),
        mapping=Mapping(**_TEP4),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=32, index_head_dim=128, index_topk=2048
        ),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        spec_config=MTPDecodingConfig(max_draft_len=3) if mtp_enabled else None,
    )
    resolved = modeling_v2_resolve(model_config, _llm_args("auto"))
    if missed is None:
        assert _qualified(resolved) == _TEP4_TARGET
        assert modeling_v2_resolve(model_config, _llm_args("require")) is resolved
    else:
        assert resolved is None
        with pytest.raises(ValueError, match=re.escape(missed)):
            modeling_v2_resolve(model_config, _llm_args("require"))


@pytest.mark.parametrize(
    "indexer_types,missed",
    [
        pytest.param(None, "indexer_types absent", id="missing"),
        pytest.param(["full"] * 77, "indexer_types covers 77 of 78", id="short"),
        pytest.param(17, "indexer_types is int", id="scalar"),
        pytest.param(["shared"] * 78, "indexer_types[0]='shared'", id="all-shared"),
        pytest.param(
            ["shared"] + ["full"] * 77, "indexer_types[0]='shared'", id="shared-before-owner"
        ),
        pytest.param(["full"] + ["shared"] * 77, None, id="shared-after-owner"),
    ],
)
def test_tep4_requires_indexer_schedule(
    tmp_path: Path, indexer_types: list[str] | int | None, missed: str | None
) -> None:
    pretrained_config = _glm52_config(first_k_dense_replace=3)
    if indexer_types is not None:
        pretrained_config.indexer_types = indexer_types
    quant_config, quant_config_dict = _load_tep4_quantization(tmp_path, [], {})
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**_TEP4),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=32, index_head_dim=128, index_topk=2048
        ),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
    )
    if missed is None:
        resolved = modeling_v2_resolve(model_config, _llm_args("auto"))
        assert _qualified(resolved) == _TEP4_TARGET
        assert modeling_v2_resolve(model_config, _llm_args("require")) is resolved
        return
    assert modeling_v2_resolve(model_config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=re.escape(missed)):
        modeling_v2_resolve(model_config, _llm_args("require"))


def test_non_tep4_ignores_indexer_schedule(tmp_path: Path) -> None:
    quant_config, quant_config_dict = _load_tep4_quantization(tmp_path, [], {})
    model_config = ModelConfig(
        pretrained_config=_glm52_config(first_k_dense_replace=3, indexer_types=17),
        mapping=Mapping(world_size=2, tp_size=2, moe_ep_size=2, moe_tp_size=1),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
    )
    assert modeling_v2_resolve(model_config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="tep4 checkpoint not read"):
        modeling_v2_resolve(model_config, _llm_args("require"))


@pytest.mark.parametrize(
    "sharing_config,owns_cache",
    [
        pytest.param(
            dict(index_topk_freq=4, index_skip_topk_offset=3),
            True,
            id="frequency-provides-owner",
        ),
        pytest.param(
            dict(index_topk_freq=8, index_skip_topk_offset=3),
            False,
            id="frequency-omits-owner",
        ),
        pytest.param(
            dict(
                index_topk_freq=4,
                index_skip_topk_offset=3,
                index_topk_pattern=["F"] * 6 + ["S"],
            ),
            False,
            id="pattern-omits-owner",
        ),
        pytest.param(
            dict(
                index_topk_freq=8,
                index_skip_topk_offset=3,
                index_topk_pattern=["F"] * 7,
            ),
            True,
            id="pattern-provides-owner",
        ),
    ],
)
def test_tep4_requires_indexer_cache_pool(
    tmp_path: Path, sharing_config: dict, owns_cache: bool
) -> None:
    pretrained_config = _glm52_config(
        first_k_dense_replace=3,
        indexer_types=["full" if layer_idx in (0, 6) else "shared" for layer_idx in range(78)],
        **sharing_config,
    )
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=32, index_head_dim=128, index_topk=2048
    )
    assert (
        sparse_config.to_sparse_params(
            pretrained_config=pretrained_config, layer_idx=6
        ).is_full_indexer_layer
        is owns_cache
    )
    assert sparse_config.to_sparse_params(
        pretrained_config=pretrained_config, layer_idx=78
    ).is_full_indexer_layer
    quant_config, quant_config_dict = _load_tep4_quantization(tmp_path, [], {})
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**_TEP4),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        sparse_attention_config=sparse_config,
        spec_config=MTPDecodingConfig(max_draft_len=3),
    )
    resolved = modeling_v2_resolve(model_config, _llm_args("auto"))
    if owns_cache:
        assert _qualified(resolved) == _TEP4_TARGET
        assert modeling_v2_resolve(model_config, _llm_args("require")) is resolved
    else:
        assert resolved is None
        with pytest.raises(
            ValueError,
            match="indexer_types layer 6 has no indexer K-cache.*index_topk_pattern.*index_topk_freq",
        ):
            modeling_v2_resolve(model_config, _llm_args("require"))


@pytest.mark.parametrize(
    "omitted_layer",
    [None, 6, 78],
    ids=["missing-config", "trunk-instance-policy", "mtp-instance-policy"],
)
def test_tep4_uses_deployment_sparse_config(tmp_path: Path, omitted_layer: int | None) -> None:
    class SparseConfigWithoutPool(DeepSeekSparseAttentionConfig):
        @staticmethod
        def _is_full_indexer_layer(pretrained_config: object, layer_idx: int) -> bool:
            return layer_idx != omitted_layer

    quant_config, quant_config_dict = _load_tep4_quantization(tmp_path, [], {})
    model_config = ModelConfig(
        pretrained_config=_glm52_config(first_k_dense_replace=3, indexer_types=["full"] * 78),
        mapping=Mapping(**_TEP4),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        sparse_attention_config=SparseConfigWithoutPool(
            index_n_heads=32, index_head_dim=128, index_topk=2048
        )
        if omitted_layer is not None
        else None,
        spec_config=MTPDecodingConfig(max_draft_len=3) if omitted_layer == 78 else None,
    )
    missed = (
        f"layer {omitted_layer} has no indexer K-cache"
        if omitted_layer is not None
        else "sparse_attention_config absent"
    )
    assert modeling_v2_resolve(model_config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(model_config, _llm_args("require"))


@pytest.mark.parametrize("mixed_precision", [False, True], ids=["homogeneous", "per-linear"])
def test_tep4_preserves_kv_cache_override(tmp_path: Path, mixed_precision: bool) -> None:
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant

    packed = "model.layers.2.mlp.down_proj"
    excluded = [name for name in _TEP4_DENSE_PROJECTIONS if name != packed]
    quant_config, quant_config_dict = _load_tep4_quantization(
        tmp_path,
        [] if mixed_precision else [*excluded, *_TEP4_SHARED_MODULES],
        {packed: "NVFP4"} if mixed_precision else None,
        kv_cache_quant_algo=None,
    )
    model_config = ModelConfig(
        pretrained_config=_glm52_config(first_k_dense_replace=3, indexer_types=["full"] * 78),
        mapping=Mapping(**_TEP4),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=32, index_head_dim=128, index_topk=2048
        ),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
    )
    args = _llm_args("auto", kv_cache_config=KvCacheConfig(dtype="fp8"))
    resolved = modeling_v2_resolve(model_config, args)
    validate_and_set_kv_cache_quant(model_config, args.kv_cache_config.dtype)
    assert model_config.quant_config.layer_quant_mode.has_fp8_kv_cache()
    assert model_config.quant_config.quant_mode.has_fp8_kv_cache()
    if model_config.quant_config_dict is not None:
        assert all(
            config.layer_quant_mode.has_fp8_kv_cache() and config.quant_mode.has_fp8_kv_cache()
            for config in model_config.quant_config_dict.values()
        )
    assert resolved is None


_TP8_TARGET = "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.modeling.ModelingV2Glm52Nvfp4Sm103Tp8"


def _stub_target(monkeypatch, qualified_name):
    """Route to ``qualified_name`` on CPU without loading the target's native ops."""
    module_name, class_name = qualified_name.rsplit(".", 1)
    stub = ModuleType(f"tensorrt_llm._torch.modeling_v2.{module_name}")
    setattr(stub, class_name, type(class_name, (), {"__module__": stub.__name__}))
    monkeypatch.setitem(sys.modules, stub.__name__, stub)


def _glm53_tp8_config(**changes):
    """GLM-5.3-NVFP4 on tp8 with MTP5: its MIXED_PRECISION export has NVFP4 routed experts in
    layers 3-77 and FP8 dense MLPs before them. ``changes`` maps a module to an algorithm, or
    to None to drop it."""
    layers = {"model.layers.0.mlp.gate_up_proj": {"quant_algo": "FP8"}}
    layers.update({f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4"} for i in range(3, 78)})
    for name, algo in changes.items():
        if algo is None:
            layers.pop(name)
        else:
            layers[name] = {"quant_algo": algo}
    quantization = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "modelopt"},
        "ignore": ["model.layers.78*"],
        "quantized_layers": layers,
    }
    pretrained = _glm52_config(
        quantization_config=quantization,
        num_attention_heads=64,
        index_topk=2048,
        first_k_dense_replace=3,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    return _model_config(
        pretrained,
        spec_config=MTPDecodingConfig(max_draft_len=5),
        world_size=8,
        tp_size=8,
        moe_ep_size=1,
        moe_tp_size=8,
    )


def test_tp8_matches_glm53(monkeypatch):
    _stub_target(monkeypatch, _TP8_TARGET)
    config = _glm53_tp8_config()
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TP8_TARGET


@pytest.mark.parametrize(
    "changes, missed",
    [
        ({"model.layers.10.mlp.shared_experts.gate_up_proj": "NVFP4"}, "shared_experts"),
        ({"model.layers.10.mlp.experts": None}, r"routed experts in layers \[10\]"),
        ({"model.layers.78.mlp.experts": "NVFP4"}, "MTP layer quantized"),
        ({"model.layers.10.mlp.experts": "FP8"}, r"'model\.layers\.10\.mlp\.experts' as FP8"),
        ({"model.layers.10.self_attn.q_a_proj": "FP8"}, "self_attn.q_a_proj"),
        ({"lm_head": "FP8"}, "'lm_head'"),
    ],
    ids=["shared-experts", "missing-experts", "mtp-layer", "fp8-experts", "attention", "lm-head"],
)
def test_tp8_declines_other_mixed_precision(monkeypatch, changes, missed):
    _stub_target(monkeypatch, _TP8_TARGET)
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(_glm53_tp8_config(**changes), _llm_args("require"))
