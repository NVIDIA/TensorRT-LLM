# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 target routing, driven by synthetic configs on CPU."""

import json
import os
import re
import tempfile

import pytest
import torch
from transformers import PretrainedConfig

import tensorrt_llm._utils
from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.routing import (
    DEFAULT_TARGET_ENV,
    _mapping_mismatch,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MoeLoadBalancerConfig,
    MoePerLayerDynamicLoadBalancerConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.quant_config_utils import update_quant_config_from_compressed_tensors
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = "models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling.ModelingV2KimiK31ProMxfp4Sm10xDefault"
_TEP_TARGET = (
    "models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.modeling.ModelingV2KimiK31ProMxfp4Sm103Tep"
)
_DEP_TARGET = (
    "models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.modeling.ModelingV2KimiK31ProMxfp4Sm103Dep"
)
_TEP1 = dict(world_size=1, tp_size=1)
_TEP2 = dict(world_size=2, tp_size=2)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=2, moe_tp_size=2)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1)
_TEP8_MOETP2 = dict(world_size=8, tp_size=8, moe_ep_size=4, moe_tp_size=2)
_TEP8_MOETP4 = dict(world_size=8, tp_size=8, moe_ep_size=2, moe_tp_size=4)
_TEP16 = dict(world_size=16, tp_size=16, moe_ep_size=4, moe_tp_size=4)
_TEP32 = dict(world_size=32, tp_size=32, moe_ep_size=8, moe_tp_size=4)


def _dep(size: int) -> dict:
    return dict(world_size=size, tp_size=size, enable_attention_dp=True)


_DEP16 = _dep(16)


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
    """Route as if this were a GB300, wherever the test actually runs, with no target override."""
    _on_gpu(monkeypatch, (10, 3))
    monkeypatch.delenv(DEFAULT_TARGET_ENV, raising=False)


_DSA_CONFIG = dict(
    dsa_layers=[4, 8], index_topk=2048, index_num_heads=32, index_head_dim=128, index_rope_dim=0
)
# Layer kinds of the 8-layer test config (1-based, as the checkpoint numbers them): MLA on the
# DSA layers 4 and 8, KDA on the rest.
_LAYER_KINDS = dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7])


def _k31_config(text_overrides=None, quant_format="mxfp4-pack-quantized"):
    """The shape and layer layout Kimi K3.1 Pro's own config.json declares (layer count of a
    truncated copy)."""
    text = dict(
        num_hidden_layers=8,
        hidden_size=7168,
        num_experts=896,
        num_experts_per_token=16,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        vocab_size=163840,
        dsa_config=_DSA_CONFIG,
        linear_attn_config=_LAYER_KINDS,
        attn_res_block_size=12,
        routed_expert_hidden_size=3584,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
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
    use_cute_dsl_topk=False,
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
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            indexer_k_dtype=indexer_k_dtype, use_cute_dsl_topk=use_cute_dsl_topk
        )
        if sparse
        else None,
    )


_DSPARK = DSparkDecodingConfig(max_draft_len=5, speculative_model=_DUMMY_MODEL)


def _dflash_drafter(
    attention_mode="mla",
    max_draft_len=4,
    architecture="DFlash2DraftModel",
    context_kv_dtype=None,
    **extra,
) -> DFlashDecodingConfig:
    """DFlash over a drafter directory holding a DFlash2 config.json of ``attention_mode`` (plus ``extra``
    dflash_config entries)."""
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
    dflash_config.update(extra)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(
            dict(architectures=[architecture], model_type="qwen3", dflash_config=dflash_config),
            f,
        )
    config = DFlashDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=path, context_kv_dtype=context_kv_dtype
    )
    config.resolve_from_checkpoint()
    return config


_DFLASH2 = _dflash_drafter()
_DFLASH2_W8 = _dflash_drafter(max_draft_len=7)


_LAYOUTS = {
    "tep1": _TEP1,
    "tep2": _TEP2,
    "tep4-moetp2ep2": _TEP4,
    "tep8": _TEP8,
    "tep8-moetp2ep4": _TEP8_MOETP2,
    "tep8-moetp4ep2": _TEP8_MOETP4,
    "tep16-moetp4ep4": _TEP16,
    "tep32-moetp4ep8": _TEP32,
    "dep2": _dep(2),
    "dep4": _dep(4),
    "dep8": _dep(8),
    "dep16": _DEP16,
    "dep32": _dep(32),
}


# The layouts the SM 10.3 perf targets take; every other one runs the general target.
_SM103_PERF = {
    "tep8": _TEP_TARGET,
    "tep8-moetp2ep4": _TEP_TARGET,
    "tep8-moetp4ep2": _TEP_TARGET,
    "tep16-moetp4ep4": _TEP_TARGET,
    "tep32-moetp4ep8": _TEP_TARGET,
    "dep16": _DEP_TARGET,
}


@pytest.mark.parametrize("capability", [(10, 3), (10, 7)], ids=["sm103", "sm107"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize(
    "spec_config", [None, _DFLASH2, _DFLASH2_W8], ids=["no-spec", "dflash2-w5", "dflash2-w8"]
)
def test_k31_layouts_match(monkeypatch, capability, layout, spec_config):
    _on_gpu(monkeypatch, capability)
    config = _model_config(_k31_config(), spec_config=spec_config, **_LAYOUTS[layout])
    expected = _SM103_PERF.get(layout, _TARGET) if capability == (10, 3) else _TARGET
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == expected


@pytest.mark.parametrize(
    "capability, mapping_kwargs, expected",
    [
        ((10, 3), _TEP16, _TEP_TARGET),
        ((10, 3), _DEP16, _DEP_TARGET),
        ((10, 7), _TEP16, _TARGET),
        ((10, 7), _DEP16, _TARGET),
    ],
    ids=["sm103-tep16-tp", "sm103-dep16-dp", "sm107-tep16-general", "sm107-dep16-general"],
)
def test_k31_sm103_prefers_the_perf_target_of_its_layout(
    monkeypatch, capability, mapping_kwargs, expected
):
    _on_gpu(monkeypatch, capability)
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == expected


@pytest.mark.parametrize("layout", list(_SM103_PERF))
@pytest.mark.parametrize("value", ["1", "true", "ON"])
def test_k31_default_target_env_routes_to_the_general_target(monkeypatch, layout, value):
    monkeypatch.setenv(DEFAULT_TARGET_ENV, value)
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_LAYOUTS[layout])
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


@pytest.mark.parametrize("value", ["", "0", "false"])
def test_k31_default_target_env_off_keeps_the_perf_target(monkeypatch, value):
    monkeypatch.setenv(DEFAULT_TARGET_ENV, value)
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_TEP16)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TEP_TARGET


def test_k31_default_target_env_is_the_family_variable(monkeypatch):
    """The switch is K3.1's own TLLM_K31_DEFAULT_TARGET; the engine-wide name it replaced selects nothing."""
    assert DEFAULT_TARGET_ENV == "TLLM_K31_DEFAULT_TARGET"
    monkeypatch.setenv("TRTLLM_MODELING_V2_DEFAULT_TARGET", "1")
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_TEP16)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TEP_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs, env, expected",
    [(_TEP8, None, _TEP_TARGET), (_TEP8, "1", _TARGET), (_DEP16, None, _DEP_TARGET)],
    ids=["tep8", "tep8-default-target-env", "dep16"],
)
def test_k31_route_logs_the_target_it_picks(monkeypatch, mapping_kwargs, env, expected):
    if env is not None:
        monkeypatch.setenv(DEFAULT_TARGET_ENV, env)
    infos = []
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: infos.append(str(msg)))
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **mapping_kwargs)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(target) == expected
    line = f"modeling_v2: 'KimiK31ForConditionalGeneration' routes to {target.__module__}.{target.__qualname__}"
    assert line in infos, infos


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(world_size=16, tp_size=16),
        dict(world_size=16, tp_size=16, moe_ep_size=16, moe_tp_size=1),
        dict(world_size=8, tp_size=8, moe_ep_size=1, moe_tp_size=8),
        dict(world_size=32, tp_size=32, moe_ep_size=32, moe_tp_size=1),
        _dep(8),
        _dep(32),
    ],
    ids=["tep16-ep16", "tep16-moetp1ep16", "tep8-moetp8ep1", "tep32-ep32", "dep8", "dep32"],
)
def test_k31_sm103_other_layouts_run_the_general_target(mapping_kwargs):
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


_NVFP4_KV = dict(kv_dtype="nvfp4", kv_algo=QuantAlgo.NVFP4)
_FP4_INDEX_KEYS = dict(indexer_k_dtype="fp4")
# What the NoTarget of a latent KV cache other than FP8 or the checkpoint's BF16 asks for.
_FP8_KV_ONLY = re.escape(
    "latent KV cache (kv_cache_config.dtype='fp8') or the checkpoint's BF16 latents ('auto')"
)


def _layout_case(case: dict) -> tuple[dict, str]:
    """(model-config kwargs, kv_cache_config.dtype) of a routing case."""
    case = dict(case)
    return case, case.pop("kv_dtype", "fp8")


@pytest.mark.parametrize(
    "case, missed",
    [
        (_NVFP4_KV, _FP8_KV_ONLY),
        (_FP4_INDEX_KEYS, "indexer_k_dtype 'fp4'"),
    ],
    ids=["nvfp4-kv", "fp4-index-keys"],
)
@pytest.mark.parametrize("where", ["tep16", "dep16", "dep8", "sm107"])
def test_k31_refuses_nvfp4_kv_and_fp4_index_keys(monkeypatch, case, missed, where):
    """Every target runs FP8 latents and FP8 index keys: an NVFP4 latent KV cache or FP4 index keys find no
    target, on the perf targets' layouts too."""
    mapping_kwargs = {"tep16": _TEP16, "dep16": _DEP16, "dep8": _dep(8), "sm107": _DEP16}[where]
    if where == "sm107":
        _on_gpu(monkeypatch, (10, 7))
    case, kv_dtype = _layout_case(case)
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **case, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", kv_dtype)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", kv_dtype))


_STATIC_PLACEMENT = MoeLoadBalancerConfig(
    num_slots=896,
    initial_global_assignments={layer: list(range(896))[::-1] for layer in range(1, 8)},
)


@pytest.mark.parametrize(
    "load_balancer",
    [_STATIC_PLACEMENT, MoeLoadBalancerConfig(num_slots=896)],
    ids=["assignments", "no-assignments"],
)
def test_k31_dep_perf_target_admits_a_static_expert_placement(load_balancer):
    """DEP16 on sm_103 runs the attention-DP perf target with a static expert placement from the engine's EP load
    balancer configuration (one slot per expert, no online updates)."""
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_DEP16)
    args = _llm_args("require", moe_config=MoeConfig(load_balancer=load_balancer))
    assert _qualified(modeling_v2_resolve(config, args)) == _DEP_TARGET


@pytest.mark.parametrize(
    "load_balancer, missed",
    [
        (MoeLoadBalancerConfig(num_slots=912), "redundant expert slots"),
        (
            MoeLoadBalancerConfig(num_slots=896, layer_updates_per_iter=2),
            "layer_updates_per_iter=2",
        ),
    ],
    ids=["redundant-slots", "online"],
)
def test_k31_dep_perf_target_refuses_redundant_or_online_eplb(load_balancer, missed):
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_DEP16)
    args_kwargs = dict(moe_config=MoeConfig(load_balancer=load_balancer))
    assert modeling_v2_resolve(config, _llm_args("auto", **args_kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **args_kwargs))


@pytest.mark.parametrize("where", ["sm107", "default-target-env", "dep8", "tep16", "bf16-latents"])
def test_k31_general_and_tp_targets_refuse_a_static_expert_placement(monkeypatch, where):
    mapping_kwargs = _DEP16
    kv_algo, kv_dtype = QuantAlgo.FP8, "fp8"
    if where == "sm107":
        _on_gpu(monkeypatch, (10, 7))
    elif where == "default-target-env":
        monkeypatch.setenv(DEFAULT_TARGET_ENV, "1")
    elif where == "dep8":
        mapping_kwargs = _dep(8)
    elif where == "bf16-latents":
        kv_algo, kv_dtype = None, "auto"
    else:
        mapping_kwargs = _TEP16
    config = _model_config(_k31_config(), spec_config=_DFLASH2, kv_algo=kv_algo, **mapping_kwargs)
    args = _llm_args("require", kv_dtype, moe_config=MoeConfig(load_balancer=_STATIC_PLACEMENT))
    with pytest.raises(ValueError, match="sm_103 DEP16 perf target") as refusal:
        modeling_v2_resolve(config, args)
    # Each refusal names its fix; on the DEP16 layout that is unsetting the default-target switch.
    fix = (
        f"unset {DEFAULT_TARGET_ENV}"
        if where == "default-target-env"
        else "drop moe_config.load_balancer"
    )
    assert fix in str(refusal.value)


# Main's acceptance configuration: 1024 slots over 16 ranks (56 resident experts + 8 helper slots each).
_PER_LAYER_DYNAMIC = MoePerLayerDynamicLoadBalancerConfig(
    num_slots=1024, auxiliary_sms=8, helper_slot_mode="shared_banks"
)


@pytest.mark.parametrize(
    "load_balancer",
    [
        _PER_LAYER_DYNAMIC,
        MoePerLayerDynamicLoadBalancerConfig(num_slots=912),
        dict(
            mode="per_layer_dynamic",
            num_slots=1024,
            auxiliary_sms=8,
            helper_slot_mode="shared_banks",
        ),
    ],
    ids=["1024-shared-banks", "912-per-layer-resident", "yaml-dict"],
)
def test_k31_dep_perf_target_admits_per_layer_dynamic_eplb(load_balancer):
    """DEP16 on sm_103 runs the attention-DP perf target with per-layer dynamic EPLB: helper slots on every EP
    rank, num_slots above the 896 experts and a multiple of the 16 ranks."""
    config = _model_config(_k31_config(), **_DEP16)
    kwargs = dict(moe_config=MoeConfig(load_balancer=load_balancer))
    args = _llm_args("require", **kwargs)
    assert args.moe_config.backend == "TRTLLM"
    assert args.cuda_graph_config is None
    assert _qualified(modeling_v2_resolve(config, args)) == _DEP_TARGET


@pytest.mark.parametrize(
    "num_slots, missed",
    [(896, "num_slots=896"), (900, "num_slots=900"), (1000, "split evenly over the 16 EP ranks")],
    ids=["no-helpers", "uneven", "not-a-multiple-of-16"],
)
def test_k31_dep_perf_target_refuses_per_layer_dynamic_eplb_without_even_helpers(num_slots, missed):
    config = _model_config(_k31_config(), **_DEP16)
    args_kwargs = dict(
        moe_config=MoeConfig(
            load_balancer=MoePerLayerDynamicLoadBalancerConfig(num_slots=num_slots)
        )
    )
    assert modeling_v2_resolve(config, _llm_args("auto", **args_kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **args_kwargs))


@pytest.mark.parametrize("where", ["sm107", "default-target-env", "dep8", "tep16", "bf16-latents"])
def test_k31_general_and_tp_targets_refuse_per_layer_dynamic_eplb(monkeypatch, where):
    mapping_kwargs = _DEP16
    kv_algo, kv_dtype = QuantAlgo.FP8, "fp8"
    if where == "sm107":
        _on_gpu(monkeypatch, (10, 7))
    elif where == "default-target-env":
        monkeypatch.setenv(DEFAULT_TARGET_ENV, "1")
    elif where == "dep8":
        mapping_kwargs = _dep(8)
    elif where == "bf16-latents":
        kv_algo, kv_dtype = None, "auto"
    else:
        mapping_kwargs = _TEP16
    config = _model_config(_k31_config(), kv_algo=kv_algo, **mapping_kwargs)
    args = _llm_args("require", kv_dtype, moe_config=MoeConfig(load_balancer=_PER_LAYER_DYNAMIC))
    with pytest.raises(
        ValueError, match="per-layer dynamic EPLB .*runs on the sm_103 DEP16 perf target"
    ):
        modeling_v2_resolve(config, args)


def test_k31_dep_perf_target_refuses_fp8_ds_mla():
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_DEP16)
    with pytest.raises(ValueError, match=_FP8_KV_ONLY):
        modeling_v2_resolve(config, _llm_args("require", "fp8_ds_mla"))


@pytest.mark.parametrize(
    "mapping_kwargs, expected",
    [(_TEP8, _TEP_TARGET), (_DEP16, _DEP_TARGET), (_TEP2, _TARGET)],
    ids=["tep8", "dep16", "tep2"],
)
def test_k31_fp8_kv_matches(mapping_kwargs, expected):
    config = _model_config(_k31_config(), kv_algo=None, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", "fp8"))) == expected


@pytest.mark.parametrize(
    "state_dtype, admitted",
    [("auto", True), ("float32", True), ("float16", False), ("bfloat16", False)],
)
def test_k31_kda_state_dtype(state_dtype, admitted):
    """The KDA recurrent state runs in FP32: 'auto' (FP32 for Kimi) and 'float32' route, half precision does not."""
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_TEP8)
    kv = dict(mamba_ssm_cache_dtype=state_dtype)
    if admitted:
        assert _qualified(modeling_v2_resolve(config, _llm_args("require", kv=kv))) == _TEP_TARGET
    else:
        with pytest.raises(ValueError, match="KDA recurrent state in FP32"):
            modeling_v2_resolve(config, _llm_args("require", kv=kv))


@pytest.mark.parametrize("capability", [(10, 3), (10, 7)], ids=["sm103", "sm107"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("spec_config", [None, _DFLASH2], ids=["no-spec", "dflash2"])
def test_k31_auto_kv_keeps_bf16_latents_on_the_general_target(
    monkeypatch, capability, layout, spec_config
):
    """``auto`` over a checkpoint without a KV-cache scheme keeps its BF16 latents: every layout, the perf layouts
    on sm_103 included, runs the general target, and routing says so at INFO."""
    _on_gpu(monkeypatch, capability)
    infos = []
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: infos.append(str(msg)))
    config = _model_config(_k31_config(), spec_config=spec_config, kv_algo=None, **_LAYOUTS[layout])
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode, "auto"))) == _TARGET
    assert any("BF16 latent KV cache" in msg for msg in infos), infos


@pytest.mark.parametrize("layout", list(_SM103_PERF))
def test_k31_auto_kv_over_a_declared_fp8_scheme_keeps_the_perf_target(layout):
    """``auto`` is the checkpoint's KV-cache algorithm: an FP8 one runs the perf targets as ``fp8`` does."""
    config = _model_config(_k31_config(), kv_algo=QuantAlgo.FP8, **_LAYOUTS[layout])
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args("require", "auto"))) == _SM103_PERF[layout]
    )


@pytest.mark.parametrize(
    "kv_dtype, kv_algo",
    [("bfloat16", None), ("nvfp4", QuantAlgo.NVFP4), ("auto", QuantAlgo.NVFP4)],
    ids=["bfloat16", "nvfp4", "auto-over-nvfp4-checkpoint"],
)
@pytest.mark.parametrize("mapping_kwargs", [_TEP8, _DEP16, _TEP2], ids=["tep8", "dep16", "tep2"])
def test_k31_other_kv_dtypes_do_not_match(mapping_kwargs, kv_dtype, kv_algo):
    """Only FP8 latents and the checkpoint's BF16 latents (``auto``) route; ``bfloat16`` is not an engine KV dtype."""
    config = _model_config(_k31_config(), kv_algo=kv_algo, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", kv_dtype)) is None
    with pytest.raises(ValueError, match=_FP8_KV_ONLY):
        modeling_v2_resolve(config, _llm_args("require", kv_dtype))


def test_k31_refusal_logs_a_warning_with_the_fix(monkeypatch):
    """K3.1 has no Modeling V1 fallback, so a refusal is logged with its fix, not only raised."""
    config = _model_config(_k31_config(), kv_algo=None, **_TEP8)
    args = _llm_args("auto", "bfloat16")
    warnings = []
    monkeypatch.setattr(logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    assert modeling_v2_resolve(config, args) is None
    refusals = [msg for msg in warnings if "Kimi K3.1" in msg]
    assert len(refusals) == 1, warnings
    assert "kv_cache_config.dtype='fp8'" in refusals[0]


def test_k31_disagg_with_python_nixl_matches():
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_TEP8)
    transceiver = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    args = _llm_args("require", cache_transceiver_config=transceiver)
    assert _qualified(modeling_v2_resolve(config, args)) == _TEP_TARGET


def _near_miss(config_kwargs=None, text=None, quant_format="mxfp4-pack-quantized", **args_kwargs):
    config_kwargs = dict(config_kwargs or {})
    mapping = config_kwargs.pop("mapping", _TEP8)
    config = _model_config(_k31_config(text, quant_format), **config_kwargs, **mapping)
    return config, args_kwargs


# (config / LLM-argument changes from a matching TEP8 deployment, what the NoTarget names).
_NEAR_MISSES = {
    "other-shape": (dict(text=dict(hidden_size=6144)), "shape"),
    "no-dsa-config": (dict(text=dict(dsa_config=None)), "dsa_config"),
    "no-layer-kinds": (dict(text=dict(linear_attn_config=None)), "no linear_attn_config"),
    "layer-of-two-kinds": (
        dict(text=dict(linear_attn_config=dict(_LAYER_KINDS, kda_layers=[1, 2, 3, 4, 5, 6, 7]))),
        "layer 4 must be exactly one of",
    ),
    "layer-of-no-kind": (
        dict(text=dict(linear_attn_config=dict(_LAYER_KINDS, kda_layers=[1, 2, 3, 5, 6]))),
        "layer 7 must be exactly one of",
    ),
    "dense-mla-layer": (
        dict(text=dict(dsa_config=dict(_DSA_CONFIG, dsa_layers=[4]))),
        "MLA layers \\[8\\] are not in dsa_config.dsa_layers",
    ),
    "no-attn-res-block-size": (
        dict(text=dict(attn_res_block_size=None)),
        "attn_res_block_size None",
    ),
    "no-latent-moe": (
        dict(text=dict(routed_expert_hidden_size=None)),
        "routed_expert_hidden_size",
    ),
    "latent-moe-without-norm": (
        dict(text=dict(latent_moe_use_norm=False)),
        "latent_moe_use_norm",
    ),
    "no-situ-linear-beta": (
        dict(text=dict(activation_situ_linear_beta=None)),
        "activation_situ_linear_beta",
    ),
    "nvfp4-experts": (dict(quant_format="nvfp4"), "MXFP4"),
    "no-dsa-sparse": (dict(config_kwargs=dict(sparse=False), sparse=False), "DSA"),
    "fp4-index-keys": (dict(config_kwargs=dict(indexer_k_dtype="fp4")), "indexer_k_dtype 'fp4'"),
    "cute-dsl-topk": (dict(config_kwargs=dict(use_cute_dsl_topk=True)), "use_cute_dsl_topk"),
    "flashinfer-attention": (dict(attn_backend="FLASHINFER"), "attn_backend 'FLASHINFER'"),
    "rotated-index-keys": (
        dict(text=dict(dsa_config=dict(_DSA_CONFIG, index_rope_dim=64))),
        "index_rope_dim 64",
    ),
    "no-index-rope-dim": (
        dict(text=dict(dsa_config={k: v for k, v in _DSA_CONFIG.items() if k != "index_rope_dim"})),
        "index_rope_dim None",
    ),
    "nvfp4-kv": (dict(kv_dtype="nvfp4"), _FP8_KV_ONLY),
    "fp8_ds_mla-kv": (dict(kv_dtype="fp8_ds_mla"), _FP8_KV_ONLY),
    "bf16-kda-state": (
        dict(kv=dict(mamba_ssm_cache_dtype="bfloat16")),
        "KDA recurrent state in FP32",
    ),
    "mtp": (
        dict(config_kwargs=dict(spec_config=MTPDecodingConfig(max_draft_len=3))),
        "MTP",
    ),
    "dspark": (dict(config_kwargs=dict(spec_config=_DSPARK)), "DSPARK"),
    "dflash-gqa-drafter": (
        dict(config_kwargs=dict(spec_config=_dflash_drafter(attention_mode=None))),
        "MLA",
    ),
    "dflash-dspark-heads": (
        dict(config_kwargs=dict(spec_config=_dflash_drafter(markov_rank=8))),
        "DSpark heads",
    ),
    "dflash1-drafter": (
        dict(
            config_kwargs=dict(
                spec_config=_dflash_drafter(
                    architecture="DFlashDraftModel", conv_kernel_size=0, selector_rank=0
                )
            )
        ),
        "spec_dec_variant 1",
    ),
    "fp8-drafter-context": (
        dict(config_kwargs=dict(spec_config=_dflash_drafter(context_kv_dtype="fp8"))),
        "context_kv_dtype='fp8'",
    ),
    "pp2": (
        dict(config_kwargs=dict(mapping=dict(world_size=16, tp_size=8, pp_size=2))),
        "pipeline",
    ),
    "tep3": (
        dict(config_kwargs=dict(mapping=dict(world_size=3, tp_size=3))),
        "attention TP 3",
    ),
    "tep64": (
        dict(config_kwargs=dict(mapping=dict(world_size=64, tp_size=64))),
        "attention TP 64",
    ),
    "dep1": (dict(config_kwargs=dict(mapping=_dep(1))), "attention DP over 1"),
    "dep6": (dict(config_kwargs=dict(mapping=_dep(6))), "attention DP over 6"),
    "dep64": (dict(config_kwargs=dict(mapping=_dep(64))), "attention DP over 64"),
    "dep8-moe-tp": (
        dict(
            config_kwargs=dict(
                mapping=dict(
                    world_size=8,
                    tp_size=8,
                    moe_tp_size=2,
                    moe_ep_size=4,
                    enable_attention_dp=True,
                )
            )
        ),
        "EP-only",
    ),
    "moe-cutlass": (
        dict(moe_config=MoeConfig(backend="CUTLASS")),
        "TRTLLM MoE backend",
    ),
    "softmax-router": (dict(text=dict(moe_router_activation_func="softmax")), "sigmoid"),
    "no-renormalize": (dict(text=dict(moe_renormalize=False)), "renormalizes"),
    "expert-groups": (dict(text=dict(num_expert_group=8, topk_group=4)), "one expert group"),
    "routed-expert-width": (dict(text=dict(moe_intermediate_size=2880)), "3072 wide"),
    "eplb": (
        dict(moe_config=MoeConfig(load_balancer=MoeLoadBalancerConfig(num_slots=896))),
        "load balancer",
    ),
    "kv-manager-v1": (dict(kv=dict(use_kv_cache_manager_v2=False)), "V2"),
    "beam2": (dict(max_beam_width=2), "beam width"),
    "kv-pool-rebalance": (dict(kv=dict(enable_kv_pool_rebalance=True)), "rebalance"),
    "tokens-per-block-16": (dict(kv=dict(tokens_per_block=16)), "tokens_per_block 16"),
    "disagg-ucx": (
        dict(cache_transceiver_config=CacheTransceiverConfig(backend="UCX")),
        "Python NIXL",
    ),
    "disagg-cpp-transceiver": (
        dict(
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="CPP"
            )
        ),
        "Python NIXL",
    ),
}


@pytest.mark.parametrize("case", list(_NEAR_MISSES.values()), ids=list(_NEAR_MISSES))
def test_k31_near_misses_do_not_match(case):
    changes, missed = case
    changes = dict(changes)
    config, args_kwargs = _near_miss(
        changes.pop("config_kwargs", None),
        changes.pop("text", None),
        changes.pop("quant_format", "mxfp4-pack-quantized"),
    )
    kv_dtype = changes.pop("kv_dtype", "fp8")
    sparse = changes.pop("sparse", True)
    assert modeling_v2_resolve(config, _llm_args("auto", kv_dtype, sparse, **changes)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", kv_dtype, sparse, **changes))


@pytest.mark.parametrize("value", ["DEEPEP", "nvlink_two_sided", "NCCL_EP"])
def test_k31_refuses_moe_exchanges_it_does_not_run(monkeypatch, value):
    """Under attention DP the routed MoE exchanges tokens with the one-sided all-to-all or the all-gather only."""
    monkeypatch.setenv("TRTLLM_FORCE_COMM_METHOD", value)
    config = _model_config(_k31_config(), **_DEP16)
    with pytest.raises(ValueError, match="TRTLLM_FORCE_COMM_METHOD"):
        modeling_v2_resolve(config, _llm_args("require"))
    # Attention TP exchanges no tokens: the variable is not read.
    assert modeling_v2_resolve(_model_config(_k31_config(), **_TEP8), _llm_args("require"))


@pytest.mark.parametrize("value", ["NVLINK_ONE_SIDED", "allgather"])
def test_k31_admits_the_moe_exchanges_it_runs(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_FORCE_COMM_METHOD", value)
    config = _model_config(_k31_config(), **_DEP16)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _DEP_TARGET


def test_k31_attention_dp_needs_an_mpi_session(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31 import routing

    monkeypatch.setattr(routing, "mpi_disabled", lambda: True)
    with pytest.raises(ValueError, match="MPI session"):
        modeling_v2_resolve(_model_config(_k31_config(), **_DEP16), _llm_args("require"))
    assert modeling_v2_resolve(_model_config(_k31_config(), **_TEP8), _llm_args("require"))


@pytest.mark.parametrize("value", ["auto", "cute-dsl", "flashinfer"])
def test_k31_refuses_the_engine_low_m_gemm_dispatcher(monkeypatch, value):
    """The target's BF16 projections take no branch of the engine's low-M GEMM dispatcher, so its switch is refused."""
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", value)
    config = _model_config(_k31_config(), **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="TRTLLM_LOW_M_GEMM_BACKEND"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_k31_refuses_the_cute_dsl_bf16_gemm():
    """The target's BF16 projections take no CuTe DSL BF16 GEMM branch, so the switch is refused."""
    config = _model_config(_k31_config(), **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto", use_cute_dsl_bf16_gemm=True)) is None
    with pytest.raises(ValueError, match="use_cute_dsl_bf16_gemm"):
        modeling_v2_resolve(config, _llm_args("require", use_cute_dsl_bf16_gemm=True))


@pytest.mark.parametrize("value", ["off", "cublas", ""])
def test_k31_admits_the_engine_low_m_gemm_dispatcher_off(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", value)
    config = _model_config(_k31_config(), **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("require")) is not None


# Kimi K3.1 Pro's own text quantization_config (compressed-tensors): MXFP4 routed experts, the
# attention, shared and dense MLPs, router, latent projections, LM head and vision left BF16.
_CHECKPOINT_IGNORE = [
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    "re:.*block_sparse_moe\\.gate$",
    "re:.*block_sparse_moe\\.routed_expert_(up|down)_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
]


def _checkpoint_quant_config(ignore=tuple(_CHECKPOINT_IGNORE)) -> QuantConfig:
    """The engine's QuantConfig for the checkpoint's recipe, with ``ignore`` as its BF16 modules."""
    quant = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    update_quant_config_from_compressed_tensors(
        quant,
        {
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "group_size": 32,
                        "num_bits": 4,
                        "strategy": "group",
                        "type": "float",
                    },
                }
            },
            "format": "mxfp4-pack-quantized",
            "ignore": list(ignore),
            "quant_method": "compressed-tensors",
        },
    )
    return quant


def test_k31_checkpoint_quantization_matches(monkeypatch):
    """The checkpoint's recipe keeps every attention projection BF16, so it routes."""
    # Pin the general target: on SM 10.3 this layout would otherwise pick the attention-TP perf target.
    monkeypatch.setenv(DEFAULT_TARGET_ENV, "1")
    config = _model_config(_k31_config(), quant_config=_checkpoint_quant_config(), **_TEP8)
    assert config.quant_config.quant_algo is not None
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


# (quant_config, quant_config_dict, what the NoTarget names) of attention the target does not load.
_QUANTIZED_ATTENTION = {
    "fp8-block-scales": (
        QuantConfig(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
            group_size=128,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=_CHECKPOINT_IGNORE[1:],
        ),
        None,
        "model.layers.0.self_attn.q_proj is quantized as FP8_BLOCK_SCALES",
    ),
    "fp8-block-scaled-mla-kv-b": (
        QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=QuantAlgo.FP8),
        {
            "language_model.model.layers.7.self_attn.kv_b_proj": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
        },
        "model.layers.7.self_attn.kv_b_proj is quantized as FP8_BLOCK_SCALES",
    ),
    "mxfp4-attention": (
        _checkpoint_quant_config(ignore=_CHECKPOINT_IGNORE[1:]),
        None,
        "is quantized as W4A16_MXFP4",
    ),
}


@pytest.mark.parametrize(
    "quant_config, quant_config_dict, missed",
    list(_QUANTIZED_ATTENTION.values()),
    ids=list(_QUANTIZED_ATTENTION),
)
def test_k31_quantized_attention_does_not_match(quant_config, quant_config_dict, missed):
    """The target loads BF16 attention projections: an FP8 block-scaled (or any quantized) one,
    declared globally or per layer, gets a NoTarget instead of a load-time failure."""
    config = _model_config(
        _k31_config(),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        **_TEP8,
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed) as refusal:
        modeling_v2_resolve(config, _llm_args("require"))
    assert "ignore 're:.*self_attn.*'" in str(refusal.value)


@pytest.mark.parametrize(
    "excluded, missed",
    [
        (
            "re:.*block_sparse_moe\\.experts.*",
            "language_model.model.layers.0.block_sparse_moe.experts are excluded",
        ),
        (
            "language_model.layers.5.block_sparse_moe.experts",
            "model.layers.5.block_sparse_moe.experts are excluded",
        ),
    ],
    ids=["all-layers", "one-layer-wrapper-spelling"],
)
def test_k31_unquantized_routed_experts_do_not_match(excluded, missed):
    """The routed experts load as MXFP4 only: an exclusion from quantization that reaches them, in the
    checkpoint's or the wrapper's spelling, gets a NoTarget instead of a load-time failure."""
    quant = _checkpoint_quant_config(ignore=[*_CHECKPOINT_IGNORE, excluded])
    config = _model_config(_k31_config(), quant_config=quant, **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError) as refusal:
        modeling_v2_resolve(config, _llm_args("require"))
    assert missed in str(refusal.value)


@pytest.mark.parametrize("context_kv_dtype", [None, "auto"])
def test_k31_drafter_context_in_its_own_dtype_matches(context_kv_dtype):
    spec = _dflash_drafter(context_kv_dtype=context_kv_dtype)
    config = _model_config(_k31_config(), spec_config=spec, **_TEP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TEP_TARGET


def test_k31_draft_len_schedule_does_not_match():
    spec = _dflash_drafter()
    spec.draft_len_schedule = {4: 4, 64: 2}
    config = _model_config(_k31_config(), spec_config=spec, **_TEP8)
    with pytest.raises(ValueError, match="draft_len_schedule"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "gate",
    [
        dict(acceptance_rate_window_size=16, acceptance_rate_threshold=0.3),
        dict(acceptance_rate_threshold=0.3),
        dict(acceptance_rate_window_size=16),
    ],
    ids=["window-and-threshold", "threshold", "window"],
)
def test_k31_speculation_gate_does_not_match(gate):
    """A gate trip turns drafting off mid-flight and changes the KDA replay width."""
    spec = _dflash_drafter()
    for key, value in gate.items():
        setattr(spec, key, value)
    config = _model_config(_k31_config(), spec_config=spec, **_TEP8)
    with pytest.raises(ValueError, match="speculation gate"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_k31_lm_head_tp_in_adp_with_spec_does_not_match():
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_DEP16)
    args = _llm_args("require", enable_lm_head_tp_in_adp=True)
    with pytest.raises(ValueError, match="enable_lm_head_tp_in_adp"):
        modeling_v2_resolve(config, args)
    no_spec = _model_config(_k31_config(), **_DEP16)
    assert _qualified(modeling_v2_resolve(no_spec, args)) == _DEP_TARGET


@pytest.mark.parametrize(
    "capability, missed",
    [((10, 0), "sm_100 is not validated yet"), ((9, 0), "sm_103"), ((12, 0), "sm_103")],
    ids=["sm100", "sm90", "sm120"],
)
def test_k31_compute_capability(monkeypatch, capability, missed):
    _on_gpu(monkeypatch, capability)
    config = _model_config(_k31_config(), **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def _mapping(tp, adp=False, moe=None, pp=1, cp=1):
    """A Mapping stand-in: the routing reads only these fields."""
    from types import SimpleNamespace

    moe_tp, moe_ep = moe if moe is not None else (None, None)
    return SimpleNamespace(
        tp_size=tp,
        pp_size=pp,
        cp_size=cp,
        has_cp_helix=lambda: False,
        cp_config={},
        enable_attention_dp=adp,
        moe_tp_ep_user_specified=moe is not None,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )


@pytest.mark.parametrize(
    "mapping, missed",
    [
        (_mapping(8, moe=(2, 2)), "must equal tp_size"),
        (_mapping(32, moe=(1, 32)), None),
        (_mapping(32, moe=(32, 1)), None),
        (_mapping(16, adp=True, moe=(1, 16)), None),
        (_mapping(8, cp=2), "context parallelism ULYSSES: the Kimi K3.1 target runs Helix CP"),
    ],
    ids=["moe-split-product", "ep32", "moe-tp-32", "dep16-ep16", "cp2"],
)
def test_k31_moe_split_rules(mapping, missed):
    """MoE split rules no Mapping built from LLM arguments reaches (the split product)."""
    reason = _mapping_mismatch(mapping)
    if missed is None:
        assert reason is None
    else:
        assert reason is not None and missed in reason


@pytest.mark.parametrize(
    "mapping, missed",
    [
        (_mapping(64, moe=(64, 1)), "must divide 96"),
        (_mapping(3, moe=(1, 3)), "must divide the 896"),
    ],
    ids=["moe-tp-64", "moe-ep-3"],
)
def test_k31_head_and_expert_divisibility_rules(monkeypatch, mapping, missed):
    """96 % moe_tp and 896 % moe_ep: every admitted attention TP size satisfies both, so widen the
    admitted sizes to reach the rules themselves."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31 import routing

    monkeypatch.setattr(routing, "_ATTENTION_TP_SIZES", (1, 2, 3, 4, 8, 16, 32, 64))
    reason = _mapping_mismatch(mapping)
    assert reason is not None and missed in reason


def _helix_config(cp, spec_config=None):
    return _model_config(
        _k31_config(),
        spec_config=spec_config,
        world_size=cp,
        tp_size=1,
        cp_size=cp,
        cp_config={"cp_type": "HELIX"},
        moe_ep_size=cp,
        moe_tp_size=1,
    )


@pytest.mark.parametrize("cp", [2, 4, 8])
@pytest.mark.parametrize("spec_config", [None, _DFLASH2, _DFLASH2_W8])
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("overlap", [False, True], ids=["no-overlap", "overlap"])
def test_k31_helix_routes_with_managed_cache(cp, spec_config, graphs, overlap):
    # With graphs: the padded batch sizes a helix generation server at batch 16 captures.
    graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True)
    args = _llm_args(
        mode="require",
        cuda_graph_config=graph_config if graphs else None,
        disable_overlap_scheduler=not overlap,
        kv={"enable_block_reuse": False},
    )
    assert _qualified(modeling_v2_resolve(_helix_config(cp, spec_config), args)) == _TARGET


@pytest.mark.parametrize("cp", [2, 8])
def test_k31_helix_with_block_reuse_does_not_match(cp):
    config = _helix_config(cp, _DFLASH2)
    assert modeling_v2_resolve(config, _llm_args(kv={"enable_block_reuse": True})) is None
    with pytest.raises(ValueError, match="Helix requires block reuse disabled"):
        modeling_v2_resolve(config, _llm_args("require", kv={"enable_block_reuse": True}))
