# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 target routing, driven by synthetic configs on CPU.

Moonshot's MXFP4 checkpoint routes to the general target (``kimi_k31_pro_mxfp4__sm_10x__default``)
and, on sm_103's TEP and DEP16 layouts, to the perf targets (``kimi_k31_pro_mxfp4__sm_103__{tep,dep}``).
The ModelOpt NVFP4 export routes on sm_107 to the specialized targets
(``kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8}``) first, and on sm_103 and sm_107 to the NVFP4
default target (``kimi_k31_pro_nvfp4__sm_10x__default``) on the validated cells
(``routing._NVFP4_DEFAULT_CELLS``) the specialized targets do not take. The four NVFP4 targets are
stand-ins, so this exercises the decision in ``routing.py``, not the model. The ModelOpt configs are
shaped as the export's ``config.json`` and ``hf_quant_config.json`` and read through the engine's
ModelOpt parser, as the loader reads them; the server role reaches the config the model is built
from through the model loader's own config path. The two sides of a disaggregated pair the table
admits build the same KV cache layout (the specialized targets imported).
"""

import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import types
from dataclasses import replace

import pytest
import yaml
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import (
    ModelingV2Mode,
    engine_kv_cache_quant_algo,
    modeling_v2_resolve,
    resolved_mode,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31 import routing
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MoeLoadBalancerConfig,
    RocketSparseAttentionConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.quant_config_utils import update_quant_config_from_compressed_tensors
from tensorrt_llm.quantization.mode import QuantAlgo

__extra_import_path__ = ["."]
from _k31_builders import (  # noqa: E402 — needs the path declared above
    SM103,
    SM107,
    HfConfigOnly,
    checkpoint_dir,
    hf_quant_config,
    inline_quantization_config,
    on_gpu,
    quantized_layers,
)

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_FAMILY = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"


def _qualified(target) -> str:
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    on_gpu(monkeypatch, (10, 3))


def _dflash2(
    max_draft_len=4,
    attention_mode="mla",
    architecture="DFlash2DraftModel",
    fields=None,
    changes=None,
    drafter=None,
    **extra,
) -> DFlashDecodingConfig:
    """One-engine DFlash over a drafter directory holding a DFlash2 config.json of block
    ``max_draft_len + 1`` and ``attention_mode`` (plus ``extra`` dflash_config entries and
    ``drafter`` top-level entries). ``fields`` are further config fields, ``changes`` fields set
    once it read the checkpoint."""
    path = tempfile.mkdtemp(prefix="k31_dflash2_")
    dflash_config = dict(
        block_size=max_draft_len + 1,
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
            dict(
                architectures=[architecture],
                model_type="qwen3",
                dflash_config=dflash_config,
                **(drafter or {}),
            ),
            f,
        )
    config = DFlashDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=path, **(fields or {})
    )
    config.resolve_from_checkpoint()
    for field, value in (changes or {}).items():
        setattr(config, field, value)
    return config


_DFLASH2 = _dflash2()
_DSPARK = DSparkDecodingConfig(max_draft_len=5, speculative_model=_DUMMY_MODEL)

# --- Moonshot's MXFP4 checkpoint: the general target and the sm_103 perf targets ---

_TARGET = "models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling.ModelingV2KimiK31ProMxfp4Sm10xDefault"
_TEP_TARGET = (
    "models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.modeling.ModelingV2KimiK31ProMxfp4Sm103Tep"
)
_DEP_TARGET = (
    "models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.modeling.ModelingV2KimiK31ProMxfp4Sm103Dep"
)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1)


def _dep(size: int) -> dict:
    return dict(world_size=size, tp_size=size, enable_attention_dp=True)


def _helix(cp: int) -> dict:
    return dict(
        world_size=cp,
        tp_size=1,
        cp_size=cp,
        cp_config={"cp_type": "HELIX"},
        moe_ep_size=cp,
        moe_tp_size=1,
    )


_DEP16 = _dep(16)
_LAYOUTS = {
    "tep2": dict(world_size=2, tp_size=2),
    "tep8": _TEP8,
    "tep16-moetp4ep4": dict(world_size=16, tp_size=16, moe_ep_size=4, moe_tp_size=4),
    "dep8": _dep(8),
    "dep16": _DEP16,
}
# Of these layouts, the ones the SM 10.3 perf targets take; the others run the general target.
_SM103_PERF = {
    "tep8": _TEP_TARGET,
    "tep16-moetp4ep4": _TEP_TARGET,
    "dep16": _DEP_TARGET,
}

_DSA_CONFIG = dict(
    dsa_layers=[4, 8],
    index_topk=2048,
    index_num_heads=32,
    index_head_dim=128,
    index_rope_dim=0,
    index_k_norm_type="layernorm",
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


def _llm_args(
    mode="auto", kv_dtype="fp8", sparse=True, kv=None, indexer_k_dtype=None, **kwargs
) -> TorchLlmArgs:
    """``sparse``: True for a DSA config (naming ``indexer_k_dtype`` if given), False for none, else
    the sparse attention config itself."""
    if sparse is True:
        named = {} if indexer_k_dtype is None else dict(indexer_k_dtype=indexer_k_dtype)
        sparse = DeepSeekSparseAttentionConfig(**named)
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=KvCacheConfig(dtype=kv_dtype, **(kv or {})),
        sparse_attention_config=None if sparse is False else sparse,
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


def _deployment(monkeypatch, changes) -> tuple:
    """(model config, ``_llm_args`` arguments) of a TEP8 deployment on sm_103 with ``changes``:
    ``config_kwargs`` (``_model_config`` arguments, ``mapping`` among them), ``text`` and
    ``quant_format`` (the checkpoint's config), ``capability``, ``env`` (environment variables)
    and ``patch`` (``routing`` names replaced); the rest are ``_llm_args`` arguments."""
    changes = dict(changes)
    on_gpu(monkeypatch, changes.pop("capability", (10, 3)))
    for name, value in changes.pop("env", {}).items():
        monkeypatch.setenv(name, value)
    for name, value in changes.pop("patch", {}).items():
        monkeypatch.setattr(routing, name, value)
    config_kwargs = dict(changes.pop("config_kwargs", {}))
    mapping = config_kwargs.pop("mapping", _TEP8)
    pretrained = _k31_config(
        changes.pop("text", None), changes.pop("quant_format", "mxfp4-pack-quantized")
    )
    return _model_config(pretrained, **config_kwargs, **mapping), changes


# (capability, layout): each layout on sm_103, where each picks its own target; on sm_107, which
# has no perf target, one TEP layout. The speculation checks read neither the layout
# nor the GPU, so the layouts route without a drafter.
_LAYOUT_CASES = {
    **{f"sm103-{layout}": ((10, 3), layout) for layout in _LAYOUTS},
    "sm107-tep8": ((10, 7), "tep8"),
}


@pytest.mark.parametrize(
    "capability, layout", list(_LAYOUT_CASES.values()), ids=list(_LAYOUT_CASES)
)
def test_k31_layouts_match(monkeypatch, capability, layout):
    on_gpu(monkeypatch, capability)
    config = _model_config(_k31_config(), **_LAYOUTS[layout])
    expected = _SM103_PERF.get(layout, _TARGET) if capability == (10, 3) else _TARGET
    for mode in ("auto", "require"):
        target = modeling_v2_resolve(config, _llm_args(mode))
        assert _qualified(target) == expected
        assert resolved_mode(target) is ModelingV2Mode(mode)


# (checkpoint KV-cache algorithm, kv_cache_config.dtype, layout, capability, target): FP8 latents
# the arguments name or ``auto`` reads off the checkpoint run the perf targets; ``auto`` over a
# checkpoint without a KV-cache scheme keeps its BF16 latents on the general target. One condition
# guards every perf return, so one perf layout stands for the others.
_KV_CASES = {
    "fp8-tep8": (None, "fp8", "tep8", (10, 3), _TEP_TARGET),
    "auto-bf16-tep8-sm103": (None, "auto", "tep8", (10, 3), _TARGET),
    "auto-fp8-tep8": (QuantAlgo.FP8, "auto", "tep8", (10, 3), _TEP_TARGET),
}


@pytest.mark.parametrize("case", list(_KV_CASES.values()), ids=list(_KV_CASES))
def test_k31_kv_cache_dtype_picks_the_target(monkeypatch, case):
    kv_algo, kv_dtype, layout, capability, expected = case
    on_gpu(monkeypatch, capability)
    config = _model_config(_k31_config(), kv_algo=kv_algo, **_LAYOUTS[layout])
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode, kv_dtype))) == expected


# A static expert placement of every routed-MoE layer of the 8-layer config (layers 0-7), and the
# identity placement.
_STATIC_PLACEMENT = MoeLoadBalancerConfig(
    num_slots=896,
    initial_global_assignments={layer: list(range(896))[::-1] for layer in range(8)},
)
_IDENTITY_PLACEMENT = MoeLoadBalancerConfig(
    num_slots=896,
    initial_global_assignments={layer: list(range(896)) for layer in range(8)},
)
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


_DEP16_DFLASH2 = dict(config_kwargs=dict(mapping=_DEP16, spec_config=_DFLASH2))
# Deployments a target takes beyond the plain layouts: (``_deployment`` changes, target).
_ADMITTED = {
    # A placed expert runs on the fused routed MoE only: 4096 rows over 16 ranks.
    "dep16-static-expert-placement": (
        dict(
            _DEP16_DFLASH2,
            moe_config=MoeConfig(load_balancer=_STATIC_PLACEMENT),
            max_num_tokens=256,
        ),
        _DEP_TARGET,
    ),
    "dep16-identity-placement": (
        dict(_DEP16_DFLASH2, moe_config=MoeConfig(load_balancer=_IDENTITY_PLACEMENT)),
        _DEP_TARGET,
    ),
    "moe-backend-trtllm": (dict(moe_config=MoeConfig(backend="TRTLLM")), _TEP_TARGET),
    # Rejection sampling with the rule the targets run, named or left at its default.
    "rejection-sampling-default-method": (
        dict(config_kwargs=dict(spec_config=_dflash2(changes=dict(use_rejection_sampling=True)))),
        _TEP_TARGET,
    ),
    "rejection-sampling-block": (
        dict(
            config_kwargs=dict(
                spec_config=_dflash2(
                    changes=dict(use_rejection_sampling=True, rejection_sample_method="block")
                )
            )
        ),
        _TEP_TARGET,
    ),
    # Attention TP exchanges no tokens: neither the exchange variable nor the MPI session is read.
    "tep8-exchange-and-mpi-unread": (
        dict(
            env=dict(TRTLLM_FORCE_COMM_METHOD="DEEPEP"),
            patch=dict(mpi_disabled=lambda: True),
        ),
        _TEP_TARGET,
    ),
    "disagg-python-nixl": (
        dict(
            config_kwargs=dict(spec_config=_DFLASH2),
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="PYTHON"
            ),
        ),
        _TEP_TARGET,
    ),
    # The checkpoint's recipe keeps every attention projection BF16.
    "checkpoint-quantization": (
        dict(config_kwargs=dict(quant_config=_checkpoint_quant_config())),
        _TEP_TARGET,
    ),
}


@pytest.mark.parametrize("case", list(_ADMITTED.values()), ids=list(_ADMITTED))
def test_k31_admitted_deployments(monkeypatch, case):
    changes, expected = case
    config, args = _deployment(monkeypatch, changes)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", **args))) == expected


# What the NoTarget of a latent KV cache other than FP8 or the checkpoint's BF16 asks for.
_FP8_KV_ONLY = re.escape(
    "latent KV cache (kv_cache_config.dtype='fp8') or the checkpoint's BF16 latents ('auto')"
)
_LOAD_BALANCER = ("sm_103 DEP16 perf target", "drop moe_config.load_balancer")

# (``_deployment`` changes, what the NoTarget names: a pattern or a tuple of them).
_NEAR_MISSES = {
    "other-shape": (dict(text=dict(hidden_size=6144)), "shape"),
    "no-dsa-config": (dict(text=dict(dsa_config=None)), "dsa_config"),
    "layer-of-two-kinds": (
        dict(text=dict(linear_attn_config=dict(_LAYER_KINDS, kda_layers=[1, 2, 3, 4, 5, 6, 7]))),
        "layer 4 must be exactly one of",
    ),
    "dense-mla-layer": (
        dict(text=dict(dsa_config=dict(_DSA_CONFIG, dsa_layers=[4]))),
        "MLA layers \\[8\\] are not in dsa_config.dsa_layers",
    ),
    "no-latent-moe": (
        dict(text=dict(routed_expert_hidden_size=None)),
        "routed_expert_hidden_size",
    ),
    "nvfp4-experts": (dict(quant_format="nvfp4"), "MXFP4"),
    "no-dsa-sparse": (
        dict(
            config_kwargs=dict(sparse=False),
            sparse=RocketSparseAttentionConfig(),
            kv=dict(enable_block_reuse=False),
        ),
        re.escape("algorithm 'dsa' (got 'rocket')"),
    ),
    # Every target runs FP8 index keys and FP8 latents (checked before the layout and the GPU).
    "fp4-index-keys": (dict(config_kwargs=dict(indexer_k_dtype="fp4")), "indexer_k_dtype 'fp4'"),
    "flashinfer-attention": (dict(attn_backend="FLASHINFER"), "attn_backend 'FLASHINFER'"),
    "nvfp4-kv": (dict(kv_dtype="nvfp4"), _FP8_KV_ONLY),
    "bfloat16-kda-state": (
        dict(kv=dict(mamba_ssm_cache_dtype="bfloat16")),
        "KDA recurrent state in FP32",
    ),
    "dspark": (dict(config_kwargs=dict(spec_config=_DSPARK)), "DSPARK"),
    "dflash1-drafter": (
        dict(
            config_kwargs=dict(
                spec_config=_dflash2(
                    architecture="DFlashDraftModel", conv_kernel_size=0, selector_rank=0
                )
            )
        ),
        "spec_dec_variant 1",
    ),
    # A gate trip turns drafting off mid-flight and changes the KDA replay width.
    "speculation-gate": (
        dict(config_kwargs=dict(spec_config=_dflash2(changes=dict(acceptance_rate_threshold=0.3)))),
        "speculation gate",
    ),
    "pp2": (
        dict(config_kwargs=dict(mapping=dict(world_size=16, tp_size=8, pp_size=2))),
        "pipeline",
    ),
    "tep64": (
        dict(config_kwargs=dict(mapping=dict(world_size=64, tp_size=64))),
        "attention TP 64",
    ),
    "dep1": (dict(config_kwargs=dict(mapping=_dep(1))), "attention DP over 1"),
    "helix8-block-reuse": (
        dict(
            config_kwargs=dict(mapping=_helix(8), spec_config=_DFLASH2),
            kv=dict(enable_block_reuse=True),
        ),
        "Helix requires block reuse disabled",
    ),
    "moe-cutlass": (
        dict(moe_config=MoeConfig(backend="CUTLASS")),
        "TRTLLM MoE backend",
    ),
    "softmax-router": (dict(text=dict(moe_router_activation_func="softmax")), "sigmoid"),
    "dep16-without-mpi": (
        dict(config_kwargs=dict(mapping=_DEP16), patch=dict(mpi_disabled=lambda: True)),
        "MPI session",
    ),
    # A static expert placement runs on the sm_103 DEP16 perf target only.
    "load-balancer-tep8": (
        dict(moe_config=MoeConfig(load_balancer=MoeLoadBalancerConfig(num_slots=896))),
        _LOAD_BALANCER,
    ),
    "kv-manager-v1": (dict(kv=dict(use_kv_cache_manager_v2=False)), "V2"),
    "beam2": (dict(max_beam_width=2), "beam width"),
    "disagg-ucx": (
        dict(cache_transceiver_config=CacheTransceiverConfig(backend="UCX")),
        "Python NIXL",
    ),
    "compute-capability-sm100": (dict(capability=(10, 0)), "sm_100 is not validated yet"),
    # The target loads BF16 attention projections: one quantized per layer gets a NoTarget instead
    # of a load-time failure.
    "quantized-attention": (
        dict(
            config_kwargs=dict(
                quant_config=QuantConfig(
                    quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=QuantAlgo.FP8
                ),
                quant_config_dict={
                    "language_model.model.layers.7.self_attn.kv_b_proj": QuantConfig(
                        quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
                    )
                },
            )
        ),
        (
            "model.layers.7.self_attn.kv_b_proj is quantized as FP8_BLOCK_SCALES",
            re.escape("ignore 're:.*self_attn.*'"),
        ),
    ),
    # The routed experts load as MXFP4 only.
    "unquantized-routed-experts": (
        dict(
            config_kwargs=dict(
                quant_config=_checkpoint_quant_config(
                    ignore=[*_CHECKPOINT_IGNORE, "re:.*block_sparse_moe\\.experts.*"]
                )
            )
        ),
        re.escape("language_model.model.layers.0.block_sparse_moe.experts are excluded"),
    ),
}


@pytest.mark.parametrize("case", list(_NEAR_MISSES.values()), ids=list(_NEAR_MISSES))
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_k31_near_misses_do_not_match(monkeypatch, case, mode):
    changes, missed = case
    config, args = _deployment(monkeypatch, changes)
    with pytest.raises(ValueError) as refusal:
        modeling_v2_resolve(config, _llm_args(mode, **args))
    for pattern in missed if isinstance(missed, tuple) else (missed,):
        assert re.search(pattern, str(refusal.value)), (pattern, str(refusal.value))


_HELIX8_NO_REUSE = dict(config_kwargs=dict(mapping=_helix(8)), kv=dict(enable_block_reuse=False))
_DISAGG = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

# Settings a user sets (the arguments, an environment variable, the layout) that the MXFP4 targets
# refuse at load, refused by routing instead: (``_deployment`` changes, what the NoTarget names).
_LOAD_TIME_REFUSALS = {
    # An AUTO the arguments set stays AUTO, which the engine resolves to CUTLASS for Kimi K3.1.
    "moe-backend-auto-named": (
        dict(moe_config=MoeConfig(backend="AUTO")),
        "TRTLLM MoE backend, got AUTO",
    ),
    "drafter-attention-backend": (
        dict(config_kwargs=dict(spec_config=_dflash2(changes=dict(attention_backend="TRTLLM")))),
        "speculative_config.attention_backend 'TRTLLM'",
    ),
    "rejection-sampling-standard-named": (
        dict(
            config_kwargs=dict(
                spec_config=_dflash2(
                    changes=dict(use_rejection_sampling=True, rejection_sample_method="standard")
                )
            )
        ),
        "rejection_sample_method 'standard'",
    ),
    "k8": (
        dict(config_kwargs=dict(spec_config=_dflash2(8))),
        "max_draft_len 8: the Kimi K3.1 targets verify at most 7 drafts",
    ),
    "kda-heads-over-tep8": (
        dict(text=dict(linear_attn_config=dict(_LAYER_KINDS, num_heads=36))),
        "36 KDA heads",
    ),
    "drafter-heads-over-helix8": (
        dict(
            _HELIX8_NO_REUSE,
            config_kwargs=dict(
                mapping=_helix(8), spec_config=_dflash2(drafter=dict(num_attention_heads=36))
            ),
        ),
        "its 36 attention heads over the 8 Helix ranks",
    ),
    "context-server-without-transceiver": (
        dict(env=dict(TRTLLM_DISAGG_ROLE="context"), config_kwargs=dict(spec_config=_DFLASH2)),
        "without cache_transceiver_config",
    ),
    "unified-draft-cache-swa-scratch-reuse": (
        dict(
            config_kwargs=dict(spec_config=_DFLASH2),
            cache_transceiver_config=_DISAGG,
            kv=dict(enable_swa_scratch_reuse=True),
        ),
        "kv_cache_config.enable_swa_scratch_reuse",
    ),
    "unified-draft-cache-pool-ratio": (
        dict(
            config_kwargs=dict(mapping=_helix(8), spec_config=_DFLASH2),
            kv=dict(enable_block_reuse=False, pool_ratio=[0.5, 0.5]),
        ),
        "kv_cache_config.pool_ratio",
    ),
    "placement-max-num-tokens": (
        dict(_DEP16_DFLASH2, moe_config=MoeConfig(load_balancer=_STATIC_PLACEMENT)),
        "lower max_num_tokens to 256 or use identity placement",
    ),
    "placement-without-a-layer": (
        dict(
            _DEP16_DFLASH2,
            moe_config=MoeConfig(
                load_balancer=MoeLoadBalancerConfig(
                    num_slots=896,
                    initial_global_assignments={layer: list(range(896)) for layer in range(1, 8)},
                )
            ),
        ),
        "no entry for routed-MoE layer 0",
    ),
    "placement-not-a-permutation": (
        dict(
            _DEP16_DFLASH2,
            moe_config=MoeConfig(
                load_balancer=MoeLoadBalancerConfig(
                    num_slots=896,
                    initial_global_assignments={layer: [0] * 896 for layer in range(8)},
                )
            ),
        ),
        "layer 0 is not a permutation of the 896 experts",
    ),
    "dep16-odd-gpus-per-node": (
        dict(config_kwargs=dict(mapping=dict(_DEP16, gpus_per_node=1), spec_config=_DFLASH2)),
        "gpus_per_node 1",
    ),
}


@pytest.mark.parametrize("case", list(_LOAD_TIME_REFUSALS.values()), ids=list(_LOAD_TIME_REFUSALS))
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_k31_refuses_what_the_mxfp4_targets_refuse_at_load(monkeypatch, case, mode):
    changes, missed = case
    config, args = _deployment(monkeypatch, changes)
    with pytest.raises(ValueError, match=re.escape(missed)):
        modeling_v2_resolve(config, _llm_args(mode, **args))


def _mapping(tp, adp=False, moe=None, pp=1, cp=1):
    """A Mapping stand-in: the routing reads only these fields."""
    moe_tp, moe_ep = moe if moe is not None else (None, None)
    return types.SimpleNamespace(
        tp_size=tp,
        pp_size=pp,
        cp_size=cp,
        has_cp_helix=lambda: False,
        enable_attention_dp=adp,
        moe_tp_ep_user_specified=moe is not None,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )


@pytest.mark.parametrize(
    "mapping, missed, widened",
    [
        (_mapping(8, moe=(2, 2)), "must equal tp_size", False),
        (_mapping(16, adp=True, moe=(1, 16)), None, False),
        (_mapping(8, cp=2), "Helix requires TP1", False),
        (_mapping(3, moe=(1, 3)), "must divide the 896", True),
    ],
    ids=["moe-split-product", "dep16-ep16", "cp2", "moe-ep-3"],
)
def test_k31_mapping_rules(monkeypatch, mapping, missed, widened):
    """MoE split and divisibility rules no Mapping built from LLM arguments reaches. Every admitted
    attention TP size divides the 96 heads and the 896 experts, so ``widened`` admits more sizes to
    reach those rules themselves."""
    if widened:
        monkeypatch.setattr(routing, "_ATTENTION_TP_SIZES", (1, 2, 3, 4, 8, 16, 32, 64))
    reason = routing._mapping_mismatch(mapping)
    if missed is None:
        assert reason is None
    else:
        assert reason is not None and missed in reason


@pytest.mark.parametrize(
    "setting, options",
    [
        ("enable_return_routed_experts", {"enable_return_routed_experts": True}),
        ("TRTLLM_ENABLE_PDL", {"env_overrides": {"TRTLLM_ENABLE_PDL": "0"}}),
        (
            "TRTLLM_DISABLE_FUSED_DSA_METADATA",
            {"env_overrides": {"TRTLLM_DISABLE_FUSED_DSA_METADATA": "1"}},
        ),
    ],
)
def test_routing_rejects_unsupported_execution_settings(setting, options):
    config = _model_config(_k31_config(), **_TEP8)
    with pytest.raises(ValueError, match=setting):
        modeling_v2_resolve(config, _llm_args(**options))


@pytest.mark.parametrize("reject", [False, True])
def test_dflash2_watermark_rejected(reject):
    spec = _dflash2(fields={"use_rejection_sampling": reject})
    config = _model_config(_k31_config(), spec_config=spec, **_TEP8)
    with pytest.raises(ValueError, match="DFlash2 verification does not support watermark_config"):
        modeling_v2_resolve(config, _llm_args(watermark_config={"key": 1}))
    assert modeling_v2_resolve(config, _llm_args()) is not None


@pytest.mark.parametrize("kv", [{}, {"enable_block_reuse": False}])
def test_k31_helix_routes_with_managed_cache(kv):
    # CP 8 with graphs, overlap and DFlash2: the padded batch sizes a helix generation server at
    # batch 16 captures. Routing reads neither the overlap nor the graph settings.
    graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True)
    args = _llm_args(
        mode="require",
        cuda_graph_config=graph_config,
        disable_overlap_scheduler=False,
        kv=kv,
    )
    config = _model_config(_k31_config(), spec_config=_DFLASH2, **_helix(8))
    assert _qualified(modeling_v2_resolve(config, args)) == _TARGET


# --- the ModelOpt NVFP4 export on sm_107: the specialized targets ---

_SPECIALIZED = {
    "dep16": ("kimi_k31_pro_nvfp4__sm_107__dep16", "ModelingV2KimiK31ProNvfp4Sm107Dep16"),
    "helix8": ("kimi_k31_pro_nvfp4__sm_107__helix8", "ModelingV2KimiK31ProNvfp4Sm107Helix8"),
    "tep8": ("kimi_k31_pro_nvfp4__sm_107__tep8", "ModelingV2KimiK31ProNvfp4Sm107Tep8"),
}
_DEFAULT = ("kimi_k31_pro_nvfp4__sm_10x__default", "ModelingV2KimiK31ProNvfp4Sm10xDefault")
# The real role reader; the ``rubin`` fixtures stand a fixed role in for it.
_SERVER_ROLE = routing.server_role
# The variable the runtime publishes the server role in.
_ROLE_ENV = "TRTLLM_DISAGG_ROLE"

_HELIX8 = dict(
    world_size=8, tp_size=1, cp_size=8, cp_config={"cp_type": "HELIX"}, moe_tp_size=2, moe_ep_size=4
)
# Helix CP 8 with the experts EP 8: the NVFP4 default target's helix8 split.
_HELIX8_EP8 = dict(_HELIX8, moe_tp_size=1, moe_ep_size=8)
_RUBIN_LAYOUTS = {"dep16": dict(_DEP16, moe_ep_size=16), "helix8": _HELIX8, "tep8": _TEP8}

# 8 layers, numbered as linear_attn_config numbers them (from 1): MLA (DSA) on 4 and 8, KDA on
# the rest; layer 0's MLP is dense, the others are the latent MoE. The NVFP4 default target reads
# the layer layout and the indexer's rotary width.
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
    latent_moe_use_norm=True,
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    attn_res_block_size=12,
    num_shared_experts=2,
    first_k_dense_replace=1,
    linear_attn_config=dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7]),
    dsa_config=dict(
        dsa_layers=[4, 8],
        index_topk=2048,
        index_num_heads=32,
        index_head_dim=128,
        index_rope_dim=0,
        index_k_norm_type="layernorm",
    ),
)


def _modelopt_config(
    tmp_path, mapping_kwargs, spec_config=None, shared=True, attn_backend="TRTLLM"
) -> ModelConfig:
    """The export's config, its recipes read from hf_quant_config.json as the loader reads them."""
    layers = quantized_layers(shared=shared)
    path = tmp_path / "hf_quant_config.json"
    path.write_text(json.dumps(hf_quant_config(layers)))
    quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
        str(path), str(tmp_path), "TRTLLM"
    )
    pretrained = PretrainedConfig(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=PretrainedConfig(**_TEXT),
        quantization_config=inline_quantization_config(layers),
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


def _export_args(
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
    moe_config fields; ``backend`` None leaves moe_config.backend unset."""
    sparse = {} if indexer_k_dtype is None else dict(indexer_k_dtype=indexer_k_dtype)
    moe = dict(moe or {}) if backend is None else dict(moe or {}, backend=backend)
    cache = dict(kv_cache or {})
    if ssm_dtype is not None:
        cache["mamba_ssm_cache_dtype"] = ssm_dtype
    if disaggregated:
        kwargs.setdefault(
            "cache_transceiver_config",
            CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON"),
        )
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=KvCacheConfig(dtype=kv_dtype, **cache),
        sparse_attention_config=DeepSeekSparseAttentionConfig(**sparse),
        moe_config=MoeConfig(**moe),
        **kwargs,
    )


@pytest.fixture
def rubin(monkeypatch):
    """Route as on Rubin, with stand-ins for the four NVFP4 targets (importing one needs its GPU
    kernels), on a generation server."""
    on_gpu(monkeypatch, (10, 7))
    for directory, name in (*_SPECIALIZED.values(), _DEFAULT):
        module = types.ModuleType(f"{_FAMILY}.{directory}.modeling")
        setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(routing, "server_role", lambda: "generation")


def _as_role(monkeypatch, role):
    monkeypatch.setattr(routing, "server_role", lambda: role)


def _resolves(config, **args) -> str:
    """The target class name both modes resolve to (they must agree)."""
    names = {
        modeling_v2_resolve(config, _export_args(mode, **args)).__name__
        for mode in ("auto", "require")
    }
    assert len(names) == 1, names
    return names.pop()


def _refused(config, named, **args) -> str:
    """Both modes report the setting that ruled out the deployment."""
    for mode in ("auto", "require"):
        with pytest.raises(ValueError, match=re.escape(named)) as refusal:
            modeling_v2_resolve(config, _export_args(mode, **args))
    return str(refusal.value)


# Each specialized server: (layout, role, the MoE backend it runs). An aggregated server takes
# the generation role.
_SERVERS = {
    "dep16-context": ("dep16", "context", "CUTEDSL"),
    "dep16-generation": ("dep16", "generation", "MEGAMOE_CUTEDSL"),
    "helix8-generation": ("helix8", "generation", "CUTEDSL_FC12"),
    "tep8-context": ("tep8", "context", "CUTEDSL"),
    "tep8-generation": ("tep8", "generation", "CUTEDSL_FC12"),
}


def _export_deployment(monkeypatch, tmp_path, server, changes=None) -> tuple:
    """(model config, MoE backend) of ``server`` in its role, drafting 7 tokens with DFlash2, with
    ``changes``: ``role``, ``draft_len``, ``spec`` ("dspark" or None), ``spec_fields``,
    ``spec_changes`` and ``drafter`` (``_dflash2``'s ``fields``, ``changes`` and ``drafter``),
    ``mapping``, ``dsa_layers`` (of the text config), and ``_modelopt_config`` arguments."""
    layout, role, backend = _SERVERS[server]
    changes = dict(changes or {})
    _as_role(monkeypatch, changes.pop("role", role))
    draft_len = changes.pop("draft_len", 7)
    spec = changes.pop("spec", "dflash2")
    spec_config = None
    if spec == "dflash2":
        spec_config = _dflash2(
            draft_len,
            fields=changes.pop("spec_fields", None),
            changes=changes.pop("spec_changes", None),
            drafter=changes.pop("drafter", None),
        )
    elif spec == "dspark":
        spec_config = DSparkDecodingConfig(max_draft_len=draft_len, speculative_model=_DUMMY_MODEL)
    dsa_layers = changes.pop("dsa_layers", None)
    mapping = changes.pop("mapping", _RUBIN_LAYOUTS[layout])
    config = _modelopt_config(tmp_path, mapping, spec_config, **changes)
    if dsa_layers is not None:
        text = config.pretrained_config.text_config
        text.dsa_config = dict(text.dsa_config, dsa_layers=dsa_layers)
    return config, backend


@pytest.mark.parametrize("server", list(_SERVERS))
def test_every_server_takes_the_recipe(rubin, monkeypatch, tmp_path, server):
    """The export with an FP8 KV cache, FP8 index keys, an FP32 KDA state and DFlash2 at 7 drafts
    lands on the server's specialized target; a context server runs the one of the generation
    server it feeds."""
    config, backend = _export_deployment(monkeypatch, tmp_path, server)
    assert engine_kv_cache_quant_algo(config, _export_args(backend=backend)) is QuantAlgo.FP8
    name = _resolves(config, backend=backend, ssm_dtype="float32")
    assert name == _SPECIALIZED[_SERVERS[server][0]][1]


@pytest.mark.parametrize("draft_len", [2, 8], ids=lambda k: f"k{k}")
def test_every_server_refuses_other_draft_lengths(rubin, monkeypatch, tmp_path, draft_len):
    """The reason names the draft lengths the server takes and, above 7, why a longer draft does
    not fit: the KDA verify kernels and one indexer-logits atom hold at most 7 drafts. Every cell
    holds the same draft lengths, so one server stands for all."""
    server = "tep8-generation"
    config, backend = _export_deployment(monkeypatch, tmp_path, server, dict(draft_len=draft_len))
    layout, role, _ = _SERVERS[server]
    named = f"max_draft_len {draft_len}: a {layout} {role} server runs one of (3, 4, 5, 6, 7)"
    reason = _refused(config, named, backend=backend)
    assert ("KDA verify kernels" in reason and "indexer-logits atom" in reason) == (draft_len > 7)


@pytest.mark.parametrize(
    "setting",
    [
        "use_cute_dsl_topk",
        "index_rope_dim",
        "linear_attn_config",
        "use_cute_dsl_bf16_gemm",
        "TRTLLM_LOW_M_GEMM_BACKEND",
    ],
)
def test_specialized_target_validates_projection_and_index_settings(
    rubin, monkeypatch, tmp_path, setting
):
    config, backend = _export_deployment(monkeypatch, tmp_path, "tep8-generation")
    text = config.pretrained_config.text_config
    args = {}
    if setting == "use_cute_dsl_topk":
        config.sparse_attention_config.use_cute_dsl_topk = True
    elif setting == "index_rope_dim":
        text.dsa_config = dict(text.dsa_config, index_rope_dim=32)
    elif setting == "linear_attn_config":
        text.linear_attn_config = None
    elif setting == "TRTLLM_LOW_M_GEMM_BACKEND":
        monkeypatch.setenv(setting, "cutlass")
    else:
        args[setting] = True
    _refused(config, setting, backend=backend, **args)


def test_an_aggregated_server_takes_the_generation_role(rubin, monkeypatch, tmp_path):
    """Without a cache transceiver a server is aggregated, and runs the generation constants."""
    server = "tep8-generation"
    config, backend = _export_deployment(monkeypatch, tmp_path, server)
    monkeypatch.setattr(routing, "server_role", _SERVER_ROLE)
    monkeypatch.delenv(_ROLE_ENV, raising=False)
    name = _resolves(config, backend=backend, disaggregated=False)
    assert name == _SPECIALIZED[_SERVERS[server][0]][1]


# Refusals: (server, ``_export_deployment`` changes, argument changes, what the reason names).
_SPECIALIZED_REFUSALS = {
    "dspark": ("tep8-generation", dict(spec="dspark"), {}, "DSpark speculative decoding"),
    "flashinfer-attention": (
        "tep8-generation",
        dict(attn_backend="FLASHINFER"),
        {},
        "attention backend FLASHINFER",
    ),
    "bf16-shared-experts": (
        "tep8-generation",
        dict(shared=False),
        {},
        "module(s) the NVFP4 export quantizes are BF16 here",
    ),
    "helix8-context": (
        "helix8-generation",
        dict(role="context"),
        {},
        "helix8 serves generation servers only, not a context server",
    ),
    "tep8-generation-cutedsl": (
        "tep8-generation",
        {},
        dict(backend="CUTEDSL"),
        "a tep8 generation server runs CUTEDSL_FC12",
    ),
    "draft-len-schedule": (
        "tep8-generation",
        dict(spec_changes=dict(draft_len_schedule={4: 7, 8: 3})),
        {},
        "no draft_len_schedule",
    ),
    "torch-compile": (
        "tep8-generation",
        {},
        dict(torch_compile_config=TorchCompileConfig()),
        "no torch_compile_config",
    ),
    "kv-cache-manager-v1": (
        "tep8-generation",
        {},
        dict(kv_cache=dict(use_kv_cache_manager_v2=False)),
        "keep the index keys and the drafter history in KV-V2",
    ),
    "ucx-transceiver": (
        "tep8-generation",
        {},
        dict(cache_transceiver_config=CacheTransceiverConfig(backend="UCX")),
        "Python NIXL transceiver only",
    ),
    "helix8-aggregated": (
        "helix8-generation",
        {},
        dict(disaggregated=False),
        "helix8 serves a disaggregated generation server",
    ),
}


@pytest.mark.parametrize(
    "case", list(_SPECIALIZED_REFUSALS.values()), ids=list(_SPECIALIZED_REFUSALS)
)
def test_specialized_refusals(rubin, monkeypatch, tmp_path, case):
    server, changes, args_changes, named = case
    config, backend = _export_deployment(monkeypatch, tmp_path, server, changes)
    _refused(config, named, **{"backend": backend, **args_changes})


# Settings a user sets that the NVFP4 targets refuse at load, refused by routing instead, and a
# config whose layer layout the export's module list cannot be read off: (GPU, server,
# ``_export_deployment`` changes, argument changes, environment, text config changes, what the
# reason names). On sm_107 the specialized targets' reason comes first.
_NVFP4_LOAD_TIME_REFUSALS = {
    "sm107-py-mamba": (
        SM107,
        "tep8-generation",
        {},
        {},
        dict(TRTLLM_USE_PY_MAMBA="1"),
        {},
        "TRTLLM_USE_PY_MAMBA=1",
    ),
    "sm107-context-server-without-transceiver": (
        SM107,
        "tep8-context",
        {},
        dict(disaggregated=False),
        {},
        {},
        "without cache_transceiver_config",
    ),
    "sm107-swa-scratch-reuse": (
        SM107,
        "tep8-generation",
        {},
        dict(kv_cache=dict(enable_swa_scratch_reuse=True)),
        {},
        {},
        "kv_cache_config.enable_swa_scratch_reuse",
    ),
    "sm103-drafter-attention-backend": (
        SM103,
        "tep8-aggregated",
        dict(spec_changes=dict(attention_backend="FA4")),
        {},
        {},
        {},
        "speculative_config.attention_backend 'FA4'",
    ),
    "sm103-rejection-sampling-standard-named": (
        SM103,
        "tep8-aggregated",
        dict(spec_changes=dict(use_rejection_sampling=True, rejection_sample_method="standard")),
        {},
        {},
        {},
        "rejection_sample_method 'standard'",
    ),
    "sm103-kda-heads-over-tep8": (
        SM103,
        "tep8-aggregated",
        {},
        {},
        {},
        dict(linear_attn_config=dict(_TEXT["linear_attn_config"], num_heads=36)),
        "36 KDA heads",
    ),
    "sm103-drafter-heads-over-helix8": (
        SM103,
        "helix8-generation",
        dict(mapping=_HELIX8_EP8, drafter=dict(num_attention_heads=36)),
        {},
        {},
        {},
        "its 36 attention heads over the 8 Helix ranks",
    ),
    "sm103-context-server-without-transceiver": (
        SM103,
        "tep8-context",
        {},
        dict(disaggregated=False, disable_mm_encoder=True),
        {},
        {},
        "without cache_transceiver_config",
    ),
    "sm103-no-linear-attn-config": (
        SM103,
        "tep8-aggregated",
        {},
        {},
        {},
        dict(linear_attn_config=None),
        "the text config has no linear_attn_config",
    ),
}


@pytest.mark.parametrize(
    "case", list(_NVFP4_LOAD_TIME_REFUSALS.values()), ids=list(_NVFP4_LOAD_TIME_REFUSALS)
)
def test_routing_refuses_what_the_nvfp4_targets_refuse_at_load(rubin, monkeypatch, tmp_path, case):
    capability, server, changes, args_changes, env, text_changes, named = case
    on_gpu(monkeypatch, capability)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    layout, kind = server.split("-")
    role = "generation" if kind == "aggregated" else kind
    config, backend = _export_deployment(monkeypatch, tmp_path, f"{layout}-{role}", changes)
    for name, value in text_changes.items():
        setattr(config.pretrained_config.text_config, name, value)
    # The NVFP4 default target runs CUTEDSL; a specialized server its own backend.
    backend = "CUTEDSL" if capability == SM103 else backend
    args = {"backend": backend, "disaggregated": kind != "aggregated", **args_changes}
    _refused(config, named, **args)


# --- the ModelOpt NVFP4 export: the NVFP4 default target ---

_ON_DEFAULT = "not validated on the NVFP4 default target"
# The layouts the NVFP4 default target runs, as a refusal of another one lists them.
_DEFAULT_TARGET_LAYOUTS = (
    "which runs tep8 (TP 8, MoE EP 8), dep16 (attention DP 16, MoE EP 16) and helix8 (Helix CP 8 "
    "over TP 1, MoE EP 8)"
)
# Cases of the settings the NVFP4 default target reads: (GPU, server, ``_export_deployment``
# changes, argument changes, expected). ``server`` is "<layout>-<kind>"; an aggregated server is the
# generation server's deployment without a cache transceiver. The arguments name the CUTEDSL MoE
# backend and the drafter drafts 4 tokens unless the changes say otherwise. ``expected`` is the
# layout of the target the deployment lands on ("default" or a specialized layout), else the texts
# its refusal holds. On sm_103 a refusal past the GPU check is the default target's; on sm_107 the
# specialized targets' reason comes first.
_NVFP4_DEFAULT_CASES = {
    # sm_103: validated cells.
    "sm103-tep8-aggregated": (SM103, "tep8-aggregated", {}, {}, "default"),
    "sm103-tep8-context-without-the-tower": (
        SM103,
        "tep8-context",
        {},
        dict(disable_mm_encoder=True),
        "default",
    ),
    "sm103-dep16-generation": (SM103, "dep16-generation", {}, {}, "default"),
    "sm103-helix8-generation": (
        SM103,
        "helix8-generation",
        dict(mapping=_HELIX8_EP8),
        {},
        "default",
    ),
    # sm_103: the refusals, each of one setting outside the validated cells.
    "sm100": ((10, 0), "tep8-aggregated", {}, {}, ("sm_100 is not validated yet",)),
    "sm103-tep8-context-with-the-tower": (
        SM103,
        "tep8-context",
        {},
        {},
        (
            "the NVFP4 default target: a context server with the vision tower: image requests "
            f"through disaggregated serving are {_ON_DEFAULT}",
        ),
    ),
    "sm103-helix8-moe-tp2": (
        SM103,
        "helix8-generation",
        {},
        {},
        (
            "the NVFP4 default target: layout world_size=8 tp=1 pp=1 cp=8 helix=True "
            f"attention_dp=False moe_tp=2 moe_ep=4: {_ON_DEFAULT}, {_DEFAULT_TARGET_LAYOUTS}",
        ),
    ),
    "sm103-moe-trtllm": (
        SM103,
        "tep8-aggregated",
        {},
        dict(backend="TRTLLM"),
        (
            f"the NVFP4 default target: moe_config.backend TRTLLM: {_ON_DEFAULT}, which runs CUTEDSL",
        ),
    ),
    "sm103-nvfp4-kv-cache": (
        SM103,
        "tep8-aggregated",
        {},
        dict(kv_dtype="nvfp4"),
        (
            "the NVFP4 default target: kv_cache_config.dtype 'nvfp4' (KV cache NVFP4): "
            f"{_ON_DEFAULT}",
        ),
    ),
    "sm103-k8": (
        SM103,
        "tep8-aggregated",
        dict(draft_len=8),
        {},
        (
            "the NVFP4 default target: max_draft_len 8: the NVFP4 default target runs one of "
            "(1, 2, 3, 4, 5, 6, 7); a draft of k fills 2k + 1 rows of the KDA verify kernels",
        ),
    ),
    # sm_107: the aggregated servers the specialized targets refuse for their MoE backend.
    "sm107-tep8-aggregated": (SM107, "tep8-aggregated", {}, {}, "default"),
    # The specialized dep16 target keeps the aggregated server of its own backend.
    "sm107-dep16-aggregated-megamoe-cutedsl": (
        SM107,
        "dep16-aggregated",
        {},
        dict(backend="MEGAMOE_CUTEDSL"),
        "dep16",
    ),
    # sm_107: no disaggregated server on the default target.
    "sm107-tep8-generation": (
        SM107,
        "tep8-generation",
        {},
        {},
        (
            "moe_config.backend CUTEDSL: a tep8 generation server runs CUTEDSL_FC12; the NVFP4 "
            f"default target: a tep8 generation server on sm_107: {_ON_DEFAULT} there",
        ),
    ),
}


@pytest.mark.parametrize(
    "case", list(_NVFP4_DEFAULT_CASES.values()), ids=list(_NVFP4_DEFAULT_CASES)
)
def test_the_nvfp4_default_target_takes_its_validated_cells_only(
    rubin, monkeypatch, tmp_path, case
):
    """The export lands on the NVFP4 default target on each validated cell the specialized targets
    do not take, and a deployment one setting away from one is refused naming that setting."""
    capability, server, changes, args_changes, expected = case
    on_gpu(monkeypatch, capability)
    layout, kind = server.split("-")
    config, _ = _export_deployment(
        monkeypatch,
        tmp_path,
        f"{layout}-{'generation' if kind == 'aggregated' else kind}",
        {"draft_len": 4, **changes},
    )
    args = {"backend": "CUTEDSL", "disaggregated": kind != "aggregated", **args_changes}
    if isinstance(expected, str):
        target = _DEFAULT if expected == "default" else _SPECIALIZED[expected]
        assert _resolves(config, **args) == target[1]
        return
    reason = _refused(config, expected[0], **args)
    for named in expected[1:]:
        assert named in reason, (named, reason)


# --- the server role ---


@pytest.mark.parametrize(
    "environ, role",
    [("context", "context"), (None, "generation")],
    ids=["context", "aggregated"],
)
def test_server_role_is_the_role_the_runtime_published(monkeypatch, environ, role):
    """``trtllm-serve --server_role`` publishes the role to every rank; an aggregated server, which
    publishes none, takes the generation role."""
    if environ is None:
        monkeypatch.delenv(_ROLE_ENV, raising=False)
    else:
        monkeypatch.setenv(_ROLE_ENV, environ)
    assert _SERVER_ROLE() == role


def test_the_published_role_reaches_routing(rubin, monkeypatch, tmp_path):
    """The server's role picks its row of the table: helix8 serves generation only."""
    monkeypatch.setattr(routing, "server_role", _SERVER_ROLE)
    monkeypatch.delenv(_ROLE_ENV, raising=False)
    config = _modelopt_config(tmp_path, _HELIX8, _dflash2(5))
    monkeypatch.setenv(_ROLE_ENV, "context")
    _refused(config, "not a context server", backend="CUTEDSL_FC12")
    monkeypatch.delenv(_ROLE_ENV)
    assert _resolves(config, backend="CUTEDSL_FC12") == _SPECIALIZED["helix8"][1]


# --- both loads of a server YAML ---
# The engine routes the config loaded before the target's model defaults apply (the arguments as
# written), then the config the model is built from.


#: What decides the target in a server YAML of the recipe, as ``trtllm-serve --config`` reads it,
#: but its speculative_config (``_SERVER_DRAFT_LEN``): validating a DFlash speculative_config
#: against a local checkpoint imports the engine's drafter ops, which a CPU host lacks.
_SERVER_DRAFT_LEN = {"tep8-generation": 7, "helix8-generation": 5, "dep16-context": 5}
_SERVER_YAML = {
    "tep8-generation": """\
        tensor_parallel_size: 8
        moe_expert_parallel_size: 8
        enable_attention_dp: false
        max_batch_size: 1
        modeling_v2: require
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
def test_a_server_yaml_routes_both_loads(rubin, monkeypatch, tmp_path, server):
    layout, role = server.split("-")
    _as_role(monkeypatch, role)
    checkpoint = checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **yaml.safe_load(textwrap.dedent(_SERVER_YAML[server])))
    spec_config = _dflash2(_SERVER_DRAFT_LEN[server])
    for config in _loaded_configs(checkpoint, args, spec_config):
        assert config.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert engine_kv_cache_quant_algo(config, args) is QuantAlgo.FP8
        assert modeling_v2_resolve(config, args).__name__ == _SPECIALIZED[layout][1]


# --- the role reaches the target ---
# The runtime publishes the server role to every rank; routing and the target read the same role.

#: A TEP8 context server, in front of the TEP8 generation server above.
_TEP8_CONTEXT_YAML = """\
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    enable_attention_dp: false
    max_batch_size: 32
    max_num_tokens: 16384
    enable_chunked_prefill: true
    modeling_v2: require
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
#: length, the ``--server_role`` given (None: none), the role the target builds, the target: a
#: specialized target's layout or "default"). The aggregated TEP8 server is the generation server
#: without a cache transceiver, with chunked prefill; on the CUTEDSL MoE backend, which the
#: specialized tep8 target refuses there, it runs the NVFP4 default target.
_ROLE_SERVERS = {
    "tep8-context": (_TEP8_CONTEXT_YAML, {}, 7, "context", "context", "tep8"),
    "helix8-generation": (_SERVER_YAML["helix8-generation"], {}, 5, None, "generation", "helix8"),
    "tep8-aggregated-cutedsl": (
        _SERVER_YAML["tep8-generation"],
        dict(
            cache_transceiver_config=None,
            max_num_tokens=2048,
            enable_chunked_prefill=True,
            moe_config=dict(backend="CUTEDSL"),
        ),
        7,
        None,
        "generation",
        "default",
    ),
}


def _model_loader():
    """The engine's model loader module; skips on a CPU host whose stand-ins cannot import it."""
    try:
        from tensorrt_llm._torch.pyexecutor import model_loader
    except Exception as exc:  # noqa: BLE001 - the engine runtime is not importable on this host
        pytest.skip(f"the model loader does not import here: {type(exc).__name__}: {exc}")
    return model_loader


@pytest.mark.parametrize("server", list(_ROLE_SERVERS))
def test_the_role_reaches_the_target_through_the_loader(rubin, monkeypatch, tmp_path, server):
    """``ModelLoader._load_and_validate_config`` on the export directory, with the role published
    as ``trtllm-serve --server_role`` publishes it: the config it returns carries the target of
    that role, and the role reads back as published.

    The KDA state check, which imports the KDA package (``fla``), is stood in; it gets the config
    the loader returns."""
    model_loader = _model_loader()
    # The resolved DFlash budget reads device memory on this simulated GPU.
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda device: types.SimpleNamespace(total_memory=128 << 30),
    )
    state_checked = []
    monkeypatch.setattr(
        model_loader,
        "validate_and_set_mamba_ssm_cache_dtype",
        lambda config, *args: state_checked.append(config),
    )
    text, changes, draft_len, server_role, role, target = _ROLE_SERVERS[server]
    monkeypatch.setattr(routing, "server_role", _SERVER_ROLE)
    monkeypatch.delenv(_ROLE_ENV, raising=False)
    options = yaml.safe_load(textwrap.dedent(text))
    for key, value in changes.items():
        if value is None:
            options.pop(key)
        else:
            options[key] = value
    if server_role is not None:
        # commands/serve.py: the environment and the overrides the workers apply.
        monkeypatch.setenv(_ROLE_ENV, server_role)
        options["env_overrides"] = {_ROLE_ENV: server_role}
    checkpoint = checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **options)
    loader = object.__new__(model_loader.ModelLoader)
    loader.llm_args = args
    loader.mapping = args.parallel_config.to_mapping()
    loader.spec_config = _dflash2(draft_len)
    loader.sparse_attention_config = args.sparse_attention_config
    loader.max_num_tokens = args.max_num_tokens
    loader.max_seq_len = args.max_seq_len
    loader.lora_config = None
    config = loader._load_and_validate_config(checkpoint, HfConfigOnly())
    assert len(state_checked) == 1 and state_checked[0] is config
    expected = _DEFAULT[1] if target == "default" else _SPECIALIZED[target][1]
    assert config.modeling_v2_target.__name__ == expected
    assert config.extra_attrs["kv_cache_dtype"] == "fp8"  # the loader's own keys stay
    assert _SERVER_ROLE() == role


# --- the NVFP4 default target on both loads ---


def test_a_default_server_yaml_routes_both_loads(rubin, monkeypatch, tmp_path):
    """The tep8 generation server's YAML on sm_103 with the CUTEDSL MoE backend lands on the NVFP4
    default target on both loads. The first load has no sparse attention config of its own, so
    routing reads the arguments' index keys: FP4 there refuses both loads alike."""
    on_gpu(monkeypatch, SM103)
    server = "tep8-generation"
    checkpoint = checkpoint_dir(tmp_path)
    options = yaml.safe_load(textwrap.dedent(_SERVER_YAML[server]))
    options["moe_config"]["backend"] = "CUTEDSL"
    spec_config = _dflash2(_SERVER_DRAFT_LEN[server])
    args = TorchLlmArgs(model=checkpoint, **options)
    for config in _loaded_configs(checkpoint, args, spec_config):
        assert modeling_v2_resolve(config, args).__name__ == _DEFAULT[1]
    options["sparse_attention_config"]["indexer_k_dtype"] = "fp4"
    args = TorchLlmArgs(model=checkpoint, **options)
    first, built = _loaded_configs(checkpoint, args, spec_config)
    assert getattr(first.sparse_attention_config, "indexer_k_dtype", "fp8") == "fp8"
    assert built.sparse_attention_config.indexer_k_dtype == "fp4"
    refused = f"the NVFP4 default target: indexer_k_dtype 'fp4': {_ON_DEFAULT}"
    for config in (first, built):
        with pytest.raises(ValueError, match=re.escape(refused)):
            modeling_v2_resolve(config, args)


# --- a disaggregated pair of specialized targets builds one cache layout ---
# The context server's target writes the pages the generation server's target reads: the MLA latent
# KV, the DSA index keys, the KDA state and the DFlash2 drafter's history. Each target binds its cache
# manager's arguments per role (``_CACHE_ARGS``, through ``bind_kv_cache_args``), so the two sides of
# a pair the table above admits are built from two tables that must agree on every argument that
# shapes a page. The targets are imported.

#: (context target, generation target): one pair within a layout, one across layouts.
_PAIRS = (("tep8", "tep8"), ("dep16", "helix8"))
#: The one bound argument that shapes no page: the pools are zeroed once at construction.
_NOT_LAYOUT = frozenset({"zero_gpu_pools"})
#: Recipe and model constants of modeling.py that shape a page on either side.
_PAGE_CONSTANTS = (
    "_INDEXER_K_DTYPE",
    "_STATE_DTYPE",
    "_DRAFT_HISTORY_DTYPE",
    "_K31",
    "_K31_KDA",
    "_K31_DSA",
)
#: Engine defaults that shape a page.
_KV_DEFAULTS = ("tokens_per_block", "dtype", "mamba_ssm_cache_dtype")
#: Shell hooks whose answer must not depend on the side, and the configs they are asked about.
_HOOKS = ("requires_kv_cache_manager_v2", "get_preferred_transceiver_runtime")
_HOOK_CONFIGS = (
    None,
    types.SimpleNamespace(),
    types.SimpleNamespace(dsa_config={}),
    types.SimpleNamespace(text_config=types.SimpleNamespace(dsa_config={})),
)
#: The shells' class attributes the executor reads. Their runtime hooks agree too, apart from the attention backend:
#: helix8 declares the Helix one.
_SHELL_ATTRS = ("mamba_metadata_cls",)


def _specialized_modeling(layout: str):
    return importlib.import_module(f"{_FAMILY}.{_SPECIALIZED[layout][0]}.modeling")


def _shell(module) -> type:
    """The target's shell: its one ``DecoderModelForCausalLM`` class."""
    (shell,) = [
        cls
        for cls in vars(module).values()
        if isinstance(cls, type)
        and cls.__module__ == module.__name__
        and issubclass(cls, module.DecoderModelForCausalLM)
    ]
    return shell


@pytest.mark.parametrize("ctx, gen", _PAIRS, ids=[f"{c}-ctx-to-{g}-gen" for c, g in _PAIRS])
def test_context_and_generation_sides_build_the_same_cache(ctx, gen):
    """The two sides of a disaggregated pair agree on everything that shapes a cache page."""
    c, g = _specialized_modeling(ctx), _specialized_modeling(gen)
    assert "context" in c._ROLES, f"{ctx} serves no context role"
    assert "generation" in g._ROLES, f"{gen} serves no generation role"
    manager = importlib.import_module(f"{_FAMILY}.ops.ssm.rubin_kda_dsa_cache_manager")
    params = inspect.signature(manager.bind_kv_cache_args).parameters.values()
    bound = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
    c_args, g_args = c._CACHE_ARGS["context"], g._CACHE_ARGS["generation"]
    differ = {k: (c_args[k], g_args[k]) for k in bound - _NOT_LAYOUT if c_args[k] != g_args[k]}
    assert not differ, f"cache arguments differ (context, generation): {differ}"
    for name in _PAGE_CONSTANTS:
        assert hasattr(c, name) and hasattr(g, name), f"{name} is not a constant of both targets"
        assert getattr(c, name) == getattr(g, name), (
            f"{name}: {getattr(c, name)} vs {getattr(g, name)}"
        )
    c_shell, g_shell = _shell(c), _shell(g)
    c_kv = c_shell.get_model_defaults(None).get("kv_cache_config", {})
    g_kv = g_shell.get_model_defaults(None).get("kv_cache_config", {})
    for key in _KV_DEFAULTS:
        assert c_kv.get(key) == g_kv.get(key), f"kv_cache_config.{key}"
    for hook in _HOOKS:
        for config in _HOOK_CONFIGS:
            assert getattr(c_shell, hook)(config) == getattr(g_shell, hook)(config), (hook, config)
    for attr in _SHELL_ATTRS:
        assert getattr(c_shell, attr) is getattr(g_shell, attr), attr
    assert replace(c_shell.runtime_hooks, attention_backend_cls=None) == replace(
        g_shell.runtime_hooks, attention_backend_cls=None
    )
