# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H target routing, driven by synthetic configs on CPU.

No checkpoint and no weights: routing reads the config's shape, the mapping, the
compute capability and the LLM API arguments, all of which are stated here. The
basename is family-specific because neither this directory nor GLM's has an
``__init__.py``: two ``test_routing.py`` files would clash in one local session.
"""

from dataclasses import replace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MoeConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM100 = (10, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)

_ULTRA_BF16_TARGET = (
    "models.nemotron_h.nemotron3_ultra_bf16__sm_100__tep8.modeling."
    "ModelingV2Nemotron3UltraBf16Sm100Tep8"
)
_ULTRA_BF16_TP8EP2_TARGET = (
    "models.nemotron_h.nemotron3_ultra_bf16__sm_100__tp8ep2.modeling."
    "ModelingV2Nemotron3UltraBf16Sm100Tp8ep2"
)
# Two GB200 trays of four GPUs: the deployment the BF16 targets were gated on.
_TEP8 = dict(
    world_size=8,
    tp_size=8,
    moe_ep_size=8,
    moe_tp_size=1,
    enable_attention_dp=False,
    gpus_per_node=4,
)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a GB200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def _kv(**overrides) -> KvCacheConfig:
    """The KV cache configuration the gates ran with: block reuse off (the loader turns it
    off for a hybrid model with no snapshot policy before routing), fp32 recurrent state."""
    fields = dict(enable_block_reuse=False, mamba_ssm_cache_dtype="float32")
    fields.update(overrides)
    return KvCacheConfig(**fields)


def _llm_args(mode="auto", kv_cache_config=None, **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it."""
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        modeling_v2=mode,
        kv_cache_config=kv_cache_config or _kv(),
        **kwargs,
    )


def _model_config(pretrained_config, spec_config=None, kv_cache_quant_algo=None, **mapping):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping),
        spec_config=spec_config,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo),
    )


def _ultra_config(**overrides):
    """The shape Nemotron-3 Ultra 550B-A55B NVFP4's own config.json declares."""
    fields = dict(
        architectures=["NemotronHForCausalLM"],
        model_type="nemotron_h",
        hidden_size=8192,
        mamba_num_heads=256,
        mamba_head_dim=64,
        n_groups=8,
        ssm_state_size=128,
        num_attention_heads=64,
        num_key_value_heads=2,
        head_dim=128,
        n_routed_experts=512,
        num_experts_per_tok=22,
        moe_intermediate_size=5120,
        moe_latent_size=2048,
        moe_shared_expert_intermediate_size=10240,
        layers_block_type=["mamba", "moe", "mamba", "attention", "moe"] * 2,
        quantization_config=dict(
            quant_algo="MIXED_PRECISION",
            kv_cache_scheme=dict(dynamic=False, num_bits=8, type="float"),
        ),
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _ultra_bf16_config(**overrides):
    """The shape the BF16 release's config.json declares: the NVFP4 release's, with no
    quantization_config at all."""
    fields = dict(quantization_config=None)
    fields.update(overrides)
    return _ultra_config(**fields)


def _refused(config, missed, **llm_kwargs) -> None:
    """``auto`` falls back to the built-in model; ``require`` names the missed fact."""
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **llm_kwargs))


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_tep8_matches(mode):
    config = _model_config(_ultra_bf16_config(), **_TEP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_TARGET


def test_ultra_bf16_depth_is_not_part_of_the_identity():
    """Every weight the target declares is per layer and the stack is built from the
    config's own block list, so a prefix of the layers runs the same code."""
    config = _model_config(
        _ultra_bf16_config(layers_block_type=["mamba", "moe", "attention"]), **_TEP8
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _ULTRA_BF16_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a Hugging Face native quantization of the same model: any declared quantization is
        # a different weight layout, even when the config carries no `quant_algo`
        (dict(quantization_config=dict(quant_method="fp8")), {}, r"quantization \('fp8', None\)"),
        # the right checkpoint on 4 ranks: 1.1 TB of weights do not fit
        (dict(), dict(_TEP8, world_size=4, tp_size=4, moe_ep_size=4), "world_size=4"),
        # attention data parallel: the replicated non-expert weights would not fit
        (dict(), dict(_TEP8, enable_attention_dp=True), "attention_dp=True"),
        # every expert split over all 8 ranks
        (dict(), dict(_TEP8, moe_ep_size=1, moe_tp_size=8), "moe_ep=1"),
        # pipeline parallelism
        (dict(), dict(_TEP8, tp_size=4, pp_size=2, moe_ep_size=4), "pp=2"),
        # Nemotron-3 Super's geometry
        (dict(hidden_size=4096), {}, r"config shape \(4096,"),
        # a dense-MLP block the target does not implement
        (dict(layers_block_type=["mamba", "mlp", "attention"]), {}, "block kinds"),
        # one 8-GPU node: only two GB200 trays of four were gated
        (dict(), dict(_TEP8, gpus_per_node=8), "inside one node"),
    ],
    ids=[
        "quant-method",
        "four-ranks",
        "attention-dp",
        "moe-tp8",
        "pipeline",
        "super-shape",
        "mlp-block",
        "single-node",
    ],
)
def test_ultra_bf16_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_ultra_bf16_config(**config_kwargs), **(mapping_kwargs or _TEP8))
    _refused(config, missed)


@pytest.mark.parametrize("capability", [(10, 3), (10, 7), (9, 0)])
def test_ultra_bf16_on_another_architecture_does_not_match(capability, monkeypatch):
    """Certification is per architecture: the BF16 targets are gated on sm_100 only."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    config = _model_config(_ultra_bf16_config(), **_TEP8)
    _refused(config, rf"compute capability \({capability[0]}, {capability[1]}\)")


def test_ultra_nvfp4_does_not_route_to_a_bf16_target():
    """The NVFP4 release of the same model on sm_100 falls back."""
    config = _model_config(_ultra_config(), **_TEP8)
    _refused(config, r"MIXED_PRECISION")


def test_ultra_bf16_with_speculative_decoding_does_not_match():
    config = _model_config(_ultra_bf16_config(), spec_config=_NGRAM, **_TEP8)
    _refused(config, "speculative decoding")


@pytest.mark.parametrize(
    "llm_kwargs, missed",
    [
        (dict(kv_cache_config=_kv(dtype="fp8")), "KV cache"),
        (dict(kv_cache_config=_kv(mamba_ssm_cache_dtype="bfloat16")), "recurrent-state"),
        (dict(kv_cache_config=_kv(use_kv_cache_manager_v2=False)), "use_kv_cache_manager_v2"),
        (dict(kv_cache_config=_kv(enable_block_reuse=True)), "enable_block_reuse"),
        (dict(moe_config=MoeConfig(load_balancer={"num_slots": 512})), "EPLB"),
        (dict(prefill_cuda_graph_backend="piecewise"), "piecewise"),
        (dict(max_beam_width=2), "max_beam_width=2"),
    ],
    ids=["fp8-kv", "bf16-state", "kv-manager-v1", "block-reuse", "eplb", "piecewise", "beam"],
)
def test_ultra_bf16_tep8_refuses_engine_choices_it_never_ran(llm_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **_TEP8)
    _refused(config, missed, **llm_kwargs)


@pytest.mark.parametrize("backend", ["CUTEDSL", "DEEPGEMM", "MEGAMOE_DEEPGEMM", "TRITON"])
def test_ultra_bf16_refuses_another_moe_backend(backend):
    config = replace(_model_config(_ultra_bf16_config(), **_TEP8), moe_backend_requested=backend)
    _refused(config, "moe_config.backend")


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("backend", ["AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"])
def test_ultra_bf16_tep8_takes_its_moe_backends(backend, mode):
    """fused_moe by default (AUTO, CUTLASS); the engine's MegaMoE or TRTLLM-Gen BF16 MoE layer
    when asked for by name. The target, not routing, checks that the leaf serves the layer."""
    config = replace(_model_config(_ultra_bf16_config(), **_TEP8), moe_backend_requested=backend)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_TARGET


def test_ultra_bf16_refuses_an_fp16_engine():
    config = _model_config(_ultra_bf16_config(torch_dtype=torch.float16), **_TEP8)
    _refused(config, "torch.float16")


_TP8EP2 = dict(_TEP8, moe_ep_size=2, moe_tp_size=4)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_tp8ep2_matches(mode):
    """Two expert windows of four tensor-parallel ranks select the tp8ep2 target."""
    config = _model_config(_ultra_bf16_config(), **_TP8EP2)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_TP8EP2_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        # attention data parallel: the replicated non-expert weights would not fit
        (dict(_TP8EP2, enable_attention_dp=True), "attention_dp=True"),
        # four windows of two: a layout neither target was built for
        (dict(_TP8EP2, moe_ep_size=4, moe_tp_size=2), "moe_ep=4"),
        # one 8-GPU node
        (dict(_TP8EP2, gpus_per_node=8), "inside one node"),
    ],
    ids=["attention-dp", "moe-ep4-tp2", "single-node"],
)
def test_ultra_bf16_tp8ep2_near_misses_do_not_match(mapping_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **mapping_kwargs)
    _refused(config, missed)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_tp8ep2_matches_over_an_fp8_kv_cache(mode):
    """tp8ep2 binds an fp8-e4m3 pool at unit scales; tep8 does not (see above)."""
    config = _model_config(_ultra_bf16_config(), **_TP8EP2)
    resolved = modeling_v2_resolve(config, _llm_args(mode, kv_cache_config=_kv(dtype="fp8")))
    assert _qualified(resolved) == _ULTRA_BF16_TP8EP2_TARGET


@pytest.mark.parametrize(
    "backend, missed",
    [
        # EP-only: tp8ep2 splits every expert over the 4 ranks of its window
        ("MEGAMOE_CUTEDSL", "expert-parallel only"),
        ("TRTLLM", "fused_moe only"),
        ("CUTEDSL", "moe_config.backend"),
    ],
)
def test_ultra_bf16_tp8ep2_refuses_the_engine_moe_backends(backend, missed):
    config = replace(_model_config(_ultra_bf16_config(), **_TP8EP2), moe_backend_requested=backend)
    _refused(config, missed)


def test_ultra_bf16_tp8ep2_refuses_an_nvfp4_kv_cache():
    config = _model_config(_ultra_bf16_config(), **_TP8EP2)
    _refused(config, "KV cache", kv_cache_config=_kv(dtype="nvfp4"))


# --- BF16 dep16: attention DP over four GB200 trays -------------------------------

_ULTRA_BF16_DEP16_TARGET = (
    "models.nemotron_h.nemotron3_ultra_bf16__sm_100__dep16.modeling."
    "ModelingV2Nemotron3UltraBf16Sm100Dep16"
)
# Four GB200 trays of four: the disaggregated serving tiers' layout (ctx DEP16, gen DEP16).
_DEP16 = dict(
    world_size=16,
    tp_size=16,
    moe_ep_size=16,
    moe_tp_size=1,
    enable_attention_dp=True,
    gpus_per_node=4,
)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_dep16_matches(mode):
    """Attention DP over 16 ranks with the experts in sixteen windows selects dep16."""
    config = _model_config(_ultra_bf16_config(), **_DEP16)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_DEP16_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("backend", ["AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"])
def test_ultra_bf16_dep16_takes_its_moe_backends(backend, mode):
    """Every request runs in the engine's MoE layer; the target checks the leaf it builds."""
    config = replace(_model_config(_ultra_bf16_config(), **_DEP16), moe_backend_requested=backend)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_DEP16_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        # 16 ranks of tensor-parallel attention: a layout no target was built for
        (dict(_DEP16, enable_attention_dp=False), "attention_dp=False"),
        # the experts in eight windows split two ways
        (dict(_DEP16, moe_ep_size=8, moe_tp_size=2), "moe_ep=8"),
        # attention DP over 8 ranks: the replicated non-expert weights would not fit
        (dict(_DEP16, world_size=8, tp_size=8, moe_ep_size=8), "world_size=8"),
        # one 16-GPU node: only GB200 trays of four were gated
        (dict(_DEP16, gpus_per_node=16), "inside one node"),
        # a vocab-parallel lm_head under attention DP
        (dict(_DEP16, enable_lm_head_tp_in_adp=True), "enable_lm_head_tp_in_adp"),
    ],
    ids=["tp16", "moe-ep8-tp2", "dep8", "single-node", "lm-head-tp"],
)
def test_ultra_bf16_dep16_near_misses_do_not_match(mapping_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **mapping_kwargs)
    _refused(config, missed)


@pytest.mark.parametrize(
    "llm_kwargs, missed",
    [
        # SWE accuracy needs the bf16 KV cache and the fp32 state
        (dict(kv_cache_config=_kv(dtype="fp8")), "KV cache"),
        (dict(kv_cache_config=_kv(mamba_ssm_cache_dtype="float16")), "recurrent-state"),
        (dict(kv_cache_config=_kv(enable_block_reuse=True)), "enable_block_reuse"),
        (dict(moe_config=MoeConfig(load_balancer={"num_slots": 512})), "EPLB"),
        (dict(prefill_cuda_graph_backend="piecewise"), "piecewise"),
    ],
    ids=["fp8-kv", "fp16-state", "block-reuse", "eplb", "piecewise"],
)
def test_ultra_bf16_dep16_refuses_engine_choices_it_never_ran(llm_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **_DEP16)
    _refused(config, missed, **llm_kwargs)


def test_ultra_bf16_dep16_block_reuse_is_opt_in(monkeypatch):
    """The disaggregated context tier reuses prefixes; dep16 admits block reuse only when asked
    for, since no gate has run it, and no other target admits it at all."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.routing import (
        BF16_DEP16_BLOCK_REUSE_ENV,
    )

    # trtllm-llmapi-launch drops MV2_, SLURM_, UCX_ and OMPI_ variables before the ranks start.
    assert not BF16_DEP16_BLOCK_REUSE_ENV.startswith(("MV2_", "SLURM_", "UCX_", "OMPI_"))
    reuse = dict(kv_cache_config=_kv(enable_block_reuse=True))
    dep16 = _model_config(_ultra_bf16_config(), **_DEP16)
    monkeypatch.delenv(BF16_DEP16_BLOCK_REUSE_ENV, raising=False)
    _refused(dep16, BF16_DEP16_BLOCK_REUSE_ENV, **reuse)
    monkeypatch.setenv(BF16_DEP16_BLOCK_REUSE_ENV, "1")
    resolved = modeling_v2_resolve(dep16, _llm_args("require", **reuse))
    assert _qualified(resolved) == _ULTRA_BF16_DEP16_TARGET
    _refused(_model_config(_ultra_bf16_config(), **_TEP8), "enable_block_reuse", **reuse)


@pytest.mark.parametrize("backend", ["CUTEDSL", "DEEPGEMM", "MEGAMOE_DEEPGEMM", "TRITON"])
def test_ultra_bf16_dep16_refuses_another_moe_backend(backend):
    config = replace(_model_config(_ultra_bf16_config(), **_DEP16), moe_backend_requested=backend)
    _refused(config, "moe_config.backend")


def test_ultra_bf16_dep16_with_speculative_decoding_does_not_match():
    config = _model_config(_ultra_bf16_config(), spec_config=_NGRAM, **_DEP16)
    _refused(config, "speculative decoding")


def test_ultra_nvfp4_does_not_route_to_dep16():
    """The NVFP4 release on 16 sm_100 ranks with attention DP falls back."""
    config = _model_config(_ultra_config(), **_DEP16)
    _refused(config, r"MIXED_PRECISION")


# --- BF16 dep32: attention DP over eight GB200 trays ------------------------------

_ULTRA_BF16_DEP32_TARGET = (
    "models.nemotron_h.nemotron3_ultra_bf16__sm_100__dep32.modeling."
    "ModelingV2Nemotron3UltraBf16Sm100Dep32"
)
# Eight GB200 trays of four: the disaggregated context tier at DEP32 (its generation tiers at
# dep16 route on their own mappings).
_DEP32 = dict(_DEP16, world_size=32, tp_size=32, moe_ep_size=32)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_dep32_matches(mode):
    """Attention DP over 32 ranks with the experts in thirty-two windows selects dep32."""
    config = _model_config(_ultra_bf16_config(), **_DEP32)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_DEP32_TARGET


def test_ultra_bf16_dep16_and_dep32_each_take_their_own_mapping():
    """A disaggregated deployment's context engine at DEP32 and generation engine at DEP16 are
    two LLM instances; each routes on its own mapping."""
    ctx = _model_config(_ultra_bf16_config(), **_DEP32)
    gen = _model_config(_ultra_bf16_config(), **_DEP16)
    assert _qualified(modeling_v2_resolve(ctx, _llm_args("require"))) == _ULTRA_BF16_DEP32_TARGET
    assert _qualified(modeling_v2_resolve(gen, _llm_args("require"))) == _ULTRA_BF16_DEP16_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("backend", ["AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"])
def test_ultra_bf16_dep32_takes_its_moe_backends(backend, mode):
    """dep16's backends: every request runs in the engine's MoE layer."""
    config = replace(_model_config(_ultra_bf16_config(), **_DEP32), moe_backend_requested=backend)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_DEP32_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        # 32 ranks of tensor-parallel attention: a layout no target was built for
        (dict(_DEP32, enable_attention_dp=False), "attention_dp=False"),
        # the experts in sixteen windows split two ways
        (dict(_DEP32, moe_ep_size=16, moe_tp_size=2), "moe_ep=16"),
        # attention DP over 24 ranks: 512 experts do not tile, and no target was built for it
        (dict(_DEP32, world_size=24, tp_size=24, moe_ep_size=24), "world_size=24"),
        # one 32-GPU node: only GB200 trays of four were measured
        (dict(_DEP32, gpus_per_node=32), "inside one node"),
        # a vocab-parallel lm_head under attention DP
        (dict(_DEP32, enable_lm_head_tp_in_adp=True), "enable_lm_head_tp_in_adp"),
    ],
    ids=["tp32", "moe-ep16-tp2", "dep24", "single-node", "lm-head-tp"],
)
def test_ultra_bf16_dep32_near_misses_do_not_match(mapping_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **mapping_kwargs)
    _refused(config, missed)


@pytest.mark.parametrize(
    "llm_kwargs, missed",
    [
        # dep16's: the bf16 KV cache and the fp32 state only
        (dict(kv_cache_config=_kv(dtype="fp8")), "KV cache"),
        (dict(kv_cache_config=_kv(mamba_ssm_cache_dtype="float16")), "recurrent-state"),
        (dict(kv_cache_config=_kv(enable_block_reuse=True)), "enable_block_reuse"),
        (dict(moe_config=MoeConfig(load_balancer={"num_slots": 512})), "EPLB"),
        (dict(prefill_cuda_graph_backend="piecewise"), "piecewise"),
    ],
    ids=["fp8-kv", "fp16-state", "block-reuse", "eplb", "piecewise"],
)
def test_ultra_bf16_dep32_refuses_engine_choices_it_never_ran(llm_kwargs, missed):
    config = _model_config(_ultra_bf16_config(), **_DEP32)
    _refused(config, missed, **llm_kwargs)


def test_ultra_bf16_dep32_block_reuse_is_opt_in_with_its_own_switch(monkeypatch):
    """dep32 admits block reuse only when its own switch asks for it; dep16's switch does not
    turn it on for dep32, nor dep32's for dep16."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.routing import (
        BF16_DEP16_BLOCK_REUSE_ENV,
        BF16_DEP32_BLOCK_REUSE_ENV,
    )

    assert BF16_DEP32_BLOCK_REUSE_ENV != BF16_DEP16_BLOCK_REUSE_ENV
    # trtllm-llmapi-launch drops MV2_, SLURM_, UCX_ and OMPI_ variables before the ranks start.
    assert not BF16_DEP32_BLOCK_REUSE_ENV.startswith(("MV2_", "SLURM_", "UCX_", "OMPI_"))
    reuse = dict(kv_cache_config=_kv(enable_block_reuse=True))
    dep32 = _model_config(_ultra_bf16_config(), **_DEP32)
    dep16 = _model_config(_ultra_bf16_config(), **_DEP16)
    monkeypatch.delenv(BF16_DEP32_BLOCK_REUSE_ENV, raising=False)
    monkeypatch.setenv(BF16_DEP16_BLOCK_REUSE_ENV, "1")
    _refused(dep32, BF16_DEP32_BLOCK_REUSE_ENV, **reuse)
    monkeypatch.delenv(BF16_DEP16_BLOCK_REUSE_ENV)
    monkeypatch.setenv(BF16_DEP32_BLOCK_REUSE_ENV, "1")
    resolved = modeling_v2_resolve(dep32, _llm_args("require", **reuse))
    assert _qualified(resolved) == _ULTRA_BF16_DEP32_TARGET
    _refused(dep16, BF16_DEP16_BLOCK_REUSE_ENV, **reuse)
    _refused(_model_config(_ultra_bf16_config(), **_TEP8), "enable_block_reuse", **reuse)


@pytest.mark.parametrize("backend", ["CUTEDSL", "DEEPGEMM", "MEGAMOE_DEEPGEMM", "TRITON"])
def test_ultra_bf16_dep32_refuses_another_moe_backend(backend):
    config = replace(_model_config(_ultra_bf16_config(), **_DEP32), moe_backend_requested=backend)
    _refused(config, "moe_config.backend")


def test_ultra_bf16_dep32_with_speculative_decoding_does_not_match():
    config = _model_config(_ultra_bf16_config(), spec_config=_NGRAM, **_DEP32)
    _refused(config, "speculative decoding")


def test_ultra_nvfp4_does_not_route_to_dep32():
    """The NVFP4 release on 32 sm_100 ranks with attention DP falls back."""
    config = _model_config(_ultra_config(), **_DEP32)
    _refused(config, r"MIXED_PRECISION")


# --- NVFP4 on sm_107 -------------------------------------------------------------

_ULTRA_NVFP4_TEP4_TARGET = (
    "models.nemotron_h.nemotron3_ultra_nvfp4__sm_107__tep4.modeling."
    "ModelingV2Nemotron3UltraNvfp4Sm107Tep4"
)
_ULTRA_NVFP4_DEP4_TARGET = (
    "models.nemotron_h.nemotron3_ultra_nvfp4__sm_107__dep4.modeling."
    "ModelingV2Nemotron3UltraNvfp4Sm107Dep4"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=False)
_DEP4 = dict(_TEP4, enable_attention_dp=True)


@pytest.fixture
def _on_sm107(monkeypatch):
    """Route as if this were a Rubin GPU."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def _nvfp4_config(**mapping):
    """The NVFP4 release as the loader resolves it: its config declares an FP8 KV cache."""
    return _model_config(_ultra_config(), kv_cache_quant_algo=QuantAlgo.FP8, **mapping)


def _nvfp4_kv(**overrides) -> KvCacheConfig:
    """The KV cache configuration the NVFP4 gates ran with: block reuse off, fp16 state."""
    return _kv(**dict(dict(mamba_ssm_cache_dtype="float16"), **overrides))


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping, target",
    [(_TEP4, _ULTRA_NVFP4_TEP4_TARGET), (_DEP4, _ULTRA_NVFP4_DEP4_TARGET)],
    ids=["tep4", "dep4"],
)
def test_ultra_nvfp4_matches(_on_sm107, mapping, target, mode):
    """Attention DP is an identity criterion: it selects the replicated-attention target."""
    resolved = modeling_v2_resolve(
        _nvfp4_config(**mapping), _llm_args(mode, kv_cache_config=_nvfp4_kv())
    )
    assert _qualified(resolved) == target


@pytest.mark.parametrize(
    "mapping, llm_kwargs, missed",
    [
        (dict(_TEP4, moe_ep_size=1, moe_tp_size=4), {}, "moe_ep=1"),
        (dict(world_size=8, tp_size=8, moe_ep_size=8), {}, "world_size=8"),
        (_TEP4, dict(kv_cache_config=_nvfp4_kv(mamba_ssm_cache_dtype="float32")), "float32"),
        (
            _TEP4,
            dict(kv_cache_config=_nvfp4_kv(mamba_ssm_stochastic_rounding=True)),
            "stochastic_rounding",
        ),
        (_DEP4, dict(kv_cache_config=_nvfp4_kv(dtype="nvfp4")), "KV cache"),
        (_TEP4, dict(kv_cache_config=_nvfp4_kv(enable_block_reuse=True)), "enable_block_reuse"),
    ],
    ids=["moe-tp4", "eight-ranks", "fp32-state", "stochastic-rounding", "nvfp4-kv", "reuse"],
)
def test_ultra_nvfp4_near_misses_do_not_match(_on_sm107, mapping, llm_kwargs, missed):
    llm_kwargs.setdefault("kv_cache_config", _nvfp4_kv())
    _refused(_nvfp4_config(**mapping), missed, **llm_kwargs)


def test_ultra_nvfp4_refuses_another_moe_backend(_on_sm107):
    config = replace(_nvfp4_config(**_TEP4), moe_backend_requested="CUTLASS")
    _refused(config, "moe_config.backend", kv_cache_config=_nvfp4_kv())


def test_ultra_nvfp4_on_sm103_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    _refused(_nvfp4_config(**_TEP4), r"compute capability \(10, 3\)", kv_cache_config=_nvfp4_kv())
