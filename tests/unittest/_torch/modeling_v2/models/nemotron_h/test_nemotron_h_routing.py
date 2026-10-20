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


def test_ultra_bf16_refuses_another_moe_backend():
    config = replace(_model_config(_ultra_bf16_config(), **_TEP8), moe_backend_requested="TRTLLM")
    _refused(config, "moe_config.backend")


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


def test_ultra_bf16_tp8ep2_refuses_an_fp8_kv_cache():
    config = _model_config(_ultra_bf16_config(), **_TP8EP2)
    _refused(config, "KV cache", kv_cache_config=_kv(dtype="fp8"))
