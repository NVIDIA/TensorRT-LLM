# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``modeling_v2_resolve`` actually does, driven by synthetic configs.

No checkpoint and no weights: routing reads config *shape*, the mapping, the
SM version and the LLM API arguments the deployment was configured with, all
of which can be stated directly. The SM version is monkeypatched so these run
on any device -- the point here is the decision logic, not the kernels.

The things worth proving:

* ``off`` changes nothing. This is the whole safety argument for making the
  decision in the model loader at all.
* a matching configuration resolves to the target class itself, imported
  from its own directory; no registry lookup stands between the decision and
  the class.
* ``require`` raises on a near-miss and says, in words, which criterion
  missed; ``auto`` falls back to the built-in implementation instead.
* speculative decoding is an identity criterion: a target that does not
  implement the requested mode does not claim the deployment.
* ``AutoModelForCausalLM._resolve_class`` reads the decision off the
  ``ModelConfig`` rather than deciding again.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    engine_facts,
    modeling_v2_resolve,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import (
    routing as qwen4_exp_routing,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_TARGET = "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling.ModelingV2GptOss120bSm103Tp1"
_R1_TARGET = (
    "models.deepseek_v3.r1_0528_nvfp4__sm_103__dep4.modeling.ModelingV2DeepseekR10528Nvfp4Sm103Dep4"
)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _gpt_oss_config(**overrides):
    """The shape gpt-oss-120b's own config.json declares."""
    fields = dict(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=36,
        hidden_size=2880,
        num_local_experts=128,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _r1_config(**overrides):
    """The shape DeepSeek-R1-0528-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=61,
        hidden_size=7168,
        n_routed_experts=256,
        q_lora_rank=1536,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it.

    The switch is an LLM API argument, so the tests set one too -- that is the
    surface under test.
    """
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config, mapping=mapping, spec_config=spec_config
    )


# A speculative mode no target implements.
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qwen4_exp_config(**vision_overrides):
    """The shape Qwen3.8-Flash-Next-NVFP4's normalized composite config declares."""
    text = PretrainedConfig(
        num_hidden_layers=48,
        hidden_size=2560,
        num_experts=512,
        num_attention_heads=24,
        linear_num_value_heads=48,
        vocab_size=248320,
    )
    vision = dict(
        depth=27,
        hidden_size=1152,
        num_heads=16,
        spatial_merge_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh",
    )
    vision.update(vision_overrides)
    return PretrainedConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        vision_config=PretrainedConfig(**vision),
    )


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("mode", [None, "off"], ids=["default", "off"])
def test_off_resolves_nothing(mode):
    """The default must be indistinguishable from modeling_v2 not existing.

    Leaving the argument alone and an explicit "off" have to behave
    identically: the common case is that nobody has heard of this package.
    """
    args = TorchLlmArgs(model=_DUMMY_MODEL) if mode is None else _llm_args(mode)
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, args) is None


def test_off_still_reaches_the_builtin_implementation():
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == _BUILTIN_GPT_OSS


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(mode):
    config = _model_config(_gpt_oss_config())
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GPT_OSS_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(mode):
    config = _model_config(_r1_config(), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _R1_TARGET


def test_resolve_class_reads_the_decision_off_the_config():
    """The loader decides and writes the class on the ModelConfig; the class
    lookup only has to honour it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args())
    decided = replace(config, modeling_v2_target=target)
    assert AutoModelForCausalLM._resolve_class(decided) is target


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, r"shape \(24, 2880, 32\)"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "2 ranks"),
    ],
)
def test_gpt_oss_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args()) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "no target fits" in message and "GptOssForCausalLM" in message
    assert "compute capability (10, 0)" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_an_unrouted_architecture_raises_under_require():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_an_unknown_mode_is_rejected_when_the_arguments_are_built():
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent. Pydantic refuses it before anything is built.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    with pytest.raises(ValidationError, match="modeling_v2"):
        _llm_args("yes")


# --- speculative decoding is an identity criterion ------------------------------


def test_gpt_oss_with_speculative_decoding_falls_back_under_auto():
    """The tp1 target implements no speculative path, so claiming the
    deployment would trade a fallback for a crash inside the target."""
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_gpt_oss_with_speculative_decoding_raises_under_require():
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "NGramDecodingConfig" in message


def test_r1_dep4_with_mtp_matches():
    config = _model_config(_r1_config(), spec_config=MTPDecodingConfig(max_draft_len=3), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _R1_TARGET


def test_r1_dep4_with_another_speculative_mode_does_not_match():
    """The dep4 target carries the checkpoint's MTP module and nothing else."""
    config = _model_config(_r1_config(), spec_config=_NGRAM, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_r1_dep4_with_vanilla_mtp_does_not_match():
    """``use_mtp_vanilla`` resolves to the multi-model MTP loop; the target
    asserts the one-model MTP-Eagle loop at construction, so route must say
    no before that assert can."""
    vanilla = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
    config = _model_config(_r1_config(), spec_config=vanilla, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- qwen4_exp: one-model MTP-Eagle over a linear chain, one rank ---------------

_QWEN4_EXP_TARGET = "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."


@pytest.mark.parametrize(
    "spec_config",
    [
        None,
        MTPDecodingConfig(max_draft_len=1),
        MTPDecodingConfig(max_draft_len=3),
        MTPDecodingConfig(max_draft_len=7),
    ],
    ids=["no-spec", "mtp1", "mtp3", "mtp7"],
)
def test_qwen4_exp_tp1_matches(spec_config):
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    # The recurrent-state cache at either dtype the target serves; `auto` is bf16 here.
    for dtype in ("float32", "bfloat16", "auto"):
        kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype=dtype)
        target = modeling_v2_resolve(config, _llm_args("require", kv_cache_config=kv_cache_config))
        assert _qualified(target).startswith(_QWEN4_EXP_TARGET), dtype


def _mtp(max_draft_len=3, **attributes):
    """An MTP config with fields the engine fills in after load set directly: the
    constructor remaps or recomputes them, so passing them would test nothing."""
    config = MTPDecodingConfig(max_draft_len=max_draft_len)
    for name, value in attributes.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize(
    "spec_config",
    [
        _NGRAM,
        MTPDecodingConfig(max_draft_len=8),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        _mtp(max_total_draft_tokens=5),
        _mtp(num_nextn_predict_layers=2),
    ],
    ids=[
        "ngram",
        "draft8",
        "vanilla-mtp",
        "max-total-draft",
        "two-predictors",
    ],
)
def test_qwen4_exp_unsupported_speculation_falls_back(spec_config):
    """The target asserts each of these at construction; route must decline first."""
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError):
        modeling_v2_resolve(config, _llm_args("require"))


def test_qwen4_exp_deepstack_vision_tower_does_not_match():
    config = _model_config(_qwen4_exp_config(deepstack_visual_indexes=[5, 11, 17]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_qwen4_exp_on_two_ranks_does_not_match():
    config = _model_config(_qwen4_exp_config(), world_size=2, tp_size=2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


# --- qwen4_exp: engine choices the constructor would refuse -----------------------


def _declined(config, llm_args_kwargs=None, match=None):
    """``auto`` falls back to the built-in model and ``require`` says why."""
    kwargs = llm_args_kwargs or {}
    assert modeling_v2_resolve(config, _llm_args("auto", **kwargs)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", **kwargs))


def _claimed(config, llm_args_kwargs=None):
    target = modeling_v2_resolve(config, _llm_args("require", **(llm_args_kwargs or {})))
    assert _qualified(target).startswith(_QWEN4_EXP_TARGET)


@pytest.mark.parametrize("algo", ["fp8", "nvfp4"])
def test_qwen4_exp_quantized_kv_cache_dtypes_match(algo):
    _claimed(_model_config(_qwen4_exp_config()), dict(kv_cache_config=KvCacheConfig(dtype=algo)))


@pytest.mark.parametrize("algo", [QuantAlgo.INT8, QuantAlgo.FP8_BLOCK_SCALES])
def test_qwen4_exp_other_kv_cache_algorithms_are_declined(algo):
    facts = engine_facts(_model_config(_qwen4_exp_config()), _llm_args("auto"))
    with pytest.raises(NoTarget, match="kv_cache_quant_algo"):
        qwen4_exp_routing._check_engine(None, facts._replace(kv_cache_quant_algo=algo))


def test_qwen4_exp_piecewise_prefill_cuda_graph_falls_back():
    config = _model_config(_qwen4_exp_config())
    _declined(config, dict(prefill_cuda_graph_backend="piecewise"), match="piecewise")


@pytest.mark.parametrize("backend", ["disabled", "breakable"])
def test_qwen4_exp_matches_without_or_with_the_breakable_prefill_cuda_graph(backend):
    _claimed(
        _model_config(_qwen4_exp_config()),
        dict(prefill_cuda_graph_backend=backend),
    )


@pytest.mark.parametrize("backend", ["CUTLASS", "TRITON", "DEEPGEMM", "VANILLA"])
def test_qwen4_exp_unimplemented_moe_backend_falls_back(backend):
    config = replace(_model_config(_qwen4_exp_config()), moe_backend_requested=backend)
    _declined(config, match="moe_config.backend")


@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM", "CUTEDSL"])
def test_qwen4_exp_implemented_moe_backend_matches(backend):
    _claimed(replace(_model_config(_qwen4_exp_config()), moe_backend_requested=backend))


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        dict(max_concurrency=8),
        dict(draft_len_schedule={4: 3, 8: 1}),
        dict(acceptance_rate_threshold=0.5),
    ],
    ids=["max-concurrency", "draft-len-schedule", "acceptance-rate"],
)
def test_qwen4_exp_runtime_draft_length_falls_back(spec_kwargs):
    """The verify step runs at one width; a draft length that changes at runtime
    would trip the width assert in the first reduced step."""
    spec = MTPDecodingConfig(max_draft_len=3, **spec_kwargs)
    _declined(_model_config(_qwen4_exp_config(), spec_config=spec), match="draft length")


@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_qwen4_exp_draft_beyond_three_needs_the_cached_replay(monkeypatch, max_draft_len):
    """The legacy per-step recurrent states are certified at up to three drafted tokens and
    hold one fp32 state per verify row."""
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(
        _qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=max_draft_len)
    )
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float32")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="TRTLLM_USE_GDN_REPLAY=0")
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    _claimed(config, dict(kv_cache_config=kv_cache_config))


def test_qwen4_exp_legacy_verify_matches_up_to_three_drafted_tokens(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32")))


@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_qwen4_exp_cutedsl_declines_drafts_beyond_three(monkeypatch, max_draft_len):
    """The CuteDSL expert entries capture up to 128 rows (32 requests of 4 verify rows)."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float32")
    spec = MTPDecodingConfig(max_draft_len=max_draft_len)
    cutedsl = replace(
        _model_config(_qwen4_exp_config(), spec_config=spec), moe_backend_requested="CUTEDSL"
    )
    _declined(cutedsl, dict(kv_cache_config=kv_cache_config), match="CUTEDSL")
    _claimed(
        _model_config(_qwen4_exp_config(), spec_config=spec), dict(kv_cache_config=kv_cache_config)
    )


def test_qwen4_exp_cutedsl_matches_up_to_three_drafted_tokens(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    spec = MTPDecodingConfig(max_draft_len=3)
    cutedsl = replace(
        _model_config(_qwen4_exp_config(), spec_config=spec), moe_backend_requested="CUTEDSL"
    )
    _claimed(cutedsl, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32")))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "auto"])
def test_qwen4_exp_mtp_matches_a_float32_or_bfloat16_state_cache(monkeypatch, dtype):
    """GDN cached replay runs on either checkpoint dtype; `auto` resolves to bf16 here."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype)))


@pytest.mark.parametrize(
    "spec_config", [None, MTPDecodingConfig(max_draft_len=3)], ids=["no-spec", "mtp3"]
)
def test_qwen4_exp_declines_a_float16_state_cache(monkeypatch, spec_config):
    """The engine's replay gate accepts fp16; the target's pool contract does not."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float16")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="recurrent-state cache")


def test_qwen4_exp_mtp_on_a_float16_state_cache_declines_with_replay_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float16")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="recurrent-state cache")


def test_qwen4_exp_mtp_on_a_bf16_state_cache_matches_with_replay_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config)


def test_qwen4_exp_no_speculation_matches_the_default_state_cache(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    _claimed(_model_config(_qwen4_exp_config()))
