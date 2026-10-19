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
    ModelingV2Mode,
    modeling_v2_resolve,
    resolved_mode,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

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


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_the_target_reads_the_mode_it_was_resolved_under(mode):
    """A target raises instead of falling back under require, so it has to know which mode built it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert resolved_mode(target) is ModelingV2Mode(mode)


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


@pytest.mark.parametrize(
    "mapping_kwargs, parallel",
    [
        (dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1), "tp4"),
        (dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1), "tp8"),
        # experts split by expert rather than by tensor: another weight layout, no target
        (dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8), None),
        (dict(world_size=8, tp_size=8, moe_tp_size=8, enable_attention_dp=True), None),
        (dict(world_size=2, tp_size=2, moe_tp_size=2), None),
    ],
)
def test_minimax_m3_parallel_names_the_target_topology(mapping_kwargs, parallel):
    """MiniMax-M3 has one target per tensor-parallel width: tp4 and tp8 load different per-rank shapes."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    assert routing._parallel(Mapping(**mapping_kwargs)) == parallel


# The MiniMax-M3 Eagle3 drafter of the released checkpoint (inferact MiniMax-M3-EAGLE3-GQA), as its config.json reads.
_M3_DRAFTER = dict(
    num_hidden_layers=1,
    fc_norm=True,
    hidden_size=6144,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    intermediate_size=18432,
    vocab_size=200064,
    draft_vocab_size=200064,
    attention_bias=False,
    tie_word_embeddings=False,
)


def _m3_drafter_spec(tmp_path, captured=None, **overrides):
    import json
    from types import SimpleNamespace

    (tmp_path / "config.json").write_text(json.dumps({**_M3_DRAFTER, **overrides}))
    return SimpleNamespace(speculative_model=str(tmp_path), eagle3_layers_to_capture=captured)


@pytest.mark.parametrize(
    "overrides, captured, reason",
    [
        ({}, None, None),
        ({}, {1, 29, 56}, None),
        (dict(num_hidden_layers=2), None, "2 drafter layers, not 1"),
        (dict(fc_norm=False), None, "no per-layer fc_norm"),
        (dict(eagle_config={"eh_proj_before_attn": True}), None, "eh_proj before attention"),
        (dict(eagle_config={"norm_before_fc": True}), None, "norm before the fc"),
        (dict(kv_lora_rank=512), None, "an MLA drafter layer"),
        (dict(target_hidden_size=4096), None, "a target hidden size other than the drafter's"),
        (dict(draft_vocab_size=32000), None, "a draft vocabulary other than the target's"),
        (dict(mlp_bias=True), None, "biased projections"),
        (dict(tie_word_embeddings=True), None, "a tied head"),
        ({}, {-1}, "a single captured target layer"),
        (dict(num_key_value_heads=1), None, "num_key_value_heads=1, not 4"),
        (dict(intermediate_size=14336), None, "intermediate_size=14336, not 18432"),
        (
            dict(hidden_size=4096, head_dim=None),
            None,
            "hidden_size=4096, not 6144; head_dim=64, not 128",
        ),
    ],
    ids=[
        "released_drafter",
        "three_captured_layers",
        "two_layers",
        "no_fc_norm",
        "eh_proj_before_attn",
        "norm_before_fc",
        "mla",
        "target_hidden",
        "draft_vocab",
        "bias",
        "tied_head",
        "one_captured_layer",
        "one_kv_head",
        "mlp_width",
        "hidden_size",
    ],
)
def test_minimax_m3_routes_only_the_drafter_its_targets_build(
    tmp_path, overrides, captured, reason
):
    """The targets draft with their own one-layer fc_norm Eagle3 drafter: any other drafter is declined in routing
    (``auto`` takes the built-in model, ``require`` names the criterion) instead of failing at construction."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    spec = _m3_drafter_spec(tmp_path, captured, **overrides)
    assert routing._drafter_mismatch(spec) == reason
    assert routing._drafter_mismatch(None) is None


@pytest.mark.parametrize("drafter_layers", [1, 3], ids=["flat_drafter", "multi_layer_drafter"])
def test_minimax_m3_route_declines_a_drafter_it_does_not_build(
    tmp_path, monkeypatch, drafter_layers
):
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    monkeypatch.setattr(routing, "_eagle3_one_model_or_none", lambda spec: True)
    text_config = dict(
        zip(
            (
                "num_hidden_layers",
                "hidden_size",
                "num_local_experts",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
            ),
            routing._SHAPE,
        )
    )
    spec = _m3_drafter_spec(tmp_path, num_hidden_layers=drafter_layers)
    spec.spec_dec_mode = SimpleNamespace(name="EAGLE3_ONE_MODEL")
    mapping = Mapping(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1)
    pretrained = SimpleNamespace(text_config=text_config)
    runtime = _m3_runtime()
    if drafter_layers == 1:
        target = routing.route(pretrained, mapping, routing._SM, spec, runtime=runtime)
        assert target.__name__ == "ModelingV2MiniMaxM3Nvfp4Sm103Tp4"
    else:
        with pytest.raises(NoTarget, match="Eagle3 drafter: 3 drafter layers, not 1"):
            routing.route(pretrained, mapping, routing._SM, spec, runtime=runtime)


def _m3_sparse(**overrides):
    """The sparse-attention runtime of the PR deployment (MSA, FP8 indexer, fused projection), with overrides."""
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    fields = dict(
        implementation="msa",
        indexer_kv_dtype="fp8",
        fuse_qkv_index_projection=True,
        sparse_disable_index_value=True,
    )
    fields.update(overrides)
    return MiniMaxM3SparseAttentionConfig(**fields)


def _m3_runtime(sparse=None, kv="NVFP4"):
    from tensorrt_llm._torch._experimental.modeling_v2._router_index import RuntimeFacts

    return RuntimeFacts(
        sparse_attention_config=_m3_sparse() if sparse is None else sparse, kv_cache_quant_algo=kv
    )


# Every runtime the engine accepts for MiniMax-M3 that the targets are not written for; the built-in model runs them.
_M3_RUNTIME_DECLINES = [
    (
        dict(implementation="triton", indexer_kv_dtype="bf16", fuse_qkv_index_projection=False),
        "FP8",
        "sparse attention implementation 'triton', not 'msa'; indexer_kv_dtype 'bf16', not 'fp8'",
    ),
    (
        dict(indexer_kv_dtype="bf16", fuse_qkv_index_projection=False),
        "FP8",
        "indexer_kv_dtype 'bf16', not 'fp8'",
    ),
    (
        dict(fuse_qkv_index_projection=False),
        None,
        "a BF16 main KV cache, not FP8 or NVFP4",
    ),
]
_M3_RUNTIME_DECLINE_IDS = ["triton", "bf16_indexer", "bf16_main_kv"]


@pytest.mark.parametrize(
    "sparse, kv",
    [
        (dict(), "NVFP4"),
        (dict(), "FP8"),
        (dict(fuse_qkv_index_projection=False), "FP8"),
        (dict(fuse_qkv_index_projection=False), "NVFP4"),
    ],
    ids=["fused_nvfp4", "fused_fp8", "unfused_fp8", "unfused_nvfp4"],
)
def test_minimax_m3_runtime_the_targets_are_written_for(sparse, kv):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    assert routing._runtime_mismatch(_m3_runtime(_m3_sparse(**sparse), kv)) is None


@pytest.mark.parametrize("sparse, kv, reason", _M3_RUNTIME_DECLINES, ids=_M3_RUNTIME_DECLINE_IDS)
def test_minimax_m3_runtime_mismatch_names_the_setting(sparse, kv, reason):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    assert routing._runtime_mismatch(_m3_runtime(_m3_sparse(**sparse), kv)) == reason


def test_minimax_m3_runtime_without_its_sparse_config_is_declined():
    from tensorrt_llm._torch._experimental.modeling_v2._router_index import RuntimeFacts
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    runtime = RuntimeFacts(sparse_attention_config=None, kv_cache_quant_algo="NVFP4")
    assert routing._runtime_mismatch(runtime) == "no MiniMax-M3 sparse_attention_config"


def _m3_resolve_config(sparse, kv_checkpoint, tp=4):
    """A ModelConfig as the loader holds it when it resolves: the MiniMax-M3 VL wrapper with the target text shape,
    the checkpoint's KV quantization (before kv_cache_config.dtype is applied) and the sparse-attention runtime."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.models.modeling_utils import QuantConfig

    text_config = dict(
        zip(
            (
                "num_hidden_layers",
                "hidden_size",
                "num_local_experts",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
            ),
            routing._SHAPE,
        )
    )
    pretrained = PretrainedConfig(
        architectures=["MiniMaxM3SparseForConditionalGeneration"], text_config=text_config
    )
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(world_size=tp, tp_size=tp, moe_tp_size=tp, moe_ep_size=1),
        quant_config=QuantConfig(kv_cache_quant_algo=kv_checkpoint),
        sparse_attention_config=sparse,
    )


def _m3_llm_args(mode, kv_dtype):
    return _llm_args(mode, kv_cache_config=dict(dtype=kv_dtype))


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_minimax_m3_pr_runtime_resolves_to_the_target(monkeypatch, mode, tp):
    """The PR deployment (MSA, FP8 indexer, fused projection, kv_cache_config.dtype='nvfp4') takes the target."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(), None, tp)
    target = modeling_v2_resolve(config, _m3_llm_args(mode, "nvfp4"))
    assert target.__name__ == f"ModelingV2MiniMaxM3Nvfp4Sm103Tp{tp}"


@pytest.mark.parametrize("sparse, kv, reason", _M3_RUNTIME_DECLINES, ids=_M3_RUNTIME_DECLINE_IDS)
@pytest.mark.parametrize("tp", [4, 8])
def test_minimax_m3_other_runtimes_fall_back_under_auto(monkeypatch, sparse, kv, reason, tp):
    """``auto`` resolves nothing, so the loader builds the registered built-in MiniMax-M3 model."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    # kv_cache_config.dtype 'auto' keeps the checkpoint's KV quantization (None: the BF16 cache).
    config = _m3_resolve_config(_m3_sparse(**sparse), kv, tp)
    assert modeling_v2_resolve(config, _m3_llm_args("auto", "auto")) is None
    resolved = replace(config, modeling_v2_target=None)
    assert AutoModelForCausalLM._resolve_class(resolved) is MiniMaxM3VLForConditionalGeneration


@pytest.mark.parametrize("sparse, kv, reason", _M3_RUNTIME_DECLINES, ids=_M3_RUNTIME_DECLINE_IDS)
def test_minimax_m3_other_runtimes_raise_under_require(monkeypatch, sparse, kv, reason):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(**sparse), kv)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _m3_llm_args("require", "auto"))
    message = str(excinfo.value)
    assert "MiniMaxM3SparseForConditionalGeneration" in message
    assert f"runtime: {reason}" in message


def test_minimax_m3_kv_cache_dtype_overrides_the_checkpoint(monkeypatch):
    """An explicit kv_cache_config.dtype is the KV cache the engine runs with, whatever the checkpoint declares."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(fuse_qkv_index_projection=False), None)
    assert modeling_v2_resolve(config, _m3_llm_args("auto", "auto")) is None
    target = modeling_v2_resolve(config, _m3_llm_args("auto", "fp8"))
    assert target.__name__ == "ModelingV2MiniMaxM3Nvfp4Sm103Tp4"


# --- qwen4_exp: one-model MTP-Eagle over a linear chain, one rank ---------------

_QWEN4_EXP_TARGET = "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."


@pytest.mark.parametrize(
    "spec_config",
    [None, MTPDecodingConfig(max_draft_len=1), MTPDecodingConfig(max_draft_len=3)],
    ids=["no-spec", "mtp1", "mtp3"],
)
def test_qwen4_exp_tp1_matches(spec_config):
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(target).startswith(_QWEN4_EXP_TARGET)


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
        MTPDecodingConfig(max_draft_len=4),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        _mtp(max_total_draft_tokens=5),
        _mtp(num_nextn_predict_layers=2),
    ],
    ids=[
        "ngram",
        "draft4",
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
