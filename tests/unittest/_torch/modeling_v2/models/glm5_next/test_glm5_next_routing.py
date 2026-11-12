# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM100 = (10, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_SPLIT = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling."
    "ModelingV2Glm53FlashNvfp4Sm100Default"
)
_FUSED = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default_mtp_fused.modeling."
    "ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a B200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def _llm_args(mode: str, **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _flash_config(**text_overrides) -> PretrainedConfig:
    """The composite config GLM-5.3-Flash's config.json declares, its text_config cut to the routed fields."""
    text_config = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        hidden_size=4096,
        n_routed_experts=288,
        q_lora_rank=1536,
        linear_num_heads=64,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        hc_mult=4,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        n_shared_experts=1,
        tie_word_embeddings=False,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    )
    text_config.update(text_overrides)
    return PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=text_config,
    )


def _model_config(
    pretrained_config,
    quant_algo=QuantAlgo.NVFP4,
    spec_config=None,
    kv_cache_quant_algo=None,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TEP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
    )


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, enable_attention_dp=True)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config, mapping_kwargs",
    [
        (MTPDecodingConfig(max_draft_len=3), _TEP4),
        (MTPDecodingConfig(max_draft_len=1), dict(world_size=8, tp_size=8, moe_tp_size=8)),
        (MTPDecodingConfig(max_draft_len=3), _DEP4),
    ],
    ids=["tep4-mtp3", "tp8-mtp1", "dep4-mtp3"],
)
def test_linear_mtp_eagle_routes_to_the_fused_target(mode, spec_config, mapping_kwargs):
    """The fused target takes every linear one-model MTP-Eagle deployment it supports."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _FUSED


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config, mapping_kwargs, llm_kwargs",
    [
        (None, _TEP4, {}),
        (None, _DEP4, {}),
        (MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3, use_rejection_sampling=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3, use_relaxed_acceptance_for_thinking=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3), _TEP4, dict(guided_decoding_backend="xgrammar")),
        (SADecodingConfig(max_draft_len=3), _TEP4, {}),
    ],
    ids=[
        "no-spec",
        "dep4-no-spec",
        "vanilla-mtp",
        "rejection-sampling",
        "relaxed-acceptance",
        "guided-decoding",
        "suffix-automaton",
    ],
)
def test_other_deployments_route_to_the_split_target(mode, spec_config, mapping_kwargs, llm_kwargs):
    """What the fused target does not compute falls to the split target."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **llm_kwargs))) == _SPLIT


@pytest.mark.parametrize(
    "text_overrides, quant_algo, spec_config, mapping_kwargs, missed",
    [
        (dict(num_hidden_layers=44), QuantAlgo.NVFP4, None, _TEP4, r"shape \(44, 4096"),
        (dict(), QuantAlgo.FP8_BLOCK_SCALES, None, _TEP4, "quantization FP8_BLOCK_SCALES"),
        (dict(), QuantAlgo.NVFP4, None, dict(world_size=8, tp_size=4, pp_size=2), "pp=2"),
        (
            dict(),
            QuantAlgo.NVFP4,
            NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            _TEP4,
            "NGRAM",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused"),
            _TEP4,
            "Eagle3DecodingConfig",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            MTPDecodingConfig(max_draft_len=3),
            dict(_DEP4, enable_lm_head_tp_in_adp=True),
            "LM head split",
        ),
    ],
    ids=["other-shape", "fp8-checkpoint", "pipeline-parallel", "ngram", "eagle3", "adp-split-head"],
)
def test_flash_near_misses_do_not_match(
    text_overrides, quant_algo, spec_config, mapping_kwargs, missed
):
    config = _model_config(
        _flash_config(**text_overrides),
        quant_algo=quant_algo,
        spec_config=spec_config,
        **mapping_kwargs,
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_flash_on_sm103_does_not_match(monkeypatch):
    """The targets have been run on B200 only; a GB300 falls back."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_flash_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


# The block-FP8 target: the FP8_BLOCK_SCALES checkpoint on sm_100, tp4, FP8 KV cache, text only,
# with no speculation or one-model MTP-Eagle at three linear drafts.

_FP8 = "models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling.ModelingV2Glm53FlashFp8Sm10xTp4"
_TP4 = dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, gpus_per_node=8)


def _fp8_config(
    spec_config=None, kv_cache_quant_algo=QuantAlgo.FP8, text_overrides=None, **mapping
):
    """The block-FP8 checkpoint at the deployment its target admits, unless overridden."""
    return _model_config(
        _flash_config(**(text_overrides or {})),
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        spec_config=spec_config,
        kv_cache_quant_algo=kv_cache_quant_algo,
        **(mapping or _TP4),
    )


def _fp8_llm_args(mode: str, **kwargs) -> TorchLlmArgs:
    kwargs.setdefault("disable_mm_encoder", True)
    return _llm_args(mode, **kwargs)


def _mtp3(**fields) -> MTPDecodingConfig:
    """MTP-Eagle at three drafts with ``fields`` set as given, bypassing the validators that would
    reject some of them on their own (a draft checkpoint path that does not exist, a tree without
    its sizes): routing reads the fields, not their consistency."""
    return MTPDecodingConfig(max_draft_len=3).model_copy(update=fields)


def _fp8_declined(config, llm_args_of, missed: str) -> str:
    """``auto`` falls back to the built-in model; ``require`` raises with ``missed`` in the reason,
    which it returns."""
    assert modeling_v2_resolve(config, llm_args_of("auto")) is None
    with pytest.raises(ValueError, match=missed) as raised:
        modeling_v2_resolve(config, llm_args_of("require"))
    return str(raised.value)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("spec_config", [None, _mtp3()], ids=["no-spec", "mtp3"])
@pytest.mark.parametrize("moe_backend", ["AUTO", "TRTLLM"])
@pytest.mark.parametrize("allreduce_strategy", ["AUTO", "ONESHOT"])
def test_fp8_tp4_routes_to_the_fp8_target(allreduce_strategy, moe_backend, spec_config, mode):
    llm_args = _fp8_llm_args(
        mode, moe_config={"backend": moe_backend}, allreduce_strategy=allreduce_strategy
    )
    assert _qualified(modeling_v2_resolve(_fp8_config(spec_config), llm_args)) == _FP8


def test_fp8_kv_cache_dtype_override_routes_to_the_fp8_target():
    """Deployments ask for the FP8 latent cache with kv_cache_config.dtype; the checkpoint declares none."""
    llm_args = _fp8_llm_args("require", kv_cache_config={"dtype": "fp8"})
    config = _fp8_config(kv_cache_quant_algo=None)
    assert _qualified(modeling_v2_resolve(config, llm_args)) == _FP8


def test_fp8_target_is_built_only_when_modeling_v2_is_set():
    """The built-in model's defaults are applied before routing; they do not count as setting modeling_v2."""
    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVLM
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    llm_args = TorchLlmArgs(model=_DUMMY_MODEL, disable_mm_encoder=True)
    apply_model_defaults_to_llm_args(llm_args, Glm5NextVLM.get_model_defaults(llm_args))
    assert llm_args.allreduce_strategy == "ONESHOT"
    assert modeling_v2_resolve(_fp8_config(_mtp3()), llm_args) is None
    llm_args.modeling_v2 = "auto"
    assert _qualified(modeling_v2_resolve(_fp8_config(_mtp3()), llm_args)) == _FP8


@pytest.mark.parametrize(
    "spec_config, missed",
    [
        (_mtp3(use_dynamic_tree=True), "a dynamic draft tree"),
        (_mtp3(max_total_draft_tokens=6), "draft trees"),
        (_mtp3(speculative_model="/draft"), "a separate speculative checkpoint"),
        (_mtp3(draft_lm_head_dtype="mxfp8"), r"draft_lm_head_dtype=mxfp8 \(only auto\)"),
        (_mtp3(moe_backend="CUTLASS"), "draft MoE backend CUTLASS"),
        (_mtp3(draft_len_schedule={8: 3, 32: 2}), "a draft-length schedule"),
        (MTPDecodingConfig(max_draft_len=3, max_concurrency=16), "a draft-length schedule"),
        (
            _mtp3(acceptance_rate_window_size=8, acceptance_rate_threshold=0.3),
            "the acceptance-rate gate",
        ),
        (_mtp3(use_rejection_sampling=True), "rejection sampling"),
        (_mtp3(use_relaxed_acceptance_for_thinking=True), "relaxed acceptance"),
        (
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused"),
            r"Eagle3DecodingConfig \(only MTP\)",
        ),
        (
            NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            r"NGramDecodingConfig \(only MTP\)",
        ),
    ],
    ids=[
        "dynamic-tree",
        "static-tree",
        "draft-checkpoint",
        "mxfp8-draft-head",
        "draft-moe-backend",
        "draft-length-schedule",
        "max-concurrency",
        "acceptance-rate-gate",
        "rejection-sampling",
        "relaxed-acceptance",
        "eagle3",
        "ngram",
    ],
)
def test_fp8_drafter_options_it_does_not_build_are_declined(spec_config, missed):
    _fp8_declined(_fp8_config(spec_config), _fp8_llm_args, missed)


@pytest.mark.parametrize(
    "gate",
    [dict(acceptance_rate_window_size=8), dict(acceptance_rate_threshold=0.3)],
    ids=["window-only", "threshold-only"],
)
def test_fp8_half_set_acceptance_rate_gate_is_admitted(gate):
    """The executor builds the gate only when both fields are set; one alone changes nothing."""
    llm_args = _fp8_llm_args("require")
    assert _qualified(modeling_v2_resolve(_fp8_config(_mtp3(**gate)), llm_args)) == _FP8


@pytest.mark.parametrize(
    "moe_update, args_update, missed",
    [
        (dict(backend="DEEPGEMM"), {}, r"MoE backend DEEPGEMM \(only AUTO or TRTLLM\)"),
        (dict(backend="CUTLASS"), {}, r"MoE backend CUTLASS \(only AUTO or TRTLLM\)"),
        (dict(load_balancer="eplb.yaml"), {}, r"a MoE load balancer \(eplb\.yaml\)"),
        (dict(max_num_tokens=4096), {}, r"moe_config\.max_num_tokens=4096"),
        ({}, dict(dwdp_config="dwdp"), r"DWDP \(dwdp\)"),
    ],
    ids=["deepgemm", "cutlass", "load-balancer", "moe-chunking", "dwdp"],
)
def test_fp8_moe_options_it_does_not_honour_are_declined(moe_update, args_update, missed):
    def llm_args_of(mode):
        args = _fp8_llm_args(mode)
        moe_config = args.moe_config.model_copy(update=moe_update)
        return args.model_copy(update=dict(args_update, moe_config=moe_config))

    _fp8_declined(_fp8_config(), llm_args_of, missed)


def test_fp8_target_ignores_the_built_in_shared_expert_fusion_flag(monkeypatch):
    """The flag configures the built-in MoE backends, which the target never builds."""
    monkeypatch.setenv("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION", "1")
    assert _qualified(modeling_v2_resolve(_fp8_config(), _fp8_llm_args("require"))) == _FP8


@pytest.mark.parametrize(
    "llm_kwargs, missed",
    [
        (dict(prefill_cuda_graph_backend="breakable"), r"prefill CUDA graphs \(breakable\)"),
        (dict(prefill_cuda_graph_backend="piecewise"), r"prefill CUDA graphs \(piecewise\)"),
        (dict(torch_compile_config={}), "torch.compile"),
        (dict(load_format="dummy"), r"load_format DUMMY \(only AUTO\)"),
        (dict(lm_head_output_dtype="float32"), r"lm_head_output_dtype=float32 \(only auto\)"),
        (dict(allreduce_strategy="NCCL"), r"allreduce_strategy=NCCL \(only AUTO or ONESHOT\)"),
        (dict(kv_cache_config={"dtype": "fp8_ds_mla"}), r"KV cache fp8_ds_mla \(only FP8\)"),
    ],
    ids=[
        "breakable-prefill-graphs",
        "piecewise-prefill-graphs",
        "torch-compile",
        "dummy-weights",
        "fp32-logits",
        "nccl-allreduce",
        "fp8-ds-mla-cache",
    ],
)
def test_fp8_engine_settings_it_does_not_run_are_declined(llm_kwargs, missed):
    def llm_args_of(mode):
        return _fp8_llm_args(mode, **llm_kwargs)

    _fp8_declined(_fp8_config(), llm_args_of, missed)


@pytest.mark.parametrize(
    "config_kwargs, llm_kwargs, missed",
    [
        (
            dict(text_overrides=dict(num_hidden_layers=78)),
            {},
            r"text_config shape \(78, 4096, 288, 1536, 64\) \(only",
        ),
        (dict(kv_cache_quant_algo=None), {}, r"KV cache bf16 \(only FP8\)"),
        (dict(kv_cache_quant_algo=QuantAlgo.NVFP4), {}, r"KV cache NVFP4 \(only FP8\)"),
        (dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1), {}, "moe_tp=1 moe_ep=4"),
        (
            dict(world_size=4, tp_size=4, moe_tp_size=4, enable_attention_dp=True),
            {},
            "attention DP",
        ),
        (dict(world_size=8, tp_size=8, moe_tp_size=8), {}, "world_size=8 tp=8"),
        (
            dict(world_size=4, tp_size=4, moe_tp_size=4, gpus_per_node=2),
            {},
            r"gpus_per_node=2 \(only the 4 ranks of one node\)",
        ),
        (
            dict(text_overrides=dict(num_experts_per_tok=6)),
            {},
            r"text_config geometry .*'num_experts_per_tok': 6",
        ),
        (
            dict(text_overrides=dict(moe_intermediate_size=1024)),
            {},
            r"text_config geometry .*'moe_intermediate_size': 1024",
        ),
        (dict(text_overrides=dict(hc_mult=2)), {}, r"text_config geometry .*'hc_mult': 2"),
        (dict(text_overrides=dict(scoring_func="softmax")), {}, "MoE routing .*softmax"),
        (
            dict(
                text_overrides=dict(
                    linear_attn_config={
                        "num_heads": 64,
                        "head_dim": 128,
                        "short_conv_kernel_size": 4,
                    }
                )
            ),
            {},
            "a KDA gate without gate_lower_bound",
        ),
        (
            dict(
                text_overrides=dict(
                    linear_attn_config={
                        "num_heads": 64,
                        "head_dim": 64,
                        "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0,
                    }
                )
            ),
            {},
            r"KDA .*'head_dim': 64",
        ),
        (
            dict(
                text_overrides=dict(
                    linear_attn_config={
                        "num_heads": 64,
                        "head_dim": 128,
                        "short_conv_kernel_size": 3,
                        "gate_lower_bound": -5.0,
                    }
                )
            ),
            {},
            r"KDA .*'short_conv_kernel_size': 3",
        ),
        (
            dict(text_overrides=dict(tie_word_embeddings=True)),
            {},
            "an LM head tied to the embedding",
        ),
        (dict(spec_config=MTPDecodingConfig(max_draft_len=2)), {}, r"max_draft_len=2 \(only 3\)"),
        (
            dict(spec_config=MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)),
            {},
            "only one-model MTP-Eagle",
        ),
        ({}, dict(disable_mm_encoder=False), "the vision encoder"),
    ],
    ids=[
        "other-shape",
        "bf16-kv",
        "nvfp4-kv",
        "expert-parallel",
        "attention-dp",
        "tp8",
        "two-gpus-per-node",
        "top-k",
        "expert-intermediate",
        "mhc-streams",
        "softmax-routing",
        "unbounded-gate",
        "kda-head-dim",
        "kda-conv-width",
        "tied-embeddings",
        "mtp2",
        "vanilla-mtp",
        "vision-tower",
    ],
)
def test_fp8_near_misses_do_not_match(config_kwargs, llm_kwargs, missed):
    def llm_args_of(mode):
        return _fp8_llm_args(mode, **llm_kwargs)

    _fp8_declined(_fp8_config(**config_kwargs), llm_args_of, missed)


def test_fp8_refusal_names_only_what_failed():
    """A deployment that misses one requirement is told that one, not every fact it has."""
    reason = _fp8_declined(
        _fp8_config(kv_cache_quant_algo=None), _fp8_llm_args, r"KV cache bf16 \(only FP8\)"
    )
    for admitted in ("world_size", "moe_tp", "attention DP", "MoE backend", "vision", "draft"):
        assert admitted not in reason, f"{admitted!r} is admitted but reported: {reason}"


@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, None], ids=["nvfp4", "unquantized"])
def test_other_checkpoints_do_not_reach_the_fp8_target(quant_algo):
    """Another quantization of the same model never lands on the block-FP8 target."""
    config = _model_config(
        _flash_config(), quant_algo=quant_algo, kv_cache_quant_algo=QuantAlgo.FP8, **_TP4
    )
    target = modeling_v2_resolve(config, _fp8_llm_args("auto"))
    assert target is None or _qualified(target) != _FP8
    try:
        target = modeling_v2_resolve(config, _fp8_llm_args("require"))
    except ValueError:
        return
    assert _qualified(target) != _FP8


@pytest.mark.parametrize("sm", [(9, 0), (10, 3), (12, 0)], ids=["sm90", "sm103", "sm120"])
def test_fp8_on_other_gpus_does_not_match(monkeypatch, sm):
    """The target has run on B200 only; a B300 falls back."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    missed = rf"compute capability \({sm[0]}, {sm[1]}\) \(only sm_100\)"
    _fp8_declined(_fp8_config(), _fp8_llm_args, missed)


def test_fp8_constructor_refuses_what_route_declines():
    """A direct construction raises on the facts ``route`` declines, before building anything."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling import (
        ModelingV2Glm53FlashFp8Sm10xTp4,
    )

    config = _fp8_config(world_size=8, tp_size=8, moe_tp_size=8)
    config.disable_mm_encoder = True
    with pytest.raises(ValueError, match="world_size=8 tp=8"):
        ModelingV2Glm53FlashFp8Sm10xTp4(config)
    config = _fp8_config(_mtp3(use_rejection_sampling=True))
    with pytest.raises(ValueError, match="rejection sampling; the vision encoder"):
        ModelingV2Glm53FlashFp8Sm10xTp4(config)


@pytest.mark.parametrize(
    "multimodal_data, refused",
    [
        (None, None),
        ({}, None),
        ({"image": {"pixel_values": torch.zeros(1)}}, "image"),
        ({"video": {"pixel_values_videos": torch.zeros(1)}}, "video"),
    ],
    ids=["text", "empty-payload", "image", "video"],
)
def test_fp8_target_refuses_media_on_admission(multimodal_data, refused):
    """The text-only target fails a media request on admission, as that request's own error."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling import (
        ModelingV2Glm53FlashFp8Sm10xTp4,
    )

    request = SimpleNamespace(py_multimodal_data=multimodal_data)
    if refused is None:
        ModelingV2Glm53FlashFp8Sm10xTp4.validate_request(None, request)
        return
    with pytest.raises(ValueError, match=f"carries {refused} input"):
        ModelingV2Glm53FlashFp8Sm10xTp4.validate_request(None, request)
