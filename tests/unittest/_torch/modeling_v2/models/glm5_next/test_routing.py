# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = "models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling.ModelingV2Glm53FlashFp8Sm10xTp4"
_TP4 = dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, gpus_per_node=8)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a B200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    kwargs.setdefault("disable_mm_encoder", True)
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _glm53_config(**text_overrides):
    """The composite config the GLM-5.3-Flash release declares (text half only)."""
    text = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        hidden_size=4096,
        n_routed_experts=288,
        q_lora_rank=1536,
    )
    text.update(text_overrides)
    config = PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"], model_type="glm5_next"
    )
    config.text_config = PretrainedConfig(**text)
    return config


def _model_config(
    pretrained_config,
    spec_config=None,
    quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
    kv_cache_quant_algo=QuantAlgo.FP8,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
    )


def _mtp3(**fields) -> MTPDecodingConfig:
    """MTP-Eagle at three drafts with ``fields`` set as given, bypassing the validators that would
    reject some of them on their own (a draft checkpoint path that does not exist, a tree without
    its sizes): routing reads the fields, not their consistency."""
    return MTPDecodingConfig(max_draft_len=3).model_copy(update=fields)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("spec_config", [None, _mtp3()], ids=["no-spec", "mtp3"])
@pytest.mark.parametrize("moe_backend", ["AUTO", "TRTLLM"])
@pytest.mark.parametrize("sm", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
def test_fp8_tp4_matches(monkeypatch, sm, moe_backend, spec_config, mode):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(_glm53_config(), spec_config=spec_config, **_TP4)
    llm_args = _llm_args(mode, moe_config={"backend": moe_backend})
    assert _qualified(modeling_v2_resolve(config, llm_args)) == _TARGET


def _declined(config, llm_args_of, missed) -> None:
    """``auto`` falls back to the built-in model and ``require`` raises with ``missed`` in the reason."""
    assert modeling_v2_resolve(config, llm_args_of("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, llm_args_of("require"))


@pytest.mark.parametrize(
    "spec_config, missed",
    [
        (_mtp3(use_dynamic_tree=True), "use_dynamic_tree=True"),
        (_mtp3(speculative_model="/draft"), "speculative_model=/draft"),
        (_mtp3(draft_lm_head_dtype="mxfp8"), "draft_lm_head_dtype=mxfp8"),
        (_mtp3(moe_backend="CUTLASS"), "draft moe_backend=CUTLASS"),
    ],
    ids=["dynamic-tree", "draft-checkpoint", "mxfp8-draft-head", "draft-moe-backend"],
)
def test_drafter_options_it_does_not_build_are_declined(spec_config, missed):
    config = _model_config(_glm53_config(), spec_config=spec_config)
    _declined(config, _llm_args, missed)


@pytest.mark.parametrize(
    "moe_update, args_update, missed",
    [
        (dict(backend="DEEPGEMM"), {}, "moe backend=DEEPGEMM"),
        (dict(backend="CUTLASS"), {}, "moe backend=CUTLASS"),
        (dict(load_balancer="eplb.yaml"), {}, "load_balancer=eplb.yaml"),
        ({}, dict(dwdp_config="dwdp"), "dwdp_config=dwdp"),
    ],
    ids=["deepgemm", "cutlass", "load-balancer", "dwdp"],
)
def test_moe_options_it_does_not_honour_are_declined(moe_update, args_update, missed):
    def llm_args_of(mode):
        args = _llm_args(mode)
        moe_config = args.moe_config.model_copy(update=moe_update)
        return args.model_copy(update=dict(args_update, moe_config=moe_config))

    _declined(_model_config(_glm53_config()), llm_args_of, missed)


def test_shared_expert_fusion_is_declined(monkeypatch):
    monkeypatch.setenv("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION", "1")
    _declined(_model_config(_glm53_config()), _llm_args, "TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1")


@pytest.mark.parametrize(
    "config_kwargs, model_kwargs, llm_kwargs, missed",
    [
        (dict(num_hidden_layers=78), {}, {}, r"shape \(78, 4096, 288, 1536\)"),
        (dict(), dict(kv_cache_quant_algo=None), {}, "KV cache bf16"),
        (dict(), dict(kv_cache_quant_algo=QuantAlgo.NVFP4), {}, r"KV cache (QuantAlgo\.)?NVFP4"),
        (dict(), dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1), {}, "moe_ep=4"),
        (
            dict(),
            dict(world_size=4, tp_size=4, enable_attention_dp=True, moe_tp_size=4),
            {},
            "attention_dp=True",
        ),
        (dict(), dict(world_size=8, tp_size=8, moe_tp_size=8), {}, "world_size=8"),
        (dict(), dict(spec_config=MTPDecodingConfig(max_draft_len=2)), {}, "max_draft_len=2"),
        (
            dict(),
            dict(spec_config=MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)),
            {},
            "MTP",
        ),
        (dict(), {}, dict(disable_mm_encoder=False), "disable_mm_encoder=False"),
    ],
    ids=[
        "other-shape",
        "bf16-kv",
        "nvfp4-kv",
        "expert-parallel",
        "attention-dp",
        "tp8",
        "mtp2",
        "vanilla-mtp",
        "vision-tower",
    ],
)
def test_near_misses_do_not_match(config_kwargs, model_kwargs, llm_kwargs, missed):
    config = _model_config(_glm53_config(**config_kwargs), **model_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **llm_kwargs))


@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, None], ids=["nvfp4", "unquantized"])
def test_other_checkpoints_do_not_reach_the_fp8_target(quant_algo):
    """Another quantization of the same model never lands on the block-FP8 target.

    Whether another glm5_next target takes it is that target's routing test's concern.
    """
    config = _model_config(_glm53_config(), quant_algo=quant_algo)
    for mode in ("auto", "require"):
        try:
            target = modeling_v2_resolve(config, _llm_args(mode))
        except ValueError:
            continue
        assert target is None or _qualified(target) != _TARGET


@pytest.mark.parametrize("sm", [(9, 0), (12, 0)], ids=["sm90", "sm120"])
def test_other_gpus_do_not_match(monkeypatch, sm):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(_glm53_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
