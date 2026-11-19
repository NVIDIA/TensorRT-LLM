# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Inkling 2 route: the gated deployment is admitted, and each other one is
refused with the condition it lacks."""

import pytest

from tensorrt_llm._torch.configs import Inkling2Config
from tensorrt_llm._torch.model_config import ModelConfig, _mirror_text_subconfig_attrs
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.models.inkling2 import routing
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

pytestmark = pytest.mark.cpu_only

_GB300 = (10, 3)
_TEP4 = dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4)
_TARGET = "ModelingV2Inkling2SmallNvfp4Sm103Tep4"
#: The settings the target was gated on.
_GATED = dict(
    max_seq_len=32768,
    cuda_graph_config=None,
    disable_overlap_scheduler=True,
    enable_chunked_prefill=False,
    kv_cache_config=KvCacheConfig(enable_block_reuse=False),
)


def _pretrained(**text_overrides) -> Inkling2Config:
    """The released config, mirrored onto the top level as `ModelConfig.from_pretrained` does."""
    config = Inkling2Config(text_config=text_overrides)
    _mirror_text_subconfig_attrs(config)
    return config


def _model_config(pretrained=None, spec_config=None, **mapping) -> ModelConfig:
    return ModelConfig(
        pretrained_config=pretrained if pretrained is not None else _pretrained(),
        mapping=Mapping(**{**_TEP4, **mapping}),
        spec_config=spec_config,
        quant_config=QuantConfig(),
    )


def _route(model_config=None, capability=_GB300, **llm_args):
    args = TorchLlmArgs(model="/tmp/dummy_model", **{**_GATED, **llm_args})
    return routing.route(model_config or _model_config(), args, capability)


def _refusal(**kwargs) -> str:
    with pytest.raises(NoTarget) as refused:
        _route(**kwargs)
    return str(refused.value)


@pytest.mark.parametrize("max_seq_len", [1, routing.MAX_SEQ_LEN])
def test_admits_the_gated_deployment(max_seq_len):
    assert _route(max_seq_len=max_seq_len).__name__ == _TARGET


@pytest.mark.parametrize(
    "kwargs, condition",
    [
        (dict(max_seq_len=None), "an explicit max_seq_len"),
        (dict(max_seq_len=0), "an explicit max_seq_len"),
        (dict(max_seq_len=routing.MAX_SEQ_LEN + 1), "an explicit max_seq_len"),
        (dict(capability=(10, 0)), "GB300 (sm_103)"),
        (
            dict(model_config=_model_config(world_size=2, tp_size=2, moe_ep_size=2)),
            "attention TP4 with MoE EP4 on 4 ranks",
        ),
        (
            dict(model_config=_model_config(enable_attention_dp=True)),
            "no pipeline, context or attention-data parallelism",
        ),
        (
            dict(model_config=_model_config(spec_config=MTPDecodingConfig(max_draft_len=1))),
            "no speculative decoding",
        ),
        (dict(kv_cache_config=KvCacheConfig(enable_block_reuse=True)), "no KV cache block reuse"),
        (
            dict(
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False, enable_swa_scratch_reuse=True
                )
            ),
            "no SWA scratch reuse",
        ),
        (
            dict(kv_cache_config=KvCacheConfig(enable_block_reuse=False, dtype="fp8")),
            "a BF16 KV cache",
        ),
        (
            dict(
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False, use_kv_cache_manager_v2=False
                )
            ),
            "KV cache manager V2",
        ),
        (dict(enable_chunked_prefill=True), "no chunked prefill"),
        (dict(cuda_graph_config=CudaGraphConfig()), "no CUDA graphs"),
        (dict(disable_overlap_scheduler=False), "the overlap scheduler off"),
    ],
    ids=[
        "max_seq_len_unset",
        "max_seq_len_zero",
        "max_seq_len_too_long",
        "sm_100",
        "tp2_ep2",
        "attention_dp",
        "speculative_decoding",
        "block_reuse",
        "swa_scratch_reuse",
        "fp8_kv_cache",
        "kv_cache_manager_v1",
        "chunked_prefill",
        "cuda_graphs",
        "overlap_scheduler",
    ],
)
def test_refuses_each_other_deployment_naming_the_condition(kwargs, condition):
    message = _refusal(**kwargs)
    assert f"lacks: {condition}" in message or f"; {condition}" in message, message


def test_refuses_an_explicit_kv_cache_window_smaller_than_the_models():
    # An explicit max_attention_window replaces the windows the KV cache derives from layer_types.
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_attention_window=[512])
    assert "attention windows" in _refusal(kv_cache_config=kv_cache_config)


def test_admits_an_explicit_kv_cache_window_equal_to_the_models():
    # Every sixth layer full context (a window of max_seq_len), the others 1024: the release's.
    windows = [1024] * 5 + [_GATED["max_seq_len"]]
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_attention_window=windows)
    assert _route(kv_cache_config=kv_cache_config).__name__ == _TARGET


def test_refuses_a_kv_cache_window_the_model_does_not_apply():
    # Only the copy the KV cache reads changes; the model keeps the release's window.
    pretrained = _pretrained()
    pretrained.sliding_window = 512
    assert "attention windows" in _refusal(model_config=_model_config(pretrained))


def test_refuses_a_window_the_kv_cache_does_not_keep():
    # A field changed after loading (model_kwargs) reaches the text config, which the model
    # reads, but not the copy on the top-level config, which sizes the KV cache's windows.
    pretrained = _pretrained()
    pretrained.text_config.sliding_window_size = 2048
    assert "attention windows" in _refusal(model_config=_model_config(pretrained))


@pytest.mark.parametrize(
    "text_overrides",
    [dict(sliding_window_size=2048), dict(local_layer_ids=list(range(47)))],
    ids=["another_window", "another_layer_pattern"],
)
def test_refuses_windows_other_than_the_release(text_overrides):
    pretrained = _pretrained(**text_overrides)
    assert "attention windows" in _refusal(model_config=_model_config(pretrained))
