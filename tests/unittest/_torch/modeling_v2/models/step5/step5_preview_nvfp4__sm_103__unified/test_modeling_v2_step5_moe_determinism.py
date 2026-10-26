# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Step-5 target's routed experts run deterministically when the engine asks for it.

``fused_moe``'s FC2 + finalize fusion adds each token's top-8 expert outputs with
atomics, so two identical runs can disagree in the last bit and, through a near-tie,
in the tokens. The engine asks for an ordered reduction in two ways:
``FORCE_DETERMINISTIC=1`` in the environment, or ``moe_config.disable_finalize_fusion``
(``ModelConfig.moe_disable_finalize_fusion``). Under either the target's ``_experts``
passes ``use_fused_finalize=False`` to ``fused_moe``; otherwise ``True``.

The switch is read once, when the model is built, so the forward holds no host branch
on it: changing the environment afterwards does not change what the built model passes.

All on the CPU: the shell is built on the meta device from a synthetic config of the
checkpoint's shape, and ``_experts`` runs with its router, top-k and ``fused_moe``
entries replaced by recorders.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.configs import Step5Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DECODER_LAYERS = 92
_MTP_LAYERS = 3
_HIDDEN = 4096
_TOPK = 8


def _step5_config() -> Step5Config:
    """A composite of the checkpoint's shape, as the loader builds it."""
    total = _DECODER_LAYERS + _MTP_LAYERS
    kinds = ["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(total)]
    text = dict(
        architectures=["Step5ForCausalLM"],
        layer_types=kinds,
        rope_theta=[1e7 if kind == "full_attention" else 1e4 for kind in kinds],
        partial_rotary_factors=[1 / 3 if kind == "full_attention" else 1.0 for kind in kinds],
        rope_scaling={"rope_type": "llama3", "factor": 1.0},
        yarn_only_types=["full_attention"],
        moe_layer_list=list(range(3, 91)),
        swiglu_limits=[7.0] * total,
        swiglu_limits_shared=[7.0] * total,
        num_nextn_predict_layers=_MTP_LAYERS,
        sparse_config=dict(
            enabled=True,
            proxy_dim=256,
            sparse_indexer_rope_dim=32,
            sparse_indexer_num_heads=16,
            sparse_indexer_num_k_heads=1,
            num_provider_groups=4,
            topk=512,
            region_block_size=8,
            compression_method="csa_block_compress",
            attention_impl="sparse_gqa",
            apply_to_layer_types=["full_attention"],
        ),
        torch_dtype="bfloat16",
    )
    config = Step5Config(
        architectures=["Step5ForConditionalGeneration"],
        text_config=text,
        vision_config=dict(image_size=728, patch_size=14, width=1536, layers=47, heads=16),
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    config.torch_dtype = torch.bfloat16
    return config


def _core(**model_config_kwargs) -> M.Step5PreviewModelingV2:
    """The core of a shell built on the meta device, at one rank (the switch does not
    depend on the layout)."""
    model_config = ModelConfig(
        pretrained_config=_step5_config(),
        mapping=Mapping(world_size=1, tp_size=1),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=None),
        **model_config_kwargs,
    )
    with torch.device("meta"):
        shell = M.ModelingV2Step5PreviewNvfp4Sm103Unified(model_config)
    return shell.model


@pytest.fixture
def fused_moe_calls(monkeypatch) -> list[dict]:
    """Replace the entries `_experts` calls with recorders; every `fused_moe` call's
    keyword arguments land in the returned list."""
    calls: list[dict] = []

    def cublas_mm(a, b, bias=None, out_dtype=None):
        return torch.zeros(a.shape[0], 352, dtype=out_dtype or a.dtype)

    def noaux_tc_op(logits, bias, n_group, topk_group, topk, scale):
        rows = logits.shape[0]
        return (
            torch.full((rows, topk), 1.0 / topk, dtype=torch.float32),
            torch.zeros(rows, topk, dtype=torch.int32),
        )

    def fused_moe(x, ids, weights, *args, **kwargs):
        calls.append(kwargs)
        return [torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype)]

    monkeypatch.setattr(M, "cublas_mm", cublas_mm)
    monkeypatch.setattr(M, "noaux_tc_op", noaux_tc_op)
    monkeypatch.setattr(M, "fused_moe", fused_moe)
    return calls


def _layer() -> SimpleNamespace:
    """The fields of `_Layer` `_experts` reads, for a bf16 expert layer (no input
    quantization)."""
    return SimpleNamespace(
        router=None,
        router_bias=None,
        gs_in=None,
        fc1=None,
        fc2=None,
        quant_scales=[],
        expert_limits=None,
    )


def _run_experts(core, rows: int = 3) -> None:
    x = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16)
    out = M._experts(core, _layer(), x)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "env, model_config_kwargs, fused",
    [
        pytest.param(None, {}, True, id="default"),
        pytest.param("0", {}, True, id="force_deterministic_0"),
        pytest.param("1", {}, False, id="force_deterministic_1"),
        pytest.param(None, dict(moe_disable_finalize_fusion=True), False, id="disable_fusion"),
        pytest.param("1", dict(moe_disable_finalize_fusion=True), False, id="both"),
    ],
)
def test_the_switch_reaches_fused_moe(
    monkeypatch, fused_moe_calls, env, model_config_kwargs, fused
):
    if env is None:
        monkeypatch.delenv("FORCE_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("FORCE_DETERMINISTIC", env)
    core = _core(**model_config_kwargs)
    assert core.topk == _TOPK
    _run_experts(core)
    assert len(fused_moe_calls) == 1
    assert fused_moe_calls[0]["use_fused_finalize"] is fused


def test_every_chunk_carries_the_switch(monkeypatch, fused_moe_calls):
    """A call over more rows than one `fused_moe` call takes is split; each chunk runs
    with the same setting."""
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    core = _core()
    _run_experts(core, rows=M._MOE_MAX_T + 1)
    assert [call["use_fused_finalize"] for call in fused_moe_calls] == [False, False]


@pytest.mark.parametrize("built_with, flipped_to", [(None, "1"), ("1", "0")])
def test_the_switch_is_read_when_the_model_is_built(
    monkeypatch, fused_moe_calls, built_with, flipped_to
):
    """The forward passes what the model was built with, whatever the environment says
    by the time it runs."""
    if built_with is None:
        monkeypatch.delenv("FORCE_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("FORCE_DETERMINISTIC", built_with)
    core = _core()
    built = core.moe_fused_finalize
    monkeypatch.setenv("FORCE_DETERMINISTIC", flipped_to)
    _run_experts(core)
    assert fused_moe_calls[0]["use_fused_finalize"] is built
    assert built is (built_with != "1")
