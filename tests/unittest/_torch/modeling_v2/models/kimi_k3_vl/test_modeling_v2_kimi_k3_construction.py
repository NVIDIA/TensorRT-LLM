# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' construction checks, host-side: each target requires its own parallel layout, and
``tp16_moetp16ep1`` also an explicit expert split and no speculative decoding."""

import inspect
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.models import modeling_kimi_linear as shared
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.quantization.mode import QuantAlgo

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        modeling as route_a,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (
        spec_worker as spec_a,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        modeling as route_b,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (
        spec_worker as spec_b,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)


def _config(moe_tp, moe_ep, split_set=True, spec_config=None, attention_dp=False):
    """The ModelConfig fields the construction checks read."""
    mapping = types.SimpleNamespace(
        world_size=16,
        tp_size=16,
        pp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        enable_attention_dp=attention_dp,
        moe_tp_ep_user_specified=split_set,
    )
    return types.SimpleNamespace(
        mapping=mapping,
        spec_config=spec_config,
        torch_dtype=torch.bfloat16,
        # The MXFP4 checkpoint's quantization, as the model config reads it.
        quant_config=types.SimpleNamespace(
            quant_algo=QuantAlgo.W4A16_MXFP4, kv_cache_quant_algo=None
        ),
        quant_config_dict=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
    )


@pytest.fixture(autouse=True)
def sm_100(monkeypatch):
    """The targets' architecture, on any host."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))


@pytest.mark.parametrize(
    "target,parallel,split,other",
    [
        (route_a, "tp16_moetp4ep4", (4, 4), (16, 1)),
        (route_b, "tp16_moetp16ep1", (16, 1), (4, 4)),
    ],
    ids=["tp16_moetp4ep4", "tp16_moetp16ep1"],
)
def test_each_target_requires_its_layout(target, parallel, split, other):
    target._check_construction(_config(*split))
    with pytest.raises(AssertionError, match=f"the {parallel} target needs"):
        target._check_construction(_config(*other))
    with pytest.raises(AssertionError, match="enable_attention_dp false"):
        target._check_construction(_config(*split, attention_dp=True))


@pytest.mark.parametrize("target", [route_a, route_b], ids=["tp16_moetp4ep4", "tp16_moetp16ep1"])
def test_expert_ckpt_spec_meets_the_shared_loader_contract(target):
    spec = target._k3_expert_ckpt_spec(QuantAlgo.W4A8_MXFP4_MXFP8)
    assert set(shared._K3ExpertCkptSpec._fields) <= set(spec._fields)
    assert tuple(spec.optional_suffixes) == ()

    loaded = {}
    backend = types.SimpleNamespace(
        quant_method=types.SimpleNamespace(
            load_packed_mxfp4_expert=lambda backend, **kwargs: loaded.update(kwargs)
        )
    )
    spec.loader(backend, "experts", 3, 0, lambda key: key, lambda key: None)
    assert loaded["global_expert_id"] == 3
    assert loaded["w1_weight"] == "experts.3.w1.weight_packed"


@pytest.mark.parametrize("target", [spec_a, spec_b], ids=["tp16_moetp4ep4", "tp16_moetp16ep1"])
def test_dspark_markov_bias_meets_the_shared_worker_contract(target, monkeypatch):
    worker_cls = target.KimiK3DSparkWorker
    assert inspect.signature(worker_cls._apply_dspark_markov_bias) == inspect.signature(
        DSparkWorker._apply_dspark_markov_bias
    )
    calls = []
    monkeypatch.setattr(
        DSparkWorker,
        "_apply_dspark_markov_bias",
        lambda self, *args: calls.append(args) or "shared",
    )
    worker = worker_cls.__new__(worker_cls)
    worker._k3_sharded_block_logits = False
    draft_model = types.SimpleNamespace(has_markov_head=True, markov_w2=torch.empty(8, 2))
    logits = torch.zeros(1, 2, 8)
    prev = torch.zeros(1, dtype=torch.long)
    assert worker._refine_block_logits(draft_model, logits, {"first_prev_tokens": prev}, None) == (
        "shared"
    )
    assert calls == [(draft_model, logits, prev)]


def test_tp16_moetp16ep1_requires_the_split_set_explicitly():
    with pytest.raises(AssertionError, match="set explicitly"):
        route_b._check_construction(_config(16, 1, split_set=False))


def test_tp16_moetp16ep1_decodes_without_speculation():
    with pytest.raises(
        AssertionError, match="moe_tensor_parallel_size 4 and moe_expert_parallel_size 4"
    ):
        route_b._check_construction(_config(16, 1, spec_config=object()))


def test_tp16_moetp16ep1_keeps_no_per_token_kda_states(monkeypatch):
    """The text model answers ``kda_draft_records`` (False, without speculation), which a captured step's latent-push
    decision reads: a pure decode step on the decode kernels pushes, as on ``tp16_moetp4ep4`` without per-token
    states."""
    model = route_b.KimiLinearModel.__new__(route_b.KimiLinearModel)
    assert model.kda_draft_records is False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(route_b, "is_in_breakable_cuda_graph", lambda: False)
    kda = types.SimpleNamespace(
        is_kda=True,
        linear_attn=types.SimpleNamespace(takes_k3_kernels=True, k3_buffers=object()),
    )
    mla = types.SimpleNamespace(
        is_kda=False,
        self_attn=types.SimpleNamespace(will_run_decode_branch=lambda metadata, step: True),
    )
    object.__setattr__(model, "layers", [kda, mla])
    assert model._latent_push(None, route_b.DecodeStep(1, 1, 1))
    assert not model._latent_push(None, route_b.DecodeStep(8, 1, 8))
