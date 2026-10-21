# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' construction-time checks, host-side, each run by the target's test directory on its own
modules: the ModelConfig fields the construction checks read, the targets' architecture on any host, the routed
experts' checkpoint spec against the shared loader and the DSpark worker's markov bias against the shared worker."""

import inspect
import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_kimi_k3_loader import _K3ExpertCkptSpec
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.quantization.mode import QuantAlgo


def config(moe_tp, moe_ep, split_set=True, spec_config=None, attention_dp=False):
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


@pytest.fixture
def sm_100(monkeypatch):
    """The targets' architecture, on any host."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))


def check_expert_ckpt_spec_meets_the_shared_loader_contract(modeling):
    """The routed experts' checkpoint spec of a target's ``modeling`` module against the shared loader."""
    spec = modeling._k3_expert_ckpt_spec(QuantAlgo.W4A8_MXFP4_MXFP8)
    assert set(_K3ExpertCkptSpec._fields) <= set(spec._fields)
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


def check_dspark_markov_bias_meets_the_shared_worker_contract(spec_worker, monkeypatch):
    """The DSpark worker of a target's ``spec_worker`` module applies the markov bias through the shared worker."""
    worker_cls = spec_worker.KimiK3DSparkWorker
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
