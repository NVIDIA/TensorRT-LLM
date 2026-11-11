# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's construction-time checks, host-side: its own parallel layout, an explicit expert
split and no speculative decoding, so no per-token KDA states; its routed experts' checkpoint spec meets the shared
loader and its DSpark worker's markov bias the shared worker."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_construction import (  # noqa: E402, F401
    check_dspark_markov_bias_meets_the_shared_worker_contract,
    check_expert_ckpt_spec_meets_the_shared_loader_contract,
    config,
    sm_100,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        modeling as target,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        spec_worker,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.usefixtures("sm_100")


def test_requires_its_layout():
    target._check_construction(config(16, 1))
    with pytest.raises(AssertionError, match="the tp16_moetp16ep1 target needs"):
        target._check_construction(config(4, 4))
    with pytest.raises(AssertionError, match="enable_attention_dp false"):
        target._check_construction(config(16, 1, attention_dp=True))


def test_requires_the_split_set_explicitly():
    with pytest.raises(AssertionError, match="set explicitly"):
        target._check_construction(config(16, 1, split_set=False))


def test_decodes_without_speculation():
    with pytest.raises(
        AssertionError, match="moe_tensor_parallel_size 4 and moe_expert_parallel_size 4"
    ):
        target._check_construction(config(16, 1, spec_config=object()))


def test_expert_ckpt_spec_meets_the_shared_loader_contract():
    check_expert_ckpt_spec_meets_the_shared_loader_contract(target)


def test_dspark_markov_bias_meets_the_shared_worker_contract(monkeypatch):
    check_dspark_markov_bias_meets_the_shared_worker_contract(spec_worker, monkeypatch)


def test_keeps_no_per_token_kda_states(monkeypatch):
    """The text model answers ``kda_draft_records`` (False, without speculation), which a captured step's latent-push
    decision reads: a pure decode step on the decode kernels pushes, as on ``tp16_moetp4ep4`` without per-token
    states."""
    model = target.KimiLinearModel.__new__(target.KimiLinearModel)
    assert model.kda_draft_records is False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(target, "is_in_breakable_cuda_graph", lambda: False)
    kda = types.SimpleNamespace(
        is_kda=True,
        linear_attn=types.SimpleNamespace(takes_k3_kernels=True, k3_buffers=object()),
    )
    mla = types.SimpleNamespace(
        is_kda=False,
        self_attn=types.SimpleNamespace(will_run_decode_branch=lambda metadata, step: True),
    )
    object.__setattr__(model, "layers", [kda, mla])
    assert model._latent_push(None, target.DecodeStep(1, 1, 1))
    assert not model._latent_push(None, target.DecodeStep(8, 1, 8))
