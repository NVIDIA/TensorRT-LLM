# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe import nccl_ep_utils
from tensorrt_llm._torch.modules.fused_moe.communication import communication_factory
from tensorrt_llm._torch.modules.fused_moe.communication.allgather_reducescatter import (
    AllGatherReduceScatter,
)
from tensorrt_llm._torch.modules.fused_moe.communication.nccl_ep import NcclEP


def _make_model_config(
    act_dtype: torch.dtype = torch.bfloat16,
    moe_max_num_tokens: int | None = 1024,
):
    mapping = SimpleNamespace(
        enable_attention_dp=True,
        dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_ep_rank=0,
    )
    return SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(hidden_size=4096),
        torch_dtype=act_dtype,
        quant_config=None,
        max_num_tokens=1024,
        moe_max_num_tokens=moe_max_num_tokens,
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
        moe_load_balancer=None,
    )


def _strategy_unavailable(*args, **kwargs):
    raise RuntimeError("strategy unavailable")


def _install_failing_nccl_module(monkeypatch: pytest.MonkeyPatch, error: BaseException):
    def fail_get_version():
        raise error

    monkeypatch.setattr(nccl_ep_utils, "_nccl_ep_installed", None)
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(get_version=fail_get_version))
    monkeypatch.delitem(sys.modules, "nccl.ep", raising=False)


def test_nccl_ep_installed_handles_runtime_probe_failure(monkeypatch: pytest.MonkeyPatch):
    _install_failing_nccl_module(monkeypatch, RuntimeError("missing libnccl_ep"))

    assert nccl_ep_utils.is_nccl_ep_installed() is False
    assert nccl_ep_utils._nccl_ep_installed is False


class _FakeNcclEP:
    def __init__(
        self,
        mapping,
        num_slots,
        hidden_size,
        max_num_tokens,
        moe_max_num_tokens,
        top_k=8,
    ):
        self.mapping = mapping
        self.num_slots = num_slots
        self.hidden_size = hidden_size
        self.max_num_tokens = max_num_tokens
        self.moe_max_num_tokens = moe_max_num_tokens
        self.top_k = top_k


@pytest.mark.parametrize(
    ("act_dtype", "moe_max_num_tokens", "match"),
    [
        (torch.float16, 1024, "act_dtype=torch.bfloat16"),
    ],
)
def test_forced_nccl_ep_validates_preconditions(
    act_dtype: torch.dtype,
    moe_max_num_tokens: int | None,
    match: str,
):
    model_config = _make_model_config(act_dtype, moe_max_num_tokens)

    with pytest.raises(ValueError, match=match):
        communication_factory.CommunicationFactory._create_forced_method(
            "NCCL_EP",
            model_config,
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            payload_in_workspace=False,
            alltoall_result_do_sum=True,
            use_flashinfer=False,
            hidden_size=4096,
        )


def test_forced_nccl_ep_allows_missing_moe_max_num_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = _make_model_config(torch.bfloat16, None)
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory._create_forced_method(
        "NCCL_EP",
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        payload_in_workspace=False,
        alltoall_result_do_sum=True,
        use_flashinfer=False,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)
    assert strategy.max_num_tokens == model_config.max_num_tokens
    assert strategy.moe_max_num_tokens is None


def test_auto_selection_uses_nccl_ep_with_missing_moe_max_num_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = _make_model_config(torch.bfloat16, None)

    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)
    assert strategy.max_num_tokens == model_config.max_num_tokens
    assert strategy.moe_max_num_tokens is None


@pytest.mark.parametrize(
    ("act_dtype", "moe_max_num_tokens"),
    [
        (torch.float16, 1024),
    ],
)
def test_auto_selection_skips_nccl_ep_when_preconditions_fail(
    monkeypatch: pytest.MonkeyPatch,
    act_dtype: torch.dtype,
    moe_max_num_tokens: int | None,
):
    model_config = _make_model_config(act_dtype, moe_max_num_tokens)

    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("NcclEP should not be constructed")

    monkeypatch.setattr(communication_factory, "NcclEP", fail_if_called)

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, AllGatherReduceScatter)


def test_auto_selection_skips_nccl_ep_for_quantized_moe(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = _make_model_config()
    model_config.quant_config = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(has_any_quant=lambda **_: True)
    )
    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    monkeypatch.setattr(
        communication_factory,
        "NcclEP",
        lambda *args, **kwargs: pytest.fail("NcclEP should not be constructed for quantized MoE"),
    )

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, AllGatherReduceScatter)


def test_auto_selection_falls_back_when_nccl_probe_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = _make_model_config(torch.bfloat16, None)
    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    _install_failing_nccl_module(monkeypatch, OSError("missing native NCCL EP library"))

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, AllGatherReduceScatter)


def test_nccl_ep_context_init_rejects_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = object.__new__(NcclEP)
    strategy._ctx = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="context must be initialized before CUDA graph capture"):
        strategy._get_context()


def test_nccl_ep_handle_init_rejects_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = object.__new__(NcclEP)
    strategy._handle = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fail_create_handle(*args, **kwargs):
        raise AssertionError("create_handle should not run during CUDA graph capture")

    ctx = SimpleNamespace(
        ep_group=SimpleNamespace(create_handle=fail_create_handle),
        layout=object(),
    )

    with pytest.raises(
        RuntimeError, match="dispatch handle must be initialized before CUDA graph capture"
    ):
        strategy._setup_handle(ctx, object(), 0)
