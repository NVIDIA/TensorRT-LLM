# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-path plumbing tests for routed-expert MoE LoRA.

The op-level tests exercise the fused_moe op and _extract_moe_lora_tensors in
isolation, so they cannot catch a regression where lora_params never reaches
the routed-expert call in the real model.forward to self.experts to
CutlassFusedMoE.run_moe path.

These CPU-only tests (no GPU or built C++ op required) assert, at each hop,
that a non-empty lora_params is forwarded:

  1. QwenMoE.forward to the routed self.experts call (legacy wrapper).
  2. ConfigurableMoE.forward_impl to scheduler.forward (the default
     ENABLE_CONFIGURABLE_MOE=1 path).
  3. ExternalCommMoEScheduler._get_backend_kwargs to the CutlassFusedMoE
     run_moe kwargs, and not to backends that cannot carry LoRA.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen_moe import QwenMoE
from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType

# A unique sentinel so the assertions can verify object identity rather than
# mere truthiness; any drop/replace along the chain fails the identity check.
_LORA_PARAMS_SENTINEL = {"num_seqs": 1, "_marker": object()}


def test_qwen_moe_forward_passes_lora_params_to_routed_experts():
    """QwenMoE.forward must forward lora_params to the routed self.experts
    call, not only to the shared expert."""
    num_tokens, hidden_dim = 4, 8

    hidden_states = torch.randn(num_tokens, hidden_dim)
    router_logits = torch.randn(num_tokens, 2)
    expert_out = torch.randn(num_tokens, hidden_dim)
    shared_out = torch.randn(num_tokens, hidden_dim)

    experts = MagicMock(return_value=expert_out)

    # Call the unbound method against a lightweight stand-in so we do not need
    # to construct real weights or CUDA streams.
    fake_self = SimpleNamespace(
        hidden_dim=hidden_dim,
        gate=MagicMock(return_value=router_logits),
        experts=experts,
        shared_expert=MagicMock(return_value=shared_out),
        shared_expert_gate=MagicMock(return_value=torch.zeros(num_tokens, 1)),
    )
    attn_metadata = SimpleNamespace(all_rank_num_tokens=[num_tokens])

    QwenMoE.forward(
        fake_self,
        hidden_states,
        attn_metadata,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    experts.assert_called_once()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL, (
        "QwenMoE.forward dropped lora_params on the routed-expert call; "
        "routed-expert MoE LoRA would be silently disabled."
    )


def test_configurable_moe_forward_impl_forwards_lora_params_to_scheduler():
    """ConfigurableMoE.forward_impl must forward lora_params to the scheduler
    so routed-expert MoE LoRA is not dropped on the default
    ENABLE_CONFIGURABLE_MOE=1 path."""
    x = torch.randn(4, 8)
    router_logits = torch.randn(4, 2)

    scheduler = MagicMock()
    scheduler.forward = MagicMock(return_value=torch.zeros_like(x))

    fake_self = SimpleNamespace(
        scheduler=scheduler,
        enable_dwdp=False,
        repeat_idx=0,
        repeat_count=1,
    )

    ConfigurableMoE.forward_impl(
        fake_self,
        x,
        router_logits,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    scheduler.forward.assert_called_once()
    assert scheduler.forward.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL, (
        "ConfigurableMoE.forward_impl dropped lora_params before the scheduler; "
        "routed-expert MoE LoRA would be silently disabled on the default path."
    )


def _make_external_comm_scheduler(backend_cls):
    """Build an ExternalCommMoEScheduler whose moe.backend is an uninitialized
    instance of backend_cls, sufficient for _get_backend_kwargs class dispatch
    without constructing weights."""
    backend = backend_cls.__new__(backend_cls)
    moe = SimpleNamespace(
        backend=backend,
        comm=None,
        enable_alltoall=False,
        mapping=SimpleNamespace(tp_size=1),
        routing_method=SimpleNamespace(top_k=2),
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = moe
    return scheduler


def test_scheduler_threads_lora_params_to_cutlass_run_moe_kwargs():
    """_get_backend_kwargs must thread lora_params into the
    CutlassFusedMoE.run_moe kwargs."""
    scheduler = _make_external_comm_scheduler(CutlassFusedMoE)

    kwargs = scheduler._get_backend_kwargs(
        output_dtype=torch.bfloat16,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL, (
        "Scheduler dropped lora_params before CutlassFusedMoE.run_moe; "
        "routed-expert MoE LoRA would be silently disabled."
    )


def test_scheduler_does_not_thread_lora_params_to_non_cutlass_backend():
    """Only CutlassFusedMoE.run_moe accepts lora_params. Other backends must
    not receive it, since it is not in their run_moe signature."""
    scheduler = _make_external_comm_scheduler(DeepGemmFusedMoE)

    kwargs = scheduler._get_backend_kwargs(
        output_dtype=torch.bfloat16,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert "lora_params" not in kwargs, (
        "lora_params must only be forwarded to CutlassFusedMoE.run_moe; "
        f"DeepGemmFusedMoE.run_moe does not accept it. Got kwargs: {list(kwargs)}"
    )


def _moe_lora_params_for_layer(layer_idx):
    """A minimal lora_params carrying a routed-expert MoE module (moe_h_to_4h)
    for layer_idx, enough for _moe_lora_active."""
    module_id = int(LoraModuleType.from_string("moe_h_to_4h"))
    return {
        "num_seqs": 1,
        layer_idx: {module_id: {"adapter_size": None, "weight_pointers": None}},
    }


def test_cutlass_moe_lora_active_detects_layer_modules():
    """_moe_lora_active is the predicate the multi-chunk guard relies on: True
    only when this layer has a routed-expert MoE LoRA module."""
    backend = CutlassFusedMoE.__new__(CutlassFusedMoE)
    backend.layer_idx = 3

    assert backend._moe_lora_active(_moe_lora_params_for_layer(3)) is True
    # No MoE modules for this layer, or empty params, means inactive.
    assert backend._moe_lora_active(_moe_lora_params_for_layer(5)) is False
    assert backend._moe_lora_active(None) is False
    assert backend._moe_lora_active({"num_seqs": 1}) is False


def test_scheduler_rejects_multichunk_with_moe_lora():
    """The ConfigurableMoE scheduler must reject multi-chunk execution when
    routed-expert MoE LoRA is active, with an actionable message rather than a
    deep C++ kernel failure."""
    backend = CutlassFusedMoE.__new__(CutlassFusedMoE)
    backend.layer_idx = 0

    moe = SimpleNamespace(
        backend=backend,
        calculate_num_chunks=lambda *_args, **_kw: 2,
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = moe

    x = torch.randn(8, 4)
    router_logits = torch.randn(8, 2)

    with pytest.raises(NotImplementedError, match="multi-chunk"):
        scheduler.forward(
            x,
            router_logits,
            do_finalize=True,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=None,
            use_dp_padding=False,
            lora_params=_moe_lora_params_for_layer(0),
        )
