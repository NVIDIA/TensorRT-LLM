# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep finalize fusion confined to the supported model execution paths."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE


@pytest.mark.parametrize(
    "tokens,workspace_available,compiling,expect_fused",
    [
        (1, True, False, True),
        (128, True, False, True),
        (129, True, False, False),
        (8, False, False, False),
        (8, True, True, False),
    ],
)
def test_finalize_dispatch_preserves_fallback(
    tokens: int, workspace_available: bool, compiling: bool, expect_fused: bool
) -> None:
    """Exercise the real MoE forward selection and prevent double reduction."""
    hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 8, dtype=torch.int32)
    weights = torch.full((tokens, 8), 1 / 8)
    routed = torch.full_like(hidden, 2)
    finalized = torch.full_like(hidden, 3)
    reduced = torch.full_like(hidden, 4)
    routing = SimpleNamespace(
        apply=Mock(return_value=(indices, weights)), clear_routing_memo=Mock()
    )
    experts = Mock(return_value=(routed, weights, indices))
    experts.routing_method = routing
    experts.comm = None
    workspace = SimpleNamespace(finalize=Mock(return_value=finalized))
    module = SimpleNamespace(
        experts=experts,
        prepare_moe_inputs=Mock(return_value=(weights, hidden)),
        nvtx=lambda _: nullcontext(),
        _finalize_ar_workspace=workspace if workspace_available else None,
        post_expert_norm_weight=torch.zeros(4),
        post_expert_norm_eps=1e-8,
        finalize=Mock(return_value=finalized),
        all_reduce=Mock(return_value=reduced),
        unpooler=Mock(side_effect=lambda value: value),
        _unpool_gather_workspace=None,
        mapping=SimpleNamespace(enable_attention_dp=False, has_tp=lambda: True, moe_tp_size=1),
    )
    module._replicated_projection = MethodType(MuseSparkOpenMoE._replicated_projection, module)
    with patch("torch.compiler.is_compiling", return_value=compiling):
        result = MuseSparkOpenMoE.forward(module, hidden, None)
    if expect_fused:
        workspace.finalize.assert_called_once()
        module.finalize.assert_not_called()
        module.all_reduce.assert_not_called()
        assert result is finalized
    else:
        workspace.finalize.assert_not_called()
        module.finalize.assert_called_once()
        module.all_reduce.assert_called_once()
        assert result is reduced
    routing.clear_routing_memo.assert_called_once()


@pytest.mark.parametrize(
    "tokens,max_tokens,chunked_available,expect",
    [
        (128, 512, True, "finalize"),
        (129, 512, True, "chunked"),
        (512, 512, True, "chunked"),
        (513, 512, True, "unfused"),
        (256, 128, True, "unfused"),
        (256, 512, False, "unfused"),
    ],
)
def test_chunked_finalize_dispatch(
    tokens: int, max_tokens: int, chunked_available: bool, expect: str
) -> None:
    """MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED serves 129..512 tokens in fused 128-token calls."""
    hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 8, dtype=torch.int32)
    weights = torch.full((tokens, 8), 1 / 8)
    routed = torch.full_like(hidden, 2)
    finalized = torch.full_like(hidden, 3)
    reduced = torch.full_like(hidden, 4)
    routing = SimpleNamespace(
        apply=Mock(return_value=(indices, weights)), clear_routing_memo=Mock()
    )
    experts = Mock(return_value=(routed, weights, indices))
    experts.routing_method = routing
    experts.comm = None
    workspace = SimpleNamespace(finalize=Mock(return_value=finalized))
    if chunked_available:
        workspace.finalize_chunked = Mock(return_value=finalized)
    module = SimpleNamespace(
        experts=experts,
        prepare_moe_inputs=Mock(return_value=(weights, hidden)),
        nvtx=lambda _: nullcontext(),
        _finalize_ar_workspace=workspace,
        _finalize_ar_max_tokens=max_tokens,
        post_expert_norm_weight=torch.zeros(4),
        post_expert_norm_eps=1e-8,
        finalize=Mock(return_value=finalized),
        all_reduce=Mock(return_value=reduced),
        unpooler=Mock(side_effect=lambda value: value),
        _unpool_gather_workspace=None,
        mapping=SimpleNamespace(enable_attention_dp=False, has_tp=lambda: True, moe_tp_size=1),
    )
    module._replicated_projection = MethodType(MuseSparkOpenMoE._replicated_projection, module)
    result = MuseSparkOpenMoE.forward(module, hidden, None)
    calls = {
        "finalize": workspace.finalize.call_count,
        "chunked": workspace.finalize_chunked.call_count if chunked_available else 0,
        "unfused": module.finalize.call_count,
    }
    assert calls == {name: int(name == expect) for name in calls}
    if expect == "unfused":
        module.all_reduce.assert_called_once()
        assert result is reduced
    else:
        module.all_reduce.assert_not_called()
        assert result is finalized


@pytest.mark.parametrize(
    "tokens,max_tokens,expect,sparse",
    [
        (129, 512, "chunked", None),
        (512, 512, "chunked", None),
        (513, 512, "unfused", True),
        (129, 128, "unfused", False),
        (159, 128, "unfused", False),
        (160, 128, "unfused", True),
    ],
)
def test_sparse_finalize_reaches_only_unfused_batches_of_160_tokens_or_more(
    monkeypatch, tokens: int, max_tokens: int, expect: str, sparse
) -> None:
    """MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE never takes a batch the fused finalize serves.

    With MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED (``max_tokens`` 512) the
    sparse kernel only sees batches above 512 tokens; without it, unfused
    batches of 129..159 tokens keep the tile kernel.
    """
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    combine_calls = []

    def spy_combine(*args, **kwargs):
        combine_calls.append(kwargs["sparse_slots"])
        return torch.full((kwargs["num_tokens"], kwargs["hidden_size"]), 3.0)

    monkeypatch.setattr(builtin, "post_expert_norm_combine", spy_combine)
    hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 8, dtype=torch.int32)
    weights = torch.full((tokens, 8), 1 / 8)
    routed = torch.full_like(hidden, 2)
    finalized = torch.full_like(hidden, 3)
    routing = SimpleNamespace(
        apply=Mock(return_value=(indices, weights)), clear_routing_memo=Mock()
    )
    experts = Mock(return_value=(routed, weights, indices))
    experts.routing_method = routing
    experts.comm = None
    workspace = SimpleNamespace(
        finalize=Mock(return_value=finalized), finalize_chunked=Mock(return_value=finalized)
    )
    module = SimpleNamespace(
        experts=experts,
        prepare_moe_inputs=Mock(return_value=(weights, hidden)),
        nvtx=lambda _: nullcontext(),
        _finalize_ar_workspace=workspace,
        _finalize_ar_max_tokens=max_tokens,
        _large_finalize_sparse=True,
        _large_finalize_4warps=False,
        expert_hidden_size=4,
        top_k=8,
        post_expert_norm_weight=torch.zeros(4),
        post_expert_norm_eps=1e-8,
        all_reduce=Mock(side_effect=lambda value: value),
        unpooler=Mock(side_effect=lambda value: value),
        _unpool_gather_workspace=None,
        mapping=SimpleNamespace(enable_attention_dp=False, has_tp=lambda: True, moe_tp_size=1),
    )
    module.finalize = MethodType(MuseSparkOpenMoE.finalize, module)
    module._replicated_projection = MethodType(MuseSparkOpenMoE._replicated_projection, module)
    MuseSparkOpenMoE.forward(module, hidden, None)
    assert workspace.finalize_chunked.call_count == int(expect == "chunked")
    assert combine_calls == ([] if sparse is None else [sparse])


@pytest.mark.parametrize("value,enabled", [(None, False), ("0", False), ("1", True), ("on", True)])
def test_large_finalize_sparse_switch_values(monkeypatch, value, enabled) -> None:
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    if value is None:
        monkeypatch.delenv("MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE", raising=False)
    else:
        monkeypatch.setenv("MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE", value)
    assert builtin._large_finalize_sparse_enabled() is enabled


@pytest.mark.parametrize("value,max_tokens", [(None, 128), ("0", 128), ("1", 512), ("on", 512)])
def test_chunked_finalize_switch_values(monkeypatch, value, max_tokens) -> None:
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    if value is None:
        monkeypatch.delenv("MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED", raising=False)
    else:
        monkeypatch.setenv("MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED", value)
    assert builtin._fused_finalize_ar_chunked_enabled() is (max_tokens == 512)
    assert builtin._FINALIZE_AR_CHUNKED_MAX_TOKENS == 512
