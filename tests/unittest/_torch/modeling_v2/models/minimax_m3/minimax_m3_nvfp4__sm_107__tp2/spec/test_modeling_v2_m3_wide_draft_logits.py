# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W2b (SpecTail, wide target): the in-tree draft loop drafts from the drafter head's local vocab shard."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.modeling import (  # noqa: E501
    M3Eagle3Worker,
    _M3ShardDraftLogits,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode

pytestmark = pytest.mark.cpu_only


class _Head(nn.Module):
    tp_mode = TensorParallelMode.COLUMN
    padding_size = 0
    tp_size = 2
    bias = None

    def apply_linear(self, x, bias):
        assert bias is None
        return x + 1  # stands for this rank's shard GEMM


class _Original(nn.Module):
    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        return hidden_states * 0 - 5


def test_the_shard_processor_returns_the_local_shard_for_selected_rows() -> None:
    proc = _M3ShardDraftLogits(_Original())
    x = torch.zeros(3, 4, dtype=torch.bfloat16)
    assert torch.equal(proc(x, _Head(), None, True), x + 1)
    # Rows not selected by the caller take the original processor (it selects them itself).
    assert torch.equal(proc(x, _Head(), None, False), x * 0 - 5)
    # The shared original is not a submodule of the swap (the target model owns it).
    assert list(proc.children()) == []


def _worker(**over):
    w = M3Eagle3Worker.__new__(M3Eagle3Worker)
    if isinstance(w, nn.Module):
        nn.Module.__init__(
            w
        )  # attribute storage only; the worker's own init needs a full model config
    w.is_mtp_eagle = over.get("mtp", False)
    w.guided_decoder = over.get("guided", None)
    w.mapping = SimpleNamespace(enable_attention_dp=over.get("adp", False), tp_size=2, tp_rank=0)
    return w


def _draft(head=None):
    return SimpleNamespace(
        lm_head=head if head is not None else _Head(), logits_processor=_Original()
    )


def test_the_wide_draft_logits_engage_only_where_the_shard_argmax_is_valid() -> None:
    meta = SimpleNamespace(wants_advanced_draft_sampling=False)
    ok = _worker()._m3_wide_draft_logits_ok(meta, _draft())
    assert ok == bool(dispatch.WIDE_SPEC_TAIL)
    if not dispatch.WIDE_SPEC_TAIL:
        return
    assert not _worker()._m3_wide_draft_logits_ok(
        SimpleNamespace(wants_advanced_draft_sampling=True), _draft()
    )
    assert not _worker(guided=object())._m3_wide_draft_logits_ok(meta, _draft())
    assert not _worker(adp=True)._m3_wide_draft_logits_ok(meta, _draft())
    assert not _worker(mtp=True)._m3_wide_draft_logits_ok(meta, _draft())
    assert not _worker()._m3_wide_draft_logits_ok(meta, None)
    padded = _Head()
    padded.padding_size = 8
    assert not _worker()._m3_wide_draft_logits_ok(meta, _draft(padded))


def test_the_base_draft_loop_routes_the_head_through_the_swapped_processor() -> None:
    """The swap only matters if the in-tree loop calls draft_model.logits_processor and then sample_draft_tokens; if
    the base loop stops doing so, W2b goes inert again (the frozen 2e0a73900b was: the head's processor all-gathered
    and cast to fp32, so the sharded check never passed). Fail here instead of silently in-model."""
    import inspect

    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

    # _forward_draft_loop dispatches to the linear loop, which runs the head and the pick.
    dispatcher = inspect.getsource(Eagle3OneModelWorker._forward_draft_loop)
    assert "self._forward_linear_draft_loop(" in dispatcher
    src = inspect.getsource(Eagle3OneModelWorker._forward_linear_draft_loop)
    assert "draft_model.logits_processor(" in src
    assert "self.sample_draft_tokens(" in src


def test_a_wide_draft_pick_reaches_the_shard_argmax(monkeypatch) -> None:
    """The path the in-tree loop takes on a wide step: the swapped processor returns the BF16 shard, and
    sample_draft_tokens -> greedy_sample_draft_with_tp_gather hands it to the fused shard argmax, whose tokens come
    back (engagement at wide rows, without a GPU)."""
    if not dispatch.WIDE_SPEC_TAIL:
        pytest.skip("WIDE_SPEC_TAIL is off")
    calls = []

    def fake_shard_accept(logits, vocab_offset, mapping, draft, batch_idx, force_total, *a, **k):
        calls.append((tuple(logits.shape), logits.dtype, vocab_offset, tuple(draft.shape)))
        m = logits.shape[0]
        acc = (torch.arange(m, dtype=torch.int32) + 7).view(m, 1)
        return acc, torch.ones(m, dtype=torch.int32), torch.zeros(m, dtype=torch.int64)

    monkeypatch.setattr(dispatch.m3_shard_accept_kernel, "shard_accept", fake_shard_accept)
    w = _worker()
    w._d2t = None
    w.mapping.tp_rank = 1
    shard, rows = 100032, 46
    meta = SimpleNamespace(wants_advanced_draft_sampling=False, draft_vocab_size=2 * shard)
    draft_model = _draft()
    assert w._m3_wide_draft_logits_ok(meta, draft_model)

    class _ShardHead(_Head):
        def apply_linear(self, x, bias):
            return torch.zeros(x.shape[0], shard, dtype=torch.bfloat16)

    proc = _M3ShardDraftLogits(draft_model.logits_processor)
    logits = proc(torch.zeros(rows, 8), _ShardHead(), None, True)
    tokens = w.sample_draft_tokens(logits, meta, rows, draft_step=0)
    assert calls == [((rows, shard), torch.bfloat16, shard, (rows, 0))]
    assert torch.equal(tokens, torch.arange(rows, dtype=torch.int32) + 7)
