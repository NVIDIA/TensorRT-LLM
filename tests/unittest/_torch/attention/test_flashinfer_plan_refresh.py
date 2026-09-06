# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the FlashInfer decode plan refresh under CUDA graphs.

Graph replay never re-enters ``forward_impl``: whatever split-KV schedule
``plan()`` computed at capture time is what every replay runs. A model with
more than one decode wrapper (per-layer sliding window plus global attention)
defers its plan to ``forward_impl``, so the recorded page tuple and
``_refresh_fa2_cuda_graph_plans()`` are the only things that keep a replayed
plan current. This is plan bookkeeping rather than kernel maths, so it is
testable without a GPU.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from tensorrt_llm._torch.attention.backends.flashinfer import (
    FlashInferAttentionMetadata,
    FlashInferWrappers,
    _records_fa2_plan,
)


# --------------------------------------------------------------------------
# Which decode plans record a page tuple
# --------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["fa2", "auto", "cutlass"])
def test_every_non_trtllm_gen_decode_plan_is_recorded_under_graphs(backend):
    """The wrapper this fix is about is neither tensor-core nor 'fa2'-labelled.

    A multi-wrapper (variable sliding window) model plans decode wrappers whose
    backend flashinfer resolves to 'auto', and whose head_dim needs no
    tensor-core wrapper. Those used to go unrecorded, so the refresh skipped
    them and every replay reused the capture-time plan.
    """
    assert _records_fa2_plan(True, backend) is True


def test_trtllm_gen_is_not_recorded():
    """trtllm-gen keeps its own per-step block-table/kv_lens refresh."""
    assert _records_fa2_plan(True, "trtllm-gen") is False


@pytest.mark.parametrize("backend", ["fa2", "auto", "trtllm-gen"])
def test_eager_plans_are_not_recorded(backend):
    """Without graphs, ``forward_impl`` re-plans every step; nothing to record."""
    assert _records_fa2_plan(False, backend) is False


# --------------------------------------------------------------------------
# _refresh_fa2_cuda_graph_plans
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _FakePlanParams:
    """Only the two fields ``_refresh_fa2_cuda_graph_plans`` looks at.

    ``eq=False`` keeps identity semantics, so two default instances are two
    distinct cache keys -- as two real per-layer plan params are.
    """

    attention_mask_data: Optional[object] = None
    multi_item_params: Optional[object] = None


class _FakeMetadata:
    """The three attributes the refresh reads, plus a recording re-plan.

    Constructing real metadata needs a KV cache manager and a device, and the
    method under test touches none of that: it reads ``num_blocks``,
    ``num_contexts`` and the plan cache, and calls ``_plan_with_params``.
    """

    def __init__(self, num_blocks, num_contexts, wrappers_by_params):
        self.num_blocks = num_blocks
        self.num_contexts = num_contexts
        self._plan_params_to_wrappers = wrappers_by_params
        self.replanned = []

    def _plan_with_params(self, plan_params):
        self.replanned.append(plan_params)
        self._plan_params_to_wrappers[plan_params].is_planned = True

    refresh = FlashInferAttentionMetadata._refresh_fa2_cuda_graph_plans


def _wrappers(page_tuple):
    return FlashInferWrappers(is_planned=True, fa2_plan_num_blocks=page_tuple)


def test_page_table_change_replans_every_recorded_wrapper():
    """The multi-wrapper (variable sliding window) case the fix was written for.

    Every decode wrapper that was graph-captured now carries a page tuple, so a
    change to the generation page counts must re-plan all of them -- not just
    the first, and not just the tensor-core one.
    """
    windowed, global_ = _FakePlanParams(), _FakePlanParams()
    cache = {windowed: _wrappers((3, 3)), global_: _wrappers((3, 3))}
    # num_contexts=1: the first entry is a context request and is not part of
    # the generation page tuple.
    meta = _FakeMetadata([9, 4, 4], 1, cache)

    meta.refresh()

    assert meta.replanned == [windowed, global_]
    assert cache[windowed].fa2_plan_num_blocks == (3, 3)
    assert all(w.is_planned for w in cache.values())


def test_an_unchanged_page_table_does_not_replan():
    """The plan and its stream sync run only on a page-count change."""
    params = _FakePlanParams()
    cache = {params: _wrappers((4, 4))}
    meta = _FakeMetadata([4, 4], 0, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].is_planned is True


def test_no_generations_clears_the_recorded_page_tuple():
    """A context-only step has no decode plan to keep current."""
    params = _FakePlanParams()
    cache = {params: _wrappers((4, 4))}
    meta = _FakeMetadata([7], 1, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].fa2_plan_num_blocks is None
