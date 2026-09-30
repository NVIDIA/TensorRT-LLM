# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the FlashInfer plan refresh under CUDA graphs.

Under CUDA graphs, replay never re-enters ``forward_impl``: whatever split-KV
schedule ``plan()`` computed at capture time is what every replay runs. A model
with more than one decode wrapper (per-layer sliding window plus global
attention) deferred its plan and only recorded the page tuple for
``head_dim > 128``, so every replay reused the warmup batch's plan -- one
full-length dummy in slot 0 and one-token dummies elsewhere. Slots past the
first attended only their first ``kv_chunk_size`` tokens.
``_refresh_fa2_cuda_graph_plans()`` is what re-plans them. This is plan
bookkeeping rather than kernel maths, so it is testable without a GPU.
"""

import ast
import inspect
import math
import pathlib
from dataclasses import dataclass
from typing import Optional

import pytest

from tensorrt_llm._torch.attention.backends import flashinfer as fi_backend
from tensorrt_llm._torch.attention.backends.flashinfer import (
    FlashInferAttentionMetadata,
    FlashInferWrappers,
)


def _method_source(module, class_name: str, method: str) -> str:
    """The source text of one method, sliced from the module file with ast.

    ``inspect.getsource`` on a method needs the function object; going through
    the file keeps this a check on the code as written, and works the same
    whichever way the module was loaded.
    """
    path = inspect.getsourcefile(module)
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    return ast.get_source_segment(
                        pathlib.Path(path).read_text(encoding="utf-8"), sub
                    )
    raise AssertionError(f"{class_name}.{method} not found in {path}")


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


def test_wrappers_without_a_recorded_tuple_are_left_alone():
    """Eager and trtllm-gen wrappers never record one, so nothing to refresh."""
    params = _FakePlanParams()
    cache = {params: FlashInferWrappers(is_planned=True)}
    meta = _FakeMetadata([4, 4], 0, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].fa2_plan_num_blocks is None


@pytest.mark.parametrize("field", ["attention_mask_data", "multi_item_params"])
def test_masked_plans_are_skipped(field):
    """Masked and multi-item plans are per-forward and are flushed, not refreshed."""
    params = _FakePlanParams(**{field: object()})
    cache = {params: _wrappers((3, 3))}
    meta = _FakeMetadata([4, 4], 0, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].fa2_plan_num_blocks == (3, 3)


def test_clean_cached_plans_keeps_graph_captured_wrappers_planned():
    """``_clean_cached_plans`` must not unplan a wrapper the refresh owns.

    The two run back to back in ``prepare()``. If the cleanup cleared
    ``is_planned`` on a wrapper carrying a page tuple, the deferred path would
    plan it again in ``forward_impl`` -- which graph replay never reaches.
    """
    kept, ordinary = _FakePlanParams(), _FakePlanParams()
    cache = {kept: _wrappers((3, 3)), ordinary: FlashInferWrappers(is_planned=True)}
    meta = _FakeMetadata([3, 3], 0, cache)

    FlashInferAttentionMetadata._clean_cached_plans(meta, defer_plan=True)

    assert cache[kept].is_planned is True
    assert cache[ordinary].is_planned is False
    assert meta.replanned == []


def test_refresh_runs_before_the_cleanup_in_prepare():
    """Order matters: the refresh reads a page tuple the cleanup skips over.

    A source check, because ``prepare()`` itself needs a live KV cache manager.
    """
    source = _method_source(fi_backend, "FlashInferAttentionMetadata", "prepare")
    assert ("self._refresh_fa2_cuda_graph_plans()\n            self._clean_cached_plans(") in source


def test_neutral_lse_is_finite_and_very_negative():
    """The split-KV scratch fill value.

    Zero is not neutral (it competes with a real partial) and -inf is fatal
    (``exp2(-inf - -inf)`` is NaN and poisons the row for good, which is the
    failure the scratch handling exists to stop).
    """
    assert math.isfinite(fi_backend._FI_NEUTRAL_LSE)
    assert fi_backend._FI_NEUTRAL_LSE < -1e30
