# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RuntimeHooks prefill capture fields, CPU.

- ``apply_prefill_capture_hooks``: on the breakable backend only, a model's extra buckets join a non-empty configured
  list whose top bucket reaches ``prefill_capture_extra_min_top`` (a shorter list keeps its own buckets: it would
  otherwise pad its gap up to the first extra bucket), each at most the capturable ceiling, and the cap removes every
  larger entry, configured ones included.
- ``prefill_graphs_released`` runs with the model when the decoder runner releases its breakable graphs, after they are
  cleared.
- The engine applies it once, right after its own filter and before the list reaches the piecewise backend and the
  runners.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import model_engine
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, apply_prefill_capture_hooks

pytestmark = pytest.mark.cpu_only

TODAY = list(range(256, 4097, 256))
GRID = (5120, 6144, 7168, 8192, 9216, 10240, 11264, 12287, 12288, 13312, 14336, 15360, 16384)


def test_the_fields_default_off():
    hooks = RuntimeHooks()
    assert hooks.prefill_capture_extra_num_tokens == ()
    assert hooks.prefill_capture_max_num_tokens is None
    assert hooks.prefill_capture_extra_min_top == 0 and hooks.prefill_graphs_released is None
    assert apply_prefill_capture_hooks(TODAY, hooks, max_num_tokens=16384, breakable=True) == (
        TODAY,
        [],
        [],
    )


def test_extra_buckets_extend_a_breakable_list():
    hooks = RuntimeHooks(prefill_capture_extra_num_tokens=GRID)
    final, added, dropped = apply_prefill_capture_hooks(
        TODAY, hooks, max_num_tokens=16384, breakable=True
    )
    assert final == TODAY + list(GRID) and added == list(GRID) and dropped == []
    # Above the capturable ceiling they are left out; a configured bucket is not re-added.
    final, added, _ = apply_prefill_capture_hooks(
        TODAY + [8192], hooks, max_num_tokens=12288, breakable=True
    )
    assert final == sorted(set(TODAY) | {b for b in GRID if b <= 12288})
    assert 8192 not in added and max(added) == 12288


@pytest.mark.parametrize("kept, breakable", [(TODAY, False), ([], True)])
def test_extra_buckets_never_start_or_switch_backends(kept, breakable):
    hooks = RuntimeHooks(prefill_capture_extra_num_tokens=GRID)
    assert apply_prefill_capture_hooks(kept, hooks, max_num_tokens=16384, breakable=breakable) == (
        kept,
        [],
        [],
    )


def test_extra_buckets_skip_a_list_below_the_minimum_top():
    """A [256, 512] list with the 5120.. grid would pad a 600-token step to 5120 (it used to run eager, unpadded)."""
    hooks = RuntimeHooks(prefill_capture_extra_num_tokens=GRID, prefill_capture_extra_min_top=4096)
    assert apply_prefill_capture_hooks([256, 512], hooks, max_num_tokens=16384, breakable=True) == (
        [256, 512],
        [],
        [],
    )
    assert apply_prefill_capture_hooks(
        list(range(256, 3841, 256)), hooks, max_num_tokens=16384, breakable=True
    ) == (list(range(256, 3841, 256)), [], [])
    final, added, _ = apply_prefill_capture_hooks(
        TODAY, hooks, max_num_tokens=16384, breakable=True
    )
    assert final == TODAY + list(GRID) and added == list(GRID)
    # A list that already goes past the minimum also takes them.
    final, added, _ = apply_prefill_capture_hooks(
        [256, 8192], hooks, max_num_tokens=16384, breakable=True
    )
    assert added == [b for b in GRID if b != 8192]


def test_the_cap_drops_configured_buckets_too():
    hooks = RuntimeHooks(prefill_capture_max_num_tokens=4096)
    final, added, dropped = apply_prefill_capture_hooks(
        TODAY + [8192, 16384], hooks, max_num_tokens=16384, breakable=True
    )
    assert final == TODAY and added == [] and dropped == [8192, 16384]
    both = RuntimeHooks(prefill_capture_extra_num_tokens=GRID, prefill_capture_max_num_tokens=8192)
    final, added, dropped = apply_prefill_capture_hooks(
        TODAY, both, max_num_tokens=16384, breakable=True
    )
    assert final[-1] == 8192 and added == [5120, 6144, 7168, 8192]
    assert dropped == [b for b in GRID if b > 8192]
    # Another backend's list is the configured one (the hooks are breakable-graph buckets).
    assert apply_prefill_capture_hooks(
        TODAY + [8192], hooks, max_num_tokens=16384, breakable=False
    ) == (TODAY + [8192], [], [])


def test_the_engine_applies_the_hooks_after_its_plan():
    src = Path(model_engine.__file__).read_text()
    plan = src.index("self._prefill_cuda_graph_num_tokens = plan.prefill_num_tokens")
    # The hooked list replaces the engine's own list (every consumer reads self._prefill_cuda_graph_num_tokens).
    hook = src.index(
        "self._prefill_cuda_graph_num_tokens, added, dropped = apply_prefill_capture_hooks(\n"
        "            self._prefill_cuda_graph_num_tokens,"
    )
    first_use = src.index("prefill_capture_num_tokens=self._prefill_cuda_graph_num_tokens")
    assert plan < hook < first_use
    assert "runtime_hooks_of(self.model)" in src[hook : hook + 200]
    assert "PrefillCudaGraphBackend.BREAKABLE" in src[hook : hook + 500]


def test_releasing_the_graphs_calls_the_release_hook_after_clearing_them():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs

    events = []

    def clear(**kwargs):
        events.append("prefill")

    model = SimpleNamespace(
        runtime_hooks=RuntimeHooks(prefill_graphs_released=lambda m: events.append(("hook", m)))
    )
    graphs = SimpleNamespace(
        model=model,
        cuda_graph_runner=SimpleNamespace(clear=lambda **kwargs: events.append("decode")),
        breakable_cuda_graph_runner=SimpleNamespace(clear=clear),
    )
    DecoderCudaGraphs.release(graphs)
    assert events == ["decode", "prefill", ("hook", model)]

    # Without breakable graphs there is nothing to release, so the hook does not run.
    events.clear()
    graphs.breakable_cuda_graph_runner = None
    DecoderCudaGraphs.release(graphs)
    assert events == ["decode"]

    # A model without the hook releases its graphs as before.
    events.clear()
    graphs.breakable_cuda_graph_runner = SimpleNamespace(clear=clear)
    graphs.model = SimpleNamespace(runtime_hooks=RuntimeHooks())
    DecoderCudaGraphs.release(graphs)
    assert events == ["decode", "prefill"]
