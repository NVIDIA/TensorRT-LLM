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
"""One op for every batch size: ``spec_dsa.op`` from 1 to 1,024 rows (SM100).

The op picks the megakernel, the tail-attention hybrid or the large-batch kernels by batch size (``implementation_for``;
the FP8 spec has no hybrid), for the FP8 and the MXFP4 indexer K cache (the ``idx`` fixture). The case builder and the
fp64 references of the large-batch test check every row's top-512 and attention here too, so the implementations are
covered through the one interface: batch sizes around the switches and between the powers of two, the CUDA-graph and
in-place refresh path, the megakernel forced at the FP4 sizes the hybrid took over, and the argument checks of the op.
The hybrid launches the production FP4 indexer logits op of the repository (and its scheduler metadata): the tests that
run it skip when that op is not registered.
"""

import dataclasses
import re

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import hybrid_op as _hybrid
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import large_batch_op as _lbo
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op as _tail
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta_op as _topk

from . import test_cute_dsl_spec_dsa_large_batch as H

pytestmark = H.pytestmark
idx = H.idx
# The batch sizes at which the FP4 spec switches: the first and last size of the hybrid with the size outside them, its
# megakernel's last size, and the first size of every further compile of the large-batch indexer kernel
# (``large_batch_op._VARIANTS``) with the size before it.
FP4_LAST = H.SPEC_FP4.megakernel_max_batch
HYBRID_FIRST, HYBRID_LAST = H.SPEC_FP4.tail_hybrid_range
FP4_VARIANTS = [first for first, _ in _lbo._VARIANTS.get(H.SPEC_FP4.kernel_key, ())] or [64]
FP4_VARIANT = min(FP4_VARIANTS)
SWITCHES = {
    FP4_LAST,
    FP4_LAST + 1,
    HYBRID_FIRST - 1,
    HYBRID_FIRST,
    HYBRID_LAST,
    HYBRID_LAST + 1,
} | {b for first in FP4_VARIANTS for b in (first - 1, first)}
# The FP4 sizes the megakernel served by default before the hybrid took them over; it stays selectable there.
FORCED_MEGAKERNEL = [HYBRID_FIRST, 8, 9, 13, 16, 17, FP4_LAST]
# The hybrid's launch variants change with the batch size: the CTAs per row of its top-K (``topk_multicta_op``) and the
# PDL / poll limits of its tail attention. HYBRID_SIZES holds its first and last size, both sides of every limit and the
# megakernel's last size; VARIANT_SIZES one size per variant.
_LIMITS = [m for m, _ in _topk._CLUSTER_LIMITS] + [_tail.PDL_MAX_BATCH, _tail.POLL_MAX_BATCH]
HYBRID_SIZES = sorted(
    {HYBRID_FIRST, HYBRID_LAST, 13, FP4_LAST}
    | {b for limit in _LIMITS for b in (limit, limit + 1) if HYBRID_FIRST <= b <= HYBRID_LAST}
)
VARIANT_SIZES = sorted(
    {
        (_topk.cluster_size(b), _tail.default_pdl(b), _tail.default_poll(b)): b
        for b in range(HYBRID_FIRST, HYBRID_LAST + 1)
    }.values()
)
# Row lengths of the hybrid tests: mixed (short and long), long (every row has 16,384 positions or more: the top-K
# stage the hybrid is tuned for), short (no row has), and rows with planted ties and all-equal logits.
HYBRID_ROWS = {
    "mixed": H._plain(H.LENGTHS),
    "long": H._plain(
        [262144, 262143, 200000, 131072, 100003, 70000, 65535, 32769, 24576, 16385, 16384, 40000]
    ),
    "short": H._plain([512, 513, 1000, 4097, 8192, 8193, 8197, 12000, 16383, 700]),
    "ties": [
        {"n": 100000, "ties": 3},
        {"n": 262144, "ties": 6000},
        {"n": 5000, "flat": True},
        {"n": 262143, "flat": True},
        {"n": 700, "ties": 100},
        {"n": 512},
        {"n": 90000},
        {"n": 40000, "ties": 600},
    ],
}


def _op():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import op

    return op


def _skip_without_hybrid():
    ok, why = _hybrid.is_supported(config=H.SPEC_FP4)
    if not ok:
        pytest.skip(why)


def _needs_hybrid(batch, idx="fp4", implementation=None):
    """Skip the test if it launches the tail hybrid and the production logits op is not available."""
    if (implementation or _op().implementation_for(batch, H._spec_of(idx))) == "tail_hybrid":
        _skip_without_hybrid()


def _prepare(op, inp, state=None, idx="fp8", implementation=None):
    # host lengths: prepare() reads them without a device sync
    return op.prepare(
        inp["ctx_lens"].cpu(),
        inp["kv_lens"].cpu(),
        inp["idx_block_table"],
        inp["sinks"],
        state=state,
        config=H._spec_of(idx),
        implementation=implementation,
    )


def _launch(op, state, inp, topk, out):
    op.run(
        state,
        inp["q_idx"],
        inp["w_idx"],
        inp["k_cache"],
        inp["idx_block_table"],
        inp["ctx_lens"],
        inp["kv_lens"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        topk,
        out,
        softmax_scale=H.SM_SCALE,
        **({"q_scale": inp["q_scale"]} if "q_scale" in inp else {}),
    )


def _capture(fn):
    """Warm up on a side stream (as the other specDSA tests do), then capture ``fn`` into a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _check_error_bits(op, state):
    """No error bit is set. The hybrid has no device flag (``error_bits`` is a constant zero): nothing to check."""
    if state.implementation != "tail_hybrid":
        assert int(op.error_bits(state).max()) == 0


def test_implementation_for(idx):
    op = _op()
    spec = H._spec_of(idx)
    assert op.implementation_for(1, spec) == "megakernel"
    assert op.implementation_for(spec.max_batch, spec) == "large_batch"
    if spec.tail_hybrid_range is None:
        last = spec.megakernel_max_batch
        assert op.implementation_for(last, spec) == "megakernel"
        assert op.implementation_for(last + 1, spec) == "large_batch"
    else:
        first, last = spec.tail_hybrid_range
        assert op.implementation_for(first - 1, spec) == "megakernel"
        assert op.implementation_for(first, spec) == "tail_hybrid"
        assert op.implementation_for(last, spec) == "tail_hybrid"
        assert op.implementation_for(last + 1, spec) == "large_batch"
    for bad in (-1, 0, spec.max_batch + 1):
        with pytest.raises(ValueError, match="batch size"):
            op.implementation_for(bad, spec)


def _check_matches_reference(batch, idx, implementation=None):
    op = _op()
    _needs_hybrid(batch, idx, implementation)
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=100 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx, implementation=implementation)
    assert state.batch_size == batch
    assert state.implementation == (implementation or op.implementation_for(batch, H._spec_of(idx)))
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    _check_error_bits(op, state)
    H._check_rows(inp, bases, topk, out, "realistic")


@pytest.mark.parametrize(
    "batch",
    sorted({1, 2, 3, 4, 5, 8, 9, 13, 16, 17, 24, 32, 33, 40, 47, 48, 49, 64, 149, 1024} | SWITCHES),
)
def test_matches_reference(batch, idx):
    _check_matches_reference(batch, idx)


@pytest.mark.parametrize("batch", FORCED_MEGAKERNEL)
def test_forced_megakernel_matches_reference(batch):
    """The FP4 megakernel at the sizes the hybrid serves by default: ``implementation="megakernel"``."""
    _check_matches_reference(batch, "fp4", "megakernel")


@pytest.mark.parametrize("batch", [HYBRID_FIRST, FP4_VARIANT])
def test_forced_large_batch_matches_reference(batch):
    """The large-batch kernels below their default range: at the hybrid's first size and at the first size of their
    first knob range (also the hybrid's)."""
    _check_matches_reference(batch, "fp4", "large_batch")


def _check_refresh_in_place_and_cuda_graph(batch, idx, implementation=None):
    op = _op()
    _needs_hybrid(batch, idx, implementation)
    ns = H.LENGTHS[:batch] if batch <= len(H.LENGTHS) else H.LENGTHS
    inp, bases = H._make_case(batch, H._plain(ns), seed=7 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx, implementation=implementation)
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")
    # the same state refreshed in place for another step, then one captured launch replayed
    inp2, bases2 = H._make_case(batch, H._plain(ns[::-1]), seed=70 + batch, idx=idx)
    assert _prepare(op, inp2, state=state, idx=idx, implementation=implementation) is state
    topk.fill_(-1)
    out.zero_()
    _launch(op, state, inp2, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp2, bases2, topk, out, "realistic")
    graph = _capture(lambda: _launch(op, state, inp2, topk, out))
    for _ in range(2):
        topk.fill_(-1)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        H._check_rows(inp2, bases2, topk, out, "realistic")
    _check_error_bits(op, state)


@pytest.mark.parametrize("batch", sorted({4, 5, 8, 13, 24, 40, 47, 48, 60} | SWITCHES))
def test_refresh_in_place_and_cuda_graph(batch, idx):
    _check_refresh_in_place_and_cuda_graph(batch, idx)


@pytest.mark.parametrize("batch", [HYBRID_FIRST, 8, 13, FP4_LAST])
def test_forced_megakernel_refresh_in_place_and_cuda_graph(batch):
    _check_refresh_in_place_and_cuda_graph(batch, "fp4", "megakernel")


def _check_negative_logits(batch, idx, implementation=None):
    """Rows whose logits are all <= 0 through the op."""
    op = _op()
    _needs_hybrid(batch, idx, implementation)
    ns = H.LENGTHS[:batch] if batch <= len(H.LENGTHS) else H.LENGTHS
    inp, bases = H._make_case(batch, [{"n": n, "neg": True} for n in ns], seed=170 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx, implementation=implementation)
    assert state.implementation == (implementation or op.implementation_for(batch, H._spec_of(idx)))
    topk, out = H._new_outputs(batch)
    for launch in range(3):
        topk.fill_(-5)
        _launch(op, state, inp, topk, out)
        torch.cuda.synchronize()
        _check_error_bits(op, state)
        if launch == 0:
            H._check_rows(inp, bases, topk, out, "realistic")
        else:
            H._check_selection(bases, topk)


@pytest.mark.parametrize("batch", sorted({1, 3, 9, 64} | SWITCHES))
def test_negative_logits(batch, idx):
    """The megakernel sizes, the hybrid sizes and the large-batch sizes of both specs."""
    _check_negative_logits(batch, idx)


@pytest.mark.parametrize("batch", [9, FP4_LAST])
def test_forced_megakernel_negative_logits(batch):
    _check_negative_logits(batch, "fp4", "megakernel")


# --------------------------------------------------------------------------------------------------------------------
# the tail hybrid through the op


@pytest.mark.parametrize("kind", list(HYBRID_ROWS))
@pytest.mark.parametrize("batch", HYBRID_SIZES)
def test_hybrid_matches_reference(batch, kind):
    """The hybrid against the fp64 reference at its first and last size and on both sides of every limit of its launch
    variants (``HYBRID_SIZES``), with mixed, long, short rows and rows with planted ties and all-equal logits: every
    row's exact top-512 set and attention."""
    _skip_without_hybrid()
    op = _op()
    inp, bases = H._make_case(batch, HYBRID_ROWS[kind], seed=3000 + batch + len(kind), idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    assert state.implementation == "tail_hybrid"
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


@pytest.mark.parametrize("batch", [13, 24, HYBRID_LAST])
def test_hybrid_graph_replays_over_in_place_refreshed_inputs(batch):
    """One captured launch replays over several steps while the inputs change in place and ``prepare`` refreshes the
    state in place: the schedule of the logits kernel follows the lengths (rows of 512 to 262,144 positions rotate over
    the rows, so a stale schedule would show). The three sizes launch the hybrid with polling, with PDL only, plain."""
    _skip_without_hybrid()
    op = _op()
    ns = (H.LENGTHS * 2)[:batch]
    steps = [ns[r:] + ns[:r] for r in (0, 5, 9)]
    inp, bases = H._make_case(batch, H._plain(steps[0]), seed=2000 + batch, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    assert state.implementation == "tail_hybrid"
    topk, out = H._new_outputs(batch)
    op.precompile(torch.device("cuda"), batch, H.SPEC_FP4)
    graph = _capture(lambda: _launch(op, state, inp, topk, out))
    lengths, sched = state.state.ctx_lens, state.state.sched
    for step, rows in enumerate(steps):
        if step:
            new, bases = H._make_case(batch, H._plain(rows), seed=2000 + batch + step, idx="fp4")
            for key, buf in inp.items():
                assert buf.shape == new[key].shape, key
                buf.copy_(new[key])
            before = sched.clone()
            assert _prepare(op, inp, state=state, idx="fp4") is state
            assert state.state.ctx_lens is lengths, "prepare must refresh the lengths in place"
            assert state.state.sched is sched, "prepare must refresh the schedule in place"
            assert not torch.equal(sched, before), "the schedule follows the new lengths"
        assert torch.equal(lengths, inp["ctx_lens"]), "the state holds the lengths of the step"
        topk.fill_(-1)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        H._check_rows(inp, bases, topk, out, "realistic")


def _k1_runner():
    """The compile cache holder of the production FP4 logits op."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    runner = getattr(cute_dsl_custom_ops, "CuteDSLFP4PagedMQALogitsRunner", None)
    if runner is None:
        pytest.skip("the production FP4 logits runner is not in this build")
    return runner


@pytest.mark.parametrize("batch", VARIANT_SIZES)
def test_hybrid_precompile_then_captured_op(batch, monkeypatch):
    """``op.precompile`` followed by a CUDA-graph capture of ``op.run`` compiles nothing during the capture, the
    production logits kernel included: there is no warm-up launch in between. One size per launch variant (CTAs per
    row of the top-K, PDL, polling)."""
    _skip_without_hybrid()
    op = _op()
    runner = _k1_runner()
    # as in a fresh process: nothing compiled, the logits kernel not known to be launched
    monkeypatch.setattr(runner, "kernel_cache", {})
    monkeypatch.setattr(_tail, "_compiled", {})
    monkeypatch.setattr(_topk, "_compiled", {})
    monkeypatch.setattr(_hybrid, "_k1_ready", set())
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=2500, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    op.precompile(torch.device("cuda"), batch, H.SPEC_FP4)
    compiled = (len(runner.kernel_cache), len(_tail._compiled), len(_topk._compiled))
    assert compiled[0] >= 1 and compiled[1:] == (1, 1), "precompile compiles all three kernels"
    topk, out = H._new_outputs(batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(op, state, inp, topk, out)
    assert (len(runner.kernel_cache), len(_tail._compiled), len(_topk._compiled)) == compiled
    graph.replay()
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_hybrid_capture_before_the_logits_kernel_is_compiled_is_refused(monkeypatch):
    """The production logits kernel cannot be compiled inside a capture: precompile (or launch) first."""
    _skip_without_hybrid()
    op = _op()
    batch = 8
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=2600, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)  # warm-up: compiles everything
    monkeypatch.setattr(_hybrid, "_k1_ready", set())  # as in a fresh process
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="precompile"):
        with torch.cuda.graph(graph):
            _launch(op, state, inp, topk, out)
    monkeypatch.undo()
    topk.fill_(-1)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


def test_hybrid_prepare_takes_device_lengths():
    """Like the other implementations: ``ctx_lens`` and ``kv_lens`` may be device tensors (``prepare`` reads them)."""
    _skip_without_hybrid()
    op = _op()
    batch = 9
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=2700, idx="fp4")
    state = op.prepare(
        inp["ctx_lens"],
        inp["kv_lens"],
        inp["idx_block_table"],
        inp["sinks"],
        config=H.SPEC_FP4,
    )
    assert state.implementation == "tail_hybrid"
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


def _guard_hybrid_lengths(monkeypatch, state):
    """Make the logits op and the top-K fail, before they launch, on any lengths tensor but the state's own copy."""
    lengths = state.state.ctx_lens
    k1_op, topk_run = _hybrid._k1_op, _topk.run

    def checked(ctx_lens):
        assert ctx_lens.data_ptr() == lengths.data_ptr(), (
            "the launch must use the lengths of the state"
        )

    def guarded_k1_op():
        k1 = k1_op()

        def k1_checked(q, sf_q, kv, weights, ctx_lens, *args, **kwargs):
            checked(ctx_lens)
            return k1(q, sf_q, kv, weights, ctx_lens, *args, **kwargs)

        return k1_checked

    def topk_checked(logits, topk, ctx_lens, **kwargs):
        checked(ctx_lens)
        return topk_run(logits, topk, ctx_lens, **kwargs)

    monkeypatch.setattr(_hybrid, "_k1_op", guarded_k1_op)
    monkeypatch.setattr(_topk, "run", topk_checked)


@pytest.mark.parametrize("device_lengths", ["shorter", "longer"])
def test_hybrid_launch_uses_the_lengths_of_prepare(device_lengths, monkeypatch):
    """The logits kernel and the top-K read the lengths ``prepare`` kept in the state, never the ``ctx_lens`` of the
    launch (the logits kernel can hang on context lengths that differ from the ones its schedule was built from). With
    device lengths 2,000 positions short or the longest possible, the launch completes and is exact for the lengths of
    ``prepare``."""
    _skip_without_hybrid()
    op = _op()
    ns = [10492, 9000, 12000, 8200, 20000, 16500, 11111, 13000]
    batch = len(ns)
    inp, bases = H._make_case(batch, H._plain(ns), seed=3300, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    assert torch.equal(state.state.ctx_lens, inp["ctx_lens"])
    other = dict(inp)
    other["ctx_lens"] = (
        inp["ctx_lens"] - 2000
        if device_lengths == "shorter"
        else torch.full_like(inp["ctx_lens"], H.N_MAX)
    )
    _guard_hybrid_lengths(monkeypatch, state)
    topk, out = H._new_outputs(batch)
    _launch(op, state, other, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


def _strided(table):
    """A view of ``table`` with the same values whose rows are twice as far apart."""
    wide = torch.zeros((table.shape[0], 2 * table.shape[1]), dtype=table.dtype, device=table.device)
    wide[:, : table.shape[1]] = table
    view = wide[:, : table.shape[1]]
    assert not view.is_contiguous()
    return view


@pytest.mark.parametrize("name", ["idx_block_table", "bt_swa", "bt_c"])
def test_hybrid_refuses_strided_block_tables(name):
    """The logits kernel and the tail attention are compiled for compact tensors: the hybrid refuses a strided block
    table before it launches anything (``prepare`` the indexer table, ``run`` all three), unlike the megakernel
    (``test_forced_megakernel_takes_strided_block_tables``)."""
    _skip_without_hybrid()
    op = _op()
    batch = 9
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=3400, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    strided = dict(inp)
    strided[name] = _strided(inp[name])
    topk, out = H._new_outputs(batch)
    if name == "idx_block_table":
        with pytest.raises(ValueError, match=f"{name} must be a contiguous int32"):
            _prepare(op, strided, idx="fp4")
    with pytest.raises(ValueError, match=f"{name} must be a contiguous int32"):
        _launch(op, state, strided, topk, out)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


def test_forced_megakernel_takes_strided_block_tables():
    """The megakernel reads strided block tables (``bt_c``: 16-byte aligned rows)."""
    op = _op()
    batch = 9
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=3401, idx="fp4")
    strided = dict(inp)
    for name in ("idx_block_table", "bt_swa", "bt_c"):
        strided[name] = _strided(inp[name])
    state = _prepare(op, strided, idx="fp4", implementation="megakernel")
    topk, out = H._new_outputs(batch)
    _launch(op, state, strided, topk, out)
    torch.cuda.synchronize()
    _check_error_bits(op, state)
    H._check_rows(inp, bases, topk, out, "realistic")


def test_hybrid_error_bits_is_a_device_zero():
    """The hybrid has no device flag: ``error_bits`` is a persistent int32 [1] zero on the device."""
    _skip_without_hybrid()
    op = _op()
    inp, _ = H._make_case(6, H._plain(H.LENGTHS), seed=2800, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    bits = op.error_bits(state)
    assert bits.dtype == torch.int32 and tuple(bits.shape) == (1,) and bits.is_cuda
    assert int(bits) == 0
    assert op.error_bits(state).data_ptr() == bits.data_ptr()


def test_state_remembers_its_implementation_and_refuses_mismatches():
    """A state keeps the implementation it was built for: a refresh without ``implementation`` keeps it, a request for
    another one or another batch size is an error."""
    _skip_without_hybrid()
    op = _op()
    inp, _ = H._make_case(8, H._plain(H.LENGTHS), seed=2900, idx="fp4")
    hybrid = _prepare(op, inp, idx="fp4")
    assert hybrid.implementation == "tail_hybrid"
    assert _prepare(op, inp, state=hybrid, idx="fp4") is hybrid
    assert _prepare(op, inp, state=hybrid, idx="fp4", implementation="tail_hybrid") is hybrid
    with pytest.raises(
        ValueError, match=re.escape("state is for 8 rows (tail_hybrid), got 8 rows (megakernel)")
    ):
        _prepare(op, inp, state=hybrid, idx="fp4", implementation="megakernel")
    forced = _prepare(op, inp, idx="fp4", implementation="megakernel")
    assert forced.implementation == "megakernel"
    assert _prepare(op, inp, state=forced, idx="fp4") is forced  # no keyword: the state's own
    with pytest.raises(
        ValueError, match=re.escape("state is for 8 rows (megakernel), got 8 rows (tail_hybrid)")
    ):
        _prepare(op, inp, state=forced, idx="fp4", implementation="tail_hybrid")
    inp9, _ = H._make_case(9, H._plain(H.LENGTHS), seed=2901, idx="fp4")
    with pytest.raises(
        ValueError, match=re.escape("state is for 8 rows (tail_hybrid), got 9 rows (tail_hybrid)")
    ):
        _prepare(op, inp9, state=hybrid, idx="fp4")
    with pytest.raises(
        ValueError, match=re.escape("state is for 8 rows (megakernel), got 9 rows (tail_hybrid)")
    ):
        _prepare(op, inp9, state=forced, idx="fp4")
    # a request that cannot serve the batch size is refused whatever the state
    inp21, _ = H._make_case(21, H._plain(H.LENGTHS), seed=2902, idx="fp4")
    with pytest.raises(ValueError, match="serves 1 to 20 rows"):
        _prepare(op, inp21, idx="fp4", implementation="megakernel")


def test_hybrid_rejects_bad_arguments():
    _skip_without_hybrid()
    op = _op()
    batch = 6
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=3100, idx="fp4")
    state = _prepare(op, inp, idx="fp4")
    topk, out = H._new_outputs(batch)
    dev = torch.device("cuda")

    def launch(**changes):
        op.run(
            state,
            *(
                changes.get(k, inp[k])
                for k in (
                    "q_idx",
                    "w_idx",
                    "k_cache",
                    "idx_block_table",
                    "ctx_lens",
                    "kv_lens",
                    "q_att",
                    "swa_pool",
                    "bt_swa",
                    "c_pool",
                    "bt_c",
                )
            ),
            changes.get("topk", topk),
            changes.get("out", out),
            softmax_scale=H.SM_SCALE,
            q_scale=changes.get("q_scale", inp["q_scale"]),
        )

    with pytest.raises(ValueError, match="needs q_scale"):
        launch(q_scale=None)
    with pytest.raises(ValueError, match="q_scale must be contiguous"):
        launch(q_scale=inp["q_scale"].float())
    with pytest.raises(ValueError, match="k_cache must hold 2176 bytes per page"):
        launch(k_cache=torch.zeros((10, 2000), dtype=torch.uint8, device=dev))
    with pytest.raises(ValueError, match="ctx_lens must be contiguous"):
        launch(ctx_lens=inp["ctx_lens"].long())
    with pytest.raises(ValueError, match="ctx_lens must be on the device"):
        launch(ctx_lens=inp["ctx_lens"].cpu())
    with pytest.raises(ValueError, match="kv_lens must be contiguous"):
        launch(kv_lens=inp["kv_lens"][:-1])
    with pytest.raises(ValueError, match="idx_block_table must be a contiguous int32"):
        launch(idx_block_table=inp["idx_block_table"].long())
    with pytest.raises(ValueError, match="bt_swa must be a contiguous int32"):
        launch(bt_swa=inp["bt_swa"].long())
    with pytest.raises(ValueError, match="bt_c must be a contiguous int32"):
        launch(bt_c=inp["bt_c"].long())
    with pytest.raises(ValueError, match="topk must be contiguous"):
        launch(topk=topk.long())
    with pytest.raises(ValueError, match="out must be a contiguous bfloat16"):
        launch(out=out.float())
    with pytest.raises(ValueError, match="w_idx must be contiguous"):
        launch(w_idx=inp["w_idx"].half())
    # prepare
    ctx, kv, bt, sinks = (
        inp["ctx_lens"].cpu(),
        inp["kv_lens"].cpu(),
        inp["idx_block_table"],
        inp["sinks"],
    )
    short = ctx.clone()
    short[0] = 300
    with pytest.raises(ValueError, match="compressed lengths"):
        op.prepare(short, kv, bt, sinks, config=H.SPEC_FP4)
    low = kv.clone()
    low[0] = 100
    with pytest.raises(ValueError, match="uncompressed lengths"):
        op.prepare(ctx, low, bt, sinks, config=H.SPEC_FP4)
    with pytest.raises(ValueError, match="kv_lens must have 6 entries"):
        op.prepare(ctx, kv[:-1], bt, sinks, config=H.SPEC_FP4)
    with pytest.raises(ValueError, match="pages per row"):
        op.prepare(ctx, kv, bt[:, :4].contiguous(), sinks, config=H.SPEC_FP4)
    with pytest.raises(ValueError, match="CUDA tensor"):
        op.prepare(ctx, kv, bt.cpu(), sinks, config=H.SPEC_FP4)
    with pytest.raises(ValueError, match="sinks must have 64 entries"):
        op.prepare(ctx, kv, bt, sinks[:8], config=H.SPEC_FP4)
    # the refused launches changed nothing: the state still serves the step
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")


def test_rejects_bad_arguments():
    op = _op()
    dev = torch.device("cuda")
    i32 = torch.int32
    sinks = torch.zeros(64, dtype=torch.float32, device=dev)
    bt = torch.zeros((2, 128), dtype=i32, device=dev)
    with pytest.raises(ValueError, match="sinks"):
        op.prepare(
            torch.tensor([4096], dtype=i32),
            torch.tensor([16384], dtype=i32),
            bt[:1],
            sinks[:8],
        )
    with pytest.raises(ValueError, match="compressed lengths"):
        op.prepare(torch.tensor([300], dtype=i32), torch.tensor([1200], dtype=i32), bt[:1], sinks)
    state = op.prepare(
        torch.tensor([4096], dtype=i32), torch.tensor([16384], dtype=i32), bt[:1], sinks
    )
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.full((2,), 4096, dtype=i32),
            torch.full((2,), 16384, dtype=i32),
            bt,
            sinks,
            state=state,
        )
    # a state keeps the spec it was built for
    other = dataclasses.replace(H.SPEC, name="other", topk=1024)
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.tensor([4096], dtype=i32),
            torch.tensor([16384], dtype=i32),
            bt[:1],
            sinks,
            state=state,
            config=other,
        )
    assert state.config is H.SPEC
    many = H.SPEC.megakernel_max_batch + 1
    bt_many = torch.zeros((many, 128), dtype=i32, device=dev)
    large_state = op.prepare(
        torch.full((many,), 4096, dtype=i32), torch.full((many,), 16384, dtype=i32), bt_many, sinks
    )
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.full((1,), 4096, dtype=i32),
            torch.full((1,), 16384, dtype=i32),
            bt[:1],
            sinks,
            state=large_state,
        )
