# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILGLUE_B: the DSpark step's next_new_tokens and the forward-tail spec fold in one launch (kernels/m3_nnt_fold).

The launch must write exactly what the stock pair writes -- the catalog next_new_tokens (N3 A2; its in-tree torch
reference where the CuTe DSL op cannot run) and then m3_fold_tail (this target's C++ foldTail and its Triton twin) --
into next_new_tokens and into every buffer the fold tail writes (the sampler store's new / next-new / draft tokens and
lengths, the engine's input ids, draft ids, position and KV-length offsets), for the rows the step armed and nothing
for the others: at b1 K7 and the other decode shapes, with shuffled batch indices, a wider accepted buffer, a wider
store, int32 / int64 batch indices, unarmed and partly armed steps, inside a CUDA graph, and over a run of steps whose
armed rows change (the first decode step after an admission). In the model (M3DSparkStepFoldFwd + the DSpark worker):
the tail skips its own launch only for the outputs the fused launch stored, the sampler's stored-rows check still
holds, and the worker takes the launch only on decode-only steps under the switch.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
    m3_fold_tail,
    m3_nnt_fold,
)

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_DIR = Path(_family.__file__).resolve().parent / TARGET
VOCAB = 200064
NUM_SLOTS = 12
MAX_ROWS = 8
STALE = -4

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _sm107() -> bool:
    """The C++ fold tail op is this target's (built for SM107 in its wheel): its cases run there, and fail there if
    the op is missing; elsewhere they are skipped."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


def _stock_next_new_tokens(accepted, num_accepted, batch_idx, next_draft):
    """N3 A2's catalog op where it runs (sm_90+: its kernel waits on the grid dependency), else its exact reference
    (the in-tree index + concat)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
        strict_accept,
    )

    op = strict_accept.spec_next_new_tokens
    if torch.cuda.get_device_capability()[0] >= 9 and strict_accept.supports_next_new_tokens(
        accepted, num_accepted, batch_idx, next_draft
    ):
        return op(accepted, num_accepted, batch_idx, next_draft)
    return op.reference(accepted, num_accepted, batch_idx, next_draft)


class _Bufs:
    """The sampler store (as M3SpecStepFold views it) and the engine's next-step input buffers."""

    def __init__(self, widths, t, g):
        wn, wx, wd = widths
        i32 = dict(dtype=torch.int32, device="cuda")
        self.new_tokens = torch.randint(0, VOCAB, (wn, NUM_SLOTS, 1), generator=g, **i32)
        self.next_new_tokens = torch.randint(0, VOCAB, (wx, NUM_SLOTS, 1), generator=g, **i32)
        self.next_draft_tokens = torch.randint(0, VOCAB, (NUM_SLOTS, wd), generator=g, **i32)
        self.new_tokens_lens = torch.randint(1, 5, (NUM_SLOTS,), generator=g, **i32)
        self.ids = torch.full((MAX_ROWS * t,), STALE, **i32)
        self.draft_ids = torch.full((MAX_ROWS * (t - 1),), STALE, **i32)
        self.pos_off = torch.full((MAX_ROWS * t,), STALE, **i32)
        self.kv_off = torch.full((MAX_ROWS,), STALE, **i32)

    def clone(self):
        c = _Bufs.__new__(_Bufs)
        c.__dict__ = {k: v.clone() for k, v in self.__dict__.items()}
        return c

    def store_views(self):
        return (
            self.new_tokens.squeeze(-1).T,
            self.next_new_tokens.squeeze(-1).T,
            self.next_draft_tokens,
            self.new_tokens_lens,
        )

    def engine(self):
        return (self.ids, self.draft_ids, self.pos_off, self.kv_off)

    def equal(self, other):
        return all(
            torch.equal(a, b) for a, b in zip(self.__dict__.values(), other.__dict__.values())
        )


def _step(n, d, g, acc_width=None, bidx_dtype=torch.int32, shuffle=True):
    acc_width = acc_width or d + 1
    accepted = torch.randint(
        0, VOCAB, (n, acc_width), generator=g, device="cuda", dtype=torch.int32
    )
    num = torch.randint(1, d + 2, (n,), generator=g, device="cuda", dtype=torch.int32)
    if shuffle:
        bidx = torch.randperm(n, generator=g, device="cuda")
    else:
        bidx = torch.arange(n, device="cuda")
    batch_indices = torch.arange(32, device="cuda", dtype=bidx_dtype)
    batch_indices[:n] = bidx.to(bidx_dtype)
    next_draft = torch.randint(0, VOCAB, (n, d), generator=g, device="cuda", dtype=torch.int32)
    return accepted, num, batch_indices[:n], next_draft


def _stock_pair(bufs, ctrl, slots, accepted, num, bidx, next_draft, cpp):
    out = _stock_next_new_tokens(accepted, num, bidx, next_draft)
    rows = min(accepted.shape[0], slots.numel())
    dn, dx, dd, dl = bufs.store_views()
    assert m3_fold_tail.fold_tail(
        ctrl, slots, rows, accepted, dn, out, dx, next_draft, dd, num, dl, *bufs.engine(), out.shape[1], cpp=cpp
    )  # fmt: skip
    return out


def _fused(bufs, ctrl, slots, accepted, num, bidx, next_draft):
    dn, dx, dd, dl = bufs.store_views()
    ops = (accepted, num, bidx, next_draft, ctrl, slots, dn, dx, dd, dl, *bufs.engine())
    assert m3_nnt_fold.supports(*ops)
    return m3_nnt_fold.nnt_fold(*ops)


_CPP = pytest.param(
    True,
    id="cpp",
    marks=pytest.mark.skipif(not _sm107(), reason="the C++ fold tail op is the SM107 wheel's"),
)


@needs_gpu
@pytest.mark.parametrize("cpp", [False, _CPP], ids=["triton", "cpp"])
@pytest.mark.parametrize("bidx_dtype", [torch.int32, torch.int64], ids=["bidx32", "bidx64"])
@pytest.mark.parametrize("armed", ["all", "none", "part"])
@pytest.mark.parametrize(
    "n,d,widths,acc_extra",
    [
        (1, 7, (8, 8, 7), 0),  # b1 K7: the deployed TP4 step
        (2, 7, (8, 8, 7), 0),
        (8, 7, (8, 8, 7), 0),
        (1, 3, (8, 8, 7), 4),  # K3 into a K7-wide store, accepted buffer wider than D + 1
        (8, 3, (4, 4, 3), 0),
    ],
    ids=["b1k7", "b2k7", "b8k7", "b1k3wide", "b8k3"],
)
def test_nnt_fold_equals_the_stock_pair(n, d, widths, acc_extra, armed, bidx_dtype, cpp):
    g = torch.Generator(device="cuda").manual_seed(n * 100 + d * 10 + len(armed))
    t = d + 1
    accepted, num, bidx, next_draft = _step(n, d, g, d + 1 + acc_extra, bidx_dtype)
    rows_armed = {"all": n, "none": 0, "part": max(n - 1, 0)}[armed]
    ctrl = torch.tensor([rows_armed], dtype=torch.int32, device="cuda")
    slots = torch.randperm(NUM_SLOTS, generator=g, device="cuda")[:MAX_ROWS].long()
    ref_bufs = _Bufs(widths, t, g)
    got_bufs = ref_bufs.clone()
    want = _stock_pair(ref_bufs, ctrl, slots, accepted, num, bidx, next_draft, cpp)
    got = _fused(got_bufs, ctrl, slots, accepted, num, bidx, next_draft)
    torch.cuda.synchronize()
    assert got.dtype == want.dtype == torch.int32 and torch.equal(got, want)
    assert got_bufs.equal(ref_bufs), "a store or engine buffer differs from the stock pair"


@needs_gpu
def test_nnt_fold_in_a_cuda_graph():
    """One captured b1 K7 launch replayed over steps whose arming, slots and values change, against the eager pair."""
    g = torch.Generator(device="cuda").manual_seed(7)
    n, d, t = 1, 7, 8
    accepted, num, bidx, next_draft = _step(n, d, g)
    ctrl = torch.ones(1, dtype=torch.int32, device="cuda")
    slots = torch.zeros(MAX_ROWS, dtype=torch.long, device="cuda")
    bufs = _Bufs((8, 8, 7), t, g)
    _fused(bufs.clone(), ctrl, slots, accepted, num, bidx, next_draft)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            out = _fused(bufs, ctrl, slots, accepted, num, bidx, next_draft)
    torch.cuda.current_stream().wait_stream(stream)
    ref = bufs.clone()
    for step, (armed, slot) in enumerate([(1, 3), (0, 3), (1, 5), (1, 0), (1, 11)]):
        a2, n2, _, nd2 = _step(n, d, g)
        accepted.copy_(a2)
        num.copy_(n2)
        next_draft.copy_(nd2)
        ctrl.fill_(armed)
        slots[0] = slot
        want = _stock_pair(ref, ctrl, slots, accepted, num, bidx, next_draft, False)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want), step
        assert bufs.equal(ref), step


@needs_gpu
def test_supports_declines_what_the_pair_does_not_share():
    g = torch.Generator(device="cuda").manual_seed(3)
    accepted, num, bidx, next_draft = _step(2, 7, g)
    bufs = _Bufs((8, 8, 7), 8, g)
    ctrl = torch.ones(1, dtype=torch.int32, device="cuda")
    slots = torch.arange(MAX_ROWS, dtype=torch.long, device="cuda")
    dn, dx, dd, dl = bufs.store_views()
    base = dict(
        accepted=accepted, num_accepted=num, batch_idx=bidx, next_draft=next_draft, ctrl=ctrl, slots=slots,
        dst_new=dn, dst_next=dx, dst_draft=dd, dst_lens=dl, ids=bufs.ids, draft_ids=bufs.draft_ids,
        pos_off=bufs.pos_off, kv_off=bufs.kv_off,
    )  # fmt: skip
    assert m3_nnt_fold.supports(**base)
    wide = torch.zeros(2, 40, dtype=torch.int32, device="cuda")
    for change in (
        dict(accepted=accepted.long()),  # an int64 next_new_tokens: the fold needs int32
        dict(next_draft=next_draft.long()),
        dict(slots=slots.int()),
        dict(num_accepted=num.long()),
        dict(accepted=wide),  # past one 32-lane CTA
        dict(next_draft=torch.zeros(2, 32, dtype=torch.int32, device="cuda")),
        dict(accepted=accepted[:1]),  # rows differ from the drafts'
        dict(next_draft=next_draft.t().contiguous().t()),  # columns not contiguous
        dict(ids=bufs.ids[::2]),
        dict(ids=bufs.ids[:7]),  # too short for the armed rows
        dict(accepted=accepted.cpu()),
        dict(batch_idx=bidx.float()),
    ):
        assert not m3_nnt_fold.supports(**{**base, **change}), list(change)
    big = torch.zeros(m3_nnt_fold.MAX_ROWS + 1, 3, dtype=torch.int32, device="cuda")
    assert not m3_nnt_fold.supports(**{**base, "accepted": torch.zeros(65, 4, dtype=torch.int32, device="cuda"),
                                       "next_draft": big})  # fmt: skip


# ------------------------------------------------------------------------------------------- in the model


def _fold_classes():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        modeling,
    )

    return modeling


def _bound_fold(modeling, bufs):
    fold = modeling.M3DSparkStepFoldFwd(store_copy_wait=False, cpp_tail=False)
    fold.bind_engine_buffers(bufs.ids, bufs.draft_ids, bufs.pos_off, bufs.kv_off, MAX_ROWS)
    fold.bind_sampler_store(
        SimpleNamespace(
            new_tokens=bufs.new_tokens,
            next_new_tokens=bufs.next_new_tokens,
            next_draft_tokens=bufs.next_draft_tokens,
            new_tokens_lens=bufs.new_tokens_lens,
        )
    )
    return fold


@needs_gpu
def test_fold_object_over_steps_after_admissions(monkeypatch):
    """Two in-forward folds over the same run of decode steps: one takes the fused launch (then its tail skips), the
    other the stock next_new_tokens + store_in_forward. Every buffer, every begin_step / sampler_rows_stored answer and
    every next_new_tokens are equal, including the first step after an admission (new slots) and a step whose slots
    are the previous ones (staged)."""
    modeling = _fold_classes()
    m3_nnt_fold.precompile(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(11)
    d, t = 7, 8
    ref_bufs = _Bufs((8, 8, 7), t, g)
    got_bufs = ref_bufs.clone()
    ref, got = _bound_fold(modeling, ref_bufs), _bound_fold(modeling, got_bufs)
    tail_calls = []
    real_fold_tail = m3_fold_tail.fold_tail

    def counted(*a, **k):
        tail_calls.append(a[2])
        return real_fold_tail(*a, **k)

    monkeypatch.setattr(m3_fold_tail, "fold_tail", counted)
    runs = [[3], [3], [5], [5, 7], [5, 7], [2]]  # admissions at steps 2, 3 and 5
    for step, seq_slots in enumerate(runs):
        n = len(seq_slots)
        accepted, num, bidx, next_draft = _step(n, d, g)
        assert ref.begin_step(seq_slots, n) == got.begin_step(seq_slots, n), step
        before = len(tail_calls)
        want = _stock_next_new_tokens(accepted, num, bidx, next_draft)
        ref.store_in_forward(
            {
                "new_tokens": accepted,
                "next_new_tokens": want,
                "new_tokens_lens": num,
                "next_draft_tokens": next_draft,
            }
        )
        assert len(tail_calls) == before + 1
        out = got.fold_next_new_tokens(accepted, num, bidx, next_draft)
        assert out is not None, step
        outputs = {"new_tokens": accepted, "next_new_tokens": out, "new_tokens_lens": num,
                   "next_draft_tokens": next_draft}  # fmt: skip
        got.store_in_forward(outputs)
        assert len(tail_calls) == before + 1, (
            "the fused step's tail must not launch the fold tail again"
        )
        torch.cuda.synchronize()
        assert torch.equal(out, want), step
        assert got_bufs.equal(ref_bufs), step
        ref_out = {"new_tokens": accepted}
        assert ref.sampler_rows_stored(ref_out, 0, seq_slots) == got.sampler_rows_stored(
            outputs, 0, seq_slots
        )


@needs_gpu
def test_fold_object_runs_its_tail_for_other_outputs():
    """Outputs that are not the ones the fused launch stored (another new_tokens tensor) get the stock tail."""
    modeling = _fold_classes()
    m3_nnt_fold.precompile(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(5)
    bufs = _Bufs((8, 8, 7), 8, g)
    fold = _bound_fold(modeling, bufs)
    fold.begin_step([4], 1)
    accepted, num, bidx, next_draft = _step(1, 7, g)
    out = fold.fold_next_new_tokens(accepted, num, bidx, next_draft)
    assert out is not None and fold._tail_done is not None
    other = accepted.clone()
    seen = []
    fold.store_in_forward(
        {
            "new_tokens": other,
            "next_new_tokens": out,
            "new_tokens_lens": num,
            "next_draft_tokens": next_draft,
        }
    )
    seen.append(fold._eager_outputs is other)
    assert seen == [True] and fold._tail_done is None
    # An unbound fold declines without waiting or launching.
    bare = modeling.M3DSparkStepFoldFwd(store_copy_wait=False, cpp_tail=False)
    assert bare.fold_next_new_tokens(accepted, num, bidx, next_draft) is None
    assert "nnt_fold" in modeling.M3DSparkStepFoldFwd.STORE_WAIT_BEFORE


def _external_events() -> bool:
    """Whether this torch has the external CUDA events the deployed store wait uses (torch.cuda.Event(external=))."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.Event(external=True)
    except TypeError:
        return False
    return True


@needs_gpu
@pytest.mark.skipif(not _external_events(), reason="needs torch.cuda.Event(external=True)")
def test_fold_object_with_the_store_wait_in_a_cuda_graph(monkeypatch):
    """The deployed configuration (STEP_SPEC_STORE_COPY_SIDE: store_copy_wait=True): the store wait (an external-event
    wait node under capture) and then the fused launch, captured in one graph with the tail's store_in_forward (which
    skips its own launch), replayed over armed, re-armed and unarmed steps while the sampler copies the store to the
    host on its side stream (late, behind a sleep) and records the event. The replay must wait for that copy (the
    host copy holds the store before the replay's writes), and every buffer and next_new_tokens must equal the stock
    pair's."""
    modeling = _fold_classes()
    m3_nnt_fold.precompile(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(13)
    n, d, t = 1, 7, 8
    bufs = _Bufs((8, 8, 7), t, g)
    ref = bufs.clone()
    fold = modeling.M3DSparkStepFoldFwd(store_copy_wait=True, cpp_tail=False)
    fold.bind_engine_buffers(bufs.ids, bufs.draft_ids, bufs.pos_off, bufs.kv_off, MAX_ROWS)
    assert fold.store_copies_done is not None
    fold.bind_sampler_store(
        SimpleNamespace(
            new_tokens=bufs.new_tokens,
            next_new_tokens=bufs.next_new_tokens,
            next_draft_tokens=bufs.next_draft_tokens,
            new_tokens_lens=bufs.new_tokens_lens,
        )
    )
    accepted, num, bidx, next_draft = _step(n, d, g)
    tail_calls = []
    real_fold_tail = m3_fold_tail.fold_tail

    def counted(*a, **k):
        tail_calls.append(a[2])
        return real_fold_tail(*a, **k)

    monkeypatch.setattr(m3_fold_tail, "fold_tail", counted)
    fold.begin_step([3], n)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fold.fold_next_new_tokens(accepted, num, bidx, next_draft)
            assert out is not None
            fold.store_in_forward(
                {
                    "new_tokens": accepted,
                    "next_new_tokens": out,
                    "new_tokens_lens": num,
                    "next_draft_tokens": next_draft,
                }
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert tail_calls == [], "the captured tail must not launch the fold tail again"
    assert fold._tail_done is None and accepted.data_ptr() in fold._graph_outputs
    side = torch.cuda.Stream()
    host = torch.empty(bufs.new_tokens.shape, dtype=torch.int32).pin_memory()
    for step, seq_slots in enumerate([[3], [3], [5], None, [11], [0]]):
        a2, n2, _, nd2 = _step(n, d, g)
        accepted.copy_(a2)
        num.copy_(n2)
        next_draft.copy_(nd2)
        fold.begin_step(seq_slots, n)
        want = _stock_pair(ref, fold._ctrl, fold._slots, accepted, num, bidx, next_draft, False)
        before = bufs.new_tokens.clone()
        torch.cuda.synchronize()
        # The sampler: its host copy of the store on its side stream, late, then the event the tail waits on.
        with torch.cuda.stream(side):
            torch.cuda._sleep(20_000_000)
            host.copy_(bufs.new_tokens, non_blocking=True)
            fold.store_copies_done.record(side)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(host, before.cpu()), (
            step,
            "the replay rewrote the store before the host copy",
        )
        assert torch.equal(out, want), step
        assert bufs.equal(ref), step


def _dspark():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark,
    )

    return dspark


class _FakeFold:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def fold_next_new_tokens(self, accepted, num, bidx, next_draft):
        self.calls.append((accepted, num, bidx, next_draft))
        return self.result


def _bare_worker(dspark):
    from torch import nn

    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    nn.Module.__init__(w)
    w.mapping = SimpleNamespace(tp_rank=1)
    return w


@needs_gpu
def test_worker_takes_the_launch_on_decode_only_steps(monkeypatch):
    dspark = _dspark()
    g = torch.Generator(device="cuda").manual_seed(2)
    accepted, num, _, next_draft = _step(1, 7, g)
    batch_indices = torch.arange(32, dtype=torch.int32, device="cuda")
    sentinel = torch.full((1, 8), -9, dtype=torch.int32, device="cuda")
    lines = []
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: lines.append(msg))
    w = _bare_worker(dspark)
    fake = _FakeFold(sentinel)
    w.bind_tail_fold(fake)
    stock = _stock_next_new_tokens(accepted, num, batch_indices[:1], next_draft)
    # Mixed (context + generation) step: the forward tail does not store it, so the stock path runs.
    w._nnt_fold_step = False
    assert torch.equal(
        w._prepare_next_new_tokens(accepted, next_draft, batch_indices, 1, num), stock
    )
    assert fake.calls == []
    # Decode-only step: the fused launch.
    w._nnt_fold_step = True
    for _ in range(2):
        assert w._prepare_next_new_tokens(accepted, next_draft, batch_indices, 1, num) is sentinel
    assert len(fake.calls) == 2 and fake.calls[0][2].numel() == 1
    assert [m for m in lines if "TAILGLUE_B" in m] == [
        "[M3 dispatch] TAILGLUE_B taken: next_new_tokens + forward-tail fold in one launch (1 requests) on rank 1"
    ]
    # The fold declines (operands it does not take): the stock path.
    w.bind_tail_fold(_FakeFold(None))
    assert torch.equal(
        w._prepare_next_new_tokens(accepted, next_draft, batch_indices, 1, num), stock
    )
    # Kill switch.
    monkeypatch.setattr(dspark, "TAILGLUE_B", False)
    w.bind_tail_fold(fake)
    assert torch.equal(
        w._prepare_next_new_tokens(accepted, next_draft, batch_indices, 1, num), stock
    )
    assert len(fake.calls) == 2


def test_worker_forward_marks_decode_only_steps(monkeypatch):
    dspark = _dspark()
    seen = []

    def inner(self, *args, **kwargs):
        seen.append(self._nnt_fold_step)
        return "out"

    monkeypatch.setattr(dspark.S4WorkerMixin, "_forward_impl", inner)
    w = _bare_worker(dspark)
    w._tf = None
    assert w._forward_impl(attn_metadata=SimpleNamespace(num_contexts=0)) == "out"
    assert w._forward_impl(attn_metadata=SimpleNamespace(num_contexts=2)) == "out"
    assert (
        w._forward_impl(None, None, None, None, SimpleNamespace(num_contexts=0), None, None)
        == "out"
    )
    assert seen == [True, False, True] and w._nnt_fold_step is False


def _eval_switch(monkeypatch, env):
    """dspark.py's TAILGLUE_B assignment evaluated against a private copy of m31_switches under ``env``."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_tailglue_{id(env)}", _DIR / "m31_switches.py"
    )
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse((_DIR / "dspark.py").read_text())
    stmts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TAILGLUE_B" for t in n.targets)
    ]
    assert len(stmts) == 1
    ns = {"m31_disabled": switches.m31_disabled}
    exec(compile(ast.Module(stmts, []), "dspark.py", "exec"), ns)
    return ns["TAILGLUE_B"]


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_TAILGLUE_B": "0"}, True),
        ({"TRTLLM_M31_DISABLE_TAILGLUE_B": "1"}, False),
        # N3A's switch restores the in-tree next_new_tokens, so it turns the fused launch off too.
        ({"TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED": "1"}, False),
        ({"TRTLLM_M31_DISABLE_DRAFT_KV_REWIND_FUSED": "1"}, True),
    ],
)
def test_tailglue_b_switch(monkeypatch, env, want):
    assert _eval_switch(monkeypatch, env) is want
