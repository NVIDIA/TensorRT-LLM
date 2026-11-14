# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_prefill_store (DEP2CTX_DSPARK_STORE): CPU-only tests (no CUDA).

What is proven here:

1. Plumbing exactness: with the hook installed, DFlashWorker._store_prefill_context hands it, per chunk, exactly the
   (projected rows, slot-local start / end, block-table row, table, cache) the stock lines would have passed to
   precompute_context_kv and _store_context_kv_paged; host state after the call (ctx_len / base / stored, the
   request -> slot map, the continuation positions) is identical with and without the hook; a declined chunk runs
   the stock lines; overflow / no-slot / empty chunks never reach it; the vanilla (unpaged) arena never calls it.
2. Launch hygiene: with host positions the hooked loop never touches position_ids (the wasted .long() copy is gone);
   without them the first position is read exactly as stock reads it.
3. The wrapper: positions == cols == arange(start, end) int64 views of one persistent buffer (no allocation per
   chunk), num_accepted all-accepting, request_rows == [row], tokens_per_request == chunk length; geometric growth;
   it builds the drafter's fused-KV geometry before the support check; declines bad shapes.
4. Install routing and every kill switch, and the process-wide fold flag with its co-install guard (a decode store
   on the same worker keeps the flag; this install never writes it then).
5. Addressing oracle: the kernel's given-column page / slot arithmetic (dsparkCtxStore.cu generic build) equals
   flashinfer's append addressing (kv_indices[kv_indptr[b] + pos // page], pos % page) for the tables
   _ctx_paged_index_args builds, on prefill-shaped chunks, and no row is skipped as a shared-column write.

Items 1 and 2 drive the engine's DFlashWorker through its per-chunk prefill hook (enable_fused_prefill_store, with
the lean glue and decode-store hooks the installs read); they skip on an engine without those hooks.
"""

import collections
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as dcs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_prefill_store as dps,
)
from tensorrt_llm._torch.speculative import dflash

pytestmark = pytest.mark.cpu_only

DFlashWorker = dflash.DFlashWorker

_ENGINE_HOOKS = ("enable_fused_prefill_store", "enable_lean_glue", "enable_fused_ctx_store")
_MISSING_HOOKS = [h for h in _ENGINE_HOOKS if not hasattr(DFlashWorker, h)]
_needs_engine_hooks = pytest.mark.skipif(
    bool(_MISSING_HOOKS),
    reason="the stand-in worker models DFlashWorker with its prefill store hooks; this engine lacks "
    + ", ".join(f"DFlashWorker.{h}" for h in _MISSING_HOOKS),
)


class _Logger:
    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        return lambda msg, *a, **k: self.lines.append((level, str(msg)))


LOGGER = _Logger()


# ----------------------------------------------------------------------------------------------------------------------
# Stand-ins for the worker state, the drafter and the step's metadata
# ----------------------------------------------------------------------------------------------------------------------


class _TorchNoCuda:
    """dflash's torch with device="cuda" mapped to the CPU (the stock lines hardcode it)."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def arange(*a, device=None, **k):
        return torch.arange(*a, **k)

    @staticmethod
    def full(*a, device=None, **k):
        return torch.full(*a, **k)


class _Draft:
    """fc / hidden_norm present; projection is the identity; precompute_context_kv records its call."""

    fc = hidden_norm = object()

    def __init__(self):
        self.precompute = []

    def project_target_hidden(self, hs):
        return hs

    def precompute_context_kv(self, proj, pos):
        self.precompute.append((proj.clone(), pos.clone()))
        n = proj.shape[0]
        return torch.zeros(n, 2, 1, 4), torch.zeros(n, 2, 1, 4)


class _Hook:
    """The enable_fused_prefill_store contract, recorded: take (True) or decline (False) per call."""

    def __init__(self, take=lambda call: True):
        self.calls = []
        self.take = take

    def __call__(self, draft_model, projected, start, end, row, block_table, kv_buf):
        call = dict(
            proj=projected.clone(), start=start, end=end, row=row, table=block_table, kv=kv_buf
        )
        self.calls.append(call)
        return self.take(call)


def _worker(bound: bool, slots=5, max_ctx=256, page=16, paged=True, reuse_addressable=False):
    w = object.__new__(DFlashWorker)
    torch.nn.Module.__init__(w)
    w._ctx_paged = paged
    w._max_ctx = max_ctx
    w._ctx_page_size = page
    w._ctx_reuse_addressable = reuse_addressable
    w._req_ctx_pos = {}
    w._parked_ctx_base = collections.OrderedDict()
    w._req_to_slot = {}
    w._free_slots = collections.deque(range(slots))
    w._ctx_len_host = [0] * slots
    w._ctx_base_host = [0] * slots
    w._ctx_stored_host = [False] * slots
    w._ctx_len = torch.zeros(slots, dtype=torch.long)
    w._ctx_base_pages = torch.zeros(slots, dtype=torch.long)
    w._ctx_stored = torch.zeros(slots, dtype=torch.bool)
    pages = max_ctx // page
    w._ctx_kv_buf = [torch.zeros(slots * pages, 2, 1, page, 4)]
    w._ctx_page_table = torch.arange(slots * pages, dtype=torch.int32).view(slots, pages)
    w._ctx_block_tables = (
        torch.arange(slots * pages, dtype=torch.int32).flip(0).view(slots, pages).contiguous()
        if bound
        else None
    )
    w._ctx_block_counts = torch.full((slots,), pages, dtype=torch.long) if bound else None
    w._fused_prefill_store = None
    w._lean_glue = False
    w.stock_stores = []
    w._store_context_kv_paged = lambda k, v, rows, pos: w.stock_stores.append(
        (rows.clone(), pos.clone())
    )
    return w


def _step(chunks, hidden=8, host_pos=True):
    """chunks: [(req_id, first_pos, slen)] -> (spec_metadata, attn_metadata, position_ids, total tokens)."""
    n = sum(s for _, _, s in chunks)
    hs = torch.arange(n * hidden, dtype=torch.float32).view(n, hidden)
    pos = (
        torch.cat([torch.arange(f, f + s, dtype=torch.int32) for _, f, s in chunks])
        if n
        else torch.zeros(0)
    )
    spec = types.SimpleNamespace(
        request_ids=[r for r, _, _ in chunks],
        host_position_ids=pos.tolist() if host_pos else None,
        get_hidden_states=lambda total: hs,
    )
    attn = types.SimpleNamespace(
        num_ctx_tokens=n,
        num_contexts=len(chunks),
        _seq_lens=torch.tensor([s for _, _, s in chunks], dtype=torch.int32),
    )
    return spec, attn, pos, n


def _run(w, draft, chunks, host_pos=True, position_ids=None):
    spec, attn, pos, n = _step(chunks, host_pos=host_pos)
    w._store_prefill_context(draft, spec, attn, pos if position_ids is None else position_ids, n)


def _state(w):
    return (
        dict(w._req_to_slot),
        dict(w._req_ctx_pos),
        list(w._ctx_len_host),
        list(w._ctx_base_host),
        list(w._ctx_stored_host),
        w._ctx_len.tolist(),
        w._ctx_base_pages.tolist(),
        w._ctx_stored.tolist(),
        list(w._free_slots),
    )


# A schedule of steps: fresh requests, a chunked continuation, a reuse start (first_pos > 0 on a fresh slot), an
# overflow, an empty chunk, and a reused request id. It needs the worker's five slots: requests 10 to 13 (13 with
# an empty chunk) take four, and 14 takes the fifth.
STEPS = [
    [(10, 0, 37), (11, 0, 5)],
    [(10, 37, 64), (12, 48, 20), (13, 0, 0)],
    [(14, 0, 250), (11, 5, 1)],  # 14 fits max_ctx 256 and stores its 250 rows
    [
        (14, 250, 30),
        (10, 101, 3),
    ],  # 14: 250 + 30 > 256 -> overflow: it loses its slot and stores nothing
    [(11, 0, 9)],  # request id reused from 0: reset
]


@_needs_engine_hooks
def test_the_schedule_reaches_the_overflow():
    """Request 14 takes the fifth slot and stores 250 rows; its next chunk overflows (250 + 30 > 256), so it loses the
    slot and stores nothing more (request 10's chunk is that step's only store)."""
    w, d = _worker(False), _Draft()
    for step in STEPS[:3]:
        _run(w, d, step)
    row = w._req_to_slot[14]
    assert w._req_ctx_pos[14] == 250
    assert [len(rows) for rows, _ in w.stock_stores if bool((rows == row).all())] == [250]
    before = len(w.stock_stores)
    _run(w, d, STEPS[3])
    assert 14 not in w._req_to_slot and 14 not in w._req_ctx_pos
    assert [(len(rows), int(rows[0])) for rows, _ in w.stock_stores[before:]] == [
        (3, w._req_to_slot[10])
    ]


@_needs_engine_hooks
def test_with_every_slot_taken_a_new_request_stores_nothing():
    """Four slots: requests 10 to 13 hold them all, so 14 never gets a slot and none of its chunks is stored."""
    w, d = _worker(False, slots=4), _Draft()
    for step in STEPS[:4]:
        _run(w, d, step)
    assert 14 not in w._req_to_slot
    assert not [rows for rows, _ in w.stock_stores if len(rows) in (250, 30)]


@pytest.fixture(autouse=True)
def _cpu_torch(monkeypatch):
    monkeypatch.setattr(dflash, "torch", _TorchNoCuda())
    yield


@_needs_engine_hooks
@pytest.mark.parametrize("bound", [False, True])
def test_hook_gets_exactly_the_stock_store_inputs(bound):
    """Plumbing exactness: same per-chunk (rows, positions, row) and same host / device bookkeeping."""
    ref_w, ref_d = _worker(bound), _Draft()
    got_w, got_d = _worker(bound), _Draft()
    hook = _Hook()
    assert got_w.enable_fused_prefill_store(hook) is True
    for step in STEPS:
        _run(ref_w, ref_d, step)
        _run(got_w, got_d, step)
        assert _state(ref_w) == _state(got_w)
    assert (
        got_d.precompute == [] and got_w.stock_stores == []
    )  # every chunk taken: no stock line ran
    assert len(hook.calls) == len(ref_d.precompute) == len(ref_w.stock_stores) > 0
    for call, (proj, pos), (rows, spos) in zip(hook.calls, ref_d.precompute, ref_w.stock_stores):
        assert torch.equal(call["proj"], proj)
        assert torch.equal(torch.arange(call["start"], call["end"]), pos)
        assert torch.equal(pos, spos)
        assert torch.equal(rows, torch.full_like(rows, call["row"]))
        assert call["table"] is (got_w._ctx_block_tables if bound else got_w._ctx_page_table)
        assert call["kv"] is got_w._ctx_kv_buf


@_needs_engine_hooks
@pytest.mark.parametrize("bound", [False, True])
def test_declined_chunks_run_the_stock_lines(bound):
    ref_w, ref_d = _worker(bound), _Draft()
    got_w, got_d = _worker(bound), _Draft()
    got_w.enable_fused_prefill_store(_Hook(take=lambda c: c["start"] % 2 == 0))
    for step in STEPS:
        _run(ref_w, ref_d, step)
        _run(got_w, got_d, step)
        assert _state(ref_w) == _state(got_w)
    taken = [c for c in got_w._fused_prefill_store.calls if c["start"] % 2 == 0]
    assert len(got_d.precompute) + len(taken) == len(ref_d.precompute)
    assert 0 < len(got_d.precompute) < len(ref_d.precompute)


@_needs_engine_hooks
def test_no_hook_and_kill_restore_stock():
    """Hook None (the default and every kill switch): the stock lines, identical to the base tree."""
    w, d = _worker(True), _Draft()
    assert w.enable_fused_prefill_store(None) is False
    for step in STEPS:
        _run(w, d, step)
    assert len(d.precompute) == len(w.stock_stores) > 0


@_needs_engine_hooks
def test_unpaged_arena_never_calls_the_hook():
    w, d = _worker(False, paged=False), _Draft()
    hook = _Hook()
    w.enable_fused_prefill_store(hook)
    w._ctx_k_buf = torch.zeros(4, 2, 256, 1, 4)
    w._ctx_v_buf = torch.zeros(4, 2, 256, 1, 4)
    _run(w, d, STEPS[0])
    assert hook.calls == [] and len(d.precompute) == 2


class _NoTouch:
    def __getitem__(self, key):
        raise AssertionError("position_ids read although host positions exist")


@_needs_engine_hooks
def test_hooked_loop_never_touches_position_ids_with_host_positions():
    w, d = _worker(True), _Draft()
    w.enable_fused_prefill_store(_Hook())
    for step in STEPS:
        _run(w, d, step, position_ids=_NoTouch())


@_needs_engine_hooks
def test_first_position_without_host_positions_matches_stock():
    ref_w, ref_d = _worker(True), _Draft()
    got_w, got_d = _worker(True), _Draft()
    got_w.enable_fused_prefill_store(_Hook())
    for step in STEPS:
        _run(ref_w, ref_d, step, host_pos=False)
        _run(got_w, got_d, step, host_pos=False)
        assert _state(ref_w) == _state(got_w)


# ----------------------------------------------------------------------------------------------------------------------
# The wrapper
# ----------------------------------------------------------------------------------------------------------------------


class _Rec:
    def __init__(self, ret=True):
        self.args = []
        self.ret = ret

    def __call__(self, *a, **k):
        self.args.append((a, k))
        return self.ret


class _FakeDraft:
    def __init__(self):
        self._fused_kv_weight = None
        self.builds = 0

    def _build_fused_kv_buffers(self):
        self.builds += 1
        self._fused_kv_weight = torch.zeros(1)


def test_wrapper_maps_a_chunk_onto_the_decode_op(monkeypatch):
    monkeypatch.setattr(sys.modules["tensorrt_llm.logger"], "logger", LOGGER)
    LOGGER.lines.clear()
    rec = _Rec()
    s = dps.PrefillStore(store=rec)
    d = _FakeDraft()
    proj = torch.randn(7, 8)
    assert s(d, proj, 5, 12, 3, "table", "kv") is True
    assert d.builds == 1  # fused-KV geometry built before the support check
    ((a, k),) = rec.args
    model, projected, pos, nacc, rows, cols, table, kv, kp1 = a
    assert model is d and projected is proj and table == "table" and kv == "kv" and k == {}
    assert (
        pos.dtype == torch.int64 and torch.equal(pos, torch.arange(5, 12)) and pos.is_contiguous()
    )
    assert cols is pos  # given columns: no capacity clamp
    assert nacc.dtype == torch.int64 and nacc.tolist() == [2**31 - 1] and kp1 == 7
    assert rows.dtype == torch.int64 and rows.tolist() == [3]
    base = s._arange.data_ptr()
    assert s(d, torch.randn(4, 8), 0, 4, 0, "t", "k") and d.builds == 1
    a2 = rec.args[-1][0]
    assert (
        a2[2].data_ptr() == base and a2[4].data_ptr() == base
    )  # views of one buffer: nothing allocated per chunk
    assert s.calls == s.taken == 2
    taken = [m for _, m in LOGGER.lines if "DEP2CTX_DSPARK_STORE taken" in m]
    assert len(taken) == 1 and "n=7" in taken[0]  # one line per store, for its first chunk taken


def test_wrapper_grows_and_declines():
    rec = _Rec()
    s = dps.PrefillStore(store=rec)
    d = _FakeDraft()
    assert s(d, torch.randn(10, 8), 9000, 9010, 1, "t", "k")
    assert s._arange.numel() >= 9010 and torch.equal(rec.args[-1][0][2], torch.arange(9000, 9010))
    for args in (
        (torch.randn(3, 8), 0, 4, 0),
        (torch.randn(0, 8), 4, 4, 0),
        (torch.randn(2, 8), -1, 1, 0),
    ):
        assert s(d, *args, "t", "k") is False
    s2 = dps.PrefillStore(store=_Rec(ret=False))
    assert s2(d, torch.randn(2, 8), 0, 2, 0, "t", "k") is False and s2.taken == 0


def test_wrapper_declines_under_capture(monkeypatch):
    """The hook contract: while the stream captures, every chunk declines before anything allocates or launches (no
    geometry build, no index buffers, no store call), cold or with the buffers already grown; a graph-pool tensor must
    not outlive its graph. Eagerly again, the same store takes chunks."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    capturing = [True]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    rec = _Rec()
    s = dps.PrefillStore(store=rec)
    d = _FakeDraft()
    assert s(d, torch.randn(4, 8), 0, 4, 0, "t", "k") is False
    assert s._arange is None and s._nacc is None and d.builds == 0 and rec.args == []
    capturing[0] = False
    assert s(d, torch.randn(4, 8), 0, 4, 0, "t", "k") and d.builds == 1
    grown = s._arange
    capturing[0] = True
    for start in (8, grown.numel()):  # warm (inside the buffer) and cold (needs growth)
        assert s(d, torch.randn(4, 8), start, start + 4, 1, "t", "k") is False
        assert s._arange is grown and len(rec.args) == 1
    assert s.taken == 1
    capturing[0] = False
    assert s(d, torch.randn(4, 8), 8, 12, 1, "t", "k") and s.taken == 2


# ----------------------------------------------------------------------------------------------------------------------
# Install and kill switches
# ----------------------------------------------------------------------------------------------------------------------


class _W:
    def __init__(self):
        self.got = {}

    def enable_lean_glue(self, draft_model, enabled=True):
        self.got["S10"] = bool(enabled)

    def enable_fused_prefill_store(self, store):
        self.got["STORE"] = store
        return store is not None


@pytest.mark.parametrize(
    "off, want",
    [
        ((), dict(S10=True, STORE=True, FOLD=True)),
        (("DRAFTER_LEAN_GLUE",), dict(S10=False, STORE=True, FOLD=True)),
        (("DEP2CTX_DSPARK_STORE_FOLD",), dict(S10=True, STORE=True, FOLD=False)),
        (("DEP2CTX_DSPARK_STORE",), dict(S10=False, STORE=False, FOLD=False)),
    ],
)
def test_install_items_and_switches(off, want):
    w = _W()
    try:
        on = dps.install(w, object(), disabled=lambda item: item in off)
        assert on == want
        assert dcs._fold_w[0] is want["FOLD"]
        if "DEP2CTX_DSPARK_STORE" in off:
            assert w.got == {}  # nothing touched
        else:
            assert isinstance(w.got["STORE"], dps.PrefillStore) and w.got["S10"] is want["S10"]
        assert ("on" in dps.engagement(on)) is any(want.values())
    finally:
        dcs.fold_k_norm_weight(False)


def test_install_without_the_hook(monkeypatch):
    """No worker: nothing, silently. A worker without the hook (an older engine, another spec worker): nothing, and
    one line naming the worker type, so the install log does not blame a missing worker."""
    monkeypatch.setattr(sys.modules["tensorrt_llm.logger"], "logger", LOGGER)
    LOGGER.lines.clear()
    assert dps.install(None, None, disabled=lambda i: False) == {}
    assert LOGGER.lines == []

    class _StaleWorker:
        pass

    assert dps.install(_StaleWorker(), None, disabled=lambda i: False) == {}
    assert [m for _, m in LOGGER.lines] == [
        "[M3 dispatch] DEP2CTX_DSPARK_STORE not installed: _StaleWorker has no enable_fused_prefill_store"
    ]
    assert "enable_fused_prefill_store" in dps.engagement({})


@_needs_engine_hooks
def test_real_worker_install():
    w = _worker(True)
    try:
        on = dps.install(
            w, types.SimpleNamespace(_dflash_lean_glue=False), disabled=lambda i: False
        )
        assert on == dict(S10=True, STORE=True, FOLD=True)
        assert isinstance(w._fused_prefill_store, dps.PrefillStore) and w._lean_glue is True
    finally:
        dcs.fold_k_norm_weight(False)


@_needs_engine_hooks
def test_install_leaves_the_fold_to_a_decode_store(monkeypatch):
    """Co-install guard: with the decode store on the same worker, install never writes the
    process-wide fold flag (whatever that store's installer set stays) and reports the flag as it stands, since
    fused_ctx_store reads it for the prefill chunks too; one log line says whose flag it is. The FOLD kill switch is
    the decode installer's business then, so it changes neither the flag nor the report."""
    monkeypatch.setattr(sys.modules["tensorrt_llm.logger"], "logger", LOGGER)
    try:
        for prior in (False, True):
            for off in ((), ("DEP2CTX_DSPARK_STORE_FOLD",)):
                LOGGER.lines.clear()
                w = _worker(True)
                assert w.enable_fused_ctx_store(dcs.fused_ctx_store) is True
                dcs.fold_k_norm_weight(prior)
                on = dps.install(
                    w,
                    types.SimpleNamespace(_dflash_lean_glue=False),
                    disabled=lambda i, off=off: i in off,
                )
                assert on == dict(S10=True, STORE=True, FOLD=prior)
                assert dcs._fold_w[0] is prior  # untouched
                assert isinstance(w._fused_prefill_store, dps.PrefillStore)
                state = "on" if prior else "off"
                assert any(f"FOLD={state} (left to S10C_A)" in m for _, m in LOGGER.lines)
                assert f"FOLD={state}" in dps.engagement(on)
    finally:
        dcs.fold_k_norm_weight(False)


def test_op_kill_switch_declines_without_cuda(monkeypatch):
    """TRTLLM_M31_DISABLE_DSPARK_CTX_STORE_CPP=1: fused_ctx_store's support check fails, so the wrapper declines
    (the stock lines run). Checked at the module flag the env var sets at import."""
    monkeypatch.setattr(dcs, "_CPP_OFF", True)
    assert dcs._cpp_op() is None


# ----------------------------------------------------------------------------------------------------------------------
# Addressing oracle: kernel generic build (given columns) vs flashinfer append, as _ctx_paged_index_args sets it up
# ----------------------------------------------------------------------------------------------------------------------


def _flashinfer_cells(table, row, positions, page):
    indices = table.flatten().tolist()
    width = table.shape[1]
    indptr = [r * width for r in range(table.shape[0] + 1)]
    return [(indices[indptr[row] + p // page], p % page) for p in positions]


def _kernel_cells(table, row, positions, page):
    """dsparkCtxStoreGenericKernel with colFromPos false: col = cols[r]; a row whose next row has the same column
    and is not the request's last skips its write; page = table[row * tableStride + col / page]."""
    stride = table.stride(0)
    flat = table.flatten().tolist()
    out, n = [], len(positions)
    for r, col in enumerate(positions):
        last = r == n - 1
        nxt = positions[r] if last else positions[r + 1]
        if not (last or nxt != col):
            out.append(None)
            continue
        out.append((flat[row * stride + col // page], col % page))
    return out


@pytest.mark.parametrize("page", [16, 32, 64])
@pytest.mark.parametrize("start,n", [(0, 1), (0, 7), (5, 64), (31, 3500), (0, 16384), (8192, 4097)])
def test_kernel_addressing_is_flashinfer_append_addressing(page, start, n):
    rows, width = 3, -(-(start + n) // page) + 2
    table = (
        torch.randperm(rows * width, generator=torch.Generator().manual_seed(n))
        .int()
        .view(rows, width)
    )
    positions = list(range(start, start + n))
    for row in range(rows):
        k = _kernel_cells(table, row, positions, page)
        assert None not in k  # distinct consecutive columns: every row writes
        assert k == _flashinfer_cells(table, row, positions, page)
