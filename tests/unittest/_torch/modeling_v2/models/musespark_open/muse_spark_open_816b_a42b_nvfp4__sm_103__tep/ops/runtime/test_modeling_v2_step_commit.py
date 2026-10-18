# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step-commit launch and the executor hooks, bitwise against the built-in writes.

GPU. Each test builds the inputs of one built-in write group, runs the
built-in torch / C++ code on one set of buffers and the hook's launch on a
clone, and compares every word. The call sites are emulated as the executor
runs them (``_copy``): the hook gets the write first, and on a decline the
caller runs its own statement. ``_step_input_shadow.ShadowStager`` runs a window into
shadow buffers next to the built-in writes and compares them; it is checked
here against the built-in branch and against a planted dropped write. These
cover the kernel and the edge cases a serving run does not reach on demand
(padding, several slots, page holes, stale tails).
"""

from __future__ import annotations

import math
import types
from collections import Counter

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.runtime import (  # noqa: E501
    step_commit as sc,
)

__extra_import_path__ = ["."]
from _step_input_shadow import ShadowStager  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ptr(t, offset=0):
    return t.data_ptr() + offset * t.element_size()


def _window(runner=None, attn=None, staging=None):
    """One step-input window over ``staging`` (a fresh block by default), as the executor opens it."""
    staging = staging if staging is not None else sc.StepStaging()
    staging.reset()
    return runtime._PrepareStager(staging, Counter(), runner, attn, None)


def _copy(stager, dst, src):
    """A copy call site of the executor: the hook first, the original statement on a decline."""
    if stager is None or not stager.copy(dst, src):
        dst.copy_(src, non_blocking=True)


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------
def test_copy_gather_fill_blocks_against_torch():
    g = torch.Generator().manual_seed(0)
    staging = sc.StepStaging(max_descriptors=64, data_bytes=1 << 16)
    # COPY: int32 and int64 payloads (int64 as two words each), 3000 elements -> 3 chunks.
    a32 = torch.randint(-(2**31), 2**31 - 1, (3000,), generator=g, dtype=torch.int64).to(
        torch.int32
    )
    a64 = torch.randint(-(2**62), 2**62, (17,), generator=g, dtype=torch.int64)
    d32 = torch.full((3001,), -5, dtype=torch.int32, device="cuda")
    d64 = torch.zeros(20, dtype=torch.int64, device="cuda")
    staging.copy_bytes(_ptr(d32), a32.view(torch.uint8).numpy())
    staging.copy_bytes(_ptr(d64, 2), a64.view(torch.uint8).numpy())
    # GATHER: strided source, strided destination, add.
    src = torch.randint(-1000, 1000, (64, 9), generator=g, dtype=torch.int32).cuda()
    dst = torch.full((8, 40), 3, dtype=torch.int32, device="cuda")
    staging.gather(_ptr(dst, 1), 40, _ptr(src, 5 * 9 + 2), 9, 8, add=-7)
    bcast = torch.zeros(40, dtype=torch.int32, device="cuda")
    staging.gather(_ptr(bcast), 1, _ptr(src, 2 * 9), 0, 40)  # one word broadcast
    # FILL with stride.
    fill = torch.ones(100, dtype=torch.int32, device="cuda")
    staging.fill(_ptr(fill, 4), 3, 30, -1234)
    # BLOCKS: page indices with BAD_PAGE_INDEX holes, read from pinned host memory.
    pages = torch.randint(0, 2**20, (2500,), generator=g, dtype=torch.int32)
    pages[::7] = -1
    pages = pages.pin_memory()
    bk = torch.full((2600,), 9, dtype=torch.int32, device="cuda")
    bv = torch.full((2600,), 9, dtype=torch.int32, device="cuda")
    staging.blocks(_ptr(bk), _ptr(bv), _ptr(pages), 2500, 6, 3)
    staging.launch()
    torch.cuda.synchronize()

    assert torch.equal(d32[:3000].cpu(), a32) and d32[3000].item() == -5
    assert torch.equal(d64[2:19].cpu(), a64) and d64[:2].abs().sum() == 0 and d64[19] == 0
    exp = torch.full((8, 40), 3, dtype=torch.int32)
    exp.view(-1)[1 : 1 + 8 * 40 : 40] = src.cpu().view(-1)[5 * 9 + 2 : 5 * 9 + 2 + 8 * 9 : 9] - 7
    assert torch.equal(dst.cpu(), exp)
    assert (bcast == src[2, 0]).all()
    exp_fill = torch.ones(100, dtype=torch.int32)
    exp_fill[4 : 4 + 30 * 3 : 3] = -1234
    assert torch.equal(fill.cpu(), exp_fill)
    bad = pages == -1
    assert torch.equal(bk[:2500].cpu(), torch.where(bad, 0, pages * 6))
    assert torch.equal(bv[:2500].cpu(), torch.where(bad, 0, pages * 6 + 3))
    assert (bk[2500:] == 9).all() and (bv[2500:] == 9).all()
    staging.reset()


def test_launch_runs_only_pending_and_orders_launches():
    staging = sc.StepStaging(max_descriptors=8, data_bytes=256)
    buf = torch.zeros(4, dtype=torch.int32, device="cuda")
    staging.fill(_ptr(buf), 1, 4, 1)
    staging.launch()
    # A second launch in the same step writes after the first.
    staging.fill(_ptr(buf, 2), 1, 2, 2)
    assert staging.pending == 1
    staging.launch()
    assert not staging.launch()
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 1, 2, 2]
    staging.reset()
    assert staging.num == 0 and staging.data_used == 0


# --------------------------------------------------------------------------
# Block offsets vs the C++ copy
# --------------------------------------------------------------------------
def test_blocks_match_cpp_copy_on_readable_columns():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    g = torch.Generator().manual_seed(1)
    pools, rows, nb = 2, 6, 1024
    host = torch.randint(0, 2**22, (pools, rows, 2, nb), generator=g, dtype=torch.int32)
    host[:, :, :, ::5] = -1
    host = host.pin_memory()
    scales = torch.tensor([4, 2], dtype=torch.int32).pin_memory()
    kvoff = torch.tensor([1, 77], dtype=torch.int32).pin_memory()
    copy_idx = torch.tensor([4, 1, 3], dtype=torch.int32).pin_memory()
    stale = torch.randint(0, 2**22, (pools, 5, 2, nb), generator=g, dtype=torch.int32).cuda()
    legacy = stale.clone()
    mine = stale.clone()
    copy_batch_block_offsets_to_device(
        host, legacy, copy_idx, scales, kvoff, torch.cuda.current_stream().cuda_stream
    )
    cols = 301
    staging = sc.StepStaging()
    for p in range(pools):
        for s, row in enumerate(copy_idx.tolist()):
            src = _ptr(host, (p * rows + row) * 2 * nb)
            k = _ptr(mine, (p * 5 + s) * 2 * nb)
            v = _ptr(mine, ((p * 5 + s) * 2 + 1) * nb)
            staging.blocks(k, v, src, cols, int(scales[p]), int(kvoff[p]))
    staging.launch()
    torch.cuda.synchronize()
    assert torch.equal(mine[:, :3, :, :cols], legacy[:, :3, :, :cols])
    assert torch.equal(mine[:, :3, :, cols:], stale[:, :3, :, cols:])
    assert torch.equal(mine[:, 3:], legacy[:, 3:])


# --------------------------------------------------------------------------
# The window's overlap-scheduler branch vs the built-in branch
# --------------------------------------------------------------------------
def _legacy_previous_tensors(
    r,
    new_tokens_device,
    new_tokens_lens_device,
    next_draft_tokens_device,
    previous_batch_indices,
    previous_pos_indices,
    num_tokens,
    num_draft_tokens,
    T,
    E,
):
    """The built-in branch of ``DecoderRunner._prepare_tp_inputs``, statement for statement."""
    r.previous_pos_id_offsets_cuda *= 0
    r.previous_kv_lens_offsets_cuda *= 0
    W = T - 1
    P = len(previous_batch_indices)
    if P > 0:
        previous_slots = r.previous_batch_indices_cuda[:P]
        previous_slots.copy_(
            torch.tensor(previous_batch_indices, dtype=torch.int).pin_memory(), non_blocking=True
        )
        new_tokens = new_tokens_device.transpose(0, 1)[previous_slots, :T].flatten()
        r.input_ids_cuda[num_tokens : num_tokens + P * T].copy_(new_tokens, non_blocking=True)
        if W > 0:
            r.draft_tokens_cuda[num_draft_tokens : num_draft_tokens + P * W].copy_(
                next_draft_tokens_device[previous_slots, :W].flatten(), non_blocking=True
            )
        kv_len_offsets_device = new_tokens_lens_device - T
        r.previous_pos_indices_cuda[0 : P * T].copy_(
            torch.tensor(previous_pos_indices, dtype=torch.int).pin_memory(), non_blocking=True
        )
        r.previous_pos_id_offsets_cuda[(E - P) * T : E * T].copy_(
            new_tokens_lens_device[r.previous_pos_indices_cuda[0 : P * T]], non_blocking=True
        )
        r.previous_kv_lens_offsets_cuda[E - P : E].copy_(
            kv_len_offsets_device[previous_slots], non_blocking=True
        )


def _fake_runner(seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(n):
        return torch.randint(-50, 50, (n,), generator=g, dtype=torch.int32).cuda()

    return types.SimpleNamespace(
        previous_pos_id_offsets_cuda=rnd(256),
        previous_kv_lens_offsets_cuda=rnd(8),
        previous_batch_indices_cuda=rnd(256),
        previous_pos_indices_cuda=rnd(256),
        input_ids_cuda=rnd(256),
        draft_tokens_cuda=rnd(64),
    )


def _buffers(r):
    return {k: v for k, v in vars(r).items() if isinstance(v, torch.Tensor)}


def _clone_runner(r):
    return types.SimpleNamespace(**{k: v.clone() for k, v in _buffers(r).items()})


def _previous_tensors(T=8, S=4, D=7, seed=7):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 200000, (T, S, 1), generator=g, dtype=torch.int32).cuda(),
        torch.randint(1, 9, (S,), generator=g, dtype=torch.int32).cuda(),
        torch.randint(0, 200000, (S, D), generator=g, dtype=torch.int32).cuda(),
    )


@pytest.mark.parametrize(
    "slots, num_tokens, num_draft_tokens, E, T",
    [
        ([0], 0, 0, 1, 8),
        ([3, 1], 0, 0, 2, 8),
        ([2], 5, 7, 3, 8),
        ([1, 0, 2], 0, 0, 3, 1),
        ([], 0, 0, 0, 8),
    ],
)
def test_overlap_inputs_matches_builtin(slots, num_tokens, num_draft_tokens, E, T):
    new_tokens_device, new_tokens_lens, next_draft = _previous_tensors(T=max(T, 8))
    pos_idx = [s for s in slots for _ in range(T)]
    base = _fake_runner(3)
    legacy, mine = _clone_runner(base), _clone_runner(base)
    _legacy_previous_tensors(
        legacy,
        new_tokens_device,
        new_tokens_lens,
        next_draft,
        slots,
        pos_idx,
        num_tokens,
        num_draft_tokens,
        T,
        E,
    )
    stager = _window(mine)
    assert stager.overlap_inputs(
        mine,
        new_tokens_device=new_tokens_device,
        new_tokens_lens_device=new_tokens_lens,
        next_draft_tokens_device=next_draft,
        previous_batch_indices=slots,
        previous_pos_indices=pos_idx,
        num_tokens=num_tokens,
        num_draft_tokens=num_draft_tokens,
        tokens_per_gen_step=T,
        num_extend_requests=E,
    )
    stager.staging.launch()
    torch.cuda.synchronize()
    for name in _buffers(base):
        assert torch.equal(getattr(mine, name), getattr(legacy, name)), name


def test_copy_stager_and_hook_off():
    dst = torch.zeros(5, dtype=torch.int32, device="cuda")
    _copy(None, dst, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4, 5]

    stager = _window()
    d64 = torch.zeros(3, dtype=torch.int64, device="cuda")
    _copy(stager, dst[1:4], torch.tensor([7, 8, 9], dtype=torch.int64))  # int64 -> int32
    _copy(stager, d64, torch.tensor([2**40, -1, 5], dtype=torch.int64).pin_memory())
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4, 5], "a staged write must not land before the launch"
    # A write the stager cannot take: what was recorded is issued first, then the caller's copy.
    f32 = torch.zeros(2, dtype=torch.float32, device="cuda")
    _copy(stager, f32, torch.tensor([1, 2], dtype=torch.int32))
    assert stager.declined
    stager.end({})
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 7, 8, 9, 5]
    assert d64.tolist() == [2**40, -1, 5]
    assert f32.tolist() == [1.0, 2.0]


def _static_step(seed, positions_seed=None):
    g = torch.Generator().manual_seed(seed)
    S, D, T = 2, 7, 8
    r = _fake_runner(seed + 4)
    r.position_ids_cuda = (
        torch.randint(0, 9, (256,), generator=g, dtype=torch.int32).cuda()
        if positions_seed is None
        else torch.zeros(256, dtype=torch.int32, device="cuda")
    )
    static = {
        "input_ids": torch.ones(64, dtype=torch.int32, device="cuda"),
        "position_ids": torch.zeros((1, 64), dtype=torch.int32, device="cuda"),
    }
    r.cuda_graph_runner = types.SimpleNamespace(shared_static_tensors=static)
    tokens = torch.randint(0, 200000, (T, S, 1), generator=g, dtype=torch.int32).cuda()
    lens = torch.randint(1, 9, (S,), generator=g, dtype=torch.int32).cuda()
    drafts = torch.randint(0, 200000, (S, D), generator=g, dtype=torch.int32).cuda()
    overlap = dict(
        new_tokens_device=tokens,
        new_tokens_lens_device=lens,
        next_draft_tokens_device=drafts,
        previous_batch_indices=[1],
        previous_pos_indices=[1] * T,
        num_tokens=0,
        num_draft_tokens=0,
        tokens_per_gen_step=T,
        num_extend_requests=1,
    )
    return r, static, tokens, overlap, T


def test_prepare_writes_static_inputs_and_replay_skips_its_copies():
    """A step whose recorded writes cover input_ids[:n] and position_ids[:n] also fills the static buffers."""
    r, static, tokens, overlap, T = _static_step(5)
    static_before = {k: v.clone() for k, v in static.items()}
    stager = _window(r, types.SimpleNamespace(padded_num_tokens=None))
    assert stager.overlap_inputs(r, **overlap)
    positions = torch.arange(1000, 1000 + T, dtype=torch.int32).pin_memory()
    _copy(stager, r.position_ids_cuda[:T], positions)
    inputs = {
        "input_ids": r.input_ids_cuda[:T],
        "position_ids": r.position_ids_cuda[:T].unsqueeze(0),
    }
    stager.end(inputs)
    torch.cuda.synchronize()
    assert torch.equal(static["input_ids"][:T], tokens[:, 1, 0])
    assert torch.equal(r.input_ids_cuda[:T], tokens[:, 1, 0])
    assert torch.equal(static["position_ids"][0, :T], positions.cuda())
    assert torch.equal(static["input_ids"][T:], static_before["input_ids"][T:])
    assert stager.static_inputs() == (
        inputs["input_ids"].data_ptr(),
        inputs["position_ids"].data_ptr(),
        T,
    )


def test_static_mirror_falls_back_when_block_is_full():
    r, static, tokens, overlap, T = _static_step(23, positions_seed=0)
    static_before = {k: v.clone() for k, v in static.items()}
    # The previous-tensor group asks room for 10 (up to 4 fills) and records 8 (2 fills, 2 uploads,
    # 4 gathers); the position copy makes 9; the mirror's 2 rows do not fit in 10.
    stager = _window(
        r,
        types.SimpleNamespace(padded_num_tokens=None),
        staging=sc.StepStaging(max_descriptors=10, data_bytes=1024),
    )
    assert stager.overlap_inputs(r, **overlap)
    _copy(stager, r.position_ids_cuda[:T], torch.arange(5, 5 + T, dtype=torch.int32))
    inputs = {
        "input_ids": r.input_ids_cuda[:T],
        "position_ids": r.position_ids_cuda[:T].unsqueeze(0),
    }
    stager.end(inputs)
    torch.cuda.synchronize()
    assert stager.counts["static inputs copied by replay"] == 1
    assert stager.static_inputs() is None
    assert all(torch.equal(static[k], static_before[k]) for k in static)
    assert torch.equal(r.input_ids_cuda[:T], tokens[:, 1, 0])


# --------------------------------------------------------------------------
# The sampler store
# --------------------------------------------------------------------------
def _store(g, Tn, Tnn, D, num_slots):
    return types.SimpleNamespace(
        new_tokens=torch.randint(0, 9, (Tn, num_slots, 1), generator=g, dtype=torch.int32).cuda(),
        next_new_tokens=torch.randint(
            0, 9, (Tnn, num_slots, 1), generator=g, dtype=torch.int32
        ).cuda(),
        new_tokens_lens=torch.randint(0, 9, (num_slots,), generator=g, dtype=torch.int32).cuda(),
        next_draft_tokens=torch.randint(
            0, 9, (num_slots, D), generator=g, dtype=torch.int32
        ).cuda(),
    )


@pytest.mark.parametrize("slots, num_slots", [([0], 1), ([2], 4), ([3, 0], 4), ([1, 2, 0, 3], 4)])
def test_store_and_copy_out_matches_index_copy_and_snapshots(slots, num_slots):
    g = torch.Generator().manual_seed(11)
    Tn, Tnn, D = 8, 8, 7
    n = len(slots)
    base = _store(g, Tn, Tnn, D, num_slots)
    rows = 2 * n + 1  # output rows wider than the sampled ones, sliced like sample_async does
    outs = [
        torch.randint(10, 99, shape, generator=g, dtype=torch.int32).cuda()
        for shape in ((rows, Tn), (rows, Tnn), (rows,), (rows, D))
    ]
    o = [t[1 : 1 + n] for t in outs]

    legacy = types.SimpleNamespace(**{k: v.clone() for k, v in vars(base).items()})
    sl = torch.tensor(slots, dtype=torch.long, device="cuda")
    legacy.new_tokens.squeeze(-1).T.index_copy_(0, sl, o[0])
    legacy.next_new_tokens.squeeze(-1).T.index_copy_(0, sl, o[1])
    legacy.new_tokens_lens.index_copy_(0, sl, o[2])
    legacy.next_draft_tokens.index_copy_(0, sl, o[3])
    legacy_host = [
        t.cpu() for t in (legacy.new_tokens, legacy.new_tokens_lens, legacy.next_draft_tokens)
    ]

    mine_store = types.SimpleNamespace(**{k: v.clone() for k, v in vars(base).items()})
    sampler = types.SimpleNamespace(store=mine_store, _async_worker_active=lambda: False)
    host = runtime.SampleStoreWriter().store_and_copy_out(sampler, slots, *o, None)
    assert host is not None
    torch.cuda.synchronize()
    for k in vars(base):
        assert torch.equal(getattr(mine_store, k), getattr(legacy, k)), k
    for mine, ref in zip(host, legacy_host):
        assert mine.is_pinned() and torch.equal(mine, ref)


def test_store_and_copy_out_declines_outside_its_case():
    writer = runtime.SampleStoreWriter()
    st = types.SimpleNamespace(
        new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32, device="cuda"),
        next_new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32, device="cuda"),
        new_tokens_lens=torch.zeros((2,), dtype=torch.int32, device="cuda"),
        next_draft_tokens=torch.zeros((2, 7), dtype=torch.int32, device="cuda"),
    )
    sampler = types.SimpleNamespace(store=st, _async_worker_active=lambda: False)
    o = (torch.zeros((1, 8), dtype=torch.int32, device="cuda"),) * 2 + (
        torch.zeros((1,), dtype=torch.int32, device="cuda"),
        torch.zeros((1, 7), dtype=torch.int32, device="cuda"),
    )
    verify = torch.ones(1, dtype=torch.int32, device="cuda")
    assert writer.store_and_copy_out(sampler, [0], *o, verify) is None
    assert writer.store_and_copy_out(sampler, [5], *o, None) is None
    sampler._async_worker_active = lambda: True
    assert writer.store_and_copy_out(sampler, [0], *o, None) is None
    assert math.isclose(
        sum(v for k, v in writer.counts.items() if k.startswith("sample declined")), 3
    )


def test_store_and_copy_out_declines_on_full_block():
    writer = runtime.SampleStoreWriter()
    writer._staging = [sc.StepStaging(max_descriptors=4), sc.StepStaging(max_descriptors=4)]
    st = types.SimpleNamespace(
        new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32, device="cuda"),
        next_new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32, device="cuda"),
        new_tokens_lens=torch.zeros((2,), dtype=torch.int32, device="cuda"),
        next_draft_tokens=torch.zeros((2, 7), dtype=torch.int32, device="cuda"),
    )
    sampler = types.SimpleNamespace(store=st, _async_worker_active=lambda: False)
    o = (torch.ones((1, 8), dtype=torch.int32, device="cuda"),) * 2 + (
        torch.ones((1,), dtype=torch.int32, device="cuda"),
        torch.ones((1, 7), dtype=torch.int32, device="cuda"),
    )
    assert writer.store_and_copy_out(sampler, [1], *o, None) is None  # needs 3 * 2 + 4 = 10
    assert writer.counts["sample declined: staging block full"] == 1
    assert all(int(t.abs().sum()) == 0 for t in vars(st).values()), "nothing written"


# --------------------------------------------------------------------------
# Capacity: a full staging block declines (the caller writes eagerly, in order)
# and the block grows at the next reset
# --------------------------------------------------------------------------
def test_staging_room_and_growth():
    staging = sc.StepStaging(max_descriptors=4, data_bytes=64)
    assert staging.capacity == (4, 64)
    assert staging.room(4) and staging.room(1, 32, 1)
    assert not staging.short
    assert not staging.room(5)
    assert staging.short
    staging.reset()
    assert staging.capacity == (8, 128) and not staging.short
    assert not staging.room(1, 128, 1)  # alignment padding counts
    staging.reset()
    assert staging.capacity == (16, 256)
    big = sc.StepStaging(max_descriptors=sc.MAX_DESCRIPTORS, data_bytes=sc.MAX_DATA_BYTES)
    assert not big.room(sc.MAX_DESCRIPTORS + 1)
    big.reset()
    assert big.capacity == (sc.MAX_DESCRIPTORS, sc.MAX_DATA_BYTES)
    # Recording past the end without asking is a programming error.
    tiny = sc.StepStaging(max_descriptors=1, data_bytes=16)
    buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    tiny.fill(_ptr(buf), 1, 1, 1)
    with pytest.raises(RuntimeError):
        tiny.fill(_ptr(buf), 1, 1, 2)


def test_copy_declines_on_full_block_in_order():
    """Overlapping writes across a decline land in program order, as the built-in copies would."""
    staging = sc.StepStaging(max_descriptors=2, data_bytes=1024)
    stager = _window(staging=staging)
    dst = torch.zeros(8, dtype=torch.int32, device="cuda")
    _copy(stager, dst[0:4], torch.full((4,), 1, dtype=torch.int32))  # staged
    _copy(stager, dst[4:8], torch.full((4,), 2, dtype=torch.int32))  # staged
    _copy(stager, dst[2:6], torch.full((4,), 3, dtype=torch.int32))  # full: launch, then eager
    _copy(stager, dst[0:2], torch.full((2,), 4, dtype=torch.int32))  # after a decline: eager
    stager.end({})
    torch.cuda.synchronize()
    assert dst.tolist() == [4, 4, 3, 3, 3, 3, 2, 2]
    assert stager.declined == ["staging block full"]
    assert staging.num == 2, "nothing recorded after the decline"
    staging.reset()
    assert staging.capacity == (4, 2048), "grows for the next step"


def test_overlap_inputs_declines_on_full_block():
    T, slots = 8, [2, 0]
    new_tokens_device, lens, drafts = _previous_tensors(seed=17)
    pos_idx = [s for s in slots for _ in range(T)]
    base = _fake_runner(21)
    legacy, mine = _clone_runner(base), _clone_runner(base)
    _legacy_previous_tensors(legacy, new_tokens_device, lens, drafts, slots, pos_idx, 0, 0, T, 2)
    # needs 4 + 2 + 2 * 4
    stager = _window(mine, staging=sc.StepStaging(max_descriptors=8, data_bytes=1024))
    kw = dict(
        new_tokens_device=new_tokens_device,
        new_tokens_lens_device=lens,
        next_draft_tokens_device=drafts,
        previous_batch_indices=slots,
        previous_pos_indices=pos_idx,
        num_tokens=0,
        num_draft_tokens=0,
        tokens_per_gen_step=T,
        num_extend_requests=2,
    )
    if not stager.overlap_inputs(mine, **kw):
        _legacy_previous_tensors(mine, new_tokens_device, lens, drafts, slots, pos_idx, 0, 0, T, 2)
    stager.end({})
    torch.cuda.synchronize()
    assert stager.declined == ["staging block full"] and stager.staging.num == 0
    for name in _buffers(base):
        assert torch.equal(getattr(mine, name), getattr(legacy, name)), name


def test_overlap_inputs_declines_unexpected_position_indices():
    T = 8
    r = _fake_runner(4)
    stager = _window(r)
    tokens = torch.zeros((T, 4, 1), dtype=torch.int32, device="cuda")
    lens = torch.ones(4, dtype=torch.int32, device="cuda")
    drafts = torch.zeros((4, 7), dtype=torch.int32, device="cuda")
    kw = dict(
        new_tokens_device=tokens,
        new_tokens_lens_device=lens,
        next_draft_tokens_device=drafts,
        previous_batch_indices=[1, 3],
        num_tokens=0,
        num_draft_tokens=0,
        tokens_per_gen_step=T,
        num_extend_requests=2,
    )
    bad = [1] * T + [3] * (T - 1) + [1]
    assert not stager.overlap_inputs(r, previous_pos_indices=bad, **kw)
    assert stager.declined and "position indices" in stager.declined[0]
    assert stager.staging.num == 0
    assert not stager.overlap_inputs(r, previous_pos_indices=[1] * T + [3] * (T + 1), **kw)
    ok = _window(r)
    assert ok.overlap_inputs(r, previous_pos_indices=[1] * T + [3] * T, **kw)


def _fake_kv_manager(seed, pools=2, rows=6, nb=512, capacities=(1000, 70)):
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp

    g = torch.Generator().manual_seed(seed)
    host = torch.randint(0, 2**20, (pools, rows, 2, nb), generator=g, dtype=torch.int32)
    host[:, :, :, ::9] = -1
    host = host.pin_memory()
    copy_rows = torch.tensor([3, 0], dtype=torch.int32)
    return types.SimpleNamespace(
        host_kv_cache_block_offsets=host,
        sparse_metadata_batch=None,
        _use_per_layer_page_tables=False,
        max_copy_beam_width=1,
        kv_cache_type=CacheTypeCpp.SELF,
        tokens_per_block=32,
        kv_cache_map={
            10: types.SimpleNamespace(capacity=capacities[0]),
            11: types.SimpleNamespace(capacity=capacities[1]),
        },
        index_mapper=types.SimpleNamespace(get_copy_index=lambda ids, nctx, bw, flag: copy_rows),
        _index_scale_ints=[4, 2],
        kv_offset=torch.tensor([1, 77], dtype=torch.int32),
    )


@pytest.mark.parametrize("room", [True, False])
def test_block_offsets_stager_matches_cpp_or_declines(room):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    manager = _fake_kv_manager(31)
    host, nb = manager.host_kv_cache_block_offsets, manager.host_kv_cache_block_offsets.shape[3]
    stale = torch.randint(0, 2**20, (2, 4, 2, nb), dtype=torch.int32).cuda()
    legacy, mine = stale.clone(), stale.clone()
    copy_idx = torch.tensor([3, 0], dtype=torch.int32).pin_memory()
    scales = torch.tensor([4, 2], dtype=torch.int32).pin_memory()
    kvoff = torch.tensor([1, 77], dtype=torch.int32).pin_memory()
    copy_batch_block_offsets_to_device(
        host, legacy, copy_idx, scales, kvoff, torch.cuda.current_stream().cuda_stream
    )
    stager = _window(staging=sc.StepStaging(max_descriptors=4 if room else 3, data_bytes=256))
    staged = stager.block_offsets(manager, mine, [10, 11], 1, 0, 2)
    if not staged:
        copy_batch_block_offsets_to_device(
            host, mine, copy_idx, scales, kvoff, torch.cuda.current_stream().cuda_stream
        )
    stager.end({})
    torch.cuda.synchronize()
    assert staged is room
    cols = math.ceil(1000 / 32) + 1
    if room:
        assert stager.block_cols == cols and not stager.declined
        assert torch.equal(mine[:, :2, :, :cols], legacy[:, :2, :, :cols])
        assert torch.equal(mine[:, :2, :, cols:], stale[:, :2, :, cols:])
        assert torch.equal(mine[:, 2:], stale[:, 2:])
    else:
        assert stager.declined == ["staging block full"]
        assert torch.equal(mine, legacy)


# --------------------------------------------------------------------------
# The test-only shadow: the window's launch into shadow buffers next to the built-in writes
# --------------------------------------------------------------------------
def _shadow_step(drop_last_gather=False):
    T, slots = 8, [3, 1]
    new_tokens_device, lens, drafts = _previous_tensors(seed=29)
    pos_idx = [s for s in slots for _ in range(T)]
    r = _fake_runner(13)
    # A value the built-in branch never writes, so a dropped offset write shows.
    r.previous_kv_lens_offsets_cuda.fill_(1000)
    staging = sc.StepStaging()
    if drop_last_gather:
        recorded = staging.gather

        def gather(*args, **kwargs):
            gather.calls += 1
            if gather.calls < 4 * len(slots):
                recorded(*args, **kwargs)

        gather.calls = 0
        staging.gather = gather
    bases = {f"runner.{name}": t for name, t in _buffers(r).items()}
    shadow = ShadowStager(bases, staging, r)
    kw = dict(
        new_tokens_device=new_tokens_device,
        new_tokens_lens_device=lens,
        next_draft_tokens_device=drafts,
        previous_batch_indices=slots,
        previous_pos_indices=pos_idx,
        num_tokens=0,
        num_draft_tokens=0,
        tokens_per_gen_step=T,
        num_extend_requests=2,
    )
    # The call site: the shadow records into its copies and declines, so the built-in runs too.
    if not shadow.overlap_inputs(r, **kw):
        _legacy_previous_tensors(r, new_tokens_device, lens, drafts, slots, pos_idx, 0, 0, T, 2)
    _copy(shadow, r.input_ids_cuda[100:104], torch.arange(4, dtype=torch.int32))
    shadow.end({})
    return shadow.compare()


def test_the_shadow_agrees_with_the_builtin_writes():
    assert _shadow_step() == []


def test_the_shadow_reports_a_dropped_write():
    mismatches = _shadow_step(drop_last_gather=True)
    assert mismatches and "runner.previous_kv_lens_offsets_cuda" in mismatches[0]
