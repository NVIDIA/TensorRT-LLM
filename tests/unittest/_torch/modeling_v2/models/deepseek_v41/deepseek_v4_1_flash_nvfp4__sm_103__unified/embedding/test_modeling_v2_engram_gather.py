# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_gather entry of the DeepSeek-V4.1 target.

What is certified: one Triton launch gathers rows of an FP8 E4M3 ``[rows, 256]``
table with one E8M0 scale per 32 channels, dequantizes them in fp32 and rounds
once to bf16, writing zeros for ids outside the caller's row range -- the
DeepSeek-V4.1 Engram lookup (``ParallelEngramEmbedding.forward`` of the
checkpoint's ``inference/model.py``). Two residencies go through the same
kernel: a table in device memory and a table in pinned host memory read
through the address ``cudaHostGetDevicePointer`` returns (zero-copy).

The entry owns what is certified: ``CELLS`` is the list, ``reference`` and
``compare`` are the gate. This file turns a cell's spec into tables and ids and
keeps what a cell cannot carry: the prefix sweeps over every row count, the
largest-shard and far-end runs, CUDA-graph capture, the layout guards, the
wrong-variant controls, the checkpoint's real rows and the real-size
node-shared host table (one process, four processes, one-GPU processes).

Four sources of truth:

* synthetic tables, against the entry's ``reference`` (torch ops in the
  reference's own order: embedding lookup, fp32 multiply, one bf16 cast,
  masked fill);
* REAL rows of the checkpoint's layer-1 table, read row by row with
  ``safetensors`` ``get_slice`` at the ids the golden reference hashed for its
  prompts (``golden/<case>/engram_hash.pt``) -- never the whole table;
* the golden layer dumps: the gathered rows pushed through the reference's own
  next step (FP8 1x32 activation quantize, fp32 GEMM with ``wkv``) must
  reproduce the golden ``engram_value``;
* the checkpoint's OWN two tables (202.76 GB) in the node's one shared host copy
  of the target's ``engram_host_table.py`` (host_offload), through both of
  its backends: ``vmm`` (the default: CUDA VMM host-NUMA allocations, CUDA views
  handed to the gather) and ``shm`` (a ``/dev/shm`` file registered with
  ``cudaHostRegister``, pinned CPU views), each checked against the checkpoint's
  bytes read through an independent ``mmap`` of its safetensors shards, and each
  gather held to a reference built from the table's bytes as the CPU reads them.
The last three need external directories and skip, naming them, when one is
absent. The real-size host tests need about 270 GB of available host memory
(the table plus the fill's headroom) unless the table is already there. The
four-rank case runs one process per GPU and test_vmm_table_created_by_a_one_gpu_process
runs processes that each see one GPU of each socket: both need four GPUs visible
on two NUMA nodes, and so does the check that the vmm slabs span both sockets.
Each skips, saying what it lacks, when the node or the visible GPUs fall short.

The gate is bit equality. Every product of an E4M3 value and a power of two is
exact in fp32 (subnormals included, as the byte sweep shows), and the single
fp32 -> bf16 rounding is round-to-nearest-even on both sides, so any
difference is a defect. Deliberately wrong variants (scale per 16 channels,
out-of-shard rows not zeroed, E4M3 bytes read as E5M2) are run through the
same gate and must fail it.
"""

import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_gather import (  # noqa: E501
    engram_gather as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "engram_gather requires a CUDA device"

DIM = 256  # engram_head_dim
BLOCK = 32  # one E8M0 scale per 32 channels
LOOKUPS = 24  # (engram_max_ngram_size - 1) * engram_n_heads
WORLD = 4
REAL_ROWS_L1 = 384_006_168  # engram_num_embeddings[0], the layer-1 table
REAL_ROWS_L14 = 384_016_682

# Reference activations dumped from the checkpoint's own inference code; no default
# location, so the cases that compare against them skip unless this is set.
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_GOLDEN_DIR", "")
# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_REAL_CASES = ("capital_fr", "repeat_ngram")

DEV = torch.device("cuda")


# ─── reference: the entry's own; controls: deliberately wrong variants of it ───


# The three deliberately wrong variants every gate below must reject.


def _wrong_scale_per_16(ids, weight, scale, row_start, row_end):
    """The same 8 scale bytes applied per 16 channels: channel c takes byte (c // 16) % 8."""
    ids = ids.to(DEV).long()
    w8, s8 = weight.to(DEV).view(torch.uint8), scale.to(DEV).view(torch.uint8)
    mask = (ids < row_start) | (ids >= row_end)
    local = (ids - row_start).masked_fill(mask, 0)
    s = s8[local].view(torch.float8_e8m0fnu).float()
    s = s.repeat_interleave(16, -1).repeat(*([1] * (s.dim() - 1)), 2)  # [.., 256]: (c // 16) % 8
    v = w8[local].view(torch.float8_e4m3fn).float() * s
    return v.to(torch.bfloat16).masked_fill(mask.unsqueeze(-1), 0)


def _wrong_no_zero_fill(ids, weight, scale, row_start, row_end):
    """Out-of-shard ids read local row 0 and are left in the output (no final masked fill)."""
    ids = ids.to(DEV).long()
    w8, s8 = weight.to(DEV).view(torch.uint8), scale.to(DEV).view(torch.uint8)
    mask = (ids < row_start) | (ids >= row_end)
    local = (ids - row_start).masked_fill(mask, 0)
    v = w8[local].view(torch.float8_e4m3fn).float().unflatten(-1, (-1, BLOCK))
    v = v * s8[local].view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return v.flatten(-2).to(torch.bfloat16)


def _wrong_e5m2(ids, weight, scale, row_start, row_end):
    """The table bytes decoded as float8_e5m2 instead of float8_e4m3fn."""
    ids = ids.to(DEV).long()
    w8, s8 = weight.to(DEV).view(torch.uint8), scale.to(DEV).view(torch.uint8)
    mask = (ids < row_start) | (ids >= row_end)
    local = (ids - row_start).masked_fill(mask, 0)
    v = w8[local].view(torch.float8_e5m2).float().unflatten(-1, (-1, BLOCK))
    v = v * s8[local].view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return v.flatten(-2).to(torch.bfloat16).masked_fill(mask.unsqueeze(-1), 0)


_WRONG = {
    "scale_per_16": _wrong_scale_per_16,
    "no_zero_fill": _wrong_no_zero_fill,
    "e5m2": _wrong_e5m2,
}


# ─── gate ───


def _mismatch(actual, expected) -> float:
    """Fraction of elements whose bf16 bit pattern differs (NaN == NaN whatever the payload)."""
    a, e = actual.view(torch.int16), expected.view(torch.int16)
    same = (a == e) | (torch.isnan(actual) & torch.isnan(expected))
    return 1.0 - same.float().mean().item()


def _summed(expected):
    """What a bf16 sum over ranks of ``expected`` (one owner, +0 elsewhere) yields:
    identical except that -0.0 becomes +0.0 (IEEE -0 + +0 = +0). The reference's own
    all-reduce does the same, so the rank sum is compared against this."""
    return torch.zeros_like(expected) + expected


def _assert_controls_fail(actual, ids, weight, scale, row_start, row_end, variants, label):
    for name in variants:
        wrong = _WRONG[name](ids, weight, scale, row_start, row_end)
        with pytest.raises(AssertionError):
            op.compare(actual, wrong)
        print(
            f"  control {label} {name}: rejected, mismatch fraction {_mismatch(actual, wrong):.6f}"
        )


# ─── table builders ───


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def _table(rows: int, seed: int):
    """Device [rows, 256] E4M3 bytes (every byte except the two NaN codes) and
    [rows, 8] E8M0 bytes in 110..134 (the checkpoint's layer-1 rows read here
    use 117..121)."""
    g = _gen(seed)
    w = torch.randint(0, 256, (rows, DIM), dtype=torch.uint8, device=DEV, generator=g)
    w[(w & 0x7F) == 0x7F] = 0x38  # 0x7F / 0xFF are E4M3 NaN; the byte sweep covers them
    s = torch.randint(110, 135, (rows, DIM // BLOCK), dtype=torch.uint8, device=DEV, generator=g)
    return w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)


def _pinned(t: torch.Tensor) -> torch.Tensor:
    h = torch.empty(t.shape, dtype=torch.uint8, pin_memory=True)
    h.copy_(t.view(torch.uint8))
    assert h.is_pinned()
    return h.view(t.dtype)


def _ids(tokens: int, rows: int, seed: int, extra=()):
    """[tokens, 24] int64 ids uniform over [0, rows), with the listed ids written
    over the first positions (boundaries, padding)."""
    ids = torch.randint(
        0, rows, (tokens, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(seed)
    )
    extra = torch.tensor(list(extra), dtype=torch.int64, device=DEV)
    n = min(extra.numel(), ids.numel())
    ids.view(-1)[:n] = extra[:n]
    return ids


def _shard_bounds(rows: int, rank: int):
    part = (rows + WORLD - 1) // WORLD  # ceil(rows / world_size), as the reference allocates
    return rank * part, (rank + 1) * part, part


# ─── every byte pair ───


def _drive_byte_pairs(spec):
    """All 256 E4M3 bytes x all 256 E8M0 bytes: NaN codes, -0, subnormal products
    (scale bytes 0..2), overflow to inf (scale >= 2^121 x 448), E8M0 255 = NaN."""
    residency = spec["residency"]
    assert (spec["T"], spec["rows"]) == (11, 256)
    w = torch.arange(256, dtype=torch.uint8, device=DEV).repeat(256, 1).view(torch.float8_e4m3fn)
    s = (
        torch.arange(256, dtype=torch.uint8, device=DEV)[:, None]
        .repeat(1, DIM // BLOCK)
        .view(torch.float8_e8m0fnu)
    )
    ids = torch.full((11, LOOKUPS), -1, dtype=torch.int64, device=DEV)  # 264 slots, 8 padding
    ids.view(-1)[:256] = torch.arange(256, device=DEV)
    tw, ts = (w, s) if residency == "device" else (_pinned(w), _pinned(s))
    out = op(ids, tw, ts, 0, 256)
    expected = op.reference(ids, w, s, 0, 256)
    op.compare(out, expected)
    flat = out.view(-1, DIM)
    subnormal = (flat != 0) & (flat.abs() < torch.finfo(torch.bfloat16).tiny)
    print(
        f"\n  {residency}: {torch.isnan(out).sum().item()} NaN, {torch.isinf(out).sum().item()} inf, "
        f"{subnormal.sum().item()} bf16 subnormals, bit-exact"
    )
    assert subnormal.any() and torch.isinf(out).any() and torch.isnan(out).any()
    assert (flat[256:].view(torch.int16) == 0).all()


# ─── device table, row-sharded by rank ───


def _drive_device_shards(spec):
    """Four [ceil(rows/4), 256] shards (the last padded past the table's end with
    rows no valid id names): each rank's output is its owned rows exactly and
    +0 elsewhere, and the bf16 sum over ranks is the unsharded gather. The four
    ranks' calls run in this process, on one device; the sum is the arithmetic the
    all-reduce performs, not the collective."""
    tokens, rows = spec["T"], spec["rows"]
    assert spec["world"] == WORLD
    full_w, full_s = _table(rows, seed=tokens)
    _, _, part = _shard_bounds(rows, 0)
    boundaries = [
        0,
        part - 1,
        part,
        2 * part - 1,
        2 * part,
        3 * part - 1,
        3 * part,
        rows - 1,
        -1,
        -1,
    ]
    ids = _ids(tokens, rows, seed=100 + tokens, extra=boundaries)
    total = torch.zeros(tokens, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    owners = torch.zeros(tokens, LOOKUPS, dtype=torch.int64, device=DEV)
    for rank in range(WORLD):
        lo, hi, _ = _shard_bounds(rows, rank)
        w_r = torch.empty(part, DIM, dtype=torch.uint8, device=DEV)
        s_r = torch.empty(part, DIM // BLOCK, dtype=torch.uint8, device=DEV)
        w_r.copy_(
            torch.randint(0, 0x70, w_r.shape, dtype=torch.uint8, device=DEV, generator=_gen(rank))
        )  # pad rows
        s_r.fill_(127)
        n = min(hi, rows) - lo
        w_r[:n] = full_w.view(torch.uint8)[lo : lo + n]
        s_r[:n] = full_s.view(torch.uint8)[lo : lo + n]
        w_r, s_r = w_r.view(torch.float8_e4m3fn), s_r.view(torch.float8_e8m0fnu)
        out = op(ids, w_r, s_r, lo, hi)
        op.compare(out, op.reference(ids, w_r, s_r, lo, hi))
        owned = (ids >= lo) & (ids < hi)
        assert (out[~owned].view(torch.int16) == 0).all(), "an out-of-shard row is not +0"
        if rank == 1 and tokens == 32:
            _assert_controls_fail(out, ids, w_r, s_r, lo, hi, _WRONG, "device rank-1 shard")
        owners += owned.long()
        total += out
    assert ((owners == 1) | (ids < 0)).all() and (owners[ids < 0] == 0).all()
    op.compare(total, _summed(op.reference(ids, full_w, full_s, 0, rows)))


# ─── pinned host table (host_offload) ───


def _drive_host_shared(spec):
    """host_offload's shared layout: one pinned copy of the whole table, row range
    [0, rows), read through the mapped pointer. No device copy of the table is made:
    device memory allocated across the call is unchanged with a caller-owned out."""
    tokens, rows = spec["T"], spec["rows"]
    w, s = _table(rows, seed=7 + tokens)
    hw, hs = _pinned(w), _pinned(s)
    ids = _ids(tokens, rows, seed=tokens, extra=[0, rows - 1, -1, rows])
    out = torch.empty(tokens, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    op(ids, hw, hs, 0, rows, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert hw.device.type == "cpu" and hw.is_pinned() and hs.is_pinned()
    op.compare(out, op.reference(ids, w, s, 0, rows))


def _drive_host_shards(spec):
    """Per-rank host shards as row views (storage offsets) of one pinned buffer:
    the mapped pointer of an interior address, zero-fill and the rank sum."""
    tokens, rows = spec["T"], spec["rows"]
    assert spec["world"] == WORLD
    w, s = _table(rows, seed=31)
    _, _, part = _shard_bounds(rows, 0)
    hw = torch.zeros(WORLD * part, DIM, dtype=torch.uint8, pin_memory=True)
    hs = torch.full((WORLD * part, DIM // BLOCK), 127, dtype=torch.uint8, pin_memory=True)
    hw[:rows].copy_(w.view(torch.uint8))
    hs[:rows].copy_(s.view(torch.uint8))
    hw, hs = hw.view(torch.float8_e4m3fn), hs.view(torch.float8_e8m0fnu)
    ids = _ids(tokens, rows, seed=5, extra=[part - 1, part, 3 * part, rows - 1, -1])
    total = torch.zeros(tokens, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    for rank in range(WORLD):
        lo, hi, _ = _shard_bounds(rows, rank)
        out = op(ids, hw[lo:hi], hs[lo:hi], lo, hi)
        op.compare(out, op.reference(ids, hw[lo:hi], hs[lo:hi], lo, hi))
        total += out
    op.compare(total, _summed(op.reference(ids, w, s, 0, rows)))


# ─── byte offsets past 2 GiB (real tables are 98 GB) ───


def _drive_offsets_2gib(spec):
    """A 9,000,000-row shard (2.30 GB) whose first global row is rank 3's start of
    the layer-14 table; ids at local rows above 2^31 / 256 = 8,388,608 are read at
    byte offsets past 2 GiB. Ids just outside the shard on both sides give zeros."""
    residency, rows = spec["residency"], spec["rows"]
    lo = 3 * ((REAL_ROWS_L14 + WORLD - 1) // WORLD)  # 288,012,513
    assert lo == spec["row_start"] and spec["T"] == 32
    hi = lo + rows
    w, s = _table(rows, seed=77)
    tw, ts = (w, s) if residency == "device" else (_pinned(w), _pinned(s))
    local = torch.randint(8_388_608, rows, (32, LOOKUPS), device=DEV, generator=_gen(3))
    local[:, :4] = torch.randint(0, rows, (32, 4), device=DEV, generator=_gen(4))
    ids = lo + local
    ids[0, :6] = torch.tensor([lo, hi - 1, lo - 1, hi, 0, REAL_ROWS_L14 - 1], device=DEV)
    out = op(ids, tw, ts, lo, hi)
    op.compare(out, op.reference(ids, w, s, lo, hi))
    assert (out[0, 2:6].view(torch.int16) == 0).all()


# ─── CUDA graphs ───


@pytest.mark.parametrize("tokens", [1, 32, 4096])
@pytest.mark.parametrize("residency", ["device", "host"])
def test_cuda_graph_capture_replay(residency, tokens):
    """Capture one call (the host pointer is resolved during capture), replay with
    new ids, then rewrite the table in place and replay again: the replay reads the
    live table (a captured staging copy would return stale rows)."""
    rows = 50_000
    w, s = _table(rows, seed=11)
    tw, ts = (w.clone(), s.clone()) if residency == "device" else (_pinned(w), _pinned(s))
    ids = _ids(tokens, rows, seed=12, extra=[-1, rows])
    out = torch.empty(tokens, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    op(ids, tw, ts, 0, rows, out=out)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(ids, tw, ts, 0, rows, out=out)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    op.compare(out, op.reference(ids, tw, ts, 0, rows))

    ids.copy_(_ids(tokens, rows, seed=13, extra=[rows - 1, -5]))
    graph.replay()
    torch.cuda.synchronize()
    op.compare(out, op.reference(ids, tw, ts, 0, rows))

    new_w, new_s = _table(rows, seed=14)
    tw.view(torch.uint8).copy_(new_w.view(torch.uint8))
    ts.view(torch.uint8).copy_(new_s.view(torch.uint8))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    op.compare(out, op.reference(ids, new_w, new_s, 0, rows))
    eager = op(ids, tw, ts, 0, rows)
    assert torch.equal(eager.view(torch.int16), out.view(torch.int16))


# ─── every row count 1..N_max is the same per-row function ───
#
# The core target calls this op on one rank's T tokens (tep4 / tp4: the
# [:, layer, :] view of the hash output, row stride 48; T = 1..max_num_tokens, up to
# 16384 per rank) and, under dep4, on the all-gathered ids of the four ranks (the
# contiguous int64 view of an int32 [4 * rows, 48] all-gather buffer; up to
# 4 x 16384 = 65536 rows). T reaches
# the kernel only as ``n_rows = T * 24``, which sizes the grid and masks the last
# program. The sweeps below check that every T from 1 to N_max returns exactly the
# first T rows of the N_max call, which is itself held bit-exact to the reference.


def _jit_variants(fn) -> int:
    """Compiled variants Triton holds for ``fn`` on every device (Triton >= 3.2 keeps them
    in ``device_caches[device][0]``, older releases in ``cache[device]``)."""
    dc = getattr(fn, "device_caches", None)
    if dc:
        return sum(len(v[0]) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


_POISON = (
    0x7FA5  # a bf16 NaN bit pattern the kernel never produces (the tables here hold no NaN bytes)
)
_GUARD = 64  # rows past the prefix that must still hold the poison after each call


def _core_shard(full_w, full_s, rows: int, rank: int, seed: int):
    """Rank ``rank``'s [ceil(rows / 4), 256] device shard, as the core holds it: bounds
    ``row_start = rank * part``, ``row_end = min(row_start + part, rows)``; the last
    shard's rows past the table's end hold bytes no valid id names."""
    part = (rows + WORLD - 1) // WORLD
    lo, hi = rank * part, min(rank * part + part, rows)
    w_r = torch.randint(0, 0x70, (part, DIM), dtype=torch.uint8, device=DEV, generator=_gen(seed))
    s_r = torch.full((part, DIM // BLOCK), 127, dtype=torch.uint8, device=DEV)
    w_r[: hi - lo] = full_w.view(torch.uint8)[lo:hi]
    s_r[: hi - lo] = full_s.view(torch.uint8)[lo:hi]
    return w_r.view(torch.float8_e4m3fn), s_r.view(torch.float8_e8m0fnu), lo, hi


def _sprinkle(ids: torch.Tensor, values, seed: int) -> None:
    """Write each of ``values`` at 64 random positions of ``ids`` (in place), and all of
    them into the first token row, so even r = 1 sees every edge."""
    g = _gen(seed)
    for v in values:
        t = torch.randint(0, ids.shape[0], (64,), device=DEV, generator=g)
        k = torch.randint(0, ids.shape[1], (64,), device=DEV, generator=g)
        ids[t, k] = v
    ids[0, : len(values)] = torch.tensor(list(values), dtype=ids.dtype, device=DEV)


@pytest.mark.parametrize(
    "form,n_max,rank", [("tp_view", 16384, 1), ("dep4_gathered", 4 * 16384, 3)]
)
def test_every_row_count_is_a_prefix_of_n_max(form, n_max, rank):
    """Every prefix ids[:r], r = 1..N_max, returns exactly the first r rows of the N_max call.

    ``tp_view``: rank 1's shard (row_start = part > 0) over a [:, 1, :] view of a
    [N, 2, 24] int64 hash buffer (row stride 48), N_max 16384. ``dep4_gathered``:
    rank 3's (padded, last) shard over the int64 view of an int32 [N, 48] all-gather
    buffer, N_max 65536, with the zero padding rows the core's pad-to-max leaves.
    Ids include -1, row_start - 1, row_start, row_end - 1, row_end, 0 and rows - 1.
    Each call writes into a row-prefix view of one poisoned buffer: its r rows must
    equal the N_max rows bit for bit and the next 64 rows must still hold the poison
    (mismatches are summed on the device, one synchronization at the end). The Triton
    kernel's compiled variants are counted: after r = 1 and r = 2 (n_rows = 24 r is
    divisible by 16 exactly for even r, Triton's integer specialization) no further
    variant may appear for any r."""
    rows = 100_003
    full_w, full_s = _table(rows, seed=4000 + rank)
    w_r, s_r, lo, hi = _core_shard(full_w, full_s, rows, rank, seed=4100 + rank)
    if form == "tp_view":
        hashes = torch.randint(
            0, rows, (n_max, 2, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(4200)
        )
        ids = hashes[:, 1, :]
        assert ids.stride() == (2 * LOOKUPS, 1)
    else:
        g32 = torch.empty(n_max, 2 * LOOKUPS, dtype=torch.int32, device=DEV)
        ids = g32.view(torch.int64)
        ids.copy_(
            torch.randint(
                0, rows, (n_max, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(4201)
            )
        )
        per_rank = n_max // WORLD
        for r_ in range(
            WORLD
        ):  # pad-to-max rows at the end of each rank's block: all-zero int32 words
            g32[(r_ + 1) * per_rank - 5 * (r_ + 1) : (r_ + 1) * per_rank] = 0
        assert ids.is_contiguous() and ids.shape == (n_max, LOOKUPS)
    _sprinkle(ids, [-1, lo - 1, lo, hi - 1, hi, 0, rows - 1], seed=4300)
    owned = (ids >= lo) & (ids < hi)
    assert 0 < owned.float().mean().item() < 0.5 and (ids == -1).any()

    full = op(ids, w_r, s_r, lo, hi)
    op.compare(full, op.reference(ids, w_r, s_r, lo, hi))
    full_bits = full.view(torch.int16)
    buf = torch.empty(n_max + _GUARD, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    bits = buf.view(torch.int16)
    bits.fill_(_POISON)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    over = torch.zeros((), dtype=torch.int64, device=DEV)
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_gather import (  # noqa: E501
        _engram_gather_kernel as fn,
    )

    before = _jit_variants(fn)
    settled = None
    for r in range(1, n_max + 1):
        res = op(ids[:r], w_r, s_r, lo, hi, out=buf[:r])
        assert res.data_ptr() == buf.data_ptr()
        bad += (bits[:r] != full_bits[:r]).sum()
        over += (bits[r : r + _GUARD] != _POISON).sum()
        if r == 2:
            settled = _jit_variants(fn)
    torch.cuda.synchronize()
    after = _jit_variants(fn)
    print(
        f"\n  {form}: r = 1..{n_max}, rank {rank} [{lo}, {hi}): {bad.item()} elements differ from the "
        f"{n_max}-row result, {over.item()} guard elements written; kernel variants {before} before, "
        f"{settled} after r = 2, {after} after r = {n_max}"
    )
    assert bad.item() == 0, f"{bad.item()} elements differ from the {n_max}-row result's rows"
    assert over.item() == 0, f"{over.item()} elements past a prefix were written"
    assert after == settled and settled - before <= 2, (before, settled, after)
    del buf, full


def test_row_sweep_harness_sees_a_one_row_shift():
    """Control for the sweeps above: the same bitwise per-row check rejects a call that
    reads the ids one token row off, at every r tried, including r = 1."""
    rows = 100_003
    full_w, full_s = _table(rows, seed=4400)
    w_r, s_r, lo, hi = _core_shard(full_w, full_s, rows, 1, seed=4401)
    hashes = torch.randint(
        0, rows, (65, 2, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(4402)
    )
    hashes[:, 1, :8] = lo + torch.arange(65 * 8, device=DEV).view(
        65, 8
    )  # distinct owned rows per token
    ids = hashes[:, 1, :]
    full = op(ids[:64], w_r, s_r, lo, hi)
    for r in (1, 2, 7, 63, 64):
        shifted = op(ids[1 : r + 1], w_r, s_r, lo, hi)
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
        print(f"  [control row shift] r={r}: {n_bad} / {r * LOOKUPS * DIM} elements differ")
        assert n_bad > 0


# ─── table size: any per-rank shard up to the largest the target allocates ───

# The target allocates every rank's shard as [ceil(rows / 4), 256] (+ [.., 8] scale bytes):
# 96,001,542 rows for layer 1 and 96,004,171 for layer 14 (the largest), whatever the
# free memory; rank 3's layer-14 shard holds 96,004,169 loaded rows (row_end = rows).
_PART_L1 = (REAL_ROWS_L1 + WORLD - 1) // WORLD
_PART_L14 = (REAL_ROWS_L14 + WORLD - 1) // WORLD
_PART_MAX = max(_PART_L1, _PART_L14)


def _fill_rows(buf_w, buf_s, local: torch.Tensor, seed: int) -> None:
    """Random E4M3 bytes (no NaN codes) and E8M0 bytes 110..134 into the listed local rows."""
    g = _gen(seed)
    w = torch.randint(0, 256, (local.numel(), DIM), dtype=torch.uint8, device=DEV, generator=g)
    w[(w & 0x7F) == 0x7F] = 0x38
    buf_w[local] = w
    buf_s[local] = torch.randint(
        110, 135, (local.numel(), DIM // BLOCK), dtype=torch.uint8, device=DEV, generator=g
    )


def _wrong_wrap32(ids, weight, scale, row_start, row_end):
    """The gather with a 32-bit byte offset: local row r read at (r * 256) mod 2^32."""
    ids = ids.to(DEV).long()
    wrapped = row_start + (ids - row_start) % ((1 << 32) // DIM)
    return op.reference(
        torch.where((ids >= row_start) & (ids < row_end), wrapped, ids),
        weight,
        scale,
        row_start,
        row_end,
    )


def test_table_size_invariance_and_far_end_at_the_largest_shard():
    """The table-row axis. The table is a checkpoint constant, not a pool the
    runtime sizes from free memory, and the kernel receives it as a raw address plus
    (row_start, row_end); this runs that instead of arguing it.

    (1) Invariance: one device buffer of the largest per-rank shard the target allocates
    (layer 14, 96,004,171 rows, 24.6 GB + 0.77 GB scale bytes), rows 0..4095 random and
    the rest a constant row. The same 16384-token call (the tp [:, 1, :] view, rank 3's
    layer-14 row_start 288,012,513; ids in the first 4096 local rows plus -1, ids below
    row_start and at row_end) on views [:R] for R = 4096 (smallest holding them),
    2^24 (mid) and 96,004,171, and on a separate exact 4096-row allocation, is bit-exact
    to the reference and bitwise equal across all four -- once with row_end fixed at
    row_start + 4096 and once with row_end = row_start + R.
    (2) Far end: for every rank of both layers, with the target's own bounds
    (row_end = min(row_start + part, rows)), ids among the last 2^20 loaded rows of the
    shard plus the last loaded row, row_end - 1, row_end, row_start, row_start - 1 and -1,
    in the tp view (T 16384) and, for rank 3, the dep4 all-gathered form (T 65536): bit-exact.
    Control: a gather with a 32-bit byte offset (row r read at (256 r) mod 2^32) fails."""
    buf_w = torch.full((_PART_MAX, DIM), 0x38, dtype=torch.uint8, device=DEV)
    buf_s = torch.full((_PART_MAX, DIM // BLOCK), 127, dtype=torch.uint8, device=DEV)
    try:
        small = 4096
        _fill_rows(buf_w, buf_s, torch.arange(small, device=DEV), seed=9100)
        lo = 3 * _PART_L14  # 288,012,513
        hashes = torch.randint(
            lo, lo + small, (16384, 2, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(9101)
        )
        ids = hashes[:, 1, :]
        _sprinkle(ids, [-1, lo - 1, lo, lo + small - 1, 0], seed=9102)
        exact_w, exact_s = buf_w[:small].clone(), buf_s[:small].clone()
        views = [
            (small, buf_w[:small], buf_s[:small]),
            (1 << 24, buf_w[: 1 << 24], buf_s[: 1 << 24]),
            (_PART_MAX, buf_w, buf_s),
            ("exact", exact_w, exact_s),
        ]
        for fixed_end in (True, False):
            outs = []
            for size, w, s in views:
                hi = lo + small if fixed_end or size == "exact" else lo + w.shape[0]
                tw, ts = w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)
                out = op(ids, tw, ts, lo, hi)
                op.compare(out, op.reference(ids, tw, ts, lo, hi))
                outs.append(out)
            for out in outs[1:]:
                assert torch.equal(out.view(torch.int16), outs[0].view(torch.int16))
            print(
                f"\n  invariance (row_end {'fixed' if fixed_end else '= row_start + R'}): R = "
                f"{[v[0] for v in views]}: bit-exact to the reference and bitwise equal"
            )
        # zero-fill: ids below row_start and -1 read nothing
        assert (outs[0].view(torch.int16)[0, 0] == 0).all() and (
            outs[0].view(torch.int16)[0, 1] == 0
        ).all()

        buf_w[:small] = 0x38
        buf_s[:small] = 127
        for layer, rows, part in (
            ("L1", REAL_ROWS_L1, _PART_L1),
            ("L14", REAL_ROWS_L14, _PART_L14),
        ):
            tw, ts = buf_w[:part].view(torch.float8_e4m3fn), buf_s[:part].view(torch.float8_e8m0fnu)
            for rank in range(WORLD):
                lo = rank * part
                hi = min(lo + part, rows)
                n_loaded = hi - lo
                tail = torch.randint(
                    n_loaded - (1 << 20), n_loaded, (4096,), device=DEV, generator=_gen(9200 + rank)
                )
                local = torch.unique(torch.cat([tail, torch.tensor([n_loaded - 1, 0], device=DEV)]))
                _fill_rows(buf_w, buf_s, local, seed=9300 + rank)
                forms = [("tp_view", 16384)] + (
                    [("dep4_gathered", 4 * 16384)] if rank == WORLD - 1 else []
                )
                for form, t in forms:
                    pick = torch.randint(
                        0, local.numel(), (t, LOOKUPS), device=DEV, generator=_gen(9400 + rank)
                    )
                    if form == "tp_view":
                        hashes = torch.empty(t, 2, LOOKUPS, dtype=torch.int64, device=DEV)
                        hashes[:, 1, :] = lo + local[pick]
                        ids = hashes[:, 1, :]
                    else:
                        g32 = torch.empty(t, 2 * LOOKUPS, dtype=torch.int32, device=DEV)
                        ids = g32.view(torch.int64)
                        ids.copy_(lo + local[pick])
                    _sprinkle(
                        ids, [-1, lo - 1, lo, hi - 1, hi, lo + n_loaded - 1], seed=9500 + rank
                    )
                    out = op(ids, tw, ts, lo, hi)
                    op.compare(out, op.reference(ids, tw, ts, lo, hi))
                    far = int((ids - lo).clamp(max=n_loaded - 1).max().item())
                    print(
                        f"  far end {layer} rank {rank} {form} T {t}: [{lo}, {hi}) of a {part}-row shard, "
                        f"max local row {far} (byte offset {far * DIM}): bit-exact"
                    )
                    if rank == 0 and form == "tp_view":
                        with pytest.raises(AssertionError):
                            op.compare(out, _wrong_wrap32(ids, tw, ts, lo, hi))
                        print(
                            f"  control {layer} 32-bit byte offset: rejected, mismatch fraction "
                            f"{_mismatch(out, _wrong_wrap32(ids, tw, ts, lo, hi)):.6f}"
                        )
                buf_w[local] = 0x38
                buf_s[local] = 127
    finally:
        del buf_w, buf_s
        torch.cuda.empty_cache()


# ─── capture at every graph batch the gates capture ───

# The generation-only CUDA-graph batches an engine captures per rank, Gset: every batch
# 1..32 plus TRT-LLM's padded default list for cuda_graph_config {enable_padding: true,
# max_batch_size: 256} (1, 2, 4, 8, 16..128 step 8, 192, 256; the padding-off list at 256,
# 1..32, 64, 128, 256, is a subset). A decode step carries G tokens and a DSpark
# target-verify step 6 G (1 + 5 drafts): T = G and 6 G, 82 counts up to 1536. tep4 / tp4
# and host_offload gather the rank's T tokens through a [:, layer, :] view of the hash
# output (row stride 48); dep4 with device tables gathers the all-gathered ids of the four
# ranks, 4 x T rows (82 counts up to 6144), contiguous.
_GRAPH_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_T = sorted(set(_GRAPH_BATCHES) | {6 * b for b in _GRAPH_BATCHES})
assert len(_GRAPH_BATCHES) == 46 and len(_CAPTURE_T) == 82 and _CAPTURE_T[-1] == 1536
_CAPTURE_CASES = [("tp_view", t) for t in _CAPTURE_T] + [
    ("dep4_gathered", 4 * t) for t in _CAPTURE_T
]


def _capture_ids(form: str, tokens: int, rows: int, seed: int):
    """[tokens, 2, 24] ids for ``tp_view`` (both layers' views are gathered) or
    [tokens, 24] for ``dep4_gathered`` (each rank's block of tokens / 4 rows ends in
    zero padding rows, as the core's pad-to-max before the all-gather leaves it)."""
    part = (rows + WORLD - 1) // WORLD
    edge = [0, part - 1, part, 2 * part - 1, 2 * part, 3 * part - 1, 3 * part, rows - 1, -1]
    if form == "tp_view":
        ids = torch.randint(
            0, rows, (tokens, 2, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(seed)
        )
        ids[0, 0, : len(edge)] = torch.tensor(edge, device=DEV)
        ids[-1, 1, : len(edge)] = torch.tensor(edge[::-1], device=DEV)
        return ids
    ids = torch.randint(
        0, rows, (tokens, LOOKUPS), dtype=torch.int64, device=DEV, generator=_gen(seed)
    )
    ids.view(-1)[: len(edge)] = torch.tensor(edge, device=DEV)
    per_rank = tokens // WORLD
    for rank in range(WORLD):
        pad = (
            (seed + rank) % 3 if per_rank > 2 else 0
        )  # 0..2 padding rows at the end of each rank's block
        if pad:
            ids[(rank + 1) * per_rank - pad : (rank + 1) * per_rank] = 0
    return ids


@pytest.mark.parametrize("form,tokens", _CAPTURE_CASES)
def test_cuda_graph_at_every_capture_batch(form, tokens):
    """The core's calls, captured per rank with ``out`` allocated inside the capture
    (``engram_gather(ids, shard_w, shard_s, row_start, row_end)``), after an eager
    warm-up at the same shape. ``tp_view``: the two Engram layers' ``hashes[:, l, :]``
    views of one static ``[T, 2, 24]`` int64 buffer; ``dep4_gathered``: the int64
    view of a static ``[4 * rows, 48]`` int32 all-gather buffer. Three replays on new
    ids written in place: every rank's replay equals the same eager call bit for bit
    and the reference, out-of-shard rows are +0, and the bf16 sum over the four ranks
    is the unsharded gather."""
    rows = 100_003
    full_w, full_s = _table(rows, seed=900 + tokens)
    if form == "tp_view":
        static = _capture_ids(form, tokens, rows, seed=0)
        views = [static[:, 0, :], static[:, 1, :]]
        assert all(v.stride() == (2 * LOOKUPS, 1) for v in views)
    else:
        g32 = torch.empty(tokens, 2 * LOOKUPS, dtype=torch.int32, device=DEV)
        g32.view(torch.int64).copy_(_capture_ids(form, tokens, rows, seed=0))
        views = [g32.view(torch.int64)]
        assert views[0].shape == (tokens, LOOKUPS) and views[0].is_contiguous()
    capture_ids = [v.clone() for v in views]
    shards = [_core_shard(full_w, full_s, rows, rank, seed=50 + rank) for rank in range(WORLD)]
    graphs, outs = [], []
    for w_r, s_r, lo, hi in shards:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for v in views:
                op(v, w_r, s_r, lo, hi)  # engine warm-up at this shape
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append([op(v, w_r, s_r, lo, hi) for v in views])
        graphs.append(graph)
    for rep in (1, 2, 3):
        new = _capture_ids(form, tokens, rows, seed=rep)
        if form == "tp_view":
            static.copy_(new)
        else:
            g32.view(torch.int64).copy_(new)
        for rank, graph in enumerate(graphs):
            for o in outs[rank]:
                o.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        for li, v in enumerate(views):
            total = torch.zeros(tokens, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
            owners = torch.zeros(tokens, LOOKUPS, dtype=torch.int64, device=DEV)
            for rank, (w_r, s_r, lo, hi) in enumerate(shards):
                out = outs[rank][li]
                eager = op(v, w_r, s_r, lo, hi)
                assert torch.equal(out.view(torch.int16), eager.view(torch.int16)), (
                    f"rank {rank}: replay != eager"
                )
                expected = op.reference(v, w_r, s_r, lo, hi)
                op.compare(out, expected)
                owned = (v >= lo) & (v < hi)
                assert (out[~owned].view(torch.int16) == 0).all(), "an out-of-shard row is not +0"
                stale = op.reference(capture_ids[li], w_r, s_r, lo, hi)
                if owned.any() or ((capture_ids[li] >= lo) & (capture_ids[li] < hi)).any():
                    # the replay read the new ids: the capture-time ids' rows would differ
                    assert not torch.equal(expected.view(torch.int16), stale.view(torch.int16)), (
                        f"rank {rank}"
                    )
                if rep == 1 and li == 0 and rank == 1:
                    print(
                        f"\n  {form} T={tokens}: 4 ranks x {len(views)} calls x 3 replays, replay == eager == "
                        f"reference bit for bit; rank 1 differs from the capture-time ids' rows on "
                        f"{_mismatch(out, stale):.6f}"
                    )
                    _assert_controls_fail(
                        out, v, w_r, s_r, lo, hi, _WRONG, f"graph {form} T={tokens} rank 1"
                    )
                owners += owned.long()
                total += out
            assert ((owners == 1) | (v < 0)).all() and (owners[v < 0] == 0).all()
            op.compare(total, _summed(op.reference(v, full_w, full_s, 0, rows)))


# ─── capture through the node's shared pinned host table (host_offload) ───


def _write_synthetic_checkpoint(ckpt_dir: str, tables) -> None:
    """A two-layer 'checkpoint' holding only ``layers.{1,14}.engram.embed.{weight,scale}``,
    in one safetensors shard written by hand (header length, JSON header, raw bytes),
    plus its ``model.safetensors.index.json``: what ``engram_host_table`` reads."""
    import struct

    header, blobs, at = {}, [], 0
    for layer, (w, s) in tables.items():
        for kind, t, dtype in (("weight", w, "F8_E4M3"), ("scale", s, "U8")):
            b = t.view(torch.uint8).cpu().contiguous().numpy().tobytes()
            header[f"layers.{layer}.engram.embed.{kind}"] = {
                "dtype": dtype,
                "shape": list(t.shape),
                "data_offsets": [at, at + len(b)],
            }
            blobs.append(b)
            at += len(b)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    shard = "model-00001-of-00001.safetensors"
    with open(os.path.join(ckpt_dir, shard), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: shard for k in header}}, f)


@pytest.mark.parametrize("tokens", [1, 5, 8, 32, 48, 192])
def test_cuda_graph_through_the_shared_host_table(tokens, tmp_path):
    """host_offload's residency, end to end with the target's host-table module: a
    synthetic two-layer checkpoint (100,003 rows per layer; never the real table) is
    filled into a node-shared file with ``fill_shared_engram_table``, mapped and
    ``cudaHostRegister``ed with ``attach_shared_engram_table``, and its
    ``weight(layer)`` / ``scale(layer)`` views (uint8 scale bytes) are gathered with
    ``row_start = 0, row_end = rows``, captured after a warm-up and replayed on new
    ids for both layers: replay == eager == reference bit for bit, no device copy of
    the table. The file is removed afterwards."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.engram_host_table import (  # noqa: E501
        attach_shared_engram_table,
        fill_shared_engram_table,
        remove_shared_engram_table,
    )

    rows = 100_003
    tables = {1: _table(rows, seed=6001), 14: _table(rows, seed=6014)}
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_synthetic_checkpoint(str(ckpt), tables)
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else str(tmp_path)
    path = os.path.join(shm, f"g4-engram-gather-test-{os.getpid()}-{tokens}.bin")
    table = None
    try:
        report = fill_shared_engram_table(str(ckpt), path, threads=4, headroom_bytes=1 << 30)
        assert not report.reused
        table = attach_shared_engram_table(path, checkpoint_dir=str(ckpt), device=DEV)
        hw = {layer: table.weight(layer) for layer in (1, 14)}
        hs = {layer: table.scale(layer) for layer in (1, 14)}
        for layer in (1, 14):
            assert (
                hw[layer].device.type == "cpu" and hw[layer].is_pinned() and hs[layer].is_pinned()
            )
            assert torch.equal(
                hw[layer].view(torch.uint8), tables[layer][0].view(torch.uint8).cpu()
            )
        static = _capture_ids("tp_view", tokens, rows, seed=0)
        views = [static[:, 0, :], static[:, 1, :]]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for li, layer in enumerate((1, 14)):
                op(views[li], hw[layer], hs[layer], 0, rows)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [
                op(views[li], hw[layer], hs[layer], 0, rows) for li, layer in enumerate((1, 14))
            ]
        for rep in (1, 2):
            static.copy_(_capture_ids("tp_view", tokens, rows, seed=rep))
            for o in outs:
                o.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for li, layer in enumerate((1, 14)):
                eager = op(views[li], hw[layer], hs[layer], 0, rows)
                assert torch.equal(outs[li].view(torch.int16), eager.view(torch.int16))
                op.compare(
                    outs[li], op.reference(views[li], tables[layer][0], tables[layer][1], 0, rows)
                )
        # the graph's own output pool aside, no device memory holds a table copy
        assert torch.cuda.memory_allocated() - before < rows * DIM
        print(
            f"\n  shared host table ({report.nbytes} table bytes, file {path}): T={tokens} x 2 layers, "
            f"2 replays == eager == reference bit for bit"
        )
        del graph, outs
        torch.cuda.synchronize()
    finally:
        hw = hs = None
        if table is not None:
            table.close()
        remove_shared_engram_table(path, force=True)
        for p in (path, path + ".lock"):
            if os.path.exists(p):
                os.unlink(p)


# ─── layouts and edges ───


def _drive_layouts(spec):
    """T = 0, one layer's column of the hash output as a strided view, int32 ids, uint8
    scale bytes, a caller-owned out as a row prefix of a larger buffer, inputs unmodified."""
    rows = spec["rows"]
    assert spec["T"] == 37
    w, s = _table(rows, seed=21)
    # T = 0: nothing launched, an empty [0, 24, 256] result
    empty = op(torch.empty(0, LOOKUPS, dtype=torch.int64, device=DEV), w, s, 0, rows)
    assert empty.shape == (0, LOOKUPS, DIM) and empty.dtype == torch.bfloat16
    # the hash kernel's [T, 2, 24] output, one layer's column taken as a view (row stride 48)
    hashes = torch.randint(0, rows, (37, 2, LOOKUPS), device=DEV, generator=_gen(22))
    for layer in range(2):
        view = hashes[:, layer, :]
        assert view.stride() == (2 * LOOKUPS, 1)
        op.compare(op(view, w, s, 0, rows), op.reference(view, w, s, 0, rows))
    # int32 ids
    ids = _ids(37, rows, seed=23, extra=[-1, rows, rows - 1])
    op.compare(op(ids.int(), w, s, 0, rows), op.reference(ids, w, s, 0, rows))
    # uint8 scale bytes are read the same as float8_e8m0fnu
    op.compare(op(ids, w, s.view(torch.uint8), 0, rows), op.reference(ids, w, s, 0, rows))
    # caller-owned out as a row prefix of a NaN-filled buffer: every element of the
    # prefix is written, nothing past it is
    big = torch.full((40, LOOKUPS, DIM), float("nan"), dtype=torch.bfloat16, device=DEV)
    sentinel = big[37:].clone()
    res = op(ids, w, s, 0, rows, out=big[:37])
    assert res.data_ptr() == big.data_ptr()
    op.compare(big[:37], op.reference(ids, w, s, 0, rows))
    assert torch.equal(big[37:].view(torch.int16), sentinel.view(torch.int16))
    # ids and tables are not modified
    ids_before, w_before = ids.clone(), w.view(torch.uint8).clone()
    op(ids, w, s, 0, rows)
    assert torch.equal(ids, ids_before) and torch.equal(w.view(torch.uint8), w_before)


_DRIVERS = {
    "byte_pairs": _drive_byte_pairs,
    "device_shards": _drive_device_shards,
    "host_shared": _drive_host_shared,
    "host_shards": _drive_host_shards,
    "offsets_2gib": _drive_offsets_2gib,
    "layouts": _drive_layouts,
}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every cell through ``op`` (guarded by ``is_valid``), ``op.reference`` and ``op.compare``,
    with the structural checks (zero fill, rank sum, untouched memory) its configuration carries."""
    certifying(op)
    spec = cell.spec
    assert (spec["D"], spec["B"], spec["K"]) == (DIM, BLOCK, LOOKUPS)
    _DRIVERS[spec["kind"]](spec)


def test_is_valid_rejects_layouts_the_kernel_misreads():
    """Each case ``is_valid`` refuses (inside ``validating``) was run through the launch
    without the guard on sm_103 and returned wrong values (or wrote outside ``out``) without
    an error; the measured results are in ``is_valid``'s docstring."""
    rows = 1000
    w, s = _table(rows, seed=41)
    ids = _ids(7, rows, seed=42)
    out = torch.empty(7, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    bad = {
        "ids inner stride": lambda: op(ids.t().contiguous().t(), w, s, 0, rows),
        "ids float32": lambda: op(ids.float(), w, s, 0, rows),
        "weight float8_e5m2": lambda: op(
            ids, w.view(torch.uint8).view(torch.float8_e5m2), s, 0, rows
        ),
        "scale float32": lambda: op(ids, w, s.float(), 0, rows),
        "weight column slice": lambda: op(
            ids,
            torch.cat([w.view(torch.uint8)] * 2, 1).view(torch.float8_e4m3fn)[:, :DIM],
            s,
            0,
            rows,
        ),
        "scale column slice": lambda: op(
            ids, w, torch.cat([s.view(torch.uint8)] * 2, 1)[:, : DIM // BLOCK], 0, rows
        ),
        "scale with fewer rows than weight": lambda: op(ids, w, s[: rows // 2], 0, rows),
        "scale width not dividing D": lambda: op(
            ids, w, s.view(torch.uint8)[:, :7].contiguous(), 0, rows
        ),
        "row range beyond the rows held": lambda: op(ids, w[:100], s[:100], 0, 200),
        "out last-dim slice": lambda: op(
            ids,
            w,
            s,
            0,
            rows,
            out=torch.empty(7, LOOKUPS, 2 * DIM, dtype=torch.bfloat16, device=DEV)[..., :DIM],
        ),
        "out too few token rows": lambda: op(ids, w, s, 0, rows, out=out[:3]),
    }
    match = {
        "ids inner stride": r"ids must be \[T, K\]",
        "ids float32": "ids must be int64 or int32",
        "weight float8_e5m2": "weight must be float8_e4m3fn",
        "scale float32": "scale must be float8_e8m0fnu or uint8",
        "weight column slice": "weight must be a contiguous",
        "scale column slice": "scale must be a contiguous",
        "scale with fewer rows than weight": "does not match weight",
        "scale width not dividing D": "does not match weight",
        "row range beyond the rows held": "exceeds the",
        "out last-dim slice": "out must be a contiguous",
        "out too few token rows": "out must be a contiguous",
    }
    assert set(match) == set(bad)
    for name, call in bad.items():
        with pytest.raises(AssertionError, match=match[name]):
            with validating(op):
                call()
        print(f"  rejected: {name}")
    # CPU ids are refused by Triton's own launcher, loudly
    with pytest.raises(ValueError, match="cannot be accessed from Triton"):
        op(ids.cpu(), w, s, 0, rows)


# ─── REAL rows of the checkpoint's layer-1 table ───


def _real_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_GOLDEN_DIR, c, "engram_hash.pt")) for c in _REAL_CASES
    ) and os.path.isfile(os.path.join(_CKPT_DIR, "model.safetensors.index.json"))


def _skip_without_real():
    if not _real_available():
        pytest.skip(f"golden dumps ({_GOLDEN_DIR}) or checkpoint ({_CKPT_DIR}) not present")


@functools.lru_cache(maxsize=None)
def _weight_file(name: str) -> str:
    with open(os.path.join(_CKPT_DIR, "model.safetensors.index.json")) as f:
        return os.path.join(_CKPT_DIR, json.load(f)["weight_map"][name])


@functools.lru_cache(maxsize=None)
def _real_rows(cases: tuple):
    """Layer-1 hash ids of the golden cases ([T, 24] each, offsets applied), and the
    rows they name, read one row at a time through get_slice (never the table)."""
    from safetensors import safe_open

    hids = [
        torch.load(os.path.join(_GOLDEN_DIR, c, "engram_hash.pt"), weights_only=True)["hash_ids"][
            :, 0, :
        ]
        for c in cases
    ]
    uniq = torch.unique(torch.cat([h.reshape(-1) for h in hids]))
    got = []
    for name in ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale"):
        with safe_open(_weight_file(name), framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            assert sl.get_shape()[0] == REAL_ROWS_L1
            got.append(torch.cat([sl[r : r + 1] for r in uniq.tolist()]))
    w, s = got
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu, (w.dtype, s.dtype)
    return hids, uniq, w, s


@pytest.mark.parametrize("residency", ["device", "host"])
@pytest.mark.parametrize("case", _REAL_CASES)
def test_real_rows_compact_table(case, residency):
    """The golden prompt's real rows in a compact table (ids remapped): the device
    and the pinned host path both reproduce the reference dequantization bit for bit."""
    _skip_without_real()
    (hid,), uniq, w, s = _real_rows((case,))
    ids = torch.searchsorted(uniq, hid.contiguous()).to(DEV)
    tw, ts = (w.to(DEV), s.to(DEV)) if residency == "device" else (_pinned(w), _pinned(s))
    out = op(ids, tw, ts, 0, uniq.numel())
    op.compare(out, op.reference(ids, w, s, 0, uniq.numel()))
    print(
        f"\n  {case} {residency}: {tuple(hid.shape)} ids, {uniq.numel()} distinct real rows, bit-exact"
    )
    _assert_controls_fail(out, ids, w, s, 0, uniq.numel(), ("scale_per_16", "e5m2"), f"real {case}")


def test_real_ids_at_the_real_shard_size():
    """The real global ids against a device buffer of the real per-rank shard size
    (ceil(384,006,168 / 4) = 96,001,542 rows, 24.6 GB): each rank's call reads its
    own rows at their true local offsets (up to 24.6 GB into the buffer) and zeroes
    the rest, and the rank sum equals the unsharded gather."""
    _skip_without_real()
    hids, uniq, w, s = _real_rows(_REAL_CASES)
    ids = torch.cat(hids).to(DEV)
    part = (REAL_ROWS_L1 + WORLD - 1) // WORLD
    buf_w = torch.empty(part, DIM, dtype=torch.uint8, device=DEV)
    buf_s = torch.empty(part, DIM // BLOCK, dtype=torch.uint8, device=DEV)
    # local row 0 of every rank is a known nonzero row unless a real id lands there,
    # so the no-zero-fill control (out-of-shard ids read local row 0) has something to show
    buf_w[0].fill_(0x38)
    buf_s[0].fill_(127)
    uw, us = w.view(torch.uint8).to(DEV), s.view(torch.uint8).to(DEV)
    uniq_d = uniq.to(DEV)
    expected = op.reference(torch.searchsorted(uniq_d, ids), w, s, 0, uniq.numel())
    total = torch.zeros_like(expected)
    for rank in range(WORLD):
        lo, hi = rank * part, (rank + 1) * part
        mine = (uniq_d >= lo) & (uniq_d < hi)
        buf_w[uniq_d[mine] - lo] = uw[mine]
        buf_s[uniq_d[mine] - lo] = us[mine]
        tw, ts = buf_w.view(torch.float8_e4m3fn), buf_s.view(torch.float8_e8m0fnu)
        out = op(ids, tw, ts, lo, hi)
        owned = (ids >= lo) & (ids < hi)
        want = expected.masked_fill(~owned.unsqueeze(-1), 0)
        op.compare(out, want)
        print(
            f"\n  rank {rank}: [{lo}, {hi}) owns {owned.sum().item()} of {ids.numel()} lookups "
            f"(max local row {(uniq_d[mine] - lo).max().item() if mine.any() else -1})"
        )
        if 0 < owned.sum().item() < ids.numel():
            for name in _WRONG:
                wrong = _WRONG[name](ids, tw, ts, lo, hi)
                with pytest.raises(AssertionError):
                    op.compare(out, wrong)
                print(
                    f"  control real rank-{rank} {name}: rejected, mismatch fraction {_mismatch(out, wrong):.6f}"
                )
        total += out
    op.compare(total, _summed(expected))
    del buf_w, buf_s
    torch.cuda.empty_cache()


@contextmanager
def _ieee_fp32_matmul():
    """The golden reference runs its fp32 GEMMs with TF32 off."""
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def _fp8_act_qdq(x):
    """The reference's FP8 1x32 activation quantize-dequantize: amax floored at 1e-4,
    scale 2^ceil(log2(amax / 448)) from the fp32 bits, E4M3 of the clamped quotient."""
    xf = x.float().unflatten(-1, (-1, 32))
    v = xf.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = v.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    sc = ((e + 127) << 23).view(torch.float32)
    q = (xf / sc.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * sc.unsqueeze(-1)).flatten(-2)


@pytest.mark.parametrize("case", _REAL_CASES)
def test_real_rows_reproduce_golden_engram_value(case):
    """Gathered rows -> the reference's next step (FP8 1x32 activation quantize,
    IEEE fp32 GEMM with the dequantized 32x32-block ``wkv``, bf16) -> the value
    slice must match the golden ``engram_value`` of layer 1."""
    _skip_without_real()
    from safetensors import safe_open

    (hid,), uniq, w, s = _real_rows((case,))
    ids = torch.searchsorted(uniq, hid.contiguous()).to(DEV)
    out = op(ids, w.to(DEV), s.to(DEV), 0, uniq.numel())
    wkv = {}
    for name in ("layers.1.engram.wkv.weight", "layers.1.engram.wkv.scale"):
        with safe_open(_weight_file(name), framework="pt", device="cpu") as f:
            wkv[name] = f.get_tensor(name).to(DEV)
    ww, ws = wkv["layers.1.engram.wkv.weight"], wkv["layers.1.engram.wkv.scale"]
    n, k = ww.shape
    sc = ws.float().repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]
    wkv_f32 = ww.float() * sc
    golden = torch.load(
        os.path.join(_GOLDEN_DIR, case, "layers", "L01.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    golden_value = golden["engram_value"].to(DEV)

    def value_of(rows_bf16):
        with _ieee_fp32_matmul():
            kv = torch.matmul(_fp8_act_qdq(rows_bf16.flatten(-2)), wkv_f32.t()).to(torch.bfloat16)
        return kv[:, n - golden_value.shape[-1] :]

    value = value_of(out)
    exact = 1.0 - _mismatch(value, golden_value)
    print(f"\n  {case}: engram_value {tuple(value.shape)} bit-exact vs golden {exact:.6f}")
    torch.testing.assert_close(value, golden_value)
    for name in ("scale_per_16", "e5m2"):
        wrong = value_of(_WRONG[name](ids, w, s, 0, uniq.numel()))
        with pytest.raises(AssertionError):
            torch.testing.assert_close(wrong, golden_value)
        rel = ((wrong.float() - golden_value.float()).norm() / golden_value.float().norm()).item()
        print(
            f"  control golden {case} {name}: rejected, rel L2 {rel:.4g}, "
            f"bit-exact {1.0 - _mismatch(wrong, golden_value):.6f}"
        )


# ─── host_offload at the real size: the checkpoint's own tables in the node-shared pinned file ───
#
# host_offload keeps ONE copy of both Engram tables per node -- [384,006,168, 256] and
# [384,016,682, 256] float8_e4m3fn plus [rows, 8] E8M0 scale bytes, 202.76 GB -- that
# The target's engram_host_table.py fills from the checkpoint, in one of two backends
# ($TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND): ``vmm`` (the default) allocates CUDA VMM host-NUMA
# slabs, 1 GiB each alternating between the sockets, shares them between the node's processes as
# file descriptors, and hands out CUDA views of that host memory; ``shm`` fills a /dev/shm file
# that every rank maps (MAP_SHARED) and cudaHostRegisters, and hands out pinned CPU views. Either
# way every rank gathers its OWN tokens over ALL rows (row_start 0, row_end rows; no zero fill, no
# collective) in the target's form: the [:, layer, :] view of the [T, 2, 24] hash output (row
# stride 48), with the scale table passed as uint8 bytes. One call carries at most max_num_tokens = 16384 tokens per
# rank (this path has no all-gather, on every member). Weight rows reach byte offsets past 2^36 of their table, scale
# rows past 2^31.

_HOST_LAYERS = ((1, REAL_ROWS_L1), (14, REAL_ROWS_L14))
_HOST_T_MAX = 16384
_COMMON_SEED = 31337  # the ids every rank of the four-rank case gathers
_RANK_DEADLINE_S = 1800  # a rank that has not exited by then is killed (by PID) and the case fails


def _offset_line_rows(rows: int):
    """Rows on both sides of each 2^k byte line (k = 31..36) of the 256-byte-row weight table and
    of the 2^31 line of the 8-byte-row scale table (row 2^28, also the weight table's 2^36 line)."""
    lines = [(1 << k) // DIM for k in range(31, 37)] + [(1 << 31) // (DIM // BLOCK)]
    return sorted({r + d for r in lines for d in (-2, -1, 0, 1) if 0 <= r + d < rows})


def _token0_rows(rows: int):
    """24 edge rows for token 0: the last two rows, row 0, the rows just below and at each
    2^31..2^36 weight byte line, and the next-to-last rows to fill the 24 lookups."""
    edge = [rows - 1, rows - 2, 0] + [
        x for k in range(31, 37) for x in ((1 << k) // DIM - 1, (1 << k) // DIM)
    ]
    edge += list(range(rows - 3, rows - 3 - (LOOKUPS - len(edge)), -1))
    assert len(edge) == LOOKUPS
    return edge


def _host_ids(tokens: int, seed: int) -> torch.Tensor:
    """[T, 2, 24] int64 ids in the hash output's layout (layer 1 in column 0, layer 14 in column 1),
    uniform over each table's rows. Token 0 holds ``_token0_rows`` in a seed-dependent order (so
    even T = 1 reaches the last row and every offset line, and consecutive seeds differ), and the
    last 4096 rows, the first 1024 and the offset-line rows are also written at random positions.
    Drawn on the CPU from a seeded generator, so every process draws the same ids."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(tokens, 2, LOOKUPS, dtype=torch.int64)
    for li, (_, rows) in enumerate(_HOST_LAYERS):
        col = torch.randint(0, rows, (tokens * LOOKUPS,), generator=g)
        special = torch.cat(
            [
                torch.arange(rows - 4096, rows),
                torch.arange(1024),
                torch.tensor(_offset_line_rows(rows)),
            ]
        )
        k = min(col.numel() // 4, special.numel())
        col[torch.randperm(col.numel(), generator=g)[:k]] = special[
            torch.randperm(special.numel(), generator=g)[:k]
        ]
        col[:LOOKUPS] = torch.tensor(_token0_rows(rows))[torch.randperm(LOOKUPS, generator=g)]
        ids[:, li, :] = col.view(tokens, LOOKUPS)
    return ids.to(DEV)


def _host_ref(ids: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The reference on a host table without copying the table to the device: the rows ``ids``
    name are read on the CPU through this process's own mapping (``weight`` / ``scale`` are the
    CPU byte views of ``_cpu_bytes``) and dequantized by ``_ref``."""
    flat = ids.reshape(-1).cpu()
    local = torch.arange(flat.numel(), device=DEV).view(ids.shape)
    return op.reference(
        local, weight.view(torch.uint8)[flat], scale.view(torch.uint8)[flat], 0, flat.numel()
    )


def _host_wrong_wrap(
    ids: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, weight_mod: int, scale_mod: int
):
    """The gather with truncated byte offsets: weight row r read at byte (256 r) mod ``weight_mod``
    and scale row r at byte (8 r) mod ``scale_mod`` -- what a 32-bit (or signed 32-bit) offset reads."""
    flat = ids.reshape(-1).cpu()
    local = torch.arange(flat.numel(), device=DEV).view(ids.shape)
    w = weight.view(torch.uint8)[flat % (weight_mod // DIM)]
    s = scale.view(torch.uint8)[flat % (scale_mod // (DIM // BLOCK))]
    return op.reference(local, w, s, 0, flat.numel())


def _digest(out: torch.Tensor) -> str:
    return hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()


_CKPT_TABLES = {}


def _checkpoint_table(layer: int):
    """(weight [rows, 256], scale [rows, 8]) uint8 CPU views of layer ``layer``'s Engram table in
    the checkpoint's own safetensors shard, mapped read-only with ``mmap`` at the offsets the
    shard's header gives -- independent of engram_host_table's fill and of ``safetensors``."""
    import mmap
    import struct
    import warnings

    if layer not in _CKPT_TABLES:
        views = []
        for kind, dtype in (("weight", "F8_E4M3"), ("scale", "F8_E8M0")):
            name = f"layers.{layer}.engram.embed.{kind}"
            with open(_weight_file(name), "rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                entry = json.loads(f.read(length))[name]
                mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            begin, end = entry["data_offsets"]
            assert (
                entry["dtype"] == dtype and end - begin == entry["shape"][0] * entry["shape"][1]
            ), entry
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", UserWarning
                )  # torch warns that the buffer is read-only
                views.append(
                    torch.frombuffer(
                        mm, dtype=torch.uint8, count=end - begin, offset=8 + length + begin
                    ).view(*entry["shape"])
                )
        _CKPT_TABLES[layer] = tuple(views)
    return _CKPT_TABLES[layer]


def _eht():
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
        engram_host_table,
    )

    return engram_host_table


def _table_backend(table) -> str:
    """``vmm`` for a VmmEngramHostTable, ``shm`` for the /dev/shm file's SharedEngramHostTable."""
    return "vmm" if type(table).__name__ == "VmmEngramHostTable" else "shm"


def _gather_views(table, layer: int):
    """(weight, scale bytes) of ``layer`` exactly as the target hands them to engram_gather:
    ``table.weight(layer)`` and ``table.scale(layer)`` viewed as uint8 (CUDA views of host memory
    under vmm, pinned CPU views under shm)."""
    return table.weight(layer), table.scale(layer).view(torch.uint8)


def _cpu_bytes(table, layer: int):
    """uint8 CPU tensors ``[rows, 256]`` / ``[rows, 8]`` over ``layer``'s host bytes, read by the CPU
    through this process's own host mapping (never through the GPU): the shm views themselves, or
    under vmm the host view ``host_buffer`` exposes of the same VMM allocations."""
    if _table_backend(table) == "shm":
        return table.weight(layer).view(torch.uint8), table.scale(layer).view(torch.uint8)
    out = []
    for kind, cols in (("weight", DIM), ("scale", DIM // BLOCK)):
        buf = table.host_buffer(f"layers.{layer}.engram.embed.{kind}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # torch warns that the buffer is read-only
            out.append(torch.frombuffer(buf, dtype=torch.uint8).view(-1, cols))
    return tuple(out)


def _check_views(table, device) -> None:
    """Each layer's gather views are what the backend promises: CUDA views on ``device`` (vmm;
    ``check_gather_views`` too) or pinned CPU tensors (shm)."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for layer, rows in _HOST_LAYERS:
        hw, hs = table.weight(layer), table.scale(layer)
        assert hw.shape == (rows, DIM) and hw.dtype == torch.float8_e4m3fn, (hw.shape, hw.dtype)
        assert hs.shape == (rows, DIM // BLOCK) and hs.dtype == torch.float8_e8m0fnu, (
            hs.shape,
            hs.dtype,
        )
        if _table_backend(table) == "vmm":
            assert hw.is_cuda and hs.is_cuda and hw.device == hs.device == device, (
                hw.device,
                device,
            )
        else:
            assert hw.device.type == "cpu" and hw.is_pinned() and hs.is_pinned()
    if _table_backend(table) == "vmm":
        table.check_gather_views(device)


@contextmanager
def _backend_env(backend: str):
    """``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND`` for the duration: unset for ``vmm`` (the default
    the target runs with), ``shm`` for the file backend; the previous value restored after."""
    key = "TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND"
    saved = os.environ.get(key)
    if backend == "vmm":
        os.environ.pop(key, None)
    else:
        os.environ[key] = backend
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


@pytest.fixture(scope="module", params=["vmm", "shm"])
def real_host_table(request):
    """The checkpoint's own node-shared table, opened through the target's entry point
    ``open_shared_engram_table`` as local rank 0 on this process's device, under each backend:

    * ``vmm`` (``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND`` unset, the default): import the table
      from a process on this node that already serves it, or create the VMM host-NUMA slabs, fill
      them from the checkpoint and serve them on the node's abstract socket. Closed afterwards;
      the host memory goes when the last process mapping it closes.
    * ``shm``: fill, or reuse a complete file whose manifest names the same checkpoint bytes, at
      the module's default path; then map and ``cudaHostRegister``. A file this fixture filled is
      removed afterwards if no other process holds it; a file that was already there is left in
      place, as the module's reuse contract has it.

    The backend variable stays set for the fixture's lifetime, so the four-rank case's processes
    inherit it."""
    if not os.path.isfile(os.path.join(_CKPT_DIR, "model.safetensors.index.json")):
        pytest.skip(f"checkpoint ({_CKPT_DIR}) not present")
    backend = request.param
    with _backend_env(backend):
        eht = _eht()
        assert eht.host_table_backend() == backend
        path = eht.default_table_path()
        existed = backend == "shm" and eht.describe_shared_engram_table(path)["state"] == "ready"
        torch.cuda.synchronize()
        alloc0, free0 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
        t0 = time.monotonic()
        table = eht.open_shared_engram_table(_CKPT_DIR, local_rank=0, path=path, device=DEV)
        seconds = time.monotonic() - t0
        torch.cuda.synchronize()
        assert _table_backend(table) == backend, type(table).__name__
        info = {
            "backend": backend,
            "path": table.path,
            "alloc_delta": torch.cuda.memory_allocated() - alloc0,
            "free_delta": free0 - torch.cuda.mem_get_info()[0],
            "fill": table.manifest.get("fill") or {},
        }
        fill = info["fill"]
        if backend == "shm":
            info["existed"] = existed
            info["inode"] = os.stat(path).st_ino
            print(
                f"\n  shm host table {path} (inode {info['inode']}): {'reused' if existed else 'filled'}, opened in "
                f"{seconds:.2f} s (registration {table.register_seconds:.2f} s, read_only {table.read_only}); "
                f"{table.nbytes} table bytes; fill record: {fill.get('seconds', 0):.1f} s, numa {fill.get('numa')}, "
                f"huge-page bytes {fill.get('huge_page_bytes')}; device memory_allocated delta "
                f"{info['alloc_delta']} B, "
                f"cudaMemGetInfo delta {info['free_delta'] / 2**20:.1f} MiB"
            )
        else:
            info["existed"] = not table.created
            print(
                f"\n  vmm host table: {table.describe()}; opened in {seconds:.2f} s; {table.nbytes} table bytes in "
                f"{table.allocated_bytes} allocated; fill record: pid {fill.get('pid')}, alloc "
                f"{fill.get('alloc_seconds', 0):.1f} s, "
                f"fill {fill.get('seconds', 0):.1f} s, granularity {fill.get('granularity')}, numa bytes "
                f"{fill.get('numa_bytes')}; device memory_allocated delta {info['alloc_delta']} B, cudaMemGetInfo "
                f"delta {info['free_delta'] / 2**20:.1f} MiB"
            )
        try:
            yield table, info
        finally:
            torch.cuda.synchronize()
            table.close()
            if backend == "shm" and not existed:
                eht.remove_shared_engram_table(path)


# ─── the vmm table created by a process that sees ONE GPU ───
#
# The driver refuses cuMemCreate(HOST_NUMA, node) with CUDA_ERROR_INVALID_VALUE for a CPU socket that
# none of the process's visible GPUs is attached to. The vmm backend
# probes the sockets first (_vmm_usable_nodes) and places every slab on the ones it may allocate
# on: both with GPUs of both sockets visible, else its GPUs' socket; a process that sees only a GPU
# of the other socket imports the same slabs and reads them over the socket link. The cases below
# run that in separate processes, each with exactly one GPU visible (CUDA_VISIBLE_DEVICES), because
# visibility is fixed when CUDA initializes.

_ONE_GPU_CAPTURE_T = (
    1,
    6,
    8,
    32,
    48,
    128,
    192,
    256,
    768,
    1536,
)  # G and 6 G for G in 1, 8, 32, 128, 256
_ONE_GPU_WAIT_S = 900  # how long a creator keeps serving while the importer runs


def _vmm_create_probe() -> dict:
    """In this process (its current device, with whatever GPUs are visible): per CPU NUMA node,
    whether ``cuMemCreate`` of one allocation-granularity host-NUMA slab, exactly as the vmm backend
    asks for it (``_vmm_prop``), succeeds; the device's own NUMA node; and the sockets the backend's
    own ``_vmm_usable_nodes`` finds usable."""
    from cuda.bindings import driver as cu

    eht = _eht()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()  # the primary context is current
    dev = torch.cuda.current_device()
    err, numa = cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    out = {
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_numa": int(numa),
        "create": {},
    }
    nodes = eht._cpu_memory_nodes()
    for node in nodes:
        gran = eht._vmm_granularity([node])
        err, handle = cu.cuMemCreate(gran, eht._vmm_prop(node), 0)
        out["create"][str(node)] = err.name
        if err == cu.CUresult.CUDA_SUCCESS:
            cu.cuMemRelease(handle)
    usable, refused = eht._vmm_usable_nodes(nodes)
    out["usable"] = usable
    out["refused"] = {str(n): why for n, why in refused.items()}
    return out


def _one_gpu_create_main(name: str, out_dir: str) -> None:
    """The creator, run as ``python <this file> --vmm-one-gpu create --name <name> --out <dir>`` with one
    GPU visible: create the table under ``name`` (no process serves it), check the slab placement and
    the checkpoint's bytes, run every T 1..16384 through both layers against the reference and the
    captured steps, write ``create.json``, then keep serving until ``<dir>/done`` appears (the
    importer's turn) and close. Any failed check raises (exit code 1)."""
    eht = _eht()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    alloc0 = torch.cuda.memory_allocated()
    t0 = time.monotonic()
    table = eht.open_vmm_engram_table(_CKPT_DIR, local_rank=0, device=dev, wait_s=300.0, name=name)
    open_s = time.monotonic() - t0
    torch.cuda.synchronize()
    alloc_delta = torch.cuda.memory_allocated() - alloc0
    probe = _vmm_create_probe()
    own = probe["device_numa"]
    fill = table.manifest.get("fill") or {}
    numa = table.numa_bytes()
    assert table.created and table.path == name and alloc_delta == 0, (
        table.created,
        table.path,
        alloc_delta,
    )
    assert numa == {own: table.allocated_bytes}, (numa, own)  # every slab on this GPU's socket
    assert fill.get("device_host_nodes") == {"0": own}, fill.get("device_host_nodes")
    assert fill.get("numa_nodes_refused") == {
        str(n): "cuMemCreate: CUDA_ERROR_INVALID_VALUE" for n in (0, 1) if n != own
    }, fill.get("numa_nodes_refused")
    _check_views(table, dev)
    checked, bad = table.verify(full=False)
    assert checked > 0 and not bad, (checked, bad)
    result = {
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "device_numa": own,
        "open_s": round(open_s, 2),
        "alloc_s": round(fill.get("alloc_seconds", 0.0), 2),
        "fill_s": round(fill.get("seconds", 0.0), 2),
        "numa_bytes": {str(k): v for k, v in numa.items()},
        "allocated_bytes": table.allocated_bytes,
        "slabs": len(table._slabs),
        "refused": fill.get("numa_nodes_refused"),
        "crc_chunks_checked": checked,
        "sweep_s": {},
        "digest": {},
    }
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_gather import (  # noqa: E501
        _engram_gather_kernel as fn,
    )

    tabs = [_gather_views(table, layer) + (rows,) for layer, rows in _HOST_LAYERS]
    cpu = [_cpu_bytes(table, layer) for layer, _ in _HOST_LAYERS]
    # the table's bytes on far, edge and random rows equal the checkpoint's own (independent mmap)
    for li, (layer, rows) in enumerate(_HOST_LAYERS):
        ck_w, ck_s = _checkpoint_table(layer)
        g = torch.Generator().manual_seed(7300 + layer)
        picked = torch.cat(
            [
                torch.arange(rows - 4096, rows),
                torch.arange(1024),
                torch.tensor(_offset_line_rows(rows)),
                torch.randint(0, rows, (8192,), generator=g),
            ]
        )
        assert torch.equal(cpu[li][0][picked], ck_w[picked]) and torch.equal(
            cpu[li][1][picked], ck_s[picked]
        ), layer
    # every T 1..16384: the 16384-token call == reference, every prefix its first rows, the guard rows untouched
    hashes = _host_ids(_HOST_T_MAX, seed=8100)
    buf = torch.empty(_HOST_T_MAX + _GUARD, LOOKUPS, DIM, dtype=torch.bfloat16, device=dev)
    bits = buf.view(torch.int16)
    for li, (w, s, rows) in enumerate(tabs):
        ids = hashes[:, li, :]
        full = op(ids, w, s, 0, rows)
        op.compare(full, _host_ref(ids, *cpu[li]))
        wrong = _host_wrong_wrap(ids, *cpu[li], 1 << 32, 1 << 64)
        assert _mismatch(full, wrong) > 0.5, "the 32-bit weight-offset control was not rejected"
        result["digest"][str(_HOST_LAYERS[li][0])] = _digest(full)
        full_bits = full.view(torch.int16)
        bits.fill_(_POISON)
        bad_t = torch.zeros((), dtype=torch.int64, device=dev)
        over = torch.zeros((), dtype=torch.int64, device=dev)
        t1 = time.monotonic()
        before, settled = _jit_variants(fn), None
        for r in range(1, _HOST_T_MAX + 1):
            op(ids[:r], w, s, 0, rows, out=buf[:r])
            bad_t += (bits[:r] != full_bits[:r]).sum()
            over += (bits[r : r + _GUARD] != _POISON).sum()
            if r == 2:
                settled = _jit_variants(fn)
        torch.cuda.synchronize()
        assert bad_t.item() == 0 and over.item() == 0, (bad_t.item(), over.item())
        assert _jit_variants(fn) == settled and settled - before <= 2, (
            before,
            settled,
            _jit_variants(fn),
        )
        result["sweep_s"][str(_HOST_LAYERS[li][0])] = round(time.monotonic() - t1, 1)
    del buf
    # capture: both layers in one graph at G and 6 G counts, 3 replays on new ids each
    stale_min = 1.0
    for tokens in _ONE_GPU_CAPTURE_T:
        static = _host_ids(tokens, seed=0)
        views = [static[:, 0, :], static[:, 1, :]]
        capture_ids = static.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for v, (w, s, rows) in zip(views, tabs):
                op(v, w, s, 0, rows)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [op(v, w, s, 0, rows) for v, (w, s, rows) in zip(views, tabs)]
        for rep in (1, 2, 3):
            static.copy_(_host_ids(tokens, seed=rep))
            for o in outs:
                o.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for li, (v, (w, s, rows)) in enumerate(zip(views, tabs)):
                assert torch.equal(
                    outs[li].view(torch.int16), op(v, w, s, 0, rows).view(torch.int16)
                )
                expected = _host_ref(v, *cpu[li])
                op.compare(outs[li], expected)
                stale_min = min(
                    stale_min, _mismatch(expected, _host_ref(capture_ids[:, li, :], *cpu[li]))
                )
        del graph, outs
    assert stale_min > 0.5, stale_min
    result["captured_T"] = list(_ONE_GPU_CAPTURE_T)
    result["stale_min"] = round(stale_min, 6)
    with open(os.path.join(out_dir, "create.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(out_dir, "serving"), "w") as f:
        f.write(str(os.getpid()))
    deadline = time.monotonic() + _ONE_GPU_WAIT_S
    while not os.path.exists(os.path.join(out_dir, "done")) and time.monotonic() < deadline:
        time.sleep(0.5)
    del tabs, cpu
    torch.cuda.synchronize()
    table.close()
    assert os.path.exists(os.path.join(out_dir, "done")), (
        "the importer did not finish while this process served"
    )


def _one_gpu_import_main(name: str, out_dir: str) -> None:
    """The importer, run as ``python <this file> --vmm-one-gpu import --name <name> --out <dir>`` with one GPU
    of the OTHER socket visible, while the creator serves: import the table, check that no slab is on this
    GPU's socket (it reads every row over the socket link), and gather the creator's 16384 tokens on both
    layers (bit-exact to the reference and to the creator's digest) and the captured steps."""
    eht = _eht()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    alloc0 = torch.cuda.memory_allocated()
    t0 = time.monotonic()
    table = eht.open_vmm_engram_table(_CKPT_DIR, local_rank=1, device=dev, wait_s=120.0, name=name)
    open_s = time.monotonic() - t0
    torch.cuda.synchronize()
    probe = _vmm_create_probe()
    numa = table.numa_bytes()
    assert not table.created and table.path == name and torch.cuda.memory_allocated() == alloc0
    assert probe["device_numa"] not in numa and len(numa) == 1, (probe["device_numa"], numa)
    _check_views(table, dev)
    tabs = [_gather_views(table, layer) + (rows,) for layer, rows in _HOST_LAYERS]
    cpu = [_cpu_bytes(table, layer) for layer, _ in _HOST_LAYERS]
    hashes = _host_ids(_HOST_T_MAX, seed=8100)
    digest = {}
    for li, (w, s, rows) in enumerate(tabs):
        out = op(hashes[:, li, :], w, s, 0, rows)
        op.compare(out, _host_ref(hashes[:, li, :], *cpu[li]))
        digest[str(_HOST_LAYERS[li][0])] = _digest(out)
    for tokens in (1, 48, 1536):
        static = _host_ids(tokens, seed=100)
        views = [static[:, 0, :], static[:, 1, :]]
        for v, (w, s, rows) in zip(views, tabs):
            op(v, w, s, 0, rows)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [op(v, w, s, 0, rows) for v, (w, s, rows) in zip(views, tabs)]
        for rep in (1, 2):
            static.copy_(_host_ids(tokens, seed=1000 * rep))
            graph.replay()
            torch.cuda.synchronize()
            for li, (v, (w, s, rows)) in enumerate(zip(views, tabs)):
                assert torch.equal(
                    outs[li].view(torch.int16), op(v, w, s, 0, rows).view(torch.int16)
                )
                op.compare(outs[li], _host_ref(v, *cpu[li]))
        del graph, outs
    result = {
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "device_numa": probe["device_numa"],
        "numa_bytes": {str(k): v for k, v in numa.items()},
        "open_s": round(open_s, 2),
        "digest": digest,
        "captured_T": [1, 48, 1536],
    }
    del tabs, cpu
    torch.cuda.synchronize()
    table.close()
    with open(os.path.join(out_dir, "import.json"), "w") as f:
        json.dump(result, f)


def _spawn(args, env, log_path):
    with open(log_path, "w") as log:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())] + args,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _wait_all(procs, logs, deadline_s):
    """Wait for ``procs`` up to ``deadline_s``; kill (by PID) any still running; fail with the logs' tails."""
    deadline = time.monotonic() + deadline_s
    while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
        time.sleep(1)
    wedged = [i for i, p in enumerate(procs) if p.poll() is None]
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    tails = "".join(f"\n--- {log.name} ---\n{log.read_text()[-3000:]}" for log in logs)
    assert not wedged, f"processes {wedged} did not exit within {deadline_s} s{tails}"
    assert [p.returncode for p in procs] == [0] * len(procs), (
        f"exit codes {[p.returncode for p in procs]}{tails}"
    )


def test_vmm_table_created_by_a_one_gpu_process(tmp_path):
    """The vmm table with one GPU visible.

    (1) Probe. In this process (four GPUs visible) ``cuMemCreate`` of a host-NUMA slab, as the backend
    asks for it, succeeds on both CPU NUMA nodes, and ``_vmm_usable_nodes`` finds both usable. In a
    process that sees one GPU, run once per GPU: it succeeds on that GPU's own node, returns
    CUDA_ERROR_INVALID_VALUE on the other socket's node, and ``_vmm_usable_nodes`` finds only the own
    node usable (the other refused with that error).
    (2) Create and read, once with a GPU of each socket visible (the first and the last visible GPU):
    a process that sees only that GPU creates the table under a name no process serves; it succeeds,
    every slab lands on that GPU's socket, the manifest records the other socket as refused, the views
    are CUDA views with no device memory behind them, sampled chunk CRCs re-verify, and the bytes on
    far, edge and random rows equal the checkpoint's. Through that table, both layers: the 16384-token
    call equals the reference (and rejects the 32-bit offset control), every prefix r = 1..16384 returns
    exactly its first r rows with the guard rows untouched and no new Triton variant after r = 2, and
    both layers in one graph captured at T = G and 6 G for G in 1, 8, 32, 128, 256 replay 3 times on new
    ids equal to eager and to the reference.
    (3) While it serves, a process that sees only a GPU of the OTHER socket imports the table (no slab
    on its own socket) and gathers the same 16384 tokens: bit-exact to the reference and the same
    SHA-256 as the creator; captured at T 1, 48, 1536, replay == eager == reference.
    A process that has not exited within the deadline is killed by PID and the case fails."""
    if not os.path.isfile(os.path.join(_CKPT_DIR, "model.safetensors.index.json")):
        pytest.skip(f"checkpoint ({_CKPT_DIR}) not present")
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    visible = visible or [str(i) for i in range(torch.cuda.device_count())]
    if len(visible) < WORLD:
        pytest.skip(f"needs {WORLD} visible GPUs, one process per GPU; {len(visible)} visible")
    here = _vmm_create_probe()
    if len(here["usable"]) < 2:
        pytest.skip(
            f"needs visible GPUs on two NUMA nodes (a creator on one socket, an importer on the "
            f"other); the visible GPUs reach NUMA nodes {here['usable']} only"
        )
    print(
        f"\n  {len(visible)} GPUs visible (current device NUMA node {here['device_numa']}): cuMemCreate per node "
        f"{here['create']}; usable {here['usable']}, refused {here['refused']}"
    )
    assert set(here["create"].values()) == {"CUDA_SUCCESS"} and len(here["create"]) == 2, here
    assert here["usable"] == [0, 1] and not here["refused"], here
    repo_root = Path(__file__).resolve().parents[8]
    base_env = dict(os.environ)
    # the repo root (tensorrt_llm), tests/unittest (the utils package this file imports), tests
    # (test_common, which utils imports; the unittest conftest adds it for pytest) and
    # tests/unittest/_torch/modeling_v2 (_validating, which pytest reaches via __extra_import_path__)
    base_env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            str(repo_root),
            str(Path(__file__).resolve().parents[6]),
            str(Path(__file__).resolve().parents[7]),
            str(Path(__file__).resolve().parents[4]),
            base_env.get("PYTHONPATH", ""),
        )
        if p
    )
    base_env.pop("TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND", None)
    # (1) the probe, one process per GPU
    procs, logs = [], []
    for i, dev in enumerate(visible[:WORLD]):
        logs.append(tmp_path / f"probe{i}.log")
        procs.append(
            _spawn(
                ["--vmm-create-probe", str(tmp_path / f"probe{i}.json")],
                dict(base_env, CUDA_VISIBLE_DEVICES=dev),
                logs[-1],
            )
        )
    _wait_all(procs, logs, _RANK_DEADLINE_S)
    node_of = {}
    for i, dev in enumerate(visible[:WORLD]):
        r = json.loads((tmp_path / f"probe{i}.json").read_text())
        own = r["device_numa"]
        node_of[dev] = own
        print(
            f"  one GPU visible ({r['visible']}, NUMA node {own}): cuMemCreate per node {r['create']}; usable "
            f"{r['usable']}, refused {r['refused']}"
        )
        assert r["create"][str(own)] == "CUDA_SUCCESS", r
        assert all(
            v == "CUDA_ERROR_INVALID_VALUE" for n, v in r["create"].items() if n != str(own)
        ), r
        assert r["usable"] == [own] and r["refused"] == {
            str(1 - own): "cuMemCreate: CUDA_ERROR_INVALID_VALUE"
        }, r
    # (2) + (3): a creator on each socket, an importer on the other
    creators = [visible[0], visible[WORLD - 1]]
    assert node_of[creators[0]] != node_of[creators[1]], node_of
    for k, cdev in enumerate(creators):
        idev = next(d for d in visible[:WORLD] if node_of[d] != node_of[cdev])
        name = f"cg-engram-gather-onegpu-{os.getpid()}-{k}"
        run = tmp_path / f"run{k}"
        run.mkdir()
        clog, ilog = run / "create.log", run / "import.log"
        creator = _spawn(
            ["--vmm-one-gpu", "create", "--name", name, "--out", str(run)],
            dict(base_env, CUDA_VISIBLE_DEVICES=cdev),
            clog,
        )
        deadline = time.monotonic() + _RANK_DEADLINE_S
        while (
            not (run / "serving").exists()
            and creator.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(1)
        try:
            if (run / "serving").exists():
                importer = _spawn(
                    ["--vmm-one-gpu", "import", "--name", name, "--out", str(run)],
                    dict(base_env, CUDA_VISIBLE_DEVICES=idev),
                    ilog,
                )
                _wait_all([importer], [ilog], 600)
        finally:
            (run / "done").write_text("done")  # the creator stops serving and exits either way
            _wait_all(
                [creator],
                [clog] + ([ilog] if ilog.exists() else []),
                max(60.0, deadline - time.monotonic()),
            )
        c = json.loads((run / "create.json").read_text())
        i_ = json.loads((run / "import.json").read_text())
        print(
            f"  one GPU visible ({cdev}, NUMA node {c['device_numa']}) created the table in {c['open_s']} s (alloc "
            f"{c['alloc_s']} s, fill {c['fill_s']} s): {c['slabs']} slabs, {c['allocated_bytes']} bytes, NUMA bytes "
            f"{c['numa_bytes']}, refused {c['refused']}; {c['crc_chunks_checked']} chunk CRCs re-verified; every T "
            f"1..{_HOST_T_MAX} bit-exact (sweep {c['sweep_s']} s per layer); captured at T {c['captured_T']}, replay "
            f"== eager == reference (min fraction differing from the capture-time ids {c['stale_min']})"
        )
        print(
            f"  one GPU visible ({idev}, NUMA node {i_['device_numa']}) imported it in {i_['open_s']} s (slabs on "
            f"{list(i_['numa_bytes'])}): {_HOST_T_MAX} tokens sha256 L1 {i_['digest']['1'][:16]} L14 "
            f"{i_['digest']['14'][:16]} == the creator's, bit-exact; captured at T {i_['captured_T']}"
        )
        assert c["numa_bytes"] == {str(c["device_numa"]): c["allocated_bytes"]}, c
        assert i_["device_numa"] != c["device_numa"] and list(i_["numa_bytes"]) == [
            str(c["device_numa"])
        ], (c, i_)
        assert i_["digest"] == c["digest"], (i_["digest"], c["digest"])


def test_real_host_table_far_and_random_rows_equal_the_checkpoint(real_host_table):
    """Both of the checkpoint's tables through the node's shared host copy (each backend), in the
    target's call form (``[:, layer, :]`` view of a ``[T, 2, 24]`` hash buffer, uint8 scale view,
    rows ``[0, rows)``): the last 4096 rows (the deepest, 384,006,167 / 384,016,681, start 98.3 GB
    into the weight table), the first 1024, the rows on both sides of the 2^31 .. 2^36 byte lines of
    the weight table and of the 2^31 line of the scale table, the four-rank shard boundaries and 8192
    random rows are gathered and compared bit for bit with the dequantization of the SAME rows read
    from the checkpoint's safetensors shards through an independent mmap; the table's bytes on those
    rows, read by the CPU, equal the checkpoint's. The views are CUDA views on this device (vmm) or
    pinned CPU tensors (shm); no device memory holds the table. Controls, each rejected by the same
    gate: a 32-bit weight byte offset, a signed 32-bit scale byte offset, the scale applied per 16
    channels, E5M2 decoding."""
    table, info = real_host_table
    assert table.layers == [1, 14]
    assert (
        table.nbytes
        == sum(rows * (DIM + DIM // BLOCK) for _, rows in _HOST_LAYERS)
        == 202_758_032_400
    )
    assert info["alloc_delta"] == 0 and info["free_delta"] < (1 << 30), (
        info
    )  # no device copy of 202.76 GB
    _check_views(table, DEV)
    if info["backend"] == "vmm":
        print(
            f"\n  vmm: {table.allocated_bytes} bytes allocated over NUMA nodes {table.numa_bytes()}"
        )
    for li, (layer, rows) in enumerate(_HOST_LAYERS):
        hw, hs = _gather_views(table, layer)
        cw, cs = _cpu_bytes(table, layer)
        ck_w, ck_s = _checkpoint_table(layer)
        assert ck_w.shape == (rows, DIM) and ck_s.shape == (rows, DIM // BLOCK)
        part = (rows + WORLD - 1) // WORLD
        g = torch.Generator().manual_seed(7100 + layer)
        picked = torch.cat(
            [
                torch.arange(rows - 4096, rows),
                torch.arange(1024),
                torch.tensor(_offset_line_rows(rows)),
                torch.tensor([b + d for b in (part, 2 * part, 3 * part) for d in (-2, -1, 0, 1)]),
                torch.randint(0, rows, (8192,), generator=g),
            ]
        )
        picked = torch.cat(
            [picked, torch.randint(0, rows, (-picked.numel() % LOOKUPS,), generator=g)]
        )
        tokens = picked.numel() // LOOKUPS
        hashes = torch.randint(
            0,
            REAL_ROWS_L1,
            (tokens, 2, LOOKUPS),
            dtype=torch.int64,
            device=DEV,
            generator=_gen(7200 + layer),
        )
        hashes[:, li, :] = picked.view(tokens, LOOKUPS).to(DEV)
        ids = hashes[:, li, :]
        assert ids.stride() == (2 * LOOKUPS, 1)
        out = op(ids, hw, hs, 0, rows)
        t0 = time.monotonic()
        ck_rows_w, ck_rows_s = ck_w[picked], ck_s[picked]  # page faults into the checkpoint shards
        read_s = time.monotonic() - t0
        local = torch.arange(picked.numel(), device=DEV).view(tokens, LOOKUPS)
        expected = op.reference(local, ck_rows_w, ck_rows_s, 0, picked.numel())
        op.compare(out, expected)
        assert torch.equal(cw[picked], ck_rows_w) and torch.equal(cs[picked], ck_rows_s)
        deepest = int(picked.max())
        print(
            f"\n  {info['backend']} layer {layer}: {picked.numel()} rows ({tokens} tokens x 24) incl. the last "
            "4096, the 2^31..2^36 "
            f"byte lines and 8192 random rows == checkpoint (mmap, {read_s:.1f} s), bit-exact; deepest row "
            f"{deepest}: weight byte offset {deepest * DIM}, scale byte offset {deepest * (DIM // BLOCK)}"
        )
        for label, weight_mod, scale_mod in (
            ("32-bit weight byte offset", 1 << 32, 1 << 64),
            ("signed 32-bit scale byte offset", 1 << 64, 1 << 31),
        ):
            wrong = _host_wrong_wrap(ids, cw, cs, weight_mod, scale_mod)
            with pytest.raises(AssertionError):
                op.compare(out, wrong)
            print(
                f"  control {info['backend']} L{layer} {label}: rejected, mismatch fraction {_mismatch(out, wrong):.6f}"
            )
        _assert_controls_fail(
            out,
            local,
            ck_rows_w,
            ck_rows_s,
            0,
            picked.numel(),
            ("scale_per_16", "e5m2"),
            f"real host {info['backend']} L{layer}",
        )


def test_real_host_table_every_row_count_is_a_prefix_of_16384(real_host_table):
    """The prefix rule through the real mapping, per layer, in the target's form: the 16384-token call
    (``_host_ids``: far rows and every offset line included) is bit-exact to the reference built from
    the table's own rows (read on the CPU through the mapping; the previous case ties those bytes to
    the checkpoint) and rejects the 32-bit offset control, under each backend; then every prefix ``ids[:r]``,
    r = 1..16384, returns exactly the first r rows of that call into a row-prefix view of one
    poisoned buffer whose 64 rows past the prefix keep the poison. No Triton variant appears
    after r = 2 (n_rows = 24 r is divisible by 16 exactly for even r)."""
    table, info = real_host_table
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_gather import (  # noqa: E501
        _engram_gather_kernel as fn,
    )

    hashes = _host_ids(_HOST_T_MAX, seed=8100)
    buf = torch.empty(_HOST_T_MAX + _GUARD, LOOKUPS, DIM, dtype=torch.bfloat16, device=DEV)
    bits = buf.view(torch.int16)
    for li, (layer, rows) in enumerate(_HOST_LAYERS):
        ids = hashes[:, li, :]
        hw, hs = _gather_views(table, layer)
        cw, cs = _cpu_bytes(table, layer)
        full = op(ids, hw, hs, 0, rows)
        op.compare(full, _host_ref(ids, cw, cs))
        with pytest.raises(AssertionError):
            op.compare(full, _host_wrong_wrap(ids, cw, cs, 1 << 32, 1 << 64))
        full_bits = full.view(torch.int16)
        bits.fill_(_POISON)
        bad = torch.zeros((), dtype=torch.int64, device=DEV)
        over = torch.zeros((), dtype=torch.int64, device=DEV)
        before = _jit_variants(fn)
        settled = None
        t0 = time.monotonic()
        for r in range(1, _HOST_T_MAX + 1):
            res = op(ids[:r], hw, hs, 0, rows, out=buf[:r])
            assert res.data_ptr() == buf.data_ptr()
            bad += (bits[:r] != full_bits[:r]).sum()
            over += (bits[r : r + _GUARD] != _POISON).sum()
            if r == 2:
                settled = _jit_variants(fn)
        torch.cuda.synchronize()
        after = _jit_variants(fn)
        far = int(ids.max().item())
        print(
            f"\n  {info['backend']} layer {layer}: T {_HOST_T_MAX} bit-exact (deepest row {far}, byte offset "
            f"{far * DIM}); r = 1..{_HOST_T_MAX} "
            f"in {time.monotonic() - t0:.1f} s: {bad.item()} elements differ from the {_HOST_T_MAX}-row result, "
            f"{over.item()} guard elements written; kernel variants {before} before, {settled} after r = 2, {after} "
            f"after r = {_HOST_T_MAX}"
        )
        assert bad.item() == 0 and over.item() == 0, (bad.item(), over.item())
        assert after == settled and settled - before <= 2, (before, settled, after)
    del buf


@pytest.mark.parametrize("tokens", _CAPTURE_T)
def test_real_host_table_cuda_graph_at_every_capture_batch(real_host_table, tokens):
    """The target's captured Engram step at every T of the capture set (Gset graph batches G and 6 G
    for the DSpark verify step, 82 counts up to 1536), under each backend: both layers' gathers through the real
    mapping in one graph, ``out`` allocated inside the capture, after an eager warm-up on a side stream. Three replays
    on new ids (far rows and every offset line in token 0) written in place into the static
    ``[T, 2, 24]`` buffer: each replay equals the eager call bit for bit and the reference, and
    differs from the capture-time ids' rows (it read the new ids)."""
    table, info = real_host_table
    tabs = [_gather_views(table, layer) + (rows,) for layer, rows in _HOST_LAYERS]
    cpu = [_cpu_bytes(table, layer) for layer, _ in _HOST_LAYERS]
    static = _host_ids(tokens, seed=0)
    views = [static[:, 0, :], static[:, 1, :]]
    capture_ids = static.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for v, (w, s, rows) in zip(views, tabs):
            op(v, w, s, 0, rows)  # engine warm-up at this shape
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [op(v, w, s, 0, rows) for v, (w, s, rows) in zip(views, tabs)]
    for rep in (1, 2, 3):
        static.copy_(_host_ids(tokens, seed=rep))
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for li, (v, (w, s, rows)) in enumerate(zip(views, tabs)):
            eager = op(v, w, s, 0, rows)
            assert torch.equal(outs[li].view(torch.int16), eager.view(torch.int16)), (
                f"layer column {li}: replay != eager"
            )
            expected = _host_ref(v, *cpu[li])
            op.compare(outs[li], expected)
            stale = _host_ref(capture_ids[:, li, :], *cpu[li])
            assert not torch.equal(expected.view(torch.int16), stale.view(torch.int16)), (
                "the replay did not read new ids"
            )
            if rep == 1 and li == 0:
                print(
                    f"\n  {info['backend']} T={tokens}: 2 layers x 3 replays == eager == reference bit for bit; "
                    "layer 1 differs from "
                    f"the capture-time ids' rows on {_mismatch(outs[li], stale):.6f}"
                )
    del graph, outs


def _host_rank_main(rank: int, out_dir: str) -> None:
    """One engine-like rank of the four-rank case, run as ``python <this file> --engram-host-rank
    <rank> --out <dir>``: attach through ``open_shared_engram_table`` on its own GPU, gather, and
    write what it saw to ``<dir>/rank<rank>.json``. Any failed check raises (exit code 1)."""
    eht = _eht()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    alloc0 = torch.cuda.memory_allocated()
    path = eht.default_table_path()
    backend = eht.host_table_backend()
    t0 = time.monotonic()
    table = eht.open_shared_engram_table(
        _CKPT_DIR, local_rank=rank, path=path, device=dev, wait_s=600.0
    )
    open_s = time.monotonic() - t0
    torch.cuda.synchronize()
    alloc_delta = torch.cuda.memory_allocated() - alloc0
    assert _table_backend(table) == backend, (type(table).__name__, backend)
    _check_views(table, dev)
    assert alloc_delta == 0, alloc_delta
    tabs = [_gather_views(table, layer) + (rows,) for layer, rows in _HOST_LAYERS]
    cpu = [_cpu_bytes(table, layer) for layer, _ in _HOST_LAYERS]
    common = _host_ids(_HOST_T_MAX, seed=_COMMON_SEED)
    digest = {}
    for li, (w, s, rows) in enumerate(tabs):
        out = op(common[:, li, :], w, s, 0, rows)
        op.compare(out, _host_ref(common[:, li, :], *cpu[li]))
        digest[str(_HOST_LAYERS[li][0])] = _digest(out)
    own_tokens = 4096 - 7 * rank  # the ranks' own, unequal token counts
    own = _host_ids(own_tokens, seed=9000 + rank)
    for li, (w, s, rows) in enumerate(tabs):
        op.compare(op(own[:, li, :], w, s, 0, rows), _host_ref(own[:, li, :], *cpu[li]))
    captured = []
    for tokens in (1, 48, 192, 1536):
        static = _host_ids(tokens, seed=100 + rank)
        views = [static[:, 0, :], static[:, 1, :]]
        for v, (w, s, rows) in zip(views, tabs):
            op(v, w, s, 0, rows)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [op(v, w, s, 0, rows) for v, (w, s, rows) in zip(views, tabs)]
        for rep in (1, 2):
            static.copy_(_host_ids(tokens, seed=1000 * rep + rank))
            graph.replay()
            torch.cuda.synchronize()
            for li, (v, (w, s, rows)) in enumerate(zip(views, tabs)):
                assert torch.equal(
                    outs[li].view(torch.int16), op(v, w, s, 0, rows).view(torch.int16)
                )
                op.compare(outs[li], _host_ref(v, *cpu[li]))
        captured.append(tokens)
        del graph, outs
    props = torch.cuda.get_device_properties(dev)
    fill = table.manifest.get("fill") or {}
    result = {
        "rank": rank,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": str(getattr(props, "uuid", rank)),
        "backend": backend,
        "path": table.path,
        "inode": os.stat(table.path).st_ino if backend == "shm" else None,
        "created": bool(getattr(table, "created", False)),
        "fill_pid": fill.get("pid"),
        "fill_time": fill.get("unix_time"),
        "weight_l1_address": tabs[0][0].data_ptr(),
        "read_only": table.read_only,
        "open_s": round(open_s, 2),
        "register_s": round(table.register_seconds, 2),
        "alloc_delta": alloc_delta,
        "digest": digest,
        "own_tokens": own_tokens,
        "captured_T": captured,
    }
    del tabs, cpu, w, s
    torch.cuda.synchronize()
    table.close()  # the file stays: the pytest process that launched this rank still holds it
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(result, f)


def test_real_host_table_vmm_slabs_span_both_sockets(real_host_table):
    """Under vmm the table's creator places its slabs on every CPU NUMA node its visible GPUs let it
    allocate on (the driver refuses a host-NUMA allocation on a socket none of them is attached
    to). A creator that saw GPUs of both sockets therefore splits the table about evenly over both,
    which this checks; a table created from one socket cannot, and the case skips."""
    table, info = real_host_table
    if info["backend"] != "vmm":
        pytest.skip("the shm backend's placement is the kernel's page policy, not per-socket slabs")
    eht = _eht()
    nodes = list(eht._cpu_memory_nodes())
    refused = info["fill"].get("numa_nodes_refused") or {}
    usable = [n for n in nodes if str(n) not in refused]
    if len(usable) < 2:
        pytest.skip(
            f"the table's creator could allocate on NUMA nodes {usable} of {nodes} only (refused "
            f"{refused}); spanning both sockets needs visible GPUs on two NUMA nodes"
        )
    numa = table.numa_bytes()
    assert len(numa) == 2 and min(numa.values()) > 0.45 * table.allocated_bytes, numa


def test_real_host_table_four_ranks_read_the_same_mapping(real_host_table, tmp_path):
    """One process per GPU on four GPUs, as the engine's ranks run, under each backend: each opens the
    SAME node copy through ``open_shared_engram_table`` (local rank 0 fills or reuses it, the others
    attach; here every rank reuses the table this process holds: under vmm each imports the
    fds this process serves, under shm each maps the same /dev/shm file), with no device memory for
    the table, and gathers
    (1) the same 16384-token ids on both layers -- every rank's result has the same SHA-256 as this
    process's own call, which is bit-exact to the reference; (2) its own 4096 - 7 rank tokens,
    bit-exact; (3) both layers captured in one graph at T 1, 48, 192 and 1536, two replays on new ids,
    replay == eager == reference. A rank that has not exited within the deadline is killed by PID
    and the case fails."""
    table, info = real_host_table
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    visible = visible or [str(i) for i in range(torch.cuda.device_count())]
    if len(visible) < WORLD:
        pytest.skip(f"runs one process per GPU on {WORLD} GPUs; {len(visible)} visible")
    common = _host_ids(_HOST_T_MAX, seed=_COMMON_SEED)
    want = {}
    for li, (layer, rows) in enumerate(_HOST_LAYERS):
        w, s = _gather_views(table, layer)
        out = op(common[:, li, :], w, s, 0, rows)
        op.compare(out, _host_ref(common[:, li, :], *_cpu_bytes(table, layer)))
        want[str(layer)] = _digest(out)
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), repo_root
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(visible[:WORLD]))
    # the repo root (tensorrt_llm), tests/unittest (the utils package this file imports), tests
    # (test_common, which utils imports; the unittest conftest adds it for pytest) and
    # tests/unittest/_torch/modeling_v2 (_validating, which pytest reaches via __extra_import_path__)
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            str(repo_root),
            str(Path(__file__).resolve().parents[6]),
            str(Path(__file__).resolve().parents[7]),
            str(Path(__file__).resolve().parents[4]),
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    procs = []
    for rank in range(WORLD):
        with open(tmp_path / f"rank{rank}.log", "w") as log:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--engram-host-rank",
                        str(rank),
                        "--out",
                        str(tmp_path),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    deadline = time.monotonic() + _RANK_DEADLINE_S
    while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
        time.sleep(1)
    wedged = [rank for rank, p in enumerate(procs) if p.poll() is None]
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    tails = "".join(
        f"\n--- rank {r} ---\n{(tmp_path / f'rank{r}.log').read_text()[-3000:]}"
        for r in range(WORLD)
    )
    assert not wedged, f"ranks {wedged} did not exit within {_RANK_DEADLINE_S} s{tails}"
    codes = [p.returncode for p in procs]
    assert codes == [0] * WORLD, f"rank exit codes {codes}{tails}"
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(WORLD)]
    print()
    for r in results:
        print(
            f"  {r['backend']} rank {r['rank']} pid {r['pid']} device {r['device_uuid']}: {r['path']} "
            + (
                f"inode {r['inode']} "
                if r["backend"] == "shm"
                else f"imported (created {r['created']}; fill pid {r['fill_pid']}) "
            )
            + f"at 0x{r['weight_l1_address']:x}, opened {r['open_s']} s (map/registration {r['register_s']} s), "
            f"device delta {r['alloc_delta']} B; common {_HOST_T_MAX} tokens sha256 L1 {r['digest']['1'][:16]} "
            f"L14 {r['digest']['14'][:16]}; own {r['own_tokens']} tokens and captures at T {r['captured_T']} bit-exact"
        )
        assert r["backend"] == info["backend"] and r["path"] == info["path"], (r, info)
        if info["backend"] == "shm":
            assert r["inode"] == info["inode"], (r, info)
        else:  # the same allocations: imported, from the fill this process's fixture made or imported
            assert not r["created"] and (r["fill_pid"], r["fill_time"]) == (
                info["fill"].get("pid"),
                info["fill"].get("unix_time"),
            )
        assert r["alloc_delta"] == 0
        assert r["digest"] == want, (r["rank"], r["digest"], want)
    assert (
        len({r["pid"] for r in results}) == WORLD
        and len({r["device_uuid"] for r in results}) == WORLD
    )
    print(
        f"  this process (cuda:0): sha256 L1 {want['1'][:16]} L14 {want['14'][:16]}, bit-exact to the reference"
    )


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="a rank of test_real_host_table_four_ranks_read_the_same_mapping, "
        "or a process of test_vmm_table_created_by_a_one_gpu_process"
    )
    _parser.add_argument("--engram-host-rank", type=int)
    _parser.add_argument("--out")
    _parser.add_argument("--vmm-create-probe")
    _parser.add_argument("--vmm-one-gpu", choices=("create", "import"))
    _parser.add_argument("--name")
    _args = _parser.parse_args()
    if _args.vmm_create_probe:
        with open(_args.vmm_create_probe, "w") as _f:
            json.dump(_vmm_create_probe(), _f)
    elif _args.vmm_one_gpu == "create":
        _one_gpu_create_main(_args.name, _args.out)
    elif _args.vmm_one_gpu == "import":
        _one_gpu_import_main(_args.name, _args.out)
    else:
        _host_rank_main(_args.engram_host_rank, _args.out)
