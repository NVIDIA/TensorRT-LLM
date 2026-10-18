# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``context_kv_commit`` entry (fast path ``context_kv_fork``).

The entry's claim is bitwise equality with the built-in ops it replaces -- the
mask multiply of the stored rows, the context-length update and the read of
the new lengths -- so every cell is gated with an exact comparison of the
whole arenas (as int16), the lengths and the new-length output against the
entry's ``reference`` (the built-in ops on the same arena). The stored rows
are seeded with every BF16 class the multiply treats differently (negative
values, both zeros, subnormals, infinities, NaN), and the cells are replayed
under a CUDA graph with new counts and lengths. The one program's threads all
read the lengths to address the zeroed columns while one thread writes them, so
the compiled kernel must hold a CTA barrier between the two, and many launches
with up to ``MAX_REQUESTS`` requests must all equal the reference.
"""

import re
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    context_kv_commit as commit_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.context_kv_commit import (  # noqa: E501
    context_kv_commit as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "context_kv_commit requires a CUDA device"

#: BF16 bit patterns the zeroing must treat as the BF16 multiply does: both
#: zeros, subnormals, both infinities, the NaN the GPU's conversions produce
#: (the kernels that compute the stored rows emit no other), the largest
#: finite value and an ordinary negative one.
SPECIALS = (0x8000, 0x0000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FFF, 0x7F7F, 0xC2F7)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _arena(slots, layers, capacity, kv_heads, head_dim, g, specials=True):
    arena = 3.0 * torch.randn(
        slots, layers, capacity, kv_heads, head_dim, generator=g, device="cuda"
    )
    arena = arena.to(torch.bfloat16)
    if not specials:
        return arena
    bits = arena.view(torch.int16).view(-1)
    patterns = torch.tensor([p - (1 << 16) if p >= 1 << 15 else p for p in SPECIALS], device="cuda")
    where = torch.randint(0, bits.numel(), (bits.numel() // 7,), generator=g, device="cuda")
    pick = torch.randint(0, len(SPECIALS), where.shape, generator=g, device="cuda")
    bits[where] = patterns[pick].to(torch.int16)
    return arena


def _case(spec, g):
    slots, rows = spec["slots"], spec["rows"]
    layers, kv_heads, head_dim = spec["layers"], spec["kv_heads"], spec["head_dim"]
    gens = spec["gens"]
    capacity = 300 + rows
    max_ctx = capacity - rows
    k_arena = _arena(slots, layers, capacity, kv_heads, head_dim, g)
    v_arena = None if spec.get("no_v") else _arena(slots, layers, capacity, kv_heads, head_dim, g)
    chosen = torch.randperm(slots, generator=g, device="cuda")[:gens]
    ctx_len = torch.randint(0, max_ctx - rows, (slots,), generator=g, device="cuda")
    if spec.get("at_end"):
        ctx_len[chosen] = max_ctx - 2
    accepted = torch.tensor(spec["accepted"], device="cuda")
    accepted = accepted.to(torch.int64 if spec.get("int64") else torch.int32)
    num_ctx = torch.full((gens,), -1, dtype=torch.int64, device="cuda")
    return k_arena, v_arena, ctx_len, chosen.to(torch.int64), accepted, num_ctx, rows, max_ctx


def _clone(case):
    return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in case)


def _bits(t):
    return None if t is None else t.view(torch.int16)


def _assert_same(out, ref):
    """``out`` / ``ref``: a case after the call (arenas, lengths, new-length output compared)."""
    (k, v, ctx_len, num_ctx), (rk, rv, rctx, rnum) = (
        (case[0], case[1], case[2], case[5]) for case in (out, ref)
    )
    assert torch.equal(_bits(k), _bits(rk)), "K arena bytes differ"
    if v is not None:
        assert torch.equal(_bits(v), _bits(rv)), "V arena bytes differ"
    assert torch.equal(ctx_len, rctx), (ctx_len, rctx)
    assert torch.equal(num_ctx, rnum), (num_ctx, rnum)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    for _ in range(4):
        case = _case(cell.spec, g)
        mine, theirs = _clone(case), _clone(case)
        op(*mine)
        op.reference(*theirs)
        _assert_same(mine, theirs)


def test_masked_columns_are_the_ones_zeroed():
    """Rows at and past the accepted count, at the columns the store used; nothing else."""
    g = torch.Generator(device="cuda").manual_seed(7)
    spec = dict(gens=1, rows=8, layers=5, kv_heads=1, head_dim=128, slots=2, accepted=(3,))
    _, _, ctx_len, slots, accepted, num_ctx, rows, max_ctx = _case(spec, g)
    k_arena = _arena(2, 5, max_ctx + rows, 1, 128, g, specials=False)
    k_arena = (k_arena.float().abs() + 1.0).to(torch.bfloat16)  # no zeros, no NaN
    before = k_arena.clone()
    start = int(ctx_len[slots[0]])
    op(k_arena, None, ctx_len, slots, accepted, num_ctx, rows, max_ctx)
    changed = (k_arena != before).nonzero()
    cols = set(changed[:, 2].tolist())
    assert cols == set(range(start + 3, start + rows)), cols
    assert set(changed[:, 0].tolist()) == {int(slots[0])}
    assert int(ctx_len[slots[0]]) == start + 3
    assert num_ctx.tolist() == [start + 3]


@pytest.mark.parametrize("cell", op.CELLS[:1] + op.CELLS[3:4], ids=["serving", "at_end"])
def test_cuda_graph_replay(cell):
    """One capture, replayed with new stored rows, counts and lengths: each replay is the reference."""
    g = torch.Generator(device="cuda").manual_seed(_seed("graph" + cell.why))
    static = _case(cell.spec, g)
    k_arena, v_arena, ctx_len, slots, accepted, num_ctx, rows, max_ctx = static
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*static)
    for step in range(8):
        fresh = _case(cell.spec, g)
        accepted.copy_(
            torch.tensor([1 + (step + i) % rows for i in range(slots.numel())], device="cuda")
        )
        for static_t, new_t in ((k_arena, fresh[0]), (v_arena, fresh[1]), (ctx_len, fresh[2])):
            static_t.copy_(new_t)
        num_ctx.fill_(-1)
        expected = _clone(static)
        op.reference(*expected)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(static, expected)


def _served_launch():
    """The wrapper's launch at the served shape (1 request, 8 rows, 5 layers, 1 x 128, K and V)."""
    rows, layers, width, max_ctx = 8, 5, 128, 4096
    k = torch.zeros(2, layers, max_ctx + rows, 1, width, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    ctx_len = torch.tensor([100, 7], dtype=torch.int64, device="cuda")
    slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    accepted = torch.tensor([3], dtype=torch.int32, device="cuda")
    num_ctx = torch.zeros(1, dtype=torch.int64, device="cuda")
    new_len = torch.empty(1, dtype=torch.int64, device="cuda")
    compiled = commit_module._context_kv_commit_kernel[(1,)](
        k, v, ctx_len, slots, accepted, new_len, num_ctx, 1, ctx_len.numel(), max_ctx,
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        ROWS=rows, LAYERS=layers, WIDTH=width, BLOCK=width,
        PAIRS=commit_module.tile_pairs(rows, layers, width), HAS_V=True, num_warps=4,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert ctx_len.tolist() == [103, 7] and num_ctx.tolist() == [103]
    return compiled


def test_lengths_are_written_after_a_cta_barrier():
    """Every arena store (the zeroing) precedes the first barrier; 64-bit stores (lengths) follow it."""
    ptx = _served_launch().asm["ptx"].splitlines()
    barriers = [i for i, line in enumerate(ptx) if re.search(r"\b(bar|barrier)\.sync\b", line)]
    stores = [(i, line) for i, line in enumerate(ptx) if re.search(r"\bst\.global", line)]
    arena = [i for i, line in stores if not re.search(r"st\.global\.b64\b", line)]
    lengths = [i for i, line in stores if re.search(r"st\.global\.b64\b", line)]
    print(
        f"context_kv_commit PTX: {len(ptx)} lines, {len(barriers)} barrier(s) at {barriers}, "
        f"{len(arena)} arena stores, {len(lengths)} 64-bit stores at {lengths}"
    )
    assert barriers, "no CTA barrier between the length reads and the length writes"
    assert arena and max(arena) < barriers[0], "an arena store follows the barrier"
    assert any(i > barriers[0] for i in lengths), "no length is written after the barrier"


@pytest.mark.parametrize("gens", [1, 8, 64])
def test_many_launches_equal_the_reference(gens):
    """New lengths and counts every launch: a length read after its write in one launch would show."""
    rows = 8
    g = torch.Generator(device="cuda").manual_seed(gens)
    num_slots = gens + 1
    shape = (num_slots, 5, 4096 + rows, 1, 128)
    k0 = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    v0 = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    perm = torch.randperm(num_slots, generator=torch.Generator().manual_seed(gens))[:gens]
    slots = perm.to(torch.int64).cuda()
    launches = 2000 if gens <= 8 else 400
    bad = 0
    for _ in range(launches):
        ctx0 = torch.randint(0, 4000, (num_slots,), generator=g, device="cuda", dtype=torch.int64)
        accepted = torch.randint(
            1, rows + 1, (gens,), generator=g, device="cuda", dtype=torch.int32
        )
        mine = (k0.clone(), v0.clone(), ctx0.clone(), slots, accepted,
                torch.zeros(gens, dtype=torch.int64, device="cuda"), rows, 4096)  # fmt: skip
        theirs = _clone(mine)
        op(*mine)
        op.reference(*theirs)
        same = (
            torch.equal(_bits(mine[0]), _bits(theirs[0]))
            and torch.equal(_bits(mine[1]), _bits(theirs[1]))
            and torch.equal(mine[2], theirs[2])
            and torch.equal(mine[5], theirs[5])
        )
        bad += 0 if same else 1
    print(f"context_kv_commit gens={gens}: {launches} launches, {bad} differing from the reference")
    assert bad == 0
