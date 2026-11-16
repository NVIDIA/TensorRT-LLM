# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_fp8_paged_mqa_logits entry."""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm import deep_gemm  # noqa: E402
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E402, E501
    ATOM_SPLIT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E402, E501
    cute_dsl_fp8_paged_mqa_logits as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

E4M3_MAX = 448.0
#: The schedule block the DSA indexer builds its decode schedule with.
SCHEDULE_BLOCK_KV = 64

# Gate, in units of the row's largest |logit|. See `_compare`.
ELEMENT_TOL = 2.0**-12
RMS_TOL = 2.0**-14


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2**-12 of the row's largest |logit| per element, 2**-14 relative RMS,
    over the causal part of each row only.

    Same arithmetic as `fp8_mqa_logits`: exact fp8 products, then fp32 sums
    over 128 dims and 32 heads, so kernel and reference differ by a few
    fp32 ulps of the row scale; a wrong formula moves logits by O(1).
    Columns past the causal end hold whatever the shared buffer held and are
    not compared: NaN in `ref` means unspecified.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    live = ~torch.isnan(ref)
    assert bool(torch.isfinite(out[live]).all()), "a causal entry is not finite"
    o = torch.where(live, out, torch.zeros_like(out)).double()
    r = torch.where(live, ref, torch.zeros_like(ref)).double()
    row_scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    err = (o - r).abs() / row_scale
    assert float(err.max()) <= ELEMENT_TOL, (
        f"max error {float(err.max()):.3e} of the row scale exceeds {ELEMENT_TOL:.3e}"
    )
    rms = float(((o - r) ** 2).sum().sqrt() / (r**2).sum().sqrt().clamp_min(1e-30))
    assert rms <= RMS_TOL, f"relative RMS {rms:.3e} exceeds {RMS_TOL:.3e}"


def _schedule(context_lens: torch.Tensor) -> torch.Tensor:
    """The decode schedule, built the way the DSA indexer builds it."""
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.view(-1, 1), SCHEDULE_BLOCK_KV, deep_gemm.get_num_sms()
    )


def _build(spec, seed, num_pages=None, width=None):
    """`(q, kv_fused, weights, context_lens, block_table, schedule, max_context_len)`.

    Every key is quantized per token to e4m3 with an fp32 scale and written to
    a random page, data first and scales after, as the indexer cache stores
    them. Unused block-table entries point at page 0. `num_pages` / `width`
    size the cache and the table past what the lengths need (equal shapes for
    a second set of lengths copied into captured buffers).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    heads, dim, tpb = spec["heads"], spec["head_dim"], spec["tokens_per_block"]
    next_n = spec["next_n"]
    lens = spec["context_lens"]
    batch = len(lens)
    pages = [math.ceil(n / tpb) for n in lens]
    num_pages = max(sum(pages) + 1, num_pages or 0)
    order = torch.randperm(num_pages, generator=cpu).to(torch.int32)
    table = torch.zeros(batch, max(max(pages), width or 0), dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages):
        table[b, :n] = order[used : used + n]
        used += n

    flat = torch.zeros(num_pages, tpb * (dim + 4), dtype=torch.uint8, device="cuda")
    keys = torch.randn(num_pages, tpb, dim, generator=g, device="cuda")
    scale = (keys.abs().amax(-1) / E4M3_MAX).clamp_min(1e-12)
    codes = (keys / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    flat[:, : tpb * dim] = codes.view(torch.uint8).reshape(num_pages, -1)
    flat[:, tpb * dim :] = scale.contiguous().view(torch.uint8).reshape(num_pages, -1)
    kv_fused = flat.view(num_pages, tpb, 1, dim + 4)

    q = torch.randn(batch, next_n, heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(batch * next_n, heads, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    context_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return (
        q,
        kv_fused,
        w.contiguous(),
        context_lens,
        table.cuda(),
        _schedule(context_lens),
        spec["max_context_len"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    cache_before = args[1].clone()
    with validating(op):
        out = op(*args)
    assert torch.equal(args[1], cache_before), "scoring modified the cache"
    batch, next_n = args[0].shape[:2]
    assert tuple(out.shape) == (batch * next_n, args[6])
    ref = op.reference(*args)
    # Not degenerate: every row with a causal key has a nonzero logit, so the row-scaled
    # gate in `_compare` measures against a real scale.
    live = ~torch.isnan(ref)
    scale = torch.where(live, ref, torch.zeros_like(ref)).abs().amax(dim=1)
    assert bool((scale[live.any(dim=1)] > 0).all()), "a row's causal logits are all zero"
    _compare(out, ref)


@pytest.mark.parametrize("next_n", [2, 4, 6], ids=["mtp1", "mtp3", "mtp5"])
def test_graph_replay_is_bitwise_eager(next_n) -> None:
    """The target captures this op in its decode graph at each MTP width's full batch: a
    replay must write the eager call's causal logits bit for bit, and a second replay on new
    queries, weights, keys and block table copied into the same buffers must write that
    call's. The result is the arena buffer, so each eager result is copied out at once."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=next_n,
        context_lens=[100 + 251 * i for i in range(32)],
        max_context_len=8192,
    )
    args = list(_build(spec, seed=90 + next_n))
    fresh = _build(spec, seed=95 + next_n)
    eager = [op(*a).clone() for a in (args, fresh)]
    live = ~torch.isnan(op.reference(*args))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    first = [t.clone() for t in args[:3]] + [args[4].clone()]
    for operands, want in zip((first, [fresh[0], fresh[1], fresh[2], fresh[4]]), eager):
        for i, src in zip((0, 1, 2, 4), operands):
            args[i].copy_(src)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[live].view(torch.int32), want[live].view(torch.int32))
    assert not torch.equal(eager[0][live], eager[1][live])


def _split_call(args, atom):
    """The target's MTP5 call: `next_n // atom` pseudo-sequences per sequence, `atom` rows
    each, on the metadata `split_metadata` builds from the step's lengths and table."""
    q, kv, w, lens, table, _, max_len = args
    batch, next_n = q.shape[:2]
    factor = next_n // atom
    meta = op.split_metadata(lens, table, factor)
    return op(q.reshape(batch * factor, atom, *q.shape[2:]), kv, w, *meta, max_len)


#: MTP5 batches the target splits: one sequence (the expansion's reshape is a stride-0 view
#: there), 3 with a 6-key context, 11 with ends around top-k and long rows, the full 32.
_SPLIT_LENS = {
    "b1": ([7], 256),
    "b3": ([6, 33, 1000], 2048),
    "b11": ([6, 7, 33, 2047, 2048, 2053, 4097, 9001, 20000, 40000, 65536], 65536),
    "b32": ([106 + 253 * i for i in range(32)], 8192),
}


@pytest.mark.parametrize("name", list(_SPLIT_LENS))
def test_atom_split_writes_the_native_causal_logits(name) -> None:
    """MTP5 at next_n 6 as two pseudo-sequences of 3 per sequence: every causal logit of the
    native next_n 6 call, bit for bit, and within the fp64 reference's band. The split's
    metadata is contiguous (the kernel reads it densely) and doubles every sequence."""
    assert ATOM_SPLIT == {6: 3}
    lens, max_len = _SPLIT_LENS[name]
    spec = dict(
        heads=32, head_dim=128, tokens_per_block=32, next_n=6, context_lens=lens,
        max_context_len=max_len,
    )  # fmt: skip
    args = _build(spec, seed=zlib.crc32(name.encode()) % 2**31)
    ref = op.reference(*args)
    live = ~torch.isnan(ref)
    # Not degenerate: every row with a causal key has a nonzero logit, and from 11 sequences
    # on some rows are longer than the decode top-k (2048), so a selection is real.
    scale = torch.where(live, ref, torch.zeros_like(ref)).abs().amax(dim=1)
    assert bool((scale[live.any(dim=1)] > 0).all()), "a row's causal logits are all zero"
    if len(lens) >= 11:
        assert int(live.sum(dim=1).max()) > 2048
    native = op(*args).clone()
    lens_x, table_x, schedule_x = op.split_metadata(args[3], args[4], 2)
    assert lens_x.is_contiguous() and table_x.is_contiguous()
    assert lens_x.tolist() == [n for n in lens for _ in range(2)]
    assert torch.equal(table_x, args[4].repeat_interleave(2, dim=0))
    assert torch.equal(schedule_x, _schedule(lens_x))
    with validating(op):
        split = _split_call(args, ATOM_SPLIT[6]).clone()
    assert torch.equal(split[live].view(torch.int32), native[live].view(torch.int32))
    _compare(split, ref)


def test_atom_split_graph_replay_is_bitwise_eager() -> None:
    """The target captures the split, metadata included, in its MTP5 decode graph: a replay
    must write the eager split's (and the native call's) causal logits bit for bit, and a
    second replay on new queries, weights, keys, block table and context lengths copied into
    the same buffers must write that call's: the lengths, table and schedule are rebuilt in
    the graph, not frozen at capture."""
    lens = [106 + 253 * i for i in range(32)]
    spec = dict(
        heads=32, head_dim=128, tokens_per_block=32, next_n=6, context_lens=lens,
        max_context_len=8192,
    )  # fmt: skip
    # The second step's lengths differ per sequence (some shorter, some longer).
    spec2 = dict(spec, context_lens=[8 + (977 * i) % 8100 for i in range(32)])
    pages = max(sum(math.ceil(n / 32) for n in s["context_lens"]) for s in (spec, spec2)) + 1
    args = list(_build(spec, seed=96, num_pages=pages, width=256))
    fresh = _build(spec2, seed=97, num_pages=pages, width=256)
    atom = ATOM_SPLIT[6]
    lives = [~torch.isnan(op.reference(*a)) for a in (args, fresh)]
    natives = [op(*a).clone() for a in (args, fresh)]
    eager = [_split_call(a, atom).clone() for a in (args, fresh)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _split_call(args, atom)
    first = [t.clone() for t in args[:5]]
    for operands, want, native, live in zip((first, fresh), eager, natives, lives):
        for i in range(5):
            args[i].copy_(operands[i])
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[live].view(torch.int32), want[live].view(torch.int32))
        assert torch.equal(out[live].view(torch.int32), native[live].view(torch.int32))
    assert not torch.equal(lives[0], lives[1])


def test_the_result_is_a_shared_buffer() -> None:
    """Two calls of one shape return the same storage, and the second call
    overwrites the first result. A caller that keeps a result across calls
    reads the later scores without an error."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[500, 300],
        max_context_len=1024,
    )
    first_args = _build(spec, seed=21)
    second_args = _build(spec, seed=22)
    first = op(*first_args)
    _compare(first, op.reference(*first_args))
    second = op(*second_args)
    assert first.data_ptr() == second.data_ptr()
    _compare(first, op.reference(*second_args))


def test_mtp_rows_have_their_own_causal_end() -> None:
    """At MTP 2 the first token of a sequence ends one key before the second.

    The decode top-k reads each row only up to its own causal end, so that is
    the part the entry certifies; the last key is defined for the second row
    only.
    """
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=2,
        context_lens=[200],
        max_context_len=256,
    )
    args = _build(spec, seed=23)
    out = op(*args)
    ref = op.reference(*args)
    assert torch.isnan(ref[0, 199]) and not torch.isnan(ref[1, 199])
    _compare(out, ref)


def _ok():
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[100, 64],
        max_context_len=256,
    )
    return _build(spec, seed=25)


def test_guard_refuses_a_schedule_for_another_sm_count() -> None:
    q, kv, w, lens, table, schedule, max_len = _ok()
    for rows in (schedule.shape[0] - 1, schedule.shape[0] + 1):
        bad = torch.zeros(rows, 2, dtype=torch.int32, device="cuda")
        with pytest.raises(AssertionError, match="SM count \\+ 1 rows"):
            with validating(op):
                op(q, kv, w, lens, table, bad, max_len)


def test_guard_refuses_a_context_past_max_context_len() -> None:
    q, kv, w, lens, table, schedule, _ = _ok()
    with pytest.raises(AssertionError, match="exceeds max_context_len"):
        with validating(op):
            op(q, kv, w, lens, table, schedule, 64)
