# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa catalog entry.

The paged state is real: one `[pages, 2, kv, tpb, D]` HND pool sized from the
pages its requests map, and the engine's slot table -- a row per request slot,
more slots than requests, assigned by a seeded permutation, `-1` past each
sequence. Selection rows are built the way QSA's top-k + expansion lay them
out: complete groups expanded to tokens, then the trailing partial group, then
`-1`, all packed at the front.

The call has two kernels behind it (and a merge). The `launches` fixture
records which ones a call indexed with a grid, so every cell asserts its
branch.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    The kernel rounds its online-softmax probabilities to the cache dtype
    before P.V, so the error tracks the magnitude of the values averaged,
    not of the output: against an fp32 reference it measured 1.2e-3 on rows
    averaging a thousand tokens (outputs 0.14 to 0.4) and 1.2e-2 on rows
    averaging a handful (outputs to 5). Scaled by each head row's own
    magnitude that is one band. Against this reference (fp32, rounded once)
    every cell measured at most 2.0 element ulp -- a single final-rounding
    flip, since bf16's step is 2 of these units at the bottom of a binade --
    and at most 0.62 ulp RMS: 2x and 3.2x margin. Six wrong references (no
    scale, interleaved heads, padding attended, identity pages, K/V
    swapped, page/slot transposed) are each more than 0.1 off and rejected.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


DEV = "cuda"
ENGINE_MAX_SEQ_LEN = 8192

_KERNELS = {
    "_qsa_paged_sparse_gqa_kernel": "prefill",
    "_qsa_paged_sparse_gqa_splitk_kernel": "splitk",
    "_qsa_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing, then delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order, with grids."""
    log: list = []
    for name, tag in _KERNELS.items():
        inner = getattr(_qsa_kernels, name)
        monkeypatch.setattr(_qsa_kernels, name, _LaunchRecorder(inner, tag, log))
    return log


# --------------------------------------------------------------------------
# building real state from a cell spec
# --------------------------------------------------------------------------


def _segments(spec):
    """`(first position, rows)` per request in row order, and a generation mask."""
    if "mixed" not in spec:
        segments = list(spec["segments"])
        return segments, torch.zeros(sum(n for _, n in segments), dtype=torch.bool, device=DEV)
    requests, context_lens = spec["mixed"]
    # Chunk 0 starts at position 0, so one launch spans rows that see one token
    # and rows at the full budget; later chunks are chunked-prefill continuations.
    segments = [
        (0 if i == 0 else (907 * i) % (ENGINE_MAX_SEQ_LEN - n + 1), n)
        for i, n in enumerate(context_lens)
    ]
    # MTP: two rows per generation request, positions across the budget both ways.
    segments += [(100 + (499 * g) % 8000, 2) for g in range(requests - len(context_lens))]
    generation = torch.cat(
        [torch.full((n,), i >= len(context_lens), device=DEV) for i, (_, n) in enumerate(segments)]
    )
    return segments, generation


def _selected_tokens(positions, seed, compress_ratio, token_topk):
    """QSA's expansion layout, built in one pass.

    A random `min(available, block_topk)`-subset of the complete visible
    groups, each expanded to its `compress_ratio` tokens, then the incomplete
    trailing group, then `-1` -- packed at the front, as the producer emits.
    """
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)

    scores = torch.rand(rows, width, generator=gen, device=DEV)
    group_ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(group_ids < available[:, None], scores, torch.inf)
    chosen = scores.argsort(dim=1)[:, :block_topk]

    column = torch.arange(final_topk, device=DEV)[None, :]
    group_of_column = (column // compress_ratio).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(chosen, 1, group_of_column) * compress_ratio + column % compress_ratio
    in_body = column < (taken * compress_ratio)[:, None]
    tail_column = column - (taken * compress_ratio)[:, None]
    tail = (available * compress_ratio)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (
        spec["tokens_per_block"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["q_heads"],
    )
    segments, generation = _segments(spec)
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots

    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    # `min_seq_len` maps pages past the last query row's position, so a probe
    # can plant a token that is resident but not causally visible.
    floor = spec.get("min_seq_len", 0)
    mapped = [math.ceil(max(s + n, floor) / tpb) for s, n in segments]
    assert max(mapped) <= spec["pages"]
    pool_pages = sum(mapped) + 7
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    pool = torch.randn(pool_pages, 2, kv, tpb, d, generator=gen, device=DEV).to(spec["cache_dtype"])
    k_cache, v_cache = pool[:, 0], pool[:, 1]
    if spec["cache_layout"] == "contiguous":
        k_cache, v_cache = k_cache.contiguous(), v_cache.contiguous()

    q_dtype = spec["q_dtype"]
    if spec["q_layout"] == "qkv_slice":
        qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(q_dtype)
        q = qkv[:, :q_heads]
    else:
        q = torch.randn(rows, q_heads, d, generator=gen, device=DEV).to(q_dtype)

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    if "width" in spec:
        selected = selected[:, : spec["width"]].contiguous()
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    if spec.get("holes"):
        selected[1] = -1  # an all-padding row
        request_indices[2] = slots + 67  # a request the table does not have
        block_table[:, 0] = -1  # every request's first page unmapped

    args = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )
    if spec["prefill_pair"]:
        args.update(query_positions=positions, compress_ratio=spec["compress_ratio"])
    return args, generation


def _bound(positions, spec):
    """The prefill kernel's per-row scan bound, from the positions alone."""
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args, _ = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    if spec["prefill_pair"] and not spec.get("holes"):
        # The precondition the op cannot check: front-packed, and exactly as
        # many valid entries as the prefill kernel will scan.
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))

    out = op(**args)
    assert [name for name, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))


# --------------------------------------------------------------------------
# the prefill kernel's bound
# --------------------------------------------------------------------------

#: The first cell is this checkpoint's own decode geometry; the probes below
#: start from it.
CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(**overrides):
    return {**CKPT, **overrides}


def _first_unscanned_column(positions, tile=64):
    """First column of each row the prefill kernel never reads (bound rounded
    up to its 64-wide tile); none when that lands at or past the row's width."""
    final_topk = CKPT["token_topk"] + CKPT["compress_ratio"] - 1
    bound = _bound(positions, CKPT).clamp(max=final_topk)
    return ((bound + tile - 1) // tile) * tile, final_topk


def _plant_past_the_bound(args):
    """Put each row's own newest token in its first unscanned column."""
    positions = args["query_positions"]
    first, final_topk = _first_unscanned_column(positions)
    gap_rows = (first < final_topk).nonzero().flatten()
    assert gap_rows.numel() > 0, "no row has an unscanned column to plant into"
    planted = args["selected_tokens"].clone()
    planted[gap_rows, first[gap_rows]] = positions[gap_rows].to(torch.int32)
    return planted, gap_rows


def _sixty_four_token_rows(rows, seed):
    """Rows seeing 64 tokens: the bound is exactly one tile and column 64 the
    first unread one. It is planted with token 200 -- resident (each request
    maps 256 tokens) and not otherwise selected."""
    spec = _with(segments=((63, 1),) * rows, slots=rows + 8, min_seq_len=256, prefill_pair=True)
    args, _ = _build(spec, seed)
    packed = args["selected_tokens"]
    planted = packed.clone()
    planted[:, 64] = 200
    return {**args, "selected_tokens": planted}, packed


def test_prefill_kernel_stops_at_the_causal_prefix(launches) -> None:
    """An index past the bound is silently not attended to.

    The op matches the reference computed WITHOUT the planted index, and the
    planted index moves the correct answer by more than 0.1, so the first
    comparison is not vacuous.
    """
    args, packed = _sixty_four_token_rows(300, seed=7000)
    out = op(**args)
    assert [name for name, _ in launches] == ["prefill"]
    without_it = op.reference(**{**args, "selected_tokens": packed})
    _compare(out, without_it)
    with_it = op.reference(**args)
    assert (with_it.float() - without_it.float()).abs().max().item() > 0.1


def test_split_k_reads_the_whole_row(launches) -> None:
    """Below the program threshold the same operands attend to every column."""
    args, packed = _sixty_four_token_rows(100, seed=7100)
    out = op(**args)
    assert [name for name, _ in launches][0] == "splitk"
    _compare(out, op.reference(**args))
    without_it = op.reference(**{**args, "selected_tokens": packed})
    assert (out.float() - without_it.float()).abs().max().item() > 0.1


def test_prefill_bound_at_8191_rows() -> None:
    """The bound is the same at 8191 rows, and reaches much further: a row
    seeing 64 tokens is scanned for 64 of 2051 columns."""
    args, _ = _build(_with(segments=((0, 8191),), slots=8, prefill_pair=True), seed=11000)
    planted, _ = _plant_past_the_bound(args)
    out = op(**{**args, "selected_tokens": planted})
    without_it = op.reference(**args)
    _compare(out, without_it)
    with_it = op.reference(**{**args, "selected_tokens": planted})
    assert (with_it.float() - without_it.float()).abs().max().item() > 0.1


def test_prefill_bound_reaches_the_generation_rows() -> None:
    """A mixed launch's generation rows ride the prefill kernel, bound and all."""
    args, generation = _build(
        _with(mixed=(32, (900, 200, 150, 28)), slots=32, prefill_pair=True), seed=22000
    )
    planted, gap_rows = _plant_past_the_bound(args)
    assert bool(generation[gap_rows].any()), "no generation row has an unscanned column"
    out = op(**{**args, "selected_tokens": planted})
    without_it = op.reference(**args)
    _compare(out, without_it)
    with_it = op.reference(**{**args, "selected_tokens": planted})
    generation_gaps = gap_rows[generation[gap_rows]]
    moved = (with_it[generation_gaps].float() - without_it[generation_gaps].float()).abs().max()
    assert moved.item() > 0.1, f"the planted indices must matter to the generation rows: {moved}"


# --------------------------------------------------------------------------
# semantics a cell cannot state
# --------------------------------------------------------------------------


def test_repeated_index_is_attended_twice() -> None:
    """Selected indices are not deduplicated; each column is its own key."""
    args, _ = _build(_with(segments=((900, 1),), prefill_pair=False), seed=8700)
    selected = torch.full((1, 64), -1, dtype=torch.int32, device=DEV)
    selected[0, :3] = torch.tensor([5, 5, 40], dtype=torch.int32, device=DEV)
    twice = op(**{**args, "selected_tokens": selected})
    _compare(twice, op.reference(**{**args, "selected_tokens": selected}))
    once = selected.clone()
    once[0, 1] = -1
    single = op.reference(**{**args, "selected_tokens": once})
    assert (twice.float() - single.float()).abs().max().item() > 1e-2


def test_calls_are_bitwise_deterministic() -> None:
    """Including a 32-split launch whose merge is a second kernel."""
    args, _ = _build(_with(segments=tuple((4000 + 37 * i, 2) for i in range(4))), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_zero_rows_return_an_empty_output() -> None:
    args, _ = _build(_with(segments=((900, 1),) * 4), seed=8400)
    for name in ("q", "selected_tokens", "request_indices"):
        args[name] = args[name][:0]
    out = op(**args)
    assert out.shape == (0, CKPT["q_heads"], CKPT["head_dim"]) and out.dtype == torch.bfloat16
    _compare(out, op.reference(**args))


def test_gate_discriminates_wrong_variants() -> None:
    """Six references, each wrong one way, all rejected by `_compare`.

    Built by feeding the entry's own reference a mutated input, so each differs
    from the correct one in exactly the way it names.
    """
    args, _ = _build(_with(segments=((2032, 272),), prefill_pair=True), seed=9000)
    out = op(**args)
    _compare(out, op.reference(**args))
    tpb, groups = CKPT["tokens_per_block"], CKPT["q_heads"] // CKPT["kv_heads"]
    selected, table = args["selected_tokens"], args["block_table"]

    # Head h reads KV head h % Hkv instead of h // groups.
    interleave = torch.arange(CKPT["q_heads"], device=DEV)
    interleave = (interleave % groups) * CKPT["kv_heads"] + interleave // groups
    identity = torch.arange(table.shape[1], dtype=torch.int32, device=DEV).expand_as(table)
    tokens = selected.clamp_min(0)
    transposed = (tokens % tpb) * tpb + (tokens // tpb).clamp(max=tpb - 1)
    variants = {
        "no_softmax_scale": dict(softmax_scale=1.0),
        "interleaved_heads": dict(q=args["q"][:, interleave]),
        "padding_attended": dict(selected_tokens=tokens),
        "identity_pages": dict(block_table=identity.contiguous()),
        "k_and_v_swapped": dict(k_cache=args["v_cache"], v_cache=args["k_cache"]),
        "page_and_slot_transposed": dict(
            selected_tokens=torch.where(selected >= 0, transposed, selected)
        ),
    }
    for name, override in variants.items():
        wrong = op.reference(**{**args, **override})
        if name == "interleaved_heads":
            wrong = wrong[:, torch.argsort(interleave)]
        with pytest.raises(AssertionError):
            _compare(out, wrong)
        assert (out.float() - wrong.float()).abs().max().item() > 0.1, name


# --------------------------------------------------------------------------
# what the op refuses
# --------------------------------------------------------------------------

_REJECTIONS = {
    "float32_query": (lambda a: {**a, "q": a["q"].float()}, "QSA queries must use BF16 or FP16"),
    "flat_query": (
        lambda a: {**a, "q": a["q"].reshape(a["q"].shape[0], -1)},
        "QSA queries must have shape [rows, query_heads, head_dim]",
    ),
    "int64_selection": (
        lambda a: {**a, "selected_tokens": a["selected_tokens"].long()},
        "QSA selected tokens must be a two-dimensional int32 tensor",
    ),
    "int64_block_table": (
        lambda a: {**a, "block_table": a["block_table"].long()},
        "QSA sparse GQA requires a two-dimensional int32 block table",
    ),
    "wrong_tokens_per_block": (
        lambda a: {**a, "tokens_per_block": a["tokens_per_block"] // 2},
        "QSA cache token dimension must match positive tokens_per_block",
    ),
    "negative_scale": (
        lambda a: {**a, "softmax_scale": -1.0},
        "QSA sparse GQA softmax scale must be finite and positive",
    ),
    "strided_request_indices": (
        lambda a: {**a, "request_indices": torch.zeros(8, dtype=torch.int32, device=DEV)[::2]},
        "QSA request indices must be contiguous",
    ),
    "float32_cache": (
        lambda a: {**a, "k_cache": a["k_cache"].float(), "v_cache": a["v_cache"].float()},
        "QSA K/V cache must use BF16, FP16, or FP8 E4M3",
    ),
    "positions_without_ratio": (
        lambda a: {**a, "query_positions": torch.zeros(4, dtype=torch.int64, device=DEV)},
        "QSA causal-prefix bounds require positions and compression ratio",
    ),
    "head_dim_192": (
        lambda a: {
            **a,
            "q": a["q"][..., :192].contiguous(),
            "k_cache": a["k_cache"][..., :192],
            "v_cache": a["v_cache"][..., :192],
        },
        "QSA sparse GQA requires a power-of-two head dimension",
    ),
    "host_query": (
        lambda a: {**a, "q": a["q"].cpu()},
        "QSA paged sparse GQA requires CUDA tensors",
    ),
}


@pytest.mark.parametrize("case", list(_REJECTIONS), ids=list(_REJECTIONS))
def test_rejects(case) -> None:
    """The op's own loud refusals, verbatim -- the reason `is_valid` is empty.
    192 is a legal head size elsewhere in this engine and not here."""
    args, _ = _build(_with(segments=((600, 1),) * 4), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    mutate, message = _REJECTIONS[case]
    with pytest.raises(ValueError, match=re.escape(message)):
        op(**mutate(args))


def test_selection_width_is_checked_on_the_prefill_kernel_only() -> None:
    """A width the compress ratio cannot produce: refused by the prefill
    kernel, accepted (the pair is unused) by split-K."""
    args, _ = _build(_with(segments=((200, 272),), prefill_pair=True), seed=9300)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    with pytest.raises(ValueError, match="incompatible with its compression ratio"):
        op(**args)

    args, _ = _build(_with(segments=((200, 100),), prefill_pair=True), seed=9400)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    _compare(op(**args), op.reference(**args))
