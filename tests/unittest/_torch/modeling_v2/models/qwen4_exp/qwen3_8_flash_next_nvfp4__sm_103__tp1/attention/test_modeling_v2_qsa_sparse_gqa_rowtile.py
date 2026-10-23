# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_sparse_gqa_rowtile entry.

Same paged state as `qsa_paged_sparse_gqa`'s test: one `[pages, 2, kv, tpb, D]`
HND pool, the engine's slot table (more slots than requests, a seeded
permutation, `-1` past each sequence), and selection rows laid out the way
QSA's top-k + expansion emit them (complete groups expanded to tokens, the
trailing partial group, then `-1`, packed at the front). The band and the
reference are that entry's: this op replaces it on context launches.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention import (  # noqa: E501
    qsa_sparse_gqa_rowtile as _module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa as per_row,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_sparse_gqa_rowtile import (  # noqa: E501
    qsa_sparse_gqa_rowtile as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_sparse_gqa_rowtile requires a CUDA device"

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`qsa_paged_sparse_gqa`'s band: 4 ulp of each (row, head)'s magnitude,
    2 ulp relative RMS, in q's dtype, against the fp32 reference rounded once."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


class _LaunchRecorder:
    """Records which of the entry's kernels a call indexed with a grid; the bitmap
    pass (`_row_bits`, a plain call) is recorded with its one-CTA-a-row grid."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]

    def __call__(self, selected_tokens, *args, **kwargs):
        self._log.append((self._name, (selected_tokens.shape[0],)))
        return self._inner(selected_tokens, *args, **kwargs)


_KERNELS = (
    "_row_bits",
    "_rowtile_plan_kernel",
    "_rowtile_lone_kernel",
    "_rowtile_attention_cute",
)


@pytest.fixture
def launches(monkeypatch):
    log: list = []
    for name in _KERNELS:
        inner = getattr(_module, name)
        monkeypatch.setattr(_module, name, _LaunchRecorder(inner, name, log))
    return log


def _selected_tokens(positions, seed, compress_ratio, token_topk, shared_scores=False):
    """QSA's expansion layout: a random `min(available, block_topk)`-subset of
    the complete visible groups, expanded, then the trailing partial group,
    then `-1`, packed at the front. With `shared_scores` every row ranks the
    groups by one score vector, so adjacent rows pick nearly the same set (a
    sparse union, as on real selections)."""
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)
    out = torch.empty(rows, final_topk, dtype=torch.int32, device=DEV)
    column = torch.arange(final_topk, device=DEV)[None, :]
    for start in range(0, rows, 2048):
        stop = min(start + 2048, rows)
        avail, take, vis = available[start:stop], taken[start:stop], visible[start:stop]
        if shared_scores:
            scores = torch.rand(1, width, generator=gen, device=DEV).expand(stop - start, -1)
            gen.manual_seed(seed)
        else:
            scores = torch.rand(stop - start, width, generator=gen, device=DEV)
        group_ids = torch.arange(width, device=DEV)[None, :]
        scores = torch.where(group_ids < avail[:, None], scores, torch.inf)
        chosen = scores.argsort(dim=1)[:, :block_topk]
        group_of_column = (column // compress_ratio).clamp(max=block_topk - 1)
        body = torch.gather(chosen, 1, group_of_column.expand(stop - start, -1))
        body = body * compress_ratio + column % compress_ratio
        in_body = column < (take * compress_ratio)[:, None]
        tail_column = column - (take * compress_ratio)[:, None]
        tail = (avail * compress_ratio)[:, None] + tail_column
        in_tail = ~in_body & (tail_column >= 0) & (tail < vis[:, None])
        out[start:stop] = torch.where(in_body, body, torch.where(in_tail, tail, -1)).to(torch.int32)
    return out


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (
        spec["tokens_per_block"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["q_heads"],
    )
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
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

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"],
                                spec.get("shared_scores", False))  # fmt: skip
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])
    if spec.get("holes"):
        selected[1] = -1  # an all-padding row
        request_indices[2] = slots + 67  # a request the table does not have
        block_table[:, 0] = -1  # every request's first page unmapped
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
        query_positions=positions.to(spec["request_dtype"]),
        compress_ratio=spec["compress_ratio"],
    )


def _bound(positions, spec):
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


def _shared_groups(request_indices, rows_per_tile=10):
    """How many 10-row groups name a single request (served by the row-tile pass)."""
    rows = request_indices.numel()
    padded = torch.full((math.ceil(rows / rows_per_tile) * rows_per_tile,), -1, device=DEV)
    padded[:rows] = request_indices.long()
    grid = padded.view(-1, rows_per_tile)
    live = torch.arange(padded.numel(), device=DEV).view(-1, rows_per_tile) < rows
    low = torch.where(live, grid, 1 << 40).min(1).values
    high = torch.where(live, grid, -1).max(1).values
    return int((low == high).sum())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    if not spec.get("holes"):
        # The per-row kernel's precondition, which this op shares.
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        assert torch.equal((selected >= 0).sum(1), _bound(args["query_positions"], spec))
    out = op(**args)
    assert [name for name, _ in launches] == list(_KERNELS)
    groups = math.ceil(rows / 10)
    assert launches[0][1] == (rows,) and launches[1][1] == (groups,)
    assert launches[2][1] == (min(rows, 256), spec["kv_heads"])
    assert launches[3][1] == (groups, spec["kv_heads"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))


# --------------------------------------------------------------------------
# semantics a cell cannot state
# --------------------------------------------------------------------------

CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(segments, **overrides):
    return {**CKPT, "segments": tuple(segments), **overrides}


def _plant_past_the_bound(args):
    """Put each row's own newest token in its first unscanned column (bound
    rounded up to 64), where the row has one."""
    positions = args["query_positions"]
    final_topk = args["selected_tokens"].shape[1]
    bound = _bound(positions, CKPT).clamp(max=final_topk)
    first = (bound + 63) // 64 * 64
    gap_rows = (first < final_topk).nonzero().flatten()
    assert gap_rows.numel() > 0
    planted = args["selected_tokens"].clone()
    planted[gap_rows, first[gap_rows]] = positions[gap_rows].to(torch.int32)
    return planted, gap_rows


@pytest.mark.parametrize(
    "segments",
    [
        [(0, 8191)],
        [(0, 900), (907, 200), (1814, 150), (2721, 28)]
        + [(100 + (499 * g) % 8000, 1) for g in range(28)],
    ],
    ids=["shared-groups", "mixed-with-lone-rows"],
)
def test_prefill_bound_is_honoured(segments, launches) -> None:
    """An index past the per-row kernel's bound is not attended, on the
    row-tile pass and on the lone-row pass alike; the planted index moves the
    right answer by more than 0.1, so the comparison is not vacuous."""
    args = _build(_with(segments, slots=64), seed=11000)
    planted, gap_rows = _plant_past_the_bound(args)
    out = op(**{**args, "selected_tokens": planted})
    without_it = op.reference(**args)
    _compare(out, without_it)
    with_it = op.reference(**{**args, "selected_tokens": planted})
    assert (with_it[gap_rows].float() - without_it[gap_rows].float()).abs().max().item() > 0.1
    assert _shared_groups(args["request_indices"]) > 0


def test_repeated_index_on_a_shared_group_is_attended_once() -> None:
    """The row-tile pass attends to the set a row names (the documented
    precondition): a repeated column matches the reference without the
    repeat. Rows of a group that spans requests (rows 10..15 here: 15 rows of
    one request, then one of another) keep per-column semantics."""
    args = _build(_with([(900, 15), (5000, 1)], slots=8), seed=8700)
    selected = args["selected_tokens"].clone()
    once = selected.clone()
    shared_row, lone_row = 3, 15
    for r in (shared_row, lone_row):
        assert int(selected[r, 0]) >= 0
        last = int((selected[r] >= 0).sum()) - 1
        selected[r, last] = selected[r, 0]  # the row's last token becomes a repeat of its first
        once[r, last] = -1
    out = op(**{**args, "selected_tokens": selected})
    _compare(out[:10], op.reference(**{**args, "selected_tokens": once})[:10])
    _compare(out[10:], op.reference(**{**args, "selected_tokens": selected})[10:])


def test_matches_the_per_row_entry_on_dumped_layout() -> None:
    """Same arguments, same answer as `qsa_paged_sparse_gqa` within twice the band."""
    args = _build(
        _with([(0, 1200), (3000, 900)] + [(8300 + g, 1) for g in range(9)], slots=16), seed=4242
    )
    out, theirs = op(**args), per_row(**args)
    assert_within_ulp(out, theirs, element_ulp=8.0, rms_ulp=4.0)


def test_calls_are_bitwise_deterministic() -> None:
    args = _build(_with([(0, 2000), (500, 300), (7000, 1)], slots=8), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_zero_rows_return_an_empty_output() -> None:
    args = _build(_with([(900, 4)], slots=8), seed=8400)
    for name in ("q", "selected_tokens", "request_indices", "query_positions"):
        args[name] = args[name][:0]
    out = op(**args)
    assert out.shape == (0, CKPT["q_heads"], CKPT["head_dim"]) and out.dtype == torch.bfloat16


_REJECTIONS = {
    "no_prefill_pair": (
        lambda a: {**a, "query_positions": None, "compress_ratio": None},
        "the row-tile kernel needs the prefill pair",
    ),
    "float32_query": (lambda a: {**a, "q": a["q"].float()}, "QSA queries must use BF16"),
    "int64_selection": (
        lambda a: {**a, "selected_tokens": a["selected_tokens"].long()},
        "QSA selected tokens must be a two-dimensional int32 tensor",
    ),
    "width_vs_ratio": (
        lambda a: {**a, "selected_tokens": a["selected_tokens"][:, :64].contiguous()},
        "incompatible with its compression ratio",
    ),
    "strided_request_indices": (
        lambda a: {**a, "request_indices": torch.zeros(8, dtype=torch.int32, device=DEV)[::2]},
        "QSA request indices must be contiguous",
    ),
    "host_query": (lambda a: {**a, "q": a["q"].cpu()}, "needs every tensor on one CUDA device"),
}


@pytest.mark.parametrize("case", list(_REJECTIONS), ids=list(_REJECTIONS))
def test_rejects(case) -> None:
    args = _build(_with([(600, 4)], slots=8), seed=9100)
    mutate, message = _REJECTIONS[case]
    with pytest.raises(ValueError, match=re.escape(message)):
        op(**mutate(args))


class _PlanRecorder:
    """Captures the plan (`group_plan`) the row-tile launch reads."""

    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self._log.append(args[5].clone())
            return self._inner[grid](*args, **kwargs)

        return launch


def _expected_union(args, rows_per_tile=10):
    """Per 10-row group: the number of distinct selected tokens within each
    row's bound whose page the group's request maps, when the group names
    one request (else -1), and the dense walk's tiles."""
    selected, positions = args["selected_tokens"], args["query_positions"]
    request_indices, table = args["request_indices"].long(), args["block_table"]
    tpb, rows = args["tokens_per_block"], selected.shape[0]
    bound = _bound(positions, CKPT).clamp(max=selected.shape[1])
    columns = torch.arange(selected.shape[1], device=DEV)[None, :]
    counts, dense = [], []
    for start in range(0, rows, rows_per_tile):
        stop = min(start + rows_per_tile, rows)
        group_requests = request_indices[start:stop].unique()
        if group_requests.numel() != 1:
            counts.append(-1)
            dense.append(0)
            continue
        tokens = selected[start:stop].long()
        tokens = tokens[(columns < bound[start:stop, None]) & (tokens >= 0)].unique()
        pages = table[int(group_requests)][(tokens // tpb).clamp(max=table.shape[1] - 1)]
        kept = tokens[(tokens // tpb < table.shape[1]) & (pages >= 0)]
        counts.append(kept.numel())
        dense.append((int(tokens.max()) + 64) // 64 if tokens.numel() else 0)
    return torch.tensor(counts), torch.tensor(dense)


@pytest.mark.parametrize(
    "segments, shared_scores",
    [([(4096, 4096)], True), ([(900, 40), (950, 40), (3000, 25)], False)],
    ids=["sparse-union", "two-requests-unmapped-page"],
)
def test_packed_walk_is_the_resident_union(segments, shared_scores, monkeypatch) -> None:
    """The row-tile pass walks exactly the union of its rows' resident
    selected tokens, 64 per tile: the plan's list length is that union's
    size for every one-request group, and on rows ranking one score vector
    the walk is far shorter than every tile up to the group's highest token."""
    args = _build(_with(segments, slots=16, shared_scores=shared_scores), seed=7300)
    if not shared_scores:
        args["block_table"][:, 3] = -1  # an unmapped page inside every request's span
    log: list = []
    name = "_rowtile_attention_cute"
    monkeypatch.setattr(_module, name, _PlanRecorder(getattr(_module, name), log))
    out = op(**args)
    _compare(out, op.reference(**args))
    plan = log[0].cpu()
    counts, dense = _expected_union(args)
    one_request = counts >= 0
    assert torch.equal(plan[one_request, 1], counts[one_request].int())
    taken = plan[:, 0] >= 0
    assert bool(taken.any())
    assert torch.equal(plan[taken, 0], (plan[taken, 1] + 63) // 64)
    if shared_scores:
        assert int(plan[taken, 0].sum()) * 2 < int(dense[taken].sum())
