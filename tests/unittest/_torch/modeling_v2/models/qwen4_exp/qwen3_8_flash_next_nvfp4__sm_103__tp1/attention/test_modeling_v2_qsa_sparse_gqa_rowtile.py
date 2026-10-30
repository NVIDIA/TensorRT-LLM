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
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
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
    assert spec["cache_layout"] == "pool_planes" and spec["q_layout"] == "qkv_slice"
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(spec["q_dtype"])
    q = qkv[:, :q_heads]

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"],
                                spec.get("shared_scores", False))  # fmt: skip
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])
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


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    # The per-row kernel's precondition, which this op shares.
    selected = args["selected_tokens"]
    assert bool(_is_prefix_packed(selected).all())
    assert torch.equal((selected >= 0).sum(1), _bound(args["query_positions"], spec))
    out = op(**args)
    assert [name for name, _ in launches] == list(_KERNELS)
    groups = math.ceil(rows / 10)
    assert launches[0][1] == (rows,) and launches[1][1] == (groups,)
    assert launches[2][1] == (min(rows, 256), spec["kv_heads"])
    # The row-tile pass is persistent: one CTA per SM, never more than (group, KV head) items.
    items = groups * spec["kv_heads"]
    assert launches[3][1] == (_module._rowtile_ctas(items, args["q"].device),)
    assert launches[3][1][0] == min(
        items, torch.cuda.get_device_properties(0).multi_processor_count
    )
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))


CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(segments, **overrides):
    return {**CKPT, "segments": tuple(segments), **overrides}


@pytest.mark.parametrize(
    "segments",
    [
        # Fewer items (2 groups x 2 KV heads) than SMs: a 4-CTA grid, mostly one item a CTA.
        [(5000, 17)],
        # A lone group (rows of two requests) between row-tile groups of each request.
        [(3000, 25), (7000, 25)],
    ],
    ids=["fewer items than SMs", "a lone group between row-tile groups"],
)
def test_small_launches_match_the_reference(segments) -> None:
    """The persistent pass's edges: CTAs left without a second item, and an item the pass
    skips (a lone group) between items it serves."""
    args = _build(_with(segments, slots=8), seed=7700 + len(segments))
    _compare(op(**args), op.reference(**args))


def test_calls_are_bitwise_deterministic() -> None:
    args = _build(_with([(0, 2000), (500, 300), (7000, 1)], slots=8), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_rejects_a_call_without_the_prefill_pair() -> None:
    """The op's own loud refusal: the row-tile pass is bounded by the prefill pair."""
    args = _build(_with([(600, 4)], slots=8), seed=9100)
    args.update(query_positions=None, compress_ratio=None)
    with pytest.raises(ValueError, match=re.escape("the row-tile kernel needs the prefill pair")):
        op(**args)
