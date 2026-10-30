# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qsa_paged_sparse_gqa variant (`__r26`).

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

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_paged_sparse_gqa__r26 import (  # noqa: E501
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
    swapped, page/slot transposed) were each measured more than 0.1 off.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


DEV = "cuda"

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

    q_dtype = spec["q_dtype"]
    row_heads = q_heads + 2 * kv if spec["q_layout"] == "qkv_slice" else 55
    qkv = torch.randn(rows, row_heads, d, generator=gen, device=DEV).to(q_dtype)
    q = qkv[:, :q_heads]

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    gate = None
    if spec.get("gate") == "fused_row":
        # The gate columns of the target's fused projection row (55 heads of
        # D): heads 28..51, after q, k and v.
        fused = torch.randn(rows, 55 * d, generator=gen, device=DEV).to(q_dtype)
        gate = fused[:, 28 * d : (28 + q_heads) * d]

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
    if gate is not None:
        args.update(output_gate=gate)
    return args


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
    args = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    if spec["prefill_pair"]:
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


#: The first cell is this checkpoint's own decode geometry; the tests below
#: start from it.
CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(**overrides):
    return {**CKPT, **overrides}


def test_calls_are_bitwise_deterministic() -> None:
    """Including a 32-split launch whose merge is a second kernel."""
    spec = _with(segments=tuple((4000 + 37 * i, 2) for i in range(4)), gate=None)
    args = _build(spec, seed=8800)
    assert torch.equal(op(**args), op(**args))


# --------------------------------------------------------------------------
# what the op refuses
# --------------------------------------------------------------------------


def test_rejects_positions_without_a_compress_ratio() -> None:
    """The op's own loud refusal, verbatim -- the reason `is_valid` is empty: half of
    the prefill pair."""
    args = _build(_with(segments=((600, 1),) * 4, gate=None), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    args["query_positions"] = torch.zeros(4, dtype=torch.int64, device=DEV)
    message = "QSA causal-prefix bounds require positions and compression ratio"
    with pytest.raises(ValueError, match=re.escape(message)):
        op(**args)
