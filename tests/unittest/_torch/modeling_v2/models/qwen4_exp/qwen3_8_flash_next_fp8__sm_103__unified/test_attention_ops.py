# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's attention entries other than `thop_attention`: the
main and index rope, and the QSA plane store, K/V store and sparse GQA.

- `fused_qk_norm_rope__rank_heads`: the catalog entry's test keyed to the head counts one rank
  of this target holds (`QK_NORM_ROPE_TARGET_CELLS`, driven with the entry's own cells), the
  `[11, 11, 10]` mRoPE split against the wrong wirings, and the plain path it reduces to.
- `qsa_paged_index_store`: real index-key and position planes of `pages` pages and the engine's
  `[slots, width]` table -- more slots than requests, each live row's pages drawn from a seeded
  permutation, `-1` past its allocation, rows past the batch holding a previous step's page ids
  -- pre-filled with a sentinel so a stray write shows; the stock bf16 layout (one layer's plane
  of the coalesced index pool, the others checked untouched). The stock manager itself is built
  through `test_routing_and_layout.py` and stored through its own views, the draft layer's
  plane included, and a captured store replays rewritten metadata.
- `qsa_paged_kv_store`: an HND pool whose K and V role views are the caches, either contiguous
  and compared whole, or the stock manager's layout at an engine's measured page count (every
  attention layer's K and V planes in one page, ~100 GB at ~257k pages), a reserved virtual
  range with memory mapped under the touched pages only (`_KvEnginePool`), every other mapped
  byte checked unchanged; a captured step over that pool replays new positions.
- `qsa_paged_sparse_gqa`: one `[pages, slot_pages, kv, tpb, D]` pool whose layer planes are the
  call's K and V, and selection rows built the way QSA's top-k + expansion lay them out. The
  `launches` fixture records which of the call's kernels ran with which grid and PDL, so each
  cell asserts its branch; a `capture` cell is replayed in a CUDA graph over refilled inputs.
"""

import importlib.util
import math
import zlib
from pathlib import Path

import cuda.bindings.driver as cuda_drv
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.fused_qk_norm_rope__rank_heads import (  # noqa: E501
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_index_store import (  # noqa: E501
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_kv_store import (  # noqa: E501
    qsa_paged_kv_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = ["../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "the attention entries require a CUDA device"

DEV = "cuda"


def _seed(text: str) -> int:
    """A seed that is the same in every process; `hash()` of a str is not."""
    return zlib.crc32(text.encode())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(as_int)


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


# ------------------------------------------------ fused_qk_norm_rope__rank_heads ---

# Spec keys that shape the test's inputs or gate rather than the op call.
_ROPE_NOT_OP_ARGS = ("tokens", "max_position", "gemma_prefill_band")


#: The element gate of a Gemma-norm cell at prefill token counts; see `_rope_compare`.
GEMMA_PREFILL_ATOL = 0.125


def _rope_compare(
    out: torch.Tensor, ref: torch.Tensor, *, element_atol: float | None = None
) -> None:
    """A rotation is a two-term combination, and two terms can cancel.

    `x1 * cos - x2 * sin` goes to zero wherever the two products nearly
    match, and there torch's per-element rtol is meaningless: at 2048 tokens
    roughly 2e-4 of the outputs land close enough that a 7.3e-4 absolute
    difference reads as a relative difference of 23. The decode and 37-token
    cells pass under torch's default band only because they are too small to
    sample one.

    So: 2 ulp element-wise against the row scale, 1 of relative RMS -- tight,
    because one rotation is the shortest chain in this catalog. Over 16
    seeds the gpt-oss cells' worst is 0.0156 absolute, one output rounding
    step at magnitude 2-4: 1.3 ulp of the row scale, 0.04 RMS.

    The qwen4_exp cells sit closer, for two reasons that are not defects:
    `1 + w` with `w` near 1 doubles the q/k magnitude to 8-16, where one
    bf16 ulp is 0.0625, and the kernel's fast sincos loses phase as
    `position * freq` grows, so positions up to 32768 rotate a large pair a
    little off. Below a hundred tokens, 16 seeds per cell, that is 0.03125
    max abs and 1.5 ulp of the row scale (0.21 RMS). At 8192 tokens a few
    elements per tensor reach 0.0625 -- one ulp of the rotated pair -- and
    1.98 ulp of the row scale (6 seeds per cell), which a 2-ulp gate passes
    only by luck of where the row's largest element falls. A cell that
    names `gemma_prefill_band` (GEMMA_PREFILL_ATOL = 0.125, one ulp above that
    ceiling) gates each element at `GEMMA_PREFILL_ATOL + 1.6e-2 |ref|` instead
    and keeps the RMS gate (0.13 measured). Every wrong mRoPE wiring lands
    11-19 away, ~100x outside it.

    The 2-ulp element gate admits exactly one output rounding step: two
    adjacent bf16 values differ by at most 2^-7 of the reference element, which
    is at most 2 ulp of its row's largest magnitude. At this target's rank
    heads (12 / 1 / 16, 6 / 1 / 10, 6 / 1 / 1), over 16 seeds per cell (6 at
    8192 rows), the worst element is 2.00 ulp of the row scale (0.0625 at
    magnitude 8-16, one step), the worst relative RMS 0.196 ulp, and no seed
    fails its cell's gate.
    """
    if element_atol is None:
        assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=1.0)
        return
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    o, r = out.float(), ref.float()
    torch.testing.assert_close(o, r, rtol=1.6e-2, atol=element_atol)
    rel_rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    assert rel_rms <= ULP[ref.dtype], f"relative RMS {rel_rms:.3e} exceeds 1 ulp"


def _distinct_rows(tokens: int, max_position: int, g: torch.Generator) -> torch.Tensor:
    """A contiguous int32 `[3, T]` whose three rows differ at every token.

    Rows that agree make every section wiring compute the same rotation, so a
    cell that drew them would certify nothing about the split.
    """
    row0 = torch.randint(0, max_position, (tokens,), generator=g, device="cuda")
    d1 = torch.randint(1, max_position, (tokens,), generator=g, device="cuda")
    d2 = torch.randint(1, max_position - 1, (tokens,), generator=g, device="cuda")
    d2 = d2 + (d2 >= d1).long()
    rows = torch.stack([row0, (row0 + d1) % max_position, (row0 + d2) % max_position])
    assert (rows[0] != rows[1]).all() and (rows[0] != rows[2]).all()
    assert (rows[1] != rows[2]).all()
    return rows.to(torch.int32).contiguous()


def _rope_build(spec, seed):
    """Materialize one cell as (qkv, op kwargs, compare kwargs), with qkv
    separate because it is the buffer the op writes and the caller has to keep
    a copy of.

    Norm weights are drawn near 1, which under the Gemma form puts the q/k
    heads at their largest magnitude. Positions are `0..T-1` unless the spec
    names a `max_position`, in which case they are uniform below it -- three
    distinct rows under mRoPE, one flat row otherwise.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = spec["tokens"]
    heads = spec["num_heads_q"] + spec["num_heads_k"] + spec["num_heads_v"]
    width = heads * spec["head_dim"]
    qkv = torch.randn(tokens, width, generator=g, device="cuda").to(torch.bfloat16)
    q_w, k_w = (
        (torch.randn(spec["head_dim"], generator=g, device="cuda") * 0.1 + 1.0).to(torch.bfloat16)
        for _ in range(2)
    )
    max_position = spec.get("max_position")
    if max_position is None:
        position_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    elif spec.get("use_mrope", False):
        position_ids = _distinct_rows(tokens, max_position, g)
    else:
        position_ids = torch.randint(
            0, max_position, (tokens,), generator=g, device="cuda", dtype=torch.int32
        )
    kwargs = {k: v for k, v in spec.items() if k not in _ROPE_NOT_OP_ARGS}
    kwargs.update(eps=1e-6, q_weight=q_w, k_weight=k_w, position_ids=position_ids)
    compare_kwargs = {"element_atol": GEMMA_PREFILL_ATOL} if spec.get("gemma_prefill_band") else {}
    return qkv, kwargs, compare_kwargs


def _rope_deviation(out: torch.Tensor, ref: torch.Tensor) -> str:
    """Max abs, worst element against the row scale and relative RMS, in bf16 ulp."""
    ulp = 2.0**-8
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / row).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ulp
    return f"max abs {(o - r).abs().max().item():.6f}, elt {elt:.2f} ulp, rms {rms:.3f} ulp"


# qwen4_exp (Qwen3.8-Flash-Next) runs Gemma QK norm and an unscaled rope
# on the first 64 dims (32 pairs), base 1e7, at two geometries. Its
# mrope_section is [11, 11, 10], passed as sections 11 / 11. Positions are
# drawn uniformly below `max_position` (a test axis, not an op argument):
# the kernel's fast sincos loses phase as `position * freq` grows, so the
# position range is part of what a cell certifies.
_QWEN_ROPE = dict(
    rotary_dim=64,
    base=1e7,
    is_neox=True,
    is_qk_norm=True,
    use_gemma=True,
    max_position=32768,
)
_QWEN_MROPE = dict(use_mrope=True, mrope_section1=11, mrope_section2=11, **_QWEN_ROPE)
# Main attention (trunk and MTP): 24 query heads, 2 kv, head_dim 256.
_QWEN_MAIN = dict(num_heads_q=24, num_heads_k=2, num_heads_v=2, head_dim=256, **_QWEN_MROPE)
# The QSA index plane's query rope: 4 index heads plus the one k and one v
# slot of its scratch layout, over index_head_dim 128 (== 2 * rotary_dim,
# so the partial factor is not reapplied).
_QWEN_INDEX_HEADS = dict(num_heads_q=4, num_heads_k=1, num_heads_v=1, head_dim=128)
_QWEN_INDEX_MROPE = dict(**_QWEN_INDEX_HEADS, **_QWEN_MROPE)
_QWEN_INDEX_PLAIN = dict(**_QWEN_INDEX_HEADS, **_QWEN_ROPE)

# One rank's heads under attention TP: tp2 / tep2 hold 12 query heads and 1 kv head,
# tp4 / tep4 hold 6 and 1. On the fused in-proj row everything after k is a "v" head
# that only sets the row stride: 1 v + 12 gate + 3 index-scratch heads at tp2 (16),
# 1 v + 6 gate + 3 at tp4 (10). The MTP draft layer's compact row carries 1 v head.
_QWEN_TP2_ROW = dict(_QWEN_MAIN, num_heads_q=12, num_heads_k=1, num_heads_v=16)
_QWEN_TP4_ROW = dict(_QWEN_MAIN, num_heads_q=6, num_heads_k=1, num_heads_v=10)
_QWEN_TP4_MTP = dict(_QWEN_MAIN, num_heads_q=6, num_heads_k=1, num_heads_v=1)

# The shapes this target calls the variant with, by target directory under
# `models/`. The entry's own `CELLS` are the op's whoever calls it.
QK_NORM_ROPE_TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified": (
        Cell(
            why="tp2 fused-row main rope, 1 token: 12 q / 1 k, 16 tail heads set the stride",
            spec=dict(tokens=1, **_QWEN_TP2_ROW),
        ),
        Cell(
            why="tp2 fused-row main rope, 128 tokens: decode at max_batch_size 128",
            spec=dict(tokens=128, **_QWEN_TP2_ROW),
        ),
        Cell(
            why="tp2 fused-row main rope, 8192 tokens: a max_num_tokens prefill chunk, "
            "under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_TP2_ROW),
        ),
        Cell(
            why="tp4 fused-row main rope, 1 token: 6 q / 1 k, 10 tail heads set the stride",
            spec=dict(tokens=1, **_QWEN_TP4_ROW),
        ),
        Cell(
            why="tp4 fused-row main rope, 1024 tokens: tep4's MTP-7 verify, 128 x 8 rows",
            spec=dict(tokens=1024, **_QWEN_TP4_ROW),
        ),
        Cell(
            why="tp4 fused-row main rope, 8192 tokens: a max_num_tokens prefill chunk, "
            "under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_TP4_ROW),
        ),
        Cell(
            why="tep4 MTP draft layer compact qkv, 1 token: 6 q / 1 k / 1 v",
            spec=dict(tokens=1, **_QWEN_TP4_MTP),
        ),
        Cell(
            why="tep4 MTP draft layer compact qkv, 1024 tokens: the first draft over MTP-7's "
            "128 x 8 verify rows",
            spec=dict(tokens=1024, **_QWEN_TP4_MTP),
        ),
        Cell(
            why="tep4 MTP draft layer compact qkv, 8192 tokens: the draft layer over a "
            "max_num_tokens prefill chunk, under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_TP4_MTP),
        ),
        # Every head on one rank (tp1, and dep2 / dep4 under attention DP): the main
        # rope on the MTP layer's compact row (24 / 2 / 2) and on the fused row
        # (24 / 2 / 29), and the QSA index plane's scratch (4 / 1 / 1 over 128).
        Cell(
            why="qwen4_exp main rope, decode: one token, Gemma norm and interleaved mRoPE "
            "with three distinct position rows",
            spec=dict(tokens=1, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 19 tokens: the size its wiring and saturation tests run at",
            spec=dict(tokens=19, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 8192-token prefill: the Gemma one-ulp tail, under "
            "GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_MAIN),
        ),
        # The main rope run in place on qwen4_exp's fused input projection, whose
        # row is [q | k | v | gate | index | pad] = 55 heads of 256: everything
        # after k is passed as v heads, which only sets the row stride.
        Cell(
            why="qwen4_exp fused-row main rope, 1 token: 29 v heads only set the row stride",
            spec=dict(tokens=1, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp fused-row main rope, 128 tokens: decode at max concurrency",
            spec=dict(tokens=128, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp fused-row main rope, 8192 tokens: prefill, under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 1 token: one k and one v head, the "
            "narrowest layout; 5 warps of an 8-warp block",
            spec=dict(tokens=1, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 2 tokens: the second token's heads "
            "straddle a block boundary",
            spec=dict(tokens=2, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 5 tokens: the last block runs one warp",
            spec=dict(tokens=5, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 17 tokens: one past a power of two",
            spec=dict(tokens=17, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 19 tokens: the size its wiring and "
            "saturation tests run at",
            spec=dict(tokens=19, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 33 tokens: one past a power of two",
            spec=dict(tokens=33, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 65 tokens: one past a power of two",
            spec=dict(tokens=65, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 8192-token prefill: the index site sees the "
            "whole packed batch; under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (plain), decode: the form the index plane runs "
            "today, row 0 alone as a flat [T] position_ids",
            spec=dict(tokens=1, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 19 tokens: an odd count across blocks",
            spec=dict(tokens=19, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 8192-token prefill: under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_PLAIN),
        ),
    ),
}

QK_NORM_ROPE_CELLS = certified_cells(fused_qk_norm_rope, QK_NORM_ROPE_TARGET_CELLS)


@pytest.mark.parametrize("cell", QK_NORM_ROPE_CELLS, ids=[c.why[:44] for c in QK_NORM_ROPE_CELLS])
def test_fused_qk_norm_rope_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(fused_qk_norm_rope)
    qkv, kwargs, compare_kwargs = _rope_build(cell.spec, seed=_seed(cell.why))
    before = qkv.clone()
    ref = fused_qk_norm_rope.reference(qkv, **kwargs)
    out = fused_qk_norm_rope(qkv, **kwargs)
    assert out is qkv, "the op writes in place"
    print(f"\n  {cell.why[:60]}: {_rope_deviation(out, ref)}")
    _rope_compare(out, ref, **compare_kwargs)
    # The v heads pass through bit-exactly; `_rope_compare` alone would allow them
    # an ulp.
    v_start = (kwargs["num_heads_q"] + kwargs["num_heads_k"]) * kwargs["head_dim"]
    assert torch.equal(out[:, v_start:], before[:, v_start:]), "the v heads were written"


def _qwen_spec(num_heads_q: int, tokens: int) -> dict:
    """The certified qwen4_exp mRoPE cell at this geometry, at `tokens` rows."""
    matches = [
        c.spec
        for c in QK_NORM_ROPE_CELLS
        if c.spec["num_heads_q"] == num_heads_q and c.spec.get("use_mrope", False)
    ]
    assert matches, f"no certified mRoPE cell with {num_heads_q} query heads"
    spec = dict(matches[0])
    spec.pop("gemma_prefill_band", None)
    spec["tokens"] = tokens
    return spec


# The two rope forms this target runs mRoPE at: the main attention rope (24 query heads
# over 256) and the QSA index plane's query rope (4 over 128). The split is over frequency
# pairs, independent of the head count; the cells drive the rank geometries (12 and 6 query
# heads, whose 13 and 7 warps a token straddle the 8-warp blocks) with three distinct position
# rows against the reference, which places every pair.
_QWEN_GEOMETRIES = pytest.mark.parametrize("num_heads_q", [24, 4], ids=["main", "index"])


@_QWEN_GEOMETRIES
def test_mrope_section_split_is_11_11_10(num_heads_q: int) -> None:
    """The [11, 11, 10] split, measured against the wrong wirings as well as the right one.

    At rotary_dim 64 there are 32 frequency pairs and the checkpoint's
    mrope_section sums to exactly 32: right only if 11 pairs land on row 0, 11
    on row 1 and 10 on row 2. Every wrong wiring is invisible whenever the
    three position rows agree -- a text batch, a single image -- so this
    compares against them explicitly.
    """
    qkv0, kwargs, _ = _rope_build(_qwen_spec(num_heads_q, 19), seed=10 + num_heads_q)
    pos = kwargs["position_ids"]

    def call(**over):
        qkv = qkv0.clone()
        with validating(fused_qk_norm_rope):
            return fused_qk_norm_rope(qkv, **{**kwargs, **over})

    def reference(**over):
        return fused_qk_norm_rope.reference(qkv0, **{**kwargs, **over})

    got = call()
    _rope_compare(got, reference())

    # The correct wiring sits at ~0.03 max abs; measured, every wrong one lands
    # 11-19 away (~350-600x), so an absolute 1.0 is ~30x above the right
    # answer and an order of magnitude below every wrong one.
    for what, wrong in (
        (
            "row 0 for every pair (mRoPE inert)",
            reference(position_ids=pos[0].contiguous(), use_mrope=False),
        ),
        ("rows 1 and 2 exchanged", reference(position_ids=pos[[0, 2, 1]].contiguous())),
    ):
        far = (got.float() - wrong.float()).abs().max().item()
        print(f"\n  {what}: max abs {far:.6f}")
        assert far > 1.0, f"{what} is indistinguishable from the correct wiring ({far})"
        with pytest.raises(AssertionError):
            _rope_compare(got, wrong)

    # mrope_section1 is load-bearing here: 3*11 = 33 keeps pair 31 on row 1,
    # 3*10 = 30 pushes it back to row 0.
    assert not torch.equal(call(mrope_section1=10), got)
    # mrope_section2 saturates instead: the largest j with j % 3 == 2 below 32
    # is 29, so every mrope_section2 >= 10 selects the same 10 pairs.
    assert torch.equal(call(mrope_section2=10), got)
    assert torch.equal(call(mrope_section2=12), got)


@_QWEN_GEOMETRIES
def test_mrope_reduces_to_plain_rope_when_rows_agree(num_heads_q: int) -> None:
    """Text tokens put the same position in all three rows; that must be the plain path.

    Bitwise, which is what lets the index plane run row 0 alone on the plain
    form and lets a text batch stay where its accuracy gates measured it.
    """
    qkv0, kwargs, _ = _rope_build(_qwen_spec(num_heads_q, 5), seed=16 + num_heads_q)
    flat = kwargs["position_ids"][0].contiguous()

    def call(**over):
        qkv = qkv0.clone()
        with validating(fused_qk_norm_rope):
            return fused_qk_norm_rope(qkv, **{**kwargs, **over})

    three_equal_rows = flat.expand(3, flat.shape[0]).contiguous()
    assert torch.equal(
        call(position_ids=three_equal_rows),
        call(position_ids=flat, use_mrope=False, mrope_section1=0, mrope_section2=0),
    )


def _rope_ok(**over):
    spec = dict(QK_NORM_ROPE_CELLS[0].spec)
    spec.update(over)
    return _rope_build(spec, seed=3)[:2]


def test_the_rope_guard_refuses_a_width_that_disagrees_with_the_head_counts() -> None:
    """Read as whatever the counts say; the heads come out shifted."""
    qkv, kwargs = _rope_ok()
    with pytest.raises(AssertionError, match="last dim must be"):
        with validating(fused_qk_norm_rope):
            fused_qk_norm_rope(qkv[:, :-64], **kwargs)


def test_the_rope_guard_refuses_a_rotary_dim_past_the_head() -> None:
    qkv, kwargs = _rope_ok()
    kwargs["rotary_dim"] = 128
    with pytest.raises(AssertionError, match="exceeds head_dim"):
        with validating(fused_qk_norm_rope):
            fused_qk_norm_rope(qkv, **kwargs)


def test_the_rope_guard_refuses_a_position_id_count_that_is_not_the_token_count() -> None:
    qkv, kwargs = _rope_ok()
    kwargs["position_ids"] = torch.arange(3, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="one position per token"):
        with validating(fused_qk_norm_rope):
            fused_qk_norm_rope(qkv, **kwargs)


# ---------------------------------------------------------- qsa_paged_index_store ---


def _index_store_compare(out, ref) -> None:
    """Bitwise over both whole planes, as raw bits.

    A store is a move with one conversion that `reference` performs too, so
    there is nothing to spend a tolerance on; comparing every slot rather
    than the written ones is the stray-write check.
    """
    for name, got, want in zip(("index_cache", "position_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


KEY_FILL = -7.0
POS_FILL = -7
ENGINE_MAX_NUM_TOKENS = 16384


def _shared_page_planes(spec, extras: dict):
    """The NVFP4-KV pool's side planes: each physical page holds the `layers`
    index-K pages and then the position page, so the index cache (the last
    role) and the position cache are strided views of one byte allocation.
    Every role holds `KEY_FILL` and the position page `POS_FILL`; the other
    roles are recorded for the untouched check."""
    layers, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    index_bytes = tpb * hkv * head_dim * 2
    page = torch.empty(pages, layers * index_bytes + tpb * axes * 4, dtype=torch.uint8, device=DEV)
    page[:, : layers * index_bytes].view(torch.bfloat16).fill_(KEY_FILL)
    page[:, layers * index_bytes :].view(torch.int32).fill_(POS_FILL)
    index_cache = page[:, (layers - 1) * index_bytes : layers * index_bytes].view(torch.bfloat16)
    index_cache = index_cache.view(pages, tpb, hkv, head_dim)
    position_cache = page[:, layers * index_bytes :].view(torch.int32).view(pages, tpb, axes)
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    assert position_cache.stride() == spec["position_strides"]
    extras["roles"] = page[:, : (layers - 1) * index_bytes]
    return index_cache, position_cache


def _coalesced_planes(spec, extras: dict):
    """The bf16 index pool of the default target: the `layers` sparse layers' INDEX_KEY buffers
    coalesced into one allocation, so the cell's plane (layer `layer`) is a strided view, and the
    position plane its own contiguous pool. The whole allocation holds `KEY_FILL`; the pool is
    recorded for the untouched check."""
    layers, layer, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "layer", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    pool = torch.full(
        (pages, layers, tpb, hkv, head_dim), KEY_FILL, dtype=torch.bfloat16, device=DEV
    )
    index_cache = pool[:, layer]
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    extras["coalesced"] = (pool, layer)
    return index_cache, position_cache


def _other_layers_untouched(pool: torch.Tensor, layer: int) -> None:
    """Every other layer's plane in the coalesced pool still holds `KEY_FILL`, checked a band of
    pages at a time so the check stays a fraction of the pool's size."""
    fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
    bits = pool.view(torch.int16)
    for start in range(0, pool.shape[0], 16384):
        band = bits[start : start + 16384]
        for other in range(pool.shape[1]):
            if other != layer:
                assert bool((band[:, other] == fill).all()), (
                    f"layer {other}'s index plane moved in pages {start}..{start + band.shape[0]}"
                )


def _index_store_build(spec, seed: int, planes=None):
    """Real planes, table and rows for one cell; returns the op's kwargs and extras.

    `planes`, if given, is an `(index_cache, position_cache)` pair to store into instead of the
    cell's own (a stock manager's views)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim, axes = (spec[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]
    rows = len(positions)

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec.get("straddle"):
        # The batch's last row stores into the pool's last page; every other row keeps its page
        # from the permutation, so the rows land on both sides of 2^31 elements of the plane.
        last_start, last_rows = segments[-1]
        column = (last_start + last_rows - 1) // tpb
        live = table[: len(segments)]
        holder = (live == pages - 1).nonzero()
        if len(holder):
            live[holder[0, 0], holder[0, 1]] = live[len(segments) - 1, column]
        live[len(segments) - 1, column] = pages - 1
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)
        rows += 1
    table = table.to(DEV)

    projection = torch.randn(
        rows, spec["index_heads"] + hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["key_dtype"])
    token_k = projection[:, spec["index_heads"] :]
    if spec["key_layout"] == "fused_row":
        # The target's fused input projection row, [q | k | v | gate | index q, k |
        # pad of one index head], in heads of 128: the raw index key is the second to
        # last (108 of 110 at 14080 wide; 32 of 34 at the default target's 4352).
        heads = spec.get("row_width", 14080) // head_dim
        row = torch.randn(rows, heads, head_dim, generator=gen, dtype=torch.float32)
        token_k = row.to(device=DEV, dtype=spec["key_dtype"])[:, heads - 2 : heads - 2 + hkv]
        assert token_k.stride(0) == heads * head_dim

    request_indices = torch.tensor(requests, dtype=spec["request_dtype"], device=DEV)
    logical_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)

    # Three genuinely different axes, as an image span has: t, t + 7, t + 14.
    coords = logical_positions.to(torch.int32)[:, None] + 7 * torch.arange(
        3, dtype=torch.int32, device=DEV
    )
    # The engine's `[3, max_num_tokens]` position buffer, transposed.
    buffer = torch.full((3, ENGINE_MAX_NUM_TOKENS), -1, dtype=torch.int32, device=DEV)
    buffer[:, :rows] = coords.t()
    coords = buffer[:, :rows].t()

    extras = {}
    position_cache = None
    if planes is not None:
        index_cache, position_cache = planes
    elif spec["cache_layout"] == "shared_page":
        index_cache, position_cache = _shared_page_planes(spec, extras)
    elif spec["cache_layout"] == "coalesced":
        index_cache, position_cache = _coalesced_planes(spec, extras)
    else:
        index_cache = torch.full((pages, tpb, hkv, head_dim), KEY_FILL, device=DEV).to(
            torch.bfloat16
        )
    if position_cache is None:
        position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    args = dict(
        token_k=token_k,
        position_coordinates=coords,
        index_cache=index_cache,
        position_cache=position_cache,
        request_indices=request_indices,
        logical_positions=logical_positions,
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, extras


@pytest.mark.parametrize(
    "cell", qsa_paged_index_store.CELLS, ids=[c.why[:44] for c in qsa_paged_index_store.CELLS]
)
def test_qsa_paged_index_store_certified_cells(cell, certifying):  # noqa: F811
    certifying(qsa_paged_index_store)
    args, extras = _index_store_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if cell.spec.get("straddle"):
        # The last row's page starts past 2^31 elements of the plane it is stored through.
        assert (cell.spec["pages"] - 1) * args["index_cache"].stride(0) >= 2**31
    ref = qsa_paged_index_store.reference(**args)
    out = qsa_paged_index_store(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _index_store_compare(out, ref)
    if "roles" in extras:
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        assert bool((extras["roles"].view(torch.int16) == fill).all()), (
            "another index-K role of the shared page moved"
        )
    if "coalesced" in extras:
        _other_layers_untouched(*extras["coalesced"])


#: The cell of 32 MTP7 verify blocks (256 rows): the segments, pool and table it runs over.
_INDEX_STORE_MTP7_32 = next(
    c.spec for c in qsa_paged_index_store.CELLS if c.why.startswith("decode MTP7: 32 requests")
)


def _index_store_small(seed: int, **overrides):
    """Four decode rows at the checkpoint's call-site layouts."""
    spec = {**_INDEX_STORE_MTP7_32, "segments": ((0, 1), (7, 1), (32, 1), (40, 1)), "pages": 8}
    return _index_store_build({**spec, **overrides}, seed=seed)[0]


def test_the_index_store_refuses_unequal_metadata_before_launching() -> None:
    """Request rows and positions of different lengths raise before the launch, the planes
    untouched."""
    args = _index_store_small(seed=5)
    args["logical_positions"] = args["logical_positions"][:-1]
    before = [_bits(args["index_cache"]).clone(), _bits(args["position_cache"]).clone()]
    with pytest.raises(ValueError, match="equally long"):
        qsa_paged_index_store(**args)
    assert torch.equal(_bits(args["index_cache"]), before[0])
    assert torch.equal(_bits(args["position_cache"]), before[1])


def _pool_layout():
    """The target's layout test, loaded by path: its `_manager` builds the stock manager the
    engine builds on a rank of a member (core cache config, bf16 KV), and `ATTN_LAYERS` names the
    sparse layers. Loaded rather than restated so the two cannot disagree on what the stock pool
    is."""
    path = Path(__file__).resolve().parent / "test_routing_and_layout.py"
    spec = importlib.util.spec_from_file_location("_qsa_index_store_pool_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Each member's fused input projection row at a rank (`weights.py`'s `attn_in_rows`): the stock
#: cells at that width are the ones the member stores with.
ROW_WIDTH = {
    "tp1": 14080,
    "tp2": 7424,
    "tep2": 7424,
    "dep2": 14080,
    "tp4": 4352,
    "tep4": 4352,
    "dep4": 14080,
}


def _stock_cell(member: str) -> dict:
    """The first cell over the stock bf16 pool at `member`'s row width."""
    return next(
        c.spec
        for c in qsa_paged_index_store.CELLS
        if c.spec["cache_layout"] == "coalesced" and c.spec["row_width"] == ROW_WIDTH[member]
    )


def _fill_bits() -> int:
    return int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])


@pytest.mark.parametrize("member", ["tp1", "tp2", "tp4", "dep4"])
def test_the_stock_managers_planes_are_the_cells_layout(member) -> None:
    """The stock manager's own index and position planes at rank 0 of `member` have the layout
    the member's cells model, and the op stores through them with the member's row width: four
    rows, the last on the pool's last page, compared against `reference` over the whole plane,
    every other sparse layer's plane untouched. The member's row width is derived from its heads
    in the layout test's `LAYOUT`: `[q | k | v | gate]` heads of 256 and the index
    projection's `4 + 1` heads of 128 plus one pad head. One member per row width and attention
    layout: tep2 and tep4 bind tp2's and tp4's planes, dep2 dep4's."""
    layout = _pool_layout()
    assert set(ROW_WIDTH) == set(layout.MEMBERS)
    text, mapping = layout._TEXT, layout.MEMBERS[member]
    attn_tp = 1 if mapping.get("enable_attention_dp", False) else mapping["tp_size"]
    heads_q = text["num_attention_heads"] // attn_tp
    heads_kv = layout.LAYOUT[(member, 0)]["heads_kv"]
    index_heads = text["indexer_n_heads"] + text["indexer_kv_heads"] + 1
    assert ROW_WIDTH[member] == (
        2 * (heads_q + heads_kv) * text["head_dim"] + index_heads * text["indexer_head_dim"]
    )
    cell = _stock_cell(member)
    tpb, hkv, head_dim, axes = (cell[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    with layout._manager(member, 0) as cm:
        planes = [cm.get_index_k_buffer(i) for i in layout.ATTN_LAYERS]
        index_cache, position_cache = planes[cell["layer"]], cm.get_qsa_position_buffer()
        assert len(planes) == cell["layers"]
        assert index_cache.stride() == (cell["page_stride"], hkv * head_dim, head_dim, 1)
        assert position_cache.stride() == (tpb * axes, axes, 1)
        assert position_cache.shape[0] == index_cache.shape[0]
        plane_bytes = tpb * hkv * head_dim * index_cache.element_size()
        offsets = [p.data_ptr() - planes[0].data_ptr() for p in planes]
        assert offsets == [j * plane_bytes for j in range(len(planes))]

        for plane in planes:
            plane.fill_(KEY_FILL)
        position_cache.fill_(POS_FILL)
        spec = dict(
            cell,
            segments=((0, 1), (31, 1), (33, 2)),
            pages=index_cache.shape[0],
            slots=8,
        )
        args, _ = _index_store_build(spec, seed=71, planes=(index_cache, position_cache))
        assert int(args["block_table"][2, 1]) == index_cache.shape[0] - 1
        ref = qsa_paged_index_store.reference(**args)
        qsa_paged_index_store(**args)
        _index_store_compare((index_cache, position_cache), ref)
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        for j, plane in enumerate(planes):
            if j != cell["layer"]:
                assert bool((_bits(plane) == fill).all()), f"sparse layer {j}'s plane moved"


@pytest.mark.parametrize("member", ("tp2", "dep4"))
def test_the_draft_layers_stock_plane_takes_the_packed_key(monkeypatch, member) -> None:
    """The draft site (`_mtp_index_branch`): the draft layer's own stock index plane -- in the
    separate draft manager under attention TP (tp2), in the trunk's manager after the trunk's
    layers under attention DP (dep4) -- stored through with token_k off the draft's packed
    `[T, 4 + 1, 128]` index projection (row stride 640): three MTP3 verify blocks of 4 rows, two
    across a page edge and the last ending on the plane's last page, compared against `reference`
    over the whole plane, every other sparse layer's plane in that manager untouched."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    layout = _pool_layout()
    separate = not layout.MEMBERS[member].get("enable_attention_dp", False)
    spec_config = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    with layout._manager(member, 0, spec_config=spec_config, is_draft=separate) as cm:
        draft = layout.MTP_LAYER
        sparse = [draft] if separate else [*layout.ATTN_LAYERS, draft]
        assert draft in cm.layer_offsets
        if separate:
            assert set(cm.layer_offsets) == {draft}, sorted(cm.layer_offsets)
        planes = {i: cm.get_index_k_buffer(i) for i in sparse}
        index_cache, position_cache = planes[draft], cm.get_qsa_position_buffer()
        packed = next(
            c.spec for c in qsa_paged_index_store.CELLS if c.spec["key_layout"] == "packed"
        )
        tpb, hkv, head_dim = (packed[k] for k in ("tokens_per_block", "hkv", "head_dim"))
        assert index_cache.shape[1:] == (tpb, hkv, head_dim)
        assert index_cache.stride()[1:] == (hkv * head_dim, head_dim, 1)
        assert position_cache.shape == (index_cache.shape[0], tpb, packed["axes"])

        for plane in planes.values():
            plane.fill_(KEY_FILL)
        position_cache.fill_(POS_FILL)
        spec = dict(
            packed,
            segments=((0, 4), (29, 4), (61, 4)),
            pages=index_cache.shape[0],
            slots=8,
            straddle=True,
        )
        args, _ = _index_store_build(spec, seed=73, planes=(index_cache, position_cache))
        assert args["token_k"].stride(0) == (packed["index_heads"] + hkv) * head_dim
        assert int(args["block_table"][2, 64 // tpb]) == index_cache.shape[0] - 1
        print(
            f"  {member} draft plane: {index_cache.shape[0]} pages at page stride "
            f"{index_cache.stride(0)}, {len(planes)} sparse layer(s) in the manager"
        )
        ref = qsa_paged_index_store.reference(**args)
        qsa_paged_index_store(**args)
        _index_store_compare((index_cache, position_cache), ref)
        for i, plane in planes.items():
            if i != draft:
                assert bool((_bits(plane) == _fill_bits()).all()), f"sparse layer {i}'s plane moved"


@pytest.mark.parametrize(
    "prefix", ("tp2 decode: 128 requests", "dep4 MTP3 verify"), ids=("tp2-decode", "dep4-mtp3")
)
def test_a_captured_store_replays_rewritten_metadata(prefix) -> None:
    """The decode sites run inside the captured graph. A store captured at the cell's batch (one
    row per program at 128 rows, 16 rows per program at 512), replayed after the next step's
    keys, coordinates, request rows, positions and table are written into the captured buffers
    in place, stores that next step bitwise as `reference` computes it over refilled planes.

    The pool is sized to the batch rather than the measured one: the capture, not the offsets,
    is the claim (the cell itself holds the offsets past 2^31 elements)."""
    cell = next(c.spec for c in qsa_paged_index_store.CELLS if c.why.startswith(prefix))
    spec = dict(cell, pages=24576, straddle=False)
    args, extras = _index_store_build(spec, seed=101)
    index_cache, position_cache = args["index_cache"], args["position_cache"]
    qsa_paged_index_store(**args)  # compiles the launch before the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_paged_index_store(**args)

    shifted = dict(spec, segments=tuple((max(0, s - 3), n) for s, n in spec["segments"]))
    following, _ = _index_store_build(shifted, seed=202, planes=(index_cache, position_cache))
    before = qsa_paged_index_store.reference(**args)
    for name in (
        "token_k",
        "position_coordinates",
        "request_indices",
        "logical_positions",
        "block_table",
    ):
        args[name].copy_(following[name])
    index_cache.fill_(KEY_FILL)
    position_cache.fill_(POS_FILL)
    ref = qsa_paged_index_store.reference(**args)
    assert not torch.equal(_bits(ref[0]), _bits(before[0])), "the next step stores nothing new"
    graph.replay()
    torch.cuda.synchronize()
    _index_store_compare((index_cache, position_cache), ref)
    _other_layers_untouched(*extras["coalesced"])


# ------------------------------------------------------------- qsa_paged_kv_store ---


def _kv_store_compare(out, ref) -> None:
    """Bitwise over both role planes, compared as raw bits.

    A store is a move with one conversion, and `reference` performs the
    same conversion, so there is no rounding to spend a tolerance on. In
    the target both planes together are the whole pool allocation, so this
    is also the stray-write check.
    """
    for name, got, want in zip(("k_cache", "v_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


FILL = -2.0


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


class _ArrayInterface:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = dict(
            shape=(nbytes,), strides=None, typestr="|u1", data=(ptr, False), version=2
        )


class _KvEnginePool:
    """One attention layer's K and V views of the stock manager's QSA pool at `pages` pages.

    A page holds `role_pages` role planes of `[hkv, tpb, D]` (K then V of every attention
    layer); the views are layer `layer_slot`'s. The range is reserved whole and physical memory
    is mapped under the K and V planes of `touched` pages only, and under page 0's (a tensor's
    base pointer). Reading an unmapped address faults, so only those pages are ever read.
    """

    def __init__(self, spec, touched):
        dtype = spec["cache_dtype"]
        elem = torch.empty((), dtype=dtype).element_size()
        hkv, tpb, dim = spec["hkv"], spec["tokens_per_block"], spec["head_dim"]
        self.pages, self.touched = spec["pages"], sorted(set(touched))
        self.role = hkv * tpb * dim
        self.stride = spec["role_pages"] * self.role
        self.own = spec["layer_slot"] * 2 * self.role

        def planes(page):
            """This layer's K and V planes of `page`, in bytes of the range."""
            lo = (page * self.stride + self.own) * elem
            return lo, lo + 2 * self.role * elem

        self.written = [planes(p) for p in self.touched]

        dev = torch.cuda.current_device()
        prop = cuda_drv.CUmemAllocationProp()
        prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        gran = _check(
            cuda_drv.cuMemGetAllocationGranularity(
                prop,
                cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        self.size = -(-self.pages * self.stride * elem // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        merged = []
        for lo, hi in sorted({planes(0), *self.written}):
            lo, hi = lo // gran * gran, min(-(-hi // gran) * gran, self.size)
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        access = cuda_drv.CUmemAccessDesc()
        access.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev
        access.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.maps, self.spans = [], merged
        for lo, hi in merged:
            handle = _check(cuda_drv.cuMemCreate(hi - lo, prop, 0))
            ptr = cuda_drv.CUdeviceptr(self.va + lo)
            _check(cuda_drv.cuMemMap(ptr, hi - lo, 0, handle, 0))
            self.maps.append((ptr, hi - lo, handle))
            _check(cuda_drv.cuMemSetAccess(ptr, hi - lo, [access], 1))

        self.raw = torch.as_tensor(_ArrayInterface(self.va, self.size), device=DEV)
        pool = self.raw.view(dtype).as_strided(
            (self.pages, 2, hkv, tpb, dim),
            (self.stride, self.role, tpb * dim, dim, 1),
            self.own,
        )
        self.k, self.v = pool[:, 0], pool[:, 1]
        self.index = torch.tensor(self.touched, dtype=torch.long, device=DEV)
        self.fill()

    def fill(self) -> None:
        """Every mapped byte to the sentinel's pattern, and remember it."""
        dtype = self.k.dtype
        pattern = _bits(torch.full((2,), FILL, dtype=torch.bfloat16).to(dtype))
        for lo, hi in self.spans:
            self.raw[lo:hi].view(pattern.dtype).fill_(int(pattern[0]))
        torch.cuda.synchronize()
        self.before = [self.raw[lo:hi].clone() for lo, hi in self.spans]

    def dense(self, table: torch.Tensor):
        """`(k, v, table)`: copies of the touched pages, and `table` renumbered into them.

        An entry naming no touched page becomes `-1`. No row resolves to one, so the
        reference reads the same pages it would over the whole pool.
        """
        remap = torch.full((self.pages,), -1, dtype=torch.int32, device=DEV)
        remap[self.index] = torch.arange(len(self.touched), dtype=torch.int32, device=DEV)
        renamed = torch.where(table >= 0, remap[table.clamp_min(0).long()], -1).to(torch.int32)
        return self.k[self.index].clone(), self.v[self.index].clone(), renamed.contiguous()

    def gathered(self):
        return self.k[self.index], self.v[self.index]

    def assert_rest_unchanged(self) -> None:
        """Every mapped byte outside the touched pages' K and V planes is what it was."""
        torch.cuda.synchronize()
        for (lo, hi), before in zip(self.spans, self.before):
            keep = torch.ones(hi - lo, dtype=torch.bool, device=DEV)
            for p_lo, p_hi in self.written:
                if p_hi > lo and p_lo < hi:
                    keep[max(p_lo, lo) - lo : min(p_hi, hi) - lo] = False
            assert torch.equal(self.raw[lo:hi][keep], before[keep]), (
                f"bytes [{lo}, {hi}) of the pool changed outside the touched pages' planes"
            )

    def close(self) -> None:
        torch.cuda.synchronize()
        del self.k, self.v, self.raw
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


def _kv_store_build(spec, seed: int):
    """Real pool, table and rows for one cell: `(op kwargs, pool)`.

    `pool` is the contiguous pool tensor, or the `_KvEnginePool` under a `pool="engine"` cell.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim = spec["tokens_per_block"], spec["hkv"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    engine = spec.get("pool") == "engine"
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    if engine:
        # The first row's page is the pool's last one, whose planes are its highest bytes.
        first = segments[0][0] // tpb
        last = int((order == pages - 1).nonzero()[0])
        order[[first, last]] = order[[last, first]]
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        assert cols <= width, f"request {r} ends past the table's {width} columns"
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)

    rows = len(positions)
    assert rows == spec.get("rows", rows), (rows, spec["rows"])
    qkv = torch.randn(
        rows,
        spec["heads_q"] + 2 * hkv + spec.get("row_pad_heads", 0),
        head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(device=DEV, dtype=spec["row_dtype"])
    assert qkv.stride(0) == spec.get("row_stride", qkv.stride(0)), qkv.stride(0)
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv : spec["heads_q"] + 2 * hkv]
    touched = [int(table[r, p // tpb]) for r, p in zip(requests, positions)]
    table = table.to(DEV)

    if engine:
        pool = _KvEnginePool(spec, [t for t in touched if t >= 0])
        k_cache, v_cache = pool.k, pool.v
    else:
        pool = torch.full(
            (pages, 2, hkv, tpb, head_dim), FILL, dtype=torch.bfloat16, device=DEV
        ).to(spec["cache_dtype"])
        k_cache, v_cache = pool[:, 0], pool[:, 1]
    args = dict(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, pool


def _kv_engine_reference(args, pool: _KvEnginePool):
    """The entry's reference over a dense copy of the touched pages."""
    dense_k, dense_v, renamed = pool.dense(args["block_table"])
    return qsa_paged_kv_store.reference(
        **{**args, "k_cache": dense_k, "v_cache": dense_v, "block_table": renamed}
    )


def _assert_reaches_past_int32(pool: _KvEnginePool) -> None:
    """The highest touched page's planes lie past 2^32 elements of the view's base: a page
    offset computed in 32 bits would land elsewhere."""
    top = max(pool.touched)
    assert top == pool.pages - 1, (top, pool.pages)
    assert top * pool.k.stride(0) >= 2**32, top * pool.k.stride(0)


@pytest.mark.parametrize(
    "cell", qsa_paged_kv_store.CELLS, ids=[c.why[:44] for c in qsa_paged_kv_store.CELLS]
)
def test_qsa_paged_kv_store_certified_cells(cell, certifying):  # noqa: F811
    certifying(qsa_paged_kv_store)
    args, pool = _kv_store_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if isinstance(pool, _KvEnginePool):
        try:
            _assert_reaches_past_int32(pool)
            before = pool.gathered()
            before = (before[0].clone(), before[1].clone())
            ref = _kv_engine_reference(args, pool)
            out = qsa_paged_kv_store(**args)
            assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
            got = pool.gathered()
            _kv_store_compare(got, ref)
            pool.assert_rest_unchanged()
            wrote = any(not torch.equal(_bits(g), _bits(b)) for g, b in zip(got, before))
            assert wrote, "a cell with rows must write"
            print(
                f"[{cell.why[:44]}] {len(pool.touched)} of {pool.pages} pages touched; "
                f"top plane at byte {pool.written[-1][0]} of {pool.size}; "
                f"{sum(hi - lo for lo, hi in pool.spans)} bytes mapped"
            )
        finally:
            pool.close()
        return
    before = pool.clone()
    ref = qsa_paged_kv_store.reference(**args)
    out = qsa_paged_kv_store(**args)
    assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
    _kv_store_compare(out, ref)
    wrote = not torch.equal(_bits(pool), _bits(before))
    assert wrote == bool(cell.spec["segments"]), (
        "a cell with rows must write, an empty one must not"
    )


#: The captured steps of the FP8 unified target's members, by the cell they run.
_KV_STORE_CAPTURED = (
    "tp2 decode, 128 rows (max_batch_size) off the 7424-wide row",
    "dep4 MTP3 verify, 128 requests x 4 rows (512)",
)


@pytest.mark.parametrize("why", _KV_STORE_CAPTURED, ids=[w[:44] for w in _KV_STORE_CAPTURED])
def test_a_captured_kv_store_replays_over_the_engine_pool(why) -> None:
    """A captured decode or verify step over the engine's own pool: captured with every
    position at -1 (each row skipped, nothing read from the table or written), replayed after
    the step's positions are written into the same buffer."""
    cell = next(c for c in qsa_paged_kv_store.CELLS if c.why.startswith(why))
    args, pool = _kv_store_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    try:
        positions = args["logical_positions"]
        saved = positions.clone()
        positions.fill_(-1)
        before = tuple(t.clone() for t in pool.gathered())

        def call():
            qsa_paged_kv_store(**args)

        for _ in range(3):  # compile the Triton kernel before capture
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        torch.cuda.synchronize()
        _kv_store_compare(pool.gathered(), before)  # a row at position -1 writes nothing

        positions.copy_(saved)
        ref = _kv_engine_reference(args, pool)
        graph.replay()
        torch.cuda.synchronize()
        _kv_store_compare(pool.gathered(), ref)
        pool.assert_rest_unchanged()
    finally:
        pool.close()


# ----------------------------------------------------------- qsa_paged_sparse_gqa ---


def _sparse_gqa_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    The kernel rounds its online-softmax probabilities to the cache dtype
    before P.V, so the error tracks the magnitude of the values averaged,
    not of the output: against an fp32 reference it measured 1.2e-3 on rows
    averaging a thousand tokens (outputs 0.14 to 0.4) and 1.2e-2 on rows
    averaging a handful (outputs to 5). Scaled by each head row's own
    magnitude that is one band. Against this reference (fp32, rounded once)
    every ungated cell measured at most 2.0 element ulp -- a single final-rounding
    flip, since bf16's step is 2 of these units at the bottom of a binade --
    and at most 0.62 ulp RMS: 2x and 3.2x margin. Six wrong references (no
    scale, interleaved heads, padding attended, identity pages, K/V
    swapped, page/slot transposed) were each measured more than 0.1 off.

    The default target's cells (6/1 heads at 1 to 8192 rows, 24/2 over an
    e4m3 pool) sit in the same band, and the table shifted by one page, the
    softmax scale dropped and one selected token dropped a row each fall
    outside it. This target's tp2 / tep2 cells (12/1 off the 29-head row) and
    tp1 / dep2 cells (24/2 over a [128, 292] table) do too: gated at most 3.32
    element ulp and 0.644 RMS, ungated at most 2.00 and 0.574; the 12/1 wrong
    references sit 278 ulp or more off (`test_tp2_rank_wrong_references_...`).

    A gated output is rounded twice, the attention and then the gated
    product, each by at most one bf16 ulp of the value rounded, so its bound
    is near the band's 4 element ulp. The gate not applied, or read one head
    early, falls far outside it; one selected token dropped at the 1-row
    gated cell is refused on the RMS gate.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _sparse_gqa_ulp(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """`_sparse_gqa_compare`'s two statistics, measured: (max element, relative RMS), in ulp of q's dtype."""
    ulp = ULP[ref.dtype]
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / scale).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    return element, rms


_KERNELS = {
    "_qsa_paged_sparse_gqa_kernel": "prefill",
    "_qsa_paged_sparse_gqa_splitk_kernel": "splitk",
    "_qsa_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing and whether it launched with PDL, then
    delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        entry = [self._name, tuple(grid), False]
        self._log.append(entry)
        launch = self._inner[grid]

        def run(*args, **kwargs):
            entry[2] = bool(kwargs.get("launch_pdl", False))
            return launch(*args, **kwargs)

        return run


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order: [name, grid, PDL]."""
    log: list = []
    for name, tag in _KERNELS.items():
        inner = getattr(_qsa_kernels, name)
        monkeypatch.setattr(_qsa_kernels, name, _LaunchRecorder(inner, tag, log))
    return log


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


#: NaN's bit pattern per pool dtype: filled through an integer view, since not every fill
#: kernel takes an e4m3 tensor.
_NAN_BITS = {torch.bfloat16: (torch.int16, 0x7FC0), torch.float8_e4m3fn: (torch.uint8, 0x7F)}


def _nan_pool(shape, dtype):
    bits_dtype, nan = _NAN_BITS[dtype]
    return torch.full(shape, nan, dtype=bits_dtype, device=DEV).view(dtype)


def _sparse_gqa_build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names.

    The pool is `[pool_pages, slot_pages, kv, tpb, D]` and the call's K and V are layer
    `layer`'s two role planes: tp1's cells keep one layer's adjacent pair (2 role pages a
    page), the default target's cells the engine's trunk (24). The requests sit on the
    pool's last pages; every other page, and every other plane, is NaN, so an address that
    strayed could not come back finite.
    """
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
    needed = sum(mapped) + 7
    pool_pages = spec.get("pool_pages", needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    slot_pages, layer = spec.get("slot_pages", 2), spec.get("layer", 0)
    planes = slice(2 * layer, 2 * layer + 2)
    pool = _nan_pool((pool_pages, slot_pages, kv, tpb, d), spec["cache_dtype"])
    pool[first:, planes] = torch.randn(needed, 2, kv, tpb, d, generator=gen, device=DEV).to(
        spec["cache_dtype"]
    )
    k_cache, v_cache = pool[:, 2 * layer], pool[:, 2 * layer + 1]

    q_dtype = spec["q_dtype"]
    fused = spec["q_layout"] == "fused_row"
    # The row q is read off: the packed `[q | k | v]` buffer, or the target's fused input
    # projection row `[q | k | v | gate | index q, k | pad]` (`row_heads` heads of D).
    row_heads = spec.get("row_heads", 55 if fused else q_heads + 2 * kv)
    row = torch.randn(rows, row_heads * d, generator=gen, device=DEV).to(q_dtype)
    q = row.view(rows, row_heads, d)[:, :q_heads]

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    gate = None
    if spec.get("gate") == "fused_row":
        assert fused, "the gate is read off the fused row"
        gate = _row_gate(q, spec)

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
    if gate is not None and spec.get("gate_fused", True):
        args.update(output_gate=gate)
    return args


def _row_gate(q, spec):
    """The output gate's columns of the fused row `q` is read off: after q, k and v."""
    rows, d = q.shape[0], spec["head_dim"]
    row = q.as_strided((rows, q.stride(0)), (q.stride(0), 1))
    start = (spec["q_heads"] + 2 * spec["kv_heads"]) * d
    return row[:, start : start + spec["q_heads"] * d]


def _bound(positions, spec):
    """The prefill kernel's per-row scan bound, from the positions alone."""
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _assert_straddles(args, edge: int) -> None:
    """The pages the cell's rows map put the K and V planes' element offsets on both sides
    of `edge`: what the cell claims about its pool."""
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        ends = plane.storage_offset() + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= edge).any()), f"{name}: no mapped page lies past {edge}"
        assert bool((ends < edge).any()), f"{name}: every mapped page lies past {edge}"


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


def _refill(args, spec, seed) -> None:
    """The engine's next step, written into the call's own buffers: new q and gate values
    (the fused row's columns), a new selection at the same positions, and new K/V in every
    mapped page."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for name in ("q", "output_gate"):
        if name in args:
            view = args[name]
            view.copy_(torch.randn(view.shape, generator=gen, device=DEV).to(view.dtype))
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in spec["segments"]])
    args["selected_tokens"].copy_(
        _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    )
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        bits = _NAN_BITS[plane.dtype][0]
        fresh = torch.randn((pages.numel(), *plane.shape[1:]), generator=gen, device=DEV)
        plane.view(bits)[pages] = fresh.to(plane.dtype).view(bits)


def _replay_in_a_graph(args, spec, seed, launches, first) -> None:
    """The decode site's capture: the call captured into a CUDA graph (its split-K partials
    allocated from the graph's pool), the inputs refilled in place, the graph replayed.
    The replay must take the eager call's launches (`first` is that call's output), answer
    the new inputs within the band, and match an eager call on them bitwise."""
    eager = [tuple(entry) for entry in launches]
    del launches[:]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_paged_sparse_gqa(**args)
    assert [tuple(entry) for entry in launches] == eager, launches
    _refill(args, spec, seed)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    assert not torch.equal(replayed, first), "the replay did not read the refilled inputs"
    _sparse_gqa_compare(replayed, qsa_paged_sparse_gqa.reference(**args))
    assert torch.equal(replayed, qsa_paged_sparse_gqa(**args)), (
        "the replay differs from an eager call"
    )


@pytest.mark.parametrize(
    "cell", qsa_paged_sparse_gqa.CELLS, ids=[c.why[:44] for c in qsa_paged_sparse_gqa.CELLS]
)
def test_qsa_paged_sparse_gqa_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(qsa_paged_sparse_gqa)
    spec = cell.spec
    args = _sparse_gqa_build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    assert rows == spec.get("rows", rows)
    if "row_heads" in spec:
        # q, and the gate when fused, are views of one fused row `row_heads` heads wide.
        q = args["q"]
        assert q.stride(0) == spec["row_heads"] * spec["head_dim"]
        if "output_gate" in args:
            gate = args["output_gate"]
            assert gate.stride(0) == q.stride(0)
            assert gate.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    if "rows_per_request" in spec:
        assert {n for _, n in spec["segments"]} == {spec["rows_per_request"]}
    if "offset_edge" in spec:
        _assert_straddles(args, spec["offset_edge"])
    if spec["prefill_pair"]:
        # The precondition the op cannot check: front-packed, and exactly as
        # many valid entries as the prefill kernel will scan.
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))

    out = qsa_paged_sparse_gqa(**args)
    assert [name for name, _, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    if "splits" in spec:
        assert launches[0][1][2] == spec["splits"], launches
    assert all(pdl == spec.get("pdl", False) for _, _, pdl in launches), launches
    if "output_gate" in args:
        gated = (args["q"], args["k_cache"], args["selected_tokens"], args["output_gate"])
        assert qsa_paged_sparse_gqa.fuses_output_gate(*gated)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    ref = qsa_paged_sparse_gqa.reference(**args)
    element, rms = _sparse_gqa_ulp(out, ref)
    print(f"  {cell.why[:60]}: {element:.2f} element ulp, {rms:.3f} ulp RMS; launches {launches}")
    _sparse_gqa_compare(out, ref)
    if spec.get("gate") == "fused_row" and not spec.get("gate_fused", True):
        # The site's other branch: a launch with no merge cannot carry the gate, so the call
        # site applies it after the call -- and handing it over is refused, not ignored.
        gate = _row_gate(args["q"], spec)
        assert gate.untyped_storage().data_ptr() == args["q"].untyped_storage().data_ptr()
        assert not qsa_paged_sparse_gqa.fuses_output_gate(
            args["q"], args["k_cache"], args["selected_tokens"], gate
        )
        with pytest.raises(ValueError, match="output-gate fusion requires"):
            qsa_paged_sparse_gqa(**args, output_gate=gate)
    if spec.get("capture"):
        seed = sum(map(ord, cell.why)) + 1
        _replay_in_a_graph(args, spec, seed=seed, launches=launches, first=out)


def test_tp2_rank_wrong_references_fall_outside_the_band() -> None:
    """Negative controls at the 12 / 1 head geometry this target adds (tp2 / tep2): the
    call at the 128-row gated decode cell, against references each wrong in one way the
    call site could be -- the gate read one head early off the 29-head row, the table
    shifted by one page, the softmax scale dropped. Each has to fall outside `_sparse_gqa_compare`'s
    band while the right reference sits inside it."""
    cell = next(
        c for c in qsa_paged_sparse_gqa.CELLS if c.why.startswith("tp2/tep2 gated decode 128 x 1")
    )
    spec = cell.spec
    args = _sparse_gqa_build(spec, seed=sum(map(ord, cell.why)))
    out = qsa_paged_sparse_gqa(**args)
    _sparse_gqa_compare(out, qsa_paged_sparse_gqa.reference(**args))
    gate = args["output_gate"]
    early = gate.as_strided(gate.shape, gate.stride(), gate.storage_offset() - spec["head_dim"])
    table = args["block_table"]
    shifted = torch.where(table >= 0, table + 1, table)
    wrong = {
        "gate one head early": {**args, "output_gate": early},
        "table shifted one page": {**args, "block_table": shifted},
        "softmax scale dropped": {**args, "softmax_scale": 1.0},
    }
    for name, bad in wrong.items():
        ref = qsa_paged_sparse_gqa.reference(**bad)
        element, rms = _sparse_gqa_ulp(out, ref)
        print(f"  {name}: {element:.2f} element ulp, {rms:.3f} ulp RMS")
        with pytest.raises(AssertionError):
            _sparse_gqa_compare(out, ref)
