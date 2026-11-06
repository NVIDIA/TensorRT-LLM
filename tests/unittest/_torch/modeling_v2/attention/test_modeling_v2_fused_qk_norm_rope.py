# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_qk_norm_rope catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    FusedQkNormRope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (  # noqa: E501
    fused_qk_norm_rope as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "fused_qk_norm_rope requires a CUDA device"

# Spec keys that shape the test's inputs or gate rather than the op call.
_NOT_OP_ARGS = ("tokens", "max_position", "gemma_prefill_band")


#: The element gate of a Gemma-norm cell at prefill token counts; see `_compare`.
GEMMA_PREFILL_ATOL = 0.125


def _compare(out: torch.Tensor, ref: torch.Tensor, *, element_atol: float | None = None) -> None:
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


def _seed(text: str) -> int:
    """A seed that is the same in every process; `hash()` of a str is not."""
    return zlib.crc32(text.encode())


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


def _build(spec, seed):
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
    kwargs = {k: v for k, v in spec.items() if k not in _NOT_OP_ARGS}
    kwargs.update(eps=1e-6, q_weight=q_w, k_weight=k_w, position_ids=position_ids)
    compare_kwargs = {"element_atol": GEMMA_PREFILL_ATOL} if spec.get("gemma_prefill_band") else {}
    return qkv, kwargs, compare_kwargs


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> str:
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

# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "gpt_oss/gpt_oss_120b__sm_10x__default": (
        Cell(
            why="default target at attention TP8: 8q/1k/1v heads per rank",
            spec=dict(
                tokens=64,
                **{
                    **FusedQkNormRope.GPT_OSS,
                    "num_heads_q": 8,
                    "num_heads_k": 1,
                    "num_heads_v": 1,
                },
            ),
        ),
    ),
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1": (
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

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    qkv, kwargs, compare_kwargs = _build(cell.spec, seed=_seed(cell.why))
    before = qkv.clone()
    ref = op.reference(qkv, **kwargs)
    out = op(qkv, **kwargs)
    assert out is qkv, "the op writes in place"
    print(f"\n  {cell.why[:60]}: {_deviation(out, ref)}")
    _compare(out, ref, **compare_kwargs)
    # The v heads pass through bit-exactly; `_compare` alone would allow them
    # an ulp.
    v_start = (kwargs["num_heads_q"] + kwargs["num_heads_k"]) * kwargs["head_dim"]
    assert torch.equal(out[:, v_start:], before[:, v_start:]), "the v heads were written"


def test_the_v_heads_are_left_alone() -> None:
    """The op's boundary, and nothing about the output shape shows it.

    A kernel that rotated the v heads too would still return a buffer of the
    right size, and every cell compares against a reference that makes the same
    claim -- so this pins it against the input instead.
    """
    spec = dict(CELLS[1].spec)
    qkv, kwargs, _ = _build(spec, seed=11)
    before = qkv.clone()
    op(qkv, **kwargs)
    v_start = (kwargs["num_heads_q"] + kwargs["num_heads_k"]) * kwargs["head_dim"]
    assert torch.equal(qkv[:, v_start:], before[:, v_start:])
    assert not torch.equal(qkv[:, :v_start], before[:, :v_start]), "q and k were not touched"


def _qwen_spec(num_heads_q: int, tokens: int) -> dict:
    """The certified qwen4_exp mRoPE cell at this geometry, at `tokens` rows."""
    matches = [
        c.spec
        for c in CELLS
        if c.spec["num_heads_q"] == num_heads_q and c.spec.get("use_mrope", False)
    ]
    assert matches, f"no certified mRoPE cell with {num_heads_q} query heads"
    spec = dict(matches[0])
    spec.pop("gemma_prefill_band", None)
    spec["tokens"] = tokens
    return spec


# The two geometries qwen4_exp runs the mRoPE form at: the main attention rope
# (24 query heads over 256) and the QSA index plane's query rope (4 over 128).
_QWEN_GEOMETRIES = pytest.mark.parametrize("num_heads_q", [24, 4], ids=["main", "index"])


@_QWEN_GEOMETRIES
def test_mrope_section_split_is_11_11_10(num_heads_q: int) -> None:
    """The [11, 11, 10] split, measured against the wrong wirings as well as the right one.

    At rotary_dim 64 there are 32 frequency pairs and the checkpoint's
    mrope_section sums to exactly 32: right only if 11 pairs land on row 0, 11
    on row 1 and 10 on row 2. Every wrong wiring is invisible whenever the
    three position rows agree -- a text batch, a single image -- so this
    compares against them explicitly. The rule is over frequency pairs, so it
    is independent of head_dim and the head counts; running it at both
    geometries makes that a measurement rather than an argument.
    """
    qkv0, kwargs, _ = _build(_qwen_spec(num_heads_q, 19), seed=10 + num_heads_q)
    pos = kwargs["position_ids"]

    def call(**over):
        qkv = qkv0.clone()
        with validating(op):
            return op(qkv, **{**kwargs, **over})

    def reference(**over):
        return op.reference(qkv0, **{**kwargs, **over})

    got = call()
    _compare(got, reference())

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
            _compare(got, wrong)

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
    qkv0, kwargs, _ = _build(_qwen_spec(num_heads_q, 5), seed=16 + num_heads_q)
    flat = kwargs["position_ids"][0].contiguous()

    def call(**over):
        qkv = qkv0.clone()
        with validating(op):
            return op(qkv, **{**kwargs, **over})

    three_equal_rows = flat.expand(3, flat.shape[0]).contiguous()
    assert torch.equal(
        call(position_ids=three_equal_rows),
        call(position_ids=flat, use_mrope=False, mrope_section1=0, mrope_section2=0),
    )


def _ok(**over):
    spec = dict(CELLS[0].spec)
    spec.update(over)
    return _build(spec, seed=3)[:2]


def test_guard_refuses_a_width_that_disagrees_with_the_head_counts() -> None:
    """Read as whatever the counts say; the heads come out shifted."""
    qkv, kwargs = _ok()
    with pytest.raises(AssertionError, match="last dim must be"):
        with validating(op):
            op(qkv[:, :-64], **kwargs)


def test_guard_refuses_a_rotary_dim_past_the_head() -> None:
    qkv, kwargs = _ok()
    kwargs["rotary_dim"] = 128
    with pytest.raises(AssertionError, match="exceeds head_dim"):
        with validating(op):
            op(qkv, **kwargs)


def test_guard_refuses_a_position_id_count_that_is_not_the_token_count() -> None:
    qkv, kwargs = _ok()
    kwargs["position_ids"] = torch.arange(3, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="one position per token"):
        with validating(op):
            op(qkv, **kwargs)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="the Kimi K3 cell is certified on sm_100"
)
def test_bf16_neox_kimi_k3_drafter() -> None:
    torch.manual_seed(9)
    for num_heads_q, num_heads_kv in ((6, 1), (24, 4)):
        for num_tokens in (1, 8, 64):
            qkv = torch.randn(
                num_tokens,
                (num_heads_q + 2 * num_heads_kv) * 64,
                dtype=torch.bfloat16,
                device="cuda",
            )
            kwargs: dict = dict(
                num_heads_q=num_heads_q,
                num_heads_k=num_heads_kv,
                num_heads_v=num_heads_kv,
                head_dim=64,
                rotary_dim=64,
                eps=1e-5,
                q_weight=torch.randn(64, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0,
                k_weight=torch.randn(64, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0,
                base=10000.0,
                is_neox=True,
                position_ids=torch.randint(
                    0, 32768, (num_tokens,), dtype=torch.int32, device="cuda"
                ),
            )
            ref = op.reference(qkv, **kwargs)
            v_start = (num_heads_q + num_heads_kv) * 64
            v_before = qkv[:, v_start:].clone()
            op(qkv, **kwargs)
            torch.testing.assert_close(qkv, ref, rtol=1.6e-2, atol=0.02)
            assert torch.equal(qkv[:, v_start:], v_before)
