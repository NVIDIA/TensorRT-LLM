# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_fp8_roundtrip entry the hyv4-flash target owns.

The entry owns the contract: ``engram_fp8_roundtrip.CELLS`` is what is certified,
``reference`` -- HF's ``_fp8_round_trip`` with the quotient and the dequant multiply
kept in fp32 -- is what it is certified against, and ``_compare`` below is the band.
What a cell cannot carry is here: the input distribution the gathered table rows
actually have (per-row magnitudes spanning orders of magnitude, and slots nobody
trained), the independent bound against the input, and the measured separation from
the computations that are *not* this op.

The entry carries no ``is_valid``: it is a composition of torch calls, so a shape,
stride or width it accepts is one it computes exactly, and the reference accepts the
same ones. There is no class of input it answers silently wrongly.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.engram.engram_fp8_roundtrip import (  # noqa: E501
    engram_fp8_roundtrip as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "engram_fp8_roundtrip requires a CUDA device"

#: ``engram_n_embed_per_ngram`` (768) / ``engram_n_head_per_ngram`` (8), from config.json.
WIDTH = 96
#: One e4m3 step at the row maximum, in bf16 ulps of the row maximum: 2^-3 / 2^-8.
ONE_E4M3_STEP = 32.0
#: The band `_compare` gates with: 1.25 steps element-wise, and the measured RMS and
#: bit-identical fraction (see ``test_the_band_separates_wrong_variants``).
ELEMENT_ULP = 1.25 * ONE_E4M3_STEP
RMS_ULP = 5.0
EXACT = 0.95

# The entry is target-local, so its own CELLS already are this target's call shapes:
# rows = tokens * 16 heads at 96 wide, from one decode token to the max_num_tokens
# envelope (8192 tokens x 16 heads). There is no second caller to declare cells for.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "hyv4_flash/hy4_flash_fp8__sm_103__default": (),
}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(rows: int, seed: int) -> torch.Tensor:
    """Rows shaped like the gathered table: per-row magnitudes over ~11 binades.

    A hash table's rows are not one distribution -- each row is one head of one
    n-gram's slot, and their amaxes differ by orders of magnitude. Rows 0 and 1 are
    the two the per-row scale exists for: an untouched slot (all zero) and a slot
    whose amax is ~1e-9, below anything a fixed scale would resolve.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, WIDTH), generator=g, device="cuda")
    magnitude = torch.exp2(torch.empty((rows, 1), device="cuda").uniform_(-8.0, 3.0, generator=g))
    x = (x * magnitude).to(torch.bfloat16)
    if rows >= 2:
        x[0] = 0.0
        x[1] *= 2.0**-30
    return x


def _separation(candidate: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Where a value lands against the reference: element ulp, RMS ulp, exact fraction."""
    row_amax = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((candidate.float() - ref.float()).abs() / row_amax).max().item() / ULP[
        torch.bfloat16
    ]
    rms = (
        (candidate.float() - ref.float()).pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()
    ).item() / ULP[torch.bfloat16]
    exact = (candidate == ref).float().mean().item()
    return element, rms, exact


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The band, and where it comes from.

    The entry is HF's chain in bf16: the quotient is divided and rounded in bf16 and
    the dequantized value is multiplied in bf16; the reference does both in fp32.
    A 2^-9 relative move of the quotient is 1/16 of an e4m3 step, so an element
    within that of a rounding midpoint lands on the neighbouring e4m3 value -- one
    step, at most 2^-3 of the row maximum, 32 bf16 ulps -- and the bf16 dequant
    multiply adds 2^-9 relative of the element. Measured over every cell: 19.6 ulps
    element-wise (1.25 steps = 40, 2.0x margin), 4.05 ulps relative RMS (5.0,
    1.24x), and at least 97.0% of elements bit-identical (0.95, 1.02x).

    The element gate cannot separate the round trip from its absence: a half-step
    error (the round trip skipped) is *smaller* than the one-step flip the entry may
    make. The RMS and exactness gates are what separate them, measured in
    ``test_the_band_separates_wrong_variants``.
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 - pytest resolves the fixture by this name
    certifying(op)
    rows = _build(cell.spec["rows"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(rows)
    ref = op.reference(rows)
    assert out.dtype == torch.bfloat16 and out.shape == rows.shape
    _compare(out, ref)
    exact = (out == ref).float().mean().item()
    assert exact >= EXACT, f"{exact:.4f} of elements bit-identical to the reference"


def test_the_round_trip_stays_within_half_an_e4m3_step() -> None:
    """Independent of the reference: the round trip may not move a value far.

    e4m3 rounds to the nearest of 2^3 steps, so the error is at most half a step of
    the element, bounded here by half a step of the row maximum; the bf16 scale and
    dequant roundings add 2^-8 of the row maximum. A composition that dropped the
    dequantize, or scaled by the wrong row, is orders of magnitude out.
    """
    rows = _build(1024, seed=11)
    out = op(rows)
    err = (out.float() - rows.float()).abs()
    row_amax = rows.float().abs().amax(dim=1, keepdim=True)
    assert torch.all(err <= row_amax * (2.0**-4 + 2.0**-8)), (
        f"worst error {(err / row_amax.clamp_min(1e-9)).max().item():.3e} of the row amax"
    )


def test_the_scale_floor_keeps_a_dim_row() -> None:
    """The clamp_min(1e-12) branch, and the reason the engine's per-token op is not it.

    A row whose amax is 1e-9 must survive the round trip; a scale floored at
    1 / (448 * 512) = 4.36e-6 (torch.ops.trtllm.quantize_e4m3_activation's floor)
    quantizes it to zero. A zero row stays exactly zero rather than NaN.
    """
    rows = torch.zeros((2, WIDTH), dtype=torch.bfloat16, device="cuda")
    rows[1] = (torch.randn(WIDTH, device="cuda") * 1e-9).to(torch.bfloat16)
    out = op(rows)
    assert torch.equal(out[0], torch.zeros(WIDTH, dtype=torch.bfloat16, device="cuda"))
    row_amax = rows[1].float().abs().amax()
    assert out[1].float().abs().amax() > 0.5 * row_amax, (
        "a dim row came back empty: the scale floor is not the checkpoint's 1e-12"
    )
    ref = op.reference(rows)
    assert torch.equal(out[0], ref[0])
    assert torch.all(
        (out[1].float() - ref[1].float()).abs() <= row_amax * ONE_E4M3_STEP * ULP[torch.bfloat16]
    ), "a dim row is more than one e4m3 step from the reference"


def test_the_straight_through_form_is_the_identity_in_bf16() -> None:
    """Why ``__call__`` may return the dequantized value instead of HF's ``x + (q - x)``.

    ``q`` is a bf16 value within a few percent of ``x``, so ``q - x`` is exact
    (Sterbenz) and ``x + (q - x)`` rounds back to ``q`` in bf16. If that ever stops
    holding, the straight-through estimator has to come back into ``__call__``.
    """
    rows = _build(256, seed=5)
    q = op(rows)
    assert torch.equal(rows + (q - rows), q)


def test_the_band_separates_wrong_variants() -> None:
    """Every gate above is measured against computations that are not this op.

    ``identity`` is the round trip skipped, ``per_tensor`` is one scale for the whole
    tensor instead of one per row, and ``no_dequant`` drops the ``* scale``. Measured
    on 1024 rows (element ulp / RMS ulp / exact): entry 19.6 / 4.05 / 0.970, identity
    9.8 / 6.53 / 0.075, per_tensor 256.0 / 9.19 / 0.054, no_dequant 7.98e12 / 22037 /
    0.001. The entry sits inside the band and each of the three lands outside it.
    """
    rows = _build(1024, seed=3)
    ref = op.reference(rows)
    scale = ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    per_tensor = ref.float().abs().amax().clamp_min(1e-12) / 448.0
    candidates = {
        "entry": op(rows),
        "identity": rows,
        "per_tensor": (
            (rows.float() / per_tensor).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
            * per_tensor
        ).to(torch.bfloat16),
        "no_dequant": (rows.float() / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .to(torch.bfloat16),
    }
    measured = {name: _separation(value, ref) for name, value in candidates.items()}
    entry_element, entry_rms, entry_exact = measured.pop("entry")
    assert entry_element <= ELEMENT_ULP and entry_rms <= RMS_ULP and entry_exact >= EXACT, (
        f"the entry sits outside its own band at {entry_element:.1f} / {entry_rms:.2f} ulp / "
        f"{entry_exact:.4f} exact"
    )
    for name, (element, rms, exact) in measured.items():
        assert element > ELEMENT_ULP or rms > RMS_ULP or exact < EXACT, (
            f"{name} lands inside the band at {element:.1f} / {rms:.2f} ulp / {exact:.4f} "
            f"exact: the band does not separate it"
        )
