# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_draft_glue_rmsnorm_cat entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.spec.draft_glue_rmsnorm_cat import (  # noqa: E501
    glm_draft_glue_rmsnorm_cat as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)

HIDDEN = 6144
EPS = 1e-5


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One RMSNorm's worth of rounding differences.

    Kernel and reference normalize the same BF16 rows and round once to BF16. They differ in the order of the 6144
    squares in the FP32 sum and in the reciprocal square root (the kernel's approximate `rsqrt`, a few FP32 ulp, the
    reference's correctly rounded one): together a few 1e-7 relative on the scale factor, which moves the BF16
    rounding of an output only when it lies that close to a rounding midpoint, by one ulp of that element.

    One BF16 ulp of an element in the row maximum's binade is up to twice the band's unit (2**-8 of the row
    maximum). So: 2 ulp element-wise against the row scale, 0.25 ulp of relative RMS. A wrong row, column or weight
    is off by the size of the values themselves.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, vocab = spec["rows"], spec["vocab"]
    table = torch.randn(vocab, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    hidden = 3.0 * torch.randn(rows, HIDDEN, generator=g, device="cuda")
    if spec["ids"] == "valid":
        ids = torch.randint(0, vocab, (rows,), generator=g, device="cuda", dtype=torch.int32)
    else:
        # int64 ids: -1 padding, the first id past the table, valid ids, a large negative and a far one.
        candidates = [-1, vocab, 5, -7, vocab + 100, 0, vocab - 1, -1]
        ids = torch.tensor(candidates[:rows], dtype=torch.int64, device="cuda")
    enorm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    hnorm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    lo, hi = spec["window"]
    return table, ids, hidden.to(torch.bfloat16), enorm, hnorm, EPS, lo, hi


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    out = op(*args, True)
    assert out.is_contiguous() and out.shape == (cell.spec["rows"], args[7] - args[6])
    _compare(out, op.reference(*args, True))


def test_window_is_flashinfer_rmsnorm_bitwise() -> None:
    """Where FlashInfer runs its CuTe RMSNorm, the window is that kernel's output on each half, bit for bit."""
    import flashinfer.norm

    device = torch.device("cuda")
    if flashinfer.norm._use_cuda_norm(device):
        pytest.skip("flashinfer.norm.rmsnorm runs its CUDA JIT kernel on this device")
    for rank in (0, 15):
        table, ids, hidden, enorm, hnorm, eps, lo, hi = _build(
            dict(rows=8, vocab=4096, window=(768 * rank, 768 * (rank + 1)), ids="valid"), seed=rank
        )
        embeds = torch.nn.functional.embedding(ids.long(), table)
        enormed = flashinfer.norm.rmsnorm(embeds, enorm, eps)
        hnormed = flashinfer.norm.rmsnorm(hidden, hnorm, eps)
        unfused = torch.cat([enormed, hnormed], dim=-1)
        out = op(table, ids, hidden, enorm, hnorm, eps, lo, hi, True)
        torch.testing.assert_close(out, unfused[:, lo:hi], rtol=0, atol=0)


def test_guard_refuses_a_window_inside_a_vector() -> None:
    """The store predicate tests each 8-column vector's first column only."""
    table, ids, hidden, enorm, hnorm, eps, _, _ = _build(op.CELLS[0].spec, seed=1)
    with pytest.raises(AssertionError, match="8-column aligned"):
        with validating(op):
            op(table, ids, hidden, enorm, hnorm, eps, 4, 772, True)


def test_guard_refuses_a_misaligned_hidden_row() -> None:
    """The kernel is compiled for 16-byte aligned rows."""
    table, ids, hidden, enorm, hnorm, eps, lo, hi = _build(op.CELLS[2].spec, seed=2)
    storage = torch.zeros(8, HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    shifted = storage[:, 1 : HIDDEN + 1]
    shifted.copy_(hidden)
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(op):
            op(table, ids, shifted, enorm, hnorm, eps, lo, hi, True)
