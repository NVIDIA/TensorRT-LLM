# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the hyv4_flash compressed-indexer pool entry `indexer_pool`.

The operands are the checkpoint's: BF16 token keys at the scale `token_key_norm` leaves
(unit RMS), and the four fp32 pooling tensors at the standard deviations measured on
`model.layers.0.self_attn.compressed_indexer` of hf-003107f_orig -- `key_value_projection.weight`
0.022, `key_gate_projection.weight` 0.049, `intra_group_ape` 0.196, `key_norm.weight` 1.055.
Those scales put the four-slot softmax in its working range (logit spread of order 1, so
probabilities away from uniform), which is what makes a pooling cell meaningful.

Each cell is judged against the entry's own fp32 reference twice: at least 99.99% of the
group keys must be bitwise equal, and every element must sit inside the band `_compare`
derives. Beyond the cells: the three `is_valid` refusals, each after the unguarded call is
shown to answer differently from the reference, and the BF16-interior variant the checkpoint
warns about, measured against that band.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.indexer_pool import (  # noqa: E501
    indexer_pool,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "indexer_pool requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the hyv4_flash target's entries are certified on sm_103"
)

DEVICE = "cuda"
HD = 128
KPOOL = 4
#: The checkpoint's `rms_norm_eps` (`HYV4FlashTextConfig`), shared by `key_norm`.
EPS = 1e-5
#: Measured standard deviations of the checkpoint's four pooling tensors (see the module
#: docstring); the token keys are drawn at unit RMS like the normed keys the target passes.
VALUE_STD, GATE_STD, APE_STD, NORM_MEAN, NORM_STD = 0.022, 0.049, 0.196, 1.055, 0.122
#: One BF16 step of any element is at most `2 * ULP[bf16]` of its row's largest magnitude
#: (the step is `2**(e-7)` and the row maximum is at least `2**e`), so one legitimate rounding
#: flip fits in this many units and no more.
ELEMENT_ULP = 2.0
#: Measured relative RMS against the reference: 2.2e-5 (0.006 units) at the 5120-token cell,
#: 2.0e-5 (0.005) at the 64-request one. A quarter of a unit leaves 40x headroom for the
#: accumulation order, and a wrong pooling (uniform mean, the softmax over the wrong axis)
#: lands two orders of magnitude past it.
RMS_ULP = 0.25


def _args(spec: dict, seed: int) -> tuple:
    """`(token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight)`."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    batch, tokens = spec["batch"], spec["tokens"]
    token_keys = torch.randn(batch, tokens, HD, generator=g, device=DEVICE).to(torch.bfloat16)
    value_weight = torch.randn(HD, HD, generator=g, device=DEVICE) * VALUE_STD
    gate_weight = torch.randn(HD, HD, generator=g, device=DEVICE) * GATE_STD
    intra_group_ape = torch.randn(KPOOL, HD, generator=g, device=DEVICE) * APE_STD
    key_norm_weight = NORM_MEAN + torch.randn(HD, generator=g, device=DEVICE) * NORM_STD
    return token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """At least 99.99% of the group keys bitwise equal to the reference, and every element
    inside `ELEMENT_ULP` units of its row's largest magnitude with the relative RMS inside
    `RMS_ULP`.

    Both sides compute the whole interior in fp32 and round to BF16 once, so most elements
    cannot differ at all: measured, 0 differ at the 4- and 7-token cells, 9 of 163840 at the
    5120-token cell. The kernel takes `tl.exp` (the hardware `exp2`) where the reference takes
    torch's softmax, and `tl.rsqrt` where torch takes its own; each is within an fp32 ulp, so
    an element whose fp32 value sits on a BF16 rounding boundary can land one step away.
    Measured with a float64 recompute of the worst element, both sides are one BF16 step from
    the true value, in opposite directions -- the reference is not the more accurate side.

    The elements that move further in BF16 steps are the ones the four-slot weighted sum
    cancels to nearly zero: 36 steps measured at the 64-request cell, on elements whose
    absolute error is far inside the row-scale band below. That is why the step distance is
    reported but not bounded, and why the band is in units of the row's scale rather than of
    each element's own value.

    The band: `assert_within_ulp` with `ELEMENT_ULP = 2` admits one BF16 step of any element
    in the row (see its definition above); measured, 7 of 10485760 elements at the 64-request
    cell exceed one unit and none exceeds two. A pooling that skips the gates or takes the
    softmax over the head axis moves every element far past it.
    """
    assert out.dtype is torch.bfloat16 and ref.dtype is torch.bfloat16
    assert out.shape == ref.shape, (out.shape, ref.shape)
    a, b = out.view(torch.int16), ref.view(torch.int16)
    differs = a != b
    count = int(differs.sum())
    distance = int((a[differs].int() - b[differs].int()).abs().max()) if count else 0
    assert count <= max(16, a.numel() // 10_000), (
        f"{count} of {a.numel()} group keys differ from the reference, by up to {distance} "
        f"BF16 steps; at most one in 10000 may"
    )
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP, ulp=ULP[torch.bfloat16])


@pytest.mark.parametrize("cell", indexer_pool.CELLS, ids=[c.why[:44] for c in indexer_pool.CELLS])
def test_indexer_pool_matches_reference(cell, certifying):  # noqa: F811
    """Every group key inside the measured band of the entry's fp32 reference; see
    `_compare` for the band and its derivation."""
    certifying(indexer_pool)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = indexer_pool(*args, EPS, kpool=KPOOL)
    ref = indexer_pool.reference(*args, EPS, kpool=KPOOL)
    assert out.shape == (cell.spec["batch"], cell.spec["tokens"] // KPOOL, HD)
    _compare(out, ref)


def test_bf16_interior_is_the_variant_the_band_separates():
    """The pooling the checkpoint's `HYV4FlashCompressedIndexer` comment warns about: the
    same formula with its interior in BF16 instead of fp32, which loses the ordering of
    near-ties. Measured against the fp32 reference on the 5120-token cell it lands at
    3.763e-03 relative RMS, 3.9x outside the 9.766e-04 band `_compare` allows the kernel,
    so the band is a real gate on the interior's precision and not just on its shape."""
    spec = dict(batch=1, tokens=5120)
    args = _args(spec, zlib.crc32(b"bf16 interior") % 2**31)
    token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight = args
    ref = indexer_pool.reference(*args, EPS, kpool=KPOOL)
    grouped = token_keys.reshape(1, spec["tokens"] // KPOOL, KPOOL, HD).float()
    values = (grouped @ value_weight.float().t()).to(torch.bfloat16).float()
    gates = (grouped @ gate_weight.float().t()).to(torch.bfloat16).float() + intra_group_ape
    pooled = (values * gates.softmax(dim=2).to(torch.bfloat16).float()).sum(dim=2)
    pooled = pooled.to(torch.bfloat16).float()
    scale = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + EPS)
    bf16_interior = (pooled * scale * key_norm_weight).to(torch.bfloat16)
    rel_rms = (bf16_interior.float() - ref.float()).pow(2).mean().sqrt() / ref.float().pow(
        2
    ).mean().sqrt()
    band = RMS_ULP * ULP[torch.bfloat16]
    print(f"MEASURED bf16 interior rel RMS {float(rel_rms):.3e} against the band {band:.3e}")
    assert float(rel_rms) > band, (
        f"the BF16 interior lands at {float(rel_rms):.3e} relative RMS, inside the "
        f"{band:.3e} the kernel is allowed"
    )


def test_guard_refuses_a_strided_slice_of_token_keys():
    """The kernel walks rows at `head_dim` stride: a decode slice of a wider cache would read
    the cache's next rows, and the callable accepts the shape."""
    args = _args(dict(batch=2, tokens=8), seed=0)
    token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight = args
    g = torch.Generator(device=DEVICE).manual_seed(1)
    wide = torch.randn(2, 12, HD, generator=g, device=DEVICE).to(torch.bfloat16)
    sliced = wide[:, :8]
    assert sliced.shape == token_keys.shape and not sliced.is_contiguous()
    got = indexer_pool(sliced, value_weight, gate_weight, intra_group_ape, key_norm_weight, EPS)
    want = indexer_pool.reference(
        sliced, value_weight, gate_weight, intra_group_ape, key_norm_weight, EPS
    )
    assert not torch.equal(got, want), "the unguarded strided call was supposed to be wrong"
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(indexer_pool):
            indexer_pool(sliced, value_weight, gate_weight, intra_group_ape, key_norm_weight, EPS)


def test_guard_refuses_a_transposed_weight():
    """Both projections are square, so `weight.t()` has the shape the callable accepts; the
    kernel reads it as `[out, in]` and computes `x @ w` instead of `x @ w.T`."""
    args = _args(dict(batch=1, tokens=8), seed=0)
    token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight = args
    transposed = value_weight.t()
    assert transposed.shape == value_weight.shape
    got = indexer_pool(token_keys, transposed, gate_weight, intra_group_ape, key_norm_weight, EPS)
    want = indexer_pool.reference(
        token_keys, transposed, gate_weight, intra_group_ape, key_norm_weight, EPS
    )
    assert not torch.equal(got, want), "the unguarded transposed weight was supposed to be wrong"
    with pytest.raises(AssertionError, match="dense"):
        with validating(indexer_pool):
            indexer_pool(token_keys, transposed, gate_weight, intra_group_ape, key_norm_weight, EPS)


def test_guard_refuses_kpool_past_the_pooling_tile():
    """The tile pools at most 16 members: a `kpool` past it drops the rest of each group
    without a sign, and the shape still comes out `T // kpool` groups wide."""
    args = _args(dict(batch=1, tokens=68), seed=0)
    token_keys, value_weight, gate_weight, _, key_norm_weight = args
    g = torch.Generator(device=DEVICE).manual_seed(2)
    ape = torch.randn(17, HD, generator=g, device=DEVICE) * APE_STD
    got = indexer_pool(token_keys, value_weight, gate_weight, ape, key_norm_weight, EPS, kpool=17)
    want = indexer_pool.reference(
        token_keys, value_weight, gate_weight, ape, key_norm_weight, EPS, kpool=17
    )
    assert got.shape == want.shape == (1, 4, HD)
    assert not torch.equal(got, want), "the unguarded 17-slot pool was supposed to be wrong"
    with pytest.raises(AssertionError, match="16-row pooling tile"):
        with validating(indexer_pool):
            indexer_pool(token_keys, value_weight, gate_weight, ape, key_norm_weight, EPS, kpool=17)
