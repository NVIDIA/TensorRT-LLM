# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_prefill_conv entry.

Each cell packs ragged sequences into the [T, 4D] fused [q | k | v | g] projection the KDA layer hands the
convolution, against a padded-slot convolution pool; outputs within the engine convolution test's band of an fp32
causal convolution, the updated windows bit for bit, every other slot and the inputs untouched.
"""

import pytest
import torch
from _kda_ops_util import HEAD_DIM, bits, clone_strided, generator, padded_slots, requires_sm103

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

# One bf16 rounding (2**-9) on top of the kernel's fast SiLU, h + h * tanh.approx(h) with h = x / 2, whose relative
# error the kernel bounds by 1.3% wherever it keeps the fast form (h >= -2; groups below are recomputed exactly).
RTOL, ATOL = 1.5e-2, 1e-4


def _inputs(spec, gen):
    heads, lengths = spec["heads"], spec["seq_lens"]
    dim = heads * HEAD_DIM
    total = sum(lengths)
    # Pre-activations of standard deviation ~1.4: both the fast SiLU and its exact recompute (x < -4) run.
    src = torch.randn(total, 4 * dim, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(3 * dim, 4, generator=gen, device="cuda") * 0.7).to(torch.bfloat16)
    pool = padded_slots(spec["pool_slots"], (3 * dim, 3), torch.bfloat16)
    pool.copy_(torch.randn(pool.shape, generator=gen, device="cuda") * 0.5)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    starts = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    n = len(lengths)
    slots = torch.arange(spec["pool_slots"] - 1, spec["pool_slots"] - 1 - n, -1, dtype=torch.int32)
    has_init = torch.tensor(spec["has_init"], dtype=torch.bool, device="cuda")
    return src, weight, pool, starts, slots.cuda(), has_init, max(lengths)


@pytest.mark.parametrize(
    "cell", kda_prefill_conv.CELLS, ids=[c.why[:60] for c in kda_prefill_conv.CELLS]
)
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_prefill_conv)
    src, weight, pool, starts, slots, has_init, max_seqlen = _inputs(cell.spec, generator(cell))
    ref_pool = clone_strided(pool)
    src_before, weight_before = src.clone(), weight.clone()
    outputs = kda_prefill_conv(src, weight, pool, starts, slots, has_init, max_seqlen)
    expected = kda_prefill_conv.reference(
        src, weight, ref_pool, starts, slots, has_init, max_seqlen
    )
    for name, got, ref in zip("qkv", outputs, expected):
        assert got.shape == ref.shape and got.dtype == torch.bfloat16 and got.is_contiguous(), name
        torch.testing.assert_close(
            got, ref, rtol=RTOL, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}"
        )
    assert torch.equal(bits(pool), bits(ref_pool)), (
        "convolution pool differs (selected or other slots)"
    )
    assert torch.equal(bits(src), bits(src_before)) and torch.equal(
        bits(weight), bits(weight_before)
    )


@torch.inference_mode()
def test_guard_refuses_repeated_slots() -> None:
    spec = dict(heads=6, seq_lens=(5, 7), has_init=(False, True), pool_slots=4)
    src, weight, pool, starts, _, has_init, max_seqlen = _inputs(
        spec, torch.Generator("cuda").manual_seed(2)
    )
    slots = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="unique"):
        with validating(kda_prefill_conv):
            kda_prefill_conv(src, weight, pool, starts, slots, has_init, max_seqlen)
