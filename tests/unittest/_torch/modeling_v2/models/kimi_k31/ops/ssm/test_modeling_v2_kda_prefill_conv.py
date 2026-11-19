# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_prefill_conv entry.

Each cell packs ragged sequences into the [T, 4D] fused [q | k | v | g] projection the KDA layer hands the
convolution, against a padded-slot convolution pool; outputs within the engine convolution test's band of an fp32
causal convolution, the updated windows bit for bit, every other slot untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)

from ._kda_ops_util import HEAD_DIM, bits, clone_strided, generator, padded_slots, requires_sm103

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

#: A boundary and a typical cell of ``kda_prefill_conv.CELLS``.
_CELLS = sample_cells(kda_prefill_conv.CELLS, size_key="pool_slots")[:2]


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


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_prefill_conv)
    src, weight, pool, starts, slots, has_init, max_seqlen = _inputs(cell.spec, generator(cell))
    ref_pool = clone_strided(pool)
    outputs = kda_prefill_conv(src, weight, pool, starts, slots, has_init, max_seqlen)
    expected = kda_prefill_conv.reference(
        src, weight, ref_pool, starts, slots, has_init, max_seqlen
    )
    for got, ref in zip(outputs, expected):
        torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)
    assert torch.equal(bits(pool), bits(ref_pool)), "convolution pool differs"
