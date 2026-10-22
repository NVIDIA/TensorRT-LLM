# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mtp_norm_shard entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_mtp_norm_shard as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

op = entry.cute_dsl_mtp_norm_shard
HIDDEN, LOCAL = 6144, 3072


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    The same derivation as the trunk's q_a/kv_a norm: one bf16 rounding on
    each side, a few fp32 ulps of difference before it (sum-of-squares order,
    the hardware rsqrt), so at most a one-ulp flip on the rare element near a
    rounding midpoint -- up to 2 units of `ULP[bf16]` against the row scale.
    The wrong half, or a statistic over 3072 columns instead of 6144, misses
    by far more.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def _build(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    source = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    # Make the two halves differ in scale, so a statistic taken over one half
    # alone, or the wrong half written, cannot pass by symmetry.
    source[:, LOCAL:] *= 3
    weight = (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return source, weight


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    rows, offset, eps = cell.spec["rows"], cell.spec["shard_offset"], cell.spec["eps"]
    compiled = entry.compile(offset)
    source, weight = _build(rows, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = source.clone()
    out = torch.full((rows, LOCAL), float("nan"), dtype=torch.bfloat16, device="cuda")
    assert op(source, weight, out, offset, eps, compiled) is out, "the op writes the caller's out"
    assert torch.equal(source, before), "the op must not write its input"
    _compare(out, op.reference(source, weight, out, offset, eps, compiled))


def test_guard_refuses_a_handle_for_the_other_half() -> None:
    """The offset is baked into the handle: the other half is written without error."""
    source, weight = _build(2, seed=1)
    out = torch.empty(2, LOCAL, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="other offset"):
        with validating(op):
            op(source, weight, out, LOCAL, 1e-5, entry.compile(0))


def test_guard_refuses_another_eps() -> None:
    """eps is a kernel constant; another value would be silently ignored."""
    source, weight = _build(2, seed=2)
    out = torch.empty(2, LOCAL, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="would be ignored"):
        with validating(op):
            op(source, weight, out, 0, 1e-6, entry.compile(0))
