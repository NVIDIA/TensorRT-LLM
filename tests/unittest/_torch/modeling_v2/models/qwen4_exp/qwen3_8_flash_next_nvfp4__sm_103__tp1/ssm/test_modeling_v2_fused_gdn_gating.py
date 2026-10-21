# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_gdn_gating entry.

Geometry is a Qwen3.8-Flash-Next linear-attention layer: 48 value heads, `a`
bf16, `A_log` / `dt_bias` fp32 drawn across the checkpoint's own ranges.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_gating import (  # noqa: E501
    A_LOG_RANGE,
    APPROX_ULP,
    BA_WIDTH,
    DT_BIAS_RANGE,
    NUM_V_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_gating import (  # noqa: E501
    fused_gdn_gating as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_gdn_gating requires a CUDA device"


#: The softplus `beta` the absolute floor is sized for; the tail error scales as
#: 1 / beta. The target calls 1.0, so 0.5 leaves a factor 2 of headroom.
_MIN_CERTIFIED_BETA = 0.5


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both fp32. `g` multiplies two approximate transcendentals, so the
    relative term is `4 * 2^-22`. The absolute floor is the unstabilised
    softplus's tail error -- about one approximate ulp of the log's
    argument -- scaled by `1 / beta` and by the `exp(A_log)` it multiplies;
    `_compare` cannot see either, so it takes exp(A_log) <= 139.42 (the
    checkpoint's extreme) and beta >= 0.5 (`_MIN_CERTIFIED_BETA`): atol =
    4 * 2^-22 * 139.42 / 0.5 = 2.66e-04.

    Measured against this gate on 256 checkpoint-range rows: worst error /
    allowance 0.215. Five wrong formulas (dt_bias dropped, sign dropped,
    A_log without exp, exp(-A_log), sigmoid for softplus) land 8.7e+05 to
    1.5e+07 allowances away.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    atol = 4.0 * APPROX_ULP * math.exp(A_LOG_RANGE[1]) / _MIN_CERTIFIED_BETA
    torch.testing.assert_close(out, ref, rtol=4.0 * APPROX_ULP, atol=atol)


DEV = "cuda"
SOFTPLUS_THRESHOLD = 20.0


def _params(g, num_heads=NUM_V_HEADS, dtype=torch.float32):
    lo, hi = A_LOG_RANGE
    A_log = torch.rand(num_heads, generator=g, device=DEV) * (hi - lo) + lo
    lo, hi = DT_BIAS_RANGE
    dt_bias = torch.rand(num_heads, generator=g, device=DEV) * (hi - lo) + lo
    return A_log.to(dtype), dt_bias.to(dtype)


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, heads = spec["rows"], NUM_V_HEADS

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(torch.bfloat16)

    if "fill" in spec:
        a = torch.full((rows, heads), spec["fill"], dtype=torch.bfloat16, device=DEV)
    elif spec.get("layout", "contiguous") == "target_rows":
        # `_gdn_block` splits the [T, 48] projection into prefill and decode rows.
        a = randn(rows + 23, heads)[23:]
        assert a.is_contiguous() and a.storage_offset() == 23 * heads
    else:
        a = randn(rows, heads)
    A_log, dt_bias = _params(g, heads)
    return A_log, a, dt_bias, 1.0, SOFTPLUS_THRESHOLD


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    g = op(*args)
    _compare(g, op.reference(*args))
    assert g.is_contiguous() and torch.isfinite(g).all()


def test_guard_refuses_what_the_kernel_reads_at_the_wrong_addresses() -> None:
    """Unguarded, an `a` with head stride 2 returns a finite `g` -- exactly the
    one for the memory the kernel actually addressed (`row * stride(0) +
    head`: the first 48 columns of each row), not the tensor it was handed;
    the guard refuses it."""
    g = torch.Generator(device=DEV).manual_seed(111)
    A_log, dt_bias = _params(g)
    wide = torch.randn(32, BA_WIDTH, generator=g, device=DEV).to(torch.bfloat16)
    bad = (A_log, wide[:, ::2], dt_bias)
    addressed = (A_log, wide[:, :NUM_V_HEADS], dt_bias)
    unguarded = op(*bad)
    assert torch.isfinite(unguarded).all()
    assert torch.equal(unguarded, op(*addressed)), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(*bad))
    with pytest.raises(AssertionError, match="head axis must be unit-stride"):
        with validating(op):
            op(*bad)
