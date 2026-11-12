# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_w4a16_gemv_conv entry.

Cells against the reference (the GEMV reference, then the conv in fp32 with torch's
SiLU) in the band derived in `_compare`, on a conv pool with padding between slots
(a strided view, as the cache manager hands it over). Beyond the cells, the
exactness claim: bitwise the composition `nvfp4_w4a16_gemv` -> the in-tree
`causal_conv1d_update`, output and whole pool (padding bytes and untouched slots
included), over several stateful calls; and the slot guards.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_w4a16_gemv import (  # noqa: E501
    nvfp4_w4a16_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_w4a16_gemv_conv import (  # noqa: E501
    CONV_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_w4a16_gemv_conv import (  # noqa: E501
    nvfp4_w4a16_gemv_conv as op,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
N, K = 16384, 5120
SLOTS = 6
PAD = -1
#: The pool's slot stride: CONV_DIM * 3 plus padding, as a strided pool view has.
SLOT_STRIDE = CONV_DIM * 3 + 64


def _operands(seed: int):
    """Random NVFP4 weight bytes with normal block scales, global scale 1/128 (a power
    of two, as the decode recipe produces), conv weights, a padded pool."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    weight = torch.randint(0, 256, (N, K // 2), generator=g, device=DEV, dtype=torch.int32).to(
        torch.uint8
    )
    weight_sf = torch.randint(0x28, 0x40, (N, K // 16), generator=g, device=DEV, dtype=torch.int32)
    weight_sf = weight_sf.to(torch.uint8)
    conv_weight = (torch.randn(CONV_DIM, 4, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    raw = torch.randn(SLOTS, SLOT_STRIDE, generator=g, device=DEV).to(torch.bfloat16)
    return weight, weight_sf, 1.0 / 128.0, conv_weight, raw


def _pool(raw: torch.Tensor) -> torch.Tensor:
    pool = raw[:, : CONV_DIM * 3].view(SLOTS, CONV_DIM, 3)
    assert pool.stride() == (SLOT_STRIDE, 3, 1)
    return pool


def _x(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(m, K, generator=g, device=DEV) * 0.5).to(torch.bfloat16)


def _compare(out, ref, pool_after, ref_pool, raw_before, raw_after, slots) -> None:
    """The projection is the GEMV's (4 units element-wise, 2 of RMS: accumulation order
    only). The conv columns take that BF16 projection through four fp32 taps and SiLU:
    a one-ulp projection flip moves them by |w3| (<= 2.5 here) ulps of the projection,
    and `__expf` against torch's exp by about one fp32 ulp -- inside the same band
    measured against the row maximum. The state copies values: taps 0-1 are the old
    state shifted (bitwise), tap 2 is the BF16 projection (within the GEMV band).
    Every byte the call must not touch is bitwise unchanged."""
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)
    live = [s for s in slots if s != PAD]
    for s in live:
        assert torch.equal(pool_after[s, :, :2], ref_pool[s, :, :2]), f"slot {s} shifted taps"
        assert_within_ulp(
            pool_after[s, :, 2].reshape(1, -1), ref_pool[s, :, 2].reshape(1, -1), 4.0, 2.0
        )
    for s in range(SLOTS):
        if s not in live:
            assert torch.equal(raw_after[s], raw_before[s]), f"untouched slot {s} changed"
    assert torch.equal(raw_after[:, CONV_DIM * 3 :], raw_before[:, CONV_DIM * 3 :]), "pool padding"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    weight, weight_sf, gs, conv_weight, raw = _operands(seed_of(cell.why))
    x = _x(s["m"], seed_of(cell.why) + 1)
    slots = list(s["slots"])
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    raw_before = raw.clone()
    pool = _pool(raw)
    ref_out, ref_pool = op.reference(
        x, weight, weight_sf, gs, pool.clone(), conv_weight, idx, CONV_DIM, PAD
    )
    out = op(x, weight, weight_sf, gs, pool, conv_weight, idx, CONV_DIM, PAD)
    assert out.shape == (s["m"], N)
    _compare(out, ref_out, pool, ref_pool, raw_before, raw, slots)


@pytest.mark.parametrize("slots", [[3], [4, 1], [1, PAD, 3]], ids=["m1", "m2", "m3-pad"])
def test_bitwise_the_gemv_then_conv_update(slots) -> None:
    """Ten stateful calls; after each, output and the whole raw pool (padding and
    untouched slots included) bitwise equal to the two-op path on its own pool."""
    weight, weight_sf, gs, conv_weight, raw = _operands(seed=len(slots))
    raw_two = raw.clone()
    pool, pool_two = _pool(raw), _pool(raw_two)
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    for step in range(10):
        x = _x(len(slots), seed=100 + step)
        proj = nvfp4_w4a16_gemv(x, weight, weight_sf, gs)
        mixed = causal_conv1d_update(
            proj[:, :CONV_DIM],
            pool_two,
            conv_weight,
            None,
            "silu",
            conv_state_indices=idx,
            pad_slot_id=PAD,
        )
        got = op(x, weight, weight_sf, gs, pool, conv_weight, idx, CONV_DIM, PAD)
        assert torch.equal(got[:, CONV_DIM:], proj[:, CONV_DIM:]), step
        assert torch.equal(got[:, :CONV_DIM], mixed), step
        assert torch.equal(raw, raw_two), step


def test_guard_refuses_two_live_rows_on_one_slot() -> None:
    weight, weight_sf, gs, conv_weight, raw = _operands(seed=7)
    idx = torch.tensor([2, 2], dtype=torch.int32, device=DEV)
    with validating(op), pytest.raises(AssertionError, match="share a conv-state slot"):
        op(_x(2, 8), weight, weight_sf, gs, _pool(raw), conv_weight, idx, CONV_DIM, PAD)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    weight, weight_sf, gs, conv_weight, raw = _operands(seed=9)
    idx = torch.tensor([SLOTS], dtype=torch.int32, device=DEV)
    with validating(op), pytest.raises(AssertionError, match="outside the pool"):
        op(_x(1, 10), weight, weight_sf, gs, _pool(raw), conv_weight, idx, CONV_DIM, PAD)
