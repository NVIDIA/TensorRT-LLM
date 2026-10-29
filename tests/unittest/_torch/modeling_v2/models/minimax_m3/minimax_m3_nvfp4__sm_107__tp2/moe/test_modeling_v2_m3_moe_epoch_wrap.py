# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the m3_moe engine's per-call epoch wraps past 2^31 without changing a bit.

The engine tags its hand-off records with the call epoch (masked low bits plus a marker bit) and counts the B2IMG group
counters up by CNT_PER_CALL per call, waiting for ``(epoch + 1) * CNT_PER_CALL`` in wrapping int32 arithmetic. A state
seeded at epoch 0x7FFFFFF0 (counters seeded consistently, as after that many calls) runs 32 calls across the int32
wrap; every output equals a fresh state's, bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    M3MoeState,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    m3_moe_engine as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

assert torch.cuda.is_available(), "m3_moe_engine requires a CUDA device"

EPOCH0 = 0x7FFFFFF0
CALLS = 32


def _int32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _seed(state: M3MoeState, epoch: int) -> None:
    """The state as after ``epoch`` calls: every CTA's epoch, and each B2IMG group counter at epoch x CNT_PER_CALL."""
    state.epochs.fill_(_int32(epoch))
    mod = state.mod
    if mod.B2IMG:
        value = _int32(epoch * mod.CNT_PER_CALL)
        for g in range(mod.NG_CAP):
            state.cnt[8 * g] = value
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [1, 4, 8])
def test_the_epoch_wraps_bit_for_bit(m) -> None:
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    fresh = M3MoeState(dev, num_tokens=m, num_ctas=sms - 24, shared=True)
    wrapped = M3MoeState(dev, num_tokens=m, num_ctas=sms - 24, shared=True)
    _seed(wrapped, EPOCH0)
    lf, lw = layer_for(fresh, "random"), layer_for(wrapped, "random")
    for i in range(CALLS):
        x = hidden_states(m, 4000 + i)
        a, b = op(x, lf), op(x, lw)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"call {i} (epoch {EPOCH0 + i:#x})"
    assert int(wrapped.epochs[0]) == _int32(EPOCH0 + CALLS) < 0  # crossed 2^31
