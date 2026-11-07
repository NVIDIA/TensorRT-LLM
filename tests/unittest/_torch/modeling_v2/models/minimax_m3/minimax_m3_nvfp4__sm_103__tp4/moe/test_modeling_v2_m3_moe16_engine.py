# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/m3_moe16_engine entry: the 9..13-row MoE engine on its caller-owned state.

One GB300 rank of the tp4 deployment, the target's build (128 routed CTAs + 24 shared-expert CTAs). Operands and the
band: ``_m3_moe_operands`` and ``test_modeling_v2_m3_moe_engine`` (the same math).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.m3_moe.op16 import (  # noqa: E501
    M16_NUM_CTAS,
    SHARED_CTAS,
    M3Moe16State,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_moe16_engine import (  # noqa: E501
    m3_moe16_engine as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import H, check_routed_part, hidden_states, layer_for  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_moe16_engine requires a CUDA device"

# As test_modeling_v2_m3_moe_engine.compare (same arithmetic, same quantization boundaries).
BAND_ELEMENT = 4.0
BAND_RMS = 0.5

_STATES = {}


def _state(m: int) -> M3Moe16State:
    sms = torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    if sms < M16_NUM_CTAS + SHARED_CTAS:
        pytest.skip(f"m3_moe16 needs {M16_NUM_CTAS + SHARED_CTAS} resident SMs; device has {sms}")
    if m not in _STATES:
        _STATES[m] = M3Moe16State(torch.device("cuda"), num_tokens=m)
    return _STATES[m]


@pytest.mark.parametrize("case", ("random", "concentrated"))
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, case) -> None:
    m = cell.spec["tokens"]
    layer = layer_for(_state(m), case)
    hidden = hidden_states(m, seed=zlib.crc32((cell.why + case).encode()) % 2**31)
    with validating(op):
        out = op(hidden, layer)
        again = op(hidden, layer)
    assert out.shape == (m, H) and out.dtype is torch.bfloat16
    assert torch.equal(out.view(torch.int16), again.view(torch.int16)), (
        "a repeated call changed the output"
    )
    assert_within_ulp(out, op.reference(hidden, layer), element_ulp=BAND_ELEMENT, rms_ulp=BAND_RMS)


@pytest.mark.parametrize("m", (9, 13))
def test_routed_part_is_trtllm_gen_bit_for_bit(m) -> None:
    check_routed_part(op, _state(m), m, seed=100 + m)


def test_insufficient_sm_budget_is_rejected() -> None:
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(device).multi_processor_count >= M16_NUM_CTAS + SHARED_CTAS:
        pytest.skip("device has enough SMs for the MoE16 schedule")
    with pytest.raises(ValueError, match="routed CTAs"):
        M3Moe16State(device, num_tokens=9)
