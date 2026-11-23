# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/m3_moe_engine entry (the 1..8-token MoE engine) on each MiniMax-M3.1-Flash sm_107 TP target's
caller-owned state, and for the moe/m3_moe16_engine entry (the 9..13-row engine: 128 routed CTAs + 24 shared-expert
CTAs) on the tp4 target's.

One rank of each deployment: 128 experts (all local; intermediate 1536 per rank on tp2, 768 on tp4, 384 on tp8, the
shared expert's the same), top 4, hidden 6144, the shared expert folded in, ``num_ctas`` = SMs - 24 as the targets
build it. The certified cells are the entry's ``CELLS_BY_TP[tp]``. Operands: ``_m31_flash_moe_operands``.
"""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m3_moe16_engine import (
    m3_moe16_engine as op16,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m3_moe_engine import (
    m3_moe_engine as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m31_flash_moe_operands import H, check_routed_part, hidden_states, layer_for  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.minimax_m3"
# Target -> its TP size (the engine's CELLS_BY_TP key).
TARGETS = {
    "minimax_m3_1_flash_nvfp4__sm_107__tp2": 2,
    "minimax_m3_1_flash_nvfp4__sm_107__tp4": 4,
    "minimax_m3_1_flash_nvfp4__sm_107__tp8": 8,
}
_TP4 = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
# The malformed buffers each target's compiled call is held to refuse.
_MALFORMED = {
    "minimax_m3_1_flash_nvfp4__sm_107__tp2": (
        "misaligned hidden", "fp32 out", "short out", "long out", "int32 arp"),
    "minimax_m3_1_flash_nvfp4__sm_107__tp4": ("misaligned hidden", "short out", "int32 arp"),
    "minimax_m3_1_flash_nvfp4__sm_107__tp8": (
        "misaligned hidden", "fp32 out", "short out", "long out", "int32 arp"),
}  # fmt: skip

BAND_ELEMENT = 4.0
BAND_RMS = 0.5


def _target_module(target: str, name: str):
    try:
        return importlib.import_module(f"{_PACKAGE}.{target}.kernels.m3_moe.{name}")
    except NoTarget as exc:
        pytest.skip(str(exc))


_STATES = {}


def _state(target: str, m: int):
    """The target's 1..8-token engine state for ``m`` tokens, and its expert width per rank."""
    module = _target_module(target, "op")
    if (target, m) not in _STATES:
        dev = torch.device("cuda")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        _STATES[target, m] = module.M3MoeState(dev, num_tokens=m, num_ctas=sms - 24, shared=True)
    return _STATES[target, m], module.INTER


def _state16(m: int):
    """The tp4 target's 9..13-row engine state for ``m`` tokens, and its expert width per rank."""
    module = _target_module(_TP4, "op16")
    if (_TP4, "16", m) not in _STATES:
        _STATES[_TP4, "16", m] = module.M3Moe16State(torch.device("cuda"), num_tokens=m)
    return _STATES[_TP4, "16", m], _target_module(_TP4, "op").INTER


def compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The engine against the plain-torch MoE: BAND_ELEMENT bf16 ulp of the row's largest magnitude per element,
    BAND_RMS ulp relative RMS.

    Both sides quantize the same operands with the same recipes (the activation's and the intermediate's NVFP4 codes
    and block scales, the shared expert's MXFP8); they differ in summation order (tensor-core fp32 against fp64) and in
    the PTX approximations (rcp / ex2 / tanh). Such a difference changes the output only where it moves a value across
    a rounding boundary: one bf16 ulp of a slot output or of the sum, or, rarely, one intermediate NVFP4 code, which
    reaches the output as one weight column's share. Measured over every cell, both routing cases and 3 seeds each on
    the tp4 engine (GB300, intermediate 768): at most 1.48 ulp element, 0.18 ulp RMS; the tp2 and tp8 engines run the
    same recipes over other widths. The band leaves ~3x on both; a misread recipe (the activation's
    1 / (sf x (1 / sc)) spelling of the reciprocal on ~0.06% of its codes) measured 12 ulp element, 7 ulp RMS.
    """
    assert_within_ulp(out, ref, element_ulp=BAND_ELEMENT, rms_ulp=BAND_RMS)


def _check_cell(engine, state, inter: int, cell, case: str) -> None:
    m = cell.spec["tokens"]
    layer = layer_for(state, inter, case)
    hidden = hidden_states(m, seed=zlib.crc32((cell.why + case).encode()) % 2**31)
    with validating(engine):
        out = engine(hidden, layer)
        again = engine(hidden, layer)
    assert out.shape == (m, H) and out.dtype is torch.bfloat16
    # The workspace is tagged per call and never reset: a repeated call is the same call.
    assert torch.equal(out.view(torch.int16), again.view(torch.int16)), (
        "a repeated call changed the output"
    )
    compare(out, engine.reference(hidden, layer))


_CELL_CASES = [
    pytest.param(target, cell, id=f"{target.rsplit('__', 1)[1]}-{cell.why[:44]}")
    for target, tp in TARGETS.items()
    for cell in op.CELLS_BY_TP[tp]
]


@pytest.mark.parametrize("case", ("random", "concentrated"))
@pytest.mark.parametrize("target,cell", _CELL_CASES)
def test_certified_cells(target, cell, case) -> None:
    state, inter = _state(target, cell.spec["tokens"])
    _check_cell(op, state, inter, cell, case)


@pytest.mark.parametrize("m", (1, 2, 8))
@pytest.mark.parametrize("target", TARGETS)
def test_routed_part_is_trtllm_gen_bit_for_bit(target, m) -> None:
    state, inter = _state(target, m)
    check_routed_part(op, state, inter, m, seed=100 + m)


@pytest.mark.parametrize("target", TARGETS)
def test_layers_on_one_state_do_not_leak_into_each_other(target) -> None:
    """Two layers share one state's workspace, in stream order: A, B, A gives A's output twice, bit for bit, and the
    caller's ``out`` receives the same rows as a fresh tensor."""
    state, inter = _state(target, 4)
    a, b = layer_for(state, inter, "random"), layer_for(state, inter, "concentrated")
    hidden = hidden_states(4, seed=7)
    first = op(hidden, a)
    op(hidden, b)
    out = torch.empty(4, H, dtype=torch.bfloat16, device="cuda")
    second = op(hidden, a, out=out)
    assert second.data_ptr() == out.data_ptr()
    assert torch.equal(first.view(torch.int16), out.view(torch.int16))


@pytest.mark.parametrize(
    "target,bad", [(target, bad) for target, bads in _MALFORMED.items() for bad in bads]
)
def test_malformed_buffers_are_rejected_by_the_call(target, bad) -> None:
    """Why the entry has no is_valid: the compiled call checks each buffer against its signature and raises."""
    state, inter = _state(target, 2)
    layer = layer_for(state, inter, "random")
    hidden, kw = hidden_states(2, seed=3), {}
    if bad == "misaligned hidden":
        hidden = torch.randn(2 * H + 8, device="cuda").to(torch.bfloat16)[2 : 2 + 2 * H].view(2, H)
    elif bad == "fp32 out":
        kw["out"] = torch.empty(2, H, dtype=torch.float32, device="cuda")
    elif bad == "short out":
        kw["out"] = torch.empty(1, H, dtype=torch.bfloat16, device="cuda")
    elif bad == "long out":
        kw["out"] = torch.empty(3, H, dtype=torch.bfloat16, device="cuda")
    else:
        kw["arp"] = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        op(hidden, layer, **kw)


@pytest.mark.parametrize("case", ("random", "concentrated"))
@pytest.mark.parametrize("cell", op16.CELLS, ids=[c.why[:44] for c in op16.CELLS])
def test_moe16_certified_cells(cell, case) -> None:
    """The 9..13-row engine: the same arithmetic and quantization boundaries, so the same band."""
    state, inter = _state16(cell.spec["tokens"])
    _check_cell(op16, state, inter, cell, case)


@pytest.mark.parametrize("m", (9, 13))
def test_moe16_routed_part_is_trtllm_gen_bit_for_bit(m) -> None:
    state, inter = _state16(m)
    check_routed_part(op16, state, inter, m, seed=100 + m)
