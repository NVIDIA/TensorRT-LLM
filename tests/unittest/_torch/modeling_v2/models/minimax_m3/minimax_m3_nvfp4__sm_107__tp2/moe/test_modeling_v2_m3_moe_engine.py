# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/m3_moe_engine entry: the 1..8-token MoE engine on its caller-owned state.

One Rubin (SM107) rank of the tp2 deployment: 128 experts (all local, routed and shared intermediate 1536 per rank),
top 4, hidden 6144, the shared expert folded in, ``num_ctas`` = SMs - 24 as the target builds it (212 - 24 = 188).
Operands: ``_m3_moe_operands``.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    HIDDEN,
    INTER,
    SHARED_INTER,
    M3MoeState,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    m3_moe_engine as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import (  # noqa: E402
    I_TP,
    SHARED_I,
    H,
    check_full_output,
    check_routed_part,
    hidden_states,
    layer_for,
)
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_moe_engine requires a CUDA device"

_STATES = {}


def test_the_operands_are_the_tp2_engine_geometry() -> None:
    assert (H, I_TP, SHARED_I) == (HIDDEN, INTER, SHARED_INTER) == (6144, 1536, 1536)


def test_the_engine_refuses_the_tp4_width() -> None:
    with pytest.raises(ValueError, match="1536"):
        M3MoeState(torch.device("cuda"), num_tokens=1, inter=768)


def test_the_engine_refuses_more_routed_ctas_than_fc2_blocks() -> None:
    """A routed CTA owning none of the 192 FC2 output blocks would deadlock the hand-off (routed-only, 212 SMs)."""
    with pytest.raises(ValueError, match="192"):
        M3MoeState(torch.device("cuda"), num_tokens=4, num_ctas=193, shared=False)


def _state(m: int) -> M3MoeState:
    if m not in _STATES:
        dev = torch.device("cuda")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        _STATES[m] = M3MoeState(dev, num_tokens=m, num_ctas=sms - 24, shared=True)
    return _STATES[m]


def test_a_module_built_with_other_trace_config_compiles_its_own_kernel(monkeypatch) -> None:
    """The compile cache keys on the config the kernel module was built with: a state whose module carries an extra
    trace-time option (``timing``, which only adds per-CTA stamps) compiles its own kernel, never the plain build's."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_moe import (  # noqa: E501
        op as kop,
    )

    hidden = hidden_states(1, seed=5)
    op(hidden, layer_for(_state(1), "random"))  # the plain build is compiled
    n0 = len(kop._compiled)
    build = kop._kernel_module
    monkeypatch.setattr(
        kop, "_kernel_module", lambda config, path: build({**config, "timing": 1}, path)
    )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    state = M3MoeState(torch.device("cuda"), num_tokens=1, num_ctas=sms - 24, shared=True)
    op(hidden, layer_for(state, "random"))
    assert state.mod.TIMING and not _state(1).mod.TIMING
    assert len(kop._compiled) == n0 + 1, "the timing build reused another build's compiled kernel"


def compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The engine against the plain-torch MoE: BAND_ELEMENT bf16 ulp of the row's largest magnitude per element,
    BAND_RMS ulp relative RMS.

    Both sides quantize the same operands with the same recipes (the activation's and the intermediate's NVFP4 codes
    and block scales, the shared expert's MXFP8); they differ in summation order (tensor-core fp32 against fp64) and in
    the PTX approximations (rcp / ex2 / tanh). Such a difference changes the output only where it moves a value across
    a rounding boundary: one bf16 ulp of a slot output or of the sum, or, rarely, one intermediate NVFP4 code, which
    reaches the output as one weight column's share. Measured over every cell, both routing cases and 3 seeds each
    on the tp4 engine (GB300, intermediate 768): at most 1.48 ulp element, 0.18 ulp RMS; the tp2 engine runs the
    same recipes over twice the intermediate. The band leaves ~3x on both; a misread recipe (the activation's
    1 / (sf x (1 / sc)) spelling of the reciprocal on ~0.06% of its codes) measured 12 ulp element, 7 ulp RMS.
    """
    assert_within_ulp(out, ref, element_ulp=BAND_ELEMENT, rms_ulp=BAND_RMS)


BAND_ELEMENT = 4.0
BAND_RMS = 0.5


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
    # The workspace is tagged per call and never reset: a repeated call is the same call.
    assert torch.equal(out.view(torch.int16), again.view(torch.int16)), (
        "a repeated call changed the output"
    )
    compare(out, op.reference(hidden, layer))


@pytest.mark.parametrize("m", (1, 2, 4, 8))
def test_routed_part_is_trtllm_gen_bit_for_bit(m) -> None:
    check_routed_part(op, _state(m), m, seed=100 + m)


@pytest.mark.parametrize("seed", (1, 2, 3))
@pytest.mark.parametrize("m", (4, 8))
def test_full_output_is_the_stock_block_bit_for_bit(m, seed) -> None:
    """The TP2 operating points (one / two requests verifying 3 drafts): the whole layer output, shared expert
    included, against the stock block (TRTLLM-Gen routed + MXFP8 GatedMLP shared, shared.add_(routed))."""
    check_full_output(op, _state(m), m, seed=200 + 10 * m + seed)


def test_layers_on_one_state_do_not_leak_into_each_other() -> None:
    """Two layers share one state's workspace, in stream order: A, B, A gives A's output twice, bit for bit, and the
    caller's ``out`` receives the same rows as a fresh tensor."""
    state = _state(4)
    a, b = layer_for(state, "random"), layer_for(state, "concentrated")
    hidden = hidden_states(4, seed=7)
    first = op(hidden, a)
    op(hidden, b)
    out = torch.empty(4, H, dtype=torch.bfloat16, device="cuda")
    second = op(hidden, a, out=out)
    assert second.data_ptr() == out.data_ptr()
    assert torch.equal(first.view(torch.int16), out.view(torch.int16))


@pytest.mark.parametrize(
    "bad",
    ("misaligned hidden", "fp32 out", "short out", "long out", "int32 arp"),
)
def test_malformed_buffers_are_rejected_by_the_call(bad) -> None:
    """Why the entry has no is_valid: the compiled call checks each buffer against its signature and raises."""
    state = _state(2)
    layer = layer_for(state, "random")
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
