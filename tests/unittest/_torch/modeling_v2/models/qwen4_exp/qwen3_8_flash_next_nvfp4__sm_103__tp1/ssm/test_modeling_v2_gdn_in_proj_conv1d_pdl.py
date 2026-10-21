# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_in_proj_conv1d_pdl entry.

Cells against the entry's fp32 reference: the output within the band the
decode projections use (4 bf16 ulp element-wise against the row scale, 2 ulp
relative RMS -- the GEMV's summation order is the only difference), the pool
the same, a pad row's slot bitwise untouched. Beyond the cells: two decode
steps under CUDA-graph replay, the second reading the window the first one
wrote; and the row guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_in_proj_conv1d_pdl import (  # noqa: E501
    MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_in_proj_conv1d_pdl import (  # noqa: E501
    gdn_in_proj_conv1d_pdl as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "gdn_in_proj_conv1d_pdl requires a CUDA device"

N, K, CONV_DIM, PAD = 16480, 2560, 10240, -1


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(rows, pool, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(N, K, generator=g, device="cuda") * K**-0.5).to(torch.bfloat16)
    state = torch.randn(pool, CONV_DIM, 3, generator=g, device="cuda").to(torch.bfloat16)
    taps = (torch.randn(CONV_DIM, 4, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    return x, weight.t(), state, taps


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    x, mat_b, state, taps = _build(s["rows"], s["pool"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    slots = torch.tensor(s["slots"], dtype=torch.int32, device="cuda")
    pool = state.clone()
    out = op(x, mat_b, pool, taps, slots, CONV_DIM, PAD, s["pdl"])
    ref_out, ref_state = op.reference(x, mat_b, state, taps, slots, CONV_DIM, PAD, s["pdl"])
    assert out.shape == (s["rows"], N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _compare(out, ref_out)
    _compare(pool, ref_state)
    touched = {v for v in s["slots"] if v != PAD}
    for slot in range(s["pool"]):
        if slot not in touched:
            assert torch.equal(pool[slot], state[slot]), f"slot {slot} was written"


def test_two_steps_under_graph_replay() -> None:
    """Decode runs this under CUDA graphs with PDL, layer after layer: capture one
    step, replay it twice with new activations. The second replay's conv reads the
    window the first replay advanced, so a stale or misplaced window write fails
    the second comparison."""
    pdl = True
    x, mat_b, state, taps = _build(1, 64, seed=3)
    slots = torch.tensor([9], dtype=torch.int32, device="cuda")
    pool = state.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(x, mat_b, pool.clone(), taps, slots, CONV_DIM, PAD, pdl)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, mat_b, pool, taps, slots, CONV_DIM, PAD, pdl)
    want_state = state.clone()
    for trial in range(2):
        g = torch.Generator(device="cuda").manual_seed(50 + trial)
        x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
        graph.replay()
        torch.cuda.synchronize()
        ref_out, want_state = op.reference(x, mat_b, want_state, taps, slots, CONV_DIM, PAD)
        _compare(out, ref_out)
        _compare(pool, want_state)


def test_guards() -> None:
    """More rows than `MAX_ROWS` are refused."""
    x, mat_b, state, taps = _build(MAX_ROWS + 1, 16, seed=21)
    slots = torch.zeros(MAX_ROWS + 1, dtype=torch.int32, device="cuda")
    with validating(op), pytest.raises(AssertionError):
        op(x, mat_b, state, taps, slots, CONV_DIM, PAD)
