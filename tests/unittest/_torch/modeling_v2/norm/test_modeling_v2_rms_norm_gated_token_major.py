# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the norm/rms_norm_gated_token_major catalog entry.

The gate is a column section of a wider projection output, as the KDA layer reads it: ``z[t]`` is token t's
``[heads, N]`` block of the output-gate GEMM, and ``x`` holds the recurrence's per-head rows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 -- needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

# The shapes the shipped target normalizes, by target directory under `models/`: GLM-5.3-Flash's KDA output
# gate at TP4 (16 of 64 heads, head_dim 128, rms_norm_eps 1e-5, sigmoid gate), on the prefill path (decode and
# verify fuse the norm into their kernels).
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4": (
        Cell(
            why="a 13,312-token prompt: 212,992 rows, more programs than SMs (no PDL)",
            spec=dict(tokens=13312, heads=16, width=128, eps=1e-5, gate_activation="sigmoid"),
        ),
        Cell(
            why="a 32,768-token chunked-prefill step",
            spec=dict(tokens=32768, heads=16, width=128, eps=1e-5, gate_activation="sigmoid"),
        ),
        Cell(
            why="a 17-token prompt: fewer programs than SMs, so the PDL launch",
            spec=dict(tokens=17, heads=16, width=128, eps=1e-5, gate_activation="sigmoid"),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec: dict, seed: int):
    """``(x, z, weight, eps, fp8_scale, gate_activation)``: z is the second half of a ``[2, tokens, heads * N]``
    batched GEMM output (the forget gate is the first), so its token stride is ``heads * N`` and it starts
    ``tokens * heads * N`` elements into the buffer."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens, heads, width = spec["tokens"], spec["heads"], spec["width"]
    x = torch.randn(tokens * heads, width, generator=gen, device="cuda").to(torch.bfloat16)
    gates = torch.randn(2, tokens, heads * width, generator=gen, device="cuda").to(torch.bfloat16)
    z = gates[1].view(tokens, heads, width)
    weight = (1 + 0.1 * torch.randn(width, generator=gen, device="cuda")).to(torch.bfloat16)
    fp8_scale = None
    if "fp8_scale" in spec:
        fp8_scale = torch.tensor([spec["fp8_scale"]], dtype=torch.float32, device="cuda")
    return x, z, weight, spec["eps"], fp8_scale, spec["gate_activation"]


def _assert_matches(out: torch.Tensor, ref: torch.Tensor, fp8: bool) -> None:
    """Both sides compute in fp32 and round once, so they differ by the kernel's reduction order and its
    sigmoid / sqrt implementations (relative 1e-6), which can move the rounded output by one ulp of bf16
    (2^-8 to 2^-7 relative): ``rtol`` is two such ulps (2 * bf16 eps), ``atol`` 1e-5 covers outputs near zero.
    With fp8 the same one-ulp bf16 difference before quantization can move an e4m3 code by one step (2^-3
    relative): ``rtol`` 2^-3 plus a bf16 ulp, ``atol`` one e4m3 subnormal step (2^-9) for codes near zero."""
    if fp8:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-3 + 2.0**-8, atol=2.0**-9)
    else:
        torch.testing.assert_close(
            out.float(), ref.float(), rtol=2 * torch.finfo(torch.bfloat16).eps, atol=1e-5
        )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    x, z = args[0].clone(), args[1].clone()
    out = op(*args)
    fp8 = args[4] is not None
    assert out.shape == x.shape and out.is_contiguous()
    assert out.dtype == (torch.float8_e4m3fn if fp8 else x.dtype)
    assert torch.equal(args[0], x) and torch.equal(args[1], z), "the op mutated its inputs"
    _assert_matches(out, op.reference(*args), fp8)


def test_cuda_graph_replay() -> None:
    """Compiled outside capture, the launch replays rewritten inputs bit for bit (the PDL form: 8 tokens)."""
    x, z, weight, eps, _, gate = _build(
        dict(tokens=8, heads=16, width=128, eps=1e-5, gate_activation="sigmoid"), seed=4
    )
    op(x, z, weight, eps, None, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, z, weight, eps, None, gate)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        z.copy_(torch.randn_like(z))
        graph.replay()
        assert torch.equal(out, op(x, z, weight, eps, None, gate))


def test_guard_refuses_a_head_strided_gate() -> None:
    """A gate whose heads are not packed per token (here every head padded to 2N) is read at ``h * N`` anyway:
    the kernel returns a plausible norm gated by the wrong columns."""
    x, _, weight, eps, _, gate = _build(
        dict(tokens=16, heads=6, width=128, eps=1e-5, gate_activation="sigmoid"), seed=5
    )
    z = torch.randn(16, 6, 256, device="cuda").to(torch.bfloat16)[:, :, :128]
    with pytest.raises(AssertionError, match="dense per token"):
        with validating(op):
            op(x, z, weight, eps, None, gate)


def test_guard_refuses_an_unknown_gate() -> None:
    """Any gate name but 'sigmoid' compiles the silu gate: 'relu' would return a silu-gated norm."""
    args = list(
        _build(dict(tokens=4, heads=16, width=128, eps=1e-5, gate_activation="silu"), seed=6)
    )
    args[5] = "relu"
    with pytest.raises(AssertionError, match="gate_activation must be"):
        with validating(op):
            op(*args)


def test_guard_refuses_per_column_fp8_scale() -> None:
    """The kernel reads only the first element of the scale: a per-column scale quantizes every column by it."""
    args = list(
        _build(dict(tokens=8, heads=16, width=128, eps=1e-5, gate_activation="sigmoid"), seed=7)
    )
    args[4] = torch.linspace(0.5, 3.0, 128, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(*args)
