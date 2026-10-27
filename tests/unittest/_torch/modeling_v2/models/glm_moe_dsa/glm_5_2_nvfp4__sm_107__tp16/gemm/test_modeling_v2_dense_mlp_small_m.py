# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 tp16 dense_mlp_small_m entry (sm_107).

The cells gate the kernels against the entry's reference within a derived band.
The rest is what a band cannot show: a row's result does not depend on the
other rows, repeats are bitwise equal, a call chained right behind another one
reads its input only after the first call wrote it (programmatic dependent
launch), CUDA-graph replay, and the shape domain the op serves and refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.dense_mlp_small_m import (  # noqa: E501
    HIDDEN,
    INTER_PER_RANK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.dense_mlp_small_m import (  # noqa: E501
    dense_mlp_small_m as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "dense_mlp_small_m requires a CUDA device"

#: The c1 cell: the shape and scales every non-cell test below reuses.
C1 = op.CELLS[0]


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """x [rows, 6144], gate_up_weight [2 * 768, 6144] and down_weight [6144, 768], bf16.

    Normal draws at the cell's scales; x is a unit-RMS row times the norm's
    gamma, as post_attention_layernorm hands it over.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["rows"], HIDDEN, generator=g, device="cuda") * spec["x_scale"]
    gate_up = torch.randn(2 * INTER_PER_RANK, HIDDEN, generator=g, device="cuda")
    down = torch.randn(HIDDEN, INTER_PER_RANK, generator=g, device="cuda")
    return (
        x.to(torch.bfloat16),
        (gate_up * spec["gate_up_std"]).to(torch.bfloat16),
        (down * spec["down_std"]).to(torch.bfloat16),
    )


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Summation order and the fast sigmoid, nothing else.

    Kernel and reference take the same bf16 operands and round at the same
    three points (gate and up sums, h, out). They differ in the fp32 order of
    the 6144- and 768-long sums (about 1e-6 relative) and in the kernel's fast
    exp and division inside the sigmoid (a few fp32 ulp; tens at the saturated
    cell's |gate| of 40). Either can carry a value across a bf16 rounding
    boundary. An output that flips moves by one bf16 ulp of itself: at most
    2^-7 of its row's largest magnitude, 2 units of the 2^-8 band unit. An h
    that flips moves each output by 2^-7 |h_k w_nk|, a few hundredths of a
    unit. Flips are rare (the gaps above are 1e-3 of a bf16 ulp), so the
    relative RMS sits well under one unit.

    So: 4 units per element against the row scale, 2 units of relative RMS --
    twice the single-flip bound, and three orders below a wrong product (a
    swapped gate/up, a lost row, a missing SwiGLU), which misses by O(1).
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, gate_up_weight, down_weight = _build(cell.spec, seed=_seed(cell.why))
    before = x.clone()
    out = op(x, gate_up_weight, down_weight)
    assert out.shape == (x.shape[0], HIDDEN) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    assert torch.equal(x, before), "the op must not modify x"
    _compare(out, op.reference(x, gate_up_weight, down_weight))


def test_a_row_does_not_see_the_other_rows() -> None:
    """Row m's result is bitwise the same with 1 to 8 rows present.

    The empty token columns of a short call read as zero and are never
    stored, and the arithmetic of a row does not depend on the other rows.
    """
    x, gate_up_weight, down_weight = _build(C1.spec, seed=_seed("rows"))
    full = op(x, gate_up_weight, down_weight)
    for rows in range(1, 8):
        part = op(x[:rows].contiguous(), gate_up_weight, down_weight)
        assert torch.equal(part, full[:rows]), f"{rows} rows differ from the 8-row call"


def test_repeats_are_bitwise_equal() -> None:
    """Fixed summation orders inside a CTA, across warps and across the cluster."""
    x, gate_up_weight, down_weight = _build(C1.spec, seed=_seed("repeat"))
    first = op(x, gate_up_weight, down_weight)
    for _ in range(4):
        assert torch.equal(op(x, gate_up_weight, down_weight), first)


def _chain(x, layer_a, layer_b):
    out_a = op(x, *layer_a)
    return out_a, op(out_a, *layer_b)


def test_a_chained_call_reads_its_input_after_the_wait() -> None:
    """The second of two back-to-back calls reads what the first one wrote.

    With PDL the down kernel of the first call releases the gate/up kernel of
    the second before it computes its output; that kernel stages its weights
    early and must read x only after its grid-dependency wait. Every iteration
    draws a new x, so a read of the previous iteration's output (often at the
    same address) or of an unwritten buffer shows up against a standalone
    recompute of the second call after a device sync. Eager and graph-replayed.
    """
    x, *layer_a = _build(C1.spec, seed=_seed("chain a"))
    _, *layer_b = _build(C1.spec, seed=_seed("chain b"))
    g = torch.Generator(device="cuda").manual_seed(_seed("chain x"))

    for _ in range(8):
        x.copy_(torch.randn(x.shape, generator=g, device="cuda") * C1.spec["x_scale"])
        out_a, out_b = _chain(x, layer_a, layer_b)
        torch.cuda.synchronize()
        assert torch.equal(out_b, op(out_a.clone(), *layer_b)), "eager chain read a stale input"

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _chain(x, layer_a, layer_b)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_a, out_b = _chain(x, layer_a, layer_b)
    for _ in range(8):
        x.copy_(torch.randn(x.shape, generator=g, device="cuda") * C1.spec["x_scale"])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_b, op(out_a.clone(), *layer_b)), "replayed chain read a stale input"


def test_cuda_graph_capture_and_replay() -> None:
    """The c1 forward runs inside one CUDA graph: replay with new x contents
    gives the eager result bitwise, and the reference within the band."""
    x, gate_up_weight, down_weight = _build(C1.spec, seed=_seed("graph"))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(x, gate_up_weight, down_weight)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, gate_up_weight, down_weight)

    fresh, _, _ = _build(C1.spec, seed=_seed("graph replay"))
    x.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(x, gate_up_weight, down_weight))
    _compare(captured, op.reference(x, gate_up_weight, down_weight))


@pytest.mark.parametrize(
    "rows, hidden, inter, served",
    [
        (1, HIDDEN, INTER_PER_RANK, True),
        (8, HIDDEN, INTER_PER_RANK, True),
        (0, HIDDEN, INTER_PER_RANK, False),
        (9, HIDDEN, INTER_PER_RANK, False),
        (8, HIDDEN, 2 * INTER_PER_RANK, False),
        (8, HIDDEN, 4 * INTER_PER_RANK, False),
        (8, HIDDEN, 16 * INTER_PER_RANK, False),
        (8, 7168, INTER_PER_RANK, False),
    ],
    ids=["1 row", "8 rows", "0 rows", "9 rows", "tp8", "tp4", "tp1", "hidden 7168"],
)
def test_the_supported_predicate(rows, hidden, inter, served) -> None:
    """The host-side answer the caller routes on: 1-8 rows of the TP16 slice
    only. TP4 is what a dense MLP capped at the node's 4 GPUs would hand over."""
    assert torch.ops.trtllm.glm_dense_mlp_small_m_supported(rows, hidden, inter) is served


def _strided_x():
    x, w13, w2 = _build(C1.spec, seed=_seed("strided"))
    wide = torch.zeros(x.shape[0], 2 * HIDDEN, dtype=x.dtype, device=x.device)
    wide[:, :HIDDEN] = x
    return wide[:, :HIDDEN], w13, w2


def _misaligned_x():
    x, w13, w2 = _build(C1.spec, seed=_seed("misaligned"))
    flat = torch.zeros(x.numel() + 1, dtype=x.dtype, device=x.device)
    shifted = flat[1:].view(x.shape)
    shifted.copy_(x)
    return shifted, w13, w2


def _too_many_rows():
    x, w13, w2 = _build(dict(C1.spec, rows=9), seed=_seed("9 rows"))
    return x, w13, w2


def _tp8_slice():
    x, _, _ = _build(C1.spec, seed=_seed("tp8"))
    w13 = torch.zeros(4 * INTER_PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w2 = torch.zeros(HIDDEN, 2 * INTER_PER_RANK, dtype=torch.bfloat16, device="cuda")
    return x, w13, w2


def _fp16_x():
    x, w13, w2 = _build(C1.spec, seed=_seed("fp16"))
    return x.to(torch.float16), w13, w2


def _mismatched_down():
    x, w13, w2 = _build(C1.spec, seed=_seed("mismatch"))
    return x, w13, w2[:, : INTER_PER_RANK // 2].contiguous()


@pytest.mark.parametrize(
    "make",
    [_strided_x, _misaligned_x, _too_many_rows, _tp8_slice, _fp16_x, _mismatched_down],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_inputs_outside_the_domain_raise(make) -> None:
    """Loud, never a silently wrong answer: these are why `is_valid` has
    nothing left to refuse."""
    x, gate_up_weight, down_weight = make()
    with pytest.raises(RuntimeError):
        with validating(op):
            op(x, gate_up_weight, down_weight)
