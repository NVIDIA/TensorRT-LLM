# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection_mxf8_decode entry.

A producer-written operand (`pre`) holds the activation quant's bytes, so the GEMM alone
must give the quant + GEMM pair's output bitwise. The wide runner (`build_wide`, the MTP3 /
MTP5 decode rows 65..192 of o_proj) must give the context MXF8 GEMM's output bitwise: those
rows took that GEMM before, on the same operands in the same K order."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection_mxf8_decode as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — path declared above

op = entry.cute_dsl_projection_mxf8_decode
_SHAPES = {"q_b": (4096, 2048), "o_proj": (6144, 4096)}
_RUNNERS: dict = {}


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    n, k = _SHAPES[kernel]
    return (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)


def _runner(kernel: str, wide: bool = False) -> entry.DecodeMxf8Runner:
    if (kernel, wide) not in _RUNNERS:
        g = torch.Generator(device="cuda").manual_seed(1)
        w8 = entry._kernel.quantize_weight(_weight(kernel, g))
        _RUNNERS[kernel, wide] = (entry.build_wide if wide else entry.build)(kernel, w8)
    return _RUNNERS[kernel, wide]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    runner = _runner(kernel, bool(cell.spec.get("wide")))
    assert runner.rows[0] <= rows <= runner.rows[1]
    n, k = _SHAPES[kernel]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    weight = entry._kernel.quantize_weight(_weight(kernel, g))
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    pre = bool(cell.spec.get("pre"))
    if pre:
        # The producer's bytes: the activation quant's, written into the launcher's operand.
        xq, sf = quant.reference(x)
        runner.operand[0][:rows].copy_(xq)
        need = quant.sf_bytes(rows, k)
        runner.operand[1][:need].copy_(sf[:need])
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        assert op(x, weight, out, runner, pre=pre) is out
    torch.cuda.synchronize()
    ref = op.reference(x, weight, out, runner, pre=pre)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)
    if pre:
        plain = torch.full_like(out, float("nan"))
        op(x, weight, plain, runner)
        torch.cuda.synchronize()
        assert torch.equal(out, plain), "the producer's operand must give the quant path's output"


#: MTP3 / MTP5 decode row counts: each instance's ends, the first wide count, the 24-request
#: batches (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 68, 96, 126, 127, 128, 129, 144, 150, 191, 192)


def test_wide_is_the_context_gemm_bitwise() -> None:
    """At every MTP3 / MTP5 decode row count the wide runner's output -- through its own
    quant and on a producer's operand -- is the context MXF8 GEMM's (the path those rows
    took before), bit for bit, on a non-degenerate product; stale bytes in the operand's
    rows past the call's are never stored; 64 and 193 rows raise."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E501
        cute_dsl_projection_mxf8 as ctx,
    )

    runner = _runner("o_proj", wide=True)
    n, k = _SHAPES["o_proj"]
    g = torch.Generator(device="cuda").manual_seed(11)
    weight = entry._kernel.quantize_weight(_weight("o_proj", g))
    ctx_runner = ctx.build("o_proj", weight)
    for rows in WIDE_ROWS:
        x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
        want = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        ctx.cute_dsl_projection_mxf8(x, weight, want, ctx_runner)
        runner.operand[0].fill_(0x7F)  # NaN bytes in every row the call does not own
        got = torch.full_like(want, float("nan"))
        op(x, weight, got, runner)
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"{rows} rows: not the context GEMM's output"
        ref = op.reference(x, weight, got, runner).float()
        assert float(ref.abs().max()) > 0 and float((ref == 0).float().mean()) < 0.01
        xq, sf = quant.reference(x)
        runner.operand[0][:rows].copy_(xq)
        runner.operand[1][: sf.numel()].copy_(sf)
        pre = torch.full_like(want, float("nan"))
        op(x, weight, pre, runner, pre=True)
        torch.cuda.synchronize()
        assert torch.equal(pre, want), f"{rows} rows: the producer's operand differs"
    for rows in (64, 193):
        x = torch.zeros(rows, k, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="rows"):
            runner.launch(x, weight, out)


@pytest.mark.parametrize("rows", (128, 150, 192))
def test_wide_replays_bitwise_in_a_cuda_graph(rows) -> None:
    """The wide GEMM on a producer's operand runs inside the decode CUDA graph: a replay
    writes the eager launch's bytes, and after new operand bytes are written, the eager
    launch's product on them."""
    runner = _runner("o_proj", wide=True)
    n, k = _SHAPES["o_proj"]
    g = torch.Generator(device="cuda").manual_seed(rows)
    weight = entry._kernel.quantize_weight(_weight("o_proj", g))

    def put(seed: int) -> torch.Tensor:
        gx = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, k, generator=gx, device="cuda").to(torch.bfloat16)
        xq, sf = quant.reference(x)
        runner.operand[0][:rows].copy_(xq)
        runner.operand[1][: sf.numel()].copy_(sf)
        return x

    x = put(1)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    op(x, weight, out, runner, pre=True)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, out, runner, pre=True)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the replay differs from eager"
    x = put(2)
    again = torch.empty_like(out)
    op(x, weight, again, runner, pre=True)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, again) and not torch.equal(again, eager), "replay on new operands"


def test_shift_scales_is_the_scaled_weight() -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    w = _weight("q_b", g)
    shifted = entry.shift_scales(entry._kernel.quantize_weight(w), 3)
    direct = entry._kernel.quantize_weight((w.float() * 8).to(torch.bfloat16))
    assert torch.equal(shifted.q, direct.q) and torch.equal(shifted.sf, direct.sf)
