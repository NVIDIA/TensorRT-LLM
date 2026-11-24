# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused routing + MXFP8 prologue: every cell bitwise against the separated routing and
quantization it replaces, and CUDA-graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (  # noqa: E501  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    k31_moe_quant,
    k31_moe_route,
    quant,
    route,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (  # noqa: E501
    k31_moe_prologue as op,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("moe_prologue runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "moe_prologue requires a CUDA device"

# The rows per rank the targets that call the prologue route (TEP32 8; TEP16 6 / 14 / 24 / 72, with the runner's
# routing weights under attention TP; DEP16 96 / 140 / 168 rows per rank, routed and quantized on the source rank).
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__tep": (
        Cell(why="TEP8 prefill chunk of 8192 tokens", spec=dict(rows=8192, scale=1.0)),
        *(
            Cell(
                why=f"attention TP rows {r}: in-kernel routing weights",
                spec=dict(rows=r, scale=1.0, runner_weights=True),
            )
            for r in (6, 14, 24)
        ),
    ),
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__dep": tuple(
        Cell(why=f"DEP16 rows {r}", spec=dict(rows=r, scale=1.0))
        for r in (6, 14, 24, 72, 96, 140, 168)
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = spec["rows"]
    logits = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32)
    bias = torch.randn(NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32) * 0.1
    x = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 3).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block: scale byte 0, zero codes
    if m > 1:
        x[1, 32:64] *= 1e4  # a block near the E4M3 range edge
    if spec.get("runner_weights"):
        return logits, bias, spec["scale"], x, True
    return logits, bias, spec["scale"], x


def _same(out, ref):
    for name, a, b in zip(("ids", "weights", "x_fp8", "x_sf"), out, ref):
        assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (
            f"{name} differs from the reference"
        )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    _same(out, op.reference(*args))


def test_cuda_graph_replay() -> None:
    args = _build(dict(rows=168, scale=1.0), seed=9)
    op(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = op(*args)
    args[0].add_(0.25 * torch.randn_like(args[0]))  # new routing inputs, same buffers
    args[3].mul_(-1)
    g.replay()
    torch.cuda.synchronize()
    _same(out, op.reference(*args))


def test_rejects_wrong_inputs() -> None:
    logits, bias, scale, x = _build(dict(rows=16, scale=1.0), seed=4)
    with pytest.raises((ValueError, RuntimeError)):
        op(logits, bias, scale, x.float())
    with pytest.raises((ValueError, RuntimeError)):
        op(logits, bias, scale, x[:8])
    with pytest.raises((ValueError, RuntimeError)):
        op(logits.half(), bias, scale, x)


def test_fp64_rounded_weights_at_bf16_midpoints() -> None:
    """Up to 64 rows the weights are the FP64 expression rounded once to BF16 (the fused C++ op's),
    computed in FP32 except where the FP32 quotient is a BF16 rounding midpoint. Many random
    batches make such quotients occur; every weight must match the C++ op's bits."""
    g = torch.Generator(device="cuda").manual_seed(1234)
    midpoints = 0
    for _ in range(400):
        logits = torch.randn(64, NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32) * 2
        bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)
        x = torch.randn(64, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        ids, w, _, _ = op(logits, bias, 1.0, x)
        ref_ids, ref_w, _, _ = op.reference(logits, bias, 1.0, x)
        assert torch.equal(ids, ref_ids)
        assert torch.equal(w.view(torch.int16), ref_w.view(torch.int16))
        scores = torch.sigmoid(torch.gather(logits, 1, ids.long()))
        q = scores / scores.sum(1, keepdim=True)
        midpoints += int(((q.view(torch.int32) & 0xFFFF) == 0x8000).sum())
    print(f"FP32 quotients at a BF16 midpoint (approximate count): {midpoints}")


def _runner_routing_weights(logits, bias):
    """The TRTLLM-Gen runner's own in-kernel routing weights (one-group DeepSeek-V3 over FP32 logits), read back
    from the NVFP4 runner, which returns them unfinalized (the MXFP4 runner shares the routing code but not this
    output). Eight dummy local experts: the routing does not depend on them."""
    rows, (loc, hid, inter) = logits.shape[0], (8, 2560, 640)
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randint(
        0, 256, (loc, 2 * inter, hid // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w2 = torch.randint(
        0, 256, (loc, hid, inter // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    s1 = torch.ones(loc, 2 * inter, hid // 16, device="cuda").to(torch.float8_e4m3fn)
    s2 = torch.ones(loc, hid, inter // 16, device="cuda").to(torch.float8_e4m3fn)
    xq = torch.randint(0, 256, (rows, hid // 2), dtype=torch.uint8, device="cuda", generator=g)
    xsf = torch.ones(rows * hid // 16, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones(loc, dtype=torch.float32, device="cuda")
    outs = torch.ops.trtllm.fp4_block_scale_moe_runner(
        logits,
        bias,
        xq,
        xsf,
        w1,
        s1,
        None,
        None,
        None,
        None,
        w2,
        s2,
        None,
        one,
        one,
        one,
        NUM_EXPERTS,
        16,
        1,
        1,
        inter,
        0,
        loc,
        1.0,
        2,
        False,
        0,
    )
    return outs[1].to(torch.bfloat16)


# Every routing-kernel family the runner dispatches by row count: the cooperative block kernel (1..4), the
# single-cluster kernel (5..16 and on), the histogram / cooperative multi-CTA path (prefill sizes).
@pytest.mark.parametrize("rows", [1, 2, 4, 6, 8, 13, 16, 24, 72, 2048, 4096])
def test_runner_weights_match_the_runner_routing(rows) -> None:
    """runner_weights reproduces the TRTLLM-Gen runner's in-kernel routing weights bit for bit (the ground
    truth the attention-TP scaffold uses); the separated paths' weights do not always."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    mismatches = {"runner": 0, "separated": 0}
    for _ in range(max(1, 200_000 // (rows * 16))):
        logits = torch.randn(rows, NUM_EXPERTS, generator=g, device="cuda") * 1.5
        bias = torch.randn(NUM_EXPERTS, generator=g, device="cuda") * 0.08
        x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        ids, w, _, _ = op(logits, bias, 1.0, x, True)
        sep_ids, sep_w, _, _ = op(logits, bias, 1.0, x)
        truth = _runner_routing_weights(logits, bias)
        assert torch.equal(ids, sep_ids), "selection differs"
        mismatches["runner"] += int((w.view(torch.int16) != truth.view(torch.int16)).sum())
        mismatches["separated"] += int((sep_w.view(torch.int16) != truth.view(torch.int16)).sum())
    print(f"rows {rows}: weights differing from the runner's routing: {mismatches}")
    assert mismatches["runner"] == 0


@pytest.mark.parametrize("runner_weights", [True, False])
@pytest.mark.parametrize("rows", [33, 40, 80, 105, 130, 145, 192, 300])
def test_route_and_quant_halves_are_bitwise(rows, runner_weights) -> None:
    """The routing half and the quantization half, run as two launches through the ``route`` / ``quant`` entries
    (the tep target's early routing above the front's rows), give the bits of the one-launch prologue."""
    scale = 1.0 if runner_weights else 2.5
    logits, bias, _, x = _build(dict(rows=rows, scale=scale), seed=rows * 7 + runner_weights)[:4]
    full = op(logits, bias, scale, x, runner_weights)
    ids, weights = route(logits, bias, scale, runner_weights)
    x_fp8, x_sf = quant(x)
    _same((ids, weights, x_fp8, x_sf), full)


@pytest.mark.parametrize("cell", k31_moe_route.CELLS, ids=[c.why[:44] for c in k31_moe_route.CELLS])
def test_route_cells(cell) -> None:
    spec = cell.spec
    logits, bias, scale, x = _build(
        dict(rows=spec["rows"], scale=spec["scale"]), seed=spec["rows"] * 11
    )[:4]
    args = (logits, bias, scale, spec["runner_weights"])
    _same(k31_moe_route(*args), k31_moe_route.reference(*args))


@pytest.mark.parametrize("cell", k31_moe_quant.CELLS, ids=[c.why[:44] for c in k31_moe_quant.CELLS])
def test_quant_cells(cell) -> None:
    x = _build(dict(rows=cell.spec["rows"], scale=1.0), seed=cell.spec["rows"] * 13)[3]
    _same(k31_moe_quant(x), k31_moe_quant.reference(x))


def test_prerouted_sort_matches_the_op_sort() -> None:
    """The sort a caller issues ahead of the CuTe DSL experts (``prerouted_sort``) is the one the op runs itself:
    the op's unfinalized output with it equals the op's own sort, at TEP16 rows with 16-row tiles."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        w4a8_moe_cute as mc,
    )

    g = torch.Generator(device="cuda").manual_seed(3)
    local, inter = 224, 768
    w1 = torch.randint(
        0, 256, (local, 2 * inter, HIDDEN // 2), generator=g, device="cuda", dtype=torch.uint8
    )
    s1 = torch.randint(
        118, 124, (local, 2 * inter, HIDDEN // 32), generator=g, device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (local, HIDDEN, inter // 2), generator=g, device="cuda", dtype=torch.uint8
    )
    s2 = torch.randint(
        118, 124, (local, HIDDEN, inter // 32), generator=g, device="cuda", dtype=torch.uint8
    )
    for rows in (40, 105, 145):
        logits, bias, _, x = _build(dict(rows=rows, scale=1.0), seed=rows)[:4]
        ids, weights, x_fp8, x_sf = op(logits, bias, 1.0, x, True)
        n_tile = mc.pick_n_tile(rows)
        ws = torch.zeros(mc.moe_workspace_bytes(mc.max_tiles_for(rows, local, n_tile), inter, n_tile),
                         dtype=torch.uint8, device="cuda")  # fmt: skip
        kw = dict(
            finalize=False,
            local_expert_offset=0,
            local_num_experts=local,
            n_tile=n_tile,
            workspace=ws,
        )
        own = mc.k31_w4a8_moe_cute(x_fp8, x_sf, ids, weights, w1, s1, w2, s2, **kw)
        pre = mc.prerouted_sort(ids, 0, local, n_tile)
        assert pre is not None
        early = mc.k31_w4a8_moe_cute(x_fp8, x_sf, ids, weights, w1, s1, w2, s2, prerouted=pre, **kw)
        assert torch.equal(own[2], early[2])
        sel = own[2].reshape(-1) >= 0
        rows_own = own[0][own[2].reshape(-1)[sel].long()]
        rows_early = early[0][early[2].reshape(-1)[sel].long()]
        assert torch.equal(rows_own.view(torch.int16), rows_early.view(torch.int16))
