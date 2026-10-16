# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_ctm_gemv_wide catalog entry.

The Kimi K3 TP16 per-rank shapes of the projections of a wide decode step, at every M in 1..64: error of the plain
(sig_col0=-1) output against an fp64 product of the same bf16 inputs, the sigmoid columns torch.sigmoid of that
output bit for bit, identical bits on a repeated call, and each M's rows bit-identical to the same rows of the
64-row call. Where the device's split equals a k3_ctm_gemv_long call site's, every token's row carries that op's
bits; a CUDA-graph replay returns the eager bits; calls outside the op's preconditions raise ValueError.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_long import (
    k3_ctm_gemv_long,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_wide import (
    k3_ctm_gemv_wide,
)

assert torch.cuda.is_available(), "k3_ctm_gemv_wide requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

M_ALL = range(1, 65)
TOL = 8e-3  # bf16 output: max |y - ref| / max |ref| against an fp64 product of the same bf16 inputs
TOL_FP32 = 1e-4  # fp32 output

# (N, K, sig_col0, out_fp32): the per-rank TP16 shapes of the projections of a wide decode step. kda_qkvg is 25
# whole 128-row blocks and an 8-row last one; mla_qkv_a_gate is [W_a; W_g] with its gate columns as bf16(sigmoid);
# moe_head is [latent down slice; router rows] in fp32 (router logits); moe_tail is 5 k-tiles; dense_down's K 2112
# ends in a half k-tile.
CELLS = {
    "kda_qkvg": (3208, 7168, -1, False),
    "mla_qkv_a_gate": (2880, 7168, 2112, False),
    "o_proj": (7168, 768, -1, False),
    "moe_head": (280, 7168, -1, True),
    "moe_tail": (7168, 640, -1, False),
    "shared_gate_up": (768, 7168, -1, False),
    "dense_gate_up": (4224, 7168, -1, False),
    "dense_down": (7168, 2112, -1, False),
    "drafter_qkv": (512, 7168, -1, False),
    "drafter_gate_up": (1792, 7168, -1, False),
    "drafter_o_proj": (7168, 384, -1, False),
}
# The k3_ctm_gemv_long call (split, ring, push) of the shapes that run it at up to 8 tokens.
LONG_SITES = {
    "mla_qkv_a_gate": (6, 6, True),
    "dense_gate_up": (4, 5, False),
    "dense_down": (2, 6, False),
    "drafter_qkv": (8, 6, True),
    "drafter_gate_up": (8, 6, True),
}


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _weight(n: int, k: int) -> torch.Tensor:
    return _randn(n, k, 1, 0.02)


def _rows(k: int) -> torch.Tensor:
    return _randn(64, k, 2, 1.0)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


def test_projection_cells() -> None:
    for name, (n, k, sig, fp32) in CELLS.items():
        w = _weight(n, k)
        x64 = _rows(k)
        ref64 = x64.double() @ w.double().t()
        y64 = k3_ctm_gemv_wide(x64, w, sig, fp32)
        dtype = torch.float32 if fp32 else torch.bfloat16
        tol = TOL_FP32 if fp32 else TOL
        for m in M_ALL:
            cell = f"{name} M={m}"
            x = x64[:m].contiguous()
            y = k3_ctm_gemv_wide(x, w, sig, fp32)
            assert y.shape == (m, n) and y.dtype == dtype, cell
            again = k3_ctm_gemv_wide(x, w, sig, fp32)
            assert torch.equal(_bits(y), _bits(again)), f"{cell}: a repeated call changed bits"
            assert torch.equal(_bits(y), _bits(y64[:m])), (
                f"{cell}: rows differ from the 64-row call"
            )
            plain = k3_ctm_gemv_wide(x, w, -1, fp32) if sig >= 0 else y
            rel = _rel(plain, ref64[:m])
            assert rel <= tol, f"{cell}: max |y - ref| / max |ref| = {rel:.2e} > {tol}"
            if sig >= 0:
                assert torch.equal(_bits(y[:, :sig]), _bits(plain[:, :sig])), (
                    f"{cell}: columns before sig_col0 differ from the sig_col0=-1 call"
                )
                assert torch.equal(_bits(y[:, sig:]), _bits(plain[:, sig:].sigmoid())), (
                    f"{cell}: columns from sig_col0 on are not torch.sigmoid of the sig_col0=-1 call"
                )


def test_rows_match_long_at_the_same_split() -> None:
    """A token's row is k3_ctm_gemv_long's for it (calls of up to 8 tokens) where the splits agree."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op as ctm_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.k3_ctm_gemv_kernel import wide_tile

    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    compared = 0
    for name, (split, ring, push) in LONG_SITES.items():
        n, k, sig, _ = CELLS[name]
        w = _weight(n, k)
        x64 = _rows(k)
        groups = torch.cat(
            [
                k3_ctm_gemv_long(x64[r : r + 8], w, sig, split, ring, True, push)
                for r in range(0, 64, 8)
            ]
        )
        for m in [*range(1, 9), *range(16, 65, 8)]:
            if ctm_op.wide_config(n, k, wide_tile(m), sms)[0] != split:
                continue
            x = x64[:m].contiguous()
            y = k3_ctm_gemv_wide(x, w, sig)
            want = k3_ctm_gemv_long(x, w, sig, split, ring, True, push) if m <= 8 else groups[:m]
            assert torch.equal(_bits(y), _bits(want)), (
                f"{name} M={m}: rows differ from k3_ctm_gemv_long's at split {split}"
            )
            compared += 1
    assert compared > 0, f"no call site's split is this device's ({sms} SMs); nothing was compared"


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after an eager call per key and replayed with x rewritten in place: the eager bits."""
    calls = []
    for name in ("mla_qkv_a_gate", "moe_head"):
        n, k, sig, fp32 = CELLS[name]
        w = _weight(n, k)
        for m in (1, 64):
            x = _rows(k)[:m].clone()
            k3_ctm_gemv_wide(x, w, sig, fp32)  # compiles the key outside capture
            calls.append((f"{name} M={m}", x, w, sig, fp32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [k3_ctm_gemv_wide(x, w, sig, fp32) for _, x, w, sig, fp32 in calls]
    for seed, (_, x, *_) in enumerate(calls, start=100):
        x.copy_(_randn(x.shape[0], x.shape[1], seed, 1.0))
    graph.replay()
    for (cell, x, w, sig, fp32), y in zip(calls, outs):
        eager = k3_ctm_gemv_wide(x, w, sig, fp32)
        assert torch.equal(_bits(y), _bits(eager)), f"{cell}: the replay differs from an eager call"


def test_unsupported_calls_raise_value_error() -> None:
    """The op's own check refuses these before compiling or launching anything."""
    n, k = 3208, 7168
    w = _weight(n, k)
    x8 = _rows(k)[:8]
    flat = torch.zeros(8 * k + 8, dtype=torch.bfloat16, device="cuda")
    cases = {
        "0 rows": (x8[:0], w, -1, False),
        "65 rows": (torch.cat([_rows(k), x8[:1]]), w, -1, False),
        "x 2 bytes past a 16-byte boundary": (flat[1 : 1 + 8 * k].view(8, k), w, -1, False),
        "sig_col0 = N": (x8, w, n, False),
        "sigmoid columns with fp32 output": (x8, w, 100, True),
        "N % 8 != 0": (x8, w[:3204], -1, False),
        "K % 64 != 0": (x8[:, :7136].contiguous(), w[:, :7136].contiguous(), -1, False),
        "fp16 x": (x8.half(), w, -1, False),
    }
    for name, (x, weight, sig, fp32) in cases.items():
        try:
            k3_ctm_gemv_wide(x, weight, sig, fp32)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError, the op accepted the call")
