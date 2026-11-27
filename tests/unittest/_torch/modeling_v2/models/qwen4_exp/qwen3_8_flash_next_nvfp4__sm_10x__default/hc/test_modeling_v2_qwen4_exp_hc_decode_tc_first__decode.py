# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qwen4_exp_hc_decode_tc_first variant
(ops/hc/qwen4_exp_hc_decode_tc_first__decode): cells in the band against the float64
reference at the decode rows the members run, the row-count guard, graph replay at the
note's 1, 4, 16 and 256 rows (every tile variant)."""

import ast
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.hc import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first__decode as kernel_module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.hc.qwen4_exp_hc_decode_tc_first__decode import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_tc_first requires a CUDA device"

DEV = "cuda"
RANK = 320


def _ratio(out, ref) -> float:
    """Worst element's distance from `ref` as a fraction of the band: one ulp of
    the element (`eps * |ref|`) plus `eps / 16` of the tensor's largest value.
    An fp32-accumulated 10240-term projection rounded once can sit one ulp off
    the float64 value at a rounding midpoint; the absolute term covers SiLU's
    cancellation near zero, where an fp32 logit error of ~1e-6 relative to the
    row's scale is many ulps of a tiny output."""
    eps = torch.finfo(ref.dtype).eps
    r = ref.double()
    return ((out.double() - r).abs() / (eps * r.abs() + eps / 16 * r.abs().max())).max().item()


def _compare(out, ref) -> None:
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _ratio(got, want) <= 1.0, f"{_ratio(got, want):.4g}x the band"


def _build(spec, seed):
    hc, hidden = 4, 2560
    rows, width, inject = spec["rows"], hc * hidden, spec.get("inject", True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # This checkpoint's layer statistics: down ~N(0, 0.0296), block_inject ~N(0, 0.0107),
    # the normed stream unit-RMS per stream times (1 + gamma), gamma ~N(0, 0.477).
    down = torch.randn(RANK, width, generator=g, device=DEV) * 0.0296
    parts = [down] + ([torch.randn(hc, width, generator=g, device=DEV) * 0.0107] if inject else [])
    w = torch.cat(parts)
    w = torch.cat([w, torch.zeros(384 - w.shape[0], width, device=DEV)])
    gamma = torch.randn(width, generator=g, device=DEV) * 0.477
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.flatten(-2) * (1.0 + gamma)).to(torch.bfloat16)
    return normed, w.to(torch.bfloat16).t(), RANK, hc, inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args))


def test_guard_refuses_more_than_two_tiles_of_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=257), seed=4)
    with validating(op), pytest.raises(AssertionError, match="at most 256 rows"):
        op(normed, first, rank, c, wi)


@pytest.mark.parametrize("rows", [1, 4, 16, 256])
def test_cuda_graph_capture_and_replay(rows) -> None:
    normed, first, rank, c, wi = _build(dict(rows=rows), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(normed, first, rank, c, wi)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi)
    fresh = _build(dict(rows=rows), seed=6)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


_PEER_LOADS = ("_ld_dsmem_f32", "_ld_dsmem_f32_if")


def _dsmem_reduce_kernels():
    """`(class name, [(line, callee)])` for every kernel class of the module that reads its
    cluster peers' shared memory, its calls in source order."""
    tree = ast.parse(open(kernel_module.__file__).read())
    out = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        calls = sorted(
            (c.lineno, c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id)
            for c in ast.walk(cls)
            if isinstance(c, ast.Call) and isinstance(c.func, (ast.Attribute, ast.Name))
        )
        if any(name in _PEER_LOADS for _, name in calls):
            out.append((cls.name, calls))
    return out


_DSMEM_KERNELS = _dsmem_reduce_kernels()


@pytest.mark.parametrize("name,calls", _DSMEM_KERNELS, ids=[k[0] for k in _DSMEM_KERNELS])
def test_no_peer_load_in_flight_at_exit(name, calls) -> None:
    """A CTA leaves the cluster barrier after its reduce and exits, releasing its shared memory.
    A thread that reaches that barrier with a peer load still in flight (issued, never summed)
    lets the barrier release that peer under it: CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT, a
    sticky launch failure of the whole process. ptxas hoists a later step's unconditional loads
    ahead of the share check, so either every peer load is predicated on the thread's share
    (`_ld_dsmem_f32_if`, issued only where summed) or the arrive after them is a release
    `cluster_arrive()`, which waits for them. (The predicate itself is pinned by the bitwise
    tests above: a narrower one would drop partials.)"""
    loads = [(line, callee) for line, callee in calls if callee in _PEER_LOADS]
    last_read = max(line for line, _ in loads)
    after = [
        (line, callee) for line, callee in calls if line > last_read and "cluster_arrive" in callee
    ]
    assert after, f"{name}: no cluster barrier after the peer reads (a CTA may exit while read)"
    unpredicated = [(line, callee) for line, callee in loads if callee == "_ld_dsmem_f32"]
    relaxed = [(line, callee) for line, callee in after if callee != "cluster_arrive"]
    assert not (unpredicated and relaxed), (
        f"{name}: unpredicated peer loads {unpredicated} and a relaxed exit arrive {relaxed}"
    )
