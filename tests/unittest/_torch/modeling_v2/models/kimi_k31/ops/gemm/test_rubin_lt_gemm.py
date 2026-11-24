# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_lt_gemm entry: helix8's generation GEMMs at a pinned cuBLASLt algorithm.

GPU (sm_107): every B-class cell (a pinned helix8 GEMM) writes the words of the call it replaces
(``cublas_mm``, ``rubin_linear``, the router's ``dsv3_router_gemm_op`` or ``bmm_out``) on three
activation families, every cell (A-class too: another summation order) repeats its own words and
sits within one BF16 ulp of the row's largest magnitude of the entry's FP32 reference; a captured
call of a B-class cell replays on new inputs with the replaced call's words. CPU: cuBLASLt's
column-major description of every operand layout the target passes reproduces ``a @ b`` (a
column-major emulator over the operands' memory), and the guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.rubin_lt_pin import (  # noqa: E501
    lt_problem,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_lt_gemm import (
    LtPin,
    rubin_lt_gemm,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

_HEADS = 96  # W_UV: one batch per MLA head
_PIN = LtPin(66, 15, 35, 11, 1, 0, 0, 1, 0)
#: The reference band: the pinned and the replaced call round FP32 sums of the same products once,
#: torch's reference sums them in its own order. One ulp at the row's largest magnitude M is
#: 2^(floor(log2 M) - 7) / M <= 2^-7 of M (BF16 keeps 8 significant bits), and every element of
#: the row is within it; the aggregate within 0.125 x 2^-7 = 0.25 x 2^-8. FP32 logits sit far
#: inside that band: it checks the product, the equality with the replaced call checks the bits.
_BAND = dict(element_ulp=1.0, rms_ulp=0.125, ulp=2.0**-7)


def _operands(spec: dict, seed: int, family: str = "gauss", device: str = "cuda") -> tuple:
    """``(a, b, out)`` of the cell's call: BF16 rows times a stored weight's ``.t()`` (``out`` FP32
    for the router), or for ``uv`` the latent output's 96 heads times W_UV into the value heads,
    as strided batches. ``family``: ``gauss``, ``rows`` (each row at a scale of 2^-3..2^3) or
    ``outlier`` (four channels 2^6 larger)."""
    rows, n, k = spec["rows"], spec["n"], spec["k"]
    gen = torch.Generator().manual_seed(seed)
    if spec["today"] == "uv":
        x = torch.randn(rows, _HEADS, k, generator=gen)
        kv_b = torch.randn(_HEADS * n, k, generator=gen) * k**-0.5
        v_b_t = kv_b.to(torch.bfloat16).to(device).view(_HEADS, n, k).transpose(1, 2)
        out = torch.empty(rows, _HEADS, n, dtype=torch.bfloat16, device=device)
        a = x.to(torch.bfloat16).to(device).transpose(0, 1)
        return a, v_b_t, out.transpose(0, 1)
    x = torch.randn(rows, k, generator=gen)
    if family == "rows":
        x *= torch.pow(2.0, torch.randint(-3, 4, (rows, 1), generator=gen).float())
    elif family == "outlier":
        x[:, torch.randperm(k, generator=gen)[:4]] *= 2.0**6
    w = (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16).to(device)
    dtype = torch.float32 if spec["today"] == "router" else torch.bfloat16
    out = torch.empty(rows, n, dtype=dtype, device=device)
    return x.to(torch.bfloat16).to(device), w.t(), out


def _replaced(spec: dict, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The call the pin replaces in helix8's modeling.py, on the same operands."""
    if spec["today"] == "mm":
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

        return cublas_mm(a, b)
    if spec["today"] == "linear":
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (  # noqa: E501
            rubin_linear,
        )

        return rubin_linear(a, b.t(), False)
    if spec["today"] == "router":
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (  # noqa: E501
            dsv3_router_gemm_op,
        )

        return dsv3_router_gemm_op(a, b, None, torch.float32)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out

    return bmm_out(a, b, torch.empty_like(out))


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


@on_sm107
@pytest.mark.parametrize("cell", rubin_lt_gemm.CELLS, ids=[c.why[:48] for c in rubin_lt_gemm.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(rubin_lt_gemm)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    for family in ("gauss", "rows", "outlier"):
        a, b, out = _operands(spec, seed, family)
        got = rubin_lt_gemm(a, b, out, spec["pin"])
        want = _replaced(spec, a, b, out)
        again = rubin_lt_gemm(a, b, torch.empty_like(out), spec["pin"])
        torch.cuda.synchronize()
        if spec["exact"]:
            assert torch.equal(_words(got), _words(want)), (
                f"{family}: not the replaced call's words"
            )
        assert torch.equal(_words(again), _words(got)), f"{family}: a repeat differs"
        ref = rubin_lt_gemm.reference(a, b, out, spec["pin"])
        assert_within_ulp(got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1]), **_BAND)


@on_sm107
@pytest.mark.parametrize("today", sorted({c.spec["today"] for c in rubin_lt_gemm.CELLS}))
def test_a_captured_call_replays_the_replaced_calls_words_on_new_inputs(today):
    (cell,) = [c for c in rubin_lt_gemm.CELLS if c.spec["today"] == today and c.spec["exact"]][:1]
    spec = cell.spec
    a, b, out = _operands(spec, 1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rubin_lt_gemm(a, b, out, spec["pin"])  # the plan, outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        rubin_lt_gemm(a, b, out, spec["pin"])
    for seed, family in ((2, "rows"), (3, "outlier")):
        a2, b2, _ = _operands(spec, seed, family)
        a.copy_(a2)
        b.copy_(b2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_words(out), _words(_replaced(spec, a, b, out))), family


def _column_major(t: torch.Tensor, spec: dict, batch: int) -> torch.Tensor:
    """The column-major ``[rows, cols]`` (per batch) matrix cuBLASLt reads at ``t``'s address."""
    if batch > 1:
        return torch.as_strided(
            t, (batch, spec["rows"], spec["cols"]), (spec["stride"], 1, spec["ld"])
        )
    return torch.as_strided(t, (spec["rows"], spec["cols"]), (1, spec["ld"]))


@pytest.mark.cpu_only
@pytest.mark.parametrize("cell", rubin_lt_gemm.CELLS, ids=[c.why[:48] for c in rubin_lt_gemm.CELLS])
def test_the_column_major_description_reproduces_the_product(cell):
    """cuBLASLt's ``D = op(A) @ op(B)`` over the operands' memory (``A := b``, ``B := a``, both read
    column-major) is ``(a @ b)^T``, and ``D`` read column-major at ``out``'s address is ``out``. Entries
    of -1, 0 and 1 on the CPU: every product and sum is exact, in BF16 too."""
    spec = dict(cell.spec, rows=5, n=min(cell.spec["n"], 24), k=min(cell.spec["k"], 40))
    a, b, out = _operands(spec, 7, device="cpu")
    gen = torch.Generator().manual_seed(8)
    a.copy_(torch.randint(-1, 2, a.shape, generator=gen).to(a.dtype))
    b.copy_(torch.randint(-1, 2, b.shape, generator=gen).to(b.dtype))
    p = lt_problem(a, b, out)
    op_a = _column_major(b, p["A"], p["batch"]).float()
    op_b = _column_major(a, p["B"], p["batch"]).float()
    op_a = op_a.transpose(-1, -2) if p["transa"] else op_a
    op_b = op_b.transpose(-1, -2) if p["transb"] else op_b
    d = op_a @ op_b
    want = a.float() @ b.float()
    assert torch.equal(d, want.transpose(-1, -2)), "D is not (a @ b)^T"
    out.copy_(want.to(out.dtype))
    assert torch.equal(_column_major(out, p["D"], p["batch"]).float(), d), "D is not out's memory"
    assert (p["m"], p["n"], p["k"]) == (spec["n"], spec["rows"], spec["k"])


@pytest.mark.cpu_only
def test_guard_admits_the_cells():
    for cell in rubin_lt_gemm.CELLS:
        a, b, out = _operands(cell.spec, 3, device="cpu")
        rubin_lt_gemm.is_valid(a, b, out, cell.spec["pin"])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    ["fp16_a", "fp32_b", "fp16_out", "k", "n", "a_stride", "b_stride", "out_stride", "pin", "algo"],
)  # fmt: skip
def test_guard_refuses_what_cublaslt_would_read_or_write_wrong(case):
    rows, n, k = 6, 32, 64
    a = torch.zeros(rows, k, dtype=torch.bfloat16)
    b = torch.zeros(n, k, dtype=torch.bfloat16).t()
    out, pin = torch.empty(rows, n, dtype=torch.bfloat16), _PIN
    if case == "fp16_a":
        a = a.half()
    elif case == "fp32_b":
        b = b.float()
    elif case == "fp16_out":
        out = out.half()
    elif case == "k":
        b = torch.zeros(n, k // 2, dtype=torch.bfloat16).t()
    elif case == "n":
        out = torch.empty(rows, n // 2, dtype=torch.bfloat16)
    elif case == "a_stride":
        a = torch.zeros(rows, 2 * k, dtype=torch.bfloat16)[:, ::2]
    elif case == "b_stride":
        b = torch.zeros(2 * n, 2 * k, dtype=torch.bfloat16)[::2, ::2].t()
    elif case == "out_stride":
        out = torch.empty(n, rows, dtype=torch.bfloat16).t()
    elif case == "pin":
        pin = tuple(_PIN)
    elif case == "algo":
        pin = _PIN._replace(algo=None)
    with pytest.raises(AssertionError):
        rubin_lt_gemm.is_valid(a, b, out, pin)
