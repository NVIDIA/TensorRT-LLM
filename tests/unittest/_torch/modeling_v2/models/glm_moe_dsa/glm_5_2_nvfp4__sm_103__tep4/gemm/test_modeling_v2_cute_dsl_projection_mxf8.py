# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection_mxf8 entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_projection_mxf8 runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_projection_mxf8

#: One runner per entry, built once per session against one layer's weight.
_RUNNERS: dict = {}


def _shape(kernel: str) -> tuple[int, int]:
    abi = entry._kernel.ABIS[entry.ENTRIES[kernel]]
    return abi.out_features, abi.in_features


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    """A row-major bf16 `[N, K]` weight scaled so each output is O(1); zero where a split
    entry's tiles do not read it (`n_split_k`, the entry's contract)."""
    n, k = _shape(kernel)
    w = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    split = entry._kernel.ABIS[entry.ENTRIES[kernel]].n_split_k
    if split is not None:
        w[split[0] :, split[1] :] = 0
    return w


def _runner(kernel: str) -> entry.Mxf8ProjectionRunner:
    if kernel not in _RUNNERS:
        g = torch.Generator(device="cuda").manual_seed(1)
        _RUNNERS[kernel] = entry.build(kernel, entry.quantize_weight(_weight(kernel, g)))
    return _RUNNERS[kernel]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Against the same MXF8 operands multiplied in fp32: the products of two e4m3 values
    and power-of-two scales are exact in fp32, so the two sides differ only by the order
    of K fp32 additions before one rounding to the output. bf16 out: 4 ulp element-wise
    and 2 ulp relative RMS, as the bf16 projections. e4m3 out: an element differs only
    where the two sums straddle an e4m3 rounding boundary, by one step (3 mantissa bits),
    at no more than 1% of the elements. A wrong scale factor (a whole 32-element block off
    by a power of two) or a wrong tile misses by many steps."""
    if out.dtype == torch.float8_e4m3fn:
        got, want = out.float(), ref.float()
        assert bool(torch.isfinite(got).all()), "an element was not written"
        step = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-6))) - 3)
        diff = (got - want).abs()
        assert bool((diff <= step).all()), f"max error {float((diff / step).max()):.1f} steps"
        frac = float((diff > 0).float().mean())
        assert frac <= 0.01, f"{frac:.2%} of the elements differ from the reference"
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _e4m3_bytes(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Finite e4m3 bytes of O(1) values: a latent-like pre-quantized A operand."""
    vals = torch.randn(rows, k, generator=g, device="cuda") * 4.0
    return vals.to(torch.float8_e4m3fn).view(torch.uint8)


def _block_scales(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Blocked e8m0 scales with codes around 2^0 (2^-4 .. 2^3): every block's own scale."""
    n = entry._kernel.unit_sf_bytes(rows, k)
    return torch.randint(123, 131, (n,), generator=g, device="cuda", dtype=torch.int32).to(
        torch.uint8
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    n, k = _shape(kernel)
    runner = _runner(kernel)
    assert (runner.n, runner.k) == (n, k)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # Another layer's weight: the runner was built against a different one.
    weight = entry.quantize_weight(_weight(kernel, g))
    pre = sfa = None
    if runner.prequant:
        x = _e4m3_bytes(rows, k, g)
        if cell.spec.get("sfa"):
            sfa = _block_scales(rows, k, g)
    else:
        x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
        if cell.spec.get("x_row_stride"):
            # The tail of a wider row, as q_b reads q_a in the q_a | kv_a projection row.
            wide = torch.zeros(rows, cell.spec["x_row_stride"], dtype=torch.bfloat16, device="cuda")
            wide[:, :k] = x
            x = wide[:, :k]
        if cell.spec.get("pre"):
            # The producer's operand: the activation quant's exact bytes.
            pre = tuple(t.contiguous() for t in entry._kernel._quant.reference(x))
    # The dense down's call: the SwiGLU wrote only the operand, which is passed as `x` too.
    x_call = pre[0] if cell.spec.get("x_is_pre") else x
    # NaN, so an element the kernel never writes fails the comparison.
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = out.to(runner.out_dtype)
    with validating(op):
        assert op(x_call, weight, out, runner, pre=pre, sfa=sfa) is out, (
            "the op writes `out` and returns it"
        )
    torch.cuda.synchronize()
    _compare(out, op.reference(x_call, weight, out, runner, pre=pre, sfa=sfa))
    if pre is not None:
        # Same bytes in, same launch: the producer's operand must give the quant path's output.
        plain = torch.full_like(out, float("nan"))
        op(x, weight, plain, runner)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), plain.view(torch.uint8)), (
            "a producer-quantized operand must give the activation quant's output bitwise"
        )
    # FP8-level against the bf16 product: the MXF8 rounding of both operands only.
    if out.dtype == torch.bfloat16 and not runner.prequant:
        dense = (x.float() @ _dequantized(weight, n, k).t()).float()
        rel = ((out.float() - dense).norm() / dense.norm()).item()
        assert rel < 5e-2, f"relative RMS {rel:.3e} vs the product with x unquantized"


def _dequantized(weight, n: int, k: int) -> torch.Tensor:
    return entry._kernel.dequantize_weight(weight)


def test_weight_quantization_is_the_activation_rule() -> None:
    """The load-time weight quantization and the per-step activation quantization are the
    same kernel: equal bytes and scale factors for the same matrix."""
    g = torch.Generator(device="cuda").manual_seed(7)
    w = _weight("o_proj", g)
    wq = entry.quantize_weight(w)
    q, sf = entry._kernel._quant.reference(w)
    assert torch.equal(wq.q, q)
    assert torch.equal(wq.sf[: sf.numel()], sf)


def test_is_valid_refuses_an_unaligned_row_stride() -> None:
    """The quant kernel loads 8 bf16 per thread: a row stride off a multiple of 8 would
    read across rows."""
    runner = _runner("qkva")
    n, k = _shape("qkva")
    wide = torch.zeros(65, k + 4, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(65, n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="row stride"):
        with validating(op):
            op(wide[:, :k], entry.quantize_weight(torch.zeros(n, k, dtype=torch.bfloat16,
                                                              device="cuda")), out, runner)  # fmt: skip
