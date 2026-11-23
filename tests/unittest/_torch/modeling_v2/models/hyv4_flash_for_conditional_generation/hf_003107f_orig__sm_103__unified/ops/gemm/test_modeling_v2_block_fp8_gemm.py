# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the block_fp8_gemm entry of the Hy4-Flash unified target.

The entry owns the contract. This file turns a cell into block-FP8 operands (drawn directly as
e4m3 codes and fp32 scales, so both sides read bit-identical values), runs the block-FP8
`Linear` composition once end to end from bf16 rows through `fp8_quantize_1x128`, replays a
decode shape under CUDA-graph capture, shows the certified band separates a transposed
activation scale and a shifted weight-scale block, and drives the guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.gemm.block_fp8_gemm import (  # noqa: E501
    BLOCK,
    DRAFT_MLP_UP,
    SPARSE_LINEAR_GATE,
    SPARSE_O_PROJ,
    SPARSE_Q_B,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.gemm.block_fp8_gemm import (  # noqa: E501
    block_fp8_gemm as op,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

DEV = "cuda"

assert torch.cuda.is_available(), "block_fp8_gemm requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="block_fp8_gemm is certified on sm_103"
)

#: 4 bf16 ulp of the row max element-wise, 2 ulp of relative RMS.
#:
#: Kernel and reference read bit-identical e4m3 codes and fp32 scales; the products of codes
#: are exact in fp32, and the two differ only in where the scales are applied (per 128-wide K
#: block in the kernel, before the sum in the reference) and in the order of the fp32 sums --
#: a few fp32 ulp of the row scale, far below one bf16 ulp. What remains is the single bf16
#: rounding on each side landing on different neighbours, one ulp of an element at most. A
#: misread scale layout (a transposed activation scale, a shifted weight block) is off by a
#: whole block's contribution, many ulp RMS -- the two controls below measure that.
ELEMENT_ULP = 4.0
RMS_ULP = 2.0

#: How far outside the band a wrong layout has to land for the band to be evidence.
SEPARATION = 10.0


def _blocks(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK


def _build(spec: dict, seed: int) -> tuple[torch.Tensor, ...]:
    """Block-FP8 operands at O(1) output scale, every value exact in its storage type.

    Activation codes ~ 2 N(0, 1) in e4m3 with scales in [0.5, 1.5); weight codes ~ N(0, 1) in
    e4m3 with scales in [0.5, 1.5) / sqrt(K), so the outputs are O(1) at every K.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    m, n, k = spec["m"], spec["n"], spec["k"]
    x = (2.0 * torch.randn(m, k, generator=g, device=DEV)).to(torch.float8_e4m3fn)
    x_scale = 0.5 + torch.rand(_blocks(k), m, generator=g, device=DEV)
    w = torch.randn(n, k, generator=g, device=DEV).to(torch.float8_e4m3fn)
    w_scale = (0.5 + torch.rand(_blocks(n), _blocks(k), generator=g, device=DEV)) / k**0.5
    return x, w, x_scale, w_scale


def _rms(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS of `out` against `ref`, the aggregate gate's own metric."""
    return float(
        (out.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    )


def _compare(out: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """The certified band, plus the measured numbers in the log."""
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
    exact = float((out == ref).float().mean())
    print(f"  {label}: {100 * exact:.4f}% bit-identical, rel RMS {_rms(out, ref):.3e}")


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, w, x_scale, w_scale = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, w, x_scale, w_scale)
    assert out.shape == (cell.spec["m"], cell.spec["n"]) and out.is_contiguous()
    _compare(out, op.reference(x, w, x_scale, w_scale), cell.why[:44])


def test_block_fp8_linear_from_bf16_rows() -> None:
    """The composition a block-FP8 `Linear` runs: `fp8_quantize_1x128` on bf16 rows, then the
    op on its output, passed straight through. Against the entry's reference on that same
    quantization it holds the certified band, which is what shows the quantizer's scale layout
    is the one the GEMM reads."""
    g = torch.Generator(device=DEV).manual_seed(23)
    rows = torch.randn(64, SPARSE_Q_B["k"], generator=g, device=DEV).to(torch.bfloat16)
    x, x_scale = fp8_quantize_1x128(rows)
    _, w, _, w_scale = _build(dict(m=1, **SPARSE_Q_B), seed=29)
    with validating(op):
        out = op(x, w, x_scale, w_scale)
    _compare(out, op.reference(x, w, x_scale, w_scale), "64 bf16 rows through fp8_quantize_1x128")


def test_cuda_graph_capture_and_replay() -> None:
    """A decode step as the target runs it: `fp8_quantize_1x128`, then the op, captured in a
    CUDA graph and replayed on new rows. The replayed quantization equals an eager one bit for
    bit, and the product holds the certified band against the reference on it."""
    g = torch.Generator(device=DEV).manual_seed(47)
    rows = torch.randn(8, SPARSE_Q_B["k"], generator=g, device=DEV).to(torch.bfloat16)
    _, w, _, w_scale = _build(dict(m=1, **SPARSE_Q_B), seed=53)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm_x, warm_scale = fp8_quantize_1x128(rows)
        op(warm_x, w, warm_scale, w_scale)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x, x_scale = fp8_quantize_1x128(rows)
        out = op(x, w, x_scale, w_scale)
    for seed in (59, 61):
        fresh = torch.Generator(device=DEV).manual_seed(seed)
        rows.copy_(torch.randn(rows.shape, generator=fresh, device=DEV).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        eager_x, eager_scale = fp8_quantize_1x128(rows)
        assert torch.equal(x.view(torch.uint8), eager_x.view(torch.uint8))
        assert torch.equal(x_scale, eager_scale)
        _compare(out, op.reference(x, w, x_scale, w_scale), f"graph replay at rows seed {seed}")


def test_use_tvm_ffi_false_matches_the_default_launch() -> None:
    """The two launch paths run the same kernel on the same operands: bit-identical bf16."""
    x, w, x_scale, w_scale = _build(dict(m=64, **DRAFT_MLP_UP), seed=67)
    default = op(x, w, x_scale, w_scale)
    pointer = op(x, w, x_scale, w_scale, use_tvm_ffi=False)
    assert torch.equal(default.view(torch.int16), pointer.view(torch.int16)), (
        "the TVM FFI and pointer launch paths differ; the kernel cache is keyed by the path"
    )
    _compare(pointer, op.reference(x, w, x_scale, w_scale), "use_tvm_ffi=False")


def test_a_transposed_activation_scale_is_outside_the_band() -> None:
    """K 3072 is 24 scale blocks and 8 rows is not: a token-major `[M, K/128]` scale holds the
    same 192 values in a different order, and the kernel reads the pointer as `[24, 8]`. The
    right layout holds the band; the transposed one is many band-widths off."""
    x, w, x_scale, w_scale = _build(dict(m=8, **SPARSE_LINEAR_GATE), seed=71)
    ref = op.reference(x, w, x_scale, w_scale)
    _compare(op(x, w, x_scale, w_scale), ref, "K-block-major activation scale")
    wrong = _rms(op(x, w, x_scale.t().contiguous(), w_scale), ref)
    print(f"  transposed activation scale: rel RMS {wrong:.3e}")
    assert wrong > SEPARATION * RMS_ULP * ULP[torch.bfloat16], (
        f"a transposed activation scale is only {wrong:.3e} off: the band cannot see it"
    )


def test_a_shifted_weight_scale_block_is_outside_the_band() -> None:
    """A weight scale one K block off -- what an off-by-one block index reads -- moves every
    128-wide K block's contribution; the band is many widths away."""
    x, w, x_scale, w_scale = _build(dict(m=64, **SPARSE_O_PROJ), seed=73)
    ref = op.reference(x, w, x_scale, w_scale)
    _compare(op(x, w, x_scale, w_scale), ref, "row-major weight scale")
    wrong = _rms(op(x, w, x_scale, w_scale.roll(1, dims=1).contiguous()), ref)
    print(f"  weight scale one K block off: rel RMS {wrong:.3e}")
    assert wrong > SEPARATION * RMS_ULP * ULP[torch.bfloat16], (
        f"a shifted weight-scale block is only {wrong:.3e} off: the band cannot see it"
    )


def test_guard_refuses_a_transposed_activation_scale() -> None:
    """A token-major `[M, K/128]` scale where the shapes differ is refused before the kernel
    misreads it; at M == K/128 nothing can tell, which is why the entry says so."""
    x, w, x_scale, w_scale = _build(dict(m=8, **SPARSE_LINEAR_GATE), seed=31)
    with pytest.raises(AssertionError, match="K-block-major"):
        with validating(op):
            op(x, w, x_scale.t().contiguous(), w_scale)


def test_guard_refuses_a_strided_scale_slice() -> None:
    """A row slice of the activation needs a contiguous copy of its scale columns."""
    x, w, x_scale, w_scale = _build(dict(m=16, **SPARSE_LINEAR_GATE), seed=37)
    with pytest.raises(AssertionError, match="input_scale must be contiguous"):
        with validating(op):
            op(x[4:12], w, x_scale[:, 4:12], w_scale)


def test_guard_refuses_bf16_scales() -> None:
    """The checkpoint stores `weight_scale_inv` in bf16. The kernel reads fp32 bytes, so a bf16
    tensor of the expected shape would be read as twice as many fp32 values, past its end; the
    upcast to fp32 is exact and happens at load. The guard refuses the bf16 tensor before the
    kernel reads it."""
    x, w, x_scale, w_scale = _build(dict(m=6, **DRAFT_MLP_UP), seed=41)
    with pytest.raises(AssertionError, match="both scales must be fp32"):
        with validating(op):
            op(x, w, x_scale, w_scale.to(torch.bfloat16))


def test_guard_refuses_mismatched_k() -> None:
    x, w, x_scale, w_scale = _build(dict(m=8, n=1024, k=3072), seed=43)
    narrow = x[:, :1024].contiguous()
    with pytest.raises(AssertionError, match="takes K from"):
        with validating(op):
            op(narrow, w, x_scale[:8].contiguous(), w_scale)


def test_guard_refuses_a_three_dimensional_input() -> None:
    x, w, x_scale, w_scale = _build(dict(m=6, **DRAFT_MLP_UP), seed=45)
    with pytest.raises(AssertionError, match="must be 2-D"):
        with validating(op):
            op(x.view(1, 6, -1), w, x_scale, w_scale)
