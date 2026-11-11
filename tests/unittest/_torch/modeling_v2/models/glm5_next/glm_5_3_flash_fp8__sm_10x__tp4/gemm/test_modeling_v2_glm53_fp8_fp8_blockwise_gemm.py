# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_blockwise_gemm operation of the glm5_next FP8 target.

The entry owns the contract. This file turns a cell into block-FP8 operands (drawn directly as
e4m3 codes and fp32 scales, so both sides read bit-identical values), runs the block-FP8
`Linear` composition once end to end from bf16 rows, and drives the guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.fp8_blockwise_gemm import (  # noqa: E501
    BLOCK,
    GLM_DENSE_DOWN,
    GLM_DENSE_GATE_UP,
    GLM_SHARED_DOWN,
    GLM_SHARED_GATE_UP,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.fp8_blockwise_gemm import (  # noqa: E501
    fp8_blockwise_gemm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="fp8_blockwise_gemm is certified on sm_100"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 bf16 ulp of the row max element-wise, 2 ulp of relative RMS.

    Kernel and reference read bit-identical e4m3 codes and fp32 scales; the products of codes
    are exact in fp32, and the two differ only in where the scales are applied (per 128-wide K
    block in the kernel, before the sum in the reference) and in the order of the fp32 sums --
    a few fp32 ulp of the row scale, far below one bf16 ulp. What remains is the single bf16
    rounding on each side landing on different neighbours, one ulp of an element at most. A
    misread scale layout (a transposed activation scale, a 64-column block offset) or a wrong
    K block is off by a whole block's contribution, many ulp RMS.
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


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


# The shapes the target calls this operation with, on top of the operation's own `CELLS`.
#
# GLM-5.3-Flash at TP4 runs four block-FP8 FFN projections -- the three dense layers' MLP and
# every MoE layer's shared expert, each a gate_up and a down -- and, in the sequence-parallel
# MLA prefill, q_b and o over all 64 heads for a rank's own rows. Generation steps are MTP-3
# verifies of 4 rows per request, up to 64 requests; a 13,312-row prefill runs the FFN GEMMs
# over every gathered row and the MLA ones over a rank's quarter (3,328 rows).
_FFN = (
    ("dense gate_up", GLM_DENSE_GATE_UP),
    ("dense down", GLM_DENSE_DOWN),
    ("shared gate_up", GLM_SHARED_GATE_UP),
    ("shared down", GLM_SHARED_DOWN),
)
_ROWS = (
    (4, "MTP-3 verify of one request"),
    (256, "MTP-3 verify of 64 requests"),
    (13312, "a 13,312-row prefill step"),
)

CELLS: tuple[Cell, ...] = op.CELLS + (
    *(
        Cell(
            why=f"{name} {shape['n']}x{shape['k']}, {m} rows: {what}",
            spec=dict(m=m, **shape),
        )
        for name, shape in _FFN
        for m, what in _ROWS
    ),
    Cell(
        why="sequence-parallel MLA prefill q_b over 64 heads (16384 x 1536), 3,328 rows",
        spec=dict(m=3328, n=16384, k=1536),
    ),
    Cell(
        why="sequence-parallel MLA prefill o over 64 heads (4096 x 16384), 3,328 rows",
        spec=dict(m=3328, n=4096, k=16384),
    ),
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    x, w, x_scale, w_scale = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, w, x_scale, w_scale)
    assert out.shape == (cell.spec["m"], cell.spec["n"]) and out.is_contiguous()
    _compare(out, op.reference(x, w, x_scale, w_scale))


def test_block_fp8_linear_from_bf16_rows() -> None:
    """The composition a block-FP8 `Linear` runs: the op on `fp8_quantize_1x128`'s output,
    passed straight through. Against the entry's reference on that same quantization it holds
    the certified band, which is what shows the quantizer's scale layout is the one the GEMM
    reads."""
    g = torch.Generator(device=DEV).manual_seed(23)
    rows = torch.randn(64, 4096, generator=g, device=DEV).to(torch.bfloat16)
    x, x_scale = fp8_quantize_1x128(rows)
    _, w, _, w_scale = _build(dict(m=1, **GLM_SHARED_GATE_UP), seed=29)
    with validating(op):
        out = op(x, w, x_scale, w_scale)
    _compare(out, op.reference(x, w, x_scale, w_scale))


def test_tuned_tactic_under_cuda_graph_capture() -> None:
    """A decode step's FFN input as the target runs it: `fp8_quantize_1x128`, then the op at the
    tactic the autotuner picked in warmup, captured in a CUDA graph and replayed on new rows.
    Tuning profiles only the call's own row bucket. The replayed quantization equals an eager
    one bit for bit, and the product holds the certified band against the reference on it."""
    shape = GLM_SHARED_GATE_UP
    g = torch.Generator(device=DEV).manual_seed(47)
    rows = torch.randn(32, shape["k"], generator=g, device=DEV).to(torch.bfloat16)
    _, w, _, w_scale = _build(dict(m=1, **shape), seed=53)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        x, x_scale = fp8_quantize_1x128(rows)
        with autotune(skip_dynamic_tuning_buckets=True):
            op(x, w, x_scale, w_scale)
        assert any(
            key[0] == "trtllm::cute_dsl_fp8_gemm_blackwell::gemm"
            for key in tuner.profiling_cache.cache
        ), "the warmup tuned nothing"
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
            _compare(out, op.reference(x, w, x_scale, w_scale))
    finally:
        tuner.clear_cache()


def test_guard_refuses_a_transposed_activation_scale() -> None:
    """A token-major `[M, K/128]` scale where the shapes differ is refused before the kernel
    misreads it; at M == K/128 nothing can tell, which is why the entry says so."""
    x, w, x_scale, w_scale = _build(dict(m=8, **GLM_SHARED_GATE_UP), seed=31)
    with pytest.raises(AssertionError, match="K-block-major"):
        with validating(op):
            op(x, w, x_scale.t().contiguous(), w_scale)


def test_guard_refuses_a_strided_scale_slice() -> None:
    """A row slice of the activation needs a contiguous copy of its scale columns."""
    x, w, x_scale, w_scale = _build(dict(m=16, **GLM_SHARED_GATE_UP), seed=37)
    with pytest.raises(AssertionError, match="input_scale must be contiguous"):
        with validating(op):
            op(x[4:12], w, x_scale[:, 4:12], w_scale)


def test_guard_refuses_bf16_scales() -> None:
    x, w, x_scale, w_scale = _build(dict(m=8, **GLM_SHARED_GATE_UP), seed=41)
    with pytest.raises(AssertionError, match="both scales must be fp32"):
        with validating(op):
            op(x, w, x_scale, w_scale.to(torch.bfloat16))


def test_guard_refuses_mismatched_k() -> None:
    x, w, x_scale, w_scale = _build(dict(m=8, n=1024, k=4096), seed=43)
    narrow = x[:, :3072].contiguous()
    with pytest.raises(AssertionError, match="takes K from"):
        with validating(op):
            op(narrow, w, x_scale[:24].contiguous(), w_scale)
