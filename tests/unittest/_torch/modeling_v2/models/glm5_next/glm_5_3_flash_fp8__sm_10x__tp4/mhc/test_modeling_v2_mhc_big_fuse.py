# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 mhc_big_fuse entry.

The inputs are the reference mixer product of the cell's streams, so the kernel and the reference
start from the same FP32 values and only this op's arithmetic is compared.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_big_fuse import (  # noqa: E501
    BLOCK_SIZE_OPTIONS,
    big_fuse_block_size,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_big_fuse import (  # noqa: E501
    mhc_big_fuse as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_gemm_sqrsum import (  # noqa: E501
    mhc_gemm_sqrsum,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".", "../../../.."]
from _glm53_mhc_inputs import generator, mixer, norm_weight, scalars, streams  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

#: Absolute band of the FP32 coefficients; see `_compare`.
COEFF_ATOL = 1e-4


def _compare(out, ref, fused_norm: bool) -> None:
    """``post`` and ``comb`` within 1e-4 absolute; the layer input within 3 ulp per element of
    its row scale (4 with the fused norm) and 0.5 ulp relative RMS.

    The coefficients come from the same mixer product on both sides; the kernel's fast-math
    ``exp``, ``rsqrt`` and divisions are each a few FP32 ulp off, and the Sinkhorn tail repeats
    its divisions 39 times, so the two differ by about 1e-6 -- 1e-4 is two orders above that, on
    values in ``[0, 2]`` and ``[0, 1]``, and a misplaced scale, base or multiplier moves them by
    0.01 to 1. The layer input is rounded to BF16 at the same places on both sides (the collapse,
    and with the norm the normalized product), so it differs only where those FP32 differences
    flip a rounding: one BF16 step, at most 2 ulp of the row scale, or two steps with the norm
    when both roundings flip.
    """
    post, comb, layer_input = out
    ref_post, ref_comb, ref_layer_input = ref
    assert post.shape == ref_post.shape and comb.shape == ref_comb.shape
    torch.testing.assert_close(post, ref_post, rtol=0.0, atol=COEFF_ATOL)
    torch.testing.assert_close(comb, ref_comb, rtol=0.0, atol=COEFF_ATOL)
    element_ulp = 4.0 if fused_norm else 3.0
    assert_within_ulp(layer_input, ref_layer_input, element_ulp=element_ulp, rms_ulp=0.5)


def _matches(out, ref, fused_norm: bool) -> bool:
    try:
        _compare(out, ref, fused_norm)
    except AssertionError:
        return False
    return True


def _inputs(rows: int, norm: bool, seed: int, std: float = 1.0):
    gen = generator(seed)
    residual = streams(rows, gen, std)
    fn, scale, base = mixer(gen)
    mixes, sqrsum = mhc_gemm_sqrsum.reference(residual.view(rows, -1), fn, 0)
    return [mixes, sqrsum, residual, scale, base, norm_weight(gen) if norm else None]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    """Every block size, against the reference; the coefficients do not depend on it."""
    certifying(op)
    spec = cell.spec
    args = _inputs(spec["rows"], spec["norm"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    args += scalars(spec)
    ref = op.reference(*args, big_fuse_block_size(spec["rows"]))
    first = None
    for block_size in BLOCK_SIZE_OPTIONS:
        out = op(*args, block_size)
        _compare(out, ref, spec["norm"])
        if first is None:
            first = out
            continue
        assert torch.equal(out[0], first[0]) and torch.equal(out[1], first[1]), (
            f"block_size={block_size} changed the coefficients"
        )
        if not spec["norm"]:
            assert torch.equal(out[2], first[2]), f"block_size={block_size} changed the collapse"


def test_scalars_reach_their_kernel_parameters() -> None:
    """Distinct, exaggerated scalars on small streams, where every epsilon moves the outputs.

    GLM-5.3-Flash's two epsilon pairs are equal (``norm_eps`` and ``layer_input_eps`` 1e-5,
    ``hc_eps`` and ``sinkhorn_eps`` 1e-6) and its epsilons are too small to move unit-scale
    streams, so the cells cannot tell a swapped pair from a right one. Here the streams' mean
    square is 2.5e-3, the size of ``norm_eps``; the references with either pair swapped must fail
    the band, and the op must pass it.
    """
    args = _inputs(64, norm=True, seed=11, std=0.05)
    sinkhorn_iters, norm_eps, hc_eps, sinkhorn_eps, post_mult, layer_input_eps = (
        3,
        2.5e-3,
        0.25,
        0.05,
        1.5,
        1e-6,
    )
    right = (sinkhorn_iters, norm_eps, hc_eps, sinkhorn_eps, post_mult, layer_input_eps)
    with validating(op):
        out = op(*args, *right, 256)
    _compare(out, op.reference(*args, *right, 256), fused_norm=True)
    swapped = {
        "norm_eps and layer_input_eps": (
            sinkhorn_iters,
            layer_input_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        ),
        "hc_eps and sinkhorn_eps": (
            sinkhorn_iters,
            norm_eps,
            sinkhorn_eps,
            hc_eps,
            post_mult,
            layer_input_eps,
        ),
        "post_mult": (sinkhorn_iters, norm_eps, hc_eps, sinkhorn_eps, 2.0, layer_input_eps),
        "sinkhorn_iters": (20, norm_eps, hc_eps, sinkhorn_eps, post_mult, layer_input_eps),
    }
    for name, wrong in swapped.items():
        assert not _matches(out, op.reference(*args, *wrong, 256), fused_norm=True), (
            f"the band cannot tell {name} apart"
        )


def test_cuda_graph_capture_and_replay() -> None:
    """The decode step's first pre-mapping is captured: replay on new streams and products."""
    args = _inputs(4, norm=True, seed=31) + scalars(op.CELLS[0].spec)
    block_size = big_fuse_block_size(4)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args, block_size)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args, block_size)
    for captured, fresh in zip(args[:6], _inputs(4, norm=True, seed=32)):
        captured.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(*args, block_size), fused_norm=True)


def test_guard_refuses_what_the_kernel_misreads() -> None:
    """The kernel reads the product row-major and is compiled for 4 streams."""
    mixes, sqrsum, residual, scale, base, weight = _inputs(8, norm=True, seed=7)
    rest = [scale, base, weight, *scalars(op.CELLS[0].spec), 256]
    cases = {
        "mixes transposed in memory": [mixes.t().contiguous().t(), sqrsum, residual, *rest],
        "mixes stored [24, M]": [mixes.t().contiguous(), sqrsum, residual, *rest],
        "residual as 2 streams": [mixes, sqrsum, residual.view(8, 2, -1), *rest],
    }
    for name, args in cases.items():
        with pytest.raises(AssertionError):
            with validating(op):
                op(*args)
            pytest.fail(f"is_valid accepted {name}")
