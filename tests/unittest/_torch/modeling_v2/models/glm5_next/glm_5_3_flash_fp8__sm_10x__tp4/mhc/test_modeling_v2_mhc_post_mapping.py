# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 mhc_post_mapping entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_post_mapping import (  # noqa: E501
    mhc_post_mapping as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".", "../../../.."]
from _glm53_mhc_inputs import block_output, coefficients, generator, streams  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """3 ulp per element against the row scale, 0.5 ulp relative RMS.

    Kernel and reference form the same FP32 products and sums in the same order; the kernel fuses
    each multiply-add, so the two FP32 values differ by about one FP32 ulp, and their single BF16
    rounding differs only where that difference crosses a rounding boundary: rarely, and then by
    one BF16 step of the element, at most 2 ulp of its row's scale. A misread stream or a
    transposed ``comb`` is off by the streams' own size, hundreds of ulp.
    """
    assert_within_ulp(out, ref, element_ulp=3.0, rms_ulp=0.5)


def _inputs(rows: int, seed: int):
    gen = generator(seed)
    post, comb = coefficients(rows, gen)
    return block_output(rows, gen), streams(rows, gen), post, comb


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, residual, post, comb = _inputs(cell.spec["rows"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, residual, post, comb)
    assert out.shape == residual.shape and out.dtype == torch.bfloat16
    assert out.data_ptr() != residual.data_ptr(), "the op must return new streams"
    _compare(out, op.reference(x, residual, post, comb))
    # The model keeps post as [M, 4, 1]: the same bytes.
    assert torch.equal(op(x, residual, post.unsqueeze(-1), comb), out)


def test_cuda_graph_capture_and_replay() -> None:
    """The last boundary of a captured decode step replays on new contents of its inputs."""
    x, residual, post, comb = _inputs(4, seed=31)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, residual, post, comb)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, residual, post, comb)
    for captured, fresh in zip((x, residual, post, comb), _inputs(4, seed=32)):
        captured.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(x, residual, post, comb))


def test_guard_refuses_what_the_kernel_misreads() -> None:
    """Strided coefficients are read as dense rows, and a launch of no rows fails later."""
    x, residual, post, comb = _inputs(8, seed=7)
    wide = torch.zeros(8, 2 * 4, device="cuda")
    cases = {
        "post a column slice": (x, residual, wide[:, :4], comb),
        "comb transposed": (x, residual, post, comb.transpose(1, 2)),
        "no rows": (x[:0], residual[:0], post[:0], comb[:0]),
    }
    for name, args in cases.items():
        with pytest.raises(AssertionError):
            with validating(op):
                op(*args)
            pytest.fail(f"is_valid accepted {name}")
