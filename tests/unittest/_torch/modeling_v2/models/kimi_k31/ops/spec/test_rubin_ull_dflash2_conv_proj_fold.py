# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the DFlash2 conv kernel projection with the side-0 convolution in its epilogue.

Every certified cell is gated bit for bit against the two launches it replaces: the vendored warp split-K GEMM of
the same tactic (the projection, all of ``[rows, 2 * taps * groups]``) and ``dflash2_grouped_conv`` of its side 0;
the projection is also held to the FP64 product. The fold is driven between the kernels that feed and read it in
the drafter (PDL on), captured in a CUDA graph, and its replay must equal the eager call.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_dflash2_conv_proj_fold import (  # noqa: E501
    PreparedConvProj,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_main import (
    module as vllm_main,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (  # noqa: E501
    dflash2_grouped_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_conv_proj_fold import (  # noqa: E501
    dflash2_conv_proj_fold as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

HIDDEN, TAPS, GROUP = 7168, 2, 16
N = 2 * TAPS * HIDDEN // GROUP
# The helix16 Rubin rows' warp split-K tactic of the [1792, 7168] kernel projection.
TACTIC = (16, 8, 256, 16, 2)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.fixture(scope="module")
def engines():
    g = torch.Generator(device="cuda").manual_seed(11)
    weight = (torch.randn(N, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(torch.bfloat16)
    w = vllm_main("cutedsl.gemm.warp_splitk")
    tactic = w.WarpSplitKTactic(*TACTIC)
    fold = PreparedConvProj(weight, tactic)
    plain = w.PreparedWarpSplitK(weight, {m: tactic for m in range(1, 9)})
    return fold, plain


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    base = (0.5 * torch.randn(TAPS, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return x, base


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _served(plain, x, base, block):
    rows = x.shape[0]
    coef = plain.run(x, torch.empty(rows, N, dtype=x.dtype, device=x.device))
    delta = coef.view(rows, 2, TAPS, HIDDEN // GROUP)[:, 0]
    return coef, dflash2_grouped_conv(x, delta, base, block, GROUP, True)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, engines) -> None:
    fold, plain = engines
    rows, block = cell.spec["rows"], cell.spec["block"]
    x, base = _inputs(rows, zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        coef, conv = op(fold, x, base, block)
    ref_coef, ref_conv = _served(plain, x, base, block)
    torch.cuda.synchronize()
    assert torch.equal(_bits(coef), _bits(ref_coef)), (
        "the projection is not the warp split-K's bits"
    )
    assert torch.equal(_bits(conv), _bits(ref_conv)), (
        "the convolution is not dflash2_grouped_conv's bits"
    )
    f64_coef, f64_conv = op.reference(fold, x, base, block)
    err = (coef.double() - f64_coef.double()).abs().max() / f64_coef.double().abs().max()
    assert err < 2e-2, f"projection max-rel error {err:.3e} against the FP64 product"
    err = (conv.double() - f64_conv.double()).abs().max() / f64_conv.double().abs().max()
    assert err < 2e-2, f"convolution max-rel error {err:.3e} against the FP64 path"


def test_graph_replay_equals_eager_between_pdl_neighbours(engines) -> None:
    """norm -> fold -> next GEMM (each a PDL launch) captured in a graph: the replay's bits are the eager ones."""
    fold, plain = engines
    x0, base = _inputs(8, 5)

    def chain():
        x = x0 * 1.0  # the producer kernel the fold waits on
        coef, conv = op(fold, x, base, 8)
        nxt = plain.run(conv, torch.empty(8, N, dtype=conv.dtype, device=conv.device))
        return coef, conv, nxt

    eager = [t.clone() for t in chain()]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(outs, eager):
        assert torch.equal(_bits(got), _bits(want))


def test_is_valid_refuses_more_rows_than_a_token_tile(engines) -> None:
    fold, _ = engines
    x, base = _inputs(9, 3)
    with pytest.raises(AssertionError, match="rows"):
        op.is_valid(fold, x, base, 9)
