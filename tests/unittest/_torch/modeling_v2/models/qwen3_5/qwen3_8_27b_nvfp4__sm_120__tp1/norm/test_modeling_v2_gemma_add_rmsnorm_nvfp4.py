# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 gemma_add_rmsnorm_nvfp4 entry.

Two gates per cell:

1. Bitwise against the in-tree composition it replaces -- the stock gemma add+norm
   kernel, then the `fp4_quantize` kernel on its BF16 output: the residual, every FP4
   byte and the whole SF buffer, padding rows included. `x` is read only.
2. Against the entry's fp32 reference, through the stock BF16 output `y`: the residual
   bitwise; `y` in the stock norm's band (torch's default bf16 band); and wherever a
   16-element block of `y` agrees bitwise with the reference's, that block's scale byte
   exactly and its codes outside `fp4_quantize`'s reciprocal tie window. A block whose
   `y` differs by an ulp can quantize differently and is held by the `y` band and gate
   1 instead. Such blocks are rare: the reference and the kernel differ only in the row
   sum's order and `rsqrt.approx` (~2^-19 relative on `rstd`), which flips a bf16
   rounding of `y` with probability ~2^-11 an element, ~1% of 16-element blocks; at
   least 90% of the blocks must agree.

Then the properties: every output byte written (buffers pre-filled with 0xFF), CUDA-graph
replay, and the per-row global-scale guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm._gemma_add_rmsnorm_nvfp4 import (  # noqa: E501
    gemma_fused_add_rmsnorm_nvfp4_cute,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.flashinfer_gemma_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_gemma_fused_add_rmsnorm as stock,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.gemma_add_rmsnorm_nvfp4 import (  # noqa: E501
    gemma_add_rmsnorm_nvfp4 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """`(x, residual, weight, eps, global_scale)`: rows at magnitudes spread over e^[-6, 5],
    one all-zero and one all -0.0 16-column block per row (a zero block quantizes to scale 0
    and code +-0), a weight with entries at exactly -1 (so `1 + w == 0`) and at x20."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    scale = torch.empty(rows, 1, device="cuda").uniform_(-6.0, 5.0, generator=g).exp()
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * scale).to(torch.bfloat16)
    residual = (torch.randn(rows, hidden, generator=g, device="cuda") * 4 * scale).to(
        torch.bfloat16
    )
    x[:, 32:48] = 0.0
    residual[:, 32:48] = 0.0
    x[:, 80:96] = -0.0
    residual[:, 80:96] = -0.0
    weight = torch.randn(hidden, generator=g, device="cuda") * 0.3
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] = -1.0
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] *= 20.0
    gs = torch.tensor([spec["global_scale"]], dtype=torch.float32, device="cuda")
    return x, residual, weight.to(torch.bfloat16), spec["eps"], gs


def _composition(x, residual, weight, eps, gs):
    """The in-tree pair this kernel replaces, on copies: `(y, residual_out, fp4, sf)`."""
    y, h = stock(x.clone(), residual.clone(), weight, eps)
    fp4, sf = torch.ops.trtllm.fp4_quantize(y, gs, 16, False, True)
    return y, h, fp4, sf


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble), with -0 (code
    8) read as +0: a `y` of exactly 0 (the `1 + w == 0` columns, the zero blocks) carries
    the sign of `h`, and which zero the hardware conversion emits is not what this gate is
    about -- gate 1 holds the bytes."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    return torch.where(codes == 8, torch.zeros_like(codes), codes)


def _scale_bytes(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The 128x4-swizzled scale buffer as `[rows, cols]` bytes (padding rows dropped)."""
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    padded_cols = (cols + 3) // 4 * 4
    index = (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    return sf[index]


def _check_against_reference(fp4, sf, y, ref, gs) -> None:
    """Gate 2's quantization half: exact on the blocks whose BF16 `y` matches the
    reference's bitwise, codes up to the tie window."""
    ref_fp4, ref_sf, y_ref = ref
    rows, hidden = y.shape
    cols = hidden // 16
    agree = (y.view(torch.int16) == y_ref.view(torch.int16)).view(rows, cols, 16).all(-1)
    assert agree.float().mean().item() >= 0.9, (
        f"only {agree.float().mean().item():.3f} of the blocks agree with the reference's y"
    )
    got_sf, want_sf = _scale_bytes(sf, rows, cols), _scale_bytes(ref_sf, rows, cols)
    assert torch.equal(got_sf[agree], want_sf[agree]), "scale bytes differ on agreeing blocks"
    tie = Fp4Quantize.near_tie(fp4_quantize.scaled(y_ref, gs, 16)).view(rows, cols, 16)
    diff = (_unpack(fp4) != _unpack(ref_fp4)).view(rows, cols, 16)
    stray = diff & ~tie & agree.unsqueeze(-1)
    assert not stray.any(), f"{stray.sum().item()} codes differ outside the tie window"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, residual, weight, eps, gs = _build(cell.spec, seed=_seed(cell.why))
    rows, hidden = x.shape
    x_before = x.clone()
    y, h, want_fp4, want_sf = _composition(x, residual, weight, eps, gs)
    ref_fp4, ref_sf, ref_residual = op.reference(x, residual, weight, eps, gs)
    y_ref, _ = stock.reference(x, residual, weight, eps)

    fp4, sf, out_residual = op(x, residual, weight, eps, gs)

    assert out_residual is residual, "the residual is written in place and handed back"
    assert torch.equal(x, x_before), "x is read only"
    assert fp4.dtype == torch.uint8 and fp4.shape == (rows, hidden // 2)
    assert sf.dtype == torch.uint8 and sf.shape == want_sf.shape == ref_sf.shape
    # Gate 1: bitwise the in-tree composition.
    assert torch.equal(residual.view(torch.int16), h.view(torch.int16)), "residual"
    assert torch.equal(fp4, want_fp4), "FP4 bytes differ from fp4_quantize's"
    assert torch.equal(sf, want_sf), "SF buffer (padding included) differs from fp4_quantize's"
    # Gate 2: the fp32 reference.
    assert torch.equal(residual.view(torch.int16), ref_residual.view(torch.int16))
    torch.testing.assert_close(y, y_ref)
    _check_against_reference(fp4, sf, y, (ref_fp4, ref_sf, y_ref), gs)


@pytest.mark.parametrize("rows", [33, 129], ids=["33 rows", "129 rows"])
def test_every_output_byte_is_written(rows) -> None:
    """FP4 and SF buffers pre-filled with 0xFF come back equal to the composition: no byte,
    SF padding rows included, is left to whatever the allocator held."""
    spec = dict(op.CELLS[0].spec, rows=rows)
    x, residual, weight, eps, gs = _build(spec, seed=_seed(f"poison {rows}"))
    _, _, want_fp4, want_sf = _composition(x, residual, weight, eps, gs)
    fp4 = torch.full_like(want_fp4, 0xFF)
    sf = torch.full_like(want_sf, 0xFF)
    gemma_fused_add_rmsnorm_nvfp4_cute(x, residual, weight, gs, eps, True, fp4_out=fp4, sf_out=sf)
    assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)


def test_cuda_graph_replay() -> None:
    """The decode graph at 33 rows: replays with fresh inputs (and the residual rewritten
    before each) equal the composition."""
    spec = op.CELLS[0].spec
    x, residual, weight, eps, gs = _build(spec, seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, residual.clone(), weight, eps, gs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp4, sf, _ = op(x, residual, weight, eps, gs)

    for replay in range(3):
        fresh_x, fresh_residual, _, _, _ = _build(spec, seed=_seed(f"replay {replay}"))
        _, want_h, want_fp4, want_sf = _composition(fresh_x, fresh_residual, weight, eps, gs)
        x.copy_(fresh_x)
        residual.copy_(fresh_residual)
        fp4.fill_(0xFF)
        sf.fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(residual.view(torch.int16), want_h.view(torch.int16))
        assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf), f"replay {replay}"


def test_guard_refuses_a_per_row_global_scale() -> None:
    """The epilogue reads `global_scale[0]` only; a per-row scale would be applied as its
    first element everywhere."""
    x, residual, weight, eps, gs = _build(op.CELLS[0].spec, seed=_seed("guard"))
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(x, residual, weight, eps, gs.expand(x.shape[0]).contiguous())
