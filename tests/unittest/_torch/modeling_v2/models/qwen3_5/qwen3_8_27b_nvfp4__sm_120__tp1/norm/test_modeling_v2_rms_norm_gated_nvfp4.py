# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 rms_norm_gated_nvfp4 entry.

Two gates per cell:

1. Bitwise against the in-tree composition it replaces -- the registered multi-row gated
   norm op (`torch.ops.trtllm.rms_norm_gated_token_major`, 4 rows a program, bf16 out;
   not the catalog entry, which switches to 8 rows a program from 32768 rows) then
   `fp4_quantize` on its output: every FP4 byte and the whole SF buffer, padding rows
   included.
2. Against the entry's fp32 reference, through the composition's bf16 `y`: `y` in the
   gated norm's band (torch's default bf16 band); and wherever a 16-element block of `y`
   agrees bitwise with the reference's, that block's scale byte exactly and its codes
   outside `fp4_quantize`'s reciprocal tie window. A block whose `y` differs by an ulp
   can quantize differently and is held by the `y` band and gate 1 instead. The kernel
   and the reference differ only in the row's sum order and `1 / sqrt` (~2^-20
   relative on `rstd`), which flips a bf16 rounding of `y` with probability ~2^-11 an
   element, ~1% of the blocks; at least 90% must agree.

Then the properties: outputs drawn from a poisoned allocator, PDL behind a late
producer of `x`, CUDA-graph replay, and the global-scale guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_nvfp4 import (  # noqa: E501
    rms_norm_gated_nvfp4 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major as gated_norm,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import capture, late_producer  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
#: in_proj_qkvz's output width and where its z columns start.
_QKVZ, _Z_OFFSET = 16384, 10240


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """`(x, z, weight, eps, global_scale)`: x the contiguous `[T * 48, 128]` recurrence
    output at magnitudes spread over e^[-4, 3] per token, one all-zero 16-column block per
    head row; z the in_proj_qkvz column view."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    t, heads, n = spec["tokens"], spec["heads"], spec["width"]
    scale = torch.empty(t, 1, 1, device=DEV).uniform_(-4.0, 3.0, generator=g).exp()
    x = (torch.randn(t, heads, n, generator=g, device=DEV) * scale).to(torch.bfloat16)
    x[:, :, 32:48] = 0.0
    x = x.view(t * heads, n)
    qkvz = (torch.randn(t, _QKVZ, generator=g, device=DEV) * 2).to(torch.bfloat16)
    z = qkvz[:, _Z_OFFSET : _Z_OFFSET + heads * n].view(t, heads, n)
    weight = (1.0 + 0.3 * torch.randn(n, generator=g, device=DEV)).to(torch.bfloat16)
    gs = torch.tensor([spec["global_scale"]], dtype=torch.float32, device=DEV)
    return x, z, weight, spec["eps"], gs


def _composition(x, z, weight, eps, gs):
    """The in-tree pair this kernel replaces: `(y, fp4, sf)`. The norm is the registered
    op (registered by the gated-norm entry's import), whose 4-row programs this kernel
    copies at every row count."""
    y = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, "silu")
    fp4, sf = fp4_quantize(y.view(z.shape[0], -1), gs, 16, False, True)
    return y, fp4, sf


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble), -0 read as +0:
    which zero the conversion emits for an exact 0 is gate 1's business."""
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


def _check_against_reference(fp4, sf, y, ref, y_ref, gs) -> None:
    """Gate 2's quantization half: exact on the blocks whose bf16 `y` matches the
    reference's bitwise, codes up to the tie window."""
    ref_fp4, ref_sf = ref
    tokens = fp4.shape[0]
    y, y_ref = y.view(tokens, -1), y_ref.view(tokens, -1)
    cols = y.shape[1] // 16
    agree = (y.view(torch.int16) == y_ref.view(torch.int16)).view(tokens, cols, 16).all(-1)
    assert agree.float().mean().item() >= 0.9, (
        f"only {agree.float().mean().item():.3f} of the blocks agree with the reference's y"
    )
    got_sf, want_sf = _scale_bytes(sf, tokens, cols), _scale_bytes(ref_sf, tokens, cols)
    assert torch.equal(got_sf[agree], want_sf[agree]), "scale bytes differ on agreeing blocks"
    tie = Fp4Quantize.near_tie(fp4_quantize.scaled(y_ref, gs, 16)).view(tokens, cols, 16)
    diff = (_unpack(fp4) != _unpack(ref_fp4)).view(tokens, cols, 16)
    stray = diff & ~tie & agree.unsqueeze(-1)
    assert not stray.any(), f"{stray.sum().item()} codes differ outside the tie window"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, z, weight, eps, gs = _build(cell.spec, seed=_seed(cell.why))
    before = [x.clone(), z.clone()]
    y, want_fp4, want_sf = _composition(x, z, weight, eps, gs)
    ref = op.reference(x, z, weight, eps, gs)
    y_ref = gated_norm.reference(x, z, weight, eps, None, "silu")

    fp4, sf = op(x, z, weight, eps, gs)

    tokens = cell.spec["tokens"]
    assert torch.equal(x, before[0]) and torch.equal(z, before[1]), "inputs are read only"
    assert fp4.dtype == torch.uint8 and fp4.shape == (tokens, 48 * 128 // 2)
    assert sf.dtype == torch.uint8 and sf.shape == want_sf.shape == ref[1].shape
    # Gate 1: bitwise the in-tree composition.
    assert torch.equal(fp4, want_fp4), "FP4 bytes differ from fp4_quantize's"
    assert torch.equal(sf, want_sf), "SF buffer (padding included) differs from fp4_quantize's"
    # Gate 2: the fp32 reference.
    torch.testing.assert_close(y, y_ref)
    _check_against_reference(fp4, sf, y, ref, y_ref, gs)


def test_outputs_from_poisoned_storage(monkeypatch) -> None:
    """Every output byte is written, SF padding rows included: with both outputs handed
    out pre-filled with 0xAB, they come back equal to the composition."""
    x, z, weight, eps, gs = _build(op.CELLS[0].spec, seed=_seed("poison"))
    _, want_fp4, want_sf = _composition(x, z, weight, eps, gs)
    empty, handed = torch.empty, []

    def poisoned_empty(*args, **kwargs):
        t = empty(*args, **kwargs)
        if t.dtype == torch.uint8 and t.is_cuda:
            t.fill_(0xAB)
            handed.append(t)
        return t

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    fp4, sf = op(x, z, weight, eps, gs)
    monkeypatch.undo()
    assert any(t is fp4 for t in handed) and any(t is sf for t in handed), "outputs not poisoned"
    assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)


def test_pdl_waits_for_a_late_producer() -> None:
    """With PDL the kernel may start before its producer finishes; every load sits behind
    `griddepcontrol.wait`. A producer that releases its dependents at once and writes `x`
    only after spinning, into a NaN-filled buffer: the result equals the composition on
    the final `x`."""
    x, z, weight, eps, gs = _build(op.CELLS[0].spec, seed=_seed("pdl"))
    _, want_fp4, want_sf = _composition(x, z, weight, eps, gs)
    late_x = torch.full_like(x, float("nan"))
    # Compile both kernels first, so the timed launch is a host dispatch of a few
    # microseconds and the producer, spinning 2 ms, is still running when the consumer
    # launches.
    late_producer(x, late_x)
    op(late_x, z, weight, eps, gs, True)
    torch.cuda.synchronize()
    late_x.fill_(float("nan"))
    late_producer(x, late_x, delay_us=2000)
    fp4, sf = op(late_x, z, weight, eps, gs, True)
    torch.cuda.synchronize()
    assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)


def test_cuda_graph_replay() -> None:
    """The decode graph at 128 rows: replays with fresh `x` and `z` equal the composition."""
    spec = op.CELLS[1].spec
    x, z, weight, eps, gs = _build(spec, seed=_seed("graph"))
    graph, (fp4, sf) = capture(lambda: op(x, z, weight, eps, gs))
    for replay in range(3):
        fresh_x, fresh_z, _, _, _ = _build(spec, seed=_seed(f"replay {replay}"))
        _, want_fp4, want_sf = _composition(fresh_x, fresh_z, weight, eps, gs)
        x.copy_(fresh_x)
        z.copy_(fresh_z)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf), f"replay {replay}"


def test_guard_refuses_a_per_token_global_scale() -> None:
    """The kernel loads `global_scale[0]` only; a per-token scale is refused."""
    x, z, weight, eps, gs = _build(op.CELLS[0].spec, seed=_seed("guard"))
    with pytest.raises(AssertionError, match="one fp32 element"):
        with validating(op):
            op(x, z, weight, eps, gs.expand(z.shape[0]).contiguous())
