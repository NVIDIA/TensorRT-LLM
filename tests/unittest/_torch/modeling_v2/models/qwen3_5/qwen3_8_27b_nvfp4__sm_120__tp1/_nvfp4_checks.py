# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks shared by this target's GEMM / quantization / activation entry tests.

* `assert_nvfp4_matches_reference`: an NVFP4 output (packed codes + swizzled scales)
  against `fp4_quantize`'s native-torch quantization of the reference intermediate,
  in the band the fused kernels are certified in (stated there).
* `late_producer`: a PDL producer that releases its dependents at once and writes
  the consumer's input only after spinning, so a consumer that read its input before
  `griddepcontrol.wait` would read the NaN fill.
* `capture`: warm a callable on a side stream, then capture it in a CUDA graph.
"""

from __future__ import annotations

import zlib

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization._nvfp4_reference import (  # noqa: E501
    SF_VEC,
    swizzled_sf_index,
    unpack_codes,
    unswizzle_sf,
)


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def assert_nvfp4_matches_reference(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
    rows: int,
    cols: int,
    *,
    max_sf_frac: float,
    max_code_frac: float = 0.01,
) -> None:
    """``out`` and ``ref`` are ``(packed [rows, cols/2], swizzled sf)``.

    The fused kernels compute their BF16 intermediate with Triton's sigmoid (or a
    tensor-core GEMM) and quantize it with the CUDA quantizer's approximate
    reciprocals; the reference uses torch's sigmoid (or an fp32 matmul) and
    round-to-nearest-even. So, in units the reference cannot predict:

    * scale padding: every byte outside the real rectangle zero, exactly;
    * scales: a block's E4M3 scale moves only when its maximum element's BF16 value
      moved by an ulp across an E4M3 rounding boundary -- rare, and then to the
      neighbouring code; at most ``max_sf_frac`` of the blocks;
    * codes (blocks with equal scales): ``fp4_quantize``'s own test measured 0.13% to
      0.24% of elements inside its reciprocal tie window and bounds them at 1%; a moved
      intermediate adds far fewer. A differing code is the neighbouring magnitude with
      the same sign; at most ``max_code_frac`` of the elements.

    A wrong column order, scale, or SwiGLU pairing moves a large fraction of codes by
    more than one step and fails the last two gates.
    """
    fp4, sf = out
    ref_fp4, ref_sf = ref
    assert fp4.dtype == torch.uint8 and fp4.shape == ref_fp4.shape, (fp4.shape, ref_fp4.shape)
    assert sf.numel() == ref_sf.numel(), (sf.numel(), ref_sf.numel())
    sf_cols = cols // SF_VEC
    flat, ref_flat = sf.reshape(-1).view(torch.uint8), ref_sf.reshape(-1).view(torch.uint8)
    padding = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
    padding[swizzled_sf_index(rows, sf_cols, flat.device).reshape(-1)] = False
    assert torch.equal(flat[padding], ref_flat[padding]), "scale padding bytes differ"

    lin, ref_lin = unswizzle_sf(sf, rows, sf_cols), unswizzle_sf(ref_sf, rows, sf_cols)
    sf_diff = lin != ref_lin
    sf_frac = sf_diff.float().mean().item()
    assert sf_frac <= max_sf_frac, f"{sf_frac:.2e} of the block scales differ (> {max_sf_frac:.0e})"
    gap = (lin[sf_diff].int() - ref_lin[sf_diff].int()).abs()
    assert bool((gap <= 1).all()), "a block scale differs by more than one E4M3 code"

    codes, ref_codes = unpack_codes(fp4), unpack_codes(ref_fp4)
    same_scale = ~sf_diff.repeat_interleave(SF_VEC, dim=1)
    diff = (codes != ref_codes) & same_scale
    code_frac = diff.float().mean().item()
    assert code_frac <= max_code_frac, (
        f"{code_frac:.2e} of the codes differ (> {max_code_frac:.0e})"
    )
    mag, ref_mag = (codes & 7).int(), (ref_codes & 7).int()
    same_sign = ((codes ^ ref_codes) & 8) == 0
    neighbour = ((mag - ref_mag).abs() <= 1) & (same_sign | ((mag == 0) & (ref_mag == 0)))
    assert bool(neighbour[diff].all()), "a code differs by more than one E2M1 step"


@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_producer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    # Wait for the poisoning fill, let dependents launch at once, spin, only then write.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    d = pid.to(tl.int64)
    t0 = _globaltimer(d)
    t = t0
    while t - t0 < delay_ns:
        t = _globaltimer(d)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def late_producer(src: torch.Tensor, dst: torch.Tensor, delay_us: int = 30) -> None:
    """Copy ``src`` into ``dst`` (same shape, contiguous) as a PDL primary that lets its
    dependents start immediately and stores only after ``delay_us``. Few CTAs, so most
    SMs stay free for the consumer's CTAs to launch early."""
    n = src.numel()
    block = 8192
    _late_producer_kernel[(triton.cdiv(n, block),)](
        src.reshape(-1),
        dst.reshape(-1),
        n,
        delay_us * 1000,
        BLOCK=block,
        num_warps=8,
        launch_pdl=True,
    )


def capture(fn):
    """Warm ``fn`` twice on a side stream, then capture one call; returns (graph, outputs)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = fn()
    return graph, outs
