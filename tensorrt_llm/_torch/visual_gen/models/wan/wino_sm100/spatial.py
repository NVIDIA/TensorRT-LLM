# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(4x4, 3x3) for the frame-pair convs on Blackwell (sm100), on the CuTe DSL
kernels of this package: input transform, one batched tcgen05 GEMM over the 36 coordinates of
every component (fp16 in, fp32 accumulate, fp16 out) on the static persistent schedule, output
transform. Interpolation points (0, +-3/2, +-5/8, inf). :func:`spatial_filters` transforms a
conv's filters; the conv caches the result on first use and recomputes it when its weight
tensor changes identity or version.

The fused F(2,3)-horizontal / direct-vertical conv for the 256-output-channel level lives in
:mod:`.fused_conv`; :func:`wino_spatial_conv_fused` is its per-component entry point.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def sm100_wino_available() -> bool:
    """Blackwell (sm100) and ``nvidia-cutlass-dsl`` installed."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import (  # noqa: F401
            fused_conv,
            gemm,
            transforms,
        )
    except ImportError:
        return False
    return True


def spatial_filters(g: list[torch.Tensor]) -> torch.Tensor:
    """``U [36, components, Cout, Cin]`` fp16: ``G g G^T`` per component filter ``g[k]
    [Cout, Cin, 3, 3]`` (fp32 math, rounded once), batch index ``coord * components + k``."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.transforms import WINO

    comps = len(g)
    cout, cin = g[0].shape[:2]
    gmat = torch.tensor(WINO[4]["G"], dtype=torch.float32, device=g[0].device)
    u = torch.einsum("ia,kocab,jb->ijkoc", gmat, torch.stack(g).float(), gmat)
    return u.reshape(36, comps, cout, cin).to(torch.float16).contiguous()


def spatial_eligible(cin: int, cout: int) -> bool:
    """Channel counts the GEMM and the transforms accept (K tiles of 64, 256-row Cout tiles)."""
    return cin % 64 == 0 and cout % 256 == 0


def wino_spatial_conv(d: torch.Tensor, u: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """3x3 conv (padding 1) of the component inputs by their filters: ``D [components, pairs,
    H*W, Cin]`` (bf16 or fp16) -> ``M [components, pairs, H*W, Cout]`` (bias-free, in ``D``'s
    dtype), ``u`` from :func:`spatial_filters`. ``width`` must be a multiple of 4."""
    comps, pairs, positions, cin = d.shape
    cout = u.shape[2]
    x = d.view(comps * pairs, height, width, cin)
    y = torch.empty((comps * pairs, height, width, cout), dtype=d.dtype, device=d.device)
    _run(x, u, y, comps, pairs)
    return y.view(comps, pairs, positions, cout)


_compiled: dict = {}


def _transform(name: str, jit_fn, tensors: list[torch.Tensor], consts: tuple):
    """Compile-once launcher for the transform kernels (layouts and element types are baked
    into the code, so the key carries shapes, strides and dtypes)."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    key = (name, *((tuple(t.shape), t.stride(), t.dtype) for t in tensors), *consts)
    args = [from_dlpack(t, assumed_align=16) for t in tensors]
    fn = _compiled.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *args, *consts, make_fake_stream())
        _compiled[key] = fn
    return lambda stream: fn(*args, stream)


def _gemm_n_tile(tiles: int) -> int:
    """GEMM N tile over the Winograd tiles: least padding, ties to the larger tile."""
    return min((-(-tiles // n) * n, -n, n) for n in (256, 224, 192))[2]


def _gemm(a, b, c, n_tile: int):
    """The batched GEMM ``C[l] = A[l] B[l]^T`` (M = Cout, N = tiles, K = Cin, batched over the
    coordinates x components) on the static persistent schedule, compiled once per N tile."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import make_fake_stream

    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.gemm import PersistentDenseGemmKernel

    key = ("gemm", n_tile)
    fn = _compiled.get(key)
    if fn is None:
        max_clusters = utils.HardwareInfo().get_max_active_clusters(2)
        op = PersistentDenseGemmKernel(cutlass.Float32, True, (256, n_tile), (2, 1), True)
        fn = cute.compile(op, a, b, c, max_clusters, make_fake_stream())
        _compiled[key] = fn
    return fn


def _run(x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, comps: int, pairs: int) -> None:
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import transforms

    frames, height, width, cin = x.shape
    cout = u.shape[2]
    tiles = pairs * -(-height // 4) * (width // 4)
    dev = x.device
    v = torch.empty((36, comps * tiles, cin), dtype=torch.float16, device=dev)
    m = torch.empty((36, comps * tiles, cout), dtype=torch.float16, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _transform("in", transforms.in_xform, [x, v], (frames, height, width, cin, 4, 4, 0))(stream)
    a = from_dlpack(u.view(36 * comps, cout, cin).permute(1, 2, 0), assumed_align=16)
    b = from_dlpack(v.view(36 * comps, tiles, cin).permute(1, 2, 0), assumed_align=16)
    c = from_dlpack(m.view(36 * comps, tiles, cout).permute(2, 1, 0), assumed_align=16)
    a = a.mark_layout_dynamic(leading_dim=1)
    b = b.mark_layout_dynamic(leading_dim=1)
    c = c.mark_layout_dynamic(leading_dim=0)
    _gemm(a, b, c, _gemm_n_tile(tiles))(a, b, c, stream)
    _transform("out", transforms.out_xform, [m, y], (frames, height, width, cout, 4, 4, 0))(stream)


def fused_eligible(cin: int, cout: int, height: int, width: int) -> bool:
    """Shapes the fused conv covers: 256 output channels, at least two 32-channel input
    chunks (the MMA schedule peels a first and a last chunk), 8-row and 32-pixel tiles."""
    return cout == 256 and cin >= 64 and cin % 32 == 0 and height % 8 == 0 and width % 32 == 0


def fused_filters(g: list[torch.Tensor]) -> list[torch.Tensor]:
    """Transformed filters for :func:`wino_spatial_conv_fused`, one per component filter
    ``g[k] [Cout, Cin, 3, 3]``."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import fused_conv

    return [fused_conv.prepare(gk.contiguous().to(torch.bfloat16)) for gk in g]


def wino_spatial_conv_fused(
    d: torch.Tensor, filters: list[torch.Tensor], height: int, width: int
) -> torch.Tensor:
    """The fused F(2,3)-horizontal / direct-vertical 3x3 conv per component: ``D [components,
    pairs, H*W, Cin]`` (bf16 or fp16) -> ``M [components, pairs, H*W, 256]`` in ``D``'s dtype,
    one stacked buffer (the consumer norm kernel reads the components at one stride)."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import fused_conv

    comps, pairs, positions, cin = d.shape
    cout = filters[0].shape[0]
    y = torch.empty((comps, pairs, height, width, cout), dtype=d.dtype, device=d.device)
    x = d.view(comps, pairs, height, width, cin)
    for k in range(comps):
        fused_conv.run_prepared(x[k], filters[k], y[k])
    return y.view(comps, pairs, positions, cout)
