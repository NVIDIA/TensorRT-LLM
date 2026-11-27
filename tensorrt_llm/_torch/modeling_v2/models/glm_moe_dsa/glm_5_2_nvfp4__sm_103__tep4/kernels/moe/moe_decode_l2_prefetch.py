# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode MoE: L2 prefetch of the routed experts' FC1 weights ahead of GG1

A decode GG1 (``moe_gather_grouped_gemm_act_fusion``, tiler 128x256, raster M) streams ~600-750 MB of NVFP4
weights per EP rank and layer from DRAM. The pre-MoE all-reduce and the router that precede it leave DRAM
mostly idle. This kernel is issued in that window, on a side stream forked at the o_proj end. It pulls into L2
the weight slices GG1 consumes first: GG1's first waves are N-tile-major over the active experts, so the order
is N-tiles ``0..NTILES-1`` of the ``TOP_E`` experts that the same layer's previous decode step routed the most
tokens to (``expert_hist``, written by ``moe_topk_routing_push``).

Each issuing lane runs one ``cp.async.bulk.prefetch.L2.global`` of ``_CHUNK`` bytes. The instruction has no
destination, no completion token and no SMEM, so the kernel writes nothing and changes no result. A CTA retires
when its prefetches land (about 170-250 GB/s per SM). 128 one-warp CTAs move 64 MiB in ~10 us on a B300.

Layout of one layer's FC1 operands (the target's ``fc1_w`` / ``fc1_s``, the GG1 B / SFB):
  weights  ``[64 experts, 2*I = 4096 rows, H/2 = 3072 B]``: the (expert, N-tile) slice is 256 rows = 786432 B
  scales   ``[64, 2*I/128 = 32, H/64 = 96, 512 B]``: the (expert, N-tile) slice is 2 row blocks = 98304 B
Expert strides are runtime arguments (bytes), so a stacked or a per-expert-strided store both work.
The weights' byte order inside a slice does not matter here: in the k-block layout (``moe_fc1_kblock``, the
target's default) each 128-row block is still one contiguous range, so a 256-row slice is the same 786432 B.

Selection, identical in every CTA (64 counts, one warp): expert ``e`` takes slot ``rank(e)``. Its rank is the
number of experts with a larger count, or the same count and a smaller index. Experts with a zero count take no
slot, so a layer whose histogram was never written (all zeros) issues nothing.

``hold_ns`` > 0 with ``NTILES == 0`` (a carveout-only form): no prefetch. Each CTA sleeps for ``hold_ns`` and exits.
Launched with the max shared-memory carveout preference while the pre-MoE all-reduce's CTAs arrive, the
resident CTAs keep their SMs configured for the large carveout, for the decode router's ~200 KiB multicast
clusters. That helped only when the fork was queued ahead of the all-reduce (top-k end -0.7 / -1.2 us at
cluster 8 / 2). Behind the served PDL chain o_proj -> all-reduce it did not, and the preference on the prefetch
form delayed the cluster-2 router by 17-30 us, so the target does not use that form; ``carveout`` is a launch
option.

Both forms are launched without PDL (no ``griddepcontrol``): their stream predecessor is the fork event after
o_proj, and they read only the previous step's counts and the load-time weights.
"""

from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
from cutlass.experimental import primitives as prims

_LOCAL_EXPERTS = 64
_WARP = 32
_HIDDEN = 6144
_INTER = 2048
_TILE_N = 256  # GG1 mma_tiler_n: one N-tile = 256 rows of the [2*I, H/2] FC1 weight
N_TILES = 2 * _INTER // _TILE_N  # 16 N-tiles per expert
W_SLICE = _TILE_N * _HIDDEN // 2  # 786432 B of NVFP4 weight per (expert, N-tile)
S_SLICE = (
    (_TILE_N // 128) * (_HIDDEN // 64) * 512
)  # 98304 B of e4m3 scale atoms per (expert, N-tile)
W_EXPERT = 2 * _INTER * _HIDDEN // 2  # 12582912 B: the stacked layout's expert stride
S_EXPERT = (2 * _INTER // 128) * (_HIDDEN // 64) * 512  # 1572864 B
# Bytes per cp.async.bulk.prefetch: divides both slices, a multiple of 16.
_CHUNK = 16384
if W_SLICE % _CHUNK or S_SLICE % _CHUNK or _CHUNK % 16:
    raise ValueError("_CHUNK must divide 98304 and be a multiple of 16")
_W_CHUNKS = W_SLICE // _CHUNK
_SLICE_CHUNKS = _W_CHUNKS + S_SLICE // _CHUNK
MAX_TOP_E = _WARP  # the slot table is initialised by one lane per slot
DEFAULT_CTAS = 64
_SPIN_NS = 256

PrefetchFn = Callable[..., None]
_CACHE: Dict[Tuple, PrefetchFn] = {}


@_dsl_user_op
def _l2_bulk_prefetch(addr, nbytes, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.prefetch.L2.global [addr], nbytes`` (GMEM -> L2, no destination, ``nbytes`` % 16 == 0)."""
    _llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@_dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None) -> None:
    """``nanosleep.u32 ns`` (the hold loop's back-off; keeps the resident warp off the issue slots)."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def decode_l2_prefetch_kernel(
    expert_hist: cute.Tensor,
    w_base: cutlass.Int64,
    w_expert_stride: cutlass.Int64,
    s_base: cutlass.Int64,
    s_expert_stride: cutlass.Int64,
    hold_ns: cutlass.Int64,
    TOP_E: cutlass.Constexpr,
    NTILES: cutlass.Constexpr,
    sel_out: Optional[cute.Tensor],
):
    """One warp per CTA: rank the 64 local experts by count, then bulk-prefetch the first N-tiles (module doc)."""
    lane, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(NTILES == 0):
        t0 = cute.arch.globaltimer()
        t = t0
        while t - t0 < hold_ns:
            _nanosleep(_SPIN_NS)
            t = cute.arch.globaltimer()
    else:
        sel = cutlass.Array(cutlass.Int32, MAX_TOP_E, space=cutlass.AddressSpace.smem, alignment=16)
        sel.subview(lane).store(cutlass.Int32(-1))
        c0 = cutlass.Int32(expert_hist[lane])
        c1 = cutlass.Int32(expert_hist[lane + cutlass.Int32(_WARP)])
        r0 = cutlass.Int32(0)
        r1 = cutlass.Int32(0)
        for j in cutlass.range_constexpr(_WARP):
            v0 = cute.arch.shuffle_sync(c0, j)  # expert j
            v1 = cute.arch.shuffle_sync(c1, j)  # expert j + 32
            # expert lane: ahead of it are larger counts, or equal counts at a smaller index
            if v0 > c0 or (v0 == c0 and cutlass.Int32(j) < lane):
                r0 = r0 + cutlass.Int32(1)
            if v1 > c0:
                r0 = r0 + cutlass.Int32(1)
            # expert lane + 32: every expert j < 32 has a smaller index
            if v0 >= c1:
                r1 = r1 + cutlass.Int32(1)
            if v1 > c1 or (v1 == c1 and cutlass.Int32(j) < lane):
                r1 = r1 + cutlass.Int32(1)
        prims.barrier_cta_sync(0)
        if c0 > cutlass.Int32(0) and r0 < cutlass.Int32(TOP_E):
            sel.subview(r0).store(lane)
        if c1 > cutlass.Int32(0) and r1 < cutlass.Int32(TOP_E):
            sel.subview(r1).store(lane + cutlass.Int32(_WARP))
        prims.barrier_cta_sync(0)
        if cutlass.const_expr(sel_out is not None):
            if cta == cutlass.Int32(0) and lane < cutlass.Int32(TOP_E):
                sel_out[lane] = sel.subview(lane).load()
        total = cutlass.Int32(NTILES * TOP_E * _SLICE_CHUNKS)
        c = cta * cutlass.Int32(_WARP) + lane
        step = grid * cutlass.Int32(_WARP)
        while c < total:
            sl = c // cutlass.Int32(_SLICE_CHUNKS)
            k = c - sl * cutlass.Int32(_SLICE_CHUNKS)
            n = sl // cutlass.Int32(TOP_E)
            r = sl - n * cutlass.Int32(TOP_E)
            e = sel.subview(r).load()
            if e >= cutlass.Int32(0):
                # the slice's scale chunk, unless k names one of its weight chunks (both defined before the
                # branch: the DSL carries a value out of a dynamic `if` only when it exists before it)
                addr = (
                    s_base
                    + cutlass.Int64(e) * s_expert_stride
                    + cutlass.Int64(n) * cutlass.Int64(S_SLICE)
                    + cutlass.Int64(k - cutlass.Int32(_W_CHUNKS)) * cutlass.Int64(_CHUNK)
                )
                if k < cutlass.Int32(_W_CHUNKS):
                    addr = (
                        w_base
                        + cutlass.Int64(e) * w_expert_stride
                        + cutlass.Int64(n) * cutlass.Int64(W_SLICE)
                        + cutlass.Int64(k) * cutlass.Int64(_CHUNK)
                    )
                _l2_bulk_prefetch(addr, _CHUNK)
            c = c + step


decode_l2_prefetch_kernel.set_name_prefix(
    "glm52_moe_decode_l2_prefetch", remove_cutlass_symbol=True
)


@cute.jit
def launch_decode_l2_prefetch(
    expert_hist: cute.Tensor,
    w_base: cutlass.Int64,
    w_expert_stride: cutlass.Int64,
    s_base: cutlass.Int64,
    s_expert_stride: cutlass.Int64,
    hold_ns: cutlass.Int64,
    stream: cuda.CUstream,
    top_e: cutlass.Constexpr[int],
    ntiles: cutlass.Constexpr[int],
    ctas: cutlass.Constexpr[int],
    carveout: cutlass.Constexpr[bool],
    sel_out: Optional[cute.Tensor] = None,
):
    decode_l2_prefetch_kernel(
        expert_hist,
        w_base,
        w_expert_stride,
        s_base,
        s_expert_stride,
        hold_ns,
        top_e,
        ntiles,
        sel_out,
    ).launch(
        grid=(ctas, 1, 1),
        block=(_WARP, 1, 1),
        preferred_smem_carveout=100 if carveout else None,
        stream=stream,
    )


def _check(top_e: int, ntiles: int, ctas: int) -> None:
    if not 0 <= ntiles <= N_TILES:
        raise ValueError(f"ntiles must be in [0, {N_TILES}], got {ntiles}")
    if ntiles and not 1 <= top_e <= MAX_TOP_E:
        raise ValueError(f"top_e must be in [1, {MAX_TOP_E}], got {top_e}")
    if not 1 <= ctas <= 1024:
        raise ValueError(f"ctas must be in [1, 1024], got {ctas}")


def compile_prefetch(
    *,
    top_e: int,
    ntiles: int,
    ctas: int = DEFAULT_CTAS,
    carveout: bool = False,
    debug_sel: bool = False,
    device: Optional[int] = None,
) -> PrefetchFn:
    """Compile once per (device, top_e, ntiles, ctas, carveout, debug) the handle
    ``fn(expert_hist, w_base, w_stride, s_base, s_stride, hold_ns[, sel_out])`` (launches on the current stream)."""
    _check(top_e, ntiles, ctas)
    device = torch.cuda.current_device() if device is None else int(device)
    key = (
        device,
        int(top_e) if ntiles else 0,
        int(ntiles),
        int(ctas),
        bool(carveout),
        bool(debug_sel),
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        fake = cute.runtime.make_fake_compact_tensor
        hist = fake(cutlass.Int32, (_LOCAL_EXPERTS,), stride_order=(0,), assumed_align=16)
        extra = (
            (fake(cutlass.Int32, (MAX_TOP_E,), stride_order=(0,), assumed_align=16),)
            if debug_sel
            else ()
        )
        _CACHE[key] = cute.compile(
            launch_decode_l2_prefetch,
            hist,
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            int(top_e) if ntiles else 1,
            int(ntiles),
            int(ctas),
            bool(carveout),
            *extra,
            options="--enable-tvm-ffi",
        )
    return _CACHE[key]


def _layer_bases(fc1_w: "torch.Tensor", fc1_s: "torch.Tensor") -> Tuple[int, int, int, int]:
    """(weight base, weight expert stride, scale base, scale expert stride) in bytes of one layer's FC1 operands."""
    if (
        fc1_w.dim() < 2
        or fc1_s.dim() < 2
        or fc1_w.shape[0] != _LOCAL_EXPERTS
        or fc1_s.shape[0] != _LOCAL_EXPERTS
    ):
        raise ValueError(
            f"FC1 operands must be stacked over {_LOCAL_EXPERTS} local experts, got {tuple(fc1_w.shape)} / "
            f"{tuple(fc1_s.shape)}"
        )
    w_stride = fc1_w.stride(0) * fc1_w.element_size()
    s_stride = fc1_s.stride(0) * fc1_s.element_size()
    if (
        fc1_w[0].numel() * fc1_w.element_size() != W_EXPERT
        or fc1_s[0].numel() * fc1_s.element_size() != S_EXPERT
    ):
        raise ValueError(
            f"FC1 expert slices must be {W_EXPERT} / {S_EXPERT} B, got "
            f"{fc1_w[0].numel() * fc1_w.element_size()} / {fc1_s[0].numel() * fc1_s.element_size()}"
        )
    if not fc1_w[0].is_contiguous() or not fc1_s[0].is_contiguous():
        raise ValueError("each expert's FC1 weight / scale slice must be contiguous")
    if fc1_w.data_ptr() % 16 or fc1_s.data_ptr() % 16 or w_stride % 16 or s_stride % 16:
        raise ValueError("FC1 operands must be 16-byte aligned")
    return fc1_w.data_ptr(), w_stride, fc1_s.data_ptr(), s_stride


def run_prefetch(
    expert_hist: "torch.Tensor",
    fc1_w: "torch.Tensor",
    fc1_s: "torch.Tensor",
    *,
    top_e: int,
    ntiles: int,
    ctas: int = DEFAULT_CTAS,
    carveout: bool = False,
    sel_out: Optional["torch.Tensor"] = None,
) -> None:
    """Launch on the current stream: prefetch N-tiles ``0..ntiles-1`` of the ``top_e`` most-counted experts."""
    if (
        expert_hist.dtype != torch.int32
        or tuple(expert_hist.shape) != (_LOCAL_EXPERTS,)
        or not expert_hist.is_contiguous()
    ):
        raise ValueError(f"expert_hist must be a contiguous int32 [{_LOCAL_EXPERTS}] tensor")
    if ntiles == 0:
        raise ValueError(
            "run_prefetch needs ntiles >= 1 (run_hold launches the carveout-only form)"
        )
    w, ws, s, ss = _layer_bases(fc1_w, fc1_s)
    fn = compile_prefetch(
        top_e=top_e,
        ntiles=ntiles,
        ctas=ctas,
        carveout=carveout,
        debug_sel=sel_out is not None,
        device=expert_hist.device.index,
    )
    if sel_out is None:
        fn(expert_hist, w, ws, s, ss, 0)
    else:
        fn(expert_hist, w, ws, s, ss, 0, sel_out)


def run_hold(expert_hist: "torch.Tensor", hold_ns: int, *, ctas: int = DEFAULT_CTAS) -> None:
    """Launch on the current stream the carveout-only form (no prefetch): ``ctas`` one-warp CTAs held ``hold_ns``.

    ``expert_hist`` is any int32 [64] device tensor (not read); it only fixes the device and the argument shape."""
    if expert_hist.dtype != torch.int32 or tuple(expert_hist.shape) != (_LOCAL_EXPERTS,):
        raise ValueError(f"expert_hist must be an int32 [{_LOCAL_EXPERTS}] tensor")
    fn = compile_prefetch(
        top_e=1, ntiles=0, ctas=ctas, carveout=True, device=expert_hist.device.index
    )
    fn(expert_hist, 0, 0, 0, 0, int(hold_ns))


def reference_selection(expert_hist: "torch.Tensor", top_e: int) -> list:
    """The kernel's slot table: experts by (count desc, index asc), zero counts excluded, padded with -1."""
    h = expert_hist.detach().cpu().tolist()
    order = sorted((e for e in range(_LOCAL_EXPERTS) if h[e] > 0), key=lambda e: (-h[e], e))[:top_e]
    return order + [-1] * (top_e - len(order))
