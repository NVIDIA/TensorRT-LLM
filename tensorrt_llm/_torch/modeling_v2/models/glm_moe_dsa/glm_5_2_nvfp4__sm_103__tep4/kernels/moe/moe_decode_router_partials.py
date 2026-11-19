# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode MoE router: split-K4 FP32 partials ``[T, 4, 256]`` for T = 1..256, latency-first.

Drop-in for ``moe_frontend_router_tc.run_router_tc_partials`` (the reused
TRT-LLM persistent GEMM, ``mma_tiler (64, 32)``, split-K 4, cluster (1, 2)):
the same tcgen05 instruction (F16 kind, M64 x N32 x K16, FP32 accumulate) is
issued in the same ascending-K order over the same K slices, so the partials
are bit-identical and ``moe_topk_routing`` sums them exactly as before. What
changes is the schedule around the MMAs, all for the latency of one decode
launch behind the pre-MoE all-reduce (PDL):

* **Weights before the PDL wait.** Each of the 32 CTAs (8 N tiles x 4 K
  splits) TMA-loads its whole weight slice (24 tiles of 32 rows x 64 K, 96 KiB)
  into a dedicated SMEM buffer before ``griddepcontrol.wait`` -- the router
  weight is immutable, and the router launches well ahead of the all-reduce's
  end. Only the activation is read after the wait.
* **Activation multicast.** ``cluster`` CTAs of one K split (consecutive N
  tiles; 8 by default) form a cluster; each loads 64/cluster of the 64
  activation rows of every A tile and multicasts them to the cluster, so an SM
  requests 1/cluster of the activation bytes after the wait. The cluster size
  changes no MMA, so the partials are the same for 1, 2, 4 and 8. In serving
  an 8-CTA cluster must find 8 free SMs of one GPC behind the pre-MoE
  all-reduce and lands after the shared expert's gate/up; ``cluster=2``
  (router cluster size) places it ahead of it again: graph chain behind a
  256-CTA all-reduce stand-in with the real shared gate/up on a forked stream,
  T=64, top-k end 12.5 -> 7.8 us after the all-reduce (TRT-LLM router 8.6).
* **Few stage hand-offs.** One A pipeline stage carries ``KSUB`` 64-K tiles
  (4 by default: 6 hand-offs per CTA instead of 24); MMAs still run tile by
  tile, K16 by K16.
* **Early dependent launch.** ``griddepcontrol.launch_dependents`` at kernel
  entry: the fused top-k launches during the router and its own
  ``griddepcontrol.wait`` still orders it after this grid's completion.

Measured (fused decode router, B300, decode T=64, graph-replayed chain behind a
PDL producer, cold weights): router end - producer end 3.62 -> 3.10 us;
with ``moe_topk_routing_push`` the router + top-k chain 7.78 -> 5.95 us.

Operands: ``x`` BF16 ``[T, 6144]`` contiguous (16-byte aligned), weight BF16
``[256, 6144]`` contiguous, partials FP32 ``[T, 4, 256]`` contiguous
(32-byte aligned; 256-bit stores). Rows beyond T are TMA zero-fill and never
stored. One compile serves every T (symbolic).

T above 64 (the MTP3 / MTP5 verify steps and draft step 0 carry 128 / 192 rows):
the 32-CTA schedule above is repeated for each 64-row M tile along grid y,
``ceil(T / 64)`` of them. M tile ``m`` loads activation rows ``64 m ..`` and
stores those rows; every CTA still loads its whole weight slice before the PDL
wait (from L2 after the first M tile: 3 MiB per tile). The MMAs, their K order
and the TMEM drain are the same in every M tile, so each tile's partials are the
64-row partials of its rows, bit for bit. At T <= 64 grid y is 1 and tile 0
starts at row 0.
"""

from typing import Callable, Dict, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass.base_dsl.array import L1EvictKind
from cutlass.experimental import primitives as prims

_K = 6144
_E = 256
_TILE_M = 64
_TILE_N = 32
_TILE_K = 64
_SPLIT = 4
_KTILES = _K // (_TILE_K * _SPLIT)  # 24 64-K tiles per split
_NT = _E // _TILE_N  # 8 N tiles
_CTAS = _NT * _SPLIT  # 32
_CLUSTER = 8  # the 8 N tiles of one K split share every activation tile
# router cluster size: the activation multicast group (CTAs of one K split, consecutive N tiles). 8 is the served
# schedule; 2 / 4 / 1 issue the same MMAs in the same order (bitwise partials) and only change how many SMs one
# cluster needs at once -- an 8-CTA cluster must find 8 free SMs of one GPC behind the pre-MoE all-reduce, which
# in serving places the router after the shared expert's gate/up.
_CLUSTERS = (1, 2, 4, 8)
#: Up to four 64-row M tiles on grid y.
MAX_TOKENS = 4 * _TILE_M

# 64-K tiles per pipeline stage (divides 24, <= 8) and the activation pipeline depth.
_KSUB = 4
_STAGES = 3
if _KTILES % _KSUB or _KSUB not in (1, 2, 3, 4, 6, 8):
    raise ValueError("_KSUB must divide 24 and be <= 8")
_SMEM_LIMIT = 227 * 1024
if _STAGES * _KSUB * _TILE_M * _TILE_K * 2 + _KTILES * _TILE_N * _TILE_K * 2 + 1024 > _SMEM_LIMIT:
    raise ValueError("GLM_DEC_ROUTER_STAGES x KSUB exceeds B300 SMEM with the 96 KiB weight slice")

RouterFn = Callable[..., None]
_CACHE: Dict[Tuple[int, int, int, bool], RouterFn] = {}


@cute.kernel
def decode_router_partials_kernel(
    a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    b_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    output: cute.Tensor,
    tokens: cutlass.Int32,
    stages: cutlass.Constexpr[int],
    ksub: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    cluster: cutlass.Constexpr[int],
):
    steps: cutlass.Constexpr[int] = _KTILES // ksub
    a_rows: cutlass.Constexpr[int] = _TILE_M // cluster
    mask: cutlass.Constexpr[int] = (1 << cluster) - 1
    a_tile: cutlass.Constexpr[int] = _TILE_M * _TILE_K
    b_tile: cutlass.Constexpr[int] = _TILE_N * _TILE_K
    smem_a = cutlass.Array(
        cutlass.BFloat16,
        stages * ksub * a_tile,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_b = cutlass.Array(
        cutlass.BFloat16,
        _KTILES * b_tile,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    full = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    b_full = cutlass.Array(cutlass.Int64, _KTILES, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_location = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    tid, _, _ = cute.arch.thread_idx()
    block, m_tile, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tid % 32
    m0 = m_tile * cutlass.Int32(_TILE_M)
    if cutlass.const_expr(use_pdl):
        # Early scheduling signal only: the top-k's griddepcontrol.wait still waits for this grid to complete.
        if tid == 0:
            cute.arch.griddepcontrol_launch_dependents()
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), cluster)  # every cluster CTA's MMA commit
            for kt in cutlass.range_constexpr(_KTILES):
                prims.mbarrier_init(b_full.subview(kt), 1)
            prims.mbarrier_init(acc_full.subview(0), 1)
    prims.fence_mbarrier_init()
    # Peers' barriers must be initialised before the first multicast reaches them; the producer waits only after
    # its weight prefetch and PDL wait, the other warps right away (every arrive is paired with a wait).
    prims.barrier_cluster_arrive()
    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_alloc(tmem_location, _TILE_N, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_location.load()
    tmem = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    swizzle: cutlass.Constexpr = a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    desc_a = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=128,
        stride_byte_offset=1024,
        layout=swizzle,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        start_address=smem_b,
        leading_byte_offset=128,
        stride_byte_offset=1024,
        layout=swizzle,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        m_dim=_TILE_M,
        n_dim=_TILE_N,
    )
    split = block // _NT
    coord_n = (block % _NT) * _TILE_N
    crank = block % cluster
    k0 = split * (_KTILES * _TILE_K)

    if warp == 0:
        # TMA producer.
        if prims.elect_sync():
            for kt in cutlass.range_constexpr(_KTILES):
                prims.mbarrier_arrive_expect_tx(b_full.subview(kt), b_tile * 2)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_b.subview(kt * b_tile),
                    b_desc.get_ptr(),
                    (k0 + kt * _TILE_K, coord_n),
                    b_full.subview(kt),
                )
            if cutlass.const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()
        prims.barrier_cluster_wait()
        stage = cutlass.Int32(0)
        empty_phase = cutlass.Int32(0)
        for st in cutlass.range(steps):
            if st >= stages:
                while not prims.mbarrier_try_wait_parity(empty.subview(stage), empty_phase):
                    pass
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(full.subview(stage), ksub * a_tile * 2)
                for j in cutlass.range_constexpr(ksub):
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        smem_a.subview((stage * ksub + j) * a_tile + crank * (a_rows * _TILE_K)),
                        a_desc.get_ptr(),
                        (k0 + (st * ksub + j) * _TILE_K, m0 + crank * a_rows),
                        full.subview(stage),
                        [],
                        multicast_mask=cutlass.Int16(mask),
                    )
            stage += 1
            if stage == stages:
                stage = cutlass.Int32(0)
                if st >= stages:
                    empty_phase = empty_phase ^ 1
        # Drain: every cluster CTA has consumed the last stages (and committed to this CTA's empty barriers), so
        # no multicast commit can arrive after this CTA exits.
        for t in cutlass.range_constexpr(min(stages, steps)):
            st_last = steps - min(stages, steps) + t
            while not prims.mbarrier_try_wait_parity(
                empty.subview(st_last % stages), (st_last // stages) % 2
            ):
                pass
    else:
        prims.barrier_cluster_wait()

    if warp == 1:
        # MMA issuer: per 64-K tile, four ascending K16 MMAs (the served instruction order).
        stage = cutlass.Int32(0)
        full_phase = cutlass.Int32(0)
        acc_pointer = prims.TmemAddr.from_row_col(
            prims.TmemAddr(tmem_raw).row_id, prims.TmemAddr(tmem_raw).col_id
        ).as_ptr(cutlass.Float32)
        for st in cutlass.range(steps):
            while not prims.mbarrier_try_wait_parity(full.subview(stage), full_phase):
                pass
            for j in cutlass.range_constexpr(ksub):
                kt = st * ksub + j
                while not prims.mbarrier_try_wait_parity(b_full.subview(kt), 0):
                    pass
                stage_desc_a = desc_a.advance_start_address((stage * ksub + j) * a_tile * 2)
                stage_desc_b = desc_b.advance_start_address(kt * b_tile * 2)
                for kb in cutlass.range_constexpr(4):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            acc_pointer,
                            stage_desc_a.advance_start_address(kb * 32),
                            stage_desc_b.advance_start_address(kb * 32),
                            instruction,
                            st > 0 if (j == 0 and kb == 0) else True,
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(
                    empty.subview(stage),
                    multicast_mask=cutlass.Int16(mask),
                    group=prims.CTAGroup.CTA_1,
                )
            stage += 1
            if stage == stages:
                stage = cutlass.Int32(0)
                full_phase = full_phase ^ 1
        if prims.elect_sync():
            prims.tcgen05_commit(acc_full.subview(0), group=prims.CTAGroup.CTA_1)

    if warp >= 2:
        # Four TMEM drain warps: M64 accumulator rows are lanes 0..15 of each warp's sub-partition.
        row = m0 + (warp % 4) * (_TILE_M // 4) + lane
        owns_row = lane < _TILE_M // 4
        global_output = output.iterator.raw_ptr()
        while not prims.mbarrier_try_wait_parity(acc_full.subview(0), 0):
            pass
        address = prims.TmemAddr.from_row_col(
            prims.TmemAddr(tmem_raw).row_id, prims.TmemAddr(tmem_raw).col_id
        ).as_ptr(cutlass.Float32)
        values = prims.tcgen05_ld("32x32b", address, num=_TILE_N)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        if owns_row and row < tokens:
            registers = cutlass.Array(cutlass.Float32, _TILE_N, alignment=32)
            registers.store(values)
            for piece in cutlass.range_constexpr(_TILE_N // 8):
                col = split * _E + coord_n + piece * 8
                vector = registers.load(piece * 8, vector_size=8, alignment=32)
                prims.store_ext(
                    vector.bitcast(cutlass.Int32),
                    global_output + row * (_E * _SPLIT) + col,
                    evict=L1EvictKind.NOALLOCATE,
                )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp == 2:
            prims.tcgen05_dealloc(tmem, _TILE_N, group="cta_1")


decode_router_partials_kernel.set_name_prefix(
    "glm52_moe_decode_router_partials", remove_cutlass_symbol=True
)


@cute.jit
def launch_decode_router_partials(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    tokens: cutlass.Int32,
    stream: cuda.CUstream,
    stages: cutlass.Constexpr[int],
    ksub: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    cluster: cutlass.Constexpr[int],
):
    a_desc = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[_K, tokens],
        global_strides=[_K * 2 // 16],
        box_dims=[_TILE_K, _TILE_M // cluster],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    b_desc = cuda_exp.create_tensor_map_tiled(
        global_address=b.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[_K, _E],
        global_strides=[_K * 2 // 16],
        box_dims=[_TILE_K, _TILE_N],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    decode_router_partials_kernel(a_desc, b_desc, c, tokens, stages, ksub, use_pdl, cluster).launch(
        grid=(_CTAS, cute.ceil_div(tokens, _TILE_M), 1),
        block=(192, 1, 1),
        cluster=(cluster, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _validate(input_tensor: "torch.Tensor", weight: "torch.Tensor", output: "torch.Tensor") -> int:
    m, k = input_tensor.shape
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"decode router supports T in [1, {MAX_TOKENS}], got {m}")
    if tuple(weight.shape) != (_E, _K) or k != _K:
        raise ValueError(f"decode router requires [T,{_K}] x [{_E},{_K}]")
    if tuple(output.shape) != (m, _SPLIT, _E):
        raise ValueError(f"decode router output must be [{m},{_SPLIT},{_E}]")
    if (
        input_tensor.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or output.dtype != torch.float32
    ):
        raise TypeError("decode router: BF16 input and weight, FP32 partials")
    if not (input_tensor.is_contiguous() and weight.is_contiguous() and output.is_contiguous()):
        raise ValueError("decode router operands must be contiguous")
    if input_tensor.data_ptr() % 16 or weight.data_ptr() % 16 or output.data_ptr() % 32:
        raise ValueError("decode router: 16-byte aligned input/weight, 32-byte aligned partials")
    device = (
        input_tensor.device.index
        if input_tensor.device.index is not None
        else torch.cuda.current_device()
    )
    if weight.device != input_tensor.device or output.device != input_tensor.device:
        raise ValueError("decode router tensors must share one CUDA device")
    return device


def compile_router_partials(
    input_tensor: "torch.Tensor",
    weight: "torch.Tensor",
    output: "torch.Tensor",
    *,
    use_pdl: bool,
    cluster: int = _CLUSTER,
) -> RouterFn:
    """Compile (once per device / schedule / PDL mode / cluster) the handle ``fn(x, weight, partials, T)``."""
    device = _validate(input_tensor, weight, output)
    if cluster not in _CLUSTERS:
        raise ValueError(f"decode router cluster must be one of {_CLUSTERS}, got {cluster}")
    key = (device, _STAGES, _KSUB, bool(use_pdl), int(cluster))
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        tokens = cute.sym_int64()
        fake = cute.runtime.make_fake_compact_tensor
        a = fake(cutlass.BFloat16, (tokens, _K), stride_order=(1, 0), assumed_align=16)
        b = fake(cutlass.BFloat16, (_E, _K), stride_order=(1, 0), assumed_align=16)
        c = fake(
            cutlass.Float32,
            (tokens, _SPLIT, _E),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
        _CACHE[key] = cute.compile(
            launch_decode_router_partials,
            a,
            b,
            c,
            cutlass.Int32(MAX_TOKENS),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            stages=_STAGES,
            ksub=_KSUB,
            use_pdl=bool(use_pdl),
            cluster=int(cluster),
            options="--enable-tvm-ffi",
        )
    return _CACHE[key]


def run_router_partials(
    input_tensor: "torch.Tensor",
    weight: "torch.Tensor",
    output: "torch.Tensor",
    *,
    use_pdl: bool,
    cluster: int = _CLUSTER,
) -> None:
    """Launch on the current stream: ``output[t, s, e] = sum over K split s of x[t] * weight[e]`` (FP32)."""
    compile_router_partials(input_tensor, weight, output, use_pdl=use_pdl, cluster=cluster)(
        input_tensor, weight, output, int(input_tensor.shape[0])
    )
