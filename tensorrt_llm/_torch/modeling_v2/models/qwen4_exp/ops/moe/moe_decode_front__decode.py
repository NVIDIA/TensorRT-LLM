# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small decode MoE block's router GEMM, routing, pre-routed tables and NVFP4 activation in one
launch.

`moe_route_front__decode` starts from the router logits; this op also forms them. The kernel
(`kernels/blackwell/moe_decode_front`) runs the 512-expert bf16 router GEMM on its low-M warp-MMA
CTAs (`k_warps` warps split K: the GEMM's summation order), hands each row's logits to its router
CTA through a staging buffer (`[32, 512]` int32, valid-tagged words, all zero between launches),
which takes the top-10 with the Renormalize weights and writes the runner's pre-routed tables for
an expert window, while its quantization CTAs write the NVFP4 codes with linear e4m3 block scales
at the layer's global scale. The runner then enters pre-routed (`run_moe_prerouted`) at the
tactic whose tile the tables were built for, as after the route front.

Given its own logits everything else is bitwise the route front's (`renorm_moe_routing_op`'s
experts and weights, `fp4_quantize`'s codes and scales, the same tables up to the row order
inside an expert's tile). The logits are the kernel's GEMM: against another bf16 router GEMM
(another kernel, another summation order) a logit can differ in its last bit, and a row whose
10th and 11th logits are within that bit of each other can pick the other expert: a same-precision
reorder, not a bitwise replacement of the router launch.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    table_sizes,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)

#: The routing geometry the kernel is written for: 512 experts, top-10.
NUM_EXPERTS = 512
TOP_K = 10
#: Rows a launch is certified for: 1..4 requests of four verify rows (the kernel takes 32).
MAX_ROWS = 16
#: Warps that split the router GEMM's K: hidden 2560 = 2 x 64 x 20 (the kernel needs K a multiple
#: of 64 warps' tiles; 16 and 32 warps do not divide 2560).
K_WARPS = 20
_SF_VEC = 16

_COMPILED: dict = {}


def compile_front(rows: int, hidden: int, row_stride: int, num_local: int, tile: int) -> bool:
    """Compile (once) the launch of `rows` rows of `hidden` (row stride `row_stride`) for a window
    of `num_local` experts at tile `tile`; False where the kernel does not take the shape. Outside
    CUDA-graph capture only; `__call__` then finds it."""
    key = (rows, hidden, row_stride, num_local, tile, bool(get_env_enable_pdl()))
    if key in _COMPILED:
        return _COMPILED[key] is not None
    assert not torch.cuda.is_current_stream_capturing(), "compile outside graph capture"
    if not (
        1 <= rows <= MAX_ROWS
        and hidden % (64 * K_WARPS) == 0
        and num_local % 32 == 0
        and num_local <= 32 * K_WARPS
        and rows * TOP_K <= 32 * K_WARPS
        and row_stride % 8 == 0
        and row_stride >= hidden
    ):
        _COMPILED[key] = None
        return False
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_utils import graph_safe_cute_tensor
    from tensorrt_llm._torch.kernels.blackwell.moe_decode_front import MoeDecodeFrontKernel

    kernel = MoeDecodeFrontKernel(
        num_rows=rows,
        k_extent=hidden,
        k_warps=K_WARPS,
        use_pdl=key[-1],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local=num_local,
        tile=tile,
        a_stride=row_stride,
    )
    dev = torch.device("cuda")
    # The compile reads only the operands' shapes and strides.
    a = torch.empty([rows, row_stride], dtype=torch.bfloat16, device=dev)[:, :hidden]
    w = torch.empty([NUM_EXPERTS, hidden], dtype=torch.bfloat16, device=dev)
    c = torch.empty([rows, NUM_EXPERTS], dtype=torch.bfloat16, device=dev)
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _COMPILED[key] = cute.compile(
        kernel,
        graph_safe_cute_tensor(a),
        graph_safe_cute_tensor(w),
        graph_safe_cute_tensor(c),
        z,
        z,
        z,
        z,
        z,
        z,
        z,
        cutlass.Int32(0),
        stream,
    )
    return True


def staging_buffer(device) -> torch.Tensor:
    """A launch site's staging buffer: zero, and zero again after every launch (the router CTA
    clears what it consumed). One per site: two launches in flight must not share it."""
    return torch.zeros([32 * NUM_EXPERTS], dtype=torch.int32, device=device)


def reference_logits(x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    """The router logits `x @ router.T` (`x` `[T, H]`, `router` `[512, H]`) summed exactly
    (float64: an fp32 GEMM is TF32 under the containers' `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`)
    and rounded once to bf16, returned as fp32."""
    return (x.double() @ router.double().t()).float().to(torch.bfloat16).float()


def logits_off_bound(logits: torch.Tensor, x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    """Where the kernel's `logits` leave this op's class against `reference_logits`: more than
    two bf16 steps of the exact logit, plus the worst-case error of an fp32 sum in another
    order, `2^-16 * sum_k |x_k router_k|` (`n * u` for the fewer than 256 sequential adds of a
    warp's K slice and the warps' reduction). Without that second term a logit near zero (a
    cancelling sum, whose bf16 step is tiny) fails any summation order, cuBLAS's included."""
    ref = reference_logits(x, router)
    step = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0**-100))) - 7)
    sum_bound = (x.double().abs() @ router.double().abs().t()).float() * 2.0**-16
    return (logits.float() - ref).abs() > 2 * step + sum_bound


def _cell(rows: int, offset: int, tile: int, why: str) -> Cell:
    return Cell(
        why=why, spec=dict(rows=rows, hidden=2560, local_offset=offset, num_local=128, tile=tile)
    )


class MoeDecodeFront(OpWrapper):
    """`logits, xq, xsf, ids, wts = moe_decode_front(x, router, global_scale, tables, stage, tile,
    local_offset, num_local)`: over `x` `[T, H]` bf16 (`T` <= `MAX_ROWS`, `compile_front`d) and the
    router weight `[512, H]` bf16 (contiguous), the router logits `[T, 512]` bf16, the NVFP4 codes
    `[T, H / 2]` uint8 with linear block scales `[T, H / 16]` uint8 at `global_scale`, the top-10
    experts `[T, 10]` int32 and their Renormalize weights `[T, 10]` bf16, and the pre-routed tables
    of the window `[local_offset, local_offset + num_local)` written into `tables` (int32, at least
    `table_sizes` long) for tile `tile`. `stage` is the site's staging buffer (`staging_buffer`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        _cell(4, 0, 8, "tep4 MTP-3: 1 request (4 rows), window 0, tile 8"),
        _cell(8, 128, 8, "tep4 MTP-3: 2 requests (8 rows), window 1, tile 8"),
        _cell(12, 256, 8, "tep4 MTP-3: 3 requests (12 rows), window 2, tile 8"),
        _cell(16, 384, 8, "tep4 MTP-3: 4 requests (16 rows, the certified last), window 3"),
    )

    note = """
    Source: `kernels/blackwell/moe_decode_front` (CuTe DSL; router GEMM on `K_WARPS` warps, one
    router CTA, quantization CTAs), compiled per shape by `compile_front` outside capture.
    Certified for 128-expert windows of the 512-expert / top-10 layer at hidden 2560 on GB300
    (tep4's own 4-16 rows), tile 8. Not certified: other windows, rows past 16, other routing
    geometries. The logits are the kernel's own GEMM (class S against another router GEMM).
    """

    def __call__(
        self,
        x: torch.Tensor,
        router: torch.Tensor,
        global_scale: torch.Tensor,
        tables: torch.Tensor,
        stage: torch.Tensor,
        tile: int,
        local_offset: int,
        num_local: int,
    ):
        import cuda.bindings.driver as cuda
        import cutlass

        from tensorrt_llm._torch.cute_dsl_utils import graph_safe_cute_tensor

        rows, hidden = x.shape
        key = (rows, hidden, x.stride(0), num_local, tile, bool(get_env_enable_pdl()))
        fn = _COMPILED.get(key)
        assert fn is not None, f"no decode front compiled for {key}: `compile_front` first"
        dev = x.device
        logits = torch.empty([rows, NUM_EXPERTS], dtype=torch.bfloat16, device=dev)
        xq = torch.empty([rows, hidden // 2], dtype=torch.uint8, device=dev)
        xsf = torch.empty([rows, hidden // _SF_VEC], dtype=torch.uint8, device=dev)
        ids = torch.empty([rows, TOP_K], dtype=torch.int32, device=dev)
        wts = torch.empty([rows, TOP_K], dtype=torch.bfloat16, device=dev)
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(
            graph_safe_cute_tensor(x),
            graph_safe_cute_tensor(router),
            graph_safe_cute_tensor(logits),
            p64(global_scale),
            p64(xq),
            p64(xsf),
            p64(ids),
            p64(wts),
            p64(tables),
            p64(stage),
            cutlass.Int32(local_offset),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return logits, xq, xsf, ids, wts

    def reference(
        self,
        x: torch.Tensor,
        router: torch.Tensor,
        global_scale: torch.Tensor,
        tables: torch.Tensor,
        stage: torch.Tensor,
        tile: int,
        local_offset: int,
        num_local: int,
    ):
        """The composed path over the exact router logits rounded once to bf16
        (`reference_logits`; the GEMMs agree with it up to their summation order,
        `logits_off_bound`; the tables are checked by their meaning,
        `moe_route_front__decode.check_tables`): `fp4_quantize` with linear scales,
        `renorm_moe_routing_op` over those logits."""
        logits = reference_logits(x, router).to(torch.bfloat16)
        xq, xsf = fp4_quantize(x, global_scale, _SF_VEC, False, False)
        ids, wts = renorm_moe_routing_op(logits, TOP_K, torch.bfloat16)
        return logits, xq, xsf.reshape(x.shape[0], -1), ids, wts

    def is_valid(self, x, router, global_scale, tables, stage, tile, local_offset, num_local):
        rows, hidden = x.shape
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.data_ptr() % 16 == 0
        assert 1 <= rows <= MAX_ROWS and x.stride(0) % 8 == 0
        assert router.dtype == torch.bfloat16 and router.is_contiguous()
        assert router.shape == (NUM_EXPERTS, hidden)
        assert 0 <= local_offset and local_offset + num_local <= NUM_EXPERTS
        assert tables.dtype == torch.int32 and tables.is_contiguous()
        assert tables.numel() >= sum(table_sizes(rows, TOP_K, num_local, tile))
        assert stage.dtype == torch.int32 and stage.numel() >= 32 * NUM_EXPERTS


moe_decode_front = MoeDecodeFront()
