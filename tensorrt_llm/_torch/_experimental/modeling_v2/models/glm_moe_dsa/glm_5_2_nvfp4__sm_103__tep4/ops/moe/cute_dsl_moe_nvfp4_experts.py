# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's routed experts, NVFP4, this rank's 64-expert window: FC1 (gather + grouped GEMM
+ SwiGLU + NVFP4 requantize) then FC2 (grouped GEMM + weighted finalize, accumulated into
the output), in CuTe DSL. A decode pair compiled for the row capacity, and a compact-M128
prefill pair for context steps.

The decode GEMMs take `cute.Tensor` operands, and the operands this target holds are NVFP4
packed two values to the byte. cute counts a tensor's shape and strides in elements, torch
has no half-byte dtype, and `from_dlpack` cannot describe a `[K, N, L]` element view over
`K*N*L/2` bytes (`torch.Tensor.set_` appears to, and silently reallocates the storage).
`cute.make_tensor` builds exactly the view required, and runs only inside a `@cute.jit`
function, so `_fc1` / `_fc2` below build those views from raw pointers next to the kernel
call that consumes them. `@cute.jit` reads its function's source, so they live in this
importable module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import make_ptr

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_gather_grouped_gemm_act_fusion as _gg1
from ...kernels.moe import moe_grouped_gemm_finalize_fusion as _gg2
from ...kernels.moe import moe_prefill_nvfp4_pair as _pair
from ...kernels.moe.moe_utils import cute_tensor_from_torch as _ctf
from .cute_dsl_moe_routing import LOCAL_EXPERTS, TILE, TOP_K, allocate_maps, max_tiles

HIDDEN = 6144
INTER = 2048
#: FC1 is a 128-wide N tile: `mma_tiler_n=256` fails the kernel's own reference check at
#: this geometry. NVFP4 fixes the K tile at 256.
MMA_TILER_N = 128
MMA_TILER_K = 256
#: NVFP4: one e4m3 scale per 16 elements; the scale atom is 128 rows by 64 K in 512 bytes.
SF_VEC = 16
SF_ATOM = 512
SF_ATOM_MN = 128
SF_ATOM_K = 64
#: The gate/up interleave of the FC1 weight: 64-row blocks alternating up, gate.
PAIR_BLOCK = 64
DECODE_MAX_ROWS = 64
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192

# The grouped GEMMs take packed operands: A, B and FC1's C are `Float4E2M1FNx2` tensors
# whose K (and N_out) extent counts byte pairs. A `@cute.jit` body cannot branch on that,
# since the DSL traces both arms.
ACT_DTYPE = cutlass.Float4E2M1FNx2
# FC1's C is the exception: the kernel module's own entry recasts it back to the scalar
# element before `host_function`, whose C TMA descriptor is built for scalar FP4.
# Describing C as byte pairs halves its row pitch and the output is garbage.
C_DTYPE = cutlass.Float4E2M1FN


def rest_n_sfc(n_out: int) -> int:
    """SFC's N extent: one atom spans `4 * SF_VEC` output columns."""
    return (n_out + 4 * SF_VEC - 1) // (4 * SF_VEC)


def allocate_decode_c(max_rows: int, inter: int, device) -> torch.Tensor:
    """One decode FC1 output buffer: `[1, max_tiles(max_rows) * 128, inter]` bytes, of
    which the scalar-FP4 view (row pitch `inter` nibbles) uses half."""
    return torch.zeros((1, max_tiles(max_rows) * TILE, inter), dtype=torch.uint8, device=device)


def allocate_pair_c(max_rows: int, inter: int, device) -> torch.Tensor:
    """One prefill FC1 output buffer: compact `[1, max_tiles(max_rows) * 128, inter / 2]`
    packed rows (half the decode buffer's row pitch)."""
    cap_m = max_tiles(max_rows) * TILE
    return torch.zeros((1, cap_m, inter // 2), dtype=torch.uint8, device=device)


def allocate_sfc(max_rows: int, inter: int, device) -> torch.Tensor:
    """One FC1 output-scale buffer in 512-byte atoms, shared by both pairs."""
    cap_m = max_tiles(max_rows) * TILE
    return torch.zeros(
        (1, cap_m // SF_ATOM_MN, rest_n_sfc(inter), SF_ATOM), dtype=torch.uint8, device=device
    )


def _act_ptr(data_ptr: int):
    """A raw NVFP4 global pointer, packed two values to the byte, 16-byte aligned."""
    return make_ptr(ACT_DTYPE, data_ptr, cute.AddressSpace.gmem, assumed_align=16)


def _e4m3_ptr(data_ptr: int):
    return make_ptr(cutlass.Float8E4M3FN, data_ptr, cute.AddressSpace.gmem, assumed_align=16)


@cute.jit
def _fc1(
    b_ptr,
    sfb_ptr,
    c_ptr,
    tile_idx,
    bias,
    a_ptr,
    sfa_ptr,
    sfc,
    norm_const_ptr: cutlass.Int64,
    alpha_ptr: cutlass.Int64,
    token_mapping,
    mn_limit,
    num_tiles_m_ptr: cutlass.Int64,
    stream,
    k: cutlass.Constexpr,
    n: cutlass.Constexpr,
    ll: cutlass.Constexpr,
    # The **capacity**, never the call's row count. `rows` is a Constexpr and therefore
    # part of cute's compile-cache key: a per-call value compiles one kernel per row
    # count, and a later larger call that reuses an earlier entry runs a kernel built for
    # fewer rows over more data (2 then 64 faults, 64 then 2 does not). A's dim-1 stride
    # is `k` regardless, and dim 2 has extent 1, so the capacity is also the right layout.
    rows: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    max_tiles_m: cutlass.Constexpr,
    rest_n_sfc: cutlass.Constexpr,
    mma_tiler_n: cutlass.Constexpr,
    mma_tiler_k: cutlass.Constexpr,
    raster_along_m: cutlass.Constexpr,
    num_sms: cutlass.Constexpr,
):
    """Gather + grouped GEMM + fused SwiGLU + SFC.

    B is `(n, k, l)` per the kernel's docstring but `(k, n, l)` in the kernel module's own
    entry, which is what actually runs, so K leads here.
    """
    # `kk` is the K extent in packed byte-pair elements; the scale-factor layouts count K
    # in values.
    kk = k // 2
    b = cute.make_tensor(b_ptr, cute.make_layout((kk, n, ll), stride=(1, kk, kk * n)))
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (512, k // SF_ATOM_K, n // SF_ATOM_MN, ll),
            stride=(
                1,
                512,
                512 * (k // SF_ATOM_K),
                512 * (k // SF_ATOM_K) * (n // SF_ATOM_MN),
            ),
        ),
    )
    a = cute.make_tensor(a_ptr, cute.make_layout((kk, rows, 1), stride=(1, kk, kk * rows)))
    # SFA is three-dimensional with the scale axis contiguous, matching the
    # `(padded_tokens, sf_k, 1)` buffer the kernel's entry takes at leading_dim=1. A
    # two-dimensional view compiles and produces an all-zero C.
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout((rows, k // SF_VEC, 1), stride=(k // SF_VEC, 1, (k // SF_VEC) * rows)),
    )
    _gg1.host_function(
        b,
        sfb,
        c_ptr,
        tile_idx,
        a,
        sfa,
        sfc,
        norm_const_ptr,
        alpha_ptr,
        token_mapping,
        mn_limit,
        k,
        topk,
        num_tiles_m_ptr,
        max_tiles_m,
        rest_n_sfc,
        mma_tiler_n,
        mma_tiler_k,
        raster_along_m,
        num_sms,
        stream,
    )


@cute.jit
def _fc2(
    a_ptr,
    b_ptr,
    sfa_ptr,
    sfb_ptr,
    out_ptr: cutlass.Int64,
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    perm_idx_ptr: cutlass.Int64,
    token_scales_ptr: cutlass.Int64,
    alpha_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    num_tiles_m_ptr: cutlass.Int64,
    stream,
    k: cutlass.Constexpr,
    n: cutlass.Constexpr,
    ll: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    # Constexpr, as it is on FC1. The kernel declares it `Int32`, and k/n/max_tiles_m are
    # declared the same way and already reach it as constants. Leaving it dynamic puts it
    # next to `stream` in the compiled handle's runtime-argument list, where the two are
    # silently swappable.
    topk: cutlass.Constexpr,
    max_tiles_m: cutlass.Constexpr,
    mma_tiler_n: cutlass.Constexpr,
    mma_tiler_k: cutlass.Constexpr,
    use_blkred: cutlass.Constexpr,
    blkred_bufs: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    use_bias_smem: cutlass.Constexpr,
    epi_subtile_n: cutlass.Constexpr,
    raster_along_m: cutlass.Constexpr,
    num_sms: cutlass.Constexpr,
):
    """Grouped GEMM + fused finalize.

    `has_bias` must be False when no bias is supplied: otherwise the bias prefetch reads
    `expert_idx * problem_n` floats past the allocation, folding whatever lived there into
    the accumulator, inside torch's caching allocator slab where compute-sanitizer sees
    nothing.
    """
    kk = k // 2
    a = cute.make_tensor(a_ptr, cute.make_layout((kk, rows, 1), stride=(1, kk, kk * rows)))
    b = cute.make_tensor(b_ptr, cute.make_layout((kk, n, ll), stride=(1, kk, kk * n)))
    # FC2's SFA is in the 512-byte atom layout, not the linear one FC1 gathers from: the
    # kernel's entry takes both SFA and SFB as `permute(3, 2, 1, 0)` of a
    # `create_scale_factor_flat` buffer. Reusing FC1's linear form here is a silent wrong
    # answer, not an error.
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (512, k // SF_ATOM_K, rows // SF_ATOM_MN, 1),
            stride=(
                1,
                512,
                512 * (k // SF_ATOM_K),
                512 * (k // SF_ATOM_K) * (rows // SF_ATOM_MN),
            ),
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (512, k // SF_ATOM_K, n // SF_ATOM_MN, ll),
            stride=(
                1,
                512,
                512 * (k // SF_ATOM_K),
                512 * (k // SF_ATOM_K) * (n // SF_ATOM_MN),
            ),
        ),
    )
    _gg2.host_function(
        a,
        b,
        sfa,
        sfb,
        out_ptr,
        tile_idx_ptr,
        bias_ptr,
        perm_idx_ptr,
        token_scales_ptr,
        alpha_ptr,
        mn_limit_ptr,
        k,
        n,
        topk,
        num_tiles_m_ptr,
        max_tiles_m,
        mma_tiler_n,
        mma_tiler_k,
        use_blkred,
        blkred_bufs,
        has_bias,
        use_bias_smem,
        epi_subtile_n,
        raster_along_m,
        num_sms,
        stream,
    )


@dataclass(frozen=True)
class CompiledDecodeExperts:
    """The decode pair, compiled once for a row capacity. `bias` is the zero fp32
    `[experts, 2 * inter]` operand both GEMMs declare and neither reads."""

    fc1: object
    fc2: object
    bias: torch.Tensor
    hidden: int
    inter: int
    experts: int
    max_rows: int


@dataclass(frozen=True)
class CompiledPairExperts:
    """The prefill pair at both MMA N widths, `{128 | 256: (gg1, gg2)}`; the width is
    picked per call (`selected_width`: 128 up to 1536 rows, 256 above). `bias` is the zero
    fp32 `[experts, hidden]` operand in GG2's ABI that it does not read."""

    pair: Mapping[int, tuple[object, object]]
    bias: torch.Tensor
    hidden: int
    inter: int
    experts: int


def build_decode(
    *,
    c: torch.Tensor,
    sfc: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    xq: torch.Tensor,
    xsf: torch.Tensor,
    out: torch.Tensor,
    hidden: int,
    inter: int,
    experts: int,
    max_rows: int,
) -> CompiledDecodeExperts:
    """Compile FC1 and FC2 once, for up to `max_rows` rows. Call at load; nothing is
    launched.

    `c`, `sfc`, `maps`, `xq`, `xsf` and `out` are buffers of the shapes the target passes
    per call (one slot's): the compile reads their layouts and nothing else, and pointers
    stay runtime operands. `rows` is compiled as the capacity, `max_rows * 8` expanded
    rows, and FC2's tile bound as `max_tiles(max_rows) * 128`. FC1 rasters along M with
    a 128 x 256 MMA tile; FC2 has no bias, no block reduction, one buffer, a 0 epilogue
    subtile and rasters along N. Both use every SM. Calling the `@cute.jit` adapters
    directly instead would rebuild their IR on every call (150 builds per forward, the
    first inside the capture). Raises if a compile fails: the routed experts have no
    stock path for these packed weights.
    """
    dev = out.device
    n_fc1 = 2 * inter
    cap_m = max_tiles(max_rows) * TILE
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    bias = torch.zeros((experts, n_fc1), dtype=torch.float32, device=dev)
    scratch = xq.data_ptr()
    with torch.cuda.device(dev):
        stream = cutlass_torch.current_stream()
        fc1 = cute.compile(
            _fc1,
            _act_ptr(scratch),
            _e4m3_ptr(scratch),
            _ctf(c.permute(2, 1, 0), C_DTYPE, leading_dim=0),
            _ctf(maps["tile_idx_to_expert_idx"], None, leading_dim=0),
            _ctf(bias, None, leading_dim=1),
            _act_ptr(xq.data_ptr()),
            _e4m3_ptr(xsf.data_ptr()),
            _ctf(sfc, None, leading_dim=3),
            cutlass.Int64(scratch),
            cutlass.Int64(scratch),
            _ctf(maps["permuted_idx_to_expanded_idx"], None, leading_dim=0),
            _ctf(maps["tile_idx_to_mn_limit"], None, leading_dim=0),
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            stream,
            hidden,
            n_fc1,
            experts,
            max_rows,
            TOP_K,
            max_rows * TOP_K,
            rest_n_sfc(inter),
            MMA_TILER_N,
            MMA_TILER_K,
            True,
            sms,
        )
        fc2 = cute.compile(
            _fc2,
            _act_ptr(c.data_ptr()),
            _act_ptr(scratch),
            _e4m3_ptr(sfc.data_ptr()),
            _e4m3_ptr(scratch),
            cutlass.Int64(out.data_ptr()),
            cutlass.Int64(maps["tile_idx_to_expert_idx"].data_ptr()),
            cutlass.Int64(bias.data_ptr()),
            cutlass.Int64(maps["permuted_idx_to_expanded_idx"].data_ptr()),
            cutlass.Int64(scratch),
            cutlass.Int64(scratch),
            cutlass.Int64(maps["tile_idx_to_mn_limit"].data_ptr()),
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            stream,
            inter,
            hidden,
            experts,
            cap_m,
            TOP_K,
            max_rows * TOP_K,
            MMA_TILER_N,
            MMA_TILER_K,
            # has_bias False: True makes the prefetch read past the allocation and fold
            # whatever lived there into the accumulator, where compute-sanitizer is silent.
            False,
            1,
            False,
            False,
            0,
            False,
            sms,
        )
    return CompiledDecodeExperts(fc1, fc2, bias, hidden, inter, experts, int(max_rows))


def _run_pair(
    compiled: CompiledPairExperts,
    xq,
    xsf,
    maps,
    topk_w,
    fc1_w,
    fc1_s,
    fc2_w,
    fc2_s,
    norm_const,
    alpha1,
    alpha2,
    out,
    c,
    sfc,
    join,
) -> None:
    """One layer's two grouped GEMMs on the prefill pair.

    Operands in the pair's torch ABI: packed FP4 as `float4_e2m1fn_x2` views, the expert
    stacks as `[E, K/2, N]` views with K/2 contiguous (a permute of the stored `[E, N,
    K/2]`: the bytes the decode pair reads), E4M3 scales, and the six maps as they are.
    The tile bound is this call's `max_tiles`; the live tile count is a device pointer.
    """
    rows = int(xq.shape[0])
    hidden, inter, experts = compiled.hidden, compiled.inter, compiled.experts
    n_fc1 = 2 * inter
    gg1, gg2 = compiled.pair[_pair.selected_width(rows)]
    f4, e4 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    cap = max_tiles(rows)
    c4, sfc4 = c.view(f4), sfc.view(e4)
    gg1(
        fc1_w.view(f4).view(experts, n_fc1, hidden // 2).permute(0, 2, 1),
        fc1_s,
        c4,
        maps["tile_idx_to_expert_idx"],
        xq.view(f4).view(1, rows, hidden // 2),
        xsf.view(e4).view(rows, hidden // SF_VEC, 1),
        sfc4,
        norm_const,
        alpha1,
        maps["permuted_idx_to_expanded_idx"],
        maps["tile_idx_to_mn_limit"],
        hidden,
        maps["num_non_exiting_tiles"],
        cap,
        rest_n_sfc(n_fc1 // 2),
    )
    if join is not None:
        # The shared expert's partial must be in `out` before FC2 adds to it.
        torch.cuda.current_stream().wait_stream(join)
    gg2(
        c4,
        fc2_w.view(f4).view(experts, hidden, inter // 2).permute(0, 2, 1),
        sfc4,
        fc2_s,
        out,
        maps["tile_idx_to_expert_idx"],
        compiled.bias,
        maps["permuted_idx_to_expanded_idx"],
        topk_w,
        alpha2,
        maps["tile_idx_to_mn_limit"],
        inter,
        hidden,
        TOP_K,
        maps["num_non_exiting_tiles"],
        cap,
    )


def _run_decode(
    compiled: CompiledDecodeExperts,
    xq,
    xsf,
    maps,
    topk_w,
    fc1_w,
    fc1_s,
    fc2_w,
    fc2_s,
    norm_const,
    alpha1,
    alpha2,
    out,
    c,
    sfc,
    join,
) -> None:
    """The compiled handles take only the runtime arguments, in signature order."""
    stream = cutlass_torch.current_stream()
    compiled.fc1(
        _act_ptr(fc1_w.data_ptr()),
        _e4m3_ptr(fc1_s.data_ptr()),
        _ctf(c.permute(2, 1, 0), C_DTYPE, leading_dim=0),
        _ctf(maps["tile_idx_to_expert_idx"], None, leading_dim=0),
        _ctf(compiled.bias, None, leading_dim=1),
        _act_ptr(xq.data_ptr()),
        _e4m3_ptr(xsf.data_ptr()),
        _ctf(sfc, None, leading_dim=3),
        cutlass.Int64(norm_const.data_ptr()),
        cutlass.Int64(alpha1.data_ptr()),
        # Indexed by permuted row, so it must be `valid_m` long; the inverse array is
        # `rows * top_k` long and is read past its end once enough tiles are live.
        _ctf(maps["permuted_idx_to_expanded_idx"], None, leading_dim=0),
        _ctf(maps["tile_idx_to_mn_limit"], None, leading_dim=0),
        cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
        stream,
    )
    if join is not None:
        # The shared expert's partial must be in `out` before FC2 adds to it.
        torch.cuda.current_stream().wait_stream(join)
    compiled.fc2(
        _act_ptr(c.data_ptr()),
        _act_ptr(fc2_w.data_ptr()),
        _e4m3_ptr(sfc.data_ptr()),
        _e4m3_ptr(fc2_s.data_ptr()),
        cutlass.Int64(out.data_ptr()),
        cutlass.Int64(maps["tile_idx_to_expert_idx"].data_ptr()),
        cutlass.Int64(compiled.bias.data_ptr()),
        cutlass.Int64(maps["permuted_idx_to_expanded_idx"].data_ptr()),
        cutlass.Int64(topk_w.data_ptr()),
        cutlass.Int64(alpha2.data_ptr()),
        cutlass.Int64(maps["tile_idx_to_mn_limit"].data_ptr()),
        cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
        stream,
    )


def build_prefill_pair(hidden: int, inter: int, experts: int, device) -> CompiledPairExperts:
    """Compile the prefill pair at both MMA N widths (128, 256), every SM, PDL as the
    engine sets it, and launch each width once on zero tiles: a compiled handle still
    loads its module on its first launch, which the first context step would otherwise
    pay. Call at load. The pair is fixed at hidden 6144, intermediate 2048, 64 experts.

    The warm launch allocates weight-shaped scratch for the descriptors (nothing is read
    at zero tiles), freed on return. Raises if the device is not sm_103 or a compile
    fails: the target then runs every context step on the decode pair.
    """
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    with torch.cuda.device(device):
        pair = {
            width: (
                _pair.compile_gg1(tile_n=width, num_sms=num_sms, use_pdl=_USE_PDL),
                _pair.compile_gg2(tile_n=width, num_sms=num_sms, use_pdl=_USE_PDL),
            )
            for width in (128, 256)
        }
        compiled = CompiledPairExperts(
            pair,
            torch.zeros((experts, hidden), dtype=torch.float32, device=device),
            hidden,
            inter,
            experts,
        )
        u8 = {"dtype": torch.uint8, "device": device}
        e4 = {"dtype": torch.float8_e4m3fn, "device": device}
        n_fc1 = 2 * inter
        maps = allocate_maps(PREFILL_MAX_ROWS, device)
        for k in ("tile_idx_to_expert_idx", "tile_idx_to_mn_limit", "num_non_exiting_tiles"):
            maps[k].zero_()
        fc1_w = torch.empty((experts, n_fc1, hidden // 2), **u8)
        fc1_s = torch.empty((experts, n_fc1 // SF_ATOM_MN, hidden // SF_ATOM_K, SF_ATOM), **e4)
        fc2_w = torch.empty((experts, hidden, inter // 2), **u8)
        fc2_s = torch.empty((experts, hidden // SF_ATOM_MN, inter // SF_ATOM_K, SF_ATOM), **e4)
        one = torch.ones(1, dtype=torch.float32, device=device)
        alpha = torch.ones(experts, dtype=torch.float32, device=device)
        c = allocate_pair_c(PREFILL_MAX_ROWS, inter, device)
        sfc = allocate_sfc(PREFILL_MAX_ROWS, inter, device)
        # One row count per width: N128 up to 1536 rows, N256 above.
        for rows in (513, PREFILL_MAX_ROWS):
            _run_pair(
                compiled,
                torch.zeros((rows, hidden // 2), **u8),
                torch.zeros((rows, hidden // SF_VEC), **u8),
                maps,
                torch.zeros((rows, TOP_K), dtype=torch.float32, device=device),
                fc1_w,
                fc1_s,
                fc2_w,
                fc2_s,
                one,
                alpha,
                alpha,
                torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device),
                c,
                sfc,
                None,
            )
        torch.cuda.synchronize(device)
    return compiled


# ---- the arithmetic, in plain torch --------------------------------------------------

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """`[..., K/2]` packed bytes -> `[..., K]` fp32 values (element 2i in the low nibble)."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    value = lut[(codes & 7).long()]
    return torch.where((codes & 8).bool(), -value, value)


def atoms_to_rows(s: torch.Tensor) -> torch.Tensor:
    """`[E, N/128, K/64, 512]` scale atoms -> row-major `[E, N, K/16]` fp32 scales. An atom
    holds row `128b + 32*i1 + i0`, scale column `4kb + ka` at byte `16*i0 + 4*i1 + ka`."""
    e, nb, kb, _ = s.shape
    v = s.view(torch.float8_e4m3fn).float().view(e, nb, kb, 32, 4, 4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(e, nb * 128, kb * 4)


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """NVFP4 `[R, K/2]` codes times their `[R, K/16]` fp32 block scales -> fp32 `[R, K]`.
    No global scale: the kernels fold it into the per-expert alphas."""
    vals = _unpack_e2m1(packed)
    r, k = vals.shape[-2], vals.shape[-1]
    return (vals.view(-1, r, k // SF_VEC, SF_VEC) * scales.view(-1, r, k // SF_VEC, 1)).view(
        vals.shape
    )


def _swiglu_interleaved(acc: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """`(alpha*up) * silu(alpha*gate)` over 64-column blocks alternating up, gate."""
    r, n = acc.shape
    blocks = acc.view(r, n // (2 * PAIR_BLOCK), 2, PAIR_BLOCK)
    up = blocks[:, :, 0].reshape(r, n // 2) * alpha
    gate = blocks[:, :, 1].reshape(r, n // 2) * alpha
    return up * gate * torch.sigmoid(gate)


def expert_rows(maps: Mapping[str, torch.Tensor]) -> dict[int, torch.Tensor]:
    """`{local expert: expanded indices}` over the valid rows of the live tiles."""
    tiles = int(maps["num_non_exiting_tiles"].reshape(-1)[0].item())
    tile_expert = maps["tile_idx_to_expert_idx"][:tiles].tolist()
    mn = maps["tile_idx_to_mn_limit"][:tiles].tolist()
    p2e = maps["permuted_idx_to_expanded_idx"]
    found: dict[int, list[torch.Tensor]] = {}
    for t, (e, hi) in enumerate(zip(tile_expert, mn)):
        found.setdefault(int(e), []).append(p2e[t * TILE : int(hi)].long())
    return {e: torch.cat(v) for e, v in found.items()}


class CuteDslMoeNvfp4Experts(OpWrapper):
    """One layer's routed experts on this rank's window, accumulated into `out`. Returns
    `out`.

        h_r  = (a1*up) * silu(a1*gate),  [up | gate] = A[token] x W1[e]^T,  a1 = alpha1[e]
        out[token] += topk_w[token, slot] * alpha2[e] * (nvfp4(h_r, norm_const) x W2[e]^T)

    for every valid row r of the maps (`token = expanded // 8`, `slot = expanded % 8`).

    `xq` uint8 `[rows, 3072]` and `xsf` (`rows * 384` e4m3 bytes, linear) are the NVFP4
    input; `maps` the six maps (`cute_dsl_moe_routing`) for these rows; `topk_w` fp32
    `[rows, 8]`; `fc1_w` uint8 `[64, 4096, 3072]` with 64-row blocks alternating up, gate,
    and `fc1_s` its e4m3 scales in 512-byte atoms `[64, 32, 96, 512]`; `fc2_w` uint8
    `[64, 6144, 1024]`, `fc2_s` `[64, 48, 32, 512]`; `norm_const` fp32 `[1]` (the FC1
    output's global scale, the checkpoint's `down_proj.input_scale` reciprocal); `alpha1`
    / `alpha2` fp32 `[64]`, the per-expert dequant products. `out` bf16 `[rows, 6144]`
    holds zeros or a partial (the shared expert's) that FC2 adds onto. `c` / `sfc` are
    the FC1 output scratch (`allocate_decode_c` or `allocate_pair_c`, `allocate_sfc`).
    `join`, when given, is a stream the current stream waits for between FC1 and FC2:
    the stream writing the shared expert's partial into `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Decode: one row, an MTP-sized batch, the 64-row ceiling, and the accumulate form
    # (out holding the shared expert's partial, joined between the GEMMs). Prefill: each
    # MMA N width at its edges, a count that is not a tile multiple, and the budget.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=dict(pair=False, rows=1, offset=0)),
        Cell(
            why="8 decode rows, an FC1 output global scale other than 1",
            spec=dict(pair=False, rows=8, offset=64, norm_const=2.0),
        ),
        Cell(why="the full decode batch, 64 rows", spec=dict(pair=False, rows=64, offset=128)),
        Cell(
            why="64 decode rows accumulating onto a partial written on another stream",
            spec=dict(pair=False, rows=64, offset=192, accumulate=True),
        ),
        Cell(why="65 rows, the first prefill count: N128", spec=dict(pair=True, rows=65, offset=0)),
        Cell(
            why="513 rows on the prefill pair, N128, an FC1 output global scale other than 1",
            spec=dict(pair=True, rows=513, offset=64, norm_const=0.5),
        ),
        Cell(why="1536 rows, N128's last count", spec=dict(pair=True, rows=1536, offset=128)),
        Cell(
            why="1537 rows, N256's first count, accumulating onto a joined partial",
            spec=dict(pair=True, rows=1537, offset=192, accumulate=True),
        ),
        Cell(why="8192 rows on the prefill pair, N256", spec=dict(pair=True, rows=8192, offset=0)),
    )

    note = """
    The target builds the decode pair once for 8192 rows (`build_decode`) and the
    prefill pair once (`build_prefill_pair`). Up to 64 rows take the decode pair; 65..8192
    take the prefill pair when it built, else the decode pair. There is no stock path:
    the weights are packed for these kernels alone.

    Rows the caller padded route to `pad_expert_id` at weight 0, so no tile holds them;
    `out`'s padding rows still go through FC2's untouched. FC2 accumulates with atomics,
    so the order of a token's eight contributions, and their bf16 rounding, varies run
    to run. FC1 rounds its SwiGLU output to NVFP4 with the e4m3 scale `amax / 6 *
    norm_const` and approximate reciprocals; `reference` requantizes through
    `fp4_quantize.reference`, so values near an e2m1 midpoint may land on either side.

    The maps, ids and weights handed in must already be visible: FC1 waits on its PDL
    predecessor before reading them, so the caller adds no device work (copy, fill)
    between the producer of these operands and this call.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        norm_const: torch.Tensor,
        alpha1: torch.Tensor,
        alpha2: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        sfc: torch.Tensor,
        join: torch.cuda.Stream | None,
        compiled: CompiledDecodeExperts | CompiledPairExperts,
    ) -> torch.Tensor:
        run = _run_pair if isinstance(compiled, CompiledPairExperts) else _run_decode
        run(
            compiled,
            xq,
            xsf,
            maps,
            topk_w,
            fc1_w,
            fc1_s,
            fc2_w,
            fc2_s,
            norm_const,
            alpha1,
            alpha2,
            out,
            c,
            sfc,
            join,
        )
        return out

    def reference(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        norm_const: torch.Tensor,
        alpha1: torch.Tensor,
        alpha2: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        sfc: torch.Tensor,
        join: torch.cuda.Stream | None,
        compiled: CompiledDecodeExperts | CompiledPairExperts,
    ) -> torch.Tensor:
        """Dequantize, gather, fp32 GEMMs (TF32 off), fp32 SwiGLU, the intermediate
        requantized through `fp4_quantize.reference` with `norm_const` as its global
        scale and read back as code times e4m3 scale (as FC2 reads it), then the weighted
        sum added onto `out` in fp32 and rounded once to bf16. Fresh tensor; `out` is
        read, not written. `c`, `sfc`, `join` and `compiled` are not read."""
        rows = int(xq.shape[0])
        hidden = int(out.shape[1])
        x = dequantize(
            xq.reshape(rows, hidden // 2), xsf.reshape(rows, -1).view(torch.float8_e4m3fn).float()
        )
        s1, s2 = atoms_to_rows(fc1_s), atoms_to_rows(fc2_s)
        w_flat = topk_w.reshape(-1).float()
        acc = out.float().clone()
        gs = norm_const.float().reshape(1)
        with true_fp32_matmul():
            for e, expanded in sorted(expert_rows(maps).items()):
                tokens = expanded // TOP_K
                w1 = dequantize(fc1_w[e], s1[e])
                h = _swiglu_interleaved(x[tokens] @ w1.t(), alpha1[e].float())
                packed, sf = fp4_quantize.reference(h.contiguous(), gs, SF_VEC, False, False)
                hq = dequantize(packed, sf.view(torch.float8_e4m3fn).float().view(h.shape[0], -1))
                y = (hq @ dequantize(fc2_w[e], s2[e]).t()) * alpha2[e].float()
                acc.index_add_(0, tokens, y * w_flat[expanded].unsqueeze(1))
        return acc.to(out.dtype)

    def is_valid(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        norm_const: torch.Tensor,
        alpha1: torch.Tensor,
        alpha2: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        sfc: torch.Tensor,
        join: torch.cuda.Stream | None,
        compiled: CompiledDecodeExperts | CompiledPairExperts,
    ) -> None:
        # The decode pair reads every operand through a raw pointer: nothing checks a
        # shape, a stride or a capacity, and FC2 adds into whatever `out` points at.
        rows = int(xq.shape[0])
        hidden, inter, experts = compiled.hidden, compiled.inter, compiled.experts
        n_fc1 = 2 * inter
        pair = isinstance(compiled, CompiledPairExperts)
        if pair:
            assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS, (
                f"the prefill pair serves {PREFILL_MIN_ROWS}..{PREFILL_MAX_ROWS} rows"
            )
        else:
            assert 1 <= rows <= compiled.max_rows, (
                f"the decode pair is compiled for up to {compiled.max_rows} rows, got {rows}"
            )
        u8, e4, f32 = torch.uint8, torch.float8_e4m3fn, torch.float32
        for name, t, shape, dtype in (
            ("xq", xq, (rows, hidden // 2), u8),
            ("topk_w", topk_w, (rows, TOP_K), f32),
            ("fc1_w", fc1_w, (experts, n_fc1, hidden // 2), u8),
            ("fc1_s", fc1_s, (experts, n_fc1 // SF_ATOM_MN, hidden // SF_ATOM_K, SF_ATOM), e4),
            ("fc2_w", fc2_w, (experts, hidden, inter // 2), u8),
            ("fc2_s", fc2_s, (experts, hidden // SF_ATOM_MN, inter // SF_ATOM_K, SF_ATOM), e4),
            ("norm_const", norm_const, (1,), f32),
            ("alpha1", alpha1, (experts,), f32),
            ("alpha2", alpha2, (experts,), f32),
            ("out", out, (rows, hidden), torch.bfloat16),
        ):
            assert tuple(t.shape) == shape and t.dtype == dtype, (
                f"{name} must be {dtype} {list(shape)}, got {t.dtype} {tuple(t.shape)}"
            )
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"
        assert xsf.numel() == rows * (hidden // SF_VEC) and xsf.element_size() == 1, (
            f"xsf must be {rows * (hidden // SF_VEC)} e4m3 bytes in row-major order"
        )
        assert xsf.is_contiguous() and xsf.data_ptr() % 16 == 0
        # Up to 128 rows an expert holds at most one tile, so the fused top-k's map sets
        # hold `min(64, 8 * rows)` tiles; above, `max_tiles` bounds the count.
        tiles = min(LOCAL_EXPERTS, rows * TOP_K) if rows <= TILE else max_tiles(rows)
        for k, size in (
            ("tile_idx_to_expert_idx", tiles),
            ("tile_idx_to_mn_limit", tiles),
            ("permuted_idx_to_expanded_idx", tiles * TILE),
            ("num_non_exiting_tiles", 1),
        ):
            assert maps[k].dtype == torch.int32 and maps[k].is_contiguous()
            assert maps[k].numel() >= size, f"{k} holds {maps[k].numel()}; {rows} rows need {size}"
        c_cols = inter // 2 if pair else inter
        assert c.dtype == u8 and c.dim() == 3 and c.shape[0] == 1 and c.shape[2] == c_cols, (
            f"c must be uint8 [1, rows, {c_cols}] for the "
            f"{'prefill' if pair else 'decode'} pair, got {tuple(c.shape)}"
        )
        assert c.shape[1] >= tiles * TILE and c.is_contiguous()
        assert sfc.dtype == u8 and tuple(sfc.shape[2:]) == (rest_n_sfc(inter), SF_ATOM)
        assert sfc.shape[1] >= tiles and sfc.is_contiguous()
        assert experts == LOCAL_EXPERTS


cute_dsl_moe_nvfp4_experts = CuteDslMoeNvfp4Experts()
