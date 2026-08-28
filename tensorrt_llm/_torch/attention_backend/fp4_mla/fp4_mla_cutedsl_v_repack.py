# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E741, F841

"""CuTeDSL helpers for repacking the paged FP4 MLA V cache."""

import contextlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass as ctm
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr
from cutlass.experimental import cuda as cuda_exp
from cutlass.experimental import primitives

PREPARED_BUFFER_ALIGNMENT_BYTES = 32
TRTLLM_PAGE_SIZE = 128
SMEM_P4_V_N_PER_CTA = 128

_CUTEDSL_VERBOSE_COMPILE_ENV = "TRTLLM_CUTEDSL_VERBOSE_COMPILE"
_PYIR_STDOUT_LINES = frozenset(
    {
        "Enabling PyIR, it was False",
        "Enabling PyIR, it is now True",
        "Disabling PyIR, it was True",
        "Disabling PyIR, it is now False",
    }
)


class _PyIRStdoutFilter:
    """Drop only CuTeDSL PyIR state transitions from a text stream."""

    def __init__(self, output):
        self._output = output
        self._pending = ""

    def write(self, text):
        lines = (self._pending + text).split("\n")
        self._pending = lines.pop()
        for line in lines:
            if line.rstrip("\r") not in _PYIR_STDOUT_LINES:
                self._output.write(f"{line}\n")
        return len(text)

    def flush(self):
        self._output.flush()

    def finish(self):
        if self._pending and self._pending.rstrip("\r") not in _PYIR_STDOUT_LINES:
            self._output.write(self._pending)
        self._pending = ""
        self._output.flush()

    def __getattr__(self, name):
        return getattr(self._output, name)


def _compile_cutedsl(*args, **kwargs):
    """Compile with PyIR while suppressing only its state-transition lines."""
    verbose = os.getenv(_CUTEDSL_VERBOSE_COMPILE_ENV, "").strip().lower()
    if verbose in {"1", "true", "yes", "on"}:
        with BaseDSL.enable_pyir():
            return cute.compile(*args, **kwargs)

    stdout_filter = _PyIRStdoutFilter(sys.stdout)
    try:
        with contextlib.redirect_stdout(stdout_filter), BaseDSL.enable_pyir():
            return cute.compile(*args, **kwargs)
    finally:
        stdout_filter.finish()


_EXPLICIT_TORCH_STREAM: torch.cuda.Stream | None = None


def _current_cu_stream() -> cuda.CUstream:
    """Return the active PyTorch stream as a CUDA driver stream.

    cuda2ctl capture cannot reliably query the default null stream.  Keep the
    default behavior for normal pytest/AModel runs, but allow SMART wrappers to
    request an explicit stream through the environment.
    """
    global _EXPLICIT_TORCH_STREAM
    if os.environ.get("DKG_MLA_EXPLICIT_STREAM") == "1":
        if _EXPLICIT_TORCH_STREAM is None:
            _EXPLICIT_TORCH_STREAM = torch.cuda.Stream()
        torch.cuda.set_stream(_EXPLICIT_TORCH_STREAM)
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@dataclass(frozen=True)
class KvCache3DLayout:
    """3D view metadata for a paged FP4 KV cache tensor."""

    num_pages: int
    page_size: int
    packed_dim: int
    stride_page: int
    stride_token: int
    stride_packed_dim: int


def _validate_tensor_pointer_alignment(
    name: str,
    tensor: torch.Tensor,
    alignment_bytes: int = PREPARED_BUFFER_ALIGNMENT_BYTES,
) -> None:
    """Require the pointer alignment promised to generated GMEM accesses."""
    pointer = tensor.data_ptr()
    if pointer % alignment_bytes != 0:
        raise ValueError(
            f"{name} data pointer must be {alignment_bytes}B aligned, "
            f"got address remainder {pointer % alignment_bytes}"
        )


def _make_v_repack_ptrs(
    v_packed_data_ptr: int,
    kv_cache_data_ptr: int,
    page_ids_data_ptr: int,
    page_indptr_data_ptr: int = 0,
    kv_lens_data_ptr: int = 0,
    generation_lens_data_ptr: int = 0,
) -> tuple[cute.Pointer, ...]:
    return (
        make_ptr(
            ctm.Uint8,
            v_packed_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            ctm.Uint8,
            kv_cache_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            ctm.Int32,
            page_ids_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            ctm.Int32,
            page_indptr_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            ctm.Int32,
            kv_lens_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            ctm.Int32,
            generation_lens_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
    )


@cute.kernel
def fp4_mla_v_repack_kernel(
    mVPacked: cute.Tensor,
    mKvCache: cute.Tensor,
    mPageIds: cute.Tensor,
    page_size: ctm.Constexpr,
    v_head_dim: ctm.Constexpr,
    block_v: ctm.Constexpr,
    use_page_ids: ctm.Constexpr,
) -> None:
    """Repack token-major paged V bytes into the decode kernel's V layout.

    ``mKvCache`` exposes ``[num_pages, page_size, v_head_dim / 2]`` bytes,
    where each byte packs two adjacent V features for one token. ``mVPacked``
    instead packs one feature for two adjacent tokens, matching the decode
    kernel's ``[(page, dim_block, dim), token_pair]`` cache contract.
    """
    page_list_idx, dim_block, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bdimx, _, _ = cute.arch.block_dim()

    page_pairs: ctm.Constexpr = page_size // 2
    packed_dims_per_block: ctm.Constexpr = block_v // 2
    input_bytes_per_tile: ctm.Constexpr = page_size * packed_dims_per_block
    output_bytes_per_tile: ctm.Constexpr = block_v * page_pairs
    num_dim_blocks: ctm.Constexpr = v_head_dim // block_v
    # A one-byte pad breaks the 128-byte alias between adjacent packed
    # dimensions.  The first phase therefore reads the token-major cache with
    # coalesced GMEM accesses, while the second phase reads token pairs and
    # writes the decode-ready dim-major tile with coalesced GMEM stores.
    smem_token_stride: ctm.Constexpr = page_size + 1
    sVBytes = ctm.Array(
        ctm.Uint8,
        packed_dims_per_block * smem_token_stride,
        space=ctm.AddressSpace.smem,
        alignment=16,
    )

    page_idx = page_list_idx
    if ctm.const_expr(use_page_ids):
        page_idx = ctm.Int32(mPageIds[page_list_idx])

    for linear_idx in ctm.range(tidx, input_bytes_per_tile, bdimx):
        token = linear_idx // ctm.Int32(packed_dims_per_block)
        packed_dim_in_block = linear_idx - token * ctm.Int32(packed_dims_per_block)
        packed_dim = dim_block * ctm.Int32(packed_dims_per_block) + packed_dim_in_block
        smem_idx = packed_dim_in_block * ctm.Int32(smem_token_stride) + token
        sVBytes[smem_idx] = mKvCache[page_idx, token, packed_dim]

    primitives.barrier(barrier_id=0, number_of_threads=256)

    for linear_idx in ctm.range(tidx, output_bytes_per_tile, bdimx):
        dim_in_block = linear_idx // ctm.Int32(page_pairs)
        token_pair = linear_idx - dim_in_block * ctm.Int32(page_pairs)
        packed_dim_in_block = dim_in_block >> ctm.Int32(1)
        token0 = token_pair << ctm.Int32(1)
        token1 = token0 + ctm.Int32(1)

        smem_base = packed_dim_in_block * ctm.Int32(smem_token_stride)
        even_token_byte = ctm.Int32(sVBytes[smem_base + token0])
        odd_token_byte = ctm.Int32(sVBytes[smem_base + token1])
        out_byte = ctm.Int32(0)
        if (dim_in_block & ctm.Int32(1)) == ctm.Int32(0):
            out_byte = (even_token_byte & ctm.Int32(0x0F)) | (
                (odd_token_byte & ctm.Int32(0x0F)) << ctm.Int32(4)
            )
        else:
            out_byte = ((even_token_byte >> ctm.Int32(4)) & ctm.Int32(0x0F)) | (
                odd_token_byte & ctm.Int32(0xF0)
            )

        # V2 coalesces the per-layer cache and encodes a physical slot as
        # ``slot * num_local_layers``.  The row coordinate still fits in
        # Int32 for current pool sizes, but its byte offset can exceed 2 GiB
        # (PAGE128/V512 crosses that boundary at encoded page 65536).  Keep
        # the complete output-address chain in Int64 so large disaggregated
        # fill batches cannot wrap into an unrelated allocation.
        out_row = (
            ctm.Int64(page_idx) * ctm.Int64(num_dim_blocks) + ctm.Int64(dim_block)
        ) * ctm.Int64(block_v) + ctm.Int64(dim_in_block)
        mVPacked[out_row, token_pair] = ctm.Uint8(out_byte & ctm.Int32(0xFF))


@cute.kernel
def fp4_mla_v_repack_tma_kernel(
    src_desc: ctm.GridConstant[cuda_exp.TensorMap],
    mVPacked: cute.Tensor,
    mPageIds: cute.Tensor,
    mPageIndptr: cute.Tensor,
    mKvLens: cute.Tensor,
    mGenerationLens: cute.Tensor,
    num_pages: ctm.Int32,
    num_page_ids: ctm.Int32,
    num_dim_blocks: ctm.Constexpr,
    use_page_ids: ctm.Constexpr,
    resolve_generation_pages: ctm.Constexpr,
    max_touched_pages: ctm.Constexpr,
) -> None:
    """Bulk PAGE128/BLOCKV128 V repack for the persistent TRT cache.

    One TMA transaction reads each 8 KiB token-major source tile once.  The
    CTA performs the nibble transpose from that SMEM image and emits aligned
    16-byte stores to the physical-page-indexed packed cache.
    """
    page_list_idx, dim_block, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // ctm.Int32(32)
    page_size: ctm.Constexpr = TRTLLM_PAGE_SIZE
    block_v: ctm.Constexpr = SMEM_P4_V_N_PER_CTA
    packed_dims_per_block: ctm.Constexpr = block_v // 2
    page_pairs: ctm.Constexpr = page_size // 2
    groups_per_row: ctm.Constexpr = page_pairs // 16
    input_tile_bytes: ctm.Constexpr = page_size * packed_dims_per_block

    sVBytes = ctm.Array(
        ctm.Uint8,
        input_tile_bytes,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    tma_mbar = ctm.Array(
        ctm.Int64,
        1,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    page_idx = ctm.Int32(0)
    page_valid = ctm.Int32(1)
    if ctm.const_expr(resolve_generation_pages):
        sequence_idx = page_list_idx // ctm.Int32(max_touched_pages)
        touched_page_idx = page_list_idx - sequence_idx * ctm.Int32(max_touched_pages)
        kv_len = ctm.Int32(mKvLens[sequence_idx])
        generation_len = ctm.Int32(mGenerationLens[sequence_idx])
        if kv_len <= ctm.Int32(0) or generation_len <= ctm.Int32(0):
            page_valid = ctm.Int32(0)
        first_token = kv_len - generation_len
        if first_token < ctm.Int32(0):
            first_token = ctm.Int32(0)
        first_page = first_token // ctm.Int32(page_size)
        last_page = (kv_len - ctm.Int32(1)) // ctm.Int32(page_size)
        logical_page = first_page + touched_page_idx
        page_begin = ctm.Int32(mPageIndptr[sequence_idx])
        page_end = ctm.Int32(mPageIndptr[sequence_idx + ctm.Int32(1)])
        page_count = page_end - page_begin
        if page_begin < ctm.Int32(0) or page_end < page_begin or page_end > num_page_ids:
            page_valid = ctm.Int32(0)
        if (
            logical_page > last_page
            or logical_page < ctm.Int32(0)
            or logical_page >= page_count
            or page_count <= ctm.Int32(0)
        ):
            page_valid = ctm.Int32(0)
        if page_valid != ctm.Int32(0):
            physical_page = ctm.Int32(mPageIds[page_begin + logical_page])
            if physical_page < ctm.Int32(0) or physical_page >= num_pages:
                page_valid = ctm.Int32(0)
            else:
                page_idx = physical_page
    elif ctm.const_expr(use_page_ids):
        page_idx = ctm.Int32(mPageIds[page_list_idx])
    else:
        page_idx = ctm.Int32(page_list_idx)

    if warp_idx == ctm.Int32(0):
        if primitives.elect_sync():
            primitives.mbarrier_init(tma_mbar, 1)
    primitives.fence_mbarrier_init()
    primitives.barrier()

    if warp_idx == ctm.Int32(0):
        if primitives.elect_sync():
            primitives.mbarrier_arrive_expect_tx(tma_mbar, input_tile_bytes)
            primitives.cp_async_bulk_tensor_shared_cta_global(
                sVBytes,
                src_desc.get_ptr(),
                [
                    dim_block * ctm.Int32(packed_dims_per_block),
                    ctm.Int32(0),
                    page_idx,
                ],
                tma_mbar,
            )
    while not primitives.mbarrier_try_wait_parity(tma_mbar, 0, time_limit=10_000_000):
        pass
    primitives.barrier()

    packed_dim_in_block = tidx // ctm.Int32(groups_per_row)
    token_group = tidx - packed_dim_in_block * ctm.Int32(groups_per_row)
    low_bytes = ctm.Array(ctm.Uint8, 16, space=ctm.AddressSpace.rmem)
    high_bytes = ctm.Array(ctm.Uint8, 16, space=ctm.AddressSpace.rmem)
    smem_swizzle = ctm.Swizzle.from_name(str(src_desc.swizzle))
    for elem_idx in ctm.range_constexpr(16):
        token_pair = token_group * ctm.Int32(16) + ctm.Int32(elem_idx)
        token0 = token_pair << ctm.Int32(1)
        token1 = token0 + ctm.Int32(1)
        even_offset = token0 * ctm.Int32(packed_dims_per_block) + packed_dim_in_block
        odd_offset = token1 * ctm.Int32(packed_dims_per_block) + packed_dim_in_block
        even_token_byte = ctm.Int32(
            sVBytes.subview(even_offset).data_ptr().load_swizzled(smem_swizzle)
        )
        odd_token_byte = ctm.Int32(
            sVBytes.subview(odd_offset).data_ptr().load_swizzled(smem_swizzle)
        )
        low_bytes[elem_idx] = ctm.Uint8(
            (even_token_byte & ctm.Int32(0x0F))
            | ((odd_token_byte & ctm.Int32(0x0F)) << ctm.Int32(4))
        )
        high_bytes[elem_idx] = ctm.Uint8(
            ((even_token_byte >> ctm.Int32(4)) & ctm.Int32(0x0F))
            | (odd_token_byte & ctm.Int32(0xF0))
        )

    # ``gVPacked`` is a flattened byte view.  With PAGE128/V512 each encoded
    # page contributes 32 KiB, so Int32 byte indexing wraps at page 65536.
    # FP4 MLA V2 can exceed that encoded page ID under high-concurrency
    # disaggregated fill because every physical slot is scaled by the number
    # of local layers.  Promote before any multiply in the address chain.
    row_base = (ctm.Int64(page_idx) * ctm.Int64(num_dim_blocks) + ctm.Int64(dim_block)) * ctm.Int64(
        block_v
    ) + ctm.Int64(packed_dim_in_block) * ctm.Int64(2)
    col = ctm.Int64(token_group) * ctm.Int64(16)
    gVPacked = ctm.make_array_view(mVPacked)
    if page_valid != ctm.Int32(0):
        gVPacked.store(
            low_bytes.load(0, 16, alignment=16),
            idx=row_base * ctm.Int64(page_pairs) + col,
            vector_size=16,
            alignment=16,
        )
        gVPacked.store(
            high_bytes.load(0, 16, alignment=16),
            idx=(row_base + ctm.Int64(1)) * ctm.Int64(page_pairs) + col,
            vector_size=16,
            alignment=16,
        )


@cute.jit
def fp4_mla_v_repack_host(
    v_packed_ptr: cute.Pointer,
    kv_cache_ptr: cute.Pointer,
    page_ids_ptr: cute.Pointer,
    page_indptr_ptr: cute.Pointer,
    kv_lens_ptr: cute.Pointer,
    generation_lens_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_pages: ctm.Int32,
    page_size: ctm.Constexpr,
    v_head_dim: ctm.Constexpr,
    kv_s0: ctm.Int64,
    kv_s_token: ctm.Int64,
    kv_s_packed_dim: ctm.Int64,
    out_s0: ctm.Int64,
    out_s1: ctm.Int64,
    page_ids_stride: ctm.Int64,
    num_page_ids: ctm.Int32,
    num_generation_sequences: ctm.Int32,
    use_page_ids: ctm.Constexpr,
    resolve_generation_pages: ctm.Constexpr,
    max_touched_pages: ctm.Constexpr,
    block_v: ctm.Constexpr,
    use_tma_fast_path: ctm.Constexpr,
) -> None:
    """Launch V repack with the canonical paged-cache semantics."""
    if ctm.const_expr(use_tma_fast_path):
        # TMA descriptor construction must see one statically known leading
        # mode. Eligibility validation guarantees packed-dim stride == 1.
        kv_layout = cute.make_layout(
            (num_pages, page_size, v_head_dim // 2),
            stride=(kv_s0, kv_s_token, 1),
        )
    else:
        kv_layout = cute.make_layout(
            (num_pages, page_size, v_head_dim // 2),
            stride=(kv_s0, kv_s_token, kv_s_packed_dim),
        )
    num_dim_blocks: ctm.Constexpr = v_head_dim // block_v
    v_packed_layout = cute.make_layout(
        (num_pages * num_dim_blocks * block_v, page_size // 2),
        stride=(out_s0, out_s1),
    )
    if ctm.const_expr(resolve_generation_pages):
        page_ids_layout = cute.make_layout((num_page_ids,), stride=(page_ids_stride,))
        page_indptr_layout = cute.make_layout((num_generation_sequences + 1,), stride=(1,))
        generation_lengths_layout = cute.make_layout((num_generation_sequences,), stride=(1,))
        launch_pages = num_generation_sequences * max_touched_pages
    elif ctm.const_expr(use_page_ids):
        page_ids_layout = cute.make_layout((num_page_ids,), stride=(page_ids_stride,))
        page_indptr_layout = cute.make_layout((1,), stride=(1,))
        generation_lengths_layout = cute.make_layout((1,), stride=(1,))
        launch_pages = num_page_ids
    else:
        page_ids_layout = cute.make_layout((1,), stride=(1,))
        page_indptr_layout = cute.make_layout((1,), stride=(1,))
        generation_lengths_layout = cute.make_layout((1,), stride=(1,))
        launch_pages = num_pages
    mVPacked = cute.make_tensor(v_packed_ptr, v_packed_layout)
    mKvCache = cute.make_tensor(kv_cache_ptr, kv_layout)
    mPageIds = cute.make_tensor(page_ids_ptr, page_ids_layout)
    mPageIndptr = cute.make_tensor(page_indptr_ptr, page_indptr_layout)
    mKvLens = cute.make_tensor(kv_lens_ptr, generation_lengths_layout)
    mGenerationLens = cute.make_tensor(generation_lens_ptr, generation_lengths_layout)
    if ctm.const_expr(use_tma_fast_path):
        src_desc = cuda_exp.create_tensor_map_tiled_from_view(
            mKvCache,
            box_dims=(1, page_size, block_v // 2),
            stride_order=(2, 1, 0),
            # The transpose reads scalar bytes from different token rows, so
            # a linear SMEM image avoids per-byte XOR address de-swizzling.
            swizzle=cuda_exp.TensorMapSwizzle.none,
            tma_format=cuda_exp.TensorMapDataFormat.BYTE,
        )
        fp4_mla_v_repack_tma_kernel(
            src_desc,
            mVPacked,
            mPageIds,
            mPageIndptr,
            mKvLens,
            mGenerationLens,
            num_pages,
            num_page_ids,
            num_dim_blocks,
            use_page_ids,
            resolve_generation_pages,
            max_touched_pages,
        ).launch(
            grid=(launch_pages, num_dim_blocks, 1),
            block=(256, 1, 1),
            stream=stream,
        )
        return
    fp4_mla_v_repack_kernel(
        mVPacked,
        mKvCache,
        mPageIds,
        page_size,
        v_head_dim,
        block_v,
        use_page_ids,
    ).launch(
        grid=(launch_pages, num_dim_blocks, 1),
        block=(256, 1, 1),
        stream=stream,
    )


def _kv_cache_3d_layout(kv_cache: torch.Tensor, page_size: int) -> KvCache3DLayout:
    """Return the canonical 3D view consumed by the V repack helper."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be torch.uint8, got {kv_cache.dtype}")
    if page_size <= 0 or page_size % 2 != 0:
        raise ValueError(f"page_size must be a positive even integer, got {page_size}")
    if kv_cache.dim() == 3:
        num_pages, actual_page_size, packed_dim = kv_cache.shape
        stride_page, stride_token, stride_packed_dim = kv_cache.stride()
    elif kv_cache.dim() >= 5:
        num_pages = kv_cache.shape[0]
        actual_page_size = kv_cache.shape[2]
        packed_dim = kv_cache.shape[4]
        stride_page = kv_cache.stride(0)
        stride_token = kv_cache.stride(2)
        stride_packed_dim = kv_cache.stride(4)
    else:
        raise ValueError(
            "kv_cache must be shaped as [num_pages, page_size, packed_dim] "
            "or expose TRT-LLM's 5D paged layout with page/token/packed-dim "
            f"axes at 0/2/4, got shape={tuple(kv_cache.shape)}"
        )
    if actual_page_size != page_size:
        raise ValueError(f"page_size mismatch: argument={page_size}, cache={actual_page_size}")
    return KvCache3DLayout(
        num_pages=int(num_pages),
        page_size=int(actual_page_size),
        packed_dim=int(packed_dim),
        stride_page=int(stride_page),
        stride_token=int(stride_token),
        stride_packed_dim=int(stride_packed_dim),
    )


def _kv_cache_3d_view(kv_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    layout = _kv_cache_3d_layout(kv_cache, page_size)
    return torch.as_strided(
        kv_cache,
        size=(layout.num_pages, layout.page_size, layout.packed_dim),
        stride=(layout.stride_page, layout.stride_token, layout.stride_packed_dim),
    )


def _validate_v_repack_args(
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor | None,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
) -> KvCache3DLayout:
    if v_packed.dtype != torch.uint8:
        raise TypeError(f"v_packed must be torch.uint8, got {v_packed.dtype}")
    if v_packed.dim() != 2:
        raise ValueError(f"v_packed must be 2D, got shape={tuple(v_packed.shape)}")
    if v_head_dim <= 0 or v_head_dim % 2 != 0:
        raise ValueError(f"v_head_dim must be positive and even, got {v_head_dim}")
    if block_v <= 0 or v_head_dim % block_v != 0:
        raise ValueError(f"v_head_dim={v_head_dim} must be divisible by block_v={block_v}")
    layout = _kv_cache_3d_layout(kv_cache, page_size)
    if layout.packed_dim < v_head_dim // 2:
        raise ValueError(
            "kv_cache packed dimension does not cover v_head_dim: "
            f"packed_dim={layout.packed_dim}, v_head_dim={v_head_dim}"
        )
    expected_rows = layout.num_pages * (v_head_dim // block_v) * block_v
    expected_cols = page_size // 2
    if v_packed.shape[0] < expected_rows or v_packed.shape[1] < expected_cols:
        raise ValueError(
            "v_packed is too small for the decode V-cache layout: "
            f"got shape={tuple(v_packed.shape)}, needs at least "
            f"({expected_rows}, {expected_cols})"
        )
    out_s0, out_s1 = (int(stride) for stride in v_packed.stride())
    if out_s0 <= 0 or out_s1 <= 0:
        raise ValueError(f"v_packed strides must be positive, got {(out_s0, out_s1)}")
    minimum_row_stride = (expected_cols - 1) * out_s1 + 1
    if out_s0 < minimum_row_stride:
        raise ValueError(
            "v_packed rows must not overlap: "
            f"stride={(out_s0, out_s1)}, minimum row stride={minimum_row_stride}"
        )
    if v_packed.device != kv_cache.device:
        raise ValueError(
            "v_packed and kv_cache must be on the same device, got "
            f"{v_packed.device} and {kv_cache.device}"
        )
    if (
        min(
            layout.stride_page,
            layout.stride_token,
            layout.stride_packed_dim,
        )
        <= 0
    ):
        raise ValueError(
            "kv_cache strides must be positive, got "
            f"{(layout.stride_page, layout.stride_token, layout.stride_packed_dim)}"
        )
    _validate_tensor_pointer_alignment("v_packed", v_packed, alignment_bytes=16)
    _validate_tensor_pointer_alignment("kv_cache", kv_cache, alignment_bytes=16)
    if page_ids is not None:
        if page_ids.dtype != torch.int32:
            raise TypeError(f"page_ids must be torch.int32, got {page_ids.dtype}")
        if page_ids.dim() != 1:
            raise ValueError(f"page_ids must be 1D, got shape={tuple(page_ids.shape)}")
        if page_ids.device != kv_cache.device:
            raise ValueError(
                "page_ids and kv_cache must be on the same device, got "
                f"{page_ids.device} and {kv_cache.device}"
            )
        if page_ids.stride(0) <= 0:
            raise ValueError(f"page_ids stride must be positive, got {page_ids.stride(0)}")
        _validate_tensor_pointer_alignment("page_ids", page_ids, alignment_bytes=4)
    return layout


_V_REPACK_COMPILE_CACHE: dict[tuple[object, ...], Callable] = {}


def _compile_fp4_mla_v_repack(
    v_packed_data_ptr: int,
    kv_cache_data_ptr: int,
    page_ids_data_ptr: int,
    page_indptr_data_ptr: int,
    kv_lens_data_ptr: int,
    generation_lens_data_ptr: int,
    page_size: int,
    v_head_dim: int,
    use_page_ids: bool,
    resolve_generation_pages: bool,
    max_touched_pages: int,
    block_v: int,
    use_tma_fast_path: bool,
    stream: cuda.CUstream,
) -> Callable:
    cache_key = (
        page_size,
        v_head_dim,
        use_page_ids,
        resolve_generation_pages,
        max_touched_pages,
        block_v,
        use_tma_fast_path,
    )
    cached = _V_REPACK_COMPILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ptrs = _make_v_repack_ptrs(
        v_packed_data_ptr,
        kv_cache_data_ptr,
        page_ids_data_ptr,
        page_indptr_data_ptr,
        kv_lens_data_ptr,
        generation_lens_data_ptr,
    )
    compiled = _compile_cutedsl(
        fp4_mla_v_repack_host,
        *ptrs,
        stream,
        ctm.Int32(1),
        page_size,
        v_head_dim,
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int32(1),
        ctm.Int32(1),
        use_page_ids,
        resolve_generation_pages,
        max_touched_pages,
        block_v,
        use_tma_fast_path,
    )
    _V_REPACK_COMPILE_CACHE[cache_key] = compiled
    return compiled


def fp4_mla_repack_v_cache(
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor | None = None,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int = 128,
    page_indptr: torch.Tensor | None = None,
    kv_lens: torch.Tensor | None = None,
    generation_lens: torch.Tensor | None = None,
    max_touched_pages: int = 1,
) -> None:
    """Populate decode-ready V bytes from a paged FP4 latent KV cache.

    ``kv_cache`` may be the compact 3D view
    ``[num_pages, page_size, v_head_dim / 2]`` or TRT-LLM's 5D paged cache.
    The launch is asynchronous on the current CUDA stream.  When all three
    generation metadata tensors are supplied, the TMA CTA resolves its
    physical page directly from the device CSR table and no page-ID worklist
    is materialized.
    """
    layout = _validate_v_repack_args(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
    )
    if layout.num_pages == 0:
        return
    use_page_ids = page_ids is not None
    if use_page_ids and page_ids is not None and page_ids.numel() == 0:
        return

    generation_metadata = (page_indptr, kv_lens, generation_lens)
    resolve_generation_pages = any(tensor is not None for tensor in generation_metadata)
    if resolve_generation_pages and not all(tensor is not None for tensor in generation_metadata):
        raise ValueError("page_indptr, kv_lens, and generation_lens must be provided together")
    num_generation_sequences = 0
    if resolve_generation_pages:
        if page_ids is None:
            raise ValueError("generation-aware V repack requires page_ids")
        if max_touched_pages <= 0:
            raise ValueError(f"max_touched_pages must be positive, got {max_touched_pages}")
        assert page_indptr is not None
        assert kv_lens is not None
        assert generation_lens is not None
        for name, tensor in (
            ("page_indptr", page_indptr),
            ("kv_lens", kv_lens),
            ("generation_lens", generation_lens),
        ):
            if tensor.dtype != torch.int32 or tensor.dim() != 1:
                raise TypeError(f"{name} must be a one-dimensional torch.int32 tensor")
            if tensor.device != kv_cache.device or tensor.stride(0) != 1:
                raise ValueError(f"{name} must be contiguous and on {kv_cache.device}")
            _validate_tensor_pointer_alignment(name, tensor, alignment_bytes=4)
        num_generation_sequences = int(kv_lens.numel())
        if generation_lens.numel() != num_generation_sequences:
            raise ValueError("kv_lens and generation_lens must have equal lengths")
        if page_indptr.numel() < num_generation_sequences + 1:
            raise ValueError(
                "page_indptr must have at least one more entry than generation lengths"
            )

    page_ids_data_ptr = 0 if page_ids is None else page_ids.data_ptr()
    page_ids_stride = 1 if page_ids is None else int(page_ids.stride(0))
    num_page_ids = 0 if page_ids is None else int(page_ids.numel())
    page_indptr_data_ptr = 0 if page_indptr is None else page_indptr.data_ptr()
    kv_lens_data_ptr = 0 if kv_lens is None else kv_lens.data_ptr()
    generation_lens_data_ptr = 0 if generation_lens is None else generation_lens.data_ptr()
    use_tma_fast_path = (
        page_size == TRTLLM_PAGE_SIZE
        and block_v == SMEM_P4_V_N_PER_CTA
        and layout.stride_packed_dim == 1
        and int(v_packed.stride(0)) == page_size // 2
        and int(v_packed.stride(1)) == 1
    )
    if resolve_generation_pages and not use_tma_fast_path:
        raise ValueError("generation-aware V repack requires the PAGE128/BLOCKV128 TMA path")
    stream = _current_cu_stream()
    repack_fn = _compile_fp4_mla_v_repack(
        v_packed.data_ptr(),
        kv_cache.data_ptr(),
        page_ids_data_ptr,
        page_indptr_data_ptr,
        kv_lens_data_ptr,
        generation_lens_data_ptr,
        page_size,
        v_head_dim,
        use_page_ids,
        resolve_generation_pages,
        max_touched_pages,
        block_v,
        use_tma_fast_path,
        stream,
    )
    ptrs = _make_v_repack_ptrs(
        v_packed.data_ptr(),
        kv_cache.data_ptr(),
        page_ids_data_ptr,
        page_indptr_data_ptr,
        kv_lens_data_ptr,
        generation_lens_data_ptr,
    )
    repack_fn(
        *ptrs,
        stream,
        ctm.Int32(layout.num_pages),
        ctm.Int64(layout.stride_page),
        ctm.Int64(layout.stride_token),
        ctm.Int64(layout.stride_packed_dim),
        ctm.Int64(v_packed.stride(0)),
        ctm.Int64(v_packed.stride(1)),
        ctm.Int64(page_ids_stride),
        ctm.Int32(num_page_ids),
        ctm.Int32(num_generation_sequences),
    )


def fp4_mla_repack_v_cache_reference(
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor | None = None,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int = 128,
) -> None:
    """Torch reference for ``fp4_mla_repack_v_cache``; writes in-place."""
    layout = _validate_v_repack_args(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
    )
    kv_view = _kv_cache_3d_view(kv_cache, page_size)
    if page_ids is None:
        pages = range(layout.num_pages)
    else:
        pages = [int(page) for page in page_ids.detach().cpu().tolist()]
    num_dim_blocks = v_head_dim // block_v
    for page_idx in pages:
        for dim in range(v_head_dim):
            src_bytes = kv_view[page_idx, :, dim // 2]
            if dim % 2 == 0:
                repacked = (src_bytes[0::2] & 0x0F) | ((src_bytes[1::2] & 0x0F) << 4)
            else:
                repacked = ((src_bytes[0::2] >> 4) & 0x0F) | (src_bytes[1::2] & 0xF0)
            dim_block = dim // block_v
            dim_in_block = dim - dim_block * block_v
            out_row = (page_idx * num_dim_blocks + dim_block) * block_v + dim_in_block
            v_packed[out_row, : page_size // 2] = repacked
