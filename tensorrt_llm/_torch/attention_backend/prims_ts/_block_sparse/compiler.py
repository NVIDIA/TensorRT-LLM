# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compilation and caching for PrimTS block-sparse attention adapters."""

from collections.abc import Callable
import functools

import torch

from flashinfer.utils import ceil_div

from .config import _BlockSparseCompileKey, _make_block_sparse_config


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"


def _compile_block_sparse(key: _BlockSparseCompileKey) -> Callable[..., object]:
    """Compile one storage-specialized prepare-plus-attention adapter."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from ..kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig
    from ..kernels.fmha_decode.block_sparse_prepare import (
        _PrepareBitmaskRoutes,
        _PrepareBsrRoutes,
    )
    from ..kernels.fmha_decode.fmha_decode_kernel import (
        fmha_block_sparse_launch,
    )

    config = _make_block_sparse_config(key)
    prepare_kwargs = {
        "batch_size": key.batch_size,
        "num_kv_heads": key.num_kv_heads,
        "seq_len_q": key.seq_len_q,
        "seq_len_kv": key.seq_len_kv,
        "q_block_size": key.q_block_size,
        "kv_block_size": key.kv_block_size,
        "kv_route_size": key.kv_route_size,
        "use_proxy_routes": key.use_proxy_routes,
        "use_causal_mask": key.mask_type == "causal",
        "apply_token_mask": key.use_kv_valid_bits,
    }
    if key.page_size is not None:
        if key.sparse_format != "bsr" or key.use_proxy_routes:
            raise AssertionError("paged block-sparse supports exact BSR routes only")
        prepare_kwargs["page_size"] = key.page_size

    if key.sparse_format == "bsr":
        prepare_routes = _PrepareBsrRoutes(**prepare_kwargs)
    elif key.sparse_format == "bitmask":
        prepare_routes = _PrepareBitmaskRoutes(**prepare_kwargs)
    else:
        raise AssertionError("sparse_format must be 'bsr' or 'bitmask'")

    route_metadata_base = prepare_routes.route_metadata_base_word_offset
    Int32 = cutlass.Int32
    Int64 = cutlass.Int64
    Float32 = cutlass.Float32

    @cute.jit
    def exact_bsr_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            block_indptr,
            block_indices,
            kv_valid_bits,
            None,
            None,
            Int64(0),
            Int64(0),
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        # Live per-row route counts occupy the first words of run scratch.
        row_route_counts = route_workspace.iterator
        route_metadata = route_workspace.iterator + Int32(route_metadata_base)
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k.iterator,
            v.iterator,
            k.iterator,
            v.iterator,
            out.iterator,
            row_route_offsets.iterator,
            row_route_counts,
            route_metadata,
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
        )

    @cute.jit
    def exact_bitmask_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        exact_block_bits: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            exact_block_bits,
            kv_valid_bits,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k.iterator,
            v.iterator,
            k.iterator,
            v.iterator,
            out.iterator,
            row_route_offsets.iterator,
            route_workspace.iterator,
            route_workspace.iterator + Int32(route_metadata_base),
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
        )

    @cute.jit
    def proxy_bsr_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        k_summary: cute.Tensor,
        v_summary: cute.Tensor,
        out: cute.Tensor,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            block_indptr,
            block_indices,
            kv_valid_bits,
            None,
            None,
            Int64(0),
            Int64(0),
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k.iterator,
            v.iterator,
            k_summary.iterator,
            v_summary.iterator,
            out.iterator,
            row_route_offsets.iterator,
            route_workspace.iterator,
            route_workspace.iterator + Int32(route_metadata_base),
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
        )

    @cute.jit
    def proxy_bitmask_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        k_summary: cute.Tensor,
        v_summary: cute.Tensor,
        out: cute.Tensor,
        exact_block_bits: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            exact_block_bits,
            kv_valid_bits,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k.iterator,
            v.iterator,
            k_summary.iterator,
            v_summary.iterator,
            out.iterator,
            row_route_offsets.iterator,
            route_workspace.iterator,
            route_workspace.iterator + Int32(route_metadata_base),
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
        )

    @cute.jit
    def paged_tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        block_tables: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        num_physical_kv_pages: cutlass.Int64,
        block_table_row_stride: cutlass.Int64,
        k_page_stride: cutlass.Int64,
        v_page_stride: cutlass.Int64,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            block_indptr,
            block_indices,
            kv_valid_bits,
            seq_lens_kv,
            block_tables,
            num_physical_kv_pages,
            block_table_row_stride,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        row_route_counts = route_workspace.iterator
        route_metadata = route_workspace.iterator + Int32(route_metadata_base)
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k_cache.iterator,
            v_cache.iterator,
            k_cache.iterator,
            v_cache.iterator,
            out.iterator,
            row_route_offsets.iterator,
            row_route_counts,
            route_metadata,
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
            seq_lens_kv.iterator,
            True,
            num_physical_kv_pages,
            k_page_stride,
            v_page_stride,
        )

    def fake_compact(
        dtype: object, shape: tuple[object, ...], alignment: int
    ) -> object:
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=alignment,
        )

    logical_nnz = cute.sym_int()
    logical_workspace_words = cute.sym_int()
    q_shape = (key.batch_size, key.seq_len_q, key.num_qo_heads, key.head_dim)
    num_q_blocks = ceil_div(key.seq_len_q, key.q_block_size)
    num_kv_blocks = ceil_div(key.seq_len_kv, key.kv_block_size)
    indptr_fake = fake_compact(
        Int32,
        (key.batch_size, key.num_kv_heads, num_q_blocks + 1),
        4,
    )
    indices_fake = fake_compact(Int32, (logical_nnz,), 4)
    valid_bits_fake = fake_compact(
        cutlass.Uint32,
        (key.batch_size, ceil_div(key.seq_len_kv, 32)),
        4,
    )
    row_route_offsets_fake = fake_compact(
        Int32,
        (key.batch_size * key.num_kv_heads * num_q_blocks + 1,),
        4,
    )
    route_workspace_fake = fake_compact(
        Int32,
        (logical_workspace_words,),
        4,
    )
    q_fake = fake_compact(config.q_dtype, q_shape, 16)
    out_fake = fake_compact(config.out_dtype, q_shape, 16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    tensor_adapter: Callable[..., object]
    dynamic_args: tuple[object, ...]
    if key.page_size is None:
        kv_shape = (
            key.batch_size,
            key.seq_len_kv,
            key.num_kv_heads,
            key.head_dim,
        )
        k_fake = fake_compact(config.kv_dtype, kv_shape, 16)
        v_fake = fake_compact(config.kv_dtype, kv_shape, 16)
        exact_bits_fake = fake_compact(
            cutlass.Uint32,
            (
                key.batch_size,
                key.num_kv_heads,
                num_q_blocks,
                ceil_div(num_kv_blocks, 32),
            ),
            4,
        )
        common_tail = (
            valid_bits_fake,
            row_route_offsets_fake,
            route_workspace_fake,
            Int32(0),
            Float32(1.0),
        )
        if key.sparse_format == "bsr" and not key.use_proxy_routes:
            tensor_adapter = exact_bsr_adapter
            dynamic_args = (
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                indptr_fake,
                indices_fake,
                *common_tail,
            )
        elif key.sparse_format == "bitmask" and not key.use_proxy_routes:
            tensor_adapter = exact_bitmask_adapter
            dynamic_args = (
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                exact_bits_fake,
                *common_tail,
            )
        elif key.sparse_format == "bsr" and key.use_proxy_routes:
            summary_shape = (
                key.batch_size,
                num_kv_blocks,
                key.num_kv_heads,
                key.head_dim,
            )
            k_summary_fake = fake_compact(config.kv_dtype, summary_shape, 16)
            v_summary_fake = fake_compact(config.kv_dtype, summary_shape, 16)
            proxy_prefix = (
                q_fake,
                k_fake,
                v_fake,
                k_summary_fake,
                v_summary_fake,
                out_fake,
            )
            tensor_adapter = proxy_bsr_adapter
            dynamic_args = (
                *proxy_prefix,
                indptr_fake,
                indices_fake,
                *common_tail,
            )
        elif key.sparse_format == "bitmask" and key.use_proxy_routes:
            summary_shape = (
                key.batch_size,
                num_kv_blocks,
                key.num_kv_heads,
                key.head_dim,
            )
            k_summary_fake = fake_compact(config.kv_dtype, summary_shape, 16)
            v_summary_fake = fake_compact(config.kv_dtype, summary_shape, 16)
            tensor_adapter = proxy_bitmask_adapter
            dynamic_args = (
                q_fake,
                k_fake,
                v_fake,
                k_summary_fake,
                v_summary_fake,
                out_fake,
                exact_bits_fake,
                *common_tail,
            )
        else:
            raise AssertionError("continuous sparse_format must be 'bsr' or 'bitmask'")
    else:
        page_size = key.page_size
        assert page_size is not None
        physical_pages = cute.sym_int()
        runtime_page_columns = cute.sym_int()
        runtime_page_row_stride = cute.sym_int64(divisibility=1)
        k_outer_stride = cute.sym_int64(divisibility=1)
        v_outer_stride = cute.sym_int64(divisibility=1)
        kv_shape = (
            physical_pages,
            key.num_kv_heads,
            page_size,
            key.head_dim,
        )
        k_fake = cute.runtime.make_fake_tensor(
            config.kv_dtype,
            kv_shape,
            stride=(
                k_outer_stride,
                page_size * key.head_dim,
                key.head_dim,
                1,
            ),
            assumed_align=16,
        )
        v_fake = cute.runtime.make_fake_tensor(
            config.kv_dtype,
            kv_shape,
            stride=(
                v_outer_stride,
                page_size * key.head_dim,
                key.head_dim,
                1,
            ),
            assumed_align=16,
        )
        block_tables_fake = cute.runtime.make_fake_tensor(
            Int32,
            (key.batch_size, runtime_page_columns),
            stride=(runtime_page_row_stride, 1),
            assumed_align=4,
        )
        seq_lens_kv_fake = fake_compact(Int32, (key.batch_size,), 4)
        tensor_adapter = paged_tensor_adapter
        dynamic_args = (
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            indptr_fake,
            indices_fake,
            valid_bits_fake,
            block_tables_fake,
            seq_lens_kv_fake,
            row_route_offsets_fake,
            route_workspace_fake,
            Int32(0),
            Int64(1),
            Int64(1),
            Int64(1),
            Int64(1),
            Float32(1.0),
        )

    with torch.cuda.device(key.device_index):
        return cute.compile(
            tensor_adapter,
            *dynamic_args,
            stream_fake,
            config,
            key.batch_size,
            key.seq_len_kv,
            key.num_qo_heads,
            key.num_kv_heads,
            key.head_dim,
            options=_COMPILE_OPTIONS,
        )


@functools.cache
def _get_compiled_block_sparse(
    key: _BlockSparseCompileKey,
) -> Callable[..., object]:
    """Compile and cache one contiguous or paged prepare-plus-attention adapter."""

    return _compile_block_sparse(key)


__all__ = [
    "_compile_block_sparse",
    "_get_compiled_block_sparse",
]
