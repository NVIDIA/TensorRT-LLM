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

from .common import _num_sparse_pattern_heads
from ..sage import sage_adapter_slots, sage_scale_shapes
from .config import _BlockSparseCompileKey, _make_block_sparse_config


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"


def _compile_block_sparse(key: _BlockSparseCompileKey) -> Callable[..., object]:
    """Compile one contiguous or paged attention adapter for ``key``.

    Every contiguous plan, dense or block-sparse, exact or proxy, BSR or
    bitmask, compiles the same adapter signature. The plan decides at compile
    time which tensor slots it uses; the unused slots are ``None`` both for the
    compile-time fakes and for every later call, and the adapter body reads a
    slot only on the branch its configuration enables. A dense plan runs no
    prepare kernel and reaches the decode kernel directly; a block-sparse plan
    prepares its routes and launches the prepared-route attention. Paged
    block-sparse plans keep their own adapter because their K/V and request
    metadata differ in kind, not only in presence.
    """

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
        fmha_decode_launch,
    )

    config = _make_block_sparse_config(key)
    sparse_format = key.sparse_format
    use_proxy_routes = key.use_proxy_routes
    prepare_routes: _PrepareBsrRoutes | _PrepareBitmaskRoutes | None = None
    route_metadata_base = 0
    pattern_heads = _num_sparse_pattern_heads(
        key.num_kv_heads, key.share_pattern_across_kv_heads
    )
    if key.use_block_sparse:
        prepare_kwargs = {
            "batch_size": key.batch_size,
            "num_kv_heads": pattern_heads,
            "seq_len_q": key.seq_len_q,
            "seq_len_kv": key.seq_len_kv,
            "q_block_size": key.q_block_size,
            "kv_block_size": key.kv_block_size,
            "kv_route_size": key.kv_route_size,
            "use_proxy_routes": use_proxy_routes,
            "use_causal_mask": key.mask_type == "causal",
            "apply_token_mask": key.use_kv_valid_bits,
            "store_score_words": config.uses_prepared_score_keep_words,
        }
        if key.page_size is not None:
            if sparse_format != "bsr" or use_proxy_routes:
                raise AssertionError(
                    "paged block-sparse supports exact BSR routes only"
                )
            prepare_kwargs["page_size"] = key.page_size
        if sparse_format == "bsr":
            prepare_routes = _PrepareBsrRoutes(**prepare_kwargs)
        elif sparse_format == "bitmask":
            prepare_routes = _PrepareBitmaskRoutes(**prepare_kwargs)
        else:
            raise AssertionError("sparse_format must be 'bsr' or 'bitmask'")
        route_metadata_base = prepare_routes.route_metadata_base_word_offset
    elif key.page_size is not None:
        raise AssertionError("dense attention requires contiguous K/V")

    Int32 = cutlass.Int32
    Int64 = cutlass.Int64
    Float32 = cutlass.Float32

    @cute.jit
    def contiguous_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        k_summary: cute.Tensor | None,
        v_summary: cute.Tensor | None,
        out: cute.Tensor,
        block_indptr: cute.Tensor | None,
        block_indices: cute.Tensor | None,
        exact_block_bits: cute.Tensor | None,
        kv_valid_bits: cute.Tensor | None,
        row_route_offsets: cute.Tensor | None,
        route_workspace: cute.Tensor | None,
        max_blocks_per_row: cutlass.Int32,
        # The Sage scale slots, in ``SAGE_ADAPTER_SLOTS`` order.
        q_scale: cute.Tensor | None,
        k_scale: cute.Tensor | None,
        k_summary_scale: cute.Tensor | None,
        v_scale: cute.Tensor | None,
        v_mean: cute.Tensor | None,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        problem_shape = (
            Int32(static_batch_size),
            Int32(static_num_qo_heads),
            Int32(static_num_kv_heads),
            Int32(static_seq_len_kv),
            Int32(static_head_dim),
        )
        # Sage scale tensors are [heads, flat slots]; the head stride is the
        # flat slot count of one head. A slot the recipe does not use (the V
        # mean without a mean, the summary K scales without proxy routes) is
        # ``None`` and stays unbound.
        sage_kwargs = {}
        if cutlass.const_expr(static_config.use_sage_attention):
            sage_kwargs = {
                "sage_q_scale_iter": q_scale.iterator,
                "sage_k_scale_iter": k_scale.iterator,
                "sage_v_scale_iter": v_scale.iterator,
                "sage_q_scale_head_stride": Int32(q_scale.shape[1]),
                "sage_k_scale_head_stride": Int32(k_scale.shape[1]),
            }
            if cutlass.const_expr(static_config.sage_v_mean):
                sage_kwargs["sage_v_mean_iter"] = v_mean.iterator
            if cutlass.const_expr(static_config.use_block_sparse_proxy_routes):
                sage_kwargs["sage_k_summary_scale_iter"] = k_summary_scale.iterator
                sage_kwargs["sage_k_summary_scale_head_stride"] = Int32(
                    k_summary_scale.shape[1]
                )
        if cutlass.const_expr(static_config.use_block_sparse):
            if cutlass.const_expr(sparse_format == "bsr"):
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
            else:
                prepare_routes(
                    exact_block_bits,
                    kv_valid_bits,
                    row_route_offsets,
                    route_workspace,
                    max_blocks_per_row,
                    stream,
                )
            # Exact routes address K/V through the summary TensorMaps as well;
            # proxy routes address the caller's block summaries.
            if cutlass.const_expr(static_config.use_block_sparse_proxy_routes):
                k_summary_iter = k_summary.iterator
                v_summary_iter = v_summary.iterator
            else:
                k_summary_iter = k.iterator
                v_summary_iter = v.iterator
            # Live per-row route counts occupy the first words of run scratch.
            fmha_block_sparse_launch(
                problem_shape,
                q.iterator,
                k.iterator,
                v.iterator,
                k_summary_iter,
                v_summary_iter,
                out.iterator,
                row_route_offsets.iterator,
                route_workspace.iterator,
                route_workspace.iterator + Int32(route_metadata_base),
                sm_scale,
                stream,
                static_config,
                static_seq_len_kv,
                **sage_kwargs,
            )
        else:
            null_i32 = cute.make_ptr(Int32, 0, mem_space=cutlass.AddressSpace.gmem)
            null_f32 = cute.make_ptr(Float32, 0, mem_space=cutlass.AddressSpace.gmem)
            # Only packed NVFP4 K/V read scale factors.
            null_u8 = cute.make_ptr(
                cutlass.Uint8, 0, mem_space=cutlass.AddressSpace.gmem
            )
            fmha_decode_launch(
                problem_shape,
                q.iterator,
                k.iterator,
                v.iterator,
                null_u8,  # k_sf
                null_u8,  # v_sf
                out.iterator,
                null_i32,  # seqlens_kv
                null_i32,  # cu_seqlens_q
                Int32(0),  # total_q_tokens
                null_i32,  # page_idx_kv
                null_i32,  # q_token_kv_block_sparse_page_memberships
                out.iterator,  # partial_o
                null_f32,  # partial_stats
                null_i32,  # split_kv_counter
                null_f32,  # attention_sinks
                sm_scale,
                Float32(1.0),  # output_scale
                Int32(
                    static_seq_len_kv * static_num_kv_heads * static_head_dim
                ),  # kv_b_stride
                Int32(0),  # max_active_clusters
                stream,
                static_config,
                static_seq_len_kv,
                # Contiguous K/V has no page table; the kernel parameters
                # behind these launcher arguments are typed, so pass typed
                # zeros rather than relying on the launcher's literal defaults.
                page_table_stride=Int64(0),
                page_table_capacity=Int32(0),
                q_token_kv_block_sparse_page_membership_stride=Int32(0),
                num_physical_kv_pages=Int64(0),
                k_page_stride=Int64(0),
                k_head_stride=Int64(0),
                k_token_stride=Int64(0),
                v_page_stride=Int64(0),
                v_head_stride=Int64(0),
                v_token_stride=Int64(0),
                **sage_kwargs,
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
        dtype: object, shape: tuple[object, ...], alignment: int = 16
    ) -> object:
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=alignment,
        )

    q_shape = (key.batch_size, key.seq_len_q, key.num_qo_heads, key.head_dim)
    num_q_blocks = ceil_div(key.seq_len_q, key.q_block_size)
    num_kv_blocks = ceil_div(key.seq_len_kv, key.kv_block_size)
    q_fake = fake_compact(config.q_dtype, q_shape)
    out_fake = fake_compact(config.out_dtype, q_shape)
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
        k_fake = fake_compact(config.k_dtype, kv_shape)
        v_fake = fake_compact(config.v_dtype, kv_shape)
        k_summary_fake = None
        v_summary_fake = None
        indptr_fake = None
        indices_fake = None
        exact_bits_fake = None
        valid_bits_fake = None
        row_route_offsets_fake = None
        route_workspace_fake = None
        if use_proxy_routes:
            summary_shape = (
                key.batch_size,
                num_kv_blocks,
                key.num_kv_heads,
                key.head_dim,
            )
            k_summary_fake = fake_compact(config.k_dtype, summary_shape)
            v_summary_fake = fake_compact(config.v_dtype, summary_shape)
        if key.use_block_sparse:
            if sparse_format == "bsr":
                indptr_fake = fake_compact(
                    Int32,
                    (key.batch_size, pattern_heads, num_q_blocks + 1),
                    4,
                )
                indices_fake = fake_compact(Int32, (cute.sym_int(),), 4)
            else:
                exact_bits_fake = fake_compact(
                    cutlass.Uint32,
                    (
                        key.batch_size,
                        pattern_heads,
                        num_q_blocks,
                        ceil_div(num_kv_blocks, 32),
                    ),
                    4,
                )
            valid_bits_fake = fake_compact(
                cutlass.Uint32,
                (key.batch_size, ceil_div(key.seq_len_kv, 32)),
                4,
            )
            row_route_offsets_fake = fake_compact(
                Int32,
                (key.batch_size * pattern_heads * num_q_blocks + 1,),
                4,
            )
            route_workspace_fake = fake_compact(Int32, (cute.sym_int(),), 4)
        sage_fakes: tuple[object | None, ...] = sage_adapter_slots({})
        if key.sage is not None:
            # The plan's scale shapes fill the adapter slots through
            # ``sage_adapter_slots``, the helper that binds each run's
            # tensors: the summary K scales only for a proxy plan, and no V
            # mean for a recipe without one.
            shapes = sage_scale_shapes(
                key.sage,
                batch_size=key.batch_size,
                seq_len_q=key.seq_len_q,
                seq_len_kv=key.seq_len_kv,
                num_qo_heads=key.num_qo_heads,
                num_kv_heads=key.num_kv_heads,
                head_dim=key.head_dim,
                summary_seq_len=num_kv_blocks if use_proxy_routes else None,
            )
            sage_fakes = tuple(
                None if shape is None else fake_compact(Float32, shape)
                for shape in sage_adapter_slots(shapes)
            )
        tensor_adapter = contiguous_adapter
        dynamic_args = (
            q_fake,
            k_fake,
            v_fake,
            k_summary_fake,
            v_summary_fake,
            out_fake,
            indptr_fake,
            indices_fake,
            exact_bits_fake,
            valid_bits_fake,
            row_route_offsets_fake,
            route_workspace_fake,
            Int32(0),
            *sage_fakes,
            Float32(1.0),
        )
    else:
        page_size = key.page_size
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
            config.k_dtype,
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
            config.v_dtype,
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
        indptr_fake = fake_compact(
            Int32,
            (key.batch_size, pattern_heads, num_q_blocks + 1),
            4,
        )
        indices_fake = fake_compact(Int32, (cute.sym_int(),), 4)
        valid_bits_fake = fake_compact(
            cutlass.Uint32,
            (key.batch_size, ceil_div(key.seq_len_kv, 32)),
            4,
        )
        row_route_offsets_fake = fake_compact(
            Int32,
            (key.batch_size * pattern_heads * num_q_blocks + 1,),
            4,
        )
        route_workspace_fake = fake_compact(Int32, (cute.sym_int(),), 4)
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
    """Compile and cache the adapter of one contiguous or paged plan.

    A contiguous plan is dense or block-sparse; a paged plan is block-sparse.
    """

    return _compile_block_sparse(key)


__all__ = [
    "_compile_block_sparse",
    "_get_compiled_block_sparse",
]
