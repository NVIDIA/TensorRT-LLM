# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash pool-compressed sparse MLA in the TRTLLM backend family.

The model projects queries and selects pools; this backend owns paged latent
and indexer state, pool-key updates, and sparse attention kernels. Selection arrives
as global latent-cache row IDs in GlmKpoolBackendForwardArgs.topk_rows.

Queries have no rotary component and are absorbed into the 512-wide latent
space. Small TP generation batches use native TRTLLM-GEN query heads; other
shapes use FlashMLA. FP8 cache rows are gathered and dequantized
in bounded query chunks. Prepared GLM page tables and live TRTLLM lengths
provide the same cache contract for prefill, decode and verification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._utils import get_sm_version

from ...interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from ...trtllm import TrtllmAttention, TrtllmAttentionMetadata
from .kernels import gather_fp8_kv_rows, kpool_expand, kpool_score, kpool_update
from .native_decode import GlmKpoolNativeDecode
from .params import INDEX_SENTINEL, GlmKpoolSparseParams


def _flash_mla_sparse_fwd() -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Resolve the FlashMLA sparse kernel lazily."""
    try:
        from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
    except ImportError as exc:  # pragma: no cover - wheel always bundles it
        raise RuntimeError(
            "glm_kpool sparse MLA requires tensorrt_llm.flash_mla."
            "flash_mla_sparse_fwd, which this build does not provide"
        ) from exc
    return flash_mla_sparse_fwd


def paged_slot_indices(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (page, within_page) indices for positions in a paged pool.

    block_table is [..., max_pages]; positions matches its leading dimensions.
    Use this pair to index [slots, tokens_per_block, ...] pools: V2 coalesced page
    strides may include other buffers, so flattening would copy or misaddress
    storage. Callers must supply valid positions.
    """
    page = torch.div(positions, tokens_per_block, rounding_mode="floor")
    offset = positions - page * tokens_per_block
    return torch.gather(block_table, -1, page), offset


def latent_pool_rows(latent_pool: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """View coalesced latent storage as uniformly strided [N, 1, dim] rows.

    Input is [slots, tokens_per_block, dim]. Returns (rows, base_row, rows_per_slot),
    with row_id = base_row + slot * rows_per_slot + within_page. Only valid latent
    row IDs or -1 sentinels may reach FlashMLA; storage also contains other buffers.
    No allocation or device work is needed, so the view is CUDA-graph safe.
    """
    dim = latent_pool.shape[-1]
    if latent_pool.stride(2) != 1 or latent_pool.stride(1) != dim:
        raise ValueError(
            "glm_kpool latent pool rows must be contiguous within a page; got "
            f"strides {tuple(latent_pool.stride())} for dim {dim}"
        )
    slot_stride = latent_pool.stride(0)
    offset = latent_pool.storage_offset()
    if slot_stride % dim or offset % dim:
        raise ValueError(
            f"glm_kpool latent pool slot stride {slot_stride} / storage offset "
            f"{offset} are not multiples of dim {dim}; the pool has no uniform "
            "row view and the coalesced layout assumption broke"
        )
    total_rows = latent_pool.untyped_storage().nbytes() // (dim * latent_pool.element_size())
    rows = torch.as_strided(latent_pool, (total_rows, 1, dim), (dim, dim, 1), storage_offset=0)
    return rows, offset // dim, slot_stride // dim


@dataclass(frozen=True)
class _GlmKpoolCacheState:
    """Layer-local pool views and prepared tables/lengths in context-first order.

    Pool views are [slots, tokens_per_block, dim]. Device tables and lengths retain
    their prepared addresses for CUDA graph replay.
    """

    latent_pool: torch.Tensor
    index_pool: torch.Tensor
    block_tables: torch.Tensor
    kv_lens: torch.Tensor
    tokens_per_block: int
    num_contexts: int


class GlmKpoolSparseAttention(TrtllmAttention):
    """NoPE k-pool MLA backend selected by SparseParams(algorithm="glm_kpool")."""

    Metadata = TrtllmAttentionMetadata

    #: FlashMLA tiles the top-k axis in blocks of 64 (``B_TOPK``); index rows
    #: are padded to that multiple with ``-1`` (invalid) entries, which is
    #: semantics-free by the kernel's own contract.
    _KERNEL_TOPK_ALIGN = 64

    # FlashMLA instantiates 64/128 query heads. Pad TP-local heads to the next
    # size and discard the extra outputs; heads attend independently.
    _KERNEL_HEAD_COUNTS = (64, 128)

    # Bound selected-KV staging independently of the configured context length.
    # At the published 2112-entry selection this uses 132 MiB of BF16 rows.
    _FP8_QUERY_CHUNK_SIZE = 64

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        quant_config=None,
        q_scaling: float | None = None,
        pos_embd_params: PositionalEmbeddingParams | None = None,
        mla_params: MLAParams | None = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: int | None = None,
        sparse_params: GlmKpoolSparseParams | None = None,
        dtype: torch.dtype | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **kwargs,
    ) -> None:
        # dtype/aux_stream arrive from the standard create_attention kwargs;
        # this branch has no dtype-dependent weights and no side stream.
        del dtype, aux_stream
        # The engine-level construction path passes the llmapi config object;
        # keep it (as DSA does) without forwarding it into the base class.
        self.sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if sparse_params is None:
            raise ValueError("sparse_params is required for GlmKpoolSparseAttention")
        if not isinstance(sparse_params, GlmKpoolSparseParams):
            raise TypeError(
                f"GlmKpoolSparseAttention needs GlmKpoolSparseParams, got {type(sparse_params)}"
            )
        if head_dim != sparse_params.kv_lora_rank:
            raise ValueError(
                "glm_kpool consumes absorbed latent-space queries: head_dim "
                f"({head_dim}) must equal kv_lora_rank ({sparse_params.kv_lora_rank})"
            )
        if pos_embd_params is not None:
            raise ValueError(
                "glm_kpool is fully NoPE; positional embedding parameters have no "
                "meaning on this branch"
            )
        if mla_params is None:
            # The standard create_attention MLA path asserts qk_rope_head_dim>0
            # (the rope'd DeepSeek geometry), so this fully-NoPE branch states
            # its MLA identity itself instead of loosening the shared assert.
            mla_params = MLAParams(
                q_lora_rank=sparse_params.q_lora_rank,
                kv_lora_rank=sparse_params.kv_lora_rank,
                qk_rope_head_dim=0,
                qk_nope_head_dim=sparse_params.qk_nope_head_dim,
                v_head_dim=sparse_params.v_head_dim,
                rope_append=False,
            )
        if mla_params.qk_rope_head_dim != 0:
            raise ValueError(
                f"glm_kpool is fully NoPE; got qk_rope_head_dim={mla_params.qk_rope_head_dim}"
            )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=None,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            sparse_params=sparse_params,
            **kwargs,
        )
        if self.num_kv_heads != 1:
            raise ValueError(
                f"glm_kpool latent cache is MQA-style (one KV head), got {self.num_kv_heads}"
            )
        #: Softmax scale of the *unabsorbed* q . k product over
        #: ``qk_nope_head_dim``; absorption reassociates the matmuls but the
        #: score scale is unchanged.
        self.softmax_scale = float(sparse_params.qk_nope_head_dim) ** -0.5
        self._native_decode = GlmKpoolNativeDecode()

    @classmethod
    def support_fused_rope(cls) -> bool:
        # Fully NoPE: there is no rotary embedding anywhere on this path.
        return False

    @classmethod
    def support_fused_qkv(cls) -> bool:
        # The model layer owns the low-rank q/kv projections and absorption.
        return False

    # -- metadata-derived cache state ----------------------------------------

    def _cache_state(self, metadata) -> _GlmKpoolCacheState:
        """Read slot-major pool views, prepared GLM tables and live TRTLLM lengths.

        kv_lens_cuda includes overlap corrections and MTP rewinds. Constructing this
        state uses views only, with no allocation, H2D copy or host synchronization.
        """
        if metadata is None:
            raise ValueError(
                "GlmKpoolSparseAttention requires the engine's prepared attention "
                "metadata (TrtllmAttentionMetadata); got None. The backend derives "
                "its cache pools, block tables, and visible lengths from it."
            )
        manager = getattr(metadata, "kv_cache_manager", None)
        if manager is None:
            raise ValueError(
                "glm_kpool metadata has no kv_cache_manager; the hybrid "
                "KVCacheManagerV2 owns the latent/indexer pools"
            )
        mamba_metadata = getattr(metadata, "mamba_metadata", None)
        if mamba_metadata is None or mamba_metadata is False:
            raise ValueError(
                "glm_kpool requires prepared metadata: call metadata.prepare() "
                "with the Glm5NextCacheManager attached (mamba_metadata is missing)"
            )
        latent = manager.get_latent_state_buffer(self.layer_idx)
        index = manager.get_index_state_buffer(self.layer_idx)
        if latent is None or index is None:
            raise ValueError(
                f"glm_kpool layer {self.layer_idx} has no latent/indexer pool on "
                "this manager; the layer schedule and the cache layout disagree"
            )
        latent = latent[:, :, 0, :]
        index = index[:, :, 0, :]
        tokens_per_block = int(manager.tokens_per_block)
        batch = int(metadata.seq_lens.shape[0])
        num_contexts = int(metadata.num_contexts)

        tables = getattr(mamba_metadata, "glm_block_tables", None)
        if tables is None:
            raise RuntimeError(
                "glm_kpool requires prepared glm_block_tables; call metadata.prepare() "
                "with Glm5NextMamba2Metadata before eager execution or CUDA graph capture"
            )
        kv_lens = getattr(metadata, "kv_lens_cuda", None)
        if kv_lens is None:
            raise ValueError("glm_kpool requires prepared metadata.kv_lens_cuda")
        return _GlmKpoolCacheState(
            latent_pool=latent,
            index_pool=index,
            block_tables=tables[:batch],
            kv_lens=kv_lens[:batch],
            tokens_per_block=tokens_per_block,
            num_contexts=num_contexts,
        )

    # -- paged cache path -----------------------------------------------------

    def append_paged_state(
        self,
        latent: torch.Tensor,
        packed: torch.Tensor,
        positions: torch.Tensor,
        metadata,
        *,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        """Write latent rows and packed [k | gate] state into the paged cache.

        For packed context, positions and request_ids are [tokens], identifying each
        row's request. Otherwise positions is [generation_requests, tokens_per_request]
        and uses the generation slice of the prepared block tables.
        """
        state = self._cache_state(metadata)
        if request_ids is not None:
            # Packed context rows of several requests: row i's page comes from
            # block table request_ids[i] (no [tokens, max_pages] gather).
            page_idx = torch.div(positions, state.tokens_per_block, rounding_mode="floor")
            offset = positions - page_idx * state.tokens_per_block
            page = state.block_tables[request_ids.long(), page_idx]
        else:
            table = state.block_tables[state.num_contexts :]
            page, offset = paged_slot_indices(table, positions, state.tokens_per_block)
        if state.latent_pool.dtype == torch.float8_e4m3fn:
            quantized = (latent.float() * self.kv_scale_orig_quant).clamp(-448, 448)
            # PyTorch advanced indexing does not implement FP8 payloads. Index
            # their byte views, preserving the manager's coalesced page strides.
            state.latent_pool.view(torch.uint8)[page, offset] = quantized.to(
                torch.float8_e4m3fn
            ).view(torch.uint8)
        else:
            state.latent_pool[page, offset] = latent.to(state.latent_pool.dtype)
        # Only the [k | gate] columns; the pool-key slice is maintained by
        # update_pool_keys.
        packed_dim = self.sparse_params.packed_state_dim
        state.index_pool[page, offset, :packed_dim] = packed.to(state.index_pool.dtype)

    def _rows(
        self,
        state: _GlmKpoolCacheState,
        rows_per_request: int = 1,
        request_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select generation tables, or all tables for packed context request IDs.

        Verification repeats generation tables per token; callers supply each
        query's visible length for verification and packed context rows.
        """
        if request_ids is not None:
            return state.block_tables, None
        gen = slice(state.num_contexts, None)
        if rows_per_request > 1:
            return state.block_tables[gen].repeat_interleave(rows_per_request, dim=0), None
        return state.block_tables[gen], state.kv_lens[gen]

    def update_pool_keys(
        self,
        positions: torch.Tensor,
        ape: torch.Tensor,
        metadata,
        *,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        """Refresh the pool containing each position using compression APE.

        positions is [rows]; ape is [kpool, head_dim]. Generation has one position per
        request. For packed context, request_ids maps each row to a request and only
        pool-final positions are supplied, avoiding concurrent writes to one pool.
        """
        state = self._cache_state(metadata)
        tables, _ = self._rows(state, request_ids=request_ids)
        kpool_update(
            state.index_pool,
            tables,
            positions,
            ape,
            state.tokens_per_block,
            head_dim=self.sparse_params.index_head_dim,
            kpool=self.sparse_params.index_kpool,
            request_ids=request_ids,
        )

    def score_pools(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        metadata,
        *,
        q_scale: float,
        w_scale: float,
        kv_lens: torch.Tensor | None = None,
        rows_per_request: int = 1,
        request_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score complete pools into FP32 [rows, pool_capacity].

        q is [rows, heads, head_dim], weights is [rows, heads]. Packed context uses
        request_ids; verification repeats generation tables rows_per_request times.
        Both supply per-query kv_lens. Single-token decode uses live metadata lengths.
        Incomplete/invisible pools receive the FP32 minimum before top-k.
        """
        state = self._cache_state(metadata)
        tables, gen_lens = self._rows(state, rows_per_request, request_ids=request_ids)
        capacity = state.block_tables.shape[1] * state.tokens_per_block
        kpool = self.sparse_params.index_kpool
        return kpool_score(
            q,
            weights,
            state.index_pool,
            tables,
            gen_lens if kv_lens is None else kv_lens,
            state.tokens_per_block,
            num_pools_max=(capacity + kpool - 1) // kpool,
            head_dim=self.sparse_params.index_head_dim,
            kpool=self.sparse_params.index_kpool,
            q_scale=q_scale,
            w_scale=w_scale,
            # bf16 inputs are exact in tf32, so the tensor-core dot is an fp32
            # accumulation of exact products (measured: identical selections).
            precision="tf32",
            # Context rows: the query tokens of a request share its block
            # table, so a program gathers each pool-key block once for 16 rows.
            rows_per_program=1 if request_ids is None else 16,
            request_ids=request_ids,
        )

    def expand_selection(
        self,
        selected: torch.Tensor,
        metadata,
        *,
        kv_lens: torch.Tensor | None = None,
        rows_per_request: int = 1,
        request_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Selected pools ``[N, select_k]`` -> latent row ids for the kernel.

        Expands each valid pool into its members, appends the always-visible
        tail, pads with ``-1`` to :attr:`GlmKpoolSparseParams.kernel_output_width`
        and translates positions to latent-cache row ids in one kernel. Same
        row addressing as :meth:`score_pools`.
        """
        state = self._cache_state(metadata)
        tables, gen_lens = self._rows(state, rows_per_request, request_ids=request_ids)
        _, base_row, rows_per_slot = latent_pool_rows(state.latent_pool)
        return kpool_expand(
            selected,
            gen_lens if kv_lens is None else kv_lens,
            tables,
            state.tokens_per_block,
            base_row=base_row,
            rows_per_slot=rows_per_slot,
            kpool=self.sparse_params.index_kpool,
            out_width=self.sparse_params.kernel_output_width,
            request_ids=request_ids,
        )

    # -- sparse core ----------------------------------------------------------

    def create_output(
        self,
        q: torch.Tensor,
        *,
        is_quantize_output: bool,
        metadata=None,
        attention_mask=None,
        is_gen_only: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Allocate [tokens, num_heads * kv_lora_rank] for absorbed attention.

        Both phases emit latent-width outputs; the model applies the V projection.
        Quantized outputs are unsupported.
        """
        del metadata, attention_mask, is_gen_only, kwargs
        if is_quantize_output:
            raise ValueError(
                "glm_kpool produces a bf16 latent-space output; quantized "
                "attention output (out_scale/output_sf) is not supported"
            )
        return [q.new_empty((q.shape[0], self.num_heads * self.kv_lora_rank), dtype=q.dtype)]

    def _dispatch_sparse_core(
        self,
        q_latent: torch.Tensor,
        kv_rows: torch.Tensor,
        topk_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Pad the index rows to the kernel's tiles and run the sparse kernel.

        ``q_latent`` is ``[T, H, kv_lora]``, ``kv_rows`` is ``[N, 1,
        kv_lora]``, ``topk_rows`` is int32 ``[T, width]`` with
        :data:`INDEX_SENTINEL` invalid slots. Returns the kernel's latent-space
        output ``[T, H, kv_lora]``; :meth:`_finalize_output` flattens it to the
        base-contract shape at the ``forward`` boundary.
        """
        if kv_rows.dtype == torch.float8_e4m3fn:
            # Like DeepSeek-V4's BF16 context adapter, dequantize only selected
            # rows and reuse FlashMLA. Chunk query rows so prefill never stages
            # the full cache or a [max_num_tokens, topk, latent_dim] allocation.
            output = torch.empty_like(q_latent)
            for start in range(0, q_latent.shape[0], self._FP8_QUERY_CHUNK_SIZE):
                stop = min(start + self._FP8_QUERY_CHUNK_SIZE, q_latent.shape[0])
                indices = topk_rows[start:stop]
                selected, local_indices = gather_fp8_kv_rows(
                    kv_rows, indices, self.kv_scale_quant_orig
                )
                output[start:stop] = self._dispatch_sparse_core(
                    q_latent[start:stop], selected, local_indices
                )
            return output
        pad = (-topk_rows.shape[-1]) % self._KERNEL_TOPK_ALIGN
        if pad:
            topk_rows = torch.nn.functional.pad(topk_rows, (0, pad), value=INDEX_SENTINEL)
        local_heads = q_latent.shape[1]
        kernel_heads = next((h for h in self._KERNEL_HEAD_COUNTS if h >= local_heads), None)
        if kernel_heads is None:
            raise ValueError(
                f"glm_kpool: {local_heads} query heads exceed every FlashMLA "
                f"sparse-kernel instantiation {self._KERNEL_HEAD_COUNTS}"
            )
        if kernel_heads != local_heads:
            # Zero-filled query lanes; per-head attention keeps them inert and
            # the slice below discards their outputs (DSA's TP-head padding).
            q_padded = q_latent.new_zeros((q_latent.shape[0], kernel_heads, q_latent.shape[2]))
            q_padded[:, :local_heads, :] = q_latent
            q_latent = q_padded
        out, _, _ = _flash_mla_sparse_fwd()(
            q_latent,
            kv_rows,
            topk_rows.unsqueeze(1),
            self.softmax_scale,
            self.kv_lora_rank,
        )
        if kernel_heads != local_heads:
            # Strided view over the real heads; the model's absorbed V bmm
            # consumes strided batches, so no copy of [T, H, kv_lora] here.
            out = out[:, :local_heads, :]
        return out

    def _finalize_output(
        self, out_latent: torch.Tensor, output: torch.Tensor | None
    ) -> torch.Tensor:
        """Flatten to [tokens, num_heads * kv_lora_rank], filling output when supplied.

        Without a caller-owned buffer, flattening the contiguous head/feature axes
        preserves the view even when token strides include padded query heads.
        """
        if output is None:
            # The real heads remain contiguous within each token even when
            # the token stride includes padded heads. Flattening these two
            # axes keeps a view and preserves the backend's 2-D contract.
            return out_latent.flatten(1)
        output.copy_(out_latent.reshape(out_latent.shape[0], -1))
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run sparse attention against paged latent rows.

        Args:
            q: Absorbed queries [tokens, num_heads, kv_lora_rank], or the flattened
                [tokens, num_heads * kv_lora_rank] view.
            k: Must be None; KV rows come from the prepared cache metadata.
            v: Must be None; latent rows serve as both K and V.
            metadata: Prepared TRTLLM metadata with Glm5NextMamba2Metadata tables.
            forward_args: Must provide topk_rows (int32 [tokens, width], -1 invalid)
                and a context_only or generation_only phase. An optional flat output
                buffer is validated and filled in place. Quantized outputs are unsupported.
            **kwargs: Legacy argument carrier, merged by the shared base contract.

        Returns:
            Latent output [tokens, num_heads * kv_lora_rank].
        """
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        sparse_args = forward_args.sparse_backend_args
        topk_rows = getattr(sparse_args, "topk_rows", None)
        if topk_rows is None:
            raise ValueError(
                "glm_kpool requires pool-expanded selection in sparse_backend_args.topk_rows"
            )
        if sparse_args.topk_indices is not None:
            raise ValueError("glm_kpool accepts topk_rows, not request-local topk_indices")
        if k is not None:
            raise ValueError("glm_kpool reads paged latent rows from metadata; k must be None")
        if v is not None:
            raise ValueError("glm_kpool consumes latent rows as both K and V; v must be None")
        if q.dim() == 2:
            q = q.view(q.shape[0], self.num_heads, self.head_dim)

        if (
            forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
            or forward_args.output_sf is not None
        ):
            raise ValueError(
                "glm_kpool does not support quantized attention output "
                "(out_scale/out_scale_sf/output_sf); it returns the bf16 "
                "latent-space output the model layer projects"
            )
        output = forward_args.output
        if output is not None:
            expected_shape = (q.shape[0], self.num_heads * self.kv_lora_rank)
            if (
                tuple(output.shape) != expected_shape
                or output.dtype != q.dtype
                or output.device != q.device
            ):
                raise ValueError(
                    f"glm_kpool forward_args.output must be a {expected_shape} "
                    f"tensor of dtype {q.dtype} on {q.device}; got shape "
                    f"{tuple(output.shape)}, dtype {output.dtype}, device "
                    f"{output.device}"
                )

        input_type = forward_args.attention_input_type
        if input_type not in (
            AttentionInputType.context_only,
            AttentionInputType.generation_only,
        ):
            raise ValueError(f"glm_kpool forward is phase-explicit, got {input_type!r}")
        state = self._cache_state(metadata)
        kv_rows, _, _ = latent_pool_rows(state.latent_pool)
        native_supported = (
            q.is_cuda
            and q.dtype == kv_rows.dtype == torch.bfloat16
            and q.shape[1:] == (16, 512)
            and kv_rows.shape[0] % 32 == 0
            and get_sm_version() in (100, 103)
        )
        if native_supported:
            # Reserve during prefill/profiling too, before sizing the KV pool.
            # Scratch is shared through metadata; each backend owns only counters.
            self._native_decode.prepare_workspace(q, metadata)
        if (
            native_supported
            and input_type == AttentionInputType.generation_only
            and 1 <= q.shape[0] <= 8
        ):
            out = self._native_decode(q, kv_rows, topk_rows, self.softmax_scale, metadata)
        else:
            out = self._dispatch_sparse_core(q, kv_rows, topk_rows)
        return self._finalize_output(out, output)
