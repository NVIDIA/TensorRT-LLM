# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-attention state of an encoder-decoder model's attention metadata."""

from typing import TYPE_CHECKING, Any, List, Optional

import torch

from ....memory_buffer_utils import Buffers
from ....metadata import KVCacheParams
from ..extensions import MetadataExtension

if TYPE_CHECKING:
    from ..interface import AttentionMetadata, BatchInputs


class CrossAttentionMetadata(MetadataExtension):
    """The cross-attention sub-metadata of an encoder-decoder model.

    On a decoder's self-attention metadata, ``cross`` is the sub-metadata its
    cross-attention layers use: it shares the decoder's query side and takes
    the encoder's lengths as its key/value side. On that sub-metadata, which
    is cross attention itself (``is_cross``), ``cross`` refers to itself.
    """

    fields = frozenset({"cross", "has_sub_metadata"})
    methods = frozenset(
        {
            "update_cross_metadata",
            "refresh_cross_query_side",
            "use_cross_sub_metadata",
            "drop_cross_sub_metadata",
        }
    )
    init_fields = {
        "cross": None,
    }

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        md = self.metadata
        if md.is_cross:
            assert self.cross is None or self.cross is md, (
                "Cross attention metadata should not have sub metadata"
            )
            self.cross = md
        else:
            assert self.cross is None or (self.cross.core_class() is md.core_class()), (
                "Top level and cross attention sub metadata type mismatched"
            )

    @property
    def has_sub_metadata(self) -> bool:
        return self.cross is not None and self.cross is not self.metadata

    def creation_value(self, name: str) -> Any:
        # The sub-metadata refers to itself; a copy of it refers to the copy.
        if name == "cross" and self.cross is self.metadata:
            return None
        return super().creation_value(name)

    def use_cross_sub_metadata(self, cross_md: "AttentionMetadata") -> None:
        """Keep ``cross_md`` as this decoder's sub-metadata (CUDA graph steps reuse it)."""
        self.cross = cross_md

    def drop_cross_sub_metadata(self) -> None:
        """Detach the sub-metadata from a derived view (a draft view) of the decoder."""
        if self.has_sub_metadata:
            self.cross = None

    def copy_sub_metadata_for_cuda_graph(
        self, max_batch_size: int, max_draft_tokens: int, buffers, encode_only: bool
    ) -> None:
        """Point a CUDA graph copy at a CUDA graph copy of the sub-metadata."""
        if self.has_sub_metadata:
            self.cross = self.cross.create_cuda_graph_metadata(
                max_batch_size, True, max_draft_tokens, buffers, encode_only
            )

    def create_sub_metadata(
        self,
        encoder_seq_lens: List[int],
        cross_kv_cache_manager=None,
        encoder_num_cached_tokens_per_seq: Optional[List[int]] = None,
    ) -> "AttentionMetadata":
        """Build a sub-metadata instance for cross-attention.

        The returned metadata shares Q-side fields (``seq_lens``,
        ``request_ids``, ``num_contexts``) with ``self`` (the decoder
        self-attention metadata) and overrides the K/V-side with the encoder
        lengths so that ``returned.is_cross is True``.

        This is intended to be called by the runtime / unit tests once the
        encoder lengths and cross-pool KV cache manager are known. The
        returned object is a *new* metadata instance (not stored on
        the decoder's ``cross``); callers can attach it there if desired.

        Args:
            encoder_seq_lens: Per-request encoder sequence lengths. On the
                first decoder context step this is the full encoder length;
                on generation steps it should be ``0`` (no new K/V tokens to
                add to the cross pool — the encoder K/V are already cached).
            cross_kv_cache_manager: KV cache manager for the cross pool.
                When ``None``, the returned metadata uses the stateless
                (no-KV-cache) path (suitable for unit tests).
            encoder_num_cached_tokens_per_seq: Per-request count of encoder
                K/V tokens already present in the cross pool. ``None``
                defaults to 0 (context phase, nothing cached yet).

        Returns:
            A new ``AttentionMetadata`` of the same family as the decoder's,
            with ``seq_lens_kv`` set to ``encoder_seq_lens`` so that
            ``is_cross`` becomes ``True``.
        """
        md = self.metadata
        # Cross-attention has K/V lengths from the encoder, while
        # self-attention has K/V lengths from the decoder. In a CUDA graph the
        # sub-metadata keeps its own buffers so preparing cross metadata cannot
        # overwrite self-attention sequence lengths.
        cross_md = md.constructed_copy(
            kv_cache_manager=cross_kv_cache_manager,
            seq_lens_kv=torch.tensor(encoder_seq_lens, dtype=torch.int),
            cross=None,
            cuda_graph_buffers=Buffers() if md.is_cuda_graph else md.cuda_graph_buffers,
        )
        self._update_sub_metadata(
            cross_md,
            encoder_seq_lens,
            cross_kv_cache_manager,
            encoder_num_cached_tokens_per_seq,
            base_kv_cache_params=md.kv_cache_params,
            block_ids_per_seq=None,
        )
        return cross_md

    def _update_sub_metadata(
        self,
        cross_md: "AttentionMetadata",
        encoder_seq_lens: List[int],
        cross_kv_cache_manager,
        encoder_num_cached_tokens_per_seq: Optional[List[int]],
        *,
        base_kv_cache_params: Optional[KVCacheParams],
        block_ids_per_seq: Optional[List[list]],
    ) -> "AttentionMetadata":
        kv_cache_params = cross_md.kv_cache_params
        if encoder_num_cached_tokens_per_seq is not None:
            kv_cache_params = KVCacheParams(
                use_cache=(
                    base_kv_cache_params.use_cache
                    if base_kv_cache_params is not None
                    else (cross_kv_cache_manager is not None)
                ),
                num_cached_tokens_per_seq=list(encoder_num_cached_tokens_per_seq),
                block_ids_per_seq=block_ids_per_seq,
                host_max_attention_window_sizes=(
                    base_kv_cache_params.host_max_attention_window_sizes
                    if base_kv_cache_params is not None
                    else None
                ),
                host_sink_token_length=(
                    base_kv_cache_params.host_sink_token_length
                    if base_kv_cache_params is not None
                    else None
                ),
                num_extra_kv_tokens=(
                    base_kv_cache_params.num_extra_kv_tokens
                    if base_kv_cache_params is not None
                    else 0
                ),
            )
        # The sub-metadata shares the decoder's query side, takes the encoder
        # lengths as its KV side and keeps the decoder-side prompt lengths for
        # the Q-side context metadata (the encoder side is in seq_lens_kv and
        # kv_cache_params.num_cached_tokens_per_seq).
        cross_md.core().kv_cache_manager = cross_kv_cache_manager
        cross_md.core().apply_batch_inputs(
            self._sub_metadata_inputs(
                cross_md,
                kv_cache_params=kv_cache_params,
                seq_lens_kv=torch.tensor(encoder_seq_lens, dtype=torch.int),
            )
        )
        return cross_md

    def _sub_metadata_inputs(self, cross_md: "AttentionMetadata", **changes) -> "BatchInputs":
        """The sub-metadata's batch: the decoder's query side, its own otherwise."""
        from ..interface import BatchInputs

        md = self.metadata
        inputs = dict(
            seq_lens=md.seq_lens,
            seq_lens_cuda=md.seq_lens_cuda,
            num_contexts=md.num_contexts,
            request_ids=md.request_ids,
            prompt_lens=md.prompt_lens,
            kv_cache_params=cross_md.kv_cache_params,
            beam_width=cross_md.beam_width,
            num_chunked_ctx_requests=cross_md.num_chunked_ctx_requests,
            q_offsets=cross_md.cu_q_seqlens,
            kv_offsets=cross_md.cu_kv_seqlens,
        )
        inputs.update(changes)
        return BatchInputs(**inputs)

    def refresh_cross_query_side(self) -> "AttentionMetadata":
        """Rebind the sub-metadata to the decoder's current query side.

        For a stable CUDA graph generation step whose encoder side did not
        change: binds the decoder's lengths, request IDs and prompt lengths
        without staging or re-deriving anything.
        """
        if not self.has_sub_metadata:
            raise RuntimeError("CUDA graph cross-attention metadata has not been initialized.")
        self.cross.core().apply_batch_inputs(self._sub_metadata_inputs(self.cross))
        return self.cross

    def prepare_sub_metadata(self, inputs: "BatchInputs") -> None:
        """Prepare the sub-metadata from its own step inputs (``BatchInputs.cross``)."""
        if not self.has_sub_metadata:
            raise RuntimeError("BatchInputs.cross given without a cross-attention sub-metadata.")
        self.cross.prepare(inputs)

    def update_cross_metadata(
        self,
        encoder_seq_lens: List[int],
        cross_kv_cache_manager,
        encoder_num_cached_tokens_per_seq: Optional[List[int]] = None,
    ) -> "AttentionMetadata":
        """Refresh an existing CUDA graph cross-attention sub-metadata."""
        if not self.has_sub_metadata:
            raise RuntimeError("CUDA graph cross-attention metadata has not been initialized.")

        cross_md = self.cross
        assert cross_md is not None
        base_kv_cache_params = cross_md.kv_cache_params
        block_ids_per_seq = (
            base_kv_cache_params.block_ids_per_seq if base_kv_cache_params is not None else None
        )
        return self._update_sub_metadata(
            cross_md,
            encoder_seq_lens,
            cross_kv_cache_manager,
            encoder_num_cached_tokens_per_seq,
            base_kv_cache_params=base_kv_cache_params,
            block_ids_per_seq=block_ids_per_seq,
        )
