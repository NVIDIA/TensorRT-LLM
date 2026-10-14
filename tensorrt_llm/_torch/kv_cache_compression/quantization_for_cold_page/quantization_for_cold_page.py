# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

from typing import TYPE_CHECKING, Any, Callable, Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CachePageInfo, _KVCache


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Common codec registration and callbacks for cold-page quantizers."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True
    config: "ColdPageQuantizationCompressionConfig"

    @property
    def uses_gpu_page_protection(self) -> bool:
        """Enable the optional policy only when a protected token count is set."""
        return any(self.gpu_resident_token_bounds)

    @property
    def gpu_resident_token_bounds(self) -> tuple[int, int]:
        """Optional first/last confirmed-token counts; both default to zero."""
        return self.config.first_n, self.config.last_n

    def validate_gpu_resident_token_bounds(self, tokens_per_block: int) -> None:
        """Validate whole-page protection against this manager's block size."""
        for name, value in zip(("first_n", "last_n"), self.gpu_resident_token_bounds):
            if value < 0 or value % tokens_per_block:
                raise ValueError(
                    f"{name} must be a nonnegative multiple of the KV block size "
                    f"({tokens_per_block}), got {value}"
                )

    def select_gpu_resident_pages(
        self,
        page_info: Sequence["CachePageInfo"],
        *,
        confirmed_length: int,
        tokens_per_block: int,
    ) -> list[tuple[int, int, int]]:
        """Select whole dense pages intersecting the head, tail or pending writes.

        Confirmed tokens include prefill and accepted generation writes.
        """
        first_n, last_n = self.gpu_resident_token_bounds
        if not (first_n or last_n):
            return []
        tail_start = max(0, confirmed_length - last_n)
        selected = []
        for page in page_info:
            if page.is_sparse or page.window_size is not None:
                continue
            start = page.block_ordinal * tokens_per_block
            end = start + tokens_per_block
            if start < first_n or start >= confirmed_length or (last_n and end > tail_start):
                selected.append((page.beam_index, page.block_ordinal, page.layer_group_id))
        return selected

    def update_gpu_page_protection(
        self,
        kv_cache: "_KVCache",
        *,
        confirmed_length: int | None = None,
    ) -> None:
        """Apply the cold-page protection policy before migration or commit."""
        if not self.uses_gpu_page_protection:
            return
        page_info = kv_cache.get_page_info()
        pages = self.select_gpu_resident_pages(
            page_info,
            confirmed_length=(
                kv_cache.history_length if confirmed_length is None else confirmed_length
            ),
            tokens_per_block=kv_cache.tokens_per_block,
        )
        selected = set(pages)
        exited = [page for page in kv_cache.get_private_pages() if page not in selected]
        if exited:
            restored = set(kv_cache.restore_shared_pages(exited))
            shared = {
                (page.beam_index, page.block_ordinal, page.layer_group_id)
                for page in page_info
                if page.has_shared_page
            }
            # Keep valid private pages until their shared replacements are ready.
            pages.extend(page for page in exited if page in shared and page not in restored)
        kv_cache.set_private_pages(pages)
        kv_cache.set_gpu_resident_pages(pages)

    def create_precision_reuse_checker(
        self,
    ) -> Callable[["_KVCache"], int | None] | None:
        """Check each claim against the same protected endpoint during admission."""
        if not self.uses_gpu_page_protection:
            return None
        protected_length: int | None = None

        def check_reuse(kv_cache: "_KVCache") -> int | None:
            nonlocal protected_length
            matched = kv_cache.num_committed_tokens
            if protected_length is None:
                protected_length = matched
            limit = self.get_precision_reuse_limit(
                kv_cache.get_page_info(),
                reused_length=matched,
                protected_length=protected_length,
                tokens_per_block=kv_cache.tokens_per_block,
            )
            return limit if limit < matched else None

        return check_reuse

    def get_precision_reuse_limit(
        self,
        page_info: Sequence["CachePageInfo"],
        *,
        reused_length: int,
        protected_length: int,
        tokens_per_block: int,
    ) -> int:
        """Stop reuse before a protected page with lossy or unknown history.

        Keep the original match endpoint fixed when retrying a shorter prefix.
        """
        protected = set(
            self.select_gpu_resident_pages(
                page_info,
                confirmed_length=protected_length,
                tokens_per_block=tokens_per_block,
            )
        )
        limit = reused_length
        for page in page_info:
            start = page.block_ordinal * tokens_per_block
            coordinates = (page.beam_index, page.block_ordinal, page.layer_group_id)
            if start < limit and coordinates in protected and page.lossy_encode_count != 0:
                limit = start
        return limit

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Create one native codec with state isolated to this KVCM."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state = self.build_codec_state(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        return native.create_python_cold_page_codec(self, codec_state)

    def configure(self, codec_state: Any, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers and publish each lifecycle's cold-page size."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state.lifecycle_metadata = tuple(
            self.build_lifecycle_metadata(codec_state, lifecycle) for lifecycle in lifecycles
        )
        properties = []
        for metadata in codec_state.lifecycle_metadata:
            lifecycle = native.ColdPageLifecycleProperties()
            lifecycle.cold_page_bytes = metadata.cold_page_bytes
            lifecycle.page_index_location = native.ColdPageIndexLocation.HOST
            lifecycle.is_lossless = self.is_lifecycle_lossless(codec_state, metadata)
            properties.append(lifecycle)
        return properties

    def is_lifecycle_lossless(self, codec_state: object, metadata: object) -> bool:
        """Whether the format preserves every value in this lifecycle exactly."""
        return False

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Build the format-specific state owned by one native codec."""
        raise NotImplementedError

    def build_lifecycle_metadata(self, codec_state: object, lifecycle: object) -> object:
        """Resolve one KVCM lifecycle into format-specific launch metadata."""
        raise NotImplementedError
