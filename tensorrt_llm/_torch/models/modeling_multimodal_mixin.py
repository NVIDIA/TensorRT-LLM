# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Hashable, Iterable, Iterator, Optional, Sequence

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.logger import logger

from .modeling_multimodal_utils import (
    _cache_multimodal_embeddings,
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)

if TYPE_CHECKING:
    from ..pyexecutor.llm_request import LlmRequest


_MM_DATA_INPUT_MODALITY_KEYS = frozenset({"audio", "image", "video"})
_MM_AUX_STREAM: Optional[tuple[int, torch.cuda.Stream]] = None
_MM_ENCODER_CACHE_LOG_NAME = "mm_encoder_cache"


def _get_mm_aux_stream(max_prefetch_ahead: int = 0) -> Optional[torch.cuda.Stream]:
    """Return the side CUDA stream used for multimodal encoder prefetch.

    Returns `None` when side-stream prefetch is disabled, CUDA is unavailable,
    or the current stream is being captured. The cache intentionally keeps only
    one stream because executor processes are expected to run on one current
    CUDA device; if the current device changes, the cached stream is replaced.
    """
    global _MM_AUX_STREAM

    if max_prefetch_ahead <= 0:
        return None
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    device = torch.cuda.current_device()
    if _MM_AUX_STREAM is None or _MM_AUX_STREAM[0] != device:
        _MM_AUX_STREAM = (device, torch.cuda.Stream(device=device))
        logger.warning_once(
            f"Using multimodal encoder side stream on CUDA device {device} "
            f"with encoder_side_stream_max_ahead={max_prefetch_ahead}. "
            "This may increase peak GPU memory usage because raw multimodal "
            "encoder inputs and computed embeddings can be resident before "
            "request prefill.",
            key=f"mm_aux_stream_used_device_{device}",
        )
    return _MM_AUX_STREAM[1]


@contextlib.contextmanager
def _run_on_aux_stream(aux_stream: torch.cuda.Stream) -> Iterator[torch.cuda.Event]:
    """Run a block on `aux_stream` independently of the caller stream.

    Yields a CUDA event recorded on `aux_stream` when the block exits. Callers
    can wait on that event from another stream before consuming tensors written
    in the block.

    No entrance barrier is enforced: cross-iter MM encoder prefetch operates on
    data disjoint from the current iteration's batch, so serializing aux-stream
    work behind the caller stream would eliminate the overlap this stream
    exists for.
    """
    exit_event = torch.cuda.Event()
    with torch.cuda.stream(aux_stream):
        try:
            yield exit_event
        finally:
            # Keep the sync point valid even if the block raises after queuing
            # aux-stream work.
            exit_event.record()


@dataclass(frozen=True)
class PreparedLlmInputs:
    """Prepared inputs returned by `MultimodalModelMixin`."""

    input_ids: Optional[torch.Tensor]
    inputs_embeds: Optional[torch.Tensor]
    extra_embeds: Sequence[torch.Tensor] = ()


class MultimodalModelMixin:
    """Template-method mixin for PyTorch multimodal causal LM models.

    Concrete model forwards can call `prepare_multimodal_inputs` while keeping their explicit
    language-model delegation.

    Current limitations:

    * For the time being, the persistent multimodal encoder cache stores per-item embeddings for
      single-modality `MultimodalParams` objects.
    * Cache reuse is all-or-nothing for each `MultimodalParams` object: every item in that object
      hit the cache before cached embeddings are reused. Mixed-modality `MultimodalParams` objects
      bypass the persistent cache.
    """

    model_config: ModelConfig
    _multimodal_encoder_cache: Optional[TensorLRUCache] = None

    @classmethod
    def _cast_multimodal_encoder_dtype(
        cls,
        module: torch.nn.Module,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Cast a multimodal encoder dtype without materializing meta tensors."""

        def convert(tensor: torch.Tensor) -> torch.Tensor:
            if not (tensor.is_floating_point() or tensor.is_complex()):
                return tensor
            if tensor.device == torch.device("meta"):
                return torch.empty_like(tensor, dtype=dtype)
            return tensor.to(dtype=dtype)

        return module._apply(convert)

    def encode_multimodal_inputs(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> torch.Tensor:
        """Run model-specific multimodal encoder work.

        Returns the single primary multimodal embedding tensor for the supplied params. Rows are
        expected to be concatenated in request order, and special multimodal tokens occupy token
        positions but do not have rows here.
        """
        raise NotImplementedError

    @property
    def multimodal_token_ids(self) -> Optional[Sequence[int] | torch.Tensor]:
        """Return placeholder token ids in `input_ids` replaced by MM embeds.

        These are sentinel token positions whose text embeddings are replaced
        by multimodal embeddings. Return `None` to use the out-of-vocabulary
        sentinel behavior in `fuse_input_embeds`.
        """
        raise NotImplementedError

    @property
    def text_embedding_layer(self):
        """Return the token embedding layer used by `fuse_input_embeds`."""
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        """Return the width of each cached multimodal embedding row."""
        raise NotImplementedError

    @property
    def embedding_dtype(self) -> torch.dtype:
        """Return the dtype of each cached multimodal embedding row."""
        raise NotImplementedError

    def select_multimodal_params(
        self,
        multimodal_params: Sequence[MultimodalParams],
        num_context_requests: int,
    ) -> Sequence[MultimodalParams]:
        """Select the params that participate in multimodal encoder work.

        Returns the context-slice params with multimodal content. Helpers below
        this method (`get_multimodal_embeddings`, `find_input_mm_embeds`,
        `fuse_input_embeds`) operate on the returned list and therefore see
        only `has_content()` params. Models overriding this hook must
        preserve that invariant.
        """
        return [
            param for param in list(multimodal_params)[:num_context_requests] if param.has_content()
        ]

    def after_full_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_params: Sequence[MultimodalParams],
        embeddings: torch.Tensor,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional hook before active chunk rows are selected.

        Runs after cache lookup or encoder execution has produced full
        per-request multimodal embeddings, but before the mixin selects rows
        active in the current forward chunk.
        """
        return input_ids, embeddings

    def after_active_multimodal_embeddings(
        self,
        *,
        active_embeddings: list[torch.Tensor],
        multimodal_params: Sequence[MultimodalParams],
        **forward_kwargs: Any,
    ) -> tuple[list[torch.Tensor], Sequence[torch.Tensor]]:
        """Optional hook after active chunk rows are selected and before fusion.

        Models can transform or split the active multimodal embeddings here
        and return additional embedding tensors to fuse alongside the primary
        multimodal embeddings.
        """
        # Models with packed auxiliary features (e.g. Qwen3-VL) can split them here and return
        # them as extra embeds without changing the base flow.
        return active_embeddings, ()

    # A future optional mixin-owned forward can build on the same template method.
    def prepare_multimodal_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor],
        multimodal_params: Optional[Sequence[MultimodalParams]],
        num_context_requests: int,
        **forward_kwargs: Any,
    ) -> PreparedLlmInputs:
        """Prepare multimodal inputs for a concrete model forward.

        This method owns the common framework sequence around a model-specific
        encoder hook: retrieve/cache full request embeddings, select active
        chunk rows, run optional model hooks, and fuse rows into text embeds.
        """
        context_params = list(
            self.select_multimodal_params(
                multimodal_params or [],
                num_context_requests,
            )
        )
        if not context_params:
            return PreparedLlmInputs(input_ids=input_ids, inputs_embeds=None)

        full_embeddings = self._get_or_encode_multimodal_embeddings(context_params)

        input_ids, full_embeddings = self.after_full_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_params=context_params,
            embeddings=full_embeddings,
            **forward_kwargs,
        )

        active_embeddings = find_input_mm_embeds([full_embeddings], list(context_params))
        active_embeddings, extra_embeds = self.after_active_multimodal_embeddings(
            active_embeddings=active_embeddings,
            multimodal_params=context_params,
            **forward_kwargs,
        )

        fused_input_ids, inputs_embeds, fused_extra_embeds = self._fuse_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=active_embeddings,
            mm_token_ids=self.multimodal_token_ids,
            embedding_layer=self.text_embedding_layer,
            extra_embeds=extra_embeds,
            # `text_token_indices` / `mm_token_indices` are pre-computed by the
            # executor (see model_engine._prepare_inputs) and must reach
            # `fuse_input_embeds` to (a) preserve the active-chunk subset
            # contract when MM rows are a subset of visible MM tokens and
            # (b) avoid the torch.where host sync inside
            # `filter_mm_token_from_input_ids`.
            text_token_indices=forward_kwargs.get("text_token_indices"),
            mm_token_indices=forward_kwargs.get("mm_token_indices"),
        )
        return PreparedLlmInputs(
            input_ids=fused_input_ids,
            inputs_embeds=inputs_embeds,
            extra_embeds=fused_extra_embeds,
        )

    def _get_or_encode_multimodal_embeddings(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> torch.Tensor:
        """Return cached multimodal embeddings or run the encoder for misses.

        Delegates cache lookup and gather behavior to `get_multimodal_embeddings`, then validates
        the single tensor contract for both encoded and cached-only paths.
        """
        encoder_cache = self._get_multimodal_encoder_cache()
        if encoder_cache is not None:
            for param in multimodal_params:
                self._attach_encoder_cache_hit(param, encoder_cache)

        embeddings = get_multimodal_embeddings(
            encoder_forward_fn=self.encode_multimodal_inputs,
            multimodal_params=list(multimodal_params),
        )
        if encoder_cache is not None:
            for param in multimodal_params:
                self._write_encoder_cache_entries(param, encoder_cache)

        # Validate post-gather so cached-only paths (KV reuse, all-cached chunked prefill) are also
        # checked, not just paths that ran the encoder.
        self._validate_embeddings(embeddings, multimodal_params)
        return embeddings[0]

    def _get_multimodal_encoder_cache(self) -> Optional[TensorLRUCache]:
        """Return the per-model encoder cache, if enabled.

        The cache stores per-item embeddings for params that can be represented by one modality.
        See `_encoder_cache_keys` for the mixed-modality skip path and its technical limitation.
        """
        multimodal_config = self.model_config.multimodal_config
        if multimodal_config is None:
            return None

        max_bytes = multimodal_config.encoder_cache_max_bytes
        if max_bytes <= 0:
            logger.debug_once(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: disabled because "
                "multimodal_config.encoder_cache_max_bytes=0.",
                key="mm_encoder_cache_disabled",
            )
            return None

        if self._multimodal_encoder_cache is None:
            self._multimodal_encoder_cache = TensorLRUCache(
                max_bytes,
                clone_on_insert=True,
                name=_MM_ENCODER_CACHE_LOG_NAME,
            )
            try:
                embedding_dim = self.embedding_dim
                embedding_dtype = self.embedding_dtype
            except NotImplementedError:
                logger.info(
                    f"{_MM_ENCODER_CACHE_LOG_NAME}: created with max_bytes={max_bytes}, "
                    "clone_on_insert=True; embedding row capacity unavailable because the "
                    "model does not implement embedding_dim and embedding_dtype."
                )
            else:
                bytes_per_embedding_row = (
                    embedding_dim * torch.empty((), dtype=embedding_dtype).element_size()
                )
                max_embedding_rows = max_bytes // bytes_per_embedding_row
                logger.info(
                    f"{_MM_ENCODER_CACHE_LOG_NAME}: created with max_bytes={max_bytes}, "
                    f"max_embedding_rows={max_embedding_rows}, embedding_dim={embedding_dim}, "
                    f"embedding_dtype={embedding_dtype}, clone_on_insert=True"
                )
        return self._multimodal_encoder_cache

    @staticmethod
    def _encoder_cache_modality(param: MultimodalParams) -> Optional[str]:
        """Return the single modality represented by `param`, if cacheable.

        `None` means the params either do not identify a modality or contain
        multiple modality inputs. The persistent encoder cache deliberately does
        not cache mixed-modality params today.
        """
        mm_data = param.multimodal_data or {}
        modality = mm_data.get("modality_type")
        if isinstance(modality, str):
            return modality

        modalities = [key for key in _MM_DATA_INPUT_MODALITY_KEYS if key in mm_data]
        if len(modalities) != 1:
            # Mixed-modality params are skipped because the cache key metadata is request-item
            # oriented: `multimodal_hashes` and `multimodal_embedding_lengths` are parallel per
            # item, but there is no parallel per-item modality list. Without that, a cache key
            # cannot unambiguously distinguish, for example, an image item from an audio item inside
            # the same params object.
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping params with {len(modalities)} detected "
                "modalities."
            )
            return None
        return modalities[0]

    @classmethod
    def _encoder_cache_keys(
        cls,
        param: MultimodalParams,
    ) -> Optional[list[Hashable]]:
        """Build per-item encoder cache keys for single-modality params.

        The returned keys split one request's concatenated encoder output by
        `multimodal_embedding_lengths`, using the same modality for every item.

        Mixed-modality params are not cacheable until runtime metadata carries a
        modality per item alongside `multimodal_hashes` and embedding lengths.
        """
        mm_input = param.multimodal_input
        mm_data = param.multimodal_data or {}
        if mm_input is None:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping params without multimodal hashes."
            )
            return None

        modality = cls._encoder_cache_modality(param)
        embedding_lengths = mm_data.get("multimodal_embedding_lengths")
        kwargs_hash = mm_data.get("mm_processor_kwargs_hash")
        if modality is None or not isinstance(embedding_lengths, list) or kwargs_hash is None:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping unkeyable params, "
                f"has_modality={modality is not None}, "
                f"has_embedding_lengths={isinstance(embedding_lengths, list)}, "
                f"has_processor_kwargs_hash={kwargs_hash is not None}"
            )
            return None
        if len(mm_input.multimodal_hashes) != len(embedding_lengths):
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping params with mismatched "
                "multimodal_hashes and multimodal_embedding_lengths counts"
            )
            return None

        # The item hash, embedding row count, processor kwargs, and modality fully describe a
        # reusable item embedding. Request order is excluded so the same item can be reused from a
        # different request layout; the current request order is restored when cached item tensors
        # are concatenated below.
        return [
            (
                modality,
                tuple(item_hash),
                int(embedding_length),
                kwargs_hash,
            )
            for item_hash, embedding_length in zip(
                mm_input.multimodal_hashes,
                embedding_lengths,
                strict=True,
            )
        ]

    @classmethod
    def _attach_encoder_cache_hit(
        cls,
        param: MultimodalParams,
        encoder_cache: TensorLRUCache,
    ) -> None:
        if param.multimodal_data.get("multimodal_embedding") is not None:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: request-local multimodal embedding present; "
                "skipping persistent cache lookup"
            )
            return

        keys = cls._encoder_cache_keys(param)
        if not keys:
            return

        cached_embeddings = []
        for key in keys:
            cached_embedding = encoder_cache.get(key)
            if cached_embedding is None:
                # `get_multimodal_embeddings` treats a param as either fully cached or uncached.
                # Attaching partial hits would make the later concatenated tensor ambiguous because
                # there is no placeholder for missing item rows inside `multimodal_embedding`.
                logger.debug(
                    f"{_MM_ENCODER_CACHE_LOG_NAME}: cache miss; hit_items={len(cached_embeddings)},"
                    f" total_items={len(keys)}."
                )
                return
            cached_embeddings.append(cached_embedding)

        param.multimodal_data["multimodal_embedding"] = torch.cat(cached_embeddings, dim=0)
        logger.debug(
            f"{_MM_ENCODER_CACHE_LOG_NAME}: full cache hit for {len(keys)} item entries, "
            f"rows={param.multimodal_data['multimodal_embedding'].shape[0]}."
        )

    @classmethod
    def _write_encoder_cache_entries(
        cls,
        param: MultimodalParams,
        encoder_cache: TensorLRUCache,
    ) -> None:
        keys = cls._encoder_cache_keys(param)
        if not keys:
            return

        embedding = param.multimodal_data.get("multimodal_embedding")
        if isinstance(embedding, list):
            embedding = torch.cat(embedding, dim=0)
            param.multimodal_data["multimodal_embedding"] = embedding
        if not isinstance(embedding, torch.Tensor):
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping write because no tensor embedding was "
                "attached after encoder execution."
            )
            return

        embedding_lengths = param.multimodal_data["multimodal_embedding_lengths"]
        if sum(embedding_lengths) != embedding.shape[0]:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping write because embedding row count "
                "does not match multimodal_embedding_lengths."
            )
            return

        # Encoder outputs are concatenated per params object. Splitting by item length lets future
        # requests reuse matching items independently, even when their request-level item order
        # differs.
        inserted_entries = 0
        rejected_entries = 0
        for key, item_embedding in zip(
            keys,
            torch.split(embedding, embedding_lengths, dim=0),
            strict=True,
        ):
            if encoder_cache.put(key, item_embedding):
                inserted_entries += 1
            else:
                rejected_entries += 1
        logger.debug(
            f"{_MM_ENCODER_CACHE_LOG_NAME}: wrote {inserted_entries} item entries, "
            f"rejected={rejected_entries}, rows={embedding.shape[0]}."
        )
        encoder_cache.log_stats("multimodal encoder cache write.")

    def _fuse_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor],
        mm_token_ids: Optional[Sequence[int] | torch.Tensor],
        embedding_layer,
        extra_embeds: Sequence[torch.Tensor],
        text_token_indices: Optional[torch.Tensor] = None,
        mm_token_indices: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Sequence[torch.Tensor]]:
        """Thin adapter over `fuse_input_embeds`.

        The framework does not forward `prepare_multimodal_inputs` kwargs
        into `fuse_input_embeds`; only inputs the helper actually consumes
        are surfaced here. Models needing to bypass token filtering should
        pass pre-computed `text_token_indices`/`mm_token_indices`.
        """
        if mm_token_ids is not None and not isinstance(mm_token_ids, torch.Tensor):
            mm_token_ids = torch.tensor(
                list(mm_token_ids), dtype=input_ids.dtype, device=input_ids.device
            )

        result = fuse_input_embeds(
            embedding_layer=embedding_layer,
            input_ids=input_ids,
            mm_embeds=multimodal_embeddings,
            mm_token_ids=mm_token_ids,
            text_token_indices=text_token_indices,
            mm_token_indices=mm_token_indices,
            extra_embeds=list(extra_embeds) if extra_embeds else None,
        )
        if len(result) == 3:
            fused_input_ids, inputs_embeds, fused_extra_embeds = result
            return fused_input_ids, inputs_embeds, fused_extra_embeds or ()

        fused_input_ids, inputs_embeds = result
        return fused_input_ids, inputs_embeds, ()

    @staticmethod
    def _validate_embeddings(
        embeddings: list[torch.Tensor],
        multimodal_params: Sequence[MultimodalParams],
    ) -> None:
        """Validate gathered embeddings' row count against runtime metadata.

        Skipped if any param lacks `multimodal_runtime.total_embeds_in_request`, since the contract
        cannot be evaluated without complete metadata.
        """
        if len(embeddings) != 1:
            raise ValueError(
                f"MultimodalModelMixin requires a single embedding tensor, got {len(embeddings)} "
                "tensors."
            )

        embeddings_tensor = embeddings[0]
        expected_rows = 0
        has_runtime_metadata = []
        for param in multimodal_params:
            runtime = param.multimodal_runtime
            has_runtime = runtime is not None and runtime.total_embeds_in_request is not None
            has_runtime_metadata.append(has_runtime)
            if has_runtime:
                expected_rows += runtime.total_embeds_in_request

        if any(has_runtime_metadata) and not all(has_runtime_metadata):
            raise ValueError(
                "Multimodal runtime metadata must be present for every param or none of them."
            )
        if not all(has_runtime_metadata):
            logger.debug(
                "Skipping multimodal embedding row-count validation: runtime metadata missing "
                "for all params."
            )
            return

        actual_rows = embeddings_tensor.shape[0]
        if actual_rows != expected_rows:
            raise ValueError(
                f"Multimodal embedding row count mismatch: expected {expected_rows}, got {actual_rows}."
            )


def _collect_cross_iter_prefetch_candidates(
    pending_requests: Sequence["LlmRequest"],
    in_flight_request_ids: Iterable[int],
    max_prefetch: int,
    max_prefetch_ahead: int,
) -> list[tuple["LlmRequest", Dict[str, Any], torch.Tensor]]:
    """Select cross-iteration prefetch candidates without touching CUDA.

    Returns up to `max_prefetch` `(request, multimodal_data, cumsum)` tuples
    while keeping total not-in-flight prefetched requests at or below
    `max_prefetch_ahead`. A new candidate must have a supported raw input
    modality in `multimodal_data`, no cached `multimodal_embedding`, and a valid
    `multimodal_embed_mask_cumsum`. Existing cached embeddings or pending encoder
    events count against the ahead limit only when attached to a request with
    real MM encoder work.
    """
    if max_prefetch <= 0 or max_prefetch_ahead <= 0:
        return []

    in_flight = set(in_flight_request_ids)
    outstanding_prefetches = 0
    candidates: list[tuple["LlmRequest", Dict[str, Any], torch.Tensor]] = []

    for req in pending_requests:
        if req.py_request_id in in_flight:
            continue
        mm_data = req.py_multimodal_data or {}
        has_cached_embedding = mm_data.get("multimodal_embedding") is not None
        has_raw_mm_input = any(key in mm_data for key in _MM_DATA_INPUT_MODALITY_KEYS)
        if req.py_mm_encoder_event is not None:
            if has_cached_embedding or has_raw_mm_input:
                outstanding_prefetches += 1
            continue
        if has_cached_embedding:
            outstanding_prefetches += 1
            continue
        if not has_raw_mm_input:
            continue
        cumsum = mm_data.get("multimodal_embed_mask_cumsum")
        if cumsum is None:
            continue
        candidates.append((req, mm_data, cumsum))

    available_slots = min(max_prefetch, max_prefetch_ahead - outstanding_prefetches)
    if available_slots <= 0:
        return []
    return candidates[:available_slots]


def _dispatch_cross_iter_prefetch(
    model: "MultimodalModelMixin",
    candidates: Sequence[tuple["LlmRequest", Dict[str, Any], torch.Tensor]],
    aux_stream: "torch.cuda.Stream",
) -> None:
    """H2D-copy MM data, run the encoder, and cache embeddings on `aux_stream`.

    Stamps a CUDA event on every candidate's `py_mm_encoder_event` so the
    next iteration's consume site waits on it before reading cached tensors.
    The event covers all work queued in the aux-stream block, so the same
    event object is shared across all candidates.
    """
    params_list = [
        MultimodalParams(
            multimodal_data=mm_data,
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=0,
                chunk_end_pos=cumsum.numel(),
                embed_mask_cumsum=cumsum,
            ),
        )
        for _, mm_data, cumsum in candidates
    ]

    # Prefetch targets requests outside the current iteration, so their
    # multimodal tensors are not touched by the main stream. The caller queues
    # this after the iteration's LLM kernels so aux-stream H2D copies and
    # encoder work can overlap them.
    #
    # Ordering is handled by `encoder_event`, and tensor lifetime is anchored
    # by `req.py_multimodal_data` until the request is consumed or terminated.
    # Keeping `record_stream` out also keeps this path modality-neutral.
    encoder_event = None
    try:
        with _run_on_aux_stream(aux_stream) as encoder_event:
            for p in params_list:
                p.to_device(
                    "multimodal_data",
                    "cuda",
                    pin_memory=prefer_pinned(),
                    target_keywords=getattr(model, "multimodal_data_device_paths", None),
                )
            # `to_device` may replace `multimodal_data` with a new dict; reattach
            # the (possibly new) dict to each request so the next iteration's
            # `_prepare_inputs` sees the cached embedding stamped below. Mirrors
            # the reassignment at the canonical to_device call site in
            # model_engine._prepare_inputs.
            for (req, _, _), p in zip(candidates, params_list):
                req.py_multimodal_data = p.multimodal_data
            encoder_output = model.encode_multimodal_inputs(params_list)
            _cache_multimodal_embeddings(params_list, [encoder_output])
    finally:
        # Stash the event on every candidate's durable LlmRequest (not the
        # per-iter `MultimodalParams`), since `_prepare_inputs` rebuilds the
        # wrapper each iteration. The transfer to `MultimodalParams.encoder_event`
        # happens in `_prepare_inputs` when the request is next scheduled.
        #
        # This runs in `finally` (and `_run_on_aux_stream` records the event in
        # its own `finally`) so that on partial failure -- e.g. `to_device`
        # mutated `req.py_multimodal_data` in place to aux-stream CUDA tensors,
        # then `encode_multimodal_inputs` raised -- the consumer still has an
        # event to wait on before reading those tensors on the main stream.
        # Without this, the request would carry aux-stream tensors with no sync
        # point, producing a cross-stream data race in the next iteration's
        # in-iter encode path.
        if encoder_event is not None:
            for req, _, _ in candidates:
                req.py_mm_encoder_event = encoder_event


def maybe_prefetch_mm_encoder_for_next_iter(
    model: Any,
    pending_requests: Sequence["LlmRequest"],
    in_flight_request_ids: Iterable[int] = (),
    max_prefetch: int = 1,
    max_prefetch_ahead: Optional[int] = None,
) -> int:
    """Speculative cross-iteration MM encoder prefetch on a side CUDA stream.

    For up to `max_prefetch` `pending_requests`, subject to the outstanding ahead cap, runs
    `model.encode_multimodal_inputs` on a side CUDA stream.

    The resulting embeddings are written into `request.py_multimodal_data` so the next iteration's
    `_prepare_inputs` picks them up via the standard cache path, and a CUDA event is stamped on
    `request.py_mm_encoder_event` for `_prepare_inputs` to transfer onto the new `MultimodalParams`.

    The mixin consume sites (e.g. `get_multimodal_embeddings)` need to wait on the event before
    reading the cached tensors.

    While the current iteration's LLM kernels run on the main stream, this queues encoder work for
    an "admit-likely context" request on the aux stream.

    Mis-predictions waste GPU time, but cached embeddings remain valid until the request is admitted
    or terminated. Examples include:

    - The prefetched request can be terminated before admission (client cancel / disconnect,
      timeout, validation failure, ...).
    - With `max_prefetch < len(pending)`, if the head is bumped by budget reasons, the next-admitted
      request is one we did not prefetch.

    Gated by `MultimodalConfig.encoder_side_stream_max_ahead`: 0 disables the side stream; a
    positive integer enables it and caps the total number of not-in-flight requests with
    prefetched MM encoder work.

    Returns the number of requests for which an encoder kick-off was queued.
    """
    if not isinstance(model, MultimodalModelMixin):
        return 0
    if max_prefetch <= 0:
        return 0
    if max_prefetch_ahead is None:
        max_prefetch_ahead = 0
    if max_prefetch_ahead <= 0:
        return 0

    aux_stream = _get_mm_aux_stream(max_prefetch_ahead)
    if aux_stream is None:
        return 0

    candidates = _collect_cross_iter_prefetch_candidates(
        pending_requests, in_flight_request_ids, max_prefetch, max_prefetch_ahead
    )
    if not candidates:
        return 0
    _dispatch_cross_iter_prefetch(model, candidates, aux_stream)
    return len(candidates)
