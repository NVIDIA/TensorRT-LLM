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

"""Encoder embedding storage, active-row selection, and token fusion."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, cast

import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger


class TokenEmbedding(Protocol):
    """Token lookup required by the default multimodal fusion path."""

    num_embeddings: int

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...


def _get_uncached_multimodal_params(
    multimodal_params: List[MultimodalParams],
) -> List[MultimodalParams]:
    """
    Get uncached multimodal params that need encoder processing for chunk prefill.
    """
    params_to_run = []

    for param in multimodal_params:
        # Skip if no multimodal content
        if not param.has_content():
            continue

        # Check if embeddings are already cached
        if (
            param.multimodal_data
            and "multimodal_embedding" in param.multimodal_data
            and param.multimodal_data["multimodal_embedding"] is not None
        ):
            logger.debug("Skipping encoder forward for param with cached multimodal_embedding")
            continue

        # This param needs encoder processing
        params_to_run.append(param)

    return params_to_run


def _store_chunked_prefill_embeddings(
    multimodal_params: List[MultimodalParams],
    embeddings: List[torch.Tensor],
) -> None:
    """
    Store computed multimodal embeddings back to multimodal_data to avoid recomputation.

    NOTE: this function only stores multimodal embeddings within the current request context,
    mostly for chunked prefill. It does not persist embeddings across different requests or sessions.
    """
    # TODO: support multiple multimodal modalities per request
    if len(embeddings) > 1:
        raise ValueError("Multiple modalities caching is not supported yet.")
    mm_embed = embeddings[0]

    # Collect embedding lengths for each parameter
    embed_lengths = []
    for param in multimodal_params:
        if param.multimodal_runtime is not None:
            embed_lengths.append(param.multimodal_runtime.total_embeds_in_request)

    # Validate total length matches
    total_expected = sum(embed_lengths)
    assert len(mm_embed) == total_expected, (
        f"Number of mm_embeds ({len(mm_embed)}) does not match expected total ({total_expected})"
    )

    # Use torch.split for efficient tensor splitting
    split_embeddings = torch.split(mm_embed, embed_lengths, dim=0)
    valid_params = [p for p in multimodal_params if p.multimodal_runtime is not None]

    # Cache split embeddings to each parameter
    logger.debug(
        f"Caching {len(split_embeddings)} multimodal embedding chunks in {len(multimodal_params)} params"
    )
    for param, embed_chunk in zip(valid_params, split_embeddings, strict=True):
        param.multimodal_data["multimodal_embedding"] = embed_chunk

    logger.debug(f"Cached {len(split_embeddings)} multimodal embedding chunks in this iteration")


def _normalize_encoder_embeddings(
    encoder_embeddings: Union[torch.Tensor, List[torch.Tensor]],
) -> List[torch.Tensor]:
    if isinstance(encoder_embeddings, torch.Tensor):
        return [encoder_embeddings]

    if not isinstance(encoder_embeddings, list) or not all(
        isinstance(embedding, torch.Tensor) for embedding in encoder_embeddings
    ):
        raise TypeError("encoder_forward_fn must return a torch.Tensor or a list of torch.Tensor.")

    return encoder_embeddings


def _join_embeddings(
    embeddings: List[torch.Tensor], *, start: int = 0, end: Optional[int] = None
) -> torch.Tensor:
    """Join per-item/per-request embeddings along dim 0 without a needless copy.

    ``torch.cat`` always allocates, so calling it on a one-element list
    duplicates the whole embedding. That matters here because the same rows
    are joined several times on the way to the LLM input (per request, per
    batch, and again in ``fuse_input_embeds``), and each redundant copy is a
    full extra allocation live at the prefill peak. Multimodal embeddings are
    read-only downstream -- they are sliced and scattered into the input
    embedding buffer, never written in place -- so returning the sole tensor
    is equivalent to concatenating it. If a row range is given, slice each
    segment first so rows outside the current prefill chunk are never copied.
    """
    if start != 0 or end is not None:
        slices = []
        offset = 0
        for embedding in embeddings:
            segment_end = offset + embedding.shape[0]
            slice_end = segment_end if end is None else min(end, segment_end)
            if max(start, offset) < slice_end:
                slices.append(embedding[max(start - offset, 0) : slice_end - offset])
            offset = segment_end
            if end is not None and offset >= end:
                break
        if not slices:
            return embeddings[0][:0]
        embeddings = slices

    return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=0)


def get_multimodal_embeddings(
    encoder_forward_fn: Callable[
        ...,
        torch.Tensor | List[torch.Tensor],
    ],
    multimodal_params: List[MultimodalParams],
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    """
    High-level utility to get multimodal embeddings from encoder or cached embeddings.

    This function will:
    1. Identify which parameters need encoder processing
    2. Run encoder forward only on uncached parameters
    3. Cache newly computed embeddings (if enabled)
    4. Gather all embeddings for the batch

    Args:
        encoder_forward_fn: Callable that performs encoder forward pass.
                           Should accept List[MultimodalParams] and return either
                           a single torch.Tensor or List[torch.Tensor].
        multimodal_params: All multimodal parameters in the batch.
        encoder_kwargs: Optional kwargs to pass to encoder_forward_fn.
    Returns:
        List of multimodal embeddings for all multimodal params in the batch.
    """
    if not multimodal_params:
        return []

    # Wait before touching tensors produced on the MM side stream. Do not clear the event here;
    # repeated stream-side waits are cheap, and leaving the event field untouched avoids races if a
    # caller accidentally reuses it.
    # Register attached embeddings with the consumer stream as well: the request drops its Python
    # references after prefill, potentially before the asynchronous gather below has finished
    # reading them.
    for param in multimodal_params:
        if param.encoder_event is not None:
            consumer_stream = torch.cuda.current_stream()
            consumer_stream.wait_event(param.encoder_event)
            embeds = param.multimodal_data.get("multimodal_embedding")
            if isinstance(embeds, torch.Tensor):
                embeds = [embeds]
            if isinstance(embeds, list):
                for embed in embeds:
                    if isinstance(embed, torch.Tensor) and embed.is_cuda:
                        embed.record_stream(consumer_stream)

    # Step 1: Find uncached multimodal params that need encoder processing
    uncached_multimodal_params = _get_uncached_multimodal_params(multimodal_params)

    # Step 2: Run encoder forward only on uncached parameters
    if uncached_multimodal_params:
        kwargs = encoder_kwargs or {}
        encoder_embeddings = encoder_forward_fn(uncached_multimodal_params, **kwargs)
        encoder_embeddings = _normalize_encoder_embeddings(encoder_embeddings)

        # TODO: support multiple multimodal modalities per request
        if len(encoder_embeddings) > 1:
            logger.warning(
                f"Multiple modalities caching is not supported yet. "
                f"encoder returned {len(encoder_embeddings)} embeddings "
                f"(types: {[type(e).__name__ for e in encoder_embeddings]}, "
                f"shapes: {[e.shape if hasattr(e, 'shape') else 'N/A' for e in encoder_embeddings]}) "
                f"for {len(uncached_multimodal_params)} uncached params. "
                f"encoder_forward_fn={encoder_forward_fn}"
            )
            return encoder_embeddings

        # Validate that multimodal_runtime has required attributes for caching
        if (
            not hasattr(uncached_multimodal_params[0], "multimodal_runtime")
            or uncached_multimodal_params[0].multimodal_runtime is None
            or uncached_multimodal_params[0].multimodal_runtime.total_embeds_in_request is None
        ):
            logger.warning(
                "Multimodal runtime data missing or incomplete, will not cache embeddings."
            )
            return encoder_embeddings

        # Step 3: Cache the computed embeddings to multimodal_data["multimodal_embedding"].
        # A TP-sharded encoder can return an NCCL window tensor whose lease ends
        # with the enclosing ModelCaller scope, while chunked prefill reads the
        # cached rows in later iterations, so cache an owned copy instead of
        # views of the encoder output.
        encoder_embeddings = [encoder_embeddings[0].clone()]
        _store_chunked_prefill_embeddings(uncached_multimodal_params, encoder_embeddings)

    # Step 4: Gather after the stream wait and any encoder/cache writes.
    return get_attached_multimodal_embeddings(multimodal_params)


def get_attached_multimodal_embeddings(
    multimodal_params: List[MultimodalParams],
) -> List[torch.Tensor]:
    """Gather embeddings already stored on MultimodalParams.

    Use this on E/P prefill workers and cached-only paths. The encoder already ran somewhere else.
    Normally `find_input_mm_embeds` selects active rows afterwards. If item
    scheduling attached current-chunk embeddings, select the same chunk from
    any full-request embeddings before joining the batch. This keeps mixed
    batches in one row layout without copying unrelated future chunks.

    Side-stream-prefetched requests must use `get_multimodal_embeddings`, which waits on
    `encoder_event` and registers attached tensors with the consuming stream before gathering them.
    """
    current_chunk_only = any(
        param.multimodal_data.get("multimodal_embedding_is_chunk", False)
        for param in multimodal_params
    )
    attached_embeddings = []
    for param in multimodal_params:
        embeds = param.multimodal_data.get("multimodal_embedding")
        # No attached embedding for this request.
        if embeds is None:
            continue
        runtime = param.multimodal_runtime
        if (
            current_chunk_only
            and runtime is not None
            and not param.multimodal_data.get("multimodal_embedding_is_chunk", False)
        ):
            # Keep the full attached payload intact for later chunks/cache
            # writes; only this forward's gathered tensor is chunk-local.
            start = runtime.num_cached_mm_tokens
            end = start + runtime.num_mm_tokens_in_chunk
            if isinstance(embeds, list):
                embeds = _join_embeddings(embeds, start=start, end=end)
            else:
                embeds = embeds[start:end]
        elif isinstance(embeds, list):
            # Legacy paths can attach a list of full-request output pieces.
            embeds = _join_embeddings(embeds)
            param.multimodal_data["multimodal_embedding"] = embeds
        if not isinstance(embeds, torch.Tensor):
            raise TypeError("multimodal_embedding must be a torch.Tensor")
        attached_embeddings.append(embeds)

    if not attached_embeddings:
        return []
    # Match get_multimodal_embeddings output: one concatenated tensor.
    return [_join_embeddings(attached_embeddings)]


def find_input_mm_embeds(
    mm_embeds: List[torch.Tensor], multimodal_params: List[MultimodalParams]
) -> List[torch.Tensor]:
    """
    Find the multimodal mm_embeds that need processing from multimodal_params for each batch.
    Supports both KV cache reuse and chunked prefill scenarios.

    Args:
        - mm_embeds: List[torch.Tensor] - Multimodal embeddings for each batch
        - multimodal_params: List[MultimodalParams] - Multimodal parameters with runtime data

    Returns:
        - List[torch.Tensor] - Sliced mm_embeds containing only tokens that need processing:
          - For KV cache reuse: tokens that are not cached
          - For chunked prefill: tokens that are in the current chunk
          - For mixed scenarios: both uncached and current chunk tokens
          - Empty list if all tokens are cached or beyond current chunk

    Note:
        - Supports both individual batching (len(mm_embeds) == len(multimodal_params))
          and pre-concatenated batching (len(mm_embeds) == 1)
        - Call get_attached_multimodal_embeddings before this helper when
          embeddings are already attached to multimodal_params.
        - Handles chunked prefill by considering chunk boundaries and current chunk tokens
        - Example: if a request has 8 MM embed rows, 2 cached rows, and 3 rows
          in the current chunk, this keeps rows [2:5].
        - This function reads only CPU-resident counts from
          ``MultimodalParams.multimodal_runtime`` and never touches GPU
          tensors, so it does not introduce host-device synchronization.
          The companion ``torch.where`` host sync sits in ``fuse_input_embeds``
          (via ``filter_mm_token_from_input_ids``) and is avoided by routing
          executor-precomputed ``mm_token_indices`` / ``text_token_indices``
          through to that call — see the contract there.
    """
    if not isinstance(mm_embeds, list):
        raise TypeError("mm_embeds must be a list")

    # Current support two batching modes:
    # 1. Pre-concatenated mm_embeds for each batch, i.e., len(mm_embeds) == 1
    # 2. Individual mm_embeds for each multimodal param, i.e., len(mm_embeds) == len(multimodal_params)
    if len(mm_embeds) > 1 and len(mm_embeds) != len(multimodal_params):
        raise ValueError(
            f"Number of mm_embeds ({len(mm_embeds)}) does not match number of "
            f"multimodal params ({len(multimodal_params)})."
        )

    if not multimodal_params or multimodal_params[0].multimodal_runtime is None:
        # No slicing, return the full mm_embeds
        return mm_embeds

    total_mm_tokens = sum(
        param.multimodal_runtime.num_mm_tokens_in_chunk
        for param in multimodal_params
        if param.multimodal_runtime is not None
    )

    if total_mm_tokens == 0:
        logger.debug(
            "All multimodal tokens are cached or beyond current chunk, skipping vision encoder forward"
        )
        return []

    if not mm_embeds:
        raise ValueError(
            "No multimodal embeddings were provided or cached for active multimodal tokens."
        )

    if total_mm_tokens == sum(mm_embed.shape[0] for mm_embed in mm_embeds):
        # Item-scheduled batches were already sliced by the producer/gather
        # path. Do not apply the cached-prefix offset a second time.
        return mm_embeds

    current_pos = 0
    slices = []
    for param in multimodal_params:
        runtime = param.multimodal_runtime
        if runtime is None:
            continue
        local_start_pos = runtime.num_cached_mm_tokens
        local_end_pos = local_start_pos + runtime.num_mm_tokens_in_chunk
        slices.append((current_pos + local_start_pos, current_pos + local_end_pos))
        if len(mm_embeds) == 1:  # pre-concatenated; advance global cursor
            current_pos += runtime.total_embeds_in_request

    if len(mm_embeds) == 1:
        sliced = [mm_embeds[0][start:end] for start, end in slices]
        return [_join_embeddings(sliced)]
    return [mm_embeds[i][start:end] for i, (start, end) in enumerate(slices)]


def filter_mm_token_from_input_ids(
    input_ids: torch.IntTensor,
    vocab_size: int,
    mm_token_ids: Optional[torch.IntTensor] = None,
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    """
    Filter multimodal tokens from input_ids.
    Args:
        input_ids: shape [text_total_length + mm_total_length].
        vocab_size: size of the model's vocabulary
        mm_token_ids: possible token ids for multimodal tokens, if known. If
            not known and set to None, it is assumed that the multimodal tokens
            are out-of-vocabulary tokens i.e. the ``input_ids`` contains tokens
            >= vocab_size that represent the multimodal tokens.
    Note:
        Example: input_ids=[1, 55, 2, 101], vocab_size=100, and
        mm_token_ids=[55] returns mm_token_indices=[1]; token 101 is text
        because explicit mm_token_ids overrides the OOV fallback.
        This function involves host-device synchronization due to torch.where() (= torch.nonzero) requiring
        host allocation. The output indices reside on the same device as input_ids.
    Returns:
        text_token_indices: indices of text tokens in the input_ids
        mm_token_indices: indices of multimodal tokens in the input_ids
    """
    if mm_token_ids is None:
        # If mm_token_ids is None, assume the multimodal tokens are out-of-vocab
        # (input_ids >= vocab_size). Avoids torch.isin() over a potentially
        # unbounded mm_token_ids set.
        mm_token_mask = input_ids >= vocab_size
    else:
        mm_token_ids = mm_token_ids.to(input_ids.device, dtype=input_ids.dtype)
        mm_token_mask = torch.isin(input_ids, mm_token_ids)
    # NOTE: torch.where() enforces a host sync
    text_token_indices = torch.where(~mm_token_mask)[0]
    mm_token_indices = torch.where(mm_token_mask)[0]
    return text_token_indices, mm_token_indices


def _as_copy_index(index: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    """Return ``index`` as the int64 index ``index_copy_`` needs on ``destination``.

    Explicit indices can come from callers with another dtype or device; for the
    executor's int64 device indices this is a no-op.
    """
    return index.to(device=destination.device, dtype=torch.long, non_blocking=True)


def fuse_input_embeds(
    embedding_layer: TokenEmbedding,
    input_ids: torch.IntTensor,
    mm_embeds: List[torch.Tensor],
    mm_token_ids: Optional[torch.IntTensor] = None,
    text_token_indices: Optional[torch.IntTensor] = None,
    mm_token_indices: Optional[torch.IntTensor] = None,
    extra_embeds: Optional[List[torch.Tensor]] = None,
    **kwargs,
    # TODO: make unified return type for all models
) -> Union[
    Tuple[Optional[torch.IntTensor], Optional[torch.FloatTensor]],
    Tuple[
        Optional[torch.IntTensor], Optional[torch.FloatTensor], Optional[List[torch.FloatTensor]]
    ],
]:
    """
    Fuse text and multimodal rows into [text_total_length + mm_total_length, hidden_dim].

    Look up text embeddings and scatter multimodal rows into their token positions.

    Args:
        embedding_layer: embedding layer of the model.
        input_ids: Packed token IDs of shape [text_total_length + mm_total_length],
            with padding removed across the in-flight batch.
        mm_embeds: List[(mm_total_length1, hidden_dim), ..., (mm_total_lengthi, hidden_dim)].
        mm_token_ids: Known multimodal token IDs. None selects the out-of-vocabulary
            placeholder convention.
        extra_embeds: Optional list of extra embed tensors for models that support it (e.g., Qwen3-VL/Qwen3-MoE-VL).
    Returns:
        With no multimodal rows, return input_ids and no input embeddings.
        Otherwise return no input IDs and fused embeddings of shape [total_length, hidden_dim].
        Auxiliary embeddings, when present, form the third return value.
    Note:
        - Use explicit indices when both are supplied. If either is missing, derive both
          from mm_token_ids, or from the OOV convention when mm_token_ids is None.
        - Example: len(torch.cat(mm_embeds)) must match len(mm_token_indices);
          for chunked prefill, pass only the current chunk's mm_embeds or
          explicit indices for the active MM token positions.
        - Sync-free contract: passing both ``text_token_indices`` and
          ``mm_token_indices`` skips the GPU ``torch.where`` host sync. The
          executor (``DecoderInputPreparer.prepare_inputs``) precomputes them on a CPU
          ``input_ids`` copy via ``runners.prepare_multimodal_indices`` (which uses
          ``filter_mm_token_from_input_ids`` against ``self.model.mm_token_ids``
          when present, else the OOV fallback ``>= vocab_size``) and ships
          them as pinned async H2D tensors in the inputs dict. VLM forwards
          are expected to forward these explicitly rather than relying on a
          ``**kwargs`` splat — if they don't, this function falls back to the
          host-syncing ``filter_mm_token_from_input_ids`` on GPU ``input_ids``.
        - In-vocab fast path: when ``mm_token_ids`` is provided, the caller is
          declaring that mm placeholder IDs are real vocabulary entries. In
          that case the text path skips its ``index_select`` + ``torch.empty``
          + text scatter and embeds the full ``input_ids`` once, overwriting
          mm rows afterwards. This is the path taken by every VLM currently
          in tree EXCEPT Hyperclovax, which keeps the legacy ``vocab_size + 1``
          OOV remap in its input processor and therefore passes
          ``mm_token_ids=None`` so the OOV branch is taken — embedding an OOV
          id would be out-of-bounds.
    """
    if len(mm_embeds) == 0:
        if extra_embeds is not None and len(extra_embeds) > 0:
            return input_ids, None, extra_embeds
        return input_ids, None

    mm_embed = _join_embeddings(mm_embeds)

    # TODO: derive a missing index tensor from the supplied complement without host synchronization.
    if text_token_indices is None or mm_token_indices is None:
        # Filtering uses torch.where(), which synchronizes CUDA inputs with the host.
        text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
            input_ids, vocab_size=embedding_layer.num_embeddings, mm_token_ids=mm_token_ids
        )
    if mm_token_indices.shape[0] != mm_embed.shape[0]:
        raise ValueError(
            f"Multimodal token count mismatch: found {len(mm_token_indices)} image tokens in input_ids "
            f"but received {mm_embed.shape[0]} image embeddings."
        )

    if mm_token_ids is not None:
        # In-vocab fast path: caller declared mm tokens are real vocabulary
        # IDs, so a single full embedding lookup is safe. Saves the
        # ``index_select`` over text positions, the ``torch.empty`` alloc, and
        # the text scatter (~2 GPU kernels per VLM forward) at the cost of
        # ~mm-fraction extra embedding lookups whose result is overwritten.
        input_embeds = embedding_layer(input_ids)
    else:
        # OOV path: input_ids holds placeholder IDs >= num_embeddings at mm
        # positions (e.g. ``vocab_size + 1``), so embedding the full sequence
        # would be out-of-bounds. Gather only the in-vocab text positions and
        # scatter into a fresh buffer.
        text_embed = embedding_layer(input_ids[text_token_indices])
        input_embeds = torch.empty(
            input_ids.shape[0], mm_embed.shape[-1], device=text_embed.device, dtype=text_embed.dtype
        )
        input_embeds.index_copy_(0, _as_copy_index(text_token_indices, input_embeds), text_embed)
    if extra_embeds is not None and len(extra_embeds) > 0:
        # only support single modality for deepstack features for now
        for i, extra_feature in enumerate(extra_embeds):
            extra_embed = torch.zeros(
                input_ids.shape[0],
                mm_embed.shape[-1],
                device=extra_feature.device,
                dtype=extra_feature.dtype,
            )
            extra_embed.index_copy_(0, _as_copy_index(mm_token_indices, extra_embed), extra_feature)
            extra_embeds[i] = extra_embed

    input_embeds.index_copy_(
        0,
        _as_copy_index(mm_token_indices, input_embeds),
        mm_embed.to(dtype=input_embeds.dtype, device=input_embeds.device),
    )
    if extra_embeds is not None and len(extra_embeds) > 0:
        return None, cast(torch.FloatTensor, input_embeds), extra_embeds
    return None, cast(torch.FloatTensor, input_embeds)
