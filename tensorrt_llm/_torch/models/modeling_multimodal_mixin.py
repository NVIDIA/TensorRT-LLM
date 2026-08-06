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
import copy
import itertools
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.inputs.registry import (
    MultimodalEncoderItemMetadata,
    get_multimodal_encoder_item_metadata,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .modeling_multimodal_utils import (
    _store_chunked_prefill_embeddings,
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)


class MultimodalEncoderContractError(ValueError):
    """A request-local MM encoder input or output contract violation."""


def _assemble_multimodal_encoder_embeddings(
    item_tensors: Dict[int, torch.Tensor], total_items: int
) -> torch.Tensor:
    """Copy prompt-ordered item embeddings into one contiguous tensor."""
    ordered = [item_tensors[i] for i in range(total_items)]
    first = ordered[0]
    if any(
        output.shape[1:] != first.shape[1:]
        or output.dtype != first.dtype
        or output.device != first.device
        for output in ordered[1:]
    ):
        raise ValueError(
            "MM encoder items for one request must have matching output shape, dtype, and device"
        )

    embeddings = torch.empty(
        (sum(output.shape[0] for output in ordered), *first.shape[1:]),
        dtype=first.dtype,
        device=first.device,
    )
    start = 0
    for output in ordered:
        end = start + output.shape[0]
        embeddings[start:end].copy_(output.detach())
        start = end
    return embeddings


@dataclass(frozen=True)
class EncoderGroup:
    """Modalities that share a single encoder call.

    Batching all items in a group into one encoder invocation amortizes
    fixed costs (kernel launches, dispatch) across items. The framework
    splits the output back per-modality and reorders it into prompt order
    via each request's `mm_item_order` manifest.

    Contract between `build_batched_input` and `encoder_fn`:

    * `build_batched_input` must concatenate items across requests in
      `modalities` order (all items for the first modality across requests,
      then all items for the second modality, etc.).
    * Within a modality, items must appear in the same per-request iteration
      order that `_lengths_by_modality` uses (i.e. the order of
      `multimodal_params` passed in).
    * `encoder_fn` must return one tensor whose rows correspond 1:1 to the
      input layout produced by `build_batched_input`, so the framework can
      split the output by `_lengths_by_modality` and reorder into prompt
      order via each request's `mm_item_order` manifest.
    """

    modalities: Tuple[str, ...]
    """Ordered modality names that share this encoder. Defines the row
    layout of the encoder output tensor: first all items of
    `modalities[0]`, then all items of `modalities[1]`, etc."""

    encoder_fn: Callable[..., torch.Tensor]
    """Encoder call invoked as `encoder_fn(**build_batched_input(params))`.
    Returns a single tensor with one row per embedding, laid out per the
    contract above."""

    build_batched_input: Callable[[List[MultimodalParams]], Dict[str, Any]]
    """Builds the kwargs dict passed to `encoder_fn`. Responsible for
    concatenating raw per-item tensors from `multimodal_data` across
    requests in the order described in the class docstring."""


def _lengths_by_modality(
    multimodal_params: List[MultimodalParams],
    modalities: Tuple[str, ...],
) -> Dict[str, List[int]]:
    """Invert prompt-ordered `multimodal_embedding_lengths` (number of
    embedding rows per item) into per-modality per-item lengths, matching
    the per-modality item order used by `EncoderGroup.build_batched_input`.

    Glossary used below:

    * **group** — one `EncoderGroup` entry (a single tuple of modalities
      that share an encoder call). E.g. Qwen3-VL registers one group
      `("image", "video")`; Nano registers three groups
      `("image",)`, `("video",)`, `("audio",)`.
    * **group-local modalities** — the `modalities` tuple of the group
      whose encoder is being invoked (this function's `modalities` arg).
    * **cross-group mix** — a single request whose items span modalities
      belonging to two *different* groups (e.g. one image + one audio
      item for Nano, where image and audio live in separate groups).
    * **multi-group model** — a model that registers more than one
      `EncoderGroup`, so its group-local `modalities` tuples are each
      strictly narrower than the full request-level modality set.
    """
    by_modality: Dict[str, List[int]] = {m: [] for m in modalities}
    for mp in multimodal_params:
        flat = mp.multimodal_data.get("multimodal_embedding_lengths") or []
        if mp.mm_item_order:
            for entry, length in zip(mp.mm_item_order, flat, strict=True):
                if entry["modality"] in by_modality:
                    by_modality[entry["modality"]].append(length)
            continue
        # Raw-prompt entrypoints (non chat-parsing) do not attach a manifest,
        # so this is the single enforcement point that a >1-modality request
        # must carry `mm_item_order` to make prompt-order reordering possible.
        # Check the *request-level* modality set so multi-group models (each
        # group with a single modality) still catch cross-group mixes — the
        # group-local `modalities` tuple can have length 1 and never trip.
        present = [m for m in _MM_DATA_INPUT_MODALITY_KEYS if mp.multimodal_data.get(m) is not None]
        if len(present) > 1:
            raise ValueError(
                "Request with multiple modalities present "
                f"({present}) must carry mm_item_order on MultimodalParams."
            )
        if present and present[0] in by_modality:
            by_modality[present[0]].extend(flat)
    return by_modality


def _reorder_embeds_by_manifest(
    multimodal_params: List[MultimodalParams],
    per_modality_embeds: Dict[str, torch.Tensor],
    per_modality_lengths: Dict[str, List[int]],
) -> torch.Tensor:
    """Slice per-modality tensors item-by-item and concat in prompt order."""
    per_modality_row_starts: Dict[str, List[int]] = {
        m: list(itertools.accumulate(lens, initial=0)) for m, lens in per_modality_lengths.items()
    }

    slices: List[torch.Tensor] = []
    # `entry["index"]` is per-request per-modality; advance a cursor to
    # translate it into a global item index within `per_modality_embeds`.
    per_modality_cursor: Dict[str, int] = {m: 0 for m in per_modality_embeds}
    for mp in multimodal_params:
        manifest = mp.mm_item_order or _synthesize_single_modality_manifest(
            mp, per_modality_embeds.keys()
        )
        req_counts: Dict[str, int] = {}
        for entry in manifest:
            m = entry["modality"]
            if m not in per_modality_embeds:
                continue
            i = per_modality_cursor[m] + entry["index"]
            starts = per_modality_row_starts[m]
            slices.append(per_modality_embeds[m][starts[i] : starts[i + 1]])
            req_counts[m] = req_counts.get(m, 0) + 1
        for m, c in req_counts.items():
            per_modality_cursor[m] += c
    if not slices:
        # No items resolved for any request. This happens on the executor's
        # KV-cache profiling pass: `_encode_dummy_inputs` runs the encoder on a
        # worst-case dummy batch that carries the encoder tensors but no
        # `multimodal_embedding_lengths`, so the per-modality lengths (and thus
        # the sliced `per_modality_embeds`) come back empty. The encoder forward
        # still ran (its activation is what peak-memory profiling captures), so
        # return a correctly-typed empty embedding tensor instead of crashing on
        # `torch.cat([])`. `per_modality_embeds` values are already zero-row
        # slices of the encoder output, so their concat preserves dtype/device
        # and the hidden dim.
        if per_modality_embeds:
            return torch.cat(list(per_modality_embeds.values()), dim=0)
        return torch.empty(0)
    return torch.cat(slices, dim=0)


def _synthesize_single_modality_manifest(
    mp: MultimodalParams,
    modalities: Iterable[str],
) -> List[Dict[str, Union[str, int]]]:
    """Trivial manifest for requests with only one modality present."""
    flat = mp.multimodal_data.get("multimodal_embedding_lengths") or []
    for m in modalities:
        if mp.multimodal_data.get(m) is not None:
            return [{"modality": m, "index": i} for i in range(len(flat))]
    return []


def encode_multimodal_by_groups(
    mm_encoder_groups: Sequence["EncoderGroup"],
    multimodal_params: List[MultimodalParams],
) -> torch.Tensor:
    """Run each group's encoder over its batched items and reorder into
    per-request prompt order.

    For each group present in the batch, one encoder call is issued over all
    items across all requests belonging to that group's modalities
    (arithmetic-intensity win). The output is split back per-modality using
    the prompt-ordered `multimodal_embedding_lengths` already stashed on
    `multimodal_data`, then reordered into each request's `mm_item_order`
    prompt sequence.

    Shared entry point for both the aggregated (`MultimodalModelMixin`) and
    mm-encoder-only (`Qwen3VisionModelBase.forward`) paths so the ordering
    contract lives in one place.
    """
    per_modality_embeds: Dict[str, torch.Tensor] = {}
    per_modality_lengths: Dict[str, List[int]] = {}
    for group in mm_encoder_groups:
        group_params = [
            mp
            for mp in multimodal_params
            if any(mp.multimodal_data.get(m) is not None for m in group.modalities)
        ]
        if not group_params:
            continue
        out = group.encoder_fn(**group.build_batched_input(group_params))
        lengths = _lengths_by_modality(group_params, group.modalities)
        cursor = 0
        for m in group.modalities:
            total = sum(lengths[m])
            per_modality_embeds[m] = out[cursor : cursor + total]
            cursor += total
        per_modality_lengths.update(lengths)
    return _reorder_embeds_by_manifest(multimodal_params, per_modality_embeds, per_modality_lengths)


def reorder_multimodal_embeddings_by_modality(
    multimodal_params: List[MultimodalParams],
    modalities: Tuple[str, ...],
    embeddings: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Reorder modality-grouped encoder outputs into request/prompt order."""
    if len(embeddings) != len(modalities):
        raise ValueError(
            f"Expected {len(modalities)} multimodal encoder outputs, got {len(embeddings)}."
        )
    if not all(
        isinstance(param.multimodal_data.get("multimodal_embedding_lengths"), list)
        for param in multimodal_params
    ):
        # Encoder memory profiling intentionally omits request layout metadata.
        return torch.cat(list(embeddings), dim=0)
    per_modality_embeds = dict(zip(modalities, embeddings, strict=True))
    per_modality_lengths = _lengths_by_modality(multimodal_params, modalities)
    return _reorder_embeds_by_manifest(
        multimodal_params,
        per_modality_embeds,
        per_modality_lengths,
    )


if TYPE_CHECKING:
    from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
    from ..pyexecutor.llm_request import LlmRequest


_MM_DATA_INPUT_MODALITY_KEYS = frozenset({"audio", "image", "video"})
_MM_AUX_STREAM: Optional[tuple[int, torch.cuda.Stream]] = None
_MM_ENCODER_CACHE_LOG_NAME = "mm_encoder_cache"


def _encoder_data_parallel_size(model_config: ModelConfig) -> int:
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return 1
    return multimodal_config.encoder_data_parallel_size


def make_multimodal_encoder_model_config(model_config: ModelConfig) -> ModelConfig:
    """Copy ``model_config`` and replicate encoder weights when MM DP is active.

    Attention DP already distributes requests between ranks. Explicit encoder
    data parallelism distributes encoder items over the ordinary TP group.
    Both modes need an unsharded encoder on every participating rank, while the
    language model keeps the original mapping.
    """
    encoder_config = copy.deepcopy(model_config)
    mapping = model_config.mapping
    encoder_dp_size = _encoder_data_parallel_size(model_config)

    if mapping.enable_attention_dp and encoder_dp_size > 1:
        raise ValueError(
            "Explicit multimodal encoder data parallelism cannot be combined "
            "with attention data parallelism. Hierarchical encoder DP is not supported."
        )

    replicate_encoder = mapping.enable_attention_dp or encoder_dp_size > 1
    if not replicate_encoder:
        return encoder_config

    if mapping.pp_size != 1 or mapping.cp_size != 1:
        raise NotImplementedError(
            "Multimodal encoder data parallelism currently requires pipeline "
            "parallel size 1 and context parallel size 1."
        )
    if not mapping.enable_attention_dp and encoder_dp_size != mapping.tp_size:
        raise ValueError(
            "multimodal_config.encoder_data_parallel_size must equal the tensor "
            f"parallel size ({mapping.tp_size}), got {encoder_dp_size}."
        )

    # Keep the process rank (and therefore local_rank) while making every TP
    # group a singleton. pp_size is used only to satisfy Mapping's world-size
    # invariant; multimodal encoders do not pipeline-partition their layers.
    replicated_mapping = Mapping(
        world_size=mapping.world_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=1,
        pp_size=mapping.world_size,
    )
    encoder_config._frozen = False
    encoder_config.mapping = replicated_mapping
    encoder_config._frozen = model_config._frozen
    return encoder_config


@dataclass(frozen=True)
class _EncoderDpWork:
    param_index: int
    item_index: Optional[int]
    global_row_start: int
    row_count: int


@dataclass(frozen=True)
class _EncoderDpPlacement:
    local_row_start: int
    global_row_start: int
    row_count: int


def _build_request_multimodal_input(
    request: "LlmRequest", cache_enabled: bool
) -> Optional[MultimodalInput]:
    """Build the encoder-cache key metadata carried by one request."""
    # Skip construction (and `from_components` validation) when no persistent cache consumes it.
    if not cache_enabled or request.multimodal_hashes is None:
        return None
    # `MultimodalModelMixin._encoder_cache_keys` uses UUID-aware multimodal hashes internally.
    # Although the UUIDs are not exposed as an attribute, they remain in the backing C++ request
    # for KV-cache block keys and cache events.
    return MultimodalInput.from_components(
        request.multimodal_hashes,
        request.multimodal_positions,
        request.multimodal_lengths,
        mm_item_run_cu_offsets=request.multimodal_item_run_cu_offsets,
        mm_run_positions=request.multimodal_run_positions,
        mm_run_lengths=request.multimodal_run_lengths,
    )


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


@dataclass(frozen=True)
class EncoderCachePartition:
    """Per-item cache partition for a single `MultimodalParams`.

    `hits` maps item index to its cached embedding row-block; `miss_indices` lists item
    indices that still require encoder work; `keys` is aligned to item order so miss
    embeddings can be written back after they are computed.

    `keys` always spans every item in the request -- full embedding assembly needs the total
    item count, and misses are written back by item index. `hits` and `miss_indices` instead
    cover only `looked_up`, the indices the caller asked about. The item scheduler passes the
    subset it picked for this iteration, so items it has no budget for yet are not touched: an
    LRU `get` refreshes recency, and probing an item the caller cannot encode would reorder
    eviction against items that are actually in flight.
    """

    hits: Dict[int, torch.Tensor]
    miss_indices: list[int]
    keys: list[Hashable]
    looked_up: list[int]

    @property
    def is_full_hit(self) -> bool:
        return bool(self.looked_up) and not self.miss_indices

    @property
    def is_full_miss(self) -> bool:
        return bool(self.looked_up) and not self.hits


class MultimodalModelMixin:
    """Template-method mixin for PyTorch multimodal causal LM models.

    Concrete model forwards can call `prepare_multimodal_inputs` while keeping their explicit
    language-model delegation.

    Current limitations:

    * For the time being, the persistent multimodal encoder cache stores per-item embeddings for
      single-modality `MultimodalParams` objects. Mixed-modality objects bypass the cache.
    * A partially cached `MultimodalParams` is handled by encoding only its miss items
      and interleaving cached items back in original per-item order. The default
      `build_multimodal_encoder_input` handles stacked-on-dim-0 and packed-with-grid-thw
      layouts; models with other layouts override that method.
    """

    supports_encoder_cache: ClassVar[bool] = False
    """Whether the model's production forward path uses the persistent encoder cache."""

    supports_mm_encoder_item_scheduling: ClassVar[bool] = False
    """Whether the model supports item-level MM encoder scheduling: it implements the item-encode
    forward and pairs with a processor that overrides
    `BaseMultimodalInputProcessor.get_mm_encoder_item_metadata`. This is the executor-layer gate for
    item scheduling."""

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

    # Per-model registration of encoder-batching groups. Each `EncoderGroup`
    # bundles a set of modalities that share one encoder call. Set as a class
    # attribute or on `self` in `__init__` (when `encoder_fn` binds to instance
    # methods). Consumers call the module-level `encode_multimodal_by_groups`
    # with these groups; both the aggregated and mm-encoder-only paths share
    # that helper so the ordering contract lives in one place.
    mm_encoder_groups: Sequence[EncoderGroup] = ()

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

    def prepare_multimodal_encoder_inputs(
        self,
        selected_items: Sequence[tuple[MultimodalParams, int]],
    ) -> list[tuple[MultimodalParams, list[int], str]]:
        """Build selected item encoder inputs before the caller performs H2D.

        Adjacent items from the same request and modality are sliced in one
        call. That is not just tidier: the packed-layout slicer splits the
        request's whole pixel payload and concatenates the chosen pieces, so
        slicing item-by-item re-splits the payload N times and copies each
        item separately.

        Args:
            selected_items: `(request params, item index)` pairs in scheduler-selected order.

        Returns:
            Tuples of `(sliced encoder params, per-item embedding row counts, modality)`, in
            input order. Flattening the row-count lists recovers the per-item sequence.
        """
        encoder_inputs: list[tuple[MultimodalParams, list[int], str]] = []
        for (
            multimodal_param,
            run_indices,
            modality,
            item_metadata,
        ) in self._runs_by_request_modality(selected_items):
            item_refs = item_metadata.item_refs
            # The two slicers take different index spaces. The raw-tensor
            # slicer indexes the modality's own payload, so it gets the
            # modality-local indices (plus the modality, which also lets it
            # slice out of an interleaved request); the parallel metadata is
            # prompt-ordered across modalities, so it gets the global ones.
            try:
                residual = self.build_multimodal_encoder_input(
                    multimodal_param,
                    [item_refs[i][1] for i in run_indices],
                    modality=modality,
                )
                self._apply_metadata_slice(residual, multimodal_param, run_indices)
            except MultimodalEncoderContractError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise MultimodalEncoderContractError(
                    f"Invalid multimodal encoder item input: {error}"
                ) from error
            encoder_inputs.append(
                (
                    residual,
                    [int(item_metadata.output_embedding_lengths[i]) for i in run_indices],
                    modality,
                )
            )
        return encoder_inputs

    @staticmethod
    def _runs_by_request_modality(
        selected_items: Sequence[tuple[MultimodalParams, int]],
    ) -> Iterator[tuple[MultimodalParams, list[int], str, "MultimodalEncoderItemMetadata"]]:
        """Split scheduler order into maximal same-request, same-modality runs.

        Only adjacent items merge, so the flattened result keeps the
        scheduler's order and outputs map back positionally. Each run also
        carries its request's item metadata: fetching it validates the whole
        record, so it is read once per request rather than once per item, and
        the caller reuses it instead of fetching again.
        """
        run_param: Optional[MultimodalParams] = None
        run_modality: Optional[str] = None
        run_metadata: Optional[MultimodalEncoderItemMetadata] = None
        run_indices: list[int] = []
        for multimodal_param, item_idx in selected_items:
            try:
                metadata = (
                    run_metadata
                    if multimodal_param is run_param
                    else get_multimodal_encoder_item_metadata(
                        multimodal_param.multimodal_data or {}
                    )
                )
            except (TypeError, ValueError) as error:
                raise MultimodalEncoderContractError(str(error)) from error
            if metadata is None:
                raise MultimodalEncoderContractError(
                    "MM item metadata is required for item encoding"
                )
            if item_idx < 0 or item_idx >= len(metadata.item_refs):
                raise MultimodalEncoderContractError(
                    f"MM item index {item_idx} is out of range for "
                    f"{len(metadata.item_refs)} item(s)"
                )
            modality = metadata.item_refs[item_idx][0]
            if run_indices and (multimodal_param is not run_param or modality != run_modality):
                yield run_param, run_indices, run_modality, run_metadata
                run_indices = []
            run_param, run_modality, run_metadata = multimodal_param, modality, metadata
            run_indices.append(item_idx)
        if run_indices:
            yield run_param, run_indices, run_modality, run_metadata

    def forward_multimodal_encoder_items(
        self,
        encoder_inputs: Sequence[tuple[MultimodalParams, list[int], str]],
    ) -> list[torch.Tensor]:
        """Forward prepared MM encoder inputs in scheduler item order.

        Args:
            encoder_inputs: Tuples returned by `prepare_multimodal_encoder_inputs`. Consecutive
                inputs with the same modality must be batch-compatible.

        Returns:
            One encoder output tensor **per item** (not per input tuple). Each tensor has the
            declared embedding row count and retains scheduler input order.
        """
        outputs: list[torch.Tensor] = []
        group_params: list[MultimodalParams] = []
        group_lengths: list[int] = []
        group_modality: Optional[str] = None

        def flush_group() -> None:
            if not group_params:
                return
            embeddings = self.encode_multimodal_inputs(group_params)
            expected_length = sum(group_lengths)
            if embeddings.shape[0] != expected_length:
                raise MultimodalEncoderContractError(
                    f"MM encoder output length {embeddings.shape[0]} does not "
                    f"match the {expected_length} rows declared by the "
                    "selected items"
                )
            outputs.extend(torch.split(embeddings, group_lengths, dim=0))
            group_params.clear()
            group_lengths.clear()

        for multimodal_param, embedding_lengths, modality in encoder_inputs:
            if group_modality is not None and modality != group_modality:
                flush_group()
            group_modality = modality
            group_params.append(multimodal_param)
            group_lengths.extend(embedding_lengths)
        flush_group()
        return outputs

    @property
    def multimodal_token_ids(self) -> Optional[Sequence[int] | torch.Tensor]:
        """Return placeholder token ids in `input_ids` replaced by MM embeds.

        These are sentinel token positions whose text embeddings are replaced
        by multimodal embeddings. Return `None` to use the out-of-vocabulary
        sentinel behavior in `fuse_input_embeds`.
        """
        return None

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

    @property
    def encoder_data_parallel_size(self) -> int:
        """Return the explicitly configured encoder DP size."""
        return _encoder_data_parallel_size(self.model_config)

    @property
    def encoder_data_parallel_active(self) -> bool:
        """Whether encoder items are distributed over the model's TP group."""
        mapping = self.model_config.mapping
        return not mapping.enable_attention_dp and self.encoder_data_parallel_size > 1

    def _get_encoder_dp_allreduce(self) -> AllReduce:
        allreduce = getattr(self, "_encoder_dp_allreduce", None)
        if allreduce is None:
            allreduce = AllReduce(
                self.model_config.mapping,
                strategy=AllReduceStrategy.NCCL,
                dtype=self.embedding_dtype,
            )
            self._encoder_dp_allreduce = allreduce
        return allreduce

    def _allreduce_encoder_dp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._get_encoder_dp_allreduce()(tensor)

    @staticmethod
    def _supports_encoder_item_partition(param: MultimodalParams) -> bool:
        """Return whether the common slicer understands this request layout."""
        modality = MultimodalModelMixin._encoder_cache_modality(param)
        if modality is None:
            return False
        modality_data = param.multimodal_data.get(modality)
        if not isinstance(modality_data, dict):
            return False
        embedding_lengths = param.multimodal_data.get("multimodal_embedding_lengths")
        if not isinstance(embedding_lengths, list):
            return False
        item_count = len(embedding_lengths)
        if modality in ("image", "video"):
            grid_key = f"{modality}_grid_thw"
            packed_pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
            pixel_values = modality_data.get("pixel_values")
            grids = modality_data.get(grid_key)
            return (
                isinstance(grids, torch.Tensor)
                and grids.shape[0] == item_count
                and packed_pixel_key in modality_data
            ) or (
                isinstance(pixel_values, torch.Tensor)
                and pixel_values.ndim >= 2
                and pixel_values.shape[0] == item_count
            )
        if modality != "audio":
            return False
        feature_key = "input_features" if "input_features" in modality_data else "audio_features"
        features = modality_data.get(feature_key)
        return isinstance(features, torch.Tensor) and features.shape[0] == item_count

    def _plan_encoder_dp_work(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> Optional[tuple[list[MultimodalParams], list[_EncoderDpPlacement], int]]:
        """Assign atomic encoder items to this TP rank.

        Returns ``None`` when row metadata is unavailable. That path is used by
        encoder memory profiling and deliberately executes the full dummy input
        on every rank.
        """
        mapping = self.model_config.mapping
        works: list[_EncoderDpWork] = []
        global_row_start = 0

        for param_index, param in enumerate(multimodal_params):
            raw_lengths = param.multimodal_data.get("multimodal_embedding_lengths")
            if not isinstance(raw_lengths, list) or not raw_lengths:
                return None
            lengths = [int(length) for length in raw_lengths]
            if any(length <= 0 for length in lengths):
                raise ValueError("multimodal_embedding_lengths must contain positive values.")

            if self._supports_encoder_item_partition(param):
                for item_index, row_count in enumerate(lengths):
                    works.append(
                        _EncoderDpWork(
                            param_index=param_index,
                            item_index=item_index,
                            global_row_start=global_row_start,
                            row_count=row_count,
                        )
                    )
                    global_row_start += row_count
            else:
                row_count = sum(lengths)
                works.append(
                    _EncoderDpWork(
                        param_index=param_index,
                        item_index=None,
                        global_row_start=global_row_start,
                        row_count=row_count,
                    )
                )
                global_row_start += row_count

        rank_loads = [0] * mapping.tp_size
        rank_works: list[list[_EncoderDpWork]] = [[] for _ in range(mapping.tp_size)]
        for work in sorted(works, key=lambda item: (-item.row_count, item.global_row_start)):
            target_rank = min(range(mapping.tp_size), key=lambda rank: (rank_loads[rank], rank))
            rank_works[target_rank].append(work)
            rank_loads[target_rank] += work.row_count

        selected = sorted(rank_works[mapping.tp_rank], key=lambda item: item.global_row_start)
        selected_by_param: dict[int, list[_EncoderDpWork]] = {}
        for work in selected:
            selected_by_param.setdefault(work.param_index, []).append(work)

        local_params: list[MultimodalParams] = []
        placements: list[_EncoderDpPlacement] = []
        local_row_start = 0
        for param_index, param in enumerate(multimodal_params):
            param_works = selected_by_param.get(param_index)
            if not param_works:
                continue

            if param_works[0].item_index is None:
                local_param = param
            else:
                item_indices = [work.item_index for work in param_works]
                local_param = self.build_multimodal_encoder_input(param, item_indices)
                self._apply_metadata_slice(local_param, param, item_indices)
            local_params.append(local_param)

            for work in param_works:
                placements.append(
                    _EncoderDpPlacement(
                        local_row_start=local_row_start,
                        global_row_start=work.global_row_start,
                        row_count=work.row_count,
                    )
                )
                local_row_start += work.row_count

        return local_params, placements, global_row_start

    def _run_multimodal_encoder(
        self,
        multimodal_params: Sequence[MultimodalParams],
        **encoder_kwargs: Any,
    ) -> torch.Tensor:
        """Run the model encoder with the configured multimodal DP behavior."""
        params = list(multimodal_params)
        mapping = self.model_config.mapping

        # Attention DP has already routed different requests to each rank.
        # Its encoder weights are replicated, so no inner partition or gather
        # is needed here.
        if mapping.enable_attention_dp or not self.encoder_data_parallel_active:
            return self.encode_multimodal_inputs(params, **encoder_kwargs)

        if self.encoder_data_parallel_size != mapping.tp_size:
            raise ValueError(
                "multimodal_config.encoder_data_parallel_size must equal the tensor "
                f"parallel size ({mapping.tp_size}), got {self.encoder_data_parallel_size}."
            )

        local_error: Optional[Exception] = None
        local_output: Optional[torch.Tensor] = None
        plan: Optional[tuple[list[MultimodalParams], list[_EncoderDpPlacement], int]] = None
        try:
            plan = self._plan_encoder_dp_work(params)
            if plan is None:
                # Memory profiling omits the row metadata needed for work
                # partitioning, so every rank measures the full dummy encoder.
                local_output = self.encode_multimodal_inputs(params, **encoder_kwargs)
            else:
                local_params, placements, _ = plan
                if local_params:
                    local_output = self.encode_multimodal_inputs(local_params, **encoder_kwargs)
                    expected_local_rows = sum(placement.row_count for placement in placements)
                    if local_output.shape[0] != expected_local_rows:
                        raise ValueError(
                            "Multimodal encoder returned an unexpected number of rows: "
                            f"expected {expected_local_rows}, got {local_output.shape[0]}."
                        )
                    if local_output.ndim != 2 or local_output.shape[1] != self.embedding_dim:
                        raise ValueError(
                            "Multimodal encoder output shape does not match the model embedding "
                            f"shape: expected (*, {self.embedding_dim}), "
                            f"got {tuple(local_output.shape)}."
                        )
        except Exception as error:
            # Every rank must reach the status collective; otherwise a local
            # preprocessing/encoder failure would strand its peers in the data
            # collective below.
            local_error = error

        embedding_weight = self.text_embedding_layer.weight
        error_flag = torch.tensor(
            [local_error is not None],
            dtype=torch.int32,
            device=embedding_weight.device,
        )
        any_error = self._allreduce_encoder_dp_tensor(error_flag)
        if bool(any_error.item()):
            if local_error is not None:
                raise RuntimeError("Multimodal encoder data-parallel rank failed.") from local_error
            raise RuntimeError("Multimodal encoder data-parallel peer rank failed.")

        if plan is None:
            assert local_output is not None
            return local_output
        _, placements, total_rows = plan

        output = torch.zeros(
            (total_rows, self.embedding_dim),
            dtype=self.embedding_dtype,
            device=embedding_weight.device,
        )
        if local_output is not None:
            for placement in placements:
                local_slice = slice(
                    placement.local_row_start,
                    placement.local_row_start + placement.row_count,
                )
                global_slice = slice(
                    placement.global_row_start,
                    placement.global_row_start + placement.row_count,
                )
                output[global_slice].copy_(local_output[local_slice])
        return self._allreduce_encoder_dp_tensor(output)

    @property
    def encoder_cache_active(self) -> bool:
        """Whether the persistent encoder cache is active for this model.

        Single source of truth shared by:

        * the in-iter consume path
        * the side-stream prefetch dispatch
        * the engine's cache-related gate
        * the KV-cache memory reservation.

        The cache is only active when the model opts in via `supports_encoder_cache` and configures
        a positive capacity.
        """
        if not self.supports_encoder_cache:
            return False
        multimodal_config = self.model_config.multimodal_config
        return multimodal_config is not None and multimodal_config.encoder_cache_max_bytes > 0

    def select_multimodal_params(
        self,
        multimodal_params: Sequence[MultimodalParams],
        num_context_requests: int,
    ) -> Sequence[MultimodalParams]:
        """Select the params that participate in multimodal encoder work.

        Returns the context-slice params with active multimodal content. Helpers below
        this method (`get_multimodal_embeddings`, `find_input_mm_embeds`,
        `fuse_input_embeds`) operate on the returned list and therefore see
        only `has_content()` params. Models overriding this hook must
        preserve that invariant.
        """
        return [
            param
            for param in list(multimodal_params)[:num_context_requests]
            if param.has_content()
            and (
                param.multimodal_runtime is None
                or param.multimodal_runtime.num_mm_tokens_in_chunk != 0
            )
        ]

    @property
    def language_model(self) -> torch.nn.Module:
        """Return the inner language model that receives prepared inputs."""
        raise NotImplementedError

    @property
    def vocab_size_padded(self) -> int:
        """Return the inner language model's padded vocabulary size."""
        return self.language_model.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        """Return the inner language model's maximum sequence length."""
        return self.language_model.infer_max_seq_len()

    def set_guided_decoder(self, guided_decoder: "CapturableGuidedDecoder") -> bool:
        """Install a guided decoder on the inner language model.

        Returns False when the inner model does not support guided decoding,
        matching the contract ModelEngine.set_guided_decoder() expects.
        """
        inner = self.language_model
        if not hasattr(inner, "set_guided_decoder"):
            return False
        return inner.set_guided_decoder(guided_decoder)

    def get_language_model_extra_forward_kwargs(
        self,
        *,
        raw_input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        mm_inputs: PreparedLlmInputs,
        **forward_kwargs: Any,
    ) -> dict[str, Any]:
        """Return model-specific arguments for the inner language-model forward."""
        return {}

    def get_language_model_forward_kwargs(
        self,
        *,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor],
        raw_input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        mm_inputs: PreparedLlmInputs,
        return_context_logits: bool,
        **forward_kwargs: Any,
    ) -> dict[str, Any]:
        """Build common and model-specific inner language-model forward arguments."""
        llm_kwargs = {
            "attn_metadata": attn_metadata,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "return_context_logits": return_context_logits,
        }
        llm_kwargs.update(
            self.get_language_model_extra_forward_kwargs(
                raw_input_ids=raw_input_ids,
                position_ids=position_ids,
                mm_inputs=mm_inputs,
                **forward_kwargs,
            )
        )
        return llm_kwargs

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prepare multimodal inputs and dispatch them to the language model."""
        multimodal_params = kwargs.pop("multimodal_params", [])
        mm_inputs = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            positions=position_ids,
            multimodal_params=multimodal_params,
            num_context_requests=attn_metadata.num_contexts,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        if inputs_embeds is not None:
            if mm_inputs.inputs_embeds is not None:
                raise ValueError(
                    "MultimodalModelMixin.forward received both caller-supplied inputs_embeds "
                    "and multimodal-derived inputs_embeds. These paths are mutually exclusive; "
                    "pass at most one."
                )
            mm_inputs = PreparedLlmInputs(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                extra_embeds=mm_inputs.extra_embeds,
            )

        llm_kwargs = self.get_language_model_forward_kwargs(
            attn_metadata=attn_metadata,
            input_ids=mm_inputs.input_ids,
            raw_input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=mm_inputs.inputs_embeds,
            mm_inputs=mm_inputs,
            return_context_logits=return_context_logits,
            multimodal_params=multimodal_params,
            num_generation_requests=attn_metadata.num_generations,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        return self.language_model.forward(**llm_kwargs)

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

    def build_multimodal_encoder_input(
        self,
        param: MultimodalParams,
        item_indices: Sequence[int],
        modality: Optional[str] = None,
    ) -> MultimodalParams:
        """Return a `MultimodalParams` whose raw modality inputs contain only
        `item_indices` from `param`, in that order.

        `item_indices` are indices into `modality`'s own payload. When `modality`
        is None it is inferred from `param`, which requires the request to hold a
        single modality -- the full-request path's case, where an item's global
        index and its index within its modality coincide. Callers that already
        know an item's modality (item scheduling reads it from `item_refs`) pass
        it explicitly along with the modality-local indices, which is what lets
        a single item be sliced out of an interleaved image+video request.

        Default handles three common single-modality layouts:

        - Image / video, stacked on dim 0 (Mistral 3 / Pixtral / Gemma 4):
          `pixel_values` with optional parallel `image_sizes`, position IDs, and sequence lengths.
        - Image / video, packed with `*_grid_thw` offsets (Qwen2-VL family):
          `pixel_values` `[total_patches, feat]` + `image_grid_thw` `[B, 3]`;
          prefix-summed patch counts locate each item's slice, and `image_grid_thw`
          is sliced in parallel.
        - Audio, stacked on dim 0 (Whisper / Qwen2-Audio / Gemma 4):
          `input_features` or `audio_features` sliced by item.

        Any additional sibling field in the modality dict whose first-axis length equals
        the item count is also sliced -- covers per-item metadata such as
        `second_per_grid_ts` (Qwen2.5-VL video) or `input_features_mask` /
        `feature_attention_mask` (audio) without model-specific code.

        Models with a different layout (e.g. mixed-modality per param, custom packed
        formats) should override this method. The parallel per-item metadata
        (`multimodal_embedding_lengths`, `multimodal_hashes`) is model-agnostic and is
        re-sliced by the mixin after this returns, so overrides need only handle the
        modality-specific raw data.
        """
        if modality is None:
            modality = self._encoder_cache_modality(param)
        if modality is None:
            raise NotImplementedError(
                "Default `build_multimodal_encoder_input` cannot infer the modality of a "
                "mixed-modality param. Pass `modality` with modality-local indices, or "
                "override for other layouts."
            )
        modality_data = param.multimodal_data[modality]
        if not isinstance(modality_data, dict):
            raise TypeError(
                f"multimodal_data[{modality!r}] must be a dict, got {type(modality_data).__name__}"
            )

        indices = list(item_indices)
        grid_key = {"image": "image_grid_thw", "video": "video_grid_thw"}.get(modality)
        pixel_key = {"image": "pixel_values", "video": "pixel_values_videos"}.get(modality)

        if (
            (grid_key and pixel_key)
            and (grid_key in modality_data)
            and (pixel_key in modality_data)
        ):
            # Packed layout: prefix-sum patch counts to locate each item's slab, then
            # concat the requested subset in item-index order.
            grids = modality_data[grid_key]
            n_items = grids.shape[0]
            embedding_lengths = param.multimodal_data.get("multimodal_embedding_lengths")
            if isinstance(embedding_lengths, list) and n_items != len(embedding_lengths):
                raise NotImplementedError(
                    f"Default `build_multimodal_encoder_input` cannot map {n_items} "
                    f"{modality} grids to {len(embedding_lengths)} items."
                )
            patch_counts = [int(c) for c in torch.prod(grids, dim=1).tolist()]
            row_starts = list(itertools.accumulate(patch_counts, initial=0))
            if indices == list(range(indices[0], indices[0] + len(indices))):
                # Contiguous run: the concatenation is just a row range, so take
                # a view instead of copying the payload. The common case for
                # both callers -- a scheduler picks items in order, and cache
                # misses cluster.
                pixel_slice = modality_data[pixel_key][
                    row_starts[indices[0]] : row_starts[indices[-1] + 1]
                ]
            else:
                per_item = torch.split(modality_data[pixel_key], patch_counts, dim=0)
                pixel_slice = torch.cat([per_item[i] for i in indices], dim=0)
            sliced = {
                pixel_key: pixel_slice,
                grid_key: grids[indices],
            }
        elif (
            modality in ("image", "video")
            and isinstance(modality_data.get("pixel_values"), torch.Tensor)
            and modality_data["pixel_values"].ndim >= 2
            and (
                not isinstance(param.multimodal_data.get("multimodal_embedding_lengths"), list)
                or modality_data["pixel_values"].shape[0]
                == len(param.multimodal_data["multimodal_embedding_lengths"])
            )
        ):
            # Stacked layout: dim-0 select from pixel values and parallel metadata.
            n_items = modality_data["pixel_values"].shape[0]
            miss_pixel = modality_data["pixel_values"][indices]
            # `pixel_values` was padded to the request-wide max H/W by the input
            # processor. After keeping only the miss subset, crop the trailing H/W back
            # down to that subset's own max true size -- otherwise a downstream re-batch
            # step (e.g. Mistral 3's `batch_pixel_values`) that pads to
            # `max(residual.image_sizes)` would compute a negative pad amount whenever
            # the omitted items were the largest in the original request.
            image_sizes = modality_data.get("image_sizes")
            if image_sizes is not None:
                miss_sizes = [image_sizes[i] for i in indices]
                if miss_sizes and miss_pixel.dim() >= 4:
                    max_h = max(int(size[0]) for size in miss_sizes)
                    max_w = max(int(size[1]) for size in miss_sizes)
                    miss_pixel = miss_pixel[..., :max_h, :max_w]
                sliced = {
                    "pixel_values": miss_pixel,
                    "image_sizes": miss_sizes,
                }
            else:
                sliced = {"pixel_values": miss_pixel}
        elif modality == "audio" and (
            "input_features" in modality_data or "audio_features" in modality_data
        ):
            # Stacked audio layout: slice the leading item dimension.
            # Per-item masks (`input_features_mask`, `feature_attention_mask`, ...)
            # are handled by the sibling-slice pass below.
            feature_key = (
                "input_features" if "input_features" in modality_data else "audio_features"
            )
            n_items = modality_data[feature_key].shape[0]
            embedding_lengths = param.multimodal_data.get("multimodal_embedding_lengths")
            if isinstance(embedding_lengths, list) and n_items != len(embedding_lengths):
                raise NotImplementedError(
                    f"Default `build_multimodal_encoder_input` cannot map {n_items} "
                    f"{modality} input rows to {len(embedding_lengths)} items."
                )
            sliced = {feature_key: modality_data[feature_key][indices]}
        else:
            raise NotImplementedError(
                f"Default `build_multimodal_encoder_input` cannot slice {modality} layout "
                f"with fields {sorted(modality_data)}; override this method."
            )

        # Sibling per-item fields (e.g. `second_per_grid_ts` on Qwen2.5-VL video)
        # must be sliced alongside the load-bearing keys above, or the residual
        # carries a shape-mismatched encoder input.
        sliced = {
            **modality_data,
            **sliced,
            **self._slice_per_item_sibling_fields(modality_data, n_items, indices, sliced.keys()),
        }

        # Shallow-copy `multimodal_input` so `_apply_metadata_slice` can rewrite
        # `multimodal_hashes` on the residual without mutating the source.
        residual_input = (
            copy.copy(param.multimodal_input) if param.multimodal_input is not None else None
        )
        # Carry the source's other entries through, but keep only the sliced
        # modality's raw payload: leaving a sibling modality's unsliced tensors
        # on the residual would make it look mixed-modality to the encoder
        # (`_lengths_by_modality` rejects that without an `mm_item_order`
        # manifest) while its rows were never requested.
        residual_data = {
            key: value
            for key, value in param.multimodal_data.items()
            if key not in _MM_DATA_INPUT_MODALITY_KEYS or key == modality
        }
        residual_data[modality] = sliced
        return MultimodalParams(
            multimodal_data=residual_data,
            multimodal_input=residual_input,
        )

    @staticmethod
    def _slice_per_item_sibling_fields(
        modality_data: Dict[str, Any],
        n_items: int,
        item_indices: Sequence[int],
        already_sliced: Iterable[str],
    ) -> Dict[str, Any]:
        """Slice modality-dict siblings whose first axis is parallel to items.

        Anything with `shape[0] == n_items` (tensor) or `len == n_items` (list) is
        assumed to be per-item metadata and sliced by `item_indices`. Fields already
        handled by the caller (`already_sliced`) and everything else pass through.
        """
        skip = set(already_sliced)
        sliced: Dict[str, Any] = {}
        for key, value in modality_data.items():
            if key in skip:
                continue
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == n_items:
                sliced[key] = value[item_indices]
            elif isinstance(value, list) and len(value) == n_items:
                sliced[key] = [value[i] for i in item_indices]
        return sliced

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

        During side-stream prefetch, this runs with the auxiliary stream current, so the H2D copies,
        the encoder, and every persistent-cache `put()` are issued on that stream. `TensorLRUCache`
        records each entry's producer event on the issuing (aux) stream; the next iteration's
        main-stream consumer waits on the request-level `encoder_event` for ordering.
        """
        encoder_cache = self._get_multimodal_encoder_cache()
        cache_misses: list[MultimodalParams] = []
        partial_hits: list[tuple[MultimodalParams, EncoderCachePartition]] = []
        if encoder_cache is not None:
            for param in multimodal_params:
                if param.multimodal_data.get("multimodal_embedding") is not None:
                    # A present embedding means either an earlier forward already wrote the
                    # persistent cache, or a prefetch hit attached a cache-owned tensor. Either
                    # way, skip re-lookup and re-write. `get_multimodal_embeddings` waits on the
                    # request event and records the attached tensor on the consuming stream before
                    # gathering it.
                    continue
                partition = self.partition_encoder_cache(param, encoder_cache)
                if partition is None or partition.is_full_miss:
                    cache_misses.append(param)
                    continue
                if partition.is_full_hit:
                    param.multimodal_data["multimodal_embedding"] = (
                        _assemble_multimodal_encoder_embeddings(partition.hits, len(partition.keys))
                    )
                    continue
                partial_hits.append((param, partition))

        if partial_hits:
            # `encoder_cache` is non-None here because partitions are only produced when the cache
            # exists.
            self._encode_with_partial_cache(partial_hits, encoder_cache)

        embeddings = get_multimodal_embeddings(
            encoder_forward_fn=self._run_multimodal_encoder,
            multimodal_params=list(multimodal_params),
        )
        if encoder_cache is not None:
            for param in cache_misses:
                self._write_encoder_cache_entries(param, encoder_cache)

        # Validate post-gather so cached-only paths (KV reuse, all-cached chunked prefill) are also
        # checked, not just paths that ran the encoder.
        self._validate_embeddings(embeddings, multimodal_params)
        return embeddings[0]

    def _get_multimodal_encoder_cache(self) -> Optional[TensorLRUCache]:
        """Return the per-model full-request-path encoder clone cache, if enabled.

        The cache stores per-item embeddings for params that can be represented by one modality.
        See `_encoder_cache_keys` for the mixed-modality skip path and its technical limitation.

        Scope: the single encoder cache instance for a cache-enabled model
        (`supports_encoder_cache`). The full-request (legacy inline-encode) consumers — side-stream
        prefetch, `mm_encoder_only`/disagg encoding — populate and read it inline; the
        item-scheduling path consumes the same instance read-through at encode time
        (`ModelEngine.forward_multimodal_encoder_items`). The key format is shared
        (`_encoder_cache_item_key`) so hits cross between paths. The item path's recorded outputs
        are cloned, so cache eviction never invalidates an in-flight request.
        """
        if not self.encoder_cache_active:
            logger.debug_once(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: disabled because the model does not opt in via "
                "supports_encoder_cache or multimodal_config.encoder_cache_max_bytes=0.",
                key="mm_encoder_cache_disabled",
            )
            return None

        multimodal_config = self.model_config.multimodal_config
        max_bytes = multimodal_config.encoder_cache_max_bytes
        if self._multimodal_encoder_cache is None:
            # Per-item embeddings are views produced by splitting a request-level encoder output.
            # Clone them so a cached item neither aliases mutable caller output nor retains the
            # entire batch allocation while cache accounting charges only its logical size. This
            # briefly needs source and clone memory during insertion, but preserves existing cache
            # entries when the copy cannot be allocated.
            self._multimodal_encoder_cache = TensorLRUCache(
                max_bytes,
                name=_MM_ENCODER_CACHE_LOG_NAME,
                cuda_stream_aware=multimodal_config.encoder_side_stream_max_ahead > 0,
            )
            try:
                embedding_dim = self.embedding_dim
                embedding_dtype = self.embedding_dtype
            except NotImplementedError:
                logger.info(
                    f"{_MM_ENCODER_CACHE_LOG_NAME}: created with max_bytes={max_bytes}, "
                    "embedding row capacity unavailable because the model does not implement "
                    "embedding_dim and embedding_dtype."
                )
            else:
                bytes_per_embedding_row = (
                    embedding_dim * torch.empty((), dtype=embedding_dtype).element_size()
                )
                max_embedding_rows = max_bytes // bytes_per_embedding_row
                logger.info(
                    f"{_MM_ENCODER_CACHE_LOG_NAME}: created with max_bytes={max_bytes}, "
                    f"max_embedding_rows={max_embedding_rows}, embedding_dim={embedding_dim}, "
                    f"embedding_dtype={embedding_dtype}"
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
        modalities = [key for key in _MM_DATA_INPUT_MODALITY_KEYS if key in mm_data]

        modality = mm_data.get("modality_type")
        if isinstance(modality, str):
            # Trust the explicit `modality_type` only when it agrees with the actual data keys.
            # Otherwise fall through to the mixed-modality skip so an inconsistent producer (e.g.
            # `modality_type="image"` while both image and audio data are present) cannot bypass the
            # safety check below and have the cache serve embeddings for the wrong modality.
            if modalities == [modality]:
                return modality

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
        """Build per-item encoder cache keys for `param`.

        The returned keys split one request's concatenated encoder output by
        `multimodal_embedding_lengths`.

        When the request carries atomic-item metadata, each key takes its modality from
        that item's `item_refs` entry, so mixed-modality requests are keyable per item.
        Without that metadata the whole param must resolve to one modality, since there
        is nothing else that says which item is which.
        """
        mm_input = param.multimodal_input
        mm_data = param.multimodal_data or {}
        if mm_input is None:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping params without multimodal hashes."
            )
            return None

        item_metadata = get_multimodal_encoder_item_metadata(mm_data)
        if item_metadata is not None:
            return cls.build_encoder_cache_item_keys(
                mm_input.multimodal_hashes,
                item_metadata.item_refs,
                item_metadata.output_embedding_lengths,
                mm_data.get("mm_processor_kwargs_hash"),
            )

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
            cls._encoder_cache_item_key(modality, item_hash, embedding_length, kwargs_hash)
            for item_hash, embedding_length in zip(
                mm_input.multimodal_hashes,
                embedding_lengths,
                strict=True,
            )
        ]

    @staticmethod
    def _encoder_cache_item_key(
        modality: str,
        item_hash: Sequence[int],
        embedding_length: int,
        kwargs_hash: str,
    ) -> Hashable:
        # Sole definition of the key format, so entries written by the
        # full-request path and the item-scheduling path hit from either.
        return (modality, tuple(item_hash), int(embedding_length), kwargs_hash)

    @classmethod
    def build_encoder_cache_item_keys(
        cls,
        multimodal_hashes: Optional[Sequence[Sequence[int]]],
        item_refs: Sequence[tuple[str, int]],
        embedding_lengths: Sequence[int],
        kwargs_hash: Optional[str],
    ) -> Optional[list[Hashable]]:
        """Build per-item cache keys from request-level item metadata.

        The modality comes from each item's `item_refs` entry, so mixed-modality requests are
        keyable per item. Returns `None` when the request cannot participate in the cache (missing
        hashes or kwargs hash, or item counts that do not line up).
        """
        if multimodal_hashes is None or kwargs_hash is None:
            return None
        if not (len(multimodal_hashes) == len(item_refs) == len(embedding_lengths)):
            # Malformed metadata rather than a normal miss: the request loses
            # cache participation silently, so say so once with the counts.
            logger.warning_once(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: skipping item keys because "
                f"multimodal_hashes ({len(multimodal_hashes)}), item_refs "
                f"({len(item_refs)}) and multimodal_embedding_lengths "
                f"({len(embedding_lengths)}) counts disagree",
                key="mm_encoder_cache_item_key_count_mismatch",
            )
            return None
        return [
            cls._encoder_cache_item_key(modality, item_hash, embedding_length, kwargs_hash)
            for (modality, _), item_hash, embedding_length in zip(
                item_refs,
                multimodal_hashes,
                embedding_lengths,
                strict=True,
            )
        ]

    @classmethod
    def partition_encoder_cache(
        cls,
        param: MultimodalParams,
        encoder_cache: TensorLRUCache,
        item_indices: Optional[Sequence[int]] = None,
        keys: Optional[Sequence[Hashable]] = None,
    ) -> Optional[EncoderCachePartition]:
        """Look up `param`'s items in the cache and return the per-item partition.

        Args:
            item_indices: restrict the lookup to these items. `None` looks up every item,
                which is what the full-request path wants. The item scheduler passes the
                subset it selected for this iteration so unscheduled items keep their LRU
                recency -- see `EncoderCachePartition`.
            keys: precomputed per-item keys. `None` derives them from `param`, which needs
                `multimodal_input` to carry the content hashes. The executor builds params
                from `LlmRequest.py_multimodal_data` alone -- the hashes live on the
                request, not the params -- so it derives keys there and passes them in.

        Returns `None` when the param is not keyable (missing item metadata, content
        hashes or processor-kwargs hash, or a request-local embedding is already
        attached); the caller should treat that as a full miss.
        """
        if param.multimodal_data.get("multimodal_embedding") is not None:
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: request-local multimodal embedding present; "
                "skipping persistent cache lookup"
            )
            return None

        keys = list(keys) if keys is not None else cls._encoder_cache_keys(param)
        if not keys:
            return None

        looked_up = list(range(len(keys))) if item_indices is None else list(item_indices)
        out_of_range = [i for i in looked_up if not 0 <= i < len(keys)]
        if out_of_range:
            raise MultimodalEncoderContractError(
                f"item_indices {out_of_range} out of range for {len(keys)} cache keys"
            )

        hits: Dict[int, torch.Tensor] = {}
        miss_indices: list[int] = []
        for i in looked_up:
            cached = encoder_cache.get(keys[i])
            if cached is None:
                miss_indices.append(i)
            else:
                hits[i] = cached

        logger.debug(
            f"{_MM_ENCODER_CACHE_LOG_NAME}: partition hit_items={len(hits)}, "
            f"miss_items={len(miss_indices)}, looked_up={len(looked_up)}, "
            f"total_items={len(keys)}."
        )
        return EncoderCachePartition(
            hits=hits, miss_indices=miss_indices, keys=keys, looked_up=looked_up
        )

    @staticmethod
    def _apply_metadata_slice(
        residual: MultimodalParams,
        source: MultimodalParams,
        item_indices: Sequence[int],
    ) -> None:
        """Overwrite `residual`'s per-item metadata to match the sliced items.

        Models slice raw modality tensors in `build_multimodal_encoder_input`; the mixin owns
        the parallel per-item metadata slice so every model gets it identically.
        """
        source_lengths = source.multimodal_data["multimodal_embedding_lengths"]
        residual.multimodal_data["multimodal_embedding_lengths"] = [
            source_lengths[i] for i in item_indices
        ]
        if residual.multimodal_input is not None and source.multimodal_input is not None:
            source_hashes = source.multimodal_input.multimodal_hashes
            residual.multimodal_input.multimodal_hashes = [source_hashes[i] for i in item_indices]

    def _encode_with_partial_cache(
        self,
        partials: Sequence[tuple[MultimodalParams, EncoderCachePartition]],
        encoder_cache: TensorLRUCache,
    ) -> None:
        """Encode only the miss items of each partial-hit param and stitch results.

        Miss residuals from all partial-hit params in the batch are encoded in a
        single call and the concatenated output is split back per param, mirroring
        how `get_multimodal_embeddings` batches full-miss params. After this returns,
        each param's `multimodal_embedding` has the same shape as a full encoder run
        so downstream `get_multimodal_embeddings` treats it as fully cached.
        """
        if not partials:
            return

        # Cross-iter prefetch may have staged some params' raw MM tensors on the aux
        # stream. If a prefetch encoder call then raised, the request reaches this
        # iteration with an `encoder_event` but no `multimodal_embedding`; slicing
        # those tensors on the main stream before the event would race the aux-stream
        # H2D copy. Wait per param up front, before any raw-tensor read.
        for param, _ in partials:
            if param.encoder_event is not None:
                torch.cuda.current_stream().wait_event(param.encoder_event)

        # Build every residual, then run one batched encoder call over the whole set.
        residuals: list[MultimodalParams] = []
        per_param_miss_lengths: list[list[int]] = []
        for param, partition in partials:
            residual = self.build_multimodal_encoder_input(param, partition.miss_indices)
            self._apply_metadata_slice(residual, param, partition.miss_indices)
            residuals.append(residual)
            per_param_miss_lengths.append(
                [
                    param.multimodal_data["multimodal_embedding_lengths"][i]
                    for i in partition.miss_indices
                ]
            )

        batched_output = self._run_multimodal_encoder(residuals)
        per_param_slabs = torch.split(
            batched_output, [sum(lengths) for lengths in per_param_miss_lengths], dim=0
        )

        for (param, partition), slab, miss_lengths in zip(
            partials, per_param_slabs, per_param_miss_lengths, strict=True
        ):
            miss_tensors = torch.split(slab, miss_lengths, dim=0)

            by_item: Dict[int, torch.Tensor] = dict(partition.hits)
            for miss_idx, tensor in zip(partition.miss_indices, miss_tensors, strict=True):
                by_item[miss_idx] = tensor
            param.multimodal_data["multimodal_embedding"] = _assemble_multimodal_encoder_embeddings(
                by_item, len(partition.keys)
            )

            inserted = 0
            rejected = 0
            for miss_idx, tensor in zip(partition.miss_indices, miss_tensors, strict=True):
                if encoder_cache.put(partition.keys[miss_idx], tensor):
                    inserted += 1
                else:
                    rejected += 1
            logger.debug(
                f"{_MM_ENCODER_CACHE_LOG_NAME}: partial-hit encode "
                f"total_items={len(partition.keys)} "
                f"hit_items={len(partition.hits)} "
                f"encoded_items={len(partition.miss_indices)} "
                f"cache_writes_inserted={inserted} "
                f"cache_writes_rejected={rejected}"
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
    encoder_cache_enabled = model.encoder_cache_active
    params_list = [
        MultimodalParams(
            multimodal_input=_build_request_multimodal_input(req, encoder_cache_enabled),
            multimodal_data=mm_data,
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=0,
                chunk_end_pos=cumsum.numel(),
                embed_mask_cumsum=cumsum,
            ),
            mm_item_order=req.py_mm_item_order,
        )
        for req, mm_data, cumsum in candidates
    ]

    # Prefetch targets requests outside the current iteration, so their
    # multimodal tensors are not touched by the main stream. The caller queues
    # this after the iteration's LLM kernels so aux-stream H2D copies and
    # encoder work can overlap them.
    #
    # Request-local ordering is handled by `encoder_event`; the consume path also `record_stream`s
    # attached embeddings before gathering them so post-prefill request cleanup cannot release
    # storage while main-stream work is pending. Persistent-cache clones use their own producer
    # events and consumer `record_stream` calls inside `TensorLRUCache`.
    encoder_event = None
    try:
        with _run_on_aux_stream(aux_stream) as encoder_event:
            encoder_cache = model._get_multimodal_encoder_cache() if encoder_cache_enabled else None
            cache_misses: list[MultimodalParams] = []
            partial_hits: list[tuple[MultimodalParams, EncoderCachePartition]] = []
            if encoder_cache is None:
                cache_misses = params_list
            else:
                for param in params_list:
                    partition = model.partition_encoder_cache(param, encoder_cache)
                    if partition is None or partition.is_full_miss:
                        cache_misses.append(param)
                    elif partition.is_full_hit:
                        param.multimodal_data["multimodal_embedding"] = (
                            _assemble_multimodal_encoder_embeddings(
                                partition.hits, len(partition.keys)
                            )
                        )
                    else:
                        partial_hits.append((param, partition))

            params_to_transfer = cache_misses + [param for param, _ in partial_hits]
            for param in params_to_transfer:
                param.to_device(
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

            if partial_hits and encoder_cache is not None:
                model._encode_with_partial_cache(partial_hits, encoder_cache)
            if cache_misses:
                encoder_output = model._run_multimodal_encoder(cache_misses)
                _store_chunked_prefill_embeddings(cache_misses, [encoder_output])
                if encoder_cache is not None:
                    for param in cache_misses:
                        model._write_encoder_cache_entries(param, encoder_cache)

            # Prefetch only needs to attach each request's embedding. Validate each request
            # independently instead of gathering the unused batch output with `torch.cat`.
            for param in params_list:
                embedding = param.multimodal_data.get("multimodal_embedding")
                if not isinstance(embedding, torch.Tensor):
                    raise ValueError("Multimodal encoder prefetch did not produce an embedding.")
                model._validate_embeddings([embedding], [param])
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
