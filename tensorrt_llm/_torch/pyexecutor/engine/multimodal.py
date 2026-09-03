"""MM encoder item scheduling for decoder-family multimodal engines.

The second driving surface of a multimodal LLM: the executor encodes
scheduler-selected multimodal items through an LRU read-through cache under a
byte budget before each decode step. This is a capability facet, orthogonal to
the runner family axis -- it has no warmup or graph-capture phase.
"""

from collections.abc import Hashable
from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    InputProcessor,
    get_multimodal_encoder_item_metadata,
)
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy, TorchLlmArgs
from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest, MultimodalEncoderRequestError, _Unset


def resolve_mm_encoder_token_budget(base_budget: int, model_max_atomic_item_tokens: int) -> int:
    """Keep the model's largest indivisible MM item schedulable."""
    return max(base_budget, model_max_atomic_item_tokens)


def validate_mm_encoder_scheduling_compatibility(
    llm_args: TorchLlmArgs, item_scheduling_enabled: bool
) -> None:
    """Validate item-scheduling combinations after model capability is known."""
    policy = llm_args.multimodal_config.encoder_scheduling_policy
    if not item_scheduling_enabled:
        return
    if llm_args.multimodal_config.encoder_side_stream_max_ahead > 0:
        raise ValueError(
            "MM encoder item scheduling does not yet support side-stream "
            "prefetch (multimodal_config.encoder_side_stream_max_ahead > 0); "
            "set encoder_scheduling_policy=DISABLED or "
            "encoder_side_stream_max_ahead=0"
        )
    if policy != MultimodalEncoderSchedulingPolicy.EAGER:
        return
    if llm_args.enable_attention_dp:
        raise ValueError(
            "multimodal_config.encoder_scheduling_policy=EAGER does not yet "
            "support attention DP (enable_attention_dp=True)"
        )
    cache_transceiver_config = llm_args.cache_transceiver_config
    if cache_transceiver_config is not None and cache_transceiver_config.backend is not None:
        raise ValueError(
            "multimodal_config.encoder_scheduling_policy=EAGER does not yet "
            "support disaggregated serving (cache_transceiver_config)"
        )


def is_multimodal(model: nn.Module, input_processor: InputProcessor) -> bool:
    """True iff this engine drives a multimodal model.

    Primary signal: ``MultimodalModelMixin`` is the canonical marker --
    multimodal LM classes inherit from it. Until every model has migrated
    (Mistral done; Qwen-VL, Nemotron, Gemma, Phi-4-MM, etc. pending), fall back
    to whether the input processor subclasses ``BaseMultimodalInputProcessor``,
    which every multimodal model necessarily provides at the data boundary.

    TODO(TRTLLM-13542): Once all multimodal models inherit
    ``MultimodalModelMixin``, drop the input-processor fallback so the model
    class itself is the single source of truth.
    """
    if isinstance(model, MultimodalModelMixin):
        return True
    return isinstance(input_processor, BaseMultimodalInputProcessor)


def mm_item_scheduling_enabled(llm_args: TorchLlmArgs, model: nn.Module) -> bool:
    """Whether the item-scheduling wiring is engaged this run.

    Item scheduling is declared once, as a model capability (the ``MultimodalModelMixin``
    ClassVar). This is the actionable flag derived from it: the engine stores it because
    the executor and the scheduler wrap both read it back off the engine, and the setup
    below, the scheduler wrap and the executor encoder step must all agree. Both
    ``disable_mm_encoder`` and a ``DISABLED`` policy keep the capability but run only the
    base LLM scheduler.
    """
    mm_config = getattr(llm_args, "multimodal_config", None)
    policy = (
        mm_config.encoder_scheduling_policy
        if mm_config is not None
        else MultimodalEncoderSchedulingPolicy.DEFAULT
    )
    return (
        not llm_args.disable_mm_encoder
        and isinstance(model, MultimodalModelMixin)
        and model.supports_mm_encoder_item_scheduling
        and policy != MultimodalEncoderSchedulingPolicy.DISABLED
    )


def mm_encoder_cache_enabled(model: nn.Module) -> bool:
    """Whether the multimodal encoder cache is active for this model."""
    return isinstance(model, MultimodalModelMixin) and model.encoder_cache_active


def setup_mm_encoder_attn_metadata(
    model: nn.Module,
    input_processor: InputProcessor,
    encoder_max_num_tokens: int,
    attention_metadata_capacity: dict[str, int] | None,
) -> None:
    """Construct AttentionMetadata for any multimodal encoders inside the loaded model,
    using the engine's encoder token budget (``encoder_max_num_tokens``, falling back to
    the LLM-side ``max_num_tokens``).

    Mirrors ``_set_up_attn_metadata`` for the LLM backbone: encoders opt in by inheriting
    ``MultimodalEncoderMixin``, and the engine drives the construction so the sizes match
    the resolved encoder token budget rather than being hardcoded inside each encoder's
    ``__init__``. The optional per-segment capacity combines the encoder token budget with
    the input processor's largest supported item.

    Runs for every multimodal model, not just item-scheduled ones, so it is a module-level
    function rather than a ``MultimodalItemScheduler`` method: the scheduler is ``None``
    whenever item scheduling is off.
    """
    max_seq_len = encoder_max_num_tokens
    if isinstance(input_processor, BaseMultimodalDummyInputsBuilder):
        max_tokens_per_item = input_processor.get_mm_max_tokens_per_item()
        max_seq_len = max(max_seq_len, max(max_tokens_per_item.values(), default=0))

    for module in model.modules():
        if isinstance(module, MultimodalEncoderMixin):
            setup_kwargs: dict[str, Any] = dict(max_num_tokens=encoder_max_num_tokens)
            if attention_metadata_capacity is not None:
                setup_kwargs["attention_metadata_capacity"] = attention_metadata_capacity
            module.setup_attn_metadata(**setup_kwargs)
            module.set_attn_max_seq_len(max_seq_len)


def resolve_bytes_per_mm_encoder_embedding(model: MultimodalModelMixin) -> int:
    """Bytes occupied by one multimodal encoder output embedding.

    Prefers the mixin's explicit ``embedding_dim``/``embedding_dtype`` contract, then the
    text embedding layer's weight, then the pretrained config's hidden size with the
    loaded weights' dtype -- both mixin properties are optional and most VLMs implement
    neither.
    """
    try:
        return model.embedding_dim * torch.empty((), dtype=model.embedding_dtype).element_size()
    except NotImplementedError:
        pass
    try:
        weight = model.text_embedding_layer.weight
        return weight.shape[-1] * weight.element_size()
    except NotImplementedError:
        pass
    pretrained = model.model_config.pretrained_config
    hidden_size = getattr(pretrained, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(pretrained, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Cannot derive the MM embedding row size: the model "
            "implements neither embedding_dim/embedding_dtype nor "
            "text_embedding_layer, and its pretrained config exposes "
            "no (text_config.)hidden_size"
        )
    element_size = next(model.parameters()).dtype.itemsize
    return hidden_size * element_size


def resolve_mm_encoder_output_budget(
    input_processor: BaseMultimodalDummyInputsBuilder,
    encoder_max_num_tokens: int,
    model: MultimodalModelMixin,
) -> tuple[int, int]:
    """Resolve ``(output_budget_bytes, bytes_per_embedding)`` for resident MM encoder outputs.

    Encoder attention tokens and output embeddings use different units:
    ``encoder_max_num_tokens`` bounds pre-merge encoder attention tokens in one encoder
    iteration, while the input processor converts that runtime capacity into a
    model-specific upper bound on aggregate post-encoder embeddings. The byte budget is
    that embedding capacity multiplied by bytes per encoder embedding; the LLM-side
    ``max_num_tokens`` does not participate.

    It caps outputs held between encode and prefill and is reserved during KV-capacity
    estimation. It is separate from the optional reuse cache (``encoder_cache_max_bytes``).
    A request whose total embedding exceeds this budget is rejected at admission.

    The embedding capacity is validated before the model is consulted, so a processor that
    cannot report one raises regardless of what the model implements.
    """
    max_output_embeddings = input_processor.get_max_mm_encoder_output_embeddings(
        encoder_max_num_tokens
    )
    if max_output_embeddings is None or max_output_embeddings <= 0:
        raise ValueError(
            "A model with MM encoder item scheduling must implement "
            "get_max_mm_encoder_output_embeddings() and return a positive "
            "aggregate embedding capacity"
        )
    bytes_per_embedding = resolve_bytes_per_mm_encoder_embedding(model)
    return max_output_embeddings * bytes_per_embedding, bytes_per_embedding


class MultimodalItemScheduler:
    """Encodes scheduler-selected MM items through an optional read-through cache.

    Constructed once, at engine startup, and only when item scheduling is engaged. It
    holds no reference to the engine: everything it reads is passed in.

    It also carries the budgets ``create`` resolved, which the engine copies back onto
    itself because they are external contract (``_util.py`` and ``py_executor.py`` read
    them off the engine). ``encoder_max_num_tokens`` may exceed the value handed to
    ``create``: an atomic MM item cannot be split, so the budget is raised to the model's
    largest profiled item.

    The budget fields default to ``None`` so the item path can be exercised without a
    resolution pass; nothing outside ``create`` sets them.
    """

    def __init__(
        self,
        *,
        model: MultimodalModelMixin,
        input_processor: BaseMultimodalDummyInputsBuilder | None = None,
        attention_metadata_capacity: dict[str, int] | None = None,
        encoder_max_num_tokens: int | None = None,
        output_budget_bytes: int | None = None,
        bytes_per_embedding: int | None = None,
    ) -> None:
        self.model = model
        self.input_processor = input_processor
        self.attention_metadata_capacity = attention_metadata_capacity
        self.encoder_max_num_tokens = encoder_max_num_tokens
        self.output_budget_bytes = output_budget_bytes
        self.bytes_per_embedding = bytes_per_embedding

    @classmethod
    def maybe_create(
        cls,
        *,
        llm_args: TorchLlmArgs,
        model: nn.Module,
        input_processor: BaseMultimodalDummyInputsBuilder,
        encoder_max_num_tokens: int | None,
    ) -> "MultimodalItemScheduler | None":
        """Build the scheduler if this run engages item scheduling, else ``None``.

        The whole decision -- policy resolution, the capability predicate, the
        feature-combination validation and the budget resolution -- lives here rather than
        in the engine, which only records what comes back.

        The validation runs whether or not scheduling is engaged: it reads
        ``llm_args.multimodal_config`` before its own early return, so a run with no
        multimodal config raises here exactly as it did inline.
        """
        enabled = mm_item_scheduling_enabled(llm_args, model)
        validate_mm_encoder_scheduling_compatibility(llm_args, enabled)
        if not enabled:
            return None
        assert isinstance(model, MultimodalModelMixin)  # narrowed by the predicate
        return cls.create(
            model=model,
            input_processor=input_processor,
            encoder_max_num_tokens=encoder_max_num_tokens,
            configured_encoder_max_num_tokens=llm_args.encoder_max_num_tokens,
        )

    @classmethod
    def create(
        cls,
        *,
        model: MultimodalModelMixin,
        input_processor: BaseMultimodalDummyInputsBuilder,
        encoder_max_num_tokens: int | None,
        configured_encoder_max_num_tokens: int | None,
    ) -> "MultimodalItemScheduler":
        """Resolve the encoder budgets and build the scheduler.

        Item scheduling bounds four distinct MM encoder resources. They are owned in
        different places and measured in different units, so they are enumerated here once:

        * (A) encoder batch cardinality -- ``encoder_batch_size``, counting atomic MM items
          rather than model-internal attention sequences.
        * (B) encoder-forward workspace -- ``encoder_max_num_tokens``, in encoder attention
          tokens, clamped up to the largest atomic item; profiled by a direct full-budget
          encoder warmup.
        * (C) resident output bytes -- ``output_budget_bytes``, the maximum post-encoder
          embeddings produced by one legal encoder iteration, converted to bytes. Enforced
          by the scheduler; any capacity not materialized by warmup is reserved in
          KV-capacity estimation.
        * (D) reuse cache bytes -- ``encoder_cache_max_bytes`` on the mixin's
          ``TensorLRUCache``, self-bounded by LRU; unprofiled capacity is reserved on top
          of (C) for cache-enabled models.

        Prefill currently waits for every item in a request, so admission rejects a request
        whose complete MM embedding exceeds (C).
        """
        if encoder_max_num_tokens is None:
            raise ValueError(
                "MM encoder item scheduling requires a token budget; set "
                "encoder_max_num_tokens or max_num_tokens"
            )
        max_tokens_per_item = input_processor.get_mm_max_tokens_per_item()
        if not max_tokens_per_item:
            raise ValueError(
                "A model with MM encoder item scheduling must implement "
                "get_mm_max_tokens_per_item()"
            )
        if any(value <= 0 for value in max_tokens_per_item.values()):
            raise ValueError("get_mm_max_tokens_per_item() must return positive token counts")
        model_max_atomic_item_tokens = max(max_tokens_per_item.values())
        encoder_token_budget_base = encoder_max_num_tokens
        effective_encoder_token_budget = resolve_mm_encoder_token_budget(
            encoder_token_budget_base, model_max_atomic_item_tokens
        )
        if effective_encoder_token_budget > encoder_max_num_tokens:
            logger.warning_once(
                f"encoder_max_num_tokens={encoder_max_num_tokens} "
                "is smaller than the model's largest profiled atomic "
                f"multimodal item ({model_max_atomic_item_tokens}); "
                f"using {model_max_atomic_item_tokens} as the "
                "effective encoder runtime budget.",
                key="raise_encoder_max_num_tokens_for_atomic_item",
            )
            encoder_max_num_tokens = effective_encoder_token_budget

        attention_metadata_capacity = input_processor.get_mm_encoder_attention_metadata_capacity(
            encoder_max_num_tokens
        )
        if attention_metadata_capacity is not None:
            if not attention_metadata_capacity or any(
                value <= 0 for value in attention_metadata_capacity.values()
            ):
                raise ValueError(
                    "get_mm_encoder_attention_metadata_capacity() must "
                    "return nonempty positive capacities or None"
                )
        logger.info(
            "Multimodal encoder token budget: "
            f"configured={configured_encoder_max_num_tokens}, "
            f"base={encoder_token_budget_base}, "
            f"effective={encoder_max_num_tokens}, "
            f"model_atomic_max={model_max_atomic_item_tokens}, "
            "attention_capacity="
            f"{attention_metadata_capacity}."
        )
        output_budget_bytes, bytes_per_embedding = resolve_mm_encoder_output_budget(
            input_processor, encoder_max_num_tokens, model
        )

        return cls(
            model=model,
            input_processor=input_processor,
            attention_metadata_capacity=attention_metadata_capacity,
            encoder_max_num_tokens=encoder_max_num_tokens,
            output_budget_bytes=output_budget_bytes,
            bytes_per_embedding=bytes_per_embedding,
        )

    @property
    def encoder_cache(self) -> TensorLRUCache[Any] | None:
        """The encoder cache the item path reads through, or ``None``.

        Only models that opt into the encoder cache (``supports_encoder_cache`` with
        ``encoder_cache_max_bytes > 0``) participate -- the same lazily created
        ``TensorLRUCache`` instance the legacy inline path uses. Item-scheduled models
        without the flag (e.g. Qwen today) get ``None`` and never touch the cache;
        cross-request reuse for them is out of scope here.
        """
        model = self.model
        if not getattr(model, "supports_encoder_cache", False):
            return None
        getter = getattr(model, "_get_multimodal_encoder_cache", None)
        return getter() if getter is not None else None

    def item_keys(self, request: LlmRequest) -> list[Hashable] | None:
        """Return the request's per-item cache keys, or ``None``.

        ``None`` means the request cannot build stable content keys (no item metadata,
        content hashes, or processor-kwargs hash), so its items bypass the cache (always
        encoded; outputs live only on the request). The key format is shared with the
        legacy full-request path (``_encoder_cache_item_key``) so entries hit across both.

        The engine only builds this component when item scheduling is engaged, so there is
        no "is scheduling on" guard here.
        """
        # Memoized on the request's encoder state: the inputs are fixed at
        # admission, and a request whose items span several iterations would
        # otherwise rebuild the same keys on each one.
        state = request.py_mm_encoder_state
        if state is not None and not isinstance(state.cache_item_keys, _Unset):
            return state.cache_item_keys
        mm_data = request.py_multimodal_data
        try:
            item_metadata = get_multimodal_encoder_item_metadata(mm_data)
        except (TypeError, ValueError) as error:
            raise MultimodalEncoderRequestError(str(error)) from error
        keys = (
            None
            if item_metadata is None
            else (
                self.model.build_encoder_cache_item_keys(
                    request.multimodal_hashes,
                    item_metadata.item_refs,
                    item_metadata.output_embedding_lengths,
                    mm_data.get("mm_processor_kwargs_hash"),
                )
            )
        )
        if state is not None:
            state.cache_item_keys = keys
        return keys

    @torch.inference_mode()
    def forward_items(
        self,
        requests: list[LlmRequest],
        scheduled_items: dict[int, list[int]],
    ) -> None:
        """Forward selected MM encoder items and commit request-local outputs."""
        if not scheduled_items:
            return
        if not isinstance(self.model, MultimodalModelMixin):
            raise TypeError("Item-level MM scheduling requires MultimodalModelMixin")

        # Read-through against the model's encoder cache when enabled
        # (`supports_encoder_cache` + `encoder_cache_max_bytes > 0`): a hit
        # records a clone and skips the encode; a miss encodes, records, and
        # populates the cache. The hit/miss branch runs here -- at encode time,
        # on every rank against rank-local cache state -- so ranks perform
        # identical get/put sequences and stay in sync. When the cache is off
        # (`encoder_cache` is None), every item is a miss and outputs live only
        # on the request. Records take a clone regardless, so a recorded slot
        # never aliases a batch output or an evictable cache entry.
        encoder_cache = self.encoder_cache
        request_by_id = {request.request_id: request for request in requests}
        miss_items = []
        miss_owners: list[tuple[LlmRequest, int, Hashable | None]] = []
        touched_requests: dict[int, LlmRequest] = {}
        for request_id, item_indices in scheduled_items.items():
            request = request_by_id.get(request_id)
            if request is None:
                raise MultimodalEncoderRequestError(
                    f"Scheduled MM request {request_id} is no longer active"
                )
            state = request.py_mm_encoder_state
            if state is None:
                raise MultimodalEncoderRequestError(
                    f"Scheduled MM request {request_id} has no encoder item state"
                )
            touched_requests[request_id] = request
            multimodal_param = MultimodalParams(multimodal_data=request.py_multimodal_data)
            # Scope the lookup to this iteration's items: probing an item the
            # budget cannot encode yet would still refresh its LRU recency and
            # reorder eviction against items actually in flight.
            # Keys come from the request: the executor's params carry
            # `py_multimodal_data` only, while the content hashes live on the
            # `LlmRequest`.
            item_keys = self.item_keys(request) if encoder_cache is not None else None
            try:
                partition = (
                    self.model.partition_encoder_cache(
                        multimodal_param, encoder_cache, item_indices=item_indices, keys=item_keys
                    )
                    if item_keys is not None
                    else None
                )
            except MultimodalEncoderContractError as error:
                raise MultimodalEncoderRequestError(str(error)) from error
            if partition is None:
                # No cache, or the request cannot build stable content keys:
                # every scheduled item is a miss and its output lives only on
                # the request.
                for item_idx in item_indices:
                    miss_items.append((multimodal_param, item_idx))
                    miss_owners.append((request, item_idx, None))
                continue
            for item_idx, cached in partition.hits.items():
                state.record(item_idx, cached)
            for item_idx in partition.miss_indices:
                miss_items.append((multimodal_param, item_idx))
                miss_owners.append((request, item_idx, partition.keys[item_idx]))

        if miss_items:
            try:
                encoder_inputs = self.model.prepare_multimodal_encoder_inputs(miss_items)
            except MultimodalEncoderContractError as error:
                raise MultimodalEncoderRequestError(str(error)) from error
            for encoder_input, _, _ in encoder_inputs:
                encoder_input.to_device(
                    "multimodal_data",
                    "cuda",
                    pin_memory=prefer_pinned(),
                    target_keywords=getattr(self.model, "multimodal_data_device_paths", None),
                )

            try:
                outputs = self.model.forward_multimodal_encoder_items(encoder_inputs)
            except MultimodalEncoderContractError as error:
                raise MultimodalEncoderRequestError(str(error)) from error
            if len(outputs) != len(miss_owners):
                raise MultimodalEncoderRequestError(
                    "MM item encoder must return one output per item"
                )

            for output, (request, item_idx, key) in zip(outputs, miss_owners, strict=True):
                request.py_mm_encoder_state.record(item_idx, output)
                if encoder_cache is not None and key is not None:
                    encoder_cache.put(key, output)

        for request in touched_requests.values():
            request.py_mm_encoder_state.finalize(request.py_multimodal_data)
