# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampler orchestration: the entry points the executor drives each step.

Holds the :class:`Sampler` ABC and its ``SampleState`` types, the trivial
early-stop samplers (:class:`EarlyStopSampler` for non-generation models,
:class:`EarlyStopWithMMResult` for the multimodal-encoder-only engine), and
:class:`TorchSampler` -- the PyTorch sampling path. ``TorchSampler`` owns no
sampling logic of its own beyond batching and orchestration: each feature
(beam search, penalties, token bans, top-p decay, finish reasons, log-probs,
two-model speculation, seeds) lives in its own module and is held here as a
handler, driven through ``setup_sampler_step`` / ``sample_async`` /
``update_requests``.
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    cast,
)

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.bindings.internal.batch_manager import add_new_tokens_to_requests
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

from ...utils import torch_multi_arange
from ..llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from ..resource_manager import ResourceManager, ResourceManagerType
from ..scheduler import ScheduledRequests
from .beam_search import BeamHistoryBuilder, BeamSearchHandler, finalize_beam, prepare_beam_search
from .finish_reasons import FinishReasonsHandler
from .greedy_sample_kernels import supports_greedy_argmax_scatter
from .greedy_tail_graph import GreedyTailGraph
from .logprobs import LogProbsHandler, LogProbsState, LogProbsStateList, LogProbsStore
from .penalties import PenaltyHandler, has_occurrence_penalty
from .sampler_common import (
    DEFAULT_BEAM_IDX,
    DEFAULT_STEP_IDX,
    FinishReasonsList,
    _BatchedSamplingResult,
    _get_beam_width_out,
    _request_get_sampling_params,
    add_token,
    int_tensor,
)
from .sampler_features import (
    AsyncWorkerMixin,
    SamplerEvent,
    _PackedStepIndexer,
    _UnpackedStepIndexer,
    apply_d2t,
    apply_embedding_bias,
    check_stop_words_length,
    fast_greedy_sample_kernel,
)
from .sampler_strategy import (
    GREEDY,
    BeamHistory,
    BeamSearchMetadata,
    BeamSearchStore,
    FlashInferGroupedStrategySampler,
    GenericStrategyKeyType,
    RequestGroupKey,
    RequestGroupValue,
    RequestGroupValueWithMetadata,
    RequestSeeds,
    StrategyMetadata,
    TopPDecayMetadata,
    _CachingRequestGrouper,
    _request_strategy,
)
from .seed_manager import _SeedManager
from .token_ban import (
    OverlappedTokenBanHandler,
    SynchronousTokenBanHandler,
    TokenBanHandler,
    has_min_length,
)
from .top_p_decay import TopPDecayHandler
from .two_model_spec_dec import (
    TwoModelSpecDecHandler,
    get_rejected_indices,  # noqa: F401  (re-exported for tests via the package __getattr__)
    sample_rejected,  # noqa: F401
)

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    # Type-only: importing the speculative package at module level would
    # re-create the import cycle sampler.sampler -> speculative ->
    # (draft_target/mtp) -> pyexecutor.sampler that this package's lazy
    # __init__ exists to avoid. The cycle only resolves when speculative is
    # imported first; a process whose first touch is sampler.sampler (e.g. a
    # test module, with the top-level package now lazy) would break.
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

T = TypeVar("T")


@dataclass(kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor
    log_probs: torch.Tensor | None = None


GenericSampleStateTensorsHost = TypeVar("GenericSampleStateTensorsHost", bound=SampleStateTensors)
GenericSampleStateTensorsDevice = TypeVar(
    "GenericSampleStateTensorsDevice", bound=SampleStateTensors
)


@dataclass(kw_only=True)
class SampleState(Generic[GenericSampleStateTensorsHost, GenericSampleStateTensorsDevice]):
    requests: list[LlmRequest]
    device: Optional[GenericSampleStateTensorsDevice] = None
    host: Optional[GenericSampleStateTensorsHost] = None
    sampler_event: Optional[SamplerEvent] = None
    runtime_draft_len: Optional[int] = None


# Generic bounds not supported, https://github.com/python/typing/issues/548
GenericSampleState = TypeVar("GenericSampleState", bound=SampleState)  # type: ignore


class Sampler(ABC, Generic[GenericSampleState]):
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests) -> None:
        pass

    def get_cache_indirection(self) -> torch.Tensor | None:
        return None

    @abstractmethod
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> GenericSampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(
        self,
        state: GenericSampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def beam_width(requests: Iterable[LlmRequest]) -> int:
        for req in requests:
            return req.py_beam_width
        return 0

    @abstractmethod
    def is_generation_model(self) -> bool:
        raise NotImplementedError

    def validate_request(self, request: LlmRequest) -> None:
        """Validate that the request can be processed by the sampler.

        If the request is not supported by the sampler, this should raise an
        appropriate exception.

        Args:
            request: The request to validate

        Returns:
            None if request is valid

        Raises:
            Appropriate exception if request is not supported by sampler.
        """

    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        """Check if sampler wants to receive draft token probabilities."""
        return True  # conservative default


class EarlyStopSampler(Sampler[SampleState[SampleStateTensors, SampleStateTensors]]):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    SampleState: TypeAlias = SampleState[SampleStateTensors, SampleStateTensors]

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        host = SampleStateTensors(new_tokens=torch.empty(0))
        assert not scheduled_requests.generation_requests
        return self.SampleState(requests=scheduled_requests.context_requests, host=host)

    @override
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        assert isinstance(state, SampleState)
        requests = state.requests
        for idx, request in enumerate(requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

    @override
    def is_generation_model(self) -> bool:
        return False


@dataclass(kw_only=True)
class MultimodalResult:
    mm_embeddings: List[torch.Tensor]
    # needed to torch.split the mm_embeddings into item-wise chunks
    mm_embedding_lengths: List[List[int]]
    # needed when requests mix text-only and multimodal ones
    mm_embedding_request_indices: List[int]
    # number of context requests in the batch
    num_context_requests: int
    # Can be used to include e.g. `mrope_position_ids`, etc.
    extra_data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        num_embeddings = len(self.mm_embeddings)
        num_lengths = len(self.mm_embedding_lengths)
        if num_lengths != num_embeddings:
            raise ValueError(
                "mm_embedding_lengths batch size does not match mm_embeddings: "
                f"{num_lengths} != {num_embeddings}"
            )
        num_request_indices = len(self.mm_embedding_request_indices)
        if num_request_indices != num_embeddings:
            raise ValueError(
                "mm_embedding_request_indices batch size does not match "
                f"mm_embeddings: {num_request_indices} != {num_embeddings}"
            )
        for result_index, (mm_embedding, mm_embedding_lengths) in enumerate(
            zip(self.mm_embeddings, self.mm_embedding_lengths, strict=True)
        ):
            actual_rows = len(mm_embedding)
            expected_rows = sum(mm_embedding_lengths)
            if actual_rows != expected_rows:
                raise ValueError(
                    f"mm_embedding shape mismatch for result {result_index}: "
                    f"{actual_rows} != {expected_rows}"
                )
        for request_index in self.mm_embedding_request_indices:
            if request_index < 0 or request_index >= self.num_context_requests:
                raise ValueError(
                    "mm_embedding_request_indices contains an invalid request "
                    f"index: {request_index} not in [0, {self.num_context_requests})"
                )

    @classmethod
    def from_model_outputs(
        cls, model_outputs: Dict[str, Any], num_context_requests: int
    ) -> "MultimodalResult":
        result_keys = {
            "mm_embeddings",
            "mm_embedding_lengths",
            "mm_embedding_request_indices",
        }
        return cls(
            mm_embeddings=model_outputs["mm_embeddings"],
            mm_embedding_lengths=model_outputs["mm_embedding_lengths"],
            mm_embedding_request_indices=model_outputs["mm_embedding_request_indices"],
            num_context_requests=num_context_requests,
            extra_data={
                key: value for key, value in model_outputs.items() if key not in result_keys
            },
        )


@dataclass(kw_only=True)
class SampleStateWithMMResult(SampleState[SampleStateTensors, SampleStateTensors]):
    data: MultimodalResult


class EarlyStopWithMMResult(Sampler[SampleStateWithMMResult]):
    """
    Use for skipping decoding step for non generation model, and return the batch_output (such as mm_embeddings)
    """

    SampleState: TypeAlias = SampleStateWithMMResult

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        # from model_outputs to MultimodalResult
        assert not scheduled_requests.generation_requests
        data = MultimodalResult.from_model_outputs(
            model_outputs, scheduled_requests.num_context_requests
        )
        return self.SampleState(requests=scheduled_requests.context_requests, data=data)

    @override
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        # resource_manager will not be used in this function, just for interface consistency.
        assert isinstance(state, SampleState)
        requests = state.requests
        mm_embeddings = state.data.mm_embeddings
        extra_data = state.data.extra_data or {}
        mrope_position_ids = extra_data.get("mrope_position_ids", None)
        mrope_position_deltas = extra_data.get("mrope_position_deltas", None)
        for request in requests:
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

        request_indices = state.data.mm_embedding_request_indices
        for result_index, (request_index, mm_embedding) in enumerate(
            zip(request_indices, mm_embeddings, strict=True)
        ):
            request = requests[request_index]
            mm_embedding_lengths = state.data.mm_embedding_lengths[result_index]

            request.py_result.append_mm_embeddings(mm_embedding, mm_embedding_lengths)

            # Store mrope data if available
            if mrope_position_ids is not None and mrope_position_deltas is not None:
                mrope_index = (
                    request_index if len(mrope_position_ids) == len(requests) else result_index
                )
                request.py_result.set_mrope_position(
                    mrope_position_ids[mrope_index],
                    mrope_position_deltas[mrope_index],
                )

    @override
    def is_generation_model(self) -> bool:
        return False


@dataclass(kw_only=True)
class SamplingRequestsMetadata:
    """Metadata for the sampling requests."""

    req_num_generated_tokens: torch.Tensor
    """The number of generated tokens for each sampling request.
    In beam search, this uses the incoming beam width."""
    req_num_generated_tokens_output: torch.Tensor
    """The number of generated tokens for each sampling request.
    In beam search, this uses the outgoing beam width."""
    req_num_beams: torch.Tensor
    """The number of beams for each sampling request."""
    req_num_steps: torch.Tensor
    """The number of generation steps for each sampling request."""
    req_offsets: torch.Tensor
    """The start offsets of the sampling requests in the raw logits."""


@dataclass(kw_only=True)
class SampleStateTensorsHostTorch(SampleStateTensors):
    finish_reasons: torch.Tensor | None
    first_finish_reasons: torch.Tensor | None
    logprobs_state: LogProbsState | None = None
    single_step_greedy: bool = False
    """Whether `new_tokens` uses the compact `(num_requests,)` layout instead of
    `[step, slot, beam]`. Describes these host tensors, so it must live here rather
    than on `SampleStateTorch`: under pipeline parallelism only this object crosses
    the ring hand-off, and a receiving rank would otherwise pair the compact buffer
    with an outer flag left at its default."""

    def finish_reasons_list(self) -> FinishReasonsList:
        """`(num_seq_slots, num_steps)`"""
        # step, slot, beam => slot, step, beam
        finish_reasons = self.finish_reasons
        if finish_reasons is None:
            return []
        else:
            return finish_reasons.permute(1, 0, 2).tolist()


@dataclass(kw_only=True)
class SampleStateTorch(SampleState[SampleStateTensorsHostTorch, SampleStateTensors]):
    beam_history_builders: list[BeamHistoryBuilder | None] | None = None
    greedy_tail_slot: int | None = None
    """Ring slot backing `host.new_tokens` when the greedy tail was replayed
    from a graph. Returned to the ring once this state has been consumed; a
    slot that is never returned simply leaves the ring one buffer shorter."""


class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
    DEFAULT_MAX_STOP_WORD_LENGTH: Final[int] = 20
    DEFAULT_MAX_STOP_WORDS: Final[int] = 10

    SampleState = SampleStateTorch

    @override
    def get_cache_indirection(self) -> torch.Tensor | None:
        if (beam_search_store := self.store.beam_search_store) is not None:
            return beam_search_store.cache_indirection
        return None

    @override
    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    @dataclass(kw_only=True)
    class Store:
        new_tokens: torch.Tensor
        """Device tensor containing latest sampled tokens.

        Shape: ``NEW_TOKENS_SHAPE`` -- (max_tokens, max_num_sequences, max_beam_width).
        """
        beam_search_store: "BeamSearchStore | None" = None
        """Holds data related to beam search."""
        log_probs_store: LogProbsStore
        """Holds data related to log-probs handling."""

    def _create_store(self) -> Store:
        # Tensors necessary for all sampling methods
        new_tokens = int_tensor(self.NEW_TOKENS_SHAPE)

        # Only used for logprobs processing or beam search
        sampled_log_probs = torch.empty(self.LOGPROBS_SHAPE, device="cuda", dtype=torch.float32)
        # Only used for logprobs processing
        sampled_log_prob_indices = torch.empty(
            self.LOGPROBS_SHAPE, device="cuda", dtype=torch.int32
        )
        sampled_log_prob_ranks = torch.empty(self.LOGPROBS_SHAPE, device="cuda", dtype=torch.int32)
        # These are 0 sized tensors, if topk-logprobs are not used
        topk_indices = torch.empty(
            self._log_probs.TOPK_LOGPROBS_SHAPE, device="cuda", dtype=torch.int32
        )
        topk_vals = torch.empty(
            self._log_probs.TOPK_LOGPROBS_SHAPE, device="cuda", dtype=torch.float32
        )
        log_probs_store = LogProbsStore(
            sampled_log_prob_indices=sampled_log_prob_indices,
            sampled_log_probs=sampled_log_probs,
            sampled_log_prob_ranks=sampled_log_prob_ranks,
            topk_indices=topk_indices,
            topk_vals=topk_vals,
        )

        beam_search_store = None
        if self._use_beam_search:
            beam_search_store = BeamSearchStore.create(
                cache_indirection_shape=self.CACHE_INDIRECTION_SHAPE,
                max_num_sequences=self.max_num_sequences,
                max_beam_width=self.max_beam_width,
            )
        return self.Store(
            new_tokens=new_tokens,
            log_probs_store=log_probs_store,
            beam_search_store=beam_search_store,
        )

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int
        max_total_draft_tokens: int
        disable_overlap_scheduler: bool = False
        enable_async_worker: bool = False
        enable_speculative_beam_history_d2h: bool = False

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.max_tokens = args.max_total_draft_tokens + 1
        self.max_beam_width = args.max_beam_width
        # Snapshot of `not self._use_beam_search` so the update_requests
        # fast-path avoids a property call per iteration.
        self._batch_fastpath_eligible: bool = self.max_beam_width == 1
        if args.max_total_draft_tokens > 0 and args.max_beam_width > 1:
            raise ValueError("TorchSampler does not support beam search with speculative decoding")
        self.max_num_sequences = args.max_num_sequences
        # With the overlap scheduler, sample_async for step i runs before
        # update_requests for step i-1, so the host-side token lists lag the
        # device state. Track, per seq slot, how many sampled steps have not
        # been folded back into the request yet; bad-words handling uses this
        # to decide whether the newest token must be read device-side.
        self._track_pending_steps = not args.disable_overlap_scheduler
        self._pending_steps = [0] * self.max_num_sequences
        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.max_num_sequences, self.max_beam_width)
        self.CACHE_INDIRECTION_SHAPE = (
            self.max_num_sequences,
            self.max_beam_width,
            self.max_seq_len,
        )
        self.LOGPROBS_SHAPE = (self.max_num_sequences, self.max_beam_width, self.max_tokens)
        # Owns max_topk_logprobs / batch_max_topk_logprobs / TOPK_LOGPROBS_SHAPE;
        # constructed before _create_store, which reads the shape.
        self._log_probs = LogProbsHandler(self)

        # The Torch sampler hard-depends on flashinfer. Enforce it once here, at
        # construction, so the check stays out of the CUDA-graph-captured
        # sampling loop.
        if not IS_FLASHINFER_AVAILABLE:
            raise ImportError(
                "flashinfer is not available, please install the version pinned "
                "in requirements.txt."
            )
        self._grouped_sampler_cls = FlashInferGroupedStrategySampler
        # Per-slot Top-P Decay runtime state (FlashInfer path). Allocated for all
        # sampler instances; only decay-admitted slots are ever read.
        self._top_p_decay = TopPDecayHandler(self.max_num_sequences)

        # Token-ban handling (bad words, no-repeat ngram). The overlap-aware
        # variant is selected once here from whether the overlap scheduler is
        # enabled; only it produces the conditional (stale-host) bans.
        self._token_ban_handler: TokenBanHandler = (
            OverlappedTokenBanHandler()
            if self._track_pending_steps
            else SynchronousTokenBanHandler()
        )

        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self._create_store()
            self._request_grouper: _CachingRequestGrouper[Any] = _CachingRequestGrouper(
                self.max_num_sequences
            )
            self._finish_reasons_handler = FinishReasonsHandler(
                max_stop_word_length=self.DEFAULT_MAX_STOP_WORD_LENGTH,
                max_num_stop_words=self.DEFAULT_MAX_STOP_WORDS,
                max_num_sequences=self.max_num_sequences,
                max_beam_width=self.max_beam_width,
                max_tokens=self.max_tokens,
                max_seq_len=self.max_seq_len,
            )
            assert (
                self.store.new_tokens.shape
                == self._finish_reasons_handler.store.finish_reasons_cuda.shape
            )
            self._penalty_handler = PenaltyHandler(
                max_num_sequences=self.max_num_sequences,
                max_beam_width=self.max_beam_width,
                device="cuda",
            )

        # Initialize seed for multi-GPU consistency
        self._global_seed = 42
        self._generator: torch.Generator | None = None

        # Per-request RNG state backing SamplingParams.seed. Kept per sequence
        # slot so a seeded request's stream depends only on its own step count,
        # not on batch composition.
        self._seed_manager = _SeedManager(
            max_num_sequences=self.max_num_sequences,
            global_seed=self._global_seed,
        )

        # Force number of accepted tokens for speculative decoding testing.
        # Imported here (not at module level) to keep sampler.sampler off the
        # speculative import cycle; see the TYPE_CHECKING note above.
        from ...speculative.interface import get_force_num_accepted_tokens

        self._force_num_accepted_tokens = get_force_num_accepted_tokens()

        self._two_model_spec_dec = TwoModelSpecDecHandler(self)

        self._async_worker_init(args.enable_async_worker)

        # The speculative path bypasses _copy_to_host, so it cannot coexist
        # with the async worker. LlmArgs validation rejects the explicit
        # conflict with sampler_force_async_worker. Confidential compute may
        # also enable the async worker at runtime; if so, disable the path
        # here with a warning.
        self._use_speculative_beam_history_d2h: bool = args.enable_speculative_beam_history_d2h
        if self._use_speculative_beam_history_d2h and self.async_worker_enabled():
            logger.warning(
                "enable_speculative_beam_history_d2h is incompatible with the "
                "sampler async worker (likely auto-enabled by confidential "
                "compute); disabling the speculative beam-history D2H path."
            )
            self._use_speculative_beam_history_d2h = False

        self._stable_greedy_request_ids: list[int] = []
        self._stable_greedy_seq_slots: list[int] = []
        self._stable_greedy_seq_slots_host: Optional[torch.Tensor] = None
        self._stable_greedy_seq_slots_cuda: Optional[torch.Tensor] = None
        self._greedy_tail = GreedyTailGraph()
        # Ring slot the current _process_requests took, handed to sample_async
        # for the sample state that owns the read-back buffer.
        self._greedy_tail_slot: Optional[int] = None

        # BeamSearchHandler owns the lagged first_finish_reasons snapshots the
        # speculative predictor reads, so no separate host mirror is kept here.
        self._beam_search = BeamSearchHandler(
            store=self.store.beam_search_store,
            max_seq_len=self.max_seq_len,
            max_num_sequences=self.max_num_sequences,
            use_speculative_d2h=self._use_speculative_beam_history_d2h,
            has_multi_token_stop_words=check_stop_words_length,
            copy_to_host=self._copy_to_host,
            make_side_stream_copier=self._make_side_stream_copier,
        )

    @staticmethod
    def _is_draft_batch(requests: list[LlmRequest]) -> bool:
        """Whether this batch belongs to the draft model.

        Batches are homogeneous by construction: ModelDrafter builds all-draft
        batches for its sample_async/update_requests calls on this shared
        sampler, and PyExecutor's batches are all-target. The pending-steps
        accounting relies on this to skip draft batches wholesale; assert it so
        a mixed batch fails loudly instead of silently corrupting the counters.
        """
        is_draft: bool = requests[0].py_is_draft
        assert all(r.py_is_draft == is_draft for r in requests), (
            "sampler batch must be homogeneous (all-draft or all-target)"
        )
        return is_draft

    def get_generator(self, device: torch.device) -> torch.Generator:
        """Get a deterministic generator for the specified device.

        Args:
            device: The device to create the generator on

        Returns:
            A torch.Generator with the global seed set
        """
        if self._generator is None:
            # Fallback to a default seed if not set
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._global_seed)
        assert self._generator.device == device
        return self._generator

    def get_spec_tree_manager(
        self, resource_manager: Optional[ResourceManager]
    ) -> Optional["SpecTreeManager"]:
        if resource_manager is None:
            return None
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if spec_resource_manager is None or not hasattr(spec_resource_manager, "spec_tree_manager"):
            return None
        return spec_resource_manager.spec_tree_manager  # type: ignore

    @property
    def _use_beam_search(self) -> bool:
        return self.max_beam_width > 1

    def _can_use_fast_greedy_path(self, requests: list[LlmRequest]) -> bool:
        """
        Check if we can use the fast argmax path for greedy sampling.
        """
        # Check if all requests use greedy sampling and don't require features
        # that the fast path skips
        for req in requests:
            # vocab_size doesn't affect greediness check
            if _request_strategy(req, vocab_size=2**31) != GREEDY:
                return False

            # Fast path skips logprobs handling
            if req.py_return_log_probs:
                return False
        return True

    @staticmethod
    def _finished_beam_prefix_lengths(finish_reasons: torch.Tensor) -> list[int]:
        """Count the leading finished beams of every slot in one batched reduction.

        A request is complete once all of the beams it actually uses have a finish
        reason, i.e. once its leading ``py_beam_width`` entries are all set. Rather
        than reducing each request's row separately, reduce the whole tensor once
        and return, per slot, how many leading beams have finished. The per-request
        check then degenerates to ``prefix_length >= beam_width``, which needs no
        tensor work and stays correct for mixed beam widths regardless of what the
        columns past a request's width hold.

        Args:
            finish_reasons: Shape ``(max_batch_size, max_beam_width)``. The finish
                reasons of every beam of every slot.

        Returns:
            Per slot, the number of leading beams whose finish reason is set.
        """
        unfinished = finish_reasons == FinishReason.NOT_FINISHED.value
        # A beam belongs to the finished prefix iff no unfinished beam precedes it
        # and it is finished itself, i.e. iff the running count of unfinished beams
        # up to and including it is still zero. Counting those positions yields the
        # prefix length directly, and needs no special case for a fully finished
        # row (every position counts) or a row finishing at beam 0 (none do).
        return (unfinished.cumsum(dim=1) == 0).sum(dim=1).tolist()

    def _handle_first_finish_reasons(
        self,
        request: LlmRequest,
        finished_beam_prefix_lengths: list[int],
        finish_reasons_list: list[list[int]],
    ) -> bool:
        """Check if all beams of a request have finished and set the request state accordingly

        Args:
            request: LlmRequest. The request to check.
            finished_beam_prefix_lengths: Per slot, the number of leading beams that
                have finished, as returned by ``_finished_beam_prefix_lengths``.
            finish_reasons_list: list[list[int]]. The finish reasons for each beam.
        Returns:
            True if all beams have finished, False otherwise.
        """
        assert request.py_seq_slot is not None
        beam_width = request.py_beam_width
        if finished_beam_prefix_lengths[request.py_seq_slot] < beam_width:
            return False
        request.state = LlmRequestState.GENERATION_COMPLETE
        request_finish_reasons = finish_reasons_list[request.py_seq_slot]
        for beam_idx in range(beam_width):
            request.set_finished_reason(
                FinishReason(request_finish_reasons[beam_idx]),
                beam_idx,
            )
        return True

    @staticmethod
    @nvtx_range("update_original_tokens")
    def _update_original_tokens(
        original_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor,
        new_tokens: torch.Tensor,
    ) -> None:
        """Update the original tokens storage for the request with the newly sampled tokens

        When using streaming a requests tokens may be altered, leading to wrong results when called multiple times.
        Store the original tokens in a separate buffer to use them as a consistent basis
        when updating the tokens in a request."""
        assert new_tokens.device == original_tokens.device, (
            "new_tokens and original_tokens must be on the same device"
        )
        original_tokens[seq_slots, :, seq_lens] = new_tokens[0, seq_slots, :]

    def finish_if_reason(
        self, request: LlmRequest, finish_reasons: FinishReasonsList, *, step: int, beam_idx: int
    ) -> bool:
        assert request.py_seq_slot is not None
        reason = FinishReason(finish_reasons[request.py_seq_slot][step][beam_idx])
        valid_reasons = {FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS}
        if reason in valid_reasons:
            request.finish_by(reason, beam_idx)
            return True
        return False

    @classmethod
    def _collect_new_requests_for_setup(
        cls, scheduled_requests: ScheduledRequests
    ) -> list[LlmRequest]:
        # ADP can inject generation-phase dummy requests after request activation.
        # Those still need sampler-side slot initialization before grouping/sampling.
        # The two source lists are disjoint by construction.
        context_new_requests = [
            request
            for request in scheduled_requests.context_requests_last_chunk
            if not request.is_finished and not request.py_is_draft
        ]
        adp_dummy_generation_requests = [
            request
            for request in scheduled_requests.generation_requests
            if request.is_attention_dp_dummy
        ]
        return context_new_requests + adp_dummy_generation_requests

    @override
    def validate_request(self, request: LlmRequest) -> None:
        # Reject unsupported top-p-decay combinations at admission, so only the offending
        # request fails (raising later, inside setup_sampler_step or sampling, would abort
        # the whole executor step). Occurrence penalties have no unsupported combination
        # left: beam search is supported, and beam search with speculative decoding is
        # rejected for the whole sampler in __init__.
        self._top_p_decay.validate_request(request)
        if self._use_beam_search:
            if request.py_return_log_probs:
                if request.py_num_logprobs > 1:
                    raise ValueError(
                        "Beam search does not support returning multiple logprobs per request"
                    )
                if request.py_num_logprobs != 0:
                    raise ValueError(
                        "Beam search only supports returning the sampled logprob per token"
                    )
            # Every early_stopping mode is served by the candidate-beams-array
            # path (see beam_search_sampling_batch_cba); the mode only selects
            # the done verdict computed there.

    @override
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests) -> None:
        """Setup the sampler step for the requests

        Args:
            scheduled_requests: The scheduled requests to set up the sampler step for.
        """
        new_requests = self._collect_new_requests_for_setup(scheduled_requests)

        if not new_requests:
            return

        # Used for all store updates
        seq_slots: list[int] = []
        # Used for beam search updates
        max_prompt_len: int = 0

        # Prepare finish reasons handler
        self._finish_reasons_handler.setup_new_request_handling()
        for request in new_requests:
            slot = request.py_seq_slot
            assert slot is not None
            seq_slots.append(slot)
            # update temp_data with this requests data
            self._finish_reasons_handler.prepare_for_new_request(request)

            if self._use_beam_search:
                assert not (request.py_return_log_probs and request.py_num_logprobs > 1), (
                    "Beam search does not support returning multiple logprobs per request"
                )
                max_prompt_len = max(max_prompt_len, request.py_prompt_len)
                if self._use_speculative_beam_history_d2h:
                    # Drop stale predictor state from any prior occupant of this slot.
                    self._beam_search.clear_slot(slot)

            self._request_grouper.prepare_for_new_request(request, slot)
            self._penalty_handler.prepare_for_new_request(request, slot)

        max_lens = self._finish_reasons_handler.new_max_lens
        end_ids = self._finish_reasons_handler.new_end_ids
        prompt_lens = [request.py_prompt_len for request in new_requests]
        beam_caps = [request.py_beam_width for request in new_requests]
        # Perform updates to the stores
        full_list = [seq_slots, max_lens, end_ids, prompt_lens, beam_caps]
        # perform only a single copy
        full_list_tensor_host = torch.tensor(
            full_list, device="cpu", dtype=torch.int32, pin_memory=prefer_pinned()
        )
        full_list_tensor_cuda = full_list_tensor_host.to(device="cuda", non_blocking=True)
        seq_slots_tensor_host = full_list_tensor_host[0]
        seq_slots_tensor_cuda = full_list_tensor_cuda[0]
        max_lens_tensor_cuda = full_list_tensor_cuda[1]
        end_ids_tensor_cuda = full_list_tensor_cuda[2]
        prompt_lens_tensor_cuda = full_list_tensor_cuda[3]
        beam_caps_tensor_cuda = full_list_tensor_cuda[4]

        # Cast to int64 once for downstream ``index_copy_`` / ``index_fill_`` calls.
        seq_slots_tensor_cuda_long = seq_slots_tensor_cuda.long()

        self._finish_reasons_handler.update_for_new_request(
            seq_slots_cuda_long=seq_slots_tensor_cuda_long,
            max_lengths_cuda=max_lens_tensor_cuda,
            end_ids_cuda=end_ids_tensor_cuda,
            seq_slots_host=seq_slots_tensor_host,
            all_sampling_requests=new_requests + scheduled_requests.generation_requests,
        )

        self._top_p_decay.setup_for_new_requests(
            new_requests, new_seq_slots_cuda_long=seq_slots_tensor_cuda_long
        )

        self._penalty_handler.update_for_new_requests(
            new_seq_slots_cuda_long=seq_slots_tensor_cuda_long
        )

        if self._use_beam_search:
            beam_search_store = self.store.beam_search_store
            assert beam_search_store is not None
            # Allocate the CBA tensors on the first beam-search request: every
            # early_stopping mode runs on that path.
            beam_search_store.ensure_cba()
            prepare_beam_search(
                beam_search_store,
                self.store.log_probs_store,
                seq_slots_long=seq_slots_tensor_cuda_long,
                max_prompt_len=max_prompt_len,
                prompt_lens_cuda=prompt_lens_tensor_cuda,
                beam_caps_cuda=beam_caps_tensor_cuda,
            )

    def _add_metadata_to_grouped_requests(
        self,
        requests: list[LlmRequest],
        grouped_requests: dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue],
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor | None,
        get_metadata_type_for_group_fn: Callable[
            [GenericStrategyKeyType], Type[StrategyMetadata] | None
        ],
        *,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        req_num_steps: torch.Tensor,
    ) -> dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata]:
        grouped_requests_with_metadata: dict[
            RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata
        ] = {}
        log_probs_store = self.store.log_probs_store
        num_requests = len(requests)
        for key, value in grouped_requests.items():
            metadata_type = get_metadata_type_for_group_fn(key.strategy_key)
            metadata: StrategyMetadata | None
            if metadata_type is BeamSearchMetadata:
                metadata = self._beam_search.build_metadata(
                    requests=requests,
                    group_req_indices=value.indices,
                    seq_slots=seq_slots,
                    seq_lens=seq_lens,
                    seq_slots_cuda=seq_slots_cuda,
                    seq_lens_cuda=seq_lens_cuda,
                    num_requests=num_requests,
                    new_log_probs=log_probs_store.sampled_log_probs[..., DEFAULT_STEP_IDX],
                    end_ids_cuda=self._finish_reasons_handler.store.end_ids_cuda,
                    past_tokens_cuda=self._finish_reasons_handler.store.past_tokens_cuda,
                )
            elif metadata_type is TopPDecayMetadata:
                metadata = self._top_p_decay.build_metadata(
                    group_req_indices=value.indices,
                    req_num_steps=req_num_steps,
                    seq_slots=seq_slots,
                    seq_slots_cuda=seq_slots_cuda,
                )
            elif metadata_type is None:
                metadata = None
            else:
                raise ValueError(f"Unsupported metadata type: {metadata_type}")
            grouped_requests_with_metadata[key] = RequestGroupValueWithMetadata(
                indices=value.indices,
                strategies=value.strategies,
                speculation_needs_probs_indices=value.speculation_needs_probs_indices,
                need_processed_logprobs=value.need_processed_logprobs,
                metadata=metadata,
            )
        return grouped_requests_with_metadata

    @override
    @nvtx_range("update_requests")
    @torch.inference_mode()
    def update_requests(
        self,
        state: SampleStateTorch,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        try:
            self._update_requests(state, resource_manager)
        finally:
            # The replayed tail wrote its token into a ring buffer that is
            # reusable only now that this state has been fully consumed.
            if state.greedy_tail_slot is not None:
                self._greedy_tail.release(state.greedy_tail_slot)
                state.greedy_tail_slot = None

    def _update_requests(
        self,
        state: SampleStateTorch,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        if state.sampler_event:
            state.sampler_event.synchronize()

        if not state.requests:
            return

        if self._track_pending_steps and not self._is_draft_batch(state.requests):
            for req in state.requests:
                slot = req.py_seq_slot
                if slot is not None and self._pending_steps[slot] > 0:
                    self._pending_steps[slot] -= 1

        assert state.host is not None
        # Reuse sample_async's qualification instead of rechecking every
        # request after the asynchronous sample completes.
        if state.host.single_step_greedy:
            self._update_requests_single_beam_single_step(state)
            return

        new_tokens = state.host.new_tokens
        finish_reasons = state.host.finish_reasons_list()
        first_finish_reasons_host = state.host.first_finish_reasons
        if first_finish_reasons_host is not None:
            first_finish_reasons = first_finish_reasons_host.tolist()
            # Reduce every slot at once; the per-request loop below only reads the
            # result and updates the request objects.
            finished_beam_prefix_lengths = self._finished_beam_prefix_lengths(
                first_finish_reasons_host
            )
        else:
            first_finish_reasons = []
            finished_beam_prefix_lengths = []

        new_tokens_list = new_tokens.tolist()

        logprobs_state_list: LogProbsStateList | None = None
        if state.host.logprobs_state is not None:
            logprobs_state_list = LogProbsStateList.from_logprobs_state(state.host.logprobs_state)

        beam_history_builders = state.beam_history_builders
        assert (beam_history_builders is not None) == self._use_beam_search

        def _maybe_build_beam_history(req_idx: int) -> BeamHistory | None:
            if (
                beam_history_builders is not None
                and (beam_history_builder := beam_history_builders[req_idx]) is not None
            ):
                return beam_history_builder()
            else:
                return None

        finalized_token_updates: list[tuple[int, list[int]]] = []
        # Fast-path (batched pybind): when the batch is greedy with no beam
        # search, no logprobs, no draft tokens, no stop-words, and no
        # speculative tree, collapse per-request pybind chatter into one
        # batched add_new_tokens_to_requests call. Single-pass eligibility
        # check with early-break; falls through when any invariant breaks.
        if (
            self._batch_fastpath_eligible
            and logprobs_state_list is None
            and self.get_spec_tree_manager(resource_manager) is None
        ):
            alive_reqs: list[LlmRequest] = []
            tokens_flat: list[int] = []
            fastpath_ok = True
            new_tokens_step0 = new_tokens_list[0]
            for req in state.requests:
                if req.state == LlmRequestState.GENERATION_COMPLETE:
                    continue
                if get_draft_token_length(req) != 0 or req.py_stop_words_list:
                    fastpath_ok = False
                    break
                assert req.py_seq_slot is not None
                alive_reqs.append(req)
                tokens_flat.append(new_tokens_step0[req.py_seq_slot][DEFAULT_BEAM_IDX])
            if fastpath_ok and alive_reqs:
                add_new_tokens_to_requests(alive_reqs, tokens_flat, DEFAULT_BEAM_IDX)
                _valid_finish_reasons = {
                    FinishReason.END_ID,
                    FinishReason.LENGTH,
                    FinishReason.STOP_WORDS,
                }
                for req in alive_reqs:
                    assert req.py_seq_slot is not None
                    reason_val = finish_reasons[req.py_seq_slot][0][DEFAULT_BEAM_IDX]
                    if reason_val != 0:
                        reason = FinishReason(reason_val)
                        if reason in _valid_finish_reasons:
                            req.finish_by(reason, DEFAULT_BEAM_IDX)
                    req.py_num_accepted_draft_tokens = 0
                    req.py_num_draft_tokens_verified = 0
                    req.py_rewind_len = 0
                    req.py_decoding_iter += 1
                return

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                self._top_p_decay.retire_slot(req)
                continue

            if req.py_beam_width > 1:
                # Context-only (disaggregated prefill) requests reach the
                # sampler already flagged as finished -- the context server is
                # done with them -- but they are migrating, not completing:
                # the generation server decodes the rest. Finalizing here
                # would rewrite the beam rows from the beam history, and after
                # a single step only beam 0 has history; every other beam is
                # all BEAM_SEARCH_PAD_TOKEN, so set_generated_tokens would
                # give it an empty generated sequence. The handoff reads the
                # per-beam first token back out via getTokens().back()
                # (llmRequest.cpp), which would then hand the generation
                # server the prompt tail instead of that beam's token. Append
                # this step's tokens instead and let the generation side own
                # finalization.
                if (
                    not req.is_context_only_request
                    and (beam_history := _maybe_build_beam_history(req_idx)) is not None
                ):
                    finalize_beam(req, beam_history)
                else:
                    # Only the leading beam_width_out columns hold real tokens;
                    # the op pads the rest of the store-width row with
                    # BEAM_SEARCH_PAD_TOKEN. Appending those would put the
                    # sentinel into the request's token history, which is
                    # visible to streaming consumers and to anything reading
                    # get_tokens() mid-flight (e.g. the token-ban suffix
                    # matching) even though finalization later rewrites it.
                    for beam_idx in range(_get_beam_width_out(req)):
                        # Beam search does not support speculative decoding.
                        add_token(req, new_tokens_list, beam_idx=beam_idx)
                    self._log_probs.handle_logprobs(
                        req, logprobs_state_list=logprobs_state_list, count=1
                    )
                assert first_finish_reasons_host is not None
                self._handle_first_finish_reasons(
                    req, finished_beam_prefix_lengths, first_finish_reasons
                )
                if self._use_speculative_beam_history_d2h:
                    # Snapshot for the next step's predictor.
                    assert req.py_seq_slot is not None
                    self._beam_search.record_first_finish_reasons(
                        req.py_seq_slot, first_finish_reasons_host[req.py_seq_slot]
                    )
                if req.is_context_only_request:
                    beam_search_store = self.store.beam_search_store
                    assert beam_search_store is not None
                    assert req.py_seq_slot is not None
                    beam_width = req.py_beam_width
                    first_gen_scores = (
                        beam_search_store.cum_log_probs[req.py_seq_slot, :beam_width]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    first_gen_tokens = [
                        new_tokens_list[0][req.py_seq_slot][beam_idx]
                        for beam_idx in range(beam_width)
                    ]
                    first_gen_log_probs = [
                        {token_id: Logprob(logprob=log_prob, rank=None)}
                        for token_id, log_prob in zip(first_gen_tokens, first_gen_scores)
                    ]
                    req.py_result.set_first_gen_log_probs(first_gen_log_probs)
                req.py_num_accepted_draft_tokens = 0
                req.py_num_draft_tokens_verified = 0
                req.py_rewind_len = 0
            else:
                processed = 1
                num_tokens_before = req.get_num_tokens(DEFAULT_BEAM_IDX)
                num_accepted = self._two_model_spec_dec.process_draft_tokens(
                    req,
                    new_tokens_tensor=new_tokens,
                    new_tokens_list=new_tokens_list,
                    finish_reasons=finish_reasons,
                    resource_manager=resource_manager,
                )
                if (actual_draft_len := get_draft_token_length(req)) > 0:
                    req.py_num_accepted_draft_tokens = num_accepted
                    # Pair the acceptance count with the real proposal count
                    # of the same step: py_draft_tokens is padded to the
                    # static max for CUDA graphs, so use the pre-padding
                    # length the drafter recorded. py_rewind_len keeps the
                    # padded length (padding occupies KV cache).
                    effective_len = req.py_draft_tokens_effective_len
                    req.py_num_draft_tokens_verified = (
                        min(effective_len, actual_draft_len)
                        if effective_len is not None
                        else actual_draft_len
                    )
                    req.py_rewind_len = actual_draft_len - num_accepted
                else:
                    req.py_num_accepted_draft_tokens = 0
                    req.py_num_draft_tokens_verified = 0
                    req.py_rewind_len = 0
                processed += num_accepted
                if actual_draft_len > 0:
                    num_new_tokens = req.get_num_tokens(DEFAULT_BEAM_IDX) - num_tokens_before
                    if num_new_tokens > 0:
                        assert req.py_seq_slot is not None
                        confirmed_tokens = req.get_tokens(DEFAULT_BEAM_IDX)[-num_new_tokens:]
                        finalized_token_updates.append((req.py_seq_slot, confirmed_tokens))
                self._log_probs.handle_logprobs(
                    req, logprobs_state_list=logprobs_state_list, count=processed
                )
            req.py_decoding_iter += 1
            # Check None or empty list
            if req.py_stop_words_list:
                self._finish_reasons_handler.store.num_accepted_draft_tokens_host[
                    req.py_seq_slot
                ] = req.py_num_accepted_draft_tokens
            # req.state can become GENERATION_COMPLETE within this loop iteration
            # (e.g. process_draft_tokens -> _handle_stop_criteria -> finish_by),
            # so the comparison is valid at runtime; mypy narrowed it away via the
            # `continue` at the top of the loop and cannot see the mutating calls.
            if req.state == LlmRequestState.GENERATION_COMPLETE:  # type: ignore[comparison-overlap]
                self._top_p_decay.retire_slot(req)

        self._penalty_handler.update_token_counts(finalized_token_updates)

    @nvtx_range("_update_requests_single_beam_single_step")
    def _update_requests_single_beam_single_step(self, state: SampleStateTorch) -> None:
        """Update the common greedy, single-token case without draft machinery."""
        assert state.host is not None
        requests = [
            request
            for request in state.requests
            if request.state != LlmRequestState.GENERATION_COMPLETE
        ]
        if not requests:
            return

        all_new_tokens = state.host.new_tokens.tolist()
        if len(requests) == len(state.requests):
            new_tokens = all_new_tokens
        else:
            new_tokens = [
                new_token
                for request, new_token in zip(state.requests, all_new_tokens)
                if request.state != LlmRequestState.GENERATION_COMPLETE
            ]
        add_new_tokens_to_requests(requests, new_tokens, DEFAULT_BEAM_IDX)

        # sample_async deliberately omits the device finish-reason tensor for
        # this qualified path; completion is derived from compact host tokens.
        assert state.host.finish_reasons is None
        for request, new_token in zip(requests, new_tokens):
            # The stable greedy path excludes stop words. Keep EOS ahead of the
            # length check so a terminal EOS at the token limit is reported as
            # END_ID, matching _handle_stop_criteria.
            if new_token == request.py_end_id:
                request.finish_by(FinishReason.END_ID, DEFAULT_BEAM_IDX)
            elif (
                request.max_beam_num_tokens - request.py_orig_prompt_len
                >= request.py_max_new_tokens
                or request.max_beam_num_tokens >= self.max_seq_len
            ):
                request.finish_by(FinishReason.LENGTH, DEFAULT_BEAM_IDX)
            request.py_num_accepted_draft_tokens = 0
            request.py_rewind_len = 0
            request.py_decoding_iter += 1

    @override
    @torch.inference_mode()
    @nvtx_range("sample_async")
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTorch:
        # NB: The sampler is either called directly by PyExecutor, for the target model,
        #     or by ModelDrafter.prepare_draft_tokens(), for the draft model. In the former
        #     case there are 1 + get_draft_token_length(request) tokens per request. In the
        #     latter case, there is always only 1 token per request because draft
        #     tokens are sampled one-by-one.
        self.setup_sampler_step(scheduled_requests)
        new_tokens = self.store.new_tokens

        if self._track_pending_steps:
            # A context request claims a (possibly reused) slot: clear any
            # counter leaked by a prior occupant that never got its final
            # update_requests. Must happen before _process_requests, which
            # reads the counters for bad-words staleness.
            for r in scheduled_requests.context_requests:
                if not r.py_is_draft:
                    assert r.py_seq_slot is not None
                    self._pending_steps[r.py_seq_slot] = 0

        # seq_slots_cuda / seq_lens_cuda are cast once inside
        # _process_requests and shared with the beam-search metadata builder.
        (
            requests,
            seq_slots_host,
            seq_lens_host,
            seq_slots_cuda,
            seq_lens_cuda,
            new_tokens_host,
            single_step_greedy,
        ) = self._process_requests(
            scheduled_requests,
            model_outputs,
            new_tokens,
            num_context_logits_prefix_sum,
        )

        if self._track_pending_steps and requests and not self._is_draft_batch(requests):
            for r in requests:
                assert r.py_seq_slot is not None
                self._pending_steps[r.py_seq_slot] += 1

        finish_reasons_host: torch.Tensor | None = None
        first_finish_reasons_host: torch.Tensor | None = None
        beam_history_builders: list[BeamHistoryBuilder | None] | None = None
        # Forwarded to _record_sampler_event so SamplerEvent.synchronize
        # awaits any side-stream D2H copies host-side.
        side_stream_event: torch.cuda.Event | None = None
        if requests:
            beam_search_store = self.store.beam_search_store
            assert self._use_beam_search == (beam_search_store is not None)
            # Prepare stop word handling
            # Draft requests need to be ignored for stop word handling as they never set up
            # their buffers in the store.
            # Assume that either all requests are drafts or none are drafts
            is_draft_batch = requests[0].py_is_draft
            if not single_step_greedy:
                assert seq_lens_host is not None
                assert seq_lens_cuda is not None
                finish_reasons_device = self._finish_reasons_handler.write_finish_reasons(
                    seq_slots_host=seq_slots_host,
                    is_draft_batch=is_draft_batch,
                    seq_slots_cuda=seq_slots_cuda,
                    seq_lens_cuda=seq_lens_cuda,
                    new_tokens_cuda=new_tokens,
                    first_finish_reasons_cuda=(
                        beam_search_store.first_finish_reasons
                        if beam_search_store is not None
                        else None
                    ),
                    pending_harvest_cuda=(
                        beam_search_store.pending_harvest if beam_search_store is not None else None
                    ),
                )
                finish_reasons_host = self._copy_to_host(finish_reasons_device)

            if self._use_beam_search:
                assert beam_search_store is not None
                assert seq_lens_cuda is not None
                first_finish_reasons = beam_search_store.first_finish_reasons
                first_finish_reasons_host = self._copy_to_host(first_finish_reasons)
                self._update_original_tokens(
                    beam_search_store.original_tokens, seq_slots_cuda, seq_lens_cuda, new_tokens
                )
                beam_history_builders, side_stream_event = self._beam_search.prepare_beam_histories(
                    requests, finish_reasons=first_finish_reasons
                )

        # copy logprobs to host
        logprobs_state: LogProbsState | None = None
        if self._log_probs._return_log_probs(requests):
            log_probs_store = self.store.log_probs_store
            host_topk_vals = self._copy_to_host(
                log_probs_store.topk_vals[..., : self._log_probs.batch_max_topk_logprobs]
            )
            host_topk_indices = self._copy_to_host(
                log_probs_store.topk_indices[..., : self._log_probs.batch_max_topk_logprobs]
            )
            host_sampled_vals = self._copy_to_host(log_probs_store.sampled_log_probs)
            host_sampled_indices = self._copy_to_host(log_probs_store.sampled_log_prob_indices)
            host_sampled_rank = self._copy_to_host(log_probs_store.sampled_log_prob_ranks)
            logprobs_state = LogProbsState(
                topk_vals=host_topk_vals,
                topk_indices=host_topk_indices,
                sampled_vals=host_sampled_vals,
                sampled_indices=host_sampled_indices,
                sampled_rank=host_sampled_rank,
            )

        sampler_event = self._record_sampler_event(side_stream_event=side_stream_event)
        greedy_tail_slot = self._greedy_tail_slot
        self._greedy_tail_slot = None
        return SampleStateTorch(
            requests=requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens_host,
                finish_reasons=finish_reasons_host,
                first_finish_reasons=first_finish_reasons_host,
                logprobs_state=logprobs_state,
                single_step_greedy=single_step_greedy,
            ),
            sampler_event=sampler_event,
            beam_history_builders=beam_history_builders,
            greedy_tail_slot=greedy_tail_slot,
        )

    @nvtx_range("sample_batched_by_strategy")
    @torch.inference_mode()
    def _sample_batched_by_strategy(
        self,
        logits_cuda: torch.Tensor,
        requests: list[LlmRequest],
        model_outputs: dict[str, Any],
        *,
        logits_cuda_indexer: _PackedStepIndexer,
        req_num_generated_tokens: torch.Tensor,
        req_num_steps: torch.Tensor,
        req_offsets: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        token_dtype: torch.dtype,
        return_log_probs: bool,
    ) -> _BatchedSamplingResult:
        cuda_device = logits_cuda.device

        self._seed_manager.observe(requests)

        grouped_requests, need_raw_logprobs = self._request_grouper.group_requests_by_strategy_key(
            requests,
            pin_memory=prefer_pinned(),
            strategy_to_key=self._grouped_sampler_cls.strategy_grouping_key,
            seq_slots=seq_slots,
            vocab_size=logits_cuda.size(1),  # Dummy value; strategy should already be cached
        )
        grouped_requests_with_metadata = self._add_metadata_to_grouped_requests(
            requests,
            grouped_requests,
            seq_slots,
            seq_lens,
            get_metadata_type_for_group_fn=self._grouped_sampler_cls.get_metadata_type_for_group,
            seq_slots_cuda=seq_slots_cuda,
            seq_lens_cuda=seq_lens_cuda,
            req_num_steps=req_num_steps,
        )
        generator_cuda = self.get_generator(cuda_device)

        # NB: Currently, "d2t" is applied to draft tokens, but not to draft logits,
        #     breaking _process_draft_tokens_rejection_sampling.
        needs_d2t = "d2t" in model_outputs
        if needs_d2t and (
            len(grouped_requests_with_metadata) > 1
            or (
                grouped_requests_with_metadata
                and next(iter(grouped_requests_with_metadata.values())).strategies[0] != GREEDY
            )
        ):
            raise ValueError("d2t does not yet support non-greedy sampling")

        # Tensors for collecting sampling results (in batch ordering)
        batch_req_indices = torch.empty((len(requests),), dtype=torch.int32)
        batch_next_tokens_cuda_int = torch.empty(
            (logits_cuda.size(0), self.max_beam_width), device=cuda_device, dtype=token_dtype
        )
        batch_processed_logprobs_cuda = (
            torch.empty(  # NB: overprovisioning buffer for simplicity and later reuse in _process_logprobs
                (logits_cuda.size(0), logits_cuda.size(1)), device=cuda_device, dtype=torch.float32
            )
            if return_log_probs
            else None
        )
        reqs_indices_needing_processed_logprobs: list[int] = []
        batch_processed_logprobs_offset = 0
        batch_req_idx_offset_start = 0
        batch_next_tokens_offset_start = 0
        for group_key, group_val_with_metadata in grouped_requests_with_metadata.items():
            strategy_key = group_key.strategy_key
            needs_probs = group_key.needs_probs
            group_req_indices = group_val_with_metadata.indices
            group_strategies = group_val_with_metadata.strategies
            group_speculation_needs_probs_indices = (
                group_val_with_metadata.speculation_needs_probs_indices
            )
            group_need_processed_logprobs = group_val_with_metadata.need_processed_logprobs
            group_metadata = group_val_with_metadata.metadata

            # group_req_indices: Indices of 'requests' entries having the same sampling
            # strategy, ordered ascending.
            batch_req_idx_offset_end = batch_req_idx_offset_start + group_req_indices.size(0)
            batch_req_indices[batch_req_idx_offset_start:batch_req_idx_offset_end] = (
                group_req_indices
            )

            # NB: Assuming that group_req_indices are sorted
            group_req_1st_index = cast(int, group_req_indices[0].item())
            group_req_last_index = cast(int, group_req_indices[-1].item())
            group_logits_cuda_indices_cuda: torch.Tensor | slice
            logit_indices_for_sampler: Optional[torch.Tensor]
            if group_req_last_index - group_req_1st_index + 1 == len(group_req_indices):
                # Avoid data movement if indices are contiguous
                group_logits_cuda_indices_cuda = slice(
                    req_offsets[group_req_1st_index],
                    req_offsets[group_req_last_index]
                    + req_num_generated_tokens[group_req_last_index],
                )
                group_logits_cuda = logits_cuda[group_logits_cuda_indices_cuda]
                logit_indices_for_sampler = None
            else:
                group_logits_cuda_indices = logits_cuda_indexer[group_req_indices]
                group_logits_cuda_indices_cuda = group_logits_cuda_indices.to(
                    device=logits_cuda.device, non_blocking=True
                )
                group_logits_cuda = logits_cuda
                logit_indices_for_sampler = group_logits_cuda_indices_cuda

            group_steps_per_request = req_num_steps[group_req_indices].tolist()
            group_strategies_per_step = [  # convert from per-request to per-step
                strat
                for strat, steps in zip(group_strategies, group_steps_per_request)
                for _ in range(steps)
            ]

            # Per-request seeds only when some live request actually asked for
            # one; otherwise the shared generator keeps the previous behavior.
            group_seeds: Optional[RequestSeeds] = None
            if self._seed_manager.any_seeded:
                group_slots_per_step = [
                    slot
                    for slot, steps in zip(
                        seq_slots[group_req_indices].tolist(), group_steps_per_request
                    )
                    for _ in range(steps)
                ]
                group_seeds = self._seed_manager.make_row_seeds(
                    group_slots_per_step, device=cuda_device
                )
                self._seed_manager.advance(group_slots_per_step)

            group_next_tokens_cuda, group_softmax_cuda, group_temperature_cuda = (
                self._grouped_sampler_cls.sample_grouped_strategies(
                    strategy_key,
                    group_strategies_per_step,
                    group_logits_cuda,
                    generator=generator_cuda,
                    return_probs=needs_probs,
                    group_logit_indices=logit_indices_for_sampler,
                    group_metadata=group_metadata,
                    seeds=group_seeds,
                )
            )
            batch_next_tokens_offset_end = (
                batch_next_tokens_offset_start + group_next_tokens_cuda.size(0)
            )
            # if no beam search is used, the shape is (batch_size,), so we need to unsqueeze it to (batch_size, 1)
            if group_next_tokens_cuda.dim() == 1:
                group_next_tokens_cuda = group_next_tokens_cuda.unsqueeze(1)
            batch_next_tokens_cuda_int[
                batch_next_tokens_offset_start:batch_next_tokens_offset_end
            ].copy_(group_next_tokens_cuda, non_blocking=True)

            # --- Prepare data for logprobs ---
            if return_log_probs:
                assert batch_processed_logprobs_cuda is not None
                if logits_cuda.dtype != batch_processed_logprobs_cuda.dtype:
                    # NB: tensorrt_llm._torch.modules.logits_processor.LogitsProcessor forces logits
                    #     to float32, so this warning is not expected to get triggered. To support, e.g.,
                    #     bfloat16, unittest/_torch/sampler/test_logits_logprobs.py::TestLogsprobsInBatchedSampling
                    #     should be parametrized accordingly and dtype handling in TorchSampler needs
                    #     to be cleaned up to ensure consistent dtypes for logits, temperature, softmax, etc.
                    logger.warning_once(
                        "Processed logprobs calculation is only tested with float32 logits. Results for lower "
                        "precision logits may be inaccurate.",
                        key="WARN_INACCURATE_PROCESSED_LOGPROBS",
                    )
                need_processed_logprobs_req_indices = group_req_indices[
                    group_need_processed_logprobs
                ]
                group_num_requests_needing_processed_logprobs = (
                    need_processed_logprobs_req_indices.size(0)
                )
                if group_num_requests_needing_processed_logprobs > 0:
                    need_processed_logprobs_req_indices_list = (
                        need_processed_logprobs_req_indices.tolist()
                    )
                    reqs_indices_needing_processed_logprobs += (
                        need_processed_logprobs_req_indices_list
                    )

                    # Gather logprobs only for non-beam search requests
                    if self._use_beam_search:
                        skip_processed_logprobs_group_req_indices_list = [
                            grp_idx
                            for grp_idx, req_idx in enumerate(
                                need_processed_logprobs_req_indices_list
                            )
                            if requests[req_idx].py_beam_width != 1
                        ]

                        # repurpose group_need_processed_logprobs
                        group_gather_processed_logprobs = group_need_processed_logprobs
                        del group_need_processed_logprobs
                        skip_processed_logprobs_group_req_indices = torch.tensor(
                            skip_processed_logprobs_group_req_indices_list,
                            dtype=need_processed_logprobs_req_indices.dtype,
                        )
                        group_gather_processed_logprobs[
                            skip_processed_logprobs_group_req_indices
                        ] = False
                        num_gather_processed_logprobs_req_indices = len(
                            need_processed_logprobs_req_indices_list
                        ) - len(skip_processed_logprobs_group_req_indices_list)
                    else:
                        # repurpose group_need_processed_logprobs
                        group_gather_processed_logprobs = group_need_processed_logprobs
                        del group_need_processed_logprobs
                        num_gather_processed_logprobs_req_indices = len(
                            need_processed_logprobs_req_indices_list
                        )

                    # Gather relevant logits and probs; using prefix 'proc_lp' for subset of
                    # requests in current sampling requests group which require processed logprobs.
                    assert group_softmax_cuda is not None
                    if num_gather_processed_logprobs_req_indices == group_req_indices.size(0):
                        if logit_indices_for_sampler is None:
                            proc_lp_logits_cuda = group_logits_cuda
                        else:
                            proc_lp_logits_cuda = group_logits_cuda[logit_indices_for_sampler]
                        proc_lp_softmax_cuda = group_softmax_cuda
                        proc_lp_temperature_cuda = group_temperature_cuda
                        if isinstance(proc_lp_temperature_cuda, torch.Tensor):
                            proc_lp_temperature_cuda = proc_lp_temperature_cuda.unsqueeze(-1)
                    else:
                        proc_lp_group_steps = req_num_generated_tokens[group_req_indices]
                        proc_lp_step_mask_cuda = torch.repeat_interleave(
                            group_gather_processed_logprobs.to(
                                device=logits_cuda.device, non_blocking=True
                            ),
                            proc_lp_group_steps.to(device=logits_cuda.device, non_blocking=True),
                            output_size=cast(int, proc_lp_group_steps.sum().item()),
                        ).unsqueeze(-1)
                        proc_lp_steps_num_selected = cast(
                            int,
                            (
                                group_gather_processed_logprobs.to(dtype=proc_lp_group_steps.dtype)
                                * proc_lp_group_steps
                            )
                            .sum()
                            .item(),
                        )
                        if logit_indices_for_sampler is None:
                            proc_lp_logits_cuda = group_logits_cuda.new_empty(
                                (proc_lp_steps_num_selected, *group_logits_cuda.shape[1:]),
                            )
                            torch.masked_select(
                                group_logits_cuda,
                                proc_lp_step_mask_cuda,
                                out=proc_lp_logits_cuda.view(-1),
                            )
                        else:
                            logit_indices_for_sampler_cuda = logit_indices_for_sampler.to(
                                device=logits_cuda.device, non_blocking=True
                            )
                            proc_lp_logits_indices_cuda = logit_indices_for_sampler_cuda.new_empty(
                                (
                                    proc_lp_steps_num_selected,
                                    *logit_indices_for_sampler_cuda.shape[1:],
                                ),
                            )
                            torch.masked_select(
                                logit_indices_for_sampler_cuda,
                                proc_lp_step_mask_cuda.squeeze(-1),
                                out=proc_lp_logits_indices_cuda.view(-1),
                            )
                            proc_lp_logits_cuda = group_logits_cuda[proc_lp_logits_indices_cuda]
                        proc_lp_softmax_cuda = group_softmax_cuda.new_empty(
                            (proc_lp_steps_num_selected, *group_softmax_cuda.shape[1:]),
                        )
                        torch.masked_select(
                            group_softmax_cuda,
                            proc_lp_step_mask_cuda,
                            out=proc_lp_softmax_cuda.view(-1),
                        )
                        if isinstance(group_temperature_cuda, torch.Tensor):
                            proc_lp_temperature_cuda = group_temperature_cuda.new_empty(
                                (proc_lp_steps_num_selected, *group_temperature_cuda.shape[1:]),
                            )
                            torch.masked_select(
                                group_temperature_cuda,
                                proc_lp_step_mask_cuda.squeeze(-1),
                                out=proc_lp_temperature_cuda.view(-1),
                            )
                            proc_lp_temperature_cuda = proc_lp_temperature_cuda.unsqueeze(-1)
                        else:
                            proc_lp_temperature_cuda = group_temperature_cuda

                    # Apply temperature
                    if proc_lp_temperature_cuda is not None:
                        if proc_lp_logits_cuda is group_logits_cuda:
                            proc_lp_logits_cuda = proc_lp_logits_cuda / proc_lp_temperature_cuda
                        else:
                            proc_lp_logits_cuda /= proc_lp_temperature_cuda

                    # Reconstruct processed logprobs from softmax and logits
                    proc_lp_max_logits_cuda, proc_lp_max_logit_indices = proc_lp_logits_cuda.max(
                        dim=-1
                    )
                    # NB: The operations below could be fused using torch.compile (cf. Fusions) for performance.
                    proc_lp_renorm_offset_cuda = (
                        proc_lp_softmax_cuda.gather(
                            dim=1, index=proc_lp_max_logit_indices.unsqueeze(-1)
                        )
                        .log()
                        .squeeze(-1)
                        - proc_lp_max_logits_cuda
                    )

                    batch_processed_logprobs_offset_next = (
                        batch_processed_logprobs_offset + proc_lp_max_logits_cuda.size(0)
                    )
                    proc_lp_dest_cuda_view = batch_processed_logprobs_cuda[
                        batch_processed_logprobs_offset:batch_processed_logprobs_offset_next
                    ]
                    batch_processed_logprobs_offset = batch_processed_logprobs_offset_next
                    torch.add(
                        proc_lp_logits_cuda,
                        proc_lp_renorm_offset_cuda.unsqueeze(-1),
                        out=proc_lp_dest_cuda_view,
                    )
                    # NB: In principle. copying and masking whole-vocab tensors could be avoided by storing:
                    #        -  the number of unmasked tokens 'nu'
                    #        -  r := log(max(probs)) - max(logits)
                    #   Later, processed logprobs can be reconstructed from logits _after_ applying
                    #   top-k: Add 'r' and mask smallest entries so that only min(k, nu) tokens remain.
                    #   Currently, this approach is not viable because various tie-breaking behaviors are
                    #   unspecified for several of the kernels involved.
                    proc_lp_dest_cuda_view.masked_fill_(proc_lp_softmax_cuda == 0, float("-inf"))

            # Set LlmRequest.py_target_probs
            if group_speculation_needs_probs_indices.size(0) > 0:
                assert group_softmax_cuda is not None
                current_offset = 0
                for req_idx, steps in zip(
                    group_speculation_needs_probs_indices.tolist(),
                    req_num_steps[group_speculation_needs_probs_indices].tolist(),
                ):
                    next_offset = current_offset + steps
                    # using view avoids copy
                    requests[req_idx].py_target_probs = group_softmax_cuda[
                        current_offset:next_offset
                    ]
                    current_offset = next_offset

            batch_next_tokens_offset_start = batch_next_tokens_offset_end
            batch_req_idx_offset_start = batch_req_idx_offset_end

        # Prepare logit indices for raw logprobs
        reqs_indices_needing_raw_logprobs = []
        batch_raw_logprob_indices_cuda = None
        if return_log_probs:
            reqs_indices_needing_raw_logprobs_tensor = torch.nonzero(need_raw_logprobs).view(-1)
            num_requests_needing_raw_logprobs = reqs_indices_needing_raw_logprobs_tensor.size(0)
            if num_requests_needing_raw_logprobs > 0:
                # Gather logprobs only for non-beam search requests
                if self._use_beam_search:
                    gather_raw_logprobs_req_indices_list = [
                        idx
                        for idx in reqs_indices_needing_raw_logprobs_tensor.tolist()
                        if requests[idx].py_beam_width == 1
                    ]
                    gather_raw_logprobs_req_indices_tensor = torch.tensor(
                        gather_raw_logprobs_req_indices_list,
                        dtype=reqs_indices_needing_raw_logprobs_tensor.dtype,
                    )
                else:
                    gather_raw_logprobs_req_indices_tensor = (
                        reqs_indices_needing_raw_logprobs_tensor
                    )
                    gather_raw_logprobs_req_indices_list = (
                        gather_raw_logprobs_req_indices_tensor.tolist()
                    )

                batch_raw_logprob_indices_cuda = logits_cuda_indexer[
                    gather_raw_logprobs_req_indices_tensor
                ].to(device=logits_cuda.device, non_blocking=True)
                reqs_indices_needing_raw_logprobs = gather_raw_logprobs_req_indices_list

        # NB: 'd2t' contains offsets for transforming draft vocab token IDs into
        #     the target vocab. This is used by Eagle3ForCausalLM, whose input domain
        #     is the target vocab, whereas the output logits correspond to the draft
        #     vocab. Since the inputs/outputs are linked by TorchSampler.update_requests,
        #     they currently need to be handled within TorchSampler.
        if needs_d2t:
            apply_d2t(batch_next_tokens_cuda_int, model_outputs)

        return _BatchedSamplingResult(
            req_indices=batch_req_indices,
            next_tokens_cuda_int=batch_next_tokens_cuda_int,
            raw_logprobs_reqs_indices=reqs_indices_needing_raw_logprobs,
            raw_logprobs_logit_indices_cuda=batch_raw_logprob_indices_cuda,
            processed_logprobs_reqs_indices=reqs_indices_needing_processed_logprobs,
            processed_logprobs_end=batch_processed_logprobs_offset,
            logprobs_cuda=batch_processed_logprobs_cuda,
        )

    def _unbatch_sampling_results(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        *,
        new_tokens_cuda: torch.Tensor,
        req_num_generated_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_slots_cuda: torch.Tensor,
    ) -> torch.Tensor:
        batch_req_indices = batched_sampling_result.req_indices
        batch_next_tokens_cuda_int = batched_sampling_result.next_tokens_cuda_int

        def _dims_canonically_ordered(t: torch.Tensor) -> bool:
            return len(t.dim_order(ambiguity_check=[torch.contiguous_format])) == t.ndim

        # Assert destination tensor dimensions are canonically ordered ("row"-major); this
        # matters for element ordering in the .view(...).scatter_(...) calls below.
        assert _dims_canonically_ordered(new_tokens_cuda)

        # Construct index mapping from slice indices of computed tensors
        # (packed request_idx and step dimensions) to linearized indices
        # in (steps, seq_slot).
        batch_destination_cuda_indexer = _UnpackedStepIndexer(
            seq_slots=seq_slots[batch_req_indices],
            num_steps=req_num_generated_tokens[batch_req_indices],
            steps_dim_size=new_tokens_cuda.size(0),
            slots_dim_size=new_tokens_cuda.size(1),
            dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
            index_dtype=torch.int64,  # enforced by Tensor.scatter_
        )

        # Batch update output tensors
        batch_dest_indices_1d_cuda = (
            batch_destination_cuda_indexer[:]
            .to(new_tokens_cuda.device, non_blocking=True)
            .unsqueeze(1)
            .expand(-1, self.max_beam_width)
        )
        new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:]).scatter_(
            0, batch_dest_indices_1d_cuda, batch_next_tokens_cuda_int
        )
        # Post-sample: decay the runtime top-p for any decay-active slots that were
        # sampled this iteration (must run after tokens land in new_tokens_cuda).
        # batch_req_indices is a permutation of all sampled requests, so the set of
        # sampled slots is exactly seq_slots (the kernel updates each slot
        # independently; order is irrelevant) -- pass the resident device copy
        # instead of gathering seq_slots[batch_req_indices] on host and copying it.
        self._top_p_decay.update_after_sample(
            step_tokens=new_tokens_cuda[DEFAULT_STEP_IDX, :, DEFAULT_BEAM_IDX],
            sampled_slots_cuda=seq_slots_cuda,
        )
        return self._copy_to_host(new_tokens_cuda)

    def _compute_pending_steps(self, requests: list[LlmRequest]) -> list[int] | None:
        """Per-request count of tokens sampled but not yet written back, or None.

        With the overlap scheduler ``sample_async`` for step ``i`` runs before
        ``update_requests`` for step ``i - 1``, so the host token list — and
        hence ``get_num_tokens()`` — lags the true sequence by this many tokens.
        Length-based bans (min_length) add it back to recover the real generated
        length. Unlike the suffix-rule staleness this needs no device-side
        lookup, so it is not restricted to the single-step / single-beam case.
        Returns None when the overlap scheduler is off, on a draft batch, or
        when nothing is pending, so callers can skip the correction entirely.
        """
        if not self._track_pending_steps or self._is_draft_batch(requests):
            return None
        pending = [
            self._pending_steps[r.py_seq_slot] if r.py_seq_slot is not None else 0 for r in requests
        ]
        return pending if any(pending) else None

    def _compute_stale_by_one(self, requests: list[LlmRequest]) -> list[bool] | None:
        """Per-request overlap-scheduler stale flags, or None when not applicable.

        Returns a list where entry ``i`` is True when request ``i``'s host token
        history lags the device state by exactly one token (the previous step's
        token was sampled but not yet written back). Only the single-step,
        single-beam overlap case is reconstructible on the device side; under
        speculative decoding or beam search the missing history cannot be
        recovered, so bans are matched against the lagging host history and may
        be enforced one step late (warned once). Returns None when the overlap
        scheduler is off, on a draft batch, or when nothing is pending.
        """
        if not self._track_pending_steps or self._is_draft_batch(requests):
            return None
        pending = [
            self._pending_steps[r.py_seq_slot] if r.py_seq_slot is not None else 0 for r in requests
        ]
        if not any(pending):
            return None
        if self.max_tokens == 1 and self.max_beam_width == 1 and max(pending) == 1:
            return [p > 0 for p in pending]
        logger.warning_once(
            "bad_words / no_repeat_ngram_size with the overlap scheduler and "
            "speculative decoding or beam search: bans are matched against a "
            "host token history that lags the device state and may be enforced "
            "inexactly.",
            key="bad_words_stale_overlap",
        )
        return None

    @staticmethod
    def _select_generated_logits(
        scheduled_requests: ScheduledRequests,
        raw_logits_cuda: torch.Tensor,
        *,
        num_context_logits_prefix_sum: list[int],
    ) -> tuple[list[LlmRequest], SamplingRequestsMetadata, torch.Tensor]:
        """Select the sampling requests and the corresponding logits from the raw logits.

        Args:
            scheduled_requests: The scheduled requests. Sampling requests will be selected from this list.
            raw_logits_cuda: The raw logits corresponding to the scheduled requests.
            num_context_logits_prefix_sum: The prefix sum of the number of logits for each context request.

        Returns:
            A tuple containing the following:
            - sampling requests: The requests that are selected for sampling.
            - sampling requests metadata: The metadata for the sampling requests.
            - logits: The logits for the sampling requests.
        """
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests

        req_num_generation_steps_list = [
            1 + get_draft_token_length(req) for req in sampling_requests
        ]
        req_num_generation_steps = torch.tensor(
            req_num_generation_steps_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )

        # Rows in the logits tensor, i.e. how many rows each request occupies.
        # ModelEngine lays out generation requests at the *static* admission
        # width (py_beam_width), not the per-iteration width: with a variable
        # beam width array the two differ, and offsetting by the narrower
        # per-iteration width would make every request after the first read
        # another request's rows. logits.view() succeeds for any shape whose
        # element count divides, so that is silent corruption rather than an
        # error. Match the layout here and slice down to the per-iteration
        # width where the beams are actually consumed.
        # NB: context requests do not have multiple beams yet, hence the 1s.
        req_num_beams_list = [1] * len(finished_context_requests) + [
            req.py_beam_width for req in scheduled_requests.generation_requests
        ]
        req_num_beams = torch.tensor(
            req_num_beams_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        # context requests do not have multiple beams yet, so beam width may differ after sampling
        req_num_output_beams_list = [req.get_beam_width_by_iter(True) for req in sampling_requests]
        req_num_beams_output = torch.tensor(
            req_num_output_beams_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )

        req_num_generated_tokens = req_num_generation_steps * req_num_beams
        req_num_generated_tokens_output = req_num_generation_steps * req_num_beams_output
        # NB: These offsets consider generated tokens _only_ (draft and target, but not context).
        #     Filter out the context tokens below.
        req_offsets, sum_num_generated_tokens = _PackedStepIndexer.calculate_request_offsets(
            req_num_generated_tokens, pin_memory=prefer_pinned()
        )

        generation_requests_total_steps = (
            # NB: requests == finished_context_requests + scheduled_requests.generation_requests
            sum_num_generated_tokens - cast(int, req_offsets[len(finished_context_requests)].item())
            if scheduled_requests.generation_requests
            else 0
        )

        sampling_requests_metadata = SamplingRequestsMetadata(
            req_num_generated_tokens=req_num_generated_tokens,
            req_num_generated_tokens_output=req_num_generated_tokens_output,
            req_num_beams=req_num_beams,
            req_num_steps=req_num_generation_steps,
            req_offsets=req_offsets,
        )

        # logits_cuda should contain only the generated logits for the sampling requests.
        # If return context logits is requested, select only the generated logits of the context requests.
        #
        # NB: context_requests_chunking precede finished_context_requests, which precede generation requests.
        #     context_requests_chunking do not sample new tokens, so they should be skipped.
        #     sampling_requests == finished_context_requests + scheduled_requests.generation_requests
        num_skipped_requests = len(scheduled_requests.context_requests_chunking)
        if any(r.py_return_context_logits for r in finished_context_requests):
            assert len(num_context_logits_prefix_sum) == scheduled_requests.num_context_requests + 1
            logits_end_offsets = num_context_logits_prefix_sum[num_skipped_requests + 1 :]

            if scheduled_requests.generation_requests:
                # Since logits for generation requests are densely packed, add them all as one contiguous block
                logits_end_offsets.append(
                    num_context_logits_prefix_sum[-1] + generation_requests_total_steps
                )
                num_logits = req_num_generated_tokens[: len(finished_context_requests) + 1].clone()
                num_logits[-1] = generation_requests_total_steps
            else:
                num_logits = req_num_generated_tokens[: len(finished_context_requests)]

            logits_end_offsets_cuda = torch.tensor(
                logits_end_offsets, dtype=torch.int32, pin_memory=prefer_pinned()
            ).to(device=raw_logits_cuda.device, non_blocking=True)
            num_logits_cuda = num_logits.to(raw_logits_cuda.device, non_blocking=True)

            num_logits_to_keep = sum_num_generated_tokens

            # Now, the generated tokens for context request i are at indices
            #    range(logits_end_offsets_cuda[i] - num_logits_cuda[i],
            #          logits_end_offsets_cuda[i])
            # And if generation requests are present, those tensors each include a trailing entry selecting
            # all tokens/logits generated by all generation requests.
            indices_to_keep_cuda = torch_multi_arange(
                starts=(logits_end_offsets_cuda - num_logits_cuda),
                ends=logits_end_offsets_cuda,
                output_length=num_logits_to_keep,
            )

            logits_cuda = raw_logits_cuda[indices_to_keep_cuda]
        else:
            logits_begin_offset = num_context_logits_prefix_sum[num_skipped_requests]
            logits_cuda = raw_logits_cuda[logits_begin_offset:]

        return sampling_requests, sampling_requests_metadata, logits_cuda

    @nvtx_range("_process_requests")
    def _process_requests(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        new_tokens_cuda: torch.Tensor,
        num_context_logits_prefix_sum: list[int],
    ) -> tuple[
        list[LlmRequest],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        bool,
    ]:
        raw_logits_cuda = model_outputs["logits"]

        generation_requests = scheduled_requests.generation_requests
        request_ids = [request.py_request_id for request in generation_requests]
        maybe_seq_slots = [request.py_seq_slot for request in generation_requests]
        has_stable_greedy_batch = (
            self._stable_greedy_request_ids == request_ids
            and self._stable_greedy_seq_slots == maybe_seq_slots
        )
        can_use_stable_greedy_path = (
            bool(generation_requests)
            and self.max_beam_width == 1
            and scheduled_requests.num_context_requests == 0
            and len(generation_requests) <= raw_logits_cuda.shape[0]
            and model_outputs.get("d2t") is None
            and all(
                not request.is_dummy and get_draft_token_length(request) == 0
                for request in generation_requests
            )
            and (
                has_stable_greedy_batch
                or all(
                    request._py_embedding_bias_1d is None
                    and not getattr(request, "py_bad_words", None)
                    and not getattr(request, "py_no_repeat_ngram_size", None)
                    and not has_occurrence_penalty(request)
                    and not has_min_length(request)
                    and not request.py_return_log_probs
                    and not request.py_stop_words_list
                    and _request_strategy(request, vocab_size=2**31) == GREEDY
                    for request in generation_requests
                )
            )
        )
        if can_use_stable_greedy_path:
            if has_stable_greedy_batch:
                assert self._stable_greedy_seq_slots_host is not None
                assert self._stable_greedy_seq_slots_cuda is not None
                seq_slots_host = self._stable_greedy_seq_slots_host
                seq_slots_cuda = self._stable_greedy_seq_slots_cuda
            else:
                assert all(seq_slot is not None for seq_slot in maybe_seq_slots)
                seq_slots = [cast(int, seq_slot) for seq_slot in maybe_seq_slots]
                seq_slots_host = torch.tensor(
                    seq_slots, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                seq_slots_cuda = seq_slots_host.to(
                    device="cuda", dtype=torch.int64, non_blocking=True
                )
                self._stable_greedy_request_ids = request_ids
                self._stable_greedy_seq_slots = seq_slots
                self._stable_greedy_seq_slots_host = seq_slots_host
                self._stable_greedy_seq_slots_cuda = seq_slots_cuda

            logits_cuda = raw_logits_cuda[: len(generation_requests)]
            new_tokens_host: torch.Tensor | None = None
            if not self._async_worker_active() and supports_greedy_argmax_scatter(
                logits_cuda, new_tokens_cuda
            ):
                # The whole tail -- sample, scatter, read the token back -- is
                # replayable from a graph while the batch holds still, which
                # is the common case for this path.
                replayed = self._greedy_tail.run(
                    logits_cuda, new_tokens_cuda, seq_slots_cuda, self.max_beam_width
                )
                if replayed is not None:
                    new_tokens_host, self._greedy_tail_slot = replayed
            if new_tokens_host is None:
                next_tokens = fast_greedy_sample_kernel(
                    logits_cuda,
                    new_tokens_cuda,
                    seq_slots_cuda,
                    self.max_beam_width,
                    None,
                )
                new_tokens_host = self._copy_to_host(next_tokens)
            return (
                generation_requests,
                seq_slots_host,
                None,
                seq_slots_cuda,
                None,
                new_tokens_host,
                True,
            )

        self._stable_greedy_request_ids = []
        self._stable_greedy_seq_slots = []

        sampling_requests, sampling_requests_metadata, logits_cuda = self._select_generated_logits(
            scheduled_requests,
            raw_logits_cuda,
            num_context_logits_prefix_sum=num_context_logits_prefix_sum,
        )
        return_log_probs = self._log_probs._return_log_probs(sampling_requests)
        if return_log_probs:
            self._log_probs._prepare_log_probs(sampling_requests)

        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in sampling_requests],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

        # necessary for beam search and max_length checks
        seq_lens_host = torch.tensor(
            [r.max_beam_num_tokens for r in sampling_requests],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

        # Cast seq_slots / seq_lens to CUDA exactly once; consumed by both
        # the per-group beam-search metadata builder and the finish-reasons
        # handler in sample_async. int64 is required for the index_*_ ops
        # downstream.
        seq_slots_cuda = seq_slots_host.to(device="cuda", dtype=torch.int64, non_blocking=True)
        seq_lens_cuda = seq_lens_host.to(device="cuda", non_blocking=True)

        # Handle embedding bias
        apply_embedding_bias(
            logits_cuda, sampling_requests, sampling_requests_metadata.req_num_steps
        )

        # Apply repetition/presence/frequency penalties in place, before the greedy fast
        # path, so both greedy and grouped-sampling logits are penalized. With beam search
        # this also re-parents the per-beam counts, which reads the predecessor map the
        # step's sampling is about to overwrite -- so it has to stay ahead of sampling.
        beam_search_store = self.store.beam_search_store
        self._penalty_handler.apply(
            logits_cuda,
            sampling_requests,
            new_tokens=new_tokens_cuda,
            seq_slots=seq_slots_cuda,
            request_offsets=sampling_requests_metadata.req_offsets,
            request_num_steps=sampling_requests_metadata.req_num_steps,
            request_num_beams=sampling_requests_metadata.req_num_beams,
            predecessor_beams=(
                beam_search_store.predecessor_beams if beam_search_store is not None else None
            ),
            # _is_draft_batch reads requests[0]; an empty batch has no penalties to apply
            # anyway, so short-circuit rather than index into it.
            is_draft_batch=bool(sampling_requests) and self._is_draft_batch(sampling_requests),
        )

        batch_has_min_length = any(has_min_length(r) for r in sampling_requests)
        has_bad_words = any(getattr(r, "py_bad_words", None) for r in sampling_requests)
        # Normalized in executor_request_to_llm_request: a positive int, or
        # None when the restriction is disabled for the request.
        ngram_sizes = [getattr(r, "py_no_repeat_ngram_size", None) for r in sampling_requests]
        has_no_repeat_ngram = any(size is not None for size in ngram_sizes)
        if batch_has_min_length or has_bad_words or has_no_repeat_ngram:
            # Overlap-scheduler stale flags (per request): True when the host
            # token history lags the device by one token. Only the overlap
            # handler consumes them; computed here as it needs sampler state.
            # Only the suffix-matching bans (bad words, no-repeat ngram) care:
            # min_length bans EOS from a length count, never from token values,
            # so a lagging history cannot mismatch it. Skipping the call for
            # min-length-only batches also avoids emitting the stale-history
            # warning, which names features such a batch does not use.
            stale_by_one = (
                self._compute_stale_by_one(sampling_requests)
                if (has_bad_words or has_no_repeat_ngram)
                else None
            )
            # min_length compares against get_num_tokens(), which counts only
            # the host history; add back the tokens still pending write-back so
            # the generated length is exact under the overlap scheduler.
            pending_steps = (
                self._compute_pending_steps(sampling_requests) if batch_has_min_length else None
            )
            bans = self._token_ban_handler.generate_ban_list(
                sampling_requests,
                sampling_requests_metadata.req_num_steps.tolist(),
                sampling_requests_metadata.req_num_beams.tolist(),
                ngram_sizes,
                stale_by_one=stale_by_one,
                pending_steps=pending_steps,
            )
            self._token_ban_handler.apply_ban_list(
                logits_cuda, bans, new_tokens_cuda=new_tokens_cuda
            )

        # Fast path for greedy sampling
        if self._can_use_fast_greedy_path(sampling_requests):
            # Compute destination indices on CPU (same pattern as _unbatch_sampling_results)
            batch_destination_indexer = _UnpackedStepIndexer(
                seq_slots=seq_slots_host,
                num_steps=sampling_requests_metadata.req_num_generated_tokens,
                steps_dim_size=new_tokens_cuda.size(0),
                slots_dim_size=new_tokens_cuda.size(1),
                dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
                index_dtype=torch.int64,
            )
            batch_dest_indices_cuda = batch_destination_indexer[:].to(
                new_tokens_cuda.device, non_blocking=True
            )

            # Get d2t tensor if present
            d2t = model_outputs.get("d2t", None)

            # Run compiled kernel for argmax, d2t application, and scatter
            fast_greedy_sample_kernel(
                logits_cuda,
                new_tokens_cuda,
                batch_dest_indices_cuda,
                self.max_beam_width,
                d2t,
            )

            new_tokens_host = self._copy_to_host(new_tokens_cuda)
            return (
                sampling_requests,
                seq_slots_host,
                seq_lens_host,
                seq_slots_cuda,
                seq_lens_cuda,
                new_tokens_host,
                False,
            )

        # Indexer for accessing tokens in 'logits_cuda', corresponding to the
        # requests in 'requests'.
        steps_dim_size = new_tokens_cuda.size(0)
        logits_cuda_indexer = _PackedStepIndexer(
            num_steps=sampling_requests_metadata.req_num_generated_tokens,
            max_steps=steps_dim_size * self.max_beam_width,
            req_offsets=sampling_requests_metadata.req_offsets,
        )

        # Perform sampling in batches
        batched_sampling_result = self._sample_batched_by_strategy(
            logits_cuda,
            sampling_requests,
            model_outputs,
            logits_cuda_indexer=logits_cuda_indexer,
            req_offsets=sampling_requests_metadata.req_offsets,
            seq_slots=seq_slots_host,
            seq_lens=seq_lens_host,
            seq_slots_cuda=seq_slots_cuda,
            seq_lens_cuda=seq_lens_cuda,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            req_num_steps=sampling_requests_metadata.req_num_steps,
            token_dtype=new_tokens_cuda.dtype,
            return_log_probs=return_log_probs,
        )

        # Fill results into output buffers
        new_tokens_host = self._unbatch_sampling_results(
            batched_sampling_result,
            new_tokens_cuda=new_tokens_cuda,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            seq_slots=seq_slots_host,
            seq_slots_cuda=seq_slots_cuda,
        )

        if return_log_probs:
            self._log_probs._process_logprobs(
                batched_sampling_result,
                logits_cuda=logits_cuda,
                new_tokens_cuda=new_tokens_cuda,
                seq_slots=seq_slots_host,
                requests=sampling_requests,
                req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens_output,
            )

        # NB: update_requests syncs w/ device computation and async D2H copies
        return (
            sampling_requests,
            seq_slots_host,
            seq_lens_host,
            seq_slots_cuda,
            seq_lens_cuda,
            new_tokens_host,
            False,
        )

    @override
    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        params = _request_get_sampling_params(request)
        temperature = params.temperature
        top_p = params.top_p
        top_k = params.top_k
        min_p = params.min_p

        # Do not request draft probs when sampling is greedy.
        return not SamplingParams.params_imply_greedy_decoding(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=self._use_beam_search,
            min_p=min_p,
        )
