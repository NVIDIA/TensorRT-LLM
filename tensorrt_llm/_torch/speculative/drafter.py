from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Dict, List, Optional, final

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, get_draft_token_length
from ..pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

    def __init__(self,
                 max_draft_len: int = None,
                 max_total_draft_tokens: int = None,
                 max_concurrency: Optional[int] = None,
                 draft_len_schedule: Optional[Dict[int, int]] = None) -> None:
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self._static_max_total_draft_tokens = max_total_draft_tokens
        self.max_concurrency = max_concurrency
        self.draft_len_schedule = draft_len_schedule

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.

        Args:
            scheduled_requests: The scheduled requests for this iteration
        """
        raise NotImplementedError

    @final
    def should_use_spec_decode(self, requests: List[LlmRequest],
                               max_batch_size: int, max_num_tokens: int,
                               max_total_draft_tokens: int) -> bool:
        """
        You probably don't want to override this. ModelEngine
        assumes that speculation is always on if max_concurrency
        is not specified by the user's spec config.
        """

        # Inputs typically validated upstream: max_batch_size>0, max_num_tokens>0, max_total_draft_tokens>=0

        if self.max_concurrency is None:
            return True

        # Defensive guards; keep behavior explicit for zero/empty cases
        if not requests or max_batch_size <= 0 or max_num_tokens <= 0:
            return False

        tokens_per_request = 1 + max_total_draft_tokens
        token_cap = max_num_tokens // tokens_per_request
        if token_cap <= 0:
            return False

        num_effective_requests = min(len(requests), max_batch_size, token_cap)
        return num_effective_requests <= self.max_concurrency

    # Drafters that use TorchSampler (NGram, two-model) compute py_rewind_len
    # from len(py_draft_tokens), which includes padding.  They must set this
    # to True so that extend_capacity_for_tokens is called after padding.
    # One-model drafters (MTP / Eagle3 / SA) use SpecSamplerBase which
    # computes rewind from runtime_draft_len, so padding is harmless.
    _needs_padding_kv_extension: bool = False

    @final
    def pad_draft_tokens_for_cuda_graph(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """Pad draft tokens to the static max for CUDA graph compatibility.

        When draft_len_schedule reduces draft length below the static max,
        prepare_resources allocates KV cache capacity for the shorter length.
        After padding restores py_draft_tokens to the static max, drafters
        whose sampler uses the padded length for rewind need the KV cache
        capacity extended to match.  The KV cache manager itself computes
        the exact delta from its internal tracking.
        """
        pad_to = self._static_max_total_draft_tokens
        for req in scheduled_requests.generation_requests:
            num_draft_tokens = get_draft_token_length(req)
            req.py_draft_tokens.extend(
                0 for _ in range(pad_to - num_draft_tokens))

        if self._needs_padding_kv_extension and resource_manager is not None:
            kv_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.KV_CACHE_MANAGER)
            draft_kv_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.DRAFT_KV_CACHE_MANAGER)
            for req in scheduled_requests.generation_requests:
                if kv_mgr is not None:
                    kv_mgr.extend_capacity_for_tokens(req)
                if draft_kv_mgr is not None:
                    draft_kv_mgr.extend_capacity_for_tokens(req)

    def get_draft_len_for_batch_size(self, batch_size: int) -> int:
        """
        Get the appropriate draft length for the given batch size using binary search.
        Args:
            batch_size: Current batch size (has been sorted by config validator)
        Returns:
            The draft length to use for this batch size
        """

        # Binary search to find the largest threshold <= batch_size
        # draft_len_schedule is already sorted by config validator
        thresholds = list(self.draft_len_schedule.keys())

        # bisect_right finds where to insert batch_size to keep list sorted
        # The element before insertion point is the largest threshold <= batch_size
        idx = bisect_right(thresholds, batch_size)

        if idx == 0:
            # batch_size is smaller than smallest threshold (batch_size smaller than 1)
            # This shouldn't happen in practice, but handle defensively
            logger.warning(
                f"get_draft_len_for_batch_size called with batch_size={batch_size} < 1. "
                f"This is unexpected. Disabling speculation (returning draft_len=0)."
            )
            return 0

        # Return draft_len for the largest threshold <= batch_size
        threshold = thresholds[idx - 1]
        return self.draft_len_schedule[threshold]

    def update_max_total_draft_tokens(self,
                                      new_max_total_draft_tokens: int) -> None:
        """
        Used when draft_len_schedule is provided in spec_config (dynamic draft length based on runtime batch size is enabled)
        Update max_total_draft_tokens in drafter and propagate to any dependent components.
        Subclasses can override to propagate to their resource managers if needed.
        Args:
            new_max_total_draft_tokens: The new max total draft tokens
        """
        self.max_total_draft_tokens = new_max_total_draft_tokens
        self.max_draft_len = new_max_total_draft_tokens
