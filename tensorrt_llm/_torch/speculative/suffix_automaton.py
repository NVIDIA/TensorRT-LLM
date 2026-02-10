"""
Suffix Automaton Speculative Decoding Module

This module provides a native implementation of the suffix automaton-based
speculative decoding approach, originally developed by Baseten (sa_spec).

The suffix automaton is a compact state machine that recognizes all suffixes
of a string. It's used to find longest patterns in previously generated tokens
to predict future tokens, which can boost acceptance rates in speculative decoding
by up to 40% when combined with MTP (Multi-Token Prediction).

Key component:
- SuffixAutomatonManager: Manages per-request suffix automaton states and
  integrates with TRT-LLM's resource management via BaseResourceManager.

This module requires the native C++/CUDA kernel for GPU-native, CUDA graph
compatible operations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests

logger = logging.getLogger(__name__)

# Import native suffix automaton bindings (required)
_sa_native = None
try:
    from tensorrt_llm.bindings.internal import suffix_automaton as _sa_native

    logger.info("Native suffix automaton kernel available")
except ImportError:
    logger.error(
        "Native suffix automaton kernel not available. "
        "The suffix automaton module requires the native C++/CUDA kernel."
    )


@dataclass
class SAConfig:
    """Configuration for suffix automaton speculative decoding."""

    # Maximum sequence length supported (runtime configurable)
    max_seq_len: int = 262144

    # Maximum number of concurrent requests
    max_slots: int = 256

    # Minimum match length to use SA draft tokens
    threshold: int = 4


class SuffixAutomatonManager(BaseResourceManager):
    """
    Manages suffix automaton states for multiple requests.

    This class handles:
    - Creating and destroying SA states per request
    - Copying SA states from host (pinned) to device memory
    - Batch operations for efficient GPU utilization
    - Integration with TRT-LLM's resource management (BaseResourceManager)

    SA states are built on CPU using native code, then copied to GPU during
    prepare(). The extend() and extend_ngram() methods run entirely on GPU
    and are CUDA graph compatible.

    Used as the resource manager for both NGram and MTP+SA speculative decoding.
    """

    def __init__(
        self,
        config,
        max_num_requests: int,
        max_seq_len: int = 262144,
    ):
        if _sa_native is None:
            raise RuntimeError(
                "Native suffix automaton kernel is required but not available. "
                "Please ensure the native bindings are properly built."
            )

        # SA configuration
        sa_config = (
            config
            if isinstance(config, SAConfig)
            else SAConfig(
                max_seq_len=max_seq_len,
                max_slots=max_num_requests,
                threshold=getattr(config, "sa_spec_threshold", 4),
            )
        )

        self.config = sa_config
        self.max_num_requests = max_num_requests
        self.max_seq_len = sa_config.max_seq_len

        # Calculate per-state size based on max_seq_len
        self.state_size = _sa_native.get_state_size(self.max_seq_len)

        # Log memory usage
        total_memory_mb = max_num_requests * self.state_size / 1024 / 1024
        logger.info(
            f"Allocating {max_num_requests} SA slots with max_seq_len={self.max_seq_len} "
            f"({self.state_size / 1024 / 1024:.1f} MB/slot, {total_memory_mb:.1f} MB total)"
        )

        # Request ID -> slot index mapping
        self._request_to_slot: Dict[int, int] = {}

        # Free slots for reuse
        self._free_slots: List[int] = list(range(max_num_requests))

        # Host-side SA states as pinned memory tensors
        self._host_states_native: Dict[int, torch.Tensor] = {}

        # GPU workspace for SA states
        self._gpu_slots: Optional[torch.Tensor] = None

        # Track which slots need to be copied to GPU
        self._pending_copies: Set[int] = set()

        # GPU output buffers
        self._workspace_allocated = False
        self._gpu_match_len: Optional[torch.Tensor] = None
        self._gpu_draft_tokens: Optional[torch.Tensor] = None
        self._gpu_batch_indices: Optional[torch.Tensor] = None

        # Track which requests have been initialized (for prepare_resources)
        self._initialized_requests: Set[int] = set()

    def _ensure_workspace(self, max_draft_len: int):
        """Ensure GPU workspace is allocated."""
        if not self._workspace_allocated:
            self._gpu_match_len = torch.zeros(
                (self.max_num_requests,), dtype=torch.int32, device="cuda"
            )
            self._gpu_draft_tokens = torch.zeros(
                (self.max_num_requests, max_draft_len), dtype=torch.int32, device="cuda"
            )
            self._gpu_batch_indices = torch.zeros(
                (self.max_num_requests,), dtype=torch.int32, device="cuda"
            )

            # Allocate GPU workspace for SA states with dynamic size
            self._gpu_slots = _sa_native.allocate_workspace(self.max_num_requests, self.max_seq_len)

            self._workspace_allocated = True

    # --- Core SA operations ---

    def add_request(self, request_id: int, context_tokens: List[int]):
        """
        Add a new request and build its initial suffix automaton from context.

        Args:
            request_id: Unique identifier for the request
            context_tokens: The initial context tokens to build the SA from
        """
        if request_id in self._request_to_slot:
            # Request already exists, rebuild its state
            self._host_states_native[request_id] = _sa_native.build_automaton_host(
                context_tokens, self.max_seq_len
            )
            self._pending_copies.add(request_id)
            return

        if not self._free_slots:
            raise RuntimeError("No free slots available for new request")

        # Allocate a slot
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot

        # Build SA state on host using native code
        self._host_states_native[request_id] = _sa_native.build_automaton_host(
            context_tokens, self.max_seq_len
        )
        self._pending_copies.add(request_id)

    def remove_request(self, request_id: int):
        """Remove a request and free its resources."""
        if request_id not in self._request_to_slot:
            return

        slot = self._request_to_slot.pop(request_id)
        self._free_slots.append(slot)

        self._host_states_native.pop(request_id, None)
        self._pending_copies.discard(request_id)
        self._initialized_requests.discard(request_id)

        # Clear the GPU slot
        if self._gpu_slots is not None:
            _sa_native.clear_slot(self._gpu_slots, slot, self.max_seq_len)

    def prepare(self, request_ids: List[int], max_draft_len: int):
        """
        Prepare batch indices for the upcoming extend() call.

        This copies pending host states to GPU.

        Args:
            request_ids: List of request IDs in the current batch
            max_draft_len: Maximum draft length for output buffers
        """
        self._ensure_workspace(max_draft_len)

        # Copy pending host states to GPU
        if self._pending_copies:
            for rid in list(self._pending_copies):
                if rid in self._host_states_native and rid in self._request_to_slot:
                    slot = self._request_to_slot[rid]
                    _sa_native.copy_state_to_slot(
                        self._host_states_native[rid],
                        self._gpu_slots,
                        slot,
                        self.max_seq_len,
                    )
            self._pending_copies.clear()

        # Prepare batch indices
        batch_indices = torch.tensor(
            [self._request_to_slot.get(rid, 0) for rid in request_ids],
            dtype=torch.int32,
            pin_memory=True,
        )

        num_requests = len(request_ids)
        self._gpu_batch_indices[:num_requests].copy_(batch_indices, non_blocking=True)

    def extend(
        self,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        max_draft_len: int,
    ) -> tuple:
        """
        Extend suffix automaton states with accepted tokens and get draft tokens.

        This is the main entry point called during generation. It:
        1. Extends each request's SA with newly accepted tokens
        2. Performs lookup to find longest matching suffix
        3. Returns draft tokens based on the match

        CUDA graph compatible.

        Args:
            request_ids: List of request IDs in the batch
            accepted_tokens: [batch_size, max_draft_len + 1] accepted token tensor
            num_accepted_tokens: [batch_size] number of accepted tokens per request
            max_draft_len: Maximum draft length

        Returns:
            Tuple of (match_len, draft_tokens) tensors
        """
        self._ensure_workspace(max_draft_len)

        batch_size = len(request_ids)

        # Native GPU kernel - CUDA graph compatible
        match_len = self._gpu_match_len[:batch_size]
        draft_tokens = self._gpu_draft_tokens[:batch_size, :max_draft_len]

        # Ensure accepted_tokens is contiguous and int32
        if accepted_tokens.dtype != torch.int32:
            accepted_tokens = accepted_tokens.to(torch.int32)
        if num_accepted_tokens.dtype != torch.int32:
            num_accepted_tokens = num_accepted_tokens.to(torch.int32)

        _sa_native.invoke_extend(
            batch_size,
            max_draft_len,
            self.max_num_requests,
            self.max_seq_len,
            self._gpu_slots,
            self._gpu_batch_indices[:batch_size],
            match_len,
            draft_tokens,
            accepted_tokens,
            num_accepted_tokens,
        )

        return match_len, draft_tokens

    def extend_ngram(
        self,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        max_draft_len: int,
        max_ngram_size: int = -1,
    ) -> tuple:
        """
        Extend suffix automaton states with accepted tokens and get draft tokens
        using ngram matching.

        This method supports both longest match mode and fixed-size ngram matching.
        CUDA graph compatible.

        Args:
            request_ids: List of request IDs in the batch
            accepted_tokens: [batch_size, max_draft_len + 1] accepted token tensor
            num_accepted_tokens: [batch_size] number of accepted tokens per request
            max_draft_len: Maximum draft length
            max_ngram_size: Max ngram size for matching:
                           -1 = longest match mode (default)
                           >0 = try ngram sizes from max down to 1

        Returns:
            Tuple of (match_len, draft_tokens) tensors
        """
        self._ensure_workspace(max_draft_len)

        batch_size = len(request_ids)

        # Native GPU kernel - CUDA graph compatible
        match_len = self._gpu_match_len[:batch_size]
        draft_tokens = self._gpu_draft_tokens[:batch_size, :max_draft_len]

        # Ensure accepted_tokens is contiguous and int32
        if accepted_tokens.dtype != torch.int32:
            accepted_tokens = accepted_tokens.to(torch.int32)
        if num_accepted_tokens.dtype != torch.int32:
            num_accepted_tokens = num_accepted_tokens.to(torch.int32)

        _sa_native.invoke_extend_ngram(
            batch_size,
            max_draft_len,
            max_ngram_size,
            self.max_num_requests,
            self.max_seq_len,
            self._gpu_slots,
            self._gpu_batch_indices[:batch_size],
            match_len,
            draft_tokens,
            accepted_tokens,
            num_accepted_tokens,
        )

        return match_len, draft_tokens

    # --- BaseResourceManager interface ---

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        """Prepare SA states for new context requests."""
        for req in scheduled_batch.context_requests:
            if req.is_first_context_chunk:
                if req.request_id not in self._initialized_requests:
                    context_tokens = req.get_tokens(0)
                    self.add_request(req.request_id, context_tokens)
                    self._initialized_requests.add(req.request_id)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        """Update resources after forward pass (no-op for SA)."""
        pass

    def free_resources(self, request: LlmRequest):
        """Free SA state for completed request."""
        self.remove_request(request.request_id)

    def add_dummy_requests(self, request_ids: List[int]):
        """Add dummy requests for CUDA graph warmup."""
        for rid in request_ids:
            self.add_request(rid, [1])  # Dummy token

    def shutdown(self):
        """Clean up all resources."""
        self._request_to_slot.clear()
        self._free_slots = list(range(self.max_num_requests))

        self._host_states_native.clear()
        self._pending_copies.clear()
        self._initialized_requests.clear()
        self._gpu_slots = None

        # Free GPU memory
        self._gpu_match_len = None
        self._gpu_draft_tokens = None
        self._gpu_batch_indices = None
        self._workspace_allocated = False

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0


def is_native_available() -> bool:
    """Check if native suffix automaton kernel is available."""
    return _sa_native is not None
