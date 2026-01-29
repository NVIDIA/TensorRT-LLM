"""
Suffix Automaton Speculative Decoding Module

This module provides a native implementation of the suffix automaton-based
speculative decoding approach, originally developed by Baseten (sa_spec).

The suffix automaton is a compact state machine that recognizes all suffixes
of a string. It's used to find longest patterns in previously generated tokens
to predict future tokens, which can boost acceptance rates in speculative decoding
by up to 40% when combined with MTP (Multi-Token Prediction).

Key components:
- SuffixAutomatonManager: Manages per-request suffix automaton states
- SAResourceManager: Integrates with TRT-LLM's resource management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.scheduler import ScheduledRequests

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig


@dataclass
class SAConfig:
    """Configuration for suffix automaton speculative decoding."""

    # Maximum sequence length supported
    max_seq_len: int = 262144

    # Maximum number of concurrent requests
    max_slots: int = 256

    # Minimum match length to use SA draft tokens
    threshold: int = 4


class SuffixAutomatonState:
    """
    Represents the state of a suffix automaton for a single request.

    This is a Python-side placeholder that will be replaced by the native
    C++ SuffixAutomaton when pybind bindings are available.
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokens: List[int] = []

    def extend(self, token: int):
        """Extend the automaton with a new token."""
        self.tokens.append(token)

    def extend_batch(self, tokens: List[int]):
        """Extend the automaton with multiple tokens."""
        self.tokens.extend(tokens)

    def lookup(self) -> Optional[tuple]:
        """
        Find the longest suffix that appears earlier in the text.

        Returns:
            Optional tuple of (position, length) if a match is found,
            None otherwise.
        """
        if len(self.tokens) < 2:
            return None

        # Simple O(n^2) implementation for now
        # Will be replaced by efficient C++ implementation
        text = self.tokens
        n = len(text)

        best_pos = None
        best_len = 0

        # Look for the longest suffix ending before the current position
        for suffix_len in range(1, n):
            suffix_start = n - suffix_len
            suffix = text[suffix_start:]

            # Search for this suffix in the earlier text
            for pos in range(n - suffix_len):
                if text[pos:pos + suffix_len] == suffix:
                    if suffix_len > best_len:
                        best_len = suffix_len
                        best_pos = pos + suffix_len  # Position after the match
                    break

        if best_pos is not None:
            return (best_pos, best_len)
        return None

    def get_draft_tokens(self, start_pos: int, num_tokens: int) -> List[int]:
        """Get draft tokens starting from a position."""
        end_pos = min(start_pos + num_tokens, len(self.tokens))
        return self.tokens[start_pos:end_pos]

    def clear(self):
        """Clear the automaton state."""
        self.tokens = []


class SuffixAutomatonManager:
    """
    Manages suffix automaton states for multiple requests.

    This class handles:
    - Creating and destroying SA states per request
    - Copying SA states from host (pinned) to device memory
    - Batch operations for efficient GPU utilization
    """

    def __init__(self, config: SAConfig, max_num_requests: int):
        self.config = config
        self.max_num_requests = max_num_requests

        # Host-side SA states (one per request)
        self._host_states: Dict[int, SuffixAutomatonState] = {}

        # Free slots for reuse
        self._free_slots: List[int] = list(range(max_num_requests))

        # Request ID -> slot index mapping
        self._request_to_slot: Dict[int, int] = {}

        # GPU workspace - will be allocated when needed
        # For now we use CPU tensors as placeholders
        self._workspace_allocated = False
        self._gpu_match_len: Optional[torch.Tensor] = None
        self._gpu_draft_tokens: Optional[torch.Tensor] = None
        self._gpu_batch_indices: Optional[torch.Tensor] = None

    def _ensure_workspace(self, max_draft_len: int):
        """Ensure GPU workspace is allocated."""
        if not self._workspace_allocated:
            self._gpu_match_len = torch.zeros((self.max_num_requests,),
                                              dtype=torch.int32,
                                              device='cuda')
            self._gpu_draft_tokens = torch.zeros(
                (self.max_num_requests, max_draft_len),
                dtype=torch.int32,
                device='cuda')
            self._gpu_batch_indices = torch.zeros((self.max_num_requests,),
                                                  dtype=torch.int32,
                                                  device='cuda')
            self._workspace_allocated = True

    def add_request(self, request_id: int, context_tokens: List[int]):
        """
        Add a new request and build its initial suffix automaton from context.

        Args:
            request_id: Unique identifier for the request
            context_tokens: The initial context tokens to build the SA from
        """
        if request_id in self._request_to_slot:
            # Request already exists, just update its tokens
            state = self._host_states[request_id]
            state.clear()
            state.extend_batch(context_tokens)
            return

        if not self._free_slots:
            raise RuntimeError("No free slots available for new request")

        # Allocate a slot
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot

        # Create and populate the SA state
        state = SuffixAutomatonState(self.config.max_seq_len)
        state.extend_batch(context_tokens)
        self._host_states[request_id] = state

    def remove_request(self, request_id: int):
        """Remove a request and free its resources."""
        if request_id not in self._request_to_slot:
            return

        slot = self._request_to_slot.pop(request_id)
        self._free_slots.append(slot)

        if request_id in self._host_states:
            del self._host_states[request_id]

    def prepare(self, request_ids: List[int], max_draft_len: int):
        """
        Prepare batch indices for the upcoming extend() call.

        This copies the request_id -> slot mapping to GPU.

        Args:
            request_ids: List of request IDs in the current batch
            max_draft_len: Maximum draft length for output buffers
        """
        self._ensure_workspace(max_draft_len)

        batch_indices = torch.tensor(
            [self._request_to_slot.get(rid, 0) for rid in request_ids],
            dtype=torch.int32,
            pin_memory=True)

        num_requests = len(request_ids)
        self._gpu_batch_indices[:num_requests].copy_(batch_indices,
                                                     non_blocking=True)

    def extend(self, request_ids: List[int], accepted_tokens: torch.Tensor,
               num_accepted_tokens: torch.Tensor,
               max_draft_len: int) -> tuple:
        """
        Extend suffix automaton states with accepted tokens and get draft tokens.

        This is the main entry point called during generation. It:
        1. Extends each request's SA with newly accepted tokens
        2. Performs lookup to find longest matching suffix
        3. Returns draft tokens based on the match

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
        match_len = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
        draft_tokens = torch.zeros((batch_size, max_draft_len),
                                   dtype=torch.int32,
                                   device='cuda')

        # Process each request
        accepted_tokens_cpu = accepted_tokens.cpu()
        num_accepted_cpu = num_accepted_tokens.cpu()

        for i, rid in enumerate(request_ids):
            if rid not in self._host_states:
                continue

            state = self._host_states[rid]
            num_new = num_accepted_cpu[i].item()

            # Extend with accepted tokens
            for j in range(num_new):
                token = accepted_tokens_cpu[i, j].item()
                state.extend(token)

            # Lookup longest suffix
            result = state.lookup()
            if result is not None:
                pos, length = result
                match_len[i] = length

                # Get draft tokens
                drafts = state.get_draft_tokens(pos, max_draft_len)
                for j, tok in enumerate(drafts):
                    draft_tokens[i, j] = tok

        return match_len, draft_tokens

    def shutdown(self):
        """Clean up all resources."""
        self._host_states.clear()
        self._request_to_slot.clear()
        self._free_slots = list(range(self.max_num_requests))

        # Free GPU memory
        self._gpu_match_len = None
        self._gpu_draft_tokens = None
        self._gpu_batch_indices = None
        self._workspace_allocated = False


class SAResourceManager(BaseResourceManager):
    """
    Resource manager for suffix automaton speculative decoding.

    Integrates with TRT-LLM's resource management system to:
    - Allocate SA states for new requests
    - Free SA states when requests complete
    - Prepare SA states for batch processing
    """

    def __init__(self, config: "MTPDecodingConfig", max_num_requests: int):
        self.max_num_requests = max_num_requests
        self.slot_manager = SlotManager(max_num_requests)

        # SA configuration
        sa_config = SAConfig(
            max_seq_len=262144,
            max_slots=max_num_requests,
            threshold=getattr(config, 'sa_spec_threshold', 4))

        # Initialize the SA manager
        self.sa_manager = SuffixAutomatonManager(sa_config, max_num_requests)

        # Track which requests have been initialized
        self._initialized_requests: Set[int] = set()

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        """Prepare SA states for new context requests."""
        context_batch = scheduled_batch.context_requests

        for req in context_batch:
            if req.is_first_context_chunk:
                slot_id = self.slot_manager.add_slot(req.request_id)

                # Build initial SA from context tokens
                if req.request_id not in self._initialized_requests:
                    context_tokens = req.get_tokens(0)
                    self.sa_manager.add_request(req.request_id, context_tokens)
                    self._initialized_requests.add(req.request_id)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        """Update resources after forward pass (no-op for SA)."""
        pass

    def free_resources(self, request: LlmRequest):
        """Free SA state for completed request."""
        self.slot_manager.remove_slot(request.request_id)
        self.sa_manager.remove_request(request.request_id)
        self._initialized_requests.discard(request.request_id)

    def add_dummy_requests(self, request_ids: List[int]):
        """Add dummy requests for CUDA graph warmup."""
        for rid in request_ids:
            self.slot_manager.add_slot(rid)
            self.sa_manager.add_request(rid, [1])  # Dummy token

    def shutdown(self):
        """Clean up all resources."""
        self.slot_manager.shutdown()
        self.sa_manager.shutdown()
        self._initialized_requests.clear()

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0

    def prepare_batch(self, request_ids: List[int], max_draft_len: int):
        """Prepare batch indices for extend operation."""
        self.sa_manager.prepare(request_ids, max_draft_len)

    def extend_and_get_drafts(self, request_ids: List[int],
                              accepted_tokens: torch.Tensor,
                              num_accepted_tokens: torch.Tensor,
                              max_draft_len: int) -> tuple:
        """
        Extend SA states and get draft tokens.

        Returns:
            Tuple of (match_len, draft_tokens) tensors
        """
        return self.sa_manager.extend(request_ids, accepted_tokens,
                                      num_accepted_tokens, max_draft_len)


# Module-level interface for compatibility with existing sa_spec usage
_global_sa_manager: Optional[SuffixAutomatonManager] = None
_seen_requests: Set[int] = set()


def init(max_num_requests: int = 256, max_seq_len: int = 262144):
    """Initialize the global SA manager."""
    global _global_sa_manager, _seen_requests
    config = SAConfig(max_seq_len=max_seq_len, max_slots=max_num_requests)
    _global_sa_manager = SuffixAutomatonManager(config, max_num_requests)
    _seen_requests = set()


def add_request(request_id: int, context_tokens: List[int]):
    """Add a request with its context tokens."""
    global _global_sa_manager
    if _global_sa_manager is None:
        init()
    _global_sa_manager.add_request(request_id, context_tokens)


def prepare(request_ids: List[int], max_draft_len: int = 8):
    """Prepare batch for processing."""
    global _global_sa_manager
    if _global_sa_manager is None:
        return
    _global_sa_manager.prepare(request_ids, max_draft_len)


def extend(match_len_out: torch.Tensor, draft_tokens_out: torch.Tensor,
           accepted_tokens: torch.Tensor, num_accepted_tokens: torch.Tensor):
    """
    Extend SA states and populate output tensors.

    This matches the interface of the external sa_spec.extend() function.
    """
    global _global_sa_manager
    if _global_sa_manager is None:
        match_len_out.zero_()
        return

    # Get batch size from input tensors
    batch_size = accepted_tokens.shape[0]
    max_draft_len = draft_tokens_out.shape[1]

    # For now, use placeholder request IDs
    # In the actual integration, these will come from SpecMetadata
    request_ids = list(range(batch_size))

    match_len, draft_tokens = _global_sa_manager.extend(request_ids,
                                                        accepted_tokens,
                                                        num_accepted_tokens,
                                                        max_draft_len)

    match_len_out.copy_(match_len)
    draft_tokens_out.copy_(draft_tokens)


def remove_request(request_id: int):
    """Remove a completed request."""
    global _global_sa_manager
    if _global_sa_manager is not None:
        _global_sa_manager.remove_request(request_id)


def shutdown():
    """Shutdown the global SA manager."""
    global _global_sa_manager, _seen_requests
    if _global_sa_manager is not None:
        _global_sa_manager.shutdown()
        _global_sa_manager = None
    _seen_requests = set()
