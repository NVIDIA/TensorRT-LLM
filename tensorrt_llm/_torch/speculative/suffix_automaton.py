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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.internal import suffix_automaton as _sa_native

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests

logger = logging.getLogger(__name__)


_DEFAULT_POOL_SIZE = 64


@dataclass
class SAConfig:
    """Configuration for suffix automaton speculative decoding."""

    # Maximum sequence length supported (runtime configurable)
    max_seq_len: int = 262144

    # Maximum number of concurrent requests
    max_slots: int = 256

    # Minimum match length to use SA draft tokens
    threshold: int = 4

    # Enable global pool search across all active SA states
    enable_global_pool: bool = False

    # Explicit global pool size (None = use default heuristic)
    global_pool_size: Optional[int] = None

    @property
    def effective_pool_size(self) -> int:
        """Actual number of SA slots to allocate.

        When global pool is enabled but no explicit size is given,
        defaults to max(_DEFAULT_POOL_SIZE, max_slots) — a fixed-size
        pool independent of batch size, floored to at least max_slots
        so there's always room for the full batch.
        """
        if self.global_pool_size is not None:
            return self.global_pool_size
        if self.enable_global_pool:
            return max(_DEFAULT_POOL_SIZE, self.max_slots)
        return self.max_slots


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

        from tensorrt_llm.llmapi.llm_args import SADecodingConfig, SAEnhancerConfig

        # SA configuration
        if isinstance(config, SAConfig):
            sa_config = config
        elif isinstance(config, SAEnhancerConfig):
            sa_config = SAConfig(
                max_seq_len=max_seq_len,
                max_slots=max_num_requests,
                threshold=config.threshold,
                enable_global_pool=config.enable_global_pool,
            )
        elif isinstance(config, SADecodingConfig):
            sa_config = SAConfig(
                max_seq_len=max_seq_len,
                max_slots=max_num_requests,
                enable_global_pool=config.enable_global_pool,
                global_pool_size=config.global_pool_size,
            )
        else:
            raise TypeError(
                f"SuffixAutomatonManager received unsupported config type "
                f"{type(config).__name__}. Expected SAConfig, SAEnhancerConfig,"
                f" or SADecodingConfig."
            )

        self.config = sa_config
        self.max_num_requests = max_num_requests
        self.max_seq_len = sa_config.max_seq_len
        self.enable_global_pool = sa_config.enable_global_pool

        # Pool sizing: effective_pool_size returns max_num_requests when
        # global pool is off, or max(64, max_num_requests) / explicit
        # value when on. All slot-indexed sizing uses pool_size.
        self.pool_size = sa_config.effective_pool_size
        if self.pool_size < max_num_requests:
            raise ValueError(
                f"global_pool_size ({self.pool_size}) must be >= "
                f"max_batch_size ({max_num_requests})"
            )

        # Calculate per-state size based on max_seq_len
        self.state_size = _sa_native.get_state_size(self.max_seq_len)

        logger.info(
            f"SA pool: {self.pool_size} slots "
            f"({self.pool_size - max_num_requests} retained capacity, "
            f"{self.pool_size * self.state_size / 1024 / 1024:.1f} MB total)"
        )

        # Request ID -> slot index mapping
        self._request_to_slot: Dict[int, int] = {}

        # Free slots now range over the full pool
        self._free_slots: List[int] = list(range(self.pool_size))

        # Host-side SA states as pinned memory tensors
        self._host_states_native: Dict[int, torch.Tensor] = {}

        # GPU workspace for SA states
        self._gpu_slots: Optional[torch.Tensor] = None

        # Track which slots need to be copied to GPU
        self._pending_copies: Set[int] = set()

        # GPU output buffers
        self._workspace_allocated = False
        self._allocated_max_draft_len: int = 0  # Track allocated draft length
        self._gpu_match_len: Optional[torch.Tensor] = None
        self._gpu_draft_tokens: Optional[torch.Tensor] = None
        self._gpu_batch_indices: Optional[torch.Tensor] = None

        # Global pool buffers (allocated lazily in _ensure_workspace)
        self._gpu_active_slot_mask: Optional[torch.Tensor] = None
        self._gpu_match_slot: Optional[torch.Tensor] = None
        self._pending_mask_updates: Dict[int, int] = {}

        # Track which requests have been initialized (for prepare_resources)
        self._initialized_requests: Set[int] = set()

        # Dummy slot lives right after the pool — always use pool_size,
        # not max_num_requests, so dummies never collide with pool slots.
        self._dummy_slot_index: int = self.pool_size
        self._dummy_request_ids: Set[int] = set()

        # Pre-allocated CPU staging buffers for prepare() to avoid
        # repeated pinned-memory allocation every round.
        self._cpu_batch_indices: Optional[torch.Tensor] = None
        self._cpu_nondummy_mask: Optional[torch.Tensor] = None

        # Retained slots: completed requests whose SA states remain in the
        # pool for cross-request search. OrderedDict preserves insertion
        # (completion) order for FIFO eviction.
        #   key: slot index
        #   value: original request_id (for debugging/logging)
        self._retained_slots: OrderedDict[int, int] = OrderedDict()

        # Track which slots have active (in-flight) requests.
        self._active_slots: Set[int] = set()

    def _ensure_workspace(self, max_draft_len: int):
        """Ensure GPU workspace is allocated with sufficient capacity.

        Args:
            max_draft_len: Required maximum draft length for output buffers.

        Raises:
            ValueError: If called with max_draft_len larger than previously allocated.
        """
        if not self._workspace_allocated:
            # Batch-indexed buffers: sized to max_num_requests (max batch size)
            self._gpu_match_len = torch.zeros(
                (self.max_num_requests,), dtype=torch.int32, device="cuda"
            )
            self._gpu_draft_tokens = torch.zeros(
                (self.max_num_requests, max_draft_len), dtype=torch.int32, device="cuda"
            )
            self._gpu_batch_indices = torch.zeros(
                (self.max_num_requests,), dtype=torch.int32, device="cuda"
            )
            # Mask: 1 for real requests, 0 for dummies. Populated by
            # prepare() (outside CUDA graph) and used by extend() (inside
            # CUDA graph) to zero out dummy entries without Python control
            # flow that would break graph capture.
            self._gpu_nondummy_mask = torch.ones(
                (self.max_num_requests,), dtype=torch.int32, device="cuda"
            )

            # Slot-indexed buffers: pool_size + 1 (pool slots + dummy slot).
            # When global pool is off, pool_size == max_num_requests.
            self._gpu_slots = _sa_native.allocate_workspace(self.pool_size + 1, self.max_seq_len)

            # Global pool buffers
            if self.enable_global_pool:
                self._gpu_active_slot_mask = torch.zeros(
                    (self.pool_size + 1,), dtype=torch.int32, device="cuda"
                )
                self._gpu_match_slot = torch.zeros(
                    (self.max_num_requests,), dtype=torch.int32, device="cuda"
                )

            self._allocated_max_draft_len = max_draft_len
            self._workspace_allocated = True
        elif max_draft_len > self._allocated_max_draft_len:
            # Subsequent call with larger draft length - need to re-allocate draft tokens buffer
            logger.warning(
                f"SuffixAutomatonManager: Re-allocating _gpu_draft_tokens buffer "
                f"(old max_draft_len={self._allocated_max_draft_len}, new={max_draft_len})"
            )
            self._gpu_draft_tokens = torch.zeros(
                (self.max_num_requests, max_draft_len), dtype=torch.int32, device="cuda"
            )
            self._allocated_max_draft_len = max_draft_len

    # --- Core SA operations ---

    def _allocate_slot(self) -> int:
        """Allocate a slot, evicting the oldest retained request if needed."""
        if self._free_slots:
            return self._free_slots.pop()

        if self._retained_slots:
            slot, old_rid = self._retained_slots.popitem(last=False)  # FIFO
            if self._gpu_slots is not None:
                _sa_native.clear_slot(self._gpu_slots, slot, self.max_seq_len)
            self._pending_mask_updates[slot] = 0
            logger.debug(
                f"Evicted retained SA slot {slot} (request {old_rid}) to make room for new request"
            )
            return slot

        raise RuntimeError(
            f"No free or retained slots available. pool_size={self.pool_size}, "
            f"active={len(self._active_slots)}"
        )

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

        slot = self._allocate_slot()
        self._request_to_slot[request_id] = slot
        self._active_slots.add(slot)

        # Build SA state on host using native code
        self._host_states_native[request_id] = _sa_native.build_automaton_host(
            context_tokens, self.max_seq_len
        )
        self._pending_copies.add(request_id)

        if self.enable_global_pool:
            self._pending_mask_updates[slot] = 1

    def remove_request(self, request_id: int):
        """Remove a request, retaining its SA state for cross-request search
        when global pool is enabled and the pool has retention capacity."""
        if request_id not in self._request_to_slot:
            return

        slot = self._request_to_slot.pop(request_id)

        if request_id in self._dummy_request_ids:
            self._dummy_request_ids.discard(request_id)
            # Dummy slot is reserved; never retain or free.
            return

        self._active_slots.discard(slot)

        # If the GPU copy was never flushed, the slot contains stale data —
        # skip retention and free immediately to avoid searching garbage.
        stale = request_id in self._pending_copies

        if self.enable_global_pool and self.pool_size > self.max_num_requests and not stale:
            # Retain: keep SA state alive for cross-request search.
            # Active mask stays ON — the slot is still searchable.
            self._retained_slots[slot] = request_id
        else:
            # Free immediately
            if self.enable_global_pool:
                self._pending_mask_updates[slot] = 0
            self._free_slots.append(slot)
            if self._gpu_slots is not None:
                _sa_native.clear_slot(self._gpu_slots, slot, self.max_seq_len)

        self._host_states_native.pop(request_id, None)
        self._pending_copies.discard(request_id)
        self._initialized_requests.discard(request_id)

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

        # Flush deferred active-slot-mask updates.  add_request/remove_request
        # queue updates because the GPU tensor may not exist yet at that point;
        # here _ensure_workspace has already run so the tensor is available.
        if self._pending_mask_updates and self._gpu_active_slot_mask is not None:
            for slot, value in self._pending_mask_updates.items():
                self._gpu_active_slot_mask[slot] = value
            self._pending_mask_updates.clear()

        # Map each request ID to its slot. Unknown IDs (e.g. CUDA graph
        # warmup dummies that skipped the context phase) are routed to the
        # reserved dummy slot so the kernel still runs on valid memory.
        num_requests = len(request_ids)
        slots = [self._request_to_slot.get(rid, self._dummy_slot_index) for rid in request_ids]

        # Reuse pre-allocated pinned CPU buffers to avoid costly
        # pinned-memory allocation every round.
        if self._cpu_batch_indices is None or self._cpu_batch_indices.shape[0] < num_requests:
            buf_size = self.max_num_requests
            self._cpu_batch_indices = torch.zeros(
                buf_size, dtype=torch.int32, pin_memory=prefer_pinned()
            )
            self._cpu_nondummy_mask = torch.zeros(
                buf_size, dtype=torch.int32, pin_memory=prefer_pinned()
            )

        batch_indices = self._cpu_batch_indices[:num_requests]
        nondummy_mask = self._cpu_nondummy_mask[:num_requests]
        for i, s in enumerate(slots):
            batch_indices[i] = s
            nondummy_mask[i] = 0 if s == self._dummy_slot_index else 1

        self._gpu_batch_indices[:num_requests].copy_(batch_indices, non_blocking=True)
        self._gpu_nondummy_mask[:num_requests].copy_(nondummy_mask, non_blocking=True)
        # Stream-ordered: the non_blocking copies above are on the current
        # stream, so any kernel launched on the same stream afterwards will
        # see the updated values. No device-wide sync needed.
        torch.cuda.current_stream().synchronize()

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

        # Zero out accepted-token counts for dummy entries so the kernel's
        # extend() loop is a no-op for them, avoiding the concurrent-write
        # race when multiple dummies share one slot.  The mask is populated
        # by prepare() (outside CUDA graph); the multiply is graph-safe.
        num_accepted_tokens = num_accepted_tokens * self._gpu_nondummy_mask[:batch_size]

        _sa_native.invoke_extend(
            batch_size,
            max_draft_len,
            self.pool_size + 1,
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

        # Zero out dummy entries (see extend() for rationale).
        num_accepted_tokens = num_accepted_tokens * self._gpu_nondummy_mask[:batch_size]

        _sa_native.invoke_extend_ngram(
            batch_size,
            max_draft_len,
            max_ngram_size,
            self.pool_size + 1,
            self.max_seq_len,
            self._gpu_slots,
            self._gpu_batch_indices[:batch_size],
            match_len,
            draft_tokens,
            accepted_tokens,
            num_accepted_tokens,
        )

        return match_len, draft_tokens

    def extend_global(
        self,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        max_draft_len: int,
        max_ngram_size: int = -1,
    ) -> tuple:
        """
        Extend SA states and search across all active SAs for the best match.

        Each request's SA is extended with accepted tokens, then all active
        SA states are searched in parallel to find the longest match across
        the pool. CUDA graph compatible (two kernel launches on same stream).

        Args:
            request_ids: List of request IDs in the batch
            accepted_tokens: [batch_size, max_draft_len + 1] accepted token tensor
            num_accepted_tokens: [batch_size] number of accepted tokens per request
            max_draft_len: Maximum draft length
            max_ngram_size: Max ngram size for suffix extraction (-1 = full)

        Returns:
            Tuple of (match_len, draft_tokens) tensors
        """
        self._ensure_workspace(max_draft_len)

        batch_size = len(request_ids)

        match_len = self._gpu_match_len[:batch_size]
        match_slot = self._gpu_match_slot[:batch_size]
        draft_tokens = self._gpu_draft_tokens[:batch_size, :max_draft_len]

        if accepted_tokens.dtype != torch.int32:
            accepted_tokens = accepted_tokens.to(torch.int32)
        if num_accepted_tokens.dtype != torch.int32:
            num_accepted_tokens = num_accepted_tokens.to(torch.int32)

        # Zero out dummy entries (see extend() for rationale).
        num_accepted_tokens = num_accepted_tokens * self._gpu_nondummy_mask[:batch_size]

        _sa_native.invoke_global_search(
            batch_size,
            max_draft_len,
            max_ngram_size,
            self.pool_size + 1,
            self.max_seq_len,
            self._gpu_slots,
            self._gpu_batch_indices[:batch_size],
            self._gpu_active_slot_mask,
            match_len,
            match_slot,
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
        """Add dummy requests for CUDA graph padding.

        Dummy requests are mapped to a single reserved slot
        (index = pool_size) that lives outside the real slot pool.
        This prevents CUDA graph padding from exhausting slots that real
        requests need.

        No host automaton is built -- the GPU slot is already zeroed by
        allocate_workspace (at::zeros), so the kernel safely produces
        match_len = 0 for dummies.
        """
        for rid in request_ids:
            if rid in self._request_to_slot:
                continue
            self._request_to_slot[rid] = self._dummy_slot_index
            self._dummy_request_ids.add(rid)

    def shutdown(self):
        """Clean up all resources."""
        if self._workspace_allocated:
            torch.cuda.synchronize()

        self._request_to_slot.clear()
        self._free_slots = list(range(self.pool_size))
        self._dummy_request_ids.clear()
        self._retained_slots.clear()
        self._active_slots.clear()

        self._host_states_native.clear()
        self._pending_copies.clear()
        self._initialized_requests.clear()
        self._gpu_slots = None

        self._gpu_match_len = None
        self._gpu_draft_tokens = None
        self._gpu_batch_indices = None
        self._gpu_active_slot_mask = None
        self._gpu_match_slot = None
        self._pending_mask_updates.clear()
        self._workspace_allocated = False
        self._allocated_max_draft_len = 0

        torch.cuda.empty_cache()

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0
