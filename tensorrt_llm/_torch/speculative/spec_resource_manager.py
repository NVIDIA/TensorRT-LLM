"""
Composite resource manager for speculative decoding.

This module provides SpecResourceManager which combines multiple spec resource
managers (e.g., MTP, Eagle3, SA) into a single unified interface.
"""

from typing import TYPE_CHECKING, List, Optional

import torch

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests

if TYPE_CHECKING:
    from .suffix_automaton import SAResourceManager


class SpecResourceManager(BaseResourceManager):
    """
    Composite resource manager that combines a primary spec resource manager
    (e.g., MTPHiddenStatesManager, Eagle3ResourceManager) with SA (Suffix Automaton).

    This allows SA to be used standalone or combined with other speculative
    decoding techniques.

    Examples:
        - SA only: SpecResourceManager(primary_manager=None, sa_manager=sa_mgr)
        - MTP only: Use MTPHiddenStatesManager directly (no wrapping needed)
        - MTP + SA: SpecResourceManager(primary_manager=mtp_mgr, sa_manager=sa_mgr)
        - Eagle3 + SA: SpecResourceManager(primary_manager=eagle3_mgr, sa_manager=sa_mgr)
    """

    def __init__(
        self,
        primary_manager: Optional[BaseResourceManager] = None,
        sa_manager: Optional['SAResourceManager'] = None
    ):
        """
        Args:
            primary_manager: The primary spec resource manager (MTP, Eagle3, etc.)
            sa_manager: Optional SA resource manager for pattern-based drafting
        """
        self.primary_manager = primary_manager
        self.sa_manager = sa_manager

    @property
    def has_primary(self) -> bool:
        return self.primary_manager is not None

    @property
    def has_sa(self) -> bool:
        return self.sa_manager is not None

    # Forward common attributes from primary manager
    def __getattr__(self, name: str):
        """Forward attribute access to primary manager for compatibility."""
        if name in ('primary_manager', 'sa_manager', 'has_primary', 'has_sa'):
            raise AttributeError(name)
        if self.primary_manager is not None:
            return getattr(self.primary_manager, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' "
            f"(primary_manager is None)"
        )

    def get_max_resource_count(self) -> int:
        if self.primary_manager is not None:
            return self.primary_manager.get_max_resource_count()
        if self.sa_manager is not None:
            return self.sa_manager.get_max_resource_count()
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        if self.primary_manager is not None:
            return self.primary_manager.get_needed_resource_to_completion(
                request)
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        """Prepare resources for both managers."""
        if self.primary_manager is not None:
            self.primary_manager.prepare_resources(scheduled_batch)
        if self.sa_manager is not None:
            self.sa_manager.prepare_resources(scheduled_batch)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        """Update resources for both managers."""
        if self.primary_manager is not None:
            self.primary_manager.update_resources(scheduled_batch)
        if self.sa_manager is not None:
            self.sa_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        """Free resources from both managers."""
        if self.primary_manager is not None:
            self.primary_manager.free_resources(request)
        if self.sa_manager is not None:
            self.sa_manager.free_resources(request)

    def add_dummy_requests(self, request_ids: List[int]):
        """Add dummy requests to both managers for CUDA graph warmup."""
        if self.primary_manager is not None:
            self.primary_manager.add_dummy_requests(request_ids)
        if self.sa_manager is not None:
            self.sa_manager.add_dummy_requests(request_ids)

    def shutdown(self):
        """Shutdown both managers."""
        if self.primary_manager is not None and hasattr(self.primary_manager,
                                                         'shutdown'):
            self.primary_manager.shutdown()
        if self.sa_manager is not None:
            self.sa_manager.shutdown()

    # SA-specific methods for external access
    def get_sa_manager(self) -> Optional['SAResourceManager']:
        """Get the SA manager for SA-specific operations."""
        return self.sa_manager

    def sa_add_request(self, request_id: int, context_tokens: List[int]):
        """Add a request to SA manager."""
        if self.sa_manager is not None:
            self.sa_manager.add_request(request_id, context_tokens)

    def sa_prepare_batch(self, request_ids: List[int], max_draft_len: int):
        """Prepare SA batch for extend operation."""
        if self.sa_manager is not None:
            self.sa_manager.prepare_batch(request_ids, max_draft_len)

    def sa_extend_and_get_drafts(
        self,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        max_draft_len: int
    ) -> Optional[tuple]:
        """Extend SA states and get draft tokens."""
        if self.sa_manager is not None:
            return self.sa_manager.extend_and_get_drafts(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len
            )
        return None
