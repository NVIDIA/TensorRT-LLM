from abc import ABC, abstractmethod
from typing import List, Optional, final

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

    def __init__(self, max_concurrency: Optional[int] = None) -> None:
        self.max_concurrency = max_concurrency

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
    def should_use_spec_decode(self, requests: List[LlmRequest]) -> bool:
        """
        You probably don't want to override this. ModelEngine
        assumes that speculation is always on if max_concurrency
        is not specified by the user's spec config.
        """
        if self.max_concurrency is not None:
            return len(requests) <= self.max_concurrency
        return True
