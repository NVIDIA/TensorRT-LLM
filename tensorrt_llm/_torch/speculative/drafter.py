from abc import ABC, abstractmethod
from typing import List, Optional

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

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

    def should_use_spec_decode(self, requests: List[LlmRequest]) -> bool:
        """Check if spec decode should be used for the current iteration."""
        return True
