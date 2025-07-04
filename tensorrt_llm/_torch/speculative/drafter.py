from abc import ABC, abstractmethod
from typing import Optional

from tensorrt_llm.bindings.executor import IterationStats

from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import SampleState
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):

    def __init__(
        self,
        spec_resource_manager: Optional[BaseResourceManager] = None,
    ):
        self.spec_resource_manager = spec_resource_manager

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        state: SampleState,
        iter_stats: IterationStats = None,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.
        """
        raise NotImplementedError
