from abc import ABC, abstractmethod

from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.
        """
        raise NotImplementedError
