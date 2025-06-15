from abc import ABC, abstractmethod

from ..pyexecutor.sampler import SampleState
from .scheduler import ScheduledRequests


class Drafter(ABC):

    def __init__():
        pass

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.
        """
        raise NotImplementedError

    @abstractmethod
    def update_drafter(
        self,
        state: SampleState,
    ) -> None:
        """
        Update the state of the drafter after forward computation this step.
        """
        raise NotImplementedError
