from abc import ABC, abstractmethod

from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpeculativeDecodingMode


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.

        Args:
            scheduled_requests: The scheduled requests for this iteration
        """
        raise NotImplementedError


def create_drafter(spec_decoding_mode: SpeculativeDecodingMode,
                   **kwargs) -> Drafter:
    """
    Factory function to create the appropriate drafter based on the mode.

    Args:
        spec_decoding_mode: The speculative decoding mode
        **kwargs: Additional arguments for drafter construction

    Returns:
        Drafter: The appropriate drafter instance

    Raises:
        ValueError: If the speculative decoding mode is not supported
    """
    match spec_decoding_mode:
        case SpeculativeDecodingMode.NGRAM:
            from .ngram import NGramDrafter
            return NGramDrafter(**kwargs)
        case SpeculativeDecodingMode.EAGLE3 | SpeculativeDecodingMode.DRAFT_TARGET:
            # Import here to avoid circular import
            from .model_drafter import ModelDrafter
            return ModelDrafter(**kwargs)
        case _:
            raise ValueError(
                f"Unsupported speculative decoding mode: {spec_decoding_mode}")
