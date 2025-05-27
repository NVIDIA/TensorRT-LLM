from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

StateDict = Dict[str, Any]  # Typically Dict[str, torch.Tensor]
TRTLLMStateDict = Dict[str, Any]  # Typically Dict[str, torch.Tensor]


class WeightsMapperInterface(ABC):

    @abstractmethod
    def map_weights(self, state_dict: StateDict) -> TRTLLMStateDict:
        """
        Maps weights from a source state dictionary (e.g., Hugging Face)
        to a TRT-LLM compatible state dictionary.

        Args:
            state_dict: The input state dictionary.

        Returns:
            A new state dictionary mapped for TRT-LLM.
        """

    @abstractmethod
    def apply_transformations(
            self, callback: Callable[[StateDict], StateDict]) -> None:
        """
        Applies a series of transformation functions to an internal representation
        of weights or to guide the mapping process. The exact behavior might depend
        on the implementation (e.g., storing callbacks to be applied later).

        Args:
            callback: A callable that takes a StateDict and returns a transformed StateDict.
        """
