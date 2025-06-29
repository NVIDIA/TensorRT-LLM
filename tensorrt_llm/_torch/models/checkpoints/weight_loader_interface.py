from abc import ABC, abstractmethod
from typing import Any, Optional

from tensorrt_llm.mapping import Mapping


class WeightLoaderInterface(ABC):

    def __init__(self, mapping: Optional[Mapping] = None):
        """
        Initializes the WeightLoader.

        Args:
            mapping: Mapping object for distributed environments.
        """
        self._mapping = mapping

    @abstractmethod
    def load_weights(self, checkpoint_dir: str) -> dict[str, Any]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.

        Returns:
            A dictionary where keys are tensor names and values are the tensors.
        """

    def set_mapping(self, mapping: Mapping):
        self._mapping = mapping

    def ensure_mapping(self, mapping: Optional[Mapping] = None):
        """
        Ensures that mapping is set, will use default Mapping() if not provided and not already set.
        """
        if self._mapping is None:
            if mapping is None:
                mapping = Mapping()
            self.set_mapping(mapping)
        else:
            raise ValueError("Mapping already set.")
