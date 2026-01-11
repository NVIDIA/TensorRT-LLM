from abc import ABC, abstractmethod
from typing import Any

from tensorrt_llm.mapping import Mapping


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str,
                     mapping: Mapping) -> dict[str, Any]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.

        Returns:
            A dictionary where keys are tensor names and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
