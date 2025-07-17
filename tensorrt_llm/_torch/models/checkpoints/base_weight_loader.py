from abc import ABC, abstractmethod
from typing import Any


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str) -> dict[str, Any]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.

        Returns:
            A dictionary where keys are tensor names and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
