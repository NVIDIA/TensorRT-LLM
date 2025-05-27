from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tensorrt_llm.mapping import Mapping


class FileSystemWeightsLoaderInterface(ABC):

    def __init__(self, mapping: Optional[Mapping] = None):
        """
        Initializes the FileSystemWeightsLoader.

        Args:
            mapping: Mapping object for distributed environments.
                     If None, assumes single process (world_size=1, rank=0).
        """
        self._mapping = mapping

    @abstractmethod
    def load_weights(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A string path to the checkpoint directory.

        Returns:
            A dictionary where keys are tensor names and values are the tensors.
        """
