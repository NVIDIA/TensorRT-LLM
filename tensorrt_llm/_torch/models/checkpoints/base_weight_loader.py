from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple, Union

from tensorrt_llm.mapping import Mapping


class ConsumableWeightsDict:
    """
    Wrapper around a weights dictionary that allows marking keys as consumed
    to free memory during model loading.

    This reduces peak memory usage by deleting weight tensors from the dictionary
    after they have been copied to the model, rather than keeping all weights
    in memory until loading completes.
    """

    def __init__(self, weights: Dict[str, Any]):
        self._weights = weights

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._weights[key] = value

    def __delitem__(self, key: str) -> None:
        del self._weights[key]

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def keys(self):
        return self._weights.keys()

    def values(self):
        return self._weights.values()

    def items(self) -> Iterator[Tuple[str, Any]]:
        return self._weights.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._weights.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self._weights.update(other)

    def mark_consumed(self, prefix: str) -> int:
        """
        Delete all keys starting with the given prefix to free memory.

        Args:
            prefix: The prefix to match. Keys starting with "{prefix}." will be deleted.

        Returns:
            The number of keys deleted.
        """
        keys_to_delete = [
            k for k in self._weights.keys() if k.startswith(prefix + ".")
        ]
        for key in keys_to_delete:
            del self._weights[key]
        return len(keys_to_delete)


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(
            self, checkpoint_dir: str,
            mapping: Mapping) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
