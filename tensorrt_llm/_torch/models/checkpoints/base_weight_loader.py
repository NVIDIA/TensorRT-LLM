# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from typing import Any, Dict, Iterator, Tuple, Union

from tensorrt_llm.mapping import Mapping


class ConsumableWeightsDict:
    """
    Wrapper around a weights dictionary that allows marking keys as consumed
    to free memory during model loading.

    This reduces peak memory usage by deleting weight tensors from the dictionary
    after they have been copied to the model, rather than keeping all weights
    in memory until loading completes.

    Thread-safe: uses a lock to protect concurrent access. Iteration methods
    (keys, values, items, __iter__) return snapshot copies to allow safe
    concurrent iteration while other threads may modify the dictionary.
    """

    def __init__(self, weights: Dict[str, Any]):
        self._weights = weights
        self._lock = threading.Lock()
        self._key_index: list[str] | None = None

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._weights:
                self._key_index = None
            self._weights[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._weights[key]

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __iter__(self) -> Iterator[str]:
        # Return iterator over a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return iter(list(self._weights.keys()))

    def keys(self):
        # Return a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return list(self._weights.keys())

    def values(self):
        # Return a snapshot copy of values to allow concurrent modification
        with self._lock:
            return list(self._weights.values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        # Return a snapshot copy of items to allow concurrent modification
        with self._lock:
            return list(self._weights.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._weights.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            if any(key not in self._weights for key in other):
                self._key_index = None
            self._weights.update(other)

    def clear(self) -> None:
        """Drop every remaining reference.

        Use once a downstream dict owns the tensors: a derived dict aliases the
        source tensors it did not rewrite, so consuming it frees nothing while
        this dict still holds them.
        """
        with self._lock:
            self._weights.clear()
            self._key_index = []

    def filter_prefix(self, prefix: str) -> Dict[str, Any]:
        """Return weights below ``prefix`` without scanning the full mapping."""
        with self._lock:
            if not prefix:
                return dict(self._weights)
            prefix_with_separator = prefix + "."
            return {
                key[len(prefix_with_separator):]: self._weights[key]
                for key in self._matching_keys_locked(prefix_with_separator)
            }

    def _matching_keys_locked(self, prefix_with_separator: str) -> list[str]:
        if self._key_index is None:
            self._key_index = sorted(self._weights)
        begin = bisect_left(self._key_index, prefix_with_separator)
        end = bisect_right(self._key_index,
                           prefix_with_separator + chr(0x10FFFF))
        return [
            key for key in self._key_index[begin:end] if key in self._weights
        ]

    def _delete_keys_locked(self, keys: list[str]) -> int:
        for key in keys:
            del self._weights[key]
        return len(keys)

    def mark_consumed_keys(self, keys) -> int:
        """Delete an exact set of keys to free memory.

        Use instead of :meth:`mark_consumed` when a module consumed specific
        tensors rather than a whole ``name.*`` subtree.
        """
        with self._lock:
            keys_to_delete = [
                key for key in dict.fromkeys(keys) if key in self._weights
            ]
            return self._delete_keys_locked(keys_to_delete)

    def mark_consumed(self, prefix: str) -> int:
        """
        Delete all keys starting with the given prefix to free memory.

        Args:
            prefix: The prefix to match. Keys starting with "{prefix}." will be deleted.

        Returns:
            The number of keys deleted.

        Thread-safe: uses a lock to prevent concurrent modification issues.
        """
        with self._lock:
            keys_to_delete = self._matching_keys_locked(prefix + ".")
            return self._delete_keys_locked(keys_to_delete)


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.
            **kwargs: Optional format-specific loader arguments.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
