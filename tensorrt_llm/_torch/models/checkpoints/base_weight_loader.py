# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from abc import ABC, abstractmethod
from bisect import bisect_left
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

    @classmethod
    def take_ownership(cls, source: Union[Dict[str, Any],
                                          "ConsumableWeightsDict"],
                       derived: Dict[str, Any]) -> Dict[str, Any]:
        """Hand ``derived`` the tensors ``source`` was holding.

        A renamed or filtered mapping aliases the tensors it was built from, so
        while the source is alive it holds a second reference to each one and
        consuming the alias frees nothing. Emptying the source makes the alias
        the last reference, which is what lets the loader release weights
        module by module instead of pinning the whole checkpoint.

        A plain dict source is returned unchanged -- it was never doing
        incremental release. **The caller must not use ``source`` afterwards.**
        """
        if not isinstance(source, cls):
            return derived
        source.clear()
        return cls(derived)

    def filter_prefix(self, prefix: str) -> Dict[str, Any]:
        """Same result as a ``startswith(prefix)`` scan, without the scan.

        ``prefix`` must be non-empty. Callers that may pass an empty prefix
        keep their own scan; only the loading loop, which always names a
        module, comes through here.
        """
        with self._lock:
            start = len(prefix) + 1
            return {
                key[start:]: self._weights[key]
                for key in self._keys_with_prefix_locked(prefix)
            }

    def _keys_with_prefix_locked(self, prefix: str) -> list[str]:
        """Return the live keys starting with ``prefix``, in sorted order.

        Deletions deliberately do not invalidate the index: a stale index is
        always a superset of the live keys, so filtering on membership keeps
        every reader correct while each lookup stays proportional to the keys
        it matched rather than to the size of the checkpoint.
        """
        if self._key_index is None:
            self._key_index = sorted(self._weights)
        begin = bisect_left(self._key_index, prefix)
        # Exclusive upper bound: every key starting with the prefix sorts
        # before the prefix with its last character incremented.
        upper_bound = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        end = bisect_left(self._key_index, upper_bound, begin)
        return [
            key for key in self._key_index[begin:end] if key in self._weights
        ]

    def mark_consumed_keys(self, keys) -> int:
        """Delete an exact set of keys to free memory.

        Use instead of :meth:`mark_consumed` when a module consumed specific
        tensors rather than a whole ``name.*`` subtree.
        """
        deleted = 0
        with self._lock:
            for key in keys:
                if key in self._weights:
                    del self._weights[key]
                    deleted += 1
        return deleted

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
            keys_to_delete = self._keys_with_prefix_locked(prefix + ".")
            for key in keys_to_delete:
                del self._weights[key]
            return len(keys_to_delete)


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
