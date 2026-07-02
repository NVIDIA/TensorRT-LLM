# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import safetensors

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

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
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
            self._weights.update(other)

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
            keys_to_delete = [
                k for k in self._weights.keys() if k.startswith(prefix + ".")
            ]
            for key in keys_to_delete:
                del self._weights[key]
            return len(keys_to_delete)


class MmappedSafetensorsWeights:
    """
    Memory-mapped safetensors checkpoint access for integrated GPU systems.

    Opens checkpoint shards via ``safetensors.safe_open`` without loading full
    tensors into RAM. ``mark_consumed`` drops index entries after weights are
    copied into the model to reduce peak memory on unified-memory machines.
    """

    def __init__(self, weight_files: List[str]):
        self._files = weight_files
        self._handles: list = []
        self._key_locations: dict[str, tuple[int, str]] = {}
        self._lock = threading.Lock()
        for file_idx, path in enumerate(weight_files):
            handle = safetensors.safe_open(path, framework="pt", device="cpu")
            self._handles.append(handle)
            for key in handle.keys():
                if key in self._key_locations:
                    raise RuntimeError(
                        f"Duplicate tensor name {key!r} in checkpoint files.")
                self._key_locations[key] = (file_idx, key)

    def _get_slice(self, key: str):
        file_idx, tensor_name = self._key_locations[key]
        return self._handles[file_idx].get_slice(tensor_name)

    def __getitem__(self, key: str) -> Any:
        return self._get_slice(key)

    def __contains__(self, key: str) -> bool:
        return key in self._key_locations

    def __len__(self) -> int:
        return len(self._key_locations)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._key_locations.keys()))

    def keys(self):
        with self._lock:
            return list(self._key_locations.keys())

    def values(self):
        with self._lock:
            return [self._get_slice(k) for k in self._key_locations.keys()]

    def items(self) -> Iterator[Tuple[str, Any]]:
        with self._lock:
            keys = list(self._key_locations.keys())
        return [(k, self._get_slice(k)) for k in keys]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._key_locations:
            return self._get_slice(key)
        return default

    def mark_consumed(self, prefix: str) -> int:
        with self._lock:
            keys_to_delete = [
                k for k in self._key_locations if k.startswith(prefix + ".")
            ]
            for key in keys_to_delete:
                del self._key_locations[key]
            return len(keys_to_delete)

    def remap_keys(self, key_mapping: dict[str,
                                           str]) -> "MmappedSafetensorsWeights":
        remapped = object.__new__(MmappedSafetensorsWeights)
        remapped._files = self._files
        remapped._handles = self._handles
        remapped._lock = threading.Lock()
        remapped._key_locations = {}
        for old_key, loc in self._key_locations.items():
            new_key = key_mapping.get(old_key, old_key)
            if new_key in remapped._key_locations:
                raise RuntimeError(
                    f"Duplicate tensor name {new_key!r} after key remap.")
            remapped._key_locations[new_key] = loc
        return remapped

    def transform_keys(
            self, transform_fn: Callable[[str],
                                         str]) -> "MmappedSafetensorsWeights":
        key_mapping = {
            old_key: transform_fn(old_key)
            for old_key in self._key_locations
        }
        return self.remap_keys(key_mapping)

    def rename_by_regex(
            self, pattern_mapping: dict[str,
                                        str]) -> "MmappedSafetensorsWeights":
        key_mapping = {}
        matched_keys = set()
        for key in self._key_locations:
            new_key = key
            for pattern, replacement in pattern_mapping.items():
                if re.match(pattern, key):
                    new_key = re.sub(pattern, replacement, key)
                    matched_keys.add(key)
                    break
            key_mapping[key] = new_key
        return self.remap_keys(key_mapping)


def rename_weight_keys_with_regex(
    weights: Union[Dict[str, Any], ConsumableWeightsDict,
                   MmappedSafetensorsWeights],
    pattern_mapping: dict[str, str],
) -> Union[Dict[str, Any], ConsumableWeightsDict, MmappedSafetensorsWeights]:
    if isinstance(weights, MmappedSafetensorsWeights):
        return weights.rename_by_regex(pattern_mapping)

    is_consumable = isinstance(weights, ConsumableWeightsDict)
    renamed_weights = {}
    matched_keys = set()
    for key in list(weights.keys()):
        new_key = key
        for pattern, replacement in pattern_mapping.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                matched_keys.add(key)
                break
        renamed_weights[new_key] = weights[key]
    if is_consumable:
        return ConsumableWeightsDict(renamed_weights)
    return renamed_weights


def remap_weight_keys(
    weights: Union[Dict[str, Any], ConsumableWeightsDict,
                   MmappedSafetensorsWeights],
    key_mapping: dict[str, str],
) -> Union[Dict[str, Any], ConsumableWeightsDict, MmappedSafetensorsWeights]:
    if isinstance(weights, MmappedSafetensorsWeights):
        return weights.remap_keys(key_mapping)

    is_consumable = isinstance(weights, ConsumableWeightsDict)
    renamed_weights = {}
    for key in weights.keys():
        renamed_weights[key_mapping.get(key, key)] = weights[key]
    if is_consumable:
        return ConsumableWeightsDict(renamed_weights)
    return renamed_weights


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
