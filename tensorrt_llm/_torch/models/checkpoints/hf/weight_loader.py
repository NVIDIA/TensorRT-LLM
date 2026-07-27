# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import multiprocessing
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, local_mpi_barrier,
                                 local_mpi_comm, local_mpi_rank, local_mpi_size)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_WEIGHT_CACHE_ENV = "TRTLLM_HF_WEIGHT_CACHE"
_WEIGHT_CACHE_MAX_ENTRIES_ENV = "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"
# Default to a single cached checkpoint: each entry pins a full copy of the
# raw weights in CPU RAM, so callers wanting cross-model caching must opt in
# via TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES.
_DEFAULT_WEIGHT_CACHE_MAX_ENTRIES = 1
_WEIGHT_CACHE_LOCK = threading.Lock()
_WEIGHT_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()


@register_checkpoint_weight_loader("MX")
@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("mistral_large_3")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    @staticmethod
    def _is_weight_cache_enabled() -> bool:
        return os.environ.get(_WEIGHT_CACHE_ENV,
                              "0").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _weight_cache_max_entries() -> int:
        try:
            return max(
                0,
                int(
                    os.environ.get(_WEIGHT_CACHE_MAX_ENTRIES_ENV,
                                   _DEFAULT_WEIGHT_CACHE_MAX_ENTRIES)))
        except ValueError:
            logger.warning(
                f"Invalid {_WEIGHT_CACHE_MAX_ENTRIES_ENV} value; disabling HF weight cache."
            )
            return 0

    @staticmethod
    def _weight_files_cache_key(weight_files: List[str],
                                use_consolidated: bool) -> tuple:
        file_fingerprint = []
        for file_name in sorted(weight_files):
            stat = os.stat(file_name)
            file_fingerprint.append(
                (os.path.abspath(file_name), stat.st_size, stat.st_mtime_ns))
        return (tuple(file_fingerprint), use_consolidated)

    @staticmethod
    def _clear_weight_cache() -> None:
        with _WEIGHT_CACHE_LOCK:
            _WEIGHT_CACHE.clear()

    @staticmethod
    def _evict_to_make_room() -> None:
        """Evict LRU entries on a miss BEFORE the new load, so CPU never holds
        the old (cached) and new (loading) weights at once (a ~2x peak)."""
        max_entries = HfWeightLoader._weight_cache_max_entries()
        if max_entries <= 0:
            return
        with _WEIGHT_CACHE_LOCK:
            while len(_WEIGHT_CACHE) >= max_entries:
                _WEIGHT_CACHE.popitem(last=False)

    @staticmethod
    def _tensor_sig(t: torch.Tensor) -> tuple:
        """A cheap integrity fingerprint: shape, dtype and a sampled sum.

        Recomputing the same sum over the same (unmutated) memory is exactly
        deterministic, so plain equality detects in-place mutation. Sampling
        up to 1024 strided elements keeps this at microseconds per tensor.
        """
        flat = t.detach().reshape(-1)
        stride = max(1, flat.numel() // 1024)
        sample = flat[::stride][:1024]
        return (tuple(t.shape), str(t.dtype),
                float(torch.nan_to_num(sample.float()).sum()))

    @staticmethod
    def _fingerprint(weights: dict[str, Any]) -> dict[str, tuple]:
        return {
            key: HfWeightLoader._tensor_sig(value)
            for key, value in weights.items() if torch.is_tensor(value)
        }

    @staticmethod
    def _cache_loaded_weights(cache_key: tuple,
                              loaded_weights: dict[str, Any]) -> None:
        max_entries = HfWeightLoader._weight_cache_max_entries()
        if max_entries <= 0:
            return

        weights = dict(loaded_weights)
        # Fingerprint outside the lock; the cache shares tensors across loads
        # (read-only by contract), and the fingerprint turns a violation of
        # that contract into a detected, self-healing miss instead of
        # silently corrupted weights (see _get_cached_weights).
        sigs = HfWeightLoader._fingerprint(weights)
        # Room was already made by the caller-side evict-before-load in
        # _with_weight_cache (the load-bearing one for the memory peak).
        with _WEIGHT_CACHE_LOCK:
            _WEIGHT_CACHE[cache_key] = (weights, sigs)

    @staticmethod
    def _get_cached_weights(cache_key: tuple) -> ConsumableWeightsDict | None:
        with _WEIGHT_CACHE_LOCK:
            entry = _WEIGHT_CACHE.get(cache_key)
            if entry is None:
                return None
            weights, sigs = entry
            _WEIGHT_CACHE.move_to_end(cache_key)
        # Integrity check: cached tensors are shared, so an earlier consumer
        # mutating them in place (e.g. an in-place transform in a weight
        # mapper) would poison every later load. Detect it, name the culprit
        # keys, drop the entry and let the caller reload from disk.
        mutated = [
            key for key, sig in sigs.items()
            if HfWeightLoader._tensor_sig(weights[key]) != sig
        ]
        if mutated:
            logger.warning(
                "HF weight cache entry was mutated in place since it was "
                f"stored (keys: {mutated[:5]}{'...' if len(mutated) > 5 else ''}); "
                "dropping it and reloading from disk. Weight preprocessing "
                "must not mutate raw checkpoint tensors.")
            with _WEIGHT_CACHE_LOCK:
                if _WEIGHT_CACHE.get(cache_key) is entry:
                    del _WEIGHT_CACHE[cache_key]
            return None
        # Return a fresh dict wrapper because model loaders call
        # mark_consumed(). Tensor values are intentionally shared: this
        # cache targets read-only raw checkpoint tensors, not per-config
        # materialized module weights.
        return ConsumableWeightsDict(dict(weights))

    @staticmethod
    def _get_local_available_host_memory() -> int:
        """Determine the minimum available memory observed on the local node
        and distribute it to all local ranks

        Because psutil.virtual_memory().available is just a snapshot in time,
        it is possible for the local ranks to get different numbers due to
        timing differences. This can lead to disagreement among the local ranks
        as to whether prefetch should be enabled, which causes a deadlock,
        because the ranks that think prefetch is enabled will wait at a local
        mpi barrier indefinitely for the ranks that do not.
        """
        available_host_memory = psutil.virtual_memory().available
        if ENABLE_MULTI_DEVICE:
            return local_mpi_comm().allreduce(available_host_memory,
                                              op=_MPI.MIN)
        return available_host_memory

    def _with_weight_cache(self, weight_files: List[str],
                           use_consolidated: bool,
                           mirror_load_collectives: bool,
                           load_fn) -> ConsumableWeightsDict:
        """Wrap ``load_fn`` with the optional raw-weight cache.

        Key -> hit (optionally joining the local barrier the miss path is
        about to enter) -> evict-before-load (so CPU never holds the old
        cached and the new loading weights at once) -> load -> store.
        """
        cache_key = self._weight_files_cache_key(
            weight_files,
            use_consolidated) if self._is_weight_cache_enabled() else None
        if cache_key is not None:
            cached_weights = self._get_cached_weights(cache_key)
            if cached_weights is not None:
                logger.info("Reusing cached HF checkpoint weights.")
                if mirror_load_collectives:
                    # Rank-local caches can diverge, so a hit on one rank must
                    # enqueue EXACTLY the collectives a miss on another rank
                    # enqueues, in the same order, or the job deadlocks. The
                    # safetensors miss path performs an Allreduce (inside
                    # _get_local_available_host_memory) and then a Barrier;
                    # mirror both here (the allreduce result is unused).
                    self._get_local_available_host_memory()
                    local_mpi_barrier()
                return cached_weights
            self._evict_to_make_room()
        weights = load_fn()
        if cache_key is not None:
            self._cache_loaded_weights(cache_key, weights)
        return weights

    def load_weights(self,
                     checkpoint_dir: str,
                     mapping: Mapping,
                     use_consolidated: bool = False,
                     **kwargs) -> dict[str, Any]:
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former unless specified explicitly, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files
            if ("consolidated" in os.path.split(x)[1]) == use_consolidated
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if weight_files:
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=True,
                load_fn=lambda: self._prefetch_and_load(weight_files))

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=False,
                load_fn=lambda: self._load_weights_in_parallel(
                    weight_files, self._load_bin_or_path_file,
                    "Loading bin weights in parallel"))

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _prefetch_and_load(self,
                           weight_files: List[str]) -> ConsumableWeightsDict:
        # Prefetch the weight files to CPU memory if the size is less than 90% of the available memory.
        # This is a heuristic to avoid prefetching files that are too large and causing file cache thrashing.
        prefetch_size = sum(os.path.getsize(file) for file in weight_files)
        # If the layer number is overridden, it indicates that only a subset of layers are loaded.
        # Prefetching all layers is unnecessary.
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        enable_prefetch = (prefetch_size
                           < self._get_local_available_host_memory() * 0.9
                           and num_layers == 0)
        if enable_prefetch:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
            )
            self.prefetch_files(weight_files)
        # Sync all local ranks unconditionally. `enable_prefetch` depends on
        # `psutil.virtual_memory().available`, a per-rank volatile value, so
        # different ranks may take different branches; gating the barrier on
        # it would deadlock between ranks that prefetched and ranks that
        # skipped. Ranks that didn't prefetch reach the barrier immediately.
        local_mpi_barrier()

        return self._load_weights_in_parallel(
            weight_files, self._load_safetensors_file,
            "Loading safetensors weights in parallel")

    def _load_weights_in_parallel(self, weight_files: List[str], load_func,
                                  description: str) -> ConsumableWeightsDict:
        """
        Load weight files in parallel using the specified loading function.

        Args:
            weight_files: List of weight file paths
            load_func: Function to load individual weight files
            description: Description for the progress bar

        Returns:
            ConsumableWeightsDict containing all loaded weights
        """
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), desc=description)

        # Note that the function is called with a tuple of arguments, hence we need to wrap the arguments in a tuple via [(w,) for w in weight_files]
        # specifically the comma right after the w is important to make it a tuple.
        run_concurrently(load_func, [(w, ) for w in weight_files],
                         reduce_func=weights.update,
                         pbar=pbar)

        return ConsumableWeightsDict(weights)

    @staticmethod
    def _load_safetensors_file(file):
        logger.info(f"Start to load safetensor file {file}")
        return safetensors.torch.load_file(file)

    @staticmethod
    def _load_bin_or_path_file(file):
        try:
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=True)
        except Exception:
            logger.warning(
                f"Failed to load {file} with mmap=True, fallback to mmap=False")
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=False)
        finally:
            return part_weights

    def _prefetch_one_file(self, file_name):
        if os.path.exists(file_name):
            logger.info(f"Prefetching {file_name} to memory...")
            with open(file_name, 'rb') as f:
                f.read()
            logger.info(f"Finished prefetching {file_name}.")

    def prefetch_files(self, file_names: List[str]):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices local_rank, local_rank + local_mpi_size, local_rank + 2*local_mpi_size, etc.
        local_file_names = file_names[local_mpi_rank()::local_mpi_size()]
        if len(local_file_names) == 0:
            return

        max_workers = min(multiprocessing.cpu_count() * 2, 16,
                          len(local_file_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._prefetch_one_file, local_file_names))
