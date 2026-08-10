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
import json
import multiprocessing
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, List

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.mmap_utils import populate_file_pages
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict)
from tensorrt_llm._torch.models.checkpoints.hf.rank_striped_read_ahead import (
    RankStripedReadAheadSession, build_local_plan, close_node_communicator,
    coordinate_error, effective_available_host_memory, memory_admission)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, local_mpi_barrier,
                                 local_mpi_comm, local_mpi_rank, local_mpi_size,
                                 mpi_comm, mpi_disabled)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_WEIGHT_CACHE_ENV = "TRTLLM_HF_WEIGHT_CACHE"
_WEIGHT_CACHE_MAX_ENTRIES_ENV = "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"
_NATIVE_IO_POLICY = "native"
_RANK_STRIPED_IO_POLICY = "rank_striped_read_ahead"
_SUPPORTED_IO_POLICIES = (_NATIVE_IO_POLICY, _RANK_STRIPED_IO_POLICY)
# Default to a single cached checkpoint: each entry pins a full copy of the
# raw weights in CPU RAM, so callers wanting cross-model caching must opt in
# via TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES.
_DEFAULT_WEIGHT_CACHE_MAX_ENTRIES = 1
_WEIGHT_CACHE_LOCK = threading.Lock()
_WEIGHT_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()

# Prefetch warms the OS page cache with mmap + madvise(MADV_POPULATE_READ)
# (no user-space copy, no anonymous buffer), falling back to chunked reads
# through one bounded buffer per in-flight file where unsupported. Both paths
# work in fixed-size windows and emit a progress log at a fixed cadence so
# that a slow prefetch produces observable output instead of minutes of
# silence.
_PREFETCH_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
_PREFETCH_LOG_INTERVAL_SEC = 60.0
# Log the chunked-read fallback once per process instead of once per file;
# the lock makes the check-and-set atomic across concurrent prefetch threads.
_PREFETCH_FALLBACK_LOGGED = threading.Event()
_PREFETCH_FALLBACK_LOG_LOCK = threading.Lock()


@dataclass
class _CheckpointIOStatus:
    requested: str
    selected: str
    activated: bool = False
    effective: str = "none"
    fallback_reason: str | None = None


@register_checkpoint_weight_loader("MX")
@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("mistral_large_3")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    def __init__(self,
                 checkpoint_io_policy: str = _NATIVE_IO_POLICY,
                 *,
                 partial_model_loading: bool = False) -> None:
        if checkpoint_io_policy not in _SUPPORTED_IO_POLICIES:
            raise ValueError("checkpoint_io_policy must be one of "
                             f"{_SUPPORTED_IO_POLICIES}, got "
                             f"{checkpoint_io_policy!r}")
        self._checkpoint_io_policy = checkpoint_io_policy
        self._partial_model_loading = partial_model_loading
        self._last_checkpoint_io_status = _CheckpointIOStatus(
            requested=checkpoint_io_policy, selected=checkpoint_io_policy)

    @property
    def checkpoint_io_policy(self) -> str:
        return self._checkpoint_io_policy

    @property
    def last_checkpoint_io_status(self) -> _CheckpointIOStatus:
        return replace(self._last_checkpoint_io_status)

    def _reset_checkpoint_io_status(self) -> None:
        self._last_checkpoint_io_status = _CheckpointIOStatus(
            requested=self._checkpoint_io_policy,
            selected=self._checkpoint_io_policy,
        )

    def _log_checkpoint_io_status(self) -> None:
        status = self._last_checkpoint_io_status
        logger.info(
            f"Checkpoint I/O policy: requested={status.requested}, "
            f"selected={status.selected}, activated={status.activated}, "
            f"effective={status.effective}, fallback_reason="
            f"{status.fallback_reason or 'none'}.")

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
    def _get_local_available_host_memory(local_communicator=None) -> int:
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
            communicator = (local_mpi_comm() if local_communicator is None else
                            local_communicator)
            return communicator.allreduce(available_host_memory, op=_MPI.MIN)
        return available_host_memory

    def _with_weight_cache(self,
                           weight_files: List[str],
                           use_consolidated: bool,
                           mirror_load_collectives: bool,
                           load_fn,
                           local_communicator=None) -> ConsumableWeightsDict:
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
                    if local_communicator is None:
                        self._get_local_available_host_memory()
                        local_mpi_barrier()
                    else:
                        self._get_local_available_host_memory(
                            local_communicator)
                        local_communicator.Barrier()
                return cached_weights
            self._evict_to_make_room()
        weights = load_fn()
        if cache_key is not None:
            self._cache_loaded_weights(cache_key, weights)
        return weights

    def cleanup(self) -> None:
        # Drop lazy safetensors handles (if any) so the mmaps are released.
        self._lazy_handles = []
        super().cleanup()

    @staticmethod
    def _is_kimi_k3_checkpoint(checkpoint_dir: str) -> bool:
        """Kimi K3 checkpoints (~1.5 TB) must not be materialized in host RAM."""
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.isfile(config_path):
            return False
        # Do not swallow read/parse failures: every rank must take the same
        # branch here (the non-Kimi path enqueues collectives), so a
        # rank-local transient error routing one rank differently would
        # deadlock the job. Propagating fails fast on all ranks instead.
        with open(config_path) as f:
            model_type = json.load(f).get("model_type")
        return model_type in ("kimi_k3", "kimi_linear")

    def _load_lazy_safetensors(
            self,
            checkpoint_dir: str,
            use_consolidated: bool = False) -> dict[str, Any]:
        """Return a dict of name -> lazy safetensors slices.

        Values are ``safetensors`` PySafeSlice objects: ``v[:]`` (or any
        indexing) materializes only the requested bytes from the mmapped
        file. This lets a model's ``load_weights`` stream a huge checkpoint
        and read only its rank-local shard (e.g. Kimi K3 expert-parallel
        expert slices) without ever holding the full checkpoint in RAM.
        """
        weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No safetensors files in {checkpoint_dir}.")
        # Same sharded-vs-consolidated selection as the eager path below:
        # when both flavors are present, keep only the requested one.
        filtered_weight_files = [
            x for x in weight_files
            if ("consolidated" in os.path.split(x)[1]) == use_consolidated
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        weights: dict[str, Any] = {}
        handles = []
        for file_name in weight_files:
            handle = safetensors.safe_open(file_name,
                                           framework="pt",
                                           device="cpu")
            handles.append(handle)
            for name in handle.keys():
                weights[name] = handle.get_slice(name)
        # Keep the file handles alive for as long as the loader lives; the
        # slices reference them. Released in cleanup().
        self._lazy_handles = handles
        logger.info(f"Lazily opened {len(weight_files)} safetensors files "
                    f"({len(weights)} tensors) from {checkpoint_dir}")
        lazy_weights = ConsumableWeightsDict(weights)
        # A lazy slice does not carry the file it came from, and a model that
        # wants to re-open shards itself (Kimi K3 streams rank-local experts
        # per shard file, precisely to avoid holding this mapping open) has no
        # other reliable source: transformers no longer sets
        # ``PretrainedConfig._name_or_path``.
        lazy_weights.checkpoint_dir = checkpoint_dir
        return lazy_weights

    def load_weights(self,
                     checkpoint_dir: str,
                     mapping: Mapping,
                     use_consolidated: bool = False,
                     **kwargs) -> dict[str, Any]:
        """Load synchronously for callers without a materialization session."""
        if self._checkpoint_io_policy == _NATIVE_IO_POLICY:
            self._reset_checkpoint_io_status()
            weights = self._load_weights_native(checkpoint_dir, mapping,
                                                use_consolidated, **kwargs)
            self._last_checkpoint_io_status.effective = _NATIVE_IO_POLICY
            return weights
        with self.open_weight_session(checkpoint_dir,
                                      mapping=mapping,
                                      use_consolidated=use_consolidated,
                                      **kwargs) as weights:
            return weights

    @contextmanager
    def open_weight_session(self,
                            checkpoint_dir: str,
                            mapping: Mapping,
                            use_consolidated: bool = False,
                            **kwargs) -> Iterator[dict[str, Any]]:
        """Keep opt-in read-ahead alive through model materialization."""
        self._reset_checkpoint_io_status()
        if self._checkpoint_io_policy == _NATIVE_IO_POLICY:
            weights = self._load_weights_native(checkpoint_dir, mapping,
                                                use_consolidated, **kwargs)
            self._last_checkpoint_io_status.effective = _NATIVE_IO_POLICY
            yield weights
            return

        active_communicator = self._active_communicator(mapping)
        if mapping.world_size > 1 and active_communicator is None:
            self._last_checkpoint_io_status.selected = _NATIVE_IO_POLICY
            weights = self._fallback_to_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                "distributed rank-striped read-ahead requires an active MPI communicator",
                **kwargs,
            )
            yield weights
            return

        weights, session = self._start_rank_striped_read_ahead(
            checkpoint_dir,
            mapping,
            use_consolidated,
            active_communicator,
            **kwargs,
        )
        if session is None:
            yield weights
            return

        body_error = None
        try:
            yield weights
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                read_error = session.finish(body_error)
            except Exception as error:
                status = self._last_checkpoint_io_status
                status.effective = "none"
                status.fallback_reason = str(error)
                self._log_checkpoint_io_status()
                if body_error is None:
                    raise
                logger.error(
                    "Suppressing rank-striped cleanup failure to preserve "
                    "the model-materialization exception: "
                    f"{type(error).__name__}: {error}")
            else:
                status = self._last_checkpoint_io_status
                if read_error is None:
                    status.effective = _RANK_STRIPED_IO_POLICY
                else:
                    # Read-ahead is advisory. The native mmap/materialization
                    # path succeeded, so never retry a partially mutated model.
                    status.effective = _NATIVE_IO_POLICY
                    status.fallback_reason = str(read_error)
                    logger.warning(
                        "Rank-striped read-ahead degraded after activation; "
                        "keeping the successfully materialized model: "
                        f"{read_error}")
                self._log_checkpoint_io_status()

    @staticmethod
    def _active_communicator(mapping: Mapping):
        """Return and validate the communicator used by the opt-in policy."""
        if not ENABLE_MULTI_DEVICE or mpi_disabled():
            return None
        communicator = mpi_comm()
        communicator_size = communicator.Get_size()
        if mapping.world_size != communicator_size:
            raise RuntimeError(
                "Rank-striped read-ahead requires mapping.world_size to "
                "match the active MPI communicator size "
                f"({communicator_size}); received {mapping.world_size}.")
        return communicator

    @staticmethod
    def _selected_safetensors_files(checkpoint_dir: str,
                                    use_consolidated: bool) -> list[str]:
        weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
        filtered_weight_files = [
            path for path in weight_files
            if ("consolidated" in os.path.split(path)[1]) == use_consolidated
        ]
        return filtered_weight_files or weight_files

    def _close_unactivated_session(
            self,
            session,
            node_communicator,
            active_communicator,
            phase: str,
            *,
            preserve_node_communicator: bool = False) -> None:
        cleanup_error = None
        try:
            if session is not None:
                cleanup_error = (session.cancel_reads()
                                 if preserve_node_communicator else
                                 session.cancel_and_close())
            elif not preserve_node_communicator:
                close_node_communicator(node_communicator)
        except Exception as error:
            cleanup_error = error
        coordinated_cleanup_error = coordinate_error(active_communicator, phase,
                                                     cleanup_error)
        if coordinated_cleanup_error is not None:
            raise coordinated_cleanup_error

    def _fallback_to_native(self,
                            checkpoint_dir: str,
                            mapping: Mapping,
                            use_consolidated: bool,
                            reason: str,
                            active_communicator=None,
                            node_communicator=None,
                            session=None,
                            **kwargs) -> dict[str, Any]:
        status = self._last_checkpoint_io_status
        status.activated = False
        status.fallback_reason = reason
        try:
            self._close_unactivated_session(
                session,
                node_communicator,
                active_communicator,
                "rank-striped fallback reader cleanup",
                preserve_node_communicator=True,
            )
        except BaseException:
            try:
                close_node_communicator(node_communicator)
            except Exception as close_error:
                logger.error(
                    "Failed to close the rank-striped node communicator after "
                    "reader cleanup failed: "
                    f"{type(close_error).__name__}: {close_error}")
            raise
        fallback_communicator = node_communicator
        if (fallback_communicator is None
                and (active_communicator is None
                     or active_communicator.Get_size() == 1)):
            fallback_communicator = active_communicator
        allow_prefetch = not (mapping.world_size > 1
                              and fallback_communicator is None)
        load_error = None
        try:
            weights = self._load_weights_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                _local_communicator=fallback_communicator,
                _allow_prefetch=allow_prefetch,
                **kwargs)
        except BaseException as error:
            load_error = error
            weights = None
        close_error = None
        try:
            close_node_communicator(node_communicator)
        except Exception as error:
            close_error = error
        coordinated_load_error = coordinate_error(
            active_communicator, "rank-striped native fallback", load_error)
        coordinated_close_error = coordinate_error(
            active_communicator, "rank-striped fallback cleanup", close_error)
        if coordinated_load_error is not None:
            status.effective = "none"
            status.fallback_reason = str(coordinated_load_error)
            self._log_checkpoint_io_status()
            if coordinated_close_error is not None:
                logger.error("Rank-striped cleanup also failed during native "
                             f"fallback: {coordinated_close_error}")
            raise coordinated_load_error
        if coordinated_close_error is not None:
            status.effective = "none"
            status.fallback_reason = str(coordinated_close_error)
            self._log_checkpoint_io_status()
            raise coordinated_close_error
        assert weights is not None
        status.effective = _NATIVE_IO_POLICY
        self._log_checkpoint_io_status()
        return weights

    def _start_rank_striped_read_ahead(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        use_consolidated: bool,
        active_communicator,
        **kwargs,
    ) -> tuple[dict[str, Any], RankStripedReadAheadSession | None]:
        node_communicator = None
        split_error = None
        try:
            if (active_communicator is not None
                    and active_communicator.Get_size() > 1):
                node_communicator = active_communicator.Split_type(
                    _MPI.COMM_TYPE_SHARED)
        except Exception as error:
            split_error = error
        coordinated_split_error = coordinate_error(
            active_communicator, "rank-striped node communicator creation",
            split_error)
        if coordinated_split_error is not None:
            self._close_unactivated_session(
                None, node_communicator, active_communicator,
                "rank-striped node communicator cleanup")
            self._last_checkpoint_io_status.selected = _NATIVE_IO_POLICY
            return self._fallback_to_native(checkpoint_dir, mapping,
                                            use_consolidated,
                                            str(coordinated_split_error),
                                            active_communicator, **kwargs), None

        weight_files = []
        stats = []
        available_memory = 0
        eligibility_reason = None
        preflight_error = None
        try:
            if self._is_kimi_k3_checkpoint(checkpoint_dir):
                eligibility_reason = (
                    "the checkpoint requires model-specific lazy SafeTensors loading"
                )
            weight_files = self._selected_safetensors_files(
                checkpoint_dir, use_consolidated)
            stats = [(path, os.stat(path)) for path in weight_files]
            if not weight_files:
                eligibility_reason = "no SafeTensors checkpoint files were found"
            elif (self._is_weight_cache_enabled()
                  and self._weight_cache_max_entries() > 0):
                eligibility_reason = "the raw HF weight cache is enabled"
            elif (self._partial_model_loading
                  or int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0")) != 0):
                eligibility_reason = "partial model loading was requested"
            available_memory = effective_available_host_memory()
        except Exception as error:
            preflight_error = error

        coordinated_preflight_error = coordinate_error(
            active_communicator, "rank-striped preflight", preflight_error)
        if coordinated_preflight_error is not None:
            self._last_checkpoint_io_status.selected = _NATIVE_IO_POLICY
            return self._fallback_to_native(checkpoint_dir, mapping,
                                            use_consolidated,
                                            str(coordinated_preflight_error),
                                            active_communicator,
                                            node_communicator, **kwargs), None

        local_rank = 0
        local_size = 1
        node_error = None
        try:
            backing_signature = tuple(
                (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                for _, stat in stats)
            if node_communicator is not None:
                local_rank = node_communicator.Get_rank()
                local_size = node_communicator.Get_size()
                observations = node_communicator.allgather(
                    (backing_signature, available_memory))
                if any(observation[0] != observations[0][0]
                       for observation in observations[1:]):
                    eligibility_reason = (
                        "node-local ranks resolved different backing files")
                available_memory = min(observation[1]
                                       for observation in observations)
        except Exception as error:
            node_error = error

        coordinated_node_error = coordinate_error(
            active_communicator, "rank-striped node preflight", node_error)
        if coordinated_node_error is not None:
            self._close_unactivated_session(
                None, node_communicator, active_communicator,
                "rank-striped node preflight cleanup")
            self._last_checkpoint_io_status.selected = _NATIVE_IO_POLICY
            return self._fallback_to_native(checkpoint_dir, mapping,
                                            use_consolidated,
                                            str(coordinated_node_error),
                                            active_communicator, **kwargs), None

        file_sizes = [(path, stat.st_size) for path, stat in stats]
        checkpoint_bytes = sum(size for _, size in file_sizes)
        admitted, headroom = memory_admission(checkpoint_bytes,
                                              available_memory)
        if eligibility_reason is None and not admitted:
            eligibility_reason = (
                f"checkpoint bytes ({checkpoint_bytes}) exceed available "
                f"host memory ({available_memory}) after startup headroom "
                f"({headroom})")

        discovery_signature = tuple(
            (os.path.basename(path), size) for path, size in file_sizes)
        selections = ([
            (discovery_signature, eligibility_reason)
        ] if active_communicator is None else active_communicator.allgather(
            (discovery_signature, eligibility_reason)))
        if any(selection[0] != selections[0][0]
               for selection in selections[1:]):
            self._close_unactivated_session(
                None, node_communicator, active_communicator,
                "rank-striped discovery mismatch cleanup")
            raise RuntimeError(
                "SafeTensors checkpoint discovery must match "
                f"across model-load ranks; received {selections}.")
        fallback_reasons = [(rank, selection[1])
                            for rank, selection in enumerate(selections)
                            if selection[1] is not None]
        if fallback_reasons:
            rank, reason = fallback_reasons[0]
            self._last_checkpoint_io_status.selected = _NATIVE_IO_POLICY
            return self._fallback_to_native(checkpoint_dir, mapping,
                                            use_consolidated,
                                            f"rank {rank}: {reason}",
                                            active_communicator,
                                            node_communicator, **kwargs), None

        self._last_checkpoint_io_status.selected = _RANK_STRIPED_IO_POLICY
        session = None
        setup_error = None
        try:
            plan = build_local_plan(file_sizes, local_rank, local_size)
            session = RankStripedReadAheadSession(active_communicator,
                                                  node_communicator,
                                                  plan).start()
        except Exception as error:
            setup_error = error
        coordinated_setup_error = coordinate_error(active_communicator,
                                                   "rank-striped reader setup",
                                                   setup_error)
        if coordinated_setup_error is not None:
            return self._fallback_to_native(checkpoint_dir, mapping,
                                            use_consolidated,
                                            str(coordinated_setup_error),
                                            active_communicator,
                                            node_communicator, session,
                                            **kwargs), None
        assert session is not None

        status = self._last_checkpoint_io_status
        status.activated = True
        logger.info("Rank-striped checkpoint read-ahead activated: "
                    f"local_rank={local_rank}, local_size={local_size}, "
                    f"workers={plan.workers}, assigned_bytes="
                    f"{plan.assigned_bytes}, checkpoint_bytes="
                    f"{checkpoint_bytes}.")

        weights = None
        mapping_error = None
        try:
            weights = self._load_weights_in_parallel(
                weight_files, self._load_safetensors_file,
                "Mapping safetensors weights in parallel")
        except BaseException as error:
            mapping_error = error
        coordinated_mapping_error = coordinate_error(
            active_communicator, "rank-striped SafeTensors mapping",
            mapping_error)
        if coordinated_mapping_error is not None:
            cleanup_error = session.cancel_and_close()
            coordinated_cleanup_error = coordinate_error(
                active_communicator, "rank-striped mapping cleanup",
                cleanup_error)
            if coordinated_cleanup_error is not None:
                logger.error("Rank-striped cleanup also failed after mapping "
                             f"failure: {coordinated_cleanup_error}")
            status.effective = "none"
            status.fallback_reason = str(coordinated_mapping_error)
            self._log_checkpoint_io_status()
            raise coordinated_mapping_error
        assert weights is not None
        return weights, session

    def _load_weights_native(self,
                             checkpoint_dir: str,
                             mapping: Mapping,
                             use_consolidated: bool = False,
                             *,
                             _local_communicator=None,
                             _allow_prefetch: bool = True,
                             **kwargs) -> dict[str, Any]:
        if self._is_kimi_k3_checkpoint(checkpoint_dir):
            return self._load_lazy_safetensors(checkpoint_dir, use_consolidated)
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
            if _local_communicator is None and _allow_prefetch:
                return self._with_weight_cache(
                    weight_files,
                    use_consolidated,
                    mirror_load_collectives=True,
                    load_fn=lambda: self._prefetch_and_load(weight_files))
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=_allow_prefetch,
                load_fn=lambda: self._prefetch_and_load(
                    weight_files, _local_communicator, _allow_prefetch),
                local_communicator=_local_communicator)

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

    def _prefetch_and_load(
            self,
            weight_files: List[str],
            local_communicator=None,
            allow_prefetch: bool = True) -> ConsumableWeightsDict:
        # Prefetch the weight files to CPU memory if the size is less than 90% of the available memory.
        # This is a heuristic to avoid prefetching files that are too large and causing file cache thrashing.
        prefetch_size = sum(os.path.getsize(file) for file in weight_files)
        # If the layer number is overridden, it indicates that only a subset of layers are loaded.
        # Prefetching all layers is unnecessary.
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        if allow_prefetch:
            available_memory = (
                self._get_local_available_host_memory()
                if local_communicator is None else
                self._get_local_available_host_memory(local_communicator))
            enable_prefetch = (prefetch_size < available_memory * 0.9
                               and num_layers == 0)
        else:
            enable_prefetch = False
        if enable_prefetch:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
            )
            if local_communicator is None:
                self.prefetch_files(weight_files)
            else:
                self.prefetch_files(weight_files, local_communicator)
        # Sync all local ranks unconditionally. `enable_prefetch` depends on
        # `psutil.virtual_memory().available`, a per-rank volatile value, so
        # different ranks may take different branches; gating the barrier on
        # it would deadlock between ranks that prefetched and ranks that
        # skipped. Ranks that didn't prefetch reach the barrier immediately.
        if allow_prefetch:
            if local_communicator is None:
                local_mpi_barrier()
            else:
                local_communicator.Barrier()

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
    def _load_safetensors_file(file: str) -> dict[str, torch.Tensor]:
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

    def _prefetch_one_file(
            self,
            file_name: str,
            report_progress: Callable[[int], None] | None = None) -> None:
        if os.path.exists(file_name):
            logger.info(f"Prefetching {file_name} to memory...")
            populated = populate_file_pages(file_name,
                                            _PREFETCH_CHUNK_SIZE_BYTES,
                                            report_progress)
            # Chunked reads resume from wherever population stopped: from
            # zero when MADV_POPULATE_READ is unsupported, reading nothing
            # when population already covered the whole file.
            read_back = self._read_file_in_chunks(file_name, populated,
                                                  report_progress)
            # Gate the fallback log on bytes actually read so a fully
            # populated (or empty) file never logs, and name the cause:
            # a partial populate would otherwise silently pay for both a
            # populate and a chunked read of the remainder on every file.
            should_log = False
            if read_back > 0:
                with _PREFETCH_FALLBACK_LOG_LOCK:
                    should_log = not _PREFETCH_FALLBACK_LOGGED.is_set()
                    if should_log:
                        _PREFETCH_FALLBACK_LOGGED.set()
            if should_log:
                if populated == 0:
                    logger.info(
                        "madvise(MADV_POPULATE_READ) did not populate "
                        f"{file_name} (kernel < 5.14, filesystem without "
                        "mmap support, or open/mmap failure; enable debug "
                        "logging for the errno); prefetching via chunked "
                        "reads.")
                else:
                    logger.info(
                        "madvise(MADV_POPULATE_READ) stopped after "
                        f"{populated} bytes of {file_name} (enable debug "
                        "logging for the errno); finishing affected files "
                        "via chunked reads.")
            logger.info(f"Finished prefetching {file_name}.")

    @staticmethod
    def _read_file_in_chunks(
            file_name: str,
            offset: int,
            report_progress: Callable[[int], None] | None = None) -> int:
        # Read in fixed-size chunks into a reusable buffer instead of one
        # whole-file read: a whole-file read pins the entire file in
        # anonymous memory until it completes, and with up to 16 concurrent
        # multi-GB files per local rank, slow storage lets those buffers
        # accumulate into hundreds of GB across the local ranks, which can
        # OOM the host. Chunked reads warm the OS page cache identically
        # with a constant per-thread footprint. Returns the number of bytes
        # read, i.e. how much of the file population did not cover.
        total_read = 0
        with open(file_name, 'rb') as f:
            # Allocate the buffer only when there is something left to read:
            # when population covered the whole file (the common case on
            # capable kernels), eagerly allocating 64 MiB in each of up to 16
            # workers would transiently waste ~1 GiB per rank to read EOF.
            if offset >= os.fstat(f.fileno()).st_size:
                return 0
            buffer = memoryview(bytearray(_PREFETCH_CHUNK_SIZE_BYTES))
            f.seek(offset)
            while num_read := f.readinto(buffer):
                total_read += num_read
                if report_progress is not None:
                    report_progress(num_read)
        return total_read

    def prefetch_files(self, file_names: List[str], local_communicator=None):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices local_rank, local_rank + local_mpi_size, local_rank + 2*local_mpi_size, etc.
        if local_communicator is None:
            rank, size = local_mpi_rank(), local_mpi_size()
        else:
            rank = local_communicator.Get_rank()
            size = local_communicator.Get_size()
        local_file_names = file_names[rank::size]
        if len(local_file_names) == 0:
            return

        total_size = 0
        for file_name in local_file_names:
            try:
                total_size += os.path.getsize(file_name)
            except OSError:
                pass  # Missing files are tolerated, as in _prefetch_one_file.
        progress_lock = threading.Lock()
        prefetched_size = 0
        last_log_time = time.monotonic()

        def report_progress(num_bytes: int) -> None:
            # Periodic heartbeat: on slow storage, prefetching can run for
            # tens of minutes without completing a single file, and log
            # silence gets the process killed by output-stall watchdogs.
            # Deliberately progress-gated: it proves liveness only while
            # bytes are actually moving, so a fully hung mount still goes
            # silent and such watchdogs retain the ability to kill it.
            nonlocal prefetched_size, last_log_time
            with progress_lock:
                prefetched_size += num_bytes
                now = time.monotonic()
                if now - last_log_time < _PREFETCH_LOG_INTERVAL_SEC:
                    return
                last_log_time = now
                current_size = prefetched_size
            logger.info(
                f"Prefetch progress: {current_size / (1024**3):.2f}GB / "
                f"{total_size / (1024**3):.2f}GB (this rank's share).")

        max_workers = min(multiprocessing.cpu_count() * 2, 16,
                          len(local_file_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                executor.map(
                    lambda file_name: self._prefetch_one_file(
                        file_name, report_progress), local_file_names))
