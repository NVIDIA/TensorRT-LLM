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
from pathlib import Path
from typing import Any, Callable, List

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.mmap_utils import populate_file_pages
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import ENABLE_MULTI_DEVICE, mpi_comm, mpi_disabled
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_WEIGHT_CACHE_ENV = "TRTLLM_HF_WEIGHT_CACHE"
_WEIGHT_CACHE_MAX_ENTRIES_ENV = "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"
_NATIVE_IO_POLICY = "native"
_RANK_STRIPED_IO_POLICY = "rank_striped_read_ahead"
_NO_EFFECTIVE_IO_POLICY = "none"
_SUPPORTED_IO_POLICIES = frozenset({_NATIVE_IO_POLICY, _RANK_STRIPED_IO_POLICY})
_DEFAULT_PREFETCH_CHUNK_SIZE = 256 * 1024 * 1024
_PREFETCH_READ_SIZE = 8 * 1024 * 1024
_DEFAULT_PREFETCH_WORKERS_PER_RANK = 16
_DEFAULT_PREFETCH_WORKERS_PER_NODE = 64
_DEFAULT_HOST_MEMORY_HEADROOM_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_HOST_MEMORY_HEADROOM_FRACTION = 0.1
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
    """Last-load state used for logs, tests, and startup diagnostics."""

    requested: str
    selected: str
    activated: bool
    effective: str
    fallback_reason: str | None = None
    local_workers: int = 0
    assigned_bytes: int = 0
    completed_bytes: int = 0
    read_ahead_seconds: float = 0.0
    exposed_tail_seconds: float = 0.0


@dataclass(frozen=True)
class _CheckpointFilePlan:
    """A rank-coherent checkpoint discovery result."""

    file_kind: str
    weight_files: tuple[str, ...]
    discovery_signature: tuple[Any, ...]


class _RankStripedReadAheadSession:
    """Own background host reads and their node-local communicator.

    Chunk planning and all MPI operations stay on the caller thread. The
    coordinator thread executes only precomputed POSIX reads.
    """

    def __init__(
        self,
        loader: "HfWeightLoader",
        *,
        node_communicator,
        local_chunks: list[tuple[str, int, int]],
        max_workers: int,
        local_rank: int,
    ) -> None:
        self._loader = loader
        self._node_communicator = node_communicator
        self._local_chunks = local_chunks
        self._max_workers = max_workers
        self._local_rank = local_rank
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._completed_bytes = 0
        self._completed_bytes_lock = threading.Lock()
        self._started_at = time.perf_counter()
        self._completed_at: float | None = None
        self._read_error: Exception | None = None
        self._finished = False
        self._communicator_freed = False

    def start(self) -> None:
        if not self._local_chunks:
            self._completed_at = time.perf_counter()
            return
        self._thread = threading.Thread(
            target=self._run,
            name="trtllm-rank-striped-read-ahead",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._completed_at = time.perf_counter()
            raise

    def _record_chunk(self, length: int) -> None:
        with self._completed_bytes_lock:
            self._completed_bytes += length

    def _run(self) -> None:
        try:
            self._loader._prefetch_chunks(
                self._local_chunks,
                self._max_workers,
                cancel_event=self._cancel_event,
                completion_callback=self._record_chunk,
            )
        except Exception as error:
            self._read_error = error
        finally:
            self._completed_at = time.perf_counter()

    def cancel_and_close(self) -> Exception | None:
        """Cancel local work, join it, and release the communicator."""
        self._cancel_event.set()
        cleanup_error = None
        if self._thread is not None:
            self._thread.join()
        if self._completed_at is None:
            self._completed_at = time.perf_counter()
        try:
            self._free_communicator()
        except Exception as error:
            cleanup_error = error
        return cleanup_error

    def _free_communicator(self) -> None:
        if (self._node_communicator is not None
                and not self._communicator_freed):
            self._node_communicator.Free()
            self._communicator_freed = True

    def finish(self, body_error: BaseException | None = None) -> None:
        """Join readers and coordinate failures after materialization."""
        if self._finished:
            return
        self._finished = True
        if body_error is not None:
            self._cancel_event.set()

        tail_started = time.perf_counter()
        coordinated_body_error = self._loader._coordinate_rank_error(
            "rank-striped model materialization", body_error)
        if coordinated_body_error is not None:
            self._cancel_event.set()

        if self._thread is not None:
            self._thread.join()
        if self._completed_at is None:
            self._completed_at = time.perf_counter()

        coordinated_read_error = self._loader._coordinate_rank_error(
            "rank-striped background read-ahead", self._read_error)
        status = self._loader._last_checkpoint_io_status
        status.completed_bytes = self._completed_bytes
        status.read_ahead_seconds = self._completed_at - self._started_at
        status.exposed_tail_seconds = max(0.0,
                                          self._completed_at - tail_started)

        communicator_error = None
        try:
            self._free_communicator()
        except Exception as error:
            communicator_error = error
        coordinated_communicator_error = self._loader._coordinate_rank_error(
            "rank-striped communicator cleanup", communicator_error)

        if coordinated_body_error is not None:
            status.effective = _NO_EFFECTIVE_IO_POLICY
            status.fallback_reason = str(coordinated_body_error)
            self._loader._log_checkpoint_io_status()
            raise coordinated_body_error

        if coordinated_communicator_error is not None:
            status.effective = _NO_EFFECTIVE_IO_POLICY
            status.fallback_reason = str(coordinated_communicator_error)
            self._loader._log_checkpoint_io_status()
            raise coordinated_communicator_error

        if coordinated_read_error is None:
            status.effective = _RANK_STRIPED_IO_POLICY
        else:
            # Read-ahead is advisory. Native SafeTensors materialization has
            # already succeeded, so never reload a partially mutated model.
            status.effective = _NATIVE_IO_POLICY
            status.fallback_reason = str(coordinated_read_error)
            logger.warning(
                "Rank-striped checkpoint read-ahead degraded after "
                "activation; keeping the successfully materialized model "
                "without retrying the HF loader: "
                f"{coordinated_read_error}")

        self._loader._log_checkpoint_io_status()


@register_checkpoint_weight_loader("MX")
@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("mistral_large_3")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    def __init__(
        self,
        *,
        checkpoint_io_policy: str = _NATIVE_IO_POLICY,
        prefetch_chunk_size: int = _DEFAULT_PREFETCH_CHUNK_SIZE,
        prefetch_workers_per_node: int = _DEFAULT_PREFETCH_WORKERS_PER_NODE,
        prefetch_workers_per_rank: int = _DEFAULT_PREFETCH_WORKERS_PER_RANK,
        host_memory_headroom_bytes: int = _DEFAULT_HOST_MEMORY_HEADROOM_BYTES,
        host_memory_headroom_fraction:
        float = _DEFAULT_HOST_MEMORY_HEADROOM_FRACTION,
    ) -> None:
        if checkpoint_io_policy not in _SUPPORTED_IO_POLICIES:
            raise ValueError("checkpoint_io_policy must be one of "
                             f"{sorted(_SUPPORTED_IO_POLICIES)}, got "
                             f"{checkpoint_io_policy!r}")
        if prefetch_chunk_size <= 0:
            raise ValueError("prefetch_chunk_size must be positive")
        if prefetch_workers_per_node <= 0:
            raise ValueError("prefetch_workers_per_node must be positive")
        if prefetch_workers_per_rank <= 0:
            raise ValueError("prefetch_workers_per_rank must be positive")
        if host_memory_headroom_bytes < 0:
            raise ValueError("host_memory_headroom_bytes must be nonnegative")
        if not 0.0 <= host_memory_headroom_fraction < 1.0:
            raise ValueError(
                "host_memory_headroom_fraction must be in [0.0, 1.0)")

        self._checkpoint_io_policy = checkpoint_io_policy
        self._prefetch_chunk_size = prefetch_chunk_size
        self._prefetch_workers_per_node = prefetch_workers_per_node
        self._prefetch_workers_per_rank = prefetch_workers_per_rank
        self._host_memory_headroom_bytes = host_memory_headroom_bytes
        self._host_memory_headroom_fraction = host_memory_headroom_fraction
        self._last_checkpoint_io_status = _CheckpointIOStatus(
            requested=checkpoint_io_policy,
            selected=_NATIVE_IO_POLICY,
            activated=False,
            effective=_NO_EFFECTIVE_IO_POLICY,
        )

    @property
    def checkpoint_io_policy(self) -> str:
        return self._checkpoint_io_policy

    @property
    def last_checkpoint_io_status(self) -> _CheckpointIOStatus:
        """Return a snapshot of the most recent checkpoint I/O decision."""
        return replace(self._last_checkpoint_io_status)

    def _reset_checkpoint_io_status(self) -> None:
        selected = (self._checkpoint_io_policy if self._checkpoint_io_policy
                    == _NATIVE_IO_POLICY else _RANK_STRIPED_IO_POLICY)
        self._last_checkpoint_io_status = _CheckpointIOStatus(
            requested=self._checkpoint_io_policy,
            selected=selected,
            activated=False,
            effective=_NO_EFFECTIVE_IO_POLICY,
        )

    def _select_native_io(self, reason: str | None = None) -> None:
        status = self._last_checkpoint_io_status
        status.selected = _NATIVE_IO_POLICY
        status.activated = False
        status.effective = _NATIVE_IO_POLICY
        status.fallback_reason = reason
        self._log_checkpoint_io_status()

    def _log_checkpoint_io_status(self) -> None:
        status = self._last_checkpoint_io_status
        logger.info(
            f"Checkpoint I/O policy: requested={status.requested}, "
            f"selected={status.selected}, activated={status.activated}, "
            f"effective={status.effective}, fallback_reason="
            f"{status.fallback_reason or 'none'}, "
            f"local_workers={status.local_workers}, "
            f"assigned_bytes={status.assigned_bytes}, "
            f"completed_bytes={status.completed_bytes}, "
            f"read_ahead_seconds={status.read_ahead_seconds:.3f}, "
            f"exposed_tail_seconds={status.exposed_tail_seconds:.3f}.", )

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

    @classmethod
    def _get_active_node_load_context(cls) -> tuple[int, int, int]:
        """Return active-group local rank/size and conservative free memory.

        The process-wide ``local_comm`` is created at module import and may
        include colocated ranks that later select different model-load
        subcommunicators. Derive node membership from the active communicator
        instead so independent TRT-LLM instances cannot deadlock each other.
        """
        if (not ENABLE_MULTI_DEVICE or mpi_disabled()
                or mpi_comm().Get_size() == 1):
            return 0, 1, cls._get_effective_available_host_memory()

        communicator = mpi_comm()
        processor_name = _MPI.Get_processor_name()
        available_host_memory = None
        admission_error = None
        try:
            available_host_memory = cls._get_effective_available_host_memory()
        except Exception as error:
            admission_error = f"{type(error).__name__}: {error}"
        observations = communicator.allgather(
            (processor_name, available_host_memory, admission_error))
        for rank, (_, _, peer_error) in enumerate(observations):
            if peer_error is not None:
                raise RuntimeError(
                    "Rank "
                    f"{rank} failed during native checkpoint memory "
                    f"admission: {peer_error}")
        assert available_host_memory is not None
        local_ranks = [
            rank for rank, (peer_name, _, _) in enumerate(observations)
            if peer_name == processor_name
        ]
        active_rank = communicator.Get_rank()
        return (
            local_ranks.index(active_rank),
            len(local_ranks),
            min(observations[rank][1] for rank in local_ranks),
        )

    def _with_weight_cache(self, weight_files: List[str],
                           use_consolidated: bool,
                           load_fn) -> ConsumableWeightsDict:
        """Wrap ``load_fn`` with the optional raw-weight cache.

        Key -> hit -> evict-before-load (so CPU never holds the old cached and
        the new loading weights at once) -> load -> store. Distributed
        synchronization is owned by ``_load_weights_native`` so cache hits and
        misses execute the same active-communicator collective sequence.
        """
        cache_key = self._weight_files_cache_key(
            weight_files,
            use_consolidated) if self._is_weight_cache_enabled() else None
        if cache_key is not None:
            cached_weights = self._get_cached_weights(cache_key)
            if cached_weights is not None:
                logger.info("Reusing cached HF checkpoint weights.")
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
            use_consolidated: bool = False,
            weight_files: List[str] | None = None) -> dict[str, Any]:
        """Return a dict of name -> lazy safetensors slices.

        Values are ``safetensors`` PySafeSlice objects: ``v[:]`` (or any
        indexing) materializes only the requested bytes from the mmapped
        file. This lets a model's ``load_weights`` stream a huge checkpoint
        and read only its rank-local shard (e.g. Kimi K3 expert-parallel
        expert slices) without ever holding the full checkpoint in RAM.
        """
        if weight_files is None:
            weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
            # Same sharded-vs-consolidated selection as the eager path below:
            # when both flavors are present, keep only the requested one.
            filtered_weight_files = [
                x for x in weight_files
                if ("consolidated" in os.path.split(x)[1]) == use_consolidated
            ]
            if filtered_weight_files:
                weight_files = filtered_weight_files
        if not weight_files:
            raise RuntimeError(f"No safetensors files in {checkpoint_dir}.")
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
        """Load synchronously for direct callers.

        ModelLoader uses :meth:`open_weight_session` so rank-striped host reads
        may remain active while the returned tensors are materialized.
        """
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
        """Keep rank-striped read-ahead alive through model materialization."""
        self._reset_checkpoint_io_status()
        if self._checkpoint_io_policy == _NATIVE_IO_POLICY:
            weights = self._load_weights_native(checkpoint_dir, mapping,
                                                use_consolidated, **kwargs)
            self._select_native_io()
            yield weights
            return

        weights, session = self._open_rank_striped_read_ahead(
            checkpoint_dir, mapping, use_consolidated, **kwargs)
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
                session.finish(body_error)
            except Exception:
                if body_error is None:
                    raise
                logger.exception(
                    "Suppressing rank-striped session cleanup failure to "
                    "preserve the model-materialization exception.")

    def _load_weights_native(self,
                             checkpoint_dir: str,
                             mapping: Mapping,
                             use_consolidated: bool = False,
                             checkpoint_plan: _CheckpointFilePlan | None = None,
                             allow_prefetch: bool = True,
                             **kwargs) -> dict[str, Any]:
        del mapping, kwargs
        if checkpoint_plan is None:
            checkpoint_plan = self._get_coherent_checkpoint_plan(
                checkpoint_dir, use_consolidated, "native checkpoint discovery")

        weight_files = list(checkpoint_plan.weight_files)
        weights = None
        load_error = None
        try:
            if checkpoint_plan.file_kind == "lazy_safetensors":
                weights = self._load_lazy_safetensors(
                    checkpoint_dir,
                    use_consolidated,
                    weight_files=weight_files,
                )
            elif checkpoint_plan.file_kind == "safetensors":
                local_rank, local_size, effective_available = (
                    self._get_active_node_load_context())
                checkpoint_size = sum(
                    size for _, size in checkpoint_plan.discovery_signature[1])
                weights = self._load_native_safetensors(
                    weight_files,
                    use_consolidated,
                    allow_prefetch=allow_prefetch,
                    checkpoint_size=checkpoint_size,
                    effective_available=effective_available,
                    local_rank=local_rank,
                    local_size=local_size,
                )
            elif checkpoint_plan.file_kind in ("bin", "pth"):
                weights = self._with_weight_cache(
                    weight_files,
                    use_consolidated,
                    load_fn=lambda: self._load_weights_in_parallel(
                        weight_files,
                        self._load_bin_or_path_file,
                        "Loading bin weights in parallel",
                    ),
                )
            else:
                raise RuntimeError(
                    f"No weight files found in {checkpoint_dir}.")
        except Exception as error:
            load_error = error

        if (not ENABLE_MULTI_DEVICE or mpi_disabled()
                or mpi_comm().Get_size() == 1):
            if load_error is not None:
                raise load_error
            coordinated_load_error = None
        else:
            coordinated_load_error = self._coordinate_rank_error(
                "native checkpoint load", load_error)
        if coordinated_load_error is not None:
            raise coordinated_load_error
        assert weights is not None
        return weights

    def _discover_checkpoint_plan(
        self,
        checkpoint_dir: str,
        use_consolidated: bool,
    ) -> _CheckpointFilePlan:
        is_lazy_checkpoint = self._is_kimi_k3_checkpoint(checkpoint_dir)
        weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
        filtered_weight_files = [
            file_name for file_name in weight_files
            if ("consolidated" in os.path.split(file_name)[1]
                ) == use_consolidated
        ]
        if filtered_weight_files:
            weight_files = filtered_weight_files

        if is_lazy_checkpoint:
            file_kind = "lazy_safetensors"
        elif weight_files:
            file_kind = "safetensors"
        else:
            weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.bin"))
            if weight_files:
                file_kind = "bin"
            else:
                weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.pth"))
                file_kind = "pth" if weight_files else "missing"

        discovery_signature = (
            file_kind,
            tuple((os.path.basename(file_name), os.path.getsize(file_name))
                  for file_name in weight_files),
        )
        return _CheckpointFilePlan(
            file_kind=file_kind,
            weight_files=tuple(weight_files),
            discovery_signature=discovery_signature,
        )

    def _get_coherent_checkpoint_plan(
        self,
        checkpoint_dir: str,
        use_consolidated: bool,
        phase: str,
    ) -> _CheckpointFilePlan:
        checkpoint_plan = None
        discovery_error = None
        try:
            checkpoint_plan = self._discover_checkpoint_plan(
                checkpoint_dir, use_consolidated)
        except Exception as error:
            discovery_error = error

        coordinated_discovery_error = self._coordinate_rank_error(
            phase, discovery_error)
        if coordinated_discovery_error is not None:
            raise coordinated_discovery_error
        assert checkpoint_plan is not None

        signatures = self._allgather_rank_values(
            checkpoint_plan.discovery_signature)
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError(
                f"{phase} must match across all model-load ranks; received "
                f"{signatures}.")
        return checkpoint_plan

    def _open_rank_striped_read_ahead(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        use_consolidated: bool,
        **kwargs,
    ) -> tuple[dict[str, Any], _RankStripedReadAheadSession | None]:
        """Start bounded rank-striped reads before native materialization."""
        del kwargs
        if mapping.world_size > 1 and (not ENABLE_MULTI_DEVICE
                                       or mpi_disabled()):
            return self._fallback_to_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                "distributed rank-striped read-ahead requires the active MPI "
                "model-load communicator",
            )
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            communicator = mpi_comm()
            mapping_sizes = communicator.allgather(mapping.world_size)
            communicator_size = communicator.Get_size()
            if any(world_size != communicator_size
                   for world_size in mapping_sizes):
                raise RuntimeError(
                    "Rank-striped checkpoint I/O requires every "
                    "mapping.world_size to match the active MPI "
                    f"communicator size ({communicator_size}); received "
                    f"{mapping_sizes}.")

        checkpoint_plan = self._get_coherent_checkpoint_plan(
            checkpoint_dir,
            use_consolidated,
            "rank-striped checkpoint discovery",
        )
        weight_files = list(checkpoint_plan.weight_files)
        file_kind = checkpoint_plan.file_kind
        discovery_signature = checkpoint_plan.discovery_signature

        if file_kind != "safetensors":
            reasons = {
                "lazy_safetensors":
                "the checkpoint requires model-specific lazy SafeTensors "
                "loading",
                "bin":
                ".bin checkpoints use native checkpoint I/O",
                "pth":
                ".pth checkpoints use native checkpoint I/O",
                "missing":
                "no SafeTensors checkpoint files were found",
            }
            return self._fallback_to_native(checkpoint_dir,
                                            mapping,
                                            use_consolidated,
                                            reasons[file_kind],
                                            checkpoint_plan=checkpoint_plan)

        node_communicator = None
        split_error = None
        try:
            node_communicator = self._get_active_node_communicator()
        except Exception as error:
            split_error = error
        coordinated_split_error = self._coordinate_rank_error(
            "rank-striped node communicator creation", split_error)
        if coordinated_split_error is not None:
            self._close_node_communicator(
                node_communicator, "rank-striped communicator creation cleanup")
            return self._fallback_to_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                str(coordinated_split_error),
                checkpoint_plan=checkpoint_plan,
            )

        stats = None
        num_layers = None
        cache_enabled = None
        effective_available = None
        preflight_error = None
        try:
            stats = [(file_name, os.stat(file_name))
                     for file_name in weight_files]
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            cache_enabled = self._is_weight_cache_enabled()
            effective_available = self._get_effective_available_host_memory()
        except Exception as error:
            preflight_error = error

        coordinated_preflight_error = self._coordinate_rank_error(
            "rank-striped preflight", preflight_error)
        if coordinated_preflight_error is not None:
            self._close_node_communicator(node_communicator,
                                          "rank-striped preflight cleanup")
            raise coordinated_preflight_error
        assert stats is not None
        assert num_layers is not None
        assert cache_enabled is not None
        assert effective_available is not None

        checkpoint_size = sum(stat.st_size for _, stat in stats)
        backing_signature = tuple(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            for _, stat in stats)
        node_reason = None
        node_collective_error = None
        try:
            if node_communicator is not None:
                backing_signatures = node_communicator.allgather(
                    backing_signature)
                if any(signature != backing_signatures[0]
                       for signature in backing_signatures[1:]):
                    node_reason = (
                        "node-local ranks resolved the checkpoint to "
                        "different backing files")
                effective_available = node_communicator.allreduce(
                    effective_available, op=_MPI.MIN)
        except Exception as error:
            node_collective_error = error
        coordinated_node_error = self._coordinate_rank_error(
            "rank-striped node preflight", node_collective_error)
        if coordinated_node_error is not None:
            self._close_node_communicator(
                node_communicator, "rank-striped node preflight cleanup")
            return self._fallback_to_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                str(coordinated_node_error),
                checkpoint_plan=checkpoint_plan,
            )

        headroom = max(
            self._host_memory_headroom_bytes,
            int(effective_available * self._host_memory_headroom_fraction),
        )
        if cache_enabled:
            node_reason = (
                "the raw HF weight cache is enabled and requires the native "
                "collective sequence")
        elif num_layers != 0:
            node_reason = (
                "TLLM_OVERRIDE_LAYER_NUM requests partial checkpoint loading")
        elif checkpoint_size > max(0, effective_available - headroom):
            node_reason = (
                f"checkpoint bytes ({checkpoint_size}) exceed effective host "
                f"memory ({effective_available}) after startup headroom "
                f"({headroom})")

        policy_inputs = self._allgather_rank_values(
            (discovery_signature, num_layers, cache_enabled, node_reason))
        if any(inputs[0] != policy_inputs[0][0]
               for inputs in policy_inputs[1:]):
            self._close_node_communicator(node_communicator,
                                          "rank-striped selection cleanup")
            raise RuntimeError(
                "Rank-striped checkpoint selection changed during preflight.")
        fallback_reasons = [(rank, inputs[3])
                            for rank, inputs in enumerate(policy_inputs)
                            if inputs[3] is not None]
        if fallback_reasons:
            self._close_node_communicator(node_communicator,
                                          "rank-striped admission cleanup")
            rank, reason = fallback_reasons[0]
            return self._fallback_to_native(
                checkpoint_dir,
                mapping,
                use_consolidated,
                f"rank {rank}: {reason}",
                checkpoint_plan=checkpoint_plan,
            )

        local_rank, local_size = self._get_local_rank_and_size(
            node_communicator)
        session = None
        planning_error = None
        try:
            local_chunks, max_workers = self._local_prefetch_plan(
                weight_files, local_rank, local_size)
            session = _RankStripedReadAheadSession(
                self,
                node_communicator=node_communicator,
                local_chunks=local_chunks,
                max_workers=max_workers,
                local_rank=local_rank,
            )
            status = self._last_checkpoint_io_status
            status.selected = _RANK_STRIPED_IO_POLICY
            status.local_workers = max_workers
            status.assigned_bytes = sum(length for _, _, length in local_chunks)
        except Exception as error:
            planning_error = error

        coordinated_planning_error = self._coordinate_rank_error(
            "rank-striped read planning", planning_error)
        if coordinated_planning_error is not None:
            cleanup_error = None
            if session is not None:
                cleanup_error = session.cancel_and_close()
            else:
                try:
                    self._free_node_communicator(node_communicator)
                except Exception as error:
                    cleanup_error = error
            coordinated_cleanup_error = self._coordinate_rank_error(
                "rank-striped planning cleanup", cleanup_error)
            if coordinated_cleanup_error is not None:
                raise coordinated_cleanup_error
            return self._fallback_to_native_after_selection(
                checkpoint_dir,
                mapping,
                use_consolidated,
                str(coordinated_planning_error),
                checkpoint_plan=checkpoint_plan,
            )
        assert session is not None

        start_error = None
        try:
            session.start()
        except Exception as error:
            start_error = error
        coordinated_start_error = self._coordinate_rank_error(
            "rank-striped read-ahead start", start_error)
        if coordinated_start_error is not None:
            cleanup_error = session.cancel_and_close()
            coordinated_cleanup_error = self._coordinate_rank_error(
                "rank-striped start cleanup", cleanup_error)
            if coordinated_cleanup_error is not None:
                raise coordinated_cleanup_error
            return self._fallback_to_native_after_selection(
                checkpoint_dir,
                mapping,
                use_consolidated,
                str(coordinated_start_error),
                checkpoint_plan=checkpoint_plan,
            )

        self._last_checkpoint_io_status.activated = True
        logger.info("Rank-striped checkpoint read-ahead activated: "
                    f"local_rank={local_rank}, local_size={local_size}, "
                    f"workers={max_workers}, assigned_bytes="
                    f"{self._last_checkpoint_io_status.assigned_bytes}, "
                    f"checkpoint_bytes={checkpoint_size}.")

        weights = None
        mapping_error = None
        try:
            weights = self._load_weights_in_parallel(
                weight_files,
                self._load_safetensors_file,
                "Mapping safetensors weights in parallel",
            )
        except Exception as error:
            mapping_error = error
        coordinated_mapping_error = self._coordinate_rank_error(
            "rank-striped SafeTensors mapping", mapping_error)
        if coordinated_mapping_error is not None:
            cleanup_error = session.cancel_and_close()
            coordinated_cleanup_error = self._coordinate_rank_error(
                "rank-striped mapping cleanup", cleanup_error)
            if coordinated_cleanup_error is not None:
                logger.error(
                    "Rank-striped cleanup also failed after SafeTensors "
                    f"mapping failure: {coordinated_cleanup_error}")
            status = self._last_checkpoint_io_status
            status.effective = _NO_EFFECTIVE_IO_POLICY
            status.fallback_reason = str(coordinated_mapping_error)
            self._log_checkpoint_io_status()
            raise coordinated_mapping_error
        assert weights is not None
        return weights, session

    def _fallback_to_native(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        use_consolidated: bool,
        reason: str,
        *,
        checkpoint_plan: _CheckpointFilePlan | None = None,
    ) -> tuple[dict[str, Any], None]:
        weights = self._load_weights_native(
            checkpoint_dir,
            mapping,
            use_consolidated,
            checkpoint_plan=checkpoint_plan,
            allow_prefetch=False,
        )
        self._select_native_io(reason)
        return weights, None

    def _fallback_to_native_after_selection(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        use_consolidated: bool,
        reason: str,
        *,
        checkpoint_plan: _CheckpointFilePlan,
    ) -> tuple[dict[str, Any], None]:
        weights = self._load_weights_native(
            checkpoint_dir,
            mapping,
            use_consolidated,
            checkpoint_plan=checkpoint_plan,
            allow_prefetch=False,
        )
        status = self._last_checkpoint_io_status
        status.activated = False
        status.effective = _NATIVE_IO_POLICY
        status.fallback_reason = reason
        self._log_checkpoint_io_status()
        return weights, None

    @staticmethod
    def _allgather_rank_values(value):
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            return mpi_comm().allgather(value)
        return [value]

    def coordinate_checkpoint_io_request(self, mapping: Mapping) -> None:
        """Reject rank-divergent policies before either path enqueues work."""
        if not ENABLE_MULTI_DEVICE or mpi_disabled():
            return
        communicator = mpi_comm()
        communicator_size = communicator.Get_size()
        requests = communicator.allgather(
            (self._checkpoint_io_policy, mapping.world_size, communicator_size))
        mapping_sizes = [request[1] for request in requests]
        if any(world_size != communicator_size for world_size in mapping_sizes):
            raise RuntimeError(
                "Checkpoint I/O requires every mapping.world_size to match "
                f"the active MPI communicator size ({communicator_size}); "
                f"received {mapping_sizes}.")
        policies = [request[0] for request in requests]
        if any(peer_policy != policies[0] for peer_policy in policies[1:]):
            raise RuntimeError("checkpoint_io_policy must match across "
                               f"all model-load ranks; received {policies}.")

    @classmethod
    def _coordinate_rank_error(cls, phase: str,
                               error: BaseException | None) -> Exception | None:
        error_message = (None if error is None else
                         f"{type(error).__name__}: {error}")
        error_messages = cls._allgather_rank_values(error_message)
        for rank, rank_error in enumerate(error_messages):
            if rank_error is not None:
                return RuntimeError(
                    f"Rank {rank} failed during {phase}: {rank_error}")
        return None

    @staticmethod
    def _get_active_node_communicator():
        if (ENABLE_MULTI_DEVICE and not mpi_disabled()
                and mpi_comm().Get_size() > 1):
            return mpi_comm().Split_type(_MPI.COMM_TYPE_SHARED)
        return None

    @staticmethod
    def _free_node_communicator(node_communicator) -> None:
        if node_communicator is not None:
            node_communicator.Free()

    @classmethod
    def _close_node_communicator(cls, node_communicator, phase: str) -> None:
        cleanup_error = None
        try:
            cls._free_node_communicator(node_communicator)
        except Exception as error:
            cleanup_error = error
        coordinated_cleanup_error = cls._coordinate_rank_error(
            phase, cleanup_error)
        if coordinated_cleanup_error is not None:
            raise coordinated_cleanup_error

    @staticmethod
    def _get_local_rank_and_size(node_communicator=None) -> tuple[int, int]:
        if node_communicator is None:
            return 0, 1
        return node_communicator.Get_rank(), node_communicator.Get_size()

    def _load_native_safetensors(
        self,
        weight_files: List[str],
        use_consolidated: bool,
        *,
        allow_prefetch: bool,
        checkpoint_size: int,
        effective_available: int,
        local_rank: int,
        local_size: int,
    ) -> ConsumableWeightsDict:
        """Coordinate cache lookup and synchronous native prefetch."""
        cache_key = None
        cached_weights = None
        preparation_error = None
        try:
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            enable_prefetch = (allow_prefetch and num_layers == 0
                               and checkpoint_size < effective_available * 0.9)
            if self._is_weight_cache_enabled():
                cache_key = self._weight_files_cache_key(
                    weight_files, use_consolidated)
                cached_weights = self._get_cached_weights(cache_key)
                if cached_weights is None:
                    self._evict_to_make_room()
            if cached_weights is None and enable_prefetch:
                prefetch_size = sum(
                    os.path.getsize(file) for file in weight_files)
                logger.info(
                    f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
                )
                self.prefetch_files(weight_files,
                                    local_rank=local_rank,
                                    local_size=local_size)
        except Exception as error:
            preparation_error = error
        coordinated_preparation_error = self._coordinate_rank_error(
            "native checkpoint cache lookup and prefetch", preparation_error)
        if coordinated_preparation_error is not None:
            raise coordinated_preparation_error

        if cached_weights is not None:
            logger.info("Reusing cached HF checkpoint weights.")
            return cached_weights

        weights = self._load_weights_in_parallel(
            weight_files, self._load_safetensors_file,
            "Loading safetensors weights in parallel")
        if cache_key is not None:
            self._cache_loaded_weights(cache_key, weights)
        return weights

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

    @classmethod
    def _get_effective_available_host_memory(cls) -> int:
        """Return reclaim-aware host availability capped by cgroup limits."""
        host_available = psutil.virtual_memory().available
        cgroup_available = cls._get_cgroup_available_host_memory()
        if cgroup_available is None:
            return host_available
        return min(host_available, cgroup_available)

    @staticmethod
    def _get_cgroup_available_host_memory() -> int | None:
        """Best-effort cgroup-v1/v2 remaining-memory discovery."""
        relative_path = ""
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                fields = line.split(":", 2)
                if len(fields) == 3 and fields[0] == "0":
                    relative_path = fields[2].lstrip("/")
                    break
                if len(fields) == 3 and "memory" in fields[1].split(","):
                    relative_path = fields[2].lstrip("/")
        except (OSError, UnicodeError):
            pass

        roots = [Path("/sys/fs/cgroup")]
        if relative_path:
            roots.append(Path("/sys/fs/cgroup") / relative_path)
        v1_root = Path("/sys/fs/cgroup/memory")
        roots.append(v1_root / relative_path if relative_path else v1_root)

        available_values = []
        seen = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            for limit_name, current_name, unlimited_threshold in (
                ("memory.max", "memory.current", None),
                ("memory.limit_in_bytes", "memory.usage_in_bytes", 1 << 60),
            ):
                try:
                    raw_limit = (root / limit_name).read_text().strip()
                    if raw_limit == "max":
                        continue
                    limit = int(raw_limit)
                    if (unlimited_threshold is not None
                            and limit >= unlimited_threshold):
                        continue
                    current = int((root / current_name).read_text().strip())
                except (OSError, UnicodeError, ValueError):
                    continue
                available_values.append(max(0, limit - current))

        if not available_values:
            return None
        return min(available_values)

    @staticmethod
    def _distribute_worker_budget(
        local_size: int,
        workers_per_node: int,
        workers_per_rank: int,
    ) -> tuple[int, ...]:
        """Distribute an exact node budget as evenly as possible."""
        if local_size <= 0:
            raise ValueError("local_size must be positive")
        if workers_per_node <= 0 or workers_per_rank <= 0:
            raise ValueError("worker budgets must be positive")
        worker_budget = min(workers_per_node, local_size * workers_per_rank)
        base_workers, extra_workers = divmod(worker_budget, local_size)
        return tuple(base_workers + (rank < extra_workers)
                     for rank in range(local_size))

    def _local_prefetch_plan(
        self,
        file_names: List[str],
        local_rank: int,
        local_size: int,
    ) -> tuple[list[tuple[str, int, int]], int]:
        """Return complete, disjoint extents and this rank's worker count."""
        chunks = []
        for file_name in sorted(file_names):
            file_size = os.path.getsize(file_name)
            for offset in range(0, file_size, self._prefetch_chunk_size):
                chunks.append((file_name, offset,
                               min(self._prefetch_chunk_size,
                                   file_size - offset)))

        worker_counts = self._distribute_worker_budget(
            local_size,
            self._prefetch_workers_per_node,
            self._prefetch_workers_per_rank,
        )
        active_ranks = [
            rank for rank, worker_count in enumerate(worker_counts)
            if worker_count > 0
        ]
        if local_rank not in active_ranks or not chunks:
            return [], 0
        active_ordinal = active_ranks.index(local_rank)
        local_chunks = chunks[active_ordinal::len(active_ranks)]
        return local_chunks, min(worker_counts[local_rank], len(local_chunks))

    @staticmethod
    def _prefetch_one_chunk(
        file_name: str,
        offset: int,
        length: int,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Read one bounded file extent into the Linux page cache."""
        with open(file_name, "rb", buffering=0) as checkpoint_file:
            file_descriptor = checkpoint_file.fileno()
            read_offset = offset
            remaining = length
            while remaining > 0:
                if cancel_event is not None and cancel_event.is_set():
                    return
                read_size = min(remaining, _PREFETCH_READ_SIZE)
                data = os.pread(file_descriptor, read_size, read_offset)
                if not data:
                    raise OSError(
                        f"Unexpected EOF while reading {file_name} at byte "
                        f"offset {read_offset}.")
                bytes_read = len(data)
                read_offset += bytes_read
                remaining -= bytes_read

    def _prefetch_chunks(
        self,
        local_chunks: list[tuple[str, int, int]],
        max_workers: int,
        *,
        cancel_event: threading.Event,
        completion_callback: Callable[[int], None],
    ) -> None:
        """Execute a precomputed plan without MPI calls on worker threads."""
        if not local_chunks:
            return
        if max_workers <= 0:
            raise ValueError(
                "a nonempty read plan requires at least one worker")

        def prefetch_chunk(chunk: tuple[str, int, int]) -> None:
            try:
                self._prefetch_one_chunk(*chunk, cancel_event=cancel_event)
                if not cancel_event.is_set():
                    completion_callback(chunk[2])
            except Exception:
                cancel_event.set()
                raise

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(prefetch_chunk, chunk) for chunk in local_chunks
            ]
            try:
                for index, future in enumerate(futures):
                    if cancel_event.is_set():
                        for pending_future in futures[index:]:
                            pending_future.cancel()
                    if not future.cancelled():
                        future.result()
            except Exception:
                cancel_event.set()
                for pending_future in futures:
                    pending_future.cancel()
                raise

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

    def prefetch_files(self,
                       file_names: List[str],
                       *,
                       local_rank: int = 0,
                       local_size: int = 1):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        if local_size <= 0 or not 0 <= local_rank < local_size:
            raise ValueError("local rank and size must describe a valid group")
        # Each active-group rank prefetches a disjoint file stripe.
        local_file_names = file_names[local_rank::local_size]
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
