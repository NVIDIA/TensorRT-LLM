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
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, List, Sequence

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict, WeightBatchStream, WeightGroup)
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightLoadPlan, WeightLoadPolicy, normalize_weight_load_plan)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, local_mpi_barrier,
                                 local_mpi_comm, local_mpi_rank, local_mpi_size,
                                 mpi_comm, mpi_disabled)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_WEIGHT_CACHE_ENV = "TRTLLM_HF_WEIGHT_CACHE"
_WEIGHT_CACHE_MAX_ENTRIES_ENV = "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"
_WEIGHT_LOAD_PLAN_ENV = "TRTLLM_HF_WEIGHT_LOAD_PLAN"
_SHARED_HOST_BUFFER_BUDGET_ENV = "TRTLLM_HF_SHARED_HOST_BUFFER_BUDGET_BYTES"
_DEFAULT_PREFETCH_CHUNK_SIZE = 256 * 1024 * 1024
_DEFAULT_SHARED_HOST_BUFFER_BUDGET = 64 * 1024 * 1024 * 1024
_PREFETCH_READ_SIZE = 8 * 1024 * 1024
_DEFAULT_PREFETCH_WORKERS_PER_RANK = 16
_DEFAULT_PREFETCH_WORKERS_PER_NODE = 64
_GPU_BROADCAST_UNAVAILABLE_REASON = (
    "gpu_broadcast is not implemented: efficient GPU fan-out requires "
    "rank-aware final tensor placement and a topology-aware NCCL/P2P transport")
# Default to a single cached checkpoint: each entry pins a full copy of the
# raw weights in CPU RAM, so callers wanting cross-model caching must opt in
# via TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES.
_DEFAULT_WEIGHT_CACHE_MAX_ENTRIES = 1
_WEIGHT_CACHE_LOCK = threading.Lock()
_WEIGHT_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()


class _DirectReadAheadSession:
    """Own one direct-rank background read-ahead operation.

    Chunk planning and every MPI operation happen on the caller thread. The
    coordinator thread below only executes precomputed host reads.
    """

    def __init__(
        self,
        loader,
        *,
        node_communicator,
        local_chunks: list[tuple[str, int, int]],
        max_workers: int,
        local_rank: int,
        enabled: bool,
    ) -> None:
        self._loader = loader
        self._node_communicator = node_communicator
        self._local_chunks = local_chunks
        self._max_workers = max_workers
        self._local_rank = local_rank
        self._enabled = enabled
        self._started_at = time.perf_counter()
        self._completed_at: float | None = None
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._finished = False

    def start(self) -> None:
        if not self._enabled or not self._local_chunks:
            self._completed_at = time.perf_counter()
            return
        self._thread = threading.Thread(
            target=self._run,
            name="trtllm-direct-rank-read-ahead",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._completed_at = time.perf_counter()
            raise

    def _run(self) -> None:
        try:
            self._loader._prefetch_chunks(self._local_chunks,
                                          self._max_workers,
                                          cancel_event=self._cancel_event)
        except Exception as error:
            self._error = error
        finally:
            self._completed_at = time.perf_counter()

    def finish(self, body_error: BaseException | None = None) -> None:
        """Join and coordinate the operation on the caller thread."""
        if self._finished:
            return
        self._finished = True
        if body_error is not None:
            self._cancel_event.set()
        tail_started = time.perf_counter()
        coordinated_body_error = None
        coordinated_read_error = None
        finish_error = None
        communicator_error = None
        try:
            try:
                self._loader._raise_on_rank_error(
                    "direct_rank_read model materialization", body_error)
            except Exception as error:
                coordinated_body_error = error
                # A peer failed materialization. Stop this rank's remaining
                # host reads before waiting for the coordinator to exit.
                self._cancel_event.set()

            if self._thread is not None:
                self._thread.join()
            assert self._completed_at is not None

            try:
                self._loader._raise_on_rank_error(
                    "direct_rank_read background read-ahead", self._error)
            except Exception as error:
                coordinated_read_error = error

            read_ahead_elapsed = self._completed_at - self._started_at
            successful = (coordinated_body_error is None
                          and coordinated_read_error is None
                          and not self._cancel_event.is_set())
            if successful and self._node_communicator is not None:
                self._node_communicator.Barrier()

            if self._enabled and successful:
                exposed_read_tail = max(0.0, self._completed_at - tail_started)
                local_prefetch_size = sum(
                    length for _, _, length in self._local_chunks)
                local_throughput = (local_prefetch_size /
                                    max(read_ahead_elapsed, 1e-9) / (1024**3))
                logger.info(
                    "direct_rank_read local rank "
                    f"{self._local_rank} background read-ahead assigned "
                    f"{local_prefetch_size / (1024**3):.2f}GB and completed "
                    f"in {read_ahead_elapsed:.2f}s "
                    f"({local_throughput:.2f}GB/s rank-local read rate); "
                    "exposed read-ahead tail after model materialization was "
                    f"{exposed_read_tail:.2f}s. This is system-level "
                    "filesystem I/O and model-materialization/H2D overlap; "
                    "it does not use pinned asynchronous DMA.")
        except Exception as error:
            finish_error = error
        finally:
            if self._node_communicator is not None:
                try:
                    self._node_communicator.Free()
                except Exception as error:
                    communicator_error = error

        if coordinated_body_error is not None:
            raise coordinated_body_error
        if coordinated_read_error is not None:
            raise coordinated_read_error
        if finish_error is not None:
            raise finish_error
        if communicator_error is not None:
            raise communicator_error


class _SharedHostStreamSession:
    """Own transport cleanup and the caller-created node communicator."""

    def __init__(self, stream: WeightBatchStream, node_communicator) -> None:
        self._stream = stream
        self._node_communicator = node_communicator
        self._finished = False
        self._started_at = time.perf_counter()

    def finish(self, body_error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        cleanup_error = None
        communicator_error = None
        telemetry = None
        try:
            if body_error is not None:
                try:
                    self._stream.abort(body_error)
                except Exception:
                    logger.exception(
                        "Suppressing shared_host_producer abort failure to "
                        "preserve the model-load exception.")
            try:
                telemetry = getattr(self._stream, "telemetry", None)
            except Exception:
                logger.exception(
                    "Unable to snapshot shared_host_producer telemetry.")
            try:
                self._stream.finalize()
            except Exception as error:
                cleanup_error = error
        finally:
            if self._node_communicator is not None:
                try:
                    self._node_communicator.Free()
                except Exception as error:
                    communicator_error = error

        if body_error is not None:
            return
        if cleanup_error is not None:
            raise cleanup_error
        if communicator_error is not None:
            raise communicator_error
        if telemetry is not None:
            materialized_bytes = (telemetry.direct_view_bytes +
                                  telemetry.staged_bytes)
            direct_fraction = (telemetry.direct_view_bytes / materialized_bytes
                               if materialized_bytes else 0.0)
            logger.info("shared_host_producer rank materialization: "
                        f"world_rank={telemetry.world_rank}, "
                        f"node_rank={telemetry.node_rank}, "
                        "all_ranks_host_registered="
                        f"{telemetry.all_ranks_host_registered}, direct_groups="
                        f"{telemetry.direct_view_groups}, manifest_groups="
                        f"{telemetry.group_count}, direct_bytes="
                        f"{telemetry.direct_view_bytes}, staged_groups="
                        f"{telemetry.staged_groups}, staged_bytes="
                        f"{telemetry.staged_bytes}, direct_byte_fraction="
                        f"{direct_fraction:.6f}.")
        if telemetry is not None and telemetry.is_node_producer:
            elapsed = time.perf_counter() - self._started_at
            throughput = telemetry.bytes_published / max(elapsed, 1e-9) / (1024
                                                                           **3)
            logger.info(
                "shared_host_producer node stream published "
                f"{telemetry.bytes_published / (1024**3):.2f}GB in "
                f"{telemetry.batches_published} batches over {elapsed:.2f}s "
                f"({throughput:.2f}GB/s logical rate); shared double buffer="
                f"{telemetry.slot_count} x "
                f"{telemetry.slot_bytes / (1024**2):.0f}MiB "
                f"(configured={telemetry.configured_slot_bytes / (1024**2):.0f}MiB, "
                f"budget={telemetry.buffer_budget_bytes / (1024**3):.1f}GiB, "
                f"largest group={telemetry.largest_group_nbytes / (1024**2):.0f}MiB, "
                f"single-slot groups={telemetry.groups_fitting_single_slot}/"
                f"{telemetry.group_count}), "
                f"I/O workers={telemetry.io_workers}, host registration="
                f"{telemetry.host_registered} "
                f"({telemetry.host_registration_detail}), all-rank registration="
                f"{telemetry.all_ranks_host_registered}; direct materialization="
                f"{telemetry.direct_view_groups} groups/"
                f"{telemetry.direct_view_bytes / (1024**3):.2f}GB, staged="
                f"{telemetry.staged_groups} groups/"
                f"{telemetry.staged_bytes / (1024**3):.2f}GB.")


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
        weight_load_plan: (WeightLoadPolicy | str
                           | Sequence[WeightLoadPolicy | str] | None) = None,
        prefetch_chunk_size: int = _DEFAULT_PREFETCH_CHUNK_SIZE,
        prefetch_workers_per_rank: int | None = None,
        shared_host_buffer_budget: int | None = None,
    ) -> None:
        self._configured_weight_load_plan = weight_load_plan
        self._prefetch_chunk_size = prefetch_chunk_size
        self._prefetch_workers_per_rank = prefetch_workers_per_rank
        self._configured_shared_host_buffer_budget = (shared_host_buffer_budget)
        self._validate_weight_load_options()

    def _validate_weight_load_options(self) -> None:
        weight_load_plan = self._get_weight_load_plan()
        if self._prefetch_chunk_size <= 0:
            raise ValueError("prefetch_chunk_size must be positive")
        if (self._prefetch_workers_per_rank is not None
                and self._prefetch_workers_per_rank <= 0):
            raise ValueError("prefetch_workers_per_rank must be positive")
        if (WeightLoadPolicy.SHARED_HOST_PRODUCER in weight_load_plan
                and self._get_shared_host_buffer_budget()
                < 2 * self._prefetch_chunk_size):
            raise ValueError(
                "shared_host_buffer_budget must hold two prefetch chunks")

    def _get_weight_load_plan(self) -> WeightLoadPlan:
        value = self._configured_weight_load_plan
        if value is None:
            value = os.environ.get(_WEIGHT_LOAD_PLAN_ENV)
        if value is None and self._is_weight_cache_enabled():
            return (WeightLoadPolicy.LEGACY_FALLBACK, )
        return normalize_weight_load_plan(value)

    def _get_shared_host_buffer_budget(self) -> int:
        value = self._configured_shared_host_buffer_budget
        if value is None:
            raw_value = os.environ.get(_SHARED_HOST_BUFFER_BUDGET_ENV)
            if raw_value is not None:
                try:
                    value = int(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"{_SHARED_HOST_BUFFER_BUDGET_ENV} must be an integer"
                    ) from error
        if value is None:
            value = _DEFAULT_SHARED_HOST_BUFFER_BUDGET
        if value <= 0:
            raise ValueError("shared_host_buffer_budget must be positive")
        return value

    def requires_initialized_mapper_for_session(self) -> bool:
        """Whether shared-host preflight can be reached by this plan."""
        first_non_gpu_policy = next(
            (policy for policy in self._get_weight_load_plan()
             if policy != WeightLoadPolicy.GPU_BROADCAST), None)
        return first_non_gpu_policy == WeightLoadPolicy.SHARED_HOST_PRODUCER

    def _get_coordinated_weight_load_plan(
            self,
            mapping: Mapping,
            checkpoint_format: str,
            load_format: Any = None) -> tuple[WeightLoadPlan, str | None]:
        """Select one path after validating the active MPI communicator."""
        weight_load_plan = self._get_weight_load_plan()
        load_format_name = getattr(load_format, "name", load_format)
        # ModelLoader selects one load format for the whole model-loading
        # group. Coordinate only the HF disk modes: GMS and format-specific
        # loaders are not guaranteed to call this disk path on every rank.
        # Including the disk-format value still catches AUTO/None divergence
        # in direct HfWeightLoader calls, where both values enter this branch.
        requires_consensus = (checkpoint_format == "HF"
                              and load_format_name in (None, "AUTO"))
        coordination_error = self._cooperative_coordination_error(
            mapping, checkpoint_format, load_format_name)
        if (requires_consensus and ENABLE_MULTI_DEVICE and not mpi_disabled()
                and mpi_comm().Get_size() == mapping.world_size):
            selection = (tuple(policy.value for policy in weight_load_plan),
                         load_format_name)
            selections = mpi_comm().allgather(selection)
            if any(selection != selections[0] for selection in selections[1:]):
                raise RuntimeError(
                    "Weight-load plan and load format must match across "
                    f"all MPI ranks; received {selections}")
        return weight_load_plan, coordination_error

    @staticmethod
    def _cooperative_coordination_error(mapping: Mapping,
                                        checkpoint_format: str,
                                        load_format: Any) -> str | None:
        if checkpoint_format != "HF":
            return None
        if (load_format in (None, "AUTO") and ENABLE_MULTI_DEVICE
                and not mpi_disabled()):
            communicator_size = mpi_comm().Get_size()
            if communicator_size != mapping.world_size:
                return (
                    "the active MPI communicator size "
                    f"({communicator_size}) does not match mapping.world_size "
                    f"({mapping.world_size})")
        if mapping.world_size > 1 and load_format is None:
            return ("distributed cooperative loading requires an explicit "
                    "AUTO load format")
        return None

    @staticmethod
    def _cooperative_ineligibility_reason(
            model: Any,
            mapping: Mapping,
            *,
            checkpoint_format: str | None,
            uses_custom_weight_mapper: bool,
            load_format: Any = None) -> str | None:
        # These policies only read immutable file extents into the OS page
        # cache and then return the same complete raw tensor dictionary as the
        # legacy mmap path. Model architecture, mapper, quantization and
        # parallelism therefore do not affect byte-read correctness.
        del model, uses_custom_weight_mapper
        if checkpoint_format != "HF":
            return f"checkpoint format {checkpoint_format or 'unknown'} is not supported"
        if load_format is not None and getattr(load_format, "name",
                                               load_format) != "AUTO":
            return f"load format {getattr(load_format, 'name', load_format)} is not supported"
        if mapping.world_size > 1 and mpi_disabled():
            return "distributed cooperative loading requires MPI-launched ranks"
        return None

    @staticmethod
    def _coordinate_cooperative_ineligibility_reason(
            reason: str | None) -> str | None:
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            reasons = mpi_comm().allgather(reason)
        else:
            reasons = [reason]
        for rank, rank_reason in enumerate(reasons):
            if rank_reason is not None:
                return f"rank {rank}: {rank_reason}"
        return None

    @staticmethod
    def _nested_partial_load_ineligibility_reason(model: Any) -> str | None:
        """Return the first destination module that rejects partial loading."""
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            return None

        for module_name, module in named_modules():
            display_name = module_name or "<root>"
            try:
                capability = getattr(module, "supports_partial_weight_loading",
                                     None)
            except Exception as error:
                return (f"destination module {display_name!r} "
                        f"({type(module).__name__}) failed its partial-load "
                        f"capability check: {type(error).__name__}: {error}")
            # Only modules that explicitly publish this contract participate.
            # Unknown module types remain eligible for their existing loading
            # path rather than being rejected by absence of an opt-in marker.
            if capability is not False:
                continue
            quant_method = getattr(module, "quant_method", None)
            quant_details = (f", quant_method={type(quant_method).__name__}"
                             if quant_method is not None else "")
            return (
                f"destination module {display_name!r} "
                f"({type(module).__name__}{quant_details}) does not support "
                "allow_partial_loading")
        return None

    @classmethod
    def _shared_host_ineligibility_reason(
        cls,
        model: Any,
        mapping: Mapping,
        *,
        checkpoint_format: str,
        uses_custom_weight_mapper: bool,
        load_format: Any,
        supports_weight_stream: bool,
        model_supports_partial_loading: bool,
        weight_mapper: Any,
        weight_groups: Sequence[Any] | None,
        preflight_error: str | None,
    ) -> str | None:
        """Preflight model-side requirements for incremental materialization."""
        reason = cls._cooperative_ineligibility_reason(
            model,
            mapping,
            checkpoint_format=checkpoint_format,
            uses_custom_weight_mapper=uses_custom_weight_mapper,
            load_format=load_format)
        if reason is not None:
            return reason
        if not ENABLE_MULTI_DEVICE or mpi_disabled():
            return ("shared_host_producer requires active MPI world and "
                    "node-local communicators")
        if not supports_weight_stream:
            return ("shared_host_producer requires an open weight session; "
                    "synchronous load_weights callers cannot consume a stream")
        if not model_supports_partial_loading:
            return ("the model load_weights method does not support "
                    "allow_partial_loading")
        nested_reason = cls._nested_partial_load_ineligibility_reason(model)
        if nested_reason is not None:
            return nested_reason
        model_config = getattr(model, "model_config", None)
        load_balancer = getattr(model_config, "moe_load_balancer", None)
        if (load_balancer is not None
                and getattr(load_balancer, "layer_updates_per_iter", 0) > 0):
            return ("shared_host_producer does not support dynamic MoE load "
                    "balancing because it retains raw checkpoint tensors "
                    "beyond a bounded stream batch")
        if getattr(model_config, "enable_min_latency", False):
            return ("shared_host_producer does not support min-latency model "
                    "loading because its eager weight derivation is not "
                    "compatible with deferred partial-load finalization")
        if weight_mapper is None:
            return "an initialized weight mapper is required"
        if not callable(getattr(weight_mapper, "get_weight_groups", None)):
            return ("the initialized weight mapper does not provide an "
                    "incremental weight-group manifest")
        if preflight_error is not None:
            return preflight_error
        if weight_groups is None:
            return ("the initialized weight mapper did not declare a safe "
                    "incremental weight-group manifest")
        return None

    def _resolve_weight_load_policy(
        self,
        plan: WeightLoadPlan,
        model: Any,
        mapping: Mapping,
        *,
        checkpoint_format: str,
        uses_custom_weight_mapper: bool,
        load_format: Any,
        coordination_error: str | None,
        supports_weight_stream: bool = False,
        model_supports_partial_loading: bool = False,
        weight_mapper: Any = None,
        shared_host_weight_groups: Sequence[Any] | None = None,
        shared_host_preflight_error: str | None = None,
    ) -> tuple[WeightLoadPolicy, list[tuple[WeightLoadPolicy, str]]]:
        """Resolve an ordered policy list before strategy collectives begin."""
        if coordination_error is not None:
            raise RuntimeError("Weight loading cannot coordinate ranks: "
                               f"{coordination_error}")
        skipped = []
        policy_reasons: dict[WeightLoadPolicy, str | None] = {}
        for policy in plan:
            if policy == WeightLoadPolicy.LEGACY_FALLBACK:
                return policy, skipped
            if policy == WeightLoadPolicy.GPU_BROADCAST:
                skipped.append((policy, _GPU_BROADCAST_UNAVAILABLE_REASON))
                continue

            if policy not in policy_reasons:
                if policy == WeightLoadPolicy.SHARED_HOST_PRODUCER:
                    reason = self._shared_host_ineligibility_reason(
                        model,
                        mapping,
                        checkpoint_format=checkpoint_format,
                        uses_custom_weight_mapper=uses_custom_weight_mapper,
                        load_format=load_format,
                        supports_weight_stream=supports_weight_stream,
                        model_supports_partial_loading=(
                            model_supports_partial_loading),
                        weight_mapper=weight_mapper,
                        weight_groups=shared_host_weight_groups,
                        preflight_error=shared_host_preflight_error)
                else:
                    reason = self._cooperative_ineligibility_reason(
                        model,
                        mapping,
                        checkpoint_format=checkpoint_format,
                        uses_custom_weight_mapper=uses_custom_weight_mapper,
                        load_format=load_format)
                if getattr(load_format, "name", load_format) == "AUTO":
                    reason = self._coordinate_cooperative_ineligibility_reason(
                        reason)
                policy_reasons[policy] = reason
            reason = policy_reasons[policy]
            if reason is None:
                return policy, skipped
            skipped.append((policy, reason))

        details = "; ".join(f"{policy.value}: {reason}"
                            for policy, reason in skipped)
        raise RuntimeError("No executable weight-load policy remains in the "
                           f"configured plan ({details})")

    @staticmethod
    def _coordinate_checkpoint_discovery(weight_files: List[str],
                                         file_kind: str, mapping: Mapping,
                                         checkpoint_format: str,
                                         load_format: Any) -> None:
        """Reject rank-divergent HF/AUTO file discovery before branching."""
        load_format_name = getattr(load_format, "name", load_format)
        if (checkpoint_format != "HF" or load_format_name not in (None, "AUTO")
                or not ENABLE_MULTI_DEVICE or mpi_disabled()):
            return

        coordination_error = HfWeightLoader._cooperative_coordination_error(
            mapping, checkpoint_format, load_format_name)
        if coordination_error is not None:
            raise RuntimeError("Weight loading cannot coordinate ranks: "
                               f"{coordination_error}")

        try:
            signature = (
                file_kind,
                tuple((os.path.basename(file), os.path.getsize(file))
                      for file in weight_files),
            )
        except Exception as error:
            signature = ("error", type(error).__name__, str(error))
        signatures = mpi_comm().allgather(signature)
        if any(value != signatures[0] for value in signatures[1:]):
            raise RuntimeError(
                "Checkpoint file discovery must match across all MPI ranks; "
                f"received {signatures}")
        if signature[0] == "error":
            raise RuntimeError("Checkpoint file discovery failed: "
                               f"{signature[1]}: {signature[2]}")

    @staticmethod
    def _get_shared_host_weight_groups(
            keys: Sequence[str], weight_mapper: Any) -> tuple[WeightGroup, ...]:
        """Build and validate an atomic manifest without starting transport."""
        if len(set(keys)) != len(keys):
            raise ValueError(
                "SafeTensors checkpoint contains duplicate tensor names")

        groups = weight_mapper.get_weight_groups(keys)
        if groups is None:
            if not getattr(weight_mapper, "single_tensor_groups_safe", False):
                raise ValueError(
                    "the initialized weight mapper did not declare a safe "
                    "incremental weight-group manifest")
            groups = [WeightGroup(group_id=key, keys=(key, )) for key in keys]
        groups = tuple(groups)
        if not groups:
            raise ValueError("the incremental weight-group manifest is empty")

        group_ids = [group.group_id for group in groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("incremental weight-group IDs must be unique")
        grouped_keys = [key for group in groups for key in group.keys]
        grouped_key_set = set(grouped_keys)
        checkpoint_key_set = set(keys)
        if len(grouped_keys) != len(grouped_key_set):
            raise ValueError(
                "incremental weight groups contain duplicate tensor names")
        if grouped_key_set != checkpoint_key_set:
            missing = sorted(checkpoint_key_set - grouped_key_set)
            unexpected = sorted(grouped_key_set - checkpoint_key_set)
            details = []
            if missing:
                details.append(f"missing {missing[:5]}")
            if unexpected:
                details.append(f"unexpected {unexpected[:5]}")
            raise ValueError(
                "incremental weight groups must partition checkpoint keys "
                "exactly (" + "; ".join(details) + ")")
        return groups

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
    def _get_local_available_host_memory(node_communicator=None) -> int:
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
        if node_communicator is not None:
            return node_communicator.allreduce(available_host_memory,
                                               op=_MPI.MIN)
        if ENABLE_MULTI_DEVICE:
            return local_mpi_comm().allreduce(available_host_memory,
                                              op=_MPI.MIN)
        return available_host_memory

    def _with_weight_cache(self,
                           weight_files: List[str],
                           use_consolidated: bool,
                           mirror_load_collectives: bool,
                           load_fn,
                           node_communicator=None) -> ConsumableWeightsDict:
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
                    self._get_local_available_host_memory(node_communicator)
                    self._node_barrier(node_communicator)
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
        """Load weights synchronously for public/direct callers."""
        weights, pending_session = self._load_weights_impl(
            checkpoint_dir,
            mapping,
            use_consolidated,
            defer_direct_read_ahead=False,
            **kwargs)
        assert pending_session is None
        return weights

    @contextmanager
    def open_weight_session(
            self,
            checkpoint_dir: str,
            mapping: Mapping,
            use_consolidated: bool = False,
            **kwargs) -> Iterator[dict[str, Any]
                                  | WeightBatchStream]:
        """Keep policy-owned I/O resources alive during materialization."""
        weights, pending_session = self._load_weights_impl(
            checkpoint_dir,
            mapping,
            use_consolidated,
            defer_direct_read_ahead=True,
            **kwargs)
        body_error = None
        try:
            yield weights
        except BaseException as error:
            body_error = error
            raise
        finally:
            if pending_session is not None:
                try:
                    pending_session.finish(body_error)
                except Exception:
                    if body_error is None:
                        raise
                    logger.exception(
                        "Suppressing weight-session cleanup failure to preserve "
                        "the model-load exception.")

    def _load_weights_impl(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        use_consolidated: bool,
        *,
        defer_direct_read_ahead: bool,
        **kwargs,
    ) -> tuple[dict[str, Any] | WeightBatchStream, _DirectReadAheadSession
               | _SharedHostStreamSession | None]:
        load_format = kwargs.get("_load_format")
        checkpoint_format = kwargs.get("_checkpoint_format", "HF")
        weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former unless specified explicitly, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files
            if ("consolidated" in os.path.split(x)[1]) == use_consolidated
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        file_kind = "safetensors"
        if not weight_files:
            weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.bin"))
            file_kind = "bin"
        if not weight_files:
            weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.pth"))
            file_kind = "pth"
        if not weight_files:
            file_kind = "missing"

        self._coordinate_checkpoint_discovery(weight_files, file_kind, mapping,
                                              checkpoint_format, load_format)

        if file_kind == "safetensors":
            weight_load_plan, coordination_error = (
                self._get_coordinated_weight_load_plan(mapping,
                                                       checkpoint_format,
                                                       load_format))
            model = kwargs.get("model")
            weight_mapper = kwargs.get("_weight_mapper")
            model_supports_partial_loading = kwargs.get(
                "_model_supports_partial_loading", False)
            shared_host_preflight = None
            shared_host_weight_groups = None
            shared_host_preflight_error = None

            # Avoid even SafeTensors header parsing unless shared-host
            # streaming is the first potentially executable policy. Direct
            # read-ahead and page-cache prefetch have the same distributed
            # eligibility requirements, so a shared policy after either one
            # cannot become the selected policy on its own.
            first_non_gpu_policy = next(
                (candidate for candidate in weight_load_plan
                 if candidate != WeightLoadPolicy.GPU_BROADCAST), None)
            if (coordination_error is None and first_non_gpu_policy
                    == WeightLoadPolicy.SHARED_HOST_PRODUCER):
                cheap_reason = self._shared_host_ineligibility_reason(
                    model,
                    mapping,
                    checkpoint_format=checkpoint_format,
                    uses_custom_weight_mapper=kwargs.get(
                        "_uses_custom_weight_mapper", False),
                    load_format=load_format,
                    supports_weight_stream=defer_direct_read_ahead,
                    model_supports_partial_loading=(
                        model_supports_partial_loading),
                    weight_mapper=weight_mapper,
                    weight_groups=(),
                    preflight_error=None)
                if getattr(load_format, "name", load_format) == "AUTO":
                    cheap_reason = (
                        self._coordinate_cooperative_ineligibility_reason(
                            cheap_reason))
                if cheap_reason is None:
                    try:
                        from tensorrt_llm._torch.models.checkpoints.hf.shared_host_stream import \
                            prepare_shared_host_weight_stream

                        shared_host_preflight = (
                            prepare_shared_host_weight_stream(weight_files))
                        shared_host_weight_groups = (
                            self._get_shared_host_weight_groups(
                                shared_host_preflight.keys, weight_mapper))
                    except Exception as error:
                        shared_host_preflight_error = (
                            "shared-host stream preflight failed: "
                            f"{type(error).__name__}: {error}")

                    if getattr(load_format, "name", load_format) == "AUTO":
                        shared_host_preflight_error = (
                            self._coordinate_cooperative_ineligibility_reason(
                                shared_host_preflight_error))
                    if shared_host_preflight_error is None:
                        assert shared_host_weight_groups is not None
                        if ENABLE_MULTI_DEVICE and not mpi_disabled():
                            manifest_signature = tuple(
                                (group.group_id, group.keys)
                                for group in shared_host_weight_groups)
                            manifest_signatures = mpi_comm().allgather(
                                manifest_signature)
                            if any(signature != manifest_signatures[0]
                                   for signature in manifest_signatures[1:]):
                                shared_host_preflight_error = (
                                    "incremental weight-group manifests must "
                                    "match across all MPI ranks")

            def resolve_policy(plan: WeightLoadPlan):
                return self._resolve_weight_load_policy(
                    plan,
                    model,
                    mapping,
                    checkpoint_format=checkpoint_format,
                    uses_custom_weight_mapper=kwargs.get(
                        "_uses_custom_weight_mapper", False),
                    load_format=load_format,
                    coordination_error=coordination_error,
                    supports_weight_stream=defer_direct_read_ahead,
                    model_supports_partial_loading=(
                        model_supports_partial_loading),
                    weight_mapper=weight_mapper,
                    shared_host_weight_groups=shared_host_weight_groups,
                    shared_host_preflight_error=(shared_host_preflight_error))

            policy, skipped = resolve_policy(weight_load_plan)
            if policy == WeightLoadPolicy.SHARED_HOST_PRODUCER:
                assert shared_host_preflight is not None
                assert shared_host_weight_groups is not None
                node_communicator = self._get_active_node_communicator()
                try:
                    from tensorrt_llm._torch.models.checkpoints.hf.shared_host_stream import \
                        open_shared_host_weight_stream

                    world_communicator = (mpi_comm() if ENABLE_MULTI_DEVICE
                                          and not mpi_disabled() else None)
                    stream = open_shared_host_weight_stream(
                        shared_host_preflight,
                        node_communicator,
                        world_communicator,
                        group_manifest=shared_host_weight_groups,
                        slot_bytes=self._prefetch_chunk_size,
                        buffer_budget_bytes=(
                            self._get_shared_host_buffer_budget()),
                        io_workers=(self._prefetch_workers_per_rank
                                    or _DEFAULT_PREFETCH_WORKERS_PER_NODE),
                        strict=len(weight_load_plan) == 1)
                except BaseException:
                    if node_communicator is not None:
                        node_communicator.Free()
                    raise
                if stream is not None:
                    telemetry = stream.telemetry
                    if self._is_weight_cache_enabled():
                        logger.warning(
                            "The HF raw-weight cache is ignored by "
                            "shared_host_producer because it cannot mirror "
                            "the incremental collective sequence.")
                    logger.info(
                        "Using shared_host_producer bounded SafeTensors "
                        f"streaming for {type(model).__name__} "
                        f"(TP={mapping.tp_size}, PP={mapping.pp_size}); slots="
                        f"{telemetry.slot_count} x "
                        f"{telemetry.slot_bytes / (1024**2):.0f}MiB "
                        f"(configured {telemetry.configured_slot_bytes / (1024**2):.0f}MiB), "
                        f"single-slot atomic groups="
                        f"{telemetry.groups_fitting_single_slot}/"
                        f"{telemetry.group_count}, largest group="
                        f"{telemetry.largest_group_nbytes / (1024**2):.0f}MiB, "
                        f"all-rank CUDA host registration="
                        f"{telemetry.all_ranks_host_registered}.")
                    return stream, _SharedHostStreamSession(
                        stream, node_communicator)

                if node_communicator is not None:
                    node_communicator.Free()
                skipped.append(
                    (WeightLoadPolicy.SHARED_HOST_PRODUCER,
                     "the shared-host transport is unavailable on this host"))
                shared_index = weight_load_plan.index(
                    WeightLoadPolicy.SHARED_HOST_PRODUCER)
                remaining_plan = weight_load_plan[shared_index + 1:]
                policy, additionally_skipped = resolve_policy(remaining_plan)
                skipped.extend(additionally_skipped)

            if skipped and checkpoint_format == "HF":
                skipped_details = "; ".join(
                    f"{skipped_policy.value}: {reason}"
                    for skipped_policy, reason in skipped)
                logger.info(
                    f"Resolved weight-load policy to {policy.value}; skipped "
                    f"{skipped_details}.")
            if policy in (WeightLoadPolicy.DIRECT_RANK_READ,
                          WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH):
                if self._is_weight_cache_enabled():
                    logger.warning(
                        "The HF raw-weight cache is ignored by cooperative "
                        "SafeTensors policies because it does not yet mirror "
                        "their collective sequence.")
                model_type = type(model).__name__
                logger.info(f"Using {policy.value} SafeTensors loading for "
                            f"{model_type} (TP={mapping.tp_size}, "
                            f"PP={mapping.pp_size}).")
                if (defer_direct_read_ahead
                        and policy == WeightLoadPolicy.DIRECT_RANK_READ):
                    return self._load_direct_rank_read_with_background_readahead(
                        weight_files)
                return self._load_cooperative_policy(weight_files, policy), None
            if policy != WeightLoadPolicy.LEGACY_FALLBACK:
                raise RuntimeError(
                    f"Weight-load policy {policy.value} has no executor")
            use_active_communicator = (checkpoint_format == "HF" and getattr(
                load_format, "name", load_format) in (None, "AUTO"))
            return (self._load_legacy_safetensors(
                weight_files,
                use_consolidated,
                use_active_communicator=use_active_communicator), None)

        if file_kind in ("bin", "pth"):
            weight_load_plan, coordination_error = (
                self._get_coordinated_weight_load_plan(mapping,
                                                       checkpoint_format,
                                                       load_format))
            if coordination_error is not None:
                raise RuntimeError("Weight loading cannot coordinate ranks: "
                                   f"{coordination_error}")
            if WeightLoadPolicy.LEGACY_FALLBACK not in weight_load_plan:
                configured = ", ".join(policy.value
                                       for policy in weight_load_plan)
                raise RuntimeError(
                    ".bin/.pth checkpoints require legacy_fallback in the "
                    f"weight-load plan; configured policies: {configured}")
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=False,
                load_fn=lambda: self._load_weights_in_parallel(
                    weight_files, self._load_bin_or_path_file,
                    "Loading bin weights in parallel")), None

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _load_legacy_safetensors(
            self, weight_files: List[str], use_consolidated: bool, *,
            use_active_communicator: bool) -> ConsumableWeightsDict:
        if not use_active_communicator:
            # MX/GMS/format-specific disk fallback can be rank-local. Do not
            # enter any MPI collective that assumes peer ranks also fell back.
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=False,
                load_fn=lambda: self._load_weights_in_parallel(
                    weight_files, self._load_safetensors_file,
                    "Loading safetensors weights in parallel"))

        node_communicator = self._get_active_node_communicator()
        try:
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=True,
                load_fn=lambda: self._prefetch_and_load(weight_files,
                                                        node_communicator),
                node_communicator=node_communicator)
        finally:
            if node_communicator is not None:
                node_communicator.Free()

    def _prefetch_and_load(self,
                           weight_files: List[str],
                           node_communicator=None) -> ConsumableWeightsDict:
        prefetch_size, enable_prefetch = self._get_prefetch_policy(
            weight_files, node_communicator)
        if enable_prefetch:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
            )
            self.prefetch_files(weight_files, node_communicator)
        # Sync all local ranks unconditionally. `enable_prefetch` depends on
        # `psutil.virtual_memory().available`, a per-rank volatile value, so
        # different ranks may take different branches; gating the barrier on
        # it would deadlock between ranks that prefetched and ranks that
        # skipped. Ranks that didn't prefetch reach the barrier immediately.
        self._node_barrier(node_communicator)

        return self._load_weights_in_parallel(
            weight_files, self._load_safetensors_file,
            "Loading safetensors weights in parallel")

    def _load_cooperative_policy(
            self, weight_files: List[str],
            policy: WeightLoadPolicy) -> ConsumableWeightsDict:
        """Prefetch SafeTensors with one selected host I/O assignment."""
        if policy not in (WeightLoadPolicy.DIRECT_RANK_READ,
                          WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH):
            raise ValueError(
                f"Policy {policy.value} is not a cooperative host policy")
        node_communicator = self._get_active_node_communicator()
        try:
            return self._load_cooperative_policy_with_communicator(
                weight_files, policy, node_communicator)
        finally:
            if node_communicator is not None:
                node_communicator.Free()

    def _load_direct_rank_read_with_background_readahead(
        self, weight_files: List[str]
    ) -> tuple[ConsumableWeightsDict, _DirectReadAheadSession]:
        """Map weights now and defer direct-rank read-ahead completion.

        The caller must finish the returned session after it consumes the
        weights. That lets page-cache reads overlap tensor materialization and
        the model's existing H2D copies without changing tensor placement.
        """
        node_communicator = self._get_active_node_communicator()
        session = None
        start_attempted = False
        try:
            prefetch_size, enable_prefetch = (
                self._get_cooperative_prefetch_policy(weight_files,
                                                      node_communicator))
            local_rank = None
            planning_error = None
            try:
                local_rank, local_size = self._get_local_rank_and_size(
                    node_communicator)
                local_chunks = self._local_prefetch_chunks(
                    weight_files, WeightLoadPolicy.DIRECT_RANK_READ,
                    node_communicator)
                max_workers = self._get_prefetch_worker_count(
                    WeightLoadPolicy.DIRECT_RANK_READ, local_size,
                    len(local_chunks))
                session = _DirectReadAheadSession(
                    self,
                    node_communicator=node_communicator,
                    local_chunks=local_chunks,
                    max_workers=max_workers,
                    local_rank=local_rank,
                    enabled=enable_prefetch,
                )
            except Exception as error:
                planning_error = error
            self._raise_on_rank_error("direct_rank_read read-ahead planning",
                                      planning_error)
            assert session is not None
            assert local_rank is not None
            if enable_prefetch:
                logger.info(
                    "direct_rank_read is starting background read-ahead for "
                    f"{prefetch_size / (1024**3):.2f}GB of checkpoint data "
                    "in bounded chunks.")
            else:
                logger.info(
                    "Skipping direct_rank_read full-checkpoint background "
                    "read-ahead; weights will use the existing mmap-backed "
                    "path.")
            start_attempted = True
            start_error = None
            try:
                session.start()
            except Exception as error:
                start_error = error
            self._raise_on_rank_error("direct_rank_read read-ahead start",
                                      start_error)

            mmap_started = time.perf_counter()
            weights = None
            mmap_error = None
            try:
                weights = self._load_weights_in_parallel(
                    weight_files,
                    self._load_safetensors_file,
                    "Mapping safetensors weights in parallel",
                    reject_duplicate_keys=True)
            except Exception as error:
                mmap_error = error
            self._raise_on_rank_error("direct_rank_read SafeTensors mmap setup",
                                      mmap_error)
            assert weights is not None
            if local_rank == 0:
                logger.info(
                    "direct_rank_read SafeTensors mmap setup completed in "
                    f"{time.perf_counter() - mmap_started:.2f}s; background "
                    "read-ahead remains active during model materialization.")
            return weights, session
        except BaseException as setup_error:
            # A mmap/planning failure remains the primary exception even if
            # joining or communicator cleanup also encounters an error.
            if session is not None and start_attempted:
                try:
                    session.finish(setup_error)
                except Exception:
                    logger.exception(
                        "Suppressing direct_rank_read cleanup failure while "
                        "propagating weight-session setup failure.")
            elif node_communicator is not None:
                node_communicator.Free()
            raise

    def _load_cooperative_policy_with_communicator(
            self, weight_files: List[str], policy: WeightLoadPolicy,
            node_communicator) -> ConsumableWeightsDict:
        """Run cooperative stages using one active-communicator node group."""
        prefetch_size, enable_prefetch = self._get_cooperative_prefetch_policy(
            weight_files, node_communicator)
        local_rank, _ = self._get_local_rank_and_size(node_communicator)
        prefetch_started = time.perf_counter()
        prefetch_error = None
        try:
            if enable_prefetch:
                logger.info(
                    f"{policy.value} is prefetching "
                    f"{prefetch_size / (1024**3):.2f}GB of checkpoint data "
                    "in bounded chunks.")
                self.prefetch_file_chunks(weight_files, policy,
                                          node_communicator)
            else:
                logger.info(
                    f"Skipping {policy.value} full-checkpoint prefetch; "
                    "weights will be loaded through the existing mmap-backed "
                    "path.")
        except Exception as error:
            # Coordinate ordinary failures before entering a node-local
            # barrier. Otherwise peers can wait forever after one rank exits.
            prefetch_error = error
        self._raise_on_rank_error(f"{policy.value} checkpoint prefetch",
                                  prefetch_error)

        # Available memory is reduced within each node. Different nodes may
        # make different prefetch decisions, but every rank follows the same
        # collective sequence and the barrier remains node-local.
        if node_communicator is not None:
            node_communicator.Barrier()
        if enable_prefetch and local_rank == 0:
            prefetch_elapsed = time.perf_counter() - prefetch_started
            prefetch_throughput = prefetch_size / prefetch_elapsed / (1024**3)
            logger.info(
                f"{policy.value} checkpoint prefetch completed in "
                f"{prefetch_elapsed:.2f}s ({prefetch_throughput:.2f}GB/s "
                "logical read rate).")

        mmap_started = time.perf_counter()
        weights = None
        mmap_error = None
        try:
            weights = self._load_weights_in_parallel(
                weight_files,
                self._load_safetensors_file,
                "Mapping safetensors weights in parallel",
                reject_duplicate_keys=True)
        except Exception as error:
            # Complete world-rank consensus before any healthy rank returns
            # weights and begins mutating model parameters.
            mmap_error = error
        self._raise_on_rank_error(f"{policy.value} SafeTensors mmap setup",
                                  mmap_error)
        assert weights is not None
        if local_rank == 0:
            mmap_elapsed = time.perf_counter() - mmap_started
            logger.info(f"{policy.value} SafeTensors mmap setup completed in "
                        f"{mmap_elapsed:.2f}s.")
        return weights

    @staticmethod
    def _raise_on_rank_error(phase: str, error: BaseException | None) -> None:
        error_message = (None if error is None else
                         f"{type(error).__name__}: {error}")
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            error_messages = mpi_comm().allgather(error_message)
        else:
            error_messages = [error_message]
        for rank, rank_error in enumerate(error_messages):
            if rank_error is not None:
                raise RuntimeError(f"Rank {rank} failed during {phase}: "
                                   f"{rank_error}") from error

    @staticmethod
    def _get_active_node_communicator():
        """Derive a node-local group from the active model-load communicator."""
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            return mpi_comm().Split_type(_MPI.COMM_TYPE_SHARED)
        return None

    @staticmethod
    def _node_barrier(node_communicator=None) -> None:
        if node_communicator is not None:
            node_communicator.Barrier()
        else:
            local_mpi_barrier()

    def _get_cooperative_prefetch_policy(
            self,
            weight_files: List[str],
            node_communicator=None) -> tuple[int, bool]:
        """Compute a policy without entering a collective after local errors."""
        prefetch_size = None
        num_layers = None
        available_host_memory = None
        checkpoint_signature = None
        backing_file_signature = None
        policy_error = None
        try:
            file_stats = [(file, os.stat(file)) for file in weight_files]
            file_sizes = [(os.path.basename(file), stat.st_size)
                          for file, stat in file_stats]
            checkpoint_signature = tuple(file_sizes)
            backing_file_signature = tuple(
                (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                for _, stat in file_stats)
            prefetch_size = sum(size for _, size in file_sizes)
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            available_host_memory = psutil.virtual_memory().available
        except Exception as error:
            policy_error = error

        # This world collective occurs before the node-local memory allreduce,
        # so a stat/env/psutil error cannot strand healthy local peers there.
        self._raise_on_rank_error("cooperative prefetch policy", policy_error)
        assert checkpoint_signature is not None
        assert backing_file_signature is not None
        assert prefetch_size is not None
        assert num_layers is not None
        assert available_host_memory is not None

        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            policy_inputs = mpi_comm().allgather(
                (checkpoint_signature, num_layers))
            if any(inputs != policy_inputs[0] for inputs in policy_inputs[1:]):
                raise RuntimeError(
                    "Cooperative SafeTensors checkpoint selection and "
                    "TLLM_OVERRIDE_LAYER_NUM must match across MPI ranks")
        if node_communicator is not None:
            backing_file_signatures = node_communicator.allgather(
                backing_file_signature)
            backing_file_error = None
            if any(signature != backing_file_signatures[0]
                   for signature in backing_file_signatures[1:]):
                backing_file_error = RuntimeError(
                    "node-local ranks resolved checkpoint paths to different "
                    "backing files")
            self._raise_on_rank_error("cooperative backing-file validation",
                                      backing_file_error)
            available_host_memory = node_communicator.allreduce(
                available_host_memory, op=_MPI.MIN)

        enable_prefetch = (prefetch_size < available_host_memory * 0.9
                           and num_layers == 0)
        return prefetch_size, enable_prefetch

    def _get_prefetch_policy(self,
                             weight_files: List[str],
                             node_communicator=None) -> tuple[int, bool]:
        """Return checkpoint size and the node-consistent prefetch decision."""
        # Prefetch only when the files use less than 90% of available host
        # memory. This avoids page-cache thrashing for oversized checkpoints.
        prefetch_size = sum(os.path.getsize(file) for file in weight_files)
        # A layer override means only a model subset is loaded, so staging the
        # complete checkpoint would be wasted I/O.
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        enable_prefetch = (
            prefetch_size
            < self._get_local_available_host_memory(node_communicator) * 0.9
            and num_layers == 0)
        return prefetch_size, enable_prefetch

    def _load_weights_in_parallel(
            self,
            weight_files: List[str],
            load_func,
            description: str,
            reject_duplicate_keys: bool = False) -> ConsumableWeightsDict:
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
        def update_weights(part_weights: dict[str, Any]) -> None:
            if reject_duplicate_keys:
                duplicate_keys = weights.keys() & part_weights.keys()
                if duplicate_keys:
                    duplicate_key = min(duplicate_keys)
                    raise RuntimeError(
                        "Duplicate SafeTensors key found across checkpoint "
                        f"shards: {duplicate_key}")
            weights.update(part_weights)

        run_concurrently(load_func, [(w, ) for w in weight_files],
                         reduce_func=update_weights,
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

    @staticmethod
    def _prefetch_one_chunk(
            file_name: str,
            offset: int,
            length: int,
            cancel_event: threading.Event | None = None) -> None:
        """Read a bounded file extent into the OS page cache."""
        with open(file_name, "rb", buffering=0) as f:
            file_descriptor = f.fileno()
            read_offset = offset
            remaining = length
            while remaining > 0:
                if cancel_event is not None and cancel_event.is_set():
                    return
                read_size = min(remaining, _PREFETCH_READ_SIZE)
                data = os.pread(file_descriptor, read_size, read_offset)
                if not data:
                    raise OSError(
                        f"Unexpected EOF while prefetching {file_name} at "
                        f"offset {read_offset}")
                bytes_read = len(data)
                read_offset += bytes_read
                remaining -= bytes_read

    def _local_prefetch_chunks(
            self,
            file_names: List[str],
            policy: WeightLoadPolicy,
            node_communicator=None) -> list[tuple[str, int, int]]:
        """Build this rank's deterministic policy-assigned file extents."""
        chunks = []
        for file_name in sorted(file_names):
            file_size = os.path.getsize(file_name)
            for offset in range(0, file_size, self._prefetch_chunk_size):
                length = min(self._prefetch_chunk_size, file_size - offset)
                chunks.append((file_name, offset, length))
        local_rank, local_size = self._get_local_rank_and_size(
            node_communicator)
        if policy == WeightLoadPolicy.DIRECT_RANK_READ:
            return chunks[local_rank::local_size]
        if policy == WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH:
            return chunks if local_rank == 0 else []
        raise ValueError(f"Policy {policy.value} does not assign host reads")

    @staticmethod
    def _get_local_rank_and_size(node_communicator=None) -> tuple[int, int]:
        if node_communicator is not None:
            return (node_communicator.Get_rank(), node_communicator.Get_size())
        return 0, 1

    def prefetch_file_chunks(self,
                             file_names: List[str],
                             policy: WeightLoadPolicy,
                             node_communicator=None) -> None:
        """Prefetch policy-assigned chunks with a bounded CPU worker pool."""
        local_chunks = self._local_prefetch_chunks(file_names, policy,
                                                   node_communicator)
        if not local_chunks:
            return

        local_rank, local_size = self._get_local_rank_and_size(
            node_communicator)
        max_workers = self._get_prefetch_worker_count(policy, local_size,
                                                      len(local_chunks))
        logger.debug(f"{policy.value} local rank {local_rank} prefetching "
                     f"{len(local_chunks)} chunks with {max_workers} workers.")
        self._prefetch_chunks(local_chunks, max_workers)

    def _get_prefetch_worker_count(self, policy: WeightLoadPolicy,
                                   local_size: int, num_chunks: int) -> int:
        """Choose a bounded worker count before background work starts."""
        if num_chunks == 0:
            return 0
        if self._prefetch_workers_per_rank is None:
            worker_budget = _DEFAULT_PREFETCH_WORKERS_PER_RANK
            if policy == WeightLoadPolicy.DIRECT_RANK_READ:
                worker_budget = min(
                    worker_budget,
                    max(1, _DEFAULT_PREFETCH_WORKERS_PER_NODE // local_size))
            return min(worker_budget, num_chunks)
        else:
            return min(self._prefetch_workers_per_rank, num_chunks)

    def _prefetch_chunks(self,
                         local_chunks: list[tuple[str, int, int]],
                         max_workers: int,
                         cancel_event: threading.Event | None = None) -> None:
        """Execute a precomputed read plan without MPI or rank discovery."""
        if not local_chunks:
            return

        def prefetch_chunk(chunk: tuple[str, int, int]) -> None:
            try:
                if cancel_event is None:
                    self._prefetch_one_chunk(*chunk)
                else:
                    self._prefetch_one_chunk(*chunk, cancel_event)
            except Exception:
                if cancel_event is not None:
                    cancel_event.set()
                raise

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(prefetch_chunk, chunk) for chunk in local_chunks
            ]
            try:
                for index, future in enumerate(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        for pending_future in futures[index:]:
                            pending_future.cancel()
                    if not future.cancelled():
                        future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise

    def prefetch_files(self, file_names: List[str], node_communicator=None):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices local_rank, local_rank + local_mpi_size, local_rank + 2*local_mpi_size, etc.
        if node_communicator is not None:
            local_rank, local_size = self._get_local_rank_and_size(
                node_communicator)
        else:
            local_rank, local_size = local_mpi_rank(), local_mpi_size()
        local_file_names = file_names[local_rank::local_size]
        if len(local_file_names) == 0:
            return

        max_workers = min(multiprocessing.cpu_count() * 2, 16,
                          len(local_file_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._prefetch_one_file, local_file_names))
