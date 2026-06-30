# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import pickle  # nosec B403
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache, wraps
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (_object_to_tensor,
                                                _tensor_to_object)

try:
    from mpi4py import MPI
except Exception:
    MPI = None  # deferred; functions will error if used when ENABLE_MULTI_DEVICE is True

from tensorrt_llm._mnnvl_utils import (get_helix_cp_mnnvl_topology,
                                       init_helix_cp_comm)
from tensorrt_llm._torch.distributed.nccl_fault_tolerance import (
    NCCL_FAULT_TOLERANCE_ENABLED, _canonical_recovery_generation,
    _recovery_rendezvous_id)
from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_comm,
                                 mpi_disabled, mpi_isend, mpi_isend_object,
                                 mpi_recv, mpi_recv_object, mpi_send,
                                 mpi_send_object, mpi_world_size,
                                 torch_pybind11_abi)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.internal.process_group import init_pg
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm import ray_stub as ray


class ReduceOp(IntEnum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6


_reduce_op_to_torch_dict = {
    ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
    ReduceOp.PRODUCT: torch.distributed.ReduceOp.PRODUCT,
    ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
    ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
    ReduceOp.BAND: torch.distributed.ReduceOp.BAND,
    ReduceOp.BOR: torch.distributed.ReduceOp.BOR,
    ReduceOp.BXOR: torch.distributed.ReduceOp.BXOR,
}


def reduce_op_to_torch(op: ReduceOp) -> torch.distributed.ReduceOp:
    return _reduce_op_to_torch_dict[op]


_reduce_op_to_mpi_dict = {
    ReduceOp.SUM: MPI.SUM,
    ReduceOp.PRODUCT: MPI.PROD,
    ReduceOp.MIN: MPI.MIN,
    ReduceOp.MAX: MPI.MAX,
    ReduceOp.BAND: MPI.BAND,
    ReduceOp.BOR: MPI.BOR,
    ReduceOp.BXOR: MPI.BXOR,
}


def reduce_op_to_mpi(op: ReduceOp) -> MPI.Op:
    return _reduce_op_to_mpi_dict[op]


@dataclass(frozen=True)
class MpiFtSubcommSetup:
    """Collectively validated WideEP FT communicator startup state."""

    comm: Any
    local_rank: int
    ep_size: int
    ulfm_available: bool


_MPI_FT_PROCESS_LIFETIME_REFS: list[Any] = []
_MPI_FT_PROCESS_LIFETIME_REFS_LOCK = threading.Lock()


def _retain_mpi_ft_comm(comm: Any) -> None:
    """Keep a communicator with unsafe cleanup reachable until process exit."""
    with _MPI_FT_PROCESS_LIFETIME_REFS_LOCK:
        _MPI_FT_PROCESS_LIFETIME_REFS.append(comm)


def _is_mpi_ft_int(value: Any) -> bool:
    """Reject bool/float values that compare equal to MPI integer fields."""
    return isinstance(value, int) and not isinstance(value, bool)


def _mpi_ft_local_topology(mapping: Mapping, parent_rank: int, parent_size: int,
                           provided_thread_level: int,
                           health_size: Optional[int]) -> dict[str, Any]:
    """Build a serializable startup record without raising locally."""
    world_size = None
    mapping_rank = None
    ep_group = None
    ep_size = None
    ep_rank = None
    mapping_error = None
    try:
        world_size = mapping.world_size
        mapping_rank = mapping.rank
        ep_group = tuple(mapping.moe_ep_group)
        ep_size = mapping.moe_ep_size
        ep_rank = mapping.moe_ep_rank
    except Exception as error:
        mapping_error = ("invalid WideEP FT mapping topology: "
                         f"{type(error).__name__}: {error}")

    error_kind = None
    error_message = None
    if provided_thread_level < MPI.THREAD_MULTIPLE:
        error_kind = "runtime"
        error_message = (
            "WideEP FT requires MPI.THREAD_MULTIPLE because its control-plane "
            "thread overlaps other MPI traffic "
            f"(provided={provided_thread_level}, required={MPI.THREAD_MULTIPLE})"
        )
    elif mapping_error is not None:
        error_kind = "value"
        error_message = mapping_error
    elif not ep_group:
        error_kind = "value"
        error_message = "mapping.moe_ep_group must not be empty"
    elif len(ep_group) != ep_size:
        error_kind = "value"
        error_message = (
            "mapping.moe_ep_group size must match mapping.moe_ep_size, "
            f"got {len(ep_group)} and {ep_size}")
    else:
        try:
            has_duplicate_ranks = len(set(ep_group)) != len(ep_group)
            has_invalid_rank = any(
                not isinstance(rank, int) or rank < 0 or rank >= parent_size
                for rank in ep_group)
        except TypeError as error:
            error_kind = "value"
            error_message = f"mapping.moe_ep_group contains invalid ranks: {error}"
        else:
            if has_duplicate_ranks:
                error_kind = "value"
                error_message = (
                    "mapping.moe_ep_group contains duplicate ranks: "
                    f"{ep_group}")
            elif parent_size != world_size:
                error_kind = "runtime"
                error_message = (
                    "WideEP FT parent communicator size must match "
                    f"mapping.world_size, got {parent_size} and {world_size}")
            elif parent_rank != mapping_rank:
                error_kind = "runtime"
                error_message = (
                    "WideEP FT parent communicator rank must match mapping.rank, "
                    f"got {parent_rank} and {mapping_rank}")
            elif has_invalid_rank:
                error_kind = "value"
                error_message = (
                    "mapping.moe_ep_group contains a rank outside the parent "
                    f"communicator: {ep_group}")
            elif ep_size != world_size or set(ep_group) != set(
                    range(parent_size)):
                error_kind = "value"
                error_message = (
                    "WideEP FT MVP requires one MoE EP group spanning the "
                    "full MPI world; "
                    f"got world_size={world_size}, moe_ep_group={ep_group}")
            elif health_size is not None and health_size != ep_size:
                error_kind = "value"
                error_message = (
                    "EPGroupHealth size must match mapping.moe_ep_size, "
                    f"got {health_size} and {ep_size}")
            elif parent_rank not in ep_group:
                error_kind = "value"
                error_message = (
                    f"local rank {parent_rank} is not in its MoE EP group "
                    f"{ep_group}")
            elif ep_group.index(parent_rank) != ep_rank:
                error_kind = "value"
                error_message = (
                    "mapping.moe_ep_group order must match mapping.moe_ep_rank, "
                    f"got group index {ep_group.index(parent_rank)} and EP rank "
                    f"{ep_rank}")

    return {
        "parent_rank": parent_rank,
        "parent_size": parent_size,
        "world_size": world_size,
        "mapping_rank": mapping_rank,
        "ep_group": ep_group,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "health_size": health_size,
        "error_kind": error_kind,
        "error_message": error_message,
    }


def _validate_mpi_ft_topologies(topologies: List[Any],
                                parent_size: int) -> None:
    """Raise the same validation error on every parent-communicator rank."""
    if len(topologies) != parent_size:
        raise RuntimeError(
            "WideEP FT startup allgather returned an unexpected number of "
            f"topologies: got {len(topologies)}, expected {parent_size}")

    for gather_rank, topology in enumerate(topologies):
        if not isinstance(topology, dict):
            raise RuntimeError(
                "WideEP FT startup allgather returned an invalid topology for "
                f"parent rank {gather_rank}: {topology!r}")

    # Object allgather returns records in communicator-rank order. Selecting the
    # first error makes every rank raise the same exception before Split.
    for gather_rank, topology in enumerate(topologies):
        error_message = topology.get("error_message")
        if error_message is None:
            continue
        reporting_rank = topology.get("parent_rank", gather_rank)
        message = ("WideEP FT startup validation failed on parent rank "
                   f"{reporting_rank}: {error_message}")
        if topology.get("error_kind") == "value":
            raise ValueError(message)
        raise RuntimeError(message)

    for gather_rank, topology in enumerate(topologies):
        reported_parent_rank = topology.get("parent_rank")
        if (not _is_mpi_ft_int(reported_parent_rank)
                or reported_parent_rank != gather_rank):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: allgather slot "
                f"{gather_rank} reports parent rank "
                f"{reported_parent_rank!r}")
        reported_parent_size = topology.get("parent_size")
        if (not _is_mpi_ft_int(reported_parent_size)
                or reported_parent_size != parent_size):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports communicator size "
                f"{reported_parent_size!r}, expected {parent_size}")

        ep_group = topology.get("ep_group")
        if not isinstance(ep_group, (list, tuple)):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports invalid EP group {ep_group!r}")
        if len(ep_group) != parent_size:
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports EP group size {len(ep_group)}, "
                f"expected {parent_size}")
        if any(
                isinstance(peer_rank, bool) or not _is_mpi_ft_int(peer_rank)
                or peer_rank < 0 or peer_rank >= parent_size
                for peer_rank in ep_group):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports invalid EP group ranks {ep_group!r}")
        if len(set(ep_group)) != parent_size:
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports duplicate EP group ranks {ep_group!r}")
        world_size = topology.get("world_size")
        if not _is_mpi_ft_int(world_size) or world_size != parent_size:
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports world size "
                f"{world_size!r}, expected {parent_size}")
        mapping_rank = topology.get("mapping_rank")
        if not _is_mpi_ft_int(mapping_rank) or mapping_rank != gather_rank:
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports mapping rank "
                f"{mapping_rank!r}")
        ep_size = topology.get("ep_size")
        if not _is_mpi_ft_int(ep_size) or ep_size != parent_size:
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports EP size {ep_size!r}, "
                f"expected {parent_size}")
        ep_rank = topology.get("ep_rank")
        if (not _is_mpi_ft_int(ep_rank) or ep_rank < 0 or ep_rank >= parent_size
                or ep_group[ep_rank] != gather_rank):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports invalid EP rank {ep_rank!r} for "
                f"group {ep_group!r}")
        health_size = topology.get("health_size")
        if health_size is not None and (not _is_mpi_ft_int(health_size)
                                        or health_size != parent_size):
            raise RuntimeError(
                "WideEP FT startup topology is inconsistent: parent rank "
                f"{gather_rank} reports health size {health_size}, "
                f"expected {parent_size}")
    # Only dereference peer groups after every gathered record has passed the
    # schema/range checks above. This keeps malformed records rank-attributed and
    # prevents a valid earlier record from indexing into unchecked peer data.
    for gather_rank, topology in enumerate(topologies):
        ep_group = topology["ep_group"]
        for peer_rank in ep_group:
            peer_group = topologies[peer_rank].get("ep_group")
            if peer_group != ep_group:
                raise RuntimeError(
                    "WideEP FT startup topology is inconsistent: parent rank "
                    f"{gather_rank} reports EP group {ep_group}, but member "
                    f"rank {peer_rank} reports {peer_group}")


def _mpi_ft_post_split_status(ft_comm: Any, parent_rank: int, ep_rank: int,
                              ep_size: int) -> dict[str, Any]:
    """Configure a derived communicator without raising before reconciliation."""
    error_message = None
    ulfm_available = False
    try:
        # Install the non-fatal handler before any other operation on the new
        # communicator. In particular, do not inspect a possibly broken handle
        # while it still inherits the parent's fatal error policy.
        ft_comm.Set_errhandler(MPI.ERRORS_RETURN)
        actual_rank = ft_comm.Get_rank()
        actual_size = ft_comm.Get_size()
        if actual_size != ep_size or actual_rank != ep_rank:
            error_message = (
                "MPI_Comm_split created an unexpected WideEP FT communicator: "
                f"rank={actual_rank}, size={actual_size}, "
                f"expected rank={ep_rank}, size={ep_size}")
    except Exception as error:
        error_message = ("WideEP FT communicator setup raised "
                         f"{type(error).__name__}: {error}")

    if error_message is None and callable(getattr(
            ft_comm, "Is_revoked", None)) and callable(
                getattr(ft_comm, "Revoke", None)):
        try:
            already_revoked = ft_comm.Is_revoked()
        except Exception as error:
            if not _is_unsupported_mpi_ulfm_error(error):
                error_message = ("WideEP FT ULFM probe raised "
                                 f"{type(error).__name__}: {error}")
        else:
            if already_revoked:
                error_message = (
                    "WideEP FT communicator is already revoked at construction")
            else:
                ulfm_available = True
    return {
        "parent_rank": parent_rank,
        "error_message": error_message,
        "ulfm_available": ulfm_available,
    }


def _is_unsupported_mpi_ulfm_error(error: BaseException) -> bool:
    if isinstance(error, NotImplementedError):
        return True
    mpi_exception = getattr(MPI, "Exception", None)
    if not isinstance(mpi_exception, type) or not isinstance(
            error, mpi_exception):
        return False
    get_error_class = getattr(error, "Get_error_class", None)
    if not callable(get_error_class):
        return False
    unsupported_classes = {
        value
        for value in (
            getattr(MPI, "ERR_NOT_SUPPORTED", None),
            getattr(MPI, "ERR_UNSUPPORTED_OPERATION", None),
        ) if value is not None
    }
    try:
        error_class = get_error_class()
    except Exception:
        # This helper runs while building a rank-local status that must reach
        # the parent allgather. A broken exception classifier is not evidence of
        # an unsupported operation, but it must not strand peer ranks either.
        return False
    return error_class in unsupported_classes


def _mpi_ft_post_split_result(
    statuses: List[Any],
    parent_size: int,
) -> tuple[Optional[str], bool]:
    """Reconcile setup errors and the global ULFM capability."""
    if len(statuses) != parent_size:
        return (
            "WideEP FT post-split allgather returned an unexpected number of "
            f"statuses: got {len(statuses)}, expected {parent_size}",
            False,
        )
    for gather_rank, status in enumerate(statuses):
        if not isinstance(status, dict):
            return (
                "WideEP FT post-split allgather returned an invalid status for "
                f"parent rank {gather_rank}: {status!r}",
                False,
            )
        if status.get("parent_rank") != gather_rank:
            return (
                "WideEP FT post-split status is inconsistent: allgather slot "
                f"{gather_rank} reports parent rank "
                f"{status.get('parent_rank')}",
                False,
            )
        if not isinstance(status.get("ulfm_available"), bool):
            return (
                "WideEP FT post-split allgather returned incomplete capability "
                f"state for parent rank {gather_rank}: {status!r}",
                False,
            )

    ulfm_available = all(status["ulfm_available"] for status in statuses)
    for gather_rank, status in enumerate(statuses):
        error_message = status.get("error_message")
        if error_message is not None:
            return (
                "WideEP FT communicator setup failed on parent rank "
                f"{gather_rank}: {error_message}",
                False,
            )
    return None, ulfm_available


def create_mpi_ft_subcomm(
        mapping: Mapping,
        parent_comm: Optional[Any] = None,
        health_size: Optional[int] = None) -> MpiFtSubcommSetup:
    """Create the dedicated MPI control-plane communicator for WideEP FT.

    This operation is collective across ``parent_comm`` and must run on every
    rank during startup, before any background failure-broadcast threads are
    launched. The MVP requires one MoE EP group spanning the parent world,
    ordered by the EP-local rank used by :class:`EPGroupHealth`.

    Before splitting, ranks exchange their local validation outcome and EP
    topology on the healthy parent communicator. This prevents one rank from
    raising locally while its peers block forever inside ``MPI_Comm_split``.

    Args:
        mapping: Distributed topology for the local rank.
        parent_comm: Parent MPI communicator. Defaults to TRT-LLM's active
            ``mpi_comm()``, which wraps ``MPI.COMM_WORLD`` in the standard
            launch path and preserves custom communicator sessions.
        health_size: Optional local ``EPGroupHealth`` size to validate
            collectively before splitting.

    Returns:
        The world-spanning EP MPI communicator and its collectively validated
        rank, size, and ULFM capability.

    Raises:
        RuntimeError: If MPI is unavailable, does not provide
            ``MPI.THREAD_MULTIPLE``, or creates an unexpected communicator.
        ValueError: If the mapping's EP group is inconsistent.
    """
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required to create the WideEP FT communicator")

    parent_comm = mpi_comm() if parent_comm is None else parent_comm
    parent_rank = parent_comm.Get_rank()
    parent_size = parent_comm.Get_size()
    local_topology = _mpi_ft_local_topology(mapping, parent_rank, parent_size,
                                            MPI.Query_thread(), health_size)
    topologies = parent_comm.allgather(local_topology)
    _validate_mpi_ft_topologies(topologies, parent_size)

    ep_group = local_topology["ep_group"]
    ep_rank = local_topology["ep_rank"]
    ep_size = local_topology["ep_size"]

    # The MVP topology gate above makes the world-spanning group's first rank
    # a stable color shared by every process.
    ft_comm = parent_comm.Split(color=ep_group[0], key=ep_rank)
    try:
        setup_status = _mpi_ft_post_split_status(ft_comm, parent_rank, ep_rank,
                                                 ep_size)
        setup_statuses = parent_comm.allgather(setup_status)
        setup_error, ulfm_available = _mpi_ft_post_split_result(
            setup_statuses, parent_size)
        if setup_error is None:
            setup = MpiFtSubcommSetup(
                comm=ft_comm,
                local_rank=ep_rank,
                ep_size=ep_size,
                ulfm_available=ulfm_available,
            )
            # Enforce process-lifetime ownership at the creation boundary. A
            # direct or future caller must not be able to drop the last Python
            # reference and trigger rank-local collective MPI_Comm_free.
            _retain_mpi_ft_comm(ft_comm)
            return setup
    except Exception:
        # Once Split succeeds, never let an unexpected post-Split exception
        # drop the last reference and trigger rank-local communicator cleanup.
        _retain_mpi_ft_comm(ft_comm)
        raise

    # MPI_Comm_free is collective. Once setup has reported any communicator
    # invariant failure, attempting collective cleanup on that same handle is
    # not provably bounded, even with MPI_ERRORS_RETURN installed. Keep the
    # handle reachable and let MPI reclaim it at process teardown.
    _retain_mpi_ft_comm(ft_comm)
    logger.warning("WideEP FT retained a communicator after startup failure: "
                   f"{setup_error}")
    raise RuntimeError(setup_error)


class Distributed(ABC):

    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    @staticmethod
    @lru_cache(maxsize=None)
    def get(mapping: Mapping) -> "Distributed":
        if mpi_disabled():
            return TorchDist(mapping)
        else:
            return MPIDist(mapping)

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size

    @property
    def has_tp(self):
        return self.mapping.has_tp()

    @property
    def has_pp(self):
        return self.mapping.has_pp()

    @property
    def cp_size(self):
        return self.mapping.cp_size

    @property
    def pp_size(self):
        return self.mapping.pp_size

    @property
    def tp_size(self):
        return self.mapping.tp_size

    @property
    def cp_rank(self):
        return self.mapping.cp_rank

    @property
    def tp_rank(self):
        return self.mapping.tp_rank

    @property
    def pp_rank(self):
        return self.mapping.pp_rank

    @property
    def is_last_pp_rank(self):
        return self.mapping.is_last_pp_rank()

    @property
    def is_second_last_pp_rank(self):
        return self.mapping.is_second_last_pp_rank()

    @property
    def is_first_pp_rank(self):
        return self.mapping.is_first_pp_rank()

    @property
    def next_pp_rank(self):
        return self.mapping.next_pp_rank()

    @property
    def prev_pp_rank(self):
        return self.mapping.prev_pp_rank()

    @property
    def has_cp_ulysses(self):
        return self.mapping.has_cp_ulysses()

    @property
    def has_cp_helix(self):
        return self.mapping.has_cp_helix()

    @property
    def cp_config(self):
        return self.mapping.cp_config

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def tp_barrier(self):
        pass

    @abstractmethod
    def broadcast(self, obj, root=0):
        pass

    @abstractmethod
    def allgather(self, obj, root=0):
        pass

    @abstractmethod
    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        pass

    @abstractmethod
    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        pass

    @abstractmethod
    def tp_broadcast(self, obj, root=0, **kwargs):
        pass

    @abstractmethod
    def cp_broadcast(self, obj, root=0, **kwargs):
        pass

    def tp_cp_broadcast(self, obj, root=0, **kwargs):
        """Broadcast object across both TP and CP groups.

        This is used when both TP and CP parallelism are enabled (e.g., helix parallelism).
        First broadcasts within the TP group, then within the CP group.
        """
        if self.tp_size > 1:
            obj = self.tp_broadcast(obj, root=root, **kwargs)
        if self.cp_size > 1:
            obj = self.cp_broadcast(obj, root=root, **kwargs)
        return obj

    @abstractmethod
    def tp_allgather(self, obj):
        pass

    @abstractmethod
    def cp_allgather(self, obj):
        pass

    def tp_cp_allgather(self, obj):
        """Allgather across both TP and CP dimensions.

        First gathers within CP group, then across TP groups, returning
        a flattened list with tp_size * cp_size entries.
        """
        # Gather across CP dimension.
        if self.cp_size > 1:
            obj = self.cp_allgather(obj)
        else:
            obj = [obj]  # Wrap to match cp_allgather output format.

        # Gather across TP dimension.
        if self.tp_size > 1:
            obj = self.tp_allgather(obj)
        else:
            obj = [obj]  # Wrap to match tp_allgather output format.

        # Flatten: [[cp0, cp1], [cp0, cp1], ...] -> [tp0_cp0, tp0_cp1, tp1_cp0, ...]
        return [entry for tp_group in obj for entry in tp_group]


def safe_broadcast(comm, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
    """
    Safely broadcasts potentially large objects by splitting into fixed-size chunks,
    using raw-byte MPI.Bcast to avoid pickle5's out-of-band buffer allocations.

    Args:
        comm: communicator to broadcast
        obj: Python object to broadcast
        root: Rank of the broadcasting process
        chunk_size: Maximum size of each chunk in bytes (default: 4MB)

    Returns:
        The broadcasted object on all ranks
    """
    if not ENABLE_MULTI_DEVICE:
        return obj
    if ENABLE_MULTI_DEVICE and MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    rank = comm.Get_rank()

    # ---- Serialization phase (root only) ----
    # Header layout: [ok_flag, total_size, num_chunks] as int64
    header = np.zeros(3, dtype=np.int64)
    if rank == root:
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            total_size = len(serialized)
            num_chunks = math.ceil(total_size /
                                   chunk_size) if total_size > 0 else 0
            header[:] = (1, total_size, num_chunks)
        except Exception as e:
            # Signal failure to all ranks, then raise
            header[:] = (0, 0, 0)
            comm.Bcast([header, MPI.INT64_T], root=root)
            raise RuntimeError(f"Serialization failed: {str(e)}") from e
    else:
        serialized = None  # not used on non-root before Bcast

    # ---- Metadata broadcast (Bcast the fixed-size header) ----
    comm.Bcast([header, MPI.INT64_T], root=root)
    ok_flag, total_size, num_chunks = int(header[0]), int(header[1]), int(
        header[2])
    if not ok_flag:
        raise RuntimeError("Root rank failed during serialization")

    # ---- Allocate receive buffer (non-root) or build a view (root) ----
    # We broadcast raw bytes chunk by chunk.
    if rank == root:
        src_view = memoryview(serialized)
        dst_buf = None
        dst_view = None
    else:
        # Pre-allocate a contiguous byte buffer to receive the payload
        dst_buf = bytearray(total_size)
        dst_view = memoryview(dst_buf)
        src_view = None  # not used on non-root

    # ---- Chunked raw-byte broadcast with MPI.Bcast ----
    # Each round sends exactly `cur` bytes of the global payload.
    offset = 0
    for i in range(num_chunks):
        cur = min(chunk_size, total_size - offset)
        if cur <= 0:
            break  # safety guard for zero-size payloads

        if rank == root:
            # Root sends a slice of the source view
            part = src_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)
        else:
            # Non-root receives directly into the destination view
            part = dst_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)

        offset += cur

    # ---- Reconstruction and deserialization ----
    # Validate the received byte count and unpickle.
    if rank == root:
        # Root already has `serialized`
        if len(serialized) != total_size:
            raise RuntimeError(
                f"Data size mismatch at root: expected {total_size}, got {len(serialized)}"
            )
        try:
            return pickle.loads(serialized)  # nosec B301
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e
    else:
        if len(dst_buf) != total_size:
            raise RuntimeError(
                f"Data size mismatch at rank {rank}: expected {total_size}, got {len(dst_buf)}"
            )
        try:
            return pickle.loads(dst_buf)  # nosec B301
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e


def _serialize_and_exchange_lengths(
    comm: Any,
    obj: Any,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Serialize *obj* and exchange payload lengths across all ranks.

    Uses buffer-based ``MPI_Allgather`` (uppercase) for the length
    exchange — a single MPI collective with no pickle overhead, which
    is the same work that mpi4py does internally inside
    ``comm.allgather(obj)``.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance).
        obj: Python object to transfer (must be picklable).

    Returns:
        Tuple of ``(rank, size, lengths, displs, sendbuf)`` where:

        - **rank** (*int*) — this process's rank in *comm*.
        - **size** (*int*) — total number of ranks in *comm*.
        - **lengths** (*np.ndarray[int64]*) — per-rank serialized payload
          sizes.  A value of ``-1`` signals a serialization failure.
        - **displs** (*np.ndarray[int64]*) — per-rank byte offsets into a
          concatenated receive buffer.
        - **sendbuf** (*np.ndarray[uint8]*) — this rank's serialized
          payload as a contiguous byte array (empty when serialization
          failed).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_ser_error = None
    try:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        local_len = np.array([len(payload)], dtype=np.int64)
    except Exception as exc:
        payload = b""
        local_len = np.array([-1], dtype=np.int64)
        local_ser_error = exc

    # Buffer-based Allgather: 1 MPI collective, no pickle overhead.
    lengths = np.empty(size, dtype=np.int64)
    comm.Allgather([local_len, MPI.INT64_T], [lengths, MPI.INT64_T])

    if (lengths < 0).any():
        raise RuntimeError(
            f"Rank {rank}: serialization failed on at least one rank "
            f"(lengths={lengths})") from local_ser_error

    displs = np.zeros(size, dtype=np.int64)
    if size > 1:
        displs[1:] = np.cumsum(lengths[:-1])

    sendbuf = np.frombuffer(payload, dtype=np.uint8)
    return rank, size, lengths, displs, sendbuf


def _chunked_transfer_loop(
    comm: Any,
    rank: int,
    size: int,
    lengths: np.ndarray,
    displs: np.ndarray,
    sendbuf: np.ndarray,
    num_rounds: int,
    chunk_size: int,
    recvbuf: Optional[np.ndarray],
    collective_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                            None],
) -> None:
    """Run the chunked MPI transfer loop used by safe_gather/safe_allgather.

    Each round transfers at most ``chunk_size`` bytes per rank using a
    per-round temporary receive buffer with 0-based int32 displacements,
    then copies the received data into ``recvbuf`` at the correct absolute
    offsets using 64-bit Python-level indexing.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance).
        rank: This rank's index in *comm*.
        size: Total number of ranks in *comm*.
        lengths: Per-rank serialized payload sizes (int64 array of shape
            ``(size,)``).
        displs: Per-rank byte offsets into *recvbuf* (int64 array of shape
            ``(size,)``).
        sendbuf: This rank's serialized payload (uint8 array).
        num_rounds: Number of chunked transfer rounds to execute.
        chunk_size: Per-round max bytes each rank contributes.
        recvbuf: Final contiguous receive buffer (uint8 array), or
            ``None`` for non-root ranks in gather mode (copy-back is
            skipped).
        collective_fn: Callable that performs the per-round MPI
            collective.  Signature: ``collective_fn(send_part,
            round_recvbuf, counts_this_round, round_displs)``.
    """
    for r in range(num_rounds):
        round_offs = r * chunk_size
        counts_this_round = np.minimum(np.maximum(lengths - round_offs, 0),
                                       chunk_size).astype(np.int32)
        sent_so_far = np.minimum(lengths, round_offs)

        round_recvbuf = (np.empty(counts_this_round.sum(), dtype=np.uint8)
                         if recvbuf is not None else None)
        round_displs = np.zeros(size, dtype=np.int32)
        if size > 1:
            round_displs[1:] = np.cumsum(counts_this_round[:-1])

        send_part = sendbuf[sent_so_far[rank]:sent_so_far[rank] +
                            counts_this_round[rank]]

        collective_fn(send_part, round_recvbuf, counts_this_round, round_displs)

        # Copy received chunks into the final buffer at correct
        # absolute offsets (using 64-bit Python-level indexing).
        if recvbuf is not None:
            src_offset = 0
            for i in range(size):
                n = counts_this_round[i]
                if n > 0:
                    dst = displs[i] + sent_so_far[i]
                    recvbuf[dst:dst +
                            n] = (round_recvbuf[src_offset:src_offset + n])
                src_offset += n


def _deserialize_recvbuf(
    recvbuf: np.ndarray,
    lengths: np.ndarray,
    displs: np.ndarray,
    size: int,
) -> List[Any]:
    """Deserialize gathered payloads from a contiguous receive buffer.

    Args:
        recvbuf: Contiguous receive buffer (uint8 array) containing the
            concatenated serialized payloads from all ranks.
        lengths: Per-rank serialized payload sizes (int64 array of shape
            ``(size,)``).
        displs: Per-rank byte offsets into *recvbuf* (int64 array of shape
            ``(size,)``).
        size: Total number of ranks.

    Returns:
        List of deserialized Python objects (``len == size``). Ranks whose
        payload length is zero are represented as ``None``.
    """
    # Zero-length payloads (e.g. from pickling None) are returned as None
    # without calling pickle.loads, which would fail on empty bytes.
    return [
        pickle.loads(recvbuf[displs[i]:displs[i] + lengths[i]])  # nosec B301
        if lengths[i] > 0 else None for i in range(size)
    ]


def safe_gather(
    comm: Any,
    obj: Any,
    root: int = 0,
    chunk_size: int = 4 * 1024 * 1024,
) -> Optional[List[Any]]:
    """Safely gather potentially large objects by splitting into fixed-size
    chunks, using raw-byte MPI.Gatherv with a per-round temp buffer to
    keep counts and displacements within int32.

    The function serializes *obj* once with ``pickle.dumps``, exchanges
    payload lengths via buffer-based ``MPI_Allgather`` (1 MPI collective),
    then transfers the raw bytes with ``MPI_Gatherv`` (1 MPI collective).
    This matches the number of MPI collectives that mpi4py's
    ``comm.gather(obj)`` performs internally, while adding chunking
    safety for payloads whose total exceeds the int32 displacement
    limit (~2 GB).

    Args:
        comm: MPI communicator (``MPI.Comm`` instance) to gather over.
        obj: Python object to gather (must be picklable).
        root: Rank that receives the gathered objects.
        chunk_size: Per-round max bytes each rank contributes (default:
            4 MB).

    Returns:
        On *root*: list of deserialized objects (``len == comm.size``).
        On non-root ranks: ``None``.
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Step 1: serialize once and exchange lengths (1 MPI collective).
    rank, size, lengths, displs, sendbuf = \
        _serialize_and_exchange_lengths(comm, obj)

    total = int(lengths.sum())
    int32_max = np.iinfo(np.int32).max

    # Step 2a: total fits in int32 — single Gatherv (1 MPI collective).
    if total < int32_max:
        counts = lengths.astype(np.int32)
        displs32 = displs.astype(np.int32)
        if rank == root:
            recvbuf = np.empty(total, dtype=np.uint8)
            comm.Gatherv([sendbuf, MPI.BYTE],
                         [recvbuf, counts, displs32, MPI.BYTE],
                         root=root)
            return _deserialize_recvbuf(recvbuf, lengths, displs, size)
        else:
            comm.Gatherv([sendbuf, MPI.BYTE], None, root=root)
            return None

    # Step 2b: total exceeds int32 — chunked Gatherv.
    logger.info(
        "safe_gather: total payload %d bytes exceeds int32 limit, "
        "using chunked Gatherv (size=%d)", total, size)
    max_safe_chunk = int32_max // size
    chunk_size = min(chunk_size, max_safe_chunk)
    max_len = int(lengths.max())
    num_rounds = math.ceil(max_len / chunk_size) if max_len > 0 else 0

    recvbuf = np.empty(total, dtype=np.uint8) if rank == root else None

    def _gatherv(send_part, round_recvbuf, counts, round_displs):
        if rank == root:
            comm.Gatherv([send_part, MPI.BYTE],
                         [round_recvbuf, counts, round_displs, MPI.BYTE],
                         root=root)
        else:
            comm.Gatherv([send_part, MPI.BYTE], None, root=root)

    _chunked_transfer_loop(comm, rank, size, lengths, displs, sendbuf,
                           num_rounds, chunk_size, recvbuf, _gatherv)

    if rank == root:
        return _deserialize_recvbuf(recvbuf, lengths, displs, size)
    return None


def safe_allgather(
    comm: Any,
    obj: Any,
    chunk_size: int = 4 * 1024 * 1024,
) -> List[Any]:
    """Safely allgather potentially large objects by splitting into
    fixed-size chunks, using raw-byte MPI.Allgatherv.

    The function serializes *obj* once with ``pickle.dumps``, exchanges
    payload lengths via buffer-based ``MPI_Allgather`` (1 MPI collective),
    then transfers the raw bytes with ``MPI_Allgatherv`` (1 MPI
    collective).  This matches the number of MPI collectives that
    mpi4py's ``comm.allgather(obj)`` performs internally, while adding
    chunking safety for payloads whose total exceeds the int32
    displacement limit (~2 GB) and avoiding mpi4py's pickle5
    out-of-band buffers that can cause unexpected memory spikes.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance) to allgather over.
        obj: Python object to allgather (must be picklable).
        chunk_size: Per-round max bytes each rank contributes (default:
            4 MB).

    Returns:
        List of deserialized objects from all ranks
        (``len == comm.size``).
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Step 1: serialize once and exchange lengths (1 MPI collective).
    rank, size, lengths, displs, sendbuf = \
        _serialize_and_exchange_lengths(comm, obj)

    total = int(lengths.sum())
    int32_max = np.iinfo(np.int32).max

    # Step 2a: total fits in int32 — single Allgatherv (1 MPI collective).
    if total < int32_max:
        counts = lengths.astype(np.int32)
        displs32 = displs.astype(np.int32)
        recvbuf = np.empty(total, dtype=np.uint8)
        comm.Allgatherv([sendbuf, MPI.BYTE],
                        [recvbuf, counts, displs32, MPI.BYTE])
        return _deserialize_recvbuf(recvbuf, lengths, displs, size)

    # Step 2b: total exceeds int32 — chunked Allgatherv.
    logger.info(
        "safe_allgather: total payload %d bytes exceeds int32 limit, "
        "using chunked Allgatherv (size=%d)", total, size)
    max_safe_chunk = int32_max // size
    chunk_size = min(chunk_size, max_safe_chunk)
    max_len = int(lengths.max())
    num_rounds = math.ceil(max_len / chunk_size) if max_len > 0 else 0

    recvbuf = np.empty(total, dtype=np.uint8)

    def _allgatherv(send_part, round_recvbuf, counts, round_displs):
        comm.Allgatherv([send_part, MPI.BYTE],
                        [round_recvbuf, counts, round_displs, MPI.BYTE])

    _chunked_transfer_loop(comm, rank, size, lengths, displs, sendbuf,
                           num_rounds, chunk_size, recvbuf, _allgatherv)

    return _deserialize_recvbuf(recvbuf, lengths, displs, size)


class MPIDist(Distributed):
    tp_comm: MPI.Comm

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self._cp_comm = None
        self._tp_comm = None
        self._pp_comm = None

    def _validate_world_size(self):
        """Validate world size before creating sub-communicators to prevent segfaults."""

        if ENABLE_MULTI_DEVICE:
            actual_world_size = mpi_world_size()
            max_rank_needed = self.mapping.world_size

            if max_rank_needed > actual_world_size:
                raise RuntimeError(
                    f"Mapping requires world_size={max_rank_needed} "
                    f"(tp_size={self.mapping.tp_size} * pp_size={self.mapping.pp_size} * cp_size={self.mapping.cp_size}), "
                    f"but MPI world size is only {actual_world_size}. ")

    def broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = mpi_comm()
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    def allgather(self, obj):
        return mpi_allgather(obj)

    def barrier(self):
        mpi_barrier()

    def tp_barrier(self):
        self.tp_comm.Barrier()

    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        return mpi_isend(buf, dest, tag)

    def send(self, buf: np.ndarray, dest, tag=0):
        # blocking send numpy buffer
        mpi_send(buf, dest, tag)

    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        return mpi_recv(buf, src, tag)

    def send_object(self, obj, dest, tag=0):
        mpi_send_object(obj, dest, tag)

    def isend_object(self, obj, dest, tag=0):
        return mpi_isend_object(obj, dest, tag)

    def recv_object(self, src, tag=0):
        return mpi_recv_object(src, tag)

    @property
    def tp_comm(self):
        if self._tp_comm is None:
            self._validate_world_size()
            mapping = self.mapping
            new_group = mpi_comm().group.Incl(mapping.tp_group)
            self._tp_comm = mpi_comm().Create_group(new_group)
        return self._tp_comm

    @property
    def pp_comm(self):
        if self._pp_comm is None:
            self._validate_world_size()
            mapping = self.mapping
            new_group = mpi_comm().group.Incl(mapping.pp_group)
            self._pp_comm = mpi_comm().Create_group(new_group)
        return self._pp_comm

    @property
    def cp_comm(self):
        if self._cp_comm is None:
            self._validate_world_size()
            new_group = mpi_comm().group.Incl(self.mapping.cp_group)
            self._cp_comm = mpi_comm().Create_group(new_group)
        return self._cp_comm

    def cp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024):
        comm = self.cp_comm
        return safe_allgather(comm, obj, chunk_size=chunk_size)

    def cp_broadcast(self,
                     obj,
                     root=0,
                     chunk_size: int = 4 * 1024 * 1024,
                     **kwargs):
        comm = self.cp_comm
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    def tp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_allgather(comm, obj, chunk_size=chunk_size)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_gather(comm, obj, root=root, chunk_size=chunk_size)

    def tp_broadcast(self,
                     obj,
                     root=0,
                     chunk_size: int = 4 * 1024 * 1024,
                     **kwargs):
        comm = self.tp_comm
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    def pp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024):
        comm = self.pp_comm
        return safe_allgather(comm, obj, chunk_size=chunk_size)

    def pp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.pp_comm
        return safe_gather(comm, obj, root=root, chunk_size=chunk_size)

    def pp_broadcast(self, obj, root=0):
        return self.pp_comm.bcast(obj, root)

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        reduce_op = reduce_op_to_mpi(op)
        return mpi_comm().allreduce(obj, reduce_op)

    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        reduce_op = reduce_op_to_mpi(op)
        return self.tp_comm.allreduce(obj, reduce_op)


class MultiHandleWrapper:
    """
    Wrapper that encapsulates multiple handles and provides a single wait() interface
    to unify the API between MPIDist and TorchDist.
    """

    def __init__(self, handles):
        self.handles = handles if isinstance(handles, list) else [handles]

    def wait(self):
        for handle in self.handles:
            try:
                handle.wait()
            except Exception as e:
                raise RuntimeError(f"Asynchronous operation failed: {e}") from e


class TorchDist(Distributed):

    @property
    def rank(self):
        return torch.distributed.get_rank()

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        assert dist.is_initialized(
        ), "torch.distributed should be initialized before TorchDist"

        self.cluster_info = None

        from tensorrt_llm._utils import set_torch_comm
        set_torch_comm(self)  # Set as global instance
        mapping.build_mesh()

        self.setup_local_comm()
        self.default_store = torch.distributed.distributed_c10d._get_default_store(
        )

        init_pg(torch.distributed.group.WORLD, self.local_comm,
                torch_pybind11_abi())

    def setup_local_comm(self):
        self._get_cluster_info()

        # node IP -> list of ranks
        ip_to_ranks = {}
        for rank, (node_ip, _) in enumerate(self.cluster_info):
            ip_to_ranks.setdefault(node_ip, []).append(int(rank))

        self.local_comm = None
        for ranks in ip_to_ranks.values():
            # All global ranks from the default process group to participate in the call,
            # even if some ranks are not part of the new process group being created
            pg = dist.new_group(ranks=ranks, backend='cuda:nccl,cpu:gloo')
            if int(self.rank) in ranks:
                logger.debug(
                    f"[Rank {self.rank}] Done setting local comm. ip_to_ranks: {ip_to_ranks}"
                )
                self.local_comm = pg

    def _get_cluster_info(self):
        if self.cluster_info is not None:
            return self.cluster_info

        if ray.is_initialized():
            node_ip = ray.util.get_node_ip_address()
        else:
            raise RuntimeError("Ray is not initialized")

        gpu_index = [int(id) for id in ray.get_gpu_ids()]

        assert len(gpu_index) == 1

        # Gather node ip
        node_list = [None] * torch.distributed.get_world_size()

        torch.distributed.all_gather_object(node_list, node_ip)

        # Gather gpu index
        gpu_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gpu_list, gpu_index[0])

        # Gather rank
        rank_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_list,
                                            torch.distributed.get_rank())

        rank_info_list = [None] * torch.distributed.get_world_size()
        for i in range(len(rank_list)):
            rank_info_list[rank_list[i]] = (node_list[i], gpu_list[i])

        self.cluster_info = rank_info_list

        logger.debug(f"Cluster info: {self.cluster_info}")
        return self.cluster_info

    @staticmethod
    def log_op(func, enable_log=False):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_log:
                logger.debug(
                    f"{func.__name__} enter: {args[1:]}, {kwargs}, rank: {torch.distributed.get_rank()}"
                )
            ret = func(*args, **kwargs)

            if enable_log:
                logger.debug(f"{func.__name__} exit: {ret}")
            return ret

        return wrapper

    @log_op
    def broadcast(self, obj, root=0):
        assert not (self.mapping.has_cp_ulysses() and self.mapping.has_tp()
                    ), 'Unsupported mix of Ulysses CP and TP.'

        if mpi_disabled():
            if isinstance(obj, torch.Tensor):
                dist.broadcast(obj, src=root)
                return obj
            else:
                obj_list = [obj]
                dist.broadcast_object_list(obj_list, src=root)
                return obj_list[0]

        if self.mapping.has_cp_ulysses():
            self.broadcast_cp(obj, root)
        elif self.mapping.has_tp():
            self.broadcast_tp(obj, root)

    @log_op
    def allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj) for _ in range(self.world_size)
            ]
            dist.all_gather(output_list, obj)
            return output_list
        else:
            obj_list = [None] * self.world_size
            dist.all_gather_object(obj_list, obj)
            return obj_list

    @log_op
    def barrier(self):
        dist.barrier()

    @log_op
    def tp_barrier(self):
        dist.barrier(group=self.mapping.tp_group_pg)

    @log_op
    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        tensor = torch.from_numpy(buf)
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def send(self, buf: np.ndarray, dest, tag=0):
        raise NotImplementedError(
            "blocking send is not implemented for TorchDist")

    @log_op
    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        tensor = torch.empty_like(torch.from_numpy(buf))
        dist.recv(tensor, src=src, tag=tag)
        return tensor.numpy()

    @log_op
    def isend_tensor(self, tensor: torch.Tensor, dest, tag=0):
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def recv_tensor(self, tensor: torch.Tensor, src, tag=0):
        dist.recv(tensor, src=src, tag=tag)
        return tensor

    @log_op
    def recv_object(self, src, tag=0):
        size_tensor = torch.tensor([0], dtype=torch.int32)
        torch.distributed.recv(size_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        bytes_size = size_tensor.item()
        recv_tensor = torch.empty(bytes_size, dtype=torch.uint8)
        torch.distributed.recv(recv_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        return _tensor_to_object(recv_tensor, bytes_size,
                                 torch.distributed.group.WORLD)

    @log_op
    def send_object(self, obj, dest, tag=0):
        self.isend_object(obj, dest, tag).wait()

    @log_op
    def isend_object(self, obj, dest, tag=0):
        input_tensor, local_size = _object_to_tensor(
            obj, torch.device("cpu"), torch.distributed.group.WORLD)

        # Send object size
        works = []
        works.append(
            torch.distributed.isend(torch.tensor([local_size],
                                                 dtype=torch.int32),
                                    dst=dest,
                                    tag=tag))
        works.append(torch.distributed.isend(input_tensor, dst=dest, tag=tag))
        return MultiHandleWrapper(works)

    @log_op
    def allreduce(
        self,
        obj: int | float | torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        is_base_type = isinstance(obj, int) or isinstance(obj, float)
        if is_base_type:
            obj = torch.tensor(obj)

        dist.all_reduce(obj, op=reduce_op_to_torch(op))

        if is_base_type:
            obj = obj.item()

        return obj

    @log_op
    def tp_allreduce(
        self,
        obj: int | float | torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        is_base_type = isinstance(obj, int) or isinstance(obj, float)
        if is_base_type:
            obj = torch.tensor(obj)

        dist.all_reduce(obj,
                        op=reduce_op_to_torch(op),
                        group=self.mapping.tp_group_pg)

        if is_base_type:
            obj = obj.item()

        return obj

    @log_op
    def tp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.tp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.tp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.tp_group_pg)
            return output_list

    @log_op
    def tp_gather(self, obj, dst=0):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.tp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.tp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.tp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.tp_group_pg)
            return output_list

    @log_op
    def tp_broadcast(self, obj, root=0, **kwargs):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.tp_group_pg)
            return obj
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.tp_group_pg,
                device=torch.device("cpu"))
            return ret[0]

    @log_op
    def cp_broadcast(self, obj, root=0, **kwargs):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.cp_group_pg)
            return obj
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.cp_group_pg,
                device=torch.device("cpu"))
            return ret[0]

    @log_op
    def cp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.cp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.cp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.cp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.cp_group_pg)
            return output_list

    @log_op
    def pp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.pp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.pp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_gather(self, obj, dst=0):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.pp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.pp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.pp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_broadcast(self, obj, root=0):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.pp_group_pg)
            return obj
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.pp_group_pg,
                device=torch.device("cpu"))
            return ret[0]


class PPCommNCCL:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self._topology = self._mapping_topology(global_mapping)
        self._reconfigure_lock = threading.Lock()
        self._reconfigure_generation = 0
        self._completed_recovery = None
        self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
            self.mapping.world_size,
            self.mapping.rank,
        )
        if NCCL_FAULT_TOLERANCE_ENABLED:
            # The FT native communicator survives process-local Python wrapper
            # churn. Mirror its actual survivor membership instead of assuming
            # a newly-created wrapper always represents the initial world.
            self._active_ranks = tuple(
                int(rank) for rank in self.nccl_comm.get_active_ranks())
        else:
            # Preserve the legacy constructor path when FT is disabled.
            self._active_ranks = tuple(range(self.mapping.world_size))
        self.tensor_ready_event = torch.cuda.Event()
        self.send_stream = torch.cuda.Stream()

    @staticmethod
    def _mapping_topology(mapping: Mapping) -> tuple:
        return (
            int(mapping.world_size),
            int(mapping.rank),
            tuple(mapping.pp_group),
            get_helix_cp_mnnvl_topology(mapping),
        )

    def is_compatible(self, mapping: Mapping) -> bool:
        """Whether another engine can share this process-local communicator."""
        return self._topology == self._mapping_topology(mapping)

    def _validate_peer(self, peer: int) -> int:
        peer = int(peer)
        if peer not in self._topology[2]:
            raise RuntimeError(
                f"NCCL error: peer world rank {peer} is not in this rank's PP group "
                f"{list(self._topology[2])}")
        if peer not in self._active_ranks:
            raise RuntimeError(
                f"NCCL error: peer world rank {peer} is not active in the current PP communicator"
            )
        return peer

    def _required_pp_peer(self, *, next_peer: bool) -> int:
        # Do not silently skip a failed stage: connecting non-adjacent stages
        # would bypass model layers. A higher-level topology reconstruction may
        # pass an explicit remapped peer once it has reassigned those layers.
        peer = (self.mapping.next_pp_rank()
                if next_peer else self.mapping.prev_pp_rank())
        return self._validate_peer(peer)

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if not NCCL_FAULT_TOLERANCE_ENABLED:
            if dest is None:
                dest = self.mapping.next_pp_rank()
        elif dest is None:
            dest = self._required_pp_peer(next_peer=True)
        else:
            dest = self._validate_peer(dest)

        # NCCL send kernel in send_stream cannot be captured,
        # so we send in the current stream instead in CUDA graph cases.
        if torch.cuda.is_current_stream_capturing():
            self.nccl_comm.send(tensor, dest)
            return

        # If the tensor is allocated from non-default memory pool
        # like userbuffers, its underlying memory may be reused
        # before the send operation is completed.
        # We clone the tensor to avoid write-write conflicts.
        tensor = tensor.clone()
        self.send_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.send_stream):
            self.nccl_comm.send(tensor, dest)

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if not NCCL_FAULT_TOLERANCE_ENABLED:
            if src is None:
                src = self.mapping.prev_pp_rank()
        elif src is None:
            src = self._required_pp_peer(next_peer=False)
        else:
            src = self._validate_peer(src)
        self.nccl_comm.recv(tensor, src)

    def abort(self, generation: Optional[int] = None) -> None:
        """Abort the PP communicator on this survivor."""
        # Capture the generation before waiting for a concurrent rebuild. If
        # that rebuild succeeds, this abort belongs to the old communicator
        # and must not tear down its replacement. An abort that starts after
        # the rebuild observes the new generation and still takes effect.
        if generation is None:
            generation = self._reconfigure_generation
        with self._reconfigure_lock:
            if generation != self._reconfigure_generation:
                return
            self.nccl_comm.abort()

    def abort_and_reinit(self,
                         active_ranks: List[int],
                         generation: Optional[int] = None) -> None:
        """Rebuild the PP communicator using only surviving world ranks.

        Every survivor must call this with the same rank set. The C++ wrapper
        exchanges a fresh NCCL unique ID point-to-point, so excluded ranks do
        not participate in communicator initialization.

        ``generation`` is the coordinator's shared monotonic recovery-event
        generation, advanced for every distinct attempt. It is required for
        same-membership recovery so every rank enters the native rendezvous
        even when local watchdog observations differ. Repeated callbacks with
        the same generation and target are idempotent. A retry after any native
        failure must use a newly advanced generation so stale rendezvous traffic
        cannot pair different recovery attempts.
        """
        canonical_ranks = tuple(sorted(int(rank) for rank in active_ranks))
        if not canonical_ranks:
            raise ValueError("active_ranks must not be empty")
        if len(canonical_ranks) != len(set(canonical_ranks)):
            raise ValueError("active_ranks must not contain duplicates")
        if self.mapping.rank not in canonical_ranks:
            raise ValueError(
                f"current world rank {self.mapping.rank} is not active")
        generation = _canonical_recovery_generation(generation)
        rendezvous_id = _recovery_rendezvous_id(generation)

        # Failure notifications can be duplicated or delivered concurrently by
        # independent control paths. Keep the native rebuild and its Python
        # membership mirror in one transaction so a stale callback cannot
        # widen membership after a newer shrink has completed.
        with self._reconfigure_lock:
            current_ranks = tuple(
                int(rank) for rank in self.nccl_comm.get_active_ranks())
            self._active_ranks = current_ranks
            if generation is not None and self._completed_recovery is not None:
                completed_generation, completed_target = self._completed_recovery
                if generation < completed_generation:
                    return
                if generation == completed_generation:
                    if canonical_ranks != completed_target:
                        raise RuntimeError(
                            "NCCL error: conflicting PP communicator recovery "
                            f"target for generation {generation}: completed "
                            f"{list(completed_target)}, requested {list(canonical_ranks)}"
                        )
                    return
            if canonical_ranks == current_ranks and generation is None:
                raise ValueError(
                    "generation is required for same-membership PP communicator "
                    "recovery so every survivor makes the same rebuild decision"
                )
            if not set(canonical_ranks).issubset(current_ranks):
                raise ValueError(
                    "abort_and_reinit cannot reactivate a removed rank")

            self.nccl_comm.abort_and_reinit(list(canonical_ranks),
                                            rendezvous_id)
            native_ranks = tuple(
                int(rank) for rank in self.nccl_comm.get_active_ranks())
            self._active_ranks = native_ranks
            self._reconfigure_generation += 1
            if native_ranks != canonical_ranks:
                raise RuntimeError(
                    "NCCL error: rebuilt PP communicator returned unexpected "
                    f"membership {list(native_ranks)}; expected {list(canonical_ranks)}"
                )
            if generation is not None:
                self._completed_recovery = (generation, canonical_ranks)

    def get_async_error(self) -> str:
        """Return the C++ wrapper's latched NCCL abort/error reason, if any."""
        return self.nccl_comm.get_async_error()

    def get_active_ranks(self) -> List[int]:
        """Return original world-rank IDs in the current PP communicator."""
        with self._reconfigure_lock:
            self._active_ranks = tuple(
                int(rank) for rank in self.nccl_comm.get_active_ranks())
            return list(self._active_ranks)


class PPCommTorch:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.pg = self.mapping.pp_group_pg
        self.pg_group = self.mapping.pp_group

    def _global_to_local_rank(self, global_rank: int):
        assert global_rank in self.pg_group
        return self.pg_group.index(global_rank)

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()

        work = self.pg.send([tensor], self._global_to_local_rank(dest), tag=0)
        # Send operation cannot be captured without blocking wait,
        # so we block the current stream in CUDA graph cases.
        if torch.cuda.is_current_stream_capturing():
            work.block_current_stream()

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()

        work = self.pg.recv([tensor], self._global_to_local_rank(src), tag=0)
        work.block_current_stream()


_pp_comm = None
_pp_comm_refcount = 0
_pp_comm_lock = threading.Lock()
_pp_comm_condition = threading.Condition(_pp_comm_lock)
_pp_comm_control_refcount = 0
_pp_comm_final_release_pending = False


def init_pp_comm(mapping):
    """Acquire the process-local PP communicator for one model engine."""
    global _pp_comm, _pp_comm_refcount
    with _pp_comm_condition:
        while (_pp_comm_control_refcount > 0 or _pp_comm_final_release_pending):
            _pp_comm_condition.wait()
        created = False
        if mpi_disabled():
            if _pp_comm is None:
                _pp_comm = PPCommTorch(mapping)
                created = True
            elif not isinstance(_pp_comm, PPCommTorch):
                raise RuntimeError(
                    "PP communicator is already initialized with a different backend"
                )
            elif _pp_comm.mapping != mapping:
                raise RuntimeError(
                    "PP communicator is already initialized for a different topology"
                )
        else:
            if _pp_comm is None:
                _pp_comm = PPCommNCCL(mapping)
                created = True
            elif not isinstance(_pp_comm, PPCommNCCL):
                raise RuntimeError(
                    "NCCL error: PP communicator is already initialized with a different backend"
                )
            elif not _pp_comm.is_compatible(mapping):
                raise RuntimeError(
                    "NCCL error: PP communicator is already initialized for a "
                    "different topology (PP or Helix MNNVL CP topology)")
        try:
            init_helix_cp_comm(mapping)
        except Exception:
            if created and _pp_comm_refcount == 0:
                _pp_comm = None
            raise
        _pp_comm_refcount += 1


def release_pp_comm() -> None:
    """Release one model engine's ownership of the shared PP communicator."""
    global _pp_comm, _pp_comm_refcount, _pp_comm_final_release_pending
    with _pp_comm_condition:
        if _pp_comm_refcount <= 0:
            raise RuntimeError(
                "PP communicator release has no matching acquisition")
        _pp_comm_refcount -= 1
        if _pp_comm_refcount == 0:
            _pp_comm_final_release_pending = True
            while _pp_comm_control_refcount > 0:
                _pp_comm_condition.wait()
            if not (NCCL_FAULT_TOLERANCE_ENABLED
                    and isinstance(_pp_comm, PPCommNCCL)):
                # Preserve the legacy teardown behavior when FT is disabled.
                _pp_comm = None
            # In FT mode both the native communicator and its Python recovery
            # generation state are process-lifetime. A rank-local final release
            # is not a distributed teardown barrier, and dropping either state
            # could make only one rank enter a duplicate-generation rendezvous.
            _pp_comm_final_release_pending = False
            _pp_comm_condition.notify_all()


def _get_pp_nccl_comm_locked() -> PPCommNCCL:
    """Return the NCCL PP communicator while ``_pp_comm_lock`` is held."""
    if not isinstance(_pp_comm, PPCommNCCL):
        raise RuntimeError("NCCL error: PP communicator is not initialized")
    return _pp_comm


def _acquire_pp_nccl_control(
        *,
        capture_generation: bool = False) -> tuple[PPCommNCCL, Optional[int]]:
    """Keep the shared communicator alive without holding its lifecycle lock."""
    global _pp_comm_control_refcount
    with _pp_comm_lock:
        comm = _get_pp_nccl_comm_locked()
        if _pp_comm_refcount <= 0 or _pp_comm_final_release_pending:
            raise RuntimeError("NCCL error: PP communicator is being released")
        _pp_comm_control_refcount += 1
        generation = (comm._reconfigure_generation
                      if capture_generation else None)
        return comm, generation


def _release_pp_nccl_control() -> None:
    global _pp_comm_control_refcount
    with _pp_comm_condition:
        if _pp_comm_control_refcount <= 0:
            raise RuntimeError(
                "NCCL error: PP communicator control reference underflow")
        _pp_comm_control_refcount -= 1
        if _pp_comm_control_refcount == 0:
            _pp_comm_condition.notify_all()


def pp_comm_abort() -> None:
    """Abort the process-local NCCL PP communicator."""
    comm, generation = _acquire_pp_nccl_control(capture_generation=True)
    try:
        comm.abort(generation)
    finally:
        comm = None
        _release_pp_nccl_control()


def pp_comm_abort_and_reinit(active_ranks: List[int],
                             generation: Optional[int] = None) -> None:
    """Rebuild the NCCL PP communicator using surviving original world ranks."""
    comm, _ = _acquire_pp_nccl_control()
    try:
        comm.abort_and_reinit(active_ranks, generation)
    finally:
        comm = None
        _release_pp_nccl_control()


def pp_comm_get_async_error() -> str:
    """Return the NCCL PP communicator's latched abort/error reason."""
    comm, _ = _acquire_pp_nccl_control()
    try:
        return comm.get_async_error()
    finally:
        comm = None
        _release_pp_nccl_control()


def pp_comm_get_active_ranks() -> List[int]:
    """Return original world-rank IDs in the NCCL PP communicator."""
    comm, _ = _acquire_pp_nccl_control()
    try:
        return comm.get_active_ranks()
    finally:
        comm = None
        _release_pp_nccl_control()


@TorchDist.log_op
def pp_recv(tensor):
    """Receive tensors from previous pp rank."""
    _pp_comm.recv(tensor)


@TorchDist.log_op
def pp_send(tensor):
    """Send tensors to next pp rank."""
    _pp_comm.send(tensor)


@torch.library.custom_op("trtllm::pp_recv_tensors", mutates_args=("tensors", ))
def pp_recv_tensors(tensors: List[torch.Tensor]) -> None:
    """
    Receive tensors from previous pp rank.
    """
    for tensor in tensors:
        pp_recv(tensor)


@torch.library.custom_op("trtllm::pp_send_tensors", mutates_args=("tensors", ))
def pp_send_tensors(tensors: List[torch.Tensor]) -> None:
    """Send tensors to next pp rank."""
    for tensor in tensors:
        pp_send(tensor)
