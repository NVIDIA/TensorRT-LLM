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
"""Start Wide-EP ranks with fault tolerance, and the membership authority on rank 0.

Every rank runs as its own one-process MPI world, so a failed rank can restart alone. The ranks
reach each other through the rendezvous store that the membership authority hosts.
"""

from __future__ import annotations

import functools
import hashlib
import ipaddress
import logging
import os
import pickle  # nosec B403
import secrets
import select
import socket
import subprocess  # nosec B404 - starts the membership authority with sys.executable and a fixed argv, no shell
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Optional

from tensorrt_llm._utils import global_mpi_size, local_mpi_comm, set_mpi_world_comm
from tensorrt_llm._wide_ep_fault_tolerance import (
    FAULT_TOLERANCE_RANK_ENV,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    TURN_FAULT_TOLERANCE_OFF,
    RankIdentity,
    check_fault_tolerance_requirements,
    wide_ep_fault_tolerance_rank,
)
from tensorrt_llm.executor.utils import WorkerCommIpcAddrs, get_spawn_proxy_process_env
from tensorrt_llm.executor.worker import worker_main
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy
from tensorrt_llm.logger import logger
from tensorrt_llm.version import __version__

from ...autotuner import AutoTuner
from ...pyexecutor.hang_detector import RANK_CRASH_KILL_GRACE_ENV
from ..communicator import Distributed
from ..host_exchange import MnnvlDistFactory
from ..startup_record import REPLACEMENT_CANNOT_START, ReplacementCannotStart, StartupRecord
from .liveness import MembershipClient, close_in_forked_child, rendezvous_store
from .records import (
    MAX_RANKS,
    RENDEZVOUS_TIMEOUT_S,
    MembershipRecord,
    MembershipState,
    launcher_free_env,
    rank_mask,
)
from .rendezvous import StoreGroup, rendezvous_host_port
from .signed_store import SHARED_SECRET_BYTES, SignedStore, parse_shared_secret

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

_EPHEMERAL_PORT_RANGE = "/proc/sys/net/ipv4/ip_local_port_range"


@dataclass
class RankStartState:
    """The state of this process's start as a fault-tolerant rank."""

    record: StartupRecord
    client: Optional[MembershipClient] = None
    # The authority process, which only rank 0 starts.
    authority: Optional[subprocess.Popen] = None
    fingerprint: Optional[str] = None
    # The startup steps this rank completed, and the peers its running startup collective waits for.
    steps: int = 0
    waiting_for: int = 0
    every_rank_started: threading.Event = field(default_factory=threading.Event)
    predecessor_memory_released: threading.Event = field(default_factory=threading.Event)


# Every rank reports its startup steps, so a long checkpoint load is not taken for a stalled startup.
_state = RankStartState(StartupRecord(report_progress=lambda: _report_startup_wait(0)))


def ensure_rendezvous(
    orchestrator_type: Optional[str],
) -> tuple[dict[str, str], Optional[socket.socket]]:
    """Return the shared secret and rendezvous address for the ranks this process starts.

    Exported values are kept. Otherwise the secret is new, and an SO_REUSEADDR socket that never
    listens reserves the port, so the authority can still bind it. Until the authority binds it,
    another process that binds this loopback port with SO_REUSEADDR can take it, and the authority
    then fails to start. Keep the returned socket open while the deployment runs. An attached
    serving frontend starts no rank, so it gets no values.

    Raises:
        ValueError: The deployment does not use the MPI executor, this process is a rank of an MPI
            job, or trtllm-llmapi-launch started the ranks without its node supervisor.
    """
    if os.environ.get("TLLM_EXECUTOR_ATTACH_INFO") is not None:
        return {}, None
    check_fault_tolerance_requirements(
        [
            (
                orchestrator_type is None,
                f"the MPI executor (orchestrator_type unset, not {orchestrator_type!r}), because "
                "only its worker starts the membership authority and the membership of each rank",
            ),
            (
                global_mpi_size() == 1,
                "one start of the server or script, not one on every rank of an MPI job, because "
                "Wide-EP fault tolerance starts each rank as a process of its own",
            ),
        ]
    )
    if get_spawn_proxy_process_env() and wide_ep_fault_tolerance_rank() is None:
        raise ValueError(
            "Wide-EP fault tolerance does not support ranks launched by trtllm-llmapi-launch yet, "
            "so it runs on one node only. Start trtllm-serve or the LLM API without a launcher."
        )
    env = {
        SHARED_SECRET_ENV: os.environ.get(SHARED_SECRET_ENV)
        or secrets.token_hex(SHARED_SECRET_BYTES)
    }
    reserved = None
    if RENDEZVOUS_ENV in os.environ:
        env[RENDEZVOUS_ENV] = os.environ[RENDEZVOUS_ENV]
        _warn_if_ephemeral(env[RENDEZVOUS_ENV])
    else:
        reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reserved.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Without an exported address, this process starts every rank on its own node, so the store
        # is reachable on loopback only.
        reserved.bind(("127.0.0.1", 0))
        env[RENDEZVOUS_ENV] = f"127.0.0.1:{reserved.getsockname()[1]}"
    return env, reserved


def _warn_if_ephemeral(address: str) -> None:
    """Warn when an exported rendezvous port is in this host's ephemeral port range.

    The authority binds the port only once rank 0 starts it, so an outgoing connection may take it
    first and fail the startup.
    """
    _, port = rendezvous_host_port(address)
    try:
        with open(_EPHEMERAL_PORT_RANGE) as ports:
            low, high = (int(bound) for bound in ports.read().split())
    except (OSError, ValueError):
        return
    if low <= port <= high:
        logger.warning(
            f"{RENDEZVOUS_ENV}={address} names port {port}, inside this host's ephemeral port "
            f"range {low}-{high}, where an outgoing connection can take it before the membership "
            "authority binds it. Choose a port outside net.ipv4.ip_local_port_range"
        )


def _launched_rank() -> RankIdentity:
    """Return this process's rank after checking the shared secret and rendezvous address."""
    try:
        secret = os.environ[SHARED_SECRET_ENV]
        rendezvous_host_port(os.environ[RENDEZVOUS_ENV])
    except (KeyError, ValueError):
        raise ValueError(
            f"Wide-EP fault tolerance requires a shared secret in {SHARED_SECRET_ENV} and "
            f"host:port in {RENDEZVOUS_ENV} on every rank, or {TURN_FAULT_TOLERANCE_OFF}"
        ) from None
    try:
        parse_shared_secret(secret)
    except ValueError as error:
        raise ValueError(f"{error}. Set it on every rank, or {TURN_FAULT_TOLERANCE_OFF}") from None
    identity = _rank_identity()
    if identity.world_size > identity.gpus_per_node:
        _check_rendezvous_reachable(os.environ[RENDEZVOUS_ENV])
    return identity


def _check_rendezvous_reachable(address: str) -> None:
    """Refuse a rendezvous address that other nodes cannot reach.

    The authority listens on the address the host resolves to on rank 0's node, and each rank
    connects to the address it resolves to on its own node.
    """
    host, _ = rendezvous_host_port(address)
    try:
        resolved = socket.gethostbyname(host)
    except OSError as error:
        raise ValueError(
            f"Cannot resolve {RENDEZVOUS_ENV}={address} on this node ({error})"
        ) from None
    if ipaddress.ip_address(resolved).is_loopback:
        raise ValueError(
            f"{RENDEZVOUS_ENV}={address} resolves to the loopback address {resolved} on this "
            f"node. Set a host that every node reaches, or {TURN_FAULT_TOLERANCE_OFF}"
        )


def _rank_identity() -> RankIdentity:
    identity = wide_ep_fault_tolerance_rank()
    if identity is None:
        raise ValueError(
            "Wide-EP fault tolerance requires '<rank>,<world size>,<GPUs per node>,<replacement>' "
            f"in {FAULT_TOLERANCE_RANK_ENV}, which the trtllm-llmapi-launch node supervisor sets"
        )
    return identity


# Rank 0's startup record and startup fingerprint, which a replacement replays.
AGREED_STARTUP_KEY = "agreed_startup"
WORKER_ARGS_KEY = "worker_args"
# Each rank's tuned tactics, which a replacement of the rank takes. Format with the rank.
TUNED_TACTICS_KEY = "tuned_tactics/{}"


def startup_fingerprint(worker_main_args: dict) -> str:
    """Return a digest of the TensorRT-LLM version and the checkpoint's ``config.json``."""
    digest = hashlib.sha256(__version__.encode())
    model_dir = worker_main_args.get("hf_model_dir")
    if model_dir is not None and Path(model_dir, "config.json").is_file():
        digest.update(Path(model_dir, "config.json").read_bytes())
    return digest.hexdigest()


def start_rank(worker_main_args: dict) -> None:
    """Start this rank's membership in ``worker_main``.

    On rank 0, start the authority, publish the worker arguments for the other ranks, join the
    membership and record the startup. On another rank, check that :func:`worker_node_main` joined.
    """
    _state.fingerprint = startup_fingerprint(worker_main_args)
    identity = _launched_rank()
    if identity.rank == 0:
        _state.authority = _start_authority(
            identity.world_size, identity.gpus_per_node, RENDEZVOUS_TIMEOUT_S
        )
        store = _startup_store()
        store.set(WORKER_ARGS_KEY, _worker_args_for_other_ranks(worker_main_args))
        _join_at_entry(store, identity)
        _state.record.record()
    elif _state.client is None:
        raise ValueError(
            f"Rank {identity.rank} must start through worker_node_main, as trtllm-llmapi-launch "
            "does"
        )


def _worker_args_for_other_ranks(worker_main_args: dict) -> bytes:
    """Return rank 0's worker arguments without IPC addresses and keys, for the plaintext store."""
    return pickle.dumps(
        {
            **worker_main_args,
            "worker_queues": WorkerCommIpcAddrs(None, None, None),
            "rpc_addr": None,
            "hmac_key": b"",
        }
    )


def worker_node_main() -> None:
    """Run a rank other than 0 with fault tolerance, as trtllm-llmapi-launch does.

    A replacement that does not match the deployment, or whose startup failed in a way that a
    restart would repeat, ends with exit status ``REPLACEMENT_CANNOT_START``.
    """
    identity = _launched_rank()
    if identity.rank == 0:
        raise ValueError(
            "Rank 0 runs in the process that starts the deployment, not through worker_node_main"
        )
    store = _startup_store()
    _join_at_entry(store, identity)
    try:
        # Only this deployment's processes hold the shared secret that signs the store's values.
        worker_main_args = pickle.loads(store.get(WORKER_ARGS_KEY))  # nosec B301
        if identity.replacement:
            agreed_startup = pickle.loads(store.get(AGREED_STARTUP_KEY))  # nosec B301
            if startup_fingerprint(worker_main_args) != agreed_startup["fingerprint"]:
                raise ReplacementCannotStart(
                    "The replacement's TensorRT-LLM version or checkpoint differs from the "
                    "deployment's"
                )
            _state.record.replay(agreed_startup)
            _take_tuned_tactics(store, identity.rank)
            _start_alone(worker_main_args)
        worker_main(**worker_main_args)
    except ReplacementCannotStart as error:
        end_process(
            identity.rank,
            REPLACEMENT_CANNOT_START,
            f"no replacement of this rank can start: {error}",
        )


def publish_tuned_tactics(store: SignedStore, rank: int) -> None:
    """Publish the tactics this rank had tuned where the recorded startup collectives start.

    A replacement of the rank starts from them, so its startup makes the same tuning collectives as
    the original ranks' later warmup. Each rank publishes its own, because the cache keys of some
    MoE ops hold the rank's expert offset.
    """
    store.set(TUNED_TACTICS_KEY.format(rank), pickle.dumps(_state.record.tuned_tactics))


def _take_tuned_tactics(store: SignedStore, rank: int) -> None:
    """Start this replacement of ``rank`` from its predecessor's tuned tactics, and keep them."""
    tuned_tactics = pickle.loads(store.get(TUNED_TACTICS_KEY.format(rank)))  # nosec B301
    _state.record.tuned_tactics = tuned_tactics
    AutoTuner.get().profiling_cache.merge_cache_data(tuned_tactics)


def _start_alone(worker_main_args: dict) -> None:
    """Make a replacement rank load its checkpoint alone and skip the KV cache estimation.

    It takes the KV cache capacity from the startup record and the tuned tactics from its
    predecessor instead. So it neither loads nor saves the autotuner cache file of the original
    ranks, whether the environment or ``env_overrides`` sets it: loading the file would replace the
    tactics it took.
    """
    os.environ["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] = "1"
    os.environ.pop("TLLM_AUTOTUNER_CACHE_PATH", None)
    llm_args = worker_main_args["llm_args"]
    update = {"checkpoint_io_policy": CheckpointIOPolicy.NATIVE}
    if llm_args.env_overrides and "TLLM_AUTOTUNER_CACHE_PATH" in llm_args.env_overrides:
        update["env_overrides"] = {
            name: value
            for name, value in llm_args.env_overrides.items()
            if name != "TLLM_AUTOTUNER_CACHE_PATH"
        }
    requested = llm_args.checkpoint_io_policy
    if requested != CheckpointIOPolicy.NATIVE:
        log = (
            logger.warning
            if requested == CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD
            else logger.info
        )
        log(
            "Checkpoint I/O policy resolved before loading: "
            f"requested={requested}, "
            f"selected={CheckpointIOPolicy.NATIVE}, "
            "reason=a replacement rank loads its checkpoint alone, and rank-striped read-ahead "
            "splits the checkpoint reads among the ranks of a node."
        )
    worker_main_args["llm_args"] = llm_args.model_copy(update=update)


@functools.lru_cache(maxsize=1)
def _startup_store() -> SignedStore:
    return rendezvous_store(RENDEZVOUS_TIMEOUT_S)


def _report_startup_wait(waiting_for: int) -> None:
    """Report each change in the peers that a startup collective waits for, 0 once it completed."""
    if waiting_for and waiting_for == _state.waiting_for:
        return
    _state.waiting_for = waiting_for
    if not waiting_for:
        _state.steps += 1
    _state.client.report(waiting_for=waiting_for, progress=_state.steps)


def _on_tick_at_startup(host_collectives: MnnvlDistFactory) -> None:
    """Report the peers that a startup collective of the startup group or ``MnnvlDist`` waits for.

    The peers a host collective waits for are the ranks of its waited epochs. As in serving, each
    new host collective round that waits for a peer counts as the peer's progress, so a peer that
    keeps completing host collectives is not taken as hung.
    """
    waited_epochs = host_collectives.waited_epochs
    _state.client.report(
        waiting_for=_state.waiting_for | rank_mask((r for r, _ in waited_epochs), MAX_RANKS),
        waited_epochs=waited_epochs,
    )


def join_startup_group(mapping: Mapping) -> None:
    """Make the startup group this rank's ``mpi_comm()`` and MPI world, and set ``TLLM_RANK``.

    ``Distributed`` then runs the host collectives over MNNVL memory (``MnnvlDist``). It also sets
    ``TLLM_RANK_CRASH_HARD_KILL_GRACE=-1``, so that a crashed rank does not kill its peers.
    """
    identity = _rank_identity()
    one_node = identity.world_size <= min(identity.gpus_per_node, mapping.gpus_per_node)
    nodes = identity.world_size // identity.gpus_per_node
    check_fault_tolerance_requirements(
        [
            (
                mapping.gpus_per_node == identity.gpus_per_node or one_node,
                f"gpus_per_node {identity.gpus_per_node}, the GPUs that the node supervisor sees "
                f"on each node (CUDA_VISIBLE_DEVICES), not {mapping.gpus_per_node}",
            ),
            (
                mapping.world_size == identity.world_size,
                f"a world size of {identity.world_size}, the ranks that the node supervisors start "
                f"({nodes} nodes with {identity.gpus_per_node} GPUs each), not the model's "
                f"{mapping.world_size}",
            ),
        ]
    )
    os.environ["TLLM_RANK"] = str(identity.rank)
    _turn_off_rank_crash_kill()
    startup_group = StoreGroup(
        _startup_store(),
        "startup/world",
        identity.rank,
        range(identity.world_size),
        _report_startup_wait,
        active=[identity.rank] if identity.replacement else None,
        gpus_per_node=identity.gpus_per_node,
        startup=True,
    )
    set_mpi_world_comm(startup_group)
    host_collectives = MnnvlDistFactory()
    Distributed.set_implementation(host_collectives, _state.record)
    # A host collective waits in native code, so the client's thread reports whom it waits for.
    _state.client.bind(
        _on_record_at_startup,
        functools.partial(_on_tick_at_startup, host_collectives),
        _on_error_at_startup,
    )
    if not isinstance(local_mpi_comm(), StoreGroup):
        raise RuntimeError(
            "The node's communicator was split from MPI before the startup group was set"
        )


def _turn_off_rank_crash_kill() -> None:
    """Keep a crashed rank from killing its peers: its exit ends only itself.

    The membership removes a rank that ends, so no peer waits for it, and the error that ended the
    rank is reported as it is.
    """
    grace = os.environ.get(RANK_CRASH_KILL_GRACE_ENV)
    try:
        turned_off = grace is None or float(grace) < 0
    except ValueError:
        turned_off = False
    if not turned_off:
        logger.warning(
            f"{RANK_CRASH_KILL_GRACE_ENV}={grace} is replaced by -1: with Wide-EP fault tolerance a "
            "crashed rank ends only itself, and the membership removes it"
        )
    os.environ[RANK_CRASH_KILL_GRACE_ENV] = "-1"


def _join_at_entry(store: SignedStore, identity: RankIdentity) -> None:
    """Join the membership, and exit if the deployment ends before the executor takes it over.

    A replacement then waits for a committed membership without its rank. Its predecessor has
    ended, and the active ranks release the memory they imported from it before that commit, so
    the replacement sizes its KV cache from all device memory.

    Raises:
        TimeoutError: A replacement saw no such commit within the rendezvous timeout.
    """
    _state.client = MembershipClient(
        identity.rank,
        store,
        _on_record_at_startup,
        replacement=identity.replacement,
        on_error=_on_error_at_startup,
    )
    _state.client.start()
    if identity.replacement and not _state.predecessor_memory_released.wait(RENDEZVOUS_TIMEOUT_S):
        raise TimeoutError(
            f"Rank {identity.rank}'s replacement waited {RENDEZVOUS_TIMEOUT_S:.0f} s for the "
            "active ranks to release its predecessor's memory"
        )


def _on_record_at_startup(record: MembershipRecord) -> None:
    if record.state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
        status = 0 if record.state == MembershipState.SHUTDOWN else 1
        end_process(
            _state.client.rank, status, f"the membership record is {record} during the startup"
        )
    if record.state != MembershipState.STARTING:
        _state.every_rank_started.set()
    if record.state == MembershipState.COMMITTED and not record.is_active(_state.client.rank):
        _state.predecessor_memory_released.set()


def _on_error_at_startup(error: Exception) -> None:
    end_process(
        _state.client.rank, 1, f"the membership client failed during the startup ({error!r})"
    )


def membership_client() -> MembershipClient:
    """Return the membership client this process started when its worker began."""
    return _state.client


def authority_process() -> Optional[subprocess.Popen]:
    """Return the authority process that this process started. Only rank 0 starts one."""
    return _state.authority


def publish_startup_record(store: SignedStore) -> None:
    """Publish rank 0's startup record and startup fingerprint for a replacement to replay."""
    agreed_startup = {**_state.record.published(), "fingerprint": _state.fingerprint}
    store.set(AGREED_STARTUP_KEY, pickle.dumps(agreed_startup))


def wait_for_every_rank() -> None:
    """Report this rank started and wait until every rank has."""
    _state.client.report(started=True)
    _state.every_rank_started.wait()


def end_process(rank: int, status: int, reason: str) -> NoReturn:
    """End this process with ``status`` at once, logging ``reason``."""
    (logger.info if status == 0 else logger.error)(f"Rank {rank} exits: {reason}")
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def _start_authority(world_size: int, gpus_per_node: int, timeout_s: float) -> subprocess.Popen:
    """Start the membership authority and return its process once its store listens.

    This process holds the write end of the authority's standard input pipe, so the authority ends
    with it. The authority runs without MPI, launcher variables or a GPU.
    """
    env = launcher_free_env(os.environ)
    env.update(
        TRT_LLM_NO_LIB_INIT="1",
        MPI4PY_RC_INITIALIZE="false",
        CUDA_VISIBLE_DEVICES="",
        TLLM_LOG_LEVEL=logger.level,
    )
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as address_pipe:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorrt_llm._torch.distributed.membership",
                    f"--world-size={world_size}",
                    f"--gpus-per-node={gpus_per_node}",
                    f"--address-fd={write_fd}",
                ],
                env=env,
                pass_fds=(write_fd,),
                stdin=subprocess.PIPE,
            )
        finally:
            os.close(write_fd)
        close_in_forked_child(process.stdin)
        ready, _, _ = select.select([address_pipe], [], [], timeout_s)
        address = address_pipe.readline().strip() if ready else ""
    if not address:
        process.kill()
        raise RuntimeError(
            f"The membership authority did not report its store address within {timeout_s} s "
            f"(exit status {process.wait()}), check that "
            f"{RENDEZVOUS_ENV}={os.environ.get(RENDEZVOUS_ENV)} names a free port on rank 0's node"
        )
    return process
