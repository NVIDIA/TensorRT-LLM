# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-instance multi-node disaggregated serving setup helper.

Background: when a disagg test spawns multiple LLM instances (e.g. 2 ctx + 1 gen)
across a multi-node MPI allocation managed by `trtllm-llmapi-launch`, every
`trtllm-serve serve` subprocess connects to the *same* mgmn proxy whose pool
size equals `SLURM_NTASKS`. Because each LLM's world size (TP*PP*CP) is
smaller than the pool size, `Mapping(rank=mpi_rank(), world_size=tp*pp*cp)`
ends up with `rank >= world_size` on the spill-over ranks, and the very first
collective `mapping.tp_group` call raises `IndexError: list index out of range`.

The fix is to use the same pattern as `trtllm-serve disaggregated_mpi_worker`:
each rank calls `split_world_comm` to partition `MPI_COMM_WORLD` into per-
instance sub-communicators. The instance leader rank then launches its own
LLM bound to its sub-comm; non-leader ranks join the sub-comm's
`MPICommExecutor`. The frontend (OpenAI-compatible HTTP server that fans out
to ctx/gen workers) runs as a separate subprocess on the pytest process.

This helper plugs that wiring on top of the existing mgmn proxy that
`trtllm-llmapi-launch` has already set up — no extra mpirun / srun is
required.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time

import yaml
from disagg_test_utils import ProcessWrapper, wait_for_disagg_server_ready

# ---------------------------------------------------------------------------
# Task dispatched to every mgmn worker rank. We reuse the publicly-exposed
# `disaggregated_mpi_worker_main` from tensorrt_llm.commands.serve so the
# function reference can be unpickled on workers without needing the test
# directory on PYTHONPATH. `disaggregated_mpi_worker_main` itself fixes up
# sys.argv[0] internally for the non-CLI entry case.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hostname / port plumbing
# ---------------------------------------------------------------------------
def _resolve_rank_to_hostname(mpi_session, mpi_world_size: int) -> list[str]:
    """Return a list of hostnames indexed by global MPI rank.

    Dispatches `allgather_hostnames` to all MPI ranks. Each rank returns the
    full rank-ordered hostname list (mpi_allgather guarantees identical
    output on every rank); we just take the first one returned.
    """
    from tensorrt_llm.llmapi.disagg_utils import allgather_hostnames

    results = mpi_session.submit_sync(allgather_hostnames)
    if not results:
        raise RuntimeError("allgather_hostnames returned no results")
    hostnames = list(results[0])
    if len(hostnames) != mpi_world_size:
        raise RuntimeError(
            f"allgather_hostnames returned {len(hostnames)} entries, expected {mpi_world_size}"
        )
    return hostnames


def _allocate_free_port_on_host(host: str) -> int:
    """Allocate a port for a leader rank's HTTP server.

    Uses the same `get_free_port_in_ci` helper as other disagg tests so that
    CI's `CONTAINER_PORT_START` / `CONTAINER_PORT_NUM` allocation is honored.
    We can only verify the port is free locally (on pytest's node); when the
    leader rank for an instance is on a different node we accept the small
    risk of collision in the CI-allocated range.
    """
    from defs.common import get_free_port_in_ci

    return get_free_port_in_ci()


def _gpus_per_instance(server_cfg: dict) -> int:
    return (
        server_cfg.get("tensor_parallel_size", 1)
        * server_cfg.get("pipeline_parallel_size", 1)
        * server_cfg.get("context_parallel_size", 1)
    )


def _required_worker_ranks(base_config: dict) -> int:
    ctx_servers = base_config.get("context_servers", {}) or {}
    gen_servers = base_config.get("generation_servers", {}) or {}
    return ctx_servers.get("num_instances", 0) * _gpus_per_instance(ctx_servers) + gen_servers.get(
        "num_instances", 0
    ) * _gpus_per_instance(gen_servers)


def _candidate_shared_roots(cwd: str | None) -> list[str]:
    candidates = [
        os.environ.get("TRTLLM_DISAGG_SHARED_TMPDIR"),
        os.environ.get("jobWorkspace"),
        os.environ.get("JOB_WORKSPACE"),
    ]
    llm_models_root = os.environ.get("LLM_MODELS_ROOT")
    if llm_models_root:
        candidates.append(os.path.dirname(llm_models_root))
    if cwd:
        candidates.append(cwd)

    deduped = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _create_work_dir(cwd: str | None) -> str:
    """Create a work dir visible to MPI ranks when CI provides one."""
    errors = []
    for shared_root in _candidate_shared_roots(cwd):
        if not os.path.isdir(shared_root):
            errors.append(f"{shared_root}: not a directory")
            continue
        try:
            work_dir = tempfile.mkdtemp(prefix="multi_disagg_", dir=shared_root)
            probe_path = os.path.join(work_dir, ".write_probe")
            with open(probe_path, "w") as f:
                f.write("ok")
            os.remove(probe_path)
            print(f"Using disagg shared work dir: {work_dir}")
            return work_dir
        except OSError as e:
            errors.append(f"{shared_root}: {e}")

    # Single-node local runs can still use node-local /tmp. Multi-node CI is
    # expected to provide jobWorkspace or TRTLLM_DISAGG_SHARED_TMPDIR.
    work_dir = tempfile.mkdtemp(prefix="multi_disagg_")
    print(f"Falling back to node-local disagg work dir {work_dir}; shared root attempts: {errors}")
    return work_dir


class _MpiWorkerStopProcess:
    """Process-like shim used by disagg_test_utils.terminate()."""

    pid = "mpi-disagg-workers"

    def __init__(self, stop_file: str):
        self._stop_file = stop_file
        self._stopped = False

    def poll(self):
        return 0 if self._stopped else None

    def kill(self):
        os.makedirs(os.path.dirname(self._stop_file), exist_ok=True)
        with open(self._stop_file, "w") as f:
            f.write("stop\n")
        self._stopped = True

    def wait(self, timeout=None):
        # Give leader ranks a chance to observe the stop file before pytest
        # removes the temporary work dir.
        time.sleep(min(timeout or 5, 5))
        return 0


# ---------------------------------------------------------------------------
# Combined config writer + setup entry point
# ---------------------------------------------------------------------------
def _build_combined_config(
    base_config: dict, work_dir: str, rank_to_host: list[str]
) -> tuple[dict, list[dict]]:
    """Return (combined_config_dict, [per_instance_meta]).

    Each per-instance meta dict has keys: type, url, hostname, port, leader_rank.
    """
    ctx_servers = base_config.get("context_servers", {}) or {}
    gen_servers = base_config.get("generation_servers", {}) or {}

    num_ctx = ctx_servers.get("num_instances", 0)
    num_gen = gen_servers.get("num_instances", 0)
    gpus_per_ctx = _gpus_per_instance(ctx_servers)
    gpus_per_gen = _gpus_per_instance(gen_servers)

    # Plan rank ranges: ctx instances first, then gen instances. This matches
    # `extract_disagg_cfg`'s ordering, which `split_world_comm` consumes.
    next_rank = 0
    instances: list[dict] = []
    ctx_urls: list[str] = []
    gen_urls: list[str] = []

    for _ in range(num_ctx):
        leader_rank = next_rank
        host = rank_to_host[leader_rank]
        port = _allocate_free_port_on_host(host)
        url = f"{host}:{port}"
        instances.append(
            dict(type="ctx", url=url, hostname=host, port=port, leader_rank=leader_rank)
        )
        ctx_urls.append(url)
        next_rank += gpus_per_ctx

    for _ in range(num_gen):
        leader_rank = next_rank
        host = rank_to_host[leader_rank]
        port = _allocate_free_port_on_host(host)
        url = f"{host}:{port}"
        instances.append(
            dict(type="gen", url=url, hostname=host, port=port, leader_rank=leader_rank)
        )
        gen_urls.append(url)
        next_rank += gpus_per_gen

    # Combined config = same shape as input but with explicit URLs so
    # disaggregated_mpi_worker can read them per instance via
    # extract_ctx_gen_cfgs().
    combined = dict(base_config)
    if "context_servers" in combined:
        combined["context_servers"] = dict(ctx_servers)
        combined["context_servers"]["urls"] = ctx_urls
    if "generation_servers" in combined:
        combined["generation_servers"] = dict(gen_servers)
        combined["generation_servers"]["urls"] = gen_urls

    return combined, instances


def setup_multi_instance_disagg_cluster(
    config_file: str,
    model_name: str | None = None,
    env: dict | None = None,
    cwd: str | None = None,
    server_start_timeout: int = 1200,
):
    """Set up a multi-instance multi-node disagg cluster.

    Returns the same tuple shape as ``setup_disagg_cluster`` for drop-in
    compatibility:
        (config, ctx_workers, gen_workers, disagg_server, server_port, work_dir)

    ``ctx_workers`` carries a process-like stop shim because the real workers
    run as in-MPI-pool tasks rather than local subprocesses.
    """
    # Lazy imports to avoid breaking single-node tests on import.
    from tensorrt_llm.executor.utils import create_mpi_comm_session

    with open(config_file, "r") as f:
        base_config = yaml.safe_load(f)

    mpi_world_size = int(os.environ.get("tllm_mpi_size", "1"))
    required_ranks = _required_worker_ranks(base_config)
    if required_ranks != mpi_world_size:
        raise RuntimeError(
            f"Disagg config requires {required_ranks} MPI ranks, "
            f"but tllm_mpi_size is {mpi_world_size}"
        )

    # Step 0: create the shared mgmn proxy client up-front so we can use it
    # for both the hostname gather and the long-running disagg task.
    mpi_session = create_mpi_comm_session(n_workers=mpi_world_size)

    rank_to_host = _resolve_rank_to_hostname(mpi_session, mpi_world_size)

    combined, instances = _build_combined_config(
        base_config, work_dir="", rank_to_host=rank_to_host
    )

    # Each MPI rank — including ranks on remote nodes — rereads the combined
    # config after the task is dispatched. ``/tmp`` is node-local in CI, so use
    # a known shared job directory when one is available.
    work_dir = _create_work_dir(cwd)
    combined_cfg_path = os.path.join(work_dir, "combined.yaml")
    worker_stop_file = os.path.join(work_dir, "stop_workers")
    with open(combined_cfg_path, "w") as f:
        yaml.dump(combined, f, sort_keys=False)

    # Frontend config: bind on pytest's loopback so the rest of the test
    # infrastructure (which polls localhost) keeps working unchanged.
    frontend_host = "localhost"
    frontend_port = _allocate_free_port_on_host(frontend_host)
    frontend_cfg = dict(base_config)
    frontend_cfg["hostname"] = frontend_host
    frontend_cfg["port"] = frontend_port
    # Strip the per-worker LLM params; the frontend only needs routing info.
    if "context_servers" in frontend_cfg:
        ctx_cfg = combined.get("context_servers", {})
        frontend_cfg["context_servers"] = {
            "num_instances": ctx_cfg.get("num_instances", 0),
            "urls": ctx_cfg.get("urls", []),
            "router": ctx_cfg.get("router", {}),
        }
    if "generation_servers" in frontend_cfg:
        gen_cfg = combined.get("generation_servers", {})
        frontend_cfg["generation_servers"] = {
            "num_instances": gen_cfg.get("num_instances", 0),
            "urls": gen_cfg.get("urls", []),
            "router": gen_cfg.get("router", {}),
        }
    frontend_cfg_path = os.path.join(work_dir, "frontend.yaml")
    with open(frontend_cfg_path, "w") as f:
        yaml.dump(frontend_cfg, f, sort_keys=False)

    # --- 1) Dispatch the disaggregated_mpi_worker task to all MPI ranks via
    # the shared mgmn proxy. The function lives in tensorrt_llm.commands.serve
    # so workers can unpickle it without the test dir on PYTHONPATH.
    # `submit` is fire-and-forget on RemoteMpiCommSessionClient; the task
    # blocks each rank for the lifetime of the disagg cluster.
    from tensorrt_llm.commands.serve import disaggregated_mpi_worker_main

    mpi_session.submit(disaggregated_mpi_worker_main, combined_cfg_path, "info", worker_stop_file)

    # --- 2) Launch the disagg frontend on the pytest process's node.
    sub_env = dict(env) if env else None
    # Drop MPI-related env so the frontend python process doesn't try to
    # MPI_Init into the shared world.
    if sub_env is None:
        sub_env = dict(os.environ)
    for k in list(sub_env.keys()):
        if k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_PROCID")):
            sub_env.pop(k, None)
    # `--server_start_timeout` is the time the frontend itself will wait for
    # ctx/gen workers to come up before its own Application startup fails. We
    # pin it to the caller's server_start_timeout so model-loading time
    # (~30 min for DSv4-Pro across 24 GPUs) does not exceed the frontend's
    # internal default (~60s) and abort.
    frontend_proc = subprocess.Popen(
        [
            "trtllm-serve",
            "disaggregated",
            "-c",
            frontend_cfg_path,
            "--server_start_timeout",
            str(server_start_timeout),
            "-r",
            "360000",
        ],
        env=sub_env,
        cwd=cwd,
    )

    # --- 3) Wait for frontend readiness (it polls workers via /cluster_info).
    try:
        asyncio.run(wait_for_disagg_server_ready(frontend_port, timeout=server_start_timeout))
    except Exception:
        try:
            frontend_proc.terminate()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    disagg_server = ProcessWrapper(frontend_proc, port=frontend_port)

    # Make a config dict shaped like the existing setup_disagg_cluster output.
    config_for_return = dict(base_config)
    config_for_return["hostname"] = frontend_host
    config_for_return["port"] = frontend_port

    # The workers live inside the shared MPI pool. Return a process-like shim
    # so the normal test cleanup path can ask them to exit before pytest
    # removes the shared config directory.
    worker_stopper = ProcessWrapper(_MpiWorkerStopProcess(worker_stop_file))
    return (config_for_return, [worker_stopper], [], disagg_server, frontend_port, work_dir)


def should_use_multi_instance_path(base_config: dict) -> bool:
    """Return True iff this config needs the MPI-worker helper."""
    if int(os.environ.get("tllm_mpi_size", "1")) <= 1:
        # No shared mgmn pool — old path handles single-node multi-instance
        # by giving each subprocess its own CUDA_VISIBLE_DEVICES.
        return False
    return _required_worker_ranks(base_config) > 1
