# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The launch's default directory for shared JIT builds (tensorrt_llm._jit_share): without
TLLM_JIT_SHARE_DIR, the processes of a launch that run on one node share their builds through a
directory on the node's local disk.

Nothing is compiled: the compiler is /bin/true.
"""

import os
import stat
import tempfile
import time

import pytest

from tensorrt_llm import _jit_share

pytestmark = pytest.mark.cpu_only

_LAUNCH_VARS = (
    "PMIX_NAMESPACE",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_STEP_TASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "TORCHELASTIC_RUN_ID",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "LOCAL_WORLD_SIZE",
    "MPI_LOCALNRANKS",
    "TRTLLM_DWDP_RANK",
    "TRTLLM_DWDP_MASTER_ADDR",
    "TRTLLM_DWDP_MASTER_PORT",
)
_INSTALL_VARS = (
    _jit_share.SHARE_DIR_ENV,
    _jit_share._DEFAULT_ENV,
    _jit_share._KERNEL_BUILDER_ENV,
    _jit_share._PROCESS_ENV,
    "DG_JIT_NVCC_COMPILER",
    "DG_JIT_CACHE_DIR",
)


@pytest.fixture
def local_disk(monkeypatch, tmp_path):
    """A process of no launch yet that has not installed, whose default directories go under
    the returned directory. The environment install() changes is restored after the test."""
    saved = dict(os.environ)
    for name in _LAUNCH_VARS + _INSTALL_VARS:
        os.environ.pop(name, None)
    root = tmp_path / "local"
    root.mkdir()
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(_jit_share, "_DEFAULT_ROOT", str(root))
    monkeypatch.setattr(_jit_share, "_installed", False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    os.environ[_jit_share._REAL_COMPILER_ENV] = "/bin/true"
    os.environ[_jit_share._SALT_ENV] = "test"
    yield root
    os.environ.clear()
    os.environ.update(saved)


def _launch(namespace="slurm.pmix.7.0", per_node="4"):
    os.environ["PMIX_NAMESPACE"] = namespace
    os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = per_node


def test_a_launch_of_several_processes_per_node_shares_a_directory_on_local_disk(local_disk):
    _launch()
    assert _jit_share.install() and _jit_share.enabled()
    share_dir = os.environ[_jit_share.SHARE_DIR_ENV]
    assert os.path.dirname(share_dir) == str(local_disk / f"tllm-jit-{os.getuid()}")
    assert os.environ[_jit_share._DEFAULT_ENV] == "1"
    assert stat.S_IMODE(os.stat(share_dir).st_mode) == 0o700
    assert os.path.exists(os.path.join(share_dir, "procs", str(os.getpid())))
    # DeepGEMM keeps its own disk cache, which outlives the launch.
    assert "DG_JIT_CACHE_DIR" not in os.environ


def test_the_processes_of_a_launch_choose_one_directory_and_other_launches_another(local_disk):
    _launch("prterun-node-1@1")
    first = _jit_share._default_share_dir()
    assert first is not None and _jit_share._default_share_dir() == first
    _launch("prterun-node-1@2")
    assert _jit_share._default_share_dir() != first


@pytest.mark.parametrize(
    "env",
    [
        {"PMIX_NAMESPACE": "slurm.pmix.7.0", "OMPI_COMM_WORLD_LOCAL_SIZE": "1"},
        {"OMPI_COMM_WORLD_LOCAL_SIZE": "4"},
        {
            "PMIX_NAMESPACE": "slurm.pmix.7.0",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "4",
            "TLLM_KERNEL_BUILDER": "0",
        },
        {
            "PMIX_NAMESPACE": "slurm.pmix.7.0",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "4",
            "TLLM_JIT_SHARE_DEFAULT": "0",
        },
    ],
    ids=["one_process_per_node", "no_launch", "kernel_builder_off", "default_off"],
)
def test_no_default_directory_without_a_launch_of_several_processes_per_node(local_disk, env):
    for name, value in env.items():
        os.environ[name] = value
    assert _jit_share._default_share_dir() is None
    assert not _jit_share.install()


def test_slurm_and_torchrun_launches_count_their_processes_per_node(local_disk):
    os.environ["SLURM_JOB_ID"] = "7"
    os.environ["SLURM_STEP_ID"] = "0"
    os.environ["SLURM_STEP_TASKS_PER_NODE"] = "4(x2)"
    assert _jit_share._default_share_dir() is not None
    os.environ["SLURM_STEP_TASKS_PER_NODE"] = "1,1"
    assert _jit_share._default_share_dir() is None
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_STEP_TASKS_PER_NODE"):
        del os.environ[name]
    os.environ["TORCHELASTIC_RUN_ID"] = "run"
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    assert _jit_share._default_share_dir() is not None


def test_the_workers_of_a_dwdp_rendezvous_share_a_directory(local_disk):
    # Context workers launched one trtllm-serve each, without the launcher's variables.
    os.environ["TRTLLM_DWDP_RANK"] = "3"
    os.environ["TRTLLM_DWDP_MASTER_ADDR"] = "node-0"
    os.environ["TRTLLM_DWDP_MASTER_PORT"] = "29611"
    first = _jit_share._default_share_dir()
    os.environ["TRTLLM_DWDP_RANK"] = "0"
    assert first is not None and _jit_share._default_share_dir() == first
    os.environ["TRTLLM_DWDP_MASTER_PORT"] = "29612"
    assert _jit_share._default_share_dir() != first


def test_a_root_other_users_can_write_to_is_not_used(local_disk):
    _launch()
    root = local_disk / f"tllm-jit-{os.getuid()}"
    root.mkdir()
    os.chmod(root, 0o777)
    assert _jit_share._default_share_dir() is None


def test_a_process_a_launch_process_starts_keeps_the_launch_directory(local_disk, monkeypatch):
    _launch()
    assert _jit_share.install()
    share_dir = os.environ[_jit_share.SHARE_DIR_ENV]
    # A BuildPool worker process: no launcher variables, the launch's directory inherited.
    for name in _LAUNCH_VARS:
        os.environ.pop(name, None)
    monkeypatch.setattr(_jit_share, "_installed", False)
    assert _jit_share.install()
    assert os.environ[_jit_share.SHARE_DIR_ENV] == share_dir
    assert "DG_JIT_CACHE_DIR" not in os.environ


def test_a_share_dir_the_user_sets_holds_deepgemm_disk_cache(local_disk, tmp_path):
    _launch()
    share = tmp_path / "share"
    os.environ[_jit_share.SHARE_DIR_ENV] = str(share)
    assert _jit_share.install()
    assert os.environ[_jit_share.SHARE_DIR_ENV] == str(share)
    assert os.environ["DG_JIT_CACHE_DIR"] == str(share / "deep_gemm")


def test_launch_directories_whose_processes_all_exited_are_removed(local_disk):
    root = local_disk / f"tllm-jit-{os.getuid()}"
    old = time.time() - 2 * _jit_share._PRUNE_AFTER_S
    dirs = {}
    for name, pid in (("ended", "999999999"), ("running", str(os.getpid())), ("new", "999999999")):
        procs = root / name / "procs"
        procs.mkdir(parents=True)
        (procs / pid).touch()
        dirs[name] = root / name
    for name in ("ended", "running"):
        os.utime(dirs[name], (old, old))
    _jit_share._prune_launch_dirs(str(root))
    assert not dirs["ended"].exists()
    assert dirs["running"].exists() and dirs["new"].exists()
