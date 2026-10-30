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
"""Tests for sharing FlashInfer runtime cubins across isolated MPI ranks."""

import hashlib
import sys
import threading
import types
from pathlib import Path

import pytest

from tensorrt_llm._torch import flashinfer_cubin_pool as pool

pytestmark = pytest.mark.cpu_only

_FMHA = "deadbeef/fmha/trtllm-gen/"
# FlashInfer joins the artifact directory and the kernel name with a doubled slash.
_CUBIN = _FMHA + "/fmhaSm100fKernel_Test.cubin"
_HEADER = _FMHA + "/include/flashInferMetaInfo.h"
_GEMM_CUBIN = "deadbeef/gemm-0000000-0000000/Gemm_Test.cubin"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Rank:
    """One rank's view of FlashInfer: a private cubin dir and a recording downloader."""

    def __init__(self, workspace_root: Path, rank: int, downloads: list):
        self.workspace = workspace_root / f"rank-{rank}"
        self.cubin_dir = self.workspace / ".cache" / "flashinfer" / "cubins"
        self.rank = rank
        self.downloads = downloads
        self.env = {
            "TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1",
            "FLASHINFER_WORKSPACE_BASE": str(self.workspace),
        }

    def download(self, file_name: str, sha256: str, session=None) -> bytes:
        self.downloads.append((self.rank, file_name))
        data = file_name.encode()
        path = self.cubin_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def pooled(self):
        shared = pool.resolve_shared_cubin_dir(self.env)
        assert shared is not None
        return pool.make_pooled_get_artifact(self.download, self.cubin_dir, shared, _FMHA)


@pytest.fixture
def ranks(tmp_path: Path):
    downloads: list = []
    root = tmp_path / "flashinfer"
    return root, downloads, [_Rank(root, rank, downloads) for rank in range(4)]


def test_resolve_shared_dir_sits_beside_rank_slots(tmp_path: Path) -> None:
    env = {
        "TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1",
        "FLASHINFER_WORKSPACE_BASE": str(tmp_path / "rank-3"),
    }
    assert pool.resolve_shared_cubin_dir(env) == tmp_path / "cubins"


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({"FLASHINFER_WORKSPACE_BASE": "/x/rank-0"}, id="unmanaged-workspace"),
        pytest.param({"TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1"}, id="no-workspace"),
        pytest.param(
            {
                "TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1",
                "FLASHINFER_WORKSPACE_BASE": "/tmp/trtllm-flashinfer-rank-0.abcd1234",
            },
            id="temporary-fallback-workspace",
        ),
        pytest.param(
            {
                "TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1",
                "FLASHINFER_WORKSPACE_BASE": "/x/rank-0",
                "FLASHINFER_CUBIN_DIR": "/shared/cubins",
            },
            id="explicit-cubin-dir",
        ),
        pytest.param(
            {
                "TRTLLM_FLASHINFER_WORKSPACE_MANAGED": "1",
                "FLASHINFER_WORKSPACE_BASE": "/x/rank-0",
                "TRTLLM_FLASHINFER_SHARED_CUBINS": "0",
            },
            id="disabled",
        ),
    ],
)
def test_resolve_shared_dir_is_off_outside_managed_rank_slots(env: dict) -> None:
    assert pool.resolve_shared_cubin_dir(env) is None


def test_cubin_downloaded_once_is_reused_by_peer_ranks(ranks) -> None:
    root, downloads, (r0, r1, r2, _) = ranks
    data = _CUBIN.encode()

    assert r0.pooled()(_CUBIN, _sha(data)) == data
    assert r1.pooled()(_CUBIN, _sha(data)) == data
    assert r2.pooled()(_CUBIN, _sha(data)) == data

    assert downloads == [(0, _CUBIN)]
    assert (root / "cubins" / _CUBIN).read_bytes() == data
    # Peers read the pooled cubin in place; nothing is written to their caches.
    assert not (r1.cubin_dir / _CUBIN).exists()
    assert not (r2.cubin_dir / _CUBIN).exists()


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param(_HEADER, id="fmha-header"),
        pytest.param(_FMHA + "/checksums.txt", id="fmha-checksums"),
        pytest.param(_FMHA + "/include/Test.cubin", id="fmha-subdirectory"),
        pytest.param(_GEMM_CUBIN, id="other-artifact-cubin"),
    ],
)
def test_only_fmha_kernel_cubins_are_shared(ranks, file_name: str) -> None:
    root, downloads, (r0, r1, _, _) = ranks
    data = file_name.encode()

    assert r0.pooled()(file_name, _sha(data)) == data
    assert r1.pooled()(file_name, _sha(data)) == data

    assert downloads == [(0, file_name), (1, file_name)]
    assert not (root / "cubins").exists()


def test_corrupt_pool_entry_is_ignored_and_replaced(ranks) -> None:
    root, downloads, (r0, r1, _, _) = ranks
    data = _CUBIN.encode()
    shared = root / "cubins" / _CUBIN
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"truncated")

    assert r0.pooled()(_CUBIN, _sha(data)) == data
    assert downloads == [(0, _CUBIN)]
    assert shared.read_bytes() == data

    assert r1.pooled()(_CUBIN, _sha(data)) == data
    assert downloads == [(0, _CUBIN)]


def test_failed_download_is_not_published(ranks) -> None:
    root, _, (r0, _, _, _) = ranks

    def failing_download(file_name: str, sha256: str, session=None) -> bytes:
        return b""

    shared = pool.resolve_shared_cubin_dir(r0.env)
    pooled = pool.make_pooled_get_artifact(failing_download, r0.cubin_dir, shared, _FMHA)
    assert pooled(_CUBIN, _sha(_CUBIN.encode())) == b""
    assert not (root / "cubins" / _CUBIN).exists()


def test_concurrent_ranks_never_read_a_partial_cubin(tmp_path: Path) -> None:
    downloads: list = []
    root = tmp_path / "flashinfer"
    names = [_CUBIN.replace("Test", f"Test{i}") for i in range(4)]
    errors: list = []

    def run(rank: int) -> None:
        pooled = _Rank(root, rank, downloads).pooled()
        try:
            for _ in range(20):
                for name in names:
                    assert pooled(name, _sha(name.encode())) == name.encode()
        except AssertionError as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert not list((root / "cubins").rglob("*.tmp"))


def test_install_wraps_flashinfer_loader_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "rank-0"
    monkeypatch.setenv("TRTLLM_FLASHINFER_WORKSPACE_MANAGED", "1")
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(workspace))
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_FLASHINFER_SHARED_CUBINS", raising=False)

    def get_artifact(file_name: str, sha256: str, session=None) -> bytes:
        return b""

    cubin_loader = types.SimpleNamespace(
        get_artifact=get_artifact,
        get_cubin=get_artifact,
        FLASHINFER_CUBIN_DIR=workspace / ".cache" / "flashinfer" / "cubins",
    )
    artifacts = types.SimpleNamespace(ArtifactPath=types.SimpleNamespace(TRTLLM_GEN_FMHA=_FMHA))
    jit = types.ModuleType("flashinfer.jit")
    jit.cubin_loader = cubin_loader
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.jit = jit
    flashinfer.artifacts = artifacts
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", jit)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cubin_loader", cubin_loader)
    monkeypatch.setitem(sys.modules, "flashinfer.artifacts", artifacts)

    assert pool.install_shared_cubin_pool()
    wrapped = cubin_loader.get_artifact
    assert wrapped is not get_artifact
    assert cubin_loader.get_cubin is wrapped

    assert pool.install_shared_cubin_pool()
    assert cubin_loader.get_artifact is wrapped
