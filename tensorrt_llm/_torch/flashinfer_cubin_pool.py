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
"""Share FlashInfer runtime cubins between isolated MPI ranks.

``trtllm-llmapi-launch`` and ``MpiPoolSession`` give every rank its own
FlashInfer workspace, and with ``FLASHINFER_CUBIN_DIR`` unset FlashInfer keeps
its downloaded artifacts inside that workspace too. That isolation is required
for compiler inputs: FlashInfer downloads trtllm-gen export headers into the
same directory and compiles against them, and a header replaced by a peer
rank's download mid-compilation breaks the build.

Runtime cubins are different. A cubin is never handed to a compiler; FlashInfer
only reads its bytes and passes them to ``cuModuleLoadData``. Keeping them
per-rank means every rank downloads every kernel it uses separately, and a
TRTLLM-Gen FMHA kernel that warmup does not reach is downloaded on the serving
path the first time a request shape selects it. Under attention DP only the
rank that served a request has already fetched its kernels, so the other ranks
stall on the download in lockstep and the whole step waits for them.

This module installs a read-through pool for the TRTLLM-Gen FMHA kernel cubins
only, the ``*.cubin`` files directly under FlashInfer's ``TRTLLM_GEN_FMHA``
artifact directory. A lookup first tries the rank-local cache, then a pool
directory that sits next to the per-rank workspace slots, and only then
downloads. A rank that downloads a cubin publishes it to the pool with an
atomic rename, so peers see either a complete file or none. Peers read pooled
cubins in place, and every byte read from the pool is checked against the
artifact's SHA-256 before use. Headers, checksum files and every other
artifact take FlashInfer's own path unchanged and stay per-rank.
"""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..logger import logger

_CUBIN_SUFFIX = ".cubin"
_POOL_DIR_NAME = "cubins"
_MANAGED_ENV = "TRTLLM_FLASHINFER_WORKSPACE_MANAGED"
_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_CUBIN_DIR_ENV = "FLASHINFER_CUBIN_DIR"
_DISABLE_ENV = "TRTLLM_FLASHINFER_SHARED_CUBINS"
_INSTALLED_MARKER = "_trtllm_shared_cubin_pool"


def resolve_shared_cubin_dir(environ: Optional[dict] = None) -> Optional[Path]:
    """Return the shared cubin pool for this process, or None when not applicable.

    The pool exists only for TensorRT-LLM-managed per-rank workspaces: the
    workspace is a ``rank-N`` slot and the pool is the ``cubins`` directory
    beside it. A user-configured ``FLASHINFER_CUBIN_DIR`` is already shared and
    is left alone, as is an unmanaged ``FLASHINFER_WORKSPACE_BASE``.
    ``TRTLLM_FLASHINFER_SHARED_CUBINS=0`` disables the pool.

    Args:
        environ: Environment to read; defaults to ``os.environ``.
    """
    env = os.environ if environ is None else environ
    if env.get(_DISABLE_ENV, "1") == "0":
        return None
    if env.get(_CUBIN_DIR_ENV):
        return None
    if env.get(_MANAGED_ENV) != "1":
        return None
    workspace = env.get(_WORKSPACE_ENV)
    if not workspace:
        return None
    workspace_path = Path(workspace)
    if not workspace_path.name.startswith("rank-"):
        # Temporary fallback workspaces are process-unique and removed at exit,
        # so there is no stable directory to share with peers.
        return None
    return workspace_path.parent / _POOL_DIR_NAME


def _read_verified(path: Path, sha256: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError:
        return b""
    if hashlib.sha256(data).hexdigest() != sha256:
        return b""
    return data


def _is_pooled_artifact(file_name: str, pooled_prefix: str) -> bool:
    if not file_name.endswith(_CUBIN_SUFFIX):
        return False
    prefix = pooled_prefix.rstrip("/") + "/"
    if not file_name.startswith(prefix):
        return False
    # Only the kernel cubins directly under the artifact directory are pooled;
    # FlashInfer joins the directory and the kernel name with a doubled slash.
    return "/" not in file_name[len(prefix) :].lstrip("/")


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def make_pooled_get_artifact(
    get_artifact: Callable[..., bytes],
    local_cubin_dir: Path,
    shared_cubin_dir: Path,
    pooled_prefix: str,
) -> Callable[..., bytes]:
    """Wrap FlashInfer's ``get_artifact`` with a shared pool for runtime cubins.

    Args:
        get_artifact: FlashInfer's ``flashinfer.jit.cubin_loader.get_artifact``.
        local_cubin_dir: This rank's FlashInfer cubin directory.
        shared_cubin_dir: Pool directory shared by the job's ranks.
        pooled_prefix: Artifact directory whose kernel cubins are pooled.
    """

    def pooled_get_artifact(file_name: str, sha256: str, *args, **kwargs) -> bytes:
        if not _is_pooled_artifact(file_name, pooled_prefix):
            return get_artifact(file_name, sha256, *args, **kwargs)

        data = _read_verified(local_cubin_dir / file_name, sha256)
        if data:
            return data

        shared_path = shared_cubin_dir / file_name
        data = _read_verified(shared_path, sha256)
        if data:
            return data

        data = get_artifact(file_name, sha256, *args, **kwargs)
        if data:
            try:
                _write_atomically(shared_path, data)
            except OSError as error:
                logger.debug(f"Could not share FlashInfer cubin {file_name}: {error}")
        return data

    setattr(pooled_get_artifact, _INSTALLED_MARKER, True)
    return pooled_get_artifact


def install_shared_cubin_pool() -> bool:
    """Route FlashInfer's runtime cubin lookups through the shared pool.

    Returns True when the pool was installed. A no-op when the process does not
    use a TensorRT-LLM-managed per-rank workspace, when the pool is disabled, or
    when FlashInfer does not expose the expected loader.
    """
    shared_cubin_dir = resolve_shared_cubin_dir()
    if shared_cubin_dir is None:
        return False
    try:
        from flashinfer.jit import cubin_loader
    except ImportError:
        return False

    try:
        from flashinfer.artifacts import ArtifactPath
    except ImportError:
        return False

    get_artifact = getattr(cubin_loader, "get_artifact", None)
    local_cubin_dir = getattr(cubin_loader, "FLASHINFER_CUBIN_DIR", None)
    pooled_prefix = getattr(ArtifactPath, "TRTLLM_GEN_FMHA", None)
    if not callable(get_artifact) or local_cubin_dir is None or not pooled_prefix:
        return False
    if getattr(get_artifact, _INSTALLED_MARKER, False):
        return True
    local_cubin_dir = Path(local_cubin_dir)
    if local_cubin_dir.resolve() == shared_cubin_dir.resolve():
        return False

    pooled = make_pooled_get_artifact(
        get_artifact, local_cubin_dir, shared_cubin_dir, pooled_prefix
    )
    # The C++ cubin callback installed by setup_cubin_loader resolves
    # get_artifact from this module's globals on every call, so rebinding the
    # module attribute reaches the TRTLLM-Gen FMHA runtime loader.
    cubin_loader.get_artifact = pooled
    if getattr(cubin_loader, "get_cubin", None) is get_artifact:
        cubin_loader.get_cubin = pooled
    logger.info(f"FlashInfer runtime cubins are shared across ranks via {shared_cubin_dir}")
    return True
