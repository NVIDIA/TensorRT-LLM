# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a checkpoint from object storage (S3, GCS or Azure Blob) with Run:ai Model Streamer.

The weight tensors are never copied to local disk. They are streamed from the
bucket into the model by ``checkpoint_io_policy="model_streamer"``. Config,
tokenizer and remote-code files are different: Hugging Face loaders read them
from a local directory, so `pull_metadata()` copies every file except the
weights into a node-local cache directory. Config and tokenizer loading use that
directory, and weight loading keeps the original URI.

The cache directory is named after a hash of the URI, so every process and every
node derives the same path. `pull_metadata()` uses the Model Streamer
``ObjectStorageModel``, which writes a completion marker (``.runai_complete``)
after a successful download. A later call finds the marker and downloads
nothing. The cache is therefore not refreshed when the bucket changes. Set
``TRTLLM_OBJECT_STORAGE_FORCE_PULL=1`` to download again, or delete the cache
directory.

Credentials come from the environment, as for Model Streamer itself, for example
``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY`` and ``AWS_ENDPOINT_URL`` for S3.
"""

import fcntl
import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from tensorrt_llm.logger import logger

OBJECT_STORAGE_SCHEMES = ("s3://", "gs://", "az://")

# Set to "1" to download the metadata again, even when the cache is complete.
FORCE_PULL_ENV = "TRTLLM_OBJECT_STORAGE_FORCE_PULL"
# Identifies one launch that forces a download. Every process of the launch
# shares it, so only the first one downloads and the others find the cache current.
PULL_TOKEN_ENV = "TRTLLM_OBJECT_STORAGE_PULL_TOKEN"
# Set in a process that serves requests for an already running executor.
_ATTACHED_FRONTEND_ENV = "TLLM_EXECUTOR_ATTACH_INFO"

_PULL_TOKEN_FILE = ".trtllm_pull_token"

# Everything except the weights. Model Streamer reads the weights itself. The list also
# names other formats that hold a model, so that a bucket that mirrors a whole repository
# does not fill the local disk with a copy of the model in a format that is not used.
_WEIGHT_PATTERNS = [
    "*.pt",
    "*.safetensors",
    "*.bin",
    "*.tensors",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "*.h5",
    "*.msgpack",
    "*.ckpt",
    "*.npz",
    "*.ot",
]

# File systems that are shared between nodes. The lock that makes concurrent
# downloads safe is only reliable on one host.
_NETWORK_FILESYSTEMS = (
    "nfs",
    "nfs4",
    "lustre",
    "gpfs",
    "cifs",
    "smbfs",
    "beegfs",
    "ceph",
    "glusterfs",
)


def is_object_storage_uri(path: Any) -> bool:
    """Whether ``path`` is an ``s3://``, ``gs://`` or ``az://`` URI.

    A `pathlib.Path` is never an object-storage URI: it collapses ``s3://`` to
    ``s3:/``, so only a `str` can carry one.
    """
    return isinstance(path, str) and path.lower().startswith(OBJECT_STORAGE_SCHEMES)


def get_cache_dir(uri: str) -> Path:
    """Return the node-local directory that holds the metadata files of ``uri``.

    The directory is under ``$TRTLLM_CACHE_DIR`` when that is set, otherwise
    under ``~/.cache/tensorrt_llm``.
    """
    root = os.environ.get("TRTLLM_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "tensorrt_llm"
    return base / "object_storage" / hashlib.sha256(uri.encode()).hexdigest()[:16]


def get_force_pull_token() -> Optional[str]:
    """Return the token of this launch's forced download, or None if not forced.

    The first call in a forced launch creates the token and exports it, so child
    processes that inherit the environment use the same one. A process attached
    to an executor that is already running never forces a download: the launcher
    owns the cache.
    """
    if os.environ.get(FORCE_PULL_ENV) != "1" or os.environ.get(_ATTACHED_FRONTEND_ENV) is not None:
        return None
    token = os.environ.get(PULL_TOKEN_ENV)
    if not token:
        token = uuid.uuid4().hex
        os.environ[PULL_TOKEN_ENV] = token
    return token


def pull_metadata(uri: str, force_token: Optional[str] = None) -> Path:
    """Copy the non-weight files of ``uri`` to the local cache and return its path.

    Does nothing when the cache is already complete. Safe to call from several
    processes on one node: the first downloads and the others wait.

    Args:
        uri: Object-storage prefix of the checkpoint.
        force_token: Token from `get_force_pull_token()`. When it differs from the
            token recorded in the cache, the cache is deleted and downloaded
            again. None keeps a complete cache.

    Raises:
        ImportError: If the installed runai-model-streamer has no
            ``ObjectStorageModel``.
        ValueError: If ``uri`` is not an object-storage URI.
    """
    if not is_object_storage_uri(uri):
        raise ValueError(f"Not an object storage URI: {uri!r}")
    cache_dir = get_cache_dir(uri)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    _warn_if_network_filesystem(cache_dir.parent)

    if force_token is None:
        _pull(uri, cache_dir)
        return cache_dir

    # One lock for the whole forced download. Without it, a second process that
    # also holds this token could delete the directory under the first.
    with open(f"{cache_dir}.force.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        token_file = cache_dir / _PULL_TOKEN_FILE
        if not token_file.is_file() or token_file.read_text() != force_token:
            logger.info(f"Downloading the metadata of {uri} again ({FORCE_PULL_ENV}=1)")
            _remove_cache(cache_dir)
            _pull(uri, cache_dir)
            token_file.write_text(force_token)
    return cache_dir


def _remove_cache(cache_dir: Path) -> None:
    """Delete the cache directory once no Model Streamer process works in it.

    ``ObjectStorageModel`` holds a shared lock on ``<cache_dir>.lock`` while it uses the
    directory, so the exclusive lock here waits for those processes. The lock is released
    before the next download: ``ObjectStorageModel`` takes the same lock itself, and a lock
    held by the caller would block it for good, in this process too.
    """
    with open(f"{cache_dir}.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _pull(uri: str, cache_dir: Path) -> None:
    try:
        from runai_model_streamer import ObjectStorageModel
    except ImportError as error:
        raise ImportError(
            "Loading a model from object storage requires a runai-model-streamer "
            "version that provides ObjectStorageModel"
        ) from error
    with ObjectStorageModel(model_path=uri, dst=str(cache_dir)) as storage:
        storage.pull_files(ignore_pattern=_WEIGHT_PATTERNS)


def _warn_if_network_filesystem(path: Path) -> None:
    """Warn when ``path`` is on a file system that nodes share.

    Several nodes would then download into one directory, and the lock that
    guards a download works on one host only.
    """
    try:
        with open("/proc/mounts") as mounts:
            entries = [line.split() for line in mounts]
    except OSError:
        return
    resolved = str(path.resolve())
    best = ("", "")
    for entry in entries:
        if len(entry) < 3:
            continue
        mount_point, fs_type = entry[1], entry[2]
        within = resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/")
        if within and len(mount_point) > len(best[0]):
            best = (mount_point, fs_type)
    if best[1].startswith(_NETWORK_FILESYSTEMS):
        logger.warning(
            f"The object storage metadata cache {path} is on a network file system "
            f"({best[1]}). Nodes that share it can download into the same directory "
            "at once. Set TRTLLM_CACHE_DIR to a node-local directory."
        )
