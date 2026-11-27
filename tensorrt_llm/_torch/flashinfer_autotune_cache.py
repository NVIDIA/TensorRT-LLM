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
"""A per-rank file cache for FlashInfer's own autotuner (its CuTe DSL MXFP8 GEMM configs).

FlashInfer keeps the configs it profiles inside ``flashinfer.autotune()`` in a process-wide table, so
every server start profiles them again and may pick a different config per shape. With
``TLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`` set, every ``autotune(cache=None)`` this package opens reads
(and, while tuning, extends) ``<dir>/rank<R>.json`` instead. With
``TLLM_FLASHINFER_AUTOTUNE_FROZEN=1`` the file is only loaded, never profiled into: the configs of a
previous run are served as they are. Without the directory, a model may install a bundled file
(``use_bundled_file``), which is always only loaded. Neither set, ``autotune`` is FlashInfer's own.
"""

import os
from typing import Optional

from tensorrt_llm._utils import env_switch, mpi_rank
from tensorrt_llm.logger import logger

CACHE_DIR_ENV = "TLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
FROZEN_ENV = "TLLM_FLASHINFER_AUTOTUNE_FROZEN"

# FlashInfer keeps a loaded file's configs after the context exits, so a frozen process reads each
# file once ("loaded" holds the last file read); later calls for it pass no cache and see the same table.
# A tuning process ("loaded" holds the last file and its modification time) hands FlashInfer the file
# on every tuning call, so the configs it profiles are saved, and otherwise only when the file changed.
_STATE = {"loaded": False, "logged": False, "bundled": None}


def _mtime(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def use_bundled_file(path: Optional[str]) -> None:
    """Serve this rank's FlashInfer configs from ``path`` (load only) when no cache directory is configured."""
    _STATE["bundled"] = path


# This rank's file per cache directory, resolved once per process.
_RANK_FILES: dict = {}


def _flashinfer_autotune():
    # Looked up on every call (a module-table lookup), so this always enters the
    # ``flashinfer.autotune`` currently installed, as a direct call would.
    from flashinfer import autotune as fi_autotune

    return fi_autotune


def cache_file() -> Optional[str]:
    """This rank's cache file: the configured directory's, else the bundled file, else None."""
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return _STATE["bundled"]
    path = _RANK_FILES.get(cache_dir)
    if path is None:
        path = _RANK_FILES[cache_dir] = os.path.join(cache_dir, f"rank{mpi_rank()}.json")
    return path


def frozen() -> bool:
    """Whether the cache file is only loaded (never profiled into)."""
    if not os.environ.get(CACHE_DIR_ENV):
        return True
    return env_switch(FROZEN_ENV) is True


def autotune(*args, **kwargs):
    """``flashinfer.autotune`` with this rank's cache file filled in when none is given.

    Takes FlashInfer's arguments (``tune_mode=True, cache=None, ...``). Without a file to serve, or
    with an explicit ``cache``, FlashInfer's own ``autotune`` is called exactly as this one was.
    """
    _autotune = _flashinfer_autotune()
    cache = args[1] if len(args) > 1 else kwargs.get("cache")
    if cache is None and _STATE.get("bundled") is None and not os.environ.get(CACHE_DIR_ENV):
        # Nothing to serve: FlashInfer's own autotune, as called.
        return _autotune(*args, **kwargs)
    path = cache_file() if cache is None else None
    if path is None:
        if cache is not None:
            # FlashInfer replaces its file configs with the given cache's, so the next call
            # without one hands this rank's file over again.
            _STATE["loaded"] = False
        return _autotune(*args, **kwargs)
    tune_mode = args[0] if args else kwargs.pop("tune_mode", True)
    kwargs.pop("cache", None)
    args = args[2:]
    if frozen():
        tune_mode = False
        if _STATE["loaded"] == path:
            return _autotune(tune_mode, None, *args, **kwargs)
        _STATE["loaded"] = path
    else:
        version = (path, _mtime(path))
        if not tune_mode and _STATE["loaded"] == version:
            return _autotune(tune_mode, None, *args, **kwargs)
        _STATE["loaded"] = version
    if not _STATE["logged"]:
        _STATE["logged"] = True
        logger.info(
            f"FlashInfer autotune cache: rank={mpi_rank()} file={path} "
            f"exists={os.path.isfile(path)} frozen={frozen()}"
        )
    return _autotune(tune_mode, path, *args, **kwargs)
