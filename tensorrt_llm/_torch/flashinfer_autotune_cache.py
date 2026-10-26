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

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.logger import logger

CACHE_DIR_ENV = "TLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
FROZEN_ENV = "TLLM_FLASHINFER_AUTOTUNE_FROZEN"

# FlashInfer keeps a loaded file's configs after the context exits, so a frozen process reads the
# file once; later calls pass no cache and see the same table.
_STATE = {"loaded": False, "logged": False, "bundled": None}


def use_bundled_file(path: Optional[str]) -> None:
    """Serve this rank's FlashInfer configs from ``path`` (load only) when no cache directory is configured."""
    _STATE["bundled"] = path


def cache_file() -> Optional[str]:
    """This rank's cache file: the configured directory's, else the bundled file, else None."""
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return _STATE["bundled"]
    return os.path.join(cache_dir, f"rank{mpi_rank()}.json")


def frozen() -> bool:
    """Whether the cache file is only loaded (never profiled into)."""
    if not os.environ.get(CACHE_DIR_ENV):
        return True
    return os.environ.get(FROZEN_ENV, "0") == "1"


def autotune(tune_mode: bool = True, cache: Optional[str] = None, *args, **kwargs):
    """``flashinfer.autotune`` with this rank's cache file filled in when none is given."""
    from flashinfer import autotune as _autotune

    path = cache_file() if cache is None else None
    if path is not None:
        if frozen():
            tune_mode = False
            if _STATE["loaded"]:
                return _autotune(tune_mode, None, *args, **kwargs)
            _STATE["loaded"] = True
        cache = path
        if not _STATE["logged"]:
            _STATE["logged"] = True
            logger.info(
                f"FlashInfer autotune cache: rank={mpi_rank()} file={path} "
                f"exists={os.path.isfile(path)} frozen={frozen()}"
            )
    return _autotune(tune_mode, cache, *args, **kwargs)
