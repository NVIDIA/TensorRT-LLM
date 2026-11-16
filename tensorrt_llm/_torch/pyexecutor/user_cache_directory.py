# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The user cache directory, made to exist before an engine compiles kernels at run time.

The run-time compiled attention kernels (the FMHA kernels the kernel generator builds with NVRTC) are
cached by name. When the generator does not find the user cache directory, it keeps the compiled
kernels in `/tmp/jitify2_cache`, a directory every process of the host shares. A kernel name does
not determine the kernel: a generation kernel is built for one query token per sequence in an
engine without draft tokens and for several in an engine with them, under one name. An engine with
draft tokens that starts after an engine without them then loads the single-token kernel from that
directory, writes NaN into its KV cache during warm-up and returns token id 0 from small-batch
decode steps for its whole life.

With the user cache directory in existence the generator keeps its kernels in the process, which is
what happens in an ordinary environment. The directory is missing when `XDG_CACHE_HOME` names a path
nobody created (test harnesses do that to isolate runs) or when the home directory has no `.cache`.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from tensorrt_llm.logger import logger


def user_cache_directory(environ: Mapping[str, str] = os.environ) -> Optional[Path]:
    """The directory the XDG base directory rules name for user caches.

    `$XDG_CACHE_HOME` when it is set and not empty, else `$HOME/.cache`; `None` when neither is known.
    """
    cache_home = environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home)
    home = environ.get("HOME")
    return Path(home) / ".cache" if home else None


def ensure_user_cache_directory(environ: Mapping[str, str] = os.environ) -> Optional[Path]:
    """Create the user cache directory if it does not exist, and return it.

    Returns `None`, with a warning, when the directory is not known or cannot be created: the kernel
    generator then shares compiled kernels with every other process of the host through
    `/tmp/jitify2_cache`.
    """
    path = user_cache_directory(environ)
    if path is None:
        logger.warning(
            "No user cache directory (XDG_CACHE_HOME and HOME are unset): run-time compiled kernels "
            "are shared with other processes through /tmp/jitify2_cache"
        )
        return None
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.warning(
            f"Cannot create the user cache directory {path} ({error}): run-time compiled kernels "
            "are shared with other processes through /tmp/jitify2_cache"
        )
        return None
    return path
