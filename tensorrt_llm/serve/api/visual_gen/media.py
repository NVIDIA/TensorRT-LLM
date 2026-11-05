# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from datetime import datetime
from pathlib import Path


def _new_media_dir(root: Path) -> Path:
    """Create a directory under ``root`` stamped with the current time.

    ``mkdir`` without ``exist_ok`` is what makes this safe: the kernel either
    creates the directory or raises, so two servers starting in the same
    second take separate names instead of sharing one.
    """
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1


def _resolve_media_dir() -> Path:
    """Create and return the directory to store generated media in.

    ``TRTLLM_MEDIA_STORAGE_PATH`` names the directory outright, empty meaning
    unset. Otherwise it goes beside the working directory, and where that
    cannot be written it goes to a private temporary one: a shared ``/tmp``
    holds directories owned by other users, so the fallback takes a name
    nobody else can hold rather than a fixed one.
    """
    explicit = os.getenv("TRTLLM_MEDIA_STORAGE_PATH")
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    try:
        return _new_media_dir(Path.cwd() / "trtllm_generated")
    except OSError:
        # OSError rather than PermissionError: a read-only mount raises
        # EROFS, which is not one.
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        return Path(tempfile.mkdtemp(prefix=f"trtllm_generated-{stamp}-"))
