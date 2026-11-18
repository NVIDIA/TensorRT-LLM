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
"""Pause words, through which a node supervisor sees its ranks' hang detector pauses.

A rank's pause word is a 4-byte integer in /dev/shm. It holds the PID of the rank's process while
the rank's hang detector is paused, and 0 while it is armed. The rank writes it in the call that
pauses or arms the detector, so the word stays right also while the rank holds the GIL or is
stopped. Its node supervisor reads it and unlinks it once the deployment ends.
"""

import mmap
import os
import struct

from .signed_store import shared_secret_digest

_SHM_DIR = "/dev/shm"
# One native int, which holds any PID.
_FORMAT = "i"
_SIZE = struct.calcsize(_FORMAT)


def pause_word_path(rank: int) -> str:
    """Return the /dev/shm path of ``rank``'s pause word in this deployment."""
    digest = shared_secret_digest(b"hang detector pause word")[:16]
    return os.path.join(_SHM_DIR, f"trtllm_pause_{digest}_rank{rank}")


def open_pause_word(path: str) -> memoryview:
    """Create or open the pause word at ``path``, set it to 0, and return it.

    A replacement opens its predecessor's word, so the reset drops the PID that the predecessor
    left in it.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, _SIZE)
        mapping = mmap.mmap(fd, _SIZE)
    finally:
        os.close(fd)
    word = memoryview(mapping).cast(_FORMAT)
    word[0] = 0
    return word


def read_pause_word(path: str) -> int:
    """Return the PID that the pause word at ``path`` holds, or 0 if it is armed or absent."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return 0
    try:
        data = os.pread(fd, _SIZE, 0)
    finally:
        os.close(fd)
    return memoryview(data).cast(_FORMAT)[0] if len(data) == _SIZE else 0
