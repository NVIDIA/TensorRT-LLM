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

import mmap
import os

import pytest

from tensorrt_llm._torch.mmap_utils import populate_file_pages

pytestmark = pytest.mark.cpu_only


def test_populate_file_pages_full_or_unsupported(tmp_path):
    # MADV_POPULATE_READ requires Linux >= 5.14: population either warms the
    # whole file (reporting every window, including the trailing partial one)
    # or signals "unsupported" by stopping at 0 bytes.
    payload = os.urandom(2 * mmap.PAGESIZE + 123)
    file = tmp_path / "blob.bin"
    file.write_bytes(payload)

    windows: list[int] = []
    populated = populate_file_pages(str(file), mmap.PAGESIZE, windows.append)

    assert populated in (0, len(payload))
    assert sum(windows) == populated
    if windows:
        assert max(windows) <= mmap.PAGESIZE


def test_populate_file_pages_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.touch()
    assert populate_file_pages(str(empty), mmap.PAGESIZE) == 0
    assert populate_file_pages(str(tmp_path / "missing.bin"), mmap.PAGESIZE) == 0
