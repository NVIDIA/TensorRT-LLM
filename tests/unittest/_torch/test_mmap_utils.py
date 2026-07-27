# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

from tensorrt_llm._torch import mmap_utils


class _FakeCpuTensor:
    class _Device:
        type = "cpu"

    device = _Device()

    def __init__(self, address: int, nbytes: int, *, contiguous: bool = True):
        self._address = address
        self._nbytes = nbytes
        self._contiguous = contiguous

    def data_ptr(self) -> int:
        return self._address

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def is_contiguous(self) -> bool:
        return self._contiguous


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ("1000-4000 rw-p 00000000 00:00 1 /models/model.safetensors\n", True),
        ("1000-4000 rw-s 00000000 00:00 1 /dev/shm/mpi-window\n", False),
        ("1000-4000 rw-p 00000000 00:00 0\n", False),
        ("1000-2800 rw-p 00000000 00:00 1 /models/model.safetensors\n", False),
    ],
)
def test_reloadable_file_backed_tensor_check(monkeypatch, mapping, expected):
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(mapping))

    tensor = _FakeCpuTensor(0x2000, 0x1000)

    assert mmap_utils.is_reloadable_file_backed_tensor(tensor) is expected


def test_advise_tensor_pageout_skips_non_file_storage(monkeypatch):
    tensor = _FakeCpuTensor(0x2000, 0x1000)
    monkeypatch.setattr(mmap_utils, "is_reloadable_file_backed_tensor", lambda _: False)

    def fail_madvise(*args, **kwargs):
        raise AssertionError("anonymous stream storage must not be discarded")

    monkeypatch.setattr(mmap_utils, "madvise_range", fail_madvise)

    mmap_utils.advise_tensor_pageout(tensor)
