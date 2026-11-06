# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host-resident PLE n-gram table (`weights._host_resident_empty`): a mapping of its own,
advised `MADV_HUGEPAGE` before the registration faults it in, and still allocated and
registered, with a warning, on a kernel that refuses the advice."""

import errno
import mmap

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.weights import (  # noqa: E501
    _host_resident_empty,
)
from tensorrt_llm.logger import logger

_ROWS = 1 << 16
_HEAD_DIM = 256


def _huge_page_advice_accepted() -> bool:
    """Whether this kernel takes MADV_HUGEPAGE (it has transparent huge pages)."""
    probe = mmap.mmap(-1, mmap.PAGESIZE)
    try:
        probe.madvise(mmap.MADV_HUGEPAGE)
    except OSError:
        return False
    finally:
        probe.close()
    return True


@pytest.mark.skipif(not _huge_page_advice_accepted(), reason="the kernel refuses MADV_HUGEPAGE")
def test_host_table_is_advised_transparent_huge_pages() -> None:
    # The host-resident table is a mapping of its own, advised MADV_HUGEPAGE
    # before it is faulted in, whatever torch's CPU allocator does with huge pages.
    table, host = _host_resident_empty(_ROWS, _HEAD_DIM, torch.float8_e4m3fn, torch.device("cuda"))
    flags, inside = None, False
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            fields = line.split()
            if line[:1] in "0123456789abcdef" and "-" in fields[0]:
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                inside = lo <= host.pointer < hi
            elif inside and line.startswith("VmFlags:"):
                flags = fields[1:]
                break
    assert flags is not None and "hg" in flags, flags
    assert table.shape == (_ROWS, _HEAD_DIM)


def test_host_table_loads_when_huge_page_advice_is_refused(monkeypatch) -> None:
    # A kernel without transparent huge pages refuses the advice: the table is
    # still allocated and registered, on base pages, with a warning.
    class _Refused(mmap.mmap):
        def madvise(self, *args):
            raise OSError(errno.EINVAL, "Invalid argument")

    warnings = []
    monkeypatch.setattr(mmap, "mmap", _Refused)
    monkeypatch.setattr(logger, "warning", warnings.append)
    table, host = _host_resident_empty(_ROWS, _HEAD_DIM, torch.float8_e4m3fn, torch.device("cuda"))
    assert isinstance(host._mapping, _Refused) and host.pointer
    assert table.shape == (_ROWS, _HEAD_DIM)
    # Other modules on the path (the NUMA placement) may warn too; count only this one.
    assert len([w for w in warnings if "MADV_HUGEPAGE" in str(w)]) == 1, warnings
