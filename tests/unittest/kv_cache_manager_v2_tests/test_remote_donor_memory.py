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
"""The remote-level donor makes all of its memory resident before it publishes it. Linux only, no GPU."""

import ctypes
import errno
import gc
import mmap
import os
import shutil
import sys
import tempfile
import unittest
import weakref
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import host_memory, remote_donor

_SIZE = 96 << 20


# Host-only (no GPU): runs in the CPU stages.
pytestmark = pytest.mark.cpu_only


def _resident_bytes(address: int, size: int) -> int:
    """Bytes of [address, address + size) in memory, page by page with mincore(2).

    /proc/self/smaps is not exact here: the kernel may merge the test mapping with a neighbouring anonymous one.
    """
    page = mmap.PAGESIZE
    pages = (size + page - 1) // page
    vec = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(v & 1 for v in vec) * page


@unittest.skipUnless(sys.platform.startswith("linux"), "uses mincore(2)")
class TestPopulate(unittest.TestCase):
    def _region(self):
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | remote_donor._MAP_NORESERVE
        region = mmap.mmap(-1, _SIZE, flags=flags)
        self.addCleanup(region.close)
        return region, ctypes.addressof(ctypes.c_char.from_buffer(region))

    def test_every_page_is_resident(self):
        region, address = self._region()
        self.assertEqual(_resident_bytes(address, _SIZE), 0)
        host_memory.populate(region, address, _SIZE)
        self.assertEqual(_resident_bytes(address, _SIZE), _SIZE)

    def test_fallback_without_madv_populate_write(self):
        region, address = self._region()
        self.assertEqual(_resident_bytes(address, _SIZE), 0)
        # A kernel without MADV_POPULATE_WRITE (EINVAL) takes the zero-fill path.
        with mock.patch.object(
            host_memory, "madvise_range", side_effect=OSError(errno.EINVAL, "EINVAL")
        ):
            host_memory.populate(region, address, _SIZE)
        self.assertEqual(_resident_bytes(address, _SIZE), _SIZE)


class _RecordingTransport(remote_donor.DonorTransport):
    """Publishes without memory and counts releases."""

    kind = "test"

    def __init__(self):
        self.releases = 0

    def start(self, donor):
        donor._publish([], ["segment 0 0"], "test")

    def release_unowned(self):
        self.releases += 1


class TestUnclosedDonor(unittest.TestCase):
    """A donor nobody closed frees its transport's memory too, not just its files: the process may live on."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="kvcm2_unclosed_")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def _started(self, transport):
        donor = remote_donor.RemoteMemDonor(self.directory, 1 << 20, transport, chunk_bytes=1 << 20)
        donor.start()
        self.assertTrue(os.path.exists(donor._path("ready")))
        return donor

    def test_a_collected_donor_releases_its_transport(self):
        transport = _RecordingTransport()
        donor = self._started(transport)
        ready, failed = donor._path("ready"), donor._path("failed")
        del donor
        gc.collect()
        self.assertEqual(transport.releases, 1)
        self.assertFalse(os.path.exists(ready))
        self.assertTrue(os.path.exists(failed))

    def test_a_closed_donor_releases_once(self):
        transport = _RecordingTransport()
        donor = self._started(transport)
        donor.close()
        del donor
        gc.collect()
        self.assertEqual(transport.releases, 1)

    def test_the_finalizer_does_not_keep_the_donor_alive(self):
        donor = self._started(_RecordingTransport())
        ref = weakref.ref(donor)
        del donor
        gc.collect()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
