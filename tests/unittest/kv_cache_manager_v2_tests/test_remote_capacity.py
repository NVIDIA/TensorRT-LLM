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
"""The remote level's capacity check, donor failure markers and fabric chunk placement.

No GPU, except TestFabricPlacement (the fabric transport) and TestFailedStartReleaseCudaState.

check_remote_capacity waits until the expected donors have published (donor.<name>.ready + .meta) or failed
(donor.<name>.failed), then compares what they offer with what the clients claim. It must fail as soon as the
last donor reports with a shortfall, pass as soon as the offer covers the need, and name failed and short donors.
"""

import contextlib
import errno
import gc
import mmap
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import host_memory, remote_donor
from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import (
    FabricDonorTransport,
    check_remote_capacity,
    make_donor,
    scan_donors,
)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from remote_test_utils import fabric_available, gpu_count
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

_GIB = 1 << 30


def _publish(
    directory, name, gib, chunk_gib=1, short_gib=0, host="hostA", extra=(), transport="fabric"
):
    lines = [
        "version 1",
        f"name {name}",
        f"host {host}",
        "pid 1",
        f"chunk_bytes {chunk_gib * _GIB}",
        f"transport {transport}",
        "numa 0",
        f"segment 0 {gib * _GIB}",
        *extra,
    ]
    if short_gib:
        lines.append(f"short_bytes {short_gib * _GIB}")
    with open(os.path.join(directory, f"donor.{name}.meta"), "w") as f:
        f.write("\n".join(lines) + "\n")
    open(os.path.join(directory, f"donor.{name}.ready"), "w").close()


@pytest.mark.cpu_only
class TestCapacityCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_capacity_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_scan_counts_whole_chunks_and_failures(self):
        _publish(self.dir, "a", 160)
        _publish(self.dir, "b", 100, short_gib=60)
        donor = make_donor(self.dir, 160 * _GIB, chunk_bytes=_GIB, name="c")
        donor.publish_failure(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
        open(os.path.join(self.dir, "claim.a.0"), "w").close()  # claims are not donors
        published, failed = scan_donors(self.dir)
        self.assertEqual(sorted(p["name"] for p in published), ["a", "b"])
        self.assertEqual(sum(p["offered_bytes"] for p in published), 260 * _GIB)
        self.assertEqual([p["short_bytes"] for p in published if p["name"] == "b"], [60 * _GIB])
        self.assertEqual([f["name"] for f in failed], ["c"])
        self.assertEqual(failed[0]["planned_bytes"], 160 * _GIB)
        self.assertIn("OUT_OF_MEMORY", failed[0]["error"])

    def test_a_donor_file_without_its_transport_is_malformed(self):
        """Every writer names its transport; a .meta or .failed without it counts as no donor."""
        _publish(self.dir, "ok", 4)
        _publish(self.dir, "bare", 4)
        meta = os.path.join(self.dir, "donor.bare.meta")
        lines = [line for line in open(meta) if not line.startswith("transport ")]
        with open(meta, "w") as f:
            f.writelines(lines)
        with open(os.path.join(self.dir, "donor.lost.failed"), "w") as f:
            f.write("version 1\nname lost\nhost h\npid 1\nplanned_bytes 1\nerror e\n")
        published, failed = scan_donors(self.dir)
        self.assertEqual([p["name"] for p in published], ["ok"])
        self.assertEqual(failed, [])
        # The capacity check warns once per malformed file and names them, instead of blaming a missing donor.
        with mock.patch.object(remote_donor.logger, "warning") as warning:
            with self.assertRaisesRegex(
                RuntimeError, "malformed.*donor.bare.meta, donor.lost.failed"
            ):
                check_remote_capacity(
                    self.dir, 3, 20 * _GIB, timeout_s=0.3, poll_s=0.05, log=lambda _: None
                )
        self.assertEqual(
            sorted(c.args[0].split()[4] for c in warning.call_args_list),
            ["donor.bare.meta", "donor.lost.failed"],
        )

    def test_every_donor_names_its_transport(self):
        for kind in remote_donor.TRANSPORTS:
            donor = make_donor(self.dir, _GIB, kind, chunk_bytes=_GIB, name=f"named-{kind}")
            donor._publish([], ["segment 0 1"], "test")
            donor.publish_failure(RuntimeError("e"))
            fields = remote_donor._meta_fields(os.path.join(self.dir, f"donor.named-{kind}.meta"))
            self.assertEqual(fields["transport"], kind)
            failed = remote_donor._meta_fields(os.path.join(self.dir, f"donor.named-{kind}.failed"))
            self.assertEqual(failed["transport"], kind)
            donor.close()

    def test_only_donors_of_the_clients_transport_count(self):
        """Clients skip donors of another transport, so their capacity does not cover the need."""
        _publish(self.dir, "fabric", 160)
        _publish(self.dir, "nixl", 40, transport="nixl")
        check_remote_capacity(self.dir, 2, 100 * _GIB, timeout_s=10, log=lambda _: None)
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                2,
                100 * _GIB,
                transport="nixl",
                timeout_s=0.2,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertIn("1 of 2 planned donors published 40 GiB", str(ctx.exception))
        self.assertIn("donor.size", str(ctx.exception))

    def test_passes_when_offer_covers_need(self):
        for i in range(8):
            _publish(self.dir, f"l{i}", 160)
        result = check_remote_capacity(self.dir, 8, 1218 * _GIB, timeout_s=10, log=lambda _: None)
        self.assertEqual(result["offered_bytes"], 1280 * _GIB)

    def test_fails_fast_when_all_donors_reported_short(self):
        # 5 of 8 donors published 160 GiB, 3 failed; the clients claim 6 x 203 GiB.
        for i in range(5):
            _publish(self.dir, f"ok{i}", 160)
        for i in range(3):
            make_donor(self.dir, 160 * _GIB, chunk_bytes=_GIB, name=f"bad{i}").publish_failure(
                RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
            )
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                8,
                1218 * _GIB,
                timeout_s=600,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertLess(
            time.monotonic() - t0, 5.0
        )  # no wait for the timeout: every donor has reported
        message = str(ctx.exception)
        self.assertIn("5 of 8 planned donors published 800 GiB, 3 failed", message)
        self.assertIn("claim 1218 GiB", message)
        self.assertIn("bad0", message)
        self.assertIn("OUT_OF_MEMORY", message)

    def test_names_short_donors(self):
        _publish(self.dir, "a", 100, short_gib=60)
        _publish(self.dir, "b", 160)
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                2,
                300 * _GIB,
                timeout_s=10,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertIn("a (hostA) is 60 GiB short", str(ctx.exception))

    def test_short_donor_note_names_its_numa_placement(self):
        _publish(self.dir, "a", 100, short_gib=60, extra=["numa_chunks 0:60,1:40"])
        _publish(self.dir, "b", 160)
        published, _ = scan_donors(self.dir)
        self.assertEqual(
            {p["name"]: p["numa_chunks"] for p in published}, {"a": "0:60,1:40", "b": ""}
        )
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                2,
                300 * _GIB,
                timeout_s=10,
                poll_s=0.05,
                log=lambda _: None,
            )
        message = str(ctx.exception)
        self.assertIn("a (hostA) is 60 GiB short (chunks per NUMA node 0:60,1:40)", message)
        self.assertNotIn("\n", message)

    def test_times_out_when_donors_do_not_report(self):
        _publish(self.dir, "a", 10)
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            check_remote_capacity(
                self.dir,
                3,
                20 * _GIB,
                timeout_s=0.3,
                poll_s=0.05,
                log=lambda _: None,
            )
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        self.assertIn("2 not reported", str(ctx.exception))

    def test_passes_before_all_report_once_covered(self):
        _publish(self.dir, "a", 160)
        _publish(self.dir, "b", 160)
        t0 = time.monotonic()
        check_remote_capacity(
            self.dir,
            8,
            300 * _GIB,
            timeout_s=600,
            poll_s=0.05,
            log=lambda _: None,
        )
        self.assertLess(time.monotonic() - t0, 5.0)

    def test_waits_for_a_donor_that_publishes_later(self):
        _publish(self.dir, "a", 100)
        threading.Timer(0.3, lambda: _publish(self.dir, "b", 100)).start()
        result = check_remote_capacity(
            self.dir,
            2,
            200 * _GIB,
            timeout_s=10,
            poll_s=0.05,
            log=lambda _: None,
        )
        self.assertEqual(result["offered_bytes"], 200 * _GIB)

    def test_failure_marker_survives_close(self):
        donor = make_donor(self.dir, 4 * _GIB, chunk_bytes=_GIB, name="gone")
        donor.close()
        donor.publish_failure(RuntimeError("boom\nsecond line"))
        path = os.path.join(self.dir, "donor.gone.failed")
        self.assertTrue(os.path.exists(path))
        text = open(path).read()
        self.assertIn("planned_bytes 4294967296", text)
        self.assertNotIn("\nsecond line", text)  # the error stays on one line

    def test_close_waits_for_a_failed_background_start_closing_itself(self):
        # The start thread closes the donor itself on failure; a concurrent close() must not withdraw alongside it.
        donor = make_donor(self.dir, 4 * _GIB, chunk_bytes=_GIB, name="racy")
        entered, release = threading.Event(), threading.Event()
        active, depths = [0], []
        real_withdraw = donor._withdraw

        def withdraw():
            active[0] += 1
            depths.append(active[0])
            if threading.current_thread().name == "kvcm2-remote-donor":
                entered.set()
                release.wait(10)
            real_withdraw()
            active[0] -= 1

        start_threads = []

        def failing_start():
            start_threads.append(threading.current_thread())
            raise RuntimeError("no memory")

        donor._withdraw = withdraw
        donor.start = failing_start
        donor.start_in_background()
        self.assertTrue(entered.wait(10))
        closer = threading.Thread(target=donor.close)
        closer.start()
        closer.join(0.3)
        self.assertTrue(
            closer.is_alive(), "close() returned while the start thread was still closing"
        )
        release.set()
        closer.join(10)
        self.assertFalse(closer.is_alive())
        self.assertEqual(depths, [1, 1])
        start_threads[0].join(10)  # it still writes the failure marker after its close
        self.assertFalse(start_threads[0].is_alive())

    def test_wait_started_returns_once_a_failed_start_is_released(self):
        """A failed background start closes itself; wait_started() must not return while that release still runs."""
        released = threading.Event()

        class _SlowRelease(remote_donor.DonorTransport):
            kind = "nixl"

            def start(self, donor):
                time.sleep(0.2)
                raise RuntimeError("registration failed")

            def release_unowned(self):
                time.sleep(0.8)
                released.set()

        donor = remote_donor.RemoteMemDonor(
            self.dir, 4 * _GIB, _SlowRelease(), chunk_bytes=_GIB, name="slow"
        )
        donor.start_in_background()
        time.sleep(0.4)  # the start failed at 0.2 s; its release runs until 1.0 s
        donor.wait_started()
        self.assertTrue(
            released.is_set(), "wait_started() returned during the failed start's release"
        )
        _, failed = scan_donors(self.dir)
        self.assertEqual([f["name"] for f in failed], ["slow"])
        donor.close()

    def _started(self, name):
        donor = make_donor(self.dir, 4 * _GIB, "fabric", chunk_bytes=_GIB, name=name)
        with mock.patch.object(
            FabricDonorTransport,
            "start",
            lambda self, donor: donor._publish([], ["segment 0 1"], "test"),
        ):
            donor.start()
        return donor

    def test_a_donor_nobody_closed_withdraws_and_leaves_a_marker(self):
        # E.g. startup failed between the donation start and the manager that would own it.
        donor = self._started("orphan")
        ready = os.path.join(self.dir, "donor.orphan.ready")
        self.assertTrue(os.path.exists(ready))
        del donor
        gc.collect()
        self.assertFalse(os.path.exists(ready))
        _, failed = scan_donors(self.dir)
        self.assertEqual([f["name"] for f in failed], ["orphan"])

    def test_a_start_still_running_when_the_finalizer_withdraws_publishes_nothing(self):
        """A start still running when the finalizer withdrew the donor publishes nothing.

        At exit the finalizer runs while a daemon start thread may still be past its registration: once it withdrew
        the donor and left the failure marker, the start must not publish .ready for memory it released.
        """
        registered, finalized = threading.Event(), threading.Event()

        class _Racing(remote_donor.DonorTransport):
            kind = "fabric"

            def start(self, donor):
                registered.set()
                finalized.wait(10)
                donor._publish([], ["segment 0 1"], "test")

        donor = remote_donor.RemoteMemDonor(
            self.dir, 4 * _GIB, _Racing(), chunk_bytes=_GIB, name="exiting"
        )
        donor.start_in_background()
        self.assertTrue(registered.wait(10))
        donor._finalizer()  # what atexit runs for a donor nobody closed
        finalized.set()
        donor.wait_started()
        published, failed = scan_donors(self.dir)
        self.assertEqual(published, [])
        self.assertEqual([f["name"] for f in failed], ["exiting"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "donor.exiting.meta")))

    def test_a_closed_donor_leaves_no_marker(self):
        donor = self._started("done")
        donor.close()
        del donor
        gc.collect()
        self.assertEqual(scan_donors(self.dir), ([], []))

    def test_close_with_an_error_leaves_a_marker(self):
        donor = self._started("aborted")
        donor.close(RuntimeError("initialization failed on another rank"))
        _, failed = scan_donors(self.dir)
        self.assertIn("another rank", failed[0]["error"])

    def test_donor_names_are_unique_within_a_process(self):
        names = {make_donor(self.dir, _GIB, chunk_bytes=_GIB).name for _ in range(64)}
        self.assertEqual(len(names), 64)


@pytest.mark.cpu_only
class TestBindingDefaults(unittest.TestCase):
    def test_the_binding_timeouts_default_to_remote_cache_config(self):
        """The C++ defaults (config.h) match RemoteCacheConfig's; a launcher that omits a field gets one."""
        from tensorrt_llm.llmapi.llm_args import RemoteCacheConfig
        from tensorrt_llm.runtime.kv_cache_manager_v2 import (
            RemoteCacheTierConfig,
            RemoteTransferConfig,
            RemoteTransport,
        )

        defaults = RemoteCacheConfig(directory="d", level={"size": 1})
        transfer = RemoteTransferConfig()
        self.assertEqual(transfer.timeout_ms, defaults.transport.transfer_timeout_s * 1000)
        self.assertEqual(transfer.transport, RemoteTransport[defaults.transport.kind.upper()])
        tier = RemoteCacheTierConfig(quota=0, path="")
        self.assertEqual(tier.wait_timeout_ms, defaults.level.donor_wait_timeout_s * 1000)


def _capture_mode():
    """The calling thread's stream-capture mode."""
    from cuda.bindings import driver as drv

    relaxed = drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    err, mode = drv.cuThreadExchangeStreamCaptureMode(relaxed)
    assert err == drv.CUresult.CUDA_SUCCESS, err
    drv.cuThreadExchangeStreamCaptureMode(mode)
    return mode


@unittest.skipUnless(gpu_count() > 0, "needs a GPU")
class TestFailedStartReleaseCudaState(unittest.TestCase):
    """A start that fails after registration releases on the donor thread, in the start's relaxed capture mode.

    The serving thread may capture a CUDA graph in global mode meanwhile, which a capture-unsafe call made in global
    mode on another thread (e.g. deregistering host memory) would invalidate.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_release_mode_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_nixl_donation_deregisters_in_relaxed_mode(self):
        modes = []

        class _Agent:
            def __init__(self, config):
                pass

            def register_memory(self, descs):
                pass

            def deregister_memory(self, descs):
                modes.append(_capture_mode())

            def get_local_agent_desc(self):
                raise RuntimeError("no agent descriptor")

        binding = types.ModuleType("tensorrt_llm.tensorrt_llm_transfer_agent_binding")
        binding.BaseAgentConfig = types.SimpleNamespace
        binding.NixlTransferAgent = _Agent
        binding.MemoryDesc = lambda *args: args
        binding.MemoryDescs = lambda *args: args
        binding.MemoryType = types.SimpleNamespace(DRAM="DRAM")
        donor = make_donor(self.dir, 4 << 20, "nixl", chunk_bytes=2 << 20, device=0)
        with mock.patch.dict(sys.modules, {binding.__name__: binding}):
            donor.start_in_background()
            donor.wait_started()
        from cuda.bindings import driver as drv

        self.assertEqual(modes, [drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED])
        _, failed = scan_donors(self.dir)
        self.assertIn("no agent descriptor", failed[0]["error"])

    @unittest.skipUnless(fabric_available(), "needs the fabric transport")
    def test_a_fabric_donation_releases_its_chunks_in_relaxed_mode(self):
        modes = []
        real_release = FabricDonorTransport._release_fabric

        def release(transport):
            if transport._fabric:
                modes.append(_capture_mode())
            real_release(transport)

        def failing_publish(*args, **kwargs):
            raise OSError("publish failed")

        donor = make_donor(self.dir, 2 * (16 << 20), "fabric", chunk_bytes=16 << 20, device=0)
        donor._publish = failing_publish
        with mock.patch.object(FabricDonorTransport, "_release_fabric", release):
            donor.start_in_background()
            donor.wait_started()
        from cuda.bindings import driver as drv

        self.assertEqual(modes, [drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED])
        self.assertEqual(donor.transport._fabric, [])


@pytest.mark.cpu_only
class TestRendezvousPermissions(unittest.TestCase):
    """Donor files hold NIXL registration keys and fabric handles: the directory and its files are the user's."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_perm_")

    def tearDown(self):
        os.chmod(self.dir, 0o700)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_new_directory_is_private(self):
        path = os.path.join(self.dir, "rendezvous")
        remote_donor.ensure_rendezvous_directory(path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o700)

    def test_a_directory_others_can_write_is_refused(self):
        os.chmod(self.dir, 0o777)
        with self.assertRaises(PermissionError):
            remote_donor.ensure_rendezvous_directory(self.dir)
        os.chmod(self.dir, 0o755)
        remote_donor.ensure_rendezvous_directory(self.dir)

    def test_a_symlinked_directory_is_refused(self):
        """The leaf is checked without following a link: a link can be pointed elsewhere after the check."""
        target = os.path.join(self.dir, "target")
        os.mkdir(target, 0o700)
        link = os.path.join(self.dir, "rendezvous")
        os.symlink(target, link)
        with self.assertRaises(PermissionError):
            remote_donor.ensure_rendezvous_directory(link)

    def test_a_symlinked_directory_with_a_trailing_slash_is_refused(self):
        """A path ending in a separator follows the link in lstat and in the C++ client's O_NOFOLLOW open."""
        target = os.path.join(self.dir, "target")
        os.mkdir(target, 0o700)
        link = os.path.join(self.dir, "rendezvous")
        os.symlink(target, link)
        with self.assertRaises(PermissionError):
            remote_donor.ensure_rendezvous_directory(link + "/")
        self.assertEqual(remote_donor.resolve_rendezvous_directory(link + "//"), link)

    def test_missing_parents_are_private_too(self):
        path = os.path.join(self.dir, "a", "b", "rendezvous")
        remote_donor.ensure_rendezvous_directory(path)
        for d in (os.path.join(self.dir, "a"), os.path.join(self.dir, "a", "b"), path):
            self.assertEqual(os.stat(d).st_mode & 0o077, 0, d)

    def test_an_unset_variable_in_the_directory_is_refused(self):
        """An unset variable would leave one literal directory that every run without it shares."""
        with mock.patch.dict(os.environ, {"KVCM2_TEST_JOB": "7"}):
            os.environ.pop("KVCM2_TEST_UNSET", None)
            self.assertEqual(
                remote_donor.resolve_rendezvous_directory(f"{self.dir}/${{KVCM2_TEST_JOB}}/"),
                f"{self.dir}/7",
            )
            for unset in ("${KVCM2_TEST_UNSET}", "$KVCM2_TEST_UNSET"):
                with self.assertRaisesRegex(ValueError, re.escape(unset)):
                    remote_donor.resolve_rendezvous_directory(f"{self.dir}/{unset}/x")

    def test_the_compaction_lock_is_not_opened_through_a_link(self):
        victim = os.path.join(self.dir, "victim")
        open(victim, "w").close()
        os.symlink(victim, os.path.join(self.dir, "compact.h.node0.lock"))
        with host_memory.node_compaction_lock(self.dir, "h", 0) as held:
            self.assertFalse(held)

    def test_a_planted_tmp_link_is_not_written_through(self):
        """Files are written through fresh exclusive temporaries, never through a file or link already there."""
        victim = os.path.join(self.dir, "victim")
        with open(victim, "w") as f:
            f.write("keep")
        path = os.path.join(self.dir, "donor.x.failed")
        os.symlink(victim, f"{path}.tmp.{os.getpid()}")
        remote_donor._write_atomically(path, b"marker")
        self.assertEqual(open(victim).read(), "keep")
        self.assertEqual(open(path, "rb").read(), b"marker")
        self.assertFalse(os.path.islink(path))

    def test_donor_files_are_private(self):
        make_donor(self.dir, _GIB, chunk_bytes=_GIB, name="x").publish_failure(RuntimeError("e"))
        self.assertEqual(os.stat(os.path.join(self.dir, "donor.x.failed")).st_mode & 0o077, 0)


@pytest.mark.cpu_only
class TestPopulate(unittest.TestCase):
    """Populate errors other than a missing MADV_POPULATE_WRITE (test_remote_donor_memory) raise."""

    def _populate_with(self, error):
        region = mmap.mmap(-1, 1 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            with mock.patch.object(host_memory, "madvise_range", side_effect=error):
                host_memory.populate(region, 0, 1 << 20)
        finally:
            region.close()

    def test_other_errors_raise(self):
        for code in (errno.ENOMEM, getattr(errno, "EHWPOISON", 133)):
            with self.assertRaises(OSError):
                self._populate_with(OSError(code, os.strerror(code)))


@pytest.mark.cpu_only
class TestPlaceChunks(unittest.TestCase):
    """Chunk placement when the driver refuses chunks on a node that still has room.

    Pure Python: the allocator, the node memory, the reclaim-and-compact pass, the per-node lock and the clock are
    fakes. The policy is the module's (2 s of retries, passes of at least 8 GiB) with a 24 GiB reserve per node.
    """

    def setUp(self):
        self.t = [0.0]
        self.room = {0: 200.0, 1: 200.0}  # GiB free per node (no page cache)
        # GiB free in >= 2 MiB blocks; passes raise it only where a test says so.
        self.contig = {0: 0.0, 1: 0.0}
        self.passes = []
        self.locked = set()  # nodes whose compaction lock is held
        self.on_lock = lambda node: True  # runs as the lock is taken; False refuses it

    @contextlib.contextmanager
    def _node_lock(self, node):
        granted = self.on_lock(node)
        if granted:
            self.locked.add(node)
        try:
            yield granted
        finally:
            self.locked.discard(node)

    def _sleep(self, seconds):
        self.t[0] += seconds

    def _meminfo(self, node):
        return {"MemFree": self.room[node], "MemTotal": 480.0}

    def _compact(self, node, gib):
        assert node in self.locked, "a pass ran without its node's lock"
        self.passes.append((node, gib))

    def _place(self, n_chunks, create, **overrides):
        args = dict(
            chunk_gib=1.0,
            reserve_gib=24.0,
            meminfo=self._meminfo,
            compact=self._compact,
            contiguous=lambda node: self.contig[node],
            sleep=self._sleep,
            clock=lambda: self.t[0],
            node_lock=self._node_lock,
        )
        args.update(overrides)
        return host_memory.place_chunks(n_chunks, [0, 1], create, **args)

    def _allocator(self, refuse):
        """create(node) refuses when refuse(node, call number) says so; an accepted chunk takes 1 GiB of the node."""
        calls = []

        def create(node):
            calls.append(node)
            if refuse(node, len(calls)):
                return None
            self.room[node] -= 1.0
            return object()

        return create, calls

    def _passes_on(self, node):
        return [gib for n, gib in self.passes if n == node]

    def test_without_compaction_passes_a_refused_chunk_moves_on_after_its_retries(self):
        """compact_host_memory off (the default): no pass and no node lock; what fits is placed."""
        create, _ = self._allocator(lambda node, call: node == 0)
        handles, placement = self._place(4, create, compact_passes=False)
        self.assertEqual(self.passes, [])
        self.assertEqual(placement, {0: 0, 1: 4})
        self.assertEqual(len(handles), 4)

    def test_all_accepted(self):
        create, calls = self._allocator(lambda node, i: False)
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 10)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(calls, [0] * 10)

    def test_transient_refusal_is_retried(self):
        # Refuse chunk 3's first try and its first retry.
        create, calls = self._allocator(lambda node, i: i in (4, 5))
        handles, placement = self._place(10, create)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(calls, [0] * 12)
        self.assertEqual(self.passes, [])
        self.assertGreater(self.t[0], 0.0)  # backed off before the retries

    def test_pass_recovers_a_refusal(self):
        # Node 0 refuses from chunk 5 on until one reclaim-and-compact pass has run on it.
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        handles, placement = self._place(10, create)
        self.assertEqual(placement, {0: 10, 1: 0})
        self.assertEqual(self.passes, [(0, 8.0)])  # the smallest pass: the node has plenty of room
        self.assertGreaterEqual(self.t[0], 2.0)  # the retries came first

    def test_pass_is_sized_to_keep_the_reserve_and_margin(self):
        self.room[0] = 24.0 + 1.0 + 8.0 + 3.5  # reserve + chunk + margin + 3.5 GiB
        create, _ = self._allocator(lambda node, i: node == 0 and not self.passes)
        handles, placement = self._place(1, create)
        self.assertEqual(placement[0], 1)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 3.5)

    def test_no_room_for_a_pass_moves_on_and_stops_trying_the_node(self):
        self.room[0] = 24.0 + 1.0 + 8.0 + 0.5  # less than 1 GiB left for a pass
        create, calls = self._allocator(lambda node, i: node == 0)
        handles, placement = self._place(5, create)
        self.assertEqual(placement, {0: 0, 1: 5})
        self.assertEqual(self.passes, [])
        # Refused (and retried) for the first chunk only, then skipped for the later ones.
        self.assertNotIn(0, calls[calls.index(1) :])

    def test_short_when_every_node_refuses(self):
        create, _ = self._allocator(lambda node, i: i > 3)  # three chunks, then nothing anywhere
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 3)
        self.assertEqual(placement, {0: 3, 1: 0})
        self.assertEqual(len(self._passes_on(0)), host_memory._MAX_PASSES_PER_REFUSAL)

    def test_a_failing_pass_stops_passes_on_the_node(self):
        attempts = []

        def broken(node, gib):
            attempts.append(node)
            raise RuntimeError("libnuma is unavailable")

        create, _ = self._allocator(lambda node, i: node == 0 and i > 2)
        handles, placement = self._place(4, create, compact=broken)
        self.assertEqual(placement, {0: 2, 1: 2})
        self.assertEqual(attempts, [0])

    def test_a_pass_is_sized_to_the_donation_still_missing(self):
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(40, create)
        self.assertEqual(self.passes, [(0, 35.0)])  # 40 - 5 chunks, more than the smallest pass

    def test_a_pass_stays_within_its_cap(self):
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(150, create)
        self.assertEqual(self.passes, [(0, 64.0)])  # _MAX_PASS_CHUNKS chunks

    def test_pass_floor_and_margin_follow_the_reserve(self):
        # A third of a 48 GiB reserve: passes of at least 16 GiB, leaving 16 GiB above the reserve and the chunk.
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(10, create, reserve_gib=48.0)
        self.assertEqual(self.passes, [(0, 16.0)])
        self.passes.clear()
        self.room[0] = 48.0 + 1.0 + 16.0 + 3.5
        create, _ = self._allocator(lambda node, i: node == 0 and not self.passes)
        self._place(1, create, reserve_gib=48.0)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 3.5)

    def test_passes_go_on_while_they_raise_contiguous_memory(self):
        # Node 0 refuses until five passes have run; each pass leaves its size free in >= 2 MiB blocks.
        def compact(node, gib):
            self.passes.append((node, gib))
            self.contig[node] += gib

        create, _ = self._allocator(lambda node, i: node == 0 and i > 2 and len(self.passes) < 5)
        handles, placement = self._place(4, create, compact=compact)
        self.assertEqual(placement, {0: 4, 1: 0})
        self.assertEqual(len(self.passes), 5)

    def test_a_node_given_up_on_is_swept_once_more(self):
        # Node 0 is fragmented while node 1 still has room; once node 1 is full, node 0 holds memory in >= 2 MiB blocks
        # again after the donor left it.
        self.room[1] = 24.0 + 4.0  # four chunks above the reserve

        def contiguous(node):
            return 10.0 if node == 0 and self.room[1] < 25.0 else 0.0

        create, _ = self._allocator(lambda node, i: node == 0 and contiguous(0) < 1.0)
        handles, placement = self._place(8, create, contiguous=contiguous)
        self.assertEqual(placement, {0: 4, 1: 4})
        self.assertEqual(len(self._passes_on(0)), host_memory._MAX_PASSES_PER_REFUSAL)

    def test_the_sweep_runs_once(self):
        # Nodes that look recovered but refuse every chunk: the donation ends short after one sweep, without looping.
        create, calls = self._allocator(lambda node, i: i > 3)
        self.contig = {0: 5.0, 1: 5.0}
        handles, placement = self._place(10, create)
        self.assertEqual(len(handles), 3)
        self.assertLess(len(calls), 200)

    def test_a_pass_is_sized_from_the_memory_left_once_it_holds_the_lock(self):
        # Another donor on the node fills 160 GiB of it while this one waits for the lock.
        def other_donor_ran(node):
            self.room[node] -= 160.0
            return True

        self.on_lock = other_donor_ran
        create, _ = self._allocator(lambda node, i: node == 0 and i > 5 and not self.passes)
        self._place(150, create)
        self.assertEqual(len(self.passes), 1)
        self.assertAlmostEqual(self.passes[0][1], 200.0 - 5 - 160.0 - 24.0 - 1.0 - 8.0)

    def test_no_pass_without_the_lock(self):
        self.on_lock = lambda node: False
        create, _ = self._allocator(lambda node, i: node == 0 and i > 2)
        handles, placement = self._place(4, create)
        self.assertEqual(placement, {0: 2, 1: 2})
        self.assertEqual(self.passes, [])

    def test_reserve_skips_a_node_without_trying_it(self):
        self.room[0] = 24.5  # below reserve + chunk
        create, calls = self._allocator(lambda node, i: False)
        handles, placement = self._place(3, create)
        self.assertEqual(placement, {0: 0, 1: 3})
        self.assertNotIn(0, calls)


@pytest.mark.cpu_only
class TestNodeCompactionLock(unittest.TestCase):
    """The flock that lets one donor per host and NUMA node run reclaim-and-compact passes at a time."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_compaction_lock_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_donors_on_one_node_take_turns(self):
        order = []
        holding = threading.Event()

        def second():
            holding.wait(10)
            with host_memory.node_compaction_lock(self.dir, "hostA", 0) as locked:
                order.append(("second", locked))

        thread = threading.Thread(target=second)
        thread.start()
        with host_memory.node_compaction_lock(self.dir, "hostA", 0) as locked:
            holding.set()
            thread.join(0.3)
            self.assertTrue(thread.is_alive(), "the second donor passed a held lock")
            order.append(("first", locked))
        thread.join(10)
        self.assertEqual(order, [("first", True), ("second", True)])

    def test_other_nodes_and_hosts_do_not_wait(self):
        with host_memory.node_compaction_lock(self.dir, "hostA", 0) as locked:
            self.assertTrue(locked)
            for host, node in (("hostA", 1), ("hostB", 0)):
                with host_memory.node_compaction_lock(self.dir, host, node) as other:
                    self.assertTrue(other)

    def test_unusable_directory_refuses_the_lock(self):
        missing = os.path.join(self.dir, "missing")
        with host_memory.node_compaction_lock(missing, "hostA", 0) as locked:
            self.assertFalse(locked)


@pytest.mark.cpu_only
class TestBuddyInfo(unittest.TestCase):
    TEXT = (
        "Node 0, zone   Normal     10      5      3      2      1      0      0      0      0      4      2\n"
        "Node 1, zone   Normal    100     50     30     20     10      5      0      0      0      0      0\n"
    )

    def test_free_memory_in_blocks_of_2_mib_or_more(self):
        f = host_memory.contiguous_free_gib
        # 4 KiB pages: order 9 is 2 MiB, order 10 is 4 MiB: 4 x 2 + 2 x 4 = 16 MiB on node 0, none on node 1.
        self.assertAlmostEqual(f(0, text=self.TEXT, page=4096), 16 / 1024)
        self.assertEqual(f(1, text=self.TEXT, page=4096), 0.0)
        # 64 KiB pages (aarch64): order 5 is 2 MiB. Node 0: 4 x 32 + 2 x 64 MiB; node 1: 5 x 2 MiB.
        self.assertAlmostEqual(f(0, text=self.TEXT, page=65536), 256 / 1024)
        self.assertAlmostEqual(f(1, text=self.TEXT, page=65536), 10 / 1024)
        self.assertEqual(f(7, text=self.TEXT, page=4096), 0.0)


@unittest.skipUnless(
    len(host_memory.host_numa_nodes()) >= 2 and fabric_available(),
    "needs the fabric transport (a Grace GPU with an IMEX channel) and two NUMA nodes with memory",
)
class TestFabricPlacement(unittest.TestCase):
    """Where a fabric donation's chunks go when its node is full or the driver refuses them.

    A HOST_NUMA allocation never leaves its node, and the driver can refuse chunks with room left. Node fullness and
    refusals are faked; the allocations are real.
    """

    CHUNK = 16 << 20

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="kvcm2_fabric_placement_")
        self.real = host_memory.node_meminfo

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _full(self, node):
        return dict(self.real(node), MemFree=0.0, **{"Active(file)": 0.0, "Inactive(file)": 0.0})

    def _meta(self, donor):
        return remote_donor._meta_fields(os.path.join(self.dir, f"donor.{donor.name}.meta"))

    def test_full_preferred_node_spills_to_the_other(self):
        donor = make_donor(
            self.dir,
            4 * self.CHUNK,
            transport="fabric",
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        fake = lambda n: self._full(n) if n == 0 else self.real(n)  # noqa: E731
        with mock.patch.object(host_memory, "node_meminfo", fake):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "1:4")
            self.assertEqual(fields["segments"], [[0, 4 * self.CHUNK]])
            self.assertNotIn("short_bytes", fields)
            fabric = os.path.join(self.dir, f"donor.{donor.name}.fabric")
            self.assertEqual(os.path.getsize(fabric), 4 * 64)
        finally:
            donor.close()

    def test_donation_that_only_partly_fits_is_published_short(self):
        donor = make_donor(
            self.dir,
            4 * self.CHUNK,
            transport="fabric",
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )

        # Node 0 is full, node 1 has room for two chunks.
        def fake(n):
            return self._full(n) if n == 0 or len(donor.transport._fabric) >= 2 else self.real(n)

        with mock.patch.object(host_memory, "node_meminfo", fake):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["segments"], [[0, 2 * self.CHUNK]])
            self.assertEqual(int(fields["short_bytes"]), 2 * self.CHUNK)
            published, _ = scan_donors(self.dir)
            self.assertEqual(published[0]["offered_bytes"], 2 * self.CHUNK)
        finally:
            donor.close()

    def test_refused_chunks_are_retried_with_real_allocations(self):
        # The driver refuses two calls (chunk 1's first try and its first retry); the donation still completes on
        # node 0.
        real_create = FabricDonorTransport._try_create
        calls = [0]

        def flaky(transport, drv, prop):
            calls[0] += 1
            return None if calls[0] in (2, 3) else real_create(transport, drv, prop)

        donor = make_donor(
            self.dir,
            4 * self.CHUNK,
            transport="fabric",
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
        )
        with (
            mock.patch.object(FabricDonorTransport, "_try_create", flaky),
            mock.patch.object(host_memory, "_REFUSAL_RETRY_S", 0.5),
        ):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "0:4")
            self.assertEqual(calls[0], 6)
        finally:
            donor.close()

    def test_a_real_reclaim_pass_recovers_a_refusal(self):
        # Refuse until one real reclaim-and-compact pass (64 chunks of 16 MiB, libnuma-bound, huge pages) has run on
        # node 0.
        real_create = FabricDonorTransport._try_create
        real_compact = host_memory.compact_node
        passes = []

        def counted(node, gib):
            passes.append((node, gib))
            return real_compact(node, gib)

        def refusing(transport, drv, prop):
            return real_create(transport, drv, prop) if passes or not transport._fabric else None

        donor = make_donor(
            self.dir,
            3 * self.CHUNK,
            transport="fabric",
            chunk_bytes=self.CHUNK,
            numa_node=0,
            device=0,
            compact_host_memory=True,
        )
        with (
            mock.patch.object(FabricDonorTransport, "_try_create", refusing),
            mock.patch.object(host_memory, "compact_node", counted),
            mock.patch.object(host_memory, "_REFUSAL_RETRY_S", 0.2),
        ):
            donor.start()
        try:
            fields = self._meta(donor)
            self.assertEqual(fields["numa_chunks"], "0:3")
            self.assertEqual(passes, [(0, 1.0)])
        finally:
            donor.close()

    def test_no_room_anywhere_leaves_a_failure_marker(self):
        donor = make_donor(self.dir, 4 * self.CHUNK, "fabric", chunk_bytes=self.CHUNK)
        with mock.patch.object(host_memory, "node_meminfo", self._full):
            donor.start_or_report()
        self.assertFalse(os.path.exists(os.path.join(self.dir, f"donor.{donor.name}.ready")))
        published, failed = scan_donors(self.dir)
        self.assertEqual(published, [])
        self.assertEqual([f["name"] for f in failed], [donor.name])
        self.assertIn("no NUMA node can hold", failed[0]["error"])


if __name__ == "__main__":
    unittest.main()
