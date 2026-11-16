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
"""KV cache manager v2 with a remote cold level (``RemoteCacheTierConfig``).

The level keeps its slots in host memory that donor processes (``remote_donor.py``) publish in a rendezvous
directory, and moves pages with NIXL READ / WRITE or over NVLink fabric memory (``storage/remoteMem.h``). Every test
class runs once per transport; a transport this machine lacks (remote_test_utils) is skipped.
"""

import contextlib
import gc
import glob
import itertools
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from typing import cast
from unittest import mock

import cuda.bindings.driver as drv
import pytest
from parameterized import parameterized_class

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    import test_kv_cache_manager_v2 as base  # the suite's fixtures, helpers and imports
    from remote_test_utils import available_transports, gpu_count
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

AttentionLayerConfig = base.AttentionLayerConfig
BufferConfig = base.BufferConfig
CacheLevel = base.CacheLevel
CudaStream = base.CudaStream
DataRole = base.DataRole
DiskCacheTierConfig = base.DiskCacheTierConfig
FakeEngine = base.FakeEngine
GpuCacheTierConfig = base.GpuCacheTierConfig
HostCacheTierConfig = base.HostCacheTierConfig
KVCacheManager = base.KVCacheManager
KVCacheManagerConfig = base.KVCacheManagerConfig
LayerId = base.LayerId
SsmLayerConfig = base.SsmLayerConfig
Step = base.Step
TokenId = base.TokenId
_introspection = base._introspection
RemoteCacheTierConfig = base.RemoteCacheTierConfig
RemoteMemError = base.RemoteMemError
RemoteTransferConfig = base.RemoteTransferConfig
RemoteTransport = base.RemoteTransport


def _sync_device() -> None:
    """Wait for all device work, and with it every fabric copy queued behind it."""
    (err,) = drv.cuCtxSynchronize()
    assert err == drv.CUresult.CUDA_SUCCESS, err


_TRANSPORTS = available_transports()
pytestmark = pytest.mark.skipif(
    not _TRANSPORTS, reason="needs a GPU and the NIXL or fabric transport"
)
# One concrete class per transport: <Class>_0_nixl, <Class>_1_fabric.
_per_transport = parameterized_class(("transport",), [("nixl",), ("fabric",)])
# The largest claim of one test (the suite's 1 GiB disk quota, rounded up to whole chunks per pool group) and room.
_DONOR_BYTES = 2 << 30
_CHUNK_BYTES = 8 << 20
_counter = itertools.count()
_NIXL_DEATH_ONLY = (
    "fabric: a dead donor's effect is the driver's; this test checks NIXL's fail-stop"
)


class _Donor:
    """A donor in its own process, as on a GEN tray.

    Inside a Slurm step the subprocess joins the step's PMIx job (`import tensorrt_llm` runs MPI_Init), so it is
    stopped with SIGTERM and exits normally: a SIGKILLed donor makes Slurm abort the whole step. A stopped donor has
    deregistered its memory and destroyed its NIXL agent, which is what a client sees of a dead one.
    """

    def __init__(
        self,
        directory: str,
        nbytes: int,
        transport: str,
        chunk_bytes: int = _CHUNK_BYTES,
        env: "dict[str, str] | None" = None,
    ):
        self.directory = directory
        self.name = f"test-donor-{os.getpid()}-{next(_counter)}"
        self.log_path = os.path.join(directory, f"donor_{self.name}.log")
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(_TEST_DIR, "remote_donor_cli.py"),
                "--dir",
                directory,
                "--bytes",
                str(nbytes),
                "--chunk-mib",
                str(chunk_bytes / (1 << 20)),
                "--name",
                self.name,
                "--transport",
                transport,
                # The last GPU, so that on a multi-GPU node donor and client sit on different devices.
                "--device",
                str(gpu_count() - 1),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            env=None if env is None else {**os.environ, **env},
        )
        ready = os.path.join(directory, f"donor.{self.name}.ready")
        deadline = time.monotonic() + 300
        while not os.path.exists(ready):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"donor exited early (rc {self.proc.returncode}):\n{self._tail()}"
                )
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"donor did not publish within 300 s:\n{self._tail()}")
            time.sleep(0.2)

    def _tail(self, n: int = 40) -> str:
        self._log.flush()
        with open(self.log_path) as f:
            return "".join(f.readlines()[-n:])

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # Last resort only: inside a Slurm step this aborts the step.
                self.proc.kill()
                self.proc.wait(timeout=60)
        self._log.close()


# Per transport: the rendezvous directory of the module's donor, the donor, and why it could not start. A transport's
# donor starts with the first test that reads its directory, so that a run selecting one transport ("-k fabric")
# starts only that donor.
_module_dirs: "dict[str, str]" = {}
_module_donors: "dict[str, _Donor]" = {}
_module_donor_errors: "dict[str, BaseException]" = {}


def _module_dir(transport: str) -> str:
    if transport in _module_donor_errors:
        raise RuntimeError(
            f"the module's {transport} donor failed to start"
        ) from _module_donor_errors[transport]
    if transport not in _module_donors:
        directory = _module_dirs.setdefault(
            transport, tempfile.mkdtemp(prefix=f"kvcm2_remote_{transport}_")
        )
        try:
            _module_donors[transport] = _Donor(directory, _DONOR_BYTES, transport)
        except BaseException as error:
            _module_donor_errors[transport] = error
            raise
    return _module_dirs[transport]


def tearDownModule() -> None:
    for donor in _module_donors.values():
        donor.stop()
    _module_donors.clear()
    _module_donor_errors.clear()
    for directory in _module_dirs.values():
        shutil.rmtree(directory, ignore_errors=True)
    _module_dirs.clear()


class _Transport:
    """A test class's transport (set per class by _per_transport), its module directory and its remote tier."""

    transport: str = "nixl"

    def setUp(self) -> None:
        if self.transport not in _TRANSPORTS:
            self.skipTest(f"the {self.transport} transport is unavailable here")
        super().setUp()

    @property
    def module_dir(self) -> str:
        return _module_dir(self.transport)

    def remote_tier(
        self,
        quota: int,
        directory: "str | None" = None,
        transfer_timeout_ms: int = 30_000,
        pool_ratio: "list[float] | None" = None,
        wait_timeout_ms: int = 60_000,
        expected_donors: int = 0,
    ):
        return RemoteCacheTierConfig(
            quota=quota,
            path=directory or self.module_dir,
            wait_timeout_ms=wait_timeout_ms,
            expected_donors=expected_donors,
            pool_ratio=pool_ratio or [],
            transfer=RemoteTransferConfig(
                transport=RemoteTransport[self.transport.upper()], timeout_ms=transfer_timeout_ms
            ),
        )


@_per_transport
class TestRemoteMemDonor(_Transport, unittest.TestCase):
    def test_donor_publishes_whole_chunks_and_withdraws_them(self) -> None:
        directory = tempfile.mkdtemp(prefix="kvcm2_donor_")
        donor = None
        try:
            donor = _Donor(directory, (64 << 20) + (3 << 20), self.transport, chunk_bytes=16 << 20)
            meta = open(os.path.join(directory, f"donor.{donor.name}.meta")).read().split("\n")
            fields = dict(line.split(" ", 1) for line in meta if line)
            self.assertEqual(fields["version"], "1")
            self.assertEqual(fields["name"], donor.name)
            self.assertEqual(int(fields["chunk_bytes"]), 16 << 20)
            base_address, nbytes = (int(x) for x in fields["segment"].split())
            self.assertEqual(nbytes, 64 << 20)  # the partial fifth chunk is not donated
            if self.transport == "fabric":
                # Each chunk is its own fabric allocation: one 64-byte handle per chunk, in chunk order.
                self.assertEqual(fields.get("transport"), "fabric")
                self.assertEqual(base_address, 0)
                self.assertEqual(
                    os.path.getsize(os.path.join(directory, f"donor.{donor.name}.fabric")), 4 * 64
                )
            else:
                self.assertGreater(base_address, 0)
                self.assertGreater(
                    os.path.getsize(os.path.join(directory, f"donor.{donor.name}.desc")), 0
                )
            donor.stop()
            self.assertEqual(glob.glob(os.path.join(directory, f"donor.{donor.name}.*")), [])
        finally:
            if donor is not None:
                donor.stop()  # a no-op once stopped
            shutil.rmtree(directory, ignore_errors=True)


@_per_transport
class TestRemoteLevelRoundTrip(_Transport, unittest.TestCase):
    """Attention pages and recurrent state go to the remote level and come back bit-exact.

    FakeEngine checks every reused page and the SSM slot.
    """

    tokens_per_block = 32
    buf_size = 256 << 10  # per block per buffer: a block of 2 attention layers x (K, V) is 1 MiB

    def setUp(self) -> None:
        super().setUp()
        base.init_cuda_once()
        self._token_ids = itertools.count(1_000_000 * (os.getpid() % 1000))
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            try:
                self.manager.shutdown()
            finally:
                del self.manager

    def _config(
        self,
        with_host: bool,
        directory: "str | None" = None,
        transfer_timeout_ms: int = 30_000,
        remote_quota: int = 512 << 20,
        wait_timeout_ms: int = 60_000,
        expected_donors: int = 0,
        gpu_quota: int = 32 << 20,
    ):
        layers = []
        for lid in range(2):
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=self.buf_size),
                        BufferConfig(role=DataRole("value"), size=self.buf_size),
                    ],
                )
            )
        for lid in range(2, 4):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[BufferConfig(role=DataRole("ssm_state"), size=self.buf_size)],
                )
            )
        tiers = [GpuCacheTierConfig(quota=gpu_quota)]
        if with_host:
            tiers.append(HostCacheTierConfig(quota=16 << 20))
        tiers.append(
            self.remote_tier(
                remote_quota,
                directory,
                transfer_timeout_ms,
                wait_timeout_ms=wait_timeout_ms,
                expected_donors=expected_donors,
            )
        )
        return KVCacheManagerConfig(
            tokens_per_block=self.tokens_per_block,
            cache_tiers=tiers,
            layers=layers,
            enable_partial_reuse=False,
            commit_min_snapshot=True,
        )

    def _prompt(self, length: int):
        return [TokenId(next(self._token_ids)) for _ in range(length)]

    def _prefill(self, prompt, stream) -> None:
        kv = self.manager.create_kv_cache()
        self.assertTrue(kv.resume(stream))
        kv.capacity = len(prompt)
        kv.history_length = len(prompt)
        self.engine.execute([Step(kv, prompt, [])], stream)
        kv.commit(prompt)
        kv.stop_committing()
        kv.close()

    def _residency(self, prompt) -> list:
        kv = self.manager.create_kv_cache(input_tokens=prompt)
        try:
            return list(kv.cached_tokens_by_level)
        finally:
            kv.close()

    def _push_to_last_level(self, prompt, stream) -> list:
        """Prefill unrelated prompts until every cached token of `prompt` sits on the last level."""
        last = len(self.cfg.cache_tiers) - 1
        for _ in range(64):
            levels = self._residency(prompt)
            if sum(levels) > 0 and levels[last] == sum(levels):
                return levels
            self._prefill(self._prompt(256), stream)
        self.fail(f"pages never reached level {last}: {self._residency(prompt)}")

    def _reuse_and_check(self, prompt, stream) -> list:
        kv = self.manager.create_kv_cache(input_tokens=prompt)
        levels = list(kv.cached_tokens_by_level)
        self.assertTrue(kv.resume(stream))
        kv.capacity = len(prompt)
        kv.history_length = len(prompt)
        # Reads every reused attention page and the SSM slot and compares them with what the prefill wrote.
        self.engine.execute([Step(kv, [], prompt)], stream)
        history = list(prompt)
        for _ in range(4):
            kv.capacity = len(history) + 1
            token = TokenId(next(self._token_ids))
            self.engine.execute([Step(kv, [token], history)], stream)
            history.append(token)
            kv.history_length = len(history)
        kv.close()
        return levels

    def _round_trip(self, with_host: bool) -> None:
        self.cfg = self._config(with_host)
        self.manager = KVCacheManager(self.cfg)
        self.engine = FakeEngine(self.cfg)
        stream_holder = base.CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        self.assertTrue(
            glob.glob(os.path.join(self.module_dir, "claim.*")), "no remote chunk was claimed"
        )

        prompt = self._prompt(4 * self.tokens_per_block)
        self._prefill(prompt, stream)
        before = self._push_to_last_level(prompt, stream)
        self.assertGreater(before[-1], 0)
        levels = self._reuse_and_check(prompt, stream)
        self.assertEqual(levels, before)
        # The pages were promoted on reuse: the prompt is cached on the GPU now.
        after = self._residency(prompt)
        self.assertGreater(after[0], 0, after)

    def test_round_trip_through_host_and_remote(self) -> None:
        """GPU -> host -> remote on eviction (codec encode, host-to-remote WRITE), remote -> GPU on reuse."""
        self._round_trip(with_host=True)

    def test_round_trip_without_host(self) -> None:
        """GPU -> remote and back directly (encode into staging + WRITE, READ + decode)."""
        self._round_trip(with_host=False)

    def test_rebase_onto_remote_copies_keeps_the_fresh_pages(self) -> None:
        """Rebasing onto remote copies installs the request's own GPU pages, and reuse reads them correctly.

        The request computes a prompt whose blocks the tree holds on the remote level (it did not reuse them).
        """
        self.cfg = self._config(with_host=True)
        self.manager = KVCacheManager(self.cfg)
        self.engine = FakeEngine(self.cfg)
        stream_holder = base.CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = self._prompt(4 * self.tokens_per_block)
        self._prefill(prompt, stream)
        self._push_to_last_level(prompt, stream)
        # Same tokens, no reuse: committing meets the tree's full blocks, whose attention pages are remote.
        self._prefill(prompt, stream)
        # Still reusable in full. (The level reported follows the SSM snapshot as well, which may sit on the host.)
        levels = self._residency(prompt)
        self.assertEqual(sum(levels), len(prompt), levels)
        # FakeEngine reads every reused page (now the second request's attention pages) and the SSM slot.
        self._reuse_and_check(prompt, stream)

    def test_rebase_onto_remote_copies_reads_nothing(self) -> None:
        """The same rebase with the donor gone: moving a remote copy back would fail the commit."""
        if self.transport == "fabric":
            self.skipTest(_NIXL_DEATH_ONLY)
        directory = tempfile.mkdtemp(prefix="kvcm2_rebase_")
        donor = _Donor(directory, 1 << 30, self.transport)
        try:
            self.cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=10_000)
            self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            prompt = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            self._push_to_last_level(prompt, stream)
            # Take the second request's GPU pages while the donor still serves the evictions that frees.
            kv = self.manager.create_kv_cache()
            self.assertTrue(kv.resume(stream))
            kv.capacity = len(prompt)
            kv.history_length = len(prompt)
            _sync_device()
            donor.stop()
            self.engine.execute([Step(kv, prompt, [])], stream)
            # Rebasing must not move the remote copies back to the GPU here: making room for them would evict to the
            # remote level and read the copies from it, so the commit would fail on a WRITE to the gone donor.
            kv.commit(prompt)
            kv.stop_committing()
            kv.close()
            del kv
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                try:
                    self.manager.shutdown()
                except Exception:
                    pass
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def _attention_layers(self, window: "int | None" = None) -> list:
        """Two attention layers; with `window`, the second one slides over that many tokens."""
        return [
            AttentionLayerConfig(
                layer_id=LayerId(lid),
                buffers=[
                    BufferConfig(role=DataRole("key"), size=self.buf_size),
                    BufferConfig(role=DataRole("value"), size=self.buf_size),
                ],
                sliding_window_size=window if lid == 1 else None,
            )
            for lid in range(2)
        ]

    def test_rebase_leaves_the_remote_copies_of_stale_sliding_window_blocks_unread(self) -> None:
        """Rebasing a stale sliding-window block onto the tree's remote copy reads nothing.

        Its page is held only to be committed: the rebase drops it instead of reading that copy back, which with the
        donor gone would fail the commit.
        """
        if self.transport == "fabric":
            self.skipTest(_NIXL_DEATH_ONLY)
        directory = tempfile.mkdtemp(prefix="kvcm2_swa_")
        donor = _Donor(directory, 1 << 30, self.transport)
        try:
            cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=10_000)
            # All-reusable commits: the sliding layer's stale blocks keep their pages (held) until committed.
            self.cfg = replace(
                cfg,
                layers=self._attention_layers(window=self.tokens_per_block),
                commit_min_snapshot=False,
            )
            self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            prompt = self._prompt(4 * self.tokens_per_block)

            def compute(tokens: list, stop_donor: bool = False) -> None:
                kv = self.manager.create_kv_cache()
                self.assertTrue(kv.resume(stream))
                kv.capacity = len(tokens)
                if stop_donor:
                    _sync_device()
                    donor.stop()
                self.engine.execute([Step(kv, tokens, [])], stream)
                # The window moves past the sliding layer's first blocks: their pages are held for the commit only.
                kv.history_length = len(tokens)
                kv.commit(tokens)
                kv.stop_committing()
                kv.close()

            # The suite's prefill sets the history before computing, which a sliding window cannot take.
            self._prefill = lambda tokens, _stream: compute(tokens)
            compute(prompt)
            self._push_to_last_level(prompt, stream)
            compute(prompt, stop_donor=True)
            self.assertIsNone(base.poison_reason())
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                with contextlib.suppress(Exception):
                    self.manager.shutdown()
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_page_that_supersedes_a_remote_copy_keeps_its_retention(self) -> None:
        """The committed page takes the superseded copy's retention priority, as reusing that copy would keep it."""
        cfg = self._config(with_host=True)
        self.cfg = replace(cfg, layers=self._attention_layers())
        self.manager = KVCacheManager(self.cfg)
        self.engine = FakeEngine(self.cfg)
        stream_holder = base.CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = self._prompt(4 * self.tokens_per_block)
        # Below the default (35) and above the cold-tier offload threshold (30), so that it reaches the remote level.
        retention = base.KvCacheRetentionConfig(
            [base.KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, 31, None)]
        )
        kv = self.manager.create_kv_cache(
            expected_prompt_length=len(prompt), kv_cache_retention_config=retention
        )
        self.assertTrue(kv.resume(stream))
        kv.capacity = len(prompt)
        kv.history_length = len(prompt)
        self.engine.execute([Step(kv, prompt, [])], stream)
        kv.commit(prompt)
        kv.stop_committing()
        kv.close()
        self._push_to_last_level(prompt, stream)
        # A default-priority request computes the same prompt and supersedes the remote copies with its pages.
        kv = self.manager.create_kv_cache()
        self.assertTrue(kv.resume(stream))
        kv.capacity = len(prompt)
        kv.history_length = len(prompt)
        self.engine.execute([Step(kv, prompt, [])], stream)
        kv.commit(prompt)
        self.assertEqual(kv.get_page_priorities(0), [31] * 4)
        kv.stop_committing()
        kv.close()

    def test_dead_donor_fails_the_promotion_instead_of_reusing(self) -> None:
        """The promotion itself raises."""
        if self.transport == "fabric":
            self.skipTest(_NIXL_DEATH_ONLY)
        directory = tempfile.mkdtemp(prefix="kvcm2_dead_")
        donor = _Donor(directory, 1 << 30, self.transport)
        try:
            self.cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=10_000)
            self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            prompt = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            self._push_to_last_level(prompt, stream)
            # The donor goes away with its memory registration and agent (its files go too; this client has loaded
            # it already), so the next READ of the prompt's pages has nowhere to go.
            donor.stop()
            kv = self.manager.create_kv_cache(input_tokens=prompt)
            # Its own class, not RuntimeError: callers tell a retired remote level from any other failure.
            with self.assertRaisesRegex(RemoteMemError, "remote level"):
                kv.resume(stream)
            try:
                kv.close()
            except Exception:
                pass
            del kv
            # The failed resume gave back the GPU slots it had taken for the promotion: with no cache open, every
            # GPU slot is free or evictable.
            gpu = cast(CacheLevel, CacheLevel(0))
            for pool in self.manager.get_storage_statistics(gpu):
                self.assertEqual(pool.available, pool.total)
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                try:
                    self.manager.shutdown()
                except Exception:
                    pass
                del self.manager
            # Fail-stop: after a failed transfer the manager is not expected to tear down cleanly. Collect what is
            # left now, so a destructor's poison lands here and is cleared, not in the next test.
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_failed_level_drops_its_pages_and_takes_no_more(self) -> None:
        """After a failed transfer the level is retired instead of failing every later request.

        Its pages leave the tree, and evictions go past it (here: they drop, since it is the last level).
        """
        if self.transport == "fabric":
            self.skipTest(_NIXL_DEATH_ONLY)
        directory = tempfile.mkdtemp(prefix="kvcm2_retired_")
        donor = _Donor(directory, 1 << 30, self.transport)
        try:
            self.cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=10_000)
            self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            prompt = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            self._push_to_last_level(prompt, stream)
            donor.stop()
            kv = self.manager.create_kv_cache(input_tokens=prompt)
            with self.assertRaisesRegex(Exception, "remote level"):
                kv.resume(stream)
            kv.close()
            del kv
            self.assertEqual(self._residency(prompt)[-1], 0, "the failed level still offers pages")
            # Prefills evict through the host level and past the failed one; a fresh prompt round-trips.
            for _ in range(8):
                self._prefill(self._prompt(256), stream)
            fresh = self._prompt(2 * self.tokens_per_block)
            self._prefill(fresh, stream)
            self._reuse_and_check(fresh, stream)
            self.assertEqual(self._residency(prompt)[-1], 0)
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                try:
                    self.manager.shutdown()
                except Exception:
                    pass
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def _with_dying_donor(self, prefix: str, body, attention_only: bool = False) -> None:
        """Run body(stream, donor) against a remote level whose donor the body stops; the manager must stay usable.

        attention_only: two attention layers, 8 MiB GPU and host tiers (8 blocks each), so that every page is tracked.
        """
        if self.transport == "fabric":
            self.skipTest(_NIXL_DEATH_ONLY)
        directory = tempfile.mkdtemp(prefix=prefix)
        donor = _Donor(directory, 1 << 30, self.transport)
        try:
            self.cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=10_000)
            if attention_only:
                tiers = list(self.cfg.cache_tiers)
                tiers[0] = GpuCacheTierConfig(quota=8 << 20)
                tiers[1] = HostCacheTierConfig(quota=8 << 20)
                self.cfg = replace(self.cfg, cache_tiers=tiers, layers=self.cfg.layers[:2])
            self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            body(cast(CudaStream, stream_holder.handle), donor)
            self.assertIsNone(base.poison_reason())
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                with contextlib.suppress(Exception):
                    self.manager.shutdown()
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_failed_eviction_write_retires_the_level_instead_of_failing_the_request(self) -> None:
        """A WRITE to a gone donor while making room retires the level instead of failing the request.

        The eviction goes past the level (here: drops, as it is the last level), so the request still gets its room.
        """

        def body(stream, donor) -> None:
            prompt = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            self._push_to_last_level(prompt, stream)
            donor.stop()
            self.assertFalse(self.manager.any_remote_level_failed)
            # The host tier is full: every prefill evicts toward the remote level.
            for _ in range(8):
                self._prefill(self._prompt(256), stream)
            self.assertTrue(self.manager.any_remote_level_failed)
            self.assertEqual(self._residency(prompt)[-1], 0, "the failed level still offers pages")
            fresh = self._prompt(2 * self.tokens_per_block)
            self._prefill(fresh, stream)
            self._reuse_and_check(fresh, stream)

        self._with_dying_donor("kvcm2_write_", body)

    def test_pages_a_request_holds_on_a_failed_level_leave_the_reuse_tree(self) -> None:
        """Pages a suspended request holds on a failed remote level are matched by no new request.

        Only the holder's resume fails; the prompt is computed again for everyone else.
        """

        def body(stream, donor) -> None:
            prompt = self._prompt(4 * self.tokens_per_block)
            holder = self.manager.create_kv_cache()
            self.assertTrue(holder.resume(stream))
            holder.capacity = len(prompt)
            holder.history_length = len(prompt)
            self.engine.execute([Step(holder, prompt, [])], stream)
            holder.commit(prompt)
            holder.suspend()
            self._push_to_last_level(prompt, stream)
            donor.stop()
            for _ in range(2):
                with self.assertRaisesRegex(RemoteMemError, "remote level"):
                    holder.resume(stream)
            self.assertEqual(
                self._residency(prompt)[-1], 0, "a new request matched the failed level's pages"
            )
            self._prefill(prompt, stream)
            self._reuse_and_check(prompt, stream)
            holder.close()
            del holder
            self._reuse_and_check(prompt, stream)

        self._with_dying_donor("kvcm2_held_", body)

    def test_a_probe_after_the_failure_counts_no_page_of_the_failed_level(self) -> None:
        """Probes stop counting a failed level's pages as soon as the failure happened.

        Schedulers and the ADP router call probe_reuse and probe_first_new_block_key before any allocation.
        """

        def body(stream, donor) -> None:
            # `prompt` is cached on the remote level only (droppable); `held` is held there by a suspended request.
            prompt = self._prompt(4 * self.tokens_per_block)
            held = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            holder = self.manager.create_kv_cache()
            self.assertTrue(holder.resume(stream))
            holder.capacity = len(held)
            holder.history_length = len(held)
            self.engine.execute([Step(holder, held, [])], stream)
            holder.commit(held)
            holder.suspend()
            self._push_to_last_level(prompt, stream)
            self._push_to_last_level(held, stream)
            # Free GPU slots (an uncommitted cache's), so that the failing resume evicts nothing: only its READ fails.
            spacer = self.manager.create_kv_cache()
            self.assertTrue(spacer.resume(stream))
            spacer.capacity = 12 * self.tokens_per_block
            spacer.close()
            self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), len(prompt))
            _sync_device()
            donor.stop()
            with self.assertRaisesRegex(RemoteMemError, "remote level"):
                holder.resume(stream)
            # No create_kv_cache(input_tokens=...) in between, which would drop and unlink the level's pages itself.
            probed = self.manager.probe_reuse(input_tokens=prompt)
            new_key = self.manager.probe_first_new_block_key(input_tokens=prompt)
            self.assertEqual(probed, 0)
            self.assertIsNotNone(new_key)
            self.assertEqual(sum(self._residency(prompt)), 0)
            holder.close()
            del holder

        self._with_dying_donor("kvcm2_probe_", body)

    def test_a_failed_eviction_write_still_offloads_to_the_host(self) -> None:
        """The GPU pages a failed host -> remote WRITE was making room for reach the host on the retry.

        The failed attempt had accepted them for migration; they must survive its unwinding so that its rollback puts
        them back, and the retry past the failed level then offloads them to the host instead of losing them.
        """

        def body(stream, donor) -> None:
            older = self._prompt(2 * self.tokens_per_block)
            self._prefill(older, stream)
            # Two-block prompts until the GPU and host tiers hold only these, the remote level the older ones.
            recent = [self._prompt(2 * self.tokens_per_block) for _ in range(16)]
            for prompt in recent:
                self._prefill(prompt, stream)
            levels = {i: self._residency(prompt) for i, prompt in enumerate(recent)}
            on_gpu = [i for i, lv in levels.items() if lv[0] == len(recent[i])]
            self.assertTrue(on_gpu, levels)
            self.assertGreater(
                self._residency(older)[-1], 0, "the older prompt is not on the remote level"
            )
            _sync_device()
            donor.stop()
            # Needs GPU room: GPU -> host, whose own eviction WRITEs to the dead donor and fails the level.
            self._prefill(self._prompt(2 * self.tokens_per_block), stream)
            self.assertEqual(
                self._residency(older)[-1], 0, "the remote level did not fail in this prefill"
            )
            # The oldest GPU prompt left the GPU for the host; no GPU prompt lost a cached token.
            after = {i: self._residency(recent[i]) for i in on_gpu}
            self.assertEqual(
                {i: sum(lv) for i, lv in after.items()}, {i: len(recent[i]) for i in on_gpu}
            )
            self.assertGreater(after[on_gpu[0]][1], 0, after)

        self._with_dying_donor("kvcm2_offload_", body, attention_only=True)

    def test_a_transfer_past_its_deadline_keeps_its_staging_until_it_quiesces(self) -> None:
        """A READ past its deadline keeps its staging lease out of use until it concludes, then is reaped.

        The READ fails the level but is not released while in progress.

        Over UCX's TCP transport a one-sided READ needs the donor's progress thread, so a stopped (SIGSTOP) donor holds
        the request in progress until it is continued.
        """
        if self.transport == "fabric":
            self.skipTest(
                "fabric copies are stream-ordered: their staging is retired on the stream"
            )
        directory = tempfile.mkdtemp(prefix="kvcm2_unquiesced_")
        # cuda_copy only because NIXL refuses a UCX_TLS without CUDA support on a GPU node; it moves no host memory.
        tcp = {"UCX_TLS": "tcp,cuda_copy"}
        donor = _Donor(directory, 1 << 30, self.transport, env=tcp)
        stopped = False
        try:
            self.cfg = self._config(with_host=True, directory=directory, transfer_timeout_ms=2_000)
            # The client's agent reads UCX_TLS when it is created, with the manager.
            with mock.patch.dict(os.environ, tcp):
                self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            prompt = self._prompt(4 * self.tokens_per_block)
            self._prefill(prompt, stream)
            self._push_to_last_level(prompt, stream)
            self.assertEqual(_introspection.remote_unquiesced_transfers(self.manager), 0)
            os.kill(donor.proc.pid, signal.SIGSTOP)
            stopped = True
            kv = self.manager.create_kv_cache(input_tokens=prompt)
            with self.assertRaisesRegex(RemoteMemError, "timed out"):
                kv.resume(stream)
            # The request is still in progress: its staging stays out of use.
            self.assertEqual(_introspection.remote_unquiesced_transfers(self.manager), 1)
            os.kill(donor.proc.pid, signal.SIGCONT)
            stopped = False
            deadline = time.monotonic() + 60
            while (
                _introspection.remote_unquiesced_transfers(self.manager)
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            self.assertEqual(_introspection.remote_unquiesced_transfers(self.manager), 0)
            with contextlib.suppress(Exception):
                kv.close()
            del kv
        finally:
            if stopped:
                os.kill(donor.proc.pid, signal.SIGCONT)
            donor.stop()
            if hasattr(self, "manager"):
                with contextlib.suppress(Exception):
                    self.manager.shutdown()
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_stalled_transfer_holding_all_staging_fails_later_resumes_with_remote_mem_error(
        self,
    ) -> None:
        """A READ past its deadline that holds the whole page-staging ring fails later remote resumes as RemoteMemError.

        The level is checked before staging is leased: otherwise the next resume of a request holding pages on it fails
        to lease staging with a plain error, which the scheduler does not take for a lost remote level.
        """
        if self.transport == "fabric":
            self.skipTest(
                "fabric copies are stream-ordered: their staging is retired on the stream"
            )
        # 16 MiB attention pages: the 64 MiB ring is 4 of them, and the 8-block READ below asks for all of it.
        self.buf_size = 4 << 20
        directory = tempfile.mkdtemp(prefix="kvcm2_ring_")
        tcp = {"UCX_TLS": "tcp,cuda_copy"}
        # Room for both prompts and the pushes on the remote level, so that neither is evicted from it; chunks hold
        # whole 16 MiB pages.
        donor = _Donor(directory, 2 << 30, self.transport, chunk_bytes=64 << 20, env=tcp)
        stopped = False
        kvs = []
        try:
            self.cfg = self._config(
                with_host=False,
                directory=directory,
                transfer_timeout_ms=2_000,
                remote_quota=1 << 30,
                gpu_quota=256 << 20,
            )
            with mock.patch.dict(os.environ, tcp):
                self.manager = KVCacheManager(self.cfg)
            self.engine = FakeEngine(self.cfg)
            stream_holder = base.CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)
            first = self._prompt(8 * self.tokens_per_block)
            second = self._prompt(4 * self.tokens_per_block)
            self._prefill(first, stream)
            self._prefill(second, stream)
            self._push_to_last_level(first, stream)
            self._push_to_last_level(second, stream)
            self.assertEqual(self._residency(first)[-1], len(first))
            # Room on the GPU, so that the first transfer to stall is the 8-page READ, which leases the whole ring,
            # rather than an eviction WRITE of a few pages.
            self.assertTrue(self.manager.resize(cast(CacheLevel, CacheLevel(0)), 1 << 30))
            # Both match the level's pages before it fails.
            kvs = [self.manager.create_kv_cache(input_tokens=p) for p in (first, second)]
            os.kill(donor.proc.pid, signal.SIGSTOP)
            stopped = True
            with self.assertRaises(RemoteMemError):
                kvs[0].resume(stream)
            self.assertGreaterEqual(_introspection.remote_unquiesced_transfers(self.manager), 1)
            # Not a staging-lease failure: the scheduler re-matches or recomputes on RemoteMemError only.
            with self.assertRaisesRegex(RemoteMemError, "remote level"):
                kvs[1].resume(stream)
        finally:
            if stopped:
                os.kill(donor.proc.pid, signal.SIGCONT)
            if hasattr(self, "manager"):
                deadline = time.monotonic() + 60
                while (
                    _introspection.remote_unquiesced_transfers(self.manager)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
            for kv in kvs:
                with contextlib.suppress(Exception):
                    kv.close()
            kvs.clear()
            donor.stop()
            if hasattr(self, "manager"):
                with contextlib.suppress(Exception):
                    self.manager.shutdown()
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_held_pages_stop_at_the_host_level_above_a_failed_remote_level(self) -> None:
        """Once the trailing remote level failed, the host level is the last one.

        Held pages falling to it are kept, and the droppable pages it evicts are counted as dropped. Treated as a
        middle level, the host let pages fall to the failed level: held ones, which nothing there can hold (every
        later allocation then failed with OutOfPagesError), and droppable ones, cleared there uncounted.
        """

        def body(stream, donor) -> None:
            lost = self._prompt(4 * self.tokens_per_block)
            self._prefill(lost, stream)
            self._push_to_last_level(lost, stream)
            donor.stop()
            kv = self.manager.create_kv_cache(input_tokens=lost)
            with self.assertRaisesRegex(RemoteMemError, "remote level"):
                kv.resume(stream)
            kv.close()
            del kv
            # Held pages fill the host level's attention pool but one slot, so when the last of them fall the host
            # has room for them and not for the droppable pages falling with them.
            groups = _introspection.level_pool_groups(self.manager, 1)
            attention_slots = max(groups, key=lambda group: group[1])[0]
            ssm_slots = min(groups, key=lambda group: group[1])[0]
            num_holders = max(1, min(4, ssm_slots - 1))
            held_blocks = (attention_slots - 1) // num_holders
            self.assertGreater(held_blocks, 0, groups)
            holders = []
            for _ in range(num_holders):
                prompt = self._prompt(held_blocks * self.tokens_per_block)
                holder = self.manager.create_kv_cache()
                self.assertTrue(holder.resume(stream))
                holder.capacity = len(prompt)
                holder.history_length = len(prompt)
                self.engine.execute([Step(holder, prompt, [])], stream)
                holder.commit(prompt)
                holder.suspend()
                holders.append((holder, prompt))
            # The GPU fills with droppable pages: the held ones go down to the host, the droppable ones drop.
            self.manager.get_and_reset_iteration_stats()
            for _ in range(8):
                self._prefill(self._prompt(256), stream)
            stats = self.manager.get_and_reset_iteration_stats()
            self.assertGreater(
                sum(delta.iter_host_dropped_blocks for delta in stats.values()),
                0,
                "pages the host level evicted were dropped uncounted",
            )
            for holder, prompt in holders:
                self.assertTrue(holder.resume(stream), "a held page fell past the host level")
                history = list(prompt)
                holder.capacity = len(history) + 1
                token = TokenId(next(self._token_ids))
                # Reads every page of the holder's history and compares it with what its prefill wrote.
                self.engine.execute([Step(holder, [token], history)], stream)
                holder.suspend()
            for holder, _ in holders:
                holder.close()
            holders.clear()

        self._with_dying_donor("kvcm2_held_host_", body)

    def test_withdrawn_donor_takes_no_new_claims(self) -> None:
        """Growing the level after its donor withdrew claims nothing from it, though the client knew its chunks."""
        directory = tempfile.mkdtemp(prefix="kvcm2_withdrawn_")
        donor = _Donor(directory, 128 << 20, self.transport)
        try:
            self.cfg = self._config(
                with_host=False, directory=directory, remote_quota=32 << 20, wait_timeout_ms=2_000
            )
            self.manager = KVCacheManager(self.cfg)
            claims = sorted(glob.glob(os.path.join(directory, f"claim.{donor.name}.*")))
            self.assertTrue(claims, "no chunk was claimed")
            donor.stop()
            last = cast(CacheLevel, CacheLevel(len(self.cfg.cache_tiers) - 1))
            self.assertFalse(self.manager.resize(last, 96 << 20), "grew on a withdrawn donor")
            self.assertEqual(
                sorted(glob.glob(os.path.join(directory, f"claim.{donor.name}.*"))), claims
            )
        finally:
            donor.stop()
            if hasattr(self, "manager"):
                self.manager.shutdown()
                del self.manager
            gc.collect()
            shutil.rmtree(directory, ignore_errors=True)

    def test_claims_stop_once_every_expected_donor_reported(self) -> None:
        """With every expected donor reported (published or failed) and no free chunk, a claim fails at once."""
        directory = tempfile.mkdtemp(prefix="kvcm2_reported_")
        donor = _Donor(directory, 128 << 20, self.transport)
        with open(os.path.join(directory, "donor.gone.failed"), "w") as f:
            f.write(f"version 1\nname gone\ntransport {self.transport}\n")
        try:
            self.cfg = self._config(
                with_host=False,
                directory=directory,
                wait_timeout_ms=600_000,
                expected_donors=2,
            )
            start = time.monotonic()
            with self.assertRaisesRegex(Exception, "expected donors reported"):
                self.manager = KVCacheManager(self.cfg)
            self.assertLess(time.monotonic() - start, 120)
        finally:
            donor.stop()
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_donor_reported_twice_counts_once(self) -> None:
        """A donor whose .ready and .failed are both listed counts once.

        A listing can see both when the donor exits between the two; a claim then keeps waiting for the other expected
        donor instead of failing as if all had reported.
        """
        directory = tempfile.mkdtemp(prefix="kvcm2_twice_")
        try:
            meta = f"version 1\nname x\nhost h\npid 1\nchunk_bytes {_CHUNK_BYTES}\ntransport {self.transport}\n"
            for suffix, text in (
                ("meta", meta + f"segment 4096 {4 * _CHUNK_BYTES}\n"),
                ("ready", ""),
                ("failed", f"version 1\nname x\ntransport {self.transport}\n"),
            ):
                with open(os.path.join(directory, f"donor.x.{suffix}"), "w") as f:
                    f.write(text)
            self.cfg = self._config(
                with_host=False, directory=directory, wait_timeout_ms=1_500, expected_donors=2
            )
            start = time.monotonic()
            with self.assertRaisesRegex(Exception, "after 1500 ms"):
                self.manager = KVCacheManager(self.cfg)
            self.assertGreaterEqual(time.monotonic() - start, 1.0)
        finally:
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_donor_found_full_earlier_is_probed_again_before_a_claim_fails(self) -> None:
        """A donor found full earlier is probed again before a claim fails at once.

        Every expected donor reported, but a peer may have returned its chunks since the donor was found full while a
        claim was served elsewhere, e.g. an abandoned construction.
        """
        directory = tempfile.mkdtemp(prefix="kvcm2_refull_")
        donors = [_Donor(directory, _CHUNK_BYTES, self.transport) for _ in range(2)]
        peer_claim = os.path.join(directory, f"claim.{donors[0].name}.0")
        last = cast(CacheLevel, CacheLevel(1))
        try:
            cfg = self._config(
                with_host=False,
                directory=directory,
                remote_quota=_CHUNK_BYTES,
                wait_timeout_ms=600_000,
                expected_donors=2,
            )
            self.cfg = replace(cfg, layers=self._attention_layers())
            # Whether the first claim passes over the full donor depends on where this client starts probing, which
            # its random name decides: repeat with new clients.
            for attempt in range(8):
                with open(peer_claim, "w") as f:
                    f.write("peer\n")
                self.manager = KVCacheManager(self.cfg)
                os.unlink(peer_claim)  # the peer lets go of its chunk
                self.assertTrue(self.manager.resize(last, 2 * _CHUNK_BYTES), f"attempt {attempt}")
                self.manager.shutdown()
                del self.manager
                gc.collect()
        finally:
            for donor in donors:
                donor.stop()
            if hasattr(self, "manager"):
                with contextlib.suppress(Exception):
                    self.manager.shutdown()
                del self.manager
            gc.collect()
            base.take_poison()
            shutil.rmtree(directory, ignore_errors=True)

    def test_crashed_donor_is_passed_over(self) -> None:
        """A donor that died without withdrawing (its files stay) gets no claims; the next donor serves them."""
        if self.transport != "fabric":
            self.skipTest("a dead NIXL donor shows at its first transfer, not at the claim")
        directory = tempfile.mkdtemp(prefix="kvcm2_crashed_")
        dead = _Donor(directory, 128 << 20, self.transport)
        files = sorted(glob.glob(os.path.join(directory, f"donor.{dead.name}.*")))
        contents = {path: open(path, "rb").read() for path in files}
        dead.stop()
        for path in sorted(files, key=lambda path: path.endswith(".ready")):  # .ready last
            with open(path, "wb") as f:
                f.write(contents[path])
        # The live donor publishes while the client already waits, so the client meets the dead one first.
        live = []
        thread = threading.Thread(
            target=lambda: live.append(_Donor(directory, 128 << 20, self.transport))
        )
        thread.start()
        try:
            self.cfg = self._config(
                with_host=False, directory=directory, remote_quota=32 << 20, wait_timeout_ms=300_000
            )
            self.manager = KVCacheManager(self.cfg)
            thread.join()
            self.assertEqual(glob.glob(os.path.join(directory, f"claim.{dead.name}.*")), [])
            self.assertTrue(glob.glob(os.path.join(directory, f"claim.{live[0].name}.*")))
        finally:
            thread.join()
            if hasattr(self, "manager"):
                self.manager.shutdown()
                del self.manager
            for donor in live:
                donor.stop()
            gc.collect()
            shutil.rmtree(directory, ignore_errors=True)


@_per_transport
class TestRemotePoolRatio(_Transport, unittest.TestCase):
    """RemoteCacheTierConfig.pool_ratio splits only the remote level's quota across layer groups.

    The model has an attention layer group (a 1 MiB block) and an SSM group (a 512 KiB state), at initial_pool_ratio
    0.5 / 0.5. Without the ratio, every cold level follows that split. With it, the remote level gives each layer
    group the requested share of its own bytes, and the GPU and host levels keep their split.
    """

    buf_size = 256 << 10
    remote_quota = 256 << 20

    def setUp(self) -> None:
        super().setUp()
        base.init_cuda_once()

    def _config(self, pool_ratio):
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(lid),
                buffers=[
                    BufferConfig(role=DataRole("key"), size=self.buf_size),
                    BufferConfig(role=DataRole("value"), size=self.buf_size),
                ],
            )
            for lid in range(2)
        ]
        layers += [
            SsmLayerConfig(
                layer_id=LayerId(lid),
                buffers=[BufferConfig(role=DataRole("ssm_state"), size=self.buf_size)],
            )
            for lid in range(2, 4)
        ]
        return KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[
                GpuCacheTierConfig(quota=32 << 20),
                HostCacheTierConfig(quota=64 << 20),
                self.remote_tier(self.remote_quota, pool_ratio=pool_ratio),
            ],
            layers=layers,
            enable_partial_reuse=False,
            commit_min_snapshot=True,
            initial_pool_ratio=[0.5, 0.5],
        )

    def _levels(self, pool_ratio: "list[float] | None"):
        """Build a manager whose remote level has `pool_ratio` (None: unset); return its levels' pool groups."""
        manager = KVCacheManager(self._config(pool_ratio))
        try:
            levels = [_introspection.level_pool_groups(manager, level) for level in range(3)]
            ssm = _introspection.ssm_life_cycle_id(manager)
        finally:
            manager.shutdown()
            gc.collect()
        self.assertIsNotNone(ssm)
        return levels, ssm

    @staticmethod
    def _bytes(groups) -> int:
        return sum(slots * slot_bytes for slots, slot_bytes, _ in groups)

    @classmethod
    def _share(cls, groups, life_cycle: int) -> float:
        """The share of a level's bytes held by the pool group of `life_cycle`."""
        held = sum(slots * slot_bytes for slots, slot_bytes, lcs in groups if life_cycle in lcs)
        return held / cls._bytes(groups)

    def test_unset_keeps_the_hot_split_on_every_cold_level(self) -> None:
        (_, host, remote), ssm = self._levels(None)
        self.assertAlmostEqual(self._share(remote, ssm), self._share(host, ssm), delta=0.05)
        self.assertGreater(self._share(remote, ssm), 0.3)

    def test_override_splits_only_the_remote_level(self) -> None:
        (gpu0, host0, remote0), ssm = self._levels(None)
        num_life_cycles = sum(len(lcs) for _, _, lcs in gpu0)
        weights = [9.0] * num_life_cycles
        weights[ssm] = 1.0
        target = 1 / (1 + 9 * (num_life_cycles - 1))
        (gpu1, host1, remote1), _ = self._levels(weights)
        self.assertEqual(gpu1, gpu0)
        self.assertEqual(host1, host0)
        self.assertAlmostEqual(self._share(remote1, ssm), target, delta=0.05)
        self.assertLess(self._share(remote1, ssm), self._share(remote0, ssm) - 0.2)
        # The same quota, only split differently.
        self.assertAlmostEqual(
            self._bytes(remote1), self._bytes(remote0), delta=0.05 * self.remote_quota
        )
        # Weights are normalized.
        (gpu2, host2, remote2), _ = self._levels([w / 10 for w in weights])
        self.assertEqual((gpu2, host2, remote2), (gpu1, host1, remote1))

    def test_malformed_values_fail_before_remote_memory_is_claimed(self) -> None:
        claims = sorted(glob.glob(os.path.join(self.module_dir, "claim.*")))
        for value in (
            [0.5],
            [0.5, 0.0],
            [0.5, -1.0],
            [float("nan"), 1.0],
            [float("inf"), 1.0],
            [1.0, 1.0, 1.0],
        ):
            # TLLM_CHECK, from the tier config (a bad entry) or the storage manager (a wrong entry count).
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "pool ratio"),
            ):
                self._levels(value)
        self.assertEqual(sorted(glob.glob(os.path.join(self.module_dir, "claim.*"))), claims)


@_per_transport
class TestRemoteTierCombinations(_Transport, unittest.TestCase):
    """A remote level stages pages through pinned host memory, so it cannot sit next to a file-backed disk level."""

    def _single_level_manager(self, directory: str):
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=DataRole("key"), size=64 << 10)],
            )
        ]
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[
                    GpuCacheTierConfig(quota=32 << 20),
                    self.remote_tier(64 << 20, directory=directory, wait_timeout_ms=5_000),
                ],
                layers=layers,
            )
        )

    def test_the_rendezvous_directory_must_be_this_users_private_directory(self) -> None:
        """The client reads donor files only through the directory it checked.

        Never through a symlink in its place, never from a directory others can write.
        """
        parent = tempfile.mkdtemp(prefix="kvcm2_pin_")
        try:
            link = os.path.join(parent, "link")
            os.symlink(self.module_dir, link)
            with self.assertRaisesRegex(Exception, "symlink"):
                self._single_level_manager(link)
            shared = os.path.join(parent, "shared")
            os.mkdir(shared)
            os.chmod(shared, 0o770)
            with self.assertRaisesRegex(Exception, "not writable by group or others"):
                self._single_level_manager(shared)
        finally:
            gc.collect()
            shutil.rmtree(parent, ignore_errors=True)

    def test_a_file_backed_disk_level_is_rejected(self) -> None:
        disk_dir = tempfile.mkdtemp(prefix="kvcm2_disk_")
        try:
            layers = [
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=DataRole("key"), size=64 << 10)],
                )
            ]
            with self.assertRaisesRegex(Exception, "file-backed disk level"):
                KVCacheManager(
                    KVCacheManagerConfig(
                        tokens_per_block=32,
                        cache_tiers=[
                            GpuCacheTierConfig(quota=32 << 20),
                            HostCacheTierConfig(quota=32 << 20),
                            DiskCacheTierConfig(quota=64 << 20, path=disk_dir),
                            self.remote_tier(64 << 20),
                        ],
                        layers=layers,
                    )
                )
        finally:
            gc.collect()
            shutil.rmtree(disk_dir, ignore_errors=True)


class TestRemoteInPlaceOfDisk(_Transport, base.TestKVCacheManagerV2):
    """The suite's cache-reuse tests and one shrink case, with the remote level in the disk level's place."""

    Request = base.TestNoBatching.Request
    new_request = base.TestNoBatching.new_request
    run_request = base.TestNoBatching.run_request
    run_naive = base.TestNoBatching.run_naive

    def prepare(
        self,
        gpu_quota,
        host_quota,
        disk_quota,
        num_layers,
        window_size,
        sink_tokens,
        tokens_per_block=32,
        kv_buf_size=8192,
        block_quant_buf_size=None,
    ) -> None:
        cfg = base.create_config(
            tokens_per_block,
            gpu_quota,
            host_quota,
            disk_quota,
            num_layers,
            window_size,
            sink_tokens,
            kv_buf_size,
            block_quant_buf_size,
        )
        cfg = replace(
            cfg,
            cache_tiers=[
                self.remote_tier(t.quota) if isinstance(t, DiskCacheTierConfig) else t
                for t in cfg.cache_tiers
            ],
        )
        self.cfg = cfg
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)


# The suite's cache-reuse tests, whose pages reach the disk level and come back (each generated name), and one
# shrink case; the base suite covers the rest of the manager.
_cases = sorted(
    name
    for name, test in vars(base.TestNoBatching).items()
    if name.startswith("test_cache_reuse") and callable(test)
)
_cases += sorted(
    name
    for name, test in vars(base.TestNoBatching).items()
    if name.startswith("test_shrink_capacity") and "truncates" not in name and callable(test)
)[:1]
for _name in _cases:
    setattr(TestRemoteInPlaceOfDisk, _name, getattr(base.TestNoBatching, _name))
del _name, _cases
TestRemoteInPlaceOfDisk = _per_transport(TestRemoteInPlaceOfDisk)
