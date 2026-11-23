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
"""Tests of the opt-in remote host KV tier (TLLM_KV_REMOTE_HOST_*).

The knob parsing, the handle-file protocol (lender lock, ready marker, importer claim), the
importer's and lender's refusals, the stats line, the host-tier collapse exclusion and the
several-lender import (indexed files, waiting and re-reading, checks, all-or-nothing claims and
their exit hooks, interleave, re-stamp with a fake chunk memory, one chunk's driver calls with a
fake driver, the per-pool-group spread line) run on the CPU. The cross-rank failure
sync needs the KV cache manager module (torch) but no GPU. The tests marked GPU build real KV cache
managers; the two-process test and the two-lender tests also need fabric host memory (HOST_NUMA +
FABRIC with an IMEX channel) and skip without it. The C++ level itself is covered by the
kvCacheManagerV2RemoteHostTierTest gtest.
"""

import ctypes
import json
import os
import queue
import socket
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from typing import List

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import host_tier_memory
from tensorrt_llm._torch.pyexecutor.kv_cache import remote_host_tier as rht

ALL_ENVS = (
    rht.LEND_GIB_ENV,
    rht.CHUNK_GIB_ENV,
    rht.TIER_ENV,
    rht.DIR_ENV,
    rht.WAIT_S_ENV,
    rht.EXPECT_GIB_ENV,
    rht.LENDER_INDEX_ENV,
    rht.NUM_LENDERS_ENV,
    rht.JOB_ENV,
)
MIB = 1 << 20
GIB = 1 << 30
JOB = "4242"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ALL_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rht, "_LENDER", None)
    monkeypatch.setattr(rht, "_MERGED", None)
    # Claims and lender files register exit hooks; keep them out of the test process.
    monkeypatch.setattr(rht.atexit, "register", lambda *args, **kwargs: None)


def _lender_identity(pid: int = 1234, host: str = "gen-node") -> dict:
    return {"role": "lender", "host": host, "rank": 5, "local_rank": 0, "pid": pid}


def _payload(
    rank: int = 0, num_chunks: int = 3, chunk_bytes: int = 2 * MIB, job: str = JOB
) -> dict:
    return {
        "format": rht.FORMAT_VERSION,
        "rank": rank,
        "job": job,
        "device": rank,
        "numa": rank // 2,
        "chunk_bytes": chunk_bytes,
        "nonce": 0x1234,
        "lender": _lender_identity(),
        "handles": [(bytes([i]) * rht.FABRIC_HANDLE_BYTES).hex() for i in range(num_chunks)],
    }


def _me(role: str = "importer", pid: int = None, host: str = None) -> rht.Identity:
    return rht.Identity(
        role=role,
        host=host if host is not None else socket.gethostname(),
        rank=0,
        local_rank=0,
        pid=pid if pid is not None else os.getpid(),
    )


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


# ---- knobs -----------------------------------------------------------------------------------------


def test_knobs_unset_mean_off():
    assert rht.parse_lend_config({}) is None
    assert rht.parse_tier_config({}) is None
    assert rht.parse_lend_config({rht.LEND_GIB_ENV: "0"}) is None
    assert rht.parse_tier_config({rht.TIER_ENV: "0"}) is None
    assert rht.parse_tier_config({rht.TIER_ENV: ""}) is None
    assert not rht.lend_requested({})
    assert not rht.tier_requested({rht.TIER_ENV: "0"})
    assert rht.lend_requested({rht.LEND_GIB_ENV: "220"})
    assert rht.tier_requested({rht.TIER_ENV: "1"})


def test_lend_config():
    config = rht.parse_lend_config({rht.LEND_GIB_ENV: "220", rht.DIR_ENV: "/logs/rh"})
    assert config == rht.LendConfig(lend_bytes=220 * GIB, chunk_bytes=GIB, directory="/logs/rh")
    assert config.num_chunks == 220
    # Rounded down to whole chunks.
    config = rht.parse_lend_config(
        {rht.LEND_GIB_ENV: "2.5", rht.CHUNK_GIB_ENV: "1", rht.DIR_ENV: "/d"}
    )
    assert config.lend_bytes == 2 * GIB
    with pytest.raises(ValueError, match=rht.DIR_ENV):
        rht.parse_lend_config({rht.LEND_GIB_ENV: "220"})
    with pytest.raises(ValueError):
        rht.parse_lend_config({rht.LEND_GIB_ENV: "lots", rht.DIR_ENV: "/d"})
    with pytest.raises(ValueError):
        rht.parse_lend_config({rht.LEND_GIB_ENV: "-1", rht.DIR_ENV: "/d"})
    with pytest.raises(ValueError, match="2 MiB"):
        rht.parse_lend_config(
            {rht.LEND_GIB_ENV: "1", rht.CHUNK_GIB_ENV: "0.0001", rht.DIR_ENV: "/d"}
        )
    with pytest.raises(ValueError, match="smaller than one chunk"):
        rht.parse_lend_config({rht.LEND_GIB_ENV: "0.5", rht.DIR_ENV: "/d"})


def test_tier_config():
    config = rht.parse_tier_config({rht.TIER_ENV: "1", rht.DIR_ENV: "/logs/rh"})
    assert config == rht.TierConfig(directory="/logs/rh", wait_s=rht.DEFAULT_WAIT_S)
    config = rht.parse_tier_config({rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.WAIT_S_ENV: "30"})
    assert config.wait_s == 30.0
    config = rht.parse_tier_config(
        {rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.EXPECT_GIB_ENV: "220"}
    )
    assert config.expect_bytes == 220 * GIB
    with pytest.raises(ValueError, match=rht.DIR_ENV):
        rht.parse_tier_config({rht.TIER_ENV: "1"})
    with pytest.raises(ValueError, match="use 1"):
        rht.parse_tier_config({rht.TIER_ENV: "2", rht.DIR_ENV: "/d"})
    with pytest.raises(ValueError):
        rht.parse_tier_config({rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.WAIT_S_ENV: "0"})
    with pytest.raises(ValueError, match=rht.EXPECT_GIB_ENV):
        rht.parse_tier_config({rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.EXPECT_GIB_ENV: "x"})


def test_job_tag_must_be_set_and_file_safe():
    assert rht.job_tag({rht.JOB_ENV: "123456"}) == "123456"
    with pytest.raises(ValueError, match=rht.JOB_ENV):
        rht.job_tag({})
    with pytest.raises(ValueError, match=rht.JOB_ENV):
        rht.job_tag({rht.JOB_ENV: "  "})
    with pytest.raises(ValueError, match="file name"):
        rht.job_tag({rht.JOB_ENV: "../x"})


# ---- handle files ----------------------------------------------------------------------------------


def test_publish_and_wait_round_trip(tmp_path):
    payload = _payload(rank=2, job="77")
    rht.publish(str(tmp_path), 2, payload, job="77")
    assert sorted(os.listdir(tmp_path)) == ["rank2.job77.handles.json", "rank2.job77.ready"]
    with open(rht.ready_path(str(tmp_path), 2, "77")) as f:
        assert json.load(f)["lender"] == payload["lender"]
    clock = FakeClock()
    got = rht.wait_for_payload(str(tmp_path), 2, 5.0, "77", clock=clock, sleep=clock.sleep)
    assert got == payload
    assert clock.now == 0.0


def test_wait_times_out_without_marker(tmp_path):
    clock = FakeClock()
    with pytest.raises(TimeoutError, match="lender rank 1"):
        rht.wait_for_payload(str(tmp_path), 1, 5.0, "77", clock=clock, sleep=clock.sleep)
    assert clock.now >= 5.0


def test_files_of_another_job_are_never_read(tmp_path):
    rht.publish(str(tmp_path), 0, _payload(job="old"), job="old")
    clock = FakeClock()
    with pytest.raises(TimeoutError):
        rht.wait_for_payload(str(tmp_path), 0, 3.0, "new", clock=clock, sleep=clock.sleep)


def test_wait_picks_up_a_late_lender(tmp_path):
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 2.0 and not os.path.exists(rht.ready_path(str(tmp_path), 0, "9")):
            rht.publish(str(tmp_path), 0, _payload(job="9"), job="9")

    got = rht.wait_for_payload(str(tmp_path), 0, 10.0, "9", clock=clock, sleep=sleep)
    assert got["rank"] == 0


def test_digest_mismatch_is_transient_until_the_deadline(tmp_path):
    rht.publish(str(tmp_path), 0, _payload(job="9"), job="9")
    with open(rht.handle_path(str(tmp_path), 0, "9"), "a") as f:
        f.write(" ")
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="does not match"):
        rht.wait_for_payload(str(tmp_path), 0, 3.0, "9", clock=clock, sleep=clock.sleep)
    assert clock.now >= 3.0

    # The lender re-publishes while the importer waits: the importer takes the new files.
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 2.0:
            rht.publish(str(tmp_path), 0, _payload(job="9", num_chunks=5), job="9")

    got = rht.wait_for_payload(str(tmp_path), 0, 10.0, "9", clock=clock, sleep=sleep)
    assert len(got["handles"]) == 5


def test_wait_rejects_bad_markers_and_malformed_files(tmp_path):
    directory = str(tmp_path)
    # A marker that is not a JSON object, or names another job, never becomes valid.
    with open(rht.ready_path(directory, 0, "9"), "w") as f:
        f.write("[1, 2]")
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="not a JSON object"):
        rht.wait_for_payload(directory, 0, 2.0, "9", clock=clock, sleep=clock.sleep)
    with open(rht.ready_path(directory, 0, "9"), "w") as f:
        json.dump({"job": "other", "format": rht.FORMAT_VERSION}, f)
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="names job 'other'"):
        rht.wait_for_payload(directory, 0, 2.0, "9", clock=clock, sleep=clock.sleep)

    # A file that matches its marker but is malformed raises at once.
    for bad in (
        dict(_payload(job="9"), handles=["00" * 10]),
        dict(_payload(job="9"), handles=["zz" * rht.FABRIC_HANDLE_BYTES]),
        _payload(rank=3, job="9"),
        dict(_payload(job="9"), job="8"),
        dict(_payload(job="9"), lender="gen-node"),
        dict(_payload(job="9"), nonce="0x1234"),
    ):
        rht.publish(directory, 0, bad, job="9")
        with pytest.raises(RuntimeError, match="malformed"):
            rht.wait_for_payload(directory, 0, 1.0, "9")

    marker = {"job": "9", "format": rht.FORMAT_VERSION + 1, "sha256": ""}
    with open(rht.ready_path(directory, 0, "9"), "w") as f:
        json.dump(marker, f)
    with pytest.raises(RuntimeError, match="handle format"):
        rht.wait_for_payload(directory, 0, 1.0, "9")


def test_publish_removes_the_old_marker_first(tmp_path, monkeypatch):
    directory = str(tmp_path)
    rht.publish(directory, 0, _payload(job="9"), job="9")
    real_write = rht._write_atomic

    def failing_write(path, data):
        if path.endswith(".handles.json"):
            raise OSError("disk full")
        real_write(path, data)

    monkeypatch.setattr(rht, "_write_atomic", failing_write)
    with pytest.raises(OSError):
        rht.publish(directory, 0, _payload(job="9", num_chunks=4), job="9")
    # No marker is left that names a handle file it may not match.
    assert not os.path.exists(rht.ready_path(directory, 0, "9"))


# ---- lender lock and importer claim ----------------------------------------------------------------


def test_second_lender_of_a_rank_is_refused(tmp_path):
    directory = str(tmp_path)
    first = _me("lender", pid=os.getppid(), host="gen-a")
    rht.acquire_lender_lock(directory, 0, "9", first)
    # Another host, or a live process of this host, holding the lock: refused.
    with pytest.raises(RuntimeError, match="one context and one generation instance"):
        rht.acquire_lender_lock(directory, 0, "9", _me("lender", host="gen-b"))
    os.unlink(rht.lender_lock_path(directory, 0, "9"))
    rht.acquire_lender_lock(directory, 0, "9", _me("lender", pid=os.getppid()))
    with pytest.raises(RuntimeError, match="held by"):
        rht.acquire_lender_lock(directory, 0, "9", _me("lender"))
    # Another rank or another job is independent.
    rht.acquire_lender_lock(directory, 1, "9", _me("lender"))
    rht.acquire_lender_lock(directory, 0, "10", _me("lender"))


def test_lender_lock_of_an_exited_process_on_this_host_is_replaced(tmp_path):
    directory = str(tmp_path)
    rht.acquire_lender_lock(directory, 0, "9", _me("lender", pid=_dead_pid()))
    rht.acquire_lender_lock(directory, 0, "9", _me("lender"))
    with open(rht.lender_lock_path(directory, 0, "9")) as f:
        assert json.load(f)["owner"]["pid"] == os.getpid()


def test_second_importer_of_a_lender_is_refused(tmp_path):
    directory = str(tmp_path)
    payload = _payload(job="9")
    path = rht.claim_lender(directory, 0, "9", payload, _me(pid=os.getppid()))
    assert path == rht.claim_path(directory, 0, "9", payload["nonce"])
    with open(path) as f:
        claim = json.load(f)
    assert claim["lender"] == payload["lender"]
    assert claim["owner"]["role"] == "importer"
    with pytest.raises(RuntimeError, match="held by"):
        rht.claim_lender(directory, 0, "9", payload, _me())
    with pytest.raises(RuntimeError, match="held by"):
        rht.claim_lender(directory, 0, "9", payload, _me(host="ctx-b"))
    # A new lender incarnation (another nonce) is a different claim.
    rht.claim_lender(directory, 0, "9", dict(payload, nonce=0x5678), _me())


def test_claim_of_an_exited_importer_on_this_host_is_replaced(tmp_path):
    directory = str(tmp_path)
    payload = _payload(job="9")
    rht.claim_lender(directory, 0, "9", payload, _me(pid=_dead_pid()))
    rht.claim_lender(directory, 0, "9", payload, _me())


# ---- importer refusals -----------------------------------------------------------------------------


def _build(**overrides):
    kwargs = dict(
        local_rank=0,
        is_draft=False,
        is_estimating=False,
        local_host_quota=280 * GIB,
        codec_configured=False,
        has_disk_tier=False,
    )
    kwargs.update(overrides)
    return rht.build_remote_tier(**kwargs)


def test_importer_off_when_unset(tmp_path):
    # Unset: nothing is read, whatever the other arguments say.
    assert _build(codec_configured=True, local_host_quota=0, has_disk_tier=True) is None


def test_importer_refusals(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.WAIT_S_ENV, "0.01")
    with pytest.raises(ValueError, match="cold-page codec"):
        _build(codec_configured=True)
    with pytest.raises(ValueError, match="local host tier"):
        _build(local_host_quota=0)
    with pytest.raises(ValueError, match="disk tier"):
        _build(has_disk_tier=True)
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    with pytest.raises(ValueError, match="different workers"):
        _build()
    monkeypatch.delenv(rht.LEND_GIB_ENV)
    # No job id: refused before anything is read.
    with pytest.raises(ValueError, match=rht.JOB_ENV):
        _build()
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    # Nothing published: the timeout raises; there is no fallback to the local tiers.
    with pytest.raises(TimeoutError):
        _build()


def test_importer_refuses_an_unexpected_lent_size(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setenv(rht.EXPECT_GIB_ENV, str(4 * 2 * MIB / GIB))
    rht.publish(str(tmp_path), 0, _payload(num_chunks=3), job=JOB)
    with pytest.raises(ValueError, match=rht.EXPECT_GIB_ENV):
        _build()
    # Refused before claiming: the lender stays free for a correctly configured importer.
    assert not any(name.endswith(".claim") for name in os.listdir(tmp_path))


def test_importer_refuses_a_claimed_lender(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    payload = _payload()
    rht.publish(str(tmp_path), 0, payload, job=JOB)
    rht.claim_lender(str(tmp_path), 0, JOB, payload, _me(host="other-ctx"))
    with pytest.raises(RuntimeError, match="held by"):
        _build()


def test_importer_skips_draft_and_estimation_managers(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    # Neither waits for handles (none are published) nor refuses.
    assert _build(is_draft=True, codec_configured=True) is None
    assert _build(is_estimating=True) is None


def test_importer_builds_the_tier(monkeypatch, tmp_path):
    pytest.importorskip("tensorrt_llm.bindings")
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setenv(rht.EXPECT_GIB_ENV, str(3 * 2 * MIB / GIB))
    payload = _payload(num_chunks=3, chunk_bytes=2 * MIB)
    rht.publish(str(tmp_path), 0, payload, job=JOB)
    tier = _build()
    assert tier.is_remote
    assert tier.quota == 3 * 2 * MIB
    assert tier.remote.chunk_bytes == 2 * MIB
    assert tier.remote.verify_nonce == payload["nonce"]
    assert [h.hex() for h in tier.remote.fabric_handles] == payload["handles"]
    tier.assert_valid()
    assert os.path.exists(rht.claim_path(str(tmp_path), 0, JOB, payload["nonce"]))


# ---- lender guards ---------------------------------------------------------------------------------


def test_lender_off_when_unset():
    def device_fn():
        raise AssertionError("the device must not be queried when lending is off")

    assert rht.maybe_lend(local_rank=0, device_fn=device_fn, is_draft=False) is None


def test_lender_skips_draft_and_refuses_both_roles(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    assert rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=True) is None
    monkeypatch.setenv(rht.TIER_ENV, "1")
    with pytest.raises(ValueError, match="different workers"):
        rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)


def test_lender_refuses_a_cold_page_codec_and_a_missing_job(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))

    def device_fn():
        raise AssertionError("nothing may be allocated before the refusal")

    with pytest.raises(ValueError, match=rht.JOB_ENV):
        rht.maybe_lend(local_rank=0, device_fn=device_fn, is_draft=False)
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    with pytest.raises(ValueError, match="cold-page codec"):
        rht.maybe_lend(local_rank=0, device_fn=device_fn, is_draft=False, codec_configured=True)
    assert os.listdir(tmp_path) == []


def test_lender_refuses_a_second_lender_before_allocating(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    rht.acquire_lender_lock(str(tmp_path), 0, JOB, _me("lender", host="other-gen"))

    def device_fn():
        raise AssertionError("nothing may be allocated before the refusal")

    with pytest.raises(RuntimeError, match="held by"):
        rht.maybe_lend(local_rank=0, device_fn=device_fn, is_draft=False)
    # The other lender's lock is untouched.
    assert os.path.exists(rht.lender_lock_path(str(tmp_path), 0, JOB))


class _FakeLender:
    instances: List["_FakeLender"] = []

    def __init__(self, config, rank, device):
        self.config = config
        self.rank = rank
        self.numa = 0
        self.lend_s = 0.0
        self.released = False
        _FakeLender.instances.append(self)

    def lend(self):
        return dict(_payload(rank=self.rank), nonce=99)

    def release(self):
        self.released = True


@pytest.fixture
def fake_lender(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setattr(rht, "RemoteHostLender", _FakeLender)
    _FakeLender.instances.clear()
    return str(tmp_path)


def test_lender_releases_its_chunks_when_publishing_fails(monkeypatch, fake_lender):
    def failing_publish(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(rht, "publish", failing_publish)
    with pytest.raises(OSError):
        rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False, global_rank=4)
    assert _FakeLender.instances[-1].released
    # No file of this lender is left behind, its lock included.
    assert os.listdir(fake_lender) == []
    assert rht._LENDER is None


def test_lender_publishes_its_identity(fake_lender):
    lender = rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False, global_rank=4)
    assert lender is _FakeLender.instances[-1] and not lender.released
    payload = rht.wait_for_payload(fake_lender, 0, 1.0, JOB)
    assert payload["job"] == JOB
    assert payload["lender"]["role"] == "lender"
    assert payload["lender"]["rank"] == 4
    assert payload["lender"]["pid"] == os.getpid()
    assert payload["lender"]["host"] == socket.gethostname()
    with open(rht.lender_lock_path(fake_lender, 0, JOB)) as f:
        assert json.load(f)["owner"] == payload["lender"]
    # Once per process.
    assert rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False) is lender
    # The exit hook removes the lender's files.
    rht._remove_published(fake_lender, 0, JOB)
    assert os.listdir(fake_lender) == []


def test_lender_lends_once_per_process(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    sentinel = object()
    monkeypatch.setattr(rht, "_LENDER", sentinel)
    assert rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False) is sentinel


# ---- several lenders: knobs and files --------------------------------------------------------------


def test_lender_index_knob():
    base = {rht.LEND_GIB_ENV: "1", rht.DIR_ENV: "/d"}
    assert rht.parse_lend_config(base).lender_index is None
    assert rht.parse_lend_config(dict(base, **{rht.LENDER_INDEX_ENV: " "})).lender_index is None
    # 0 is an index, not "off".
    assert rht.parse_lend_config(dict(base, **{rht.LENDER_INDEX_ENV: "0"})).lender_index == 0
    assert rht.parse_lend_config(dict(base, **{rht.LENDER_INDEX_ENV: "15"})).lender_index == 15
    for bad in ("x", "-1", "16", "1.5"):
        with pytest.raises(ValueError, match=rht.LENDER_INDEX_ENV):
            rht.parse_lend_config(dict(base, **{rht.LENDER_INDEX_ENV: bad}))
    # The index alone lends nothing.
    assert rht.parse_lend_config({rht.LENDER_INDEX_ENV: "1", rht.DIR_ENV: "/d"}) is None


def test_tier_config_num_lenders_and_expect_list():
    base = {rht.TIER_ENV: "1", rht.DIR_ENV: "/d"}
    config = rht.parse_tier_config(base)
    assert config.num_lenders is None and config.expect_bytes_per_lender is None
    config = rht.parse_tier_config(dict(base, **{rht.NUM_LENDERS_ENV: "2"}))
    assert config.num_lenders == 2
    assert config.expect_bytes is None and config.expect_bytes_per_lender is None
    config = rht.parse_tier_config(
        dict(base, **{rht.NUM_LENDERS_ENV: "2", rht.EXPECT_GIB_ENV: "180, 120"})
    )
    assert config.expect_bytes_per_lender == (180 * GIB, 120 * GIB)
    assert config.expect_bytes is None
    config = rht.parse_tier_config(
        dict(base, **{rht.NUM_LENDERS_ENV: "1", rht.EXPECT_GIB_ENV: "180"})
    )
    assert config.num_lenders == 1 and config.expect_bytes_per_lender == (180 * GIB,)
    # A single number with N >= 2 is ambiguous; a list must have exactly N entries.
    for expect in ("180", "180,120,60"):
        with pytest.raises(ValueError, match="2 comma-separated per-lender sizes"):
            rht.parse_tier_config(
                dict(base, **{rht.NUM_LENDERS_ENV: "2", rht.EXPECT_GIB_ENV: expect})
            )
    with pytest.raises(ValueError, match=rht.EXPECT_GIB_ENV):
        rht.parse_tier_config(dict(base, **{rht.NUM_LENDERS_ENV: "2", rht.EXPECT_GIB_ENV: "1,x"}))
    for bad in ("0", "17", "x"):
        with pytest.raises(ValueError, match=rht.NUM_LENDERS_ENV):
            rht.parse_tier_config(dict(base, **{rht.NUM_LENDERS_ENV: bad}))
    # Without NUM_LENDERS the expected size stays one number.
    with pytest.raises(ValueError, match=rht.EXPECT_GIB_ENV):
        rht.parse_tier_config(dict(base, **{rht.EXPECT_GIB_ENV: "180,120"}))


def test_file_names_with_and_without_a_lender_index():
    d = "/d"
    # Unset index: the single-lender names, unchanged.
    assert rht.handle_path(d, 1, "9") == "/d/rank1.job9.handles.json"
    assert rht.ready_path(d, 1, "9") == "/d/rank1.job9.ready"
    assert rht.lender_lock_path(d, 1, "9") == "/d/rank1.job9.lender"
    assert rht.claim_path(d, 1, "9", 0xAB) == "/d/rank1.job9.00000000000000ab.claim"
    assert rht.handle_path(d, 1, "9", 2) == "/d/rank1.job9.l2.handles.json"
    assert rht.ready_path(d, 1, "9", 0) == "/d/rank1.job9.l0.ready"
    assert rht.lender_lock_path(d, 1, "9", 2) == "/d/rank1.job9.l2.lender"
    assert rht.claim_path(d, 1, "9", 0xAB, 2) == "/d/rank1.job9.l2.00000000000000ab.claim"


def test_unset_index_keeps_the_single_lender_files(fake_lender):
    rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)
    assert sorted(os.listdir(fake_lender)) == [
        "rank0.job4242.handles.json",
        "rank0.job4242.lender",
        "rank0.job4242.ready",
    ]
    with open(rht.handle_path(fake_lender, 0, JOB)) as f:
        assert "lender_index" not in json.load(f)
    with open(rht.ready_path(fake_lender, 0, JOB)) as f:
        assert sorted(json.load(f)) == ["format", "job", "lender", "sha256"]


def test_indexed_lender_publishes_indexed_files(monkeypatch, fake_lender):
    monkeypatch.setenv(rht.LENDER_INDEX_ENV, "1")
    rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False, global_rank=4)
    assert sorted(os.listdir(fake_lender)) == [
        "rank0.job4242.l1.handles.json",
        "rank0.job4242.l1.lender",
        "rank0.job4242.l1.ready",
    ]
    payload = rht.wait_for_payload(fake_lender, 0, 1.0, JOB, lender_index=1)
    assert payload["lender_index"] == 1
    with open(rht.ready_path(fake_lender, 0, JOB, 1)) as f:
        assert json.load(f)["lender_index"] == 1
    # A single-lender importer does not see it.
    clock = FakeClock()
    with pytest.raises(TimeoutError):
        rht.wait_for_payload(fake_lender, 0, 2.0, JOB, clock=clock, sleep=clock.sleep)
    rht._remove_published(fake_lender, 0, JOB, 1)
    assert os.listdir(fake_lender) == []


def test_indexed_files_must_name_their_index(tmp_path):
    directory = str(tmp_path)
    # A handle file under index 1 whose payload says index 0 is malformed.
    rht.publish(directory, 0, dict(_payload(job="9"), lender_index=0), job="9", lender_index=1)
    with pytest.raises(RuntimeError, match="lender index 1: malformed"):
        rht.wait_for_payload(directory, 0, 1.0, "9", lender_index=1)


def test_lender_refuses_the_importer_count(monkeypatch, fake_lender):
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "2")
    with pytest.raises(ValueError, match="importer knob"):
        rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)
    assert os.listdir(fake_lender) == []


def test_lender_locks_are_per_index(tmp_path):
    directory = str(tmp_path)
    rht.acquire_lender_lock(directory, 0, "9", _me("lender", host="gen-a"), 0)
    # Another index of the same rank and job coexists; the same index is refused.
    rht.acquire_lender_lock(directory, 0, "9", _me("lender", host="gen-b"), 1)
    with pytest.raises(RuntimeError, match="held by"):
        rht.acquire_lender_lock(directory, 0, "9", _me("lender", host="gen-c"), 1)
    # The single-lender lock is a different file.
    rht.acquire_lender_lock(directory, 0, "9", _me("lender", host="gen-c"))


def test_interleave_order():
    assert rht.interleave_order([2, 2]) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert rht.interleave_order([3, 2]) == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    assert rht.interleave_order([4]) == [(0, i) for i in range(4)]
    order = rht.interleave_order([180, 120])
    assert len(order) == 300
    # Every run of 5 chunks holds 3 of the 180 GiB lender and 2 of the 120 GiB lender.
    for start in range(0, 300, 5):
        assert sum(k == 0 for k, _ in order[start : start + 5]) == 3
    # Each lender's chunks keep their order.
    assert [i for k, i in order if k == 1] == list(range(120))
    assert [i for k, i in order if k == 0] == list(range(180))


# ---- several lenders: the importer -----------------------------------------------------------------


def _lender_payload(index: int, num_chunks: int, nonce: int, chunk_bytes: int = 2 * MIB) -> dict:
    payload = _payload(num_chunks=num_chunks, chunk_bytes=chunk_bytes)
    payload["nonce"] = nonce
    payload["lender"] = _lender_identity(pid=1000 + index, host=f"gen-{'ab'[index]}")
    payload["lender_index"] = index
    payload["handles"] = [
        (bytes([index + 1, i]) * (rht.FABRIC_HANDLE_BYTES // 2)).hex() for i in range(num_chunks)
    ]
    return payload


class FakeChunkMemory:
    """The chunk heads of lent memory by fabric handle, standing in for the importer's CUDA pass."""

    def __init__(self) -> None:
        self.heads = {}
        self.calls: List[tuple] = []

    def lend(self, payload: dict) -> None:
        for i, handle in enumerate(payload["handles"]):
            self.heads[bytes.fromhex(handle)] = payload["nonce"] ^ i

    def __call__(self, fabric_handle, chunk_bytes, cu_device, expected, new):
        self.calls.append((fabric_handle, expected, new))
        head = self.heads[fabric_handle]
        if head != expected:
            raise RuntimeError(
                f"reads 0x{head:016x}, expected 0x{expected:016x} (stale or foreign handles)"
            )
        if new is not None:
            self.heads[fabric_handle] = new


@pytest.fixture
def merged_env(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "2")
    monkeypatch.setattr(rht, "_POLL_S", 0.01)
    memory = FakeChunkMemory()
    monkeypatch.setattr(rht, "_verify_and_stamp_chunk", memory)
    monkeypatch.setattr(rht, "_current_device", lambda: 0)
    return SimpleNamespace(directory=str(tmp_path), memory=memory)


def _lend(env, *payloads):
    for payload in payloads:
        rht.publish(env.directory, 0, payload, job=JOB, lender_index=payload["lender_index"])
        env.memory.lend(payload)


def _claims(directory):
    return sorted(name for name in os.listdir(directory) if name.endswith(".claim"))


def _import():
    return rht.import_lenders(rht.parse_tier_config(), local_rank=0, global_rank=7)


def test_wait_for_lenders_takes_a_late_lender_and_names_a_missing_one(tmp_path):
    directory = str(tmp_path)
    rht.publish(directory, 0, _lender_payload(0, 2, 0x11), job=JOB, lender_index=0)
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 3.0 and not os.path.exists(rht.ready_path(directory, 0, JOB, 1)):
            rht.publish(directory, 0, _lender_payload(1, 2, 0x22), job=JOB, lender_index=1)

    got = rht.wait_for_lenders(directory, 0, 2, 10.0, JOB, clock=clock, sleep=sleep)
    assert [p["lender_index"] for p in got] == [0, 1]

    # One deadline for all; the timeout names the missing index.
    other = str(tmp_path / "other")
    rht.publish(other, 0, _lender_payload(0, 2, 0x11), job=JOB, lender_index=0)
    clock = FakeClock()
    with pytest.raises(TimeoutError, match=r"index\(es\) \[1\] of 2.*index 1: no ready marker"):
        rht.wait_for_lenders(other, 0, 2, 5.0, JOB, clock=clock, sleep=clock.sleep)
    assert 5.0 <= clock.now < 7.0
    # A single-lender file does not stand in for index 0.
    legacy = str(tmp_path / "legacy")
    rht.publish(legacy, 0, _payload(job=JOB), job=JOB)
    clock = FakeClock()
    with pytest.raises(TimeoutError, match=r"\[0, 1\] of 2"):
        rht.wait_for_lenders(legacy, 0, 2, 2.0, JOB, clock=clock, sleep=clock.sleep)


def test_wait_for_lenders_names_a_missing_handle_file(tmp_path):
    directory = str(tmp_path)
    rht.publish(directory, 0, _lender_payload(0, 2, 0x11), job=JOB, lender_index=0)
    rht.publish(directory, 0, _lender_payload(1, 2, 0x22), job=JOB, lender_index=1)
    # Lender 1's ready marker stays, its handle file is gone.
    os.unlink(rht.handle_path(directory, 0, JOB, 1))
    clock = FakeClock()
    with pytest.raises(
        TimeoutError, match=r"index\(es\) \[1\] of 2.*index 1: the handle file is missing"
    ):
        rht.wait_for_lenders(directory, 0, 2, 2.0, JOB, clock=clock, sleep=clock.sleep)


def test_wait_for_lenders_takes_a_lender_that_republished_meanwhile(tmp_path):
    directory = str(tmp_path)
    rht.publish(directory, 0, _lender_payload(0, 2, 0x11), job=JOB, lender_index=0)
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        # While the importer waits for lender 1, lender 0 restarts and publishes new handles.
        clock.sleep(seconds)
        if not os.path.exists(rht.ready_path(directory, 0, JOB, 1)):
            rht.publish(directory, 0, _lender_payload(0, 2, 0x33), job=JOB, lender_index=0)
            rht.publish(directory, 0, _lender_payload(1, 2, 0x22), job=JOB, lender_index=1)

    got = rht.wait_for_lenders(directory, 0, 2, 10.0, JOB, clock=clock, sleep=sleep)
    assert [p["nonce"] for p in got] == [0x33, 0x22]

    # A lender that keeps re-publishing until the deadline is named.
    churn = str(tmp_path / "churn")
    rht.publish(churn, 0, _lender_payload(0, 2, 0x11), job=JOB, lender_index=0)
    rht.publish(churn, 0, _lender_payload(1, 2, 0x22), job=JOB, lender_index=1)
    nonce = [0x100]
    real_read = rht._try_read_payload

    def churning_read(directory, rank, job, lender_index=None):
        payload, reason = real_read(directory, rank, job, lender_index)
        if payload is not None and lender_index == 1:
            nonce[0] += 1
            payload = dict(payload, nonce=nonce[0])
        return payload, reason

    clock = FakeClock()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rht, "_try_read_payload", churning_read)
        with pytest.raises(
            TimeoutError, match=r"index\(es\) \[1\] of 2.*index 1: re-published during the wait"
        ):
            rht.wait_for_lenders(churn, 0, 2, 1.0, JOB, clock=clock, sleep=clock.sleep)


def test_merged_import_interleaves_and_restamps(monkeypatch, merged_env):
    monkeypatch.setenv(rht.EXPECT_GIB_ENV, f"{6 * MIB / GIB},{6 * MIB / GIB}")
    p0 = _lender_payload(0, 3, 0x1111)
    p1 = _lender_payload(1, 3, 0x2222)
    _lend(merged_env, p0, p1)
    logged: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    merged = _import()
    assert merged.owners == (0, 1, 0, 1, 0, 1)
    expected_order = [(p0, 0), (p1, 0), (p0, 1), (p1, 1), (p0, 2), (p1, 2)]
    assert merged.handles == tuple(bytes.fromhex(p["handles"][i]) for p, i in expected_order)
    assert merged.lent_bytes == 12 * MIB and merged.chunk_bytes == 2 * MIB
    # Each chunk was checked against its lender's word, then re-stamped to V ^ position.
    assert [expected for _, expected, _ in merged_env.memory.calls] == [
        p["nonce"] ^ i for p, i in expected_order
    ]
    for position, handle in enumerate(merged.handles):
        assert merged_env.memory.heads[handle] == merged.verify_nonce ^ position
    assert _claims(merged_env.directory) == sorted(
        os.path.basename(rht.claim_path(merged_env.directory, 0, JOB, p["nonce"], k))
        for k, p in enumerate((p0, p1))
    )
    assert rht.merged_import() is merged
    (line,) = [m for m in logged if "lenders" in m]
    for field in (
        "rank=0 global_rank=7",
        "importing 6 chunks",
        "(0.01 GiB) from 2 lenders",
        "k=0 host=gen-a pid=1000",
        "k=1 host=gen-b pid=1001",
        "chunks=3",
        "order=interleaved restamp_s=",
    ):
        assert field in line, field


def test_merged_import_of_one_lender_keeps_its_words(monkeypatch, merged_env):
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "1")
    p0 = _lender_payload(0, 3, 0x1111)
    _lend(merged_env, p0)
    merged = _import()
    assert merged.verify_nonce == 0x1111
    assert merged.owners == (0, 0, 0)
    # Checked, not re-stamped: a restarted importer could import the chunks again.
    assert [new for _, _, new in merged_env.memory.calls] == [None, None, None]
    for i, handle in enumerate(merged.handles):
        assert merged_env.memory.heads[handle] == 0x1111 ^ i


@pytest.mark.parametrize(
    "case, match",
    [
        ("chunk_size", "chunks of different sizes"),
        ("expect", r"lender index 1 host=gen-b pid=1001 lent 0.01 GiB, .*EXPECT_GIB expects 0.00"),
        ("same_process", "lender indices 0 and 1 name one process"),
        ("same_handle", "publish the same chunk handle"),
    ],
)
def test_merged_import_refusals_claim_nothing(monkeypatch, merged_env, case, match):
    p0 = _lender_payload(0, 3, 0x1111)
    p1 = _lender_payload(1, 3, 0x2222)
    if case == "chunk_size":
        p1 = dict(p1, chunk_bytes=4 * MIB)
    elif case == "expect":
        monkeypatch.setenv(rht.EXPECT_GIB_ENV, f"{6 * MIB / GIB},{4 * MIB / GIB}")
    elif case == "same_process":
        p1 = dict(p1, lender=p0["lender"])
    else:
        p1 = dict(p1, handles=p0["handles"])
    _lend(merged_env, p0, p1)
    with pytest.raises(ValueError, match=match):
        _import()
    assert _claims(merged_env.directory) == []
    assert merged_env.memory.calls == []
    assert rht.merged_import() is None


def test_merged_import_claims_all_or_nothing(merged_env):
    p0 = _lender_payload(0, 3, 0x1111)
    p1 = _lender_payload(1, 3, 0x2222)
    _lend(merged_env, p0, p1)
    foreign = rht.claim_lender(merged_env.directory, 0, JOB, p1, _me(host="other-ctx"), 1)
    with pytest.raises(RuntimeError, match="held by"):
        _import()
    # Lender 0's claim was taken, then removed; only the other importer's claim is left.
    assert _claims(merged_env.directory) == [os.path.basename(foreign)]
    assert merged_env.memory.calls == []


def test_exit_hook_leaves_a_claim_taken_after_the_rollback(monkeypatch, tmp_path):
    hooks = []
    monkeypatch.setattr(rht.atexit, "register", lambda fn, *args: hooks.append((fn, args)))
    directory = str(tmp_path)
    p0 = _lender_payload(0, 3, 0x1111)
    p1 = _lender_payload(1, 3, 0x2222)
    me = _me()
    # Importer A claims lender 0, is refused on lender 1 (held by B) and rolls back.
    rht.claim_lender(directory, 0, JOB, p1, _me(host="ctx-b"), 1)
    with pytest.raises(RuntimeError, match="held by"):
        rht.claim_lenders(directory, 0, JOB, [p0, p1], me)
    claim0 = rht.claim_path(directory, 0, JOB, p0["nonce"], 0)
    assert not os.path.exists(claim0)
    # Importer C takes lender 0; A's exit hooks must leave C's claim alone.
    rht.claim_lender(directory, 0, JOB, p0, _me(host="ctx-c"), 0)
    # The hooks the three importers registered all run in this process; A's are those naming A.
    hooks_of_a = [(fn, args) for fn, args in hooks if args[1:] == (me,)]
    assert [fn for fn, _ in hooks_of_a] == [rht._unlink_own]
    for fn, args in hooks_of_a:
        fn(*args)
    with open(claim0) as f:
        assert json.load(f)["owner"]["host"] == "ctx-c"

    # An importer's own claims are removed by its exit hooks.
    own = str(tmp_path / "own")
    os.makedirs(own)
    hooks.clear()
    rht.claim_lenders(own, 0, JOB, [p0, p1], me)
    assert len(_claims(own)) == 2
    for fn, args in hooks:
        fn(*args)
    assert _claims(own) == []


def test_single_lender_claim_keeps_its_exit_hook(monkeypatch, tmp_path):
    hooks = []
    monkeypatch.setattr(rht.atexit, "register", lambda fn, *args: hooks.append((fn, args)))
    path = rht.claim_lender(str(tmp_path), 0, JOB, _payload(), _me())
    # Unset knobs: the same unconditional removal at exit as before several lenders existed.
    assert hooks == [(rht._unlink, (path,))]


def test_merged_import_fails_without_every_lender(monkeypatch, merged_env):
    monkeypatch.setenv(rht.WAIT_S_ENV, "0.05")
    _lend(merged_env, _lender_payload(0, 3, 0x1111))
    with pytest.raises(TimeoutError, match=r"\[1\] of 2"):
        _build()
    assert _claims(merged_env.directory) == []


def test_merged_import_names_the_lender_of_a_stale_chunk(merged_env):
    p0 = _lender_payload(0, 3, 0x1111)
    p1 = _lender_payload(1, 3, 0x2222)
    _lend(merged_env, p0, p1)
    # Lender 1's memory no longer carries the words its handle file promises (a restarted or
    # foreign lender): the start fails and names that lender.
    merged_env.memory.heads[bytes.fromhex(p1["handles"][1])] ^= 0xFF
    with pytest.raises(
        RuntimeError,
        match=r"lender index 1 host=gen-b pid=1001 chunk 1 \(merged position 3\).*stale or foreign",
    ):
        _import()
    assert rht.merged_import() is None


class _FakeDriver:
    """The CUDA driver calls of one chunk import, on one 8-byte chunk head held in this process."""

    SUCCESS = "CUDA_SUCCESS"

    def __init__(self, head: int, drop_writes: bool = False) -> None:
        self.head = head
        self.drop_writes = drop_writes
        self.calls: List[tuple] = []
        self.CUresult = SimpleNamespace(CUDA_SUCCESS=self.SUCCESS)
        self.CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_FABRIC="FABRIC")
        self.CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE="DEVICE")
        self.CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE="RW")

    def CUmemAccessDesc(self):
        return SimpleNamespace(location=SimpleNamespace(type=None, id=None), flags=None)

    def cuMemAddressReserve(self, size, alignment, addr, flags):
        self.calls.append(("reserve", size))
        return self.SUCCESS, 0xA000

    def cuMemMap(self, va, size, offset, handle, flags):
        self.calls.append(("map", va, handle))
        return (self.SUCCESS,)

    def cuMemSetAccess(self, va, size, descs, count):
        self.calls.append(("access", va, descs[0].location.id, descs[0].flags))
        return (self.SUCCESS,)

    def cuMemcpyDtoH(self, dst, src, size):
        self.calls.append(("read", src))
        ctypes.c_uint64.from_address(dst).value = self.head
        return (self.SUCCESS,)

    def cuMemcpyHtoD(self, dst, src, size):
        self.calls.append(("write", dst))
        if not self.drop_writes:
            self.head = ctypes.c_uint64.from_address(src).value
        return (self.SUCCESS,)

    def cuMemUnmap(self, va, size):
        self.calls.append(("unmap", va))
        return (self.SUCCESS,)

    def cuMemAddressFree(self, va, size):
        self.calls.append(("free", va))
        return (self.SUCCESS,)

    def cuMemRelease(self, handle):
        self.calls.append(("release", handle))
        return (self.SUCCESS,)


@pytest.mark.parametrize(
    "case, head, new, match, writes",
    [
        ("check_only", 0x1234, None, None, 0),
        ("restamp", 0x1234, 0xBEEF, None, 1),
        ("stale", 0x9999, 0xBEEF, "stale or foreign", 0),
        ("dropped_write", 0x1234, 0xBEEF, "re-stamp reads back", 1),
    ],
)
def test_verify_and_stamp_chunk_with_a_fake_driver(monkeypatch, case, head, new, match, writes):
    driver = _FakeDriver(head, drop_writes=case == "dropped_write")
    imported = []

    def import_handle(fabric_handle, handle_type):
        imported.append((fabric_handle, handle_type))
        return 77

    transport = SimpleNamespace(_import_handle=import_handle)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.modules.dwdp.transport", transport)
    monkeypatch.setattr(rht, "_cuda_driver", lambda: driver)
    fabric_handle = b"\x05" * rht.FABRIC_HANDLE_BYTES
    if match is None:
        rht._verify_and_stamp_chunk(fabric_handle, 2 * MIB, 3, 0x1234, new)
    else:
        with pytest.raises(RuntimeError, match=match):
            rht._verify_and_stamp_chunk(fabric_handle, 2 * MIB, 3, 0x1234, new)
    assert imported == [(fabric_handle, "FABRIC")]
    names = [call[0] for call in driver.calls]
    assert names[:4] == ["reserve", "map", "access", "read"]
    assert ("map", 0xA000, 77) in driver.calls and ("access", 0xA000, 3, "RW") in driver.calls
    assert names.count("write") == writes
    if writes:
        # The write is read back before the chunk is unmapped.
        assert names[names.index("write") + 1] == "read"
    if case == "restamp":
        assert driver.head == 0xBEEF
    elif case != "dropped_write":
        assert driver.head == head
    # Every outcome unmaps, frees the address range and releases the import, once each, in order.
    assert names[-3:] == ["unmap", "free", "release"]
    assert [names.count(n) for n in ("unmap", "free", "release")] == [1, 1, 1]
    assert driver.calls[-1] == ("release", 77)


def test_importer_refuses_the_lender_index_knob(monkeypatch, merged_env):
    monkeypatch.setenv(rht.LENDER_INDEX_ENV, "0")
    with pytest.raises(ValueError, match="lender knob"):
        _build()
    monkeypatch.delenv(rht.NUM_LENDERS_ENV)
    with pytest.raises(ValueError, match="lender knob"):
        _build()


def test_merged_importer_builds_the_tier(monkeypatch, merged_env):
    pytest.importorskip("tensorrt_llm.bindings")
    p0 = _lender_payload(0, 2, 0x1111)
    p1 = _lender_payload(1, 1, 0x2222)
    _lend(merged_env, p0, p1)
    tier = _build()
    merged = rht.merged_import()
    assert tier.is_remote
    assert tier.quota == 3 * 2 * MIB
    assert tier.remote.verify_nonce == merged.verify_nonce
    assert list(tier.remote.fabric_handles) == list(merged.handles)
    tier.assert_valid()


def test_lender_spread_line(monkeypatch):
    handles = tuple(bytes([i]) * rht.FABRIC_HANDLE_BYTES for i in range(6))
    merged = rht.MergedImport(
        rank=0,
        global_rank=0,
        chunk_bytes=2 * MIB,
        verify_nonce=0xABC,
        handles=handles,
        owners=(0, 1, 0, 1, 0, 1),
        lenders=({"host": "gen-a"}, {"host": "gen-b"}),
    )
    pool_a, pool_b = 0x10_0000_0000, 0x20_0000_0000
    # Pool group 0 maps merged positions 0-2, pool group 1 positions 3-4; position 5 is free.
    memory = {pool_a + j * 2 * MIB: 0xABC ^ j for j in range(3)}
    memory.update({pool_b + j * 2 * MIB: 0xABC ^ (3 + j) for j in range(2)})
    ranges = [(pool_a, 12 * MIB), (pool_b, 12 * MIB)]
    groups = [SimpleNamespace(slot_sizes=[64]), SimpleNamespace(slot_sizes=[32])]
    monkeypatch.setattr(rht, "_mapped", lambda address: address in memory)
    monkeypatch.setattr(rht, "_read_word", lambda address: memory[address])
    logged: List[str] = []
    warned: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    monkeypatch.setattr(rht.logger, "warning", warned.append)

    # Single-lender mode: nothing is read or logged.
    assert rht.log_lender_spread(ranges, groups) is None
    assert logged == []

    monkeypatch.setattr(rht, "_MERGED", merged)
    line = rht.log_lender_spread(ranges, groups)
    assert logged == [line]
    for field in (
        "rank=0 chunks per lender per pool group:",
        "pg0=[l0:2,l1:1]",
        "pg1=[l0:1,l1:1]",
        "unmapped=[l0:0,l1:1]",
        "unknown=0",
        "lenders=[l0=gen-a,l1=gen-b]",
    ):
        assert field in line, field

    # A pool range per pool is required; otherwise a warning and no line.
    assert rht.log_lender_spread(ranges[:1], groups) is None

    # Diagnostic only: a read error is a warning, not a failure.
    def failing_read(address):
        raise RuntimeError("CUDA driver error 1")

    monkeypatch.setattr(rht, "_read_word", failing_read)
    assert rht.log_lender_spread(ranges, groups) is None
    assert len(warned) == 2 and "CUDA driver error 1" in warned[1]


def test_lender_spread_counts_unknown_heads_and_full_pools(monkeypatch):
    merged = rht.MergedImport(
        rank=0,
        global_rank=0,
        chunk_bytes=2 * MIB,
        verify_nonce=0xABC,
        handles=tuple(bytes([i]) * rht.FABRIC_HANDLE_BYTES for i in range(6)),
        owners=(0, 1, 0, 1, 0, 1),
        lenders=({"host": "gen-a"}, {"host": "gen-b"}),
    )
    pool_a, pool_b, pool_c = 0x10_0000_0000, 0x20_0000_0000, 0x30_0000_0000
    # Pool group 0: pool a is fully mapped (positions 0-2, the scan ends at the range end), pool b
    # maps position 3, then position 3 again and position 9 (past the merged list): two unknowns.
    # Pool group 1: pool c maps positions 4 and 5, then stops at an unmapped slot.
    memory = {pool_a + j * 2 * MIB: 0xABC ^ j for j in range(3)}
    memory.update({pool_b + j * 2 * MIB: 0xABC ^ p for j, p in enumerate((3, 3, 9))})
    memory.update({pool_c + j * 2 * MIB: 0xABC ^ (4 + j) for j in range(2)})
    ranges = [(pool_a, 6 * MIB), (pool_b, 8 * MIB), (pool_c, 8 * MIB)]
    groups = [SimpleNamespace(slot_sizes=[64, 64]), SimpleNamespace(slot_sizes=[32])]
    reads: List[int] = []

    def read_word(address):
        reads.append(address)
        return memory[address]

    monkeypatch.setattr(rht, "_mapped", lambda address: address in memory)
    monkeypatch.setattr(rht, "_read_word", read_word)
    monkeypatch.setattr(rht, "_MERGED", merged)
    logged: List[str] = []
    warned: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    monkeypatch.setattr(rht.logger, "warning", warned.append)
    line = rht.log_lender_spread(ranges, groups)
    assert warned == [] and logged == [line]
    for field in ("pg0=[l0:2,l1:2]", "pg1=[l0:1,l1:1]", "unmapped=[l0:0,l1:0]", "unknown=2"):
        assert field in line, field
    # The fully mapped pool a was read to its end and no further.
    assert [a for a in reads if pool_a <= a < pool_b] == [pool_a + j * 2 * MIB for j in range(3)]

    # Storage statistics or ranges of an unexpected shape: a warning, not a failed start.
    assert rht.log_lender_spread(ranges, [SimpleNamespace(), SimpleNamespace()]) is None
    assert "unexpected storage statistics" in warned[-1]
    assert rht.log_lender_spread([(pool_a, None), (pool_b, 0), (pool_c, 0)], groups) is None
    assert len(warned) == 2 and len(logged) == 1


# ---- stats line ------------------------------------------------------------------------------------


def _stats() -> dict:
    return {
        "level": 2,
        "lent_bytes": 220 * GIB,
        "num_chunks": 220,
        "demoted_pages": 10,
        "demoted_bytes": 10 * MIB,
        "promoted_pages": 4,
        "promoted_bytes": 4 * MIB,
        "gpu_kernel_host_copies": 7,
        "host_function_host_copies": 3,
        "hit_tokens": 128,
    }


def test_stats_line_fields_and_interval(monkeypatch):
    pools = [SimpleNamespace(total=1000, free=250), SimpleNamespace(total=500, free=500)]
    levels: List[int] = []

    def storage_fn(level):
        levels.append(level)
        return pools

    logged: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    stats = rht.RemoteHostTierStats(rank=1, stats_fn=_stats, storage_fn=storage_fn, interval_s=60)
    line = stats.format()
    for field in (
        "rank=1",
        "level=2",
        "lent_gib=220.00",
        "chunks=220",
        "slots=1500",
        "used=750",
        "fill=0.5000",
        "hit_tokens=128",
        "promoted_pages=4",
        "demoted_pages=10",
        "gpu_kernel_h2h=7",
        "host_fn_h2h=3",
        "rss_mb=",
        "rss_gib_per_m_used=",
    ):
        assert field in line, field
    assert levels == [2]
    assert stats.maybe_log(100.0)
    assert not stats.maybe_log(130.0)
    assert stats.maybe_log(161.0)
    assert stats.maybe_log(162.0, force=True)
    assert len(logged) == 3


def test_stats_line_reports_rss_growth_per_used_slot():
    used = {"slots": 0}
    rss = {"mb": 300 * 1024}  # the pinned local host tier is already in the RSS

    def storage_fn(level):
        return [SimpleNamespace(total=4_000_000, free=4_000_000 - used["slots"])]

    stats = rht.RemoteHostTierStats(
        rank=0, stats_fn=_stats, storage_fn=storage_fn, rss_fn=lambda: rss["mb"]
    )
    first = stats.format()
    assert "rss_base_mb=307200" in first and "used_base=0" in first
    assert "rss_gib_per_m_used=0.00" in first
    # 2 M slots filled and 3 GiB more RSS: 1.5 GiB per million, whatever the base RSS.
    used["slots"] = 2_000_000
    rss["mb"] += 3 * 1024
    assert "rss_gib_per_m_used=1.50" in stats.format()


# ---- host-tier collapse ----------------------------------------------------------------------------


def test_collapse_skips_excluded_ranges(monkeypatch):
    huge = 2 * MIB
    local = (0x10_0000_0000, 0x10_0000_0000 + 64 * MIB)
    remote = (0x20_0000_0000, 0x20_0000_0000 + 64 * MIB)
    monkeypatch.setattr(host_tier_memory, "anonymous_mappings", lambda: [local, remote])
    monkeypatch.setattr(host_tier_memory, "_huge_page_bytes", lambda: huge)
    monkeypatch.setattr(host_tier_memory, "_anon_huge_bytes", lambda: 0)
    advised: List[int] = []

    class FakeLibc:
        class _Madvise:
            argtypes = None
            restype = None

            def __call__(self, addr, length, advice):
                advised.append(addr.value)
                return 0

        madvise = _Madvise()

    monkeypatch.setattr(host_tier_memory.ctypes, "CDLL", lambda *a, **k: FakeLibc())
    host_tier_memory.collapse_new_mappings([], exclude=[(remote[0], remote[1] - remote[0])])
    assert advised and all(local[0] <= a < local[1] for a in advised)
    advised.clear()
    host_tier_memory.collapse_new_mappings([])
    assert any(remote[0] <= a < remote[1] for a in advised)


# ---- cross-rank failure sync (no GPU) --------------------------------------------------------------


def _kv_module():
    pytest.importorskip("torch")
    return pytest.importorskip("tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2")


class _FakeDistributed:
    def __init__(self, other_rank_failed: bool) -> None:
        self.other_rank_failed = other_rank_failed
        self.calls: List[tuple] = []

    def allreduce(self, value, op):
        self.calls.append((value, op))
        return max(value, int(self.other_rank_failed))


def test_remote_host_step_failure_raises_on_every_rank(monkeypatch):
    kv = _kv_module()
    mapping = SimpleNamespace(world_size=4)

    def use(dist):
        monkeypatch.setattr(kv.Distributed, "get", lambda _mapping: dist)
        return dist

    # This rank succeeded and another failed: this rank raises too instead of moving on to the
    # next collective.
    dist = use(_FakeDistributed(other_rank_failed=True))
    with pytest.raises(RuntimeError, match="failed on another rank"):
        kv._sync_remote_host_tier_step(lambda: 1, requested=True, mapping=mapping, what="Lend")
    assert dist.calls == [(0, kv.ReduceOp.MAX)]

    # This rank failed: it reports the failure to the others, then raises its own error.
    dist = use(_FakeDistributed(other_rank_failed=False))

    def failing_step():
        raise TimeoutError("no handles")

    with pytest.raises(TimeoutError, match="no handles"):
        kv._sync_remote_host_tier_step(failing_step, requested=True, mapping=mapping, what="Import")
    assert dist.calls == [(1, kv.ReduceOp.MAX)]

    # Every rank succeeded.
    dist = use(_FakeDistributed(other_rank_failed=False))
    assert kv._sync_remote_host_tier_step(lambda: 7, True, mapping, "Import") == 7

    # Not requested (knobs unset): no collective at all.
    dist = use(_FakeDistributed(other_rank_failed=True))
    assert kv._sync_remote_host_tier_step(lambda: None, False, mapping, "Import") is None
    assert dist.calls == []


def test_remote_host_step_runs_only_when_requested():
    kv = _kv_module()
    calls: List[str] = []

    def step():
        calls.append("step")
        return "tier"

    # Not requested: the step is never called and the mapping is never read (it has no fields here),
    # so the knob-off path stays inert even for a manager built with a partial mapping.
    assert kv._sync_remote_host_tier_step(step, False, SimpleNamespace(), "Lend") is None
    assert calls == []

    # Requested: the step runs once and its result is returned (one rank, no collective).
    assert (
        kv._sync_remote_host_tier_step(step, True, SimpleNamespace(world_size=1), "Lend") == "tier"
    )
    assert calls == ["step"]


# ---- KV cache manager (GPU) ------------------------------------------------------------------------


def _gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return torch


def _make_manager(host_bytes: int = 64 * MIB, **extra):
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=2048, enable_block_reuse=True, host_cache_size=host_bytes
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=512,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
        **extra,
    )


def test_manager_unchanged_when_unset():
    _gpu()
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    manager = _make_manager()
    try:
        assert list(manager.impl.cache_tier_list) == [CacheTier.GPU_MEM, CacheTier.HOST_MEM]
        assert manager.impl.get_remote_host_tier_stats()["level"] == -1
        assert manager.impl.get_remote_host_address_ranges() == []
        assert manager._remote_host_level is None
        assert manager._remote_host_stats is None
        assert manager._stats_cache_level_tier_names() == ["gpu", "host"]
    finally:
        manager.shutdown()


def test_manager_refuses_codec_with_remote_tier(monkeypatch, tmp_path):
    _gpu()
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="cold-page codec"):
        _make_manager(cold_page_codec_provider=object())


def test_manager_times_out_without_lender(monkeypatch, tmp_path):
    _gpu()
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.WAIT_S_ENV, "1")
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    with pytest.raises(TimeoutError):
        _make_manager()


def test_manager_never_falls_back_without_the_remote_tier(monkeypatch, tmp_path):
    """A handle that cannot be imported raises a CUDA error, which without the remote tier would
    rebuild the manager without host tiers; with the tier requested the construction must fail."""
    _gpu()
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, "garbage")
    payload = _payload(num_chunks=2, job="garbage")
    payload["handles"] = [os.urandom(rht.FABRIC_HANDLE_BYTES).hex() for _ in range(2)]
    rht.publish(str(tmp_path), 0, payload, job="garbage")
    with pytest.raises(Exception):
        _make_manager()


LENDER_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    from tensorrt_llm._torch.pyexecutor.kv_cache import remote_host_tier as rht
    try:
        lender = rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)
    except Exception as error:  # reported to the parent, which skips
        print("FAILED", repr(error), flush=True)
        sys.exit(0)
    print("LENT", len(lender.fabric_handles), flush=True)
    sys.stdin.read()  # hold the chunks until the parent closes stdin
    """
)

# Upper bound on the lender's start-up (CUDA init, the fabric host allocation, the handle export).
LENDER_REPORT_TIMEOUT_S = 300


def _read_lender_report(lender, timeout_s):
    """Return the lender's LENT / FAILED line split into words, [] if it exits without one, or None
    if it reports nothing within timeout_s (the lender is then killed).

    Importing tensorrt_llm prints a version banner first, so earlier lines are skipped. A daemon
    thread reads the lines, so a lender that hangs before reporting cannot hang the test. The
    deadline is checked on every line, so a lender that keeps printing other lines is killed too.
    The thread keeps draining stdout afterwards, discarding the lines, so the lender never blocks
    on a full pipe.
    """
    lines = queue.Queue()
    done = threading.Event()

    def pump():
        for line in lender.stdout:
            if not done.is_set():
                lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                return []
            words = line.split()
            if words and words[0] in ("LENT", "FAILED"):
                return words
        lender.kill()
        lender.wait()
        return None
    finally:
        done.set()


def _demote_and_promote(manager, torch, chain_blocks: int = 8, tokens_per_block: int = 32):
    """Commit a chain, push it down to the remote level with newer chains, then match it again."""
    impl = manager.impl
    stream = torch.cuda.current_stream().cuda_stream

    def commit(tokens):
        cache = impl.create_kv_cache(None, tokens)
        try:
            assert cache.resume(stream)
            assert cache.resize(len(tokens))
            rest = tokens[cache.num_committed_tokens :]
            if rest:
                cache.commit(rest)
        finally:
            cache.close()
        torch.cuda.synchronize()

    chain_tokens = chain_blocks * tokens_per_block
    first = list(range(1000, 1000 + chain_tokens))
    commit(first)
    # One cold page per block (both layers share one life cycle). LRU eviction demotes the oldest
    # chain first, so once as many pages as it has blocks have been demoted it sits in the remote
    # level; stop there, before the remote level itself starts to drop pages.
    for index in range(1, 512):
        if impl.get_remote_host_tier_stats()["demoted_pages"] >= chain_blocks:
            break
        base = 100_000 + index * chain_tokens
        commit(list(range(base, base + chain_tokens)))
    else:
        pytest.fail("no page was demoted into the remote level")
    before = impl.get_remote_host_tier_stats()
    assert before["promoted_pages"] == 0

    cache = impl.create_kv_cache(None, first)
    try:
        assert list(cache.cached_tokens_by_level)[2] > 0
        assert cache.resume(stream)
        assert cache.num_committed_tokens > 0
    finally:
        cache.close()
    torch.cuda.synchronize()
    after = impl.get_remote_host_tier_stats()
    assert after["demoted_pages"] >= chain_blocks
    assert after["promoted_pages"] > 0
    assert after["hit_tokens"] > 0
    assert after["gpu_kernel_host_copies"] > 0
    assert after["host_function_host_copies"] == 0
    assert f"hit_tokens={after['hit_tokens']}" in manager._remote_host_stats.format()


def test_two_process_lender_and_importer(monkeypatch, tmp_path):
    torch = _gpu()
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    job = f"rh-test-{os.getpid()}"
    env = dict(os.environ)
    for name in ALL_ENVS:
        env.pop(name, None)
    env.update(
        {
            rht.LEND_GIB_ENV: str(16 * MIB / GIB),
            rht.CHUNK_GIB_ENV: str(2 * MIB / GIB),
            rht.DIR_ENV: str(tmp_path),
            rht.JOB_ENV: job,
        }
    )
    lender = subprocess.Popen(
        [sys.executable, "-c", LENDER_SCRIPT],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        first = _read_lender_report(lender, LENDER_REPORT_TIMEOUT_S)
        if first is None:
            # Like a FAILED report, a lender stuck in its start-up is the environment, not the
            # importer under test.
            pytest.skip(
                f"the lender reported neither LENT nor FAILED in {LENDER_REPORT_TIMEOUT_S} s"
            )
        if not first or first[0] != "LENT":
            pytest.skip(f"fabric host memory unavailable to the lender: {first}")
        assert int(first[1]) == 8
        with open(rht.handle_path(str(tmp_path), 0, job)) as f:
            handles = json.load(f)
        assert handles["chunk_bytes"] == 2 * MIB
        assert handles["lender"]["pid"] == lender.pid

        monkeypatch.setenv(rht.TIER_ENV, "1")
        monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
        monkeypatch.setenv(rht.WAIT_S_ENV, "30")
        monkeypatch.setenv(rht.EXPECT_GIB_ENV, str(16 * MIB / GIB))
        monkeypatch.setenv(rht.JOB_ENV, job)
        try:
            # A small local host tier, so committed chains cascade into the remote level quickly.
            manager = _make_manager(host_bytes=2 * MIB)
        except Exception as error:  # same-node import is a driver feature, not this code's
            if "CUDA driver error" in str(error) and ("801" in str(error) or "800" in str(error)):
                pytest.skip(f"fabric import unavailable: {error}")
            raise
        try:
            tiers = list(manager.impl.cache_tier_list)
            # The long-chain protector's guard (last tier is host memory) still holds.
            assert tiers == [CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.HOST_MEM]
            stats = manager.impl.get_remote_host_tier_stats()
            assert stats["level"] == 2
            assert stats["num_chunks"] == 8
            assert stats["lent_bytes"] == 16 * MIB
            assert stats["host_function_host_copies"] == 0
            assert manager._remote_host_level == 2
            assert manager._stats_cache_level_tier_names() == ["gpu", "host", "host_remote"]
            assert "level=2" in manager._remote_host_stats.format()
            assert len(manager.impl.get_remote_host_address_ranges()) >= 1
            # The claim on this lender is held.
            assert os.path.exists(rht.claim_path(str(tmp_path), 0, job, handles["nonce"]))

            _demote_and_promote(manager, torch)
        finally:
            manager.shutdown()
    finally:
        lender.stdin.close()
        try:
            lender.wait(timeout=60)
        except subprocess.TimeoutExpired:
            lender.kill()
            lender.wait()


# ---- several lenders (GPU, fabric host memory) -----------------------------------------------------

TWO_LENDERS = ((0, 16 * MIB), (1, 8 * MIB))


def _skip_without_fabric_import(error: Exception) -> None:
    """Same-node fabric import is a driver feature, not this code's: skip on its error codes.

    With several lenders the first import is the importer's own (dwdp ``check_cu_result``: "CUDA
    error: <CUresult>"); with one lender it is KVCMv2's ("CUDA driver error").
    """
    text = str(error)
    if ("CUDA driver error" in text or "CUDA error" in text) and any(
        code in text for code in ("800", "801", "NOT_PERMITTED", "NOT_SUPPORTED")
    ):
        pytest.skip(f"fabric import unavailable: {error}")


def _start_lenders(tmp_path, job: str, started: List[subprocess.Popen]) -> List[dict]:
    """Start one lender subprocess per (index, size) of TWO_LENDERS on GPU 0 (2 MiB chunks), wait for
    each to lend, and return their handle files. Appends each process to ``started`` at once, so
    the caller can stop them whatever happens."""
    payloads = []
    for index, size in TWO_LENDERS:
        env = dict(os.environ)
        for name in ALL_ENVS:
            env.pop(name, None)
        env.update(
            {
                rht.LEND_GIB_ENV: str(size / GIB),
                rht.CHUNK_GIB_ENV: str(2 * MIB / GIB),
                rht.DIR_ENV: str(tmp_path),
                rht.JOB_ENV: job,
                rht.LENDER_INDEX_ENV: str(index),
            }
        )
        lender = subprocess.Popen(
            [sys.executable, "-c", LENDER_SCRIPT],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(lender)
        report = _read_lender_report(lender, LENDER_REPORT_TIMEOUT_S)
        if report is None:
            pytest.skip(
                f"lender {index} reported neither LENT nor FAILED in {LENDER_REPORT_TIMEOUT_S} s"
            )
        if not report or report[0] != "LENT":
            pytest.skip(f"fabric host memory unavailable to lender {index}: {report}")
        assert int(report[1]) == size // (2 * MIB)
        with open(rht.handle_path(str(tmp_path), 0, job, index)) as f:
            payload = json.load(f)
        assert payload["lender_index"] == index and payload["lender"]["pid"] == lender.pid
        payloads.append(payload)
    return payloads


def _stop_lenders(started: List[subprocess.Popen]) -> None:
    for lender in started:
        lender.stdin.close()
        try:
            lender.wait(timeout=60)
        except subprocess.TimeoutExpired:
            lender.kill()
            lender.wait()


def _set_two_lender_importer(monkeypatch, tmp_path, job: str) -> None:
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.WAIT_S_ENV, "30")
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "2")
    monkeypatch.setenv(rht.EXPECT_GIB_ENV, ",".join(str(size / GIB) for _, size in TWO_LENDERS))
    monkeypatch.setenv(rht.JOB_ENV, job)


def test_two_lenders_one_merged_level(monkeypatch, tmp_path):
    """Two lender processes (16 MiB and 8 MiB) and one importer on one GPU: one remote level of 12
    chunks, interleaved 2:1, re-stamped by the importer (fabric import, map, write, release, then
    KVCMv2 imports the same handles again), both claims held, and a demoted chain promotes back."""
    torch = _gpu()
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    job = f"rh-test2-{os.getpid()}"
    started: List[subprocess.Popen] = []
    try:
        payloads = _start_lenders(tmp_path, job, started)
        _set_two_lender_importer(monkeypatch, tmp_path, job)
        spread: List[str] = []
        real_spread = rht.log_lender_spread

        def recording_spread(*args, **kwargs):
            line = real_spread(*args, **kwargs)
            spread.append(line)
            return line

        monkeypatch.setattr(rht, "log_lender_spread", recording_spread)
        try:
            manager = _make_manager(host_bytes=2 * MIB)
        except Exception as error:
            _skip_without_fabric_import(error)
            raise
        try:
            assert list(manager.impl.cache_tier_list) == [
                CacheTier.GPU_MEM,
                CacheTier.HOST_MEM,
                CacheTier.HOST_MEM,
            ]
            stats = manager.impl.get_remote_host_tier_stats()
            assert stats["level"] == 2
            assert stats["num_chunks"] == 12
            assert stats["lent_bytes"] == 24 * MIB
            merged = rht.merged_import()
            assert merged.owners == tuple(k for k, _ in rht.interleave_order([8, 4]))
            assert [lender["pid"] for lender in merged.lenders] == [
                p["lender"]["pid"] for p in payloads
            ]
            for index, payload in enumerate(payloads):
                assert os.path.exists(
                    rht.claim_path(str(tmp_path), 0, job, payload["nonce"], index)
                )
            # The per-pool-group spread line was written, and every mapped chunk named its lender.
            assert len(spread) == 1 and spread[0] is not None
            assert "unknown=0" in spread[0] and "chunks per lender per pool group" in spread[0]
            _demote_and_promote(manager, torch)
        finally:
            manager.shutdown()
    finally:
        _stop_lenders(started)


def test_two_lenders_stale_handles_fail_the_start(monkeypatch, tmp_path):
    """A handle file whose nonce does not match the chunk words (a restarted or foreign lender) fails
    the importer's start, and the error names that lender."""
    _gpu()
    job = f"rh-test3-{os.getpid()}"
    started: List[subprocess.Popen] = []
    try:
        payloads = _start_lenders(tmp_path, job, started)
        stale = dict(payloads[1], nonce=payloads[1]["nonce"] ^ 0x5A)
        rht.publish(str(tmp_path), 0, stale, job=job, lender_index=1)
        _set_two_lender_importer(monkeypatch, tmp_path, job)
        with pytest.raises(RuntimeError) as raised:
            _make_manager(host_bytes=2 * MIB)
        _skip_without_fabric_import(raised.value)
        message = str(raised.value)
        assert "lender index 1 host=" in message and "stale or foreign" in message
        assert rht.merged_import() is None
    finally:
        _stop_lenders(started)
