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
"""Tests of split lending in the remote host KV tier (TLLM_KV_REMOTE_HOST_LEND_PARTS / _PART).

The knobs, the per-part files, nonces and chunk words of the lender, and the importer's part
choice (wait, checks, ``auto`` and fixed part claims, the timeout hints) run on the CPU with the
fakes of ``test_kv_remote_host_tier``. The GPU tests start real lender processes and two importers
(one in a subprocess, one in the test process) that each take one part of the same lenders; they
also need fabric host memory and skip without it.
"""

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from typing import List, Optional

import pytest
import test_kv_remote_host_tier as base
from test_kv_remote_host_tier import _clean_env  # noqa: F401  (autouse fixture)

from tensorrt_llm._torch.pyexecutor.kv_cache import remote_host_tier as rht

MIB = base.MIB
GIB = base.GIB
JOB = base.JOB
SPLIT_ENVS = (rht.LEND_PARTS_ENV, rht.PART_ENV)


@pytest.fixture(autouse=True)
def _clean_split_env(monkeypatch):
    for name in SPLIT_ENVS:
        monkeypatch.delenv(name, raising=False)


# ---- knobs and file names --------------------------------------------------------------------------


def test_lend_parts_knob():
    base_env = {rht.LEND_GIB_ENV: "180", rht.DIR_ENV: "/d", rht.LENDER_INDEX_ENV: "0"}
    for unsplit in ("", " ", "1"):
        config = rht.parse_lend_config(dict(base_env, **{rht.LEND_PARTS_ENV: unsplit}))
        assert config.num_parts is None and config.part_chunks == 180
    config = rht.parse_lend_config(dict(base_env, **{rht.LEND_PARTS_ENV: "2"}))
    assert config.num_parts == 2 and config.num_chunks == 180 and config.part_chunks == 90
    sixteen = dict(base_env, **{rht.LEND_GIB_ENV: "160", rht.LEND_PARTS_ENV: "16"})
    assert rht.parse_lend_config(sixteen).part_chunks == 10
    for bad in ("0", "17", "x", "1.5", "-2"):
        with pytest.raises(ValueError, match=rht.LEND_PARTS_ENV):
            rht.parse_lend_config(dict(base_env, **{rht.LEND_PARTS_ENV: bad}))
    # Parts are indexed files: no split without a lender index.
    unindexed = {rht.LEND_GIB_ENV: "180", rht.DIR_ENV: "/d", rht.LEND_PARTS_ENV: "2"}
    with pytest.raises(ValueError, match=f"needs {rht.LENDER_INDEX_ENV}"):
        rht.parse_lend_config(unindexed)
    # Whole chunks per part, never a silent truncation.
    with pytest.raises(ValueError, match="do not split into 2 equal parts"):
        rht.parse_lend_config(dict(base_env, **{rht.LEND_GIB_ENV: "181", rht.LEND_PARTS_ENV: "2"}))
    # The knob alone lends nothing.
    assert rht.parse_lend_config({rht.LEND_PARTS_ENV: "2", rht.DIR_ENV: "/d"}) is None


def test_part_knob():
    base_env = {rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.NUM_LENDERS_ENV: "3"}
    assert rht.parse_tier_config(base_env).part is None
    assert rht.parse_tier_config(dict(base_env, **{rht.PART_ENV: " "})).part is None
    assert rht.parse_tier_config(dict(base_env, **{rht.PART_ENV: "auto"})).part == rht.PART_AUTO
    assert rht.parse_tier_config(dict(base_env, **{rht.PART_ENV: "0"})).part == 0
    assert rht.parse_tier_config(dict(base_env, **{rht.PART_ENV: "15"})).part == 15
    for bad in ("16", "-1", "x", "AUTO"):
        with pytest.raises(ValueError, match=f"{rht.PART_ENV}.*use auto or a part index"):
            rht.parse_tier_config(dict(base_env, **{rht.PART_ENV: bad}))
    config = rht.parse_tier_config(
        dict(base_env, **{rht.PART_ENV: "auto", rht.EXPECT_GIB_ENV: "90,90,90"})
    )
    assert config.expect_bytes_per_lender == (90 * GIB,) * 3
    with pytest.raises(ValueError, match=f"needs {rht.NUM_LENDERS_ENV}"):
        rht.parse_tier_config({rht.TIER_ENV: "1", rht.DIR_ENV: "/d", rht.PART_ENV: "auto"})


def test_part_file_names():
    d = "/d"
    assert rht.handle_path(d, 1, "9", 2, 0) == "/d/rank1.job9.l2.p0.handles.json"
    assert rht.ready_path(d, 1, "9", 2, 1) == "/d/rank1.job9.l2.p1.ready"
    assert rht.claim_path(d, 1, "9", 0xAB, 2, 1) == "/d/rank1.job9.l2.p1.00000000000000ab.claim"
    # The lock stays one per lender process.
    assert rht.lender_lock_path(d, 1, "9", 2) == "/d/rank1.job9.l2.lender"
    # No part: the names of the several-lender mode, unchanged.
    assert rht.ready_path(d, 1, "9", 2) == "/d/rank1.job9.l2.ready"


# ---- lender ----------------------------------------------------------------------------------------


class _FakeSplitLender(base._FakeLender):
    """A lender of 4 chunks with one nonce per part, as ``RemoteHostLender`` gives them."""

    def __init__(self, config, rank, device):
        super().__init__(config, rank, device)
        self.nonces = [0x1100 + part for part in range(config.num_parts or 1)]
        self.nonce = self.nonces[0]

    def lend(self):
        return dict(base._payload(rank=self.rank, num_chunks=4), nonce=self.nonce)


@pytest.fixture
def split_lender(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.LEND_GIB_ENV, "2")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setenv(rht.LENDER_INDEX_ENV, "1")
    monkeypatch.setenv(rht.LEND_PARTS_ENV, "2")
    monkeypatch.setattr(rht, "RemoteHostLender", _FakeSplitLender)
    base._FakeLender.instances.clear()
    return str(tmp_path)


def test_split_lender_publishes_every_part(monkeypatch, split_lender):
    hooks = []
    monkeypatch.setattr(rht.atexit, "register", lambda fn, *args: hooks.append((fn, args)))
    logged: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    lender = rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False, global_rank=4)
    # No unsuffixed handle file or marker: an importer without the part knob finds nothing.
    assert sorted(os.listdir(split_lender)) == [
        "rank0.job4242.l1.lender",
        "rank0.job4242.l1.p0.handles.json",
        "rank0.job4242.l1.p0.ready",
        "rank0.job4242.l1.p1.handles.json",
        "rank0.job4242.l1.p1.ready",
    ]
    whole = base._payload(rank=0, num_chunks=4)["handles"]
    for part in range(2):
        with open(rht.handle_path(split_lender, 0, JOB, 1, part)) as f:
            payload = json.load(f)
        assert payload["nonce"] == lender.nonces[part]
        assert payload["handles"] == whole[2 * part : 2 * part + 2]
        assert (payload["part"], payload["num_parts"], payload["lender_index"]) == (part, 2, 1)
        assert payload["lender"]["rank"] == 4 and payload["job"] == JOB
        with open(rht.ready_path(split_lender, 0, JOB, 1, part)) as f:
            marker = json.load(f)
        assert (marker["part"], marker["num_parts"], marker["lender_index"]) == (part, 2, 1)
    (line,) = [m for m in logged if " lent " in m]
    assert "lender_index=1 parts=2 part_gib=1.00 " in line
    # The exit hook removes every part file and the lock.
    assert hooks == [(rht._remove_published, (split_lender, 0, JOB, 1, 2))]
    rht._remove_published(*hooks[0][1])
    assert os.listdir(split_lender) == []


def test_split_lender_failure_removes_every_part(monkeypatch, split_lender):
    real_publish = rht.publish

    def publish(directory, rank, payload, job, lender_index=None, part=None):
        if part == 1:
            raise OSError("disk full")
        real_publish(directory, rank, payload, job, lender_index, part)

    monkeypatch.setattr(rht, "publish", publish)
    with pytest.raises(OSError, match="disk full"):
        rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)
    assert base._FakeLender.instances[-1].released
    assert os.listdir(split_lender) == []
    assert rht._LENDER is None


def test_part_payloads_refuses_an_uneven_split():
    payload = base._payload(num_chunks=3)
    with pytest.raises(ValueError, match="do not form 2 parts"):
        rht.part_payloads(payload, [1, 2], 2)
    with pytest.raises(ValueError, match="do not form 2 parts"):
        rht.part_payloads(base._payload(num_chunks=4), [1], 2)


def test_cross_role_refusals(monkeypatch, split_lender):
    monkeypatch.setenv(rht.PART_ENV, "auto")
    with pytest.raises(ValueError, match=f"{rht.PART_ENV} is an importer knob"):
        rht.maybe_lend(local_rank=0, device_fn=lambda: 0, is_draft=False)
    assert os.listdir(split_lender) == []
    for name in (rht.LEND_GIB_ENV, rht.LENDER_INDEX_ENV, rht.PART_ENV):
        monkeypatch.delenv(name)
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "3")
    with pytest.raises(ValueError, match=f"{rht.LEND_PARTS_ENV} is a lender knob"):
        base._build()


class _AllocDriver:
    """Just enough of the CUDA driver for ``RemoteHostLender.lend`` without a GPU."""

    class CUresult:
        CUDA_SUCCESS = 0

    class CUmemAllocationType:
        CU_MEM_ALLOCATION_TYPE_PINNED = 1

    class CUmemLocationType:
        CU_MEM_LOCATION_TYPE_HOST_NUMA = 2

    class CUmemAllocationHandleType:
        CU_MEM_HANDLE_TYPE_FABRIC = 3

    def __init__(self) -> None:
        self.created = 0
        self.released: List[int] = []

    def cuDeviceGet(self, ordinal):
        return (0, ordinal)

    def CUmemAllocationProp(self):
        return SimpleNamespace(location=SimpleNamespace(), allocFlags=SimpleNamespace())

    def cuMemCreate(self, size, prop, flags):
        self.created += 1
        return (0, self.created)

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return (0,)


@pytest.mark.parametrize("num_parts", [None, 2, 3])
def test_lender_words_and_nonce_draws(monkeypatch, num_parts):
    """Chunk i of part j reads nonce_j ^ i; unsplit draws one nonce and writes nonce ^ i, as before."""
    driver = _AllocDriver()
    monkeypatch.setattr(rht, "_cuda_driver", lambda: driver)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.modules.dwdp.transport",
        SimpleNamespace(_export_handle=lambda handle, kind: bytes([handle]) * 64),
    )
    draws = []

    def urandom(n):
        draws.append(n)
        return (0x1000 * len(draws)).to_bytes(n, "little")

    monkeypatch.setattr(rht.os, "urandom", urandom)
    words = []
    monkeypatch.setattr(rht.RemoteHostLender, "_host_numa", lambda self, cu_device: 0, raising=True)
    monkeypatch.setattr(
        rht.RemoteHostLender,
        "_write_head",
        lambda self, handle, value, cu_device: words.append(value),
    )
    config = rht.LendConfig(
        lend_bytes=6 * 2 * MIB,
        chunk_bytes=2 * MIB,
        directory="/d",
        lender_index=0 if num_parts else None,
        num_parts=num_parts,
    )
    lender = rht.RemoteHostLender(config, rank=0, device=0)
    payload = lender.lend()
    assert draws == [8] * (num_parts or 1)
    assert lender.nonce == lender.nonces[0] == 0x1000
    size = 6 // (num_parts or 1)
    assert words == [lender.nonces[g // size] ^ (g % size) for g in range(6)]
    assert payload["nonce"] == lender.nonces[0] and len(payload["handles"]) == 6
    if num_parts is None:
        assert words == [0x1000 ^ g for g in range(6)]
    else:
        parts = rht.part_payloads(payload, lender.nonces, num_parts)
        assert [p["nonce"] for p in parts] == lender.nonces
        assert sum((p["handles"] for p in parts), []) == payload["handles"]


# ---- importer --------------------------------------------------------------------------------------


def _part_payload(index: int, part: int, num_parts: int, num_chunks: int, nonce: int) -> dict:
    payload = base._payload(num_chunks=num_chunks)
    payload.update(
        nonce=nonce,
        lender=base._lender_identity(pid=1000 + index, host=f"gen-{index}"),
        lender_index=index,
        part=part,
        num_parts=num_parts,
        handles=[
            (bytes([index + 1, part + 1, i, 0]) * (rht.FABRIC_HANDLE_BYTES // 4)).hex()
            for i in range(num_chunks)
        ],
    )
    return payload


def _publish(env, payload) -> None:
    rht.publish(
        env.directory, 0, payload, JOB, lender_index=payload["lender_index"], part=payload["part"]
    )
    env.memory.lend(payload)


def _lend_split(env, num_lenders=3, num_parts=2, num_chunks=2) -> List[List[dict]]:
    """Publish ``num_parts`` parts of ``num_lenders`` lenders; return ``parts[j][k]``."""
    parts = [
        [
            _part_payload(k, j, num_parts, num_chunks, 0x1000 * (k + 1) + j)
            for k in range(num_lenders)
        ]
        for j in range(num_parts)
    ]
    for part_set in parts:
        for payload in part_set:
            _publish(env, payload)
    return parts


@pytest.fixture
def split_env(monkeypatch, tmp_path):
    monkeypatch.setenv(rht.TIER_ENV, "1")
    monkeypatch.setenv(rht.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(rht.JOB_ENV, JOB)
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "3")
    monkeypatch.setenv(rht.PART_ENV, "auto")
    monkeypatch.setattr(rht, "_POLL_S", 0.01)
    memory = base.FakeChunkMemory()
    monkeypatch.setattr(rht, "_verify_and_stamp_chunk", memory)
    monkeypatch.setattr(rht, "_current_device", lambda: 0)
    importers: List[str] = []

    def identity(role, rank, local_rank):
        # Each import of the test is a different context worker (host ctx-<n>).
        importers.append(f"ctx-{len(importers)}")
        return rht.Identity(role=role, host=importers[-1], rank=rank, local_rank=0, pid=4321)

    monkeypatch.setattr(rht.Identity, "of", staticmethod(identity))
    return SimpleNamespace(directory=str(tmp_path), memory=memory)


def _import():
    return rht.import_lenders(rht.parse_tier_config(), local_rank=0, global_rank=7)


def _claims_of(directory) -> dict:
    """Claim file name -> owner host."""
    claims = {}
    for name in base._claims(directory):
        with open(os.path.join(directory, name)) as f:
            claims[name] = json.load(f)["owner"]["host"]
    return claims


def test_auto_gives_two_importers_disjoint_parts(monkeypatch, split_env):
    monkeypatch.setenv(rht.EXPECT_GIB_ENV, ",".join([str(4 * MIB / GIB)] * 3))
    parts = _lend_split(split_env)
    logged: List[str] = []
    monkeypatch.setattr(rht.logger, "info", logged.append)
    first = _import()
    second = _import()
    assert (first.part, second.part) == (0, 1)
    # Each took one part of every lender, interleaved and re-stamped to its own V ^ position.
    for merged, part_set in ((first, parts[0]), (second, parts[1])):
        assert merged.owners == (0, 1, 2, 0, 1, 2)
        assert set(merged.handles) == {
            bytes.fromhex(h) for payload in part_set for h in payload["handles"]
        }
        for position, handle in enumerate(merged.handles):
            assert split_env.memory.heads[handle] == merged.verify_nonce ^ position
    assert not set(first.handles) & set(second.handles)
    claims = _claims_of(split_env.directory)
    assert claims == {
        os.path.basename(
            rht.claim_path(split_env.directory, 0, JOB, payload["nonce"], k, j)
        ): f"ctx-{j}"
        for j, part_set in enumerate(parts)
        for k, payload in enumerate(part_set)
    }
    lines = [m for m in logged if "importing" in m]
    assert "from 3 lenders" in lines[0] and "part=0 of 2 order=interleaved restamp_s=" in lines[0]
    assert "part=1 of 2 " in lines[1]
    assert f"nonce=0x{parts[1][2]['nonce']:016x}" in lines[1]

    # A third importer finds every part held and fails closed, naming the holders.
    calls = len(split_env.memory.calls)
    with pytest.raises(rht.ClaimHeldError, match=r"all 2 parts of the lenders are held") as raised:
        _import()
    assert "ctx-0" in str(raised.value) and "ctx-1" in str(raised.value)
    assert _claims_of(split_env.directory) == claims
    assert len(split_env.memory.calls) == calls


def test_auto_rolls_back_a_part_it_cannot_take_whole(split_env):
    parts = _lend_split(split_env)
    # Another importer holds lender 1's part 0 only.
    foreign = rht.claim_lender(
        split_env.directory, 0, JOB, parts[0][1], base._me(host="ctx-x"), 1, 0
    )
    merged = _import()
    assert merged.part == 1
    claims = _claims_of(split_env.directory)
    # Lender 0's part 0, taken first, was rolled back; the foreign claim is untouched.
    assert claims[os.path.basename(foreign)] == "ctx-x"
    assert sorted(host for name, host in claims.items() if ".p1." in name) == ["ctx-0"] * 3
    assert not any(".p0." in name and host == "ctx-0" for name, host in claims.items())


def test_fixed_part_never_moves_on(monkeypatch, split_env):
    parts = _lend_split(split_env)
    monkeypatch.setenv(rht.PART_ENV, "1")
    assert _import().part == 1
    assert all(".p1." in name for name in _claims_of(split_env.directory))
    with pytest.raises(rht.ClaimHeldError, match=r"\.p1\..*held by"):
        _import()
    monkeypatch.setenv(rht.PART_ENV, "0")
    assert _import().part == 0
    assert sorted(_claims_of(split_env.directory)) == sorted(
        os.path.basename(rht.claim_path(split_env.directory, 0, JOB, payload["nonce"], k, j))
        for j, part_set in enumerate(parts)
        for k, payload in enumerate(part_set)
    )


@pytest.mark.parametrize(
    "case, match",
    [
        ("part_too_high", r"PART=2 but the lenders lent 2 parts"),
        ("expect", r"lender index 0 part 0 host=gen-0 pid=1000 lent 0.00 GiB.*expects 0.01"),
        ("other_process", r"lender index 2 part 1 host=gen-9 .*does not match part 0"),
        ("same_handle", r"lender index 0 part 0 and lender index 0 part 1 publish the same"),
        ("chunk_size", "chunks of different sizes"),
    ],
)
def test_part_refusals_claim_nothing(monkeypatch, split_env, case, match):
    parts = _lend_split(split_env)
    if case == "part_too_high":
        monkeypatch.setenv(rht.PART_ENV, "2")
    elif case == "expect":
        monkeypatch.setenv(rht.EXPECT_GIB_ENV, ",".join([str(8 * MIB / GIB)] * 3))
    else:
        bad = dict(parts[1][2] if case == "other_process" else parts[1][0])
        if case == "other_process":
            bad["lender"] = base._lender_identity(pid=9, host="gen-9")
        elif case == "same_handle":
            bad["handles"] = parts[0][0]["handles"]
        else:
            bad["chunk_bytes"] = 4 * MIB
        rht.publish(split_env.directory, 0, bad, JOB, bad["lender_index"], bad["part"])
    with pytest.raises(ValueError, match=match):
        _import()
    assert base._claims(split_env.directory) == []
    assert split_env.memory.calls == []


def test_lenders_must_agree_on_the_number_of_parts(split_env):
    _lend_split(split_env, num_lenders=2, num_parts=2)
    for part in range(3):
        _publish(split_env, _part_payload(2, part, 3, 2, 0x7000 + part))
    with pytest.raises(ValueError, match=r"different numbers of parts \[2, 2, 3\]"):
        _import()
    assert base._claims(split_env.directory) == []


def test_one_split_lender_keeps_its_words(monkeypatch, split_env):
    monkeypatch.setenv(rht.NUM_LENDERS_ENV, "1")
    parts = _lend_split(split_env, num_lenders=1, num_chunks=3)
    first, second = _import(), _import()
    for merged, part_set in ((first, parts[0]), (second, parts[1])):
        # V is the part's nonce and positions are the chunk indices within the part: no re-stamp.
        assert merged.verify_nonce == part_set[0]["nonce"]
        assert merged.handles == tuple(bytes.fromhex(h) for h in part_set[0]["handles"])
        for i, handle in enumerate(merged.handles):
            assert split_env.memory.heads[handle] == part_set[0]["nonce"] ^ i
    assert [new for _, _, new in split_env.memory.calls] == [None] * 6


def test_stale_part_names_lender_and_part(split_env):
    parts = _lend_split(split_env)
    split_env.memory.heads[bytes.fromhex(parts[0][2]["handles"][1])] ^= 0xFF
    with pytest.raises(RuntimeError, match=r"lender index 2 part 0 host=gen-2 .*chunk 1 .*stale"):
        _import()


def test_wait_for_parts_shares_one_deadline(tmp_path):
    env = SimpleNamespace(directory=str(tmp_path), memory=base.FakeChunkMemory())
    for k in range(2):
        _publish(env, _part_payload(k, 0, 2, 2, 0x10 + k))
    clock = base.FakeClock()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 3.0 and not os.path.exists(rht.ready_path(env.directory, 0, JOB, 1, 1)):
            for k in range(2):
                _publish(env, _part_payload(k, 1, 2, 2, 0x20 + k))

    parts = rht.wait_for_parts(env.directory, 0, 2, 10.0, JOB, clock=clock, sleep=sleep)
    assert [[p["nonce"] for p in part_set] for part_set in parts] == [[0x10, 0x11], [0x20, 0x21]]
    # Part 1 never comes: the timeout falls at the one deadline and names the part.
    other = SimpleNamespace(directory=str(tmp_path / "other"), memory=base.FakeChunkMemory())
    for k in range(2):
        _publish(other, _part_payload(k, 0, 2, 2, 0x10 + k))
    clock = base.FakeClock()
    with pytest.raises(TimeoutError, match=r"\[0, 1\] of 2 \(part 1\)"):
        rht.wait_for_parts(other.directory, 0, 2, 5.0, JOB, clock=clock, sleep=clock.sleep)
    assert 5.0 <= clock.now < 7.0


def test_timeouts_name_the_file_of_the_other_mode(tmp_path):
    directory = str(tmp_path)
    clock = base.FakeClock()
    # PART set, lender not split.
    rht.publish(directory, 0, base._lender_payload(0, 2, 0x11), JOB, lender_index=0)
    with pytest.raises(
        TimeoutError,
        match=r"index 0: no ready marker; rank0\.job4242\.l0\.ready exists \(the lender is not split",
    ):
        rht.wait_for_parts(directory, 0, 1, 2.0, JOB, clock=clock, sleep=clock.sleep)
    # Lender split, PART unset.
    split = str(tmp_path / "split")
    rht.publish(split, 0, _part_payload(0, 0, 2, 2, 0x11), JOB, lender_index=0, part=0)
    clock = base.FakeClock()
    with pytest.raises(
        TimeoutError,
        match=rf"index 0: no ready marker; rank0\.job4242\.l0\.p0\.ready exists \(the lender is "
        rf"split, set {rht.PART_ENV}\)",
    ):
        rht.wait_for_lenders(split, 0, 1, 2.0, JOB, clock=clock, sleep=clock.sleep)


def test_part_file_must_name_its_part(tmp_path):
    directory = str(tmp_path)
    rht.publish(directory, 0, _part_payload(0, 0, 2, 2, 0x11), JOB, lender_index=0, part=1)
    with pytest.raises(RuntimeError, match="lender index 0 part 1: malformed"):
        rht.wait_for_lenders(directory, 0, 1, 1.0, JOB, part=1)


# ---- GPU: real lenders, two importers --------------------------------------------------------------

IMPORTER_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    import torch
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    import test_kv_remote_host_tier as base
    from tensorrt_llm._torch.pyexecutor.kv_cache import remote_host_tier as rht
    try:
        manager = base._make_manager(host_bytes=2 * base.MIB)
    except Exception as error:  # reported to the parent
        print("FAILED", repr(error).replace("\\n", " "), flush=True)
        sys.exit(0)
    try:
        stats = manager.impl.get_remote_host_tier_stats()
        print("IMPORTED", rht.merged_import().part, stats["level"], stats["num_chunks"], flush=True)
        base._demote_and_promote(manager, torch)
        print("PROMOTED", manager.impl.get_remote_host_tier_stats()["hit_tokens"], flush=True)
        sys.stdin.read()  # hold the claims until the parent closes stdin
    finally:
        manager.shutdown()
    """
)


def _pump(proc) -> "queue.Queue":
    """Drain ``proc``'s stdout into a queue from a daemon thread (None at the end), so the process
    never blocks on a full pipe and a reader can wait with a timeout."""
    lines: "queue.Queue" = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    return lines


def _read_report(proc, lines, keys, timeout_s) -> Optional[List[str]]:
    """The next line of ``proc`` (from ``lines``) whose first word is in ``keys``, split; [] if it
    exits without one; None (the process killed) if none comes within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        if line is None:
            return []
        words = line.split()
        if words and words[0] in keys:
            return words
    proc.kill()
    proc.wait()
    return None


def _start_split_lenders(tmp_path, job, sizes, started) -> List[List[dict]]:
    """Start one lender per size (index = position, 2 parts of 2 MiB chunks) on GPU 0; return the
    handle files as ``parts[j][k]``. Each process is appended to ``started`` at once."""
    for index, size in enumerate(sizes):
        env = dict(os.environ)
        for name in base.ALL_ENVS + SPLIT_ENVS:
            env.pop(name, None)
        env.update(
            {
                rht.LEND_GIB_ENV: str(size / GIB),
                rht.CHUNK_GIB_ENV: str(2 * MIB / GIB),
                rht.DIR_ENV: str(tmp_path),
                rht.JOB_ENV: job,
                rht.LENDER_INDEX_ENV: str(index),
                rht.LEND_PARTS_ENV: "2",
            }
        )
        lender = subprocess.Popen(
            [sys.executable, "-c", base.LENDER_SCRIPT],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(lender)
        report = base._read_lender_report(lender, base.LENDER_REPORT_TIMEOUT_S)
        if report is None:
            pytest.skip(f"lender {index} reported neither LENT nor FAILED")
        if not report or report[0] != "LENT":
            pytest.skip(f"fabric host memory unavailable to lender {index}: {report}")
        assert int(report[1]) == size // (2 * MIB)
    parts = []
    for part in range(2):
        part_set = []
        for index in range(len(sizes)):
            with open(rht.handle_path(str(tmp_path), 0, job, index, part)) as f:
                payload = json.load(f)
            assert (payload["part"], payload["num_parts"]) == (part, 2)
            part_set.append(payload)
        parts.append(part_set)
    return parts


def _importer_env(tmp_path, job, sizes) -> dict:
    return {
        rht.TIER_ENV: "1",
        rht.DIR_ENV: str(tmp_path),
        rht.WAIT_S_ENV: "60",
        rht.NUM_LENDERS_ENV: str(len(sizes)),
        rht.EXPECT_GIB_ENV: ",".join(str(size / 2 / GIB) for size in sizes),
        rht.PART_ENV: "auto",
        rht.JOB_ENV: job,
    }


def _two_importers_one_part_each(monkeypatch, tmp_path, sizes) -> None:
    """Lenders of ``sizes`` (2 parts each) and two importers with PART=auto, one in a subprocess
    and one here, started together: each builds a level of one part of every lender, the parts
    differ, every claim is held by its importer, and both promote a demoted chain."""
    torch = base._gpu()
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    job = f"rh-split-{len(sizes)}-{os.getpid()}"
    started: List[subprocess.Popen] = []
    importers: List[subprocess.Popen] = []
    try:
        parts = _start_split_lenders(tmp_path, job, sizes, started)
        importer_env = _importer_env(tmp_path, job, sizes)
        env = dict(os.environ)
        for name in base.ALL_ENVS + SPLIT_ENVS:
            env.pop(name, None)
        env.update(importer_env)
        other = subprocess.Popen(
            [sys.executable, "-c", IMPORTER_SCRIPT, os.path.dirname(os.path.abspath(__file__))],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        importers.append(other)
        lines = _pump(other)
        for name, value in importer_env.items():
            monkeypatch.setenv(name, value)
        try:
            manager = base._make_manager(host_bytes=2 * MIB)
        except Exception as error:
            base._skip_without_fabric_import(error)
            raise
        try:
            part_chunks = sum(size // (2 * MIB) for size in sizes) // 2
            assert list(manager.impl.cache_tier_list) == [
                CacheTier.GPU_MEM,
                CacheTier.HOST_MEM,
                CacheTier.HOST_MEM,
            ]
            stats = manager.impl.get_remote_host_tier_stats()
            assert stats["level"] == 2 and stats["num_chunks"] == part_chunks
            mine = rht.merged_import().part
            assert mine in (0, 1)
            report = _read_report(
                other, lines, ("IMPORTED", "FAILED"), base.LENDER_REPORT_TIMEOUT_S
            )
            assert report and report[0] == "IMPORTED", report
            assert int(report[1]) == 1 - mine
            assert (int(report[2]), int(report[3])) == (2, part_chunks)
            # Every lender's two parts are claimed, one by each importer.
            for part, part_set in enumerate(parts):
                for index, payload in enumerate(part_set):
                    path = rht.claim_path(str(tmp_path), 0, job, payload["nonce"], index, part)
                    with open(path) as f:
                        owner = json.load(f)["owner"]["pid"]
                    assert owner == (os.getpid() if part == mine else other.pid)
            base._demote_and_promote(manager, torch)
            report = _read_report(
                other, lines, ("PROMOTED", "FAILED"), base.LENDER_REPORT_TIMEOUT_S
            )
            assert report and report[0] == "PROMOTED" and int(report[1]) > 0, report
        finally:
            manager.shutdown()
    finally:
        # The importers let go of the chunks before the lenders exit.
        base._stop_lenders(importers)
        base._stop_lenders(started)


def test_one_lender_two_parts_two_importers(monkeypatch, tmp_path):
    _two_importers_one_part_each(monkeypatch, tmp_path, [16 * MIB])


def test_two_lenders_two_parts_two_importers(monkeypatch, tmp_path):
    _two_importers_one_part_each(monkeypatch, tmp_path, [16 * MIB, 8 * MIB])


def test_stale_part_fails_the_start(monkeypatch, tmp_path):
    """A part whose handle file does not match its chunk words fails the start and names the
    lender and the part."""
    base._gpu()
    job = f"rh-split-stale-{os.getpid()}"
    started: List[subprocess.Popen] = []
    try:
        parts = _start_split_lenders(tmp_path, job, [16 * MIB, 8 * MIB], started)
        stale = dict(parts[1][1], nonce=parts[1][1]["nonce"] ^ 0x5A)
        rht.publish(str(tmp_path), 0, stale, job, lender_index=1, part=1)
        for name, value in _importer_env(tmp_path, job, [16 * MIB, 8 * MIB]).items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv(rht.PART_ENV, "1")
        with pytest.raises(RuntimeError) as raised:
            base._make_manager(host_bytes=2 * MIB)
        base._skip_without_fabric_import(raised.value)
        message = str(raised.value)
        assert "lender index 1 part 1 host=" in message and "stale or foreign" in message
        assert rht.merged_import() is None
    finally:
        base._stop_lenders(started)
