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
"""CPU-only tests for tensorrt_llm/_rank_local_cache.py (TLLM_K31_RANK_LOCAL_CACHE).

The module and _bootstrap.py are loaded from the source tree into a throwaway package, so these tests run
without importing (or building) tensorrt_llm itself.
"""

import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_SRC = Path(__file__).resolve().parents[3] / "tensorrt_llm"

_ENV_VARS = (
    "TLLM_K31_RANK_LOCAL_CACHE",
    "TLLM_K31_RANK_LOCAL_CACHE_BASE",
    "TLLM_K31_RANK_LOCAL_CACHE_SEED",
    "TLLM_K31_RANK_LOCAL_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRTLLM_CACHE_DIR",
    "SLURM_PROCID",
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_STEP_ID",
    "SLURM_STEPID",
    "SLURM_LOCALID",
)

LATER = 600.0  # pretend this many seconds passed since the test entries were written


def _load_pkg(tmp_path):
    """Copy _bootstrap.py and _rank_local_cache.py into a fresh package and import it."""
    name = f"rlc_pkg_{os.getpid()}_{time.monotonic_ns()}"
    pkg = tmp_path / "pkgs" / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    shutil.copy(_SRC / "_rank_local_cache.py", pkg / "_rank_local_cache.py")
    shutil.copy(_SRC / "_bootstrap.py", pkg / "_bootstrap.py")
    sys.path.insert(0, str(pkg.parent))
    try:
        return (
            importlib.import_module(f"{name}._rank_local_cache"),
            importlib.import_module(f"{name}._bootstrap"),
        )
    finally:
        sys.path.remove(str(pkg.parent))


@pytest.fixture
def mods(tmp_path):
    saved = dict(os.environ)
    for var in _ENV_VARS:
        os.environ.pop(var, None)
    yield _load_pkg(tmp_path)
    os.environ.clear()
    os.environ.update(saved)


def _elf64(payload=b"\x90" * 64):
    """A minimal little-endian ELF64 object: header, one PROGBITS section, section header table."""
    data_off = 64
    shoff = data_off + len(payload)
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4], header[5], header[6] = 2, 1, 1
    struct.pack_into("<HHIQQQIHHHHHH", header, 16, 1, 190, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 2, 0)
    null_sh = bytes(64)
    sh = struct.pack("<IIQQQQIIQQ", 0, 1, 0, 0, data_off, len(payload), 0, 0, 1, 0)
    return bytes(header) + payload + null_sh + sh


def _make_kernel_entry(root, key, name="k", cubin=None, meta=None, extra=None):
    """A Triton-style entry: IR files, metadata JSON, cubin and the group file Triton writes last."""
    d = Path(root) / key
    d.mkdir(parents=True)
    files = {
        f"{name}.ttir": b"ttir",
        f"{name}.ptx": b"ptx",
        f"{name}.json": b'{"name": "k"}' if meta is None else meta,
        f"{name}.cubin": _elf64() if cubin is None else cubin,
    }
    files.update(extra or {})
    for fname, data in files.items():
        (d / fname).write_bytes(data)
    group = {"child_paths": {fname: str(d / fname) for fname in files}}
    (d / f"__grp__{name}.json").write_text(json.dumps(group))
    return d


def _make_future(path, seconds=3600.0):
    """Give an entry's files an mtime in the future: it stays 'young' for any test clock."""
    t = time.time() + seconds
    for p in Path(path).iterdir():
        os.utime(p, (t, t))


def test_switch_off_changes_nothing(mods, monkeypatch, tmp_path):
    rlc, boot = mods
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(tmp_path / "base"))
    for value in (None, "", "0"):
        if value is None:
            monkeypatch.delenv("TLLM_K31_RANK_LOCAL_CACHE", raising=False)
        else:
            monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", value)
        boot._setup_rank_local_compile_cache()
        rlc.setup()
        assert "TRITON_CACHE_DIR" not in os.environ
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
        assert "TLLM_K31_RANK_LOCAL_CACHE_DIR" not in os.environ
    assert not (tmp_path / "base").exists()


def test_bootstrap_calls_it_first(mods):
    _, boot = mods
    import inspect

    body = inspect.getsource(boot._prepare_environment)
    calls = [line.strip() for line in body.splitlines() if line.strip().startswith("_")]
    assert calls[0] == "_setup_rank_local_compile_cache()"
    assert calls.index("_setup_rank_local_compile_cache()") < calls.index("_setup_unified_cache()")


def test_rank_identity(mods):
    rlc, _ = mods
    env = {
        "SLURM_PROCID": "13",
        "SLURM_JOB_ID": "1234",
        "SLURM_STEP_ID": "4",
        "SLURM_LOCALID": "1",
    }
    assert rlc.rank_identity(env) == ("1234.4", "r13", "1")
    assert rlc.rank_identity({}) == ("nojob", f"p{os.getpid()}", "")


def test_elf_complete(mods):
    rlc, _ = mods
    good = _elf64()
    assert rlc.elf_complete(good)
    for cut in (0, 10, 63, 64, 100, len(good) - 1):
        assert not rlc.elf_complete(good[:cut])
    assert not rlc.elf_complete(b"\x00" * len(good))
    # Section data reaching past the end of the file (header table intact) is caught too.
    bad = bytearray(good)
    struct.pack_into("<Q", bad, len(good) - 64 + 32, 10**6)
    assert not rlc.elf_complete(bytes(bad))


def test_seed_copies_only_good_entries(mods, tmp_path):
    rlc, _ = mods
    shared, local = tmp_path / "shared", tmp_path / "local"
    local.mkdir()
    good = _make_kernel_entry(shared, "GOOD")
    _make_kernel_entry(shared, "EMPTYJSON", meta=b"")
    _make_kernel_entry(shared, "BADJSON", meta=b'{"name": ')
    _make_kernel_entry(shared, "SHORTCUBIN", cubin=_elf64()[:150])
    missing = _make_kernel_entry(shared, "MISSINGCHILD")
    (missing / "k.ptx").unlink()
    nogroup = _make_kernel_entry(shared, "NOGROUP")
    (nogroup / "__grp__k.json").unlink()
    (shared / "SOONLY").mkdir()
    (shared / "SOONLY" / "cuda_utils.cpython-312-aarch64-linux-gnu.so").write_bytes(_elf64())
    (shared / "tmp.pid_1_x").mkdir()
    (shared / ".trtllm.host.1.abc").mkdir()
    _make_future(_make_kernel_entry(shared, "YOUNG"))

    now = time.time() + LATER
    stats = rlc.seed(str(shared), str(local), min_age_s=60, now=now)
    assert stats == {"copied": 2, "present": 0, "young": 1, "incomplete": 2, "invalid": 3}
    assert sorted(p.name for p in local.iterdir()) == ["GOOD", "SOONLY"]
    group = json.loads((local / "GOOD" / "__grp__k.json").read_text())
    for fname, path in group["child_paths"].items():
        assert path == str(local / "GOOD" / fname)
        assert Path(path).read_bytes() == (good / fname).read_bytes()
    # Entries written just now are young under the real clock.
    assert rlc.seed(str(shared), str(tmp_path / "other"), min_age_s=60)["copied"] == 0
    # A second pass finds everything in place and changes nothing.
    stats = rlc.seed(str(shared), str(local), min_age_s=60, now=now)
    assert stats["copied"] == 0 and stats["present"] == 2


def test_seed_missing_source(mods, tmp_path):
    rlc, _ = mods
    assert rlc.seed(str(tmp_path / "nope"), str(tmp_path))["missing"] == 1


def test_publish_is_write_once(mods, tmp_path):
    rlc, _ = mods
    local, target = tmp_path / "local", tmp_path / "shared"
    _make_kernel_entry(local, "A")
    _make_kernel_entry(local, "B")
    inflight = _make_kernel_entry(local, "INFLIGHT")
    (inflight / "__grp__k.json").unlink()
    _make_future(_make_kernel_entry(local, "JUSTNOW"))
    target.mkdir()
    (target / "B").mkdir()
    (target / "B" / "marker").write_text("theirs")

    done = set()
    now = time.time() + LATER
    assert rlc.publish_pass(str(local), str(target), done, now=now) == 1
    assert (target / "B" / "marker").read_text() == "theirs"  # never replaced
    assert not (target / "INFLIGHT").exists() and not (target / "JUSTNOW").exists()
    assert {"A", "B"} <= done and not {"INFLIGHT", "JUSTNOW"} & done  # retried on the next pass
    group = json.loads((target / "A" / "__grp__k.json").read_text())
    assert all(p.startswith(str(target / "A") + os.sep) for p in group["child_paths"].values())
    assert not [p for p in target.iterdir() if p.name.startswith(".")]  # no temp dirs left behind
    assert rlc.publish_pass(str(local), str(target), done, now=now) == 0


def test_setup_private_dirs_seed_and_inherit(mods, monkeypatch, tmp_path):
    rlc, boot = mods
    jit = tmp_path / "jit"
    _make_kernel_entry(jit / "triton", "WARM")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", "1")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(tmp_path / "base"))
    monkeypatch.setattr(rlc, "SEED_MIN_AGE_S", 0.0)
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(jit))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_job" / "GEN_0_node"))
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("SLURM_STEP_ID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "1")  # not a publisher

    boot._setup_rank_local_compile_cache()
    root = tmp_path / "base" / "77.3" / "r5"
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    assert os.environ["TLLM_K31_RANK_LOCAL_CACHE_DIR"] == str(root)
    assert (root / "triton" / "WARM" / "__grp__k.json").is_file()
    assert (root / "inductor").is_dir()
    # _setup_unified_cache (setdefault) keeps the private directories.
    boot._setup_unified_cache()
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    # A descendant of this rank keeps the inherited directories, whatever its own Slurm variables say.
    monkeypatch.setenv("SLURM_PROCID", "6")
    rlc.setup()
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert rlc._publisher_thread is None


def test_setup_seed_off(mods, monkeypatch, tmp_path):
    rlc, _ = mods
    jit = tmp_path / "jit"
    _make_kernel_entry(jit / "triton", "WARM")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", "1")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(tmp_path / "base"))
    monkeypatch.setattr(rlc, "SEED_MIN_AGE_S", 0.0)
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_SEED", "0")
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(jit))
    rlc.setup()
    triton = Path(os.environ["TRITON_CACHE_DIR"])
    assert triton.parent.name == f"p{os.getpid()}" and triton.parent.parent.name == "nojob"
    assert list(triton.iterdir()) == []
    # Only local rank 0 of a Slurm step publishes; this process has no Slurm rank.
    assert rlc._publisher_thread is None


def test_setup_unwritable_base_keeps_shared(mods, monkeypatch, tmp_path):
    rlc, _ = mods
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", "1")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(blocker / "sub"))
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(tmp_path / "jit"))
    rlc.setup()
    assert "TRITON_CACHE_DIR" not in os.environ
    assert "TLLM_K31_RANK_LOCAL_CACHE_DIR" not in os.environ


def test_publisher_thread(mods, monkeypatch, tmp_path):
    rlc, _ = mods
    jit = tmp_path / "jit"
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", "1")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(tmp_path / "base"))
    monkeypatch.setattr(rlc, "PUBLISH_EVERY_S", 1.0)
    monkeypatch.setattr(rlc, "PUBLISH_FOR_S", 3.0)
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(jit))
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr(rlc, "PUBLISH_SETTLE_S", 0.0)
    rlc.setup()
    _make_kernel_entry(os.environ["TRITON_CACHE_DIR"], "NEW")
    rlc._publisher_thread.join(timeout=20)
    assert not rlc._publisher_thread.is_alive()
    assert (jit / "triton" / "NEW" / "__grp__k.json").is_file()


@pytest.mark.parametrize(
    "value, on",
    [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True)],
)
def test_switch_values(mods, value, on):
    rlc, _ = mods
    env = {} if value is None else {"TLLM_K31_RANK_LOCAL_CACHE": value}
    assert rlc.enabled(env) is on


def test_seed_from_another_directory(mods, monkeypatch, tmp_path):
    rlc, _ = mods
    warm = tmp_path / "warm"
    _make_kernel_entry(warm, "WARM")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE", "1")
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_BASE", str(tmp_path / "base"))
    monkeypatch.setattr(rlc, "SEED_MIN_AGE_S", 0.0)
    monkeypatch.setenv("TLLM_K31_RANK_LOCAL_CACHE_SEED", str(warm))
    rlc.setup()
    assert (Path(os.environ["TRITON_CACHE_DIR"]) / "WARM" / "__grp__k.json").is_file()


_RACE = r"""
import importlib, os, sys
sys.path.insert(0, sys.argv[1])
rlc = importlib.import_module("rlc_race_mod")
shared, local = sys.argv[2], sys.argv[3]
bad = 0
for _ in range(3):
    rlc.publish_pass(local, shared, set(), settle_s=0)
    os.makedirs(local + ".seeded", exist_ok=True)
    stats = rlc.seed(shared, local + ".seeded", min_age_s=0)
    bad += stats["invalid"] + stats["incomplete"]
print(bad)
"""


def test_concurrent_publish_and_seed(tmp_path):
    """Four processes publish the same keys into one directory while seeding from it: no torn entries."""
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    shutil.copy(_SRC / "_rank_local_cache.py", mod_dir / "rlc_race_mod.py")
    shared = tmp_path / "shared"
    shared.mkdir()
    procs = []
    for w in range(4):
        local = tmp_path / f"local{w}"
        for k in range(40):
            _make_kernel_entry(local, f"K{k:02d}", extra={"pad.bin": os.urandom(4096)})
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", _RACE, str(mod_dir), str(shared), str(local)],
                stdout=subprocess.PIPE,
                text=True,
            )
        )
    outs = [p.communicate(timeout=120)[0].strip() for p in procs]
    assert [p.returncode for p in procs] == [0] * 4 and outs == ["0"] * 4
    assert sorted(p.name for p in shared.iterdir()) == [f"K{k:02d}" for k in range(40)]
    for d in shared.iterdir():
        group = json.loads((d / "__grp__k.json").read_text())
        assert all(
            Path(p).parent == d and Path(p).stat().st_size > 0
            for p in group["child_paths"].values()
        )
