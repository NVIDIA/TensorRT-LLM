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
"""Unit tests for the BOLT OSS engine helpers (scripts/bolt).

Covers the pure logic most likely to regress silently:
- manifest.select_workloads: the manifest records the workloads ACTUALLY
  profiled (explicit list) rather than the full suite declaration.
- apply_bolt.profile_for: ELF-basename -> profile-file mapping (.yaml preferred,
  .fdata fallback, multi-dot names like the python bindings, empty/missing).
- apply_bolt.repack_wheel: member permission bits survive the unzip/rezip round
  trip, which zipfile does NOT give us for free.
"""

from __future__ import annotations

import importlib.util
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BOLT_DIR = REPO_ROOT / "scripts" / "bolt"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BOLT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return _load("manifest")


@pytest.fixture(scope="module")
def apply_bolt():
    return _load("apply_bolt")


# --------------------------- manifest.select_workloads ---------------------------
def test_select_workloads_explicit_overrides_suite(manifest, tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text("workloads:\n  - name: from_suite\n")
    # The explicit list (what actually ran) must win over the suite declaration.
    assert manifest.select_workloads("a,b,c", suite) == ["a", "b", "c"]


def test_select_workloads_strips_and_drops_empty(manifest):
    assert manifest.select_workloads(" a , ,b ,", None) == ["a", "b"]


def test_select_workloads_no_arg_no_suite(manifest):
    assert manifest.select_workloads(None, None) == []


def test_select_workloads_falls_back_to_enabled_suite_entries(manifest, tmp_path):
    pytest.importorskip("yaml")
    suite = tmp_path / "suite.yaml"
    suite.write_text("workloads:\n  - name: w_enabled\n  - name: w_disabled\n    enabled: false\n")
    assert manifest.select_workloads(None, suite) == ["w_enabled"]


# ------------------------------ apply_bolt.profile_for ---------------------------
def test_profile_for_prefers_yaml_over_fdata(apply_bolt, tmp_path):
    (tmp_path / "libtensorrt_llm.yaml").write_text("x")
    (tmp_path / "libtensorrt_llm.fdata").write_text("y")
    got = apply_bolt.profile_for("libtensorrt_llm.so", tmp_path)
    assert got is not None and got.name == "libtensorrt_llm.yaml"


def test_profile_for_falls_back_to_fdata(apply_bolt, tmp_path):
    (tmp_path / "libth_common.fdata").write_text("y")
    got = apply_bolt.profile_for("libth_common.so", tmp_path)
    assert got is not None and got.name == "libth_common.fdata"


def test_profile_for_strips_only_trailing_so(apply_bolt, tmp_path):
    # Python bindings carry dots in the stem; only the final `.so` is stripped.
    stem = "bindings.cpython-312-aarch64-linux-gnu"
    (tmp_path / f"{stem}.yaml").write_text("x")
    got = apply_bolt.profile_for(f"{stem}.so", tmp_path)
    assert got is not None and got.name == f"{stem}.yaml"


def test_profile_for_missing_returns_none(apply_bolt, tmp_path):
    assert apply_bolt.profile_for("no_such_lib.so", tmp_path) is None


def test_profile_for_ignores_empty_profile(apply_bolt, tmp_path):
    (tmp_path / "lib.yaml").write_text("")  # zero-size is treated as absent
    assert apply_bolt.profile_for("lib.so", tmp_path) is None


# --------------------------- apply_bolt.is_bolt_applicable -----------------------
def _compile_so(path: Path, emit_relocs: bool) -> bool:
    """Build a trivial .so, optionally BOLT-compatible. False if no toolchain."""
    if shutil.which("gcc") is None:
        return False
    src = path.with_suffix(".c")
    src.write_text("int f(void){ return 1; }\n")
    cmd = ["gcc", "-shared", "-fPIC", "-o", str(path), str(src)]
    if emit_relocs:
        cmd.insert(3, "-Wl,--emit-relocs")
    return subprocess.run(cmd, capture_output=True).returncode == 0


needs_toolchain = pytest.mark.skipif(
    shutil.which("gcc") is None or shutil.which("readelf") is None,
    reason="needs gcc and readelf to produce/inspect a real ELF",
)


@needs_toolchain
def test_is_bolt_applicable_requires_emit_relocs(apply_bolt, tmp_path):
    # The distinction the guard exists for: both are valid ELFs that llvm-bolt
    # will happily process, but only one carries the .rela.text it needs to do
    # so correctly. A build that silently loses ENABLE_BOLT_COMPATIBLE=ON looks
    # exactly like the first one.
    plain = tmp_path / "plain.so"
    assert _compile_so(plain, emit_relocs=False)
    assert apply_bolt.is_bolt_applicable(plain) is False

    compat = tmp_path / "compat.so"
    assert _compile_so(compat, emit_relocs=True)
    assert apply_bolt.is_bolt_applicable(compat) is True


def test_is_bolt_applicable_without_readelf_does_not_block(apply_bolt, tmp_path, monkeypatch):
    # No readelf means "cannot tell", and refusing to optimize on that would
    # turn a missing utility into a build failure.
    def _no_readelf(*_a, **_k):
        raise OSError("readelf not found")

    monkeypatch.setattr(apply_bolt.subprocess, "run", _no_readelf)
    probe = tmp_path / "anything.so"
    probe.write_bytes(b"\x7fELF")
    assert apply_bolt.is_bolt_applicable(probe) is True


# ------------------------------ apply_bolt.repack_wheel --------------------------
def _mode_of(zip_path: Path, member: str) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.getinfo(member).external_attr >> 16


def _build_wheel(path: Path, members: dict) -> dict:
    """Write a zip whose members carry explicit unix modes; return their ZipInfos."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, (data, mode) in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i for i in zf.infolist()}


def test_repack_wheel_preserves_member_modes(apply_bolt, tmp_path):
    src = tmp_path / "src.whl"
    infos = _build_wheel(
        src,
        {
            "pkg/runner.sh": ("#!/bin/sh\n", 0o755),
            "pkg/lib.so": ("\x7fELF-ish", 0o644),
        },
    )
    # Mimic process_wheel: extract (which drops modes), mutate, then repack.
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    (work / "pkg" / "lib.so").write_text("bolted payload")

    out = tmp_path / "out.whl"
    apply_bolt.repack_wheel(work, out, infos)

    # The executable keeps its exec bit; the untouched member keeps its mode.
    assert _mode_of(out, "pkg/runner.sh") & stat.S_IXUSR
    assert stat.S_IMODE(_mode_of(out, "pkg/runner.sh")) == 0o755
    assert stat.S_IMODE(_mode_of(out, "pkg/lib.so")) == 0o644
    with zipfile.ZipFile(out) as zf:
        assert zf.read("pkg/lib.so") == b"bolted payload"


def test_repack_wheel_regression_plain_write_loses_exec_bit(apply_bolt, tmp_path):
    """Guards the reason repack_wheel exists: zf.write() would drop the mode."""
    src = tmp_path / "src.whl"
    _build_wheel(src, {"pkg/runner.sh": ("#!/bin/sh\n", 0o755)})
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    naive = tmp_path / "naive.whl"
    with zipfile.ZipFile(naive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(work / "pkg" / "runner.sh", "pkg/runner.sh")
    assert not _mode_of(naive, "pkg/runner.sh") & stat.S_IXUSR


def test_repack_wheel_falls_back_for_new_members(apply_bolt, tmp_path):
    src = tmp_path / "src.whl"
    infos = _build_wheel(src, {"pkg/lib.so": ("x", 0o644)})
    work = tmp_path / "work"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)
    # A member created after extraction has no original ZipInfo to honor.
    (work / "pkg" / "GENERATED").write_text("new")
    out = tmp_path / "out.whl"
    apply_bolt.repack_wheel(work, out, infos)
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["pkg/GENERATED", "pkg/lib.so"]


# --------------------------- verify_bolted_wheel.py ------------------------------
VERIFY = BOLT_DIR / "internal" / "verify_bolted_wheel.py"


def _wheel_of(path: Path, libs: list[Path], prefix: str = "tensorrt_llm/libs") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for lib in libs:
            zf.write(lib, f"{prefix}/{lib.name}")
    return path


def _verify(wheel: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY), str(wheel)], capture_output=True, text=True, timeout=300
    )


@needs_toolchain
def test_verify_rejects_library_that_crashes_the_loader(tmp_path):
    # The failure this whole check exists for: llvm-bolt exits 0, the wheel is
    # well formed, and the only symptom is a signal raised inside ld.so while it
    # runs .init_array on the first dlopen.
    src = tmp_path / "boom.c"
    src.write_text(
        "__attribute__((constructor)) static void b(void){ *(volatile int*)0x1234 = 1; }\n"
    )
    lib = tmp_path / "libboom.so"
    assert (
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(lib), str(src)], capture_output=True
        ).returncode
        == 0
    )

    result = _verify(_wheel_of(tmp_path / "bad.whl", [lib]))
    assert result.returncode == 2
    assert "CRASHED libboom.so" in result.stdout


@needs_toolchain
def test_verify_accepts_loadable_library(tmp_path):
    lib = tmp_path / "libgood.so"
    assert _compile_so(lib, emit_relocs=False)
    result = _verify(_wheel_of(tmp_path / "good.whl", [lib]))
    assert result.returncode == 0
    assert "loaded libgood.so" in result.stdout


def test_verify_does_not_fail_on_unloadable_but_uncrashing_library(tmp_path):
    # A library whose dependencies this container cannot resolve is the normal
    # case outside a full build image. Reporting it as a BOLT failure would make
    # the check fire on everything, so only a signal is a verdict.
    lib = tmp_path / "libtruncated.so"
    lib.write_bytes(b"\x7fELF")
    result = _verify(_wheel_of(tmp_path / "odd.whl", [lib]))
    assert result.returncode == 0
    assert "UNVERIFIED libtruncated.so" in result.stdout


def test_verify_passes_wheel_with_no_libs(tmp_path):
    empty = tmp_path / "empty.whl"
    zipfile.ZipFile(empty, "w").close()
    assert _verify(empty).returncode == 0


# --------------------------- check_a53_veneers.find_veneers ----------------------
# GNU ld's erratum-843419 workaround leaves a branch out to a stub that performs
# a displaced load and branches straight back. BOLT mistakes that for a tail call
# and clobbers x16 at the return site, so these have to be refused before BOLT
# runs rather than detected afterwards.
EM_AARCH64 = 183
EM_X86_64 = 62


@pytest.fixture(scope="module")
def a53():
    spec = importlib.util.spec_from_file_location(
        "check_a53_veneers", BOLT_DIR / "internal" / "check_a53_veneers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _b(src: int, dst: int) -> int:
    """Encode an unconditional AArch64 `b` from src to dst."""
    return 0x14000000 | (((dst - src) >> 2) & 0x03FFFFFF)


def _elf(path: Path, words: list[int], base: int = 0x1000, machine: int = EM_AARCH64) -> Path:
    """Minimal ELF64 carrying `words` in a single executable section."""
    import struct

    code = b"".join(struct.pack("<I", w) for w in words)
    code_off = 64
    shoff = code_off + len(code)

    hdr = bytearray(64)
    hdr[0:16] = b"\x7fELF\x02\x01\x01\x00" + bytes(8)
    struct.pack_into("<HHI", hdr, 16, 3, machine, 1)  # type, machine, ver
    struct.pack_into("<Q", hdr, 40, shoff)  # e_shoff
    struct.pack_into("<HHHHHH", hdr, 52, 64, 0, 0, 64, 2, 0)

    null_sh = bytes(64)
    text_sh = bytearray(64)
    # name, type=PROGBITS, flags=ALLOC|EXECINSTR, addr, offset, size
    struct.pack_into("<IIQQQQ", text_sh, 0, 0, 1, 0x2 | 0x4, base, code_off, len(code))

    path.write_bytes(bytes(hdr) + code + null_sh + bytes(text_sh))
    return path


def test_find_veneers_detects_erratum_round_trip(a53, tmp_path):
    base = 0x1000
    # b 0x1000 -> 0x1010; 0x1010 does the load; 0x1014 branches back to 0x1004.
    words = [_b(base, base + 0x10)] + [0xD503201F] * 3 + [0xF9400001, _b(base + 0x14, base + 4)]
    elf = _elf(tmp_path / "veneer.so", words, base)
    assert a53.find_veneers(str(elf)) == [(base, base + 0x10)]


def test_find_veneers_ignores_branch_that_does_not_return(a53, tmp_path):
    base = 0x1000
    words = [_b(base, base + 0x10)] + [0xD503201F] * 5
    elf = _elf(tmp_path / "plain.so", words, base)
    assert a53.find_veneers(str(elf)) == []


def test_find_veneers_ignores_non_aarch64(a53, tmp_path):
    base = 0x1000
    words = [_b(base, base + 0x10)] + [0xD503201F] * 3 + [0xF9400001, _b(base + 0x14, base + 4)]
    elf = _elf(tmp_path / "x86.so", words, base, machine=EM_X86_64)
    assert a53.find_veneers(str(elf)) == []


def test_find_veneers_ignores_non_elf(a53, tmp_path):
    junk = tmp_path / "notelf.so"
    junk.write_bytes(b"this is not an ELF file")
    assert a53.find_veneers(str(junk)) == []


def test_bolt_elf_refuses_binary_with_a53_veneers(apply_bolt, tmp_path, monkeypatch):
    """BOLT must never see these: it miscompiles them and still exits 0."""
    base = 0x1000
    words = [_b(base, base + 0x10)] + [0xD503201F] * 3 + [0xF9400001, _b(base + 0x14, base + 4)]
    elf = _elf(tmp_path / "libwith.so", words, base)
    profile = tmp_path / "libwith.yaml"
    profile.write_text("---\n")

    calls = []
    monkeypatch.setattr(apply_bolt, "run", lambda cmd: calls.append(cmd) or (0, ""))

    with pytest.raises(apply_bolt.BoltApplyError, match="cortex-a53-843419"):
        apply_bolt.bolt_elf(elf, profile, [], strip=False, dry_run=False)
    assert calls == [], "llvm-bolt must not run on a binary with these veneers"


def test_is_bolt_applicable_treats_readelf_failure_as_no_verdict(apply_bolt, tmp_path, monkeypatch):
    """A readelf that runs and fails says nothing about ENABLE_BOLT_COMPATIBLE."""

    def _failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="readelf: Error: bad")

    monkeypatch.setattr(apply_bolt.subprocess, "run", _failing)
    probe = tmp_path / "lib.so"
    probe.write_bytes(b"\x7fELF")
    assert apply_bolt.is_bolt_applicable(probe) is True


def test_find_veneers_tolerates_truncated_elf_header(a53, tmp_path):
    """apply_bolt.is_elf only reads 4 bytes, so this reaches find_veneers."""
    full = b"\x7fELF\x02\x01\x01\x00"
    for size in range(4, 64):
        stub = tmp_path / f"trunc{size}.so"
        stub.write_bytes(full.ljust(size, b"\x00")[:size])
        assert a53.find_veneers(str(stub)) == []


def test_bolt_elf_does_not_crash_on_truncated_elf(apply_bolt, tmp_path, monkeypatch):
    elf = tmp_path / "libtrunc.so"
    elf.write_bytes(b"\x7fELF")
    profile = tmp_path / "libtrunc.yaml"
    profile.write_text("---\n")

    def fake_llvm_bolt(cmd):
        # Stand in for the real tool by producing the -o file, so the chmod and
        # os.replace that follow run for real instead of being patched out.
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x7fELF")
        return 0, ""

    monkeypatch.setattr(apply_bolt, "run", fake_llvm_bolt)
    # No veneers to find, so this must reach llvm-bolt rather than raise.
    apply_bolt.bolt_elf(elf, profile, [], strip=False, dry_run=False)
    assert elf.read_bytes() == b"\x7fELF"


@needs_toolchain
def test_verify_checks_libraries_outside_the_libs_directory(tmp_path):
    """process_wheel rglobs the whole tree, so the verifier has to as well."""
    src = tmp_path / "boom2.c"
    src.write_text(
        "__attribute__((constructor)) static void b(void){ *(volatile int*)0x1234 = 1; }\n"
    )
    lib = tmp_path / "libnested.so"
    assert (
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(lib), str(src)], capture_output=True
        ).returncode
        == 0
    )

    wheel = _wheel_of(tmp_path / "nested.whl", [lib], prefix="tensorrt_llm/deep/ep")
    result = _verify(wheel)
    assert result.returncode == 2
    assert "CRASHED libnested.so" in result.stdout
