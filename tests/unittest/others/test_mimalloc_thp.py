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
"""The mimalloc THP default that ``import tensorrt_llm`` applies while torch loads.

``_bootstrap._mimalloc_arenas_off_thp`` sets ``MIMALLOC_ALLOW_THP=0`` around phase 1
and ``import torch``, then withdraws it and puts the process's THP-disable flag back.
The in-process tests drive the context manager; the fresh-interpreter ones import the
package the way a worker process does, because torch is already loaded in this one.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tensorrt_llm import _bootstrap

pytestmark = pytest.mark.cpu_only

_ENV = _bootstrap._MIMALLOC_ALLOW_THP_ENV

# Exit codes of a fresh-interpreter check that finds nothing to check.
_NO_PRCTL = 75
_THP_ALREADY_DISABLED = 76
_NO_ADVISED_ARENA = 77
_SKIP_REASONS = {
    _NO_PRCTL: "PR_GET_THP_DISABLE is unavailable here",
    _THP_ALREADY_DISABLED: "THP is already disabled for processes started here",
    _NO_ADVISED_ARENA: "torch's CPU allocator does not advise MADV_HUGEPAGE here",
}

# Shared start of every check: libc's prctl, the variable's name and a lookup of the
# VmFlags of the mapping that holds an address. Every check needs THP allowed at start.
_PREAMBLE = textwrap.dedent(f"""\
    import ctypes, os, sys
    libc = ctypes.CDLL(None)
    ENV = {_ENV!r}

    def thp_disable_flag():
        return libc.prctl({_bootstrap._PR_GET_THP_DISABLE}, 0, 0, 0, 0)

    def disable_thp():
        assert libc.prctl({_bootstrap._PR_SET_THP_DISABLE}, 1, 0, 0, 0) == 0

    def advised(address):
        inside = False
        for line in open("/proc/self/smaps"):
            fields = line.split()
            if line[:1] in "0123456789abcdef" and "-" in fields[0]:
                low, high = (int(x, 16) for x in fields[0].split("-"))
                inside = low <= address < high
            elif inside and line.startswith("VmFlags:"):
                return "hg" in fields[1:]
        raise AssertionError(f"no mapping holds {{address:#x}}")

    if thp_disable_flag() < 0:
        sys.exit({_NO_PRCTL})
    if thp_disable_flag() != 0:
        sys.exit({_THP_ALREADY_DISABLED})
""")


def _check_in_fresh_interpreter(body: str) -> None:
    """Run the preamble and ``body`` in a fresh interpreter; a skip exit skips the test."""
    code = _PREAMBLE + textwrap.dedent(body)
    result = subprocess.run([sys.executable, "-c", code], timeout=600)
    if result.returncode in _SKIP_REASONS:
        pytest.skip(_SKIP_REASONS[result.returncode])
    assert result.returncode == 0


@pytest.fixture(autouse=True)
def _no_mimalloc_thp_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # The context manager sets the variable with a plain os.environ assignment.
    # monkeypatch restores only what it recorded, and delenv records nothing for
    # an absent variable, so setenv records the original value first and teardown
    # restores it (or removes the variable) whatever a test leaves behind.
    monkeypatch.setenv(_ENV, "")
    monkeypatch.delenv(_ENV)


@pytest.fixture
def _on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_bootstrap.sys, "platform", "linux")


def _fake_thp_disable_flag(monkeypatch: pytest.MonkeyPatch, *reads: int | None) -> list:
    """Make _thp_disable_flag return ``reads`` in turn; return the flags set meanwhile."""
    values = iter(reads)
    monkeypatch.setattr(_bootstrap, "_thp_disable_flag", lambda: next(values))
    written = []
    monkeypatch.setattr(_bootstrap, "_set_thp_disable_flag", written.append)
    return written


@pytest.mark.usefixtures("_on_linux")
def test_default_is_set_for_the_block_and_withdrawn(monkeypatch: pytest.MonkeyPatch) -> None:
    written = _fake_thp_disable_flag(monkeypatch, 0, 0)

    with _bootstrap._mimalloc_arenas_off_thp():
        assert os.environ[_ENV] == "0"

    assert _ENV not in os.environ
    assert written == []


@pytest.mark.usefixtures("_on_linux")
def test_default_is_withdrawn_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    written = _fake_thp_disable_flag(monkeypatch, 0, 1)

    with pytest.raises(ImportError):
        with _bootstrap._mimalloc_arenas_off_thp():
            raise ImportError("torch cannot be imported")

    assert _ENV not in os.environ
    assert written == [0]


@pytest.mark.usefixtures("_on_linux")
def test_explicit_setting_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    written = _fake_thp_disable_flag(monkeypatch)

    with _bootstrap._mimalloc_arenas_off_thp():
        assert os.environ[_ENV] == "1"

    assert os.environ[_ENV] == "1"
    assert written == []


def test_default_is_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_bootstrap.sys, "platform", "win32")

    with _bootstrap._mimalloc_arenas_off_thp():
        assert _ENV not in os.environ


@pytest.mark.usefixtures("_on_linux")
@pytest.mark.parametrize(
    ("before", "after", "restored"),
    [
        (0, 1, 0),  # the torch import disabled THP: undo it
        (0, 0, None),  # nothing changed
        (1, 1, None),  # disabled before TensorRT LLM was imported: keep it
        # disabled except where advised before, and the import made it plain: put it back
        (3, 1, 3),
        (None, 1, None),  # not readable before the import: keep it
        (0, None, None),  # not readable after the import: keep it
    ],
)
def test_thp_disable_flag_is_put_back(
    monkeypatch: pytest.MonkeyPatch, before: int | None, after: int | None, restored: int | None
) -> None:
    written = _fake_thp_disable_flag(monkeypatch, before, after)

    with _bootstrap._mimalloc_arenas_off_thp():
        pass

    assert written == ([] if restored is None else [restored])


def test_refused_flag_change_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    # e.g. a seccomp policy that lets the flag be read but not set
    monkeypatch.setattr(_bootstrap, "_libc_prctl", lambda: lambda *args: -1)

    with pytest.warns(RuntimeWarning, match="huge page disable flag"):
        _bootstrap._set_thp_disable_flag(0)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_THP_DISABLE is Linux-only")
def test_thp_disable_flag_round_trip() -> None:
    # 3 is PR_THP_DISABLE_EXCEPT_ADVISED (Linux 6.18+), possibly inherited from the
    # parent; it is restored exactly like the others.
    original = _bootstrap._thp_disable_flag()
    if original is None:
        pytest.skip(_SKIP_REASONS[_NO_PRCTL])
    assert original in (0, 1, 1 | _bootstrap._PR_THP_DISABLE_EXCEPT_ADVISED)
    try:
        _bootstrap._set_thp_disable_flag(1)
        assert _bootstrap._thp_disable_flag() == 1
        _bootstrap._set_thp_disable_flag(0)
        assert _bootstrap._thp_disable_flag() == 0
    finally:
        _bootstrap._set_thp_disable_flag(original)
    assert _bootstrap._thp_disable_flag() == original


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_THP_DISABLE is Linux-only")
def test_importing_tensorrt_llm_keeps_torch_arenas_off_thp() -> None:
    # Control: torch alone keeps mimalloc's default and advises the mapping of a
    # fresh tensor. Where it does not (torch's allocator is not mimalloc there),
    # there is nothing to keep off, and the test skips.
    _check_in_fresh_interpreter(f"""
        import torch
        keep = torch.empty(64 << 20, dtype=torch.uint8)
        if not advised(keep.data_ptr()):
            sys.exit({_NO_ADVISED_ARENA})
    """)
    # As a worker process starts: importing the package first leaves the mapping
    # unadvised, and neither the process's THP setting nor a later library with its
    # own mimalloc (pyarrow) is affected by the default.
    _check_in_fresh_interpreter("""
        import tensorrt_llm
        assert ENV not in os.environ
        assert thp_disable_flag() == 0
        try:
            import pyarrow
        except ImportError:
            pass
        assert thp_disable_flag() == 0
        import torch
        keep = torch.empty(64 << 20, dtype=torch.uint8)
        assert not advised(keep.data_ptr())
    """)


@pytest.mark.skipif(sys.platform != "linux", reason="the mimalloc default is Linux-only")
@pytest.mark.parametrize(
    "module",
    [
        "torch",  # the torch import raises
        "triton_kernels",  # phase 1 raises, at its vendored triton_kernels import
    ],
)
def test_failed_import_withdraws_the_mimalloc_default(module: str) -> None:
    # The import fails at ``module`` after disabling THP for the process, as torch's
    # mimalloc does when it loads. The default must have been in place there and be
    # gone afterwards, with the flag clear again, as a process that handles the
    # error and goes on sees them.
    _check_in_fresh_interpreter(f"""
        class _Fail:
            mimalloc_allow_thp = None
            def find_spec(self, name, path=None, target=None):
                if name == {module!r}:
                    _Fail.mimalloc_allow_thp = os.environ.get(ENV)
                    disable_thp()
                    raise ImportError(name + " cannot be imported")
        sys.meta_path.insert(0, _Fail())
        try:
            import tensorrt_llm
        except ImportError as e:
            assert str(e) == {module + " cannot be imported"!r}, e
        else:
            raise SystemExit("tensorrt_llm was imported")
        assert _Fail.mimalloc_allow_thp == "0", _Fail.mimalloc_allow_thp
        assert ENV not in os.environ
        assert thp_disable_flag() == 0
    """)
