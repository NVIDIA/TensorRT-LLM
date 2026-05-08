# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Modules in tensorrt_llm/_torch/models/ and tensorrt_llm/inputs/ must
import cleanly without torchvision installed.

torchvision is an optional dependency of TRT-LLM: only a handful of multimodal
/ image-preprocessing call sites need it, and those are expected to use
lazy_loader to import on first actual use. This test pins that contract: any
future regression that re-introduces a top-level torchvision import in one
of the audited directories will fail loudly here rather than silently making
torchvision a hard dependency of TRT-LLM.

The audited set is computed by globbing ``*.py`` (excluding
``__init__.py``) under the directories listed in ``_AUDITED_DIRS`` so
any new file dropped into those directories is automatically covered.

Mechanics:
  * ONE Python subprocess is spawned per pytest session.  It marks
    ``torchvision`` (and any cached submodules) as ``None`` in
    ``sys.modules`` -- Python's documented contract makes this raise
    ``ModuleNotFoundError`` on any subsequent ``import torchvision``
    while still letting ``importlib.util.find_spec("torchvision")``
    return ``None`` (third-party libraries notably ``transformers``
    probe availability with ``find_spec`` and would surface a raised
    exception otherwise).
  * The subprocess attempts ``importlib.import_module(m)`` on every
    audited module in turn and serialises the per-module outcome as
    JSON on stdout.  A subprocess is required because once any
    torchvision module is loaded into the parent pytest process,
    neither ``sys.modules`` manipulation nor a meta-path finder can
    un-load it.
  * Pytest parametrises one test per module so failures are reported
    with the offending module's name and full traceback.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Resolve repo paths off this file so the test doesn't have to import
# tensorrt_llm at collection time (which would itself fail in a
# torchvision-less env, defeating the test).  This file lives at
# ``<repo>/tests/unittest/test_lazy_torchvision_imports.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRTLLM_PKG = _REPO_ROOT / "tensorrt_llm"

# Directories whose .py files must import cleanly with torchvision absent.
# Add new directories here when their contract should also be enforced.
_AUDITED_DIRS = [
    _TRTLLM_PKG / "_torch" / "models",
    _TRTLLM_PKG / "inputs",
]


def _enumerate_modules() -> list[str]:
    """Glob ``*.py`` under each audited directory and return the
    corresponding dotted module names.  ``__init__.py`` files are
    excluded -- they are exercised transitively by any submodule
    import in the same package, so adding them here would just
    duplicate failures."""
    out: list[str] = []
    for d in _AUDITED_DIRS:
        if not d.exists():
            raise RuntimeError(
                f"audited directory {d} not found; the test layout has "
                f"shifted and _AUDITED_DIRS needs updating"
            )
        for path in sorted(d.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            rel = path.relative_to(_REPO_ROOT).with_suffix("")
            out.append(".".join(rel.parts))
    if not out:
        raise RuntimeError(
            f"no modules discovered under {[str(d) for d in _AUDITED_DIRS]}; "
            f"this test would silently pass with no coverage"
        )
    return out


# Computed once at collection time so pytest can parametrize and so
# the per-session subprocess sees the exact same module list.
MODULES_REQUIRING_LAZY_TORCHVISION = _enumerate_modules()


_PROBE = textwrap.dedent(
    """\
    import importlib, json, sys, traceback

    sys.modules["torchvision"] = None
    for k in [k for k in list(sys.modules) if k.startswith("torchvision.")]:
        sys.modules[k] = None
    importlib.invalidate_caches()

    modules = json.loads({modules_json!r})
    results = {{}}
    for m in modules:
        try:
            importlib.import_module(m)
            results[m] = None
        except BaseException:
            results[m] = traceback.format_exc()
            # Clean up the failed module entry so an unrelated module
            # imported next doesn't observe its partially-initialised
            # state.  Parent packages may still be in a broken state if
            # their __init__ fired the failing import; we accept that
            # downstream failures may share the same root cause.
            sys.modules.pop(m, None)
    # Write to a tempfile path the parent passed in -- stdout is
    # unreliable because tensorrt_llm and friends print banners
    # ("[TensorRT-LLM] TensorRT LLM version: ...") that would
    # contaminate parsing.
    with open({outfile!r}, "w") as _fh:
        _fh.write(json.dumps(results))
    """
)


@pytest.fixture(scope="session")
def _module_import_results(tmp_path_factory):
    """Run all audited imports in ONE subprocess and return a dict of
    ``{module: error_traceback_or_None}``.

    Single-subprocess design is deliberate: the heavy ``import
    tensorrt_llm`` cost (transformers, torch, etc.) is paid once
    instead of once-per-module, taking the run from ~minutes to ~10s.
    """
    outfile = tmp_path_factory.mktemp("tv_probe") / "results.json"
    script = _PROBE.format(
        modules_json=json.dumps(MODULES_REQUIRING_LAZY_TORCHVISION),
        outfile=str(outfile),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    # The probe writes its JSON to a tempfile *before* the interpreter
    # tears down.  Some C-extension shutdown paths in this stack
    # (notably libnccl's atexit / __cxa_finalize) can abort the
    # process at exit even after the probe finished successfully, so
    # we trust the JSON file if it parses regardless of the subprocess
    # exit code.  Only treat the run as a hard probe failure when the
    # tempfile is missing or unparseable.
    if not outfile.exists():
        pytest.fail(
            "torchvision-hidden import probe did not produce a result "
            f"file (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )
    try:
        return json.loads(outfile.read_text())
    except json.JSONDecodeError as e:
        pytest.fail(
            f"torchvision-hidden import probe wrote unparseable JSON: {e}\n"
            f"file contents: {outfile.read_text()!r}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )


@pytest.mark.parametrize("module", MODULES_REQUIRING_LAZY_TORCHVISION)
def test_module_imports_without_torchvision(module, _module_import_results):
    """The module must load cleanly when torchvision is unavailable."""
    error = _module_import_results.get(module, "MISSING_FROM_PROBE")
    if error is None:
        return
    pytest.fail(
        f"{module} failed to import without torchvision installed.\n\n"
        f"This means {module} (or one of its module-level imports) is "
        f"calling ``import torchvision`` / ``from torchvision...`` at "
        f"import time.  Move that import inside the function that uses "
        f"it so torchvision stays an optional runtime dependency.\n\n"
        f"Subprocess traceback:\n{error}"
    )
