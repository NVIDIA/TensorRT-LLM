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
"""Tests for the offline import fallback of the standalone CuTe DSL drivers.

The "install" and the "checkout" are small fake ``tensorrt_llm`` trees under
``tmp_path``. ``install`` changes ``sys.modules``, so every test calls it in a
fresh interpreter started with ``-I -S``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
LOADER_PATH = REPO_ROOT / "tests" / "microbenchmarks" / "cute_dsl_kernels" / "_offline_loader.py"
VERBOSE_ENV = "TRTLLM_OFFLINE_LOADER_VERBOSE"

PROBE = "tensorrt_llm._torch.moe.kernels.probe"

# Body of ``tensorrt_llm/__init__.py`` and ``tensorrt_llm/_torch/__init__.py`` in
# both trees: each reports on stderr that it ran.
_SEEDED_INIT = "import sys\nsys.stderr.write('[init] {} {}\\n')\n"

# Runs in the child interpreter: imports ``pre_import``, optionally attempts
# ``first_attempt`` (the drivers' first import, expected to fail), calls
# ``install`` twice, and reports the resulting state as JSON on stdout.
_CHILD = r"""
import importlib
import importlib.util
import json
import sys

spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["site"])
loader_spec = importlib.util.spec_from_file_location("_offline_loader", spec["loader"])
loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(loader)


def package_paths():
    return {
        name: list(vars(module)["__path__"])
        for name, module in sorted(sys.modules.items())
        if name.split(".")[0] == "tensorrt_llm" and "__path__" in vars(module)
    }


report = {}
for name in spec["pre_import"]:
    importlib.import_module(name)
if spec["first_attempt"]:
    try:
        importlib.import_module(spec["first_attempt"])
    except ImportError as exc:
        report["first_attempt"] = type(exc).__name__
    else:
        report["first_attempt"] = "imported"

loader.install(spec["checkout"])
report["stub"] = {
    name: "__file__" not in vars(sys.modules[name])
    for name in ("tensorrt_llm", "tensorrt_llm._torch")
}
report["attached"] = vars(sys.modules["tensorrt_llm"]).get("_torch") is sys.modules[
    "tensorrt_llm._torch"
]
report["files"] = {
    name: importlib.import_module(name).__file__
    for name in spec["pre_import"] + spec["import_after"]
}
report["paths"] = package_paths()
loader.install(spec["checkout"])
report["paths_after_second_install"] = package_paths()
print(json.dumps(report))
"""


def _write(root, relpath, text=""):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def trees(tmp_path):
    """Build a fake install without the kernel trees, and a checkout that has them."""
    site = (tmp_path / "site").resolve()
    for relpath in (
        "tensorrt_llm/bindings.py",
        "tensorrt_llm/executor/__init__.py",
        "tensorrt_llm/executor/both.py",
        "tensorrt_llm/executor/sub/__init__.py",
        "tensorrt_llm/_torch_sibling/__init__.py",
        "tensorrt_llm/_torch/install_only/__init__.py",
        "tensorrt_llm/_torch/moe/__init__.py",
        "tensorrt_llm/_torch/moe/both.py",
        "tensorrt_llm/_torch/moe/nested/__init__.py",
    ):
        _write(site, relpath)
    # A module whose module-level ``__getattr__`` raises for every name,
    # ``__path__`` included.
    _write(
        site,
        "tensorrt_llm/_torch/raising.py",
        "def __getattr__(name):\n    raise RuntimeError(name)\n",
    )

    checkout = (tmp_path / "checkout").resolve()
    for root, tree in ((site, "install"), (checkout, "checkout")):
        for package in ("tensorrt_llm", "tensorrt_llm._torch"):
            relpath = Path(*package.split(".")) / "__init__.py"
            _write(root, relpath, _SEEDED_INIT.format(tree, package))
    for relpath in (
        "tensorrt_llm/executor/__init__.py",
        "tensorrt_llm/executor/both.py",
        "tensorrt_llm/executor/sub/__init__.py",
        "tensorrt_llm/_torch_sibling/__init__.py",
        "tensorrt_llm/_torch/moe/__init__.py",
        "tensorrt_llm/_torch/moe/both.py",
        "tensorrt_llm/_torch/moe/nested/__init__.py",
        "tensorrt_llm/_torch/moe/kernels/__init__.py",
        "tensorrt_llm/_torch/moe/kernels/probe.py",
    ):
        _write(checkout, relpath)
    return site, checkout


def _run_child(site, checkout, *, pre_import, first_attempt, import_after, verbose=False):
    spec = {
        "site": str(site),
        "checkout": str(checkout),
        "loader": str(LOADER_PATH),
        "pre_import": pre_import,
        "first_attempt": first_attempt,
        "import_after": import_after,
    }
    env = {key: value for key, value in os.environ.items() if key != VERBOSE_ENV}
    if verbose:
        env[VERBOSE_ENV] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", _CHILD, json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _run(site, checkout, **kwargs):
    result = _run_child(site, checkout, **kwargs)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result.stderr


def _init_lines(stderr):
    return [line for line in stderr.splitlines() if line.startswith("[init]")]


def _seeded_path_lines(paths):
    """The lines one verbose ``install`` call prints for the two seeded packages."""
    return [
        "[offline-loader] %s.__path__ = %r" % (name, paths[name])
        for name in ("tensorrt_llm", "tensorrt_llm._torch")
    ]


def _loader_lines(stderr):
    return [line for line in stderr.splitlines() if line.startswith("[offline-loader]")]


def _origin(path, site, checkout):
    path = Path(path)
    if path.is_relative_to(checkout):
        return "checkout"
    if path.is_relative_to(site):
        return "install"
    return str(path)


@pytest.mark.parametrize("verbose", [False, True])
def test_install_after_failed_first_import(trees, verbose):
    """The driver path: a first import from the install fails, then ``install`` runs."""
    site, checkout = trees
    report, stderr = _run(
        site,
        checkout,
        pre_import=[
            "tensorrt_llm._torch.install_only",
            "tensorrt_llm._torch.moe",
            "tensorrt_llm._torch.moe.nested",
            "tensorrt_llm._torch.raising",
            "tensorrt_llm._torch_sibling",
            "tensorrt_llm.executor",
            "tensorrt_llm.executor.sub",
        ],
        first_attempt=PROBE,
        import_after=[
            PROBE,
            "tensorrt_llm._torch.moe.both",
            "tensorrt_llm.executor.both",
            "tensorrt_llm.bindings",
        ],
        verbose=verbose,
    )

    assert report["first_attempt"] == "ModuleNotFoundError"
    # Already imported from the install, so extended rather than replaced by stubs.
    assert report["stub"] == {"tensorrt_llm": False, "tensorrt_llm._torch": False}
    assert _init_lines(stderr) == [
        "[init] install tensorrt_llm",
        "[init] install tensorrt_llm._torch",
    ]
    origins = {name: _origin(path, site, checkout) for name, path in report["files"].items()}
    assert origins == {
        # Imported before ``install``: stay the install's copies.
        "tensorrt_llm._torch.install_only": "install",
        "tensorrt_llm._torch.moe": "install",
        "tensorrt_llm._torch.moe.nested": "install",
        "tensorrt_llm._torch.raising": "install",
        "tensorrt_llm._torch_sibling": "install",
        "tensorrt_llm.executor": "install",
        "tensorrt_llm.executor.sub": "install",
        # Exists only in the checkout.
        PROBE: "checkout",
        # Exists in both; first imported after ``install`` beneath a cached
        # ``tensorrt_llm._torch.*`` package, which now searches the checkout first.
        "tensorrt_llm._torch.moe.both": "checkout",
        # Exists in both, but beneath a cached package outside ``tensorrt_llm._torch``.
        "tensorrt_llm.executor.both": "install",
        # Exists only in the install.
        "tensorrt_llm.bindings": "install",
    }

    paths = report["paths"]
    for name in (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.moe",
        "tensorrt_llm._torch.moe.nested",
    ):
        relpath = Path(*name.split("."))
        assert paths[name] == [str(checkout / relpath), str(site / relpath)], name
    # Cached packages that the checkout lacks, or that lie outside
    # ``tensorrt_llm._torch``, keep their installed ``__path__``.
    for name in (
        "tensorrt_llm._torch.install_only",
        "tensorrt_llm._torch_sibling",
        "tensorrt_llm.executor",
        "tensorrt_llm.executor.sub",
    ):
        assert paths[name] == [str(site / Path(*name.split(".")))], name

    assert report["paths_after_second_install"] == paths
    # The second call changes no cached package, so it prints only the seeded paths.
    assert _loader_lines(stderr) == (
        _seeded_path_lines(paths)
        + [
            "[offline-loader] checkout prepended to cached %s.__path__" % name
            for name in ("tensorrt_llm._torch.moe", "tensorrt_llm._torch.moe.nested")
        ]
        + _seeded_path_lines(paths)
        if verbose
        else []
    )


@pytest.mark.parametrize("verbose", [False, True])
def test_install_before_any_import(trees, verbose):
    """Nothing imported yet: both packages are seeded as stubs over the install."""
    site, checkout = trees
    report, stderr = _run(
        site,
        checkout,
        pre_import=[],
        first_attempt=None,
        import_after=[PROBE, "tensorrt_llm._torch.moe.both", "tensorrt_llm.bindings"],
        verbose=verbose,
    )

    # Neither ``__init__.py`` -- the checkout's nor the install's -- was executed.
    assert _init_lines(stderr) == []
    assert report["stub"] == {"tensorrt_llm": True, "tensorrt_llm._torch": True}
    assert report["attached"]
    origins = {name: _origin(path, site, checkout) for name, path in report["files"].items()}
    assert origins == {
        PROBE: "checkout",
        "tensorrt_llm._torch.moe.both": "checkout",
        "tensorrt_llm.bindings": "install",
    }
    paths = report["paths"]
    for name in ("tensorrt_llm", "tensorrt_llm._torch"):
        relpath = Path(*name.split("."))
        assert paths[name] == [str(checkout / relpath), str(site / relpath)], name
    # Imported by the regular machinery, which keeps only the first directory
    # that has the package.
    assert paths["tensorrt_llm._torch.moe"] == [str(checkout / "tensorrt_llm" / "_torch" / "moe")]
    assert report["paths_after_second_install"] == paths

    # Neither call changes a cached package's ``__path__``.
    assert _loader_lines(stderr) == (_seeded_path_lines(paths) * 2 if verbose else [])


def test_install_rejects_a_directory_without_tensorrt_llm(tmp_path):
    empty = tmp_path.resolve()
    result = _run_child(empty, empty, pre_import=[], first_attempt=None, import_after=[])
    assert result.returncode != 0
    assert "RuntimeError: offline import fallback:" in result.stderr
    assert "is not a TensorRT-LLM checkout" in result.stderr
