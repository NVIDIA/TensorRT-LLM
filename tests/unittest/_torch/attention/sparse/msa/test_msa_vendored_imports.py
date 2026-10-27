# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vendored MSA packages are the only MSA code TensorRT-LLM imports.

The packages live under ``tensorrt_llm._torch.attention.backends.fmha.msa_kernels``; the top-level ``fmha_sm100`` /
``fmha_sm100_q8kv8`` packages they came from are not installed with TensorRT-LLM, so any import of those names fails
at run time (often inside an ``except ImportError`` fallback, where it silently turns a feature off).
"""

import ast
import importlib
import importlib.abc
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import tensorrt_llm

TOP_LEVEL = ("fmha_sm100", "fmha_sm100_q8kv8")
PACKAGE_ROOT = Path(tensorrt_llm.__file__).resolve().parent
MSA_KERNELS = "tensorrt_llm._torch.attention.backends.fmha.msa_kernels"
MINIMAX_M3 = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"


def _top_level(name: str) -> bool:
    return name.split(".")[0] in TOP_LEVEL


def _module_names_in(node: ast.AST):
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        yield node.module
    elif isinstance(node, ast.Call) and node.args:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        arg = node.args[0]
        if name in ("import_module", "find_spec", "__import__") and isinstance(arg, ast.Constant):
            if isinstance(arg.value, str):
                yield arg.value


def test_no_source_file_imports_a_top_level_msa_package():
    hits = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            hits += [
                f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {name}"
                for name in _module_names_in(node)
                if _top_level(name)
            ]
    assert not hits, "imports of the top-level MSA packages:\n" + "\n".join(hits)


class _Recorder(importlib.abc.MetaPathFinder):
    """Records (and refuses) every attempt to import a top-level MSA package name."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if _top_level(fullname):
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"blocked top-level MSA package {fullname}")
        return None


def _modules_to_sweep():
    """Every module under msa_kernels/ and under the MiniMax-M3.1 targets' kernels/ directories."""
    package = importlib.import_module(MSA_KERNELS)
    yield from (
        info.name for info in pkgutil.walk_packages(package.__path__, prefix=f"{MSA_KERNELS}.")
    )
    family = importlib.import_module(MINIMAX_M3)
    for target in sorted(Path(family.__path__[0]).glob("minimax_m31_*/kernels")):
        prefix = f"{MINIMAX_M3}.{target.parent.name}.kernels"
        yield prefix
        yield from (info.name for info in pkgutil.walk_packages([str(target)], prefix=f"{prefix}."))


@pytest.fixture
def blocked_top_level(monkeypatch):
    for name in list(sys.modules):
        if _top_level(name):
            monkeypatch.delitem(sys.modules, name)
    recorder = _Recorder()
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    return recorder


def test_vendored_and_kernel_modules_import_without_the_top_level_packages(blocked_top_level):
    pytest.importorskip("cutlass")
    imported, skipped = [], []
    for name in _modules_to_sweep():
        if ".test_" in name or name.endswith((".build", ".cute")):
            continue  # standalone scripts and the cute/ directory loaded through sys.path
        try:
            importlib.import_module(name)
            imported.append(name)
        except Exception as err:  # noqa: BLE001  (a module this environment cannot load, e.g. without a GPU)
            skipped.append(f"{name}: {type(err).__name__}: {err}")
        if blocked_top_level.attempts:
            break
    assert not blocked_top_level.attempts, f"{name} tried to import {blocked_top_level.attempts}"
    # The vendored packages must import wherever the CuTe DSL does (the targets' kernels may need a device).
    vendored_failures = [entry for entry in skipped if entry.startswith(f"{MSA_KERNELS}.")]
    assert not vendored_failures, "\n".join(vendored_failures)
    assert any(name.endswith("fmha_sm100_q8kv8.q8_indexer_interface") for name in imported), skipped


# A fresh interpreter: nothing imported first (no require_msa_module), the top-level names blocked. Every module file
# under msa_kernels/ is imported, the CuTe modules after sparse.py (they import each other as top-level src.*, which
# sparse.py puts on sys.path), build_k2q_csr only when its prebuilt library exists (it compiles at import otherwise).
_FRESH_SWEEP = r"""
import importlib, importlib.abc, json, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("fmha_sm100", "fmha_sm100_q8kv8"):
            raise ModuleNotFoundError("blocked top-level MSA package " + fullname)
sys.meta_path.insert(0, Block())
import tensorrt_llm  # noqa: F401
package, reader_module = sys.argv[1], sys.argv[2]
root = importlib.util.find_spec(package).submodule_search_locations[0]
libs = os.path.join(os.path.dirname(root), *([os.pardir] * 4), "libs", "msa_fmha_sm100")
names = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for f in sorted(filenames):
        if not f.endswith(".py") or f.startswith("test_") or f == "example.py":
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), root)[: -len(".py")].split(os.sep)
        if rel[-1] == "__init__":
            rel = rel[:-1]
        if "build_k2q_csr" in rel and not os.path.isfile(os.path.join(libs, "k2q", "sparse_build_k2q_csr_ext.so")):
            continue
        names.append(".".join([package] + rel))
names.sort(key=lambda n: (".cute." in n, n))
failed = {}
for name in names:
    try:
        importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        failed[name] = f"{type(e).__name__}: {e}"[:300]
reader = importlib.import_module(reader_module).Nvfp4IndexDecodeScorer(2).available()
print("RESULT " + json.dumps({"imported": len(names) - len(failed), "failed": failed, "reader": reader}))
"""


def test_a_fresh_interpreter_imports_every_vendored_module():
    """Nothing set up by other imports (such as require_msa_module) may be needed first."""
    pytest.importorskip("cutlass")
    reader = f"{MINIMAX_M3}.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_nvfp4_index_decode"
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_SWEEP, MSA_KERNELS, reader],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stderr[-4000:]
    result = json.loads(lines[-1][len("RESULT ") :])
    assert result["imported"] > 0
    assert not result["failed"], result["failed"]
    assert result["reader"] is True


def test_the_cute_name_shim_is_installed_by_the_package_import():
    pytest.importorskip("cutlass")
    code = (
        "import importlib, cutlass.cute as cute\n"
        f"importlib.import_module('{MSA_KERNELS}')\n"
        "print(cute.core.ThrMma is cute.ThrMma, cute.core.ThrCopy is cute.ThrCopy)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.stdout.strip().endswith("True True"), proc.stderr[-2000:]


def test_the_nvfp4_index_k_decode_reader_is_available(blocked_top_level):
    pytest.importorskip("cutlass")
    module = importlib.import_module(
        f"{MINIMAX_M3}.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_nvfp4_index_decode"
    )
    assert module.Nvfp4IndexDecodeScorer(2).available()
    assert not blocked_top_level.attempts
