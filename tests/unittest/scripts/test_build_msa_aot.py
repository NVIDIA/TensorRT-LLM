# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for selecting and validating wheel-time MSA artifacts."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/build_msa_aot.py"
spec = importlib.util.spec_from_file_location("build_msa_aot", _SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize(
    ("requested", "supported", "expected"),
    [
        ("all", {"sm_100a", "sm_103a", "sm_100f"}, ["sm_100a", "sm_103a", "sm_100f"]),
        ("100", {"sm_100a"}, ["sm_100a"]),
        ("103-real", {"sm_100a", "sm_103a"}, ["sm_103a"]),
        ("90;100", {"sm_100a", "sm_103a"}, ["sm_100a"]),
        ("90", {"sm_100a", "sm_103a"}, []),
    ],
)
def test_selects_requested_architectures_supported_by_nvcc(requested, supported, expected):
    assert builder.select_cuda_architectures(requested, supported) == expected


def test_requested_unsupported_sm_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="103"):
        builder.select_cuda_architectures("103", {"sm_100a"})


@pytest.mark.parametrize(
    ("system", "requested", "skip_wheel", "expected"),
    [("Linux", "all", False, True), ("Linux", "90;100", False, True),
     ("Linux", "90", False, False), ("Windows", "100", False, False),
     ("Linux", "100", True, False)],
)
def test_wheel_build_is_gated_by_platform_and_architecture(system, requested, skip_wheel, expected):
    assert builder.should_build(system, requested, skip_wheel) is expected


def test_native_requires_architectures_recorded_by_cmake(tmp_path):
    with pytest.raises(ValueError, match="reconfigure"):
        builder.resolve_requested_architectures("native", tmp_path)
    assert builder.resolve_requested_architectures("100", tmp_path) == "100"


@pytest.mark.parametrize(("requested", "configured"), [("90", "103"), ("103", "90")])
def test_final_cmake_architecture_overrides_cli(tmp_path, requested, configured):
    (tmp_path / "CMakeCache.txt").write_text(
        f"CMAKE_CUDA_ARCHITECTURES:STRING={configured}\n"
    )
    assert builder.resolve_requested_architectures(requested, tmp_path) == configured


def test_native_uses_architectures_recorded_when_cmake_configured(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=native\n")
    (tmp_path / "trtllm_cuda_architectures.txt").write_text("103")
    detector = tmp_path / "detect_cuda_arch"
    detector.write_text("#!/bin/sh\nprintf 90\n")
    detector.chmod(0o755)
    assert builder.resolve_requested_architectures("native", tmp_path) == "103"


def test_cmake_records_resolved_architectures(tmp_path):
    module = _SCRIPT.parents[1] / "cpp/cmake/modules/cuda_configuration.cmake"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(msa_arch NONE)\n"
        "set(CMAKE_CUDA_COMPILER_VERSION 12.9)\n"
        "set(CMAKE_CUDA_ARCHITECTURES 103)\n"
        f'include("{module}")\n'
        "setup_cuda_architectures()\n"
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build)], check=True,
                   capture_output=True, text=True)
    assert (build / "trtllm_cuda_architectures.txt").read_text() == "103"
    assert builder.resolve_requested_architectures("90", build) == "103"


def test_cli_skips_non_blackwell_without_nvcc(tmp_path):
    env = {"PATH": ""}
    subprocess.run([sys.executable, str(_SCRIPT), str(tmp_path), str(tmp_path / "unused.py"),
                    "90", "1", "--build-dir", str(tmp_path)], env=env, check=True)
    assert list(tmp_path.iterdir()) == []


def _shared_library(path, symbols):
    source = path.with_suffix(".c")
    source.write_text("\n".join(f"void {symbol}(void) {{}}" for symbol in symbols))
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(path)], check=True)


def test_artifact_validation_checks_each_export_and_helper(tmp_path):
    files = {name: tmp_path / f"{name}.so" for name in ("variants", "plan", "sparse_topk", "reduction")}
    _shared_library(files["variants"], ["__tvm_ffi_run_0_0", "__tvm_ffi_run_0_1"])
    _shared_library(files["plan"], ["__tvm_ffi_plan"])
    _shared_library(files["sparse_topk"], ["__tvm_ffi_sparse_topk_select"])
    _shared_library(files["reduction"], ["__tvm_ffi_reduction"])
    builder.validate_artifacts(files, ["0_0", "0_1"])

    with pytest.raises(ValueError, match="0_2"):
        builder.validate_artifacts(files, ["0_0", "0_1", "0_2"])
    files["plan"].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.validate_artifacts(files, ["0_0", "0_1"])


def test_build_stages_complete_private_artifacts(tmp_path, monkeypatch):
    package = tmp_path / "fmha_sm100"
    package.mkdir()
    (package / "jit.py").write_text('''
import os
from pathlib import Path
CACHE_BASE = Path(os.environ["MINFER_FMHA_CACHE_DIR"]) / "fingerprint"
def _aot_variant_names(): return ["0_0", "0_1"]
def _aot_source_digest(): return "sourcehash"
def _aot_ffi_version(): return "ffi-version"
def _get_nvcc_flags(*args): return "-gencode=arch=compute_100a,code=sm_100a -gencode=arch=compute_103a,code=sm_103a"
''')
    warmup = tmp_path / "warmup.py"
    warmup.write_text('''
import os
import subprocess
import fmha_sm100.jit as jit
from fmha_sm100.jit import _get_nvcc_flags
def _cleanup_nvcc_temps(): raise AssertionError("unsafe global cleanup called")
def main():
    _cleanup_nvcc_temps()
    flags = _get_nvcc_flags(None)
    assert "sm_100a" in flags and "sm_103a" not in flags
    assert os.environ["TMPDIR"].startswith(os.environ["MINFER_FMHA_CACHE_DIR"])
    assert "GPU_TRACE" not in os.environ
    libraries = {
        "_all_variants/all_variants.so": ["run_0_0", "run_0_1"],
        "plan/fmha_sm100_plan.so": ["plan"],
        "sparse_topk/sparse_topk_select.so": ["sparse_topk_select"],
        "reduction/fmha_sm100_reduction.so": ["reduction"],
    }
    for name, symbols in libraries.items():
        target = jit.CACHE_BASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = target.with_suffix(".c")
        source.write_text("\\n".join("void __tvm_ffi_" + s + "(void) {}" for s in symbols))
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(target)], check=True)
''')
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then printf 'CUDA release 12.9\\n'; "
                    "else printf 'sm_100a\\nsm_103a\\n'; fi\n")
    nvcc.chmod(0o755)
    user_cache = tmp_path / "user-cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(user_cache))
    monkeypatch.setenv("GPU_TRACE", "1")

    builder.build(package, warmup, "100", nvcc, jobs=2)

    manifest_path = package / "_aot/sourcehash/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["variants"] == ["0_0", "0_1"]
    assert manifest["cuda_architectures"] == ["sm_100a"]
    assert manifest["cuda_toolkit_version"] == "CUDA release 12.9"
    assert all((manifest_path.parent / name).is_file() for name in manifest["files"].values())
    assert not user_cache.exists()


@pytest.mark.skipif(not os.environ.get("MSA_WARMUP_SCRIPT"), reason="patched MSA source not staged")
def test_real_warmup_handles_spaces_in_all_build_paths(tmp_path, monkeypatch):
    root = tmp_path / "build path $with: spaces"
    package = root / "fmha_sm100"
    csrc = package / "csrc"
    (csrc / "include").mkdir(parents=True)
    cutlass = package / "cutlass/include"
    cutlass.mkdir(parents=True)
    cutlass_util = package / "cutlass/tools/util/include"
    cutlass_util.mkdir(parents=True)
    ffi_include = root / "ffi include"
    ffi_include.mkdir()
    for name in ("fmha_sm100_params.h", "tvm_ffi_utils.h", "gmem_bounds_check.h",
                 "fmha_sm100_plan.cu", "sparse_topk_select.cu", "fmha_sm100_reduction.cu"):
        (csrc / name).write_text("fixture\n")
    (csrc / "include/sparse_topk_select.cuh").write_text("fixture\n")
    for name in ("fmha_sm100_inst.jinja", "fmha_sm100_variant_run.cu.jinja"):
        (csrc / name).write_text("// {{ variant_name }}\n")
    (package / "jit.py").write_text(f'''
import os
from pathlib import Path
CACHE_BASE = Path(os.environ["MINFER_FMHA_CACHE_DIR"]) / "fingerprint"
_FMHA_VARLEN_DIR = Path({str(csrc)!r})
_CUTLASS_INCLUDE = Path({str(cutlass)!r})
_CUTLASS_UTIL_INCLUDE = Path({str(cutlass_util)!r})
_FMHA_SM100_DISPATCH = [("only", [("value", {{}})])]
def _FMHA_SM100_IMPOSSIBLE(params): return False
def _aot_variant_names(): return ["0"]
def _aot_source_digest(): return "sourcehash"
def _aot_ffi_version(): return "ffi-version"
def _get_tvm_ffi_include(): return {str(ffi_include)!r}
def _get_cuda_home(): return os.environ["CUDA_HOME"]
class FMHAVariantManager: pass
def _get_nvcc_flags(cache, fmha=True):
    return "-gencode=arch=compute_103a,code=sm_103a " + " ".join(
        "-I" + str(p) for p in (_FMHA_VARLEN_DIR / "include", _CUTLASS_INCLUDE,
                                _CUTLASS_UTIL_INCLUDE, _get_tvm_ffi_include(), cache))
''')
    nvcc = root / "cuda toolkit/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(f'''#!/usr/bin/env python3
import subprocess,sys
args = sys.argv[1:]
if "--list-gpu-code" in args:
    print("sm_103a")
    raise SystemExit
if "--version" in args:
    print("CUDA release 12.9")
    raise SystemExit
out = args[args.index("-o") + 1]
if "-c" in args:
    assert {('-I' + str(csrc / 'include'))!r} in args, args
    subprocess.run(["cc", "-x", "c", "-c", "-o", out, "-"], input="", text=True, check=True)
else:
    symbols = {{"all_variants.so": "run_0", "fmha_sm100_plan.so": "plan",
               "sparse_topk_select.so": "sparse_topk_select", "fmha_sm100_reduction.so": "reduction"}}
    name = out.rsplit("/", 1)[-1]
    source = "void __tvm_ffi_" + symbols[name] + "(void) {{}}"
    subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-o", out, "-"], input=source, text=True, check=True)
''')
    nvcc.chmod(0o755)
    temp = root / "private temp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr(builder.tempfile, "tempdir", None)

    builder.build(package, Path(os.environ["MSA_WARMUP_SCRIPT"]), "103", nvcc, jobs=2)

    assert (package / "_aot/sourcehash/manifest.json").is_file()


def test_wheel_contains_staged_fmha_artifacts(tmp_path):
    """Exercise the real setup.py package-data patterns with setuptools."""
    setup_path = _SCRIPT.parents[1] / "setup.py"
    setup_tree = ast.parse(setup_path.read_text())
    setup_call = next(node for node in ast.walk(setup_tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "setup")
    package_data = next(kw.value for kw in setup_call.keywords if kw.arg == "package_data")
    patterns = next(ast.literal_eval(value) for key, value in zip(package_data.keys, package_data.values)
                    if isinstance(key, ast.Constant) and key.value == "fmha_sm100")
    package = tmp_path / "fmha_sm100"
    artifacts = package / "_aot/sourcehash"
    artifacts.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "csrc/include").mkdir(parents=True)
    (package / "csrc/kernel.cu").write_text("kernel fixture")
    (package / "csrc/include/kernel.h").write_text("header fixture")
    (package / "cutlass/include/cutlass").mkdir(parents=True)
    (package / "cutlass/include/cutlass/cutlass.h").write_text("cutlass fixture")
    for name in ("manifest.json", "all_variants.so", "plan.so", "sparse_topk.so", "reduction.so"):
        (artifacts / name).write_text("fixture")
    (tmp_path / "setup.py").write_text(
        "from setuptools import setup\n"
        f"setup(name='msa-package-check', version='0.0.1', packages=['fmha_sm100'], "
        f"package_data={{'fmha_sm100': {patterns!r}}})\n"
    )
    subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as archive:
        entries = set(archive.namelist())
    assert "fmha_sm100/csrc/kernel.cu" in entries
    assert "fmha_sm100/csrc/include/kernel.h" in entries
    assert "fmha_sm100/cutlass/include/cutlass/cutlass.h" in entries
    for name in ("manifest.json", "all_variants.so", "plan.so", "sparse_topk.so", "reduction.so"):
        assert f"fmha_sm100/_aot/sourcehash/{name}" in entries
