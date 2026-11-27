# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for selecting and validating wheel-time MSA artifacts."""

import ast
import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/build_msa_aot.py"
spec = importlib.util.spec_from_file_location("build_msa_aot", _SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize(
    ("requested", "supported", "expected"),
    [
        ("all", {"sm_100a", "sm_103a", "sm_100f"}, ["sm_100a", "sm_103a"]),
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


def test_nvcc_probe_detects_architecture_specific_targets(tmp_path):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("""#!/usr/bin/env python3
import sys
if "-arch=sm_100f" in sys.argv:
    print("nvcc fatal: Unsupported gpu architecture 'compute_100f'", file=sys.stderr)
    raise SystemExit(1)
assert "-c" in sys.argv
""")
    nvcc.chmod(0o755)
    assert builder.probe_cuda_architectures(nvcc, ["sm_100a", "sm_103a", "sm_100f"]) == {
        "sm_100a",
        "sm_103a",
    }


def test_nvcc_probe_does_not_hide_compiler_failure(tmp_path):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text('#!/bin/sh\necho "host compiler failed" >&2\nexit 1\n')
    nvcc.chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError, match="nvcc"):
        builder.probe_cuda_architectures(nvcc, ["sm_103a"])


@pytest.mark.parametrize(
    ("system", "requested", "skip_wheel", "expected"),
    [
        ("Linux", "all", False, True),
        ("Linux", "90;100", False, True),
        ("Linux", "90", False, False),
        ("Windows", "100", False, False),
        ("Linux", "100", True, False),
    ],
)
def test_wheel_build_is_gated_by_platform_and_architecture(system, requested, skip_wheel, expected):
    assert builder.should_build(system, requested, skip_wheel) is expected


def test_native_requires_architectures_recorded_by_cmake(tmp_path):
    with pytest.raises(ValueError, match="reconfigure"):
        builder.resolve_requested_architectures("native", tmp_path)
    assert builder.resolve_requested_architectures("100", tmp_path) == "100"


@pytest.mark.parametrize(("requested", "configured"), [("90", "103"), ("103", "90")])
def test_final_cmake_architecture_overrides_cli(tmp_path, requested, configured):
    (tmp_path / "CMakeCache.txt").write_text(f"CMAKE_CUDA_ARCHITECTURES:STRING={configured}\n")
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
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (build / "trtllm_cuda_architectures.txt").read_text() == "103"
    assert builder.resolve_requested_architectures("90", build) == "103"


def test_cli_skips_non_blackwell_without_nvcc(tmp_path):
    env = {"PATH": ""}
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(tmp_path),
            str(tmp_path / "unused.py"),
            "90",
            "1",
            "--build-dir",
            str(tmp_path),
        ],
        env=env,
        check=True,
        timeout=60,
    )
    assert list(tmp_path.iterdir()) == []


def _shared_library(path, symbols):
    source = path.with_suffix(".c")
    source.write_text("\n".join(f"void {symbol}(void) {{}}" for symbol in symbols))
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(path)], check=True, timeout=60)


def test_artifact_validation_checks_each_export_and_helper(tmp_path):
    files = {
        name: tmp_path / f"{name}.so" for name in ("variants", "plan", "sparse_topk", "reduction")
    }
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
    (package / "jit.py").write_text("""
import os
from pathlib import Path
CACHE_BASE = Path(os.environ["MINFER_FMHA_CACHE_DIR"]) / "fingerprint"
def _aot_variant_names(): return ["0_0", "0_1"]
def _aot_source_digest(): return "sourcehash"
def _aot_ffi_version(): return "ffi-version"
def _get_nvcc_flags(*args): return "-gencode=arch=compute_100a,code=sm_100a -gencode=arch=compute_103a,code=sm_103a"
""")
    warmup = tmp_path / "warmup.py"
    warmup.write_text("""
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
""")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(
        "#!/bin/sh\nif [ \"$1\" = --version ]; then printf 'CUDA release 12.9\\n'; "
        "else printf 'sm_100a\\nsm_103a\\n'; fi\n"
    )
    nvcc.chmod(0o755)
    user_cache = tmp_path / "user-cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(user_cache))
    monkeypatch.setenv("GPU_TRACE", "1")

    builder.build(package, warmup, "100", nvcc, jobs=2)

    manifest_path = package / "_aot/sourcehash/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["variants"] == ["0_0", "0_1"]
    assert manifest["cuda_architectures"] == ["sm_100a"]
    assert manifest["nvcc_version"] == "CUDA release 12.9"
    assert all((manifest_path.parent / name).is_file() for name in manifest["files"].values())
    assert not user_cache.exists()


def test_installed_package_builder_loader_round_trip(tmp_path, monkeypatch):
    """CI exercises the real installed jit and worker without CUDA or an external warmup path."""
    spec = importlib.util.find_spec("fmha_sm100")
    assert spec is not None and spec.submodule_search_locations, "installed MSA package is required"
    source = Path(next(iter(spec.submodule_search_locations)))
    package = tmp_path / "fmha_sm100"
    package.mkdir()
    shutil.copy2(source / "jit.py", package / "jit.py")
    shutil.copytree(source / "csrc", package / "csrc")
    headers = package / "cutlass/include/cutlass"
    headers.mkdir(parents=True)
    (headers / "version.h").write_text(
        "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 4\n#define CUTLASS_PATCH 2\n"
    )
    (package / "cutlass/tools/util/include").mkdir(parents=True)
    warmup = tmp_path / "warmup.py"
    warmup.write_text("""
import os
import subprocess
import fmha_sm100.jit as jit

def _cleanup_nvcc_temps():
    raise AssertionError("unsafe cleanup")

def main():
    _cleanup_nvcc_temps()
    assert os.environ["MINFER_FMHA_CACHE_DIR"].startswith(os.environ["EXPECTED_BUILD_DIR"])
    libraries = {
        "_all_variants/all_variants.so": ["run_" + name for name in jit._aot_variant_names()],
        "plan/fmha_sm100_plan.so": ["plan"],
        "sparse_topk/sparse_topk_select.so": ["sparse_topk_select"],
        "reduction/fmha_sm100_reduction.so": ["reduction"],
    }
    for name, symbols in libraries.items():
        target = jit.CACHE_BASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        code = "\\n".join("int __tvm_ffi_" + symbol + "(void) {return 7;}" for symbol in symbols)
        subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-o", str(target), "-"],
                       input=code, text=True, check=True, timeout=60)
""")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("#!/bin/sh\nprintf 'CPU compiler fixture\\n'\n")
    nvcc.chmod(0o755)
    # Native ELF fixtures do not use the real TVM ABI; use controlled package metadata.
    dependencies = tmp_path / "metadata"
    distribution = dependencies / "apache_tvm_ffi-0.0.0.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text("Name: apache-tvm-ffi\nVersion: 0.0.0\n")
    monkeypatch.syspath_prepend(str(dependencies))
    monkeypatch.setenv("PYTHONPATH", str(dependencies))
    build_dir = tmp_path / "configured build"
    monkeypatch.setenv("EXPECTED_BUILD_DIR", str(build_dir))
    builder.build(package, warmup, "100;103", nvcc, jobs=2, build_dir=build_dir)
    jit = builder._load_module("_msa_roundtrip", package / "jit.py")
    monkeypatch.setattr(jit, "_proxy_modes", lambda: (True, True))
    monkeypatch.setattr(jit, "_aot_device_archs", lambda: {"sm_100a", "sm_103a"})
    manifests = list((package / "_aot").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert len(manifest["variants"]) == 350
    for kind in ("variants", "plan", "sparse_topk", "reduction"):
        assert jit._aot_module_path(kind) == manifests[0].parent / manifest["files"][kind]
    assert not list(build_dir.iterdir())


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
    for name in (
        "fmha_sm100_params.h",
        "tvm_ffi_utils.h",
        "gmem_bounds_check.h",
        "fmha_sm100_plan.cu",
        "sparse_topk_select.cu",
        "fmha_sm100_reduction.cu",
    ):
        (csrc / name).write_text("fixture\n")
    (csrc / "include/sparse_topk_select.cuh").write_text("fixture\n")
    for name in ("fmha_sm100_inst.jinja", "fmha_sm100_variant_run.cu.jinja"):
        (csrc / name).write_text("// {{ variant_name }}\n")
    (package / "jit.py").write_text(f"""
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
def _cutlass_dir(): return _CUTLASS_INCLUDE.parent
def _get_cuda_home(): return os.environ["CUDA_HOME"]
class FMHAVariantManager: pass
def _get_nvcc_flags(cache, fmha=True):
    return "-gencode=arch=compute_103a,code=sm_103a " + " ".join(
        "-I" + str(p) for p in (_FMHA_VARLEN_DIR / "include", _CUTLASS_INCLUDE,
                                _CUTLASS_UTIL_INCLUDE, _get_tvm_ffi_include(), cache))
""")
    nvcc = root / "cuda toolkit/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(f"""#!/usr/bin/env python3
import os,subprocess,sys
args = sys.argv[1:]
if "--list-gpu-code" in args:
    print("sm_103a")
    raise SystemExit
if "--version" in args:
    print("CUDA release 12.9")
    raise SystemExit
if any(arg.endswith("/probe.cu") for arg in args):
    raise SystemExit
out = args[args.index("-o") + 1]
if "-c" in args:
    # Relative object paths keep the 700-input aggregate link below ARG_MAX.
    assert not os.path.isabs(out), out
    assert {("-I" + str(csrc / "include"))!r} in args, args
    subprocess.run(["cc", "-x", "c", "-c", "-o", out, "-"], input="", text=True, check=True)
else:
    symbols = {{"all_variants.so": "run_0", "fmha_sm100_plan.so": "plan",
               "sparse_topk_select.so": "sparse_topk_select", "fmha_sm100_reduction.so": "reduction"}}
    name = out.rsplit("/", 1)[-1]
    source = "void __tvm_ffi_" + symbols[name] + "(void) {{}}"
    subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-o", out, "-"], input=source, text=True, check=True)
""")
    nvcc.chmod(0o755)
    temp = root / "private temp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr(builder.tempfile, "tempdir", None)

    builder.build(package, Path(os.environ["MSA_WARMUP_SCRIPT"]), "103", nvcc, jobs=2)

    assert (package / "_aot/sourcehash/manifest.json").is_file()


@pytest.mark.skipif(not os.environ.get("MSA_WARMUP_SCRIPT"), reason="patched MSA source not staged")
def test_real_patched_worker_builds_and_loads_complete_stock_artifacts(tmp_path, monkeypatch):
    """Exercise real dispatch, templates, flags, Ninja, manifest and loaders with native ELF fixtures."""
    warmup = Path(os.environ["MSA_WARMUP_SCRIPT"])
    source = warmup.parents[1] / "python/fmha_sm100"
    root = tmp_path / "build path $with: spaces"
    package = root / "fmha_sm100"
    package.mkdir(parents=True)
    shutil.copy2(source / "jit.py", package / "jit.py")
    shutil.copytree(source / "csrc", package / "csrc")
    headers = package / "cutlass/include/cutlass"
    headers.mkdir(parents=True)
    (headers / "version.h").write_text(
        "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 4\n#define CUTLASS_PATCH 2\n"
    )
    (package / "cutlass/tools/util/include").mkdir(parents=True)
    dependencies = tmp_path / "dependencies"
    ffi = dependencies / "tvm_ffi"
    (ffi / "include").mkdir(parents=True)
    (ffi / "__init__.py").write_text("")
    (dependencies / "torch.py").write_text("raise AssertionError('build imported torch')\n")
    distribution = dependencies / "apache_tvm_ffi-0.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: apache-tvm-ffi\nVersion: 0.0.0\n")
    monkeypatch.setenv("PYTHONPATH", str(dependencies))
    monkeypatch.syspath_prepend(str(dependencies))
    monkeypatch.setenv("CUTLASS_ROOT", str(root / "wrong inherited CUTLASS"))
    monkeypatch.setenv("MINFER_FMHA_CUDA_ARCH", "107a")
    monkeypatch.setenv("TLLM_FMHA_PROXY_CORRECTION_CHAIN", "0")
    monkeypatch.setenv("TLLM_FMHA_PROXY_S_DOUBLE_BUFFER", "1")
    nvcc = root / "cuda toolkit/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("""#!/usr/bin/env python3
import os,re,subprocess,sys
from pathlib import Path
args = sys.argv[1:]
if '--version' in args:
    print('CPU nvcc fixture, no CUDA compilation')
    raise SystemExit
if any(arg.endswith('/probe.cu') for arg in args):
    raise SystemExit
out = args[args.index('-o') + 1]
if '-c' in args:
    assert '-DFMHA_ONLYSCORE_PER_TILE_CORRECTION' in args, args
    assert '-DFMHA_ONLYSCORE_SINGLE_S' in args, args
    assert '-gencode=arch=compute_103a,code=sm_103a' in args, args
    assert not any('compute_107' in arg for arg in args), args
    source = Path(args[args.index('-c') + 1]).name
    match = re.fullmatch(r'run_([0-9_]+)\\.cu', source)
    symbols = {'fmha_sm100_plan.cu': 'plan', 'sparse_topk_select.cu': 'sparse_topk_select',
               'fmha_sm100_reduction.cu': 'reduction'}
    symbol = 'run_' + match[1] if match else symbols.get(source)
    code = 'int __tvm_ffi_' + symbol + '(void) {return 7;}' if symbol else ''
    subprocess.run(['cc','-fPIC','-x','c','-c','-o',out,'-'], input=code, text=True, check=True)
else:
    objects = args[1:args.index('-o')]
    subprocess.run(['cc','-shared',*objects,'-o',out], check=True)
""")
    nvcc.chmod(0o755)
    builder.build(package, warmup, "103", nvcc, jobs=8)
    manifests = list((package / "_aot").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert len(manifest["variants"]) == 350
    assert manifest["proxy_modes"] == [True, True]
    assert manifest["cuda_architectures"] == ["sm_103a"]
    assert manifest["tvm_ffi_version"] == "0.0.0"
    monkeypatch.delenv("CUTLASS_ROOT")
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(root / "unused runtime cache"))
    jit = builder._load_module("_real_msa_stock_runtime", package / "jit.py")
    monkeypatch.setattr(jit, "_device_is_sm107", lambda: False)
    monkeypatch.setattr(jit, "_aot_device_archs", lambda: {"sm_103a"})
    monkeypatch.setattr(jit, "_prebuilt_dir", lambda: None)

    class NativeModule:
        def __init__(self, path):
            self.library = ctypes.CDLL(path)

        def __getattr__(self, name):
            return getattr(self.library, "__tvm_ffi_" + name)

    monkeypatch.setitem(sys.modules, "tvm_ffi", types.SimpleNamespace(load_module=NativeModule))
    manager = jit.FMHAVariantManager()
    variant = manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert variant.run() == 7
    assert jit.get_plan_fn().plan() == 7
    assert jit.get_sparse_topk_module().sparse_topk_select() == 7
    assert jit.get_reduction_module().reduction() == 7
    assert not (root / "unused runtime cache").exists()


def test_wheel_contains_staged_fmha_artifacts(tmp_path):
    """Exercise the real setup.py package-data patterns with setuptools."""
    setup_path = _SCRIPT.parents[1] / "setup.py"
    setup_tree = ast.parse(setup_path.read_text())
    setup_call = next(
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    package_data = next(kw.value for kw in setup_call.keywords if kw.arg == "package_data")
    patterns = next(
        ast.literal_eval(value)
        for key, value in zip(package_data.keys, package_data.values)
        if isinstance(key, ast.Constant) and key.value == "fmha_sm100"
    )
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
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as archive:
        entries = set(archive.namelist())
    assert "fmha_sm100/csrc/kernel.cu" in entries
    assert "fmha_sm100/csrc/include/kernel.h" in entries
    assert "fmha_sm100/cutlass/include/cutlass/cutlass.h" in entries
    for name in ("manifest.json", "all_variants.so", "plan.so", "sparse_topk.so", "reduction.so"):
        assert f"fmha_sm100/_aot/sourcehash/{name}" in entries


@pytest.fixture
def staging_command():
    from setuptools.command.build_py import build_py

    setup_path = _SCRIPT.parents[1] / "setup.py"
    setup_tree = ast.parse(setup_path.read_text())
    command = next(
        node
        for node in setup_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MsaBuildPy"
    )
    setup_call = next(
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    package_data = next(kw.value for kw in setup_call.keywords if kw.arg == "package_data")
    patterns = next(
        ast.literal_eval(value)
        for key, value in zip(package_data.keys, package_data.values)
        if isinstance(key, ast.Constant) and key.value == "fmha_sm100"
    )
    namespace = {"Path": Path, "build_py": build_py}
    exec(compile(ast.Module(body=[command], type_ignores=[]), str(setup_path), "exec"), namespace)
    return namespace["MsaBuildPy"], patterns


def _staging_instance(staging_command, source, build):
    from setuptools.dist import Distribution

    command, patterns = staging_command
    distribution = Distribution(
        {
            "script_name": "setup.py",
            "packages": ["fmha_sm100"],
            "package_dir": {"fmha_sm100": str(source)},
            "package_data": {"fmha_sm100": patterns},
        }
    )
    instance = command(distribution)
    instance.build_lib = str(build)
    instance.ensure_finalized()
    return instance


@pytest.mark.parametrize("layout", ["package_symlink", "build_symlink", "ancestor_symlink"])
def test_wheel_staging_allows_symlinks_without_touching_package_target(
    tmp_path, staging_command, layout
):
    source = tmp_path / "source/fmha_sm100"
    artifact = source / "_aot/hash/kernel.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"keep")
    (source / "__init__.py").touch()
    real_build = tmp_path / "real/build"
    real_build.mkdir(parents=True)
    build = real_build
    if layout == "package_symlink":
        target = tmp_path / "shared"
        target.mkdir()
        (target / "marker").write_text("untouched")
        (build / "fmha_sm100").symlink_to(target, target_is_directory=True)
    elif layout == "build_symlink":
        build = tmp_path / "build-link"
        build.symlink_to(real_build, target_is_directory=True)
    else:
        ancestor = tmp_path / "ancestor-link"
        ancestor.symlink_to(real_build.parent, target_is_directory=True)
        build = ancestor / "build"
    _staging_instance(staging_command, source, build).run()
    staged = build / "fmha_sm100"
    assert not staged.is_symlink()
    assert (staged / "_aot/hash/kernel.so").read_bytes() == b"keep"
    assert artifact.read_bytes() == b"keep"
    if layout == "package_symlink":
        assert (target / "marker").read_text() == "untouched"
        assert not (target / "_aot").exists()


@pytest.mark.parametrize("layout", ["same", "same_through_symlink", "inside", "above"])
def test_wheel_staging_never_cleans_or_stages_inside_source(tmp_path, staging_command, layout):
    source = tmp_path / "source/fmha_sm100"
    artifact = source / "_aot/hash/kernel.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"keep")
    (source / "__init__.py").touch()
    if layout == "same":
        build = source.parent
    elif layout == "same_through_symlink":
        build = tmp_path / "linked-source"
        build.symlink_to(source.parent, target_is_directory=True)
    elif layout == "inside":
        build = source / "fresh-build"
    else:
        build = tmp_path / "build"
        source.rename(tmp_path / "original")
        source = build / "fmha_sm100/nested/fmha_sm100"
        source.parent.mkdir(parents=True)
        (tmp_path / "original").rename(source)
        artifact = source / "_aot/hash/kernel.so"
    with pytest.raises(RuntimeError, match="FMHA source package"):
        _staging_instance(staging_command, source, build).run()
    assert artifact.read_bytes() == b"keep"
    assert not (source / "fresh-build").exists()


def test_editable_build_does_not_clean_symlinked_source(tmp_path, staging_command):
    source = tmp_path / "source/fmha_sm100"
    artifact = source / "_aot/hash/kernel.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"keep")
    (source / "__init__.py").touch()
    build = tmp_path / "build"
    build.mkdir()
    (build / "fmha_sm100").symlink_to(source, target_is_directory=True)
    command = _staging_instance(staging_command, source, build)
    command.editable_mode = True
    command.run()
    assert artifact.read_bytes() == b"keep"
