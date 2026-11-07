# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for loading packaged MSA artifacts."""

import importlib.util
import itertools
import json
import platform
import shutil
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def jit_package(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("fmha_sm100")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("FMHA package must be staged or installed")

    source = Path(next(iter(spec.submodule_search_locations)))
    package = tmp_path / "fmha_sm100"
    package.mkdir()
    shutil.copy2(source / "jit.py", package / "jit.py")
    shutil.copytree(source / "csrc", package / "csrc")
    for relative in ("cutlass/include/cutlass", "cutlass/tools/util/include"):
        (package / relative).mkdir(parents=True)
    (package / "cutlass/include/cutlass/version.h").write_text("test-cutlass\n")
    (package / "cutlass/tools/util/include/util.h").write_text("test-util\n")

    cache = tmp_path / "user-cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(cache))
    for name in ("GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK"):
        monkeypatch.delenv(name, raising=False)

    loaded = []
    ffi = types.ModuleType("tvm_ffi")

    def load_module(path):
        loaded.append(Path(path))
        return types.SimpleNamespace(**{"run_1_0_1_3_1_0_0": lambda value: ("compiled", value)})

    ffi.load_module = load_module
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)
    module_spec = importlib.util.spec_from_file_location("_msa_jit_under_test", package / "jit.py")
    jit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(jit)
    monkeypatch.setattr(jit, "_aot_ffi_version", lambda: "0.1.13.post2", raising=False)
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_103a", raising=False)
    return jit, package, cache, loaded


def _all_variant_names(jit):
    names = []
    dimensions = [values for _, values in jit._FMHA_SM100_DISPATCH]
    for indices in itertools.product(*(range(len(values)) for values in dimensions)):
        params = {}
        for dimension, index in zip(dimensions, indices):
            params.update(dimension[index][1])
        if not jit._FMHA_SM100_IMPOSSIBLE(params):
            names.append("_".join(map(str, indices)))
    return names


def _stage_artifacts(jit, package):
    assert callable(getattr(jit, "_aot_source_digest", None)), (
        "runtime must calculate the packaged source and CUTLASS fingerprint"
    )
    digest = jit._aot_source_digest()
    artifact_dir = package / "_aot" / digest
    artifact_dir.mkdir(parents=True)
    files = {
        "variants": "all_variants.so",
        "plan": "plan.so",
        "sparse_topk": "sparse_topk.so",
        "reduction": "reduction.so",
    }
    for name in files.values():
        (artifact_dir / name).write_bytes(b"fixture library")
    names = _all_variant_names(jit)
    assert len(names) == 350
    manifest = {
        "format_version": 1,
        "source_sha256": digest,
        "tvm_ffi_version": "0.1.13.post2",
        "host_arch": platform.machine(),
        "cuda_architectures": ["sm_100a", "sm_103a", "sm_100f"],
        "variants": names,
        "files": files,
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_dir


def test_packaged_variant_avoids_compiler_and_user_cache(jit_package, monkeypatch):
    jit, package, cache, loaded = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    manager = jit.FMHAVariantManager()
    monkeypatch.setattr(
        manager, "_compile_only", lambda *_: pytest.fail("unexpected JIT compilation")
    )
    monkeypatch.setattr(
        jit.subprocess, "run", lambda *_a, **_k: pytest.fail("unexpected subprocess")
    )

    variant = manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)

    assert variant.run("input") == ("compiled", "input")
    assert loaded == [artifact_dir / "all_variants.so"]
    assert not cache.exists()


def test_packaged_helpers_avoid_compiler_and_user_cache(jit_package, monkeypatch):
    jit, package, cache, loaded = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    for name in ("_do_compile_plan", "_do_compile_sparse_topk", "_do_compile_reduction"):
        monkeypatch.setattr(jit, name, lambda: pytest.fail("unexpected helper compilation"))

    jit.get_plan_fn()
    jit.get_sparse_topk_module()
    jit.get_reduction_module()

    assert loaded == [
        artifact_dir / "plan.so",
        artifact_dir / "sparse_topk.so",
        artifact_dir / "reduction.so",
    ]
    assert not cache.exists()


def test_broken_packaged_library_never_silently_falls_back(jit_package, monkeypatch):
    jit, package, cache, _ = jit_package
    _stage_artifacts(jit, package)
    manager = jit.FMHAVariantManager()
    monkeypatch.setattr(
        manager, "_compile_only", lambda *_: pytest.fail("unexpected JIT compilation")
    )

    def load_broken_library(_path):
        raise OSError("bad library")

    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", load_broken_library)
    for _ in range(2):
        with pytest.raises(OSError, match="bad library"):
            manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert not cache.exists()


def test_packaged_library_missing_export_never_silently_falls_back(jit_package, monkeypatch):
    jit, package, cache, _ = jit_package
    _stage_artifacts(jit, package)
    manager = jit.FMHAVariantManager()
    monkeypatch.setattr(
        manager, "_compile_only", lambda *_: pytest.fail("unexpected JIT compilation")
    )
    monkeypatch.setattr(
        sys.modules["tvm_ffi"], "load_module", lambda _path: types.SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="1_0_1_3_1_0_0"):
        manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert not cache.exists()


def test_removed_packaged_library_never_silently_falls_back(jit_package, monkeypatch):
    jit, package, cache, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    assert jit._aot_module_path("variants") == artifact_dir / "all_variants.so"
    (artifact_dir / "all_variants.so").unlink()
    manager = jit.FMHAVariantManager()
    monkeypatch.setattr(
        manager, "_compile_only", lambda *_: pytest.fail("unexpected JIT compilation")
    )
    with pytest.raises(FileNotFoundError):
        manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert not cache.exists()


def test_packaged_source_verification_is_reused(jit_package, monkeypatch):
    jit, package, _, _ = jit_package
    _stage_artifacts(jit, package)
    digest = jit._aot_source_digest
    calls = []

    def count_digest():
        calls.append(1)
        return digest()

    monkeypatch.setattr(jit, "_aot_source_digest", count_digest)
    for kind in ("variants", "plan", "sparse_topk", "reduction"):
        assert jit._aot_module_path(kind) is not None
    assert len(calls) == 1


def test_jit_fallback_cache_tracks_cutlass_headers(jit_package):
    jit, package, _, _ = jit_package
    original = jit._compute_cache_base()
    (package / "cutlass/include/cutlass/version.h").write_text("modified-cutlass\n")
    assert jit._compute_cache_base() != original


def test_jit_fallback_cache_tracks_ffi_version(jit_package, monkeypatch):
    jit, _, _, _ = jit_package
    original = jit._compute_cache_base()
    monkeypatch.setattr(jit, "_aot_ffi_version", lambda: "different")
    assert jit._compute_cache_base() != original


def test_jit_fallback_cache_isolated_by_host_architecture(jit_package, monkeypatch):
    jit, _, _, _ = jit_package
    monkeypatch.setattr(jit.platform, "machine", lambda: "x86_64")
    original = jit._compute_cache_base()
    monkeypatch.setattr(jit.platform, "machine", lambda: "aarch64")
    assert jit._compute_cache_base() != original


def test_source_install_without_artifacts_retains_jit_fallback(jit_package, monkeypatch):
    jit, _, cache, loaded = jit_package
    manager = jit.FMHAVariantManager()

    def compile_fixture(name, _params):
        path = cache / jit.CACHE_BASE.name / name / f"{name}.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(manager, "_compile_only", compile_fixture)
    variant = manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)

    assert variant.run("input") == ("compiled", "input")
    assert loaded == [cache / jit.CACHE_BASE.name / "1_0_1_3_1_0_0/1_0_1_3_1_0_0.so"]


def test_incomplete_legacy_aggregate_still_uses_per_variant_jit(jit_package, monkeypatch):
    jit, _, cache, _ = jit_package
    manager = jit.FMHAVariantManager()
    jit._ALL_VARIANTS_SO.parent.mkdir(parents=True)
    jit._ALL_VARIANTS_SO.write_bytes(b"old partial aggregate")

    def load_module(path):
        if Path(path) == jit._ALL_VARIANTS_SO:
            return types.SimpleNamespace()
        return types.SimpleNamespace(**{"run_1_0_1_3_1_0_0": lambda value: ("compiled", value)})

    def compile_fixture(name, _params):
        path = cache / jit.CACHE_BASE.name / name / f"{name}.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", load_module)
    monkeypatch.setattr(manager, "_compile_only", compile_fixture)
    variant = manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert variant.run("input") == ("compiled", "input")


def test_packaged_gpu_compatibility_is_checked_for_each_device(jit_package, monkeypatch):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    assert jit._aot_module_path("variants") == artifact_dir / "all_variants.so"
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_unsupported")
    assert jit._aot_module_path("variants") is None


def test_variant_manager_does_not_reuse_packaged_kernel_on_incompatible_device(
    jit_package, monkeypatch
):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cuda_architectures"] = ["sm_103a"]
    manifest_path.write_text(json.dumps(manifest))
    manager = jit.FMHAVariantManager()

    def load_module(path):
        kind = "packaged" if Path(path).parent == artifact_dir else "jit"
        return types.SimpleNamespace(**{"run_1_0_1_3_1_0_0": lambda: kind})

    def compile_fixture(name, _params):
        path = jit.CACHE_BASE / name / f"{name}.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", load_module)
    monkeypatch.setattr(manager, "_compile_only", compile_fixture)
    args = (jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert manager.get_variant(*args).run() == "packaged"
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_100a")
    assert manager.get_variant(*args).run() == "jit"
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_103a")
    assert manager.get_variant(*args).run() == "packaged"


def test_helpers_do_not_reuse_packaged_modules_on_incompatible_device(jit_package, monkeypatch):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cuda_architectures"] = ["sm_103a"]
    manifest_path.write_text(json.dumps(manifest))

    def load_module(path):
        kind = "packaged" if Path(path).parent == artifact_dir else "jit"
        return types.SimpleNamespace(kind=kind)

    def compile_fixture(relative):
        path = jit.CACHE_BASE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", load_module)
    monkeypatch.setattr(jit, "_do_compile_plan", lambda: compile_fixture("plan/fmha_sm100_plan.so"))
    monkeypatch.setattr(
        jit, "_do_compile_sparse_topk", lambda: compile_fixture("sparse_topk/sparse_topk_select.so")
    )
    monkeypatch.setattr(
        jit, "_do_compile_reduction", lambda: compile_fixture("reduction/fmha_sm100_reduction.so")
    )
    helpers = (jit.get_plan_fn, jit.get_sparse_topk_module, jit.get_reduction_module)
    assert [fn().kind for fn in helpers] == ["packaged"] * 3
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_100a")
    assert [fn().kind for fn in helpers] == ["jit"] * 3
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_103a")
    assert [fn().kind for fn in helpers] == ["packaged"] * 3


def test_transient_device_query_failure_cannot_cache_a_packaged_kernel_as_jit(
    jit_package, monkeypatch
):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    manager = jit.FMHAVariantManager()
    calls = iter([RuntimeError("device not ready"), "sm_103a"])

    def device_arch():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    def load_module(path):
        kind = "packaged" if Path(path).parent == artifact_dir else "jit"
        return types.SimpleNamespace(**{"run_1_0_1_3_1_0_0": lambda: kind})

    def compile_fixture(name, _params):
        path = jit.CACHE_BASE / name / f"{name}.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(jit, "_aot_gpu_arch", device_arch)
    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", load_module)
    monkeypatch.setattr(manager, "_compile_only", compile_fixture)
    args = (jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)
    assert manager.get_variant(*args).run() == "jit"
    assert manager.get_variant(*args).run() == "packaged"
    monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_unsupported")
    assert manager.get_variant(*args).run() == "jit"


@pytest.mark.parametrize(
    "corruption", ["missing_manifest", "invalid_json", "unsafe_path", "duplicate_variant"]
)
def test_corrupt_packaged_manifest_is_rejected(jit_package, corruption):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    path = artifact_dir / "manifest.json"
    if corruption == "missing_manifest":
        path.unlink()
    elif corruption == "invalid_json":
        path.write_text("{")
    else:
        manifest = json.loads(path.read_text())
        if corruption == "unsafe_path":
            manifest["files"]["plan"] = "../outside.so"
        else:
            manifest["variants"][0] = manifest["variants"][1]
        path.write_text(json.dumps(manifest))
    assert jit._aot_module_path("variants") is None


@pytest.mark.parametrize("missing", ["variant", "helper"])
def test_incomplete_packaged_set_falls_back_to_jit(jit_package, monkeypatch, missing):
    jit, package, cache, loaded = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    if missing == "variant":
        manifest_path = artifact_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["variants"].remove("1_0_1_3_1_0_0")
        manifest_path.write_text(json.dumps(manifest))
    else:
        (artifact_dir / "plan.so").unlink()

    manager = jit.FMHAVariantManager()

    def compile_fixture(name, _params):
        path = cache / jit.CACHE_BASE.name / name / f"{name}.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"JIT fixture")

    monkeypatch.setattr(manager, "_compile_only", compile_fixture)
    variant = manager.get_variant(jit._FLOAT8_E4M3FN_CODE, 128, False, -1, 128, False, 1)

    assert variant.run("input") == ("compiled", "input")
    assert loaded == [cache / jit.CACHE_BASE.name / "1_0_1_3_1_0_0/1_0_1_3_1_0_0.so"]


@pytest.mark.parametrize("flag", ["GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK"])
def test_debug_modes_do_not_use_production_artifacts(jit_package, monkeypatch, flag):
    jit, package, _, _ = jit_package
    _stage_artifacts(jit, package)
    monkeypatch.setenv(flag, "1")
    assert jit._aot_module_path("variants") is None


@pytest.mark.parametrize("change", ["fmha", "cutlass", "ffi", "host", "gpu"])
def test_incompatible_packaged_set_is_rejected(jit_package, monkeypatch, change):
    jit, package, _, _ = jit_package
    artifact_dir = _stage_artifacts(jit, package)
    if change == "fmha":
        (package / "csrc/fmha_sm100_plan.h").write_text("changed\n")
    elif change == "cutlass":
        (package / "cutlass/include/cutlass/version.h").write_text("changed\n")
    elif change == "ffi":
        monkeypatch.setattr(jit, "_aot_ffi_version", lambda: "different")
    elif change == "host":
        manifest_path = artifact_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["host_arch"] = "incompatible"
        manifest_path.write_text(json.dumps(manifest))
    else:
        monkeypatch.setattr(jit, "_aot_gpu_arch", lambda: "sm_unsupported")

    assert jit._aot_module_path("variants") is None
