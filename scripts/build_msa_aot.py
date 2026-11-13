#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and validate the FMHA libraries shipped in supported wheels."""

import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path


def resolve_requested_architectures(requested, build_dir):
    recorded = Path(build_dir) / "trtllm_cuda_architectures.txt"
    if recorded.is_file():
        architectures = recorded.read_text().strip()
        if not architectures:
            raise ValueError(
                "CMake recorded an empty CUDA architecture list; reconfigure the build"
            )
        return architectures
    cache = Path(build_dir) / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(r"(?m)^CMAKE_CUDA_ARCHITECTURES:[A-Z_]+=(.*)$", cache.read_text())
        if match:
            requested = match.group(1).strip() or "all"
    if requested != "native":
        return requested
    raise ValueError("native CUDA architectures were not recorded; reconfigure the build")


def should_build(system, requested, skip_wheel):
    if system != "Linux" or skip_wheel:
        return False
    return requested == "all" or bool(select_cuda_architectures(requested, {"sm_100a", "sm_103a"}))


def select_cuda_architectures(requested, supported):
    """Select real SM images required by the wheel and available in nvcc."""
    supported = set(supported)
    if requested == "all":
        return [arch for arch in ("sm_100a", "sm_103a", "sm_100f") if arch in supported]
    wanted = []
    for item in requested.split(";"):
        match = re.fullmatch(r"(100|103)(?:a)?(?:-real)?", item.strip())
        if not match:
            continue
        arch = f"sm_{match.group(1)}a"
        if arch not in supported:
            raise ValueError(f"nvcc cannot build requested architecture {arch}")
        if arch not in wanted:
            wanted.append(arch)
    return wanted


def probe_cuda_architectures(nvcc, candidates):
    """Compile tiny kernels: nvcc's architecture list omits a/f suffixes."""
    supported = set()
    with tempfile.TemporaryDirectory(prefix="trtllm-msa-arch-probe-") as directory:
        source = Path(directory) / "probe.cu"
        source.write_text("__global__ void trtllm_msa_arch_probe() {}\n")
        for arch in candidates:
            command = [
                str(nvcc),
                f"-arch={arch}",
                "-c",
                str(source),
                "-o",
                str(Path(directory) / "probe.o"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                supported.add(arch)
            elif re.search(
                r"unsupported gpu architecture|not defined for option", result.stderr, re.IGNORECASE
            ):
                continue
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stdout, result.stderr
                )
    return supported


def validate_artifacts(files, variant_names):
    """Require every generated entry point, including auxiliary kernels."""
    required = {
        "variants": {f"__tvm_ffi_run_{name}" for name in variant_names},
        "plan": {"__tvm_ffi_plan"},
        "sparse_topk": {"__tvm_ffi_sparse_topk_select"},
        "reduction": {"__tvm_ffi_reduction"},
    }
    for kind, expected in required.items():
        path = Path(files[kind])
        if not path.is_file():
            raise FileNotFoundError(path)
        output = subprocess.run(
            ["nm", "-D", "--defined-only", str(path)], check=True, capture_output=True, text=True
        ).stdout
        exported = {line.split()[-1] for line in output.splitlines() if line.split()}
        missing = expected - exported
        if missing:
            raise ValueError(f"{kind} missing exports: {', '.join(sorted(missing))}")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _worker(package, warmup, architectures, jobs):
    # Avoid importing the package's GPU-facing API on a CPU-only build host.
    package_module = types.ModuleType("fmha_sm100")
    package_module.__path__ = [str(package)]
    sys.modules["fmha_sm100"] = package_module
    jit = _load_module("fmha_sm100.jit", package / "jit.py")
    package_module.jit = jit

    flags = jit._get_nvcc_flags

    def selected_flags(*args, **kwargs):
        original = flags(*args, **kwargs)
        original = re.sub(r"-gencode=arch=compute_\w+,code=sm_\w+", "", original)
        if "-I" in original:
            include_dirs = (
                jit._FMHA_VARLEN_DIR / "include",
                jit._CUTLASS_INCLUDE,
                jit._CUTLASS_UTIL_INCLUDE,
                jit._get_tvm_ffi_include(),
                args[0],
            )
            directories = sorted(map(str, include_dirs), key=len, reverse=True)
            pattern = "-I(?:" + "|".join(map(re.escape, directories)) + ")"
            original = re.sub(pattern, lambda match: shlex.quote(match.group()), original)
        targets = [f"-gencode=arch=compute_{arch[3:]},code={arch}" for arch in architectures]
        return " ".join([original, *targets]).replace("$", "$$")

    jit._get_nvcc_flags = selected_flags
    compile_script = _load_module("msa_warmup_for_wheel", warmup)
    # Upstream's cleanup traverses the shared /tmp; our TMPDIR is private.
    compile_script._cleanup_nvcc_temps = lambda: None
    sys.argv = [str(warmup), "--preset", "all", "--jobs", str(jobs)]
    compile_script.main()

    files = {
        "variants": jit.CACHE_BASE / "_all_variants/all_variants.so",
        "plan": jit.CACHE_BASE / "plan/fmha_sm100_plan.so",
        "sparse_topk": jit.CACHE_BASE / "sparse_topk/sparse_topk_select.so",
        "reduction": jit.CACHE_BASE / "reduction/fmha_sm100_reduction.so",
    }
    variants = jit._aot_variant_names()
    validate_artifacts(files, variants)
    digest = jit._aot_source_digest()
    destination = package / "_aot" / digest
    destination.mkdir(parents=True)
    packaged_files = {}
    for kind, path in files.items():
        packaged_files[kind] = path.name
        shutil.copy2(path, destination / path.name)
    manifest = {
        "format_version": 1,
        "source_sha256": digest,
        "tvm_ffi_version": jit._aot_ffi_version(),
        "host_arch": platform.machine(),
        "cuda_toolkit_version": os.environ["TRTLLM_MSA_BUILD_CUDA_VERSION"],
        "cuda_architectures": architectures,
        "variants": variants,
        "files": packaged_files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build(package, warmup, requested, nvcc, jobs):
    package, warmup, nvcc = Path(package).resolve(), Path(warmup).resolve(), Path(nvcc).resolve()
    candidates = select_cuda_architectures(requested, {"sm_100a", "sm_103a", "sm_100f"})
    supported = probe_cuda_architectures(nvcc, candidates)
    architectures = select_cuda_architectures(requested, supported)
    if not architectures:
        return
    version = subprocess.run(
        [str(nvcc), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if not version:
        raise ValueError("nvcc did not report a CUDA toolkit version")
    with tempfile.TemporaryDirectory(prefix="trtllm-msa-aot-") as directory:
        cache = Path(directory) / "cache"
        temporary = cache / "tmp"
        temporary.mkdir(parents=True)
        env = os.environ.copy()
        for name in ("GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK"):
            env.pop(name, None)
        env.update(
            MINFER_FMHA_CACHE_DIR=str(cache),
            TMPDIR=str(temporary),
            CUDA_HOME=str(nvcc.parent.parent),
            TRTLLM_MSA_BUILD_CUDA_VERSION=version,
        )
        stub_dir = nvcc.parent.parent / "lib64/stubs"
        if stub_dir.is_dir():
            env["LIBRARY_PATH"] = str(stub_dir) + os.pathsep + env.get("LIBRARY_PATH", "")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(package),
                str(warmup),
                ";".join(architectures),
                str(jobs),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path.cwd())
    parser.add_argument("package", type=Path)
    parser.add_argument("warmup", type=Path)
    parser.add_argument("architectures")
    parser.add_argument("jobs", type=int)
    args = parser.parse_args()
    if args.worker:
        _worker(args.package, args.warmup, args.architectures.split(";"), args.jobs)
    else:
        architectures = resolve_requested_architectures(args.architectures, args.build_dir)
        if not should_build(platform.system(), architectures, False):
            sys.exit(0)
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        nvcc = (
            os.environ.get("CMAKE_CUDA_COMPILER")
            or (str(Path(cuda_home) / "bin/nvcc") if cuda_home else None)
            or shutil.which("nvcc")
        )
        if not nvcc:
            parser.error("nvcc is required to build MSA artifacts")
        build(args.package, args.warmup, architectures, nvcc, args.jobs)
