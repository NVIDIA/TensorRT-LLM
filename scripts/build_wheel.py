#!/usr/bin/env python3
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

import os
import platform
import re
import shutil
import sys
import sysconfig
import tempfile
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy, copytree, rmtree
from subprocess import (DEVNULL, PIPE, CalledProcessError, Popen, check_output,
                        run)
from typing import Optional, Sequence

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except (ImportError, ModuleNotFoundError):
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

build_run = partial(run, shell=True, check=True)


def get_available_cpu_count() -> int:
    """Return the number of CPUs available to this process.

    Respects the process CPU affinity mask (Linux) so that builds launched
    inside a cgroup or taskset-constrained environment don't over-subscribe.
    Falls back to the total CPU count on platforms that don't expose affinity.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return cpu_count() or 1


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_project_dir():
    return Path(__file__).parent.resolve().parent


def apply_version_override(project_dir: Path,
                           version_override: Optional[str]) -> None:
    """Apply the requested package version before building the wheel."""
    if not version_override:
        return

    version_file = project_dir / "tensorrt_llm" / "version.py"
    version_content = version_file.read_text()
    version_match = re.search(r'(?m)^__version__ = "([^"]+)"$', version_content)
    current_version = version_match.group(1)

    resolved_version = version_override
    if version_override.startswith((".", "+")):
        resolved_version = current_version
        if not current_version.endswith(version_override):
            resolved_version += version_override
    resolved_version = str(Version(resolved_version))
    version_file.write_text(
        version_content.replace(f'__version__ = "{current_version}"',
                                f'__version__ = "{resolved_version}"', 1))


def get_source_dir():
    return get_project_dir() / "cpp"


def get_build_dir(build_dir, build_type, build_root=None, out_of_tree=False):
    if build_dir is None:
        dir_name = "build" if build_type == "Release" else f"build_{build_type}"
        if build_root is not None:
            # Isolate out-of-tree build state in its own CMake directory so
            # that toggling --out-of-tree against an existing conventional
            # build under the same --build_root always triggers a fresh
            # configure (first_build) instead of reusing a CMakeCache whose
            # redirected FMHA/version.h paths don't match the mode. Without
            # this, switching modes without --clean/--configure_cmake would
            # skip configure and silently write into the checkout despite
            # --out-of-tree.
            suffix = "-oot" if out_of_tree else ""
            build_dir = Path(build_root).resolve() / f"cpp-{dir_name}{suffix}"
        else:
            build_dir = get_source_dir() / dir_name
    else:
        build_dir = Path(build_dir).resolve()
    return build_dir


def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isdir(item_path) and not os.path.islink(item_path):
                rmtree(item_path)
            else:
                os.remove(item_path)
        except (OSError, IOError) as e:
            print(f"Failed to remove {item_path}: {e}", file=sys.stderr)


def sysconfig_scheme(override_vars=None):
    # Backported 'venv' scheme from Python 3.11+
    if os.name == 'nt':
        scheme = {
            'purelib': '{base}/Lib/site-packages',
            'scripts': '{base}/Scripts',
        }
    else:
        scheme = {
            'purelib': '{base}/lib/python{py_version_short}/site-packages',
            'scripts': '{base}/bin',
        }

    vars_ = sysconfig.get_config_vars()
    if override_vars:
        vars_.update(override_vars)
    return {key: value.format(**vars_) for key, value in scheme.items()}


def create_venv(venv_prefix: Path):
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    print(
        f"-- Using virtual environment at: {venv_prefix} (Python {py_major}.{py_minor})"
    )

    # Ensure compatible virtualenv version is installed (>=20.29.1, <22.0)
    print("-- Ensuring virtualenv version >=20.29.1,<22.0 is installed...")
    build_run(f'"{sys.executable}" -m pip install "virtualenv>=20.29.1,<22.0"')

    # Create venv if it doesn't exist
    if not venv_prefix.exists():
        print(f"-- Creating virtual environment in {venv_prefix}...")
        build_run(
            f'"{sys.executable}" -m virtualenv --system-site-packages "{venv_prefix}"'
        )
    else:
        print("-- Virtual environment already exists.")

    return venv_prefix


def setup_venv(project_dir: Path,
               requirements_file: Path,
               no_venv: bool,
               yes: bool = False,
               build_root: Optional[Path] = None) -> tuple[Path, Path]:
    """Creates/updates a venv and installs requirements.

    Args:
        project_dir: The root directory of the project.
        requirements_file: Path to the requirements file.
        no_venv: Use current Python environment as is.
        build_root: Directory for out-of-tree build state; when set, the venv
            is created there instead of inside the checkout.

    Returns:
        Tuple[Path, Path]: Paths to the python and conan executables in the venv.
    """
    if no_venv or sys.prefix != sys.base_prefix:
        reason = "Explicitly requested by user" if no_venv else "Already inside virtual environment"
        print(f"-- {reason}, using environment {sys.prefix} as is.")
        venv_prefix = Path(sys.prefix)
    else:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if build_root is not None:
            venv_prefix = build_root / f"venv-{py_version}"
        else:
            venv_prefix = project_dir / f".venv-{py_version}"
        venv_prefix = create_venv(venv_prefix)

    scheme = sysconfig_scheme({'base': venv_prefix})
    # Determine venv executable paths
    scripts_dir = Path(scheme["scripts"])
    venv_python = venv_prefix / sys.executable.removeprefix(sys.prefix)[1:]

    if os.environ.get("NVIDIA_PYTORCH_VERSION"):
        # Ensure PyPI PyTorch is not installed in the venv
        purelib_dir = Path(scheme["purelib"])
        pytorch_package_dir = purelib_dir / "torch"
        if str(venv_prefix) != sys.base_prefix and pytorch_package_dir.exists():
            warnings.warn(
                f"Using the NVIDIA PyTorch container with PyPI distributed PyTorch may lead to compatibility issues.\n"
                f"If you encounter any problems, please delete the environment at `{venv_prefix}` so that "
                f"`build_wheel.py` can recreate a virtual environment using container-provided PyTorch installation."
            )
            print("^^^^^^^^^^ IMPORTANT WARNING ^^^^^^^^^^", file=sys.stderr)
            if not yes:
                input("Press Ctrl+C to stop, any key to continue...\n")

        # Ensure inherited PyTorch version is compatible
        try:
            info = check_output(
                [str(venv_python), "-m", "pip", "show", "torch"])
        except CalledProcessError:
            raise RuntimeError(
                "NVIDIA PyTorch container detected, but cannot find PyTorch installation. "
                "The environment is corrupted. Please recreate your container.")
        version_installed = next(
            line.removeprefix("Version: ")
            for line in info.decode().splitlines()
            if line.startswith("Version: "))
        version_required = None
        try:
            with open(requirements_file) as fp:
                for line in fp:
                    if line.startswith("torch"):
                        version_required = Requirement(line)
                        break
        except FileNotFoundError:
            pass

        if version_required is not None:
            if version_installed not in version_required.specifier:
                raise RuntimeError(
                    f"Incompatible NVIDIA PyTorch container detected. "
                    f"The container provides PyTorch version {version_installed}, "
                    f"but current revision requires {version_required}. "
                    f"Please recreate your container using image specified in jenkins/current_image_tags.properties. "
                    f"NOTE: Please don't try install PyTorch using pip. "
                    f"Using the NVIDIA PyTorch container with PyPI distributed PyTorch may lead to compatibility issues."
                )

    # Install/update requirements
    print(
        f"-- Installing requirements from {requirements_file} into {venv_prefix}..."
    )
    build_run(f'"{venv_python}" -m pip install -r "{requirements_file}"')

    venv_conan = setup_conan(scripts_dir, venv_python)

    return venv_python, venv_conan


def setup_conan(scripts_dir, venv_python):
    build_run(f'"{venv_python}" -m pip install conan==2.14.0')
    # Determine the path to the conan executable within the venv
    venv_conan = scripts_dir / "conan"
    if not venv_conan.exists():
        # Attempt to find it using shutil.which as a fallback, in case it's already installed in the system
        try:
            result = build_run(
                f'''{venv_python} -c "import shutil; print(shutil.which('conan'))" ''',
                capture_output=True,
                text=True)
            conan_path_str = result.stdout.strip()

            if conan_path_str:
                venv_conan = Path(conan_path_str)
                print(
                    f"-- Found conan executable via PATH search at: {venv_conan}"
                )
            else:
                raise RuntimeError(
                    f"Failed to locate conan executable in virtual environment {scripts_dir} or system PATH."
                )

        except CalledProcessError as e:
            print(f"Fallback search command output: {e.stdout}",
                  file=sys.stderr)
            print(f"Fallback search command error: {e.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"Failed to locate conan executable in virtual environment {scripts_dir} or system PATH."
            )
    else:
        print(f"-- Found conan executable at: {venv_conan}")

    # Create default profile
    build_run(f'"{venv_conan}" profile detect -f')

    # Add the TensorRT LLM remote if it doesn't exist
    build_run(
        f'"{venv_conan}" remote add --force TensorRT-LLM https://edge.urm.nvidia.com/artifactory/api/conan/sw-tensorrt-llm-conan',
        stdout=DEVNULL,
        stderr=DEVNULL)

    return venv_conan


def _fmha_generation_stamp(fmha_v2_cu_dir: Path) -> Path:
    # Written as the last step of generate_fmha_cu; its absence means a
    # previous generation was interrupted and the directory contents cannot
    # be trusted (the bare directory exists from the moment generation
    # starts).
    return fmha_v2_cu_dir / ".generation_complete"


def get_fmha_gen_dirs(project_dir, gen_root=None):
    """Return (fmha_v2_cu_dir, cubin_dir) for generated FMHA sources.

    gen_root=None keeps the historical in-source locations; otherwise both
    live under gen_root (consumed by CMake via TRTLLM_FMHA_GEN_DIR).
    """
    base = (project_dir /
            "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention"
            if gen_root is None else gen_root)
    return base / "fmha_v2_cu", base / "cubin"


def generate_fmha_cu(project_dir, venv_python, gen_root=None):
    fmha_v2_cu_dir, cubin_dir = get_fmha_gen_dirs(project_dir, gen_root)
    fmha_v2_cu_dir.mkdir(parents=True, exist_ok=True)
    cubin_dir.mkdir(parents=True, exist_ok=True)
    _fmha_generation_stamp(fmha_v2_cu_dir).unlink(missing_ok=True)

    fmha_v2_dir = project_dir / "cpp/kernels/fmha_v2"
    if gen_root is not None:
        # The generator writes ./generated, ./temp and ./obj relative to its
        # own directory; run it from a scratch copy so a (possibly read-only)
        # checkout is never written.
        work_dir = gen_root / "fmha_v2-work"
        if work_dir.exists():
            rmtree(work_dir)
        copytree(fmha_v2_dir,
                 work_dir,
                 symlinks=False,
                 ignore=shutil.ignore_patterns("generated", "temp", "obj",
                                               "__pycache__"))
        fmha_v2_dir = work_dir

    env = os.environ.copy()
    env.update({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "ENABLE_SM89_QMMA": "1",
        "ENABLE_HMMA_FP32": "1",
        "GENERATE_CUBIN": "1",
        "SCHEDULING_MODE": "1",
        "ENABLE_SM100": "1",
        "ENABLE_SM120": "1",
        "GENERATE_CU_TRTLLM": "true"
    })

    shutil.rmtree(fmha_v2_dir / "generated", ignore_errors=True)
    shutil.rmtree(fmha_v2_dir / "temp", ignore_errors=True)
    shutil.rmtree(fmha_v2_dir / "obj", ignore_errors=True)
    build_run("python3 setup.py", env=env, cwd=fmha_v2_dir)

    # Only touches generated source files if content is updated
    def move_if_updated(src, dst):
        with open(src, "rb") as f:
            new_content = f.read()
        try:
            with open(dst, "rb") as f:
                old_content = f.read()
        except FileNotFoundError:
            old_content = None

        if old_content != new_content:
            shutil.move(src, dst)

    # Copy generated header file when cu path is active and cubins are deleted.
    move_if_updated(fmha_v2_dir / "generated/fmha_cubin.h",
                    cubin_dir / "fmha_cubin.h")

    # Copy generated source file (fmha_cubin.cpp) to the same directory as header
    cpp_src = fmha_v2_dir / "generated/fmha_cubin.cpp"
    if cpp_src.exists():
        move_if_updated(cpp_src, cubin_dir / "fmha_cubin.cpp")

    generated_files = set()
    for cu_file in (fmha_v2_dir / "generated").glob("*sm*.cu"):
        dst_file = fmha_v2_cu_dir / os.path.basename(cu_file)
        move_if_updated(cu_file, dst_file)
        generated_files.add(str(dst_file.resolve()))

    if not generated_files:
        raise RuntimeError(
            f"FMHA generation produced no *_sm*.cu files in {fmha_v2_cu_dir}; "
            "generation may have failed silently.")

    # Remove extra files
    for root, _, files in os.walk(fmha_v2_cu_dir):
        for file in files:
            file_path = os.path.realpath(os.path.join(root, file))
            if file_path not in generated_files:
                os.remove(file_path)

    _fmha_generation_stamp(fmha_v2_cu_dir).touch()


def create_cuda_stub_links(cuda_stub_dir: str, missing_libs: list[str]) -> str:
    """
    Creates symbolic links for CUDA stub libraries in a temporary directory.

    Args:
        cuda_stub_dir (str): Path to the directory containing CUDA stubs.
        missing_libs: Versioned names of the missing libraries.

    Returns:
        str: Path to the temporary directory where links were created.
    """
    cuda_stub_path = Path(cuda_stub_dir)
    if not cuda_stub_path.exists():
        raise RuntimeError(
            f"CUDA stub directory '{cuda_stub_dir}' does not exist.")

    # Create a temporary directory for the symbolic links
    temp_dir = tempfile.mkdtemp(prefix="cuda_stub_links_")
    temp_dir_path = Path(temp_dir)

    version_pattern = r'\.\d+'
    for missing_lib in filter(lambda x: re.search(version_pattern, x),
                              missing_libs):
        # Define `so` as the first part of `missing_lib` with trailing '.' and digits removed
        so = cuda_stub_path / re.sub(version_pattern, '', missing_lib)
        so_versioned = temp_dir_path / missing_lib

        # Check if the library exists in the original directory
        if so.exists():
            try:
                # Create the symbolic link in the temporary directory
                so_versioned.symlink_to(so)
            except OSError as e:
                # Clean up the temporary directory on error
                rmtree(temp_dir)
                raise RuntimeError(
                    f"Failed to create symbolic link for '{missing_lib}' in temporary directory '{temp_dir}': {e}"
                )
        else:
            warnings.warn(
                f"Warning: Source library '{so}' does not exist and was skipped."
            )

    # Return the path to the temporary directory where the links were created
    return str(temp_dir_path)


def check_missing_libs(lib_name: str) -> list[str]:
    result = build_run(f"ldd {lib_name}", capture_output=True, text=True)
    missing = []
    for line in result.stdout.splitlines():
        if "not found" in line:
            lib_name = line.split()[
                0]  # Extract the library name before "=> not found"
            if lib_name not in missing:
                missing.append(lib_name)
    return missing


def generate_python_stubs_linux(venv_python: Path, deep_ep: bool,
                                flash_mla: bool, transfer_agent_binding: bool,
                                binding_lib_name: str):
    build_run(f"\"{venv_python}\" -m pip install nanobind")
    build_run(f"\"{venv_python}\" -m pip install pybind11-stubgen")
    nanobind_stubgen_patterns = get_project_dir(
    ) / "scripts" / "nanobind_stubgen.patterns"

    env_stub_gen = os.environ.copy()
    cuda_home_dir = env_stub_gen.get("CUDA_HOME") or env_stub_gen.get(
        "CUDA_PATH") or "/usr/local/cuda"
    missing_libs = check_missing_libs(binding_lib_name)
    cuda_stub_dir = f"{cuda_home_dir}/lib64/stubs"

    if missing_libs and Path(cuda_stub_dir).exists():
        # Create symbolic links for the CUDA stubs
        link_dir = create_cuda_stub_links(cuda_stub_dir, missing_libs)
        ld_library_path = env_stub_gen.get("LD_LIBRARY_PATH")
        env_stub_gen["LD_LIBRARY_PATH"] = ":".join(
            filter(None, [link_dir, cuda_stub_dir, ld_library_path]))
    else:
        link_dir = None

    try:
        build_run(
            f"\"{venv_python}\" -m nanobind.stubgen -m bindings -r -O . "
            f"-p \"{nanobind_stubgen_patterns}\" -q",
            env=env_stub_gen)
        # Pre-import torch so deep_gemm_cpp_tllm's FP4 scalar-type registration
        # succeeds; CLI args after `-c ...` land in sys.argv[1:] for argparse.
        build_run(
            f"\"{venv_python}\" -c 'import torch; from pybind11_stubgen import main; main()' "
            "-o . deep_gemm_cpp_tllm --exit-code",
            env=env_stub_gen)
        if flash_mla:
            build_run(
                f"\"{venv_python}\" -m pybind11_stubgen -o . flash_mla_cpp_tllm --exit-code",
                env=env_stub_gen)
        if deep_ep:
            build_run(
                f"\"{venv_python}\" -m pybind11_stubgen -o . deep_ep_cpp_tllm --exit-code",
                env=env_stub_gen)
        if transfer_agent_binding:
            # Generate stubs for tensorrt_llm_transfer_agent_binding

            build_run(
                f"\"{venv_python}\" -m nanobind.stubgen -m tensorrt_llm_transfer_agent_binding -O .",
                env=env_stub_gen)

    finally:
        if link_dir:
            rmtree(link_dir)


def generate_python_stubs_windows(venv_python: Path, pkg_dir: Path,
                                  lib_dir: Path):

    print("Windows not supported for nanobind stubs")
    exit(1)


def build_kv_cache_manager_v2(project_dir,
                              venv_python,
                              use_mypyc=False,
                              build_root=None):
    print("-- Building kv_cache_manager_v2...")
    kv_cache_mgr_dir = project_dir / "tensorrt_llm/runtime/kv_cache_manager_v2"
    runtime_dir = project_dir / "tensorrt_llm/runtime"

    # The produced .so files always land in-place (they are final artifacts);
    # only the intermediate object files are redirected out of the checkout.
    build_temp_arg = ""
    if build_root is not None:
        build_temp_arg = f' --build-temp "{build_root / "kv_cache_manager_v2-temp"}"'

    # Clean up any existing mypyc artifacts in runtime directory to prevent stale inclusion
    # when switching from --mypyc to standard build
    if not use_mypyc:
        for so_file in runtime_dir.glob("*__mypyc*.so"):
            print(f"Removing stale mypyc artifact: {so_file}")
            so_file.unlink()

        # Also clean up any .so files inside kv_cache_manager_v2
        for so_file in kv_cache_mgr_dir.rglob("*.so"):
            print(f"Removing stale artifact: {so_file}")
            so_file.unlink()

    # Build rawref
    print("-- Building kv_cache_manager_v2 rawref extension...", end=" ")
    rawref_dir = kv_cache_mgr_dir / "rawref"
    build_run(f'"{venv_python}" setup.py build_ext --inplace{build_temp_arg}',
              cwd=rawref_dir)
    print("Done")

    if use_mypyc:
        # Build mypyc
        print("-- Building kv_cache_manager_v2 mypyc extensions...", end=" ")
        # setup_mypyc.py is in kv_cache_manager_v2 but executed from runtime dir
        setup_mypyc = kv_cache_mgr_dir / "setup_mypyc.py"
        build_run(
            f'"{venv_python}" "{setup_mypyc}" build_ext --inplace{build_temp_arg}',
            cwd=runtime_dir)

        # Verify that the shared library was generated
        if not list(runtime_dir.glob("*__mypyc*.so")):
            raise RuntimeError(
                "Failed to build kv_cache_manager_v2: no shared library generated."
            )
        print("Done")
    print("-- Done building kv_cache_manager_v2.")


def _tar_pipe_copy(src: Path, dst: Path) -> bool:
    """Populate dst from src as one streamed tar pipeline.

    A single reader/writer pair with kernel-buffered pipe I/O is much faster
    than per-file copies on network filesystems. Dereferences symlinks (-h)
    and preserves mtimes, matching copytree(symlinks=False) + copystat.
    Returns False if tar is unavailable or fails, so callers can fall back.
    """
    tar_bin = shutil.which("tar")
    if tar_bin is None:
        return False
    dst.mkdir(parents=True, exist_ok=True)
    # posix (pax) format keeps sub-second mtimes; the gnu default truncates
    # to whole seconds, which would defeat sync_tree's mtime comparison.
    #
    # Chain the producer and consumer tars directly through an OS pipe rather
    # than a shell string. Checking both return codes reports a failing
    # producer (e.g. an unreadable source file) that a shell pipeline would
    # mask behind the consumer's exit status, without depending on a bash that
    # supports `set -o pipefail`; it also avoids shell quoting entirely.
    producer = Popen(
        [tar_bin, "--format=posix", "-C",
         str(src), "-chf", "-", "."],
        stdout=PIPE)
    consumer = Popen([tar_bin, "-C", str(dst), "-xf", "-"],
                     stdin=producer.stdout)
    # Close our copy of the write end so the consumer sees EOF when the
    # producer exits (and the producer gets SIGPIPE if the consumer dies).
    producer.stdout.close()
    consumer.wait()
    producer.wait()
    return producer.returncode == 0 and consumer.returncode == 0


# How recently a source file must have been written for its mtime to be
# untrustworthy as a change marker. Inode timestamps come from a coarse clock
# (one timer tick on Linux) and some filesystems store whole seconds, so a file
# rewritten shortly after being copied can still report the mtime the copy
# recorded. A size+mtime comparison would then call it unchanged and leave a
# stale copy behind. Two seconds covers a whole-second-granularity destination
# (which can make an mtime look up to a second older than it is) on top of the
# tick granularity of the source.
_MTIME_RACE_WINDOW = 2.0


def _demote_racy_mtime(dst_file: Path, src_stat: Optional[os.stat_result],
                       now: float) -> None:
    """Break the mtime match for a copy whose source was just written.

    A copy normally records the source's mtime so the next sync can skip it.
    That is only sound once the source mtime has aged out of the window above;
    before that the source can change again without its mtime moving. Backdating
    the copy makes the next sync's comparison mismatch, so the file is re-copied
    instead of silently kept stale. It costs one extra copy of files written
    right before a sync, and converges: the re-copy records the real mtime.
    """
    if src_stat is None or src_stat.st_mtime <= now - _MTIME_RACE_WINDOW:
        return
    try:
        os.utime(dst_file,
                 (src_stat.st_atime, src_stat.st_mtime - _MTIME_RACE_WINDOW))
    except OSError:
        pass


def sync_tree(src: Path, dst: Path, exclude: Sequence[str] = ()) -> None:
    """Mirror the src directory into dst, touching only what changed.

    Replaces the rmtree+copytree pattern for artifact copy-back: files are
    rewritten only when size or mtime differs and entries missing from src
    are deleted, so incremental rebuilds cause almost no I/O on the
    destination (which may be a slow network filesystem). A missing dst is
    populated via a streamed tar pipeline instead of per-file copies.
    Symlinks are dereferenced like copytree(symlinks=False); mtimes are
    preserved so the next sync can compare against them, except for sources
    written within _MTIME_RACE_WINDOW of the copy, whose mtimes cannot yet
    prove the content settled. exclude lists fnmatch patterns for entry names
    to skip.
    """
    import fnmatch

    src = Path(src).resolve()
    dst = Path(dst)
    now = time.time()

    def excluded(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in exclude)

    def demote_racy_mtimes() -> None:
        # A cold populate (tar or copytree) copies source mtimes verbatim, so
        # apply the same guard the incremental path applies per file. Walk the
        # source rather than the freshly written destination: the source is
        # local and warm, and only the few racy entries need a write.
        for root, dirs, files in os.walk(src, followlinks=True):
            dirs[:] = [d for d in dirs if not excluded(d)]
            rel = Path(root).relative_to(src)
            for name in files:
                if excluded(name):
                    continue
                try:
                    src_stat = (Path(root) / name).stat()
                except OSError:
                    continue
                _demote_racy_mtime(dst / rel / name, src_stat, now)

    if dst.is_symlink():
        dst.unlink()
    elif dst.exists() and src == dst.resolve():
        return

    if not dst.exists():
        if not exclude and _tar_pipe_copy(src, dst):
            demote_racy_mtimes()
            return
        copytree(src,
                 dst,
                 symlinks=False,
                 ignore=shutil.ignore_patterns(*exclude) if exclude else None)
        demote_racy_mtimes()
        return

    for root, dirs, files in os.walk(src, followlinks=True):
        rel = Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if not excluded(d)]
        files = [f for f in files if not excluded(f)]
        dst_root = dst / rel
        if dst_root.exists() and not dst_root.is_dir():
            dst_root.unlink()
        dst_root.mkdir(exist_ok=True)
        keep = set(dirs) | set(files)
        for stale in os.listdir(dst_root):
            if stale not in keep:
                stale_path = dst_root / stale
                if stale_path.is_dir() and not stale_path.is_symlink():
                    rmtree(stale_path)
                else:
                    stale_path.unlink()
        for name in files:
            src_file = Path(root) / name
            dst_file = dst_root / name
            src_stat = None
            try:
                src_stat = src_file.stat()
                dst_stat = dst_file.stat()
                # Trust the match only once the source mtime has aged past the
                # race window; a just-written source can be rewritten again
                # without the mtime moving, which would strand a stale copy.
                if (src_stat.st_size == dst_stat.st_size
                        and abs(src_stat.st_mtime - dst_stat.st_mtime) < 1e-3
                        and src_stat.st_mtime <= now - _MTIME_RACE_WINDOW):
                    continue
            except OSError:
                pass
            if dst_file.is_dir() and not dst_file.is_symlink():
                rmtree(dst_file)
            # copy2: mtime must survive for the next sync's comparison.
            shutil.copy2(src_file, dst_file)
            _demote_racy_mtime(dst_file, src_stat, now)


def stage_python_package(project_dir: Path, staging_dir: Path) -> None:
    """Copy the sources setup.py packages into an out-of-tree staging project.

    Out-of-tree builds install compiled artifacts into this staging tree and
    build the wheel from it, so nothing is ever written into the checkout.
    """
    print(f"-- Staging python package sources into {staging_dir} ...")
    staging_dir.mkdir(parents=True, exist_ok=True)
    # examples: setup.py's root-level find_packages() ships the
    # examples.configs.database package from it.
    for tree in ("tensorrt_llm", "triton_kernels", "examples"):
        sync_tree(project_dir / tree,
                  staging_dir / tree,
                  exclude=("__pycache__", "*.pyc"))
    top_level_files = [
        "setup.py", "pyproject.toml", "requirements.txt",
        "requirements-dev.txt", "constraints.txt", "LICENSE", "README.md"
    ]
    top_level_files += [
        f.name for f in project_dir.glob("ATTRIBUTIONS-CPP-*.md")
    ]
    for name in top_level_files:
        src = project_dir / name
        if src.exists():
            copy(src, staging_dir / name)


# Console scripts `setup.py` declares. A Python-only setup cannot run the
# editable install that would create them, so it writes them itself.
_CONSOLE_SCRIPTS = {
    "trtllm-bench": "tensorrt_llm.commands.bench",
    "trtllm-serve": "tensorrt_llm.commands.serve",
    "trtllm-eval": "tensorrt_llm.commands.eval",
}


def setup_python_only(project_dir: Path, venv_python: Path):
    """Make a checkout importable and runnable without compiling C++.

    A checkout whose compiled artifacts are supplied from elsewhere -- a
    worktree sharing another build's `.so` files, a dev or CI container that
    already ships them, or a check that only touches Python -- still needs a
    venv it can import from, and there was no way to get just that.
    `--cpp_only` is the opposite of what is wanted, `--configure_only` still
    runs conan and the cmake configure, and a plain `pip install -e .[devel]`
    fails during the build because `fmha_sm100` is generated by the build and
    is not there yet:

        ImportError: The `fmha_sm100` package is missing.
                     Please execute scripts/build_wheel.py first

    So the requirements go in through the venv, the checkout is put on the
    venv's path with a `.pth`, and the console scripts `setup.py` declares are
    written by hand. Nothing is copied from any other checkout.
    """
    site_packages = check_output([
        str(venv_python), "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    ],
                                 text=True).strip()
    pth = Path(site_packages) / "tensorrt_llm_checkout.pth"
    pth.write_text(f"{project_dir}\n", encoding="utf-8")
    print(f"-- python-only: {pth} -> {project_dir}")

    bin_dir = Path(venv_python).parent
    for name, module in _CONSOLE_SCRIPTS.items():
        target = bin_dir / name
        target.write_text(
            f"#!{venv_python}\n"
            "import sys\n"
            f"from {module} import main\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(main())\n",
            encoding="utf-8")
        target.chmod(0o755)
        print(f"-- python-only: wrote {target}")

    # A plain script rather than an entry point, so it is symlinked from the
    # checkout it belongs to.
    launcher = project_dir / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"
    if launcher.is_file():
        installed = bin_dir / launcher.name
        if installed.exists() or installed.is_symlink():
            installed.unlink()
        installed.symlink_to(launcher)
        print(f"-- python-only: {installed} -> {os.readlink(installed)}")
    else:
        print(f"-- python-only: no {launcher}; skipping the launcher shim")


def main(*,
         build_type: str = "Release",
         generator: str = "",
         build_root: Optional[Path] = None,
         build_dir: Optional[Path] = None,
         dist_dir: Optional[Path] = None,
         cuda_architectures: Optional[str] = None,
         job_count: Optional[int] = None,
         extra_cmake_vars: Sequence[str] = tuple(),
         extra_make_targets: str = "",
         nccl_root: Optional[str] = None,
         nixl_root: Optional[str] = None,
         mooncake_root: Optional[str] = None,
         internal_cutlass_kernels_root: Optional[str] = None,
         clean: bool = False,
         clean_wheel: bool = False,
         configure_cmake: bool = False,
         configure_only: bool = False,
         use_ccache: bool = False,
         out_of_tree: bool = False,
         use_3rdparty_cache: bool = False,
         fast_build: bool = False,
         cpp_only: bool = False,
         python_only: bool = False,
         install: bool = False,
         skip_building_wheel: bool = False,
         linking_install_binary: bool = False,
         micro_benchmarks: bool = False,
         nvtx: bool = False,
         skip_stubs: bool = False,
         generate_fmha: bool = False,
         no_venv: bool = False,
         nvrtc_dynamic_linking: bool = False,
         mypyc: bool = False,
         require_dynamic_attributions: bool = False,
         plat_name: Optional[str] = None,
         yes: bool = False,
         version_override: Optional[str] = None):

    if clean:
        clean_wheel = True

    project_dir = get_project_dir()

    # Out-of-tree build state: everything metadata-heavy (venv, wheel
    # staging, ccache, intermediate objects) goes under build_root, keeping
    # the checkout free of high-churn I/O (important on network filesystems).
    # Resolve before chdir so a relative path stays anchored to the caller's
    # working directory.
    if build_root is None and os.environ.get("TRTLLM_BUILD_ROOT"):
        build_root = Path(os.environ["TRTLLM_BUILD_ROOT"])
    if build_root is not None:
        build_root = build_root.resolve()
        build_root.mkdir(parents=True, exist_ok=True)
        print(f"-- Out-of-tree build state under: {build_root}")
        # setup.py redirects the setuptools staging tree and *.egg-info
        # to this directory; an explicit env var set by the user wins.
        os.environ.setdefault("TRTLLM_WHEEL_STAGING_DIR",
                              str(build_root / "wheel-staging"))

    if out_of_tree:
        # Out-of-tree: never write into the checkout; the wheel is assembled in
        # an out-of-tree staging project. Validated by building with the
        # checkout mounted read-only.
        if build_root is None:
            raise RuntimeError("--out-of-tree requires --build_root")
        if platform.system() == "Windows":
            raise RuntimeError("--out-of-tree is not supported on Windows")
        if skip_building_wheel or linking_install_binary or install:
            raise RuntimeError(
                "--out-of-tree is incompatible with --skip_building_wheel, "
                "--linking_install_binary and --install: editable installs "
                "import compiled artifacts from the checkout, which a "
                "out-of-tree build never writes.")
        if version_override:
            raise RuntimeError(
                "--out-of-tree does not support --version-override (it would "
                "modify tensorrt_llm/version.py in the checkout)")

    apply_version_override(project_dir, version_override)
    os.chdir(project_dir)

    on_windows = platform.system() == "Windows"
    requirements_filename = "requirements-dev-windows.txt" if on_windows else "requirements-dev.txt"

    # Setup venv and install requirements
    venv_python, venv_conan = setup_venv(project_dir,
                                         project_dir / requirements_filename,
                                         no_venv,
                                         yes=yes,
                                         build_root=build_root)

    if python_only:
        setup_python_only(project_dir, venv_python)
        print("-- python-only setup complete: no C++ was built. Supply the "
              "compiled artifacts separately (they are not part of this mode).")
        return

    if cuda_architectures is not None:
        if "70-real" in cuda_architectures:
            raise RuntimeError("Volta architecture is deprecated support.")

    # Debug and RelWithDebInfo enable CUDA `--generate-line-info`, which
    # inflates .text so much that linking against every supported arch
    # overflows section limits. Require an explicit arch list so the build
    # won't silently fail deep into compilation.
    if build_type in ("Debug", "RelWithDebInfo") and not cuda_architectures:
        raise RuntimeError(
            f"Building {build_type} requires --cuda_architectures to be set "
            "explicitly (e.g. --cuda_architectures=90-real). Building for all "
            "architectures with line info enabled exceeds linker section "
            "limits. Pass a narrow arch list matching the GPU you intend to "
            "debug/profile.")

    if build_type == "Debug":
        print(
            "-- Debug build: only --generate-line-info is enabled by default. "
            "Full device debug info (-G) is NOT enabled because it can make "
            "ptxas memory usage explode and get OOM-killed on some kernels. "
            "If you need cuda-gdb stepping inside kernels, opt in with "
            "`--extra-cmake-vars CMAKE_CUDA_FLAGS_DEBUG=-G` and make sure "
            "the build host has enough memory for ptxas.")

    cuda_architectures = cuda_architectures or 'all'
    cmake_cuda_architectures = f'"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}"'

    cmake_def_args = []
    cmake_generator = ""

    if on_windows:
        # Windows does not support multi-device currently.
        extra_cmake_vars = list(extra_cmake_vars) + ["ENABLE_MULTI_DEVICE=0"]

        # The Ninja CMake generator is used for our Windows build
        # (Easier than MSBuild to make compatible with our Docker image)

    if generator:
        cmake_generator = "-G" + generator

    if job_count is None:
        job_count = get_available_cpu_count()

    if len(extra_cmake_vars):
        # Backwards compatibility, we also support semicolon expansion for each value.
        # However, it is best to use flag multiple-times due to issues with spaces in CLI.
        expanded_args = []
        for var in extra_cmake_vars:
            expanded_args += var.split(";")

        extra_cmake_vars = ["\"-D{}\"".format(var) for var in expanded_args]
        # Don't include duplicate conditions
        cmake_def_args.extend(set(extra_cmake_vars))

    if nccl_root is not None:
        cmake_def_args.append(f"-DNCCL_ROOT={nccl_root}")

    if nixl_root is not None:
        cmake_def_args.append(f"-DNIXL_ROOT={nixl_root}")

    if mooncake_root is not None:
        if on_windows:
            raise RuntimeError("Mooncake is not supported on Windows.")
        cmake_def_args.append(f"-DMOONCAKE_ROOT={mooncake_root}")

    build_dir = get_build_dir(build_dir, build_type, build_root, out_of_tree)
    first_build = not Path(build_dir, "CMakeFiles").exists()

    if clean and build_dir.exists():
        clear_folder(build_dir)  # Keep the folder in case it is mounted.
    build_dir.mkdir(parents=True, exist_ok=True)

    if use_ccache:
        if build_root is not None and "CCACHE_DIR" not in os.environ:
            # Default the cache next to the rest of the out-of-tree build
            # state. Point CCACHE_DIR at persistent storage instead to keep
            # compile results across ephemeral nodes/containers.
            ccache_dir = build_root / "ccache"
            ccache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["CCACHE_DIR"] = str(ccache_dir)
            print(f"-- ccache directory: {ccache_dir}")
        cmake_def_args.append(
            f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )

    if fast_build:
        cmake_def_args.append(f"-DFAST_BUILD=ON")

    # FetchContent bare-repo cache (see 3rdparty/CMakeLists.txt).  Forwarded
    # as -D vars so the configuration is reproducible from CMakeCache.txt
    # alone and doesn't depend on the caller's env.  The env var hand-off
    # lets a wrapping agent point at a shared cache without patching
    # build_wheel.py.
    if use_3rdparty_cache:
        cache_dir = os.environ.get("TRTLLM_FETCHCONTENT_CACHE") or str(
            project_dir / "3rdparty" / ".cache_3rdparty")
        cmake_def_args.append(f"-DTRTLLM_FETCHCONTENT_CACHE={cache_dir}")
        update_cmd = os.environ.get("TRTLLM_FETCHCONTENT_UPDATE_CMD", "")
        if update_cmd:
            cmake_def_args.append(
                f"-DTRTLLM_FETCHCONTENT_UPDATE_CMD={update_cmd}")

    if nvrtc_dynamic_linking:
        cmake_def_args.append(f"-DNVRTC_DYNAMIC_LINKING=ON")

    # BOLT compatibility: Force dynamic linking for NVIDIA libraries
    # Static NVIDIA libraries (libnvrtc_static.a, etc.) lack --emit-relocs,
    # which BOLT requires for proper binary optimization.
    bolt_enabled = any("ENABLE_BOLT_COMPATIBLE=ON" in var
                       for var in extra_cmake_vars)
    if bolt_enabled:
        if not nvrtc_dynamic_linking:
            cmake_def_args.append("-DNVRTC_DYNAMIC_LINKING=ON")
            print(
                "-- BOLT: Forcing NVRTC_DYNAMIC_LINKING=ON (static NVIDIA libs lack relocations)"
            )

    targets = ["tensorrt_llm"]

    if cpp_only:
        build_pyt = "OFF"
        build_deep_ep = "OFF"
        build_deep_gemm = "OFF"
        build_flash_mla = "OFF"
    else:
        targets.extend([
            "th_common", "bindings", "deep_ep", "deep_gemm", "pg_utils",
            "flash_mla"
        ])
        build_pyt = "ON"
        build_deep_ep = "ON"
        build_deep_gemm = "ON"
        build_flash_mla = "ON"

    if micro_benchmarks:
        targets.append("micro_benchmarks")
        build_micro_benchmarks = "ON"
    else:
        build_micro_benchmarks = "OFF"

    disable_nvtx = "OFF" if nvtx else "ON"

    source_dir = get_source_dir()

    fmha_gen_root = build_root / "fmha-gen" if out_of_tree else None
    fmha_v2_cu_dir, _ = get_fmha_gen_dirs(project_dir, fmha_gen_root)
    if (clean or generate_fmha
            or not _fmha_generation_stamp(fmha_v2_cu_dir).exists()):
        generate_fmha_cu(project_dir, venv_python, fmha_gen_root)

    if out_of_tree:
        cmake_def_args.append(f"-DTRTLLM_FMHA_GEN_DIR={fmha_gen_root}")
        # Write the configured executor/version.h into the build tree
        # instead of cpp/include (the checkout may be read-only).
        cmake_def_args.append(
            f"-DTRTLLM_VERSION_H_INCLUDE_DIR={build_dir}/generated-include")

    with working_directory(build_dir):
        if clean or first_build or configure_cmake or configure_only:
            # Conan writes a CMakeUserPresets.json convenience file next to
            # cpp/CMakeLists.txt; with out-of-tree build state it would be
            # the only build file left in the checkout (and would point at a
            # possibly ephemeral location), so skip generating it.
            conan_extra_args = (
                " -c tools.cmake.cmaketoolchain:user_presets=False"
                if build_root is not None else "")
            build_run(
                f"\"{venv_conan}\" install --build=missing --no-remote --output-folder={build_dir}/conan -s 'build_type={build_type}'{conan_extra_args} {source_dir}"
            )
            cmake_def_args.append(
                f"-DCMAKE_TOOLCHAIN_FILE={build_dir}/conan/conan_toolchain.cmake"
            )
            if internal_cutlass_kernels_root:
                cmake_def_args.append(
                    f"-DINTERNAL_CUTLASS_KERNELS_PATH={internal_cutlass_kernels_root}"
                )
            cmake_def_args = " ".join(cmake_def_args)
            cmake_configure_command = (
                f'cmake -DCMAKE_BUILD_TYPE="{build_type}" -DBUILD_PYT="{build_pyt}" -DBUILD_DEEP_EP="{build_deep_ep}" -DBUILD_DEEP_GEMM="{build_deep_gemm}" -DBUILD_FLASH_MLA="{build_flash_mla}"'
                f' -DNVTX_DISABLE="{disable_nvtx}" -DBUILD_MICRO_BENCHMARKS={build_micro_benchmarks}'
                f' -DBUILD_WHEEL_TARGETS="{";".join(targets)}"'
                f' -DPython_EXECUTABLE={venv_python} -DPython3_EXECUTABLE={venv_python}'
                f' {cmake_cuda_architectures} {cmake_def_args} {cmake_generator} -S "{source_dir}"'
            )
            print("CMake Configure command: ")
            print(cmake_configure_command)
            build_run(cmake_configure_command)

        if configure_only:
            return

        maybe_keep_depfile = " -- -d keepdepfile" if generator == "Ninja" else ""
        cmake_build_command = (
            f'cmake --build . --config {build_type} --parallel {job_count} '
            f'--target build_wheel_targets {" ".join(extra_make_targets)}{maybe_keep_depfile}'
        )
        print("CMake Build command: ")
        print(cmake_build_command)
        build_run(cmake_build_command)

    if cpp_only:
        assert not install, "Installing is not supported for cpp_only builds"
        return

    if out_of_tree:
        # Assemble the wheel in an out-of-tree staging project; the checkout
        # is only read from this point on.
        wheel_project_dir = build_root / "package"
        stage_python_package(project_dir, wheel_project_dir)
    else:
        wheel_project_dir = project_dir

    pkg_dir = wheel_project_dir / "tensorrt_llm"
    assert pkg_dir.is_dir(), f"{pkg_dir} is not a directory"
    lib_dir = pkg_dir / "libs"
    include_dir = pkg_dir / "include"
    if lib_dir.exists():
        clear_folder(lib_dir)
    # include_dir is not cleared: its subtrees are synced with deletion of
    # extraneous entries (sync_tree) or guarded by generation stamps, so
    # incremental rebuilds skip the ~10k-file rewrite of the include tree.
    # Remove auto-generated attributions file from previous builds
    auto_attr_file = wheel_project_dir / "ATTRIBUTIONS.md"
    if auto_attr_file.exists():
        os.remove(auto_attr_file)

    cache_dir = os.getenv("TRTLLM_DG_CACHE_DIR")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif on_windows:
        if os.getenv("APPDATA") is not None:
            cache_dir = Path(os.getenv("APPDATA")) / "tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP")) / "tensorrt_llm"
    else:
        if os.getenv("HOME") is not None:
            cache_dir = Path(os.getenv("HOME")) / ".tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP"), "/tmp") / "tensorrt_llm"
    if cache_dir.exists():
        clear_folder(cache_dir)

    def safe_copy(src, dst):
        """Copy a file, replacing a destination symlink with a real file."""
        src_path = Path(src)
        dst_path = Path(dst)
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name
        if dst_path.is_symlink():
            dst_path.unlink()
        return copy(src_path, dst_path)

    install_file = safe_copy

    install_tree = sync_tree
    if skip_building_wheel and linking_install_binary:

        def symlink_remove_dst(src, dst):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if os.path.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_file = symlink_remove_dst

        def symlink_remove_dst_tree(src, dst):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if os.path.isdir(dst) and not os.path.islink(dst):
                rmtree(dst)  # left behind by a previous copy-mode build
            elif os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_tree = symlink_remove_dst_tree

    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    install_tree(get_source_dir() / "include" / "tensorrt_llm" / "deep_gemm",
                 include_dir / "deep_gemm")

    # Copy FMHA kernel generation headers for JIT compilation
    fmha_build_dir = build_dir / "tensorrt_llm" / "kernels" / "trtllmGenKernels" / "fmha"
    fmha_include_dir = include_dir / "trtllm_gen_kernels" / "fmha"
    if fmha_build_dir.exists():
        fmha_include_dir.mkdir(parents=True, exist_ok=True)

        # Helper function to resolve symlinks and copy actual content
        def copy_resolving_symlink(src_path, dst_path):
            """Copy file or directory, resolving symlinks to copy actual content.

            Skips the copy when dst already exists and a stamp file confirms the
            previous copy completed successfully, and the stamp is at least as new
            as src.  Using a stamp (rather than dst mtime) avoids two pitfalls:
            - Directory mtime on Linux only reflects direct-child changes, not
              deeply nested ones, so a modified source file may not update it.
            - An interrupted copy leaves dst with a fresh mtime that would
              incorrectly satisfy an mtime check, causing the next build to skip.
            """
            if src_path.is_symlink():
                resolved_src = src_path.resolve()
            else:
                resolved_src = src_path

            if resolved_src.is_dir():
                stamp = dst_path.parent / f".{dst_path.name}.stamp"
                if dst_path.exists() and stamp.exists():
                    if stamp.stat().st_mtime >= resolved_src.stat().st_mtime:
                        return  # destination is up to date
                    rmtree(dst_path)
                elif dst_path.exists():
                    rmtree(
                        dst_path)  # dst exists but no stamp — interrupted copy
                stamp.unlink(missing_ok=True)
                # Shell out to cp -rL: dereferences symlinks like copytree(symlinks=False)
                # but uses kernel-level copy primitives, which is significantly faster
                # than Python's file-by-file copytree on network-mounted filesystems.
                # Fall back to copytree if cp is unavailable (non-Linux or minimal containers).
                cp_bin = shutil.which("cp")
                if cp_bin is not None:
                    try:
                        run([cp_bin, "-rL",
                             str(resolved_src),
                             str(dst_path)],
                            check=True)
                    except FileNotFoundError:
                        copytree(resolved_src, dst_path, symlinks=False)
                else:
                    copytree(resolved_src, dst_path, symlinks=False)
                stamp.touch()
            else:
                if dst_path.is_dir():
                    dst_path = dst_path / src_path.name
                copy(resolved_src, dst_path)

        # Copy cuda_ptx directory (actual directory, not symlink)
        cuda_ptx_src = fmha_build_dir / "cuda_ptx"
        if cuda_ptx_src.exists():
            copy_resolving_symlink(cuda_ptx_src, fmha_include_dir / "cuda_ptx")

        # Copy cutlass (symlink, need to resolve)
        cutlass_src = fmha_build_dir / "cutlass"
        if cutlass_src.exists():
            copy_resolving_symlink(cutlass_src, fmha_include_dir / "cutlass")

        # Copy trtllm directory (contains dev symlink)
        trtllm_src = fmha_build_dir / "trtllm"
        if trtllm_src.exists():
            copy_resolving_symlink(trtllm_src, fmha_include_dir / "trtllm")

        # Copy cuda (symlink, need to resolve)
        cuda_src = fmha_build_dir / "cuda"
        if cuda_src.exists():
            copy_resolving_symlink(cuda_src, fmha_include_dir / "cuda")

        # Copy KernelParams.h (actual file)
        kernel_params_src = fmha_build_dir / "KernelParams.h"
        if kernel_params_src.exists():
            copy(kernel_params_src, fmha_include_dir / "KernelParams.h")

        # Copy KernelParamsDecl.h (actual file)
        kernel_params_decl_src = fmha_build_dir / "KernelParamsDecl.h"
        if kernel_params_decl_src.exists():
            copy(kernel_params_decl_src,
                 fmha_include_dir / "KernelParamsDecl.h")

    required_cuda_headers = [
        "cuda_fp16.h", "cuda_fp16.hpp", "cuda_bf16.h", "cuda_bf16.hpp",
        "cuda_fp8.h", "cuda_fp8.hpp"
    ]
    if os.getenv("CUDA_HOME") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_HOME")) / "include"
    elif os.getenv("CUDA_PATH") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_PATH")) / "include"
    elif not on_windows:
        cuda_include_dir = Path("/usr/local/cuda/include")
    else:
        cuda_include_dir = None

    if cuda_include_dir is None or not cuda_include_dir.exists():
        print(
            "CUDA_HOME or CUDA_PATH should be set to enable DeepGEMM JIT compilation"
        )
    else:
        cuda_include_target_dir = include_dir / "cuda" / "include"
        cuda_include_target_dir.mkdir(parents=True, exist_ok=True)
        for header in required_cuda_headers:
            install_file(cuda_include_dir / header, include_dir / header)

    if on_windows:
        install_file(build_dir / "tensorrt_llm/tensorrt_llm.dll",
                     lib_dir / "tensorrt_llm.dll")
        install_file(build_dir / f"tensorrt_llm/thop/th_common.dll",
                     lib_dir / "th_common.dll")
    else:
        install_file(build_dir / "tensorrt_llm/libtensorrt_llm.so",
                     lib_dir / "libtensorrt_llm.so")
        install_file(build_dir / "tensorrt_llm/thop/libth_common.so",
                     lib_dir / "libth_common.so")
        if os.path.exists(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so"
        ):
            install_file(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so",
                lib_dir / "libtensorrt_llm_ucx_wrapper.so")
            build_run(
                f'patchelf --set-rpath \'$ORIGIN/ucx/\' {lib_dir / "libtensorrt_llm_ucx_wrapper.so"}'
            )
            if os.path.exists("/usr/local/ucx"):
                ucx_dir = lib_dir / "ucx"
                if ucx_dir.exists():
                    clear_folder(ucx_dir)
                install_tree("/usr/local/ucx/lib", ucx_dir)
                build_run(
                    f"find {ucx_dir} -type f -name '*.so*' -exec patchelf --set-rpath \'$ORIGIN:$ORIGIN/ucx:$ORIGIN/../\' {{}} \\;"
                )
        # NIXL wrapper and libraries
        nixl_utils_dir = build_dir / "tensorrt_llm/executor/cache_transmission/nixl_utils"
        if os.path.exists(nixl_utils_dir / "libtensorrt_llm_nixl_wrapper.so"):
            install_file(nixl_utils_dir / "libtensorrt_llm_nixl_wrapper.so",
                         lib_dir / "libtensorrt_llm_nixl_wrapper.so")
            build_run(
                f'patchelf --set-rpath \'$ORIGIN/nixl/\' {lib_dir / "libtensorrt_llm_nixl_wrapper.so"}'
            )
            # Copy NIXL libraries
            if os.path.exists("/opt/nvidia/nvda_nixl"):
                nixl_dir = lib_dir / "nixl"
                if nixl_dir.exists():
                    clear_folder(nixl_dir)
                nixl_lib_path = "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
                if not os.path.exists(nixl_lib_path):
                    nixl_lib_path = "/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu"
                if not os.path.exists(nixl_lib_path):
                    nixl_lib_path = "/opt/nvidia/nvda_nixl/lib64"
                install_tree(nixl_lib_path, nixl_dir)
                build_run(
                    f"find {nixl_dir} -type f -name '*.so*' -exec patchelf --set-rpath \'$ORIGIN:$ORIGIN/plugins:$ORIGIN/../:$ORIGIN/../ucx/:$ORIGIN/../../ucx/\' {{}} \\;"
                )
        # Install tensorrt_llm_transfer_agent_binding Python module (standalone agent bindings)
        # This is built when either NIXL or Mooncake is enabled
        # Install to tensorrt_llm/ (same level as bindings.so)
        agent_binding_so = list(
            nixl_utils_dir.glob("tensorrt_llm_transfer_agent_binding*.so"))
        if agent_binding_so:
            install_file(agent_binding_so[0],
                         pkg_dir / agent_binding_so[0].name)
        if os.path.exists(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/mooncake_utils/libtensorrt_llm_mooncake_wrapper.so"
        ):
            install_file(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/mooncake_utils/libtensorrt_llm_mooncake_wrapper.so",
                lib_dir / "libtensorrt_llm_mooncake_wrapper.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so",
            lib_dir / "libdecoder_attention_0.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so",
            lib_dir / "libdecoder_attention_1.so")
        install_file(build_dir / "tensorrt_llm/runtime/utils/libpg_utils.so",
                     lib_dir / "libpg_utils.so")

    # deep_ep/deep_gemm are synced in place below (sync_tree removes stale
    # entries); deep_ep is deleted explicitly when this build does not
    # produce it.
    deep_ep_dir = pkg_dir / "deep_ep"
    deep_gemm_dir = pkg_dir / "deep_gemm"

    scripts_dir = pkg_dir / "scripts"
    if scripts_dir.exists():
        clear_folder(scripts_dir)

    if not cpp_only:

        def get_binding_lib(subdirectory, name):
            binding_build_dir = (build_dir / "tensorrt_llm" / subdirectory)
            if on_windows:
                binding_lib = list(binding_build_dir.glob(f"{name}.*.pyd"))
            else:
                binding_lib = list(binding_build_dir.glob(f"{name}.*.so"))

            assert len(
                binding_lib
            ) == 1, f"Exactly one binding library should be present: {binding_lib}"
            return binding_lib[0]

        binding_lib_dir = get_binding_lib("nanobind", "bindings")
        binding_lib_file_name = binding_lib_dir.name
        install_file(binding_lib_dir, pkg_dir)

        with (build_dir / "tensorrt_llm" / "deep_ep" /
              "cuda_architectures.txt").open() as f:
            deep_ep_cuda_architectures = f.read().strip().strip(";")
        if not deep_ep_cuda_architectures and deep_ep_dir.exists():
            if deep_ep_dir.is_symlink():
                deep_ep_dir.unlink()
            else:
                rmtree(deep_ep_dir)
        if deep_ep_cuda_architectures:
            install_file(get_binding_lib("deep_ep", "deep_ep_cpp_tllm"),
                         pkg_dir)
            install_tree(
                build_dir / "tensorrt_llm" / "deep_ep" / "python" / "deep_ep",
                deep_ep_dir)
            (lib_dir / "nvshmem").mkdir(exist_ok=True)
            install_file(
                build_dir / "tensorrt_llm/deep_ep/nvshmem-build/License.txt",
                lib_dir / "nvshmem")
            install_file(
                build_dir /
                "tensorrt_llm/deep_ep/nvshmem-build/src/lib/nvshmem_bootstrap_uid.so.3",
                lib_dir / "nvshmem")
            install_file(
                build_dir /
                "tensorrt_llm/deep_ep/nvshmem-build/src/lib/nvshmem_transport_ibgda.so.103",
                lib_dir / "nvshmem")

        install_file(get_binding_lib("deep_gemm", "deep_gemm_cpp_tllm"),
                     pkg_dir)
        install_tree(
            build_dir / "tensorrt_llm" / "deep_gemm" / "python" / "deep_gemm",
            deep_gemm_dir)

        with (build_dir / "tensorrt_llm" / "flash_mla" /
              "cuda_architectures.txt").open() as f:
            flash_mla_cuda_architectures = f.read().strip().strip(";")
        if flash_mla_cuda_architectures:
            install_file(get_binding_lib("flash_mla", "flash_mla_cpp_tllm"),
                         pkg_dir)
            install_tree(
                build_dir / "tensorrt_llm" / "flash_mla" / "python" /
                "flash_mla", pkg_dir / "flash_mla")

        # Stage the FetchContent-patched MSA package for setup.py packaging.
        msa_src = build_dir / "_deps" / "msa-src" / "python" / "fmha_sm100"
        cutlass_src = build_dir / "_deps" / "cutlass-src"
        msa_dst = wheel_project_dir / "3rdparty" / "fmha_sm100"
        if not (msa_src / "cute" / "interface.py").is_file():
            raise FileNotFoundError(
                f"MSA package missing at {msa_src}; CMake FetchContent for msa "
                "did not populate the expected sources.")
        if msa_dst.is_symlink():
            msa_dst.unlink()
        elif msa_dst.exists():
            rmtree(msa_dst)
        msa_dst.mkdir(parents=True)
        for python_source in msa_src.glob("*.py"):
            install_file(python_source, msa_dst)
        for source_dir, relative_dir in (
            (msa_src / "csrc", Path("csrc")),
            (msa_src / "cute", Path("cute")),
            (cutlass_src / "include", Path("cutlass/include")),
            (cutlass_src / "tools/util/include",
             Path("cutlass/tools/util/include")),
        ):
            (msa_dst / relative_dir).parent.mkdir(parents=True, exist_ok=True)
            install_tree(
                source_dir,
                msa_dst / relative_dir,
            )
        install_file(cutlass_src / "LICENSE.txt", msa_dst / "cutlass")

        if not skip_stubs:
            with working_directory(pkg_dir):
                if on_windows:
                    generate_python_stubs_windows(venv_python, pkg_dir, lib_dir)
                else:  # on linux
                    generate_python_stubs_linux(
                        venv_python, bool(deep_ep_cuda_architectures),
                        bool(flash_mla_cuda_architectures),
                        nixl_root is not None or mooncake_root is not None,
                        binding_lib_file_name)

    build_kv_cache_manager_v2(wheel_project_dir,
                              venv_python,
                              use_mypyc=mypyc,
                              build_root=build_root)

    if not skip_building_wheel:
        if dist_dir is None:
            dist_dir = build_root / "dist" if out_of_tree else project_dir / "build"
        else:
            dist_dir = Path(dist_dir)

        if not dist_dir.exists():
            dist_dir.mkdir(parents=True)

        if clean_wheel:
            # For incremental build, the python build module adds
            # the new files but does not remove the deleted files.
            #
            # This breaks the Windows CI/CD pipeline when building
            # and validating python changes in the whl.
            clear_folder(dist_dir)
            # Without --build_root the setuptools staging tree (build_base)
            # lives at project_dir/build == dist_dir, so the clear above
            # already wipes it. With --build_root it moves under
            # TRTLLM_WHEEL_STAGING_DIR, so clearing dist_dir alone would
            # leave stale copies of deleted package files there to be
            # re-packed into the next "clean" wheel. Clear it too.
            staging_dir = os.environ.get("TRTLLM_WHEEL_STAGING_DIR")
            if staging_dir:
                staging_build = Path(staging_dir) / "build"
                if staging_build.exists():
                    clear_folder(staging_build)

        extra_wheel_build_args = os.getenv("EXTRA_WHEEL_BUILD_ARGS", "")
        plat_name_arg = ""
        if plat_name:
            plat_name_arg = f'--config-setting="--build-option=--plat-name={plat_name}"'
            extra_wheel_build_args = " ".join(
                arg for arg in (extra_wheel_build_args, plat_name_arg) if arg)

        # Attempt to generate attributions using the dependency database
        # Skip if output already exists and the build system hasn't changed
        auto_attr = build_dir / "attribution" / "ATTRIBUTIONS.md"
        if auto_attr.exists() and not (clean or first_build or configure_cmake):
            print(f"Using cached attributions from {auto_attr}")
        else:
            try:
                # Activate venv so that 'trtllm-sbom' CLI can be found after pip installs it
                venv_bin = venv_python.parent
                build_run(
                    f'. "{venv_bin / "activate"}" && python {project_dir}/scripts/attribute.py --build-dir "{build_dir}" -j {job_count}'
                )
            except Exception as e:
                if require_dynamic_attributions:
                    raise RuntimeError(
                        f"Attribution generation failed and --require_dynamic_attributions was set: {e}"
                    ) from e
                print(
                    f"Warning: Attribution generation step failed with error: {e}",
                    file=sys.stderr)
                print(
                    "You can run the dependency scanner manually and then use 'trtllm-sbom generate' as described in scripts/attribution/sbom/README.md.",
                    file=sys.stderr)

        # Copy auto-generated ATTRIBUTIONS.md to project root for wheel packaging
        if auto_attr.exists():
            install_file(auto_attr, wheel_project_dir / "ATTRIBUTIONS.md")
            print(
                f"Copied auto-generated attributions to {wheel_project_dir / 'ATTRIBUTIONS.md'}"
            )

        env = os.environ.copy()
        if mypyc:
            env["TRTLLM_ENABLE_MYPYC"] = "1"
        else:
            env["TRTLLM_ENABLE_MYPYC"] = "0"

        build_run(
            f'\"{venv_python}\" -m build {wheel_project_dir} --skip-dependency-check {extra_wheel_build_args} --no-isolation --wheel --outdir "{dist_dir}"',
            env=env)

    if install:
        # The venv this build just created, not the interpreter that started
        # the build. On a fresh checkout `build_wheel.py` has to be launched
        # with the system python, so `sys.executable` here installed the
        # package into the system site-packages and left the venv untouched.
        build_run(f"\"{venv_python}\" -m pip install -e .[devel]")


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--build_type",
        "-b",
        default="Release",
        choices=["Release", "RelWithDebInfo", "Debug"],
        help="Build type, will be passed to cmake `CMAKE_BUILD_TYPE` variable")
    parser.add_argument(
        "--generator",
        "-G",
        default="",
        help="CMake generator to use (e.g., 'Ninja', 'Unix Makefiles')")
    parser.add_argument(
        "--cuda_architectures",
        "-a",
        help=
        "CUDA architectures to build for, will be passed to cmake `CUDA_ARCHITECTURES` variable. Example: `--cuda_architectures=90-real;100-real`"
    )
    parser.add_argument("--install",
                        "-i",
                        action="store_true",
                        help="Install the built python package after building")
    parser.add_argument("--clean",
                        "-c",
                        action="store_true",
                        help="Clean the build directory before building")
    parser.add_argument(
        "--clean_wheel",
        action="store_true",
        help=
        "Clear dist_dir folder when creating wheel. Will be set to `true` if `--clean` is set"
    )
    parser.add_argument("--configure_cmake",
                        action="store_true",
                        help="Always configure cmake before building")
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Run cmake configure and exit, skipping build and wheel packaging")
    parser.add_argument("--use_ccache",
                        default=False,
                        action="store_true",
                        help="Use ccache compiler driver for faster rebuilds")
    parser.add_argument(
        "--use-3rdparty-cache",
        default=False,
        action="store_true",
        help="Accelerate FetchContent git clones via bare reference "
        "repos under $TRTLLM_FETCHCONTENT_CACHE "
        "(default: <project>/3rdparty/.cache_3rdparty).")
    parser.add_argument(
        "--fast_build",
        "-f",
        default=False,
        action="store_true",
        help=
        "Skip compiling some kernels to accelerate compilation -- for development only"
    )
    parser.add_argument(
        "--job_count",
        "-j",
        const=get_available_cpu_count(),
        nargs="?",
        help=
        "Number of parallel jobs for compilation (default: number of CPUs available to this process, respecting affinity)"
    )
    parser.add_argument(
        "--cpp_only",
        "-l",
        action="store_true",
        help="Only build the C++ library without Python dependencies")
    parser.add_argument(
        "--python_only",
        action="store_true",
        help="The opposite of --cpp_only: build the venv, install the "
        "requirements, put this checkout on the venv's path, write the "
        "console scripts, and build no C++ at all. For a checkout whose "
        "compiled artifacts come from elsewhere -- a worktree sharing another "
        "build's .so files, or a dev/CI container that already ships them. "
        "Supply the artifacts separately; this mode does not.")
    parser.add_argument(
        "--extra-cmake-vars",
        "-D",
        action="append",
        help=
        "Extra cmake variable definitions which can be specified multiple times. Example: -D \"key1=value1\" -D \"key2=value2\"",
        default=[])
    parser.add_argument(
        "--extra-make-targets",
        help="Additional make targets to build. Example: \"target_1 target_2\"",
        nargs="+",
        default=[])
    parser.add_argument("--nccl_root",
                        help="Directory containing NCCL headers and libraries")
    parser.add_argument("--nixl_root",
                        help="Directory containing NIXL headers and libraries")
    parser.add_argument(
        "--mooncake_root",
        help=
        "Directory containing Mooncake transfer engine headers and libraries")
    parser.add_argument(
        "--internal-cutlass-kernels-root",
        default="",
        help=
        "Directory containing internal_cutlass_kernels sources. If specified, the internal_cutlass_kernels and NVRTC wrapper libraries will be built from source."
    )
    parser.add_argument(
        "--build_root",
        type=Path,
        help=
        "Directory for all out-of-tree build state (also via TRTLLM_BUILD_ROOT env var). "
        "When set, the CMake build dir, build venv, wheel staging, intermediate "
        "objects and (with --use_ccache) the ccache directory default under this "
        "directory instead of the checkout. Point it at fast local storage (e.g. "
        "/tmp) when the checkout lives on a network filesystem. Individual "
        "options like --build_dir and CCACHE_DIR still override their piece.")
    parser.add_argument(
        "--out-of-tree",
        action="store_true",
        help=
        "Never write into the checkout: generated FMHA sources and version.h "
        "go to the build tree, and the wheel is assembled from an out-of-tree "
        "staging copy of the Python package (default wheel output: "
        "<build_root>/dist). Requires --build_root; the checkout may be "
        "mounted read-only. Incompatible with editable-install workflows "
        "(--skip_building_wheel, --linking_install_binary, --install) and "
        "with --version-override (it edits tensorrt_llm/version.py in place).")
    parser.add_argument(
        "--build_dir",
        type=Path,
        help=
        "Directory where C++ sources are built (default: cpp/build or cpp/build_<build_type>, "
        "or <build_root>/cpp-build* when --build_root is set)")
    parser.add_argument(
        "--dist_dir",
        type=Path,
        help="Directory where Python wheels are built (default: build/)")
    parser.add_argument(
        "--skip_building_wheel",
        "-s",
        action="store_true",
        help=
        "Skip building the *.whl files (they are only needed for distribution)")
    parser.add_argument(
        "--linking_install_binary",
        action="store_true",
        help=
        "Install the built binary by creating symbolic links instead of copying files"
    )
    parser.add_argument("--micro_benchmarks",
                        action="store_true",
                        help="Build the micro benchmarks for C++ components")
    parser.add_argument("--nvtx",
                        action="store_true",
                        help="Enable NVTX profiling features")
    parser.add_argument("--skip-stubs",
                        action="store_true",
                        help="Skip building Python type stubs")
    parser.add_argument("--generate_fmha",
                        action="store_true",
                        help="Generate the FMHA CUDA files")
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help=
        "Use the current Python interpreter without creating a virtual environment"
    )
    parser.add_argument(
        "--nvrtc_dynamic_linking",
        action="store_true",
        help="Link against dynamic NVRTC libraries instead of static ones")
    parser.add_argument("--mypyc",
                        action="store_true",
                        help="Compile kv_cache_manager_v2 with mypyc")
    parser.add_argument("--require_dynamic_attributions",
                        action="store_true",
                        help="Fail the build if attribution generation fails")

    def _plat_name_type(value):
        import re
        if not re.fullmatch(r'[a-zA-Z0-9_]+', value):
            raise ArgumentTypeError(
                f"Invalid plat name '{value}': only alphanumerics and underscores are allowed"
            )
        return value

    parser.add_argument(
        "--plat-name",
        type=_plat_name_type,
        help=
        "Wheel platform tag passed to bdist_wheel --plat-name (e.g. linux_x86_64, manylinux_2_28_x86_64)"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help=
        "Skip interactive confirmation prompts (useful for non-interactive builds)",
    )
    parser.add_argument(
        "--version-override",
        help="Package version override. A leading '.' or '+' appends to the "
        "current version; any other value replaces it.",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))
