"""Build and load the package's two native extensions.

Both are compiled on first use and cached per (source, argv, interpreter) key,
so a user compiles once per node and later processes load the same .so.  Neither
pulls in libtorch, CUTLASS or CuTe: the scheduler extension is ordinary CUDA
C++ and the SAMI submitter is C11, which is what keeps steady-state launch to
one short native call.

The build/cache/load machinery was duplicated across two modules; it lives in
``_build_and_load`` now and the two loaders differ only in what they must
differ in -- compiler argv, cache-key inputs, and the ABI each validates.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess  # nosec B404 -- compiler argv is validated and shell=False
import sys
import sysconfig
import tempfile
from typing import Callable, Sequence

_ROOT = Path(__file__).resolve().parent

_SAMI_SOURCE = _ROOT / "csrc" / "in_switch_copy" / "sami_hierarchical_release.c"
_SAMI_HEADER = _SAMI_SOURCE.with_name("sami_shards.h")
_SAMI_MODULE = "megamoe_sami_hierarchical"
_TMA_SOURCE = _SAMI_SOURCE.with_name("tma_copy.cu")
_TMA_HEADER = _SAMI_SOURCE.with_name("tma_copy.h")
_TMA_MODULE = "megamoe_sami_tma"

_HALO_Q_SOURCE = _ROOT / "cuda_scheduler" / "csrc" / "halo_q_scheduler.cu"
_COPY_POLICY_HEADER = _SAMI_SOURCE.with_name("plan_policy.cuh")
_HALO_Q_MODULE = "megamoe_halo_q_cuda"

_ARCH_ENV = "MEGAMOE_HALO_Q_ARCH"
_DEFAULT_ARCH = "sm_100"
# Only targets this scheduler has actually been built and run on.  An unknown
# value is rejected rather than passed through: nvcc would accept a wrong-but-
# valid arch and produce a module that loads, launches, and computes garbage.
_SUPPORTED_ARCHES = ("sm_100", "sm_100a", "sm_103", "sm_107", "sm_107a")

_NATIVE: dict[str, object] = {}
_DEFAULT_BUILD_ROOT = Path(tempfile.gettempdir()) / f"tensorrt_llm-{os.getuid()}"


def _secure_cache_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"native cache directory is not owned by this user: {directory}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        directory.chmod(0o700)
        info = directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError(f"cannot secure native cache directory: {directory}")


def _is_private_regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
        and info.st_nlink == 1
    )


def _remove_unsafe_cache_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError(f"unsafe native cache entry cannot be replaced: {path}")
    path.unlink()


def _open_cache_lock(path: Path):
    if not _is_private_regular_file(path):
        _remove_unsafe_cache_file(path)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise RuntimeError(f"native cache lock is not private: {path}")
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _target_arch() -> str:
    """Compilation target, defaulting to Blackwell.

    Deliberately an explicit override rather than device auto-detection: this
    module's contract is that it pulls in neither libtorch nor CUTLASS, and
    detection would need one of them (or a ctypes driver call) just to read a
    compute capability the caller already knows.  Callers that have torch in
    hand -- the framework seam, the standalone runners -- set it from
    ``torch.cuda.get_device_capability()``.

    The value reaches the cache key through the compiler argv, so switching
    targets produces a different key instead of silently reusing the other
    architecture's cubin.
    """

    arch = os.environ.get(_ARCH_ENV, _DEFAULT_ARCH).strip()
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"{_ARCH_ENV}={arch!r} is not a supported HALO-Q scheduler target; "
            f"expected one of {', '.join(_SUPPORTED_ARCHES)}"
        )
    return arch


def _sami_argv(output: Path) -> list[str]:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    return [
        os.environ.get("CC", "gcc"),
        "-O3",
        "-fPIC",
        "-shared",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-fno-plt",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{cuda_home}/include",
        str(_SAMI_SOURCE),
        "-o",
        str(output),
        f"-L{cuda_home}/lib64/stubs",
        "-lcuda",
        "-Wl,--no-as-needed",
    ]


def _halo_q_argv(output: Path) -> list[str]:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    return [
        os.path.join(cuda_home, "bin", "nvcc"),
        "-O3",
        "-std=c++17",
        "--shared",
        "--threads",
        "4",
        "-lineinfo",
        f"-arch={_target_arch()}",
        "-Xcompiler",
        "-fPIC",
        "-Xcompiler",
        "-fvisibility=hidden",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{cuda_home}/include",
        str(_HALO_Q_SOURCE),
        "-o",
        str(output),
        f"-L{cuda_home}/lib64",
        "-lcudart",
    ]


def _tma_commands(output: Path) -> list[list[str]]:
    """Keep the existing submitter C11; only the optional copy kernel is CUDA."""
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    host = output.with_suffix(".host.o")
    device = output.with_suffix(".device.o")
    return [
        [os.environ.get("CC", "gcc"), "-O3", "-std=c11", "-fPIC", "-c",
         "-DSAMI_ENABLE_TMA", f"-I{sysconfig.get_paths()['include']}",
         f"-I{cuda_home}/include", str(_SAMI_SOURCE), "-o", str(host)],
        [nvcc, "-O3", "-std=c++17", "-lineinfo", f"-arch={_target_arch()}",
         "-Xcompiler", "-fPIC", "-c", str(_TMA_SOURCE), "-o", str(device)],
        [nvcc, "--shared", str(host), str(device), "-o", str(output),
         f"-L{cuda_home}/lib64/stubs", "-lcuda", "-lcudart"],
    ]


def _tma_argv(output: Path) -> list[str]:
    # Include every compiler option in the ordinary source/cache digest.
    return [arg for command in _tma_commands(output) for arg in command]


def _cache_key(argv: Callable[[Path], Sequence[str]], sources: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    digest.update("\0".join(argv(Path("extension.so"))).encode())
    digest.update(sys.version.encode())
    return digest.hexdigest()[:20]


def _build_and_load(
    *,
    module_name: str,
    argv: Callable[[Path], Sequence[str]],
    sources: Sequence[Path],
    build_dir_env: str,
    default_build_dir: str,
    failure: str,
    validate: Callable[[object], None],
    force: bool,
    commands: Callable[[Path], Sequence[Sequence[str]]] | None = None,
):
    """Compile once per node/cache key, then load the private CPython module."""

    cached = _NATIVE.get(module_name)
    if cached is not None and not force:
        return cached
    directory = Path(os.environ.get(build_dir_env, default_build_dir)).expanduser()
    if build_dir_env not in os.environ:
        _secure_cache_directory(_DEFAULT_BUILD_ROOT)
    _secure_cache_directory(directory)
    key = _cache_key(argv, sources)
    output = directory / f"{module_name}_{key}.so"
    lock_path = directory / f"{module_name}_{key}.lock"
    with _open_cache_lock(lock_path) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reusable_output = _is_private_regular_file(output)
        if not reusable_output:
            _remove_unsafe_cache_file(output)
        if force or not reusable_output:
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".so", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                for command in (commands(temporary_path) if commands else [argv(temporary_path)]):
                    process = subprocess.run(command, capture_output=True, text=True)
                    if process.returncode:
                        raise RuntimeError(
                            failure + "\n" + " ".join(command)
                            + "\n--- stdout ---\n" + process.stdout
                            + "\n--- stderr ---\n" + process.stderr
                        )
                temporary_path.chmod(0o600)
                os.replace(temporary_path, output)
                if not _is_private_regular_file(output):
                    raise RuntimeError(f"native build produced an unsafe cache file: {output}")
            finally:
                temporary_path.unlink(missing_ok=True)
                if commands:
                    temporary_path.with_suffix(".host.o").unlink(missing_ok=True)
                    temporary_path.with_suffix(".device.o").unlink(missing_ok=True)

    specification = importlib.util.spec_from_file_location(module_name, output)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {output}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    validate(module)
    _NATIVE[module_name] = module
    return module


def _validate_sami(module: object) -> None:
    if int(getattr(module, "PLAN_ABI_VERSION", 0)) != 6:
        raise RuntimeError("hierarchical SAMI plan ABI mismatch")
    if int(getattr(module, "REMOTE_VISIBILITY_RELEASE", 0)) != 1:
        raise RuntimeError("hierarchical SAMI lacks the SYS-release guarantee")
    if not int(module.driver_symbol()):
        raise RuntimeError("hierarchical SAMI linked a null payload-launch symbol")


def _validate_halo_q(module: object) -> None:
    if int(getattr(module, "ABI_VERSION", 0)) != 6:
        raise RuntimeError("pure-CUDA scheduler extension ABI mismatch")


def load_hierarchical_native(force: bool = False):
    """Build and load the C-only hierarchical SAMI submitter."""

    return _build_and_load(
        module_name=_SAMI_MODULE,
        argv=_sami_argv,
        sources=(_SAMI_SOURCE, _SAMI_HEADER),
        build_dir_env="MEGAMOE_SAMI_HIERARCHICAL_BUILD_DIR",
        default_build_dir=str(_DEFAULT_BUILD_ROOT / "megamoe_sami_hierarchical"),
        failure="hierarchical SAMI native build failed",
        validate=_validate_sami,
        force=force,
    )


def load_tma_native(force: bool = False):
    """Optional SM copy backend, requiring CUDA 13.1+ for multicast TMA."""
    return _build_and_load(
        module_name=_TMA_MODULE, argv=_tma_argv,
        sources=(_SAMI_SOURCE, _SAMI_HEADER, _TMA_SOURCE, _TMA_HEADER,
                 _COPY_POLICY_HEADER, _SAMI_SOURCE.with_name("gpu_plan.h"),
                 _SAMI_SOURCE.with_name("gpu_plan_builder.cuh")),
        build_dir_env="MEGAMOE_SAMI_TMA_BUILD_DIR",
        default_build_dir=str(_DEFAULT_BUILD_ROOT / "megamoe_sami_tma"),
        failure="SM TMA copy build failed (CUDA 13.1+ required)",
        validate=_validate_sami, force=force, commands=_tma_commands,
    )


def load_scheduler_native(force: bool = False):
    """Build and load the AOT-style pure-CUDA HALO-Q scheduler extension."""

    return _build_and_load(
        module_name=_HALO_Q_MODULE,
        argv=_halo_q_argv,
        sources=(_HALO_Q_SOURCE, _COPY_POLICY_HEADER),
        build_dir_env="MEGAMOE_HALO_Q_BUILD_DIR",
        default_build_dir=str(_DEFAULT_BUILD_ROOT / "megamoe_halo_q_cuda"),
        failure="pure-CUDA scheduler build failed",
        validate=_validate_halo_q,
        force=force,
    )


__all__ = ["load_hierarchical_native", "load_scheduler_native", "load_tma_native"]
