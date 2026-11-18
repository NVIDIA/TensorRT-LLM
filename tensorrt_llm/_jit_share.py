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
"""Build each JIT kernel once across the processes of a launch.

The ranks of a launch compile the same kernels, each on its own. With
``TLLM_JIT_SHARE_DIR`` set to a directory that every process of the launch
can see (on a multi-node launch, a shared file system; use a fresh directory
per launch), the first process to claim a kernel builds it and the others
reuse the result. Every process runs the kernels it finds there, so
:func:`install` shares builds only when the directory, and the ones it holds
kernels in, belong to this user and other users cannot write to them (group
members who can write to them must be trusted). Where the directory cannot be
used (permissions, quota, I/O errors), a process builds each kernel itself, as
without sharing. Two kinds of kernels are shared:

* DeepGEMM's. DeepGEMM compiles a kernel the first time a shape selects it:
  it writes ``<tmp>/kernel.cu``, runs its compiler (``DG_JIT_NVCC_COMPILER``,
  by default ``$CUDA_HOME/bin/nvcc``) as ``nvcc <tmp>/kernel.cu --cubin
  --output-file <tmp>/kernel.cubin <flags>``, then renames ``<tmp>`` into its
  disk cache (``DG_JIT_CACHE_DIR``). :func:`install`, which runs at ``import
  tensorrt_llm``, makes this file DeepGEMM's compiler and, unless
  ``DG_JIT_CACHE_DIR`` is set, moves DeepGEMM's disk cache to
  ``<dir>/deep_gemm``, so that a kernel one process published is a cache hit
  for the others. The compiler claims a kernel as ``<dir>/kernels/<key>.claim``
  and publishes its cubin as ``<dir>/kernels/<key>.cubin``.
* CuTe DSL's. A runner compiles a kernel with ``cute.compile`` and keeps it in
  its class's ``kernel_cache`` dict. :func:`share_kernel_caches` makes that
  dict a :class:`SharedKernelCache`: while :class:`sharing_kernels` is active
  (the autotuner turns it on while it tunes), a lookup that misses loads the
  kernel another process exported as ``<dir>/cute/<name>.o``, and a kernel
  stored in it is exported there.

Either way, a process that finds a kernel claimed by another waits for it (at
most ``TLLM_JIT_SHARE_WAIT_S`` seconds, default 60, and not past the deadline
of an enclosing :class:`wait_until`) or, inside :class:`defer_busy_builds`,
fails at once with :data:`BUSY_MARK` in its error so that the caller can do
other work first. It builds the kernel itself when the claimant failed or the
wait ran out, so a claimant that died or hangs (for instance in I/O to the
shared file system) delays the others by one wait at most. A process never
reads the files another process claimed with, only checks whether results
exist: on a network file system, reading a file another client wrote can block
for as long as that client is unreachable.

A DeepGEMM key hashes the kernel source, the compiler arguments with the
temporary directory left out, and the contents of the compiler and of
DeepGEMM's headers. A CuTe DSL name hashes the runner class, the ``repr`` of
the cache key (keys whose ``repr`` holds an object address are not shared),
the TensorRT-LLM and CuTe DSL versions, and the GPU's compute capability.
Every other compiler invocation (``--version`` and the like) runs the real
compiler unchanged. With ``TLLM_JIT_SHARE_LOG=1`` each kernel adds a line
``<start> <seconds> <action> <key>`` to ``<dir>/log/<host>-<pid>.txt``.

A caller that runs a list of builds (the autotuner) splits the list across
the processes with :class:`SharedSweep` claims, and starts at an offset of its
own (:func:`start_fraction`).

Only the standard library may be imported at module scope: :func:`install`
runs before ``torch`` is imported, and the compiler runs this file in a bare
interpreter. Modules that only some paths need are imported where they are
used, to keep the compiler's start-up short.
"""

import hashlib
import os
import stat
import sys
import threading
import time

SHARE_DIR_ENV = "TLLM_JIT_SHARE_DIR"
BUSY_MARK = "TLLM_JIT_SHARE_BUSY"
BUSY_EXIT_CODE = 75  # EX_TEMPFAIL

_REAL_COMPILER_ENV = "TLLM_JIT_SHARE_NVCC"
_SALT_ENV = "TLLM_JIT_SHARE_SALT"
_DEFER_ENV = "TLLM_JIT_SHARE_DEFER"
_WAIT_ENV = "TLLM_JIT_SHARE_WAIT_S"
_DEADLINE_ENV = "TLLM_JIT_SHARE_DEADLINE"
_LOG_ENV = "TLLM_JIT_SHARE_LOG"
_PROCESS_ENV = "TLLM_JIT_SHARE_PROCESS"
# Linux reads at most this much of a script's "#!" line on older kernels.
_SHEBANG_LIMIT = 127

_installed = False
_slot: int | None = None
_kernel_sharing_depth = 0
_cute_salt: str | None = None
# .claim: (claim, failed) of the kernel this thread claimed and has not stored.
_unstored = threading.local()


class KernelBusy(RuntimeError):
    """A build refused because another process is building that kernel (see :class:`defer_busy_builds`)."""


def enabled() -> bool:
    """True when this process builds JIT kernels through the shared directory."""
    return _installed


def is_busy(error: BaseException) -> bool:
    """True for a build refused because another process is building that kernel."""
    return BUSY_MARK in str(error)


class defer_busy_builds:
    """Within the block, a build of a kernel that another process is building fails at
    once (see :func:`is_busy`) instead of waiting for that process.

    The setting is process-wide: it reaches DeepGEMM's compiler through the environment.
    """

    def __enter__(self):
        self._previous = os.environ.get(_DEFER_ENV)
        os.environ[_DEFER_ENV] = "1"
        return self

    def __exit__(self, *exc_info):
        if self._previous is None:
            os.environ.pop(_DEFER_ENV, None)
        else:
            os.environ[_DEFER_ENV] = self._previous
        return False


class wait_until:
    """Within the block, no wait for another process's build lasts past ``deadline``
    (a :func:`time.time` value), so that a block that waits for many builds waits at
    most until then in total; after it, the process builds what it still needs.

    The setting is process-wide: it reaches DeepGEMM's compiler through the environment.
    """

    def __init__(self, deadline: float):
        self._deadline = deadline

    def __enter__(self):
        self._previous = os.environ.get(_DEADLINE_ENV)
        os.environ[_DEADLINE_ENV] = repr(self._deadline)
        return self

    def __exit__(self, *exc_info):
        if self._previous is None:
            os.environ.pop(_DEADLINE_ENV, None)
        else:
            os.environ[_DEADLINE_ENV] = self._previous
        return False


def wait_seconds() -> float:
    """How long a process waits at most for a build of another process."""
    return float(os.environ.get(_WAIT_ENV, "60"))


def _wait_deadline() -> float:
    deadline = time.time() + wait_seconds()
    if os.environ.get(_DEADLINE_ENV):
        deadline = min(deadline, float(os.environ[_DEADLINE_ENV]))
    return deadline


def start_fraction() -> float:
    """Where this process starts a pass over a list of builds, as a fraction of it.

    The processes that share the directory take consecutive slots in it, in
    whatever way the launch started them; slot ``k`` starts at the fractional
    part of ``k`` times the golden ratio, which spreads any number of processes
    evenly over the list.
    """
    global _slot
    if _slot is None:
        slot = 0
        share_dir = os.environ.get(SHARE_DIR_ENV)
        if share_dir:
            slots = os.path.join(share_dir, "slots")
            try:
                os.makedirs(slots, exist_ok=True)
                slot = len(os.listdir(slots))
                while not _claim(os.path.join(slots, str(slot))):
                    slot += 1
            except OSError:
                slot = 0  # The directory is not usable here: start at the front.
        _slot = slot
    return (_slot * 0.6180339887498949) % 1.0


class SharedSweep:
    """Claims on the items of a list of builds that every process walks, such as the
    profiles of one autotuner sweep. ``name`` must be the same in every process.

    Where the directory is not usable (permissions, quota, I/O errors), this
    process claims every item, as it would without sharing.
    """

    def __init__(self, name: str):
        self.dir = os.path.join(
            os.environ[SHARE_DIR_ENV], "sweeps", hashlib.sha256(name.encode()).hexdigest()[:24]
        )
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError:
            self.dir = None

    def claim(self, index: int) -> bool:
        """Claim item ``index``; False if another process claimed it."""
        if self.dir is None:
            return True
        try:
            return _claim(os.path.join(self.dir, f"{index}.claim"))
        except OSError:
            self.dir = None
            return True

    def done(self, index: int) -> None:
        """Record that item ``index`` has run, so its builds are published."""
        if self.dir is None:
            return
        try:
            os.close(
                os.open(os.path.join(self.dir, f"{index}.done"), os.O_CREAT | os.O_WRONLY, 0o644)
            )
        except OSError:
            pass  # The others wait for this item until their wait runs out.

    def wait_done(self, index: int) -> bool:
        """Wait until the process that claimed item ``index`` has run it, or the wait runs
        out (see :func:`wait_seconds` and :class:`wait_until`); True if it has run."""
        deadline = _wait_deadline()
        while not self._is_done(index):
            if time.time() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def _is_done(self, index: int) -> bool:
        return self.dir is None or os.path.exists(os.path.join(self.dir, f"{index}.done"))


class sharing_kernels:
    """While active, :class:`SharedKernelCache` lookups and stores go through the
    shared directory; otherwise those caches are plain dicts."""

    def __enter__(self):
        global _kernel_sharing_depth
        _kernel_sharing_depth += 1
        return self

    def __exit__(self, *exc_info):
        global _kernel_sharing_depth
        _kernel_sharing_depth -= 1
        _release_unstored_claim()
        return False


def _kernel_sharing_active() -> bool:
    return _kernel_sharing_depth > 0 and bool(os.environ.get(SHARE_DIR_ENV))


def _release_unstored_claim() -> str | None:
    """Mark the kernel this thread claimed failed if it has not stored it, and return its claim.

    A runner stores the kernel it claimed right after compiling it, before it
    looks up another, so a claim still open at the thread's next lookup or store
    is one whose compile raised. The ``.failed`` file stops the other processes
    waiting for it.
    """
    pending = getattr(_unstored, "claim", None)
    if pending is None:
        return None
    _unstored.claim = None
    claim, failed = pending
    _mark_failed(failed, "compile raised")
    return claim


def _mark_failed(failed: str, reason: str) -> None:
    """Write the ``.failed`` file that stops the other processes waiting for a kernel, if the directory allows."""
    try:
        with open(failed, "w") as f:
            f.write(reason + "\n")
    except OSError:
        pass


class SharedKernelCache(dict):
    """A runner class's ``kernel_cache`` whose CuTe DSL kernels are shared through the directory.

    The runner's code is unchanged: it tests ``key in kernel_cache``, compiles
    on a miss, stores the result, and reads it back. Here a miss loads the
    kernel when another process exported it, waits for it when another process
    is compiling it (or, inside :class:`defer_busy_builds`, raises
    :class:`KernelBusy`), and otherwise claims it, so that the store that
    follows exports it. A claimed kernel whose compile raises is marked failed
    at the thread's next lookup or store, which the waiting processes see.
    """

    def __init__(self, owner: str, entries=()):
        super().__init__(entries)
        self.owner = owner

    def _paths(self, key):
        text = repr(key)
        if " at 0x" in text:
            return None
        name = (
            "k"
            + hashlib.sha256(f"{_cute_identity()}|{self.owner}|{text}".encode()).hexdigest()[:32]
        )
        base = os.path.join(os.environ[SHARE_DIR_ENV], "cute", name)
        return name, base + ".o", base + ".claim", base + ".failed", base + ".json"

    def __contains__(self, key) -> bool:
        failed_here = _release_unstored_claim()
        if dict.__contains__(self, key):
            return True
        if not _kernel_sharing_active():
            return False
        paths = self._paths(key)
        if paths is None:
            return False
        name, obj, claim, failed, meta = paths
        if failed_here == claim:
            return False  # Its compile raised here; the runner compiles it again.
        start = time.time()
        action = "cute-loaded"
        if not os.path.exists(obj):
            try:
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                claimed = _claim(claim)
            except OSError:
                return False  # The directory is not usable here: the runner compiles it.
            if claimed:
                _unstored.claim = (claim, failed)
                return False  # This process compiles it; the store exports it.
            if os.environ.get(_DEFER_ENV) == "1" and not os.path.exists(failed):
                _log(os.environ[SHARE_DIR_ENV], start, "cute-busy", name)
                raise KernelBusy(f"{BUSY_MARK} {name}")
            if not _wait(obj, failed):
                return False
            action = "cute-waited"
        try:
            kernel = _load_kernel(obj, name, meta)
        except Exception:  # noqa: BLE001 - a kernel that does not load is compiled here instead
            return False
        dict.__setitem__(self, key, kernel)
        _log(os.environ[SHARE_DIR_ENV], start, action, name)
        return True

    def __setitem__(self, key, kernel) -> None:
        dict.__setitem__(self, key, kernel)
        if not _kernel_sharing_active():
            return
        paths = self._paths(key)
        if paths is None:
            return
        name, obj, claim, failed, meta = paths
        pending = getattr(_unstored, "claim", None)
        if pending is not None and pending[0] == claim:
            _unstored.claim = None
        _release_unstored_claim()
        if os.path.exists(obj):
            return
        start = time.time()
        try:
            _export_kernel(kernel, obj, name, meta)
        except Exception as e:  # noqa: BLE001 - the kernel works here; the others compile it
            _mark_failed(failed, f"{type(e).__name__}: {e}")
            return
        _log(os.environ[SHARE_DIR_ENV], start, "cute-built", name)


def share_kernel_caches(runners) -> None:
    """Make the ``kernel_cache`` dict of each runner's class a :class:`SharedKernelCache`."""
    for runner in runners:
        for cls in type(runner).__mro__:
            cache = cls.__dict__.get("kernel_cache")
            if cache is None:
                continue
            if isinstance(cache, dict) and not isinstance(cache, SharedKernelCache):
                setattr(
                    cls,
                    "kernel_cache",
                    SharedKernelCache(f"{cls.__module__}.{cls.__qualname__}", cache),
                )
            break


def _cute_identity() -> str:
    """The TensorRT-LLM and CuTe DSL versions and the GPU's compute capability, which a kernel depends on."""
    global _cute_salt
    if _cute_salt is None:
        import importlib.metadata

        import torch

        from tensorrt_llm.version import __version__

        try:
            dsl = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            dsl = ""
        _cute_salt = f"{__version__}|{dsl}|{torch.cuda.get_device_capability()}"
    return _cute_salt


def _export_kernel(kernel, obj: str, name: str, meta: str) -> None:
    """Write a compiled CuTe DSL kernel to ``obj`` as an object file with entry symbol ``name``."""
    import json

    tvm_ffi = any(c.__name__.startswith("TVMFFI") for c in type(kernel).__mro__)
    part = f"{obj[:-2]}.{os.getpid()}.part.o"  # Ends in ".o": the TVM-FFI export picks its format by suffix.
    if tvm_ffi:
        # TVM-FFI functions export their entry as __tvm_ffi_<name>, which a TVM-FFI load looks up by name.
        kernel.export_to_c(part, function_name=name)
    else:
        with open(part, "wb") as f:
            f.write(kernel.dump_to_object(name))
    with open(f"{meta}.{os.getpid()}.part", "w") as f:
        json.dump({"tvm_ffi": tvm_ffi}, f)
    os.replace(f"{meta}.{os.getpid()}.part", meta)
    _fsync(part)
    os.replace(part, obj)


def _load_kernel(obj: str, name: str, meta: str):
    """Load the kernel another process exported with :func:`_export_kernel`."""
    import json

    import cutlass.runtime

    with open(meta) as f:
        tvm_ffi = json.load(f)["tvm_ffi"]
    return cutlass.runtime.load_module(obj, enable_tvm_ffi=tvm_ffi)[name]


def install() -> bool:
    """Make this file DeepGEMM's compiler if ``TLLM_JIT_SHARE_DIR`` is set.

    Must run before the process's first DeepGEMM call, which reads
    ``DG_JIT_NVCC_COMPILER``. Returns whether sharing is active.
    """
    global _installed
    share_dir = os.environ.get(SHARE_DIR_ENV)
    if _installed or not share_dir or os.name != "posix":
        return _installed
    # A process started by one that installed inherits the variables below;
    # its compiler is still the real one, not the parent's wrapper.
    compiler = (
        os.environ.get(_REAL_COMPILER_ENV)
        or os.environ.get("DG_JIT_NVCC_COMPILER")
        or _default_compiler()
    )
    if not compiler or not os.access(compiler, os.X_OK):
        return False
    # Every process runs the kernels it finds in these directories.
    deep_gemm = os.path.join(share_dir, "deep_gemm")
    code_dirs = [share_dir, os.path.join(share_dir, "kernels"), os.path.join(share_dir, "cute")]
    if os.environ.get("DG_JIT_CACHE_DIR", deep_gemm) == deep_gemm:
        code_dirs.append(deep_gemm)
    try:
        for path in code_dirs:
            os.makedirs(path, mode=0o755, exist_ok=True)
    except OSError:
        return False
    untrusted = next((path for path in code_dirs if not _trusted_dir(path)), None)
    if untrusted is not None:
        import warnings

        warnings.warn(
            f"{SHARE_DIR_ENV} is ignored: {untrusted} must be a directory of this user "
            "that other users cannot write to",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    wrapper = None
    for bin_dir in _wrapper_dirs(share_dir):
        try:
            wrapper = _write_wrapper(bin_dir)
        except OSError:
            continue
        if wrapper is not None:
            break
    if wrapper is None:
        return False
    os.environ[_REAL_COMPILER_ENV] = compiler
    if not os.environ.get(_SALT_ENV):
        os.environ[_SALT_ENV] = _salt(compiler)
    os.environ[_PROCESS_ENV] = f"{os.uname().nodename}-{os.getpid()}"
    os.environ["DG_JIT_NVCC_COMPILER"] = wrapper
    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(share_dir, "deep_gemm"))
    _installed = True
    return True


def _default_compiler() -> str | None:
    """DeepGEMM's own choice: ``$CUDA_HOME``, ``$CUDA_PATH``, ``which nvcc``, ``/usr/local/cuda``."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        import shutil

        nvcc = shutil.which("nvcc")
        if nvcc:
            home = os.path.dirname(os.path.dirname(nvcc))
    if not home and os.path.exists("/usr/local/cuda"):
        home = "/usr/local/cuda"
    return os.path.join(home, "bin", "nvcc") if home else None


def _wrapper_dirs(share_dir: str) -> list[str]:
    """Directories for the compiler wrapper, best first.

    DeepGEMM runs the wrapper at its path for as long as the process lives, so the
    wrapper must stay where no other user can replace it or a directory above it:
    this user's directory in the system temporary directory, whose parent is sticky
    (or private), first; the share directory's ``bin`` only if the share directory
    is this user's alone (a temporary directory may not allow executing files).
    """
    import tempfile

    tmp = tempfile.gettempdir()
    dirs = []
    try:
        info = os.stat(tmp)
        if info.st_mode & stat.S_ISVTX or not info.st_mode & 0o022:
            dirs.append(os.path.join(tmp, f"tllm_jit_share_{os.getuid()}"))
    except OSError:
        pass
    if _private_dir(share_dir):
        dirs.append(os.path.join(share_dir, "bin"))
    return dirs


def _trusted_dir(path: str) -> bool:
    """True for a directory that this user owns and other users cannot write to."""
    try:
        info = os.stat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and not info.st_mode & stat.S_IWOTH
    )


def _private_dir(path: str) -> bool:
    """True for a directory (not a symlink) that this user owns and no one else can write to."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o022


def _write_wrapper(bin_dir: str) -> str | None:
    """Write (once per interpreter and module path) the script DeepGEMM runs as its compiler.

    The script is a Python script, so that no shell starts per compiler call;
    a shell script is the fallback for an interpreter path too long for "#!".
    """
    interpreter = sys.executable or "/usr/bin/python3"
    module = os.path.abspath(__file__)
    if len(f"#!{interpreter} -IS") <= _SHEBANG_LIMIT:
        script = (
            f"#!{interpreter} -IS\n"
            f"import sys\n"
            f"sys.argv[0] = {module!r}\n"
            f"exec(compile(open({module!r}).read(), {module!r}, 'exec'),\n"
            f"     {{'__name__': '__main__', '__file__': {module!r}}})\n"
        )
    else:
        import shlex

        script = f'#!/bin/sh\nexec {shlex.join([interpreter, "-I", "-S", module])} "$@"\n'
    path = os.path.join(bin_dir, "nvcc-" + hashlib.sha256(script.encode()).hexdigest()[:16])
    os.makedirs(bin_dir, mode=0o700, exist_ok=True)
    if not _private_dir(bin_dir):
        return None
    if not os.path.exists(path):
        part = f"{path}.{os.getpid()}.part"
        with open(part, "w") as f:
            f.write(script)
        os.chmod(part, 0o700)  # The processes of a launch run as this user.
        os.replace(part, path)
    if os.statvfs(bin_dir).f_flag & os.ST_NOEXEC or not os.access(path, os.X_OK):
        return None
    return path


def _salt(compiler: str) -> str:
    """Hash of the contents of the compiler and of the headers DeepGEMM compiles against.

    Contents, not file times: each node installs its own copy, at its own time.
    """
    digest = hashlib.sha256()
    for path in (
        compiler,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_gemm", "include"),
    ):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for name in sorted(files):
                    file = os.path.join(root, name)
                    digest.update(os.path.relpath(file, path).encode() + b"\0")
                    _update_with_file(digest, file)
        elif os.path.isfile(path):
            _update_with_file(digest, path)
    return digest.hexdigest()


def _update_with_file(digest, path: str) -> None:
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
    except OSError:
        pass


def _log(share_dir: str, start: float, action: str, key: str) -> None:
    if os.environ.get(_LOG_ENV) != "1":
        return
    process = os.environ.get(_PROCESS_ENV) or str(os.getppid())
    try:
        os.makedirs(os.path.join(share_dir, "log"), exist_ok=True)
        with open(os.path.join(share_dir, "log", f"{process}.txt"), "a") as f:
            f.write(f"{start:.3f} {time.time() - start:.3f} {action} {key}\n")
    except OSError:
        pass


def _fsync(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_file(source: str, destination: str) -> None:
    with open(source, "rb") as src, open(destination, "wb") as dst:
        dst.write(src.read())


def _copy(cubin: str, output: str) -> int:
    _copy_file(cubin, output)
    return 0 if os.path.getsize(output) > 0 else 1


def _publish(output: str, cubin: str) -> None:
    part = f"{os.path.dirname(cubin)}/.{os.path.basename(cubin)}.{os.getpid()}.part"
    _copy_file(output, part)
    _fsync(part)
    os.replace(part, cubin)


def _build(compiler: str, argv: list[str], output: str, cubin: str, failed: str) -> int:
    import subprocess  # nosec B404 - runs the real compiler on the arguments DeepGEMM passed, no shell

    returncode = subprocess.run([compiler] + argv).returncode
    if returncode == 0 and os.path.isfile(output) and os.path.getsize(output) > 0:
        try:
            _publish(output, cubin)
        except OSError as e:  # The kernel is built here; the others build it themselves.
            _mark_failed(failed, f"publish: {e}")
    elif not os.path.exists(cubin):
        _mark_failed(failed, f"exit code {returncode}")
    return returncode


def _claim(claim: str) -> bool:
    """Create ``claim``; False if it exists. Other errors (permissions, quota, I/O) raise OSError."""
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{os.uname().nodename} {os.getpid()}\n".encode())
    except OSError:
        pass  # The claim stands; its text only says who made it.
    finally:
        os.close(fd)
    return True


def _wait(result: str, failed: str) -> bool:
    deadline = _wait_deadline()
    while time.time() < deadline and not os.path.exists(failed):
        if os.path.exists(result):
            return True
        time.sleep(0.02)
    return False


def _kernel_key(source: str, argv: list[str]) -> str:
    """Hash of the kernel source, the arguments without its temporary directory, and the salt."""
    tmp_dir = os.path.dirname(os.path.abspath(source))
    digest = hashlib.sha256(os.environ.get(_SALT_ENV, "").encode())
    with open(source, "rb") as f:
        digest.update(f.read())
    for arg in argv:
        digest.update(b"\0" + arg.replace(tmp_dir, "@").encode())
    return digest.hexdigest()[:40]


def main(argv: list[str]) -> int:
    """Compile like nvcc, sharing ``--cubin`` builds of ``kernel.cu`` through the directory."""
    compiler = os.environ.get(_REAL_COMPILER_ENV) or _default_compiler() or "nvcc"
    share_dir = os.environ.get(SHARE_DIR_ENV)
    source = next((a for a in argv if os.path.basename(a) == "kernel.cu"), None)
    if not share_dir or source is None or "--cubin" not in argv or "--output-file" not in argv:
        os.execvp(compiler, [compiler] + argv)  # nosec B606 - becomes the real compiler, same arguments, no shell
    start = time.time()
    output = argv[argv.index("--output-file") + 1]
    key = _kernel_key(source, argv)
    kernels = os.path.join(share_dir, "kernels")
    cubin, claim, failed = (os.path.join(kernels, key + s) for s in (".cubin", ".claim", ".failed"))
    try:
        os.makedirs(kernels, exist_ok=True)
        if os.path.exists(cubin):
            action, returncode = "copied", _copy(cubin, output)
        elif _claim(claim):
            action, returncode = "built", _build(compiler, argv, output, cubin, failed)
        elif os.environ.get(_DEFER_ENV) == "1" and not os.path.exists(failed):
            print(f"{BUSY_MARK} {key}", flush=True)
            action, returncode = "busy", BUSY_EXIT_CODE
        elif _wait(cubin, failed):
            action, returncode = "waited", _copy(cubin, output)
        else:
            action, returncode = "fallback", _build(compiler, argv, output, cubin, failed)
    except OSError:
        # The directory is not usable here (permissions, quota, I/O): build as without sharing.
        os.execvp(compiler, [compiler] + argv)  # nosec B606 - becomes the real compiler, same arguments, no shell
    _log(share_dir, start, action, key)
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
