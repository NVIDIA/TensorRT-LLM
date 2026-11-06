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
"""Build each DeepGEMM JIT kernel once across the processes of a launch.

DeepGEMM compiles a kernel the first time a shape selects it: it writes
``<tmp>/kernel.cu``, runs its compiler (``DG_JIT_NVCC_COMPILER``, by default
``$CUDA_HOME/bin/nvcc``) as ``nvcc <tmp>/kernel.cu --cubin --output-file
<tmp>/kernel.cubin <flags>``, then renames ``<tmp>`` into its disk cache
(``DG_JIT_CACHE_DIR``). The ranks of a launch select the same kernels, so each
rank compiles every kernel on its own.

``TLLM_DEEP_GEMM_JIT_SHARE_DIR`` names a directory that every process of the
launch can see (on a multi-node launch, a shared file system; use a fresh
directory per launch). When it is set, :func:`install`, which runs at
``import tensorrt_llm``, makes this file DeepGEMM's compiler and, unless
``DG_JIT_CACHE_DIR`` is set, moves DeepGEMM's disk cache to
``<dir>/deep_gemm`` so that a kernel one process published is a cache hit for
the others. For each kernel, the compiler then

* builds it in the first process that claims it (exclusive create of
  ``<dir>/kernels/<key>.claim``) and publishes the cubin as
  ``<dir>/kernels/<key>.cubin``;
* in a process that finds it claimed, waits for that cubin and copies it
  (at most ``TLLM_DEEP_GEMM_JIT_SHARE_WAIT_S`` seconds, default 300), or,
  inside :class:`defer_busy_builds`, fails at once with :data:`BUSY_MARK` in
  its output so that the caller can do other work first;
* builds it itself when the claimant failed, the wait ran out, or the claim is
  older than ``TLLM_DEEP_GEMM_JIT_SHARE_CLAIM_TTL_S`` seconds (default 600).

The key hashes the kernel source, the compiler arguments with the temporary
directory left out, and the identity of the compiler and of DeepGEMM's
headers. Every other compiler invocation (``--version`` and the like) runs the
real compiler unchanged. With ``TLLM_DEEP_GEMM_JIT_SHARE_LOG=1`` each kernel
adds a line ``<start> <seconds> <action> <key>`` to
``<dir>/log/<host>-<pid>.txt``.

A caller that runs a list of builds (the autotuner's swap-AB GEMM sweeps)
splits the list across the processes with :class:`SharedSweep` claims, and
starts at an offset of its own (:func:`start_fraction`).

Only the standard library may be imported here: :func:`install` runs before
``torch`` is imported, and the compiler runs this file in a bare interpreter.
Modules that only some paths need are imported where they are used, to keep
the compiler's start-up short.
"""

import hashlib
import os
import sys
import time

SHARE_DIR_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_DIR"
BUSY_MARK = "TLLM_DEEP_GEMM_JIT_BUSY"
BUSY_EXIT_CODE = 75  # EX_TEMPFAIL

_REAL_COMPILER_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_NVCC"
_SALT_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_SALT"
_DEFER_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_DEFER"
_WAIT_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_WAIT_S"
_CLAIM_TTL_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_CLAIM_TTL_S"
_LOG_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_LOG"
_PROCESS_ENV = "TLLM_DEEP_GEMM_JIT_SHARE_PROCESS"
# Linux reads at most this much of a script's "#!" line on older kernels.
_SHEBANG_LIMIT = 127

_installed = False
_slot: int | None = None


def enabled() -> bool:
    """True when this process builds DeepGEMM kernels through the shared directory."""
    return _installed


def is_busy(error: BaseException) -> bool:
    """True for a DeepGEMM build refused because another process is building that kernel."""
    return BUSY_MARK in str(error)


class defer_busy_builds:
    """Within the block, a build of a kernel that another process is building fails at
    once (see :func:`is_busy`) instead of waiting for that process.

    The setting is process-wide: it reaches the compiler through the environment.
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
            os.makedirs(slots, exist_ok=True)
            slot = len(os.listdir(slots))
            while not _claim(os.path.join(slots, str(slot))):
                slot += 1
        _slot = slot
    return (_slot * 0.6180339887498949) % 1.0


class SharedSweep:
    """Claims on the items of a list of builds that every process walks, such as the
    profiles of one autotuner sweep. ``name`` must be the same in every process."""

    def __init__(self, name: str):
        self.dir = os.path.join(
            os.environ[SHARE_DIR_ENV], "sweeps", hashlib.sha256(name.encode()).hexdigest()[:24]
        )
        os.makedirs(self.dir, exist_ok=True)

    def claim(self, index: int) -> bool:
        """Claim item ``index``; False if another process claimed it (and the claim is not stale)."""
        path = os.path.join(self.dir, f"{index}.claim")
        return _claim(path) or (_claim_is_stale(path) and not self._is_done(index))

    def done(self, index: int) -> None:
        """Record that item ``index`` has run, so its builds are published."""
        os.close(os.open(os.path.join(self.dir, f"{index}.done"), os.O_CREAT | os.O_WRONLY, 0o644))

    def wait_done(self, index: int) -> None:
        """Wait until the process that claimed item ``index`` has run it, or its claim is stale."""
        claim = os.path.join(self.dir, f"{index}.claim")
        while not self._is_done(index) and not _claim_is_stale(claim):
            time.sleep(0.02)

    def _is_done(self, index: int) -> bool:
        return os.path.exists(os.path.join(self.dir, f"{index}.done"))


def install() -> bool:
    """Make this file DeepGEMM's compiler if ``TLLM_DEEP_GEMM_JIT_SHARE_DIR`` is set.

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
    try:
        os.makedirs(os.path.join(share_dir, "kernels"), exist_ok=True)
    except OSError:
        return False
    import tempfile

    # The wrapper must sit on a file system that allows executing it.
    wrapper = None
    for bin_dir in (
        os.path.join(share_dir, "bin"),
        os.path.join(tempfile.gettempdir(), f"tllm_deep_gemm_jit_{os.getuid()}"),
    ):
        try:
            wrapper = _write_wrapper(bin_dir)
        except OSError:
            continue
        if wrapper is not None:
            break
    if wrapper is None:
        return False
    os.environ[_REAL_COMPILER_ENV] = compiler
    os.environ.setdefault(_SALT_ENV, _salt(compiler))
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
    if not os.path.exists(path):
        os.makedirs(bin_dir, exist_ok=True)
        part = f"{path}.{os.getpid()}.part"
        with open(part, "w") as f:
            f.write(script)
        os.chmod(part, 0o755)
        os.replace(part, path)
    if os.statvfs(bin_dir).f_flag & os.ST_NOEXEC or not os.access(path, os.X_OK):
        return None
    return path


def _salt(compiler: str) -> str:
    """Identity of the compiler and of the headers DeepGEMM compiles against."""
    parts = []
    for path in (
        compiler,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_gemm", "include"),
    ):
        count, size, newest = 0, 0, 0
        for root, _, files in os.walk(path) if os.path.isdir(path) else [("", [], [path])]:
            for name in files:
                try:
                    st = os.stat(os.path.join(root, name))
                except OSError:
                    continue
                count, size, newest = count + 1, size + st.st_size, max(newest, st.st_mtime_ns)
        parts.append(f"{path}:{count}:{size}:{newest}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


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


def _copy_file(source: str, destination: str) -> None:
    with open(source, "rb") as src, open(destination, "wb") as dst:
        dst.write(src.read())


def _copy(cubin: str, output: str) -> int:
    _copy_file(cubin, output)
    return 0 if os.path.getsize(output) > 0 else 1


def _publish(output: str, cubin: str) -> None:
    part = f"{os.path.dirname(cubin)}/.{os.path.basename(cubin)}.{os.getpid()}.part"
    _copy_file(output, part)
    fd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(part, cubin)


def _build(compiler: str, argv: list[str], output: str, cubin: str, failed: str) -> int:
    import subprocess

    returncode = subprocess.run([compiler] + argv).returncode
    if returncode == 0 and os.path.isfile(output) and os.path.getsize(output) > 0:
        _publish(output, cubin)
    elif not os.path.exists(cubin):
        with open(failed, "w") as f:
            f.write(f"exit code {returncode}\n")
    return returncode


def _claim(claim: str) -> bool:
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, f"{os.uname().nodename} {os.getpid()}\n".encode())
    os.close(fd)
    return True


def _claim_is_stale(claim: str) -> bool:
    try:
        age = time.time() - os.stat(claim).st_mtime
    except FileNotFoundError:
        return True
    return age > float(os.environ.get(_CLAIM_TTL_ENV, "600"))


def _wait(cubin: str, failed: str) -> bool:
    deadline = time.monotonic() + float(os.environ.get(_WAIT_ENV, "300"))
    while time.monotonic() < deadline and not os.path.exists(failed):
        if os.path.exists(cubin):
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
        os.execvp(compiler, [compiler] + argv)
    start = time.time()
    output = argv[argv.index("--output-file") + 1]
    key = _kernel_key(source, argv)
    kernels = os.path.join(share_dir, "kernels")
    os.makedirs(kernels, exist_ok=True)
    cubin, claim, failed = (os.path.join(kernels, key + s) for s in (".cubin", ".claim", ".failed"))

    if os.path.exists(cubin):
        action, returncode = "copied", _copy(cubin, output)
    elif _claim(claim):
        action, returncode = "built", _build(compiler, argv, output, cubin, failed)
    elif _claim_is_stale(claim):
        action, returncode = "stale", _build(compiler, argv, output, cubin, failed)
    elif os.environ.get(_DEFER_ENV) == "1" and not os.path.exists(failed):
        print(f"{BUSY_MARK} {key}", flush=True)
        action, returncode = "busy", BUSY_EXIT_CODE
    elif _wait(cubin, failed):
        action, returncode = "waited", _copy(cubin, output)
    else:
        action, returncode = "fallback", _build(compiler, argv, output, cubin, failed)
    _log(share_dir, start, action, key)
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
