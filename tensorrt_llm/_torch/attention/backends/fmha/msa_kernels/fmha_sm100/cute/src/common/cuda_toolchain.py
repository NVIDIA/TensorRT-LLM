# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Capability probes for the CUDA toolchain that assembles our inline PTX.

``cutlass.CUDA_VERSION`` reports the CUDA version the CuTe DSL was *built*
against, not the version of the libNVVM/ptxas that will actually compile the
generated device IR. Those differ whenever a wheel built against a newer CUDA
is installed next to an older toolkit -- and when they do, version-gated inline
PTX is emitted that the assembler cannot parse. The only diagnostic surfaced in
that case is a bare::

    NVVM backend compilation failed
    libNVVM failed while compiling generated device IR.

with no mention of the offending instruction, which is extremely hard to
diagnose from the kernel side.

``ptxas_supports()`` answers the question directly by assembling a
one-instruction kernel with the real ptxas. That is correct across mixed
toolkit installations and self-maintaining as instructions land in new
releases, so gates built on it do not need to encode a CUDA version threshold
that may turn out to be wrong.
"""

import functools
import os
import shutil
import subprocess
import tempfile

__all__ = [
    "find_ptxas",
    "toolkit_cuda_version",
    "max_ptx_isa_version",
    "ptxas_supports",
]

# Highest first: the probe takes the first version ptxas accepts.
_CANDIDATE_ISA_VERSIONS = (
    "9.5", "9.4", "9.3", "9.2", "9.1", "9.0",
    "8.8", "8.7", "8.6", "8.5", "8.4", "8.3", "8.2", "8.1", "8.0",
)

_PROBE_TIMEOUT_SECONDS = 30


@functools.lru_cache(maxsize=1)
def find_ptxas() -> str | None:
    """Locate the ptxas that belongs to the toolkit the DSL will use."""
    for var in ("CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            candidate = os.path.join(root, "bin", "ptxas")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which("ptxas")
    if found:
        return found
    candidate = "/usr/local/cuda/bin/ptxas"
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


@functools.lru_cache(maxsize=1)
def toolkit_cuda_version() -> tuple[int, int] | None:
    """(major, minor) of the local ptxas, or None if it cannot be determined."""
    ptxas = find_ptxas()
    if ptxas is None:
        return None
    try:
        proc = subprocess.run(
            [ptxas, "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # "Cuda compilation tools, release 13.2, V13.2.51"
    for token in proc.stdout.replace(",", " ").split():
        if token.startswith("V") and token.count(".") >= 1:
            parts = token[1:].split(".")
            try:
                return int(parts[0]), int(parts[1])
            except (IndexError, ValueError):
                continue
    return None


def _default_target() -> str:
    """Arch string to probe with; matches the device we will compile for."""
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}a"
    except Exception:
        return "sm_100a"


def _assemble(body: str, isa_version: str, target: str) -> bool:
    """True iff ptxas accepts a kernel whose body is ``body``."""
    ptxas = find_ptxas()
    if ptxas is None:
        return False
    source = (
        f".version {isa_version}\n"
        f".target {target}\n"
        ".address_size 64\n"
        ".visible .entry _msa_probe()\n"
        "{\n"
        "  .reg .b8 %b<8>;\n"
        "  .reg .b16 %h<8>;\n"
        "  .reg .b32 %r<8>;\n"
        "  .reg .b64 %rd<8>;\n"
        "  mov.u32 %r1, 0;\n"
        "  mov.u64 %rd1, 0;\n"
        f"{body}\n"
        "  ret;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.ptx")
        with open(path, "w") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                [ptxas, f"-arch={target}", path, "-o", os.path.join(tmp, "probe.cubin")],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return proc.returncode == 0


@functools.lru_cache(maxsize=8)
def max_ptx_isa_version(target: str | None = None) -> str | None:
    """Highest ``.version`` the local ptxas accepts for ``target``.

    Probing the instruction at too low an ISA version would report a false
    negative for an instruction the toolkit actually supports, so the capability
    probe must first find the ceiling.
    """
    target = target or _default_target()
    for isa_version in _CANDIDATE_ISA_VERSIONS:
        if _assemble("", isa_version, target):
            return isa_version
    return None


@functools.lru_cache(maxsize=64)
def ptxas_supports(body: str, target: str | None = None) -> bool | None:
    """Whether ptxas assembles ``body``.

    Returns True/False when ptxas could be run, and None when it could not, so
    callers can distinguish "unsupported" from "unknown" and fall back to a
    version comparison rather than silently taking the slow path.
    """
    if find_ptxas() is None:
        return None
    target = target or _default_target()
    isa_version = max_ptx_isa_version(target)
    if isa_version is None:
        return None
    return _assemble(body, isa_version, target)
