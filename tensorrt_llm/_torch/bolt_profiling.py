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
"""POC: constrain LLVM BOLT instrumentation profiles to steady state.

When TensorRT-LLM libraries are BOLT-instrumented, the profile counters
accumulate from process start, so a single workload run produces `.fdata`
dominated by one-time startup (weight load, NVRTC/jitify compiles, autotuning,
CUDA-graph capture) rather than steady-state serving. BOLT's instrumentation
runtime exposes ``__bolt_instr_clear_counters()`` to reset counters and begin a
fresh profiling phase (see llvm ``bolt/runtime/instr.cpp``).

This module calls that function, once, at the end of engine warmup -- after all
JIT / autotune / graph-capture has happened but before the measured requests --
so the resulting profile reflects steady state only.

Key facts driving the design:
  * The runtime (and thus the symbol) is embedded *into each instrumented .so*
    at instrument time. Nothing is downloaded or dlopen'd at runtime; we only
    resolve + call a symbol that already lives in the loaded libraries.
  * A single BOLT run instruments MULTIPLE objects (libtensorrt_llm.so,
    libnvinfer_plugin_tensorrt_llm.so, libth_common.so, the Python bindings,
    libtriton_tensorrtllm.so). Each embeds its own runtime and its own copy of
    the symbol + counters, so we must clear EVERY instrumented lib loaded in the
    process, not just one.
  * The symbol may be hidden-visibility (not in .dynsym), so a plain dlsym can
    fail; we fall back to resolving its address from the ELF symbol table plus
    the library's load base in /proc/self/maps.

Entirely gated by ``TLLM_BOLT_CLEAR_COUNTERS=1`` -- a complete no-op otherwise,
so it is inert on every normal (non-instrumented) build.
"""

from __future__ import annotations

import ctypes
import os
import subprocess  # nosec B404
from typing import Dict, List, Optional

from tensorrt_llm.logger import logger

#: CI sets this to "1" only in the BOLT profile-generation job.
CLEAR_COUNTERS_ENV = "TLLM_BOLT_CLEAR_COUNTERS"

#: When "1", raise instead of warn-and-skip if the counters can't be cleared.
#: Used for the POC so a run that silently fails to clear fails LOUDLY (letting
#: us confirm whether the symbol is resolvable) rather than producing an
#: un-cleared profile that looks fine. Off by default -> production stays inert.
CLEAR_STRICT_ENV = "TLLM_BOLT_CLEAR_STRICT"

#: The BOLT runtime entry point that zeroes all instrumentation counters.
CLEAR_SYMBOL = "__bolt_instr_clear_counters"

#: File written at instrument time mapping "<lib_basename> <hex_offset>", where
#: the offset is BOLT's printed "clear procedure is 0x..." address. This is the
#: ONLY reliable way to locate the clear routine: BOLT emits it as a LOCAL
#: symbol (absent from .dynsym AND .symtab), so dlsym/nm can't find it -- but it
#: prints the address at instrument time (scripts/bolt/bolt_lib.sh captures it).
CLEAR_OFFSETS_ENV = "BOLT_CLEAR_OFFSETS_FILE"

#: Basenames (substring match) of the objects we BOLT-instrument. MUST mirror the
#: target set actually instrumented in scripts/bolt/bolt_lib.sh, otherwise a lib
#: listed here but NOT instrumented is still found loaded, fails clear-symbol
#: resolution, and (with TLLM_BOLT_CLEAR_STRICT=1) fails the run. Currently P0-only
#: -- the P1 entries (bindings, libtriton_tensorrtllm) are commented out to match
#: bolt_lib.sh, which disabled P1 while isolating the disagg GEN worker-init hang.
#: Re-enable these IN LOCKSTEP with the P1 lines in bolt_lib.sh.
_TARGET_LIB_SUBSTRINGS = (
    "libtensorrt_llm.so",
    "libnvinfer_plugin_tensorrt_llm.so",
    "libth_common.so",
    # --- P1 (disabled; re-enable together with bolt_lib.sh P1) ---
    # "libtriton_tensorrtllm.so",
    # "bindings",  # bindings.cpython-<ver>-<arch>.so
)

# One-shot latch: warmup can be re-entered in some paths; only clear once.
_already_cleared = False


def _is_enabled() -> bool:
    return os.environ.get(CLEAR_COUNTERS_ENV, "").strip() == "1"


def _iter_loaded_target_libs() -> Dict[str, int]:
    """Map {realpath: load_base} for loaded libs matching the instrument set.

    load_base is the lowest mapped address for the file in /proc/self/maps,
    which for a shared object is what a symbol's ELF st_value is relative to.
    """
    libs: Dict[str, int] = {}
    try:
        with open("/proc/self/maps", "r") as f:
            maps = f.read()
    except OSError as e:
        logger.warning(f"[bolt] cannot read /proc/self/maps: {e}")
        return libs

    for line in maps.splitlines():
        # Format: <start>-<end> perms offset dev inode  pathname
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if not path.startswith("/"):
            continue
        base_name = os.path.basename(path)
        if not any(s in base_name for s in _TARGET_LIB_SUBSTRINGS):
            continue
        try:
            real = os.path.realpath(path)
        except OSError:
            real = path
        # Only our instrumented objects, which all live under the tensorrt_llm
        # package. This excludes look-alikes that match "bindings" but aren't
        # ours (e.g. xgrammar/xgrammar_bindings.cpython-*.so).
        if "tensorrt_llm" not in real:
            continue
        start = int(line.split("-", 1)[0], 16)
        # Keep the lowest start seen for this file == load base.
        if real not in libs or start < libs[real]:
            libs[real] = start
    return libs


def _call_via_dlsym(path: str) -> bool:
    """Try to resolve + call the symbol via a per-lib handle (dynsym path)."""
    try:
        # RTLD_NOLOAD: get a handle to the already-loaded lib, don't reload it.
        handle = ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
    except OSError:
        return False
    try:
        fn = getattr(handle, CLEAR_SYMBOL)
    except AttributeError:
        return False  # hidden visibility -> not in .dynsym
    fn.restype = None
    fn.argtypes = []
    fn()
    return True


def _nm_tools() -> List[str]:
    """`nm` binaries to try, preferring the staged LLVM toolchain.

    The symbol is hidden (absent from .dynsym) so we read .symtab via nm. The
    workload container (a pytorch base image) usually has NO binutils, but the
    BOLT profile-gen job stages an LLVM toolchain and exports BOLT_LLVM_DIR into
    the container -- llvm-nm lives there. Fall back to any system nm.
    """
    tools: List[str] = []
    llvm_dir = os.environ.get("BOLT_LLVM_DIR", "").strip()
    if llvm_dir:
        tools.append(os.path.join(llvm_dir, "bin", "llvm-nm"))
    tools += ["llvm-nm", "nm"]
    return tools


def _symbol_offset(path: str) -> Optional[int]:
    """Read st_value for CLEAR_SYMBOL from .symtab via (llvm-)nm.

    nm rows: ``<hex_addr> <type> <name>``; a defined text symbol has type t/T.
    Returns the .so-relative offset, or None if no tool ran or the symbol is
    genuinely absent (stripped).
    """
    for tool in _nm_tools():
        try:
            out = subprocess.check_output([tool, path], stderr=subprocess.DEVNULL, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue  # tool missing/failed -> try the next
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == CLEAR_SYMBOL and parts[-2] in ("t", "T"):
                try:
                    return int(parts[0], 16)
                except ValueError:
                    pass
        return None  # a tool ran; symbol simply isn't in .symtab
    return None


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _addr_in_lib(addr: int, path: str) -> bool:
    """dladdr sanity check: confirm `addr` maps into `path`'s image.

    Calling a mis-computed absolute address would SIGSEGV the workload (a long,
    expensive run), so gate the raw call on dladdr agreeing the address belongs
    to the expected library.
    """
    try:
        libc = ctypes.CDLL(None)
        info = _DlInfo()
        if libc.dladdr(ctypes.c_void_p(addr), ctypes.byref(info)) == 0:
            return False
        fname = (info.dli_fname or b"").decode(errors="replace")
        return os.path.basename(os.path.realpath(fname)) == os.path.basename(os.path.realpath(path))
    except Exception:
        return False


def _call_at_offset(path: str, load_base: int, offset: int) -> bool:
    """Call the clear routine at load_base+offset, gated by a dladdr check.

    A mis-computed absolute address would SIGSEGV the (long) run, so only call
    once dladdr confirms the address maps into the expected library.
    """
    addr = load_base + offset
    if not _addr_in_lib(addr, path):
        logger.warning(
            f"[bolt] computed {CLEAR_SYMBOL} addr 0x{addr:x} not in {path} "
            "(dladdr mismatch); skipping to avoid a crash"
        )
        return False
    proto = ctypes.CFUNCTYPE(None)
    try:
        proto(addr)()
    except (ValueError, OSError):
        return False
    return True


def _clear_offsets_from_file() -> Dict[str, int]:
    """Parse the instrument-time "<lib_basename> <hex_offset>" file.

    The offset is BOLT's printed "clear procedure is 0x..." address (the local,
    unexported __bolt_instr_clear_counters). Returns {basename: offset}; empty
    if the file is unset/absent.
    """
    out: Dict[str, int] = {}
    fpath = os.environ.get(CLEAR_OFFSETS_ENV, "").strip()
    if not fpath or not os.path.exists(fpath):
        return out
    try:
        with open(fpath, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, hexoff = parts
                try:
                    out[name] = int(hexoff, 16)
                except ValueError:
                    continue
    except OSError as e:
        logger.warning(f"[bolt] cannot read {CLEAR_OFFSETS_ENV}={fpath}: {e}")
    return out


def _call_via_address(path: str, load_base: int) -> bool:
    """Fallback: derive the offset from .symtab via (llvm-)nm, then call it."""
    offset = _symbol_offset(path)
    if offset is None:
        return False
    return _call_at_offset(path, load_base, offset)


def maybe_bolt_clear_counters() -> int:
    """Clear BOLT counters for every instrumented lib in this process (once).

    Returns the number of libraries successfully cleared (0 when disabled or on
    a non-instrumented build). Safe no-op unless ``TLLM_BOLT_CLEAR_COUNTERS=1``.
    """
    global _already_cleared
    if _already_cleared or not _is_enabled():
        return 0

    strict = os.environ.get(CLEAR_STRICT_ENV, "").strip() == "1"
    _already_cleared = True

    libs = _iter_loaded_target_libs()
    if not libs:
        msg = (
            "[bolt] TLLM_BOLT_CLEAR_COUNTERS set but no instrumented target "
            "libs found in /proc/self/maps (non-instrumented build?)"
        )
        if strict:
            raise RuntimeError(msg)
        logger.info(msg)
        return 0

    # PRIMARY: BOLT-printed "clear procedure" offsets captured at instrument
    # time (the routine is a local symbol, so dlsym/nm can't see it). dlsym/nm
    # are kept only as best-effort fallbacks for other BOLT builds.
    offsets = _clear_offsets_from_file()

    cleared: List[str] = []
    failed: List[str] = []
    for path, load_base in libs.items():
        base = os.path.basename(path)
        off = offsets.get(base)
        ok = (
            (off is not None and _call_at_offset(path, load_base, off))
            or _call_via_dlsym(path)
            or _call_via_address(path, load_base)
        )
        if ok:
            cleared.append(base)
        else:
            failed.append(path)
            logger.warning(
                f"[bolt] could not resolve {CLEAR_SYMBOL} in {path} "
                f"(offsets-file hit={off is not None}; dlsym/nm also failed)"
            )

    if cleared:
        logger.info(
            f"[bolt] cleared instrumentation counters after warmup for "
            f"{len(cleared)} lib(s): {', '.join(sorted(cleared))}"
        )
    if failed and strict:
        raise RuntimeError(
            f"[bolt] TLLM_BOLT_CLEAR_STRICT=1 and could not clear {CLEAR_SYMBOL} "
            f"in {len(failed)} lib(s): {', '.join(sorted(os.path.basename(p) for p in failed))}. "
            "Symbol is likely hidden AND stripped from .symtab, or (llvm-)nm is "
            "unavailable. Failing fast so the profile isn't silently un-cleared."
        )
    return len(cleared)


def _self_check(paths: List[str]) -> int:
    """CLI dry-run: report symbol resolvability for each given .so, no calling.

    Lets you validate on the cluster against an INSTRUMENTED lib before spending
    a CI run:  python -m tensorrt_llm._torch.bolt_profiling <inst-lib.so> ...
    (dlsym needs the lib loadable; the nm-offset path only reads the file.)
    """
    rc = 0
    for path in paths:
        offset = _symbol_offset(path)
        dlsym_ok = False
        try:
            h = ctypes.CDLL(path)  # dlopen (loads it) to test dlsym visibility
            dlsym_ok = hasattr(h, CLEAR_SYMBOL)
        except OSError as e:
            dlsym_ok = f"load-failed: {e}"
        status = "OK" if (offset is not None or dlsym_ok is True) else "UNRESOLVABLE"
        if status != "OK":
            rc = 1
        print(
            f"[bolt-selfcheck] {status}  {path}\n"
            f"    nm offset : {'0x%x' % offset if offset is not None else 'NOT FOUND'}\n"
            f"    dlsym     : {dlsym_ok}"
        )
    return rc


if __name__ == "__main__":
    import sys

    sys.exit(_self_check(sys.argv[1:]))
