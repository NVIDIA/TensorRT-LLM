#!/usr/bin/env python3
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
"""Check that the shared libraries in a BOLTed wheel can still be loaded.

`llvm-bolt` exiting 0 says it produced an output file, not that the output file
is loadable. A profile that no longer lines up with the binary can yield an ELF
whose `.init_array` points at code the loader cannot reach, and the only symptom
is a SIGSEGV raised inside ld.so the first time anything dlopens it -- long after
the wheel has been uploaded.

So dlopen each library here, in a child process, while the un-BOLTed wheel is
still the one on disk.

Only death by signal is treated as a verdict. These libraries pull in torch and
the CUDA runtime, so an ordinary non-zero exit means "this container could not
resolve the dependencies" far more often than it means "BOLT broke it", and
failing on that would make the check useless wherever it runs outside a full
build image. A segfault, by contrast, is never ambiguous.

Exit codes:
  0  no library died on a signal (some may have been unverifiable)
  2  at least one library crashed the loader
  3  usage/extraction error
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import zipfile

# Anywhere in the wheel, not just tensorrt_llm/libs/: apply_bolt.process_wheel
# walks the extracted tree with rglob and optimizes every ELF that has a
# matching profile, so anything narrower would leave members modified but
# unverified.
LIB_GLOB = "**/*.so*"


def log(msg: str) -> None:
    print(f"[bolt-verify] {msg}", flush=True)


def extract_libs(wheel: str, dest: str) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        members = [n for n in zf.namelist() if ".so" in os.path.basename(n)]
        for n in members:
            zf.extract(n, dest)
    return sorted(glob.glob(os.path.join(dest, LIB_GLOB), recursive=True))


def _no_core_dump() -> None:
    # The crash this check exists to catch happens with libtensorrt_llm.so and
    # the CUDA runtime mapped in, so the core would be multi-gigabyte. Writing it
    # took longer than the rest of the check combined when this was first tried,
    # and nothing reads it: the signal number is the whole result.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def load_in_child(lib: str, libdir: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Siblings resolve out of the same directory the wheel ships them in; without
    # this every library with a DT_NEEDED on another one of ours is unverifiable.
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return subprocess.run(
        [sys.executable, "-c", "import ctypes, sys; ctypes.CDLL(sys.argv[1]); print('ok')", lib],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        preexec_fn=_no_core_dump,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", help="the .bolted wheel to check")
    args = ap.parse_args()

    if not os.path.isfile(args.wheel):
        log(f"ERROR: no such wheel: {args.wheel}")
        return 3

    with tempfile.TemporaryDirectory(prefix="bolt-verify-") as tmp:
        try:
            libs = extract_libs(args.wheel, tmp)
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"ERROR: could not read {args.wheel}: {exc}")
            return 3
        if not libs:
            # Nothing was BOLTed into place, so nothing to clear. The caller
            # already failed if apply did not run.
            log(f"no libraries matched {LIB_GLOB}; nothing to verify")
            return 0

        libdir = os.path.dirname(libs[0])
        crashed, unverified, loaded = [], [], []
        for lib in libs:
            name = os.path.basename(lib)
            try:
                proc = load_in_child(lib, libdir)
            except subprocess.TimeoutExpired:
                unverified.append((name, "timed out"))
                continue
            if proc.returncode < 0:
                crashed.append((name, f"killed by signal {-proc.returncode}"))
            elif proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()
                unverified.append((name, tail[-1] if tail else f"exit {proc.returncode}"))
            else:
                loaded.append(name)

        for name in loaded:
            log(f"  loaded {name}")
        for name, why in unverified:
            log(f"  UNVERIFIED {name}: {why}")
        for name, why in crashed:
            log(f"  CRASHED {name}: {why}")

        if crashed:
            log(
                f"ERROR: {len(crashed)} BOLTed library(ies) crash the dynamic loader. "
                "Refusing to publish this wheel."
            )
            return 2
        log(f"{len(loaded)} loaded, {len(unverified)} unverifiable, 0 crashed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
