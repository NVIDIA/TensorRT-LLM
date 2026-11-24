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
"""Rank-private Triton and Inductor compile caches (opt-in: ``TLLM_K31_RANK_LOCAL_CACHE=1``).

Problem. With ``TRTLLM_CACHE_DIR`` set, every rank of every job that uses the same cache root compiles
Triton kernels into one ``<root>/triton`` directory on the shared file system, and the ranks of a node
share one Inductor directory. Triton writes an entry as a temp file plus ``os.replace``, which is atomic
on one host. A cold start, however, has 16-24 ranks on 4-6 nodes compile the same kernels within the same
seconds and replace the same file names from several file-system clients, and a rank that reads an entry
another node has just replaced can get an empty or short file: ``json.loads`` fails on an empty metadata
file in ``CompiledKernel.__init__``, or loading the cubin fails with ``CUDA_ERROR_INVALID_IMAGE`` (in
Triton's ``_init_handles`` or in Inductor's static launcher). Inductor also writes into the Triton
directory on the assumption that it has it to itself (``TritonBundler.read_and_emit`` and
``reload_cubin_path``, which rewrites a missing cubin in place).

Fix. The first process of each rank that imports ``tensorrt_llm`` points ``TRITON_CACHE_DIR`` and
``TORCHINDUCTOR_CACHE_DIR`` at directories of its own on node-local storage::

    ${TLLM_K31_RANK_LOCAL_CACHE_BASE:-/tmp/trtllm_rank_cache}/<SLURM_JOB_ID>.<SLURM_STEP_ID>/r<SLURM_PROCID>/{triton,inductor}

A process without a Slurm rank gets ``.../nojob/p<pid>`` (trtllm-llmapi-launch starts the trtllm-serve
frontend without the MPI and Slurm environment). Child processes, such as Inductor's compile workers,
inherit the directories through the environment; ``TLLM_K31_RANK_LOCAL_CACHE_DIR`` marks them as set up.

The shared Triton directory the rank would otherwise have used (``$TRITON_CACHE_DIR`` if set, else
``$TRTLLM_CACHE_DIR/triton``) stays a cache that warms up across jobs:

* Read-only seed. Before Triton runs, its complete entries are copied into the private directory. Each
  entry is checked first: group and metadata JSON parse, every child file is present and non-empty, the
  sections of every ``.cubin`` / ``.so`` lie inside the file, and its newest file is at least
  ``SEED_MIN_AGE_S`` (60) seconds old. Group child paths are rewritten to the private copy. An entry that
  fails a check is skipped and compiled locally. Triton keys entries by a hash of the kernel source,
  options and environment, so a seed from another job or overlay is safe.
  ``TLLM_K31_RANK_LOCAL_CACHE_SEED=<dir>`` seeds from another directory instead; ``=0`` turns seeding off.
* Write-once publish target. Local rank 0 of each step publishes complete new entries back to the shared
  directory every ``PUBLISH_EVERY_S`` (60) seconds for ``PUBLISH_FOR_S`` (2700) seconds. An entry is written
  into a temporary directory and renamed into place only if the key does not exist yet, so a published file
  is never replaced and no reader sees a half-written entry.

Inductor's directory is private and starts empty for each job; its Triton kernels come from the seeded
Triton directory. Other caches under ``TRTLLM_CACHE_DIR`` (FlashInfer, CuTe DSL, DeepGEMM, CUDA) are
untouched. With the switch unset or ``0`` this module does nothing.

The switch is an environment variable because it has to act before ``import torch``, long before any
model configuration exists.

Module scope is limited to the standard library: ``tensorrt_llm._bootstrap`` calls :func:`setup` before
``import torch``.
"""

import json
import os
import shutil
import socket
import stat
import struct
import sys
import threading
import time
import uuid

SWITCH_ENV = "TLLM_K31_RANK_LOCAL_CACHE"
BASE_ENV = "TLLM_K31_RANK_LOCAL_CACHE_BASE"
SEED_ENV = "TLLM_K31_RANK_LOCAL_CACHE_SEED"
DIR_ENV = "TLLM_K31_RANK_LOCAL_CACHE_DIR"

DEFAULT_BASE = "/tmp/trtllm_rank_cache"
# Seed only entries whose newest file is at least this old; publish every PUBLISH_EVERY_S seconds for the
# first PUBLISH_FOR_S seconds of the process.
SEED_MIN_AGE_S = 60.0
PUBLISH_EVERY_S = 60.0
PUBLISH_FOR_S = 2700.0
SEED_TIME_CAP_S = 300.0
PUBLISH_SETTLE_S = 5.0

# Our own temporary directories start with ".trtllm."; Triton's in-flight puts use "tmp.pid_*" and
# TritonBundler's "tmp.<uuid>". None of them is a cache entry.
_SKIP_PREFIXES = (".", "tmp.")
_ELF_SUFFIXES = (".cubin", ".so", ".hsaco")
_GROUP_PREFIX = "__grp__"
_SHT_NOBITS = 8

# read_entry() results other than a {name: bytes} dict.
YOUNG = "young"  # newest file younger than the minimum age; try again later
INCOMPLETE = "incomplete"  # a group child is missing or no group yet; try again later
INVALID = "invalid"  # empty file, unparsable JSON or truncated ELF; never copy

_publisher_thread = None


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get(SWITCH_ENV) or "0").strip() not in ("", "0")


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[rank-local-cache] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def rank_identity(env=None):
    """Return ``(namespace, name, local_id)`` naming this process's private directory."""
    env = os.environ if env is None else env
    procid = env.get("SLURM_PROCID", "")
    if procid.isdigit():
        job = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID") or "nojob"
        step = env.get("SLURM_STEP_ID") or env.get("SLURM_STEPID") or "nostep"
        return f"{job}.{step}", f"r{procid}", env.get("SLURM_LOCALID", "")
    return "nojob", f"p{os.getpid()}", ""


def shared_triton_dir(env=None):
    """The Triton directory this process would use without the switch (None: Triton's own default)."""
    env = os.environ if env is None else env
    explicit = env.get("TRITON_CACHE_DIR")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    root = env.get("TRTLLM_CACHE_DIR")
    if root:
        return os.path.join(os.path.abspath(os.path.expanduser(root)), "triton")
    return None


def _resolve(value, default):
    """Unset or "1": the default directory; "0": off; anything else: that directory."""
    if value is None or value.strip() in ("", "1"):
        return default
    if value.strip() == "0":
        return None
    return os.path.abspath(os.path.expanduser(value.strip()))


def elf_complete(data: bytes) -> bool:
    """True if the ELF header, both header tables and every section's bytes lie inside ``data``."""
    if len(data) < 52 or data[:4] != b"\x7fELF" or data[5] not in (1, 2):
        return False
    end = "<" if data[5] == 1 else ">"
    try:
        if data[4] == 2:  # ELF64
            if len(data) < 64:
                return False
            phoff, shoff = struct.unpack_from(end + "QQ", data, 32)
            phentsize, phnum, shentsize, shnum = struct.unpack_from(end + "HHHH", data, 54)
            sh_fmt = end + "IIQQQQ"  # name, type, flags, addr, offset, size
        elif data[4] == 1:  # ELF32
            phoff, shoff = struct.unpack_from(end + "II", data, 28)
            phentsize, phnum, shentsize, shnum = struct.unpack_from(end + "HHHH", data, 42)
            sh_fmt = end + "IIIIII"
        else:
            return False
        size = len(data)
        if phnum and phoff + phentsize * phnum > size:
            return False
        if shnum:
            if shentsize < struct.calcsize(sh_fmt) or shoff + shentsize * shnum > size:
                return False
            for i in range(shnum):
                _, sh_type, _, _, offset, length = struct.unpack_from(
                    sh_fmt, data, shoff + i * shentsize
                )
                if sh_type != _SHT_NOBITS and length and offset + length > size:
                    return False
        return True
    except struct.error:
        return False


def validate_entry(files: dict, dst_dir: str):
    """Check one entry's files and rewrite its group child paths to ``dst_dir``.

    Returns the files to write (a new dict), INCOMPLETE or INVALID.
    """
    if not files:
        return INVALID
    out = dict(files)
    groups = [n for n in files if n.startswith(_GROUP_PREFIX) and n.endswith(".json")]
    for name, data in files.items():
        if not data:
            return INVALID
        if name in groups:
            try:
                grp = json.loads(data)
            except ValueError:
                return INVALID
            children = grp.get("child_paths") if isinstance(grp, dict) else None
            if not isinstance(children, dict) or not children:
                return INVALID
            rewritten = {}
            for child, path in children.items():
                base = os.path.basename(str(path))
                if base not in files:
                    return INCOMPLETE
                rewritten[child] = os.path.join(dst_dir, base)
            out[name] = json.dumps({"child_paths": rewritten}).encode()
        elif name.endswith(".json"):
            try:
                json.loads(data)
            except ValueError:
                return INVALID
        elif name.endswith(_ELF_SUFFIXES):
            if not elf_complete(data):
                return INVALID
    # A kernel entry is usable only once Triton has written its group file (last). An entry made only of
    # ELF objects (Triton's cuda_utils / launcher modules) has no group.
    if not groups and any(not n.endswith(_ELF_SUFFIXES) for n in files):
        return INCOMPLETE
    return out


def read_entry(src_dir: str, dst_dir: str, min_age_s: float = 0.0, now=None):
    """Read and validate the entry in ``src_dir`` for installation at ``dst_dir``.

    Returns ``{name: bytes}``, YOUNG, INCOMPLETE or INVALID.
    """
    files = {}
    newest = 0.0
    for name in os.listdir(src_dir):
        if name.startswith(_SKIP_PREFIXES):
            continue
        path = os.path.join(src_dir, name)
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            continue
        newest = max(newest, st.st_mtime, st.st_ctime)
        with open(path, "rb") as f:
            files[name] = f.read()
    if not files:
        return INCOMPLETE
    if min_age_s > 0 and (time.time() if now is None else now) - newest < min_age_s:
        return YOUNG
    return validate_entry(files, dst_dir)


def install_entry(files: dict, dst_root: str, key: str) -> bool:
    """Write ``files`` as ``dst_root/key`` with one directory rename; False if the key already exists."""
    tmp = os.path.join(dst_root, f".trtllm.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}")
    os.makedirs(tmp)
    try:
        for name, data in files.items():
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
        try:
            os.rename(tmp, os.path.join(dst_root, key))
        except OSError:
            return False  # somebody installed this key first (or left an empty directory there)
        return True
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def seed(
    src_root: str,
    dst_root: str,
    min_age_s: float = SEED_MIN_AGE_S,
    time_cap_s: float = SEED_TIME_CAP_S,
    now=None,
) -> dict:
    """Copy the complete, settled, valid entries of ``src_root`` that ``dst_root`` lacks."""
    stats = {"copied": 0, "present": 0, YOUNG: 0, INCOMPLETE: 0, INVALID: 0}
    try:
        keys = sorted(os.listdir(src_root))
    except OSError:
        stats["missing"] = 1
        return stats
    now = time.time() if now is None else now
    t0 = time.monotonic()
    for key in keys:
        if key.startswith(_SKIP_PREFIXES):
            continue
        src = os.path.join(src_root, key)
        dst = os.path.join(dst_root, key)
        try:
            if not os.path.isdir(src):
                continue
            if os.path.exists(dst):
                stats["present"] += 1
                continue
            entry = read_entry(src, dst, min_age_s=min_age_s, now=now)
            if isinstance(entry, dict):
                if install_entry(entry, dst_root, key):
                    stats["copied"] += 1
                else:
                    stats["present"] += 1
            else:
                stats[entry] += 1
        except OSError:
            stats[INVALID] += 1  # vanished or unreadable while we looked at it
        if time.monotonic() - t0 > time_cap_s:
            stats["time_capped"] = 1
            break
    return stats


def publish_pass(
    local_root: str, target_root: str, done: set, settle_s: float = PUBLISH_SETTLE_S, now=None
) -> int:
    """Publish the complete local entries ``target_root`` lacks; returns how many this pass added."""
    try:
        keys = os.listdir(local_root)
    except OSError:
        return 0
    os.makedirs(target_root, exist_ok=True)
    now = time.time() if now is None else now
    published = 0
    for key in keys:
        if key in done or key.startswith(_SKIP_PREFIXES):
            continue
        src = os.path.join(local_root, key)
        dst = os.path.join(target_root, key)
        try:
            if not os.path.isdir(src):
                continue
            if os.path.exists(dst):
                done.add(key)
                continue
            entry = read_entry(src, dst, min_age_s=settle_s, now=now)
            if entry in (YOUNG, INCOMPLETE):
                continue  # a put in flight; look again next pass
            if isinstance(entry, dict) and install_entry(entry, target_root, key):
                published += 1
        except OSError:
            continue
        done.add(key)
    return published


def _start_publisher(local_root: str, target_root: str, every_s: float, for_s: float) -> None:
    global _publisher_thread
    if _publisher_thread is not None:
        return

    def run():
        done = set()
        total = 0
        deadline = time.monotonic() + for_s
        while True:
            time.sleep(every_s)
            try:
                n = publish_pass(local_root, target_root, done, settle_s=PUBLISH_SETTLE_S)
            except Exception as exc:  # keep the rank alive whatever the shared file system does
                _log(f"publish pass to {target_root} failed: {exc!r}")
                n = 0
            if n:
                total += n
                _log(f"published {n} new entries to {target_root} ({total} so far)")
            if time.monotonic() >= deadline:
                break
        _log(f"publisher finished: {total} entries published to {target_root}")

    _publisher_thread = threading.Thread(target=run, name="trtllm-rank-cache-publish", daemon=True)
    _publisher_thread.start()


def setup() -> None:
    """Point this rank's Triton and Inductor caches at private node-local directories (when switched on).

    Called first thing in ``tensorrt_llm._bootstrap._prepare_environment()``, before ``import torch``.
    Never raises: on an error the shared caches stay in place and an ERROR line is printed.
    """
    if os.environ.get(DIR_ENV) or not enabled():
        return  # inherited from the process of this rank that set it up, or off
    t0 = time.monotonic()
    try:
        namespace, name, local_id = rank_identity()
        base = os.path.abspath(os.path.expanduser(os.environ.get(BASE_ENV) or DEFAULT_BASE))
        root = os.path.join(base, namespace, name)
        triton_dir = os.path.join(root, "triton")
        inductor_dir = os.path.join(root, "inductor")
        os.makedirs(triton_dir, exist_ok=True)
        os.makedirs(inductor_dir, exist_ok=True)
    except Exception as exc:
        _log(
            f"ERROR: cannot create the private cache directories ({exc!r}); "
            "keeping the shared Triton/Inductor caches"
        )
        return

    shared = shared_triton_dir()
    seed_dir = _resolve(os.environ.get(SEED_ENV), shared)
    publish_dir = shared
    if seed_dir == triton_dir:
        seed_dir = None
    if publish_dir == triton_dir:
        publish_dir = None

    seed_msg = "off"
    if seed_dir:
        try:
            stats = seed(seed_dir, triton_dir, min_age_s=SEED_MIN_AGE_S)
            seed_msg = f"{seed_dir} {stats}"
        except Exception as exc:
            seed_msg = f"{seed_dir} failed ({exc!r}); starting cold"

    os.environ["TRITON_CACHE_DIR"] = triton_dir
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
    os.environ[DIR_ENV] = root

    publish_msg = "off"
    if publish_dir:
        if local_id == "0":
            every_s, for_s = PUBLISH_EVERY_S, PUBLISH_FOR_S
            try:
                _start_publisher(triton_dir, publish_dir, every_s, for_s)
                publish_msg = f"{publish_dir} every {every_s:g}s for {for_s:g}s"
            except Exception as exc:
                publish_msg = f"{publish_dir} failed to start ({exc!r})"
        else:
            publish_msg = f"{publish_dir} by local rank 0"
    _log(
        f"{namespace}/{name} on {socket.gethostname()}: TRITON_CACHE_DIR={triton_dir} "
        f"TORCHINDUCTOR_CACHE_DIR={inductor_dir}; seed {seed_msg}; publish {publish_msg}; "
        f"{time.monotonic() - t0:.2f}s"
    )
