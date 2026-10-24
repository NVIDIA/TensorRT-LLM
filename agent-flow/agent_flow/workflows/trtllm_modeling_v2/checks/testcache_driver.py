#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a tree's pytest tests and record, per test, its result and every input it ran on.

Usage, in the tree under test with that tree's python (standard library and
pytest only, so it runs by path in any job)::

    <python> testcache_driver.py run --cache DIR [--repo TREE] [--selection FILE] [ITEM ...] [-- PYTEST ARGS]
    <python> testcache_driver.py session --cache DIR [--repo TREE] [--select FILE] -- PYTEST ARGS
    <python> testcache_driver.py env

``run`` takes test files and node ids (``ITEM``, or one per line in
``--selection``) and runs one pytest process per test file, each a
``session``, so a test's inputs are what its own file's process ran, not
what an earlier file left loaded. With nothing selected it still records the
job's environment. ``env`` prints the environment fingerprint and exits.

A session writes one JSON record to ``<cache>/sessions/``: the tree's commit
and whether its tracked files were clean, the environment, and per test its
outcome, duration and failure message, with its index in the run. What the
process read is recorded with the index of the test during which it was
first seen (``-1``: before the first test, while collecting), and a test's
inputs are everything first seen at or before its index. Within one process
a test can read state an earlier test left (a memoized result, a captured
graph), so cumulative is what is sound. Inputs are paths relative to the
tree, each tracked file with the git blob id of its content:

- ``loaded``: the tree's modules in ``sys.modules``;
- ``executed``: per file, the qualified names of the functions that started
  running (``sys.monitoring``, else a profile hook);
- ``read``: files opened for reading other than by the import system: data,
  and Python source read as text (``inspect.getsource`` under a JIT);
- ``listed``: directories listed other than by the import system,
  ``importlib.metadata`` or pytest walking to the files it was given;
- ``probed``: paths whose existence was asked (``os.stat``, ``os.lstat``,
  ``os.access``, so ``os.path.exists`` and ``Path.is_file`` too);
- ``imports``: module names the import system searched for that did not
  resolve inside the tree (an added file could now satisfy one);
- ``native``: the tree's shared libraries the process loaded.

A test from which on the process may run code it cannot see is **opaque**,
and its record is reused only at the very commit it ran on: from the first
process it starts that can run Python or build code (an interpreter, a
launcher, a compiler; ``git`` is not one), from the first process a pool
starts (``multiprocessing``, ``concurrent.futures``, an MPI pool, Ray), from
the first frame ``torch.compile`` compiled, in every rank of a multi-rank launch, and from the start when the
tree's own package was imported from outside the tree.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

#: Bumped when a record's meaning changes; the reader skips other versions.
RECORD_VERSION = 1
SESSIONS_DIRNAME = "sessions"
#: Files larger than this are recorded without a blob id, and so always read as changed.
MAX_HASH_BYTES = 256 << 20
#: Outcomes, the worse the higher: a test's is its worst phase's.
SEVERITY = {None: 0, "passed": 1, "skipped": 2, "xfailed": 2, "xpassed": 3, "failed": 4, "error": 5}

#: Environment variables that change what a test computes, by prefix.
ENV_PREFIXES = (
    "TLLM_",
    "TRTLLM_",
    "TRT_LLM_",
    "TORCH_",
    "PYTORCH_",
    "NCCL_",
    "CUDA_",
    "TRITON_",
    "FLASHINFER_",
    "HF_",
    "TRANSFORMERS_",
)
ENV_NAMES = ("LLM_MODELS_ROOT", "PYTHONHASHSEED")
#: ... less the ones that only say where something is kept or how it is reached.
ENV_SKIP_SUFFIXES = ("_DIR", "_PATH", "_HOME", "_FILE", "_LOG", "_PORT", "_ADDR", "_IFNAME")
ENV_SKIP_NAMES = ("CUDA_VISIBLE_DEVICES",)

#: A program that can run Python or build native code, by its name.
_RUNS_CODE = re.compile(
    r"^(?:python[\d.]*|pytest|py\.test|mpirun|mpiexec|orterun|prterun|srun|torchrun|"
    r"trtllm-[\w-]+|deepspeed|ray|accelerate|ninja|make|cmake|nvcc|gcc|g\+\+|cc|c\+\+|"
    r"clang|clang\+\+)$"
)
#: What separates the commands of a shell command line.
_SHELL_SEPARATORS = re.compile(r"[;&|()\n`]+|\$\(")
#: Words that run the command after them.
_PREFIXES = frozenset({"env", "exec", "nohup", "time", "command", "builtin"})
#: Programs that run the command line they are given: what they start is what counts.
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})


def starts_code(argv: list[str]) -> bool:
    """Whether the process ``argv`` starts can run Python or build code.

    Its program decides, not every word it is given: ``file -b
    /usr/bin/python3.12`` (which ``platform.architecture()`` runs) names an
    interpreter and runs none. ``env`` and its kin hand on to the program
    after their options and assignments, and a shell's ``-c`` script is read
    as a command line (:func:`starts_code_script`).
    """
    words = list(argv)
    while words and os.path.basename(words[0]) in _PREFIXES:
        words = words[1:]
        while words and (words[0].startswith("-") or "=" in words[0]):
            words = words[1:]
    if not words:
        return False
    name = os.path.basename(words[0])
    if _RUNS_CODE.match(name) or name.endswith(".py"):
        return True
    if name in _SHELLS and "-c" in words[1:]:
        script = words[words.index("-c", 1) + 1 :]
        return bool(script) and starts_code_script(script[0])
    return False


def starts_code_script(script: str) -> bool:
    """Whether a shell command line runs, as one of its commands, a program that can run code.

    Each command's program is its first word past variable assignments; an
    argument naming an interpreter runs nothing.
    """
    for command in _SHELL_SEPARATORS.split(script):
        words = [w for w in command.split() if w]
        while words and "=" in words[0] and not words[0].startswith("="):
            words = words[1:]
        if words and starts_code(words):
            return True
    return False


#: Out-of-tree functions that hand work to other processes, by file and qualified name:
#: every multiprocessing start (pools, ``ProcessPoolExecutor``, ``torch.multiprocessing``)
#: goes through ``BaseProcess.start``; MPI pools and Ray start their own.
_LAUNCHERS = (
    (re.compile(r"[/\\]multiprocessing[/\\]process\.py$"), frozenset({"BaseProcess.start"})),
    (
        re.compile(r"[/\\]mpi4py[/\\]futures[/\\]"),
        frozenset({"MPIPoolExecutor.__init__", "MPICommExecutor.__enter__"}),
    ),
    (
        re.compile(r"[/\\]ray[/\\]"),
        frozenset({"init", "RemoteFunction._remote", "ActorClass._remote"}),
    ),
)
#: Import-system functions that search for a module rather than run one.
_IMPORT_SEARCH = frozenset(
    {
        "_find_spec",
        "_find_spec_legacy",
        "find_spec",
        "_get_spec",
        "_path_hooks",
        "_path_importer_cache",
        "_fill_cache",
        "path_hook_for_FileFinder",
        "_search_registry",
        "_get_module_info",
        "__init__",
    }
)
#: ``importlib.metadata`` looking for installed distributions, which ``environment()`` holds.
_METADATA = re.compile(r"[/\\]importlib(?:[/\\]|_)metadata[/\\]")
#: Listings that decide nothing a test sees: the import system's (an import
#: that a new file could satisfy is in ``imports``) and pytest's own while it
#: walks to the test files it was given.
_NOT_A_LISTING = re.compile(r"^<frozen importlib|[/\\](?:_pytest|pluggy)[/\\]")
_RANK_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "WORLD_SIZE")
_SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache"}
_CO_OPTIMIZED = 0x1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def blob_id(data: bytes) -> str:
    """The git (SHA-1) blob id of ``data``: what ``git hash-object`` prints for it."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_blob_id(path: str) -> str | None:
    """The blob id of the file at ``path``; None when it is not a readable file or is too large."""
    try:
        if os.path.getsize(path) > MAX_HASH_BYTES:
            return None
        with open(path, "rb") as handle:
            return blob_id(handle.read())
    except OSError:
        return None


def _env_var_kept(name: str) -> bool:
    if name in ENV_SKIP_NAMES or name.endswith(ENV_SKIP_SUFFIXES):
        return name in ENV_NAMES
    return name in ENV_NAMES or name.startswith(ENV_PREFIXES)


def _gpus() -> tuple[list[str], str]:
    """The GPU models the driver lists, and its version; empty on a host without one."""
    models = []
    root = "/proc/driver/nvidia/gpus"
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        entries = []
    for entry in entries:
        try:
            with open(os.path.join(root, entry, "information"), encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("Model:"):
                        models.append(line.split(":", 1)[1].strip())
        except OSError:
            continue
    driver = ""
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as handle:
            found = re.search(r"Kernel Module\s+(?:for \w+\s+)?([\d.]+)", handle.readline())
            driver = found.group(1) if found else ""
    except OSError:
        pass
    return sorted(models), driver


def environment() -> dict:
    """What, besides the tree, decides a test's result: interpreter, packages, GPUs, settings."""
    import importlib.metadata as metadata
    import platform

    packages = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower()
        if name:
            packages.add(f"{name}=={dist.version}")
    models, driver = _gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = (
        len([d for d in visible.split(",") if d.strip()]) if visible is not None else len(models)
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "packages": sorted(packages),
        "gpus": models,
        "visible_gpus": count,
        "driver": driver,
        "vars": {k: v for k, v in sorted(os.environ.items()) if _env_var_kept(k)},
    }


def env_digest(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _rank() -> tuple[int, int]:
    """This process's rank and world size under a multi-rank launcher; (0, 1) otherwise."""
    for name in _RANK_SIZE_VARS:
        try:
            size = int(os.environ.get(name, "") or 0)
        except ValueError:
            continue
        if size > 1:
            for rank_name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "RANK"):
                try:
                    return int(os.environ.get(rank_name, "") or 0), size
                except ValueError:
                    continue
            return 0, size
    return 0, 1


def _git(repo: str, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def tree_state(repo: str) -> tuple[str | None, bool | None]:
    """The tree's commit and whether its tracked files match it (None: git cannot say)."""
    head = (_git(repo, "rev-parse", "--verify", "HEAD") or "").strip() or None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return head, (None if status is None else not status.strip())


# ----------------------------------------------------------------- recording
class Recorder:
    """What one process ran on, each input with the index of the test it was first seen in."""

    def __init__(self, repo: str, tracer: str = "auto") -> None:
        self.repo = os.path.abspath(repo)
        self.repo_real = os.path.realpath(repo)
        self.idx = -1
        self.loaded: dict[str, int] = {}
        self.executed: dict[str, dict[str, int]] = {}
        self.read: dict[str, int] = {}
        self.listed: dict[str, int] = {}
        self.probed: dict[str, int] = {}
        self.imports: dict[str, int] = {}
        self.native: dict[str, int] = {}
        self.opaque: tuple[int, str] | None = None
        self.tracer = ""
        self._requested_tracer = tracer
        self._files: dict[str, object] = {}
        self._modules_seen: set[str] = set()
        self._tls = threading.local()
        self._packages = self._top_packages()

    # -------------------------------------------------------------- paths
    def _top_packages(self) -> set[str]:
        try:
            names = os.listdir(self.repo)
        except OSError:
            return set()
        return {n for n in names if os.path.isfile(os.path.join(self.repo, n, "__init__.py"))}

    def rel(self, path) -> str | None:
        """``path`` relative to the tree, ``/``-separated; None outside it or in a cache directory."""
        try:
            text = os.fsdecode(path)
        except TypeError:
            return None
        if not text or text.startswith("<"):
            return None
        full = os.path.abspath(text)
        for root in (self.repo, self.repo_real):
            if full == root or full.startswith(root + os.sep):
                break
        else:
            full = os.path.realpath(full)
            if not (full == self.repo_real or full.startswith(self.repo_real + os.sep)):
                return None
            root = self.repo_real
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        if rel == "." or _SKIP_PARTS.intersection(rel.split("/")) or rel.endswith(".pyc"):
            return None
        return rel

    def latch(self, reason: str, idx: int | None = None) -> None:
        """Every test from ``idx`` (now, by default) on runs code this process cannot see."""
        at = self.idx if idx is None else idx
        if self.opaque is None or at < self.opaque[0]:
            self.opaque = (at, reason)

    # -------------------------------------------------------------- hooks
    def install(self) -> None:
        """Start recording: the audit hook, the existence probes and the function tracer."""
        rank, size = _rank()
        if size > 1:
            self.latch(f"runs as rank {rank} of {size}", -1)
        sys.addaudithook(self._audit)
        sys.meta_path.insert(0, _Searched(self))
        self._patch_probes()
        self._start_tracer()

    def _busy(self) -> bool:
        return getattr(self._tls, "busy", False)

    @contextlib.contextmanager
    def quiet(self):
        """Nothing the recorder itself does in here is recorded."""
        before = self._busy()
        self._tls.busy = True
        try:
            yield
        finally:
            self._tls.busy = before

    def _audit(self, event: str, args: tuple) -> None:
        handler = _AUDIT.get(event)
        if handler is None or self._busy():
            return
        self._tls.busy = True
        try:
            handler(self, event, args)
        except Exception:  # a recorder must never break the test it watches
            pass
        finally:
            self._tls.busy = False

    @staticmethod
    def _caller() -> tuple[str, bool]:
        """Who raised the event being handled: the caller's file, and whether a module is being looked for.

        The import system searches (``sys.path`` hooks, finders, their
        directory caches) beneath ``_find_spec`` and its kin, and a module it
        executes runs beneath ``_call_with_frames_removed``: the first frozen
        import frame above the caller says which. ``importlib.metadata``
        looks for installed distributions, which the environment holds.
        """
        try:
            frame = sys._getframe(3)
        except ValueError:
            return "", False
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        caller = frame.f_code.co_filename if frame is not None else ""
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename.startswith(("<frozen importlib", "<frozen zipimport")):
                return caller, frame.f_code.co_name in _IMPORT_SEARCH
            if _METADATA.search(filename):
                return caller, True
            frame = frame.f_back
        return caller, False

    def _on_open(self, _event, args) -> None:
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int):
            return
        if isinstance(mode, str):
            if "w" in mode or "x" in mode:
                return
        elif isinstance(flags, int):
            if flags & os.O_ACCMODE == os.O_WRONLY or flags & getattr(os, "O_TRUNC", 0):
                return
        rel = self.rel(path)
        if rel is None:
            return
        caller, searching = self._caller()
        if searching or caller.startswith(("<frozen importlib", "<frozen zipimport")):
            return
        self.read.setdefault(rel, self.idx)

    def _on_listing(self, _event, args) -> None:
        path = args[0] if args else "."
        if isinstance(path, int):
            return
        rel = self.rel("." if path is None else path)
        if rel is None:
            full = os.path.abspath(os.fsdecode("." if path is None else path))
            rel = "" if full in (self.repo, self.repo_real) else None
        if rel is None:
            return
        caller, searching = self._caller()
        if searching or _NOT_A_LISTING.search(caller):
            return
        self.listed.setdefault(rel, self.idx)

    def _on_process(self, event, args) -> None:
        if event in ("os.fork", "os.forkpty"):
            self.latch(f"forks ({event})")
            return
        if event == "os.system":
            words = [str(args[0]) if args else ""]
            runs = starts_code_script(words[0])
        else:
            executable, argv = (args[1], args[2]) if event == "os.spawn" else (args[0], args[1])
            if isinstance(argv, (str, bytes)):
                argv = [argv]
            words = [
                os.fsdecode(w) for w in (argv or ()) if isinstance(w, (str, bytes, os.PathLike))
            ]
            if executable is not None:
                words = [os.fsdecode(executable), *words[1:]]
            runs = starts_code(words)
        if runs:
            self.latch(f"starts `{' '.join(words)[:80]}`")

    def _on_dlopen(self, _event, args) -> None:
        rel = self.rel(args[0]) if args else None
        if rel is not None:
            self.native.setdefault(rel, self.idx)

    def probe(self, path) -> None:
        if self._busy():
            return
        self._tls.busy = True
        try:
            if not isinstance(path, int):
                rel = self.rel(path)
                if rel is not None and not self._caller()[1]:
                    self.probed.setdefault(rel, self.idx)
        except Exception:  # never the test's failure
            pass
        finally:
            self._tls.busy = False

    def _patch_probes(self) -> None:
        for name in ("stat", "lstat", "access"):
            real = getattr(os, name)

            def wrapper(path, *args, _real=real, **kwargs):
                if kwargs.get("dir_fd") is None:
                    self.probe(path)
                return _real(path, *args, **kwargs)

            wrapper.__name__ = name
            wrapper.__wrapped__ = real
            for support in ("supports_dir_fd", "supports_fd", "supports_follow_symlinks"):
                group = getattr(os, support, None)
                if isinstance(group, set) and real in group:
                    group.add(wrapper)
            setattr(os, name, wrapper)

    # ------------------------------------------------------------ tracer
    def _classify(self, filename: str):
        """The tree-relative path of a code object's file, a launcher's names, or None."""
        self._tls.busy = True
        try:
            rel = self.rel(filename)
            if rel is None:
                for pattern, names in _LAUNCHERS:
                    if pattern.search(filename or ""):
                        return names
            return rel
        finally:
            self._tls.busy = False

    def _code_started(self, code) -> bool:
        """Record ``code`` starting; False while the recorder itself is running."""
        if self._busy():
            return False
        filename = code.co_filename
        where = self._files.get(filename, _UNSEEN)
        if where is _UNSEEN:
            where = self._files[filename] = self._classify(filename)
        if isinstance(where, frozenset):
            name = getattr(code, "co_qualname", code.co_name)
            if name in where or code.co_name in where:
                self.latch(f"hands work to other processes ({name})")
        elif where is not None and code.co_flags & _CO_OPTIMIZED:
            name = getattr(code, "co_qualname", None) or f"@{code.co_firstlineno}:{code.co_name}"
            self.executed.setdefault(where, {}).setdefault(name, self.idx)
        elif where is not None and code.co_name == "<module>":
            # Seen here as well as in sys.modules: a test may drop a module it imported.
            self.loaded.setdefault(where, self.idx)
        return True

    def _start_tracer(self) -> None:
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is not None and self._requested_tracer in ("auto", "monitoring"):
            for tool in (3, 4, monitoring.PROFILER_ID, monitoring.OPTIMIZER_ID):
                try:
                    monitoring.use_tool_id(tool, "trtllm-modeling-v2 testcache")
                except ValueError:
                    continue
                disable = monitoring.DISABLE

                def started(code, _offset):
                    return disable if self._code_started(code) else None

                monitoring.register_callback(tool, monitoring.events.PY_START, started)
                monitoring.set_events(tool, monitoring.events.PY_START)
                self.tracer = "monitoring"
                return
        seen: set = set()

        def profile(frame, event, _arg):
            if event == "call":
                code = frame.f_code
                if code not in seen and self._code_started(code):
                    seen.add(code)

        threading.setprofile(profile)
        everywhere = getattr(threading, "setprofile_all_threads", None)
        if everywhere is not None:
            everywhere(profile)
        sys.setprofile(profile)
        self.tracer = "profile"

    # ------------------------------------------------------------- scans
    def scan(self) -> None:
        """What the test that just ended left: new modules, shared libraries, compiled frames."""
        self._tls.busy = True
        try:
            for name, module in list(sys.modules.items()):
                if name in self._modules_seen:
                    continue
                self._modules_seen.add(name)
                where = getattr(module, "__file__", None)
                if not where:
                    continue
                rel = self.rel(where)
                if rel is None:
                    top = name.partition(".")[0]
                    if top in self._packages and name == top:
                        self.latch(f"imported `{top}` from outside the tree ({where})", -1)
                    continue
                if rel.endswith((".so", ".pyd")) or ".so." in rel:
                    self.native.setdefault(rel, self.idx)
                else:
                    self.loaded.setdefault(rel, self.idx)
            self._scan_maps()
            dynamo = sys.modules.get("torch._dynamo.utils")
            counters = getattr(dynamo, "counters", None) if dynamo is not None else None
            if counters and sum((counters.get("frames") or {}).values()):
                self.latch("torch.compile compiled frames")
        except Exception:  # never the test's failure
            pass
        finally:
            self._tls.busy = False

    def _scan_maps(self) -> None:
        try:
            with open("/proc/self/maps", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        for line in lines:
            parts = line.split(None, 5)
            if len(parts) == 6 and (parts[5].rstrip().endswith(".so") or ".so." in parts[5]):
                rel = self.rel(parts[5].strip())
                if rel is not None:
                    self.native.setdefault(rel, self.idx)

    def unresolved_imports(self) -> dict[str, int]:
        """The searched-for names that did not resolve inside the tree."""
        out = {}
        for name, idx in self.imports.items():
            module = sys.modules.get(name)
            where = getattr(module, "__file__", None) if module is not None else None
            if where and self.rel(where) is not None:
                continue
            out[name] = idx
        return out

    def deps(self) -> dict:
        """The record's ``deps``: each input with its first index and, for a file, its blob id."""
        self._tls.busy = True
        try:
            files = set(self.loaded) | set(self.executed) | set(self.read)
            blobs = {rel: file_blob_id(os.path.join(self.repo, rel)) for rel in sorted(files)}
            native = {}
            for rel, idx in self.native.items():
                native[rel] = {"idx": idx, "blob": file_blob_id(os.path.join(self.repo, rel))}
            return {
                "blobs": blobs,
                "loaded": dict(sorted(self.loaded.items())),
                "executed": {k: dict(sorted(v.items())) for k, v in sorted(self.executed.items())},
                "read": dict(sorted(self.read.items())),
                "listed": dict(sorted(self.listed.items())),
                "probed": dict(sorted(self.probed.items())),
                "imports": dict(sorted(self.unresolved_imports().items())),
                "native": native,
            }
        finally:
            self._tls.busy = False

    def sys_path(self) -> list[str]:
        """The directories on ``sys.path`` inside the tree, ``""`` for its root."""
        out = []
        with self.quiet():
            for entry in sys.path:
                full = os.path.abspath(entry or ".")
                if not os.path.isdir(full):
                    continue
                rel = "" if full in (self.repo, self.repo_real) else self.rel(full)
                if rel is not None and rel not in out:
                    out.append(rel)
        return out


class _Searched:
    """A meta path finder that finds nothing: it records each module name the import system looks for.

    It sees ``importlib.import_module`` as well as the import statement, and
    only names not already in ``sys.modules``.
    """

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def find_spec(self, name, path=None, target=None):
        self.recorder.imports.setdefault(name, self.recorder.idx)
        return None


_UNSEEN = object()

_AUDIT = {
    "open": Recorder._on_open,
    "os.listdir": Recorder._on_listing,
    "os.scandir": Recorder._on_listing,
    "subprocess.Popen": Recorder._on_process,
    "os.posix_spawn": Recorder._on_process,
    "os.spawn": Recorder._on_process,
    "os.exec": Recorder._on_process,
    "os.system": Recorder._on_process,
    "os.fork": Recorder._on_process,
    "os.forkpty": Recorder._on_process,
    "ctypes.dlopen": Recorder._on_dlopen,
}


# ------------------------------------------------------------------ session
def _outcome(previous: str | None, report) -> str:
    """A test's outcome so far, given one more of its phase reports."""
    if getattr(report, "wasxfail", None) is not None:
        now = "xfailed" if report.skipped else "xpassed"
    elif report.failed:
        now = "failed" if report.when == "call" else "error"
    elif report.skipped:
        now = "skipped"
    else:
        now = "passed" if report.when == "call" else None
    return now if SEVERITY.get(now, 0) > SEVERITY.get(previous, 0) else previous


def _message(report) -> str:
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    text = getattr(crash, "message", None) or getattr(report, "longreprtext", "") or ""
    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        text = str(report.longrepr[2])
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return (lines[0] if lines else "")[:400]


def _plugin(recorder: Recorder, record: dict, selected: set[str] | None):
    import pytest

    tests: dict[str, dict] = record["tests"]
    ids: dict[str, str] = {}
    root: list[str] = [os.getcwd()]

    def repo_id(path, nodeid: str) -> str:
        with recorder.quiet():
            rel = recorder.rel(str(path)) or nodeid.split("::", 1)[0]
        name = nodeid.split("::", 1)[1] if "::" in nodeid else ""
        return f"{rel}::{name}" if name else rel

    class RecorderPlugin:
        def pytest_configure(self, config):
            root[0] = str(config.rootpath)

        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            for item in items:
                ids[item.nodeid] = repo_id(item.path, item.nodeid)
                file, _, name = ids[item.nodeid].partition("::")
                record["collected"].setdefault(file, []).append(name)
            if selected is None:
                return
            files = {s for s in selected if "::" not in s}
            keep, drop = [], []
            for item in items:
                rid = ids[item.nodeid]
                (keep if rid in selected or rid.partition("::")[0] in files else drop).append(item)
            record["missing"] = sorted(
                s for s in selected if "::" in s and s not in set(ids.values())
            )
            if drop:
                config.hook.pytest_deselected(items=drop)
                items[:] = keep

        def pytest_collectreport(self, report):
            if report.failed:
                file = report.nodeid.split("::", 1)[0]
                rid = repo_id(os.path.join(root[0], file), report.nodeid) if file else "<session>"
                tests.setdefault(
                    rid,
                    {"index": -1, "outcome": "error", "duration": 0.0, "message": _message(report)},
                )

        def pytest_collection_finish(self, session):
            recorder.scan()

        def pytest_runtest_logstart(self, nodeid, location):
            rid = ids.get(nodeid, nodeid)
            recorder.idx = record["count"]
            record["count"] += 1
            tests[rid] = {"index": recorder.idx, "outcome": None, "duration": 0.0, "message": ""}

        def pytest_runtest_logreport(self, report):
            entry = tests.get(ids.get(report.nodeid, report.nodeid))
            if entry is None:
                return
            before = entry["outcome"]
            entry["outcome"] = _outcome(before, report)
            entry["duration"] = round(entry["duration"] + float(report.duration or 0.0), 3)
            if entry["outcome"] != before and entry["outcome"] not in ("passed", None):
                entry["message"] = _message(report)

        def pytest_runtest_logfinish(self, nodeid, location):
            recorder.scan()

    return RecorderPlugin()


def _write(cache: str, record: dict) -> str:
    directory = os.path.join(cache, SESSIONS_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = os.path.join(directory, f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}.json")
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=None, separators=(",", ":"))
    os.replace(partial, path)
    return path


def _read_selection(path: str | None, items: list[str]) -> list[str]:
    lines = list(items)
    if path:
        with open(path, encoding="utf-8") as handle:
            lines += handle.read().splitlines()
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]


def session(args: argparse.Namespace) -> int:
    """One pytest process, recorded: what each test ran on and how it ended."""
    repo = os.path.abspath(args.repo)
    commit, clean = tree_state(repo)
    if args.commit and commit is None:
        commit = args.commit
    env = environment()
    record = {
        "version": RECORD_VERSION,
        "commit": commit,
        "clean": clean,
        "repo": repo,
        "run_id": args.run_id or os.environ.get("TESTCACHE_RUN_ID") or uuid.uuid4().hex[:12],
        "python": sys.executable,
        "argv": list(args.pytest_args),
        "started": _now(),
        "env": env,
        "env_digest": env_digest(env),
        "tests": {},
        "collected": {},
        "missing": [],
        "count": 0,
    }
    selected = None
    if args.select:
        selected = set(_read_selection(args.select, []))
    # As `python -m pytest` from the tree: the tree first on sys.path, this
    # script's directory nowhere on it.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, repo)
    os.chdir(repo)
    recorder = Recorder(repo, tracer=args.tracer)
    recorder.install()
    import pytest

    code = int(pytest.main(list(args.pytest_args), plugins=[_plugin(recorder, record, selected)]))
    recorder.scan()
    record["exit_code"] = code
    record["ended"] = _now()
    record["tracer"] = recorder.tracer
    record["deps"] = recorder.deps()
    record["sys_path"] = recorder.sys_path()
    record["opaque"] = (
        None
        if recorder.opaque is None
        else {"idx": recorder.opaque[0], "reason": recorder.opaque[1]}
    )
    path = _write(args.cache, record)
    print(
        f"testcache: recorded {len(record['tests'])} test(s) at {commit or 'an unknown commit'} -> {path}"
    )
    return code


def _group(selection: list[str]) -> dict[str, list[str]]:
    """Selection lines by test file, in order; a bare file selects the whole file."""
    groups: dict[str, list[str]] = {}
    for item in selection:
        file = item.split("::", 1)[0]
        groups.setdefault(file, [])
        if item not in groups[file]:
            groups[file].append(item)
    return groups


def run(args: argparse.Namespace) -> int:
    """One recorded session per test file of the selection, in order."""
    repo = os.path.abspath(args.repo)
    selection = _read_selection(args.selection, args.items)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    groups = _group(selection)
    if not groups:
        env = environment()
        commit, clean = tree_state(repo)
        record = {
            "version": RECORD_VERSION,
            "commit": commit or args.commit,
            "clean": clean,
            "repo": repo,
            "run_id": run_id,
            "python": sys.executable,
            "argv": [],
            "started": _now(),
            "ended": _now(),
            "env": env,
            "env_digest": env_digest(env),
            "tests": {},
            "collected": {},
            "missing": [],
            "count": 0,
            "exit_code": 5,
            "deps": {},
            "opaque": None,
        }
        print(
            f"testcache: nothing selected; recorded the environment -> {_write(args.cache, record)}"
        )
        return 0
    worst = 0
    env = dict(os.environ, TESTCACHE_RUN_ID=run_id)
    for file, items in groups.items():
        whole = file in items
        select = None
        command = [sys.executable, os.path.abspath(__file__), "session", "--cache", args.cache]
        command += ["--repo", repo, "--run-id", run_id, "--tracer", args.tracer]
        if args.commit:
            command += ["--commit", args.commit]
        if not whole:
            select = os.path.join(args.cache, f".select-{os.getpid()}-{uuid.uuid4().hex[:6]}")
            os.makedirs(args.cache, exist_ok=True)
            with open(select, "w", encoding="utf-8") as handle:
                handle.write("\n".join(items) + "\n")
            command += ["--select", select]
        command += ["--", *args.pytest_args, file]
        print(
            f"testcache: {file}: {'whole file' if whole else f'{len(items)} test(s)'}", flush=True
        )
        started = time.monotonic()
        try:
            code = subprocess.run(command, cwd=repo, env=env, check=False).returncode
        finally:
            if select:
                try:
                    os.unlink(select)
                except OSError:
                    pass
        print(
            f"testcache: {file}: pytest exit {code} in {time.monotonic() - started:.0f}s",
            flush=True,
        )
        # 0 all passed, 1 some failed, 5 nothing collected: recorded either way.
        if code not in (0, 1, 5):
            worst = 2
    return worst


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="testcache_driver.py", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "session"):
        p = sub.add_parser(name)
        p.add_argument("--cache", required=True, help="The test cache directory.")
        p.add_argument("--repo", default=".", help="The tree under test (default: here).")
        p.add_argument("--commit", default="", help="The tree's commit, where git cannot say.")
        p.add_argument("--run-id", default="", help="Groups the sessions of one launch.")
        p.add_argument(
            "--tracer",
            choices=("auto", "monitoring", "profile"),
            default="auto",
            help="How function starts are seen: sys.monitoring where there is one.",
        )
        if name == "run":
            p.add_argument("--selection", default="", help="A file of test files and node ids.")
            p.add_argument("items", nargs="*", help="Test files and node ids.")
        else:
            p.add_argument("--select", default="", help="Keep only these node ids.")
    sub.add_parser("env")
    if "--" in argv:
        at = argv.index("--")
        args = parser.parse_args(argv[:at])
        args.pytest_args = argv[at + 1 :]
    else:
        args = parser.parse_args(argv)
        args.pytest_args = []
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    if getattr(args, "cache", None):
        args.cache = os.path.abspath(args.cache)  # a session runs in the tree
    if args.action == "env":
        env = environment()
        print(json.dumps({"env": env, "env_digest": env_digest(env)}, indent=1, sort_keys=True))
        return 0
    if args.action == "session":
        return session(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
