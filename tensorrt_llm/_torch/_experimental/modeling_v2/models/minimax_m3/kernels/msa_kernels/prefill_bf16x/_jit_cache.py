# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Content-validated cache of JIT-built native libraries.

Layout, under a placement-only root (``MINFER_FMHA_CACHE_DIR``, default
``~/.cache/minfer/fmha_sm100``)::

    <root>/v2/<toolchain>/            sha256 of the toolchain manifest (nvcc, CUDA, host compiler,
      manifest.json                   TVM-FFI): another toolchain is another namespace
      <name>-<recipe>/                one directory per build recipe (flags, defines, parameters)
        build.ninja, *.cu, *.o        the build tree, reused by incremental rebuilds
        <name>-<inputs>.so            immutable libraries, named by the inputs they were built from
        entry.json                    the recipe and one record per library: every file the
                                      compiler read, with its content hash

A library is served while every file of its record still has the recorded content. The files
are what the compiler read (``ninja -t deps``) plus the generator's templates, minus the
toolchain's own headers, which the namespace accounts for. Editing a header therefore rebuilds
only the libraries that include it, and going back to earlier sources finds the library built
from them (an entry keeps its last ``MAX_RECORDS`` records). Contents decide, not timestamps,
so a checkout or an install that keeps mtimes cannot serve a stale kernel.

Lookups take no lock and are memoized per process. A miss takes the recipe's file lock, deletes
the objects whose inputs changed, rebuilds, and publishes the record last with an atomic
``os.replace``, so a crashed build is a miss, never a corrupt hit. Unreadable, foreign or
mismatched entries are misses, never errors.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from functools import cached_property
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SCHEMA = 2
# Records kept per recipe: switching among this many versions of the sources rebuilds nothing.
MAX_RECORDS = 4
# Host system headers: part of the toolchain, so the namespace accounts for them.
_SYSTEM_PREFIXES = ("/usr/include/", "/usr/lib/", "/usr/lib64/", "/usr/local/include/", "/lib/")

_MISSING = object()
_file_hashes: dict[str, str | None] = {}
_hits: dict[Path, Path] = {}
_memo_lock = threading.Lock()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj, length: int) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()[:length]


def file_hash(path) -> str | None:
    """sha256 of a file's content (``None`` if unreadable), memoized per process."""
    path = os.fspath(path)
    digest = _file_hashes.get(path, _MISSING)
    if digest is _MISSING:
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            digest = None
        _file_hashes[path] = digest
    return digest


def clear_memo() -> None:
    """Forget memoized file hashes and hits, as a new process would (sources changed)."""
    with _memo_lock:
        _file_hashes.clear()
        _hits.clear()


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a same-directory tempfile and ``os.replace``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_if_changed(path: Path, text: str) -> None:
    """Rewrite a generated source only when it changes, so ninja keeps its object."""
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    atomic_write(path, text)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def toolchain_manifest(cuda_home: Path | str | None) -> dict:
    """The toolchain identity a namespace is keyed by. Reads files only; runs no compiler."""
    manifest: dict = {"schema": SCHEMA, "machine": platform.machine()}
    if cuda_home is not None:
        cuda_home = Path(os.path.realpath(cuda_home))
        manifest["nvcc"] = str(cuda_home / "bin" / "nvcc")
        try:
            version = json.loads((cuda_home / "version.json").read_text())["cuda"]["version"]
        except (OSError, KeyError, TypeError, ValueError):
            version = "unknown"
        manifest["cuda"] = version
    host = shutil.which("c++") or shutil.which("g++")
    manifest["host_compiler"] = os.path.realpath(host) if host else None
    try:
        import tvm_ffi

        manifest["tvm_ffi"] = getattr(tvm_ffi, "__version__", "unknown")
    except ImportError:
        manifest["tvm_ffi"] = None
    return manifest


class Namespace:
    """``<root>/v2/<toolchain>/``: where every recipe built by one toolchain lives."""

    def __init__(self, root: Path | str, cuda_home: Path | str | None, extra: dict | None = None):
        manifest = toolchain_manifest(cuda_home) | (extra or {})
        self.manifest = manifest
        self.path = Path(root) / f"v{SCHEMA}" / _digest(manifest, 16)
        covered = list(_SYSTEM_PREFIXES)
        if cuda_home is not None:
            covered.append(os.path.realpath(cuda_home) + os.sep)
        self.covered = tuple(covered)

    def ensure(self) -> None:
        manifest = self.path / "manifest.json"
        if not manifest.is_file():
            self.path.mkdir(parents=True, exist_ok=True)
            atomic_write(manifest, json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")


class Recipe:
    """One build recipe in a namespace and the libraries built from it.

    ``key`` holds everything that selects the build besides file contents (flags, defines,
    template parameters); ``templates`` are generator inputs the compiler never sees;
    ``covered`` lists further directory prefixes the key accounts for (e.g. the torch headers
    of a torch extension whose key holds the torch version).
    """

    def __init__(self, namespace: Namespace, name: str, key: dict, *,
                 templates=(), covered=()):
        self.namespace = namespace
        self.name = name
        self.recipe = {"name": name, "key": key}
        self.templates = tuple(str(Path(path)) for path in templates)
        self.covered = namespace.covered + tuple(covered)

    @cached_property
    def dir(self) -> Path:
        return self.namespace.path / f"{self.name}-{_digest(self.recipe, 12)}"

    @property
    def _entry_path(self) -> Path:
        return self.dir / "entry.json"

    def _read_entry(self) -> dict | None:
        try:
            entry = json.loads(self._entry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (not isinstance(entry, dict) or entry.get("schema") != SCHEMA
                or entry.get("recipe") != json.loads(_canonical(self.recipe))
                or not isinstance(entry.get("records"), list)):
            return None
        records = [record for record in entry["records"]
                   if isinstance(record, dict) and isinstance(record.get("library"), str)
                   and isinstance(record.get("inputs"), dict)]
        return {**entry, "records": records}

    def lookup(self) -> Path | None:
        """The library whose recorded inputs all match the current files, or ``None``."""
        hit = _hits.get(self.dir)
        if hit is not None:
            return hit
        entry = self._read_entry()
        if entry is None:
            return None
        for record in entry["records"]:
            if all(file_hash(path) == digest for path, digest in record["inputs"].items()):
                library = self.dir / record["library"]
                if library.is_file():
                    _hits[self.dir] = library
                    return library
        return None

    def build(self, builder: Callable[[Path], Path]) -> Path:
        """Return a valid library, running ``builder(build_dir)`` under the recipe's lock when
        no record matches. ``builder`` writes and runs the ninja build (compile rules with
        ``deps = gcc``) and returns the library it produced."""
        library = self.lookup()
        if library is not None:
            return library
        self.namespace.ensure()
        self.dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.dir / ".lock"):
            library = self.lookup()  # another process may have built it meanwhile
            if library is not None:
                return library
            self._drop_stale_objects()
            built = Path(builder(self.dir))
            return self._publish(built)

    def _drop_stale_objects(self) -> None:
        """Delete every object the latest record does not vouch for, so ninja, which compares
        only timestamps, rebuilds exactly the objects whose inputs changed."""
        entry = self._read_entry()
        latest = entry["records"][0] if entry and entry["records"] else {}
        objects = latest.get("objects") if isinstance(latest.get("objects"), dict) else {}
        for path in self.dir.rglob("*.o"):
            deps = objects.get(str(path.relative_to(self.dir)))
            if not isinstance(deps, dict) or any(
                    file_hash(dep) != digest for dep, digest in deps.items()):
                path.unlink()

    def _publish(self, built: Path) -> Path:
        objects = {}
        inputs = {}
        build_dir = str(self.dir) + os.sep
        for obj, deps in ninja_deps(self.dir).items():
            hashed = {}
            for dep in deps:
                if dep.startswith(self.covered):
                    continue
                digest = file_hash(dep)
                if digest is not None:
                    hashed[dep] = digest
            objects[obj] = hashed
            # Generated sources are covered by the templates and the key, so earlier records
            # stay checkable after the build tree has moved on.
            inputs.update((dep, digest) for dep, digest in hashed.items()
                          if not dep.startswith(build_dir))
        for template in self.templates:
            inputs[template] = file_hash(template)
        name = f"{self.name}-{_digest({'recipe': self.recipe, 'inputs': inputs}, 12)}{built.suffix}"
        library = self.dir / name
        # Rename, never rewrite: processes that mapped an earlier library keep their inode.
        os.replace(built, library)
        entry = self._read_entry()
        earlier = [record for record in (entry["records"] if entry else [])
                   if record["library"] != name]
        records = [{"library": name, "inputs": inputs, "objects": objects}]
        records += earlier[:MAX_RECORDS - 1]
        atomic_write(self._entry_path, _canonical(
            {"schema": SCHEMA, "recipe": self.recipe, "records": records}))
        kept = {record["library"] for record in records}
        for path in self.dir.glob(f"{self.name}-*{built.suffix}"):
            if path.name not in kept:
                with contextlib.suppress(OSError):
                    path.unlink()
        _hits[self.dir] = library
        logger.info("Published %s (%d inputs)", library, len(inputs))
        return library


def ninja_deps(build_dir: Path) -> dict[str, list[str]]:
    """``{object relative to build_dir: [absolute real paths it was compiled from]}``, from the
    deps ninja recorded for the rules declared with ``deps = gcc``."""
    result = subprocess.run(["ninja", "-t", "deps"], cwd=build_dir, capture_output=True,
                            text=True, check=True)
    deps: dict[str, list[str]] = {}
    current = None
    for line in result.stdout.splitlines():
        if not line.strip():
            current = None
        elif not line[0].isspace():
            target = os.path.realpath(os.path.join(build_dir, line.split(": #deps", 1)[0]))
            current = deps.setdefault(os.path.relpath(target, build_dir), [])
        elif current is not None:
            current.append(os.path.realpath(os.path.join(build_dir, line.strip())))
    return deps


def run_ninja(build_dir: Path, label: str, jobs: int = 1) -> None:
    result = subprocess.run(["ninja", f"-j{jobs}"], cwd=build_dir, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} compilation failed:\n"
                           f"stdout: {result.stdout}\nstderr: {result.stderr}")
