# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declarative, file-scoped import paths for the test tree.

A file declares the directories its own imports may come from, rather than
mutating the process-global ``sys.path``::

    __extra_import_path__ = [".."]
    from utils.llm_data import llm_models_root

which replaces::

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils.llm_data import llm_models_root

Modules are imported as usual; the file level scoping is achieved *magically*,
with zero extra code.

Choosing between ``__extra_import_path__`` and ``sys.path``
--------------------------------------------------------

Use ``__extra_import_path__`` to satisfy an import written in the declaring file.
Use ``sys.path`` to influence imports performed by other code.

A ``__extra_import_path__`` declaration is scoped to the file that makes it, or,
in a ``conftest.py``, to the directory that file governs. It cannot, and is
not designed to change how a module resolves for code elsewhere in the process.
Consequently, it never takes precedence over a module that is already importable.
Therefore, the following remains ``sys.path`` work, though rather than being a
raw top-level modification, they should be fixtures instead, to restore what
they made to the environment:

* Overriding an installed distribution, such as preferring a source checkout of
  ``diffusers`` to an installed copy in site-packages.
* Providing a module that another component imports by name on the test's
  behalf, such as a plugin named in a configuration object and imported later
  by the library under test.
* Any adjustment intended to be visible process-wide.

Applying the rule also removes a common accident. A ``sys.path.insert(0, ...)``
added only to satisfy the inserting file's own import silently reorders module
resolution for everything else in the process; a declaration cannot.


Declaring a path
----------------

Assign a list of paths at module level::

    __extra_import_path__ = [".", "..", "~/examples/auto_deploy"]

Each entry is one of the following:

* ``.``, ``..``, ``sub/dir`` -- relative to the declaring file's own directory,
  so the declaration remains correct if the file moves with its neighbors.
* ``~/...`` -- relative to the PROJECT root, that is, the repository checkout.
  This is **not** the user's home directory; ``os.path.expanduser`` is
  deliberately not applied. Use this form for directories in a fixed location
  in the tree, such as ``~/examples``.
* An absolute path, used unchanged.

A bare string is rejected. Always use a list, even for a single entry.

The value is read from the live module namespace and may therefore be
computed::

    __extra_import_path__ = [os.environ["DIFFUSERS_MAIN_PATH"] + "/src"]


Where to declare
----------------

In a test file or helper, the declaration applies to that file alone. Place it
above the imports it serves: it takes effect only for imports executed after
the assignment, exactly as the ``sys.path`` line it replaces did.

In a ``conftest.py``, the declaration applies to every file in that directory
and below, and is read once when the conftest finishes loading. Mutating
``__extra_import_path__`` afterward, for instance, by appending to it from a
fixture, has no effect. Declare it once, at module level.

A conftest may import helper modules of its own. Those resolve while the
conftest is still executing, hence before it has been registered, so the finder
falls back to matching ``sys.modules`` by file in order to read the declaration
mid-flight. Place the declaration above such an import: it applies only to
imports that run after it, the same rule that governs a test file.

Declarations must be placed at or below the pytest rootdir, that is, the
directory containing ``pytest.ini``. Conftest files above that point are not
 consulted because pytest does not load them either. In this repository the
rootdirs are ``tests/unittest`` and ``tests/integration/defs``, so a
declaration in ``tests/conftest.py`` would have no effect.

A file's own declaration and the conftest declarations above it are merged,
nearest first.


Running a test file directly
----------------------------

``python tests/.../test_foo.py``, the ``if __name__ == "__main__"`` debugging
idiom, does not load the pytest plugin, so the file must enable the mechanism
itself. Importing this module does so::

    import test_common.magic_import  # noqa: F401 -- enables __extra_import_path__

    __extra_import_path__ = ["~/tests/unittest"]
    from utils.llm_data import llm_models_root

The import may be left in place permanently; the finder is inert until some
file declares a path. It requires ``tests/`` on ``PYTHONPATH``, which is the
entry ``tests/unittest/pytest.ini`` already lists under ``pythonpath``, so the
cost is one environment variable per checkout rather than one line per file.

Only the file's own declaration applies in this mode, since pytest is what
imports conftest files. Declare what a directly runnable file needs in that
file; under pytest the two sources merge, so nothing is lost by being explicit.

Note that this makes direct execution work where it commonly does not today. A
``sys.path.append(os.path.dirname(__file__) + "/..")`` line is frequently
satisfied not by the directory it names but by the rootdir pytest injects into
``sys.path``, so the file runs under pytest and fails standalone. A declaration
must name the directory that actually holds the module.


Design notes
------------

``MagicFinder`` is installed last in ``sys.meta_path``. It is therefore offered
only those names that every ordinary finder has already failed to resolve,
which is what guarantees it cannot shadow an installed distribution or anything
reachable from ``sys.path``. When it is consulted it:

1. Identifies the source file that requested the import, namely the nearest
   frame that belongs to neither importlib nor this module.
2. Collects ``__extra_import_path__`` from that file and from each ``conftest.py``
   above it, nearest first, stopping at the directory holding ``pytest.ini``.
3. Resolves the name against those directories only.
4. Records the name so that it can be removed from ``sys.modules`` once the
   importing file is finished, retaining the module object in a private cache.

Step 4 enables generic helper names to be used safely. Two directories may each
hold a ``utils.py``; because we made neither name persist in ``sys.modules``
beyond the scope of the file that requested it, the second file receives its own
``utils`` rather than whatever the first file happened to import. The private
cache keeps re-imports inexpensive and, more importantly, ensures module-level
side effects such as custom operator registration execute exactly once per
module.

Attribution is lexical: the requesting file is the nearest frame that ran the
import, and if that frame lies outside the test tree, the lookup is abandoned
rather than continued up the stack. See :func:`_requesting_source_file` for the
alternatives considered and why they were rejected.

Conftest declarations are supplied by the pytest plugin through
``pytest_plugin_registered`` rather than looked up in ``sys.modules``. The
latter cannot answer the question: under pytest's default ``prepend`` import
mode every conftest is imported under the bare name ``conftest``, so each
evicts its ancestors and only the most recently imported one remains
reachable. Outside pytest the finder falls back to matching ``sys.modules`` by
file. This is also why the walk stops at ``pytest.ini``: pytest loads conftest
files from its rootdir downwards and no further, so precisely the files in that
range are guaranteed to have been imported by the time a file below them runs
an import.

Assertion rewriting is preserved. Being last in ``sys.meta_path`` would
normally forfeit it, as pytest's rewriting hook runs first and rewrites only
what it locates itself; moving ahead of that hook would not help, since
whichever finder returns the spec determines the loader, and it would surrender
the guarantee above. The resolved directories are instead passed to that hook
explicitly, so a helper reached through ``__extra_import_path__`` is rewritten on
the same terms as one reached through ``sys.path``, including
``pytest.register_assert_rewrite``.
"""

import contextlib
import importlib
import os
import sys
from importlib.abc import Loader
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType

__all__ = ["MagicFinder"]

# The module-level name a file assigns to declare its own import directories.
_DECLARATION = "__extra_import_path__"

_SELF_FILE: str = os.path.realpath(__file__)
_IMPORTLIB_DIR: str = os.path.dirname(os.path.realpath(importlib.__file__))

# {frame co_filename -> is it importlib's or ours}. A frame's co_filename is
# whatever spelling the module was imported under, which need not match
# ``__file__`` normalised: an unnormalised sys.path entry such as
# ``tests/integration/defs/../..`` -- which is exactly what pytest.ini's
# ``pythonpath = ../../`` produces -- leaves ``/../..`` in co_filename while
# ``realpath`` collapses it. Comparing the two as strings then fails to
# recognise this module's own frame, the walk stops inside the finder, and
# every declaration silently stops applying. Resolved once per distinct
# spelling, since this runs for every import that reaches the finder.
_INTERNAL_FRAME: dict[str, bool] = {}


def _is_internal_frame(filename: str) -> bool:
    """True for importlib's frames and this module's, whatever the spelling."""
    known = _INTERNAL_FRAME.get(filename)
    if known is None:
        if filename.startswith("<frozen importlib"):
            known = True
        else:
            resolved = os.path.realpath(filename)
            known = resolved == _SELF_FILE or os.path.dirname(resolved) == _IMPORTLIB_DIR
        _INTERNAL_FRAME[filename] = known
    return known


def _resolve_entry(entry, base_dir: str, project_root: str) -> str:
    """Resolves one declared path to an absolute directory.

    ``~/x`` is project-root relative -- deliberately NOT ``expanduser``; test
    declarations never want the invoking user's home directory.
    """
    if entry == "~" or entry.startswith(("~/", "~" + os.sep)):
        return os.path.normpath(os.path.join(project_root, entry[1:].lstrip("/" + os.sep)))
    return os.path.normpath(os.path.join(base_dir, entry))


def _declared_paths(value, where: str) -> tuple[str, ...]:
    """Validates a declaration into raw, unresolved path strings."""
    if value is None:
        return ()
    # A bare str is rejected rather than wrapped: iterating it would silently
    # search one directory per character.
    if isinstance(value, (str, bytes, os.PathLike)) or not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{where}: {_DECLARATION} must be a list of paths, got {type(value).__name__}"
        )
    return tuple(os.fspath(entry) for entry in value)


def _module_file(module: ModuleType) -> str | None:
    """A module's file, resolved the same way requester paths are.

    ``realpath``, not ``abspath``: these values are matched against paths
    derived from :meth:`MagicFinder._source_in_test_tree`, which resolves
    symlinks. A checkout reached through a symlinked parent -- a CI workspace,
    typically -- otherwise yields two spellings of the same file, and every
    conftest lookup misses.
    """
    try:
        origin = module.__file__
    except AttributeError:
        return None
    return os.path.realpath(origin) if origin else None


class _CachedLoader(Loader):
    """Hands back an already-executed module instead of running it again."""

    def __init__(self, module: ModuleType):
        self._module = module

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:
        pass  # already executed on first import


class MagicFinder(PathFinder):
    """Resolves declared imports for files under the test tree.

    Enabled for a pytest run by ``test_common.magic_import_hooks``, and for a
    directly executed file by importing this module. Declaring
    ``__extra_import_path__`` is the entire user-facing interface; the methods
    below exist for the plugin and for tests of this module.
    """

    # Boundary of the test tree. An import requested from a file outside it is
    # never handled, which is what keeps declarations from reaching library
    # code. Also the backstop for the conftest walk, which normally stops
    # earlier, at the directory holding pytest.ini.
    test_root: str = str(Path(_SELF_FILE).resolve().parents[1])

    # Repository checkout root, against which a "~/..." entry is resolved.
    project_root: str = str(Path(_SELF_FILE).resolve().parents[2])

    # Modules resolved by this finder, keyed by absolute file path, retained
    # across purges so that each is executed at most once per interpreter.
    _magic_cache: dict[str, ModuleType] = {}

    # Submodules pulled in under a cached top-level package, keyed by the same
    # absolute file path: {"/dir/utils/__init__.py": {"utils.llm_data": <module>}}.
    # Restored together with their parent so `utils.llm_data` is not re-executed.
    _magic_submodules: dict[str, dict[str, ModuleType]] = {}

    # Names resolved by this finder for the source file currently being
    # handled, held until purge_magic_sys_modules ends their scope.
    _active_sys_modules: set[str] = set()

    # Resolved directories, keyed by (source file, its own raw declaration) so
    # that a declaration assigned part-way down a file invalidates the entry.
    _search_path_cache: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}

    # Resolved conftest declarations, keyed by the directory whose chain they
    # were collected for. Only filled once every conftest.py in that chain has
    # been seen, so a chain resolved while a conftest is still executing (and
    # therefore not yet registered) is not frozen in.
    _conftest_cache: dict[str, tuple[str, ...]] = {}

    # {absolute conftest.py path -> resolved directories}, snapshotted once the
    # conftest has finished loading. Recording the resolved value (rather than
    # keeping the module and re-reading it) is what makes a conftest
    # declaration immutable: see the note on mutation in the module docstring.
    #
    # It is filled from pytest_plugin_registered because sys.modules cannot
    # serve the lookup: in pytest's default "prepend" import mode every
    # conftest is imported as the bare name "conftest", so each one evicts its
    # ancestors and only the last import remains reachable there.
    _conftest_paths_by_file: dict[str, tuple[str, ...]] = {}

    # {raw frame filename -> resolved path under the test root, "" if outside}.
    _source_cache: dict[str, str] = {}

    # Lookups recorded inside a :meth:`trace` block, as
    # (name, requester, outcome). Off by default: recording everything would
    # cost on every failed import and would bury the interesting entries under
    # the optional-dependency probes library code makes constantly (``_wmi``,
    # ``msvcrt``, ``winreg``, and ~20 in a row from torch's trace_rules).
    _tracing: bool = False
    _trace: list = []

    # Reverse index into sys.modules ({absolute file -> module name}) plus the
    # sys.modules size it was built at, so it is only rebuilt when it can help.
    _file_to_module: dict[str, str] = {}
    _scanned_size: int = -1

    # pytest's AssertionRewritingHook, when the plugin has handed it over. It
    # locates modules only on the path it is given, so consulting it cannot
    # widen what is resolved; it determines only which loader executes them.
    rewrite_hook = None

    # Guards against re-entering find_spec during a resolution in progress.
    _resolving: bool = False

    @classmethod
    def install(cls) -> None:
        """Enables ``__extra_import_path__`` for the current interpreter.

        Idempotent, and called on import of this module, so a directly executed
        test file needs only to import it. See "Running a test file directly".

        The finder is appended rather than inserted: being last is what
        prevents it from shadowing anything already importable.
        """
        if cls not in sys.meta_path:
            sys.meta_path.append(cls)

    @classmethod
    def register_conftest(cls, module: ModuleType) -> None:
        """Records a conftest's declaration once that conftest has loaded.

        Called by the pytest plugin for every registered plugin; non-conftest
        modules are ignored. Reading the declaration here, once, is what fixes
        it at load time.
        """
        origin = _module_file(module)
        if origin and os.path.basename(origin) == "conftest.py":
            cls._snapshot_conftest(origin, module)

    @classmethod
    def _snapshot_conftest(cls, conftest: str, module: ModuleType) -> tuple[str, ...]:
        declared = _declared_paths(getattr(module, _DECLARATION, None), conftest)
        base = os.path.dirname(conftest)
        resolved = tuple(_resolve_entry(entry, base, cls.project_root) for entry in declared)
        previous = cls._conftest_paths_by_file.get(conftest)
        cls._conftest_paths_by_file[conftest] = resolved
        if previous is not None and previous != resolved:
            # An earlier lookup read this conftest while it was still
            # executing -- which happens when a conftest imports one of its
            # own helper modules, since that helper resolves before the
            # conftest finishes and gets registered. If the declaration has
            # changed since, everything derived from the earlier reading is
            # stale and would otherwise persist for the whole directory.
            cls._conftest_cache.clear()
            cls._search_path_cache.clear()
        return resolved

    @classmethod
    @contextlib.contextmanager
    def trace(cls):
        """Records the lookups made inside the block; yields the record.

        For answering "why did my declaration not apply?" -- wrap a retry of
        the failing import and read back what the finder was asked and what it
        decided::

            with MagicFinder.trace() as seen:
                try:
                    import the_missing_module
                except ImportError:
                    pass
            print(seen)  # [(name, requester, outcome), ...]
        """
        previous_trace, previous_tracing = cls._trace, cls._tracing
        cls._trace, cls._tracing = [], True
        try:
            # The caller keeps a reference, so the record stays readable after
            # the block restores the finder's own state.
            yield cls._trace
        finally:
            cls._trace, cls._tracing = previous_trace, previous_tracing

    @classmethod
    def _record(cls, fullname: str, requester, outcome: str) -> None:
        """Notes one lookup when tracing is on. A no-op otherwise."""
        if cls._tracing:
            cls._trace.append((fullname, requester, outcome))

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        # Submodules resolve through their parent package's __path__, which the
        # stock PathFinder handles; only top-level names are claimed here.
        if "." in fullname or cls._resolving:
            return None

        cls._resolving = True
        try:
            requester, namespace = _requesting_source_file()
            if requester is None:
                cls._record(fullname, None, "no-requesting-frame")
                return None
            search_paths = cls._search_paths(requester, namespace)
            if not search_paths:
                cls._record(
                    fullname,
                    requester,
                    "no-search-paths"
                    if cls._source_in_test_tree(requester)
                    else "requester-outside-test-root",
                )
                return None

            # Offer the name to pytest's assertion rewriter first, pointed at
            # the declared directories. It returns a spec only for modules it
            # would rewrite in any case (test files, conftest files, and
            # anything passed to pytest.register_assert_rewrite), so a helper
            # receives the same introspective assertion messages it would
            # receive when reached through sys.path.
            spec = None
            if cls.rewrite_hook is not None:
                spec = cls.rewrite_hook.find_spec(fullname, list(search_paths), target)
            if spec is None:
                spec = super().find_spec(fullname, list(search_paths), target)
            if spec is None:
                cls._record(fullname, requester, f"not-found-in={list(search_paths)}")
                return None
            cls._record(fullname, requester, f"resolved={spec.origin}")

            # Namespace packages have no origin and nothing to execute; track
            # them for purging, but there is no module object worth caching.
            #
            # realpath, matching _module_file: this reads the same cache that
            # purge_magic_sys_modules writes through that function. Under
            # abspath the two spellings disagree whenever a declared directory
            # is reached through a symlink, the lookup misses, and the module
            # is executed a second time -- exactly what the cache exists to
            # prevent for side effects such as custom operator registration.
            origin = os.path.realpath(spec.origin) if spec.origin else None
            cached = cls._magic_cache.get(origin) if origin else None
            if cached is not None and getattr(cached, "__name__", None) == fullname:
                for name, module in cls._magic_submodules.get(origin, {}).items():
                    sys.modules.setdefault(name, module)
                spec.loader = _CachedLoader(cached)

            # Registered before execution: if the module raises, the import
            # machinery drops it from sys.modules and purging simply finds
            # nothing to harvest.
            cls._active_sys_modules.add(fullname)
            return spec
        finally:
            cls._resolving = False

    @classmethod
    def purge_magic_sys_modules(cls):
        """Ends the scope of the declared imports resolved so far.

        Called by the pytest plugin once a file has finished importing and
        again after each test. This is what confines a generic module name such
        as ``utils`` to the file that requested it, so that a second file
        receives its own module rather than the first file's.

        Only names this finder resolved are removed, together with any
        submodules loaded beneath them. The module objects are retained in
        :attr:`_magic_cache`, so a later file importing the same path receives
        the same, already-executed module instead of re-running its top-level
        side effects.
        """
        for name in sorted(cls._active_sys_modules):
            prefix = name + "."
            subtree = {
                key: module
                for key, module in list(sys.modules.items())
                if key == name or key.startswith(prefix)
            }
            top = subtree.get(name)
            origin = _module_file(top) if top is not None else None
            if origin:
                cls._magic_cache[origin] = top
                cls._magic_submodules[origin] = {
                    key: module for key, module in subtree.items() if key != name
                }
            for key in subtree:
                sys.modules.pop(key, None)
        cls._active_sys_modules.clear()

    @classmethod
    def _search_paths(cls, source_file: str, namespace: dict) -> tuple[str, ...]:
        """Extra directories visible to ``source_file``, nearest declaration first."""
        source_file = cls._source_in_test_tree(source_file)
        if not source_file:
            return ()
        source = Path(source_file)

        # The importing file's own declaration comes straight off the frame
        # that ran the import, which is the one namespace always available.
        own = _declared_paths(namespace.get(_DECLARATION), source_file)
        key = (source_file, own)
        cached = cls._search_path_cache.get(key)
        if cached is not None:
            return cached

        directory = str(source.parent)
        collected = [_resolve_entry(entry, directory, cls.project_root) for entry in own]
        chain, complete = cls._conftest_paths(source.parent, skip=source_file)
        collected.extend(chain)
        # De-duplicate keeping order; drop entries that are not directories.
        result = tuple(dict.fromkeys(p for p in collected if os.path.isdir(p)))
        if complete:
            cls._search_path_cache[key] = result
        return result

    @classmethod
    def _conftest_paths(cls, directory: Path, skip: str = "") -> tuple[tuple[str, ...], bool]:
        """Declarations from the conftest.py chain above ``directory``.

        Returns ``(paths, complete)``; ``complete`` is False when a conftest.py
        exists on disk but no module for it could be found -- typically because
        it is the file currently executing, so its registration has not
        happened yet. An incomplete answer is usable but must not be cached.
        """
        key = str(directory)
        cached = cls._conftest_cache.get(key)
        if cached is not None:
            return cached, True

        # Resolved, because ``directory`` derives from a resolved source path.
        # Comparing a resolved path against an unresolved test_root never
        # matches when the checkout is reached through a symlink, and the walk
        # then climbs past the tree it was meant to stop in.
        root = Path(os.path.realpath(cls.test_root))
        collected: list[str] = []
        complete = True
        current = directory
        while True:
            base = str(current)
            conftest = os.path.join(base, "conftest.py")
            if os.path.isfile(conftest):
                if conftest == skip:
                    # Its declaration came from the live frame instead, and it
                    # is not snapshotted yet, so this answer is for now only.
                    complete = False
                else:
                    declared = cls._conftest_declaration(conftest)
                    if declared is None:
                        complete = False
                    else:
                        collected.extend(declared)
            # pytest loads conftests from its rootdir downwards only, so a
            # declaration above that point would never be readable here.
            if os.path.isfile(os.path.join(base, "pytest.ini")) or current == root:
                break
            if current.parent == current:  # filesystem root; never loop forever
                break
            current = current.parent

        result = tuple(collected)
        if complete:
            cls._conftest_cache[key] = result
        return result, complete

    @classmethod
    def _source_in_test_tree(cls, filename: str) -> str:
        """Symlink-resolved path if ``filename`` is a file under the test root.

        Empty string otherwise, which is how a requester gets rejected. Both
        steps matter and both used to be missing:

        * ``test_root`` is resolved, so an unresolved requester never matched
          it when the checkout was reached through a symlinked parent -- the
          finder silently stopped applying.
        * A frame can carry a pseudo-name (``<string>``, ``<stdin>``), which
          ``abspath`` happily turns into a path under the current directory.
          Requiring a real file rejects those.

        Cached per raw frame filename: this runs for every import that reaches
        the finder, and the resolution costs syscalls.
        """
        resolved = cls._source_cache.get(filename)
        if resolved is None:
            resolved = os.path.realpath(filename)
            if not (
                os.path.isfile(resolved)
                and Path(resolved).is_relative_to(Path(os.path.realpath(cls.test_root)))
            ):
                resolved = ""
            cls._source_cache[filename] = resolved
        return resolved

    @classmethod
    def _conftest_declaration(cls, conftest: str) -> tuple[str, ...] | None:
        """Snapshotted directories for a conftest, or None if it is not loaded."""
        resolved = cls._conftest_paths_by_file.get(conftest)
        if resolved is not None:
            return resolved
        # Nothing registered means pytest is not driving (a plain script, or a
        # unit test of this module); sys.modules is unambiguous in that case.
        module = cls._module_for_file(conftest)
        return None if module is None else cls._snapshot_conftest(conftest, module)

    @classmethod
    def _module_for_file(cls, file_path: str) -> ModuleType | None:
        """The imported module for ``file_path``, or None if it is not loaded."""
        module = cls._lookup(file_path)
        if module is not None:
            return module
        # A miss can only turn into a hit if sys.modules has changed since the
        # index was built, which keeps repeated misses from rescanning.
        if len(sys.modules) == cls._scanned_size:
            return None
        for name, candidate in list(sys.modules.items()):
            origin = _module_file(candidate)
            if origin:
                cls._file_to_module[origin] = name
        cls._scanned_size = len(sys.modules)
        return cls._lookup(file_path)

    @classmethod
    def _lookup(cls, file_path: str) -> ModuleType | None:
        name = cls._file_to_module.get(file_path)
        if name is None:
            return None
        module = sys.modules.get(name)
        # The name may have been rebound (or purged) since the index was built.
        return module if module is not None and _module_file(module) == file_path else None

    @classmethod
    def reset(cls) -> None:
        """Drops every cache and purges outstanding modules (for self-tests)."""
        cls.purge_magic_sys_modules()
        cls._magic_cache.clear()
        cls._magic_submodules.clear()
        cls._search_path_cache.clear()
        cls._conftest_cache.clear()
        cls._conftest_paths_by_file.clear()
        cls._source_cache.clear()
        cls._file_to_module.clear()
        cls._scanned_size = -1


def _requesting_source_file() -> tuple[str | None, dict]:
    """Returns ``(file, globals)`` for the frame that requested the import.

    That is the file whose ``import`` statement triggered this lookup -- a test
    module, a helper it imported, or a ``conftest.py`` pytest is executing --
    together with the namespace it is executing in, which is where its own
    ``__extra_import_path__`` lives.

    Deliberately the NEAREST such frame, not the nearest test file: the search
    stops at whatever ran the import, and if that turns out to sit outside the
    test tree the lookup is abandoned rather than continued up the stack. The
    rule is lexical -- a file's imports resolve against where that file lives.

    Walking further, to the first frame under ``tests/`` or the first
    ``conftest.py``/``test_*.py``, would make it dynamic instead, with three
    costs. It would let a test's declaration steer imports made *inside*
    library code it called, which is exactly where optional-dependency probes
    (``try: import ray``) live, and those are reached only when the import
    would otherwise have failed -- the one case this finder handles. It would
    make the answer depend on the caller chain, so the per-file caches below
    would be wrong. And it would deny a helper its own declaration, since a
    magic-imported helper importing its siblings would be attributed to the
    test module that pulled it in.

    The cost of the lexical rule is that an import performed on a test's behalf
    by library code (a plugin loaded by name from ``tensorrt_llm``, say) is not
    covered; that path still needs an explicit ``sys.path`` entry.
    """
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not _is_internal_frame(filename):
            return os.path.abspath(filename), frame.f_globals
        frame = frame.f_back
    return None, {}


# Importing this module is the whole bootstrap for a directly-executed test
# file. Harmless under pytest too, where the plugin installs it earlier.
MagicFinder.install()
