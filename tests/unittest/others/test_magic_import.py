# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``__extra_import_path__`` meta path finder.

The behaviour under test is documented in :mod:`test_common.magic_import`;
these cases are the executable form of the rules stated there. Each test names
the rule it pins, so a failure identifies which guarantee was broken.

The ``tree`` fixture builds a throwaway project and points the finder at it.
``_Tree.run`` then reproduces the sequence pytest performs -- importing each
ancestor conftest under the bare name ``conftest`` and registering it with the
finder -- because several of the guarantees depend on that sequence rather than
on the finder alone.
"""

import importlib.util
import itertools
import os
import subprocess
import sys
import textwrap
import threading
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_common.magic_import import MagicFinder
from test_common.magic_import_hooks import (
    _NON_TEST_TREES,
    _check_sys_path,
    _relative_pythonpath_entries,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A throwaway project/test root with MagicFinder installed against it."""
    project_root = tmp_path / "repo"
    test_root = project_root / "tests"
    test_root.mkdir(parents=True)

    monkeypatch.setattr(MagicFinder, "test_root", str(test_root))
    monkeypatch.setattr(MagicFinder, "project_root", str(project_root))
    monkeypatch.setattr(MagicFinder, "_magic_cache", {})
    monkeypatch.setattr(MagicFinder, "_magic_submodules", {})
    monkeypatch.setattr(MagicFinder, "_active_sys_modules", set())
    monkeypatch.setattr(MagicFinder, "_search_path_cache", {})
    monkeypatch.setattr(MagicFinder, "_conftest_cache", {})
    monkeypatch.setattr(MagicFinder, "_conftest_paths_by_file", {})
    monkeypatch.setattr(MagicFinder, "_source_cache", {})
    monkeypatch.setattr(MagicFinder, "_file_to_module", {})
    monkeypatch.setattr(MagicFinder, "_scanned_size", -1)

    meta_path = list(sys.meta_path)
    before = set(sys.modules)
    MagicFinder.install()
    try:
        yield _Tree(project_root, test_root)
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]
        sys.meta_path[:] = meta_path


class _Tree:
    """Builds a fixture repo and executes files the way the import system does."""

    _counter = itertools.count()

    # Fixture module names that are allowed to collide with a real module,
    # because the test is specifically about not shadowing one.
    _MAY_COLLIDE = frozenset({"json"})

    def __init__(self, project_root, test_root):
        self.project_root = project_root
        self.test_root = test_root

    def write(self, rel: str, body: str) -> Path:
        """Writes a file under the project root; returns its path.

        Refuses a name that is already importable. The finder is last in
        sys.meta_path, so for such a name it correctly declines to act and the
        test would silently assert against the wrong module.
        """
        path = self.project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))

        name = path.stem
        if (
            path.suffix == ".py"
            and name not in self._MAY_COLLIDE
            and not name.startswith(("test_", "__"))
            and name != "conftest"
        ):
            try:
                existing = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                existing = None
            assert existing is None, (
                f"fixture module {name!r} is already importable from "
                f"{existing.origin!r}; the finder will not shadow it, so this "
                f"fixture cannot test what it intends -- pick a unique name"
            )
        return path

    def run(self, path: Path, register: bool = True) -> dict:
        """Imports every ancestor conftest, then executes ``path``.

        Mirrors pytest: conftests are imported outermost-first before anything
        in their directory runs, all under the bare module name ``conftest``
        (so each evicts the previous from ``sys.modules``), and each is handed
        to the finder the way ``pytest_plugin_registered`` does.

        ``register=False`` skips that registration and gives every conftest a
        distinct module name instead, exercising the ``sys.modules`` fallback
        used when pytest is not driving.
        """
        for conftest in self._conftest_chain(path.parent):
            self._load_conftest(conftest, register)
        return self._exec(path)

    def _conftest_chain(self, directory: Path) -> list[Path]:
        """Ancestor conftests, outermost first, down from the test root."""
        if not directory.is_relative_to(Path(self.test_root)):
            raise ValueError(f"{directory} outside of test_root {self.test_root}")
        chain = []
        current = directory
        while True:
            conftest = current / "conftest.py"
            if conftest.is_file():
                chain.append(conftest)
            if current == self.test_root:
                break
            current = current.parent
        return list(reversed(chain))

    def _load_conftest(self, path: Path, register: bool) -> None:
        name = "conftest" if register else f"_conftest_{next(self._counter)}"
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        if register:
            MagicFinder.register_conftest(module)

    def _exec(self, path: Path) -> dict:
        """Runs a file; its frame globals are what the finder reads."""
        namespace = {"__file__": str(path), "__name__": "__fixture__"}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace


def test_conftest_declaration_applies_to_whole_subtree(tree):
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    deep = tree.write("tests/a/b/c/test_deep.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    assert tree.run(deep)["RESULT"] == "shared"


def test_walk_stops_at_the_directory_holding_pytest_ini(tree):
    """Pytest does not load conftests above its rootdir, and neither does this.

    Mirrors this repo: ``tests/unittest/pytest.ini`` makes ``tests/conftest.py``
    invisible to a normal run.
    """
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    tree.write("tests/unittest/pytest.ini", "[pytest]\n")
    case = tree.write("tests/unittest/deep/test_deep.py", "import mi_helper\n")

    with pytest.raises(ModuleNotFoundError):
        tree.run(case)


def test_declaration_at_the_pytest_ini_directory_is_used(tree):
    """The rootdir conftest itself is in range -- the walk stops after it."""
    tree.write("tests/unittest/shared/mi_helper.py", "VALUE = 'root-level'\n")
    tree.write("tests/unittest/pytest.ini", "[pytest]\n")
    tree.write("tests/unittest/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write(
        "tests/unittest/deep/test_deep.py", "import mi_helper\nRESULT = mi_helper.VALUE\n"
    )

    assert tree.run(case)["RESULT"] == "root-level"


def test_nearest_declaration_wins_over_ancestor(tree):
    tree.write("tests/far/mi_helper.py", "VALUE = 'far'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['far']\n")
    tree.write("tests/near/local/mi_helper.py", "VALUE = 'near'\n")
    tree.write("tests/near/conftest.py", "__extra_import_path__ = ['local']\n")
    case = tree.write("tests/near/test_near.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    assert tree.run(case)["RESULT"] == "near"


def test_same_name_in_sibling_trees_stays_isolated(tree):
    """Scoping rule: a generic name resolves per file, not process-wide.

    This is the defect ``sys.path.append`` produces -- the first importer
    wins for the whole process.
    """
    tree.write("tests/one/helpers/mi_utils.py", "VALUE = 'one'\n")
    tree.write("tests/one/conftest.py", "__extra_import_path__ = ['helpers']\n")
    tree.write("tests/two/helpers/mi_utils.py", "VALUE = 'two'\n")
    tree.write("tests/two/conftest.py", "__extra_import_path__ = ['helpers']\n")
    first = tree.write("tests/one/test_one.py", "import mi_utils\nRESULT = mi_utils.VALUE\n")
    second = tree.write("tests/two/test_two.py", "import mi_utils\nRESULT = mi_utils.VALUE\n")

    assert tree.run(first)["RESULT"] == "one"
    MagicFinder.purge_magic_sys_modules()
    assert tree.run(second)["RESULT"] == "two"


def test_purge_removes_package_and_its_submodules(tree):
    tree.write("tests/shared/mi_pkg/__init__.py", "")
    tree.write("tests/shared/mi_pkg/mi_sub.py", "VALUE = 'mi_sub'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/test_pkg.py", "from mi_pkg.mi_sub import VALUE\nRESULT = VALUE\n")

    assert tree.run(case)["RESULT"] == "mi_sub"
    assert {"mi_pkg", "mi_pkg.mi_sub"} <= set(sys.modules)

    MagicFinder.purge_magic_sys_modules()
    assert not {"mi_pkg", "mi_pkg.mi_sub"} & set(sys.modules)


def test_reimport_reuses_cached_module_without_re_executing(tree):
    """Caching rule: a module is executed at most once per interpreter.

    Module-level side effects such as custom operator registration would
    fail on a second execution.
    """
    tree.write("tests/shared/mi_pkg/__init__.py", "RUNS = []\n")
    tree.write("tests/shared/mi_pkg/mi_sub.py", "import mi_pkg\nmi_pkg.RUNS.append('mi_sub')\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/test_pkg.py", "import mi_pkg.mi_sub\nRESULT = mi_pkg\n")

    first = tree.run(case)["RESULT"]
    MagicFinder.purge_magic_sys_modules()
    second = tree.run(case)["RESULT"]
    MagicFinder.purge_magic_sys_modules()

    assert first is second
    assert first.RUNS == ["mi_sub"]


def test_tilde_entry_is_project_root_relative(tree):
    tree.write("examples/apps/mi_demo.py", "VALUE = 'example'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['~/examples/apps']\n")
    case = tree.write("tests/test_example.py", "import mi_demo\nRESULT = mi_demo.VALUE\n")

    assert tree.run(case)["RESULT"] == "example"


def test_tilde_is_not_the_users_home_directory(tree, monkeypatch):
    monkeypatch.setenv("HOME", str(tree.project_root / "not-here"))
    tree.write("examples/mi_demo.py", "VALUE = 'example'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['~/examples']\n")
    case = tree.write("tests/test_example.py", "import mi_demo\nRESULT = mi_demo.VALUE\n")

    assert tree.run(case)["RESULT"] == "example"


def test_dot_and_dotdot_are_file_relative(tree):
    tree.write("tests/a/mi_sibling.py", "VALUE = 'mi_sibling'\n")
    tree.write("tests/mi_parent_level.py", "VALUE = 'parent'\n")
    case = tree.write(
        "tests/a/test_rel.py",
        """
        __extra_import_path__ = ['.', '..']
        import mi_sibling
        import mi_parent_level
        RESULT = (mi_sibling.VALUE, mi_parent_level.VALUE)
        """,
    )

    assert tree.run(case)["RESULT"] == ("mi_sibling", "parent")


def test_declaration_in_the_importing_file_may_be_computed(tree, monkeypatch):
    tree.write("tests/generated/mi_helper.py", "VALUE = 'computed'\n")
    case = tree.write(
        "tests/test_computed.py",
        """
        import os
        __extra_import_path__ = [os.environ['FIXTURE_DIR']]
        import mi_helper
        RESULT = mi_helper.VALUE
        """,
    )

    monkeypatch.setenv("FIXTURE_DIR", "generated")
    assert tree.run(case)["RESULT"] == "computed"


def test_declaration_takes_effect_only_for_later_imports(tree):
    """An earlier failed import must not freeze the not-yet-declared value."""
    tree.write("tests/generated/mi_helper.py", "VALUE = 'late'\n")
    case = tree.write(
        "tests/test_late.py",
        """
        try:
            import an_optional_dependency
        except ImportError:
            an_optional_dependency = None
        __extra_import_path__ = ['generated']
        import mi_helper
        RESULT = mi_helper.VALUE
        """,
    )

    assert tree.run(case)["RESULT"] == "late"


def test_conftest_may_use_its_own_declaration(tree):
    tree.write("tests/support/mi_fixtures.py", "VALUE = 'conftest-visible'\n")
    conftest = tree.write(
        "tests/mi_pkg/conftest.py",
        """
        __extra_import_path__ = ['../support']
        import mi_fixtures
        RESULT = mi_fixtures.VALUE
        """,
    )

    assert tree.run(conftest)["RESULT"] == "conftest-visible"


def test_ancestor_conftest_survives_the_bare_conftest_module_name(tree):
    """Pytest imports every conftest as ``conftest``, so they evict each other.

    The nearest conftest is the last one imported and is the only one left in
    ``sys.modules``; the ancestor's declaration has to come from the registry
    that ``pytest_plugin_registered`` fills.
    """
    tree.write("tests/shared/mi_from_root.py", "VALUE = 'root'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    tree.write("tests/leaf/helpers/mi_from_leaf.py", "VALUE = 'leaf'\n")
    tree.write("tests/leaf/conftest.py", "__extra_import_path__ = ['helpers']\n")
    case = tree.write(
        "tests/leaf/test_leaf.py",
        "import mi_from_root, mi_from_leaf\nRESULT = (mi_from_root.VALUE, mi_from_leaf.VALUE)\n",
    )

    assert tree.run(case)["RESULT"] == ("root", "leaf")
    # Precondition of the test: only the nearest conftest is reachable there.
    assert sys.modules["conftest"].__file__ == str(tree.test_root / "leaf" / "conftest.py")


def test_conftests_resolve_via_sys_modules_when_pytest_is_not_driving(tree):
    """Plain `python mi_script.py` under the test tree: no plugin registrations."""
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/mi_script.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    assert tree.run(case, register=False)["RESULT"] == "shared"
    # Resolved without any registration, then snapshotted like a registered one.
    assert MagicFinder._conftest_paths_by_file == {
        str(tree.test_root / "conftest.py"): (str(tree.test_root / "shared"),)
    }


def test_conftest_declaration_is_snapshotted_at_load_time(tree):
    """Mutating a conftest declaration after it loads is explicitly not supported."""
    tree.write("tests/early/mi_helper.py", "VALUE = 'early'\n")
    tree.write("tests/late/mi_late_helper.py", "VALUE = 'late'\n")
    conftest = tree.write("tests/conftest.py", "__extra_import_path__ = ['early']\n")
    case = tree.write("tests/test_snapshot.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    assert tree.run(case)["RESULT"] == "early"

    # Appending after load must not extend the search path.
    assert sys.modules["conftest"].__file__ == str(conftest)
    sys.modules["conftest"].__extra_import_path__.append("late")
    late = tree.write("tests/test_late_add.py", "import mi_late_helper\n")
    with pytest.raises(ModuleNotFoundError):
        tree._exec(late)


def test_rewrite_hook_gets_first_refusal(tree, monkeypatch):
    """Rewriting rule: the assertion rewriter is offered the declared directories."""
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/test_hook.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    seen = []

    class _Hook:
        def find_spec(self, name, path=None, target=None):
            seen.append((name, tuple(path or ())))
            return None  # "not a module I would rewrite" -> plain load

    monkeypatch.setattr(MagicFinder, "rewrite_hook", _Hook())

    assert tree.run(case)["RESULT"] == "shared"
    assert seen == [("mi_helper", (str(tree.test_root / "shared"),))]


def test_rewrite_hook_spec_is_what_executes(tree, monkeypatch):
    """When the rewriter claims the module, its loader is the one that runs."""
    tree.write("tests/shared/mi_helper.py", "VALUE = 'plain'\n")
    tree.write("tests/rewritten/mi_helper.py", "VALUE = 'rewritten'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/test_hook.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    elsewhere = str(tree.test_root / "rewritten")

    class _Hook:
        def find_spec(self, name, path=None, target=None):
            return PathFinder.find_spec(name, [elsewhere])

    monkeypatch.setattr(MagicFinder, "rewrite_hook", _Hook())

    assert tree.run(case)["RESULT"] == "rewritten"


def test_standalone_file_declaration_works_with_no_conftest_loaded(tree):
    """`python test_foo.py`: no pytest, so no conftest is ever imported.

    The file's own declaration still applies (it comes off the executing
    frame), and an unloaded conftest in the chain must not break the lookup.
    """
    tree.write("tests/helpers/mi_helper.py", "VALUE = 'standalone'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['never-loaded']\n")
    case = tree.write(
        "tests/deep/test_direct.py",
        """
        __extra_import_path__ = ['~/tests/helpers']
        import mi_helper
        RESULT = mi_helper.VALUE
        """,
    )

    # _exec, not run: nothing imports the conftest chain.
    assert tree._exec(case)["RESULT"] == "standalone"


def test_importing_the_module_installs_the_finder():
    """The one-line bootstrap a directly-executed test file relies on."""
    tests_root = Path(MagicFinder.test_root)
    probe = "import sys, test_common.magic_import as m; print(sys.meta_path[-1] is m.MagicFinder)"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "PYTHONPATH": str(tests_root)},
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "True", result.stderr


def test_resolution_on_one_thread_does_not_silence_another(tree):
    """The re-entrancy guard is per-thread.

    A file imported from a worker thread must resolve its declared helper even
    while another thread is already resolving that same name, which a guard
    shared across threads would suppress.
    """
    tree.write("tests/helpers/mi_threaded.py", "VALUE = 'threaded'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['helpers']\n")
    case = tree.write("tests/test_threaded.py", "import mi_threaded\nRESULT = mi_threaded.VALUE\n")

    result = {}

    def worker():
        try:
            result["namespace"] = tree.run(case)
        except BaseException as exc:  # reported below rather than swallowed
            result["error"] = exc

    with MagicFinder._recursive_guard("mi_threaded") as claimed:
        assert claimed, "this thread should own the name"
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert "error" not in result, result["error"]
    assert result["namespace"]["RESULT"] == "threaded"


def test_a_declared_helper_may_import_another_declared_helper(tree):
    """Integration: nested helpers both resolve.

    Note this does not exercise the re-entrancy guard -- find_spec has already
    returned, and the guard released the name, by the time the loader executes
    the module and its own imports run. The guard's behaviour is pinned
    directly below.
    """
    tree.write("tests/helpers/mi_inner.py", "VALUE = 'inner'\n")
    tree.write("tests/helpers/mi_outer.py", "import mi_inner\nVALUE = 'outer+' + mi_inner.VALUE\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['helpers']\n")
    case = tree.write("tests/test_nested.py", "import mi_outer\nRESULT = mi_outer.VALUE\n")

    assert tree.run(case)["RESULT"] == "outer+inner"


def test_recursive_guard_declines_a_name_already_in_flight():
    """A request for the name currently being resolved is refused, not retried."""
    with MagicFinder._recursive_guard("mi_selfish") as claimed:
        assert claimed, "a free name must be claimable"
        with MagicFinder._recursive_guard("mi_selfish") as reentered:
            assert not reentered, "a name already in flight must not be claimed again"


def test_recursive_guard_is_per_name():
    """Holding one name must not block resolution of a different one.

    A resolution can legitimately trigger an import of something else -- the
    assertion-rewriting hook and conftest loading both do -- and that must
    still be served.
    """
    with MagicFinder._recursive_guard("mi_held") as claimed:
        assert claimed
        with MagicFinder._recursive_guard("mi_other") as other:
            assert other, "a different name must still be claimable"


def test_recursive_guard_releases_the_name_after_use():
    """The name is freed on exit, including when the body raises."""
    with pytest.raises(RuntimeError):
        with MagicFinder._recursive_guard("mi_transient") as claimed:
            assert claimed
            raise RuntimeError("boom")

    with MagicFinder._recursive_guard("mi_transient") as claimed:
        assert claimed, "the name must be claimable again after a failed resolution"


def test_undeclared_name_still_raises(tree):
    tree.write("tests/conftest.py", "__extra_import_path__ = []\n")
    case = tree.write("tests/test_missing.py", "import nowhere_at_all\n")

    with pytest.raises(ModuleNotFoundError):
        tree.run(case)


def test_files_outside_the_test_root_are_never_handled(tree):
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    outside = tree.write("src/mi_prod.py", "import mi_helper\n")

    with pytest.raises(ModuleNotFoundError):
        tree._exec(outside)


def test_requester_reached_through_a_symlink_still_matches(tree):
    """test_root is resolved, so an unresolved requester would never match it."""
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/deep/test_deep.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")
    for conftest in tree._conftest_chain(case.parent):
        tree._load_conftest(conftest, register=True)

    link = tree.project_root.parent / "link"
    link.symlink_to(tree.project_root)
    through_link = link / "tests" / "deep" / "test_deep.py"

    assert tree._exec(through_link)["RESULT"] == "shared"


def test_pseudo_filename_frames_are_not_treated_as_files(tree, monkeypatch):
    """`python -c` from inside the tree must not inherit a conftest chain."""
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    conftest = tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    tree._load_conftest(conftest, register=True)
    (tree.test_root / "deep").mkdir()
    monkeypatch.chdir(tree.test_root / "deep")

    # abspath("<string>") lands under the test root; it is still not a file.
    with pytest.raises(ModuleNotFoundError):
        exec(compile("import mi_helper", "<string>", "exec"), {"__name__": "__fixture__"})


def test_finder_never_shadows_an_importable_module(tree):
    """Precedence rule: an already importable module is never shadowed."""
    tree.write("tests/shared/json.py", "VALUE = 'impostor'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = ['shared']\n")
    case = tree.write("tests/test_stdlib.py", "import json\nRESULT = json.dumps([1])\n")

    assert tree.run(case)["RESULT"] == "[1]"


def test_bare_string_declaration_is_rejected(tree):
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/conftest.py", "__extra_import_path__ = 'shared'\n")
    case = tree.write("tests/test_bad.py", "import mi_helper\n")

    with pytest.raises(TypeError, match="must be a list of paths"):
        tree.run(case)


def test_conftest_helper_imported_during_conftest_execution(tree):
    """A conftest may import a mi_helper of its own, before it is registered.

    Models tests/integration/defs/cpp: conftest.py imports cpp_common, so the
    mi_helper resolves while the conftest is still executing and therefore not yet
    registered. The mi_helper must see the declaration, and -- the part that is
    easy to get wrong -- so must every later file in that directory.
    """
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write("tests/mi_pkg/mi_sibling.py", "import mi_helper\nVALUE = mi_helper.VALUE\n")
    tree.write(
        "tests/mi_pkg/conftest.py",
        """
        __extra_import_path__ = ['.', '../shared']
        import mi_sibling
        RESULT = mi_sibling.VALUE
        """,
    )
    case = tree.write("tests/mi_pkg/test_after.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    # The conftest's own mi_helper import resolves mid-execution...
    assert tree.run(case)["RESULT"] == "shared"
    # ...and the declaration is still live for files collected afterwards.
    MagicFinder.purge_magic_sys_modules()
    assert tree._exec(case)["RESULT"] == "shared"


def test_early_lookup_does_not_poison_the_directory(tree):
    """A reading taken mid-execution must not outlive the real declaration.

    A conftest that imports its own mi_helper is read before it finishes, so the
    value seen then can be incomplete. Registration has to discard anything
    derived from it, or every later file in the directory inherits the stale
    answer.
    """
    tree.write("tests/shared/mi_helper.py", "VALUE = 'shared'\n")
    tree.write(
        "tests/mi_pkg/mi_sibling.py",
        "try:\n    import mi_helper\nexcept ImportError:\n    mi_helper = None\n",
    )
    tree.write(
        "tests/mi_pkg/conftest.py",
        """
        __extra_import_path__ = ['.']
        import mi_sibling                  # resolves before the real declaration
        __extra_import_path__ = ['.', '../shared']
        """,
    )
    case = tree.write("tests/mi_pkg/test_after.py", "import mi_helper\nRESULT = mi_helper.VALUE\n")

    assert tree.run(case)["RESULT"] == "shared"


def test_conftest_declaration_survives_a_symlinked_checkout(tree, tmp_path):
    """A CI workspace is often reached through a symlink.

    pytest then imports the conftest under the symlinked spelling while the
    requester resolves to the real one. If the two are normalised differently
    the conftest lookup misses -- and if test_root is compared unresolved, the
    directory walk never reaches it and climbs to the filesystem root.
    """
    tree.write("tests/scripts/mi_shared_helper.py", "VALUE = 'resolved'\n")
    tree.write("tests/mi_pkg/conftest.py", "__extra_import_path__ = ['~/tests/scripts']\n")
    case = tree.write(
        "tests/mi_pkg/test_case.py", "import mi_shared_helper\nRESULT = mi_shared_helper.VALUE\n"
    )

    link = tmp_path / "workspace_link"
    link.symlink_to(tree.project_root)
    # conftest imported through the symlink, exactly as pytest would see it
    tree._load_conftest(link / "tests" / "mi_pkg" / "conftest.py", register=True)

    assert tree._exec(link / "tests" / "mi_pkg" / "test_case.py")["RESULT"] == "resolved"
    assert case.exists()


def test_module_cache_hits_when_the_helper_dir_is_symlinked(tree, tmp_path):
    """The execute-once guarantee must survive a symlinked declared directory.

    find_spec looks the module up by spec.origin; purge writes that cache
    through _module_file. If the two normalise differently the lookup misses
    and the module runs twice, which is what the cache exists to prevent.
    """
    real = tmp_path / "real_helpers"
    real.mkdir()
    (real / "mi_pkg").mkdir()
    (real / "mi_pkg" / "__init__.py").write_text("RUNS = []\n")
    (real / "mi_pkg" / "mi_sub.py").write_text("import mi_pkg\nmi_pkg.RUNS.append('x')\n")
    link = tree.test_root / "linked_helpers"
    link.symlink_to(real)

    tree.write("tests/conftest.py", "__extra_import_path__ = ['linked_helpers']\n")
    case = tree.write("tests/test_once.py", "import mi_pkg.mi_sub\nRESULT = mi_pkg\n")

    first = tree.run(case)["RESULT"]
    MagicFinder.purge_magic_sys_modules()
    second = tree.run(case)["RESULT"]
    MagicFinder.purge_magic_sys_modules()

    assert first is second, "cache missed: module re-executed"
    assert first.RUNS == ["x"], f"side effect ran {len(first.RUNS)} times"


class _FakeConfig:
    """The three attributes ``_check_sys_path`` reads off a pytest Config."""

    def __init__(self, rootpath, pythonpath=()):
        self.rootpath = rootpath
        self.invocation_params = SimpleNamespace(dir=rootpath)
        self._pythonpath = [str(entry) for entry in pythonpath]

    def getini(self, name):
        assert name == "pythonpath", name
        return self._pythonpath


@pytest.fixture
def sys_path_report(tree, monkeypatch):
    """Runs the sys.path check over an explicit sys.path against the fixture repo."""

    def run(*entries, pythonpath=()):
        monkeypatch.setattr(sys, "path", [str(entry) for entry in entries])
        return _check_sys_path(_FakeConfig(tree.project_root, pythonpath))

    return run


def test_sys_path_check_passes_on_a_clean_run(sys_path_report, tree):
    assert sys_path_report(tree.project_root, tree.test_root, "/usr/lib/python3") == []


def test_sys_path_check_reports_a_helper_directory(sys_path_report, tree):
    """The pattern the check exists to catch: a bare directory of helpers."""
    helpers = tree.project_root / "tests" / "scripts" / "helpers"
    helpers.mkdir(parents=True)
    (helpers / "mi_helper.py").write_text("VALUE = 1\n")

    assert sys_path_report(helpers) == [os.path.join("tests", "scripts", "helpers")]


def test_sys_path_check_accepts_a_directory_holding_test_files(sys_path_report, tree):
    """Pytest prepends the basedir of every module it imports, and never removes it."""
    basedir = tree.test_root / "suite"
    basedir.mkdir()
    (basedir / "test_thing.py").write_text("")

    assert sys_path_report(basedir) == []


def test_sys_path_check_accepts_a_directory_holding_only_a_conftest(sys_path_report, tree):
    basedir = tree.test_root / "suite"
    basedir.mkdir()
    (basedir / "conftest.py").write_text("")

    assert sys_path_report(basedir) == []


def test_sys_path_check_accepts_the_parent_of_a_package(sys_path_report, tree):
    """A basedir need not hold the test file itself.

    Pytest inserts the first ancestor that is not a package, so for a test
    inside a package the entry is that package's parent, which may contain no
    test file of its own -- tests/integration is exactly this case.
    """
    parent = tree.test_root / "integration"
    package = parent / "defs"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "test_thing.py").write_text("")

    assert sys_path_report(parent) == []


def test_sys_path_check_ignores_paths_outside_the_project(sys_path_report, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    assert sys_path_report(outside, "/usr/lib/python3", "") == []


@pytest.mark.parametrize("ignored_tree", _NON_TEST_TREES)
def test_sys_path_check_ignores_every_non_test_tree(sys_path_report, tree, ignored_tree):
    """Entries the product or a vendored package adds when imported.

    A test author cannot act on these, so reporting them would be noise. Driven
    from _NON_TEST_TREES itself so that editing that list cannot leave this
    test asserting something the check no longer does.
    """
    directory = tree.project_root / ignored_tree / "nested" / "python"
    directory.mkdir(parents=True)

    assert sys_path_report(directory) == []


@pytest.mark.parametrize("ignored_tree", _NON_TEST_TREES)
def test_sys_path_check_ignores_the_exempt_directory_itself(sys_path_report, tree, ignored_tree):
    """An entry names a directory, not only the things below it."""
    directory = tree.project_root / ignored_tree
    directory.mkdir(parents=True)

    assert sys_path_report(directory) == []


def test_sys_path_check_matches_exempt_trees_by_path_component(sys_path_report, tree):
    """A sibling whose name merely starts with an exempt entry is not exempt."""
    exempt = _NON_TEST_TREES[-1]
    sibling = tree.project_root / (exempt + "_other")
    sibling.mkdir(parents=True)

    assert sys_path_report(sibling) == [(exempt + "_other").replace("/", os.sep)]


def test_sys_path_check_accepts_pythonpath_ini_entries(sys_path_report, tree):
    """pytest.ini's pythonpath is a declared, reviewed entry, not a leak."""
    declared = tree.project_root / "tests" / "declared"
    declared.mkdir(parents=True)
    (declared / "mi_helper.py").write_text("")

    assert sys_path_report(declared) != [], "precondition: reported when undeclared"
    assert sys_path_report(declared, pythonpath=[declared]) == []


def test_sys_path_check_accepts_an_absolute_pythonpath_entry(sys_path_report, tree, monkeypatch):
    """A CI harness may point PYTHONPATH at a directory inside the project."""
    declared = tree.project_root / "ci-output" / "overrides"
    declared.mkdir(parents=True)

    assert sys_path_report(declared) != [], "precondition: reported when undeclared"
    monkeypatch.setenv("PYTHONPATH", str(declared))
    assert sys_path_report(declared) == []


@pytest.mark.parametrize(
    "pythonpath,expected",
    [
        pytest.param("", [], id="unset"),
        pytest.param("/abs/one:/abs/two", [], id="absolute-only"),
        pytest.param(":/abs/one", [], id="empty-entry-from-appending-to-unset"),
        pytest.param("relative/dir", ["relative/dir"], id="relative"),
        # What a PYTHONPATH entry holding a pytest node id splits into: the
        # "::" contains os.pathsep, so one entry becomes three.
        pytest.param(
            "/out/overrides-test_mod.py::test_case:/abs/two",
            ["test_case"],
            id="node-id-split",
        ),
    ],
)
def test_relative_pythonpath_entries_are_identified(monkeypatch, pythonpath, expected):
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    assert _relative_pythonpath_entries() == expected


def test_sys_path_check_still_reports_a_relative_pythonpath_entry(
    sys_path_report, tree, monkeypatch
):
    """Bypassing the early failure must not turn a relative entry into a declaration.

    pytest_configure rejects these outright, so this is the second line of
    defence: only a bypassed run reaches it, and there the entry stays
    reportable rather than being excused as PYTHONPATH-declared.
    """
    leaked = tree.project_root / "test_some_case"
    leaked.mkdir()
    monkeypatch.chdir(tree.project_root)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/abs/dir.py", "", "test_some_case"]))

    assert sys_path_report("test_some_case") == ["test_some_case"]


def _run_pytest_with_the_check(
    project_root, leak: bool, extra_pythonpath: str = ""
) -> subprocess.CompletedProcess:
    """Runs a real pytest over a throwaway repo with the plugin enabled.

    The plugin resolves the project root from the real magic_import module, so
    the generated conftest repoints it at ``project_root``; otherwise every
    path here would sit outside the project root and be ignored.
    """
    suite = project_root / "tests" / "suite"
    suite.mkdir(parents=True)
    (project_root / "tests" / "conftest.py").write_text(
        textwrap.dedent(f"""
        from test_common.magic_import import MagicFinder

        MagicFinder.project_root = {str(project_root)!r}
        MagicFinder.test_root = {str(project_root / "tests")!r}
        """)
    )
    if leak:
        helpers = project_root / "tests" / "helpers"
        helpers.mkdir()
        (helpers / "mi_leaked_helper.py").write_text("VALUE = 1\n")
        (suite / "conftest.py").write_text(
            textwrap.dedent(f"""
            import sys

            sys.path.insert(0, {str(helpers)!r})
            """)
        )
    (suite / "test_case.py").write_text("def test_ok():\n    assert True\n")

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "test_common.magic_import_hooks",
        ],
        cwd=str(project_root),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [MagicFinder.test_root, *([extra_pythonpath] if extra_pythonpath else [])]
            ),
            "TRTLLM_SKIP_SYS_PATH_CHECK": "",
        },
        capture_output=True,
        text=True,
    )


def test_check_passes_end_to_end_on_a_real_pytest_run(tmp_path):
    """The basedirs pytest inserts for its own imports must not trip the check.

    Pytest does not remove them at the end of a run, so they are still on
    sys.path when the check runs at pytest_sessionfinish. This is the case that
    would make the check unusable if _is_pytest_basedir were wrong.
    """
    result = _run_pytest_with_the_check(tmp_path / "repo", leak=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "sys.path check failed" not in result.stdout


def test_relative_pythonpath_is_reported_before_any_test(tmp_path):
    """A malformed environment is named at configure time, not as a later symptom.

    Only the report is asserted: failing the run on this is currently commented
    out in pytest_configure, so the run still proceeds. Assert the exit status
    here once that is turned on.
    """
    result = _run_pytest_with_the_check(
        tmp_path / "repo", leak=False, extra_pythonpath="relative/dir"
    )
    output = result.stdout + result.stderr

    assert "PYTHONPATH check failed" in output, output
    assert "not absolute paths" in output and "'relative/dir'" in output


def test_check_reports_end_to_end_when_a_conftest_leaks(tmp_path):
    """The whole wiring: a leak at import time is named at session finish."""
    result = _run_pytest_with_the_check(tmp_path / "repo", leak=True)
    output = result.stdout + result.stderr

    # TODO: enable return code assertion
    # assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert "sys.path check failed" in output, output
    assert os.path.join("tests", "helpers") in output


def test_sys_path_check_reports_each_entry_once(sys_path_report, tree):
    """sys.path routinely holds duplicates; the report should not."""
    helpers = tree.project_root / "tests" / "helpers"
    helpers.mkdir(parents=True)
    (helpers / "mi_helper.py").write_text("")

    assert sys_path_report(helpers, helpers, str(helpers) + os.sep) == [
        os.path.join("tests", "helpers")
    ]
