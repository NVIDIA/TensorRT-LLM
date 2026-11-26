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
"""Identity, FutureWarning and old-path-import checks of the kept serve shims.

``SHIM_ROWS`` starts empty: only the pull requests that keep an old import path
as a shim (old paths that code outside the repository imports) add rows here.
Each row then checks that the old name is the same object as the new one, that
the shim warns with the expected ``FutureWarning``, and that nothing inside the
repository imports the old path. The self-tests below prove the three checks
can fail, so the empty table is not vacuous.
"""

import sys
import textwrap
from pathlib import Path

import pytest
from test_common.serve.shims import ShimRow, check_identity, check_warnings, find_old_path_imports

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

# Rows are added only by the pull requests that create the kept shims.
SHIM_ROWS: tuple = ()


def test_shim_identity():
    assert check_identity(SHIM_ROWS) == []


def test_shim_warnings():
    assert check_warnings(SHIM_ROWS) == []


def test_no_old_path_imports():
    assert find_old_path_imports(SHIM_ROWS, REPO_ROOT) == []


# ---------------------------------------------------------------------------
# Self-test on a synthetic package: each check must be able to fail.
# ---------------------------------------------------------------------------


@pytest.fixture
def shim_package(tmp_path, monkeypatch):
    """A synthetic package with good, silent, copying, lazy and interface shims."""
    pkg = "shim_selftest_pkg"
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "new.py").write_text(
        textwrap.dedent(
            """
            class X:
                pass

            TABLE = {"a": 1}

            def main():
                pass
            """
        )
    )
    (root / "good.py").write_text(
        textwrap.dedent(
            f"""
            import warnings

            from {pkg}.new import TABLE, X

            warnings.warn(
                "{pkg}.good has moved to {pkg}.new and will be removed in 1.6.0.",
                FutureWarning,
                stacklevel=2,
            )
            __all__ = ["TABLE", "X"]
            """
        )
    )
    (root / "silent.py").write_text(f"from {pkg}.new import X\n__all__ = ['X']\n")
    (root / "bad.py").write_text("class X:\n    pass\n__all__ = ['X']\n")
    (root / "lazy.py").write_text(
        textwrap.dedent(
            f"""
            import warnings

            from {pkg}.new import main

            _LAZY_NAMES = {{"X": "{pkg}.new"}}

            def __getattr__(name):
                if name in _LAZY_NAMES:
                    warnings.warn(
                        f"{pkg}.lazy.{{name}} has moved to {pkg}.new "
                        "and will be removed in 1.6.0.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    import importlib

                    value = getattr(importlib.import_module(_LAZY_NAMES[name]), name)
                    globals()[name] = value  # warn once: later reads skip __getattr__
                    return value
                raise AttributeError(name)
            """
        )
    )
    (root / "iface.py").write_text(
        textwrap.dedent(
            f"""
            def __getattr__(name):
                if name == "X":
                    from {pkg}.new import X

                    return X
                raise AttributeError(name)
            """
        )
    )
    sub = root / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            import warnings

            from {pkg}.new import X

            warnings.warn(
                "{pkg}.sub has moved to {pkg}.new and will be removed in 1.6.0.",
                FutureWarning,
                stacklevel=2,
            )
            __all__ = ["X"]
            """
        )
    )
    (sub / "mod.py").write_text(
        textwrap.dedent(
            f"""
            import warnings

            from {pkg}.new import X

            warnings.warn(
                "{pkg}.sub.mod has moved to {pkg}.new and will be removed in 1.6.0.",
                FutureWarning,
                stacklevel=2,
            )
            __all__ = ["X"]
            """
        )
    )
    (root / "importer.py").write_text(f"from {pkg}.good import X\n")
    (root / "module_user.py").write_text(
        f"from {pkg} import good\n\n\ndef use():\n    return good.X\n"
    )
    (root / "other_name_user.py").write_text(
        f"from {pkg} import good\n\n\ndef use():\n    return good.TABLE\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield pkg, tmp_path
    for name in list(sys.modules):
        if name == pkg or name.startswith(pkg + "."):
            del sys.modules[name]


def _row(pkg, module, name, **kwargs):
    retire_in = kwargs.pop("retire_in", "1.6.0")
    return ShimRow(f"{pkg}.{module}", name, f"{pkg}.new", retire_in=retire_in, **kwargs)


def test_identity_check_passes_and_fails(shim_package):
    pkg, _ = shim_package
    good_rows = [_row(pkg, "good", "X"), _row(pkg, "good", "TABLE")]
    assert check_identity(good_rows) == []
    problems = check_identity([_row(pkg, "bad", "X")])
    assert len(problems) == 1 and "not the same object" in problems[0]
    problems = check_identity([_row(pkg, "good", "GONE")])
    assert len(problems) == 1 and "no longer exists" in problems[0]


def test_warning_check_passes_for_shim_styles(shim_package):
    pkg, _ = shim_package
    rows = [
        _row(pkg, "good", "X"),
        _row(pkg, "good", "TABLE"),
        _row(pkg, "lazy", "X"),  # warns on the first read only
        _row(pkg, "lazy", "main"),  # eager name: never warns
        _row(pkg, "iface", "X", warns=False, ban_old_path=False, retire_in="not retired"),
        # A fresh import of a shim submodule also imports (and warns for) its
        # parent shim package; the message filter must not count that warning.
        _row(pkg, "sub.mod", "X"),
    ]
    assert f"{pkg}.sub" not in sys.modules
    assert check_warnings(rows) == []


def test_warning_check_reports_silent_and_wrong_release(shim_package):
    pkg, _ = shim_package
    problems = check_warnings([_row(pkg, "silent", "X")])
    assert len(problems) == 1 and "emitted no FutureWarning" in problems[0]
    wrong_release = ShimRow(f"{pkg}.good", "X", f"{pkg}.new", retire_in="9.9.9")
    problems = check_warnings([wrong_release])
    assert len(problems) == 1 and "names the new location and the removal release" in problems[0]


def test_warning_check_restores_sys_modules(shim_package):
    pkg, _ = shim_package
    import importlib

    before = importlib.import_module(f"{pkg}.good")
    assert check_warnings([_row(pkg, "good", "X")]) == []
    assert sys.modules[f"{pkg}.good"] is before
    assert getattr(sys.modules[pkg], "good") is before


def test_old_path_import_scan(shim_package):
    pkg, tmp_path = shim_package
    rows = [_row(pkg, "good", "X")]
    problems = find_old_path_imports(rows, tmp_path, roots=(pkg,))
    importing_lines = [p for p in problems if "importer.py" in p]
    module_attribute_lines = [p for p in problems if "module_user.py" in p]
    other_name_lines = [p for p in problems if "other_name_user.py" in p]
    assert len(importing_lines) == 1, problems
    assert f"import it from `{pkg}.new`" in importing_lines[0]
    # `from <pkg> import good` + `good.X` loads the shim as surely as a from-import.
    assert len(module_attribute_lines) == 1, problems
    # Reading another name of the module is not banned by this row.
    assert other_name_lines == [], problems
    # The shim's own file is exempt.
    assert not any("good.py" in p for p in problems), problems
