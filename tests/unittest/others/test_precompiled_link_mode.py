# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_PRECOMPILED_LINK`` in ``setup.py``'s ``extract_from_precompiled``.

The local-directory source normally copies the compiled artifacts into the
checkout. ``TRTLLM_PRECOMPILED_LINK=1`` symlinks them instead so several
checkouts can share one build tree.

``setup.py`` cannot be imported (module scope runs ``setup()``), so the two
functions under test are pulled out of its AST and executed on their own. They
are self-contained: every import they need is inside the function body, apart
from ``os``.
"""

import ast
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_SETUP_PY = Path(__file__).resolve().parents[3] / "setup.py"
_WANTED = ("should_skip_precompiled_package_data", "warn_on_build_skew", "extract_from_precompiled")


def _load(name):
    assert _SETUP_PY.is_file(), _SETUP_PY
    tree = ast.parse(_SETUP_PY.read_text(encoding="utf-8"))
    wanted = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _WANTED
    ]
    assert {node.name for node in wanted} == set(_WANTED)
    namespace = {"os": os}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(_SETUP_PY), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def extract_from_precompiled():
    return _load("extract_from_precompiled")


@pytest.fixture(scope="module")
def warn_on_build_skew():
    return _load("warn_on_build_skew")


@pytest.fixture
def build_tree(tmp_path, monkeypatch):
    """A source checkout holding compiled artifacts, and an empty destination."""
    source = tmp_path / "built-checkout"
    (source / "tensorrt_llm" / "libs").mkdir(parents=True)
    (source / "tensorrt_llm" / "libs" / "libtensorrt_llm.so").write_text("so")
    (source / "3rdparty" / "fmha_sm100").mkdir(parents=True)
    (source / "3rdparty" / "fmha_sm100" / "__init__.py").write_text("")

    destination = tmp_path / "checkout"
    destination.mkdir()
    monkeypatch.chdir(destination)
    return source


PACKAGE_DATA = ["libs/*.so"]


def _run(extract, source, workspace, link):
    previous = os.environ.get("TRTLLM_PRECOMPILED_LINK")
    os.environ["TRTLLM_PRECOMPILED_LINK"] = "1" if link else "0"
    try:
        extract(str(source), PACKAGE_DATA, str(workspace))
    finally:
        if previous is None:
            os.environ.pop("TRTLLM_PRECOMPILED_LINK", None)
        else:
            os.environ["TRTLLM_PRECOMPILED_LINK"] = previous


def test_link_mode_symlinks_the_artifacts(extract_from_precompiled, build_tree, tmp_path):
    _run(extract_from_precompiled, build_tree, tmp_path, link=True)

    lib = Path("tensorrt_llm/libs/libtensorrt_llm.so")
    assert lib.is_symlink()
    assert Path(os.readlink(lib)) == (build_tree / "tensorrt_llm" / "libs" / "libtensorrt_llm.so")
    fmha = Path("3rdparty/fmha_sm100")
    assert fmha.is_symlink()
    assert Path(os.readlink(fmha)) == build_tree / "3rdparty" / "fmha_sm100"


def test_link_mode_keeps_an_existing_fmha_symlink(extract_from_precompiled, build_tree, tmp_path):
    """A link already pointing at this source is left untouched."""
    Path("3rdparty").mkdir()
    os.symlink(build_tree / "3rdparty" / "fmha_sm100", "3rdparty/fmha_sm100")

    _run(extract_from_precompiled, build_tree, tmp_path, link=True)

    assert Path(os.readlink("3rdparty/fmha_sm100")) == build_tree / "3rdparty" / "fmha_sm100"


def test_link_mode_relinks_fmha_when_the_source_changed(
    extract_from_precompiled, build_tree, tmp_path
):
    """A link to a different source is repointed, not kept stale."""
    other = tmp_path / "other-fmha"
    other.mkdir()
    Path("3rdparty").mkdir()
    os.symlink(other, "3rdparty/fmha_sm100")

    _run(extract_from_precompiled, build_tree, tmp_path, link=True)

    assert Path(os.readlink("3rdparty/fmha_sm100")) == build_tree / "3rdparty" / "fmha_sm100"


def test_link_mode_rejects_the_current_checkout(extract_from_precompiled, build_tree, tmp_path):
    """Linking a checkout onto itself would destroy its own artifacts."""
    from setuptools.errors import SetupError

    with pytest.raises(SetupError, match="current directory"):
        _run(extract_from_precompiled, Path.cwd(), tmp_path, link=True)


def test_link_mode_replaces_a_stale_artifact(extract_from_precompiled, build_tree, tmp_path):
    """A real file left by an earlier copy-mode install is not kept."""
    Path("tensorrt_llm/libs").mkdir(parents=True)
    Path("tensorrt_llm/libs/libtensorrt_llm.so").write_text("stale")

    _run(extract_from_precompiled, build_tree, tmp_path, link=True)

    assert Path("tensorrt_llm/libs/libtensorrt_llm.so").is_symlink()


def test_copy_mode_is_unchanged(extract_from_precompiled, build_tree, tmp_path):
    _run(extract_from_precompiled, build_tree, tmp_path, link=False)

    lib = Path("tensorrt_llm/libs/libtensorrt_llm.so")
    assert lib.is_file() and not lib.is_symlink()
    assert lib.read_text() == "so"
    fmha = Path("3rdparty/fmha_sm100")
    assert fmha.is_dir() and not fmha.is_symlink()
    assert (fmha / "__init__.py").is_file()


def test_link_mode_rejects_a_wheel(extract_from_precompiled, build_tree, tmp_path):
    """There is no build tree to point at, so fail instead of copying."""
    from setuptools.errors import SetupError

    wheel = tmp_path / "tensorrt_llm-0.0.0.whl"
    wheel.write_text("")

    with pytest.raises(SetupError, match="TRTLLM_PRECOMPILED_LINK"):
        _run(extract_from_precompiled, wheel, tmp_path, link=True)


# --------------------------------------------------------------------------
# Build-skew warning
# --------------------------------------------------------------------------
def _fake_git(monkeypatch, heads, diff_output="", diff_fails=False):
    """Stub ``git rev-parse`` / ``git diff`` with canned output."""
    import subprocess

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(argv, **kwargs):
        if "rev-parse" in argv:
            return Done(heads[argv[argv.index("-C") + 1]] + "\n")
        assert argv[1] == "diff", argv
        if diff_fails:
            raise subprocess.CalledProcessError(1, argv)
        return Done(diff_output)

    monkeypatch.setattr(subprocess, "run", run)


def test_no_warning_when_the_checkouts_match(warn_on_build_skew, monkeypatch, capfd):
    _fake_git(monkeypatch, {"/src": "a" * 40, ".": "a" * 40})

    warn_on_build_skew("/src")

    assert capfd.readouterr().out == ""


def test_no_warning_when_nothing_native_differs(warn_on_build_skew, monkeypatch, capfd):
    """The two checkouts are meant to differ; only the native inputs matter."""
    _fake_git(monkeypatch, {"/src": "a" * 40, ".": "b" * 40}, diff_output="")

    warn_on_build_skew("/src")

    assert capfd.readouterr().out == ""


def test_warns_when_native_inputs_differ(warn_on_build_skew, monkeypatch, capfd):
    _fake_git(
        monkeypatch,
        {"/src": "a" * 40, ".": "b" * 40},
        diff_output="cpp/one.cu\ncpp/two.cu\nsetup.py\n3rdparty/x\n",
    )

    warn_on_build_skew("/src")

    out = capfd.readouterr().out
    assert "WARNING" in out
    assert "aaaaaaaaaaaa" in out and "bbbbbbbbbbbb" in out
    assert "4 files feeding" in out
    assert "cpp/one.cu" in out and "..." in out
    assert "ABI skew" in out


def test_warns_generically_when_the_diff_cannot_be_taken(warn_on_build_skew, monkeypatch, capfd):
    """Two unrelated clones do not share an object store."""
    _fake_git(monkeypatch, {"/src": "a" * 40, ".": "b" * 40}, diff_fails=True)

    warn_on_build_skew("/src")

    out = capfd.readouterr().out
    assert "WARNING" in out and "could not be inspected" in out


def test_skew_check_is_skipped_outside_a_repo(warn_on_build_skew, monkeypatch, capfd):
    import subprocess

    def run(argv, **kwargs):
        raise subprocess.CalledProcessError(128, argv)

    monkeypatch.setattr(subprocess, "run", run)

    warn_on_build_skew("/src")

    assert "Cannot check for build skew" in capfd.readouterr().out
