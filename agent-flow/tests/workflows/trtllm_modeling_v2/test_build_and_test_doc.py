# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`BUILD_AND_TEST.md`, held to the tree it describes.

The document's whole value is that it points at the places in TensorRT-LLM
that already do each job correctly, rather than restating them. That only
works while the pointers resolve — a reference doc whose paths have moved
costs the reader the trip and then their trust in the rest of the page.

So: every repo path it names must exist, every `build_wheel.py` flag it
documents must still be a flag, and every test list it names must still be a
list. What it *says* about them is prose and is not checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import agent_flow.workflows.trtllm_modeling_v2 as pkg

PACKAGE = Path(pkg.__file__).parent
DOC = PACKAGE / "BUILD_AND_TEST.md"
# agent-flow/tests/workflows/trtllm_modeling_v2 -> the TensorRT-LLM checkout.
REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "scripts" / "build_wheel.py").is_file(),
    reason="not inside a TensorRT-LLM checkout; nothing to resolve against",
)


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_doc_ships_and_is_linked():
    assert DOC.is_file()
    assert "BUILD_AND_TEST.md" in (PACKAGE / "README.md").read_text(encoding="utf-8")


def test_it_is_declared_as_package_data():
    """A doc absent from a non-editable install is a doc that does not exist."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    declared = data["tool"]["setuptools"]["package-data"]["agent_flow.workflows.trtllm_modeling_v2"]
    assert "BUILD_AND_TEST.md" in declared


def test_every_repo_path_it_names_exists():
    missing = []
    for path in sorted(
        set(
            re.findall(
                r"`((?:scripts|tests|jenkins|cpp|tensorrt_llm|\.claude)/[A-Za-z0-9_./*-]+)`",
                _text(),
            )
        )
    ):
        if "*" in path:  # a shape, not a file
            continue
        if not (REPO_ROOT / path.rstrip("/")).exists():
            missing.append(path)
    assert not missing, "BUILD_AND_TEST.md names paths that do not exist:\n  " + "\n  ".join(
        missing
    )


def _section(title: str) -> str:
    """The text of one `## ` section, up to the next one."""
    text = _text()
    start = text.index(f"## {title}")
    rest = text[start + 1 :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def test_every_build_flag_it_documents_is_still_a_flag():
    """The flag table is the part a reader copies verbatim.

    Scoped to the Build section: the multi-node section quotes `srun` flags
    in the same backticked shape, and those belong to Slurm.
    """
    source = (REPO_ROOT / "scripts" / "build_wheel.py").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(--[a-z_]+)`", _section("1. Build")))
    # `-j`, `-c`, `-f` are short forms; check the long ones, which are the
    # names `build_wheel.py` actually declares.
    unknown = sorted(f for f in documented if f'"{f}"' not in source)
    assert documented, "the Build section documents no flags; did it change shape?"
    assert not unknown, f"documented flags build_wheel.py does not declare: {unknown}"


def test_every_test_list_it_names_is_still_a_list():
    lists = REPO_ROOT / "tests/integration/test_lists/test-db"
    named = set(re.findall(r"`(l0_[a-z0-9_]+)\.yml`", _text()))
    assert named, "the scale ladder names no lists; did the table change shape?"
    missing = sorted(n for n in named if not (lists / f"{n}.yml").is_file())
    assert not missing, f"named test lists that no longer exist: {missing}"


def test_the_multi_node_launch_rules_agree_with_the_code_that_emits_them():
    """The doc and `environment.py` must not drift into two different recipes."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import (
        LLMAPI_LAUNCH,
        RunEnvironment,
    )

    text = _text()
    assert LLMAPI_LAUNCH in text
    emitted = RunEnvironment(kind="session_spool", python="/p", world_size=8, spool="/s").command(
        "python x.py"
    )
    assert "mpirun -n 8" in emitted and LLMAPI_LAUNCH in emitted
    assert "`mpirun -n <world> ... trtllm-llmapi-launch`" in text


def test_the_rebuild_table_still_reflects_a_python_only_target():
    """If a target ever needs a native rebuild, this page is the first lie."""
    assert "almost certainly not" in _text()
    assert "Triton is JIT" in _text()
