# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prompt-bundle snapshot (``agent_flow.dump_prompt_bundle``)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_flow import dump_prompt_bundle


@dataclass(frozen=True)
class _Bundle:
    planner: str
    reviewer: str


# Non-ASCII text and trailing whitespace are what a lossy write would mangle.
_BUNDLE = _Bundle(
    planner="You plan.\n\n## Slurm\n\nBootstrap the container → then plan.\n",
    reviewer="You review.  \n\n",
)


def _snapshot(directory):
    return {path.name: path.read_text(encoding="utf-8") for path in directory.iterdir()}


def test_every_role_is_written_verbatim(tmp_path):
    """A snapshot a reader can diff against the prompt modules."""
    directory = tmp_path / "workspace" / "prompts"

    dump_prompt_bundle(_BUNDLE, directory)

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == {
        "planner.md": _BUNDLE.planner.encode("utf-8"),
        "reviewer.md": _BUNDLE.reviewer.encode("utf-8"),
    }


def test_a_later_call_overwrites_the_previous_prompts(tmp_path):
    directory = tmp_path / "prompts"
    dump_prompt_bundle(_Bundle(planner="old planner", reviewer="old reviewer"), directory)

    dump_prompt_bundle(_BUNDLE, directory)

    assert _snapshot(directory) == {"planner.md": _BUNDLE.planner, "reviewer.md": _BUNDLE.reviewer}


def test_a_role_the_bundle_no_longer_has_is_dropped(tmp_path):
    """Two versions' prompts in one directory would read as one bundle's."""
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "retired_role.md").write_text("from an older version\n", encoding="utf-8")
    (directory / "notes.txt").write_text("not a prompt\n", encoding="utf-8")

    dump_prompt_bundle(_BUNDLE, directory)

    assert _snapshot(directory) == {
        "planner.md": _BUNDLE.planner,
        "reviewer.md": _BUNDLE.reviewer,
        "notes.txt": "not a prompt\n",
    }


def test_a_string_directory_is_accepted(tmp_path):
    dump_prompt_bundle(_BUNDLE, str(tmp_path / "prompts"))

    assert (tmp_path / "prompts" / "planner.md").read_text(encoding="utf-8") == _BUNDLE.planner


@pytest.mark.parametrize(
    "bundle",
    [
        pytest.param({"planner": "You plan.", "reviewer": "You review."}, id="mapping"),
        pytest.param(_Bundle, id="dataclass-type"),
    ],
)
def test_anything_but_a_dataclass_instance_is_rejected_before_writing(tmp_path, bundle):
    """A rejected call must leave the snapshot already on disk intact."""
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "retired_role.md").write_text("the previous run's\n", encoding="utf-8")

    with pytest.raises(TypeError, match="dataclass instance"):
        dump_prompt_bundle(bundle, directory)

    assert _snapshot(directory) == {"retired_role.md": "the previous run's\n"}
