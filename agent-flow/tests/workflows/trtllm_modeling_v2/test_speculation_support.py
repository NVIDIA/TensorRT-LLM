# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight refuses a speculation algorithm the gates' launch path cannot build.

When ``quickstart_advanced.py`` has no branch for an algorithm, e.g.
``--spec_decode_algo=DSPARK``, the flag falls through to ``spec_config =
None`` and every quickstart gate of that unit runs without speculation --
surfacing only at a boot or acceptance gate, as a unit that "drafts
nothing". Preflight names the missing branch and the file instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import preflight

from .test_workflow import Script, _install, _workflow

REPO_ROOT = Path(__file__).resolve().parents[4]

_QUICKSTART = """\
def setup_llm(args):
    spec_decode_algo = args.spec_decode_algo.upper() if args.spec_decode_algo else None
    if spec_decode_algo == "MTP":
        spec_config = 1
    elif spec_decode_algo == "EAGLE3":
        spec_config = 2
{extra}    else:
        spec_config = None
    return spec_config
"""
_DSPARK_BRANCH = """    elif spec_decode_algo in ("DSPARK",):
        spec_config = 3
"""

_LLM_ARGS = """\
class MTPDecodingConfig:
    decoding_type: Literal["MTP"] = Field(default="MTP")


class DSparkDecodingConfig:
    decoding_type: Literal["DSpark"] = Field(default="DSpark")


class PrivateDecodingConfig:
    decoding_type: Literal["Private"] = Field(default="Private")


SpeculativeConfig: TypeAlias = Annotated[
    Union[MTPDecodingConfig, DSparkDecodingConfig],
    Field(discriminator="decoding_type"),
]
"""


def _repo(tmp_path: Path, *, dspark_branch: bool) -> Path:
    repo = tmp_path / "repo"
    quickstart = repo / preflight.QUICKSTART_REL
    quickstart.parent.mkdir(parents=True, exist_ok=True)
    quickstart.write_text(
        _QUICKSTART.format(extra=_DSPARK_BRANCH if dspark_branch else ""), encoding="utf-8"
    )
    llm_args = repo / preflight.LLM_ARGS_REL
    llm_args.parent.mkdir(parents=True, exist_ok=True)
    llm_args.write_text(_LLM_ARGS, encoding="utf-8")
    return repo


def _task(repo: Path, features, options: Path | None = None) -> dict:
    accuracy: dict = {}
    if options is not None:
        accuracy["llm_api_options"] = {"default": str(options), "core": str(options)}
    return {"trtllm_repo_path": str(repo), "features": list(features), "accuracy": accuracy}


def _options(tmp_path: Path, decoding_type: str) -> Path:
    path = tmp_path / "spec_llm_api.yaml"
    path.write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": decoding_type}}), encoding="utf-8"
    )
    return path


def test_the_launchers_are_read_for_the_algorithms_they_handle(tmp_path):
    repo = _repo(tmp_path, dspark_branch=True)
    assert preflight.quickstart_spec_algorithms(repo / preflight.QUICKSTART_REL) == {
        "MTP",
        "EAGLE3",
        "DSPARK",
    }
    # Only the union's members: a config class outside it is not accepted.
    assert preflight.llm_api_decoding_types(repo / preflight.LLM_ARGS_REL) == {"MTP", "DSpark"}
    assert preflight.quickstart_spec_algorithms(tmp_path / "absent.py") is None


def test_a_dspark_unit_without_a_quickstart_branch_is_a_named_error(tmp_path):
    repo = _repo(tmp_path, dspark_branch=False)
    with pytest.raises(preflight.UnsupportedSpeculationError) as excinfo:
        preflight.check_speculation_support(_task(repo, ["dspark", "multimodal"]))
    message = str(excinfo.value)
    assert "--spec_decode_algo=DSPARK" in message
    assert str(repo / preflight.QUICKSTART_REL) in message
    assert "no `spec_decode_algo == 'DSPARK'` branch (it handles EAGLE3, MTP)" in message
    assert "spec_config = None" in message
    # Both units that inherit the speculator are named.
    assert "unit(s) dspark, multimodal" in message
    # Not an infrastructure failure: the workflow skips the troubleshooting hint.
    assert isinstance(excinfo.value, preflight.SpecInconsistencyError)


def test_a_tree_with_the_branch_passes_and_says_so(tmp_path):
    repo = _repo(tmp_path, dspark_branch=True)
    notes = preflight.check_speculation_support(
        _task(repo, ["dspark"], _options(tmp_path, "DSpark"))
    )
    assert notes == ["speculation: DSPARK, DSpark supported by the gates' launch paths"]


def test_a_decoding_type_the_llm_api_does_not_accept_is_a_named_error(tmp_path):
    repo = _repo(tmp_path, dspark_branch=True)
    options = _options(tmp_path, "Private")
    with pytest.raises(preflight.UnsupportedSpeculationError) as excinfo:
        preflight.check_speculation_support(_task(repo, ["mtp"], options))
    message = str(excinfo.value)
    assert "`speculative_config.decoding_type: Private`" in message
    assert str(options) in message
    assert (
        f"`SpeculativeConfig` in {repo / preflight.LLM_ARGS_REL} accepts only DSpark, MTP"
        in message
    )


def test_no_speculator_reads_nothing_and_a_missing_launcher_is_only_a_note(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert preflight.check_speculation_support(_task(empty, [])) == []
    notes = preflight.check_speculation_support(_task(empty, ["dspark"]))
    assert len(notes) == 1 and "cannot read" in notes[0] and "DSPARK" in notes[0]


def test_this_tree_handles_the_speculators_upstream_ships():
    """Read the real launchers, so a renamed variable fails here and not in a run."""
    algorithms = preflight.quickstart_spec_algorithms(REPO_ROOT / preflight.QUICKSTART_REL)
    assert algorithms is not None and {"MTP", "EAGLE3"} <= algorithms
    decoding = preflight.llm_api_decoding_types(REPO_ROOT / preflight.LLM_ARGS_REL)
    assert decoding is not None and {"MTP", "Eagle3"} <= decoding


def test_the_workflow_stops_at_preflight_with_the_named_branch(
    tmp_path, task_file, repo, monkeypatch
):
    quickstart = repo / preflight.QUICKSTART_REL
    quickstart.parent.mkdir(parents=True, exist_ok=True)
    quickstart.write_text(_QUICKSTART.format(extra=""), encoding="utf-8")
    spec = yaml.safe_load(task_file.read_text())
    spec["features"] = ["dspark"]
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")

    workflow = _workflow(tmp_path, dry_run=True)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert calls == []
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "preflight failed" in journal
    assert "no `spec_decode_algo == 'DSPARK'` branch" in journal
    assert "infrastructure failure" not in journal
