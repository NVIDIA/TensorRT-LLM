"""Tests for staircase-bringup's ``task.yaml`` schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_bringup.task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    load_and_validate_task_yaml,
    target_relpath,
    world_size,
)


def _write(tmp_path: Path, data: dict, name: str = "task.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _valid(tmp_path: Path) -> dict:
    """A minimal spec whose three paths exist (they point at tmp_path itself)."""
    return {
        "reference_code_path": str(tmp_path),
        "checkpoint_path": str(tmp_path),
        "trtllm_repo_path": str(tmp_path),
        "target": {
            "family": "deepseek_v3",
            "checkpoint": "r1_0528_nvfp4",
            "arch": "sm_103",
            "parallel": "dep4",
        },
    }


def test_valid_spec_round_trips(tmp_path: Path):
    data = load_and_validate_task_yaml(_write(tmp_path, _valid(tmp_path)))
    assert data["completion_criteria"] == []
    assert data["implements_tips"] == []
    assert not has_slurm_environment(data)


def test_world_size_reads_the_parallel_segment(tmp_path: Path):
    for segment, expected in (("tp1", 1), ("tep4", 4), ("dep8", 8), ("tp16", 16)):
        spec = _valid(tmp_path)
        spec["target"]["parallel"] = segment
        data = load_and_validate_task_yaml(_write(tmp_path, spec))
        assert world_size(data) == expected


def test_target_relpath_is_the_in_tree_layout(tmp_path: Path):
    data = load_and_validate_task_yaml(_write(tmp_path, _valid(tmp_path)))
    assert target_relpath(data) == (
        "tensorrt_llm/_torch/staircase/models/deepseek_v3/targets/r1_0528_nvfp4/sm_103/dep4"
    )


@pytest.mark.parametrize("segment", ["single", "tp0", "dp4", "tep", "4tep", "tep4x"])
def test_malformed_parallel_segment_is_rejected(tmp_path: Path, segment: str):
    """'single' is deliberately not a spelling; a segment always carries its rank count."""
    spec = _valid(tmp_path)
    spec["target"]["parallel"] = segment
    with pytest.raises(TaskSchemaError, match="parallel"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


@pytest.mark.parametrize("arch", ["sm100", "b200", "sm_", "SM_100"])
def test_malformed_arch_is_rejected(tmp_path: Path, arch: str):
    spec = _valid(tmp_path)
    spec["target"]["arch"] = arch
    with pytest.raises(TaskSchemaError, match="arch"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


def test_missing_target_is_rejected(tmp_path: Path):
    spec = _valid(tmp_path)
    del spec["target"]
    with pytest.raises(TaskSchemaError, match="target"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


def test_every_problem_is_reported_at_once(tmp_path: Path):
    """One pass must surface all gaps, not the first one."""
    spec = {
        "reference_code_path": "/does/not/exist",
        "target": {"family": "f", "arch": "nope", "parallel": "single"},
    }
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(exc.value)
    for fragment in ("checkpoint_path", "trtllm_repo_path", "non-existent", "arch", "parallel"):
        assert fragment in message, fragment


def test_anchor_without_a_source_is_rejected(tmp_path: Path):
    """An anchor with no written source cannot serve as a release criterion."""
    spec = _valid(tmp_path)
    spec["accuracy_anchor"] = {"benchmark": "mmlu", "score": 95.07, "tol": 5.0, "source": "  "}
    with pytest.raises(TaskSchemaError, match="source"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


def test_anchor_score_must_be_numeric(tmp_path: Path):
    spec = _valid(tmp_path)
    spec["accuracy_anchor"] = {
        "benchmark": "mmlu",
        "score": "about ninety five",
        "tol": 5.0,
        "source": "model card",
    }
    with pytest.raises(TaskSchemaError, match="score"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


def test_slurm_environment_is_detected_and_validated(tmp_path: Path):
    spec = _valid(tmp_path)
    spec["slurm-environment"] = {"slurm_partition": "batch", "docker_image": "/img.sqsh"}
    data = load_and_validate_task_yaml(_write(tmp_path, spec))
    assert has_slurm_environment(data)

    spec["slurm-environment"] = {"slurm_partition": "batch"}
    with pytest.raises(TaskSchemaError, match="docker_image"):
        load_and_validate_task_yaml(_write(tmp_path, spec))


def test_missing_file_and_bad_yaml(tmp_path: Path):
    with pytest.raises(TaskSchemaError, match="not found"):
        load_and_validate_task_yaml(tmp_path / "absent.yaml")

    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="not valid YAML"):
        load_and_validate_task_yaml(bad)


def test_shipped_example_parses(tmp_path: Path):
    """The committed example must itself be schema-valid once its paths resolve."""
    import agent_flow.workflows.staircase_bringup as pkg

    example = Path(pkg.__file__).parent / "task.example.yaml"
    spec = yaml.safe_load(example.read_text(encoding="utf-8"))
    for field in ("reference_code_path", "checkpoint_path", "trtllm_repo_path"):
        spec[field] = str(tmp_path)
    data = load_and_validate_task_yaml(_write(tmp_path, spec))
    assert world_size(data) == 4
    assert data["accuracy_anchor"]["benchmark"] == "mmlu"
