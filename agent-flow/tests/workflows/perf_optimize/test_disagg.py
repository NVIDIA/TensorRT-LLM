"""Tests for the disagg block: validation, backfill, and the tuning seed.

The property under test throughout is the authority rule: the disagg
harness config owns the measurement conditions, ``task.yaml`` owns only
the campaign knobs, and the copy runs one way into the workspace's
resolved spec.
"""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.perf_optimize import task_schema
from agent_flow.workflows.perf_optimize.prompts import build_perf_optimize_prompts
from agent_flow.workflows.perf_optimize.prompts._common import DISAGG_CAMPAIGN


def _flat(text: str) -> str:
    """Whitespace-collapsed view of a prompt fragment.

    The assertions below are about what the prose *says*; hard-wrapping it
    differently must not fail them.
    """
    return " ".join(text.split())


HARNESS = {
    "slurm": {"partition": "p", "account": "a", "job_time": "02:00:00"},
    "benchmark": {
        "mode": "e2e",
        "multi_round": 5,
        "streaming": True,
        "concurrency_list": "16 64",
        "input_length": 1024,
        "output_length": 2048,
        "dataset_file": "/data/random-1k2k.json",
    },
    "hardware": {"gpus_per_node": 4, "num_ctx_servers": 1, "num_gen_servers": 2},
    "environment": {"model_path": "/ckpt", "trtllm_repo": "/repo"},
    "profiling": {
        "nsys_on": False,
        "ctx_profile_range": "10-30",
        "gen_profile_range": "200-250",
    },
    "worker_config": {
        "ctx": {"tensor_parallel_size": 4, "max_batch_size": 16},
        "gen": {"tensor_parallel_size": 8, "max_batch_size": 256},
    },
}


def _write_harness(tmp_path, overrides: dict | None = None):
    cfg = yaml.safe_load(yaml.safe_dump(HARNESS))  # deep copy
    for key, value in (overrides or {}).items():
        section, _, field = key.partition(".")
        if field:
            cfg[section][field] = value
        else:
            cfg[section] = value
    path = tmp_path / "harness.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_task(tmp_path, extra: dict | None = None):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    data = {"checkpoint_path": str(ckpt), "trtllm_repo_path": str(repo)}
    data.update(extra or {})
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------- the backfill


def test_concurrency_is_the_total_not_the_per_gen_server_value(tmp_path):
    """run_benchmark.sh drives the client at list_entry * num_gen_servers.

    The resolved spec must carry that product: it is what the client
    actually runs, what names the concurrency_<c> result directory, and
    what an aggregate campaign's `concurrency` means.
    """
    harness = _write_harness(tmp_path)
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {"disagg": {"config": str(harness)}})
    )
    assert data["benchmark"]["concurrency"] == [32, 128]
    # num_prompts = concurrency * multi_round, paired index-by-index.
    assert data["benchmark"]["num_prompts"] == [160, 640]
    assert task_schema.is_curve_mode(data)


def test_a_condition_the_user_wrote_that_disagrees_is_an_error(tmp_path):
    """Not a silent overwrite.

    The harness config owns the measurement conditions, but a `task.yaml`
    whose stated operating point is quietly replaced is a file you cannot
    read: "my setting did nothing" is exactly the failure this avoids. So
    an omitted field is filled, and a contradicting one stops the run
    naming both values.
    """
    harness = _write_harness(tmp_path)
    with pytest.raises(task_schema.TaskSchemaError, match="contradicts the disagg config"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    "disagg": {"config": str(harness)},
                    "benchmark": {"concurrency": 999},
                },
            )
        )

    # The same value the harness implies is not a conflict, and what was
    # filled is recorded so "where did this come from" is answerable.
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                "disagg": {"config": str(harness)},
                "benchmark": {"concurrency": [32, 128]},
            },
        )
    )
    assert data["benchmark"]["random_input_len"] == 1024
    filled = " ".join(data["disagg"]["filled_from_disagg_config"])
    assert "random_input_len" in filled and "concurrency" not in filled.split("num_prompts")[0]


def test_profiling_reduces_to_nsys_on_the_gen_window(tmp_path):
    """The harness only wraps workers in nsys; torch/ncu have no path in it."""
    harness = _write_harness(tmp_path)
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                "disagg": {"config": str(harness)},
                "profile": {"methods": ["nsys", "torch", "ncu"], "kernel_coverage": {}},
            },
        )
    )
    assert data["profile"]["methods"] == ["nsys"]
    assert data["profile"]["nsys_iter_range"] == "200-250"
    assert "kernel_coverage" not in data["profile"]
    notes = " ".join(data["disagg"]["filled_from_disagg_config"])
    assert "torch" in notes and "ncu" in notes


def test_focus_concurrencies_validate_against_the_backfilled_points(tmp_path):
    """The backfill must land before the blocks validated against it."""
    harness = _write_harness(tmp_path)
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                "disagg": {"config": str(harness)},
                "optimize": {"focus_concurrencies": [128]},
            },
        )
    )
    assert data["optimize"]["focus_concurrencies"] == [128]

    with pytest.raises(task_schema.TaskSchemaError, match="focus_concurrencies"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    "disagg": {"config": str(harness)},
                    # 64 is a per-gen-server entry, not a measured point.
                    "optimize": {"focus_concurrencies": [64]},
                },
            )
        )


# --------------------------------------------------------------- validation


def test_missing_config_key_is_rejected(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="disagg.config"):
        task_schema.load_and_validate_task_yaml(_write_task(tmp_path, {"disagg": {}}))


def test_nonexistent_config_is_rejected(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="not a file"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {"disagg": {"config": str(tmp_path / "nope.yaml")}})
        )


def test_extra_llm_api_options_cannot_be_combined(tmp_path):
    """Two seeds for one live tuning file — the disagg one would silently win."""
    harness = _write_harness(tmp_path)
    extra = tmp_path / "extra.yaml"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(task_schema.TaskSchemaError, match="cannot be combined"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    "disagg": {"config": str(harness)},
                    "extra_llm_api_options": str(extra),
                },
            )
        )


# --------------------------------------------------------------- tuning seed


# --------------------------------------------------------------- the prompts


def test_the_disagg_section_is_composed_only_for_a_disagg_campaign():
    """Selective composition, the same way slurm / sol / kernel_coverage work.

    An aggregate campaign must not carry it at all: a section that is
    always present and gated on "check task.yaml first" makes every
    aggregate run pay for it and turns a deployment-time fact into a
    per-turn inference the agent can get wrong.
    """
    aggregate = build_perf_optimize_prompts()
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa", "reporter"):
        assert DISAGG_CAMPAIGN not in getattr(aggregate, role)

    disagg_bundle = build_perf_optimize_prompts(include_disagg=True)
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa"):
        assert DISAGG_CAMPAIGN in getattr(disagg_bundle, role)
    # The reporter only synthesizes artifacts, the projector launches nothing.
    assert DISAGG_CAMPAIGN not in disagg_bundle.reporter
    assert DISAGG_CAMPAIGN not in disagg_bundle.projector


def test_the_disagg_section_is_composed_last_so_its_overrides_win():
    """It supersedes the single-server lifecycle stated earlier in the prompt.

    Ordering is the whole mechanism: the section only works if the role
    reads it after the guidance it replaces.
    """
    bundle = build_perf_optimize_prompts(include_disagg=True)
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa"):
        assert getattr(bundle, role).rstrip().endswith(DISAGG_CAMPAIGN.rstrip())


def test_the_disagg_section_freezes_the_topology():
    flat = _flat(DISAGG_CAMPAIGN)
    assert "frozen for this campaign" in flat
    assert "is a REJECT" in flat
    # num_gpus is the sum over roles, not one server's world size.
    assert "sum over roles" in flat


def test_the_disagg_section_scopes_profiling_to_nsys_on_workers():
    flat = _flat(DISAGG_CAMPAIGN)
    assert "no path through this harness" in flat
    assert "workers only" in flat
    assert "communication" in flat


# ------------------------------------------------- the two-layer prompt override


def test_stage_prompts_state_the_mode_so_the_system_prompt_section_wins(tmp_path):
    """The per-stage instruction also names trtllm-serve, and it arrives last.

    Without this the agent gets a contradiction: the system prompt says
    "submit a job", the more specific stage instruction says "launch
    trtllm-serve with --extra_llm_api_options and poll it".
    """
    from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

    harness = _write_harness(tmp_path)
    task = _write_task(tmp_path, {"disagg": {"config": str(harness)}})
    wf = PerfOptimizeWorkflow.__new__(PerfOptimizeWorkflow)
    wf._task_data = lambda: task_schema.load_and_validate_task_yaml(task)

    directive = wf._campaign_directive()
    assert "DISAGGREGATED" in directive
    assert "replaces all of it" in directive
    assert str(harness) in directive
    # It must name the guidance it overrides, or a reader cannot tell what to skip.
    assert "trtllm-serve" in directive


# ------------------------------------------------- profiling wording
