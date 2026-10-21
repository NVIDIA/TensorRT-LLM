# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decode CUDA-graph capture list per unit, whichever gate builds the engine.

The accuracy gate's ``trtllm-eval`` reads the unit's options file, and
``quickstart_advanced.py`` (boot, acceptance, envelope) builds its
``CudaGraphConfig`` from its own flags. Unaligned, one unit captured 19 padded
batches under accuracy and 34 unpadded ones under boot. These tests hold the
resolver to the LLM API's rule, the quickstart flags to the resolved list,
preflight to refusing what still disagrees, and the survey turn to printing
the lists it plans against.
"""

from __future__ import annotations

import ast
import json
import math
import re
import shlex
from pathlib import Path
from typing import List

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import plan_drafter
from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.stages.plan import PlanStage
from agent_flow.workflows.trtllm_modeling_v2.spec.variants import variant_spec

REPO_ROOT = Path(__file__).resolve().parents[4]

#: What the LLM API captures with nothing set, and with padding at its default limit.
DEFAULT_34 = (*range(1, 33), 64, 128)
PADDED_19 = (1, 2, 4, *range(8, 129, 8))


def _options(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.llm_api.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _task(tmp_path: Path, *, options: dict | None = None, runtime: dict | None = None, **extra):
    task: dict = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": ["mtp"],
        "runtime": runtime
        if runtime is not None
        else {"cuda_graph": True, "extra_flags": ["--max_batch_size=128"]},
        "accuracy": {"benchmark": "gsm8k"},
    }
    if options is not None:
        task["accuracy"]["llm_api_options"] = {
            unit: str(_options(tmp_path, unit, data)) for unit, data in options.items()
        }
    task.update(extra)
    return task


_PADDED_128 = {"cuda_graph_config": {"enable_padding": True, "max_batch_size": 128}}


# ------------------------------------------------------------- the LLM API rule


def test_the_llm_api_default_is_34_unpadded_batches():
    assert gates.llm_api_batch_sizes() == DEFAULT_34
    assert gates.LLM_API_DEFAULT_CAPTURE.batch_sizes == DEFAULT_34
    assert not gates.LLM_API_DEFAULT_CAPTURE.enable_padding
    assert gates.LLM_API_DEFAULT_CAPTURE.describe() == "34 batches [1-32, 64, 128], unpadded"


def test_padding_generates_the_19_and_steps_of_64_above_128():
    assert gates.llm_api_batch_sizes(None, 128, True) == PADDED_19
    assert gates.llm_api_batch_sizes(None, 300, True)[-4:] == (128, 192, 256, 300)
    assert gates.llm_api_batch_sizes(None, 512, False)[-3:] == (128, 256, 512)
    assert gates.llm_api_batch_sizes(None, 100, False)[-3:] == (32, 64, 100)


def test_explicit_batch_sizes_are_sorted_and_must_end_at_max_batch_size():
    assert gates.llm_api_batch_sizes([8, 1, 4], 0, False) == (1, 4, 8)
    assert gates.llm_api_batch_sizes([8, 1, 4], 8, True) == (1, 4, 8)
    with pytest.raises(ValueError, match="is not max"):
        gates.llm_api_batch_sizes([1, 4], 8, False)


def _checkout_generator():
    """``BaseCudaGraphConfig._generate_cuda_graph_batch_sizes`` from this checkout, or None."""
    path = REPO_ROOT / "tensorrt_llm" / "llmapi" / "llm_args.py"
    if not path.is_file():
        return None
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == "BaseCudaGraphConfig":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == (
                    "_generate_cuda_graph_batch_sizes"
                ):
                    item.decorator_list = []
                    namespace: dict = {"math": math, "List": List}
                    module = ast.Module(body=[item], type_ignores=[])
                    exec(compile(module, str(path), "exec"), namespace)
                    return namespace[item.name]
    return None


def test_the_restated_rule_is_the_checkouts_own():
    """Read the real generator, so a changed LLM API rule fails here and not in a review."""
    generate = _checkout_generator()
    if generate is None:
        pytest.skip("this checkout carries no tensorrt_llm/llmapi/llm_args.py")
    for limit in [*range(1, 300), 512, 1000, 2048]:
        for padding in (False, True):
            assert gates.llm_api_batch_sizes(None, limit, padding) == tuple(
                generate(limit, padding)
            ), (limit, padding)


def test_a_batch_list_is_printed_with_its_runs_joined():
    assert gates.format_batches(DEFAULT_34) == "1-32, 64, 128"
    assert gates.format_batches((1, 2, 4, 8)) == "1, 2, 4, 8"
    assert gates.format_batches(()) == ""


# ----------------------------------------------------------------- per path


def test_options_without_a_cuda_graph_config_get_the_llm_api_default(tmp_path):
    capture = gates.options_capture({"max_num_tokens": 8192}, "core.yaml")
    assert capture == gates.LLM_API_DEFAULT_CAPTURE
    assert "CudaGraphConfig()" in capture.source and "core.yaml" in capture.source


def test_a_null_cuda_graph_config_captures_nothing():
    assert gates.options_capture({"cuda_graph_config": None}) is None


def test_a_draft_len_schedule_pads_and_adds_its_thresholds():
    """``TorchLlmArgs`` turns padding on and merges the schedule's keys into the list."""
    capture = gates.options_capture(
        {
            "cuda_graph_config": {"max_batch_size": 16},
            "speculative_config": {"draft_len_schedule": {6: 4, 12: 2, 64: 1}},
        }
    )
    assert capture.enable_padding
    assert capture.batch_sizes == tuple(range(1, 17))
    capture = gates.options_capture(
        {
            "cuda_graph_config": {"enable_padding": True, "max_batch_size": 32},
            "speculative_config": {"draft_len_schedule": {6: 4, 12: 2}},
        }
    )
    assert capture.batch_sizes == (1, 2, 4, 6, 8, 12, 16, 24, 32)


def test_the_quickstart_list_comes_from_its_flags_alone():
    assert gates.quickstart_capture(["--max_batch_size=8"]) is None
    assert gates.quickstart_capture(["--use_cuda_graph", "--max_batch_size=8"]) == (
        gates.LLM_API_DEFAULT_CAPTURE
    )
    one_string = gates.quickstart_capture(
        ["--use_cuda_graph", "--cuda_graph_padding_enabled", "--cuda_graph_batch_sizes 4 1 2"]
    )
    as_words = gates.quickstart_capture(
        ["--use_cuda_graph", "--cuda_graph_batch_sizes", "1", "2", "4", "--padding_after=1"]
    )
    assert one_string.batch_sizes == as_words.batch_sizes == (1, 2, 4)
    assert one_string.enable_padding and not as_words.enable_padding
    # argparse keeps the last spelling.
    last = gates.quickstart_capture(
        ["--use_cuda_graph", "--cuda_graph_batch_sizes 1 2", "--cuda_graph_batch_sizes=8"]
    )
    assert last.batch_sizes == (8,)


# ---------------------------------------------------------------- the resolver


def test_the_options_file_decides_when_it_sets_a_cuda_graph_config(tmp_path):
    task = _task(
        tmp_path,
        options={"mtp": _PADDED_128},
        runtime={"cuda_graph": True, "extra_flags": ["--max_batch_size=64"]},
    )
    capture = gates.resolve_capture_list(task, "mtp")
    assert capture.batch_sizes == PADDED_19 and capture.enable_padding
    assert "mtp.llm_api.yaml" in capture.source


def test_without_one_the_runtime_flags_decide_by_the_llm_api_rule(tmp_path):
    task = _task(
        tmp_path,
        options={"mtp": {"max_num_tokens": 8192}},
        runtime={
            "cuda_graph": True,
            "extra_flags": ["--cuda_graph_padding_enabled", "--cuda_graph_batch_sizes 1 2 4"],
        },
    )
    capture = gates.resolve_capture_list(task, "mtp")
    assert capture.batch_sizes == (1, 2, 4) and capture.enable_padding
    assert capture.source == "the runtime block's extra_flags"


def test_with_neither_the_list_is_the_llm_api_default(tmp_path):
    capture = gates.resolve_capture_list(_task(tmp_path), "core")
    assert capture == gates.LLM_API_DEFAULT_CAPTURE
    assert "neither the unit (no options file) nor the runtime block" in capture.source


def test_a_null_cuda_graph_config_resolves_to_no_list(tmp_path):
    task = _task(tmp_path, options={"core": {"cuda_graph_config": None}})
    assert gates.resolve_capture_list(task, "core") is None


# ---------------------------------------------------------------- the flags


def test_runtime_flags_spell_the_list_under_cuda_graphs_only():
    capture = gates.CaptureList(PADDED_19, True)
    flags = gates.runtime_flags({"cuda_graph": True}, capture=capture)
    assert flags == [
        "--use_cuda_graph",
        "--cuda_graph_padding_enabled",
        "--cuda_graph_batch_sizes " + " ".join(str(b) for b in PADDED_19),
    ]
    assert gates.runtime_flags({"cuda_graph": False}, capture=capture) == []
    # Without a list the flags are what they always were.
    assert gates.runtime_flags({"cuda_graph": True}) == ["--use_cuda_graph"]


def test_a_spelling_the_runtime_block_carries_is_not_repeated():
    capture = gates.CaptureList((1, 2, 4), True)
    flags = gates.runtime_flags(
        {"cuda_graph": True, "extra_flags": ["--cuda_graph_batch_sizes=8"]}, capture=capture
    )
    assert flags == [
        "--use_cuda_graph",
        "--cuda_graph_padding_enabled",
        "--cuda_graph_batch_sizes=8",
    ]
    flags = gates.runtime_flags(
        {"cuda_graph": True, "extra_flags": ["--cuda_graph_padding_enabled"]}, capture=capture
    )
    assert flags == [
        "--use_cuda_graph",
        "--cuda_graph_batch_sizes 1 2 4",
        "--cuda_graph_padding_enabled",
    ]


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=(0,),
        world_size=1,
    )


def test_boot_acceptance_and_accuracy_capture_the_same_list(tmp_path):
    """The live defect: a padded options file, and a runtime block that says nothing of it."""
    options = {**_PADDED_128, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    task = _task(
        tmp_path,
        options={"mtp": options},
        runtime={"cuda_graph": True, "extra_flags": ["--max_batch_size=128"]},
    )
    words = gates.quickstart_words(
        task, _env(), unit="mtp", features=["mtp"], max_tokens=16, member=None
    )
    quickstart = gates.quickstart_capture(words)
    accuracy = gates.options_capture(
        gates.load_llm_api_options(gates.resolve_llm_api_options(task["accuracy"], "mtp"))
    )
    assert quickstart == accuracy
    assert quickstart.batch_sizes == PADDED_19 and quickstart.enable_padding
    assert "--spec_decode_max_draft_len=3" in words


def test_the_witness_engine_captures_the_unit_list(tmp_path):
    """The witness builds its own quickstart command; it runs the list the other gates run."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.witness import witness_command

    task = _task(tmp_path, options={"mtp": _PADDED_128})
    words = shlex.split(witness_command(task, _env(), unit="mtp", features=["mtp"]))
    witness = gates.quickstart_capture(words[: words.index("--prompt")])
    boot = gates.quickstart_words(
        task, _env(), unit="mtp", features=["mtp"], max_tokens=16, member=None
    )
    assert witness == gates.quickstart_capture(boot)
    assert witness.batch_sizes == PADDED_19 and witness.enable_padding


def _driver_runner(seen: list[dict]):
    def run(command, env=None, timeout=0):
        spec_path = re.search(r"--spec (\S+)", command).group(1)
        seen.append(json.loads(Path(spec_path).read_text(encoding="utf-8")))
        return CommandResult(returncode=1, stdout="")

    return run


def test_the_boot_gate_engine_is_built_with_the_unit_list(tmp_path):
    task = _task(tmp_path, options={"mtp": _PADDED_128})
    seen: list[dict] = []
    gates.run_boot_gate(
        _env(),
        task_data=task,
        runner=_driver_runner(seen),
        unit="mtp",
        allow_dirty=True,
        workdir=tmp_path / "gates",
    )
    argv = seen[0]["quickstart_argv"]
    assert "--cuda_graph_padding_enabled" in argv
    start = argv.index("--cuda_graph_batch_sizes") + 1
    assert tuple(int(w) for w in argv[start : start + len(PADDED_19)]) == PADDED_19


def test_a_variant_gate_resolves_the_variant_list(tmp_path):
    task = _task(
        tmp_path,
        options={"core": _PADDED_128},
        variants={
            "nopad": {
                "note": "unpadded capture",
                "llm_api_options": {"cuda_graph_config": {"enable_padding": False}},
            }
        },
    )
    spec = variant_spec(task, "nopad", world_size=1, options_dir=tmp_path / "variants")
    capture = gates.resolve_capture_list(spec, "nopad")
    assert capture.batch_sizes == DEFAULT_34 and not capture.enable_padding
    by_name = {c.name: c for c in gates.launch_captures(task)}
    assert by_name["nopad"].unit == "core"
    assert by_name["nopad"].quickstart == by_name["nopad"].llm_api == capture
    assert by_name["core"].llm_api.batch_sizes == PADDED_19


# ------------------------------------------------------------------- preflight


def test_preflight_refuses_runtime_flags_against_the_options_file(tmp_path):
    task = _task(
        tmp_path,
        options={"core": _PADDED_128},
        runtime={"cuda_graph": True, "extra_flags": ["--cuda_graph_batch_sizes 1 2 4"]},
    )
    with pytest.raises(preflight.CaptureListMismatchError) as excinfo:
        preflight.check_capture_lists(task)
    message = str(excinfo.value)
    assert "`core`" in message
    assert "3 batches [1, 2, 4], padded" in message
    assert "19 batches [1, 2, 4, 8, 16, 24," in message
    assert "core.llm_api.yaml" in message
    assert "`cuda_graph_config` in the unit's options file" in message
    assert isinstance(excinfo.value, preflight.SpecInconsistencyError)


def test_preflight_refuses_runtime_flags_where_the_options_file_states_none(tmp_path):
    """``trtllm-eval`` then captures ``CudaGraphConfig()``'s default, not the flags' list."""
    task = _task(
        tmp_path,
        runtime={"cuda_graph": True, "extra_flags": ["--cuda_graph_padding_enabled"]},
    )
    with pytest.raises(preflight.CaptureListMismatchError) as excinfo:
        preflight.check_capture_lists(task)
    message = str(excinfo.value)
    # The flags pad `CudaGraphConfig`'s generated list; trtllm-eval builds the bare default.
    assert "would capture 19 batches [1, 2, 4, 8, 16," in message
    assert "would capture 34 batches [1-32, 64, 128], unpadded" in message
    assert "CudaGraphConfig()" in message


def test_preflight_names_a_variant_that_disagrees(tmp_path):
    task = _task(
        tmp_path,
        options={"core": _PADDED_128},
        variants={
            "small": {
                "note": "a runtime list of its own",
                "runtime": {"extra_flags": ["--cuda_graph_batch_sizes 1 2"]},
            }
        },
    )
    with pytest.raises(preflight.CaptureListMismatchError, match="`small \\(core\\)`"):
        preflight.check_capture_lists(task)


def test_preflight_refuses_a_cuda_graph_config_the_llm_api_refuses(tmp_path):
    bad = {"cuda_graph_config": {"batch_sizes": [1, 2], "max_batch_size": 8}}
    with pytest.raises(preflight.CaptureListMismatchError, match="is not max"):
        preflight.check_capture_lists(_task(tmp_path, options={"core": bad}))


def test_aligned_units_pass_with_their_list_stated(tmp_path):
    task = _task(tmp_path, options={"core": _PADDED_128, "mtp": _PADDED_128})
    notes = preflight.check_capture_lists(task)
    assert notes == [
        "cuda graphs: core, mtp: 19 batches [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, "
        "88, 96, 104, 112, 120, 128], padded, at boot, acceptance, envelope and accuracy alike"
    ]


def test_an_eager_path_is_a_note_not_a_refusal(tmp_path):
    """The schema's default runtime has CUDA graphs off; ``trtllm-eval`` captures anyway."""
    notes = preflight.check_capture_lists(_task(tmp_path, runtime={}))
    assert len(notes) == 2
    assert all("run eager (runtime `cuda_graph` off)" in note for note in notes)
    notes = preflight.check_capture_lists(
        _task(tmp_path, options={"core": {"cuda_graph_config": None}}, features=[])
    )
    assert len(notes) == 1 and "accuracy (trtllm-eval) runs eager" in notes[0]


def test_run_preflight_stops_before_probing_the_host(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    task = _task(
        tmp_path,
        options={"core": _PADDED_128},
        runtime={"cuda_graph": True, "extra_flags": ["--cuda_graph_batch_sizes 1 2 4"]},
        checkpoint_path=str(checkpoint),
        arch="sm_103",
        features=[],
        environment={
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": "/scratch/x", "CLAUDE_CODE_TMPDIR": "/scratch/x"},
        },
    )

    def runner(argv, env=None, timeout=0):  # pragma: no cover - must not run
        raise AssertionError(f"probed the host before checking the spec: {argv}")

    with pytest.raises(preflight.CaptureListMismatchError):
        preflight.run_preflight(task, runner=runner)


# --------------------------------------------------------------- the survey turn


def test_the_survey_turn_is_given_every_list_it_plans_against(tmp_path):
    task = _task(
        tmp_path,
        options={"core": _PADDED_128, "mtp": {}},
        runtime={"cuda_graph": True, "extra_flags": []},
        variants={
            "kv_fp8": {"note": "fp8 KV", "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}},
            "nopad": {
                "note": "unpadded",
                "llm_api_options": {"cuda_graph_config": {"enable_padding": False}},
            },
        },
        perf={
            "enabled": True,
            "configs": [
                {"name": "tp1", "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 16]}},
                {
                    "name": "fixed",
                    "workload": {"concurrency": [64]},
                    "llm_api_options": {"cuda_graph_config": {"enable_padding": True}},
                },
            ],
        },
    )
    block = plan_drafter.capture_lists_block(task)
    assert block.startswith(plan_drafter.CAPTURE_LISTS_HEADING)
    lines = block.splitlines()
    core = next(line for line in lines if line.startswith("- `core`"))
    assert "19 batches [1, 2, 4, 8, 16," in core and "padded" in core
    assert "core.llm_api.yaml" in core
    mtp = next(line for line in lines if line.startswith("- `mtp`"))
    assert "34 batches [1-32, 64, 128], unpadded" in mtp
    nopad = next(line for line in lines if line.startswith("- variant `nopad` (unit `core`)"))
    assert "34 batches [1-32, 64, 128], unpadded" in nopad
    assert "- variants kv_fp8: their unit's list" in lines
    default = next(line for line in lines if line.startswith("- LLM API default"))
    assert default.endswith("34 batches [1-32, 64, 128], unpadded")
    assert (
        "- tune `tp1`: 16 batches [1-16], padded, from `capture_batch_sizes` up to concurrency 16"
        in lines
    )
    fixed = next(line for line in lines if line.startswith("- tune `fixed`"))
    assert fixed.endswith("padded, from its `llm_api_options`")


def test_without_perf_the_block_says_tune_measures_nothing(tmp_path):
    block = plan_drafter.capture_lists_block(_task(tmp_path))
    assert "- tune: `perf` is off, so no serving config is measured" in block


def test_the_system_prompt_points_the_envelope_at_the_printed_lists():
    prompt = plan_drafter.SYSTEM_PROMPT
    assert f"`{plan_drafter.CAPTURE_LISTS_HEADING}`" in prompt
    assert "The capture batches are read, never recalled." in prompt
    assert "## Open risks" in prompt


class _Survey(PlanStage):
    def __init__(self, tmp_path: Path, task_data: dict) -> None:
        self.workspace = tmp_path / "ws"
        self.task_path = tmp_path / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self._surface_block = "## Surface survey\n"
        self._runtime_block = ""
        self.task_data = task_data

    def _report_instruction(self, role: str) -> str:
        return f"report as {role}"


def test_the_plan_drafter_turn_carries_the_block(tmp_path):
    task = _task(tmp_path, options={"core": _PADDED_128})
    prompt = _Survey(tmp_path, task)._plan_drafter_prompt()
    assert plan_drafter.CAPTURE_LISTS_HEADING in prompt
    assert "19 batches [1, 2, 4, 8, 16," in prompt
