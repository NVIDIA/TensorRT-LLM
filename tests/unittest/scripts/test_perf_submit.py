#!/usr/bin/env python3
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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pytest_split.algorithms import LeastDurationAlgorithm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CI_SUBMIT_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py"
SUBMIT_PATHS = (
    CI_SUBMIT_PATH,
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
EXAMPLE_SUBMIT_PATH = REPO_ROOT / "examples" / "disaggregated" / "slurm" / "benchmark" / "submit.py"


class _FakePytestItem:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid

    def __str__(self) -> str:
        return self.nodeid


def _load_module(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(f"perf_submit_{path.parent.name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=SUBMIT_PATHS, ids=("ci", "local"))
def submit_module(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(request.param, monkeypatch)


@pytest.fixture
def ci_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(CI_SUBMIT_PATH, monkeypatch)


@pytest.fixture
def example_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(EXAMPLE_SUBMIT_PATH, monkeypatch)


@pytest.fixture
def local_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The local harness only; the CI submit.py has no SLURM GRES helpers."""
    return _load_module(SUBMIT_PATHS[1], monkeypatch)


def _get_benchmark_config(module: ModuleType, concurrency):
    config = {"benchmark": {"concurrency_list": concurrency}}
    if Path(module.__file__).parent.name == "local":
        return module.get_benchmark_config(config, "gen_only")
    return module.get_benchmark_config(config)


@pytest.mark.parametrize("concurrency", ("1", 1, "4301"))
def test_get_benchmark_config_accepts_positive_integer(submit_module: ModuleType, concurrency):
    benchmark_config = _get_benchmark_config(submit_module, concurrency)

    assert benchmark_config["concurrency"] == int(concurrency)


def _select_ci_test_case_line(
    ci_submit_module: ModuleType,
    tmp_path: Path,
    pytest_options: str,
    split_group: int,
) -> str:
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text(
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300-kimi]\n",
        encoding="utf-8",
    )
    script_prefix_lines = [
        f'export pytestCommand="pytest --splitting-algorithm least_duration {pytest_options}"'
    ]
    return ci_submit_module.select_test_case_line(
        test_list_path,
        tmp_path,
        script_prefix_lines,
        split_group=split_group,
    )


@pytest.mark.parametrize(
    "concurrency",
    (True, 1.5, [], {}, "0", 0, "-1", -1, "1.5", "not-an-integer", None),
)
def test_get_benchmark_config_rejects_invalid_concurrency(submit_module: ModuleType, concurrency):
    with pytest.raises(ValueError, match="benchmark.concurrency_list must be a positive integer"):
        _get_benchmark_config(submit_module, concurrency)


@pytest.mark.parametrize(
    "concurrency",
    (True, 1.5, [], {}, "0", 0, "-1", -1, "1.5", "not-an-integer", None),
)
def test_example_worker_environment_rejects_invalid_concurrency(
    example_submit_module: ModuleType, concurrency
):
    with pytest.raises(ValueError, match="benchmark.concurrency_list must be a positive integer"):
        example_submit_module.build_worker_environment(
            worker_config={},
            env_config={},
            role="GEN",
            benchmark_mode="gen_only",
            nsys_on=False,
            profile_range="",
            concurrency=concurrency,
        )


def test_example_worker_environment_exports_positive_concurrency(example_submit_module: ModuleType):
    worker_environment = example_submit_module.build_worker_environment(
        worker_config={},
        env_config={},
        role="GEN",
        benchmark_mode="gen_only",
        nsys_on=False,
        profile_range="",
        concurrency="4301",
    )

    assert worker_environment["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] == "4301"


def test_ci_submit_selects_same_least_duration_shard_as_pytest_split(
    ci_submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    test_lines = [
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_kimi-k25] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_kimi-k25] TIMEOUT (90)",
    ]
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text("\n".join(test_lines), encoding="utf-8")

    durations_dir = tmp_path / "tests" / "integration" / "defs"
    durations_dir.mkdir(parents=True)
    durations = {
        ci_submit_module._test_nodeid(test_lines[0]): 836.268,
        ci_submit_module._test_nodeid(test_lines[1]): 1462.754,
        ci_submit_module._test_nodeid(test_lines[2]): 2211.1548,
        ci_submit_module._test_nodeid(test_lines[3]): 2548.912,
    }
    (durations_dir / ".test_durations").write_text(json.dumps(durations), encoding="utf-8")
    script_prefix_lines = [
        'export pytestCommand="pytest --splitting-algorithm least_duration '
        "--splits 4 --group 3 "
        '--durations-path /remote/tests/integration/defs/.test_durations"'
    ]

    selected = ci_submit_module.select_test_case_line(
        test_list_path,
        tmp_path,
        script_prefix_lines,
        split_group=3,
    )

    assert selected == test_lines[1]


@pytest.mark.parametrize(
    ("selected_suffix", "waive_line", "expected"),
    (
        (" SKIP (inline)", "", True),
        (" SKIP(inline)", "", True),
        ("", "{nodeid} SKIP (global)", True),
        ("", "{nodeid} SKIP(global)", True),
        ("", "{nodeid} XFAIL (known failure)", False),
        ("", "perf/test_other.py::test_other SKIP (other)", False),
    ),
)
def test_ci_submit_skips_precheck_only_for_selected_skip_waive(
    ci_submit_module: ModuleType,
    tmp_path: Path,
    selected_suffix: str,
    waive_line: str,
    expected: bool,
) -> None:
    nodeid = "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-case]"
    waives = tmp_path / "waives.txt"
    waives.write_text(waive_line.format(nodeid=nodeid), encoding="utf-8")

    assert (
        ci_submit_module.selected_test_is_skip_waived(f"{nodeid}{selected_suffix}", waives)
        is expected
    )


def test_ci_submit_honors_matching_platform_scoped_skip_waive(
    ci_submit_module: ModuleType, tmp_path: Path
) -> None:
    nodeid = "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-case]"
    waives = tmp_path / "waives.txt"
    waives.write_text(f"full:GB300/{nodeid} SKIP (platform)\n", encoding="utf-8")

    assert ci_submit_module.selected_test_is_skip_waived(
        nodeid, waives, test_prefix="GB300-Disagg-Perf"
    )
    assert not ci_submit_module.selected_test_is_skip_waived(
        nodeid, waives, test_prefix="DGX_B200-Disagg-Perf"
    )


def test_ci_submit_selector_matches_installed_pytest_split(
    ci_submit_module: ModuleType,
) -> None:
    lines = [
        f"perf/test_perf_sanity.py::test_e2e[case-{case_name}] TIMEOUT (90)"
        for case_name in ("zeta", "alpha", "gamma", "beta", "epsilon", "delta")
    ]
    nodeids = [ci_submit_module._test_nodeid(line) for line in lines]
    items = [_FakePytestItem(nodeid) for nodeid in nodeids]
    duration_sets = (
        {nodeid: float(index + 1) for index, nodeid in enumerate(nodeids)},
        dict.fromkeys(nodeids, 4.0),
        {nodeids[1]: 8.0, nodeids[4]: 2.0, "irrelevant::test": 1000.0},
        {},
    )

    for durations in duration_sets:
        for splits in (2, 3, 4):
            expected_groups = LeastDurationAlgorithm()(splits, items, durations)
            for group, expected_group in enumerate(expected_groups, start=1):
                selected = ci_submit_module._select_least_duration_group(
                    lines,
                    durations,
                    splits,
                    group,
                )
                assert [ci_submit_module._test_nodeid(line) for line in selected] == [
                    item.nodeid for item in expected_group.selected
                ]


@pytest.mark.parametrize(
    ("splits", "group", "match"),
    (
        (0, 1, "--splits"),
        (2, 0, "--group"),
        (2, 3, "--group"),
    ),
)
def test_ci_submit_rejects_invalid_least_duration_groups(
    ci_submit_module: ModuleType,
    splits: int,
    group: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ci_submit_module._select_least_duration_group(
            ["perf/test_perf_sanity.py::test_e2e[case]"],
            {},
            splits,
            group,
        )


def test_ci_submit_ignores_test_list_comments(
    ci_submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text(
        "# section comment\n\nperf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )

    assert ci_submit_module._read_test_list_lines(test_list_path) == [
        "perf/test_perf_sanity.py::test_e2e[case]"
    ]


def test_ci_submit_rejects_split_group_disagreement(
    ci_submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="disagrees with pytest --group"):
        _select_ci_test_case_line(
            ci_submit_module,
            tmp_path,
            pytest_options="--splits 1 --group 1",
            split_group=2,
        )


def test_ci_submit_rejects_missing_pytest_split_durations(
    ci_submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "tests" / "integration" / "defs" / ".test_durations"
    with pytest.raises(FileNotFoundError, match=f"durations file not found: {expected_path}"):
        _select_ci_test_case_line(
            ci_submit_module,
            tmp_path,
            pytest_options="--splits 1 --group 1 --durations-path /remote/.test_durations",
            split_group=1,
        )


@pytest.mark.parametrize(
    ("assignment", "expected"),
    (
        ("LLM_MODELS_ROOT=/models", "/models"),
        ("LLM_MODELS_ROOT='/models with spaces'", "/models with spaces"),
        ("LLM_MODELS_ROOT=/models/cache=production", "/models/cache=production"),
    ),
)
def test_extract_pytest_command_env(ci_submit_module: ModuleType, assignment: str, expected: str):
    lines = [f'export pytestCommand="LLM_ROOT=/src {assignment} COLUMNS=300 pytest -vv"']

    assert ci_submit_module.extract_pytest_command_env(lines, "LLM_MODELS_ROOT") == expected


def test_extract_pytest_command_env_rejects_missing_leading_assignment(
    ci_submit_module: ModuleType,
):
    lines = ['export pytestCommand="LLM_ROOT=/src pytest LLM_MODELS_ROOT=/too-late"']

    with pytest.raises(ValueError, match="does not set leading environment variable"):
        ci_submit_module.extract_pytest_command_env(lines, "LLM_MODELS_ROOT")


def test_extract_pytest_command_env_rejects_malformed_export(ci_submit_module: ModuleType):
    lines = ['export pytestCommand="LLM_ROOT=/src LLM_MODELS_ROOT=/models pytest']

    with pytest.raises(ValueError, match="cannot parse exported pytestCommand"):
        ci_submit_module.extract_pytest_command_env(lines, "LLM_MODELS_ROOT")


def test_resolve_llm_models_root_falls_back_to_submitter_env(
    ci_submit_module: ModuleType, monkeypatch
):
    monkeypatch.setenv("LLM_MODELS_ROOT", "/models/from-env")
    lines = ['export pytestCommand="LLM_ROOT=/src pytest LLM_MODELS_ROOT=/too-late"']

    assert ci_submit_module._resolve_llm_models_root(lines) == "/models/from-env"


def test_resolve_llm_models_root_explains_both_missing_sources(
    ci_submit_module: ModuleType, monkeypatch
):
    monkeypatch.delenv("LLM_MODELS_ROOT", raising=False)
    lines = ['export pytestCommand="LLM_ROOT=/src pytest"']

    with pytest.raises(ValueError, match="getPytestBaseCommandLine in L0_Test.groovy"):
        ci_submit_module._resolve_llm_models_root(lines)


def test_resolve_llm_models_root_does_not_mask_malformed_command(
    ci_submit_module: ModuleType, monkeypatch
):
    monkeypatch.setenv("LLM_MODELS_ROOT", "/models/from-env")
    lines = ['export pytestCommand="LLM_ROOT=/src pytest']

    with pytest.raises(ValueError, match="cannot parse exported pytestCommand"):
        ci_submit_module._resolve_llm_models_root(lines)


# --------------------- gen-only request-queue capacity cap -----------------------
# The fill loop cannot reach a target above the GEN executor's active capacity,
# so TLLM_BENCHMARK_REQ_QUEUES_SIZE is clamped to it. This applies to every
# gen_only run of the local harness, not just BOLT ones.
@pytest.mark.parametrize(
    "gen_config, concurrency, expected",
    [
        # Capacity above the ask: concurrency wins (no clamp).
        ({"max_batch_size": 512}, 128, 128),
        # Capacity below the ask: clamped to max_batch_size.
        ({"max_batch_size": 64}, 128, 64),
        # Exactly equal: no clamp.
        ({"max_batch_size": 128}, 128, 128),
        # attention_dp multiplies capacity by TP.
        ({"max_batch_size": 64, "enable_attention_dp": True, "tensor_parallel_size": 4}, 128, 128),
        # TP is ignored unless attention_dp is on.
        ({"max_batch_size": 64, "tensor_parallel_size": 4}, 128, 64),
        # attention_dp on but TP still too small to cover the ask.
        ({"max_batch_size": 8, "enable_attention_dp": True, "tensor_parallel_size": 2}, 128, 16),
        # Missing max_batch_size defaults to concurrency, i.e. no clamp.
        ({}, 128, 128),
    ],
)
def test_get_benchmark_request_queue_size(
    local_submit_module: ModuleType, gen_config, concurrency, expected
):
    config = {"worker_config": {"gen": gen_config}}
    assert local_submit_module.get_benchmark_request_queue_size(config, concurrency) == expected


@pytest.mark.parametrize("worker_config", [{}, {"gen": None}, None])
def test_get_benchmark_request_queue_size_tolerates_absent_gen_config(
    local_submit_module: ModuleType, worker_config
):
    config = {} if worker_config is None else {"worker_config": worker_config}
    assert local_submit_module.get_benchmark_request_queue_size(config, 32) == 32


# ------------------------- partition GRES tri-state / #SBATCH --------------------
def _fake_sinfo(monkeypatch, module, gres_out, raises=False):
    def fake_check_output(cmd, **kwargs):
        if raises:
            raise OSError("sinfo unavailable")
        return gres_out

    monkeypatch.setattr(module.subprocess, "check_output", fake_check_output)


def test_partition_gpu_gres_prefers_gpu_row_over_null(local_submit_module: ModuleType, monkeypatch):
    _fake_sinfo(monkeypatch, local_submit_module, "(null)\ngpu:8\n")
    assert local_submit_module.partition_gpu_gres("batch") == "gpu:8"


def test_partition_gpu_gres_reports_null_as_definitive(
    local_submit_module: ModuleType, monkeypatch
):
    # NOT None: "(null)" means the partition really has no GPU GRES (e.g. EOS),
    # which generate_gpu_request must distinguish from "sinfo could not answer".
    _fake_sinfo(monkeypatch, local_submit_module, "(null)\n")
    assert local_submit_module.partition_gpu_gres("batch") == "(null)"


def test_partition_gpu_gres_returns_none_when_sinfo_fails(
    local_submit_module: ModuleType, monkeypatch
):
    _fake_sinfo(monkeypatch, local_submit_module, "", raises=True)
    assert local_submit_module.partition_gpu_gres("batch") is None


def test_partition_gpu_gres_returns_none_for_sentinel_partition(
    local_submit_module: ModuleType,
):
    assert local_submit_module.partition_gpu_gres("unspecified") is None


def test_generate_gpu_request_adds_gres_when_partition_advertises_gpus(
    local_submit_module: ModuleType, monkeypatch
):
    _fake_sinfo(monkeypatch, local_submit_module, "gpu:4\n")
    assert local_submit_module.generate_gpu_request("batch", 4) == [
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --gres=gpu:4",
    ]


def test_generate_gpu_request_omits_everything_on_gpu_less_partition(
    local_submit_module: ModuleType, monkeypatch
):
    # "(null)" is definitive, so ask for nothing -- --gres would be rejected as
    # an invalid generic resource on a cluster that registers no GRES at all.
    _fake_sinfo(monkeypatch, local_submit_module, "(null)\n")
    assert local_submit_module.generate_gpu_request("batch", 4) == []


def test_generate_gpu_request_still_asks_when_sinfo_cannot_answer(
    local_submit_module: ModuleType, monkeypatch
):
    # Undeterminable must NOT be read as "no GPUs": request --gpus-per-node
    # (but not --gres, which we cannot justify without a GRES reading).
    _fake_sinfo(monkeypatch, local_submit_module, "", raises=True)
    assert local_submit_module.generate_gpu_request("batch", 8) == ["#SBATCH --gpus-per-node=8"]


def test_default_slurm_partition_picks_the_starred_entry(
    local_submit_module: ModuleType, monkeypatch
):
    _fake_sinfo(monkeypatch, local_submit_module, "batch\ninteractive*\n")
    assert local_submit_module.default_slurm_partition() == "interactive"


def test_default_slurm_partition_empty_when_none_flagged(
    local_submit_module: ModuleType, monkeypatch
):
    _fake_sinfo(monkeypatch, local_submit_module, "batch\ninteractive\n")
    assert local_submit_module.default_slurm_partition() == ""


# --------------------------------------------------------------------------- #
# Test-id grammar: <prefix>-<mode>[-<modifier>]-<stem>
#
# The modifier segment is what makes the grammar non-trivial: disagg stems
# routinely contain "-" (".._ccb-NIXL"), so the stem cannot be recovered by
# counting segments -- only by peeling a known modifier off the front. Three
# hand-duplicated parsers implement this (the two generators here plus
# test_perf_sanity.py:parse_test_string) and there is no shared module they can
# import, so the agreement test below is the only thing keeping them in step.
# --------------------------------------------------------------------------- #
_DISAGG_STEM = "gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL"

# (test id, config stem, benchmark_mode, runtime_mode, time_breakdown)
TEST_ID_GRAMMAR = (
    ("disagg-e2e-" + _DISAGG_STEM, _DISAGG_STEM, "e2e", "disaggregated", False),
    (
        "disagg_upload-e2e-time_breakdown-" + _DISAGG_STEM,
        _DISAGG_STEM,
        "e2e",
        "disaggregated",
        True,
    ),
    ("disagg-gen_only-" + _DISAGG_STEM, _DISAGG_STEM, "gen_only", "disaggregated", False),
    (
        "disagg_upload-gen_only-time_breakdown-" + _DISAGG_STEM,
        _DISAGG_STEM,
        "gen_only",
        "disaggregated",
        True,
    ),
    ("aggr-ctx_only-" + _DISAGG_STEM, _DISAGG_STEM, "ctx_only", "aggregated", False),
    (
        "aggr_upload-ctx_only-time_breakdown-" + _DISAGG_STEM,
        _DISAGG_STEM,
        "ctx_only",
        "aggregated",
        True,
    ),
    # Plain aggregated: parts[1] alone is the stem, the remainder is the server
    # name, and no modifier segment exists -- so a "-" in the stem is illegal
    # there and this shape must stay untouched by the modifier logic.
    (
        "aggr-deepseek_r1_fp4_v2-r1_fp4_v2_dep4_mtp1_8k1k",
        "deepseek_r1_fp4_v2",
        None,
        "aggregated",
        False,
    ),
)


def _parse_with_module(module: ModuleType, tmp_path: Path, test_id: str):
    """Normalise the two generators' parsers to one tuple.

    The CI parser takes a test-list line and resolves the yaml on disk; the local
    parser takes the bracket content alone. Both are fed the same id here.
    """
    if Path(module.__file__).parent.name == "local":
        stem, _select_pattern, runtime_mode, benchmark_mode, time_breakdown = (
            module.parse_test_string(test_id)
        )
        return stem, benchmark_mode, runtime_mode, time_breakdown

    for folder in (module.AGG_CONFIG_FOLDER, module.DISAGG_CONFIG_FOLDER):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for _id, stem, _mode, _runtime, _tb in TEST_ID_GRAMMAR:
        for folder in (module.AGG_CONFIG_FOLDER, module.DISAGG_CONFIG_FOLDER):
            (tmp_path / folder / f"{stem}.yaml").write_text("{}\n", encoding="utf-8")
    config_yaml, _server_name, benchmark_mode, runtime_mode, time_breakdown = (
        module.parse_test_case_name(
            str(tmp_path), f"perf/test_perf_sanity.py::test_e2e[{test_id}] TIMEOUT (90)"
        )
    )
    return Path(config_yaml).stem, benchmark_mode, runtime_mode, time_breakdown


@pytest.mark.parametrize(
    ("test_id", "stem", "benchmark_mode", "runtime_mode", "time_breakdown"),
    TEST_ID_GRAMMAR,
    ids=[entry[0][:40] for entry in TEST_ID_GRAMMAR],
)
def test_both_generators_parse_the_id_grammar(
    submit_module: ModuleType,
    tmp_path: Path,
    test_id: str,
    stem: str,
    benchmark_mode,
    runtime_mode: str,
    time_breakdown: bool,
) -> None:
    assert _parse_with_module(submit_module, tmp_path, test_id) == (
        stem,
        benchmark_mode,
        runtime_mode,
        time_breakdown,
    )


def test_all_three_parsers_agree_on_the_id_grammar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner and both generators must read every id identically.

    They are three hand-written copies of one grammar with no shared module. If
    they drift, the launch script runs a different case than the one pytest
    collects -- or writes its artifacts into a different directory -- and nothing
    else in the tree notices.
    """
    pytest.importorskip("torch._inductor")
    sys.path.insert(0, str(REPO_ROOT / "tests" / "integration"))
    from defs.perf import test_perf_sanity as runner

    modules = [_load_module(path, monkeypatch) for path in SUBMIT_PATHS]
    for index, (test_id, stem, benchmark_mode, runtime_mode, time_breakdown) in enumerate(
        TEST_ID_GRAMMAR
    ):
        runner_stem, _select, runner_runtime, runner_mode, runner_tb = runner.parse_test_string(
            test_id
        )
        assert (runner_stem, runner_mode, runner_runtime, runner_tb) == (
            stem,
            benchmark_mode,
            runtime_mode,
            time_breakdown,
        ), test_id
        for module in modules:
            assert _parse_with_module(module, tmp_path / str(index), test_id) == (
                runner_stem,
                runner_mode,
                runner_runtime,
                runner_tb,
            ), f"{test_id} in {module.__file__}"


@pytest.mark.parametrize(
    "test_id",
    (
        "disagg-e2e-time_breakdown",
        "aggr-ctx_only-time_breakdown",
    ),
)
def test_a_modifier_with_no_config_is_rejected(
    submit_module: ModuleType, tmp_path: Path, test_id: str
) -> None:
    """An id ending at the modifier has no stem left to look up.

    Without this check the stem would come out empty and the failure would
    surface as a FileNotFoundError for "<folder>/.yaml".
    """
    with pytest.raises((AssertionError, ValueError)):
        _parse_with_module(submit_module, tmp_path, test_id)


def test_format_test_label_round_trips_through_the_local_parser(
    local_submit_module: ModuleType,
) -> None:
    """The generator regenerates the id it was handed, modifier included.

    local/submit.py rebuilds the test id from the parsed components (see the
    comment above its test_case_name reconstruction); if the formatter and the
    parser disagree the run silently writes two divergent output folders.
    """
    for test_id, stem, benchmark_mode, runtime_mode, time_breakdown in TEST_ID_GRAMMAR:
        if benchmark_mode is None:
            continue
        prefix = "disagg" if runtime_mode == "disaggregated" else "aggr"
        label = local_submit_module.format_test_label(benchmark_mode, time_breakdown)
        rebuilt = f"{prefix}-{label}-{stem}"
        assert rebuilt == test_id.replace("_upload", "", 1)
