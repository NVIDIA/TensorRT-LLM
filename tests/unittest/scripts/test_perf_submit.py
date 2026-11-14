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


def _dwdp_disagg_config(num_ctx_servers=16, dwdp_size=16, num_groups=1, ctx_tp=1):
    return {
        "hardware": {"gpus_per_node": 4, "num_ctx_servers": num_ctx_servers, "num_gen_servers": 3},
        "worker_config": {
            "ctx": {
                "tensor_parallel_size": ctx_tp,
                "dwdp_config": {"dwdp_size": dwdp_size, "num_groups": num_groups},
            },
            "gen": {"tensor_parallel_size": 1, "context_parallel_size": 8},
        },
    }


def _disagg_hardware(module: ModuleType, config: dict, benchmark_mode: str = "e2e") -> dict:
    if Path(module.__file__).parent.name == "local":
        return module.get_hardware_config(config, "disaggregated", benchmark_mode)
    return module.get_hardware_config(config, "disaggregated", benchmark_mode, None)


def test_dwdp_ctx_servers_pack_into_one_mpi_step(submit_module: ModuleType):
    # 16 single-GPU ctx servers fill 4 nodes as one MPI step instead of taking
    # a node each; the 3 helix CP8 gen servers keep 2 nodes apiece.
    hardware = _disagg_hardware(submit_module, _dwdp_disagg_config())
    assert hardware["dwdp_size"] == 16
    assert hardware["ctx_nodes"] == 4
    assert hardware["total_nodes"] == 4 + 3 * 2
    assert hardware["total_gpus"] == 40


def test_non_dwdp_ctx_servers_keep_one_node_slice_each(submit_module: ModuleType):
    config = _dwdp_disagg_config()
    del config["worker_config"]["ctx"]["dwdp_config"]
    hardware = _disagg_hardware(submit_module, config)
    assert hardware["dwdp_size"] == 0
    assert hardware["ctx_nodes"] == 16
    assert hardware["total_nodes"] == 16 + 3 * 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"ctx_tp": 2}, r"tp \* pp \* cp == 1"),
        ({"num_ctx_servers": 8}, "num_ctx_servers must equal dwdp_size"),
        ({"num_ctx_servers": 6, "dwdp_size": 3, "num_groups": 2}, "fill whole nodes"),
    ),
)
def test_dwdp_ctx_layout_rejects_unlaunchable_configs(
    submit_module: ModuleType, overrides: dict, message: str
):
    with pytest.raises(ValueError, match=message):
        _disagg_hardware(submit_module, _dwdp_disagg_config(**overrides))


def test_dwdp_ctx_worker_command_replaces_llmapi_launch(ci_submit_module: ModuleType):
    worker_line = (
        'export partialPytestCommandWorker="LLM_ROOT=/src COLUMNS=300 '
        '/src/tensorrt_llm/llmapi/trtllm-llmapi-launch pytest -vv --test-list=/w/list.txt"'
    )
    assert ci_submit_module.get_dwdp_ctx_worker_command(worker_line, "bash /rank.sh") == (
        'export partialPytestCommandDWDPCtxWorker="LLM_ROOT=/src COLUMNS=300 '
        'bash /rank.sh pytest -vv --test-list=/w/list.txt"'
    )


def _aggregated_hardware(module: ModuleType, config: dict, benchmark_mode: str) -> dict:
    if Path(module.__file__).parent.name == "local":
        return module.get_hardware_config(config, "aggregated", benchmark_mode)
    return module.get_hardware_config(config, "aggregated", benchmark_mode, None)


def test_ctx_only_folds_dwdp_ctx_servers_into_one_server(submit_module: ModuleType):
    # ctx_only runs the 16 single-GPU DWDP ctx servers as one 16-GPU
    # attention-DP server, not as a single one-GPU server.
    hardware = _aggregated_hardware(submit_module, _dwdp_disagg_config(), "ctx_only")
    assert hardware["gpus_per_server"] == 16
    assert hardware["total_nodes"] == 4


def test_ctx_only_without_dwdp_sizes_from_ctx_parallelism(submit_module: ModuleType):
    config = _dwdp_disagg_config(num_ctx_servers=2, ctx_tp=8)
    del config["worker_config"]["ctx"]["dwdp_config"]
    hardware = _aggregated_hardware(submit_module, config, "ctx_only")
    assert hardware["gpus_per_server"] == 8
    assert hardware["total_nodes"] == 2


# ------------------------- online-EPLB /dev/shm mount ----------------------------
BENCHMARK_UTILS_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "benchmark_utils.py"
KIMI_K25_EPLB416_CONFIG = (
    "tests/scripts/perf-sanity/disaggregated/"
    "gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml"
)
DSV4_STATIC_EPLB_CONFIG = (
    "tests/scripts/perf-sanity/disaggregated/"
    "gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml"
)


@pytest.fixture
def benchmark_utils_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(BENCHMARK_UTILS_PATH, monkeypatch)


def _load_repo_yaml(relative_path: str) -> dict:
    import yaml

    with open(REPO_ROOT / relative_path) as f:
        return yaml.safe_load(f)


def _lb_config(load_balancer, role="gen") -> dict:
    return {"worker_config": {role: {"moe_config": {"load_balancer": load_balancer}}}}


@pytest.mark.parametrize(
    "config, expected",
    [
        (_lb_config({"num_slots": 416, "layer_updates_per_iter": 1}), True),
        (_lb_config({"num_slots": 416, "layer_updates_per_iter": 1}, role="ctx"), True),
        (_lb_config({"num_slots": 384, "layer_updates_per_iter": 0}), False),
        (_lb_config({"num_slots": 384}), False),
        (_lb_config(None), False),
        ({"worker_config": {"gen": {"moe_config": None}}}, False),
        ({"worker_config": {"gen": None}}, False),
        (
            {
                "server_configs": [
                    {"moe_config": {"load_balancer": {"layer_updates_per_iter": 0}}},
                    {"moe_config": {"load_balancer": {"layer_updates_per_iter": 2}}},
                ]
            },
            True,
        ),
        ({}, False),
        (None, False),
    ],
)
def test_uses_online_eplb(benchmark_utils_module: ModuleType, config, expected):
    assert benchmark_utils_module.uses_online_eplb(config) is expected


def test_uses_online_eplb_reads_load_balancer_file(
    benchmark_utils_module: ModuleType, tmp_path: Path
):
    (tmp_path / "online.yaml").write_text("num_slots: 288\nlayer_updates_per_iter: 1\n")
    (tmp_path / "static.yaml").write_text("num_slots: 288\nlayer_updates_per_iter: 0\n")
    assert benchmark_utils_module.uses_online_eplb(_lb_config("online.yaml"), str(tmp_path))
    assert not benchmark_utils_module.uses_online_eplb(_lb_config("static.yaml"), str(tmp_path))
    assert benchmark_utils_module.uses_online_eplb(
        _lb_config(str(tmp_path / "online.yaml")), "/nonexistent"
    )


@pytest.mark.parametrize(
    "server_config, expected",
    [
        ({"extra_llm_api_config_path": "online_extra.yaml"}, True),
        ({"extra_llm_api_config_path": "lb_file_extra.yaml"}, True),
        ({"extra_llm_api_config_path": "static_extra.yaml"}, False),
        ({"extra_llm_api_config_path": "empty_extra.yaml"}, False),
        # Inline fields replace the file's top-level keys, so this moe_config wins.
        (
            {
                "extra_llm_api_config_path": "online_extra.yaml",
                "moe_config": {"backend": "CUTEDSL"},
            },
            False,
        ),
    ],
)
def test_uses_online_eplb_reads_extra_llm_api_config(
    benchmark_utils_module: ModuleType, tmp_path: Path, server_config, expected
):
    (tmp_path / "online_extra.yaml").write_text(
        "moe_config:\n  load_balancer:\n    num_slots: 288\n    layer_updates_per_iter: 1\n"
    )
    (tmp_path / "static_extra.yaml").write_text(
        "moe_config:\n  load_balancer:\n    num_slots: 288\n    layer_updates_per_iter: 0\n"
    )
    (tmp_path / "lb_file_extra.yaml").write_text("moe_config:\n  load_balancer: online.yaml\n")
    (tmp_path / "online.yaml").write_text("num_slots: 288\nlayer_updates_per_iter: 1\n")
    (tmp_path / "empty_extra.yaml").write_text("")
    disagg_config = {"worker_config": {"gen": server_config}}
    aggr_config = {"server_configs": [server_config]}
    assert benchmark_utils_module.uses_online_eplb(disagg_config, str(tmp_path)) is expected
    assert benchmark_utils_module.uses_online_eplb(aggr_config, str(tmp_path)) is expected


def test_uses_online_eplb_rejects_non_mapping_extra_llm_api_config(
    benchmark_utils_module: ModuleType, tmp_path: Path
):
    (tmp_path / "list_extra.yaml").write_text("- moe_config\n")
    config = {"worker_config": {"gen": {"extra_llm_api_config_path": "list_extra.yaml"}}}
    with pytest.raises(ValueError, match="does not contain a YAML mapping"):
        benchmark_utils_module.uses_online_eplb(config, str(tmp_path))


def test_uses_online_eplb_on_checked_in_configs(benchmark_utils_module: ModuleType):
    assert benchmark_utils_module.uses_online_eplb(
        _load_repo_yaml(KIMI_K25_EPLB416_CONFIG), str(REPO_ROOT)
    )
    assert not benchmark_utils_module.uses_online_eplb(
        _load_repo_yaml(DSV4_STATIC_EPLB_CONFIG), str(REPO_ROOT)
    )


@pytest.mark.parametrize(
    "mounts, expected_suffix",
    [
        ("", "tmpfs:/dev/shm:size=640G"),
        ("/a:/a", "/a:/a,tmpfs:/dev/shm:size=640G"),
        ("/a:/a,", "/a:/a,tmpfs:/dev/shm:size=640G"),
        ("/x:/dev/shm/sub", "/x:/dev/shm/sub,tmpfs:/dev/shm:size=640G"),
        ("/dev/shm:/host_shm", "/dev/shm:/host_shm,tmpfs:/dev/shm:size=640G"),
    ],
)
def test_add_eplb_shm_mount_appends_for_online_eplb(
    benchmark_utils_module: ModuleType, mounts, expected_suffix
):
    config = _lb_config({"num_slots": 416, "layer_updates_per_iter": 1})
    assert benchmark_utils_module.add_eplb_shm_mount(mounts, config) == expected_suffix


@pytest.mark.parametrize(
    "mounts",
    [
        "/a:/a,tmpfs:/dev/shm:size=900G",
        "/dev/shm",
        "/a:/a,/dev/shm",
        "/host/shm:/dev/shm/",
    ],
)
def test_add_eplb_shm_mount_keeps_caller_dev_shm(benchmark_utils_module: ModuleType, mounts):
    config = _lb_config({"num_slots": 416, "layer_updates_per_iter": 1})
    assert benchmark_utils_module.add_eplb_shm_mount(mounts, config) == mounts


def test_add_eplb_shm_mount_noop_without_online_eplb(benchmark_utils_module: ModuleType):
    config = _lb_config({"num_slots": 384, "layer_updates_per_iter": 0})
    assert benchmark_utils_module.add_eplb_shm_mount("/a:/a", config) == "/a:/a"


def test_ci_submit_adds_eplb_shm_mount_to_launch_script(
    ci_submit_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    test_id = (
        "perf/test_perf_sanity.py::test_e2e[disagg-gen_only-"
        "gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL]"
    )
    test_list = tmp_path / "test_list.txt"
    test_list.write_text(test_id + "\n")
    prefix = tmp_path / "prefix.sh"
    prefix.write_text(
        "#!/bin/bash\n"
        "export jobWorkspace=/ws\n"
        'export pytestCommand="LLM_ROOT=/src LLM_MODELS_ROOT=/models '
        "/src/tensorrt_llm/llmapi/trtllm-llmapi-launch pytest perf/test_perf_sanity.py "
        '--test-prefix=/src/tests/integration/defs --test-list=/ws/test_list.txt --output-dir=/ws"\n'
    )
    srun_args = tmp_path / "srun_args.txt"
    srun_args.write_text("--container-image=/img.sqsh --container-mounts=/a:/a,/b:/b\n")
    launch = tmp_path / "launch.sh"
    draft = REPO_ROOT / "jenkins" / "scripts" / "perf" / "disaggregated" / "slurm_launch_draft.sh"
    monkeypatch.setattr(
        "sys.argv",
        [
            "submit.py",
            "--draft-launch-sh",
            str(draft),
            "--launch-sh",
            str(launch),
            "--run-sh",
            "/run.sh",
            "--install-sh",
            "/install.sh",
            "--llm-src",
            str(REPO_ROOT),
            "--test-list",
            str(test_list),
            "--script-prefix",
            str(prefix),
            "--srun-args",
            str(srun_args),
            "--stage-name",
            "GB300-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-Post-Merge-1",
        ],
    )
    ci_submit_module.main()
    assert '"--container-mounts=/a:/a,/b:/b,tmpfs:/dev/shm:size=640G"' in launch.read_text()


def test_local_submit_adds_eplb_shm_mount_to_launch_script(
    local_submit_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    test_id = (
        "perf/test_perf_sanity.py::test_e2e[disagg-e2e-"
        "gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL]"
    )
    _fake_sinfo(monkeypatch, local_submit_module, "gpu:4\n")
    launch = tmp_path / "launch.sh"
    monkeypatch.setattr(
        "sys.argv",
        [
            "submit.py",
            "--test-list",
            test_id,
            "--partition",
            "batch",
            "--account",
            "acct",
            "--job-name",
            "job",
            "--image",
            "/img.sqsh",
            "--mounts",
            "/a:/a",
            "--work-dir",
            str(tmp_path),
            "--launch-sh",
            str(launch),
            "--llm-src",
            str(REPO_ROOT),
            "--llm-models-root",
            "/models",
            "--cluster-name",
            "oci-aga",
        ],
    )
    local_submit_module.main()
    content = launch.read_text()
    assert f'"--container-mounts=/a:/a,{REPO_ROOT}:{REPO_ROOT},tmpfs:/dev/shm:size=640G"' in content
