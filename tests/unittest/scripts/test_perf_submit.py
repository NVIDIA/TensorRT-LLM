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
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLE_SUBMIT_PATH = REPO_ROOT / "examples" / "disaggregated" / "slurm" / "benchmark" / "submit.py"


def _load_module(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(f"perf_submit_{path.parent.name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(EXAMPLE_SUBMIT_PATH, monkeypatch)


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


# ------------------------- online-EPLB /dev/shm mount ----------------------------
CI_SUBMIT_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py"
LOCAL_SUBMIT_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py"
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


@pytest.fixture
def ci_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(CI_SUBMIT_PATH, monkeypatch)


@pytest.fixture
def local_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(LOCAL_SUBMIT_PATH, monkeypatch)


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


def _fake_sinfo(monkeypatch, module, gres_out):
    monkeypatch.setattr(module.subprocess, "check_output", lambda cmd, **kwargs: gres_out)


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
