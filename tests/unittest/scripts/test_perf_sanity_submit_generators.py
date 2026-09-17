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

"""Tests for the perf-sanity SLURM launch-script generators."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PERF_SCRIPTS = REPO_ROOT / "jenkins" / "scripts" / "perf"
LOCAL_SUBMIT = PERF_SCRIPTS / "local" / "submit.py"
CI_SUBMIT = PERF_SCRIPTS / "submit.py"
DISAGG_DRAFT = PERF_SCRIPTS / "disaggregated" / "slurm_launch_draft.sh"
AGG_DRAFT = PERF_SCRIPTS / "aggregated" / "slurm_launch_draft.sh"

CON4301 = "gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL"

CON8 = "gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL"
CON8_GEN_ONLY_NODES, CON8_GEN_ONLY_GPUS = 9, 36
CON8_NO_CONTEXT_NODES, CON8_NO_CONTEXT_GPUS = 2, 8

GEN_ONLY_NODES, GEN_ONLY_GPUS = 14, 56
NO_CONTEXT_NODES, NO_CONTEXT_GPUS = 2, 8
CTX_ONLY_NODES, CTX_ONLY_GPUS = 1, 4

AGGREGATED_PATH_MODES = ("ctx_only", "gen_only_no_context")

FAKE_KV_VAR = "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"
FAKE_KV_ASSIGNMENT = re.compile(rf'(?:^|[\s"]){FAKE_KV_VAR}=1(?:[\s"]|$)', re.MULTILINE)

FOUR_ROLE_EXPORTS = (
    "pytestCommandCTXWorker",
    "pytestCommandGENWorker",
    "pytestCommandDisaggServer",
    "pytestCommandBenchmark",
    "DISAGG_SERVING_TYPE",
)


def _test_id_prefix(benchmark_mode: str) -> str:
    """The id prefix a mode is reachable through."""
    return "aggr" if benchmark_mode in AGGREGATED_PATH_MODES else "disagg"


def _require_yaml() -> None:
    pytest.importorskip("yaml", reason="the generators parse the config YAML")


def _generate_local(
    tmp_path: Path, benchmark_mode: str, prefix: str = "", config: str = CON4301
) -> str:
    """Run the local generator for one test id and return the launch script text."""
    _require_yaml()
    work_dir = tmp_path / (f"{prefix}_{benchmark_mode}" if prefix else benchmark_mode)
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"
    prefix = prefix or _test_id_prefix(benchmark_mode)

    subprocess.run(
        [
            sys.executable,
            str(LOCAL_SUBMIT),
            "--test-list",
            f"perf/test_perf_sanity.py::test_e2e[{prefix}-{benchmark_mode}-{config}]",
            "--partition",
            "unspecified",
            "--account",
            "unit_test_account",
            "--job-name",
            "unit_test_job",
            "--image",
            "/dev/null/image.sqsh",
            "--mounts",
            "/dev/null:/dev/null",
            "--work-dir",
            str(work_dir),
            "--launch-sh",
            str(launch_sh),
            "--llm-src",
            str(REPO_ROOT),
            "--llm-models-root",
            "/dev/null/models",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return launch_sh.read_text()


def _generate_ci(tmp_path: Path, benchmark_mode: str, prefix: str = "") -> str:
    """Run the CI generator for one test id and return the launch script text."""
    _require_yaml()
    work_dir = tmp_path / (f"ci_{prefix}_{benchmark_mode}" if prefix else f"ci_{benchmark_mode}")
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"
    on_aggregated_path = (prefix or _test_id_prefix(benchmark_mode)).startswith("aggr")
    prefix = prefix or _test_id_prefix(benchmark_mode)

    test_list = work_dir / "test_list.txt"
    test_list.write_text(
        f"perf/test_perf_sanity.py::test_e2e[{prefix}_upload-{benchmark_mode}-{CON4301}]\n"
    )
    script_prefix = work_dir / "prefix.sh"
    script_prefix.write_text(
        'export pytestCommand="LLM_MODELS_ROOT=/dev/null/models '
        "/dev/null/llm-src/tensorrt_llm/llmapi/trtllm-llmapi-launch "
        "pytest -v perf/test_perf_sanity.py "
        f'--output-dir={work_dir}/out -o junit_logging=out-err"\n'
    )
    srun_args = work_dir / "srun_args.txt"
    srun_args.write_text("")

    stage_name = (
        "GB300-8_GPUs-2_Nodes-PyTorch-PerfSanity-Post-Merge"
        if on_aggregated_path
        else "GB300-56_GPUs-14_Nodes-PyTorch-Disagg-PerfSanity-Post-Merge"
    )
    subprocess.run(
        [
            sys.executable,
            str(CI_SUBMIT),
            "--draft-launch-sh",
            str(AGG_DRAFT if on_aggregated_path else DISAGG_DRAFT),
            "--launch-sh",
            str(launch_sh),
            "--run-sh",
            str(PERF_SCRIPTS / "local" / "slurm_run.sh"),
            "--install-sh",
            str(PERF_SCRIPTS / "local" / "slurm_install.sh"),
            "--llm-src",
            str(REPO_ROOT),
            "--test-list",
            str(test_list),
            "--script-prefix",
            str(script_prefix),
            "--srun-args",
            str(srun_args),
            "--split-group",
            "1",
            "--stage-name",
            stage_name,
            "--cluster-name",
            "aws-cmh",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return launch_sh.read_text()


def _generate_local_from_config_file(
    tmp_path: Path, benchmark_mode: str, config: str = CON4301
) -> str:
    """Run the local generator through its *other* entry path: --config-file."""
    _require_yaml()
    work_dir = tmp_path / f"cfgfile_{benchmark_mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"
    config_yaml = (
        REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "disaggregated" / f"{config}.yaml"
    )

    subprocess.run(
        [
            sys.executable,
            str(LOCAL_SUBMIT),
            "--config-file",
            str(config_yaml),
            "--benchmark-mode",
            benchmark_mode,
            "--partition",
            "unspecified",
            "--account",
            "unit_test_account",
            "--job-name",
            "unit_test_job",
            "--image",
            "/dev/null/image.sqsh",
            "--mounts",
            "/dev/null:/dev/null",
            "--work-dir",
            str(work_dir),
            "--launch-sh",
            str(launch_sh),
            "--llm-src",
            str(REPO_ROOT),
            "--llm-models-root",
            "/dev/null/models",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return launch_sh.read_text()


def _export(script: str, name: str) -> str:
    """Return the effective value of `export <name>=...` in a generated script."""
    matches = re.findall(rf"^export {re.escape(name)}=(.*)$", script, re.MULTILINE)
    assert matches, f"generated script never exports {name}"
    return matches[-1]


def test_no_context_takes_the_single_pytest_aggregated_launch_path(tmp_path: Path) -> None:
    """One pytestCommand, none of the four role commands, no DISAGG_SERVING_TYPE."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert "export pytestCommand=" in script
    for name in FOUR_ROLE_EXPORTS:
        assert f"export {name}=" not in script, (
            f"{name} means the four-role disaggregated path was taken"
        )


def test_gen_only_takes_the_four_role_disaggregated_launch_path(tmp_path: Path) -> None:
    """Negative control for the path: gen_only over the same yaml still splits roles."""
    script = _generate_local(tmp_path, "gen_only")

    for name in FOUR_ROLE_EXPORTS:
        assert f"export {name}=" in script


def test_no_context_shares_the_launch_path_with_ctx_only(tmp_path: Path) -> None:
    """The two modes that read a disagg yaml on the aggregated path agree on shape."""
    no_context = _generate_local(tmp_path, "gen_only_no_context")
    ctx_only = _generate_local(tmp_path, "ctx_only")

    for name in ("pytestCommand", "WORKER_ENV_VARS", "SERVER_ENV_VARS", "totalNodes"):
        assert f"export {name}=" in no_context
        assert f"export {name}=" in ctx_only
    assert FAKE_KV_ASSIGNMENT.search(no_context)
    assert not FAKE_KV_ASSIGNMENT.search(ctx_only)


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_no_context_is_not_reachable_through_the_disaggregated_prefix(
    tmp_path: Path, generator: str
) -> None:
    """`disagg-gen_only_no_context-...` must be a hard rejection, not a fallback."""
    generate = _generate_local if generator == "local" else _generate_ci
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        generate(tmp_path, "gen_only_no_context", prefix="disagg")

    assert "gen_only_no_context" in exc_info.value.stderr


def test_local_generator_sizes_gen_only_no_context_from_the_gen_worker_alone(
    tmp_path: Path,
) -> None:
    """The allocation is the gen worker's tp*pp*cp, and nothing else."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert _export(script, "totalNodes") == str(NO_CONTEXT_NODES)
    assert _export(script, "totalGpus") == str(NO_CONTEXT_GPUS)
    assert f"#SBATCH --nodes={NO_CONTEXT_NODES}" in script
    assert f"#SBATCH --ntasks={NO_CONTEXT_GPUS}" in script


def test_local_generator_keeps_the_ctx_fleet_for_gen_only(tmp_path: Path) -> None:
    """Negative control for the sizing: plain gen_only still allocates 14 nodes."""
    script = _generate_local(tmp_path, "gen_only")

    assert _export(script, "numCtxServers") == "12"
    assert _export(script, "totalNodes") == str(GEN_ONLY_NODES)
    assert _export(script, "totalGpus") == str(GEN_ONLY_GPUS)
    assert f"#SBATCH --nodes={GEN_ONLY_NODES}" in script
    assert f"#SBATCH --ntasks={GEN_ONLY_GPUS}" in script


def test_no_context_launches_one_of_a_replicated_gen_fleet(tmp_path: Path) -> None:
    """`num_gen_servers: 4` sizes for ONE replica, and is not refused."""
    script = _generate_local(tmp_path, "gen_only_no_context", config=CON8)

    assert _export(script, "totalNodes") == str(CON8_NO_CONTEXT_NODES)
    assert _export(script, "totalGpus") == str(CON8_NO_CONTEXT_GPUS)
    assert f"#SBATCH --nodes={CON8_NO_CONTEXT_NODES}" in script
    assert f"#SBATCH --ntasks={CON8_NO_CONTEXT_GPUS}" in script


def test_gen_only_keeps_the_replicated_gen_fleet(tmp_path: Path) -> None:
    """Control for the override: plain gen_only on con8 still allocates all four."""
    script = _generate_local(tmp_path, "gen_only", config=CON8)

    assert _export(script, "numCtxServers") == "1"
    assert _export(script, "numGenServers") == "4"
    assert _export(script, "totalNodes") == str(CON8_GEN_ONLY_NODES)
    assert _export(script, "totalGpus") == str(CON8_GEN_ONLY_GPUS)


@pytest.mark.parametrize("benchmark_mode", AGGREGATED_PATH_MODES)
def test_the_config_file_entry_path_picks_the_same_launch_path(
    tmp_path: Path, benchmark_mode: str
) -> None:
    """--config-file must route both aggregated-path modes exactly as --test-list does."""
    script = _generate_local_from_config_file(tmp_path, benchmark_mode)

    assert f"aggr-{benchmark_mode}-{CON4301}" in script
    assert f"disagg-{benchmark_mode}-" not in script
    for export in FOUR_ROLE_EXPORTS:
        assert f"export {export}=" not in script, (
            f"--config-file routed {benchmark_mode} to the four-role template"
        )


def test_local_generator_sizes_ctx_only_from_the_ctx_worker(tmp_path: Path) -> None:
    """Second control for the sizing: the sibling mode reads the *ctx* worker."""
    script = _generate_local(tmp_path, "ctx_only")

    assert _export(script, "totalNodes") == str(CTX_ONLY_NODES)
    assert _export(script, "totalGpus") == str(CTX_ONLY_GPUS)


def test_ci_generator_sizes_gen_only_no_context_from_the_gen_worker_alone(
    tmp_path: Path,
) -> None:
    """Same sizing on the CI path, with gen_only as the control."""
    no_context = _generate_ci(tmp_path, "gen_only_no_context")
    assert _export(no_context, "totalNodes") == str(NO_CONTEXT_NODES)
    assert _export(no_context, "world_size") == str(NO_CONTEXT_GPUS)

    assert _export(_generate_ci(tmp_path, "gen_only"), "numCtxServers") == "12"


def test_local_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """The regression test for the dual-source-of-truth bug."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert f"{FAKE_KV_VAR}=1" in _export(script, "SERVER_ENV_VARS")
    assert f"export {FAKE_KV_VAR}=1" in script
    assert f"--container-env={FAKE_KV_VAR}" in script


def test_local_generator_never_fabricates_kv_blocks_for_gen_only(tmp_path: Path) -> None:
    """Negative control for the env injection, and the throttle vars it replaces."""
    script = _generate_local(tmp_path, "gen_only")

    assert not FAKE_KV_ASSIGNMENT.search(script)
    gen_worker_env = _export(script, "GEN_WORKER_ENV_VARS")
    assert "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1" in gen_worker_env
    assert "TLLM_BENCHMARK_REQ_QUEUES_SIZE=" in gen_worker_env


def test_local_generator_omits_the_kv_transfer_throttle_for_no_context(tmp_path: Path) -> None:
    """No KV transfer to overlap or throttle when there is no ctx fleet."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert "TLLM_BENCHMARK_REQ_QUEUES_SIZE" not in script
    assert "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP" not in script


def test_ci_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """Env injection on the CI path, with gen_only as the control."""
    no_context = _generate_ci(tmp_path, "gen_only_no_context")
    assert f"{FAKE_KV_VAR}=1" in _export(no_context, "pytestCommand")
    assert f"--container-env={FAKE_KV_VAR}" in no_context

    assert not FAKE_KV_ASSIGNMENT.search(_generate_ci(tmp_path, "gen_only"))


_UCX_MUST_PRECEDE = {"local": "$SERVER_ENV_VARS", "ci": FAKE_KV_VAR}


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_the_ucx_pin_leads_the_whole_pytest_command(tmp_path: Path, generator: str) -> None:
    """`get_ucx_tls_cmd` is a shell *command* prefix, so it must come first."""
    generate = _generate_local if generator == "local" else _generate_ci
    command = _export(generate(tmp_path, "gen_only_no_context"), "pytestCommand").strip('"')

    assert command.startswith("unset "), f"pytestCommand does not lead with the UCX pin: {command}"
    assert "UCX_TLS" in command
    follows = _UCX_MUST_PRECEDE[generator]
    assert command.index("UCX_TLS") < command.index(follows), command


def test_no_context_keeps_the_launcher_for_a_multi_node_gen_worker(tmp_path: Path) -> None:
    """A gen worker spanning nodes needs trtllm-llmapi-launch to be one worker."""
    script = _generate_ci(tmp_path, "gen_only_no_context")

    assert "trtllm-llmapi-launch" in _export(script, "partialPytestCommand")
    assert "$partialPytestCommand" in _export(script, "pytestCommand")


def test_aggregated_draft_lands_the_srun_aggregate_in_the_gen_worker_log() -> None:
    """The redirect exists, is append-only, and is armed by the mode's own var."""
    draft = AGG_DRAFT.read_text()

    assert '>> "$genWorkerLog"' in draft, "the srun aggregate is not appended to the gen log"
    assert 'genWorkerLog="$testOutputDir/gen_server_0.log"' in draft
    assert '> "$genWorkerLog"' not in draft.replace('>> "$genWorkerLog"', "")
    assert f'if [ "${{{FAKE_KV_VAR}:-0}}" = "1" ]; then' in draft
    assert "slurm-${SLURM_JOB_ID}.out" in draft


def test_aggregated_draft_still_shows_a_failure_on_the_console() -> None:
    """Redirecting everything hides the failure unless the tail is echoed back."""
    draft = AGG_DRAFT.read_text()

    assert 'tail -n 200 "$genWorkerLog"' in draft


@pytest.mark.parametrize("benchmark_mode", ["gen_only", "gen_only_no_context", "ctx_only"])
def test_local_generator_emits_a_valid_shell_script(tmp_path: Path, benchmark_mode: str) -> None:
    """A generated script that cannot parse fails at srun time, not at generation."""
    _generate_local(tmp_path, benchmark_mode)
    script_path = tmp_path / benchmark_mode / "slurm_launch.sh"
    subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True)


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_generators_reject_an_unknown_benchmark_mode(tmp_path: Path, generator: str) -> None:
    """A near-miss spelling must fail loudly rather than pick a silent default."""
    generate = _generate_local if generator == "local" else _generate_ci
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        generate(tmp_path, "gen_only_nocontext")

    assert "gen_only_nocontext" in exc_info.value.stderr
