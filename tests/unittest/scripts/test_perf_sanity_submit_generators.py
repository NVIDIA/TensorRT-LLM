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

"""Tests for the perf-sanity SLURM launch-script generators.

The generators under ``jenkins/scripts/perf/`` are invoked as scripts by CI and by
ad-hoc local runs, and their only output is a ``slurm_launch.sh``. So these tests
assert on the generated text: it is the artifact that decides how many nodes Slurm
allocates and which env vars the workers see.

The subject is ``gen_only_no_context`` -- a disaggregated topology with **zero**
context workers, where the gen worker fabricates its own KV blocks instead of
receiving them over the cache transceiver. Two facts have to hold together, from
two separate call sites in each generator:

* the job is sized for no ctx fleet (con4301: 14 nodes/56 GPUs -> 2 nodes/8 GPUs), and
* ``TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1`` reaches the workers.

Half a fix is worse than none: sized-but-not-told leaves every request parked in
DISAGG_GENERATION_INIT waiting for KV that no ctx worker will ever send. Each test
therefore checks the ``gen_only`` arm as a negative control -- without that pairing
the assertions cannot distinguish a working fix from a vacuously passing one.

No GPU, no cluster and no built ``tensorrt_llm`` is required.
"""

import difflib
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

# A real, checked-in config. Its own `benchmark.mode` is `e2e`, which is the point:
# the test id alone selects the benchmark mode, so one file serves both arms of the
# A/B with no YAML edit and no duplicate config to drift.
CON4301 = "gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL"

# con4301 hardware: 12 ctx servers x tp4 (1 node each) + 1 gen server x dep8 (2 nodes),
# at 4 GPUs per node.
GEN_ONLY_NODES, GEN_ONLY_GPUS = 14, 56
NO_CONTEXT_NODES, NO_CONTEXT_GPUS = 2, 8

FAKE_KV_VAR = "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"
# Match the *assignment* only. The draft template also mentions the var in
# `${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}` guards, which are present in every arm --
# a bare substring search for the name matches those and silently passes.
FAKE_KV_ASSIGNMENT = re.compile(rf'(?:^|[\s"]){FAKE_KV_VAR}=1(?:[\s"]|$)', re.MULTILINE)


def _require_yaml() -> None:
    pytest.importorskip("yaml", reason="the generators parse the config YAML")


def _generate_local(tmp_path: Path, benchmark_mode: str) -> str:
    """Run the local generator for one test id and return the launch script text."""
    _require_yaml()
    work_dir = tmp_path / benchmark_mode
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"

    subprocess.run(
        [
            sys.executable,
            str(LOCAL_SUBMIT),
            "--test-list",
            f"perf/test_perf_sanity.py::test_e2e[disagg-{benchmark_mode}-{CON4301}]",
            # 'unspecified' omits #SBATCH --partition, so no cluster lookup happens.
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


def _generate_ci(tmp_path: Path, benchmark_mode: str) -> str:
    """Run the CI generator for one test id and return the launch script text.

    The CI generator emits no ``#SBATCH`` header -- the allocation comes from the
    Jenkins stage tuple -- so only the worker/topology exports are asserted here.
    """
    _require_yaml()
    work_dir = tmp_path / f"ci_{benchmark_mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"

    test_list = work_dir / "test_list.txt"
    test_list.write_text(
        f"perf/test_perf_sanity.py::test_e2e[disagg_upload-{benchmark_mode}-{CON4301}]\n"
    )
    # Jenkins' getPytestBaseCommandLine normally builds this. The generator reads
    # --output-dir and the leading LLM_MODELS_ROOT assignment out of it, and hard
    # fails if either is missing.
    script_prefix = work_dir / "prefix.sh"
    script_prefix.write_text(
        'export pytestCommand="LLM_MODELS_ROOT=/dev/null/models '
        "pytest -v perf/test_perf_sanity.py "
        f'--output-dir={work_dir}/out -o junit_logging=out-err"\n'
    )
    srun_args = work_dir / "srun_args.txt"
    srun_args.write_text("")

    subprocess.run(
        [
            sys.executable,
            str(CI_SUBMIT),
            "--draft-launch-sh",
            str(DISAGG_DRAFT),
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
            "GB300-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-Post-Merge",
            "--cluster-name",
            "aws-cmh",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return launch_sh.read_text()


def _export(script: str, name: str) -> str:
    """Return the value of `export <name>=...` from a generated launch script."""
    match = re.search(rf"^export {re.escape(name)}=(.*)$", script, re.MULTILINE)
    assert match is not None, f"generated script never exports {name}"
    return match.group(1)


# --------------------------------------------------------------------------------
# The local generator: the path used for ad-hoc runs, and the one that was broken.
#
# Its env-injection branch used to test `bm_config["mode"]`, which holds the mode
# parsed from the *test id*, while the same key in the CI generator's namesake holds
# the mode read from the config *yaml*. Two dicts, one key name, opposite meanings --
# so the gen_only_no_context branch could never match and the `gen_only` elif always
# won. Meanwhile the node arithmetic read the yaml correctly and did drop the ctx
# fleet, producing exactly the sized-but-not-told hang described in the module
# docstring.
# --------------------------------------------------------------------------------


def test_local_generator_sizes_gen_only_no_context_without_a_ctx_fleet(tmp_path: Path) -> None:
    """Zero ctx servers, and the allocation shrinks to just the gen worker."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert _export(script, "numCtxServers") == "0"
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


def test_local_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """The regression test for the dual-source-of-truth bug.

    Sizing the job for no ctx fleet is only half the change; without the env var the
    gen worker still waits for a KV transfer that can never arrive.
    """
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert f"{FAKE_KV_VAR}=1" in _export(script, "GEN_WORKER_ENV_VARS")
    assert f"{FAKE_KV_VAR}=1" in _export(script, "SERVER_ENV_VARS")
    # Also exported at script scope and forwarded into the container, so the
    # precheck gate and slurm_launch_draft.sh guards observe it too.
    assert f"export {FAKE_KV_VAR}=1" in script
    assert f"--container-env={FAKE_KV_VAR}" in script


def test_local_generator_never_fabricates_kv_blocks_for_gen_only(tmp_path: Path) -> None:
    """Negative control for the env injection, and the throttle vars it replaces.

    gen_only gets the KV-transfer throttle pair instead. The two arms must differ by
    exactly this swap: anything else confounds an A/B between them.
    """
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


def test_local_generator_emits_a_valid_shell_script(tmp_path: Path) -> None:
    """A generated script that cannot parse fails at srun time, not at generation."""
    for benchmark_mode in ("gen_only", "gen_only_no_context"):
        script_path = tmp_path / benchmark_mode / "slurm_launch.sh"
        _generate_local(tmp_path, benchmark_mode)
        subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True)


def test_the_two_gen_arms_differ_only_where_they_must(tmp_path: Path) -> None:
    """The A/B contract, asserted as a whole-file diff.

    Everything outside this set -- model path, parallelism, dataset, UCX transport,
    JIT cache dirs, worker counts on the gen side -- must be byte-identical, or a
    comparison between the two arms measures the harness instead of the mode.
    """
    no_context = _generate_local(tmp_path, "gen_only_no_context")
    gen_only = _generate_local(tmp_path, "gen_only")

    # Normalise the paths that differ only because each arm has its own work dir.
    def normalise(script: str, benchmark_mode: str) -> list:
        script = script.replace(str(tmp_path / benchmark_mode), "WORKDIR")
        script = script.replace(f"disagg-{benchmark_mode}-", "TESTID-")
        return script.splitlines()

    changed = {
        line[2:].split("=")[0].strip()
        for line in difflib.ndiff(
            normalise(no_context, "gen_only_no_context"), normalise(gen_only, "gen_only")
        )
        if line.startswith(("- ", "+ "))
    }
    allowed = {
        "#SBATCH --nodes",
        "#SBATCH --segment",
        "#SBATCH --ntasks",
        "export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
        "export GEN_WORKER_ENV_VARS",
        "export SERVER_ENV_VARS",
        "export numCtxServers",
        "export totalNodes",
        "export totalGpus",
        "export testOutputDir",
        # run_precheck.py is passed --benchmark-mode, so the string differs.
        "export pytestCommandCTXPrecheck",
        "export pytestCommandGENPrecheck",
        '"--container-env',
    }
    assert changed <= allowed, (
        f"unexpected divergence between the arms: {sorted(changed - allowed)}"
    )


# --------------------------------------------------------------------------------
# The CI generator. Same two facts, different code path.
# --------------------------------------------------------------------------------


def test_ci_generator_sizes_gen_only_no_context_without_a_ctx_fleet(tmp_path: Path) -> None:
    """Zero ctx servers on the CI path too, with gen_only as the control."""
    assert _export(_generate_ci(tmp_path, "gen_only_no_context"), "numCtxServers") == "0"
    assert _export(_generate_ci(tmp_path, "gen_only"), "numCtxServers") == "12"


def test_ci_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """Env injection on the CI path, with gen_only as the control."""
    no_context = _generate_ci(tmp_path, "gen_only_no_context")
    assert f"{FAKE_KV_VAR}=1" in _export(no_context, "GEN_WORKER_ENV_VARS")
    assert f"{FAKE_KV_VAR}=1" in _export(no_context, "SERVER_ENV_VARS")
    assert f"--container-env={FAKE_KV_VAR}" in no_context

    assert not FAKE_KV_ASSIGNMENT.search(_generate_ci(tmp_path, "gen_only"))


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_generators_reject_an_unknown_benchmark_mode(tmp_path: Path, generator: str) -> None:
    """A near-miss spelling must fail loudly rather than pick a silent default.

    Both generators allowlist the mode, so a typo has to be an error. Failing open
    would size the job as plain gen_only while the test id claims otherwise.
    """
    generate = _generate_local if generator == "local" else _generate_ci
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        generate(tmp_path, "gen_only_nocontext")

    assert "gen_only_nocontext" in exc_info.value.stderr
