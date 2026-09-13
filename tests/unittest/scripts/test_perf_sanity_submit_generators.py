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

The subject is ``gen_only_no_context``: a **disaggregated topology** (a proxy in
front of one gen worker, with zero context workers, the gen worker fabricating its
own KV blocks) that runs on the **aggregated launch path** (one pytest process owns
the worker, the proxy and the benchmark client). That split is the thing these tests
exist to pin, because the two halves are decided at different call sites:

* the launch path comes from the id prefix -- ``aggr-gen_only_no_context-<config>``,
  the same arrangement ``ctx_only`` already uses out of a disaggregated yaml;
* the config folder comes from the mode -- the *disaggregated* one, because only a
  disagg yaml has a ``worker_config.gen`` to size the job from;
* the sizing comes from the gen worker's ``tp*pp*cp`` alone (con4301: 14 nodes/56
  GPUs -> 2 nodes/8 GPUs, since the 12 ctx workers are not launched); and
* ``TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1`` has to reach the workers.

Half a fix is worse than none: sized-but-not-told leaves every request parked in
DISAGG_GENERATION_INIT waiting for KV that no ctx worker will ever send. Each test
therefore checks a control arm -- ``gen_only`` (the four-role disaggregated path over
the same yaml) or ``ctx_only`` (the same aggregated path over the same yaml) --
because without that pairing the assertions cannot distinguish a working fix from a
vacuously passing one.

No GPU, no cluster and no built ``tensorrt_llm`` is required.
"""

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

# A real, checked-in config. Its own `benchmark.mode` is `e2e`, which is the point:
# the test id alone selects the benchmark mode, so one file serves both arms of the
# A/B with no YAML edit and no duplicate config to drift.
CON4301 = "gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL"

# A second real config, and the only shape the first cannot exercise: it scales the
# gen fleet out by *replicating* the gen server (`num_gen_servers: 4`, tep8 each)
# rather than by widening one. gen_only_no_context overrides that count to 1, so the
# job is one replica's 8 GPUs -- see test_no_context_launches_one_of_a_replicated_gen_fleet.
CON8 = "gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL"
# con8 hardware: 1 ctx server x dep4 (1 node) + 4 gen servers x tep8 (2 nodes each),
# at 4 GPUs per node.
CON8_GEN_ONLY_NODES, CON8_GEN_ONLY_GPUS = 9, 36
CON8_NO_CONTEXT_NODES, CON8_NO_CONTEXT_GPUS = 2, 8

# con4301 hardware: 12 ctx servers x tp4 (1 node each) + 1 gen server x dep8 (2 nodes),
# at 4 GPUs per node.
GEN_ONLY_NODES, GEN_ONLY_GPUS = 14, 56
NO_CONTEXT_NODES, NO_CONTEXT_GPUS = 2, 8
# ctx_only, the sibling mode on this same launch path: one ctx worker at tp4.
CTX_ONLY_NODES, CTX_ONLY_GPUS = 1, 4

# Which launch path each mode takes. This mapping *is* the reshape: the prefix names
# the launch path, not the topology, so gen_only_no_context sits with ctx_only.
AGGREGATED_PATH_MODES = ("ctx_only", "gen_only_no_context")

FAKE_KV_VAR = "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"
# Match the *assignment* only. Both draft templates also mention the var in
# `${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}` guards, which are present in every arm --
# a bare substring search for the name matches those and silently passes.
FAKE_KV_ASSIGNMENT = re.compile(rf'(?:^|[\s"]){FAKE_KV_VAR}=1(?:[\s"]|$)', re.MULTILINE)

# The four-role exports. Their *absence* is how these tests prove the aggregated
# launch path was taken: a mode routed to the disaggregated generator would emit
# these instead of a single `pytestCommand`, and would need a DISAGG_SERVING_TYPE
# to tell one srun's role from another's. gen_only_no_context has one pytest that
# is every role at once, so it must have none of them.
FOUR_ROLE_EXPORTS = (
    "pytestCommandCTXWorker",
    "pytestCommandGENWorker",
    "pytestCommandDisaggServer",
    "pytestCommandBenchmark",
    "DISAGG_SERVING_TYPE",
)


def _test_id_prefix(benchmark_mode: str) -> str:
    """The id prefix a mode is reachable through.

    Not cosmetic: the prefix is what selects the launch path in both generators, and
    the wrong one is a hard rejection rather than a fallback (see
    test_no_context_is_not_reachable_through_the_disaggregated_prefix).
    """
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


def _generate_ci(tmp_path: Path, benchmark_mode: str, prefix: str = "") -> str:
    """Run the CI generator for one test id and return the launch script text.

    The CI generator emits no ``#SBATCH`` header -- the allocation comes from the
    Jenkins stage tuple -- so only the worker/topology exports are asserted here.

    The draft template and the stage name are picked the way ``L0_Test.groovy``
    picks them, from the launch path: ``disaggMultiNodeMode`` is
    ``stageName.contains("Disagg-PerfSanity")`` with no mode or node-count
    condition, and it alone selects the template. So a ``gen_only_no_context``
    stage must *not* carry ``Disagg-`` in its name, or it would be handed the
    four-role template.
    """
    _require_yaml()
    work_dir = tmp_path / (f"ci_{prefix}_{benchmark_mode}" if prefix else f"ci_{benchmark_mode}")
    work_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = work_dir / "slurm_launch.sh"
    # An explicit prefix overrides the mapping so a test can assert on an id form
    # the mapping deliberately never produces.
    on_aggregated_path = (prefix or _test_id_prefix(benchmark_mode)).startswith("aggr")
    prefix = prefix or _test_id_prefix(benchmark_mode)

    test_list = work_dir / "test_list.txt"
    test_list.write_text(
        f"perf/test_perf_sanity.py::test_e2e[{prefix}_upload-{benchmark_mode}-{CON4301}]\n"
    )
    # Jenkins' getPytestBaseCommandLine normally builds this. The generator reads
    # --output-dir and the leading LLM_MODELS_ROOT assignment out of it, and hard
    # fails if either is missing. trtllm-llmapi-launch is present because Jenkins
    # adds it for any stage with nodeCount > 1, and this case is 2 nodes; the
    # aggregated branch keeps whatever Jenkins put there, which is exactly how
    # "launcher iff multi-node" is honoured without the generator deciding it.
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
    """Run the local generator through its *other* entry path: --config-file.

    The two entry paths derive the same three values from different places, and only
    one of them is exercised by the tests above:

    * ``--test-list`` parses the mode, the runtime and the config stem out of a
      pytest id via ``parse_test_string``;
    * ``--config-file --benchmark-mode`` is handed the mode directly and derives the
      runtime itself, from ``detect_config_type`` plus a mode check.

    That second derivation is a second place the "which launch path?" question is
    answered, so it is a second place it can be answered wrongly -- and it was: it
    keyed on ``== "ctx_only"``, so gen_only_no_context fell to the disaggregated
    branch and composed a ``disagg-gen_only_no_context-`` id that nothing mints.
    """
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
    """Return the effective value of `export <name>=...` in a generated script.

    The *last* assignment, because that is what the shell ends up with: the CI
    generator exports `pytestCommand` twice -- Jenkins' inbound line is left in
    place and the rewritten one is appended after it -- so taking the first match
    would assert against the value that never runs.
    """
    matches = re.findall(rf"^export {re.escape(name)}=(.*)$", script, re.MULTILINE)
    assert matches, f"generated script never exports {name}"
    return matches[-1]


# --------------------------------------------------------------------------------
# Which launch path each mode takes.
#
# This is the half of the change that has no other test: everything else here would
# still pass if the mode were wired to the four-role disaggregated generator with a
# ctx fleet of zero. That arrangement starts a proxy and a worker under separate
# sruns keyed by DISAGG_SERVING_TYPE, and there is no role for the process that must
# be worker, proxy and client at once -- so the stage would hang to its TIMEOUT with
# no error to read.
# --------------------------------------------------------------------------------


def test_no_context_takes_the_single_pytest_aggregated_launch_path(tmp_path: Path) -> None:
    """One pytestCommand, none of the four role commands, no DISAGG_SERVING_TYPE."""
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert "export pytestCommand=" in script
    for name in FOUR_ROLE_EXPORTS:
        assert f"export {name}=" not in script, (
            f"{name} means the four-role disaggregated path was taken"
        )


def test_gen_only_takes_the_four_role_disaggregated_launch_path(tmp_path: Path) -> None:
    """Negative control for the path: gen_only over the same yaml still splits roles.

    Without this the assertions above could pass because the generator emits nothing
    at all for either arm.
    """
    script = _generate_local(tmp_path, "gen_only")

    for name in FOUR_ROLE_EXPORTS:
        assert f"export {name}=" in script


def test_no_context_shares_the_launch_path_with_ctx_only(tmp_path: Path) -> None:
    """The two modes that read a disagg yaml on the aggregated path agree on shape.

    ctx_only is the standing proof that the launch path and the topology are
    separable, so it is the right control for gen_only_no_context: whatever differs
    between these two arms is the mode, not the path.
    """
    no_context = _generate_local(tmp_path, "gen_only_no_context")
    ctx_only = _generate_local(tmp_path, "ctx_only")

    for name in ("pytestCommand", "WORKER_ENV_VARS", "SERVER_ENV_VARS", "totalNodes"):
        assert f"export {name}=" in no_context
        assert f"export {name}=" in ctx_only
    # The single lever that separates them, so the shared-path claim above cannot be
    # confused with the two arms being identical.
    assert FAKE_KV_ASSIGNMENT.search(no_context)
    assert not FAKE_KV_ASSIGNMENT.search(ctx_only)


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_no_context_is_not_reachable_through_the_disaggregated_prefix(
    tmp_path: Path, generator: str
) -> None:
    """`disagg-gen_only_no_context-...` must be a hard rejection, not a fallback.

    Both generators allowlist the disagg prefix's modes, and the mode is no longer
    in that list. If it silently fell through instead, a stale test-db line would
    size a 14-node allocation and split roles for a topology that has one.
    """
    generate = _generate_local if generator == "local" else _generate_ci
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        generate(tmp_path, "gen_only_no_context", prefix="disagg")

    assert "gen_only_no_context" in exc_info.value.stderr


# --------------------------------------------------------------------------------
# Sizing. The local generator is the one that emits #SBATCH, so it owns the
# allocation; the CI generator's counts reach Slurm through the Jenkins stage tuple
# and only its exported topology is checkable here.
# --------------------------------------------------------------------------------


def test_local_generator_sizes_gen_only_no_context_from_the_gen_worker_alone(
    tmp_path: Path,
) -> None:
    """The allocation is the gen worker's tp*pp*cp, and nothing else.

    The proxy and the benchmark client are single-process non-MPI children of the
    same pytest, so they add no GPUs and no nodes -- which is why 8 GPUs is the
    whole job even though the topology has three components.
    """
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
    """`num_gen_servers: 4` sizes for ONE replica, and is not refused.

    con4301 cannot test this: its fleet is one wide gen server, so
    ``num_gen_servers`` is already 1 and every assertion about overriding it passes
    vacuously. con8 is the other shape -- four tep8 replicas -- and it is the shape
    the mode used to reject outright, which made the cheap arm unavailable on
    exactly the config that most wanted it (36 GPUs -> 8).

    The override is sound because the per-worker tp/pp/cp is identical across
    replicas, so one replica is a faithful sample of the fleet's decode loop; and it
    is *required* because the proxy is handed the same count in
    server_config.<idx>.yaml, so allocating for 4 while launching 1 would leave it
    waiting on three urls that never bind.
    """
    script = _generate_local(tmp_path, "gen_only_no_context", config=CON8)

    # The aggregated template has no numCtxServers/numGenServers exports -- there is
    # one pytest, not a fleet -- so the allocation is the observable. 2 nodes rather
    # than 8 is exactly the "one replica, not four" claim.
    assert _export(script, "totalNodes") == str(CON8_NO_CONTEXT_NODES)
    assert _export(script, "totalGpus") == str(CON8_NO_CONTEXT_GPUS)
    assert f"#SBATCH --nodes={CON8_NO_CONTEXT_NODES}" in script
    assert f"#SBATCH --ntasks={CON8_NO_CONTEXT_GPUS}" in script


def test_gen_only_keeps_the_replicated_gen_fleet(tmp_path: Path) -> None:
    """Control for the override: plain gen_only on con8 still allocates all four.

    Without this the previous test cannot tell "overrides num_gen_servers for this
    mode" from "ignores num_gen_servers everywhere", which would silently under-size
    every disaggregated job that replicates its gen server.
    """
    script = _generate_local(tmp_path, "gen_only", config=CON8)

    assert _export(script, "numCtxServers") == "1"
    assert _export(script, "numGenServers") == "4"
    assert _export(script, "totalNodes") == str(CON8_GEN_ONLY_NODES)
    assert _export(script, "totalGpus") == str(CON8_GEN_ONLY_GPUS)


@pytest.mark.parametrize("benchmark_mode", AGGREGATED_PATH_MODES)
def test_the_config_file_entry_path_picks_the_same_launch_path(
    tmp_path: Path, benchmark_mode: str
) -> None:
    """--config-file must route both aggregated-path modes exactly as --test-list does.

    Parametrized over *both* modes on purpose: ctx_only is the arm that already
    worked, so it is the control that says this asserts "the branch is keyed on the
    set of modes" rather than "the branch happens to be right for the new mode".

    Asserting on the composed test id and not just the node count is deliberate --
    a wrong runtime here still sizes the allocation correctly (both branches read
    the same worker_config), and only shows up as an id that pytest cannot collect
    after the whole multi-node job has been queued, built and allocated.
    """
    script = _generate_local_from_config_file(tmp_path, benchmark_mode)

    assert f"aggr-{benchmark_mode}-{CON4301}" in script
    assert f"disagg-{benchmark_mode}-" not in script
    for export in FOUR_ROLE_EXPORTS:
        assert f"export {export}=" not in script, (
            f"--config-file routed {benchmark_mode} to the four-role template"
        )


def test_local_generator_sizes_ctx_only_from_the_ctx_worker(tmp_path: Path) -> None:
    """Second control for the sizing: the sibling mode reads the *ctx* worker.

    Both modes take the same branch of get_hardware_config, so this is what proves
    the branch reads the right half of the yaml rather than a shared default that
    happens to be right for one of them.
    """
    script = _generate_local(tmp_path, "ctx_only")

    assert _export(script, "totalNodes") == str(CTX_ONLY_NODES)
    assert _export(script, "totalGpus") == str(CTX_ONLY_GPUS)


def test_ci_generator_sizes_gen_only_no_context_from_the_gen_worker_alone(
    tmp_path: Path,
) -> None:
    """Same sizing on the CI path, with gen_only as the control.

    The CI generator exports `world_size` rather than a GPU total: it is the
    --ntasks the aggregated draft's srun asks for, so a wrong value here launches
    the worker with the wrong rank count rather than mis-sizing the allocation.
    """
    no_context = _generate_ci(tmp_path, "gen_only_no_context")
    assert _export(no_context, "totalNodes") == str(NO_CONTEXT_NODES)
    assert _export(no_context, "world_size") == str(NO_CONTEXT_GPUS)

    assert _export(_generate_ci(tmp_path, "gen_only"), "numCtxServers") == "12"


# --------------------------------------------------------------------------------
# Telling the workers to fabricate KV blocks.
#
# The local generator's env-injection branch used to test `bm_config["mode"]`, which
# holds the mode parsed from the *test id*, while the same key in the CI generator's
# namesake holds the mode read from the config *yaml*. Two dicts, one key name,
# opposite meanings -- so the gen_only_no_context branch could never match and the
# `gen_only` elif always won. Meanwhile the node arithmetic read the yaml correctly
# and did drop the ctx fleet, producing exactly the sized-but-not-told hang
# described in the module docstring.
# --------------------------------------------------------------------------------


def test_local_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """The regression test for the dual-source-of-truth bug.

    One export reaches both children of the single pytest, which is the whole
    reason the aggregated path needs no per-role env plumbing: the gen worker reads
    it for the fake-KV shortcut, and the proxy reads it both to stamp
    request_type="generation_only" onto every request and to pass /health with zero
    ctx servers.
    """
    script = _generate_local(tmp_path, "gen_only_no_context")

    assert f"{FAKE_KV_VAR}=1" in _export(script, "SERVER_ENV_VARS")
    # Also exported at script scope and forwarded into the container, so the
    # precheck gate and slurm_launch_draft.sh guards observe it too.
    assert f"export {FAKE_KV_VAR}=1" in script
    assert f"--container-env={FAKE_KV_VAR}" in script


def test_local_generator_never_fabricates_kv_blocks_for_gen_only(tmp_path: Path) -> None:
    """Negative control for the env injection, and the throttle vars it replaces.

    gen_only gets the KV-transfer throttle pair instead. Both arms must otherwise
    agree: anything else confounds an A/B between them.
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


def test_ci_generator_tells_the_workers_to_fabricate_kv_blocks(tmp_path: Path) -> None:
    """Env injection on the CI path, with gen_only as the control."""
    no_context = _generate_ci(tmp_path, "gen_only_no_context")
    assert f"{FAKE_KV_VAR}=1" in _export(no_context, "pytestCommand")
    assert f"--container-env={FAKE_KV_VAR}" in no_context

    assert not FAKE_KV_ASSIGNMENT.search(_generate_ci(tmp_path, "gen_only"))


# --------------------------------------------------------------------------------
# The two things the aggregated path does not get for free, because it has no
# per-role env prefixes of its own to hang them on.
# --------------------------------------------------------------------------------


# The token that must come *after* the UCX pin in each generator's pytestCommand.
# The pin is a command prefix, so what follows it is whatever carries the worker's
# environment: an assignment inline on the CI path, an already-built variable on the
# local one.
_UCX_MUST_PRECEDE = {"local": "$SERVER_ENV_VARS", "ci": FAKE_KV_VAR}


@pytest.mark.parametrize("generator", ["local", "ci"])
def test_the_ucx_pin_leads_the_whole_pytest_command(tmp_path: Path, generator: str) -> None:
    """`get_ucx_tls_cmd` is a shell *command* prefix, so it must come first.

    It ends in `&&`, not a VAR=VALUE list. Put an assignment ahead of it and that
    assignment binds to the `unset` special builtin -- where it survives as an
    unexported shell variable and never reaches the worker -- so the run would look
    correct and silently drop the pin. The gen worker still starts *with*
    cache_transceiver_config (that is what makes kv_cache_transceiver non-None, and
    the fake-KV path is gated on it), so NIXL still initialises and an unset
    UCX_TLS fails backend creation on every mlx5 device.
    """
    generate = _generate_local if generator == "local" else _generate_ci
    command = _export(generate(tmp_path, "gen_only_no_context"), "pytestCommand").strip('"')

    assert command.startswith("unset "), f"pytestCommand does not lead with the UCX pin: {command}"
    assert "UCX_TLS" in command
    # In front of the environment specifically: that is what a naive ordering would
    # swallow into `unset`.
    follows = _UCX_MUST_PRECEDE[generator]
    assert command.index("UCX_TLS") < command.index(follows), command


def test_no_context_keeps_the_launcher_for_a_multi_node_gen_worker(tmp_path: Path) -> None:
    """A gen worker spanning nodes needs trtllm-llmapi-launch to be one worker.

    con4301's gen worker is dep8 on 4-GPU nodes, so it spans two. On the CI path the
    launcher comes from Jenkins (`pytestUtil`, set iff nodeCount > 1) and the
    aggregated branch must preserve it rather than strip it the way the four-role
    path strips it from the proxy and the client commands -- which is how "launcher
    iff multi-node" is honoured without the generator deciding it.
    """
    script = _generate_ci(tmp_path, "gen_only_no_context")

    assert "trtllm-llmapi-launch" in _export(script, "partialPytestCommand")
    # And the rewritten command runs it, rather than shadowing it with a bare pytest.
    assert "$partialPytestCommand" in _export(script, "pytestCommand")


# --------------------------------------------------------------------------------
# Where the primary metric's log lines land.
#
# `prev_device_step_time` is emitted by the gen worker's *rank 0* only
# (PROFILE_LOG_RANKS defaults to "0"). Under trtllm-llmapi-launch rank 0's executor
# lives in the mgmn_leader_node process, which the launcher runs in the foreground
# as a **sibling** of pytest -- not the trtllm-serve child whose stdout the runner
# redirects. So on any multi-node case those lines go to the srun's stdout and
# nowhere else, and a runner-only redirect cannot capture the one metric this mode
# exists to produce. The aggregated draft therefore redirects the srun aggregate
# into gen_server_0.log: the same filename, and the same role, as the redirect on
# the disaggregated path, which is what lets the parser, the byte-offset windowing
# and the whole upload path stay identical between the two gen modes.
# --------------------------------------------------------------------------------


def test_aggregated_draft_lands_the_srun_aggregate_in_the_gen_worker_log() -> None:
    """The redirect exists, is append-only, and is armed by the mode's own var."""
    draft = AGG_DRAFT.read_text()

    assert '>> "$genWorkerLog"' in draft, "the srun aggregate is not appended to the gen log"
    assert 'genWorkerLog="$testOutputDir/gen_server_0.log"' in draft
    # Appended, not truncated: the pytest inside this srun opens the same path to
    # add the trtllm-serve child's output. Truncating would drop what the worker
    # ranks already logged and would leave one writer's offset past the end, so its
    # next write would punch a NUL-filled hole through the parsed window.
    assert '> "$genWorkerLog"' not in draft.replace('>> "$genWorkerLog"', "")
    # Armed by the mode, not unconditional: a plain aggregated case must keep its
    # output on the console, where cleanup_on_failure's message still means
    # something.
    assert f'if [ "${{{FAKE_KV_VAR}:-0}}" = "1" ]; then' in draft
    assert "slurm-${SLURM_JOB_ID}.out" in draft


def test_aggregated_draft_still_shows_a_failure_on_the_console() -> None:
    """Redirecting everything hides the failure unless the tail is echoed back.

    cleanup_on_failure scancels the job immediately, so without this the stage's
    console would hold nothing but "Aggregated test failed".
    """
    draft = AGG_DRAFT.read_text()

    assert 'tail -n 200 "$genWorkerLog"' in draft


@pytest.mark.parametrize("benchmark_mode", ["gen_only", "gen_only_no_context", "ctx_only"])
def test_local_generator_emits_a_valid_shell_script(tmp_path: Path, benchmark_mode: str) -> None:
    """A generated script that cannot parse fails at srun time, not at generation.

    Whole-file `bash -n`, not a dry run: an early `exit 0` would leave the half of
    the script that actually launches anything unparsed.
    """
    _generate_local(tmp_path, benchmark_mode)
    script_path = tmp_path / benchmark_mode / "slurm_launch.sh"
    subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True)


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
