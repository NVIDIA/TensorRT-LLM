# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic target and scripted workflow helpers shared by the domain tests."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import anchors
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.run.state import load_state
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

TARGET_REL = f"{catalog_paths.MODELING_V2_PACKAGE}/models/gpt_oss/gpt_oss_120b__sm_103__tp1"


_MODELING = '''\
"""A synthetic target, just enough for the static checks."""

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.silu_and_mul import (
    SiluAndMul,
)

from .ops.moe.router import Router


class SyntheticCore(DecoderModel):
    def forward(self, attn_metadata, *args, **kwargs):
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    def step_args(self, md):
        return dict(num_contexts=md.num_contexts, num_ctx_tokens=md.num_ctx_tokens)

    def forward(self, attn_metadata, *args, **kwargs): ...


class DecodeTarget:
    def step_args(self, md):
        return dict(num_contexts=0, num_ctx_tokens=0)

    def forward(self, attn_metadata, *args, **kwargs): ...
'''


_WEIGHTS = '''\
"""A synthetic weight table."""


class SyntheticWeights:
    WEIGHTS: tuple = ()


MODEL_WEIGHTS = SyntheticWeights()
'''


_ENTRY = """\
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Entry(OpWrapper):
    ARCHS = frozenset({Arch.SM_103})
    CELLS = (Cell(why="decode", spec=dict(rows=1)), Cell(why="prefill", spec=dict(rows=2048)))

    def __call__(self, x):
        return x

    def reference(self, x):
        return x
"""


_ENTRY_TEST = """\
import pytest


@pytest.mark.parametrize("cell", op.CELLS)
def test_certified_cells(cell): ...
"""


_SHARED_ENTRIES = (
    ("norm", "flashinfer_rmsnorm"),
    ("gemm", "cublas_mm"),
    ("activation", "silu_and_mul"),
)


def _write_entry(wrapper: Path, test: Path, *, entry: str = _ENTRY) -> None:
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(entry, encoding="utf-8")
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text(_ENTRY_TEST, encoding="utf-8")


def _build_repo(tmp_path: Path, *, modeling: str = _MODELING) -> Path:
    root = tmp_path / "repo"
    target = root / TARGET_REL
    target.mkdir(parents=True)
    (target / "modeling.py").write_text(modeling, encoding="utf-8")
    (target / "weights.py").write_text(_WEIGHTS, encoding="utf-8")

    catalog = root / catalog_paths.CATALOG_DIR
    for category, name in _SHARED_ENTRIES:
        _write_entry(
            catalog / category / f"{name}.py",
            root / catalog_paths.CATALOG_TESTS_DIR / category / f"test_modeling_v2_{name}.py",
        )
    # The one op only this target calls: in its own ops/, tested under its path.
    _write_entry(
        target / catalog_paths.MODEL_OPS_DIRNAME / "moe" / "router.py",
        root / catalog_paths.model_ops_tests_dir(TARGET_REL) / "moe" / "test_modeling_v2_router.py",
    )
    routing = root / catalog_paths.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "routing.py"
    routing.write_text(
        "def route(*args):\n"
        "    from .gpt_oss_120b__sm_103__tp1.modeling import ModelingV2GptOss120bSm103Tp1\n"
        "    return ModelingV2GptOss120bSm103Tp1\n",
        encoding="utf-8",
    )

    refs = root / anchors.REFERENCES_DIR
    refs.mkdir(parents=True)
    (refs / "gsm8k.yaml").write_text(
        yaml.safe_dump({"openai/gpt-oss-120b": [{"accuracy": 85.6}]}), encoding="utf-8"
    )
    return root


_REPO_ROOT = Path(__file__).resolve().parents[4]


_STATE = ".trtllm_modeling_v2_state.json"


_DECISION_PATH = re.compile(
    r"write your decision file, as YAML, to exactly this path[^`]*`([^`]+)`"
)


class FakeLayer:
    """Stands in for one ``AgentLayer``. Writes the decision file the real role would write."""

    def __init__(self, name, script, calls):
        self.name = name
        self.script = script
        self.calls = calls

    async def aforward(self, prompt: str) -> str:
        self.calls.append((self.name, prompt))
        turn = sum(1 for n, _ in self.calls if n == self.name)
        entry = self.script(self.name, turn, prompt)
        if entry is not None:
            match = _DECISION_PATH.search(prompt)
            assert match, f"{self.name}'s dispatch reserved no decision file"
            Path(match.group(1)).write_text(yaml.safe_dump(dict(entry)), encoding="utf-8")
        return f"{self.name} turn {turn} done"

    def __exit__(self, *args) -> None:
        return None


class Script:
    """Per-role scripted decisions, keyed by role and turn number.

    A role given as a callable ``(turn, prompt) -> decision mapping | None`` answers for
    itself. Unscripted, every worker finishes its task.
    """

    def __init__(self, ctx=None, **roles):
        self.ctx = ctx
        self.roles = roles

    def __call__(self, role, turn, prompt):
        override = self.roles.get(role)
        if callable(override):
            return override(turn, prompt)
        if role == "surveyor":
            return {"decision": "ready", "summary": "drafted"}
        if role == "kernel_writer":
            target = _target_from(prompt)
            return {"decision": "ready", "summary": f"landed {target}", "entry": target}
        if role == "assembler":
            return {"decision": "ready", "summary": "assembled"}
        raise AssertionError(f"unscripted role: {role}")


def _target_from(prompt: str) -> str:
    named = re.search(r"the entry `([^`]+)`", prompt)
    return named.group(1) if named else "unknown"


def _install(monkeypatch, workflow, script):
    calls: list[tuple[str, str]] = []

    def fake_make_agent(self, name, system_prompt):
        return FakeLayer(name, script, calls)

    monkeypatch.setattr(TrtllmModelingV2Workflow, "_make_agent", fake_make_agent)
    return calls


def fake_git(*, head: str = "c0ffee1", dirt: str = ""):
    """Answer the git questions the orchestrator asks of the checkout, as `vcs.git_output` does."""

    def run(repo, *args, stdin=None):
        if args[:2] == ("log", "-1"):
            return head + "\n"
        if args[:2] == ("status", "--porcelain"):
            return dirt
        if args[:1] == ("rev-parse",):
            return head + "\n"
        return ""

    return run


def healthy_smi(count: int = 1, free: int = 180000, cap: str = "10.3", **said):
    """A runner of preflight's probe: ``count`` devices of 183359 MiB, a tree that imports."""
    devices = [[str(i), str(free), "183359", cap] for i in range(count)]
    record = {"host": "node-1", "devices": devices, **said}
    seen: list[str] = []

    def runner(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=f"MV2FLOW_PREFLIGHT {json.dumps(record)}\n")

    runner.seen = seen  # type: ignore[attr-defined]
    return runner


def driver_records(accepted: int = 320, iterations: int = 100, prompts: int = 32) -> str:
    """What the engine driver prints for ``prompts`` prompts accepting alike."""
    lines = []
    for n in range(prompts):
        record = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": f"p{n}",
            "tokens": accepted,
            "iterations": iterations,
            "token_ids": list(range(accepted)),
            "text": "Paris is the capital of France",
            "finish_reason": "stop",
            # What a repeated prompt resumes from the cache, where the boot sends one.
            "cached_tokens": 64,
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


#: What each side of a stated disaggregated deployment must say: how the cache moves.
MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
#: A model class under the Modeling V2 package, as a rank that built a target reports it.
TARGET_CLASS = "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.model.GptOssModel"


def rank_lines(
    options: dict, /, *, tree: str = "/repo", ranks: int | None = None, **changes
) -> str:
    """The record each rank of an engine running ``options`` from ``tree`` prints."""
    count = ranks if ranks is not None else int(options.get("tensor_parallel_size") or 1)
    lines = []
    for rank in range(count):
        record = {
            "rank": rank,
            "options": options,
            "target": TARGET_CLASS,
            "origin": f"{tree}/tensorrt_llm/__init__.py",
            **changes,
        }
        lines.append(f"{workload_io.RANK_ENGINE_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def attested(command: str, **changes) -> str:
    """What the ranks of the engine ``command`` starts print: the options it was given, as run."""
    tree = re.match(r"cd (\S+) &&", command)
    spec = re.search(r"--spec '?([^\s']+)", command)
    config = re.search(r"--config=([^\s']+)", command)
    if spec and "llm" not in json.loads(Path(spec.group(1)).read_text(encoding="utf-8")):
        return ""  # a driver that starts no engine in its own process
    if spec:
        options = dict(json.loads(Path(spec.group(1)).read_text(encoding="utf-8"))["llm"])
        options.pop("model", None)
        mode = re.search(r"TRTLLM_MODELING_V2=(\w+)", command)
        options["modeling_v2"] = mode.group(1) if mode else "off"
    elif config:
        options = yaml.safe_load(Path(config.group(1)).read_text(encoding="utf-8"))
    else:
        return ""
    return rank_lines(options, tree=tree.group(1) if tree else "/repo", **changes)


def passing_shell(score: float = 90.0, accepted: int = 320, iterations: int = 100):
    def runner(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            return CommandResult(
                returncode=0,
                stdout=f"lm-eval gsm8k average accuracy: {score}\n" + attested(command),
            )
        # The gates that need per-request structure run the engine driver,
        # which prints one record per prompt.
        if "engine_driver.py" in command:
            return CommandResult(
                returncode=0, stdout=driver_records(accepted, iterations) + attested(command)
            )
        return CommandResult(
            returncode=0,
            stdout=(
                "Paris is the capital of France\n"
                f"[0] Accept rate: {accepted / iterations:.2f} "
                f"(tokens={accepted}, iterations={iterations})\n"
            ),
        )

    return runner


def _workflow(tmp_path, **kwargs):
    kwargs.setdefault("command_runner", healthy_smi())
    kwargs.setdefault("shell_runner", passing_shell())
    # Scripted like every other outside answer: the fixture repo is not a
    # checkout. An explicit `git_runner=None` selects the real, local git.
    kwargs.setdefault("git_runner", fake_git())
    return TrtllmModelingV2Workflow(workspace=tmp_path / "ws", **kwargs)


def _write_vocabulary(workspace: Path, queue):
    doc = {
        "verdict": vocab.VERDICT_AFTER_ADDITIONS if queue else vocab.VERDICT_ASSEMBLABLE,
        "steps": [
            {
                "step": "norm",
                "state": "OK",
                "entry": "catalog/norm/flashinfer_rmsnorm.py",
                "contract_line": "certified sm_103 bf16",
            }
        ],
        "queue": queue,
    }
    (workspace / "vocabulary.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    """Run one git command in ``repo``, with an identity the host may not have."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=mv2flow",
            "-c",
            "user.email=mv2flow@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _state(tmp_path):
    return load_state(tmp_path / "ws" / _STATE)


def _tasks(tmp_path) -> list[dict]:
    return yaml.safe_load((tmp_path / "ws" / "tasks.yaml").read_text())["tasks"]


def _task_id(prompt: str) -> str:
    """The task a worker's dispatch is for, from its decision path."""
    return re.search(r"decisions/([A-Za-z0-9_.-]+)-\d+(?:\.\d+)?\.yaml", prompt).group(1)


def _run(tmp_path, task_file, monkeypatch, script, **kwargs):
    """One process of the run; its agent calls, role and prompt, in order."""
    workflow = _workflow(tmp_path, **kwargs)
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        workflow.run(task_file)
    return calls


def facts(**changes) -> dict:
    """Every fact a task's message may name, as the pipeline supplies them; ``changes`` over them."""
    said = dict(
        task_id="kv_fp8",
        goal="repair what the final pass found, every item:\n- check `kv_fp8:boot` failed",
        attempt=1,
        attempts=3,
        note="",
        worktree="/ws/worktrees/kv_fp8",
        repo="/repo",
        head="",
        behind="",
        target=TARGET_REL,
        arch="sm_103",
        judge="`tensorrt_llm/evaluate/`, `tensorrt_llm/commands/eval.py`",
        package="/flow/trtllm_modeling_v2",
        notes="/ws/operator_notes.md",
        spec="/ws/task.yaml",
        plan="/ws/plan.md",
        vocabulary="/ws/vocabulary.yaml",
        gate="/ws/bin/trtllm-modeling-v2 gate <boot|accuracy|tests> --task /ws/task.yaml "
        "--deployment kv_fp8 --workdir /ws/gates/kv_fp8 --tree /ws/worktrees/kv_fp8",
        history=[],
        earlier=0,
        tasks="/ws/tasks.yaml",
        decisions="/ws/decisions",
        host_facts="/ws/host_facts.md",
        rebuilt=[],
        kept="",
        python="/opt/venv/bin/python",
        prefix="CUDA_VISIBLE_DEVICES=0 TMPDIR=/scratch/tmp",
        devices="0",
        text_only=False,
        spec_notes="",
        path="/ws/decisions/kv_fp8-1.yaml",
        # a deployment
        deployment="kv_fp8",
        first="tp1",
        described="- `kv_fp8`: aggregated, 1 rank(s); judged by its boot, gsm8k",
        options="`/spec/engines/kv_fp8.yaml`",
        checks="`kv_fp8:boot`, `kv_fp8:accuracy[gsm8k]`",
        judged=["`kv_fp8:accuracy[gsm8k]`: relative (the score of `tp1`)"],
        anchors="/ws/anchors.yaml",
        relative="tp1",
        onboarded="c0ffee1",
        cards=[],
        models="tensorrt_llm/_torch/_experimental/modeling_v2/models",
        tests="tests/unittest/_torch/modeling_v2/models/gpt_oss/gpt_oss_120b__sm_103__tp1",
        # the survey
        checkpoint="/ckpt/gpt-oss-120b",
        inventory="## Surface survey\n\n12 entries.",
        # a kernel row
        n=4,
        mode="verify",
        entry="norm/rmsnorm",
        onboard=False,
        row_note="certify rows",
        surface="bf16",
        cells=[{"rows": 2048}],
        tier=3,
        paths=(f"{TARGET_REL}/ops/norm/rmsnorm__r4.py", "tests/x/test_modeling_v2_rmsnorm__r4.py"),
        shared="`catalog/norm/rmsnorm.py` and `tests/catalog/norm/test_modeling_v2_rmsnorm.py`",
        name="rmsnorm",
        module="ops.norm.rmsnorm__r4",
        test_command="cd /ws/worktrees/k4 && PYTHONPATH=/ws/worktrees/k4 python -m pytest t.py -q",
    )
    return {**said, **changes}
