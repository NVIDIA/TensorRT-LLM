# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic target and scripted workflow helpers shared by the domain tests."""

from __future__ import annotations

import json
import re
import socket
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import delivery
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import session_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import load_state
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
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

    refs = root / delivery.ACCURACY_REFERENCES_DIR
    refs.mkdir(parents=True)
    (refs / "gsm8k.yaml").write_text(
        yaml.safe_dump({"openai/gpt-oss-120b": [{"accuracy": 85.6}]}), encoding="utf-8"
    )
    tests_dir = root / delivery.ACCURACY_TESTS_DIR
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_modeling_v2_gpt_oss.py").write_text("def test(): ...\n", encoding="utf-8")
    db = root / delivery.TEST_DB_DIR
    db.mkdir(parents=True)
    (db / "l0_gb300.yml").write_text(
        "- condition: {}\n  tests:\n  - accuracy/test_modeling_v2_gpt_oss.py::T::t\n",
        encoding="utf-8",
    )
    return root


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _verify(root: Path) -> delivery.DeliveryReport:
    return delivery.verify_delivery(
        repo_root=root,
        target_rel=TARGET_REL,
        family="gpt_oss",
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key="openai/gpt-oss-120b",
        accuracy_test="test_modeling_v2_gpt_oss.py",
    )


_STATE = ".trtllm_modeling_v2_state.json"


_DECISION_PATH = re.compile(r"Write your decision file to exactly this path[^`]*`([^`]+)`")


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
    """Per-role scripted decisions, keyed by role and turn number."""

    def __init__(self, ctx, plan=("ready",), **roles):
        self.ctx = ctx
        self.plan = list(plan)
        self.roles = roles

    def __call__(self, role, turn, prompt):
        override = self.roles.get(role)
        if role == "surveyor" and not callable(override):
            return {"decision": self.plan[min(turn, len(self.plan)) - 1], "summary": "drafted"}
        if callable(override):
            return override(turn, prompt)
        if role == "kernel_writer":
            target = _target_from(prompt)
            return {"decision": "ready", "summary": f"landed {target}", "entry": target}
        if role == "assembler":
            return {"decision": "ready", "summary": "assembled"}
        raise AssertionError(f"unscripted role: {role}")


def _target_from(prompt: str) -> str:
    for token in prompt.split("`"):
        if token and " " not in token and not token.startswith("/"):
            return token
    return "unknown"


def _install(monkeypatch, workflow, script):
    calls: list[tuple[str, str]] = []

    def fake_make_agent(
        self, name, system_prompt, *, requires_decision=True, session_mode="stateless"
    ):
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


def healthy_smi(count: int = 1):
    rows = "\n".join(f"{i}, 180000, 183359, 0, 10.3" for i in range(count))

    def runner(argv, env=None, timeout=0):
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=rows + "\n")
        return CommandResult(returncode=0, stdout="ok\n")

    return runner


TARGET_CLASS = "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.model.GptOssModel"


def session_spec(command: str) -> dict:
    """The spec of the session ``command`` starts: what `checks/session.py` wrote for its driver."""
    # The driver's launch is wrapped (`bash -c '<scrub>; exec <python> <driver> --spec <path>'`).
    found = re.search(r"--spec\s+([^\s'\"]+)", command)
    assert found, f"no session spec in: {command}"
    return json.loads(Path(found.group(1)).read_text(encoding="utf-8"))


def session_stdout(
    command: str,
    *,
    score: float = 90.0,
    accepted: int = 320,
    iterations: int = 100,
    text: str = "Paris is the capital of France",
    change=None,
) -> str:
    """What the session driver prints for a session in which everything answers.

    The stand-in for one whole session job: every server starts on devices of its own,
    every rank says it runs the options it was given from the tree measured, every
    request is answered with ``text``, a benchmark scores ``score`` and a drafter accepts
    ``accepted`` tokens over ``iterations`` iterations of each prompt. ``change`` may
    edit each marker's payload before it is printed (``change(marker, payload)``; return
    ``None`` to drop the line): how a test makes one check of the session fail.
    """
    spec = session_spec(command)
    lines: list[str] = []

    def emit(marker: str, payload: dict) -> None:
        if change is not None:
            payload = change(marker, payload)
        if payload is not None:
            lines.append(f"{marker} {json.dumps(payload)}")

    host = socket.gethostname()
    device = 0
    for instance in spec["instances"]:
        own = [str(device + n) for n in range(instance["ranks"])]
        device += instance["ranks"]
        emit(
            session_driver.DEVICES_MARKER,
            {
                "instance": instance["name"],
                "role": instance["role"],
                "index": instance["index"],
                "host": host,
                "devices": own,
                "ranks": instance["ranks"],
                "held_before": [],
            },
        )
        emit(session_driver.SERVER_MARKER, {"name": instance["name"], "healthy": True})
        options = yaml.safe_load(Path(instance["options"]).read_text(encoding="utf-8")) or {}
        for rank in range(instance["ranks"]):
            emit(
                workload_io.RANK_ENGINE_MARKER,
                {
                    "rank": rank,
                    "instance": instance["name"],
                    "role": instance["role"],
                    "options": options,
                    "target": TARGET_CLASS,
                    "origin": f"{spec['tree']}/tensorrt_llm/__init__.py",
                    "host": host,
                    "devices": own,
                },
            )
    seen: list[str] = []
    for step in spec["steps"]:
        ended = {"step": step["id"], "kind": step["kind"], "error": "", "seconds": 0.1}
        if step["kind"] == "requests":
            requests = list(step.get("requests") or ())
            drawn = int((step.get("dataset") or {}).get("num") or 0)
            requests += [
                {"id": f"d{n}", "endpoint": "/v1/completions", "body": {"prompt": f"question {n}"}}
                for n in range(drawn)
            ]
            for index, request in enumerate(requests):
                body = request["body"]
                prompt = body.get("prompt")
                key = json.dumps(prompt)
                tokens = int(body.get("max_tokens") or 1)
                record = {
                    "step": step["id"],
                    "index": index,
                    "id": request["id"],
                    "endpoint": request["endpoint"],
                    "status": 200,
                    "text": text,
                    "finish_reason": "stop",
                    "error": "",
                    "tokens": tokens,
                    "prompt_tokens": len(prompt) if isinstance(prompt, list) else 8,
                    "cached_tokens": 448 if key in seen else 0,
                    "speculative_decoding": {
                        "num_spec_steps": iterations - 1,
                        "total_accepted_draft_tokens": accepted - iterations,
                        "total_draft_tokens": 3 * iterations,
                    },
                }
                seen.append(key)
                emit(workload_io.RECORD_MARKER, record)
            if step.get("decode"):
                ended["decoded"] = {str(i): f"t{i}" for i in step["decode"]}
        elif step["kind"] == "stream":
            record = {
                "step": step["id"],
                "index": 0,
                "id": "stream",
                "endpoint": step["request"]["endpoint"],
                "status": 200,
                "text": text,
                "finish_reason": "stop",
                "events": 2,
                "done": True,
                "error": "",
            }
            emit(workload_io.RECORD_MARKER, record)
        elif step["kind"] == "command":
            Path(step["log"]).parent.mkdir(parents=True, exist_ok=True)
            Path(step["log"]).write_text(f"lm-eval gsm8k average accuracy: {score}\n")
            ended.update(returncode=0, log=step["log"], command=" ".join(step["argv"]))
        emit(session_driver.STEP_MARKER, ended)
    emit(
        session_driver.SESSION_MARKER,
        {
            "deployment": spec["deployment"],
            "host": host,
            "instances": len(spec["instances"]),
            "loads": len(spec["instances"]),
            "started": True,
            "error": "",
            "reaped": [],
            "left": [],
        },
    )
    return "\n".join(lines) + "\n"


def eval_stdout(command: str, score: float = 90.0) -> str:
    """What `trtllm-eval` in a load of its own prints: each rank's record, then its score."""
    config = re.search(r"--config=(\S+)", command)
    tree = re.match(r"cd (\S+) &&", command)
    options = yaml.safe_load(Path(config.group(1)).read_text()) if config else {}
    lines = []
    for rank in range(int((options or {}).get("tensor_parallel_size") or 1)):
        record = {
            "rank": rank,
            "options": options,
            "target": TARGET_CLASS,
            "origin": f"{tree.group(1) if tree else ''}/tensorrt_llm/__init__.py",
        }
        lines.append(f"{workload_io.RANK_ENGINE_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + f"\nlm-eval gsm8k average accuracy: {score}\n"


def passing_shell(score: float = 90.0, accepted: int = 320, iterations: int = 100, change=None):
    """A shell in which every job of the run answers: a session, `trtllm-eval`, a native test."""

    def runner(command, env=None, timeout=0):
        if "session_driver.py" in command:
            said = session_stdout(
                command, score=score, accepted=accepted, iterations=iterations, change=change
            )
            return CommandResult(returncode=0, stdout=said)
        if "trtllm-eval" in command:
            return CommandResult(returncode=0, stdout=eval_stdout(command, score))
        return CommandResult(returncode=0, stdout="Paris is the capital of France\n")

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


def seed_graph(workspace: Path, tasks: list[dict]) -> None:
    """Start a run from ``tasks`` instead of the task list code makes."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "graph.yaml").write_text(yaml.safe_dump({"tasks": tasks}), encoding="utf-8")


def unit_tasks(workspace: Path) -> list[dict]:
    """One assemble/feature task per unit, in order, claiming that unit's checks."""
    contract = yaml.safe_load((workspace / contract_mod.CONTRACT_FILENAME).read_text())
    units: dict[str, list[str]] = {}
    for check in contract["checks"]:
        units.setdefault(check["unit"], []).append(
            f"{check['unit']}:{check['gate']}@{check['member']}"
        )
    tasks, previous = [], None
    for unit, checks in units.items():
        tasks.append(
            {
                "id": unit,
                "kind": "assemble" if unit == "core" else "feature",
                "goal": f"build {unit}",
                "ref": unit,
                "depends_on": [previous] if previous else [],
                # The target's own code: no cross-model regression is owed for it.
                "touches": [f"{TARGET_REL}/{unit}.py"],
                "satisfies": checks,
            }
        )
        previous = unit
    return tasks


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
