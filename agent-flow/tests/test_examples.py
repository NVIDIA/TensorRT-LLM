from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _load_example_module():
    path = (Path(__file__).resolve().parents[1] / "examples" /
            "planner_generator_evaluator_workflow.py")
    spec = importlib.util.spec_from_file_location(
        "planner_generator_evaluator_workflow", path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_feeds_agent_final_responses_forward(tmp_path):
    module = _load_example_module()
    workflow = module.PlannerGeneratorEvaluatorWorkflow(tmp_path)

    planner_progress = "## planner\n\nPlanned the work.\n\n"
    generator_progress = planner_progress + "## generator\n\nImplemented it.\n\n"
    evaluator_progress = (generator_progress +
                          "## evaluator\n\nScore: 9/10.\n\n")

    class ScriptedAgent:

        def __init__(self, response_text, action):
            self.response_text = response_text
            self.action = action
            self.prompts = []

        def __call__(self, content):
            self.prompts.append(content)
            self.action()
            return self.response_text

    workflow.planner = ScriptedAgent(
        "planner final answer",
        lambda: (
            workflow.plan_path.write_text(
                "# Plan From File\n\n- Build the feature.\n",
                encoding="utf-8",
            ),
            workflow.progress_path.write_text(planner_progress,
                                              encoding="utf-8"),
        ),
    )
    workflow.generator = ScriptedAgent(
        "generator final answer",
        lambda: workflow.progress_path.write_text(generator_progress,
                                                  encoding="utf-8"),
    )
    workflow.evaluator = ScriptedAgent(
        "evaluator final answer",
        lambda: workflow.progress_path.write_text(evaluator_progress,
                                                  encoding="utf-8"),
    )

    workflow.run("Build a small Hello World script.")

    planner_prompt = workflow.planner.prompts[0]
    generator_prompt = workflow.generator.prompts[0]
    evaluator_prompt = workflow.evaluator.prompts[0]

    assert f"`{workflow.plan_path}`" in planner_prompt
    assert "forwarded directly to the Generator" in planner_prompt
    assert "append a progress entry" in planner_prompt
    assert "Planner final response:" in generator_prompt
    assert "planner final answer" in generator_prompt
    assert "Generator final response:" in evaluator_prompt
    assert "planner final answer" in evaluator_prompt
    assert "generator final answer" in evaluator_prompt
    assert workflow.progress_path.read_text(
        encoding="utf-8") == evaluator_progress


def test_default_model_constants_respect_env_overrides():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CLAUDE_CODE_DEFAULT_MODEL"] = "claude-test-model"
    env["CODEX_DEFAULT_MODEL"] = "codex-test-model"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("from agent_flow import CLAUDE_CODE_DEFAULT_MODEL, "
             "CODEX_DEFAULT_MODEL; "
             "print(CLAUDE_CODE_DEFAULT_MODEL); "
             "print(CODEX_DEFAULT_MODEL)"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "claude-test-model",
        "codex-test-model",
    ]


def test_examples_use_unified_default_model_constants():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "examples").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "claude-opus-4-7" not in text
        assert "gpt-5.4" not in text


def test_example_help_command_runs_from_repo_root():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "examples/planner_generator_evaluator_workflow.py",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "ModuleNotFoundError: No module named 'agent_flow'" not in combined_output


def test_quick_start_entrypoint_imports_without_module_error():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; "
            "import sys; sys.argv = ['examples/quick_start.py']; "
            "mod = runpy.run_path("
            "'examples/quick_start.py', run_name='__not_main__')",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "ModuleNotFoundError: No module named 'agent_flow'" not in combined_output


def test_workflow_entrypoint_modules_run_without_import_warnings():
    root = Path(__file__).resolve().parents[1]
    for module in (
            "agent_flow.workflows.agent_team.cli",
            "agent_flow.workflows.agent_team.workflow",
            "agent_flow.workflows.modeling_bringup.cli",
    ):
        result = subprocess.run(
            [sys.executable, "-W", "error", "-m", module, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
