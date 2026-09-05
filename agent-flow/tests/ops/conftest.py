"""Fixtures: a throwaway run directory plus a generated ops config."""

from __future__ import annotations

import textwrap

import pytest

from agent_flow.ops.config import load_config

CONFIG_TEMPLATE = """
[project]
name = "test-project"
root = "{root}"
workspace = "workspace"
log_dir = "logs"

[roles]
names = ["coder", "reviewer"]

[roles.checkouts]
coder = "{root}/main"
reviewer = "{root}/worktrees/wt-2"

[allocations.dev1]
job_id = "1001"
description = "first"
aliases = ["old1"]

[allocations.dev2]
job_id = "1002"
description = "second"

[worktrees]
dir = "{root}/worktrees"
slots = ["wt-1", "wt-2"]

[container]
name = "test-container"
image = "/images/test.sqsh"
mounts = ["{root}:{root}:rw"]
repo = "{root}/main"
default_allocation = "dev1"
env_prologue = ["export PATH=$REPO/bin:$PATH"]

[container.env]
PYTHONUNBUFFERED = "1"
"""


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "main").mkdir()
    (root / "worktrees" / "wt-2").mkdir(parents=True)
    return root


@pytest.fixture
def config_path(tmp_path, run_root):
    path = tmp_path / "agent-flow-ops.toml"
    path.write_text(textwrap.dedent(CONFIG_TEMPLATE).format(root=run_root))
    return path


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch, tmp_path):
    """Never let a real config on this machine leak into a test."""
    for var in (
        "AGENT_FLOW_OPS_CONFIG",
        "AGENT_FLOW_OPS_PROJECT",
        "AGENT_FLOW_OPS_SHARED",
        "AGENT_FLOW_PROJECTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def cfg(config_path):
    return load_config(config_path)


SHARED_TEMPLATE = """
[projects]
root = "{projects}"

[allocations.dev1]
job_id = "1001"

[container]
name = "shared-container"
image = "/images/shared.sqsh"
repo = "{root}/main"
"""


@pytest.fixture
def shared_config_path(tmp_path, run_root):
    path = tmp_path / "shared.toml"
    path.write_text(
        textwrap.dedent(SHARED_TEMPLATE).format(root=run_root, projects=tmp_path / "projects")
    )
    return path
