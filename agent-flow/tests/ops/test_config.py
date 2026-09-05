import pytest

from agent_flow.ops.config import ENV_VAR, OpsConfigError, load_config


def test_loads_declared_sections(cfg, run_root):
    assert cfg.run_root == run_root
    assert cfg.workspace == run_root / "workspace"
    assert cfg.log_dir == run_root / "logs"
    assert cfg.roles == ("coder", "reviewer")
    assert cfg.role_checkouts["reviewer"] == run_root / "worktrees" / "wt-2"
    assert cfg.container_name == "test-container"
    assert cfg.container_env == {"PYTHONUNBUFFERED": "1"}
    assert cfg.env_prologue == ["export PATH=$REPO/bin:$PATH"]
    assert cfg.default_allocation == "dev1"
    assert cfg.worktree_slots == ["wt-1", "wt-2"]


def test_allocations_and_aliases(cfg):
    assert set(cfg.allocations) == {"dev1", "dev2"}
    assert cfg.allocations["dev1"].job_id == "1001"
    assert cfg.alloc_aliases == {"old1": "dev1"}


def test_missing_config_names_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    with pytest.raises(OpsConfigError) as exc:
        load_config()
    msg = str(exc.value)
    assert "agent-flow-ops.toml" in msg
    assert "agent-flow/ops.toml" in msg
    assert "--config" in msg


def test_explicit_path_that_does_not_exist_is_reported(tmp_path):
    with pytest.raises(OpsConfigError, match="no agent-flow ops config found"):
        load_config(tmp_path / "nope.toml")


def test_missing_required_key(tmp_path):
    p = tmp_path / "ops.toml"
    p.write_text('[project]\nworkspace = "ws"\n')
    cfg = load_config(p)
    with pytest.raises(OpsConfigError, match=r"\[project\].root"):
        _ = cfg.project_root
