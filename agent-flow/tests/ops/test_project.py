import textwrap

import pytest

from agent_flow.ops import project
from agent_flow.ops.config import PROJECT_ENV_VAR, load_config, merge


def test_shared_and_project_are_layered(tmp_path, shared_config_path, config_path, run_root):
    cfg = load_config(config_path, shared_config_path)
    # project half comes from the overlay
    assert cfg.project_root == run_root
    assert cfg.project_name == "test-project"
    # shared half comes from the shared file, and the project overlay wins
    # key by key: it names its own container and image, but the shared
    # projects root and allocation come through untouched.
    assert cfg.container_name == "test-container"
    assert cfg.container_image == "/images/test.sqsh"
    assert cfg.allocations["dev1"].description == "first"
    assert cfg.projects_root == tmp_path / "projects"
    assert cfg.shared_path == shared_config_path.resolve()
    assert cfg.project_path == config_path.resolve()


def test_shared_alone_is_enough_when_it_has_everything(tmp_path, shared_config_path):
    cfg = load_config(tmp_path / "missing.toml", shared_config_path)
    assert cfg.container_name == "shared-container"


def test_merge_is_one_level_deep():
    out = merge({"a": {"x": 1, "y": 2}, "b": {"k": 1}}, {"a": {"y": 3}})
    assert out == {"a": {"x": 1, "y": 3}, "b": {"k": 1}}


def test_project_env_var_may_name_a_directory(tmp_path, monkeypatch, run_root):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "agent-flow-ops.toml").write_text(f'[project]\nroot = "{run_root}"\n')
    monkeypatch.setenv(PROJECT_ENV_VAR, str(d))
    assert load_config().project_root == run_root


def test_legacy_run_section_still_loads_with_a_warning(tmp_path, run_root):
    p = tmp_path / "old.toml"
    p.write_text(f'[run]\nroot = "{run_root}"\nworkspace = "ws"\n')
    with pytest.warns(DeprecationWarning, match=r"\[run\] is the old name"):
        cfg = load_config(p)
    assert cfg.project_root == run_root
    assert cfg.workspace == run_root / "ws"


def test_run_root_is_still_available_as_an_alias(cfg, run_root):
    assert cfg.run_root == cfg.project_root == run_root


def test_project_new_scaffolds_and_refuses_to_clobber(tmp_path, capsys):
    root = tmp_path / "projects"
    args = ["--projects-root", str(root)]
    assert project.main([*args, "new", "alpha"]) == 0
    made = root / "alpha"
    assert (made / "agent-flow-ops.toml").is_file()
    assert (made / "workspace").is_dir() and (made / "logs").is_dir()
    body = (made / "agent-flow-ops.toml").read_text()
    assert "[project]" in body and 'name = "alpha"' in body
    assert project.main([*args, "new", "alpha"]) == 3


def test_project_list_reports_state(tmp_path, capsys):
    root = tmp_path / "projects"
    project.main(["--projects-root", str(root), "new", "beta"])
    (root / "beta" / "workspace" / "PASS-LEDGER.md").write_text(
        textwrap.dedent(
            """
            | 09-04 10:00 | AC-01 | | PASS | a.log |
            | 09-04 11:00 | AC-02 | | FAIL | b.log |
            """
        )
    )
    assert project.main(["--projects-root", str(root), "list"]) == 0
    out = capsys.readouterr().out
    assert "beta" in out
    assert "1/2" in out  # the index row: passing/total gates


def test_project_list_with_no_projects(tmp_path, capsys):
    assert project.main(["--projects-root", str(tmp_path / "none"), "list"]) == 0
    assert "no projects under" in capsys.readouterr().out


def test_project_without_a_root_explains_itself(capsys):
    assert project.main(["list"]) == 2
    assert "no projects root" in capsys.readouterr().err
