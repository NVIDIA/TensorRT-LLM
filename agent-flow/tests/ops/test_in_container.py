"""No scheduler, no container: only the command and script assembly."""

from agent_flow.ops import in_container


def test_remote_script_order(cfg, run_root):
    script = in_container.remote_script(cfg, ["pytest", "-q", "a b"])
    lines = script.splitlines()
    assert lines[0] == "set -eo pipefail"
    assert lines[1] == f"export REPO={run_root}/main"
    # env exports come before the prologue, so a prologue line can override them
    assert lines.index("export PYTHONUNBUFFERED=1") < lines.index("export PATH=$REPO/bin:$PATH")
    assert lines[-2] == "cd $REPO"
    assert lines[-1] == "exec pytest -q 'a b'"  # arguments are quoted


def test_remote_script_repo_override(cfg, tmp_path):
    script = in_container.remote_script(cfg, ["true"], repo=tmp_path / "other")
    assert f"export REPO={tmp_path}/other" in script


def test_srun_command_carries_name_image_and_mounts(cfg, run_root):
    cmd = in_container.srun_command(cfg, "1001", 4, 120, "echo hi")
    assert cmd[:2] == ["srun", "--overlap"]
    assert "--jobid=1001" in cmd
    assert "--container-name=test-container" in cmd
    assert "--container-image=/images/test.sqsh" in cmd
    assert f"--container-mounts={run_root}:{run_root}:rw" in cmd
    assert "--time=120" in cmd and "--ntasks=4" in cmd
    assert cmd[-3:] == ["bash", "-c", "echo hi"]


def test_srun_command_omits_an_unset_image(tmp_path, run_root):
    from agent_flow.ops.config import load_config

    p = tmp_path / "ops.toml"
    p.write_text(
        f'[project]\nroot = "{run_root}"\n\n[container]\nname = "c"\nrepo = "{run_root}/main"\n'
    )
    cmd = in_container.srun_command(load_config(p), "7", 1, 10, "true")
    assert not any(x.startswith("--container-image") for x in cmd)
    assert not any(x.startswith("--container-mounts") for x in cmd)
