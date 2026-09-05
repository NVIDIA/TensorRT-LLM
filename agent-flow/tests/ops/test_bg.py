"""Background runner, against a tmux server private to the test.

``TMUX_TMPDIR`` puts the server's socket in the temp dir, so nothing here can
see, reuse or kill a session belonging to the person running the suite. Skipped
when tmux is not installed rather than faked: what is being tested is the
handoff to tmux, and a fake tmux would test the fake.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from agent_flow.ops import bg, notices

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


@pytest.fixture(autouse=True)
def private_tmux(tmp_path, monkeypatch, cfg):
    socket_dir = tmp_path / "tmux"
    socket_dir.mkdir()
    monkeypatch.setenv("TMUX_TMPDIR", str(socket_dir))
    monkeypatch.delenv("TMUX", raising=False)
    bg.configure(cfg)
    yield
    subprocess.run(["tmux", "kill-server"], capture_output=True)


def _wait_for(name, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not bg.running(name):
            return True
        time.sleep(0.1)
    return False


def test_start_records_the_command_and_separates_the_streams(cfg, capsys):
    assert (
        bg.main(
            [
                "--config",
                str(cfg.path),
                "start",
                "job1",
                "--",
                "sh",
                "-c",
                "echo to-out; echo to-err >&2",
            ]
        )
        == 0
    )
    assert _wait_for("job1")
    p = bg.paths("job1")
    assert p["out"].read_text().strip() == "to-out"
    assert p["err"].read_text().strip() == "to-err"
    assert p["rc"].read_text().strip() == "0"
    assert p["cmd"].read_text().strip().startswith("sh -c")


def test_status_reports_running_then_the_verdict(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job2", "--", "sh", "-c", "sleep 5"])
    capsys.readouterr()
    assert bg.main(["--config", str(cfg.path), "status", "job2"]) == 0
    assert "job2: RUNNING" in capsys.readouterr().out
    subprocess.run(["tmux", "kill-session", "-t", "bg-job2"], capture_output=True)
    assert _wait_for("job2")
    bg.main(["--config", str(cfg.path), "status", "job2"])
    assert "job2:" in capsys.readouterr().out


def test_a_failing_run_reads_as_failed(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job3", "--", "sh", "-c", "exit 4"])
    assert _wait_for("job3")
    capsys.readouterr()
    bg.main(["--config", str(cfg.path), "status", "job3"])
    assert "FAILED rc=4" in capsys.readouterr().out


def test_status_of_an_unknown_run(cfg, capsys):
    bg.main(["--config", str(cfg.path), "status", "nope"])
    assert "no such run" in capsys.readouterr().out


def test_tail_shows_both_streams(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job4", "--", "sh", "-c", "echo a; echo b >&2"])
    assert _wait_for("job4")
    capsys.readouterr()
    assert bg.main(["--config", str(cfg.path), "tail", "job4", "--lines", "5"]) == 0
    out = capsys.readouterr().out
    assert "--- out" in out and "--- err" in out and "a" in out and "b" in out


def test_list_reports_every_recorded_run(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job5", "--", "true"])
    assert _wait_for("job5")
    capsys.readouterr()
    assert bg.main(["--config", str(cfg.path), "list"]) == 0
    assert "job5" in capsys.readouterr().out


def test_start_without_a_command_is_an_error(cfg, capsys):
    assert bg.main(["--config", str(cfg.path), "start", "job6"]) == 2
    assert "no command given" in capsys.readouterr().err


def test_a_blocking_notice_refuses_the_start(cfg, capsys):
    notices.configure(cfg)
    notices.add("stop everything", blocking=True, to="coder")
    with pytest.raises(SystemExit) as exc:
        bg.main(["--config", str(cfg.path), "start", "job7", "--", "true"])
    assert exc.value.code == bg.EXIT_BLOCKED
    assert "REFUSING TO START" in capsys.readouterr().err
    assert not bg.paths("job7")["cmd"].exists()


def test_wait_returns_when_the_run_finishes(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job8", "--", "sh", "-c", "exit 2"])
    assert bg.main(["--config", str(cfg.path), "wait", "job8", "--timeout", "30"]) == 1
    assert "FAILED rc=2" in capsys.readouterr().out


def test_wait_on_an_unknown_run(cfg, capsys):
    assert bg.main(["--config", str(cfg.path), "wait", "nope"]) == 2
    assert "no such run" in capsys.readouterr().err


def test_wait_stops_early_on_a_pending_blocking_notice(cfg, capsys):
    """A notice already pending when the wait starts only interrupts if blocking.

    A plain notice the agent is deliberately holding until a result lands must
    not make every wait return instantly; a blocking one must stop it at once.
    """
    notices.configure(cfg)
    bg.main(["--config", str(cfg.path), "start", "job9", "--", "sh", "-c", "sleep 30"])
    notices.add("hold this until the run lands", to="coder")
    assert bg.main(["--config", str(cfg.path), "wait", "job9", "--timeout", "6"]) == 124
    notices.add("stop now", blocking=True, to="coder")
    assert bg.main(["--config", str(cfg.path), "wait", "job9", "--timeout", "20"]) == 125
    assert "wait stopped early" in capsys.readouterr().out
    subprocess.run(["tmux", "kill-session", "-t", "bg-job9"], capture_output=True)


def test_sleep_wakes_when_a_watched_job_finishes(cfg, capsys):
    bg.main(["--config", str(cfg.path), "start", "job10", "--", "true"])
    assert _wait_for("job10")
    capsys.readouterr()
    assert bg.main(["--config", str(cfg.path), "sleep", "30", "--job", "job10"]) == 0
    assert "job10" in capsys.readouterr().out


def test_sleep_on_an_unknown_job(cfg, capsys):
    assert bg.main(["--config", str(cfg.path), "sleep", "1", "--job", "nope"]) == 2
    assert "no such run" in capsys.readouterr().err


def test_sleep_with_nothing_to_watch_runs_out_the_clock(cfg, capsys):
    assert bg.main(["--config", str(cfg.path), "sleep", "0.2"]) == 124
    assert "nothing watched" in capsys.readouterr().out
