import json
import os
import signal
import time
from pathlib import Path

from agent_flow.jobs.registry import JobRegistry
from agent_flow.jobs.run_detached import main


def test_local_launch_records_running_entry_and_writes_sentinel(tmp_path: Path):
    reg_path = tmp_path / "jobs.json"
    # a trivial command that exits 0
    rc = main(
        [
            "--registry",
            str(reg_path),
            "--handle",
            "hk",
            "--kind",
            "local",
            "--name",
            "n-hk",
            "--",
            "true",
        ]
    )
    assert rc == 0
    entry = JobRegistry(reg_path).get("hk")
    assert entry is not None
    assert entry.kind == "local"
    assert entry.state == "SUBMITTED"
    assert "pid" in entry.handle and "sentinel" in entry.handle


def test_local_launch_passes_shell_active_args_literally(tmp_path: Path):
    # Shell metacharacters must reach the launched program verbatim, not be
    # split/expanded/injected by bash. list2cmdline (Windows quoting) would mangle these.
    reg_path = tmp_path / "jobs.json"
    payload = "a;b|c$x`d`"  # metacharacters that bash would mangle if unquoted
    rc = main(
        [
            "--registry",
            str(reg_path),
            "--handle",
            "hk2",
            "--kind",
            "local",
            "--name",
            "n-hk2",
            "--",
            "printf",
            "%s",
            payload,
        ]
    )
    assert rc == 0
    entry = JobRegistry(reg_path).get("hk2")
    assert entry is not None
    sentinel = Path(entry.handle["sentinel"])
    out = sentinel.parent / "n-hk2.out"
    for _ in range(200):  # detached process — poll up to ~10s for completion
        if sentinel.is_file():
            break
        time.sleep(0.05)
    assert sentinel.is_file(), "sentinel never written"
    assert json.loads(sentinel.read_text())["exit_code"] == 0
    assert out.read_text() == payload  # literal passthrough — fails if quoting is broken


def test_local_launch_records_the_live_process_pid(tmp_path):
    reg_path = tmp_path / "jobs.json"
    rc = main(
        [
            "--registry",
            str(reg_path),
            "--handle",
            "hk3",
            "--kind",
            "local",
            "--name",
            "n-hk3",
            "--",
            "sleep",
            "30",
        ]
    )
    assert rc == 0
    entry = JobRegistry(reg_path).get("hk3")
    pid = int(entry.handle["pid"])
    try:
        # The RECORDED pid must be the live command interpreter, not an
        # already-exited `setsid` parent. A dead setsid parent lingers as a
        # zombie in-process, so ``os.kill(pid, 0)`` alone cannot tell them
        # apart -- discriminate on the process image: the fixed launch records
        # the live ``bash`` pid; the old ``setsid`` launch recorded a defunct
        # ``setsid`` parent whose ``comm`` reads ``setsid`` and state ``Z``.
        os.kill(pid, 0)  # raises OSError if dead
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0]
        assert comm != "setsid", f"recorded the setsid parent, not the command (comm={comm!r})"
        assert state != "Z", f"recorded pid is a zombie, not a live process (state={state!r})"
    finally:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def test_expected_entry_written_before_launch_on_failure(tmp_path: Path, monkeypatch):
    # If the launch step raises, the name-first EXPECTED entry must already be on disk.
    reg_path = tmp_path / "jobs.json"
    import agent_flow.jobs.run_detached as rd

    monkeypatch.setattr(rd, "_launch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        main(
            [
                "--registry",
                str(reg_path),
                "--handle",
                "hk",
                "--kind",
                "slurm",
                "--name",
                "n-hk",
                "--",
                "sbatch",
                "x.sh",
            ]
        )
    except RuntimeError:
        pass
    entry = JobRegistry(reg_path).get("hk")
    assert entry is not None and entry.state == "EXPECTED"
    assert entry.handle.get("name") == "n-hk"
