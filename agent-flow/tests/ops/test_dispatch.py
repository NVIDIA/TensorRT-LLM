"""Dispatcher and client, CPU-only, against stubbed srun / squeue / nvidia-smi.

The stubs are the point. ``srun`` here drops its own options and execs the rest,
so the daemon really starts, really claims a request and really writes an exit
code — the parts that break are the file protocol and the environment, and both
are exercised without a scheduler, a container or a GPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from agent_flow.ops import container_dispatch as cd
from agent_flow.ops import dispatch_client as dc

SRUN_STUB = """#!/bin/sh
# Drop srun's own options, then run whatever it was asked to run, here.
while [ $# -gt 0 ]; do
  case "$1" in --*) shift ;; *) break ;; esac
done
exec "$@"
"""


def _stub(bindir: Path, name: str, body: str) -> Path:
    p = bindir / name
    p.write_text(body)
    p.chmod(0o755)
    return p


@pytest.fixture
def bindir(tmp_path, monkeypatch):
    """A PATH front with srun / squeue / python3 stubs."""
    d = tmp_path / "bin"
    d.mkdir()
    _stub(d, "srun", SRUN_STUB)
    _stub(d, "squeue", "#!/bin/sh\necho RUNNING\n")
    _stub(d, "python3", f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__import__("agent_flow").__file__).resolve().parent.parent)
    )
    return d


@pytest.fixture
def spool(tmp_path):
    d = tmp_path / "spool" / "1001" / "n1"
    d.mkdir(parents=True)
    return d


# -- pure pieces -----------------------------------------------------------


def test_spool_layout_is_one_dir_per_job_and_ntasks(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("DISPATCH_SPOOL_ROOT", raising=False)
    root = dc.spool_root(cfg)
    assert root == cfg.run_root / "dispatch"
    assert dc.spool_dir(root, "1001", 4) == root / "1001" / "n4"
    monkeypatch.setenv("DISPATCH_SPOOL_ROOT", str(tmp_path / "elsewhere"))
    assert dc.spool_root(cfg) == tmp_path / "elsewhere"


def test_liveness_is_the_freshness_of_every_rank_heartbeat(spool):
    assert dc.is_live(spool, 2) is False
    cd._beat(spool, 0)
    assert dc.is_live(spool, 2) is False  # rank 1 still missing
    cd._beat(spool, 1)
    assert dc.is_live(spool, 2) is True
    old = time.time() - dc.STALE_S - 10
    os.utime(spool / "alive.rank1", (old, old))
    assert dc.is_live(spool, 2) is False


def test_start_command_is_one_step_carrying_the_daemon(cfg, spool):
    cmd = dc.start_cmd(cfg, "1001", 4, spool, 60)
    assert cmd[0] == "srun"
    assert "--jobid=1001" in cmd and "--ntasks=4" in cmd and "--time=60" in cmd
    assert f"--container-name={cfg.container_name}" in cmd
    script = cmd[-1]
    assert dc.DAEMON_MODULE in script and str(spool) in script


def test_child_script_applies_cwd_env_and_a_repo_override(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "CFG", cfg)
    other = tmp_path / "other-checkout"
    script = cd.child_script(
        {"argv": ["echo", "hi"], "cwd": str(tmp_path), "env": {"K": "v"}, "repo": str(other)}
    )
    assert f"export REPO={other}" in script  # the override, not cfg.repo
    assert f"cd {tmp_path}" in script and "export K=v" in script
    assert script.rstrip().endswith("exec echo hi'")
    # cwd and env come after the prefix, so they win over the defaults
    assert script.index("export REPO=") < script.index("export K=v")


def test_claim_next_is_fifo_and_renames(spool):
    for i, name in enumerate(("b", "a")):
        p = spool / f"req-{i}-{name}.json"
        p.write_text("{}")
        os.utime(p, (100 + i, 100 + i))
    claimed = cd.claim_next(spool, 0)
    assert claimed == spool / "run-0.json"
    assert not (spool / "req-0-b.json").exists()
    assert cd.claim_next(spool, 1) == spool / "run-1.json"
    assert cd.claim_next(spool, 2) is None


def test_rc_is_written_last_and_atomically(spool):
    cd._write_rc(spool, "abc", 0, 7)
    assert (spool / "rc-abc.rank0").read_text().strip() == "7"
    assert not list(spool.glob(".rc-*.tmp"))


# -- the GPU probe ---------------------------------------------------------


def test_gpu_sample_parses_nvidia_smi(bindir, monkeypatch):
    _stub(bindir, "nvidia-smi", "#!/bin/sh\necho '0, 512, 81920, 37'\necho '1, 0, 81920, 0'\n")
    out = cd.gpu_sample()
    assert out["gpus"] == [
        {"index": 0, "mem_used_mib": 512, "mem_total_mib": 81920, "util_pct": 37},
        {"index": 1, "mem_used_mib": 0, "mem_total_mib": 81920, "util_pct": 0},
    ]


@pytest.mark.parametrize(
    "body, marker",
    [
        ("#!/bin/sh\necho 'boom' >&2\nexit 9\n", "nvidia-smi rc=9"),
        ("#!/bin/sh\necho 'a, b, c, d'\n", "unparsable"),
        ("#!/bin/sh\n", "no GPU rows"),
    ],
)
def test_a_broken_gpu_probe_becomes_the_payload_not_an_exception(bindir, body, marker):
    _stub(bindir, "nvidia-smi", body)
    assert marker in cd.gpu_sample()["error"]


def test_a_missing_nvidia_smi_is_an_error_field(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert "error" in cd.gpu_sample()


def test_write_gpu_publishes_at_both_levels(spool):
    cd.write_gpu(spool, {"at": 1.0, "gpus": []})
    assert json.loads((spool / "gpu.json").read_text())["at"] == 1.0
    assert json.loads((spool.parent / "gpu.json").read_text())["at"] == 1.0
    assert not list(spool.glob(".gpu.json.tmp*"))


# -- running one request ---------------------------------------------------


def test_run_request_writes_the_log_and_the_exit_code(cfg, spool, monkeypatch):
    monkeypatch.setattr(cd, "CFG", cfg)
    rc = cd.run_request(
        spool, 0, {"id": "r1", "argv": ["sh", "-c", "echo out; echo err >&2; exit 3"]}
    )
    assert rc == 3
    assert (spool / "rc-r1.rank0").read_text().strip() == "3"
    body = (spool / "out-r1.rank0.log").read_text()
    assert "out" in body and "err" in body  # streams merged, by design


def test_a_cancel_file_kills_the_child(cfg, spool, monkeypatch):
    monkeypatch.setattr(cd, "CFG", cfg)
    (spool / "cancel-r2").write_text("1\n")
    rc = cd.run_request(spool, 0, {"id": "r2", "argv": ["sleep", "30"]})
    assert rc != 0
    assert "cancelled" in (spool / "out-r2.rank0.log").read_text()


def test_a_timeout_kills_the_child(cfg, spool, monkeypatch):
    monkeypatch.setattr(cd, "CFG", cfg)
    rc = cd.run_request(spool, 0, {"id": "r3", "argv": ["sleep", "30"], "timeout_s": 0.2})
    assert rc != 0
    assert "timeout" in (spool / "out-r3.rank0.log").read_text()


# -- the client ------------------------------------------------------------


def test_status_without_a_daemon_says_how_to_start_one(spool, capsys):
    assert dc.do_status("1001", 1, spool) == dc.EXIT_NO_DAEMON
    out = capsys.readouterr().out
    assert "live: False" in out and "--start" in out


def test_the_client_refuses_a_per_command_fallback_step(cfg, config_path, spool, bindir, capsys):
    """No live dispatcher plus --no-auto-start must NOT spend a step."""
    rc = dc.main(
        [
            "--config",
            str(config_path),
            "--jobid",
            "1001",
            "--no-auto-start",
            "--",
            "true",
        ]
    )
    assert rc == dc.EXIT_NO_DAEMON
    assert "Refusing a per-command fallback step" in capsys.readouterr().err


def test_tailer_prefixes_whole_lines_only(tmp_path, capsys):
    log = tmp_path / "t.log"
    log.write_text("one\ntw")
    t = dc.Tailer(log, sys.stdout, "[r0] ")
    t.pump()
    assert capsys.readouterr().out == "[r0] one\n"
    with log.open("a") as fh:
        fh.write("o\n")
    t.close()
    assert capsys.readouterr().out == "[r0] two\n"


def test_end_to_end_start_dispatch_stop(cfg, config_path, bindir, tmp_path, monkeypatch, capsys):
    """The whole protocol, with srun replaced by 'run it right here'."""
    monkeypatch.setenv("DISPATCH_SPOOL_ROOT", str(tmp_path / "spool"))
    monkeypatch.setattr(dc, "START_WAIT_S", 60.0)
    args = ["--config", str(config_path), "--jobid", "1001"]
    assert dc.main([*args, "--start"]) == 0
    spool = dc.spool_dir(Path(os.environ["DISPATCH_SPOOL_ROOT"]), "1001", 1)
    assert dc.is_live(spool, 1)
    try:
        assert dc.main([*args, "--status"]) == 0
        capsys.readouterr()
        assert dc.main([*args, "--", "sh", "-c", "echo hello-from-the-daemon"]) == 0
        assert "hello-from-the-daemon" in capsys.readouterr().out
        assert dc.main([*args, "--", "sh", "-c", "exit 5"]) == 5
    finally:
        assert dc.main([*args, "--stop"]) == 0
    assert not dc.is_live(spool, 1)
