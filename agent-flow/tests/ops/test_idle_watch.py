"""The idle-allocation watcher, one pass at a time, with squeue stubbed."""

from __future__ import annotations

import json
import os
import time

import pytest

from agent_flow.ops import idle_watch, notices, tray


def _stub_squeue(tmp_path, monkeypatch, state="RUNNING"):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / "squeue"
    p.write_text(f"#!/bin/sh\necho {state}\n")
    p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")


@pytest.fixture
def table(cfg):
    def write(slots):
        tray.table_path(cfg).write_text(json.dumps({"slots": slots, "history": []}))

    return write


def _free_since(cfg, minutes):
    """Pretend the watcher has already seen the slot free for N minutes."""
    return {"dev1": time.time() - minutes * 60}


def test_a_held_allocation_is_never_flagged(cfg, config_path, table, tmp_path, monkeypatch):
    _stub_squeue(tmp_path, monkeypatch)
    table({"dev1": {"job_id": "1001", "holder": "coder", "purpose": "build"}})
    idle_watch.main(["--config", str(config_path), "--once"])
    notices.configure(cfg)
    assert notices.pending() == []


def test_a_freshly_free_allocation_waits_out_the_debounce(
    cfg, config_path, table, tmp_path, monkeypatch
):
    """One pass sees it free for zero minutes, which is a handoff, not an idle."""
    _stub_squeue(tmp_path, monkeypatch)
    table({"dev1": {"job_id": "1001", "holder": None}})
    idle_watch.main(["--config", str(config_path), "--once"])
    notices.configure(cfg)
    assert notices.pending() == []


def test_an_idle_running_allocation_is_notified(
    cfg, config_path, table, tmp_path, monkeypatch, capsys
):
    _stub_squeue(tmp_path, monkeypatch)
    monkeypatch.setattr(idle_watch, "IDLE_MIN", 0)
    table({"dev1": {"job_id": "1001", "holder": None}})
    idle_watch.main(["--config", str(config_path), "--once"])
    notices.configure(cfg)
    pend = notices.pending()
    assert len(pend) == 1
    assert "dev1" in pend[0]["message"] and "UNRESERVED" in pend[0]["message"]
    assert pend[0]["to"] == ["coder"]  # the first configured role
    assert "notified: dev1" in capsys.readouterr().out


def test_the_alert_can_be_addressed_elsewhere(cfg, config_path, table, tmp_path, monkeypatch):
    _stub_squeue(tmp_path, monkeypatch)
    monkeypatch.setattr(idle_watch, "IDLE_MIN", 0)
    table({"dev1": {"job_id": "1001", "holder": None}})
    idle_watch.main(["--config", str(config_path), "--once", "--to", "reviewer"])
    notices.configure(cfg)
    assert notices.pending()[0]["to"] == ["reviewer"]


def test_an_allocation_whose_job_is_gone_is_not_flagged(
    cfg, config_path, table, tmp_path, monkeypatch
):
    """Nothing to reserve: the job is not RUNNING, so silence is correct."""
    _stub_squeue(tmp_path, monkeypatch, state="")
    monkeypatch.setattr(idle_watch, "IDLE_MIN", 0)
    table({"dev1": {"job_id": "1001", "holder": None}})
    idle_watch.main(["--config", str(config_path), "--once"])
    notices.configure(cfg)
    assert notices.pending() == []


def test_squeue_state_reports_gone_and_unknown(tmp_path, monkeypatch):
    _stub_squeue(tmp_path, monkeypatch, state="")
    assert idle_watch.squeue_state("1001") == "GONE"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert idle_watch.squeue_state("1001") == "UNKNOWN"


def test_an_unreadable_table_is_logged_and_survived(config_path, cfg, capsys):
    tray.table_path(cfg).write_text("{ not json")
    idle_watch.main(["--config", str(config_path), "--once"])
    assert "table unreadable" in capsys.readouterr().out


def test_a_missing_table_is_survived(config_path, capsys):
    idle_watch.main(["--config", str(config_path), "--once"])
    assert "table unreadable" in capsys.readouterr().out
