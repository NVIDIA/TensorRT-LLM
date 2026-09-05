"""Named mailboxes: registration, delivery, idempotent sends, fsck, nag."""

from __future__ import annotations

import json
import time

import pytest

from agent_flow.ops import mailbox, notices


@pytest.fixture(autouse=True)
def _bind(cfg):
    mailbox.configure(cfg)
    return cfg


def test_roles_are_mailboxes_and_more_can_be_registered(cfg):
    assert set(mailbox.mailboxes()) == {"coder", "reviewer"}
    mailbox.register("oncall", kind="human", description="the human")
    assert mailbox.mailboxes()["oncall"].kind == "human"
    # persisted, so a second process sees it
    assert "oncall" in json.loads(mailbox.registry_path().read_text())
    assert set(mailbox.configure(cfg)) == {"coder", "reviewer", "oncall"}


def test_registration_rejects_a_bad_name_or_kind():
    with pytest.raises(ValueError, match="single word"):
        mailbox.register("two words")
    with pytest.raises(ValueError, match="kind must be"):
        mailbox.register("bot", kind="robot")


def test_send_to_a_subset_is_pending_only_there():
    mailbox.register("watchdog", kind="service")
    mailbox.send("only these two", to="coder,watchdog")
    assert [r["id"] for r in mailbox.recv("coder")] == ["n1"]
    assert [r["id"] for r in mailbox.recv("watchdog")] == ["n1"]
    assert mailbox.recv("reviewer") == []


def test_send_to_an_unknown_mailbox_names_the_registered_ones():
    with pytest.raises(ValueError, match="unknown mailbox"):
        mailbox.send("hi", to="nobody")


def test_send_is_idempotent_under_a_client_key():
    first, dup, report = mailbox.send("retry me", to="coder", key="abc")
    assert dup is False and report == []
    second, dup2, report2 = mailbox.send("retry me", to="coder", key="abc")
    assert dup2 is True and second["id"] == first["id"] and report2 == []
    assert len([r for r in notices._read() if r.get("type") == "notice"]) == 1
    assert mailbox.find_by_key("abc")["id"] == first["id"]
    assert mailbox.find_by_key("other") is None


def test_ids_are_monotonic_across_record_types():
    """An ack answers a message's id; anything minting its own takes the next.

    Counting only messages was a real bug: a report minted before the message
    that would later take its number pre-settled that message.
    """
    mailbox.send("one", to="coder")
    ids, _, _ = mailbox.ack("answers nothing", "n99", as_="coder")
    assert ids == ["r2"]  # a report, not an ack for a message that may appear
    rec, _, _ = mailbox.send("two", to="coder")
    assert rec["id"] == "n3"


def test_delivery_hooks_run_once_per_project_file(cfg):
    (cfg.workspace / "test_command.md").write_text("# commands\n")
    (cfg.workspace / "LIVE-NOTES.md").write_text("# live\n")
    rec, _, report = mailbox.send("mirror me", to="all", hooks=("command_cache", "live_notes"))
    assert all(d["ok"] for d in report)
    cache = (cfg.workspace / "test_command.md").read_text()
    live = (cfg.workspace / "LIVE-NOTES.md").read_text()
    assert cache.count("mirror me") == 1  # not once per addressee
    assert live.count("mirror me") == 1
    assert rec["id"] in cache


def test_a_missing_delivery_target_is_reported_not_raised(cfg):
    # no command cache on disk: best effort, and the message is still pending
    _, _, report = mailbox.send("no cache here", to="coder", hooks=("command_cache",))
    assert report[0]["ok"] is True and "skipped" in report[0]["detail"]
    assert [r["id"] for r in mailbox.pending("coder")] == ["n1"]


def test_an_unknown_hook_is_reported_not_raised():
    _, _, report = mailbox.send("x", to="coder", hooks=("nope",))
    assert report == [{"to": "coder", "hook": "nope", "ok": False, "detail": "no such hook"}]


def test_a_failing_hook_never_changes_pending_state():
    def boom(box, rec, cfg):
        raise OSError("disk full")

    mailbox.register_hook("boom", boom)
    _, _, report = mailbox.send("x", to="coder", hooks=("boom",))
    assert report[0]["ok"] is False and "disk full" in report[0]["detail"]
    assert len(mailbox.pending("coder")) == 1


def test_tool_result_hook_writes_a_file_for_the_harness(cfg):
    mailbox.send("inject me", to="coder", hooks=("tool_result",))
    body = (cfg.workspace / ".pending-messages.md").read_text()
    assert "inject me" in body


def test_ack_records_the_mailbox_and_how_it_was_decided(monkeypatch):
    mailbox.send("hello", to="coder")
    ids, name, source = mailbox.ack("done", "n1", as_="coder")
    assert (ids, name, source) == (["n1"], "coder", "explicit")
    rec = [r for r in notices._read() if r.get("type") == "ack"][-1]
    assert rec["mailbox"] == "coder" and rec["ack_source"] == "explicit"
    assert mailbox.pending("coder") == []


def test_cwd_inference_is_only_a_logged_fallback(cfg, monkeypatch, run_root):
    monkeypatch.delenv("AGENT_NOTICE_ROLE", raising=False)
    monkeypatch.chdir(run_root / "worktrees" / "wt-2")
    mailbox.send("hello", to="reviewer")
    ids, name, source = mailbox.ack("done")
    assert (name, source) == ("reviewer", "cwd")
    assert [r for r in notices._read() if r.get("type") == "ack"][-1]["ack_source"] == "cwd"
    monkeypatch.setenv("AGENT_NOTICE_ROLE", "coder")
    assert mailbox.whoami() == ("coder", "env")
    assert mailbox.whoami("reviewer") == ("reviewer", "explicit")


def test_cwd_inference_outside_any_checkout_is_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_NOTICE_ROLE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mailbox.whoami() == ("unknown", "unknown")


def test_fsck_is_clean_on_a_healthy_queue():
    mailbox.send("hi", to="coder")
    mailbox.ack("ok", "n1", as_="coder")
    out = mailbox.fsck()
    assert out["ok"] is True and out["messages"] == 1


def test_fsck_finds_orphan_and_early_acks_and_empty_addressees():
    mailbox.send("hi", to="coder")
    notices._append({"type": "ack", "id": "n99", "ts": time.time(), "role": "coder", "text": "?"})
    notices._append({"type": "ack", "id": "n1", "ts": 1.0, "role": "coder", "text": "early"})
    notices._append({"type": "notice", "id": "n50", "ts": time.time(), "to": [], "message": "x"})
    notices._append(
        {"type": "notice", "id": "n51", "ts": time.time(), "to": ["ghost"], "message": "x"}
    )
    out = mailbox.fsck()
    assert out["ok"] is False
    assert [r["id"] for r in out["orphan_acks"]] == ["n99"]
    assert [r["id"] for r in out["early_acks"]] == ["n1"]
    assert [r["id"] for r in out["unaddressed"]] == ["n50"]
    assert out["unknown_addressees"] == [{"id": "n51", "unknown": ["ghost"]}]


def test_a_message_without_due_is_never_overdue():
    mailbox.send("no deadline", to="coder")
    assert mailbox.overdue(time.time() + 86400)["messages"] == []


def test_overdue_lists_unacked_messages_past_their_due():
    mailbox.send("answer me", to="coder", due_minutes=10)
    now = time.time()
    assert mailbox.overdue(now)["messages"] == []
    late = mailbox.overdue(now + 700)["messages"]
    assert [r["id"] for r in late] == ["n1"]
    assert late[0]["late_seconds"] >= 100
    mailbox.ack("done", "n1", as_="coder")
    assert mailbox.overdue(now + 700)["messages"] == []


def test_overdue_lists_promised_followups_that_never_arrived(cfg):
    cfg.raw.setdefault("mailboxes", {})["default_due_minutes"] = 5
    mailbox.send("long job", to="coder")
    mailbox.ack("started", "n1", as_="coder", later=True)
    now = time.time()
    assert mailbox.overdue(now)["followups"] == []
    assert [r["id"] for r in mailbox.overdue(now + 400)["followups"]] == ["n1"]
    notices.followup("finished", "n1")
    assert mailbox.overdue(now + 400)["followups"] == []


def test_status_is_one_call_for_the_dashboard(cfg):
    mailbox.send("plain", to="coder")
    mailbox.send("urgent", to="reviewer", blocking=True, due_minutes=1)
    st = mailbox.status(time.time() + 300)
    assert st["counts"] == {"pending": 2, "blocking": 1, "overdue": 1}
    assert st["overdue_ids"] == ["n2"]
    assert {b["name"] for b in st["mailboxes"]} == {"coder", "reviewer"}


def test_nag_escalates_one_step_per_round(cfg):
    (cfg.workspace / "LIVE-NOTES.md").write_text("# live\n")
    mailbox.register("oncall", kind="human")
    mailbox.register("coder", delivery=["live_notes"], checkout=str(cfg.run_root / "main"))
    mailbox.send("do the thing", to="coder", due_minutes=1)

    live = cfg.workspace / "LIVE-NOTES.md"
    assert live.read_text().count("do the thing") == 1  # the original send

    first = mailbox.nag(time.time() + 300)
    assert [(a["id"], a["step"]) for a in first] == [("n1", "redeliver")]
    assert first[0]["detail"] == "1/1 hooks"
    assert live.read_text().count("do the thing") == 2  # pushed at it again

    second = mailbox.nag(time.time() + 400)
    assert [a["step"] for a in second] == ["mark-overdue"]

    third = mailbox.nag(time.time() + 500)
    assert [a["step"] for a in third] == ["notify-human"]
    assert any("n1 is still unacknowledged" in r["message"] for r in mailbox.recv("oncall"))

    # the ladder stops at the top rung and does not re-post to the human
    fourth = mailbox.nag(time.time() + 600)
    assert [a["step"] for a in fourth] == ["notify-human"]
    assert len(mailbox.recv("oncall")) == 1


def test_nag_without_a_human_mailbox_stays_at_mark_overdue():
    mailbox.send("x", to="coder", due_minutes=1)
    for _ in range(3):
        acted = mailbox.nag(time.time() + 300)
    assert acted[0]["step"] == "mark-overdue"
    assert "no human mailbox" in acted[0]["detail"]


def test_nothing_overdue_means_nothing_to_do():
    mailbox.send("x", to="coder")
    assert mailbox.nag() == []


def test_cli_round_trip(config_path, cfg, capsys):
    args = ["--config", str(config_path)]
    assert mailbox.main([*args, "register", "oncall", "--kind", "human"]) == 0
    assert mailbox.main([*args, "list"]) == 0
    assert "oncall" in capsys.readouterr().out
    assert mailbox.main([*args, "send", "--to", "coder", "--key", "k1", "hello", "there"]) == 0
    assert "n1 -> coder" in capsys.readouterr().out
    assert mailbox.main([*args, "send", "--to", "coder", "--key", "k1", "hello", "there"]) == 0
    assert "already sent as n1" in capsys.readouterr().out
    assert mailbox.main([*args, "recv", "--as", "coder"]) == 0
    assert "hello there" in capsys.readouterr().out
    assert mailbox.main([*args, "ack", "--as", "coder", "--id", "n1", "done"]) == 0
    assert "acknowledged n1 as coder" in capsys.readouterr().out
    assert mailbox.main([*args, "recv", "--as", "coder"]) == 0
    assert "nothing pending" in capsys.readouterr().out
    assert mailbox.main([*args, "fsck"]) == 0
    assert "ok" in capsys.readouterr().out
    assert mailbox.main([*args, "nag"]) == 0
    assert "nothing overdue" in capsys.readouterr().out
    assert mailbox.main([*args, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["pending"] == 0


def test_cli_send_to_an_unknown_mailbox_exits_two(config_path, capsys):
    assert mailbox.main(["--config", str(config_path), "send", "--to", "ghost", "x"]) == 2
    assert "unknown mailbox" in capsys.readouterr().err


def test_cli_fsck_reports_problems_with_exit_one(config_path, capsys):
    notices._append({"type": "ack", "id": "n42", "ts": time.time(), "role": "coder", "text": "?"})
    assert mailbox.main(["--config", str(config_path), "fsck"]) == 1
    assert "PROBLEMS FOUND" in capsys.readouterr().out
    assert mailbox.main(["--config", str(config_path), "fsck", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["orphan_acks"][0]["id"] == "n42"


def test_notice_cli_is_a_thin_wrapper(config_path, cfg, monkeypatch, capsys):
    """The old CLIs keep their behaviour on top of the mailbox layer."""
    from agent_flow.ops import ack_notice, notify_agent

    (cfg.workspace / "test_command.md").write_text("# commands\n")
    args = ["--config", str(config_path)]
    assert notify_agent.main([*args, "--to", "coder", "--due", "1", "switch now"]) == 0
    assert "HUMAN NOTICE" in (cfg.workspace / "test_command.md").read_text()
    assert mailbox.overdue(time.time() + 120)["messages"][0]["id"] == "n1"
    monkeypatch.setenv("AGENT_NOTICE_ROLE", "coder")
    assert ack_notice.main([*args, "--id", "n1", "switched"]) == 0
    assert "acknowledged n1 as coder (mailbox from env)" in capsys.readouterr().out
    assert mailbox.overdue(time.time() + 120)["messages"] == []


def test_notify_cli_dedupes_on_a_key(config_path, capsys):
    args = ["--config", str(config_path), "--key", "once"]
    assert notify_main(args, "first try") == 0
    assert notify_main(args, "first try") == 0
    assert "already sent as n1" in capsys.readouterr().out


def notify_main(args, message):
    from agent_flow.ops import notify_agent

    return notify_agent.main([*args, message])
