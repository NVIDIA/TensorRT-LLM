import json

import pytest

from agent_flow.ops import ack_notice, notices, notify_agent


@pytest.fixture(autouse=True)
def _bind(cfg):
    notices.configure(cfg)
    return cfg


def test_addressed_notice_is_pending_only_for_its_addressee():
    notices.add("only the coder", to="coder")
    assert [r["id"] for r in notices.pending("coder")] == ["n1"]
    assert notices.pending("reviewer") == []
    assert len(notices.pending()) == 1


def test_notice_to_both_stays_pending_until_both_ack():
    notices.add("both of you", to="all")
    notices.ack("done", "n1", role="coder")
    assert notices.pending("coder") == []
    assert [r["id"] for r in notices.pending("reviewer")] == ["n1"]
    assert notices.pending()[0]["owed"] == ["reviewer"]
    notices.ack("done too", "n1", role="reviewer")
    assert notices.pending() == []


def test_role_less_ack_settles_the_notice_for_everyone():
    notices.add("both of you", to="all")
    notices.ack("legacy client", "n1", role="unknown")
    assert notices.pending() == []


def test_ack_with_no_id_acks_everything_pending_for_that_role():
    notices.add("a", to="coder")
    notices.add("b", to="all")
    notices.add("c", to="reviewer")
    acked = notices.ack("all mine", role="coder")
    assert acked == ["n1", "n2"]
    assert [r["id"] for r in notices.pending("reviewer")] == ["n2", "n3"]


def test_ack_for_an_unknown_id_becomes_a_report_and_cannot_pre_ack():
    ids = notices.ack("I have something to say", "n41", role="coder")
    assert ids[0].startswith("r")
    rec = notices.add("the real n41 arrives later", to="coder")
    assert rec["id"] != "n41"  # ids are minted across every record
    assert notices.pending("coder")  # and the notice is genuinely pending


def test_ack_with_nothing_pending_is_kept_as_a_report():
    ids = notices.ack("nothing to ack", role="coder")
    assert ids[0].startswith("r")
    assert notices.reports()[0]["text"] == "nothing to ack"


def test_followup_promise_and_delivery():
    notices.add("run the thing", to="coder")
    notices.ack("started, result later", "n1", followup=True, role="coder")
    assert [r["id"] for r in notices.awaiting_followup()] == ["n1"]
    notices.followup("it passed", "n1")
    assert notices.awaiting_followup() == []


def test_blocking_pending_only_lists_blocking_notices():
    notices.add("fyi", to="all")
    notices.add("stop now", blocking=True, to="all")
    assert [r["id"] for r in notices.blocking_pending()] == ["n2"]
    assert "BLOCKING notice n2" in notices.render(notices.blocking_pending())


def test_parse_to_rejects_an_unknown_role():
    assert notices.parse_to("coder,reviewer") == ["coder", "reviewer"]
    assert notices.parse_to(None) == ["coder", "reviewer"]
    with pytest.raises(ValueError):
        notices.parse_to("qa")


def test_role_is_inferred_from_the_configured_checkout(cfg, monkeypatch, run_root):
    monkeypatch.delenv("AGENT_NOTICE_ROLE", raising=False)
    monkeypatch.chdir(run_root / "worktrees" / "wt-2")
    assert notices.infer_role() == "reviewer"
    monkeypatch.chdir(run_root / "main")
    assert notices.infer_role() == "coder"
    monkeypatch.chdir(run_root)
    assert notices.infer_role() == "unknown"
    monkeypatch.setenv("AGENT_NOTICE_ROLE", "reviewer")
    assert notices.infer_role() == "reviewer"


def test_queue_is_append_only_jsonl():
    notices.add("one", to="coder")
    notices.ack("ok", "n1", role="coder")
    lines = [json.loads(x) for x in notices.queue_path().read_text().splitlines()]
    assert [r["type"] for r in lines] == ["notice", "ack"]


def test_notify_and_ack_cli_round_trip(config_path, cfg, monkeypatch, capsys):
    cache = cfg.workspace / "test_command.md"
    cache.write_text("# commands\n")
    live = cfg.workspace / "LIVE-NOTES.md"
    live.write_text("# live\n")
    args = ["--config", str(config_path)]
    assert notify_agent.main([*args, "--to", "coder", "--block", "stop and switch"]) == 0
    assert "human-notice" in cache.read_text()
    assert "stop and switch" in live.read_text()
    assert len(notices.blocking_pending()) == 1

    monkeypatch.setenv("AGENT_NOTICE_ROLE", "coder")
    assert ack_notice.main([*args, "--id", "n1", "switched"]) == 0
    assert notices.pending() == []
    assert "human-notice" not in cache.read_text()  # mirrored block dropped
    assert "stop and switch" in live.read_text()  # live notes stay as history
    out = capsys.readouterr().out
    assert "acknowledged n1 as coder" in out


def test_ack_cli_later_then_followup(config_path, monkeypatch):
    args = ["--config", str(config_path)]
    notices.add("long job", to="coder")
    monkeypatch.setenv("AGENT_NOTICE_ROLE", "coder")
    assert ack_notice.main([*args, "--id", "n1", "--later", "started"]) == 0
    assert [r["id"] for r in notices.awaiting_followup()] == ["n1"]
    assert ack_notice.main([*args, "--id", "n1", "--followup", "finished"]) == 0
    assert notices.awaiting_followup() == []
