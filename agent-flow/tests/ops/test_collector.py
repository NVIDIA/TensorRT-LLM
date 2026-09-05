"""The status dict, the cross-project index, and the plain-text renderers."""

from __future__ import annotations

import json
import textwrap
import time

import pytest

from agent_flow.ops import collector, dashboard, mailbox, project, tray, worktree

LEDGER = """
| 09-04 10:00 | AC-01 | | abc1234def | PASS | a.log |
| 09-04 11:00 | AC-02 | | abc1234def | FAIL | b.log |
"""


@pytest.fixture
def filled(cfg):
    (cfg.workspace / "PASS-LEDGER.md").write_text(textwrap.dedent(LEDGER))
    (cfg.workspace / "GATE-REASONS.md").write_text(
        "- AC-02 | 09-04 11:05 | reviewer | kernel mismatch; rerun after the fix\n"
    )
    with tray.open_table(cfg) as t:
        t.data["slots"] = {
            "dev1": {"job_id": "1001", "holder": "coder", "purpose": "build", "since": "now"},
            "dev2": {"job_id": "1002", "holder": None, "purpose": None, "since": None},
        }
    with worktree.open_table(cfg) as t:
        t.data["slots"] = {"wt-1": {"holder": "reviewer", "purpose": "review", "since": "now"}}
    return cfg


def test_collect_gathers_every_source(filled):
    mailbox.configure(filled)
    mailbox.send("please look", to="coder", blocking=True)
    st = collector.collect(filled)
    assert st["project"] == "test-project"
    assert st["scoreboard"] == {"green": 1, "total": 2}
    assert [g["id"] for g in st["gates"]] == ["AC-01", "AC-02"]
    assert st["gates"][1]["reason"] == "kernel mismatch; rerun after the fix"
    assert [r["key"] for r in st["allocations"]] == ["dev1", "dev2"]
    assert st["allocations"][0]["holder"] == "coder"
    assert [r["key"] for r in st["worktrees"]] == ["wt-1"]
    assert st["mailboxes"]["counts"] == {"pending": 1, "blocking": 1, "overdue": 0}
    assert st["flow"]["state"] in ("idle", "running")


def test_collect_on_an_empty_project_still_returns_every_key(cfg):
    st = collector.collect(cfg)
    assert st["scoreboard"] == {"green": 0, "total": 0}
    assert st["gates"] == [] and st["allocations"] == [] and st["worktrees"] == []
    assert st["mailboxes"]["counts"]["pending"] == 0


def test_reservation_tables_are_read_without_the_lock(filled):
    # a stale lock file must not stop a viewer
    tray.table_path(filled).with_suffix(".lock").write_text("")
    assert collector.allocation_rows(filled)[0]["key"] == "dev1"


def test_a_corrupt_table_reads_as_empty(cfg):
    tray.table_path(cfg).write_text("{ not json")
    assert collector.allocation_rows(cfg) == []


def test_narration_is_off_by_default_and_degrades(cfg):
    assert collector.collect(cfg)["narration"] == collector.NARRATION_UNAVAILABLE
    # asked for, but nothing configured
    assert collector.collect(cfg, narration=True)["narration"] == collector.NARRATION_UNAVAILABLE


def test_narration_uses_the_configured_command(cfg, tmp_path):
    fake = tmp_path / "narrator"
    fake.write_text("#!/bin/sh\ncat >/dev/null\necho 'two gates, one red'\n")
    fake.chmod(0o755)
    cfg.raw["dashboard"] = {"narrator_command": str(fake)}
    assert collector.collect(cfg, narration=True)["narration"] == "two gates, one red"


def test_a_broken_narrator_never_breaks_the_status(cfg, tmp_path):
    for body in ("#!/bin/sh\nexit 3\n", "#!/bin/sh\n"):  # non-zero, then empty output
        fake = tmp_path / "narrator"
        fake.write_text(body)
        fake.chmod(0o755)
        cfg.raw["dashboard"] = {"narrator_command": str(fake)}
        st = collector.collect(cfg, narration=True)
        assert st["narration"] == collector.NARRATION_UNAVAILABLE
        assert st["scoreboard"] == {"green": 0, "total": 0}
    cfg.raw["dashboard"] = {"narrator_command": str(tmp_path / "does-not-exist")}
    assert collector.collect(cfg, narration=True)["narration"] == collector.NARRATION_UNAVAILABLE


# -- the cross-project index ----------------------------------------------


def _make_project(root, name, ledger=None, project_keys="", roles=None):
    project.main(["--projects-root", str(root), "new", name, "--no-parent"])
    d = root / name
    if ledger:
        (d / "workspace" / "PASS-LEDGER.md").write_text(textwrap.dedent(ledger))
    toml = d / "agent-flow-ops.toml"
    body = toml.read_text()
    if project_keys:
        body = body.replace('log_dir = "logs"', 'log_dir = "logs"\n' + project_keys)
    if roles is not None:
        body = body.replace(
            'names = ["coder", "reviewer"]', "names = [" + ", ".join(f'"{r}"' for r in roles) + "]"
        )
    toml.write_text(body)
    return d


def test_index_reports_every_column(tmp_path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", LEDGER, 'parent = "zero"\nstart_commit = "deadbeef1234"')
    _make_project(root, "beta")
    (root / "alpha" / "TRAY-RESERVATIONS.json").write_text(
        json.dumps({"slots": {"dev1": {"holder": "alpha"}, "dev2": {"holder": "someone-else"}}})
    )
    (root / "alpha" / "AGENT-NOTICES.jsonl").write_text(
        json.dumps({"type": "notice", "id": "n1", "ts": 1.0, "to": ["coder"], "message": "x"})
        + "\n"
    )
    rows = {r["name"]: r for r in project.index(root)}
    a = rows["alpha"]
    assert (a["parent"], a["start_commit"]) == ("zero", "deadbeef1234")
    assert (a["passing"], a["gates"]) == (1, 2)
    assert a["allocations"] == ["dev1"]
    assert a["pending_notices"] == 1
    assert a["state"] == "idle" and a["final_commit"] is None
    assert rows["beta"]["summary"] == "no ledger rows"


def test_an_acked_notice_stops_counting(tmp_path):
    root = tmp_path / "projects"
    d = _make_project(root, "alpha")
    (d / "AGENT-NOTICES.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"type": "notice", "id": "n1", "ts": 1.0, "to": ["coder"], "message": "x"},
                {"type": "ack", "id": "n1", "ts": 2.0, "role": "coder", "text": "ok"},
                {"type": "notice", "id": "n2", "ts": 3.0, "to": ["coder"], "message": "y"},
                {"type": "not-json-below", "id": "n3"},
            )
        )
        + "\nnot json at all\n"
    )
    assert project.index(root)[0]["pending_notices"] == 1


def test_an_archived_project_shows_its_final_commit(tmp_path):
    root = tmp_path / "projects"
    _make_project(root, "alpha")
    archive_root = tmp_path / "runs"
    folder = archive_root / "run-2026-09-05-alpha"
    folder.mkdir(parents=True)
    (folder / "MANIFEST.json").write_text(
        json.dumps({"run": "alpha", "folder": folder.name, "repo_head": "f" * 40})
    )
    row = project.index(root, archive_root)[0]
    assert row["state"] == "archived"
    assert row["final_commit"] == "f" * 40
    assert row["archived_as"] == "run-2026-09-05-alpha"


def test_a_shared_role_name_is_not_attributed_to_either_project(tmp_path):
    root = tmp_path / "projects"
    for name in ("alpha", "beta"):
        _make_project(root, name)
    slots = {
        "dev1": {"holder": "coder"},  # both projects declare a coder
        "dev2": {"holder": "alpha/coder"},  # unambiguous
    }
    rows = {r["name"]: r for r in project.index(root, alloc_slots=slots)}
    assert rows["alpha"]["allocations"] == ["dev2"]
    assert rows["beta"]["allocations"] == []


def test_a_role_only_one_project_declares_is_attributed(tmp_path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", roles=["surveyor"])
    _make_project(root, "beta")
    rows = {r["name"]: r for r in project.index(root, alloc_slots={"d": {"holder": "surveyor"}})}
    assert rows["alpha"]["allocations"] == ["d"]


def test_holds_rules():
    assert project.holds("alpha", "alpha") is True
    assert project.holds("alpha/coder", "alpha") is True
    assert project.holds("alpha:coder", "alpha") is True
    assert project.holds("coder", "alpha") is False
    assert project.holds("coder", "alpha", ("coder",)) is True
    assert project.holds(None, "alpha") is False


# -- renderers -------------------------------------------------------------


def test_render_status_shows_reasons_only_for_red_gates(filled):
    mailbox.configure(filled)
    mailbox.send("look now", to="coder", blocking=True, due_minutes=1)
    text = dashboard.render_status(collector.collect(filled, now=time.time() + 300))
    assert "test-project" in text
    assert "1/2" in text and "#" in text  # progress bar
    assert "AC-01" in text and "pass" in text and "FAIL" in text
    assert "kernel mismatch" in text
    assert text.count("why:") == 1
    assert "dev1" in text and "wt-1" in text
    assert "1 pending, 1 blocking, 1 overdue" in text
    assert "B! " in text or "B!" in text  # blocking and overdue flags
    assert collector.NARRATION_UNAVAILABLE in text


def test_render_status_on_an_empty_project(cfg):
    text = dashboard.render_status(collector.collect(cfg))
    assert "(no ledger rows)" in text and "(none declared)" in text


def test_progress_bar_edges():
    assert dashboard.progress_bar(0, 0).endswith("0/0")
    assert dashboard.progress_bar(2, 2) == "[" + "#" * 24 + "] 2/2"
    assert dashboard.progress_bar(0, 2) == "[" + "." * 24 + "] 0/2"


def test_render_projects_floats_the_live_project_to_the_top(tmp_path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", LEDGER)
    _make_project(root, "beta")
    rows = project.index(root)
    body = dashboard.render_projects(rows, live="beta")
    lines = [ln for ln in body.splitlines() if ln and not ln.startswith(("project", "-"))]
    assert lines[0].startswith("beta")
    assert "alpha" in body and "1/2" in body
    assert dashboard.render_projects([]) == "no projects\n"


def test_projects_cli_mode(tmp_path, capsys):
    root = tmp_path / "projects"
    _make_project(root, "alpha", LEDGER)
    assert dashboard.main(["--projects", "--projects-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "1/2" in out


def test_projects_cli_without_a_root_explains_itself(capsys):
    assert dashboard.main(["--projects"]) == 2
    assert "no projects root" in capsys.readouterr().err


def test_dashboard_cli_renders_and_dumps_json(filled, config_path, capsys):
    assert dashboard.main(["--config", str(config_path)]) == 0
    assert "GATES" in capsys.readouterr().out
    assert dashboard.main(["--config", str(config_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["scoreboard"] == {"green": 1, "total": 2}


def test_project_list_json_mode(tmp_path, capsys):
    root = tmp_path / "projects"
    _make_project(root, "alpha", LEDGER)
    capsys.readouterr()
    assert project.main(["--projects-root", str(root), "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "alpha" and rows[0]["passing"] == 1
