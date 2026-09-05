import json

from agent_flow.ops import tray, worktree


def run(argv, config_path):
    return tray.main(["--config", str(config_path), *argv])


def test_claim_release_cycle(config_path, run_root):
    assert run(["claim", "dev1", "--holder", "a", "--purpose", "p"], config_path) == 0
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert table["slots"]["dev1"]["holder"] == "a"
    assert (run_root / "TRAY-RESERVATIONS.md").read_text().count("HELD") == 1
    assert run(["release", "dev1", "--holder", "a"], config_path) == 0
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert table["slots"]["dev1"]["holder"] is None


def test_second_holder_is_refused(config_path):
    run(["claim", "dev1", "--holder", "a", "--purpose", "p"], config_path)
    assert run(["claim", "dev1", "--holder", "b", "--purpose", "q"], config_path) == 3
    # the same holder re-claiming is idempotent, not a conflict
    assert run(["claim", "dev1", "--holder", "a", "--purpose", "p2"], config_path) == 0


def test_release_by_a_non_holder_needs_force(config_path):
    run(["claim", "dev1", "--holder", "a", "--purpose", "p"], config_path)
    assert run(["release", "dev1", "--holder", "b"], config_path) == 3
    assert run(["release", "dev1", "--holder", "b", "--force"], config_path) == 0


def test_release_when_free_is_a_no_op(config_path):
    assert run(["release", "dev2", "--holder", "a"], config_path) == 0


def test_alias_resolves_to_the_canonical_slot(config_path, run_root):
    assert run(["claim", "old1", "--holder", "a", "--purpose", "p"], config_path) == 0
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert table["slots"]["dev1"]["holder"] == "a"
    assert "old1" not in table["slots"]


def test_a_table_written_under_an_old_name_is_migrated(config_path, run_root):
    (run_root / "TRAY-RESERVATIONS.json").write_text(
        json.dumps(
            {
                "slots": {"old1": {"holder": "a", "purpose": "p", "since": "then", "job_id": "9"}},
                "history": [],
            }
        )
    )
    assert run(["status", "--no-scheduler"], config_path) == 0
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert "old1" not in table["slots"]
    assert table["slots"]["dev1"]["holder"] == "a"  # the live reservation survived
    assert table["slots"]["dev1"]["job_id"] == "9"  # and so did its job id
    assert any("renamed slot old1 -> dev1" in h for h in table["history"])


def test_undeclared_slots_on_disk_are_kept(config_path, run_root):
    (run_root / "TRAY-RESERVATIONS.json").write_text(
        json.dumps({"slots": {"retired": {"holder": "a"}}, "history": []})
    )
    run(["status", "--no-scheduler"], config_path)
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert "retired" in table["slots"]


def test_set_job(config_path, run_root):
    assert run(["set-job", "dev1", "4242"], config_path) == 0
    table = json.loads((run_root / "TRAY-RESERVATIONS.json").read_text())
    assert table["slots"]["dev1"]["job_id"] == "4242"


def test_wait_times_out_when_the_slot_stays_busy(config_path):
    run(["claim", "dev1", "--holder", "a", "--purpose", "p"], config_path)
    rc = run(
        ["wait", "dev1", "--holder", "b", "--purpose", "q", "--timeout", "0", "--poll", "1"],
        config_path,
    )
    assert rc == 124


def test_unknown_slot_exits(config_path):
    try:
        run(["claim", "nope", "--holder", "a", "--purpose", "p"], config_path)
    except SystemExit as exc:
        assert "unknown allocation" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_worktree_slots(config_path, run_root):
    wt = run_root / "worktrees"
    args = ["--config", str(config_path)]
    assert (
        worktree.main(
            [*args, "claim", "wt-1", "--holder", "a", "--purpose", "p", "--branch", "feat/x"]
        )
        == 0
    )
    table = json.loads((wt / "WORKTREE-RESERVATIONS.json").read_text())
    assert table["slots"]["wt-1"]["branch"] == "feat/x"
    assert worktree.main([*args, "claim", "wt-1", "--holder", "b", "--purpose", "q"]) == 3
    assert worktree.main([*args, "status"]) == 0
    assert worktree.main([*args, "release", "wt-1", "--holder", "a"]) == 0
    table = json.loads((wt / "WORKTREE-RESERVATIONS.json").read_text())
    assert table["slots"]["wt-1"]["holder"] is None
    assert "wt-2" in table["slots"]
