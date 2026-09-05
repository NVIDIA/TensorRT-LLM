from agent_flow.ops.ledger import LedgerRules, gate_reasons, latest_passes, parse_rows, scoreboard

RULES = LedgerRules(year=2026)


def rows(text):
    return parse_rows(text, RULES)


def test_plain_pass_and_fail_rows():
    out = rows(
        "| 09-04 10:00 | AC-01 | abc1234 | **PASS** | logs/ac01.log |\n"
        "| 09-04 11:00 | AC-02 | def5678 | **FAIL** | logs/ac02.log |\n"
    )
    assert scoreboard(out) == {"AC-01": "pass", "AC-02": "fail"}
    assert latest_passes(out)["AC-01"]["commit"] == "abc1234"
    assert "AC-02" not in latest_passes(out)


def test_newest_row_wins_in_both_directions():
    out = rows(
        "| 09-04 10:00 | AC-01 | | PASS | a.log |\n| 09-04 12:00 | AC-01 | | FAIL | b.log |\n"
    )
    assert scoreboard(out) == {"AC-01": "fail"}
    out = rows(
        "| 09-04 10:00 | AC-01 | | FAIL | a.log |\n| 09-04 12:00 | AC-01 | | PASS | b.log |\n"
    )
    assert scoreboard(out) == {"AC-01": "pass"}


def test_config_variant_pass_counts_but_supporting_pass_does_not():
    out = rows("| 09-04 10:00 | AC-05 enabled full prefill | | PASS | a.log |\n")
    assert scoreboard(out) == {"AC-05": "pass"}
    out = rows("| 09-04 10:00 | AC-05 supporting assets | | PASS | a.log |\n")
    assert out == {}


def test_qualification_overrides_the_not_gate_word_list():
    # "review" is a not-a-gate word, but "qualification" keeps the row a gate row.
    out = rows("| 09-04 10:00 | AC-06 qualification review soak | | PASS | a.log |\n")
    assert scoreboard(out) == {"AC-06": "pass"}


def test_supporting_fail_still_reds_the_gate():
    out = rows("| 09-04 10:00 | AC-07 analysis | | FAIL | a.log |\n")
    assert scoreboard(out) == {"AC-07": "fail"}


def test_mixed_verdict_cell_counts_as_fail():
    out = rows("| 09-04 10:00 | AC-08 | | baseline PASS / enabled UNRUN | a.log |\n")
    assert scoreboard(out) == {"AC-08": "fail"}
    out = rows("| 09-04 10:00 | AC-09 | | DOES NOT HOLD | a.log |\n")
    assert scoreboard(out) == {"AC-09": "fail"}
    out = rows("| 09-04 10:00 | AC-10 | | FALSE GREEN, was PASS | a.log |\n")
    assert scoreboard(out) == {"AC-10": "fail"}


def test_negated_pass_is_not_a_pass():
    assert rows("| 09-04 10:00 | AC-11 | | does not pass yet | a.log |\n") == {}


def test_rows_without_a_verdict_or_a_timestamp_are_skipped():
    assert rows("| 09-04 10:00 | AC-12 | | RUNNING | a.log |\n") == {}
    assert rows("| not-a-date | AC-12 | | PASS | a.log |\n") == {}
    assert rows("no pipe at all\n") == {}
    assert rows("| 09-04 10:00 | AC-12 |\n") == {}


def test_gate_id_and_commit_are_matched_by_pattern_not_position():
    out = rows("| 09-04 10:00 | run 7 | AC-13 | 0123456789ab | PASS | x/y/z.log | note |")
    assert out["AC-13"][0]["commit"] == "0123456789ab"
    assert out["AC-13"][0]["run"] == "ledger:z"


def test_ledger_rows_sort_oldest_first_and_land_at_end_of_minute():
    out = rows(
        "| 09-04 12:00 | AC-14 | | PASS | b.log |\n| 09-04 10:00 | AC-14 | | PASS | a.log |\n"
    )
    epochs = [r["epoch"] for r in out["AC-14"]]
    assert epochs == sorted(epochs)
    assert all(e % 60 == 59 for e in epochs)


def test_gate_reasons_last_line_per_id_wins(tmp_path):
    (tmp_path / "GATE-REASONS.md").write_text(
        "- AC-03 | 09-04 05:10 | reviewer | first reason\n"
        "- AC-03 | 09-04 06:10 | coder | second reason\n"
        "not a reason line\n"
    )
    out = gate_reasons(tmp_path)
    assert out["AC-03"] == {"time": "09-04 06:10", "role": "coder", "text": "second reason"}


def test_gate_reasons_missing_file(tmp_path):
    assert gate_reasons(tmp_path) == {}
