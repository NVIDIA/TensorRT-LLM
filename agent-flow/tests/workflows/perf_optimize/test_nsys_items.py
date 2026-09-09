"""Tests for the ``nsys_analysis/items.json`` coverage contract.

The timeline analysis ends by writing the opportunities it found, each
with a stable id. These tests pin the guarantee that gives: a campaign
cannot conclude with a measured opportunity that was neither planned as
a roadmap item nor dismissed with evidence.
"""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.perf_optimize import nsys_items


def _write_items(tmp_path, *ids):
    path = tmp_path / nsys_items.ITEMS_FILENAME
    path.write_text(
        json.dumps({"items": [{"id": i, "title": i, "claim": "c"} for i in ids]}),
        encoding="utf-8",
    )
    return path


def _roadmap(rows, item_ids=("opt-001",)):
    return {
        "items": [{"id": i, "status": "pending"} for i in item_ids],
        nsys_items.ROADMAP_KEY: rows,
    }


# --------------------------------------------------------------------------- #
# Reading the skill's artifact
# --------------------------------------------------------------------------- #


def test_load_item_ids_reads_ids_in_file_order(tmp_path):
    path = _write_items(tmp_path, "nsys-02", "nsys-01")
    assert nsys_items.load_item_ids(path) == ["nsys-02", "nsys-01"]


def test_load_item_ids_accepts_the_empty_list(tmp_path):
    # "The analysis found nothing" is a real outcome the skill spells.
    path = tmp_path / nsys_items.ITEMS_FILENAME
    path.write_text(json.dumps({"items": []}), encoding="utf-8")
    assert nsys_items.load_item_ids(path) == []


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps([{"id": "nsys-01"}]),  # a bare list, not the documented object
        json.dumps({"items": {"id": "nsys-01"}}),
    ],
)
def test_load_item_ids_rejects_a_malformed_file(tmp_path, payload):
    path = tmp_path / nsys_items.ITEMS_FILENAME
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(nsys_items.NsysItemsError):
        nsys_items.load_item_ids(path)


def test_load_item_ids_rejects_an_entry_without_an_id(tmp_path):
    # The id is what coverage keys on, so an entry lacking one cannot be
    # accounted for at all — better to stop than to skip it silently.
    path = tmp_path / nsys_items.ITEMS_FILENAME
    path.write_text(json.dumps({"items": [{"title": "no id here"}]}), encoding="utf-8")
    with pytest.raises(nsys_items.NsysItemsError, match="no non-empty 'id'"):
        nsys_items.load_item_ids(path)


def test_items_path_sits_inside_the_pipelines_output_directory(tmp_path):
    assert nsys_items.items_path(tmp_path) == (
        tmp_path / nsys_items.NSYS_ANALYSIS_DIRNAME / nsys_items.ITEMS_FILENAME
    )


# --------------------------------------------------------------------------- #
# The coverage check
# --------------------------------------------------------------------------- #


def test_every_opportunity_planned_or_dismissed_passes():
    roadmap = _roadmap(
        [
            {"id": "nsys-01", "disposition": "item", "ref": "opt-001"},
            {"id": "nsys-02", "disposition": "dismissed", "ref": "below the noise floor"},
        ]
    )
    assert nsys_items.cross_validate(roadmap, ["nsys-01", "nsys-02"]) == []


def test_an_unaccounted_opportunity_is_reported():
    roadmap = _roadmap([{"id": "nsys-01", "disposition": "item", "ref": "opt-001"}])
    problems = nsys_items.cross_validate(roadmap, ["nsys-01", "nsys-02"])
    assert len(problems) == 1
    assert "nsys-02" in problems[0]
    assert "unaccounted for" in problems[0]


def test_an_item_ref_must_name_a_real_roadmap_entry():
    roadmap = _roadmap([{"id": "nsys-01", "disposition": "item", "ref": "opt-999"}])
    problems = nsys_items.cross_validate(roadmap, ["nsys-01"])
    assert any("does not match any roadmap item id" in p for p in problems)


def test_an_item_ref_may_name_an_already_failed_entry():
    # An opportunity whose fix was tried and rejected *was* considered —
    # which is exactly what the block exists to prove.
    roadmap = {
        "items": [{"id": "opt-001", "status": "failed"}],
        nsys_items.ROADMAP_KEY: [{"id": "nsys-01", "disposition": "item", "ref": "opt-001"}],
    }
    assert nsys_items.cross_validate(roadmap, ["nsys-01"]) == []


def test_a_dismissal_needs_evidence_not_an_empty_ref():
    roadmap = _roadmap([{"id": "nsys-01", "disposition": "dismissed", "ref": "  "}])
    problems = nsys_items.cross_validate(roadmap, ["nsys-01"])
    assert any(".ref' must be a non-empty string" in p for p in problems)


def test_an_unknown_disposition_is_rejected():
    roadmap = _roadmap([{"id": "nsys-01", "disposition": "maybe", "ref": "opt-001"}])
    problems = nsys_items.cross_validate(roadmap, ["nsys-01"])
    assert any("must be one of ['item', 'dismissed']" in p for p in problems)


def test_a_duplicate_row_is_rejected():
    roadmap = _roadmap(
        [
            {"id": "nsys-01", "disposition": "item", "ref": "opt-001"},
            {"id": "nsys-01", "disposition": "dismissed", "ref": "also this"},
        ]
    )
    problems = nsys_items.cross_validate(roadmap, ["nsys-01"])
    assert any("duplicates" in p for p in problems)


def test_a_row_naming_an_id_the_analysis_never_found_is_rejected():
    # The block accounts for items.json; it does not extend it. An
    # invented id would otherwise let a real one hide behind the count.
    roadmap = _roadmap([{"id": "nsys-99", "disposition": "dismissed", "ref": "invented"}])
    problems = nsys_items.cross_validate(roadmap, ["nsys-01"])
    assert any("absent from" in p for p in problems)
    assert any("unaccounted for" in p for p in problems)


def test_a_missing_block_is_reported_with_the_count_it_owes():
    problems = nsys_items.cross_validate({"items": []}, ["nsys-01", "nsys-02"])
    assert len(problems) == 1
    assert "2 to account for" in problems[0]


def test_no_opportunities_needs_no_block():
    # The skill writes `{"items": []}` when it found nothing; an absent
    # block is then correct rather than a violation.
    assert nsys_items.cross_validate({"items": [], nsys_items.ROADMAP_KEY: []}, []) == []
