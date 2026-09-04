"""Tests for the perf-optimize ``kernel_ledger.yaml`` schema."""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_flow.workflows.perf_optimize import kernel_ledger
from agent_flow.workflows.perf_optimize.kernel_ledger import (
    LedgerError,
    cross_validate,
    load_ledger,
)

# --------------------------------------------------------------------- helpers


def _row(kernel: str = "gdn_bf16_state", share: float = 60.0, **overrides) -> dict:
    row = {
        "kernel": kernel,
        "full_name": f"void tensorrt_llm::kernels::{kernel}<...>",
        "share_pct": share,
        "ncu": {
            "duration_us": 41.2,
            "sm_sol_pct": 12.1,
            "mem_sol_pct": 78.5,
            "occupancy_pct": 62.0,
            "bound": "memory",
        },
        "faster": {"disposition": "item", "ref": "opt-001"},
        "fusion": {
            "disposition": "dismissed",
            "neighbors": "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace, step 120)",
            "ref": "multi-consumer-pinned: intermediate feeds residual + norm (torch_trace)",
        },
    }
    row.update(overrides)
    return row


def _ledger(**overrides) -> dict:
    data = {
        "version": 1,
        "source": "rounds/round_1/analysis/nsys_stats.txt",
        "coverage": {
            "enumerated_share_pct": 96.0,
            "other_share_pct": 4.0,
            "min_share_pct": 0.5,
        },
        "kernels": [
            _row(),
            _row(
                "attention_fmha",
                36.0,
                faster={
                    "disposition": "dismissed",
                    "ref": "at-sol-floor: mem SOL 91% (ncu_details_pass1.txt)",
                },
            ),
        ],
    }
    data.update(overrides)
    return data


def _write(tmp_path, data) -> str:
    path = tmp_path / "kernel_ledger.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def _roadmap(*item_ids: str) -> dict:
    return {"items": [{"id": item_id, "status": "pending"} for item_id in item_ids]}


# ------------------------------------------------------------------ load_ledger


def test_valid_ledger_loads(tmp_path):
    data = load_ledger(_write(tmp_path, _ledger()))
    assert [row["kernel"] for row in data["kernels"]] == ["gdn_bf16_state", "attention_fmha"]
    assert data["coverage"]["enumerated_share_pct"] == pytest.approx(96.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(LedgerError, match="not found"):
        load_ledger(tmp_path / "kernel_ledger.yaml")


def test_non_mapping_top_level_raises(tmp_path):
    path = tmp_path / "kernel_ledger.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="mapping at the top level"):
        load_ledger(path)


def test_bound_shorthand_is_normalized(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = "memory-bound"
    ledger["kernels"][1]["ncu"]["bound"] = "SM"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "memory"
    assert data["kernels"][1]["ncu"]["bound"] == "compute"


def test_unknown_bound_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = "quantum"
    with pytest.raises(LedgerError, match="ncu.bound"):
        load_ledger(_write(tmp_path, ledger))


def test_comm_bound_class_is_accepted(tmp_path):
    # Collectives are never put under ncu (kernel replay deadlocks them), so an
    # allreduce row is dispositioned from nsys evidence and reports `comm`.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = {
        "duration_us": 88.0,
        "sm_sol_pct": None,
        "mem_sol_pct": None,
        "occupancy_pct": None,
        "bound": "comm",
        "note": "collective — not captured under ncu (kernel replay deadlocks NCCL)",
    }
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "comm"


@pytest.mark.parametrize("shorthand", ["communication", "comm-bound", "NCCL", "Collective"])
def test_comm_bound_shorthand_is_normalized(tmp_path, shorthand):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = shorthand
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "comm"


def test_ncu_degrade_string_is_allowed(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: no pass captured this stem (3 passes exhausted)"
    ledger["kernels"][0]["bound"] = "memory"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"].startswith("unavailable")
    assert data["kernels"][0]["bound"] == "memory"


def test_ncu_degrade_string_still_owes_a_row_level_bound(tmp_path):
    # A collective never goes under ncu, so the row-level `bound` beside the
    # degrade string is the only bound class it will ever have.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    with pytest.raises(LedgerError, match=r"kernels\[0\].bound"):
        load_ledger(_write(tmp_path, ledger))


def test_collective_row_records_comm_beside_the_degrade_string(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    ledger["kernels"][0]["bound"] = "comm"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["bound"] == "comm"


@pytest.mark.parametrize("shorthand", ["communication", "comm-bound", "NCCL", "Collective"])
def test_row_level_bound_shorthand_is_normalized(tmp_path, shorthand):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    ledger["kernels"][0]["bound"] = shorthand
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["bound"] == "comm"


def test_row_level_bound_must_be_a_known_class(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: no pass captured this stem"
    ledger["kernels"][0]["bound"] = "quantum"
    with pytest.raises(LedgerError, match=r"kernels\[0\].bound"):
        load_ledger(_write(tmp_path, ledger))


def test_empty_ncu_degrade_string_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "  "
    with pytest.raises(LedgerError, match="ncu.*non-empty"):
        load_ledger(_write(tmp_path, ledger))


def test_ncu_metrics_must_be_numbers(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = "high"
    with pytest.raises(LedgerError, match="ncu.mem_sol_pct"):
        load_ledger(_write(tmp_path, ledger))


def test_null_ncu_metric_allowed_with_a_note(tmp_path):
    # A partial capture keeps what it measured; the note says what it lost.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["sm_sol_pct"] = None
    ledger["kernels"][0]["ncu"]["occupancy_pct"] = None
    ledger["kernels"][0]["ncu"]["note"] = "SOL section empty — replay stalled, 3 passes"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["sm_sol_pct"] is None
    assert data["kernels"][0]["ncu"]["duration_us"] == pytest.approx(41.2)


def test_null_ncu_metric_without_a_note_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = None
    with pytest.raises(LedgerError, match="mem_sol_pct.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_blank_note_does_not_excuse_a_null_metric(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["duration_us"] = None
    ledger["kernels"][0]["ncu"]["note"] = "   "
    with pytest.raises(LedgerError, match="duration_us.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_absent_ncu_metric_is_treated_as_null(tmp_path):
    ledger = _ledger()
    del ledger["kernels"][0]["ncu"]["occupancy_pct"]
    with pytest.raises(LedgerError, match="occupancy_pct.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_note_does_not_excuse_a_missing_bound(tmp_path):
    # `bound` is required on a partial capture too — null a metric so this
    # crosses that branch rather than the all-populated one.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["sm_sol_pct"] = None
    ledger["kernels"][0]["ncu"]["bound"] = None
    ledger["kernels"][0]["ncu"]["note"] = "SOL sections came back empty"
    with pytest.raises(LedgerError, match="ncu.bound"):
        load_ledger(_write(tmp_path, ledger))


def test_note_does_not_excuse_a_non_numeric_metric(tmp_path):
    # The note licenses "not measured", not "measured, badly typed".
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = "high"
    ledger["kernels"][0]["ncu"]["note"] = "ncu printed a bare string"
    with pytest.raises(LedgerError, match="ncu.mem_sol_pct"):
        load_ledger(_write(tmp_path, ledger))


def test_both_questions_required_per_row(tmp_path):
    ledger = _ledger()
    del ledger["kernels"][0]["fusion"]
    with pytest.raises(LedgerError, match="answers both questions"):
        load_ledger(_write(tmp_path, ledger))


def test_disposition_enum_is_exact(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["faster"]["disposition"] = "maybe"
    with pytest.raises(LedgerError, match="faster.disposition"):
        load_ledger(_write(tmp_path, ledger))


def test_empty_ref_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["faster"]["ref"] = ""
    with pytest.raises(LedgerError, match="faster.ref"):
        load_ledger(_write(tmp_path, ledger))


def test_fusion_dismissal_requires_observed_neighbors(tmp_path):
    ledger = _ledger()  # the default fusion block is a dismissal
    del ledger["kernels"][0]["fusion"]["neighbors"]
    with pytest.raises(LedgerError, match="fusion.neighbors"):
        load_ledger(_write(tmp_path, ledger))


def test_fusion_item_does_not_require_neighbors(tmp_path):
    # A promoted fusion keeps its adjacency in the roadmap item `ref` names;
    # only the dismissal owes the evidence here.
    ledger = _ledger()
    ledger["kernels"][0]["fusion"] = {"disposition": "item", "ref": "opt-001"}
    data = load_ledger(_write(tmp_path, ledger))
    assert "neighbors" not in data["kernels"][0]["fusion"]


def test_duplicate_kernel_keys_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"].append(copy.deepcopy(ledger["kernels"][0]))
    with pytest.raises(LedgerError, match="duplicates"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_buckets_must_account_for_100(tmp_path):
    # Kernels dropped from the ledger must be rolled into other_share_pct.
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 80.0
    ledger["coverage"]["other_share_pct"] = 4.0
    with pytest.raises(LedgerError, match="~100%"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_sum_tolerates_rounding(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 96.4
    ledger["coverage"]["other_share_pct"] = 4.8  # 101.2 — within tolerance
    load_ledger(_write(tmp_path, ledger))


def test_empty_kernels_list_rejected(tmp_path):
    with pytest.raises(LedgerError, match="'kernels' must be a non-empty list"):
        load_ledger(_write(tmp_path, _ledger(kernels=[])))


def test_errors_are_batched(tmp_path):
    ledger = _ledger(version=2, source="")
    ledger["kernels"][0]["share_pct"] = -1
    with pytest.raises(LedgerError) as excinfo:
        load_ledger(_write(tmp_path, ledger))
    message = str(excinfo.value)
    assert "'version'" in message
    assert "'source'" in message
    assert "share_pct" in message


# --------------------------------------------------------------- cross_validate


def test_item_refs_must_resolve_to_roadmap_ids(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    problems = cross_validate(ledger, _roadmap("opt-999"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "faster.ref" in problems[0]
    assert "opt-001" in problems[0]


def test_refs_to_terminal_items_are_considered(tmp_path):
    # An accepted/failed item still proves the possibility was considered.
    ledger = load_ledger(_write(tmp_path, _ledger()))
    roadmap = {"items": [{"id": "opt-001", "status": "failed"}]}
    assert cross_validate(ledger, roadmap, coverage_target_pct=95.0) == []


def test_coverage_below_target_is_a_problem(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 90.0
    ledger["coverage"]["other_share_pct"] = 10.0
    loaded = load_ledger(_write(tmp_path, ledger))
    problems = cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "coverage_target_pct" in problems[0]


def test_coverage_target_tolerates_rounding(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 94.7
    ledger["coverage"]["other_share_pct"] = 5.3
    loaded = load_ledger(_write(tmp_path, ledger))
    assert cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0) == []


def test_clean_ledger_cross_validates_clean(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    assert cross_validate(ledger, _roadmap("opt-001"), coverage_target_pct=95.0) == []


def test_filename_constant_matches_contract():
    assert kernel_ledger.LEDGER_FILENAME == "kernel_ledger.yaml"
