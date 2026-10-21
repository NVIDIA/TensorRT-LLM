# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vocabulary.yaml`` — the typed dispatch queue."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.onboard import vocabulary as vocab


def _ok_step(**overrides):
    step = {
        "step": "rmsnorm_q",
        "values": {"dtype": "bf16", "head_dim": 128},
        "state": vocab.STATE_OK,
        "entry": "catalog/norm/flashinfer_rmsnorm.py",
        "contract_line": "certified: sm_103, bf16, head_dim in {64, 128}",
    }
    step.update(overrides)
    return step


def _doc(**overrides):
    doc = {
        "verdict": vocab.VERDICT_ASSEMBLABLE,
        "steps": [_ok_step()],
        "queue": [],
    }
    doc.update(overrides)
    return doc


def test_minimal_assemblable_document_parses():
    parsed = vocab.parse_vocabulary(_doc())
    assert parsed.verdict == vocab.VERDICT_ASSEMBLABLE
    assert parsed.queue == ()
    assert parsed.is_ceiling is False
    assert parsed.counts()[vocab.STATE_OK] == 1


def test_ok_row_without_contract_line_is_rejected():
    """The quoted line is the evidence; without it nobody can audit the claim."""
    doc = _doc(steps=[_ok_step(contract_line="")])
    with pytest.raises(vocab.VocabularyError, match="contract_line"):
        vocab.parse_vocabulary(doc)


def test_gap_row_without_a_tier_is_rejected():
    """A GAP has to say which rung supplies it before anything else."""
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[{"step": "moe_gelu", "state": vocab.STATE_GAP}],
        queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe_gelu"}],
    )
    with pytest.raises(vocab.VocabularyError, match="tier"):
        vocab.parse_vocabulary(doc)


def test_a_wrapping_tier_must_name_what_it_wraps():
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[{"step": "moe_gelu", "state": vocab.STATE_GAP, "tier": vocab.TIER_TRTLLM_OP}],
        queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe_gelu"}],
    )
    with pytest.raises(vocab.VocabularyError, match="candidate"):
        vocab.parse_vocabulary(doc)


def test_a_writing_tier_must_say_what_it_rejected():
    """Otherwise the ladder is a way to skip straight to writing a kernel."""
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[{"step": "exotic_norm", "state": vocab.STATE_GAP, "tier": vocab.TIER_TRITON}],
        queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "exotic_norm"}],
    )
    with pytest.raises(vocab.VocabularyError, match="tried"):
        vocab.parse_vocabulary(doc)


def test_a_writing_tier_with_its_search_recorded_parses():
    """The case that matters for a model the engine does not implement."""
    parsed = vocab.parse_vocabulary(
        _doc(
            verdict=vocab.VERDICT_AFTER_ADDITIONS,
            steps=[
                {
                    "step": "exotic_norm",
                    "state": vocab.STATE_GAP,
                    "tier": vocab.TIER_NATIVE_TORCH,
                    "tried": "no trtllm op; nothing under modules/** computes it",
                }
            ],
            queue=[
                {
                    "n": 1,
                    "mode": vocab.MODE_ONBOARD,
                    "target": "exotic_norm",
                    "tier": vocab.TIER_NATIVE_TORCH,
                }
            ],
        )
    )
    item = parsed.queue[0]
    assert item.tier == vocab.TIER_NATIVE_TORCH
    # The orchestrator needs to know this dispatch owes an implementation,
    # not just a wrapper, so it can budget turns differently.
    assert item.writes_implementation is True
    assert "native torch" not in item.tier_brief.lower() or "compose" in item.tier_brief


def test_a_wrapping_dispatch_is_not_marked_as_writing():
    parsed = vocab.parse_vocabulary(
        _doc(
            verdict=vocab.VERDICT_AFTER_ADDITIONS,
            steps=[
                {
                    "step": "moe_gelu",
                    "state": vocab.STATE_GAP,
                    "tier": vocab.TIER_INTREE_CALLABLE,
                    "candidate": "tensorrt_llm._torch.modules.x.y",
                }
            ],
            queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe_gelu", "tier": 2}],
        )
    )
    assert parsed.queue[0].writes_implementation is False


def _library_doc(candidate):
    return _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[
            {
                "step": "gemma_rmsnorm",
                "state": vocab.STATE_GAP,
                "tier": vocab.TIER_KERNEL_LIBRARY,
                "candidate": candidate,
            }
        ],
        queue=[
            {
                "n": 1,
                "mode": vocab.MODE_ONBOARD,
                "target": "norm/flashinfer_gemma_rmsnorm",
                "tier": vocab.TIER_KERNEL_LIBRARY,
            }
        ],
    )


def test_a_flashinfer_callable_is_a_wrapping_rung_below_native_torch():
    """A kernel the engine already ships is reused before anyone writes one."""
    parsed = vocab.parse_vocabulary(_library_doc("flashinfer.norm.gemma_rmsnorm"))
    item = parsed.queue[0]
    assert item.writes_implementation is False
    assert "FlashInfer" in item.tier_brief
    assert vocab.TIER_INTREE_CALLABLE < vocab.TIER_KERNEL_LIBRARY < vocab.TIER_NATIVE_TORCH


@pytest.mark.parametrize(
    "candidate", ["torch.ops.trtllm.flashinfer_rmsnorm", "some_pip_package.rmsnorm"]
)
def test_the_kernel_library_rung_names_a_pinned_library_callable(candidate):
    """Tier 3 reuses a library the engine pins; it is not a door to a new dependency."""
    with pytest.raises(vocab.VocabularyError, match="not in a kernel library the engine pins"):
        vocab.parse_vocabulary(_library_doc(candidate))


def test_the_tier_survives_a_round_trip_through_run_state():
    """A resume must not silently demote a writing dispatch to a wrapping one."""
    item = vocab.QueueItem(n=1, mode=vocab.MODE_ONBOARD, target="x", tier=vocab.TIER_TRITON)
    assert vocab.QueueItem(**item.as_dict()).tier == vocab.TIER_TRITON


def test_verify_dispatch_without_surface_is_rejected():
    """The excluded value is what smith is dispatched with; vagueness costs a run."""
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        queue=[{"n": 1, "mode": vocab.MODE_VERIFY, "target": "thop_attention"}],
    )
    with pytest.raises(vocab.VocabularyError, match="surface"):
        vocab.parse_vocabulary(doc)


def test_the_two_chapters_may_not_overlap():
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        queue=[
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "fused_moe"},
            {"n": 2, "mode": vocab.MODE_VERIFY, "target": "fused_moe", "surface": "h=72"},
        ],
    )
    with pytest.raises(vocab.VocabularyError, match="both chapters"):
        vocab.parse_vocabulary(doc)


def test_duplicate_dispatch_numbers_are_rejected():
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        queue=[
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "a"},
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "b"},
        ],
    )
    with pytest.raises(vocab.VocabularyError, match="duplicate"):
        vocab.parse_vocabulary(doc)


def test_dependency_on_a_missing_item_is_rejected():
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "a", "depends_on": [7]}],
    )
    with pytest.raises(vocab.VocabularyError, match="depends on 7"):
        vocab.parse_vocabulary(doc)


def test_ceiling_verdict_requires_its_evidence():
    doc = _doc(verdict=vocab.VERDICT_CEILING, ceiling={})
    with pytest.raises(vocab.VocabularyError, match="ceiling"):
        vocab.parse_vocabulary(doc)


def test_ceiling_with_evidence_parses():
    parsed = vocab.parse_vocabulary(
        _doc(
            verdict=vocab.VERDICT_CEILING,
            ceiling={
                "step": "exotic_norm",
                "searched": ["inventory", "catalog"],
                "why": "none",
                "exhausted": {
                    1: "no torch.ops.trtllm candidate",
                    2: "nothing under modules/** computes it",
                    3: "no flashinfer callable computes it (surface.md, FlashInfer API)",
                    4: "not expressible in torch ops without a scan",
                    5: "a Triton kernel needs a spec nobody has",
                },
            },
        )
    )
    assert parsed.is_ceiling
    assert parsed.ceiling is not None and parsed.ceiling["step"] == "exotic_norm"


def test_a_ceiling_that_ignores_the_writing_tiers_is_rejected():
    """The expensive mistake: reporting "nobody wrote it yet" as "it cannot exist"."""
    doc = _doc(
        verdict=vocab.VERDICT_CEILING,
        ceiling={
            "step": "exotic_norm",
            "searched": ["inventory"],
            "exhausted": {1: "no trtllm op", 2: "nothing in modules/**"},
        },
    )
    with pytest.raises(vocab.VocabularyError, match=r"tier\(s\) \[3, 4, 5\]"):
        vocab.parse_vocabulary(doc)


def test_a_ceiling_must_say_it_searched_flashinfer():
    """FlashInfer's API is invisible to the op inventory, so a ceiling must name it."""
    doc = _doc(
        verdict=vocab.VERDICT_CEILING,
        ceiling={
            "step": "exotic_norm",
            "searched": ["inventory"],
            "exhausted": {1: "no op", 2: "no callable", 4: "needs a scan", 5: "no spec"},
        },
    )
    with pytest.raises(vocab.VocabularyError, match=r"tier\(s\) \[3\]\. Tier 3 is FlashInfer"):
        vocab.parse_vocabulary(doc)


def test_ceiling_block_without_a_ceiling_verdict_is_rejected():
    doc = _doc(ceiling={"step": "x"})
    with pytest.raises(vocab.VocabularyError, match="must agree"):
        vocab.parse_vocabulary(doc)


def test_assemblable_with_a_queue_is_contradictory():
    doc = _doc(queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "a"}])
    with pytest.raises(vocab.VocabularyError, match="assemblable_after_additions"):
        vocab.parse_vocabulary(doc)


def test_after_additions_with_an_empty_queue_is_contradictory():
    doc = _doc(verdict=vocab.VERDICT_AFTER_ADDITIONS, queue=[])
    with pytest.raises(vocab.VocabularyError, match="names none"):
        vocab.parse_vocabulary(doc)


def test_every_problem_is_batched():
    doc = {
        "verdict": "maybe",
        "steps": [{"step": "", "state": "NOPE"}],
        "queue": [{"n": 1, "mode": "guess", "target": ""}],
    }
    with pytest.raises(vocab.VocabularyError) as excinfo:
        vocab.parse_vocabulary(doc)
    message = str(excinfo.value)
    assert "verdict" in message and "state" in message and "mode" in message


def test_queue_item_maps_mode_to_skill():
    onboard = vocab.QueueItem(n=1, mode=vocab.MODE_ONBOARD, target="a")
    verify = vocab.QueueItem(n=2, mode=vocab.MODE_VERIFY, target="b", surface="s")
    assert onboard.skill == "trtllm-modeling-v2-kernel-onboard"
    assert verify.skill == "trtllm-modeling-v2-kernel-verify"


def test_independent_items_land_in_one_batch():
    """Everything without a dependency runs at once — this is the wall-clock win."""
    items = [
        vocab.QueueItem(n=1, mode=vocab.MODE_ONBOARD, target="a"),
        vocab.QueueItem(n=2, mode=vocab.MODE_ONBOARD, target="b"),
        vocab.QueueItem(n=3, mode=vocab.MODE_ONBOARD, target="c"),
    ]
    batches = vocab.dependency_batches(items)
    assert len(batches) == 1
    assert [i.n for i in batches[0]] == [1, 2, 3]


def test_dependencies_layer_into_ordered_batches():
    items = [
        vocab.QueueItem(n=1, mode=vocab.MODE_ONBOARD, target="a"),
        vocab.QueueItem(n=2, mode=vocab.MODE_VERIFY, target="b", surface="s", depends_on=(1,)),
        vocab.QueueItem(n=3, mode=vocab.MODE_ONBOARD, target="c"),
        vocab.QueueItem(n=4, mode=vocab.MODE_VERIFY, target="d", surface="s", depends_on=(2, 3)),
    ]
    batches = vocab.dependency_batches(items)
    assert [[i.n for i in batch] for batch in batches] == [[1, 3], [2], [4]]


def test_a_dependency_cycle_is_named_not_broken_arbitrarily():
    items = [
        vocab.QueueItem(n=1, mode=vocab.MODE_ONBOARD, target="a", depends_on=(2,)),
        vocab.QueueItem(n=2, mode=vocab.MODE_ONBOARD, target="b", depends_on=(1,)),
    ]
    with pytest.raises(vocab.VocabularyError, match="cycle"):
        vocab.dependency_batches(items)


def test_empty_queue_yields_no_batches():
    assert vocab.dependency_batches([]) == []


def test_load_from_disk_round_trips(tmp_path):
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[_ok_step(state=vocab.STATE_CERT)],
        queue=[
            {
                "n": 1,
                "mode": vocab.MODE_VERIFY,
                "target": "flashinfer_rmsnorm",
                "surface": "head_dim=72",
            }
        ],
    )
    path = tmp_path / "vocabulary.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    parsed = vocab.load_vocabulary(path)
    assert parsed.queue[0].surface == "head_dim=72"
    reparsed = vocab.parse_vocabulary(yaml.safe_load(vocab.dump_vocabulary(parsed)))
    assert reparsed.queue[0].target == "flashinfer_rmsnorm"


def test_missing_file_names_the_producer(tmp_path):
    with pytest.raises(vocab.VocabularyError, match="plan phase"):
        vocab.load_vocabulary(tmp_path / "absent.yaml")


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "vocabulary.yaml"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(vocab.VocabularyError, match="empty"):
        vocab.load_vocabulary(path)


# ------------------------------------------------------------------ unit


_UNITS = ("core", "dspark", "multimodal", "kv_quant")


def _queue_doc(*rows):
    return _doc(verdict=vocab.VERDICT_AFTER_ADDITIONS, queue=list(rows))


def test_a_queue_row_may_name_the_unit_that_first_needs_it():
    parsed = vocab.parse_vocabulary(
        _queue_doc(
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe/drafter_moe", "unit": "dspark"},
            {"n": 2, "mode": vocab.MODE_ONBOARD, "target": "norm/core_only"},
        ),
        units=_UNITS,
    )
    assert [item.unit for item in parsed.queue] == ["dspark", vocab.UNIT_CORE]


def test_a_unit_the_run_does_not_build_is_rejected():
    """A misspelt tag could only be guessed at, and guessing late lets a core failure through."""
    doc = _queue_doc({"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "unit": "dspakr"})
    with pytest.raises(vocab.VocabularyError, match="not one of this run's assembly units"):
        vocab.parse_vocabulary(doc, units=_UNITS)


@pytest.mark.parametrize("bad", ["", "  ", 3, ["dspark"]])
def test_a_unit_must_be_a_name(bad):
    doc = _queue_doc({"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "unit": bad})
    with pytest.raises(vocab.VocabularyError, match=r"queue\[0\]\.unit"):
        vocab.parse_vocabulary(doc)


def test_without_the_run_units_only_the_shape_is_checked():
    """The validator cannot see task.yaml on its own; the orchestrator passes the units."""
    doc = _queue_doc({"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "unit": "anything"})
    assert vocab.parse_vocabulary(doc).queue[0].unit == "anything"


def test_the_unit_survives_a_round_trip_through_run_state():
    """A checkpointed row also carries the orchestrator's bookkeeping, which is not the item."""
    item = vocab.QueueItem(n=4, mode=vocab.MODE_ONBOARD, target="x", depends_on=(1,), unit="dspark")
    row = {
        **item.as_dict(),
        "status": "deferred",
        "deferred_to": "dspark",
        "deferred_why": "smith decided BLOCKED",
    }
    assert vocab.QueueItem.from_row(row) == item


def test_a_row_without_a_unit_reads_as_core():
    row = {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "depends_on": [], "status": "failed"}
    assert vocab.QueueItem.from_row(row).unit == vocab.UNIT_CORE


def _item(n, unit="core", depends_on=()):
    return vocab.QueueItem(
        n=n, mode=vocab.MODE_ONBOARD, target=f"t{n}", unit=unit, depends_on=depends_on
    )


def test_a_row_is_needed_by_its_own_unit():
    needed = vocab.needed_by([_item(1), _item(2, "dspark")], _UNITS)
    assert needed == {1: "core", 2: "dspark"}


def test_a_dependency_of_a_core_row_is_core_transitively():
    """Chain: core row 3 -> row 2 (kv_quant) -> row 1 (multimodal). Core needs all three."""
    queue = [
        _item(1, "multimodal"),
        _item(2, "kv_quant", depends_on=(1,)),
        _item(3, "core", depends_on=(2,)),
    ]
    assert vocab.needed_by(queue, _UNITS) == {1: "core", 2: "core", 3: "core"}


def test_a_dependency_is_needed_by_its_earliest_dependent():
    """Tagged kv_quant, but a dspark row waits on it: it is owed before dspark, not kv_quant."""
    queue = [_item(1, "kv_quant"), _item(2, "dspark", depends_on=(1,)), _item(3, "kv_quant")]
    assert vocab.needed_by(queue, _UNITS) == {1: "dspark", 2: "dspark", 3: "kv_quant"}


def test_a_dependent_is_not_pulled_earlier_by_what_it_waits_on():
    queue = [_item(1, "core"), _item(2, "kv_quant", depends_on=(1,))]
    assert vocab.needed_by(queue, _UNITS) == {1: "core", 2: "kv_quant"}


def test_a_unit_outside_the_run_is_needed_by_core():
    """The blocking direction: an unrecognised tag never lets a failure slip past assembly."""
    assert vocab.needed_by([_item(1, "eagle3")], _UNITS) == {1: "core"}
