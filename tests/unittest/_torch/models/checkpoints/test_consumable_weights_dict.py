# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import torch

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper


class _Mapper(BaseWeightMapper):
    """Concrete stand-in; the methods under test do not use the abstract half."""

    def map_weights(self) -> None:
        pass

    def apply_callbacks(self, module, module_name, module_names_breakdown, weights) -> list[dict]:
        raise NotImplementedError


def _scan(weights: dict, prefix: str) -> dict:
    """The pre-index implementation, kept as the reference for filter_prefix."""
    return {k[len(prefix) + 1 :]: v for k, v in weights.items() if k.startswith(prefix)}


def test_filter_prefix_matches_the_scan_it_replaces():
    plain = {
        "model.layers.1.weight": 1,
        "model.layers.1.self_attn.bias": 2,
        "model.layers.10.weight": 3,
        "model.norm.weight": 4,
    }
    weights = ConsumableWeightsDict(dict(plain))

    for prefix in ("model", "model.layers.1", "model.layers.10", "model.norm", "absent"):
        assert weights.filter_prefix(prefix) == _scan(plain, prefix), prefix


def test_filter_prefix_spans_keys_that_sort_past_the_separator():
    """The subtree bound must not stop at a key ordered above ``prefix.``."""
    weights = ConsumableWeightsDict({"a.weight": 1, "a/other": 2, "azz": 3, "b.weight": 4})

    assert weights.filter_prefix("a") == _scan(
        {"a.weight": 1, "a/other": 2, "azz": 3, "b.weight": 4}, "a"
    )


def test_index_tracks_added_and_consumed_keys():
    weights = ConsumableWeightsDict({"a.weight": 1})

    assert weights.filter_prefix("a") == {"weight": 1}
    weights["a.bias"] = 2
    weights.update({"a.scale": 3, "b.weight": 4})
    assert weights.filter_prefix("a") == {"weight": 1, "bias": 2, "scale": 3}
    assert weights.mark_consumed("a") == 3
    assert weights.filter_prefix("a") == {}
    assert weights.filter_prefix("b") == {"weight": 4}


def test_index_stays_correct_across_deletion_and_clear():
    """Deletions leave the index stale on purpose; it must stay a superset."""
    weights = ConsumableWeightsDict({"a.weight": 1, "a.bias": 2, "b.weight": 3})

    assert weights.filter_prefix("a") == {"weight": 1, "bias": 2}
    del weights["a.weight"]
    assert weights.mark_consumed_keys(["a.bias", "missing"]) == 1
    assert weights.filter_prefix("a") == {}
    weights.clear()
    assert weights.mark_consumed("b") == 0


def test_take_ownership_empties_a_consumable_source():
    """The derived mapping aliases the source, so the source must let go.

    Without this the loader holds a second reference to every tensor and
    mark_consumed() on the derived mapping frees nothing -- the whole
    checkpoint stays resident for the length of the load.
    """
    source = ConsumableWeightsDict({"a.weight": 1, "b.weight": 2})

    result = ConsumableWeightsDict.take_ownership(source, {"renamed.a.weight": 1})

    assert isinstance(result, ConsumableWeightsDict)
    assert len(source) == 0
    assert result.mark_consumed("renamed.a") == 1


def test_take_ownership_leaves_a_plain_dict_source_alone():
    """A plain dict was never releasing incrementally; do not surprise its owner."""
    source = {"a.weight": 1}
    derived = {"renamed.a.weight": 1}

    result = ConsumableWeightsDict.take_ownership(source, derived)

    assert result is derived
    assert source == {"a.weight": 1}


def test_rename_by_params_map_transfers_ownership():
    tensor = object()
    weights = ConsumableWeightsDict(
        {"model.language_model.layer.weight": tensor, "lm_head.weight": object()}
    )

    renamed = _Mapper().rename_by_params_map(
        {r"^model\.language_model\.(.*)$": r"model.\1"}, weights
    )

    assert isinstance(renamed, ConsumableWeightsDict)
    assert len(weights) == 0
    assert renamed["model.layer.weight"] is tensor
    assert "lm_head.weight" in renamed


def test_rename_by_params_map_leaves_a_plain_dict_alone():
    weights = {"model.language_model.layer.weight": object()}

    renamed = _Mapper().rename_by_params_map(
        {r"^model\.language_model\.(.*)$": r"model.\1"}, weights
    )

    assert not isinstance(renamed, ConsumableWeightsDict)
    assert len(weights) == 1


def test_consumption_observer_survives_identity_preserving_rename():
    reports = []
    tensor = torch.zeros(8, dtype=torch.uint8)
    source = ConsumableWeightsDict({"old.weight": tensor})
    source.set_consumption_observer(lambda *report: reports.append(report))

    renamed = ConsumableWeightsDict.take_ownership(source, {"new.weight": tensor})
    assert renamed.mark_consumed("new") == 1

    assert reports == [(tensor.nbytes, 1, 1)]


def test_consumption_observer_does_not_credit_transformed_replacement():
    reports = []
    raw = torch.zeros(8, dtype=torch.uint8)
    transformed = raw.to(torch.float32)
    source = ConsumableWeightsDict({"old.weight": raw})
    source.set_consumption_observer(lambda *report: reports.append(report))

    derived = ConsumableWeightsDict.take_ownership(source, {"new.weight": transformed})
    assert derived.mark_consumed("new") == 1

    assert reports == [(0, 1, 0)]


def test_consumption_observer_handles_concurrent_consumers():
    reports = []
    report_lock = threading.Lock()

    def observe(*report):
        with report_lock:
            reports.append(report)

    weights = ConsumableWeightsDict(
        {
            "a.weight": torch.zeros(3, dtype=torch.uint8),
            "b.weight": torch.zeros(5, dtype=torch.uint8),
        },
        consumption_observer=observe,
    )
    threads = [
        threading.Thread(target=weights.mark_consumed, args=(prefix,)) for prefix in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(report[0] for report in reports) == 8
    assert sum(report[1] for report in reports) == 2
    assert sum(report[2] for report in reports) == 2


def test_deletion_and_clear_do_not_imply_consumption():
    reports = []
    weights = ConsumableWeightsDict(
        {
            "a.weight": torch.zeros(3, dtype=torch.uint8),
            "b.weight": torch.zeros(5, dtype=torch.uint8),
        },
        consumption_observer=lambda *report: reports.append(report),
    )

    del weights["a.weight"]
    weights.clear()

    assert reports == []
