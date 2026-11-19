# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every multimodal registration and route to its reviewed gaps."""

import ast
import re
import warnings

import pytest
from _torch.multimodal.conformance import known_gaps as gaps
from _torch.multimodal.conformance import models
from _torch.multimodal.conformance.models import (
    _B300_TARGETS,
    Key,
    ModelEntry,
    _returns,
    inventory,
    inventory_or_none,
)
from _torch.multimodal.conformance.tiers import FEATURE_TIERS, NOT_DERIVABLE, STRUCTURAL_ONLY

from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE
from tensorrt_llm._torch.models.multimodal import MultimodalFeature as Feature
from tensorrt_llm._torch.models.multimodal import supports_feature

pytestmark = pytest.mark.cpu_only

_SKILL = ".claude/skills/trtllm-model-onboard-multimodal"
_TRACKING_ID = re.compile(r"\b(?:TRTLLM-\d+|(?:https?://)?nvbugs?[\s:/#-]+\d+)\b", re.I)


def _label(key: Key) -> str:
    arch, target = key
    return f"{arch}/{target}" if target else arch


_FIELDS = {"base", "fixture", "text_only", *(feature.value for feature in Feature)}
# The snapshot sizes, as (key, feature) pairs for KNOWN_OPT_OUTS. A change that deletes
# entries lowers these; the snapshot never grows.
_SNAPSHOT_SIZES = {
    "SNAPSHOT_KEYS": 49,
    "KNOWN_OPT_OUTS": 301,
    "NOT_ON_BASE": 35,
    "MISSING_FIXTURE": 33,
    "MISSING_ADMISSION_CHECK": 12,
}


def _contract_problems(key: Key, entry: ModelEntry | None) -> list[str]:
    where = _label(key)
    known = gaps.KNOWN_OPT_OUTS.get(key, ())
    new = key not in gaps.SNAPSHOT_KEYS
    approved = {
        field: record for (owner, field), record in gaps.APPROVED_EXCEPTIONS.items() if owner == key
    }
    problems = []
    for field, (reason, ticket, approver) in approved.items():
        warnings.warn(f"{where}: approved {field}: {reason} ({ticket}, {approver})", stacklevel=2)
        if field not in _FIELDS:
            problems.append(f"{where}: P0 contract has an unknown exception field '{field}'")
        if not (reason.strip() and _TRACKING_ID.fullmatch(ticket) and approver.strip()):
            problems.append(
                f"{where}: P0 contract exception needs a reason, tracking ID and approver"
            )

    if entry is None:
        return [*problems, f"{where}: P0 inventory key is missing: stale entry, delete it"]
    snapshots = {
        "KNOWN_OPT_OUTS": gaps.KNOWN_OPT_OUTS,
        "NOT_ON_BASE": gaps.NOT_ON_BASE,
        "MISSING_FIXTURE": gaps.MISSING_FIXTURE,
        "MISSING_ADMISSION_CHECK": gaps.MISSING_ADMISSION_CHECK,
    }
    for name, snapshot in snapshots.items():
        if new and key in snapshot:
            problems.append(f"{where}: P0 contract: {name} cannot gain a key (see {_SKILL})")

    if not entry.media:
        # A text-only target serves no media, so only its admission check applies.
        for name in ("KNOWN_OPT_OUTS", "NOT_ON_BASE", "MISSING_FIXTURE"):
            if key in snapshots[name]:
                problems.append(f"{where}: text-only target in {name}: stale entry, delete it")
        listed = key in gaps.MISSING_ADMISSION_CHECK or "text_only" in approved
        if entry.refuses_media and listed:
            problems.append(f"{where}: P0 text_only admission is present: stale entry, delete it")
        elif not entry.refuses_media and not listed:
            problems.append(
                f"{where}: P0 text_only: run media through the multimodal runtime, or raise on "
                f"media and rows in validate_multimodal_request_data (see {_SKILL})"
            )
        return problems
    if key in gaps.MISSING_ADMISSION_CHECK:
        problems.append(f"{where}: takes media, not text-only: stale entry, delete it")

    if new and not entry.marked:
        problems.append(f"{where}: P0 contract requires supports_multimodal = True (see {_SKILL})")
    for field, present, snapshot in (
        ("base", entry.on_base, gaps.NOT_ON_BASE),
        ("fixture", bool(entry.fixture), gaps.MISSING_FIXTURE),
    ):
        listed = key in snapshot or field in approved
        if present and listed:
            problems.append(f"{where}: P0 {field} is present: stale entry, delete it")
        elif not present and not listed and not (new and field in STRUCTURAL_ONLY):
            problems.append(
                f"{where}: missing P0 {field} without APPROVED_EXCEPTIONS (see {_SKILL})"
            )

    opt_outs = getattr(entry.model_cls, "multimodal_opt_outs", {})
    limits = getattr(entry.model_cls, "multimodal_feature_limits", None)
    # Limits for an unknown deployment apply to every deployment, like opt-outs.
    declared = {**(limits(None, None) if callable(limits) else {}), **opt_outs}
    for feature, reason in declared.items():
        tier = FEATURE_TIERS[feature][0]
        if not (
            isinstance(reason, str)
            and _TRACKING_ID.search(reason)
            and re.search(r"\w", _TRACKING_ID.sub("", reason))
        ):
            problems.append(
                f"{where}: {tier} '{feature}' needs a reason and tracking ID (see {_SKILL})"
            )

    for feature in Feature:
        tier = FEATURE_TIERS[feature][0]
        label = f"{where}: {tier} feature '{feature}'"
        supported = entry.model_cls is not None and supports_feature(entry.model_cls, feature)
        if new and feature == Feature.ENCODER_CUDA_GRAPH:
            supported = supported and entry.graph_tower
        if supported:
            if feature in approved or (feature in known and feature not in NOT_DERIVABLE):
                problems.append(f"{label} is supported: stale entry, delete it")
        elif feature in known or feature in approved:
            continue
        elif new and feature in STRUCTURAL_ONLY and feature not in declared:
            continue
        elif tier == "P0":
            problems.append(
                f"{label} is opted out but has no APPROVED_EXCEPTIONS entry (see {_SKILL})"
            )
        elif feature not in declared:
            problems.append(
                f"{label} needs a multimodal_opt_outs reason and tracking ID (see {_SKILL})"
            )
    return problems


def _test_id(key: Key, entries: dict[Key, ModelEntry]) -> str:
    entry = entries.get(key)
    label = _label(key)
    return f"{label}-{entry.class_name}" if entry and entry.class_name != key[0] else label


def _keys() -> list:
    entries = inventory_or_none()
    if entries is None:
        # The test builds the inventory again and fails with its error.
        return [pytest.param(None, id="inventory")]
    keys = (
        entries.keys()
        | gaps.SNAPSHOT_KEYS
        | gaps.KNOWN_OPT_OUTS.keys()
        | gaps.NOT_ON_BASE.keys()
        | gaps.MISSING_FIXTURE
        | gaps.MISSING_ADMISSION_CHECK
        | {key for key, _ in gaps.APPROVED_EXCEPTIONS}
    )
    return [pytest.param(key, id=_test_id(key, entries)) for key in sorted(keys, key=_label)]


@pytest.mark.parametrize("key", _keys())
def test_multimodal_contract(key: Key) -> None:
    assert not (problems := _contract_problems(key, inventory().get(key))), problems


def test_snapshot_only_shrinks() -> None:
    sizes = {
        "SNAPSHOT_KEYS": len(gaps.SNAPSHOT_KEYS),
        "KNOWN_OPT_OUTS": sum(map(len, gaps.KNOWN_OPT_OUTS.values())),
        "NOT_ON_BASE": len(gaps.NOT_ON_BASE),
        "MISSING_FIXTURE": len(gaps.MISSING_FIXTURE),
        "MISSING_ADMISSION_CHECK": len(gaps.MISSING_ADMISSION_CHECK),
    }
    grown = {name: size for name, size in sizes.items() if size > _SNAPSHOT_SIZES[name]}
    assert not grown, f"known_gaps.py cannot grow {grown}; add new gaps to APPROVED_EXCEPTIONS"
    assert sizes == _SNAPSHOT_SIZES, f"Lower _SNAPSHOT_SIZES to the new sizes: {sizes}"


def test_registered_architectures_and_media_targets() -> None:
    entries = inventory()
    registered = {
        key for key in gaps.SNAPSHOT_KEYS if key[1] is None and key[0] in MODEL_ARCH_TO_MODULE
    }
    assert registered <= entries.keys()

    # These native targets predate the marker. Their media paths have B300 qualification jobs.
    for key, entry in entries.items():
        arch, target = key
        if key in gaps.SNAPSHOT_KEYS and target is not None and not (entry.marked or entry.on_base):
            assert entry.media == (target in _B300_TARGETS), (arch, target)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("    def helper(): return 0", "no target return"),
        (
            "    if flag:\n        from .one.modeling import A as T\n"
            "    else:\n        from .two.modeling import B as T\n    return T",
            "ambiguous target return",
        ),
        (
            "    from .one.modeling import A, B\n    if flag: return A\n    return B",
            "different classes returned from one",
        ),
    ],
)
def test_route_returns_account_for_every_target(body: str, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        _returns(f"@routes('Example')\ndef route():\n{body}", "example")


class MultimodalForConditionalGeneration:
    supports_multimodal = True
    supports_mm_encoder_item_scheduling = True
    supports_encoder_cache = True
    support_mm_disagg = True


class _Model(MultimodalForConditionalGeneration):
    pass


_KEY = ("ExampleVL", None)


def _entry(**fields) -> ModelEntry:
    return ModelEntry(**{"model_cls": _Model, "fixture": "tiny", "graph_tower": True, **fields})


def test_new_model_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    # New models can leave structural-only features off.
    assert _contract_problems(_KEY, _entry()) == []
    monkeypatch.setattr(_Model, "supports_mm_encoder_item_scheduling", False)
    monkeypatch.setattr(_Model, "support_mm_disagg", False)
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == 2
    assert any("P0 feature 'item_scheduling'" in p and "APPROVED_EXCEPTIONS" in p for p in problems)
    assert any("P1 feature 'disagg_e_to_p'" in p and "multimodal_opt_outs" in p for p in problems)
    assert all(_SKILL in p for p in problems)

    feature = Feature.DISAGG_E_TO_P
    monkeypatch.setattr(
        _Model, "multimodal_opt_outs", {feature: "No layout (nvbug 12345)."}, raising=False
    )
    monkeypatch.setitem(
        gaps.APPROVED_EXCEPTIONS,
        (_KEY, Feature.ITEM_SCHEDULING),
        ("Needs item costs.", "TRTLLM-1234", "model owner"),
    )
    with pytest.warns(UserWarning, match=r"ExampleVL: approved item_scheduling: .*TRTLLM-1234"):
        assert _contract_problems(_KEY, _entry()) == []


@pytest.mark.parametrize("hook", ["multimodal_opt_outs", "multimodal_feature_limits"])
@pytest.mark.parametrize("feature", sorted(set(STRUCTURAL_ONLY) & set(Feature)))
def test_declared_p0_skip_needs_an_exception_before_its_runtime_lands(
    hook: str, feature: Feature, monkeypatch: pytest.MonkeyPatch
) -> None:
    skip = {feature: "Not wired yet (TRTLLM-1234)."}
    value = skip if hook == "multimodal_opt_outs" else classmethod(lambda cls, config, args: skip)
    monkeypatch.setattr(_Model, hook, value, raising=False)
    problems = _contract_problems(("FooVL", None), _entry())
    if FEATURE_TIERS[feature][0] == "P1":
        assert problems == []
        return
    assert problems == [
        f"FooVL: P0 feature '{feature}' is opted out but has no APPROVED_EXCEPTIONS entry "
        f"(see {_SKILL})"
    ]


def test_new_model_needs_a_graph_tower_or_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    feature = Feature.ENCODER_CUDA_GRAPH
    problems = _contract_problems(_KEY, _entry(graph_tower=False))
    assert problems == [
        f"ExampleVL: P1 feature '{feature}' needs a multimodal_opt_outs reason and tracking ID "
        f"(see {_SKILL})"
    ]
    monkeypatch.setattr(
        _Model, "multimodal_opt_outs", {feature: "No block adapter (TRTLLM-1234)."}, raising=False
    )
    assert _contract_problems(_KEY, _entry(graph_tower=False)) == []


_TEXT_ONLY_TARGETS = """
class Refuses:
    def validate_multimodal_request_data(self, mm_data):
        if mm_data:
            raise ValueError("text only")


class Silent:
    def validate_multimodal_request_data(self, mm_data):
        pass
"""


@pytest.mark.parametrize(("name", "refuses"), [("Refuses", True), ("Silent", False)])
def test_new_text_only_target_needs_only_admission(
    name: str, refuses: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    classes = {node.name: node for node in ast.parse(_TEXT_ONLY_TARGETS).body}
    monkeypatch.setattr(models, "_source", lambda module: (classes, {}))
    model_cls, media, admission = models._target("example.text_only.modeling", name)
    assert (model_cls, media, admission) == (None, False, refuses)
    key = ("ExampleVL", "example/text_only")
    entry = ModelEntry(model_cls, media=media, class_name=name, refuses_media=admission)
    problems = _contract_problems(key, entry)
    assert len(problems) == (0 if refuses else 1)
    assert all("P0 text_only" in p and _SKILL in p for p in problems)


def test_text_only_snapshot_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    key = ("ExampleVL", "example/text_only")
    monkeypatch.setattr(gaps, "SNAPSHOT_KEYS", gaps.SNAPSHOT_KEYS | {key})
    monkeypatch.setattr(gaps, "MISSING_ADMISSION_CHECK", gaps.MISSING_ADMISSION_CHECK | {key})
    monkeypatch.setitem(gaps.NOT_ON_BASE, key, "Text only.")
    entry = ModelEntry(None, media=False, refuses_media=True)
    assert _contract_problems(key, entry) == [
        "ExampleVL/example/text_only: text-only target in NOT_ON_BASE: stale entry, delete it",
        "ExampleVL/example/text_only: P0 text_only admission is present: stale entry, delete it",
    ]
    stale = "ExampleVL/example/text_only: takes media, not text-only: stale entry, delete it"
    assert stale in _contract_problems(key, _entry())


@pytest.mark.parametrize("reason", ["", "No layout.", "TRTLLM-1234"])
def test_declared_opt_out_needs_reason_and_id(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _Model, "multimodal_opt_outs", {Feature.DISAGG_E_TO_P: reason}, raising=False
    )
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == 1
    assert "P1 'disagg_e_to_p' needs a reason and tracking ID" in problems[0]


def test_new_key_requires_marker_base_and_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    problems = _contract_problems(_KEY, ModelEntry(model_cls=object))
    assert any("P0 contract requires supports_multimodal" in p and _SKILL in p for p in problems)
    assert not any("missing P0 base" in p or "missing P0 fixture" in p for p in problems)
    # Once the base and fixtures are available, new keys need them too.
    monkeypatch.delitem(STRUCTURAL_ONLY, "base")
    monkeypatch.delitem(STRUCTURAL_ONLY, "fixture")
    problems = _contract_problems(_KEY, ModelEntry(model_cls=object))
    for rule in ("missing P0 base", "missing P0 fixture"):
        assert any(rule in p and _SKILL in p for p in problems)


def test_snapshot_rejects_new_and_stale_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gaps.KNOWN_OPT_OUTS, _KEY, frozenset({Feature.ITEM_SCHEDULING}))
    monkeypatch.setitem(gaps.NOT_ON_BASE, _KEY, "Uses its own wrapper.")
    monkeypatch.setattr(gaps, "MISSING_FIXTURE", frozenset({_KEY}))
    problems = _contract_problems(_KEY, _entry())
    for snapshot in ("KNOWN_OPT_OUTS", "NOT_ON_BASE", "MISSING_FIXTURE"):
        assert any(f"P0 contract: {snapshot} cannot gain a key" in p for p in problems)
    for rule in (
        "P0 feature 'item_scheduling' is supported",
        "P0 base is present",
        "P0 fixture is present",
    ):
        assert any(f"{rule}: stale entry, delete it" in p for p in problems)
    assert _contract_problems(_KEY, None) == [
        "ExampleVL: P0 inventory key is missing: stale entry, delete it"
    ]


def test_snapshot_size_is_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    key = ("Qwen3VLForConditionalGeneration", None)
    grown = dict(gaps.KNOWN_OPT_OUTS)
    grown[key] |= {Feature.ENCODER_CACHE}
    monkeypatch.setattr(gaps, "KNOWN_OPT_OUTS", grown)
    with pytest.raises(AssertionError, match="cannot grow"):
        test_snapshot_only_shrinks()
    del grown[key]
    with pytest.raises(AssertionError, match="Lower _SNAPSHOT_SIZES"):
        test_snapshot_only_shrinks()


def test_exception_requires_a_known_field_and_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gaps.APPROVED_EXCEPTIONS, (_KEY, "typo"), ("", "", ""))
    with pytest.warns(UserWarning, match="ExampleVL: approved typo"):
        problems = _contract_problems(_KEY, _entry())
    assert any("P0 contract has an unknown exception field 'typo'" in p for p in problems)
    assert any(
        "P0 contract exception needs a reason, tracking ID and approver" in p for p in problems
    )
