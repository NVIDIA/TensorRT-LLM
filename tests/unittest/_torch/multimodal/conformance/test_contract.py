# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every multimodal registration and route to its reviewed gaps."""

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


_FOLLOW_UP_FLAGS = {
    "supports_vision_gpu_normalize": "supports_vision_gpu_normalize",
    "encoder_attention_backend": "supported_encoder_attention_backends",
}
_FIELDS = {"base", "fixture"} | {f for f, (tier, _) in FEATURE_TIERS.items() if tier == "P0"}
# The snapshot sizes, as (key, feature) pairs for KNOWN_OPT_OUTS. A change that deletes
# entries lowers these; the snapshot never grows.
_SNAPSHOT_SIZES = {
    "SNAPSHOT_KEYS": 56,
    "KNOWN_OPT_OUTS": 225,
    "NOT_ON_BASE": 36,
    "MISSING_FIXTURE": 34,
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
    }
    for name, snapshot in snapshots.items():
        if new and key in snapshot:
            problems.append(f"{where}: P0 contract: {name} cannot gain a key (see {_SKILL})")

    if not entry.media:
        # The executor owns V2 admission. Its behavior is exercised in test_admission.py.
        for name in ("KNOWN_OPT_OUTS", "NOT_ON_BASE", "MISSING_FIXTURE"):
            if key in snapshots[name]:
                problems.append(f"{where}: text-only target in {name}: stale entry, delete it")
        return problems

    if new and not entry.marked and "contract" not in approved:
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
        if feature not in set(Feature) and feature not in _FOLLOW_UP_FLAGS:
            problems.append(f"{where}: unknown multimodal feature '{feature}' (see {_SKILL})")
            continue
        tier = FEATURE_TIERS[feature][0]
        if tier == "P2":
            continue
        if not (
            isinstance(reason, str)
            and _TRACKING_ID.search(reason)
            and re.search(r"\w", _TRACKING_ID.sub("", reason))
        ):
            problems.append(
                f"{where}: {tier} '{feature}' needs a reason and tracking ID (see {_SKILL})"
            )

    eligible = {
        feature: entry.model_cls is not None and supports_feature(entry.model_cls, feature)
        for feature in Feature
        if FEATURE_TIERS[feature][0] != "P2"
    }
    if new:
        # A tower can declare an empty set of selectable backends for fixed attention.
        for feature, flag in _FOLLOW_UP_FLAGS.items():
            eligible[feature] = (
                bool(getattr(entry.model_cls, flag, False)) and feature not in opt_outs
            )
    for feature, supported in eligible.items():
        tier = FEATURE_TIERS[feature][0]
        label = f"{where}: {tier} feature '{feature}'"
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
        elif feature not in opt_outs:
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
        | {key for key, _ in gaps.APPROVED_EXCEPTIONS}
    )
    return [pytest.param(key, id=_test_id(key, entries)) for key in sorted(keys, key=_label)]


@pytest.mark.parametrize("key", _keys())
def test_multimodal_contract(key: Key) -> None:
    assert not (problems := _contract_problems(key, inventory().get(key))), problems


def test_snapshot_only_shrinks() -> None:
    for key, missing in gaps.KNOWN_OPT_OUTS.items():
        assert all(FEATURE_TIERS[feature][0] != "P2" for feature in missing), (
            f"{_label(key)}: remove optional features from KNOWN_OPT_OUTS"
        )
    sizes = {
        "SNAPSHOT_KEYS": len(gaps.SNAPSHOT_KEYS),
        "KNOWN_OPT_OUTS": sum(map(len, gaps.KNOWN_OPT_OUTS.values())),
        "NOT_ON_BASE": len(gaps.NOT_ON_BASE),
        "MISSING_FIXTURE": len(gaps.MISSING_FIXTURE),
    }
    grown = {name: size for name, size in sizes.items() if size > _SNAPSHOT_SIZES[name]}
    assert not grown, f"known_gaps.py cannot grow {grown}. Follow the tier rules in {_SKILL}."
    assert sizes == _SNAPSHOT_SIZES, f"Lower _SNAPSHOT_SIZES to the new sizes: {sizes}"


def test_registered_architectures_and_media_targets() -> None:
    entries = inventory()
    registered = {
        key for key in gaps.SNAPSHOT_KEYS if key[1] is None and key[0] in MODEL_ARCH_TO_MODULE
    }
    assert registered <= entries.keys()

    # The snapshot records pre-marker media targets as wrappers awaiting the shared base.
    for key, entry in entries.items():
        _, target = key
        if key in gaps.SNAPSHOT_KEYS and target is not None and not (entry.marked or entry.on_base):
            assert entry.media == (key in gaps.NOT_ON_BASE), key
    assert {target for (_, target), entry in entries.items() if entry.media and entry.fixture} >= (
        _B300_TARGETS
    )


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


class _Model:
    supports_multimodal = True
    supports_mm_encoder_item_scheduling = True
    supports_encoder_cache = True
    supports_vision_gpu_normalize = True
    supported_encoder_attention_backends = {"vision": frozenset()}


_KEY = ("ExampleVL", None)


def _entry(**fields) -> ModelEntry:
    return ModelEntry(**{"model_cls": _Model, "fixture": "tiny", **fields})


def test_new_model_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    # This tower supports the required features and omits the structural and optional hooks.
    assert _contract_problems(_KEY, _entry()) == []
    monkeypatch.setattr(_Model, "supports_mm_encoder_item_scheduling", False)
    monkeypatch.setattr(_Model, "supports_encoder_cache", False)
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == 2
    assert any("P0 feature 'encoder_cache'" in p and "APPROVED_EXCEPTIONS" in p for p in problems)
    assert any("P1 feature 'item_scheduling'" in p and "multimodal_opt_outs" in p for p in problems)
    assert all(_SKILL in p for p in problems)

    monkeypatch.setattr(
        _Model,
        "multimodal_opt_outs",
        {Feature.ITEM_SCHEDULING: "No item costs (nvbug 12345)."},
        raising=False,
    )
    monkeypatch.setitem(
        gaps.APPROVED_EXCEPTIONS,
        (_KEY, Feature.ENCODER_CACHE),
        ("No stable cache keys.", "TRTLLM-1234", "model owner"),
    )
    with pytest.warns(UserWarning, match=r"ExampleVL: approved encoder_cache: .*TRTLLM-1234"):
        assert _contract_problems(_KEY, _entry()) == []


@pytest.mark.parametrize("hook", ["multimodal_opt_outs", "multimodal_feature_limits"])
@pytest.mark.parametrize("feature", [Feature.ITEM_SCHEDULING, Feature.ENCODER_DP])
def test_declared_skip_obeys_feature_tier(
    hook: str, feature: Feature, monkeypatch: pytest.MonkeyPatch
) -> None:
    skip = {feature: "Not wired yet (TRTLLM-1234)."}
    value = skip if hook == "multimodal_opt_outs" else classmethod(lambda cls, config, args: skip)
    monkeypatch.setattr(_Model, hook, value, raising=False)
    problems = _contract_problems(_KEY, _entry())
    if feature == Feature.ENCODER_DP:
        assert len(problems) == 1 and "APPROVED_EXCEPTIONS" in problems[0]
    elif hook == "multimodal_feature_limits":
        assert len(problems) == 1 and "multimodal_opt_outs" in problems[0]
    else:
        assert problems == []


def test_new_model_declares_follow_ups_or_tracks_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Model, "supports_vision_gpu_normalize", False)
    monkeypatch.setattr(_Model, "supported_encoder_attention_backends", {})
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == len(_FOLLOW_UP_FLAGS)
    for feature in _FOLLOW_UP_FLAGS:
        assert any(f"P1 feature '{feature}'" in p and "multimodal_opt_outs" in p for p in problems)

    opt_outs = dict.fromkeys(_FOLLOW_UP_FLAGS, "Needs tower support (TRTLLM-1234).")
    monkeypatch.setattr(_Model, "multimodal_opt_outs", opt_outs, raising=False)
    assert _contract_problems(_KEY, _entry()) == []
    opt_outs["typo"] = "Needs tower support (TRTLLM-1234)."
    assert any("unknown multimodal feature 'typo'" in p for p in _contract_problems(_KEY, _entry()))


def test_structural_features_require_approval_once_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = (
        Feature.LLM_BREAKABLE_CUDA_GRAPH,
        Feature.LLM_PIECEWISE_CUDA_GRAPH,
        Feature.ENCODER_DP,
    )
    assert _contract_problems(_KEY, _entry()) == []
    for feature in features:
        monkeypatch.delitem(STRUCTURAL_ONLY, feature)
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == len(features)
    for feature in features:
        assert any(f"P0 feature '{feature}'" in p and "APPROVED_EXCEPTIONS" in p for p in problems)


def test_text_only_target_has_no_media_feature_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ("ExampleVL", "example/text_only")
    entry = ModelEntry(None, media=False)
    assert _contract_problems(key, entry) == []
    monkeypatch.setattr(gaps, "SNAPSHOT_KEYS", gaps.SNAPSHOT_KEYS | {key})
    monkeypatch.setitem(gaps.NOT_ON_BASE, key, "Text only.")
    assert _contract_problems(key, entry) == [
        "ExampleVL/example/text_only: text-only target in NOT_ON_BASE: stale entry, delete it",
    ]


@pytest.mark.parametrize("reason", ["", "No layout.", "TRTLLM-1234"])
def test_declared_opt_out_needs_reason_and_id(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _Model, "multimodal_opt_outs", {Feature.DISAGG_P_TO_D: reason}, raising=False
    )
    problems = _contract_problems(_KEY, _entry())
    assert len(problems) == 1
    assert "P1 'disagg_p_to_d' needs a reason and tracking ID" in problems[0]


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
    monkeypatch.setattr(models, "_V2_BASE", _Model)
    monkeypatch.setitem(gaps.KNOWN_OPT_OUTS, _KEY, frozenset({Feature.ITEM_SCHEDULING}))
    monkeypatch.setitem(gaps.NOT_ON_BASE, _KEY, "Uses its own wrapper.")
    monkeypatch.setattr(gaps, "MISSING_FIXTURE", frozenset({_KEY}))
    problems = _contract_problems(_KEY, _entry())
    for snapshot in ("KNOWN_OPT_OUTS", "NOT_ON_BASE", "MISSING_FIXTURE"):
        assert any(f"P0 contract: {snapshot} cannot gain a key" in p for p in problems)
    for rule in (
        "P1 feature 'item_scheduling' is supported",
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
