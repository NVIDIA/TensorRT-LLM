# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend contract's surface, pinned: its exports, the members and signatures of its
protocols, ``Capabilities`` and ``Outcome``, and a base package that imports nothing of the KV cache
lender. Also what native's per-request types in ``native/extent.py`` promise: ``reports_pending``
says only whether a report is owed, success cannot leave one owed, ``is_last`` and each group's
kind must be stated, and native's ``Attempt`` needs only ``poll``."""

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import os
import types
import typing

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.base as package
import tensorrt_llm._torch.disaggregation.base.backend as contract_module
import tensorrt_llm._torch.disaggregation.native.extent as extent_module
from tensorrt_llm._torch.disaggregation.base import (
    CacheKind,
    LayerGroupRows,
    Outcome,
    SlotRegion,
    TokenRange,
    View,
)
from tensorrt_llm._torch.disaggregation.native.extent import (
    CacheExtent,
    Cancelled,
    Chunk,
    Delivered,
    Failed,
)

pytestmark = pytest.mark.cpu_only


def _reports_may_be_pending(outcome) -> bool:
    """Reports may still be pending; this says nothing about safe memory release."""
    return outcome is None or outcome.reports_pending


def _chunk(**overrides) -> Chunk:
    fields = dict(
        pool_slot_ids_per_layer_group=[np.array([1, 2], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=8),
        is_last=True,
    )
    fields.update(overrides)
    return Chunk(**fields)


# Minimal structural implementations. Groups and views compare by identity.
@dataclasses.dataclass(frozen=True)
class _SlotRegion:
    layout_label: str
    address: int
    slot_bytes: int
    slot_count: int

    @property
    def nbytes(self) -> int:
        return self.slot_bytes * self.slot_count


@dataclasses.dataclass(frozen=True, eq=False)
class _LayerGroupRows:
    layer_group_index: int
    row_count: int
    object_keys: typing.Optional[np.ndarray] = None
    addresses: typing.Optional[np.ndarray] = None
    region_index: typing.Optional[int] = None

    def __len__(self) -> int:
        return self.row_count


@dataclasses.dataclass(frozen=True, eq=False)
class _View:
    groups: typing.Tuple[_LayerGroupRows, ...]

    def row_masks(self, fill_value: bool = False) -> typing.Tuple[np.ndarray, ...]:
        return tuple(np.full(len(group), fill_value, dtype=bool) for group in self.groups)


def _keyed_view(*row_counts: int) -> _View:
    """One group per row count, in order: ``_keyed_view(2, 3)`` has two rows in layer group 0
    and three in layer group 1. Rows occupy consecutive slots in region 0; their keys encode
    their slot indices."""
    region = _SlotRegion("pool", address=4096, slot_bytes=64, slot_count=sum(row_counts))
    groups, first_slot_index = [], 0
    for layer_group_index, row_count in enumerate(row_counts):
        slot_indices = np.arange(first_slot_index, first_slot_index + row_count, dtype=np.int64)
        object_keys = slot_indices.astype("<u4").view(np.uint8).reshape(row_count, 4)
        addresses = region.address + region.slot_bytes * slot_indices
        groups.append(
            _LayerGroupRows(layer_group_index, row_count, object_keys, addresses, region_index=0)
        )
        first_slot_index += row_count
    return _View(tuple(groups))


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def test_token_range_may_be_empty_but_not_reversed():
    assert TokenRange(start=8, end=8).start == 8
    with pytest.raises(ValueError):
        TokenRange(start=10, end=3)
    with pytest.raises(ValueError):
        TokenRange(start=-1, end=5)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_names_shared_with_the_contract_are_other_objects_in_their_own_module():
    """The contract has its own ``Attempt`` and ``Outcome``; native's share the names but are
    different objects, each defined in its own module. ``CacheExtent`` is native's alone."""
    assert set(extent_module.__all__) == {
        "Attempt",
        "CacheExtent",
        "Cancelled",
        "Chunk",
        "Delivered",
        "Failed",
        "Outcome",
    }
    for name in ("Outcome", "Attempt"):
        native, contract = getattr(extent_module, name), getattr(contract_module, name)
        assert native is not contract
        assert contract.__module__ == contract_module.__name__
        # Native's ``Outcome`` is the union of its endings, each defined beside it.
        for defined in typing.get_args(native) or (native,):
            assert defined.__module__ == extent_module.__name__
    for name in ("CacheExtent", "Chunk", "Delivered", "Failed", "Cancelled"):
        assert not hasattr(contract_module, name)
    for name in ("Fetches", "Publishes"):
        assert not hasattr(extent_module, name)
        assert not hasattr(contract_module, name)


# ---------------------------------------------------------------------------
# The contract's surface
# ---------------------------------------------------------------------------

EXPORTS = {
    "Attempt",
    "Backend",
    "CacheKind",
    "Capabilities",
    "Hint",
    "InvalidSubmission",
    "LayerGroupRows",
    "Outcome",
    "SlotRegion",
    "SubmissionRejected",
    "TokenRange",
    "View",
    "WakeFn",
}

# Each protocol's members: a method's signature, or a property's type after "->".
PROTOCOLS = {
    "Backend": {
        "capabilities": "-> Capabilities",
        "fetch": "(self, view: 'View', source: 'Hint') -> 'Attempt'",
        "publish": "(self, view: 'View', destination: 'Hint') -> 'Attempt'",
        "shutdown": "(self, timeout: 'float') -> 'FrozenSet'",
    },
    "Attempt": {
        "poll": "(self) -> 'Optional[Outcome]'",
        "quiesced": "(self) -> 'bool'",
        "cancel": "(self) -> 'None'",
    },
    "View": {
        "groups": "-> Sequence[LayerGroupRows]",
        "row_masks": "(self, fill_value: 'bool' = False) -> 'Tuple[np.ndarray, ...]'",
    },
    "LayerGroupRows": {
        "layer_group_index": "-> int",
        "object_keys": "-> Optional[np.ndarray]",
        "addresses": "-> Optional[np.ndarray]",
        "region_index": "-> Optional[int]",
        "__len__": "(self) -> 'int'",
    },
    "SlotRegion": {
        "layout_label": "-> str",
        "address": "-> int",
        "nbytes": "-> int",
        "slot_bytes": "-> int",
        "slot_count": "-> int",
    },
}


def _protocol_member_signatures(protocol) -> dict:
    members = {}
    for name, value in vars(protocol).items():
        if name.startswith("_") and name != "__len__":
            continue
        if isinstance(value, property):
            members[name] = f"-> {value.fget.__annotations__['return']}"
        else:
            members[name] = str(inspect.signature(value))
    return members


def test_the_contract_exports_thirteen_names():
    assert len(EXPORTS) == 13
    assert set(contract_module.__all__) == EXPORTS
    assert set(package.__all__) == EXPORTS
    for name in EXPORTS:
        assert getattr(package, name) is getattr(contract_module, name)
    assert contract_module.Hint is typing.Any
    assert contract_module.WakeFn == typing.Callable[[], None]


@pytest.mark.parametrize("name", sorted(PROTOCOLS))
def test_each_protocol_has_expected_members_and_signatures(name):
    protocol = getattr(contract_module, name)
    assert _protocol_member_signatures(protocol) == PROTOCOLS[name]
    assert not isinstance(object(), protocol)


def test_minimal_structural_implementations_pass_runtime_protocol_checks():
    region = _SlotRegion("pool", address=4096, slot_bytes=64, slot_count=8)
    view = _View((_keyed_view(2).groups[0], _LayerGroupRows(1, 3)))
    assert isinstance(region, SlotRegion) and region.nbytes == 512
    assert isinstance(view, View)
    assert all(isinstance(group, LayerGroupRows) for group in view.groups)
    masks = view.row_masks(True)
    assert [mask.tolist() for mask in masks] == [[True] * 2, [True] * 3]
    assert all(mask.flags.writeable for mask in masks)


@pytest.mark.parametrize("name", ["SlotRegion", "LayerGroupRows", "View"])
def test_runtime_protocol_checks_require_every_member(name):
    protocol = getattr(contract_module, name)
    whole = {
        "SlotRegion": _SlotRegion("pool", 4096, 64, 8),
        "LayerGroupRows": _keyed_view(2).groups[0],
        "View": _keyed_view(2, 3),
    }
    complete = {
        member: getattr(whole[name], member) for member in _protocol_member_signatures(protocol)
    }
    assert isinstance(types.SimpleNamespace(**complete), protocol)
    for dropped in complete:
        rest = {member: value for member, value in complete.items() if member != dropped}
        assert not isinstance(types.SimpleNamespace(**rest), protocol), dropped


# The constructors and methods of the concrete types a backend builds or calls.
SIGNATURES = {
    "Capabilities": "(fetch: 'bool', publish: 'bool', cancel: 'bool' = False) -> None",
    "Outcome": "(delivered_masks: 'Tuple[np.ndarray, ...]', fault: 'Optional[str]' = None) -> None",
    "Outcome.for_view": (
        "(view: 'View', delivered_masks: 'Optional[Sequence[np.ndarray]]' = None,"
        " fault: 'Optional[str]' = None) -> 'Outcome'"
    ),
    "Outcome.check_mask_shapes": "(self, view: 'View') -> 'None'",
}


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_each_concrete_type_has_its_signature(name):
    owner, _, member = name.partition(".")
    target = getattr(contract_module, owner)
    target = getattr(target, member) if member else target
    assert str(inspect.signature(target)) == SIGNATURES[name]


def test_capabilities_fields_defaults_and_validation():
    capabilities = contract_module.Capabilities
    fields = {f.name: f.default for f in dataclasses.fields(capabilities)}
    assert list(fields) == ["fetch", "publish", "cancel"]
    assert fields["fetch"] is fields["publish"] is dataclasses.MISSING
    assert fields["cancel"] is False
    assert capabilities.__dataclass_params__.frozen
    caps = capabilities(fetch=True, publish=False)
    assert [name for name in dir(caps) if not name.startswith("_")] == sorted(fields)
    assert caps.cancel is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.cancel = True
    assert capabilities(fetch=False, publish=True).publish
    with pytest.raises(ValueError):
        capabilities(fetch=False, publish=False)


def test_outcome_fields_and_identity():
    """Equal masks do not make two outcomes one, and an outcome hashes as itself."""
    assert [f.name for f in dataclasses.fields(Outcome)] == ["delivered_masks", "fault"]
    assert Outcome.__dataclass_params__.frozen and not Outcome.__dataclass_params__.eq
    view = _keyed_view(2)
    first, second = Outcome.for_view(view), Outcome.for_view(view)
    public = [name for name in dir(first) if not name.startswith("_")]
    assert public == ["check_mask_shapes", "delivered_masks", "fault", "for_view"]
    assert first == first and first != second
    assert len({first, second}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.fault = "peer reset"


def test_outcome_for_view_without_masks_delivers_nothing():
    assert list(inspect.signature(Outcome.for_view).parameters) == [
        "view",
        "delivered_masks",
        "fault",
    ]
    view = _keyed_view(2, 0, 3)
    outcome = Outcome.for_view(view)
    assert [mask.tolist() for mask in outcome.delivered_masks] == [[False] * 2, [], [False] * 3]
    assert outcome.fault is None
    outcome.check_mask_shapes(view)
    assert Outcome.for_view(view, fault="peer reset").fault == "peer reset"


def test_outcome_for_view_with_masks_and_a_fault():
    view = _keyed_view(2, 3)
    outcome = Outcome.for_view(view, [np.array([True, False]), np.array([0, 1, 1])], "peer reset")
    assert all(mask.dtype == np.bool_ for mask in outcome.delivered_masks)
    assert [mask.tolist() for mask in outcome.delivered_masks] == [
        [True, False],
        [False, True, True],
    ]
    assert outcome.fault == "peer reset"
    outcome.check_mask_shapes(view)


def test_outcome_masks_are_read_only_copies():
    """A backend may go on writing the masks it gave; the outcome does not change."""
    view = _keyed_view(3)
    masks = view.row_masks(True)
    outcome = Outcome.for_view(view, masks)
    masks[0][0] = False
    assert isinstance(outcome.delivered_masks, tuple)
    assert outcome.delivered_masks[0].tolist() == [True] * 3
    for mask in outcome.delivered_masks:
        assert not mask.flags.writeable
        with pytest.raises(ValueError):
            mask[0] = False


@pytest.mark.parametrize(
    "delivered_masks",
    [
        (np.zeros(2, dtype=bool),),
        (np.zeros(2, dtype=bool), np.zeros(3, dtype=bool), np.zeros(0, dtype=bool)),
        (np.zeros(1, dtype=bool), np.zeros(3, dtype=bool)),
        (np.zeros(2, dtype=bool), np.zeros(4, dtype=bool)),
        (np.zeros((2, 1), dtype=bool), np.zeros(3, dtype=bool)),
        (np.zeros(2, dtype=bool), np.zeros((1, 3), dtype=bool)),
    ],
    ids=["fewer-masks", "more-masks", "short-mask", "long-mask", "2d-mask", "2d-row-mask"],
)
def test_check_mask_shapes_refuses_masks_that_do_not_fit_the_view(delivered_masks):
    with pytest.raises(ValueError):
        Outcome(delivered_masks).check_mask_shapes(_keyed_view(2, 3))


def test_check_mask_shapes_refuses_a_mask_for_an_empty_group():
    outcome = Outcome((np.zeros(2, dtype=bool), np.zeros(1, dtype=bool)))
    with pytest.raises(ValueError):
        outcome.check_mask_shapes(_keyed_view(2, 0))


def test_an_invalid_submission_is_a_rejection_and_a_value_error():
    invalid = contract_module.InvalidSubmission
    assert contract_module.SubmissionRejected.__bases__ == (Exception,)
    assert invalid.__bases__ == (contract_module.SubmissionRejected, ValueError)
    assert issubclass(invalid, contract_module.SubmissionRejected)
    assert issubclass(invalid, ValueError)
    assert not issubclass(contract_module.SubmissionRejected, ValueError)


LENDER_MODULE = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"


def _imported_names(path: str, package_name: str) -> list:
    """Qualified module and member names imported by ``path``, including deferred imports."""
    with open(path, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                anchor = package_name.rsplit(".", node.level - 1)[0]
                module = f"{anchor}.{module}" if module else anchor
            found.append(module)
            found.extend(f"{module}.{alias.name}" for alias in node.names)
    return found


def _lender_imports(imported_names: list) -> list:
    return [
        name
        for name in imported_names
        if name == LENDER_MODULE or name.startswith(LENDER_MODULE + ".")
    ]


def test_base_imports_nothing_of_the_lender():
    root = os.path.dirname(package.__file__)
    files = sorted(name for name in os.listdir(root) if name.endswith(".py"))
    assert {"__init__.py", "backend.py"} <= set(files)
    for name in files:
        imported = _imported_names(os.path.join(root, name), package.__name__)
        assert _lender_imports(imported) == [], f"base/{name} imports {_lender_imports(imported)}"


def test_the_import_scan_sees_every_kind_of_import(tmp_path):
    source = tmp_path / "module.py"
    source.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from ...pyexecutor.kv_cache.sharing import Hidden\n"
        "def later():\n"
        "    import tensorrt_llm._torch.pyexecutor.kv_cache.sharing._types\n"
        "    from ...pyexecutor.kv_cache import sharing\n"
    )
    found = _lender_imports(_imported_names(str(source), package.__name__))
    assert f"{LENDER_MODULE}.Hidden" in found
    assert f"{LENDER_MODULE}._types" in found
    assert LENDER_MODULE in found


def test_lender_layout_and_arrays_fit_the_contract_with_explicit_field_mapping():
    """The lender's staging/block vocabulary needs a holder-side projection to generic rows.

    This test uses a structural fixture to check the mapping, not a production holder adapter.
    The raw lender view and groups do not implement the generic address/mask member names.
    """
    if importlib.util.find_spec(LENDER_MODULE) is None:
        pytest.skip("the KV cache lender is not part of this tree")
    sharing = importlib.import_module(LENDER_MODULE)
    region = sharing.StagingRegion("pool", 4096, 512, 64, 8, (7, 2))
    object_keys = np.zeros((2, 54), dtype=np.uint8)
    placed = sharing.LayerGroupBlocks(7, np.array([3, 9]), object_keys, np.array([4096, 4352]), 0)
    lease_view = sharing.LeaseView((placed, sharing.LayerGroupBlocks(2, np.array([0]))))
    view = _View(
        tuple(
            _LayerGroupRows(
                blocks.layer_group_index,
                len(blocks),
                blocks.object_keys,
                blocks.staging_addresses,
                blocks.region_index,
            )
            for blocks in lease_view.groups
        )
    )
    assert isinstance(region, SlotRegion)
    assert isinstance(view, View)
    assert all(isinstance(group, LayerGroupRows) for group in view.groups)
    assert [group.layer_group_index for group in view.groups] == [7, 2]
    assert view.groups[0].object_keys is placed.object_keys
    assert view.groups[0].addresses is placed.staging_addresses
    assert view.groups[1].addresses is view.groups[1].region_index is None
    outcome = Outcome.for_view(view, view.row_masks(fill_value=True))
    outcome.check_mask_shapes(view)
    assert [mask.tolist() for mask in outcome.delivered_masks] == [[True, True], [True]]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_is_last_has_no_default():
    """Getting it wrong ends a transfer early and in silence, so it cannot be left out."""
    with pytest.raises(TypeError):
        Chunk(
            pool_slot_ids_per_layer_group=[],
            kind_per_layer_group=[],
            token_range=TokenRange(start=0, end=8),
        )


def test_every_layer_group_is_given_a_kind():
    """Both kinds are arrays of integers, so a missing kind cannot be noticed by reading the ids."""
    with pytest.raises(ValueError):
        _chunk(
            pool_slot_ids_per_layer_group=[
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ],
        )


def test_kinds_stay_in_step_with_the_pool_slot_tables():
    """The pairing is positional, which is the only thing that ties a slot id to being a slot."""
    chunk = _chunk(
        pool_slot_ids_per_layer_group=[
            np.array([1, 2], dtype=np.int64),
            np.array([9], dtype=np.int64),
        ],
        kind_per_layer_group=[CacheKind.PAGED, CacheKind.STATE],
    )
    assert chunk.kind_per_layer_group[1] is CacheKind.STATE


def test_an_extent_identifies_the_request_and_carries_its_local_chunk():
    chunk = _chunk()
    extent = CacheExtent(request_id=7, local=chunk)
    assert extent.request_id == 7
    assert extent.local is chunk


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_delivered_can_never_be_owed_a_report():
    """Success on the strength of a report that has not arrived is refused."""
    assert Delivered(token_end=8).reports_pending is False
    with pytest.raises(ValueError):
        Delivered(token_end=8, reports_pending=True)


def test_delivered_pending_is_spelled_out_not_absent():
    """Every member answers ``pending``, so reading it never needs a default to fall back on."""
    for outcome in (
        Delivered(token_end=1),
        Failed(reason="x", reports_pending=False),
        Cancelled(by_peer=False, reports_pending=False),
    ):
        assert outcome.reports_pending is False


def test_cancelled_is_pending_until_told_otherwise():
    """A cancellation stops the ask, not the writers already holding the destination."""
    assert Cancelled().reports_pending is True
    assert Cancelled().by_peer is False


def test_failed_states_what_is_owed():
    """A failure has no default for ``reports_pending``: nobody can guess it for the caller."""
    with pytest.raises(TypeError):
        Failed(reason="x")


def test_outcomes_are_frozen():
    """A stored outcome must not be mutated into a stale claim."""
    outcome = Delivered(token_end=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.token_end = 5


# ---------------------------------------------------------------------------
# Whether a report is still owed
# ---------------------------------------------------------------------------


def test_no_conclusion_may_still_have_reports_pending():
    """A transfer that has not concluded may be queued and about to write."""
    assert _reports_may_be_pending(None) is True


def test_a_delivery_has_no_reports_pending():
    assert _reports_may_be_pending(Delivered(token_end=8)) is False


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (Failed(reason="peer died", reports_pending=True), True),
        (Failed(reason="peer died", reports_pending=False), False),
        (Cancelled(reports_pending=True), True),
        (Cancelled(reports_pending=False), False),
    ],
)
def test_reports_pending_does_not_follow_from_the_kind_of_ending(outcome, expected):
    """How a delivery ended and what it still owes are separate; neither is read off the other."""
    assert _reports_may_be_pending(outcome) is expected


# ---------------------------------------------------------------------------
# Native's Attempt
# ---------------------------------------------------------------------------


def test_poll_alone_satisfies_attempt():
    class OnePiece:
        def poll(self):
            return None

    assert isinstance(OnePiece(), extent_module.Attempt)


def test_something_without_poll_is_not_an_attempt():
    class NoPoll:
        pass

    assert not isinstance(NoPoll(), extent_module.Attempt)
