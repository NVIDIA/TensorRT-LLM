# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The lender's fast paths against eager references and against its rules alone, on stand-ins."""

import ast
import copy
import numbers
import random
import struct
import textwrap
import traceback
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    MemoryKind,
    RowSelection,
    _fetch_ledger,
    _lender,
    _manager,
    _ranges,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import DevicePool
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._page_locks import LockMode
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._types import LentLayerGroupRows

pytestmark = pytest.mark.cpu_only
# The sources whose methods the eager-reference differentials run, read by path.
_SHARING = Path(_lender.__file__).resolve().parent
TPB = 8
RID = 7
# Slots per device pool: past every page a stand-in cache draws.
SLOTS = 64
DEVICE, ALL = MemoryKind.DEVICE, RowSelection.ALL


def _load(file, qualified, ns):
    source = (_SHARING / file).read_text()
    node = ast.parse(source)
    for part in qualified.split("."):
        node = next(child for child in node.body if getattr(child, "name", None) == part)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    code = textwrap.dedent("\n".join(source.splitlines()[first - 1 : node.end_lineno]))
    exec(compile("from __future__ import annotations\n" + code, str(_SHARING / file), "exec"), ns)
    return ns[node.name]


def _sources(immutable_mutant=None):
    manager = SimpleNamespace(
        # A cache's committed tokens: none where no layer group committed any.
        cache_state=lambda kv: SimpleNamespace(
            history=kv.history,
            active=kv.active,
            committed=max(getattr(kv, "committed", ()), default=0),
        ),
        committed_by_group=lambda kv, count: kv.committed,
        pages=lambda kv, lg: kv.pages[lg],
        locked_pages=lambda kv, lg: kv.pages[lg],
        state_slot=lambda kv, lg: kv.state_slots[lg],
        num_blocks=lambda kv: kv.num_blocks,
        stale_range=lambda mgr, lg, history: (
            mgr.stale_at(lg, history) if hasattr(mgr, "stale_at") else mgr.stale[lg]
        ),
        context_end=lambda request, mgr: request.prompt,
        context_computed=lambda request: request.context_current_position,
        bidirectional_runs=lambda request: [],
        _MAX_TOKENS=2**31 - 1,
    )
    ns = dict(__name__=__name__, np=np, struct=struct, numbers=numbers, traceback=traceback)
    ns.update(dataclass=dataclass, field=field, _manager=manager, REUSE_KEY_BYTES=32)
    ns.update(_OBJECT_KEY_BYTES=54, OBJECT_KEY_BYTES=54, logger=SimpleNamespace(error=pytest.fail))
    for name in ("_immutable", "_frozen_int64", "_integer", "_entry_integers", "_region_indices"):
        _load("_types.py", name, ns)
    if immutable_mutant == "identity":
        ns["_immutable"] = lambda array: array
    elif immutable_mutant == "readonly_flag":
        good = ns["_immutable"]
        ns["_immutable"] = lambda array: array if not array.flags.writeable else good(array)
    for name in ("LentLayerGroupRows", "LeaseView"):
        _load("_types.py", name, ns)
    _load("_lender.py", "_LentBlockEntries", ns)
    _load("_lender.py", "_frozen_array", ns)
    _load("_lender.py", "_unshared", ns)
    ns["_NO_ROWS"] = ns["_frozen_int64"](())
    ns["_STATE_ROW"] = ns["_frozen_int64"]((0,))
    ranges = SimpleNamespace()
    for name in ("_stale_range", "_needed_ranges", "_last_history_keeping"):
        setattr(ranges, name, _load("_ranges.py", name, ns))
    ns["_ranges"] = ranges
    _load("_identity.py", "Identity.object_keys", ns)
    result = SimpleNamespace(
        object_keys=ns["object_keys"],
        ns=ns,
        manager=manager,
        ranges=ranges,
        entries=ns["_LentBlockEntries"],
        rows=ns["LentLayerGroupRows"],
        view=ns["LeaseView"],
    )
    _load("_fetch_ledger.py", "_writable_mask", ns)
    ledger_methods = """_unheld_block _computed_tokens _known _prefix_kept_by_fetch
        _invalidate_delivered_blocks_past_capacity"""
    for name in ledger_methods.split():
        setattr(result, name, _load("_fetch_ledger.py", f"_FetchLedger.{name}", ns))
    lender_methods = """_locked_block_masks _history_cap _without_short_windows_mid_context
        _mid_context_end
        _range_ordinals _block_entries_of_range _lendable_block_entries _committed_entries
        _build_view _in_place_view _on_free _on_reset _on_shutdown"""
    for name in lender_methods.split():
        setattr(result, name, _load("_lender.py", f"_ManagerLender.{name}", ns))
    return result


@pytest.fixture(scope="module")
def sources():
    return _sources()


def _bind(obj, sources, *names):
    for name in names:
        setattr(obj, name, getattr(sources, name).__get__(obj))


def _outcome(call):
    try:
        return "ok", call()
    except Exception as error:  # noqa: BLE001 - compare complete exception outcomes
        return "raise", type(error), str(error)


def _root(array):
    """The object at the base of ``array``'s chain of views."""
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


def _slow_unheld(ledger, rid, kv, entries, top):
    """The eager path: STATE and mixed tails, then computed prefixes, then delivered rows."""
    delivered = ledger._deliveries.get(rid)
    if delivered is not None and delivered.kv is not kv:
        delivered = None
    tpb = ledger._layout.tokens_per_block
    for lg, ordinals in zip(entries.layer_group_indices, entries.block_ordinals):
        if not len(ordinals):
            continue
        if lg in ledger._recurrent_groups:
            if delivered is not None and delivered.state_end != top:
                return lg, 0
        elif ledger._mixed_tail_caches.get(rid) is kv:
            tail, partial = divmod(kv.committed[lg], tpb)
            if partial and tail in ordinals:
                return lg, tail
    computed = ledger._computed_tokens(rid, kv)
    if ledger._known(computed) >= top:
        return None
    rank, size = ledger._layout.cp
    page = tpb // size
    for lg, ordinals in zip(entries.layer_group_indices, entries.block_ordinals):
        if lg in ledger._recurrent_groups:
            continue
        for block in ordinals.tolist():
            first = block * tpb + rank * page
            if first >= top:
                continue
            valid = computed[lg]
            if delivered is not None:
                mask = delivered.delivered_block_masks[lg]
                if block < len(mask) and mask[block]:
                    valid = max(valid, delivered.partial_ends.get((lg, block), first + page))
            if valid < min(first + page, top):
                return lg, block
    return None


def _ledger(sources, layout, kv, *, advanced=None, delivered=None, tainted=False):
    ledger = SimpleNamespace(
        _layout=layout,
        _recurrent_groups=frozenset(
            lg for lg, recurrent in enumerate(layout.recurrent) if recurrent
        ),
        _paged_groups=tuple(lg for lg, recurrent in enumerate(layout.recurrent) if not recurrent),
        _kept_prefixes={} if advanced is None else {7: advanced},
        _deliveries={} if delivered is None else {7: delivered},
        _mixed_tail_caches={7: kv} if tainted else {},
    )
    _bind(ledger, sources, "_known", "_prefix_kept_by_fetch", "_computed_tokens")
    return ledger


def _delivery(kv, masks, top=0):
    return SimpleNamespace(
        kv=kv,
        delivered_block_masks=masks,
        state_end=top,
        end=top,
        partial_ends={},
        resume_bounds_memo=("memo",),
    )


def _delivery_key(delivered):
    return (
        tuple(mask.tobytes() for mask in delivered.delivered_block_masks),
        delivered.end,
        tuple(sorted(delivered.partial_ends.items())),
        delivered.resume_bounds_memo,
    )


def test_unheld_block_and_ledger_shortcuts_match_eager_paths(sources):
    rng = random.Random(20261016)
    for _ in range(600):
        count, tpb = rng.randint(1, 3), rng.choice((4, 8))
        size = rng.choice((1, 2))
        layout = SimpleNamespace(
            num_layer_groups=count,
            tokens_per_block=tpb,
            recurrent=tuple(rng.random() < 0.25 for _ in range(count)),
            cp=(rng.randrange(size), size),
        )
        kv = SimpleNamespace(
            history=rng.randint(0, 64),
            active=True,
            committed=tuple(rng.randint(0, 48) for _ in range(count)),
            num_blocks=rng.randint(0, 10),
        )
        other = copy.copy(kv)
        advanced = rng.choice(
            (
                None,
                (kv, (0,) * count, kv.history),
                (kv, (0,) * count, rng.randint(0, 64)),
                (other, (0,) * count, kv.history),
            )
        )
        delivered = _delivery(
            rng.choice((kv, kv, other)),
            [
                np.array([rng.random() < 0.7 for _ in range(rng.randint(0, 14))], dtype=bool)
                for _ in range(count)
            ],
            rng.randint(0, 72),
        )
        delivered.partial_ends = {(rng.randrange(count), rng.randrange(14)): rng.randrange(72)}
        entries = sources.entries(
            list(range(count)),
            [
                np.array(rng.sample(range(14), rng.randint(0, 7)), dtype=np.int64)
                for _ in range(count)
            ],
            [],
        )
        top, tainted = rng.randint(0, 72), rng.random() < 0.25
        ledger = _ledger(
            sources, layout, kv, advanced=advanced, delivered=delivered, tainted=tainted
        )
        before = _delivery_key(delivered)
        kept_before = dict(ledger._kept_prefixes)
        tainted_before = dict(ledger._mixed_tail_caches)
        entries_before = tuple(array.tobytes() for array in entries.block_ordinals)
        old_result = _outcome(lambda: _slow_unheld(ledger, 7, kv, entries, top))
        assert old_result == _outcome(lambda: sources._unheld_block(ledger, 7, kv, entries, top))
        assert old_result == _outcome(
            lambda: sources._unheld_block(ledger, 7, kv, entries, top, kv.history)
        )
        assert ledger._kept_prefixes == kept_before and ledger._mixed_tail_caches == tainted_before
        assert tuple(array.tobytes() for array in entries.block_ordinals) == entries_before
        assert _delivery_key(delivered) == before
        start = rng.choice((0, 0, rng.randint(0, 64)))
        assert ledger._prefix_kept_by_fetch(7, kv, start) == tuple(
            min(tokens, start) for tokens in ledger._computed_tokens(7, kv)
        )
        old, new = copy.deepcopy(delivered), copy.deepcopy(delivered)
        kept = kv.num_blocks
        old.end = min(old.end, kept * tpb)
        old.partial_ends = {key: end for key, end in old.partial_ends.items() if key[1] < kept}
        if old.end != delivered.end:
            old.resume_bounds_memo = None
        for mask in old.delivered_block_masks:
            if mask[kept:].any():
                mask[kept:] = False
                old.resume_bounds_memo = None
        sources._invalidate_delivered_blocks_past_capacity(ledger, new, kv)
        assert _delivery_key(new) == _delivery_key(old)
        sources._invalidate_delivered_blocks_past_capacity(ledger, new, kv)
        assert _delivery_key(new) == _delivery_key(old)


@pytest.mark.parametrize(
    "case", ["before_fetch_end", "at_fetch_end", "after_fetch_end", "missing_state", "mixed_tail"]
)
def test_unheld_block_guards_history_state_and_mixed_tail(sources, case):
    """A read refuses the first unreadable block across history and fetch boundaries."""
    kv = SimpleNamespace(history=16, active=True, committed=(5, 0))
    delivered = _delivery(kv, [np.array([True, True, False, True]), np.ones(1, bool)], 8)
    lg, blocks = (1, [0]) if case == "missing_state" else (0, [1] if case == "mixed_tail" else [2])
    entries = sources.entries([lg], [np.array(blocks, dtype=np.int64)], [])
    ledger = _ledger(
        sources,
        layout([False, True], tpb=4),
        kv,
        delivered=delivered,
        tainted=case == "mixed_tail",
        advanced=None if case in ("missing_state", "mixed_tail") else (kv, (0, 0), 16),
    )
    kv.history = {"before_fetch_end": 15, "after_fetch_end": 17}.get(case, 16)
    expected = {
        "before_fetch_end": (0, 2),
        "at_fetch_end": (0, 2),
        "after_fetch_end": None,
        "missing_state": (1, 0),
        "mixed_tail": (0, 1),
    }[case]
    assert _slow_unheld(ledger, 7, kv, entries, 16) == expected
    assert sources._unheld_block(ledger, 7, kv, entries, 16) == expected


def test_taint_from_a_replaced_cache_does_not_block_the_current_cache(sources):
    kv = SimpleNamespace(history=8, active=True, committed=(5,))
    ledger = _ledger(sources, layout([False], tpb=4), kv)
    ledger._mixed_tail_caches[7] = object()
    entries = sources.entries([0], [np.array([1], dtype=np.int64)], [])
    assert _slow_unheld(ledger, 7, kv, entries, 8) is None
    assert sources._unheld_block(ledger, 7, kv, entries, 8) is None


def _slow_masks(lender, kv, lease):
    masks = []
    for lg, ordinals, slots in zip(
        lease._block_entries.layer_group_indices,
        lease._block_entries.block_ordinals,
        lease._block_entries.device_pages,
    ):
        same = np.zeros(len(ordinals), dtype=bool)
        if kv is not None and kv is lease._kv and kv.active:
            if lg in lender._recurrent_groups:
                same[:] = slots == kv.state_slots[lg]
            else:
                pages = kv.pages[lg]
                inside = ordinals < len(pages)
                same[inside] = pages[ordinals[inside]] == slots[inside]
                begin, end = (
                    lender._manager_ref().stale[lg]
                    if lender._layout.windows[lg] is not None
                    else (0, 0)
                )
                same &= (ordinals < begin) | (ordinals >= end)
        masks.append(same)
    return masks


def test_locked_masks_match_eager_paths(sources):
    rng = random.Random(18)
    for _ in range(600):
        groups = rng.randint(1, 3)
        windows = [rng.choice((None, 8, 16)) for _ in range(groups)]
        manager = SimpleNamespace(
            stale=[tuple(sorted((rng.randrange(4), rng.randrange(8)))) for _ in range(groups)]
        )
        layout = SimpleNamespace(windows=windows)
        lender = SimpleNamespace(
            _layout=layout,
            _manager_ref=lambda: manager,
            _recurrent_groups=frozenset(lg for lg in range(groups) if rng.random() < 0.2),
            _state_slot=lambda kv, request_id, lg: kv.state_slots[lg],
        )
        kv = SimpleNamespace(
            active=rng.random() < 0.8,
            history=rng.randrange(64),
            pages=[
                np.array([rng.randrange(20) for _ in range(rng.randrange(12))], dtype=np.int64)
                for _ in range(groups)
            ],
            state_slots=[rng.randrange(20) for _ in range(groups)],
        )
        ordinals = [
            np.array(rng.sample(range(15), rng.randrange(8)), dtype=np.int64) for _ in range(groups)
        ]
        entries = sources.entries(
            list(range(groups)),
            ordinals,
            [np.array([rng.randrange(20) for _ in blocks], dtype=np.int64) for blocks in ordinals],
        )
        lease = SimpleNamespace(
            _kv=kv if rng.random() < 0.85 else object(), _block_entries=entries, _request_id=7
        )
        current = kv if rng.random() < 0.9 else None
        before = [a.tobytes() for a in entries.block_ordinals + entries.device_pages]
        expected = _slow_masks(lender, current, lease)
        actual = sources._locked_block_masks(lender, current, lease)
        assert [a.tobytes() for a in actual] == [a.tobytes() for a in expected]
        assert before == [a.tobytes() for a in entries.block_ordinals + entries.device_pages]


@pytest.mark.parametrize(
    "blocks", [[6, 1, 0], [0, 2, 6], [5, 0, 9, 3, 1]], ids=["down", "up", "mixed"]
)
def test_ordinals_past_the_cache_s_end_in_any_order_get_and_lock_no_page(sources, blocks):
    manager = SimpleNamespace(stale=[(1, 3)])
    lender = _metadata_lender(sources, layout([False], windows=(8,)), manager)
    kv = SimpleNamespace(active=True, history=16, pages=[np.array([10, 11, 12, 13])])
    ordinals = [np.array(blocks, dtype=np.int64)]
    supplied = sources._block_entries_of_range(lender, manager, kv, 0, 16, ordinals=ordinals)
    assert supplied.device_pages[0].tolist() == [10 + b if b < 4 else -1 for b in blocks]
    slots = [np.array([10 + b if b < 4 else 10 for b in blocks])]
    lease = SimpleNamespace(_kv=kv, _block_entries=sources.entries([0], ordinals, slots))
    expected = [[b < 4 and not 1 <= b < 3 for b in blocks]]
    assert [a.tolist() for a in _slow_masks(lender, kv, lease)] == expected
    assert [a.tolist() for a in sources._locked_block_masks(lender, kv, lease)] == expected


def _bytes_key(array):
    return None if array is None else (array.dtype.str, array.shape, array.tobytes())


def _entry_key(entries):
    return (
        tuple(entries.layer_group_indices),
        tuple(map(_bytes_key, entries.block_ordinals)),
        tuple(map(_bytes_key, entries.device_pages)),
    )


def _assert_built_frozen(out, incoming):
    """Every array of ``out`` other than ``incoming``'s own lies on immutable bytes, as the locks a
    lend asks for take its ordinals."""
    given = {id(array) for array in (*incoming.block_ordinals, *incoming.device_pages)}
    for array in (*out.block_ordinals, *out.device_pages):
        if id(array) not in given:
            root = _root(array)
            assert not array.flags.writeable and isinstance(root, bytes), type(root).__name__


def _view_key(view):
    return tuple(
        (
            g.layer_group_index,
            _bytes_key(g.page_ordinals),
            _bytes_key(g.object_keys),
            _bytes_key(g.addresses),
            g.region_index,
            _bytes_key(g.slots),
            g.region_indices,
        )
        for g in view.groups
    )


def _assert_frozen_view(view):
    for group in view.groups:
        for array in (group.page_ordinals, group.object_keys, group.addresses, group.slots):
            if array is not None:
                assert not array.flags.writeable
                with pytest.raises(ValueError):
                    array.flags.writeable = True


def _metadata_lender(sources, layout, manager):
    lender = SimpleNamespace(
        _layout=layout,
        _ordinal_ranges={},
        _recurrent_groups=frozenset(lg for lg, flag in enumerate(layout.recurrent) if flag),
        _named_groups=layout.named_groups,
        _manager_ref=lambda: manager,
        _any_window=any(window is not None for window in layout.windows),
        _regions=[
            SimpleNamespace(address=0x10000 + i * 0x10000, slot_bytes=128 << i) for i in range(2)
        ],
        _region_index_of_pool_group={0: 0, 1: 1},
        _device_regions_of_pool_group={0: (2, 3), 1: (4, 5)},
        # A manager without a record of state slots: the cache answers.
        _state_slot=lambda kv, request_id, lg: kv.state_slots[lg],
    )
    lender._object_keys = lambda lg, keys: np.pad(keys, ((0, 0), (0, 22)), constant_values=lg)
    _bind(lender, sources, "_range_ordinals", "_mid_context_end")
    return lender


def _slow_entries(sources, lender, manager, kv, starts, end, named):
    """The eager path: ordinal arrays made anew, device pages filled through a bounds mask."""
    layout, tpb = lender._layout, lender._layout.tokens_per_block
    out = sources.entries([], [], [])
    rank, size = layout.cp
    for lg, start in enumerate(starts):
        if lg in lender._recurrent_groups:
            blocks, slots = np.zeros(1, dtype=np.int64), np.array([kv.state_slots[lg]], np.int64)
        else:
            runs = sources.ranges._needed_ranges(
                manager, layout, lg, start // tpb, -(-end // tpb), end
            )
            blocks = np.array(
                [block for begin, stop in runs for block in range(begin, stop)], np.int64
            )
            if named and not layout.named_groups[lg]:
                blocks = blocks[:0]
            if size > 1:
                blocks = blocks[blocks * tpb + rank * (tpb // size) < end]
            slots = np.full(len(blocks), -1, dtype=np.int64)
            inside = blocks < len(kv.pages[lg])
            slots[inside] = kv.pages[lg][blocks[inside]]
        out.layer_group_indices.append(lg)
        out.block_ordinals.append(blocks)
        out.device_pages.append(slots)
    return out


def _slow_view(sources, lender, entries, keys, device):
    """The eager view; the public row constructor freezes its inputs."""
    groups = []
    rank, size = lender._layout.cp
    for index, (lg, blocks) in enumerate(zip(entries.layer_group_indices, entries.block_ordinals)):
        group = lender._layout.pool_group_of[lg]
        if device:
            groups.append(
                sources.rows(
                    lg,
                    blocks * size + rank if size > 1 else blocks,
                    object_keys=None if keys is None else lender._object_keys(lg, keys[index]),
                    slots=entries.device_pages[index],
                    region_indices=lender._device_regions_of_pool_group[group],
                )
            )
        else:
            region_index = lender._region_index_of_pool_group[group]
            region = lender._regions[region_index]
            groups.append(
                sources.rows(
                    lg,
                    blocks,
                    object_keys=lender._object_keys(lg, keys[index]),
                    addresses=region.address + entries.staging_slots[index] * region.slot_bytes,
                    region_index=region_index,
                )
            )
    return sources.view(tuple(groups), 0, 0)


def test_cached_ranges_and_views_match_eager_paths(sources):
    rng = random.Random(23)
    for _ in range(160):
        count, tpb, size = rng.randint(1, 3), rng.choice((4, 8)), rng.choice((1, 2))
        layout = SimpleNamespace(
            num_layer_groups=count,
            tokens_per_block=tpb,
            windows=[rng.choice((None, 8, 16)) for _ in range(count)],
            recurrent=[rng.random() < 0.2 for _ in range(count)],
            named_groups=[rng.random() < 0.8 for _ in range(count)],
            pool_group_of=[lg % 2 for lg in range(count)],
            cp=(rng.randrange(size), size),
            sink_blocks=[rng.randrange(3) for _ in range(count)],
        )
        manager = SimpleNamespace()
        manager.stale_at = lambda lg, h: (
            min(-(-h // tpb), layout.sink_blocks[lg]),
            max(
                min(-(-h // tpb), layout.sink_blocks[lg]),
                int((h + 1 - (layout.windows[lg] or h + 1)) / tpb),
            ),
        )
        lender = _metadata_lender(sources, layout, manager)
        kv = SimpleNamespace(
            state_slots=[rng.randrange(32) for _ in range(count)],
            history=rng.randrange(64),
            active=True,
        )
        snapshots = []
        for step in range(3):
            end = rng.randint(1, 48)
            starts = [rng.randrange(end + 1) for _ in range(count)]
            kv.pages = [
                np.array([rng.randrange(32) for _ in range(rng.randrange(15))], np.int64)
                for _ in range(count)
            ]
            named = rng.random() < 0.5
            entries = sources._block_entries_of_range(
                lender, manager, kv, starts, end, named=named, request_id=7
            )
            expected = _slow_entries(sources, lender, manager, kv, starts, end, named)
            assert _entry_key(entries) == _entry_key(expected)
            assert len(lender._ordinal_ranges.get(7, {})) <= 2 * count
            # A later call reuses ordinal templates but must reread this cache's current slots.
            for pages in kv.pages:
                pages += 100
            refreshed = sources._block_entries_of_range(
                lender, manager, kv, starts, end, named=named, request_id=7
            )
            assert _entry_key(refreshed) == _entry_key(
                _slow_entries(sources, lender, manager, kv, starts, end, named)
            )
            entries.staging_slots = [
                np.arange(len(blocks), dtype=np.int64) + step for blocks in entries.block_ordinals
            ]
            keys = [
                np.full((len(blocks), 32), step + lg, np.uint8)
                for lg, blocks in zip(entries.layer_group_indices, entries.block_ordinals)
            ]
            view = sources._build_view(lender, entries, keys, 0, 0)
            assert _view_key(view) == _view_key(_slow_view(sources, lender, entries, keys, False))
            _assert_frozen_view(view)
            snapshots.append((view, _view_key(view)))
            device = sources.entries(entries.layer_group_indices, [], [])
            for blocks, pages in zip(entries.block_ordinals, entries.device_pages):
                device.block_ordinals.append(blocks[pages >= 0])
                device.device_pages.append(pages[pages >= 0])
            device_view = sources._in_place_view(lender, device, None, 0, 0)
            assert _view_key(device_view) == _view_key(
                _slow_view(sources, lender, device, None, True)
            )
            _assert_frozen_view(device_view)
            snapshots.append((device_view, _view_key(device_view)))
            for values in (
                entries.staging_slots + keys + device.block_ordinals + device.device_pages
            ):
                values[:] = 0
            assert all(_view_key(old) == value for old, value in snapshots)
            # The window filter keeps each group's rows and the history limit of the eager path.
            prompt = rng.randrange(64)
            computed = prompt if rng.random() < 0.3 else rng.randrange(prompt + 1)
            request = SimpleNamespace(
                py_request_id=7, prompt=prompt, context_current_position=computed
            )
            filtered = sources._without_short_windows_mid_context(
                lender, manager, request, SimpleNamespace(history=kv.history), entries
            )
            keep = [
                kv.history >= request.prompt
                or computed >= request.prompt
                or window is None
                or window >= request.prompt
                for window in layout.windows
            ]
            wanted = sources.entries(
                entries.layer_group_indices,
                [
                    blocks if flag else blocks[:0]
                    for blocks, flag in zip(entries.block_ordinals, keep)
                ],
                [pages if flag else pages[:0] for pages, flag in zip(entries.device_pages, keep)],
            )
            assert _entry_key(filtered) == _entry_key(wanted)
            lowest = {lg: rng.randrange(5) for lg in range(count)}
            lowest = {lg: block for lg, block in lowest.items() if block >= layout.sink_blocks[lg]}
            lender._locks = SimpleNamespace(lowest_locked_blocks=lambda cache, floor: lowest)
            limits = [
                sources.ranges._last_history_keeping(manager, layout, lg, block)
                for lg, block in lowest.items()
                if layout.windows[lg] is not None
            ]
            assert sources._history_cap(lender, kv) == min(limits, default=None)


def _storage(array):
    return np.frombuffer(_root(array), dtype=np.uint8)


def test_ordinal_templates_share_capacity_and_leave_old_arrays_unchanged(sources):
    lender = _metadata_lender(sources, layout([False]), object())
    first = lender._range_ordinals(7, 0, [(0, 4)])
    second = lender._range_ordinals(7, 0, [(4, 8)])
    assert np.shares_memory(_storage(first), _storage(second))
    before = first.tobytes(), second.tobytes()
    third = lender._range_ordinals(7, 0, [(8, 12)])
    assert third.tolist() == [8, 9, 10, 11]
    assert not np.shares_memory(_storage(first), _storage(third))
    assert (first.tobytes(), second.tobytes()) == before
    for values in (first, second, third):
        with pytest.raises(ValueError):
            values.flags.writeable = True
    assert lender._range_ordinals(None, 0, [(1, 3), (8, 10)]).tolist() == [1, 2, 8, 9]
    assert set(lender._ordinal_ranges) == {7}
    large = lender._range_ordinals(8, 0, [(10_000, 10_004)])
    assert large.tolist() == [10_000, 10_001, 10_002, 10_003]
    assert _storage(large).nbytes <= 2 * len(large) * large.itemsize, (
        "range storage must be bounded by row count, not absolute ordinal"
    )


@pytest.mark.parametrize("mutant", ["identity", "readonly_flag"])
def test_metadata_mutants_expose_writable_or_changing_old_views(mutant):
    correct, broken = _sources(), _sources(mutant)
    for methods, should_fail in ((correct, False), (broken, True)):
        lender = _metadata_lender(methods, layout([False]), object())
        raw_blocks, raw_slots = np.array([1, 2]), np.array([3, 4])
        blocks, slots = raw_blocks.view(), raw_slots.view()
        blocks.flags.writeable = slots.flags.writeable = False
        entries = methods.entries([0], [blocks], [slots])
        view = methods._in_place_view(lender, entries, None, 0, 0)
        before = _view_key(view)
        raw_blocks[:] = 9
        raw_slots[:] = 8
        if should_fail:
            assert _view_key(view) != before, "the mutant failed to expose a template alias"
            view.groups[0].page_ordinals.flags.writeable = True
            assert view.groups[0].page_ordinals.flags.writeable
        else:
            assert _view_key(view) == before
            _assert_frozen_view(view)


def test_ordinal_cache_is_cleared_on_free_reset_and_shutdown(sources):
    lender = SimpleNamespace(
        _ordinal_ranges={7: {}, 8: {}},
        _shrink_epochs={7: 2, 8: 1},
        _writes_at_end={},
        _state_unverified={7, 8},
        _parked={},
        _parked_closing=[],
        _stop_reason=None,
        _shut_down=False,
        _waiting_for_slots=(),
        _serving_manager=lambda: object(),
        _fail_waiting=lambda rid, reason: pytest.fail("no lease waits for slots"),
        _ledger=SimpleNamespace(_drop_records=lambda rid: None),
        _progress=lambda: pytest.fail("the free hook runs no round of progress"),
    )
    sources._on_free(lender, 7, object())
    assert lender._ordinal_ranges == {8: {}} and lender._shrink_epochs == {8: 1}
    assert lender._state_unverified == {8}
    sources._on_reset(lender)
    assert lender._ordinal_ranges == {} and lender._reuse_was_reset
    lender._ordinal_ranges[7] = {}
    lender._shut_down = False
    lender._device_regions_of_pool_group = {}
    lender._leases_holding_slots = lender._waiting_for_slots = lender._open_leases = ()
    lender._progress_round = 0
    lender._recycle = lambda: None
    lender._memory_ref = lambda: None
    sources._on_shutdown(lender)
    assert lender._ordinal_ranges == {} and lender._shut_down


def _slow_object_keys(identity, group, keys):
    if not 0 <= group < len(identity._groups):
        raise ValueError(f"unknown layer group {group}")
    keys = np.ascontiguousarray(keys, dtype=np.uint8).reshape(-1, 32)
    out = np.empty((len(keys), 54), dtype=np.uint8)
    out[:, :16] = np.frombuffer(identity.namespace, dtype=np.uint8)
    out[:, 16:48] = keys
    out[:, 48:] = identity._groups[group]
    return out


def test_object_key_bytes_and_snapshots_match_eager_layout(sources):
    rng = random.Random(29)
    for _ in range(200):
        count = rng.choice((0, 0, 1, 4, 15))
        raw = np.frombuffer(rng.randbytes(count * 64), dtype=np.uint8).copy().reshape(count, 64)
        keys = raw[:, ::2]
        suffix = struct.pack(">HHH", rng.randrange(65536), 4, 2)
        identity = SimpleNamespace(
            namespace=bytes(range(16)), _groups=[np.frombuffer(suffix, dtype=np.uint8).copy()]
        )
        expected = _slow_object_keys(identity, 0, keys)
        result = sources.object_keys(identity, 0, keys)
        assert result.shape == (count, 54) and result.dtype == np.uint8
        assert result.tobytes() == expected.tobytes()
        assert not result.flags.writeable
        with pytest.raises(ValueError):
            result.flags.writeable = True
        assert all(row[:16].tobytes() == identity.namespace for row in result)
        assert np.array_equal(result[:, 16:48], keys)
        assert np.array_equal(result[:, 48:], np.broadcast_to(identity._groups[0], (count, 6)))
        snapshot = result.tobytes()
        raw[:] = 0
        identity._groups[0][:] = 0
        identity.namespace = b"!" * 16
        assert result.tobytes() == snapshot
    for invalid in (-1, 1):
        assert _outcome(lambda: sources.object_keys(identity, invalid, np.zeros((0, 32)))) == (
            _outcome(lambda: _slow_object_keys(identity, invalid, np.zeros((0, 32))))
        )
    malformed = np.zeros(31, np.uint8)
    assert _outcome(lambda: sources.object_keys(identity, 0, malformed)) == (
        _outcome(lambda: _slow_object_keys(identity, 0, malformed))
    )


def test_an_all_write_preserves_records_and_does_not_grow(sources, kit):
    manager_api = copy.copy(sources.manager)
    ns = dict(
        sources.ns,
        _manager=manager_api,
        MemoryKind=MemoryKind,
        RowSelection=RowSelection,
        LockMode=LockMode,
        _CUDA_EVENT=Event,
        _check_ready_event=lambda event: None,
    )
    lend_write = _load("_lender.py", "_ManagerLender.lend_write", ns.copy())
    manager_api.prompt_length = lambda request, **kwargs: request.prompt
    manager_api.kv_of = lambda manager, rid: manager.kv
    manager_api.connector_load_pending = lambda *args: False
    manager_api.grow = lambda *args: pytest.fail("ALL writes must not grow their cache")
    # The stand-in manager holds back no commit for a fetch to drop.
    manager_api.drop_held_commit = lambda *args: None

    def run(start, end, pages):
        kv = SimpleNamespace(pages=[pages.copy()], committed=(0,), active=True, history=0)
        manager = SimpleNamespace(kv=kv, stale=[(0, 0)])
        lender = _metadata_lender(sources, layout([False], tpb=4), manager)
        lender._check_memory_and_rows = lambda *args: None
        lender._whole_blocks = lambda start, end, whole_end=True: (start, end)
        lender._lending_manager = lambda: manager
        lender._progress = lender._check_not_stopped = lambda: None
        lender._lends_fast = lambda: False  # the rules alone, which this test compares
        lender._required_slots_per_pool_group = lambda counts: {0: sum(counts)}
        lender._write_range_refusal = lender._bidirectional_split_refusal = lambda *args: None
        lender._scratch_reuse_on = lambda cache: False
        lender._history_refusal = lender._unwritten_pages_refusal = lambda *args: None
        lender._history_cap = sources._history_cap.__get__(lender)
        lender._locks = SimpleNamespace(
            check=lambda cache, blocks, mode, **kwargs: SimpleNamespace(blocks=blocks)
        )
        lender._pageless = lambda entries: (
            "missing" if any((slots < 0).any() for slots in entries.device_pages) else None
        )
        lender._ledger = SimpleNamespace(
            _unsettled_fetch_refusal=lambda *args: None,
            _prefix_kept_by_fetch=lambda *args: (0,),
            _record_fetch=lambda *args, **kwargs: SimpleNamespace(
                record=(args[0], args[2:], kwargs)
            ),
        )
        lender._lend_in_place = (
            lambda mgr, req, cache, kind, entries, keys, fetch, doomed, **kwargs: SimpleNamespace(
                entries=entries, fetch=fetch, doomed=doomed, checked=kwargs.get("checked")
            )
        )

        lender._block_entries_of_range = sources._block_entries_of_range.__get__(lender)
        request = SimpleNamespace(py_request_id=7, prompt=48)
        lease = lend_write(lender, request, start, end, **kit.DEVICE_ALL)
        assert np.array_equal(kv.pages[0], pages) and kv.history == 0
        assert lease.doomed is None and lease.fetch.entries is lease.entries
        expected_ordinals = list(range(start // 4, -(-end // 4)))
        assert lease.entries.layer_group_indices == [0]
        assert lease.entries.block_ordinals[0].tolist() == expected_ordinals
        assert lease.entries.device_pages[0].tolist() == pages[expected_ordinals].tolist()
        # The write locks the same ordinal array it lends, without another cache lookup.
        assert list(lease.checked.blocks) == [0]
        assert lease.checked.blocks[0] is lease.entries.block_ordinals[0]
        assert lease.fetch.record == (
            7,
            (start, (0,), 0, (0,)),
            {"state_end": None, "end": end, "prompt": 48, "runs": []},
        )

    rng = random.Random(31)
    for _ in range(100):
        start = 4 * rng.randrange(12)
        end = rng.randint(start, 48)
        pages = np.array(rng.sample(range(1000), 12), dtype=np.int64)
        run(start, end, pages)


def _slow_lendable(sources, lender, manager, entries):
    out = sources.entries([], [], [])
    for lg, blocks, pages in zip(
        entries.layer_group_indices, entries.block_ordinals, entries.device_pages
    ):
        begin, end = (0, 0) if lender._layout.windows[lg] is None else manager.stale[lg]
        kept = [
            (block, page)
            for block, page in zip(blocks.tolist(), pages.tolist())
            if page >= 0 and not begin <= block < end
        ]
        out.layer_group_indices.append(lg)
        out.block_ordinals.append(np.array([block for block, _ in kept], dtype=np.int64))
        out.device_pages.append(np.array([page for _, page in kept], dtype=np.int64))
    return out


def test_lendable_and_committed_rows_match_scalar_filters(sources):
    rng = random.Random(5820)
    covered = set()
    for _ in range(300):
        count, tpb = rng.randrange(5), rng.choice((4, 8))
        # A window with nothing behind it yet has an empty stale range.
        manager = SimpleNamespace(
            stale=[
                tuple(sorted(rng.sample(range(16), 2))) if rng.random() < 0.8 else (3, 3)
                for _ in range(count)
            ]
        )
        named = [rng.random() < 0.75 for _ in range(count)]
        lender = SimpleNamespace(
            _layout=SimpleNamespace(
                windows=[rng.choice((None, 8, 16)) for _ in range(count)],
                num_layer_groups=count,
                tokens_per_block=tpb,
            ),
            _named_groups=named,
        )
        kv = SimpleNamespace(committed=tuple(rng.randrange(24) * tpb for _ in range(count)))
        blocks = [
            np.array([rng.randrange(20) for _ in range(rng.randrange(15))], dtype=np.int64)
            for _ in range(count)
        ]
        entries = sources.entries(
            list(range(count)),
            blocks,
            [np.array([rng.randrange(-2, 20) for _ in group], dtype=np.int64) for group in blocks],
        )
        before = _entry_key(entries)
        expected = _slow_lendable(sources, lender, manager, entries)
        history = rng.randrange(100)
        actual = sources._lendable_block_entries(lender, manager, history, entries)
        assert _entry_key(actual) == _entry_key(expected), (
            "lendable rows include stale/missing pages"
        )
        _assert_built_frozen(actual, entries)
        for incoming in (entries, actual):
            expected = sources.entries([], [], [])
            for lg, ordinals, slots in zip(
                incoming.layer_group_indices, incoming.block_ordinals, incoming.device_pages
            ):
                rows = list(zip(ordinals.tolist(), slots.tolist()))
                kept = [(b, s) for b, s in rows if named[lg] and b * tpb + tpb <= kv.committed[lg]]
                expected.layer_group_indices.append(lg)
                expected.block_ordinals.append(np.array([b for b, _ in kept], dtype=np.int64))
                expected.device_pages.append(np.array([s for _, s in kept], dtype=np.int64))
                covered.add(
                    "empty"
                    if not rows
                    else "whole"
                    if len(kept) == len(rows)
                    else "none"
                    if not kept
                    else "partial"
                )
                if not named[lg]:
                    covered.add("unnamed")
                if any(a > b for a, b in zip(ordinals.tolist(), ordinals.tolist()[1:])):
                    covered.add("unsorted")
            committed = sources._committed_entries(lender, kv, incoming)
            assert _entry_key(committed) == _entry_key(expected), (
                "committed rows differ from prefix"
            )
            _assert_built_frozen(committed, incoming)
        assert _entry_key(entries) == before
    assert covered == {"empty", "whole", "none", "partial", "unnamed", "unsorted"}


def test_committed_filter_keeps_the_eager_shapes_and_error_outcomes(sources):
    lender = SimpleNamespace(
        _layout=SimpleNamespace(tokens_per_block=4, num_layer_groups=1), _named_groups=[True]
    )

    def eager(kv, entries):
        ordinals, slots = entries.block_ordinals[0], entries.device_pages[0]
        keep = (ordinals < kv.committed[0] // 4) & lender._named_groups[0]
        return _entry_key(sources.entries([0], [ordinals[keep]], [slots[keep]]))

    shapes = ((), (0,), (1,), (3,), (1, 3), (3, 1), (2, 2), (0, 2))
    for shape in shapes:
        for slot_shape in shapes:
            for committed in (0, 4, 64):
                for named in (False, True):
                    lender._named_groups[0] = named
                    ordinals = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
                    slots = np.arange(int(np.prod(slot_shape)), dtype=np.int64).reshape(slot_shape)
                    entries = sources.entries([0], [ordinals], [slots])
                    kv = SimpleNamespace(committed=(committed,))
                    expected = _outcome(lambda: eager(kv, entries))
                    actual = _outcome(
                        lambda: _entry_key(sources._committed_entries(lender, kv, entries))
                    )
                    assert actual == expected, (shape, slot_shape, committed, named)


@pytest.mark.parametrize("device", [False, True], ids=["HOST", "DEVICE"])
def test_named_pipeline_reuses_frozen_metadata_and_preserves_old_views(
    sources, monkeypatch, device
):
    manager = SimpleNamespace(stale=[(0, 0), (0, 0)])
    lender = _metadata_lender(sources, layout([False, False], tpb=4), manager)
    identity = SimpleNamespace(
        namespace=bytes(range(16)),
        _groups=[np.frombuffer(struct.pack(">HHH", lg, 4, 2), dtype=np.uint8) for lg in range(2)],
    )
    emitted = {}

    def object_keys(lg, keys):
        result = sources.object_keys(identity, lg, keys)
        emitted[lg] = result
        return result

    lender._object_keys = object_keys
    observations = []
    immutable = sources.ns["_immutable"]

    def observe(array):
        result = immutable(array)
        observations.append((np.shares_memory(array, result), array.flags.writeable))
        return result

    monkeypatch.setitem(sources.ns, "_immutable", observe)
    kv = SimpleNamespace(
        pages=[np.arange(8, dtype=np.int64) + 20 + lg * 100 for lg in range(2)],
        committed=(32, 32),
    )
    snapshots = []
    for start, end in ((0, 16), (16, 32)):
        raw = sources._block_entries_of_range(
            lender, manager, kv, start, end, named=True, request_id=7
        )
        lendable = sources._lendable_block_entries(lender, manager, 32, raw)
        entries = sources._committed_entries(lender, kv, lendable)
        for blocks, slots, original_blocks, original_slots in zip(
            entries.block_ordinals, entries.device_pages, raw.block_ordinals, raw.device_pages
        ):
            assert np.shares_memory(blocks, original_blocks), "NAMED all-keep ordinals were copied"
            assert np.shares_memory(slots, original_slots), "NAMED all-keep slots were copied"
        entries.staging_slots = [np.arange(4, dtype=np.int64) for _ in range(2)]
        keys = [np.arange(128, dtype=np.uint8).reshape(4, 32) + lg for lg in range(2)]
        view = (
            sources._in_place_view(lender, entries, keys, 0, 0)
            if device
            else sources._build_view(lender, entries, keys, 0, 0)
        )
        for lg, rows in enumerate(view.groups):
            assert rows.page_ordinals.tolist() == list(range(start // 4, end // 4))
            assert np.shares_memory(rows.page_ordinals, entries.block_ordinals[lg])
            assert np.shares_memory(rows.object_keys, emitted[lg])
            expected_keys = b"".join(
                identity.namespace + row.tobytes() + identity._groups[lg].tobytes()
                for row in keys[lg]
            )
            assert rows.object_keys.tobytes() == expected_keys
            if device:
                assert np.shares_memory(rows.slots, entries.device_pages[lg])
                assert rows.slots.tolist() == kv.pages[lg][start // 4 : end // 4].tolist()
            else:
                region = lender._regions[lg]
                assert rows.addresses.tolist() == [
                    region.address + i * region.slot_bytes for i in range(4)
                ]
        _assert_frozen_view(view)
        assert observations and all(shared and not writable for shared, writable in observations), (
            "production metadata must enter DTOs already immutable without another copy"
        )
        snapshots.append((view, _view_key(view)))
        for values in entries.staging_slots + keys + kv.pages:
            values[:] = 0
        assert all(_view_key(old) == before for old, before in snapshots)
    assert all(_view_key(old) == before for old, before in snapshots)


def test_two_groups_and_two_spans_reuse_bounded_storage(sources):
    lender = _metadata_lender(sources, layout([False, False]), object())
    first_runs = [[(0, 2), (100, 104)], [(20, 23), (200, 205)]]
    next_runs = [[(0, 2), (104, 108)], [(23, 26), (205, 210)]]
    old_entries = sources.entries([0, 1], [], [])
    for lg, runs in enumerate(first_runs):
        values = lender._range_ordinals(7, lg, runs)
        assert values.tolist() == [b for begin, end in runs for b in range(begin, end)]
        old_entries.block_ordinals.append(values)
        old_entries.device_pages.append(sources.ns["_frozen_int64"](range(len(values))))
    old_view = sources._in_place_view(lender, old_entries, None, 0, 0)
    before = _view_key(old_view)
    storage = [_storage(values) for _, values in lender._ordinal_ranges[7].values()]
    for lg, runs in enumerate(next_runs):
        values = lender._range_ordinals(7, lg, runs)
        assert values.tolist() == [b for begin, end in runs for b in range(begin, end)]
    later = [_storage(values) for _, values in lender._ordinal_ranges[7].values()]
    assert all(any(np.shares_memory(a, b) for b in later) for a in storage), (
        "retained span storage must survive consecutive chunks across layer groups"
    )
    rows = sum(end - begin for runs in next_runs for begin, end in runs)
    assert sum(buf.nbytes for buf in later) <= 2 * rows * np.dtype(np.int64).itemsize
    for lg, runs in enumerate([[(0, 2), (108, 112)], [(26, 29), (210, 215)]]):
        values = lender._range_ordinals(7, lg, runs)
        assert values.tolist() == [b for begin, end in runs for b in range(begin, end)]
    assert _view_key(old_view) == before
    _assert_frozen_view(old_view)


def _slow_resume_without_windows(
    tpb: int,
    groups: tuple[int, ...],
    masks: list[np.ndarray],
    partial_ends: dict[tuple[int, int], int],
    computed: tuple[int, ...],
    cap: int | None,
) -> int:
    known = min((computed[lg] for lg in groups), default=max(computed, default=0))
    if not groups:
        return known if cap is None else cap
    if cap is None:
        cap = max(len(masks[lg]) for lg in groups) * tpb
    for end in range(cap, known, -1):
        if all(
            all(
                max(
                    computed[lg],
                    partial_ends[(lg, ordinal)]
                    if (lg, ordinal) in partial_ends
                    else (ordinal + 1) * tpb
                    if ordinal < len(masks[lg]) and masks[lg][ordinal]
                    else 0,
                )
                >= min(end, (ordinal + 1) * tpb)
                for ordinal in range(known // tpb, -(-end // tpb))
            )
            for lg in groups
        ):
            return end
    return known


def test_resume_bound_without_windows_matches_scalar_validity(sources):
    """The largest resume endpoint has all rows computed or delivered in every paged group."""
    current = _load("_fetch_ledger.py", "_FetchLedger._max_resume_position", sources.ns.copy())
    closed = _load(
        "_fetch_ledger.py", "_FetchLedger._max_resume_without_windows", sources.ns.copy()
    )
    known_of = _load("_fetch_ledger.py", "_FetchLedger._known", sources.ns.copy())
    rng = random.Random(126)
    seen = set()
    for _ in range(4000):
        tpb, count = rng.choice((4, 8, 32)), rng.randint(1, 3)
        recurrent = frozenset(lg for lg in range(count) if rng.random() < 0.25)
        layout = SimpleNamespace(tokens_per_block=tpb, windows=[None] * count, cp=(0, 1))
        paged = tuple(lg for lg in range(count) if lg not in recurrent)
        ledger = SimpleNamespace(_layout=layout, _recurrent_groups=recurrent, _paged_groups=paged)
        ledger._known = known_of.__get__(ledger)
        ledger._max_resume_without_windows = closed.__get__(ledger)
        blocks = rng.randint(0, 24)
        masks = []
        for _ in range(count):
            arrived = rng.choice((1.0, 1.0, 0.95, 0.7, 0.3))
            length = max(0, blocks + rng.randint(-3, 3))
            masks.append(np.array([rng.random() < arrived for _ in range(length)], dtype=bool))
        partial_ends = {}
        for _ in range(rng.choice((0, 0, 1, 2, 3))):
            lg, ordinal = rng.randrange(count), rng.randrange(blocks + 2)
            partial_ends[(lg, ordinal)] = ordinal * tpb + rng.randint(1, tpb - 1)
        delivered = SimpleNamespace(delivered_block_masks=masks, partial_ends=partial_ends)
        computed = tuple(
            rng.choice((0, 0, rng.randint(0, (blocks + 2) * tpb))) for _ in range(count)
        )
        cap = rng.choice((None, rng.randint(0, (blocks + 3) * tpb)))
        want = _slow_resume_without_windows(tpb, paged, masks, partial_ends, computed, cap)
        got = current(ledger, None, delivered, computed, cap)
        assert got == want, (tpb, sorted(recurrent), masks, partial_ends, computed, cap)
        known = ledger._known(computed)
        seen.add(
            "known"
            if got == known
            else "partial end"
            if got in partial_ends.values()
            else "computed"
            if got in computed
            else "block"
            if got % tpb == 0
            else "cap"
        )
    assert seen == {"known", "partial end", "computed", "block", "cap"}, seen


class Event:
    """A stand-in CUDA event, complete at once, named by who made it."""

    def __init__(self, name):
        self.name = name

    def query(self):
        return True

    def synchronize(self):
        raise AssertionError("a lease waited on its event")


class Connector:
    """A stand-in KV cache connector manager."""

    def __init__(self, capacity_only, loading, reserved):
        self.capacity_only = capacity_only
        self.loading = loading
        self.reserved = reserved

    def has_pending_load(self, request):
        return self.loading

    def get_prefix_reservation(self, request):
        return SimpleNamespace() if self.reserved else None


def runtime_stale_range(window, sinks, history):
    """The runtime's stale range of a ``window``-token window with ``sinks`` sink blocks at
    ``history``, in its 32-bit arithmetic, the division truncating toward zero."""
    covered = (history + TPB - 1) // TPB
    start = min(covered, sinks)
    if window is None:
        return start, start
    behind = history + 1 - window
    window_start = int(behind / TPB)
    return start, max(start, window_start)


class Manager:
    """A stand-in manager: its layer groups' pools, windows and sinks, its caches and two rules of
    its configuration; a commit it holds back for a cache runs at the unlock."""

    # No cache of this stand-in is on an allocation side stream.
    _on_allocation_stream = frozenset()

    def __init__(self, groups, connector, floor_follows_history, keeps_stale_pages, windows, sinks):
        self.impl = SimpleNamespace(get_life_cycle_pool_group_indices=lambda: list(range(groups)))
        self.kv_cache_map = {}
        self._held_commits = {}
        self.kv_connector_manager = connector
        self.floor_follows_history = floor_follows_history
        self.keeps_stale_pages = keeps_stale_pages
        self.tokens_per_block = TPB
        self.windows = windows
        self.sinks = sinks

    def _stale_block_range(self, layer_group, history):
        return runtime_stale_range(self.windows[layer_group], self.sinks[layer_group], history)

    def _commit_held_back(self, kv):
        """Commit what the locks held back: more tokens, the committed blocks on other pages."""
        commit = getattr(kv, "commit_at_unlock", None)
        if self._held_commits.pop(id(kv), None) is None or commit is None:
            return
        kv.commit_at_unlock = None
        kv.num_committed_tokens = commit
        for pages in kv.pages:
            whole = min(commit // TPB, len(pages))
            pages[:whole] = np.where(pages[:whole] >= 0, (pages[:whole] + 17) % SLOTS, -1)


class Cache:
    """A stand-in runtime cache: per layer group its blocks' pages, -1 where none and past its
    blocks, as the page-index buffer holds them; a state slot per group of recurrent state."""

    def __init__(self, rng, groups, history, committed, statuses, plain):
        self.num_blocks = max(rng.randint(0, 7), -(-history // TPB) if plain else 0)
        self.pages = []
        for _ in range(groups):
            missing = not plain and rng.random() < 0.25
            pages = [
                -1 if missing and rng.random() < 0.3 else rng.randrange(SLOTS)
                for _ in range(self.num_blocks)
            ]
            self.pages.append(np.array(pages + [-1] * 4, dtype=np.int32))
        self.state_slots = [rng.choice((3, 9) if plain else (-1, 3, 3, 9)) for _ in range(groups)]
        self.history_length = history
        self.Status = SimpleNamespace(OPEN=0, CLOSED=1)
        self.status = self.Status.OPEN
        self.num_committed_tokens = committed
        self.reuse_status = statuses
        self.is_active = plain or rng.random() < 0.9
        self.tokens_per_block = TPB
        self.enable_swa_scratch_reuse = not plain and rng.random() < 0.1
        self.commit_at_unlock = None

    def get_base_page_indices(self, layer_group, beam_id=0):
        view = self.pages[layer_group][:]
        view.flags.writeable = False
        return view

    def get_aggregated_page_indices(self, layer_group, beam_id=0, valid_only=False):
        pages = [int(p) for p in self.pages[layer_group][: self.num_blocks]]
        return [p for p in pages if p >= 0] if valid_only else pages

    def get_ssm_block_base_index(self, layer_group, beam_id=0):
        return self.state_slots[layer_group]


class Rules(_lender._ManagerLender):
    """A lender of a class of its own, which lends and returns through the rules alone."""


def layout(
    recurrent, num_slots=SLOTS, windows=None, sinks=None, read_windows=(), formula=False, tpb=TPB
):
    """One layer group per pool group, each with a device pool of ``num_slots`` slots or its pool
    group's count of them; ``formula``: stale ranges by the window formula, as an attach finds."""
    count = len(recurrent)
    if isinstance(num_slots, int):
        num_slots = (num_slots,) * count
    return SimpleNamespace(
        tokens_per_block=tpb,
        windows=tuple(windows or (None,) * count),
        sink_blocks=tuple(sinks or (0,) * count),
        read_windows=tuple(read_windows),
        stale_ranges_by_formula=formula,
        recurrent=tuple(recurrent),
        pool_groups=tuple(range(count)),
        pool_group_of=tuple(range(count)),
        num_layer_groups=count,
        named_groups=(True,) * count,
        cp=(0, 1),
        device_pools={
            g: (DevicePool(g, 0, (g + 1) << 30, 256, num_slots[g]),) for g in range(count)
        },
    )


def lender_over(cls, manager, lay, no_names=None):
    """A lender of class ``cls`` over ``manager`` with layout ``lay``, its identity and regions."""
    count = lay.num_layer_groups
    ident = Identity(b"scope", bytes(range(32)), [(lg,) for lg in range(count)], [(1, 0)] * count)
    ref = weakref.ref(manager)
    regions = tuple(_lender._device_regions(manager, lay, ident))
    return cls(ref, lay, ident, regions, SlotQueue({}), ref, no_names=no_names)


def plain_world(cls, lay, history, rng, floor=False, no_names=None):
    """A lender of class ``cls`` with layout ``lay`` over a manager of its windows and sinks, and
    an active cache of ``history`` tokens with every page, drawn from ``rng``."""
    count = lay.num_layer_groups
    manager = Manager(count, None, floor, False, lay.windows, lay.sink_blocks)
    lender = lender_over(cls, manager, lay, no_names)
    status = SimpleNamespace(complete=True, endpoint=0)
    kv = Cache(rng, count, history, 0, [status] * count, True)
    manager.kv_cache_map[RID] = kv
    return lender, manager, kv


def request_of(prompt, position=0, remaining=None, runs=None, served=0):
    """A request of ``prompt`` tokens whose context computed ``position``, ``remaining`` to go."""
    return SimpleNamespace(
        py_request_id=RID,
        prompt_len=prompt,
        total_input_len_cp=prompt,
        py_multimodal_data=runs,
        py_served_prefix_end=served,
        context_current_position=position,
        context_remaining_length=prompt - position if remaining is None else remaining,
    )


def world_of(lender, manager, kv, request, start, end, kind="read", ready_event=None):
    """A ``kind`` lend of ``[start, end)`` by ``lender`` over ``manager``, its cache and request."""
    return SimpleNamespace(lease=None, **locals())


def world(seed, cls):
    """A lend by a lender of class ``cls``, the same draw for the same seed: locks, fetch records,
    windows, sinks and pools short of slots now and then; half the draws plain, with every page."""
    rng = random.Random(seed)
    plain = rng.random() < 0.5
    count = rng.randint(1, 3)
    recurrent = [rng.random() < 0.3 for _ in range(count)]
    windowed = rng.random() < 0.4
    windows = [
        rng.choice((None, rng.randint(1, 6 * TPB))) if windowed and not state else None
        for state in recurrent
    ]
    sinks = [int(w is not None and rng.random() < 0.15) for w in windows]
    read_windows = ()
    if any(w is not None for w in windows) and rng.random() < 0.4:
        read_windows = tuple(w if w is None else rng.randint(1, w) for w in windows)
    connector = (
        None
        if rng.random() < 0.6
        else Connector(rng.random() < 0.3, rng.random() < 0.3, rng.random() < 0.2)
    )
    manager = Manager(
        count, connector, rng.random() < 0.5, rng.random() < 0.5, tuple(windows), tuple(sinks)
    )
    # A few pools hold fewer slots than the pages and state slots the cache draws.
    num_slots = [SLOTS if rng.random() < 0.9 else rng.randint(1, SLOTS - 1) for _ in range(count)]
    formula = cls is _lender._ManagerLender
    lay = layout(recurrent, num_slots, windows, sinks, read_windows, formula=formula)
    no_names = "this lender names nothing" if any(recurrent) else None
    lender = lender_over(cls, manager, lay, no_names)
    prompt = rng.randint(1, 6 * TPB)
    history = prompt if plain else rng.choice((prompt, prompt, rng.randint(0, 7 * TPB)))
    committed = rng.choice((0, 0, rng.randint(0, history)))
    statuses = [
        SimpleNamespace(complete=rng.random() < 0.8, endpoint=rng.randint(0, committed))
        for _ in range(count)
    ]
    kv = Cache(rng, count, history, committed, statuses, plain)
    manager.kv_cache_map[RID] = kv
    ready_event = rng.choice((None, None, Event("caller")))
    kind = rng.choice(("read", "write"))
    if plain:
        end = rng.choice((prompt, TPB * rng.randint(0, prompt // TPB)))
        start = TPB * rng.randint(0, end // TPB)
    else:
        start = TPB * rng.randint(0, 6)
        end = rng.choice((prompt, prompt, history, rng.randint(0, 7 * TPB)))
    runs = None
    if rng.random() < 0.3:
        # A run of multimodal tokens, often one the range's end falls inside.
        begin = rng.randint(max(0, end - TPB), max(0, end - 1)) if rng.random() < 0.7 else 0
        length = rng.randint(2, 2 * TPB)
        runs = {
            "mm_bidirectional_blocks": True,
            "multimodal_embed_mask_cumsum": np.cumsum(
                [1 if begin <= i < begin + length else 0 for i in range(prompt)]
            ),
        }
    position = rng.randint(0, prompt)
    served = rng.choice((0, 0, rng.randint(0, prompt)))
    remaining = rng.choice((prompt - position, rng.randint(0, prompt - position)))
    request = request_of(prompt, position, remaining, runs, served)
    for _ in range(0 if plain else rng.choice((0, 0, 1, 2))):
        lg = rng.randrange(count)
        blocks = sorted(rng.sample(range(8), rng.randint(1, 3)))
        mode = rng.choice((LockMode.READ, LockMode.WRITE))
        try:
            lender._locks.lock(kv, {lg: blocks}, mode, ready=Event("held"))
        except ValueError:
            pass
    ledger = lender._ledger
    if not plain and rng.random() < 0.2:
        fetch = _fetch_ledger._FetchRecord(
            rng.choice((0, TPB)), kv, marks_queued=rng.random() < 0.5, end=rng.randint(0, prompt)
        )
        ledger._latest_fetches[RID] = fetch
    if not plain and rng.random() < 0.15:
        masks = [np.array([rng.random() < 0.7 for _ in range(6)]) for _ in range(count)]
        ledger._deliveries[RID] = _fetch_ledger._Deliveries(kv, 0, masks)
    if not plain and rng.random() < 0.1:
        ledger._kept_prefixes[RID] = (kv, (rng.randint(0, history),) * count, history)
    if not plain and rng.random() < 0.05:
        ledger._mixed_tail_caches[RID] = kv
    return world_of(lender, manager, kv, request, start, end, kind, ready_event)


def mid_context_world(cls, window, history, position, end, prompt):
    """A read of ``[0, end)`` over a full layer group and one of a ``window``-token window, its
    cache of ``history`` tokens of ``prompt``, whose context computed ``position`` of them."""
    lay = layout([False, False], windows=(None, window), formula=cls is _lender._ManagerLender)
    lender, manager, kv = plain_world(cls, lay, history, random.Random(history))
    return world_of(lender, manager, kv, request_of(prompt, position), 0, end)


def windowed_receive_world(seed, cls):
    """A receive of every row over windows of the formula without sinks, now and then a full group
    beside them and narrower read windows, into a cache whose history lies up to the prompt."""
    rng = random.Random(seed)
    count = rng.randint(1, 3)
    windows = [rng.randint(1, 6 * TPB) for _ in range(count)]
    if count > 1 and rng.random() < 0.3:
        windows[rng.randrange(count)] = None
    read_windows = ()
    if rng.random() < 0.6:
        read_windows = tuple(w if w is None else rng.randint(1, w) for w in windows)
    floor = rng.random() < 0.5
    fast = cls is _lender._ManagerLender
    lay = layout([False] * count, SLOTS, windows, None, read_windows, formula=fast)
    prompt = rng.randint(1, 6 * TPB)
    history = rng.randint(0, prompt)
    lender, manager, kv = plain_world(cls, lay, history, rng, floor)
    kv.num_blocks = -(-prompt // TPB)
    kv.pages = [
        np.array([rng.randrange(SLOTS) for _ in range(kv.num_blocks)] + [-1] * 4, dtype=np.int32)
        for _ in range(count)
    ]
    end = rng.choice((prompt, prompt, TPB * rng.randint(0, prompt // TPB)))
    start = TPB * rng.randint(0, end // TPB)
    return world_of(lender, manager, kv, request_of(prompt), start, end, "write")


def hybrid_world(cls, kind, num_slots, page, state_slot):
    """A ``kind`` lend of a 4-block prompt over a group of recurrent state in slot ``state_slot``
    and one of attention on pages 0, 1, ``page`` and 3, their pools of ``num_slots`` slots."""
    prompt, no_names = 4 * TPB, "this lender names nothing"
    lay = layout([True, False], num_slots)
    lender, manager, kv = plain_world(cls, lay, prompt, random.Random(0), no_names=no_names)
    kv.num_blocks = 4
    kv.pages = [np.array([-1] * 8, dtype=np.int32), np.array([0, 1, page, 3] + [-1] * 4, np.int32)]
    kv.state_slots = [state_slot, -1]
    return world_of(lender, manager, kv, request_of(prompt, prompt), 0, prompt, kind)


def row_lender(blocks):
    """A lender of one layer group over a cache of ``blocks + 1`` blocks: lender, cache, pages."""
    history, rng = (blocks + 1) * TPB, random.Random(blocks)
    lender, _, kv = plain_world(_lender._ManagerLender, layout([False]), history, rng)
    return lender, kv, kv.pages[0]


def buffer_row(lender, row):
    """Row ``row`` of the lender's page-index buffer, its key half, as a lend reads a row."""
    group_words, row_words, row_blocks = lender._index_strides
    return lender._index_words, row * row_words, group_words, row_blocks


def array_key(array):
    """``array``'s dtype, shape, values and writability, and the type of the memory under it."""
    if array is None:
        return None
    root = type(_root(array)).__name__
    return array.dtype.str, array.shape, array.tolist(), bool(array.flags.writeable), root


def lease_key(lease):
    view, lock, fetch, entries = lease._view, lease._lock, lease._fetch, lease._block_entries
    return dict(
        cls=type(lease).__name__,
        kind=lease._kind,
        failure=lease.failure,
        first_poll_failure=lease._first_poll_failure,
        loading=lease._loading is not None,
        view=None
        if view is None
        else [
            (
                rows.layer_group_index,
                array_key(rows.page_ordinals),
                array_key(rows.slots),
                rows.region_indices,
                array_key(rows.object_keys),
            )
            for rows in view.groups
        ],
        lock=None
        if lock is None
        else (
            lock.mode.value,
            {lg: array_key(blocks) for lg, blocks in lock.blocks.items()},
            getattr(lock._ready, "name", lock._ready),
        ),
        entries=None
        if entries is None
        else (
            entries.layer_group_indices,
            [array_key(o) for o in entries.block_ordinals],
            [array_key(p) for p in entries.device_pages],
        ),
        fetch=None
        if fetch is None
        else (
            fetch.start,
            fetch.end,
            fetch.state_end,
            fetch.marks_queued,
            fetch.entries is entries,
        ),
    )


def state_key(w):
    ledger, locks = w.lender._ledger, w.lender._locks
    fetch = ledger._latest_fetches.get(RID)
    kept = ledger._kept_prefixes.get(RID)
    return dict(
        progress_round=w.lender._progress_round,
        latest_fetch=None
        if fetch is None
        else (fetch.start, fetch.end, fetch.state_end, fetch.marks_queued),
        kept_prefix=None if kept is None else (kept[1], kept[2]),
        deliveries=RID in ledger._deliveries,
        mixed_tail=RID in ledger._mixed_tail_caches,
        locks_version=locks.version,
        locks_holding=locks.holding,
        locks=[
            (lock.mode.value, {lg: blocks.tolist() for lg, blocks in lock.blocks.items()})
            for cache in locks._caches.values()
            for lock in cache.locks
        ],
    )


def outcome(w):
    lend = w.lender.lend_read if w.kind == "read" else w.lender.lend_write
    try:
        lease = lend(
            w.request, w.start, w.end, memory_kind=DEVICE, rows=ALL, ready_event=w.ready_event
        )
    except Exception as error:  # noqa: BLE001 - compare complete exception outcomes
        return ("raise", type(error).__name__, str(error)), state_key(w)
    w.lease = lease
    return ("lease", lease_key(lease)), state_key(w)


@pytest.fixture(autouse=True)
def stand_in_rules(monkeypatch):
    """The stand-in manager's own answers for the facade's configuration rules."""
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: manager.keeps_stale_pages)
    monkeypatch.setattr(
        _ranges, "_floor_follows_history", lambda manager: manager.floor_follows_history
    )


@pytest.fixture
def taken(monkeypatch):
    """Fast lends that gave a lease, by kind, and the whole-delivery bounds fast returns gave."""
    counts = {"read": 0, "write": 0, "bounds": 0}
    for owner, name, key in (
        (_lender._ManagerLender, "_read_every_row_fast", "read"),
        (_lender._ManagerLender, "_write_every_row_fast", "write"),
        (_fetch_ledger._FetchLedger, "_whole_delivery_bounds", "bounds"),
    ):

        def counted(self, *args, _taken=getattr(owner, name), _key=key, **kwargs):
            found = _taken(self, *args, **kwargs)
            counts[_key] += found is not None
            return found

        monkeypatch.setattr(owner, name, counted)
    return counts


def test_a_windowed_read_mid_context_lends_short_windows_as_the_rules_do(taken):
    """Mid-context reads over short windows are never lent fast and lend what the rules lend."""
    kept = 0
    prompt = 6 * TPB
    edges = sorted(
        {
            v
            for b in range(1, prompt // TPB + 1)
            for v in (b * TPB - 1, b * TPB, b * TPB + 1)
            if TPB <= v < prompt
        }
    )
    boundary_windows = {
        1,
        TPB - 1,
        TPB,
        TPB + 1,
        2 * TPB - 1,
        2 * TPB,
        2 * TPB + 1,
        prompt - TPB - 1,
        prompt - TPB,
        prompt - TPB + 1,
        prompt - 1,
    }
    windows = sorted(boundary_windows | set(random.Random(0).sample(range(1, prompt), 16)))
    for window in windows:
        for history in edges:
            for position in edges:
                for end in range(TPB, min(history, position) + 1, TPB):
                    fast = mid_context_world(
                        _lender._ManagerLender, window, history, position, end, prompt
                    )
                    rules = mid_context_world(Rules, window, history, position, end, prompt)
                    assert fast.lender._lends_fast()
                    got, want = outcome(fast), outcome(rules)
                    assert got == want, (window, history, position, end)
                    behind = history + 1 - window
                    kept += behind < 0 or behind // TPB < end // TPB
    assert kept >= 100, f"inconclusive: {kept} cases where the window keeps rows of the range"
    assert taken["read"] == 0, "a read mid-context with a short window lent fast"


@pytest.mark.parametrize("listed_blocks", [64, 3], ids=["pages_listed", "pages_viewed"])
def test_fast_lends_give_what_the_rules_give(monkeypatch, taken, listed_blocks):
    """On drawn worlds, a lend that may take a fast path gives what the rules alone give."""
    monkeypatch.setattr(_lender, "_LISTED_BLOCKS", listed_blocks)
    fell_back = past_the_slots = windowed = 0
    for seed in range(3000):
        fast, rules = world(seed, _lender._ManagerLender), world(seed, Rules)
        assert fast.lender._lends_fast() == fast.lender._fast_layout
        assert not rules.lender._lends_fast()
        before = taken["read"] + taken["write"]
        got, want = outcome(fast), outcome(rules)
        assert got == want, f"seed {seed}"
        took = taken["read"] + taken["write"] > before
        fell_back += not took
        windowed += took and any(w is not None for w in fast.manager.windows)
        past_the_slots += "slots of its device regions" in repr(want)
    assert min(taken["read"], taken["write"]) >= 400 and fell_back >= 800, (taken, fell_back)
    assert windowed >= 100, windowed
    assert past_the_slots >= 50, past_the_slots


# A lend's bounds the call refuses, or takes as another integer type, by case: (start, end) of a
# prompt of ``prompt`` tokens.
ODD_BOUNDS = {
    "unaligned_start": lambda prompt: (1, prompt),
    "start_past_end": lambda prompt: (TPB, 0),
    "negative_start": lambda prompt: (-TPB, prompt),
    "end_past_prompt": lambda prompt: (0, prompt + 1),
    "numpy_bounds": lambda prompt: (np.int64(0), np.int64(prompt)),
}


@pytest.mark.parametrize("case", sorted(ODD_BOUNDS))
@pytest.mark.parametrize("kind", ["read", "write"])
def test_a_lend_the_fast_entry_could_take_checks_its_bounds_as_the_rules_do(kind, case):
    """Odd bounds of a lend the fast entry could take give what the rules alone give."""
    compared = 0
    for seed in range(400):
        fast, rules = world(seed, _lender._ManagerLender), world(seed, Rules)
        if not fast.lender._lends_fast():
            continue
        for w in (fast, rules):
            w.kind = kind
            w.start, w.end = ODD_BOUNDS[case](w.request.prompt_len)
        assert outcome(fast) == outcome(rules), f"seed {seed}"
        compared += 1
    assert compared >= 100, compared


# How the returns' draws change the world between the lend and the return, with their weights.
PERTURBATIONS = (
    ("none", 40),
    ("shrink_reported", 6),
    ("shrink_reported_and_regrown", 6),
    ("shrink_unreported", 5),
    ("cache_replaced", 5),
    ("cache_freed", 5),
    ("cache_suspended", 4),
    ("history_moved", 6),
    ("commit_at_unlock", 6),
    ("lender_stopped", 3),
    ("earlier_deliveries", 5),
    ("fetch_replaced", 4),
)


def perturb(w, rng):
    """Change ``w``'s world as the draw says, the same way in both worlds of a seed; the name."""
    name = rng.choices([p for p, _ in PERTURBATIONS], [n for _, n in PERTURBATIONS])[0]
    kv, lender, ledger = w.kv, w.lender, w.lender._ledger
    if name == "shrink_reported":
        lender._on_shrink(RID, kv)
    elif name == "shrink_reported_and_regrown":
        if kv.num_blocks:
            kv.num_blocks -= 1
            lender._on_shrink(RID, kv)
            kv.num_blocks += 1
            # A block regrown takes a page of its own.
            for pages in kv.pages:
                if pages[kv.num_blocks - 1] >= 0:
                    pages[kv.num_blocks - 1] = (pages[kv.num_blocks - 1] + 29) % SLOTS
    elif name == "shrink_unreported":
        kv.num_blocks = max(0, kv.num_blocks - rng.randint(1, 3))
    elif name == "cache_replaced":
        status = SimpleNamespace(complete=True, endpoint=0)
        other = Cache(rng, len(kv.pages), kv.history_length, 0, [status] * len(kv.pages), True)
        w.manager.kv_cache_map[RID] = other
    elif name == "cache_freed":
        del w.manager.kv_cache_map[RID]
        lender._on_free(RID, kv)
    elif name == "cache_suspended":
        kv.is_active = False
    elif name == "history_moved":
        kv.history_length = rng.randint(0, 7 * TPB)
    elif name == "commit_at_unlock":
        kv.commit_at_unlock = rng.randint(kv.num_committed_tokens, 7 * TPB)
        w.manager._held_commits[id(kv)] = (kv, w.request)
    elif name == "lender_stopped":
        lender._stop_reason = "the lender stopped"
    elif name == "earlier_deliveries":
        masks = [np.array([rng.random() < 0.7 for _ in range(8)]) for _ in kv.pages]
        ledger._deliveries[RID] = _fetch_ledger._Deliveries(kv, 0, masks)
    elif name == "fetch_replaced":
        ledger._latest_fetches[RID] = _fetch_ledger._FetchRecord(
            0, kv, marks_queued=True, end=rng.randint(0, w.request.prompt_len)
        )
    return name


# What reads or changes the request's records between its return and the lender's next round,
# with weights: nothing, the round, or a reader of the deliveries a fast return holds until then.
AFTER_RETURN = (
    ("none", 40),
    ("progress", 10),
    ("records_read", 10),
    ("shrink_reported", 10),
    ("cache_freed", 10),
)


def after_return(w, rng):
    """Read or change ``w``'s records after the return as the draw says, the same way in both
    worlds of a seed; the name."""
    name = rng.choices([p for p, _ in AFTER_RETURN], [n for _, n in AFTER_RETURN])[0]
    kv, lender = w.kv, w.lender
    if name == "progress":
        lender.progress()
    elif name == "records_read":
        lender._ledger._deliveries_of(RID)
    elif name == "shrink_reported":
        if kv.num_blocks:
            kv.num_blocks -= 1
        lender._on_shrink(RID, kv)
    elif name == "cache_freed":
        if w.manager.kv_cache_map.get(RID) is kv:
            del w.manager.kv_cache_map[RID]
            lender._on_free(RID, kv)
    return name


def arrivals(view, rng):
    """What the draw says arrived of ``view``: every row, as the constant or as true masks, writable
    or read-only, some rows, or none."""
    how = rng.choice(("constant", "true", "true_read_only", "some", "some", "none"))
    if how == "constant":
        return ALL_ARRIVED
    masks = [np.full(len(rows), how != "none", dtype=bool) for rows in view.groups]
    if how == "some":
        masks = [np.array([rng.random() < 0.7 for _ in range(len(m))], dtype=bool) for m in masks]
    if how == "true_read_only":
        for mask in masks:
            mask.flags.writeable = False
    return masks


def label(w, obj):
    """``obj`` named by what it is in ``w``'s world, so the two worlds compare."""
    if obj is w.kv:
        return "cache"
    if obj is None:
        return None
    return type(obj).__name__


def records_key(w):
    """The ledger's records of the request, the request's shrink count and the locks."""
    ledger = w.lender._ledger
    fetch = ledger._latest_fetches.get(RID)
    delivered = ledger._deliveries.get(RID)
    kept = ledger._kept_prefixes.get(RID)
    return dict(
        latest=None
        if fetch is None
        else (
            label(w, fetch.kv),
            fetch.start,
            fetch.end,
            fetch.state_end,
            fetch.marks_queued,
            fetch.copy,
        ),
        deliveries=None
        if delivered is None
        else (
            label(w, delivered.kv),
            delivered.lowest_start,
            [mask.tolist() for mask in delivered.delivered_block_masks],
            delivered.state_end,
            delivered.end,
            sorted(delivered.partial_ends.items()),
            delivered.resume_bounds_memo,
        ),
        kept=None if kept is None else (label(w, kept[0]), kept[1], kept[2]),
        mixed=label(w, ledger._mixed_tail_caches.get(RID)),
        shrinks=w.lender._shrink_epochs.get(RID),
        state=state_key(w),
    )


def return_outcome(w, rng):
    """Poll ``w``'s lease, change the world as drawn, return it with what the draw says arrived,
    then query the bounds again: what each gave and the records left."""
    lease = w.lease
    view = lease.poll()
    if view is None:
        return ("no view", lease.failure)
    perturbation = perturb(w, rng)
    arrived = arrivals(view, rng)
    got = _outcome(lambda: w.lender.return_writes([(lease, arrived, True)]))
    after = after_return(w, rng)
    try:
        again = ("bounds", w.lender.poll_resume_bounds(w.request))
    except ValueError as error:
        again = ("raise", str(error))
    return perturbation, got, after, again, (lease._released, lease._marked), records_key(w)


def generic_bounds(w):
    """``w`` with its ledger giving a whole delivery's bounds through ``_ranges``."""
    w.lender._ledger._closed_form = False
    return w


# How a fast return's ledger gives a whole delivery's bounds: in closed form, as the layouts fast
# lends cover allow, or through ``_ranges``, as for any other layout.
BOUNDS_FORMS = {"closed_form": lambda w: w, "by_ranges": generic_bounds}


@pytest.mark.parametrize("form", sorted(BOUNDS_FORMS))
def test_fast_returns_give_what_the_rules_give(taken, form):
    """Returns after drawn changes to the world give the bounds and records the rules give."""
    compared = fast_taken = fell_back = windowed = committed = agreed = 0
    perturbed = {name: 0 for name, _ in PERTURBATIONS}
    afterwards = {name: 0 for name, _ in AFTER_RETURN}
    for seed in range(7000):
        fast, rules = BOUNDS_FORMS[form](world(seed, _lender._ManagerLender)), world(seed, Rules)
        got, want = outcome(fast), outcome(rules)
        assert got == want, f"seed {seed}"
        if fast.lease is None or fast.kind != "write":
            continue
        before = taken["bounds"]
        got = return_outcome(fast, random.Random(seed * 7919 + 1))
        want = return_outcome(rules, random.Random(seed * 7919 + 1))
        assert got == want, f"seed {seed}"
        if got[0] == "no view":
            continue
        compared += 1
        perturbed[got[0]] += 1
        afterwards[got[2]] += 1
        if got[2] == "none" and got[1][0] == "ok" and got[1][1][0] is not None:
            # The bounds a return gives are those the next query gives.
            assert got[3] == ("bounds", got[1][1][0]), f"seed {seed}"
            agreed += 1
        took = taken["bounds"] > before
        fast_taken += took
        fell_back += not took
        windowed += took and any(w is not None for w in fast.manager.windows)
        committed += got[0] == "commit_at_unlock"
    assert compared >= 1000 and fast_taken >= 300 and fell_back >= 300, (
        compared,
        fast_taken,
        fell_back,
    )
    assert windowed >= 30 and committed >= 5 and agreed >= 300, (windowed, committed, agreed)
    assert min(perturbed.values()) >= 10, perturbed
    assert min(afterwards.values()) >= 50, afterwards


def test_a_write_view_lends_no_region_slot_twice():
    """No write view over caches of distinct pages, fast or by the rules, lends a slot twice."""
    checked = 0
    for seed in range(2000):
        for cls in (_lender._ManagerLender, Rules):
            w = world(seed, cls)
            pages = [p[: w.kv.num_blocks] for p in w.kv.pages]
            if w.kind != "write" or any(
                len(set(p[p >= 0].tolist())) < (p >= 0).sum() for p in pages
            ):
                continue
            outcome(w)
            view = None if w.lease is None else w.lease._view
            if view is None or not view.num_rows:
                continue
            lent = [
                (region, slot)
                for rows in view.groups
                for slot in rows.slots.tolist()
                for region in rows.region_indices
            ]
            assert len(lent) == len(set(lent)), f"seed {seed}"
            checked += 1
    assert checked >= 300, checked


@pytest.mark.parametrize("form", sorted(BOUNDS_FORMS))
def test_a_fast_return_over_windows_gives_the_floor_the_rules_give(kit, taken, form):
    """Fast returns over windows with read slack give the floor and bounds the rules give."""
    compared = floored = left = 0
    for seed in range(3000):
        outcomes = []
        for cls in (_lender._ManagerLender, Rules):
            w = windowed_receive_world(seed, cls)
            if cls is _lender._ManagerLender:
                BOUNDS_FORMS[form](w)
            lease = w.lender.lend_write(w.request, w.start, w.end, **kit.DEVICE_ALL)
            view = lease.poll()
            if view is None:
                outcomes.append(("no view", lease.failure))
                continue
            before = taken["bounds"]
            got = w.lender.return_writes([(lease, ALL_ARRIVED, True)])
            delivered = w.lender._ledger._deliveries.get(RID)
            took = taken["bounds"] > before
            uncomputed = took and delivered is not None and delivered.resume_bounds_memo is None
            outcomes.append((got, w.lender.poll_resume_bounds(w.request), took, uncomputed))
        fast, rules = outcomes
        assert fast[:2] == rules[:2], f"seed {seed}"
        if fast[0] == "no view":
            continue
        compared += 1
        bounds = fast[0][0]
        floored += fast[2] and bounds is not None and bounds.min_position > w.kv.history_length
        left += fast[3]
    assert compared >= 1500, compared
    assert floored >= 50, floored
    if form == "closed_form":
        assert left >= 50, left


@pytest.mark.parametrize("replace", ["class", "instance", "ledger"])
@pytest.mark.parametrize("refusal", ["", "the replaced rule refuses"])
def test_fast_lends_use_canonical_range_rules(monkeypatch, kit, replace, refusal):
    """Fast lends use the range rules directly; a ledger subclass takes the general path."""
    lender, manager, kv = plain_world(
        _lender._ManagerLender, layout([False]), TPB, random.Random(4)
    )
    w = world_of(lender, manager, kv, request_of(TPB, position=TPB), 0, TPB)
    assert w.lender._lends_fast()
    if replace == "class":
        monkeypatch.setattr(_lender._ManagerLender, "_read_end_refusal", lambda *args: refusal)
        monkeypatch.setattr(_lender._ManagerLender, "_write_range_refusal", lambda *args: refusal)
    elif replace == "instance":
        w.lender._read_end_refusal = lambda *args: refusal
        w.lender._write_range_refusal = lambda *args: refusal
    else:
        w.lender._ledger = type("Ledger", (_fetch_ledger._FetchLedger,), {})(
            w.lender._layout, w.lender._ledger._copy_done
        )
    assert w.lender._lends_fast() == (replace != "ledger")
    if replace != "ledger":
        for lend in (w.lender.lend_read, w.lender.lend_write):
            lease = lend(w.request, 0, w.request.prompt_len, **kit.DEVICE_ALL)
            assert lease.failure == refusal, lease.failure
            lease.release()


def test_a_replaced_mark_rule_takes_effect_on_a_return():
    """A fast write returned through the general path counts only still held rows."""
    seen = []
    for seed in range(400):
        w = world(seed, _lender._ManagerLender)
        if not w.lender._lends_fast() or w.kind != "write":
            continue
        outcome(w)
        if w.lease is None or w.lease.poll() is None:
            continue
        w.lender._locked_block_masks = lambda kv, lease: [
            np.zeros(len(o), dtype=bool) for o in lease._block_entries.block_ordinals
        ]
        w.lender._fast_layout = False
        assert not w.lender._lends_fast()
        w.lender.return_writes([(w.lease, ALL_ARRIVED, True)])
        delivered = w.lender._ledger._deliveries.get(RID)
        seen.append(delivered is None or not any(m.any() for m in delivered.delivered_block_masks))
    assert len(seen) >= 50 and all(seen), (len(seen), seen.count(False))


def test_a_lender_s_own_state_leaves_its_attributes_free():
    """Lender state leaves the instance dict available, and subclasses take the general path."""
    w = world(4, _lender._ManagerLender)
    assert not vars(w.lender)
    w.lender.note = "set on the lender"
    assert w.lender._lends_fast()

    assert not world(4, Rules).lender._lends_fast()
    assert w.lender._lends_fast()


@pytest.mark.parametrize("source", ["cache", "row"])
@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("offset", [-1, 0, 1, 2])
@pytest.mark.parametrize(
    "boundary", [_lender._LISTED_BLOCKS, _lender._ORDINAL_BLOCKS], ids=["listed", "ordinals"]
)
def test_fast_rows_hold_each_block_around_the_shared_ordinals(first, offset, boundary, source):
    """Rows near the shared-ordinal and listed-page bounds hold each block's page, in own arrays."""
    last = boundary + offset
    lender, kv, pages = row_lender(last)
    row = None
    if source == "row":
        # The cache's row of a page-index buffer between others: its pages, then the bad index.
        rows = np.full((1, 3, 2, len(pages) + 4), -1, dtype=np.int32)
        rows[0, 1, 0, : len(pages)] = pages
        lender._read_index_rows(rows)
        row = buffer_row(lender, 1)
    entries, view, blocks = lender._every_row(kv, RID, first * TPB, last * TPB, row=row)
    assert entries.block_ordinals[0].tolist() == list(range(first, last))
    assert entries.device_pages[0].tolist() == kv.pages[0][first:last].tolist()
    assert entries.spans == [(first, last)]
    assert blocks == {0: entries.block_ordinals[0]}
    (rows,) = view.groups
    assert rows.page_ordinals.tolist() == list(range(first, last))
    assert rows.slots.tolist() == entries.device_pages[0].tolist()
    assert rows.page_ordinals is not entries.block_ordinals[0]
    assert rows.slots is not entries.device_pages[0]


@pytest.mark.parametrize("blocks", [3, _lender._LISTED_BLOCKS + 3], ids=["listed", "viewed"])
def test_a_range_past_the_end_of_its_row_is_not_read_from_the_buffer(blocks):
    """A buffer row a block shorter than the range lends none of it; the shorter range, all."""
    lender, kv, pages = row_lender(blocks)
    # Every row and half of a row holds the cache's pages of its first ``blocks - 1`` blocks.
    rows = np.empty((1, 3, 2, blocks - 1), dtype=np.int32)
    rows[...] = pages[: blocks - 1]
    lender._read_index_rows(rows)
    shorter = lender._every_row(kv, RID, 0, (blocks - 1) * TPB, row=buffer_row(lender, 1))
    assert shorter is not None, "inconclusive: the row lends nothing"
    assert shorter[0].device_pages[0].tolist() == pages[: blocks - 1].tolist()
    assert lender._every_row(kv, RID, 0, blocks * TPB, row=buffer_row(lender, 1)) is None


@pytest.mark.parametrize("past", ["no_page", "at_the_slot_count"])
@pytest.mark.parametrize("blocks", [3, _lender._LISTED_BLOCKS + 3], ids=["listed", "viewed"])
def test_a_row_with_a_page_outside_the_pool_lends_nothing_from_the_buffer(blocks, past):
    """A buffer row with no page, or a page at the slot count, for one block lends nothing."""
    lender, kv, pages = row_lender(blocks)
    rows = np.full((1, 3, 2, len(pages)), -1, dtype=np.int32)
    rows[0, 1, 0] = pages
    lender._read_index_rows(rows)
    entries = lender._every_row(kv, RID, 0, blocks * TPB, row=buffer_row(lender, 1))[0]
    assert entries.device_pages[0].tolist() == pages[:blocks].tolist()
    ((_, _, limit, _, _),) = lender._fast_groups
    rows[0, 1, 0, blocks // 2] = -1 if past == "no_page" else limit
    assert lender._every_row(kv, RID, 0, blocks * TPB, row=buffer_row(lender, 1)) is None


def test_the_views_of_the_shared_ordinals_stay_bounded(monkeypatch):
    """At most ``_ORDINAL_VIEWS_KEPT`` views of the shared ordinals stay, each of its own range."""
    monkeypatch.setattr(_lender, "_ordinal_views", {})
    monkeypatch.setattr(_lender, "_ORDINAL_VIEWS_KEPT", 4)
    lender, kv, _ = row_lender(12)
    emptied = 0
    for first in range(3):
        for last in range(first + 1, 12):
            held = len(_lender._ordinal_views)
            entries = lender._every_row(kv, RID, first * TPB, last * TPB)[0]
            assert entries.block_ordinals[0].tolist() == list(range(first, last))
            emptied += len(_lender._ordinal_views) <= held
            assert len(_lender._ordinal_views) <= 4
    assert emptied >= 5, emptied


def view_arrays(view):
    """The arrays of ``view``'s rows."""
    return [
        array
        for rows in view.groups
        for array in (rows.page_ordinals, rows.slots, rows.addresses, rows.object_keys)
        if array is not None
    ]


def lender_arrays(lease):
    """The arrays the lender keeps for the lease: its entries' and its lock's."""
    entries = lease._block_entries
    held = [*entries.block_ordinals, *entries.device_pages, *entries.staging_slots]
    return held + ([] if lease._lock is None else list(lease._lock.blocks.values()))


@pytest.mark.parametrize("listed_blocks", [64, 3], ids=["pages_listed", "pages_viewed"])
@pytest.mark.parametrize("cls", [_lender._ManagerLender, Rules], ids=["fast", "rules"])
def test_each_view_holds_array_objects_of_its_own(monkeypatch, kit, taken, listed_blocks, cls):
    """Views of two reads of the same rows share no array object with each other or the lender."""
    monkeypatch.setattr(_lender, "_LISTED_BLOCKS", listed_blocks)
    compared = 0
    for seed in range(600):
        w = world(seed, cls)
        if w.kind != "read" or not 0 <= w.start <= w.end:
            continue
        leases = [w.lender.lend_read(w.request, w.start, w.end, **kit.DEVICE_ALL) for _ in range(2)]
        if all(lease._view is not None for lease in leases):
            first, second = (view_arrays(lease._view) for lease in leases)
            assert len({id(a) for a in first + second}) == len(first) + len(second), f"seed {seed}"
            kept = {id(a) for lease in leases for a in lender_arrays(lease)}
            assert not kept & {id(a) for a in first + second}, f"seed {seed}"
            compared += 1
        for lease in leases:
            lease.release()
    assert compared >= 100, compared
    assert taken["read"] >= (100 if cls is _lender._ManagerLender else 0), taken


@pytest.mark.parametrize("listed_blocks", [64, 3], ids=["pages_listed", "pages_viewed"])
@pytest.mark.parametrize("kind", ["read", "write"])
@pytest.mark.parametrize("past", ["page", "state_slot"])
def test_a_slot_at_the_limit_leaves_the_lend_to_the_rules(
    monkeypatch, taken, listed_blocks, kind, past
):
    """A page or state slot at its group's device slot limit leaves the lend to the rules."""
    monkeypatch.setattr(_lender, "_LISTED_BLOCKS", listed_blocks)
    slots_per_group = (4, 6)  # the state group's pool, then the attention group's
    limit = slots_per_group[1 if past == "page" else 0]
    for value in (limit - 1, limit, limit + 1):
        page, slot = (value, 2) if past == "page" else (2, value)
        fast, rules = (
            hybrid_world(cls, kind, slots_per_group, page, slot)
            for cls in (_lender._ManagerLender, Rules)
        )
        before = taken[kind]
        got, want = outcome(fast), outcome(rules)
        assert got == want
        assert taken[kind] == before + (value < limit)
        (shape, lease), state = got
        assert shape == "lease"
        if value < limit:
            assert lease["failure"] is None and lease["first_poll_failure"] is None
            assert [view[2] for view in lease["view"]] == [
                ("<i8", (1,), [slot], False, "bytes"),
                ("<i8", (4,), [0, 1, page, 3], False, "bytes"),
            ]
            continue
        lg = 1 if past == "page" else 0
        reason = f"layer group {lg}: blocks [{2 if past == 'page' else 0}] have pages past the "
        reason += f"{limit} slots of its device regions"
        if kind == "read":
            assert lease["failure"] is not None and reason in lease["failure"], lease["failure"]
            assert lease["lock"] is None and not state["locks"]
        else:
            assert lease["failure"] is None and reason in lease["first_poll_failure"]
            assert lease["lock"] is None and not state["locks"]


def test_a_next_fetch_records_what_the_settled_check_and_the_record_give():
    """Recording the next fetch in one call leaves what the settled check and the record leave."""
    refused = kept_past_start = 0
    for seed in range(2000):
        outcomes = []
        for together in (True, False):
            w = world(seed, _lender._ManagerLender)
            ledger, kv = w.lender._ledger, w.kv
            if seed % 3 == 0 and RID in ledger._latest_fetches:
                ledger._latest_fetches[RID].kv = object()
            history = _manager.cache_state(kv).history
            committed = _manager.committed_by_group(kv, len(w.lender._group_regions))
            state_end = w.end if w.lender._recurrent_groups else None
            before = ledger._kept_prefixes.get(RID)
            if together:
                fetch = ledger._record_next_fetch(
                    RID, kv, w.start, history, committed, state_end=state_end, end=w.end
                )
            elif ledger._unsettled_fetch_refusal(RID, kv) is not None:
                fetch = None
            else:
                kept = ledger._prefix_kept_by_fetch(RID, kv, w.start)
                fetch = ledger._record_fetch(
                    RID, kv, w.start, kept, history, committed, state_end=state_end, end=w.end
                )
            latest = ledger._latest_fetches.get(RID)
            kept_past_start += (
                together and w.start > 0 and ledger._kept_prefixes.get(RID) is not before
            )
            outcomes.append(
                (
                    fetch is None,
                    None if latest is None else (latest.start, latest.end, latest.state_end),
                    latest is fetch,
                    ledger._kept_prefixes.get(RID, (None,))[1:],
                )
            )
        assert outcomes[0] == outcomes[1], f"seed {seed}"
        refused += outcomes[0][0]
    assert refused >= 20 and kept_past_start >= 20, (refused, kept_past_start)


@pytest.mark.parametrize("sinks", [0, 1, 3])
def test_the_window_formula_answers_as_the_stand_in_runtime_and_the_last_keeping_history(sinks):
    """The window formula gives the runtime's stale ranges and the searched last keeping history."""
    for window in (1, 2, TPB - 1, TPB, TPB + 1, 3 * TPB + 5, 64 * TPB):
        manager = Manager(1, None, False, False, (window,), (sinks,))
        searched = layout([False], windows=(window,), sinks=(sinks,), formula=False)
        formula = layout([False], windows=(window,), sinks=(sinks,), formula=True)
        for history in range(0, 6 * window + 8 * TPB):
            assert _ranges._formula_stale_range(window, sinks, TPB, history) == (
                runtime_stale_range(window, sinks, history)
            ), (window, sinks, history)
        for block in range(sinks, sinks + 12):
            assert _ranges._last_history_keeping(manager, formula, 0, block) == (
                _ranges._last_history_keeping(manager, searched, 0, block)
            ), (window, sinks, block)
    huge = _manager._MAX_TOKENS - 3 * TPB
    manager = Manager(1, None, False, False, (huge,), (0,))
    for block in (0, 1, 5):
        assert _ranges._last_history_keeping(
            manager, layout([False], windows=(huge,), formula=True), 0, block
        ) == _ranges._last_history_keeping(
            manager, layout([False], windows=(huge,), formula=False), 0, block
        )


def test_an_attach_probe_finds_a_manager_off_the_formula():
    """The attach probe accepts a manager on the window formula and refuses one a block off."""
    lay = layout([False, False], windows=(3 * TPB + 2, None), formula=False)
    manager = Manager(2, None, False, False, lay.windows, (0, 0))
    assert _ranges._stale_ranges_follow_formula(manager, lay)
    honest = manager._stale_block_range

    def off_by_one(lg, history):
        beg, end = honest(lg, history)
        return (beg, end + 1) if history == 3 * TPB + 2 + TPB - 1 else (beg, end)

    manager._stale_block_range = off_by_one
    assert not _ranges._stale_ranges_follow_formula(manager, lay)


def test_a_fast_lend_builds_its_rows_as_the_unchecked_constructor_does():
    """A fast lend's rows are what ``LentLayerGroupRows._unchecked`` makes of the same values."""
    ordinals, slots = np.arange(3, dtype=np.int64), np.arange(5, 8, dtype=np.int64)
    fast = _lender._lent_rows(2, ordinals, slots, (4, 5))
    unchecked = LentLayerGroupRows._unchecked(2, ordinals, slots=slots, region_indices=(4, 5))
    assert type(fast) is type(unchecked)
    assert vars(fast).keys() == vars(unchecked).keys()
    assert all(vars(fast)[name] is vars(unchecked)[name] for name in vars(unchecked))
