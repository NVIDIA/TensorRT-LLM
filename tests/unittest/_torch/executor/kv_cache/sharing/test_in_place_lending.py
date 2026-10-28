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
"""Leases in place over real KV cache managers (``gpu``) and over stand-ins (``cpu_only``)."""

import random
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    LeaseView,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    RowSelection,
    StagingOptions,
    _lender,
    _manager,
    _ranges,
    _types,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import DevicePool, derive_layout
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._page_locks import LockMode
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
PROMPT_LEN = len(PROMPT)
OTHER_PROMPT = list(range(5000, 5097))
LONG_PROMPT = list(range(7000, 7000 + 6 * TPB))  # six whole blocks
LONG = 10 * TPB
END = 96
BLOCKS = END // TPB
SOURCE, TARGET, OTHER = 1, 2, 3
RID = 7  # the stand-ins' request
DEVICE, HOST = MemoryKind.DEVICE, MemoryKind.HOST
NAMED, ALL = RowSelection.NAMED, RowSelection.ALL
STALE, FRESH = 0x3C, 0xA7


# -- stand-ins ----------------------------------------------------------------------------------


class Event:
    """A stand-in CUDA event: complete once ``done`` is set; waiting on it fails the test."""

    def __init__(self, done=False):
        self.done = done
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.done

    def synchronize(self):
        raise AssertionError("a lease waited on its event")


class Connector:
    """A stand-in KV cache connector manager, loading into the requests in ``loading`` and holding
    a prefix reservation for those in ``reserved``."""

    def __init__(self, capacity_only=False):
        self.capacity_only = capacity_only
        self.loading = set()
        self.reserved = set()

    def has_pending_load(self, request):
        return request.py_request_id in self.loading

    def get_prefix_reservation(self, request):
        return SimpleNamespace() if request.py_request_id in self.reserved else None


class Manager:
    """A stand-in manager: a layer group per window, ``None`` for full attention, on pool groups 0
    and 1 in turn, with the runtime's stale ranges, and the requests' caches by request."""

    # No cache of this stand-in is on an allocation side stream.
    _on_allocation_stream = frozenset()

    def _commit_held_back(self, kv):
        """No commit is held back here."""

    def __init__(self, windows=(None,) * 3, sinks=None, connector=None):
        sinks = sinks or (0,) * len(windows)
        cycles = [AttnLifeCycle.make(w, s * TPB, TPB) for w, s in zip(windows, sinks)]
        self.tokens_per_block = TPB
        self.impl = SimpleNamespace(
            get_life_cycle_pool_group_indices=lambda: [lg % 2 for lg in range(len(windows))]
        )
        self.kv_cache_map = {}
        self.kv_connector_manager = connector
        self.enable_block_reuse = False
        self._can_publish_block_reuse = False

        def stale_block_range(lg, history):
            stale = cycles[lg].get_stale_range(history, TPB)
            return stale.beg, stale.end

        self._stale_block_range = stale_block_range


def layout(windows=(None,) * 3, sinks=None):
    """The layout of ``Manager(windows, sinks)``: pool group 0 of two pools, 1 of one, ten slots
    each."""
    count = len(windows)
    pool_group_of = tuple(lg % 2 for lg in range(count))
    return SimpleNamespace(
        tokens_per_block=TPB,
        windows=tuple(windows),
        sink_blocks=tuple(sinks or (0,) * count),
        recurrent=(False,) * count,
        pool_groups=tuple(sorted(set(pool_group_of))),
        pool_group_of=pool_group_of,
        num_layer_groups=count,
        named_groups=(True,) * count,
        device_pools={
            0: (DevicePool(0, 0, 1 << 30, 256, 10), DevicePool(0, 1, 2 << 30, 128, 10)),
            1: (DevicePool(1, 0, 3 << 30, 512, 10),),
        },
    )


def identity(scope=b"scope", count=3):
    return Identity(scope, bytes(range(32)), [(lg,) for lg in range(count)], [(1, 0)] * count)


def lender_over(manager, lay=None, no_staging=None):
    """A lender over ``manager`` and ``lay``, ``layout()`` if not given: a staging region per pool
    group, none where ``no_staging`` says why, then the device regions an attach lists."""
    lay = layout() if lay is None else lay
    ident = identity(count=lay.num_layer_groups)
    regions = []
    for g in lay.pool_groups if no_staging is None else ():
        groups = tuple(lg for lg, pg in enumerate(lay.pool_group_of) if pg == g)
        regions.append(
            MemoryRegion(f"staging{g}", 4096 * (g + 1), 4096, 1024, 4, groups, MemoryKind.HOST)
        )
    regions += _lender._device_regions(manager, lay, ident)
    ref = weakref.ref(manager)
    slots = SlotQueue({g: 4 for g in lay.pool_groups})
    return _lender._ManagerLender(
        ref, lay, ident, tuple(regions), slots, ref, no_staging=no_staging
    )


def cache(blocks=4, history=None, committed=0, pages=None):
    """A stand-in active cache of ``history`` tokens, ``committed`` by default, whose blocks lock
    ``pages``, ``0, 1, ...`` by default, in every layer group, its committed tokens counting in
    each."""
    pages = list(range(blocks)) if pages is None else list(pages)
    return SimpleNamespace(
        Status=SimpleNamespace(OPEN=0, CLOSED=1),
        status=0,
        num_committed_tokens=committed,
        reuse_status=[SimpleNamespace(complete=True, endpoint=committed)] * 4,
        history_length=committed if history is None else history,
        is_active=True,
        num_blocks=blocks,
        tokens_per_block=TPB,
        enable_swa_scratch_reuse=False,
        get_base_page_indices=lambda lg: pages,
        get_aggregated_page_indices=lambda lg, valid_only=False: pages,
    )


def entries(ordinals=(0, 1, 2), pages=(5, 6, 7)):
    """The same blocks on the same pages in each of the three layer groups."""
    o, p = np.array(ordinals, dtype=np.int64), np.array(pages, dtype=np.int64)
    return _lender._LentBlockEntries(
        [0, 1, 2], [o.copy() for _ in range(3)], [p.copy() for _ in range(3)]
    )


def request(computed=0, served=0, prompt=PROMPT_LEN, end=None):
    """Stand-in request ``RID`` of a ``prompt``-token prompt whose context on this worker runs to
    ``end``, the prompt's by default, ``computed`` tokens of it computed, ``served`` served."""
    return SimpleNamespace(
        py_request_id=RID,
        prompt_len=prompt,
        py_multimodal_data=None,
        py_served_prefix_end=served,
        context_current_position=computed,
        context_remaining_length=(prompt if end is None else end) - computed,
    )


def in_place(lender, manager, kv, kind, lent=None, fetch=None, end=END, **kwargs):
    """``_lend_in_place`` of ``request()`` over ``[0, end)``, ``entries()`` by default."""
    lent = entries() if lent is None else lent
    return lender._lend_in_place(
        manager, request(), kv, kind, lent, None, fetch, start=0, end=end, **kwargs
    )


def return_write(lender, req, lease, arrived):
    """``return_writes`` of the one entry ``(lease, arrived, True)``: its bounds, which
    ``poll_resume_bounds`` then gives ``req`` too."""
    (bounds,) = lender.return_writes([(lease, arrived, True)])
    assert lender.poll_resume_bounds(req) == bounds, bounds
    return bounds


def refused(lease, why):
    """``lease`` failed at the call for ``why`` and holds no lock; released."""
    assert lease.failure is not None and why in lease.failure, lease.failure
    assert lease._lock is None, "a lease failed at the call holds a lock"
    lease.release()


def inputs_refuse_nothing(monkeypatch, lender):
    """``lender`` with no refusal from a request's inputs: reuse keys, unnamed input, context
    outputs or a bidirectional split."""
    monkeypatch.setattr(lender, "_reuse_keys_for", lambda *args: None)
    monkeypatch.setattr(lender, "_unnamed_input_refusal", lambda request: None)
    monkeypatch.setattr(lender, "_returns_context_outputs", lambda request: False)
    monkeypatch.setattr(lender, "_bidirectional_split_refusal", lambda request, end: None)


@pytest.fixture
def recorded(monkeypatch):
    """The CUDA events made while the test runs: a lend makes none, as its readiness is its
    caller's event."""
    events = []

    class Made:
        def __init__(self, *args, **kwargs):
            events.append(self)

        def record(self, stream=None):
            pass

        def query(self):
            return True

    monkeypatch.setattr(torch.cuda, "Event", Made)
    return events


# -- real managers ------------------------------------------------------------------------------


def attach(mgr, fetches=1):
    return attach_lender(
        mgr, scope=b"in-place", staging=StagingOptions(END, capacity_in_fetches=fetches)
    )


def kept_page_ordinals(window, end, tokens_per_block, cached=0):
    """Pages intersecting the next query's retained token interval after ``end`` tokens."""
    first_token = 0 if window is None else max(0, end + 1 - window)
    return list(
        range(
            max(cached // tokens_per_block, first_token // tokens_per_block),
            (end + tokens_per_block - 1) // tokens_per_block,
        )
    )


def device_tensor(address: int, nbytes: int) -> torch.Tensor:
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    return convert_to_torch_tensor(TensorWrapper(int(address), torch.uint8, shape=(int(nbytes),)))


def slot_parts(lender, run, slot):
    """``(address, nbytes)`` of ``slot`` in each region of ``run``, in region order."""
    regions = lender.regions
    return [
        (regions[r].address + int(slot) * regions[r].slot_bytes, regions[r].slot_bytes)
        for r in run.region_indices
    ]


def entry_bytes(lender, run, i) -> bytes:
    return b"".join(
        device_tensor(address, nbytes).cpu().numpy().tobytes()
        for address, nbytes in slot_parts(lender, run, run.slots[i])
    )


def write_through(lender, view, byte, masks=None) -> None:
    """Fill each entry of ``view`` that ``masks`` marks, every entry by default, with ``byte``
    through its addresses, on a stream of its own."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for run, mask in zip(view.groups, masks or [None] * len(view.groups)):
            for i in range(len(run)):
                if mask is None or mask[i]:
                    for address, nbytes in slot_parts(lender, run, run.slots[i]):
                        device_tensor(address, nbytes).fill_(byte)
    side.synchronize()


def device_regions(lender):
    return tuple(r for r in lender.regions if r.memory_kind is MemoryKind.DEVICE)


def rows_giving(patched, lender, mgr, requests, block, page):
    """Make the rows of the page-index buffer that the lender reads a write's pages from give
    ``page`` at ``block`` of each of ``requests``, in every layer group: a copy of the buffer, the
    manager's own rows unchanged."""
    rows = lender._index_rows.copy()
    for request in requests:
        rows[:, _manager.index_row(mgr, request.py_request_id), 0, block] = page
    # Set to themselves first, so the context's end puts back the manager's rows.
    for name in ("_index_rows", "_index_words"):
        patched.setattr(lender, name, getattr(lender, name))
    lender._read_index_rows(rows)


def pages_giving(patched, lender, mgr, requests, block, page, readers=("pages", "page_list")):
    """Make every way the lender reads the pages it lends give ``page`` at ``block`` of each of
    ``requests``: the cache's ``readers`` and its page-index row."""
    caches = [mgr.kv_cache_map[r.py_request_id] for r in requests]

    def giving(read):
        def pages(kv_cache, lg):
            found = np.array(read(kv_cache, lg), dtype=np.int64)
            if any(kv_cache is lent for lent in caches) and len(found) > block:
                found[block] = page
            return found

        return pages

    for name in readers:
        patched.setattr(_manager, name, giving(getattr(_manager, name)))
    rows_giving(patched, lender, mgr, requests, block, page)


def step(mgr, request, tokens):
    """One context step of ``tokens``, as the executor ends it: the position moves, then the
    manager updates the cache."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    request.context_chunk_size = tokens
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.update_context_resources(batch)


def computed_to(mgr, request, end):
    """``request``, admitted first where the manager holds no cache of it, with its context
    computed to ``end`` in one step, as the executor runs it."""
    if request.py_request_id not in mgr.kv_cache_map:
        assert mgr.prepare_context(request)
    tokens = end - request.context_current_position
    assert mgr.resize_context(request, tokens)
    step(mgr, request, tokens)
    assert mgr.kv_cache_map[request.py_request_id].history_length == end
    return request


def generate(mgr, request, token=7):
    """One generation step as the executor runs it: the scheduler grows the cache for a token, the
    sampler adds the token, then the manager updates the cache."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    assert mgr.try_allocate_generation(request)
    request.add_new_token(token, 0)
    batch = ScheduledRequests()
    batch.append_generation_request(request)
    mgr.update_resources(batch)


def receiving(kit, mgr, request_id, tokens, prepare=False, **attributes):
    """A generation-init request of ``tokens`` with ``attributes``, admitted to receive them, and
    the event a receive into it waits for: the admission's, or with ``prepare`` one recorded after
    the preparation a hybrid manager's executor makes, which resets its state slots."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    request = kit.make_request(
        request_id, tokens, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    )
    for name, value in attributes.items():
        setattr(request, name, value)
    assert mgr.prepare_disagg_gen_init(request)
    if not prepare:
        return request, mgr.take_disagg_receive_ready(request)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    mgr.prepare_resources(batch)
    # The preparation waits for the admission's event and resets the state slots on this stream.
    reset = torch.cuda.Event()
    reset.record(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()
    return request, reset


def two_beams(request_id, **kwargs):
    """A request of ``PROMPT`` searching two beams."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(PROMPT),
        sampling_config=SamplingConfig(2),
        is_streaming=False,
        **kwargs,
    )
    assert request.py_beam_width == 2, request.py_beam_width
    return request


def slots_by_block(view):
    """``(layer group, block ordinal) -> slot`` of each entry of ``view``."""
    return {
        (run.layer_group_index, int(ordinal)): int(slot)
        for run in view.groups
        for ordinal, slot in zip(run.page_ordinals.tolist(), run.slots.tolist())
    }


def holds(dev, lent, byte_of) -> bool:
    """Whether each lent page holds the bytes its block was given."""
    return all(
        dev.read(lg, slot) == bytes([byte_of(ordinal)]) * dev.page_bytes(lg)
        for (lg, ordinal), slot in lent.items()
    )


def state_group(mgr) -> int:
    (lg,) = [lg for lg, flag in enumerate(derive_layout(mgr).recurrent) if flag]
    return lg


def state_tensors(mgr, slot):
    """The state each Mamba layer's kernels read for ``slot``: SSM and convolution states."""
    return [t[slot] for t in list(mgr.all_ssm_states) + list(mgr.all_conv_states)]


def helix_rank(rank):
    from tensorrt_llm.mapping import CpType, Mapping

    return Mapping(world_size=2, rank=rank, cp_size=2, cp_config={"cp_type": CpType.HELIX})


def buffers(lender, view):
    """``(buffer, block ordinal, bytes)`` of each public buffer descriptor of each lent row,
    padding included."""
    layout = lender.page_layout
    for rows in view.groups:
        group = layout.layer_groups[rows.layer_group_index]
        for buffer in layout.buffers:
            if buffer.pool_group != group.pool_group or buffer.layer not in group.layers:
                continue
            index = layout.device_region_index[buffer.pool_group, buffer.pool_index]
            assert index in rows.region_indices, (index, rows.region_indices, buffer)
            region = lender.regions[index]
            for ordinal, slot in zip(rows.page_ordinals.tolist(), rows.slots.tolist()):
                address = region.address + slot * region.slot_bytes + buffer.slot_offset
                yield buffer, ordinal, device_tensor(address, buffer.size)


def fill_buffers(lender, view, stream, byte):
    with torch.cuda.stream(stream):
        for _, _, tensor in buffers(lender, view):
            tensor.fill_(byte)
    stream.synchronize()


def remote_tail_layout():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout

    return CSA2Layout((0, 2, 2, 1, 1, 1), (1, 3), (1, 3, 4), candidate_source_layer_id=3)


def csa2_manager(kit, model, reuse=False, **kwargs):
    """``with csa2_manager(...) as mgr:`` a small FP8 CSA2 manager of ``model`` for disaggregated
    serving, of 128-token blocks, ``kwargs`` over its defaults."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = dict(
        num_layers=len(model.compress_ratios),
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=1024,
        vocab_size=129280,
        dtype=DataType.BF16,
        mapping=Mapping(),
        layout=model,
        is_disagg=True,
    )
    config.update(kwargs)
    return kit.managed(
        lambda: CSA2CacheManager(
            KvCacheConfig(max_gpu_total_bytes=128 << 20, enable_block_reuse=reuse, dtype="fp8"),
            CacheType.SELFKONLY,
            **config,
        )
    )


# -- regions and the attach ---------------------------------------------------------------------


@pytest.mark.cpu_only
def test_an_attach_lists_one_device_region_per_pool_after_the_staging_regions():
    manager = Manager()
    regions = _lender._device_regions(manager, layout(), identity())
    assert [(r.address, r.slot_bytes, r.slot_count, r.nbytes) for r in regions] == [
        (1 << 30, 256, 10, 2560),
        (2 << 30, 128, 10, 1280),
        (3 << 30, 512, 10, 5120),
    ]
    assert [r.layer_group_indices for r in regions] == [(0, 2), (0, 2), (1,)]
    assert all(r.memory_kind is MemoryKind.DEVICE for r in regions)
    labels = [r.layout_label for r in regions]
    staging = identity().region_label((0, 2))
    assert labels[0] == f"{staging}:pool0" and labels[1] == f"{staging}:pool1"
    assert len(set(labels)) == 3 and staging not in labels
    other = _lender._device_regions(manager, layout(), identity(b"another scope"))
    assert [r.layout_label for r in other] == labels, "a label follows the layout, not the scope"
    # A manager naming no rebalance policy gets them too: a lender's pools are not rebalanced.
    assert _lender._device_regions(SimpleNamespace(), layout(), identity()) == regions
    lender = lender_over(manager)
    assert [r.memory_kind for r in lender.regions] == [HOST, HOST, DEVICE, DEVICE, DEVICE]
    assert lender._device_regions_of_pool_group == {0: (2, 3), 1: (4,)}


@gpu
def test_the_device_regions_are_the_manager_s_pools_after_the_staging_regions(kit, real_manager):
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:  # two pool groups
        lender = attach(mgr)
        assert lender.regions == kit.staging_regions(lender) + device_regions(lender)
        group_of = kit.pool_group_of(mgr)
        expected = []
        for pg in sorted(mgr.impl.pool_group_descs, key=lambda d: int(d.pool_group_index)):
            g = int(pg.pool_group_index)
            groups = tuple(lg for lg, pool_group in enumerate(group_of) if pool_group == g)
            for pool in sorted(pg.pools, key=lambda p: int(p.pool_index)):
                slots = int(pg.num_slots)
                expected.append(
                    (int(pool.base_address), slots * int(pool.slot_bytes), slots, groups)
                )
        got = [
            (r.address, r.nbytes, r.slot_count, r.layer_group_indices)
            for r in device_regions(lender)
        ]
        assert got == expected
        labels = [r.layout_label for r in lender.regions]
        assert len(set(labels)) == len(labels), "two regions share a label"


@gpu
def test_device_memory_is_lent_with_pool_rebalancing_configured(kit, real_manager):
    """The executor does not rebalance a lender's manager's pools, so the lender lends in place."""
    with real_manager(enable_kv_pool_rebalance=True) as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        regions = [(r.memory_kind, r.layout_label) for r in lender.regions]
        assert device_regions(lender), f"no device region among {regions}"
        lease = lender.lend_read(source, 0, END, **kit.DEVICE_NAMED)
        assert lease.failure is None, lease.failure
        assert kit.ready_view(lease, mgr).num_rows > 0
        lease.release()


def attach_checks(monkeypatch, plain, **attributes):
    """A stand-in manager with ``attributes`` whose layout is ``plain``, for the attach's checks."""
    count = plain.num_layer_groups
    manager = SimpleNamespace(
        mapping=SimpleNamespace(cp_size=1, pp_size=1, has_cp_helix=lambda: False),
        _model_has_attention_window=False,
        _model_has_sparse_buffers=False,
        _model_attention_window_pattern=(),
        num_layers=count,
        kv_cache_manager_py_config=SimpleNamespace(layers=[]),
        impl=SimpleNamespace(layer_grouping=[[] for _ in range(count)]),
        **attributes,
    )
    monkeypatch.setattr(_manager, "require_v2", lambda manager: None)
    monkeypatch.setattr(_lender, "derive_layout", lambda manager: plain)
    return manager


def checked_layout(sparse):
    """A layout of full-attention layer groups, those ``sparse`` marks holding sparse buffers."""
    count = len(sparse)
    return SimpleNamespace(
        cp=(0, 1),
        num_layer_groups=count,
        recurrent=(False,) * count,
        windows=(None,) * count,
        sink_blocks=(0,) * count,
        sparse=tuple(sparse),
    )


@pytest.mark.cpu_only
def test_a_lossy_cold_page_codec_names_nothing_and_passes_the_attach(monkeypatch):
    """A lossy cold-page codec keeps the lender from naming blocks, not from attaching."""
    plain = checked_layout((False,))
    manager = attach_checks(monkeypatch, plain, _has_lossy_cold_page_codec=True)
    monkeypatch.setattr(_manager, "commits_blocks", lambda manager: True)
    monkeypatch.setattr(_manager, "prompt_lookahead", lambda manager: 0)
    monkeypatch.setattr(_manager, "model_has_recurrent_state", lambda manager: False)
    assert _lender._checked_layout(manager) is plain
    reason = _lender._no_names(manager, plain)
    assert reason is not None and "lossy" in reason, reason
    manager._has_lossy_cold_page_codec = False
    assert _lender._no_names(manager, plain) is None


@pytest.mark.cpu_only
def test_an_attach_refuses_sparse_layer_groups_and_names_them(monkeypatch):
    manager = attach_checks(monkeypatch, checked_layout((False, True, True)))
    with pytest.raises(ValueError, match=r"layer groups \[1, 2\] hold sparse buffers"):
        _lender._checked_layout(manager)
    dense = checked_layout((False,) * 3)
    monkeypatch.setattr(_lender, "derive_layout", lambda manager: dense)
    assert _lender._checked_layout(manager) is dense


@gpu
def test_a_lossy_cold_page_codec_still_lends_a_computed_request_s_pages_in_place(kit, real_manager):
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import create_default_kv_cache_cold_page_codec

    provider = SimpleNamespace(
        config=ColdPageQuantizationCompressionConfig(),
        create_cold_page_codec=lambda *args, **kwargs: create_default_kv_cache_cold_page_codec(),
    )
    with real_manager(cold_page_codec_provider=provider) as mgr:
        assert mgr._has_lossy_cold_page_codec, "the manager took the codec for a lossless one"
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        lease = lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        kv = kit.kv(mgr, source)
        for run in view.groups:
            lg = run.layer_group_index
            assert run.object_keys is None
            assert run.page_ordinals.tolist() == list(range(BLOCKS + 1))
            assert run.slots.tolist() == kit.pages(kv, lg)[: BLOCKS + 1]
            for i, ordinal in enumerate(run.page_ordinals.tolist()):
                assert entry_bytes(lender, run, i) == kit.page(mgr, source, lg, ordinal)
        lease.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "no_staging", [None, "its connector loads"], ids=["stages", "stages_nothing"]
)
def test_device_memory_is_lent_by_name_or_not_and_host_memory_only_by_name(no_staging):
    lender = lender_over(Manager(), no_staging=no_staging)
    lender._check_memory_and_rows(DEVICE, NAMED)
    lender._check_memory_and_rows(DEVICE, ALL)
    with pytest.raises(ValueError, match="RowSelection.ALL is not lent.*staging relays named"):
        lender._check_memory_and_rows(HOST, ALL)
    if no_staging is None:
        lender._check_memory_and_rows(HOST, NAMED)
    else:
        with pytest.raises(ValueError, match="stages nothing, as its connector loads"):
            lender._check_memory_and_rows(HOST, NAMED)


# -- views --------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_an_in_place_view_places_each_row_by_its_page_in_its_pool_group_regions():
    lender = lender_over(Manager())
    lent = entries(ordinals=(0, 3), pages=(9, 4))
    view = lender._in_place_view(lent, None, 0, 0)
    assert [g.layer_group_index for g in view.groups] == [0, 1, 2]
    assert [g.region_indices for g in view.groups] == [(2, 3), (4,), (2, 3)]
    assert view.groups[0].region_indices is view.groups[2].region_indices, "one per pool group"
    for group in view.groups:
        assert isinstance(group, LentLayerGroupRows)
        assert group.slots.tolist() == [9, 4] and group.page_ordinals.tolist() == [0, 3]
        assert group.object_keys is None
    keys = [np.full((2, 32), lg, dtype=np.uint8) for lg in range(3)]
    named = lender._in_place_view(lent, keys, 0, 0)
    for lg, group in enumerate(named.groups):
        np.testing.assert_array_equal(group.object_keys, lender._identity.object_keys(lg, keys[lg]))
    # Entry i's bytes: slot ``slots[i]`` of each region, in region order.
    regions = lender.regions
    addresses = [
        regions[r].address + 9 * regions[r].slot_bytes for r in view.groups[0].region_indices
    ]
    assert addresses == [(1 << 30) + 9 * 256, (2 << 30) + 9 * 128]


@pytest.mark.cpu_only
def test_every_lend_of_a_pool_group_names_its_regions_by_one_tuple():
    """Rows of one pool group lent in place name one tuple of its device regions, and staged rows
    one tuple of its staging region, lend after lend."""
    lender = lender_over(Manager())
    lent = entries(ordinals=(0, 3), pages=(9, 4))
    first, second = (lender._in_place_view(lent, None, 0, 0) for _ in range(2))
    assert all(a.region_indices is b.region_indices for a, b in zip(first.groups, second.groups))
    lender._assign_staging_slots(lent, None)
    keys = [np.full((2, 32), lg, dtype=np.uint8) for lg in range(3)]
    staged = [lender._build_view(lent, keys, 0, 0) for _ in range(2)]
    assert all(a.region_indices is b.region_indices for a, b in zip(*(v.groups for v in staged)))
    assert [group.region_indices for group in staged[0].groups] == [(0,), (1,), (0,)]
    assert {lender.regions[r].memory_kind for r in (0, 1)} == {MemoryKind.HOST}


@gpu
def test_an_in_place_read_lends_the_request_s_own_pages_by_name(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        staged = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        staged_view = kit.ready_view(staged, mgr)
        lease = lender.lend_read(source, 0, END, **kit.DEVICE_NAMED)
        view = kit.ready_view(lease, mgr)
        kv = kit.kv(mgr, source)
        descs = {int(pg.pool_group_index): pg for pg in mgr.impl.pool_group_descs}
        expected_groups = list(range(len(kit.pool_group_of(mgr))))
        assert [group.layer_group_index for group in view.groups] == expected_groups
        assert [group.layer_group_index for group in staged_view.groups] == expected_groups
        for run, staged_run in zip(view.groups, staged_view.groups):
            assert run.page_ordinals.tolist() == [0, 1, 2]
            np.testing.assert_array_equal(run.page_ordinals, staged_run.page_ordinals)
            kinds = {lender.regions[r].memory_kind for r in run.region_indices}
            staged_kinds = {lender.regions[r].memory_kind for r in staged_run.region_indices}
            assert (kinds, staged_kinds) == ({MemoryKind.DEVICE}, {MemoryKind.HOST})
            np.testing.assert_array_equal(run.object_keys, staged_run.object_keys)
            lg = run.layer_group_index
            pages = kit.pages(kv, lg)
            assert run.slots.tolist() == [pages[o] for o in run.page_ordinals.tolist()]
            pools = sorted(descs[kit.pool_group_of(mgr)[lg]].pools, key=lambda p: int(p.pool_index))
            for i, ordinal in enumerate(run.page_ordinals.tolist()):
                manager_parts = [
                    (int(p.base_address) + pages[ordinal] * int(p.slot_bytes), int(p.slot_bytes))
                    for p in pools
                ]
                assert slot_parts(lender, run, run.slots[i]) == manager_parts
                assert entry_bytes(lender, run, i) == kit.page(mgr, source, lg, ordinal)
        lease.release()
        staged.release()


def checked_by_the_constructors(view):
    """``view`` rebuilt by the public constructors, which check what the lender's own construction
    of it does not: equal, else ``AssertionError``; values out of the rows' contract raise."""
    names = "layer_group_index page_ordinals object_keys slots region_indices".split()
    groups = tuple(LentLayerGroupRows(*(getattr(rows, n) for n in names)) for rows in view.groups)
    rebuilt = LeaseView(groups, view.start, view.end)
    assert repr(rebuilt) == repr(view)
    return rebuilt


@gpu
def test_a_read_in_place_leaves_out_blocks_behind_a_window_and_blocks_without_a_page(
    kit, real_manager, monkeypatch
):
    """A read of every row lends no block behind a window, none without a page, no negative slot."""
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        source = kit.published(mgr, SOURCE, LONG_PROMPT)
        lender = attach(mgr)
        kv = kit.kv(mgr, source)
        history, blocks = kv.history_length, len(LONG_PROMPT) // TPB
        stale = {lg: kit.stale_blocks(mgr, lg, history) for lg in range(kit.num_layer_groups(mgr))}
        assert any(begin < end for begin, end in stale.values()), f"inconclusive: {stale}"
        pageless = blocks - 2  # inside every window
        pages_giving(monkeypatch, lender, mgr, [source], pageless, -1)
        lease = lender.lend_read(source, 0, len(LONG_PROMPT), **kit.DEVICE_ALL)
        view = kit.ready_view(lease, mgr)
        for run in view.groups:
            lg = run.layer_group_index
            begin, end = stale[lg]
            expected = [b for b in range(blocks) if not begin <= b < end and b != pageless]
            assert run.page_ordinals.tolist() == expected, (lg, stale[lg])
            assert min(run.slots.tolist(), default=0) >= 0, run.slots.tolist()
            pages = kit.pages(kv, lg)
            assert run.slots.tolist() == [pages[b] for b in expected]
        checked_by_the_constructors(view)
        lease.release()


def view_arrays(view):
    """Every array of ``view``, by (layer group, field)."""
    return {
        (group.layer_group_index, name): getattr(group, name)
        for group in view.groups
        for name in ("page_ordinals", "object_keys", "slots")
        if getattr(group, name) is not None
    }


def check_snapshot(view, before, records):
    """``view``'s arrays read as ``before`` holds them, share no memory with ``records`` and can be
    neither written nor made writable."""
    arrays = view_arrays(view)
    assert arrays.keys() == before.keys(), (sorted(arrays), sorted(before))
    for key, array in arrays.items():
        np.testing.assert_array_equal(array, before[key], err_msg=f"{key} changed")
        shared = [i for i, record in enumerate(records) if np.shares_memory(array, record)]
        assert not shared, f"{key} shares memory with records {shared}"
        for write in (lambda: setattr(array.flags, "writeable", True), lambda: array.fill(0)):
            try:
                write()
            except ValueError:
                continue
            raise AssertionError(f"{key} took a write")


@pytest.mark.cpu_only
def test_an_in_place_view_is_a_snapshot_that_outlives_its_lease():
    """A release and a re-lend leave a view's arrays, copied from mutable records, as they were."""
    manager = Manager()
    lender = lender_over(manager)
    kv = cache()
    keys = [np.full((3, 32), lg, dtype=np.uint8) for lg in range(3)]
    lease = lender._lend_in_place(manager, request(), kv, "read", entries(), keys, start=0, end=END)
    view = lease.poll()
    assert view is not None, lease.failure
    assert {name for _, name in view_arrays(view)} == {"page_ordinals", "object_keys", "slots"}
    before = {key: array.copy() for key, array in view_arrays(view).items()}
    records = [*lease._block_entries.block_ordinals, *lease._block_entries.device_pages]
    lease.release()
    again = lender._lend_in_place(manager, request(), kv, "read", entries(), keys, start=0, end=END)
    assert again.poll() is not None
    again.release()
    check_snapshot(view, before, records)


@pytest.mark.cpu_only
def test_a_staging_view_is_a_snapshot_of_its_own():
    lender = lender_over(Manager())
    lent = entries()
    lender._assign_staging_slots(lent, None)
    keys = [np.full((3, 32), lg, dtype=np.uint8) for lg in range(3)]
    view = lender._build_view(lent, keys, 0, 0)
    assert {name for _, name in view_arrays(view)} == {"page_ordinals", "object_keys", "slots"}
    assert all(
        lender.regions[g.region_indices[0]].memory_kind is MemoryKind.HOST for g in view.groups
    )
    before = {key: array.copy() for key, array in view_arrays(view).items()}
    lender._build_view(lent, keys, 0, 0)
    check_snapshot(view, before, [*lent.block_ordinals, *lent.staging_slots])


@pytest.mark.cpu_only
@pytest.mark.parametrize("build", ["staging", "in_place"])
def test_views_of_the_same_frozen_entries_share_no_array_object(build):
    """Two views of the same frozen entries hold arrays of their own, not each other's or theirs."""
    lender = lender_over(Manager())
    # Entries as the lender keeps them, one group's rows the module's shared empty rows.
    kept = _lender._LentBlockEntries(
        [0, 1, 2],
        [_types._frozen_int64((0, 1, 2)), _lender._NO_ROWS, _types._frozen_int64((4,))],
        [_types._frozen_int64((5, 6, 7)), _lender._NO_ROWS, _types._frozen_int64((8,))],
    )
    keys = [np.full((len(o), 32), lg, dtype=np.uint8) for lg, o in enumerate(kept.block_ordinals)]
    if build == "staging":
        lender._assign_staging_slots(kept, None)
        views = [lender._build_view(kept, keys, 0, 0) for _ in range(2)]
    else:
        views = [lender._in_place_view(kept, keys, 0, 0) for _ in range(2)]
    held = {id(a) for a in (*kept.block_ordinals, *kept.device_pages, *kept.staging_slots)}
    lent = [id(array) for view in views for array in view_arrays(view).values()]
    assert len(set(lent)) == len(lent), "two views share an array"
    assert not held & set(lent), "a view holds an array of the entries"


@gpu
@pytest.mark.parametrize(
    "lend",
    ["DEVICE_ALL", "DEVICE_NAMED", "HOST_NAMED"],
    ids=["in_place_every_row", "in_place_named", "staged_named"],
)
def test_no_two_leases_share_an_array_and_a_reshaped_one_reaches_no_other(kit, real_manager, lend):
    """Leases share no array with each other or a lock; a reshaped one changes no other lease."""
    import warnings

    keywords = getattr(kit, lend)
    end = END if keywords["rows"] is NAMED else len(PROMPT)
    with real_manager() as one, real_manager() as two:
        lenders = tuple(attach(mgr, fetches=3) for mgr in (one, two))
        requests = [
            (one, lenders[0], kit.published(one, SOURCE, PROMPT)),
            (one, lenders[0], kit.published(one, OTHER, PROMPT)),
            (two, lenders[1], kit.published(two, SOURCE, PROMPT)),
        ]
        leases = [lender.lend_read(request, 0, end, **keywords) for _, lender, request in requests]
        views = [kit.ready_view(lease, mgr) for lease, (mgr, _, _) in zip(leases, requests)]
        lent = [id(array) for view in views for array in view_arrays(view).values()]
        assert len(set(lent)) == len(lent), "two leases share an array"
        locks = [getattr(lease, "_lock", None) for lease in leases]
        locked = {id(a) for lock in locks if lock is not None for a in lock.blocks.values()}
        assert not locked & set(lent), "a lock holds an array a view holds"
        lengths = [[len(group) for group in view.groups] for view in views]
        mine = views[0].groups[0].page_ordinals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mine.shape = (1, len(mine))
        assert [[len(group) for group in view.groups] for view in views[1:]] == lengths[1:]
        if keywords["memory_kind"] is DEVICE:
            assert lenders[0]._is_locked(kit.kv(one, requests[0][2]), BLOCKS - 1, BLOCKS)
        later = lenders[0].lend_read(requests[1][2], 0, end, **keywords)
        assert [len(group) for group in kit.ready_view(later, one).groups] == lengths[1]
        for lease in (*leases, later):
            lease.release()


# -- readiness ----------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_an_in_place_read_is_ready_once_its_event_completes_and_never_waits():
    manager = Manager()
    lender = lender_over(manager)
    kv = cache()
    ready = Event()
    assert not lender._is_locked(None)
    lease = in_place(lender, manager, kv, "read", ready_event=ready)
    assert isinstance(lease, _lender._InPlaceLease) and lease.failure is None
    assert lender._locks.is_locked(kv) and lender._is_locked(kv, 2, 3)
    assert lender._is_locked(None), "a lock went unseen by the query over every cache"
    assert lease.poll() is None and lease.poll() is None and ready.queries == 2
    ready.done = True
    view = lease.poll()
    assert view is not None and lease.poll() is view and ready.queries == 3
    with pytest.raises(RuntimeError, match="return_writes is for write leases"):
        return_write(lender, request(), lease, view.row_masks())
    assert lender._locks.is_locked(kv), "a refused return let the read go"
    lease.release()
    lease.release()
    assert not lender._locks.is_locked(kv) and not lender._is_locked(None)
    with pytest.raises(RuntimeError, match="after release"):
        lease.poll()


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["write", "read"])
@pytest.mark.parametrize("given", [False, True], ids=["no_event", "an_event"])
def test_a_lease_waits_for_the_event_its_caller_gives_and_for_none_without_one(
    recorded, kind, given
):
    """A lease is ready once its caller's event completed, at once without one."""
    manager = Manager()
    lender = lender_over(manager)
    ready = Event() if given else None
    lease = in_place(lender, manager, cache(), kind, ready_event=ready)
    if given:
        assert lease.poll() is None, f"a {kind} was ready before its caller's event completed"
        ready.done = True
    assert lease.poll() is not None, f"a {kind} was not ready"
    assert recorded == [], "the lender made an event of its own"
    lease.release()


@pytest.mark.cpu_only
def test_a_named_grow_waits_for_the_callers_pending_event_before_recycling_pages(kit, monkeypatch):
    """The grow's cache stream follows work still using pages it may recycle."""
    manager = Manager((None,))
    manager._stream = SimpleNamespace(device="cuda:0")
    cache_stream = object()
    manager._cache_stream = lambda request_id: cache_stream
    lender = lender_over(manager, layout((None,)))
    manager.kv_cache_map[RID] = cache(history=2 * TPB)
    order = []

    class Pending:
        def query(self):
            return False

        def wait(self, stream):
            assert stream is cache_stream
            order.append("wait")

    def grow(*args):
        order.append("grow")
        return False

    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda handle, device: handle)
    monkeypatch.setattr(_manager, "grow", grow)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    inputs_refuse_nothing(monkeypatch, lender)
    served = request(served=2 * TPB)
    lease = lender.lend_write(
        served, 2 * TPB, 3 * TPB, **{**kit.DEVICE_NAMED, "ready_event": Pending()}
    )
    assert lease.failure is not None and order == ["wait", "grow"]
    lease.release()

    order.clear()
    lease = lender.lend_write(
        served, 2 * TPB, 3 * TPB, **{**kit.DEVICE_NAMED, "ready_event": Event()}
    )
    assert "stream-waitable" in lease.failure and order == []
    lease.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure", ["fence", "fill"])
def test_a_failure_after_named_grow_keeps_unfetched_history_out_of_resume_bounds(
    kit, monkeypatch, failure
):
    """A failed fill or fence cannot make a grown, unfetched window look computed."""
    manager = Manager((2 * TPB,))
    manager._stream = SimpleNamespace(device="cuda:0")
    order = []

    class CacheStream:
        def synchronize(self):
            order.append("synchronize")

    cache_stream = CacheStream()
    manager._cache_stream = lambda request_id: cache_stream
    lender = lender_over(manager, layout((2 * TPB,)))
    manager.kv_cache_map[RID] = cache(history=2 * TPB)

    class FailingFence:
        def record(self, stream):
            assert stream is cache_stream
            order.append("record")
            if failure == "fence":
                raise RuntimeError("record failed")

    def grow(manager, request, kv, position, end):
        kv.history_length = position
        order.append("grow")
        if failure == "fill":
            raise RuntimeError("fill failed")
        return True

    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda handle, device: handle)
    monkeypatch.setattr(torch.cuda, "Event", FailingFence)
    monkeypatch.setattr(_manager, "grow", grow)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    inputs_refuse_nothing(monkeypatch, lender)
    with pytest.raises(RuntimeError, match="record failed|fill failed"):
        lender.lend_write(
            request(served=2 * TPB),
            2 * TPB,
            3 * TPB,
            **{**kit.DEVICE_NAMED, "ready_event": Event(done=True)},
        )
    assert order == (["grow", "record", "synchronize"] if failure == "fence" else ["grow"])
    assert RID in lender._ledger._kept_prefixes
    assert (RID in lender._ledger._latest_fetches) is (failure == "fill")
    assert lender._open_lease_count() == 0


@pytest.mark.cpu_only
def test_a_named_device_write_waits_for_the_cache_stream_after_grow(kit, monkeypatch):
    """The caller's pre-grow event alone cannot release a view whose grow queued more work."""
    manager = Manager((None,))
    manager._stream = SimpleNamespace(device="cuda:0")
    cache_stream = object()
    manager._cache_stream = lambda request_id: cache_stream
    lender = lender_over(manager, layout((None,)))
    manager.kv_cache_map[RID] = cache(history=2 * TPB)
    grown = []
    fences = []

    class Fence:
        def __init__(self):
            self.done = False
            fences.append(self)

        def record(self, stream):
            assert stream is cache_stream
            assert grown, "the fence was recorded before the grow"

        def query(self):
            return self.done

    def grow(manager, request, kv, position, end):
        grown.append((position, end))
        return True

    monkeypatch.setattr(torch.cuda, "Event", Fence)
    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda handle, device: handle)
    monkeypatch.setattr(_manager, "grow", grow)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    inputs_refuse_nothing(monkeypatch, lender)
    served = request(served=2 * TPB)
    lease = lender.lend_write(
        served, 2 * TPB, 3 * TPB, **{**kit.DEVICE_NAMED, "ready_event": Event(done=True)}
    )
    try:
        assert lease.failure is None, lease.failure
        assert grown == [(2 * TPB, 3 * TPB)] and len(fences) == 1
        assert lease.poll() is None, "the pre-grow event released a view before the grow completed"
        fences[0].done = True
        view = lease.poll()
        assert view is not None, lease.failure
        return_write(lender, served, lease, view.row_masks(True))
    finally:
        lease.release()


@gpu
def test_a_read_after_its_pages_went_to_host_and_back_waits_for_the_resume(kit, host_tier_manager):
    """A read given an event recorded after a resume waits for it and lends the pages held now."""
    with host_tier_manager() as mgr:
        kit.warm_tier_moves(mgr)
        lender = attach(mgr)
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        old = [p for p in kit.pages(kv, 0) if p >= 0]
        others = kit.Requests(mgr)
        try:
            mgr.suspend_request(source)
            assert others.allocate(kit.pool_pages(mgr), chunk=1)
            assert set(old) <= others.pages(), "inconclusive: its pages stayed on the GPU"
            others.free_all_but(set(old))
            assert mgr.resume_request(source)
            with kit.held_stream(mgr._stream, strict=False):
                ready = kit.stream_event(mgr)
                lease = lender.lend_read(source, 0, END, **{**kit.DEVICE_ALL, "ready_event": ready})
                assert lease.failure is None, lease.failure
                early = lease.poll()
            assert early is None, "a read went out before the work queued after the resume"
            view = kit.ready_view(lease, mgr)
            for run in view.groups:
                now = kit.pages(kv, run.layer_group_index)
                assert run.slots.tolist() == [now[b] for b in run.page_ordinals.tolist()]
                assert not set(run.slots.tolist()) & set(old), "lent the pages held before"
            lease.release()
        finally:
            others.free()


@gpu
def test_a_named_device_fetch_on_a_host_tier_waits_for_grow_after_its_caller_event(
    kit, host_tier_manager
):
    """A DEVICE view waits for the cache stream even when its caller's event finished first."""
    with host_tier_manager() as mgr:
        kit.warm_tier_moves(mgr)
        lender = attach(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        ready = kit.stream_event(mgr)
        ready.synchronize()
        with kit.held_stream(mgr._stream, strict=False):
            lease = lender.lend_write(target, 0, END, **{**kit.DEVICE_NAMED, "ready_event": ready})
            assert lease.failure is None, lease.failure
            assert lease.poll() is None, "the view went out before the grow's stream work"
        view = kit.ready_view(lease, mgr)
        assert view.num_rows == BLOCKS
        return_write(lender, target, lease, view.row_masks(True))


@gpu
def test_a_recycled_page_is_lent_for_writing_only_after_the_work_queued_on_it(kit, real_manager):
    """A write onto freed pages goes out after the fill queued on them, and its bytes survive it."""
    # No host tier: a freed page goes straight back to the pool. A process's first cache close
    # waits for the whole device, the held stream included: close one before the hold.
    with real_manager(max_tokens=kit.POOL_TOKENS, host_cache_size=0, max_batch_size=16) as mgr:
        warm = kit.make_request(900, list(range(40_000, 40_000 + TPB)))
        assert mgr.prepare_context(warm) and mgr.resize_context(warm, TPB)
        mgr.free_resources(warm)
        torch.cuda.synchronize()
        lender = attach(mgr)
        dev = kit.DevicePages(mgr)  # built before the hold: building it waits for the device
        old = kit.make_request(OTHER, OTHER_PROMPT)
        assert mgr.prepare_context(old) and mgr.resize_context(old, 4 * TPB)
        old_pages = [p for p in kit.pages(kit.kv(mgr, old), 0) if p >= 0]
        others = kit.Requests(mgr, dev)
        # Every other page, in requests of a few blocks, then of one: only the old request's go free.
        others.allocate(kit.pool_pages(mgr))
        while others.allocate(1, chunk=1):
            pass
        with kit.held_stream(mgr._stream, strict=False):
            for slot in old_pages:
                dev.fill_async(0, slot, STALE)
            mgr.free_resources(old)
            target = kit.admitted(mgr, TARGET, PROMPT)  # all it takes from here, it takes freed
            ready = kit.stream_event(mgr)
            lease = lender.lend_write(target, 0, END, **{**kit.DEVICE_NAMED, "ready_event": ready})
            assert lease.failure is None, lease.failure
            lent = kit.pages(kit.kv(mgr, target), 0)[:BLOCKS]
            assert set(lent) <= set(old_pages), "inconclusive: the write took other pages"
            assert lease.poll() is None, "the view went out with work still queued on its pages"
        view = kit.ready_view(lease, mgr)
        write_through(lender, view, FRESH)
        mgr._stream.synchronize()
        page_bytes = dev.page_bytes(0)
        for ordinal in range(BLOCKS):
            assert kit.page(mgr, target, 0, ordinal) == bytes([FRESH]) * page_bytes, (
                "the work queued on a recycled page overwrote what was written through the view"
            )
        return_write(lender, target, lease, view.row_masks())
        others.free()


@gpu
def test_a_chunked_context_lends_each_range_once_the_step_that_computed_it_ran(kit, real_manager):
    """Each step's range, lent with the event after the step, goes out once the step ran."""
    with real_manager(enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        dev = kit.DevicePages(mgr)  # built before any hold: building it waits for the device
        request = kit.make_request(SOURCE, PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        kv = kit.kv(mgr, request)
        leases = []
        for block in range(BLOCKS):
            with kit.held_stream(mgr._stream, strict=False):
                for lg in range(kit.num_layer_groups(mgr)):
                    dev.fill_async(lg, kit.pages(kv, lg)[block], 0x10 + block)
                step(mgr, request, TPB)
                ready = kit.stream_event(mgr)
                lease = lender.lend_read(
                    request,
                    block * TPB,
                    (block + 1) * TPB,
                    **{**kit.DEVICE_ALL, "ready_event": ready},
                )
                assert lease.failure is None, lease.failure
                assert lease.poll() is None, f"block {block}'s view went out with its step queued"
                leases.append(lease)
        # Before the last step a read may neither end past what the steps computed nor inside a
        # block a later step writes: either fails at the call.
        for end, why in ((len(PROMPT), "tokens the cache holds"), (2 * TPB + 5, "whole block")):
            refused(lender.lend_read(request, 0, end, **kit.DEVICE_ALL), why)
        step(mgr, request, len(PROMPT) - BLOCKS * TPB)
        ready = kit.stream_event(mgr)
        last = lender.lend_read(
            request, 2 * TPB, len(PROMPT), **{**kit.DEVICE_ALL, "ready_event": ready}
        )
        assert sorted(slots_by_block(kit.ready_view(last, mgr))) == [
            (lg, ordinal)
            for lg in range(kit.num_layer_groups(mgr))
            for ordinal in (BLOCKS - 1, BLOCKS)
        ]
        last.release()
        for block, lease in enumerate(leases):
            lent = slots_by_block(kit.ready_view(lease, mgr))
            assert {ordinal for _, ordinal in lent} == {block}
            assert holds(dev, lent, lambda o: 0x10 + o), f"block {block} lost its step's bytes"
            lease.release()


@gpu
def test_a_read_after_a_commit_rebases_its_pages_waits_for_the_new_pages(kit, real_manager):
    """A read after a commit rebased its page waits for the writes to the new page."""
    with real_manager(block_reuse_config={"policy": "per_request"}, host_cache_size=0) as mgr:
        lender = attach(mgr)
        dev = kit.DevicePages(mgr)
        groups = range(kit.num_layer_groups(mgr))

        def allocate_pair(first_id, tokens):
            requests = [kit.make_request(first_id + i, tokens) for i in range(2)]
            for request in requests:
                assert mgr.prepare_context(request)
                kv = kit.kv(mgr, request)
                kv.enable_swa_scratch_reuse = False
                assert mgr.resize_context(request, len(tokens))
                assert kv.num_committed_tokens == 0
            return requests

        # Warm the native rebase and first close before holding the stream.
        warm = allocate_pair(900, list(range(20_000, 20_000 + 2 * TPB)))
        for request in warm:
            for lg in groups:
                for slot in kit.pages(kit.kv(mgr, request), lg):
                    if slot >= 0:
                        dev.fill_async(lg, slot, 0x11)
        for request in reversed(warm):
            step(mgr, request, 2 * TPB)
        for request in warm:
            mgr.free_resources(request)
        mgr._stream.synchronize()

        # Both requests are admitted before either publishes their shared prefix.
        first, canonical = allocate_pair(SOURCE, PROMPT[: 2 * TPB])
        first_kv, canonical_kv = (kit.kv(mgr, request) for request in (first, canonical))
        old_pages = {lg: kit.pages(first_kv, lg)[:2] for lg in groups}
        canonical_pages = {lg: kit.pages(canonical_kv, lg)[:2] for lg in groups}
        assert all(
            len(old_pages[lg]) == len(canonical_pages[lg]) == 2
            and min(old_pages[lg] + canonical_pages[lg]) >= 0
            and set(old_pages[lg]).isdisjoint(canonical_pages[lg])
            for lg in groups
        ), f"the requests have no distinct private pages: A={old_pages}, B={canonical_pages}"
        for lg in groups:
            dev.fill_async(lg, old_pages[lg][0], 0x40)
            for slot in canonical_pages[lg]:
                dev.fill_async(lg, slot, 0xEE)
        step(mgr, first, TPB)
        blocked_write = torch.cuda.Event()
        blocked_write.record(mgr._stream)
        mgr._stream.synchronize()
        assert first.context_current_position == TPB, first.context_current_position
        assert first_kv.num_committed_tokens == 0
        assert all(
            dev.read(lg, canonical_pages[lg][0]) == bytes([0xEE]) * dev.page_bytes(lg)
            for lg in groups
        ), "B's pending canonical pages were not initialized to stale bytes"

        lease = None
        try:
            with kit.held_stream(mgr._stream, strict=False, watchdog_seconds=20.0) as gate:
                for lg in groups:
                    for block, slot in enumerate(canonical_pages[lg]):
                        dev.fill_async(lg, slot, 0x40 + block)
                blocked_write.record(mgr._stream)
                step(mgr, canonical, 2 * TPB)
                for lg in groups:
                    dev.fill_async(lg, old_pages[lg][1], 0x41)
                step(mgr, first, TPB)
                rebased = {lg: kit.pages(first_kv, lg)[0] for lg in groups}
                assert all(
                    rebased[lg] == canonical_pages[lg][0] != old_pages[lg][0] for lg in groups
                ), f"no native rebase: old A={old_pages}, B={canonical_pages}, new A={rebased}"
                assert first_kv.num_committed_tokens == canonical_kv.num_committed_tokens == 2 * TPB
                assert not gate.opened and not blocked_write.query(), "the writes ran early"
                ready = kit.stream_event(mgr)
                lease = lender.lend_read(first, 0, TPB, **{**kit.DEVICE_ALL, "ready_event": ready})
                assert lease.failure is None, lease.failure
                early = lease.poll() is not None
                assert not gate.opened and not blocked_write.query(), "a lend waited for them"
            assert not early, f"A's read was ready before B wrote its rebased pages: {rebased}"
            lent = slots_by_block(kit.ready_view(lease, mgr))
            assert lent == {(lg, 0): canonical_pages[lg][0] for lg in groups}, lent
            assert holds(dev, lent, lambda ordinal: 0x40), "the canonical page has stale bytes"
        finally:
            if lease is not None:
                lease.release()


@pytest.mark.cpu_only
def test_a_stopped_lender_fails_an_in_place_lease_no_poll_returned():
    manager = Manager()
    lender = lender_over(manager)
    kv = cache()
    seen = in_place(lender, manager, kv, "read", entries((0,), (1,)), end=TPB)
    view = seen.poll()
    assert view is not None, seen.failure
    pending = in_place(lender, manager, kv, "read", entries((2,), (3,)), ready_event=Event())
    lender._stop_reason = "stopped"
    assert pending.poll() is None and pending.failure == "stopped"
    assert not lender._is_locked(kv, 2, 3), "a failed lease kept its lock"
    assert seen.poll() is view, "a view a poll returned before the stop was taken back"
    seen.release()
    pending.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ["lend_read", "lend_write"])
def test_a_device_lease_failed_at_the_call_is_an_in_place_lease_holding_nothing(kit, name):
    """A request this rank holds no cache for: its lease fails at the call, holding nothing."""
    manager = Manager()
    lender = lender_over(manager)
    lend = getattr(lender, name)
    lease = lend(request(), 0, TPB, **kit.DEVICE_ALL)
    assert isinstance(lease, _lender._InPlaceLease) and lease._lock is None
    assert lease.failure == _lender._no_cache_reason(RID) and lease.poll() is None
    assert lender._open_lease_count() == 0, "a lease in place kept the staging memory"
    lease.release()
    lease.release()
    staged = lend(request(), 0, TPB, **kit.HOST_NAMED)
    assert isinstance(staged, _lender._StagingLease)
    assert staged.failure == _lender._no_cache_reason(RID)
    staged.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("cause", ["shut_down", "gone", "stopped"])
@pytest.mark.parametrize("name", ["lend_read", "lend_write"])
def test_a_lender_that_lends_no_more_raises_past_the_argument_checks(kit, name, cause):
    """A lender that lends no more raises RuntimeError at a lend, after the argument checks."""
    manager = Manager()
    lender = lender_over(manager)
    if cause == "shut_down":
        lender._shut_down = True
    elif cause == "gone":
        del manager
    else:
        lender._stop_reason = _lender._REASON_STOPPED
    lend = getattr(lender, name)
    for keywords in (kit.DEVICE_ALL, kit.HOST_NAMED):
        with pytest.raises(RuntimeError, match="lends no more"):
            lend(request(), 0, TPB, **keywords)
    with pytest.raises(ValueError, match="bad token range"):
        lend(request(), TPB, 0, **kit.DEVICE_ALL)
    assert lender._open_lease_count() == 0, "a lend that raised left a lease open"


# -- reads --------------------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("delivery", ["local", "complete", "missing"])
def test_an_undelivered_committed_tail_cannot_be_lent_for_reading(kit, real_manager, delivery):
    with real_manager() as mgr:
        prompt = PROMPT[: TPB + 4]
        request = kit.published(mgr, SOURCE, prompt)
        lender = attach(mgr)
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens == kv.history_length == len(prompt)
        if delivery != "local":
            write = lender.lend_write(request, TPB, len(prompt), **kit.DEVICE_ALL)
            view = kit.ready_view(write, mgr)
            assert all(group.page_ordinals.tolist() == [1] for group in view.groups)
            return_write(lender, request, write, view.row_masks(delivery == "complete"))
        read = lender.lend_read(request, TPB, len(prompt), **kit.DEVICE_ALL)
        try:
            if delivery == "missing":
                assert read.failure is not None and read.poll() is None
                assert not lender._is_locked(kv)
            else:
                assert kit.ready_view(read, mgr) is not None, read.failure
        finally:
            read.release()


@gpu
def test_a_read_of_a_two_beam_request_lends_the_pages_native_sends(kit, real_manager):
    """A read of a two-beam prompt lends the first beam's pages, not the second's tail copy."""
    with real_manager(max_beam_width=2, enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        request = computed_to(mgr, two_beams(SOURCE), len(PROMPT))
        kv = kit.kv(mgr, request)
        assert mgr.try_allocate_generation(request), "no pages for the second beam"
        assert kv.beam_width == 2, f"the cache holds {kv.beam_width} beams"
        lease = lender.lend_read(request, 0, len(PROMPT), **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        for run in view.groups:
            lg = run.layer_group_index
            first = kit.pages(kv, lg)
            second = [
                int(p) for p in kv.get_aggregated_page_indices(lg, beam_id=1, valid_only=False)
            ]
            assert second[BLOCKS] >= 0 and second[BLOCKS] != first[BLOCKS], (lg, first, second)
            ordinals = run.page_ordinals.tolist()
            assert ordinals == list(range(BLOCKS + 1)), ordinals
            assert run.slots.tolist() == first[: BLOCKS + 1], (run.slots.tolist(), first)
        lease.release()


# -- locks and refusals -------------------------------------------------------------------------


@gpu
def test_a_write_whose_blocks_are_locked_fails_at_the_call_and_leaves_no_fetch(kit, real_manager):
    """A named write over a locked block fails at the call, before its grow and its fetch record."""
    with real_manager() as mgr:
        lender = attach(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        kv = kit.kv(mgr, target)
        held = lender._locks.lock(kv, {0: [1]}, LockMode.READ)
        before = (kv.capacity, lender.poll_resume_bounds(target))
        lease = lender.lend_write(target, 0, END, **kit.DEVICE_NAMED)
        assert lease.failure is not None and "cannot be locked" in lease.failure
        assert lease.poll() is None
        assert (kv.capacity, lender.poll_resume_bounds(target)) == before
        lease.release()
        lender._locks.unlock(held)


@gpu
def test_a_write_of_every_row_over_a_block_a_read_holds_fails_at_the_call(kit, real_manager):
    """A write of every row over a read-locked block fails at the call, recording no fetch."""
    with real_manager() as mgr:
        lender = attach(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        assert mgr.resize_context(target, target.context_remaining_length)
        kv = kit.kv(mgr, target)
        before = lender.poll_resume_bounds(target)
        held = lender._locks.lock(kv, {0: [1]}, LockMode.READ)
        lease = lender.lend_write(target, 0, END, **kit.DEVICE_ALL)
        assert lease.failure is not None and "locked for read" in lease.failure, lease.failure
        assert lease.poll() is None
        lease.release()
        lender._locks.unlock(held)
        assert not lender._is_locked(kv) and lender.poll_resume_bounds(target) == before
        lease = lender.lend_write(target, 0, END, **kit.DEVICE_ALL)
        view = kit.ready_view(lease, mgr)
        assert {lg: o.tolist() for lg, o in lease._lock.blocks.items()} == {
            run.layer_group_index: run.page_ordinals.tolist() for run in view.groups
        }
        return_write(lender, target, lease, view.row_masks(False))


@gpu
def test_reads_in_place_share_their_blocks_and_none_with_a_write(kit, real_manager):
    """Two reads of the same rows hold them together; a read and a write refuse each other."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        kv = kit.kv(mgr, source)
        writer = lender._locks.lock(kv, {0: [BLOCKS]}, LockMode.WRITE)
        refused(lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL), "locked for write")
        lender._locks.unlock(writer)
        first = lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL)
        second = lender.lend_read(source, TPB, len(PROMPT), **kit.DEVICE_ALL)
        for lease in (first, second):
            assert lease.failure is None, lease.failure
            kit.ready_view(lease, mgr)
        refused(lender.lend_write(source, 0, len(PROMPT), **kit.DEVICE_ALL), "locked for read")
        first.release()
        assert lender._is_locked(kv, BLOCKS, BLOCKS + 1), "the second read let go with the first"
        second.release()
        assert not lender._is_locked(kv)


@gpu
def test_a_write_in_place_over_a_block_without_a_page_fails_at_the_call(
    kit, real_manager, monkeypatch
):
    """A write of every row over a block without a page fails at the call, recording no fetch."""
    with real_manager() as mgr:
        lender = attach(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        assert mgr.resize_context(target, target.context_remaining_length)
        kv = kit.kv(mgr, target)
        before = lender.poll_resume_bounds(target)
        with monkeypatch.context() as patched:
            readers = ("pages", "page_list", "locked_pages")
            pages_giving(patched, lender, mgr, [target], 1, -1, readers)
            lease = lender.lend_write(target, 0, END, **kit.DEVICE_ALL)
        assert lease.failure is not None and "have no page" in lease.failure, lease.failure
        assert not lender._is_locked(kv)
        assert lender.poll_resume_bounds(target) == before, "the failed write left a fetch record"
        lease.release()


@gpu
def test_a_page_at_its_pool_s_slot_count_is_never_lent_in_place(kit, real_manager, monkeypatch):
    """A page at its pool's slot count fails a read at the call and a write at its first poll."""
    with real_manager() as mgr:
        lender = attach(mgr)
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)  # matches none of the source's blocks
        assert mgr.resize_context(target, target.context_remaining_length)
        caches = (kit.kv(mgr, source), kit.kv(mgr, target))
        limit = min(region.slot_count for region in device_regions(lender))
        before = lender.poll_resume_bounds(target)
        with monkeypatch.context() as patched:
            pages_giving(patched, lender, mgr, [source, target], 1, limit)
            read = lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL)
            write = lender.lend_write(target, 0, END, **kit.DEVICE_ALL)
        reason = f"blocks [1] have pages past the {limit} slots"
        assert read.failure is not None and reason in read.failure, read.failure
        assert write.failure is None and write.poll() is None
        assert write.failure is not None and reason in write.failure, write.failure
        assert not any(lender._is_locked(lent) for lent in caches)
        assert lender.poll_resume_bounds(target) == before, "the failed write left a fetch record"
        for lease in (read, write):
            lease.release()
        # Without that page the same lends lend the request's own pages.
        for request, lease in (
            (source, lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL)),
            (target, lender.lend_write(target, 0, END, **kit.DEVICE_ALL)),
        ):
            (rows,) = kit.ready_view(lease, mgr).groups
            pages = _manager.pages(kit.kv(mgr, request), rows.layer_group_index)
            assert rows.slots.tolist() == pages[rows.page_ordinals].tolist()
            if request is target:
                return_write(lender, target, lease, (np.ones(len(rows), dtype=bool),))
            else:
                lease.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("early_release", [False, True])
def test_post_grow_refusals_wait_for_poll_and_read_lock_refusals_fail_at_call(early_release):
    """A fetch's lock refusal fails at its first poll, a read's at the call."""
    manager = Manager()
    lender = lender_over(manager)
    kv = manager.kv_cache_map[RID] = cache()
    reader = in_place(lender, manager, kv, "read")
    fetch = lender._ledger._record_fetch(RID, kv, 0, (0, 0, 0), 0, (0, 0, 0))
    writer = in_place(lender, manager, kv, "write", fetch=fetch)
    assert writer.failure is None and writer._lock is None
    assert lender._ledger._latest_fetches.get(RID) is fetch
    if early_release:
        writer.release()
        assert writer.failure is None
    else:
        assert writer.poll() is None
        assert writer.failure is not None and "cannot be locked" in writer.failure
    assert not lender._ledger._latest_fetches, "the refused fetch stayed unsettled"
    reader.release()
    assert not lender._is_locked(kv)
    missing = lender._ledger._record_fetch(RID, kv, 0, (0, 0, 0), 0, (0, 0, 0))
    doomed = in_place(lender, manager, kv, "write", fetch=missing, doomed="blocks [1] have no page")
    assert doomed.failure is None and doomed._lock is None
    assert lender._ledger._latest_fetches.get(RID) is missing
    if early_release:
        doomed.release()
        assert doomed.failure is None
    else:
        assert doomed.poll() is None and doomed.failure == "blocks [1] have no page"
    assert not lender._ledger._latest_fetches
    for lease in (writer, doomed, doomed, reader):
        lease.release()
    blocker = in_place(lender, manager, kv, "write")
    refused_read = in_place(lender, manager, kv, "read")
    assert refused_read.failure is not None and "cannot be locked" in refused_read.failure
    assert refused_read.poll() is None
    refused_read.release()
    blocker.release()
    assert not lender._locks.is_locked(kv)


@pytest.mark.cpu_only
@pytest.mark.parametrize("why", ["locked", "pageless"])
def test_a_write_in_place_refused_at_the_call_leaves_the_records_as_they_were(
    kit, monkeypatch, why
):
    """A write over a read lock, or every row over a block without a page, fails at the call."""
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(
        _lender._ManagerLender, "_bidirectional_split_refusal", lambda self, request, end: None
    )
    manager = Manager()
    lender = lender_over(manager)
    pages = (0, 1, 2, -1) if why == "pageless" else None
    kv = manager.kv_cache_map[RID] = cache(history=2 * TPB, pages=pages)
    held = lender._locks.lock(kv, {0: [3]}, LockMode.READ) if why == "locked" else None
    before = lender.poll_resume_bounds(request())
    lease = lender.lend_write(request(), 2 * TPB, PROMPT_LEN, **kit.DEVICE_ALL)
    expected = "cannot be locked" if why == "locked" else "have no page"
    assert lease.failure is not None and expected in lease.failure, lease.failure
    assert lender.poll_resume_bounds(request()) == before, "the write left a record"
    lease.release()
    if held is not None:
        lender._locks.unlock(held)


@pytest.mark.cpu_only
def test_a_lease_in_place_lends_no_slot_past_its_regions_and_fails_instead():
    """A page at its regions' least slot count fails a read at the call and a write at its poll."""
    lay = layout()
    lay.device_pools = {
        0: (DevicePool(0, 0, 1 << 30, 256, 12), DevicePool(0, 1, 2 << 30, 128, 10)),
        1: (DevicePool(1, 0, 3 << 30, 512, 6),),
    }
    manager = Manager()
    lender = lender_over(manager, lay)
    limits = (10, 6, 10)  # pool group 0's smaller pool bounds layer groups 0 and 2
    kv = manager.kv_cache_map[RID] = cache()
    highest = [limit - 1 for limit in limits]

    def rows(pages):
        return _lender._LentBlockEntries(
            [0, 1, 2], [np.array([0, 1], dtype=np.int64)] * 3, [np.array([0, p]) for p in pages]
        )

    lease = in_place(lender, manager, kv, "read", rows(highest))
    view = lease.poll()
    assert view is not None, lease.failure
    assert [group.slots.tolist() for group in view.groups] == [[0, p] for p in highest]
    lease.release()
    for lg, limit in enumerate(limits):
        pages = list(highest)
        pages[lg] = limit
        reason = f"layer group {lg}: blocks [1] have pages past the {limit} slots"
        read = in_place(lender, manager, kv, "read", rows(pages))
        assert read.failure is not None and read.failure.startswith(reason), read.failure
        assert read._lock is None and read.poll() is None and not lender._is_locked(kv)
        read.release()
        fetch = lender._ledger._record_fetch(RID, kv, 0, (0, 0, 0), 0, (0, 0, 0))
        write = in_place(lender, manager, kv, "write", rows(pages), fetch)
        assert write.failure is None and write._lock is None and not lender._is_locked(kv)
        assert write.poll() is None and write.failure.startswith(reason), write.failure
        assert RID not in lender._ledger._latest_fetches, "the refused write's fetch stayed open"
        write.release()


@pytest.mark.cpu_only
def test_the_lock_mode_follows_the_lease_kind():
    manager = Manager()
    lender = lender_over(manager)
    kv = cache()
    read = in_place(lender, manager, kv, "read")
    other = in_place(lender, manager, kv, "read")
    assert read._lock.mode is other._lock.mode is LockMode.READ and other.failure is None
    for lease in (read, other):
        lease.release()
    write = in_place(lender, manager, kv, "write")
    assert write._lock.mode is LockMode.WRITE
    write.release()


# -- writes and returns -------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("read_only", [False, True], ids=["writable_mask", "read_only_mask"])
@pytest.mark.parametrize(
    "rows,usable", [((0, 1), 64), ((0, 2), 32), ((), 0)], ids=["prefix", "gap", "nothing"]
)
def test_an_in_place_write_lands_in_the_pages_and_counts_only_marked_rows(
    kit, real_manager, rows, usable, read_only
):
    with real_manager() as mgr:
        lender = attach(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        lease = lender.lend_write(target, 0, END, **kit.DEVICE_NAMED)
        view = kit.ready_view(lease, mgr)
        assert view.num_rows == BLOCKS and all(r.object_keys is not None for r in view.groups)
        write_through(lender, view, FRESH)
        mask = np.isin(np.arange(BLOCKS), rows)
        mask.flags.writeable = not read_only
        assert return_write(lender, target, lease, (mask,)) == (usable, 0)
        mgr._stream.synchronize()
        page_bytes = kit.DevicePages(mgr).page_bytes(0)
        for ordinal in range(BLOCKS):
            assert kit.page(mgr, target, 0, ordinal) == bytes([FRESH]) * page_bytes, ordinal


@gpu
def test_a_receive_of_every_row_writes_the_admitted_pages_up_to_the_prompt_end(kit, real_manager):
    with real_manager() as mgr:
        lender = attach(mgr)
        request, admitted = receiving(kit, mgr, TARGET, PROMPT)
        assert admitted is not None, "the admission recorded no event"
        keywords = {**kit.DEVICE_ALL, "ready_event": admitted}
        with pytest.raises(ValueError, match="past the request's"):
            lender.lend_write(request, 0, len(PROMPT) + 1, **keywords)
        refused(
            lender.lend_write(request, 0, len(PROMPT) - 2, **keywords), f"ends at {len(PROMPT) - 2}"
        )
        lease = lender.lend_write(request, 0, len(PROMPT), **keywords)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        assert [run.page_ordinals.tolist() for run in view.groups] == [list(range(BLOCKS + 1))]
        assert all(run.object_keys is None for run in view.groups)
        write_through(lender, view, FRESH)
        assert return_write(lender, request, lease, view.row_masks(True)) == (len(PROMPT), 0)
        mgr._stream.synchronize()
        page_bytes = kit.DevicePages(mgr).page_bytes(0)
        for ordinal in range(BLOCKS + 1):
            assert kit.page(mgr, request, 0, ordinal) == bytes([FRESH]) * page_bytes


@gpu
def test_a_receive_may_end_at_a_remote_tail_s_start_and_resumes_no_further(kit, real_manager):
    """A receive may end at a CSA2 remote tail's start inside a block, and resumes there only."""
    tail = 2 * TPB + 6
    with real_manager(enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        request, admitted = receiving(kit, mgr, TARGET, PROMPT, py_csa2_remote_tail_start=tail)
        kv = kit.kv(mgr, request)
        assert kv.history_length == tail, f"the cache declares {kv.history_length} tokens"
        keywords = {**kit.DEVICE_ALL, "ready_event": admitted}
        refused(lender.lend_write(request, 0, tail - 1, **keywords), f"ends at {tail - 1}")
        lease = lender.lend_write(request, 0, tail, **keywords)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        assert [run.page_ordinals.tolist() for run in view.groups] == [[0, 1, 2]]
        bounds = return_write(lender, request, lease, view.row_masks(True))
        assert tuple(bounds) == (tail, tail)


@gpu
def test_a_fetch_in_place_settles_in_the_lease_call_that_ends_it(kit, real_manager, monkeypatch):
    """The lease call that ends a fetch in place settles it, as for a twin queried every step."""
    with real_manager() as mgr:
        lender = attach(mgr)
        quiet = kit.admitted(mgr, TARGET, PROMPT)
        queried = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        leases = {r: lender.lend_write(r, 0, END, **kit.DEVICE_NAMED) for r in (quiet, queried)}
        views = {}
        for _ in range(3):  # iterations: poll every open lease, query only ``queried``'s bounds
            mgr._stream.synchronize()
            for request, lease in leases.items():
                view = lease.poll()
                if view is not None:
                    views[request] = view
            assert lender.poll_resume_bounds(queried) is None, "settled before its return"
        assert len(views) == 2, [lease.failure for lease in leases.values()]
        returned = lender.return_writes(
            [(lease, views[request].row_masks(True), True) for request, lease in leases.items()]
        )
        assert returned == [(END, 0), (END, 0)], returned
        assert lender.poll_resume_bounds(quiet) == lender.poll_resume_bounds(queried) == (END, 0)
        # A release before the view, and a first poll that fails, leave the bounds as they were.
        released = kit.admitted(mgr, 4, list(range(9000, 9097)))
        failed = kit.admitted(mgr, 5, list(range(11000, 11097)))
        assert mgr.resize_context(failed, failed.context_remaining_length)
        before = {r: lender.poll_resume_bounds(r) for r in (released, failed)}
        lease = lender.lend_write(released, 0, END, **kit.DEVICE_NAMED)
        assert lease.failure is None, lease.failure
        lease.release()
        assert lender.poll_resume_bounds(released) == before[released]
        limit = min(region.slot_count for region in device_regions(lender))
        with monkeypatch.context() as patched:
            pages_giving(patched, lender, mgr, [failed], 1, limit)
            lease = lender.lend_write(failed, 0, END, **kit.DEVICE_ALL)
        assert lease.failure is None and lease.poll() is None and lease.failure is not None
        assert lender.poll_resume_bounds(failed) == before[failed]
        lease.release()


@gpu
def test_a_write_released_after_its_view_settles_as_one_returned_with_no_row_arrived(
    kit, real_manager
):
    """A write released after its view settles as a twin returned with no row arrived."""
    with real_manager() as mgr:
        lender = attach(mgr)
        released = kit.admitted(mgr, TARGET, PROMPT)
        returned = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        bounds = {}
        for request in (released, returned):
            lease = lender.lend_write(request, 0, END, **kit.DEVICE_NAMED)
            view = kit.ready_view(lease, mgr)
            assert lender.poll_resume_bounds(request) is None, "settled before the lease ended"
            if request is released:
                lease.release()
                bounds[request] = lender.poll_resume_bounds(request)
            else:
                bounds[request] = return_write(lender, request, lease, view.row_masks(False))
        assert bounds[released] is not None, "a release after the view left the fetch unsettled"
        assert bounds[released] == bounds[returned], bounds
        assert bounds[released].max_position == 0, bounds


@gpu
@pytest.mark.parametrize("reuse", [False, True], ids=["reuse_off", "reuse_on"])
def test_a_receive_into_beam_zero_reaches_the_second_beam_at_the_first_generation(
    kit, real_manager, reuse
):
    """A two-beam receive into beam 0's pages reaches beam 1 at its first generation allocation."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

    with real_manager(max_beam_width=2, enable_block_reuse=reuse) as mgr:
        lender = attach(mgr)
        request = two_beams(TARGET, llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)
        assert mgr.prepare_disagg_gen_init(request)
        ready = mgr.take_disagg_receive_ready(request)
        lease = lender.lend_write(
            request, 0, len(PROMPT), **{**kit.DEVICE_ALL, "ready_event": ready}
        )
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        assert [run.page_ordinals.tolist() for run in view.groups] == [list(range(BLOCKS + 1))]
        write_through(lender, view, FRESH)
        bounds = return_write(lender, request, lease, view.row_masks(True))
        assert bounds.max_position == len(PROMPT)
        kv = kit.kv(mgr, request)
        mine = set(kit.pages(kv, 0))
        dev = kit.DevicePages(mgr)
        # Every other page holds SENTINEL before the allocation, so a copy into one shows.
        for slot in set(range(kit.pool_pages(mgr))) - mine:
            dev.fill_async(0, slot, kit.SENTINEL)
        mgr._stream.synchronize()
        assert mgr.try_allocate_generation(request), "no pages for the second beam"
        assert kv.beam_width == 2, f"the cache holds {kv.beam_width} beams"
        first = kit.pages(kv, 0)
        second = [int(p) for p in kv.get_aggregated_page_indices(0, beam_id=1, valid_only=False)]
        assert len(second) == BLOCKS + 1 and second[BLOCKS] >= 0, (first, second)
        assert second[BLOCKS] not in mine, f"beam 1's tail page {second[BLOCKS]} is beam 0's"
        received = kit.digest(bytes([FRESH]) * dev.page_bytes(0))
        for ordinal in range(BLOCKS):
            # -1: the second beam reads beam 0's page for the block.
            page = first[ordinal] if second[ordinal] == -1 else second[ordinal]
            assert kit.digest(dev.read(0, page)) == received, f"beam 1's block {ordinal}"
        assert kit.digest(dev.read(0, second[BLOCKS])) == received, "beam 1's own tail page"
        assert kit.digest(dev.read(0, first[BLOCKS])) == received, "beam 0's tail block"


# Rows that arrived in each layer group of ``write_with_its_view``'s lease, of blocks 1 to 3.
ARRIVED_ROWS = ((False, True, True), (True, True, True), (True, False, True))


class BoolMask(np.ndarray):
    """A subclass of ``numpy.ndarray``."""


def frozen(values, dtype=bool):
    """``values`` as a read-only array of ``dtype``."""
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


def write_with_its_view():
    """A write in place of blocks 1 to 3 of a cache that commits one block and five tokens of the
    next, after a poll returned its view; with its manager, lender and fetch."""
    manager = Manager()
    lender = lender_over(manager)
    kv = manager.kv_cache_map[RID] = cache(blocks=4, committed=TPB + 5)
    fetch = lender._ledger._record_fetch(
        RID, kv, TPB, (TPB,) * 3, kv.history_length, (kv.num_committed_tokens,) * 3
    )
    lease = in_place(lender, manager, kv, "write", entries((1, 2, 3), (1, 2, 3)), fetch, 4 * TPB)
    assert lease.poll() is not None
    return manager, lender, fetch, lease


@pytest.mark.cpu_only
def test_marks_in_place_count_only_marked_rows_still_lent_and_taint_an_unmarked_tail():
    manager, lender, _, lease = write_with_its_view()
    masks = lease.poll().row_masks()
    masks[0][1:] = True  # layer group 0: blocks 2 and 3, not the committed tail block 1
    masks[1][:] = True
    masks[2][:] = True
    bounds = return_write(lender, request(), lease, masks)
    delivered = lender._ledger._deliveries_of(RID)
    assert [m.nonzero()[0].tolist() for m in delivered.delivered_block_masks] == [
        [2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert bounds.max_position == 0, f"a request with a mixed tail block resumes: {bounds}"
    assert not lender._locks.is_locked(manager.kv_cache_map[RID])


@pytest.mark.cpu_only
@pytest.mark.parametrize("form", ["read_only", "writable", "read_only_subclass", "list"])
def test_a_return_keeps_a_read_only_bool_mask_as_given_and_copies_any_other(monkeypatch, form):
    """A read-only bool ``numpy.ndarray`` mask is kept as given, any other as an unshared copy."""
    manager, lender, _, lease = write_with_its_view()
    given = {
        "read_only": lambda: [frozen(rows) for rows in ARRIVED_ROWS],
        "writable": lambda: [np.array(rows) for rows in ARRIVED_ROWS],
        "read_only_subclass": lambda: [frozen(rows).view(BoolMask) for rows in ARRIVED_ROWS],
        "list": lambda: [list(rows) for rows in ARRIVED_ROWS],
    }[form]()
    taken = []
    count = lender._return_write

    def recording_count(lease, masks, reusable):
        taken.extend(masks)
        return count(lease, masks, reusable)

    monkeypatch.setattr(lender, "_return_write", recording_count)
    assert return_write(lender, request(), lease, given) is not None, "the fetch did not settle"
    assert len(taken) == len(given) == 3
    for mask, kept in zip(given, taken):
        if form == "read_only":
            assert kept is mask, "a read-only bool mask was copied"
        else:
            assert type(kept) is np.ndarray and kept.dtype == np.bool_, type(kept)
            assert not np.shares_memory(kept, np.asarray(mask)), f"a {form} mask was kept"
            assert kept.tolist() == list(mask)
    delivered = lender._ledger._deliveries_of(RID)
    assert [m.nonzero()[0].tolist() for m in delivered.delivered_block_masks] == [
        [2, 3],
        [1, 2, 3],
        [1, 3],
    ]
    assert not lender._locks.is_locked(manager.kv_cache_map[RID]), "the return left the lease lent"


@pytest.mark.cpu_only
def test_a_return_refuses_a_read_only_mask_of_another_dtype_or_length_and_changes_nothing():
    """A read-only mask of integers or of another length is refused, the fetch and lease kept."""
    manager, lender, _, lease = write_with_its_view()
    kv = manager.kv_cache_map[RID]
    good = [frozen(rows) for rows in ARRIVED_ROWS]
    for bad in (
        [frozen([1, 1, 1], np.int64)] + good[1:],
        good[:1] + [frozen([True, True])] + good[2:],
    ):
        with pytest.raises(ValueError, match="the mask must be bool"):
            return_write(lender, request(), lease, bad)
        assert lender.poll_resume_bounds(request()) is None, "a refused return settled the fetch"
        assert lender._locks.is_locked(kv), "a refused return let the lease go"
    assert return_write(lender, request(), lease, good) is not None
    assert not lender._locks.is_locked(kv)


@pytest.mark.cpu_only
def test_a_write_released_before_its_view_abandons_its_fetch_and_one_after_delivers_nothing():
    """A write released before its view abandons its fetch; one released after it counts no row."""
    manager = Manager()
    lender = lender_over(manager)
    # The pages ``entries()`` lends: an arrived row would count.
    kv = manager.kv_cache_map[RID] = cache(pages=(5, 6, 7, 8))
    for seen in (False, True):
        fetch = lender._ledger._record_fetch(RID, kv, 0, (0, 0, 0), 0, (0, 0, 0))
        lease = in_place(lender, manager, kv, "write", fetch=fetch)
        if seen:
            assert lease.poll() is not None
        lease.release()
        latest = lender._ledger._latest_fetches.get(RID)
        assert (latest is fetch) == seen
        if seen:
            assert lender._ledger._fetch_settled(fetch), "a release after the view left it open"
            delivered = lender._ledger._deliveries_of(RID).delivered_block_masks
            assert not any(blocks.any() for blocks in delivered), "a release counted rows"
            with pytest.raises(RuntimeError, match="returned already"):
                return_write(lender, request(), lease, lease._view.row_masks(True))
            delivered = lender._ledger._deliveries_of(RID).delivered_block_masks
            assert not any(blocks.any() for blocks in delivered), "a refused return counted rows"
        assert not lender._locks.is_locked(kv)


# -- what a lent cache keeps --------------------------------------------------------------------


@gpu
def test_a_freed_request_keeps_its_lent_pages_until_the_lease_is_released(kit, real_manager):
    """A freed request's lent cache stays open, each of its pages counted, until the release."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr)
        lease = lender.lend_read(source, 0, END, **kit.DEVICE_NAMED)
        view = kit.ready_view(lease, mgr)
        kv = kit.kv(mgr, source)
        held = {}
        for lg, group in enumerate(kit.pool_group_of(mgr)):
            held[group] = held.get(group, 0) + sum(page >= 0 for page in kit.pages(kv, lg))
        lent = sum(len(run) for run in view.groups)
        assert sum(held.values()) > lent, f"inconclusive: {held} pages, {lent} of them lent"
        assert dict(lender.held_page_counts().deduct) == {}, "a live request's pages counted"
        before = [entry_bytes(lender, run, i) for run in view.groups for i in range(len(run))]
        mgr.free_resources(source)
        assert not kit.closed(kv), "the free closed a cache whose pages are lent"
        assert mgr._holds_page_locks(None), "the kept cache's lock went unseen"
        retained = dict(lender.held_page_counts().deduct)
        assert retained == {g: n for g, n in held.items() if n}, (retained, held)
        kit.overwrite_free_pages(mgr)
        after = [entry_bytes(lender, run, i) for run in view.groups for i in range(len(run))]
        assert after == before, "a freed request's lent page went to another request"
        lease.release()
        assert kit.closed(kv), "the release did not close the freed request's cache"
        assert not mgr._holds_page_locks(None)
        assert dict(lender.held_page_counts().deduct) == {}, "a closed cache's pages still count"


def live_slots(kv, lg, recurrent):
    """The slots layer group ``lg`` of the live cache ``kv`` holds over its beams: its pages, its
    recurrent state's slot, its scratch slots."""
    slots = set()
    for beam in range(int(kv.beam_width)):
        if recurrent:
            slots.add(int(kv.get_ssm_block_base_index(lg, beam_id=beam)))
        else:
            slots.update(
                int(p) for p in kv.get_aggregated_page_indices(lg, beam_id=beam, valid_only=True)
            )
    scratch = kv.get_scratch_desc(lg)
    slots.update(int(s) for s in (scratch.slot_ids if scratch is not None else ()))
    slots.discard(-1)
    return slots


@gpu
@pytest.mark.parametrize("kind", ["recurrent_state", "swa_scratch", "two_beams"])
def test_a_freed_cache_counts_its_state_scratch_and_beams_as_retained(
    kit, real_manager, hybrid_manager, kind
):
    """A freed lent cache counts as retained its state slot, scratch slots and both beams' pages."""
    if kind == "recurrent_state":
        manager = hybrid_manager()
    elif kind == "swa_scratch":
        manager = real_manager(windows=[WINDOW, 256], swa_scratch_reuse=True)
    else:
        manager = real_manager(max_beam_width=2, enable_block_reuse=False)
    with manager as mgr:
        lender = attach(mgr)
        recurrent = derive_layout(mgr).recurrent
        if kind == "two_beams":
            request = two_beams(SOURCE)
        else:
            request = kit.make_request(SOURCE, LONG_PROMPT if kind == "swa_scratch" else PROMPT)
        end = 2 * TPB if kind == "swa_scratch" else len(PROMPT)
        computed_to(mgr, request, end)
        kv = kit.kv(mgr, request)
        if kind == "two_beams":
            assert mgr.try_allocate_generation(request), "no pages for the second beam"
            assert kv.beam_width == 2, f"the cache holds {kv.beam_width} beams"
        lease = lender.lend_read(request, 0, end, **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        kit.ready_view(lease, mgr)
        if kind == "swa_scratch":
            # The next chunk, in flight at the free, takes the 64-token window past its first two
            # blocks, which the runtime backs with scratch slots, not pages.
            assert mgr.resize_context(request, 4 * TPB), "no capacity for the next chunk"
        held = {}
        for lg, group in enumerate(kit.pool_group_of(mgr)):
            slots = live_slots(kv, lg, recurrent[lg])
            if slots:
                held[group] = held.get(group, 0) + len(slots)
        if kind == "recurrent_state":
            assert any(recurrent), "inconclusive: the manager holds no recurrent state"
        elif kind == "swa_scratch":
            (windowed,) = [lg for lg, window in enumerate(kit.windows(mgr)) if window == WINDOW]
            scratch = kv.get_scratch_desc(windowed)
            assert scratch is not None and len(scratch.slot_ids), (kv.history_length, kv.capacity)
            # The count includes slots that only the scratch descriptor names.
            pages = set(kit.pages(kv, windowed))
            assert not pages & {int(s) for s in scratch.slot_ids}, (pages, scratch.slot_ids)
        else:
            (lg,) = range(kit.num_layer_groups(mgr))
            assert held == {kit.pool_group_of(mgr)[lg]: BLOCKS + 2}, f"inconclusive: {held}"
        mgr.free_resources(request)
        assert not kit.closed(kv), "the free closed a cache whose pages are lent"
        assert dict(lender.held_page_counts().deduct) == held
        lease.release()
        assert kit.closed(kv), "the release did not close the freed request's cache"
        assert dict(lender.held_page_counts().deduct) == {}


@gpu
def test_a_hold_with_device_regions_keeps_the_runtime_past_the_shutdown(kit, real_manager):
    with real_manager() as mgr:
        lender = attach(mgr)
        hold = lender.hold_regions()
        runtime = mgr.impl
        mgr.shutdown()
        assert lender._keeps_pools()
        assert any(kept is runtime for kept in kit.retained()), "the runtime is not kept"
        hold.release()
        assert not lender._keeps_pools()
    # The block's exit shuts the manager down again, and with nothing held the runtime's own runs.


@pytest.mark.cpu_only
def test_a_lender_reports_the_pages_its_leases_keep_past_their_requests_free():
    """The lender counts a lent cache's pages from its request's free to its last release."""
    manager = Manager()
    lender = lender_over(manager)
    pages = [5, 6, 7, -1]
    kv = SimpleNamespace(
        Status=SimpleNamespace(OPEN=0, CLOSED=1),
        status=0,
        num_committed_tokens=0,
        tokens_per_block=TPB,
        beam_width=1,
        closed=False,
        get_scratch_desc=lambda lg: None,
        get_aggregated_page_indices=lambda lg, beam_id=0, valid_only=False: (
            [p for p in pages if p >= 0] if valid_only else pages
        ),
    )

    def close():
        kv.closed = True
        kv.status = kv.Status.CLOSED

    kv.close = close
    lease = in_place(lender, manager, kv, "read")
    assert dict(lender.held_page_counts().deduct) == {}
    assert lender._defer_close(kv), "a locked cache was not kept past its free"
    # Pool group 0 holds layer groups 0 and 2, three pages each; pool group 1 layer group 1.
    assert dict(lender.held_page_counts().deduct) == {0: 6, 1: 3}
    lease.release()
    assert kv.closed and dict(lender.held_page_counts().deduct) == {}


# -- host fetches beside reads in place ---------------------------------------------------------


@pytest.fixture
def mixed_host_case(monkeypatch):
    """Request ``RID`` after its first context step: a history of a block, its position past it."""
    manager = Manager()
    manager.enable_block_reuse = False
    manager._can_publish_block_reuse = False
    manager.is_draft = False
    manager._fresh_page_fill = None
    manager._resize_for_served_prefix = lambda req, kv, position, end: True
    manager._stale_block_range = lambda lg, history: (0, 0)
    manager.kv_cache_map[RID] = cache(history=TPB)
    lender = lender_over(manager)
    req = request(computed=TPB)
    req.py_return_context_logits = False
    req.py_additional_outputs = None
    req.try_get_encoder_output_len = lambda: None
    monkeypatch.setattr(
        _manager, "block_keys", lambda m, r, k, n: [bytes([i]) * 32 for i in range(n)]
    )
    copied = Event()
    monkeypatch.setattr(
        lender,
        "_queue_copy",
        lambda segments, to_staging, after=None: (_lender._QueuedCopy(copied), None),
    )
    return manager, lender, req, copied


@pytest.mark.cpu_only
def test_host_fetch_refuses_pages_a_device_reader_still_holds(kit, mixed_host_case):
    _, lender, req, _ = mixed_host_case
    read = lender.lend_read(req, 0, TPB, **kit.DEVICE_ALL)
    assert read.poll() is not None
    fetch = lender.lend_write(req, 0, 2 * TPB, **kit.HOST_NAMED)
    assert fetch.failure is not None and "locked" in fetch.failure
    assert lender._ledger._latest_fetches == {}
    fetch.release()
    read.release()


@pytest.mark.cpu_only
def test_host_fetch_excludes_later_device_reads_until_its_marked_copy_finishes(
    kit, mixed_host_case
):
    """A read of rows a host fetch lent fails at the call until progress saw its copy complete."""
    _, lender, req, copied = mixed_host_case
    fetch = lender.lend_write(req, 0, 2 * TPB, **kit.HOST_NAMED)
    view = fetch.poll()
    assert view is not None, fetch.failure
    for returned in (False, True):
        if returned:
            bounds = return_write(lender, req, fetch, view.row_masks(True))
            assert bounds is None, f"settled before its copy ran: {bounds}"
        refused(lender.lend_read(req, 0, TPB, **kit.DEVICE_ALL), "unsettled")
    copied.done = True
    lender.progress()
    ready = Event()
    read = lender.lend_read(req, 0, TPB, **{**kit.DEVICE_ALL, "ready_event": ready})
    assert read.failure is None, read.failure
    assert read.poll() is None
    ready.done = True
    assert read.poll() is not None
    read.release()


@pytest.mark.cpu_only
def test_a_host_fetch_leaves_nonoverlapping_device_rows_readable(kit, mixed_host_case):
    _, lender, req, copied = mixed_host_case
    fetch = lender.lend_write(req, TPB, 2 * TPB, **kit.HOST_NAMED)
    view = fetch.poll()
    assert view is not None
    read = lender.lend_read(req, 0, TPB, **kit.DEVICE_ALL)
    assert read.poll() is not None, read.failure
    return_write(lender, req, fetch, view.row_masks(True))
    copied.done = True
    read.release()


@pytest.mark.cpu_only
def test_a_device_read_of_a_row_another_group_still_fetches_fails_at_the_call(kit, mixed_host_case):
    """A read in place overlapping rows another layer group still fetches fails at the call."""
    manager, lender, req, copied = mixed_host_case
    # Block 0 committed, a claim complete in layer group 0 alone.
    kv = manager.kv_cache_map[RID]
    kv.num_committed_tokens = TPB
    kv.reuse_status = [SimpleNamespace(complete=c, endpoint=TPB) for c in (True, False, False)]
    fetch = lender.lend_write(req, 0, 2 * TPB, **kit.HOST_NAMED)
    view = fetch.poll()
    assert view is not None, fetch.failure
    pending = {g.layer_group_index: g.page_ordinals.tolist() for g in view.groups}
    assert pending == {0: [1], 1: [0, 1], 2: [0, 1]}, pending
    refused(lender.lend_read(req, 0, TPB, **kit.DEVICE_ALL), "unsettled")
    return_write(lender, req, fetch, view.row_masks(True))
    copied.done = True
    lender.progress()
    read = lender.lend_read(req, 0, TPB, **kit.DEVICE_ALL)
    assert read.failure is None, read.failure
    read.release()


# -- a request that runs on ---------------------------------------------------------------------


@pytest.mark.cpu_only
def test_a_read_past_the_history_needs_a_finished_step_that_computed_its_end(kit, recorded):
    """A read past the history ends where the request's steps computed, after its caller's event."""
    manager = Manager()
    lender = lender_over(manager)
    manager.kv_cache_map[RID] = cache(history=TPB)
    why = "past the 32 tokens the cache holds"
    refused(lender.lend_read(request(computed=TPB), 0, 2 * TPB, **kit.DEVICE_ALL), why)
    ran = Event()
    keywords = {**kit.DEVICE_ALL, "ready_event": ran}
    lease = lender.lend_read(request(computed=2 * TPB), 0, 2 * TPB, **keywords)
    assert lease.failure is None, lease.failure
    assert recorded == []
    assert lease.poll() is None, "the view went out before the step that computed it ran"
    ran.done = True
    assert lease.poll() is not None
    lease.release()
    refused(lender.lend_read(request(computed=2 * TPB), 0, 3 * TPB, **kit.DEVICE_ALL), why)


@pytest.mark.cpu_only
def test_until_the_prompt_is_computed_a_read_of_every_row_ends_at_a_whole_block(kit, recorded):
    """A read of every row ends at a whole block until the prompt is computed."""
    manager = Manager()
    lender = lender_over(manager)
    manager.kv_cache_map[RID] = cache(history=3 * TPB)
    refused(
        lender.lend_read(request(), 0, 2 * TPB + 5, **kit.DEVICE_ALL),
        "until this worker computes its context to 97 tokens",
    )
    lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_ALL).release()
    # The prompt's last step ran, while a window held the history below the prompt's end.
    computed = request(computed=PROMPT_LEN)
    lender.lend_read(computed, 0, 2 * TPB + 5, **kit.DEVICE_ALL).release()
    # Generating, past the prompt: the history holds it.
    manager.kv_cache_map[RID] = cache(blocks=5, history=PROMPT_LEN + 3)
    lease = lender.lend_read(computed, 0, PROMPT_LEN + 2, **kit.DEVICE_ALL)
    assert lease.failure is None, lease.failure
    lease.release()
    refused(
        lender.lend_read(computed, 0, PROMPT_LEN + 4, **kit.DEVICE_ALL),
        "past the 100 tokens the cache holds",
    )
    assert recorded == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("computed", ["history", "position"])
@pytest.mark.parametrize("scratch", [False, True], ids=["scratch_off", "scratch_on"])
def test_a_read_before_the_prompt_is_computed_leaves_out_windows_shorter_than_the_prompt(
    kit, recorded, scratch, computed
):
    """Until the prompt is computed a read lends and holds no window shorter than the prompt."""
    windows = (None, 64, PROMPT_LEN)
    manager = Manager(windows)
    lender = lender_over(manager, layout(windows))
    kv = manager.kv_cache_map[RID] = cache(history=2 * TPB)
    kv.enable_swa_scratch_reuse = scratch
    lease = lender.lend_read(request(computed=2 * TPB), 0, 2 * TPB, **kit.DEVICE_ALL)
    assert lease.failure is None, lease.failure
    lent = {run.layer_group_index: run.page_ordinals.tolist() for run in lease.poll().groups}
    assert lent == {0: [0, 1], 1: [], 2: [0, 1]}, lent
    assert {lg: blocks.tolist() for lg, blocks in lease._lock.blocks.items()} == lent
    cap = lender._history_cap(kv)
    assert cap is None or cap >= PROMPT_LEN, f"the read holds the context's history at {cap}"
    lease.release()
    # The context's last step turned scratch reuse off; its history reached the prompt's end, or
    # stayed at the last whole block while the request's context position reached it.
    kv.enable_swa_scratch_reuse = False
    kv.history_length = PROMPT_LEN if computed == "history" else 3 * TPB
    ran = request(computed=3 * TPB if computed == "history" else PROMPT_LEN)
    after = lender.lend_read(ran, 0, PROMPT_LEN, **kit.DEVICE_ALL)
    assert after.failure is None, after.failure
    lent = {run.layer_group_index: run.page_ordinals.tolist() for run in after.poll().groups}
    assert lent == {0: [0, 1, 2, 3], 1: [1, 2, 3], 2: [0, 1, 2, 3]}, lent
    # The 64-token window leaves block 1 behind at a history of 127.
    assert lender._history_cap(kv) == 126, lender._history_cap(kv)
    after.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "position,remaining,helix,end",
    [
        (TPB, PROMPT_LEN - TPB, False, PROMPT_LEN),
        (2 * TPB + 6, 0, False, 2 * TPB + 6),
        (2 * TPB, PROMPT_LEN, False, PROMPT_LEN),
        (2 * TPB + 6, 0, True, LONG),
    ],
    ids=["to_the_prompt", "short_of_the_prompt", "past_the_prompt", "helix"],
)
def test_this_worker_s_context_ends_where_its_remaining_rows_end_within_the_prompt(
    position, remaining, helix, end
):
    """The context ends at position plus remaining rows, within the prompt; under Helix, at it."""
    manager = SimpleNamespace(_has_cp_helix=helix, _helix_cp_rank=1, _helix_cp_size=2)
    req = SimpleNamespace(
        prompt_len=PROMPT_LEN,
        total_input_len_cp=LONG,
        context_current_position=position,
        context_remaining_length=remaining,
    )
    assert _manager.context_end(req, manager) == end
    assert _manager.context_computed(req) == position


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tail,lent",
    [
        (2 * TPB + 6, {0: [0, 1, 2], 1: [0, 1, 2], 2: [2]}),
        (4 * TPB, {0: [0, 1, 2, 3], 1: [2, 3], 2: [3]}),
    ],
    ids=["inside_a_block", "at_a_block"],
)
def test_a_context_ending_short_of_the_prompt_is_computed_once_this_worker_computed_it(
    kit, recorded, tail, lent
):
    """A context ending short of the prompt, computed to its end, lends what each window keeps."""
    windows = (None, 64, 2)
    manager = Manager(windows)
    lender = lender_over(manager, layout(windows))
    manager.kv_cache_map[RID] = cache(blocks=-(-tail // TPB), history=tail)
    ran = request(computed=tail, prompt=LONG, end=tail)
    lease = lender.lend_read(ran, 0, tail, **kit.DEVICE_ALL)
    assert lease.failure is None, lease.failure
    got = {run.layer_group_index: run.page_ordinals.tolist() for run in lease.poll().groups}
    assert got == lent, got
    lease.release()


@pytest.mark.cpu_only
def test_while_a_context_ending_short_of_the_prompt_runs_both_rules_hold_against_its_end(
    kit, recorded
):
    """Mid-context, a read ends at a whole block and leaves out windows shorter than the end."""
    tail = 4 * TPB
    windows = (None, 64, 2, tail)
    manager = Manager(windows)
    lender = lender_over(manager, layout(windows))
    manager.kv_cache_map[RID] = cache(blocks=3, history=3 * TPB)
    running = request(computed=3 * TPB, prompt=LONG, end=tail)
    refused(
        lender.lend_read(running, 0, 2 * TPB + 5, **kit.DEVICE_ALL),
        "until this worker computes its context to 128 tokens",
    )
    # A request at the same step whose context runs to the prompt leaves out the tail's window too.
    for req, lent in (
        (running, {0: [0, 1, 2], 1: [], 2: [], 3: [0, 1, 2]}),
        (request(computed=3 * TPB, prompt=LONG), {0: [0, 1, 2], 1: [], 2: [], 3: []}),
    ):
        lease = lender.lend_read(req, 0, 3 * TPB, **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        got = {run.layer_group_index: run.page_ordinals.tolist() for run in lease.poll().groups}
        assert got == lent, (req.context_remaining_length, got)
        lease.release()


@gpu
@pytest.mark.parametrize("scratch", [False, True], ids=["scratch_off", "scratch_on"])
@pytest.mark.parametrize("reuse", [False, True], ids=["resized", "committed"])
def test_a_read_before_the_context_ends_leaves_out_a_window_shorter_than_the_prompt(
    kit, real_manager, scratch, reuse
):
    """A mid-context read leaves out the 64-token window, so each step moves its history on."""
    with real_manager(
        windows=[WINDOW, 256], enable_block_reuse=reuse, swa_scratch_reuse=scratch
    ) as mgr:
        lender = attach(mgr)
        groups = range(kit.num_layer_groups(mgr))
        (windowed,) = [lg for lg, window in enumerate(kit.windows(mgr)) if window == WINDOW]
        request = kit.make_request(SOURCE, LONG_PROMPT)
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        assert kv.enable_swa_scratch_reuse == scratch

        def chunk(tokens):
            assert mgr.resize_context(request, tokens), f"no capacity for {tokens} tokens"
            step(mgr, request, tokens)

        chunk(2 * TPB)
        first = {lg: kit.pages(kv, lg)[:2] for lg in groups}
        assert all(min(first[lg]) >= 0 for lg in groups), f"inconclusive: {first}"
        lease = lender.lend_read(request, 0, 2 * TPB, **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        lent = slots_by_block(kit.ready_view(lease, mgr))
        longer = [(lg, block) for lg in groups if lg != windowed for block in range(2)]
        assert sorted(lent) == longer, f"lent {sorted(lent)}, not the longer window's {longer}"
        for _ in range(3):
            chunk(TPB)
        position = request.context_current_position
        stale = kit.stale_blocks(mgr, windowed, position)
        assert stale[1] > 0, f"inconclusive: at {position} the window passes no block, {stale}"
        assert kv.history_length == position, "the read held the history back"
        assert kit.pages(kv, windowed)[0] < 0, "the window kept block 0"
        chunk(len(LONG_PROMPT) - position)
        lease.release()
        assert kv.history_length == len(LONG_PROMPT), kv.history_length
        stale = kit.stale_blocks(mgr, windowed, kv.history_length)
        pages = kit.pages(kv, windowed)
        kept = {(windowed, block) for block in range(stale[1], len(pages)) if pages[block] >= 0}
        assert kept, f"inconclusive: the window keeps no backed block, stale {stale}, {pages}"
        after = lender.lend_read(request, 0, len(LONG_PROMPT), **kit.DEVICE_ALL)
        assert after.failure is None, after.failure
        lent = slots_by_block(kit.ready_view(after, mgr))
        assert {key for key in lent if key[0] == windowed} == kept, sorted(lent)
        after.release()


@gpu
def test_generation_holds_a_window_where_it_keeps_a_lent_block_until_the_release(kit, real_manager):
    """Generation holds a window's history where it keeps a lent block until the release."""
    from tensorrt_llm.bindings import LlmRequestState

    with real_manager(windows=[WINDOW, 256], enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        dev = kit.DevicePages(mgr)
        (windowed,) = [lg for lg, window in enumerate(kit.windows(mgr)) if window == WINDOW]
        # The largest history at which the window keeps block 0.
        keeping = max(h for h in range(4 * WINDOW) if kit.stale_blocks(mgr, windowed, h)[1] <= 0)
        request = kit.make_request(SOURCE, LONG_PROMPT[:keeping])
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        kv.enable_swa_scratch_reuse = False
        assert mgr.resize_context(request, request.context_remaining_length)
        for lg in range(kit.num_layer_groups(mgr)):
            for block in range(2):
                dev.write(lg, kit.pages(kv, lg)[block], bytes([0x30 + block]) * dev.page_bytes(lg))
        step(mgr, request, keeping)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        lease = lender.lend_read(request, 0, 2 * TPB, **kit.DEVICE_ALL)
        lent = slots_by_block(kit.ready_view(lease, mgr))
        assert (windowed, 0) in lent, f"inconclusive: lent {sorted(lent)}"
        for _ in range(3):
            generate(mgr, request)
        last = request.max_beam_num_tokens - 1
        stale = kit.stale_blocks(mgr, windowed, last)
        assert stale[1] > 0, f"inconclusive: at {last} the window passes no block, {stale}"
        assert kv.history_length == keeping, (last, keeping)
        assert kit.pages(kv, windowed)[0] == lent[(windowed, 0)], "the window let a lent page go"
        kit.overwrite_free_pages(mgr)
        assert holds(dev, lent, lambda ordinal: 0x30 + ordinal), "another request took a lent page"
        lease.release()
        generate(mgr, request)
        assert kv.history_length == request.max_beam_num_tokens - 1
        assert kit.pages(kv, windowed)[0] < 0, "the release left the window holding the block"


# -- beside a KV cache connector ----------------------------------------------------------------


@pytest.mark.cpu_only
def test_a_read_waits_for_a_load_the_connector_has_pending(kit, recorded):
    connector = Connector()
    manager = Manager(connector=connector)
    lender = lender_over(manager)
    manager.kv_cache_map[RID] = cache(history=2 * TPB)
    connector.loading.add(RID)
    lease = lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_ALL)
    assert lease.failure is None and recorded == []
    for _ in range(2):
        assert lease.poll() is None, "a view went out while the connector was loading"
    connector.loading.clear()
    view = lease.poll()
    assert view is not None and lease.poll() is view
    lease.release()
    # A lease taken with no load pending waits for its event alone.
    quiet = lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_ALL)
    connector.loading.add(RID)  # a load the connector starts later goes past what the cache holds
    assert quiet.poll() is not None
    quiet.release()


@pytest.mark.cpu_only
def test_a_read_fails_at_the_call_while_the_connector_holds_a_prefix_reservation(kit, recorded):
    """A read fails at the call while a prefix reservation counts unfilled tokens in the history."""
    connector = Connector()
    manager = Manager(connector=connector)
    lender = lender_over(manager)
    kv = manager.kv_cache_map[RID] = cache(history=2 * TPB)
    connector.reserved.add(RID)
    lease = lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_ALL)
    assert lease.failure is not None and "prefix reservation" in lease.failure, lease.failure
    assert not lender._is_locked(kv) and lease.poll() is None
    lease.release()
    connector.reserved.clear()
    lease = lender.lend_read(request(), 0, 2 * TPB, **kit.DEVICE_ALL)
    assert lease.failure is None and lender._is_locked(kv)
    lease.release()


class PastTheRule(Exception):
    """Raised by the first check of ``lend_write`` after the connector's rule."""


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", [None, "capacity_only", "loads"])
def test_a_fetch_starts_at_or_past_the_prefix_served_beside_a_loading_connector(
    kit, recorded, monkeypatch, kind
):
    """Beside a loading connector a fetch starts at or past the prefix served, else fails."""

    def past(self, request, end):
        raise PastTheRule()

    monkeypatch.setattr(_lender._ManagerLender, "_bidirectional_split_refusal", past)
    connector = None if kind is None else Connector(capacity_only=kind == "capacity_only")
    manager = Manager(connector=connector)
    lender = lender_over(manager)
    manager.kv_cache_map[RID] = cache(history=2 * TPB)
    served = request(served=2 * TPB)
    if kind == "loads":
        refused(
            lender.lend_write(served, TPB, PROMPT_LEN, **kit.DEVICE_ALL),
            "inside the 64 tokens served to the request",
        )
    else:
        with pytest.raises(PastTheRule):
            lender.lend_write(served, TPB, PROMPT_LEN, **kit.DEVICE_ALL)
    with pytest.raises(PastTheRule):
        lender.lend_write(served, 2 * TPB, PROMPT_LEN, **kit.DEVICE_ALL)


@pytest.mark.cpu_only
@pytest.mark.parametrize("window", [None, 64], ids=["full", "window"])
@pytest.mark.parametrize("rows", [NAMED, ALL], ids=["named", "all"])
@pytest.mark.parametrize("pending", [False, True], ids=["load_done", "load_pending"])
def test_a_fetch_cannot_advance_a_window_past_a_connectors_pending_destinations(
    kit, recorded, monkeypatch, window, rows, pending
):
    """A named fetch under a window fails at the call while a connector load is pending."""
    connector = Connector()
    if pending:
        connector.loading.add(RID)
    manager = Manager((window,), connector=connector)
    lender = lender_over(manager, layout((window,)))
    kv = manager.kv_cache_map[RID] = cache(history=2 * TPB)
    grown = []

    def grow(manager, request, cache, position, end):
        grown.append((position, end))
        cache.history_length = position
        return True

    monkeypatch.setattr(_manager, "grow", grow)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(_ranges, "_floor_follows_history", lambda manager: True)
    inputs_refuse_nothing(monkeypatch, lender)
    served = request(served=2 * TPB)
    before = lender.poll_resume_bounds(served)
    lease = lender.lend_write(served, 2 * TPB, 3 * TPB, **getattr(kit, f"DEVICE_{rows.name}"))
    try:
        if pending and rows is NAMED and window is not None:
            assert lease.failure is not None and "connector load" in lease.failure
            assert lease.poll() is None and not lender._is_locked(kv)
            assert not grown and kv.history_length == 2 * TPB
            assert lender.poll_resume_bounds(served) == before, "the fetch left a record"
        else:
            view = lease.poll()
            assert view is not None, lease.failure
            lender.return_writes([(lease, ALL_ARRIVED, True)])
            assert len(grown) == (rows is NAMED)
            assert kv.history_length == (3 * TPB if rows is NAMED and window else 2 * TPB)
    finally:
        lease.release()


# -- a window under a lock ----------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_last_history_keeping_a_block_is_the_last_whose_window_does():
    """Over random windows, sinks and blocks, the window keeps the block there, not one further."""
    rng = random.Random(0)
    for _ in range(2000):
        window = rng.randint(1, 300)
        sinks = rng.choice([0, 0, 1, 2])
        block = rng.randint(sinks, 40)
        manager = Manager((window,), (sinks,))
        lay = layout((window,), (sinks,))
        last = _ranges._last_history_keeping(manager, lay, 0, block)
        keeps = _ranges._stale_range(manager, lay, 0, last)[1] <= block
        passes = _ranges._stale_range(manager, lay, 0, last + 1)[1] > block
        assert keeps and passes, (window, sinks, block, last)


@pytest.mark.cpu_only
def test_the_history_cap_keeps_the_lowest_locked_block_past_the_sinks_in_every_window(recorded):
    """The cap keeps each window's lowest locked block past its sinks; other locks set none."""
    windows, sinks = (None, 64, 96), (0, 1, 0)
    manager = Manager(windows, sinks)
    lay = layout(windows, sinks)
    lender = lender_over(manager, lay)
    kv = cache(blocks=8)
    assert lender._history_cap(kv) is None
    lender._locks.lock(kv, {0: [0, 1], 1: [0]}, LockMode.READ)  # full attention, and a sink
    assert lender._history_cap(kv) is None
    lock = lender._locks.lock(kv, {1: [0, 2, 3], 2: [4, 5]}, LockMode.READ)

    def last(lg, block):
        return _ranges._last_history_keeping(manager, lay, lg, block)

    assert lender._history_cap(kv) == min(last(1, 2), last(2, 4))
    lender._locks.unlock(lock)
    lender._locks.lock(kv, {2: [1]}, LockMode.READ)
    assert lender._history_cap(kv) == last(2, 1)
    assert lender._history_cap(cache()) is None, "another cache's locks set the cap"


@pytest.mark.cpu_only
def test_a_named_fetch_whose_grow_would_pass_a_locked_block_fails_at_the_call(
    kit, recorded, monkeypatch
):
    """A named fetch whose grow would leave a locked window block behind fails before the grow."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _fetch_ledger

    windows = (64,)
    manager = Manager(windows)
    lay = layout(windows)
    lender = lender_over(manager, lay)
    kv = manager.kv_cache_map[RID] = cache(blocks=4, history=2 * TPB, committed=2 * TPB)
    grown = []
    monkeypatch.setattr(_manager, "grow", lambda *args: grown.append(args) or True)
    monkeypatch.setattr(_manager, "keeps_stale_pages", lambda manager: False)
    monkeypatch.setattr(
        _fetch_ledger._FetchLedger, "_unsettled_fetch_refusal", lambda self, rid, kv: None
    )
    inputs_refuse_nothing(monkeypatch, lender)
    held = lender._locks.lock(kv, {0: [0]}, LockMode.READ)
    end = 3 * TPB
    assert _ranges._stale_range(manager, lay, 0, end)[1] > 0, "inconclusive: block 0 stays kept"
    lease = lender.lend_write(request(), 2 * TPB, end, **kit.DEVICE_NAMED)
    assert lease.failure is not None and "leave a locked block behind" in lease.failure
    assert grown == [], "the cache grew for a fetch that failed at the call"
    lease.release()
    lender._locks.unlock(held)


# -- recurrent state ----------------------------------------------------------------------------


@gpu
def test_a_fetch_from_a_held_commit_endpoint_caps_attention_retry(kit, real_manager):
    """Unlock cannot publish a suffix while its replacement fetch remains unsettled."""
    with real_manager() as mgr:
        lender = attach(mgr)
        source = kit.make_request(SOURCE, PROMPT)
        assert mgr.prepare_context(source)
        kv = kit.kv(mgr, source)
        read = lender._locks.lock(kv, {0: [0]}, LockMode.READ)
        assert mgr.resize_context(source, 2 * TPB)
        step(mgr, source, 2 * TPB)
        assert source.context_current_position == 2 * TPB
        assert mgr._held_commits[id(kv)][2] == 2 * TPB
        assert mgr.resize_context(source, TPB)
        step(mgr, source, TPB)
        assert source.context_current_position == 3 * TPB
        assert mgr._held_commits[id(kv)][2] == 2 * TPB

        fetch = lender.lend_write(source, 2 * TPB, 3 * TPB, **kit.HOST_NAMED)
        try:
            assert fetch.failure is None, fetch.failure
            assert id(kv) in mgr._held_commits
            lender._locks.unlock(read)
            assert kv.num_committed_tokens == 2 * TPB
        finally:
            if lender._locks.is_locked(kv):
                lender._locks.unlock(read)
            fetch.release()


@gpu
def test_a_held_attention_commit_catches_up_without_a_fetch(kit, real_manager):
    """Unlock can publish the later computed prefix when nothing replaced it."""
    prompt = PROMPT[: 3 * TPB]
    with real_manager() as mgr:
        lender = attach(mgr)
        source = kit.make_request(SOURCE, prompt)
        assert mgr.prepare_context(source)
        kv = kit.kv(mgr, source)
        read = lender._locks.lock(kv, {0: [0]}, LockMode.READ)
        assert mgr.resize_context(source, 2 * TPB)
        step(mgr, source, 2 * TPB)
        assert mgr.resize_context(source, TPB)
        step(mgr, source, TPB)
        assert source.context_current_position == 3 * TPB
        assert kv.num_committed_tokens == 0
        lender._locks.unlock(read)
        assert kv.num_committed_tokens == 3 * TPB


@gpu
def test_a_state_group_s_regions_are_the_pools_native_names_and_its_row_the_state_slot(
    kit, hybrid_manager
):
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.page import CacheKind

    with hybrid_manager() as mgr:
        lender = attach(mgr)
        assert kit.staging_regions(lender) == ()
        lg = state_group(mgr)
        page_table = build_page_table_from_manager(mgr)
        native = page_table.layer_groups[lg]
        assert native.kind == CacheKind.STATE
        pools = sorted(
            page_table.pool_groups[native.pool_group_idx].pools, key=lambda p: int(p.base_address)
        )
        source = computed_to(mgr, kit.make_request(SOURCE, PROMPT), len(PROMPT))
        lease = lender.lend_read(source, 0, len(PROMPT), **kit.DEVICE_ALL)
        view = kit.ready_view(lease, mgr)
        run = next(g for g in view.groups if g.layer_group_index == lg)
        regions = sorted((lender.regions[r] for r in run.region_indices), key=lambda r: r.address)
        assert [(r.address, r.slot_bytes, r.slot_count) for r in regions] == [
            (int(p.base_address), int(p.slot_bytes), int(p.num_slots)) for p in pools
        ]
        slot = kit.kv(mgr, source).get_ssm_block_base_index(lg)
        assert run.page_ordinals.tolist() == [0] and run.slots.tolist() == [slot]
        assert run.object_keys is None
        # Each state tensor of the slot lies in the slot's bytes of one region: the kernels read
        # and write exactly what the row lends.
        for number, tensor in enumerate(state_tensors(mgr, slot)):
            tensor.view(torch.uint8).fill_(number + 1)
        torch.cuda.synchronize()
        parts = slot_parts(lender, run, slot)
        for number, tensor in enumerate(state_tensors(mgr, slot)):
            pointer = tensor.data_ptr()
            base, nbytes = next((b, n) for b, n in parts if b <= pointer < b + n)
            offset, size = pointer - base, tensor.numel() * tensor.element_size()
            assert offset + size <= nbytes
            got = device_tensor(base, nbytes)[offset : offset + size].cpu()
            assert bool((got == number + 1).all()), f"state tensor {number} is not where lent"
        lease.release()


@gpu
def test_a_lease_with_recurrent_state_ends_where_the_state_stands(kit, hybrid_manager):
    """A fetch with a state row that ends short of the prompt fails at the call."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        target, received = receiving(kit, mgr, TARGET, PROMPT, prepare=True)
        keywords = {**kit.DEVICE_ALL, "ready_event": received}
        refused(lender.lend_write(target, 0, 3 * TPB, **keywords), "where the state stands")


@gpu
def test_a_locked_state_row_holds_the_commit_that_would_snapshot_it_and_its_suspension(
    kit, hybrid_manager
):
    """A lock on the state row holds the snapshot commit and the suspension; the unlock commits."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        source = computed_to(mgr, kit.make_request(SOURCE, PROMPT), 2 * TPB)
        kv = kit.kv(mgr, source)
        source.expect_snapshot_points = [2 * TPB]  # the commit here snapshots the state
        lg = state_group(mgr)
        slot = kv.get_ssm_block_base_index(lg)
        lock = lender._locks.lock(kv, {lg: [0]}, LockMode.READ)
        with pytest.raises(RuntimeError, match="cannot be suspended"):
            mgr.suspend_request(source)
        mgr.try_commit_blocks(source)
        assert kv.num_committed_tokens == 0, "a commit snapshot the state a lock holds"
        assert kv.get_ssm_block_base_index(lg) == slot, "a held snapshot moved the locked row"
        assert source.context_current_position == kv.history_length == 2 * TPB
        lender._locks.unlock(lock)
        assert kv.num_committed_tokens == 2 * TPB, "the unlock left the snapshot commit undone"
        assert source.context_current_position == kv.history_length == 2 * TPB


@gpu
def test_a_stale_held_state_snapshot_is_not_committed_at_a_later_chunk(kit, hybrid_manager):
    """A recurrent snapshot cannot be committed after its live state has moved past it."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        source = computed_to(mgr, kit.make_request(SOURCE, PROMPT), 2 * TPB)
        kv = kit.kv(mgr, source)
        source.expect_snapshot_points = [2 * TPB]
        lg = state_group(mgr)
        first = lender._locks.lock(kv, {lg: [0]}, LockMode.READ)
        second = lender._locks.lock(kv, {lg: [0]}, LockMode.READ)

        mgr.try_commit_blocks(source)
        assert mgr._held_commits[id(kv)][2] == 2 * TPB
        computed_to(mgr, source, 3 * TPB)
        assert source.context_current_position == 3 * TPB
        assert kv.num_committed_tokens == 0

        lender._locks.unlock(first)
        assert lender._locks.is_locked(kv)
        assert id(kv) not in mgr._held_commits
        assert kv.num_committed_tokens == 0
        lender._locks.unlock(second)
        assert kv.num_committed_tokens == 0
        assert id(kv) not in mgr._held_commits
        assert source.context_current_position == kv.history_length == 3 * TPB


@gpu
def test_a_held_state_snapshot_uses_the_latest_live_snapshot_before_stopping(kit, hybrid_manager):
    """A final snapshot at the live history replaces an earlier locked snapshot."""
    with hybrid_manager() as mgr:
        lender = attach(mgr)
        source = computed_to(mgr, kit.make_request(SOURCE, PROMPT), 2 * TPB)
        kv = kit.kv(mgr, source)
        source.expect_snapshot_points = [2 * TPB, len(PROMPT)]
        lg = state_group(mgr)
        first = lender._locks.lock(kv, {lg: [0]}, LockMode.READ)
        second = lender._locks.lock(kv, {lg: [0]}, LockMode.READ)

        mgr.try_commit_blocks(source)
        computed_to(mgr, source, len(PROMPT))
        assert kv.num_committed_tokens == 0
        assert kv.history_length == len(PROMPT)
        lender._locks.unlock(first)
        assert id(kv) in mgr._held_commits
        assert kv.num_committed_tokens == 0
        lender._locks.unlock(second)
        assert kv.num_committed_tokens == len(PROMPT)
        assert id(kv) not in mgr._held_commits


@pytest.mark.cpu_only
def test_a_fetch_inside_a_held_snapshot_uses_the_recurrent_retry_position():
    """A recurrent fetch drops a held commit that could publish the current state."""
    from tensorrt_llm._torch.pyexecutor.kv_cache import _lender_hooks

    kv = SimpleNamespace(id=RID)
    source = request(computed=3 * TPB)
    mgr = SimpleNamespace(
        _held_commits={id(kv): (kv, source, 2 * TPB, False)},
        _held_commit_drop_end=lambda held: held[1].context_current_position,
    )

    _lender_hooks.drop_held_commit(mgr, kv, 3 * TPB)
    assert id(kv) in mgr._held_commits
    _lender_hooks.drop_held_commit(mgr, kv, 2 * TPB)
    assert id(kv) not in mgr._held_commits


# -- Helix --------------------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("rank", [0, 1])
def test_each_helix_rank_lends_its_pages_of_the_prompt_and_resumes_at_its_end(
    kit, real_manager, rank
):
    """Rank ``r`` lends its page of each 64-token block ``b`` as block ``2 * b + r``."""
    from tensorrt_llm._torch.pyexecutor.request_utils import partition_context_for_helix

    with real_manager(mapping=helix_rank(rank), enable_block_reuse=False) as mgr:
        lender = attach(mgr)
        assert kit.staging_regions(lender) == () and lender.regions
        # The rank's slice of the prompt, as the executor builds it, and the whole prompt's length.
        tokens, _, total, _ = partition_context_for_helix(PROMPT, rank, 2, TPB)
        request, admitted = receiving(kit, mgr, TARGET, tokens, total_input_len_cp=total)
        assert request.prompt_len < total == len(PROMPT), "inconclusive: no rank-local prompt"
        kv = kit.kv(mgr, request)
        assert kv.history_length == len(PROMPT)
        with pytest.raises(ValueError, match="lends nothing by name"):
            lender.lend_write(request, 0, 2 * TPB, **{**kit.DEVICE_NAMED, "ready_event": admitted})
        keywords = {**kit.DEVICE_ALL, "ready_event": admitted}
        with pytest.raises(ValueError, match="starts at a whole block of 64 tokens"):
            lender.lend_write(request, TPB, len(PROMPT), **keywords)
        lease = lender.lend_write(request, 0, len(PROMPT), **keywords)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        (run,) = view.groups
        expected = [p for p in range(4) if p % 2 == rank]
        assert run.page_ordinals.tolist() == expected
        pages = kit.pages(kv, 0)
        assert run.slots.tolist() == [pages[p // 2] for p in expected]
        bounds = return_write(lender, request, lease, view.row_masks(True))
        assert tuple(bounds) == (len(PROMPT), len(PROMPT))


@gpu
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_helix_rejects_a_window_on_another_real_pipeline_stage(real_manager, pp_rank):
    from tensorrt_llm.mapping import CpType, Mapping

    mapping = Mapping(
        world_size=4, rank=2 * pp_rank, cp_size=2, pp_size=2, cp_config={"cp_type": CpType.HELIX}
    )
    with real_manager(
        mapping=mapping, windows=[256, 64], num_layers=2, enable_block_reuse=False
    ) as mgr:
        local = mgr.max_attention_window_vec
        assert local == ([None] if pp_rank == 0 else [64]), (pp_rank, mgr.pp_layers, local)
        with pytest.raises(ValueError, match="Helix"):
            attach(mgr)
        assert mgr._sharing is None


@gpu
def test_helix_ignores_a_window_pattern_entry_past_the_model(real_manager):
    with real_manager(
        mapping=helix_rank(0), windows=[256, 64], num_layers=1, enable_block_reuse=False
    ) as manager:
        assert manager.pp_layers == [0] and manager.max_attention_window_vec == [None]
        assert manager._model_attention_window_pattern == (256, 64)
        assert attach(manager).regions


@gpu
@pytest.mark.parametrize("rank", [0, 1])
def test_helix_attaches_with_a_declared_read_window_and_full_kv_retention(kit, real_manager, rank):
    declared = {0: 64, 1: 64}
    with real_manager(
        mapping=helix_rank(rank),
        windows=[256],
        attention_read_windows=declared,
        enable_block_reuse=False,
    ) as manager:
        assert manager._model_attention_windows == declared
        assert manager.max_attention_window_vec == [None, None]
        layout = derive_layout(manager)
        assert layout.windows and all(window is None for window in layout.windows)
        lender = attach(manager)
        assert manager._sharing is lender
        assert lender.regions and kit.staging_regions(lender) == ()


# -- bytes a receive need not write -------------------------------------------------------------


@gpu
@pytest.mark.parametrize("mark_rows", [True, False], ids=["arrived", "not_arrived"])
def test_a_generation_only_draft_layer_can_remain_unwritten(kit, real_manager, mark_rows):
    """A receive leaving the draft layer's buffers unwritten resumes as its rows are marked."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec = MTPDecodingConfig(max_draft_len=1)
    spec.num_nextn_predict_layers = 1
    tokens = list(range(PROMPT_LEN))
    with (
        real_manager(num_layers=2) as source,
        real_manager(num_layers=2, spec_config=spec, enable_block_reuse=False) as destination,
    ):
        sender, receiver = attach(source), attach(destination)
        source_request = kit.published(source, 1, tokens)
        destination_request, ready = receiving(kit, destination, 2, tokens)
        assert kit.kv(destination, destination_request).history_length == PROMPT_LEN
        source_layers = {b.layer for b in sender.page_layout.buffers}
        destination_layers = {b.layer for b in receiver.page_layout.buffers}
        assert source_layers == {0, 1}, source_layers
        assert destination_layers == {0, 1, 2}, (destination_layers, destination.pp_layers)
        assert destination.max_draft_len == 1 and not destination.enable_block_reuse
        read = sender.lend_read(source_request, 0, PROMPT_LEN, **kit.DEVICE_ALL)
        keywords = {**kit.DEVICE_ALL, "ready_event": ready}
        write = receiver.lend_write(destination_request, 0, PROMPT_LEN, **keywords)
        bounds = None
        try:
            source_view = kit.ready_view(read, source)
            destination_view = kit.ready_view(write, destination)
            fill_buffers(receiver, destination_view, destination._stream, STALE)
            sent = {
                (b.layer, b.role, ordinal): tensor
                for b, ordinal, tensor in buffers(sender, source_view)
            }
            copied, omitted = [], []
            with torch.cuda.stream(destination._stream):
                for buffer, ordinal, tensor in buffers(receiver, destination_view):
                    peer = sent.get((buffer.layer, buffer.role, ordinal))
                    if peer is None:
                        assert buffer.layer == 2, (buffer, ordinal)
                        omitted.append(tensor)
                    else:
                        assert peer.numel() == tensor.numel(), (buffer, peer.shape, tensor.shape)
                        tensor.copy_(peer)
                        copied.append((tensor, peer))
            destination._stream.synchronize()
            assert copied and omitted, (len(copied), len(omitted))
            assert all(torch.equal(tensor, peer) for tensor, peer in copied)
            assert all(bool(torch.all(tensor == STALE)) for tensor in omitted)
            bounds = return_write(
                receiver, destination_request, write, destination_view.row_masks(mark_rows)
            )
        finally:
            write.release()
            read.release()
        assert bounds is not None, write.failure
        if mark_rows:
            assert bounds.max_position == PROMPT_LEN, (bounds, write.failure)
        else:
            assert bounds.max_position < PROMPT_LEN, (bounds, write.failure)


@gpu
@skip_pre_blackwell
def test_nvfp4_storage_padding_can_remain_unwritten(kit, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv("TRTLLM_NVFP4_MLA_RESIDUAL_QUANTIZATION", "1")
    prompt = 257
    with kit.managed(
        lambda: DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=128,
            max_seq_len=512,
            max_batch_size=2,
            max_input_len=512,
            max_num_tokens=1024,
            vocab_size=129280,
            mapping=Mapping(),
            dtype=DataType.NVFP4,
            compressor_dtype=DataType.FLOAT,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[128]
            ),
            execution_stream=torch.cuda.Stream(),
        )
    ) as manager:
        lender = attach(manager)
        request, ready = receiving(kit, manager, 3, list(range(prompt)))
        assert kit.kv(manager, request).history_length == prompt
        padding = [b for b in lender.page_layout.buffers if not b.transferred]
        assert padding and all("padding" in b.role for b in padding), padding
        lease = lender.lend_write(request, 0, prompt, **{**kit.DEVICE_ALL, "ready_event": ready})
        bounds = None
        try:
            view = kit.ready_view(lease, manager)
            fill_buffers(lender, view, manager._stream, STALE)
            written, omitted = [], []
            with torch.cuda.stream(manager._stream):
                for buffer, _, tensor in buffers(lender, view):
                    if buffer.transferred:
                        tensor.fill_(FRESH)
                        written.append(tensor)
                    else:
                        omitted.append(tensor)
            manager._stream.synchronize()
            assert written and omitted, (len(written), len(omitted), padding)
            assert all(bool(torch.all(tensor == FRESH)) for tensor in written)
            assert all(bool(torch.all(tensor == STALE)) for tensor in omitted)
            bounds = return_write(lender, request, lease, view.row_masks(True))
        finally:
            lease.release()
        assert bounds is not None and bounds.max_position == prompt, (bounds, lease.failure)


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("tail", [256, 262], ids=["at_a_page", "inside_a_page"])
def test_csa2_remote_tail_resumes_with_decoder_buffers_unwritten(kit, monkeypatch, tail):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    model = remote_tail_layout()
    prompt, split = tail + 128, 3
    with csa2_manager(kit, model, bounded_replay_on_generation=True) as manager:
        lender = attach(manager)
        request, ready = receiving(
            kit,
            manager,
            4,
            list(range(prompt)),
            py_csa2_remote_tail_mode="destination",
            py_csa2_remote_tail_start=tail,
            py_csa2_remote_tail_split=split,
        )
        cache = kit.kv(manager, request)
        assert cache.history_length == tail, (cache.history_length, tail, cache.capacity)
        assert manager.bounded_replay_on_generation and manager._decoder_replay_window > 0
        swa_layers = sorted({b.layer for b in lender.page_layout.buffers if b.role == "csa2_swa"})
        assert len(swa_layers) == len(model.compress_ratios), swa_layers
        decoder = set(swa_layers[split:])
        lease = lender.lend_write(request, 0, tail, **{**kit.DEVICE_ALL, "ready_event": ready})
        bounds = None
        try:
            view = kit.ready_view(lease, manager)
            fill_buffers(lender, view, manager._stream, STALE)
            written, omitted = [], []
            with torch.cuda.stream(manager._stream):
                for buffer, _, tensor in buffers(lender, view):
                    if buffer.role == "csa2_swa" and buffer.layer in decoder:
                        omitted.append(tensor)
                    else:
                        tensor.fill_(FRESH)
                        written.append(tensor)
            manager._stream.synchronize()
            assert written and omitted, (len(written), len(omitted), decoder)
            assert all(bool(torch.all(tensor == FRESH)) for tensor in written)
            assert all(bool(torch.all(tensor == STALE)) for tensor in omitted)
            bounds = return_write(lender, request, lease, view.row_masks(True))
        finally:
            lease.release()
        assert bounds == (tail, tail), (bounds, lease.failure, cache.history_length)


# -- CSA2 ---------------------------------------------------------------------------------------


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("policy", ["required", "hit", "partial-eviction", "evicted"])
def test_a_csa2_receive_of_every_row_lends_each_group_what_native_receives(
    kit, monkeypatch, policy
):
    """Each CSA2 group receives its retained pages beyond its independently restored prefix."""
    from tensorrt_llm._torch.disaggregation.resource.cache_reuse import _CacheReuseAdapterV2
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0" if policy == "required" else "1")
    tpb, prompt = 128, 257

    def request(request_id, kind):
        return LlmRequest(
            request_id=request_id,
            max_new_tokens=2,
            input_tokens=list(range(prompt)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
            llm_request_type=kind,
        )

    with csa2_manager(
        kit, remote_tail_layout(), reuse=True, max_batch_size=16, max_num_tokens=512 * 16
    ) as mgr:
        lender = attach(mgr)
        warm = request(SOURCE, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
        assert mgr.prepare_context(warm)
        assert mgr.resize_context(warm, 2 * tpb)
        cache = kit.kv(mgr, warm)
        cache.commit(list(range(2 * tpb)))
        optional = list(mgr._encoder_optional_groups)
        missing = {"partial-eviction": optional[:1], "evicted": optional}.get(policy, [])
        if missing:
            _introspection.drop_optional_pages(cache, 2 * tpb, missing)
        mgr.free_resources(warm)
        target = request(TARGET, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)
        assert mgr.prepare_disagg_gen_init(target)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == 2 * tpb, "inconclusive: no prefix on the generation side"
        assert [kv.reuse_status[g].complete for g in optional] == [
            g not in missing for g in optional
        ]
        ready = mgr.take_disagg_receive_ready(target)
        lease = lender.lend_write(target, 0, prompt, **{**kit.DEVICE_ALL, "ready_event": ready})
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        lent = {run.layer_group_index: run.page_ordinals.tolist() for run in view.groups}
        page_table = build_page_table_from_manager(mgr)
        cached = _CacheReuseAdapterV2(mgr).get_cached_token_count_per_layer_group(
            target, page_table.layer_groups
        )
        native = {
            lg: kept_page_ordinals(group.sliding_window_size, prompt, tpb, cached[lg])
            for lg, group in enumerate(page_table.layer_groups)
        }
        assert lent == native, (policy, lent, native, cached)
        bounds = return_write(lender, target, lease, view.row_masks(True))
        assert bounds.max_position == prompt


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("tail", [256, 262], ids=["at_a_block", "inside_a_block"])
def test_a_csa2_remote_tail_source_lends_what_native_sends_once_its_context_is_computed(
    kit, monkeypatch, tail
):
    """A completed remote-tail source lends each group's retained pages below the tail."""
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.bindings import SamplingConfig

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    tpb = 128
    prompt = tail + tpb  # the generation side computes the rest
    with csa2_manager(kit, remote_tail_layout(), bounded_replay_on_generation=True) as mgr:
        lender = attach(mgr)
        request = LlmRequest(
            request_id=SOURCE,
            max_new_tokens=1,
            input_tokens=list(range(prompt)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
        )
        request.py_csa2_remote_tail_mode = "source"
        request.py_csa2_remote_tail_start = tail
        assert mgr.prepare_context(request)
        assert request.context_remaining_length == tail, "inconclusive: no tail on this worker"
        computed_to(mgr, request, tail)
        assert request.context_remaining_length == 0 and request.prompt_len == prompt
        lease = lender.lend_read(request, 0, tail, **kit.DEVICE_ALL)
        assert lease.failure is None, lease.failure
        view = kit.ready_view(lease, mgr)
        lent = {run.layer_group_index: run.page_ordinals.tolist() for run in view.groups}
        page_table = build_page_table_from_manager(mgr)
        native = {
            lg: kept_page_ordinals(group.sliding_window_size, tail, tpb)
            for lg, group in enumerate(page_table.layer_groups)
        }
        assert lent == native, (lent, native)
        windowed = [
            lg for lg, d in enumerate(lender.page_layout.layer_groups) if d.kept_window is not None
        ]
        assert windowed and all(lent[lg] for lg in windowed), (windowed, lent)
        lease.release()


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize(
    "hole",
    [None, "margin", "csa2_swa", "csa2_compressor_kv"],
    ids=[
        "all_kept",
        "missing_margin_page",
        "missing_swa_read_page",
        "missing_compressor_read_page",
    ],
)
def test_a_real_csa2_receive_requires_its_kept_pages_to_resume(kit, monkeypatch, hole):
    """A receive resumes with all retained pages; an absent margin or read page bounds it."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    tpb, prompt = 128, 514
    model = CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=256, index_topk=4)
    spec = DSparkDecodingConfig(max_draft_len=8)
    with csa2_manager(kit, model, max_seq_len=1024, vocab_size=8192, spec_config=spec) as mgr:
        lender = attach(mgr)
        request, ready = receiving(kit, mgr, TARGET, list(range(prompt)))
        lease = lender.lend_write(request, 0, prompt, **{**kit.DEVICE_ALL, "ready_event": ready})
        view = kit.ready_view(lease, mgr)
        masks, margin_roles, missing = [], set(), 0
        for run in view.groups:
            desc = lender.page_layout.layer_groups[run.layer_group_index]
            roles = {b.role for b in lender.page_layout.buffers if b.layer in desc.layers}
            if "csa2_swa" in roles:
                read_window = model.window_size
            elif roles & {"csa2_compressor_kv", "csa2_compressor_score"}:
                read_window = desc.kept_window - mgr.max_draft_len - mgr.reuse_match_backoff
            else:
                read_window = None
            read_pages = set(kept_page_ordinals(read_window, prompt, tpb))
            kept_pages = set(kept_page_ordinals(desc.kept_window, prompt, tpb))
            lent = set(run.page_ordinals.tolist())
            assert read_pages <= kept_pages == lent, (roles, read_pages, kept_pages, lent)
            margin = kept_pages - read_pages
            if margin:
                margin_roles.update(roles)
            if hole == "margin" and margin and not missing:
                kept_pages.remove(min(margin))
                missing += 1
            elif hole in roles:
                kept_pages.remove(min(read_pages))
                missing += 1
            masks.append(np.isin(run.page_ordinals, list(kept_pages)))
        assert margin_roles == {"csa2_swa", "csa2_compressor_kv", "csa2_compressor_score"}
        assert (missing > 0) == (hole is not None)
        write_through(lender, view, FRESH, masks)
        bounds = return_write(lender, request, lease, masks)
        if hole is None:
            assert tuple(bounds) == (prompt, prompt)
        else:
            assert bounds.max_position < prompt or bounds.max_position < bounds.min_position
