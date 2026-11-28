# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's unified standalone draft cache (``ops/ssm/unified_draft_cache.py``), off the GPU.

The draft history contract, the draft storage cost, the manager's history and helix page arithmetic, the engine
methods the mixin overrides (and the engine's rewind, which keeps the drafter's history), and the deployment classmethod
``unified_draft_cache_kwargs`` (which drafter settings get draft layers, and their constructor arguments).
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.disaggregation.native import rank_info
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.unified_draft_cache import (  # noqa: E501
    DraftHistoryUpdate,
    StandaloneDraftHistory,
    UnifiedDraftCacheMixin,
    _draft_cache_kwargs,
    draft_layer_sizes,
    estimate_draft_cache_size_components,
)
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as engine_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

pytestmark = pytest.mark.cpu_only

# The K3.1 DFlash2 MLA drafter: five layers of one 576-wide latent per token.
MLA_LAYERS = 5
MLA_LATENT = 576


class _Manager(UnifiedDraftCacheMixin, KVCacheManagerV2):
    """The mixin over the engine manager, built without its constructor."""


# --- History contract -------------------------------------------------------


def test_draft_history_accepts_valid_ranges():
    StandaloneDraftHistory(0, 0)
    StandaloneDraftHistory(5, 5)
    StandaloneDraftHistory(5, 9)


@pytest.mark.parametrize(
    "valid_length, position, match",
    [
        (5.0, 9, "integer length and position"),
        (5, True, "integer length and position"),
        (-1, 9, "Invalid standalone draft history"),
        (5, 4, "Invalid standalone draft history"),
    ],
)
def test_draft_history_refuses_invalid_ranges(valid_length, position, match):
    with pytest.raises(ValueError, match=match):
        StandaloneDraftHistory(valid_length, position)


class _FakeManager:
    def __init__(self, request_ids):
        self.kv_cache_map = {rid: SimpleNamespace(is_active=True) for rid in request_ids}
        self.published = []

    def set_draft_history(self, request_id, valid_length, position):
        self.published.append((request_id, valid_length, position))


def test_draft_history_update_publishes_only_live_unreplaced_requests():
    manager = _FakeManager([1, 2, 3, 4])
    values = torch.tensor([[3, 10], [4, 11], [5, 12], [6, 13]])
    update = DraftHistoryUpdate.capture(manager, (1, 2, 3, 4), values)
    # The readback is a copy: later device writes do not change it.
    values.zero_()
    del manager.kv_cache_map[2]  # released
    manager.kv_cache_map[3] = SimpleNamespace(is_active=True)  # replaced
    manager.kv_cache_map[4].is_active = False  # suspended
    update.publish()
    assert manager.published == [(1, 3, 10)]


def test_update_resources_publishes_the_finished_forwards_oldest_first(monkeypatch):
    """A forward's draft history is published once that forward is done, before the engine's
    resource update; a later forward waits behind an earlier one that is still running."""
    calls = []
    monkeypatch.setattr(
        KVCacheManagerV2,
        "update_resources",
        lambda self, batch, attn_metadata=None, kv_cache_dtype_byte_size=None: calls.append(
            "engine"
        ),
    )
    manager = _manager()
    running, finished = [False], [True]

    def update(name, done):
        return SimpleNamespace(is_ready=lambda: done[0], publish=lambda: calls.append(name))

    manager.queue_draft_history_update(update("first", running))
    manager.queue_draft_history_update(update("second", finished))
    manager.update_resources(None)
    assert calls == ["engine"]
    running[0] = True
    manager.update_resources(None)
    assert calls == ["engine", "first", "second", "engine"]
    assert not manager._pending_draft_history


# --- Draft storage cost -----------------------------------------------------


def test_draft_cost_charges_history_and_scratch():
    sizes = (MLA_LATENT,) * MLA_LAYERS
    context, generation, per_request = estimate_draft_cache_size_components(
        sizes,
        32,
        generation_capacity_headroom=1,
        scratch_tokens=8,
    )
    # Full history: every rank-local token costs one token of each draft layer; scratch is per request.
    assert context == generation == sum(sizes)
    assert per_request == 8 * sum(sizes)


# --- The manager's history and helix pages ----------------------------------


def _manager(*, window=None, capacity=100, draft_layers=(61,)):
    manager = object.__new__(_Manager)
    manager.kv_cache_map = {7: SimpleNamespace(is_active=True, capacity=capacity)}
    manager.draft_history = {}
    manager._draft_dummy_request_ids = set()
    manager._pending_draft_history = deque()
    manager.draft_window_size = window
    manager._standalone_draft_layer_ids = tuple(draft_layers)
    return manager


def test_set_draft_history_checks_the_allocation():
    manager = _manager(window=16)
    manager.set_draft_history(7, 10, 40)
    assert manager.get_draft_history(7) == StandaloneDraftHistory(10, 40)
    with pytest.raises(ValueError, match="exceeds allocated capacity"):
        manager.set_draft_history(7, 10, 101)
    with pytest.raises(ValueError, match="retention window"):
        manager.set_draft_history(7, 17, 40)
    with pytest.raises(ValueError, match="no active cache"):
        manager.set_draft_history(8, 0, 0)
    manager.kv_cache_map[7].is_active = False
    with pytest.raises(ValueError, match="no active cache"):
        manager.set_draft_history(7, 0, 0)


def test_a_dummy_requests_draft_table_lists_the_pages_it_holds():
    """A padding or warmup request keeps no draft history and the drafter stores its tokens from column 0, so its row
    lists only the pages it holds; a request with a history keeps its positional row, released pages and all."""
    manager = _manager(window=128)
    manager.max_blocks_per_seq = 6
    manager._draft_dummy_request_ids = {9}
    held = [BAD_PAGE_INDEX, BAD_PAGE_INDEX, 5, 6]
    with patch.object(_Manager, "get_draft_block_indices", return_value=[held, held]):
        table = manager.get_draft_block_table([9, 7])
    assert table.tolist() == [[5, 6, 0, 0, 0, 0], [BAD_PAGE_INDEX, BAD_PAGE_INDEX, 5, 6, 0, 0]]


@pytest.mark.parametrize(
    "window, helix, history",
    # The window and the helix prompt length are independent: each case takes the other side of both.
    [(None, False, (30, 30)), (16, True, (16, 60))],
)
def test_disagg_gen_init_declares_the_prompt_as_draft_history(window, helix, history):
    """A generation server's request starts with the drafter history of the prompt it receives."""
    request = SimpleNamespace(py_request_id=7, prompt_len=30, total_input_len_cp=60)
    manager = _manager(window=window)
    manager._has_cp_helix = helix
    with patch.object(KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=False):
        assert not manager._prepare_disagg_gen_init_cache(request)
    assert manager.get_draft_history(7) is None
    with patch.object(KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=True):
        assert manager._prepare_disagg_gen_init_cache(request)
        assert manager.get_draft_history(7) == StandaloneDraftHistory(*history)
        without_draft = _manager(draft_layers=())
        without_draft._has_cp_helix = helix
        assert without_draft._prepare_disagg_gen_init_cache(request)
    assert without_draft.get_draft_history(7) is None


# --- The engine methods the mixin overrides ---------------------------------


def _seamed_manager(kv_factor=1):
    manager = _manager()
    manager._draft_num_kv_heads, manager._draft_head_dim = 1, MLA_LATENT
    manager._draft_dtype, manager._draft_kv_factor = DataType.FP8, kv_factor
    manager._draft_layer_sizes = (MLA_LATENT * kv_factor,) * 2
    manager._helix_cp_size = 1
    manager._standalone_draft_reserve = 8
    manager._standalone_draft_layer_ids = (61, 62)
    manager._unified_draft_layer_specs = {}
    for offset, global_id in enumerate((61, 62)):
        manager._unified_draft_layer_specs[40 + offset] = KVCacheLayerSpec(
            global_layer_id=global_id,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=DataType.FP8,
            num_kv_heads=1,
            head_dim=MLA_LATENT,
            buffer_roles=(Role.KEY, Role.VALUE)[:kv_factor],
        )
    return manager


def test_lifecycle_seams_keep_the_draft_history(monkeypatch):
    # The engine's own work in these calls is not under test.
    monkeypatch.setattr(
        KVCacheManagerV2, "_create_kv_cache", lambda self, *args, **kwargs: object()
    )
    monkeypatch.setattr(
        KVCacheManagerV2, "free_resources", lambda self, request, pin_on_release=False: None
    )
    monkeypatch.setattr(KVCacheManagerV2, "_shutdown_caches", lambda self: None)
    manager = _seamed_manager()
    manager.set_draft_history(7, 10, 40)
    manager._create_kv_cache(5, None, None, is_dummy=True)
    assert manager._draft_dummy_request_ids == {5}
    manager.free_resources(SimpleNamespace(py_request_id=7))
    manager.free_resources(SimpleNamespace(py_request_id=5))
    assert manager.get_draft_history(7) is None and not manager._draft_dummy_request_ids
    manager.set_draft_history(7, 1, 1)
    manager._shutdown_caches()
    assert manager.draft_history == {}


def test_every_allocation_covers_the_drafters_scratch():
    """Each allocation beyond a request's history, and the generation headroom the pools are sized with, include the
    drafter's next-forward scratch; without draft layers both are the engine's."""
    manager = _seamed_manager()
    manager.num_extra_kv_tokens, manager._shared_draft_block_width = 6, 0
    # The engine assigns its headroom once, in its constructor.
    manager._generation_kv_capacity_headroom = 9
    reserve = manager._standalone_draft_reserve
    assert manager._kv_alloc_extra_tokens == 6 + reserve
    assert manager._generation_kv_capacity_headroom == 9 + reserve
    assert manager._engine_generation_kv_capacity_headroom == 9
    manager._standalone_draft_reserve = 0
    assert manager._kv_alloc_extra_tokens == KVCacheManagerV2._kv_alloc_extra_tokens.fget(manager)
    assert manager._generation_kv_capacity_headroom == 9


def _sized_manager(draft_layers):
    manager = _seamed_manager() if draft_layers else _manager(draft_layers=())
    if not draft_layers:
        manager._draft_layer_sizes = ()
        manager._standalone_draft_reserve = 0
    manager._draft_registered_as_model_layers = False
    manager.tokens_per_block = 32
    manager.max_num_tokens = 256
    manager.max_batch_size, manager.gen_preemption_headroom = 4, 0
    manager.enable_swa_scratch_reuse = False
    manager.layer_specs = None
    manager._generation_kv_capacity_headroom = 2
    manager.draft_window_size = 128
    manager._get_runtime_cache_size_layer_components = lambda: ([100, 100], [None, 512])
    manager._get_runtime_cache_size_scratch_group_keys = lambda: None
    return manager


@pytest.mark.parametrize("quota", [200_000, 10_000_000])
def test_quota_conversions_cost_the_draft_layers(quota):
    """Without draft layers the quota conversions are the engine's. A context token also costs the 2 x 576 B draft
    layers; a generation token costs only the full-history 100 B target layer (the 512-token target window and the
    128-token draft window are paid per request); every request also holds its 8 scratch tokens in the target and
    draft layers. So a quota holds no more tokens."""
    plain = _sized_manager(draft_layers=False)
    assert plain._get_max_tokens_from_quota_impl(quota) == (
        KVCacheManagerV2._get_max_tokens_from_quota_impl(plain, quota)
    )
    assert plain._get_quota_from_max_tokens_impl(quota // 100) == (
        KVCacheManagerV2._get_quota_from_max_tokens_impl(plain, quota // 100)
    )
    manager = _sized_manager(draft_layers=True)
    context, generation, per_request = manager._runtime_cache_size_components()
    plain_context, plain_generation, plain_per_request = plain._runtime_cache_size_components()
    assert manager._draft_layer_sizes == (576, 576)
    assert (plain_context, plain_generation) == (200, 100)
    assert (context, generation) == (200 + 2 * 576, 100)
    assert per_request >= plain_per_request + 8 * 200 + 8 * 2 * 576
    assert manager._get_max_tokens_from_quota_impl(quota) <= plain._get_max_tokens_from_quota_impl(
        quota
    )
    tokens = manager._get_max_tokens_from_quota_impl(quota)
    if 0 < tokens < float("inf"):
        assert manager._get_quota_from_max_tokens_impl(int(tokens)) <= quota


class _PageBounds:
    def __init__(self, pages):
        self.pages = pages
        self.queried_layers = []

    def get_page_index_upper_bound(self, layer_id, role):
        self.queried_layers.append(int(layer_id))
        return self.pages[int(layer_id)]


class _HybridManager(UnifiedDraftCacheMixin, MambaHybridCacheManagerV2):
    """The mixin over the engine's hybrid manager, built without its constructor."""


@pytest.mark.parametrize(
    "layout_policy", [None, SimpleNamespace(blocks_in_primary_pool=property(lambda manager: 7))]
)
def test_free_blocks_count_the_model_layers(layout_policy):
    """The draft layers follow the model layers' ledger blocks, so a larger draft pool never sets the count."""
    seamed = _seamed_manager()
    manager = object.__new__(_HybridManager)
    manager._unified_draft_layer_specs = seamed._unified_draft_layer_specs
    manager.kv_cache_map = {engine_manager._GUARD_PAGE_REQUEST_ID: object()}
    manager._attention_cache_layout_policy = layout_policy
    manager.num_local_layers = 42
    manager.kv_factor = 1
    manager._ssm_page_index_scale = 1
    pages = [30] * 40 + [90, 90]  # 40 model layers, then the two draft layers
    manager.impl = _PageBounds(pages)
    manager._is_local_mamba_layer = lambda layer: layer < 30
    assert manager.get_num_free_blocks() == (30 if layout_policy is None else 7)
    assert manager.impl.queried_layers == (list(range(40)) if layout_policy is None else [])

    manager.impl.queried_layers.clear()
    manager.kv_cache_map[1] = object()
    with pytest.raises(AssertionError, match="kv cache manager is empty"):
        manager.get_num_free_blocks()
    assert manager.impl.queried_layers == []
    del manager.kv_cache_map[1]

    manager._unified_draft_layer_specs = {}
    assert manager.get_num_free_blocks() == (90 if layout_policy is None else 7)
    assert manager.impl.queried_layers == (list(range(42)) if layout_policy is None else [])


# --- The engine's rewind keeps the drafter's history and scratch -----------

MAX_DRAFT = 7
RESERVE = MAX_DRAFT + 1  # the drafter's next-forward scratch
WINDOW = 64
PROMPT = 100


class _Cache:
    """A request's cache: its capacity and whether it is resident."""

    def __init__(self, capacity=0):
        self.capacity, self.is_active = capacity, True

    def resize(self, capacity, history_length=None):
        if capacity is not None:
            self.capacity = capacity
        return True

    def resume(self, stream):
        self.is_active = True
        return True


class _Generation:
    """One request through the engine's generation allocation and rewind, as the scheduler and executor drive it.

    The engine's methods run unchanged. The drafter is modelled as its workers advance it: a forward moves the
    history's position by the target tokens it accepted (one plus the accepted drafts), and the manager publishes the
    forward's history once the forward is done (here at once: the earliest publish).
    """

    def __init__(self, monkeypatch, *, extra_kv_tokens, helix, disagg):
        monkeypatch.setattr(
            engine_manager, "_update_kv_cache_draft_token_location", lambda *args, **kwargs: None
        )
        manager = _manager(window=WINDOW)
        manager.kv_cache_type = CacheType.SELF
        manager.is_draft = False
        manager.num_extra_kv_tokens, manager._shared_draft_block_width = extra_kv_tokens, 0
        manager._standalone_draft_reserve = RESERVE
        manager._kv_reserve_draft_tokens = 0
        manager.max_total_draft_tokens = MAX_DRAFT
        manager._allocated_draft_lens, manager._inflight_draft_lens = {}, {}
        manager._has_cp_helix = helix
        manager.kv_compression_manages_history = False
        manager._sharing = None
        manager._stream = SimpleNamespace(cuda_stream=0)
        for name in (
            "_ensure_generation_beam_width",
            "_restore_page_index_bufs",
            "_fill_fresh_kv_pages",
            "_log_window_crossing",
            "_set_helix_rank_fields",
            "_after_shrink",
            "try_commit_generation_blocks",
        ):
            # The engine's work in these calls is not under test; a renamed one would leave a dead stub.
            assert hasattr(KVCacheManagerV2, name), name
            setattr(manager, name, lambda *args: True)
        self.manager, self.cache = manager, _Cache()
        manager.kv_cache_map = {7: self.cache}
        self.req = SimpleNamespace(
            py_request_id=7,
            py_draft_tokens=[],
            py_num_accepted_draft_tokens=0,
            py_rewind_len=0,
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            max_beam_num_tokens=PROMPT + 1,
            is_dummy_request=False,
            py_helix_decode_group_index=0,
            is_disagg_generation_transmission_complete=disagg,
            context_phase_params=SimpleNamespace(draft_tokens=None),
            py_disable_speculative_decoding=False,
            prompt_len=PROMPT // 4 if helix else PROMPT,  # helix requests hold their rank's slice
            total_input_len_cp=PROMPT,
        )
        if disagg:
            # The engine's generation-init allocation: the prompt, the drafts it arrives with and the allocation
            # beyond them; the mixin then declares the prompt as the draft history.
            monkeypatch.setattr(
                KVCacheManagerV2,
                "_prepare_disagg_gen_init_cache",
                lambda self, req: self.kv_cache_map[7].resize(
                    PROMPT + get_draft_token_length(req) + self._kv_alloc_extra_tokens
                ),
            )
            assert manager._prepare_disagg_gen_init_cache(self.req)
        else:
            # The last context chunk's allocation (resize_context), and the drafter's context store up to it.
            self.cache.capacity = PROMPT + manager._kv_alloc_extra_tokens
            manager.set_draft_history(7, min(PROMPT, WINDOW), PROMPT)
        self.position = manager.get_draft_history(7).position
        self.in_flight = []
        self.margins = []

    def schedule(self, width=None, padded=None):
        """The scheduler grows the request for a forward verifying ``width`` drafts, then CUDA-graph padding widens
        it to ``padded``. Without a width it is the first step after a disaggregated handoff: the generation server
        reserves its own draft width for the placeholder drafts."""
        self.req.py_draft_tokens = [] if width is None else [0] * width
        assert self.manager.try_allocate_generation(self.req)
        self.req.is_disagg_generation_transmission_complete = False
        if width is None:
            self.req.py_draft_tokens = [0] * MAX_DRAFT
        if padded is not None:
            self.req.py_draft_tokens = [0] * padded
            self.manager.extend_capacity_for_tokens(self.req)

    def revert(self):
        """Attention DP skips the forward: the scheduler gives the growth back."""
        self.manager.revert_allocate_generation(self.req)

    def suspend(self):
        """The next iteration's scheduler suspends the request before this iteration's update."""
        self.cache.is_active = False

    def forward(self, accepted):
        assert accepted <= get_draft_token_length(self.req)
        self.position += 1 + accepted
        self.in_flight.append(accepted)
        values = torch.tensor([[min(self.position, WINDOW), self.position]])
        self.manager.queue_draft_history_update(
            DraftHistoryUpdate(self.manager, (7,), (self.cache,), values)
        )

    def update(self):
        """The executor's resource update of the oldest forward in flight: the queued histories, then the rewind."""
        self.req.py_num_accepted_draft_tokens = self.in_flight.pop(0)
        self.manager.update_resources(SimpleNamespace(generation_requests=[self.req]))
        history = self.manager.get_draft_history(7)
        self.margins.append(self.cache.capacity - (history.position + RESERVE))


# Each step: (scheduled width, padded width, accepted drafts). ``None`` widths are the first step after a
# disaggregated handoff. A forward verifies the width it was allocated with: the Kimi K3.1 targets refuse draft-length
# schedules (routing), so no step accepts more drafts than its allocation covers.
_STEPS = {
    "max_rejected": [(MAX_DRAFT, None, 0)] * 4,
    "all_accepted": [(MAX_DRAFT, None, MAX_DRAFT)] * 4,
    "width_changes": [(MAX_DRAFT, None, 0), (0, None, 0), (3, None, 3), (0, None, 0), (2, None, 0)],
    "padded": [(2, MAX_DRAFT, 0), (0, MAX_DRAFT, MAX_DRAFT), (5, MAX_DRAFT, 0)],
    "rejected_after_accepted": [(MAX_DRAFT, None, MAX_DRAFT), (MAX_DRAFT, None, 0)] * 2,
}


@pytest.mark.parametrize(
    "overlap, extra_kv_tokens, disagg, helix",
    [(False, 0, False, False), (True, MAX_DRAFT - 1, True, True)],
    ids=["plain", "overlap-extra_kv-disagg-helix"],
)
@pytest.mark.parametrize("steps", list(_STEPS))
def test_the_rewind_keeps_the_draft_history_and_scratch(
    monkeypatch, steps, overlap, extra_kv_tokens, disagg, helix
):
    """After every rewind the request still holds its published draft history and the drafter's next-forward scratch.

    Without the overlap scheduler the rewind leaves exactly ``num_extra_kv_tokens`` beyond them (the worst case,
    every draft rejected, included). With it, the next forward's allocation is still held when the earlier forward
    is rewound, and covers that forward's accepted tokens even when its history is published first.
    """
    run = _Generation(monkeypatch, extra_kv_tokens=extra_kv_tokens, helix=helix, disagg=disagg)
    plan = list(_STEPS[steps])
    if disagg:
        plan[0] = (None, None, 0)  # placeholder drafts, all rejected
    for width, padded, accepted in plan:
        run.schedule(width, padded)
        run.forward(accepted)
        if not overlap or len(run.in_flight) == 2:
            run.update()
    while run.in_flight:
        run.update()
    assert min(run.margins) == extra_kv_tokens
    if not overlap:
        assert set(run.margins) == {extra_kv_tokens}


def test_the_rewind_keeps_the_draft_history_through_reverts_and_suspends(monkeypatch):
    """Attention DP skipping a forward (its growth reverted) and a suspend before the update (the rewind and the
    history publish skipped) only leave more capacity beyond the draft history."""
    run = _Generation(monkeypatch, extra_kv_tokens=0, helix=False, disagg=False)
    run.schedule(MAX_DRAFT)
    run.forward(0)
    run.schedule(MAX_DRAFT)
    run.revert()
    run.update()
    run.schedule(MAX_DRAFT)
    run.forward(MAX_DRAFT)
    run.suspend()
    run.update()
    run.schedule(MAX_DRAFT)  # resumes the request
    run.forward(0)
    run.update()
    assert run.margins[0] == 0 and min(run.margins) >= 0


# --- The draft layers stay out of the target's lifecycles and pools --------


def _draft_init_kwargs(**overrides):
    kwargs = dict(
        draft_num_layers=MLA_LAYERS,
        draft_num_kv_heads=1,
        draft_head_dim=MLA_LATENT,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=8,
        draft_window_size=1024,
        layer_specs=None,
        is_draft=False,
    )
    kwargs.update(overrides)
    return kwargs


def _built_manager(layer_grouping, pool_group_lifecycles):
    """A manager whose native layout is ``layer_grouping`` (lifecycle -> local layers) in pool groups of
    ``pool_group_lifecycles`` (pool group -> lifecycles); local layers 3 and 4 are draft layers."""
    manager = object.__new__(_Manager)
    manager._standalone_draft_layer_ids = (61, 62)
    manager._unified_draft_layer_specs = dict.fromkeys((3, 4))
    manager.draft_window_size = 1024
    manager.impl = SimpleNamespace(
        layer_grouping=layer_grouping,
        pool_group_descs=[
            SimpleNamespace(
                slot_desc=SimpleNamespace(
                    variants=[SimpleNamespace(layer_group_id=lc) for lc in lifecycles]
                )
            )
            for lifecycles in pool_group_lifecycles
        ],
    )
    return manager


def test_the_built_manager_keeps_draft_layers_in_pool_groups_of_their_own():
    # Draft lifecycle 2 in a pool group of its own; target lifecycles 0 (MLA) and 1 (KDA) share another.
    _built_manager([[0, 1], [2], [3, 4]], [[2], [0, 1]])._check_standalone_draft_isolation()
    # Without a window the draft layers would join the target's lifecycle.
    with pytest.raises(
        RuntimeError, match=r"pool group 0: draft layers \[3, 4\], other layers \[0, 1\]"
    ):
        _built_manager([[0, 1, 3, 4], [2]], [[0], [1]])._check_standalone_draft_isolation()
    # A draft lifecycle of the same page sizes as a target lifecycle would share its pool group.
    with pytest.raises(RuntimeError, match="share pool groups"):
        _built_manager([[0, 1], [2], [3, 4]], [[0, 2], [1]])._check_standalone_draft_isolation()
    # A manager without draft layers is not checked.
    manager = _built_manager([[0, 1, 3, 4], [2]], [[0], [1]])
    manager._standalone_draft_layer_ids = ()
    manager._check_standalone_draft_isolation()


# --- Disaggregated transfer of the draft layers -----------------------------


def test_a_draft_layer_reports_the_drafters_dtype_to_its_peers(monkeypatch):
    """Disaggregated peers compare the dtype of the draft layers they publish (``RankInfo.draft_kv_dtype``): a draft
    layer reports the drafter's storage dtype and a model layer the manager's, so two servers whose drafters keep
    their history in different dtypes do not register as peers."""
    manager = _seamed_manager()
    manager._draft_dtype, manager.dtype = DataType.BF16, DataType.FP8
    manager.layer_specs = None
    manager._appended_draft_layer_specs = {}
    manager.layer_offsets = {0: 0, 61: 40, 62: 41}
    assert manager.get_layer_dtype(61) == manager.get_layer_dtype(62) == DataType.BF16
    assert manager.get_layer_dtype(0) == DataType.FP8

    # What a context or generation rank publishes at registration (its page table needs the pools).
    monkeypatch.setattr(rank_info, "build_page_table_from_manager", lambda kvm: None)
    manager.mapping = Mapping()
    manager.num_kv_heads_per_layer = [1] * 42
    manager.pp_layers = list(range(40))
    manager.tokens_per_block, manager.head_dim, manager.kv_factor = 64, MLA_LATENT, 1
    published = rank_info.RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert (published.num_draft_layers, published.draft_kv_dtype) == (2, DataType.BF16.name)


class _InitEngine:
    """An engine constructor that leaves a built manager with draft layer 1 in a pool group of its own."""

    def __init__(self, *, helix_cp_size, **kwargs):
        self._helix_cp_size = helix_cp_size
        self._standalone_draft_layer_ids = (61,)
        self._unified_draft_layer_specs = dict.fromkeys((1,))
        self.enable_swa_scratch_reuse = False
        self.impl = SimpleNamespace(
            layer_grouping=[[0], [1]],
            pool_group_descs=[
                SimpleNamespace(
                    slot_desc=SimpleNamespace(variants=[SimpleNamespace(layer_group_id=group)])
                )
                for group in (0, 1)
            ],
        )


class _InitManager(UnifiedDraftCacheMixin, _InitEngine):
    pass


@pytest.mark.parametrize("draft_cp_size, refused", [(8, False), (1, True)])
def test_helix_draft_pages_follow_the_targets(draft_cp_size, refused):
    """Disaggregated transfer moves the draft pages as the target's: one page per rank and ledger block."""
    kwargs = {
        key: value
        for key, value in _draft_init_kwargs().items()
        if key not in ("layer_specs", "is_draft")
    }
    kwargs.update(helix_cp_size=8, draft_cp_size=draft_cp_size)
    if refused:
        with pytest.raises(NotImplementedError, match="sharded like the target's pages"):
            _InitManager(**kwargs)
    else:
        assert _InitManager(**kwargs).has_standalone_draft_cache


# --- The deployment: which drafter settings get draft layers ----------------


def _spec():
    return SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: True),
        _use_shared_kv_cache=False,
        max_draft_len=7,
        attention_backend="VANILLA",
    )


def _mla_drafter(
    *,
    helix_mapping=None,
    context_kv_fp8=True,
    aggregated=None,
    context_kv_only=None,
):
    extra = {
        name: value
        for name, value in (
            ("unified_draft_cache_when_aggregated", aggregated),
            ("context_kv_only", context_kv_only),
        )
        if value is not None
    }
    return SimpleNamespace(
        **extra,
        _helix_mapping=helix_mapping,
        _kv_factor=1,
        _num_attn_layers=MLA_LAYERS,
        _head_dim=MLA_LATENT,
        fc=SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16)),
        context_kv_fp8=context_kv_fp8,
    )


def _draft_config(mla=True):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            dflash_config={"attention_mode": "mla" if mla else "gqa"}
        ),
        mapping=Mapping(),
    )


_HELIX = SimpleNamespace(has_cp_helix=lambda: True, helix_repurposed_cp_size=1)


def _declare(*, draft, disagg=True, mapping=None, mla=True, kv_cache_config=None):
    return UnifiedDraftCacheMixin.unified_draft_cache_kwargs(
        model_engine=SimpleNamespace(model=SimpleNamespace(draft_model=draft)),
        draft_config=_draft_config(mla),
        spec_config=_spec(),
        mapping=mapping or Mapping(),
        kv_cache_config=kv_cache_config,
        is_disagg=disagg,
    )


def test_an_aggregated_drafter_keeps_its_context_private():
    # Without helix or the aggregated opt-in the draft layers would only carry the context across the
    # cache transceiver.
    assert _declare(draft=_mla_drafter(), disagg=False) == {}


def test_a_context_only_drafter_hands_its_context_over_in_the_draft_layers():
    """A context-only drafter's context reaches the generation server in the draft layers."""
    kwargs = _declare(draft=_mla_drafter(context_kv_only=True))
    assert kwargs["draft_num_layers"] == MLA_LAYERS


@pytest.mark.parametrize("disagg", [False], ids=["aggregated"])
def test_a_helix_drafter_keeps_its_context_in_the_draft_layers(disagg):
    """Under helix the MLA drafter's context is always in the draft layers."""
    assert _declare(draft=_mla_drafter(), mapping=_HELIX, disagg=disagg)["draft_num_layers"]


def test_an_fp8_drafter_stores_fp8_context_kv():
    # The dtype follows context_kv_fp8 alone; a context-only FP8 drafter runs in
    # test_a_context_only_drafter_hands_its_context_over_in_the_draft_layers.
    draft = _mla_drafter(context_kv_only=False)
    assert _declare(draft=draft)["draft_dtype"] == DataType.FP8


# --- An MLA drafter that keeps its context in the unified draft cache when aggregated ---------

_AGGREGATED = dict(disagg=False)
_REPURPOSED = SimpleNamespace(has_cp_helix=lambda: False, helix_repurposed_cp_size=8)
_DISTRIBUTED = {
    "disagg": dict(disagg=True),
    "helix": dict(disagg=False, mapping=_HELIX),
    "helix_repurposed": dict(disagg=False, mapping=_REPURPOSED),
}


def test_an_aggregated_mla_drafter_that_opts_in_gets_draft_layers():
    kwargs = _declare(draft=_mla_drafter(aggregated=True), **_AGGREGATED)
    assert (kwargs["draft_num_layers"], kwargs["draft_kv_factor"]) == (MLA_LAYERS, 1)
    assert (kwargs["draft_cp_size"], kwargs["draft_head_dim"]) == (1, MLA_LATENT)


@pytest.mark.parametrize("kind, aggregated", [("disagg", None), ("helix_repurposed", True)])
def test_a_distributed_mla_drafter_gets_draft_layers_either_way(kind, aggregated):
    """The drafter's context is in the draft layers under disaggregated serving, where it crosses the
    cache transceiver whatever the drafter sets, and on a repurposed helix CP with the aggregated opt-in."""
    draft = _mla_drafter(aggregated=aggregated)
    kwargs = _declare(draft=draft, **_DISTRIBUTED[kind])
    assert kwargs["draft_num_layers"] == MLA_LAYERS


@pytest.mark.parametrize(
    "helix_mapping, draft_cp_size, context_kv_fp8, dtype",
    # The CP size and the dtype are set apart: each case takes the other side of both.
    [
        (None, 1, True, DataType.FP8),
        (SimpleNamespace(has_cp_helix=lambda: True, cp_size=8), 8, False, DataType.BF16),
    ],
    ids=["no_helix-fp8", "helix8-bf16"],
)
def test_draft_cache_kwargs_of_an_mla_drafter(helix_mapping, draft_cp_size, context_kv_fp8, dtype):
    kwargs = _draft_cache_kwargs(
        _mla_drafter(helix_mapping=helix_mapping, context_kv_fp8=context_kv_fp8),
        _draft_config(),
        _spec(),
    )
    assert kwargs == dict(
        draft_num_layers=MLA_LAYERS,
        draft_kv_factor=1,
        draft_cp_size=draft_cp_size,
        draft_num_kv_heads=1,
        draft_head_dim=MLA_LATENT,
        draft_dtype=dtype,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=8,  # max_draft_len + 1
    )
    # The manager and its static estimator take the same kwargs.
    latent_bytes = MLA_LATENT if context_kv_fp8 else 2 * MLA_LATENT
    assert draft_layer_sizes(**kwargs) == (latent_bytes,) * MLA_LAYERS
