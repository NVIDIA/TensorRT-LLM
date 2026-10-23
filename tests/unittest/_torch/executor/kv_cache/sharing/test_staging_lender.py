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
"""The lender's contract over real KV cache managers; a few oracles read what the API hides."""

import gc
import mmap
import threading
import time
import weakref
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    ALL_ARRIVED,
    Lease,
    LeaseView,
    Lender,
    MemoryKind,
    RegionHold,
    ResumeBounds,
    StagingOptions,
    attach_lender,
)
from tensorrt_llm._utils import prefer_pinned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
OTHER_PROMPT = list(range(5000, 5097))
END = 96
BLOCKS = END // TPB
WINDOWED_PROMPT = list(range(2000, 2161))  # five whole blocks and one token
OTHER_WINDOWED_PROMPT = list(range(6000, 6161))
WINDOWED_END = 160
SOURCE, TARGET, OTHER = 1, 2, 3
SCOPE = b"lender-contract-suite"


def attach(mgr, *, guaranteed_fetch_tokens, capacity_in_fetches=1, max_bytes=None, scope=SCOPE):
    """The public attach, as an integrator makes it."""
    options = StagingOptions(guaranteed_fetch_tokens, capacity_in_fetches, max_bytes)
    return attach_lender(mgr, scope=scope, staging=options)


def attach_breaking(rules):
    """An attach installing a lender whose ``rules`` (method name -> function, or a function of the
    manager returning them) replace the real ones, on the lender or on its fetch ledger, whichever
    defines the method: a lender breaking exactly those rules."""

    def attach_rule_breaker(
        mgr, *, guaranteed_fetch_tokens, capacity_in_fetches=1, max_bytes=None, scope=SCOPE
    ):
        from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _fetch_ledger, _lender

        chosen = dict(rules(mgr) if callable(rules) else rules)
        on_ledger = {n: f for n, f in chosen.items() if _on_ledger(n)}
        lender_rules = {n: f for n, f in chosen.items() if n not in on_ledger}
        cls = type("RuleBreaker", (_lender._ManagerLender,), lender_rules)
        options = StagingOptions(guaranteed_fetch_tokens, capacity_in_fetches, max_bytes)
        lender = _lender._attach_lender(mgr, scope=scope, staging=options, cls=cls)
        if on_ledger:
            # A ledger of the rule-breaking class in place of the real one, before it records
            # anything; its rules reach the lender through _lender_ref.
            ledger = type("LedgerRuleBreaker", (_fetch_ledger._FetchLedger,), on_ledger)
            lender._ledger = ledger(lender._layout, lender._ledger._copy_done)
            lender._ledger._lender_ref = weakref.ref(lender)
        return lender

    return attach_rule_breaker


def _on_ledger(name):
    """Whether rule ``name`` is a method of the lender's fetch ledger rather than of the lender."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _fetch_ledger

    return not name.startswith("__") and name in vars(_fetch_ledger._FetchLedger)


def by_group(view):
    return {run.layer_group_index: run.page_ordinals.tolist() for run in view.groups}


def one_fetch_bytes(kit, mgr, guaranteed_fetch_tokens):
    """Bytes one fetch of ``guaranteed_fetch_tokens`` tokens takes: a window's blocks for a window."""
    dev = kit.DevicePages(mgr)
    total = 0
    for lg, window in enumerate(kit.windows(mgr)):
        blocks = -(-guaranteed_fetch_tokens // TPB)
        if window is not None:
            blocks = min(blocks, -(-window // TPB))
        total += blocks * dev.page_bytes(lg)
    return total


def check_staged(kit, mgr, lender, request, view):
    """Every row's slot, inside its region, holds the request's device page of that block."""
    kv = kit.kv(mgr, request)
    dev = kit.DevicePages(mgr)
    rows = 0
    for run in view.groups:
        region = lender.regions[run.region_index]
        assert run.layer_group_index in lender.regions[run.region_index].layer_group_indices
        assert region.slot_bytes == dev.page_bytes(run.layer_group_index)
        slots = kit.pages(kv, run.layer_group_index)
        for address, ordinal in zip(run.addresses.tolist(), run.page_ordinals.tolist()):
            assert region.address <= address < region.address + region.nbytes
            assert (address - region.address) % region.slot_bytes == 0
            device = kit.digest(dev.read(run.layer_group_index, slots[ordinal]))
            assert kit.digest(kit.host_bytes(address, region.slot_bytes)) == device
            rows += 1
    return rows


def state_of(kit, mgr, lender, request):
    """Capacity, history, committed tokens and resume_bounds of the request, or what stands for none."""
    kv = kit.kv(mgr, request)
    try:
        resume_bounds = lender.poll_resume_bounds(request)
    except ValueError:
        resume_bounds = "no cache"
    if kv is None:
        return None, resume_bounds
    return (kv.capacity, kv.history_length, kv.num_committed_tokens), resume_bounds


def return_and_settle(lender, mgr, request, lease, arrived):
    """Return ``lease`` with ``arrived``, run its copy and a round of progress: the bounds then."""
    (given,) = lender.return_writes([(lease, arrived, True)])
    mgr._stream.synchronize()
    lender.progress()
    bounds = lender.poll_resume_bounds(request)
    assert given is None or given == bounds, f"the return gave {given}, then {bounds}"
    return bounds


def check_names_nothing(kit, lender, words, refused=None, in_place=None, mgr=None, source=None):
    """Nothing staged; named lends refuse, while a cache-backed source lends device rows."""
    assert lender.regions and kit.staging_regions(lender) == ()
    assert kit.staging_memory(lender.regions) == []
    missing_id = SOURCE if source is None else SOURCE + 10_000
    request = kit.make_request(missing_id, PROMPT)  # never admitted: no cache
    for lend in (lender.lend_read, lender.lend_write):
        for keywords in refused or (kit.HOST_NAMED, kit.DEVICE_NAMED):
            message = refusal(lambda: lend(request, 0, TPB, **keywords))
            assert all(word in message for word in words), message
    lease = lender.lend_read(request, 0, TPB, **(in_place or kit.DEVICE_ALL))
    assert lease.failure == f"request {missing_id} has no KV cache", lease.failure
    lease.release()
    if source is not None:
        lease = lender.lend_read(source, 0, TPB, **(in_place or kit.DEVICE_ALL))
        assert kit.ready_view(lease, mgr).num_rows > 0, lease.failure
        lease.release()


# -- attach -----------------------------------------------------------------------------------


def test_a_manager_takes_one_lender_for_its_life(real_manager):
    with real_manager() as mgr:
        for scope in ("text", 1, bytearray(SCOPE)):  # bytes() would take the last two
            with pytest.raises(TypeError, match="scope must be bytes"):
                attach_lender(mgr, scope=scope, staging=StagingOptions(END))
        with pytest.raises(TypeError, match="staging must be StagingOptions"):
            attach_lender(mgr, scope=SCOPE, staging=END)
        lender = attach(mgr, guaranteed_fetch_tokens=END)  # the refused calls attached nothing
        assert isinstance(lender, Lender)
        with pytest.raises(ValueError, match="already attached"):
            attach(mgr, guaranteed_fetch_tokens=END)


def test_a_manager_holding_recurrent_state_gets_a_lender_naming_nothing(kit, hybrid_manager):
    """No name covers recurrent state, which the facade reads from the manager's kind."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with hybrid_manager() as mgr:
        assert _manager.model_has_recurrent_state(mgr)
        check_names_nothing(kit, attach(mgr, guaranteed_fetch_tokens=256), ["recurrent state"])


def sparse_layer(monkeypatch, index):
    """Managers built from now on mark the buffers of their layer ``index`` sparse, which no model
    here does: the runtime may then lock that layer group's read-only pages in host memory."""
    from dataclasses import replace

    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

    build = KVCacheManagerV2._build_cache_config

    def with_a_sparse_layer(self, config):
        layers = list(config.layers)
        layer = layers[index]
        layers[index] = AttentionLayerConfig(
            layer_id=layer.layer_id,
            buffers=[
                BufferConfig(b.role, b.size, b.tokens_per_block_override, is_sparse=True)
                for b in layer.buffers
            ],
            sliding_window_size=layer.sliding_window_size,
            num_sink_tokens=layer.num_sink_tokens,
        )
        return build(self, replace(config, layers=layers))

    monkeypatch.setattr(KVCacheManagerV2, "_build_cache_config", with_a_sparse_layer)


# case: (manager keywords, the layer whose buffers are sparse); "later" puts them in a layer group
# after a dense one
SPARSE_CASES = {"first": ({}, 0), "later": ({"windows": [WINDOW, 256]}, 1)}


@pytest.mark.parametrize("case", list(SPARSE_CASES))
def test_a_layer_group_of_sparse_buffers_is_refused(kit, host_tier_manager, monkeypatch, case):
    """A cache can lock a sparse layer group's read-only pages in host memory, where a page index
    names a host slot that staging would address as a device slot: the attach refuses such a
    manager, naming the group and keeping nothing, also where it follows a dense one. The runtime
    builds sparse buffers only with host memory as the cache level below the GPU, as here."""
    keywords, index = SPARSE_CASES[case]
    sparse_layer(monkeypatch, index)
    before = len(kit.retained())
    with host_tier_manager(**keywords) as mgr:
        layers = {int(layer.layer_id): layer for layer in mgr.impl.init_config.layers}
        flags = [
            any(b.is_sparse for b in layers[int(ids[0])].buffers) for ids in mgr.impl.layer_grouping
        ]
        sparse = [lg for lg, flag in enumerate(flags) if flag]
        assert sparse and (0 in sparse) == (case == "first"), (
            f"sparse buffers in layer groups {sparse}: the check proves nothing"
        )
        message = refusal(lambda: attach(mgr, guaranteed_fetch_tokens=END))
        assert f"layer groups {sparse} hold sparse buffers" in message
        assert "_sharing" not in vars(mgr)
    assert len(kit.retained()) == before, "a refused attach kept something"


def stand_in(cp_type=None):
    """The smallest manager the attach path reads before any other work: an uninitialised
    ``KVCacheManagerV2`` whose mapping is rank 0 of two ``cp_type`` ranks, or has no mapping."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.mapping import Mapping

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    if cp_type is not None:
        manager.mapping = Mapping(world_size=2, cp_size=2, rank=0, cp_config={"cp_type": cp_type})
    return manager


def refusal(call, kind=ValueError) -> str:
    """The message of the ``kind`` error ``call`` raises; the check fails on anything else."""
    try:
        call()
    except kind as error:
        return str(error)
    except AssertionError:
        raise
    except Exception as error:  # handed to the check as its failure
        raise AssertionError(f"expected a {kind.__name__}, got {error!r}") from error
    raise AssertionError(NOT_REFUSED)


PAST_THE_CHECKS = "the attach got past its checks"
NOT_REFUSED = "the call went through"


def _without_parallelism(manager=None):
    """A mapping read as one rank: what a lender misreading the manager's parallelism sees."""
    return SimpleNamespace(cp_size=1, pp_size=1)


def _defaulting_a_missing_mapping(manager):
    return getattr(manager, "mapping", None) or _without_parallelism()


# Managers the attach refuses before any other work, as (manager, error, words), and a wrong
# reading of the manager's mapping that would let one of them through.
REFUSED = {
    "context_parallel": (
        [(lambda: stand_in("ULYSSES"), ValueError, "context parallelism")],
        _without_parallelism,
    ),
    "not_v2_or_unmapped": (
        [(SimpleNamespace, TypeError, "KVCacheManagerV2"), (stand_in, TypeError, "mapping")],
        _defaulting_a_missing_mapping,
    ),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_manager_no_lender_serves_is_refused_before_any_work(monkeypatch, case):
    """Each ``(manager, kind, words)`` of the case is refused by the attach with a ``kind`` error
    naming ``words``, before the layout is derived: a tripwire there fails the check."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def tripwire(manager):
        raise AssertionError(PAST_THE_CHECKS)

    monkeypatch.setattr(_lender, "derive_layout", tripwire)
    before = len(_lender._retained())
    for make, kind, words in REFUSED[case][0]:
        staging = make()
        assert words in refusal(lambda: attach(staging, guaranteed_fetch_tokens=END), kind)
        assert "_sharing" not in vars(staging)
    assert len(_lender._retained()) == before, "a refused attach allocated staging"


def test_an_attach_that_raises_leaves_nothing_attached(real_manager, monkeypatch, attach=attach):
    """An attach whose last step before the install raises, its log line here, leaves the manager
    without a lender, so a second attach goes through."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def raising(*args, **kwargs):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        with monkeypatch.context() as patched:
            patched.setattr(_lender, "logger", SimpleNamespace(info=raising))
            with pytest.raises(RuntimeError, match="planted"):
                attach(mgr, guaranteed_fetch_tokens=END)
        assert "_sharing" not in vars(mgr), "a failed attach left its lender installed"
        assert isinstance(attach(mgr, guaranteed_fetch_tokens=END), Lender)


def pipeline_stage(rank):
    """Stage ``rank`` of a two-stage pipeline."""
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=2, pp_size=2, rank=rank)


@pytest.mark.parametrize("rank", [0, 1], ids=["first_stage", "last_stage"])
def test_each_pipeline_stage_attaches_a_lender_of_its_own(kit, real_manager, rank, attach=attach):
    """Each stage's lender lends that stage's layers under their global ids."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with real_manager(mapping=pipeline_stage(rank), num_layers=4, windows=[256, WINDOW]) as mgr:
        assert mgr.mapping.pp_size == 2 and len(mgr.pp_layers) == 2, mgr.pp_layers
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        assert vars(mgr)["_sharing"] is lender
        assert not _manager.model_has_recurrent_state(mgr)
        layers = sorted(layer for group in lender._layout.layers for layer in group)
        assert layers == [int(layer) for layer in mgr.pp_layers] == [2 * rank, 2 * rank + 1]
        source = kit.published(mgr, SOURCE, PROMPT)
        lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(lease, mgr)
        lease.release()


def unpaired_draft(mgr) -> None:
    """The state of a draft manager without joint reuse: it never publishes blocks for reuse."""
    mgr._can_publish_block_reuse = False


COMMITS_NOTHING = {
    "block_reuse_off": ({"enable_block_reuse": False}, None),
    "unpaired_draft": ({}, unpaired_draft),
}


@pytest.mark.parametrize("case", list(COMMITS_NOTHING))
def test_a_manager_that_commits_no_blocks_gets_a_lender_naming_nothing(
    kit, real_manager, case, attach=attach
):
    """A name covers committed blocks only."""
    manager_kwargs, adjust = COMMITS_NOTHING[case]
    with real_manager(**manager_kwargs) as mgr:
        if adjust is not None:
            adjust(mgr)
        check_names_nothing(kit, attach(mgr, guaranteed_fetch_tokens=END), ["commits no blocks"])


@pytest.mark.parametrize("case", ["block_reuse_off", "draft"])
def test_mamba_final_context_saves_drop_plan_without_reusable_blocks(case):
    """A final Mamba context chunk finishes its conversation when no blocks can commit."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    saved = []
    resizes = []
    request = SimpleNamespace(
        py_request_id=7,
        is_dummy_request=False,
        context_current_position=END,
        context_remaining_length=0,
        is_last_context_chunk=True,
        expect_snapshot_points=[],
    )
    cache = SimpleNamespace(
        is_active=True,
        resize=lambda capacity, history: resizes.append((capacity, history)) or True,
        enable_swa_scratch_reuse=True,
    )
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.enable_block_reuse = case != "block_reuse_off"
    manager.is_draft = case == "draft"
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager._sharing = None
    manager.kv_cache_map = {request.py_request_id: cache}
    manager.conversation_manager = SimpleNamespace(
        save_drop_plan=lambda req, kv: saved.append((req, kv))
    )
    batch = ScheduledRequests()
    batch.append_context_request(request)

    manager.update_context_resources(batch)

    assert resizes == [(None, END)]
    assert saved == [(request, cache)]
    assert cache.enable_swa_scratch_reuse is False


def test_helix_gets_past_the_parallelism_check(monkeypatch):
    """A Helix manager reaches the layout, where a tripwire stops the attach."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def tripwire(manager):
        raise AssertionError(PAST_THE_CHECKS)

    monkeypatch.setattr(_lender, "derive_layout", tripwire)
    manager = stand_in("HELIX")
    with pytest.raises(AssertionError, match=PAST_THE_CHECKS):
        attach(manager, guaranteed_fetch_tokens=END)
    assert "_sharing" not in vars(manager)


@pytest.mark.parametrize("capacity_only", [False, True], ids=["loads", "capacity_only"])
def test_a_lender_beside_a_kv_cache_connector_that_loads_stages_nothing(
    kit, real_manager, capacity_only, attach=attach
):
    """A loading connector writes pages off the manager's stream; a capacity-only one does not."""
    connector = SimpleNamespace(capacity_only=capacity_only)
    with real_manager(windows=[WINDOW, 256], kv_connector_manager=connector) as mgr:
        assert mgr.kv_connector_manager is connector
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        device = tuple(r for r in lender.regions if r.memory_kind is MemoryKind.DEVICE)
        assert device and lender.regions == kit.staging_regions(lender) + device
        if capacity_only:
            assert kit.staging_regions(lender), "a connector transferring nothing took staging away"
        else:
            words = ["stages nothing", "KV cache connector"]
            check_names_nothing(kit, lender, words, [kit.HOST_NAMED], kit.DEVICE_NAMED)


def test_a_lender_beside_a_kv_cache_connector_lends_in_place_with_pool_rebalancing_configured(
    kit, real_manager, attach=attach
):
    """The rebalance flag leaves the lender's device regions to it."""
    connector = SimpleNamespace(
        capacity_only=False,
        prefix_reservations_enabled=False,
        get_prefix_reservation=lambda request: None,
        has_pending_load=lambda request: False,
        release_unstarted_prefix_loads=lambda request: None,
        release_prefix_reservation=lambda request: None,
        reset_request_state=lambda request: None,
    )
    with real_manager(enable_kv_pool_rebalance=True, kv_connector_manager=connector) as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        assert any(region.memory_kind is MemoryKind.DEVICE for region in lender.regions)
        source = kit.published(mgr, SOURCE, PROMPT)
        check_names_nothing(
            kit, lender, ["stages nothing"], [kit.HOST_NAMED], mgr=mgr, source=source
        )
        mgr.free_resources(source)


def test_a_lossy_cold_page_codec_names_nothing_and_lends_in_place(kit, real_manager, attach=attach):
    """A lossy codec brings a block back from a colder tier with other bytes."""
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import create_default_kv_cache_cold_page_codec

    # The stand-in supplies a native codec so the real constructor records the actual config.
    provider = SimpleNamespace(
        config=ColdPageQuantizationCompressionConfig(),
        create_cold_page_codec=lambda *args, **kwargs: create_default_kv_cache_cold_page_codec(),
    )
    with real_manager(cold_page_codec_provider=provider) as mgr:
        assert mgr._has_lossy_cold_page_codec, "the manager took the codec for a lossless one"
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        source = kit.published(mgr, SOURCE, PROMPT)
        check_names_nothing(kit, lender, ["lossy"], mgr=mgr, source=source)
        mgr.free_resources(source)


@pytest.mark.parametrize("custom_lossless", [False, True])
def test_lender_accepts_default_and_custom_lossless_cold_page_codecs(
    real_manager, custom_lossless, attach=attach
):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import create_default_kv_cache_cold_page_codec

    provider = None
    if custom_lossless:
        provider = SimpleNamespace(
            create_cold_page_codec=lambda *args, **kwargs: create_default_kv_cache_cold_page_codec()
        )
    with real_manager(cold_page_codec_provider=provider) as mgr:
        assert not mgr._has_lossy_cold_page_codec
        assert isinstance(attach(mgr, guaranteed_fetch_tokens=END), Lender)


def eagle3():
    """One-model Eagle 3: its draft at position ``i`` reads prompt token ``i + 1``."""
    from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

    return Eagle3DecodingConfig(max_draft_len=1, speculative_model="draft-model")


def mtp(draft_len=1, vanilla=False):
    """One-model MTP: MTP-Eagle reads one prompt token ahead, vanilla MTP ``draft_len``."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    config = MTPDecodingConfig(max_draft_len=draft_len, use_mtp_vanilla=vanilla)
    config.num_nextn_predict_layers = draft_len  # the checkpoint sets it at model load
    return config


def dflash():
    """A standalone DFlash drafter: its draft KV at ``i`` depends on prompt tokens ``[0, i]``."""
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

    return DFlashDecodingConfig(max_draft_len=2, speculative_model="draft-model")


def pard():
    """PARD: its drafter reads the prompt as the target does."""
    from tensorrt_llm.llmapi.llm_args import PARDDecodingConfig

    return PARDDecodingConfig(max_draft_len=2, speculative_model="draft-model")


def ngram():
    """A drafter outside the one-model engine, over its own request view."""
    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig

    return NGramDecodingConfig(max_draft_len=2, max_matching_ngram_size=2)


# Managers built for a one-model draft reading prompt tokens past a position: case -> (manager
# keywords, tokens read ahead). The draft layers share the target's manager, or have a joint-reuse
# draft pool of their own.
READS_AHEAD = {
    "eagle3_layers_in_the_target": (lambda: dict(spec_config=eagle3()), 1),
    "mtp_eagle_layers_in_the_target": (lambda: dict(spec_config=mtp()), 1),
    "vanilla_mtp_layers_in_the_target": (lambda: dict(spec_config=mtp(2, vanilla=True)), 2),
    "eagle3_joint_reuse_draft_pool": (
        lambda: dict(spec_config=eagle3(), is_draft=True, joint_kv_cache_reuse=True),
        1,
    ),
}
# Managers whose blocks depend on no token past their end, all committing blocks.
READS_NONE = {
    "no_speculation": lambda: dict(),
    "pard_target": lambda: dict(spec_config=pard()),
    "dflash_joint_reuse_draft_pool": lambda: dict(
        spec_config=dflash(), is_draft=True, joint_kv_cache_reuse=True
    ),
    "own_view_draft_pool": lambda: dict(
        spec_config=ngram(), is_draft=True, joint_kv_cache_reuse=True
    ),
}
READ_AHEAD_WORDS = "prompt tokens past a block's end"
OVER_REFUSED = "the attach refused it"


def accepted(call):
    """What ``call`` returns; the check fails on the ``ValueError`` of a refusal."""
    try:
        return call()
    except ValueError as error:
        raise AssertionError(f"{OVER_REFUSED}: {error}") from error


@pytest.mark.parametrize("case", list(READS_AHEAD))
def test_a_manager_built_for_a_draft_reading_past_a_block_gets_a_lender_naming_nothing(
    kit, real_manager, case, attach=attach
):
    make, ahead = READS_AHEAD[case]
    with real_manager(**make()) as mgr:
        assert mgr.draft_prompt_lookahead == ahead
        assert mgr.reuse_match_backoff == ahead
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        check_names_nothing(kit, lender, [READ_AHEAD_WORDS, str(ahead)])


@skip_pre_blackwell
@pytest.mark.parametrize("fp8_ds_mla", [False, True], ids=["fp8", "fp8_ds_mla"])
def test_a_deepseek_v4_manager_whose_mtp_layers_read_ahead_gets_a_lender_naming_nothing(
    kit, deepseek_v4_manager, fp8_ds_mla
):
    from tensorrt_llm._torch.speculative import draft_prompt_lookahead

    # The class opts out of the reuse backoff, so the backoff alone would not see the draft.
    spec_config = mtp()
    assert draft_prompt_lookahead(spec_config) == 1
    with deepseek_v4_manager(spec_config=spec_config, fp8_ds_mla=fp8_ds_mla) as mgr:
        assert mgr.draft_prompt_lookahead == 1
        assert mgr.reuse_match_backoff == 0
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        check_names_nothing(kit, lender, [READ_AHEAD_WORDS, "1"])


@pytest.mark.parametrize("case", list(READS_NONE))
def test_staging_accepts_a_manager_whose_blocks_read_no_token_past_their_end(
    kit, real_manager, case, attach=attach
):
    with real_manager(**READS_NONE[case]()) as mgr:
        assert mgr.draft_prompt_lookahead == 0
        assert mgr.reuse_match_backoff == 0
        assert isinstance(accepted(lambda: attach(mgr, guaranteed_fetch_tokens=END)), Lender)


# -- sizing -----------------------------------------------------------------------------------


# capacity_in_fetches, max_bytes in fetches (None: no cap), and the fetches staging holds at once
SIZING = {
    "capacity_in_fetches": (2, None, 2),
    "max_bytes_of_one_fetch": (4, 1, 1),
    "max_bytes_of_two_and_a_half_fetches": (4, 2.5, 2),
}


@pytest.mark.parametrize("case", list(SIZING))
def test_staging_holds_capacity_in_fetches_fetches_at_once_capped_by_max_bytes(
    kit, real_manager, case
):
    capacity, cap, holding = SIZING[case]
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        one = one_fetch_bytes(kit, mgr, END)
        with pytest.raises(ValueError, match="one fetch"):
            attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=4, max_bytes=one - 1)
        max_bytes = None if cap is None else int(cap * one)
        lender = attach(
            mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=capacity, max_bytes=max_bytes
        )
        (region,) = kit.staging_regions(lender)
        assert region.slot_bytes == kit.DevicePages(mgr).page_bytes(0)
        assert region.nbytes == region.slot_count * region.slot_bytes
        assert holding * one <= region.nbytes <= (max_bytes or holding * one)
        leases = [lender.lend_read(source, 0, END, **kit.HOST_NAMED) for _ in range(holding + 1)]
        for lease in leases[:holding]:
            kit.ready_view(lease, mgr)
        assert leases[-1].poll() is None and leases[-1].failure is None, "one fetch too many"
        leases[0].release()
        check_staged(kit, mgr, lender, source, kit.ready_view(leases[-1], mgr))
        for lease in leases[1:]:
            lease.release()


@pytest.mark.parametrize("num_layers", [2, 3], ids=["one_pool_group", "two_pool_groups"])
def test_a_windowed_range_of_guaranteed_fetch_tokens_always_fits(kit, real_manager, num_layers):
    with real_manager(windows=[WINDOW, 256], num_layers=num_layers) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        dev = kit.DevicePages(mgr)
        rows = {}
        for lg, (window, g) in enumerate(zip(kit.windows(mgr), kit.pool_group_of(mgr))):
            rows[g] = rows.get(g, 0) + (WINDOW // TPB if window else WINDOWED_END // TPB)
        groups = kit.pool_group_ids(mgr)
        regions = kit.staging_regions(lender)
        assert len(regions) == len(groups) == num_layers - 1
        group_of = kit.pool_group_of(mgr)
        for g, region in zip(groups, regions):
            assert (region.slot_count, region.slot_bytes) == (rows[g], dev.group_page_bytes(g))
            assert region.nbytes == region.slot_count * region.slot_bytes
            assert region.layer_group_indices == tuple(
                lg for lg, pg in enumerate(group_of) if pg == g
            )
        # The regions' layer groups are disjoint and cover every layer group.
        held = sorted(lg for region in regions for lg in region.layer_group_indices)
        assert held == list(range(len(group_of)))
        read = lender.lend_read(source, 0, WINDOWED_END, **kit.HOST_NAMED)  # takes every slot
        view = kit.ready_view(read, mgr)
        assert check_staged(kit, mgr, lender, source, view) == sum(rows.values())
        read.release()
        write = lender.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)
        view = kit.ready_view(write, mgr)
        assert view.num_rows == sum(rows.values())
        kit.return_write(lender, write, view.row_masks())


def test_the_staging_memory_is_the_parts_and_nothing_more(kit, real_manager, attach=attach):
    # Three layers: two pool groups, so two regions.
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END, capacity_in_fetches=2)
        regions = kit.staging_regions(lender)
        assert len(regions) == len(kit.pool_group_ids(mgr)) == 2
        (memory,) = kit.staging_memory(regions)
        assert memory.address <= regions[0].address < memory.address + memory.nbytes
        assert memory.nbytes == sum(p.nbytes for p in regions), (
            "staging is not the sum of its regions"
        )


class FreeSpy:
    """The CUDA runtime, recording the address of every page-locked host allocation freed."""

    def __init__(self, runtime):
        self._runtime = runtime
        self.freed = []

    def __getattr__(self, name):
        found = getattr(self._runtime, name)
        if name != "cudaFreeHost":
            return found

        def free(address):
            self.freed.append(int(address))
            return found(address)

        return free


def test_the_staging_memory_is_pinned_at_its_size_and_freed_once(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    if not prefer_pinned():
        pytest.skip("staging is pageable where pinning does not pay off")
    spy = FreeSpy(_lender.cudart)
    monkeypatch.setattr(_lender, "cudart", spy)
    with real_manager() as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=3)
        regions = kit.staging_regions(lender)
        base, total = regions[0].address, sum(p.nbytes for p in regions)
        whole_pages = -(-total // mmap.PAGESIZE) * mmap.PAGESIZE
        assert 1 << (total - 1).bit_length() > whole_pages, "a power of two would pass unseen"
        start, pinned = kit.pinned_range(base)
        assert start == base and total <= pinned <= whole_pages, (
            f"{pinned} bytes pinned for {total}"
        )
        mgr.shutdown()
        assert spy.freed == [base] and kit.pinned_range(base) is None, "not freed at the shutdown"
        mgr.shutdown()
    gc.collect()
    assert spy.freed == [base], "freed more than once"


@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("failed", [False, True], ids=["passed", "failed"])
def test_test_teardown_frees_only_added_staging(
    kit, retained_cleanup, monkeypatch, pinned: bool, failed: bool
) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    if pinned and not prefer_pinned():
        pytest.skip("staging is pageable where pinning does not pay off")
    torch.cuda.init()
    monkeypatch.setattr(_lender, "prefer_pinned", lambda: pinned)
    before = _lender._allocate_retained(mmap.PAGESIZE)
    cleanup = retained_cleanup(failed)
    next(cleanup)
    added = _lender._allocate_retained(mmap.PAGESIZE)
    address = added.address
    index_owner = np.empty(1, dtype=np.int32)
    _lender._retain_until_exit(index_owner)
    try:
        if pinned:
            assert kit.pinned_range(address) is not None
        with pytest.raises(StopIteration):
            next(cleanup)
        if failed:
            assert added.address == address, "a failed test's staging was freed"
            assert any(owner is added for owner in kit.retained())
            assert any(owner is index_owner for owner in kit.retained())
            return
        assert added.address == 0 and added._pageable is None, "teardown did not free staging"
        assert not any(owner is added or owner is index_owner for owner in kit.retained())
        assert before.address != 0 and any(owner is before for owner in kit.retained())
        if pinned:
            assert kit.pinned_range(address) is None
            assert kit.pinned_range(before.address) is not None
    finally:
        for owner in (added, before):
            owner.free()
            _lender._stop_retaining(owner)
        _lender._stop_retaining(index_owner)


def test_pageable_staging_relays_bytes_and_frees_nothing_page_locked(
    kit, real_manager, monkeypatch
):
    """Where pinning does not pay off, as under confidential computing, staging is pageable: it is
    not page-locked, each lender's stays its own while both live, a publish relayed into a fetch
    lands byte for byte, and the shutdown releases it without freeing any page-locked memory."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    spy = FreeSpy(_lender.cudart)
    monkeypatch.setattr(_lender, "cudart", spy)
    monkeypatch.setattr(_lender, "prefer_pinned", lambda: False)
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_a, lender_b = (
            attach(mgr_a, guaranteed_fetch_tokens=END),
            attach(mgr_b, guaranteed_fetch_tokens=END),
        )
        regions = kit.staging_regions(lender_a) + kit.staging_regions(lender_b)
        locked = [kit.pinned_range(region.address) for region in regions]
        assert locked == [None] * len(regions), f"pageable staging was page-locked: {locked}"
        # A pageable buffer freed early can come back to the next lender at the same address.
        spans = sorted((region.address, region.address + region.nbytes) for region in regions)
        assert all(end <= begin for (_, end), (begin, _) in zip(spans, spans[1:])), (
            f"two lenders' staging overlaps: {spans}"
        )
        memories = kit.staging_memory(regions)
        assert len(memories) == 2, f"{len(memories)} kept allocations hold two lenders' staging"
        for memory in memories:
            kept = memory._pageable
            assert kept is not None and kept.ctypes.data == memory.address, (
                "the pageable staging is not an array the lender keeps"
            )
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        fetch = lender_b.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = fetch.poll()
        assert view is not None, f"the write lease was not ready: {fetch.failure}"
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        assert return_and_settle(lender_b, mgr_b, target, fetch, masks) == (END, 0)
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        publish.release()
        for mgr in (mgr_a, mgr_b):
            mgr.shutdown()
        assert spy.freed == [], "pageable staging was freed as page-locked memory"
        assert not kit.staging_kept(regions), "the staging memory was kept past the shutdown"


# -- publish ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens", [PROMPT, PROMPT[:END]], ids=["with_a_partial_block", "on_a_block_boundary"]
)
def test_a_publish_lends_the_committed_pages_under_their_reuse_keys(kit, real_manager, tokens):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, tokens)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        assert isinstance(lease, Lease)
        view = kit.ready_view(lease, mgr)
        assert isinstance(view, LeaseView) and lease.poll() is view, "one view, every poll"
        (run,) = view.groups
        assert run.page_ordinals.tolist() == list(range(BLOCKS)) and run.region_index == 0
        assert check_staged(kit, mgr, lender, source, view) == BLOCKS
        kv = kit.kv(mgr, source)
        assert [bytes(name[16:48]) for name in run.object_keys] == kit.chain_keys(kv, tokens)[
            :BLOCKS
        ]
        assert len({bytes(name[:16]) for name in run.object_keys}) == 1, "one namespace"
        assert len({bytes(name[48:]) for name in run.object_keys}) == 1, (
            "one layer group, one shard"
        )
        # The manager committed under those keys: its tree finds the block after them by them.
        extra = list(range(7000, 7000 + TPB))
        probed = mgr.impl.probe_first_new_block_key(kv.reuse_scope, PROMPT[:END] + extra)
        assert probed == kit.chain_keys(kv, PROMPT[:END] + extra)[BLOCKS]
        lease.release()
        with pytest.raises(RuntimeError):
            lease.poll()


def test_a_multimodal_publish_is_named_by_the_keys_the_manager_commits(kit, real_manager):
    image = dict(multimodal_positions=[40], multimodal_lengths=[16])
    with real_manager() as mgr:
        first = kit.published(mgr, SOURCE, PROMPT, multimodal_hashes=[list(range(1, 9))], **image)
        second = kit.published(
            mgr, OTHER, PROMPT, multimodal_hashes=[list(range(8, 0, -1))], **image
        )
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        leases = [lender.lend_read(r, 0, END, **kit.HOST_NAMED) for r in (first, second)]
        names = [kit.ready_view(lease, mgr).groups[0].object_keys for lease in leases]
        keys = [[bytes(n[16:48]) for n in run_names] for run_names in names]
        kv = kit.kv(mgr, first)
        augmented = list(mgr._augment_tokens_for_block_reuse(PROMPT, first))
        assert keys[0] == kit.chain_keys(kv, augmented)[:BLOCKS]
        extra = list(range(7000, 7000 + TPB))
        probed = mgr.impl.probe_first_new_block_key(kv.reuse_scope, augmented[:END] + extra)
        assert probed == kit.chain_keys(kv, augmented[:END] + extra)[BLOCKS]
        # The image sits in block 1: the block before it is shared, the rest is not.
        assert keys[0][0] == keys[1][0] and not set(keys[0][1:]) & set(keys[1][1:])
        assert keys[0][1:] != kit.chain_keys(kv, PROMPT)[1:BLOCKS], "the image changes the keys"
        for lease in leases:
            lease.release()


@pytest.mark.parametrize(
    "scope", [dict(cache_salt="tenant-a"), dict(lora_task_id=7)], ids=["cache_salt", "lora_task_id"]
)
def test_a_publish_is_named_under_the_request_s_reuse_scope(kit, real_manager, scope):
    with real_manager() as mgr:
        plain = kit.published(mgr, SOURCE, PROMPT)
        scoped = kit.published(mgr, OTHER, PROMPT, **scope)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        leases = [lender.lend_read(r, 0, END, **kit.HOST_NAMED) for r in (plain, scoped)]
        keys = [
            [bytes(n[16:48]) for n in kit.ready_view(lease, mgr).groups[0].object_keys]
            for lease in leases
        ]
        kv = kit.kv(mgr, scoped)
        assert keys[1] == kit.chain_keys(kv, PROMPT)[:BLOCKS], "a name ignores the reuse scope"
        # The manager committed under those keys: its tree finds the block after them by them.
        extra = list(range(7000, 7000 + TPB))
        probed = mgr.impl.probe_first_new_block_key(kv.reuse_scope, PROMPT[:END] + extra)
        assert probed == kit.chain_keys(kv, PROMPT[:END] + extra)[BLOCKS]
        assert not set(keys[0]) & set(keys[1]), "the reuse scope changes no key"
        for lease in leases:
            lease.release()


def test_a_windowed_publish_lends_only_what_the_window_at_its_end_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        windows = kit.windows(mgr)
        sliding, full = windows.index(WINDOW), windows.index(None)
        lease = lender.lend_read(source, 0, WINDOWED_END, **kit.HOST_NAMED)
        view = kit.ready_view(lease, mgr)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        assert end > beg, "the window must leave early blocks behind"
        blocks = range(WINDOWED_END // TPB)
        in_window = [o for o in blocks if not beg <= o < end]
        assert by_group(view) == {full: list(blocks), sliding: in_window}
        addresses = np.concatenate([run.addresses for run in view.groups])
        assert len(set(addresses.tolist())) == len(addresses), "every row has its own slot"
        assert check_staged(kit, mgr, lender, source, view) == len(addresses)
        lease.release()

        # The request's own window has since passed what a window at 96 reads: none is lent.
        early = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        rows = by_group(kit.ready_view(early, mgr))
        assert rows[full] == list(range(BLOCKS))
        assert rows.get(sliding, []) == []
        early.release()


def test_a_range_without_rows_is_ready_at_once_and_waits_for_no_one(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        holder = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # every slot
        kit.ready_view(holder, mgr)
        empty = lender.lend_read(source, END, END, **kit.HOST_NAMED)
        view = empty.poll()
        assert view is not None and view.num_rows == 0
        assert len(view.groups) == kit.num_layer_groups(mgr)
        empty.release()
        holder.release()


# -- ready means copied -----------------------------------------------------------------------


def test_a_publish_is_ready_only_once_its_copy_ran(kit, real_manager, attach=attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            assert lease.poll() is None, "ready before its copy into staging ran"
            lender.progress()
            assert lease.poll() is None and lease.failure is None
            gate.open()
        check_staged(kit, mgr, lender, source, kit.ready_view(lease, mgr))
        lease.release()


def test_a_publish_is_ready_only_once_its_ready_event_completed(kit, real_manager):
    """A publish's copy into staging waits for its ``ready_event``."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        elsewhere = torch.cuda.Stream()
        with kit.held_stream(elsewhere, strict=False):
            ready = torch.cuda.Event()
            ready.record(elsewhere)
            lease = lender.lend_read(source, 0, END, **{**kit.HOST_NAMED, "ready_event": ready})
            # Long enough for a copy queued without waiting for the event to have run.
            deadline = time.monotonic() + 0.5
            while not mgr._stream.query() and time.monotonic() < deadline:
                time.sleep(0.01)
            for _ in range(3):
                lender.progress()
                assert lease.poll() is None, "ready before its ready event completed"
            assert lease.failure is None, lease.failure
        check_staged(kit, mgr, lender, source, kit.ready_view(lease, mgr))
        lease.release()


def test_a_fetch_s_copy_into_the_pages_waits_for_its_ready_event(kit, real_manager):
    """A fetch's view is ready at once; its copy into the pages waits for its ``ready_event``."""
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        elsewhere = torch.cuda.Stream()
        with kit.held_stream(elsewhere, strict=False):
            ready = torch.cuda.Event()
            ready.record(elsewhere)
            lease = lender.lend_write(target, 0, END, **{**kit.HOST_NAMED, "ready_event": ready})
            view = lease.poll()
            assert view is not None, lease.failure
            kit.stage(lender, view, 0x5A)
            settled = kit.return_write(lender, lease, ALL_ARRIVED)
            assert settled is None, "settled before its ready event"
            # Long enough for a copy queued without waiting for the event to have run.
            deadline = time.monotonic() + 0.5
            while not mgr._stream.query() and time.monotonic() < deadline:
                time.sleep(0.01)
            for _ in range(3):
                lender.progress()
                assert lender.poll_resume_bounds(target) is None, "settled before its ready event"
        assert kit.settled_bounds(lender, mgr, target).max_position == END
        kv = kit.kv(mgr, target)
        dev = kit.DevicePages(mgr)
        for run in view.groups:
            lg = run.layer_group_index
            slots = kit.pages(kv, lg)
            for ordinal in run.page_ordinals.tolist():
                got = kit.digest(dev.read(lg, slots[ordinal]))
                staged = kit.digest(kit.staged_row(0x5A, lg, ordinal, dev.page_bytes(lg)))
                assert got == staged, f"layer group {lg} block {ordinal} holds other bytes"


# -- no slot reused while a copy or a backend may touch it -----------------------------------


def test_a_slot_returns_only_after_the_copy_into_it_ran(kit, real_manager, attach=attach):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        # Grown ahead, so the held section only lends.
        assert mgr._resize_for_served_prefix(target, kit.kv(mgr, target), 0, END)
        lender = attach(mgr, guaranteed_fetch_tokens=END)  # one fetch: three slots
        with kit.held_stream(mgr._stream) as gate:
            read = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # every slot, its copy held
            read.release()
            write = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
            lender.progress()
            assert write.poll() is None, "slots handed out while a copy into them was queued"
            gate.open()
        view = kit.ready_view(write, mgr)
        kit.return_write(lender, write, view.row_masks())


def test_leases_wait_for_slots_strictly_in_order(kit, real_manager, attach=attach):
    """Waiting leases are granted in order, by the lender's own calls, without new lends."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)  # three slots
        first = lender.lend_read(source, 0, 2 * TPB, **kit.HOST_NAMED)
        kit.ready_view(first, mgr)
        second = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        third = lender.lend_read(source, 0, TPB, **kit.HOST_NAMED)
        lender.progress()
        assert second.poll() is None
        assert third.poll() is None, "a later lease that fits jumped the one waiting ahead"
        first.release()
        kit.ready_view(second, mgr)
        lender.progress()
        assert third.poll() is None
        second.release()
        assert by_group(kit.ready_view(third, mgr)) == {0: [0]}
        third.release()


def test_capacity_in_fetches_is_a_budget_and_holes_can_make_a_lease_wait(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        (group,) = kit.pool_group_ids(mgr)
        (region,) = kit.staging_regions(lender)
        assert region.slot_count == 2 * BLOCKS
        first = lender.lend_read(source, 0, 2 * TPB, **kit.HOST_NAMED)  # slots 0 and 1
        second = lender.lend_read(source, 0, TPB, **kit.HOST_NAMED)  # slot 2
        third = lender.lend_read(source, 0, TPB, **kit.HOST_NAMED)  # slot 3
        for lease in (first, second, third):
            kit.ready_view(lease, mgr)
        first.release()
        lender.progress()
        assert lender._free_slots(group) == 4  # slots 0, 1, 4 and 5
        fetch = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        assert fetch.poll() is None and fetch.failure is None, "no three free slots in a row"
        second.release()  # slots 0 to 2 are free in a row
        (run,) = kit.ready_view(fetch, mgr).groups
        assert ((run.addresses - region.address) // region.slot_bytes).tolist() == [0, 1, 2]
        for lease in (third, fetch):
            lease.release()


# -- fetch ------------------------------------------------------------------------------------


@pytest.mark.parametrize("local_blocks", [0, 2], ids=["cold", "local_prefix"])
def test_a_fetch_is_usable_once_its_copy_lands(kit, real_manager, local_blocks, attach=attach):
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        if local_blocks:
            kit.published(mgr_b, 7, PROMPT[: local_blocks * TPB] + list(range(9000, 9033)))
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        kv = kit.kv(mgr_b, target)
        local, history = kv.num_committed_tokens, kv.history_length
        assert local == local_blocks * TPB
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=END)
        assert lender_b.poll_resume_bounds(target) == (max(local, history), history)

        lease = lender_b.lend_write(target, local, END, **kit.HOST_NAMED)
        assert lender_b.poll_resume_bounds(target) is None, "a fetch counts from its call"
        assert kv.capacity >= END and kv.history_length == history
        view = lease.poll()
        assert view is not None, "a write is ready at its first poll after the grant"
        assert by_group(view) == {0: list(range(local_blocks, BLOCKS))}
        kit.fill_sentinel(mgr_b, target, first_block=local_blocks)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        with kit.held_stream(mgr_b._stream) as gate:
            assert kit.return_write(lender_b, lease, masks) is None, "settled before its copy ran"
            lender_b.progress()
            assert lender_b.poll_resume_bounds(target) is None, "settled before its copy ran"
            gate.open()
        lender_b.progress()
        resume_bounds = lender_b.poll_resume_bounds(target)
        assert isinstance(resume_bounds, ResumeBounds) and all(
            type(v) is int for v in resume_bounds
        )
        assert resume_bounds == (END, min(local, history))
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        publish.release()


@pytest.mark.parametrize(
    "rows,usable",
    [((0, 1), 64), ((0, 2), 32), ((), 0), ((1, 2), 0)],
    ids=["prefix", "gap", "nothing", "no_first_block"],
)
def test_a_partial_arrival_is_usable_up_to_its_first_gap(kit, real_manager, rows, usable):
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=END)
        lease = lender_b.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target)
        kit.relay(lender_a, publish_view, lender_b, view, deliver=lambda run, row: row in rows)
        arrived = (np.isin(np.arange(BLOCKS), rows),)
        assert return_and_settle(lender_b, mgr_b, target, lease, arrived) == (usable, 0)
        sentinel = bytes([kit.SENTINEL]) * kit.DevicePages(mgr_b).page_bytes(0)
        for ordinal in range(BLOCKS):
            expected = kit.page(mgr_a, source, 0, ordinal) if ordinal in rows else sentinel
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(expected), "a row that did not arrive was copied"
        publish.release()


@pytest.mark.parametrize("partial", [False, True], ids=["complete", "partial"])
def test_a_windowed_fetch_is_usable_only_from_its_floor(kit, real_manager, partial):
    windows = [WINDOW, 256]
    with real_manager(windows=windows) as mgr_a, real_manager(windows=windows) as mgr_b:
        source = kit.published(mgr_a, SOURCE, WINDOWED_PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=WINDOWED_END)
        publish = lender_a.lend_read(source, 0, WINDOWED_END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        target = kit.admitted(mgr_b, TARGET, WINDOWED_PROMPT)
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=WINDOWED_END)
        lease = lender_b.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)
        assert kit.kv(mgr_b, target).history_length == WINDOWED_END, "the history moves to the end"
        view = lease.poll()
        assert by_group(view) == by_group(publish_view)
        kit.fill_sentinel(mgr_b, target)
        deliver = (lambda run, row: row < 2) if partial else None
        masks = kit.relay(lender_a, publish_view, lender_b, view, deliver)
        resume_bounds = return_and_settle(lender_b, mgr_b, target, lease, masks)
        assert resume_bounds.min_position == WINDOWED_END, "the window released the early blocks"
        if partial:
            assert resume_bounds.max_position < resume_bounds.min_position, "empty: compute from 0"
            for run in view.groups:  # the two rows of each run that arrived hold the source's bytes
                for ordinal in run.page_ordinals.tolist()[:2]:
                    lg = run.layer_group_index
                    got = kit.digest(kit.page(mgr_b, target, lg, ordinal))
                    assert got == kit.digest(kit.page(mgr_a, source, lg, ordinal))
        else:
            assert resume_bounds.max_position == WINDOWED_END
            for run in view.groups:
                for ordinal in run.page_ordinals.tolist():
                    lg = run.layer_group_index
                    got = kit.digest(kit.page(mgr_b, target, lg, ordinal))
                    assert got == kit.digest(kit.page(mgr_a, source, lg, ordinal))
        publish.release()


def read_only(array):
    """``array`` made read-only."""
    array.flags.writeable = False
    return array


def test_return_writes_takes_one_mask_per_run_of_each_write_seen_ready_once(kit, real_manager):
    """A wrong entry raises before any lease returns; a write seen ready returns once."""
    with real_manager() as mgr, real_manager() as other_mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        whole = kit.admitted(mgr, OTHER, list(range(8000, 8097)))
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=3)
        read = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        read_view = kit.ready_view(read, mgr)
        with pytest.raises(RuntimeError, match="for write leases"):
            kit.return_write(lender, read, read_view.row_masks())
        write = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        with pytest.raises(RuntimeError, match="before poll"):
            kit.return_write(lender, write, ())  # before poll() gave the view
        view = write.poll()
        foreign_target = kit.admitted(other_mgr, TARGET, OTHER_PROMPT)
        foreign_lender = attach(other_mgr, guaranteed_fetch_tokens=END)
        foreign = foreign_lender.lend_write(foreign_target, 0, END, **kit.HOST_NAMED)
        foreign_view = foreign.poll()
        bad = (
            ((write, (), True), ValueError),
            ((write, (np.ones(BLOCKS - 1, bool),), True), ValueError),
            ((write, (np.ones(BLOCKS, bool),) * 2, True), ValueError),
            ((write, (np.ones(BLOCKS, np.int64),), True), ValueError),  # one per row, not bool
            # read-only, but one row short
            ((write, (read_only(np.ones(BLOCKS - 1, bool)),), True), ValueError),
            # read-only and one per row, but not bool
            ((write, (read_only(np.ones(BLOCKS, np.int64)),), True), ValueError),
            ((foreign, foreign_view.row_masks(), True), ValueError),  # another lender's
            (write, TypeError),
        )
        for entry, error in bad:
            with pytest.raises(error):
                lender.return_writes([entry])
            assert lender.poll_resume_bounds(target) is None, "a refused entry recorded something"
            assert write.poll() is view, "a refused entry returned the lease"
        kv = kit.kv(mgr, target)
        own = (kv.num_committed_tokens, kv.history_length)
        assert kit.return_write(lender, write, view.row_masks()) == own, "no row arrived"
        with pytest.raises(RuntimeError, match="returned already"):
            kit.return_write(lender, write, view.row_masks())
        with pytest.raises(RuntimeError):
            write.poll()
        every = lender.lend_write(whole, 0, END, **kit.HOST_NAMED)
        every_view = every.poll()
        kit.fill_sentinel(mgr, whole)
        kit.stage(lender, every_view, 0x5A)
        assert return_and_settle(lender, mgr, whole, every, ALL_ARRIVED) == (END, 0)
        nbytes = kit.DevicePages(mgr).page_bytes(0)
        got = kit.digest([kit.page(mgr, whole, 0, o) for o in range(BLOCKS)])
        assert got == kit.digest([kit.staged_row(0x5A, 0, o, nbytes) for o in range(BLOCKS)]), (
            "ALL_ARRIVED left a row out"
        )
        for lease in (read, foreign):
            lease.release()


@pytest.mark.parametrize("ending", ["returned_writable", "returned_read_only", "released"])
def test_a_write_seen_ready_gives_back_its_slots_once_no_copy_reads_them(kit, real_manager, ending):
    """Its slots come back once its copy ran; released without a return, it delivers no row."""
    with real_manager() as mgr:
        first = kit.admitted(mgr, TARGET, PROMPT)
        second = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)  # the second waits for the first's slots
        lease = lender.lend_write(first, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        kit.fill_sentinel(mgr, first)
        kit.stage(lender, view, 0x5A)
        waiting = lender.lend_write(second, 0, END, **kit.HOST_NAMED)
        nbytes = kit.DevicePages(mgr).page_bytes(0)
        if ending == "released":
            kv = kit.kv(mgr, first)
            lease.release()
            bounds = lender.poll_resume_bounds(first)
            assert bounds == (kv.num_committed_tokens, kv.history_length), (
                f"resume_bounds {bounds} of a write released without a return"
            )
            with pytest.raises(RuntimeError, match="returned already"):
                kit.return_write(lender, lease, view.row_masks(True))
            expected = [bytes([kit.SENTINEL]) * nbytes] * BLOCKS
        else:
            masks = view.row_masks(True)
            if ending == "returned_read_only":
                masks = tuple(read_only(mask) for mask in masks)
            with kit.held_stream(mgr._stream) as gate:
                assert kit.return_write(lender, lease, masks) is None, "settled before its copy ran"
                lender.progress()
                assert waiting.poll() is None, "slots returned before the copy out of them ran"
                gate.open()
            expected = [kit.staged_row(0x5A, 0, o, nbytes) for o in range(BLOCKS)]
        waiting_view = kit.ready_view(waiting, mgr)
        got = kit.digest([kit.page(mgr, first, 0, o) for o in range(BLOCKS)])
        assert got == kit.digest(expected), "the pages hold other bytes than the rows delivered"
        if ending != "released":
            assert lender.poll_resume_bounds(first) == (END, 0)
        kit.return_write(lender, waiting, waiting_view.row_masks())


# -- the request exits during a lease ---------------------------------------------------------


def test_arrivals_for_a_freed_request_copy_nothing(kit, real_manager, attach=attach):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        assert view is not None
        kv = kit.kv(mgr, target)
        lent = {kit.pages(kv, 0)[o] for o in view.groups[0].page_ordinals.tolist()}
        kit.stage(lender, view, 0x5A)
        mgr.free_resources(target)
        with pytest.raises(ValueError):
            lender.poll_resume_bounds(target)
        others = kit.Requests(mgr)
        try:
            assert others.allocate(kit.pool_pages(mgr))
            assert lent <= others.pages(), "no lent page was reused: the check proves nothing"
            assert kit.return_write(lender, lease, view.row_masks(True)) is None
            mgr._stream.synchronize()
            dev = kit.DevicePages(mgr)
            sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
            for slot in sorted(lent):
                got = kit.digest(dev.read(0, slot))
                assert got == kit.digest(sentinel), "an arrival overwrote another request's page"
        finally:
            lease.release()
            others.free()


def commit_fetched(mgr, request, end):
    """The manager commits ``request``'s blocks up to ``end``, as after a context chunk."""
    request.context_chunk_size = end - request.context_current_position
    request.move_to_next_context_chunk()
    mgr.try_commit_blocks(request)


def test_a_commit_during_a_fetch_copies_nothing_into_pages_it_returned(
    kit, real_manager, attach=attach
):
    """A commit while a write is open rebases the target onto blocks another request committed and
    returns the lent pages to the pool: arrivals reach neither those pages nor the shared ones."""
    with real_manager(max_tokens=kit.POOL_TOKENS, max_batch_size=16) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        assert view is not None
        kv = kit.kv(mgr, target)
        lent = {kit.pages(kv, 0)[o] for o in view.groups[0].page_ordinals.tolist()}
        kit.stage(lender, view, 0x5A)
        source = kit.published(mgr, SOURCE, PROMPT)
        shared = kit.pages(kit.kv(mgr, source), 0)[:BLOCKS]
        committed_bytes = kit.digest([kit.page(mgr, source, 0, o) for o in range(BLOCKS)])
        commit_fetched(mgr, target, END)
        assert kv.num_committed_tokens == END
        assert kit.pages(kv, 0)[:BLOCKS] == shared, "no rebase: the check proves nothing"
        others = kit.Requests(mgr)
        try:
            while others.allocate(1, chunk=1):
                pass
            assert lent <= others.pages(), "no returned page was reused: the check proves nothing"
            kit.return_write(lender, lease, view.row_masks(True))
            mgr._stream.synchronize()
            dev = kit.DevicePages(mgr)
            sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
            for slot in sorted(lent):
                got = kit.digest(dev.read(0, slot))
                assert got == kit.digest(sentinel), "an arrival overwrote another request's page"
            got = kit.digest([kit.page(mgr, source, 0, o) for o in range(BLOCKS)])
            assert got == committed_bytes, "an arrival overwrote the blocks the target rebased onto"
        finally:
            lease.release()
            others.free()


@pytest.mark.parametrize("change", ["freed", "suspended"])
def test_a_waiting_read_fails_when_its_cache_changes(kit, real_manager, change, attach=attach):
    """A read waiting for slots fails at once when its request is freed, and at its grant when its
    cache was suspended; either way its slots come back."""
    with real_manager() as mgr:
        holder = kit.published(mgr, SOURCE, PROMPT)
        other = kit.published(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        head = lender.lend_read(holder, 0, END, **kit.HOST_NAMED)
        kit.ready_view(head, mgr)
        waiting = lender.lend_read(other, 0, 2 * TPB, **kit.HOST_NAMED)
        assert waiting.poll() is None and waiting.failure is None
        if change == "freed":
            mgr.free_resources(other)
            assert waiting.failure is not None, "a lease waiting for slots outlived its request"
        else:
            mgr.suspend_request(other)
        head.release()
        lender.progress()
        assert waiting.poll() is None and waiting.failure is not None, "copied a moved cache"
        waiting.release()
        again = lender.lend_read(holder, 0, END, **kit.HOST_NAMED)  # its slots came back
        kit.ready_view(again, mgr)
        again.release()


def test_a_publish_granted_before_its_request_is_freed_copies_what_it_lent(kit, real_manager):
    # No host tier: the freed pages go straight to the other requests. With one, they would move
    # to host first, and the first such move in a process waits for the whole device: behind the
    # held stream, that is until the watchdog opens it.
    with real_manager(max_tokens=kit.POOL_TOKENS, host_cache_size=0) as mgr:
        # A process's first cache close waits for the whole device, the gated stream included:
        # close one before the gate.
        mgr.free_resources(kit.published(mgr, 900, list(range(40_000, 40_000 + TPB + 1))))
        torch.cuda.synchronize()
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        lent = set(kit.pages(kit.kv(mgr, source), 0)[:BLOCKS])
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        dev = kit.DevicePages(mgr)  # built ahead: building it waits for the device
        others = kit.Requests(mgr, dev)
        with kit.held_stream(mgr._stream, strict=False) as gate:
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # granted, its copy held
            mgr.free_resources(source)
            assert lease.failure is None and kit.kv(mgr, source) is None
            # Other requests take every page and overwrite it, on the stream behind the copy.
            assert others.allocate(kit.pool_pages(mgr), fill=False)
            assert lent <= others.pages(), "no freed page was taken: the check proves nothing"
            gate.open()
        sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
        assert all(dev.read(0, slot) == sentinel for slot in lent), "the pages were not overwritten"
        (run,) = kit.ready_view(lease, mgr).groups
        length = lender.regions[run.region_index].slot_bytes
        got = kit.digest([kit.host_bytes(a, length) for a in run.addresses.tolist()])
        assert got == kit.digest(original), "the copy read a later owner's writes"
        lease.release()
        others.free()


# -- abandoned fetches ------------------------------------------------------------------------


@pytest.mark.parametrize("waiting", [False, True], ids=["granted", "waiting_in_line"])
@pytest.mark.parametrize("windowed", [False, True], ids=["full", "windowed"])
def test_an_abandoned_fetch_leaves_only_its_growth(
    kit, real_manager, windowed, waiting, attach=attach
):
    tokens, other, end = (
        (WINDOWED_PROMPT, OTHER_WINDOWED_PROMPT, WINDOWED_END)
        if windowed
        else (PROMPT, OTHER_PROMPT, END)
    )
    with real_manager(windows=[WINDOW, 256] if windowed else None) as mgr:
        target = kit.admitted(mgr, TARGET, tokens)
        second = kit.admitted(mgr, OTHER, other)
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        head = None
        if waiting:  # a publish takes every slot first, so the write waits in line
            holder = kit.published(mgr, SOURCE, list(range(8000, 8000 + len(tokens))))
            head = lender.lend_read(holder, 0, end, **kit.HOST_NAMED)
            kit.ready_view(head, mgr)
        lease = lender.lend_write(target, 0, end, **kit.HOST_NAMED)
        assert lender.poll_resume_bounds(target) is None
        if waiting:
            assert lease.poll() is None and lease.failure is None, "the write did not wait"
        lease.release()  # before any grant or view: no backend wrote, the fetch is abandoned
        if head is not None:
            head.release()
        lender.progress()  # the slots come back; the released write takes none of them
        kv = kit.kv(mgr, target)
        own = (kv.num_committed_tokens, kv.history_length)
        assert lender.poll_resume_bounds(target) == own, "the abandoned fetch still counts"
        if windowed:
            assert kv.history_length == end, "the growth moved the window's history"
            before = state_of(kit, mgr, lender, target)
            below = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
            assert below.failure is not None and "below the history" in below.failure
            below.release()
            assert state_of(kit, mgr, lender, target) == before
        again = lender.lend_write(second, 0, end, **kit.HOST_NAMED)
        view = again.poll()
        assert view is not None, "the abandoned fetch kept its slots"
        kit.return_write(lender, again, view.row_masks())


# -- a fetch split into leases, and a shrink of the cache it went into -------------------------

SPLIT_PROMPT = list(range(3000, 3129))  # four whole blocks and one token
SMALL_POOL = dict(max_tokens=256, max_batch_size=64)


def stale_anywhere(kit, mgr, history):
    """Some windowed layer group has released blocks behind its window at ``history``."""
    for lg, window in enumerate(kit.windows(mgr)):
        if window is not None:
            beg, end = kit.stale_blocks(mgr, lg, history)
            if end > beg:
                return True
    return False


def fetch_segment(kit, mgr, lender, target, start, end, byte, arrived=None):
    """A write over ``[start, end)`` staged from ``byte``, returned with ``arrived``: its bounds."""
    lease = lender.lend_write(target, start, end, **kit.HOST_NAMED)
    view = lease.poll()
    assert view is not None, f"the write lease was not ready: {lease.failure}"
    kit.stage(lender, view, byte)
    if arrived is None:
        masks = ALL_ARRIVED
    else:
        masks = tuple(np.isin(run.page_ordinals, arrived) for run in view.groups)
    return return_and_settle(lender, mgr, target, lease, masks)


SEGMENTS = {
    # name: (manager kwargs, segments [(start, end, arrived)], max_position after each segment)
    "two_halves": ({}, [(0, 64, None), (64, 128, None)], [64, 128]),
    "three_segments": ({}, [(0, 32, None), (32, 96, None), (96, 128, None)], [32, 96, 128]),
    "windowed": ({"windows": [WINDOW, 256]}, [(0, 64, None), (64, 128, None)], [64, 128]),
    # A gap stays a gap across segments.
    "first_landed_nothing": ({}, [(0, 64, []), (64, 128, None)], [0, 0]),
    "first_landed_one_block": ({}, [(0, 64, [0]), (64, 128, None)], [32, 32]),
}


@pytest.mark.parametrize("case", list(SEGMENTS))
def test_consecutive_leases_add_up_to_one_fetch(kit, real_manager, case, attach=attach):
    """Consecutive leases into one cache, nothing committed in between: resume_bounds covers every
    segment that landed, and nothing more."""
    manager_kwargs, segments, expected = SEGMENTS[case]
    with real_manager(**manager_kwargs) as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == 0, "no local prefix: every usable token was fetched"
        lender = attach(mgr, guaranteed_fetch_tokens=64)
        bytes_of = {}
        for i, ((start, end, arrived), usable) in enumerate(zip(segments, expected)):
            byte = 0x11 * (i + 1)
            fetch_segment(kit, mgr, lender, target, start, end, byte, arrived)
            for ordinal in range(start // TPB, end // TPB):
                if arrived is None or ordinal in arrived:
                    bytes_of[ordinal] = byte
            assert kv.num_committed_tokens == 0, "a fetch committed"
            resume_bounds = lender.poll_resume_bounds(target)
            floor = kv.history_length if stale_anywhere(kit, mgr, kv.history_length) else 0
            assert resume_bounds is not None and resume_bounds.min_position == floor
            assert resume_bounds.max_position >= usable, (
                f"after segment {i} [{start}, {end}): resume_bounds {tuple(resume_bounds)}; an earlier "
                "segment's delivered prefix was dropped"
            )
            assert resume_bounds.max_position == usable, (
                f"after segment {i}: resume_bounds {tuple(resume_bounds)} counts what never landed"
            )
        # The bytes are in the pages: only the bookkeeping decides whether they count. A window's
        # blocks behind the final history are released, and no resume reads them.
        dev = kit.DevicePages(mgr)
        for lg in range(kit.num_layer_groups(mgr)):
            stale_beg, stale_end = kit.stale_blocks(mgr, lg, kv.history_length)
            slots = kit.pages(kv, lg)
            for ordinal, byte in bytes_of.items():
                if stale_beg <= ordinal < stale_end:
                    continue
                assert slots[ordinal] >= 0, f"layer group {lg} lost block {ordinal}'s page"
                got = kit.digest(dev.read(lg, slots[ordinal]))
                staged = kit.digest(kit.staged_row(byte, lg, ordinal, dev.page_bytes(lg)))
                assert got == staged, f"layer group {lg} block {ordinal} holds other bytes"


# name: the span of the lease after the first, two blocks long
EXTRA_TOKEN_SPLITS = {"window_and_a_block": 3 * TPB, "within_the_window": 2 * TPB}


@pytest.mark.parametrize("case", list(EXTRA_TOKEN_SPLITS))
def test_a_later_lease_past_the_window_and_a_block_fails_with_extra_kv_tokens(
    kit, real_manager, case, attach=attach
):
    """All-reusable, a window of ``WINDOW`` tokens and PARD's extra KV token: the first lease's grow
    gives block 2, which holds that token, a page past the history the lease leaves, and nothing
    writes it. A consecutive lease spanning at least ``WINDOW + TPB - 1`` tokens leaves that block
    behind, so it fails at the call and changes nothing in the cache; a shorter one is lent."""
    span = EXTRA_TOKEN_SPLITS[case]
    with real_manager(spec_config=pard(), windows=[WINDOW, 256]) as mgr:
        assert mgr.num_extra_kv_tokens == 1, "no extra KV token: nothing checked"
        target = kit.admitted(mgr, TARGET, WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=3 * TPB)
        fetch_segment(kit, mgr, lender, target, 0, 2 * TPB, 0x5A)
        before = state_of(kit, mgr, lender, target)
        assert before[1].max_position == 2 * TPB, f"resume_bounds after the first lease {before[1]}"
        sliding = kit.windows(mgr).index(WINDOW)
        assert kit.pages(kit.kv(mgr, target), sliding)[2] >= 0, (
            "block 2 has no page: nothing checked"
        )
        lease = lender.lend_write(target, 2 * TPB, 2 * TPB + span, **kit.HOST_NAMED)
        if span >= WINDOW + TPB - 1:
            assert lease.failure is not None and "never wrote" in lease.failure, (
                f"a later lease of {span} tokens was lent past an unwritten page: {lease.failure}"
            )
            assert state_of(kit, mgr, lender, target) == before, (
                "the failed lease changed the cache"
            )
            lease.release()
        else:
            view = kit.ready_view(lease, mgr)
            kit.stage(lender, view, 0x6B)
            resume_bounds = return_and_settle(lender, mgr, target, lease, view.row_masks(True))
            for lg in range(
                kit.num_layer_groups(mgr)
            ):  # what resume_bounds counts, the window's alone
                fetched = [([2, 3], 0x6B)] if lg == sliding else [([0, 1], 0x5A), ([2, 3], 0x6B)]
                for ordinals, byte in fetched:
                    assert holds(kit, mgr, target, lg, ordinals, staged_bytes(kit, byte)), (
                        f"layer group {lg}: fetched blocks {ordinals} lack their staged bytes"
                    )
            end = 2 * TPB + span
            assert resume_bounds == (end, end), (
                f"resume_bounds {tuple(resume_bounds)} after the later lease"
            )


def test_an_abandoned_segment_keeps_what_earlier_ones_delivered(kit, real_manager, attach=attach):
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=64)
        fetch_segment(kit, mgr, lender, target, 0, 64, 0x11)
        assert lender.poll_resume_bounds(target) == (64, 0)
        second = lender.lend_write(target, 64, 128, **kit.HOST_NAMED)
        assert lender.poll_resume_bounds(target) is None, "a fetch counts from its call"
        second.release()  # before anyone saw it ready: abandoned, nothing written
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds == (64, 0), (
            f"resume_bounds {tuple(resume_bounds)} after an abandoned segment; the first segment's "
            "delivered prefix was dropped"
        )
        assert holds(kit, mgr, target, 0, [0, 1], staged_bytes(kit, 0x11)), (
            "the first segment's rows do not hold the bytes staged for them"
        )


LOST_SEGMENTS = {
    # name: (the first lease's end, the later lease's end, resume_bounds expected; None: empty)
    "window_keeps_the_blocks": (TPB, 2 * TPB, (TPB, 0)),
    "window_released_blocks": (2 * TPB, 4 * TPB, None),
}


@pytest.mark.parametrize("lost", ["abandoned", "nothing_arrived"])
@pytest.mark.parametrize("case", list(LOST_SEGMENTS))
def test_a_lost_later_segment_empties_the_interval_only_where_a_window_released_blocks(
    kit, real_manager, case, lost, attach=attach
):
    """All-reusable, a window of 64 tokens: a fetch in two leases, the first delivered whole, the
    later one lost after moving the history to its end. Where no window released blocks there, the
    request may resume below that end, up to where the first lease's rows reach, and its next chunk
    runs from there; where one did, the floor is that end, which no row reaches."""
    first_end, later_end, expected = LOST_SEGMENTS[case]
    with real_manager(windows=[WINDOW, 256]) as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        fetch_segment(kit, mgr, lender, target, 0, first_end, 0x11)
        for lg in range(kit.num_layer_groups(mgr)):
            assert holds(kit, mgr, target, lg, range(first_end // TPB), staged_bytes(kit, 0x11))
        later = lender.lend_write(target, first_end, later_end, **kit.HOST_NAMED)
        assert kv.history_length == later_end, "the later lease did not move the history"
        if lost == "abandoned":
            later.release()  # before its first poll
            resume_bounds = lender.poll_resume_bounds(target)
        else:
            # No row arrived: the return copies nothing, so it settles the fetch.
            resume_bounds = kit.return_write(lender, later, later.poll().row_masks(False))
            assert resume_bounds == lender.poll_resume_bounds(target), resume_bounds
        assert stale_anywhere(kit, mgr, later_end) == (expected is None)
        if expected is None:
            assert resume_bounds.min_position == later_end > resume_bounds.max_position, (
                f"resume_bounds {tuple(resume_bounds)} claims a start whose window blocks were released"
            )
            return
        assert resume_bounds == expected, (
            f"resume_bounds {tuple(resume_bounds)}: no resume below the later lease's end {later_end}, "
            "where no window released blocks"
        )
        resumed(kit, mgr, target, resume_bounds.max_position)
        assert next_chunk(kit, mgr, target) is None


def tp_rank(rank):
    """Rank ``rank`` of two tensor-parallel ranks."""
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=2, tp_size=2, rank=rank)


RAISED_ON_ONE_RANK = "a ValueError on one rank only"


def test_whether_an_earlier_fetch_settled_is_each_rank_s_own(kit, real_manager, attach=attach):
    """Two TP ranks: the one whose first fetch is unsettled fails the second lease at the call."""
    with (
        real_manager(num_kv_heads=4, mapping=tp_rank(0)) as rank0,
        real_manager(num_kv_heads=4, mapping=tp_rank(1)) as rank1,
    ):
        ranks = (rank0, rank1)
        targets = [kit.admitted(mgr, TARGET, SPLIT_PROMPT) for mgr in ranks]
        lenders = [
            attach(mgr, guaranteed_fetch_tokens=2 * TPB, capacity_in_fetches=2) for mgr in ranks
        ]
        firsts = [
            lender.lend_write(t, 0, 2 * TPB, **kit.HOST_NAMED)
            for lender, t in zip(lenders, targets)
        ]
        views = [kit.ready_view(lease, mgr) for lease, mgr in zip(firsts, ranks)]
        for lender, view in zip(lenders, views):
            kit.stage(lender, view, 0x5A)  # the backend filled the first segment on both ranks
        kit.return_write(lenders[0], firsts[0], views[0].row_masks(True))
        rank0._stream.synchronize()  # rank 0's copy into its pages has run
        seconds, outcomes = [], []
        with kit.held_stream(rank1._stream, strict=False):
            # Rank 1's copy stays queued.
            kit.return_write(lenders[1], firsts[1], views[1].row_masks(True))
            for lender in lenders:
                lender.progress()
            before = [
                state_of(kit, mgr, lender, t) for mgr, lender, t in zip(ranks, lenders, targets)
            ]
            for lender, target in zip(lenders, targets):
                try:
                    lease = lender.lend_write(target, 2 * TPB, 4 * TPB, **kit.HOST_NAMED)
                except ValueError as error:
                    outcomes.append(f"ValueError: {error}")
                    continue
                seconds.append(lease)
                outcomes.append("lease" if lease.failure is None else "failed at the call")
            after = state_of(kit, rank1, lenders[1], targets[1])
        try:
            assert before[0][1] is not None and before[1][1] is None, (
                f"resume_bounds {before}: rank 1's copy was not held, so this proves nothing"
            )
            raised = [outcome.startswith("ValueError") for outcome in outcomes]
            assert not any(raised), f"{RAISED_ON_ONE_RANK}: {outcomes}"
            assert outcomes == ["lease", "failed at the call"], outcomes
            assert after == before[1], "the lease failed at the call changed rank 1's cache"
            for lease in seconds:
                lease.release()
            seconds = []
            # The caller lends the next segment once every rank's resume_bounds is not None.
            for mgr, lender in zip(ranks, lenders):
                mgr._stream.synchronize()
                lender.progress()
            assert all(
                lender.poll_resume_bounds(t) is not None for lender, t in zip(lenders, targets)
            )
            for rank, (mgr, target) in enumerate(zip(ranks, targets)):
                assert holds(kit, mgr, target, 0, [0, 1], staged_bytes(kit, 0x5A)), (
                    f"rank {rank}: the first segment's rows landed off their blocks"
                )
            seconds = [
                lender.lend_write(t, 2 * TPB, 4 * TPB, **kit.HOST_NAMED)
                for lender, t in zip(lenders, targets)
            ]
            assert [lease.failure for lease in seconds] == [None, None]
            for lease, mgr, lender in zip(seconds, ranks, lenders):
                kit.return_write(lender, lease, kit.ready_view(lease, mgr).row_masks())
        finally:
            for mgr in ranks:
                mgr._stream.synchronize()
            for lease in [*firsts, *seconds]:
                lease.release()


FRESH_FILL = 2.5  # every fp16 element 0x4100: no staged byte pattern reads as it


def test_a_resume_with_the_fresh_page_fill_on_keeps_the_fetched_blocks(
    kit, real_manager, monkeypatch
):
    """The manager's fresh-page fill writes pages as a request first gets them. The pages a fetch
    grew are the request's own once the fetch lands: resuming fills only pages past them."""
    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", str(FRESH_FILL))
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        fetch_segment(kit, mgr, lender, target, 0, 64, 0x5A)
        assert lender.poll_resume_bounds(target).max_position == 64
        # The executor resumes the request where the fetch left it usable.
        target.context_chunk_size = target.prompt_len - target.context_current_position
        target.set_prepopulated_prompt_len(64, TPB)
        target.context_chunk_size = target.prompt_len - 64
        assert mgr.resize_context(target, target.context_remaining_length)
        mgr._stream.synchronize()
        dev = kit.DevicePages(mgr)
        fetched = [kit.digest(kit.staged_row(0x5A, 0, o, dev.page_bytes(0))) for o in range(2)]
        filled = np.full(dev.page_bytes(0) // 2, FRESH_FILL, dtype=np.float16).tobytes()
        got = [kit.digest(kit.page(mgr, target, 0, ordinal)) for ordinal in range(3)]
        assert got[2] == kit.digest(filled), "the fill did not run on the resume"
        assert got[:2] == fetched, "the resume filled the fetched blocks as fresh pages"
        mgr.free_resources(target)


def test_a_draft_pool_fetch_fills_only_the_pages_its_grow_added(kit, real_manager, monkeypatch):
    """A joint-reuse draft pool's own context resizes run no fresh-page fill. Under per_request the
    draft computed ``[0, LOCAL)`` and committed nothing; a fetch of the rest through the draft
    pool's lender fills the pages its grow added and leaves the blocks the draft computed."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", str(FRESH_FILL))
    spec = dflash()
    with real_manager(
        spec_config=spec, is_draft=True, joint_kv_cache_reuse=True, **PER_REQUEST
    ) as draft:
        request = kit.admitted(draft, TARGET, SPLIT_PROMPT)
        kv = kit.kv(draft, request)
        dev = kit.DevicePages(draft)
        # The draft pool's first chunk as the executor runs it: admission, forward, update.
        request.context_chunk_size = LOCAL
        assert draft.try_allocate_draft_context(request, LOCAL)
        for lg in range(kit.num_layer_groups(draft)):
            for ordinal, slot in enumerate(kit.pages(kv, lg)):
                if ordinal < LOCAL // TPB and slot >= 0:
                    dev.write(lg, slot, chunk_bytes(dev, lg, ordinal))
        request.move_to_next_context_chunk()
        batch = ScheduledRequests()
        batch.append_context_request(request)
        draft.update_context_resources(batch)
        assert (kv.history_length, kv.num_committed_tokens) == (LOCAL, 0), "the chunk committed"
        assert request.py_request_id not in draft._fresh_pages_filled, (
            "the draft pool's own resizes ran the fill: the check proves nothing"
        )
        lender = attach(draft, guaranteed_fetch_tokens=2 * TPB, scope=SCOPE + b"/draft-pool")
        blocks_before = kv.num_blocks
        lease = lender.lend_write(request, LOCAL, 4 * TPB, **kit.HOST_NAMED)
        view = kit.ready_view(lease, draft)
        torch.cuda.synchronize()  # the fill runs on the current stream
        filled = kit.digest(np.full(dev.page_bytes(0) // 2, FRESH_FILL, np.float16).tobytes())
        added = range(blocks_before, 4)
        assert len(added) and all(
            kit.digest(kit.page(draft, request, 0, o)) == filled for o in added
        ), "the fill did not run on the pages the fetch grew"
        assert holds(kit, draft, request, 0, range(LOCAL // TPB), chunk_bytes), (
            "the fetch's fill overwrote the blocks the draft computed"
        )
        kit.stage(lender, view, 0x5A)
        bounds = return_and_settle(lender, draft, request, lease, view.row_masks(True))
        assert holds(kit, draft, request, 0, [2, 3], staged_bytes(kit, 0x5A))
        assert bounds == (4 * TPB, LOCAL)


def test_a_second_draft_pool_fetch_fills_only_the_pages_its_own_grow_added(
    kit, real_manager, monkeypatch
):
    """Under per_request, a first fetch through a joint-reuse draft pool's lender leaves the
    fresh-page fill a record of the pages it grew. The draft then computes a chunk through its own
    context resize, which runs no fill and leaves that record stale. A second fetch fills only the
    pages its own grow added and leaves the chunk the draft computed."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", str(FRESH_FILL))
    spec = dflash()
    with real_manager(
        spec_config=spec, is_draft=True, joint_kv_cache_reuse=True, **PER_REQUEST
    ) as draft:
        request = kit.admitted(draft, TARGET, LONG_PROMPT)
        kv = kit.kv(draft, request)
        dev = kit.DevicePages(draft)
        lender = attach(draft, guaranteed_fetch_tokens=2 * TPB, scope=SCOPE + b"/draft-pool")
        fetch_segment(kit, draft, lender, request, 0, LOCAL, 0x5A)
        assert lender.poll_resume_bounds(request).max_position == LOCAL
        # The draft pool computes blocks 2 to 4 as the executor runs it after the resume.
        chunk = range(LOCAL // TPB, 5)
        resumed(kit, draft, request, LOCAL)
        request.context_chunk_size = len(chunk) * TPB
        assert draft.try_allocate_draft_context(request, len(chunk) * TPB)
        for lg in range(kit.num_layer_groups(draft)):
            slots = kit.pages(kv, lg)
            for ordinal in chunk:
                if slots[ordinal] >= 0:
                    dev.write(lg, slots[ordinal], chunk_bytes(dev, lg, ordinal))
        request.move_to_next_context_chunk()
        batch = ScheduledRequests()
        batch.append_context_request(request)
        draft.update_context_resources(batch)
        assert (kv.history_length, kv.num_committed_tokens) == (5 * TPB, 0), "the chunk committed"
        record = draft._fresh_pages_filled.get(request.py_request_id, {})
        known = min((len(pages) for pages in record.values()), default=kv.num_blocks)
        assert known < chunk[-1] + 1, (
            f"the fill's record covers {known} blocks, the draft's whole chunk: nothing to check"
        )
        blocks_before = kv.num_blocks
        lease = lender.lend_write(request, 5 * TPB, 7 * TPB, **kit.HOST_NAMED)
        view = kit.ready_view(lease, draft)
        torch.cuda.synchronize()  # the fill runs on the current stream
        assert holds(kit, draft, request, 0, chunk, chunk_bytes), (
            "the second fetch's fill overwrote the chunk the draft computed"
        )
        filled = kit.digest(np.full(dev.page_bytes(0) // 2, FRESH_FILL, np.float16).tobytes())
        grown = range(blocks_before, kv.num_blocks)
        assert len(grown) and all(
            kit.digest(kit.page(draft, request, 0, o)) == filled for o in grown
        ), "the fill did not run on the pages the second fetch grew"
        kit.stage(lender, view, 0x6B)
        bounds = return_and_settle(lender, draft, request, lease, view.row_masks(True))
        assert holds(kit, draft, request, 0, [0, 1], staged_bytes(kit, 0x5A))
        assert holds(kit, draft, request, 0, [5, 6], staged_bytes(kit, 0x6B))
        assert bounds == (7 * TPB, 5 * TPB)


@pytest.mark.parametrize("taker", ["context", "fetch"])
def test_the_fresh_page_fill_lets_a_freed_publisher_s_copy_run_first(
    kit, real_manager, monkeypatch, taker, fill_liar=None
):
    """A publish queues its copy into staging on the manager's stream, then its request may exit.
    With the fresh-page fill on, a request taking its pages by the context path or a fetch's grow
    fills them on the current stream after that copy, so staging holds the publisher's bytes."""
    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", str(FRESH_FILL))
    # No host tier: the freed pages go straight to the next request.
    with real_manager(max_tokens=kit.POOL_TOKENS, host_cache_size=0, max_batch_size=16) as mgr:
        # A process's first cache close waits for the whole device, the gated stream included:
        # close one before the gate.
        mgr.free_resources(kit.published(mgr, 900, list(range(40_000, 40_000 + TPB + 1))))
        torch.cuda.synchronize()
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        lent = set(kit.pages(kit.kv(mgr, source), 0)[:BLOCKS])
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        fetching = kit.admitted(mgr, TARGET, OTHER_PROMPT) if taker == "fetch" else None
        others = kit.Requests(mgr)
        # Other requests take every free page. Their fills load the fill's kernels before the
        # gate: a first load waits for the whole device.
        while others.allocate(1, chunk=1):
            pass
        fill = mgr._fill_fresh_kv_pages
        if fill_liar is not None:
            fill = fill_liar(fill)
        began = []  # per fill: whether the gate had opened when it began

        def reject_device_sync(*args, **kwargs):
            raise AssertionError("the diagnostic fill waited for the whole device")

        def recording_fill(request_id):
            unrelated_stream = torch.cuda.Stream()
            with kit.held_stream(unrelated_stream, strict=False) as unrelated_gate:
                with monkeypatch.context() as guard:
                    guard.setattr(torch.cuda, "synchronize", reject_device_sync)
                    began.append(gate.opened)
                    fill(request_id)
                assert not unrelated_gate.opened, "the fill waited for an unrelated stream"

        mgr._fill_fresh_kv_pages = recording_fill  # dropped below: it holds the manager
        write = None
        try:
            with kit.gated_stream(mgr._stream, open_after=10.0) as gate:
                read = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # its copy held
                mgr.free_resources(source)
                if taker == "context":
                    taken = kit.make_request(TARGET, OTHER_PROMPT)
                    assert mgr.prepare_context(taken)
                    assert mgr.resize_context(taken, taken.context_remaining_length)
                else:
                    taken = fetching
                    write = lender.lend_write(taken, 0, END, **kit.HOST_NAMED)
                    assert write.failure is None, write.failure
        finally:
            del mgr._fill_fresh_kv_pages
        assert began and not began[0], "the copy ran before the fill: the check proves nothing"
        assert lent & set(kit.pages(kit.kv(mgr, taken), 0)), "no freed page was taken"
        (run,) = kit.ready_view(read, mgr).groups
        length = lender.regions[run.region_index].slot_bytes
        got = kit.digest([kit.host_bytes(a, length) for a in run.addresses.tolist()])
        assert got == kit.digest(original), "the fill overwrote a page before its copy ran"
        read.release()
        if write is not None:
            write.release()
        mgr.free_resources(taken)
        others.free()


@pytest.mark.parametrize("ask_between", [False, True], ids=["regrown_unasked", "asked_between"])
def test_a_rollback_voids_what_the_fetch_delivered(kit, real_manager, ask_between, attach=attach):
    """The manager's own context rollback undoes the growth ``lend_write`` made: the same cache
    shrinks and frees the fetched pages, which others overwrite before the cache grows back.
    ``ask_between``: resume_bounds is asked between the rollback and the regrow."""
    with real_manager() as mgr_a, real_manager(**SMALL_POOL) as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        source_pages = [kit.digest(kit.page(mgr_a, source, 0, b)) for b in range(BLOCKS)]
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=END)
        kv = kit.kv(mgr_b, target)
        lease = lender_b.lend_write(target, 0, END, **kit.HOST_NAMED)
        assert target.py_ctx_pre_resize_cap == 0, "the rollback would not undo the growth"
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        assert return_and_settle(lender_b, mgr_b, target, lease, masks) == (END, 0)
        publish.release()
        assert [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)] == source_pages

        assert mgr_b.revert_allocate_context(target) is True
        assert kit.kv(mgr_b, target) is kv, "the rollback replaced the cache"
        assert kv.capacity == 0 and kit.pages(kv, 0) == [] and kv.num_committed_tokens == 0
        if ask_between:
            resume_bounds = lender_b.poll_resume_bounds(target)
            assert resume_bounds == (0, 0), (
                f"resume_bounds {tuple(resume_bounds)} counts freed blocks"
            )
        assert kit.overwrite_free_pages(mgr_b) >= BLOCKS
        assert mgr_b.resize_context(target, END)
        regrown = [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)]
        assert regrown != source_pages, "the regrown pages still hold the fetched bytes"
        resume_bounds = lender_b.poll_resume_bounds(target)
        assert resume_bounds == (0, 0), (
            f"resume_bounds {tuple(resume_bounds)} counts blocks whose pages now hold {regrown}"
        )
        mgr_b.free_resources(target)


def test_resume_bounds_voids_the_rows_of_a_shrink_no_hook_reported(kit, real_manager):
    unreported = attach_breaking({"_on_shrink": lambda self, request_id, kv_cache: None})
    test_a_rollback_voids_what_the_fetch_delivered(
        kit, real_manager, ask_between=True, attach=unreported
    )


def test_a_rollback_to_a_local_prefix_keeps_only_the_prefix(kit, real_manager, attach=attach):
    """The rollback's other branch: a local match of two blocks, a fetch of the third; the rollback
    shrinks to the prefix and suspends the cache."""
    local = 2 * TPB
    with real_manager() as mgr_a, real_manager(**SMALL_POOL) as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        kit.published(mgr_b, 7, PROMPT[:local] + list(range(9000, 9033)))
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=END)
        kv = kit.kv(mgr_b, target)
        assert kv.num_committed_tokens == local
        lease = lender_b.lend_write(target, local, END, **kit.HOST_NAMED)
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target, first_block=2)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        assert return_and_settle(lender_b, mgr_b, target, lease, masks) == (END, local)
        publish.release()
        got = kit.digest(kit.page(mgr_b, target, 0, 2))
        assert got == kit.digest(kit.page(mgr_a, source, 0, 2)), (
            "the fetched block 2 holds bytes other than the publisher's"
        )
        assert mgr_b.revert_allocate_context(target) is True
        assert kit.kv(mgr_b, target) is kv
        assert (kv.capacity, kv.is_active, kv.num_committed_tokens) == (local, False, local)
        resume_bounds = lender_b.poll_resume_bounds(target)
        assert resume_bounds == (local, local), (
            f"resume_bounds {tuple(resume_bounds)} counts block 2, which the rollback freed"
        )
        mgr_b.free_resources(target)


GENERATION_SHRINKS = ["revert_allocate_generation", "update_resources"]
GENERATION_DRAFT = 2 * TPB - 1  # draft tokens the generation path grows the cache for


@pytest.mark.parametrize("path", GENERATION_SHRINKS)
def test_a_generation_shrink_voids_the_fetched_rows_past_the_capacity(
    kit, real_manager, path, attach=attach
):
    """The generation path grows the cache for a token and ``GENERATION_DRAFT`` draft tokens, and a
    fetch of ``[0, END)`` grows it on to ``END``. ``revert_allocate_generation``, or the generation
    update reclaiming every draft slot, shrinks the same cache and leaves fetched blocks past its
    capacity; others overwrite the freed pages before the generation path grows the cache back."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import LlmRequestState

    with real_manager() as mgr_a, real_manager(**SMALL_POOL) as mgr_b:
        source = kit.published(mgr_a, SOURCE, PROMPT)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=END)
        publish = lender_a.lend_read(source, 0, END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        source_pages = [kit.digest(kit.page(mgr_a, source, 0, b)) for b in range(BLOCKS)]
        target = kit.admitted(mgr_b, TARGET, PROMPT)
        target.py_draft_tokens = [0] * GENERATION_DRAFT
        assert mgr_b.try_allocate_generation(target)
        kv = kit.kv(mgr_b, target)
        assert kv.capacity == 1 + GENERATION_DRAFT
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=END)
        lease = lender_b.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        assert return_and_settle(lender_b, mgr_b, target, lease, masks) == (END, 0)
        publish.release()
        assert [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)] == source_pages

        if path == "revert_allocate_generation":
            mgr_b.revert_allocate_generation(target)
        else:
            # The update moves the history to the request's last token, which no capacity goes
            # below, unless KV compression manages the history.
            mgr_b.kv_compression_manages_history = True
            target.state = LlmRequestState.GENERATION_IN_PROGRESS
            batch = ScheduledRequests()
            batch.append_generation_request(target)
            mgr_b.update_resources(batch)
        assert kit.kv(mgr_b, target) is kv, "the shrink replaced the cache"
        kept = kv.num_blocks
        assert 0 < kept < BLOCKS, f"{kept} blocks left: the shrink freed no fetched block, or all"
        assert kit.overwrite_free_pages(mgr_b) >= BLOCKS - kept
        assert mgr_b.try_allocate_generation(target)
        assert kv.num_blocks >= BLOCKS
        regrown = [kit.digest(kit.page(mgr_b, target, 0, b)) for b in range(BLOCKS)]
        assert regrown[:kept] == source_pages[:kept], "a block the shrink kept lost its bytes"
        assert all(r != s for r, s in zip(regrown[kept:], source_pages[kept:])), (
            "a regrown page still holds the fetched bytes"
        )
        resume_bounds = lender_b.poll_resume_bounds(target)
        assert resume_bounds == (kept * TPB, 0), (
            f"resume_bounds {tuple(resume_bounds)} counts blocks whose pages now hold {regrown[kept:]}"
        )
        mgr_b.free_resources(target)


@pytest.mark.parametrize("path", GENERATION_SHRINKS)
def test_a_generation_shrink_no_hook_reported_is_caught(kit, real_manager, path):
    """A lender that ignores the shrink hook, as under a manager that never calls it, counts the
    rows the generation shrink freed once the cache grows back."""
    unreported = attach_breaking({"_on_shrink": lambda self, request_id, kv_cache: None})
    with pytest.raises(AssertionError, match="counts blocks whose pages now hold"):
        test_a_generation_shrink_voids_the_fetched_rows_past_the_capacity(
            kit, real_manager, path, attach=unreported
        )
        pytest.fail("a lender that ignores the shrink hook went unnoticed")


# -- a prefix computed before a fetch and not committed ---------------------------------------

LOCAL = 2 * TPB  # the context chunk computed before the fetch
PER_REQUEST = {"block_reuse_config": {"policy": "per_request"}}


def computed(kit, mgr, request_id, tokens, upto=LOCAL, conversation=None, **inputs):
    """A request carrying ``inputs`` whose first context chunk, ``[0, upto)``, ran and committed
    nothing, as under a reuse policy other than all-reusable; block ``b`` of layer group ``lg``
    holds ``chunk_bytes(dev, lg, b)``."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.conversation_params import ConversationParams

    request = kit.make_request(request_id, tokens, **inputs)
    if conversation is not None:
        request.py_conversation_params = ConversationParams(conversation_id=conversation)
    assert mgr.prepare_context(request)
    kv = kit.kv(mgr, request)
    kv.enable_swa_scratch_reuse = False  # a fetch target needs it off
    request.context_chunk_size = upto
    assert mgr.resize_context(request, upto)
    dev = kit.DevicePages(mgr)
    for lg in range(kit.num_layer_groups(mgr)):
        for ordinal, slot in enumerate(kit.pages(kv, lg)):
            if slot >= 0:
                dev.write(lg, slot, chunk_bytes(dev, lg, ordinal))
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.update_context_resources(batch)
    assert (kv.history_length, kv.num_committed_tokens) == (upto, 0), "the chunk was committed"
    return request


def chunk_bytes(dev, lg, ordinal):
    """What ``computed`` wrote into block ``ordinal`` of layer group ``lg``: bytes drawn from a
    generator seeded by both, so the same block of another layer group, or a page whose pools or
    pieces trade places, holds other bytes."""
    rng = np.random.default_rng([0x40, lg, ordinal])
    return rng.integers(0, 256, dev.page_bytes(lg), dtype=np.uint8).tobytes()


def staged_bytes(kit, byte):
    """What ``fetch_segment`` staged from ``byte`` into block ``ordinal`` of ``lg``."""
    return lambda dev, lg, ordinal: kit.staged_row(byte, lg, ordinal, dev.page_bytes(lg))


def holds(kit, mgr, request, lg, ordinals, expected):
    """Whether blocks ``ordinals`` of ``lg`` hold ``expected(dev, lg, ordinal)`` in the request's
    pages."""
    dev = kit.DevicePages(mgr)
    got = [kit.digest(kit.page(mgr, request, lg, o)) for o in ordinals]
    return got == [kit.digest(expected(dev, lg, o)) for o in ordinals]


OUTCOMES = ["abandoned", "delivered"]


@pytest.mark.parametrize("outcome", OUTCOMES)
@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_a_fetch_keeps_the_tokens_computed_before_it(
    kit, real_manager, policy, outcome, attach=attach
):
    """A chunk computed ``[0, LOCAL)`` and committed nothing; a fetch of the rest of the prompt is
    abandoned before its first poll, or delivered whole. The chunk's tokens count either way."""
    conversation = "kept-chunk" if policy == "per_conversation" else None
    with real_manager(block_reuse_config={"policy": policy}) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT, conversation=conversation)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        assert lender.poll_resume_bounds(target) == (LOCAL, LOCAL)
        if outcome == "abandoned":
            lease = lender.lend_write(target, LOCAL, 4 * TPB, **kit.HOST_NAMED)
            assert lender.poll_resume_bounds(target) is None, "a fetch counts from its call"
            lease.release()  # before its first poll: no backend wrote
            expected = (LOCAL, LOCAL)
        else:
            fetch_segment(kit, mgr, lender, target, LOCAL, 4 * TPB, 0x5A)
            assert holds(kit, mgr, target, 0, [2, 3], staged_bytes(kit, 0x5A))
            expected = (4 * TPB, LOCAL)
        assert holds(kit, mgr, target, 0, [0, 1], chunk_bytes), "the fetch changed the chunk"
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds == expected, (
            f"resume_bounds {tuple(resume_bounds)}: the tokens computed before the fetch were dropped"
        )


def test_a_windowed_fetch_counts_what_its_window_still_holds(kit, real_manager, attach=attach):
    """The chunk under a window of 96 tokens. A fetch of ``[LOCAL, 128)`` moves the history to 128:
    the window releases block 0 but keeps block 1, which it reads at 128 beside the fetched blocks 2
    and 3. Delivered whole, the request resumes at 128."""
    with real_manager(windows=[96, 256], **PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        sliding = kit.windows(mgr).index(96)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        fetch_segment(kit, mgr, lender, target, LOCAL, 4 * TPB, 0x5A)
        assert kv.history_length == 4 * TPB, "the fetch did not move the history to its end"
        assert kit.stale_blocks(mgr, sliding, 4 * TPB) == (0, 1)
        assert kit.pages(kv, sliding)[0] < 0, "the window kept block 0"
        assert holds(kit, mgr, target, sliding, [1], chunk_bytes), "the window lost block 1"
        assert holds(kit, mgr, target, sliding, [2, 3], staged_bytes(kit, 0x5A))
        assert holds(kit, mgr, target, 1 - sliding, [0, 1], chunk_bytes)
        assert holds(kit, mgr, target, 1 - sliding, [2, 3], staged_bytes(kit, 0x5A)), (
            "the full-attention group's fetched blocks do not hold the bytes staged for them"
        )
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds == (4 * TPB, 4 * TPB), (
            f"resume_bounds {tuple(resume_bounds)}: the tokens computed before the fetch were dropped"
        )


# name: (windows, the fetch's start, resume_bounds, or None for an empty interval)
WINDOW_STARTS = {
    "from_the_computed_prefix": ([WINDOW], 0, (4 * TPB, 4 * TPB)),
    "past_it": ([WINDOW], 2 * TPB, (4 * TPB, 4 * TPB)),
    "past_it_with_a_wider_window": ([WINDOW, 256], 2 * TPB, None),
}


@pytest.mark.parametrize("case", list(WINDOW_STARTS))
def test_a_fetch_counts_the_rows_each_window_reads_whatever_its_start(
    kit, real_manager, case, attach=attach
):
    """All-reusable, nothing computed, a fetch ending at 128: resume_bounds counts what each layer group
    reads at 128 among the delivered rows, whatever the fetch's start. The window of ``WINDOW``
    tokens reads blocks 2 and 3 alone; a group whose window covers the prompt reads 0 and 1 too."""
    windows, start, expected = WINDOW_STARTS[case]
    with real_manager(windows=windows) as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=4 * TPB)
        fetch_segment(kit, mgr, lender, target, start, 4 * TPB, 0x5A)
        sliding = kit.windows(mgr).index(WINDOW)
        assert holds(kit, mgr, target, sliding, [2, 3], staged_bytes(kit, 0x5A)), (
            "the window's fetched blocks lack their staged bytes"
        )
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.min_position == 4 * TPB, f"resume_bounds {tuple(resume_bounds)}"
        if expected is None:
            assert resume_bounds.max_position < resume_bounds.min_position, (
                f"resume_bounds {tuple(resume_bounds)} counts blocks 0 and 1, which the wider window reads "
                "and nothing delivered"
            )
        else:
            assert resume_bounds == expected, (
                f"resume_bounds {tuple(resume_bounds)}: the window at {4 * TPB} reads blocks 2 and 3 "
                "alone, which landed"
            )


def test_an_abandoned_windowed_fetch_claims_nothing(kit, real_manager, attach=attach):
    """The chunk under a window of 64 tokens. A fetch of ``[LOCAL, 128)`` moves the history to 128,
    and the window releases blocks 0 and 1, which a start of 64 reads; abandoned, the fetch wrote
    neither block 2 nor block 3, which a start of 128 reads. No start is usable."""
    with real_manager(windows=[WINDOW, 256], **PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        sliding = kit.windows(mgr).index(WINDOW)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        lender.lend_write(target, LOCAL, 4 * TPB, **kit.HOST_NAMED).release()  # before a poll
        assert kv.history_length == 4 * TPB, "the fetch did not move the history to its end"
        assert all(p < 0 for p in kit.pages(kv, sliding)[:2]), "the window kept the chunk"
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.min_position == 4 * TPB
        assert resume_bounds.max_position < resume_bounds.min_position, (
            f"resume_bounds {tuple(resume_bounds)} claims a start whose blocks were released or never "
            "written"
        )


def test_a_fetch_from_below_the_history_keeps_only_what_lies_below_it(
    kit, real_manager, monkeypatch, attach=attach
):
    """The chunk ``[0, LOCAL)``, then a fetch from block 1: its copy into pages writes block 1 and
    fails at block 3, so the fetch is abandoned with block 1 overwritten. Only block 0 counts."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager(**PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=3 * TPB)
        lease = lender.lend_write(target, TPB, 4 * TPB, **kit.HOST_NAMED)
        view = lease.poll()
        assert by_group(view) == {0: [1, 2, 3]}
        kit.stage(lender, view, 0x5A)
        driver = FailingDriver(_lender.drv, fail_on=2)
        arrived = tuple(np.isin(run.page_ordinals, [1, 3]) for run in view.groups)
        with monkeypatch.context() as patched:
            patched.setattr(_lender, "drv", driver)
            # Blocks 1 and 3 arrived: their staging slots are apart, so each takes its own copy.
            given = kit.return_write(lender, lease, arrived)
        assert driver.calls >= 2, "a single copy: nothing failed partway"
        assert holds(kit, mgr, target, 0, [0], chunk_bytes)
        assert holds(kit, mgr, target, 0, [1], staged_bytes(kit, 0x5A)), (
            "the copy did not bring block 1 its own row"
        )
        resume_bounds = lender.poll_resume_bounds(target)
        assert given == resume_bounds, f"the return of an abandoned fetch gave {given}"
        assert resume_bounds.max_position <= TPB, (
            f"resume_bounds {tuple(resume_bounds)} counts block 1, which the abandoned fetch overwrote"
        )
        # Nothing was delivered, so the request may resume only at its history, past TPB.
        assert resume_bounds == (TPB, LOCAL), (
            f"resume_bounds {tuple(resume_bounds)}: block 0, computed before the fetch, was dropped"
        )


def test_a_refetch_whose_copy_fails_partway_counts_none_of_the_rows_it_was_to_overwrite(
    kit, real_manager, monkeypatch, attach=attach
):
    """A refetch of blocks 1 to 3 whose copy fails after block 1's leaves only block 0 counted."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=4 * TPB)
        fetch_segment(kit, mgr, lender, target, 0, 4 * TPB, 0x11)
        assert lender.poll_resume_bounds(target) == (4 * TPB, 0)
        lease = lender.lend_write(target, TPB, 4 * TPB, **kit.HOST_NAMED)
        view = lease.poll()
        assert by_group(view) == {0: [1, 2, 3]}
        kit.stage(lender, view, 0x22)
        driver = FailingDriver(_lender.drv, fail_on=2)
        arrived = tuple(np.isin(run.page_ordinals, [1, 3]) for run in view.groups)
        with monkeypatch.context() as patched:
            patched.setattr(_lender, "drv", driver)
            # Blocks 1 and 3: their staging slots are apart, so each takes its own copy.
            given = kit.return_write(lender, lease, arrived)
        assert driver.calls >= 2, "a single copy: nothing failed partway"
        assert holds(kit, mgr, target, 0, [1], staged_bytes(kit, 0x22)), (
            "the copy did not bring block 1 the second fetch's row"
        )
        resume_bounds = lender.poll_resume_bounds(target)
        assert given == resume_bounds, f"the return of an abandoned fetch gave {given}"
        assert resume_bounds.max_position <= TPB, (
            f"resume_bounds {tuple(resume_bounds)} counts rows the failed copy was to overwrite"
        )
        assert resume_bounds == (TPB, 0), (
            f"resume_bounds {tuple(resume_bounds)}: block 0, which the second fetch left alone, was dropped"
        )


class StatsDiscardRaising:
    """A ``kv_cache_map`` entry whose ``discard_pending_stats`` raises, so the manager's free stops
    before the lender's free hook; the manager's other calls on it go to the cache."""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache

    def __getattr__(self, name):
        return getattr(self.kv_cache, name)

    def discard_pending_stats(self):
        raise RuntimeError("planted")


FAILED_COPIES = ["returned", "raised", "interrupted"]
TAIL_MIXED = "counts the committed tokens of block 1, which the failed copy may have mixed"


@pytest.mark.parametrize("failure", [None, *FAILED_COPIES], ids=["control", *FAILED_COPIES])
def test_a_copy_failing_over_the_block_the_committed_tokens_end_inside_empties_the_interval(
    kit, real_manager, monkeypatch, failure, attach=attach
):
    """A copy failing past the block the committed tokens end inside empties the interval."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager() as mgr:
        kit.published(mgr, SOURCE, SIBLING_PROMPT)
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == len(SHARED) and len(SHARED) % TPB, (
            "no partial match: the check proves nothing"
        )
        lender = attach(mgr, guaranteed_fetch_tokens=3 * TPB)
        lease = lender.lend_write(target, TPB, 4 * TPB, **kit.HOST_NAMED)
        view = lease.poll()
        assert view is not None, lease.failure
        assert by_group(view) == {0: [1, 2, 3]}
        kit.stage(lender, view, 0x5A)
        planted = {"raised": MemoryError, "interrupted": Interrupted}.get(failure)
        driver = FailingDriver(_lender.drv, fail_on=2 if failure else None, raises=planted)
        # Blocks 1 and 3: their staging slots are apart, so each takes its own copy.
        masks = tuple(np.isin(run.page_ordinals, [1, 3]) for run in view.groups)
        with monkeypatch.context() as patched:
            patched.setattr(_lender, "drv", driver)
            if planted is None:
                kit.return_write(lender, lease, masks)
            else:
                with pytest.raises(planted, match="planted"):
                    kit.return_write(lender, lease, masks)
        lease.release()
        mgr._stream.synchronize()
        lender.progress()
        assert driver.calls >= 2, "a single copy: nothing failed partway"
        for run in view.groups:  # block 1, in every layer group the view has
            assert holds(kit, mgr, target, run.layer_group_index, [1], staged_bytes(kit, 0x5A)), (
                f"group {run.layer_group_index}: the copy did not reach block 1: the check proves nothing"
            )
        if failure is None:
            resume_bounds = lender.poll_resume_bounds(target)
            assert resume_bounds == (2 * TPB, TPB), f"resume_bounds {tuple(resume_bounds)}"
            return
        for _ in range(2):  # and on a later call
            resume_bounds = lender.poll_resume_bounds(target)
            assert resume_bounds.max_position < resume_bounds.min_position, (
                f"resume_bounds {tuple(resume_bounds)} {TAIL_MIXED}"
            )
        mgr.kv_cache_map[TARGET] = StatsDiscardRaising(kv)
        with pytest.raises(RuntimeError, match="planted"):
            mgr.free_resources(target)
        assert (
            TARGET not in mgr.kv_cache_map and lender._ledger._mixed_tail_caches.get(TARGET) is kv
        ), "the free left the cache in the map or ran the lender's hook: the check proves nothing"
        # The free's steps past the raise, so the request id is admitted again.
        kv.close()
        mgr.index_mapper.remove_sequence(TARGET)
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.max_position >= resume_bounds.min_position, (
            f"resume_bounds {tuple(resume_bounds)}: the cache that replaced the failed one stays empty"
        )


def test_a_copy_failing_over_that_block_in_a_later_layer_group_empties_the_interval(
    kit, real_manager, monkeypatch, attach=attach
):
    """Arrivals in a later layer group alone, whose copy fails past block 1, empty the interval."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager(windows=[4 * TPB, 256]) as mgr:
        kit.published(mgr, SOURCE, SIBLING_PROMPT)
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        assert kit.kv(mgr, target).num_committed_tokens == len(SHARED), (
            "no partial match: the check proves nothing"
        )
        lender = attach(mgr, guaranteed_fetch_tokens=3 * TPB)
        fetch_segment(kit, mgr, lender, target, TPB, 4 * TPB, 0x11)
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.max_position >= resume_bounds.min_position, (
            f"resume_bounds {tuple(resume_bounds)} is empty before the refetch: the check proves nothing"
        )
        lease = lender.lend_write(target, TPB, 4 * TPB, **kit.HOST_NAMED)
        view = lease.poll()
        assert view is not None, lease.failure
        assert by_group(view) == {0: [1, 2, 3], 1: [1, 2, 3]}
        kit.stage(lender, view, 0x5A)
        driver = FailingDriver(_lender.drv, fail_on=2)
        # Blocks 1 and 3 of the second layer group: their staging slots are apart.
        first, *later = view.groups
        masks = (
            np.zeros(len(first), dtype=bool),
            *(np.isin(r.page_ordinals, [1, 3]) for r in later),
        )
        with monkeypatch.context() as patched:
            patched.setattr(_lender, "drv", driver)
            kit.return_write(lender, lease, masks)
        mgr._stream.synchronize()
        assert driver.calls >= 2, "a single copy: nothing failed partway"
        assert holds(kit, mgr, target, first.layer_group_index, [1], staged_bytes(kit, 0x11)), (
            "the copy reached block 1 of the first layer group: the check proves nothing"
        )
        assert holds(kit, mgr, target, later[0].layer_group_index, [1], staged_bytes(kit, 0x5A)), (
            "the copy did not reach block 1 of the second layer group: the check proves nothing"
        )
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.max_position < resume_bounds.min_position, (
            f"resume_bounds {tuple(resume_bounds)} {TAIL_MIXED}"
        )


def test_a_second_fetch_keeps_what_the_first_kept(kit, real_manager, attach=attach):
    """The chunk under a window of 64 tokens; a fetch of ``[LOCAL, 128)`` abandoned, which moved the
    history to 128; then a fetch of ``[128, 160)`` delivered. A start of 160 reads block 3, which
    nothing computed or fetched."""
    with real_manager(windows=[WINDOW, 256], **PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, WINDOWED_PROMPT)
        kv = kit.kv(mgr, target)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        lender.lend_write(target, LOCAL, 4 * TPB, **kit.HOST_NAMED).release()  # before a poll
        assert kv.history_length == 4 * TPB
        lender.progress()  # the abandoned fetch's slots come back
        fetch_segment(kit, mgr, lender, target, 4 * TPB, WINDOWED_END, 0x5A)
        for lg in range(kit.num_layer_groups(mgr)):
            assert holds(kit, mgr, target, lg, [4], staged_bytes(kit, 0x5A)), f"layer group {lg}"
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.min_position == WINDOWED_END
        assert resume_bounds.max_position < resume_bounds.min_position, (
            f"resume_bounds {tuple(resume_bounds)} counts block 3, which nothing computed or fetched"
        )


def test_a_rollback_keeps_the_computed_chunk_and_voids_the_fetches(
    kit, real_manager, attach=attach
):
    """The chunk ``[0, TPB)``, then two one-block fetches delivered; the manager's context rollback
    shrinks the cache back to the chunk, freeing both fetched blocks."""
    with real_manager(**PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT, upto=TPB)
        target.py_ctx_pre_resize_cap = None  # a later resize grew nothing
        kv = kit.kv(mgr, target)
        lender = attach(mgr, guaranteed_fetch_tokens=TPB)
        fetch_segment(kit, mgr, lender, target, TPB, 2 * TPB, 0x11)
        assert lender.poll_resume_bounds(target) == (2 * TPB, TPB)
        fetch_segment(kit, mgr, lender, target, 2 * TPB, 3 * TPB, 0x22)
        assert lender.poll_resume_bounds(target) == (3 * TPB, TPB)
        for lg in range(kit.num_layer_groups(mgr)):  # the rows that resume_bounds counts
            for ordinal, byte in ((1, 0x11), (2, 0x22)):
                assert holds(kit, mgr, target, lg, [ordinal], staged_bytes(kit, byte)), (
                    f"layer group {lg}: fetched block {ordinal} lacks its staged bytes"
                )
        assert target.py_ctx_pre_resize_cap == TPB, "the rollback would not shrink to the chunk"
        assert mgr.revert_allocate_context(target) is True
        assert kit.kv(mgr, target) is kv, "the rollback replaced the cache"
        assert (kv.capacity, kv.history_length, kv.num_blocks) == (TPB, TPB, 1)
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds == (TPB, TPB), (
            f"resume_bounds {tuple(resume_bounds)} counts block 1, which the rollback freed"
        )
        mgr.free_resources(target)


# -- where a request resumes when its context update moves the history ------------------------

LONG_PROMPT = list(range(4000, 4225))  # seven whole blocks and one token


def resumed(kit, mgr, request, at):
    """``documented_resume`` as far as the manager calls that follow read it: the history raised to
    ``at`` if below it, the context position moved there. Through the V2 scheduler a resume past the
    first chunk needs the whole step."""
    kv = kit.kv(mgr, request)
    if kv.history_length < at:
        kv.resize(None, at)
    request.context_current_position = at


def next_chunk(kit, mgr, request, tokens=TPB):
    """The request's next context chunk of ``tokens`` run as the executor runs it, block ``b``
    written with ``chunk_bytes``; the manager's error, or ``None``."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    first = request.context_current_position
    request.context_chunk_size = min(tokens, request.prompt_len - first)
    assert mgr.resize_context(request, request.context_chunk_size)
    kv = kit.kv(mgr, request)
    dev = kit.DevicePages(mgr)
    end_block = (first + request.context_chunk_size) // TPB
    for lg in range(kit.num_layer_groups(mgr)):
        for ordinal, slot in enumerate(kit.pages(kv, lg)):
            if first // TPB <= ordinal < end_block and slot >= 0:
                dev.write(lg, slot, chunk_bytes(dev, lg, ordinal))
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    try:
        mgr.update_context_resources(batch)
    except ValueError as error:
        return str(error)
    return None


RESUMES = {
    # name: (manager kwargs, prompt, tokens computed first, fetch range, block ordinals arrived
    # (None: all), resume_bounds expected (None: any interval whose starts all resume))
    "delivered": ({}, SPLIT_PROMPT, LOCAL, (LOCAL, 4 * TPB), None, (4 * TPB, LOCAL)),
    # A window that releases nothing by 128, and a fetch that delivered no row: a remote miss.
    "windowed_nothing_delivered": (
        {"windows": [4 * TPB, 256]},
        SPLIT_PROMPT,
        LOCAL,
        (LOCAL, 4 * TPB),
        [],
        None,
    ),
    # A window that releases nothing by 192, and a fetch whose first block alone arrived.
    "windowed_partial": (
        {"windows": [6 * TPB, 256]},
        LONG_PROMPT,
        LOCAL,
        (LOCAL, 6 * TPB),
        [2],
        None,
    ),
    # A fetch from below the history whose first block did not arrive.
    "below_the_history_partial": ({}, SPLIT_PROMPT, 4 * TPB, (TPB, 4 * TPB), [2, 3], None),
    # A fetch from below the history, delivered whole.
    "below_the_history_delivered": (
        {},
        LONG_PROMPT,
        4 * TPB,
        (TPB, 6 * TPB),
        None,
        (6 * TPB, 4 * TPB),
    ),
    # Nothing computed first, a window that releases nothing by 192, and two blocks arrived.
    "cold_windowed_partial": (
        {"windows": [6 * TPB, 256]},
        LONG_PROMPT,
        0,
        (0, 6 * TPB),
        [0, 1],
        None,
    ),
}


@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
@pytest.mark.parametrize("case", list(RESUMES))
def test_resume_bounds_claims_only_starts_the_context_update_takes(
    kit, real_manager, case, policy, attach=attach
):
    """Per request and per conversation, the context update moves the history to each chunk's end
    and raises for a chunk ending below it: the next chunk runs from the lowest start resume_bounds
    claims, and the floor is at least the history. No window released blocks, so none sets it."""
    from tensorrt_llm.conversation_params import ConversationParams

    kwargs, prompt, upto, (start, end), arrived, expected = RESUMES[case]
    conversation = "claims" if policy == "per_conversation" else None
    with real_manager(**kwargs, block_reuse_config={"policy": policy}) as mgr:
        if upto:
            target = computed(kit, mgr, TARGET, prompt, upto=upto, conversation=conversation)
        else:
            target = kit.make_request(TARGET, prompt)
            if conversation is not None:
                target.py_conversation_params = ConversationParams(conversation_id=conversation)
            assert mgr.prepare_context(target)
            kit.kv(mgr, target).enable_swa_scratch_reuse = False  # a fetch target needs it off
        kv = kit.kv(mgr, target)
        lender = attach(mgr, guaranteed_fetch_tokens=end - start)
        fetch_segment(kit, mgr, lender, target, start, end, 0x5A, arrived)
        delivered = range(start // TPB, end // TPB) if arrived is None else arrived
        for lg in range(kit.num_layer_groups(mgr)):
            assert holds(kit, mgr, target, lg, delivered, staged_bytes(kit, 0x5A)), f"group {lg}"
        history = kv.history_length
        assert not stale_anywhere(kit, mgr, history), "a window released blocks"
        resume_bounds = lender.poll_resume_bounds(target)
        floor = resume_bounds.min_position
        if resume_bounds.max_position >= floor:
            # The lowest start claimed: from a higher one the chunk ends no lower.
            resumed(kit, mgr, target, floor)
            error = next_chunk(kit, mgr, target)
            assert error is None, (
                f"resume_bounds {tuple(resume_bounds)} lets the request resume at {floor}, below its "
                f"history of {history}, where its next chunk fails: {error}"
            )
        assert floor >= history, (
            f"resume_bounds {tuple(resume_bounds)} floors below the history {history}"
        )
        if expected is not None:
            assert resume_bounds == expected, (
                f"resume_bounds {tuple(resume_bounds)}: what was computed or delivered was dropped"
            )


def test_a_later_fetch_keeps_what_the_request_computed_after_resuming(
    kit, real_manager, attach=attach
):
    """Per request: the chunk ``[0, LOCAL)``, a fetch of ``[LOCAL, 128)`` delivered, a resume at 128
    as a remote tier does, ``[128, 192)`` computed, then a fetch of ``[192, 224)`` delivered. Every
    block below 224 holds what the request computed or fetched, so it resumes at 224."""
    with real_manager(**PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, LONG_PROMPT)
        kv = kit.kv(mgr, target)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        fetch_segment(kit, mgr, lender, target, LOCAL, 4 * TPB, 0x5A)
        first = lender.poll_resume_bounds(target)
        assert first == (4 * TPB, LOCAL), f"resume_bounds {tuple(first)} after the first fetch"
        resumed(kit, mgr, target, first.max_position)
        assert next_chunk(kit, mgr, target, 2 * TPB) is None
        assert (kv.history_length, kv.num_committed_tokens) == (6 * TPB, 0), "the chunk committed"
        fetch_segment(kit, mgr, lender, target, 6 * TPB, 7 * TPB, 0x6B)
        assert holds(kit, mgr, target, 0, [0, 1, 4, 5], chunk_bytes)
        assert holds(kit, mgr, target, 0, [2, 3], staged_bytes(kit, 0x5A))
        assert holds(kit, mgr, target, 0, [6], staged_bytes(kit, 0x6B))
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds == (7 * TPB, 6 * TPB), (
            f"resume_bounds {tuple(resume_bounds)}: the tokens computed after resuming were dropped"
        )


# -- a joint-reuse draft pool, which shares the request's cursor ----------------------------

PAIRED_END = 128  # SPLIT_PROMPT's whole blocks before its last token


def schedule(scheduler, request) -> List[int]:
    """The ids of the context requests one round of the V2 scheduler admits."""
    return [r.py_request_id for r in scheduler.schedule_request([request], set()).context_requests]


def fill_blocks(kit, mgr, request, ordinals, byte):
    """Write the request's blocks ``ordinals`` as ``kit.stage`` stages them from ``byte``."""
    dev = kit.DevicePages(mgr)
    for lg in range(kit.num_layer_groups(mgr)):
        slots = kit.pages(kit.kv(mgr, request), lg)
        for ordinal in ordinals:
            dev.write(lg, slots[ordinal], kit.staged_row(byte, lg, ordinal, dev.page_bytes(lg)))


@pytest.mark.parametrize("draft_delivers", [True, False], ids=["draft_hit", "draft_miss"])
def test_a_joint_draft_pool_resumes_where_both_lenders_allow(
    kit, real_manager, draft_delivers, attach=attach
):
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.bindings import LlmRequestState
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

    spec, byte = dflash(), 0x5A  # what any worker computing these tokens holds
    with (
        real_manager(spec_config=spec, joint_kv_cache_reuse=True) as target,
        real_manager(spec_config=spec, is_draft=True, joint_kv_cache_reuse=True) as draft,
    ):
        pools = (target, draft)
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=2048,
            kv_cache_manager=target,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            draft_kv_cache_manager=draft,
        )
        # The draft pool under a scope of its own, as attach_lender asks of such a pair.
        lenders = [
            attach(target, guaranteed_fetch_tokens=PAIRED_END),
            attach(draft, guaranteed_fetch_tokens=PAIRED_END, scope=SCOPE + b"/draft-pool"),
        ]
        request = kit.make_request(TARGET, SPLIT_PROMPT)
        assert schedule(scheduler, request) == [TARGET]
        for mgr in pools:
            fill_blocks(kit, mgr, request, range(5), kit.SENTINEL)  # nothing computed yet
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS  # parked to fetch
        intervals = []
        for mgr, lender, delivered in zip(pools, lenders, (True, draft_delivers)):
            lease = lender.lend_write(request, 0, PAIRED_END, **kit.HOST_NAMED)
            view = lease.poll()
            kit.stage(lender, view, byte)
            intervals.append(
                return_and_settle(lender, mgr, request, lease, view.row_masks(delivered))
            )
        # Each interval covers only its own pool's blocks: the request resumes where both allow.
        usable = min(interval.max_position for interval in intervals)
        assert max(interval.min_position for interval in intervals) <= usable
        for mgr in pools:
            kv = kit.kv(mgr, request)
            if kv.history_length < usable:
                kv.resize(None, usable)
        request.py_served_prefix_end = usable
        request.state = LlmRequestState.CONTEXT_INIT
        assert schedule(scheduler, request) == [TARGET]
        assert request.context_current_position == usable
        batch = ScheduledRequests()
        batch.append_context_request(request)
        for mgr in pools:
            mgr.prepare_resources(batch)
        first = usable // TPB
        last = (usable + request.context_chunk_size - 1) // TPB
        for mgr in pools:  # the forward computes the chunk's blocks
            fill_blocks(kit, mgr, request, range(first, last + 1), byte)
        request.move_to_next_context_chunk()
        for mgr in pools:
            mgr.update_context_resources(batch)
        assert kit.kv(draft, request).num_committed_tokens == len(SPLIT_PROMPT)
        nbytes = kit.DevicePages(draft).page_bytes(0)
        blocks = range(PAIRED_END // TPB)
        held = [kit.digest(kit.staged_row(byte, 0, o, nbytes)) for o in blocks]
        committed = [kit.digest(kit.page(draft, request, 0, o)) for o in blocks]
        assert committed == held, "the draft pool committed blocks it lacks"
        for lg in range(kit.num_layer_groups(target)):
            nbytes = kit.DevicePages(target).page_bytes(lg)
            held = [kit.digest(kit.staged_row(byte, lg, o, nbytes)) for o in blocks]
            committed = [kit.digest(kit.page(target, request, lg, o)) for o in blocks]
            assert committed == held, "the target pool committed blocks it lacks"
        # The whole fetch is used where both pools' rows landed, none where the draft's did not.
        assert usable == (PAIRED_END if draft_delivers else 0), (
            f"resumed at {usable}, not where both pools' fetched blocks allow"
        )


JOINT_WINDOW = 192  # wider than PAIRED_END: no window releases a block at the fetch's end
# name: (block ordinals delivered into the target pool, into the draft pool)
JOINT_DELIVERIES = {
    "short_in_both": ([0, 1], [0, 1]),
    "draft_miss": ([0, 1, 2, 3], []),
    "full": ([0, 1, 2, 3], [0, 1, 2, 3]),
}


@pytest.mark.parametrize("case", list(JOINT_DELIVERIES))
def test_a_windowed_joint_draft_pool_resumes_only_where_its_context_resize_holds(
    kit, real_manager, case, attach=attach
):
    """All-reusable, both pools windowed wider than the fetch: the target pool may resume below the
    history the fetch moved; the draft pool's context resize sets its capacity from the chunk, so
    its floor is that history. Wherever both may resume, they prepare a chunk there smaller than the
    fetch and hold the staged bytes below it; a short delivery in either pool leaves none."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.bindings import LlmRequestState
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

    spec, byte, windows = dflash(), 0x5A, [JOINT_WINDOW]
    with (
        real_manager(spec_config=spec, joint_kv_cache_reuse=True, windows=windows) as target,
        real_manager(
            spec_config=spec, is_draft=True, joint_kv_cache_reuse=True, windows=windows
        ) as draft,
    ):
        pools = (target, draft)
        for mgr in pools:
            assert JOINT_WINDOW in kit.windows(mgr), "a pool has no window: nothing checked"
            beg, end = kit.stale_blocks(mgr, kit.windows(mgr).index(JOINT_WINDOW), PAIRED_END)
            assert end <= beg, "a window released blocks at the fetch's end: nothing checked"
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=2048,
            kv_cache_manager=target,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            draft_kv_cache_manager=draft,
        )
        lenders = [
            attach(target, guaranteed_fetch_tokens=PAIRED_END),
            attach(draft, guaranteed_fetch_tokens=PAIRED_END, scope=SCOPE + b"/draft-pool"),
        ]
        request = kit.make_request(TARGET, SPLIT_PROMPT)
        assert schedule(scheduler, request) == [TARGET]
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS  # parked to fetch
        intervals = []
        for mgr, lender, ordinals in zip(pools, lenders, JOINT_DELIVERIES[case]):
            lease = lender.lend_write(request, 0, PAIRED_END, **kit.HOST_NAMED)
            view = kit.ready_view(lease, mgr)
            kit.stage(lender, view, byte)
            arrived = tuple(np.isin(run.page_ordinals, ordinals) for run in view.groups)
            intervals.append(return_and_settle(lender, mgr, request, lease, arrived))
        floor = max(interval.min_position for interval in intervals)
        usable = min(interval.max_position for interval in intervals)
        if floor <= usable:
            # The pair resumes at the smaller usable end and runs a chunk smaller than the fetch.
            request.state = LlmRequestState.CONTEXT_INIT
            request.context_current_position = usable
            request.context_chunk_size = min(TPB, request.prompt_len - usable)
            assert target.resize_context(request, request.context_chunk_size)
            batch = ScheduledRequests()
            batch.append_context_request(request)
            try:
                draft.prepare_resources(batch)
            except (ValueError, RuntimeError) as error:
                raise AssertionError(
                    f"the draft pool's context resize at {usable} raised: {error}"
                ) from error
            for name, mgr in zip(("target", "draft"), pools):  # the resume reads every row below
                for lg in range(kit.num_layer_groups(mgr)):
                    rows = range(usable // TPB)
                    assert holds(kit, mgr, request, lg, rows, staged_bytes(kit, byte)), (
                        f"{name} pool, group {lg}: a row the resume reads lacks its staged bytes"
                    )
        if case == "full":
            assert (floor, usable) == (PAIRED_END, PAIRED_END), (floor, usable)
        else:
            assert floor > usable, f"the pair may resume in [{floor}, {usable}] after a short fetch"


# -- a request resumed past its first context chunk, through the V2 scheduler -----------------

RESUME_END = 6 * TPB  # the fetch's end in LONG_PROMPT, past the chunk the request computed first


def documented_resume(kit, pools, request, at):
    """``ResumeBounds``'s resume at ``at``, a KV cache connector's skip: the served position set, each
    cache's history raised to ``at`` where below it, then the chunk spanning to the prompt's end and
    the prepopulated length and context position moved to ``at`` together."""
    first_chunk_resume(kit, pools, request, at)
    request.context_chunk_size = request.prompt_len - request.context_current_position
    request.set_prepopulated_prompt_len(at, TPB)


def first_chunk_resume(kit, pools, request, at):  # the served position and the histories alone
    for mgr in pools:
        kv = kit.kv(mgr, request)
        if kv.history_length < at:
            kv.resize(None, at)
    request.py_served_prefix_end = at


def runs_its_chunk(kit, pools, request):
    """The chunk the scheduler gave the request, run as the executor runs it: each pool prepared,
    every block the chunk touches written with ``chunk_bytes``, the context position moved and each
    pool's context update run; the first update's error, or ``None``."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    batch = ScheduledRequests()
    batch.append_context_request(request)
    for mgr in pools:
        mgr.prepare_resources(batch)
    first = request.context_current_position
    end = first + request.context_chunk_size
    for mgr in pools:
        kv = kit.kv(mgr, request)
        dev = kit.DevicePages(mgr)
        for lg in range(kit.num_layer_groups(mgr)):
            for ordinal, slot in enumerate(kit.pages(kv, lg)):
                if first // TPB <= ordinal < -(-end // TPB) and slot >= 0:
                    dev.write(lg, slot, chunk_bytes(dev, lg, ordinal))
    request.move_to_next_context_chunk()
    try:
        for mgr in pools:
            mgr.update_context_resources(batch)
    except (ValueError, RuntimeError) as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    "target_policy,draft_policy",
    [
        ("per_request", None),
        ("per_request", "per_request"),
        ("per_request", "all_reusable"),
        ("all_reusable", "per_request"),
        ("all_reusable", "all_reusable"),
    ],
)
def test_a_request_past_its_first_chunk_resumes_where_the_documented_step_puts_it(
    kit, real_manager, target_policy, draft_policy, resume=documented_resume
):
    """Through the V2 scheduler chunking at ``LOCAL`` tokens, alone or with a same/mixed-policy
    joint-reuse draft pool: the request computes its first chunk, a fetch of ``[LOCAL, RESUME_END)``
    arrives whole, and the request resumes at ``max_position`` by the step ``ResumeBounds`` states. Its
    next chunk starts there with the prepopulated length moved alike, every block below holds what
    the request computed or the fetch delivered, and each pool's context update takes the chunk."""
    from contextlib import ExitStack

    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.bindings import LlmRequestState
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

    byte = 0x5A
    paired = draft_policy is not None
    target_config = {"block_reuse_config": {"policy": target_policy}}
    with ExitStack() as stack:
        if paired:
            spec = dflash()
            pools = (
                stack.enter_context(
                    real_manager(spec_config=spec, joint_kv_cache_reuse=True, **target_config)
                ),
                stack.enter_context(
                    real_manager(
                        spec_config=spec,
                        is_draft=True,
                        joint_kv_cache_reuse=True,
                        block_reuse_config={"policy": draft_policy},
                    )
                ),
            )
        else:
            pools = (stack.enter_context(real_manager(**target_config)),)
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=LOCAL,
            kv_cache_manager=pools[0],
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
            draft_kv_cache_manager=pools[1] if paired else None,
        )
        scopes = (SCOPE, SCOPE + b"/draft-pool")  # the draft pool under a scope of its own
        lenders = [
            attach(m, guaranteed_fetch_tokens=RESUME_END - LOCAL, scope=s)
            for m, s in zip(pools, scopes)
        ]
        request = kit.make_request(TARGET, LONG_PROMPT)
        assert schedule(scheduler, request) == [TARGET]
        for mgr in pools:
            kit.kv(mgr, request).enable_swa_scratch_reuse = False  # a fetch target needs it off
        assert runs_its_chunk(kit, pools, request) is None
        assert request.context_current_position == LOCAL, "the first chunk was not LOCAL tokens"
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS  # parked to fetch
        intervals = []
        for mgr, lender in zip(pools, lenders):
            fetch_segment(kit, mgr, lender, request, LOCAL, RESUME_END, byte)
            intervals.append(tuple(lender.poll_resume_bounds(request)))
        assert intervals == [(RESUME_END, LOCAL)] * len(pools), f"resume_bounds {intervals}"
        at = RESUME_END
        request.state = LlmRequestState.CONTEXT_INIT
        resume(kit, pools, request, at)
        assert schedule(scheduler, request) == [TARGET]
        assert request.context_current_position == at, (
            f"the next chunk starts at {request.context_current_position}, not at {at}"
        )
        assert request.prepopulated_prompt_len == at, (
            f"the prepopulated length stays at {request.prepopulated_prompt_len}, not at {at}"
        )
        computed_blocks, fetched_blocks = range(LOCAL // TPB), range(LOCAL // TPB, at // TPB)
        for mgr in pools:
            for lg in range(kit.num_layer_groups(mgr)):
                assert holds(kit, mgr, request, lg, computed_blocks, chunk_bytes), f"group {lg}"
                assert holds(kit, mgr, request, lg, fetched_blocks, staged_bytes(kit, byte)), (
                    f"group {lg}: a block the resume reads lacks its staged bytes"
                )
        error = runs_its_chunk(kit, pools, request)
        assert error is None, f"the context update did not take the chunk at {at}: {error}"
        for mgr in pools:
            assert kit.kv(mgr, request).history_length == request.prompt_len


SHIFTED_AT = 3 * TPB + 24  # inside the fetched block, past the context position


def test_a_request_resumes_at_an_unaligned_point_past_its_context_position(
    kit, real_manager, resume=documented_resume, below=False
):
    """All-reusable, through the V2 scheduler: a partial local match and a first chunk leave the
    context position inside block 3, and a fetch of block 3 starts the interval below it. Resumed
    past it at an unaligned point, the request's next chunk starts there over the bytes it reads."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.bindings import LlmRequestState
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

    byte, prompt = 0x5A, SHARED + list(range(7400, 7493))  # four whole blocks and five tokens
    with real_manager() as mgr:
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=LOCAL,
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
        )
        sibling = kit.published(mgr, SOURCE, SIBLING_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=TPB)
        request = kit.make_request(TARGET, prompt)
        assert schedule(scheduler, request) == [TARGET]
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens == len(SHARED), "no partial match: the check proves nothing"
        kv.enable_swa_scratch_reuse = False  # a fetch target needs it off
        assert runs_its_chunk(kit, [mgr], request) is None
        position = len(SHARED) + LOCAL
        assert request.context_current_position == position, "the first chunk was not LOCAL tokens"
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS  # parked to fetch
        fetch_segment(kit, mgr, lender, request, 3 * TPB, 4 * TPB, byte)
        resume_bounds = lender.poll_resume_bounds(request)
        assert tuple(resume_bounds) == (4 * TPB, 3 * TPB), f"resume_bounds {tuple(resume_bounds)}"
        at = resume_bounds.min_position if below else SHIFTED_AT
        request.state = LlmRequestState.CONTEXT_INIT
        try:
            resume(kit, [mgr], request, at)
        except RuntimeError as error:
            raise AssertionError(f"the step at {at} raised: {error}") from error
        try:
            scheduled = schedule(scheduler, request)
        except RuntimeError as error:
            raise AssertionError(f"the next scheduling pass raised: {error}") from error
        assert scheduled == [TARGET]
        assert request.context_current_position == at, (
            f"the next chunk starts at {request.context_current_position}, not at {at}"
        )
        assert request.prepopulated_prompt_len == at, (
            f"the prepopulated length stays at {request.prepopulated_prompt_len}, not at {at}"
        )
        block_0 = kit.chain_keys(kit.kv(mgr, sibling), SIBLING_PROMPT)[0]

        # what the sibling's prefill wrote, not read back from its page
        def written(dev, lg, ordinal):
            return kit.fake_page(lg, block_0, dev.page_bytes(lg))

        for lg in range(kit.num_layer_groups(mgr)):
            assert holds(kit, mgr, request, lg, [0], written), f"group {lg}: block 0"
            assert holds(kit, mgr, request, lg, [1, 2], chunk_bytes), f"group {lg}: blocks 1-2"
            assert holds(kit, mgr, request, lg, [3], staged_bytes(kit, byte)), (
                f"group {lg}: the fetched block lacks its staged bytes"
            )
        error = runs_its_chunk(kit, [mgr], request)
        assert error is None, f"the context update did not take the chunk at {at}: {error}"
        assert kv.history_length == request.prompt_len


# -- failures at the call ---------------------------------------------------------------------


@pytest.mark.parametrize("state", ["no_cache", "suspended", "shut_down"])
def test_a_lease_failed_at_the_call_changed_nothing(kit, real_manager, state):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        if state == "no_cache":
            source, target = kit.make_request(8, PROMPT), kit.make_request(9, OTHER_PROMPT)
        elif state == "suspended":
            for request in (source, target):
                mgr.suspend_request(request)
                assert not kit.kv(mgr, request).is_active
        else:
            mgr.shutdown()
        before = [state_of(kit, mgr, lender, r) for r in (source, target)]
        leases = []
        for lend, request in ((lender.lend_read, source), (lender.lend_write, target)):
            if state == "shut_down":
                with pytest.raises(RuntimeError, match="lends no more"):
                    lend(request, 0, END, **kit.HOST_NAMED)
            else:
                leases.append(lend(request, 0, END, **kit.HOST_NAMED))
        for lease in leases:
            assert lease.failure is not None, "not failed at the call"
            assert lease.poll() is None
        assert [state_of(kit, mgr, lender, r) for r in (source, target)] == before
        for lease in leases:
            lease.release()
            lease.release()


def test_a_write_the_pool_cannot_grow_fails_at_the_call_and_changed_nothing(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        others = kit.Requests(mgr)
        assert others.allocate(kit.pool_pages(mgr)), "others could not take the whole pool"
        before = state_of(kit, mgr, lender, target)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        assert "no free pages" in str(lease.failure) and lease.poll() is None
        assert state_of(kit, mgr, lender, target) == before
        lease.release()
        others.free()
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)  # pages free: it goes through
        view = lease.poll()
        assert view is not None
        kit.return_write(lender, lease, view.row_masks())


def test_a_write_the_pool_cannot_grow_leaves_a_computed_prefix_as_it_was(
    kit, real_manager, attach=attach
):
    """Per request: the chunk ``[0, LOCAL)`` computed and not committed, then a fetch of
    ``[TPB, 3 * TPB)``, from below the history, that the pool cannot grow the cache for. The lease
    fails at the call, and the cache and its resume_bounds stay as they were: the chunk still counts."""
    with real_manager(max_tokens=kit.POOL_TOKENS, **PER_REQUEST) as mgr:
        target = computed(kit, mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        others = kit.Requests(mgr)
        free = kit.pool_pages(mgr) - int(kit.kv(mgr, target).num_blocks)  # past the chunk's pages
        assert others.allocate(free), "others could not take the rest of the pool"
        before = state_of(kit, mgr, lender, target)
        assert before[1] == (LOCAL, LOCAL), f"resume_bounds before the fetch {before[1]}"
        lease = lender.lend_write(target, TPB, 3 * TPB, **kit.HOST_NAMED)
        assert "no free pages" in str(lease.failure) and lease.poll() is None, lease.failure
        after = state_of(kit, mgr, lender, target)
        assert after == before, f"the failed grow changed the cache or its resume_bounds: {after}"
        lease.release()
        others.free()


@pytest.mark.parametrize("keywords", ["HOST_NAMED", "DEVICE_NAMED"], ids=["host", "device"])
@pytest.mark.parametrize("poll", [True, False], ids=["poll_failure", "release_before_poll"])
def test_a_write_left_with_a_block_without_a_page_fails_at_its_first_poll(
    kit, real_manager, monkeypatch, keywords, poll
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        kv = kit.kv(mgr, target)

        def without_block_one(read):
            def pages(kv_cache, lg):
                found = np.array(read(kv_cache, lg), dtype=np.int64)
                if kv_cache is kv and len(found) > 1:
                    found[1] = -1
                return found

            return pages

        with monkeypatch.context() as patched:
            for name in ("pages", "locked_pages"):
                patched.setattr(_manager, name, without_block_one(getattr(_manager, name)))
            lease = lender.lend_write(target, 0, END, **getattr(kit, keywords))
            assert lease.failure is None, "the cache grew: nothing fails at the call after that"
            assert lender.poll_resume_bounds(target) is None
            if poll:
                assert lease.poll() is None and lease.failure is not None
            else:
                lease.release()
                assert lease.failure is None
        assert lender.poll_resume_bounds(target) == (kv.num_committed_tokens, kv.history_length)
        lease.release()


GROWN_THEN_FAILED = "the write failed at the call after its cache grew"


def test_a_write_whose_grant_raises_within_the_call_fails_at_its_first_poll(
    kit, real_manager, monkeypatch, attach=attach
):
    """``lend_write`` grows the cache before taking slots, so when its grant raises, the lease it
    returns has not failed and fails at its first poll, abandoning the fetch like one with a block
    left without a page. A read changes no cache, so one whose grant raises fails at the call."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    view = _lender._ManagerLender._build_view
    planted = []

    def raising_once(self, rows, keys, *bounds):
        if not planted:
            planted.append(True)
            raise MemoryError("planted")
        return view(self, rows, keys, *bounds)

    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        kv = kit.kv(mgr, target)
        with monkeypatch.context() as patched:
            patched.setattr(_lender._ManagerLender, "_build_view", raising_once)
            read = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            assert planted and "planted" in str(read.failure), "the read's grant did not raise"
            planted.clear()
            lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
            assert planted, "the write's grant ran after the call: the check proves nothing"
        assert kv.capacity >= END, "the write did not grow the cache"
        assert lease.failure is None, f"{GROWN_THEN_FAILED}: {lease.failure}"
        assert lender.poll_resume_bounds(target) is None
        assert lease.poll() is None and "planted" in str(lease.failure), lease.failure
        assert lender.poll_resume_bounds(target) == (kv.num_committed_tokens, kv.history_length)
        lease.release()
        read.release()


SHARED = list(range(7000, 7040))  # one whole block and eight tokens
SIBLING_PROMPT = SHARED + list(range(7100, 7160))  # block 1 holds other tokens past the eight
MATCHED_PROMPT = SHARED + list(range(7200, 7353))  # six whole blocks and one token
# name: the fetch's end, at which the window leaves block 1 behind or keeps it
MATCHED_FETCHES = {"window_leaves_it": 6 * TPB, "window_keeps_it": 3 * TPB}


@pytest.mark.parametrize("case", list(MATCHED_FETCHES))
def test_a_fetch_from_a_partially_matched_block(kit, real_manager, case, attach=attach):
    """All-reusable: the target's local match ends inside block 1, which its resume copied from a
    sibling, so past the match the page holds the sibling's tokens. A fetch from block 1 whose
    window leaves the block behind fails at the call, changing nothing, while one whose window keeps
    it fetches the block, every row of which then holds its staged bytes. Either way native reuse
    of the window group's block 1 later serves the target's bytes."""
    with real_manager(windows=[WINDOW, 256]) as mgr:
        window_lg = kit.windows(mgr).index(WINDOW)
        dev = kit.DevicePages(mgr)
        sibling = kit.published(mgr, SOURCE, SIBLING_PROMPT)
        sibling_bytes = kit.digest(kit.page(mgr, sibling, window_lg, 1))
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == len(SHARED), "no partial match: the check proves nothing"
        mgr._stream.synchronize()  # the resume's copy of the matched page
        assert kit.digest(kit.page(mgr, target, window_lg, 1)) == sibling_bytes, (
            "block 1 holds no copy of the sibling's page: the check proves nothing"
        )
        start, end = TPB, MATCHED_FETCHES[case]
        stale_beg, stale_end = kit.stale_blocks(mgr, window_lg, end)
        leaves = stale_beg <= 1 < stale_end
        assert leaves == (case == "window_leaves_it")
        lender = attach(mgr, guaranteed_fetch_tokens=end - start)
        before = (kv.capacity, kv.history_length, kv.num_committed_tokens)
        lease = lender.lend_write(target, start, end, **kit.HOST_NAMED)
        if lease.failure is None:
            # Granted: the fetch lands whole, and the target resumes where resume_bounds allows.
            view = kit.ready_view(lease, mgr)
            kit.stage(lender, view, 0x5A)
            return_and_settle(lender, mgr, target, lease, view.row_masks(True))
            for run in view.groups:  # every row the resume reads
                ordinals = run.page_ordinals.tolist()
                assert holds(
                    kit, mgr, target, run.layer_group_index, ordinals, staged_bytes(kit, 0x5A)
                ), f"layer group {run.layer_group_index}: a fetched row lacks its staged bytes"
            resume_bounds = lender.poll_resume_bounds(target)
            assert resume_bounds.max_position >= resume_bounds.min_position, tuple(resume_bounds)
            resumed(kit, mgr, target, resume_bounds.min_position)
        else:
            assert leaves, f"the fetch failed although the window keeps block 1: {lease.failure}"
            assert "never wrote" in lease.failure, lease.failure
            lease.release()
            assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == before, (
                "the fetch that failed at the call changed the cache"
            )
        # The target computes the rest of its prompt, and the context update commits it.
        assert next_chunk(kit, mgr, target, tokens=len(MATCHED_PROMPT)) is None
        assert kv.num_committed_tokens == len(MATCHED_PROMPT), "the prompt was not committed"
        # A later request sharing the target's first three blocks takes block 1 from the tree.
        later = kit.admitted(mgr, OTHER, MATCHED_PROMPT[: 3 * TPB + 1])
        slot = kit.pages(kit.kv(mgr, later), window_lg)[1]
        if slot < 0:
            # Native reuse holds no window page for the block, so it serves no bytes there.
            assert lease.failure is None and leaves, "no window page for block 1: nothing checked"
            return
        got = kit.digest(dev.read(window_lg, slot))
        assert got != sibling_bytes, (
            "native reuse serves block 1 with the sibling's bytes past the target's match"
        )
        expected = staged_bytes(kit, 0x5A) if lease.failure is None else chunk_bytes
        assert got == kit.digest(expected(dev, window_lg, 1)), "block 1 holds other bytes"


def test_a_fetch_past_pages_grown_before_it_fails_at_the_call(kit, real_manager, attach=attach):
    """All-reusable: the target's first chunk was sized before the fetch, so the blocks past its
    history have pages nothing wrote. A fetch whose window leaves them behind fails at the call,
    changing nothing, and native reuse of those blocks later serves what the request computed."""
    with real_manager(windows=[WINDOW, 256]) as mgr:
        window_lg = kit.windows(mgr).index(WINDOW)
        dev = kit.DevicePages(mgr)
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        kv = kit.kv(mgr, target)
        assert (kv.num_committed_tokens, kv.history_length) == (0, 0), "a local match"
        # The scheduler sizes the first chunk before the fetch.
        target.context_chunk_size = 4 * TPB
        assert mgr.resize_context(target, 4 * TPB)
        kit.fill_sentinel(mgr, target)  # what a page nothing wrote holds
        unwritten = kit.digest(bytes([kit.SENTINEL]) * dev.page_bytes(window_lg))
        end = 6 * TPB
        stale_beg, stale_end = kit.stale_blocks(mgr, window_lg, end)
        assert stale_beg <= 1 < stale_end and kit.pages(kv, window_lg)[1] >= 0, (
            "block 1 has no page or the window keeps it: the check proves nothing"
        )
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        before = (kv.capacity, kv.history_length, kv.num_committed_tokens)
        lease = lender.lend_write(target, 0, end, **kit.HOST_NAMED)
        if lease.failure is None:
            # Granted: the fetch lands whole, and the target resumes where resume_bounds allows.
            view = kit.ready_view(lease, mgr)
            kit.stage(lender, view, 0x5A)
            return_and_settle(lender, mgr, target, lease, view.row_masks(True))
            for run in view.groups:  # every row the resume reads
                ordinals = run.page_ordinals.tolist()
                assert holds(
                    kit, mgr, target, run.layer_group_index, ordinals, staged_bytes(kit, 0x5A)
                ), f"layer group {run.layer_group_index}: a fetched row lacks its staged bytes"
            resume_bounds = lender.poll_resume_bounds(target)
            assert resume_bounds.max_position >= resume_bounds.min_position, tuple(resume_bounds)
            resumed(kit, mgr, target, resume_bounds.min_position)
        else:
            assert "never wrote" in lease.failure, lease.failure
            lease.release()
            assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == before, (
                "the fetch that failed at the call changed the cache"
            )
        # The target computes the rest of its prompt, and the context update commits it.
        assert next_chunk(kit, mgr, target, tokens=len(MATCHED_PROMPT)) is None
        assert kv.num_committed_tokens == len(MATCHED_PROMPT), "the prompt was not committed"
        # A later request sharing the target's first three blocks takes block 1 from the tree.
        later = kit.admitted(mgr, OTHER, MATCHED_PROMPT[: 3 * TPB + 1])
        slot = kit.pages(kit.kv(mgr, later), window_lg)[1]
        if slot < 0:
            # Native reuse holds no window page for the block, so it serves no bytes there.
            assert lease.failure is None, "no window page for block 1: nothing checked"
            return
        got = kit.digest(dev.read(window_lg, slot))
        assert got != unwritten, (
            "native reuse serves block 1 with bytes nothing computed or fetched"
        )
        expected = staged_bytes(kit, 0x5A) if lease.failure is None else chunk_bytes
        assert got == kit.digest(expected(dev, window_lg, 1)), "block 1 holds other bytes"


def test_a_fetch_past_pages_grown_before_it_goes_through_where_the_commit_keeps_none(
    kit, real_manager, attach=attach
):
    """Per request: the manager commits only the snapshot a request's window needs, so pages a
    window leaves behind are dropped, not kept for the commit. A fetch past pages the target's first
    chunk was sized with goes through, and the target resumes with what it fetched."""
    with real_manager(windows=[WINDOW, 256], **PER_REQUEST) as mgr:
        assert mgr.impl.commit_min_snapshot, "passed pages stay: the check proves nothing"
        window_lg = kit.windows(mgr).index(WINDOW)
        target = kit.admitted(mgr, TARGET, MATCHED_PROMPT)
        kv = kit.kv(mgr, target)
        # The scheduler sizes the first chunk before the fetch.
        target.context_chunk_size = 4 * TPB
        assert mgr.resize_context(target, 4 * TPB)
        kit.fill_sentinel(mgr, target)
        end = 6 * TPB
        stale_beg, stale_end = kit.stale_blocks(mgr, window_lg, end)
        assert stale_beg <= 1 < stale_end and kit.pages(kv, window_lg)[1] >= 0, (
            "block 1 has no page or the window keeps it: the check proves nothing"
        )
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        lease = lender.lend_write(target, 0, end, **kit.HOST_NAMED)
        assert lease.failure is None, (
            f"the fetch failed although the commit keeps no page a window passed: {lease.failure}"
        )
        view = kit.ready_view(lease, mgr)
        kit.stage(lender, view, 0x5A)
        return_and_settle(lender, mgr, target, lease, view.row_masks(True))
        passed = kit.pages(kv, window_lg)[stale_beg:stale_end]
        assert all(slot < 0 for slot in passed), f"the window's passed blocks kept pages: {passed}"
        resume_bounds = lender.poll_resume_bounds(target)
        assert resume_bounds.max_position == end, tuple(resume_bounds)
        resumed(kit, mgr, target, resume_bounds.max_position)
        kept = [o for o in range(end // TPB) if not stale_beg <= o < stale_end]
        for lg in range(kit.num_layer_groups(mgr)):
            ordinals = kept if lg == window_lg else range(end // TPB)
            assert holds(kit, mgr, target, lg, ordinals, staged_bytes(kit, 0x5A)), (
                f"layer group {lg}: the target resumes without the bytes it fetched"
            )
        assert next_chunk(kit, mgr, target, tokens=len(MATCHED_PROMPT)) is None


@pytest.mark.parametrize("case", ["missed", "delivered", "computed"])
def test_a_later_lease_past_window_rows_an_earlier_one_missed_fails_at_the_call(
    kit, real_manager, case, attach=attach
):
    """All-reusable, a window of ``WINDOW`` tokens: the request computed block 0, and the first
    lease of a fetch, ``[32, 64)``, delivers block 1's full-attention row and, in "delivered" alone,
    its window row. The later lease ``[64, 128)`` leaves blocks 0 and 1 behind its window, and the
    commit stores block 1's pages whole. Where its window row was missed, nothing wrote that page
    and the later lease fails at the call, changing nothing; where it was delivered, or the request
    computed block 1 from where resume_bounds let it resume, it is lent. Native reuse of blocks 0 and 1
    later serves what the request computed or fetched."""
    with real_manager(windows=[WINDOW, 256]) as mgr:
        sliding = kit.windows(mgr).index(WINDOW)
        dev = kit.DevicePages(mgr)
        target = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
        kv = kit.kv(mgr, target)
        assert kv.num_committed_tokens == 0, "a local match: the check proves nothing"
        # Block 0 first, so the check starts past it and reads block 1's delivered row there.
        assert next_chunk(kit, mgr, target, tokens=TPB) is None
        assert kv.num_committed_tokens == TPB, "block 0 was not committed"
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        first = lender.lend_write(target, TPB, 2 * TPB, **kit.HOST_NAMED)
        kit.fill_sentinel(mgr, target, 1)  # what a page nothing wrote holds
        view = kit.ready_view(first, mgr)
        kit.stage(lender, view, 0x11)
        delivered = case == "delivered"
        arrived = tuple(
            np.full(len(run), delivered or run.layer_group_index != sliding) for run in view.groups
        )
        resume_bounds = return_and_settle(lender, mgr, target, first, arrived)
        assert resume_bounds == ((2 if delivered else 1) * TPB, TPB), tuple(resume_bounds)
        if case == "computed":  # from where resume_bounds lets it resume, through the lease's end
            resumed(kit, mgr, target, resume_bounds.max_position)
            assert next_chunk(kit, mgr, target, tokens=TPB) is None
            assert kv.num_committed_tokens == 2 * TPB, "the chunk was not committed"
        assert kit.stale_blocks(mgr, sliding, 4 * TPB) == (0, 2), "nothing left behind: no check"
        assert min(kit.pages(kv, sliding)[:2]) >= 0, "blocks 0 and 1 have no page: no check"
        before = state_of(kit, mgr, lender, target)
        later = lender.lend_write(target, 2 * TPB, 4 * TPB, **kit.HOST_NAMED)
        if case == "missed":
            assert later.failure is not None and "never wrote" in later.failure, (
                f"a later lease was lent past window rows an earlier one missed: {later.failure}"
            )
            later.release()
            assert state_of(kit, mgr, lender, target) == before, (
                "the failed lease changed the cache"
            )
        else:
            assert later.failure is None, f"the later lease failed: {later.failure}"
            view = kit.ready_view(later, mgr)
            kit.stage(lender, view, 0x22)
            resume_bounds = return_and_settle(lender, mgr, target, later, view.row_masks(True))
            assert resume_bounds == (4 * TPB, 4 * TPB), tuple(resume_bounds)
        # The target computes the rest of its prompt from where resume_bounds lets it resume, and the
        # context update commits it.
        resumed(kit, mgr, target, resume_bounds.max_position)
        assert next_chunk(kit, mgr, target, tokens=len(SPLIT_PROMPT)) is None
        assert kv.num_committed_tokens == len(SPLIT_PROMPT), "the prompt was not committed"
        # A later request sharing the target's first two blocks takes them from the tree.
        other = kit.admitted(mgr, OTHER, SPLIT_PROMPT[: 2 * TPB] + OTHER_PROMPT[:33])
        unwritten = kit.digest(bytes([kit.SENTINEL]) * dev.page_bytes(sliding))
        for ordinal in (0, 1):
            slot = kit.pages(kit.kv(mgr, other), sliding)[ordinal]
            assert slot >= 0, f"native reuse took no window page for block {ordinal}: no check"
            got = kit.digest(dev.read(sliding, slot))
            assert got != unwritten, (
                f"native reuse serves block {ordinal} with bytes nothing computed or fetched"
            )
            expected = staged_bytes(kit, 0x11) if delivered and ordinal == 1 else chunk_bytes
            assert got == kit.digest(expected(dev, sliding, ordinal)), f"block {ordinal} differs"


# -- argument errors --------------------------------------------------------------------------


def test_a_read_refuses_a_bad_range_at_the_call(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        kit.published(mgr, 7, OTHER_PROMPT[: 2 * TPB] + list(range(9000, 9033)))
        sized = kit.admitted(mgr, TARGET, OTHER_PROMPT)  # its first two blocks match locally
        # Sized for the rest of its prompt, as before its next chunk: block 2 has a page, no KV.
        assert mgr.resize_context(sized, sized.context_remaining_length)
        assert kit.kv(mgr, sized).num_committed_tokens == 2 * TPB
        lender = attach(mgr, guaranteed_fetch_tokens=4 * TPB)
        for request, start, end, why in (
            (source, -TPB, TPB, "bad token range"),
            (source, TPB, 0, "bad token range"),
            (source, 0, 50, "whole blocks"),
            (source, 16, 48, "whole blocks"),
        ):
            with pytest.raises(ValueError, match=why):
                lender.lend_read(request, start, end, **kit.HOST_NAMED)
        # Past what this rank committed: another pipeline stage may hold it, so the lease fails.
        for request, start, end, why in (
            (source, 0, 4 * TPB, "past the 97 committed tokens"),
            (source, 0, 1 << 50, "past the 97 committed tokens"),  # of any length
            (sized, 0, END, "past the 64 committed tokens"),  # inside the prompt's whole blocks
        ):
            lease = lender.lend_read(request, start, end, **kit.HOST_NAMED)
            assert lease.failure is not None and why in lease.failure, (why, lease.failure)
            assert lease.poll() is None
            lease.release()
        lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(lease, mgr)
        lease.release()


def test_a_write_refuses_a_bad_range_before_changing_anything(kit, real_manager):
    with real_manager() as mgr:
        kit.published(mgr, 7, PROMPT[: 2 * TPB] + list(range(9000, 9033)))
        matched = kit.admitted(mgr, TARGET, PROMPT)  # its first two blocks match locally
        fresh = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        assert kit.kv(mgr, matched).num_committed_tokens == 2 * TPB
        lender = attach(mgr, guaranteed_fetch_tokens=4 * TPB, capacity_in_fetches=2)
        before = [state_of(kit, mgr, lender, r) for r in (matched, fresh)]
        bad = [
            (fresh, -TPB, TPB, "bad token range"),
            (fresh, TPB, 0, "bad token range"),
            (fresh, 0, 50, "covers whole blocks"),
            (fresh, 16, 48, "covers whole blocks"),
            (fresh, 0, 4 * TPB, "past the whole blocks before the request's last prompt token"),
        ]
        for request, start, end, why in bad:
            with pytest.raises(ValueError, match=why):
                lender.lend_write(request, start, end, **kit.HOST_NAMED)
        assert [state_of(kit, mgr, lender, r) for r in (matched, fresh)] == before
        # A fetch from inside what this rank committed lends from the committed blocks' end on.
        leases = [
            lender.lend_write(matched, TPB, END, **kit.HOST_NAMED),
            lender.lend_write(fresh, 0, END, **kit.HOST_NAMED),
        ]
        unsettled = state_of(kit, mgr, lender, fresh)
        again = lender.lend_write(fresh, 0, END, **kit.HOST_NAMED)  # one unsettled fetch per cache
        assert again.failure is not None and "unsettled" in again.failure, again.failure
        assert again.poll() is None and state_of(kit, mgr, lender, fresh) == unsettled
        again.release()
        for lease in leases:
            view = lease.poll()
            if lease is leases[0]:
                lent = sorted({o for g in view.groups for o in g.page_ordinals.tolist()})
                assert lent and min(lent) == 2, f"a fetch lent committed blocks: {lent}"
            kit.return_write(lender, lease, view.row_masks())


ALIGNED_PROMPT = list(range(11000, 11128))  # exactly four whole blocks


@pytest.mark.parametrize("windowed", [False, True], ids=["full", "windowed"])
def test_a_fetch_ends_at_the_whole_blocks_before_the_last_prompt_token(kit, real_manager, windowed):
    # The request computes its last prompt token itself, for its logits: on a prompt of whole
    # blocks, fetching all of them would leave it a resume at the prompt's end. A range of any
    # length past them is refused alike.
    with real_manager(windows=[WINDOW, 256] if windowed else None) as mgr:
        target = kit.admitted(mgr, TARGET, ALIGNED_PROMPT)
        end = len(ALIGNED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        before = state_of(kit, mgr, lender, target)
        for past in (end, 1 << 50):  # 1 << 50: whole blocks whose ordinals would take 256 TiB
            with pytest.raises(ValueError, match="past the whole blocks before"):
                lender.lend_write(target, 0, past, **kit.HOST_NAMED)
        assert state_of(kit, mgr, lender, target) == before
        lease = lender.lend_write(target, 0, end - TPB, **kit.HOST_NAMED)  # the last block left out
        view = lease.poll()
        assert view is not None
        kit.return_write(lender, lease, view.row_masks())


def test_a_replaced_cache_takes_a_new_fetch(kit, real_manager):
    with real_manager(
        max_tokens=kit.POOL_TOKENS, host_cache_size=0, max_batch_size=64, max_util_for_resume=1.0
    ) as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        first = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        dev = kit.DevicePages(mgr)
        others = kit.Requests(mgr, dev)
        second = None
        try:
            first_view = first.poll()
            assert first_view is not None
            old_pages = set(kit.pages(kit.kv(mgr, target), 0))
            assert len(old_pages) == BLOCKS and min(old_pages) >= 0
            kit.stage(lender, first_view, 0x5A)
            remaining = kit.pool_pages(mgr) - len(old_pages)
            assert remaining < mgr.max_batch_size, "fixture has too few request slots"
            assert others.allocate(remaining, fill=False, chunk=1)
            occupied = others.pages()
            assert old_pages.isdisjoint(occupied)
            assert len(old_pages | occupied) == kit.pool_pages(mgr)
            mgr.free_resources(target)
            torch.cuda.synchronize()
            target = kit.admitted(mgr, TARGET, PROMPT)
            second = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
            view = second.poll()
            assert view is not None and lender.poll_resume_bounds(target) is None
            assert old_pages == set(kit.pages(kit.kv(mgr, target), 0)), "old pages were not reused"
            kit.fill_sentinel(mgr, target)
            assert kit.return_write(lender, first, first_view.row_masks(True)) is None
            sentinel = bytes([kit.SENTINEL]) * dev.page_bytes(0)
            assert all(dev.read(0, s) == sentinel for s in old_pages), "old arrivals landed"
            assert lender.poll_resume_bounds(target) is None
            kit.return_write(lender, second, view.row_masks())
        finally:
            first.release()
            if second is not None:
                second.release()
            others.free()


def test_a_windowed_target_with_scratch_reuse_fails_at_the_call(kit, real_manager, attach=attach):
    with real_manager(windows=[WINDOW, 256], swa_scratch_reuse=True) as mgr:
        target = kit.make_request(TARGET, WINDOWED_PROMPT)
        assert mgr.prepare_context(target)
        kv = kit.kv(mgr, target)
        assert kv.enable_swa_scratch_reuse, "the target must start with scratch reuse on"
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        before = state_of(kit, mgr, lender, target)
        try:
            lease = lender.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)
        except RuntimeError as error:  # the manager refusing the grow a lender let through
            raise AssertionError(f"not failed at the call: {error}") from error
        assert lease.failure is not None and "scratch" in lease.failure, "not failed at the call"
        assert lease.poll() is None
        assert state_of(kit, mgr, lender, target) == before
        lease.release()
        kv.enable_swa_scratch_reuse = False
        lease = lender.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)  # off: it goes through
        view = lease.poll()
        assert view is not None
        kit.return_write(lender, lease, view.row_masks())


def test_a_windowless_target_with_scratch_reuse_fails_at_the_call(kit, real_manager, attach=attach):
    """Without a window too, a cache with SWA scratch reuse on keeps its history within the scratch
    rewind of its old capacity at every capacity change, which a fetch's grow breaks: the fetch
    fails at the call and changes nothing, and with scratch reuse off it goes through."""
    with real_manager(swa_scratch_reuse=True) as mgr:
        assert not any(kit.windows(mgr)), "a window: the windowed check covers that"
        target = kit.make_request(TARGET, SPLIT_PROMPT)
        assert mgr.prepare_context(target)
        kv = kit.kv(mgr, target)
        assert kv.enable_swa_scratch_reuse, "the target must start with scratch reuse on"
        lender = attach(mgr, guaranteed_fetch_tokens=PAIRED_END)
        before = state_of(kit, mgr, lender, target)
        try:
            lease = lender.lend_write(target, 0, PAIRED_END, **kit.HOST_NAMED)
        except RuntimeError as error:  # the manager refusing the grow a lender let through
            raise AssertionError(f"not failed at the call: {error}") from error
        try:
            failure = lease.failure
            assert failure is not None, "not failed at the call"
            assert "scratch rewind of the old capacity" in failure, failure
            assert lease.poll() is None
            assert state_of(kit, mgr, lender, target) == before, "the fetch changed the cache"
        finally:
            lease.release()
        kv.enable_swa_scratch_reuse = False
        lease = lender.lend_write(target, 0, PAIRED_END, **kit.HOST_NAMED)  # off: it goes through
        view = lease.poll()
        assert view is not None
        kit.return_write(lender, lease, view.row_masks())


def test_a_fetch_into_a_request_returning_context_outputs_fails_at_the_call(
    kit, real_manager, attach=attach
):
    """A request returning context logits, as prompt logprobs make it do, fails at the call whether
    it holds some or none, as does one asking for additional model outputs, caches unchanged: the
    executor gives these only for positions a request computes. One returning none is lent."""
    with real_manager() as mgr:
        held = kit.make_request(TARGET, PROMPT, return_context_logits=True)
        fresh = kit.make_request(OTHER, OTHER_PROMPT, return_context_logits=True)
        outputs = kit.make_request(OTHER + 1, CONTROL_PROMPT, additional_outputs=["context_output"])
        plain = kit.make_request(OTHER + 2, SPLIT_PROMPT)
        for request in (held, fresh, outputs, plain):
            assert mgr.prepare_context(request)
        held.py_result.append_context_logits(torch.zeros(TPB, 1, 8, device="cuda"))
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        for request, returns in ((held, True), (fresh, True), (outputs, True), (plain, False)):
            before = state_of(kit, mgr, lender, request)
            lease = lender.lend_write(request, 0, END, **kit.HOST_NAMED)
            if returns:
                assert lease.failure is not None, "not failed at the call"
                assert "context" in lease.failure, lease.failure
                assert lease.poll() is None
                assert state_of(kit, mgr, lender, request) == before, "the lease changed the cache"
                lease.release()
            else:
                assert lease.failure is None, "refused a request returning no context outputs"
                kit.return_write(lender, lease, kit.ready_view(lease, mgr).row_masks())


@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_a_fetch_ending_below_a_history_the_update_moved_fails_at_the_call(
    kit, real_manager, policy, attach=attach
):
    """Full attention under a policy whose context update moves the history: the request computed
    ``[0, 128)`` without a commit, so its history stands past a fetch of ``[64, 96)``. The floor
    would follow it past all the fetch brings, an empty interval dropping the computed tokens; the
    fetch fails at the call instead, the cache unchanged, and the request resumes at its history."""
    conversation = "history-past" if policy == "per_conversation" else None
    with real_manager(block_reuse_config={"policy": policy}) as mgr:
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT, upto=4 * TPB, conversation=conversation)
        lender = attach(mgr, guaranteed_fetch_tokens=TPB)
        before = state_of(kit, mgr, lender, target)
        lease = lender.lend_write(target, 2 * TPB, 3 * TPB, **kit.HOST_NAMED)
        try:
            assert lease.failure is not None, f"not failed at the call, from {before}"
            assert "stands past" in lease.failure, lease.failure
            assert lease.poll() is None
            assert state_of(kit, mgr, lender, target) == before, "the fetch changed the cache"
        finally:
            lease.release()
        assert lender.poll_resume_bounds(target) == (4 * TPB, 4 * TPB)


def test_a_draft_pool_fetch_ending_below_its_history_fails_at_the_call(
    kit, real_manager, attach=attach
):
    """Windowless under the all-reusable policy, a joint-reuse draft pool's floor follows its
    history, so a fetch of ``[64, 96)`` into a cache whose history stands at 128 fails at the call,
    the cache unchanged, while a target pool in the same state lends. Each pool's context capacity
    is allocated as admission does and its history moved to 128 without a commit: such a pool's
    context update commits every chunk it runs, so no native path leaves the history there."""
    spec, at = dflash(), 4 * TPB
    with (
        real_manager(spec_config=spec, joint_kv_cache_reuse=True) as target,
        real_manager(spec_config=spec, is_draft=True, joint_kv_cache_reuse=True) as draft,
    ):
        failures = []
        for mgr, scope in ((target, SCOPE), (draft, SCOPE + b"/draft-pool")):
            request = kit.admitted(mgr, TARGET, SPLIT_PROMPT)
            request.context_chunk_size = at
            if mgr.is_draft:
                assert mgr.try_allocate_draft_context(request, at)
            else:
                assert mgr.resize_context(request, at)
            kv = kit.kv(mgr, request)
            kv.resize(None, at)
            assert (kv.history_length, kv.num_committed_tokens) == (at, 0)
            lender = attach(mgr, guaranteed_fetch_tokens=TPB, scope=scope)
            before = state_of(kit, mgr, lender, request)
            lease = lender.lend_write(request, 2 * TPB, 3 * TPB, **kit.HOST_NAMED)
            try:
                failures.append(lease.failure)
                if lease.failure is not None:
                    assert state_of(kit, mgr, lender, request) == before, "the fetch changed it"
            finally:
                lease.release()
        assert failures[0] is None, f"the target pool's fetch failed at the call: {failures[0]}"
        assert failures[1] is not None and "stands past" in failures[1], (
            f"the draft pool's fetch below its history was not failed at the call: {failures[1]}"
        )


def test_a_range_longer_than_staging_holds_fails_at_the_call(kit, real_manager, attach=attach):
    """A range longer than this rank's staging holds fails the lease at the call."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)  # two slots
        before = state_of(kit, mgr, lender, target)
        for lend, request in ((lender.lend_read, source), (lender.lend_write, target)):
            lease = lend(request, 0, END, **kit.HOST_NAMED)
            assert "more staging slots" in str(lease.failure), lease.failure
            assert lease.poll() is None
            lease.release()
        assert lender._open_lease_count() == 0, "a refused lease stayed open after its release"
        assert state_of(kit, mgr, lender, target) == before
        # A range of guaranteed_fetch_tokens tokens fits exactly.
        read = lender.lend_read(source, 0, 2 * TPB, **kit.HOST_NAMED)
        kit.ready_view(read, mgr)
        read.release()
        lender.progress()  # the read's slots come back
        write = lender.lend_write(target, 0, 2 * TPB, **kit.HOST_NAMED)
        view = write.poll()
        assert view is not None
        kit.return_write(lender, write, view.row_masks())


# What this rank holds for a request that fails a call whose range is right: no KV cache, a reset
# of the manager's reuse state, and multimodal data without digests.
REQUEST_STATES = ["no_cache", "reuse_reset", "unnamed"]


@pytest.mark.parametrize("state", REQUEST_STATES)
def test_a_wrong_call_raises_whatever_the_request_s_state(kit, real_manager, state, attach=attach):
    """A wrong range raises whatever the request's state, which fails a right or too long range."""
    with real_manager() as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)  # two slots
        if state == "no_cache":
            target = kit.make_request(TARGET, OTHER_PROMPT)
        elif state == "reuse_reset":
            mgr.reset_reuse_state()  # what reset_prefix_cache does when a weight update ends
            target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        else:
            target = carrying(kit, TARGET, OTHER_PROMPT, dict(py_multimodal_data=IMAGE))
            assert mgr.prepare_context(target)
        before = state_of(kit, mgr, lender, target)
        for lend, end, why in (
            (lender.lend_read, 50, "whole blocks"),
            (lender.lend_write, 50, "whole blocks"),
            (lender.lend_write, 4 * TPB, "past the whole blocks before the request's last prompt"),
        ):
            assert why in refusal(lambda: lend(target, 0, end, **kit.HOST_NAMED))
        assert state_of(kit, mgr, lender, target) == before, "a wrong call changed the cache"
        # A right range, and one past the two slots of staging: the request's state decides.
        why = {"no_cache": "no KV cache", "reuse_reset": RESET, "unnamed": "without digests"}[state]
        for end in (2 * TPB, END):
            lease = lender.lend_write(target, 0, end, **kit.HOST_NAMED)
            try:
                assert why in str(lease.failure) and lease.poll() is None, (end, lease.failure)
            finally:
                lease.release()
        assert state_of(kit, mgr, lender, target) == before, "a failed lease changed the cache"


# -- resume_bounds --------------------------------------------------------------------------------


def test_resume_bounds_without_a_fetch_is_the_cache_s_own_interval(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        kv = kit.kv(mgr, source)
        resume_bounds = lender.poll_resume_bounds(source)
        assert resume_bounds == (max(kv.num_committed_tokens, kv.history_length), kv.history_length)
        assert all(type(v) is int for v in resume_bounds)
        with pytest.raises(ValueError):
            lender.poll_resume_bounds(kit.make_request(9, PROMPT))


def test_resume_bounds_waits_on_nothing_and_settles_at_a_round_of_progress(kit, real_manager):
    """A returned write's fetch settles at the first round of progress after its copy ran."""
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        with kit.held_stream(mgr._stream) as gate:
            assert kit.return_write(lender, lease, view.row_masks(True)) is None
            for _ in range(2):
                lender.progress()
                assert lender.poll_resume_bounds(target) is None
            gate.open()
        lender.progress()
        assert lender.poll_resume_bounds(target) == (END, 0)


# -- a reset of the manager's reuse state ----------------------------------------------------

RESET = "reset its reuse state"


def test_a_reuse_reset_stops_lending_by_name(kit, real_manager, attach=attach):
    # After an in-place weight update the manager computes other bytes for the same tokens, whose
    # names stay what they were: nothing may be lent by name any more.
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        original = [kit.page(mgr, source, 0, o) for o in range(BLOCKS)]
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        publish = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # granted: its copy is queued
        fetch = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        fetch_view = fetch.poll()  # seen ready, not returned
        for request in (source, target):  # the manager resets only with every cache closed
            mgr.free_resources(request)
        mgr.reset_reuse_state()  # what reset_prefix_cache does when a weight update ends
        (run,) = kit.ready_view(publish, mgr).groups  # granted before the reset: it finishes
        length = lender.regions[run.region_index].slot_bytes
        staged = [kit.host_bytes(a, length) for a in run.addresses.tolist()]
        assert kit.digest(staged) == kit.digest(original), "the granted publish lost its bytes"
        # Its request is gone: nothing is copied, and the request has no bounds.
        assert kit.return_write(lender, fetch, fetch_view.row_masks(True)) is None
        # The same tokens, computed after the reset, would go out under the names of the old bytes.
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        before = [state_of(kit, mgr, lender, r) for r in (source, target)]
        later = [
            lender.lend_read(source, 0, END, **kit.HOST_NAMED),
            lender.lend_write(target, 0, END, **kit.HOST_NAMED),
        ]
        for lease in later:
            assert lease.poll() is None and RESET in str(lease.failure), (
                "lent by name after the reset"
            )
        assert [state_of(kit, mgr, lender, r) for r in (source, target)] == before
        for lease in (publish, fetch, *later):
            lease.release()


# -- shutdown ---------------------------------------------------------------------------------


def test_the_manager_s_shutdown_waits_for_the_lender_s_copies(kit, real_manager, monkeypatch):
    """At the shutdown the lender waits for its copies before it frees the staging memory they
    write. The manager's own teardown waits on the stream as well, so the order shows only at the
    free."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    free = _lender._HostMemory.free
    frees = []  # per free of staging memory: whether the stream had run the gated copy by then

    def recording_free(memory):
        frees.append(mgr._stream.query())
        gate.open()
        mgr._stream.synchronize()  # nothing is freed under a copy, also for a lender that is wrong
        free(memory)

    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        monkeypatch.setattr(_lender._HostMemory, "free", recording_free)
        with kit.held_stream(mgr._stream, strict=False) as gate:
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            event = lease._copy._event
            assert event is not None
            synchronize = event.synchronize

            def finish_copy():
                gate.open()
                synchronize()

            monkeypatch.setattr(event, "synchronize", finish_copy)
            lease.release()  # its copy waits behind the gate until shutdown waits on its event
            mgr.shutdown()
        assert frees == [True], "the staging memory was freed before the copy into it ran"
        assert not kit.staging_kept(regions), "no lease was open: the memory is freed"


def test_shutdown_fails_waiting_leases_and_later_calls_only_end_records(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        held = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(held, mgr)
        waiting = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        assert waiting.poll() is None
        mgr.shutdown()
        assert waiting.failure is not None and waiting.poll() is None
        for lend, request in ((lender.lend_read, source), (lender.lend_write, target)):
            with pytest.raises(RuntimeError, match="lends no more"):
                lend(request, 0, END, **kit.HOST_NAMED)
        with pytest.raises(ValueError):
            lender.poll_resume_bounds(source)
        for lease in (held, held, waiting):
            lease.release()
        mgr.shutdown()  # a second shutdown frees nothing and returns


@pytest.mark.parametrize("failed", [False, True], ids=["ready_lease", "failed_lease"])
def test_an_unreleased_lease_keeps_the_staging_memory_until_exit(
    kit, real_manager, monkeypatch, failed, attach=attach
):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        assert kit.staging_memory(regions)
        if failed:
            lease = lender.lend_read(kit.make_request(9, PROMPT), 0, END, **kit.HOST_NAMED)
            assert lease.failure is not None
            address, content = None, None
        else:
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            address = int(kit.ready_view(lease, mgr).groups[0].addresses[0])
            content = kit.host_bytes(address, regions[0].slot_bytes)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(regions), "staging freed at shutdown with a lease open"
        assert warnings, "keeping the memory until exit is logged"
        if address is not None:
            got = kit.digest(kit.host_bytes(address, regions[0].slot_bytes))
            assert got == kit.digest(content), "the kept staging memory changed"
        lease.release()
        assert kit.staging_kept(regions), "a release after the shutdown freed it"


def test_a_shutdown_hook_that_fails_keeps_the_memory_and_raises_nothing(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def planted(self):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        regions = kit.staging_regions(attach(mgr, guaranteed_fetch_tokens=END))
        monkeypatch.setattr(_lender._ManagerLender, "_must_retain_memory", planted)
        mgr.shutdown()
        assert kit.staging_kept(regions)


# -- regions holds ------------------------------------------------------------------------------


def released(hold) -> None:
    """``hold.release()``; the check fails if it raises, since release is legal in every state."""
    try:
        hold.release()
    except Exception as error:  # handed to the check as its failure
        raise AssertionError(f"release raised {error!r}") from error


def test_a_hold_dropped_unreleased_still_keeps_the_staging_memory(kit, real_manager, attach=attach):
    with real_manager() as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        assert kit.staging_memory(regions)
        lender.hold_regions()  # dropped unreleased
        gc.collect()
        mgr.shutdown()
        assert kit.staging_kept(regions), "a dropped hold let the memory go"


def test_a_hold_releases_once_in_every_state(kit, real_manager, attach=attach):
    with real_manager() as mgr:
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        assert kit.staging_memory(regions)
        first, second = lender.hold_regions(), lender.hold_regions()
        released(first)
        released(first)  # does nothing: the second hold still holds
        mgr.shutdown()
        assert kit.staging_kept(regions), "a second release let the other hold go"
        late = lender.hold_regions()  # after the shutdown: inert
        assert isinstance(late, RegionHold)
        for hold in (late, late):
            released(hold)
    gone = weakref.ref(lender)
    del mgr, lender
    gc.collect()
    assert gone() is None, "the lender outlived its manager"
    for hold in (second, second):  # its lender is gone
        released(hold)


# What is released before the manager's shutdown, in order; the rest is released after it.
ORDERS = {
    "neither": (),
    "lease": ("lease",),
    "hold": ("hold",),
    "lease_then_hold": ("lease", "hold"),
    "hold_then_lease": ("hold", "lease"),
}


@pytest.mark.parametrize("order", list(ORDERS.values()), ids=list(ORDERS))
def test_a_lease_and_a_hold_keep_the_memory_until_both_go(
    kit, real_manager, monkeypatch, order, attach=attach
):
    """A ready lease and a regions hold released in ``order`` before the shutdown, the rest after it,
    and a failed lease released at once: only with both gone does the shutdown free the memory."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        refs = [weakref.ref(m) for m in kit.staging_memory(regions)]
        assert refs
        lender.lend_read(kit.make_request(9, PROMPT), 0, END, **kit.HOST_NAMED).release()  # failed
        hold = lender.hold_regions()
        assert isinstance(hold, RegionHold)
        lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(lease, mgr)
        ends = {"lease": lease.release, "hold": lambda: released(hold)}
        for what in order:
            ends[what]()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        both = len(order) == 2
        if both:
            assert not kit.staging_kept(regions) and warnings == [], "kept with both released"
        else:
            assert kit.staging_kept(regions), f"freed at shutdown after releasing {order}"
            assert warnings, "keeping the memory until exit is logged"
        for what in [w for w in ("lease", "hold") if w not in order]:
            ends[what]()
        mgr.shutdown()
        assert kit.staging_kept(regions) == (not both), (
            "a release or a shutdown after it changed it"
        )
    gc.collect()
    assert all(ref() is None for ref in refs) == both, "kept memory went, or freed memory stayed"


@pytest.mark.parametrize("order", list(ORDERS.values()), ids=list(ORDERS))
def test_a_failed_copy_and_a_hold_keep_the_memory_in_every_order(
    kit, real_manager, monkeypatch, order, attach=attach
):
    def no_event(self, stream=None):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        (group,) = kit.pool_group_ids(mgr)
        assert kit.staging_memory(regions)
        hold = lender.hold_regions()
        with monkeypatch.context() as patched:
            patched.setattr(torch.cuda.Event, "record", no_event)
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)  # no event: slots lost
        assert lease.failure is not None
        mgr._stream.synchronize()
        ends = {"lease": lease.release, "hold": lambda: released(hold)}
        for what in order:
            ends[what]()
        lender.progress()
        assert lender._free_slots(group) == 0, "slots a copy may still touch were reused"
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(regions), f"freed at shutdown after releasing {order}"
        assert warnings, "keeping the memory until exit is logged"
        for what in [w for w in ("lease", "hold") if w not in order]:
            ends[what]()
        assert kit.staging_kept(regions), "a release after the shutdown freed it"


# -- a copy that fails partway ----------------------------------------------------------------


class FailingDriver:
    """The CUDA driver, except that its ``fail_on``-th memcpy call, of any flavour, fails: it
    returns an error, or raises ``raises("planted")``."""

    def __init__(self, real_driver, fail_on, raises=None):
        self._real = real_driver
        self._fail_on = fail_on
        self._raises = raises
        self.calls = 0

    def __getattr__(self, name):
        found = getattr(self._real, name)
        if not name.startswith("cuMemcpy"):
            return found

        def call(*args, **kwargs):
            self.calls += 1
            if self.calls == self._fail_on:
                if self._raises is not None:
                    raise self._raises("planted")
                return (self._real.CUresult.CUDA_ERROR_INVALID_VALUE,)
            return found(*args, **kwargs)

        return call


def test_a_publish_whose_copy_fails_partway_fails_and_frees_its_slots_after_the_rest(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    # Three layers: two pool groups, so the copy is several memcpys.
    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        groups = kit.pool_group_ids(mgr)
        driver = FailingDriver(_lender.drv, fail_on=2)
        monkeypatch.setattr(_lender, "drv", driver)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, WINDOWED_END, **kit.HOST_NAMED)  # every slot
            assert driver.calls >= 2, "a single memcpy: nothing failed partway"
            assert lease.failure is not None and lease.poll() is None
            lease.release()
            lender.progress()
            assert [lender._free_slots(g) for g in groups] == [0] * len(groups), (
                "slots came back while a copy into them was queued"
            )
            gate.open()
        lender.progress()
        assert [lender._free_slots(g) for g in groups] == [
            p.slot_count for p in kit.staging_regions(lender)
        ]


def test_arrivals_whose_copy_fails_partway_abandon_the_fetch_and_return_the_lease(
    kit, real_manager, monkeypatch
):
    """A return whose copy fails partway abandons the fetch and still returns the lease."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        target = kit.admitted(mgr, TARGET, WINDOWED_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END)
        groups = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)
        view = lease.poll()
        kit.stage(lender, view, 0x5A)
        driver = FailingDriver(_lender.drv, fail_on=2)
        monkeypatch.setattr(_lender, "drv", driver)
        kv = kit.kv(mgr, target)
        with kit.held_stream(mgr._stream) as gate:
            bounds = kit.return_write(lender, lease, view.row_masks(True))  # raises nothing
            assert driver.calls >= 2, "a single memcpy: nothing failed partway"
            assert bounds == (kv.num_committed_tokens, kv.history_length), bounds
            assert lender.poll_resume_bounds(target) == bounds
            with pytest.raises(RuntimeError):
                lease.poll()  # returned
            lender.progress()
            assert [lender._free_slots(g) for g in groups] == [0] * len(groups)
            gate.open()
        lender.progress()
        assert [lender._free_slots(g) for g in groups] == [
            p.slot_count for p in kit.staging_regions(lender)
        ]


# -- errors inside the lender -----------------------------------------------------------------


def test_an_event_query_that_raises_returns_no_slot_twice(
    kit, real_manager, monkeypatch, attach=attach
):
    """Two released publishes' copies have run, and the slot return's query of the later one's
    event raises: that call raises having returned no slot, and the next one returns every slot
    once."""
    with real_manager() as mgr:
        first = kit.published(mgr, SOURCE, PROMPT)
        second = kit.published(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        (group,) = kit.pool_group_ids(mgr)
        with kit.held_stream(mgr._stream):
            # Granted and released first: asked first.
            earlier = lender.lend_read(first, 0, END, **kit.HOST_NAMED)
            later = lender.lend_read(second, 0, END, **kit.HOST_NAMED)
            earlier.release()
            later.release()
            lender.progress()
            assert lender._free_slots(group) == 0
        query = torch.cuda.Event.query
        planted = later._copy._event

        def raises_once(event):
            if event is planted:
                monkeypatch.setattr(torch.cuda.Event, "query", query)
                raise RuntimeError("planted")
            return query(event)

        monkeypatch.setattr(torch.cuda.Event, "query", raises_once)
        with pytest.raises(RuntimeError, match="planted"):
            lender.progress()
        try:
            lender.progress()
        except ValueError as error:
            raise AssertionError(f"a slot came back twice: {error}") from None
        assert lender._free_slots(group) == lender.regions[0].slot_count
        again = lender.lend_read(first, 0, END, **kit.HOST_NAMED)
        kit.ready_view(again, mgr)
        again.release()


def test_arrivals_that_raise_leave_the_fetch_abandoned_and_the_slots_returnable(
    kit, real_manager, monkeypatch
):
    """Arrivals whose count raises abandon the fetch; a release that follows counts."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def planted(*args):
        raise RuntimeError("planted")

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        other = kit.admitted(mgr, OTHER, OTHER_PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        (group,) = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        kv = kit.kv(mgr, target)
        with monkeypatch.context() as patched:
            patched.setattr(_lender._ManagerLender, "_locked_block_masks", planted)
            with pytest.raises(RuntimeError, match="planted"):
                kit.return_write(lender, lease, view.row_masks(True))
        assert lender.poll_resume_bounds(target) == (kv.num_committed_tokens, kv.history_length)
        lease.release()
        lender.progress()
        waiting = lender.lend_write(other, 0, END, **kit.HOST_NAMED)
        waiting_view = waiting.poll()
        assert waiting_view is not None, "the lease's slots never came back"
        opened = lender._open_lease_count()
        with monkeypatch.context() as patched:
            patched.setattr(_lender._ManagerLender, "_progress", planted)
            waiting.release()  # runs no round of progress
            assert lender._open_lease_count() == opened - 1
            with pytest.raises(RuntimeError, match="planted"):
                lender.progress()
        other_kv = kit.kv(mgr, other)
        own = (other_kv.num_committed_tokens, other_kv.history_length)
        assert lender.poll_resume_bounds(other) == own, "the write released ready delivered rows"
        lender.progress()
        assert lender._free_slots(group) == lender.regions[0].slot_count


@pytest.mark.parametrize("interrupted", [False, True], ids=["memory_error", "interrupted"])
def test_arrivals_whose_copy_raises_once_queued_lose_the_slots_it_reads(
    kit, real_manager, monkeypatch, interrupted, attach=attach
):
    """A return whose queued copy's event raises, or is interrupted, loses the slots it reads."""
    planted = Interrupted if interrupted else MemoryError

    def no_event(*args, **kwargs):
        raise planted("planted")

    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        (group,) = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        kv = kit.kv(mgr, target)
        with kit.held_stream(mgr._stream):
            with monkeypatch.context() as patched:
                patched.setattr(torch.cuda, "Event", no_event)
                with pytest.raises(planted, match="planted"):
                    # The copy is queued, then raises.
                    kit.return_write(lender, lease, view.row_masks(True))
            assert lender.poll_resume_bounds(target) == (kv.num_committed_tokens, kv.history_length)
            lease.release()
            lender.progress()
            assert lender._free_slots(group) == 0, (
                "slots came back while a copy out of them was queued"
            )
        lender.progress()
        assert lender._free_slots(group) == 0, "slots a copy may still read were reused"


class Interrupted(BaseException):
    """An interruption that is no ``Exception``, as ``KeyboardInterrupt`` is."""


SLOTS_BACK = "slots came back while a copy on them was queued"


def interrupted_event(*args, **kwargs):
    raise Interrupted("planted")


def test_an_interrupt_in_a_read_s_grant_keeps_its_slots_out_of_use(
    kit, real_manager, monkeypatch, attach=attach
):
    """An interrupt as a waiting read's grant queues its copy keeps the slots out of use."""
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        (group,) = kit.pool_group_ids(mgr)
        head = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(head, mgr)
        waiting = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        assert waiting.poll() is None, "the read was granted at once: the check proves nothing"
        head.release()
        with kit.held_stream(mgr._stream):
            with monkeypatch.context() as patched:
                patched.setattr(torch.cuda, "Event", interrupted_event)
                with pytest.raises(Interrupted, match="planted"):
                    lender.progress()  # grants the waiting read; its copy's event is interrupted
            runs = waiting._slot_runs
            waiting.release()
            lender.progress()
            assert lender._free_slots(group) == 0, SLOTS_BACK
        lender.progress()
        assert any(r is runs for r in lender._quarantined), SLOTS_BACK
        later = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        lender.progress()
        assert later.poll() is None and later.failure is None, "a grant handed the slots out"
        later.release()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(regions), "staging freed at shutdown with slots out of use"
        assert warnings, "keeping the memory until exit is logged"


def interrupting_as_it_takes_a_copy(monkeypatch, lease):
    """Make the first setting of ``lease``'s ``_copy`` to a copy raise ``Interrupted``."""
    cls = type(lease)
    assign = cls.__setattr__
    interrupted = []

    def interrupting(self, name, value):
        if self is lease and name == "_copy" and value is not None and not interrupted:
            interrupted.append(True)
            raise Interrupted("planted")
        assign(self, name, value)

    monkeypatch.setattr(cls, "__setattr__", interrupting)


def test_an_interrupt_before_a_write_takes_its_copy_keeps_the_slots_out_of_use(
    kit, real_manager, monkeypatch, attach=attach
):
    """An interrupt as a returned write takes its copy keeps the slots out of use."""
    with real_manager() as mgr:
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        regions = kit.staging_regions(lender)
        (group,) = kit.pool_group_ids(mgr)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        view = lease.poll()
        interrupting_as_it_takes_a_copy(monkeypatch, lease)
        with kit.held_stream(mgr._stream):
            with pytest.raises(Interrupted, match="planted"):
                # The copy is queued, then interrupted.
                kit.return_write(lender, lease, view.row_masks(True))
            runs = lease._slot_runs
            lease.release()
            lender.progress()
            assert lender._free_slots(group) == 0, SLOTS_BACK
        lender.progress()
        assert any(r is runs for r in lender._quarantined), SLOTS_BACK
        later = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        lender.progress()
        assert later.poll() is None and later.failure is None, "a grant handed the slots out"
        later.release()
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(regions), "staging freed at shutdown with slots out of use"
        assert warnings, "keeping the memory until exit is logged"


def test_a_grant_that_raises_fails_only_its_own_lease(
    kit, real_manager, monkeypatch, attach=attach
):
    """A grant that raises fails its own lease alone; the read behind it is granted that round."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    with real_manager() as mgr:
        holder = kit.published(mgr, SOURCE, PROMPT)
        broken = kit.published(mgr, OTHER, OTHER_PROMPT)
        fetched = kit.admitted(mgr, TARGET, list(range(8000, 8097)))
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        write = lender.lend_write(fetched, 0, END, **kit.HOST_NAMED)
        kit.return_write(lender, write, write.poll().row_masks())
        head = lender.lend_read(holder, 0, END, **kit.HOST_NAMED)
        kit.ready_view(head, mgr)
        waiting = lender.lend_read(broken, 0, 2 * TPB, **kit.HOST_NAMED)
        assert waiting.poll() is None
        behind = lender.lend_read(holder, 0, TPB, **kit.HOST_NAMED)  # healthy, in line behind it
        assert behind.poll() is None
        broken_kv = kit.kv(mgr, broken)

        real_kv_of = _manager.kv_of

        def kv_of(manager, request_id):
            if request_id == OTHER:
                raise RuntimeError("planted")
            return real_kv_of(manager, request_id)

        def raising_for_broken(read):
            def call(kv_cache, *rest):
                if kv_cache is broken_kv:
                    raise RuntimeError("planted")
                return read(kv_cache, *rest)

            return call

        with monkeypatch.context() as patched:
            patched.setattr(_manager, "kv_of", kv_of)
            for name in ("cache_state", "pages", "locked_pages"):
                patched.setattr(_manager, name, raising_for_broken(getattr(_manager, name)))
            head.release()
            lender.progress()  # grants both waiting reads, and the first one's grant raises
            granted = behind._view is not None
            assert waiting.poll() is None and waiting.failure is not None
            assert granted and behind.failure is None, (
                f"the read behind the broken one was not granted in that round: {behind.failure}"
            )
            assert isinstance(lender.poll_resume_bounds(fetched), ResumeBounds)
        view = kit.ready_view(behind, mgr)
        assert 0 < view.num_rows == check_staged(kit, mgr, lender, holder, view)
        behind.release()
        waiting.release()
        again = lender.lend_read(holder, 0, END, **kit.HOST_NAMED)
        kit.ready_view(again, mgr)
        again.release()


@pytest.mark.parametrize("fault", ["none", "cancel"], ids=["control", "cancel_raises"])
def test_an_error_in_the_free_hook_stops_the_lender(kit, real_manager, monkeypatch, fault):
    """An error the free hook meets stops the lender for good; without it the next round grants."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    with real_manager(windows=[WINDOW, 256], num_layers=3) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_WINDOWED_PROMPT)
        seen_target = kit.admitted(mgr, 4, PROMPT)
        granted_target = kit.admitted(mgr, 5, list(range(8000, 8097)))
        other = kit.admitted(mgr, OTHER, list(range(9000, 9097)))
        lender = attach(mgr, guaranteed_fetch_tokens=WINDOWED_END, capacity_in_fetches=3)
        regions = kit.staging_regions(lender)
        groups = kit.pool_group_ids(mgr)
        seen = lender.lend_write(seen_target, 0, 2 * TPB, **kit.HOST_NAMED)
        seen_view = seen.poll()
        granted = lender.lend_write(granted_target, 0, TPB, **kit.HOST_NAMED)  # never polled
        assert seen_view is not None and granted.failure is None
        kit.fill_sentinel(mgr, seen_target)
        kit.stage(lender, seen_view, 0x5A)
        # Grown ahead, so the held section only lends.
        kv = kit.kv(mgr, target)
        assert mgr._resize_for_served_prefix(target, kv, WINDOWED_END, WINDOWED_END)
        assert mgr._resize_for_served_prefix(other, kit.kv(mgr, other), TPB, TPB)
        with kit.held_stream(mgr._stream) as gate:
            publish = lender.lend_read(source, 0, WINDOWED_END, **kit.HOST_NAMED)  # its copy held
            fetch = lender.lend_write(target, 0, WINDOWED_END, **kit.HOST_NAMED)
            assert fetch.poll() is None and fetch.failure is None, "the fetch did not wait"
            behind = lender.lend_write(other, 0, TPB, **kit.HOST_NAMED)
            assert behind.poll() is None and behind.failure is None, "the write did not wait"
            publish.release()  # its copy pends, so its slots stay out of use
            gate.open()
        cancelled = []

        def planted(cancel):
            def call(ticket):
                cancelled.append(ticket)
                if len(cancelled) == 1:  # the freed request's write, failed before its cancel
                    raise MemoryError("planted")
                return cancel(ticket)

            return call

        if fault == "cancel":
            monkeypatch.setattr(lender._slot_queue, "cancel", planted(lender._slot_queue.cancel))
        mgr.free_resources(other)  # the manager's free runs the lender's hook
        assert kit.kv(mgr, other) is None
        assert behind.failure is not None, "a write waiting for slots outlived its request"
        if fault == "none":
            lender.progress()
            assert fetch.poll() is not None and fetch.failure is None, "the round granted nothing"
            assert granted.poll() is not None
            kit.return_write(lender, seen, seen_view.row_masks(True))
            mgr._stream.synchronize()
            for run in seen_view.groups:
                nbytes = lender.regions[run.region_index].slot_bytes
                for ordinal in run.page_ordinals.tolist():
                    got = kit.digest(kit.page(mgr, seen_target, run.layer_group_index, ordinal))
                    want = kit.digest(kit.staged_row(0x5A, run.layer_group_index, ordinal, nbytes))
                    assert got == want, "a return copied nothing"
            for lease in (fetch, granted):
                kit.return_write(lender, lease, lease.poll().row_masks())
            behind.release()
            mgr.shutdown()
            assert not kit.staging_kept(regions), "nothing open, yet the staging memory stays"
            return
        assert len(cancelled) == 1, "the planted fault did not strike: proves nothing"
        free = {g: lender._free_slots(g) for g in groups}
        lender.progress()
        assert fetch.failure is not None, "the waiting fetch neither became ready nor failed"
        stop = fetch.failure
        assert stop == _lender._REASON_STOPPED, stop
        assert fetch.poll() is None
        for request in (target, seen_target):
            interval = lender.poll_resume_bounds(request)
            assert interval is not None and interval.max_position < interval.min_position, (
                f"resume_bounds {interval} after the stop, not an empty interval"
            )
        assert granted.poll() is None and granted.failure == stop, "a view went out after the stop"
        assert seen.poll() is seen_view, "a view a poll returned before the stop was taken back"
        for lend, request in ((lender.lend_read, source), (lender.lend_write, target)):
            with pytest.raises(RuntimeError, match="lends no more") as raised:
                lend(request, 0, WINDOWED_END, **kit.HOST_NAMED)
            assert stop in str(raised.value), f"{lend.__name__}: {raised.value}"
        with pytest.raises(ValueError):
            lender.lend_read(source, 0, TPB + 1, **kit.HOST_NAMED)  # not whole blocks
        with pytest.raises(ValueError):
            # Past the last prompt token's block.
            lender.lend_write(target, 0, WINDOWED_END + TPB, **kit.HOST_NAMED)
        bounds = kit.return_write(lender, seen, seen_view.row_masks(True))
        assert bounds is not None and bounds.max_position < bounds.min_position, (
            f"a return after the stop gave {bounds}, not an empty interval"
        )
        with pytest.raises(RuntimeError):
            kit.return_write(lender, seen, seen_view.row_masks(True))
        mgr._stream.synchronize()
        for run in seen_view.groups:
            sentinel = bytes([kit.SENTINEL]) * lender.regions[run.region_index].slot_bytes
            for ordinal in run.page_ordinals.tolist():
                got = kit.digest(kit.page(mgr, seen_target, run.layer_group_index, ordinal))
                assert got == kit.digest(sentinel), "a return copied rows after the stop"
        for lease in (granted, fetch, behind):
            lease.release()
        assert lender._open_lease_count() == 0
        lender.progress()
        assert {g: lender._free_slots(g) for g in groups} == free, "slots returned or granted"
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert kit.staging_kept(regions), (
            "the shutdown freed staging whose slot records may be wrong"
        )
        assert warnings, "keeping the staging memory until exit is logged"


# -- names ------------------------------------------------------------------------------------


def names_by_rank(kit, real_manager, attach, num_kv_heads, scopes=(SCOPE, SCOPE)):
    """Each rank of a TP2 pair publishes ``PROMPT``: its names, and its region names."""
    names, region_labels = [], []
    for rank, scope in enumerate(scopes):
        with real_manager(num_kv_heads=num_kv_heads, mapping=tp_rank(rank)) as mgr:
            source = kit.published(mgr, SOURCE, PROMPT)
            lender = attach(mgr, guaranteed_fetch_tokens=END, scope=scope)
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            (run,) = kit.ready_view(lease, mgr).groups
            names.append(np.array(run.object_keys))
            region_labels.append([p.layout_label for p in lender.regions])
            lease.release()
    return names, region_labels


def test_a_replicated_group_is_named_alike_on_every_rank(kit, real_manager, attach=attach):
    (rank0, rank1), (parts0, parts1) = names_by_rank(kit, real_manager, attach, num_kv_heads=1)
    assert rank0.shape == (BLOCKS, 54) and rank0.dtype == np.uint8
    assert np.array_equal(rank0, rank1), "ranks holding the same bytes must share their names"
    assert parts0 == parts1


def test_a_head_sharded_group_names_each_rank_s_share(kit, real_manager):
    (rank0, rank1), (parts0, parts1) = names_by_rank(kit, real_manager, attach, num_kv_heads=4)
    assert np.array_equal(rank0[:, :50], rank1[:, :50]), "same scope, layout and blocks"
    for rank, names in enumerate((rank0, rank1)):
        shard = (2).to_bytes(2, "big") + rank.to_bytes(2, "big")
        assert {bytes(n[50:54]) for n in names} == {shard}
    assert parts0 == parts1


def test_block_names_follow_the_scope_and_region_labels_only_the_layout(kit, real_manager):
    names = []
    for scope in (b"model-a", b"model-b"):
        with real_manager() as mgr:
            source = kit.published(mgr, SOURCE, PROMPT)
            lender = attach(mgr, guaranteed_fetch_tokens=END, scope=scope)
            lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            view = kit.ready_view(lease, mgr)
            names.append(([bytes(n) for n in view.groups[0].object_keys], lender.regions))
            lease.release()
    (first, first_regions), (second, second_regions) = names
    assert [n[16:] for n in first] == [n[16:] for n in second], "same blocks, same keys"
    assert not set(first) & set(second)
    assert [p.layout_label for p in first_regions] == [p.layout_label for p in second_regions], (
        "laid out alike"
    )
    with real_manager(tokens_per_block=2 * TPB) as mgr:
        other_layout = attach(mgr, guaranteed_fetch_tokens=END, scope=b"model-a").regions
    assert [p.layout_label for p in other_layout] != [p.layout_label for p in first_regions], (
        "another layout"
    )


CONTROL_PROMPT = list(range(9100, 9197))
IMAGE = {"image": {"pixel_values": torch.zeros(4)}}
DIGESTS = dict(
    multimodal_hashes=[list(range(1, 9))], multimodal_positions=[40], multimodal_lengths=[16]
)
# What a request carries beyond the tokens its names cover, how its lease fails, and what a control
# request whose names cover what it carries holds instead.
UNNAMED = {
    "multimodal_without_digests": (
        dict(py_multimodal_data=IMAGE),
        "without digests",
        dict(py_multimodal_data=IMAGE, **DIGESTS),
    ),
    "encoder_input": (dict(encoder_input_tokens=list(range(300, 316))), "encoder input", {}),
}


def carrying(kit, request_id, tokens, inputs):
    """A request carrying ``inputs``, at its context as the executor leaves it after any encoder."""
    from tensorrt_llm.bindings import LlmRequestState

    request = kit.make_request(request_id, tokens, **inputs)
    request.state = LlmRequestState.CONTEXT_INIT
    return request


@pytest.mark.parametrize("case", list(UNNAMED))
def test_a_request_whose_kv_its_names_miss_is_not_lent(kit, real_manager, case, attach=attach):
    inputs, why, covered = UNNAMED[case]
    with real_manager() as mgr:
        source = carrying(kit, SOURCE, PROMPT, inputs)
        kit.prefill(mgr, source)
        target = carrying(kit, TARGET, OTHER_PROMPT, inputs)
        assert mgr.prepare_context(target)
        control = kit.published(mgr, OTHER, CONTROL_PROMPT, **covered)
        lender = attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2)
        before = [state_of(kit, mgr, lender, r) for r in (source, target)]
        refused = [
            lender.lend_read(source, 0, END, **kit.HOST_NAMED),
            lender.lend_write(target, 0, END, **kit.HOST_NAMED),
        ]
        for lease in refused:
            assert lease.poll() is None and why in str(lease.failure), (
                f"lent a request carrying {case.replace('_', ' ')}"
            )
        assert [state_of(kit, mgr, lender, r) for r in (source, target)] == before
        lent = lender.lend_read(control, 0, END, **kit.HOST_NAMED)
        assert check_staged(kit, mgr, lender, control, kit.ready_view(lent, mgr)) == BLOCKS
        for lease in (*refused, lent):
            lease.release()


# name: (windows, multimodal run [b, e), the fetch's end, mm_bidirectional_blocks set, refused)
SPLIT_RUNS = {
    "end_inside_a_run_longer_than_the_window": ([WINDOW, 256], (1, 130), 128, True, True),
    "end_past_the_run": ([WINDOW, 256], (1, 130), 160, True, False),
    "end_inside_a_run_within_the_window": ([WINDOW, 256], (1, 65), 32, True, True),
    # The manager's window holds the run, or there is none: the model's may still be shorter.
    "end_inside_a_run_within_a_wider_window": ([2 * WINDOW, 256], (1, 100), 96, True, True),
    "end_inside_a_run_without_a_window": (None, (1, 130), 128, True, True),
    "run_not_bidirectional": ([WINDOW, 256], (1, 130), 128, False, False),
    "end_past_a_run_the_floor_may_fall_below": ([80, 256], (1, 87), 96, True, False),
}


def bidirectional(runs, flagged, length):
    """Multimodal data whose tokens ``runs`` attend both ways where ``flagged``, with digests."""
    mask = np.zeros(length, dtype=np.int64)
    for beg, end in runs:
        mask[beg:end] = 1
    data = dict(IMAGE, mm_bidirectional_blocks=flagged)
    data["multimodal_embed_mask_cumsum"] = torch.from_numpy(np.cumsum(mask))
    return dict(py_multimodal_data=data, **DIGESTS)


@pytest.mark.parametrize("case", list(SPLIT_RUNS))
def test_a_fetch_ending_inside_a_bidirectional_run_fails_at_the_call(
    kit, real_manager, case, attach=attach
):
    """A request whose multimodal tokens attend both ways in runs the scheduler keeps within one
    chunk: a fetch whose end falls strictly inside a run fails at the call, whatever the run's
    length and the manager's windows, which need not be the model's; every other fetch is lent."""
    windows, run, end, flagged, refused = SPLIT_RUNS[case]
    with real_manager(windows=windows) as mgr:
        inputs = bidirectional([run], flagged, len(WINDOWED_PROMPT))
        target = carrying(kit, TARGET, WINDOWED_PROMPT, inputs)
        assert mgr.prepare_context(target)
        kit.kv(mgr, target).enable_swa_scratch_reuse = False  # a fetch target needs it off
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        before = state_of(kit, mgr, lender, target)
        lease = lender.lend_write(target, 0, end, **kit.HOST_NAMED)
        if refused:
            assert lease.poll() is None and "multimodal" in str(lease.failure), (
                f"lent a fetch ending inside the run {run}: {case}"
            )
            assert state_of(kit, mgr, lender, target) == before, (
                "the failed lease changed the cache"
            )
        else:
            kit.ready_view(lease, mgr)
        lease.release()


# name: (windows, multimodal runs, the fetch's end, block ordinals that arrive (None: all),
# mm_bidirectional_blocks set, resume_bounds)
RUN_READINESS = {
    "delivered_rows_ending_inside_a_run": (None, [(40, 100)], 160, [0, 1], True, (40, 0)),
    "a_floor_below_a_run": (None, [(5, 30), (40, 100)], 160, None, True, (160, 100)),
    "a_floor_below_a_run_no_window_releases": ([80, 256], [(1, 87)], 96, None, True, (96, 87)),
    "without_the_flag": (None, [(40, 100)], 160, None, False, (160, 0)),
}


@pytest.mark.parametrize("case", list(RUN_READINESS))
def test_resume_bounds_keeps_resumes_out_of_bidirectional_runs(
    kit, real_manager, case, attach=attach
):
    """All-reusable, nothing computed, a fetch past every run into a request whose multimodal tokens
    attend both ways: resume_bounds ends at the start of a run the delivered rows end inside and starts
    at the end of the last run below its end; a request without the flag keeps its interval."""
    windows, runs, end, arrived, flagged, expected = RUN_READINESS[case]
    with real_manager(windows=windows) as mgr:
        inputs = bidirectional(runs, flagged, len(WINDOWED_PROMPT))
        target = carrying(kit, TARGET, WINDOWED_PROMPT, inputs)
        assert mgr.prepare_context(target)
        kit.kv(mgr, target).enable_swa_scratch_reuse = False  # a fetch target needs it off
        lender = attach(mgr, guaranteed_fetch_tokens=end)
        fetch_segment(kit, mgr, lender, target, 0, end, 0x5A, arrived)
        for _ in range(2):  # the second answer comes from the record resume_bounds keeps
            resume_bounds = tuple(lender.poll_resume_bounds(target))
            assert resume_bounds == expected, (
                f"resume_bounds {resume_bounds} lets the request resume inside a run of {runs}: {case}"
            )


def test_resume_bounds_keeps_a_resume_out_of_the_run_the_context_position_is_in(
    kit, real_manager, attach=attach
):
    """Per-request, a first chunk that ended inside a run, as a whole-block local match can leave
    it, and a fetch past the run abandoned: with nothing delivered, resume_bounds ends at the run's
    start, below the context position, so the request does not resume inside the run."""
    with real_manager(**PER_REQUEST) as mgr:
        inputs = bidirectional([(40, 100)], True, len(SPLIT_PROMPT))
        target = computed(kit, mgr, TARGET, SPLIT_PROMPT, **inputs)
        lender = attach(mgr, guaranteed_fetch_tokens=2 * TPB)
        lender.lend_write(target, LOCAL, 4 * TPB, **kit.HOST_NAMED).release()  # abandoned unpolled
        resume_bounds = lender.poll_resume_bounds(target)
        assert tuple(resume_bounds) == (40, LOCAL), (
            f"resume_bounds {tuple(resume_bounds)} lets the request resume inside the run [40, 100)"
        )


# -- threads ----------------------------------------------------------------------------------


@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
def test_threads_take_turns_and_the_lender_starts_none(kit, real_manager, monkeypatch, pinned):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender, "prefer_pinned", lambda: pinned)
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        target = kit.admitted(mgr, TARGET, OTHER_PROMPT)
        before = set(threading.enumerate())
        built = kit.on_thread(
            lambda: attach(mgr, guaranteed_fetch_tokens=END, capacity_in_fetches=2),
            "builder",
            cuda=True,
        )
        lender = built["value"]
        regions = kit.staging_regions(lender)
        (memory,) = kit.staging_memory(regions)
        base = regions[0].address
        if pinned:
            assert kit.pinned_range(base) is not None

        def executor_loop():
            read = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
            view = kit.ready_view(read, mgr)
            # A backend reads the arrays on its own thread until the release.
            seen = kit.on_thread(lambda: [bytes(n) for n in view.groups[0].object_keys], "backend")
            write = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
            kit.return_write(lender, write, write.poll().row_masks(True))
            read.release()
            return seen["value"]

        looped = kit.on_thread(executor_loop, "executor-loop", cuda=True)
        assert "error" not in looped and len(looped["value"]) == BLOCKS
        assert set(threading.enumerate()) <= before, "the lender started a thread"
        assert "error" not in kit.on_thread(mgr.shutdown, "shutdown", cuda=True)
        assert memory.address == 0 and memory._pageable is None
        assert not kit.staging_kept(regions), "the shutdown thread kept the staging memory"
        if pinned:
            assert kit.pinned_range(base) is None, "the shutdown thread did not free pinned memory"
        with pytest.raises(ValueError):
            lender.poll_resume_bounds(source)


# -- the manager's page-index buffer, kept from the attach until the runtime's shutdown returns


def test_the_index_buffer_is_kept_from_the_attach_until_the_shutdown(
    kit, real_manager, attach=attach
):
    with real_manager() as mgr:
        before = len(kit.retained())
        buffer = mgr.host_kv_cache_block_offsets
        attach(mgr, guaranteed_fetch_tokens=END)
        assert any(o is buffer for o in kit.retained()), "not kept from the attach"
        mgr.shutdown()
        assert not any(o is buffer for o in kit.retained()), "kept past the shutdown"
        assert len(kit.retained()) == before


class ClosingSecondTime:
    """A ``kv_cache_map`` entry whose first ``close()`` raises and whose later ones close the cache;
    the manager's other calls on it go to the cache."""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.kv_cache, name)

    def close(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("planted")
        self.kv_cache.close()


BUFFER_GONE_EARLY = "the buffer went before every cache closed"


def test_the_index_buffer_stays_kept_through_a_close_that_raises(kit, real_manager, attach=attach):
    """A close that raises stops the manager's shutdown with caches still open, which a released
    lease or a record of the lender may hold and which write their page indices into the buffer as
    they close: staging keeps the buffer until the runtime's shutdown returns, here a retry's."""
    with real_manager() as mgr:
        buffer = mgr.host_kv_cache_block_offsets
        target = kit.admitted(mgr, TARGET, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_write(target, 0, END, **kit.HOST_NAMED)
        kit.return_write(lender, lease, kit.ready_view(lease, mgr).row_masks())
        kv = kit.kv(mgr, target)
        request_id = next(rid for rid, cache in mgr.kv_cache_map.items() if cache is kv)
        mgr.kv_cache_map[request_id] = ClosingSecondTime(kv)  # the released lease's own cache
        with pytest.raises(RuntimeError, match="planted"):
            mgr.shutdown()
        assert not kit.closed(kv), "the planted close closed the cache: the check proves nothing"
        assert any(o is buffer for o in kit.retained()), BUFFER_GONE_EARLY
        mgr.shutdown()
        assert kit.closed(kv), "the retried shutdown left the cache open"
        assert not any(o is buffer for o in kit.retained()), "kept past a shutdown that closed all"


def test_the_index_buffer_stays_kept_past_a_free_whose_close_raised(
    kit, real_manager, wrap=None, attach=attach
):
    """A close that raises in a free leaves the cache open outside the manager's map, held by a
    released read, so the runtime's shutdown raises: staging keeps the buffer until a shutdown
    returns, here a retry's once the cache has closed."""
    with real_manager() as mgr:
        buffer = mgr.host_kv_cache_block_offsets
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach(mgr, guaranteed_fetch_tokens=END)
        lease = lender.lend_read(source, 0, END, **kit.HOST_NAMED)
        kit.ready_view(lease, mgr)
        lease.release()  # a released read holds its cache until it is dropped
        kv = kit.kv(mgr, source)
        mgr.kv_cache_map[source.py_request_id] = ClosingSecondTime(kv)
        with pytest.raises(RuntimeError, match="planted"):
            mgr.free_resources(source)
        assert not kit.closed(kv) and source.py_request_id not in mgr.kv_cache_map, (
            "the free closed the cache or left it in the map: the check proves nothing"
        )
        try:
            if wrap is not None:
                wrap(mgr)
            with pytest.raises(RuntimeError, match="still open"):
                mgr.shutdown()
            assert any(o is buffer for o in kit.retained()), BUFFER_GONE_EARLY
        finally:
            kv.close()
        mgr.shutdown()
        assert not any(o is buffer for o in kit.retained()), "kept past a shutdown that returned"


# A cache writes -1 for every block into its manager's host page-index buffer as it closes. A new
# tensor reclaims a freed buffer's address with a canary the cache must neither read nor write.
FREED = "touched the page-index buffer freed with its manager"


def fresh_device():
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()


def check_a_lease_outliving_its_manager(kit):
    """A staging read outlives its manager and lender, collected without a shutdown, holding the
    request's cache; it also keeps its staging memory as it was."""
    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_lender(mgr, scope=b"index-buffer", staging=StagingOptions(END))
    lease = lender.lend_read(request, 0, END, **kit.HOST_NAMED)
    mgr._stream.synchronize()  # a staging read is ready once its copy into staging ran
    lender.progress()
    view = lease.poll()
    assert view is not None
    (run,) = view.groups
    address, slot_bytes = int(run.addresses[0]), lender.regions[run.region_index].slot_bytes
    regions, content = kit.staging_regions(lender), kit.host_bytes(address, slot_bytes)
    row = kit.index_row(mgr, request)
    own = list(kv.get_base_page_indices(0)[:blocks])
    assert row.values[:blocks] == own and -1 not in own, "the cache writes elsewhere"
    watched = [weakref.ref(o) for o in (mgr.host_kv_cache_block_offsets, mgr, lender)]
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert watched[1]() is None and watched[2]() is None, "the manager or lender was not collected"
    canary = None if watched[0]() is not None else kit.reclaim(row)
    if watched[0]() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    assert kit.staging_kept(regions), "the staging memory went with its manager"
    now = kit.digest(kit.host_bytes(address, slot_bytes))
    assert now == kit.digest(content), "the kept staging memory changed"
    del kv  # the lease now holds the cache's last reference
    lease.release()
    lease.release()
    del lease  # a staging lease keeps its cache past its release
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the lent cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


def test_a_cache_outliving_its_manager_touches_no_freed_index_buffer(kit):
    check_a_lease_outliving_its_manager(kit)


# -- DeepSeek-V4 ------------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize("fp8_ds_mla", [False, True], ids=["fp8", "fp8_ds_mla"])
def test_a_deepseek_v4_prompt_is_fetched_through_staging_byte_for_byte(
    kit, deepseek_v4_manager, fp8_ds_mla
):
    """Three whole blocks published by one DeepSeek-V4 manager and fetched by another, byte for
    byte, in FP8 at 128 tokens per block and in the footer-scale cache at 256."""
    with (
        deepseek_v4_manager(fp8_ds_mla=fp8_ds_mla) as mgr_a,
        deepseek_v4_manager(fp8_ds_mla=fp8_ds_mla) as mgr_b,
    ):
        tpb = mgr_a.tokens_per_block
        assert tpb == (256 if fp8_ds_mla else 128)
        end = 3 * tpb
        prompt = list(range(3000, 3000 + end + 1))
        source = kit.published(mgr_a, SOURCE, prompt)
        lender_a = attach(mgr_a, guaranteed_fetch_tokens=end)
        publish = lender_a.lend_read(source, 0, end, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, mgr_a)
        assert len(publish_view.groups) == kit.num_layer_groups(mgr_a)
        assert check_staged(kit, mgr_a, lender_a, source, publish_view) == publish_view.num_rows
        target = kit.admitted(mgr_b, TARGET, prompt)
        lender_b = attach(mgr_b, guaranteed_fetch_tokens=end)
        assert [p.layout_label for p in lender_a.regions] == [
            p.layout_label for p in lender_b.regions
        ]
        lease = lender_b.lend_write(target, 0, end, **kit.HOST_NAMED)
        view = lease.poll()
        assert by_group(view) == by_group(publish_view)
        kit.fill_sentinel(mgr_b, target)
        masks = kit.relay(lender_a, publish_view, lender_b, view)
        assert return_and_settle(lender_b, mgr_b, target, lease, masks) == (end, end)
        for run in view.groups:
            for ordinal in run.page_ordinals.tolist():
                lg = run.layer_group_index
                got = kit.digest(kit.page(mgr_b, target, lg, ordinal))
                assert got == kit.digest(kit.page(mgr_a, source, lg, ordinal))
        publish.release()


def ngram_drafting(draft_len):
    """NGram drafting ``draft_len`` tokens; ``ngram`` drafts 2."""
    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig

    return NGramDecodingConfig(max_draft_len=draft_len, max_matching_ngram_size=2)


# The lender conservatively reads DeepSeek-V4's full kept window, including the draft margin.
# case -> (spec config, tokens the publisher computes past 512, rows the publish lacks per group)
DSV4_TPB = 128
DSV4_END = 4 * DSV4_TPB
DSV4_PROMPT = list(range(60_000, 60_000 + DSV4_END + 2))  # four whole blocks and two tokens
DSV4_TARGET = DSV4_PROMPT[:DSV4_END] + [69_999, 69_998]  # the same four blocks, then others
DSV4_DRAFT_MARGIN = {
    # A window of 128 at 512 keeps block 3 alone, which a publisher two tokens past still keeps.
    "no_speculation": (lambda: None, 2, {}),
    # A draft length of 2 keeps block 2 too. A publisher past the fetch's end lacks it, so
    # the conservative read contract cannot resume there until that margin row arrives.
    "draft2_publisher_at_the_end": (ngram, 0, {}),
    "draft2_publisher_one_past": (ngram, 1, {0: [2]}),
    # A draft length of 3 keeps block 2 for a publisher one token past, not two.
    "draft3_publisher_one_past": (lambda: ngram_drafting(3), 1, {}),
    "draft3_publisher_two_past": (lambda: ngram_drafting(3), 2, {0: [2]}),
    # PARD also grows the cache by the extra tokens embedded DSpark does.
    "pard2_publisher_two_past": (pard, 2, {0: [2]}),
}
NOT_COMPUTED = "holds bytes other than the ones the publisher computed"


@skip_pre_blackwell
@pytest.mark.parametrize("case", list(DSV4_DRAFT_MARGIN))
def test_a_deepseek_v4_fetch_requires_every_page_in_its_conservative_kept_window(
    kit, deepseek_v4_manager, case
):
    """A missing kept-margin page prevents resume until a later fetch delivers that page."""
    make, tail, lacking = DSV4_DRAFT_MARGIN[case]
    with (
        deepseek_v4_manager(spec_config=make()) as pub,
        deepseek_v4_manager(spec_config=make()) as mgr,
    ):
        assert kit.windows(mgr)[0] == DSV4_TPB + mgr.max_draft_len
        assert mgr.reuse_match_backoff == 0
        source = kit.published(pub, SOURCE, DSV4_PROMPT[: DSV4_END + tail])
        publisher = attach(pub, guaranteed_fetch_tokens=DSV4_END)
        publish = publisher.lend_read(source, 0, DSV4_END, **kit.HOST_NAMED)
        publish_view = kit.ready_view(publish, pub)
        assert check_staged(kit, pub, publisher, source, publish_view) == publish_view.num_rows
        held = {name.tobytes() for run in publish_view.groups for name in run.object_keys}
        target = kit.admitted(mgr, TARGET, DSV4_TARGET)
        lender = attach(mgr, guaranteed_fetch_tokens=DSV4_END)
        lease = lender.lend_write(target, 0, DSV4_END, **kit.HOST_NAMED)
        view = kit.ready_view(lease, mgr)
        kit.fill_sentinel(mgr, target)

        def published_row(i, row):
            return view.groups[i].object_keys[row].tobytes() in held

        masks = kit.relay(publisher, publish_view, lender, view, published_row)
        usable, floor = return_and_settle(lender, mgr, target, lease, masks)
        publish.release()
        for group in lender.page_layout.layer_groups:
            assert group.read_window == group.kept_window, group
        kv = kit.kv(mgr, target)
        keys = kit.chain_keys(kv, DSV4_TARGET)
        dev = kit.DevicePages(mgr)

        def wrong(lg, ordinal):
            """The block's page, digested, unless it holds what the publisher computed."""
            got = kit.digest(dev.read(lg, kit.pages(kv, lg)[ordinal]))
            computed = kit.digest(kit.fake_page(lg, keys[ordinal], dev.page_bytes(lg)))
            return None if got == computed else got

        for i, run in enumerate(view.groups):
            lg = run.layer_group_index
            for row, ordinal in enumerate(run.page_ordinals.tolist()):
                got = wrong(lg, ordinal) if published_row(i, row) else None
                assert got is None, f"fetched block {ordinal} of layer group {lg} {NOT_COMPUTED}"
        if floor <= usable:
            for lg in range(len(lender.page_layout.layer_groups)):
                beg, end = kit.stale_blocks(mgr, lg, usable)
                for ordinal in range(-(-usable // DSV4_TPB)):
                    got = None if beg <= ordinal < end else wrong(lg, ordinal)
                    assert got is None, (
                        f"block {ordinal} of layer group {lg}, which attention reads at the resume "
                        f"point {usable}, {NOT_COMPUTED}: {got}"
                    )
        resume_bounds = (usable, floor)
        asked = {}
        for i, run in enumerate(view.groups):
            rows = enumerate(run.page_ordinals.tolist())
            missing = [ordinal for row, ordinal in rows if not published_row(i, row)]
            if missing:
                asked[run.layer_group_index] = missing
        assert asked == lacking, f"the fetch's rows the publish lacks: {asked}, expected {lacking}"
        if not lacking:
            assert resume_bounds == (DSV4_END, DSV4_END)
            return
        assert not floor <= DSV4_END <= usable, (
            f"resume_bounds {resume_bounds} counts the end despite missing kept pages {lacking}"
        )
        assert usable < floor, f"missing kept pages still permit a resume: {resume_bounds}"
        for lg, ordinals in lacking.items():
            for ordinal in ordinals:
                assert wrong(lg, ordinal) is not None, "the missing page already holds source bytes"

        repair_source = kit.published(pub, OTHER, DSV4_PROMPT[:DSV4_END])
        repair_publish = publisher.lend_read(repair_source, 0, DSV4_END, **kit.HOST_NAMED)
        repair_publish_view = kit.ready_view(repair_publish, pub)
        repair = lender.lend_write(target, 0, DSV4_END, **kit.HOST_NAMED)
        repair_view = kit.ready_view(repair, mgr)

        def missing_row(i, row):
            run = repair_view.groups[i]
            return int(run.page_ordinals[row]) in lacking.get(run.layer_group_index, ())

        repair_masks = kit.relay(publisher, repair_publish_view, lender, repair_view, missing_row)
        assert sum(int(mask.sum()) for mask in repair_masks) == sum(map(len, lacking.values()))
        assert return_and_settle(lender, mgr, target, repair, repair_masks) == (DSV4_END, DSV4_END)
        repair_publish.release()
        for run in repair_view.groups:
            for ordinal in run.page_ordinals.tolist():
                assert wrong(run.layer_group_index, ordinal) is None, (
                    f"repaired fetch lost block {ordinal} of layer group {run.layer_group_index}"
                )
