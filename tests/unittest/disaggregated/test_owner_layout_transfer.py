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
"""Real NIXL transfers between instances whose target and drafter are laid out differently.

Each rank holds a target and a drafter in one ``KVCacheManagerV2``, each with
its own ``Mapping``, and the context and generation instances lay both out
independently. Every KV element written on a context rank encodes the layer,
K or V, global head, logical page, token and channel it stands for; every
other slot of every pool is poisoned. After the transceivers run, each
generation rank must hold exactly the elements its mappings say it holds,
every other slot must be untouched, and no destination byte may be written
twice (a replicated head or page is still written by one sender only).

Expectations come only from the two instances' mappings: a model with CP
holds logical page ``p`` on its CP rank ``p % cp_size``, and TP rank ``r`` of
``tp`` holds heads ``[r * H / tp, (r + 1) * H / tp)``, or head ``r // (tp / H)``
when heads are replicated.
"""

import threading
import time
import uuid
from typing import NamedTuple

import pytest
import torch
from test_cache_transceiver_single_process import ThreadSafeDistributed, run_concurrent

import tensorrt_llm
import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.transfer import Sender, WriteMetaType
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.mapping import CpType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

P = 8  # tokens per page
HEAD_DIM = 32
TARGET_LAYERS, DRAFT_LAYERS = 2, 2
TARGET_HEADS = 4
MAX_SEQ_LEN = 1024
MAX_PAGES = MAX_SEQ_LEN // P
TARGET, DRAFT = KVCacheLayerOwner.TARGET, KVCacheLayerOwner.DRAFT
# Prompts ending inside a page, on a page boundary, and past several CP
# cycles, up to two CP32 cycles and a partial page.
REQUEST_LENGTHS = (1, 7, 8, 9, 33, 64, 67, 100, 255, 256, 257, 520)


class Instance(NamedTuple):
    """One instance's target and drafter layouts; the target's mapping is the rank's."""

    target_tp: int
    target_cp: int
    draft_tp: int
    draft_cp: int
    mla: bool = False  # both models keep one latent per token (K only), on every TP rank

    @property
    def kv_factor(self):
        return 1 if self.mla else 2

    @property
    def world(self):
        return self.target_tp * self.target_cp

    def sizes(self, owner):
        return (
            (self.target_tp, self.target_cp) if owner == TARGET else (self.draft_tp, self.draft_cp)
        )

    def __str__(self):
        mla = "_mla" if self.mla else ""
        return f"t{self.target_tp}x{self.target_cp}_d{self.draft_tp}x{self.draft_cp}{mla}"


def mapping(instance, rank, owner):
    tp, cp = instance.sizes(owner)
    assert tp * cp == instance.world
    return Mapping(
        world_size=instance.world,
        rank=rank,
        tp_size=tp,
        cp_size=cp,
        cp_config={"cp_type": CpType.HELIX} if cp > 1 else {},
    )


def layers(owner):
    return (
        range(TARGET_LAYERS)
        if owner == TARGET
        else range(TARGET_LAYERS, TARGET_LAYERS + DRAFT_LAYERS)
    )


def owner_of(layer):
    return TARGET if layer < TARGET_LAYERS else DRAFT


def global_heads(owner, draft_heads):
    return TARGET_HEADS if owner == TARGET else draft_heads


def local_heads(instance, rank, owner, draft_heads):
    """Global heads a rank holds of a model, by its TP rank in that model's mapping."""
    if instance.mla:
        return [0]
    heads = global_heads(owner, draft_heads)
    m = mapping(instance, rank, owner)
    if m.tp_size <= heads:
        per_rank = heads // m.tp_size
        return list(range(m.tp_rank * per_rank, (m.tp_rank + 1) * per_rank))
    return [m.tp_rank // (m.tp_size // heads)]


def owned_pages(instance, rank, owner, num_tokens):
    m = mapping(instance, rank, owner)
    return [p for p in range(-(-num_tokens // P)) if p % m.cp_size == m.cp_rank]


def tags(layer, heads, page, device, kv_factor):
    """The value of every element of one page of one layer: [K/V, head, token, channel]."""
    assert max(heads) < 8 and page < MAX_PAGES
    role = torch.arange(kv_factor, device=device).view(kv_factor, 1, 1, 1)
    head = torch.tensor(heads, device=device).view(1, -1, 1, 1)
    token = torch.arange(P, device=device).view(1, 1, P, 1)
    channel = torch.arange(HEAD_DIM, device=device).view(1, 1, 1, HEAD_DIM)
    code = ((((layer * 2 + role) * 8 + head) * MAX_PAGES + page) * P + token) * HEAD_DIM + channel
    return code.to(torch.float32)


def make_manager(instance, rank, draft_heads):
    target = mapping(instance, rank, TARGET)
    specs = [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=owner_of(layer),
            dtype=DataType.FLOAT,
            num_kv_heads=len(local_heads(instance, rank, owner_of(layer), draft_heads)),
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY,) if instance.mla else (Role.KEY, Role.VALUE),
        )
        for layer in range(TARGET_LAYERS + DRAFT_LAYERS)
    ]
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            max_attention_window=[MAX_SEQ_LEN],
        ),
        CacheType.SELFKONLY if instance.mla else CacheType.SELF,
        num_layers=TARGET_LAYERS,
        num_kv_heads=len(local_heads(instance, rank, TARGET, draft_heads)),
        head_dim=HEAD_DIM,
        tokens_per_block=P,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=len(REQUEST_LENGTHS),
        mapping=target,
        dtype=DataType.FLOAT,
        vocab_size=32000,
        layer_specs=specs,
        owner_mappings={TARGET: target, DRAFT: mapping(instance, rank, DRAFT)},
    )


def make_transceivers(instance, managers, config):
    tp, cp = instance.target_tp, instance.target_cp
    shared = {
        "barrier": threading.Barrier(instance.world),
        "pp_barriers": [threading.Barrier(1) for _ in range(tp * cp)],
        "tp_barriers": [threading.Barrier(tp) for _ in range(cp)],
        "lock": threading.Lock(),
    }

    def create(rank):
        m = mapping(instance, rank, TARGET)
        dist = ThreadSafeDistributed(
            rank, instance.world, tp, 1, m.tp_rank, 0, shared, m.cp_rank, cp
        )
        return KvCacheTransceiverV2(
            mapping=m, dist=dist, kv_cache_manager=managers[rank], cache_transceiver_config=config
        )

    return run_concurrent(list(range(instance.world)), create)


def local_prompt_len(instance, rank, num_tokens):
    """A rank's prompt share: its target's pages (helix splits the prompt by the target's CP)."""
    pages = owned_pages(instance, rank, TARGET, num_tokens)
    return sum(min(P, num_tokens - p * P) for p in pages)


def make_request(instance, rank, request_id, num_tokens, request_type, **params):
    sampling = SamplingParams()
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(num_tokens)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(sampling._get_sampling_config()),
        is_streaming=False,
        llm_request_type=request_type,
    )
    # Generation-first scheduling carries the first token and drafts as auxiliary data.
    req.py_disaggregated_params = DisaggregatedParams(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST, **params
    )
    if instance.target_cp > 1:
        req.prompt_len = local_prompt_len(instance, rank, num_tokens)
        req.py_prompt_len = req.prompt_len
        req.total_input_len_cp = num_tokens
    return req


def add_sequence(manager, request_id, num_tokens, generation):
    cache = manager._create_kv_cache(request_id, None, None)
    cache.stop_committing()
    assert cache.resume(torch.cuda.current_stream().cuda_stream)
    if generation:
        assert cache.resize(num_tokens, history_length=num_tokens)
    else:
        assert cache.resize(num_tokens)
        cache.resize(None, history_length=num_tokens)
    return cache


def page_slots(manager, request_id, instance, rank, layer, num_tokens):
    """Pool slots of the logical pages a rank holds of a layer, read from its attention block table."""
    owner = owner_of(layer)
    pages = owned_pages(instance, rank, owner, num_tokens)
    table = [s for s in manager.get_batch_cache_indices([request_id], layer)[0] if s >= 0]
    assert len(table) >= len(pages)
    return dict(zip(pages, table))


def pool_memory(manager):
    """Every KV pool of a manager as one flat float tensor each."""
    pools = {}
    for group in KVRegionExtractorV1(manager).page_table.pool_groups:
        for pool in group.pools:
            pools[pool.base_address] = max(
                pools.get(pool.base_address, 0), pool.slot_bytes * pool.num_slots
            )
    return [
        convert_to_torch_tensor(TensorWrapper(base, DataType.FLOAT, [size // 4]))
        for base, size in pools.items()
    ]


def fill(managers, instance, request_ids, draft_heads, value):
    """Poison all KV memory, then tag each page a rank holds of each request."""
    for rank, manager in enumerate(managers):
        for pool in pool_memory(manager):
            pool.fill_(value)
        if not request_ids:
            continue
        for layer in range(TARGET_LAYERS + DRAFT_LAYERS):
            buffer = manager.get_buffers(layer, kv_layout="HND")
            heads = local_heads(instance, rank, owner_of(layer), draft_heads)
            for request_id, num_tokens in request_ids[rank].items():
                for page, slot in page_slots(
                    manager, request_id, instance, rank, layer, num_tokens
                ).items():
                    buffer[slot] = tags(layer, heads, page, buffer.device, instance.kv_factor)
    torch.cuda.synchronize()


def check_received(managers, instance, request_ids, draft_heads):
    """Each generation rank holds exactly its tagged pages and heads; all other KV memory is untouched."""
    for rank, manager in enumerate(managers):
        written = 0
        for layer in range(TARGET_LAYERS + DRAFT_LAYERS):
            buffer = manager.get_buffers(layer, kv_layout="HND")
            heads = local_heads(instance, rank, owner_of(layer), draft_heads)
            for request_id, num_tokens in request_ids[rank].items():
                for page, slot in page_slots(
                    manager, request_id, instance, rank, layer, num_tokens
                ).items():
                    want = tags(layer, heads, page, buffer.device, instance.kv_factor)
                    assert torch.equal(buffer[slot], want), (
                        f"rank {rank} layer {layer} page {page} of a {num_tokens}-token prompt: "
                        f"{_describe_mismatch(buffer[slot], want)}"
                    )
                    written += want.numel()
        # The pages checked above are distinct memory (their tags differ), so
        # any further non-poison element is a write to memory nobody owns.
        landed = sum(int((~torch.isnan(pool)).sum()) for pool in pool_memory(manager))
        assert landed == written, f"rank {rank}: {landed - written} elements written astray"


def _describe_mismatch(got, want):
    bad = (got != want).nonzero()[0].tolist()
    return f"first mismatch at [kv, head, token, channel]={bad}: got {got[tuple(bad)]}, want {want[tuple(bad)]}"


class WriteRecorder:
    """Destination byte ranges of every KV and auxiliary write the senders submit."""

    def __init__(self, monkeypatch):
        self.ranges = {WriteMetaType.KV: [], WriteMetaType.AUX: []}
        self.lock = threading.Lock()
        real = Sender._make_agent_request

        def record(write_meta, device_id):
            with self.lock:
                self.ranges[write_meta.meta_type].extend(
                    zip(write_meta.dst_ptrs.tolist(), write_meta.sizes.tolist())
                )
            return real(write_meta, device_id)

        monkeypatch.setattr(Sender, "_make_agent_request", staticmethod(record))

    def assert_no_byte_written_twice(self):
        for kind, ranges in self.ranges.items():
            ranges = sorted(ranges)
            assert ranges, f"no {kind} write was submitted"
            for (start, size), (next_start, _) in zip(ranges, ranges[1:]):
                assert start + size <= next_start, f"{kind} bytes at {next_start:#x} written twice"


def run_transfer(ctx, gen, draft_heads, monkeypatch):
    recorder = WriteRecorder(monkeypatch)
    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=512
    )
    ctx_managers = [make_manager(ctx, rank, draft_heads) for rank in range(ctx.world)]
    gen_managers = [make_manager(gen, rank, draft_heads) for rank in range(gen.world)]
    caches = []
    transceivers = []
    try:
        transceivers += make_transceivers(ctx, ctx_managers, config)
        ctx_tcs = transceivers[: ctx.world]
        transceivers += make_transceivers(gen, gen_managers, config)
        gen_tcs = transceivers[ctx.world :]
        endpoint = ctx_tcs[0]._context_info_endpoint

        ctx_requests = {rank: [] for rank in range(ctx.world)}
        gen_requests = {rank: [] for rank in range(gen.world)}
        for index, num_tokens in enumerate(REQUEST_LENGTHS):
            unique = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
            ctx_id, gen_id = 2 * index, 2 * index + 1
            for rank in range(ctx.world):
                req = make_request(
                    ctx,
                    rank,
                    ctx_id,
                    num_tokens,
                    LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
                    disagg_request_id=unique,
                )
                ctx_requests[rank].append((req, num_tokens))
            for rank in range(gen.world):
                req = make_request(
                    gen,
                    rank,
                    gen_id,
                    num_tokens,
                    LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
                    ctx_request_id=ctx_id,
                    ctx_dp_rank=0,
                    ctx_info_endpoint=endpoint,
                    disagg_request_id=unique,
                )
                gen_requests[rank].append((req, num_tokens))

        for managers, requests, generation in (
            (gen_managers, gen_requests, True),
            (ctx_managers, ctx_requests, False),
        ):
            for rank, reqs in requests.items():
                for req, num_tokens in reqs:
                    caches.append(
                        add_sequence(managers[rank], req.py_request_id, num_tokens, generation)
                    )

        def ids(requests):
            return {
                rank: {req.py_request_id: n for req, n in reqs} for rank, reqs in requests.items()
            }

        fill(ctx_managers, ctx, ids(ctx_requests), draft_heads, float("inf"))
        fill(gen_managers, gen, None, draft_heads, float("nan"))

        for rank, reqs in ctx_requests.items():
            for req, _ in reqs:
                req.add_new_token(1000 + req.py_request_id, 0)
        for rank, reqs in gen_requests.items():
            for req, _ in reqs:
                gen_tcs[rank].request_and_receive_async(req)
        # Generation first: a context rank sends once every generation rank it
        # pairs with has asked, so the auxiliary data reaches all of them.
        deadline = time.monotonic() + 60
        for rank, reqs in ctx_requests.items():
            worker = ctx_tcs[rank]._transfer_worker
            for req, _ in reqs:
                unique = req.py_disaggregated_params.disagg_request_id
                while not worker.has_all_peer_req_infos_for_send(unique):
                    assert time.monotonic() < deadline, "generation ranks never asked"
                    time.sleep(0.01)
        for rank, reqs in ctx_requests.items():
            for req, _ in reqs:
                ctx_tcs[rank].respond_and_send_async(req)
        run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))
        torch.cuda.synchronize()

        for rank, reqs in gen_requests.items():
            for req, _ in reqs:
                # The auxiliary data reached every generation rank.
                assert req.py_first_gen_tokens == [1000 + req.py_request_id - 1], (
                    rank,
                    req.py_request_id,
                )
        check_received(gen_managers, gen, ids(gen_requests), draft_heads)
        recorder.assert_no_byte_written_twice()
    finally:
        torch.cuda.synchronize()
        for cache in caches:
            cache.close()
        for tc in transceivers:
            tc.shutdown()
        for manager in ctx_managers + gen_managers:
            manager.shutdown()


TP4 = Instance(4, 1, 4, 1)
TP4_MLA = Instance(4, 1, 4, 1, mla=True)
# Reviewer layouts at full size, every rank a thread on this GPU:
# (A) target TP1 x CP32, drafter TP16 x CP2 (8 draft heads, each on two ranks);
# (B) the CP-replicated drafter of a TP1 x CP32 target, repurposed as TP32;
# (C) MLA target TP2 x CP8, MLA drafter TP1 x CP16.
REVIEWER_A = Instance(1, 32, 16, 2)
REVIEWER_B = Instance(1, 32, 32, 1)
REVIEWER_C = Instance(2, 8, 1, 16, mla=True)
REVIEWER_CASES = [
    pytest.param(TP4, REVIEWER_A, 8, id="a_tp4_to_a"),
    pytest.param(REVIEWER_A, TP4, 8, id="a_a_to_tp4"),
    pytest.param(Instance(4, 1, 1, 4), REVIEWER_A, 8, id="a_draft_cp4_to_a"),
    pytest.param(REVIEWER_A, Instance(4, 1, 1, 4), 8, id="a_a_to_draft_cp4"),
    pytest.param(TP4, REVIEWER_B, 8, id="b_tp4_to_b"),
    pytest.param(REVIEWER_B, TP4, 8, id="b_b_to_tp4"),
    pytest.param(TP4_MLA, REVIEWER_C, 1, id="c_tp4_to_c"),
    pytest.param(REVIEWER_C, TP4_MLA, 1, id="c_c_to_tp4"),
    pytest.param(Instance(4, 1, 1, 4, mla=True), REVIEWER_C, 1, id="c_draft_cp4_to_c"),
    pytest.param(REVIEWER_C, Instance(2, 1, 1, 2, mla=True), 1, id="c_c_to_draft_cp2"),
]
CASES = REVIEWER_CASES + [
    # (context, generation, draft KV heads)
    pytest.param(TP4, Instance(1, 4, 2, 2), 4, id="tp4_to_helix_t1x4_d2x2"),
    pytest.param(TP4, Instance(1, 4, 4, 1), 4, id="tp4_to_helix_t1x4_cp_replicated_draft"),
    pytest.param(TP4, Instance(1, 4, 4, 1), 2, id="tp4_to_helix_t1x4_cp_replicated_draft_heads"),
    pytest.param(Instance(4, 1, 1, 4), Instance(4, 1, 1, 4), 4, id="tp_target_cp_draft_both"),
    pytest.param(Instance(4, 1, 1, 4), Instance(2, 2, 4, 1), 4, id="cp_draft_to_t2x2_d4"),
    pytest.param(Instance(2, 2, 1, 4), TP4, 4, id="t2x2_d1x4_to_tp4"),
    pytest.param(TP4, Instance(1, 8, 4, 2), 4, id="tp4_to_t1x8_d4x2"),
    pytest.param(TP4, Instance(4, 2, 1, 8), 4, id="tp4_to_t4x2_d1x8"),
    # Matched layouts: unchanged behaviour.
    pytest.param(TP4, TP4, 4, id="matched_tp4"),
    pytest.param(Instance(2, 1, 2, 1), Instance(4, 1, 4, 1), 2, id="matched_tp2_to_tp4_replicated"),
    pytest.param(TP4, Instance(1, 4, 1, 4), 4, id="matched_helix"),
    pytest.param(TP4_MLA, Instance(1, 4, 1, 4, mla=True), 1, id="matched_helix_mla"),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("ctx", "gen", "draft_heads"), CASES)
def test_tagged_kv_reaches_each_owner_rank(ctx, gen, draft_heads, monkeypatch):
    run_transfer(ctx, gen, draft_heads, monkeypatch)


@pytest.mark.timeout(600)
def test_pairing_pages_by_position_is_detected(monkeypatch):
    # Negative control: pair two ranks' page tables entry by entry, as if both
    # models split pages alike; the check must see the wrong pages arrive.
    def by_position(src, dst, src_layout, dst_layout):
        count = min(src.size, dst.size)
        return src[:count], dst[:count]

    monkeypatch.setattr(transfer_mod, "pair_owned_pages", by_position)
    with pytest.raises(AssertionError, match="mismatch|astray"):
        run_transfer(TP4, Instance(1, 4, 2, 2), 4, monkeypatch)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("ctx", "gen", "draft_heads"),
    [(TP4, REVIEWER_A, 8), (TP4_MLA, REVIEWER_C, 1)],
    ids=["a", "c"],
)
def test_rank_wide_routing_is_detected(ctx, gen, draft_heads, monkeypatch):
    # Negative control: route the drafter's views by the rank-wide (target)
    # layout; either the page pairing refuses the mismatched tables or the
    # check sees wrong heads or pages arrive.
    monkeypatch.setattr(PeerRegistrar, "view_rank_infos", lambda self, peer_ri: {})
    with pytest.raises((AssertionError, ValueError), match="mismatch|astray|same prompt"):
        run_transfer(ctx, gen, draft_heads, monkeypatch)


@pytest.mark.parametrize(
    ("instance", "blocker"),
    [
        (Instance(4, 1, 1, 4), "a model with its own context parallelism"),
        (Instance(1, 4, 4, 1), r"context parallelism \(cp_size=4\)"),
        (TP4, None),
    ],
    ids=str,
)
def test_pipelined_transfer_considers_every_model(instance, blocker):
    """Pipelined pieces cannot be projected onto pages split over CP, whichever model splits them."""
    manager = make_manager(instance, 0, draft_heads=4)
    try:
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._mapping = mapping(instance, 0, TARGET)
        transceiver._kv_cache_manager = manager
        config = CacheTransceiverConfig(backend="NIXL", enable_pipelined_transfer=True)
        if blocker is None:
            assert transceiver._resolve_pipelined_transfer(config)
        else:
            with pytest.raises(ValueError, match=blocker):
                transceiver._resolve_pipelined_transfer(config)
    finally:
        manager.shutdown()
