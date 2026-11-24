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
"""Kimi K3.1: a generation server with extra receive slots (TRTLLM_DISAGG_GEN_EXTRA_SLOTS).

The family KDA + DSA hybrid cache manager of the Rubin targets, built as on the generation
servers (disaggregated, FP8 KV cache, FP8 index keys, FP32 KDA state, no block reuse), runs
under the V2 scheduler and the executor's sequence-slot manager, pass by pass in PyExecutor
order. Requests arrive, take their KV cache (gen-init), receive for a few passes, take a decode
seat and decode, as on a generation server. No model runs: a receive writes a request's own
recurrent state, and a decode step advances it, so the test can tell whose state each step reads.

Checked on real pools, with the option off and on: the decode batch and the sequence slots stay
within max_batch_size, the per-pass gen-init batch too; a request decodes from the state its
receive wrote and from no other request's; requests are seated in the order their KV landed and
then decode every pass to their end; every page, index slot and sequence slot is returned. On,
the receive overlaps the decode of the request it replaces, so the same requests finish in
fewer passes.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmRequestType,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
    GEN_EXTRA_SLOTS_ENV,
    KVCacheV2Scheduler,
)
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

# sm_107 only: l0_gb300 collects ops/ as a directory.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7),
    reason="Kimi K3.1 Rubin: sm_107",
)

TOKENS_PER_BLOCK = 32
# Layers 0 and 2 are KDA, layers 1 and 3 DSA MLA (1-indexed 2 and 4).
KDA_MASK = [True, False, True, False]
MLA_MASK = [False, True, False, True]
MAX_BATCH_SIZE = 2
PROMPT_LEN = 200
NUM_REQUESTS = 10

INIT = LlmRequestState.DISAGG_GENERATION_INIT
RECEIVING = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
RECEIVED = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
DECODING = LlmRequestState.GENERATION_IN_PROGRESS
COMPLETE = LlmRequestState.GENERATION_COMPLETE
RECEIVE_PHASE = (INIT, RECEIVING, RECEIVED)


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _gen_manager() -> RubinKdaDsaHybridCacheManager:
    """The generation servers' manager settings, at toy sizes."""
    gc.collect()
    torch.cuda.empty_cache()
    return RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=sum(KDA_MASK),
        mamba_layer_mask=KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        # A typical request well below the token budget, so the KDA pool group holds many
        # more state slots than max_batch_size, as on the generation servers.
        kv_cache_config=KvCacheConfig(max_tokens=8192, avg_seq_len=256, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=sum(MLA_MASK),
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=512,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_mask=MLA_MASK,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=_text_config(),
        is_disagg=True,
    )


def _request(request_id: int, max_new_tokens: int) -> LlmRequest:
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=[(7 * request_id + i) % 1000 for i in range(PROMPT_LEN)],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    req.state = INIT
    return req


class _GenServer:
    """PyExecutor's order for a generation server, without a model or a transceiver."""

    def __init__(self, manager, extra_slots: int, receive_passes: int, decode_steps: int):
        self.manager = manager
        self.scheduler = KVCacheV2Scheduler(
            max_batch_size=MAX_BATCH_SIZE,
            max_num_tokens=None,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            enable_recompute_pause=False,
        )
        assert self.scheduler.gen_extra_slots == extra_slots
        self.seq_slots = SeqSlotManager(MAX_BATCH_SIZE)
        self.capacity = MAX_BATCH_SIZE + extra_slots
        self.receive_passes = receive_passes
        self.decode_steps = decode_steps
        self.active: list = []
        self.countdown: dict = {}
        self.slot_of: dict = {}
        self.steps: dict = {}
        self.received_at: dict = {}
        self.seated_at: dict = {}
        self.batches: list = []
        self.max_seats = 0
        self.max_gen_init = 0
        self.max_active = 0

    # The KDA layers, stood in for: a receive writes the request's state, a step advances it.
    def _states(self):
        return list(zip(self.manager.all_ssm_states, self.manager.all_conv_states))

    def _fill(self, slot: int, value: float) -> None:
        for ssm, conv in self._states():
            ssm[slot].fill_(value)
            conv[slot].fill_(value)

    def _read(self, slot: int) -> list:
        return [
            (float(t[slot].float().min()), float(t[slot].float().max()))
            for pair in self._states()
            for t in pair
        ]

    def _expect(self, slot: int, value: float) -> None:
        got = self._read(slot)
        assert all(lo == value and hi == value for lo, hi in got), (slot, value, got)

    @staticmethod
    def _value(rid: int, step: int) -> float:
        # Exact in the BF16 conv windows: below 256 for 10 requests of 8 steps.
        return float(16 * rid + step)

    def step(self, t: int, waiting: list) -> None:
        manager = self.manager
        # 1. Admission (FIFO): max_batch_size plus the extra receive slots.
        while waiting and len(self.active) < self.capacity:
            self.active.append(waiting.pop(0))
        self.max_active = max(self.max_active, len(self.active))

        # 2. The scheduler pass.
        out = self.scheduler.schedule_request(list(self.active), set())
        gen_init = list(out.fitting_disagg_gen_init_requests)
        batch = list(out.generation_requests)
        assert not out.context_requests and not out.paused_requests
        assert len(batch) <= MAX_BATCH_SIZE
        self.max_gen_init = max(self.max_gen_init, len(gen_init))
        decoding_before = {r.py_request_id for r in self.active if r.state == DECODING}
        assert decoding_before <= {r.py_request_id for r in batch}, "a decoder lost its seat"
        self.batches.append([r.py_request_id for r in batch])

        # 3. Gen-init: the executor prepares them as one batch, then the receive writes each
        # request's state into its own slot.
        if gen_init:
            prep = ScheduledRequests()
            prep.context_requests_last_chunk = gen_init
            manager.prepare_resources(prep)
            torch.cuda.synchronize()
            for req in gen_init:
                rid = req.py_request_id
                slot = manager.get_state_indices([rid], [False])[0]
                self.slot_of[rid] = slot
                self._fill(slot, self._value(rid, 0))
                req.state = RECEIVING
                self.countdown[rid] = self.receive_passes

        # 4. Received requests in the batch take their seat (sequence slot), as in
        # _prepare_disagg_gen_transmission_complete.
        seated = [r for r in batch if r.state == RECEIVED]
        if seated:
            prep = ScheduledRequests()
            prep.context_requests_last_chunk = seated
            self.seq_slots.prepare_resources(prep)
            for req in seated:
                rid = req.py_request_id
                self._expect(self.slot_of[rid], self._value(rid, 0))
                req.state = DECODING
                req.context_current_position = req.prompt_len
                req.py_decoding_iter = 1
                req.add_new_token(rid, 0)
                self.seated_at[rid] = t
                self.steps[rid] = 0
        holders = [r for r in self.active if r.state not in RECEIVE_PHASE]
        self.max_seats = max(self.max_seats, len(holders))
        assert len(holders) <= MAX_BATCH_SIZE, "more sequence slots held than max_batch_size"

        # 5. The decode step: state rows of exactly the batch, each its own slot and state.
        if batch:
            step = ScheduledRequests()
            step.generation_requests = batch
            manager.prepare_resources(step)
            ids = [r.py_request_id for r in batch]
            slots = manager.get_state_indices(ids, [False] * len(ids))
            torch.cuda.synchronize()
            assert len(set(slots)) == len(slots)
            for req, slot in zip(batch, slots):
                rid = req.py_request_id
                # The state this step reads is the request's own: the receive's, then its
                # previous step's (wherever the manager keeps it).
                self.slot_of[rid] = slot
                self._expect(slot, self._value(rid, self.steps[rid]))
                self.steps[rid] += 1
                self._fill(slot, self._value(rid, self.steps[rid]))
                req.add_new_token(rid, 0)
                req.py_decoding_iter += 1
            manager.update_resources(step)

        # 6. Release the previous pass's finished requests, then finish this pass's.
        for req in [r for r in self.active if r.state == COMPLETE]:
            self.seq_slots.free_resources(req)
            manager.free_resources(req)
            self.active.remove(req)
        for req in batch:
            if self.steps[req.py_request_id] == self.decode_steps:
                req.state = COMPLETE

        # 7. Receives in flight land after their passes.
        for req in self.active:
            if req.state == RECEIVING:
                rid = req.py_request_id
                self.countdown[rid] -= 1
                if self.countdown[rid] == 0:
                    req.state = RECEIVED
                    self.received_at[rid] = t

    def run(self, requests: list, max_passes: int = 2000) -> int:
        waiting = list(requests)
        t = 0
        while (waiting or self.active) and t < max_passes:
            self.step(t, waiting)
            t += 1
        assert not waiting and not self.active, "requests left unfinished"
        return t


@pytest.mark.parametrize("receive_passes", [2, 6])
def test_gen_server_with_extra_receive_slots(monkeypatch, receive_passes):
    decode_steps = 8
    extra_slots = 2
    passes = {}
    for arm, extra in (("off", 0), ("on", extra_slots)):
        if extra:
            monkeypatch.setenv(GEN_EXTRA_SLOTS_ENV, str(extra))
        else:
            monkeypatch.delenv(GEN_EXTRA_SLOTS_ENV, raising=False)
        manager = _gen_manager()
        try:
            # The KV index slots and the KDA state slots cover the extra receive slots.
            assert manager.max_admissible_sequences >= MAX_BATCH_SIZE + extra_slots
            slot_capacity = manager._state_layout.slot_capacity
            dummy = manager._num_reserved_dummy_slots
            assert slot_capacity >= MAX_BATCH_SIZE + extra_slots + dummy, (slot_capacity, dummy)
            resident = set(manager.kv_cache_map)
            free_index_slots = manager.index_mapper.num_free_slots()

            server = _GenServer(manager, extra, receive_passes, decode_steps)
            requests = [_request(rid, decode_steps + 1) for rid in range(1, NUM_REQUESTS + 1)]
            passes[arm] = server.run(requests)
            print(
                f"[gen-extra-slots] arm={arm} receive_passes={receive_passes} "
                f"passes={passes[arm]} max_active={server.max_active} "
                f"max_seats={server.max_seats} max_gen_init={server.max_gen_init} "
                f"state_slots={slot_capacity} index_leases={manager.max_admissible_sequences}"
            )

            assert server.max_active == MAX_BATCH_SIZE + extra
            assert server.max_seats <= MAX_BATCH_SIZE
            assert server.max_gen_init <= MAX_BATCH_SIZE
            assert sorted(server.steps.values()) == [decode_steps] * NUM_REQUESTS
            # Seated in the order their KV landed; then a step in every pass to the end.
            landing = sorted(server.received_at, key=lambda r: (server.received_at[r], r))
            seating = sorted(server.seated_at, key=lambda r: (server.seated_at[r], r))
            assert seating == landing
            for rid, start in server.seated_at.items():
                ran = [t for t, ids in enumerate(server.batches) if rid in ids]
                assert ran == list(range(start, start + decode_steps)), rid
            # Everything given back.
            assert set(manager.kv_cache_map) == resident
            assert manager.index_mapper.num_free_slots() == free_index_slots
            assert len(server.seq_slots.slot_manager.free_slots) == MAX_BATCH_SIZE
        finally:
            manager.shutdown()
            del manager
            gc.collect()
            torch.cuda.empty_cache()
    assert passes["on"] < passes["off"], passes
