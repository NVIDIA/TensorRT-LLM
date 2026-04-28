# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import threading
import uuid
from typing import Dict, List

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MixedMambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig

# ---------------------------------------------------------------------------
# Mamba parameters (under test)
# ---------------------------------------------------------------------------
NUM_MAMBA_LAYERS = 4
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_NUM_HEADS = 4
MAMBA_N_GROUPS = 1
MAMBA_HEAD_DIM = 64
MAX_BATCH_SIZE = 4
REQUEST_LENGTHS = [16, 32]

# Internal: layer 0 is a dummy attention layer required by page table infra;
# layers 1..NUM_MAMBA_LAYERS are mamba (under test).
_NUM_TOTAL_LAYERS = NUM_MAMBA_LAYERS + 1
_MAMBA_MASK = [False] + [True] * NUM_MAMBA_LAYERS
_ATTN_MASK = [True] + [False] * NUM_MAMBA_LAYERS


# ---------------------------------------------------------------------------
# ThreadSafeDistributed: barrier-based mock for single-process testing (PP=1)
# ---------------------------------------------------------------------------
class ThreadSafeDistributed:
    def __init__(self, rank, tp_size, shared):
        self.rank = rank
        self._world_size = tp_size
        self._tp_size = tp_size
        self._tp_rank = rank
        self._s = shared
        self._bcast_idx = 0
        self._ag_idx = 0
        self._pp_ag_idx = 0
        self._tp_ag_idx = 0

    @property
    def tp_size(self):
        return self._tp_size

    @property
    def pp_size(self):
        return 1

    @property
    def world_size(self):
        return self._world_size

    def broadcast(self, obj, root=0):
        idx = self._bcast_idx
        self._bcast_idx += 1
        key = f"bcast_{idx}"
        if self.rank == root:
            self._s[key] = obj
        self._s["barrier"].wait()
        result = self._s[key]
        self._s["barrier"].wait()
        return result

    def allgather(self, obj):
        idx = self._ag_idx
        self._ag_idx += 1
        key = f"ag_{idx}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._world_size
            self._s[key][self.rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def pp_allgather(self, obj):
        idx = self._pp_ag_idx
        self._pp_ag_idx += 1
        key = f"pp_ag_{idx}_tp{self._tp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None]
            self._s[key][0] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def tp_allgather(self, obj):
        idx = self._tp_ag_idx
        self._tp_ag_idx += 1
        key = f"tp_ag_{idx}_pp0"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._tp_size
            self._s[key][self._tp_rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------
def _run_concurrent(items, fn):
    errors = [None] * len(items)

    def _worker(idx, item):
        try:
            fn(item)
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=_worker, args=(i, it)) for i, it in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for err in errors:
        if err is not None:
            raise err


def _create_transceivers(tp, managers, config):
    shared = {"barrier": threading.Barrier(tp), "lock": threading.Lock()}
    results = [None] * tp
    errors = [None] * tp

    def _init(rank):
        try:
            mapping = Mapping(world_size=tp, rank=rank, tp_size=tp, pp_size=1)
            dist = ThreadSafeDistributed(rank, tp, shared)
            results[rank] = KvCacheTransceiverV2(
                mapping=mapping,
                dist=dist,
                kv_cache_manager=managers[rank],
                cache_transceiver_config=config,
            )
        except Exception as e:
            errors[rank] = e

    threads = [threading.Thread(target=_init, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for err in errors:
        if err is not None:
            raise err
    return results


def _create_managers(tp):
    """Create MixedMambaHybridCacheManagers for all TP ranks (PP=1).

    Layer 0 is a dummy attention layer required by page table infrastructure.
    Layers 1..NUM_MAMBA_LAYERS are mamba layers under test.
    """
    managers = []
    for rank in range(tp):
        mapping = Mapping(world_size=tp, rank=rank, tp_size=tp, pp_size=1)
        mgr = MixedMambaHybridCacheManager(
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_num_heads=MAMBA_NUM_HEADS,
            mamba_n_groups=MAMBA_N_GROUPS,
            mamba_head_dim=MAMBA_HEAD_DIM,
            mamba_num_layers=NUM_MAMBA_LAYERS,
            mamba_layer_mask=_MAMBA_MASK,
            mamba_cache_dtype=torch.float32,
            mamba_ssm_cache_dtype=torch.float32,
            # dummy attention layer (page table scaffolding)
            kv_cache_config=KvCacheConfig(
                max_tokens=256 * MAX_BATCH_SIZE,
                enable_block_reuse=False,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheTypeCpp.SELF,
            num_layers=1,
            layer_mask=_ATTN_MASK,
            num_kv_heads=4,
            head_dim=64,
            tokens_per_block=8,
            max_seq_len=256,
            max_batch_size=MAX_BATCH_SIZE,
            mapping=mapping,
            dtype=DataType.FLOAT,
        )
        managers.append(mgr)
    return managers


# ---------------------------------------------------------------------------
# Ground truth: generate, shard, write, compute expected, read actual
# ---------------------------------------------------------------------------
def _full_conv_section_dims() -> List[int]:
    """Full (unsharded) first-dim sizes: [x(d_inner) | B(ng*ds) | C(ng*ds)]."""
    d_inner = MAMBA_HEAD_DIM * MAMBA_NUM_HEADS
    ng_ds = MAMBA_N_GROUPS * MAMBA_D_STATE
    return [d_inner, ng_ds, ng_ds]


def _generate_ground_truth(num_requests: int, seed: int = 12345):
    """Generate full unsharded mamba states per (request, mamba_layer).

    Returns: [{global_layer_idx: {"conv": Tensor, "ssm": Tensor}}, ...]
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    full_conv_dim = sum(_full_conv_section_dims())
    results = []
    for _ in range(num_requests):
        layers = {}
        for i, is_mamba in enumerate(_MAMBA_MASK):
            if not is_mamba:
                continue
            layers[i] = {
                "conv": torch.rand(
                    full_conv_dim, MAMBA_D_CONV - 1, generator=gen, dtype=torch.float32
                ),
                "ssm": torch.rand(
                    MAMBA_NUM_HEADS,
                    MAMBA_HEAD_DIM,
                    MAMBA_D_STATE,
                    generator=gen,
                    dtype=torch.float32,
                ),
            }
        results.append(layers)
    return results


def _shard_ssm(full_ssm: torch.Tensor, tp: int, tp_rank: int) -> torch.Tensor:
    """Shard SSM along nheads (dim 0)."""
    n = full_ssm.shape[0] // tp
    return full_ssm[tp_rank * n : (tp_rank + 1) * n].clone()


def _shard_conv(full_conv: torch.Tensor, tp: int, tp_rank: int) -> torch.Tensor:
    """Shard conv per-section along dim 0: [x | B | C] each independently."""
    parts = []
    offset = 0
    for sec_dim in _full_conv_section_dims():
        n = sec_dim // tp
        parts.append(full_conv[offset + tp_rank * n : offset + (tp_rank + 1) * n])
        offset += sec_dim
    return torch.cat(parts, dim=0).clone()


def _write_ground_truth_to_ctx(managers, tp, ground_truth, request_ids):
    """Write sharded ground truth into ctx managers' allocated mamba slots."""
    for rank, mgr in enumerate(managers):
        for req_idx, rid in enumerate(request_ids):
            slot = mgr.mamba_cache_index[rid]
            for layer_idx in mgr._impl.mamba_layer_offsets:
                full = ground_truth[req_idx][layer_idx]
                mgr.get_ssm_states(layer_idx)[slot] = _shard_ssm(full["ssm"], tp, rank)
                mgr.get_conv_states(layer_idx)[slot] = _shard_conv(full["conv"], tp, rank)


def _compute_expected(ground_truth, gen_managers, gen_tp, gen_request_ids) -> Dict:
    """Compute expected mamba states BEFORE transfer.

    Returns: {(gen_rank, req_idx, layer_idx): {"conv": Tensor, "ssm": Tensor}}
    """
    expected = {}
    for gen_rank, mgr in enumerate(gen_managers):
        for req_idx in range(len(gen_request_ids)):
            for layer_idx in mgr._impl.mamba_layer_offsets:
                full = ground_truth[req_idx][layer_idx]
                expected[(gen_rank, req_idx, layer_idx)] = {
                    "ssm": _shard_ssm(full["ssm"], gen_tp, gen_rank),
                    "conv": _shard_conv(full["conv"], gen_tp, gen_rank),
                }
    return expected


def _read_actual(gen_managers, gen_request_ids) -> Dict:
    """Read actual mamba states AFTER transfer.

    Returns: {(gen_rank, req_idx, layer_idx): {"conv": Tensor, "ssm": Tensor}}
    """
    actual = {}
    for gen_rank, mgr in enumerate(gen_managers):
        for req_idx, rid in enumerate(gen_request_ids):
            slot = mgr.mamba_cache_index[rid]
            for layer_idx in mgr._impl.mamba_layer_offsets:
                actual[(gen_rank, req_idx, layer_idx)] = {
                    "conv": mgr.get_conv_states(layer_idx)[slot].cpu().clone(),
                    "ssm": mgr.get_ssm_states(layer_idx)[slot].cpu().clone(),
                }
    return actual


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------
def run_mamba_transfer_test(ctx_tp: int, gen_tp: int):
    """Test mamba transfer: ctx_tp -> gen_tp (PP=1, no DP)."""
    # -- 1. Create managers, zero mamba caches --
    ctx_mgrs = _create_managers(ctx_tp)
    gen_mgrs = _create_managers(gen_tp)
    for mgr in ctx_mgrs + gen_mgrs:
        mgr._impl.mamba_cache.conv.zero_()
        mgr._impl.mamba_cache.temporal.zero_()

    # -- 2. Create transceivers --
    config = CacheTransceiverConfig(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        max_tokens_in_buffer=512,
    )
    ctx_tcs = _create_transceivers(ctx_tp, ctx_mgrs, config)
    gen_tcs = _create_transceivers(gen_tp, gen_mgrs, config)
    ctx_endpoint = ctx_tcs[0]._context_info_endpoint

    # -- 3. Create requests --
    sampling_params = SamplingParams()
    ctx_rids, gen_rids = [], []
    ctx_reqs, gen_reqs = [], []

    for req_idx, req_len in enumerate(REQUEST_LENGTHS):
        unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
        ctx_rid, gen_rid = req_idx * 2, req_idx * 2 + 1
        ctx_rids.append(ctx_rid)
        gen_rids.append(gen_rid)

        sc = tensorrt_llm.bindings.SamplingConfig(sampling_params._get_sampling_config())
        ctx_req = LlmRequest(
            request_id=ctx_rid,
            max_new_tokens=1,
            input_tokens=list(range(req_len)),
            sampling_config=sc,
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
        )
        ctx_req.py_disaggregated_params = DisaggregatedParams(
            disagg_request_id=unique_rid,
        )
        gen_req = LlmRequest(
            request_id=gen_rid,
            max_new_tokens=1,
            input_tokens=list(range(req_len)),
            sampling_config=sc,
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        )
        gen_req.py_disaggregated_params = DisaggregatedParams(
            ctx_request_id=ctx_rid,
            ctx_dp_rank=0,
            ctx_info_endpoint=ctx_endpoint,
            disagg_request_id=unique_rid,
        )
        ctx_reqs.append(ctx_req)
        gen_reqs.append(gen_req)

    # -- 4. Allocate slots --
    ctx_batch = ScheduledRequests()
    ctx_batch.reset_context_requests(ctx_reqs)
    for mgr in ctx_mgrs:
        mgr.prepare_resources(ctx_batch)

    gen_batch = ScheduledRequests()
    gen_batch.reset_context_requests(gen_reqs)
    for mgr in gen_mgrs:
        mgr.prepare_resources(gen_batch)

    for req in ctx_reqs + gen_reqs:
        req.context_current_position = req.prompt_len
        req.add_new_token(req.prompt_len, 0)
    for mgr in ctx_mgrs:
        mgr.update_resources(ctx_batch)
    for mgr in gen_mgrs:
        mgr.update_resources(gen_batch)

    # -- 5. Ground truth -> shard -> write to ctx --
    ground_truth = _generate_ground_truth(len(REQUEST_LENGTHS))
    _write_ground_truth_to_ctx(ctx_mgrs, ctx_tp, ground_truth, ctx_rids)

    # -- 6. Compute expected BEFORE transfer --
    expected = _compute_expected(ground_truth, gen_mgrs, gen_tp, gen_rids)

    # -- 7. Transfer --
    for rank in range(gen_tp):
        for req in gen_reqs:
            gen_tcs[rank].request_and_receive_async(req)
    for rank in range(ctx_tp):
        for req in ctx_reqs:
            ctx_tcs[rank].respond_and_send_async(req)

    _run_concurrent(
        ctx_tcs,
        lambda tc: tc.check_context_transfer_status(None, mark_complete=True),
    )
    _run_concurrent(
        gen_tcs,
        lambda tc: tc.check_gen_transfer_status(None),
    )

    # -- 8. Read actual AFTER transfer --
    actual = _read_actual(gen_mgrs, gen_rids)

    # -- 9. Compare --
    for key in sorted(expected.keys()):
        gen_rank, req_idx, layer_idx = key
        for name in ["ssm", "conv"]:
            torch.testing.assert_close(
                actual[key][name],
                expected[key][name],
                rtol=0,
                atol=0,
                msg=lambda m, n=name, r=gen_rank, ri=req_idx, li=layer_idx: (
                    f"{n} mismatch: gen_rank={r} req={ri} layer={li} "
                    f"ctx_tp={ctx_tp} gen_tp={gen_tp}: {m}"
                ),
            )

    # -- 10. Cleanup --
    for mgr in ctx_mgrs + gen_mgrs:
        mgr.shutdown()
    for tc in ctx_tcs + gen_tcs:
        tc.shutdown()


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,gen_tp",
    [(4, 4), (2, 4), (4, 2)],
    ids=["ctx_tp4_gen_tp4", "ctx_tp2_gen_tp4", "ctx_tp4_gen_tp2"],
)
def test_mamba_transfer(ctx_tp, gen_tp):
    """Test mamba state transfer: ctx_tp -> gen_tp."""
    print(f"\nMamba transfer test: ctx_tp={ctx_tp} -> gen_tp={gen_tp}")
    run_mamba_transfer_test(ctx_tp, gen_tp)
    print("PASSED")
