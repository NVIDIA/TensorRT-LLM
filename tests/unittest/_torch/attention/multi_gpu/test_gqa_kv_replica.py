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

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import gqa_kv_replica
from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import _kv_replica_exchange_sizes
from tensorrt_llm._torch.attention.backends.fmha.gqa_kv_replica import (
    kv_replica_decode,
    kv_replica_fused_decode,
)
from tensorrt_llm._torch.attention.backends.fmha.gqa_kv_replica_fused import (
    _plan,
    kv_replica_fused_state,
    kv_replica_fused_supported,
)
from tensorrt_llm._torch.distributed.ll_exchange import TPGroupExchange
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="the trtllm-gen / CuTe DSL decode kernels need an SM100 / SM103 GPU",
    ),
]

HQ, D, PAGE = 4, 256, 64
BMM1, BMM2 = 1.0 / 16, 1.0
_OUT_SCALE = 0.37
# Largest batch the test runs on the segmented path: its two partials per row fill the merge channel and its counters
# at larger batches, where GqaKvReplicaFmha._takes declines (the push-channel cases only run the fused path).
_SEGMENTED_MAX_REQUESTS = 16


def _inputs(lens, q_len, tp, rank):
    """KV pool / tables / lengths shared by the group (same seed) and this rank's q / gate."""
    b = len(lens)
    g = torch.Generator(device="cuda").manual_seed(7)
    max_pages = max((max(lens) + PAGE - 1) // PAGE, tp)
    nblocks = b * 2 * max_pages + 8
    kv = (torch.randn(nblocks, 1, PAGE, D, device="cuda", generator=g) * 0.6).to(
        torch.float8_e4m3fn
    )
    tables = torch.randperm(nblocks, device="cuda", generator=g)[: b * 2 * max_pages]
    tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    q = torch.randn(b * q_len, HQ, D, device="cuda", generator=g).to(torch.float8_e4m3fn)
    gate = torch.randn(b * q_len, HQ * D, device="cuda", generator=g).to(torch.bfloat16)
    return kv, tables, seq, q, gate


def _reference(kv, tables, lens, q, gate, q_len):
    out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    pool = kv.float()[:, 0]
    for r, L in enumerate(lens):
        pos = torch.arange(L, device="cuda")
        k = pool[tables[r, 0, pos // PAGE].long(), pos % PAGE]
        v = pool[tables[r, 1, pos // PAGE].long(), pos % PAGE]
        for t in range(q_len):
            row = r * q_len + t
            visible = L - q_len + t + 1
            s = q[row].float() @ k[:visible].T * BMM1
            out[row] = torch.softmax(s, dim=-1) @ v[:visible] * BMM2
    gated = out.to(torch.bfloat16).float() * torch.sigmoid(gate.float().view(out.shape))
    return gated.to(torch.bfloat16)


def _decode_all(ex, q, kv, tables, seq, lens, q_len, sms, gate, enable_pdl, workspace, counter):
    """The outputs of every KV-replica decode path on exchange ``ex``: segmented (up to _SEGMENTED_MAX_REQUESTS
    requests), fused (BF16) and fused (E4M3, the merge writing o_proj's input), the last two where the fused partial
    kernel takes the shape."""
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    outs = {}
    if len(lens) <= _SEGMENTED_MAX_REQUESTS:
        kv_replica_decode(
            ex,
            q,
            kv,
            tables,
            seq,
            max(lens),
            q_len,
            2,
            BMM1,
            BMM2,
            workspace,
            counter,
            out,
            enable_pdl,
            gate=gate,
        )
        outs["segmented"] = out
    if kv_replica_fused_supported(q_len, ex.group_size, HQ, len(lens), sms):
        out = torch.empty_like(out)
        kv_replica_fused_decode(
            ex,
            q,
            kv,
            tables,
            seq,
            q_len,
            BMM1,
            BMM2,
            out,
            kv_replica_fused_state(q.device, sms, len(seq)),
            enable_pdl,
            gate=gate,
        )
        outs["fused"] = out
        # The same call with freshly allocated (O, LSE) partial outputs instead of the pushed-mode placeholders:
        # the push channel never writes them, so the result is bit-identical.
        fresh = torch.empty_like(out)
        real_placeholders = gqa_kv_replica._pushed_partial_placeholders
        gqa_kv_replica._pushed_partial_placeholders = lambda cp, rows, hq, d, dtype, device: (
            torch.empty(cp, rows, hq, d, dtype=dtype, device=device),
            torch.empty(cp, rows, hq, dtype=torch.float32, device=device),
        )
        try:
            kv_replica_fused_decode(
                ex,
                q,
                kv,
                tables,
                seq,
                q_len,
                BMM1,
                BMM2,
                fresh,
                kv_replica_fused_state(q.device, sms, len(seq)),
                enable_pdl,
                gate=gate,
            )
        finally:
            gqa_kv_replica._pushed_partial_placeholders = real_placeholders
        outs["fused_fresh"] = fresh
        out_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device="cuda")
        kv_replica_fused_decode(
            ex,
            q,
            kv,
            tables,
            seq,
            q_len,
            BMM1,
            BMM2,
            torch.empty_like(out),
            kv_replica_fused_state(q.device, sms, len(seq)),
            enable_pdl,
            gate=gate,
            out_fp8=out_fp8,
            out_scale=torch.tensor([_OUT_SCALE], dtype=torch.float32, device="cuda"),
        )
        outs["fused_fp8"] = out_fp8
    torch.cuda.synchronize()
    return outs


def _kv_replica(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    MPI.COMM_WORLD.barrier()
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    max_bytes, max_merge_rows = _kv_replica_exchange_sizes(128, 7, HQ, D)
    ex = TPGroupExchange.create(mapping, tp, max_bytes, max_merge_rows)
    assert ex is not None, "no peer memory for the exchange"
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(128 * 128, dtype=torch.int32, device="cuda")
    try:
        plans = set()
        for lens, q_len in (
            ([5000, 70, 1300], 7),
            ([9000, 64, 130, 4100, 70, 12000], 1),
            # The push channel with the wide (33-64 requests) and two-slot (above 64) plans.
            ([64 + (i * 97) % 1200 for i in range(40)], 7),
            ([7 + (i * 131) % 900 for i in range(100)], 7),
        ):
            if kv_replica_fused_supported(q_len, tp, HQ, len(lens), sms):
                plans.add(_plan(len(lens), sms))
            kv, tables, seq, q, gate = _inputs(lens, q_len, tp, rank)
            ref = _reference(kv, tables, lens, q, gate, q_len)
            for enable_pdl in (False, True):
                args = (q, kv, tables, seq, lens, q_len, sms, gate, enable_pdl, workspace, counter)
                outs = _decode_all(ex, *args)
                if "fused_fp8" in outs:
                    # The merge writing o_proj's E4M3 input: the bytes of the BF16 result's static quant.
                    # static_quantize_e4m3_per_tensor: FP32(BF16) x (1 / s), round to nearest.
                    out_scale = torch.tensor([_OUT_SCALE], dtype=torch.float32, device="cuda")
                    expected_fp8 = (outs["fused"].float() * (1.0 / out_scale)).to(
                        torch.float8_e4m3fn
                    )
                    assert torch.equal(
                        outs.pop("fused_fp8").view(torch.uint8), expected_fp8.view(torch.uint8)
                    ), (lens, q_len, enable_pdl)
                if "fused_fresh" in outs:
                    assert torch.equal(
                        outs.pop("fused_fresh").view(torch.uint8), outs["fused"].view(torch.uint8)
                    ), (lens, q_len, enable_pdl)
                for name, out in outs.items():
                    # FP8 P / partials: a few percent of the row max per element
                    err = (out.float() - ref.float()).abs().amax(dim=-1)
                    scale = ref.float().abs().amax(dim=-1).clamp_min(1e-3)
                    assert (err / scale).amax().item() < 0.08, (name, lens, q_len, enable_pdl)
        assert {"lane", "wide", "two"} <= plans, plans
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _worker(tp):
    try:
        with torch.inference_mode():
            _kv_replica(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_kv_replica_decode_matches_reference(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [tp] * tp)
    assert all(results)


class _StubExchange:
    """The two TPGroupExchange calls kv_replica_fused_decode makes with a gathered q, recorded."""

    group_size, grank = 2, 0

    def __init__(self):
        self.merged = []

    def merge_push_args(self, channel, rows, d):
        return ("ll", channel, rows, d)

    def alltoall_merge(self, o, lse, out, **kwargs):
        self.merged.append((o, lse, kwargs["pushed"]))


@pytest.mark.parametrize("rows_seq", [(14, 14, 7), (21, 7, 14)])
def test_pushed_partials_reuse_placeholder_storage(monkeypatch, rows_seq):
    """With the push channel the partial kernel never writes its (O, LSE) outputs and the merge never reads them:
    every eager call up to the largest row count seen gets views of the same storage (no two allocations per layer
    and step), with the shapes and strides of fresh tensors."""
    partials = []
    monkeypatch.setattr(
        gqa_kv_replica,
        "kv_replica_fused_partial",
        lambda q_all,
        kv,
        tables,
        seq,
        q_len,
        cp_rank,
        cp,
        b1,
        b2,
        o,
        lse,
        state,
        **kw: partials.append((o, lse, kw["ll"])),
    )
    ex = _StubExchange()
    hq, d = 4, 256
    for rows in rows_seq:
        q = torch.zeros(rows, hq, d, dtype=torch.float8_e4m3fn, device="cuda")
        out = torch.empty(rows, hq, d, dtype=torch.bfloat16, device="cuda")
        q_gathered = torch.zeros(rows, 2 * hq * d, dtype=torch.float8_e4m3fn, device="cuda")
        kv_replica_fused_decode(
            ex, q, None, None, None, 7, BMM1, BMM2, out, None, False, q_gathered=q_gathered
        )
    storages = {o.untyped_storage().data_ptr() for o, _, _ in partials}
    lse_storages = {lse.untyped_storage().data_ptr() for _, lse, _ in partials}
    # Kept alive by the recorder, fresh allocations would all differ.
    assert len(storages) == 1 and len(lse_storages) == 1
    for (o, lse, ll), rows in zip(partials, rows_seq):
        ref_o = torch.empty(2, rows, hq, d, dtype=torch.bfloat16, device="cuda")
        ref_lse = torch.empty(2, rows, hq, dtype=torch.float32, device="cuda")
        assert o.shape == ref_o.shape and o.stride() == ref_o.stride() and o.dtype == ref_o.dtype
        assert lse.shape == ref_lse.shape and lse.stride() == ref_lse.stride()
        assert o.data_ptr() % 16 == 0 and lse.data_ptr() % 16 == 0
        assert ll == ("ll", 1, rows * hq, d)
    assert [m[0] is p[0] and m[2] for m, p in zip(ex.merged, partials)] == [True] * len(rows_seq)
