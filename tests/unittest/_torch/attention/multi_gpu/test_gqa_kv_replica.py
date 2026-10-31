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
from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import _kv_replica_exchange_sizes
from tensorrt_llm._torch.attention.backends.fmha.gqa_kv_replica import (
    kv_replica_decode,
    kv_replica_fused_decode,
)
from tensorrt_llm._torch.attention.backends.fmha.gqa_kv_replica_fused import (
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


def _kv_replica(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    MPI.COMM_WORLD.barrier()
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    max_bytes, max_merge_rows = _kv_replica_exchange_sizes(16, 7, HQ, D)
    ex = TPGroupExchange.create(mapping, tp, max_bytes, max_merge_rows)
    assert ex is not None, "no peer memory for the exchange"
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(128 * 128, dtype=torch.int32, device="cuda")
    try:
        for lens, q_len in (([5000, 70, 1300], 7), ([9000, 64, 130, 4100, 70, 12000], 1)):
            kv, tables, seq, q, gate = _inputs(lens, q_len, tp, rank)
            ref = _reference(kv, tables, lens, q, gate, q_len)
            fused = kv_replica_fused_supported(q_len, tp, HQ, len(lens), sms)
            for enable_pdl in (False, True):
                out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
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
                outs = {"segmented": out}
                if fused:
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
                torch.cuda.synchronize()
                for name, out in outs.items():
                    # FP8 P / partials: a few percent of the row max per element
                    err = (out.float() - ref.float()).abs().amax(dim=-1)
                    scale = ref.float().abs().amax(dim=-1).clamp_min(1e-3)
                    assert (err / scale).amax().item() < 0.08, (name, lens, q_len, enable_pdl)
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
