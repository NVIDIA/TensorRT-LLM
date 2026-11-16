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
"""``packed_prep_append`` at TP8 head counts: the packed kernel writes the grid kernel's bytes.

The TP8 rank's decode prep append carries 16 query heads and 2 KV heads per row (an 18-item
row of the (row, head) grid). Every output of a call on the packed kernel -- the normalized
projection and gate, the E4M3 query, every pool byte, the FMHA scale words and the cumulative
query lengths -- must equal the one-program-per-(row, head) grid kernel's, with and without the
o_proj L2 prefetch, at the composite's step sizes. Also timed (printed, not asserted).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires an sm107 device",
)

H, G, D, PAGE = 16, 2, 64, 32


def _inputs(seqs: int, rows_per_seq: int, seed: int, permuted: bool):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = seqs * rows_per_seq
    qkv = torch.randn(rows, 2 * (H + G) * D, device="cuda", generator=gen).to(torch.bfloat16)
    qkv[0, :8] = torch.tensor([0.0, -0.0, 1e30, -1e30, 3e-39, 448.0, 1e-3, 7.0])
    pages_per_seq = 16
    num_pages = seqs * pages_per_seq * 2 + 3
    lens = torch.randint(
        rows_per_seq, pages_per_seq * PAGE, (seqs,), device="cuda", generator=gen
    ).to(torch.int32)
    perm = torch.randperm(num_pages, device="cuda", generator=gen)[: seqs * 2 * pages_per_seq]
    tables = perm.view(seqs, 2, pages_per_seq).to(torch.int32).contiguous()
    pool = torch.randint(0, 255, (num_pages, G, PAGE, D), dtype=torch.uint8, device="cuda", generator=gen)
    cos_sin = torch.rand(1024, 2, D // 2, device="cuda", generator=gen)
    positions = (lens.long()[:, None] - rows_per_seq + torch.arange(rows_per_seq, device="cuda")).reshape(-1)
    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        position_ids=positions.to(torch.int32).view(1, -1),
        num_q_heads=H,
        num_kv_heads=G,
        eps=1e-6,
        q_scale=0.125,
        qkv_permuted=permuted,
        num_seqs=seqs,
        kv_pool=pool.view(torch.float8_e4m3fn),
        block_tables=tables,
        seq_lens=lens,
        kv_scale_orig_quant=torch.tensor([0.75], device="cuda"),
        kv_scale_quant_orig=torch.tensor([1.0 / 0.75], device="cuda"),
        q_fp8=torch.full((rows * H * D,), 0x5A, dtype=torch.uint8, device="cuda"),
        bmm1_scale=torch.zeros(2, device="cuda"),
        bmm2_scale=torch.zeros(1, device="cuda"),
        cu_seqlens=torch.full((seqs + 1,), -1, dtype=torch.int32, device="cuda"),
        bmm1_host_scale=0.125,
    )


def _run(inputs, **kw):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention.multi_request_prep_append import (  # noqa: E501
        multi_request_prep_append,
    )

    mine = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    qkv_out, gate_out = multi_request_prep_append(**mine, **kw)
    torch.cuda.synchronize()
    return mine, qkv_out, gate_out


@pytest.mark.parametrize("seqs", [5, 6, 8, 12, 16])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("with_prefetch", [False, True])
def test_packed_writes_the_grid_bytes_at_tep8(seqs, permuted, with_prefetch):
    inputs = _inputs(seqs, 8, seed=seqs * 7 + permuted, permuted=permuted)
    prefetch = (
        torch.empty(8192 * 1024, dtype=torch.bfloat16, device="cuda") if with_prefetch else None
    )
    grid, g_qkv, g_gate = _run(inputs, prefetch=prefetch)
    packed, p_qkv, p_gate = _run(inputs, prefetch=prefetch, packed_min_rows=33)
    assert torch.equal(g_qkv.view(torch.int16), p_qkv.view(torch.int16))
    assert torch.equal(g_gate.view(torch.int16), p_gate.view(torch.int16))
    for name in ("kv_pool", "q_fp8", "bmm1_scale", "bmm2_scale", "cu_seqlens"):
        a, b = grid[name], packed[name]
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


@pytest.mark.parametrize("seqs", [6, 12, 16])
def test_packed_prep_timing_at_tep8(seqs):
    """Graph-replayed time per call, grid vs packed (printed: [PACKED_PREP] lines)."""
    inputs = _inputs(seqs, 8, seed=1, permuted=False)
    times = {}
    for label, kw in (("grid", {}), ("packed", {"packed_min_rows": 33})):
        mine = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        _run(mine, **kw)  # build
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention.multi_request_prep_append import (  # noqa: E501
            multi_request_prep_append,
        )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            multi_request_prep_append(**mine, **kw)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                multi_request_prep_append(**mine, **kw)
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        best = []
        for _ in range(20):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            best.append(start.elapsed_time(end) * 1000.0 / 20)
        times[label] = sorted(best)[len(best) // 2]
    print(
        f"[PACKED_PREP] tep8 seqs={seqs} rows={seqs * 8}: grid {times['grid']:.2f} us, "
        f"packed {times['packed']:.2f} us per call (graph of 20, median of 20)"
    )
