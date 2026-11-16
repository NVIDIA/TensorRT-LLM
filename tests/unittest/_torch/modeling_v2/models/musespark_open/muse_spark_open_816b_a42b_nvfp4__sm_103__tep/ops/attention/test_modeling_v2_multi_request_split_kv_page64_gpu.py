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
"""GPU test of ``multi_request_split_kv`` over a cache of 64-token blocks (sm_103).

For each case a random FP8 pool of 64-token blocks with its K and V block tables, and the pool of
32-token blocks and tables that hold the same keys (block ``2 B + s`` is half ``s`` of block
``B``): the 32-token cache is the path the target serves without
``MUSESPARK_OPEN_MULTI_REQUEST_SPLIT_KV_PAGE64``. The tests pin that

* the op on the 64-token pool equals the op on the 32-token pool bit for bit, and so do the two
  host references;
* the selection site declines the 64-token pool with the variable unset and launches the same
  output with it set;
* a CUDA graph captured on the 64-token pool replays, at other sequence lengths, the output of the
  eager 32-token call.

Cache slots past a sequence's length hold E4M3 NaN bytes, so a read past it shows up as NaN.
"""

import importlib

import pytest
import torch

PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open"
    ".muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
ENV = "MUSESPARK_OPEN_MULTI_REQUEST_SPLIT_KV_PAGE64"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3),
    reason="requires an sm_103 GPU",
)


@pytest.fixture(scope="module")
def mr():
    return importlib.import_module(PKG + ".ops.attention.multi_request_split_kv")


@pytest.fixture(scope="module")
def site():
    return importlib.import_module(PKG + ".kernels.multi_request_split_kv.attention")


def _case(kv_heads, lengths, rows, seed=0):
    """The 64-token pool and tables, and the 32-token pool and tables holding the same keys."""
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(lengths)
    blocks = [-(-length // 64) for length in lengths]
    width = max(blocks) + 1
    count = 2 * sum(blocks) + 7
    pool64 = (torch.randn(count, kv_heads, 64, 64, device=device, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    nan = torch.tensor(0x7F, dtype=torch.uint8, device=device)
    perm = torch.randperm(count, device=device, generator=gen)
    table64 = torch.zeros(batch, 2, width, dtype=torch.int32, device=device)
    raw, used = pool64.view(torch.uint8), 0
    for b, (length, n) in enumerate(zip(lengths, blocks)):
        for kv in range(2):
            ids = perm[used : used + n]
            used += n
            table64[b, kv, :n] = ids.to(torch.int32)
            table64[b, kv, n:] = int(perm[-1])  # a valid id the kernel must not read
            raw[ids[-1], :, length - (n - 1) * 64 :] = nan  # unwritten slots past the length
    pool32 = (
        pool64.view(count, kv_heads, 2, 32, 64)
        .permute(0, 2, 1, 3, 4)
        .reshape(2 * count, kv_heads, 32, 64)
        .contiguous()
    )
    halves = torch.arange(2 * width, device=device)
    table32 = (2 * table64[:, :, halves // 2].long() + halves % 2).to(torch.int32).contiguous()
    return dict(
        pool64=pool64,
        table64=table64,
        pool32=pool32,
        table32=table32,
        seq=torch.tensor(lengths, dtype=torch.int32, device=device),
        q=(torch.randn(batch * rows, 8 * kv_heads * 64, device=device, generator=gen) * 2).to(
            torch.float8_e4m3fn
        ),
        bmm1=torch.tensor([0.125], dtype=torch.float32, device=device),
        bmm2=torch.tensor([1.0], dtype=torch.float32, device=device),
        rows=rows,
        batch=batch,
        kv_heads=kv_heads,
    )


def _run(mr, case, which, out=None):
    if out is None:
        out = torch.empty(
            case["batch"] * case["rows"],
            8 * case["kv_heads"] * 64,
            dtype=torch.bfloat16,
            device="cuda",
        )
    mr.multi_request_split_kv(
        case["q"],
        case["pool" + which],
        case["table" + which],
        case["seq"],
        case["bmm1"],
        case["bmm2"],
        out,
        enable_pdl=True,
    )
    return out


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize(
    "kv_heads, lengths, rows",
    [
        (4, [33, 72], 8),
        (4, [64, 65, 96, 97, 128, 129], 3),
        (4, [9, 4000, 64 * 37 + 31, 2048, 1], 1),
        (2, [8, 513, 2049, 4000, 6001, 8191, 9000, 100], 8),
    ],
)
def test_the_64_token_pool_gives_the_32_token_output(mr, kv_heads, lengths, rows):
    case = _case(kv_heads, lengths, rows)
    out64, out32 = _run(mr, case, "64"), _run(mr, case, "32")
    torch.cuda.synchronize()
    assert torch.isfinite(out64.float()).all()
    assert _bitwise(out64, out32)
    ref = [
        mr.multi_request_split_kv.reference(
            case["q"],
            case["pool" + w],
            case["table" + w],
            case["seq"],
            case["bmm1"],
            case["bmm2"],
            None,
        )
        for w in ("64", "32")
    ]
    assert torch.equal(*ref)


def test_the_selection_site_takes_the_64_token_pool_only_with_the_gate(mr, site, monkeypatch):
    case = _case(4, [500, 7000, 64 * 3], 8)
    out = torch.empty(case["batch"] * 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    kwargs = dict(
        query=case["q"].view(-1, 32, 64),
        kv_cache=(case["pool64"], case["pool64"]),
        workspace_buffer=None,
        block_tables=case["table64"],
        seq_lens=case["seq"],
        max_seq_len=7000,
        bmm1_scale=case["bmm1"],
        bmm2_scale=case["bmm2"],
        window_left=-1,
        out=out,
        sinks=None,
        kv_layout="HND",
        enable_pdl=True,
        backend="trtllm-gen",
        q_len_per_req=8,
        max_q_len=None,
        cum_seq_lens_q=None,
        kv_cache_sf=None,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=None,
        multi_ctas_kv_counter_buffer=None,
    )
    assert set(kwargs) == site.DECODE_KWARGS
    monkeypatch.delenv(ENV, raising=False)
    assert site.MultiRequestCall(kv_heads=4, window=0).decline(kwargs) == "another cache view"
    same_keys = dict(
        kwargs, kv_cache=(case["pool32"], case["pool32"]), block_tables=case["table32"]
    )
    assert site.MultiRequestCall(kv_heads=4, window=0).decline(same_keys) is None
    monkeypatch.setenv(ENV, "1")
    call = site.MultiRequestCall(kv_heads=4, window=0)
    assert call.decline(kwargs) is None
    call.launch(kwargs)
    assert _bitwise(out.view(case["batch"] * 8, -1), _run(mr, case, "32"))


def test_a_cuda_graph_on_the_64_token_pool_replays_at_other_lengths(mr, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    case = _case(4, [3000, 5000, 6000, 7000], 8)
    mr.precompile(0, True, 4, 0, q_fp8=True, device_scale=True, cache_page=64)
    out = torch.empty(32, 2048, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(mr, case, "64", out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(mr, case, "64", out)
    for lengths in ([3000, 5000, 6000, 7000], [10, 4999, 64, 6913], [2900, 8, 129, 6400]):
        case["seq"].copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise(out, _run(mr, case, "32"))
