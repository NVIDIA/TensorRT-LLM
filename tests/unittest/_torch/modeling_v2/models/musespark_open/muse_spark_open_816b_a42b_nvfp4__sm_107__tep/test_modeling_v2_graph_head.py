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
"""The decode graph's head on multi-request steps: the overlap-offsets fold
(``step_commit_v2__OVERLAP_FOLD``) and the input boundary's entry trigger
(``hop_entry_trigger__INPUT_BOUNDARY``).

* The fold, captured in a CUDA graph at the multi-request views (16..128 generation rows,
  2..16 requests), writes the two built-in adds' words, int32 wrap included.
* The input boundary with the entry trigger writes the exit trigger's bytes, eager and with a
  PDL-launched split-K GEMM behind it in a captured graph (the layer-0 QKV shape at 48 rows).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires an sm107 device",
)

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)


def _mod(name):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


@pytest.mark.parametrize("fold", [1, 2])
@pytest.mark.parametrize("requests", [2, 6, 16])
def test_overlap_fold_writes_the_builtin_adds_in_a_graph(fold, requests):
    ia = _mod("ops.runtime.overlap_offsets")
    applier = ia.build_applier(fold)
    rows = 8 * requests
    gen = torch.Generator(device="cuda").manual_seed(requests)
    pos0 = torch.randint(-(2**31), 2**31 - 1, (1, rows), dtype=torch.int32, device="cuda", generator=gen)
    pos0[0, 0] = 2**31 - 1
    pos_off = torch.randint(-50, 50, (rows,), dtype=torch.int32, device="cuda", generator=gen)
    pos_off[0] = 7
    kv0 = torch.randint(0, 2**30, (requests,), dtype=torch.int32, device="cuda", generator=gen)
    kv_off = torch.randint(-8, 8, (requests,), dtype=torch.int32, device="cuda", generator=gen)
    want_pos = (pos0[0].long() + pos_off.long()).to(torch.int64)
    want_pos = ((want_pos + 2**31) % 2**32 - 2**31).to(torch.int32)
    want_kv = kv0 + kv_off
    pos, kv = pos0.clone(), kv0.clone()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            assert applier.apply(pos[0, :], pos_off, kv, kv_off)
    pos.copy_(pos0)
    kv.copy_(kv0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(pos[0], want_pos)
    assert torch.equal(kv, want_kv)


@pytest.mark.parametrize("rows", [16, 48, 96, 128])
def test_input_boundary_entry_trigger_writes_the_exit_bytes(rows):
    ib = _mod("kernels.model_boundary.input_boundary")
    if ib.available() is not None:
        pytest.skip(ib.available())
    gen = torch.Generator(device="cuda").manual_seed(rows)
    table = torch.randn(2048, ib.HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, 2048, (rows,), dtype=torch.int32, device="cuda", generator=gen)
    w_e = torch.ones(ib.HIDDEN, dtype=torch.bfloat16, device="cuda")
    w_q = (torch.randn(ib.HIDDEN, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    assert ib.takes(table, ids, w_e, w_q)
    exit_out = ib.input_boundary(table, ids, w_e, 1e-6, w_q, 1e-6, trigger="exit")
    entry_out = ib.input_boundary(table, ids, w_e, 1e-6, w_q, 1e-6, trigger="entry")
    torch.cuda.synchronize()
    assert torch.equal(exit_out[0].view(torch.int32), entry_out[0].view(torch.int32))
    assert torch.equal(exit_out[1].view(torch.int16), entry_out[1].view(torch.int16))

    # In a captured graph, with a PDL-launched GEMM reading the boundary's output right behind
    # it (the layer-0 QKV shape), replayed several times: the GEMM sees the boundary's output.
    w = (torch.randn(2304, ib.HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    want = exit_out[1] @ w.t()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids_static = ids.clone()
        for _ in range(2):  # warm up (and compile) outside the capture
            out = ib.input_boundary(table, ids_static, w_e, 1e-6, w_q, 1e-6, trigger="entry")[1]
            got = out @ w.t()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = ib.input_boundary(table, ids_static, w_e, 1e-6, w_q, 1e-6, trigger="entry")[1]
            got = out @ w.t()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
