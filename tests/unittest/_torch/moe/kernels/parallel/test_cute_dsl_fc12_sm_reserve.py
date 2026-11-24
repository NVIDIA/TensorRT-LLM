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
"""``TRTLLM_K3_FC12_SM_RESERVE`` on the fused FC12 MoE kernel (Rubin, SM107).

The reserve launches FC12's persistent grid with fewer CTAs so kernels on other streams find
free SMs while it runs. FC12 hands its tiles out through atomic counters, so only the CTA that
computes a tile changes: outputs stay bit-identical wherever FC12 reproduces itself (rows with at
most two local experts), down to a grid of a single CTA.
"""

import pytest
import torch
from _torch.moe.test_cute_dsl_fc12_presorted import (
    TILE,
    _assert_same_rows,
    _k3_module,
    _local_counts,
    _logits,
    _noaux_route,
    _quantized_input,
    _tail_route,
    rubin_fc12,
)
from _torch.moe.test_cute_dsl_fc12_prezero import _few_local_routing, _rank_backend, _zero_beside

from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

ENV = "TRTLLM_K3_FC12_SM_RESERVE"


@pytest.mark.parametrize(
    "reserve,cluster_size,expected",
    [(None, 1, 212), ("0", 2, 106), ("8", 1, 204), ("8", 2, 102), ("9", 2, 101), ("1000", 1, 1)],
)
def test_fc12_max_active_clusters(monkeypatch, reserve, cluster_size, expected):
    """Unset or 0: the grid as before; else whole clusters fewer, at least one."""
    if reserve is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, reserve)
    assert ops.fc12_max_active_clusters(212 // cluster_size, cluster_size) == expected


def test_fc12_sm_reserve_rejects_negative(monkeypatch):
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValueError, match=ENV):
        ops.fc12_max_active_clusters(212, 1)


def _launched_grids():
    """Persistent cluster counts FC12 kernels were compiled for so far."""
    return {key[6] for key in ops.Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner.kernel_cache}


@rubin_fc12
@pytest.mark.parametrize("reserve", [8, 100, "all_but_one"])
@pytest.mark.parametrize("offset,num_local", [(3 * 112, 112), (7 * 56, 56)])
@pytest.mark.parametrize("m", [8, 64])
def test_fc12_sm_reserve_bit_identical(monkeypatch, m, offset, num_local, reserve):
    """A smaller FC12 grid gives the full grid's output, bit for bit."""
    backend = _rank_backend(offset, num_local, seed=offset + m)
    x, x_sf = _quantized_input(m, seed=m + 3)
    ids, weights = _few_local_routing(m, offset, num_local, seed=m + 4)

    monkeypatch.delenv(ENV, raising=False)
    expected = backend.run(x, x_sf, ids, weights)
    torch.cuda.synchronize()
    full = ops.get_max_activate_clusters(1)
    if reserve == "all_but_one":
        reserve = full - 1
    monkeypatch.setenv(ENV, str(reserve))
    actual = backend.run(x, x_sf, ids, weights)
    torch.cuda.synchronize()
    assert max(1, full - reserve) in _launched_grids()
    assert expected.abs().amax() > 0, "degenerate test: all-zero MoE output"
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@rubin_fc12
def test_fc12_sm_reserve_with_prezero_cuda_graph_replay(monkeypatch):
    """Both FC12 switches in one CUDA graph (zeroing stream, routing tail, reserved FC12 grid):
    replays follow new logits and equal the eager path without either switch."""
    k3_route_tail = _k3_module("kernels.k3_route_tail")

    assert k3_route_tail.supports_device(torch.device("cuda"))
    m, offset, num_local = 16, 3 * 112, 112
    backend = _rank_backend(offset, num_local, seed=21)
    logits, bias = _logits(m, seed=210)
    x, x_sf = _quantized_input(m, seed=211)
    static_logits = logits.clone()
    side = torch.cuda.Stream()

    def step():
        _zero_beside(backend, m, side)
        return backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))

    monkeypatch.setenv(ENV, "16")
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    assert ops.get_max_activate_clusters(1) - 16 in _launched_grids()

    monkeypatch.delenv(ENV)
    for seed in range(3):
        fresh, _ = _logits(m, seed=215 + seed)
        static_logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_weights = _noaux_route(static_logits, bias)
        expected = backend.run(x, x_sf, ref_ids, ref_weights, tile_size=TILE)
        _assert_same_rows(out, expected, _local_counts(ref_ids, offset, num_local))
