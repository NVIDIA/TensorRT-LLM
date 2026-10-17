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
"""Gate C1(b, c): per-row R invariance of the MTP layer's MoE on real weights (one GPU, one TP8 rank shard).

draft_live_rows runs the MTP layer's routed MoE, shared expert and their add on the live
rows only. That is bit-identical to the served step 0 only if each of them
computes a row the same way whatever the other rows of the call are. Checked
here, bitwise, on rank ``GLM_MOE_DSA_DRAFT_LIVE_ROWS_TP_RANK``'s shard of ``model.layers.78``:

* routed MoE, native FlashInfer TRT-LLM-Gen BF16 leaf and draft_routed_moe (each with its
  own launch configuration per row count): row g of the R = 6 call equals the
  R = 1 call on row g, for every g, with the six rows chosen to cover
  D = 8..48 distinct experts (Gaussian rows and real token-embedding rows);
  rows (g1, g2) of an R = 8 call equal the R = 2 call on them (B = 2 at K = 3);
* shared expert (the served BF16 ``GatedMLP``, TP8 shard, real weights): row g
  at M = 6 equals M = 1, rows (g1, g2) at M = 8 equal M = 2;
* the same under CUDA-graph replay (graphs captured at R = 6 / R = 1, replayed
  with new rows copied into their static inputs);
* negative control: row g of the R = 6 call against the R = 1 call on another
  row differs.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "draft_routed_moe"))
import draft_moe_fixtures as fx  # noqa: E402

TP_RANK = int(os.environ.get("GLM_MOE_DSA_DRAFT_LIVE_ROWS_TP_RANK", "0"))
D_TARGETS = (8, 12, 16, 24, 31, 40, 48)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _eq(a, b):
    return torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def layer():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100-family GPU")
    import tensorrt_llm  # noqa: F401  (loads th_common: draft_routed_moe and the native MoE ops)

    if not hasattr(torch.ops.trtllm, "glm_moe_dsa_draft_routed_moe"):
        pytest.skip("draft_routed_moe op not in this build")
    return fx.build_native(TP_RANK)


@pytest.fixture(scope="module")
def pools(layer):
    gauss = fx.RowPool.gaussian(layer, num=1024, seed=10_000)
    g = torch.Generator().manual_seed(77)
    ids = torch.randint(0, 100_000, (1024,), generator=g).tolist()
    xs = fx.embedding_rows(layer, ids)
    emb = fx.RowPool(xs, fx.top8_sets(layer.bias, layer.logits(xs)))
    return {"gauss": gauss, "emb": emb}


def _routed(kind, layer, x, logits):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    if kind == "draft_routed_moe":
        return rm.run(layer.module, layer.bias, x, logits)
    return layer.native(x, logits)


@pytest.mark.parametrize("kind", ["native", "draft_routed_moe"])
@pytest.mark.parametrize("pool", ["gauss", "emb"])
def test_routed_row_invariance_r6(layer, pools, kind, pool):
    bad, cases, negatives = [], 0, 0
    for t, target in enumerate(D_TARGETS):
        for rep in range(3):
            x6, d = pools[pool].take(6, target, seed=1000 * t + rep)
            l6 = layer.logits(x6)
            out6 = _routed(kind, layer, x6, l6)
            for g in range(6):
                out1 = _routed(kind, layer, x6[g : g + 1].contiguous(), l6[g : g + 1].contiguous())
                cases += 1
                if not _eq(out6[g : g + 1], out1):
                    bad.append((target, d, rep, g))
            # Negative control: another (different) row's R = 1 output differs from row 0's.
            others = [h for h in range(1, 6) if not _eq(x6[h], x6[0])]
            if others:
                h = others[0]
                out1o = _routed(kind, layer, x6[h : h + 1].contiguous(), l6[h : h + 1].contiguous())
                assert not _eq(out6[0:1], out1o), "negative control not caught"
                negatives += 1
    torch.cuda.synchronize()
    assert negatives >= len(D_TARGETS), negatives
    print(
        f"GLM_MOE_DSA_DRAFT_LIVE_ROWS_ORACLE routed {kind} {pool} R=6 vs R=1 cases={cases} mismatching={len(bad)} "
        f"negative_controls={negatives} {bad[:4]}",
        flush=True,
    )
    assert not bad


@pytest.mark.parametrize("kind", ["native", "draft_routed_moe"])
def test_routed_pairs_r8(layer, pools, kind):
    bad, cases = [], 0
    for t, target in enumerate((16, 31, 48, 64)):
        x8, d = pools["gauss"].take(8, target, seed=5000 + t)
        l8 = layer.logits(x8)
        out8 = _routed(kind, layer, x8, l8)
        for g1, g2 in ((0, 7), (7, 0), (3, 4), (5, 2), (1, 6)):
            idx = torch.tensor([g1, g2], device="cuda")
            out2 = _routed(kind, layer, x8[idx].contiguous(), l8[idx].contiguous())
            cases += 1
            if not _eq(out8[idx], out2):
                bad.append((target, d, g1, g2))
    torch.cuda.synchronize()
    print(
        f"GLM_MOE_DSA_DRAFT_LIVE_ROWS_ORACLE routed {kind} R=8 vs R=2 cases={cases} mismatching={len(bad)} {bad[:4]}",
        flush=True,
    )
    assert not bad


def _shared_expert(layer):
    """The served BF16 shared expert of the MTP layer, TP rank shard, real weights (Deepseekv3MoE's construction)."""
    import copy
    import math

    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
    import routed_moe_fixtures as wfx

    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    mc = wfx._served_model_config(fx.MTP_LAYER, TP_RANK)
    mc.quant_config.exclude_modules = [f"model.layers.{fx.MTP_LAYER}*"]
    shared_mc = copy.copy(mc)
    shared_mc.quant_config = QuantConfig(quant_algo=None, kv_cache_quant_algo=QuantAlgo.FP8)
    tp = math.gcd(fx.INTER // 1, mc.mapping.tp_size)
    se = GatedMLP(
        hidden_size=fx.HIDDEN,
        intermediate_size=fx.INTER,
        bias=False,
        dtype=torch.bfloat16,
        config=shared_mc,
        overridden_tp_size=tp,
        reduce_output=False,
    )
    rd = fx.Reader()
    pre = f"model.layers.{fx.MTP_LAYER}.mlp.shared_experts"
    full = {p: rd.full(f"{pre}.{p}.weight") for p in ("gate_proj", "up_proj", "down_proj")}
    if hasattr(se, "gate_up_proj"):
        se.gate_up_proj.load_weights([{"weight": full["gate_proj"]}, {"weight": full["up_proj"]}])
    else:
        se.gate_proj.load_weights([{"weight": full["gate_proj"]}])
        se.up_proj.load_weights([{"weight": full["up_proj"]}])
    se.down_proj.load_weights([{"weight": full["down_proj"]}])
    wfx._walk_post_load(se)
    return se.cuda(), tp


@pytest.fixture(scope="module")
def shared(layer):
    se, tp = _shared_expert(layer)
    print(
        f"GLM_MOE_DSA_DRAFT_LIVE_ROWS_ORACLE shared expert TP{tp} rank {TP_RANK} {type(se).__name__}",
        flush=True,
    )
    return se


def test_shared_row_invariance(layer, pools, shared):
    bad, cases = [], 0
    with torch.inference_mode():
        for seed in range(12):
            x6, _ = pools["emb" if seed % 2 else "gauss"].take(6, 8 + 4 * seed, seed=seed)
            out6 = shared(x6)
            for g in range(6):
                cases += 1
                if not _eq(out6[g : g + 1], shared(x6[g : g + 1].contiguous())):
                    bad.append((seed, g))
            x8, _ = pools["gauss"].take(8, 40, seed=100 + seed)
            out8 = shared(x8)
            for g1, g2 in ((0, 7), (6, 1), (3, 4)):
                idx = torch.tensor([g1, g2], device="cuda")
                cases += 1
                if not _eq(out8[idx], shared(x8[idx].contiguous())):
                    bad.append((seed, g1, g2))
        other = shared(x6[1:2].contiguous())
        assert not _eq(out6[0:1], other) or _eq(x6[0], x6[1]), "negative control not caught"
    torch.cuda.synchronize()
    print(
        f"GLM_MOE_DSA_DRAFT_LIVE_ROWS_ORACLE shared M=6/8 vs M=1/2 cases={cases} mismatching={len(bad)} {bad[:4]}",
        flush=True,
    )
    assert not bad


def _capture(fn, *static):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn(*static)
    torch.cuda.synchronize()
    return graph, out


@pytest.mark.parametrize("kind", ["native", "draft_routed_moe"])
def test_graph_replay(layer, pools, shared, kind):
    """R = 6 and R = 1 graphs of the routed MoE and of the shared expert, replayed with new rows."""
    x6s = torch.empty(6, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
    l6s = torch.empty(6, fx.NUM_EXPERTS, dtype=torch.float32, device="cuda")
    x1s = torch.empty(1, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
    l1s = torch.empty(1, fx.NUM_EXPERTS, dtype=torch.float32, device="cuda")
    x0, _ = pools["gauss"].take(6, 31, seed=9)
    x6s.copy_(x0)
    l6s.copy_(layer.logits(x0))
    x1s.copy_(x0[:1])
    l1s.copy_(layer.logits(x0[:1]))
    with torch.inference_mode():
        g6, o6 = _capture(lambda x, lg: _routed(kind, layer, x, lg), x6s, l6s)
        g1, o1 = _capture(lambda x, lg: _routed(kind, layer, x, lg), x1s, l1s)
        s6, so6 = _capture(shared, x6s)
        s1, so1 = _capture(shared, x1s)
        bad, cases = [], 0
        for trial, target in enumerate(D_TARGETS):
            x, _ = pools["emb" if trial % 2 else "gauss"].take(6, target, seed=300 + trial)
            lg = layer.logits(x)
            x6s.copy_(x)
            l6s.copy_(lg)
            g6.replay()
            s6.replay()
            r6, sh6 = o6.clone(), so6.clone()
            for g in range(6):
                x1s.copy_(x[g : g + 1])
                l1s.copy_(lg[g : g + 1])
                g1.replay()
                s1.replay()
                cases += 1
                if not (_eq(r6[g : g + 1], o1) and _eq(sh6[g : g + 1], so1)):
                    bad.append((target, g))
        torch.cuda.synchronize()
    print(
        f"GLM_MOE_DSA_DRAFT_LIVE_ROWS_ORACLE graph {kind} routed+shared R=6 vs R=1 cases={cases} "
        f"mismatching={len(bad)} {bad[:4]}",
        flush=True,
    )
    assert not bad
