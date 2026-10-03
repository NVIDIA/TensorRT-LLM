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
"""k3_moe for steps of up to 64 tokens (K3MoeWideState: trtllm::k3_route_quant, then the m_max 64 build of k3_moe)
on one GPU at the Kimi K3 TP16 deployment's routed-expert rank layout (experts TP4 x EP4: 224 local experts,
intermediate 768 per rank), at every M in 1..64, for these routings:
- random router logits;
- 16 local experts per token (1024 local pairs at M = 64);
- disjoint: 3 local experts per token, no two tokens sharing one;
- hot: one local expert in every token's top-16 (8 groups of it at M = 64);
- group_cap: 100 experts with 9 of the 64 tokens and 124 with one (324 groups at M = 64: the kernel's group capacity);
- none_local: no local expert;
- rtb2 (with K3_ROUTING_DUMP naming the campaign's router dump): 4 draws of 8 decode forwards of 8 tokens each, the
  step's busiest EP group mapped onto this rank (M tokens: the first M of the draw).
Checks: against the stock path (trtllm::kimi_k3_noaux_tc_mxfp8_quant, then the TRTLLM-Gen W4A8_MXFP4_MXFP8 MoE runner
with those ids) and an fp64 reference over the dequantized MXFP4 experts (op-catalog gates: 8 ulp of the row max per
element, 4 ulp relative RMS); run-to-run identical bits; the slab armed and the layer's counters zero after every
call; at M <= 8, within one bf16 ulp of trtllm::k3_fused_moe (the decode build; bit-identity reported). Then: calls
of two layers at mixed M on one stream and replayed from a CUDA graph give each call's bits alone; 0 and 65 tokens are
refused. Weights are random checkpoint-format MXFP4 experts put through TRT-LLM's own loader."""

import functools
import math
import os
from types import SimpleNamespace

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_moe needs sm_100")

H, TOP_K, NUM_EXPERTS, SV = 3584, 16, 896, 32
I_TP, E_LOCAL, MOE_TP, TP_RANK, EP_RANK = 768, 224, 4, 1, 1  # one rank of experts TP4 x EP4
OFFSET = EP_RANK * E_LOCAL
GATE_CAP, LINEAR_CAP = (
    4.0,
    25.0,
)  # the SiTU caps (activation_situ_beta, activation_situ_linear_beta)
RSF = 2.827
ULP = 2.0**-8
E4M3_MAX = 448.0
M_MAX = 64
M_ALL = list(range(1, M_MAX + 1))
ROUTING_DUMP = os.environ.get("K3_ROUTING_DUMP")


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op as _rq  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rand_mxfp4(rows, k, gen):
    """Random checkpoint-format MXFP4: packed [rows, k / 2] (low nibble = even k), E8M0 per 32 k, scaled so a k-long
    dot product lands near std 3."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


@functools.lru_cache(maxsize=None)
def _experts(seed: int = 20260928):
    """This rank's experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader (the buffers both kernels read),
    and the rank's logical slices of the checkpoint tensors (what the reference reads)."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    i_full = I_TP * MOE_TP
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=MOE_TP, tp_rank=TP_RANK, scaling_vector_size=SV, intermediate_size=i_full,
                             intermediate_size_per_partition=I_TP, hidden_size=H)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = dict(
        w31=torch.empty(E_LOCAL, 2 * I_TP, H // 2, **kw),
        w31s=torch.empty(E_LOCAL, 2 * I_TP, H // SV, **kw),
        w2=torch.empty(E_LOCAL, H, I_TP // 2, **kw),
        w2s=torch.empty(E_LOCAL, H, I_TP // SV, **kw),
    )
    raw = {name: [] for name in ("up", "up_s", "gate", "gate_s", "down", "down_s")}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = TP_RANK * I_TP, (TP_RANK + 1) * I_TP
    for e in range(E_LOCAL):
        w1, w1s = _rand_mxfp4(i_full, H, gen)  # gate
        w3, w3s = _rand_mxfp4(i_full, H, gen)  # up
        w2, w2s = _rand_mxfp4(H, i_full, gen)  # down
        method.load_expert_w3_w1_weight(module, w1, w3, proc["w31"][e])
        method.load_expert_w2_weight(module, w2, proc["w2"][e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, w1s, w3s, proc["w31s"][e])
        method.load_expert_w2_weight_scale_mxfp4(module, w2s, proc["w2s"][e])
        raw["up"].append(w3[lo:hi])
        raw["up_s"].append(w3s[lo:hi])
        raw["gate"].append(w1[lo:hi])
        raw["gate_s"].append(w1s[lo:hi])
        raw["down"].append(w2[:, lo // 2 : hi // 2].contiguous())
        raw["down_s"].append(w2s[:, lo // SV : hi // SV].contiguous())
    torch.cuda.synchronize()
    gen_b = torch.Generator(device="cuda").manual_seed(seed + 1)
    bias = (torch.randn(NUM_EXPERTS, generator=gen_b, device="cuda") * 0.05).float()
    return proc, raw, bias


@functools.lru_cache(maxsize=None)
def _wide():
    """One wide state and two layers on it (the same experts; two counter sets)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    proc, _, _ = _experts()
    state = op.K3MoeWideState(torch.device("cuda", torch.cuda.current_device()), I_TP, E_LOCAL)
    weights = (proc["w31"], proc["w31s"], proc["w2"], proc["w2s"])
    return state, state.layer(*weights), state.layer(*weights)


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _deq_w(packed, sf):
    lut = torch.tensor(_E2M1, device=packed.device)
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, device=packed.device)
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * torch.exp2(sf.float() - 127.0).repeat_interleave(SV, dim=1)


def _deq_x(x_fp8, x_sf):
    rows, k = x_fp8.shape
    return x_fp8.float() * torch.exp2(
        x_sf.reshape(rows, k // SV).float() - 127.0
    ).repeat_interleave(SV, dim=1)


def _requant(act):
    """The FC1 epilogue's MXFP8 requantization per 32 columns (round-up scale), dequantized."""
    rows, cols = act.shape
    blocks = act.reshape(rows, cols // SV, SV)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    ex = torch.ceil(torch.log2(amax / E4M3_MAX))
    ex = torch.where(amax == 0, torch.full_like(amax, -127.0), ex).clamp(-127.0, 127.0)
    scale = torch.exp2(ex)
    q8 = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (q8.float() * scale).reshape(rows, cols)


def _reference(raw, x_deq, ids, weights):
    """fp64 routed MoE over this rank's experts from the checkpoint tensors: SiTU, the MXFP8 intermediate, the down
    projection, the routing-weighted sum."""
    out = torch.zeros(x_deq.shape[0], H, device="cuda")
    for e in range(E_LOCAL):
        tok, slot = (ids == OFFSET + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = x_deq[tok].double()
        up = (xe @ _deq_w(raw["up"][e], raw["up_s"][e]).double().t()).float()
        gate = (xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).double().t()).float()
        act = (
            GATE_CAP
            * torch.tanh(gate / GATE_CAP)
            * torch.sigmoid(gate)
            * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP))
        )
        y = (_requant(act).double() @ _deq_w(raw["down"][e], raw["down_s"][e]).double().t()).float()
        out.index_add_(0, tok, y * weights[tok, slot].float().unsqueeze(1))
    return out


def _compare(y, ref):
    """Op-catalog gates: |d| <= 8 ulp of the row's max |ref| per element, relative RMS <= 4 ulp; finite."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    finite = bool(torch.isfinite(o).all())
    return dict(elt_ulp=elt, rms_ulp=rms, ok=finite and elt <= 8.0 and rms <= 4.0)


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in bf16 ulps between two bf16 tensors (bit patterns as ordered integers)."""

    def ordered(x):
        i = x.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((ordered(a) - ordered(b)).abs().max().item()) if a.numel() else 0


def _stock(proc, bias, x, logits):
    """The model's base path for these experts: fused route + MXFP8 quant, then the TRTLLM-Gen runner, pre-routed."""
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    ops = _ops()
    ids, w, x_fp8, x_sf = ops.kimi_k3_noaux_tc_mxfp8_quant(logits, bias, x, RSF)
    alpha = torch.full((E_LOCAL,), GATE_CAP, dtype=torch.float32, device="cuda")
    beta = torch.full((E_LOCAL,), LINEAR_CAP, dtype=torch.float32, device="cuda")
    y = ops.mxe4m3_mxe2m1_block_scale_moe_runner(
        None, None, x_fp8, x_sf.view(-1), proc["w31"], proc["w31s"], None, alpha, beta, None, proc["w2"],
        proc["w2s"], None, NUM_EXPERTS, TOP_K, 1, 1, I_TP, H, I_TP, OFFSET, E_LOCAL, 1.0,
        int(RoutingMethodType.DeepSeekV3), int(ActType_TrtllmGen.SiTu), topk_weights=w, topk_ids=ids,
    )  # fmt: skip
    return y, ids, w, x_fp8, x_sf


def _wide_call(layer, bias, x, logits, keep=None):
    """The wide chain: k3_route_quant (early trigger, as k3_moe's PDL producer), then k3_moe. With keep (a list), the
    route outputs are appended to it."""
    ids, w, x_fp8, x_sf = _ops().k3_route_quant(logits, bias, x, RSF, early_trigger=True)
    if keep is not None:
        keep.append((ids, w, x_fp8, x_sf))
    return layer(x_fp8, x_sf, ids, w, OFFSET)


def _scratch_rearmed(state, *layers):
    """The intermediate slab armed again (FP8 -0.0 codes, E8M0 NaN scale words) and the layers' counters zero."""
    mod = state.mod
    cs = state.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
    armed = bool((state.c == -128).all()) and bool((cs[..., :4] == -1).all())
    return armed and all(bool((layer.counters == 0).all()) for layer in layers)


CASES = ["random", "16_local", "disjoint", "hot", "group_cap", "none_local"]
RTB2_DRAWS = 4


def _chosen_logits(chosen, gen):
    """Logits whose top-16 (sigmoid + bias) is exactly each row's chosen experts: chosen ~ N(2, 0.5) (sigmoid
    0.62..0.97, so varied routing weights), the rest -20."""
    logits = torch.full((len(chosen), NUM_EXPERTS), -20.0, device="cuda")
    for t, experts in enumerate(chosen):
        idx = torch.tensor(sorted(experts), device="cuda")
        logits[t, idx] = 2.0 + 0.5 * torch.randn(len(experts), generator=gen, device="cuda")
    return logits


@functools.lru_cache(maxsize=None)
def _rtb2_steps(path, draws, seed=11):
    """R decode forwards of 8 tokens each from the router dump (one random layer per draw), R = 8 (64 tokens): each
    step's ids rotated so that its busiest EP group (most distinct experts) is this rank's."""
    import numpy as np

    z = np.load(path)
    ntok = z["fwd_ntok"]
    starts = np.concatenate([[0], np.cumsum(ntok)[:-1]])
    decode = [i for i, n in enumerate(ntok) if n == 8]
    layers = [int(v) for v in z["layers"]]
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(draws):
        fwds = rng.choice(decode, size=M_MAX // 8, replace=False)
        toks = np.concatenate([np.arange(starts[f], starts[f] + 8) for f in fwds])
        ids = z[f"ids_{rng.choice(layers)}"][toks].astype(np.int64)
        counts = np.bincount(ids.reshape(-1), minlength=NUM_EXPERTS).reshape(4, E_LOCAL)
        busiest = int(np.argmax((counts > 0).sum(axis=1)))
        ids = (ids + (EP_RANK - busiest) * E_LOCAL) % NUM_EXPERTS
        steps.append([set(int(e) for e in row) for row in ids])
    return steps


@functools.lru_cache(maxsize=None)
def _tokens(case: str, seed: int = 7):
    """64 tokens: router logits and the latent rows (M tokens use the first M)."""
    salt = CASES.index(case) if case in CASES else len(CASES) + int(case[5:])
    gen = torch.Generator(device="cuda").manual_seed(seed + salt)
    cpu = torch.Generator().manual_seed(seed + salt)
    x = torch.randn(M_MAX, H, generator=gen, device="cuda").bfloat16()
    local = list(range(OFFSET, OFFSET + E_LOCAL))
    remote = [e for e in range(NUM_EXPERTS) if e not in set(local)]

    def pick(pool, k):
        return [pool[i] for i in torch.randperm(len(pool), generator=cpu)[:k].tolist()]

    if case == "random":
        return (torch.randn(M_MAX, NUM_EXPERTS, generator=gen, device="cuda") * 3.0).float(), x
    if case == "none_local":
        logits = (torch.randn(M_MAX, NUM_EXPERTS, generator=gen, device="cuda") * 3.0).float()
        logits[:, OFFSET : OFFSET + E_LOCAL] = -30.0
        return logits, x
    if case == "16_local":
        chosen = [pick(local, TOP_K) for _ in range(M_MAX)]
    elif case == "disjoint":
        perm = pick(local, 3 * M_MAX)
        chosen = [perm[3 * t : 3 * t + 3] + pick(remote, TOP_K - 3) for t in range(M_MAX)]
    elif case == "hot":
        hot = local[17]
        chosen = [[hot] + pick([e for e in local if e != hot], TOP_K - 1) for _ in range(M_MAX)]
    elif case == "group_cap":
        # 100 experts on 9 tokens and 124 on one: 1024 pairs, 2 * 100 + 124 = 324 groups. Experts are dealt in order
        # of their count to the tokens with the most free slots, so every token gets 16 distinct experts.
        order = pick(local, E_LOCAL)
        free = [TOP_K] * M_MAX
        chosen = [[] for _ in range(M_MAX)]
        for i, e in enumerate(order):
            need = 9 if i < 100 else 1
            toks = sorted(range(M_MAX), key=lambda t: (-free[t], t))[:need]
            for t in toks:
                chosen[t].append(e)
                free[t] -= 1
        assert all(f == 0 for f in free)
    elif case.startswith("rtb2."):
        chosen = [sorted(e) for e in _rtb2_steps(ROUTING_DUMP, RTB2_DRAWS)[int(case[5:])]]
    else:
        raise ValueError(case)
    return _chosen_logits(chosen, gen), x


def _groups(ids):
    """The kernel's groups for these ids: sum over this rank's experts of ceil(tokens / 8)."""
    local = ids[(ids >= OFFSET) & (ids < OFFSET + E_LOCAL)]
    counts = torch.bincount(local - OFFSET, minlength=E_LOCAL)
    return int(((counts + 7) // 8).sum())


def _check(case, m):
    """One call of M tokens of a routing case: against the stock path and the fp64 reference, re-runs, scratch, and
    at M <= 8 the decode build."""
    _ops()
    proc, raw, bias = _experts()
    state, layer, _ = _wide()
    logits64, x64 = _tokens(case)
    logits, x = logits64[:m].contiguous(), x64[:m].contiguous()
    y = _wide_call(layer, bias, x, logits)
    torch.cuda.synchronize()
    rearmed = _scratch_rearmed(state, layer)
    reruns = [_wide_call(layer, bias, x, logits) for _ in range(2)]
    det = all(torch.equal(_bits(r), _bits(y)) for r in reruns)
    y_stock, ids, w, x_fp8, x_sf = _stock(proc, bias, x, logits)
    if case not in ("random", "none_local"):
        # The constructed logits route exactly the chosen experts.
        want = torch.topk(logits, TOP_K, dim=1).indices.sort(dim=1).values
        assert torch.equal(ids.long().sort(dim=1).values, want)
    groups = _groups(ids)
    if case == "group_cap" and m == M_MAX:
        assert groups == state.mod.G_CAP == 324
    decode = ""
    if m <= 8:
        y8 = _ops().k3_fused_moe(x, logits, bias, proc["w31"], proc["w31s"], proc["w2"], proc["w2s"], OFFSET,
                                 E_LOCAL, RSF)  # fmt: skip
        ulp_dec = _max_ulp(y, y8)
        decode = f" max_ulp_vs_decode={ulp_dec} bits_as_decode={torch.equal(_bits(y), _bits(y8))}"
        assert ulp_dec <= 1
    local = int(((ids >= OFFSET) & (ids < OFFSET + E_LOCAL)).sum())
    if local == 0:
        zeros = bool((y.float() == 0).all())
        print(f"OPCHECK op=k3_moe_wide case={case} M={m} zeros={zeros} det={det} "
              f"scratch_rearmed={rearmed}{decode}")  # fmt: skip
        assert zeros and det and rearmed
        return
    ref = _reference(raw, _deq_x(x_fp8, x_sf), ids, w).bfloat16()
    c_ref, c_stock, c_stock_ref = _compare(y, ref), _compare(y, y_stock), _compare(y_stock, ref)
    print(f"OPCHECK op=k3_moe_wide case={case} M={m} local_pairs={local} groups={groups} "
          f"vs_ref_elt_ulp={c_ref['elt_ulp']:.2f} vs_ref_rms_ulp={c_ref['rms_ulp']:.2f} "
          f"vs_stock_elt_ulp={c_stock['elt_ulp']:.2f} vs_stock_rms_ulp={c_stock['rms_ulp']:.2f} "
          f"stock_vs_ref_elt_ulp={c_stock_ref['elt_ulp']:.2f} det={det} "
          f"scratch_rearmed={rearmed}{decode}")  # fmt: skip
    assert c_ref["ok"] and c_stock["ok"] and c_stock_ref["ok"]
    assert det and rearmed


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("case", CASES)
def test_k3_moe_wide(case, m):
    _check(case, m)


@pytest.mark.skipif(not ROUTING_DUMP, reason="K3_ROUTING_DUMP names no router dump")
@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("draw", range(RTB2_DRAWS))
def test_k3_moe_wide_rtb2(draw, m):
    _check(f"rtb2.{draw}", m)


def test_k3_moe_wide_mixed_sequence():
    """Two layers of one state at M 64, 16, 1, 40, 8, 64, 23 back to back on one stream, then the same calls replayed
    from one CUDA graph with refilled inputs: each call the bits of the same call alone, the scratch re-armed."""
    _ops()
    _, _, bias = _experts()
    state, layer_a, layer_b = _wide()
    logits64, x64 = _tokens("random")
    calls = [
        (layer_a, 64),
        (layer_b, 16),
        (layer_a, 1),
        (layer_b, 40),
        (layer_a, 8),
        (layer_b, 64),
        (layer_a, 23),
    ]

    def run(layer, m):
        return _wide_call(layer, bias, x64[:m].contiguous(), logits64[:m].contiguous())

    alone = {}
    for layer, m in calls:
        key = (id(layer), m)
        if key not in alone:
            alone[key] = run(layer, m)
            torch.cuda.synchronize()
    seq = [run(layer, m) for layer, m in calls]
    torch.cuda.synchronize()
    assert all(
        torch.equal(_bits(y), _bits(alone[(id(layer), m)])) for (layer, m), y in zip(calls, seq)
    )
    assert _scratch_rearmed(state, layer_a, layer_b)

    # The same sequence from a graph: inputs in static buffers, filled after capture. The route outputs are held with
    # the graph, so that no allocation inside the graph reuses their addresses.
    static_x = torch.zeros_like(x64)
    static_logits = torch.zeros_like(logits64)
    routed = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            outs = [_wide_call(layer, bias, static_x[:m], static_logits[:m], routed) for layer, m in calls]
    static_x.copy_(x64)
    static_logits.copy_(logits64)
    graph.replay()
    torch.cuda.synchronize()
    assert all(
        torch.equal(_bits(y), _bits(alone[(id(layer), m)])) for (layer, m), y in zip(calls, outs)
    )
    assert _scratch_rearmed(state, layer_a, layer_b)


@pytest.mark.parametrize("m", [0, M_MAX + 1])
def test_token_limit(m):
    _ops()
    _, layer, _ = _wide()
    x_fp8 = torch.zeros(m, H, dtype=torch.float8_e4m3fn, device="cuda")
    x_sf = torch.zeros(m, H // SV, dtype=torch.uint8, device="cuda")
    ids = torch.zeros(m, TOP_K, dtype=torch.int32, device="cuda")
    w = torch.zeros(m, TOP_K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        layer(x_fp8, x_sf, ids, w, OFFSET)
