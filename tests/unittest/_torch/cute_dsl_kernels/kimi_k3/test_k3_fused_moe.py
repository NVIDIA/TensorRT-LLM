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
"""K3MoeLayer (trtllm::k3_route_quant + the persistent k3_moe kernel, M <= 8) on one GPU, at the Kimi K3 TP16
deployment's routed-expert rank layout (experts TP4 x EP4: 224 local experts, intermediate 768 per rank), at every M
in 1..8 with random routing, with 0, 4 and 16 of each token's experts local, and with 16 local experts per token none
shared (16 M groups: the kernel's group capacity at M = 8): against the stock path
(trtllm::kimi_k3_noaux_tc_mxfp8_quant then the TRTLLM-Gen W4A8_MXFP4_MXFP8 MoE runner with those ids) and an fp32
reference over the dequantized MXFP4 experts (op-catalog gates: 8 ulp of the row max per element, 4 ulp relative RMS),
run-to-run identical bits, each M's rows within one bf16 ulp of the same rows of the 8-token call (bit-identity
reported), and the kernel's scratch (intermediate slab, layer counters) re-armed after every call. Weights are random
checkpoint-format MXFP4 experts put through TRT-LLM's own loader."""

import functools
import math
from types import SimpleNamespace

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_fused_moe needs sm_100")

H, TOP_K, NUM_EXPERTS, SV = 3584, 16, 896, 32
I_TP, E_LOCAL, MOE_TP, TP_RANK, EP_RANK = 768, 224, 4, 1, 1  # one rank of experts TP4 x EP4
OFFSET = EP_RANK * E_LOCAL
GATE_CAP, LINEAR_CAP = 4.0, 25.0  # the SiTU caps (activation_situ_beta, activation_situ_linear_beta)
RSF = 2.827
ULP = 2.0**-8
E4M3_MAX = 448.0
M_ALL = list(range(1, 9))


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rand_mxfp4(rows, k, gen):
    """Random checkpoint-format MXFP4: packed [rows, k / 2] (low nibble = even k), E8M0 per 32 k, scaled so a k-long
    dot product lands near std 3."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k))
    exps = torch.randint(base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen)
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


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _deq_w(packed, sf):
    lut = torch.tensor(_E2M1, device=packed.device)
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, device=packed.device)
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * torch.exp2(sf.float() - 127.0).repeat_interleave(SV, dim=1)


def _deq_x(x_fp8, x_sf):
    rows, k = x_fp8.shape
    return x_fp8.float() * torch.exp2(x_sf.reshape(rows, k // SV).float() - 127.0).repeat_interleave(SV, dim=1)


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
    """fp32 routed MoE over this rank's experts from the checkpoint tensors (TF32 off): SiTU, the MXFP8 intermediate,
    the down projection, the routing-weighted sum."""
    allow = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.zeros(x_deq.shape[0], H, device="cuda")
        for e in range(E_LOCAL):
            tok, slot = (ids == OFFSET + e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = x_deq[tok].double()
            up = (xe @ _deq_w(raw["up"][e], raw["up_s"][e]).double().t()).float()
            gate = (xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).double().t()).float()
            act = GATE_CAP * torch.tanh(gate / GATE_CAP) * torch.sigmoid(gate) * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP))
            y = (_requant(act).double() @ _deq_w(raw["down"][e], raw["down_s"][e]).double().t()).float()
            out.index_add_(0, tok, y * weights[tok, slot].float().unsqueeze(1))
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow


def _compare(y, ref):
    """Op-catalog gates: |d| <= 8 ulp of the row's max |ref| per element, relative RMS <= 4 ulp; finite."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    finite = bool(torch.isfinite(o).all())
    return dict(elt_ulp=elt, rms_ulp=rms, ok=finite and elt <= 8.0 and rms <= 4.0)


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


@functools.lru_cache(maxsize=None)
def _layer():
    """One K3MoeState on this GPU and the layer of _experts()' buffers on it."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _ops()
    proc, _, _ = _experts()
    state = op.K3MoeState(torch.device("cuda", torch.cuda.current_device()), I_TP, E_LOCAL)
    return state.layer(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"])


def _fused(proc, bias, x, logits):
    return _layer()(x, logits, bias, OFFSET, RSF)


def _scratch_rearmed():
    """The intermediate slab armed again (FP8 -0.0 codes, E8M0 NaN scale words) and the layer's counters zero."""
    layer = _layer()
    st = layer.state
    mod = st.mod
    cs = st.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
    armed = bool((st.c == -128).all()) and bool((cs[..., :4] == -1).all())
    return armed and bool((layer.counters == 0).all())


CASES = ["random", "4_local", "16_local", "16_local_disjoint", "none_local"]


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in bf16 ulps between two bf16 tensors (bit patterns as ordered integers)."""

    def ordered(x):
        i = x.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((ordered(a) - ordered(b)).abs().max().item()) if a.numel() else 0


@functools.lru_cache(maxsize=None)
def _tokens(case: str, seed: int = 7):
    """8 tokens: router logits and the latent rows. "<k>_local": k of this rank's experts in each token's top-16;
    "16_local_disjoint": 16 local experts per token, no two tokens sharing one (16 M groups: 128 = the kernel's group
    capacity at M = 8); "none_local": no local expert."""
    salt = CASES.index(case)
    gen = torch.Generator(device="cuda").manual_seed(seed + salt)
    cpu = torch.Generator().manual_seed(seed + salt)
    logits = (torch.randn(8, NUM_EXPERTS, generator=gen, device="cuda") * 3.0).float()
    if case.endswith("_local") and case != "none_local":
        k = int(case.split("_")[0])
        logits = logits - 30.0
        for t in range(8):
            logits[t, OFFSET + torch.randperm(E_LOCAL, generator=cpu)[:k].cuda()] = 30.0
    elif case == "16_local_disjoint":
        logits = logits - 30.0
        perm = torch.randperm(E_LOCAL, generator=cpu)[: 8 * TOP_K].view(8, TOP_K)
        for t in range(8):
            logits[t, OFFSET + perm[t].cuda()] = 30.0
    elif case == "none_local":
        logits[:, OFFSET : OFFSET + E_LOCAL] = -30.0
    x = torch.randn(8, H, generator=gen, device="cuda").bfloat16()
    return logits, x


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("case", CASES)
def test_k3_fused_moe(case, m):
    """Each M's rows against the 8-token call: the slice FC2 adds a token's expert terms in slices whose bounds
    follow the step's group count, so a row may round differently with other tokens present (at most 1 bf16 ulp);
    bit-identity is reported."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _ops()
    proc, raw, bias = _experts()
    ok, why = op.is_supported(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"], E_LOCAL)
    assert ok, why
    logits8, x8 = _tokens(case)
    logits, x = logits8[:m].contiguous(), x8[:m].contiguous()
    y = _fused(proc, bias, x, logits)
    torch.cuda.synchronize()
    rearmed = _scratch_rearmed()
    reruns = [_fused(proc, bias, x, logits) for _ in range(2)]
    y8 = _fused(proc, bias, x8, logits8)
    y_stock, ids, w, x_fp8, x_sf = _stock(proc, bias, x, logits)
    det = all(torch.equal(_bits(r), _bits(y)) for r in reruns)
    minv = torch.equal(_bits(y), _bits(y8[:m]))
    ulp_m8 = _max_ulp(y, y8[:m])
    local = int(((ids >= OFFSET) & (ids < OFFSET + E_LOCAL)).sum())
    if local == 0:
        zeros = bool((y.float() == 0).all())
        print(f"OPCHECK op=k3_fused_moe case={case} M={m} zeros={zeros} det={det} rows_as_m8={minv} "
              f"scratch_rearmed={rearmed}")  # fmt: skip
        assert zeros and det and minv and rearmed
        return
    ref = _reference(raw, _deq_x(x_fp8, x_sf), ids, w).bfloat16()
    c_ref, c_stock, c_stock_ref = _compare(y, ref), _compare(y, y_stock), _compare(y_stock, ref)
    groups = int(torch.unique(ids[(ids >= OFFSET) & (ids < OFFSET + E_LOCAL)]).numel())
    print(f"OPCHECK op=k3_fused_moe case={case} M={m} local_pairs={local} local_experts={groups} "
          f"vs_ref_elt_ulp={c_ref['elt_ulp']:.2f} vs_ref_rms_ulp={c_ref['rms_ulp']:.2f} "
          f"vs_stock_elt_ulp={c_stock['elt_ulp']:.2f} vs_stock_rms_ulp={c_stock['rms_ulp']:.2f} "
          f"stock_vs_ref_elt_ulp={c_stock_ref['elt_ulp']:.2f} det={det} rows_as_m8={minv} max_ulp_vs_m8={ulp_m8} "
          f"scratch_rearmed={rearmed}")  # fmt: skip
    assert c_ref["ok"] and c_stock["ok"] and c_stock_ref["ok"]
    assert det and ulp_m8 <= 1 and rearmed


def test_k3_fused_moe_mixed_m_sequence():
    """M 8, 1, 5, 2, 8, 7 back to back on one stream: each call the bits of the same call alone (the scratch the
    calls share is re-armed by each)."""
    _ops()
    proc, _, bias = _experts()
    logits8, x8 = _tokens("random")
    alone = {m: _fused(proc, bias, x8[:m].contiguous(), logits8[:m].contiguous()) for m in (1, 2, 5, 7, 8)}
    seq = [(m, _fused(proc, bias, x8[:m].contiguous(), logits8[:m].contiguous())) for m in (8, 1, 5, 2, 8, 7)]
    assert all(torch.equal(_bits(y), _bits(alone[m])) for m, y in seq)
    assert _scratch_rearmed()


def test_k3_fused_moe_partial_rows_past_m():
    """The combine loads the FC2 partial rows of all 8 token slots, and a call writes only its M tokens' rows: a
    fresh state's partials start zeroed, and each M's output is the same with the rows past M poisoned (NaN) as with
    them zeroed, so no output reads a row its call did not write."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _ops()
    proc, _, bias = _experts()
    fresh = op.K3MoeState(torch.device("cuda", torch.cuda.current_device()), I_TP, E_LOCAL)
    assert bool((fresh.part == 0).all())
    st = _layer().state
    logits8, x8 = _tokens("random")
    for m in M_ALL:
        logits, x = logits8[:m].contiguous(), x8[:m].contiguous()
        st.part.fill_(float("nan"))
        y_poisoned = _fused(proc, bias, x, logits)
        st.part.zero_()
        y_zeroed = _fused(proc, bias, x, logits)
        assert torch.equal(_bits(y_poisoned), _bits(y_zeroed)), m


def test_token_limit():
    _ops()
    proc, _, bias = _experts()
    logits = torch.zeros(9, NUM_EXPERTS, device="cuda")
    x = torch.zeros(9, H, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        _fused(proc, bias, x, logits)


def test_collective_workspaces_refuse_graph_capture():
    """The head all-gather's buffers (the front's) are created collectively (an MNNVL multicast allocation over the TP
    group): creating them under CUDA-graph capture raises instead of entering the collective. The per-rank state and
    a layer's counters refuse capture too (they allocate)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op
    from tensorrt_llm.mapping import Mapping

    _ops()
    proc, _, _ = _experts()
    device = torch.device("cuda", torch.cuda.current_device())
    state = op.K3MoeState(device, I_TP, E_LOCAL)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            op.K3MoeHeadWorkspace.create(mapping)
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            op.K3MoeState(device, I_TP, E_LOCAL)
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            state.layer(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"])
