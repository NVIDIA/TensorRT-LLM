# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/k3_moe catalog entry: k3_moe and k3_moe_wide on their caller-owned state.

One GPU (sm_100), one rank of the Kimi K3 TP16 deployment's routed experts: experts TP4 x EP4, so 224 of the 896
experts are local (global ids [224, 448)) with intermediate 768 per rank. The experts are random checkpoint-format
MXFP4 tensors put through TRT-LLM's own W4A8_MXFP4_MXFP8 TRTLLM-Gen loader, which writes the buffers that both k3_moe
and the stock runner read.

References: the stock path (trtllm::kimi_k3_noaux_tc_mxfp8_quant, then the TRTLLM-Gen W4A8_MXFP4_MXFP8 MoE runner on
those ids) and an fp64 reference over the dequantized MXFP4 experts (SiTU, the MXFP8 intermediate with the round-up
scale, the down projection, the routing-weighted sum), both under the op-catalog gates: 8 bf16 ulp of the token row's
largest magnitude per element, 4 ulp relative RMS. Everything about state is compared bit for bit.

The checks run in file order and share the state objects: a K3MoeState with layers A, B, C (A and C over this rank's
experts, B over other experts), a second K3MoeState, a K3MoeWideState with layers A and B. Each check makes its own
first (compiling) call eagerly where it needs one, so it also runs alone (-k). The kernel tests under
tests/unittest/_torch/cute_dsl_kernels/kimi_k3/ remain the exhaustive numerics; this file copies what it needs.

k3_moe_fused_front needs the TP group's head workspace: its cells run in moe/k3_moe_front's 4-rank matrix,
tests/unittest/_torch/modeling_v2/comm/_k3_moe_front_op_matrix.py.
"""

import functools
import math
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers the stock path's ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe import (
    K3MoeState,
    K3MoeWideState,
    is_supported,
    k3_moe,
    k3_moe_wide,
)

H, TOP_K, NUM_EXPERTS, SV = 3584, 16, 896, 32
I_TP, E_LOCAL, MOE_TP, TP_RANK, EP_RANK = 768, 224, 4, 1, 1  # one rank of experts TP4 x EP4
OFFSET = EP_RANK * E_LOCAL  # this rank's experts: global ids [224, 448)
# The routed experts' SiTU caps: k3_moe's build constants, Kimi K3's values.
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
ULP = 2.0**-8
E4M3_MAX = 448.0
DECODE_MAX, WIDE_MAX = 8, 64
DEV = "cuda"
DECODE_CASES = ("random", "16_local_disjoint", "none_local")
WIDE_CASES = ("random", "group_cap", "none_local")
WIDE_M = (1, 2, 7, 8, 9, 16, 33, 40, 64)
ROUTE_M = (1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 64)
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8)
WIDE_DIP_STEPS = (64, 64, 16, 1, 40, 64, 8, 23, 64, 9, 64)
REPLAYS = 6
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_CASE_SALT = {"random": 0, "16_local_disjoint": 1, "none_local": 2, "group_cap": 3}


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_moe needs sm_100")


@pytest.fixture(autouse=True)
def _inference_mode():
    """Run every check as the model runs these ops, under inference mode."""
    with torch.inference_mode():
        yield


# ── operands ──────────────────────────────────────────────────────────────


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bits."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _device() -> torch.device:
    return torch.device(DEV, torch.cuda.current_device())


def _rand_mxfp4(rows, k, gen):
    """Random checkpoint-format MXFP4.

    Packed [rows, k / 2] (low nibble = even k) and one E8M0 per 32 k, scaled so a k-long dot product lands near std 3.
    """
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device=DEV, generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device=DEV, generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


@functools.lru_cache(maxsize=None)
def _experts(seed: int = 20260928):
    """This rank's experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader, and the reference's view of them.

    Returns (the loader's buffers, this rank's logical slices of the checkpoint tensors, the routing bias).
    """
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    i_full = I_TP * MOE_TP
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(
        tp_size=MOE_TP,
        tp_rank=TP_RANK,
        scaling_vector_size=SV,
        intermediate_size=i_full,
        intermediate_size_per_partition=I_TP,
        hidden_size=H,
    )
    kw = dict(dtype=torch.uint8, device=DEV)
    proc = dict(
        w31=torch.empty(E_LOCAL, 2 * I_TP, H // 2, **kw),
        w31s=torch.empty(E_LOCAL, 2 * I_TP, H // SV, **kw),
        w2=torch.empty(E_LOCAL, H, I_TP // 2, **kw),
        w2s=torch.empty(E_LOCAL, H, I_TP // SV, **kw),
    )
    raw = {name: [] for name in ("up", "up_s", "gate", "gate_s", "down", "down_s")}
    gen = torch.Generator(device=DEV).manual_seed(seed)
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
    gen_b = torch.Generator(device=DEV).manual_seed(seed + 1)
    bias = (torch.randn(NUM_EXPERTS, generator=gen_b, device=DEV) * 0.05).float()
    return proc, raw, bias


@functools.lru_cache(maxsize=None)
def _rolled():
    """Other experts in the same layout: this rank's buffers rolled by one expert (expert e holds e - 1's weights)."""
    proc, _, _ = _experts()
    return {name: torch.roll(t, 1, dims=0).contiguous() for name, t in proc.items()}


def _bias() -> torch.Tensor:
    return _experts()[2]


def _weights(p):
    """A layer's four TRTLLM-Gen buffers, in K3MoeState.layer's argument order."""
    return p["w31"], p["w31s"], p["w2"], p["w2s"]


def _chosen_logits(chosen, gen):
    """Logits whose top-16 (sigmoid + bias) is exactly each row's chosen experts.

    Chosen ~ N(2, 0.5) (sigmoid 0.62..0.97, so the routing weights vary), the rest -20.
    """
    logits = torch.full((len(chosen), NUM_EXPERTS), -20.0, device=DEV)
    for t, experts in enumerate(chosen):
        idx = torch.tensor(sorted(experts), device=DEV)
        logits[t, idx] = 2.0 + 0.5 * torch.randn(len(experts), generator=gen, device=DEV)
    return logits


@functools.lru_cache(maxsize=None)
def _tokens(case: str, rows: int):
    """``rows`` tokens of a routing case: router logits fp32 [rows, 896] and the latent bf16 [rows, 3584].

    random: logits N(0, 3^2). none_local: no local expert in any token's top-16. 16_local_disjoint: 16 local experts
    per token, no two tokens sharing one (16 M groups; 128 at M 8, the M <= 8 build's group capacity). group_cap: 100
    local experts with 9 of the 64 tokens and 124 with one (324 groups at M 64, the wide build's group capacity).
    A call of M tokens takes the first M rows.
    """
    seed = 7 + 1000 * _CASE_SALT[case] + rows
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=gen, device=DEV).bfloat16()
    logits = (torch.randn(rows, NUM_EXPERTS, generator=gen, device=DEV) * 3.0).float()
    if case == "random":
        return logits, x
    if case == "none_local":
        logits[:, OFFSET : OFFSET + E_LOCAL] = -30.0
        return logits, x
    order = [OFFSET + i for i in torch.randperm(E_LOCAL, generator=cpu).tolist()]
    if case == "16_local_disjoint":
        assert rows * TOP_K <= E_LOCAL
        chosen = [order[TOP_K * t : TOP_K * (t + 1)] for t in range(rows)]
    elif case == "group_cap":
        assert rows == WIDE_MAX
        # Experts are dealt in order of their count to the tokens with the most free slots, so every token gets 16
        # distinct experts: 100 x 9 + 124 x 1 = 1024 pairs.
        free = [TOP_K] * rows
        chosen = [[] for _ in range(rows)]
        for i, e in enumerate(order):
            need = 9 if i < 100 else 1
            for t in sorted(range(rows), key=lambda tok: (-free[tok], tok))[:need]:
                chosen[t].append(e)
                free[t] -= 1
        assert all(f == 0 for f in free)
    else:
        raise ValueError(case)
    return _chosen_logits(chosen, gen), x


def _draw(seed: int, rows: int):
    """Random router logits fp32 [rows, 896] and latent bf16 [rows, 3584] from a seed: the call sequences' inputs."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    logits = (torch.randn(rows, NUM_EXPERTS, generator=gen, device=DEV) * 3.0).float()
    return logits, torch.randn(rows, H, generator=gen, device=DEV).bfloat16()


def _zeros(rows: int):
    return (
        torch.zeros(rows, NUM_EXPERTS, device=DEV),
        torch.zeros(rows, H, dtype=torch.bfloat16, device=DEV),
    )


# ── the state objects the checks share ────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _decode():
    """The shared K3MoeState and its layers A, C (this rank's experts, two counter sets) and B (the rolled experts)."""
    proc, _, _ = _experts()
    state = K3MoeState(_device(), I_TP, E_LOCAL)
    return state, [state.layer(*_weights(p)) for p in (proc, _rolled(), proc)]


@functools.lru_cache(maxsize=None)
def _decode_b():
    """A second K3MoeState (its own scratch) with one layer over this rank's experts."""
    state = K3MoeState(_device(), I_TP, E_LOCAL)
    return state, state.layer(*_weights(_experts()[0]))


@functools.lru_cache(maxsize=None)
def _wide():
    """The shared K3MoeWideState and its layers A (this rank's experts) and B (the rolled experts)."""
    proc, _, _ = _experts()
    state = K3MoeWideState(_device(), I_TP, E_LOCAL)
    return state, [state.layer(*_weights(p)) for p in (proc, _rolled())]


def _experts_of(i: int):
    """The buffers behind layer ``i`` of _decode (A, B, C) and _wide (A, B)."""
    return (_experts()[0], _rolled(), _experts()[0])[i]


def _decode_call(layer, logits, x):
    return k3_moe(x, logits, _bias(), OFFSET, RSF, layer)


def _wide_call(layer, logits, x, out=None):
    return k3_moe_wide(x, logits, _bias(), OFFSET, RSF, layer, out=out)


def _armed(state, layers) -> bool:
    """The slab armed and every layer's counters zero.

    Armed: every intermediate value byte 0x80 (FP8 -0.0), bytes 0-3 of every 16-byte scale group 0xFF (E8M0 NaN) and
    bytes 4-15 zero.
    """
    mod = state.mod
    cs = state.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
    return (
        bool((state.c == -128).all())
        and bool((cs[..., :4] == -1).all())
        and bool((cs[..., 4:] == 0).all())
        and all(bool((layer.counters == 0).all()) for layer in layers)
    )


def _snapshot(state, layers):
    return [
        t.clone() for t in (state.c, state.cs, state.part, *(layer.counters for layer in layers))
    ]


# ── references ────────────────────────────────────────────────────────────


def _deq_w(packed, sf):
    lut = torch.tensor(_E2M1, device=packed.device)
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, device=packed.device)
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * torch.exp2(sf.float() - 127.0).repeat_interleave(SV, dim=1)


def _deq_x(x_fp8, x_sf):
    rows, k = x_fp8.shape
    scale = torch.exp2(x_sf.reshape(rows, k // SV).float() - 127.0).repeat_interleave(SV, dim=1)
    return x_fp8.float() * scale


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
    """fp64 routed MoE over this rank's experts from the checkpoint tensors.

    SiTU, the MXFP8 intermediate, the down projection, the routing-weighted sum.
    """
    out = torch.zeros(x_deq.shape[0], H, device=DEV)
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


def _stock(experts, bias, x, logits):
    """The model's base path for these experts: the fused C++ route + MXFP8 quantization, then the TRTLLM-Gen runner.

    Pre-routed, SiTU with this entry's caps. Returns (y, ids, weights, x_fp8, x_sf).
    """
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    ids, w, x_fp8, x_sf = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, bias, x, RSF)
    alpha = torch.full((E_LOCAL,), GATE_CAP, dtype=torch.float32, device=DEV)
    beta = torch.full((E_LOCAL,), LINEAR_CAP, dtype=torch.float32, device=DEV)
    y = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
        None,
        None,
        x_fp8,
        x_sf.view(-1),
        experts["w31"],
        experts["w31s"],
        None,
        alpha,
        beta,
        None,
        experts["w2"],
        experts["w2s"],
        None,
        NUM_EXPERTS,
        TOP_K,
        1,
        1,
        I_TP,
        H,
        I_TP,
        OFFSET,
        E_LOCAL,
        1.0,
        int(RoutingMethodType.DeepSeekV3),
        int(ActType_TrtllmGen.SiTu),
        topk_weights=w,
        topk_ids=ids,
    )
    return y, ids, w, x_fp8, x_sf


def _local_pairs(ids) -> int:
    return int(((ids >= OFFSET) & (ids < OFFSET + E_LOCAL)).sum())


def _groups_wide(ids) -> int:
    """The wide build's groups for these ids: over this rank's experts, ceil(tokens / 8)."""
    local = ids[(ids >= OFFSET) & (ids < OFFSET + E_LOCAL)]
    counts = torch.bincount(local - OFFSET, minlength=E_LOCAL)
    return int(((counts + 7) // 8).sum())


def _check_vs_stock(y, experts, logits, x, where: str) -> None:
    """Check y against the stock path on the same inputs (op-catalog gates); zeros when no expert here is routed."""
    y_stock, ids, *_ = _stock(experts, _bias(), x, logits)
    if _local_pairs(ids) == 0:
        assert bool((y == 0).all()), f"{where}: no local expert is routed, y is not zero"
        return
    c = _compare(y, y_stock)
    assert c["ok"], f"{where}: against the stock path {c}"


def _alone(fn, layer, logits, x, experts, where: str):
    """One call made alone (synchronized before and after), checked against the stock path."""
    torch.cuda.synchronize()
    y = fn(layer, logits, x)
    torch.cuda.synchronize()
    _check_vs_stock(y, experts, logits, x, where)
    return y


# ── checks ────────────────────────────────────────────────────────────────


def test_state_armed_and_sized():
    """New state objects are armed and sized as the contract states for 224 local experts and intermediate 768.

    K3MoeState: slab [128, 8, 768] FP8 codes and [128, 8, 96] scale bytes armed, FC2 partial rows fp32 [1024, 3584]
    zero, not compiled; a layer's counters int32 [288] zero. K3MoeWideState: slab [324, 8, 768] and [324, 8, 96]
    armed, partials fp32 [1024, 3584]; a layer's counters int32 [680] zero. The loader's buffers fit the kernels.
    """
    proc, _, _ = _experts()
    ok, why = is_supported(*_weights(proc), E_LOCAL)
    assert ok, why
    state = K3MoeState(_device(), I_TP, E_LOCAL)
    layer = state.layer(*_weights(proc))
    g = min(E_LOCAL, DECODE_MAX * TOP_K)
    assert state.mod.G_CAP == g == 128
    assert state.c.dtype == state.cs.dtype == torch.int8
    assert tuple(state.c.shape) == (g, 8, I_TP) and tuple(state.cs.shape) == (g, 8, I_TP // 8)
    assert state.part.dtype == torch.float32 and tuple(state.part.shape) == (8 * g, H)
    assert bool((state.part == 0).all())
    assert layer.counters.dtype == torch.int32 and tuple(layer.counters.shape) == (32 + 2 * g,)
    assert _armed(state, [layer]) and not state.head_flags and state.compiled is None
    assert K3MoeState(_device(), I_TP, E_LOCAL, head_flags=True).head_flags

    wide = K3MoeWideState(_device(), I_TP, E_LOCAL)
    wide_layer = wide.layer(*_weights(proc))
    gw = E_LOCAL + (WIDE_MAX * TOP_K - E_LOCAL) // 8
    assert wide.mod.G_CAP == gw == 324
    assert tuple(wide.c.shape) == (gw, 8, I_TP) and tuple(wide.cs.shape) == (gw, 8, I_TP // 8)
    assert wide.part.dtype == torch.float32 and tuple(wide.part.shape) == (16 * WIDE_MAX, H)
    assert tuple(wide_layer.counters.shape) == (32 + 2 * gw,)
    assert _armed(wide, [wide_layer]) and wide.compiled is None


@pytest.mark.parametrize("m", ROUTE_M)
def test_k3_route_quant_is_the_stock_routing(m):
    """k3_route_quant (inside k3_moe and k3_moe_wide) returns kimi_k3_noaux_tc_mxfp8_quant's four outputs, bit for bit.

    Top-16 ids, routing weights, MXFP8 codes and scales, with and without the early dependent trigger.
    """
    logits64, x64 = _tokens("random", WIDE_MAX)
    logits, x = logits64[:m].contiguous(), x64[:m].contiguous()
    want = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, _bias(), x, RSF)
    for early in (False, True):
        got = torch.ops.trtllm.k3_route_quant(logits, _bias(), x, RSF, early_trigger=early)
        same = [_same(a, b) for a, b in zip(got, want)]
        print(f"OPCHECK op=k3_route_quant M={m} early_trigger={early} same(ids,w,q,sf)={same}")
        assert all(same), f"M {m} early_trigger {early}: {same}"


@pytest.mark.parametrize("m", range(1, DECODE_MAX + 1))
@pytest.mark.parametrize("case", DECODE_CASES)
def test_k3_moe_single_call(case, m):
    """One k3_moe call of M tokens on layer A against the fp64 reference and the stock path.

    Also: run-to-run identical bits; each row within one bf16 ulp of the same row of the 8-token call (the slice FC2
    adds a token's expert terms in slices whose bounds follow the step's group count; bit-identity is reported); the
    slab armed and every counter zero after the call; at 16_local_disjoint M 8 the group count at the capacity, 128.
    """
    proc, raw, bias = _experts()
    state, layers = _decode()
    logits8, x8 = _tokens(case, DECODE_MAX)
    logits, x = logits8[:m].contiguous(), x8[:m].contiguous()
    y = _decode_call(layers[0], logits, x)
    torch.cuda.synchronize()
    rearmed = _armed(state, layers)
    assert (
        y.shape == (m, H)
        and y.dtype == torch.bfloat16
        and y.is_contiguous()
        and y.device == x.device
    )
    det = all(_same(_decode_call(layers[0], logits, x), y) for _ in range(2))
    y8 = _decode_call(layers[0], logits8, x8)
    ulp_m8 = _max_ulp(y, y8[:m])
    y_stock, ids, w, x_fp8, x_sf = _stock(proc, bias, x, logits)
    local = _local_pairs(ids)
    groups = int(torch.unique(ids[(ids >= OFFSET) & (ids < OFFSET + E_LOCAL)]).numel())
    if case == "16_local_disjoint":
        assert local == TOP_K * m and groups == TOP_K * m
        if m == DECODE_MAX:
            assert groups == state.mod.G_CAP
    if local == 0:
        zeros = bool((y == 0).all())
        print(
            f"OPCHECK op=k3_moe case={case} M={m} zeros={zeros} det={det} "
            f"rows_as_m8={_same(y, y8[:m])} scratch_rearmed={rearmed}"
        )
        assert zeros and det and ulp_m8 == 0 and rearmed
        return
    ref = _reference(raw, _deq_x(x_fp8, x_sf), ids, w).bfloat16()
    c_ref, c_stock, c_stock_ref = _compare(y, ref), _compare(y, y_stock), _compare(y_stock, ref)
    print(
        f"OPCHECK op=k3_moe case={case} M={m} local_pairs={local} groups={groups} "
        f"vs_ref_elt_ulp={c_ref['elt_ulp']:.2f} vs_ref_rms_ulp={c_ref['rms_ulp']:.2f} "
        f"vs_stock_elt_ulp={c_stock['elt_ulp']:.2f} vs_stock_rms_ulp={c_stock['rms_ulp']:.2f} "
        f"stock_vs_ref_elt_ulp={c_stock_ref['elt_ulp']:.2f} det={det} "
        f"rows_as_m8={_same(y, y8[:m])} max_ulp_vs_m8={ulp_m8} scratch_rearmed={rearmed}"
    )
    assert c_ref["ok"] and c_stock["ok"] and c_stock_ref["ok"]
    assert det and ulp_m8 <= 1 and rearmed


@pytest.mark.parametrize("m", WIDE_M)
@pytest.mark.parametrize("case", WIDE_CASES)
def test_k3_moe_wide_single_call(case, m):
    """One k3_moe_wide call of M tokens on its layer A against the fp64 reference and the stock path.

    Also: run-to-run identical bits; the slab armed and every counter zero after the call; at M <= 8 within one bf16
    ulp of k3_moe on the same experts (bit-identity reported); at group_cap M 64 the group count at the capacity, 324.
    """
    proc, raw, bias = _experts()
    state, layers = _wide()
    _, decode_layers = _decode()
    logits64, x64 = _tokens(case, WIDE_MAX)
    logits, x = logits64[:m].contiguous(), x64[:m].contiguous()
    y = _wide_call(layers[0], logits, x)
    torch.cuda.synchronize()
    rearmed = _armed(state, layers)
    assert y.shape == (m, H) and y.dtype == torch.bfloat16 and y.is_contiguous()
    det = all(_same(_wide_call(layers[0], logits, x), y) for _ in range(2))
    y_stock, ids, w, x_fp8, x_sf = _stock(proc, bias, x, logits)
    groups = _groups_wide(ids)
    if case == "group_cap" and m == WIDE_MAX:
        assert groups == state.mod.G_CAP == 324
    decode = ""
    if m <= DECODE_MAX:
        y_dec = _decode_call(decode_layers[0], logits, x)
        ulp_dec = _max_ulp(y, y_dec)
        decode = f" max_ulp_vs_k3_moe={ulp_dec} bits_as_k3_moe={_same(y, y_dec)}"
        assert ulp_dec <= 1
    local = _local_pairs(ids)
    if local == 0:
        zeros = bool((y == 0).all())
        print(
            f"OPCHECK op=k3_moe_wide case={case} M={m} zeros={zeros} det={det} "
            f"scratch_rearmed={rearmed}{decode}"
        )
        assert zeros and det and rearmed
        return
    ref = _reference(raw, _deq_x(x_fp8, x_sf), ids, w).bfloat16()
    c_ref, c_stock, c_stock_ref = _compare(y, ref), _compare(y, y_stock), _compare(y_stock, ref)
    print(
        f"OPCHECK op=k3_moe_wide case={case} M={m} local_pairs={local} groups={groups} "
        f"vs_ref_elt_ulp={c_ref['elt_ulp']:.2f} vs_ref_rms_ulp={c_ref['rms_ulp']:.2f} "
        f"vs_stock_elt_ulp={c_stock['elt_ulp']:.2f} vs_stock_rms_ulp={c_stock['rms_ulp']:.2f} "
        f"stock_vs_ref_elt_ulp={c_stock_ref['elt_ulp']:.2f} det={det} scratch_rearmed={rearmed}{decode}"
    )
    assert c_ref["ok"] and c_stock["ok"] and c_stock_ref["ok"]
    assert det and rearmed


@pytest.mark.parametrize("m", [9, WIDE_MAX])
def test_k3_moe_wide_out_buffer(m):
    """k3_moe_wide with ``out``: the result is out[:M] (the same storage), the bits of the call without ``out``.

    The rows of ``out`` past M keep their bits.
    """
    _, layers = _wide()
    logits, x = _draw(900 + m, m)
    want = _wide_call(layers[0], logits, x)
    out = torch.full((WIDE_MAX + 3, H), -3.0, dtype=torch.bfloat16, device=DEV)
    tail = out[m:].clone()
    got = _wide_call(layers[0], logits, x, out=out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr() and got.shape == (m, H)
    assert _same(got, want) and _same(out[m:], tail)


def test_layers_by_steps_dip_and_regrow():
    """Decode steps of several layers on one state, the token count dipping and growing back, back to back.

    Steps of three k3_moe calls (layers A, B, C of one state) at M 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, then steps of two
    k3_moe_wide calls (layers A, B of the wide state) at M 64, 64, 16, 1, 40, 64, 8, 23, 64, 9, 64, with new inputs
    every call. Each call is first made alone (synchronized, checked against the stock path); in the sequence, back to
    back on one stream, each returns the bits of its call alone, and afterwards both slabs are armed and every counter
    is zero: a call after a smaller one reads nothing an older, larger call left.
    """
    state, layers = _decode()
    calls = [(i, *_draw(1000 + 10 * s + i, m)) for s, m in enumerate(DIP_STEPS) for i in range(3)]
    alone = [
        _alone(
            _decode_call, layers[i], lg, x, _experts_of(i), f"k3_moe layer {i} M {x.shape[0]} alone"
        )
        for i, lg, x in calls
    ]
    seq = [_decode_call(layers[i], lg, x) for i, lg, x in calls]
    torch.cuda.synchronize()
    bad = [k for k, (y, a) in enumerate(zip(seq, alone)) if not _same(y, a)]
    assert not bad, f"k3_moe calls {bad} of the sequence differ from the same calls alone"
    assert _armed(state, layers)

    wide, wide_layers = _wide()
    wide_calls = [
        (i, *_draw(2000 + 10 * s + i, m)) for s, m in enumerate(WIDE_DIP_STEPS) for i in range(2)
    ]
    alone = [
        _alone(
            _wide_call,
            wide_layers[i],
            lg,
            x,
            _experts_of(i),
            f"k3_moe_wide layer {i} M {x.shape[0]} alone",
        )
        for i, lg, x in wide_calls
    ]
    seq = [_wide_call(wide_layers[i], lg, x) for i, lg, x in wide_calls]
    torch.cuda.synchronize()
    bad = [k for k, (y, a) in enumerate(zip(seq, alone)) if not _same(y, a)]
    assert not bad, f"k3_moe_wide calls {bad} of the sequence differ from the same calls alone"
    assert _armed(wide, wide_layers)


def test_two_states_interleaved():
    """Two K3MoeStates (each its own scratch) and the wide state, calls interleaved in an irregular pattern.

    A A B A B B A A A B, twice, with a k3_moe_wide call after every third, M varying, back to back on one stream: every
    call returns the bits of the same call alone (checked against the stock path), and every slab is armed and every
    counter zero afterwards.
    """
    state_a, layers_a = _decode()
    state_b, layer_b = _decode_b()
    wide, wide_layers = _wide()
    plan = []
    for i, which in enumerate("AABABBAAAB" * 2):
        m = (3, 8, 1, 8, 5)[i % 5]
        if which == "A":
            plan.append((_decode_call, layers_a[i % 3], _experts_of(i % 3), *_draw(3000 + i, m)))
        else:
            plan.append((_decode_call, layer_b, _experts()[0], *_draw(3000 + i, m)))
        if i % 3 == 2:
            mw = (40, 64, 9)[(i // 3) % 3]
            plan.append((_wide_call, wide_layers[i % 2], _experts_of(i % 2), *_draw(3500 + i, mw)))
    alone = [
        _alone(fn, layer, lg, x, ex, f"interleaved call {k} alone")
        for k, (fn, layer, ex, lg, x) in enumerate(plan)
    ]
    seq = [fn(layer, lg, x) for fn, layer, _, lg, x in plan]
    torch.cuda.synchronize()
    bad = [k for k, (y, a) in enumerate(zip(seq, alone)) if not _same(y, a)]
    assert not bad, f"interleaved calls {bad} differ from the same calls alone"
    assert _armed(state_a, layers_a) and _armed(state_b, [layer_b]) and _armed(wide, wide_layers)


def test_graph_capture_and_replay():
    """A captured step replayed with rewritten inputs, eager calls of other token counts between replays.

    The step: k3_moe on layers A, B, C at M 8, then k3_moe_wide on its layer A at M 64, captured once and replayed six
    times with new inputs copied into its static buffers; between replays an eager k3_moe call (M 3, 1, 6, 5, 2, 7)
    and an eager k3_moe_wide call (M 23, 9, 40, 1, 64, 16) on the same states. Every replayed and eager call returns
    the bits of the same call alone (checked against the stock path), and the slabs are armed and every counter zero
    afterwards.
    """
    state, layers = _decode()
    wide, wide_layers = _wide()
    static = [_draw(4000 + i, DECODE_MAX) for i in range(3)] + [_draw(4003, WIDE_MAX)]

    def step():
        outs = [_decode_call(layers[i], *static[i]) for i in range(3)]
        return outs + [_wide_call(wide_layers[0], *static[3])]

    step()  # every first call eager: k3_route_quant and both k3_moe builds compile here if nothing has yet
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    for rep in range(REPLAYS):
        inputs = [_draw(5000 + 10 * rep + i, DECODE_MAX) for i in range(3)] + [
            _draw(5003 + 10 * rep, WIDE_MAX)
        ]
        alone = [
            _alone(
                _decode_call, layers[i], *inputs[i], _experts_of(i), f"replay {rep} layer {i} alone"
            )
            for i in range(3)
        ]
        alone.append(
            _alone(
                _wide_call, wide_layers[0], *inputs[3], _experts_of(0), f"replay {rep} wide alone"
            )
        )
        eager = (
            _draw(6000 + rep, (3, 1, 6, 5, 2, 7)[rep]),
            _draw(6100 + rep, (23, 9, 40, 1, 64, 16)[rep]),
        )
        eager_layers = (layers[rep % 3], wide_layers[rep % 2])
        eager_alone = (
            _alone(
                _decode_call, eager_layers[0], *eager[0], _experts_of(rep % 3), f"eager {rep} alone"
            ),
            _alone(
                _wide_call,
                eager_layers[1],
                *eager[1],
                _experts_of(rep % 2),
                f"eager wide {rep} alone",
            ),
        )
        for (lg, x), (new_lg, new_x) in zip(static, inputs):
            lg.copy_(new_lg)
            x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        bad = [i for i, (y, a) in enumerate(zip(outs, alone)) if not _same(y, a)]
        assert not bad, f"replay {rep}: calls {bad} differ from the same calls alone"
        got = (_decode_call(eager_layers[0], *eager[0]), _wide_call(eager_layers[1], *eager[1]))
        torch.cuda.synchronize()
        assert all(_same(g, a) for g, a in zip(got, eager_alone)), f"eager calls after replay {rep}"
    del graph
    assert _armed(state, layers) and _armed(wide, wide_layers)


def test_unsupported_calls_refused_before_launch():
    """Unsupported calls raise ValueError before any launch, and the next call is correct.

    k3_moe at M 0 and 9, k3_moe_wide at M 0 and 65, and k3_moe on a head_flags state's layer (that build takes the
    front's ready words: k3_moe_fused_front only). The slabs, the FC2 partial rows and every counter keep their bits;
    the next calls return the bits of the same calls made before.
    """
    state, layers = _decode()
    wide, wide_layers = _wide()
    lg8, x8 = _draw(7000, DECODE_MAX)
    lg40, x40 = _draw(7001, 40)
    want = _decode_call(layers[0], lg8, x8)
    want_wide = _wide_call(wide_layers[0], lg40, x40)
    flags_state = K3MoeState(_device(), I_TP, E_LOCAL, head_flags=True)
    flags_layer = flags_state.layer(*_weights(_experts()[0]))
    torch.cuda.synchronize()
    objects = ((state, layers), (wide, wide_layers), (flags_state, [flags_layer]))
    before = [_snapshot(st, lyrs) for st, lyrs in objects]
    for m in (0, DECODE_MAX + 1):
        with pytest.raises(ValueError):
            _decode_call(layers[0], *_zeros(m))
    for m in (0, WIDE_MAX + 1):
        with pytest.raises(ValueError):
            _wide_call(wide_layers[0], *_zeros(m))
    with pytest.raises(ValueError, match="head_flags"):
        _decode_call(flags_layer, lg8, x8)
    torch.cuda.synchronize()
    after = [_snapshot(st, lyrs) for st, lyrs in objects]
    assert all(_same(a, b) for snap_a, snap_b in zip(before, after) for a, b in zip(snap_a, snap_b))
    assert flags_state.compiled is None
    assert _same(_decode_call(layers[0], lg8, x8), want)
    assert _same(_wide_call(wide_layers[0], lg40, x40), want_wide)


def test_construction_and_first_call_refuse_capture():
    """Under CUDA-graph capture the states refuse to allocate, and a new state refuses its first (compiling) call.

    K3MoeState(), K3MoeState.layer(), K3MoeWideState() and K3MoeWideState.layer() raise RuntimeError; the first call
    on a new K3MoeState and on a new K3MoeWideState raises RuntimeError before the k3_moe launch (k3_route_quant,
    compiled eagerly first, is captured and discarded with the graph); neither state is compiled afterwards.
    """
    proc, _, bias = _experts()
    fresh = K3MoeState(_device(), I_TP, E_LOCAL)
    fresh_layer = fresh.layer(*_weights(proc))
    fresh_wide = K3MoeWideState(_device(), I_TP, E_LOCAL)
    fresh_wide_layer = fresh_wide.layer(*_weights(proc))
    lg4, x4 = _draw(7100, 4)
    lg16, x16 = _draw(7101, 16)
    # Both k3_route_quant builds compiled, whichever PDL setting the layers use.
    for early in (False, True):
        torch.ops.trtllm.k3_route_quant(lg4, bias, x4, RSF, early_trigger=early)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            K3MoeState(_device(), I_TP, E_LOCAL)
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            fresh.layer(*_weights(proc))
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            K3MoeWideState(_device(), I_TP, E_LOCAL)
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            fresh_wide.layer(*_weights(proc))
        with pytest.raises(RuntimeError, match="compiles on its first call"):
            _decode_call(fresh_layer, lg4, x4)
        with pytest.raises(RuntimeError, match="compiles on its first call"):
            _wide_call(fresh_wide_layer, lg16, x16)
    del graph
    assert fresh.compiled is None and fresh_wide.compiled is None


def test_negative_control_weights_rebound_after_layer():
    """Negative control: a layer reads the weight buffers it was built over; rebinding the weights is not seen.

    Layers of both builds are built over one set of experts (E1, a copy of this rank's). The weights are then
    "reloaded" by rebinding to new tensors (E2: the experts rolled by one), as a loader that replaces its parameters
    would. Nothing raises, and each layer still returns E1's partial bit for bit, outside the op-catalog gates of the
    stock path on E2: silently stale. Layers built over E2 match its stock path; and once E2 is copied into E1's
    buffers in place, the stale layers return E2's partial bit for bit.
    """
    proc, _, bias = _experts()
    state, _ = _decode()
    wide, _ = _wide()
    e1 = {name: t.clone() for name, t in proc.items()}
    stale = state.layer(*_weights(e1))
    stale_wide = wide.layer(*_weights(e1))
    lg8, x8 = _tokens("random", DECODE_MAX)
    lg40, x40 = _draw(8000, 40)
    y_e1 = _decode_call(stale, lg8, x8)
    yw_e1 = _wide_call(stale_wide, lg40, x40)
    torch.cuda.synchronize()

    # The reload: the caller's weights are now these new tensors; E1's buffers stay as they were.
    e2 = _rolled()
    y_stale = _decode_call(stale, lg8, x8)
    yw_stale = _wide_call(stale_wide, lg40, x40)
    stock_e2 = _stock(e2, bias, x8, lg8)[0]
    stock_wide_e2 = _stock(e2, bias, x40, lg40)[0]
    c, cw = _compare(y_stale, stock_e2), _compare(yw_stale, stock_wide_e2)
    print(
        f"OPCHECK op=k3_moe case=weights_rebound stale_bits_as_e1={_same(y_stale, y_e1)} "
        f"stale_vs_e2_stock_elt_ulp={c['elt_ulp']:.1f} wide_stale_bits_as_e1={_same(yw_stale, yw_e1)} "
        f"wide_stale_vs_e2_stock_elt_ulp={cw['elt_ulp']:.1f}"
    )
    assert _same(y_stale, y_e1) and not c["ok"]
    assert _same(yw_stale, yw_e1) and not cw["ok"]

    fresh = state.layer(*_weights(e2))
    fresh_wide = wide.layer(*_weights(e2))
    y_e2 = _decode_call(fresh, lg8, x8)
    yw_e2 = _wide_call(fresh_wide, lg40, x40)
    assert _compare(y_e2, stock_e2)["ok"] and _compare(yw_e2, stock_wide_e2)["ok"]
    for name, t in e1.items():
        t.copy_(e2[name])  # in place: the buffers the stale layers read now hold E2
    assert _same(_decode_call(stale, lg8, x8), y_e2)
    assert _same(_wide_call(stale_wide, lg40, x40), yw_e2)
