# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the moe/k3_moe_m1 and moe/k3_moe_m2 tests: a rank's TP16 experts through TRT-LLM's loader,
routed tokens, the fp32 reference, and the checks both stateful entries run over their caller-owned state (single
calls, call sequences across layers and steps, capture and replay, two states interleaved, the epochs across the
int32 wrap, the explicit constructor).

Imported by the two test files beside it (this tree is not a package).
"""

import functools
import math
from types import SimpleNamespace

import torch

H, NUM_EXPERTS, TOP_K, SV = 3584, 896, 16, 32
# TP16: every expert on the rank, its 192-wide intermediate slice zero-padded to whole tiles (256) by the loader.
I_TP, I_PAD, MOE_TP = 192, 256, 16
GATE_CAP, LINEAR_CAP = (
    4.0,
    25.0,
)  # the SiTU caps (activation_situ_beta, activation_situ_linear_beta)
RSF = 2.827
ULP = 2.0**-8
E4M3_MAX = 448.0
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
LAYERS = 3  # distinct weight sets; the sequences cycle through them
TOKEN_SETS = 4
STEPS = 12


def is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def _rand_mxfp4(rows, k, k_full, gen):
    """Random checkpoint-format MXFP4: packed [rows, k / 2] (low nibble = even k), E8M0 per 32 k, scaled so a
    k_full-long dot product lands near std 3."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k_full))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


@functools.lru_cache(maxsize=None)
def experts(seed: int = 20260928):
    """Rank 0's TP16 experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader (the buffers the engines read,
    the 192-wide shard sliced from tensors that hold exactly it, then padded to 256) and the checkpoint slices the
    reference reads."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=MOE_TP, tp_rank=0, scaling_vector_size=SV, intermediate_size=I_TP * MOE_TP,
                             intermediate_size_per_partition=I_TP, hidden_size=H)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = (
        torch.zeros(NUM_EXPERTS, 2 * I_PAD, H // 2, **kw),
        torch.zeros(NUM_EXPERTS, 2 * I_PAD, H // SV, **kw),
        torch.zeros(NUM_EXPERTS, H, I_PAD // 2, **kw),
        torch.zeros(NUM_EXPERTS, H, I_PAD // SV, **kw),
    )
    raw = {name: [] for name in ("up", "up_s", "gate", "gate_s", "down", "down_s")}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for e in range(NUM_EXPERTS):
        gate, gate_s = _rand_mxfp4(I_TP, H, H, gen)
        up, up_s = _rand_mxfp4(I_TP, H, H, gen)
        down, down_s = _rand_mxfp4(H, I_TP, I_TP * MOE_TP, gen)
        method.load_expert_w3_w1_weight(module, gate, up, proc[0][e])
        method.load_expert_w2_weight(module, down, proc[2][e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, gate_s, up_s, proc[1][e])
        method.load_expert_w2_weight_scale_mxfp4(module, down_s, proc[3][e])
        for name, t in zip(raw, (up, up_s, gate, gate_s, down, down_s)):
            raw[name].append(t)
    torch.cuda.synchronize()
    return proc, raw


@functools.lru_cache(maxsize=None)
def layer_weights(index: int):
    """The buffers of layer ``index``: layer 0's experts in another order (a different layer to the engine)."""
    proc, _ = experts()
    return tuple(t.roll(index, dims=0).contiguous() for t in proc) if index else proc


def routed(m: int, seed: int):
    """m tokens' MXFP8 latents and routing, as trtllm::k3_route_quant produces them (the engines' caller)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op as _rq  # noqa: F401

    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(m, NUM_EXPERTS, generator=gen, device="cuda") * 3.0).float()
    x = torch.randn(m, H, generator=gen, device="cuda").bfloat16()
    bias = (torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.05).float()
    ids, weights, x_fp8, x_sf = torch.ops.trtllm.k3_route_quant(logits, bias, x, RSF, True)
    return x_fp8, x_sf, ids, weights


def _deq_w(packed, sf):
    lut = torch.tensor(_E2M1, device=packed.device)
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, device=packed.device)
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * torch.exp2(sf.float() - 127.0).repeat_interleave(SV, dim=1)


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


def reference(x_fp8, x_sf, ids, weights):
    """fp32 routed MoE over layer 0's experts from the checkpoint slices (f64 GEMMs): SiTU, the MXFP8 intermediate,
    the down projection, the routing-weighted sum, per token; bf16 out."""
    _, raw = experts()
    rows = x_fp8.shape[0]
    x = x_fp8.float() * torch.exp2(x_sf.reshape(rows, H // SV).float() - 127.0).repeat_interleave(
        SV, dim=1
    )
    out = torch.zeros(rows, H, device="cuda")
    for t in range(rows):
        for k in range(TOP_K):
            e = int(ids[t, k])
            xe = x[t : t + 1].double()
            up = (xe @ _deq_w(raw["up"][e], raw["up_s"][e]).double().t()).float()
            gate = (xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).double().t()).float()
            act = (GATE_CAP * torch.tanh(gate / GATE_CAP) * torch.sigmoid(gate)
                   * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP)))  # fmt: skip
            y = (
                _requant(act).double() @ _deq_w(raw["down"][e], raw["down_s"][e]).double().t()
            ).float()
            out[t] += (y * weights[t, k].float())[0]
    return out.bfloat16()


def row_ulp(y, ref):
    """Largest |y - ref| in bf16 ulps of the row's max |ref|, and the relative RMS in ulps."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    return elt, rms


class Engine:
    """One entry at one token count: its state constructor and wrapper, and how to read the state's workspace (the
    epochs every CTA advances, and the count words the next call uses, which every call leaves zero)."""

    def __init__(self, name: str, m: int):
        self.name, self.m = name, m

    def create(self, push=()):
        from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

        device = torch.device("cuda", torch.cuda.current_device())
        if self.name == "k3_moe_m1":
            return op.K3MoeM1State.create(
                device, I_TP, I_PAD, NUM_EXPERTS, num_tokens=self.m, push=push
            )
        return op.K3MoeM2State.create(device, I_TP, I_PAD, NUM_EXPERTS, push=push)

    def call(self, layer, tokens, out=None):
        from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe import k3_moe_m1, k3_moe_m2

        entry = k3_moe_m1.k3_moe_m1 if self.name == "k3_moe_m1" else k3_moe_m2.k3_moe_m2
        return entry(*tokens, 0, layer, out)

    def next_counts(self, state) -> torch.Tensor:
        """The count words the next call uses (by the parity of CTA 0's epoch)."""
        ep = int(state.epochs[0].item())
        if self.name == "k3_moe_m1":
            return state.counts[:2][ep & 1 : (ep & 1) + 1]
        words = state.mod.GROUPS2 * state.mod.CW
        return state.counts.view(2, words)[ep & 1]


@functools.lru_cache(maxsize=None)
def _reference_calls(name: str, m: int):
    """Every (layer, token set) call once, in order, on a reference state of its own: the bits each call must give
    in any sequence on any state."""
    engine = Engine(name, m)
    state = engine.create()
    out = {}
    for li in range(LAYERS):
        layer = state.layer(*layer_weights(li))
        for si in range(TOKEN_SETS):
            out[li, si] = engine.call(layer, routed(m, 100 + si))
    torch.cuda.synchronize()
    return out


def _alone(engine):
    return _reference_calls(engine.name, engine.m)


def check_single_calls(engine, seeds=range(8)):
    """Single calls over the grid against the fp32 reference (8 ulp of the row max per element, 4 ulp relative
    RMS), and run to run bit for bit."""
    state = engine.create()
    layer = state.layer(*layer_weights(0))
    for seed in seeds:
        tokens = routed(engine.m, seed)
        y = engine.call(layer, tokens)
        again = engine.call(layer, tokens)
        elt, rms = row_ulp(y, reference(*tokens))
        print(
            f"OPCHECK op={engine.name} M={engine.m} seed={seed} vs_ref_elt_ulp={elt:.2f} vs_ref_rms_ulp={rms:.2f}"
        )
        assert bool(torch.isfinite(y.float()).all())
        assert elt <= 8.0 and rms <= 4.0, (elt, rms)
        assert same(y, again), "run-to-run bits differ"


def check_call_sequences(engine):
    """12 steps of the 3 layers on one state, the token set changing every step: each call gives the bits of the
    same call on the reference state, the epochs count every call, and every call leaves the next call's count
    words zero."""
    alone = _alone(engine)
    state = engine.create()
    layers = [state.layer(*layer_weights(li)) for li in range(LAYERS)]
    calls = 0
    for step in range(STEPS):
        si = step % TOKEN_SETS
        for li in range(LAYERS):
            y = engine.call(layers[li], routed(engine.m, 100 + si))
            calls += 1
            assert same(y, alone[li, si]), (step, li)
            assert bool((engine.next_counts(state) == 0).all()), (step, li)
    assert bool((state.epochs == calls).all()), "every CTA's epoch counts every call"


def check_capture_replay(engine):
    """One step of the 3 layers captured on a created state (create compiled the build, so capture compiles nothing),
    replayed 4 times with rewritten inputs and an eager call on the same state between replays: every replayed and
    eager call gives the bits of the same call on the reference state."""
    alone = _alone(engine)
    state = engine.create()
    layers = [state.layer(*layer_weights(li)) for li in range(LAYERS)]
    static = [tuple(t.clone() for t in routed(engine.m, 100)) for _ in range(LAYERS)]
    outs = [torch.empty(engine.m, H, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for li in range(LAYERS):
            engine.call(layers[li], static[li], outs[li])
    for r in range(TOKEN_SETS):
        for li in range(LAYERS):
            for dst, src in zip(static[li], routed(engine.m, 100 + r)):
                dst.copy_(src)
        graph.replay()
        for li in range(LAYERS):
            assert same(outs[li], alone[li, r]), ("replay", r, li)
        eager_set = (r + 1) % TOKEN_SETS
        assert same(
            engine.call(layers[1], routed(engine.m, 100 + eager_set)), alone[1, eager_set]
        ), ("eager", r)


def check_two_states(engine):
    """Two states, each with its own workspace, called in an irregular order (A A B A B B A B A): every call gives the
    bits of the same call on the reference state, and each state's epochs count only its own calls."""
    alone = _alone(engine)
    states = {"A": engine.create(), "B": engine.create()}
    layers = {name: st.layer(*layer_weights(0)) for name, st in states.items()}
    counts = {"A": 0, "B": 0}
    for i, name in enumerate("AABABBABA"):
        si = i % TOKEN_SETS
        assert same(engine.call(layers[name], routed(engine.m, 100 + si)), alone[0, si]), (i, name)
        counts[name] += 1
    for name, st in states.items():
        assert bool((st.epochs == counts[name]).all()), name


def check_epoch_wrap(engine, start):
    """The CTAs' epochs (int32, + 1 per call; their parity picks the count words) across the int32 wrap: the state
    preset to where ~2^31 calls leave it (every epoch at ``start``, the next call's count words zero). Each call gives
    the bits of the same call on the reference state, the epochs wrap to -2^31 and keep counting, and every call
    leaves the next call's count words zero."""
    alone = _alone(engine)
    state = engine.create()
    layer = state.layer(*layer_weights(0))
    state.counts.zero_()
    state.epochs.fill_(start)
    for c in range(TOKEN_SETS + 1):
        si = c % TOKEN_SETS
        y = engine.call(layer, routed(engine.m, 100 + si))
        ep = (start + c + 1 + 2**31) % 2**32 - 2**31  # int32 two's complement
        assert same(y, alone[0, si]), c
        assert bool((state.epochs == ep).all()), (c, ep)
        assert bool((engine.next_counts(state) == 0).all()), (c, ep)


def check_create(engine):
    """``create`` refuses to run under capture, compiles the plain build and every push build it is given, and a
    created state's first call can be captured."""
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    raised = False
    with torch.cuda.stream(stream):
        try:
            with torch.cuda.graph(graph, stream=stream):
                engine.create()
        except RuntimeError as exc:
            raised = "before CUDA-graph capture" in str(exc)
    assert raised, "create must refuse to run under capture"
    state = engine.create(push=((4, 1), (16, 4)))
    assert state.compiled
    assert all(state.push_compiled(slots, copies) for slots, copies in ((4, 1), (16, 4)))
    layer = state.layer(*layer_weights(0))
    tokens = tuple(t.clone() for t in routed(engine.m, 100))
    out = torch.empty(engine.m, H, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        engine.call(layer, tokens, out)
    graph.replay()
    alone = _alone(engine)
    assert same(out, alone[0, 0])
