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
"""k3_moe_m2 (two decode tokens' routed experts as a weight-stream kernel) on one GPU, in the TP16 layout of the
routed experts (every rank holds all 896 experts, its 192-wide intermediate slice zero-padded to 256 by TRT-LLM's
loader). Weights are random checkpoint-format MXFP4 experts put through TRT-LLM's own loader. Per routing case,
against:
- trtllm::k3_moe (K3MoeLayer) on the same buffers and routing, which it reads at the
  padded size. The zero rows add exact
  zeros, so the two compute the same math; k3_moe_m2 keeps k3_moe's FC2 operand and combine order, so the bits match
  except where its two FC1 partial sums round an intermediate value differently. Bit identity is reported; the gate
  is one bf16 ulp of the row's max;
- the fp32 reference over the dequantized experts (op-catalog gates: 8 ulp of the row max per element, 4 ulp
  relative RMS).
The routing cases cover the tokens' experts overlapping as random routing does, fully shared (16 experts) and fully
disjoint (32, the most FC1 units). It also checks run-to-run identical bits, that calls of different layers
sharing the state's workspace, in any order, each give the bits of the same call alone, and that the same holds
across the epochs' int32 wrap.
"""

import functools
import math
from types import SimpleNamespace

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_moe_m2 needs sm_100")

H, NUM_EXPERTS, TOP_K, SV = 3584, 896, 16, 32
GATE_CAP, LINEAR_CAP = (
    4.0,
    25.0,
)  # the SiTU caps (activation_situ_beta, activation_situ_linear_beta)
RSF = 2.827
ULP = 2.0**-8
E4M3_MAX = 448.0
# Layout -> (moe_tp, the rank's logical intermediate, local experts, first local id).
LAYOUTS = {"tp16": (16, 192, 896, 0), "tp4ep4": (4, 768, 224, 224)}
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op as _rq  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


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
def _experts(layout: str, seed: int = 20260928):
    """A rank's experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader (the buffers both kernels read, padded
    as the loader pads an unaligned shard) and the rank's logical slices of the checkpoint tensors (what the reference
    reads). Only the rank's shard is generated: an unaligned shard as rank 0 of tensors that hold exactly that shard
    (the loader slices it, then pads), an aligned one as the whole tensor of a single rank."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    moe_tp, i_tp, e_local, _ = LAYOUTS[layout]
    i_pad = (i_tp + 127) // 128 * 128
    tp = moe_tp if i_pad != i_tp else 1
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=tp, tp_rank=0, scaling_vector_size=SV, intermediate_size=i_tp * tp,
                             intermediate_size_per_partition=i_tp, hidden_size=H)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = dict(
        w31=torch.zeros(e_local, 2 * i_pad, H // 2, **kw),
        w31s=torch.zeros(e_local, 2 * i_pad, H // SV, **kw),
        w2=torch.zeros(e_local, H, i_pad // 2, **kw),
        w2s=torch.zeros(e_local, H, i_pad // SV, **kw),
    )
    raw = {name: [] for name in ("up", "up_s", "gate", "gate_s", "down", "down_s")}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for e in range(e_local):
        w1, w1s = _rand_mxfp4(i_tp, H, H, gen)  # gate
        w3, w3s = _rand_mxfp4(i_tp, H, H, gen)  # up
        w2, w2s = _rand_mxfp4(H, i_tp, i_tp * moe_tp, gen)  # down
        method.load_expert_w3_w1_weight(module, w1, w3, proc["w31"][e])
        method.load_expert_w2_weight(module, w2, proc["w2"][e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, w1s, w3s, proc["w31s"][e])
        method.load_expert_w2_weight_scale_mxfp4(module, w2s, proc["w2s"][e])
        for name, t in zip(raw, (w3, w3s, w1, w1s, w2, w2s)):
            raw[name].append(t)
    torch.cuda.synchronize()
    gen_b = torch.Generator(device="cuda").manual_seed(seed + 1)
    bias = (torch.randn(NUM_EXPERTS, generator=gen_b, device="cuda") * 0.05).float()
    return proc, raw, bias


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


def _reference(layout, raw, x_deq, ids, weights):
    """fp32 routed MoE over this rank's experts from the checkpoint slices (f64 GEMMs): SiTU, the MXFP8
    intermediate, the down projection, the routing-weighted sum, per token."""
    _, _, e_local, offset = LAYOUTS[layout]
    out = torch.zeros(ids.shape[0], H, device="cuda")
    for t in range(ids.shape[0]):
        for k in range(TOP_K):
            e = int(ids[t, k]) - offset
            if not 0 <= e < e_local:
                continue
            xe = x_deq[t : t + 1].double()
            up = (xe @ _deq_w(raw["up"][e], raw["up_s"][e]).double().t()).float()
            gate = (xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).double().t()).float()
            act = (GATE_CAP * torch.tanh(gate / GATE_CAP) * torch.sigmoid(gate)
                   * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP)))  # fmt: skip
            y = (
                _requant(act).double() @ _deq_w(raw["down"][e], raw["down_s"][e]).double().t()
            ).float()
            out[t] += (y * weights[t, k].float())[0]
    return out


def _row_ulp(y, ref):
    """Largest |y - ref| in bf16 ulps of the row's max |ref|, and the relative RMS in ulps."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    return elt, rms


@functools.lru_cache(maxsize=None)
def _state(layout: str):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _, i_tp, e_local, _ = LAYOUTS[layout]
    return op.K3MoeM2State(
        torch.device("cuda", torch.cuda.current_device()), i_tp, (i_tp + 127) // 128 * 128, e_local
    )


@functools.lru_cache(maxsize=None)
def _k3_moe_state(layout: str):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _, i_tp, e_local, _ = LAYOUTS[layout]
    return op.K3MoeState(
        torch.device("cuda", torch.cuda.current_device()), (i_tp + 127) // 128 * 128, e_local
    )


def _k3_moe(layout, proc, ids, w, x_fp8, x_sf):
    """trtllm::k3_moe (K3MoeLayer) on the same buffers and routing."""
    _, _, _, offset = LAYOUTS[layout]
    layer = _k3_moe_state(layout).layer(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"])
    return layer(x_fp8, x_sf, ids, w, offset)


CASES = ["random", "shared", "disjoint"]


def _tokens(case: str, seed: int):
    """Two tokens' router logits and latent rows. "shared": both tokens route to the same 16 experts; "disjoint": to
    32 different experts."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(2, NUM_EXPERTS, generator=gen, device="cuda") * 3.0).float()
    if case == "shared":
        logits[1] = logits[0]
    elif case == "disjoint":
        top0 = torch.topk(logits[0], TOP_K).indices
        logits[1, top0] = -30.0
    x = torch.randn(2, H, generator=gen, device="cuda").bfloat16()
    return logits, x


def _m2(layout, proc, bias, logits, x):
    ops = _ops()
    _, _, _, offset = LAYOUTS[layout]
    ids, w, x_fp8, x_sf = ops.k3_route_quant(logits, bias, x, RSF, True)
    layer = _state(layout).layer(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"])
    return layer(x_fp8, x_sf, ids, w, offset), ids, w, x_fp8, x_sf


@pytest.mark.parametrize("case", CASES)
def test_k3_moe_m2(case):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _ops()
    layout = "tp16"  # k3_moe_m2 is for the TP16 slice (test_k3_moe_m2_refuses_wide_intermediate)
    moe_tp, i_tp, e_local, offset = LAYOUTS[layout]
    proc, raw, bias = _experts(layout)
    ok, why = op.m2_supported(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"], e_local, i_tp)
    assert ok, why
    n_bit = 0
    seeds = range(8)
    for seed in seeds:
        logits, x = _tokens(case, 100 * CASES.index(case) + seed)
        y, ids, w, x_fp8, x_sf = _m2(layout, proc, bias, logits, x)
        again, *_ = _m2(layout, proc, bias, logits, x)
        y_k3 = _k3_moe(layout, proc, ids, w, x_fp8, x_sf)
        torch.cuda.synchronize()
        experts = len(set(ids.view(-1).tolist()))
        assert torch.equal(_bits(y), _bits(again)), "run-to-run bits differ"
        ref = _reference(layout, raw, _deq_x(x_fp8, x_sf), ids, w).bfloat16()
        elt, rms = _row_ulp(y, ref)
        elt_k3, _ = _row_ulp(y, y_k3)
        same = torch.equal(_bits(y), _bits(y_k3))
        n_bit += same
        print(f"OPCHECK op=k3_moe_m2 layout={layout} case={case} seed={seed} experts={experts} "
              f"vs_ref_elt_ulp={elt:.2f} "
              f"vs_ref_rms_ulp={rms:.2f} vs_k3_moe_elt_ulp={elt_k3:.2f} bit_identical_to_k3_moe={same}")  # fmt: skip
        assert bool(torch.isfinite(y.float()).all())
        assert elt <= 8.0 and rms <= 4.0
        assert elt_k3 <= 1.0
    print(
        f"OPCHECK op=k3_moe_m2 layout={layout} case={case} bit_identical_to_k3_moe={n_bit}/{len(seeds)}"
    )


def test_k3_moe_m2_refuses_wide_intermediate():
    """k3_moe_m2 is for an intermediate of at most 256 (its FC2 tiles of every expert slot stay in shared memory):
    m2_supported refuses the TP4 x EP4 slice (768). Only metadata is read."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    _, i_tp, e_local, _ = LAYOUTS["tp4ep4"]
    kw = dict(dtype=torch.uint8, device="meta")
    ok, why = op.m2_supported(
        torch.empty(e_local, 2 * i_tp, H // 2, **kw),
        torch.empty(e_local, 2 * i_tp, H // SV, **kw),
        torch.empty(e_local, H, i_tp // 2, **kw),
        torch.empty(e_local, H, i_tp // SV, **kw),
        e_local,
        i_tp,
    )
    assert not ok and "256" in why, why


def test_k3_moe_m2_layers_interleaved():
    """Two layers (different weights) sharing one state, called in turns: each call gives the bits of the same call
    alone (the counter sets and epochs advance with every call)."""
    _ops()
    proc, _, bias = _experts("tp16")
    other = {
        k: v.roll(1, dims=0).contiguous() for k, v in proc.items()
    }  # the experts of a second layer
    tokens = [_tokens("random", 900 + s) for s in range(3)]
    alone = {}
    for name, weights in (("a", proc), ("b", other)):
        for t, (logits, x) in enumerate(tokens):
            alone[name, t] = _m2("tp16", weights, bias, logits, x)[0]
    seq = []
    for t, (logits, x) in enumerate(tokens):
        for name, weights in (("b", other), ("a", proc)):
            seq.append((name, t, _m2("tp16", weights, bias, logits, x)[0]))
    torch.cuda.synchronize()
    for name, t, y in seq:
        assert torch.equal(_bits(y), _bits(alone[name, t])), (name, t)


@pytest.mark.parametrize("start", [2**31 - 2, 2**31 - 1])
def test_k3_moe_m2_epoch_wrap(start):
    """The CTAs' epochs (int32, + 1 per call; only their parity picks the counter set) across the int32 wrap. The
    state is preset to where ~2^31 calls leave it: every epoch at ``start`` and the counter set the next call uses
    zero. Each call then gives the bits of the same call at small epochs, the epochs wrap to -2^31 and keep counting,
    and every call leaves the set the next one uses zero."""
    _ops()
    proc, _, bias = _experts("tp16")
    st = _state("tp16")
    words = st.mod.GROUPS2 * st.mod.CW
    tokens = [_tokens("random", 700 + c) for c in range(5)]
    alone = [_m2("tp16", proc, bias, logits, x)[0] for logits, x in tokens]
    st.counts.zero_()
    st.epochs.fill_(start)
    for c, (logits, x) in enumerate(tokens):
        y = _m2("tp16", proc, bias, logits, x)[0]
        torch.cuda.synchronize()
        ep = (start + c + 1 + 2**31) % 2**32 - 2**31  # int32 two's complement
        assert torch.equal(_bits(y), _bits(alone[c])), c
        assert bool((st.epochs == ep).all()), (c, ep)
        assert bool((st.counts.view(2, words)[ep & 1] == 0).all()), (c, ep)
