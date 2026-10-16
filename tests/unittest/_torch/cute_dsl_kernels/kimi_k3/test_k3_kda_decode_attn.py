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
"""``trtllm::k3_kda_decode_attn`` (Kimi K3's fused KDA projection + plain decode of R requests of one token) at the
per-rank TP16 shape (6 heads, K = V = 128, conv width 4), against ``trtllm::kda_decode``.

The reference is the model's plain-decode path fed the very projection rows the fused kernel computes: the stream
alone (``trtllm::k3_kda_qkvg``, the same stream clusters) on the same x, its Lamport buffers decoded into the
``[q | k | v | og | f_a | b]`` rows, f_b as a bf16 ``F.linear``, then ``trtllm::kda_decode`` (native, indexed state
pool, packed conv pool updated in place). A float64 torch decode on the same rows bounds both.

Checks over rounds of distinct slots from the same initial pools: every request's output against kda_decode and
float64 (fp32 tolerance), the state rows against both, the conv pool against kda_decode bit for bit (raw inputs),
slots outside the batch untouched; pools dense and with strided slots (the cache manager's interleaving); slots
given as a slice of a longer index tensor at any element offset; the
schedule twice bit-identical; CUDA-graph replays with rewritten inputs; the launch counter near 2^31 (nothing
written outside the buffers); launches interleaved with ``trtllm::k3_kda_attn`` on one shared buffer set, as the
model runs them.

  pytest test_k3_kda_decode_attn.py
  python3 test_k3_kda_decode_attn.py report
"""

import sys

import pytest
import torch
import torch.nn.functional as F

H = 6
K = V = 128
HK = H * K
W = 4
PROJ = 3208
K_IN = 7168
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5
POOL = 11
ROUNDS = 4
NUM_SPEC = 7  # trtllm::k3_kda_attn verifies one request's NUM_SPEC + 1 tokens per launch
TOL_OUT = 2e-2
TOL_STATE = 1e-3
CONV_PAD = 128  # extra bf16 per conv slot in the strided layout (a multiple of 8)
SSM_PAD = (
    3 * HK * (W - 1)
)  # extra fp32 per state slot in the strided layout (the manager's conv bytes)


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op  # noqa: F401


def make_weights(seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    conv = (
        rnd(3, HK, W, scale=0.3).bfloat16().float()
    )  # bf16 values: the native path reads them as bf16
    return {
        "w": rnd(PROJ, K_IN, scale=0.02).bfloat16(), "w_fb": rnd(HK, K, scale=0.05).bfloat16(),
        "w_q": conv[0].contiguous(), "w_k": conv[1].contiguous(), "w_v": conv[2].contiguous(),
        "w_t": [conv[i].t().bfloat16().contiguous() for i in range(3)],
        "a_log": rnd(H, scale=0.5), "dt_bias": rnd(HK, scale=0.5), "onorm_w": (1 + 0.1 * rnd(V)).float(),
    }  # fmt: skip


def make_pools(seed: int, layout: str) -> dict:
    """conv bf16 [POOL, 3 HK, W - 1] and the state fp32 [POOL, H, V, K]; "strided": each slot padded."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    conv = (torch.randn(POOL, 3 * HK, W - 1, generator=g, device="cuda") * 0.5).bfloat16()
    state = torch.randn(POOL, H, V, K, generator=g, device="cuda") * 0.05
    if layout == "strided":
        conv_buf = torch.zeros(
            POOL, 3 * HK * (W - 1) + CONV_PAD, dtype=torch.bfloat16, device="cuda"
        )
        conv_view = conv_buf[:, : 3 * HK * (W - 1)].view(POOL, 3 * HK, W - 1)
        conv_view.copy_(conv)
        conv = conv_view
        st_buf = torch.zeros(POOL, H * V * K + SSM_PAD, device="cuda")
        st_view = st_buf[:, : H * V * K].view(POOL, H, V, K)
        st_view.copy_(state)
        state = st_view
    return {"conv": conv, "state": state}


def clone_pools(p: dict) -> dict:
    out = {}
    for name, t in p.items():
        if t.is_contiguous():
            out[name] = t.clone()
        else:
            buf = torch.zeros(t.shape[0], t.stride(0), dtype=t.dtype, device=t.device)
            view = buf[:, : t[0].numel()].view(t.shape)
            view.copy_(t)
            out[name] = view
    return out


def make_slots(num_requests: int, seed: int) -> torch.Tensor:
    perm = torch.randperm(POOL, generator=torch.Generator().manual_seed(seed))
    return perm[:num_requests].to(torch.int32).cuda()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


class Rows:
    """The projection rows the fused kernel computes for x: the stream alone, its Lamport buffers decoded."""

    def __init__(self, wt):
        from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op

        self.wt = wt
        self.bufs = op.make_buffers(torch.device("cuda"), op.CTAS)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p1, part, epoch = self.bufs
        buf = int(epoch[0].item()) % 3
        torch.ops.trtllm.k3_kda_qkvg(x, self.wt["w"], p1, part, epoch)
        n = x.shape[0]
        qkfa = p1.view(3, 8, 2 * HK + K)[buf, :n].view(torch.bfloat16)
        parts = part.view(3, 3, 2, 8, HK)[buf].view(torch.float32)
        v, og, b = ((parts[r, 0, :n] + parts[r, 1, :n]).bfloat16() for r in range(3))
        rows = torch.zeros(n, PROJ, dtype=torch.bfloat16, device="cuda")
        rows[:, : 2 * HK] = qkfa[:, : 2 * HK]
        rows[:, 2 * HK : 3 * HK] = v
        rows[:, 3 * HK : 4 * HK] = og
        rows[:, 4 * HK : 4 * HK + K] = qkfa[:, 2 * HK :]
        rows[:, 4 * HK + K : 4 * HK + K + H] = b[:, :H]
        return rows


class FusedPath:
    def __init__(self, wt, pools):
        from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op

        self.wt, self.p = wt, pools
        self.bufs = op.make_buffers(torch.device("cuda"), op.FUSED_CTAS)

    def __call__(self, x, slots):
        wt, p = self.wt, self.p
        return torch.ops.trtllm.k3_kda_decode_attn(
            x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
            p["conv"], p["state"], slots, *self.bufs, LOWER_BOUND, SCALE, EPS,
        )  # fmt: skip


class NativePath:
    """The model's plain decode (``forward_decode``): f_b as a bf16 F.linear, then trtllm::kda_decode."""

    def __init__(self, wt, pools):
        self.wt, self.p = wt, pools

    def __call__(self, rows, slots):
        from tensorrt_llm._torch.modules.kimi_kda._kda_decode import run_kda_decode_fusion_cuda

        wt, p = self.wt, self.p
        n = rows.shape[0]

        def heads(cols):
            return cols.unflatten(-1, (H, K)).unsqueeze(0)

        g = F.linear(rows[:, 4 * HK : 4 * HK + K], wt["w_fb"])
        out = torch.empty(n, 1, H, V, dtype=torch.bfloat16, device="cuda")
        conv = p["conv"]
        run_kda_decode_fusion_cuda(
            x_q=heads(rows[:, :HK]), x_k=heads(rows[:, HK : 2 * HK]), x_v=heads(rows[:, 2 * HK : 3 * HK]),
            w_q_t=wt["w_t"][0], w_k_t=wt["w_t"][1], w_v_t=wt["w_t"][2], bias_q=None, bias_k=None, bias_v=None,
            cs_q=conv[:, :HK], cs_k=conv[:, HK : 2 * HK], cs_v=conv[:, 2 * HK :], A_log=wt["a_log"],
            g=heads(g), dt_bias=wt["dt_bias"], beta=rows[:, 4 * HK + K : 4 * HK + K + H].unsqueeze(0),
            state=p["state"], onorm_g=heads(rows[:, 3 * HK : 4 * HK]), onorm_weight=wt["onorm_w"], out=out,
            ssm_state_indices=slots, scale=SCALE, onorm_eps=EPS, lower_bound=LOWER_BOUND,
            use_beta_sigmoid_in_kernel=True, update_conv_cache=True,
        )  # fmt: skip
        return out.view(n, H, V)


class F64Path:
    """The plain KDA decode in float64 torch on the same rows (f_b in float64, rounded to bf16 as the GEMM's
    output): conv4 + SiLU, q / k L2 norm (q scaled), beta sigmoid, the lower-bound gate,
    S <- S d + beta (v - (S d) k) k^T, o = S q, the gated RMSNorm."""

    def __init__(self, wt, pools):
        self.wt = wt
        self.conv = pools["conv"].double().clone()
        self.state = pools["state"].double().clone()

    def __call__(self, rows, slots):
        wt = self.wt
        r = rows.double()
        n = rows.shape[0]
        conv_w = torch.stack([wt["w_q"], wt["w_k"], wt["w_v"]]).double()  # [3, HK, W]
        g = (r[:, 4 * HK : 4 * HK + K] @ wt["w_fb"].double().t()).bfloat16().double()
        outs = torch.empty(n, H, V, dtype=torch.float64, device="cuda")
        for i, s in enumerate(slots.tolist()):
            new = r[i, : 3 * HK].view(3, HK)
            win = self.conv[s].view(3, HK, W - 1)
            u = torch.cat([win, new.unsqueeze(-1)], dim=-1)  # [3, HK, W], oldest first
            act = (u * conv_w).sum(-1)
            act = act * torch.sigmoid(act)
            self.conv[s] = u[:, :, 1:].reshape(3 * HK, W - 1)
            q, k, v = (act[j].view(H, K) for j in range(3))
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(r[i, 4 * HK + K : 4 * HK + K + H])
            xg = torch.exp(wt["a_log"].double()).unsqueeze(-1) * (
                g[i].view(H, K) + wt["dt_bias"].double().view(H, K)
            )
            decay = torch.exp(LOWER_BOUND * torch.sigmoid(xg))
            st = self.state[s] * decay.unsqueeze(1)  # [H, V, K] * decay per key
            res = (v - (st * k.unsqueeze(1)).sum(-1)) * beta.unsqueeze(-1)  # [H, V]
            st = st + res.unsqueeze(-1) * k.unsqueeze(1)
            self.state[s] = st
            o = (st * q.unsqueeze(1)).sum(-1)  # [H, V]
            rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS)
            gate = torch.sigmoid(r[i, 3 * HK : 4 * HK].view(H, V))
            outs[i] = o * rms * wt["onorm_w"].double() * gate
        return outs


def run_schedule(num_requests, layout="dense", seed=0, rounds=ROUNDS):
    _ops()
    wt = make_weights(100 + seed)
    pools = make_pools(200 + seed, layout)
    fused = FusedPath(wt, clone_pools(pools))
    native = NativePath(wt, clone_pools(pools))
    ref = F64Path(wt, pools)
    rows_of = Rows(wt)
    gen = torch.Generator(device="cuda").manual_seed(400 + seed)
    metrics, outs = [], []
    for rnd in range(rounds):
        slots = make_slots(num_requests, 300 + seed + rnd)
        x = torch.randn(num_requests, K_IN, generator=gen, device="cuda").bfloat16()
        state_before = fused.p["state"].clone()
        conv_before = fused.p["conv"].clone()
        out_f = fused(x, slots)
        rows = rows_of(x)
        out_n = native(rows, slots)
        out_r = ref(rows, slots)
        torch.cuda.synchronize()
        idx = slots.long()
        others = torch.ones(POOL, dtype=torch.bool, device="cuda")
        others[idx] = False
        m = dict(round=rnd)
        m["out_vs_native"] = rel(out_f, out_n)
        m["out_vs_f64"] = rel(out_f, out_r)
        m["native_vs_f64"] = rel(out_n, out_r)
        m["state_vs_f64"] = max(rel(fused.p["state"][s], ref.state[s]) for s in idx.tolist())
        m["state_vs_native"] = max(
            rel(fused.p["state"][s], native.p["state"][s]) for s in idx.tolist()
        )
        m["conv_bits_vs_native"] = torch.equal(fused.p["conv"], native.p["conv"])
        m["others_untouched"] = torch.equal(
            fused.p["state"][others], state_before[others]
        ) and torch.equal(fused.p["conv"][others], conv_before[others])
        metrics.append(m)
        outs.append(out_f.clone())
    return metrics, outs


def round_ok(m) -> bool:
    return (m["out_vs_native"] <= TOL_OUT and m["out_vs_f64"] <= TOL_OUT and m["state_vs_f64"] <= TOL_STATE
            and m["state_vs_native"] <= TOL_STATE and m["conv_bits_vs_native"] and m["others_untouched"])  # fmt: skip


@pytest.mark.parametrize("num_requests", range(1, 9))
@pytest.mark.parametrize("layout", ["dense", "strided"])
def test_split(num_requests, layout):
    with torch.inference_mode():
        metrics, _ = run_schedule(num_requests, layout)
    bad = [m for m in metrics if not round_ok(m)]
    assert not bad, bad


@pytest.mark.parametrize("num_requests", [1, 8])
def test_deterministic(num_requests):
    with torch.inference_mode():
        _, a = run_schedule(num_requests, seed=5)
        _, b = run_schedule(num_requests, seed=5)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("num_requests", [1, 3, 8])
def test_graph_replay(num_requests):
    """Rounds captured once in a CUDA graph (static x and slots rewritten in place before every replay) against the
    same rounds run eagerly from the same pools."""
    _ops()
    with torch.inference_mode():
        wt = make_weights(7)
        pools = make_pools(8, "dense")
        eager = FusedPath(wt, clone_pools(pools))
        graphed = FusedPath(wt, clone_pools(pools))
        gen = torch.Generator(device="cuda").manual_seed(9)
        xs = [
            torch.randn(num_requests, K_IN, generator=gen, device="cuda").bfloat16()
            for _ in range(ROUNDS)
        ]
        slots = [make_slots(num_requests, 10 + r) for r in range(ROUNDS)]
        want = [eager(x, s).clone() for x, s in zip(xs, slots)]
        x_in = xs[0].clone()
        s_in = slots[0].clone()
        graphed(x_in, s_in)  # compiles; this launch's result is discarded with the pools below
        graphed.p = clone_pools(pools)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            y = graphed(x_in, s_in)
        torch.cuda.current_stream().wait_stream(stream)
        got = []
        for x, s in zip(xs, slots):
            x_in.copy_(x)
            s_in.copy_(s)
            graph.replay()
            got.append(y.clone())
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    assert torch.equal(eager.p["state"], graphed.p["state"]) and torch.equal(
        eager.p["conv"], graphed.p["conv"]
    )


def _alone(path, x, slots):
    """One launch, complete before the next: the head CTAs read the slots' pools before their grid-dependency wait,
    so a launch right behind another on the same slots could read them mid-update (the model never runs one layer's
    KDA twice in a row; its next launch on these pools is a step later)."""
    out = path(x, slots)
    torch.cuda.synchronize()
    return out


def test_epoch_wrap():
    """Every CTA's counter preset to 2^31 - 2, as after 2^31 launches on a device (the buffers are shared by every KDA
    layer): four launches write nothing outside the buffers (each inside guard bands), give the bits of a run from
    zero, and keep the counter in 0..2."""
    _ops()
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op

    with torch.inference_mode():
        wt = make_weights(13)
        pools = make_pools(14, "dense")
        gen = torch.Generator(device="cuda").manual_seed(15)
        xs = [torch.randn(4, K_IN, generator=gen, device="cuda").bfloat16() for _ in range(4)]
        slots = make_slots(4, 16)
        fresh = FusedPath(wt, clone_pools(pools))
        want = [_alone(fresh, x, slots) for x in xs]
        bands = []
        for numel, dtype, fill in (
            (op.P1_NUMEL, torch.int16, 0x1234),
            (op.PART_NUMEL, torch.int32, 0x12345678),
        ):
            guard = numel + 4096  # a whole set of buffers on each side
            big = torch.full((guard + numel + guard,), fill, dtype=dtype, device="cuda")
            big[guard : guard + numel] = -1
            bands.append((big, guard, numel, fill))
        epoch = torch.full((op.FUSED_CTAS,), 2**31 - 2, dtype=torch.int32, device="cuda")
        wrapped = FusedPath(wt, clone_pools(pools))
        wrapped.bufs = tuple(big[guard : guard + numel] for big, guard, numel, _ in bands) + (
            epoch,
        )
        got = [_alone(wrapped, x, slots) for x in xs]
    for big, guard, numel, fill in bands:
        assert bool((big[:guard] == fill).all()), "stores before the buffers"
        assert bool((big[guard + numel :] == fill).all()), "stores after the buffers"
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert torch.equal(wrapped.p["state"], fresh.p["state"]) and torch.equal(
        wrapped.p["conv"], fresh.p["conv"]
    )
    assert bool(((epoch >= 0) & (epoch < 3)).all()), epoch.unique().tolist()


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_slots_offset(offset):
    """``slots`` as a slice of a longer index tensor that starts at element ``offset`` (the mixer passes
    ``state_indices[num_prefills:]`` in a step with prefills, so the data pointer is only 4-byte aligned): the output
    and the pools bit for bit those of the same slots in a tensor of their own."""
    _ops()
    with torch.inference_mode():
        wt = make_weights(17)
        pools = make_pools(18, "dense")
        gen = torch.Generator(device="cuda").manual_seed(19)
        x = torch.randn(3, K_IN, generator=gen, device="cuda").bfloat16()
        slots = make_slots(3, 20)
        idx = torch.zeros(offset + slots.numel(), dtype=torch.int32, device="cuda")
        idx[offset:] = slots
        view = idx[offset:]
        ref = FusedPath(wt, clone_pools(pools))
        got = FusedPath(wt, clone_pools(pools))
        want = _alone(ref, x, slots)
        out = _alone(got, x, view)
    assert view.data_ptr() % 16 == 4 * offset % 16
    assert torch.equal(out, want)
    assert torch.equal(got.p["state"], ref.p["state"]) and torch.equal(got.p["conv"], ref.p["conv"])


def make_verify_pools(seed: int) -> dict:
    """``trtllm::k3_kda_attn``'s pools for a layer (test_k3_kda_attn.py's dense layout): the conv caches
    [POOL, HK, W - 1 + NUM_SPEC] (dim-contiguous), the SSM state, the drafts' records and the pending counts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = {name: (torch.randn(POOL, W - 1 + NUM_SPEC, HK, generator=g, device="cuda") * 0.5).transpose(1, 2)
         for name in ("cs_q", "cs_k", "cs_v")}  # fmt: skip
    p["state"] = torch.randn(POOL, H, V, K, generator=g, device="cuda") * 0.05
    p["state_tok"] = torch.zeros(POOL, 3, NUM_SPEC, H, K, device="cuda")
    p["pending"] = torch.zeros(POOL, dtype=torch.int32, device="cuda")
    return p


def clone_verify_pools(p: dict) -> dict:
    return {
        name: t.transpose(1, 2).clone().transpose(1, 2) if name.startswith("cs_") else t.clone()
        for name, t in p.items()
    }


class VerifyPath:
    """``trtllm::k3_kda_attn``: the same projection and the speculative verify of one request's NUM_SPEC + 1 tokens."""

    def __init__(self, wt, pools, bufs):
        self.wt, self.p, self.bufs = wt, pools, bufs

    def __call__(self, x, slot):
        wt, p = self.wt, self.p
        return torch.ops.trtllm.k3_kda_attn(
            x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
            p["cs_q"], p["cs_k"], p["cs_v"], p["state"], p["state_tok"], slot, p["pending"], *self.bufs, NUM_SPEC,
            LOWER_BOUND, SCALE, EPS,
        )  # fmt: skip


def test_shared_buffers_with_verify():
    """The model keeps one (p1, part, epoch) set per device for this op and ``trtllm::k3_kda_attn``
    (kimi_kda_mixer.py): plain-decode and verify launches interleaved in stream order on one set give the bits of the
    same launches on a set of their own, one at a time. Both run the same stream role (all 8 token rows published,
    the same words re-armed) and advance every CTA's index once per launch, so either may follow the other."""
    _ops()
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op

    dev = torch.device("cuda")
    with torch.inference_mode():
        wt = make_weights(23)
        pools = make_pools(24, "dense")
        vpools = make_verify_pools(25)
        gen = torch.Generator(device="cuda").manual_seed(26)
        calls = []
        for i, r in enumerate((1, 4, 8, 3)):
            x = torch.randn(r, K_IN, generator=gen, device="cuda").bfloat16()
            calls.append(("decode", x, make_slots(r, 27 + i)))
            x = torch.randn(NUM_SPEC + 1, K_IN, generator=gen, device="cuda").bfloat16()
            calls.append(("verify", x, torch.tensor([i], dtype=torch.int32, device="cuda")))
        own = {
            "decode": FusedPath(wt, clone_pools(pools)),
            "verify": VerifyPath(
                wt, clone_verify_pools(vpools), op.make_buffers(dev, op.FUSED_CTAS)
            ),
        }
        want = [_alone(own[kind], x, s) for kind, x, s in calls]
        shared_bufs = op.make_buffers(dev, op.FUSED_CTAS)
        shared = {
            "decode": FusedPath(wt, clone_pools(pools)),
            "verify": VerifyPath(wt, clone_verify_pools(vpools), shared_bufs),
        }
        shared["decode"].bufs = shared_bufs
        got = [shared[kind](x, s) for kind, x, s in calls]
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    for kind, path in shared.items():
        assert all(torch.equal(t, own[kind].p[name]) for name, t in path.p.items()), kind
    epoch = shared_bufs[2]
    assert bool((epoch == epoch[0]).all()), epoch.unique().tolist()


def report() -> int:
    """Per split and layout: the worst round's errors."""
    print(f"{torch.cuda.get_device_name()}; {ROUNDS} rounds per split; rel = max |a - b| / max |b|")
    print("| R | layout | out vs kda_decode | out vs f64 | kda_decode vs f64 | state vs f64 | state vs kda_decode | "
          "conv bits | others untouched | result |")  # fmt: skip
    print("| --: | :-- | --: | --: | --: | --: | --: | :-- | :-- | :-- |")
    ok_all = True
    with torch.inference_mode():
        for layout in ("dense", "strided"):
            for r in range(1, 9):
                metrics, _ = run_schedule(r, layout)
                ok = all(round_ok(m) for m in metrics)
                ok_all &= ok
                worst = {k: max(m[k] for m in metrics) for k in ("out_vs_native", "out_vs_f64", "native_vs_f64",
                                                                 "state_vs_f64", "state_vs_native")}  # fmt: skip
                conv_ok = all(m["conv_bits_vs_native"] for m in metrics)
                others_ok = all(m["others_untouched"] for m in metrics)
                print(f"| {r} | {layout} | {worst['out_vs_native']:.2e} | {worst['out_vs_f64']:.2e} | "
                      f"{worst['native_vs_f64']:.2e} | {worst['state_vs_f64']:.2e} | {worst['state_vs_native']:.2e} | "
                      f"{conv_ok} | {others_ok} | {'PASS' if ok else 'FAIL'} |", flush=True)  # fmt: skip
    print("ALL PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
