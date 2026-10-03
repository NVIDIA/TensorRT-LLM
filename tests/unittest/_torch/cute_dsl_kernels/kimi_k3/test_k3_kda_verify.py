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
"""``trtllm::k3_kda_verify`` (Kimi K3's KDA speculative verify with a committed state per verify token) for R requests
of T = 1 + num_spec tokens, at the per-rank TP16 shape (6 heads, K = V = 128, conv width 4).

Two references, run over the same schedule of rounds from the same initial pools, with distinct slots and a pending
(accepted-draft) count per request that changes every round:

* main's path, as ``KimiDeltaAttention.forward_verify_fused`` calls it: f_b as a bf16 ``F.linear``, then
  ``trtllm::kda_mtp_decode`` replaying each request's accepted drafts from its replay caches
  (``cu_seqlens[n] = n T - pending[n]``, ``num_accepted_tokens = pending``), then the gated RMSNorm
  (``rms_norm_gated_token_major``, sigmoid gate). The report says which ``kda_mtp_decode`` was loaded.
* a float64 torch delta rule (the "fp32 reference"; float64 so that no TF32 enters) that keeps each request's
  committed token history: conv4 + SiLU, q / k L2 norm, beta sigmoid, the lower-bound gate,
  S <- S d + beta (v - (S d) k) k^T, o = S q, the gated RMSNorm.

Checks per round: the outputs of every request against both references (fp32 tolerance), the committed pool state
(after each golden token) against the history, the conv caches against main's (raw inputs: bit-exact); with the gate
taken from the unfused f_b output (``g_ext``) the pool state against main's bit for bit. The drafts' records in
``state_tok`` are checked through the next round, which starts from the accepted ones. Also: requests isolated, the
schedule twice bit-identical, CUDA-graph replays with rewritten inputs, the slots and the pending counts given as
slices of longer index tensors at any element offset.

  pytest test_k3_kda_verify.py
  python3 test_k3_kda_verify.py report | time
"""

import inspect
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

H = 6
K = V = 128
W = 4
HK = H * K
PROJ = 4 * HK + K + H + 2  # the fused [q | k | v | onorm gate | f_a | b | pad] row (3208 columns)
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5
POOL = 11
ROUNDS = 6
TOL_OUT = 2e-2  # bf16 outputs; main rounds the core to bf16 before its norm
# fp32 states against the fp32 history: approximate exp / rcp in the kernels, and the bf16 f_b gate (one bf16 ulp
# moves a decay by up to ~1 %; main's kda_mtp_decode measures 2.05e-4 at 6x8)
TOL_STATE = 1e-3

SPLITS = [(r, t) for t in (8, 4, 2) for r in range(1, 9)]


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401  (kda_mtp_decode)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import op  # noqa: F401  (k3_kda_verify)


def main_kda_variant() -> str:
    """'main' when the loaded kda_mtp_decode has main's signature, else 'port' (a modified one)."""
    _ops()
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as mtp

    return (
        "port"
        if "accepted_by_slot" in inspect.signature(mtp.kda_mtp_decode_impl).parameters
        else "main"
    )


def make_weights(seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    return {
        "w_q": rnd(HK, W, scale=0.3), "w_k": rnd(HK, W, scale=0.3), "w_v": rnd(HK, W, scale=0.3),
        "a_log": rnd(H, scale=0.5), "dt_bias": rnd(HK, scale=0.5), "onorm_w": (1 + 0.1 * rnd(V)).float(),
        "w_fb": (rnd(HK, K) * 0.05).bfloat16(),
    }  # fmt: skip


def strided(state: torch.Tensor) -> torch.Tensor:
    """The same values in the Mamba cache manager's layout: each slot's SSM state followed by its conv state bytes."""
    pool = torch.zeros(state.shape[0], state[0].numel() + 3 * HK * (W - 1), device=state.device)
    view = pool[:, : state[0].numel()].view(state.shape)
    view.copy_(state)
    return view


def make_pools(seed: int, num_spec: int, layout: str) -> dict:
    """The initial pools: conv caches [pool, HK, W - 1 + num_spec] (dim-contiguous), the SSM state [pool, H, V, K]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = W - 1 + num_spec
    p = {name: (torch.randn(POOL, s, HK, generator=g, device="cuda") * 0.5).transpose(1, 2)
         for name in ("cs_q", "cs_k", "cs_v")}  # fmt: skip
    p["state"] = torch.randn(POOL, H, V, K, generator=g, device="cuda") * 0.05
    if layout == "strided":
        p["state"] = strided(p["state"])
    return p


def clone_pools(p: dict) -> dict:
    out = {}
    for name, t in p.items():
        if name.startswith("cs_"):
            out[name] = (
                t.transpose(1, 2).clone().transpose(1, 2)
            )  # a copy in the same dim-contiguous layout
        elif name == "state" and not t.is_contiguous():
            out[name] = strided(t)
        else:
            out[name] = t.clone()
    return out


def split_proj(proj: torch.Tensor):
    t = proj.shape[0]
    x_q, x_k, x_v, og = (proj[:, i * HK : (i + 1) * HK] for i in range(4))
    f_a = proj[:, 4 * HK : 4 * HK + K]
    beta = proj[:, 4 * HK + K : 4 * HK + K + H]
    return t, x_q, x_k, x_v, og, f_a, beta


class MainPath:
    """main's verify (forward_verify_fused): its replay caches, a per-request pending count."""

    def __init__(self, wt, pools, slots, num_spec):
        self.wt, self.p, self.slots, self.num_spec = wt, pools, slots, num_spec
        self.p["qkg_cache"] = torch.zeros(POOL, num_spec, 3, HK, device="cuda")
        self.p["v_cache"] = torch.zeros(POOL, num_spec, HK, device="cuda")
        self.p["beta_cache"] = torch.zeros(POOL, num_spec, H, device="cuda")

    def __call__(self, proj, pending_req):
        from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

        t, x_q, x_k, x_v, og, f_a, beta = split_proj(proj)
        n, steps = self.slots.numel(), self.num_spec + 1
        g = F.linear(f_a, self.wt["w_fb"])
        cu = torch.arange(0, (n + 1) * steps, steps, dtype=torch.int32, device="cuda")
        cu[:n].sub_(pending_req)
        o = torch.ops.trtllm.kda_mtp_decode(
            x_q=x_q.view(1, t, H, K), x_k=x_k.view(1, t, H, K), x_v=x_v.view(1, t, H, V), w_q=self.wt["w_q"],
            w_k=self.wt["w_k"], w_v=self.wt["w_v"], cs_q=self.p["cs_q"], cs_k=self.p["cs_k"], cs_v=self.p["cs_v"],
            g=g.view(1, t, H, K), beta=beta.contiguous().view(1, t, H), A_log=self.wt["a_log"],
            dt_bias=self.wt["dt_bias"], recurrent_state=self.p["state"], qkg_cache=self.p["qkg_cache"],
            v_cache=self.p["v_cache"], beta_cache=self.p["beta_cache"], ssm_state_indices=self.slots, cu_seqlens=cu,
            num_spec=self.num_spec, num_accepted_tokens=pending_req, lower_bound=LOWER_BOUND, scale=SCALE,
        )  # fmt: skip
        core = rms_norm_gated_token_major(o.reshape(-1, V), og.reshape(t, H, V), self.wt["onorm_w"], EPS,
                                          gate_activation="sigmoid")  # fmt: skip
        return core.view(t, H, V), g


class K3Path:
    """k3_kda_verify: a committed state per verify token, a per-slot pending count."""

    def __init__(self, wt, pools, slots, num_spec):
        self.wt, self.p, self.slots, self.num_spec = wt, pools, slots, num_spec
        self.p["state_tok"] = torch.zeros(POOL, num_spec, H, V, K, device="cuda")
        self.p["pending"] = torch.zeros(POOL, dtype=torch.int32, device="cuda")

    def __call__(self, proj, g_ext=None):
        p = self.p
        return torch.ops.trtllm.k3_kda_verify(
            proj, self.wt["w_fb"], self.wt["w_q"], self.wt["w_k"], self.wt["w_v"], self.wt["a_log"],
            self.wt["dt_bias"], self.wt["onorm_w"], p["cs_q"], p["cs_k"], p["cs_v"], p["state"], p["state_tok"],
            self.slots, p["pending"], self.num_spec, LOWER_BOUND, SCALE, EPS, g_ext,
        )  # fmt: skip


def conv_silu(win, raw, c, w):
    """Channel group ``c`` (q, k, v) of the causal conv over the window (oldest first) and the new raw input, SiLU."""
    x = win[0][c] * w[:, 0] + win[1][c] * w[:, 1] + win[2][c] * w[:, 2] + raw[c] * w[:, 3]
    return x * torch.sigmoid(x)


class Fp32Path:
    """The delta rule over each request's committed history (raw conv inputs and the state after the last committed
    token), in float64: no TF32 even where cuBLAS is told to use it."""

    def __init__(self, wt, pools, slots, num_spec):
        self.wt, self.num_spec = wt, num_spec
        self.slots = slots.tolist()
        # Committed raw inputs, oldest first: the conv caches' window columns 0..2 (q, k, v).
        self.seq = [[torch.stack([pools[c][s, :, i].double() for c in ("cs_q", "cs_k", "cs_v")]) for i in range(3)]
                    for s in self.slots]  # fmt: skip
        self.state = [pools["state"][s].double().clone() for s in self.slots]
        self.last = None

    def __call__(self, proj):
        """Outputs [T, H, V] and the per-token states of every request."""
        t_total, x_q, x_k, x_v, og, f_a, beta_raw = split_proj(proj)
        steps = self.num_spec + 1
        # f_b's output is bf16 in the model.
        g_all = F.linear(f_a.double(), self.wt["w_fb"].double()).bfloat16().double()
        wq, wk, wv = (self.wt[n].double() for n in ("w_q", "w_k", "w_v"))
        exp_a = self.wt["a_log"].double().exp()
        dt_bias = self.wt["dt_bias"].double().view(H, K)
        onorm_w = self.wt["onorm_w"].double()
        out = torch.empty(t_total, H, V, dtype=torch.float64, device="cuda")
        self.last = []
        for n in range(len(self.slots)):
            seq = list(self.seq[n])
            s_cur = self.state[n]
            states, raws = [], []
            for t in range(steps):
                row = n * steps + t
                raw = torch.stack([x_q[row].double(), x_k[row].double(), x_v[row].double()])
                win = seq[-3:]
                q = conv_silu(win, raw, 0, wq).view(H, K)
                k = conv_silu(win, raw, 1, wk).view(H, K)
                v = conv_silu(win, raw, 2, wv).view(H, V)
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
                beta = torch.sigmoid(beta_raw[row].double())
                gk = LOWER_BOUND * torch.sigmoid(exp_a[:, None] * (g_all[row].view(H, K) + dt_bias))
                decay = gk.exp()
                sd = s_cur * decay[:, None, :]
                vn = v - torch.einsum("hvk,hk->hv", sd, k)
                s_cur = sd + beta[:, None, None] * vn[:, :, None] * k[:, None, :]
                o = torch.einsum("hvk,hk->hv", s_cur, q)
                rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS)
                out[row] = o * rms * onorm_w * torch.sigmoid(og[row].double().view(H, V))
                states.append(s_cur)
                raws.append(raw)
                seq.append(raw)
            self.last.append((states, raws))
        return out

    def commit(self, pending):
        """The sampler accepted ``pending[n]`` drafts of the last round: the golden token and those drafts commit."""
        for n, p in enumerate(pending):
            states, raws = self.last[n]
            self.seq[n] = (self.seq[n] + raws[: p + 1])[-3:]
            self.state[n] = states[p]


def make_slots(num_requests: int, seed: int) -> torch.Tensor:
    perm = torch.randperm(POOL, generator=torch.Generator().manual_seed(seed))
    return perm[:num_requests].to(torch.int32).cuda()


def pending_schedule(num_requests: int, num_spec: int, rnd: int):
    """Accepted drafts per request after round ``rnd``: every count 0..num_spec, different per request."""
    return [(3 * n + 5 * rnd + 1 + (rnd * n) % 3) % (num_spec + 1) for n in range(num_requests)]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def run_schedule(num_requests, steps, fold=True, layout="dense", seed=0, rounds=ROUNDS):
    """Main's path, k3_kda_verify and the fp32 history over ``rounds`` rounds; per-round metrics."""
    _ops()
    num_spec = steps - 1
    wt = make_weights(100 + seed)
    pools = make_pools(200 + seed, num_spec, layout)
    slots = make_slots(num_requests, 300 + seed)
    main = MainPath(wt, clone_pools(pools), slots, num_spec)
    k3 = K3Path(wt, clone_pools(pools), slots, num_spec)
    ref = Fp32Path(wt, pools, slots, num_spec)
    pending_req = torch.zeros(num_requests, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(400 + seed)
    rows, outs = [], []
    for rnd in range(rounds):
        proj = torch.randn(num_requests * steps, PROJ, generator=gen, device="cuda").bfloat16()
        out_main, g = main(proj, pending_req)
        out_k3 = k3(proj, None if fold else g.contiguous())
        out_ref = ref(proj)
        torch.cuda.synchronize()
        r = dict(round=rnd, pending=pending_req.tolist())
        r["out_vs_main"] = rel(out_k3, out_main)
        r["out_vs_fp32"] = rel(out_k3, out_ref)
        r["main_vs_fp32"] = rel(out_main, out_ref)
        # The pool state after the golden token (state_tok holds the drafts' compact records, which the next round's
        # outputs check against the history).
        st_err, main_st_err, st_bits = 0.0, 0.0, True
        for n, s in enumerate(slots.tolist()):
            states, _ = ref.last[n]
            st_err = max(st_err, rel(k3.p["state"][s], states[0]))
            main_st_err = max(main_st_err, rel(main.p["state"][s], states[0]))
            st_bits &= torch.equal(k3.p["state"][s], main.p["state"][s])
        r["state_vs_fp32"] = st_err
        r["main_state_vs_fp32"] = main_st_err
        r["state_bits_vs_main"] = st_bits
        r["conv_bits_vs_main"] = all(
            torch.equal(k3.p[c], main.p[c]) for c in ("cs_q", "cs_k", "cs_v")
        )
        rows.append(r)
        outs.append(out_k3.clone())
        nxt = pending_schedule(num_requests, num_spec, rnd)
        pending_req.copy_(torch.tensor(nxt, dtype=torch.int32))
        k3.p["pending"][slots.long()] = pending_req
        ref.commit(nxt)
    return rows, outs


def round_ok(r, fold: bool) -> bool:
    ok = (r["out_vs_main"] <= TOL_OUT and r["out_vs_fp32"] <= TOL_OUT and r["state_vs_fp32"] <= TOL_STATE
          and r["conv_bits_vs_main"])  # fmt: skip
    if not fold:  # the same gate as main's: the recurrence's arithmetic is kda_mtp_decode's
        ok &= r["state_bits_vs_main"]
    return ok


@pytest.mark.parametrize("num_requests,steps", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_split(num_requests, steps):
    with torch.inference_mode():
        rows, outs = run_schedule(num_requests, steps, fold=True)
        again, outs2 = run_schedule(num_requests, steps, fold=True)
    for r in rows:
        assert round_ok(r, True), r
    assert all(torch.equal(a, b) for a, b in zip(outs, outs2)), "the schedule twice differs"


@pytest.mark.parametrize("num_requests,steps", [(1, 8), (3, 4), (8, 8), (8, 2)])
@pytest.mark.parametrize("layout", ["dense", "strided"])
def test_g_ext(num_requests, steps, layout):
    """The gate from the unfused f_b output (main's bf16 F.linear): the pool state bit-exact against main's."""
    with torch.inference_mode():
        rows, _ = run_schedule(num_requests, steps, fold=False, layout=layout, seed=1)
    for r in rows:
        assert round_ok(r, False), r


@pytest.mark.parametrize("num_requests,steps", [(4, 8), (8, 4)])
def test_strided_fold(num_requests, steps):
    with torch.inference_mode():
        rows, _ = run_schedule(num_requests, steps, fold=True, layout="strided", seed=2)
    for r in rows:
        assert round_ok(r, True), r


@pytest.mark.parametrize("num_requests,steps", [(4, 8), (8, 2)])
def test_isolation(num_requests, steps):
    """A change in request 0's rows changes only request 0's outputs, pool state and per-token states."""
    _ops()
    with torch.inference_mode():
        num_spec = steps - 1
        wt = make_weights(7)
        pools = make_pools(8, num_spec, "dense")
        slots = make_slots(num_requests, 9)
        a = K3Path(wt, clone_pools(pools), slots, num_spec)
        b = K3Path(wt, clone_pools(pools), slots, num_spec)
        pend = torch.tensor(
            pending_schedule(num_requests, num_spec, 3), dtype=torch.int32, device="cuda"
        )
        for path in (a, b):
            path.p["pending"][slots.long()] = pend
        proj = torch.randn(num_requests * steps, PROJ, generator=torch.Generator(device="cuda").manual_seed(10),
                           device="cuda").bfloat16()  # fmt: skip
        proj_c = proj.clone()
        proj_c[1, HK + 5] += (
            1.0  # request 0, token 1, k channel 5: its outputs and states from token 1 on
        )
        out_a, out_b = a(proj), b(proj_c)
        torch.cuda.synchronize()
        assert not torch.equal(out_a[:steps], out_b[:steps])
        assert torch.equal(out_a[steps:], out_b[steps:])
        s0 = int(slots[0])
        others = [int(s) for s in slots[1:]]
        assert not torch.equal(a.p["state_tok"][s0], b.p["state_tok"][s0])
        for s in others:
            assert torch.equal(a.p["state"][s], b.p["state"][s])
            assert torch.equal(a.p["state_tok"][s], b.p["state_tok"][s])


@pytest.mark.parametrize("num_requests,steps", [(8, 8), (4, 2), (1, 8)])
def test_graph_replay(num_requests, steps):
    """One captured call replayed over the schedule with the projection rows and pending counts rewritten in place,
    bit-identical to eager calls on a copy of the pools."""
    _ops()
    with torch.inference_mode():
        num_spec = steps - 1
        wt = make_weights(11)
        pools = make_pools(12, num_spec, "dense")
        slots = make_slots(num_requests, 13)
        graphed = K3Path(wt, clone_pools(pools), slots, num_spec)
        eager = K3Path(wt, clone_pools(pools), slots, num_spec)
        proj = torch.zeros(num_requests * steps, PROJ, dtype=torch.bfloat16, device="cuda")
        holder = {}
        snap = clone_pools(
            {k: v for k, v in graphed.p.items() if k in ("cs_q", "cs_k", "cs_v", "state")}
        )
        snap_tok = graphed.p["state_tok"].clone()
        graphed(proj)  # compiles outside capture
        torch.cuda.synchronize()
        for name in ("cs_q", "cs_k", "cs_v", "state"):
            graphed.p[name].copy_(snap[name])
        graphed.p["state_tok"].copy_(snap_tok)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            holder["out"] = graphed(proj)
        gen = torch.Generator(device="cuda").manual_seed(14)
        for rnd in range(ROUNDS):
            proj.copy_(torch.randn(proj.shape, generator=gen, device="cuda").bfloat16())
            graph.replay()
            want = eager(proj.clone())
            torch.cuda.synchronize()
            assert torch.equal(holder["out"], want), f"round {rnd}"
            for name in ("cs_q", "cs_k", "cs_v", "state", "state_tok"):
                assert torch.equal(graphed.p[name], eager.p[name]), f"round {rnd} {name}"
            nxt = torch.tensor(
                pending_schedule(num_requests, num_spec, rnd), dtype=torch.int32, device="cuda"
            )
            graphed.p["pending"][slots.long()] = nxt
            eager.p["pending"][slots.long()] = nxt


def _at_offset(t: torch.Tensor, offset: int) -> torch.Tensor:
    """``t``'s values in a longer tensor, starting at element ``offset`` (a slice such as
    ``state_indices[num_contexts:]``: its data pointer is aligned to the element only)."""
    buf = torch.zeros(offset + t.numel(), dtype=t.dtype, device=t.device)
    buf[offset:] = t
    view = buf[offset:]
    assert view.data_ptr() % 16 == offset * t.element_size() % 16
    return view


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("which", ["slots", "pending"])
def test_index_offset(which, offset):
    """The slots or the pending counts as a slice of a longer index tensor that starts at element ``offset``: over two
    rounds, the second on nonzero pending counts, the outputs and the pools bit for bit those of the same indices in
    tensors of their own."""
    _ops()
    with torch.inference_mode():
        num_requests, steps = 3, 8
        num_spec = steps - 1
        wt = make_weights(15)
        pools = make_pools(16, num_spec, "dense")
        slots = make_slots(num_requests, 17)
        ref = K3Path(wt, clone_pools(pools), slots, num_spec)
        got = K3Path(
            wt,
            clone_pools(pools),
            _at_offset(slots, offset) if which == "slots" else slots,
            num_spec,
        )
        if which == "pending":
            got.p["pending"] = _at_offset(got.p["pending"], offset)
        gen = torch.Generator(device="cuda").manual_seed(18)
        for rnd in range(2):
            proj = torch.randn(num_requests * steps, PROJ, generator=gen, device="cuda").bfloat16()
            want, out = ref(proj), got(proj)
            torch.cuda.synchronize()
            assert torch.equal(out, want), rnd
            for name in ("cs_q", "cs_k", "cs_v", "state", "state_tok"):
                assert torch.equal(got.p[name], ref.p[name]), (rnd, name)
            nxt = torch.tensor(
                pending_schedule(num_requests, num_spec, rnd), dtype=torch.int32, device="cuda"
            )
            ref.p["pending"][slots.long()] = nxt
            got.p["pending"][slots.long()] = nxt


# ----------------------------------------------------------------------------------------------------------------
# Report and timing (python3 test_k3_kda_verify.py report | time)
# ----------------------------------------------------------------------------------------------------------------


def report() -> int:
    with torch.inference_mode():
        print(f"{torch.cuda.get_device_name()}; kda_mtp_decode: {main_kda_variant()}")
        print("| split | mode | layout | rounds | k3 vs main | k3 vs fp32 | main vs fp32 | state vs fp32 "
              "| main state vs fp32 | state bits = main | conv bits = main | deterministic | result |")  # fmt: skip
        print("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- | :-- |")
        ok_all = True
        cases = [(r, t, True, "dense") for r, t in SPLITS]
        cases += [
            (r, t, False, layout)
            for r, t in [(1, 8), (3, 4), (8, 8), (8, 2)]
            for layout in ("dense", "strided")
        ]
        cases += [(4, 8, True, "strided"), (8, 4, True, "strided")]
        for r, t, fold, layout in cases:
            rows, outs = run_schedule(
                r, t, fold=fold, layout=layout, seed=0 if fold and layout == "dense" else 1
            )
            _, outs2 = run_schedule(
                r, t, fold=fold, layout=layout, seed=0 if fold and layout == "dense" else 1
            )
            det = all(torch.equal(a, b) for a, b in zip(outs, outs2))
            ok = all(round_ok(x, fold) for x in rows) and det
            ok_all &= ok
            print(f"| {r}x{t} | {'fold' if fold else 'g_ext'} | {layout} | {len(rows)} | "
                  f"{max(x['out_vs_main'] for x in rows):.2e} | {max(x['out_vs_fp32'] for x in rows):.2e} | "
                  f"{max(x['main_vs_fp32'] for x in rows):.2e} | {max(x['state_vs_fp32'] for x in rows):.2e} | "
                  f"{max(x['main_state_vs_fp32'] for x in rows):.2e} | "
                  f"{all(x['state_bits_vs_main'] for x in rows)} | {all(x['conv_bits_vs_main'] for x in rows)} | "
                  f"{det} | {'PASS' if ok else 'FAIL'} |", flush=True)  # fmt: skip
    print("ALL PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def time_graph(body, calls, replays=15):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body(0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls):
                body(i)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_call.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(per_call), min(per_call), max(per_call)


def timing(layers: int = 48) -> None:
    """Graphs of back-to-back calls over ``layers`` copies of the weights and pools (HBM-cold), steady pending."""
    _ops()
    print(f"{torch.cuda.get_device_name()}; kda_mtp_decode: {main_kda_variant()}; graphs over {layers} layer copies, "
          "15 replays: median (min-max) us per call")  # fmt: skip
    print("| split | main: f_b + kda_mtp_decode + gated norm | k3_kda_verify |")
    print("| :-- | --: | --: |")
    with torch.inference_mode():
        for r, t in SPLITS:
            num_spec = t - 1
            slots = make_slots(r, 1)
            mains, k3s = [], []
            for i in range(layers):
                wt = make_weights(1000 + i)
                pools = make_pools(2000 + i, num_spec, "dense")
                mains.append(MainPath(wt, clone_pools(pools), slots, num_spec))
                k3s.append(K3Path(wt, pools, slots, num_spec))
            pend = torch.tensor(pending_schedule(r, num_spec, 2), dtype=torch.int32, device="cuda")
            for k3 in k3s:
                k3.p["pending"][slots.long()] = pend
            proj = torch.randn(r * t, PROJ, device="cuda").bfloat16()
            arms = [lambda i: mains[i % layers](proj, pend), lambda i: k3s[i % layers](proj)]
            res = [[], []]
            for rep in range(3):
                for a in (0, 1) if rep % 2 == 0 else (1, 0):
                    res[a].append(time_graph(arms[a], layers))
            cells = []
            for a in range(2):
                meds = sorted(x[0] for x in res[a])
                cells.append(
                    f"{meds[1]:.2f} ({min(x[1] for x in res[a]):.2f}-{max(x[2] for x in res[a]):.2f})"
                )
            print(f"| {r}x{t} | " + " | ".join(cells) + " |", flush=True)
            mains.clear()
            k3s.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    elif len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
