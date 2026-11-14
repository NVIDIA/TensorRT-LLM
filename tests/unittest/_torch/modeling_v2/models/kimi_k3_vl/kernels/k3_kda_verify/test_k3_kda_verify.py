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

* the built-in path, as ``KimiDeltaAttention.forward_verify_fused`` calls it: f_b as a bf16 ``F.linear``, then
  ``trtllm::kda_mtp_decode`` replaying each request's accepted drafts from its replay caches
  (``cu_seqlens[n] = n T - pending[n]``, ``num_accepted_tokens = pending``), then the gated RMSNorm
  (``rms_norm_gated_token_major``, sigmoid gate). The report says which ``kda_mtp_decode`` was loaded.
* a float64 torch delta rule (the "fp32 reference", ``_kda_cells.F64Verify``; float64 so that no TF32 enters) that
  keeps each request's committed token history: conv4 + SiLU, q / k L2 norm, beta sigmoid, the lower-bound gate,
  S <- S d + beta (v - (S d) k) k^T, o = S q, the gated RMSNorm.

Checks per round: the outputs of every request against both references (fp32 tolerance), the committed pool state
(after each golden token) against the history and the built-in path's chunked 3xTF32 recurrence; the conv caches
against the built-in path's (raw inputs: bit-exact). The sequential FP32 and chunked tensor-core state updates have
different rounding order.
The drafts' records in ``state_tok`` are checked through the next round, which starts from the accepted ones.
Also: requests isolated, each schedule twice bit-identical, CUDA-graph replays with rewritten inputs, the slots
and the pending counts given as slices of longer index tensors at any element offset.

  pytest test_k3_kda_verify.py
  PYTHONPATH=<repo>/tests python3 test_k3_kda_verify.py report | time
"""

import inspect
import statistics
import sys

import pytest
import test_common.magic_import  # noqa: F401  (__extra_import_path__ when run directly)
import torch
import torch.nn.functional as F

__extra_import_path__ = ["../../ops/ssm"]
import _kda_cells as kc

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
TOL_OUT = 2e-2  # bf16 outputs; the built-in path rounds the core to bf16 before its norm
# fp32 states against the fp32 history: approximate exp / rcp in the kernels, and the bf16 f_b gate (one bf16 ulp
# moves a decay by up to ~1 %; the built-in kda_mtp_decode measures 2.05e-4 at 6x8)
TOL_STATE = 1e-3
TOL_STATE_BUILTIN = 1e-5

SPLITS = [(r, t) for t in (8, 4, 2) for r in range(1, 9)]


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401  (kda_mtp_decode)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_verify import (
        op,  # noqa: F401  (k3_kda_verify)
    )


def builtin_kda_variant() -> str:
    """'stock' when the loaded kda_mtp_decode has the stock signature, else 'accepted_by_slot' (a build whose
    kda_mtp_decode takes that argument)."""
    _ops()
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as mtp

    return (
        "accepted_by_slot"
        if "accepted_by_slot" in inspect.signature(mtp.kda_mtp_decode_impl).parameters
        else "stock"
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


class BuiltinPath:
    """The built-in verify (forward_verify_fused): its replay caches, a per-request pending count."""

    def __init__(self, wt, pools, slots, num_spec):
        self.wt, self.p, self.slots, self.num_spec = wt, pools, slots, num_spec
        self.p["k_cache"] = torch.zeros(POOL, num_spec, HK, device="cuda")
        self.p["g_cache"] = torch.zeros(POOL, num_spec, HK, device="cuda")
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
            dt_bias=self.wt["dt_bias"], recurrent_state=self.p["state"], k_cache=self.p["k_cache"],
            g_cache=self.p["g_cache"],
            v_cache=self.p["v_cache"], beta_cache=self.p["beta_cache"], ssm_state_indices=self.slots, cu_seqlens=cu,
            num_spec=self.num_spec, num_accepted_tokens=pending_req, lower_bound=LOWER_BOUND, scale=SCALE,
        )  # fmt: skip
        core = rms_norm_gated_token_major(o.reshape(-1, V), og.reshape(t, H, V), self.wt["onorm_w"], EPS,
                                          gate_activation="sigmoid")  # fmt: skip
        return core.view(t, H, V), g


class K3Path:
    """k3_kda_verify: the golden token's state and the drafts' records per slot, a per-slot pending count."""

    def __init__(self, wt, pools, slots, num_spec):
        self.wt, self.p, self.slots, self.num_spec = wt, pools, slots, num_spec
        self.p["state_tok"] = torch.zeros(POOL, 3, num_spec, H, K, device="cuda")
        self.p["pending"] = torch.zeros(POOL, dtype=torch.int32, device="cuda")

    def __call__(self, proj, g_ext=None):
        p = self.p
        return torch.ops.trtllm.k3_kda_verify(
            proj, self.wt["w_fb"], self.wt["w_q"], self.wt["w_k"], self.wt["w_v"], self.wt["a_log"],
            self.wt["dt_bias"], self.wt["onorm_w"], p["cs_q"], p["cs_k"], p["cs_v"], p["state"], p["state_tok"],
            self.slots, p["pending"], self.num_spec, LOWER_BOUND, SCALE, EPS, g_ext,
        )  # fmt: skip


def make_slots(num_requests: int, seed: int) -> torch.Tensor:
    perm = torch.randperm(POOL, generator=torch.Generator().manual_seed(seed))
    return perm[:num_requests].to(torch.int32).cuda()


def pending_schedule(num_requests: int, num_spec: int, rnd: int):
    """Accepted drafts per request after round ``rnd``: every count 0..num_spec, different per request."""
    return [(3 * n + 5 * rnd + 1 + (rnd * n) % 3) % (num_spec + 1) for n in range(num_requests)]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def run_schedule(
    num_requests, steps, fold=True, layout="dense", seed=0, rounds=ROUNDS, states=None
):
    """The built-in path, k3_kda_verify and the fp32 history over ``rounds`` rounds; per-round metrics."""
    _ops()
    num_spec = steps - 1
    wt = make_weights(100 + seed)
    pools = make_pools(200 + seed, num_spec, layout)
    slots = make_slots(num_requests, 300 + seed)
    builtin = BuiltinPath(wt, clone_pools(pools), slots, num_spec)
    k3 = K3Path(wt, clone_pools(pools), slots, num_spec)
    ref = kc.F64Verify(wt, dict(pools, ssm=pools["state"]), slots, num_spec)
    pending_req = torch.zeros(num_requests, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(400 + seed)
    rows, outs = [], []
    for rnd in range(rounds):
        proj = torch.randn(num_requests * steps, PROJ, generator=gen, device="cuda").bfloat16()
        out_builtin, g = builtin(proj, pending_req)
        out_k3 = k3(proj, None if fold else g.contiguous())
        out_ref = ref(proj)
        torch.cuda.synchronize()
        r = dict(round=rnd, pending=pending_req.tolist())
        r["out_vs_builtin"] = rel(out_k3, out_builtin)
        r["out_vs_fp32"] = rel(out_k3, out_ref)
        r["builtin_vs_fp32"] = rel(out_builtin, out_ref)
        # The pool state after the golden token (state_tok holds the drafts' compact records, which the next round's
        # outputs check against the history).
        st_err, builtin_st_err, st_builtin_err = 0.0, 0.0, 0.0
        for n, s in enumerate(slots.tolist()):
            states, _ = ref.last[n]
            st_err = max(st_err, rel(k3.p["state"][s], states[0]))
            builtin_st_err = max(builtin_st_err, rel(builtin.p["state"][s], states[0]))
            st_builtin_err = max(st_builtin_err, rel(k3.p["state"][s], builtin.p["state"][s]))
        r["state_vs_fp32"] = st_err
        r["builtin_state_vs_fp32"] = builtin_st_err
        r["state_vs_builtin"] = st_builtin_err
        r["conv_bits_vs_builtin"] = all(
            torch.equal(k3.p[c], builtin.p[c]) for c in ("cs_q", "cs_k", "cs_v")
        )
        rows.append(r)
        outs.append(out_k3.clone())
        if states is not None:
            states.append(k3.p["state"].clone())
        nxt = pending_schedule(num_requests, num_spec, rnd)
        pending_req.copy_(torch.tensor(nxt, dtype=torch.int32))
        k3.p["pending"][slots.long()] = pending_req
        ref.commit(nxt)
    return rows, outs


def round_ok(r, fold: bool) -> bool:
    ok = (r["out_vs_builtin"] <= TOL_OUT and r["out_vs_fp32"] <= TOL_OUT and r["state_vs_fp32"] <= TOL_STATE
          and r["conv_bits_vs_builtin"])  # fmt: skip
    if not fold:
        ok &= r["state_vs_builtin"] <= TOL_STATE_BUILTIN and r["builtin_state_vs_fp32"] <= TOL_STATE
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
    """An unfused gate matches the built-in path numerically and repeats bit-identically on the same
    implementation."""
    states, states2 = [], []
    with torch.inference_mode():
        rows, outs = run_schedule(
            num_requests, steps, fold=False, layout=layout, seed=1, states=states
        )
        again, outs2 = run_schedule(
            num_requests, steps, fold=False, layout=layout, seed=1, states=states2
        )
    for r in rows:
        assert round_ok(r, False), r
    assert rows == again, "the repeated g_ext state and cache metrics differ"
    assert all(torch.equal(a, b) for a, b in zip(outs, outs2)), "the repeated g_ext outputs differ"
    assert all(torch.equal(a, b) for a, b in zip(states, states2)), (
        "the repeated g_ext states differ"
    )


@pytest.mark.parametrize("num_requests,steps", [(4, 8), (8, 4)])
def test_strided_fold(num_requests, steps):
    with torch.inference_mode():
        rows, _ = run_schedule(num_requests, steps, fold=True, layout="strided", seed=2)
    for r in rows:
        assert round_ok(r, True), r


@pytest.mark.parametrize("num_requests,steps", [(4, 8), (8, 2)])
def test_isolation(num_requests, steps):
    """A change in request 0's rows changes only request 0's outputs, pool state and draft records."""
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
        print(f"{torch.cuda.get_device_name()}; kda_mtp_decode: {builtin_kda_variant()}")
        print("| split | mode | layout | rounds | k3 vs built-in | k3 vs fp32 | built-in vs fp32 "
              "| state vs fp32 | built-in state vs fp32 | state vs built-in | conv bits = built-in "
              "| deterministic | result |")  # fmt: skip
        print("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |")
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
                  f"{max(x['out_vs_builtin'] for x in rows):.2e} | {max(x['out_vs_fp32'] for x in rows):.2e} | "
                  f"{max(x['builtin_vs_fp32'] for x in rows):.2e} | {max(x['state_vs_fp32'] for x in rows):.2e} | "
                  f"{max(x['builtin_state_vs_fp32'] for x in rows):.2e} | "
                  f"{max(x['state_vs_builtin'] for x in rows):.2e} | {all(x['conv_bits_vs_builtin'] for x in rows)} | "
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
    print(f"{torch.cuda.get_device_name()}; kda_mtp_decode: {builtin_kda_variant()}; "
          f"graphs over {layers} layer copies, 15 replays: median (min-max) us per call")  # fmt: skip
    print("| split | built-in: f_b + kda_mtp_decode + gated norm | k3_kda_verify |")
    print("| :-- | --: | --: |")
    with torch.inference_mode():
        for r, t in SPLITS:
            num_spec = t - 1
            slots = make_slots(r, 1)
            builtin_paths, k3s = [], []
            for i in range(layers):
                wt = make_weights(1000 + i)
                pools = make_pools(2000 + i, num_spec, "dense")
                builtin_paths.append(BuiltinPath(wt, clone_pools(pools), slots, num_spec))
                k3s.append(K3Path(wt, pools, slots, num_spec))
            pend = torch.tensor(pending_schedule(r, num_spec, 2), dtype=torch.int32, device="cuda")
            for k3 in k3s:
                k3.p["pending"][slots.long()] = pend
            proj = torch.randn(r * t, PROJ, device="cuda").bfloat16()
            arms = [
                lambda i: builtin_paths[i % layers](proj, pend),
                lambda i: k3s[i % layers](proj),
            ]
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
            builtin_paths.clear()
            k3s.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    elif len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
