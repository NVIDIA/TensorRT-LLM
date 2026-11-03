# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``kda_verify_rec`` (FP32 CUDA-core KDA verify) against an FP64 replay-semantics reference and the tensor-core
verify kernels, over chained rounds with mixed acceptance, in the runtime layouts (strided q / k / v / gate views of a
fused projection output, strided beta, strided state slots, dim-contiguous conv caches)."""

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_sm100f():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor in (100, 103)


pytestmark = [
    pytest.mark.skipif(not _is_sm100f(), reason="needs sm100 / sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

K = V = 128
W = 4
LB = -5.0


class _Layer:
    """One layer's verify operands in the runtime layouts."""

    def __init__(self, H, B, num_spec, slots, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        d = H * K
        T = B * (num_spec + 1)
        self.H, self.B, self.T, self.num_spec, self.d = H, B, T, num_spec, d

        def r(*s, dt=torch.bfloat16, sc=1.0):
            return (torch.randn(*s, generator=gen, device="cuda") * sc).to(dt)

        self.qkvg = r(T, 4 * d)
        bfa_w = (K + H + 7) // 8 * 8
        self.bfa = r(T, bfa_w)
        self.w_fb = r(d, K, sc=0.05).contiguous()
        self.w_q, self.w_k, self.w_v = (
            r(d, W, dt=torch.float32, sc=0.5).contiguous() for _ in range(3)
        )
        self.A_log = r(H, dt=torch.float32, sc=0.3)
        self.dt_bias = r(d, dt=torch.float32, sc=0.3)
        self.onorm_w = (1.0 + 0.1 * r(V, dt=torch.float32)).contiguous()
        S = W - 1 + num_spec
        self.cs = [r(slots, S, d).transpose(1, 2) for _ in range(3)]
        # every other slot page of a larger buffer, like the runtime's per-layer state views
        self.state_buf = torch.randn(slots, 2, H, V, K, generator=gen, device="cuda") * 0.05
        self.state = self.state_buf[:, 0]
        self.k_cache = r(slots, num_spec, d, sc=0.09)
        self.g_cache = (torch.rand(slots, num_spec, d, generator=gen, device="cuda") * -0.3).float()
        self.v_cache = r(slots, num_spec, d)
        self.beta_buf = torch.rand(
            slots, num_spec, (H + 7) // 8 * 8, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        self.beta_cache = self.beta_buf[..., :H]

    def new_tokens(self, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.qkvg.copy_(
            torch.randn(self.qkvg.shape, generator=gen, device="cuda").to(torch.bfloat16)
        )
        self.bfa.copy_(torch.randn(self.bfa.shape, generator=gen, device="cuda").to(torch.bfloat16))
        self.g = torch.nn.functional.linear(self.bfa[:, :K], self.w_fb)

    def views(self):
        H, T, d = self.H, self.T, self.d
        return dict(
            x_q=self.qkvg[:, :d].view(1, T, H, K),
            x_k=self.qkvg[:, d : 2 * d].view(1, T, H, K),
            x_v=self.qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
            onorm_g=self.qkvg[:, 3 * d :].view(1, T, H, V),
            beta=self.bfa[:, K : K + H].view(1, T, H),
            g=self.g.view(1, T, H, K),
        )

    def pools(self):
        return [self.state_buf, self.k_cache, self.g_cache, self.v_cache, self.beta_buf, *self.cs]

    def snapshot(self):
        return [t.clone() for t in self.pools()]

    def restore(self, snap):
        for t, s in zip(self.pools(), snap):
            t.copy_(s)


def _args(L, slot_idx, num_acc):
    v = L.views()
    return dict(
        x_q=v["x_q"],
        x_k=v["x_k"],
        x_v=v["x_v"],
        w_q=L.w_q,
        w_k=L.w_k,
        w_v=L.w_v,
        cs_q=L.cs[0],
        cs_k=L.cs[1],
        cs_v=L.cs[2],
        beta=v["beta"],
        A_log=L.A_log,
        dt_bias=L.dt_bias,
        recurrent_state=L.state,
        k_cache=L.k_cache,
        g_cache=L.g_cache,
        v_cache=L.v_cache,
        beta_cache=L.beta_cache,
        ssm_state_indices=slot_idx,
        num_spec=L.num_spec,
        num_accepted_tokens=num_acc,
        lower_bound=LB,
        scale=K**-0.5,
        onorm_g=v["onorm_g"],
        onorm_weight=L.onorm_w,
        onorm_eps=1e-6,
    )


def _run_rec(L, slot_idx, num_acc, **kw):
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_verify_rec import (
        kda_verify_rec,
    )

    args = _args(L, slot_idx, num_acc)
    args.update(g=L.views()["g"])
    return kda_verify_rec(**args, **kw)


def _run_tc(L, slot_idx, num_acc):
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import kda_mtp_decode_impl

    return kda_mtp_decode_impl(
        **_args(L, slot_idx, num_acc),
        g=L.views()["g"],
        cu_seqlens=None,
        packed_token_layout=True,
        fuse_output_norm=True,
    )


def _reference(L, slot_idx, num_acc):
    """FP64 replay semantics; returns the output (before the BF16 store), the committed state of every slot and the
    new replay-cache rows (in their cache dtypes) and raw conv columns."""
    H, B, M = L.H, L.B, L.num_spec
    NT = M + 1
    f = lambda t: t.double()  # noqa: E731
    v = L.views()
    state = f(L.state).clone()
    cs = [f(c).clone() for c in L.cs]
    kc, gc, vc, bc = (f(t).clone() for t in (L.k_cache, L.g_cache, L.v_cache, L.beta_cache))
    w = [f(x).view(H, K, W) for x in (L.w_q, L.w_k, L.w_v)]
    xq, xk, xv = (f(v[n])[0] for n in ("x_q", "x_k", "x_v"))
    graw, braw, og = f(v["g"])[0], f(v["beta"])[0], f(v["onorm_g"])[0]
    A = torch.exp(f(L.A_log)).view(H, 1)
    dtb = f(L.dt_bias).view(H, K)
    norm_w = f(L.onorm_w)
    out = torch.zeros(B * NT, H, V, dtype=torch.float64, device="cuda")
    for n, (s, a) in enumerate(zip(slot_idx.tolist(), num_acc[slot_idx.long()].tolist())):
        a = min(a, M)
        h = state[s].transpose(1, 2).clone()
        hist = [c[s].view(H, K, -1)[:, :, : W - 1].clone() for c in cs]
        tail = [[] for _ in range(3)]
        newk, newg, newv, newb = [], [], [], []
        committed_hist = None
        for i in range(a + NT):
            if i < a:
                kt, gt, vt, bt = (
                    kc[s, i].view(H, K),
                    gc[s, i].view(H, K),
                    vc[s, i].view(H, V),
                    bc[s, i],
                )
                raws = [c[s].view(H, K, -1)[:, :, W - 1 + i] for c in cs]
                qt = None
            else:
                r = n * NT + i - a
                raws = [xq[r], xk[r], xv[r]]
                conv = []
                for j in range(3):
                    y = (torch.cat([hist[j], raws[j].unsqueeze(-1)], dim=-1) * w[j]).sum(-1)
                    conv.append(y * torch.sigmoid(y))
                qt = conv[0] / torch.sqrt((conv[0] ** 2).sum(-1, keepdim=True) + 1e-6) * K**-0.5
                kt = conv[1] / torch.sqrt((conv[1] ** 2).sum(-1, keepdim=True) + 1e-6)
                vt, gt, bt = (
                    conv[2],
                    LB * torch.sigmoid((graw[r] + dtb) * A),
                    torch.sigmoid(braw[r]),
                )
            hist = [torch.cat([hist[j][:, :, 1:], raws[j].unsqueeze(-1)], dim=-1) for j in range(3)]
            h = h * torch.exp(gt).unsqueeze(-1)
            vn = (vt - torch.einsum("hkv,hk->hv", h, kt)) * bt.unsqueeze(-1)
            h = h + kt.unsqueeze(-1) * vn.unsqueeze(1)
            if qt is not None:
                o = torch.einsum("hkv,hk->hv", h, qt).to(torch.bfloat16).double()
                rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6)
                out[n * NT + i - a] = o * norm_w * torch.sigmoid(og[n * NT + i - a]) * rms
            if i == a:
                state[s] = h.transpose(1, 2)
                committed_hist = [x.clone() for x in hist]
            if i > a:
                newk.append(kt.reshape(-1))
                newg.append(gt.reshape(-1))
                newv.append(vt.reshape(-1))
                newb.append(bt.reshape(-1))
                for j in range(3):
                    tail[j].append(raws[j].reshape(-1))
        cs_new = [
            torch.cat([committed_hist[j].reshape(H * K, -1), torch.stack(tail[j], -1)], -1)
            for j in range(3)
        ]
        kc[s], gc[s], vc[s], bc[s] = (
            torch.stack(newk),
            torch.stack(newg),
            torch.stack(newv),
            torch.stack(newb),
        )
        for j in range(3):
            cs[j][s] = cs_new[j]
    return out, state, kc, gc, vc, bc, cs


def _check_round(L, slot_idx, num_acc, out, ref):
    ref_out, ref_state, kc, gc, vc, bc, cs = ref
    s = slot_idx.long()
    st = L.state[s].double()
    rel = ((st - ref_state[s]).abs().max() / ref_state[s].abs().max()).item()
    assert rel < 2e-6, f"committed state rel err {rel}"
    torch.testing.assert_close(out.double().view_as(ref_out), ref_out, atol=3e-2, rtol=2e-2)
    # replay caches in their dtypes: k within one BF16 step (the FP32 norm), the others exact up to FP32 rounding
    torch.testing.assert_close(
        L.k_cache[s].double(), kc[s].to(torch.bfloat16).double(), atol=1e-5, rtol=8e-3
    )
    torch.testing.assert_close(L.g_cache[s].double(), gc[s], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        L.v_cache[s].double(), vc[s].to(torch.bfloat16).double(), atol=1e-5, rtol=8e-3
    )
    torch.testing.assert_close(
        L.beta_cache[s].double(), bc[s].to(torch.bfloat16).double(), atol=1e-5, rtol=8e-3
    )
    for j in range(3):
        assert torch.equal(L.cs[j][s].double(), cs[j][s]), f"conv cache {j}"


def _meta(B, slots, M, rnd, seed):
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))[:B] + 1
    slot_idx = perm.to(torch.int32).cuda()
    num_acc = torch.zeros(slots, dtype=torch.int32, device="cuda")
    if rnd:
        vals = torch.randint(0, M + 1, (B,), generator=torch.Generator().manual_seed(seed + rnd))
        num_acc[slot_idx.long()] = vals.to(torch.int32).cuda()
    return slot_idx, num_acc


@torch.no_grad()
@pytest.mark.parametrize(
    "H,num_spec,B,cluster",
    [
        (12, 6, 1, None),
        (12, 6, 4, None),
        (12, 6, 16, None),
        (12, 6, 4, 2),
        (12, 6, 4, 4),
        (6, 5, 8, None),
        (6, 5, 24, None),
        (12, 7, 3, None),
        (12, 2, 5, None),
        (8, 6, 2, 8),
    ],
)
def test_matches_reference_over_rounds(H, num_spec, B, cluster):
    """Three chained rounds (nothing pending, then random accepted counts) against the FP64 reference; slots outside
    the batch stay untouched."""
    slots = 2 * B + 3
    L = _Layer(H, B, num_spec, slots, seed=11 * B + H)
    for rnd in range(3):
        slot_idx, num_acc = _meta(B, slots, num_spec, rnd, seed=B)
        L.new_tokens(seed=100 * rnd + B)
        before = L.snapshot()
        ref = _reference(L, slot_idx, num_acc)
        out = _run_rec(L, slot_idx, num_acc, cluster=cluster, use_pdl=True)
        torch.cuda.synchronize()
        _check_round(L, slot_idx, num_acc, out, ref)
        others = torch.ones(slots, dtype=torch.bool, device="cuda")
        others[slot_idx.long()] = False
        for t, b in zip(L.pools(), before):
            assert torch.equal(t[others], b[others]), "a slot outside the batch changed"


@torch.no_grad()
@pytest.mark.parametrize("H,num_spec,B", [(12, 6, 4), (6, 5, 24)])
def test_caches_match_tensor_core_kernels(H, num_spec, B):
    """The conv windows / tails and the gate / value / beta caches are bitwise those of the tensor-core verify kernels
    (``kda_mtp_decode_impl``), outputs to BF16 rounding, the committed state to FP32 rounding."""
    slots = B + 5
    La = _Layer(H, B, num_spec, slots, seed=5)
    Lb = _Layer(H, B, num_spec, slots, seed=5)
    for rnd in range(3):
        slot_idx, num_acc = _meta(B, slots, num_spec, rnd, seed=3)
        La.new_tokens(seed=rnd)
        Lb.new_tokens(seed=rnd)
        oa = _run_rec(La, slot_idx, num_acc)
        ob = _run_tc(Lb, slot_idx, num_acc)
        torch.cuda.synchronize()
        s = slot_idx.long()
        for name, a, b in zip(
            ("g_cache", "v_cache", "beta", "cs_q", "cs_k", "cs_v"),
            [La.g_cache, La.v_cache, La.beta_buf, *La.cs],
            [Lb.g_cache, Lb.v_cache, Lb.beta_buf, *Lb.cs],
        ):
            assert torch.equal(a[s], b[s]), name
        torch.testing.assert_close(
            La.k_cache[s].float(), Lb.k_cache[s].float(), atol=1e-5, rtol=8e-3
        )
        torch.testing.assert_close(oa.float(), ob.float(), atol=3e-2, rtol=2e-2)
        torch.testing.assert_close(La.state[s], Lb.state[s], atol=1e-4, rtol=1e-4)
        # carry the same caches into the next round (the k cache may differ in a last BF16 bit)
        Lb.restore(La.snapshot())


@torch.no_grad()
def test_cuda_graph_replay():
    """Captured once, replayed over rounds: the same results as eager calls."""
    H, num_spec, B, slots = 12, 6, 4, 9
    La = _Layer(H, B, num_spec, slots, seed=21)
    Lb = _Layer(H, B, num_spec, slots, seed=21)
    slot_idx, num_acc = _meta(B, slots, num_spec, 1, seed=4)
    La.new_tokens(seed=0)
    Lb.new_tokens(seed=0)
    out_b = torch.empty(1, La.T, H, V, dtype=torch.bfloat16, device="cuda")
    snap = Lb.snapshot()
    _run_rec(Lb, slot_idx, num_acc, out=out_b, use_pdl=True)  # compile
    Lb.restore(snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_rec(Lb, slot_idx, num_acc, out=out_b, use_pdl=True)
    for rnd in range(3):
        oa = _run_rec(La, slot_idx, num_acc, use_pdl=True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(oa, out_b)
        for a, b in zip(La.pools(), Lb.pools()):
            assert torch.equal(a, b)


def test_supported():
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_verify_rec import supported

    st = torch.zeros(4, 12, V, K, device="cuda")
    assert supported(st, 6, K, V, W)
    assert not supported(st.to(torch.bfloat16), 6, K, V, W)
    assert not supported(st, 8, K, V, W)
    assert not supported(st, 6, 64, V, W)
