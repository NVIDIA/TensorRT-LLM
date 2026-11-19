# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One KDA layer's verify operands in the runtime layouts, for the recurrent-verify tests.

Strided q / k / v / output-gate views of one fused projection output, a strided beta slice of the ``[f_a | b]``
output, a strided state slot view of a larger buffer, dim-contiguous conv caches and a padded beta cache.
"""

import torch

K = V = 128
W = 4
LOWER_BOUND = -5.0


class RecLayer:
    def __init__(self, H, B, num_spec, slots, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        d = H * K
        T = B * (num_spec + 1)
        self.H, self.B, self.T, self.num_spec, self.d, self.slots = H, B, T, num_spec, d, slots

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
        # Every other slot page of a larger buffer, like the runtime's per-layer state views.
        self.state_buf = torch.randn(slots, 2, H, V, K, generator=gen, device="cuda") * 0.05
        self.state = self.state_buf[:, 0]
        self.k_cache = r(slots, num_spec, d, sc=0.09)
        self.g_cache = (torch.rand(slots, num_spec, d, generator=gen, device="cuda") * -0.3).float()
        self.v_cache = r(slots, num_spec, d)
        self.beta_buf = torch.rand(
            slots, num_spec, (H + 7) // 8 * 8, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        self.beta_cache = self.beta_buf[..., :H]
        self.out = torch.empty(1, T, H, V, dtype=torch.bfloat16, device="cuda")

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

    def args(self, slot_idx, num_acc):
        """The op's arguments, in its parameter order (``out`` last)."""
        v = self.views()
        return dict(
            x_q=v["x_q"],
            x_k=v["x_k"],
            x_v=v["x_v"],
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            cs_q=self.cs[0],
            cs_k=self.cs[1],
            cs_v=self.cs[2],
            g=v["g"],
            beta=v["beta"],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            recurrent_state=self.state,
            k_cache=self.k_cache,
            g_cache=self.g_cache,
            v_cache=self.v_cache,
            beta_cache=self.beta_cache,
            ssm_state_indices=slot_idx,
            num_spec=self.num_spec,
            num_accepted_tokens=num_acc,
            lower_bound=LOWER_BOUND,
            scale=K**-0.5,
            onorm_g=v["onorm_g"],
            onorm_weight=self.onorm_w,
            onorm_eps=1e-6,
            out=self.out,
        )


def metadata(B, slots, num_spec, rnd, seed):
    """Slots of the batch (never slot 0) and the slot-indexed accepted counts: none pending in round 0."""
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))[:B] + 1
    slot_idx = perm.to(torch.int32).cuda()
    num_acc = torch.zeros(slots, dtype=torch.int32, device="cuda")
    if rnd:
        vals = torch.randint(
            0, num_spec + 1, (B,), generator=torch.Generator().manual_seed(seed + rnd)
        )
        num_acc[slot_idx.long()] = vals.to(torch.int32).cuda()
    return slot_idx, num_acc


def check_against_reference(layer, slot_idx, ref):
    """The op's in-place results vs the FP64 reference (``VerifyRecResult``), with the bands the kernel's FP32
    arithmetic and the buffers' dtypes allow."""
    s = slot_idx.long()
    st = layer.state[s].double()
    rel = ((st - ref.recurrent_state[s]).abs().max() / ref.recurrent_state[s].abs().max()).item()
    assert rel < 2e-6, f"committed state rel err {rel}"
    torch.testing.assert_close(layer.out.double(), ref.out, atol=3e-2, rtol=2e-2)
    # Replay caches in their dtypes: k within one BF16 step (the FP32 norm), the others exact up to FP32 rounding.
    torch.testing.assert_close(
        layer.k_cache[s].double(), ref.k_cache[s].to(torch.bfloat16).double(), atol=1e-5, rtol=8e-3
    )
    torch.testing.assert_close(layer.g_cache[s].double(), ref.g_cache[s], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        layer.v_cache[s].double(), ref.v_cache[s].to(torch.bfloat16).double(), atol=1e-5, rtol=8e-3
    )
    torch.testing.assert_close(
        layer.beta_cache[s].double(),
        ref.beta_cache[s].to(torch.bfloat16).double(),
        atol=1e-5,
        rtol=8e-3,
    )
    for j, name in enumerate(("cs_q", "cs_k", "cs_v")):
        assert torch.equal(layer.cs[j][s].double(), getattr(ref, name)[s]), name
