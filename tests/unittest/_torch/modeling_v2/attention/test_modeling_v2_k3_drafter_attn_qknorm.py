# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_drafter_attn_qknorm catalog entry (sm_100)."""

import pytest
import torch

assert torch.cuda.is_available(), "k3_drafter_attn requires a CUDA device"

if torch.cuda.get_device_capability() != (10, 0):
    # The CuTe DSL kernel uses tcgen05, TMA and clusters; it is certified on sm_100 (B200 / GB200) only.
    pytest.skip("k3_drafter_attn is certified on sm_100 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_drafter_attn_qknorm import (  # noqa: E402
    k3_drafter_attn_qknorm,
)

D = 64
PAGE = 64
EPS = 1e-5
THETA = 10000.0
# max |err| / max |ref|: the kernel rounds P and the output to bf16 (~4e-3 each).
TOL = 1e-2
# Context lengths, cycled over the requests: none, a block crossing a page, page and 128-row tile boundaries, the
# cluster's 16-tile round, several tiles per CTA.
LENGTHS = (60, 2041, 0, 127, 1000, 64, 128, 5)
# R x 7: DSpark's block at max_draft_len 7 (max_draft_len tokens under shift_label).
SPLITS = ((1, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 1), (2, 8), (8, 8)) + tuple(
    (r, 7) for r in range(1, 9)
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class _Case:
    """R requests of T tokens: an HND pool [pages, 2, kv, 64, 64] that is NaN outside the requests' context rows
    (optionally with a page stride larger than a page), distinct random pages per request, page-table rows (a strided
    view into a wider table, or dense), qkv and lengths."""

    def __init__(self, seed, heads, kv, num_requests, tokens, page_pad=0, strided_table=True):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.heads, self.kv, self.r, self.t = heads, kv, num_requests, tokens
        self.lengths = [LENGTHS[(seed + 3 * r) % len(LENGTHS)] for r in range(num_requests)]
        need = [(c + PAGE - 1) // PAGE for c in self.lengths]
        self.width = max(need) + 2
        n_pool = sum(need) + 2 * num_requests + 4
        inner = 2 * kv * PAGE * D
        flat = torch.full(
            (n_pool * (inner + page_pad),), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        self.pool = flat.as_strided(
            (n_pool, 2, kv, PAGE, D), (inner + page_pad, kv * PAGE * D, PAGE * D, D, 1)
        )
        perm = torch.randperm(n_pool, generator=torch.Generator().manual_seed(seed))
        rows, used = [], 0
        for n in need:
            rows.append(torch.cat([perm[used : used + n], perm[-(self.width - n) :]]))
            used += n
        dense = torch.stack(rows).to(torch.int32).cuda()
        if strided_table:
            full = torch.full(
                (num_requests + 2, self.width + 3), -1, dtype=torch.int32, device="cuda"
            )
            full[1 : num_requests + 1, : self.width] = dense
            self.table = full[1 : num_requests + 1, : self.width]
        else:
            self.table = dense
        for r, c in enumerate(self.lengths):
            for i in range((c + PAGE - 1) // PAGE):
                n_rows = min(PAGE, c - i * PAGE)
                p = int(self.table[r, i])
                self.pool[p, :, :, :n_rows] = (
                    torch.randn(2, kv, n_rows, D, generator=gen, device="cuda") * 0.5
                ).bfloat16()
        m = num_requests * tokens
        self.qkv = (
            torch.randn(m, (heads + 2 * kv) * D, generator=gen, device="cuda") * 0.5
        ).bfloat16()
        self.ctx_len = torch.tensor(self.lengths, dtype=torch.int32, device="cuda")
        self.positions = (
            self.ctx_len.view(-1, 1) + torch.arange(tokens, device="cuda", dtype=torch.int32)
        ).reshape(-1)

    def out(self) -> torch.Tensor:
        return torch.empty(self.r * self.t, self.heads * D, dtype=torch.bfloat16, device="cuda")


def _attention_ref(case: _Case, qkv: torch.Tensor) -> torch.Tensor:
    """fp64: each request's q against its cached rows and its block's own k / v, softmax(q k^T / 8) v."""
    h, kv, t = case.heads, case.kv, case.t
    outs = []
    for r, c in enumerate(case.lengths):
        rows = qkv[r * t : (r + 1) * t].double()
        q = rows[:, : h * D].view(t, h, D)
        k_blk = rows[:, h * D : (h + kv) * D].view(t, kv, D).permute(1, 0, 2)
        v_blk = rows[:, (h + kv) * D :].view(t, kv, D).permute(1, 0, 2)
        pages = case.table[r, : (c + PAGE - 1) // PAGE].long()
        k_ctx = case.pool[pages, 0].double().permute(1, 0, 2, 3).reshape(kv, -1, D)[:, :c]
        v_ctx = case.pool[pages, 1].double().permute(1, 0, 2, 3).reshape(kv, -1, D)[:, :c]
        k = torch.cat([k_ctx, k_blk], 1).repeat_interleave(h // kv, 0)
        v = torch.cat([v_ctx, v_blk], 1).repeat_interleave(h // kv, 0)
        p = torch.softmax(torch.einsum("thd,hld->thl", q, k) / D**0.5, dim=-1)
        outs.append(torch.einsum("thl,hld->thd", p, v).reshape(t, h * D))
    return torch.cat(outs)


def _qk_norm_rope_ref(case: _Case, qkv: torch.Tensor, q_w, k_w) -> torch.Tensor:
    """Per-head RMSNorm of the q and k heads, then NeoX RoPE (base THETA) at each row's position, in fp64 and rounded
    to bf16 as fused_qk_norm_rope stores them; v as is."""
    h, kv = case.heads, case.kv
    x = qkv.double().view(qkv.shape[0], h + 2 * kv, D).clone()
    inv_freq = 1.0 / THETA ** (torch.arange(0, D, 2, dtype=torch.float64, device="cuda") / D)
    angle = case.positions.double()[:, None] * inv_freq
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    for start, count, w in ((0, h, q_w), (h, kv, k_w)):
        y = x[:, start : start + count]
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
        y1, y2 = y[..., : D // 2], y[..., D // 2 :]
        x[:, start : start + count] = torch.cat([y1 * cos - y2 * sin, y1 * sin + y2 * cos], -1)
    return x.view(qkv.shape[0], -1).bfloat16()


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_k3_drafter_attn_qknorm(num_requests, tokens) -> None:
    """The raw projection output, normed and roped in the kernel, against the fp64 norm + RoPE + attention; ``qkv``
    left as it was; int64 positions give the same bits as int32."""
    with torch.inference_mode():
        case = _Case(300 + 10 * num_requests + tokens, 6, 1, num_requests, tokens)
        gen = torch.Generator(device="cuda").manual_seed(num_requests * 9 + tokens)
        raw = (case.qkv.float() * 4.0).bfloat16()
        q_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        k_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        raw_before = raw.clone()
        out, out64 = case.out(), case.out()
        k3_drafter_attn_qknorm(
            raw,
            q_w,
            k_w,
            case.positions,
            EPS,
            THETA,
            case.pool,
            case.table,
            case.ctx_len,
            6,
            1,
            out,
        )
        k3_drafter_attn_qknorm(raw, q_w, k_w, case.positions.long(), EPS, THETA, case.pool, case.table, case.ctx_len,
                               6, 1, out64)  # fmt: skip
        assert torch.equal(_bits(raw), _bits(raw_before)), "qkv was written"
        assert torch.equal(_bits(out64), _bits(out)), "int64 positions differ from int32"
        want = _attention_ref(case, _qk_norm_rope_ref(case, raw, q_w, k_w))
        err = _rel_err(out, want)
        assert err <= TOL, f"rel err {err:.3e}"
