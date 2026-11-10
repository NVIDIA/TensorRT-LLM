# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_drafter_attn_qknorm catalog entry (sm_100)."""

import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_drafter_attn requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    # The CuTe DSL kernel uses tcgen05, TMA and clusters; it is certified on SM100 and SM103.
    pytest.skip("k3_drafter_attn is certified on sm_100 and sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_drafter_attn_qknorm import (  # noqa: E402
    k3_drafter_attn_qknorm,
)

__extra_import_path__ = ["../.."]
from _validating import validating  # noqa: E402

D = 64
PAGE = 64
# max |err| / max |ref|: the kernel rounds P and the output to bf16 (~4e-3 each).
TOL = 1e-2
# Context lengths, cycled over the requests: none, a block crossing a page, page and 128-row tile boundaries, the
# cluster's 16-tile round, several tiles per CTA.
LENGTHS = (60, 2041, 0, 127, 1000, 64, 128, 5)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class _Case:
    """R requests of T tokens: an HND pool [pages, 2, kv, 64, 64] that is NaN outside the requests' context rows
    (optionally with a page stride larger than a page), distinct random pages per request, page-table rows (a strided
    view into a wider table, or dense), qkv and lengths."""

    def __init__(
        self, seed, heads, kv, num_requests, tokens, page_pad=0, strided_table=True, lengths=None
    ):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.heads, self.kv, self.r, self.t = heads, kv, num_requests, tokens
        self.lengths = (
            list(lengths)
            if lengths is not None
            else [LENGTHS[(seed + 3 * r) % len(LENGTHS)] for r in range(num_requests)]
        )
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


@pytest.mark.parametrize("cell", k3_drafter_attn_qknorm.CELLS, ids=lambda c: c.why)
def test_k3_drafter_attn_qknorm(cell) -> None:
    """The raw projection output, normed and roped in the kernel, against the fp32 norm + RoPE + attention; ``qkv``
    left as it was; int64 positions give the same bits as int32."""
    seed = zlib.crc32(cell.why.encode()) % 2**31
    spec = cell.spec
    with torch.inference_mode():
        case = _Case(
            seed,
            spec["heads"],
            spec["kv"],
            spec["num_requests"],
            spec["tokens"],
            lengths=spec["lengths"],
        )
        gen = torch.Generator(device="cuda").manual_seed(seed + 1)
        raw = (case.qkv.float() * 4.0).bfloat16()
        q_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        k_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        raw_before = raw.clone()
        outs = []
        for dtype in spec["position_dtypes"]:
            out = case.out()
            with validating(k3_drafter_attn_qknorm):
                k3_drafter_attn_qknorm(
                    raw,
                    q_w,
                    k_w,
                    case.positions.to(dtype),
                    spec["eps"],
                    spec["rope_base"],
                    case.pool,
                    case.table,
                    case.ctx_len,
                    case.heads,
                    case.kv,
                    out,
                )
            outs.append(out)
        assert torch.equal(_bits(raw), _bits(raw_before)), "qkv was written"
        assert torch.equal(_bits(outs[1]), _bits(outs[0])), "int64 positions differ from int32"
        out = outs[0]
        want = case.out()
        k3_drafter_attn_qknorm.reference(
            raw,
            q_w,
            k_w,
            case.positions,
            spec["eps"],
            spec["rope_base"],
            case.pool,
            case.table,
            case.ctx_len,
            case.heads,
            case.kv,
            want,
        )
        err = _rel_err(out, want)
        assert err <= TOL, f"rel err {err:.3e}"
