# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/k3_ctx_kv entry of the tp16_moetp4ep4 target.

Every cell at the TP16 shape (5 drafter layers, 1 KV head, hidden 7168, page 64) on a page-major pool (each layer at
its offset inside every page group, one allocation): B requests with distinct slots and table rows, lengths whose
columns cross a page, mixed accepted counts. The op and its reference run on copies of the same pool and ctx_len.
Checked: ctx_len and num_ctx exactly; every pool element no token addresses bit for bit unchanged by both; the masked
tokens' rows zero; the live tokens' rows (each at a column of its own) within the band below.
"""

import importlib
import zlib

import pytest
import torch

__extra_import_path__ = ["../../.."]
from _validating import validating

_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4"
k3_ctx_kv = importlib.import_module(f"{_TARGET}.ops.attention.k3_ctx_kv").k3_ctx_kv
_kernel_op = importlib.import_module(f"{_TARGET}.kernels.k3_ctx_kv.op")

HIDDEN = 7168
HEAD = 64
LAYERS = 5
NKV = 1
PAGE = 64
EPS = 1e-6
MAX_POS = 8192
MAX_CTX = 4096
THETA = 1.0e6
WIDTH = 64  # table width in pages: an allocation of WIDTH PAGE = MAX_CTX columns

# The kernel test's band against its fp32 rows is max(2 e_python, 1.6e-2) of each row's largest magnitude, where
# e_python is the bf16 Python path's own error (below 8e-3 at these shapes, so the 1.6e-2 floor applies). The
# reference here rounds the fp32 rows once to bf16, which moves each element by at most half a bf16 ulp of the row's
# largest magnitude (2^-8 of it), so the op must sit within 1.6e-2 + 2^-8 of the reference.
TOL = 1.6e-2 + 2.0**-8


def _sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm10x(), reason="needs SM100 / SM103 (tcgen05, TMA, clusters)")


def _cos_sin(device) -> torch.Tensor:
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD, 2, dtype=torch.float64, device=device) / HEAD))
    ang = torch.arange(MAX_POS, dtype=torch.float64, device=device)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().contiguous()


def _views(buf: torch.Tensor) -> list:
    """Page-major per-layer HND views [pages, 2, NKV, PAGE, 64] of ``buf``: layer l at offset l per_page of every
    page group."""
    per_page = 2 * NKV * PAGE * HEAD
    pages = buf.numel() // (LAYERS * per_page)
    return [
        buf.as_strided(
            (pages, 2, NKV, PAGE, HEAD),
            (LAYERS * per_page, NKV * PAGE * HEAD, PAGE * HEAD, HEAD, 1),
            layer * per_page,
        )
        for layer in range(LAYERS)
    ]


class Step:
    """One decode step's inputs for a cell (the kernel test's ``Step``, page-major pool, TP16)."""

    def __init__(self, spec, seed):
        batch, k1 = spec["batch"], spec["k1"]
        gen = torch.Generator(device="cuda").manual_seed(seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.batch, self.k1, self.block = batch, k1, spec["block"]
        self.n = batch * k1
        n_rows = LAYERS * 2 * NKV * HEAD
        self.w = (torch.randn(n_rows, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
        self.k_norm = (
            1.0 + 0.1 * torch.randn(LAYERS, HEAD, generator=gen, device="cuda")
        ).bfloat16()
        self.cs = _cos_sin("cuda")
        table_rows, slots_total = batch + 3, batch + 4
        pages = WIDTH * table_rows + 16
        self.table = (
            torch.randperm(pages, generator=g)[: table_rows * WIDTH]
            .view(table_rows, WIDTH)
            .to(torch.int32)
            .cuda()
        )
        self.counts = torch.full((table_rows,), WIDTH, dtype=torch.int64, device="cuda")
        self.slots = torch.randperm(slots_total, generator=g)[:batch].cuda()
        self.rows = torch.randperm(table_rows, generator=g)[:batch].cuda()
        # Lengths: a page end inside the block for every other request (64 m - 3), else random.
        ctx0 = torch.randint(100, 2000, (slots_total,), generator=g)
        for b in range(0, batch, 2):
            ctx0[int(self.slots[b])] = PAGE * int(torch.randint(2, 30, (1,), generator=g)) - 3
        num_acc = (torch.arange(batch) * 3 + seed) % k1 + 1
        if spec.get("clamp"):  # the first request's last columns clamp to its allocation's end
            self.counts[int(self.rows[0])] = 20
            ctx0[int(self.slots[0])] = 20 * PAGE - 3
        if spec.get("near_max"):  # requests 2 and 3 accept every token across max_ctx
            for b in (2, 3):
                ctx0[int(self.slots[b])] = MAX_CTX - 2 - b
                num_acc[b] = k1
        self.ctx0 = ctx0.cuda()
        self.num_acc = num_acc.to(torch.int32).cuda()
        self.cpos = (
            self.ctx0[self.slots][:, None] + torch.arange(k1, device="cuda")[None, :]
        ).contiguous()
        self.x = (torch.randn(self.n, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
        self.buf = torch.randn(
            pages * LAYERS * 2 * NKV * PAGE * HEAD, generator=gen, device="cuda"
        ).bfloat16()

    def args(self, buf: torch.Tensor, ctx: torch.Tensor) -> tuple:
        base, layer_off, page_stride, kv_stride, head_stride = _kernel_op.pool_view(_views(buf))
        return (
            self.x, self.w, self.k_norm, self.cs, self.cpos, self.num_acc, ctx, self.slots, self.rows, self.table,
            self.counts, base, layer_off, page_stride, kv_stride, head_stride, EPS, MAX_CTX, PAGE, self.block, NKV,
        )  # fmt: skip

    def cols(self) -> torch.Tensor:
        col = self.ctx0[self.slots][:, None] + torch.arange(self.k1, device="cuda")[None, :]
        return (
            torch.minimum(col, (self.counts[self.rows] * PAGE - 1)[:, None])
            .clamp(min=0)
            .reshape(-1)
        )

    def rows_at(self, buf: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """The pool rows of each (token, layer, K/V, head): [N, L, 2, NKV, 64]."""
        page = self.table[
            self.rows[:, None].expand(self.batch, self.k1).reshape(-1), col // PAGE
        ].long()
        return torch.stack([v[page, :, :, col % PAGE, :] for v in _views(buf)], dim=1)

    def written(self, col: torch.Tensor) -> torch.Tensor:
        """Whether a token's rows address each pool element: bool, the shape of ``buf``."""
        idx = torch.arange(self.buf.numel(), device="cuda")
        mask = torch.zeros(self.buf.numel(), dtype=torch.bool, device="cuda")
        mask[self.rows_at(idx, col).reshape(-1)] = True
        return mask


@pytest.mark.parametrize(
    "cell",
    k3_ctx_kv.CELLS,
    ids=[f"{c.spec['batch']}x{c.spec['k1']}" + "".join(f"-{k}" for k in ("clamp", "near_max") if c.spec.get(k))
         for c in k3_ctx_kv.CELLS],
)  # fmt: skip
def test_ctx_kv_cells(cell) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    with torch.inference_mode():
        st = Step(cell.spec, seed)
        buf_op, ctx_op = st.buf.clone(), st.ctx0.clone()
        buf_ref, ctx_ref = st.buf.clone(), st.ctx0.clone()
        with validating(k3_ctx_kv):
            nc_op = k3_ctx_kv(*st.args(buf_op, ctx_op))
        nc_ref = k3_ctx_kv.reference(*st.args(buf_ref, ctx_ref))
        torch.cuda.synchronize()

        assert nc_op.dtype == nc_ref.dtype == torch.int32
        assert torch.equal(ctx_op, ctx_ref), (ctx_op[st.slots], ctx_ref[st.slots])
        assert torch.equal(nc_op, nc_ref), (nc_op, nc_ref)
        if cell.spec.get("near_max"):
            # The cell reaches the clamp it is there for.
            assert int(ctx_ref[st.slots].max()) == MAX_CTX

        col = st.cols()
        written = st.written(col)
        before = st.buf.view(torch.int16)
        for name, buf in (("op", buf_op), ("reference", buf_ref)):
            changed = buf.view(torch.int16) != before
            assert not bool((changed & ~written).any()), (
                f"{name} wrote an element no token addresses"
            )

        # A clamped column is written by several tokens (which one lands is not fixed): compare only the tokens
        # whose column is their own.
        key = st.rows[:, None].expand(st.batch, st.k1).reshape(-1) * 1_000_000 + col
        own = (key[:, None] == key[None, :]).sum(1) == 1
        offs = torch.arange(st.k1, device="cuda")
        live = (offs[None, :] < st.num_acc.long()[:, None]).reshape(-1)
        got = st.rows_at(buf_op, col).float()
        want = st.rows_at(buf_ref, col).float()
        masked = ~live & own
        assert bool((got[masked] == 0).all()) and bool((want[masked] == 0).all()), (
            "masked rows not zero"
        )
        cmp = live & own
        assert bool(cmp.any())
        scale = want.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        err = ((got - want).abs() / scale)[cmp].max().item()
        assert err <= TOL, f"op vs reference {err:.3e} > {TOL:.3e}"


def _bad_args(st: Step, buf: torch.Tensor, ctx: torch.Tensor, what: str) -> tuple:
    args = list(st.args(buf, ctx))
    if what == "slots":
        args[7] = st.slots.clone()
        args[7][1] = args[7][0]
    elif what == "cpos":
        args[4] = st.cpos.clone()
        args[4][0, 0] = MAX_POS
    elif what == "table":
        args[9] = st.table.clone()
        args[9][st.rows[0], int(st.cols()[0]) // PAGE] = 1 << 24  # the first token's page
    return tuple(args)


@pytest.mark.parametrize("what", ["slots", "cpos", "table"])
def test_is_valid_rejects(what) -> None:
    """Repeated slots, a RoPE position past ``cos_sin`` and a table page past the pool are refused; the same step
    with its own inputs is admitted."""
    with torch.inference_mode():
        st = Step(dict(batch=2, k1=8, block=8), seed=11)
        buf, ctx = st.buf.clone(), st.ctx0.clone()
        k3_ctx_kv.is_valid(*st.args(buf, ctx))
        with pytest.raises(ValueError, match="k3_ctx_kv"):
            k3_ctx_kv.is_valid(*_bad_args(st, buf, ctx, what))
