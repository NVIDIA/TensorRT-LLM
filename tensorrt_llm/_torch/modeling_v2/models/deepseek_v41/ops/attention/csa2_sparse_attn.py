# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latent (MQA) sparse attention over a sliding window plus selected compressed rows, with an attention sink.

Plain torch. Static shapes, no host sync (CUDA-graph safe).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_pack_rows import csa2_unpack_rows
from .csa2_rows_io import csa2_gather_rows, csa2_row_addr

_TOKENS = 256  # query tokens attended at once (any mix of requests)


class Csa2SparseAttn(OpWrapper):
    """`o[t, h] = softmax([q[t,h] . K_t^T * scale, sink[h]]) @ K_t` (the sink column carries no value).

        `K_t` is the key set of token `t` at position `p`: the last `window` positions `p-window+1 .. p`
        (this step's rows from `kv`, earlier ones from the SWA pool) plus the compressed rows
        `topk_idx[t]` (main pool, `-1` = none). `q` `[n, h, d]` bf16, `kv` `[n, d]` bf16 (this step's
        rotated, quantize-dequantized window rows), `sink` `[h]` fp32, `cs` `[n]` the step's first
        position of the token's request, `rid` `[n]` its page-table row. Keys and values are the same
        vector. Scores and the softmax are fp32; the probabilities are rounded to bf16 for the value
        product (accumulated in fp32), like the checkpoint reference's
        kernel. `floor` `[n]` (optional) is the first window
    position a token may read: window rows before it are masked (the truncated window of a bounded decoder replay).
        A window row before the step on an unmapped SWA page (table entry < 0) is masked; `swa_old=False` skips
        reading the SWA pool (the caller knows no request has history). Returns bf16 `[n, h, d]`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="fresh prompt, no history, sliding window only (layers without compression)",
            spec=dict(hist=0, n=40, ratio=0, heads=8),
        ),
        Cell(
            why="decode with 300 tokens of history: the window is read from the SWA pool",
            spec=dict(hist=300, n=3, ratio=0, heads=8),
        ),
        Cell(
            why="ratio-2 layer: window plus 8 selected compressed rows, one of them -1",
            spec=dict(hist=300, n=5, ratio=2, heads=8),
        ),
        Cell(
            why="ratio-1 layer (compressed rows are one per position) over a 70-token chunk",
            spec=dict(hist=500, n=70, ratio=1, heads=8),
        ),
        Cell(
            why="history just past one window (130) with a single query row",
            spec=dict(hist=130, n=1, ratio=2, heads=8),
        ),
        Cell(
            why="dep2: all 64 heads, a chunk crossing the 256-token tile edge (300 rows)",
            spec=dict(hist=0, n=300, ratio=0, heads=64),
        ),
        Cell(
            why="dep2: 64 heads, compressed rows and history, 20-token chunk",
            spec=dict(hist=300, n=20, ratio=2, heads=64),
        ),
    )
    note = """
    The window is 128 and the SWA / main pages are 128 / (128 // ratio) rows (the layout of the
    engine's CSA2 pools); other geometries are rejected by `is_valid`, not certified. One request
    per cell; requests are independent rows in the op (`rid` only picks the page-table row), and
    the multi-request path is certified through the compress and indexer entries' multi-request
    cells and the target's boot, not here.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        cs: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_old: bool = True,
        floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n, h, d = q.shape
        dev = q.device
        out = torch.zeros_like(q)
        wi = torch.arange(window, device=dev)
        for a in range(0, n, _TOKENS):
            b = min(a + _TOKENS, n)
            p = pos[a:b].long().unsqueeze(-1)  # [m, 1]
            row = rid[a:b].long().unsqueeze(-1)  # [m, 1]
            qpos = p - (window - 1) + wi  # [m, window]
            from_step = qpos >= cs[a:b].long().unsqueeze(-1)
            step_row = (torch.arange(a, b, device=dev).unsqueeze(-1) - (p - qpos)).clamp(0, n - 1)
            keys = kv[step_row]
            valid = (
                qpos >= 0 if floor is None else qpos >= floor[a:b].long().unsqueeze(-1).clamp(min=0)
            )
            if swa_old:
                swa_addr = csa2_row_addr(swa_table, row.expand_as(qpos), qpos, 128)
                old = csa2_unpack_rows(csa2_gather_rows(swa_pool, swa_addr), "swa")
                keys = torch.where(from_step.unsqueeze(-1), keys, old)
                # a history row on an unmapped page is no key (this step's rows come from `kv`, whatever the table)
                valid = valid & (from_step | (swa_addr >= 0))
            if topk_idx is not None:
                idx = topk_idx[a:b].long()
                addr = csa2_row_addr(main_table, row.expand_as(idx), idx, 128 // ratio)
                comp = csa2_unpack_rows(csa2_gather_rows(main_pool, addr), "main")
                keys = torch.cat([keys, comp], dim=1)
                valid = torch.cat([valid, (idx >= 0) & (addr >= 0)], dim=1)
            kf = keys.float()
            s = torch.einsum("mhd,mkd->mhk", q[a:b].float(), kf) * scale
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = s.amax(dim=-1, keepdim=True).clamp(min=-1e30)
            e = torch.exp(s - mx)
            denom = e.sum(dim=-1) + torch.exp(sink.view(1, h) - mx.squeeze(-1))
            num = torch.einsum("mhk,mkd->mhd", e.bfloat16().float(), kf)
            out[a:b] = (num / denom.unsqueeze(-1)).to(out.dtype)
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        cs: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_old: bool = True,
        floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The checkpoint reference's `sparse_attn`, one token at a time, keys fetched from the
        pools by Python address arithmetic and decoded with `csa2_unpack_rows.reference`."""
        n, h, d = q.shape
        out = torch.zeros_like(q)
        swa_flat = swa_pool.reshape(-1, swa_pool.shape[-1])
        swa_t = swa_table.tolist()
        for t in range(n):
            p, r, c0 = int(pos[t]), int(rid[t]), int(cs[t])
            keys, valid = [], []
            lo = 0 if floor is None else max(0, int(floor[t]))
            for qp in range(p - window + 1, p + 1):
                if qp < lo:
                    keys.append(torch.zeros(d, device=q.device, dtype=kv.dtype))
                    valid.append(False)
                elif qp >= c0:
                    keys.append(kv[t - (p - qp)])
                    valid.append(True)
                else:
                    blk, off = divmod(qp, 128)
                    if blk >= len(swa_t[r]) or swa_t[r][blk] < 0:  # unmapped history page: no key
                        keys.append(torch.zeros(d, device=q.device, dtype=kv.dtype))
                        valid.append(False)
                        continue
                    row = swa_flat[swa_t[r][blk] * 128 + off]
                    keys.append(csa2_unpack_rows.reference(row, "swa", kv.dtype))
                    valid.append(True)
            if topk_idx is not None:
                rpp = 128 // ratio
                main_flat = main_pool.reshape(-1, main_pool.shape[-1])
                for j in topk_idx[t].tolist():
                    if j < 0:
                        keys.append(torch.zeros(d, device=q.device, dtype=kv.dtype))
                        valid.append(False)
                        continue
                    blk, off = divmod(j, rpp)
                    row = main_flat[main_table.tolist()[r][blk] * rpp + off]
                    keys.append(csa2_unpack_rows.reference(row, "main", kv.dtype))
                    valid.append(True)
            k = torch.stack(keys).double()
            ok = torch.tensor(valid, device=q.device)
            s = (q[t].double() @ k.T) * scale
            s = s.masked_fill(~ok, float("-inf"))
            mx = s.amax(-1, keepdim=True).clamp(
                min=-1e30
            )  # a fully masked row gives 0, as the op and the kernel
            e = torch.exp(s - mx)
            den = e.sum(-1) + torch.exp(sink.double() - mx.squeeze(-1))
            out[t] = ((e.to(torch.bfloat16).double() @ k) / den[:, None]).to(q.dtype)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        cs: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: Optional[torch.Tensor] = None,
        main_table: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        ratio: int = 0,
        swa_old: bool = True,
        floor: Optional[torch.Tensor] = None,
    ) -> None:
        assert floor is None or floor.shape == pos.shape, (
            "floor is the per-token first readable window position [n]"
        )
        assert swa_pool.dim() == 3 and swa_pool.shape[1] == 128 and swa_pool.shape[2] == 528, (
            "the SWA pool is [pages, 128, 528]; the row addressing is hard-wired to 128-row pages"
        )
        if topk_idx is None:
            assert main_pool is None, "main_pool without topk_idx is silently ignored"
        else:
            assert ratio in (1, 2) and main_pool is not None and main_table is not None
            assert main_pool.shape[1] == 128 // ratio and main_pool.shape[2] == 288, (
                "the main pool is [pages, 128 // ratio, 288]"
            )
            assert topk_idx.shape[0] == q.shape[0]
        assert 0 < window <= 128, "the SWA pool holds the last 128 positions only"
        assert kv.shape == (q.shape[0], q.shape[2]) and q.shape[2] == 512
        assert sink.shape == (q.shape[1],) and sink.dtype == torch.float32
        if not swa_old:
            assert bool((cs == 0).all()), (
                "swa_old=False skips the SWA pool, but a request with cs > 0 has history in it"
            )


csa2_sparse_attn = Csa2SparseAttn()
