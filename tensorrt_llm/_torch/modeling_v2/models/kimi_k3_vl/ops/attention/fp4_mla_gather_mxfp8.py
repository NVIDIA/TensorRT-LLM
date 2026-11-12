# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather one cached-prefix partition of the FP4 MLA cache as MXFP8 kv_b input and FP8 K-pe."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_fp4_mla import fp4_mla_context_gather as _gather
from .k3_mla_q_fused import e2m1_values, swizzled_sf_offset, unpack_e2m1

_ROW_BYTES = 320
_SF_PER_ROW = 40


def _round_up_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Biased exponent of the smallest power of two >= ``x`` (positive FP32)."""
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    return (exponent + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(0, 254)


class Fp4MlaGatherMxfp8(OpWrapper):
    """``gather_mxfp8``: per partition token, the BF16-rounded latent of its FP4 cache row as
    MXFP8 into ``a`` (E4M3 ``[M, 512]``) / ``asc`` (UE8M0, K32 R128c4, ``pad_up(M, 128) * 16``
    bytes), and the BF16-rounded RoPE part plus its residual as unit-scale E4M3 into ``pe``
    ``[M, 64]``. Returns None."""

    ARCHS = frozenset({Arch.SM_107})
    # Spec: the cached-prefix chunk schedule of `ctx_split_k`'s tests: per-request cached
    # `prefix` and new tokens, the chunk size and buffer batch; each loop of the schedule is one
    # call. The cache is the FP4 MLA V2 cache with the backend's KV global scale.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp1 context: one request with a 1000-token prefix, 256-token chunks",
            spec=dict(prefix=[1000], new=[64], chunk_size=256, buffer_batch=1),
        ),
        Cell(
            why="tp1 context: three requests, prefixes across page boundaries, one partition each",
            spec=dict(prefix=[300, 129, 5], new=[17, 3, 40], chunk_size=4096, buffer_batch=3),
        ),
    )

    note = """
    Reads dense page-128 FP4 MLA V2 storage `kv` (320-byte rows: 512 latent + 64 RoPE + 64 RoPE
    residual E2M1 values) with its swizzled E4M3 scales `sf` (40 per row) and FP32
    `global_scale`; request r's tokens are rows `offsets[r] .. offsets[r] + cu[r + 1] - cu[r]` of
    its pages `pages[r * page_table_stride ..]`, written at rows `cu[r] ..` of the outputs. The
    bytes are those of the dense BF16 gather followed by `fp8_quantize_1x128_packed_ue8m0` and
    `static_quantize_e4m3_per_tensor` with a unit scale. Grid: ceil(max_chunk / warps) + 1 by
    the requests; prefix lengths and page strides are runtime values. A process cache holds one
    compiled kernel per (device, warps).
    """

    def __call__(
        self,
        a: torch.Tensor,
        asc: torch.Tensor,
        pe: torch.Tensor,
        kv: torch.Tensor,
        sf: torch.Tensor,
        pages: torch.Tensor,
        cu: torch.Tensor,
        offsets: torch.Tensor,
        global_scale: torch.Tensor,
        max_chunk: int,
        page_size: int,
        page_table_stride: int,
        num_contexts: int,
    ) -> None:
        _gather.gather_mxfp8(
            a,
            asc,
            pe,
            kv,
            sf,
            pages,
            cu,
            offsets,
            global_scale,
            max_chunk,
            page_size,
            page_table_stride=page_table_stride,
            num_contexts=num_contexts,
        )

    def reference(
        self,
        a: torch.Tensor,
        asc: torch.Tensor,
        pe: torch.Tensor,
        kv: torch.Tensor,
        sf: torch.Tensor,
        pages: torch.Tensor,
        cu: torch.Tensor,
        offsets: torch.Tensor,
        global_scale: torch.Tensor,
        max_chunk: int,
        page_size: int,
        page_table_stride: int,
        num_contexts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The post-state ``(a, asc, pe)``. Each cache value is ``e2m1 * e4m3 scale /
        global_scale`` rounded to BF16 (exact division where the kernel uses ``div.full``); a
        128-value block of the latent gets the UE8M0 exponent of the smallest power of two >=
        amax / 448 (amax floored at 1e-10), stored in its four K32 slots, and codes
        ``e4m3(value / 2^(e - 127))``; K-pe is ``e4m3(bf16(main + residual))``. Scale rows past
        ``M`` up to the next multiple of 128 become 0; rows of invalid pages are left alone."""
        out_a, out_asc, out_pe = a.clone(), asc.clone(), pe.clone()
        gs = global_scale.float().view(())
        num_pages = kv.shape[0]
        rows_view = kv.as_strided(
            (num_pages, page_size, _ROW_BYTES), (kv.stride(0), kv.stride(2), 1), kv.storage_offset()
        )
        sf_view = sf.view(torch.uint8).as_strided(
            (num_pages, page_size * _SF_PER_ROW), (sf.stride(0), 1), sf.storage_offset()
        )
        m = a.shape[0]
        padded = (m + 127) // 128 * 128
        if padded > m:
            pad_rows = torch.arange(m, padded, device=a.device)
            pad_offsets = swizzled_sf_offset(
                pad_rows[:, None], torch.arange(16, device=a.device), 16
            )
            out_asc.view(-1)[pad_offsets.flatten()] = 0
        cu_list, offset_list = cu.tolist(), offsets.tolist()
        table = pages.tolist()
        for r in range(num_contexts):
            begin, end = cu_list[r], cu_list[r + 1]
            for local in range(min(end - begin, max_chunk)):
                position = offset_list[r] + local
                logical = position // page_size
                if position < 0 or logical >= page_table_stride:
                    continue
                physical = table[r * page_table_stride + logical]
                if not 0 <= physical < num_pages:
                    continue
                row = position % page_size
                values = self._row_values(rows_view[physical, row], sf_view[physical], row, gs)
                out = begin + local
                latent = values[:512].view(4, 128)
                amax = latent.abs().amax(-1).clamp_min(1e-10)
                exponent = _round_up_ue8m0((amax.double() / 448.0).float())
                inv = torch.exp2(127.0 - exponent.double())
                codes = (latent.double() * inv[:, None]).clamp(-448, 448).float()
                out_a[out] = codes.to(torch.float8_e4m3fn).view(512)
                sf_cols = torch.arange(16, device=a.device)
                sf_offsets = swizzled_sf_offset(torch.tensor(out, device=a.device), sf_cols, 16)
                out_asc.view(-1)[sf_offsets] = exponent.repeat_interleave(4).to(torch.uint8)
                out_pe[out] = values[512:].clamp(-448, 448).to(torch.float8_e4m3fn)
        return out_a, out_asc, out_pe

    @staticmethod
    def _row_values(row_bytes: torch.Tensor, sf_page: torch.Tensor, row: int, gs) -> torch.Tensor:
        """The BF16-rounded 512 latent and 64 K-pe (main + residual) values of one cache row."""
        cols = torch.arange(_SF_PER_ROW, device=row_bytes.device)
        offsets = swizzled_sf_offset(torch.tensor(row, device=row_bytes.device), cols, _SF_PER_ROW)
        scales = sf_page[offsets].view(torch.float8_e4m3fn).float()
        codes = unpack_e2m1(row_bytes)
        values = e2m1_values(codes).view(_SF_PER_ROW, 16).double() * scales.double()[:, None]
        values = (values / gs.double()).float()
        latent = values[:32].flatten().to(torch.bfloat16).float()
        rope = (values[32:36].flatten() + values[36:40].flatten()).to(torch.bfloat16).float()
        return torch.cat((latent, rope))


fp4_mla_gather_mxfp8 = Fp4MlaGatherMxfp8()
