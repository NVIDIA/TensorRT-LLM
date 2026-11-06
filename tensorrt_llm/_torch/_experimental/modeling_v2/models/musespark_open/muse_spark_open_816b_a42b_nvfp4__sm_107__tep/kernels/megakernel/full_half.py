# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention half of a full-attention (NoPE) decoder layer on the megakernel: from the normed BF16 attention
input ``x [8, 8192]`` to the o_proj partial in this rank's hop1 slot, as two launches of one library
(``schedule_full``):

* launch Q: the QKV projection (``qkv.QkvStage``, the checkpoint row order), the raw projection BF16 ``[8, 2304]``
  into a plain buffer;
* launch F: the full-attention stages (``stages/fullAttention.cuh``: the step's K/V append, split-KV attention over
  the FP8 pool, the merge and the output norm-gate) and the o_proj GEMV units of the attention half
  (``stages/attnHalf.cuh`` ``AhOproj``), the gated input handed over in the arena.

The same library (a ``*_s4`` mode) also runs the sliding-window attention half (``attn_half.py``), so a slice whose
every attention half is replaced runs no stock attention kernel.

Arena copies: launch F is the only launch of ``ws_f`` that reads the arena; launch Q (``ws_q``) reads none. Each
workspace rotates over its own launches, so the launches of one forward must be a multiple of its copies (two full
layers per slice: ``copies=2``, whose clear of copy ``n + 1`` by launch ``n`` is safe only because two launches of
``ws_f`` never run back to back).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from . import full_attention, library, qkv, schedule, schedule_full

#: ``kPIFlags`` bit 0 (``kIFlagFaOutArena``).
IFLAG_FA_OUT_ARENA = 1


def _f32_bits(x: float) -> int:
    return int(torch.tensor(float(x), dtype=torch.float32).view(torch.int32))


class FullHalf:
    """The full-attention layer's attention half (two launches).

    Args:
        mode: a K-slice compile mode (``lean_s4`` timed, ``check_s4`` checks and stamps).
        units: attention units (one per CTA, from CTA 0).
        copies: arena copies of each workspace (the full layers of one forward must be a multiple of it).
    """

    def __init__(
        self,
        mode: str = "lean_s4",
        units: Optional[int] = None,
        min_pages: int = 4,
        copies: int = 2,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        assert self.lib.qkv_slices == 4, "a K-slice (_s4) build, as the sliding-window half"
        self.ws_q = library.Workspace(self.lib, copies)
        self.ws_f = library.Workspace(self.lib, copies)
        self.qkv = qkv.QkvStage(mode, workspace=self.ws_q, timeout_ms=timeout_ms)
        self.units = int(units or self.lib.ctas)
        self.min_pages = int(min_pages)
        self.region_bytes = full_attention._abi_int(self.lib, "msoMkFaRegionBytes")
        self.ring = full_attention._abi_int(self.lib, "msoMkFaRing")
        # Ring pages each FaAttn unit requests in its prologue (the rest when its Q is ready); default the whole ring
        self.entry_pages = self.ring
        self.tables = schedule_full.full_attention_tables(
            self.lib.ctas, self.units, self.region_bytes
        )
        self.mk = library.Megakernel(self.lib, self.ws_f, self.tables, timeout_ms=timeout_ms)
        dev = self.ws_f.arena.device
        n = full_attention.ROWS * 16 * self.units
        # Single-reader partial slots: kEmpty between launches.
        self.part_o = torch.full(
            (n * full_attention.HEAD_DIM // 2,), -0.0, dtype=torch.float32, device=dev
        )
        self.part_l = torch.full((n,), -0.0, dtype=torch.float32, device=dev)
        self.qkv_buf = torch.zeros(8, schedule.QKV_ROWS, dtype=torch.bfloat16, device=dev)

    def make_layer(
        self,
        w_qkv: torch.Tensor,
        w_o: torch.Tensor,
        pool: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        slot: torch.Tensor,
        *,
        kv_scale: float,
        q_scale: float = full_attention.Q_SCALE,
        eps: float = full_attention.EPS,
        check: bool = False,
        stamps: bool = False,
        prefetch_tiles: int = 0,
    ) -> Dict[str, object]:
        """One layer's parameter blocks (eager): ``dict(q=launch Q's blocks, f=launch F's blocks)``.

        Args:
            w_qkv: BF16 ``[2304, 8192]`` (checkpoint order); w_o: BF16 ``[8192, 1024]``.
            pool: the layer's pool view ``[blocks, 2, 32, 64]`` (1-byte elements); block_table: int32 ``[2, >= pages]``.
            kv_lens: int32, element 0 = the context including the step (graph input, bound by address).
            slot: BF16 ``[8, 8192]``, this rank's hop1 slot (multicast view) or a plain buffer.
        """
        assert (
            slot.dtype == torch.bfloat16
            and tuple(slot.shape) == (8, schedule.HIDDEN)
            and slot.is_contiguous()
        )
        q_blocks = self.q_layer(w_qkv, check=check, stamps=stamps)
        fields, flags, keep = self.f_fields(
            w_o,
            pool,
            block_table,
            kv_lens,
            kv_scale=kv_scale,
            q_scale=q_scale,
            eps=eps,
            check=check,
            prefetch_tiles=prefetch_tiles,
        )
        fields["AhSlot"] = slot.data_ptr()
        f_blocks = self.mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep + (slot,))
        return dict(q=q_blocks, f=f_blocks)

    def q_layer(self, w_qkv: torch.Tensor, check: bool = False, stamps: bool = False):
        """Launch Q's parameter blocks."""
        return self.qkv.make_layer(w_qkv.contiguous(), check=check, stamps=stamps, one_piece=True)

    def f_fields(
        self,
        w_o: torch.Tensor,
        pool: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        *,
        kv_scale: float,
        q_scale: float = full_attention.Q_SCALE,
        eps: float = full_attention.EPS,
        check: bool = False,
        prefetch_tiles: int = 0,
    ):
        """Launch F's parameter fields, flags and kept tensors without the o_proj destination (``AhSlot``, or the
        MoE hop workspace with ``kIFlagOprojToMh``): ``(fields, flags, keep)``."""
        assert w_o.dtype == torch.bfloat16 and tuple(w_o.shape) == (
            schedule.HIDDEN,
            full_attention.OUT_WIDTH,
        )
        pool8 = pool.view(torch.float8_e4m3fn)
        tmap = full_attention.tensor_map(self.lib, pool8)
        scale = torch.tensor([float(kv_scale)], dtype=torch.float32, device=pool.device)
        packed_o = qkv.pack_bf16(w_o.contiguous())
        fields: Dict[str, int] = dict(
            FaTmap=tmap.data_ptr(),
            FaPool=pool8.data_ptr(),
            FaTable=block_table.data_ptr(),
            FaTableStride=block_table.stride(0),
            FaKvLens=kv_lens.data_ptr(),
            FaKvScale=scale.data_ptr(),
            FaPartO=self.part_o.data_ptr(),
            FaPartL=self.part_l.data_ptr(),
            FaUnits=self.units,
            FaMinPages=self.min_pages,
            FaEntryPages=int(self.entry_pages),  # ring pages requested in the prologue
            FaQScale=_f32_bits(q_scale),
            FaEps=_f32_bits(eps),
            FaOut=0,
            FaOutF32=0,
            FaPrefetch=int(prefetch_tiles),
            AhWOproj=packed_o.data_ptr(),
            AhOReplicas=1,
            AhFlags=0,
            IFlags=IFLAG_FA_OUT_ARENA,
        )
        flags = (
            library.FLAG_EVICT_FIRST | library.FLAG_ONE_PIECE | (library.FLAG_CHECK if check else 0)
        )
        keep = (tmap, pool8, block_table, kv_lens, scale, packed_o)
        return fields, flags, keep

    def __call__(self, layer: Dict[str, object], x: torch.Tensor, pdl: bool = True):
        """Launch both halves on the current stream; returns launch F's block."""
        assert (
            x.dtype == torch.bfloat16
            and tuple(x.shape) == (8, schedule.HIDDEN)
            and x.is_contiguous()
        )
        self.qkv(layer["q"], x=x, out=self.qkv_buf, pdl=pdl)
        return self.mk.launch(layer["f"], library.KernelIo(qkv=self.qkv_buf), pdl=pdl)
