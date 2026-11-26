# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`mla_rope`: the generation rows' MLA RoPE, fp8 query and paged latent append, on the
specialised CuTe DSL kernel at the widths it is built for, else the stock op.

The K-B (`absorbed_bmm.k_b_q_base`) decides, from the same `RopeWidths`, whether the call
takes the specialised kernel and whether its decode cell already wrote the e4m3 q-base; it
hands both over in the `MlaQuery` this op reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from . import cute_dsl_mla_rope_generation as _mla_rope_op
from .cute_dsl_mla_rope_generation import cute_dsl_mla_rope_generation
from .mla_rope_generation import mla_rope_generation

#: The stock op's residual dimension: none on the MLA path.
_KV_RESIDUAL_DIM = 0


@dataclass(frozen=True)
class RopeWidths:
    """The query rows per sequence the specialised rope kernel is built for, by caller:
    `trunk` (a trunk layer of a pure decode step: `mtp_widths`) and `mtp` (the MTP layer's
    decode draft step: `mtp_widths`, and on an engine drafting more than one token also its
    1-token draft steps, where the stock op needed the strided latent packed first). A
    mixed step's generation rows (`"stock"`) always take the stock op."""

    trunk: frozenset
    mtp: frozenset

    def takes(self, rope: str, gen_p: int) -> bool:
        if rope == "trunk":
            return gen_p in self.trunk
        if rope == "mtp":
            return gen_p in self.mtp
        return False


def widths(mtp_widths: tuple[int, ...], max_draft_len: int) -> RopeWidths:
    """The engine's `RopeWidths`."""
    mtp = set(mtp_widths) | ({1} if max_draft_len > 1 else set())
    return RopeWidths(trunk=frozenset(mtp_widths), mtp=frozenset(mtp))


class MlaQuery(NamedTuple):
    """The K-B's output as the rope reads it: `fused_q` `[gen, heads, lat_dim]` (its latent
    half written unless `from_kb`), `q_pe` the rope half of q, `quant_q` the e4m3 query
    (its q-base already written where `from_kb`), and whether the call takes the
    specialised kernel (`specialised`)."""

    fused_q: torch.Tensor
    q_pe: torch.Tensor
    quant_q: torch.Tensor
    specialised: bool
    from_kb: bool


class MlaRopeCells:
    """The specialised kernels by (block-table width, query rows per sequence), compiled on
    first use. Filled lazily because a separate draft pool's block-table width is known
    only at its first (eager) draft step; the trunk's are compiled ahead of any capture
    (`_derive_first_step`). The operand layout is the target's: `q_pe` the rope half of the
    `[T, heads, qk_dim]` q_b output, the latent the tail of the `[T, q_lora + lat_dim]` kv_a
    output. `q_base_from_kb` also builds the variants whose K-B wrote the q-base."""

    def __init__(
        self, heads: int, qk_dim: int, q_lora: int, lat_dim: int, q_base_from_kb: bool
    ) -> None:
        self._compiled: dict = {}
        self._heads, self._qk_dim = heads, qk_dim
        self._latent_stride = q_lora + lat_dim
        self._q_base_from_kb = q_base_from_kb

    def get(self, width: int, dev: int, p: int) -> _mla_rope_op.CompiledMlaRopeGeneration:
        handle = self._compiled.get((width, p))
        if handle is None:
            handle = _mla_rope_op.compile(
                max_blocks_per_seq=width,
                q_pe_head_stride=self._qk_dim,
                q_pe_token_stride=self._heads * self._qk_dim,
                latent_row_stride=self._latent_stride,
                device_index=dev,
                predicted_tokens_per_seq=p,
                q_base_from_kb=self._q_base_from_kb,
            )
            self._compiled[(width, p)] = handle
        return handle


@dataclass(frozen=True)
class MlaRopeHandle:
    """What `mla_rope` launches: the specialised kernels (`cells`), the rope table
    (`cos_sin`, None before the first forward derives it), the model's MLA dims and the KV
    cache's quant mode."""

    cells: MlaRopeCells
    cos_sin: torch.Tensor | None
    quant_mode: int
    heads: int
    q_lora: int
    kv_lora: int
    nope: int
    rope_dim: int
    count: Count


class MlaRope(OpWrapper):
    """RoPE of `query.q_pe` and the latent row's append to the paged KV cache at `layer`,
    writing the e4m3 query `query.quant_q`; returns `(quant_q, cu_q, bmm1, bmm2)` for the
    sparse MLA decode. `ctx` is the step's context request count (the generation rows
    follow them). On the specialised kernel where `query.specialised` (counted
    `mla_rope_draft1` at 1 row per sequence, `mla_rope_wide` above MTP1's 2), skipping the
    q-base where `query.from_kb`; else the stock op, which takes packed latent rows only."""

    def __call__(
        self,
        query: MlaQuery,
        latent: torch.Tensor,
        md,
        layer: int,
        gen_p: int,
        ctx: int,
        handle: MlaRopeHandle,
    ):
        fused_q, q_pe, quant_q = query.fused_q, query.q_pe, query.quant_q
        gen, heads, lat_dim = (int(s) for s in fused_q.shape)
        dev = fused_q.device
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        bmm1 = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2 = torch.empty([1], dtype=torch.float32, device=dev)
        h = handle
        if query.specialised:
            if gen_p == 1:
                h.count("mla_rope_draft1", True)
            elif gen_p != _mla_rope_op.MTP_WIDTH:
                h.count("mla_rope_wide", True)
            # The specialised kernel takes the stock op's earlier positional ABI.
            cute_dsl_mla_rope_generation(
                fused_q, q_pe, latent, h.cos_sin, cu_q, cu_kv, counter, bmm1, bmm2,
                quant_q, md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.prompt_lens_cpu_runtime,
                ctx, md.kv_cache_block_offsets, md.host_kv_cache_pool_pointers,
                md.host_kv_cache_pool_mapping, None, None, None, None, [None, None], gen_p, layer,
                heads, 1, lat_dim, md.tokens_per_block, md.max_seq_len, 1, h.quant_mode,
                1.0, h.q_lora, h.kv_lora, h.nope, h.rope_dim, h.kv_lora, True,
                compiled=h.cells.get(int(md.kv_cache_block_offsets.shape[-1]), dev.index, gen_p),
                q_base_from_kb=query.from_kb,
            )  # fmt: skip
        else:
            mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent.contiguous(),
                rotary_cos_sin=h.cos_sin,
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1,
                mla_bmm2_scale=bmm2,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=ctx,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=layer,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=md.tokens_per_block,
                attention_window_size=md.max_seq_len,
                beam_width=1,
                quant_mode=h.quant_mode,
                q_scaling=1.0,
                q_lora_rank=h.q_lora,
                kv_lora_rank=h.kv_lora,
                qk_nope_head_dim=h.nope,
                qk_rope_head_dim=h.rope_dim,
                v_head_dim=h.kv_lora,
                rope_append=True,
            )
        return quant_q, cu_q, bmm1, bmm2

    def reference(
        self,
        query: MlaQuery,
        latent: torch.Tensor,
        md,
        layer: int,
        gen_p: int,
        ctx: int,
        handle: MlaRopeHandle,
    ):
        """The e4m3 query the op writes, `[gen, heads, lat_dim]`: `[q-base | rope(q_pe)]` per
        row (`mla_rope_generation.reference`, GPT-J pairs), each generation row at its
        absolute position (the sequence's KV length minus its trailing rows). The q-base is
        `fused_q`'s latent half, or `quant_q`'s already-written bytes where `from_kb`. Both
        kernels write the same bytes; the latent append and the prefix arrays are not
        modelled here."""
        fused_q, q_pe = query.fused_q, query.q_pe
        gen = int(fused_q.shape[0])
        kv = handle.kv_lora
        rows = torch.arange(gen, device=fused_q.device)
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + gen // gen_p].to(torch.long)
        pos = kv_lens[rows // gen_p] - gen_p + rows % gen_p
        table = handle.cos_sin.view(-1, handle.rope_dim, 2)[pos, : handle.rope_dim // 2]
        cos, sin = table[..., 0][:, None, :], table[..., 1][:, None, :]
        base = (
            query.quant_q[..., :kv].float().to(fused_q.dtype)
            if query.from_kb
            else fused_q[..., :kv]
        )
        row = torch.cat([base, q_pe.to(fused_q.dtype)], dim=-1)
        return mla_rope_generation.reference(row, cos, sin, handle.rope_dim)


mla_rope = MlaRope()
