# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent TP16 decoder-layer kernel for one request and eight verify tokens.

The weights and multicast buffers belong to this target instance. Each launch clears the next
of three workspace copies, so every eligible forward replaces the same complete MoE-layer sequence.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.distributed.ops import (
    _get_mnnvl_workspace_comm,
    _make_mnnvl_mcast_buffer,
    _mnnvl_workspace_all_succeeded,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from . import native_ops

if TYPE_CHECKING:
    from .modeling import DecodeTarget

# Parameter block fields, in the order of glm52::Param.
_FIELDS = [
    "Magic",
    "KvBase",
    "StrideRows",
    "LayerRows",
    "MaxBlocks",
    "Rope",
    "WFa",
    "WQb",
    "WLg",
    "WO",
    "WR",
    "WGu",
    "WD",
    "GammaKv",
    "GammaPost",
    "GammaNext",
    "RouterBias",
    "W13",
    "S13",
    "W2",
    "S2",
    "Alpha1",
    "Alpha2",
    "Arena",
    "PartO",
    "PartStats",
    "Fc2out",
    "H1Mc",
    "H1Uc",
    "H2Mc",
    "H2Uc",
    "HopBytes",
    "Rank",
    "Counters",
    "State",
    "Err",
    "Roles",
    "RoleSmem",
    "RoleIssue",
    "MaxRoles",
    "TimeoutNs",
    "Stamps",
    "Flags",
    "Copy",
    "NumCopies",
    "Leader",
    "IdxWq",
    "IdxKnW",
    "IdxKnB",
    "IdxRope",
    "IdxPool",
    "IdxBlockStride",
    "IdxTmap",
    "IdxBlocks",
    "IdxCand",
    "TopkOut",
    "IdxDebug",
    "IdxDebugLd",
    "IdxDebug2",
    "TkUc",
    "TkMc",
]
_P = {name: i for i, name in enumerate(_FIELDS)}

# Stage codes (glm52::Stage).
_ST = dict(
    FusedA=1,
    Qb=2,
    KvAppend=3,
    Attn=4,
    Merge=5,
    Oproj=7,
    Clear=8,
    H1=9,
    RouterGemv=10,
    SharedGu=12,
    Moe=13,
    Owner=14,
    H2=15,
    OwnerShared=16,
    IdxK=17,
    IdxQ=18,
    IdxSel=19,
    OprojLg=20,
)
_CTAS = 212
_MAX_ROLES = 16
_SMEM_MAIN = 200704
_SMEM_FIXED = 16384 + 3072 + 3072  # reduction, residual and small areas after the unit regions
# The attention's working set sits at the top of shared memory (into the reduction area, unused by the attention), so
# the o_proj weights of the same CTA can land at [0, 128 KiB) while it runs.
_ATTN_OFF = 131072
_HOP_BYTES = 8 * 16 * 6144 * 2
_TK_BYTES = 8 * 2048 * 4  # the leader's top-k exchange per launch copy (glm52::kTkBytes)
# The router logits by linearity (glm52::kRouterLg, GLM52_ROUTER_LG; the library reports its mode): the ranks' partial
# logits per launch copy (glm52::kLgBytes: FP32 [16 ranks][16 units][8 tokens][16 experts]), in the multicast workspace
# after the top-k exchange's copies; the logit units' weights (K = 2048 O columns + the rank's 384 residual columns).
_LG_BYTES = 16 * 16 * 8 * 16 * 4
_LG_UNITS = 16
_LG_CHUNKS = 38
_LG_OPROJ_FIRST = (
    148  # glm52::kLgOprojFirst: logit unit u rides on o_proj unit 148 + u (CTAs 148-163)
)
_ARENA_BYTES = (
    2174720 + 8 * 6144 * 2 + 3 * 8 * 4 * 512 * 2 + 8 * 6144 * 2 + 8 * 4 * 4
)  # glm52::kArenaBytes (x, O's
# three copies, h', x' sum-of-squares partials)

_FLAGS = 35520
_FLAG_EARLY_X = 32768
_FLAG_NEXT_CHAINED = 32

_HOP_SLOT_CLEAR_UNITS = 2 * _HOP_BYTES // 16384
_HOP_CLEAR_UNITS = (2 * _HOP_BYTES + _TK_BYTES) // 16384
_CLEAR_UNITS = _HOP_CLEAR_UNITS + (_ARENA_BYTES + 16383) // 16384
# Shared-memory bytes of a unit's region (weights, or the working set of the dynamic stages).
_UNIT_BYTES = {
    _ST["FusedA"]: 96 * 2048,
    _ST["Qb"]: 3 * 32 * 2048,
    _ST["Attn"]: 128 * 592 + 8 * 592 + 8 * 144 + 2 * 64 * 4 + 8 * 16 * 4,
    _ST["Oproj"]: 2 * 32 * 2048,
    _ST["RouterGemv"]: 24 * 2048,
    _ST["SharedGu"]: 2 * 24 * 2048,
    _ST["Moe"]: 3 * 55296 + 32768 + 1024,
    _ST["IdxQ"]: 2 * 32 * 2048,
    _ST["IdxSel"]: 200704,  # glm52::kSelTotal (the row shard's logits past the K rounds' region)
    _ST["OprojLg"]: 3
    * 32
    * 2048,  # o_proj's two tiles, then the logit tile's O part (its residual part: global)
}
# Stages whose shared memory is written by their own code (their region must be free when they start).
_DYNAMIC = {_ST["Attn"], _ST["Moe"], _ST["IdxSel"]}


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """[M, K] BF16 -> [M/16, K/64, 2048] bytes: per (tile, chunk) pieces p = (row half, element half) of 32 threads x
    16 B; thread (g, c) holds row g's (pieces 0-1) and row g + 8's (2-3) elements [64 q + 16 c, +16)."""
    m, k = w.shape
    assert w.dtype == torch.bfloat16 and m % 16 == 0 and k % 64 == 0
    v = w.contiguous().view(m // 16, 2, 8, k // 64, 4, 2, 8)  # tile, rh, g, q, c, eh, e8
    v = v.permute(0, 3, 1, 5, 2, 4, 6)  # tile, q, rh, eh, g, c, e8
    return v.contiguous().view(torch.uint8).reshape(m // 16, k // 64, 2048)


def swizzle_64b(t: torch.Tensor) -> torch.Tensor:
    """[..., R, 64] bytes -> the SWIZZLE_64B layout of 8-row x 64 B atoms: 16-byte chunk c of row r goes to chunk
    c ^ ((r >> 1) & 3) of its row."""
    *lead, rows, width = t.shape
    assert width == 64
    v = t.reshape(-1, rows, 4, 16)
    r = torch.arange(rows, device=t.device).view(rows, 1)
    dst = torch.arange(4, device=t.device).view(1, 4) ^ ((r >> 1) & 3)
    out = torch.empty_like(v)
    out.scatter_(2, dst.view(1, rows, 4, 1).expand(v.shape[0], rows, 4, 16), v)
    return out.view(*lead, rows, 64)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def build_roles(leader=False, smem_main=_SMEM_MAIN, qb_late=False, slice_merge=False, lg=0):
    """lg: GLM52_ROUTER_LG of the build (1: the logit units too; 2: and no router GEMV units)."""
    lists = [[] for _ in range(_CTAS)]

    def add(cta, stage, unit):
        lists[cta].append((_ST[stage], unit))

    if leader:
        # fused-A tiles 0-173 (164-173: the indexer K and head-weight rows); q_b' on CTAs 174-211 and after fused-A
        # on 164-173; the indexer Q (128 quarter-head units) after fused-A on CTAs 0-127; then every CTA's IdxSel
        # (key pages c, c + 212, ...; all CTAs meet in its top-k).
        for c in range(174):
            add(c, "FusedA", c)
        if not qb_late:
            for u in range(48):
                add(174 + u if u < 38 else 164 + (u - 38), "Qb", u)
        add(163, "KvAppend", 0)
        add(162, "IdxK", 0)
        for u in range(128):
            add(u, "IdxQ", u)
        for c in range(_CTAS):
            add(c, "IdxSel", c)
        if qb_late:
            # After IdxSel on the non-selector CTAs (the selectors 204-211 run the top-k's last steps).
            for u in range(48):
                add(164 + u if u < 40 else 128 + (u - 40), "Qb", u)
    else:
        for c in range(_CTAS):
            if c < 164:
                add(c, "FusedA", c)
            else:
                add(c, "Qb", c - 164)
        add(163, "KvAppend", 0)
    for u in range(128):
        add(u, "Attn", u)
    if not slice_merge:
        for u in range(32):
            add(164 + u, "Merge", u)
    for u in range(192):
        # The router logits by linearity: the o_proj units on CTAs 148-163 (no attention, shared-expert or q_b' unit)
        # also run logit unit u - 148 on the same O (its weights land after fused-A with o_proj's), then the CTA's MoE
        # warm-up and the ranks' sum.
        lgu = lg and _LG_OPROJ_FIRST <= u < _LG_OPROJ_FIRST + _LG_UNITS
        add(u, "OprojLg" if lgu else "Oproj", u)
    for u in range(_HOP_SLOT_CLEAR_UNITS, _CLEAR_UNITS):
        add(u if u < _HOP_CLEAR_UNITS else u - _HOP_CLEAR_UNITS, "Clear", u)
    for u in range(32):
        add(180 + u, "H1", u)
    if lg < 2:
        for u in range(64):
            add(u, "RouterGemv", u)
    for u in range(32):
        add(65 + u, "SharedGu", u)
    # On CTAs [100, 148): the kernel's MoE work cut gives these CTAs a smaller share (glm52::kOsFirst, kOsCtas).
    for j in range(48):
        add(100 + j, "OwnerShared", j)
    # MoE unit = 1 + the CTA's shared-expert unit (run inside the MoE stage with kFlagOsInMoe), or 0.
    for c in range(_CTAS):
        add(c, "Moe", c - 99 if 100 <= c < 148 else 0)
    for j in range(48):
        add(100 + j, "Owner", j)
    for u in range(32):
        add(180 + u, "H2", u)

    roles = torch.zeros(_CTAS, _MAX_ROLES, dtype=torch.int32)
    smem = torch.zeros(_CTAS, _MAX_ROLES, dtype=torch.int32)
    issue = torch.full((_CTAS, _MAX_ROLES), -2, dtype=torch.int32)
    for c, units in enumerate(lists):
        assert len(units) <= _MAX_ROLES, (c, units)
        placed = []  # (offset, size, index)
        for i, (stage, unit) in enumerate(units):
            roles[c, i] = (stage << 16) | unit
            size = _UNIT_BYTES.get(stage, 0)
            if size == 0:
                continue
            if stage in _DYNAMIC:
                off = _ATTN_OFF if stage == _ST["Attn"] else 0
                blocking = max(
                    [j for (o, s, j) in placed if o < off + size and off < o + s], default=-1
                )
            else:
                cands = sorted({0, *[((o + s + 1023) // 1024) * 1024 for (o, s, j) in placed]})
                best = None
                for off_c in cands:
                    if off_c + size > smem_main:
                        continue
                    blk = max(
                        [j for (o, s, j) in placed if o < off_c + size and off_c < o + s],
                        default=-1,
                    )
                    if best is None or blk < best[0]:
                        best = (blk, off_c)
                assert best is not None, (c, stage)
                blocking, off = best
                issue[c, i] = blocking
            smem[c, i] = off
            placed.append((off, size, i))
    return roles, smem, issue


class _Shared:
    """Buffers reused by a complete sequence of decoder-layer launches."""

    def __init__(self, mapping: Mapping) -> None:
        (
            self.magic,
            num_params,
            arena_bytes,
            clear_units,
            smem_bytes,
            num_counters,
            cand_cap,
            self.max_keys,
            self.router_lg,
            lg_bytes,
        ) = native_ops.glm_megakernel_info()
        assert num_params == len(_FIELDS), "megakernel parameter layout mismatch"
        assert arena_bytes == _ARENA_BYTES and clear_units == _CLEAR_UNITS
        assert smem_bytes == _SMEM_MAIN + _SMEM_FIXED and lg_bytes == _LG_BYTES
        self.copies = 3
        self.tables = {
            leader: build_roles(leader, qb_late=True, lg=self.router_lg) for leader in (False, True)
        }
        device = torch.device("cuda", torch.cuda.current_device())

        def empty(num_bytes: int) -> torch.Tensor:
            return torch.full((num_bytes // 4,), -0.0, dtype=torch.float32, device=device)

        self.arena = empty(self.copies * _ARENA_BYTES)
        self.part_o = empty(8 * 16 * 4 * 512 * 4)
        self.part_stats = empty(8 * 16 * 4 * 2 * 4)
        self.fc2out = empty(64 * 8 * 6144 * 2)
        self.counters = torch.zeros(self.copies * num_counters, dtype=torch.int32, device=device)
        self.idx_cand = torch.zeros(
            self.copies * 8 * cand_cap * 2, dtype=torch.int32, device=device
        )
        self.state = torch.zeros(4, dtype=torch.int32, device=device)
        self.err = torch.zeros(16 + 5 * _CTAS, dtype=torch.int32, pin_memory=prefer_pinned())
        self.leader_topk: dict[int, torch.Tensor] = {}
        self.comm = _get_mnnvl_workspace_comm(mapping)
        error = None
        try:
            total = self.copies * (2 * _HOP_BYTES + _TK_BYTES + _LG_BYTES)
            self.handle = _make_mnnvl_mcast_buffer(
                self.comm, total, mapping, use_fabric_handle=mapping.is_multi_node()
            )
            self.uc = self.handle.get_uc_buffer(mapping.tp_rank, (total // 4,), torch.float32, 0)
            self.mc = self.handle.get_mc_buffer((total // 4,), torch.float32, 0)
            self.uc.fill_(-0.0)
            torch.cuda.synchronize()
        except Exception as exc:
            # All ranks must reach the agreement even if one allocation failed.
            error = exc
        if not _mnnvl_workspace_all_succeeded(self.comm, error is None):
            raise RuntimeError(
                "GLM megakernel workspace construction failed on at least one rank"
            ) from error
        self.mutable = [
            self.arena,
            self.part_o,
            self.part_stats,
            self.fc2out,
            self.counters,
            self.state,
            self.idx_cand,
            self.uc,
            self.mc,
            self.err,
        ]


class _Layer:
    """Repacked weights and capture-stable native parameters for one MoE layer."""

    def __init__(
        self,
        decode: DecodeTarget,
        layer_idx: int,
        shared: _Shared,
        copy: int,
        chained: bool,
        next_chained: bool,
    ) -> None:
        self.layer = layer = decode.core.layers[layer_idx]
        self.layer_idx = layer_idx
        self.mapping = decode.mapping
        self.shared = shared
        self.copy = copy
        self.flags = _FLAGS
        if next_chained and self.flags & _FLAG_EARLY_X:
            self.flags |= _FLAG_NEXT_CHAINED
        if not chained:
            self.flags &= ~_FLAG_EARLY_X
        self.pdl = os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"
        self.params_host = None
        self.params = None
        self.step_params = {}
        attn = layer.self_attn
        self.indexer = attn.mqa.indexer
        self.leader = self.indexer is not None
        self.leader_idx = max(
            (
                j
                for j in range(layer_idx)
                if decode.core.layers[j].self_attn.mqa.indexer is not None
            ),
            default=None,
        )
        device = attn.o_proj.weight.device
        moe = layer.mlp
        backend = getattr(moe.experts, "backend", moe.experts)
        w_fa = attn.kv_a_proj_with_mqa.weight
        w_qb = decode._q_b_absorbed[layer_idx]
        w_vb = attn.v_b_proj
        w_o = attn.o_proj.weight
        assert tuple(w_fa.shape) == (2624, 6144)
        assert tuple(w_qb.shape) == (2304, 2048)
        assert tuple(w_vb.shape) == (4, 256, 512)
        assert tuple(w_o.shape) == (6144, 1024)
        w_ov = torch.cat(
            [w_o[:, 256 * h : 256 * h + 256].float() @ w_vb[h].float() for h in range(4)], dim=1
        )
        if self.leader:
            idx = self.indexer
            w_fa = decode._fused_a_indexer_weight[layer_idx]
            self.w_idx = pack_bf16(decode._wq_b_folded[layer_idx])
            self.idx_kn_w = idx.k_norm.weight.float().contiguous()
            self.idx_kn_b = idx.k_norm.bias.float().contiguous()
            assert abs(float(idx.k_norm.variance_epsilon) - 1e-6) < 1e-12
            assert not idx.rotary_emb.is_neox
            cs = idx.rotary_emb.rotary_cos_sin
            self.idx_rope = cs.float().contiguous().view(cs.shape[0], -1)
            assert self.idx_rope.shape[1] == 64
            assert idx.n_heads == 32 and idx.head_dim == 128 and idx.index_topk == 2048
            assert not idx.use_fp4
            self.topk_out = torch.full((8, 2048), -1, dtype=torch.int32, device=device)
            shared.leader_topk[layer_idx] = self.topk_out
        self.w_fa = pack_bf16(w_fa)
        self.w_qb = pack_bf16(w_qb)
        self.w_o = pack_bf16(w_ov.to(torch.bfloat16))
        del w_ov
        self.gamma_kv = attn.kv_a_layernorm.weight.contiguous()
        self.rope = attn.mqa.rotary_cos_sin.contiguous().view(-1)
        self.w_r = pack_bf16(moe.gate.weight.to(torch.bfloat16))
        self.router_bias = moe.gate.e_score_correction_bias.to(torch.bfloat16).contiguous()
        gu = moe.shared_experts.gate_up_proj.weight
        dn = moe.shared_experts.down_proj.weight
        assert tuple(gu.shape) == (256, 6144) and tuple(dn.shape) == (6144, 128)
        self.w_gu = pack_bf16(gu)
        self.w_d = pack_bf16(dn)
        w13 = _u8(backend.w3_w1_weight).reshape(256, 2, 128, 48, 64).permute(0, 1, 3, 2, 4)
        w13 = swizzle_64b(w13.contiguous()).reshape(256, 2, 8, 6 * 8192)
        s13 = _u8(backend.w3_w1_weight_scale).reshape(256, 2, 8, 12 * 512)
        self.w13 = torch.cat([w13, s13], dim=3).contiguous()
        self.s13 = torch.zeros(1, dtype=torch.uint8, device=device)
        w2 = swizzle_64b(_u8(backend.w2_weight).reshape(256, 48, 128, 64)).reshape(256, 8, 6 * 8192)
        s2 = _u8(backend.w2_weight_scale).reshape(256, 8, 6 * 2 * 512)
        self.w2 = torch.cat([w2, s2], dim=2).contiguous()
        self.s2 = torch.zeros(1, dtype=torch.uint8, device=device)
        del w13, s13, w2, s2
        assert getattr(backend, "fc31_act_scale", None) is None
        scales = []
        for name in ("fc31_input_scale", "fc2_input_scale"):
            value = getattr(backend, name, None)
            value = (
                torch.ones(1, device=device)
                if value is None
                else value.detach().float().reshape(-1).to(device)
            )
            assert value.numel() == 1 and bool(torch.isfinite(value).all()) and float(value) > 0
            scales.append(value)
        self.alpha1 = torch.cat([backend.fc31_alpha.float().reshape(256), scales[0]]).contiguous()
        self.alpha2 = torch.cat([backend.fc2_alpha.float().reshape(256), scales[1]]).contiguous()
        self.gamma_post = layer.post_attention_layernorm.weight.contiguous()
        self.gamma_next = layer.next_layer_layernorm.weight.contiguous()
        self.w_lg = None
        if shared.router_lg:
            rank = int(self.mapping.tp_rank)
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                wg = moe.gate.weight.float() * self.gamma_post.float().view(1, -1)
                wov = torch.cat(
                    [w_o[:, 256 * h : 256 * h + 256].float() @ w_vb[h].float() for h in range(4)],
                    dim=1,
                )
                product = torch.cat([wg @ wov, wg[:, 384 * rank : 384 * rank + 384]], dim=1)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            assert tuple(product.shape) == (256, 64 * _LG_CHUNKS)
            self.w_lg = pack_bf16(product.to(torch.bfloat16))

    def bind(self, manager) -> None:
        """Rebuild raw cache addresses after profiling replaces the KV manager, outside capture."""
        shared = self.shared
        device = self.w_fa.device
        kv_pool = manager.get_unique_primary_pool()
        if self.leader:
            kbuf = manager.get_indexer_k_cache_buffers(self.layer_idx)
            assert kbuf.dtype == torch.uint8 and tuple(kbuf.shape[1:]) == (64, 1, 132)
            assert kbuf.stride()[1:] == (132, 132, 1) and kbuf.stride(0) % 128 == 0
            self.idx_block_stride = int(kbuf.stride(0))
            rows = ((kbuf.shape[0] - 1) * self.idx_block_stride + 8448) // 128
            self.idx_tmap = native_ops.glm_megakernel_indexer_tmap(kbuf, rows).to(device)
        local_layer = manager.layer_offsets[self.layer_idx]
        scale, layer_offset = manager.get_primary_pool_page_index_params(local_layer)
        assert manager.tokens_per_block == 64
        self.stride_rows = int(scale) * 64
        self.layer_rows = int(layer_offset) * 64
        self.kv_base = kv_pool.data_ptr()
        self.layer.self_attn.mqa._ensure_rope_table_size(manager.max_seq_len)
        self.rope = self.layer.self_attn.mqa.rotary_cos_sin.contiguous().view(-1)
        roles, role_smem, role_issue = shared.tables[self.leader]
        tables = torch.cat([roles.reshape(-1), role_smem.reshape(-1), role_issue.reshape(-1)])
        wt_offset = len(_FIELDS) + tables.numel() // 2
        p = torch.zeros(
            wt_offset + _CTAS * _MAX_ROLES * 3, dtype=torch.int64, pin_memory=prefer_pinned()
        )
        p[len(_FIELDS) : wt_offset] = tables.view(torch.int64)

        def put(name: str, value: int | torch.Tensor) -> None:
            p[_P[name]] = value.data_ptr() if isinstance(value, torch.Tensor) else value

        for name, value in (
            ("Magic", shared.magic),
            ("KvBase", self.kv_base),
            ("StrideRows", self.stride_rows),
            ("LayerRows", self.layer_rows),
            ("Rope", self.rope),
            ("WFa", self.w_fa),
            ("WQb", self.w_qb),
            ("WO", self.w_o),
            ("WR", self.w_r),
            ("WGu", self.w_gu),
            ("WD", self.w_d),
            ("GammaKv", self.gamma_kv),
            ("GammaPost", self.gamma_post),
            ("GammaNext", self.gamma_next),
            ("RouterBias", self.router_bias),
            ("W13", self.w13),
            ("S13", self.s13),
            ("W2", self.w2),
            ("S2", self.s2),
            ("Alpha1", self.alpha1),
            ("Alpha2", self.alpha2),
            ("Arena", shared.arena),
            ("PartO", shared.part_o),
            ("PartStats", shared.part_stats),
            ("Fc2out", shared.fc2out),
            ("Counters", shared.counters),
            ("State", shared.state),
            ("Err", shared.err),
            ("IdxCand", shared.idx_cand),
            ("Leader", int(self.leader)),
            ("NumCopies", shared.copies),
            ("H1Uc", shared.uc.data_ptr()),
            ("H1Mc", shared.mc.data_ptr()),
            ("H2Uc", shared.uc.data_ptr() + shared.copies * _HOP_BYTES),
            ("H2Mc", shared.mc.data_ptr() + shared.copies * _HOP_BYTES),
            ("TkUc", shared.uc.data_ptr() + 2 * shared.copies * _HOP_BYTES),
            ("TkMc", shared.mc.data_ptr() + 2 * shared.copies * _HOP_BYTES),
            ("Copy", self.copy),
            ("HopBytes", _HOP_BYTES),
            ("Rank", self.mapping.tp_rank),
            ("MaxRoles", _MAX_ROLES),
            ("TimeoutNs", 20_000_000_000),
            ("Flags", self.flags),
        ):
            put(name, value)
        if self.w_lg is not None:
            put("WLg", self.w_lg)
        if self.leader:
            for name, value in (
                ("IdxWq", self.w_idx),
                ("IdxKnW", self.idx_kn_w),
                ("IdxKnB", self.idx_kn_b),
                ("IdxRope", self.idx_rope),
                ("IdxPool", kbuf),
                ("IdxTmap", self.idx_tmap),
                ("TopkOut", self.topk_out),
                ("IdxBlockStride", self.idx_block_stride),
            ):
                put(name, value)
        self.params = p.to(device)
        native_ops.glm_megakernel_weight_table(self.params, wt_offset)
        p[wt_offset:] = self.params[wt_offset:].cpu()
        self.params_host = p
        self.step_params = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        metadata,
        fallback_topk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.leader:
            topk = self.topk_out
            indexer_blocks = metadata.indexer_k_cache_block_offsets[0]
        else:
            topk = self.shared.leader_topk.get(self.leader_idx, fallback_topk)
            indexer_blocks = None
        block_table = metadata.block_table[0]
        kv_lens = metadata.kv_lens_cuda_runtime
        max_blocks = metadata.block_table.shape[1]
        key = (None if indexer_blocks is None else indexer_blocks.data_ptr(), max_blocks)
        entry = self.step_params.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the GLM megakernel metadata before CUDA graph capture")
            if indexer_blocks is not None:
                self.params_host[_P["IdxBlocks"]] = indexer_blocks.data_ptr()
            self.params_host[_P["MaxBlocks"]] = max_blocks
            params = self.params_host.to(self.params.device)
            # Keep the metadata view alive for every cached raw pointer.
            entry = (params, indexer_blocks)
            self.step_params[key] = entry
        # The executor owns KV pools and may release and rebuild them between forwards.
        manager = metadata.kv_cache_manager
        kv_cache = [manager.get_unique_primary_pool()]
        if self.leader:
            kv_cache.append(manager.get_indexer_k_cache_buffers(self.layer_idx))
        return native_ops.glm_megakernel(
            entry[0],
            hidden_states,
            residual,
            topk,
            block_table,
            kv_lens,
            self.shared.mutable,
            kv_cache,
            self.pdl,
        )


class GlmMegakernel:
    """Opt-in replacement of all C1 MoE decoder layers for an eligible verify step."""

    def __init__(self, decode: DecodeTarget, layer_indices: list[int]) -> None:
        self.shared = _Shared(decode.mapping)
        self.layers = {}
        self.manager = None
        indices = set(layer_indices)
        for ordinal, idx in enumerate(layer_indices):
            self.layers[idx] = _Layer(
                decode,
                idx,
                self.shared,
                ordinal % self.shared.copies,
                idx - 1 in indices,
                idx + 1 in indices,
            )
        logger.info(f"[glm5_megakernel] prepared {len(self.layers)} MoE layers for TP16/B1/K7")

    @classmethod
    def create(cls, decode: DecodeTarget) -> GlmMegakernel | None:
        if decode.mapping.tp_size != 16 or os.environ.get("TRTLLM_GLM5_MEGAKERNEL", "0") != "1":
            return None
        if not prefer_pinned():
            raise RuntimeError(
                "GLM megakernel requires pinned host memory for its CUDA error record"
            )
        config = decode.core.model_config.pretrained_config
        indices = [
            idx
            for idx in range(config.first_k_dense_replace, config.num_hidden_layers)
            if idx % config.moe_layer_freq == 0
        ]
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (
            not indices
            or len(indices) % 3
            or (props.major, props.minor) != (10, 7)
            or props.multi_processor_count != _CTAS
        ):
            raise ValueError(
                "GLM megakernel requires complete three-layer groups on SM107 with 212 SMs"
            )
        return cls(decode, indices)

    def begin(self, hidden_states: torch.Tensor, metadata, spec_metadata) -> bool:
        if tuple(hidden_states.shape) != (8, 6144) or hidden_states.dtype != torch.bfloat16:
            return False
        if metadata.num_contexts != 0 or metadata.num_seqs != 1 or metadata.num_tokens != 8:
            return False
        if getattr(metadata, "in_mtp_draft_loop", False):
            return False
        if spec_metadata is not None and any(
            spec_metadata.is_layer_capture(i) for i in self.layers
        ):
            return False
        manager = metadata.kv_cache_manager
        if (
            manager is None
            or manager.tokens_per_block != 64
            or manager.max_seq_len > self.shared.max_keys
        ):
            return False
        if self.manager is None or self.manager() is not manager:
            if torch.cuda.is_current_stream_capturing():
                return False
            for impl in self.layers.values():
                impl.bind(manager)
            self.manager = weakref.ref(manager)
            logger.info("[glm5_megakernel] bound KV cache for TP16/B1/K7 decode")
        return True
