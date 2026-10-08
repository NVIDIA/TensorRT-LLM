# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ulysses fused FP8 self-attention with the all-to-all hidden behind FMHA.

Same kernels and scales as ops.fp8_self_attention_ulysses; only data movement differs:
- Q/K/V FP8 and per-group FMHA outputs live in symmetric (peer-mapped) memory.
- Receivers pull their head groups with copy-engine 2D copies.
- Ready flags use stream memops, so no SMs spin.
- Local heads split into groups: copies of one group overlap FMHA of another.

Every rank must see all GPUs of its Ulysses group (one node).
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group

from . import _ext
from ._common import FP8, FP8_MAX, HEAD_DIM, rope_tables, shape_buffers

_FLAG_BYTES = 4096

_states = {}
_side_streams = {}


class _State:
    """Symmetric buffers, peer pointers and epoch for one shape and group."""

    def __init__(self, batch, seq_local, heads, groups, world, group_name, device):
        self.head_group = heads // world // groups
        seq = seq_local * world
        self.qkv_bytes = batch * seq_local * heads * HEAD_DIM
        self.out_group_bytes = batch * seq * self.head_group * HEAD_DIM * 2
        total = _FLAG_BYTES + 3 * self.qkv_bytes + groups * self.out_group_bytes
        if hasattr(symm_mem, "enable_symm_mem_for_group"):
            # Older torch needs explicit group registration.
            symm_mem.enable_symm_mem_for_group(group_name)
        self.buf = symm_mem.empty(total, dtype=torch.uint8, device=device)
        self.buf[:_FLAG_BYTES].zero_()
        torch.cuda.synchronize(device)
        handle = symm_mem.rendezvous(self.buf, group_name)
        self.ptrs = [int(p) for p in handle.buffer_ptrs]
        self.epoch = 0
        dist.barrier(group=_resolve_process_group(group_name))
        qkv = self.buf[_FLAG_BYTES : _FLAG_BYTES + 3 * self.qkv_bytes].view(FP8)
        self.q8, self.k8, self.v8 = qkv.view(3, batch, seq_local, heads, HEAD_DIM).unbind(0)
        out = self.buf[_FLAG_BYTES + 3 * self.qkv_bytes :].view(torch.bfloat16)
        self.out = out.view(groups, batch * seq, self.head_group, HEAD_DIM)

    def flag(self, owner, slot, src, world):
        # Slot written by rank src into rank owner's flags.
        return self.ptrs[owner] + 4 * (slot * world + src)

    def qkv_ptr(self, owner, which):
        return self.ptrs[owner] + _FLAG_BYTES + which * self.qkv_bytes

    def out_ptr(self, owner, group):
        return self.ptrs[owner] + _FLAG_BYTES + 3 * self.qkv_bytes + group * self.out_group_bytes


def _streams(device):
    # Peer-pull and local-copy streams, shared by all layers.
    if device not in _side_streams:
        _side_streams[device] = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    return _side_streams[device]


def supported(num_heads: int, world: int, groups: int) -> bool:
    return num_heads % (world * groups) == 0


@torch.library.custom_op("wanfused::fp8_self_attention_ulysses_overlap", mutates_args=())
def fp8_self_attention_ulysses_overlap(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
    group_name: str,
    groups: int,
) -> torch.Tensor:
    """Local [B, S/P, 3*H*D] to [B, S/P, H*D], all-to-all overlapped with FMHA."""
    peer = _ext.peer()
    pg = _resolve_process_group(group_name)
    world, me = dist.get_world_size(pg), dist.get_rank(pg)
    batch, seq_local, _ = qkv.shape
    heads, seq = num_heads, seq_local * world
    heads_local = heads // world
    dev = qkv.device
    key = (batch, seq_local, heads, groups, group_name, str(dev))
    if key not in _states:
        _states[key] = _State(batch, seq_local, heads, groups, world, group_name, dev)
    st = _states[key]
    st.epoch += 1
    epoch, hg = st.epoch, st.head_group
    cur = torch.cuda.current_stream(dev)
    side, local = _streams(dev)
    cs, ss, ls = cur.cuda_stream, side.cuda_stream, local.cuda_stream

    # Global V scale, as in ops.fp8_self_attention_ulysses.
    amax = torch.ops.wanfused.v_amax(qkv, num_heads)
    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=pg)
    scale_v = amax.clamp_min(1e-12) / FP8_MAX
    bufs = shape_buffers(batch, seq_local, dev)
    mul = torch.cat([bufs["qk_mul"], (1.0 / scale_v).float().reshape(1)])
    qkv2d = qkv.reshape(batch * seq_local, -1).contiguous()
    cos2d, sin2d, seq_per_batch = rope_tables(cos, sin, qkv2d.shape[0])
    _ext.prep().norm_rope_quant(
        qkv2d,
        num_heads,
        eps,
        norm_q_w,
        norm_k_w,
        cos2d,
        sin2d,
        interleave,
        seq_per_batch,
        mul,
        bufs["amax"],
        st.q8,
        st.k8,
        st.v8,
    )
    for p in range(world):
        if p != me:
            peer.signal(st.flag(p, 0, me, world), epoch, cs)
    ev_start = torch.cuda.Event()
    ev_start.record(cur)

    # FMHA inputs [3, G, B, S, Hg, D] and the final output, filled by copy engines.
    fin = torch.empty(3, groups, batch, seq, hg, HEAD_DIM, device=dev, dtype=FP8)
    final = torch.empty(batch, seq_local, heads, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    final_g = final.view(batch, seq_local, world, groups, hg, HEAD_DIM)
    for stream in (side, local):
        stream.wait_event(ev_start)
    for p in range(world):
        if p != me:
            peer.wait_geq(st.flag(me, 0, p, world), epoch, ss)

    # Pull Q/K/V head group g from every rank; own chunk on the local stream.
    ev_in = []
    row_bytes = heads * HEAD_DIM
    for g in range(groups):
        for i in range(3):
            for p in range(world):
                for b in range(batch):
                    dst = fin[i, g, b, p * seq_local].data_ptr()
                    src = (
                        st.qkv_ptr(p, i)
                        + (b * seq_local * heads + me * heads_local + g * hg) * HEAD_DIM
                    )
                    stream = ls if p == me else ss
                    peer.copy2d(
                        dst, hg * HEAD_DIM, src, row_bytes, hg * HEAD_DIM, seq_local, stream
                    )
        evs = (torch.cuda.Event(), torch.cuda.Event())
        evs[0].record(side)
        evs[1].record(local)
        ev_in.append(evs)

    # Compute stream: FMHA per group into symmetric output, then signal peers.
    sv = scale_v.float().reshape(1).contiguous()
    fb = shape_buffers(batch, seq, dev)
    out_bytes = hg * HEAD_DIM * 2
    for g in range(groups):
        for ev in ev_in[g]:
            cur.wait_event(ev)
        q, k, v = (fin[i, g].view(batch * seq, hg, HEAD_DIM) for i in range(3))
        _ext.fmha().fmha(
            q,
            k,
            v,
            fb["cu_seqlens"],
            fb["seqlens"],
            fb["bmm1"],
            sv,
            batch,
            seq,
            True,
            True,
            st.out[g],
        )
        for p in range(world):
            if p != me:
                peer.signal(st.flag(p, 1 + g, me, world), epoch, cs)
        ev_fmha = torch.cuda.Event()
        ev_fmha.record(cur)
        # Own rows of this group's output, copied on the local stream.
        local.wait_event(ev_fmha)
        for b in range(batch):
            src = st.out_ptr(me, g) + (b * seq + me * seq_local) * out_bytes
            dst = final_g[b, 0, me, g].data_ptr()
            peer.copy2d(dst, row_bytes * 2, src, out_bytes, out_bytes, seq_local, ls)

    # Side stream: pull my token rows of each peer's group output.
    for g in range(groups):
        for p in range(world):
            if p == me:
                continue
            peer.wait_geq(st.flag(me, 1 + g, p, world), epoch, ss)
            for b in range(batch):
                src = st.out_ptr(p, g) + (b * seq + me * seq_local) * out_bytes
                dst = final_g[b, 0, p, g].data_ptr()
                peer.copy2d(dst, row_bytes * 2, src, out_bytes, out_bytes, seq_local, ss)
    for stream in (side, local):
        ev = torch.cuda.Event()
        ev.record(stream)
        cur.wait_event(ev)
    return final.view(batch, seq_local, heads * HEAD_DIM)


@fp8_self_attention_ulysses_overlap.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, group_name, groups):
    batch, seq_local, _ = qkv.shape
    return qkv.new_empty(batch, seq_local, num_heads * HEAD_DIM)
