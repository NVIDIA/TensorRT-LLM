# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared MoE operand preparation; targets retain policy and cache state."""

from typing import Optional

import torch

HIDDEN = 6144
NUM_EXPERTS = 128


def _view(t: torch.Tensor, align: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align)


def is_supported(device: Optional[torch.device] = None) -> tuple:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
        return False, "needs the sm_100 family (tcgen05 block-scaled NVFP4 MMA)"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    return True, ""


def ar_push_desc(
    mc_ptr: int, flags_ptr: int, rank: int, world: int, max_tokens: int, device
) -> torch.Tensor:
    """The ``arp`` descriptor that makes a call also push its bf16 output into a multicast all-reduce exchange (set A
    of m3_mc_allreduce's workspace at ``mc_ptr``, slot half = the all-reduce CTA's call parity in ``flags_ptr``).
    With it the kernel releases its PDL dependents only after every push; the default (enable 0) keeps the old
    behaviour."""
    return torch.tensor(
        [1, mc_ptr, flags_ptr, rank, world, max_tokens, 0, 0], dtype=torch.int64, device=device
    )


def shared_views(shared_weights, who, *, shared_inter: int):
    """(gate_up, its scales, down, its scales) of the shared expert as flat uint8 views, checked."""
    if shared_weights is None:
        raise ValueError(f"{who} (shared): shared_weights are required")
    gu_w, gu_sf, dn_w, dn_sf = shared_weights
    want = {"gate_up": (gu_w, (2 * shared_inter, HIDDEN)), "down": (dn_w, (HIDDEN, shared_inter))}
    for name, (t, shape) in want.items():
        if t.dtype != torch.float8_e4m3fn or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"{who} (shared): {name} weight must be contiguous e4m3 {shape}")
    if (
        gu_sf.numel() != 2 * shared_inter * HIDDEN // 32
        or dn_sf.numel() != HIDDEN * shared_inter // 32
    ):
        raise ValueError(
            f"{who} (shared): the UE8M0 scales must be the swizzled [N, K / 32] buffers"
        )
    return (
        gu_w.view(torch.uint8).view(-1),
        gu_sf.view(torch.uint8).reshape(-1),
        dn_w.view(torch.uint8).view(-1),
        dn_sf.view(torch.uint8).reshape(-1),
    )


def pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta: int = 24, *, shared_inter: int) -> torch.Tensor:
    """The shared expert's weights as m3_moe's shared CTAs stream them (SH_PRETILE): per CTA s, per stage g, the 16 KB
    shared-memory image (128-byte swizzle: 16-byte chunk j of row r at j ^ (r % 8)) of the stage's 128-row MMA tile and
    its 512-byte instruction scale atom (row R's 4 codes at word 4 (R % 32) + R / 32). CTA s owns P = SHARED_INTER / 24
    = 16 positions in SPLIT = 128 / (2 P) = 4 K slices of GT = 48 / SPLIT = 12 k-tiles. Stages 0..GT-1: MMA row
    R = P O + i (O = 2 q + u: slice q, u 0 gate / 1 up) = weight row u SHARED_INTER + P s + i at k-tile GT q + g; then
    down tile j (rows 256 s + 128 j ..) at k-tile kt, j-major (SHARED_INTER / 128 k-tiles each). Built once per
    layer."""
    dev = gu8.device
    H_, I_ = HIDDEN, shared_inter
    P = I_ // n_cta
    split = 128 // (2 * P)
    gt = H_ // 128 // split
    dkt = I_ // 128
    gu = gu8.view(2 * I_, H_)
    dn = dn8.view(H_, I_)
    gkb, dkb = H_ // 32, I_ // 32

    def sf_bytes(sf, n, b, kbn):  # code of (row n, K block b), broadcast
        return sf[
            ((n >> 7) * (kbn // 4) + (b >> 2)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (b & 3)
        ]

    r = torch.arange(128, device=dev)
    q = r >> 5
    ri = r & 31
    octant = r // P
    chunk = torch.arange(8, device=dev)
    swz = chunk[None, :] ^ (r[:, None] & 7)  # source chunk of image chunk j: j ^ (r % 8)
    out = torch.empty(
        n_cta, gt + 2 * dkt, 16384 + 512, dtype=torch.uint8, device=dev
    )  # capacity: fixed: a workspace of the kernel's fixed geometry
    for s in range(n_cta):
        # gate_up: MMA row R -> weight row, slice
        nrow = (octant & 1) * I_ + P * s + r % P
        sl = octant >> 1
        for g in range(gt):
            kt = sl * gt + g  # per row
            tile = gu[
                nrow[:, None], (kt[:, None] * 128 + torch.arange(128, device=dev)[None, :])
            ]  # [128, 128]
            img = tile.view(128, 8, 16)[r[:, None], swz]  # image chunk j = tile chunk j ^ (r % 8)
            out[s, g, :16384] = img.reshape(-1)
            kb = torch.arange(4, device=dev)
            codes = sf_bytes(gusf8, nrow[:, None], kt[:, None] * 4 + kb[None, :], gkb)  # [128, 4]
            atom = torch.empty(512, dtype=torch.uint8, device=dev)
            atom[(ri[:, None] * 16 + q[:, None] * 4 + kb[None, :]).reshape(-1)] = codes.reshape(-1)
            out[s, g, 16384:] = atom
        for gd in range(2 * dkt):
            jt, kt = gd // dkt, gd % dkt
            row0 = 256 * s + 128 * jt
            tile = dn[row0 : row0 + 128, kt * 128 : (kt + 1) * 128]
            img = tile.reshape(128, 8, 16)[r[:, None], swz]
            out[s, gt + gd, :16384] = img.reshape(-1)
            out[s, gt + gd, 16384:] = dnsf8[
                ((row0 >> 7) * (dkb // 4) + kt) * 512 : ((row0 >> 7) * (dkb // 4) + kt) * 512 + 512
            ]
    return out.view(-1)


def fine_ranges(weights, total: int, *, stages: int):
    """Contiguous FC1 item ranges [s_c, e_c) over the CTAs, sized as close to ``weights`` (ideal real-valued item
    counts summing to ``total``) as the split-unit protocol allows: every unit (S1 items) has at most one CTA boundary
    strictly inside it (its head and tail parts, summed by the tail's holder), so a range that starts inside a unit
    runs at least to that unit's end. Returns the item counts."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
    for c in range(n):
        acc += weights[c]
        e = total if c == n - 1 else min(total, max(s, int(acc + 0.5)))
        if s % stages and e < total:  # started inside a unit: run to (at least) its end
            e = max(e, min(total, (s // stages + 1) * stages))
        items[c] = e - s
        s = e
    assert s == total
    return items


class M3MoeLayer:
    """One MoE layer on a :class:`M3MoeState`: ``layer(hidden)``. The expert buffers are read in
    place, never copied; the per-expert scalars are packed once (layer constants: rebuild the layer after reloading
    weights)."""

    shared_weights = None  # the kernel's view of the shared expert (a shared state)
    shared_mxfp8 = None  # the shared expert's (gate_up, its scales, down, its scales) as given

    def __init__(
        self,
        state,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        bias,
        router_weight=None,
        fc31_input_scale=None,
        *,
        inter: int,
    ):
        e = NUM_EXPERTS
        expected = {
            "w3_w1_weight": (w3_w1_weight, (e, 2 * inter, HIDDEN // 2)),
            "w3_w1_weight_scale": (w3_w1_weight_scale, (e, 2 * inter, HIDDEN // 16)),
            "w2_weight": (w2_weight, (e, HIDDEN, inter // 2)),
            "w2_weight_scale": (w2_weight_scale, (e, HIDDEN, inter // 16)),
        }
        for name, (t, shape) in expected.items():
            if t.element_size() != 1 or tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError(
                    f"m3_moe: {name} is {t.dtype} {tuple(t.shape)}, expected contiguous 1-byte {shape}"
                )
        self.state = state
        self.w3_w1_weight = w3_w1_weight.view(torch.uint8)
        self.w3_w1_weight_scale = w3_w1_weight_scale.view(torch.uint8)
        self.w2_weight = w2_weight.view(torch.uint8)
        self.w2_weight_scale = w2_weight_scale.view(torch.uint8)
        self.escal = torch.stack(
            [fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha]
        ).float()
        self.escal = self.escal.reshape(6, e).contiguous()
        self.bias = bias.float().contiguous()
        if (
            router_weight is None
            or fc31_input_scale is None
            or router_weight.dtype != torch.float32
            or tuple(router_weight.shape) != (e, HIDDEN)
            or not router_weight.is_contiguous()
            or fc31_input_scale.dtype != torch.float32
            or fc31_input_scale.numel() != 1
        ):
            raise ValueError(
                "m3_moe (fused router): needs router_weight f32 [128, 6144] and fc31_input_scale f32 [1]"
            )
        self.router_weight = router_weight
        self.fc31_input_scale = fc31_input_scale

    def __call__(
        self,
        hidden: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``hidden``: contiguous bf16 [M, 6144]. ``arp`` (int64 [8], see :func:`ar_push_desc`): also push the output
        into an all-reduce exchange."""
        return self.state.run(self, hidden, out, arp=arp)
