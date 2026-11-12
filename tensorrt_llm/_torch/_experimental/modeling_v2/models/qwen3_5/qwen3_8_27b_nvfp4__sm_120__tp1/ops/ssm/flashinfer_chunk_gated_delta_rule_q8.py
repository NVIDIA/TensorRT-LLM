# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule over whole context sequences, their states in the quantized pool:
the GDN prefill core.

One call: ``g -> exp(g)``; a Triton gather that dequantizes the named slots into a
packed fp32 ``[N, HV, V, K]`` buffer (``_gdn_state_q8``); FlashInfer's SM120 CuTe-DSL
chunked kernel with fp32 initial and final states -- context-parallel, V-split x2 or
fully fused, chosen on the host (``_fi_gdn_prefill``); a Triton scatter that requantizes
the final states into the same slots.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fi_gdn_prefill import chunk_gated_delta_rule_sm120
from ._gdn_state_q8 import (
    dequantize_slots,
    q8_gather_dequant,
    q8_quant_scatter,
    q8_state_rows,
    store_slots,
)

#: This checkpoint's Gated DeltaNet at tp1: 16 q/k heads, 48 v heads, head dims 128.
H, HV, K, V = 16, 48, 128, 128


def _cell(why: str, seqlens: tuple[int, ...], route: str, **spec) -> Cell:
    return Cell(why=why, spec=dict(seqlens=seqlens, route=route, **spec))


class FlashinferChunkGatedDeltaRuleQ8(OpWrapper):
    """Run the gated delta rule over N context sequences packed in one row, starting
    from and returning to their quantized pool slots; writes `o` into `out`.

    Per token `t` of sequence `n` and value head `i` (key head `i // (HV // H)`),
    `h` the `[V, K]` state starting at `deq(pool[slots[n]])`, fp32 accumulation:

        h *= exp(g[t]);  u = (v[t] - h k[t]) * beta[t];  h += u k[t]^T
        o[t] = h (K^-0.5 q[t])

    computed 64 tokens at a time on tensor cores (WY form, bf16 operands); then
    `pool[slots[n]] = quant(h_final)`. `g` is in log space, q/k already L2-normalized
    (the one-pass front end's outputs); `cu_seqlens` splits the row.

    Fusion boundary: gate exponentiation, the slots' dequantize-gather, the recurrence,
    the output, the requantize-scatter. q/k/v/g/beta preparation, the slot reset and
    the gated norm on `o` are the caller's.

    Returns `out` `[1, T, HV, V]` bf16, written in place.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: seqlens; route (the kernel `_fi_gdn_prefill.plan` picks on a 188-SM RTX PRO
    # 6000); pool ("compact" | "coalesced"); zero_slot (sequence 0 starts from a zeroed,
    # i.e. reset, slot). q/k unit-norm bf16, g log-space and beta fp32, as the one-pass
    # front end emits them; `out` the leading rows of a [1, T, 48, 128] step output.
    CELLS: tuple[Cell, ...] = (
        _cell("1 x 2048 context: V-split x2 (two CP chunks would not pay)", (2048,), "vsplit"),
        _cell("1 x 8192 context: context-parallel, 3 chunks of 3072", (8192,), "cp"),
        _cell(
            "a mixed step at concurrency 64, contexts [2048, 2048, 2048, 1920]: fully fused, "
            "coalesced pool",
            (2048, 2048, 2048, 1920),
            "fused",
            pool="coalesced",
        ),
        _cell("ragged contexts [700, 1300, 64, 1]: fully fused", (700, 1300, 64, 1), "fused"),
        _cell(
            "1 x 100 context from a zeroed (reset) slot: V-split", (100,), "vsplit", zero_slot=True
        ),
    )

    note = """
    Metadata consumed: `slots` (the context requests' int32 state indices,
    `state_indices_p`), `cu_seqlens` (int64 `query_start_loc_long[:N + 1]`),
    `num_long_seqs` (host int: the context count N -- decode riders of a mixed step run
    `gdn_decode_t1`, never this op, so every sequence here can be long).

    Call site (modeling): q, k `[1, P, 16, 128]`, v `[1, P, 48, 128]`, g and beta
    `[1, P, 48]` fp32 straight from `gdn_prefill_onepass`; the layer's whole int8 pool
    unsliced; `out` = `output[:, :P]`, the context rows of the contiguous
    `[1, T, 48, 128]` step output. No workspace: the fp32 state copies are allocated
    per call (2 x 3 MiB per sequence).

    Kernel choice (`_fi_gdn_prefill.plan`, host ints only): CP iff FlashInfer's
    parallelism heuristic picks it (on a GDDR card `N * 48 * 3 < SMs`: one sequence on
    188 SMs) and the wave-aware gate agrees, `ceil(P / chunk_len) >= 3` with
    `chunk_len` the CP launch's own one-wave split of the P tokens (2048 tokens -> 2
    chunks of 1024: non-CP; 3072 -> 3 chunks: CP); otherwise non-CP, V-split x2 iff
    `num_long_seqs * 48 * 2 <= SMs` (one sequence), else fully fused. V-split and fully
    fused are bitwise equal; CP sums in another order.

    Pool contract: as `gdn_decode_t1` (int8-typed `[S, 48, 260, 128]`, 16-byte aligned
    slots, any slot stride). The gather / scatter address `slot * stride(0)` in int64.

    Numerics: the initial states dequantized exactly (int16 x fp32 scale, one rounding),
    the chunked recurrence with bf16 MMA operands and fp32 state, `o` rounded to bf16,
    the final states requantized once per row (int16, `amax / 32767`).

    JIT and streams: the first call per kernel variant compiles in-process (seconds);
    FlashInfer keeps one TMA descriptor workspace per device, so two streams must not
    run this op concurrently.

    Not checked by the op (is_valid covers it): slot ids unique and in range (two
    sequences sharing a slot leave its final state nondeterministic), the pool format,
    `cu_seqlens` covering q's rows.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_long_seqs: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        q3 = q.squeeze(0).contiguous()
        k3 = k.squeeze(0).contiguous()
        v3 = v.squeeze(0).contiguous()
        alpha = torch.exp(g.squeeze(0))
        beta2 = beta.squeeze(0).to(torch.float32)
        initial = q8_gather_dequant(pool, slots, v3.shape[2])
        final = torch.empty_like(initial)
        chunk_gated_delta_rule_sm120(
            out.squeeze(0),
            final,
            q3,
            k3,
            v3,
            alpha,
            beta2,
            initial,
            cu_seqlens,
            1.0 / math.sqrt(q3.shape[2]),
            num_long_seqs,
        )
        q8_quant_scatter(final, pool, slots)
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_long_seqs: int,
        out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(o, pool_after)`: the float64 token-at-a-time recurrence from the dequantized
        slots (all sequences stepped in lockstep), `o` rounded once to `q.dtype`, the
        final states quantized into a copy of the pool. Mutates nothing; `out` and
        `num_long_seqs` are ignored (they choose where results land and the launch)."""
        dt = torch.float64
        rep = v.shape[2] // q.shape[2]
        scale = q.shape[3] ** -0.5
        bounds = cu_seqlens.tolist()
        slot_list = slots.tolist()
        starts = torch.tensor(bounds[:-1], device=q.device)
        lengths = torch.tensor([e - s for s, e in zip(bounds[:-1], bounds[1:])], device=q.device)
        h = dequantize_slots(pool, slot_list).to(dt)
        o = torch.zeros(1, q.shape[1], v.shape[2], v.shape[3], dtype=dt, device=q.device)
        for step in range(int(lengths.max().item()) if len(slot_list) else 0):
            live = (lengths > step).nonzero().squeeze(1)
            t = starts[live] + step
            q_t = (q[0, t].to(dt) * scale).repeat_interleave(rep, dim=1)
            k_t = k[0, t].to(dt).repeat_interleave(rep, dim=1)
            hs = h[live] * g[0, t].to(dt).exp()[..., None, None]
            u = (v[0, t].to(dt) - torch.einsum("nhvk,nhk->nhv", hs, k_t)) * beta[0, t].to(dt)[
                ..., None
            ]
            hs = hs + u[..., :, None] * k_t[..., None, :]
            o[0, t] = torch.einsum("nhvk,nhk->nhv", hs, q_t)
            h[live] = hs
        pool_after = pool.clone()
        store_slots(pool_after, slot_list, h.float())
        return o.to(q.dtype), pool_after

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_long_seqs: int,
        out: torch.Tensor,
    ) -> None:
        _, total, hk, kd = q.shape
        hv, vd = v.shape[2], v.shape[3]
        assert k.shape == q.shape and v.shape[:2] == (1, total), (
            "q, k [1, T, H, K]; v [1, T, HV, V]"
        )
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "bf16 q / k / v"
        assert kd == vd == 128 and hv % hk == 0, "K == V == 128, HV a multiple of H"
        for name, t in (("g", g), ("beta", beta)):
            assert t.dtype == torch.float32 and t.shape == (1, total, hv), f"{name} fp32 [1, T, HV]"
        assert out.dtype == torch.bfloat16 and out.shape == v.shape and out.is_contiguous(), (
            "out a contiguous bf16 [1, T, HV, V]: FlashInfer stores o by TMA into it"
        )
        rows = q8_state_rows(vd, kd)
        assert pool.dtype == torch.int8 and tuple(pool.shape[1:]) == (hv, rows, kd), (
            f"int8 pool [S, {hv}, {rows}, {kd}]"
        )
        assert pool.stride()[1:] == (rows * kd, kd, 1), (
            "each slot's [HV, V + 4, K] block contiguous"
        )
        assert pool.stride(0) % 16 == 0 and pool.data_ptr() % 16 == 0, "16-byte aligned slots"
        num_seqs = cu_seqlens.shape[0] - 1
        assert cu_seqlens.dtype == torch.int64 and cu_seqlens.dim() == 1, "int64 cu_seqlens"
        bounds = cu_seqlens.tolist()
        assert bounds[0] == 0 and bounds[-1] == total, "cu_seqlens must cover q's T rows"
        assert all(e >= s for s, e in zip(bounds[:-1], bounds[1:])), "cu_seqlens nondecreasing"
        assert (
            slots.dtype == torch.int32 and slots.shape == (num_seqs,) and slots.is_contiguous()
        ), "slots int32 contiguous [N]"
        slot_list = slots.tolist()
        assert all(0 <= s < pool.shape[0] for s in slot_list), "slots in [0, S)"
        assert len(set(slot_list)) == len(slot_list), (
            "two sequences sharing a slot leave its final state nondeterministic"
        )


flashinfer_chunk_gated_delta_rule_q8 = FlashinferChunkGatedDeltaRuleQ8()
