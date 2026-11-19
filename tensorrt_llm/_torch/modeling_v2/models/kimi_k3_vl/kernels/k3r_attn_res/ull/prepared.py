# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ....ops.common.k3_ull_decode_shapes import Choice, Path, Shape, rows_policy
from ....ops.common.k3_ull_cute_host import (
    architecture,
    check_registers,
    dynamic_fake,
    dynamic_view,
    fake,
    kernel_registers,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ...k3r_common.ull.policy import batch_table
from .kernel import ClusterAttnRes
from .sharded import compile_gather, owned

# Devices with more SMs than GB300's 152 (Rubin): the AttnRes stores the updated prefix
# row (and its aux copies and saved block) before its second cluster exchange instead
# of after it, so fewer of its stores are outstanding when its grid completes and the
# PDL successor is released. Same values and order per address. They also spread each
# cluster exchange's DSMEM pushes over one thread per (value, peer) (``spread_push``;
# same values; 4 real ranks, us/layer: KdaHead next -0.01..-0.02, in-proj -0.04).
STORES_FIRST_SMS = 152
# Rubin: register budgets (per thread) of the vote-polled (``poll_warp``) variants by
# block count: the LSU-poll variants' allocation, where the vote alone takes more (6
# blocks without a write: 88). The in-projection after an AttnRes loads its CTAs
# beside the AttnRes CTAs, and a larger AttnRes allocation delays them (16 ranks, B8
# serving, 88 instead of 80 registers: MLA in-projection 59% early instead of 100%,
# KDA 88%; +14.8 / +6.1 us/step). A variant over its budget is compiled again with a
# launch bound (``ClusterAttnRes.min_blocks``; same code, no spills) and checked.
RUBIN_VOTE_REGISTERS = {6: 80}
# Wide rows (TP16 Rubin, the wide MoE tail's output ring: the post-tail AttnRes site)
# whose AttnRes runs ``rows.ClusterAttnResRows`` with this many tokens per 7-CTA
# cluster instead of one: ~181 one-token clusters are resident on Rubin, so above that
# many rows the second wave repeats the loads and both cluster exchanges after the
# delta landed. Per token the same arithmetic, sums and roundings: bitwise. 16 ranks
# (axhb-sw3, mirrored, 93 layers, 32k), us/step: B28 (224 rows) -138, B32 -165; the
# AttnRes after the up projection 7.7 -> 5.9 us at B32 (nsys). On one GPU it is equal
# or slower (the second wave there costs ~1 us), and at 129-192 rows ~0.
ROWS_TOKENS: dict = {"*": "193-256:2"}


def rows_tokens(rows):
    return rows_policy(ROWS_TOKENS, Shape(rows), 1)


@cache
def compile_consumer(
    nb, delta, write, final, slots, cap, arch, aux=False, sanitize=False, carveout=None,
    stores_first=False, spread_push=False, poll_warp=False, min_blocks=0, out_mc=False,
    tokens=1, mx_out=False,
):  # fmt: skip
    """``carveout`` only keys the cache: consumers with a preferred carve-out get their
    own function (the attribute is per function). ``tokens`` > 1: the token-batched
    ``ClusterAttnResRows`` (``ROWS_TOKENS``). ``aux``: False, True (the updated prefix
    rows) or "mix" (TensorRT-LLM local: the pre-norm mixture, ``ClusterAttnRes.aux_mix``).
    ``mx_out`` (TensorRT-LLM local, one-token clusters): also the output's MXFP8 rows."""
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    options = dict(
        slots=slots, has_delta=delta, write=write, final=final, sanitize=sanitize,
        stores_first=stores_first, spread_push=spread_push, poll_warp=poll_warp,
        min_blocks=min_blocks, out_mc=out_mc, aux_mix=aux == "mix",
    )  # fmt: skip
    if mx_out:
        options.update(mx_out=True)
    kernel: ClusterAttnRes
    if tokens > 1:
        from .rows import ClusterAttnResRows

        kernel = ClusterAttnResRows(nb, tokens=tokens, **options)
    else:
        kernel = ClusterAttnRes(nb, **options)
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    # The updated prefix row, also stored to an aux stream (row-sharded: multicast).
    args += (cutlass.Int64(0) if out_mc else row,) * bool(aux)
    args += (cutlass.Int64(0), cutlass.Int32(0)) if out_mc else ()
    args += (cutlass.Int64(0), cutlass.Int64(0)) if mx_out else ()
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring.
    ``carveout``: a preferred shared-memory carve-out (percent) for its launches; 100
    lets a big-smem PDL successor become resident on this kernel's SMs while it runs
    (the default carve-out of this small-smem kernel blocks it)."""

    def __init__(self, weights, key, eps, carveout=None):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.sanitize = False
        self.carveout = carveout
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.stores_first = self.spread_push = sms > STORES_FIRST_SMS
        # Rubin: the Lamport delta poll exits by a warp vote (``ClusterAttnRes``).
        self.poll_warp = sms > STORES_FIRST_SMS
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.ring_rows: Any = None  # wide batches' ring (``bind_ring_rows``)
        self.rows_codes: dict = {}  # ROWS_TOKENS: tokens -> (code, aux code or None)
        self.aux: Any = None
        self.aux_mix = False
        self.shard: Any = None  # row-sharded batches (``bind_sharded``)
        # TensorRT-LLM local: the bound ring's MXFP8-output variants (``bind_mx``)
        self.mx: Any = None

    def _compile(self, slots, cap, aux=False, out_mc=False, tokens=1, mx=False):
        arch = architecture(self.device)
        sanitize = self.sanitize and slots > 1
        first, spread = self.stores_first, self.spread_push
        vote = self.poll_warp and slots > 0
        shape = (*self.key, slots, cap, arch, aux, sanitize, self.carveout)
        shape += (first, spread)
        code = compile_consumer(*shape, vote, 0, out_mc, tokens, mx)
        # The register budget keeps one-token CTAs resident beside the in-projection;
        # the token-batched variant has a fraction of their CTAs.
        budget = RUBIN_VOTE_REGISTERS.get(self.key[0]) if vote and tokens == 1 else None
        if budget and kernel_registers(code, self.device.index) > budget:
            bound = 64 * 1024 // (128 * budget)
            code = compile_consumer(*shape, vote, bound, out_mc, tokens, mx)
            what = f"AttnRes ({self.key[0]} blocks, vote poll)"
            check_registers(code, self.device.index, budget, what)
        if self.carveout is None:
            return code
        return prepare_shared_carveout(code, self.device.index, self.carveout)

    @classmethod
    def for_target(cls, target, *, final=False, has_delta=True, carveout=None):
        if final:
            norm, query = target.output_attn_res_norm, target.output_attn_res_proj
            output, nb, write = target.norm, target.num_attn_res_blocks, -1
        else:
            norm = target.self_attention_res_norm
            query, output = target.self_attention_res_proj, target.input_layernorm
            nb = target.prev_valid_blocks
            write = target.block_write_idx if target.is_block_write_layer else -1
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        return cls(weights, (nb, has_delta, write, final), eps, carveout)

    @classmethod
    def for_post(cls, decoder):
        """The AttnRes after the attention's o-projection (the stock decoder layer's
        ``_post_attn_norm``), for steps whose o-projection runs unfused: the delta is
        the reduced o-projection output; a block-write layer starts a new block: the
        kernel (no delta) takes that output as the new prefix and stores it."""
        write = decoder.is_block_write_layer
        norm, query = decoder.mlp_res_norm, decoder.mlp_res_proj
        output = decoder.post_attention_layernorm
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        key = (decoder.prev_valid_blocks + write, not write, -1, False)
        return cls(weights, key, eps)

    def bind_ring(self, mailbox, *, sanitize=False):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm.
        ``sanitize``: the sum maps -0 to +0 (``ClusterAttnRes``)."""
        shape = tuple(mailbox.shape[:2])
        self.sanitize = sanitize
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_ring_rows(self, mailbox):
        """The wide batches' Lamport input ring [slots, cap > 64, 7168] (the wide MoE
        tail's output); ``run_ring`` reads it for batches above the bound ring's cap."""
        slots, cap = shape = tuple(mailbox.shape[:2])
        self.ring_rows = self._compile(slots, cap), to_cute(mailbox), shape
        tokens = {t for t in batch_table(ROWS_TOKENS.get("*", "")).values() if t > 1}
        self.rows_codes = {
            t: (self._compile(slots, cap, tokens=t), None) for t in tokens
        }

    def bind_aux(self, mix=False):
        """Variants that also store the updated prefix rows (a stock aux stream);
        ``mix`` (TensorRT-LLM local): the pre-norm mixture instead (one-token clusters
        only: the token-batched rows variants keep no aux code)."""
        aux = "mix" if mix else True
        self.aux_mix = mix
        ring = rows = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, aux)
        if self.ring_rows is not None:
            slots, cap = self.ring_rows[2]
            rows = self._compile(slots, cap, aux)
            self.rows_codes = {
                t: (code, None if mix else self._compile(slots, cap, True, tokens=t))
                for t, (code, _) in self.rows_codes.items()
            }
        self.aux = self._compile(0, 0, aux), ring, rows

    def bind_mx(self):
        """TensorRT-LLM local: the bound ring's variants that also store the output's MXFP8
        rows (``ClusterAttnRes.mx_out``; with ``bind_aux``'s, bind after it), which
        ``run_ring(..., mx=(xq, xs))`` launches."""
        slots, cap = self.ring[2]
        aux = None
        if self.aux is not None:
            aux = self._compile(slots, cap, "mix" if self.aux_mix else True, mx=True)
        self.mx = self._compile(slots, cap, mx=True), aux

    def bind_sharded(self, box, batches, rank, tp):
        """Row-sharded ``batches`` (``attn_res.sharded``): this rank runs the AttnRes on
        its own rows only (plain delta, or the wide MoE tail's row-owner ring
        ``box.ring``) and multicasts their outputs into every rank's gather box 1 (aux
        rows: box 2), which ``GatherRows`` copies into the plain rows. Bind after the
        rings and ``bind_aux``."""
        aux = self.aux is not None
        codes = {
            (s, a): self._compile(s, box.per_cap if s else 0, a, out_mc=True)
            for s in (0, 1)
            for a in ((False, True) if aux else (False,))
        }
        self.shard = box, frozenset(batches), rank, tp, codes

    def run_owned(self, source, prefix, blocks, out, aux):
        """Row-sharded: this rank's rows of the AttnRes (``source``: the plain delta
        rows, or None for the row-owner ring), multicast, then every row gathered."""
        box, _, rank, tp, codes = self.shard
        rows, s = prefix.shape[0], stream(self.device)
        first, count = owned(rows, tp, rank)
        if count:
            mine = slice(first, first + count)
            code = codes[(int(source is None), aux is not None)]
            if source is None:
                view = to_cute(box.ring.buffer)
            else:
                view = dynamic_view(source[mine])
            local = map(dynamic_view, (prefix[mine], blocks[mine], prefix[mine]))
            prefix_v, blocks_v, dummy = local
            extra = () if aux is None else (cutlass.Int64(box.gathers[2].mc),)
            extra += (cutlass.Int64(box.gathers[1].mc), cutlass.Int32(first))
            args = *self.views, dummy, *self.eps, *extra
            code(view, prefix_v, blocks_v, *args, s)
        gather = compile_gather(rows, box.cap, self.device.index)
        gather(to_cute(box.gathers[1].buffer), to_cute(out), s)
        if aux is not None:
            gather(to_cute(box.gathers[2].buffer), to_cute(aux), s)
        return out

    def sharded(self, rows):
        return self.shard is not None and rows in self.shard[1]

    def run_ring(self, prefix, blocks, out, aux=None, mx=None):
        """``mx`` (TensorRT-LLM local, ``bind_mx``): the (xq, xs) rows of the output's
        MXFP8 form, stored too."""
        rows = prefix.shape[0]
        if mx is not None:
            if self.mx is None or self.sharded(rows) or rows > self.ring[2][1]:
                raise ValueError(f"AttnRes MXFP8 output of {rows} rows: not bound")
            code = self.mx[0] if aux is None else self.mx[1]
            if code is None:
                raise ValueError("AttnRes MXFP8 output with aux: bind_aux before bind_mx")
            extra = tuple(cutlass.Int64(t.data_ptr()) for t in mx)
            return self._launch(code, self.ring[1], prefix, blocks, out, aux, extra)
        if self.sharded(rows):
            return self.run_owned(None, prefix, blocks, out, aux)
        if rows > self.ring[2][1]:  # the rings re-arm what they read: no row past cap
            if self.ring_rows is None or rows > self.ring_rows[2][1]:
                raise ValueError(f"AttnRes ring input of {rows} rows: no ring holds it")
            code = self.ring_rows[0] if aux is None else self.aux[2]
            batched = self.rows_codes.get(rows_tokens(rows))
            if batched is not None:
                code = batched[0] if aux is None else batched[1]
                if code is None:
                    raise ValueError(f"AttnRes aux of {rows} rows: no token-batched mix code")
            return self._launch(code, self.ring_rows[1], prefix, blocks, out, aux)
        code = self.ring[0] if aux is None else self.aux[1]
        return self._launch(code, self.ring[1], prefix, blocks, out, aux)

    def run(self, delta, prefix, blocks, out, aux=None):
        if self.sharded(prefix.shape[0]):
            return self.run_owned(delta, prefix, blocks, out, aux)
        code = self.code if aux is None else self.aux[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux)

    def _launch(self, code, source, prefix, blocks, out, aux=None, extra=()):
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *aux, *extra
        code(source, prefix, blocks, *args, stream(self.device))
        return out


# Wide verify shapes whose AttnRes state is row-sharded over the TP ranks (rows by q,
# "*": any): each rank runs every AttnRes on its own ceil(M / TP) rows and owns those
# prefix / block rows; the plain full rows of the layer inputs are gathered for the
# rest of the step. The runtime also requires the fused wide o-projection and the wide
# MoE tail at the shape (``runtime.decode.SpecializedDecode.sharded``). Empty: at
# 256 rows the owner kernel + gather costs what the full-rows kernel does (TP16
# geometry, us: nb 2 5.4 -> 5.9, nb 4 5.2 -> 5.4, nb 8 6.7 -> 5.6) and the post-tail
# site pays one more hop; it pays from 384 rows (notes/comm3.md section 3e).
SHARDED: dict = {"*": ""}
# Wide verify shapes whose AttnRes sites run the stock op (``runtime.seams
# .StockAttnRes``: the stock forward's own kernel) instead of ClusterAttnRes (one
# 7-CTA cluster per token, 181 clusters per wave), rows by q ("*": any). Above 256
# rows ClusterAttnRes needs three or more waves: 16 ranks, DL7 B64 (M512), the
# 32-layer proxy, exposed us per layer pair, ClusterAttnRes 27.7 vs stock 21.5,
# growing with depth (sol4 section 51). Up to 256 rows it wins (1 GPU, cold HBM, comm3
# arbench.py: M64-128 by 0.7-2.5 us per call at every block count, M256 at four
# blocks or more), except M192 at six or more blocks (a second wave: +0.9-1.3 us).
# Empty (axhb2, hb192): above 256 rows the sites after a wide MoE tail are its ring
# consumer (ClusterAttnRes) anyway, so the stock op served only the head / dense / final
# sites there (and, before runtime.decode skipped pre-consumed sites, ran a second time
# on the ring consumer's output).
STOCK_ROWS: dict = {"*": ""}


def choose(shape, ctx):
    """The AttnRes of a wide decode shape (the runtime's family hook): row-sharded
    owners (GENERIC "sharded"), the stock op (STOCK), or ClusterAttnRes on every row
    (GENERIC "plain")."""
    if ctx.arch == "rubin" and rows_policy(SHARDED, shape, default=0):
        return Choice(Path.GENERIC, ("rows", "sharded"), "sharded")
    if rows_policy(STOCK_ROWS, shape, default=0):
        return Choice(Path.STOCK, "stock", "stock")
    return Choice(Path.GENERIC, "plain", "plain")
