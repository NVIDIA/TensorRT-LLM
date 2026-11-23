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
from .kernel import ClusterAttnRes

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


@cache
def compile_consumer(
    nb, delta, write, final, slots, cap, arch, aux=False, sanitize=False, carveout=None,
    stores_first=False, spread_push=False, poll_warp=False, min_blocks=0, mx_out=False,
):  # fmt: skip
    """``carveout`` only keys the cache: consumers with a preferred carve-out get their
    own function (the attribute is per function). ``aux``: False, True (the updated
    prefix rows) or "mix" (the pre-norm mixture). ``mx_out``: also store the output's
    MXFP8 rows."""
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    options = dict(
        slots=slots, has_delta=delta, write=write, final=final, sanitize=sanitize,
        stores_first=stores_first, spread_push=spread_push, poll_warp=poll_warp,
        min_blocks=min_blocks, out_mc=False, aux_mix=aux == "mix",
    )  # fmt: skip
    if mx_out:
        options.update(mx_out=True)
    kernel = ClusterAttnRes(nb, **options)
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    args += (row,) * bool(aux)
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
        self.aux: Any = None
        self.aux_mix = False
        # TensorRT-LLM local: the bound ring's MXFP8-output variants (``bind_mx``)
        self.mx: Any = None

    def _compile(self, slots, cap, aux=False, mx=False):
        arch = architecture(self.device)
        sanitize = self.sanitize and slots > 1
        first, spread = self.stores_first, self.spread_push
        vote = self.poll_warp and slots > 0
        shape = (*self.key, slots, cap, arch, aux, sanitize, self.carveout)
        shape += (first, spread)
        code = compile_consumer(*shape, vote, 0, mx)
        # The register budget keeps the CTAs resident beside the in-projection.
        budget = RUBIN_VOTE_REGISTERS.get(self.key[0]) if vote else None
        if budget and kernel_registers(code, self.device.index) > budget:
            bound = 64 * 1024 // (128 * budget)
            code = compile_consumer(*shape, vote, bound, mx)
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

    def bind_aux(self, mix=False):
        """Variants that also store the updated prefix rows (a stock aux stream);
        ``mix`` (TensorRT-LLM local): store the pre-norm mixture instead."""
        aux = "mix" if mix else True
        self.aux_mix = mix
        ring = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, aux)
        self.aux = self._compile(0, 0, aux), ring

    def bind_mx(self):
        """TensorRT-LLM local: the bound ring's variants that also store the output's MXFP8
        rows (``ClusterAttnRes.mx_out``; with ``bind_aux``'s, bind after it), which
        ``run_ring(..., mx=(xq, xs))`` launches."""
        slots, cap = self.ring[2]
        aux = None
        if self.aux is not None:
            aux = self._compile(slots, cap, "mix" if self.aux_mix else True, mx=True)
        self.mx = self._compile(slots, cap, mx=True), aux

    def run_ring(self, prefix, blocks, out, aux=None, mx=None):
        """``mx`` (TensorRT-LLM local, ``bind_mx``): the (xq, xs) rows of the output's
        MXFP8 form, stored too."""
        rows = prefix.shape[0]
        if mx is not None:
            if self.mx is None or rows > self.ring[2][1]:
                raise ValueError(f"AttnRes MXFP8 output of {rows} rows: not bound")
            code = self.mx[0] if aux is None else self.mx[1]
            if code is None:
                raise ValueError("AttnRes MXFP8 output with aux: bind_aux before bind_mx")
            extra = tuple(cutlass.Int64(t.data_ptr()) for t in mx)
            return self._launch(code, self.ring[1], prefix, blocks, out, aux, extra)
        if rows > self.ring[2][1]:
            raise ValueError(f"AttnRes ring input of {rows} rows: no ring holds it")
        code = self.ring[0] if aux is None else self.aux[1]
        return self._launch(code, self.ring[1], prefix, blocks, out, aux)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux)

    def _launch(self, code, source, prefix, blocks, out, aux=None, extra=()):
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *aux, *extra
        code(source, prefix, blocks, *args, stream(self.device))
        return out
