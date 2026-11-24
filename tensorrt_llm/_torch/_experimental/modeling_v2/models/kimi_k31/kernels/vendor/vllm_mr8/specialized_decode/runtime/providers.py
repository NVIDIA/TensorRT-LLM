# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified by NVIDIA for TensorRT-LLM (2026): import statements and comments only.
"""Per-layer attention providers (input projection, KDA/MLA core, o-projection) and
the helpers ``SpecializedDecode.build_providers`` uses."""

import contextlib
from typing import Any

import torch

from ..._vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from ..._vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from ..cutedsl.attn_res.prepared import PreparedAttnRes
from ..cutedsl.common.policy import batch_table
from ..cutedsl.gemm.projection import PreparedProjection
from ..cutedsl.kda.head import PreparedKdaHead, polled_out_batches
from ..cutedsl.kda.inproj64 import PREPARED_PITCH, PreparedKdaInProj64
from ..cutedsl.kda.prepared import PreparedKda
from ..cutedsl.kda.prepared import choose as kda_choose
from ..cutedsl.mla.dcp import PAIR_SENT, DcpBox, polled_output
from ..cutedsl.mla.prepared import SWITCH, PreparedMla
from ..cutedsl.mla.prepared import choose as mla_choose
from ..cutedsl.mla.query import POLLED_X
from ..cutedsl.mla.rows import VERIFY_ROWS
from ..cutedsl.oproj.prepared import PreparedOproj, fused_schedule, shared_x
from ..cutedsl.oproj.prepared import choose as oproj_choose
from ..shapes import Path, Shape, rows_policy
from .seams import StockAttnRes, gate_sigmoid_mul, warm_gate

# Decode batches (tokens) served by the prepared providers: every batch up to 8, the
# padded CUDA-graph batches and each further captured size; others run stock.
BATCHES = (*range(1, 9), 16, 32, 64)
PREPARED_MAX = BATCHES[-1]


# Devices with more SMs than GB300's 152 (Rubin: 212), TP16: KDA batches whose input
# projection and recurrence run as one kernel (``KdaHead``, BF16 recurrent state).
# Single-GPU chain behind an AttnRes-like predecessor, B8: -0.36 us per KDA layer.
HEAD_BATCHES = (8,)
# Rubin TP16 (BF16 state) verify rows per q whose KDA layers run the in-projection and
# the verify recurrence of several requests in one kernel (``KdaHeadMulti``: the M8
# head with a request axis) instead of the in-projection -> KdaSplit pair. 1-GPU KDA
# chain beside tail / AttnRes / o-proj footprints, M16 (2 x q8), the o-proj's PDL wait
# after the AttnRes end: 12.40 -> 9.09 us. The projection
# sums K in four split-K partials (the M8 head's split) instead of SplitKTmemGemm64's
# six: a stated FP32 accumulation reorder of the projected inputs. M8 at q4 (2 x 4,
# DL3 B2): the M8 head's projection with all sixteen CTAs running a 4-token KDA role
# (the B1 x q8 head leaves eight idle), so DL3 B2 takes DL7 B1's KDA layer (head ->
# polled fused o-projection); its projection sums K in four split-K partials instead
# of the 8-row bin's two (the same stated FP32 reorder).
HEAD_MULTI = {8: "16:1", 4: "8:1"}
# Devices with more SMs than GB300's 152 (Rubin: 212), TP16 B<=8, by rows. KDA: 102 CTAs
# (split 2) instead of 204, so every CTA starts and preloads its weights beside the
# 56-CTA AttnRes. MLA: SplitKTmemGemm, M128 tiles split over K by 7 (161 CTAs in
# clusters of 7, 16 K tiles of 64 each), 12 ring stages, 15 K tiles staged in tensor
# memory as they land, owners 4 (one-hop reduce-scatter); with the preceding
# AttnRes at carve-out 100 (``MLA_BEFORE_CARVEOUT``) every CTA is resident and preloaded
# before x lands. 1-GPU MLA chain (tail -> AttnRes -> in-proj -> MlaQuery), us/layer vs
# (64, 8, 4, 12): -0.8 cold, -0.66 warm (without the carve-out: +1.0). Split 7 sums the
# FP32 partials over K/7 in rank order (an accumulation reorder of split 4). Skinny
# B1/B2 tactics by rows on those devices: 8 outputs per CTA at B1; the KDA rows run B2
# on split-K.
WIDE_B8 = {3216: (64, 8, 2, 12), 2880: (128, 8, 7, 12, 15, None, 4)}
# Preferred shared-memory carve-out of the AttnRes before an MLA layer whose input
# projection runs SplitKTmemGemm (its ~220 KB CTAs then fit beside the AttnRes CTAs).
MLA_BEFORE_CARVEOUT = 100
# Preferred shared-memory carve-out of the AttnRes before a KdaHead layer: at the
# driver default its small-smem CTAs leave their ~56 SMs in a 32 KB configuration,
# where no KdaHead CTA fits; at 100 a KdaHead CTA (which asks only for the shared
# memory it uses) is placed beside an AttnRes CTA while the AttnRes waits for the MoE
# output.
HEAD_BEFORE_CARVEOUT = 100
# Devices with more SMs than GB300's 152 (Rubin), B64 (spec verify B8 x q8): split-K
# stages of both in-projections (64, 64, 2, stages). 1-GPU, cold weights, us per call
# after a 10 us PDL predecessor, MLA rows: 6 stages 7.75 vs 5 stages 8.34 (KDA rows
# already 6: 8.04). Same split and MMA order: bitwise identical.
WIDE_B64_STAGES = 6
# Devices with more SMs than GB300's 152 (Rubin), per TP: verify rows (8 requests x q
# tokens) whose KDA in-projection (the WIDE_B64 tactic; x through a 3-D map of 8 token
# columns per request at q < 8) runs KdaSplit's input prologue in its epilogue
# (KdaInProj64: conv, activations, q / k norms, the conv window) and whose verify
# recurrence reads those operands (KdaVerifyPrepared). One gate for both: the epilogue
# rewrites the conv window that KdaSplit would read. TP16 B8 x q8 (1-GPU, the
# in-projection -> verify chain behind an AttnRes-like predecessor and before an
# o-projection stand-in; the verify's gate warp polling F_A): 14.85 vs 16.05 us per
# layer for the plain projection -> KdaSplit. Rows 32 (q4, KdaInProjQ -> KdaVerifyCols)
# in the same chain: 11.29 vs 11.77 us per layer (the o-projection released 0.6 us
# earlier); the fold's in-projection costs ~+1.0 us at q4 and q8 alike, the q4 verify
# saves less. (A pair timed with no consumer, ended by its slowest CTA, shows the
# fold losing at every q, q8 included.) Rows 40 / 48 (q 5 / 6) fold bitwise, untimed.
KDA_FOLD64: dict[int, str] = {16: "32:1,64:1"}
# Devices with more SMs than GB300's 152 (Rubin), TP16 bins 16-64, by rows:
# SplitKTmemGemm64 (128, 64, split, x's slots, 14): the weights of each 128-row tile's K
# slice staged in tensor memory (and shared memory) before the PDL wait, clusters of
# ``split`` (Rubin co-schedules 28 clusters of 6 and 24 of 7 such CTAs: KDA's 26 tiles
# take 6, MLA's 23 take 7), one-hop DSMEM reduce-scatter. 1-GPU, cold weights, us per
# call after a 10 us carve-out-100 PDL predecessor (56 small CTAs) vs (64, 64, 2, 6):
# KDA 4.52 vs 7.72, MLA 4.29 vs 7.53. The 16- and 32-row bins too (5 us predecessor, us
# per call at 16 / 24 / 32 rows vs today's (64, 16, 2, 10) / (64, 32, 2, 8)): KDA 3.84 /
# 3.88 / 3.97 vs 4.86 / 5.68 / 5.71, MLA 3.69 / 3.71 / 3.82 vs 4.64 / 5.64 / 5.72; the
# 8-row bin keeps its tactics (KDA 4.44 vs 6.34, MLA 2.24 vs 5.47). Sums the split
# partials in rank order (an FP32 accumulation reorder of split 2).
WIDE_B64 = {3216: (128, 64, 6, 14, 14), 2880: (128, 64, 7, 16, 14)}
# Devices with more SMs than GB300's 152 (Rubin), TP16 MLA rows, 9-32 verify rows: the
# WIDE_B8 kernel (SplitKTmemGemm: 15 tensor-memory tiles, a dedicated owner mailbox, no
# cluster barrier before the reduce-scatter) with 16 / 32 token columns instead of the
# 64-column WIDE_B64 kernel, with the ring stages that fit beside three AttnRes CTAs
# (217 / 207 KB, carve-out 100). Same split-7 K slices and rank-order sums as WIDE_B64.
WIDE_B16_32 = {
    2880: {16: (128, 16, 7, 10, 15, None, 4), 32: (128, 32, 7, 6, 15, None, 4)}
}
# Devices with more SMs than GB300's 152 (Rubin), TP16 verify rows above the 64-row
# bin, by in-projection rows (KDA 3216, MLA 2880): SplitKTmemGemm64 with one N tile of
# 80 / 96 token columns (13 A tiles in tensor memory beside the accumulator, 8 x
# slots) instead of PROJECTION_GENERIC's split-K. WIDE_B64's split (6 / 7) and
# rank-order owner sums: every row's output is bitwise the 64-row kernel's on that row
# (the FP32 reorder vs split-K's 2 partials that the 64-row bin already takes). 1 GPU,
# cold weights, us per call behind a MoE-tail stand-in and an AttnRes stand-in of 7 M
# CTAs (carve-out 100), split-K -> this, KDA / MLA: M72 9.34 -> 4.99 / 8.94 -> 4.90,
# M80 9.36 -> 4.99 / 8.98 -> 5.05, M88 9.36 -> 8.03 / 9.00 -> 7.77, M96 9.36 -> 8.11 /
# 9.02 -> 7.86 (the tensor-memory CTAs wait once the stand-in puts 3 CTAs on an SM:
# N80 at M72 behind 560 / 616 stand-in CTAs 4.93 / 7.68). N128 at 97-128 rows loses
# (KDA 9.3-10.1 vs 9.3-9.5, MLA 9.4-9.8 vs 8.9-9.3): split-K stays there.
WIDE_BN = {
    3216: "65-80:128.80.6.8.13,81-96:128.96.6.8.13",
    2880: "65-80:128.80.7.8.13,81-96:128.96.7.8.13",
}
SKINNY = {1: (128, 4, 1), 2: (64, 4, 2)}
WIDE_SKINNY = {3216: {1: (128, 8, 1)}, 2880: {1: (128, 8, 1), 2: (64, 4, 2)}}
# Token-row bins of the tested in-projection tactics.
BINS = (8, 16, 32, 64)
# Rows above the tested bins (M = requests x q), by q ("*": any): a generic, untuned
# split-K tactic. The shape-dynamic split-K GEMM repeats its N tile over the rows, so
# one compile serves every such batch. Rows without an entry run cuBLAS (the stock
# GEMM): 1 GPU, cold weights, us per call, 64.64.2.6 vs cuBLAS, KDA (3216 x 7168) /
# MLA (2880 x 7168): M128 8.73 / 8.47 vs 9.30 / 10.51; M256 17.1 / 16.8 vs 9.82 /
# 9.76; M512 33.7 / 32.2 vs 13.06 / 12.92.
PROJECTION_GENERIC = {"*": "65-128:64.64.2.6"}
# The most rows the generic split-K tactics serve (dense layer included); above, the
# projections run cuBLAS.
GENERIC_ROWS = 128
# Rubin TP16 wide rows (token rows, by in-projection rows: KDA 3216, MLA 2880) whose
# cuBLAS in-projection (``Attention.run_wide``) runs with torch's cuBLASLt preference:
# cuBLAS's heuristic pick is banded and loses there. Both libraries sum K in the same
# order at these rows (outputs bitwise equal at every 4th row of 164-768). 1 GPU, 8
# cold weight copies, graphs of 16 calls, us per call cuBLAS -> cuBLASLt (ABBA): KDA
# 260-288 10.8 -> 10.4, 292-336 11.9 -> 11.1, 388-448 13.0 -> 12.4, 452-464 13.6 ->
# 12.5, 468-480 13.0 -> 12.4, 516-576 15.2 -> 13.9; MLA 292-320 11.6 -> 11.0, 324-384
# 16.3 -> 11.0-12.7, 388-512 12.6 -> 12.1, 516-576 19.9 -> 13.6. cuBLAS wins at KDA
# 340-384 / 484-512 / 580-704 and MLA 260-288 / 708-768 (left out).
INPROJ_LT = {
    3216: "260-336:1,388-480:1,516-576:1",
    2880: "292-576:1",
}


def projection_policies(tp, rows, batches, sms=152, q=1):
    """Split-K (M tile, N tile, split, stages) per execution rows' next tested bin;
    rows above the bins take ``PROJECTION_GENERIC``."""
    policies = {}
    tactic: tuple[int | None, ...]
    for batch in batches:
        n = next((b for b in BINS if batch <= b), None)
        wide = batch_table(WIDE_BN.get(rows, "")) if sms > 152 and tp == 16 else {}
        if n is None and batch in wide:
            policies[batch] = wide[batch]
            continue
        if n is None:
            shape = Shape(batch, q)
            tactic = rows_policy(PROJECTION_GENERIC, shape, default=None)
            if tactic is not None:  # else cuBLAS (``Attention.project``)
                policies[batch] = tactic
            continue
        if sms > 152 and tp == 16 and n in WIDE_B16_32.get(rows, {}):
            tactic = WIDE_B16_32[rows][n]
        elif sms > 152 and n in (16, 32, 64) and tp == 16 and rows in WIDE_B64:
            tactic = WIDE_B64[rows]
        elif sms > 152 and n == 8 and rows in WIDE_B8:
            tactic = WIDE_B8[rows]
            if tp == 16 and rows == 2880 and batch in POLLED_X[16]:
                # MlaQuery polls x: no element may be its 0xFFFF sentinel
                tactic += (1,)
        elif rows == 6288 and (tp == 8 or n > 8):
            tactic = (64, 64, 1, 6) if n == 64 else (128, n, 2, 5 if n == 32 else 6)
        else:  # TP16 KDA B64: a sixth stage, -0.7 us per layer in the KDA chain
            stages = {8: 8, 16: 10, 32: 8, 64: 6 if rows == 3216 else 5}[n]
            if sms > 152 and n == 64:  # Rubin MLA B64 too (``WIDE_B64_STAGES``)
                stages = WIDE_B64_STAGES
            tactic = (64, n, 4 if n == 8 else 2, stages)
        policies[batch] = tactic
    return policies


@contextlib.contextmanager
def blas_library(name):
    """Torch's preferred BLAS library ``name`` around the block (None: unchanged),
    restored after it. Graph captures record the kernel picked at capture."""
    if name is None:
        yield
        return
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(name)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


class SkinnyProjection:
    """B1/B2 skinny GEMM; static K loads two weight tiles before the PDL wait."""

    def __init__(self, weight, fallback, tactics=SKINNY):
        self.weight, self.fallback, k = weight, fallback, weight.shape[1]
        self.configs = {m: SkinnyGemmConfig(m, *t, 8, k) for m, t in tactics.items()}
        for m, cfg in self.configs.items():  # compile before capture
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)

    def run(self, x, out):
        cfg = self.configs.get(x.shape[0])
        if cfg is None:
            return self.fallback.run(x, out)
        return shape_dynamic_skinny_gemm(x, self.weight, cfg, None)


def bf16(runtime):
    return dict(device=runtime.device, dtype=torch.bfloat16)


def scratch(runtime, name, shape, dtype=torch.bfloat16):
    if name not in runtime.buffers:
        runtime.buffers[name] = torch.empty(shape, device=runtime.device, dtype=dtype)
    return runtime.buffers[name]


class Attention:
    """Input projection -> KDA/MLA core -> o-projection with its fused AttnRes."""

    def __init__(self, decoder, runtime, prefetch, mailbox, small_batch, twoshot=None):
        attention, tp = decoder.self_attn, runtime.tp
        self.kind = kind = "kda" if hasattr(attention, "in_proj_qkvgfab") else "mla"
        name = "in_proj_qkvgfab" if kind == "kda" else "fused_qkv_a_g_proj"
        weight = getattr(attention, name).weight
        pitch = weight.shape[0]
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        policies = projection_policies(tp, pitch, runtime.batches, sms, runtime.q)
        split_k = PreparedProjection(weight, policies).prepare()
        skinny = WIDE_SKINNY.get(pitch, SKINNY) if sms > 152 else SKINNY
        self.projection = SkinnyProjection(weight, split_k, skinny)
        self.weight = weight
        # Wide rows whose cuBLAS in-projection prefers cuBLASLt (``INPROJ_LT``)
        lt = INPROJ_LT.get(pitch, "") if sms > 152 and tp == 16 else ""
        self.lt_rows = batch_table(lt)
        self.core: Any  # PreparedKda or PreparedMla
        self.core_long: Any = None  # MLA: the step switch's body 1 core (SWITCH)
        # MLA decode batches whose o-projection polls DcpValue's output (one of two
        # per-parity Lamport buffers, mla.dcp.POLLED_OUTPUT): batch -> (own, other)
        self.polled_out: dict[int, tuple] = {}
        self.runtime = runtime
        self.head: Any = None  # PreparedKdaHead (Rubin TP16 KDA batches)
        self.head_index = -1  # this layer's KdaHead index (its polled output buffer)
        # The preceding AttnRes's preferred shared-memory carve-out (None: default).
        self.before_carveout = None
        # MLA batches whose in-projection stages its weights in tensor memory.
        self.tmem_batches = ()
        if kind == "mla" and any(len(t) > 4 for t in policies.values()):
            self.before_carveout = MLA_BEFORE_CARVEOUT
            self.tmem_batches = tuple(b for b, t in policies.items() if len(t) > 4)
        if kind == "kda" and any(len(t) > 4 for t in policies.values()):
            # SplitKTmemGemm64's CTAs are resident beside the preceding AttnRes
            self.before_carveout = MLA_BEFORE_CARVEOUT
        self.fold64: Any = None  # PreparedKdaInProj64 (KDA_FOLD64 verify rows)
        # Wide batches whose MLA runs PreparedMla (``mla.prepared.choose``); the
        # others run the stock MLA attention (seam S2, ``run_wide``).
        self.mla_wide: frozenset[int] = frozenset()
        # Wide batches whose KDA core runs the stock KDA forward (seam S1,
        # ``kda.prepared.choose``); the others run KdaSplit.
        self.kda_stock: frozenset[int] = frozenset()
        if kind == "kda":
            tactics = policies
            mode = lambda b: (1 if b == 1 else 2 if b <= 8 else 0) if tp == 8 else 0
            policies = {b: (pitch, mode(b)) for b in runtime.tuned}
            q = runtime.q  # verify steps: B requests x q tokens
            shapes = (Shape(m, q) for m in runtime.wide)
            self.kda_stock = frozenset(
                s.rows
                for s in shapes
                if kda_choose(s, runtime.context).path is Path.STOCK
            )
            kdasplit = [b for b in runtime.batches if b not in self.kda_stock]
            verify = [b // q for b in kdasplit if q > 1 and b % q == 0]
            policies.update({(b, q): (pitch, int(tp == 8)) for b in verify})
            fold = batch_table(KDA_FOLD64.get(tp, "") if sms > 152 else "")
            folded = 64 in tactics and len(tactics[64]) > 4 and pitch == 3216
            folded = folded and bool(fold.get(8 * q)) and 4 <= q <= 8 and 8 in verify
            prepared = ((8, q),) if folded else ()
            dim_first = is_conv_state_dim_first()
            self.core = PreparedKda(attention, policies, dim_first, prepared=prepared)
            if folded:
                self.fold64 = PreparedKdaInProj64(
                    weight, self.core, tactics[64][:4], tokens=q
                )
                self.prepared = scratch(
                    runtime, "kda.prepared", (64, PREPARED_PITCH), torch.float32
                )
                # The folded rows' own projection buffer, one per layer: the prepared
                # verify polls its F_A rows (armed here once, re-armed by each verify's
                # last reader), and its gate warp's first look precedes its PDL wait.
                # With one buffer per layer, the rows' previous writer is this layer's
                # previous step (a graph launch earlier), so a look that finds them
                # written sees this step's in-projection, whatever the kernels between
                # two KDA layers release early.
                index = decoder.layer_idx
                name, fa = f"kda.fold.projected.{index}", 4 * self.core.dim
                if name not in runtime.buffers:
                    rows = scratch(runtime, name, (64, pitch))
                    rows[:, fa : fa + 128].view(torch.int16).fill_(-1)  # FA_SENT
                self.fold_projected = runtime.buffers[name]
            # KdaHead batches, only with a BF16 state (``prepare_layout`` would drop the
            # head after the MoE tail before this layer, ``use_heads``, and its
            # AttnRes's carve-out were set up for it).
            batches = [b for b in HEAD_BATCHES if b in runtime.batches]
            batches = batches if runtime.bf16_state else []
            multi = batch_table(HEAD_MULTI.get(q, "")) if runtime.bf16_state else {}
            multi = [b for b in runtime.batches if multi.get(b) and b % q == 0]
            if sms > 152 and tp == 16 and pitch == 3216 and (batches or multi):
                index = runtime.kda_heads
                runtime.kda_heads += 1
                counters = runtime.buffers.setdefault(
                    "kda.head.counters",
                    torch.zeros((128, 2), dtype=torch.int32, device=runtime.device),
                )
                # one F_A buffer per layer: KdaHeadWide polls its rows (sentinel)
                fa_out = scratch(runtime, "kda.head.fa", (128, 64, pitch - 4 * 768))
                fa_out = fa_out[index]
                # KdaHead batches whose fused o-projection polls the output: one
                # buffer per KdaHead layer, armed here once (kda.head.POLLED_OUT)
                polled = [
                    b for b in polled_out_batches(tp, weight.device) if b in batches
                ]
                if polled:
                    name = "kda.head.polled"
                    if name not in runtime.buffers:
                        out = torch.empty((128, 8, 12288 // tp), **bf16(runtime))
                        out.view(torch.int16).fill_(-1)  # OUT_SENT
                        runtime.buffers[name] = out
                    self.head_index = index
                # B1 x q verify rows (q = 8: the KdaHead batch): KdaHeadVerify
                verify = [(b, runtime.q) for b in batches if b == runtime.q == 8]
                verify += [(b, q) for b in multi]  # KdaHeadMulti
                self.head = PreparedKdaHead(
                    weight, self.core, batches, runtime.kda_side, counters[index],
                    fa_out, polled=polled, verify=verify,
                )  # fmt: skip
                self.before_carveout = HEAD_BEFORE_CARVEOUT
        else:
            heads = attention.num_local_heads
            shape = (runtime.capacity, heads, 576)
            query = scratch(runtime, "mla.query", shape, torch.uint8)
            shapes = (Shape(m, runtime.q) for m in runtime.wide)
            self.mla_wide = frozenset(
                s.rows for s in shapes if mla_choose(s, runtime.context)
            )
            policy = dict(max_context=runtime.max_context, core=runtime.mla_core)
            policy.update(queries=runtime.queries)
            if runtime.dcp > 1:
                parity = runtime.mla_layers % 2
                policy.update(dcp=dcp_box(runtime, heads), parity=parity)
                self.polled_out = polled_buffers(runtime, heads, sms, tp, parity)
                policy.update(polled_out=tuple(self.polled_out))
                # the first MLA layer builds the step's shared work lists (mla.wlist)
                policy.update(wlist_build=runtime.mla_layers == 0)
            runtime.mla_layers += 1
            pitches = {b: pitch for b in (*runtime.tuned, *self.mla_wide)}
            self.core = PreparedMla(attention, pitches, query, **policy)
            # The step switch's long-context core (SWITCH): tokenspeed + DcpValue's
            # LSE fold for those batches, sharing this layer's DCP buffers and parity.
            long = [b for b in SWITCH if b in runtime.batches]
            if runtime.dcp > 1 and sms > 152 and runtime.mla_core == "auto" and long:
                policy.update(core="tokenspeed", queries=(1,))
                pitches = {b: pitch for b in long}
                self.core_long = PreparedMla(attention, pitches, query, **policy)
        parity = decoder.layer_idx % 2
        front: tuple | None = None
        if decoder.is_moe_layer:
            front = small_batch.front_views[parity], small_batch.hint_views[parity]
            front += (small_batch.batches,)
        head_polled = self.head.polled if self.head is not None else ()
        rows = runtime.capacity  # the largest decode batch
        projected = scratch(runtime, kind + ".projected", (rows, pitch))
        local = scratch(runtime, "attention.local", (rows, 12288 // tp))
        # Wide batches the fused o-projection admits (``oproj.prepared.choose``); the
        # others run the stock o-projection (seam S3, ``run_wide``).
        shapes = (Shape(m, runtime.q) for m in runtime.wide)
        self.oproj_wide = frozenset(
            s.rows for s in shapes if oproj_choose(s, runtime.context)
        )
        self.output = PreparedOproj(
            decoder,
            mailbox,
            (*runtime.tuned, *sorted(self.oproj_wide)),
            kind == "mla",
            front,
            twoshot,
            self.polled_out,
            head_polled,
            x_buffer=local,
            sharded=(runtime.sharded, runtime.shard_box),
        )
        self.prefetch, self.layer = prefetch, decoder
        # Wide batches whose o-projection runs stock: the AttnRes after it runs unfused,
        # on ClusterAttnRes or (``runtime.stock_attn_res``) the stock op.
        unfused = set(runtime.wide) - self.oproj_wide
        clustered = unfused - runtime.stock_attn_res
        self.post = PreparedAttnRes.for_post(decoder) if clustered else None
        stock = unfused & runtime.stock_attn_res
        self.post_stock = StockAttnRes.for_post(decoder) if stock else None
        if kind == "mla" and set(runtime.wide) - self.mla_wide:  # S2's output gate
            warm_gate(pitch, 12288 // tp, weight.device)
        self.storage = {b: (projected[:b], local[:b]) for b in range(1, rows + 1)}
        # Verify batches (token rows) whose MLA runs as single-query rows on the q = 1
        # path (mla.rows.VERIFY_ROWS; SpecializedDecode.verify_rows writes their
        # per-row metadata once per step).
        self.rows: tuple = ()
        if kind == "mla" and runtime.dcp > 1 and sms > 152 and runtime.q > 1:
            q = runtime.q
            self.rows = tuple(
                b for b in VERIFY_ROWS if b in runtime.batches and b % q == 0
            )
        # Batches whose MlaQuery polls x (q = 1): their own rows, armed here once and
        # re-armed by the core every layer (``mla.query.MlaQuery.poll``).
        self.polled: dict = {}
        if kind == "mla" and runtime.dcp > 1 and sms > 152:
            for b in POLLED_X.get(runtime.dcp, {}):
                if b in runtime.batches:
                    rows = scratch(runtime, f"mla.projected.polled{b}", (b, pitch))
                    PreparedMla.x_columns(rows).view(torch.int16).fill_(-1)
                    self.polled[b] = rows
            for core in (self.core, self.core_long):  # mla.core_ts re-arms them
                if core is not None:
                    core.x_rows = dict(self.polled)

    def head_polled(self, b):
        """(this KdaHead layer's polled output rows, the previous KdaHead layer's
        buffer to re-arm) for a polled batch, else None (kda.head.POLLED_OUT: the
        re-arm needs at least two KdaHead layers)."""
        count = self.runtime.kda_heads
        if b not in self.head.polled or count < 2:
            return None
        buffers = self.runtime.buffers["kda.head.polled"]
        return buffers[self.head_index, :b], buffers[(self.head_index - 1) % count]

    def prepare_layout(self, caches):
        if self.kind == "kda":
            self.core.prepare_caches(caches)
            if self.fold64 is not None:
                self.fold64.prepare_caches(caches)
                if self.fold64.tokens < 8:  # x's tensor maps, before any capture
                    for hidden in self.runtime.hidden:
                        self.fold64.x_map(hidden[: 8 * self.fold64.tokens])
            if self.head is not None and caches[1].dtype == torch.bfloat16:
                self.head.prepare_caches(caches)
            else:
                self.head = None
        else:
            cores = [c for c in (self.core, self.core_long) if c is not None]
            if self.runtime.mla_layers % 2:  # the re-arm needs alternating parities
                self.polled_out.clear()
                for core in cores:
                    core.polled_out = ()
                    core.alternating = False
            for core in cores:
                core.prepare_cache(caches[0])

    def run(
        self, x, out, prefix, blocks, caches, metadata, slot_mapping, q=1, path=0,
        positions=None,
    ):  # fmt: skip
        """``path``: the step switch's body (1: ``core_long`` for its batches)."""
        b = x.shape[0]
        if b in self.runtime.wide:
            return self.run_wide(
                x, out, prefix, blocks, caches, metadata, slot_mapping, q, positions
            )
        projected, local = self.storage[b]
        core = self.core
        verify = None  # verify rows (mla.rows): (tables, local lengths, totals)
        if q > 1 and b in self.rows:
            verify, q = self.runtime.rows_meta[b], 1
        if path and q == 1 and (b, 1) in getattr(self.core_long, "kernels", {}):
            core = self.core_long
        if q == 1 and b in self.polled and b in getattr(core, "polled_x", ()):
            projected = self.polled[b]
        head = self.head
        verify_head = head is not None and q > 1
        if verify_head and (b, q) in getattr(head, "verify_kernels", {}):
            # B1 x q verify: in-projection + verify recurrence in one kernel
            slots = metadata.spec_state_indices_tensor
            accepted = metadata.num_accepted_tokens
            rearm, polled = None, self.head_polled(b)
            if polled is not None:
                local, rearm = polled
            head.run_verify(
                x, local, caches=caches, slots=slots, accepted=accepted, rearm=rearm
            )
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            self.output.run(
                local, prefix, blocks, out, after_head=True, polled=polled is not None
            )
            self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
            return out
        if head is not None and q == 1 and b in head.kernels:
            slots = metadata.non_spec_state_indices_tensor
            rearm, polled = None, self.head_polled(b)
            if polled is not None:
                local, rearm = polled
            head.run(x, local, caches=caches, slots=slots, rearm=rearm)
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            self.output.run(
                local, prefix, blocks, out, after_head=True, polled=polled is not None
            )
            self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
            return out
        fold = self.fold64 is not None and q > 1 and b == 8 * q
        prepared = self.prepared[:b] if fold else None
        if fold:  # the projection also prepares the verify operands
            projected = self.fold_projected[:b]
            slots = metadata.spec_state_indices_tensor
            accepted = metadata.num_accepted_tokens
            self.fold64.run(
                x, projected, prepared, caches=caches, slots=slots, accepted=accepted
            )  # fmt: skip
        else:
            projected = self.projection.run(x, projected)
        if self.kind == "kda":
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            if q > 1:  # uniform q-token verify rows
                slots = metadata.spec_state_indices_tensor
                accepted = metadata.num_accepted_tokens
                operands = dict(prepared=prepared) if fold else {}
                self.core.run(
                    projected, local, caches=caches, slots=slots, accepted=accepted,
                    **operands,
                )  # fmt: skip
            else:
                slots = metadata.non_spec_state_indices_tensor
                self.core.run(projected, local, caches=caches, slots=slots)
        else:
            decode = metadata.decode
            tables, lengths = decode.block_table, decode.seq_lens
            totals = decode.dcp_tot_seq_lens
            if verify is not None:
                tables, lengths, totals = verify
            # Request metadata shorter than b: the core runs the prepared batch b
            # with the missing rows as empty requests.
            rows = min(b, q * min(lengths.numel(), tables.shape[0]))
            polled = self.polled_out.get(b) if q == 1 else None
            if polled is not None:  # DcpValue writes every row; the o-proj polls x
                local = polled[0]
            elif rows < b:
                local[rows:].zero_()
            slots = slot_mapping[self.layer.self_attn.layer_name]
            mla = dict(cache=caches[0], block_table=tables, lengths=lengths)
            # DCP: lengths are the local shard's, totals the full sequences'.
            mla.update(totals=totals)
            if polled is not None:
                mla.update(rearm=polled[1])
            core.run(projected, local, slots=slots, q=q, **mla)
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            if polled is not None:
                self.output.run(local, prefix, blocks, out, polled=True)
                self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
                return out
        self.output.run(local, prefix, blocks, out)
        self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
        return out

    def run_wide(
        self, x, out, prefix, blocks, caches, metadata, slot_mapping, q, positions
    ):  # fmt: skip
        """A verify step above the tuned providers: the generic in-projection, then the
        KDA core (KdaSplit, any requests) or the stock MLA attention from the
        in-projection rows (seam S2: W_UK, cache insert, DCP exchanges, core, W_UV;
        the output gate as stock applies it), then the stock o-projection with its
        all-reduce and the unfused AttnRes (seam S3)."""
        b = x.shape[0]
        projected, local = self.storage[b]
        if b in self.projection.fallback.kernels:
            projected = self.projection.run(x, projected)
        else:  # no generic tactic at these rows: cuBLAS, as the stock layer
            with blas_library("cublaslt" if self.lt_rows.get(b) else None):
                torch.mm(x, self.weight.t(), out=projected)
        attention = self.layer.self_attn
        if self.kind == "kda" and b in self.kda_stock:
            self.stock_kda(attention, projected, local)
        elif self.kind == "kda":
            slots = metadata.spec_state_indices_tensor
            accepted = metadata.num_accepted_tokens
            self.core.run(
                projected, local, caches=caches, slots=slots, accepted=accepted
            )
        elif b in self.mla_wide:  # PreparedMla into the plain local rows
            decode = metadata.decode
            tables, lengths = decode.block_table, decode.seq_lens
            rows = min(b, q * min(lengths.numel(), tables.shape[0]))
            if rows < b:  # requests the metadata lacks run as empty ones
                local[rows:].zero_()
            mla = dict(cache=caches[0], block_table=tables, lengths=lengths)
            mla.update(totals=decode.dcp_tot_seq_lens)
            slots = slot_mapping[attention.layer_name]
            self.core.run(projected, local, slots=slots, q=q, **mla)
        else:
            width = local.shape[1]  # the output gate's columns
            qkv, gate = projected.split([projected.shape[1] - width, width], dim=-1)
            core = attention._apply_q_lora_attention(positions, x, qkv)
            # The stock output gate's math (one kernel, FP32, rounded once) into the
            # attention.local rows, which a fused o-projection maps. Not its
            # torch.compile'd function: this strided gate view would recompile it
            # inside the capture.
            gate_sigmoid_mul(core, gate, local)
        if b in self.oproj_wide:
            return self.output.run(local, prefix, blocks, out)[0]
        # The stock o-projection, reduced over the TP ranks: GEMM-AR where the stock
        # layer runs it (VLLM_KIMI_K3_GEMM_AR, from 128 rows), else GEMM + all-reduce.
        fused = attention.gemm_rs_ar
        if fused is not None and fused.should_run(local):
            delta = fused.apply(local, attention.o_proj)
        else:
            delta = attention.o_proj(local)[0]
        if b in self.runtime.stock_attn_res:  # the stock op; returns its own rows
            return self.post_stock.run(delta, prefix, blocks)
        # At a block-write layer the post AttnRes has no delta: it takes the new
        # prefix from ``delta`` (after its PDL wait) and stores it, so no copy here.
        return self.post.run(delta, prefix, blocks, out)

    @staticmethod
    def stock_kda(attention, projected, local):
        """Seam S1: the stock KDA core (conv, gates, recurrence, gated norm) from the
        in-projection rows, as the stock layer splits them, into ``local``."""
        b, width = projected.shape[0], attention.local_projection_size
        heads, dim = attention.local_num_heads, attention.head_dim
        mixed, gates, f_a, beta = projected[:, : 4 * width + dim + heads].split(
            [3 * width, width, dim, heads], dim=-1
        )
        g1 = attention.f_b_proj(f_a)[0].view(1, b, heads, dim)
        attention._forward(
            mixed_qkv=mixed,
            g1=g1,
            g2=gates.unflatten(-1, (heads, dim)),
            beta=beta.unsqueeze(0),
            core_attn_out=local.view(1, b, heads, dim),
        )


def polled_buffers(runtime, heads, sms, tp, parity):
    """MLA decode batches whose fused o-projection polls DcpValue's output
    (mla.dcp.POLLED_OUTPUT): batch -> (this parity's buffer, the other parity's),
    rows of two [64, heads * 128] buffers armed with PAIR_SENT, shared by every MLA
    layer. Separate from MlaQuery's polled in-projection rows (``self.polled``)."""
    device = runtime.device
    batches = [
        b
        for b in runtime.tuned
        if polled_output(heads, runtime.dcp, b, sms)
        and fused_schedule(tp, b, True, device)
        and shared_x(tp, b, device)
    ]
    if not batches:
        return {}
    name = "mla.x.polled"
    if name not in runtime.buffers:
        x = torch.empty((2, 64, heads * 128), device=device, dtype=torch.bfloat16)
        x.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
        runtime.buffers[name] = x
    x = runtime.buffers[name]
    return {b: (x[parity, :b], x[1 - parity, :b]) for b in batches}


def dcp_box(runtime, heads):
    """The model's DCP mailboxes (one set for every MLA layer)."""
    if runtime.dcp_box is None:
        from ..._vllm.distributed import get_dcp_group

        group = get_dcp_group()
        assert group.world_size == runtime.dcp == runtime.tp
        from ..._vllm.config import get_current_vllm_config

        args = group.device_group, runtime.device, group.world_size, group.rank_in_group
        parallel = get_current_vllm_config().parallel_config
        interleave = parallel.cp_kv_cache_interleave_size
        runtime.dcp_box = DcpBox(*args, heads, interleave=interleave)
    return runtime.dcp_box


def pack_front(moe, rank, tp):
    """[this rank's router rows (padded to 16); this rank's latent down shard]."""
    router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
    rows, shard = 896 // tp, 3584 // tp
    router = router[rank * rows : (rank + 1) * rows]
    down = down[rank * shard : (rank + 1) * shard]
    return torch.cat((router, router.new_zeros((-rows % 16, 7168)), down)).detach()
