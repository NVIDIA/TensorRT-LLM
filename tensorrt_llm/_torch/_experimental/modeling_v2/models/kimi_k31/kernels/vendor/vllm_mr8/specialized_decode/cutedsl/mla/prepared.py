# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified by NVIDIA for TensorRT-LLM (2026): import statements only.
"""Prepared MLA core: the packed input projection feeds the query kernel, then either
tokenspeed_mla_decode (core "tokenspeed") or FlashInfer's trtllm-gen MLA decode (core
"trtllm"; FP8 Q and KV as tokenspeed, FP32 softmax and accumulation) into the BF16
latent, and MlaValue (W_UV / gate). Under DCP (``dcp``: a DcpBox) the tokenspeed core
runs over the local KV shard for every DCP rank's heads between the query's q
publication and ``DcpValue``'s merge (see mla.dcp). Spec verify: ``q`` query tokens
per request (rows grouped by request, causal among them)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...._vllm import envs

from ...shapes import Choice, Path
from ..common.host import fake, layout, prepare_shared_carveout, stream, to_cute
from ..common.policy import batch_table
from .core import RUBIN_CORE, compile_dcp_core, core_args, split_workspace
from .core_p2 import compile_p2, ds_partials
from .core_q import (
    POLLED_VERIFY,
    VALID_VERIFY,
    compile_polled_q,
    polled_partials_q,
    push_table,
    verify_table,
    wide_persistent,
)
from .core_umma import (
    POLLED_SPLIT,
    PREWAIT,
    RUBIN_UMMA,
    compile_dcp_core_umma,
    polled_split_workspace,
)
from .dcp import (
    DV_LSE_UNIFORM,
    FORK_BALANCE,
    FORK_CLUSTER,
    FORK_REPOLL,
    FORK_VALID,
    FORK_VALID_DELAY,
    GB300_SMS,
    LONG_SPLIT,
    LPC_LONG,
    LPC_ROWS,
    LPC_SLOTS,
    LSE_FOLD,
    POLLED_TS,
    RUBIN_TS,
    VERIFY_FOLD,
    compile_fork,
    compile_value,
    dcp_splits,
    fill_splits,
    fork_partials,
    fork_table,
    local_max,
    long_split,
    lse_fold_splits,
    rubin_decode,
    split_partials,
    split_partials_q,
    split_workspace_ts,
    stock_decode,
    verify_fold_splits,
    verify_splits,
)
from .query import (
    GATED_GATHER,
    GATED_VERIFY,
    POLLED_X,
    QUERY_X_UNIFORM,
    VALID_PUBLISH,
    MlaQuery,
)
from .value import MlaValue
from .wlist import WLIST, shared_list

# tokenspeed (SM-filling splits) streams long contexts fast; trtllm-gen wins from B16.
# Rubin-class devices, DCP at q = 1, core "auto": batches whose core is tokenspeed's
# generic kernel with DcpValue's LSE fold (dcp.LSE_FOLD; the polled fork mla.core_ts
# for dcp.POLLED_TS) instead of the custom cores (DcpCoreUmma, DcpCore), which run a
# fixed split count and slow down past ~800 local tokens. 4 ranks, us per MLA layer at
# local 550 / 1024 / 4096 tokens: custom 26.41 / 30.25 / 51.10, tokenspeed + fold
# 28.16 / 28.64 / 32.01.
TOKENSPEED_DCP = batch_table("8:1")
# Rubin-class devices, DCP at q = 1, core "auto": batches whose decode step is captured
# twice under a CUDA-graph IF/ELSE node (common.graph_switch, runtime.decode): the
# custom cores while every request's local length is <= the value (one 160-token pass
# per split, 5 splits), else tokenspeed + fold. Off: TOKENSPEED_DCP serves every
# length at B8.
SWITCH = batch_table("")
# "tokenspeed_rubin": under DCP on sm_107 devices, tokenspeed's Rubin kernel at q = 1
# for the batches of dcp.RUBIN_TS (else as "tokenspeed").
TRTLLM_CORE_BATCH = 16
# Rubin DCP spec verify rows (above 64, split 1, persistent core: wide_persistent)
# whose core pushes its BF16 rows and LSEs into the owners' DcpBox itself, DcpValue
# only re-arming the q mailbox and merging (core_ts ``push``). Every CTA pushes its last
# block after its loops; the CTAs left with one block push up to the value's first
# blocks of the two-block CTAs each while the second wave runs. Same values (bitwise).
# At 16 ranks DcpValue's push is ~75% of its time and the fabric idles during the core.
# 16 ranks (6 heads, q8, local 2000), us per MLA layer, DcpValue push -> core push
# (value 0 / 1 / 8): M192 74.21 -> 67.51 / 65.31 / -; M256 86.34 -> 85.21 / 84.58 /
# 100.33 (idle CTAs that take many blocks become the tail). Bitwise on every rank.
CORE_PUSH = batch_table("192:1,256:1")
CORES = ("tokenspeed", "trtllm", "tokenspeed_rubin")
_FLASHINFER: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


@cache
def compile_core(
    batch,
    heads,
    pitch,
    specs,
    device,
    dcp=0,
    sms=148,
    q=1,
    fold=0,
    poll_x=False,
    polled_out=False,
    polled_split=False,
    lse_fold=None,
    q_polled=False,
    gated_verify=False,
    q_valid=False,
    q_valid_box=False,
    lse_uniform=False,
    x_uniform=False,
    cap=64,
    pushed=False,
    lpc=None,
):
    """CuTe kernels (query, value) of one batch of rows (``q`` per request); under DCP
    (``dcp``: its size) the query publishes and the value is DcpValue (``fold``: the
    Rubin DcpCore's split count, merged by DcpValue; ``poll_x``: the query polls x;
    ``polled_out``: the o-projection polls DcpValue's output, mla.dcp.POLLED_OUTPUT;
    ``polled_split``: DcpValue polls DcpCoreUmma's split workspace, POLLED_SPLIT;
    ``lse_fold``: the output scale of tokenspeed's split partials, LSE-folded by
    DcpValue, LSE_FOLD; ``gated_verify``: spec verify rows gate the gathered q poll,
    query.GATED_VERIFY; ``lse_uniform``: dcp.DV_LSE_UNIFORM; ``x_uniform``:
    query.QUERY_X_UNIFORM; ``cap``: the rows of the DCP mailboxes, DcpBox.cap;
    ``pushed``: the core pushes the rows, DcpValue merges, CORE_PUSH)."""
    qnorm, kvnorm, qb, uk, uv, qinv, kinv, kv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    query = fake((batch, heads, 576), (heads * 576, 576, 1), torch.uint8)
    projected = fake((batch, 2112), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    lengths = fake((batch // q,), (1,), torch.int32, 4)
    gate = fake((batch, heads * 128), (pitch, 1))
    output = fake((batch, heads * 128), (heads * 128, 1))
    latent = fake((batch, heads, 512))
    f32, s = cutlass.Float32, make_fake_stream()
    cp_async = heads == 12 and batch == 1
    gated = sms > GB300_SMS and q == 1 and batch in GATED_GATHER.get(dcp, {})
    if sms > GB300_SMS and q > 1 and gated_verify:
        gated = gated_verify  # 1: gate, 2: gate + batched re-polls
    # The Rubin DcpCore (fold > 0) polls the gathered q rows itself.
    mla_query = MlaQuery(
        heads,
        batch,
        q_b_cp_async=cp_async,
        dcp=dcp,
        gate=gated,
        gather=not q_polled and (not fold or lse_fold is not None),
        sms=sms,
        poll_x=poll_x,
        valid=q_valid,
        x_uniform=x_uniform,
    )
    q_args: tuple = (projected, qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv)
    q_args += (f32(1e-6), query)
    if dcp:  # this parity's q mailbox, its multicast address, rank
        mailbox = fake((cap, dcp * heads, 576), None, torch.uint8)
        q_args += (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    with torch.accelerator.device_index(device):
        kernels = [cute.compile(mla_query, *q_args, s)]
        if batch > 1:
            kernels = [prepare_shared_carveout(kernels[0], device, 100)]
        if dcp:
            uv_spec = specs[4]
            kernels.append(
                compile_value(
                    heads,
                    dcp,
                    batch,
                    cap,
                    uv_spec,
                    pitch,
                    device,
                    sms,
                    q,
                    fold,
                    polled_out,
                    polled_split,
                    lse_fold is not None,
                    1.0 if lse_fold is None else lse_fold,
                    q_valid_box,
                    lse_uniform,
                    pushed,
                    lpc=lpc,
                )
            )
        else:
            v_args = (latent, uv, gate, lengths, output, s)
            value = MlaValue(heads, batch, q)
            kernels.append(cute.compile(value, *v_args))
    return tuple(kernels)


def core_for(batch, core="auto", q=1):
    """``core`` unless "auto": trtllm-gen from B16, else tokenspeed (spec verify,
    q > 1: tokenspeed, whose SM-filling splits stream long contexts far faster)."""
    if core == "auto":
        core = "trtllm" if batch >= TRTLLM_CORE_BATCH and q == 1 else "tokenspeed"
    if core not in CORES:
        raise ValueError(f"Kimi K3 MLA core {core!r}: expected auto or one of {CORES}")
    return core


def pad_requests(batch, block_table, lengths, totals=None):
    """Request metadata for ``batch`` rows: PIECEWISE graphs pad the tokens but not
    the requests, so missing rows become empty requests (length 0 on page 0)."""
    rows = min(lengths.shape[0], block_table.shape[0])
    if rows >= batch:
        return block_table, lengths, totals
    pad = torch.nn.functional.pad
    block_table = pad(block_table[:rows], (0, 0, 0, batch - rows))
    lengths = pad(lengths[:rows], (0, batch - rows))
    totals = None if totals is None else pad(totals[:rows], (0, batch - rows))
    return block_table, lengths, totals


def trtllm_decode(query, cache, block_table, lengths, max_seq, scales, out):
    """FlashInfer trtllm-gen MLA decode of FP8 ``query`` [B, q, H, 576] (causal among
    the q rows) over the FP8 paged cache into BF16 ``out`` [B, q, H, 512], PDL chained;
    zeroed workspace and multi-CTA counter (self-resetting) owned here, one per (device,
    heads)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    device, heads = query.device, query.shape[2]
    if (device, heads) not in _FLASHINFER:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        count = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, heads, sms)
        workspace = torch.zeros(128 << 20, dtype=torch.int8, device=device)
        counter = torch.zeros(count, dtype=torch.uint8, device=device)
        _FLASHINFER[device, heads] = workspace, counter
    workspace, counter = _FLASHINFER[device, heads]
    softmax, output_scale = scales
    trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=cache.view(torch.float8_e4m3fn).unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_table,
        seq_lens=lengths,
        max_seq_len=max_seq,
        out=out,
        bmm1_scale=softmax,
        bmm2_scale=output_scale,
        enable_pdl=True,
        backend="trtllm-gen",
        multi_ctas_kv_counter_buffer=counter,
    )


class PreparedMla:
    """``max_context`` (the deployment's longest sequence) bounds the split count;
    ``core`` forces "tokenspeed" or "trtllm". ``dcp``: the model's DcpBox, with this
    layer's mailbox ``parity``. ``queries``: the query lengths q (spec verify: 1 + the
    draft length) prepared for every batch of rows divisible by q. ``polled_out``:
    decode batches whose DcpValue output is polled by the o-projection
    (dcp.POLLED_OUTPUT); ``run`` then takes the other parity's output (``rearm``)."""

    def __init__(
        self,
        attention,
        pitches,
        query_scratch,
        *,
        max_context=None,
        core="auto",
        dcp=None,
        parity=0,
        queries=(1,),
        polled_out=(),
        wlist_build=False,
    ):
        self.heads = attention.W_UK_T.shape[0]
        norms = attention.q_a_layernorm.weight, attention.kv_a_layernorm.weight
        self.weights: tuple = (*norms, attention.q_b_proj.weight, attention.W_UK_T)
        self.weights += (attention.W_UV, attention._q_scale_inv, attention._k_scale_inv)
        self.device, self.pitches = attention.q_b_proj.weight.device, pitches
        k_scale = attention._k_scale_float
        softmax = attention.scale * attention._q_scale_float * k_scale
        self.eps = cutlass.Float32(attention.rms_norm_eps)
        self.stock_scales = softmax, k_scale
        self.views = tuple(
            to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in self.weights
        )
        self.query_scratch, self.kernels = query_scratch, {}
        shapes = [(b, q) for b in pitches for q in queries if b % q == 0]
        self.cores = {(b, q): core_for(b, core, q) for b, q in shapes}
        # Under DCP on Rubin-class devices, "auto" takes the custom cores (DcpCoreUmma,
        # DcpCore) where their policies list the batch, except TOKENSPEED_DCP batches;
        # "tokenspeed" keeps tokenspeed for every batch. The custom cores run a fixed
        # split count per request, so each split's passes grow with the local length:
        # they win at <= 800 local tokens (one 160-token pass per split at 5 splits)
        # and lose beyond (B8, 1 GPU: local 896 9.3 vs 8.1 us; 8192 61 vs 14 us at
        # 128-token passes).
        self.custom_cores = core == "auto"
        # The split count only depends on this bound; seq_lens bound the work.
        self.max_seq = max_context or 1 << 17
        self.dcp, self.parity = dcp, parity
        self.polled_out = tuple(polled_out) if dcp is not None else ()
        if dcp is None:  # tokenspeed_rubin: DCP decode only
            for shape, name in self.cores.items():
                self.cores[shape] = name.replace("tokenspeed_rubin", "tokenspeed")
        else:
            arch = torch.cuda.get_device_capability(self.device)
            rubin = core == "tokenspeed_rubin" and arch == (10, 7)
            self.cores = {
                (b, q): "tokenspeed_rubin"
                if rubin and q == 1 and b in RUBIN_TS
                else "tokenspeed"
                for b, q in shapes
            }
            self.max_seq = local_max(self.max_seq, dcp.size, dcp.interleave)
        # tokenspeed_rubin batches: their (KV splits, cluster shape), dcp.RUBIN_TS
        self.rubin = {
            b: RUBIN_TS[b]
            for (b, q), c in self.cores.items()
            if c == "tokenspeed_rubin"
        }
        shape, bf16 = (max(pitches, default=0), self.heads, 512), torch.bfloat16
        self.latent = torch.empty(shape, dtype=bf16, device=self.device)
        # Rubin-class devices: the DCP core at q = 1 of RUBIN_CORE batches is DcpCore.
        self.rubin_cores: dict = {}
        # Batches (q = 1) whose MlaQuery polls x and whose DcpCoreUmma re-arms it: the
        # caller keeps their in-projection rows in a dedicated buffer armed with
        # query.X_SENT (``x_columns``).
        self.polled_x: set = set()
        # Whether consecutive MLA layers alternate parity (an even MLA layer count), as
        # the per-parity polled split workspaces need (``polled_split``).
        self.alternating = True
        # Batches whose DcpValue polls DcpCoreUmma's split workspace of this parity
        # (POLLED_SPLIT): batch -> the other parity's (part, stats), armed by the core.
        self.polled_split: dict = {}
        # Batches whose tokenspeed core skips its reducer and DcpValue LSE-folds its
        # split partials (dcp.LSE_FOLD): batch -> the exact split count.
        self.lse_fold: dict = {}
        # LSE_FOLD batches run by mla.core_ts's polled fork (dcp.POLLED_TS): batch ->
        # its address table. ``x_rows``: the polled in-projection rows (the caller's
        # buffer, armed with query.X_SENT) of those batches.
        self.ts_fork: dict = {}
        # POLLED_TS batches whose fork splits the heads evenly (dcp.FORK_BALANCE),
        # whose q pollers re-poll their late chunks together (dcp.FORK_REPOLL).
        self.ts_balanced: set = set()
        self.ts_repoll: set = set()
        self.ts_cluster: set = set()  # (FORK_CLUSTER)
        # ... whose fork polls q with validity-reporting copies (dcp.FORK_VALID; their
        # MlaQuery writes one 16 B store per q chunk).
        self.ts_valid: dict = {}  # batch -> FORK_VALID mode
        self.x_rows: dict = {}
        # Verify shapes (batch, q > 1) whose packed tokenspeed core skips its reducer
        # and DcpValue LSE-folds the split partials (dcp.VERIFY_FOLD): -> split count.
        self.verify_fold: dict = {}
        # VERIFY_FOLD shapes whose core polls its q tiles (mla.core_q.POLLED_VERIFY;
        # MlaQuery does not gather): (batch, q) -> its address table.
        self.verify_poll: dict = {}
        # ... whose core reads q with validity-reporting copies (core_q.VALID_VERIFY;
        # their MlaQuery writes one 16 B store per q chunk).
        self.verify_valid: dict = {}  # (batch, q) -> VALID_VERIFY mode
        # Wide verify shapes (above the box's rows): (batch, q) -> the polled core's
        # split count and whether it runs tokenspeed's persistent scheduler.
        self.verify_split: dict = {}
        self.verify_persistent: dict = {}
        self.verify_push: dict = {}  # ... whose core pushes its rows (CORE_PUSH)
        self.verify_flags: dict = {}  # ... and its per-block ready flags
        # ... whose persistent core runs the per-step work list (mla.wlist.WLIST),
        # shared by every layer; the first MLA layer (``wlist_build``) builds it.
        self.verify_wlist: dict = {}
        # ... whose work list runs on DS-T2P (mla.core_p2, VLLM_KIMI_K3_DS_T2P) instead
        # of core_q's polled core, or (VLLM_KIMI_K3_DS_LPC) DS-T2P on its LPC partition:
        # (batch, q) -> (long split, packed rows, slots, LPC)
        self.verify_p2: dict = {}
        # ... LPC shapes: (batch, q) -> (groups, slots), DcpValue's ``lpc``
        self.verify_lpc: dict = {}
        self.lpc_dummy = None  # (ds_partials' unused work-list words under LPC)
        self.wlist_build = wlist_build

    def lpc_on(self, batch, q, kv, sms):
        """DS-T2P runs LPC (dcp.lpc_table) at this verify shape: VLLM_KIMI_K3_DS_T2P
        and VLLM_KIMI_K3_DS_LPC, q 8, LPC_ROWS, 64-key pages, Rubin."""
        on = envs.VLLM_KIMI_K3_DS_T2P and envs.VLLM_KIMI_K3_DS_LPC
        on = on and q == 8 and sms > GB300_SMS and batch in LPC_ROWS
        return bool(on and kv.shape[1] == 64)

    def lpc_prepare(self, batch, q, packed, sms, kv):
        """Record an LPC shape: DS-T2P's (long, packed, slots, lpc) and DcpValue's
        (groups, slots); groups as the core's (CTA pairs // items per request)."""
        tiles = packed // 128
        items = 2 * (tiles // 2) + tiles % 2
        groups = (sms // 2) // items
        slots = tiles * (batch // q) * LPC_SLOTS  # (P2Core's compile key only)
        self.verify_p2[batch, q] = LPC_LONG, packed, slots, True
        self.verify_lpc[batch, q] = groups, LPC_SLOTS
        if self.lpc_dummy is None:
            self.lpc_dummy = torch.zeros(1, dtype=torch.int32, device=kv.device)

    def work_list(self, batch, q, heads, tiles, split, sms, kv):
        """The shared per-step work list (mla.wlist) of a WLIST verify shape launched
        at its LONG_SPLIT split, or None."""
        long = long_split(batch, heads, split, sms)
        if batch not in WLIST or long is None:
            return None
        return shared_list(batch // q, tiles, split, long, sms // 2, kv.device)

    def prepare_cache(self, kv):
        specs = tuple(layout(w) for w in (*self.weights, kv.view(torch.uint8)))
        for batch, q in self.cores:
            device = self.device.index
            dcp = self.dcp.size if self.dcp is not None else 0
            sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            pitch = self.pitches[batch]
            policy = None
            pages_ok = kv.shape[1] % 64 == 0 or kv.shape[1] == 32  # DcpCore's pages
            rubin = dcp and sms > GB300_SMS and q == 1 and pages_ok
            rubin = rubin and self.custom_cores and batch not in TOKENSPEED_DCP
            umma = rubin and batch in RUBIN_UMMA and dcp * self.heads <= 128
            if umma:
                policy = (RUBIN_UMMA[batch],)
            elif rubin and batch in RUBIN_CORE:
                policy = RUBIN_CORE[batch]
            fold = policy[0] if policy else 0
            lse_fold = None  # tokenspeed's split partials LSE-folded by DcpValue
            tokenspeed = self.cores[batch, q] == "tokenspeed" and batch in LSE_FOLD
            q_polled = False
            if dcp and sms > GB300_SMS and q == 1 and not policy and tokenspeed:
                heads = dcp * self.heads
                fold = lse_fold_splits(batch, heads, self.max_seq, self.device)
                self.lse_fold[batch], lse_fold = fold, self.stock_scales[1]
                q_polled = batch in POLLED_TS and batch in self.x_rows
            wide = bool(dcp) and sms > GB300_SMS and q > 1 and batch > self.dcp.cap
            dcp_box = self.box_for(batch)
            if wide:  # the polled packed core on the wide box, split by tokenspeed
                heads = dcp * self.heads
                split, packed = verify_fold_splits(
                    batch, q, heads, self.max_seq, self.device
                )
                lpc = self.lpc_on(batch, q, kv, sms)
                if lpc:  # DS-T2P's LPC partition: LPC_SLOTS slots, no work list
                    split = LPC_SLOTS
                if split > 1:  # DcpValue folds the split partials
                    fold, lse_fold = split, self.stock_scales[1]
                    self.verify_fold[batch, q] = split
                self.verify_split[batch, q] = split
                persistent = wide_persistent(batch // q, packed // 128, split, sms)
                tiles = packed // 128
                work = None
                if not lpc:
                    work = self.work_list(batch, q, heads, tiles, split, sms, kv)
                persistent = persistent or work is not None or lpc
                self.verify_persistent[batch, q] = persistent
                self.verify_valid[batch, q] = 2
                qv = dcp_box.mailbox(True)[self.parity]
                push = None
                if batch in CORE_PUSH and split == 1 and persistent:
                    push = CORE_PUSH[batch]
                self.verify_push[batch, q] = push
                if push is not None:  # q mailbox, owners' boxes, block flags
                    blocks = 2 * (packed // 128) * (batch // q)
                    flags = torch.zeros(blocks, dtype=torch.int32, device=kv.device)
                    self.verify_flags[batch, q] = flags
                    table = push_table(
                        qv, batch // q, dcp_box, self.parity, kv.device, flags
                    )
                else:
                    table = verify_table(qv, batch // q, kv.device)
                if work is not None and push is None:
                    table = work.table(table)
                    self.verify_wlist[batch, q] = work
                    # DS-T2P: q 8 rows 128-256 on its 64-key pages; the same q
                    # mailbox (validity TMA), work list, split workspace and partitions
                    p2 = envs.VLLM_KIMI_K3_DS_T2P and q == 8 and 128 <= batch <= 256
                    if p2 and kv.shape[1] == 64:
                        long = long_split(batch, heads, split, sms)
                        self.verify_p2[batch, q] = long, packed, work.slots, False
                if lpc:
                    self.lpc_prepare(batch, q, packed, sms, kv)
                self.verify_poll[batch, q] = table
            verify = dcp and sms > GB300_SMS and q > 1 and not wide
            verify = verify and batch in VERIFY_FOLD
            verify = verify and self.cores[batch, q] == "tokenspeed"
            if verify:  # packed verify core: DcpValue folds its split partials
                heads = dcp * self.heads
                split, packed = verify_fold_splits(
                    batch, q, heads, self.max_seq, self.device
                )
                lpc = self.lpc_on(batch, q, kv, sms) and batch in POLLED_VERIFY
                lpc = lpc and VALID_VERIFY.get(batch) == 2
                if lpc:  # DS-T2P's LPC partition (as the wide rows)
                    split = LPC_SLOTS
                gathered = split > 1 and batch not in POLLED_VERIFY
                if gathered and long_split(batch, heads, split, sms):
                    # (as draft.attention: the gathered core has no long partition, so
                    # DcpValue would fold the wrong partials at short contexts)
                    raise ValueError(
                        f"verify rows {batch} (q {q}): LONG_SPLIT "
                        f"{LONG_SPLIT[batch, heads]} fires on the gathered verify "
                        "core, which has no long partition: add the rows to "
                        "POLLED_VERIFY or drop the rule"
                    )
                if split > 1:
                    fold, lse_fold = split, self.stock_scales[1]
                    self.verify_fold[batch, q] = split
                    if batch in POLLED_VERIFY:  # the core polls this parity's q
                        if batch in VALID_VERIFY:
                            self.verify_valid[batch, q] = VALID_VERIFY[batch]
                        box = self.dcp.mailbox(self.valid_box(batch, q))[self.parity]
                        table = verify_table(box, batch // q, kv.device)
                        tiles = packed // 128
                        work = None
                        if not lpc:
                            work = self.work_list(
                                batch, q, heads, tiles, split, sms, kv
                            )
                        if work is not None and VALID_VERIFY.get(batch) == 2:
                            table = work.table(table)
                            self.verify_wlist[batch, q] = work
                            self.verify_persistent[batch, q] = True
                        if lpc:
                            self.verify_persistent[batch, q] = True
                            self.lpc_prepare(batch, q, packed, sms, kv)
                        self.verify_poll[batch, q] = table
            poll = bool(umma) and batch in POLLED_X.get(dcp, {})
            if q_polled:  # the fork polls q and re-arms x: MlaQuery polls x
                poll = batch in POLLED_X.get(dcp, {})
                if poll:
                    if batch in FORK_VALID:  # the validity-polled mailbox (DcpBox.qv)
                        self.ts_valid[batch] = FORK_VALID[batch]
                    query_box = self.dcp.mailbox(self.valid_box(batch, q))[self.parity]
                    table = fork_table(query_box, self.x_rows[batch], batch, kv.device)
                    self.ts_fork[batch] = table
                    if batch in FORK_BALANCE:
                        self.ts_balanced.add(batch)
                    if batch in FORK_REPOLL:
                        self.ts_repoll.add(batch)
                    if batch in FORK_CLUSTER:
                        self.ts_cluster.add(batch)
                else:
                    q_polled = False
            if poll:
                self.polled_x.add(batch)
            polled_out = q == 1 and batch in self.polled_out
            psplit = bool(umma) and batch in POLLED_SPLIT and self.alternating
            args: tuple = batch, self.heads, pitch, specs, device, dcp, sms, q, fold
            verify = q > 1 and dcp and sms > GB300_SMS
            gate_v = GATED_VERIFY.get(dcp, {}).get(batch, 0) if verify else 0
            q_polled = q_polled or (batch, q) in self.verify_poll
            args += (poll, polled_out, psplit, lse_fold, q_polled, gate_v)
            q_valid = q == 1 and batch in self.ts_valid
            q_valid |= bool(dcp) and sms > GB300_SMS and batch in VALID_PUBLISH
            q_valid |= (batch, q) in self.verify_valid
            args += (q_valid, bool(dcp) and self.valid_box(batch, q))
            rubin_rows = q == 1 and sms > GB300_SMS
            args += (rubin_rows and batch in DV_LSE_UNIFORM,)
            args += (rubin_rows and batch in QUERY_X_UNIFORM,)
            cap = dcp_box.cap if dcp_box is not None else 64
            query = to_cute(self.query_scratch[:batch])
            latent = to_cute(self.latent[:batch])
            pushed = self.verify_push.get((batch, q)) is not None
            lpc_value = self.verify_lpc.get((batch, q))
            kernels = compile_core(*args, cap=cap, pushed=pushed, lpc=lpc_value)
            self.kernels[batch, q] = (*kernels, query, latent)
            if policy:
                compile_rubin = compile_dcp_core_umma if umma else compile_dcp_core
                extra: dict = dict(x_pitch=pitch) if poll else {}
                if psplit:
                    extra.update(polled_split=True)
                if umma and batch in PREWAIT:
                    extra.update(prewait=True)
                self.rubin_cores[batch] = compile_rubin(
                    dcp * self.heads,
                    batch,
                    policy[0] if umma else policy,
                    self.dcp.cap,
                    specs[-1],
                    device,
                    **extra,
                )
            if self.dcp is not None:
                self.prepare_dcp(batch, fold, psplit, q)
            # compile before graph capture (DcpCore is compiled; its q poll needs a
            # published q, so it is not run here)
            if q == 1 and batch in self.ts_fork:  # compiled only (its q poll needs a
                # publish)
                heads = self.dcp.size * self.heads
                options: tuple = (batch in self.ts_balanced, batch in self.ts_repoll)
                options += (batch in self.ts_cluster, self.ts_valid.get(batch, 0))
                options += (FORK_VALID_DELAY.get(batch, 0),)
                compile_fork(heads, kv.shape[1], fold, batch, device, *options)
                split_workspace_ts(self.device, heads, batch, fold)
            elif (batch, q) in self.verify_poll:  # compiled only (as the fork)
                heads, reqs = self.dcp.size * self.heads, batch // q
                where = (self.dcp.size, self.dcp.rank, self.dcp.interleave)
                valid = self.verify_valid.get((batch, q), 0)
                split = self.verify_split.get((batch, q), fold)
                persistent = self.verify_persistent.get((batch, q), False)
                if (batch, q) in self.verify_p2:  # DS-T2P: its own core (no JIT later)
                    long, _, slots, lpc = self.verify_p2[batch, q]
                    raw = kv.view(torch.uint8)
                    spec = tuple(raw.shape), tuple(raw.stride())
                    pairs = torch.cuda.get_device_properties(kv.device)
                    pairs = pairs.multi_processor_count // 2
                    compile_p2(
                        heads, q, kv.shape[1], split, where, device, True, q,
                        tuple(long), spec, reqs, slots, pairs, lpc,
                    )  # fmt: skip
                else:
                    compile_polled_q(
                        heads, kv.shape[1], split, reqs, q, where, device, valid=valid,
                        persistent=persistent, push=self.verify_push.get((batch, q)),
                    )  # fmt: skip
                rows = verify_fold_splits(batch, q, heads, self.max_seq, self.device)[1]
                split_workspace_ts(self.device, rows, reqs, fold)
            elif kv.is_cuda and not policy:
                table = torch.zeros(batch, 1, dtype=torch.int32, device=kv.device)
                ones = torch.full((batch,), q, dtype=torch.int32, device=kv.device)
                self.stock_core(kv, table, ones, batch, q, ones)
        return self

    def box_for(self, batch):
        """The DCP mailboxes of a batch: the model's box, or above its rows the box set
        of the largest prepared batch (DcpBox.widened)."""
        if self.dcp is None or batch <= self.dcp.cap:
            return self.dcp
        return self.dcp.widened(max(self.pitches))

    def prepare_dcp(self, batch, fold=0, psplit=False, q=1):
        """(MlaQuery's publish arguments, DcpValue's mailbox arguments, parity and
        rank) of one (batch, q) shape; ``fold``: DcpCore's split partials replace the
        rows; ``psplit``: DcpValue polls them (this parity's armed workspace). Keyed by
        (batch, q): a verify shape of B1 x q 8 has the rows of the q = 1 batch 8, whose
        DcpValue may LSE-fold tokenspeed's split partials (``lse_fold``) instead."""
        box, p = self.box_for(batch), self.parity
        valid = self.valid_box(batch, q)  # DcpBox.qv: validity-polled readers
        mailbox = to_cute(box.mailbox(valid)[p])
        ids = (cutlass.Int32(p), cutlass.Int32(box.rank))
        rows = to_cute(box.rows[:batch]), to_cute(box.rows_lse[:batch], 4)
        if q == 1 and batch in self.lse_fold:  # tokenspeed's split partials, LSEs
            heads = box.size * self.heads
            _, part, stats = split_workspace_ts(self.device, heads, batch, fold)
            rows = to_cute(part), to_cute(stats, 4)
        elif (batch, q) in self.verify_fold:  # the packed verify core's partials
            heads = box.size * self.heads
            _, packed = verify_fold_splits(batch, q, heads, self.max_seq, self.device)
            _, part, stats = split_workspace_ts(self.device, packed, batch // q, fold)
            rows = to_cute(part), to_cute(stats, 4)
        elif fold and psplit:
            heads = box.size * self.heads
            part, stats = polled_split_workspace(self.device, heads, fold, p)
            rows = to_cute(part[:batch]), to_cute(stats[:batch], 4)
            other = polled_split_workspace(self.device, heads, fold, 1 - p)
            self.polled_split[batch] = (
                to_cute(other[0][:batch]),
                to_cute(other[1][:batch], 4),
            )
        elif fold:
            part, stats = split_workspace(self.device, box.size * self.heads, fold)
            rows = to_cute(part[:batch]), to_cute(stats[:batch], 4)
        boxes = to_cute(box.partial[p]), to_cute(box.lse[p], 4)
        peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
        self.dcp_args = getattr(self, "dcp_args", {})
        publish = (mailbox, box.mc(p, valid), ids[1])
        self.dcp_args[batch, q] = publish, (rows, boxes, peers), ids

    def valid_box(self, batch, q):
        """Whether the (batch, q) shape's q mailbox is DcpBox.qv: its readers poll
        with validity-reporting copies (FORK_VALID's fork), so its writer keeps
        mla.qvalid's contract and its DcpValue re-arms the mla.qvalid.ARMED pattern.
        A shape never changes box (each box keeps one arm pattern)."""
        return (q == 1 and batch in self.ts_valid) or (batch, q) in self.verify_valid

    @staticmethod
    def x_columns(projected):
        """The polled x columns of in-projection rows (armed with query.X_SENT)."""
        return projected[:, :1536]

    def stock_core(
        self, cache, block_table, lengths, batch, q=1, totals=None, projected=None
    ):
        """tokenspeed_mla_decode or trtllm-gen into the BF16 latent (DCP: tokenspeed
        for every rank's heads over the local shard into the DcpBox rows + LSE; q > 1
        masks causally in global positions, ``totals``). tokenspeed: a fixed split
        count per batch (one CTA pair per split filling the SMs); both PDL chained
        between our kernels."""
        box, reqs = self.box_for(batch), batch // q
        if box is not None and q == 1 and batch in self.rubin_cores:
            publish, ((rows, rows_lse), _, _), _ = self.dcp_args[batch, 1]
            table, scales = core_args(block_table[:reqs], *self.stock_scales)
            kv, lens = to_cute(cache.view(torch.uint8)), to_cute(lengths[:reqs], 4)
            x: tuple = ()
            if batch in self.polled_x:  # the core re-arms x after its PDL wait
                x = (to_cute(self.x_columns(projected[:batch])),)
            # the other parity's split workspace (armed by the core)
            x += self.polled_split.get(batch, ())
            self.rubin_cores[batch](
                publish[0],
                kv,
                *table,
                lens,
                rows,
                rows_lse,
                *scales,
                *x,
                stream(self.device),
            )
            return
        query = self.query_scratch[:batch].view(torch.float8_e4m3fn)
        out, lse, cp = self.latent[:batch], None, {}
        if box is not None:
            qbox = box.mailbox(self.valid_box(batch, q))[self.parity, :batch]
            query, out = qbox.view(query.dtype), box.rows[:batch]
            lse = box.rows_lse[:batch].view(reqs, q, -1)
            if q > 1:  # D*H heads: packed (row, head) tiles stream the KV once per tile
                cp = dict(cp=(box.size, box.rank, box.interleave), totals=totals[:reqs])
                cp.update(pack=True)
        query, out = query.view(reqs, q, *query.shape[1:]), out.view(reqs, q, -1, 512)
        args = cache, block_table[:reqs], lengths[:reqs], self.max_seq
        if self.cores[batch, q] == "trtllm":
            trtllm_decode(query, *args, self.stock_scales, out)
            return
        if box is not None and q == 1 and batch in self.ts_fork:
            split, table = self.lse_fold[batch], self.ts_fork[batch]
            options: tuple = (batch in self.ts_balanced, batch in self.ts_repoll)
            options += (batch in self.ts_cluster, self.ts_valid.get(batch, 0))
            options += (FORK_VALID_DELAY.get(batch, 0),)
            fork = (self.stock_scales, out, lse, split, table, *options)
            fork_partials(query, *args[:3], *fork)
            return
        if box is not None and q == 1 and batch in self.lse_fold:
            split = self.lse_fold[batch]
            split_partials(query, *args[:3], self.stock_scales, out, lse, split)
            return
        if self.cores[batch, q] == "tokenspeed_rubin":
            policy = self.rubin[batch]
            rubin_decode(query, *args[:3], self.stock_scales, out, lse, policy)
            return
        if box is not None and (batch, q) in self.verify_poll:
            table = self.verify_poll[batch, q]
            split = self.verify_split.get((batch, q)) or self.verify_fold[batch, q]
            where = (box.size, box.rank, box.interleave)
            tables = block_table[:reqs], lengths[:reqs], totals[:reqs]
            scales = self.stock_scales
            call = query, cache, *tables, scales, out, lse, split, where, table
            valid = self.verify_valid.get((batch, q), 0)
            persistent = self.verify_persistent.get((batch, q), False)
            push = self.verify_push.get((batch, q))
            work = self.verify_wlist.get((batch, q))
            if work is not None and self.wlist_build:  # this step's list, all layers
                work.build(lengths[:reqs])
            if (batch, q) in self.verify_p2:  # DS-T2P on the same workspace
                long, packed, slots, lpc = self.verify_p2[batch, q]
                workspace = split_workspace_ts(self.device, packed, reqs, split)[0]
                words = self.lpc_dummy if lpc else work.buffer
                ds_partials(
                    query, cache, *tables, scales, out, lse, workspace, split, where,
                    table, x_rows=q, causal=True, long=long, wlist=(words, slots),
                    lpc=lpc,
                )  # fmt: skip
                return
            polled_partials_q(
                *call, valid=valid, persistent=persistent, push=push, wlist=bool(work)
            )
            return
        if box is not None and (batch, q) in self.verify_fold:
            split = self.verify_fold[batch, q]
            where = (box.size, box.rank, box.interleave)
            tables = block_table[:reqs], lengths[:reqs], totals[:reqs]
            scales = self.stock_scales
            split_partials_q(query, cache, *tables, scales, out, lse, split, where)
            return
        split, blocks = fill_splits(batch, self.device), None
        if box is not None and q == 1:
            split, blocks = dcp_splits(batch, self.device)
        elif box is not None:
            split, blocks = verify_splits(batch, self.device)
        stock_decode(
            query, *args, self.stock_scales, out, lse, split, blocks=blocks, **cp
        )

    def run(
        self,
        projected,
        out,
        *,
        cache,
        slots,
        block_table,
        lengths,
        totals=None,
        q=1,
        rearm=None,
    ):
        """``lengths``/``block_table``: this rank's KV (the local shard under DCP);
        ``totals``: under DCP, the requests' full lengths. ``q`` rows per request (spec
        verify; ``slots`` per row). Rows of ``projected`` beyond the requests run as
        empty requests (zero output). ``rearm``: polled batches, the other parity's
        output buffer (DcpValue re-arms it)."""
        batch, s = projected.shape[0], stream(self.device)
        reqs = batch // q
        block_table, lengths, totals = pad_requests(reqs, block_table, lengths, totals)
        kv = to_cute(cache.view(torch.uint8))
        mq, v, query, latent = self.kernels[batch, q]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = self.views
        q_slots, x = to_cute(slots[:batch], 8), to_cute(projected[:, :2112])
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, qinv, kinv, self.eps, query)
        out_view, gate = to_cute(out), to_cute(projected[:, 2112:])
        lens = to_cute(lengths[:reqs], 4)
        if self.dcp is not None:  # DcpValue zeroes padded requests (global lengths)
            publish, ((rows, rows_lse), boxes, peers), ids = self.dcp_args[batch, q]
            mq(*q_args, *publish, s)
            self.stock_core(cache, block_table, lengths, batch, q, totals, projected)
            v_args = (rows, rows_lse, lens, *boxes, *peers, publish[0], uv, gate)
            v_args += (to_cute(totals[:reqs], 4), out_view, *ids)
            if q == 1 and batch in self.polled_out:
                v_args += (to_cute(rearm),)
            v(*v_args, s)
            return out
        mq(*q_args, s)
        self.stock_core(cache, block_table, lengths, batch, q)
        v(latent, uv, gate, lens, out_view, s)
        return out


def choose(shape, ctx):
    """The MLA layer's specialized query / core / DcpValue for a captured decode shape
    (the runtime's family hook), or None: the attention runs the stock MLA attention
    from the in-projection rows (seam S2). A Choice above 64 rows makes the runtime
    prepare ``PreparedMla`` at that shape and run its core into the plain ``local``
    rows. Above 64 rows (Rubin, DCP spec verify): the wide box set (DcpBox.widened),
    MlaQuery publishing q with the validity contract (no gather), the polled packed
    core (validity Q TMA, tokenspeed's split; core_q.wide_persistent)
    and DcpValue's spread merge."""
    if shape.rows <= 64:
        return Choice(Path.GENERIC)
    if ctx.arch != "rubin" or ctx.dcp <= 1 or shape.q == 1:
        return None
    return Choice(Path.GENERIC, tactic="wide", note="validq core_q spread")
