# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode attention of a batch of requests: balanced page-split partials, then a combine.

The built-in decode step of this target with two or more sequences runs flashinfer's trtllm-gen
generation FMHA on the generation preprocess's outputs: with the FP8 KV cache an E4M3 Q and
fp32 device scales (``fp8_context_fmha``), or a BF16 Q and host scales on a ``generation_only``
call. This entry takes either and is two launches:

* ``multi_request_split_kv_kernel`` (CuTe DSL, tcgen05): one CTA per SM per pair of KV heads;
  the pages every request attends are concatenated and each CTA takes an equal contiguous share,
  cut at request boundaries into segments; one partial per segment (normalized O in FP16 and
  its log2-sum-exp) at slot ``CTA + request``;
* ``_combine_kernel`` (Triton): per (request, token, KV head's group of 8 query heads) the
  request's slot range, recomputed from ``seq_lens`` exactly as the kernel cuts the pages,
  merged into the BF16 output (times ``bmm2_scale``): the maximum log2-sum-exp over all its slots
  first, then the weighted sum of the slots' O in chunks of slots, so the merge has no serial
  chain over the slots. The chunk and the program's warps (``COMBINE_SHAPES``) follow the batch
  size: the few requests of a small batch span many slots each.

Two builds by batch size (``REQUEST_BUILDS``): batches of up to 64 requests run the 64-request
build, 65..128 requests (the decode steps of a generation worker of ``max_batch_size`` above 64)
the 128-request build. The builds differ only in the capacity of the partition's tables and in
the combine's request block; both compute the same page split, so a batch both could take gets
the same partials and the same output from either.

Semantics (the call ``trtllm_batch_decode_with_kv_cache`` makes for a uniform ``q_len_per_req``
of ``M = rows / requests`` rows per request): with ``L_b = seq_lens[b]`` (the step's rows
included) row ``t`` of request ``b`` attends keys ``j`` with ``j <= L_b - M + t`` (and ``j > L_b
- M + t - window`` when ``window > 0``), score ``q . k_j * bmm1_scale``, output ``softmax . v *
bmm2_scale`` (``bmm1_scale[0]`` and ``bmm2_scale[0]`` when they are device tensors, read by the
kernels after the dependency wait); K/V are the cache's E4M3 bytes, read through the request's
``block_tables[b, 0]`` (K) and ``block_tables[b, 1]`` (V) page ids into ``kv_pool [blocks, G,
32, 64]``.

Numerics (not bit-identical to the built-in FMHA): those of ``full_attention_split_kv`` (the
8-row one-sequence kernel this target serves to a single request): an E4M3 Q enters as is (the
same values the built-in kernel multiplies), a BF16 Q as two E4M3 terms ``Q_hi + Q_lo``
(BF16-class), K/V are the cache's E4M3 values, P is rounded to E4M3 after a scale-up by 2^4
against a running maximum raised only when it grows by more than 2^4 (log2 units), the
denominator sums the unrounded P, S and P.V accumulate in fp32, every partial is stored as FP16
normalized O with an fp32 log2-sum-exp, and the combine merges in fp32 in a fixed slot order.
The page split depends only on ``seq_lens`` and the grid, so the result is deterministic run to
run. ``nq=1`` (a BF16 Q as one E4M3 rounding) is a timing variant the target does not launch.

Ownership and progress (per launch): the partition (requests' attended page ranges, their
prefix, the CTA's range) is computed by one warp of each CTA from ``seq_lens`` and the grid;
a CTA reads pages only inside its range and writes only its own slots (rows ``token * hq +
query head`` of its pair's heads); no CTA waits on another; the combine reads a slot only after
its ``griddepcontrol.wait`` (or stream order), i.e. after the split grid completed.

PDL memory-order record (``enable_pdl``): before ``griddepcontrol.wait`` the split kernel reads
only ``seq_lens`` (the partition), the TMA descriptors and its own shared memory; the block
table (a view of the step's block-offset metadata, which no kernel of the step writes), every
page, Q and a device ``bmm1_scale`` are read after the wait (the loader warp, the softmax
warps), every partial store follows the softmax warps' wait. ``seq_lens`` is the step's
metadata, final before the layer's preprocess runs, which reads it too and completes before
this kernel's predecessor wait returns; the preprocess does not write it. The split kernel
triggers its dependents when the softmax warps reach the epilogue of their last segment; the
combine waits before its first read (a device ``bmm2_scale`` included).
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    OpWrapper,
    true_fp32_matmul,
)

#: Head width, tokens per page, query heads per KV head.
HEAD_DIM = 64
PAGE = 32
Q_PER_KV = 8
#: Rows per request the row map is written for (one accepted token and up to 7 drafts) and
#: requests per launch by build: the 64-request build (512 rows) up to 64 requests, the
#: 128-request build (1024 rows) above.
MAX_ROWS = 8
REQUEST_BUILDS = (64, 128)
MAX_REQUESTS = REQUEST_BUILDS[-1]
#: KV heads of one rank per tensor-parallel size the kernel is built for.
RANK_KV_HEADS = {8: 2, 4: 4}
#: Fewest pages per keyed CTA: a batch of N attended pages keys min(grid, ceil(N / 4)) CTAs.
MIN_PAGES = 4
#: The build the target launches: O rescale in 16-column chunks (spill-free), two Q terms.
RESCALE_COLS = 16
NQ = 2
#: Pages (8 KB of K and V each) a CTA's ring holds in flight, by the number of E4M3 terms Q enters
#: as (``ring_stages``). At deep contexts the split kernel streams the cache and the ring depth
#: sets its rate. sm107: 16 pages for either Q; ``configure`` starts from these depths.
_SM107_RING_STAGES = {1: 16, 2: 16}
#: The ring depths the builds take (``ring_stages``), as ``configure`` last set them.
RING_STAGES = dict(_SM107_RING_STAGES)

#: dieexp KF651 (knob ``MUSESPARK_OPEN_DX_MR_KF651=1``, default off): the served build (full layers, E4M3 Q, device
#: scales, 2 KV heads per rank, the 64-request build) runs the Kernel Factory fa-block-vr2 candidate 651fa0ca's split
#: kernel (``multi_request_split_kv_kernel_kf651``, ring KF651_STAGES) and its combine (``_kf_post_kernel``: the
#: kernel's tile-aligned partition, read from the slot map CTA 0 publishes). Order-only vs the served pair.
KF651 = __import__("os").environ.get("MUSESPARK_OPEN_DX_MR_KF651", "") == "1"
KF651_STAGES = 24
KF651_REGS_SOFTMAX = 208
#: dieexp KF651 v3 (KF fa-block-vr5 62963e84 port): softmax/other register split, SEG_PAD virtual tiles.
KF651_REGS_OTHER = 80
KF651_SEG_PAD = 2
#: The KF651 kernel's CTA 0 publishes the request -> slot map from ONE warp (lanes 0..31): above 32 requests the
#: combine would read unpublished slots (illegal address at 48, w256 q11), so larger batches take the served pair.
KF651_MAX_REQUESTS = 32
#: dieexp KF651 v2 (``MUSESPARK_OPEN_DX_MR_KF651_V2=1`` with KF651; KF fa-block-vr4 48f1a399): ``_kf_post_kernel``
#: recomputes
#: the request's slot range from ``seq_lens`` (immutable in the step; the same integers the split kernel's CTA 0
#: publishes)
#: before its dependency wait, triggers its dependent (o_proj) before the wait, and runs 2 warps below 8 requests. Same
#: arithmetic and order (bitwise to KF651). No W_o prefetch here: the banked direct-hop producer prefetches its weight.
KF651_V2 = __import__("os").environ.get("MUSESPARK_OPEN_DX_MR_KF651_V2", "") == "1"


def kf651_build(kv_heads: int, window: int, q_fp8: bool, device_scale: bool, requests: int) -> bool:
    """Whether a build runs the KF651 pair (``KF651`` on and the served full-attention build)."""
    return bool(
        KF651
        and kv_heads == 2
        and window == 0
        and q_fp8
        and device_scale
        and int(requests) <= KF651_MAX_REQUESTS
    )


def configure(*, sm107_stages: Optional[int] = None) -> None:
    """Set sm107's ring depth for the served one-term (E4M3) Q build (``split_kv_deep_rings``).

    ``lever_settings.apply`` sets it from the model's plan before any install. ``None`` keeps
    the table's depths; ``n`` sets the one-term build's ring to ``n`` pages (the two-term build
    keeps 16). 20 launches on sm107 and is bitwise to
    16 (outputs and partials at 5-8 requests x 256k / 512k / 1M contexts); per layer at 8 / 5
    requests: 512k -4.8 / -2.7 %, 1M -6.9 / -4.8 %, 256k ~0. Each compiled build is keyed on the
    depth it takes (``_compile``), so a build compiled at another depth is never reused.

    Raises:
        ValueError: for a depth that is not a positive multiple of 4 (pages per 128-key tile).
    """
    global RING_STAGES
    table = dict(_SM107_RING_STAGES)
    if sm107_stages is not None:
        if sm107_stages <= 0 or sm107_stages % 4:
            raise ValueError(
                f"multi_request_split_kv: the ring depth is a positive multiple of 4, got "
                f"{sm107_stages}"
            )
        table[1] = int(sm107_stages)
    RING_STAGES = table


def ring_stages(device_index: int, q_terms: int) -> int:
    """The ring depth of a build whose Q enters as ``q_terms`` E4M3 terms on the device.

    Raises:
        ValueError: on any device but an sm107 one.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    if (major, minor) != (10, 7):
        raise ValueError(
            f"multi_request_split_kv: no ring depth for sm{10 * major + minor} (supported: [107])"
        )
    return RING_STAGES[q_terms]


def default_num_work(kv_heads: int, sm_count: int) -> int:
    """CTAs per pair of KV heads: one per SM over the grid ``num_work x pairs``."""
    if kv_heads not in RANK_KV_HEADS.values():
        raise ValueError(f"multi_request_split_kv has no build for {kv_heads} KV heads per rank")
    return sm_count // (kv_heads // 2)


def request_build(requests: int) -> int:
    """The build (``REQUEST_BUILDS``) that serves a batch of ``requests`` requests (the largest
    for a batch no build takes, which the call then refuses)."""
    return next((build for build in REQUEST_BUILDS if requests <= build), REQUEST_BUILDS[-1])


def slots_needed(num_work: int, requests: int) -> int:
    """Partial slots a launch of ``num_work`` CTAs per pair over ``requests`` requests writes
    at most: slot ``CTA + request`` < ``num_work + requests``."""
    return num_work + requests


def workspace_for(
    device, kv_heads: int = 4, num_work: Optional[int] = None, requests: int = REQUEST_BUILDS[0]
):
    """Uninitialized partial buffers for any batch of up to ``request_build(requests)``
    requests: FP16 O ``[slots, 8 hq, 64]`` and fp32 log2-sum-exp ``[slots, 8 hq]``."""
    device = torch.device(device)
    if num_work is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        num_work = default_num_work(kv_heads, sms)
    slots = slots_needed(num_work, request_build(requests))
    hq = kv_heads * Q_PER_KV
    part_o = torch.empty(slots, MAX_ROWS * hq, HEAD_DIM, dtype=torch.float16, device=device)
    part_lse = torch.empty(slots, MAX_ROWS * hq, dtype=torch.float32, device=device)
    return part_o, part_lse


def attended_pages(seq_len: int, rows: int, window: int) -> Tuple[int, int]:
    """``[first, end)`` pages a request of context ``seq_len`` attends (host mirror of the
    kernel's partition)."""
    first = max(seq_len - rows - window + 1, 0) // PAGE if window > 0 else 0
    return first, -(-seq_len // PAGE)


def request_slots(seq_lens, rows: int, window: int, num_work: int):
    """Per request the ``(first, last)`` slots its partials occupy (host mirror of the
    partition and of ``_combine_kernel``)."""
    n = [e - f for f, e in (attended_pages(int(s), rows, window) for s in seq_lens)]
    total = sum(n)
    weff = min(num_work, -(-total // MIN_PAGES))
    out = []
    start = 0
    for b, nb in enumerate(n):
        c_first = ((start + 1) * weff - 1) // total
        c_last = ((start + nb) * weff - 1) // total
        out.append((c_first + b, c_last + b))
        start += nb
    return out


@functools.lru_cache(maxsize=32)
def _compiled_kernel(
    device_index: int,
    enable_pdl: bool,
    kv_heads: int,
    window: int,
    nq: int = NQ,
    rescale_cols: int = RESCALE_COLS,
    q_fp8: bool = False,
    device_scale: bool = False,
    stages: Optional[int] = None,
    max_requests: int = REQUEST_BUILDS[0],
    kf651: bool = False,
):
    """The compiled split kernel of one build (its ring depth from ``ring_stages`` unless
    ``stages`` is given; ``kf651``: the KF651 kernel)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    if kf651:
        from . import multi_request_split_kv_kernel_kf651 as mk

        assert mk.SEG_PAD == KF651_SEG_PAD
    else:
        from . import multi_request_split_kv_kernel as mk

    assert (mk.MAX_ROWS, mk.REQUEST_BUILDS) == (MAX_ROWS, REQUEST_BUILDS)
    if stages is None:
        stages = ring_stages(device_index, 1 if q_fp8 else nq)
    kernel = mk.MultiRequestSplitKvKernel(
        enable_pdl=enable_pdl,
        kv_heads=kv_heads,
        nq=nq,
        window=window,
        stages=stages,
        rescale_cols=rescale_cols,
        min_pages=MIN_PAGES,
        q_fp8=q_fp8,
        device_scale=device_scale,
        max_requests=max_requests,
        **({"regs_softmax": KF651_REGS_SOFTMAX, "regs_other": KF651_REGS_OTHER} if kf651 else {}),
    )
    dtypes = (
        cutlass.Float8E4M3FN if q_fp8 else cutlass.BFloat16,  # q
        cutlass.Float8E4M3FN,  # pool
        cutlass.Int32,  # block table
        cutlass.Int32,  # seq_lens
        cutlass.Float16,  # partial O
        cutlass.Float32,  # partial log2-sum-exp
    )
    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            kernel,
            *[make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for dtype in dtypes],
            *[cutlass.Int32(1)] * 9,
            cutlass.Float32(1.0),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
            stream,
            options="--enable-tvm-ffi",
        )


def _host_scale_log2(bmm1_scale: float) -> float:
    """``bmm1_scale * log2 e`` rounded as the device-scale builds compute it (one fp32
    product), so a host scale and a device scale of the same value give the same bits."""
    return float(
        torch.tensor(float(bmm1_scale), dtype=torch.float32)
        * torch.tensor(math.log2(math.e), dtype=torch.float32)
    )


#: Builds compiled so far (``compiled_key``); a capture may only launch one of these.
_compiled_keys: set = set()


def compiled_key(
    device_index: int,
    enable_pdl: bool,
    kv_heads: int,
    window: int,
    nq: int = NQ,
    rescale_cols: int = RESCALE_COLS,
    q_fp8: bool = False,
    device_scale: bool = False,
    requests: int = REQUEST_BUILDS[0],
) -> tuple:
    """The build a call of ``requests`` requests launches; an E4M3 Q is one term whatever
    ``nq`` says."""
    return (int(device_index), bool(enable_pdl), int(kv_heads), int(window),
            1 if q_fp8 else int(nq), int(rescale_cols), bool(q_fp8),
            bool(device_scale), request_build(int(requests)),
            kf651_build(int(kv_heads), int(window), bool(q_fp8), bool(device_scale), int(requests)))  # fmt: skip


def _compile(key: tuple):
    # The ring depth (``ring_stages``, which ``configure`` sets on sm107) is passed explicitly so
    # that it is part of the compiled build's cache key.
    return _compiled_kernel(*key[:4], nq=key[4], rescale_cols=key[5], q_fp8=key[6],
                            device_scale=key[7],
                            stages=KF651_STAGES if key[9] else ring_stages(key[0], key[4]),
                            max_requests=key[8], kf651=key[9])  # fmt: skip


def precompile(
    device_index: int,
    enable_pdl: bool,
    kv_heads: int,
    window: int,
    q_fp8: bool = False,
    device_scale: bool = False,
    requests: int = REQUEST_BUILDS[0],
) -> tuple:
    """Compile one build of the split kernel (outside any CUDA-graph capture); returns its key.

    The Triton combine compiles on its first launch, which the decode graphs' warmup forwards
    make before any capture."""
    key = compiled_key(
        device_index,
        enable_pdl,
        kv_heads,
        window,
        q_fp8=q_fp8,
        device_scale=device_scale,
        requests=requests,
    )
    _compile(key)
    _compiled_keys.add(key)
    return key


#: The most slots one request can span (a power of two at least ``num_work``, the grid's CTAs
#: per pair), and the combine's shape by batch size: for batches of at most ``requests`` requests
#: (the first entry that fits), the slots one chunk reads and the program's warps. The requests
#: of a batch of up to 4 share the ``num_work`` CTAs (76 at TEP4) and average ``num_work / B``
#: slots each, so one wide chunk replaces a chain of chunks; a larger batch has fewer slots per
#: request and more programs, where the narrow chunk is faster.
COMBINE_MAX_SLOTS = 256
COMBINE_SHAPES = ((4, 64, 8), (64, 16, 4), (128, 16, 4))


@triton.jit(do_not_specialize=["num_requests", "num_work", "rows", "out_stride", "out_scale"])
def _combine_kernel(
    part_o_ptr,
    part_lse_ptr,
    seq_lens_ptr,
    out_ptr,
    out_stride,
    out_scale,
    out_scale_ptr,
    num_requests,
    num_work,
    rows,
    HQ: tl.constexpr,
    GH: tl.constexpr,
    D: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    MIN_P: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    DEVICE_SCALE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    """One program per (request, token, group of GH query heads): the request's slots,
    recomputed from ``seq_lens`` as the split kernel cuts the pages, merged in two passes (the
    maximum log2-sum-exp over all slots, then the weighted sum of O in chunks of CHUNK slots).
    The first chunk's loads are issued with the log2-sum-exp loads, before the maximum is
    known."""
    b = tl.program_id(0)
    tok = tl.program_id(1)
    hg = tl.program_id(2)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
    ib = tl.arange(0, BLOCK_B)
    okb = ib < num_requests
    L = tl.load(seq_lens_ptr + ib, mask=okb, other=0)
    if WINDOW > 0:
        first = tl.maximum(L - rows - WINDOW + 1, 0) // P
    else:
        first = L * 0
    n = tl.where(okb, (L + P - 1) // P - first, 0)
    total = tl.sum(n, 0)
    start = tl.sum(tl.where(ib < b, n, 0), 0)
    end = start + tl.sum(tl.where(ib == b, n, 0), 0)
    weff = tl.minimum(num_work, (total + MIN_P - 1) // MIN_P)
    c_first = ((start + 1) * weff - 1) // total
    c_last = (end * weff - 1) // total
    n_slots = c_last - c_first + 1
    s0 = (c_first + b).to(tl.int64)
    hh = hg * GH + tl.arange(0, GH)
    dd = tl.arange(0, D)
    R = tok * HQ + hh
    ss = tl.arange(0, MAX_SLOTS)
    lse_all = tl.load(
        part_lse_ptr + (s0 + ss)[:, None] * SLOT_ROWS + R[None, :],
        mask=(ss < n_slots)[:, None],
        other=float("-inf"),
    )
    sc = tl.arange(0, CHUNK)
    ok = sc < n_slots
    lse = tl.load(
        part_lse_ptr + (s0 + sc)[:, None] * SLOT_ROWS + R[None, :],
        mask=ok[:, None],
        other=float("-inf"),
    )
    o = tl.load(
        part_o_ptr
        + (((s0 + sc)[:, None] * SLOT_ROWS + R[None, :]) * D)[:, :, None]
        + dd[None, None, :],
        mask=ok[:, None, None],
        other=0.0,
    ).to(tl.float32)
    m = tl.max(lse_all, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w_all = tl.where(lse_all == float("-inf"), 0.0, tl.math.exp2(lse_all - m[None, :]))
    den = tl.sum(w_all, 0)
    w = tl.where(lse == float("-inf"), 0.0, tl.math.exp2(lse - m[None, :]))
    acc = tl.sum(o * w[:, :, None], 0)
    for c0 in range(CHUNK, n_slots, CHUNK):
        sc = c0 + tl.arange(0, CHUNK)
        ok = sc < n_slots
        slot = s0 + sc
        lse = tl.load(
            part_lse_ptr + slot[:, None] * SLOT_ROWS + R[None, :],
            mask=ok[:, None],
            other=float("-inf"),
        )
        w = tl.where(lse == float("-inf"), 0.0, tl.math.exp2(lse - m[None, :]))
        o = tl.load(
            part_o_ptr
            + ((slot[:, None] * SLOT_ROWS + R[None, :]) * D)[:, :, None]
            + dd[None, None, :],
            mask=ok[:, None, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(o * w[:, :, None], 0)
    if DEVICE_SCALE:
        scale = tl.load(out_scale_ptr)
    else:
        scale = out_scale
    res = acc * (scale / den)[:, None]
    row = b * rows + tok
    tl.store(out_ptr + row * out_stride + hh[:, None] * D + dd[None, :], res.to(tl.bfloat16))


@triton.jit(do_not_specialize=["num_requests", "num_work", "rows", "out_stride", "out_scale"])
def _combine_norm_gate_kernel(
    part_o_ptr,
    part_lse_ptr,
    seq_lens_ptr,
    out_ptr,
    out_stride,
    out_scale,
    out_scale_ptr,
    num_requests,
    num_work,
    rows,
    gate_ptr,
    gated_ptr,
    gate_token_stride,
    gate_head_stride,
    eps,
    HQ: tl.constexpr,
    GH: tl.constexpr,
    D: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    MIN_P: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    DEVICE_SCALE: tl.constexpr,
    TRIGGER: tl.constexpr,
):
    """``_combine_kernel`` followed, in the same program, by ``norm_gate``'s
    ``_attention_norm_gate_pdl_kernel`` on each of the program's GH heads (s316 ``NORM_GATE_FOLD``).

    The combine part is ``_combine_kernel``'s body statement for statement (same grid, same shapes,
    same ``num_warps``), so ``out`` gets the same BF16 bits. After a CTA barrier the program reloads
    each of its heads' 64 BF16 outputs as one ``[64]`` tensor and runs the norm-gate's statements on
    it -- the same ``[64]`` layout as the standalone kernel at the same ``num_warps`` (4: the combine
    shape of 5+ requests, the only one the fold takes) -- writing the gated row ``row * HQ + head``
    of ``gated``. ``TRIGGER`` is the norm-gate's (0 exit, 1 after the wait, 2 entry)."""
    b = tl.program_id(0)
    tok = tl.program_id(1)
    hg = tl.program_id(2)
    if TRIGGER == 2:
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    if TRIGGER == 1:
        tl.extra.cuda.gdc_launch_dependents()
    ib = tl.arange(0, BLOCK_B)
    okb = ib < num_requests
    L = tl.load(seq_lens_ptr + ib, mask=okb, other=0)
    if WINDOW > 0:
        first = tl.maximum(L - rows - WINDOW + 1, 0) // P
    else:
        first = L * 0
    n = tl.where(okb, (L + P - 1) // P - first, 0)
    total = tl.sum(n, 0)
    start = tl.sum(tl.where(ib < b, n, 0), 0)
    end = start + tl.sum(tl.where(ib == b, n, 0), 0)
    weff = tl.minimum(num_work, (total + MIN_P - 1) // MIN_P)
    c_first = ((start + 1) * weff - 1) // total
    c_last = (end * weff - 1) // total
    n_slots = c_last - c_first + 1
    s0 = (c_first + b).to(tl.int64)
    hh = hg * GH + tl.arange(0, GH)
    dd = tl.arange(0, D)
    R = tok * HQ + hh
    ss = tl.arange(0, MAX_SLOTS)
    lse_all = tl.load(
        part_lse_ptr + (s0 + ss)[:, None] * SLOT_ROWS + R[None, :],
        mask=(ss < n_slots)[:, None],
        other=float("-inf"),
    )
    sc = tl.arange(0, CHUNK)
    ok = sc < n_slots
    lse = tl.load(
        part_lse_ptr + (s0 + sc)[:, None] * SLOT_ROWS + R[None, :],
        mask=ok[:, None],
        other=float("-inf"),
    )
    o = tl.load(
        part_o_ptr
        + (((s0 + sc)[:, None] * SLOT_ROWS + R[None, :]) * D)[:, :, None]
        + dd[None, None, :],
        mask=ok[:, None, None],
        other=0.0,
    ).to(tl.float32)
    m = tl.max(lse_all, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w_all = tl.where(lse_all == float("-inf"), 0.0, tl.math.exp2(lse_all - m[None, :]))
    den = tl.sum(w_all, 0)
    w = tl.where(lse == float("-inf"), 0.0, tl.math.exp2(lse - m[None, :]))
    acc = tl.sum(o * w[:, :, None], 0)
    for c0 in range(CHUNK, n_slots, CHUNK):
        sc = c0 + tl.arange(0, CHUNK)
        ok = sc < n_slots
        slot = s0 + sc
        lse = tl.load(
            part_lse_ptr + slot[:, None] * SLOT_ROWS + R[None, :],
            mask=ok[:, None],
            other=float("-inf"),
        )
        w = tl.where(lse == float("-inf"), 0.0, tl.math.exp2(lse - m[None, :]))
        o = tl.load(
            part_o_ptr
            + ((slot[:, None] * SLOT_ROWS + R[None, :]) * D)[:, :, None]
            + dd[None, None, :],
            mask=ok[:, None, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(o * w[:, :, None], 0)
    if DEVICE_SCALE:
        scale = tl.load(out_scale_ptr)
    else:
        scale = out_scale
    res = acc * (scale / den)[:, None]
    row = b * rows + tok
    tl.store(out_ptr + row * out_stride + hh[:, None] * D + dd[None, :], res.to(tl.bfloat16))
    # The norm-gate of the program's GH heads on the stored BF16 output (norm_gate's statements).
    tl.debug_barrier()
    offsets = tl.arange(0, D)
    for i in tl.static_range(GH):
        head = hg * GH + i
        x = tl.load(out_ptr + row * out_stride + head * D + offsets).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / D
        normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
        g = tl.load(gate_ptr + row * gate_token_stride + head * gate_head_stride + offsets).to(
            tl.float32
        )
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        result = normed.to(tl.float32) * sigmoid.to(tl.float32)
        tl.store(gated_ptr + (row * HQ + head) * D + offsets, result)


#: dieexp KF651: the slots one chunk of ``_kf_post_kernel`` reads, by batch size (651fa0ca's: 32 from 8 requests,
#: else 64). Unlike the candidate's single chunk, the kernel loops over further chunks (same bits whenever a request's
#: slots fit one chunk, as at equal lengths; correct when one request of an uneven batch spans more).
def _kf_chunk(requests: int) -> int:
    return 32 if requests >= 8 else 64


@triton.jit(do_not_specialize=["num_requests", "rows", "workspace_slots", "num_work"])
def _kf_post_kernel(
    part_o,
    part_lse,
    out_scale,
    gate,
    out,
    seq_lens,
    num_requests,
    rows,
    gate_stride,
    gate_head_stride,
    out_stride,
    workspace_slots,
    eps,
    num_work,
    HQ: tl.constexpr,
    GH: tl.constexpr,
    D: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
    FUSE: tl.constexpr,
    MAXR: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    SEG_PAD: tl.constexpr,
):
    """dieexp KF651: 651fa0ca's fast_post (one program per (request, token, GH adjacent query heads); fp32 merge;
    the BF16 attention row), looped over ALL chunks of CHUNK slots. FUSE: then ``bf16(rmsnorm(att)) *
    bf16(sigmoid(gate))`` into ``out`` [rows, HQ * D]; else the attention row into ``out`` (row stride
    ``out_stride``). PDL: the slot range is derived from seq_lens (immutable) before the wait; every predecessor
    output (partials, gate, scale) is read after it."""
    req = tl.program_id(0)
    tok = tl.program_id(1)
    head = tl.program_id(2) * GH + tl.arange(0, GH)
    # Before the dependency wait: the request's partial-slot range from the batch's lengths (an
    # immutable input), by the formula the KF651 kernel's CTA 0 publishes (full layer: every
    # request attends pages [0, ceil(L / PAGE)), in tiles of PAGES_PER_TILE pages).
    rq = tl.arange(0, MAXR)
    lens = tl.load(seq_lens + rq, mask=rq < num_requests, other=0)
    tiles = tl.where(
        rq < num_requests,
        ((lens + PAGE - 1) // PAGE + PAGES_PER_TILE - 1) // PAGES_PER_TILE + SEG_PAD,
        0,
    )
    total = tl.sum(tiles, 0)
    start = tl.sum(tl.where(rq < req, tiles, 0), 0)
    end = start + tl.sum(tl.where(rq == req, tiles, 0), 0)
    effective = tl.minimum(num_work, total)
    first = ((start + SEG_PAD + 1) * effective - 1) // total
    last = (end * effective - 1) // total
    slot0 = (first + req).to(tl.int64)
    nslots = last - first + 1
    d = tl.arange(0, D)
    row = req * rows + tok
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()

    if FUSE:
        g = tl.load(gate + row * gate_stride + head[:, None] * gate_head_stride + d[None, :]).to(
            tl.float32
        )
    osc = tl.load(out_scale)

    slots = tl.arange(0, CHUNK)
    valid = slots < nslots
    rr = tok * HQ + head
    lse = tl.load(
        part_lse + (slot0 + slots)[:, None] * SLOT_ROWS + rr[None, :],
        mask=valid[:, None],
        other=float("-inf"),
    )
    partial = tl.load(
        part_o
        + ((slot0 + slots)[:, None] * SLOT_ROWS + rr[None, :])[:, :, None] * D
        + d[None, None, :],
        mask=valid[:, None, None],
        other=0.0,
    ).to(tl.float32)
    m = tl.max(lse, 0)
    for c0 in range(CHUNK, nslots, CHUNK):
        lse_c = tl.load(
            part_lse + (slot0 + c0 + slots)[:, None] * SLOT_ROWS + rr[None, :],
            mask=((c0 + slots) < nslots)[:, None],
            other=float("-inf"),
        )
        m = tl.maximum(m, tl.max(lse_c, 0))
    m = tl.where(m == float("-inf"), 0.0, m)
    weight = tl.where(valid[:, None], tl.math.exp2(lse - m[None, :]), 0.0)
    den = tl.sum(weight, 0)
    acc = tl.sum(partial * weight[:, :, None], 0)
    for c0 in range(CHUNK, nslots, CHUNK):
        ok = (c0 + slots) < nslots
        slot = slot0 + c0 + slots
        lse_c = tl.load(
            part_lse + slot[:, None] * SLOT_ROWS + rr[None, :],
            mask=ok[:, None],
            other=float("-inf"),
        )
        w_c = tl.where(ok[:, None], tl.math.exp2(lse_c - m[None, :]), 0.0)
        den += tl.sum(w_c, 0)
        o_c = tl.load(
            part_o + (slot[:, None] * SLOT_ROWS + rr[None, :])[:, :, None] * D + d[None, None, :],
            mask=ok[:, None, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(o_c * w_c[:, :, None], 0)
    xb = (acc * (osc / den)[:, None]).to(tl.bfloat16)
    if FUSE:
        x = xb.to(tl.float32)
        variance = tl.sum(x * x, 1) / D
        normed = (x * tl.rsqrt(variance[:, None] + eps)).to(tl.bfloat16)
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        tl.store(
            out + (row * HQ + head[:, None]) * D + d[None, :],
            normed.to(tl.float32) * sigmoid.to(tl.float32),
        )
    else:
        tl.store(out + row * out_stride + head[:, None] * D + d[None, :], xb)


#: Heads per combine program by batch size (two adjacent heads halve the grid at the batch extremes; one head keeps
#: occupancy in the middle regime). Override from main.
POST_GH = {5: 2, 13: 2}
POST_GH_MIN_WIDE = 24


def _post_gh(requests: int) -> int:
    return POST_GH.get(requests, 2 if requests >= POST_GH_MIN_WIDE else 1)


def kf651_post(pending, gate, eps: float, out: torch.Tensor) -> torch.Tensor:
    """dieexp KF651: the combine of a KF651 launch (``pending``), with the attention norm-gate folded in when ``gate``
    is given (into ``out`` [rows, HQ * 64] BF16) or else the merged attention output into ``out`` (the call's)."""
    (
        part_o,
        part_lse,
        seq_lens,
        out2,
        bmm2_scale,
        device_scale,
        requests,
        num_work,
        per,
        kv_heads,
        hq,
        window,
        enable_pdl,
    ) = pending
    assert device_scale and window == 0
    fuse = gate is not None
    if fuse:
        assert gate.dtype == torch.bfloat16 and gate.stride(-1) == 1
    assert requests <= 64 and window == 0
    gh = _post_gh(requests)
    _kf_post_kernel[(requests, per, hq // gh)](
        part_o,
        part_lse,
        bmm2_scale,
        gate if fuse else part_lse,
        out,
        seq_lens,
        requests,
        per,
        gate.stride(0) if fuse else 0,
        (gate.stride(1) if gate.ndim == 3 else HEAD_DIM) if fuse else 0,
        out.stride(0),
        part_lse.shape[0],
        float(eps),
        num_work,
        HQ=hq,
        GH=gh,
        D=HEAD_DIM,
        SLOT_ROWS=MAX_ROWS * hq,
        CHUNK=_kf_chunk(requests),
        FUSE=fuse,
        MAXR=64,
        PAGE=32,
        PAGES_PER_TILE=4,
        SEG_PAD=KF651_SEG_PAD,
        num_warps=gh,
        launch_pdl=True,
    )
    return out


def combine_norm_gate(pending, gate: torch.Tensor, eps: float, trigger: int = 0) -> torch.Tensor:
    """The deferred combine of ``multi_request_split_kv(..., defer_combine=True)`` (its ``pending``)
    with the attention norm-gate folded in: returns ``norm_gate``'s output ``[rows, HQ * 64]`` BF16
    (fresh, contiguous) and leaves the merged attention output in the call's ``out`` as the
    combine does. ``trigger``: ``norm_gate.TRIGGERS``' value."""
    (
        part_o,
        part_lse,
        seq_lens,
        out2,
        bmm2_scale,
        device_scale,
        requests,
        num_work,
        per,
        kv_heads,
        hq,
        window,
        enable_pdl,
    ) = pending
    chunk, warps = next(s[1:] for s in COMBINE_SHAPES if requests <= s[0])
    if warps != 4 or gate.stride(-1) != 1 or gate.dtype != torch.bfloat16:
        raise ValueError(
            "combine_norm_gate (multi-request): 5+ requests (4-warp combine) and a BF16 gate"
        )
    gated = torch.empty((out2.shape[0], hq * HEAD_DIM), dtype=torch.bfloat16, device=out2.device)
    _combine_norm_gate_kernel[(requests, per, kv_heads)](
        part_o,
        part_lse,
        seq_lens,
        out2,
        out2.stride(0),
        1.0 if device_scale else float(bmm2_scale),
        bmm2_scale if device_scale else part_lse,
        requests,
        num_work,
        per,
        gate,
        gated,
        gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else HEAD_DIM,
        eps,
        HQ=hq,
        GH=Q_PER_KV,
        D=HEAD_DIM,
        SLOT_ROWS=MAX_ROWS * hq,
        BLOCK_B=request_build(requests),
        P=PAGE,
        WINDOW=window,
        MIN_P=MIN_PAGES,
        CHUNK=chunk,
        MAX_SLOTS=COMBINE_MAX_SLOTS,
        DEVICE_SCALE=device_scale,
        TRIGGER=int(trigger),
        num_warps=warps,
        launch_pdl=True,
    )
    return gated


class MultiRequestSplitKv(OpWrapper):
    """Causal (optionally sliding-window) decode attention of a batch of requests of ``M`` rows
    each over the FP8 paged cache; writes ``out`` and returns it.

    Reads Q ``[B M, H 64]`` (``H = 8 G``), the generation preprocess's query: E4M3 with
    ``bmm1_scale`` / ``bmm2_scale`` as fp32 device tensors (the served call with the FP8 KV
    cache; ``bmm1_scale[0]`` and ``bmm2_scale[0]`` are read on the device after the dependency
    wait), or BF16 with both scales as host floats. Also reads the pool ``[blocks, G, 32, 64]``
    FP8 E4M3 (which must already hold the step's K/V), ``block_tables`` int32 ``[B, 2, pages]``
    and ``seq_lens`` int32 ``[B]``. ``workspace`` is ``workspace_for``'s pair (allocated per
    call when None). ``num_work`` (CTAs per pair) defaults to ``SMs / (G / 2)``.

    Loud preconditions (ValueError): E4M3 or BF16 Q with a unit last stride and 16-byte aligned
    rows, both scales host floats or both fp32 device tensors,
    ``rows = B M`` with ``1 <= M <= 8`` and ``1 <= B <= 128``; a contiguous FP8 E4M3 pool with
    ``G`` in ``RANK_KV_HEADS``; an int32 ``[B, 2, pages]`` table with a unit last stride;
    contiguous int32 ``seq_lens [B]``; a BF16 ``out`` of Q's shape with a unit last stride;
    partial buffers of at least ``num_work + B`` slots; ``1 <= num_work <= min(SMs,
    COMBINE_MAX_SLOTS)``; one sm_107 device. The first
    call per device and build compiles the kernel, which must not happen inside a CUDA-graph
    capture (``precompile``).
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: seq_lens (device; read by both kernels, never on the host), the batch's
    K and V page ids, the layer's pool view; bmm1_scale and bmm2_scale as the generation
    preprocess's fp32 device scalars with an E4M3 Q (the served build), or as host floats with
    a BF16 Q. Grid: num_work x (G / 2) CTAs of 384 threads, one CTA per SM (about 200 KB of
    shared memory and 512 TMEM columns); combine B x M x G programs.

    Gated by its own tests: every cell within a band of the op's fp32 reference (with the op's
    Q) for both Q types, and 1, 3 and 7 rows per request on the served build; the per-request
    slot ranges against the host mirror; determinism; one captured graph replayed at other
    lengths and device scales; device scales equal to host scales of the same value; NaN bytes
    in unwritten slots of last pages and in unreferenced pages.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        out: torch.Tensor,
        window: int = 0,
        enable_pdl: bool = False,
        workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_work: Optional[int] = None,
        nq: int = NQ,
        defer_combine: bool = False,
    ):
        q2 = q.reshape(q.shape[0], -1) if q.dim() == 3 else q
        rows = q2.shape[0] if q2.dim() == 2 else 0
        kv_heads = kv_pool.shape[1] if kv_pool.dim() == 4 else 0
        requests = seq_lens.numel()
        hq = kv_heads * Q_PER_KV
        if kv_heads not in RANK_KV_HEADS.values():
            raise ValueError(
                f"multi_request_split_kv takes a pool [blocks, G, {PAGE}, {HEAD_DIM}] with G in "
                f"{sorted(RANK_KV_HEADS.values())}; got {tuple(kv_pool.shape)}"
            )
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        if num_work is None:
            num_work = default_num_work(kv_heads, sms)
        if workspace is None:
            workspace = workspace_for(q.device, kv_heads, num_work, max(requests, 1))
        part_o, part_lse = workspace
        out2 = out.reshape(out.shape[0], -1) if out.dim() == 3 else out
        per = rows // requests if requests else 0
        tensors = (q2, kv_pool, block_tables, seq_lens, part_o, part_lse, out2)
        q_fp8 = q2.dtype == torch.float8_e4m3fn
        device_scale = isinstance(bmm1_scale, torch.Tensor)
        scales = (bmm1_scale, bmm2_scale) if device_scale else ()
        if (
            q2.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or q2.dim() != 2
            or q2.shape[1] != hq * HEAD_DIM
            or q2.stride(-1) != 1
            or q2.stride(0) % (16 if q_fp8 else 8) != 0
            or isinstance(bmm2_scale, torch.Tensor) != device_scale
            or any(
                t.dtype != torch.float32
                or t.numel() < 1
                or t.device != q.device
                or t.data_ptr() % 4
                for t in scales
            )
            or not 1 <= requests <= MAX_REQUESTS
            or rows != per * requests
            or not 1 <= per <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or tuple(kv_pool.shape[2:]) != (PAGE, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 3
            or block_tables.shape[0] < requests
            or block_tables.shape[1] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()
            or out2.dtype != torch.bfloat16
            or tuple(out2.shape) != tuple(q2.shape)
            or out2.stride(-1) != 1
            or part_o.dtype != torch.float16
            or part_lse.dtype != torch.float32
            or part_o.shape[0] < slots_needed(num_work, requests)
            or tuple(part_o.shape[1:]) != (MAX_ROWS * hq, HEAD_DIM)
            or tuple(part_lse.shape) != (part_o.shape[0], MAX_ROWS * hq)
            or not part_o.is_contiguous()
            or not part_lse.is_contiguous()
            or not 1 <= num_work <= min(sms, COMBINE_MAX_SLOTS)
            or window < 0
            or nq not in (1, 2)
            or any(t.device != q.device or t.data_ptr() % 16 for t in tensors)
            or q.device.type != "cuda"
            or torch.cuda.get_device_capability(q.device) != (10, 7)
        ):
            raise ValueError(
                "multi_request_split_kv takes, on one sm_107 device and 16-byte "
                f"aligned: BF16 or E4M3 q [B M, 8 G * {HEAD_DIM}] (1 <= M <= {MAX_ROWS}, 1 <= B <= "
                f"{MAX_REQUESTS}), a contiguous FP8 E4M3 pool [blocks, G, {PAGE}, {HEAD_DIM}], "
                "an int32 [B, 2, pages] table, contiguous int32 seq_lens [B], a BF16 out of q's "
                "shape, bmm1 / bmm2 scales both host floats or both fp32 device tensors, and "
                "partial buffers of at least num_work + B slots; got "
                f"q {q.dtype} {tuple(q.shape)} strides {tuple(q.stride())}, pool {kv_pool.dtype} "
                f"{tuple(kv_pool.shape)}, table {block_tables.dtype} {tuple(block_tables.shape)} "
                f"strides {tuple(block_tables.stride())}, seq_lens {seq_lens.dtype} "
                f"{tuple(seq_lens.shape)}, out {out.dtype} {tuple(out.shape)}, partials "
                f"{part_o.dtype} {tuple(part_o.shape)} / {part_lse.dtype} "
                f"{tuple(part_lse.shape)}, num_work {num_work}, window {window}, nq {nq}"
            )
        device_index = q.device.index
        key = compiled_key(
            device_index,
            enable_pdl,
            kv_heads,
            window,
            nq,
            q_fp8=q_fp8,
            device_scale=device_scale,
            requests=requests,
        )
        if torch.cuda.is_current_stream_capturing() and key not in _compiled_keys:
            raise RuntimeError(
                "multi_request_split_kv compiles on its first call per device and build; make "
                "that call (or precompile) before capturing a CUDA graph"
            )
        compiled = _compile(key)
        _compiled_keys.add(key)
        import cuda.bindings.driver as cuda

        stream = torch.cuda.current_stream(q.device)
        compiled(
            q2.data_ptr(),
            kv_pool.data_ptr(),
            block_tables.data_ptr(),
            seq_lens.data_ptr(),
            part_o.data_ptr(),
            part_lse.data_ptr(),
            requests,
            per,
            q2.stride(0),
            kv_pool.shape[0],
            block_tables.shape[2],
            block_tables.stride(0),
            block_tables.stride(1),
            num_work,
            part_o.shape[0],
            0.0 if device_scale else _host_scale_log2(bmm1_scale),
            bmm1_scale.data_ptr() if device_scale else part_lse.data_ptr(),
            cuda.CUstream(stream.cuda_stream),
        )
        if key[9]:
            # dieexp KF651: the kernel's partition differs from the served combine's; its own combine.
            pending = (
                part_o,
                part_lse,
                seq_lens,
                out2,
                bmm2_scale,
                device_scale,
                requests,
                num_work,
                per,
                kv_heads,
                hq,
                window,
                enable_pdl,
            )
            if defer_combine:
                return pending
            kf651_post(pending, None, 0.0, out2)
            return out2
        chunk, warps = next(s[1:] for s in COMBINE_SHAPES if requests <= s[0])
        if defer_combine:
            # s316 NORM_GATE_FOLD: the combine runs later, folded into the norm-gate (combine_norm_gate).
            return (
                part_o,
                part_lse,
                seq_lens,
                out2,
                bmm2_scale,
                device_scale,
                requests,
                num_work,
                per,
                kv_heads,
                hq,
                window,
                enable_pdl,
            )
        _combine_kernel[(requests, per, kv_heads)](
            part_o,
            part_lse,
            seq_lens,
            out2,
            out2.stride(0),
            1.0 if device_scale else float(bmm2_scale),
            bmm2_scale if device_scale else part_lse,
            requests,
            num_work,
            per,
            HQ=hq,
            GH=Q_PER_KV,
            D=HEAD_DIM,
            SLOT_ROWS=MAX_ROWS * hq,
            BLOCK_B=request_build(requests),
            P=PAGE,
            WINDOW=window,
            MIN_P=MIN_PAGES,
            CHUNK=chunk,
            MAX_SLOTS=COMBINE_MAX_SLOTS,
            DEVICE_SCALE=device_scale,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=warps,
            launch_pdl=enable_pdl,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        out: torch.Tensor,
        window: int = 0,
        enable_pdl: bool = False,
        workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_work: Optional[int] = None,
        nq: int = NQ,
    ) -> torch.Tensor:
        """fp32 attention over the cache with the op's Q (an E4M3 Q as is; a BF16 Q as
        ``Q_hi + Q_lo``, or one E4M3 term for ``nq=1``, formed exactly as the kernel forms them),
        P unrounded, TF32 off. Device scales are read on the host (``bmm1_scale[0]``,
        ``bmm2_scale[0]``). Returns ``[B M, H 64]`` fp32 (times ``bmm2_scale``); ``out`` is not
        written."""
        q_fp8 = q.dtype == torch.float8_e4m3fn
        if isinstance(bmm1_scale, torch.Tensor):
            bmm1_scale = float(bmm1_scale.reshape(-1)[0])
        if isinstance(bmm2_scale, torch.Tensor):
            bmm2_scale = float(bmm2_scale.reshape(-1)[0])
        q2 = (q.reshape(q.shape[0], -1) if q.dim() == 3 else q).float()
        requests = seq_lens.numel()
        rows = q2.shape[0] // requests
        kv_heads = kv_pool.shape[1]
        heads = kv_heads * Q_PER_KV

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()

        if q_fp8:
            q_eff = q2
        elif nq == 2:
            c = q2 * 1048577.0
            hi = c - (c - q2)
            q_eff = e4m3(hi) + e4m3(q2 - hi)
        else:
            q_eff = e4m3(q2)
        q_eff = q_eff.view(-1, heads, HEAD_DIM)
        pool_bytes = kv_pool.view(torch.uint8)
        result = torch.empty(q2.shape[0], heads, HEAD_DIM, dtype=torch.float32, device=q.device)
        with true_fp32_matmul():
            for b in range(requests):
                length = int(seq_lens[b])
                keys = torch.arange(length, device=q.device)

                def gather(kv: int) -> torch.Tensor:
                    pages = block_tables[b, kv, keys // PAGE].long()
                    return pool_bytes[pages, :, keys % PAGE].view(torch.float8_e4m3fn).float()

                k, v = gather(0), gather(1)
                for t in range(rows):
                    pos = length - rows + t
                    lo = max(pos - window + 1, 0) if window > 0 else 0
                    for h in range(kv_heads):
                        qs = q_eff[b * rows + t, h * Q_PER_KV : (h + 1) * Q_PER_KV]
                        s = qs @ k[lo : pos + 1, h].T * bmm1_scale
                        result[b * rows + t, h * Q_PER_KV : (h + 1) * Q_PER_KV] = (
                            torch.softmax(s, dim=-1) @ v[lo : pos + 1, h]
                        ) * bmm2_scale
        return result.reshape(q2.shape[0], -1)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        out: torch.Tensor,
        window: int = 0,
        enable_pdl: bool = False,
        workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_work: Optional[int] = None,
        nq: int = NQ,
    ) -> None:
        """Inputs the kernels take and answer wrongly, with no error.

        A request shorter than its rows puts its keys at negative positions; a table narrower
        than a request's pages reads page ids past its row; a page id outside the pool reads
        outside it.
        """
        requests = seq_lens.numel()
        rows = q.shape[0] // requests
        for b in range(requests):
            length = int(seq_lens[b])
            if length < rows:
                raise ValueError(f"seq_lens[{b}] = {length} is shorter than its {rows} rows")
            first, end = attended_pages(length, rows, window)
            if block_tables.shape[2] < end:
                raise ValueError(
                    f"a [.., 2, {block_tables.shape[2]}] table cannot address {length} keys"
                )
            used = block_tables[b, :, first:end]
            if int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]:
                raise ValueError(
                    f"request {b}: page ids [{int(used.min())}, {int(used.max())}] outside the "
                    f"{kv_pool.shape[0]}-page pool"
                )


multi_request_split_kv = MultiRequestSplitKv()
