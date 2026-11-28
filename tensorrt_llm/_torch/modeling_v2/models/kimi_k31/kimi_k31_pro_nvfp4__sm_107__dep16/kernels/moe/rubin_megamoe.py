# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE CuteDSL NVFP4 launch of the K3.1 Rubin targets: bucket buffers, staging, launch.

A copy of the engine backend's launch path (``TrtllmCutedslMegaMoeNvfp4Impl`` in
``moe/fused_moe/mega_moe/mega_moe_cute_dsl.py``: the bucket ladder, the build-time symmetric
buffers, ``_acquire_buffers``, ``_stage_inputs``, ``_launch_megamoe_kernel``, ``_prime_ladder`` and
``_run_moe``), with nothing kept per call and nothing read from the environment:

* :func:`build_workspace` allocates every bucket's buffers once (collective, in ladder order) and
  every call takes them as an argument;
* the staging is stateless: a call writes its live rows and the whole padded tail of the bucket's
  routing buffers (expert ids -1, weights 0), so what it writes does not depend on earlier calls
  or on the order in which CUDA graphs are captured (the engine backend refreshes only the rows
  that changed since the same instance's previous call);
* the combine wire format is BF16, no tactic sweep runs and the tuning pass primes every bucket
  (the engine backend reads them from ``MEGAMOE_COMBINE_FORMAT``, ``MEGAMOE_TACTIC_AUTOTUNE`` and
  ``MEGAMOE_PRIME_LADDER``).

Two ways in. :func:`run` takes the rows as tensors and stages them into the chunk's bucket (the
engine backend's path). :func:`run_in_bucket` launches on a bucket whose buffers the producers
already wrote (:func:`bucket_for` names it): the routing kernel's rows and the quantization's,
with the routing past the rows emptied by the quantization (``rubin_megamoe_quantize.py``), so the
launch is the kernel alone.

The kernel is the engine's ``trtllm::cute_dsl_megamoe_nvfp4_blackwell``; nothing here registers
an op. The expert operands are the weight table's rows as stored (the same bytes the engine's
MegaMoE quant method builds); the block scales are viewed flat per expert.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple, Optional, Sequence

import torch

from tensorrt_llm.logger import logger

#: The engine's SM107 bucket ladder (``_resolve_maxt_buckets``): rows per rank of one launch.
SM107_RUNGS = (32, 64, 128, 192, 256, 384, 512, 1024, 4096)
#: The SM107 kernel's expert-id dtype.
TOPK_DTYPE = torch.int32

# Buckets already primed in this process, by the workspace's first symmetric buffer.
_PRIMED: set = set()


class MegaMoeBucket(NamedTuple):
    """One ladder bucket: launches of up to ``max_tokens`` rows per rank."""

    max_tokens: int
    activation: torch.Tensor  # [max_tokens, hidden / 2] U8: NVFP4 values (symmetric on EP > 1)
    activation_sf: torch.Tensor  # [max_tokens, sf bytes per row] U8: their E4M3 scales
    topk_weights: torch.Tensor  # [max_tokens, top_k] FP32
    combine_output: torch.Tensor  # [max_tokens, 1, hidden] BF16: the routed rows, complete
    shared_workspace: torch.Tensor  # the kernel's shared workspace (U8)
    peer_offsets: list  # every rank's buffer minus this rank's; [0] on one rank
    topk_idx: torch.Tensor  # [max_tokens, top_k] int32, local memory: ids, -1 past the rows
    direct_inputs: bool  # the kernel reads its inputs where they are when the bucket is full


class MegaMoeWorkspace(NamedTuple):
    """Everything a MegaMoE launch reads besides the step's rows and the layer's experts."""

    buckets: tuple  # MegaMoeBucket, ascending max_tokens
    ep_rank: int
    ep_size: int
    num_local_experts: int
    hidden: int  # the experts' input width (the routed latent)
    intermediate: int  # an expert's intermediate on this rank
    top_k: int
    chunk_tokens: int  # rows per rank one chunk of a step may hold
    situ: tuple  # (activation_situ_beta, activation_situ_linear_beta)


def ladder(top: int, rungs: Sequence[int] = SM107_RUNGS) -> tuple[int, ...]:
    """The buckets: every rung below ``top`` and ``top`` itself (4096 when ``top`` is not set)."""
    full = int(top) if int(top) > 0 else 4096
    return tuple(sorted({int(b) for b in rungs if b < full} | {full}))


def select_bucket(ws: MegaMoeWorkspace, chunk_max_tokens: int) -> MegaMoeBucket:
    """The smallest bucket that holds ``chunk_max_tokens`` rows per rank.

    ``chunk_max_tokens`` is the largest row count of the chunk over the EP ranks, so every rank
    picks the same bucket.
    """
    if chunk_max_tokens <= 0:
        return ws.buckets[-1]
    for bucket in ws.buckets:
        if chunk_max_tokens <= bucket.max_tokens:
            return bucket
    # Rank-identical input: every rank raises here, no rank waits at the barrier.
    raise RuntimeError(
        f"MegaMoE: {chunk_max_tokens} rows per rank exceed the top bucket "
        f"{ws.buckets[-1].max_tokens}"
    )


#: The generation kernel's token-exchange pusher CTAs as shipped (its ``pusher_cta_count``), and
#: the most it takes: 212 SMs less the kernel's 46 four-CTA clusters leave 28, the helper's and
#: the pushers' (the exchange CTAs leave once their count rows are out).
_DEFAULT_PUSHER_CTAS = 27
_MAX_PUSHER_CTAS = 27


def _bind_pusher_ctas(pusher_ctas: int | None) -> None:
    """Set the generation kernel's pusher CTAs to ``pusher_ctas`` (None: leave its default).

    The count is a class constant of the kernel (``RubinInferenceGenphaseMegaMoE.pusher_cta_count``)
    that its workspace size and every launch it compiles read, so it holds for the whole process:
    bind it before the first workspace query and keep it. Rebinding to another value raises.
    """
    if pusher_ctas is None:
        return
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe import RubinInferenceGenphaseMegaMoE

    pushers = int(pusher_ctas)
    if not 1 <= pushers <= _MAX_PUSHER_CTAS:
        raise ValueError(
            f"MegaMoE pusher CTAs {pushers}: the generation kernel takes 1 to {_MAX_PUSHER_CTAS}"
        )
    bound = RubinInferenceGenphaseMegaMoE.pusher_cta_count
    if bound not in (_DEFAULT_PUSHER_CTAS, pushers):
        raise ValueError(
            f"MegaMoE pusher CTAs {pushers}: this process already runs the generation kernel "
            f"with {bound}"
        )
    RubinInferenceGenphaseMegaMoE.pusher_cta_count = pushers


def exchange_grid_ctas(ep_size: int) -> int:
    """The SMs the generation kernel's token exchange takes at its launch: one exchange-and-sort
    CTA per rank, the helper and the pushers (``pusher_cta_count``, which :func:`build_workspace`
    binds), padded to whole launch clusters (``GenphaseTokenComm.launch_cluster_width``)."""
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe import RubinInferenceGenphaseMegaMoE
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mega import (
        token_comm_gen_specialized as exchange,
    )

    width = exchange.GenphaseTokenComm.launch_cluster_width
    ctas = ep_size + 1 + RubinInferenceGenphaseMegaMoE.pusher_cta_count
    return -(-ctas // width) * width


def build_workspace(
    group: Any,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    max_num_tokens: int,
    chunk_tokens: int,
    device: torch.device,
    *,
    situ: tuple,
    rungs: Sequence[int] = SM107_RUNGS,
    pusher_ctas: int | None = None,
) -> MegaMoeWorkspace:
    """Allocate every bucket's buffers; collective over ``group`` when ``ep_size`` > 1.

    The top bucket holds ``min(chunk_tokens, max_num_tokens)`` rows. On more than one rank each
    bucket's activation, scales, weights, output and shared workspace are one symmetric buffer
    rendezvoused over ``group`` (``get_megamoe_symm_provider``), bucket by bucket in ladder order,
    which every rank derives from the same arguments. The expert ids live in local memory. The
    combine is BF16, and the kernel never reduces FC2 in-kernel: K3.1 applies the routing weights
    after FC2.
    ``pusher_ctas`` (None: the kernel's default) binds the generation kernel's token-exchange
    pusher CTAs for the process before the first query (``_bind_pusher_ctas``).
    """
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        get_megamoe_symm_provider,
        megamoe_activation_sf_bytes_per_row,
        query_megamoe_kernel_properties,
    )

    _bind_pusher_ctas(pusher_ctas)
    buckets = []
    for max_t in ladder(min(chunk_tokens, max_num_tokens), rungs):
        shared_bytes, direct = query_megamoe_kernel_properties(
            world_size=ep_size,
            num_topk=top_k,
            num_experts_per_rank=num_local_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            expand_intermediate_size_per_partition=2 * intermediate,
            max_tokens_per_rank=max_t,
            apply_topk_in_fc1=False,
            gate_up_clamp=None,
            situ_beta=situ[0],
            situ_linear_beta=situ[1],
            in_kernel_fc2_reduce=False,
            combine_format="bf16",
        )
        topk_idx = torch.full((max_t, top_k), -1, dtype=TOPK_DTYPE, device=device)
        if ep_size == 1:
            sf_bytes = megamoe_activation_sf_bytes_per_row(hidden)
            u8 = dict(dtype=torch.uint8, device=device)
            bucket = MegaMoeBucket(
                max_t,
                torch.empty(max_t, hidden // 2, **u8),
                torch.empty(max_t, sf_bytes, **u8),
                torch.zeros(max_t, top_k, dtype=torch.float32, device=device),
                torch.empty(max_t, 1, hidden, dtype=torch.bfloat16, device=device),
                # Zeroed: its leading counters are atomic-add targets the first launch reads.
                torch.zeros(shared_bytes, **u8),
                [0],
                topk_idx,
                False,
            )
        else:
            regions = get_megamoe_symm_provider(
                process_group=group,
                world_size=ep_size,
                rank=ep_rank,
                hidden_size=hidden,
                max_tokens_per_rank=max_t,
                num_topk=top_k,
                output_dtype=torch.bfloat16,
                combine_format="bf16",
                shared_workspace_bytes=shared_bytes,
            ).get_regions()
            bucket = MegaMoeBucket(
                max_t,
                regions.activation,
                regions.activation_sf,
                regions.topk_weights,
                regions.combine_output,
                regions.shared_workspace,
                list(regions.peer_offsets),
                topk_idx,
                bool(direct),
            )
        buckets.append(bucket)
    if ep_rank == 0:
        logger.info(f"MegaMoE buckets (rows per rank): {[b.max_tokens for b in buckets]}")
    return MegaMoeWorkspace(
        tuple(buckets),
        ep_rank,
        ep_size,
        num_local_experts,
        hidden,
        intermediate,
        top_k,
        int(chunk_tokens),
        tuple(situ),
    )


def stage(
    bucket: MegaMoeBucket,
    x4: Optional[torch.Tensor],
    sf: Optional[torch.Tensor],
    ids: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    num_tokens: int,
    top_k: int,
    stage_activation: bool,
) -> None:
    """Write the call's rows into the bucket's buffers and set the whole padded tail.

    Expert ids past the live rows are -1 (the dispatch skips them) and their weights 0 (the
    combine reads every slot): written on every call, so the result never depends on what an
    earlier call left in the tail.
    """
    max_t = bucket.max_tokens
    if num_tokens < max_t:
        bucket.topk_idx[num_tokens:].fill_(-1)
    if num_tokens > 0:
        bucket.topk_idx[:num_tokens].copy_(ids, non_blocking=True)
        if stage_activation:
            bucket.activation[:num_tokens].copy_(x4.view(torch.uint8), non_blocking=True)
            bucket.activation_sf[:num_tokens].copy_(sf.view(torch.uint8), non_blocking=True)
        bucket.topk_weights[:num_tokens, :top_k].copy_(weights, non_blocking=True)
    if num_tokens < max_t:
        bucket.topk_weights[num_tokens:max_t, :top_k].zero_()


def _launch(
    ws: MegaMoeWorkspace,
    bucket: MegaMoeBucket,
    activation: torch.Tensor,
    activation_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: tuple,
    num_tokens: int,
) -> torch.Tensor:
    """Launch the kernel once over ``bucket``; return ``[num_tokens, hidden]`` BF16.

    The result is a view of the bucket's output, valid until the next launch on the bucket.
    """
    fc1_w, fc1_s, fc2_w, fc2_s, fc31_alpha, fc2_alpha, fc1_norm_const = experts
    num_local = fc1_w.shape[0]
    torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
        activation=activation.view(torch.float4_e2m1fn_x2),
        activation_sf=activation_sf.view(torch.float8_e4m3fn),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        fc1_weight=fc1_w.view(torch.float4_e2m1fn_x2),
        fc1_weight_sf=fc1_s.view(num_local, -1).view(torch.float8_e4m3fn),
        fc2_weight=fc2_w.view(torch.float4_e2m1fn_x2),
        fc2_weight_sf=fc2_s.view(num_local, -1).view(torch.float8_e4m3fn),
        fc1_alpha=fc31_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        combine_output=bucket.combine_output,
        shared_workspace=bucket.shared_workspace,
        world_size=ws.ep_size,
        local_rank=ws.ep_rank,
        num_topk=ws.top_k,
        num_experts_per_rank=ws.num_local_experts,
        hidden_size=ws.hidden,
        intermediate_size_per_partition=ws.intermediate,
        expand_intermediate_size_per_partition=2 * ws.intermediate,
        max_tokens_per_rank=bucket.max_tokens,
        peer_offsets=bucket.peer_offsets,
        apply_topk_in_fc1=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        gate_up_clamp=None,
        act_alpha=ws.situ[0],
        act_beta=ws.situ[1],
        in_kernel_fc2_reduce=False,
        combine_format="bf16",
        tactic_autotune=False,
        num_tokens=num_tokens,
    )
    return bucket.combine_output[:num_tokens].squeeze(1)


def _prime(ws: MegaMoeWorkspace, live: MegaMoeBucket, experts: tuple) -> None:
    """Warm-up: launch every other bucket once with no rows, so no bucket compiles in serving.

    Runs in the first tuning pass, once per process and workspace (the bucket that pass runs live
    is left out); the buckets and their order are the same on every rank.
    """
    key = int(ws.buckets[0].shared_workspace.data_ptr())
    if key in _PRIMED:
        return
    t0 = time.perf_counter()
    for bucket in ws.buckets:
        if bucket.max_tokens == live.max_tokens:
            continue
        stage(bucket, None, None, None, None, 0, ws.top_k, False)
        _launch(
            ws,
            bucket,
            bucket.activation,
            bucket.activation_sf,
            bucket.topk_idx,
            bucket.topk_weights[:, : ws.top_k],
            experts,
            0,
        )
    _PRIMED.add(key)
    if ws.ep_rank == 0:
        logger.info(
            f"MegaMoE primed {len(ws.buckets) - 1} buckets in {time.perf_counter() - t0:.1f}s"
        )


def bucket_for(ws: MegaMoeWorkspace, chunk_max_tokens: int, num_tokens: int) -> MegaMoeBucket:
    """The bucket one chunk launches over, which holds this rank's ``num_tokens`` rows of it.

    :func:`select_bucket`'s, so every rank names the same bucket.
    """
    bucket = select_bucket(ws, chunk_max_tokens)
    if num_tokens > bucket.max_tokens:
        raise RuntimeError(
            f"MegaMoE: {num_tokens} rows in a bucket of {bucket.max_tokens}: the chunk's largest "
            f"row count over the ranks was {chunk_max_tokens}"
        )
    return bucket


def _prime_once(ws: MegaMoeWorkspace, live: MegaMoeBucket, experts: tuple) -> None:
    """:func:`_prime` in the tuning pass, where the workspace primes (both ways in alike)."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    if ws.ep_size > 1 and AutoTuner.get().is_tuning_mode:
        _prime(ws, live, experts)


def run_in_bucket(
    ws: MegaMoeWorkspace, bucket: MegaMoeBucket, experts: tuple, num_tokens: int
) -> torch.Tensor:
    """This rank's rows of one chunk through MegaMoE, read from the bucket's own buffers.

    ``bucket`` (:func:`bucket_for`) already holds the rows: their NVFP4 values and linear scales in
    rows ``[0, num_tokens)`` of ``activation`` / ``activation_sf``, their routing in those rows of
    ``topk_idx`` / ``topk_weights``, and ids -1 with weights 0 in every row past them (what
    :func:`stage` writes). Nothing is staged: the launch is the kernel alone. ``experts`` as in
    :func:`run`. Every rank launches, with no rows too.
    """
    if num_tokens == 0 and ws.ep_size == 1:
        return torch.empty(0, ws.hidden, dtype=torch.bfloat16, device=bucket.topk_idx.device)
    _prime_once(ws, bucket, experts)
    return _launch(
        ws,
        bucket,
        bucket.activation,
        bucket.activation_sf,
        bucket.topk_idx,
        bucket.topk_weights[:, : ws.top_k],
        experts,
        num_tokens,
    )


def run(
    ws: MegaMoeWorkspace,
    x4: torch.Tensor,
    sf: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    experts: tuple,
    chunk_max_tokens: int,
    num_tokens: int,
) -> torch.Tensor:
    """This rank's rows of one chunk through MegaMoE: ``[num_tokens, hidden]`` BF16, complete.

    ``x4`` / ``sf``: the rows' NVFP4 values ``[num_tokens, hidden / 2]`` and linear E4M3 scales;
    ``ids`` int32 / ``weights`` FP32 ``[num_tokens, top_k]``; ``experts``: ``(fc1_w, fc1_s, fc2_w,
    fc2_s, fc31_alpha, fc2_alpha, fc1_norm_const)`` of this rank's window as the weight table
    stores them; ``chunk_max_tokens``: the chunk's largest row count over the EP ranks. Every
    rank launches, with no rows too: the kernel's exchange waits for all of them.
    """
    bucket = bucket_for(ws, chunk_max_tokens, num_tokens)
    if num_tokens == 0 and ws.ep_size == 1:
        return torch.empty(0, ws.hidden, dtype=torch.bfloat16, device=bucket.topk_idx.device)
    _prime_once(ws, bucket, experts)
    # A full bucket on the generation kernel reads the rows where they are.
    direct = (
        bucket.direct_inputs
        and num_tokens == bucket.max_tokens
        and x4.shape == bucket.activation.shape
        and sf.shape == bucket.activation_sf.shape
        and x4.is_contiguous()
        and sf.is_contiguous()
    )
    direct_routing = (
        direct
        and ids.shape == weights.shape == (bucket.max_tokens, ws.top_k)
        and ids.dtype == TOPK_DTYPE
        and weights.dtype == torch.float32
        and ids.is_contiguous()
        and weights.is_contiguous()
        and ids.data_ptr() % 16 == 0
        and weights.data_ptr() % 16 == 0
    )
    if not direct_routing:
        stage(bucket, x4, sf, ids, weights, num_tokens, ws.top_k, not direct)
    return _launch(
        ws,
        bucket,
        x4 if direct else bucket.activation,
        sf if direct else bucket.activation_sf,
        ids if direct_routing else bucket.topk_idx,
        weights if direct_routing else bucket.topk_weights[:, : ws.top_k],
        experts,
        num_tokens,
    )
