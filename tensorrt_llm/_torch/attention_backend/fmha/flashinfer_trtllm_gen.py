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

"""
FlashInfer TRTLLM-Gen FMHA

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a TRT-LLM attention FMHA library for trtllm-gen kernels
(Blackwell architecture: SM100/SM103). Enable or disable it through
``TLLM_FMHA_LIBS``.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via tensorrt_llm.bindings.internal.thop,
      same as thop.attention. Writes K/V to paged KV cache via pool pointers.
    - Attention: flashinfer trtllm-gen FMHA kernels, reading KV cache through
      the paged KV cache fields carried by FmhaParams.

Entry points:
    FlashInferTrtllmGenFmha.is_available() - Check if this FMHA library can be instantiated.
    FlashInferTrtllmGenFmha.is_supported() - Check if trtllm-gen can handle the given request.
    FlashInferTrtllmGenFmha.forward()      - Main attention method.

Example:
    fmha = FlashInferTrtllmGenFmha(attn=...)
    if fmha.is_supported(q, k, v, metadata, forward_args):
        fmha.forward(q, k, v, metadata, forward_args)
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxParams
from tensorrt_llm._torch.q35_env import register as _q35_register
from tensorrt_llm._utils import get_sm_version, is_sm_100f, torch_dtype_to_binding
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .phased import FmhaParams, PhasedFmha
from .q35_cp import q35_cp_decode_ll, q35_cp_decode_nccl
from .q35_fused_decode import (
    q35_fused_decode,
    q35_fused_decode_fold,
    q35_fused_decode_fold_supported,
    q35_fused_decode_supported,
    q35_fused_workspace_floats,
)
from .q35_segmented import q35_segmented_decode, q35_segmented_supported

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _install_flashinfer_mla_decode_tuning_config_cache() -> None:
    """Work around an autotuner cache pathology on flashinfer 0.6.15's MLA decode path.

    ``flashinfer.mla._core._build_mla_decode_tuning_config`` builds a fresh
    ``TuningConfig`` — with fresh initializer closures — on every
    ``trtllm_batch_decode_with_kv_cache_mla`` call. ``DynamicTensorSpec``'s
    custom ``__hash__`` skips ``tensor_initializers``, but its
    dataclass-generated ``__eq__`` compares them by value, so each call's
    config is hash-equal but eq-unequal to all previous ones: the
    ``lru_cache`` on ``AutoTuner._find_nearest_profile`` accumulates
    colliding keys up to the cache cap, and every autotuner cache probe on
    the MLA decode path then walks the entire collision chain in Python
    ``__eq__`` (tens of milliseconds per attention call, per layer).

    The workaround memoizes the config builder on the scalars that fully
    determine its output (kv-cache paging geometry, workspace size, runner
    set, head/rank dims, max_seq_len, device), so repeated calls reuse the
    same ``TuningConfig`` object. Cache hits then resolve through
    flashinfer's own unmodified hash/eq — no equality semantics change
    anywhere, and the scope is exactly the MLA decode path that exhibits the
    problem.

    MLA decode is the only path that needs this: every other ``TuningConfig``
    construction site in flashinfer 0.6.15 already reuses its config across
    calls (module-level constants in ``gemm``, instance-level configs in the
    cute-dsl MoE tuner, ``functools.cache``-wrapped builders in
    ``mla/_sparse_mla_sm120.py``), so those specs hit the autotuner caches by
    object identity and never accumulate collision chains. Memoizing the one
    per-call builder therefore covers the whole pathology; extending it to
    other ops would add nothing.

    The bug is upstream in flashinfer; the patch is applied only when a
    behavioral probe shows the hash/eq inconsistency and the builder still
    has the expected signature, so a fixed or refactored flashinfer release
    degrades this to a no-op.
    """
    try:
        import inspect

        from flashinfer.autotuner.autotuner import DynamicTensorSpec
        from flashinfer.mla import _core as _flashinfer_mla_core

        def _probe_spec() -> DynamicTensorSpec:
            return DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(1,),
                map_to_tuning_buckets=_probe_spec,  # any stable callable
                tensor_initializers=[lambda shapes, dtype, device: None],
            )

        a, b = _probe_spec(), _probe_spec()
        if not (hash(a) == hash(b) and a != b):
            return  # flashinfer without the inconsistency; nothing to do

        orig_build = _flashinfer_mla_core._build_mla_decode_tuning_config
        if getattr(orig_build, "_trtllm_mla_tuning_config_cache", False):
            return  # already installed

        expected_params = (
            "kv_cache",
            "block_tables",
            "workspace_buffer",
            "runner_names",
            "q_len",
            "num_heads",
            "kv_lora_rank",
            "max_seq_len",
            "device",
        )
        signature = inspect.signature(orig_build)
        if tuple(signature.parameters) != expected_params:
            return  # builder was refactored; memo key may no longer be complete
        try:
            signature.bind(**dict.fromkeys(expected_params))
        except TypeError:
            return  # parameters went positional-only; the keyword call below would fail

        cache: dict = {}

        def _cached_build(
            kv_cache,
            block_tables,
            workspace_buffer,
            runner_names,
            q_len,
            num_heads,
            kv_lora_rank,
            max_seq_len,
            device,
        ):
            # Everything the builder (and _compute_mla_decode_buckets) reads
            # from its tensor arguments reduces to these scalars.
            key = (
                kv_cache.shape[0],  # num_pages, captured by init_block_tables
                kv_cache.shape[-2],  # page_size, feeds profile_seq_len
                block_tables.shape[-1],  # pages per sequence, feeds profile_seq_len
                workspace_buffer.numel() * workspace_buffer.element_size(),  # cute-dsl bucket cap
                tuple(runner_names),
                q_len,
                num_heads,
                kv_lora_rank,
                max_seq_len,
                device,
            )
            config = cache.get(key)
            if config is None:
                if len(cache) >= 256:  # backstop; a serving process sees a handful of keys
                    cache.clear()
                config = orig_build(
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    workspace_buffer=workspace_buffer,
                    runner_names=runner_names,
                    q_len=q_len,
                    num_heads=num_heads,
                    kv_lora_rank=kv_lora_rank,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                cache[key] = config
            return config

        _cached_build._trtllm_mla_tuning_config_cache = True
        _flashinfer_mla_core._build_mla_decode_tuning_config = _cached_build
        logger.debug(
            "Memoized flashinfer _build_mla_decode_tuning_config "
            "(autotuner cache collision-chain workaround)."
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        # A future flashinfer refactor (moved module, changed fields) must not
        # break attention; it just loses the workaround.
        # tensorrt_llm's logger.debug(*msg) takes no exc_info/kwargs.
        logger.debug("Skipping flashinfer MLA decode tuning-config cache workaround.")


if IS_FLASHINFER_AVAILABLE:
    _install_flashinfer_mla_decode_tuning_config_cache()


_MULTI_CTAS_KV_COUNTER_ALIGNMENT = 8


@_q35_register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global _Q35_ATTN_SEGMENTS, _Q35_ATTN_CP, _Q35_ATTN_CP_MIN_BATCH, _Q35_ATTN_CP_MAX_BATCH
    global _Q35_CP_MAX_PAIR_BYTES, _Q35_ATTN_SEG_MAX_BATCH, _Q35_ATTN_KV_APPEND
    global _Q35_ATTN_FUSED_DECODE, _Q35_ATTN_FOLD
    # Qwen3.5 low-latency switch (default off). TRTLLM_Q35_ATTN_SEGMENTS=<S> (S >= 2): uniform-q_len generation
    # attention runs as S page-aligned context segments per request through the same trtllm-gen kernel plus a Triton
    # LSE merge (fmha/q35_segmented.py); cuts the split-KV reduction cost at batch 1-2.
    try:
        _Q35_ATTN_SEGMENTS = int(os.environ.get("TRTLLM_Q35_ATTN_SEGMENTS", "0"))
    except ValueError:
        _Q35_ATTN_SEGMENTS = 0
    # CP-lite (TRTLLM_Q35_ATTN_CP=1, see fmha/q35_cp.py): generation batches >= TRTLLM_Q35_ATTN_CP_MIN_BATCH (default
    # 2) read their KV page-strided across the TP ranks of the KV group; TRTLLM_Q35_ATTN_CP_SEGMENTS local segments per
    # rank ("auto", default: 2 at batch <= 2 or q_len 1, else 1 -- GB300 4-rank microbenchmark).
    _Q35_ATTN_CP = os.environ.get("TRTLLM_Q35_ATTN_CP", "0") == "1"
    # Batches above TRTLLM_Q35_ATTN_CP_MAX_BATCH (default 8) keep the standard path (no gain there; the LL all-to-all
    # carries at most _Q35_CP_MAX_PAIR_BYTES per (source, destination) pair).
    # TRTLLM_Q35_ATTN_CP_MIN_BATCH (default 2 at every layout): the segmented path wins at b1 at TP8 and at TP16
    # (8-rank microbenchmark seg16 31.5 vs CP 36.4 us/call at 196k; CP is also slower in-model at TEP16 b1).
    try:
        _Q35_ATTN_CP_MIN_BATCH = (
            int(os.environ["TRTLLM_Q35_ATTN_CP_MIN_BATCH"])
            if "TRTLLM_Q35_ATTN_CP_MIN_BATCH" in os.environ
            else None
        )
        _Q35_ATTN_CP_MAX_BATCH = int(os.environ.get("TRTLLM_Q35_ATTN_CP_MAX_BATCH", "8"))
    except ValueError:
        _Q35_ATTN_CP_MIN_BATCH, _Q35_ATTN_CP_MAX_BATCH = None, 8
    # = TPGroupExchange's per-pair buffer (TRTLLM_Q35_CP_EXCHANGE_KB, default 256 KiB; b32 MTP6 needs ~463 KiB).
    _Q35_CP_MAX_PAIR_BYTES = int(os.environ.get("TRTLLM_Q35_CP_EXCHANGE_KB", "256")) * 1024
    # Largest generation batch that takes the segmented path (GB300 microbench: S=16 wins ~9 us/call at b1, ~1-2 us at
    # b2, loses ~3 us at b4 where the trtllm-gen kernel is already bandwidth-bound).
    try:
        _Q35_ATTN_SEG_MAX_BATCH = int(os.environ.get("TRTLLM_Q35_ATTN_SEG_MAX_BATCH", "2"))
    except ValueError:
        _Q35_ATTN_SEG_MAX_BATCH = 2
    # TRTLLM_Q35_ATTN_KV_APPEND=1: in a generation-only step the attention module's fused qk-norm / RoPE / gate kernel
    # also writes the FP8 query, appends the FP8 K / V of the new tokens to the paged cache and prepares the FMHA
    # scales (q35_kv_append_plan gives it the destination); run_generation then skips trtllm_gen_generation_preprocess,
    # i.e. its QKV kernel and, at batch >= 2 with q_len > 1, BuildDecoderInfo. The bytes written are identical to the
    # C++ path.
    _Q35_ATTN_KV_APPEND = os.environ.get("TRTLLM_Q35_ATTN_KV_APPEND", "0") == "1"
    # Fused decode attention (TRTLLM_Q35_ATTN_FUSED_DECODE=1, default off): one CuTe DSL launch
    # (fmha/q35_fused_decode.py: TMA paged FP8 KV, tcgen05 QK / PV, online softmax with the causal draft block,
    # in-kernel split reduction, output gate) replaces the segmented chain for generation-only batches of one KV head
    # per rank up to TRTLLM_Q35_ATTN_FUSED_MAX_BATCH (default 4). CP-lite, when enabled and applicable, takes
    # precedence.
    _Q35_ATTN_FUSED_DECODE = os.environ.get("TRTLLM_Q35_ATTN_FUSED_DECODE", "0") == "1"
    # Folded prologue (TRTLLM_Q35_ATTN_FOLD=1, default off; needs TRTLLM_Q35_ATTN_KV_APPEND=1,
    # TRTLLM_Q35_ATTN_FUSED_DECODE=1 and TRTLLM_Q35_ATTN_SEG_FUSE_GATE=1): calls that take the fused decode path skip
    # the module's qk-norm / RoPE / gate / KV append Triton kernel; the fused kernel reads the raw QKV projection and
    # does that work itself (bit-identical; the attention kernel becomes the QKV GEMM's PDL dependent).
    # q35_fold_inputs decides per call, before the prologue.
    _Q35_ATTN_FOLD = os.environ.get("TRTLLM_Q35_ATTN_FOLD", "0") == "1"


def _q35_cp_min_batch(cp_group_size: int) -> int:
    del cp_group_size  # unused: the threshold is 2 at both TP8 and TP16
    return 2 if _Q35_ATTN_CP_MIN_BATCH is None else _Q35_ATTN_CP_MIN_BATCH


_Q35_ATTN_CP_SEGMENTS_ENV = os.environ.get("TRTLLM_Q35_ATTN_CP_SEGMENTS", "auto")


def _q35_cp_segments(num_requests: int, q_len: int) -> int:
    if _Q35_ATTN_CP_SEGMENTS_ENV != "auto":
        try:
            return max(1, int(_Q35_ATTN_CP_SEGMENTS_ENV))
        except ValueError:
            pass
    return 2 if (num_requests <= 2 or q_len == 1) else 1


# CP-lite exchanges: one-launch LL all-gather / all-to-all within the KV group (TPGroupExchange). Without peer memory
# for it, CP-lite is skipped (the NCCL fallback is ~2x slower than not splitting: 94.7 vs 50.1 us/call at b4 196k);
# TRTLLM_Q35_ATTN_CP_NCCL=1 forces the NCCL fallback (testing only).
_Q35_ATTN_CP_NCCL = os.environ.get("TRTLLM_Q35_ATTN_CP_NCCL", "0") == "1"
# Per-flag opt-out of the MTP (draft) layer attention from CP-lite / the segmented path.
_Q35_ATTN_CP_SKIP_MTP = os.environ.get("TRTLLM_Q35_ATTN_CP_SKIP_MTP", "0") == "1"
_Q35_ATTN_SEG_SKIP_MTP = os.environ.get("TRTLLM_Q35_ATTN_SEG_SKIP_MTP", "0") == "1"
# Counter slots for segmented / CP-lite calls: >= max(b * segments * folded heads, #SMs) (<= 64 * 128 here).
_Q35_COUNTER_SLOTS = 64 * 128
# TRTLLM_Q35_ATTN_KV_APPEND_CHECK=1 (eager calls only, testing): also runs the C++ preprocess on those calls and
# compares the query, the scales and the appended K / V bytes (counts in Q35_KV_APPEND_CHECK_STATS, mismatches logged).
_Q35_ATTN_KV_APPEND_CHECK = os.environ.get("TRTLLM_Q35_ATTN_KV_APPEND_CHECK", "0") == "1"
Q35_KV_APPEND_CHECK_STATS = {"calls": 0, "mismatches": 0}


@dataclass
class Q35FoldInputs:
    """Raw inputs of the qk-norm / RoPE / gate / KV-append prologue for TRTLLM_Q35_ATTN_FOLD (see q35_fold_inputs)."""

    qkv: (
        torch.Tensor
    )  # raw QKV projection [num_tokens, (2 * Hq + 2 * Hkv) * D] bf16 ((q, gate) per head, K, V)
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos_sin: torch.Tensor
    positions: torch.Tensor  # [T] or [3, T] (interleaved mrope), contiguous
    eps: float
    rotary_dim: int
    mrope_section: Optional[tuple]


@dataclass
class Q35KvAppendPlan:
    """Paged FP8 destination of one generation-only attention call (TRTLLM_Q35_ATTN_KV_APPEND).

    Built by q35_kv_append_plan for the attention module; the module's kernel fills q_fp8 / scales and hands the plan
    back through ``attn.q35_kv_appended``.
    """

    layer_idx: int
    num_tokens: int
    num_requests: int
    q_len: int
    kv_pool: torch.Tensor  # FP8 [blocks, num_kv_heads, tokens_per_block, head_dim]
    block_tables: torch.Tensor  # int32 [num_requests, 2 (K, V), max_pages]
    seq_lens: torch.Tensor  # int32, KV length including the new tokens
    kv_scale_orig_quant: torch.Tensor
    kv_scale_quant_orig: torch.Tensor
    host_bmm1_scale: float
    q_fp8: Optional[torch.Tensor] = None  # [num_tokens, num_heads, head_dim]
    # float32 [8]: bmm1, bmm1 * log2e at [0:2]; bmm2 at [4] (both views 16-byte aligned)
    scales: Optional[torch.Tensor] = None
    # TRTLLM_Q35_ATTN_FOLD: the module did not run the prologue kernel; run_generation runs the folded decode kernel
    # (or, if the call ends up elsewhere, the prologue kernel itself) on these inputs.
    fold: Optional[Q35FoldInputs] = None


try:
    _Q35_ATTN_FUSED_MAX_BATCH = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_MAX_BATCH", "4"))
except ValueError:
    _Q35_ATTN_FUSED_MAX_BATCH = 4
_Q35_ATTN_FUSED_SKIP_MTP = os.environ.get("TRTLLM_Q35_ATTN_FUSED_SKIP_MTP", "0") == "1"
# TRTLLM_Q35_ATTN_FOLD_CHECK=1 (eager calls only, testing): after each folded call also run the prologue kernel and the
# unfolded fused decode on the same inputs and compare the output and the new tokens' K / V bytes bit for bit
# (counts in Q35_FOLD_CHECK_STATS; mismatches are logged).
_Q35_ATTN_FOLD_CHECK = os.environ.get("TRTLLM_Q35_ATTN_FOLD_CHECK", "0") == "1"
Q35_FOLD_CHECK_STATS = {"calls": 0, "mismatches": 0}
# Shared by every attention layer (the calls are serialized on the decode stream): device -> (workspace, counters).
_Q35_FUSED_BUFFERS: dict = {}


def _q35_fused_buffers(device: torch.device, sm_count: int, capturing: bool):
    key = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _Q35_FUSED_BUFFERS.get(key)
    if bufs is None:
        if capturing:
            raise RuntimeError(
                "The Q35 fused-decode workspace must be allocated before CUDA graph capture."
            )
        ws = torch.zeros(q35_fused_workspace_floats(sm_count), dtype=torch.float32, device=device)
        # grid-barrier epoch counters: row b serves batch b; zero once, never reset (the kernel keeps them consistent)
        cnt = torch.zeros(_Q35_ATTN_FUSED_MAX_BATCH + 1, 8, dtype=torch.int64, device=device)
        bufs = (ws, cnt)
        _Q35_FUSED_BUFFERS[key] = bufs
    return bufs


def _get_multi_ctas_kv_counter_size(
    num_heads: int,
    max_num_requests: int,
    multi_processor_count: int,
) -> int:
    num_counters = max(num_heads * max_num_requests, multi_processor_count)
    aligned_num_counters = (
        (num_counters + _MULTI_CTAS_KV_COUNTER_ALIGNMENT - 1)
        // _MULTI_CTAS_KV_COUNTER_ALIGNMENT
        * _MULTI_CTAS_KV_COUNTER_ALIGNMENT
    )
    return aligned_num_counters * torch.int32.itemsize


def _get_bmm1_scale_log2(bmm1_scale: torch.Tensor) -> torch.Tensor:
    if bmm1_scale.numel() < 2:
        raise RuntimeError("trtllm-gen bmm1_scale workspace must contain raw and log2 scales.")
    return bmm1_scale.narrow(0, 1, 1)


def _trtllm_gen_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    workspace_buffer: torch.Tensor,
    multi_ctas_kv_counter_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    window_left: int,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor],
    enable_pdl: bool,
    q_len_per_req: Optional[int],
    max_q_len: Optional[int],
    cum_seq_lens_q: Optional[torch.Tensor],
    kv_scale_pool: Optional[torch.Tensor],
    uses_shared_paged_kv_idx: bool,
) -> None:
    if q_len_per_req is not None:
        decode_max_q_len = q_len_per_req
        batch_size = query.size(0) // q_len_per_req
    else:
        if max_q_len is None or cum_seq_lens_q is None:
            raise RuntimeError(
                "trtllm-gen multi-token generation requires max_q_len and cum_seq_lens_q."
            )
        decode_max_q_len = max_q_len
        batch_size = cum_seq_lens_q.size(0) - 1

    bmm1_scale_arg = (
        _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
    )

    run_func = flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
    sm_count = flashinfer.decode.get_device_sm_count(query.device)
    run_func(
        out,
        None,  # out_scale_factor
        query,
        kv_pool,
        kv_pool,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        block_tables,
        seq_lens,
        decode_max_q_len,
        max_seq_len,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        batch_size,
        window_left,
        0,  # sparse_mla_top_k
        sm_count,
        enable_pdl,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
        cum_seq_lens_q,
        kv_scale_pool,  # k_block_scales
        kv_scale_pool,  # v_block_scales
        None,  # skip_softmax_threshold_scale_factor
        uses_shared_paged_kv_idx,
        None,  # lse
        0,  # lse_stride_tokens
        0,  # lse_stride_heads
        False,  # enable_block_sparse_attention (added in flashinfer 0.6.16, flashinfer-ai/flashinfer#3955)
    )


def _trtllm_gen_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    workspace_buffer: torch.Tensor,
    multi_ctas_kv_counter_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    window_left: int,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor],
    enable_pdl: bool,
    kv_scale_pool: Optional[torch.Tensor],
    uses_shared_paged_kv_idx: bool,
    causal: bool,
) -> None:
    bmm1_scale_arg = (
        _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
    )

    run_func = flashinfer.prefill.get_trtllm_gen_fmha_module().trtllm_paged_attention_context
    sm_count = flashinfer.prefill.get_device_sm_count(query.device)
    run_func(
        out,
        None,  # out_scale_factor
        query,
        kv_pool,
        kv_pool,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        batch_size,
        window_left,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        sm_count,
        enable_pdl,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
        kv_scale_pool,  # key_block_scales
        kv_scale_pool,  # value_block_scales
        None,  # skip_softmax_threshold_scale_factor
        uses_shared_paged_kv_idx,
        causal,  # causal
        None,  # lse
        0,  # lse_stride_tokens
        0,  # lse_stride_heads
    )


@lru_cache(maxsize=128)
def _get_context_workspace_layout(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> dict[str, int]:
    return thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        batch_size,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        fp8_context_fmha,
    )


@lru_cache(maxsize=128)
def _get_context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    if max_num_tokens == 0:
        return 0
    layout = _get_context_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        fp8_context_fmha,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _get_generation_workspace_layout(
    dtype: torch.dtype,
    batch_beam: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> dict[str, int]:
    return thop.get_trtllm_gen_generation_workspace_layout(
        dtype,
        batch_beam,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        num_kv_heads,
    )


@lru_cache(maxsize=128)
def _get_generation_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> int:
    if max_num_tokens == 0:
        return 0
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    layout = _get_generation_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _get_workspace_size(
    dtype: torch.dtype,
    num_tokens: int,
    num_gen_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_num_requests: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    context_size = _get_context_workspace_size(
        dtype,
        max_num_requests,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        fp8_context_fmha,
    )
    generation_size = _get_generation_workspace_size(
        dtype,
        max_num_requests,
        num_gen_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return max(context_size, generation_size)


class FlashInferTrtllmGenFmha(PhasedFmha):
    """
    An attention backend using pure trtllm-gen kernels from flashinfer.
    """

    # Default KV layout for flashinfer
    # HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
    DEFAULT_KV_LAYOUT = "HND"
    # Keep shared paged indices disabled to match the current TensorRT-LLM
    # block-table layout used by the fused preprocessing path.
    USE_SHARED_PAGED_KV_IDX = False

    # Supported data types
    SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_KV_CACHE_DTYPES = {DataType.HALF, DataType.BF16, DataType.FP8, DataType.NVFP4}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    # Supported Q:KV:O dtype combinations for trtllm-gen kernels
    # Format: (q_dtype: torch.dtype, kv_dtype: DataType, o_dtype: torch.dtype)
    SUPPORTED_DTYPE_COMBOS_CONTEXT = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }
    SUPPORTED_DTYPE_COMBOS_GENERATION = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        (torch.bfloat16, DataType.FP8, torch.bfloat16),
        (torch.float16, DataType.FP8, torch.float16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }

    # 96 is excluded because trtllm-gen does not ship context kernels for it.
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80, 96}
    MAX_HEADS_RATIO_GENERATION = 32
    MIN_TOKENS_PER_BLOCK = 8
    SUPPORTED_TOKENS_PER_BLOCK = {16, 32, 64}
    # FlashInfer narrows key_cache.size(0) to int before constructing its TMA descriptor.
    MAX_NUM_PAGES_IN_MEM_POOL = (1 << 31) - 1
    SUPPORTED_MLA_GENERATION_HEAD_DIMS = {
        (320, 256),
        (576, 512),
    }
    SLOWER_MLA_GENERATION_KERNELS = {
        (576, 512, 32),
    }

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._layout = self.DEFAULT_KV_LAYOUT
        # Read once so the hot path is not sensitive to later environment changes.
        self._enable_pdl = get_env_enable_pdl()

        # Lazily set on the first forward() call from the query device.
        self._multi_processor_count: Optional[int] = None
        self._multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None
        # Segmented / CP-lite calls run (b * segments) "requests" of up to 128 folded heads; the launcher requires
        # max(batch * heads, SMs) counters, more than the regular buffer holds. Separate zero-initialized buffer
        # (the kernel resets its counters at the end of every launch).
        self._q35_counter_buffer: Optional[torch.Tensor] = None

    def _get_total_num_blocks(self, meta: "TrtllmAttentionMetadata") -> int:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is not None:
            get_page_index_upper_bound = getattr(
                getattr(kv_cache_manager, "impl", None), "get_page_index_upper_bound", None
            )
            # KVCacheManagerV2 exposes this implementation-only API and reports an
            # already-flattened page-index bound, unlike the legacy logical block count.
            if get_page_index_upper_bound is not None:
                return int(kv_cache_manager.blocks_in_primary_pool)
        return super()._get_total_num_blocks(meta)

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not IS_FLASHINFER_AVAILABLE:
            logger.debug("FlashInfer TRTLLM-Gen FMHA is unavailable: flashinfer is not installed.")
            return False

        missing_ops = cls._missing_fused_nanobind_ops()
        if missing_ops:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: missing fused "
                f"nanobind ops: {', '.join(missing_ops)}."
            )
            return False

        sm = get_sm_version()
        if not is_sm_100f(sm):
            logger.debug(
                f"FlashInfer TRTLLM-Gen FMHA is unavailable: requires SM100 or SM103, got SM{sm}."
            )
            return False

        has_skip_softmax = isinstance(attn.sparse_params, SkipSoftmaxParams)
        if has_skip_softmax:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: skip-softmax attention is enabled."
            )
            return False

        if attn.num_heads <= 0 or attn.num_kv_heads <= 0:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: "
                f"num_heads={attn.num_heads}, num_kv_heads={attn.num_kv_heads}."
            )
            return False

        if attn.num_heads % attn.num_kv_heads != 0:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: "
                f"num_heads ({attn.num_heads}) must be divisible by "
                f"num_kv_heads ({attn.num_kv_heads})."
            )
            return False

        return True

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

    def _get_kv_scale_params(
        self,
        forward_args: AttentionForwardArgs,
        quant_mode: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        kv_cache_quant_mode = QuantMode(quant_mode)
        kv_scale_orig_quant = forward_args.kv_scale_orig_quant
        kv_scale_quant_orig = forward_args.kv_scale_quant_orig
        if (
            not kv_cache_quant_mode.has_kv_cache_quant()
            or kv_scale_orig_quant is None
            or kv_scale_quant_orig is None
        ):
            return None, None

        if kv_cache_quant_mode.has_fp4_kv_cache():
            assert kv_scale_orig_quant.size(0) == 3, (
                f"kv_scale_orig_quant must have size(0)==3 for FP4, got {kv_scale_orig_quant.size(0)}"
            )
            assert kv_scale_quant_orig.size(0) == 3, (
                f"kv_scale_quant_orig must have size(0)==3 for FP4, got {kv_scale_quant_orig.size(0)}"
            )

        return kv_scale_orig_quant, kv_scale_quant_orig

    @staticmethod
    def _get_kv_cache_dtype(
        meta: "TrtllmAttentionMetadata",
    ) -> Optional[DataType]:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is not None:
            return kv_cache_manager.dtype
        return None

    @staticmethod
    def _get_bmm1_scale(attn: "TrtllmAttention") -> float:
        return 1.0 / (math.sqrt(attn.head_dim) * attn.q_scaling)

    @staticmethod
    def _get_attention_chunk_size(attn: "TrtllmAttention") -> int:
        return attn.attention_chunk_size if attn.attention_chunk_size is not None else 0

    @classmethod
    def _check_mla_generation_support(
        cls,
        head_size: int,
        tokens_per_block: int,
        kv_lora_rank: Optional[int],
        qk_rope_head_dim: Optional[int],
    ) -> Tuple[bool, str]:
        missing_params = [
            name
            for name, value in (
                ("kv_lora_rank", kv_lora_rank),
                ("qk_rope_head_dim", qk_rope_head_dim),
            )
            if value is None or value <= 0
        ]
        if missing_params:
            return (
                False,
                "[Generation][MLA] missing required MLA parameter(s): "
                f"{', '.join(missing_params)}.",
            )

        kv_rank = int(kv_lora_rank)
        qk_rope_dim = int(qk_rope_head_dim)
        head_dim_qk = kv_rank + qk_rope_dim
        head_dim_v = kv_rank
        if head_size != head_dim_qk:
            return (
                False,
                f"[Generation][MLA] head_size ({head_size}) that does not match "
                f"kv_lora_rank + qk_rope_head_dim ({head_dim_qk}).",
            )

        if (head_dim_qk, head_dim_v) not in cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS:
            supported = sorted(cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS)
            return (
                False,
                f"[Generation][MLA] head dimensions "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}. Supported: {supported}.",
            )

        if (head_dim_qk, head_dim_v, tokens_per_block) in cls.SLOWER_MLA_GENERATION_KERNELS:
            return (
                False,
                f"[Generation][MLA] slower TRTLLM-GEN decode kernel for "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}, "
                f"tokens_per_block={tokens_per_block}.",
            )

        return True, ""

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            k,
            v,
            self.attn,
            metadata,
            forward_args,
        )
        if not supported:
            logger.debug(f"FlashInfer trtllm-gen fmha library does not support {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        is_mla_enable = attn.is_mla_enable
        sparse_params = attn.sparse_params
        has_skip_softmax = sparse_params is not None and sparse_params.algorithm == "skip_softmax"
        has_sparse_attention = sparse_params is not None and not has_skip_softmax
        if fwd.enable_dsv4_epilogue_fusion:
            return False, "trtllm-gen does not support DSv4 epilogue fusion."
        if (
            fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
        ):
            return False, "sage attention."
        if meta.helix_position_offsets is not None:
            return False, "helix parallelism."
        sparse_kv_indices = fwd.sparse_runtime_params.sparse_kv_indices
        sparse_attn_indices = fwd.sparse_runtime_params.sparse_attn_indices
        if (
            (sparse_kv_indices is not None and sparse_kv_indices.numel() > 0)
            or (sparse_attn_indices is not None and sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
            or has_sparse_attention
        ):
            return False, "sparse attention."
        if has_skip_softmax:
            return False, "skip-softmax attention."
        if fwd.relative_attention_bias is not None:
            return False, "relative attention bias."
        if meta.use_spec_decoding and meta.is_spec_dec_tree:
            return False, "spec-dec tree/custom masks."
        if is_mla_enable and fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "MLA with non-generation-only attention."

        if meta.kv_cache_block_offsets is None:
            return False, "non-paged KV cache; paged KV cache is required."

        num_pages_in_mem_pool = self._get_total_num_blocks(meta)
        if num_pages_in_mem_pool > self.MAX_NUM_PAGES_IN_MEM_POOL:
            return False, (
                f"more than {self.MAX_NUM_PAGES_IN_MEM_POOL} flattened KV-cache "
                f"pages, but this pool requires {num_pages_in_mem_pool}."
            )

        output = fwd.output
        if output is None:
            return False, "a missing output tensor; output is required."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = 0

        attn_input_type = fwd.attention_input_type
        has_context_phase = attn_input_type != AttentionInputType.generation_only
        has_generation_phase = attn_input_type != AttentionInputType.context_only
        q_dtype = q.dtype
        o_dtype = output.dtype

        if q_dtype not in self.SUPPORTED_INPUT_DTYPES:
            return False, f"input dtype {q_dtype}. Supported: FP16, BF16, FP8 (E4M3)."

        kv_cache_dtype = self._get_kv_cache_dtype(meta)
        if kv_cache_dtype is None:
            kv_cache_dtype = torch_dtype_to_binding(q_dtype)
        if meta.is_cross:
            if kv_cache_dtype == DataType.NVFP4:
                return False, "cross attention with NVFP4 KV cache."
            if is_mla_enable:
                return False, "cross attention with MLA."
            if meta.is_spec_decoding_enabled or meta.use_spec_decoding:
                return False, "cross attention with speculative decoding."
            if fwd.update_kv_cache and fwd.cross_kv is None:
                return False, "cross attention with a missing cross_kv when update_kv_cache=True."

        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        has_fp8_kv = kv_cache_dtype == DataType.FP8
        has_fp4_kv = kv_cache_dtype == DataType.NVFP4
        fp8_context_fmha = (
            is_fp8_out or is_fp4_out or has_fp4_kv or (has_fp8_kv and has_context_phase)
        )
        if has_fp4_kv or fp8_context_fmha:
            q_dtype = torch.float8_e4m3fn

        if kv_cache_dtype not in self.SUPPORTED_KV_CACHE_DTYPES:
            return False, f"KV cache dtype {kv_cache_dtype}. Supported: FP16, BF16, FP8, NVFP4."
        if o_dtype not in self.SUPPORTED_OUT_DTYPES:
            return False, f"output dtype {o_dtype}. Supported: FP16, BF16, FP8."

        has_alibi = attn.position_embedding_type in (4, 5)
        check_context_phase = has_context_phase and not is_mla_enable
        if check_context_phase:
            if attn.head_dim in self.UNSUPPORTED_HEAD_SIZES_CONTEXT:
                return False, f"[Context] head size {attn.head_dim}."
            try:
                if AttentionMaskType(fwd.mask_type) == AttentionMaskType.custom_mask:
                    return False, "[Context] custom mask."
            except ValueError:
                return False, f"[Context] invalid mask_type: {fwd.mask_type}."
            if has_alibi:
                return False, "[Context] ALiBi."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_CONTEXT:
                return False, (
                    f"[Context] dtype combination: Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )

        if has_generation_phase:
            if meta.beam_width != 1 and not meta.is_cross:
                return False, (
                    f"[Generation] beam search (beam_width={meta.beam_width}); must be 1."
                )
            sink_token_length = 0
            if sink_token_length != 0:
                return False, (
                    f"[Generation] StreamingLLM (sink_token_length={sink_token_length})."
                )
            if tokens_per_block < self.MIN_TOKENS_PER_BLOCK:
                return False, (
                    f"[Generation] tokens_per_block ({tokens_per_block}); "
                    f"must be >= {self.MIN_TOKENS_PER_BLOCK}."
                )
            heads_ratio = attn.num_heads // attn.num_kv_heads
            if not is_mla_enable and heads_ratio > self.MAX_HEADS_RATIO_GENERATION:
                return False, (
                    f"[Generation] heads ratio ({heads_ratio}) exceeding maximum "
                    f"({self.MAX_HEADS_RATIO_GENERATION})."
                )
            if has_alibi:
                return False, "[Generation] ALiBi."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_GENERATION:
                return False, (
                    f"[Generation] dtype combination: "
                    f"Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )
            if is_mla_enable:
                supported, reason = self._check_mla_generation_support(
                    head_size=attn.head_dim,
                    tokens_per_block=tokens_per_block,
                    kv_lora_rank=attn.kv_lora_rank,
                    qk_rope_head_dim=attn.qk_rope_head_dim,
                )
                if not supported:
                    return False, reason

        if tokens_per_block <= 0:
            return False, f"non-positive tokens_per_block ({tokens_per_block})."
        if tokens_per_block & (tokens_per_block - 1) != 0:
            return False, f"tokens_per_block ({tokens_per_block}) that is not a power of 2."
        if tokens_per_block not in self.SUPPORTED_TOKENS_PER_BLOCK:
            supported = sorted(self.SUPPORTED_TOKENS_PER_BLOCK)
            return False, f"tokens_per_block ({tokens_per_block}). Supported: {supported}."

        return True, ""

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_multi_processor_count_for_device(device_index: int) -> int:
        return torch.cuda.get_device_properties(device_index).multi_processor_count

    def _get_multi_processor_count(self, device: torch.device) -> int:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("trtllm-gen requires CUDA tensors.")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return self._get_multi_processor_count_for_device(device_index)

    def get_fp8_context_fmha(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        is_gen_only: bool,
    ) -> bool:
        kv_cache_quant_mode = QuantMode(self.attn.quant_mode)
        return (
            output.dtype == torch.float8_e4m3fn
            or output.dtype == torch.uint8
            or kv_cache_quant_mode.has_fp4_kv_cache()
            or (kv_cache_quant_mode.has_fp8_kv_cache() and not is_gen_only)
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        attn = self.attn
        # Lazily cache the SM count from the first query tensor's device.
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)

        required_counter_size = _get_multi_ctas_kv_counter_size(
            attn.num_heads,
            metadata.max_num_requests,
            self._multi_processor_count,
        )
        counter_buffer = self._multi_ctas_kv_counter_buffer
        if (
            counter_buffer is None
            or counter_buffer.device != q.device
            or counter_buffer.numel() * counter_buffer.element_size() < required_counter_size
        ):
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The trtllm-gen multi-CTA KV counter buffer must be allocated "
                    "before CUDA graph capture."
                )
            self._multi_ctas_kv_counter_buffer = torch.zeros(
                required_counter_size,
                dtype=torch.uint8,
                device=q.device,
            )

        if (_Q35_ATTN_SEGMENTS > 1 or _Q35_ATTN_CP) and (
            self._q35_counter_buffer is None or self._q35_counter_buffer.device != q.device
        ):
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The Q35 segmented-attention counter buffer must be allocated before capture."
                )
            self._q35_counter_buffer = torch.zeros(
                _Q35_COUNTER_SLOTS, dtype=torch.int32, device=q.device
            )

        if _Q35_ATTN_FUSED_DECODE:
            _q35_fused_buffers(
                q.device,
                self._multi_processor_count,
                metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing(),
            )

        num_tokens = q.size(0)
        attention_input_type = forward_args.attention_input_type
        is_gen_only = attention_input_type == AttentionInputType.generation_only
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        fp8_context_fmha = self.get_fp8_context_fmha(q, output, metadata, forward_args, is_gen_only)

        workspace_max_tokens = max(num_tokens, metadata.max_context_length)
        workspace_max_gen_tokens = max(num_gen_tokens, metadata.max_num_requests)
        required_workspace_size = _get_workspace_size(
            dtype=q.dtype,
            num_tokens=workspace_max_tokens,
            num_gen_tokens=workspace_max_gen_tokens,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            max_num_requests=metadata.max_num_requests,
            rotary_embedding_dim=attn.rope_dim,
            fp8_context_fmha=fp8_context_fmha,
        )

        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the "
                    "required size for trtllm-gen."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))

    def _get_multi_ctas_kv_counter_buffer(self) -> torch.Tensor:
        counter_buffer = self._multi_ctas_kv_counter_buffer
        if counter_buffer is None:
            raise RuntimeError("The trtllm-gen multi-CTA KV counter buffer is not initialized.")
        return counter_buffer

    @staticmethod
    def _compute_window_left(
        cyclic_attention_window_size: int,
        max_kv_length: int,
        attention_chunk_size: int,
    ) -> int:
        """Compute flashinfer window_left with TRTLLM semantics alignment.

        TRTLLM's attention_window_size is exclusive, while flashinfer's
        window_left is inclusive. Keep the same user-visible behavior by
        subtracting 1 when sliding-window attention is enabled.
        """
        if attention_chunk_size != 0 and cyclic_attention_window_size < max_kv_length:
            raise ValueError(
                "Chunked-attention and sliding-window-attention should not "
                "be enabled at the same time."
            )
        if 0 < cyclic_attention_window_size < max_kv_length:
            return cyclic_attention_window_size - 1
        return -1

    @staticmethod
    def _missing_fused_nanobind_ops() -> List[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
            "build_trtllm_gen_kv_cache_metadata",
        )
        return [op for op in required_ops if not hasattr(thop, op)]

    def run_context(
        self,
        params: FmhaParams,
    ) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        bmm1_scale_static = self._get_bmm1_scale(attn)
        attention_chunk_size = self._get_attention_chunk_size(attn)
        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len,
            max_kv_len,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            attention_chunk_size,  # attention_chunk_size
            params.fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
            fwd.cross_kv,  # cross_kv
            params.is_cross,  # is_cross
        )

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or params.fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        ctx_bmm1_scale = (
            bmm1_scale if params.fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        ctx_bmm2_scale = bmm2_scale if params.fp8_context_fmha and bmm2_scale is not None else 1.0
        causal = (
            False
            if params.is_cross
            else AttentionMaskType(fwd.mask_type) == AttentionMaskType.causal
        )
        _trtllm_gen_batch_context_with_kv_cache(
            q_processed,  # query
            kv_pool,  # kv_pool
            fmha_workspace,  # workspace_buffer
            self._get_multi_ctas_kv_counter_buffer(),  # multi_ctas_kv_counter_buffer
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            max_q_len,  # max_q_len
            max_kv_len,  # max_kv_len
            ctx_bmm1_scale,  # bmm1_scale
            ctx_bmm2_scale,  # bmm2_scale
            params.batch_size,  # batch_size
            cu_q_seqlens,  # cum_seq_lens_q
            cu_kv_seqlens,  # cum_seq_lens_kv
            window_left,  # window_left
            params.context_buf,  # out
            fwd.attention_sinks,  # sinks
            self._enable_pdl,  # enable_pdl
            kv_scale_pool,  # kv_scale_pool
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
            causal,  # causal
        )

        if params.is_cross:
            return

        thop.trtllm_gen_context_postprocess(
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            params.fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
        )

    def q35_run_context_lean(self, params: FmhaParams, causal: bool) -> None:
        """run_context for TRTLLM_Q35_CTX_ATTN_PLAN=2: the same launches with this layer's constant scalars worked out
        once (bmm1 scale, chunk size, FP4-KV flag) and the processed query re-typed with one view instead of the
        flatten / slice / view chain when the preprocess returned it as a contiguous [tokens, heads, head_dim] byte
        tensor (what the chain rebuilds). ``causal`` is the caller's mask decision."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        consts = self.__dict__.get("_q35_ctx_consts")
        if consts is None or consts[0] is not attn:
            consts = (
                attn,
                self._get_bmm1_scale(attn),
                self._get_attention_chunk_size(attn),
                QuantMode(attn.quant_mode).has_fp4_kv_cache(),
            )
            self._q35_ctx_consts = consts
        _, bmm1_scale_static, attention_chunk_size, has_fp4_kv = consts
        rope_params = attn.rope_params
        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len,
            max_kv_len,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            bmm1_scale_static,
            1.0,
            attention_chunk_size,
            params.fp8_context_fmha,
            meta.use_paged_context_fmha,
            attn.is_mla_enable,
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            fwd.cross_kv,
            params.is_cross,
        )
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or params.fp8_context_fmha:
            shape = (params.num_tokens, attn.num_heads, attn.head_dim)
            if (
                q_processed.dtype == torch.uint8
                and tuple(q_processed.shape) == shape
                and q_processed.is_contiguous()
            ):
                q_processed = q_processed.view(torch.float8_e4m3fn)
            else:
                q_processed = (
                    q_processed.view(torch.uint8)
                    .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                    .view(torch.float8_e4m3fn)
                    .view(*shape)
                )
        ctx_bmm1_scale = (
            bmm1_scale if params.fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        ctx_bmm2_scale = bmm2_scale if params.fp8_context_fmha and bmm2_scale is not None else 1.0
        _trtllm_gen_batch_context_with_kv_cache(
            q_processed,
            kv_pool,
            fmha_workspace,
            self._get_multi_ctas_kv_counter_buffer(),
            block_tables,
            params.sequence_lengths,
            max_q_len,
            max_kv_len,
            ctx_bmm1_scale,
            ctx_bmm2_scale,
            params.batch_size,
            cu_q_seqlens,
            cu_kv_seqlens,
            window_left,
            params.context_buf,
            fwd.attention_sinks,
            self._enable_pdl,
            kv_scale_pool,
            self.USE_SHARED_PAGED_KV_IDX,
            causal,
        )
        thop.trtllm_gen_context_postprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            bmm1_scale_static,
            params.fp8_context_fmha,
            meta.use_paged_context_fmha,
            attn.is_mla_enable,
            attention_chunk_size,
            self._multi_processor_count,
        )

    def run_generation(
        self,
        params: FmhaParams,
    ) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        bmm1_scale_static = self._get_bmm1_scale(attn)
        attention_chunk_size = self._get_attention_chunk_size(attn)
        batch_beam = params.num_requests * meta.beam_width
        # TRTLLM_Q35_ATTN_KV_APPEND: the attention module already wrote this call's FP8 query / K / V.
        q35_plan = getattr(attn, "q35_kv_appended", None)
        preprocessed = None
        if q35_plan is not None:
            attn.q35_kv_appended = None
            if q35_plan.fold is not None:
                # TRTLLM_Q35_ATTN_FOLD: the folded fused decode, or the prologue kernel the module skipped
                if self._q35_fold_generation(params, q35_plan):
                    return
                self._q35_fold_fallback(params, q35_plan)
            preprocessed = self._q35_appended_preprocess(
                params, q35_plan, batch_beam, attention_chunk_size
            )
        q35_check = (
            preprocessed is not None
            and _Q35_ATTN_KV_APPEND_CHECK
            and not torch.cuda.is_current_stream_capturing()
        )
        if preprocessed is None or q35_check:
            q35_kv_before = _q35_new_kv_bytes(q35_plan) if q35_check else None
            stock = self._run_generation_preprocess(
                params, batch_beam, bmm1_scale_static, attention_chunk_size
            )
            if q35_check:
                self._q35_check_appended(params, q35_plan, preprocessed, stock, q35_kv_before)
            else:
                preprocessed = stock
        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            cu_seqlens,
            max_q_len,
            max_kv_len,
            window_left,
            is_multi_token_gen,
        ) = preprocessed

        q_len_per_req = None if is_multi_token_gen else params.input_seq_length
        decode_max_q_len = max_q_len if is_multi_token_gen else None
        decode_cu_seqlens = cu_seqlens if is_multi_token_gen else None
        # TRTLLM_Q35_DRAFT_ATTN_WINDOW (set on the Qwen3.5 MTP layer only): its decode
        # attention reads the last W keys of each sequence; the KV cache keeps every
        # position (KV writes above use the full, non-cyclic window).
        q35_draft_window = getattr(attn, "q35_draft_window", 0)
        if q35_draft_window:
            window_left = q35_draft_window - 1

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or params.fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        gen_bmm1_scale = (
            bmm1_scale if params.fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        gen_bmm2_scale = bmm2_scale if params.fp8_context_fmha and bmm2_scale is not None else 1.0

        q35_cp_group = getattr(attn, "q35_cp_group", None)
        if (
            _Q35_ATTN_CP
            and q35_cp_group is not None
            and not (_Q35_ATTN_CP_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
            and _q35_cp_min_batch(len(q35_cp_group))
            <= params.num_requests
            <= _Q35_ATTN_CP_MAX_BATCH
            and self._q35_cp_exchange_ok(attn, len(q35_cp_group))
            and params.num_tokens * attn.num_heads * (attn.head_dim * 2 + 4)
            <= _Q35_CP_MAX_PAIR_BYTES
            and meta.num_contexts == 0
            and not is_multi_token_gen
            and not has_fp4_kv
            and kv_scale_pool is None
            and meta.beam_width == 1
            and not params.is_cross
            and attn.num_kv_heads == 1
            and window_left < 0
            and fwd.attention_sinks is None
            and q_processed.dim() == 3
            and q_processed.is_contiguous()
        ):
            self._q35_cp_generation(
                params,
                q_processed,
                kv_pool,
                block_tables,
                max_kv_len,
                gen_bmm1_scale,
                gen_bmm2_scale,
                fmha_workspace,
                q35_cp_group,
            )
            return

        if (
            _Q35_ATTN_FUSED_DECODE
            and params.num_requests <= _Q35_ATTN_FUSED_MAX_BATCH
            and not (_Q35_ATTN_FUSED_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
            and meta.num_contexts == 0
            and not is_multi_token_gen
            and not has_fp4_kv
            and kv_scale_pool is None
            and meta.beam_width == 1
            and not params.is_cross
            and q_processed.dim() == 3
            and q_processed.is_contiguous()
            and kv_pool.dim() == 4
            and kv_pool.is_contiguous()
            and q35_fused_decode_supported(
                attn.num_heads,
                attn.num_kv_heads,
                attn.head_dim,
                params.input_seq_length,
                params.num_requests,
                window_left,
                fwd.attention_sinks is not None,
                params.tokens_per_block,
                q_processed.dtype,
                params.context_buf.dtype,
                kv_pool.dtype,
                self._multi_processor_count,
            )
        ):
            q35_gate = getattr(attn, "q35_pending_gate", None)
            if q35_gate is not None:
                if q35_gate.shape[0] < params.num_tokens:
                    q35_gate = None
                else:
                    q35_gate = q35_gate[: params.num_tokens]
            ws, cnt = _q35_fused_buffers(q_processed.device, self._multi_processor_count, False)
            q35_fused_decode(
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                params.input_seq_length,
                gen_bmm1_scale,
                gen_bmm2_scale,
                params.context_buf,
                ws,
                cnt,
                self._multi_processor_count,
                self._enable_pdl,
                gate=q35_gate,
            )
            if q35_gate is not None:
                attn.q35_gate_applied = True
            return

        if (
            _Q35_ATTN_SEGMENTS > 1
            and params.num_requests <= _Q35_ATTN_SEG_MAX_BATCH
            and not (_Q35_ATTN_SEG_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
            and not is_multi_token_gen
            and not has_fp4_kv
            and meta.beam_width == 1
            and not params.is_cross
            and q_processed.dim() == 3
            and q_processed.is_contiguous()
            and q35_segmented_supported(
                attn.num_heads,
                attn.num_kv_heads,
                attn.head_dim,
                params.input_seq_length,
                params.num_requests,
                _Q35_ATTN_SEGMENTS,
                window_left,
                fwd.attention_sinks is not None,
                params.tokens_per_block,
                q_processed.dtype,
                params.context_buf.dtype,
            )
        ):
            # TRTLLM_Q35_ATTN_SEG_FUSE_GATE: the attention module hands over its output gate
            # (attn.q35_pending_gate); fused into the merge only when every token of the call is a generation token.
            q35_gate = getattr(attn, "q35_pending_gate", None)
            if q35_gate is not None:
                if meta.num_contexts > 0 or q35_gate.shape[0] < params.num_tokens:
                    q35_gate = None
                else:
                    q35_gate = q35_gate[: params.num_tokens]
            q35_segmented_decode(
                flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode,
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                max_kv_len,
                params.input_seq_length,
                attn.num_kv_heads,
                _Q35_ATTN_SEGMENTS,
                gen_bmm1_scale,
                _get_bmm1_scale_log2(gen_bmm1_scale)
                if isinstance(gen_bmm1_scale, torch.Tensor)
                else gen_bmm1_scale,
                gen_bmm2_scale,
                fmha_workspace,
                self._q35_counter_buffer,
                params.context_buf,
                flashinfer.decode.get_device_sm_count(q_processed.device),
                kv_scale_pool,
                self._enable_pdl,
                gate=q35_gate,
            )
            if q35_gate is not None:
                attn.q35_gate_applied = True
            return

        _trtllm_gen_batch_decode_with_kv_cache(
            q_processed,  # query
            kv_pool,  # kv_pool
            fmha_workspace,  # workspace_buffer
            self._get_multi_ctas_kv_counter_buffer(),  # multi_ctas_kv_counter_buffer
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            max_kv_len,  # max_seq_len
            gen_bmm1_scale,  # bmm1_scale
            gen_bmm2_scale,  # bmm2_scale
            window_left,  # window_left
            params.context_buf,  # out
            fwd.attention_sinks,  # sinks
            self._enable_pdl,  # enable_pdl
            q_len_per_req,  # q_len_per_req
            decode_max_q_len,  # max_q_len
            decode_cu_seqlens,  # cum_seq_lens_q
            kv_scale_pool,  # kv_scale_pool
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
        )

    def _run_generation_preprocess(
        self,
        params: FmhaParams,
        batch_beam: int,
        bmm1_scale_static: float,
        attention_chunk_size: int,
    ) -> tuple:
        """C++ generation preprocess: FP8 / bf16 query, K / V append into the paged cache, FMHA scales and views."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        return thop.trtllm_gen_generation_preprocess(
            params.qkv_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.spec_decoding_generation_lengths,  # spec_decoding_generation_lengths
            params.spec_decoding_position_offsets,  # spec_decoding_position_offsets
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_position_deltas,  # mrope_position_deltas
            attn.local_layer_idx,  # layer_idx
            params.seq_offset,  # seq_offset
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            batch_beam,  # batch_beam
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            params.fp8_context_fmha,  # fp8_context_fmha
            attn.predicted_tokens_per_seq,  # predicted_tokens_per_seq
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
            params.is_cross,  # is_cross
        )

    def _q35_fold_cp_applies(self, attn, num_requests: int) -> bool:
        """Whether CP-lite may take a generation call of num_requests (it has precedence over the fused path)."""
        group = getattr(attn, "q35_cp_group", None)
        return (
            _Q35_ATTN_CP
            and group is not None
            and not (_Q35_ATTN_CP_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
            and _q35_cp_min_batch(len(group)) <= num_requests <= _Q35_ATTN_CP_MAX_BATCH
        )

    def _q35_fold_generation(self, params: FmhaParams, plan: Q35KvAppendPlan) -> bool:
        """TRTLLM_Q35_ATTN_FOLD: run the folded fused decode for this call when it is exactly a call the fused path
        takes (the conditions of run_generation's fused branch and of _q35_appended_preprocess); False otherwise."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        f = plan.fold
        gate = getattr(attn, "q35_pending_gate", None)
        if not (
            gate is not None
            and _Q35_ATTN_FUSED_DECODE
            and params.fp8_context_fmha
            and meta.num_contexts == 0
            and meta.beam_width == 1
            and not params.is_cross
            and params.seq_offset == 0
            and params.spec_decoding_generation_lengths is None
            and params.num_requests <= _Q35_ATTN_FUSED_MAX_BATCH
            and not (_Q35_ATTN_FUSED_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
            and not self._q35_fold_cp_applies(attn, params.num_requests)
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == params.num_requests
            and plan.q_len == params.input_seq_length
            and fwd.out_scale is None
            and fwd.attention_sinks is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.context_buf is not None
            and params.context_buf.dtype == torch.bfloat16
            and not QuantMode(attn.quant_mode).has_fp4_kv_cache()
            and not getattr(attn, "q35_draft_window", 0)
            and self._multi_processor_count is not None
        ):
            return False
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            self._get_attention_chunk_size(attn),
        )
        if window_left >= 0 or not q35_fused_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.input_seq_length,
            params.num_requests,
            window_left,
            False,
            params.tokens_per_block,
            torch.float8_e4m3fn,
            params.context_buf.dtype,
            plan.kv_pool.dtype,
            self._multi_processor_count,
        ):
            return False
        ws, cnt = _q35_fused_buffers(params.context_buf.device, self._multi_processor_count, False)
        q35_fused_decode_fold(
            f.qkv[: params.num_tokens],
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            f.rotary_dim,
            f.mrope_section,
            plan.kv_pool,
            plan.block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            plan.kv_scale_orig_quant,
            plan.kv_scale_quant_orig,
            plan.host_bmm1_scale,
            params.context_buf,
            ws,
            cnt,
            self._multi_processor_count,
            self._enable_pdl,
        )
        attn.q35_gate_applied = True
        if _Q35_ATTN_FOLD_CHECK and not torch.cuda.is_current_stream_capturing():
            self._q35_fold_check(params, plan, ws, cnt)
        return True

    def _q35_fold_check(self, params: FmhaParams, plan: Q35KvAppendPlan, ws, cnt) -> None:
        from ...modules.fused_ops.fused_qk_norm_rope_gate import q35_fused_qkv_rope_gate_append

        attn = params.attn
        f = plan.fold
        kv_fold = _q35_new_kv_bytes(plan).clone()
        out_fold = params.context_buf.clone()
        _, gate_ref, q_ref, scales_ref = q35_fused_qkv_rope_gate_append(
            f.qkv[: params.num_tokens],
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            f.rotary_dim,
            f.mrope_section,
            plan,
        )
        out_ref = torch.empty_like(params.context_buf)
        q35_fused_decode(
            q_ref,
            plan.kv_pool,
            plan.block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            scales_ref.narrow(0, 0, 2),
            scales_ref.narrow(0, 4, 1),
            out_ref,
            ws,
            cnt,
            self._multi_processor_count,
            self._enable_pdl,
            gate=gate_ref,
        )
        kv_ref = _q35_new_kv_bytes(plan)
        kv_ok = torch.equal(kv_fold, kv_ref)
        out_ok = torch.equal(out_fold.view(torch.int16), out_ref.view(torch.int16))
        Q35_FOLD_CHECK_STATS["calls"] += 1
        if not (kv_ok and out_ok):
            Q35_FOLD_CHECK_STATS["mismatches"] += 1
            diff = (out_fold.float() - out_ref.float()).abs().nan_to_num(0)
            logger.warning(
                f"[Q35_FOLD_CHECK] layer {plan.layer_idx} b{plan.num_requests} q{plan.q_len}: kv bytes equal {kv_ok} "
                f"({int((kv_fold != kv_ref).sum())} differ), output equal {out_ok} (max |d| {diff.max().item():.3e})"
            )
        if Q35_FOLD_CHECK_STATS["calls"] % 200 == 1 or not (kv_ok and out_ok):
            logger.info(f"[Q35_FOLD_CHECK] {Q35_FOLD_CHECK_STATS}")

    @staticmethod
    def _q35_fold_fallback(params: FmhaParams, plan: Q35KvAppendPlan) -> None:
        """A folded call that does not take the fused path: run the prologue kernel the module skipped (the same
        bytes as TRTLLM_Q35_ATTN_KV_APPEND) and continue on that path with its packed QKV and contiguous gate."""
        from ...modules.fused_ops.fused_qk_norm_rope_gate import q35_fused_qkv_rope_gate_append

        attn = params.attn
        f = plan.fold
        plan.fold = None
        qkv_out, gate_out, plan.q_fp8, plan.scales = q35_fused_qkv_rope_gate_append(
            f.qkv,
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            f.rotary_dim,
            f.mrope_section,
            plan,
        )
        params.attention_input = qkv_out[: params.num_tokens]
        params.qkv_input = params.attention_input
        if getattr(attn, "q35_pending_gate", None) is not None:
            attn.q35_pending_gate = gate_out

    def _q35_appended_preprocess(
        self,
        params: FmhaParams,
        plan: Q35KvAppendPlan,
        batch_beam: int,
        attention_chunk_size: int,
    ) -> Optional[tuple]:
        """trtllm_gen_generation_preprocess outputs for a call whose FP8 query / K / V the attention module wrote
        (TRTLLM_Q35_ATTN_KV_APPEND), or None when the call does not match the plan (the C++ path then rewrites the
        same bytes from the bf16 QKV)."""
        attn = params.attn
        fwd = params.fwd
        if not (
            params.fp8_context_fmha
            and plan.q_fp8 is not None
            and plan.scales is not None
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == batch_beam
            and plan.q_len == params.input_seq_length
            and params.seq_offset == 0
            and not params.is_cross
            and params.spec_decoding_generation_lengths is None
            and fwd.out_scale is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.qkv_input is not None
            and params.qkv_input.dtype == torch.bfloat16
        ):
            return None
        layout = _get_generation_workspace_layout(
            params.qkv_input.dtype,
            batch_beam,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.num_kv_heads,
            attn.rope_params.dim,
        )
        offset = int(layout["trtllm_gen_workspace_offset"])
        size = int(layout["trtllm_gen_workspace_size"])
        workspace = params.workspace.view(-1).view(torch.uint8)
        if size <= 0 or offset + size > workspace.numel():
            return None
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            attention_chunk_size,
        )
        return (
            plan.q_fp8,
            plan.kv_pool,
            plan.block_tables,
            None,
            plan.scales.narrow(0, 0, 2),
            plan.scales.narrow(0, 4, 1),
            workspace.narrow(0, offset, size),
            None,
            params.input_seq_length,
            params.max_past_kv_length,
            window_left,
            False,
        )

    @staticmethod
    def _q35_check_appended(
        params: FmhaParams,
        plan: Q35KvAppendPlan,
        appended: tuple,
        stock: tuple,
        kv_before: torch.Tensor,
    ) -> None:
        """TRTLLM_Q35_ATTN_KV_APPEND_CHECK: compare the module-written query / scales / K / V with the C++ path."""
        n = params.num_tokens * params.attn.num_heads * params.attn.head_dim
        stock_q = stock[0].view(torch.uint8).flatten()[:n]
        problems = []
        if not torch.equal(plan.q_fp8.view(torch.uint8).flatten(), stock_q):
            problems.append("query")
        if not torch.equal(
            plan.scales[:2].view(torch.int32), stock[4].flatten()[:2].contiguous().view(torch.int32)
        ):
            problems.append(f"bmm1 {plan.scales[:2].tolist()} vs {stock[4].flatten()[:2].tolist()}")
        if not torch.equal(
            plan.scales[4:5].view(torch.int32),
            stock[5].flatten()[:1].contiguous().view(torch.int32),
        ):
            problems.append(
                f"bmm2 {plan.scales[4:5].tolist()} vs {stock[5].flatten()[:1].tolist()}"
            )
        if not torch.equal(kv_before, _q35_new_kv_bytes(plan)):
            problems.append("kv")
        for idx, name in ((1, "kv_pool"), (2, "block_tables"), (6, "fmha_workspace")):
            a, b = appended[idx], stock[idx]
            if a.data_ptr() != b.data_ptr() or a.shape != b.shape or a.stride() != b.stride():
                problems.append(name)
        if tuple(appended[8:]) != tuple(stock[8:]):
            problems.append(f"scalars {tuple(appended[8:])} vs {tuple(stock[8:])}")
        Q35_KV_APPEND_CHECK_STATS["calls"] += 1
        if problems:
            Q35_KV_APPEND_CHECK_STATS["mismatches"] += 1
            logger.warning(
                f"TRTLLM_Q35_ATTN_KV_APPEND_CHECK mismatch (layer {plan.layer_idx}, tokens {plan.num_tokens}, "
                f"q_len {plan.q_len}): {'; '.join(problems)}"
            )
        elif Q35_KV_APPEND_CHECK_STATS["calls"] % 64 == 1:
            logger.info(f"TRTLLM_Q35_ATTN_KV_APPEND_CHECK: {Q35_KV_APPEND_CHECK_STATS}")

    @staticmethod
    def _q35_cp_exchange_ok(attn, cp: int) -> bool:
        """CP-lite only with its LL exchange (or the NCCL fallback when forced). Created lazily on the first eager
        call (graph warmup); a graph captured before it exists keeps the non-CP path."""
        if _Q35_ATTN_CP_NCCL:
            return True
        mapping = getattr(attn, "q35_cp_mapping", None)
        if mapping is None:
            return False
        from tensorrt_llm._torch.distributed.ll_exchange import get_tp_group_exchange

        return get_tp_group_exchange(mapping, group_size=cp) is not None

    def _q35_cp_generation(
        self,
        params: FmhaParams,
        q_processed: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
        group: List[int],
    ) -> None:
        """CP-lite generation attention (fmha/q35_cp.py) for the rows of this rank's own heads.

        1. all-gather q within the KV group, 2. partial attention of all group heads over this rank's pages (+ the
        draft block on CP rank 0), 3. all-to-all of the per-destination (O, LSE) partials, 4. LSE combine (+ the
        output gate offered by the attention module, TRTLLM_Q35_ATTN_SEG_FUSE_GATE). Exchanges: TPGroupExchange
        (one LL launch each) or NCCL (TRTLLM_Q35_ATTN_CP_NCCL=1, or no peer memory)."""
        attn = params.attn
        cp = len(group)
        cp_rank = attn.q35_cp_rank
        rows, hq, d = q_processed.shape
        dev = q_processed.device
        out = params.context_buf.view(rows, hq, d)
        gate = getattr(attn, "q35_pending_gate", None)
        if gate is not None:
            gate = gate[:rows] if gate.shape[0] >= rows else None
        run_func = flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        bmm1_arg = (
            _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
        )
        sm_count = flashinfer.decode.get_device_sm_count(dev)
        ex = None
        if not _Q35_ATTN_CP_NCCL and getattr(attn, "q35_cp_mapping", None) is not None:
            from tensorrt_llm._torch.distributed.ll_exchange import get_tp_group_exchange

            ex = get_tp_group_exchange(attn.q35_cp_mapping, group_size=cp)
            if ex is not None:
                assert ex.group_size == cp and ex.grank == cp_rank, (
                    "TPGroupExchange group does not match the CP-lite KV group"
                )
        if ex is not None:
            q35_cp_decode_ll(
                ex,
                run_func,
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                max_kv_len,
                params.input_seq_length,
                _q35_cp_segments(params.num_requests, params.input_seq_length),
                bmm1_scale,
                bmm1_arg,
                bmm2_scale,
                fmha_workspace,
                self._q35_counter_buffer,
                sm_count,
                self._enable_pdl,
                out,
                gate=gate,
            )
        else:
            q35_cp_decode_nccl(
                group,
                cp_rank,
                run_func,
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                max_kv_len,
                params.input_seq_length,
                _q35_cp_segments(params.num_requests, params.input_seq_length),
                bmm1_scale,
                bmm1_arg,
                bmm2_scale,
                fmha_workspace,
                self._q35_counter_buffer,
                sm_count,
                self._enable_pdl,
                out,
                gate=gate,
            )
        if gate is not None:
            attn.q35_gate_applied = True

    def run_mla_generation(
        self,
        params: FmhaParams,
    ) -> None:
        """MLA generation decode using flashinfer MLA kernel."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            raise NotImplementedError(
                "Sliding-window attention is not supported by MLA decode path."
            )
        if self._get_attention_chunk_size(attn) != 0:
            raise NotImplementedError("Chunked-attention is not supported by MLA decode path.")

        batch_beam = params.num_requests * meta.beam_width
        if params.attention_input is None:
            raise RuntimeError("MLA generation requires attention_input.")
        kv_cache, block_tables, _kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            attn.local_layer_idx,  # layer_idx
            attn.num_kv_heads,  # num_kv_heads
            params.tokens_per_block,  # tokens_per_block
            attn.head_dim,  # head_dim
            params.kv_factor,  # kv_factor
            params.total_num_blocks,  # total_num_blocks
            attn.quant_mode,  # kv_cache_quant_mode
            params.seq_offset,  # batch_start
            batch_beam,  # batch_size
            params.attention_input.dtype,  # dtype
        )

        pages_per_superblock = 128 // params.tokens_per_block
        if pages_per_superblock > 1:
            num_blocks = block_tables.size(-1)
            remainder = num_blocks % pages_per_superblock
            if remainder != 0:
                pad = pages_per_superblock - remainder
                block_tables = torch.nn.functional.pad(block_tables, (0, pad), value=0)

        kv_lora_rank = attn.kv_lora_rank or 0
        qk_nope_head_dim = attn.qk_nope_head_dim or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        mla_head_dim_qk = kv_lora_rank + qk_rope_head_dim
        q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

        if QuantMode(attn.quant_mode).has_fp8_kv_cache():
            quant_q_buffer = fwd.quant_q_buffer
            bmm1_scale_buffer = fwd.mla_bmm1_scale
            bmm2_scale_buffer = fwd.mla_bmm2_scale
            if quant_q_buffer is None or bmm1_scale_buffer is None or bmm2_scale_buffer is None:
                raise RuntimeError(
                    "FP8 MLA generation requires quant_q_buffer, "
                    "mla_bmm1_scale, and mla_bmm2_scale."
                )

            expected_q_elements = params.num_tokens * attn.num_heads * mla_head_dim_qk
            if quant_q_buffer.numel() < expected_q_elements:
                raise RuntimeError(
                    f"FP8 MLA quant_q_buffer has {quant_q_buffer.numel()} elements; "
                    f"expected at least {expected_q_elements}."
                )
            if bmm1_scale_buffer.dtype != torch.float32 or bmm1_scale_buffer.numel() < 1:
                raise RuntimeError("FP8 MLA bmm1 scale must contain a float32 value.")
            if bmm2_scale_buffer.dtype != torch.float32 or bmm2_scale_buffer.numel() < 1:
                raise RuntimeError("FP8 MLA bmm2 scale must contain a float32 value.")

            query = (
                quant_q_buffer.view(torch.uint8)
                .flatten()[:expected_q_elements]
                .view(torch.float8_e4m3fn)
                .view(batch_beam, q_len_per_req, attn.num_heads, mla_head_dim_qk)
            )
            # FlashInfer converts tensor BMM1 scales to log2 internally. The
            # producer stores the regular scale at index 0 and log2 at index 1.
            bmm1_scale = bmm1_scale_buffer.flatten()[:1]
            bmm2_scale = bmm2_scale_buffer.flatten()[:1]
        else:
            query = params.qkv_input.view(
                batch_beam, q_len_per_req, attn.num_heads, mla_head_dim_qk
            )
            bmm1_scale = 1.0 / (attn.q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
            bmm2_scale = 1.0
        workspace_buffer = params.workspace.view(-1, 4)

        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query,  # query
            kv_cache,  # kv_cache
            workspace_buffer,  # workspace_buffer
            qk_nope_head_dim,  # qk_nope_head_dim
            kv_lora_rank,  # kv_lora_rank
            qk_rope_head_dim,  # qk_rope_head_dim
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            params.max_past_kv_length,  # max_seq_len
            0,  # sparse_mla_top_k
            params.context_buf.view(batch_beam, q_len_per_req, attn.num_heads, kv_lora_rank),  # out
            bmm1_scale,  # bmm1_scale
            bmm2_scale,  # bmm2_scale
            fwd.attention_sinks,  # sinks
            None,  # skip_softmax_threshold_scale_factor
            self._enable_pdl,  # enable_pdl
            "trtllm-gen",  # backend
            True,  # is_var_seq
            self.USE_SHARED_PAGED_KV_IDX,  # uses_shared_paged_kv_idx
            multi_ctas_kv_counter_buffer=self._get_multi_ctas_kv_counter_buffer(),
        )


def q35_kv_append_plan(
    attn: "TrtllmAttention",
    metadata: "TrtllmAttentionMetadata",
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Optional[Q35KvAppendPlan]:
    """Destination for the module-side FP8 q / K / V write of a generation-only step, or None (stock path).

    Mirrors the conditions under which run_generation's C++ preprocess writes FP8 K / V and an FP8 query with a
    uniform q_len (applyBiasRopeUpdateKVCacheV2): trtllm-gen FMHA library first, FP8 KV cache, no MLA / sparse /
    cross attention, beam 1, no multi-token spec-dec layout. run_generation re-checks the call before skipping.
    """
    if not _Q35_ATTN_KV_APPEND or num_tokens <= 0:
        return None
    libs = getattr(attn, "fmha_libs", None)
    if not libs or not isinstance(libs[0], FlashInferTrtllmGenFmha):
        return None
    lib = libs[0]
    if FlashInferTrtllmGenFmha._missing_fused_nanobind_ops():
        return None
    if (
        attn.is_mla_enable
        or attn.sparse_params is not None
        or attn.num_heads != num_heads
        or attn.num_kv_heads != num_kv_heads
        or attn.head_dim != head_dim
        or head_dim % 16 != 0
    ):
        return None
    quant_mode = QuantMode(attn.quant_mode)
    if not quant_mode.has_fp8_kv_cache() or quant_mode.has_fp4_kv_cache():
        return None
    if (
        metadata.kv_cache_block_offsets is None
        or metadata.host_kv_cache_pool_pointers is None
        or metadata.host_kv_cache_pool_mapping is None
        or metadata.kv_cache_manager is None
    ):
        return None
    num_seqs = metadata.num_generations
    if (
        metadata.num_contexts != 0
        or num_seqs <= 0
        or metadata.beam_width != 1
        or metadata.is_cross
        or num_tokens % num_seqs != 0
    ):
        return None
    if metadata.is_spec_decoding_enabled and attn.predicted_tokens_per_seq > 1:
        return None
    seq_lens = metadata.kv_lens_cuda_runtime
    if seq_lens is None or seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
        return None
    kv_scale_orig_quant = attn.kv_scale_orig_quant
    kv_scale_quant_orig = attn.kv_scale_quant_orig
    if not (
        isinstance(kv_scale_orig_quant, torch.Tensor)
        and isinstance(kv_scale_quant_orig, torch.Tensor)
        and kv_scale_orig_quant.dtype == torch.float32
        and kv_scale_quant_orig.dtype == torch.float32
    ):
        return None
    tokens_per_block = metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
    if tokens_per_block <= 0 or tokens_per_block & (tokens_per_block - 1):
        return None
    layer_idx = attn.get_local_layer_idx(metadata)
    kv_pool, block_tables, kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
        metadata.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        metadata.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
        metadata.kv_cache_block_offsets,  # kv_cache_block_offsets
        layer_idx,  # layer_idx
        num_kv_heads,  # num_kv_heads
        tokens_per_block,  # tokens_per_block
        head_dim,  # head_dim
        lib.kv_factor,  # kv_factor
        lib._get_total_num_blocks(metadata),  # total_num_blocks
        attn.quant_mode,  # kv_cache_quant_mode
        0,  # batch_start
        num_seqs,  # batch_size
        torch.bfloat16,  # dtype
    )
    if (
        kv_scale_pool is not None
        or kv_pool.dtype != torch.float8_e4m3fn
        or block_tables.dtype != torch.int32
    ):
        return None
    return Q35KvAppendPlan(
        layer_idx=layer_idx,
        num_tokens=num_tokens,
        num_requests=num_seqs,
        q_len=num_tokens // num_seqs,
        kv_pool=kv_pool,
        block_tables=block_tables,
        seq_lens=seq_lens,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        host_bmm1_scale=FlashInferTrtllmGenFmha._get_bmm1_scale(attn),
    )


def q35_fold_inputs(
    attn: "TrtllmAttention",
    plan: Q35KvAppendPlan,
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rotary_dim: int,
    mrope_section: Optional[tuple],
) -> Optional[Q35FoldInputs]:
    """TRTLLM_Q35_ATTN_FOLD: the fold inputs for a generation-only call whose KV-append plan is ``plan`` when the call
    will take the fused decode path (then the module skips its prologue kernel), else None. run_generation re-checks
    the call and runs the prologue kernel itself if the call goes elsewhere after all."""
    if not (_Q35_ATTN_FOLD and _Q35_ATTN_FUSED_DECODE):
        return None
    libs = getattr(attn, "fmha_libs", None)
    if not libs or not isinstance(libs[0], FlashInferTrtllmGenFmha):
        return None
    lib = libs[0]
    if lib._multi_processor_count is None:
        lib._multi_processor_count = lib._get_multi_processor_count(qkv.device)
    if (
        plan.num_requests > _Q35_ATTN_FUSED_MAX_BATCH
        or (_Q35_ATTN_FUSED_SKIP_MTP and getattr(attn, "q35_is_mtp", False))
        or getattr(attn, "q35_draft_window", 0)
        or lib._q35_fold_cp_applies(attn, plan.num_requests)
        or plan.num_tokens != qkv.shape[0]
        or plan.num_tokens != plan.num_requests * plan.q_len
    ):
        return None
    if mrope_section is not None and positions.numel() != 3 * qkv.shape[0]:
        return None
    if not q35_fused_decode_supported(
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        plan.q_len,
        plan.num_requests,
        -1,
        False,
        plan.kv_pool.shape[2],
        torch.float8_e4m3fn,
        torch.bfloat16,
        plan.kv_pool.dtype,
        lib._multi_processor_count,
    ) or not q35_fused_decode_fold_supported(
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        rotary_dim,
        plan.q_len,
        qkv,
        positions,
        cos_sin,
    ):
        return None
    if not (
        q_weight.dtype == torch.bfloat16
        and k_weight.dtype == torch.bfloat16
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and q_weight.numel() == attn.head_dim
        and k_weight.numel() == attn.head_dim
    ):
        return None
    return Q35FoldInputs(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        positions=positions,
        eps=float(eps),
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section) if mrope_section is not None else None,
    )


def _q35_new_kv_bytes(plan: Q35KvAppendPlan) -> torch.Tensor:
    """Cache bytes at the new tokens' K / V slots: uint8 [2, num_requests, q_len, num_kv_heads, head_dim]."""
    dev = plan.kv_pool.device
    page = plan.kv_pool.shape[2]
    pos = plan.seq_lens[: plan.num_requests].to(torch.int64).unsqueeze(1) - plan.q_len
    pos = (pos + torch.arange(plan.q_len, device=dev)).clamp(min=0)
    tables = plan.block_tables[: plan.num_requests].to(torch.int64)
    pages = (pos // page).clamp(max=tables.shape[2] - 1)
    slots = pos % page
    pool = plan.kv_pool.view(torch.uint8)
    k_blocks = torch.gather(tables[:, 0, :], 1, pages)
    v_blocks = torch.gather(tables[:, 1, :], 1, pages)
    return torch.stack((pool[k_blocks, :, slots, :], pool[v_blocks, :, slots, :]))
