# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import functools
import os
import weakref
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    Q35_GDN_REPLAY_MAX_T,
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.q35_conv_replay import (
    q35_conv_replay_supported,
    q35_conv_replay_update,
)
from tensorrt_llm._torch.modules.fla.q35_gdn_core import (
    q35_gdn_core_ready,
    q35_gdn_core_supported,
    q35_gdn_core_update,
)
from tensorrt_llm._torch.modules.fla.q35_gdn_fused import (
    q35_gdn_fused_supported,
    q35_gdn_fused_update,
)
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    _can_use_flashinfer_gdn_verify,
    _flashinfer_gdn_verify,
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention_backend import AttentionMetadata
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ...speculative import SpecMetadata
from ...utils import EventType, get_model_extra_attrs, is_gdn_replay_enabled, is_torch_compiling
from ..linear import FP8QDQLinearMethod, Linear, TensorParallelMode
from ..lowm_linear import default_tactic as lowm_default_tactic
from ..lowm_linear import (
    lowm_check,
    lowm_conv_supported,
    lowm_fp8_act,
    lowm_gdn_ba_enabled,
    lowm_gemm,
    lowm_rows_ok,
    lowm_skipped,
    stock_fp8_linear,
)
from ..multi_stream_utils import do_multi_stream, maybe_execute_in_parallel
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
    pack_gdn_decode_qkv,
)
from .layernorm_gated import RMSNorm as RMSNormGated
from .layernorm_gated import rms_norm_gated_token_major
from .mamba2_metadata import Mamba2Metadata
from .mamba2_mixer import _cached_arange

# Qwen3.5 low-latency switches (default off).
# TRTLLM_Q35_GDN_ARANGE_CACHE=1: build the verify-path intermediate-state row indices once per step from the
# cache manager's persistent arange (a view, no kernel) instead of one torch.arange launch per GDN layer
# (port of upstream 1374e96c89).
_Q35_GDN_ARANGE_CACHE = os.environ.get("TRTLLM_Q35_GDN_ARANGE_CACHE", "0") == "1"
# TRTLLM_Q35_GDN_FUSED_CONV_REPLAY=1: on the small-batch cached-replay target-verify path, run the causal conv1d
# update and the cached replay as one kernel (modules/fla/q35_conv_replay.py; bitwise-equal outputs).
_Q35_GDN_FUSED_CONV_REPLAY = os.environ.get("TRTLLM_Q35_GDN_FUSED_CONV_REPLAY", "0") == "1"
# TRTLLM_Q35_GDN_CONV_PDL=1: launch the target-verify conv1d update as a PDL dependent of the in_proj GEMM (reads
# in_proj output and intermediate row indices only after griddepcontrol.wait).
_Q35_GDN_CONV_PDL = os.environ.get("TRTLLM_Q35_GDN_CONV_PDL", "0") == "1"
# TRTLLM_Q35_GDN_FUSE (default 0): on the decode-only MTP target-verify cached-replay path (fewer than 16 requests),
# fuse the GDN chain between in_proj and out_proj (modules/fla/q35_gdn_fused.py):
#   1 = conv1d update + cached replay in one launch, then the gated-RMSNorm kernel;
#   2 = conv1d update + cached replay + gated RMSNorm (+ FP8 quantization for out_proj) in one launch;
#   3 = the conv1d update kernel, then cached replay + gated RMSNorm (+ FP8) in one launch;
#   4 = the conv1d update in the low-M in_proj's epilogue (TRTLLM_Q35_LOWM_GDN_BA path), then the fused replay +
#       gated RMSNorm (+ FP8); steps whose in_proj does not take the low-M kernel run as 3.
# TRTLLM_Q35_GDN_FUSE_OPTS (b < LARGE_FROM) / TRTLLM_Q35_GDN_FUSE_OPTS_LARGE (b >= LARGE_FROM, default: the same):
# launch options; TRTLLM_Q35_GDN_FUSE_LARGE_FROM (default 3) is the first decode batch size that takes the LARGE set.
# Options are "_"-separated: bv16 / bv32 / bv64 / bv128 (value tile), w2 / w4 / w8 (warps), p (SSM state fetched
# before the PDL wait), s (both history buffers fetched by slot, no dependent round trip), t0 / t1 / t2 (PDL
# dependents released at exit / after the wait / at entry), x2 (forward substitution on row-vector tuples, FMA-only
# serial chain), z (gate rows fetched right after the wait), n (with bv128: the norm reloads the stored rows, i.e. the
# norm kernel's summation order). TRTLLM_Q35_GDN_FUSE_MAX_BATCH (default 15): larger decode batches keep the separate
# kernels.
_Q35_GDN_FUSE = int(os.environ.get("TRTLLM_Q35_GDN_FUSE", "0"))
_Q35_GDN_FUSE_MAX_BATCH = int(os.environ.get("TRTLLM_Q35_GDN_FUSE_MAX_BATCH", "15"))


def _q35_gdn_fuse_opts(spec: str) -> dict:
    opts = {
        "block_v": None,
        "num_warps": 2,
        "prefetch_state": False,
        "trigger": 0,
        "spec_hist": False,
        "solve": 0,
        "z_prefetch": False,
        "norm_reload": False,
    }
    for tok in spec.replace(",", "_").split("_"):
        if not tok:
            continue
        if tok.startswith("bv"):
            opts["block_v"] = int(tok[2:])
        elif tok.startswith("w"):
            opts["num_warps"] = int(tok[1:])
        elif tok == "p":
            opts["prefetch_state"] = True
        elif tok == "s":
            opts["spec_hist"] = True
        elif tok.startswith("t"):
            opts["trigger"] = int(tok[1:])
        elif tok.startswith("x"):
            opts["solve"] = int(tok[1:])
        elif tok == "z":
            opts["z_prefetch"] = True
        elif tok == "n":
            opts["norm_reload"] = True
        else:
            raise ValueError(f"TRTLLM_Q35_GDN_FUSE_OPTS: unknown option {tok!r}")
    return opts


_Q35_GDN_FUSE_OPTS = (
    _q35_gdn_fuse_opts(os.environ.get("TRTLLM_Q35_GDN_FUSE_OPTS", "")) if _Q35_GDN_FUSE else {}
)
_Q35_GDN_FUSE_OPTS_LARGE = (
    _q35_gdn_fuse_opts(
        os.environ.get(
            "TRTLLM_Q35_GDN_FUSE_OPTS_LARGE", os.environ.get("TRTLLM_Q35_GDN_FUSE_OPTS", "")
        )
    )
    if _Q35_GDN_FUSE
    else {}
)
_Q35_GDN_FUSE_LARGE_FROM = int(os.environ.get("TRTLLM_Q35_GDN_FUSE_LARGE_FROM", "3"))
# TRTLLM_Q35_GDN_CORE (with TRTLLM_Q35_GDN_FUSE=4): the fused replay + gated RMSNorm (+ FP8) on convolved q|k|v rows
# runs as the CUDA kernel of modules/fla/q35_gdn_core.py (one CTA per (request, value head), in-CTA norm, checkpoint
# commit computed before the PDL wait) instead of the Triton kernel; shapes it does not cover (T > 8, history != 16,
# FP32 state) keep the Triton kernel. 1 = on the FUSE=4 path only (conv in the low-M in_proj epilogue: the kernel
# launches at the in_proj's entry and its prefetch / commit run during the in_proj); 2 = also on the FUSE=3 path and
# the FUSE=4 fallback (separate conv kernel, e.g. TEP8 b4 with the nvjet in_proj: there the pre-wait window is only
# the conv kernel, and it was measurably slower at TEP8 b4). TRTLLM_Q35_GDN_CORE_TRIGGER: when its
# PDL dependents are released (2: at entry, as FUSE_OPTS t2; 3: after the final CTA barrier; 0: at exit).
_Q35_GDN_CORE = int(os.environ.get("TRTLLM_Q35_GDN_CORE", "0"))
_Q35_GDN_CORE_TRIGGER = int(os.environ.get("TRTLLM_Q35_GDN_CORE_TRIGGER", "2"))
# TRTLLM_Q35_GDN_CORE_MIN_BATCH=N (with TRTLLM_Q35_GDN_CORE=1; default 0 = off): also take the CUDA core kernel on the
# FUSE=3 path and the FUSE=4 fallback (separate conv kernel) for decode batches of at least N requests, i.e. the
# TRTLLM_Q35_GDN_CORE=2 policy above a batch threshold. At TEP8 the in_proj leaves the low-M kernel from b5 on, so b5-b8
# run the Triton fused kernel between the b <= 4 (FUSE=4) and b >= 9 (GDN_CORE_LARGE) CUDA-core ranges; standalone
# (microbenchmark, GB300) Triton is 6.8 / 7.0 / 7.2 / 11.0 us at b5-b8 and the core 2.7-2.8 us.
_Q35_GDN_CORE_MIN_BATCH = int(os.environ.get("TRTLLM_Q35_GDN_CORE_MIN_BATCH", "0"))
# TRTLLM_Q35_GDN_CORE_LARGE=1 (with TRTLLM_Q35_GDN_FUSE=3/4, TRTLLM_Q35_GDN_FUSE_MAX_BATCH and
# TRTLLM_Q35_GDN_REPLAY_PARTITION_MIN_BATCH above the batch): decode batches beyond the Triton fused kernel (more than
# 64 (request, value head) pairs, i.e. b > 8 at TP8) run the separate conv1d update and then the CUDA core kernel
# (replay + in-layer checkpoint commit + gated RMSNorm + FP8 in one launch, one CTA per pair) instead of the stock
# conv -> replay -> gated-norm chain (+ the all-layer commit at b >= 16).
_Q35_GDN_CORE_LARGE = os.environ.get("TRTLLM_Q35_GDN_CORE_LARGE", "0") == "1"
_q35_core_min_batch_logged: set = set()
if _Q35_GDN_CORE_MIN_BATCH:
    logger.info(f"TRTLLM_Q35_GDN_CORE_MIN_BATCH={_Q35_GDN_CORE_MIN_BATCH} (TRTLLM_Q35_GDN_CORE={_Q35_GDN_CORE}): "
                f"CUDA core kernel on the separate-conv GDN verify path from b{_Q35_GDN_CORE_MIN_BATCH}"
                + ("" if _Q35_GDN_CORE == 1 else " [inactive: needs TRTLLM_Q35_GDN_CORE=1]"))
# TRTLLM_Q35_GDN_BA_LATE_JOIN=1 (default off; with TRTLLM_Q35_GDN_FUSE=3/4): when in_proj_ba runs on the aux stream next
# to the in_proj_qkvz GEMM (the stock projections, i.e. decode batches above the low-M in_proj's rows: TEP8 b >= 5),
# the main stream joins the aux stream right before the first consumer of a / b (the fused replay kernel, or the stock
# chain when the fused path does not apply) instead of right after the qkvz GEMM, so the separate target-verify conv1d
# update (it reads q|k|v only) no longer waits for the BF16 ba GEMM + split-K reduce. Stream order only: bitwise the
# same outputs. Only while multi-stream is on (CUDA graph capture), as maybe_execute_in_parallel.
_Q35_GDN_BA_LATE_JOIN = os.environ.get("TRTLLM_Q35_GDN_BA_LATE_JOIN", "0") == "1"
_q35_ba_late_join_logged: set = set()
if _Q35_GDN_BA_LATE_JOIN:
    logger.info(f"TRTLLM_Q35_GDN_BA_LATE_JOIN=1 (TRTLLM_Q35_GDN_FUSE={_Q35_GDN_FUSE}): GDN in_proj_ba (aux stream) "
                f"joined after the separate target-verify conv1d update"
                + ("" if _Q35_GDN_FUSE in (3, 4) else " [inactive: needs TRTLLM_Q35_GDN_FUSE=3 or 4]"))


# FlashInfer GDN prefill is ON by default; set TLLM_USE_FLASHINFER_GDN_PREFILL=0
# to force the vendored Triton chunk_gated_delta_rule everywhere. FlashInfer only
# ships the GDN prefill kernel for Hopper (SM90) and datacenter Blackwell
# (SM100/SM103); on consumer Blackwell (SM120) and other archs it aborts at
# launch, so we fall back to Triton there. Resolution is deferred to first call
# (and cached) so importing this module never initializes CUDA.
@functools.lru_cache(maxsize=1)
def _resolve_chunk_gated_delta_rule():
    if (
        os.getenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1") == "1"
        and is_flashinfer_gdn_supported_arch()
    ):
        from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as impl
    else:
        from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as impl
    return impl


@torch.compiler.disable
def chunk_gated_delta_rule(*args, **kwargs):
    return _resolve_chunk_gated_delta_rule()(*args, **kwargs)


def _q35_split_prefill_core(query, key, value, g, beta, ssm_states, split, stage_ssm, output):
    """Chunked GDN prefill of a context-only batch with in-forward snapshots (TRTLLM_MAMBA_INFORWARD_SNAPSHOT;
    split = Mamba2Metadata.q35_split).

    Each planned sequence is split into [.., P) and its tail. One chunked prefill over all sequences (initial states
    gathered explicitly, final states returned) gives every sequence's final state, written to the live slots: the
    state at P for the planned requests. The tails are then recomputed from the state at P, as the separate tail
    iteration did; their outputs replace the first pass's and their final states (the state at the prompt end) go to
    the staging rows, from which the cache manager restores the live slots after committing the snapshot at P.
    """
    initial = ssm_states.index_select(0, split.gather)
    core_out, final = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=split.cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        output=output,
    )
    final = final.to(ssm_states.dtype)
    ssm_states.index_copy_(0, split.live, final.index_select(0, split.main_rows))
    tail = split.tail_tok
    tail_out, tail_final = chunk_gated_delta_rule(
        q=query.index_select(1, tail),
        k=key.index_select(1, tail),
        v=value.index_select(1, tail),
        g=g.index_select(1, tail),
        beta=beta.index_select(1, tail),
        initial_state=final.index_select(0, split.split_rows),
        output_final_state=True,
        cu_seqlens=split.cu_tail,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    stage_ssm.index_copy_(0, split.stage_rows, tail_final.to(stage_ssm.dtype))
    core_out[0].index_copy_(0, tail, tail_out[0].to(core_out.dtype))
    return core_out


def _extract_gdn_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata)

    gdn_layers = extra_attrs.get("gdn_layers", None)
    assert gdn_layers is not None, "GDN layer is not registered"
    gdn_layer_ref = gdn_layers.get(layer_idx, None)
    assert gdn_layer_ref is not None, f"Cannot find GDN layer for layer {layer_idx}"
    gdn_layer = gdn_layer_ref()
    assert isinstance(gdn_layer, Qwen3NextGatedDeltaNet)

    return metadata, gdn_layer, extra_attrs.get("spec_metadata", None)


@triton.jit
def _reset_gdn_states_kernel(
    ssm_states,
    conv_states,
    state_indices,
    has_initial_states,
    ssm_state_stride,
    conv_state_stride,
    NUM_CACHE_LINES: tl.constexpr,
    SSM_STATE_SIZE: tl.constexpr,
    CONV_STATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request_idx = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_idx = tl.load(state_indices + request_idx).to(tl.int64)
    needs_reset = ~tl.load(has_initial_states + request_idx).to(tl.int1)
    valid_state = (state_idx >= 0) & (state_idx < NUM_CACHE_LINES)
    ssm_row_offset = state_idx * ssm_state_stride.to(tl.int64)
    conv_row_offset = state_idx * conv_state_stride.to(tl.int64)

    tl.store(
        ssm_states + ssm_row_offset + offsets,
        0.0,
        mask=needs_reset & valid_state & (offsets < SSM_STATE_SIZE),
    )
    tl.store(
        conv_states + conv_row_offset + offsets,
        0.0,
        mask=needs_reset & valid_state & (offsets < CONV_STATE_SIZE),
    )


def _reset_gdn_states(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
) -> None:
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return

    ssm_state_size = ssm_states.numel() // ssm_states.shape[0]
    conv_state_size = conv_states.numel() // conv_states.shape[0]
    block_size = 256
    grid = (
        num_requests,
        triton.cdiv(max(ssm_state_size, conv_state_size), block_size),
    )
    _reset_gdn_states_kernel[grid](
        ssm_states,
        conv_states,
        state_indices,
        has_initial_states,
        ssm_states.stride(0),
        conv_states.stride(0),
        ssm_states.shape[0],
        ssm_state_size,
        conv_state_size,
        block_size,
    )


def _q35_direct_call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


# TRTLLM_Q35_CUSTOM_OP_EAGER=1: run the body of this piecewise-excluded custom op with dynamo's eval-frame hook off.
# The op executes inside the torch.compile'd region, where every Python frame it creates (backend wrappers, Triton
# launchers, FlashInfer glue) otherwise goes through dynamo's frame callback.
_q35_op_call = (torch._dynamo.disable(_q35_direct_call, recursive=True)
                if os.environ.get("TRTLLM_Q35_CUSTOM_OP_EAGER", "0") == "1" else _q35_direct_call)


# TRTLLM_Q35_GDN_PREFILL_GRAPH=1: under the piecewise torch.compile path the GDN core (state reset, conv1d,
# post-conv, FlashInfer chunked prefill) runs eagerly between the captured pieces, and its ~8 Python-launched
# kernels cost ~1 ms of host time per layer. For context-only batches of up to
# TRTLLM_Q35_GDN_PREFILL_GRAPH_MAX_TOKENS (padded) tokens the core is captured as one CUDA graph per (layer, padded
# token count, number of context requests) and replayed:
# - Token inputs and the output go through static buffers shared by all layers (copied in / out around each
#   replay): the piece outputs that feed the op do not keep fixed addresses on every rank.
# - The graph bakes in the addresses of the Mamba metadata buffers (has_initial_states, state_indices,
#   query_start_loc(_long)) and of the layer's state pools; a replay requires them unchanged, otherwise the key is
#   re-warmed and re-captured (e.g. a graph captured during the engine warmup on dummy-request buffers).
# The graph runs on the padded token count: cu_seqlens / query_start_loc bound the work to the real tokens, as on
# the eager path. The first call of each key runs eagerly (Triton / FlashInfer specializations, workspaces), the
# second captures. Mixed and generation batches stay eager.
# TRTLLM_Q35_GDN_PREFILL_GRAPH_STATS=N logs the replay / capture counters every N calls.
_Q35_GDN_PREFILL_GRAPH = os.environ.get("TRTLLM_Q35_GDN_PREFILL_GRAPH", "0") == "1"
_Q35_GDN_PREFILL_GRAPH_WARMUP = int(os.environ.get("TRTLLM_Q35_GDN_PREFILL_GRAPH_WARMUP", "1"))
_Q35_GDN_PREFILL_GRAPH_MAX_TOKENS = int(os.environ.get("TRTLLM_Q35_GDN_PREFILL_GRAPH_MAX_TOKENS", "8192"))
_Q35_GDN_PREFILL_GRAPH_STATS = int(os.environ.get("TRTLLM_Q35_GDN_PREFILL_GRAPH_STATS", "0"))
# TRTLLM_Q35_GDN_GRAPH_LEAN=1 (with TRTLLM_Q35_GDN_PREFILL_GRAPH): the per-call key check reads the layer's state pool
# addresses from the hybrid manager's per-layer state lists instead of building a mamba_layer_cache() dataclass (two
# tensor selects and a frozen dataclass per GDN layer and iteration).
_Q35_GDN_GRAPH_LEAN = os.environ.get("TRTLLM_Q35_GDN_GRAPH_LEAN", "0") == "1"
# TRTLLM_Q35_GDN_GRAPH_DIRECT=1 (with TRTLLM_Q35_GDN_PREFILL_GRAPH): a key whose token inputs and output have the same
# addresses and strides in its warm-up call and in its capture call is captured on those tensors and replayed without
# the four copies through the static buffers while they keep them; a call on other addresses uses the static-buffer
# graph of the key (captured on first need), as without the switch.
_Q35_GDN_GRAPH_DIRECT = os.environ.get("TRTLLM_Q35_GDN_GRAPH_DIRECT", "0") == "1"
_q35_gdn_prefill_graphs = {}
_q35_gdn_prefill_graph_pool = None
_q35_gdn_prefill_static = {}
_q35_gdn_prefill_stats = collections.Counter()


def _q35_gdn_state_ptrs(kv_cache_manager, layer_idx: int):
    """Addresses of a GDN layer's conv / SSM state pools (TRTLLM_Q35_GDN_GRAPH_LEAN).

    The hybrid V2 manager keeps the per-layer state views in lists (``all_conv_states`` / ``all_ssm_states``, the
    tensors ``mamba_layer_cache()`` returns); other managers go through ``mamba_layer_cache()``.
    """
    convs = getattr(kv_cache_manager, "all_conv_states", None)
    if convs:
        off = kv_cache_manager.mamba_layer_offsets[layer_idx]
        return convs[off].data_ptr(), kv_cache_manager.all_ssm_states[off].data_ptr()
    layer_cache = kv_cache_manager.mamba_layer_cache(layer_idx)
    return layer_cache.conv.data_ptr(), layer_cache.temporal.data_ptr()


def _q35_gdn_io_sig(mixed_qkv, a, b, output):
    """Addresses and strides of the GDN core's token inputs and output (TRTLLM_Q35_GDN_GRAPH_DIRECT)."""
    return (mixed_qkv.data_ptr(), mixed_qkv.stride(), a.data_ptr(), a.stride(), b.data_ptr(), b.stride(),
            output.data_ptr(), output.stride())


def _q35_gdn_capture(body, args) -> torch.cuda.CUDAGraph:
    """Capture body(*args) into a CUDA graph on the shared GDN prefill graph pool (no replay)."""
    global _q35_gdn_prefill_graph_pool
    if _q35_gdn_prefill_graph_pool is None:
        _q35_gdn_prefill_graph_pool = torch.cuda.graph_pool_handle()
    import gc
    from unittest.mock import patch
    graph = torch.cuda.CUDAGraph()
    # torch.cuda.graph() runs torch.cuda.synchronize(), gc.collect() and torch.cuda.empty_cache() on entry, and
    # these captures happen while serving. gc.collect() is slow (the piecewise runner patches it out the same
    # way), and empty_cache() hands cached blocks back to the driver under the KV transceiver's (NIXL / UCX)
    # memory registrations: a disagg CTX segfaulted in nixlUcxWorker::arm right after its first lazy captures.
    with patch.object(gc, "collect", lambda *args, **kwargs: 0), patch.object(torch.cuda, "empty_cache",
                                                                               lambda: None):
        with torch.cuda.graph(graph, pool=_q35_gdn_prefill_graph_pool, capture_error_mode="thread_local"):
            body(*args)
    return graph


def _q35_gdn_static_view(role: str, t: torch.Tensor, token_dim: int) -> Optional[torch.Tensor]:
    """A view shaped and strided like t over a persistent buffer sized for _Q35_GDN_PREFILL_GRAPH_MAX_TOKENS tokens.

    One buffer per (role, dtype, device, layout of the other dims); None if t does not fit it.
    """
    layout = tuple((sz, st) for i, (sz, st) in enumerate(zip(t.shape, t.stride())) if i != token_dim)
    row = t.stride(token_dim)
    key = (role, t.dtype, t.device, layout, row)
    buf = _q35_gdn_prefill_static.get(key)
    if buf is None:
        extent = 1 + sum((sz - 1) * st for sz, st in layout)
        buf = torch.empty((_Q35_GDN_PREFILL_GRAPH_MAX_TOKENS - 1) * row + extent, dtype=t.dtype, device=t.device)
        _q35_gdn_prefill_static[key] = buf
    if 1 + sum((sz - 1) * st for sz, st in zip(t.shape, t.stride())) > buf.numel():
        return None
    return buf.as_strided(t.shape, t.stride())


def _q35_gdn_prefill_graphed(gdn_layer, mixed_qkv, a, b, layer_idx, output, attn_metadata, spec_metadata) -> bool:
    """Run a context-only GDN core through its per-shape CUDA graph; False: not applicable, run it eagerly."""
    num_ctx = attn_metadata.num_contexts
    num_tokens = mixed_qkv.shape[0]
    if (num_ctx == 0 or attn_metadata.seq_lens.shape[0] != num_ctx
            or num_tokens > _Q35_GDN_PREFILL_GRAPH_MAX_TOKENS or torch.cuda.is_current_stream_capturing()):
        return False
    stats = _q35_gdn_prefill_stats
    if _Q35_GDN_PREFILL_GRAPH_STATS:
        stats["call"] += 1
        if stats["call"] % _Q35_GDN_PREFILL_GRAPH_STATS == 0:
            logger.info(f"[q35 gdn prefill graph] {dict(stats)} keys={len(_q35_gdn_prefill_graphs)}")
    mamba_metadata = attn_metadata.mamba_metadata
    if _Q35_GDN_GRAPH_LEAN:
        conv_ptr, ssm_ptr = _q35_gdn_state_ptrs(attn_metadata.kv_cache_manager, gdn_layer.layer_idx)
    else:
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(gdn_layer.layer_idx)
        conv_ptr, ssm_ptr = layer_cache.conv.data_ptr(), layer_cache.temporal.data_ptr()
    meta = (mamba_metadata.has_initial_states.data_ptr(), mamba_metadata.state_indices.data_ptr(),
            mamba_metadata.query_start_loc.data_ptr(), mamba_metadata.query_start_loc_long.data_ptr(),
            conv_ptr, ssm_ptr)
    # TRTLLM_MAMBA_INFORWARD_SNAPSHOT: a split batch runs more kernels, sized by (requests split, tail tokens)
    split_sig = getattr(mamba_metadata, "q35_split_sig", 0)
    if split_sig:
        meta += (mamba_metadata._q35_buf.data_ptr(),)
    key = (layer_idx, num_tokens, num_ctx, split_sig)
    entry = _q35_gdn_prefill_graphs.get(key)
    if entry is not None and entry[2] != meta:
        stats["meta_change"] += 1
        entry = None
    if entry is None:
        # [eager warmup calls, static-buffer graph, metadata addresses, static views,
        #  direct graph, its token I/O signature, token I/O signature of the warm-up call] (the last three:
        #  TRTLLM_Q35_GDN_GRAPH_DIRECT)
        entry = _q35_gdn_prefill_graphs[key] = [0, None, meta, None, None, None, None]

    def body(qkv, a_, b_, out):
        gdn_layer.forward_core(qkv, a_, b_, attn_metadata, mamba_metadata, spec_metadata=spec_metadata, output=out)

    if _Q35_GDN_GRAPH_DIRECT:
        sig = _q35_gdn_io_sig(mixed_qkv, a, b, output)
        if entry[4] is not None:
            if sig == entry[5]:
                entry[4].replay()
                stats["direct_replay"] += 1
                return True
            stats["direct_moved"] += 1
        elif entry[1] is None:
            if entry[0] < _Q35_GDN_PREFILL_GRAPH_WARMUP:
                entry[0] += 1
                entry[6] = sig
                stats["warmup"] += 1
                body(mixed_qkv, a, b, output)
                return True
            if sig == entry[6]:
                graph = _q35_gdn_capture(body, (mixed_qkv, a, b, output))
                entry[4], entry[5] = graph, sig
                stats["capture_direct"] += 1
                graph.replay()
                return True
    if entry[1] is None and entry[0] < _Q35_GDN_PREFILL_GRAPH_WARMUP:
        entry[0] += 1
        stats["warmup"] += 1
        body(mixed_qkv, a, b, output)
        return True
    if entry[1] is None:
        views = (_q35_gdn_static_view("qkv", mixed_qkv, 0), _q35_gdn_static_view("a", a, 0),
                 _q35_gdn_static_view("b", b, 0), _q35_gdn_static_view("out", output, 1))
        if any(v is None for v in views):
            stats["no_layout"] += 1
            return False
        for v, t in zip(views[:3], (mixed_qkv, a, b)):
            v.copy_(t)
        graph = _q35_gdn_capture(body, views)
        entry[1], entry[3] = graph, views
        stats["capture"] += 1
        graph.replay()
        output.copy_(views[3])
        return True
    views = entry[3]
    for v, t in zip(views[:3], (mixed_qkv, a, b)):
        v.copy_(t)
    entry[1].replay()
    output.copy_(views[3])
    stats["replay"] += 1
    return True


@torch.library.custom_op("trtllm::gdn_custom_op_inplace", mutates_args=("output",))
def gdn_custom_op_inplace(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    attn_metadata, gdn_layer, spec_metadata = _extract_gdn_extra_attrs(layer_idx)
    if _Q35_GDN_PREFILL_GRAPH and _q35_op_call(
        _q35_gdn_prefill_graphed, gdn_layer, mixed_qkv, a, b, layer_idx, output, attn_metadata, spec_metadata
    ):
        return
    num_tokens = attn_metadata.num_tokens
    _q35_op_call(
        gdn_layer.forward_core,
        mixed_qkv[:num_tokens],
        a[:num_tokens],
        b[:num_tokens],
        attn_metadata,
        attn_metadata.mamba_metadata,
        spec_metadata=spec_metadata,
        output=output[:, :num_tokens, :, :],
    )


# Under a breakable prefill CUDA graph (TRTLLM_PREFILL_BCG) the GDN core runs eagerly between the
# captured segments; outside a capture the wrapper is a plain call.
maybe_bcg_gdn_custom_op_inplace = eager_on_graph(gdn_custom_op_inplace)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    stride_a_row,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_d = tl.program_id(0), tl.program_id(1)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    # a may be a row-strided view sliced out of the packed ba projection;
    # g is always allocated packed.
    off_a = i_b * stride_a_row + head_off
    off_g = i_b * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off_a, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off_g, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    grid = (batch, triton.cdiv(num_heads, 8))
    g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, a.stride(0), num_heads, beta, threshold, 8, num_warps=1
    )
    return g


def _verify_intermediate_state_indices(
    kv_cache_manager, num_decodes: int, device: torch.device
) -> torch.Tensor:
    """Rows [0, num_decodes) of the batch-scoped intermediate buffers, as a view of a persistent arange.

    The verify path writes each decode request's per-draft-token states into rows [0, num_decodes) of the
    intermediate conv / SSM buffers and update_mamba_states() reads the same rows back, so the index vector is
    the same for every GDN layer of a step.
    """
    indices = getattr(kv_cache_manager, "intermediate_state_indices", None)
    if indices is None or indices.shape[0] < num_decodes or indices.device != device:
        indices = _cached_arange(max(kv_cache_manager.get_max_resource_count(), num_decodes), device)
    return indices[:num_decodes]


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.pretrained_config = config
        replay_enabled = is_gdn_replay_enabled()
        if replay_enabled:
            logger.info_once(
                "GDN MTP replay is requested; set TRTLLM_USE_GDN_REPLAY=0 to disable it",
                key="gdn_mtp_replay_cached",
            )
        else:
            logger.info_once(
                "GDN MTP replay is disabled; set TRTLLM_USE_GDN_REPLAY=1 to enable it",
                key="gdn_mtp_replay_disabled",
            )

        # tensor parallel
        tp_size = model_config.mapping.tp_size
        pp_size = model_config.mapping.pp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=model_config.mapping.rank,
            gpus_per_node=model_config.mapping.gpus_per_node,
            enable_attention_dp=model_config.mapping.enable_attention_dp,
        )
        self.mapping = mapping

        self.attn_tp_rank = mapping.tp_rank
        self.attn_tp_size = mapping.tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.num_k_heads_per_tp = divide(self.num_k_heads, self.attn_tp_size)
        self.num_v_heads_per_tp = divide(self.num_v_heads, self.attn_tp_size)
        self.key_dim_per_tp = self.head_k_dim * self.num_k_heads_per_tp
        self.value_dim_per_tp = self.head_v_dim * self.num_v_heads_per_tp
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.register_to_config = False
        if model_config is not None:
            if "gdn_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["gdn_layers"] = {}
            suffix = 0
            while self.layer_idx_str in model_config.extra_attrs["gdn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            model_config.extra_attrs["gdn_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Linear(
            self.conv_kernel_size,
            self.conv_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.in_proj_qkvz = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )
        self.in_proj_ba = Linear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(
                (self.num_v_heads // self.attn_tp_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size), dtype=torch.float32).uniform_(
            0, 16
        )
        self.A_log = nn.Parameter(
            torch.log(A),
            requires_grad=False,
        )
        self.A_log._no_weight_decay = True

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.cuda.current_device(),
            dtype=config.torch_dtype,
        )

        # gemmaNorm is not supported in fused_all_reduce kernel.
        # So, we need to do allReduce in Linear and do gemmaNorm in separate kernel.
        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            reduce_output=True,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.Attention]}
        self.aux_stream = aux_stream

        # Decode-sized projections on the low-M kernel (TRTLLM_Q35_LOWM_FP8 / _BF16 / _GDN_BA,
        # read per call; inert when off).
        self.in_proj_qkvz._q35_lowm = "gdn.in_proj_qkvz"
        self.out_proj._q35_lowm = "gdn.out_proj"
        self.in_proj_ba._q35_lowm_bf16 = "gdn.in_proj_ba"

    def cache_derived_state(self) -> None:
        """Attach downstream static quantization state after loading weights."""
        self.norm.fp8_scale = None
        if isinstance(self.out_proj.quant_method, FP8QDQLinearMethod):
            scale = self.out_proj.quant_method.get_static_input_scale(self.out_proj)
            # detach() strips the Parameter wrapper so nn.Module.__setattr__
            # does not register the derived scale into norm's state_dict; the
            # detached view still shares storage with out_proj.input_scale.
            self.norm.fp8_scale = scale.detach() if scale is not None else None
        if _Q35_GDN_FUSE in (2, 3, 4) and getattr(self, "_q35_fuse_counters", None) is None:
            # Per-(request, value head) arrival counters of the fused gated RMSNorm (TRTLLM_Q35_GDN_FUSE=2..4); the
            # kernel's last program resets its counter, so they stay zero between launches.
            self._q35_fuse_counters = torch.zeros(64, dtype=torch.int32, device=self.A_log.device)

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _lowm_in_proj(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor],
        conv: Optional[dict] = None,
    ) -> Optional[tuple]:
        """(in_proj_qkvz, in_proj_ba) outputs from one low-M launch, or None when it does not apply.

        ``TRTLLM_Q35_LOWM_GDN_BA=1`` (with ``TRTLLM_Q35_LOWM_FP8=1``): the FP8 (W8A8) ``in_proj_qkvz``
        rows and the BF16 ``in_proj_ba`` rows run as two segments of one kernel instead of a cuBLAS
        FP8 GEMM (plus its activation-quantize kernel) on the main stream and a BF16 GEMM on the aux
        stream. The FP8 segment takes ``h_fp8`` (E4M3 from the preceding all-reduce,
        TRTLLM_Q35_AR_FP8_OUT) when given, else quantizes the BF16 ``hidden_states`` in registers with
        ``in_proj_qkvz.input_scale`` (same values as the quantize kernel); the BF16 segment takes
        ``hidden_states``.
        """
        if not lowm_gdn_ba_enabled():
            return None
        qkvz, ba = self.in_proj_qkvz, self.in_proj_ba
        act = "fp8" if h_fp8 is not None else lowm_fp8_act()
        mode = {"fp8": "mixed8", "quant": "mixed"}.get(act)
        m = hidden_states.shape[0]
        if (
            mode is None
            or lowm_skipped("gdn.in_proj_qkvz", m)
            or not isinstance(qkvz.quant_method, FP8QDQLinearMethod)
            or qkvz.bias is not None
            or ba.bias is not None
            or ba.weight.dtype != torch.bfloat16
            or hidden_states.dim() != 2
            or hidden_states.dtype != torch.bfloat16
            or not lowm_rows_ok(m, qkvz.weight.shape[0], qkvz.weight.shape[1], mode)
        ):
            return None
        input_scale = qkvz.quant_method.get_static_input_scale(qkvz)
        if input_scale is None:
            return None
        tactic = lowm_default_tactic(m, qkvz.weight.shape[0], qkvz.weight.shape[1], mode=mode)
        if conv is not None and not lowm_conv_supported(m, tactic, conv["weight"].shape[0], conv["tokens"]):
            return None
        out_qkvz, out_ba = lowm_gemm(
            x16=hidden_states,
            x8=h_fp8,
            w_fp8=qkvz.weight,
            weight_scale=qkvz.weight_scale,
            input_scale=input_scale,
            fp8_act=act,
            w_bf16=ba.weight,
            split_out=qkvz.weight.shape[0],
            out_dtype=hidden_states.dtype,
            tactic=tactic,
            tactic_bf16=(tactic[0], 1, tactic[2] if len(tactic) > 2 else 1),
            conv=conv,
        )
        if conv is not None:
            # TRTLLM_Q35_GDN_FUSE=4: the conv channels left through the conv1d epilogue (no stock reference).
            return out_qkvz, out_ba
        lowm_check(f"fp8:gdn.in_proj_qkvz(fused):{act}", out_qkvz,
                   lambda: stock_fp8_linear(qkvz, h_fp8 if h_fp8 is not None else hidden_states))
        lowm_check("bf16:gdn.in_proj_ba(fused)", out_ba,
                   lambda: torch.nn.functional.linear(hidden_states, ba.weight))
        return out_qkvz, out_ba

    def _compute_tokenwise_inputs(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor] = None,
        defer_ba_join: bool = False,
    ):
        # h_fp8: the same activations already quantized to E4M3 with in_proj_qkvz's static input
        # scale (the preceding all-reduce emits it, TRTLLM_Q35_AR_FP8_OUT); in_proj_ba keeps BF16.
        # defer_ba_join (TRTLLM_Q35_GDN_BA_LATE_JOIN): when in_proj_ba goes to the aux stream, leave the join to the
        # caller (_q35_join_ba before the first consumer of a / b).
        fused = self._lowm_in_proj(hidden_states, h_fp8)
        if fused is not None:
            projected_states_qkvz, projected_states_ba = fused
        else:
            qkvz_input = hidden_states
            if h_fp8 is not None and isinstance(self.in_proj_qkvz.quant_method, FP8QDQLinearMethod):
                qkvz_input = h_fp8

            def _compute_projected_states_qkvz():
                return self.in_proj_qkvz(qkvz_input)

            def _compute_projected_states_ba():
                return self.in_proj_ba(hidden_states)

            if (
                defer_ba_join
                and do_multi_stream()
                and self.aux_stream is not None
                and not torch.compiler.is_compiling()
            ):
                # maybe_execute_in_parallel's multi-stream branch without its final join.
                main_event = self.event_dict[EventType.Main]
                ba_event = self.event_dict[EventType.Attention]
                main_event.record()
                projected_states_qkvz = _compute_projected_states_qkvz()
                with torch.cuda.stream(self.aux_stream):
                    main_event.wait()
                    projected_states_ba = _compute_projected_states_ba()
                    ba_event.record()
                self._q35_ba_join_pending = ba_event
                rows = hidden_states.shape[0]
                if rows not in _q35_ba_late_join_logged:
                    _q35_ba_late_join_logged.add(rows)
                    logger.info(f"TRTLLM_Q35_GDN_BA_LATE_JOIN: in_proj_ba join deferred at {rows} rows "
                                f"(layer {self.layer_idx}, first call)")
            else:
                projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
                    _compute_projected_states_qkvz,
                    _compute_projected_states_ba,
                    self.event_dict[EventType.Main],
                    self.event_dict[EventType.Attention],
                    self.aux_stream,
                    disable_on_compile=True,
                )

        # The weight mapper reorders in_proj rows into the dense per-rank
        # layouts [Q|K|V|Z] and [b|a] (see grouped_to_dense_in_proj_qkvz_perm),
        # so every component is a plain column slice of the projection —
        # no split/reshape kernel. Downstream consumers (causal_conv1d,
        # the GDN decode kernels, the gated norm) read these row-strided
        # views in place.
        num_tokens = projected_states_qkvz.shape[0]
        mixed_qkv = projected_states_qkvz[:, : self.conv_dim_per_tp]
        z = projected_states_qkvz[:, self.conv_dim_per_tp :].view(
            num_tokens, self.num_v_heads_per_tp, self.head_v_dim
        )
        b = projected_states_ba[:, : self.num_v_heads_per_tp]
        a = projected_states_ba[:, self.num_v_heads_per_tp :]

        return mixed_qkv, z, a, b

    def _q35_join_ba(self) -> None:
        """TRTLLM_Q35_GDN_BA_LATE_JOIN: make the current stream wait for the aux-stream in_proj_ba if its join was
        deferred (no-op otherwise)."""
        event = self.__dict__.get("_q35_ba_join_pending")
        if event is not None:
            self._q35_ba_join_pending = None
            event.wait()

    def _postprocess_gdn_output(
        self,
        attn_out: torch.Tensor,
        z: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        # z is a [num_tokens, num_v_heads, head_v_dim] view of the in_proj
        # output whose (heads, head_dim) block is contiguous per token; the
        # gated norm reads it through its token stride instead of packing a
        # copy.
        attn_out = rms_norm_gated_token_major(
            attn_out.reshape(-1, self.head_v_dim),
            z,
            self.norm.weight,
            self.norm.eps,
            fp8_scale=self.norm.fp8_scale,
        )
        attn_out = attn_out.view(-1, self.value_dim_per_tp)
        return self.out_proj(attn_out, all_reduce_params=all_reduce_params)

    def _q35_fuse_ctx(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
        num_rows: int,
    ) -> Optional[dict]:
        """Eligibility of the fused GDN target-verify path (TRTLLM_Q35_GDN_FUSE) and its per-call state, or None.

        Same conditions as the replay branch of forward_decode (pure decode, speculative cache manager with the
        cached replay, fewer requests than CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, i.e. no partitioned replay /
        all-layer commit) plus the shapes of q35_gdn_fused_supported.
        """
        if spec_metadata is None or is_torch_compiling() or attn_metadata.num_contexts != 0:
            return None
        kv_cache_manager = attn_metadata.kv_cache_manager
        num_decodes = attn_metadata.seq_lens.shape[0]
        if (
            num_decodes == 0
            or num_decodes > _Q35_GDN_FUSE_MAX_BATCH
            or num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            or not kv_cache_manager.is_speculative()
            or not getattr(kv_cache_manager, "use_replay_state_update", False)
            or self.conv1d.bias is not None
            or self.activation not in ("silu", "swish")
        ):
            return None
        draft_token_num = spec_metadata.runtime_draft_len + 1
        replay_metadata = kv_cache_manager.get_replay_state_update_metadata()
        if (
            replay_metadata is None
            or draft_token_num != replay_metadata.replay_step_width
            or num_rows != num_decodes * draft_token_num
        ):
            return None
        layer_cache = kv_cache_manager.mamba_layer_cache(self.layer_idx)
        if layer_cache is None or layer_cache.intermediate_conv_window is None:
            return None
        ssm_states = layer_cache.temporal
        # Beyond 64 (request, value head) pairs only the CUDA core kernel applies (TRTLLM_Q35_GDN_CORE_LARGE).
        core_only = num_decodes * self.num_v_heads_per_tp > 64
        if core_only:
            if not (
                _Q35_GDN_CORE_LARGE
                and _Q35_GDN_FUSE in (3, 4)
                and self.conv1d.weight.shape[-1] == 4
                and q35_gdn_core_supported(
                    num_decodes,
                    draft_token_num,
                    self.num_k_heads_per_tp,
                    self.num_v_heads_per_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                    replay_metadata.replay_history_size,
                    ssm_states.dtype,
                )
            ):
                return None
        elif not q35_gdn_fused_supported(
            num_decodes,
            draft_token_num,
            self.num_k_heads_per_tp,
            self.num_v_heads_per_tp,
            self.head_k_dim,
            self.head_v_dim,
            self.conv1d.weight.shape[-1],
            replay_metadata.replay_history_size,
            ssm_states.dtype,
        ):
            return None
        counters = getattr(self, "_q35_fuse_counters", None)
        if counters is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._q35_fuse_counters = counters = torch.zeros(64, dtype=torch.int32, device=ssm_states.device)
        conv_weight = self.conv1d.weight
        if conv_weight.dim() == 3:
            conv_weight = conv_weight.view(conv_weight.shape[0], conv_weight.shape[2])
        return dict(
            kv_cache_manager=kv_cache_manager,
            num_decodes=num_decodes,
            draft_token_num=draft_token_num,
            replay_metadata=replay_metadata,
            layer_cache=layer_cache,
            ssm_states=ssm_states,
            state_indices=attn_metadata.mamba_metadata.state_indices[:num_decodes],
            counters=counters,
            conv_weight=conv_weight,
            core_only=core_only,
        )

    def _q35_fused_call(
        self,
        ctx: dict,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_in_kernel: bool,
        fuse_norm: bool,
        conv_in_inproj: bool = False,
    ) -> torch.Tensor:
        """q35_gdn_fused_update on the per-call state of _q35_fuse_ctx; returns the out_proj input."""
        if ctx.get("core_only"):
            # TRTLLM_Q35_GDN_CORE_LARGE: the Triton fused kernel does not cover this batch and the conv1d update has
            # already run, so a step the core kernel cannot take must not fall back silently.
            y = None
            if fuse_norm and not conv_in_kernel:
                y = self._q35_core_call(ctx, mixed_qkv, z, a, b)
            if y is None:
                raise RuntimeError(
                    f"TRTLLM_Q35_GDN_CORE_LARGE: the CUDA core kernel does not cover this call "
                    f"({ctx['num_decodes']} requests x {self.num_v_heads_per_tp} value heads, fuse_norm={fuse_norm}, "
                    f"conv_in_kernel={conv_in_kernel}, mixed_qkv {tuple(mixed_qkv.shape)} {mixed_qkv.stride()})")
            return y
        min_batch = (_Q35_GDN_CORE == 1 and not conv_in_inproj and _Q35_GDN_CORE_MIN_BATCH > 0
                     and ctx["num_decodes"] >= _Q35_GDN_CORE_MIN_BATCH)
        if (_Q35_GDN_CORE >= 2 or (_Q35_GDN_CORE == 1 and conv_in_inproj) or min_batch) and fuse_norm \
                and not conv_in_kernel:
            y = self._q35_core_call(ctx, mixed_qkv, z, a, b)
            if y is not None:
                if min_batch and ctx["num_decodes"] not in _q35_core_min_batch_logged:
                    _q35_core_min_batch_logged.add(ctx["num_decodes"])
                    logger.info(f"TRTLLM_Q35_GDN_CORE_MIN_BATCH={_Q35_GDN_CORE_MIN_BATCH}: CUDA core kernel on the "
                                f"separate-conv path at b{ctx['num_decodes']} (first call)")
                return y
        num_decodes = ctx["num_decodes"]
        layer_cache = ctx["layer_cache"]
        opts = _Q35_GDN_FUSE_OPTS if num_decodes < _Q35_GDN_FUSE_LARGE_FROM else _Q35_GDN_FUSE_OPTS_LARGE
        common = dict(
            mixed_qkv=mixed_qkv,
            conv_weight=ctx["conv_weight"],
            conv_states=layer_cache.conv,
            intermediate_conv_window=layer_cache.intermediate_conv_window,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            ssm_states=ctx["ssm_states"],
            state_indices=ctx["state_indices"],
            old_u=layer_cache.old_x,
            old_k=layer_cache.old_B,
            old_G=layer_cache.old_dt,
            cache_buf_idx=layer_cache.cache_buf_idx,
            prev_num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            history_size=ctx["replay_metadata"].replay_history_size,
            num_decodes=num_decodes,
            draft_token_num=ctx["draft_token_num"],
            num_k_heads=self.num_k_heads_per_tp,
            num_v_heads=self.num_v_heads_per_tp,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv=conv_in_kernel,
            **opts,
        )
        if fuse_norm:
            return q35_gdn_fused_update(
                **common,
                z=z,
                norm_weight=self.norm.weight,
                norm_eps=self.norm.eps,
                fp8_scale=self.norm.fp8_scale,
                counters=ctx["counters"],
            )
        o = q35_gdn_fused_update(**common)
        return self._q35_gated_norm(o, z)

    def _q35_core_call(
        self,
        ctx: dict,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """TRTLLM_Q35_GDN_CORE: replay + gated RMSNorm (+ FP8) of convolved q|k|v rows on the CUDA core kernel; None
        when the shape or a layout is not covered (the Triton kernel runs instead)."""
        num_decodes, draft_token_num = ctx["num_decodes"], ctx["draft_token_num"]
        layer_cache = ctx["layer_cache"]
        ssm_states = ctx["ssm_states"]
        state_indices = ctx["state_indices"]
        history_size = ctx["replay_metadata"].replay_history_size
        fp8_scale = self.norm.fp8_scale
        if (
            not q35_gdn_core_supported(
                num_decodes,
                draft_token_num,
                self.num_k_heads_per_tp,
                self.num_v_heads_per_tp,
                self.head_k_dim,
                self.head_v_dim,
                history_size,
                ssm_states.dtype,
            )
            or state_indices.dtype != torch.int32
            or not state_indices.is_contiguous()
            or self.norm.weight.dtype != torch.bfloat16
            or mixed_qkv.dtype != torch.bfloat16
            or mixed_qkv.stride(1) != 1
            or mixed_qkv.stride(0) % 8
            or mixed_qkv.data_ptr() % 16
            or a.stride(1) != 1
            or b.stride(1) != 1
            or a.stride(0) != b.stride(0)
            or z.stride(2) != 1
            or z.stride(1) != self.head_v_dim
            or not q35_gdn_core_ready(
                mixed_qkv.device.index,
                draft_token_num,
                self.num_k_heads_per_tp,
                self.num_v_heads_per_tp,
                fp8_scale is not None,
                _Q35_GDN_CORE_TRIGGER,
            )
        ):
            return None
        logger.info_once(
            f"TRTLLM_Q35_GDN_CORE: CUDA core kernel on the GDN target-verify path (T={draft_token_num}, "
            f"key heads {self.num_k_heads_per_tp}, value heads {self.num_v_heads_per_tp} per rank, "
            f"trigger {_Q35_GDN_CORE_TRIGGER})",
            key="q35_gdn_core_active",
        )
        return q35_gdn_core_update(
            mixed_qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_states,
            state_indices,
            layer_cache.old_x,
            layer_cache.old_B,
            layer_cache.old_dt,
            layer_cache.cache_buf_idx,
            layer_cache.prev_num_accepted_tokens,
            history_size,
            num_decodes,
            draft_token_num,
            self.num_k_heads_per_tp,
            self.num_v_heads_per_tp,
            self.head_k_dim,
            self.head_v_dim,
            z=z,
            norm_weight=self.norm.weight,
            norm_eps=self.norm.eps,
            fp8_scale=fp8_scale,
            trigger=_Q35_GDN_CORE_TRIGGER,
        )

    def _q35_fused_inproj(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
    ) -> Optional[torch.Tensor]:
        """TRTLLM_Q35_GDN_FUSE=4: the low-M in_proj with the conv1d update in its epilogue (convolved q|k|v,
        intermediate windows and conv state written by the GEMM), then the fused replay + gated RMSNorm.
        None when the step or the in_proj does not qualify."""
        ctx = self._q35_fuse_ctx(attn_metadata, spec_metadata, hidden_states.shape[0])
        if ctx is None or ctx["state_indices"].dtype != torch.int32:
            return None
        layer_cache = ctx["layer_cache"]
        res = self._lowm_in_proj(
            hidden_states,
            h_fp8,
            conv=dict(
                weight=ctx["conv_weight"],
                state=layer_cache.conv,
                slots=ctx["state_indices"],
                windows=layer_cache.intermediate_conv_window,
                tokens=ctx["draft_token_num"],
            ),
        )
        if res is None:
            return None
        out_qkvz, out_ba = res
        num_tokens = out_qkvz.shape[0]
        mixed_qkv = out_qkvz[:, : self.conv_dim_per_tp]
        z = out_qkvz[:, self.conv_dim_per_tp :].view(num_tokens, self.num_v_heads_per_tp, self.head_v_dim)
        b = out_ba[:, : self.num_v_heads_per_tp]
        a = out_ba[:, self.num_v_heads_per_tp :]
        return self._q35_fused_call(ctx, mixed_qkv, z, a, b, conv_in_kernel=False, fuse_norm=True,
                                    conv_in_inproj=True)

    def _q35_fused_verify(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
    ) -> Optional[torch.Tensor]:
        """TRTLLM_Q35_GDN_FUSE=1/2/3 (and 4 when the in_proj epilogue does not apply: then as 3): the out_proj input
        of a decode-only MTP target-verify step from the fused kernels, or None (separate kernels)."""
        ctx = self._q35_fuse_ctx(attn_metadata, spec_metadata, mixed_qkv.shape[0])
        if ctx is None:
            return None
        num_decodes = ctx["num_decodes"]
        draft_token_num = ctx["draft_token_num"]
        conv_in_kernel = _Q35_GDN_FUSE in (1, 2)  # 3 (and 4 when it falls back) run the conv kernel here
        if not conv_in_kernel:
            # Separate conv1d update (as forward_decode's replay branch: PDL chain conv -> fused replay), then the
            # fused kernel reads the convolved q|k|v rows.
            state_indices = ctx["state_indices"]
            layer_cache = ctx["layer_cache"]
            intermediate_state_indices = (
                _verify_intermediate_state_indices(ctx["kv_cache_manager"], num_decodes, state_indices.device)
                if _Q35_GDN_ARANGE_CACHE
                else torch.arange(num_decodes, dtype=torch.int32, device=state_indices.device)
            )
            conv_out = causal_conv1d_update_triton(
                mixed_qkv.reshape(num_decodes, draft_token_num, -1).transpose(1, 2),
                layer_cache.conv,
                self.conv1d.weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices,
                intermediate_conv_window=layer_cache.intermediate_conv_window,
                intermediate_state_indices=intermediate_state_indices,
                launch_dependent_kernels=True,
                launch_with_pdl_wait=_Q35_GDN_CONV_PDL,
            )
            mixed_qkv = conv_out.transpose(1, 2).reshape(num_decodes * draft_token_num, -1)
        # TRTLLM_Q35_GDN_BA_LATE_JOIN: a / b (in_proj_ba on the aux stream) are read from here on.
        self._q35_join_ba()
        return self._q35_fused_call(
            ctx, mixed_qkv, z, a, b, conv_in_kernel=conv_in_kernel, fuse_norm=_Q35_GDN_FUSE != 1
        )

    def _q35_gated_norm(self, attn_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rms_norm_gated_token_major(
            attn_out.reshape(-1, self.head_v_dim),
            z,
            self.norm.weight,
            self.norm.eps,
            fp8_scale=self.norm.fp8_scale,
        ).view(-1, self.value_dim_per_tp)

    def _replay_verify_recurrent(
        self,
        query,
        key,
        value,
        a,
        b,
        ssm_states,
        state_indices_d,
        num_decodes,
        draft_token_num,
        replay_metadata,
        layer_cache,
        replay_work_items,
        replay_n_writes,
        output_d=None,
        packed_qkv=None,
        use_all_layer_commit=False,
    ):
        """Run MTP target verification via the replay kernel.

        This avoids intermediate-state writes and accepted-state copies; commits
        are deferred via the compact history cache. Cached replay reuses the
        Mamba2 fields as old_x<->U, old_B<->normalized k, and
        old_dt<->cumulative G.
        """
        assert replay_metadata is not None, (
            "GDN replay enabled but replay metadata was not allocated."
        )
        assert draft_token_num == replay_metadata.replay_step_width, (
            "GDN replay does not support dynamic draft length yet: "
            f"{draft_token_num} != {replay_metadata.replay_step_width}"
        )
        if (
            draft_token_num > Q35_GDN_REPLAY_MAX_T
            or replay_metadata.replay_history_size > 16
            or replay_metadata.replay_history_size < draft_token_num
        ):
            raise RuntimeError(
                f"GDN cached replay requires draft_token_num <= {Q35_GDN_REPLAY_MAX_T} and "
                "draft_token_num <= replay_history_size <= 16."
            )
        return fused_recurrent_gated_delta_rule_cached_replay_update(
            q=query,
            k=key,
            v=value,
            g=a,
            beta=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            launch_with_pdl=True,
            ssm_states=ssm_states,
            state_indices=state_indices_d[:num_decodes],
            old_u=layer_cache.old_x,
            old_k=layer_cache.old_B,
            old_G=layer_cache.old_dt,
            old_beta=layer_cache.old_dA_cumsum,
            cache_buf_idx=layer_cache.cache_buf_idx,
            prev_num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            history_size=replay_metadata.replay_history_size,
            replay_work_items=replay_work_items,
            n_writes=replay_n_writes,
            use_qk_l2norm_in_kernel=True,
            packed_qkv=packed_qkv,
            use_all_layer_commit=use_all_layer_commit,
            output=output_d,
        )

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        query_start_loc_long,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]
        num_decodes = kwargs["num_decodes"]

        if is_target_verify:
            draft_token_num = spec_metadata.runtime_draft_len + 1
            assert num_decodes > 0
            assert mixed_qkv.shape[0] == num_decodes * draft_token_num
            assert a.shape[0] == num_decodes * draft_token_num
            assert b.shape[0] == num_decodes * draft_token_num
            assert intermediate_conv_states is not None
            assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

            replay_metadata = kwargs.get("replay_metadata")
            if (
                _Q35_GDN_FUSED_CONV_REPLAY
                and kwargs.get("use_replay", False)
                and kwargs.get("replay_work_items") is None
                and not kwargs.get("use_cached_replay_all_layer_commit", False)
                and replay_metadata is not None
                and draft_token_num == replay_metadata.replay_step_width
                and self.conv1d.bias is None
                and self.activation in ("silu", "swish")
                and q35_conv_replay_supported(
                    num_decodes,
                    draft_token_num,
                    self.num_k_heads_per_tp,
                    self.num_v_heads_per_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                    self.conv1d.weight.shape[-1],
                    replay_metadata.replay_history_size,
                    ssm_states.dtype,
                )
            ):
                layer_cache = kwargs.get("layer_cache")
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes, draft_token_num, self.num_v_heads_per_tp, self.head_v_dim
                    )
                return q35_conv_replay_update(
                    mixed_qkv,
                    self.conv1d.weight,
                    conv_states,
                    intermediate_conv_states,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    ssm_states,
                    cache_indices[:num_decodes],
                    layer_cache.old_x,
                    layer_cache.old_B,
                    layer_cache.old_dt,
                    layer_cache.cache_buf_idx,
                    layer_cache.prev_num_accepted_tokens,
                    replay_metadata.replay_history_size,
                    num_decodes,
                    draft_token_num,
                    self.num_k_heads_per_tp,
                    self.num_v_heads_per_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                    output=output_d,
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )

            # Speculative verification path:
            # 1. run conv update with per-step intermediate cache writes
            # 2. run recurrent delta rule with intermediate SSM-state cache writes
            # 3. defer final state selection to kv_cache_manager.update_mamba_states()
            intermediate_state_indices = kwargs.get("intermediate_state_indices")
            if intermediate_state_indices is None:
                intermediate_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=cache_indices.device
                )

            mixed_qkv_reshaped = mixed_qkv.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update_triton(
                mixed_qkv_reshaped,
                conv_states,
                self.conv1d.weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=cache_indices[:num_decodes],
                intermediate_conv_window=intermediate_conv_states,
                intermediate_state_indices=intermediate_state_indices,
                # PDL chain: conv1d -> replay verify kernel (replay only)
                launch_dependent_kernels=kwargs.get("use_replay", False),
                launch_with_pdl_wait=_Q35_GDN_CONV_PDL and kwargs.get("use_replay", False),
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).reshape(
                num_decodes * draft_token_num, -1
            )

            key_size = self.key_dim // self.attn_tp_size
            query = mixed_qkv[..., :key_size]
            key = mixed_qkv[..., key_size : key_size * 2]
            value = mixed_qkv[..., key_size * 2 :]

            query = query.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key = key.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value = value.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a = a.reshape(num_decodes, draft_token_num, -1)
            b = b.reshape(num_decodes, draft_token_num, -1)

            # Prefer the FlashInfer MTP kernel (raw a/b gating in-kernel,
            # initial state gathered from the pool via cache indices, per-step
            # intermediate states written to the batch-scoped [:num_decodes]
            # prefix consumed by update_mamba_states()); fall back to the
            # Triton recurrent kernel when unavailable.
            if kwargs.get("use_replay", False):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return self._replay_verify_recurrent(
                    query,
                    key,
                    value,
                    a,
                    b,
                    ssm_states,
                    cache_indices,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    packed_qkv=mixed_qkv,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            if _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query,
                    k=key,
                    v=value,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices[:num_decodes],
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            beta = b.sigmoid()
            g = fused_gdn_gating(
                self.A_log,
                a.view(num_decodes * draft_token_num, -1),
                self.dt_bias,
            ).reshape(num_decodes, draft_token_num, -1)

            # Keep intermediate-state indexing consistent with Mamba2Mixer:
            # cache slots [0..num_decodes-1] are consumed by
            # MambaCacheManager.update_mamba_states(), while initial states are
            # gathered from real slot indices.
            recurrent_state_source = ssm_states[cache_indices[:num_decodes]]
            recurrent_state_indices = kwargs.get("intermediate_state_indices")
            if recurrent_state_indices is None:
                recurrent_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=cache_indices.device
                )

            output_d = None
            if output is not None:
                output_d = output.view(
                    num_decodes,
                    draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=recurrent_state_source,
                initial_state_indices=recurrent_state_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_ssm_states,
                cache_steps=draft_token_num,
                output=output_d,
            )
            return attn_out.view(
                1,
                num_decodes * draft_token_num,
                self.num_v_heads // self.attn_tp_size,
                self.head_v_dim,
            )

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        # Keep q/k/v as views over mixed_qkv so the fused decode kernel can
        # consume their native strides without forcing packed copies.
        query = mixed_qkv[..., : self.key_dim_per_tp]
        key = mixed_qkv[..., self.key_dim_per_tp : self.key_dim_per_tp * 2]
        value = mixed_qkv[..., self.key_dim_per_tp * 2 :]
        seq_len = query.shape[0]
        query = query.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        key = key.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        value = value.view(1, seq_len, self.num_v_heads_per_tp, self.head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc_long,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            output=output,
        )

        return core_attn_out

    def forward_extend(
        self,
        conv_states,
        ssm_states,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        batch_size = kwargs["batch_size"]
        has_initial_states = kwargs["has_initial_states"][:batch_size]
        cache_indices = kwargs["cache_indices"]
        query_start_loc = kwargs["query_start_loc"]
        query_start_loc_long = kwargs["query_start_loc_long"]
        num_prefill_tokens = kwargs["num_prefill_tokens"]
        num_decode_tokens = kwargs["num_decode_tokens"]
        state_indices_p = kwargs["state_indices_p"]
        state_indices_d = kwargs["state_indices_d"]
        num_prefill = kwargs["num_prefill"]
        num_decodes = kwargs["num_decodes"]
        q35_split = kwargs.get("q35_split")

        conv_states_to_use = conv_states
        if q35_split is not None:
            if num_decode_tokens > 0 or is_target_verify:
                raise RuntimeError("in-forward Mamba snapshot needs a context-only batch")
            # the convolution state at P: the d_conv - 1 raw inputs before P, oldest first
            q35_window = mixed_qkv.index_select(0, q35_split.conv_win)

        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        if num_decode_tokens > 0:
            mixed_qkv_p, mixed_qkv_d = torch.split(mixed_qkv, seqlen_split_size, dim=0)
            query_start_loc_p = query_start_loc[: num_prefill + 1]
            has_initial_states_p = has_initial_states[:num_prefill]

            mixed_qkv_p_t = extract_transpose_prefill_slice(
                mixed_qkv_p,
                mixed_qkv_p.shape[0],
                0,
                mixed_qkv_p.shape[1],
            )
            mixed_qkv_p_t = causal_conv1d_fn(
                mixed_qkv_p_t,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc_p,
            )

            if is_target_verify:
                a_d = a[num_prefill_tokens:]
                b_d = b[num_prefill_tokens:]
                draft_token_num = spec_metadata.runtime_draft_len + 1
                assert num_decodes > 0
                assert mixed_qkv_d.shape[0] == num_decodes * draft_token_num
                assert a_d.shape[0] == num_decodes * draft_token_num
                assert b_d.shape[0] == num_decodes * draft_token_num
                assert intermediate_conv_states is not None
                assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

                intermediate_state_indices = kwargs.get("intermediate_state_indices")
                if intermediate_state_indices is None:
                    intermediate_state_indices = torch.arange(
                        num_decodes, dtype=torch.int32, device=state_indices_d.device
                    )
                mixed_qkv_d = mixed_qkv_d.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
                mixed_qkv_d = causal_conv1d_update_triton(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                    intermediate_conv_window=intermediate_conv_states,
                    intermediate_state_indices=intermediate_state_indices,
                    # PDL chain: conv1d -> replay verify kernel (replay only)
                    launch_dependent_kernels=kwargs.get("use_replay", False),
                )
                mixed_qkv_d = mixed_qkv_d.transpose(1, 2).reshape(num_decode_tokens, -1)
            else:
                mixed_qkv_d = causal_conv1d_update(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                )
            if is_target_verify:
                if num_prefill_tokens > 0:
                    query_p, key_p, value_p, g_p, beta_p = fused_gdn_post_conv(
                        mixed_qkv_p_t,
                        None,
                        a[:num_prefill_tokens],
                        b[:num_prefill_tokens],
                        self.A_log,
                        self.dt_bias,
                        self.num_k_heads_per_tp,
                        self.head_k_dim,
                        self.num_v_heads_per_tp,
                        self.head_v_dim,
                        beta_dtype=b.dtype,
                    )
                query_d, key_d, value_d = pack_gdn_decode_qkv(
                    mixed_qkv_d,
                    self.num_k_heads_per_tp,
                    self.head_k_dim,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )
            else:
                query, key, value, g, beta = fused_gdn_post_conv(
                    mixed_qkv_p_t,
                    mixed_qkv_d,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    self.num_k_heads_per_tp,
                    self.head_k_dim,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )
        else:
            mixed_qkv_t = extract_transpose_prefill_slice(
                mixed_qkv,
                mixed_qkv.shape[0],
                0,
                mixed_qkv.shape[1],
            )
            mixed_qkv_t = causal_conv1d_fn(
                mixed_qkv_t,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if q35_split is not None:
                stage_conv = kwargs["q35_stage"][1]
                stage_conv.index_copy_(
                    0, q35_split.stage_rows, conv_states.index_select(0, q35_split.tail_slots)
                )
                conv_states.index_copy_(
                    0,
                    q35_split.tail_slots,
                    q35_window.view(q35_split.k, conv_states.shape[-1], -1)
                    .transpose(1, 2)
                    .contiguous()
                    .to(conv_states.dtype),
                )
            query, key, value, g, beta = fused_gdn_post_conv(
                mixed_qkv_t,
                None,
                a,
                b,
                self.A_log,
                self.dt_bias,
                self.num_k_heads_per_tp,
                self.head_k_dim,
                self.num_v_heads_per_tp,
                self.head_v_dim,
            )

        if is_target_verify and num_decode_tokens > 0:
            attn_out_prefill = None
            if num_prefill_tokens > 0:
                recurrent_state_p = ssm_states[state_indices_p]
                output_p = output[:, :num_prefill_tokens, :, :] if output is not None else None
                attn_out_prefill, last_recurrent_state = chunk_gated_delta_rule(
                    q=query_p,
                    k=key_p,
                    v=value_p,
                    g=g_p,
                    beta=beta_p,
                    initial_state=recurrent_state_p,
                    output_final_state=True,
                    cu_seqlens=query_start_loc_long[: num_prefill + 1],
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                    output=output_p,
                )
                last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
                ssm_states[state_indices_p] = last_recurrent_state

            draft_token_num = spec_metadata.runtime_draft_len + 1
            query_d = query_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key_d = key_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value_d = value_d.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a_d = a_d.reshape(num_decodes, draft_token_num, -1)
            b_d = b_d.reshape(num_decodes, draft_token_num, -1)
            out_v_heads = self.num_v_heads // self.attn_tp_size

            output_d = None
            if output is not None:
                output_d = output[:, num_prefill_tokens:, :, :].view(
                    num_decodes,
                    draft_token_num,
                    out_v_heads,
                    self.head_v_dim,
                )

            if kwargs.get("use_replay", False):
                attn_out_decode = self._replay_verify_recurrent(
                    query_d,
                    key_d,
                    value_d,
                    a_d,
                    b_d,
                    ssm_states,
                    state_indices_d,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            elif _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                # FI gathers the initial state from the pool via state_indices_d
                # (no host gather) and writes batch-scoped intermediate states;
                # the [:num_decodes] prefix matches update_mamba_states()'s rows.
                attn_out_decode = _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a_d,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    b=b_d,
                    initial_state_source=ssm_states,
                    initial_state_indices=state_indices_d,
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            else:
                beta_d = b_d.sigmoid()
                g_d = fused_gdn_gating(
                    self.A_log,
                    a_d.view(num_decodes * draft_token_num, -1),
                    self.dt_bias,
                ).reshape(num_decodes, draft_token_num, -1)

                recurrent_state_source = ssm_states[state_indices_d]
                recurrent_state_indices = kwargs.get("intermediate_state_indices")
                if recurrent_state_indices is None:
                    recurrent_state_indices = torch.arange(
                        num_decodes, dtype=torch.int32, device=state_indices_d.device
                    )

                attn_out_decode = fused_recurrent_gated_delta_rule_update(
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    g=g_d,
                    beta=beta_d,
                    initial_state_source=recurrent_state_source,
                    initial_state_indices=recurrent_state_indices,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_ssm_states,
                    cache_steps=draft_token_num,
                    output=output_d,
                ).view(1, num_decode_tokens, out_v_heads, self.head_v_dim)

            if output is not None:
                return output
            if attn_out_prefill is None:
                return attn_out_decode
            return torch.cat((attn_out_prefill, attn_out_decode), dim=1)

        if q35_split is not None:
            return _q35_split_prefill_core(
                query, key, value, g, beta, ssm_states, q35_split, kwargs["q35_stage"][0], output
            )

        core_attn_out, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            # This path writes recurrent state directly back into the shared
            # pool; callers **must** ensure cache_indices do not alias live slots.
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=query_start_loc_long,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            output=output,
        )

        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        h_fp8: Optional[torch.Tensor] = None,
    ):
        if _Q35_GDN_FUSE == 4:
            fused = self._q35_fused_inproj(hidden_states, h_fp8, attn_metadata, spec_metadata)
            if fused is not None:
                return self.out_proj(fused, all_reduce_params=all_reduce_params)

        # TRTLLM_Q35_GDN_BA_LATE_JOIN: on the FUSE=3/4 verify path the separate conv1d update does not read a / b, so
        # the aux-stream in_proj_ba is joined after it (_q35_fused_verify) instead of right after the qkvz GEMM.
        defer_ba_join = _Q35_GDN_BA_LATE_JOIN and _Q35_GDN_FUSE in (3, 4)
        mixed_qkv, z, a, b = self._compute_tokenwise_inputs(hidden_states, h_fp8, defer_ba_join=defer_ba_join)

        if _Q35_GDN_FUSE:
            fused = self._q35_fused_verify(mixed_qkv, z, a, b, attn_metadata, spec_metadata)
            if fused is not None:
                return self.out_proj(fused, all_reduce_params=all_reduce_params)
        # Every path below reads a / b.
        self._q35_join_ba()

        if self.register_to_config and is_torch_compiling():
            attn_out = mixed_qkv.new_empty(
                (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
            )
            gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        elif self.register_to_config and is_in_breakable_cuda_graph():
            attn_out = mixed_qkv.new_empty(
                (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
            )
            maybe_bcg_gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        else:
            attn_out = self.forward_core(
                mixed_qkv,
                a,
                b,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
            )

        return self._postprocess_gdn_output(attn_out, z, all_reduce_params)

    def forward_core(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        output: Optional[torch.Tensor] = None,
    ):
        ### sglang linear attn
        # has_initial_states = None
        # if forward_batch.extend_prefix_lens is not None:
        #     has_initial_states = forward_batch.extend_prefix_lens > 0

        ### mamba2_mixer layer
        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        batch_split_size = [num_prefills, num_decodes]
        has_initial_states = mamba_metadata.has_initial_states
        state_indices = mamba_metadata.state_indices[: num_prefills + num_decodes]
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_states = layer_cache.conv
        ssm_states = layer_cache.temporal

        state_indices_p, state_indices_d = torch.split(state_indices, batch_split_size)
        if num_prefills > 0:
            # PyExecutor guarantees prefill requests are placed before decode requests
            has_initial_states_p = has_initial_states[:num_prefills]
            _reset_gdn_states(
                ssm_states,
                conv_states,
                state_indices_p,
                has_initial_states_p,
            )

        is_target_verify = (
            num_decodes > 0
            and spec_metadata is not None
            and attn_metadata.kv_cache_manager.is_speculative()
            and layer_cache is not None
        )
        intermediate_conv_states = (
            layer_cache.intermediate_conv_window if is_target_verify else None
        )
        intermediate_ssm_states = layer_cache.intermediate_ssm if is_target_verify else None
        intermediate_state_indices = (
            _verify_intermediate_state_indices(
                attn_metadata.kv_cache_manager, num_decodes, state_indices.device
            )
            if is_target_verify and _Q35_GDN_ARANGE_CACHE
            else None
        )
        use_replay = is_target_verify and getattr(
            attn_metadata.kv_cache_manager, "use_replay_state_update", False
        )
        replay_metadata = (
            attn_metadata.kv_cache_manager.get_replay_state_update_metadata()
            if use_replay
            else None
        )

        use_cached_replay_all_layer_commit = (
            num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            and getattr(
                attn_metadata.kv_cache_manager,
                "use_gdn_cached_replay_all_layer_commit",
                False,
            )
        )
        kwargs = {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "has_initial_states": has_initial_states,
            "cache_indices": state_indices,
            "query_start_loc": mamba_metadata.query_start_loc,
            "query_start_loc_long": mamba_metadata.query_start_loc_long,
            "batch_size": attn_metadata.seq_lens.shape[0],
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "state_indices_p": state_indices_p,
            "state_indices_d": state_indices_d,
            "num_prefill": num_prefills,
            "num_decodes": num_decodes,
            "intermediate_state_indices": intermediate_state_indices,
            "use_replay": use_replay,
            "replay_metadata": replay_metadata,
            "layer_cache": layer_cache,
            "replay_work_items": mamba_metadata.replay_work_items[:num_decodes]
            if use_replay and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            else None,
            "replay_n_writes": mamba_metadata.replay_n_writes
            if use_replay and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            else None,
            "use_cached_replay_all_layer_commit": use_cached_replay_all_layer_commit,
        }
        q35_split = getattr(mamba_metadata, "q35_split", None) if num_prefills > 0 else None
        if q35_split is not None:
            kwargs["q35_split"] = q35_split
            kwargs["q35_stage"] = attn_metadata.kv_cache_manager.q35_stage_views(self.layer_idx)
        if num_prefills > 0:
            attn_out = self.forward_extend(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                **kwargs,
            )
        else:
            attn_out = self.forward_decode(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                **kwargs,
            )

        return attn_out
