# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather, cp_allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._torch.pyexecutor.input_commit import flush_input_commit, stage_or_copy
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, maybe_compile
from tensorrt_llm._utils import get_sm_version, is_sm_100f, maybe_pin_memory, prefer_pinned
from tensorrt_llm.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..params import get_indexer_mqa_logits_elem_budget, use_self_sampling_gvr
from . import prefill_share
from .cache_manager import get_indexer_k_data_bytes_per_token, resolve_indexer_k_dtype
from .helix_key_shard import (
    count_set_mismatches,
    helix_key_shard_verify_enabled,
    helix_key_shard_verify_full,
    reference_local_topk,
)
from .kernels import (
    bf16_mqa_logits,
    bf16_paged_mqa_logits,
    gather_k_cache_paged,
    triton_convert_req_index_to_global_index,
)
from .params import DSAParams, disagg_role, is_kimi_k31_dsa

ModelConfig = tensorrt_llm.bindings.ModelConfig

# Cap the per-call indexer MQA-logits transient (in elements). fp8_mqa_logits
# allocates its [q x kv] logits output via torch.empty; the KV dimension is the
# full (compressed) context and is unbounded, so for a large query chunk on a
# long-context prefill this single allocation can reach tens of GB. Under
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True such an allocation can stall
# indefinitely in cuMemCreate on the longest-context (attention_dp laggard) rank
# -> GPU idle -> peers block at the next MoE all-to-all -> watchdog hang. Tiling
# the query dimension caps the transient to q_tile x kv with identical results
# (each query row's logits/top-k are independent). Override via env if needed;
# the KV cache budget reserves the same cap (see KvCacheCreator), so both read it
# through one helper.
_INDEXER_MQA_LOGITS_ELEM_BUDGET = get_indexer_mqa_logits_elem_budget()

# Cap on the [sum(q) x sum(kv)] logits of context requests packed into one
# indexer prefill chunk (in elements; 1 << 28 is 1 GiB of fp32). A packed
# chunk gathers the KV of every request in it and scores every query against
# all of it, so its logits grow with the product of the packed totals while
# each query row only needs its own request's window. Prefix-reuse hits make
# this explode: each brings few new tokens but its whole cached context. The
# cap only decides which requests share a chunk; _packed_logits_elem_budget
# also clamps it to the largest single-request logits tile the configuration
# can reach. The memory of each logits call is bounded separately by the query
# tiling under _INDEXER_MQA_LOGITS_ELEM_BUDGET, which also accounts for the
# per-head dot products of the decoded FP4 path and tiles a request whose own
# logits exceed this cap in a chunk of its own. Override via env if needed.
_INDEXER_PACKED_LOGITS_ELEM_BUDGET = int(
    os.environ.get("TLLM_INDEXER_PACKED_LOGITS_ELEM_BUDGET", 1 << 28)
)

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

    from .metadata import DSAtrtllmAttentionMetadata

# Optional import: fast-hadamard-transform causes CI build issues (requires wheel+torch pre-installed)
try:
    from fast_hadamard_transform import hadamard_transform

    HAS_FAST_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAS_FAST_HADAMARD = False

_DG_SCHEDULE_BLOCK_KV = 64

# Prefill-sized Top-K remaps (local -> global pool indices) run as the Triton kernel with 1024-entry column tiles
# (2.5x faster than the per-entry C++ kernel at 8192 x 2048); smaller ones keep the C++ op.
_TRITON_TOPK_REMAP_MIN_ROWS = 256
_TRITON_TOPK_REMAP_BLOCK = 1024

# One in-kernel atom for the whole verify window when its MMA N tile fits (next_n * index heads <= 256 for FP8 keys,
# <= 224 for FP4 keys on Blackwell, i.e. next_n <= 8 / 7 for 32 heads): each KV block is read once instead of once
# per atom. The logits are bitwise identical to the split launches (same per-(row, key) dot products and epilogue).
# TRTLLM_DSA_SINGLE_ATOM=0 restores the split.
_DSL_SINGLE_ATOM = os.environ.get("TRTLLM_DSA_SINGLE_ATOM", "1") == "1"
# Single-atom verify windows (and the key-sharded indexer) load the index-K cache with the L2 evict-first hint: every
# key is read once per call, and the hint keeps the just-written logits L2-resident for the Top-K that reads them
# next. The logits are bitwise identical. TRTLLM_DSA_LOGITS_EVICT_FIRST=0 drops the hint.
_DSL_LOGITS_EVICT_FIRST = os.environ.get("TRTLLM_DSA_LOGITS_EVICT_FIRST", "1") == "1"
_DSL_MAX_ATOM_N = 256
# BF16-reduce prefill logits (Indexer.bf16_prefill_logits: Kimi K3.1 MXFP4 index keys on a disaggregated context
# server, SM100 / SM103; lossy): the prefill logits take BF16 head weights and reduce the heads in BF16 (DeepGEMM's
# BF16-reduce epilogue, the numerics of upstream DeepGEMM's SM100 MQA logits). With the CuTe DSL available the FP4 MQA
# logits kernel computes them and stores FP32 for the prefill Top-K (_fp4_bf16_reduce_prefill_logits); otherwise
# DeepGEMM writes BF16 logits that the Top-K reads widened to FP32. Both give the same scores bit for bit;
# TRTLLM_DSA_INDEXER_BF16_LOGITS_CUTE=0 forces the DeepGEMM kernel.
_INDEXER_BF16_LOGITS_CUTE = os.environ.get("TRTLLM_DSA_INDEXER_BF16_LOGITS_CUTE", "1") == "1"
# Prefill rows per kernel "sequence" (the MMA N tile is rows x index heads, <= 224 on Blackwell) and the key block
# of the per-call fused K copy.
_BF16_PREFILL_ROWS = int(os.environ.get("TRTLLM_DSA_INDEXER_BF16_PREFILL_ROWS", "7"))
_BF16_PREFILL_BLOCK_KV = 128
_BF16_PREFILL_EPI_SUBTILES = 1
_BF16_PREFILL_COL_OFFSET = True
_BF16_PREFILL_LD_PREFETCH = True

# BF16 decode logits (Indexer.bf16_decode_logits: Kimi K3.1 FP4 index keys on a disaggregated generation server;
# lossy): single-atom decode verify windows of at least _DECODE_LOGITS_BF16_MIN_ROWS rows reduce the heads with the BF16
# epilogue of the FP4 DSL logits kernel (ReLU as cvt.rn.relu.bf16x2.f32, BF16 FMA over head pairs: the numerics of
# upstream DeepGEMM's SM100 MQA logits). Smaller windows keep the FP32 epilogue, which is as fast there. Windows on the
# BF16 epilogue store BF16 logits when only a Top-K that reads BF16 consumes them (the deterministic self-sampling GVR;
# no Helix merge, no ragged rows), else FP32 logits. The BF16 epilogue's logits are BF16 values, so the stored values
# and the selected indices are the same either way; the logits store and the Top-K loads move half the bytes.
_DECODE_LOGITS_BF16_MIN_ROWS = 64


def _decode_logits_bf16_kwargs(
    enabled: bool, num_rows: int, single_atom: bool, emitting: bool, bf16_reader: bool = False
) -> dict:
    """Epilogue kwargs of the FP4 DSL decode logits for ``num_rows`` verify rows (see Indexer.bf16_decode_logits).

    Empty (the FP32 epilogue) unless ``enabled``, the window runs as one in-kernel atom without Top-K emission, and it
    has at least ``_DECODE_LOGITS_BF16_MIN_ROWS`` rows. With a ``bf16_reader`` (every consumer of the logits reads
    BF16) the logits are also stored as BF16. Logs once per process which epilogue the rows took.
    """
    if not enabled:
        return {}
    if single_atom and not emitting and num_rows >= _DECODE_LOGITS_BF16_MIN_ROWS:
        logger.info_once(
            f"DSA decode logits: BF16 epilogue for {num_rows} rows (>= {_DECODE_LOGITS_BF16_MIN_ROWS} rows)",
            key="dsa_decode_logits_bf16_on",
        )
        kwargs = dict(epi_dtype=torch.bfloat16, fused_relu_cvt=True)
        if bf16_reader:
            logger.info_once(
                f"DSA decode logits: BF16 logits for {num_rows} rows",
                key="dsa_decode_logits_bf16_out",
            )
            kwargs["output_dtype"] = torch.bfloat16
        return kwargs
    logger.info_once(
        f"DSA decode logits: FP32 epilogue for {num_rows} rows (BF16 epilogue gate: one atom "
        f"{single_atom}, emission {emitting}, >= {_DECODE_LOGITS_BF16_MIN_ROWS} rows)",
        key="dsa_decode_logits_bf16_off",
    )
    return {}


@functools.cache
def _fp4_max_atom_n() -> int:
    """Widest MMA N tile of the FP4 DSL logits kernel on this GPU (its TMEM budget: 224 on Blackwell)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.fp4_paged_mqa_logits import (
        _max_n_tile,
    )

    return _max_n_tile()


@functools.cache
def _device_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _bf16_reduce_prefill_rows(heads: int) -> int:
    """Prefill rows per CuTe DSL logits "sequence" for ``heads`` index heads (0: the kernel can't take them)."""
    if heads % 8 != 0:
        return 0
    return max(0, min(_BF16_PREFILL_ROWS, _fp4_max_atom_n() // heads))


def _fp4_bf16_reduce_prefill_logits(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    rows_per_seq: int,
    row_starts_zero: bool = False,
) -> torch.Tensor:
    """MXFP4 MQA prefill logits with DeepGEMM's BF16-reduce epilogue on the CuTe DSL FP4 MQA logits kernel.

    q [R, H, D/2] int8 codes with q_scale [R, H] int32 (four UE8M0 bytes), k [N, D/2] int8 with k_scale [N] int32,
    weights [R, H] float, row windows [row_starts, row_ends) [R] int32. Returns FP32 logits [R, N] (row stride padded
    to 256 keys) valid in every row's window. Equal bit for bit to fp8_fp4_mqa_logits with BF16 weights and BF16
    logits, widened to FP32.

    The kernel scores ``rows_per_seq``-row groups of queries, one MMA N tile per group, over a fused
    [blocks, block_kv * (D/2 + 4)] copy of the keys: group g walks keys [base_g, max row end of the group) with
    base_g = its smallest row start rounded down to 256, and writes them to their own columns. A short last group
    repeats the last row (its scores are dropped). ``row_starts_zero`` (host knowledge that every row start is 0)
    walks every group from key 0 on the kernel instantiation without column offsets, which is faster.
    """
    rows, heads, half_dim = q.shape
    keys = k.shape[0]
    dev = q.device
    nn = rows_per_seq
    groups = -(-rows // nn)
    pad = groups * nn - rows
    q_u8 = q.view(torch.uint8)
    w = weights
    ks = row_starts
    ke = row_ends
    if pad:
        q_u8 = torch.cat((q_u8, q_u8[-1:].expand(pad, -1, -1)))
        q_scale = torch.cat((q_scale, q_scale[-1:].expand(pad, -1)))
        w = torch.cat((w, w[-1:].expand(pad, -1)))
        ks = torch.cat((ks, ks[-1:].expand(pad)))
        ke = torch.cat((ke, ke[-1:].expand(pad)))
    # The kernel walks 256-key tiles (two math warp groups of 128): bases on 256 keep every group's padded tail
    # inside its own row, and the key copy and the block table cover align256(keys).
    if _BF16_PREFILL_COL_OFFSET and not row_starts_zero:
        base = (
            ks.view(groups, nn)
            .amin(dim=1)
            .div(256, rounding_mode="floor")
            .mul_(256)
            .to(torch.int32)
        )
        ctx = (ke.view(groups, nn).amax(dim=1) - base).to(torch.int32)
    else:
        base = None
        ctx = ke.view(groups, nn).amax(dim=1).to(torch.int32)

    blk = _BF16_PREFILL_BLOCK_KV
    aligned = -(-keys // 256) * 256
    blocks = aligned // blk
    full = keys // blk
    data_bytes = blk * half_dim
    pool = torch.empty(blocks, data_bytes + blk * 4, dtype=torch.uint8, device=dev)
    if full < blocks:
        pool[full:].zero_()
    k_u8 = k.view(torch.uint8)
    s_u8 = k_scale.view(torch.uint8).view(keys, 4)
    if full:
        pool[:full, :data_bytes].copy_(k_u8[: full * blk].view(full, data_bytes))
        pool[:full, data_bytes:].copy_(s_u8[: full * blk].view(full, blk * 4))
    tail = keys - full * blk
    if tail:
        pool[full, : tail * half_dim].copy_(k_u8[full * blk :].reshape(-1))
        pool[full, data_bytes : data_bytes + tail * 4].copy_(s_u8[full * blk :].reshape(-1))
    table = torch.arange(blocks, dtype=torch.int32, device=dev).expand(groups, blocks)
    if base is not None:
        table = (table + (base // blk)[:, None]).clamp_(max=blocks - 1)
    else:
        table = table.contiguous()
    num_sms = _device_num_sms(dev.index if dev.index is not None else torch.cuda.current_device())
    schedule = get_paged_mqa_logits_metadata(ctx.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, num_sms)
    out = torch.empty(groups * nn, aligned, dtype=torch.float32, device=dev)
    torch.ops.trtllm.cute_dsl_fp4_mqa_logits_bf16_reduce_into(
        q_u8.reshape(groups, nn, heads, half_dim),
        q_scale.contiguous().view(groups, nn, heads),
        pool.view(blocks, blk, 1, half_dim + 4),
        w.contiguous(),
        ctx,
        table,
        schedule,
        keys,
        out,
        base,
        _BF16_PREFILL_EPI_SUBTILES,
        _BF16_PREFILL_LD_PREFETCH and _BF16_PREFILL_EPI_SUBTILES == 1,
    )
    return out[:rows, :keys]


def use_cute_dsl_paged_mqa_logits(requested: bool) -> bool:
    """Whether the CuTe DSL paged MQA logits kernel runs instead of DeepGEMM.

    The indexer and the metadata must both use this: the metadata builds
    DSL-only buffers (atom-split expansion, padded logits row stride) and
    skips the sm90 MTP expansion off the same decision. is_sm_100f() is the
    custom op's own precondition.
    """
    return requested and IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()


def _pick_dsl_expand(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[int, int]:
    """Pick (expand_factor, effective_next_n) for the DSL paged kernel
    using a wave-aware strategy. Used by both FP4 and FP8 DSL paths.

    The DSL kernel natively supports ``effective_next_n ∈ kernel_atoms``
    (FP4: ``(1, 2, 3)``; FP8: ``(1, 2, 3, 4)``). For ``next_n`` not natively
    supported or when SM utilization can be improved, reshape
    ``[B, next_n, ...]`` -> ``[B * expand_factor, effective_next_n, ...]``
    caller-side.

    Strategy: enumerate ``(expand_factor, effective_next_n)`` pairs with
    ``expand_factor * effective_next_n == next_n`` and ``effective_next_n
    in kernel_atoms``. Score each by ``(waves, -expand_factor)`` where
    ``waves = ceil(B * expand_factor * ceil(max_ctx/256) / num_sms)``.
    Pick min waves; on tie, prefer LARGER expand_factor (more SMs busy per
    wave; pays HBM cost of expand_factor x KV re-reads).

    When ``batch_size == 0`` or ``max_ctx == 0`` (workload unknown), fall
    back to the legacy HBM-minimizing heuristic: largest effective_next_n
    that divides next_n cleanly (still constrained to ``kernel_atoms``).

    Examples (wave-aware, num_sms=148 [B200], SPLIT_KV=256 tokens):
        FP4, next_n=4, B=1,  ctx=4096   -> (4, 1): ntask=64<148, 1 wave, max factor
        FP4, next_n=4, B=32, ctx=4096   -> (2, 2): ntask=1024>148, multi-wave, min factor
        FP4, next_n=2, B=1,  ctx=4096   -> (2, 1): wave-tie, larger factor
        FP8, next_n=4, B=1,  ctx=4096   -> (4, 1): kernel_atoms incl. 4 doesn't change small-B pick
    """
    # Legacy fallback when workload is unknown.
    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = batch_size * factor * ((max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))  # min waves, max factor
    _, factor, eff = cands[0]
    return factor, eff


def _atom_ctx_offsets(next_n: int, num_atoms: int) -> List[int]:
    """Per-atom causal stagger of an in-kernel next_n split: atom i (0 = oldest)
    sees the context shortened by the tokens of the atoms after it. Same split
    as the kernel (first ``next_n % num_atoms`` atoms get one extra token)."""
    base, rem = divmod(next_n, num_atoms)
    return [
        next_n - (i * base + min(i, rem) + base + (1 if i < rem else 0)) for i in range(num_atoms)
    ]


# The atom sizes the FP4 DSL decode logits may split a verify window into when it does not fit one in-kernel atom (MXFP4
# index keys; the cap at 64 index heads). Each atom of the caller-side split re-reads the index keys; the logits do not
# depend on the split (bitwise).
_FP4_EXPAND_ATOMS = (1, 2, 3)


def fp4_expand_atoms() -> Tuple[int, ...]:
    """The atom sizes of the FP4 DSL decode logits' atom split."""
    return _FP4_EXPAND_ATOMS


def _pick_dsl_split(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[int, int, int]:
    """``_pick_dsl_expand`` plus the kernel's in-kernel uneven split.

    Returns ``(expand_factor, atom, num_kernel_atoms)``. When no atom > 1 in
    ``kernel_atoms`` divides ``next_n`` (e.g. 5 or 7), the caller-side split can
    only run atom 1 and reads the KV ``next_n`` times; the in-kernel split runs
    ``ceil(next_n / max(kernel_atoms))`` atoms instead. It is taken when atom 1
    would need many waves and the split needs fewer, or when the workload is unknown.
    """
    factor, atom = _pick_dsl_expand(next_n, num_sms, batch_size, max_ctx, kernel_atoms)
    if atom != 1 or next_n <= 1 or any(next_n % a == 0 for a in kernel_atoms if a > 1):
        return factor, atom, 1
    num_atoms = -(-next_n // max(kernel_atoms))
    if batch_size > 0 and max_ctx > 0:
        tiles = -(-max_ctx // 256)
        waves_split = -(-batch_size * num_atoms * tiles // num_sms)
        waves_atom1 = -(-batch_size * next_n * tiles // num_sms)
        # Below ~16 atom-1 waves the split's wider tiles cost more than the KV
        # re-reads they save (GB300, next_n=7: B=1 0.87-1.00x; >= 24 waves 1.04-2.6x).
        if waves_split >= waves_atom1 or waves_atom1 < 16:
            return factor, atom, 1
    return 1, -(-next_n // num_atoms), num_atoms


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute flat byte indices for FP8/FP4 data and scales from global token positions.

    Shared by Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (logical element count).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache
            (default: head_dim, i.e. one byte per FP8 value). For MXFP4 this
            should be ``head_dim // 2`` since two E2M1 codes pack into one byte.
            The scale layout is identical at head_dim=128: 4 bytes per token
            (one float32 for FP8, four packed UE8M0 exponents for FP4).

    Returns:
        (fp8_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    if data_bytes_per_token is None:
        data_bytes_per_token = head_dim
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings during
        # CUDA graph capture/replay with MTP + DSA.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size
    return fp8_indices, scale_indices


def _unravel_indices(
    flat_indices: torch.Tensor, shape: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
    """
    Unravel indices into multiple dimensions.
    """
    d3 = shape[3]
    i3 = flat_indices % d3
    flat_indices = flat_indices // d3
    d2 = shape[2]
    i2 = flat_indices % d2
    flat_indices = flat_indices // d2
    d1 = shape[1]
    i1 = flat_indices % d1
    flat_indices = flat_indices // d1
    i0 = flat_indices
    return i0, i1, i2, i3


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to activation tensor for DSA sparse attention."""
    assert x.dtype == torch.bfloat16

    if not HAS_FAST_HADAMARD:
        # Fallback: skip transformation (acceptable for test/dev)
        logger.warning_once(
            "fast-hadamard-transform not available. Sparse MLA will skip "
            "hadamard transformation. Install with: "
            "pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
            key="fast_hadamard_import_missing",
        )
        return x

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        "Hidden size must be a power of 2 for Hadamard transform."
    )
    return hadamard_transform(x, scale=hidden_size**-0.5)


def merge_helix_topk(
    logits: torch.Tensor,
    topk_rows: torch.Tensor,
    cp_rank: int,
    gather_cp: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Reduce per-rank Top-K rows to the global Top-K across Helix CP ranks.

    Under Helix each CP rank holds a slice of every sequence's KV cache, so its
    indexer only ranks local tokens. The global Top-K is a subset of the union
    of the local Top-Ks, so gathering every rank's candidate scores and taking
    the Top-K of that union reproduces the selection of an unsharded indexer.
    Local entries that lose to another rank's candidates become -1 in place and
    the surviving entries move to the front of each row.

    Args:
        logits: ``[num_tokens, kv_len]`` local indexer scores.
        topk_rows: ``[num_tokens, k]`` int32 local token indices, -1 for none.
        cp_rank: This rank's position in the CP group.
        gather_cp: Stacks a tensor over the CP group along a new leading dim.
    """
    num_tokens, k = topk_rows.shape
    if num_tokens == 0:
        return
    valid = topk_rows >= 0
    scores = torch.gather(logits.float(), 1, topk_rows.clamp(min=0).long())
    scores = scores.masked_fill(~valid, float("-inf"))
    # [cp, num_tokens, k] -> [num_tokens, cp * k], ordered by (rank, slot).
    all_scores = gather_cp(scores).transpose(0, 1).reshape(num_tokens, -1)
    top_scores, top_pos = torch.topk(all_scores, k, dim=1)
    owner = torch.div(top_pos, k, rounding_mode="floor")
    mine = (owner == cp_rank) & torch.isfinite(top_scores)
    hits = torch.zeros_like(topk_rows, dtype=torch.int32)
    hits.scatter_add_(1, torch.where(mine, top_pos - owner * k, 0), mine.to(torch.int32))
    keep = hits > 0
    # Stable compaction: kept entries first, in their original order.
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
    compacted = torch.gather(topk_rows.masked_fill(~keep, -1), 1, order)
    topk_rows.copy_(compacted)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()
    page_index_scale, layer_offset = (
        attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(layer_idx)
    )

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    num_rows, topk = topk_indices.shape
    if (
        num_rows >= _TRITON_TOPK_REMAP_MIN_ROWS
        and topk % _TRITON_TOPK_REMAP_BLOCK == 0
        and all(t.is_contiguous() for t in (req_idx, block_table, topk_indices))
    ):
        # Same integers as the C++ remap (one thread per entry), in 1024-entry column tiles.
        global_indices = triton_convert_req_index_to_global_index(
            req_idx,
            block_table,
            topk_indices,
            attn_metadata._cached_tokens_per_block,
            topk,
            BLOCK_N=_TRITON_TOPK_REMAP_BLOCK,
            stride_factor=page_index_scale * attn_metadata._cached_tokens_per_block,
            layer_id=layer_offset,
        ).view(num_rows, topk)
    else:
        global_indices = torch.ops.trtllm.convert_req_index_to_global(
            req_idx,
            block_table,
            topk_indices,
            attn_metadata._cached_tokens_per_block,
            topk,
            page_index_scale * attn_metadata._cached_tokens_per_block,
            layer_offset,
        )

    return global_indices, attn_metadata._cached_pool_view


def transform_local_topk_and_prepare_pool_view_grouped(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_ids: torch.Tensor,
    page_index_scale: int,
    is_generation: bool = False,
) -> torch.Tensor:
    """Grouped (cross-layer fan-out) variant of the local-topk → global-index
    remap (``TRTLLM_DISABLE_DSA_GROUP_REMAP``).

    Computes the global-index output for a whole full+shared indexer group in a
    single launch. All members of a group share the same top-k selection,
    ``req_idx`` and ``block_table`` (constant across layers within a forward);
    only the additive ``layer_offset * tokens_per_block`` term differs per
    member. ``layer_ids`` (int32 CUDA tensor of length ``group_size``) carries
    each member's layer offset; ``page_index_scale`` is the (uniform) primary-
    pool page scale for the group.

    Returns a tensor of shape ``[group_size, num_tokens, topk]``; slice ``[z]``
    is bit-identical to ``transform_local_topk_and_prepare_pool_view`` for the
    member with offset ``layer_ids[z]``. MLA-only path.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global_grouped(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        page_index_scale * attn_metadata._cached_tokens_per_block,
        layer_ids,
    )

    return global_indices


def split_prefill_chunks(
    seq_lens: torch.Tensor,
    max_chunk_size: int,
    start_idx: int = 0,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split prefill requests into chunks based on max_chunk_size.
    Supports two-level chunking:
    1. Request-boundary chunking: group multiple small requests into one chunk
    2. Intra-request chunking: split large requests into multiple Q-block chunks

    Args:
        seq_lens: Sequence lengths for all requests
        max_chunk_size: Maximum number of tokens per chunk
        start_idx: Starting index for prefill requests

    Returns:
        List of chunk groups, where each group is a list of chunk specs.
        Each chunk spec is (req_idx, token_start_in_req, token_end_in_req, req_cum_start)

        - For multi-request chunks: group contains multiple specs (one per request)
        - For intra-request chunks: each Q-block is a separate group with single spec
    """
    chunk_groups = []
    num_reqs = len(seq_lens)

    current_req = start_idx
    # Compute cumulative token positions
    query_start_loc_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cpu"), seq_lens.cumsum(dim=0).to(torch.int32)]
    )

    while current_req < num_reqs:
        seq_len = seq_lens[current_req].item()
        req_cum_start = query_start_loc_cpu[current_req].item()

        if seq_len <= max_chunk_size:
            # This request fits in one chunk - try to pack with others
            current_size = seq_len
            chunk_specs = [(current_req, 0, seq_len, req_cum_start)]
            next_req = current_req + 1

            # Try to add more requests to this chunk
            while next_req < num_reqs:
                next_seq_len = seq_lens[next_req].item()
                if next_seq_len > max_chunk_size:
                    # Next request is large, stop packing
                    break
                if current_size + next_seq_len <= max_chunk_size:
                    next_cum_start = query_start_loc_cpu[next_req].item()
                    chunk_specs.append((next_req, 0, next_seq_len, next_cum_start))
                    current_size += next_seq_len
                    next_req += 1
                else:
                    break

            # Add as one multi-request chunk group
            chunk_groups.append(chunk_specs)
            current_req = next_req
        else:
            # Large request - split into Q-blocks
            # Each Q-block is a separate chunk group (processed in separate iteration)
            num_q_blocks = (seq_len + max_chunk_size - 1) // max_chunk_size
            for q_block_idx in range(num_q_blocks):
                token_start = q_block_idx * max_chunk_size
                token_end = min(token_start + max_chunk_size, seq_len)
                q_block_spec = [(current_req, token_start, token_end, req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


def group_prefill_requests_by_logits_budget(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    elem_budget: int,
) -> List[List[int]]:
    """Group consecutive context requests into indexer prefill chunks.

    A request joins the current group only while the group's packed logits,
    ``sum(q_lens) * sum(kv_lens)``, stay within ``elem_budget``. A request that
    exceeds the budget on its own forms a single-request group.

    Args:
        q_lens: New (query) tokens per request.
        kv_lens: Compressed KV tokens per request, cached plus new.
        elem_budget: Maximum packed logits elements for a multi-request group.

    Returns:
        Groups of request indices, in order, covering every request once.
    """
    assert len(q_lens) == len(kv_lens)
    groups: List[List[int]] = []
    group_q = group_kv = 0
    for req_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if groups and (group_q + q_len) * (group_kv + kv_len) <= elem_budget:
            groups[-1].append(req_idx)
            group_q += q_len
            group_kv += kv_len
        else:
            groups.append([req_idx])
            group_q, group_kv = q_len, kv_len
    return groups


def plan_mla_chunked_prefill_chunks(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    elem_budget: int,
    q_begin: int = 0,
    q_end: Optional[int] = None,
) -> List[List[Tuple[int, int, int, int]]]:
    """Plan the indexer prefill chunks covering context query tokens ``[q_begin, q_end)``.

    Requests that lie fully inside the range are grouped with
    ``group_prefill_requests_by_logits_budget``. A request cut by the range
    boundary contributes only its covered query tokens, as a chunk of its own.

    Args:
        q_lens: New (query) tokens per request.
        kv_lens: Compressed KV tokens per request, cached plus new.
        elem_budget: Maximum packed logits elements for a multi-request group.
        q_begin: First context query token to cover.
        q_end: End of the covered context query tokens; defaults to all of them.

    Returns:
        Chunk specs for ``Indexer.prepare_one_prefill_chunk``, in token order:
        ``(req_idx, token_start_in_req, token_end_in_req, req_cum_start)``.
    """
    assert len(q_lens) == len(kv_lens)
    if q_end is None:
        q_end = sum(q_lens)
    q_starts = []
    q_start = 0
    for q_len in q_lens:
        q_starts.append(q_start)
        q_start += q_len

    chunks: List[List[Tuple[int, int, int, int]]] = []
    full_run: List[int] = []

    def flush_full_run() -> None:
        groups = group_prefill_requests_by_logits_budget(
            [q_lens[i] for i in full_run], [kv_lens[i] for i in full_run], elem_budget
        )
        for group in groups:
            chunks.append(
                [(full_run[g], 0, q_lens[full_run[g]], q_starts[full_run[g]]) for g in group]
            )
        full_run.clear()

    for req_idx, q_len in enumerate(q_lens):
        begin = max(q_begin, q_starts[req_idx])
        end = min(q_end, q_starts[req_idx] + q_len)
        if begin >= end:
            continue
        if end - begin == q_len:
            full_run.append(req_idx)
        else:
            flush_full_run()
            chunks.append(
                [
                    (
                        req_idx,
                        begin - q_starts[req_idx],
                        end - q_starts[req_idx],
                        q_starts[req_idx],
                    )
                ]
            )
    flush_full_run()
    return chunks


# Shrink the indexer prefill chunk size for very long requests to bound the
# fp8(_fp4)_mqa_logits activation memory (~ chunk_size * K_compressed), keyed on
# the largest compressed KV length in the batch. Entries are
# (k_compressed_lower_bound_exclusive, chunk_size); >512K -> 8K, [256K, 512K]
# -> 16K, otherwise unchanged.
_INDEXER_CHUNK_SIZE_HEURISTIC = (
    (512 * 1024, 8 * 1024),
    (256 * 1024 - 1, 16 * 1024),
)


def select_indexer_chunk_size(configured_chunk_size: int, max_k_compressed: int) -> int:
    """Pick the indexer prefill chunk size from the batch's largest K_compressed.

    Only reduces ``configured_chunk_size`` (never increases it).
    """
    for threshold, chunk_size in _INDEXER_CHUNK_SIZE_HEURISTIC:
        if max_k_compressed > threshold:
            return min(configured_chunk_size, chunk_size)
    return configured_chunk_size


def _select_indexer_compress_ratio(compress_ratios: List[int]) -> int:
    if 4 in compress_ratios:
        return 4
    if 1 in compress_ratios:
        return 1
    return 0


def _effective_compress_ratio_divisor(compress_ratio: int) -> int:
    return compress_ratio if compress_ratio > 1 else 1


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    compress_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)
        kv_lens: KV token lengths [num_contexts], dtype=torch.int32 (optional)
        compress_ratio: Compression ratio for KV tokens

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    if kv_lens is None:
        kv_lens = (
            seq_lens if cached_token_lens is None else cached_token_lens + seq_lens
        )  # [num_contexts]
        kv_lens = kv_lens // compress_ratio

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(kv_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32), seq_lens
    )  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens, device=device, dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens
    )  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens, seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = (
            cu_seqlen_ks + (cached_per_token + local_q_positions + 1) // compress_ratio
        )  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + (local_q_positions + 1) // compress_ratio  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


@dataclass
class IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""

    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch
    row_starts_zero: bool = (
        False  # every row's window starts at the chunk's first key (one request)
    )


def _split_prefill_queries(
    count: int, mapping: Optional[Mapping], threshold: int
) -> Tuple[int, int, Optional[List[int]]]:
    """Return local query bounds and optional sizes for a TP prefill gather."""
    if (
        threshold < 0
        or count < threshold
        or mapping is None
        or mapping.enable_attention_dp
        or mapping.tp_size <= 1
    ):
        return 0, count, None
    size, rank = mapping.tp_size, mapping.tp_rank
    sizes = [(r + 1) * count // size - r * count // size for r in range(size)]
    return count * rank // size, count * (rank + 1) // size, sizes


def _prefill_q_split_threshold(metadata: DSAtrtllmAttentionMetadata) -> int:
    """Query-token count from which prefill indexer queries are split across TP ranks."""
    sparse_metadata_params = metadata.sparse_metadata_params
    return sparse_metadata_params.q_split_threshold if sparse_metadata_params is not None else 8192


def _packed_logits_elem_budget(metadata: DSAtrtllmAttentionMetadata) -> int:
    """Packed-logits cap for grouping context requests into one prefill chunk.

    A single request's logits tile is at most ``min(_INDEXER_MQA_LOGITS_ELEM_BUDGET,
    max_num_tokens * max_seq_len)`` elements, the size the KV cache estimate
    reserves for the indexer logits workspace where it reserves one. Clamping
    the packed cap to it keeps a packed chunk within that reserve.
    """
    single_request_tile = min(
        _INDEXER_MQA_LOGITS_ELEM_BUDGET,
        metadata.max_num_tokens * metadata.kv_cache_manager.max_seq_len,
    )
    return min(_INDEXER_PACKED_LOGITS_ELEM_BUDGET, single_request_tile)


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor, s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    """Cast hidden states to float32 for TF32 GEMM computation."""
    return hidden_states.float()


@contextmanager
def _tf32_matmul_enabled():
    """Temporarily enable TF32 tensor cores for FP32 matmul in this scope.

    Forces PyTorch/cuBLASLt to use CUBLAS_COMPUTE_32F_FAST_TF32, which
    guarantees TF32 tensor cores. Plain CUBLAS_COMPUTE_32F (used by
    torch.ops.trtllm.cublas_mm) falls back to SIMT SGEMM on CUDA cores
    based on cuBLASLt heuristics for small M.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@dataclass
class IndexerParams:
    """
    Parameters for indexer.
    """

    num_contexts: int
    num_generations: int
    num_ctx_tokens: int
    head_dim: int
    quant_block_size: int
    tokens_per_block: int
    compress_ratio: int
    request_ids: List[int]
    num_past_tokens: List[int]
    seq_lens: torch.Tensor
    # Bytes of quantized data per token in the indexer K cache; defaults to
    # head_dim (one FP8 byte per element). For MXFP4 use head_dim // 2.
    data_bytes_per_token: Optional[int] = None

    def __post_init__(self):
        # Pre-compute frequently used tensors once instead of on every property access
        num_past_tokens_tensor = torch.tensor(self.num_past_tokens, dtype=torch.int32)
        self._num_past_tokens_tensor = num_past_tokens_tensor
        compress_ratio = self.compress_ratio
        self._cached_kv_tokens = num_past_tokens_tensor // compress_ratio
        self._all_kv_tokens = (self.seq_lens + num_past_tokens_tensor) // compress_ratio
        self._new_kv_tokens = self._all_kv_tokens - self._cached_kv_tokens
        self._kv_lens = self._all_kv_tokens
        if self.data_bytes_per_token is None:
            self.data_bytes_per_token = self.head_dim
        self._scale_size = self.head_dim // self.quant_block_size * 4
        self._block_stride = self.tokens_per_block * (self.data_bytes_per_token + self._scale_size)

    @property
    def batch_size(self):
        return len(self.request_ids)

    @property
    def kv_lens(self):
        return self._kv_lens

    @property
    def total_tokens(self):
        return self.seq_lens.sum().item()

    @property
    def cached_kv_tokens(self):
        return self._cached_kv_tokens

    @property
    def all_kv_tokens(self):
        return self._all_kv_tokens

    @property
    def new_kv_tokens(self):
        return self._new_kv_tokens

    @property
    def scale_size(self):
        return self._scale_size

    @property
    def block_stride(self):
        return self._block_stride


class Indexer(nn.Module):
    """DSA sparse attention indexer that selects top-K KV cache entries per token."""

    projection_free: bool = False

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DSAParams,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
        *,
        projection_free: bool = False,
    ):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        self.projection_free = projection_free
        sm_version = get_sm_version() if torch.cuda.is_available() else 0
        if not projection_free and (mla_params is None or pos_embd_params is None):
            raise ValueError("Projecting indexers require MLA and positional parameters")
        if projection_free and sparse_params.enable_heuristic_topk:
            raise ValueError("Prepared indexers require exact Top-K selection")
        self.hidden_size = mla_params.hidden_size if mla_params is not None else None
        self.q_lora_rank = mla_params.q_lora_rank if mla_params is not None else None
        if sparse_params.index_rope_dim is not None:
            self.rope_dim = sparse_params.index_rope_dim
        else:
            self.rope_dim = mla_params.qk_rope_head_dim if mla_params is not None else 0
        self.n_heads = sparse_params.index_n_heads  # 64
        self.head_dim = sparse_params.index_head_dim  # 128
        self.index_topk = sparse_params.index_topk  # 2048
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio

        self.wq_b = self.wk = self.k_norm = self.weights_proj = self.rotary_emb = None
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None
        self._indexer_bf16 = False
        if not projection_free:
            self.wq_b = Linear(
                self.q_lora_rank,
                self.n_heads * self.head_dim,
                bias=False,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )
            # When TRTLLM_DSA_INDEXER_BF16=1, run the fused wk + weights_proj
            # projection in the model dtype instead of fp32. The fp32 path has no
            # native GEMM on recent architectures (it falls back to a TF32 tensor-
            # core kernel) and requires bf16<->fp32 casts around it; the model-dtype
            # path runs a single native GEMM with no casts. The default stays fp32
            # because the indexer top-k selection is precision sensitive.
            self._indexer_bf16 = os.environ.get("TRTLLM_DSA_INDEXER_BF16", "0") == "1"
            _wk_wp_dtype = dtype if self._indexer_bf16 else torch.float32
            self.wk = Linear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                dtype=_wk_wp_dtype,
                quant_config=None,
                skip_create_weights_in_init=skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )
            k_norm_eps = sparse_params.index_k_norm_eps
            self.k_norm = LayerNorm(
                hidden_size=self.head_dim, eps=1e-6 if k_norm_eps is None else k_norm_eps
            )
            self.weights_proj = Linear(
                self.hidden_size,
                self.n_heads,
                bias=False,
                dtype=_wk_wp_dtype,
                quant_config=None,
                skip_create_weights_in_init=skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )

            # Fused wk + weights_proj weight for a single F.linear GEMM. fp32 by
            # default (TF32 tensor cores on Ampere+); model dtype when
            # TRTLLM_DSA_INDEXER_BF16 is enabled.
            self._fused_wk_wp_weight: Optional[torch.Tensor] = None

            if self.rope_dim > 0:
                indexer_rope_interleave = sparse_params.indexer_rope_interleave
                self.rotary_emb = RotaryEmbedding(
                    pos_embd_params.rope,
                    head_dim=self.rope_dim,
                    is_neox=not indexer_rope_interleave,
                )

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        # indexer_k_dtype controls both Q and K precision. DeepGEMM's
        # fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels only dispatch
        # to FP4xFP4 or FP8xFP8 (no mixed-precision variant). The DeepGEMM
        # kernel asserts SM100 + head_dim=128 at launch time under FP4. BF16
        # keeps both unquantized and scores them with the Triton kernels.
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.use_bf16 = sparse_params.indexer_k_dtype == "bf16"
        # Kimi K3.1 MXFP4 index keys: the prefill logits of a disaggregated context server and the decode logits of a
        # generation server reduce the heads in BF16 (see _INDEXER_BF16_LOGITS_CUTE and _decode_logits_bf16_kwargs).
        role = disagg_role()
        self.bf16_prefill_logits = self.use_fp4 and is_kimi_k31_dsa() and role == "context"
        self.bf16_decode_logits = self.use_fp4 and is_kimi_k31_dsa() and role == "generation"
        # Kimi K3.1: the deterministic self-sampling GVR decode of a generation-only call writes the paged-pool
        # indices (convertReqIndexToGlobal's integers) instead of the local ones where its main family serves the
        # rows, and the attention takes them without the remap kernel (same indices). Only for rows whose local
        # indices nothing else reads (no GVR prior, no shared / MTP-shared Top-K, no Helix, uniform next_n, compress
        # ratio 1).
        self.gvr_paged_remap = is_kimi_k31_dsa()
        self.k_data_bytes_per_token = get_indexer_k_data_bytes_per_token(
            self.head_dim, sparse_params.indexer_k_dtype
        )
        self.aux_stream = aux_stream
        self.ln_events = [] if projection_free else [torch.cuda.Event(), torch.cuda.Event()]
        # Fork/join events for the aux-stream prev_topk write-back.
        self.prev_topk_copy_events = (
            [] if projection_free else [torch.cuda.Event(), torch.cuda.Event()]
        )
        self._prev_topk_copy_pending = False
        self.use_cute_dsl_topk = sparse_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        self.use_cute_dsl_paged_mqa_logits = use_cute_dsl_paged_mqa_logits(
            sparse_params.use_cute_dsl_paged_mqa_logits
        )
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = sparse_params.enable_heuristic_topk and sm_version >= 100
        # Two-level GVR dispatch: enable_heuristic_topk selects the GVR
        # family over the exact radix path; use_self_sampling_topk (default
        # True) selects the hint-free self-sampling engine over the
        # temporal-hint engines. Tuning is frozen from indexer_max_seq_len
        # at capture time, so the launch is CUDA-graph-replay safe. The
        # TopK module's hardware-format gate falls back to the exact
        # insertion/radix path with a one-time warning; contract violations
        # inside the engine raise.
        self._use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self._enable_heuristic_topk,
            use_self_sampling_topk=sparse_params.use_self_sampling_topk,
            index_topk=sparse_params.index_topk,
            compress_ratio=compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=sm_version,
        )
        if os.environ.get("TRTLLM_GVR_SELF_SAMPLING") is not None:
            logger.warning_once(
                "TRTLLM_GVR_SELF_SAMPLING is retired and ignored: the "
                "self-sampling GVR engine is selected by the "
                "use_self_sampling_topk sparse-attention config field "
                "(default True) when enable_heuristic_topk is set.",
                key="gvr_self_sampling_env_retired",
            )
        if (
            self._enable_heuristic_topk
            and sparse_params.use_self_sampling_topk
            and not self._use_self_sampling_topk
        ):
            logger.warning_once(
                "use_self_sampling_topk=True but the self-sampling GVR "
                "prerequisites are not met "
                f"(cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                f"sm={sm_version}, "
                f"index_topk={sparse_params.index_topk}, "
                f"compress_ratio={compress_ratio}); falling back to the "
                "temporal GVR path (exact radix when the DSL engine is "
                "unavailable).",
                key="gvr_self_sampling_prereq_fallback",
            )
        self.mtp_index_share = sparse_params.mtp_index_share

        if (
            self._enable_heuristic_topk
            and IS_CUTLASS_DSL_AVAILABLE
            # datacenter Blackwell only; consumer Blackwell (sm_120/121)
            # lacks the thread-block clusters both GVR engines use. Rubin
            # (sm_107) runs the self-sampling engine only.
            and (sm_version in (100, 103) or (sm_version == 107 and self._use_self_sampling_topk))
        ):
            decode_top_k_implementation = TopKImplementation.CUTE_DSL_GVR
        else:
            if self._enable_heuristic_topk:
                logger.warning_once(
                    "enable_heuristic_topk=True but the DSL GVR engine is "
                    f"unavailable (cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                    f"sm={get_sm_version()}); using the exact radix decode "
                    "top-K instead.",
                    key="gvr_prereq_radix_fallback",
                )
            decode_top_k_implementation = (
                TopKImplementation.CUTE_DSL_RADIX
                if self.use_cute_dsl_topk
                else TopKImplementation.CUDA_RADIX
            )
        # Prefill uses the self-sampling engine on exactly the layers where the
        # decode dispatch picks it; the temporal-hint engine has no prefill form.
        prefill_top_k_implementation = (
            TopKImplementation.CUTE_DSL_GVR
            if (
                decode_top_k_implementation == TopKImplementation.CUTE_DSL_GVR
                and self._use_self_sampling_topk
            )
            else TopKImplementation.CUDA_RADIX
        )
        self.top_k = TopK(
            self.index_topk,
            prefill_implementation=prefill_top_k_implementation,
            decode_implementation=decode_top_k_implementation,
            compress_ratio=self.compress_ratio,
            gvr_self_sampling=self._use_self_sampling_topk,
            gvr_deterministic=sparse_params.gvr_deterministic_topk,
        )
        # Emission block-skip is a temporal-hint (V1) optimization: the FP4
        # indexer epilogue emits per-block max logits the GVR Top-K consumes to
        # skip whole blocks (see gvr_emission / gvr_routing; state lives on the
        # TopK module). Off on the self-sampling path (no cross-step state to
        # assist) and off non-FP4 / non-paged-MQA layers.
        self.use_gvr_emission = (
            sparse_params.use_gvr_emission
            and not self._use_self_sampling_topk
            and decode_top_k_implementation == TopKImplementation.CUTE_DSL_GVR
            and self.use_cute_dsl_paged_mqa_logits
            and self.use_fp4
        )

        # Fused wk + weights_proj weight for single FP32 cuBLAS GEMM
        # (populated in cache_derived_state; maps to TF32 tensor cores on Ampere+)
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

    def cache_derived_state(self) -> None:
        """Fuse wk and weights_proj for F.linear with TF32 tensor cores on Ampere+."""
        if self.projection_free:
            return
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    @staticmethod
    def prepare_one_prefill_chunk(
        metadata: DSAtrtllmAttentionMetadata,
        chunk_specs: List[Tuple[int, int, int, int]],
    ) -> IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - token_start_in_req, token_end_in_req are indices into current batch context tokens
                        - For multi-request: multiple specs from different requests (full requests)
                        - For intra-request: single spec from one request's Q-block

        Note: Cached token counts are derived from metadata.host_ctx_cached_token_indptr
        """
        device = metadata.cu_seqlen_ks.device
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1]
                - metadata.host_ctx_cached_token_indptr[req_idx]
            ).item()

            # Total compressed KV tokens for this request
            req_kv_len = (num_cached + token_end_in_req) // compress_ratio

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:req_kv_len] in compressed KV space
            cu_seqlen_ks = torch.zeros(num_q_tokens, dtype=torch.int32, device="cpu")
            cu_seqlen_ke = (
                torch.arange(
                    token_start_in_req + 1, token_end_in_req + 1, dtype=torch.int32, device="cpu"
                )
                + num_cached
            ) // compress_ratio

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping in compressed KV space
            # For req_idx=0 the offset is 0; for other indices compute compressed cumulative offset
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                // compress_ratio
                for j in range(req_idx)
            )
            total_kv_for_req = req_kv_len
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_num_past_tokens = []
            req_kv_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_past_tokens = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1]
                    - metadata.host_ctx_cached_token_indptr[req_idx]
                ).item()
                req_num_past_tokens.append(num_past_tokens)
                req_kv_lens.append((num_past_tokens + req_seq_lens[-1]) // compress_ratio)

            req_seq_lens_tensor = torch.tensor(req_seq_lens, dtype=torch.int32, device="cpu")
            req_num_past_tokens_tensor = torch.tensor(
                req_num_past_tokens, dtype=torch.int32, device="cpu"
            )
            req_kv_lens_tensor = torch.tensor(req_kv_lens, dtype=torch.int32, device="cpu")
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor,
                len(chunk_specs),
                num_q_tokens,
                req_num_past_tokens_tensor,
                req_kv_lens_tensor,
                compress_ratio,
            )

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            # Must use compressed offsets
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                // compress_ratio
                for j in range(first_req_idx)
            )
            total_kv_len = sum(req_kv_lens)
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, (
            "Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: "
            f"{cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        )
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, (
            "Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: "
            f"{cu_seqlen_ke.shape[0]} != {num_q_tokens}"
        )

        row_starts_zero = not bool(cu_seqlen_ks.any())
        return IndexerPrefillChunkMetadata(
            row_starts_zero=row_starts_zero,
            cu_seqlen_ks=maybe_pin_memory(cu_seqlen_ks).to(device, non_blocking=True),
            cu_seqlen_ke=maybe_pin_memory(cu_seqlen_ke).to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def build_indexer_params(metadata: DSAtrtllmAttentionMetadata) -> Optional[IndexerParams]:
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "index_head_dim"):
            return None

        head_dim = metadata.indexer_head_dim
        data_bytes_per_token = get_indexer_k_data_bytes_per_token(
            head_dim, resolve_indexer_k_dtype(kv_cache_manager)
        )
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        return IndexerParams(
            num_contexts=metadata.num_contexts,
            num_generations=metadata.num_generations,
            num_ctx_tokens=metadata.num_ctx_tokens,
            head_dim=head_dim,
            quant_block_size=metadata.indexer_quant_block_size,
            tokens_per_block=metadata._tokens_per_block,
            compress_ratio=compress_ratio,
            request_ids=metadata.request_ids,
            num_past_tokens=metadata.kv_cache_params.num_cached_tokens_per_seq,
            seq_lens=metadata.seq_lens,
            data_bytes_per_token=data_bytes_per_token,
        )

    @staticmethod
    def recompute_slot_mappings(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: Optional[IndexerParams] = None
    ):
        """Recompute slot_mapping_fp8/scale from current block offsets.

        This is the subset of prepare_for_update_k_cache() that maps each new
        compressed KV token to its flat cache position. It is safe to call in
        isolation after a caller has swapped the active KV cache manager and
        block-offset buffers, such as during draft KV-cache replay.
        """
        if indexer_params is None:
            indexer_params = Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        batch_size = indexer_params.batch_size
        tokens_per_block = indexer_params.tokens_per_block
        head_dim = indexer_params.head_dim
        new_kv_tokens = indexer_params.new_kv_tokens
        total_new_kv_tokens = new_kv_tokens.sum().item()
        data_bytes_per_token = indexer_params.data_bytes_per_token

        # Compute global positions for all kv tokens in the batch (fully vectorized)
        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device="cpu"), new_kv_tokens
        )
        # Vectorized token_offsets: arange(total) - cumulative start per request
        cu_new_kv = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
        cu_new_kv[1:] = new_kv_tokens.to(torch.int64).cumsum(0)
        token_offsets = torch.arange(
            total_new_kv_tokens, dtype=torch.int64, device="cpu"
        ) - cu_new_kv[:-1].repeat_interleave(new_kv_tokens)
        global_positions = indexer_params.cached_kv_tokens[req_indices] + token_offsets

        # Block indices/pos for all kv tokens in the batch
        block_indices_in_seq = global_positions // tokens_per_block

        max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            indexer_params.quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_new_kv_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_new_kv_tokens] = scale_flat_indices

        stage_or_copy(
            metadata.slot_mapping_fp8[:total_new_kv_tokens],
            metadata.host_slot_mapping_fp8[:total_new_kv_tokens],
        )
        stage_or_copy(
            metadata.slot_mapping_scale[:total_new_kv_tokens],
            metadata.host_slot_mapping_scale[:total_new_kv_tokens],
        )
        if metadata.enable_helix and metadata.num_generations > 0:
            Indexer._helix_host_gen_slot_mappings(metadata, indexer_params)

    @staticmethod
    def _helix_host_gen_slot_mappings(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ) -> None:
        """Give every generation row its own Helix-aware indexer-K slot.

        The scatter pairs K row i with slot i, but under Helix a rank appends
        only the generation tokens it owns, so the counts from new_kv_tokens
        would leave rows paired with other tokens' (or stale) slots. Here the
        active rank writes its single token at the local tail and other ranks
        get -1, which the scatter skips. Verify groups get placeholders that
        on_update_kv_lens() replaces from the per-token Helix buffers.
        """
        num_ctx_tokens = metadata.num_ctx_tokens
        num_gen_tokens = metadata.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return
        if metadata._helix_spec_tokens_valid:
            fp8_flat_indices = torch.full((num_gen_tokens,), -1, dtype=torch.int64)
            scale_flat_indices = fp8_flat_indices
        else:
            assert num_gen_tokens == metadata.num_generations, (
                "Helix generation without verify groups appends one token per request"
            )
            gen = slice(metadata.num_contexts, metadata.num_seqs)
            inactive = metadata.helix_is_inactive_rank_cpu[: metadata.num_generations]
            positions = indexer_params.cached_kv_tokens[gen].to(torch.int64)
            positions = torch.where(inactive, torch.zeros_like(positions), positions)
            req_indices = torch.arange(
                metadata.num_contexts, metadata.num_seqs, dtype=torch.int64, device="cpu"
            )
            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                positions,
                metadata.host_indexer_k_cache_block_offsets,
                req_indices,
                indexer_params.head_dim,
                indexer_params.tokens_per_block,
                indexer_params.quant_block_size,
                data_bytes_per_token=indexer_params.data_bytes_per_token,
            )
            fp8_flat_indices = fp8_flat_indices.masked_fill(inactive, -1)
            scale_flat_indices = scale_flat_indices.masked_fill(inactive, -1)
        gen_rows = slice(num_ctx_tokens, num_ctx_tokens + num_gen_tokens)
        metadata.host_slot_mapping_fp8[gen_rows] = fp8_flat_indices
        metadata.host_slot_mapping_scale[gen_rows] = scale_flat_indices
        metadata.slot_mapping_fp8[gen_rows].copy_(
            metadata.host_slot_mapping_fp8[gen_rows], non_blocking=True
        )
        metadata.slot_mapping_scale[gen_rows].copy_(
            metadata.host_slot_mapping_scale[gen_rows], non_blocking=True
        )

    @staticmethod
    def recompute_context_kv_gather_mappings(
        metadata: DSAtrtllmAttentionMetadata,
        indexer_params: Optional[IndexerParams] = None,
    ) -> None:
        """Recompute context KV-cache gather mappings for the active cache manager."""
        if not metadata.enable_context_mla_with_cached_kv:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        if indexer_params is None:
            indexer_params = Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        num_contexts = indexer_params.num_contexts
        if num_contexts == 0:
            # Generation does not consume full-KV gather mappings. In
            # particular, avoid allocating temporary zero-length tensors
            # while capturing generation CUDA graphs.
            return

        if metadata.skip_indexer_for_ctx_reqs:
            # The dense short-context path does not gather from the indexer K
            # cache, so full-KV mappings would be allocated but never consumed.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        cached_kv_tokens = indexer_params.cached_kv_tokens[:num_contexts]
        if indexer_params.compress_ratio == 1 and cached_kv_tokens.count_nonzero().item() == 0:
            # Without compression or cached tokens, the regular mappings cover
            # every context KV token in the same order and remain equivalent
            # when on_update_kv_lens() refreshes them in place.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        total_kv_per_request = indexer_params.kv_lens[:num_contexts]
        total_kv_len = total_kv_per_request.sum().item()
        if total_kv_len == 0:
            # No chunk can gather indexer KV, so avoid zero-length allocation
            # and H2D work for short compressed contexts.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        host_slot_mapping_fp8_fullkv = torch.empty(
            total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
        )
        host_slot_mapping_scale_fullkv = torch.empty(
            total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
        )

        req_indices = torch.repeat_interleave(
            torch.arange(num_contexts, dtype=torch.int64, device="cpu"), total_kv_per_request
        )

        cu_kv = torch.zeros(num_contexts + 1, dtype=torch.int64, device="cpu")
        cu_kv[1:] = total_kv_per_request.to(torch.int64).cumsum(0)
        kv_positions = torch.arange(total_kv_len, dtype=torch.int64, device="cpu") - cu_kv[
            :-1
        ].repeat_interleave(total_kv_per_request)

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            kv_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            indexer_params.head_dim,
            indexer_params.tokens_per_block,
            indexer_params.quant_block_size,
            data_bytes_per_token=indexer_params.data_bytes_per_token,
        )

        host_slot_mapping_fp8_fullkv.copy_(fp8_flat_indices)
        host_slot_mapping_scale_fullkv.copy_(scale_flat_indices)
        metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(non_blocking=True)
        metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(non_blocking=True)

    @staticmethod
    def prepare_for_update_k_cache(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the update_k_cache stage.

        Compute slot_mapping for all requests (both context and generation)
        This maps each token to its flat cache position for vectorized KV cache updates
        """
        Indexer.recompute_slot_mappings(metadata, indexer_params)

    @staticmethod
    def prepare_for_chunked_prefill(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the chunked prefill.
        """
        num_contexts = indexer_params.num_contexts
        seq_lens = indexer_params.seq_lens
        metadata.indexer_prefill_q_split = None

        # When MLA chunked prefill is active, it already bounds the new tokens per
        # request, so requests are not split by length here. From
        # q_split_threshold context query tokens on, the batch's queries are
        # split evenly across TP ranks and each rank plans chunks for its share
        # only. Within that share, requests are grouped so that a chunk's packed
        # logits stay bounded: with cached KV, a request's KV can be far longer
        # than its new tokens.
        has_mla_chunked_prefill = (
            metadata.enable_context_mla_with_cached_kv and metadata.runtime_features.chunked_prefill
        )
        if has_mla_chunked_prefill:
            compress_ratio = _effective_compress_ratio_divisor(
                _select_indexer_compress_ratio(metadata.compress_ratios)
            )
            q_lens = seq_lens[:num_contexts].tolist()
            cached_indptr = metadata.host_ctx_cached_token_indptr[: num_contexts + 1].tolist()
            kv_lens = [
                (cached_indptr[i + 1] - cached_indptr[i] + q_lens[i]) // compress_ratio
                for i in range(num_contexts)
            ]
            q_begin, q_end, q_sizes = _split_prefill_queries(
                sum(q_lens), metadata.mapping, _prefill_q_split_threshold(metadata)
            )
            if q_sizes is not None:
                metadata.indexer_prefill_q_split = (q_begin, q_end, q_sizes)
            metadata.indexer_prefill_chunks = [
                Indexer.prepare_one_prefill_chunk(metadata, chunk_specs)
                for chunk_specs in plan_mla_chunked_prefill_chunks(
                    q_lens, kv_lens, _packed_logits_elem_budget(metadata), q_begin, q_end
                )
            ]
        else:
            # Use indexer's chunking to avoid quadratic indexer MQA logits
            # computation for long sequences.
            # This is only used when MLA chunked prefill is not enabled.
            # Adapt chunk size to the batch's largest K_compressed (see
            # select_indexer_chunk_size).
            max_k_compressed = (
                int(indexer_params.kv_lens[:num_contexts].max().item()) if num_contexts > 0 else 0
            )
            effective_chunk_size = select_indexer_chunk_size(
                metadata.indexer_max_chunk_size, max_k_compressed
            )
            chunk_groups = split_prefill_chunks(
                seq_lens[:num_contexts],
                effective_chunk_size,
                start_idx=0,
            )

            if len(chunk_groups) > 1 or metadata.enable_context_mla_with_cached_kv:
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                    for chunk_specs in chunk_groups
                ]
            else:
                metadata.indexer_prefill_chunks = None

        # Chunked prefill and KV-cache reuse require gather mappings that cover
        # all cached and newly appended indexer tokens.
        Indexer.recompute_context_kv_gather_mappings(metadata, indexer_params)

    @staticmethod
    def prepare_scheduler_metadata(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare scheduler metadata for the DeepGEMM decode MQA kernel.
        """
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        gen_seq_lens = metadata.get_indexer_kv_lens(
            metadata.kv_lens_cuda_runtime[num_contexts : num_contexts + num_generations]
        )
        metadata.gen_indexer_kv_lens_cuda_runtime = gen_seq_lens
        if not metadata.use_expanded_buffers_for_mtp:
            scheduler_context_lens = metadata.update_indexer_kv_lens_2d(gen_seq_lens)
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                gen_seq_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, metadata.num_sms
            )
            metadata.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if metadata.max_draft_tokens > 0:
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    scheduler_context_lens.contiguous(),
                    _DG_SCHEDULE_BLOCK_KV,
                    metadata.num_sms,
                )
                metadata.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True
                )
        else:
            # Expand schedule metadata buffer (only generation). The DeepGEMM
            # API requires 2D; each expanded token becomes a (1,) row.
            num_tokens = sum(metadata.gen_token_repeat_list())
            scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                metadata.num_sms,
            )
            metadata.scheduler_metadata_buffer_expanded.copy_(
                scheduler_metadata_buffer_expanded, non_blocking=True
            )

    @staticmethod
    def prepare(metadata: DSAtrtllmAttentionMetadata) -> None:
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """

        # Skip indexer preparation if the kv_cache_manager doesn't have index_head_dim.
        # This can happen when the metadata is being used with a draft KV cache manager
        # during MTP speculative decoding, which uses a regular KVCacheManager instead
        # of DSACacheManager.
        indexer_params = Indexer.build_indexer_params(metadata)
        if indexer_params is None:
            return

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        # Store compressed KV token count for context requests
        metadata.num_ctx_kv_tokens = indexer_params.new_kv_tokens[:num_contexts].sum().item()

        # Prepare for update_k_cache
        Indexer.prepare_for_update_k_cache(metadata, indexer_params)

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            host_num_past_tokens = indexer_params._num_past_tokens_tensor[:num_contexts]
            host_kv_lens = indexer_params.kv_lens[:num_contexts]
            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens,
                num_contexts,
                num_ctx_tokens,
                host_num_past_tokens,
                host_kv_lens,
                compress_ratio,
            )

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ks), non_blocking=True
            )
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ke), non_blocking=True
            )
            Indexer.prepare_for_chunked_prefill(metadata, indexer_params)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            # from the device KV lengths, which a batched input commit may still hold.
            flush_input_commit()
            Indexer.prepare_scheduler_metadata(metadata)

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
    ) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        if k_scale.element_size() == 1:
            # The op expects 4 byte elements.
            k_scale = k_scale.view(torch.int32)
        if self.use_bf16:
            # The op copies 1-byte elements.
            k_fp8 = k_fp8.view(torch.uint8)
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k_fp8,
            k_scale,
            k_cache,
            metadata.slot_mapping_fp8,
            metadata.slot_mapping_scale,
            num_tokens,
        )

    def _gather_k_cache_for_chunk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        chunk: IndexerPrefillChunkMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K values from indexer cache for a specific chunk.

        Uses pre-computed extended slot mappings that cover cached + current batch context tokens.
        chunk.k_token_start/k_token_end directly index into the extended slot mapping.

        Args:
            metadata: Attention metadata
            chunk: Chunk metadata with k_token_start/end as indices into extended slot mapping

        Returns:
            k_fp8: FP8 quantized k tensor, shape [num_k_tokens, head_dim]
            k_scale: Scaling factors, shape [num_k_tokens, 1]
        """
        assert metadata.slot_mapping_fp8_fullkv is not None, (
            "_gather_k_cache_for_chunk requires extended slot mappings (only available with cached tokens)"
        )

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)

        head_dim = self.k_data_bytes_per_token
        scale_size = 4  # float32 = 4 bytes

        # Extract slot mappings using chunk's k_token_start/end
        # These indices point directly into the extended slot mapping array
        k_token_start = chunk.k_token_start
        k_token_end = chunk.k_token_end
        num_k_tokens = k_token_end - k_token_start

        slot_mapping_fp8_chunk = metadata.slot_mapping_fp8_fullkv[k_token_start:k_token_end]
        slot_mapping_scale_chunk = metadata.slot_mapping_scale_fullkv[k_token_start:k_token_end]

        # Vectorized gather using pre-computed slot mappings
        # Gather FP8 data
        byte_offsets_fp8 = torch.arange(head_dim, device=k_cache.device).unsqueeze(
            0
        )  # [1, head_dim]
        gather_indices_fp8 = (
            slot_mapping_fp8_chunk.unsqueeze(1) + byte_offsets_fp8
        )  # [num_k_tokens, head_dim]
        assert (gather_indices_fp8 >= k_cache.numel()).sum() == 0, "Out-of-bounds access detected"
        gather_indices_fp8 = _unravel_indices(gather_indices_fp8, k_cache.shape)
        k_fp8_bytes = k_cache[gather_indices_fp8]
        k_dtype = torch.bfloat16 if self.use_bf16 else torch.float8_e4m3fn
        k_fp8 = k_fp8_bytes.view(num_k_tokens, head_dim).view(k_dtype)

        # Gather scale data
        byte_offsets_scale = torch.arange(scale_size, device=k_cache.device).unsqueeze(0)  # [1, 4]
        gather_indices_scale = (
            slot_mapping_scale_chunk.unsqueeze(1) + byte_offsets_scale
        )  # [num_k_tokens, 4]
        assert (gather_indices_scale >= k_cache.numel()).sum() == 0, "Out-of-bounds access detected"
        gather_indices_scale = _unravel_indices(gather_indices_scale, k_cache.shape)
        k_scale_bytes = k_cache[gather_indices_scale]
        k_scale = k_scale_bytes.view(torch.float32).view(num_k_tokens, 1)

        return k_fp8, k_scale

    def forward_prepared(
        self,
        q_data: torch.Tensor,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        tier_rows: Optional[int] = None,
        row_starts_zero: bool = False,
    ) -> torch.Tensor:
        """Run shared dense MQA and prefill Top-K on prequantized inputs.

        Row bounds index the shared K matrix. Output preserves DSA's row-local
        unsorted offsets. Only those row windows are read by Top-K, so the
        existing no-clean logits fast path remains valid. ``tier_rows``: the
        Top-K plan of a ``tier_rows``-row tile (``TopK.forward``'s
        ``prefill_tier_rows``) for a slice of that tile. ``row_starts_zero``: host knowledge that every row start
        is 0 (one request), a hint for the BF16-reduce logits kernel choice.
        """
        if output_indices.shape != (q_data.shape[0], self.index_topk):
            raise ValueError("Prepared output must match query rows and configured Top-K")
        if q_data.shape[0] == 0 or k_data.shape[0] == 0:
            return output_indices.fill_(-1)
        logits = self._call_mqa_logits(
            q_data,
            k_data,
            k_scale,
            weights,
            row_starts,
            row_ends,
            q_scale,
            clean_logits=False,
            bf16_logits=self.bf16_prefill_logits,
            row_starts_zero=row_starts_zero,
        )
        if logits.dtype != torch.float32:
            # The prefill Top-K engines read FP32 scores; BF16 -> FP32 is exact.
            logits = logits.float()
        return self.top_k(
            logits,
            output_indices,
            is_prefill=True,
            row_starts=row_starts,
            row_ends=row_ends,
            **({"prefill_tier_rows": tier_rows} if tier_rows is not None else {}),
        )

    def prefill_q_tile(self, q_rows: int, num_k_tokens: int, q_is_cuda: bool) -> int:
        """Query rows per fp8_mqa_logits call of a prefill chunk (logits element budget)."""
        budget = _INDEXER_MQA_LOGITS_ELEM_BUDGET
        if self.use_fp4 and (self.head_dim != 128 or not q_is_cuda or get_sm_version() < 100):
            # Decoded MQA also materializes per-head dot products.
            budget = max(1, budget // self.n_heads)
        return max(1, min(q_rows, budget // max(1, num_k_tokens)))

    @staticmethod
    def prefill_tile_pieces(lo: int, hi: int, rows: int, tile: int) -> List[Tuple[int, int, int]]:
        """Rows ``[lo, hi)`` of a ``rows``-row prefill chunk cut at its ``tile``-row query tiles:
        ``(start, end, rows of the enclosing tile)`` per piece, for ``forward_prepared(tier_rows=...)``."""
        pieces = []
        t0 = lo
        while t0 < hi:
            tile_lo = t0 // tile * tile
            tile_hi = min(tile_lo + tile, rows)
            t1 = min(tile_hi, hi)
            pieces.append((t0, t1, tile_hi - tile_lo))
            t0 = t1
        return pieces

    @staticmethod
    def _decode_mxfp4(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Decode existing FP4 index inputs without introducing FP8 rounding."""
        data = data.contiguous().view(torch.uint8)
        codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
        magnitudes = torch.arange(8, device=data.device)
        levels = torch.where(
            magnitudes < 4,
            magnitudes.float() * 0.5,
            torch.exp2(((magnitudes - 2) // 2).float()) * (1 + 0.5 * (magnitudes % 2)),
        )
        values = levels[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
        groups = values.shape[-1] // 32
        scale_bytes = scales.contiguous().view(torch.uint8).reshape(*values.shape[:-1], groups)
        decoded = values.reshape(*values.shape[:-1], groups, 32)
        return (decoded * torch.exp2(scale_bytes.float() - 127)[..., None]).flatten(-2)

    def _call_mqa_logits(
        self,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        clean_logits: bool = True,
        bf16_logits: bool = False,
        row_starts_zero: bool = False,
    ) -> torch.Tensor:
        """Dispatch fp8_mqa_logits, fp8_fp4_mqa_logits or the BF16 kernel.

        For FP4 the gather output keeps the legacy float8_e4m3fn dtype for
        API compatibility; reinterpret the bytes as the int8 / int32 layout
        the DeepGEMM kernel expects. The scale tensor is collapsed to 1D for
        the kv side and 2D for the q side per the kernel's asserts.

        clean_logits=False skips DeepGEMM's smxx_clean_logits pass that fills
        everything outside each row's [ks, ke) window with -inf. Safe only
        when the consumer never reads outside that window (the custom
        indexer_topk_prefill kernel); the torch topk fallback scans the full
        padded row and needs the fill. The BF16 kernel always fills.

        bf16_logits (MXFP4 keys on SM100 / SM103): BF16 head weights and BF16 head reduce. With clean_logits=False and
        the CuTe DSL available the logits come back FP32 from _fp4_bf16_reduce_prefill_logits (columns up to each row's
        window end are valid), else as DeepGEMM's BF16 logits. row_starts_zero: the caller knows every
        cu_seqlen_ks is 0 (a faster CuTe DSL instantiation; results are the same).
        """
        if self.use_bf16:
            return bf16_mqa_logits(q_fp8, k_fp8, weights, cu_seqlen_ks, cu_seqlen_ke)
        if self.use_fp4:
            k_fp4_bytes = k_fp8.view(torch.int8)
            if q_scale is None:
                raise ValueError("FP4 indexer queries require packed per-32 scale bytes")
            if self.head_dim != 128 or not q_fp8.is_cuda or get_sm_version() < 100:
                q = self._decode_mxfp4(q_fp8, q_scale)
                k = self._decode_mxfp4(k_fp4_bytes, k_scale)
                if q.is_cuda:
                    from tensorrt_llm._torch.custom_ops import bmm_out

                    dots = torch.empty(
                        (q.shape[0], q.shape[1], k.shape[0]), dtype=q.dtype, device=q.device
                    )
                    bmm_out(q, k.T.unsqueeze(0).expand(q.shape[0], -1, -1), dots)
                else:
                    dots = torch.matmul(q, k.T)
                # Keep decoded dot products and head-weight reductions in FP32;
                # rounding either to BF16 can change the exact Top-K selection.
                logits = (dots.relu() * weights.float().unsqueeze(-1)).sum(1)
                columns = torch.arange(k.shape[0], device=q.device)[None, :]
                return logits.masked_fill(
                    (columns < cu_seqlen_ks[:, None]) | (columns >= cu_seqlen_ke[:, None]),
                    -torch.inf,
                )
            k_scale_int32 = k_scale.contiguous().view(torch.int32).reshape(-1)
            q_scale_2d = q_scale.contiguous().view(torch.int32).reshape(q_fp8.shape[:2])
            q_fp8 = q_fp8.view(torch.int8)
            heads = q_fp8.shape[1]
            if not 1 <= heads <= 64:
                raise ValueError("FP4 MQA requires between 1 and 64 index heads")
            # Consumer Blackwell lacks the 8-head FP4 MQA specialization.
            supported_heads = (16, 32, 64) if get_sm_version() in (120, 121) else (8, 16, 32, 64)
            padded_heads = next(size for size in supported_heads if size >= heads)
            if padded_heads != heads:
                q_fp8 = F.pad(q_fp8, (0, 0, 0, padded_heads - heads))
                q_scale_2d = F.pad(q_scale_2d, (0, padded_heads - heads))
                weights = F.pad(weights, (0, padded_heads - heads))
            if bf16_logits:
                rows_per_seq = _bf16_reduce_prefill_rows(padded_heads)
                if (
                    _INDEXER_BF16_LOGITS_CUTE
                    and not clean_logits
                    and rows_per_seq
                    and IS_CUTLASS_DSL_AVAILABLE
                    and is_sm_100f()
                ):
                    logger.info_once(
                        "DSA indexer prefill logits: BF16 epilogue on the CuTe DSL "
                        f"FP4 MQA logits kernel ({rows_per_seq} rows per group, {padded_heads} heads, FP32 store, "
                        f"TMEM-load prefetch {_BF16_PREFILL_LD_PREFETCH})",
                        key="dsa_indexer_bf16_logits_cute_dsl",
                    )
                    return _fp4_bf16_reduce_prefill_logits(
                        q_fp8,
                        q_scale_2d,
                        k_fp4_bytes,
                        k_scale_int32,
                        weights,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        rows_per_seq,
                        row_starts_zero,
                    )
                logger.info_once(
                    "DSA indexer prefill logits: BF16 epilogue on DeepGEMM "
                    f"fp8_fp4_mqa_logits ({padded_heads} heads, BF16 logits widened to FP32)",
                    key="dsa_indexer_bf16_logits_deepgemm",
                )
                return fp8_fp4_mqa_logits(
                    (q_fp8, q_scale_2d),
                    (k_fp4_bytes, k_scale_int32),
                    weights.to(torch.bfloat16),
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    clean_logits=clean_logits,
                    logits_dtype=torch.bfloat16,
                )
            return fp8_fp4_mqa_logits(
                (q_fp8, q_scale_2d),
                (k_fp4_bytes, k_scale_int32),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                clean_logits=clean_logits,
            )
        return fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scale.reshape(-1)),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
        )

    def _call_paged_mqa_logits(
        self,
        q_decode: torch.Tensor,
        k_cache: torch.Tensor,
        weights_decode: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        scheduler_metadata_buffer: torch.Tensor,
        max_seq_len: int,
        q_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dispatch fp8_paged_mqa_logits, fp8_fp4_paged_mqa_logits or the BF16 kernel."""
        if self.use_bf16:
            return bf16_paged_mqa_logits(
                q_decode, k_cache, weights_decode, context_lens, block_table, max_seq_len
            )
        if self.use_fp4:
            return fp8_fp4_paged_mqa_logits(
                (q_decode, q_scale),
                k_cache,
                weights_decode,
                context_lens,
                block_table,
                scheduler_metadata_buffer,
                int(max_seq_len),
            )
        return fp8_paged_mqa_logits(
            q_decode,
            k_cache,
            weights_decode,
            context_lens,
            block_table,
            scheduler_metadata_buffer,
            int(max_seq_len),
        )

    def _decode_paged_remap(self, metadata) -> Optional[tuple]:
        """``TopK`` ``paged_remap`` of this layer's generation rows: the arguments of the remap
        (transform_local_topk_and_prepare_pool_view) as ``(block_table, block_size, stride_factor, layer_offset)``."""
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None:
            return None
        metadata._ensure_pool_view_cached()
        local_layer = kv_cache_manager.layer_offsets[self.layer_idx]
        page_index_scale, layer_offset = kv_cache_manager.get_primary_pool_page_index_params(
            local_layer
        )
        tokens_per_block = metadata._cached_tokens_per_block
        block_table = metadata._cached_block_table_gen
        if block_table is None or not block_table.is_contiguous():
            return None
        return (block_table, tokens_per_block, page_index_scale * tokens_per_block, layer_offset)

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for one phase or the full batch.

        When ``is_generation`` is ``None``, the inputs contain the full mixed
        batch. Otherwise they contain only the selected context or generation
        phase. ``q_scale`` is only consumed by the FP4 dispatch.

        """
        # Set by the generation Top-K below when it wrote paged-pool indices (Indexer.gvr_paged_remap).
        self.decode_topk_paged = False
        # DSACacheManager / DeepseekV4CacheManager force quant_block_size to
        # 128 (FP8 path) or 32 (MXFP4 path); both round-trip to the same
        # 4-byte scale word per token at index_head_dim=128, so the slot
        # mapping arithmetic is unchanged in either mode.
        assert metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size in (
            32,
            128,
        ), (
            "Unexpected quant_block_size "
            f"{metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        )
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        num_gen_tokens = num_tokens - num_ctx_tokens
        gvr_prior_indices = None
        if self.top_k.needs_gvr_prior:
            assert metadata.gvr_prior_indices is not None
            local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
            gvr_prior_indices = metadata.gvr_prior_indices[local_layer]
        if is_generation is None:
            has_prefill = num_contexts > 0
            has_decode = num_generations > 0
            token_offset = num_ctx_tokens
            cache_name = "indexer_topk_out_buffer"
        else:
            has_prefill = not is_generation and num_contexts > 0
            has_decode = is_generation and num_generations > 0
            token_offset = 0
            expected_tokens = num_gen_tokens if is_generation else num_ctx_tokens
            assert hidden_states.shape[0] == expected_tokens, (
                "Phase-specific DSA prediction received "
                f"{hidden_states.shape[0]} tokens, expected {expected_tokens}."
            )
            cache_name = (
                "indexer_topk_out_buffer_gen" if is_generation else "indexer_topk_out_buffer_ctx"
            )

        topk_indices_buffer = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (hidden_states.shape[0], self.index_topk),
            cache_name=cache_name,
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )

        share_plan = (
            metadata.indexer_share_plan
            if is_generation is not True and not self.use_bf16 and not metadata.in_mtp_draft_loop
            else None
        )
        shared_chunks = (
            frozenset(share_plan.local_chunks) if share_plan is not None else frozenset()
        )
        share_pending = (
            prefill_share.start_shared_indexer(
                self, metadata, q_fp8, weights, q_scale if self.use_fp4 else None
            )
            if share_plan is not None
            else None
        )

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                q_split_threshold = _prefill_q_split_threshold(metadata)
                # Chunks planned for this rank's share of the queries only are
                # never split again; their top-k rows are gathered after the loop.
                batch_q_split = metadata.indexer_prefill_q_split
                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
                # FP4 packs two codes per byte so the gathered row holds half
                # as many bytes as in the FP8 path (BF16: twice as many). The
                # scale (4 bytes) is the same in all modes because FP4 packs
                # four UE8M0 exponents into one int32 to match FP8's float32
                # scale width.
                gather_head_dim = self.k_data_bytes_per_token

                for chunk_idx, chunk in enumerate(metadata.indexer_prefill_chunks):
                    if chunk_idx in shared_chunks:
                        continue  # computed across the ranks by run_shared_indexer
                    # Skip chunks with no compressed KV tokens (e.g., warmup
                    # sequences shorter than compress_ratio produce zero KV).
                    if chunk.k_token_start >= chunk.k_token_end:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :].fill_(-1)
                        continue
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    gathered = gather_k_cache_paged(
                        k_cache_4d,
                        metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        num_k_tokens,
                        gather_head_dim,
                    )
                    if gathered is None:
                        gathered = torch.ops.trtllm.indexer_k_cache_gather_op(
                            k_cache_4d,
                            metadata.slot_mapping_fp8_fullkv,
                            metadata.slot_mapping_scale_fullkv,
                            chunk.k_token_start,
                            num_k_tokens,
                            gather_head_dim,
                        )
                    chunk_k_fp8, chunk_k_scale = gathered
                    if self.use_bf16:
                        chunk_k_fp8 = chunk_k_fp8.view(torch.bfloat16)

                    chunk_num_token = chunk.token_end - chunk.token_start
                    if batch_q_split is None:
                        chunk_q_start, chunk_q_end, q_sizes = _split_prefill_queries(
                            chunk_num_token, metadata.mapping, q_split_threshold
                        )
                    else:
                        chunk_q_start, chunk_q_end, q_sizes = 0, chunk_num_token, None

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    # Tile the query dimension so each fp8_mqa_logits call
                    # allocates at most [q_tile x num_k_tokens] instead of the
                    # full [local_q x num_k_tokens] (which can reach tens of GB
                    # on a long context and stall cuMemCreate under
                    # expandable_segments -> engine hang; see
                    # _INDEXER_MQA_LOGITS_ELEM_BUDGET). Results are identical:
                    # each query row's logits/top-k are independent and the KV
                    # (chunk_k_fp8) is unchanged across tiles, so the per-call
                    # allocation is the same size and the caching allocator
                    # reuses one block (peak ~= one tile, no extra sync).
                    local_q_len = chunk_q_end - chunk_q_start
                    q_tile = self.prefill_q_tile(local_q_len, num_k_tokens, q_fp8.is_cuda)
                    pieces = [
                        (c0, min(c0 + q_tile, chunk_q_end), None)
                        for c0 in range(chunk_q_start, chunk_q_end, q_tile)
                    ]
                    for c0, c1, tier_rows in pieces:
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        tile_q_scale = q_scale[g0:g1, ...] if self.use_fp4 else None
                        self.forward_prepared(
                            q_fp8[g0:g1, ...],
                            chunk_k_fp8,
                            chunk_k_scale,
                            weights[g0:g1, ...],
                            chunk.cu_seqlen_ks[c0:c1],
                            chunk.cu_seqlen_ke[c0:c1],
                            topk_indices_buffer[g0:g1, :],
                            tile_q_scale,
                            tier_rows=tier_rows,
                            row_starts_zero=chunk.row_starts_zero,
                        )

                    if q_sizes is not None:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :] = allgather(
                            topk_indices_buffer[global_q_start:global_q_end, :],
                            metadata.mapping,
                            dim=0,
                            sizes=q_sizes,
                        )

                # A token-sharded sparse MLA prefill (ShardQ) attends only this rank's rows; the other rows of
                # the buffer stay unset.
                if batch_q_split is not None and not getattr(
                    metadata, "indexer_skip_q_split_gather", False
                ):
                    local_q_begin, local_q_end, q_sizes = batch_q_split
                    topk_indices_buffer[: sum(q_sizes), :] = allgather(
                        topk_indices_buffer[local_q_begin:local_q_end, :],
                        metadata.mapping,
                        dim=0,
                        sizes=q_sizes,
                    )
            elif metadata.num_ctx_kv_tokens == 0:
                # No compressed KV tokens — fill with -1 (no valid indices)
                topk_indices_buffer[:num_ctx_tokens, :].fill_(-1)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                num_ctx_kv_tokens = metadata.num_ctx_kv_tokens
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens, ...] if self.use_fp4 else None
                self.forward_prepared(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_kv_tokens, ...],
                    k_scale[:num_ctx_kv_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices_buffer[:num_ctx_tokens, :],
                    ctx_q_scale,
                )
        if share_pending is not None:
            prefill_share.finish_shared_indexer(self, metadata, share_pending, topk_indices_buffer)

        if has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = metadata.topk_indices_buffer[
                :num_ctx_tokens, :
            ]

        # Dense skip outputs remain valid priors if a later step enters GVR.
        if has_prefill:
            # Device lengths: gathering the CUDA selections with the host
            # seq_lens would force a synchronous H2D copy every layer.
            self.top_k.update_gvr_prior_from_prefill(
                topk_indices_buffer[:num_ctx_tokens],
                metadata.seq_lens_cuda[:num_contexts],
                gvr_prior_indices,
                request_offset=num_generations,
            )
            if self.use_gvr_emission:
                # reused slots cold-start the emission closed loop; stale
                # lines only mis-place cuts - counts are re-measured
                # in-kernel, so exactness never rides on this reset
                self.top_k.reset_gvr_emission_rows(
                    slice(num_generations, num_generations + num_contexts)
                )

        reuse_topk = (
            self.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and metadata.shared_topk_indices is not None
        )

        use_key_shard = (
            has_decode
            and not metadata.skip_indexer_for_gen_reqs
            and not reuse_topk
            and self._key_shard_ready(metadata, num_gen_tokens, num_generations)
        )
        if has_decode and not metadata.skip_indexer_for_gen_reqs and reuse_topk:
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.shared_topk_indices[:num_generations, :]
            )
        elif use_key_shard:
            self._key_shard_decode(
                metadata,
                q_fp8[token_offset : token_offset + num_gen_tokens],
                weights[token_offset : token_offset + num_gen_tokens],
                topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                num_generations,
                q_scale[token_offset : token_offset + num_gen_tokens] if self.use_fp4 else None,
            )
        elif has_decode and not metadata.skip_indexer_for_gen_reqs:
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
            if not metadata.is_ragged_verify:
                max_decode_len = gen_seq_lens.max().item()
                min_decode_len = gen_seq_lens.min().item()
                assert max_decode_len == min_decode_len, (
                    "max_decode_len != min_decode_len, we need padding"
                )

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[token_offset : token_offset + num_gen_tokens, ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if metadata.is_ragged_verify:
                take_strided_path = False
            else:
                take_strided_path = not metadata.use_expanded_buffers_for_mtp or next_n == 1
            if take_strided_path:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts : num_contexts + num_generations
                ]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n. The DSL path
                # uses its own schedule buffer (built with num_next_n_atoms=1
                # via a (num_gen, 1) input shape) and overrides this below.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            if metadata.is_ragged_verify:
                assert num_gen_tokens == sum(metadata.ragged_verify_lens), (
                    f"ragged gen tokens {num_gen_tokens} != "
                    f"sum(verify_lens) {sum(metadata.ragged_verify_lens)}"
                )
            else:
                assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[token_offset : token_offset + num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits / fp8_fp4_paged_mqa_logits
            # with prepared decode metadata.
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            indexer_max_seq_len = metadata.get_indexer_max_seq_len()

            # Helix verify rows carry per-token bounds the DSL schedule cannot take.
            if (
                self.use_cute_dsl_paged_mqa_logits
                and not metadata.is_ragged_verify
                and metadata.helix_spec_row_kv_lens(num_gen_tokens) is None
            ):
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `Indexer.prepare()`
                # with a (num_gen, 1) input shape, which makes DeepGEMM's wrapper
                # compute `num_next_n_atoms = 1`. (DeepGEMM uses the same buffer
                # for its own next_n=1 kernel; DSL piggy-backs on it for all
                # real next_n values.) All next_n positions of a batch share
                # the same KV length on this path (kv_lens_cuda_2d broadcasts),
                # so passing the 1D contiguous kv_lens slice for context_lens
                # avoids materializing a 2D contiguous tensor per call.
                dsl_context_lens = metadata.gen_indexer_kv_lens_cuda_runtime
                assert dsl_context_lens is not None
                # Wave-aware atom-split: the picker in `_pick_dsl_expand` caches
                # (factor, atom) on metadata with invariant
                # `factor * atom == 1 + max_draft_tokens` (the target/verify-time
                # next_n). MTPEagle reuses the same metadata for its multi-step
                # draft loop; after i=0 it mutates seq_lens to 1, so i≥1
                # iterations run with next_n=1. The reshape
                # `(num_gen, next_n, ...) -> (num_gen*factor, atom, ...)` is only
                # valid when the caller actually supplies next_n == factor * atom
                # tokens; gate here so i≥1 draft calls fall back to the
                # kernel-native next_n=1 path.
                dsl_atom_split = (
                    metadata.dsl_expand_factor > 1
                    and next_n == metadata.dsl_expand_factor * metadata.dsl_atom
                )
                # In-kernel uneven split (next_n with no exact atom divisor):
                # native Q / context_lens / block_table, schedule over the
                # staggered per-atom rows. Same full-window gate as above.
                dsl_kernel_split = (
                    metadata.dsl_kernel_atoms > 1 and next_n == metadata.dsl_kernel_next_n
                )
                if self._dsl_single_atom(next_n):
                    dsl_atom_split = dsl_kernel_split = False
                dsl_split_kwargs: dict = {}
                if dsl_kernel_split:
                    dsl_split_kwargs = dict(
                        num_next_n_atoms=metadata.dsl_kernel_atoms, atom=metadata.dsl_atom
                    )
                gvr_emit_kwargs: dict = {}
                # emitting for a step the Top-K cannot consume only churns
                # the closed-loop state, so gate on the consumable shape
                if (
                    self.use_gvr_emission
                    and gvr_prior_indices is not None
                    and next_n == 1
                    and not dsl_atom_split
                    and not dsl_kernel_split
                    and num_gen_tokens <= 256
                    # ext tiers are single-CTA/sort-path only; row reordering
                    # routes the Top-K through order_row, which excludes them
                    and metadata.kv_lens_row_reorder is None
                ):
                    gvr_emit_kwargs = self.top_k.prepare_gvr_emission(
                        num_generations,
                        indexer_max_seq_len,
                        torch.cuda.get_device_properties(q_decode.device).multi_processor_count,
                        gvr_prior_indices,
                    )
                if self.use_fp4:
                    # FP4 DSL signature splits DG's (q, sf_q) tuple into two
                    # separate args and requires q.dtype == uint8 (q_decode
                    # came in via the FP8 plumbing as int8; reinterpret with
                    # no copy). sf_q is the q_scale slice reshaped to
                    # (B, next_n, H) int32 -- mirrors the non-DSL FP4 branch.
                    decode_q_scale = q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                    dsl_q = q_decode.view(torch.uint8)
                    dsl_block_table = block_table
                    dsl_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        eff_next_n = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = dsl_q.reshape(exp_B, eff_next_n, self.n_heads, self.head_dim // 2)
                        decode_q_scale = decode_q_scale.reshape(exp_B, eff_next_n, self.n_heads)
                        dsl_context_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        dsl_block_table = metadata.block_table_expanded[:exp_B]
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    elif dsl_kernel_split:
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    if _DSL_LOGITS_EVICT_FIRST and self._dsl_single_atom(next_n):
                        dsl_split_kwargs["kv_evict_first"] = True
                    dsl_split_kwargs.update(
                        _decode_logits_bf16_kwargs(
                            self.bf16_decode_logits,
                            num_gen_tokens,
                            self._dsl_single_atom(next_n),
                            bool(gvr_emit_kwargs),
                            # the Top-K is the only reader (Helix merges the rows' scores too)
                            bf16_reader=self.top_k.decode_reads_bf16 and not metadata.enable_helix,
                        )
                    )

                    logits_decode = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q,
                        decode_q_scale,
                        k_cache,
                        weights_decode,
                        dsl_context_lens,
                        dsl_block_table,
                        dsl_schedule_meta,
                        indexer_max_seq_len,
                        **gvr_emit_kwargs,
                        **dsl_split_kwargs,
                    )
                else:
                    # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                    # Atom-split benefits small-batch / low-ntask configs by
                    # raising SM utilization at the cost of factorx KV HBM
                    # re-reads. Context lengths are indexer KV lengths, which
                    # may be compressed for DeepSeek-V4.
                    dsl_q = q_decode
                    fp8_ctx_lens = dsl_context_lens
                    fp8_block_table = block_table
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        atom = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = q_decode.reshape(exp_B, atom, self.n_heads, self.head_dim)
                        fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        fp8_block_table = metadata.block_table_expanded[:exp_B]
                        fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    elif dsl_kernel_split:
                        fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    if _DSL_LOGITS_EVICT_FIRST and self._dsl_single_atom(next_n):
                        dsl_split_kwargs["kv_evict_first"] = True
                    logits_decode = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                        dsl_q,
                        k_cache,
                        weights_decode,
                        fp8_ctx_lens,
                        fp8_block_table,
                        fp8_schedule_meta,
                        indexer_max_seq_len,
                        **gvr_emit_kwargs,
                        **dsl_split_kwargs,
                    )
            else:
                decode_q_scale = (
                    q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    if self.use_fp4
                    else None
                )
                if self.use_fp4:
                    # q_decode shape is either (num_generations, next_n, n_heads,
                    # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                    # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                logits_decode = self._call_paged_mqa_logits(
                    q_decode,
                    k_cache,
                    weights_decode,
                    context_lens,
                    block_table,
                    scheduler_metadata_buffer,
                    indexer_max_seq_len,
                    decode_q_scale,
                )

            # Native/GVR use logical lengths; radix uses score-column lengths.
            gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                num_contexts : num_contexts + num_generations
            ]
            scan_lengths = metadata.gen_indexer_kv_lens_cuda_runtime
            assert scan_lengths is not None

            gvr_ext_kwargs = (
                {
                    "gvr_prior_indices": gvr_prior_indices[:num_generations],
                    "gvr_row_order": metadata.kv_lens_row_reorder,
                }
                if gvr_prior_indices is not None
                else None
            )
            assert metadata.radix_aux_indices is not None
            assert metadata.radix_aux_logits is not None
            # Helix verify tokens carry per-token rank-local bounds, like ragged rows.
            helix_row_kv_lens = metadata.helix_spec_row_kv_lens(num_gen_tokens)
            row_kv_lens = (
                helix_row_kv_lens
                if helix_row_kv_lens is not None
                else metadata.ragged_row_kv_lens(num_gen_tokens)
            )
            decode_rows = topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :]
            paged_remap = (
                self._decode_paged_remap(metadata)
                if (
                    self.gvr_paged_remap
                    and not has_prefill
                    and row_kv_lens is None
                    and gvr_ext_kwargs is None
                    and not metadata.is_ragged_verify
                    and not metadata.enable_helix
                    and not metadata.in_mtp_draft_loop
                    and metadata.shared_topk_indices is None
                    and self.compress_ratio == 1
                )
                else None
            )
            self.top_k(
                logits_decode,
                decode_rows,
                is_prefill=False,
                sequence_lengths=gen_kv_lens_cuda,
                scan_lengths=scan_lengths,
                row_kv_lens=row_kv_lens,
                next_n=1 if metadata.is_ragged_verify or helix_row_kv_lens is not None else next_n,
                max_seq_len=indexer_max_seq_len,
                gvr_ext_kwargs=gvr_ext_kwargs,
                radix_aux_indices=metadata.radix_aux_indices,
                radix_aux_logits=metadata.radix_aux_logits,
                paged_remap=paged_remap,
            )
            self.decode_topk_paged = paged_remap is not None and self.top_k.last_decode_paged
            if metadata.enable_helix and metadata.mapping.cp_size > 1:
                if helix_row_kv_lens is not None:
                    # Tokens that see no KV on this rank were scored over a
                    # placeholder slot; they must not offer candidates.
                    decode_rows.masked_fill_((helix_row_kv_lens == 0).unsqueeze(1), -1)
                mapping = metadata.mapping
                merge_helix_topk(
                    logits_decode,
                    decode_rows,
                    mapping.cp_rank,
                    lambda t: cp_allgather(t.unsqueeze(0), mapping, dim=0),
                )
                metadata.set_helix_sparse_row_lens(decode_rows)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
            )

        # Keep the GVR prior current for computed, reused, and dense-skip TopK.
        if gvr_prior_indices is not None and has_decode:
            decode_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
            if metadata.is_ragged_verify:
                last_rows = (
                    torch.cumsum(metadata.gen_token_repeats_cuda[:num_generations], dim=0).to(
                        torch.long
                    )
                    - 1
                )
                last_mtp_topk = decode_topk[last_rows]
            else:
                next_n = num_gen_tokens // num_generations
                last_mtp_topk = decode_topk[next_n - 1 :: next_n]
            prev_topk_dst = gvr_prior_indices[:num_generations]
            if do_multi_stream() and self.aux_stream is not None:
                # Overlap the GVR prior write-back with this layer's sparse attention.
                self.prev_topk_copy_events[0].record()
                with torch.cuda.stream(self.aux_stream):
                    self.prev_topk_copy_events[0].wait()
                    prev_topk_dst.copy_(last_mtp_topk)
                    self.prev_topk_copy_events[1].record()
                self._prev_topk_copy_pending = True
            else:
                prev_topk_dst.copy_(last_mtp_topk)

        if self.mtp_index_share and metadata.in_mtp_draft_loop and not reuse_topk:
            rows = None
            if has_decode:
                next_n = num_gen_tokens // num_generations
                gen_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
                rows = self._mtp_last_accepted_rows(
                    gen_topk, metadata, num_contexts, num_generations, next_n
                )
            if has_prefill:
                ctx_last = (
                    torch.cumsum(metadata.seq_lens_cuda[:num_contexts].to(torch.long), dim=0) - 1
                )
                ctx_rows = topk_indices_buffer[ctx_last]
                rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
            if rows is not None:
                shared_topk_indices = metadata.shared_topk_indices
                if shared_topk_indices is None:
                    metadata.shared_topk_indices = rows.contiguous()
                else:
                    row_start = num_contexts if is_generation else 0
                    shared_topk_indices[
                        row_start : row_start + rows.shape[0], : rows.shape[1]
                    ].copy_(rows)
        return topk_indices_buffer

    def _dsl_single_atom(self, next_n: int) -> bool:
        """Run the DSL logits of a verify window as one in-kernel atom (see _DSL_SINGLE_ATOM)."""
        if not (_DSL_SINGLE_ATOM and next_n > 1 and self.n_heads <= 32):
            return False
        max_n = _fp4_max_atom_n() if self.use_fp4 else _DSL_MAX_ATOM_N
        return next_n * self.n_heads <= max_n

    def _key_shard_ready(
        self, metadata: DSAtrtllmAttentionMetadata, num_gen_tokens: int, num_generations: int
    ) -> bool:
        """Whether this decode step takes the key-sharded indexer (see dsa/key_shard.py)."""
        next_n = metadata.key_shard_next_n
        return (
            metadata.key_shard_world > 1
            and next_n > 0
            and num_gen_tokens == num_generations * next_n
            and self.use_cute_dsl_paged_mqa_logits
            and not self.use_bf16
            and self.compress_ratio == 1
            and self._use_self_sampling_topk
            and next_n * self.n_heads <= (_fp4_max_atom_n() if self.use_fp4 else _DSL_MAX_ATOM_N)
        )

    def _key_shard_decode(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        q_decode: torch.Tensor,
        weights_decode: torch.Tensor,
        decode_rows: torch.Tensor,
        num_generations: int,
        q_scale_decode: Optional[torch.Tensor] = None,
    ) -> None:
        """Score this rank's key blocks, take the local Top-K and merge the exact global Top-K into ``decode_rows``.

        Every TP rank holds the whole index-K cache; rank ``r`` scores the 64-token blocks ``b % tp == r`` through a
        compacted block table, so the logits kernel reads 1/tp of the keys. Local rows are bounded by their causal
        length over the owned blocks.
        """
        next_n = metadata.key_shard_next_n
        rows = decode_rows.shape[0]
        local_len = metadata.key_shard_max_local_len
        merge = metadata.key_shard_merge
        helix = metadata.enable_helix
        if helix:
            # Helix: this rank's pages are its shard; score them with the Helix decode inputs.
            block_table, ctx_lens, sched, row_lens = metadata.helix_key_shard_inputs(rows)
        else:
            block_table = metadata.key_shard_block_table[:num_generations]
            ctx_lens = metadata.key_shard_ctx_lens
            sched = metadata.key_shard_sched
            row_lens = metadata.key_shard_row_lens[:rows]
        ctx_lens = ctx_lens[:num_generations]
        # The local logits land in the merge workspace, where peers read them if a row needs the exact fallback.
        logits = merge.local_logits[:rows]
        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        if self.use_fp4:
            torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits_into(
                q_decode.view(torch.uint8).view(
                    num_generations, next_n, self.n_heads, self.head_dim // 2
                ),
                q_scale_decode.view(num_generations, next_n, self.n_heads),
                k_cache,
                weights_decode,
                ctx_lens,
                block_table,
                sched,
                local_len,
                logits,
                kv_evict_first=_DSL_LOGITS_EVICT_FIRST,
            )
        else:
            torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits_into(
                q_decode.view(num_generations, next_n, self.n_heads, self.head_dim),
                k_cache,
                weights_decode,
                ctx_lens,
                block_table,
                sched,
                local_len,
                logits,
                kv_evict_first=_DSL_LOGITS_EVICT_FIRST,
            )
        local_rows = metadata.key_shard_local_topk[:rows]
        # Exact local Top-L by score over each row's owned keys. The CuTe radix is fast on these short rows (context
        # / tp) and insensitive to ties; the merge certificate compares scores only, so its tie order is irrelevant.
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            logits[:, :local_len], row_lens, local_rows, merge.local_k, 1
        )
        merge.pack(local_rows, row_lens)
        if not helix:
            merge.merge(decode_rows)
            return
        # The sparse MLA of a Helix rank reads only its own winners, as local token indices; the merge writes them
        # (kept first, then -1) and their counts, the rows' lengths for the token-sparse MLA.
        counts = metadata.helix_sparse_row_lens_out(rows)
        merge.merge_owned(decode_rows, counts)
        if helix_key_shard_verify_enabled() and not torch.cuda.is_current_stream_capturing():
            full = None
            if helix_key_shard_verify_full():
                full = (q_decode, weights_decode, q_scale_decode, k_cache)
            self._verify_helix_key_shard(
                metadata, logits[:, :local_len], row_lens, decode_rows, full
            )
        if counts is None:
            metadata.set_helix_sparse_row_lens(decode_rows)
        else:
            metadata.mark_helix_sparse_row_lens_set()

    def _verify_helix_key_shard(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        local_logits: torch.Tensor,
        row_lens: torch.Tensor,
        merged_rows: torch.Tensor,
        full: Optional[tuple] = None,
    ) -> None:
        """Debug: recompute this layer's Helix merge with ``merge_helix_topk`` and log the differences.

        ``full`` = ``(q, weights, q_scale, k_cache)``: the reference scores come from the default Helix path's
        logits (DeepGEMM paged MQA over the per-token expanded rows, bounded by the Helix bounds) instead of
        this path's local logits; the log adds the largest score difference over the rows' valid columns.
        """
        mapping = metadata.mapping
        rows = merged_rows.shape[0]
        ref_logits = local_logits
        score_diff = 0.0
        if full is not None and metadata.use_expanded_buffers_for_mtp:
            q, weights, q_scale, k_cache = full
            ref_logits = self._call_paged_mqa_logits(
                q.view(rows, 1, *q.shape[1:]),
                k_cache,
                weights,
                metadata.kv_lens_expanded_cuda[:rows].view(-1, 1),
                metadata.block_table_expanded[:rows],
                metadata.scheduler_metadata_buffer_expanded,
                metadata.get_indexer_max_seq_len(),
                None if q_scale is None else q_scale.view(rows, 1, self.n_heads),
            )
            width = min(ref_logits.shape[1], local_logits.shape[1])
            ref_logits = ref_logits[:, :width].float()
            cols = torch.arange(width, device=ref_logits.device)
            valid = cols[None, :] < row_lens[:, None].long()
            if bool(valid.any()):
                score_diff = float(
                    (ref_logits - local_logits[:, :width].float())
                    .abs()
                    .masked_fill(~valid, 0)
                    .max()
                )
        reference = reference_local_topk(ref_logits, row_lens, merged_rows.shape[1])
        merge_helix_topk(
            ref_logits,
            reference,
            mapping.cp_rank,
            lambda t: cp_allgather(t.unsqueeze(0), mapping, dim=0),
        )
        bad_rows, bad_entries, kept = count_set_mismatches(merged_rows, reference)
        stats = torch.tensor(
            [bad_rows, bad_entries, kept, rows],
            dtype=torch.int64,
            device=merged_rows.device,
        )
        totals = cp_allgather(stats.unsqueeze(0), mapping, dim=0).sum(dim=0).tolist()
        diff_all = cp_allgather(
            torch.tensor([score_diff], dtype=torch.float32, device=merged_rows.device).unsqueeze(0),
            mapping,
            dim=0,
        )
        if mapping.cp_rank == 0:
            logger.info(
                f"DSA helix key-shard verify{' full' if full is not None else ''} layer {self.layer_idx}: "
                f"{totals[0]} of {totals[3]} rows differ (symmetric difference {totals[1]} of {totals[2]} kept "
                f"entries) vs merge_helix_topk; max |score diff| {float(diff_all.max()):.4g}"
            )

    def _mtp_last_accepted_rows(
        self,
        gen_topk: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        num_contexts: int,
        num_generations: int,
        next_n: int,
    ) -> torch.Tensor:
        """Select each generation request's last accepted TopK row."""
        num_accepted = metadata.mtp_num_accepted
        if num_accepted is None:
            return gen_topk[next_n - 1 :: next_n]
        gen_num_accepted = num_accepted[num_contexts : num_contexts + num_generations]
        base = torch.arange(num_generations, device=gen_topk.device, dtype=torch.long) * next_n
        offset = (gen_num_accepted - 1).clamp(0, next_n - 1)
        return gen_topk[base + offset]

    def maybe_join_prev_topk_copy(self) -> None:
        """Join the aux-stream GVR prior write-back, if forked."""
        if self._prev_topk_copy_pending:
            self.prev_topk_copy_events[1].wait()
            self._prev_topk_copy_pending = False

    def _weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(
        self,
        qr: torch.Tensor,
        indexer_k: torch.Tensor,
        position_ids: torch.Tensor,
        q_proj_out: Optional[torch.Tensor] = None,
    ):
        """Project Q/K and apply RoPE. ``q_proj_out`` is ``wq_b(qr)`` when the caller already computed it.

        A caller's ``q_proj_out`` is usually a column slice of a packed GEMM output; the FP8 cat-quantize
        kernel steps the [tokens, heads] rows of ``q`` with one stride, so it needs the heads of
        consecutive tokens to be adjacent.
        """
        q = self.wq_b(qr) if q_proj_out is None else q_proj_out.contiguous()
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        if self.rotary_emb is None:
            # Position-free indexer: nothing rotates. The fused cat+quantize
            # kernels take two non-empty halves of each head.
            half = self.head_dim // 2
            q_pe, q_nope = q.split([half, self.head_dim - half], dim=-1)
            k_pe, k_nope = k.split([half, self.head_dim - half], dim=-1)
            return q_pe, q_nope, k_pe, k_nope
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        k_pe = k_pe[:, 0, :]
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate and quantize for Q or K.

        FP8 mode: fused cat + FP8 quantize via CUDA kernel.
        FP4 mode: fused cat + per-block-32 FP4 E2M1 quantize via CUDA kernel.
        The returned packed bytes are int8 (two FP4 codes per byte) and the
        scale is int32 (four UE8M0 exponents packed little-endian).
        BF16 mode: plain concatenation with a unit float32 scale per row.
        """
        if self.use_fp4:
            return torch.ops.trtllm.fused_cat_fp4(qk_pe, qk_nope)
        if self.use_bf16:
            out = torch.cat([qk_pe, qk_nope], dim=-1)
            return out, out.new_ones((*out.shape[:-1], 1), dtype=torch.float32)
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        projected: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8/FP4 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in dsa.module.forward_dsa_proj sees a stable signature.
        """
        if self.projection_free:
            raise RuntimeError("Projection-free Indexer requires forward_prepared()")
        if projected is not None:
            # (indexer_k, weights, wq_b(qr)) from the caller's packed projection GEMMs.
            indexer_k, weights, q_proj_out = projected
            return self._finish_pre_indexer_proj(qr, indexer_k, weights, position_ids, q_proj_out)
        indexer_k, weights = self._project_k_and_weights(hidden_states)
        return self._finish_pre_indexer_proj(qr, indexer_k, weights, position_ids)

    def pre_indexer_k_proj(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        indexer_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The K half of pre_indexer_proj, for a batch that needs no TopK.

        Short-MHA context attention selects no keys, yet later steps read the
        batch's keys from the indexer K cache. Returns (k_fp_bytes, k_scale),
        bitwise the K outputs of pre_indexer_proj, without the wq_b projection,
        the Q quantization and the weight scaling. ``indexer_k`` is wk's output
        when the caller's packed projection GEMM already computed it.
        """
        if self.projection_free:
            raise RuntimeError("Projection-free Indexer requires forward_prepared()")
        if indexer_k is None:
            # The fused GEMM also yields the unused weights_proj columns; a wk-only
            # GEMM could round K differently from the full projection.
            indexer_k, _ = self._project_k_and_weights(hidden_states)
        k = self.k_norm(indexer_k)
        if self.rotary_emb is None:
            half = self.head_dim // 2
            k_pe, k_nope = k.split([half, self.head_dim - half], dim=-1)
        else:
            k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            # RotaryEmbedding rotates on FlashInfer only for a (q, k) pair: pair K with
            # a scratch query so it is rotated exactly as in _qk_projection_and_rope.
            _, k_pe = self.rotary_emb(position_ids, [torch.empty_like(k_pe), k_pe])
        return self._prep_q_or_k(k_pe, k_nope)

    def _project_k_and_weights(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused wk / weights_proj projection; returns (indexer_k, weights)."""
        assert self._fused_wk_wp_weight is not None, (
            "cache_derived_state() must be called before forward()"
        )
        # When the boundary fusion pre-quantized the next layer's kv_a_proj
        # NVFP4 input, hidden_states is an Fp4QuantizedTensor that also carries
        # the BF16 post-RMSNorm value. Use that BF16 view here -- the indexer
        # weight is BF16 and the matmul needs a float input.
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "pre_indexer_proj received Fp4QuantizedTensor without bf16 view; "
                "the producer fusion must request return_norm_out=True"
            )
            hidden_states_bf = hidden_states.unquantized_hidden_states
        else:
            hidden_states_bf = hidden_states
        if self._indexer_bf16:
            # hidden_states_bf and _fused_wk_wp_weight are both the model dtype,
            # so this runs a single native GEMM with no fp32 cast around it.
            fused_out = F.linear(hidden_states_bf, self._fused_wk_wp_weight)
        else:
            with _tf32_matmul_enabled():
                # F.linear computes input @ weight.T internally; no explicit .t() needed.
                # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
                # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
                # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
                # which uses its own handle and always falls back to CUDA-core SGEMM.
                fused_out = F.linear(_to_float(hidden_states_bf), self._fused_wk_wp_weight)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads], dim=-1)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE,
        # FP8 quantize); a no-op when the projection already ran in the model dtype.
        indexer_k = indexer_k.to(hidden_states_bf.dtype)
        return indexer_k, weights

    def _finish_pre_indexer_proj(
        self,
        qr: torch.Tensor,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        q_proj_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K norm, quantization and weight scaling after the wk / weights_proj / wq_b projections."""
        q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(
            qr, indexer_k, position_ids, q_proj_out
        )
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        if self.use_fp4:
            # FP4 packs two codes per byte, so the trailing dim is head_dim // 2.
            # fused_cat_fp4 flattens the leading dims to M=N*n_heads; restore
            # the (N, n_heads, ...) shape so downstream slicing in
            # sparse_attn_indexer (which indexes by token) lines up with
            # q_fp8. The DeepGEMM FP4 kernel applies the per-block q_scale
            # internally, so weights carry only softmax_scale * n_heads^-0.5.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            # DeepGEMM's fp8_fp4_(paged_)mqa_logits asserts
            # `weights.scalar_type() == kFloat`. Unlike the FP8 branch (whose
            # `_weight_scale` multiplies by the fp32 `q_scale` tensor and so
            # implicitly upcasts), this branch's only multiplier is a Python
            # float — `bf16 * float` stays bf16 and trips the kernel assert.
            # Cast explicitly so this path doesn't silently rely on an
            # upstream `_to_float`.
            weights = weights.float() * self.weight_scale_factor
        elif self.use_bf16:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = weights.float() * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights, q_scale

    def forward_from_projected(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run sparse prediction from graph-captured indexer projections.

        Projected inputs cover the full batch. Phase-specific prediction slices
        Q-side inputs here while retaining the full-batch K inputs needed by
        context prediction.
        """
        if is_generation is None:
            phase_start = 0
            phase_end = metadata.num_tokens
        elif is_generation:
            phase_start = metadata.num_ctx_tokens
            phase_end = metadata.num_tokens
        else:
            phase_start = 0
            phase_end = metadata.num_ctx_tokens

        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        q_fp8_phase = q_fp8[phase_start:phase_end]
        weights_phase = weights[phase_start:phase_end]
        q_scale_phase = q_scale[phase_start:phase_end] if q_scale is not None else None

        return self.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8_phase,
            k_fp8,
            k_scale,
            weights_phase,
            q_scale=q_scale_phase,
            is_generation=is_generation,
        )

    @torch.inference_mode()
    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ):
        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids
        )
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        self._update_k_cache(k_fp8, k_scale, metadata)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.forward_from_projected(metadata, hidden_states, indexer_intermediates)
