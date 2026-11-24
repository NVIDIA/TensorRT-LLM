# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 ULL target: the model, straight-line.

Sections, each opened by a ``# ---- <name> ----`` comment: glue, comm, kda, dsa_inproj, dsa_qproj,
mla_query, dsa_front, dsa_index, dsa_topk, dsa_core, dsa_attention, dsa, nv_moe, moe, chain, dense0,
model, spec, DFlash2 and entry.

The DFlash2 drafter (MLA drafter at max_draft_len 7, the engine ``DFlashWorker`` specialized for
the Helix CP16 layout) follows the family's helix8 target.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import json
import math
import os
import socket
import struct
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from collections.abc import Mapping as AbcMapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import timedelta
from enum import Enum
from functools import cache, lru_cache
from functools import cache as _cache
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Protocol, Tuple
from typing import Sequence as TypingSequence

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import safetensors
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import dsl_user_op
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.input_commit import active_input_commit
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.pyexecutor.ull_shell import StepShell
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, VocabParallelSelection
from tensorrt_llm._torch.speculative.dflash import DFlashWorker as _EngineDFlashWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    SpecWorkerBase,
)
from tensorrt_llm._torch.speculative.spec_head_kernels import embedding_gather
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    DraftBlockPlacement,
    dflash_paged_latent_prepare,
    store_context_latents,
)
from tensorrt_llm._torch.speculative.vocab_parallel import (
    _LOCAL_BLOCK_V,
    VocabParallelHead,
    VocabShardedLogits,
    _local_argmax_kernel,
    tp_pair_exchange,
)
from tensorrt_llm._utils import maybe_pin_memory, mpi_rank, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import (
    AllReduceFusionOp,
    PositionEmbeddingType,
    RopeEmbeddingUtils,
    RotaryScalingType,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo, QuantMode

from ..kernels.attention.rubin_ull_dsa_index import (
    MERGED,
    PACKED,
    PAGE_TOKENS,
    KvRows,
    indexer_prologue,
    key_shard_local_select,
    log_indexer_paths,
    select_pack_for,
)
from ..kernels.attention.rubin_ull_dsa_index import ROWS as DSA_INDEX_ROWS
from ..kernels.attention.rubin_ull_dsa_qproj import NormSplitKGemm, _splitk
from ..kernels.attention.rubin_ull_mla_output_gate import sigmoid_gate_mul
from ..kernels.embedding.rubin_ull_flat_begin import folds_embedding
from ..kernels.ssm.rubin_ull_kda_kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..kernels.vendor.vllm_mr8 import install, module
from ..ops.activation.rubin_ull_situ_and_mul import situ_and_mul
from ..ops.attention.input_staging import flush_input_commit, stage_or_copy, stage_scatter
from ..ops.attention.rubin_ull_dsa_dcp_value import dsa_dcp_value
from ..ops.attention.rubin_ull_dsa_index_front import dsa_index_front
from ..ops.attention.rubin_ull_dsa_indexer_prologue import dsa_indexer_prologue
from ..ops.attention.rubin_ull_dsa_mla_query import dsa_mla_query
from ..ops.attention.rubin_ull_dsa_output_gate import dsa_output_gate
from ..ops.attention.rubin_ull_dsa_query_wait import dsa_query_wait
from ..ops.attention.rubin_ull_dsa_sparse_dcp_decode import dsa_sparse_dcp_decode
from ..ops.comm.allgather_module import allgather_module
from ..ops.comm.allreduce_module import allreduce_module
from ..ops.comm.rubin_comm import rubin_all_reduce
from ..ops.comm.rubin_lamport_push import rubin_lamport_finish
from ..ops.comm.rubin_push_allreduce import rubin_lamport_push_workspace
from ..ops.comm.rubin_ull_helix_alltoall_native import helix_alltoall_native
from ..ops.comm.rubin_ull_helix_post_process_native import helix_post_process_native
from ..ops.comm.rubin_ull_mnnvl_fusion_allreduce import mnnvl_fusion_allreduce
from ..ops.common.rubin_ull_features import active as k31_features
from ..ops.embedding.rubin_ull_begin_fold import begin_fold
from ..ops.embedding.rubin_ull_step_begin import prep_and_metadata, step_begin
from ..ops.gemm.rubin_linear import rubin_linear
from ..ops.gemm.rubin_ull_bf16_gemm_push import bf16_gemm_push
from ..ops.gemm.rubin_ull_bf16_linear import bf16_linear
from ..ops.gemm.rubin_ull_bf16_low_m_gemm import bf16_low_m_gemm
from ..ops.gemm.rubin_ull_prepared_projection import prepared_projection
from ..ops.gemm.rubin_ull_warp_splitk_gemm import warp_splitk_gemm
from ..ops.moe.rubin_ull_moe_route import MoeRouteLaunch
from ..ops.moe.rubin_ull_nv_routed_experts import NvExpertsLaunch, nv_routed_experts
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.attn_res_rmsnorm_fwd import attn_res_rmsnorm_fwd
from ..ops.norm.rubin_ull_cluster_attn_res import cluster_attn_res
from ..ops.norm.rubin_ull_cluster_attn_res_ring import cluster_attn_res_ring
from ..ops.spec.dflash2_block_verify import dflash2_block_verify_seeded
from ..ops.spec.dflash2_conv_add_rmsnorm import dflash2_conv_add_rmsnorm, dflash2_rmsnorm
from ..ops.spec.dflash2_global_top32 import k31_dflash2_merge_top32, k31_dflash2_shard_top32
from ..ops.spec.dflash2_greedy_verify import dflash2_greedy_verify
from ..ops.spec.dflash2_grouped_conv import dflash2_grouped_conv
from ..ops.spec.dflash2_proposal_load import k31_dflash2_proposal_load
from ..ops.spec.dflash2_sampling import (
    DFlash2ProposalStore,
    dflash2_sampled_step,
    invalidate_dflash2_proposal_slot,
)
from ..ops.spec.dflash2_selector_draft import k31_dflash2_selector_draft
from ..ops.spec.dflash2_selector_greedy_path import k31_dflash2_selector_greedy_path
from ..ops.spec.dflash2_selector_greedy_tokens import k31_dflash2_selector_greedy_tokens
from ..ops.spec.dflash2_step_glue import k31_dflash2_noise_block, k31_draft_agreed_tokens
from ..ops.spec.draft_context_cache import DRAFT_FP8_KV_SCALE
from ..ops.spec.flashinfer_compute_probs_from_logits import flashinfer_compute_probs_from_logits
from ..ops.spec.flashinfer_sample_from_logits import flashinfer_sample_from_logits
from ..ops.spec.fused_compute_probs_from_logits import fused_compute_probs_from_logits
from ..ops.spec.fused_sample_from_logits import fused_sample_from_logits
from ..ops.spec.mla_norm_rope import dflash2_mla_block_rope, dflash2_mla_ctx_norm_rope
from ..ops.spec.rubin_dflash2_helix_combine import rubin_dflash2_helix_combine
from ..ops.spec.rubin_dflash2_window_plan import rubin_dflash2_window_plan
from ..ops.spec.rubin_dflash2_window_triton import WindowTiles, rubin_dflash2_window_triton
from ..ops.spec.rubin_ull_dflash2_conv_proj_fold import dflash2_conv_proj_fold
from ..ops.spec.rubin_ull_dflash2_headtp_decode import (
    RANKED_TILES,
    headtp_decode_ranked,
    headtp_ranked_lse_merge,
    headtp_ranked_workspace,
)
from ..ops.spec.rubin_ull_dflash2_headtp_ring import (
    build_ring,
    ring_fill_batch,
    ring_step,
    ring_store,
    window_checksums,
)
from ..ops.spec.rubin_ull_dflash2_helix_window import rubin_ull_dflash2_helix_window
from ..ops.spec.rubin_ull_dflash2_query_fold import dflash2_query_fold, fold_query_weight
from ..ops.spec.rubin_ull_dflash2_rank_topk_walk import (
    PairExchange,
    RankTopKWalk,
    k31_draft_agreed_rewind,
)
from ..ops.spec.rubin_ull_next_new_tokens_rewind import draft_next_new_tokens_rewind
from ..ops.spec.rubin_ull_select_accept import select_accept
from ..ops.spec.strict_accept import strict_accept
from ..ops.spec.vocab_parallel_argmax import vocab_parallel_argmax
from ..ops.ssm.rubin_ull_kda_decode import kda_decode
from ..ops.ssm.rubin_ull_kda_decode_checkpoint import kda_decode_checkpoint
from ..ops.ssm.rubin_ull_kda_dsa_cache_manager import KDAReplayState, KimiK3HybridCacheManager
from ..ops.ssm.rubin_ull_kda_fold_fp32 import kda_fold_fp32
from ..ops.ssm.rubin_ull_kda_head_fp32 import kda_head_verify_fp32
from ..ops.ssm.rubin_ull_kda_head_multi_fp32 import kda_head_multi_fp32
from ..ops.ssm.rubin_ull_kda_verify import kda_verify
from .weights import (
    DRAFT_DFLASH,
    DRAFT_SHAPE,
    Bf16KdaInput,
    Bf16SharedExpertWeights,
    FlatCheckpointLoader,
    KdaWeights,
)

# Identity: the generation server of a disaggregated deployment; the context servers are the
# family's dep16 / tep8 targets. The target reads no environment variable.
_TARGET_NAME = "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
_ROLES = ("generation",)
# Page layout constants the context side must match: the DSA index keys' dtype (E4M3, one scale
# per 128 values) and the KDA recurrent state's dtype. The drafter history's is _DRAFT_HISTORY_DTYPE.
_INDEXER_K_DTYPE = "fp8"
_STATE_DTYPE = torch.float32


# ---- glue ----------------------------------------
@dataclass(frozen=True)
class Bf16Weight:
    """A checkpoint BF16 matrix [out,in], already sharded by the loader."""

    weight: torch.Tensor

    def __post_init__(self) -> None:
        if self.weight.dtype != torch.bfloat16 or self.weight.ndim != 2:
            raise TypeError("Excluded projection weights must remain BF16 matrices")


def linear(x: torch.Tensor, weight: Bf16Weight) -> torch.Tensor:
    """The rank's BF16 GEMM partial; communication belongs to the decode caller."""
    if x.dtype != torch.bfloat16:
        raise TypeError("Excluded projections require BF16 input activations")
    return bf16_linear(x, weight.weight)


def embedding(ids: torch.Tensor, table: Bf16Weight) -> torch.Tensor:
    """Checkpoint embedding lookup without constructing an Embedding module: a vectorized row
    copy of the 1-D ids (``embedding_gather``), else ``F.embedding``. Both copy the rows."""
    if ids.dim() == 1 and table.weight.is_contiguous():
        return embedding_gather(ids, table.weight)
    return F.embedding(ids, table.weight)


@dataclass(frozen=True)
class NormWeight:
    """One checkpoint BF16 [hidden] norm vector and the model's epsilon."""

    weight: torch.Tensor
    eps: float

    def __post_init__(self) -> None:
        if self.weight.ndim != 1 or self.weight.dtype != torch.bfloat16:
            raise TypeError("Glue normalization weights must remain BF16 vectors")
        if not isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Normalization requires the model's positive epsilon")


def rms_norm(x: torch.Tensor, norm: NormWeight) -> torch.Tensor:
    """The same FlashInfer RMSNorm kernel as the module baseline."""
    if x.dtype != torch.bfloat16:
        raise TypeError("Glue RMSNorm requires BF16 activations")
    return flashinfer_rmsnorm(x, norm.weight, norm.eps)


@dataclass(frozen=True)
class DenseWeights:
    """L0's fused BF16 [gate;up] and row-sharded BF16 down matrices."""

    gate_up: Bf16Weight
    down: Bf16Weight

    def __post_init__(self) -> None:
        if self.gate_up.weight.shape != (2 * self.down.weight.shape[1], self.down.weight.shape[0]):
            raise ValueError("Dense gate/up and down shards must describe the same MLP")


def dense_partial(
    x: torch.Tensor, weights: DenseWeights, *, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """L0 SiTU: fused BF16 gate/up GEMM, activation, BF16 down GEMM TP partial."""
    pair = linear(x, weights.gate_up)
    hidden = situ_and_mul(pair, beta, linear_beta)
    return linear(hidden, weights.down)


@dataclass(frozen=True)
class AttnResWeights:
    """BF16 AttnRes projection/norm, followed by the layer's BF16 norm."""

    projection: Bf16Weight
    selection_norm: NormWeight
    output_norm: NormWeight

    def __post_init__(self) -> None:
        hidden = self.selection_norm.weight.numel()
        if self.projection.weight.shape != (1, hidden):
            raise ValueError("AttnRes projection must contain one checkpoint row")
        if self.output_norm.weight.shape != (hidden,):
            raise ValueError("AttnRes and output norm widths must match")


def attn_res(
    prefix: torch.Tensor, snapshots: torch.Tensor, weights: AttnResWeights
) -> torch.Tensor:
    """Mix [K,M,H] snapshots followed by [M,H] prefix, as in the ULL module baseline."""
    if prefix.dtype != torch.bfloat16 or snapshots.dtype != torch.bfloat16:
        raise TypeError("AttnRes inputs must retain BF16")
    if prefix.ndim != 2 or snapshots.ndim != 3 or snapshots.shape[1:] != prefix.shape:
        raise ValueError("AttnRes requires prefix [M,H] and snapshots [K,M,H]")
    rows, hidden = prefix.shape
    count = snapshots.shape[0]
    norm = weights.selection_norm
    if prefix.is_cuda and count + 1 <= 12 and 4096 <= hidden <= 8192 and hidden % 1024 == 0:

        def run(x, bank):
            result, _, _, _ = attn_res_fwd(
                x.reshape(x.shape[0], 1, hidden).contiguous(),
                bank.reshape(count, x.shape[0], 1, hidden).contiguous(),
                weights.projection.weight.reshape(-1).contiguous(),
                norm.weight.contiguous(),
                norm.eps,
            )
            return result.reshape(x.shape[0], hidden)

        if rows <= 16384:
            return run(prefix, snapshots)
        result = torch.empty_like(prefix)
        for start in range(0, rows, 16384):
            end = min(rows, start + 16384)
            result[start:end] = run(prefix[start:end], snapshots[:, start:end])
        return result
    values = torch.cat((snapshots.transpose(0, 1), prefix.unsqueeze(1)), dim=1).float()
    keys = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + norm.eps)
    score_weight = norm.weight.float() * weights.projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values).squeeze(1).to(prefix.dtype)


def attn_res_and_norm(
    prefix: torch.Tensor, snapshots: torch.Tensor, weights: AttnResWeights
) -> torch.Tensor:
    """Preserve the module baseline's per-token fusion gate (M<=1), then norm."""
    rows, hidden = prefix.shape
    count = snapshots.shape[0]
    if prefix.is_cuda and rows <= 1 and hidden == 7168 and count + 1 <= 12:
        output = attn_res_rmsnorm_fwd(
            prefix.reshape(rows, 1, hidden).contiguous(),
            snapshots.reshape(count, rows, 1, hidden).contiguous(),
            weights.projection.weight.reshape(-1).contiguous(),
            weights.selection_norm.weight.contiguous(),
            weights.output_norm.weight.contiguous(),
            weights.selection_norm.eps,
            weights.output_norm.eps,
        )
        return output.reshape(rows, hidden)
    return rms_norm(attn_res(prefix, snapshots, weights), weights.output_norm)


# ---- comm ----------------------------------------
#: Receive workspace of the fused DSA Helix tail per rank (two halves).
_TAIL_BYTES = 16 << 20


def _k3_helix_tail():
    from ..kernels.comm import rubin_ull_k3_helix_tail as k3_helix_tail

    return k3_helix_tail


def helix_send_rows(partial, stats, empty, world):
    """Existing eager Helix sanitization/reformat, with no collective wrapper."""
    if partial.ndim != 2 or stats.ndim != 3 or stats.shape[-1] != 2:
        raise ValueError("Helix requires partial [M,H*V] and stats [M,H,2]")
    if world <= 0 or stats.shape[1] % world or partial.shape[1] % stats.shape[1]:
        raise ValueError("Helix CP group must evenly divide attention heads")
    if partial.shape[0] != stats.shape[0]:
        raise ValueError("Helix partial and softmax statistics must cover the same rows")
    if empty is not None:
        mask = empty[: partial.shape[0]].view(-1, 1)
        partial = partial.masked_fill(mask, 0.0)
        peak = stats[..., 0].masked_fill(mask, float("-inf"))
        total = stats[..., 1].masked_fill(mask, 0.0)
        stats = torch.stack((peak, total), dim=-1)
    chunks = []
    for tensor in (partial, stats):
        tensor = tensor.transpose(1, 0).contiguous()
        chunks.extend(torch.split(tensor, tensor.shape[0] // world))
    return chunks


class FlatHelix:
    """Helix exchange/combine through the existing native operators.

    The transport follows ``cp_config`` as the stock attention path does:
    ``use_nccl_for_alltoall: false`` with ``fifo_version: 2`` uses the MNNVL
    FIFO all-to-all, whose sender applies the zero-KV mask in shared memory;
    otherwise the eager NCCL reformat/exchange runs. The FIFO workspace is
    allocated collectively here, at build time, outside graph capture.

    With the FIFO transport, ``fused_tail``
    runs K3's Helix tail kernel: the BF16 V projection of every head, the push
    into the head owner's Lamport receive buffer and the combine, in one launch
    over its own MNNVL workspace.
    """

    def __init__(self, group, mapping=None):
        self.group = tuple(group)
        if not self.group or len(set(self.group)) != len(self.group):
            raise ValueError("Helix group must be nonempty with unique ranks")
        self.native = None
        self.tail = None
        cp_config = getattr(mapping, "cp_config", None) or {}
        use_fifo = (
            len(self.group) > 1
            and not cp_config.get("use_nccl_for_alltoall", True)
            and cp_config.get("fifo_version", 2) == 2
        )
        if use_fifo:
            if mapping.cp_size != len(self.group):
                raise ValueError("Helix FIFO mapping must describe the Helix group")
            self.native = self._fifo_workspace(mapping)
            if self._tail_supported(len(self.group)):
                self.tail = self._tail_workspace(mapping)
        from tensorrt_llm.logger import logger

        if self.native is None:
            logger.info(f"K31_ULL_HELIX transport=nccl world={len(self.group)}")
        else:
            workspace = self.native[0]
            logger.info(
                f"K31_ULL_HELIX transport=fifo_v2 world={len(self.group)} "
                f"workspace_bytes_per_rank={workspace.stride(0) * workspace.element_size()} "
                f"dsa_tail={'fused' if self.tail is not None else 'off'}"
            )

    @staticmethod
    def _tail_supported(world: int) -> bool:
        from tensorrt_llm._utils import get_sm_version

        return get_sm_version() in (100, 103, 107) and _k3_helix_tail().can_run(world, 96, 512, 128)

    @staticmethod
    def _tail_workspace(mapping):
        """Collective: the fused tail's sentinel-armed receive buffers and launch counter."""
        from tensorrt_llm._torch.distributed.mnnvl_memory import (
            HelixCpMnnvlMemory,
            MnnvlMemory,
            _check_cu_result,
            cuda,
        )
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        kernel = _k3_helix_tail()
        MnnvlMemory.initialize()
        memory = HelixCpMnnvlMemory(mapping, _TAIL_BYTES)
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(memory.ptr + mapping.cp_rank * memory.rank_stride),
                kernel.SENTINEL & 0xFF,
                _TAIL_BYTES,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        state = torch.zeros(8, dtype=torch.int64, device="cuda")
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()
        return SimpleNamespace(
            memory=memory,
            state=state,
            rank=mapping.cp_rank,
            world=mapping.cp_size,
            half_bytes=_TAIL_BYTES // 2 // 16 * 16,
            max_ctas=sms,
            pdl=get_env_enable_pdl(),
        )

    def fused_tail(self, latent, stats, w_uv, empty=None, gate=None, out=None):
        """V projection + exchange + combine of latent [M,H*512] with stats [M,H,2].

        ``w_uv`` is BF16 [H,128,512]. Returns this rank's BF16 [M,H/P*128] rows, or
        None where the fused tail does not apply (the caller then projects and
        calls this object). The combine is bitwise ``helix_post_process_native``;
        the V projection accumulates in FP32 in another order than the BMM.

        ``gate`` (BF16 [M,H/P*128], unit column stride, row stride a multiple of 8): the
        rows returned are ``bf16(combined * sigmoid(gate))``, bitwise
        ``sigmoid_gate_into`` of the ungated rows, stored into ``out`` (contiguous
        [M,H/P*128]) when given. Without ``gate`` the rows are the combined ones.
        """
        tail = self.tail
        if tail is None:
            return None
        rows, heads = stats.shape[:2]
        kernel = _k3_helix_tail()
        width = heads // tail.world * kernel.V_HEAD_DIM
        if (
            latent.dtype != torch.bfloat16
            or stats.dtype != torch.float32
            or w_uv.dtype != torch.bfloat16
            or w_uv.shape != (heads, kernel.V_HEAD_DIM, kernel.LATENT_DIM)
            or latent.shape != (rows, heads * kernel.LATENT_DIM)
            or not (latent.is_contiguous() and stats.is_contiguous() and w_uv.is_contiguous())
            or rows <= 0
            or rows * tail.world * kernel.record_bytes(heads // tail.world, kernel.V_HEAD_DIM)
            > tail.half_bytes
        ):
            return None
        if out is None:
            out = latent.new_empty((rows, width))
        elif out.dtype != torch.bfloat16 or out.shape != (rows, width) or not out.is_contiguous():
            raise ValueError("The Helix tail output must be contiguous BF16 [M,H/P*128]")
        mask = None if empty is None else empty[:rows].contiguous()
        if gate is not None:
            from ..ops.comm.rubin_ull_helix_tail_gated import helix_tail_gated

            if not getattr(tail, "gate_logged", False):
                from tensorrt_llm.logger import logger

                logger.info(f"K31_ULL_DSA_TAIL gate=fused rank={tail.rank} rows={rows}")
                tail.gate_logged = True
            helix_tail_gated(
                latent.view(rows * heads, kernel.LATENT_DIM),
                stats.view(rows * heads, 2),
                w_uv,
                gate,
                out,
                mask,
                tail.rank,
                tail.world,
                tail.memory.ptr,
                tail.memory.rank_stride,
                tail.half_bytes,
                tail.state,
                tail.max_ctas,
                tail.pdl,
            )
            return out
        from ..ops.comm.rubin_ull_helix_tail import helix_tail

        helix_tail(
            latent.view(rows * heads, kernel.LATENT_DIM),
            stats.view(rows * heads, 2),
            w_uv,
            out,
            mask,
            tail.rank,
            tail.world,
            tail.memory.ptr,
            tail.memory.rank_stride,
            tail.half_bytes,
            tail.state,
            tail.max_ctas,
            tail.pdl,
        )
        return out

    @staticmethod
    def _fifo_workspace(mapping):
        """Collective: one MNNVL FIFO workspace row per CP rank, initialized and fenced."""
        from tensorrt_llm._torch.distributed.mnnvl_memory import HelixCpMnnvlMemory, MnnvlMemory
        from tensorrt_llm.bindings import internal as tllm_internal

        MnnvlMemory.initialize()
        size = tllm_internal.thop.get_helix_workspace_size_per_rank(mapping.cp_size)
        memory = HelixCpMnnvlMemory(mapping, size)
        workspace = memory.as_torch_strided_tensor(torch.uint64)
        torch.ops.trtllm.initialize_helix_workspace(workspace, mapping.cp_rank, mapping.cp_size)
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()
        # The memory object owns the mapping; keep it alive with the tensor view.
        return workspace, memory, mapping.cp_rank, mapping.cp_size

    def __call__(self, partial, stats, empty=None):
        if self.native is not None:
            return self._fifo(partial, stats, empty)
        world = len(self.group)
        chunks = helix_send_rows(partial, stats, empty, world)
        if world == 1:
            # The native collective's outputs stack the sender rank dimension.
            gathered = [tensor.unsqueeze(0) for tensor in chunks]
        else:
            gathered = torch.ops.trtllm.alltoall_helix(chunks, list(self.group), 2)
        rows, statistics = [tensor.transpose(1, 2).contiguous() for tensor in gathered]
        return torch.ops.trtllm.helix_post_process(rows, statistics, 1.0)

    def _fifo(self, partial, stats, empty):
        """Partial [M,H*V] and stats [M,H,2], heads split over the group in rank order."""
        world = len(self.group)
        if partial.ndim != 2 or stats.ndim != 3 or stats.shape[-1] != 2:
            raise ValueError("Helix requires partial [M,H*V] and stats [M,H,2]")
        rows, heads = stats.shape[:2]
        if heads % world or partial.shape[1] % heads or partial.shape[0] != rows:
            raise ValueError("Helix CP group must evenly divide attention heads")
        local, value = heads // world, partial.shape[1] // heads
        mask = None if empty is None else empty[:rows].contiguous()
        workspace, _, cp_rank, cp_size = self.native
        sent_o, sent_s = helix_alltoall_native(
            partial.contiguous().view(rows, world, local * value),
            stats.contiguous().view(rows, world, local * 2),
            workspace,
            cp_rank,
            cp_size,
            mask,
        )
        combined = helix_post_process_native(
            sent_o.view(rows, world, local, value), sent_s.view(rows, world, local, 2), 1.0, 1
        )
        return combined.view(rows, local * value)


def workspace_bytes(rows: int, hidden: int, world: int, element_size: int, threshold: int) -> int:
    """Native one-shot/two-shot payload size, without allocating tensors."""
    if min(rows, hidden, world, element_size) <= 0 or threshold < 0:
        raise ValueError("Reduction geometry must be positive and threshold nonnegative")
    one_shot = rows * hidden * world * element_size
    if one_shot <= threshold:
        return one_shot
    return 2 * math.ceil(rows / world) * world * hidden * element_size


class FlatAllReduce:
    """One serialized reduction stream, using the existing native fusion kernel.

    Construction and workspace growth are collective, outside graph capture.
    Concurrent streams must use different instances. The mapping describes a
    plain TP group (including the repurposed Helix CP group), with no CP axis.
    """

    def __init__(self, mapping, *, dtype: torch.dtype, max_rows: int, hidden: int):
        if dtype not in (torch.bfloat16, torch.float16, torch.float32) or mapping.cp_size != 1:
            raise ValueError("Flat all-reduce requires an unquantized plain TP group")
        if min(max_rows, hidden, mapping.tp_size) <= 0:
            raise ValueError("Flat all-reduce geometry must be positive")
        self.mapping, self.dtype = mapping, dtype
        self.max_rows, self.hidden = max_rows, hidden
        self.world = mapping.tp_size
        self._handle = self._comm = self._buffer = self._flags = None
        if self.world == 1:
            return
        # This constant controls workspace sizing, not execution dispatch through
        # the shared AllReduce/MNNVLAllReduce wrappers.
        from tensorrt_llm._utils import mpi_comm
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        self.threshold = _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        element = torch.empty((), dtype=dtype).element_size()
        need = workspace_bytes(max_rows, hidden, self.world, element, self.threshold)
        # Cover intermediate row counts at the one-/two-shot boundary as well.
        need = max(need, self.threshold)
        self.capacity = math.ceil(max(need, 1) / (8 * 1024**2)) * (8 * 1024**2)
        if self.capacity >= 2**32 - 1:
            raise ValueError("MNNVL workspace exceeds the kernel's uint32 address range")
        comm = mpi_comm().Split(mapping.pp_rank, mapping.tp_rank)
        self._comm = comm
        self._handle = McastGPUBuffer(
            3 * self.capacity,
            self.world,
            mapping.tp_rank,
            mapping.local_rank,
            mapping.is_multi_node(),
            comm.py2f(),
        )
        raw = self._handle.get_uc_buffer(
            mapping.tp_rank, (3 * self.capacity // 4,), torch.float32, 0
        )
        raw.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._buffer = raw.view(dtype).view(3, -1)
        self._flags = torch.tensor(
            [0, 2, self.capacity, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=raw.device,
        )

    def _run(self, partial, norm, residual, fusion):
        if partial.dtype != self.dtype or partial.ndim != 2 or partial.shape[1] != self.hidden:
            raise ValueError("Reduction operand must match its workspace dtype and width")
        if partial.shape[0] <= 0 or partial.shape[0] > self.max_rows:
            raise ValueError("Prepare a larger reduction workspace before graph capture")
        if self.world == 1:
            if residual is None:
                return partial if norm is None else rms_norm(partial, norm)
            x, res = partial.contiguous().clone(), residual.contiguous().clone()
            return flashinfer_fused_add_rmsnorm(x, res, norm.weight, norm.eps)
        if partial.device != self._buffer.device:
            raise ValueError("Reduction operand and Lamport workspace must share one device")
        return mnnvl_fusion_allreduce(
            partial.contiguous(),
            None if norm is None else norm.weight,
            residual,
            0.0 if norm is None else norm.eps,
            self._buffer,
            self._flags,
            int(fusion),
            self.world,
        )

    def __call__(self, partial: torch.Tensor) -> torch.Tensor:
        return self._run(partial, None, None, AllReduceFusionOp.NONE)

    def norm(self, partial: torch.Tensor, norm: NormWeight) -> torch.Tensor:
        return self._run(partial, norm, None, AllReduceFusionOp.RMS_NORM)

    def add_norm(self, partial: torch.Tensor, residual: torch.Tensor, norm: NormWeight):
        return self._run(partial, norm, residual, AllReduceFusionOp.RESIDUAL_RMS_NORM)


#: Two-shot o-projection rows whose consumer prefers the 100% shared-memory carve-out (MR8's
#: ``CONSUMER_CARVEOUT`` names 32 / 64 rows on Rubin, not 24).
OPROJ_CARVEOUT_ROWS = (24,)
OPROJ_CARVEOUT = 100
#: Compute capabilities of the carve-out (MR8's ``consumer_carveout``: Rubin).
OPROJ_CARVEOUT_CAPABILITIES = ((10, 7),)


def oproj_consumer_carveout(prepared, tp: int, rows: Sequence[int], device) -> list:
    """The two-shot consumer of each of ``rows`` that ``prepared``
    (MR8 ``PreparedOproj``) compiled gets the ``OPROJ_CARVEOUT`` shared-memory carve-out, as
    ``CONSUMER_CARVEOUT`` gives its 32 / 64 rows: a driver attribute of the compiled function (every
    layer's consumer of that shape), so the kernel and its outputs are the same. Only on
    ``OPROJ_CARVEOUT_CAPABILITIES`` devices, only rows ``prepared`` runs two-shot, and never a row
    the table already sets. Returns the rows set. Not collective; call outside graph capture."""
    from ..kernels.vendor.vllm_mr8 import module

    if tuple(torch.cuda.get_device_capability(device)) not in OPROJ_CARVEOUT_CAPABILITIES:
        return []
    oproj, host = module("cutedsl.oproj.prepared"), module("cutedsl.common.host")
    index = device.index if device.index is not None else torch.cuda.current_device()
    done = []
    for b in rows:
        if b not in prepared.codes or b not in prepared.twoshots or len(prepared.codes[b]) != 2:
            continue
        if oproj.consumer_carveout(tp, b, device) is not None:
            continue
        host.prepare_shared_carveout(prepared.codes[b][1], index, OPROJ_CARVEOUT)
        done.append(b)
    return done


def carve_oproj_consumers(owner: str, prepared: AbcMapping, tp: int, rank: int, device) -> None:
    """Apply the consumer carve-out over one owner's ``PreparedOproj`` per layer; logs
    ``K31_OPROJ24_CARVEOUT`` on every rank with the rows set (none off Rubin)."""
    if not prepared:
        return
    rows = set()
    for po in prepared.values():
        rows.update(oproj_consumer_carveout(po, tp, OPROJ_CARVEOUT_ROWS, device))
    logger.info(
        f"K31_OPROJ24_CARVEOUT owner={owner} rank={rank} layers={len(prepared)} "
        f"rows={sorted(rows)} carveout={OPROJ_CARVEOUT}"
    )


class FlatDsaOproj:
    """MR8's fused o-projection + TP16 all-reduce + post-attention AttnRes for DSA layers.

    One ``PreparedOproj`` per DSA layer on the BF16 [7168,768] o-projection, with
    this owner's Lamport mailbox and two-shot buffers: their parity alternates over
    DSA layers in model order, so consecutive DSA calls never share a slot set and
    the chained KDA layers' mailbox is not touched. The gated rows go through
    ``local``, the producer's mapped x. Construction is collective.
    """

    def __init__(self, owner, outputs, layers, *, block_size: int, rows, capacity: int):
        from ..kernels.vendor.vllm_mr8 import module

        if not outputs or block_size <= 0 or capacity < max(rows):
            raise ValueError("DSA o-projection needs layers, a block size and row capacity")
        oproj = module("cutedsl.oproj.prepared")
        twoshot = module("cutedsl.common.twoshot")
        widths = {w.shape[1] for w in outputs.values()}
        if len(widths) != 1 or any(w.dtype != torch.bfloat16 for w in outputs.values()):
            raise ValueError("DSA o-projections must share one BF16 local head width")
        device = next(iter(outputs.values())).device
        self.device, self.rows = device, tuple(sorted(set(rows)))
        self.local = torch.empty((capacity, widths.pop()), dtype=torch.bfloat16, device=device)
        self.box = oproj.OprojMailbox(*owner)
        self.twoshot = twoshot.TwoShot(*owner)
        self.prepared = {}
        # Per layer: the AttnRes (blocks, prefix delta) PreparedOproj compiles for.
        self.attn_res = {}
        # ``bind_value``: the core, the fused kernels per
        # (layer, rows) and the polled x buffers [2, 8, width].
        self.core, self.value, self.x_box = None, {}, None

        def norm(weight):
            return SimpleNamespace(weight=weight.weight, variance_epsilon=weight.eps)

        for ordinal, i in enumerate(sorted(outputs)):
            post = layers[i].mlp_res
            decoder = SimpleNamespace(
                layer_idx=ordinal,  # the mailbox / two-shot parity only
                self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=outputs[i])),
                mlp_res_norm=norm(post.selection_norm),
                mlp_res_proj=SimpleNamespace(weight=post.projection.weight),
                post_attention_layernorm=norm(post.output_norm),
                prev_valid_blocks=(i + block_size - 1) // block_size,
                is_block_write_layer=i % block_size == 0,
            )
            self.prepared[i] = oproj.PreparedOproj(
                decoder, self.box, tuple(rows), False, twoshot=self.twoshot, x_buffer=self.local
            )
            boundary = decoder.is_block_write_layer
            self.attn_res[i] = (decoder.prev_valid_blocks + int(boundary), not boundary)
        carve_oproj_consumers("dsa", self.prepared, self.box.tp, int(self.box.rank), device)

    def bind_value(self, core) -> None:
        """The rows at which both ``core`` (the layers'
        ``FlatDsaCore``) and the fused o-projection run (``OprojFused``, up to 8 rows) take
        ``DsaOprojFused`` instead of ``DcpValue`` + ``OprojFused``
        (``kernels/attention/rubin_ull_dsa_oproj_fused.py``): compiled here per (layer AttnRes
        shape, rows), with the two polled x buffers (armed once; parity per DSA layer as the
        core's mailboxes). A row count whose 14 clusters cannot be co-resident on this device
        keeps the two kernels (logged). Not collective; call outside graph capture."""
        from ..kernels.attention import rubin_ull_dsa_oproj_fused as fused
        from ..kernels.common import rubin_ull_spin_trap as spin_trap
        from ..kernels.vendor.vllm_mr8 import module

        oproj, host = module("cutedsl.oproj.prepared"), module("cutedsl.common.host")
        tp, width = self.box.tp, self.local.shape[1]
        if width != core.local * V_DIM or tp != core.world or set(self.prepared) - set(core.w_uv):
            raise ValueError("The fused DSA value o-projection needs the core of these layers")
        rows = [
            r
            for r in self.rows
            if r <= fused.ROWS
            and core.supports(r)
            and oproj.fused_schedule(tp, r, False, self.device)
        ]
        if not rows:
            return
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        uv = next(iter(core.w_uv.values()))
        uv_layout = (tuple(uv.shape), tuple(uv.stride()), uv.dtype)
        record = spin_trap.record_address()
        value = {}
        with torch.cuda.device(self.device):
            for r in rows:
                schedule = (oproj.drain(tp, r, self.device), oproj.oproj_vote(tp, r, self.device))
                codes = {}
                for layer, (nb, delta) in self.attn_res.items():
                    codes[layer] = fused.compile_value_oproj(
                        r, nb, delta, uv_layout, DSA_PITCH, core.world, core.local, core.cap,
                        *schedule, record, index,
                    )  # fmt: skip
                ok, line = self._value_preflight(fused, host, codes, r, index, core.rank)
                from tensorrt_llm.logger import logger

                logger.info(line)
                if ok:
                    value.update({(layer, r): code for layer, code in codes.items()})
        if value:
            box = torch.empty((2, fused.ROWS, width), dtype=torch.bfloat16, device=self.device)
            self.x_box = fused.arm(box)
            self.core, self.value = core, value
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _value_preflight(fused, host, codes, rows, index, rank):
        """(whether ``DsaOprojFused`` at ``rows`` may run here, its ``K31_DSA_VALUE_OPROJ_PREFLIGHT``
        line): the 14 clusters of 8 co-resident (``cuOccupancyMaxActiveClusters``; a failing
        query is reported, not fatal), with the registers / local memory / shared memory of the
        compiled kernel."""
        attribute = _cuda.CUfunction_attribute
        ok, fields = True, []
        for code in {id(c): c for c in codes.values()}.values():
            _, function = host.resolve_kernel(code, index)
            values = {}
            for name, key in (
                ("smem", attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                ("regs", attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
                ("local", attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            ):
                err, number = _cuda.cuFuncGetAttribute(key, function)
                values[name] = int(number) if err == _cuda.CUresult.CUDA_SUCCESS else -1
            grid = (8, fused.CLUSTERS, 1)
            clusters = max_active_clusters(code, index, grid, 512, max(values["smem"], 0), 8)
            if isinstance(clusters, int) and clusters < fused.CLUSTERS:
                ok = False
            fields.append(
                f"regs={values['regs']} local={values['local']} smem={values['smem']} "
                f"clusters={clusters}"
            )
        state = "ok" if ok else "off"
        line = f"K31_DSA_VALUE_OPROJ_PREFLIGHT {state} rows={rows} rank={rank} " + " ".join(
            sorted(set(fields))
        )
        return ok, line

    @classmethod
    def build(cls, owner, dsa, layers, **kwargs):
        """The DSA owners' fused o-projections."""
        return cls(owner, {i: d.weights.output.weight for i, d in dsa.items()}, layers, **kwargs)

    def __contains__(self, layer: int) -> bool:
        return layer in self.prepared

    def output(self, layer: int, dsa, prefix, blocks, out):
        """``FlatDsa.generation``'s output stage: the gated rows into ``local``, then the
        fused o-projection updating ``prefix`` and writing the MoE input ``out``.
        ``blocks`` is the AttnRes snapshot view [rows, blocks, hidden].

        ``run(x, combined, gate=None, gated=False)``: ``gated`` means ``combined`` already
        is the gated rows, in ``run.rows(rows)`` (``local``, where the Helix tail stores
        them); else ``gate`` (the packed projection's gate columns) or ``x``'s own gate
        projection gates ``combined`` into ``local``. ``value``: the sparse decode pushed this
        layer's partials and nothing merged them; ``DsaOprojFused`` merges, projects and gates
        them (``gate``) and runs the o-projection (only where ``run.value_rows(rows)``)."""
        prepared = self.prepared[layer]

        def run(x, combined, gate=None, gated=False, value=False):
            rows = x.shape[0]
            if value:
                from ..ops.attention.rubin_ull_dsa_value_oproj import dsa_value_oproj

                core = self.core
                parity = core.mla_query.parity[layer]
                dsa_value_oproj(
                    self.value[layer, rows], core, prepared, layer, self.x_box[parity, :rows],
                    self.x_box[1 - parity], gate, prefix, blocks, out, core.host.stream(self.device),
                )  # fmt: skip
                return out
            local = self.local[:rows]
            if gated:
                if combined.data_ptr() != local.data_ptr():
                    raise ValueError("Gated DSA rows must be in the o-projection's x buffer")
            else:
                local = dsa.gate_rows(x, combined, local, gate=gate)
            prepared.run(local, prefix, blocks, out)
            return out

        run.rows = lambda rows: self.local[:rows]
        run.value_rows = lambda rows: (layer, rows) in self.value
        return run


# ---- kda ----------------------------------------
# Verify rows (one request x q8) that run the fused head when it is enabled
# (``features.K31Features.kda_head``, "auto" by default).
HEAD_ROWS = (8,)
# The fused head runs by default only on devices with more SMs than this (Rubin). On GB300
# (152 SMs) the split-K in-projection -> ``KdaSplit`` pair is faster at q8.
HEAD_MIN_SMS = 152
# Verify rows (two requests x q8) that run the fused multi-request head on eligible layers.
MULTI_ROWS = (16,)
# Verify rows (three / four requests x q8) that run MR8's folded in-projection -> verify pair
# on eligible Rubin devices: the 64-row ``SplitKTmemGemm64`` in-projection with
# ``KdaSplit``'s prologue in its epilogue, then ``KdaVerifyCols`` on those operands.
FOLD_ROWS = (24, 32)
# The folded pair's prepared-operand scratch (FP32 [64, PREPARED_PITCH]) by device: one for every
# KDA layer, as MR8's ``kda.prepared`` (a layer's verify finishes before the next layer's
# in-projection passes its PDL wait).
_FOLD_PREPARED: dict = {}


class FlatKda:
    """One TP16 head shard, returning the BF16 output-projection TP partial.

    ``conv_cache`` is BF16 [conv_rows,3*D_local,q+2], ``state_cache`` is FP32
    [state_rows,H_local,128,128]. The manager may supply overlapping flat
    views with different row counts but a common byte stride for index zero.
    Index zero is reserved for padded requests.
    Accepted-token counts and checkpoint-index promotion belong to the cache
    manager. The caller reduces the output projection over the TP16 group.
    """

    def __init__(
        self,
        weights: KdaWeights,
        conv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        *,
        rows: Sequence[int] = (8, 16, 24, 32),
        tokens: int = 8,
        gate_lower_bound: float,
        norm_eps: float,
        projection_rows: Sequence[int] = (),
        output: torch.Tensor | None = None,
        head_side=None,
        head_polled: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """``projection_rows``: extra row counts (q1 decode) whose BF16 in-projection
        is compiled now; ``output``: caller-owned BF16 [cap, D_local] core rows
        (MR8's fused o-projection maps that buffer), which must cover every row
        count this layer will run; ``head_side``: MR8's ``KdaSide`` (the step's F_A /
        beta side stream, which the caller joins once at the end of the step), or
        None: no fused head; ``head_polled``: (this layer's [8, D_local] polled output
        buffer, the previous fused-head layer's to re-arm) for MR8's ``POLLED_OUT``
        (the fused o-projection polls the head's output; the caller arms every buffer
        once and runs the polled o-projection on it), or None."""
        if tokens < 2 or not rows or any(r <= 0 or r % tokens for r in rows):
            raise ValueError("KDA verify rows must be positive multiples of tokens >= 2")
        if (
            not isfinite(norm_eps)
            or not isfinite(gate_lower_bound)
            or norm_eps <= 0
            or gate_lower_bound >= 0
        ):
            raise ValueError(
                "KDA requires the model's positive norm epsilon and negative gate bound"
            )
        heads = weights.inputs.local_heads
        dim = heads * 128
        if weights.conv.shape != (3, 4, dim) or weights.conv.dtype != torch.float32:
            raise ValueError("MR8 KDA requires FP32 [3,4,D_local] convolution weights")
        if conv_cache.dtype != torch.bfloat16 or conv_cache.ndim != 3:
            raise TypeError("KDA convolution checkpoints must be a BF16 rank-3 tensor")
        if conv_cache.shape[1:] != (3 * dim, tokens + 2):
            raise ValueError("KDA convolution checkpoint layout must be [rows,3*D,q+2]")
        if (
            state_cache.dtype != torch.float32
            or state_cache.ndim != 4
            or state_cache.shape[1:] != (heads, 128, 128)
        ):
            raise ValueError("KDA recurrent checkpoints must be FP32 [rows,H_local,128,128]")
        # BF16 o-projection and a layer-owned [M,pitch] operand buffer, written by MR8's BF16
        # split-K in-projection.
        if not isinstance(weights.inputs, Bf16KdaInput):
            raise TypeError("KDA in-projection must be the packed BF16 bank (Bf16KdaInput)")
        device = weights.inputs.device
        if not device.type == "cuda" or conv_cache.device != device or state_cache.device != device:
            raise ValueError("KDA weights and checkpoint caches must share one GPU")
        if state_cache.stride()[-2:] != (128, 1) or conv_cache.stride(-1) != 1:
            raise ValueError("KDA cache views must have contiguous key rows and conv windows")
        if any(
            value.dtype != torch.float32
            for value in (weights.a_log, weights.dt_bias, weights.output_norm)
        ):
            raise TypeError("KDA checkpoint FP32 constants cannot be narrowed")
        constants = (weights.conv, weights.f_b, weights.a_log, weights.dt_bias, weights.output_norm)
        if any(value.device != device for value in constants):
            raise ValueError("All KDA checkpoint constants must share the projection device")
        if weights.f_b.dtype != torch.bfloat16 or weights.f_b.shape != (dim, 128):
            raise ValueError("KDA f_b must retain its BF16 [D_local,128] checkpoint shard")
        if (
            weights.a_log.shape != (heads,)
            or weights.dt_bias.shape != (dim,)
            or weights.output_norm.shape != (128,)
        ):
            raise ValueError("KDA FP32 constant shapes must match the local head partition")
        hidden = weights.inputs.hidden
        out_shape = weights.output.weight.shape
        if tuple(out_shape) != (hidden, dim):
            raise ValueError("KDA output projection must consume this rank's local head rows")
        operands = {
            "conv": weights.conv,
            "f_b": weights.f_b,
            "a_log": weights.a_log,
            "dt_bias": weights.dt_bias,
            "output_norm": weights.output_norm,
        }
        misaligned = [name for name, t in operands.items() if t.data_ptr() % 16]
        if misaligned:
            raise ValueError(f"KDA kernel operands must be 16-byte aligned: {misaligned}")
        self.weights = weights
        self.decode_core = None
        self.decode_caches = None
        self.decode_capacity = 0
        self.tokens = tokens
        self.caches = (conv_cache, state_cache)
        self.rows = tuple(sorted(set(rows)))
        cap = max((*rows, *projection_rows))
        self.shared_output = output is not None
        if output is None:
            output = torch.empty((cap, dim), dtype=torch.bfloat16, device=device)
        elif output.dtype != torch.bfloat16 or output.shape[1:] != (dim,) or output.shape[0] < cap:
            raise ValueError("Caller-owned KDA core rows must be BF16 [>=rows, D_local]")
        self.output = output
        self.projection = None
        self.projected = torch.empty(
            (cap, weights.inputs.pitch), dtype=torch.bfloat16, device=device
        )
        self._prepare_projection((*rows, *projection_rows))

        from cutlass import Float32

        from ..kernels.vendor.vllm_mr8 import module

        prepared = module("cutedsl.kda.prepared")
        # Bind the pinned PreparedKda host ABI directly. Its usual constructor
        # repacks convolution weights; these are already in the kernel layout.
        core = prepared.PreparedKda.__new__(prepared.PreparedKda)
        core.heads, core.dim, core.has_bias = heads, dim, False
        core.weights = (
            weights.conv,
            weights.dt_bias,
            weights.f_b,
            weights.a_log,
            weights.dt_bias,
            weights.output_norm,
        )
        core.device = device
        pitch = weights.inputs.pitch
        core.policies = {(r // tokens, tokens): (pitch, 0) for r in self.rows}
        core.scalars = tuple(Float32(v) for v in (gate_lower_bound, 128**-0.5, norm_eps))
        core.conv_dim_first = True
        core.kernels = {}
        # Both in-projections leave raw F_A/beta, so use KdaSplit's own prologue
        # (MR8 prepares verify operands only in KdaInProj64, Rubin 64-row fold).
        core.prepared = frozenset()
        core.prepared_kernels = {}
        # KdaVerifyCols' F_A reader counts per request (the folded pair: up to four requests)
        core.readers = torch.zeros(4, dtype=torch.int32, device=device)
        self.core = core.prepare_caches(self.caches)
        self.head = None
        self.head_rows: frozenset[int] = frozenset()
        self.head_launches = 0  # host-side count of fused-head launches (path logs)
        self.head_polled = None  # (own, previous) POLLED_OUT buffers
        self.multi = None
        self.multi_rows: frozenset[int] = frozenset()
        self._multi_side = None  # the side stream of a layer that may run the multi head
        if head_side is not None and self.head_supported(weights, tokens):
            self._prepare_head(head_side, [r for r in HEAD_ROWS if r in self.rows], head_polled)
            self._multi_side = head_side
            self._prepare_multi([r for r in MULTI_ROWS if r in self.rows])
        self.fold = None
        self.fold_rows: frozenset[int] = frozenset()
        self.fold_launches = 0  # host-side count of folded-pair launches (path logs)
        self._fold_on = self.fold_supported(weights, tokens)
        if self._fold_on:
            self._prepare_fold([r for r in FOLD_ROWS if r in self.rows])

    @staticmethod
    def fold_supported(weights: KdaWeights, tokens: int) -> bool:
        """Whether this layer can run MR8's folded pair (``FOLD_ROWS``): a resident BF16
        [3216, 7168] in-projection over 6 local heads, q8, compute capability 10.x (tensor
        memory), and more than ``HEAD_MIN_SMS`` SMs."""
        inputs = weights.inputs
        if not isinstance(inputs, Bf16KdaInput) or tokens != 8:
            return False
        weight = getattr(inputs.packed, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16:
            return False
        if inputs.local_heads != 6 or tuple(weight.shape) != (3216, 7168):
            return False
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device)[0] != 10:
            return False
        props = torch.cuda.get_device_properties(weight.device)
        return props.multi_processor_count > HEAD_MIN_SMS

    def _prepare_fold(self, rows: Sequence[int]) -> None:
        """MR8's folded pair (``kernels/ssm/rubin_ull_kda_fold_fp32.py``) for three / four
        requests x q8 verify rows (``rows``: ``FOLD_ROWS`` among this layer's rows), with the
        in-projection at MR8's 64-row Rubin tactic (``WIDE_B64``), over this layer's own
        projection buffer (F_A columns armed here once, re-armed by each verify's last reader)."""
        rows = [r for r in rows if r in FOLD_ROWS and r not in self.fold_rows]
        if not rows:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the folded KDA pair before graph capture")
        from ..kernels.ssm import rubin_ull_kda_fold_fp32 as fold
        from ..kernels.vendor.vllm_mr8 import module

        weight = self.weights.inputs.packed.weight
        device = weight.device
        if self.fold is None:
            tactic = module("runtime.providers").WIDE_B64[weight.shape[0]][:4]
            projected = torch.empty(
                (max(FOLD_ROWS), weight.shape[0]), dtype=torch.bfloat16, device=device
            )
            fold.PreparedKdaFold.arm(projected, self.core.dim)
            prepared = _FOLD_PREPARED.get(device)
            if prepared is None:
                pitch = module("cutedsl.kda.inproj64").PREPARED_PITCH
                prepared = torch.empty((fold.TILE_ROWS, pitch), dtype=torch.float32, device=device)
                _FOLD_PREPARED[device] = prepared
            self.fold = fold.PreparedKdaFold(weight, self.core, tactic, projected, prepared)
        self.fold.prepare_caches(self.caches, tuple(rows))
        self.fold_rows = self.fold_rows | frozenset(rows)

    @staticmethod
    def head_supported(weights: KdaWeights, tokens: int) -> bool:
        """Whether this layer can run ``KdaHeadVerify`` (``HEAD_ROWS``): a resident BF16
        [pitch, 7168] in-projection over 6 local heads, q8, compute capability 10.x
        (tensor memory, 16-CTA clusters) and more than ``HEAD_MIN_SMS`` SMs."""
        inputs = weights.inputs
        if not isinstance(inputs, Bf16KdaInput) or tokens != 8:
            return False
        weight = getattr(inputs.packed, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16:
            return False
        if inputs.local_heads != 6 or tuple(weight.shape) != (inputs.pitch, 7168):
            return False
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device)[0] != 10:
            return False
        props = torch.cuda.get_device_properties(weight.device)
        return props.multi_processor_count > HEAD_MIN_SMS

    @staticmethod
    def head_order(device: torch.device) -> str:
        """Use sibling launch order where the device has room for its concurrent CTAs."""
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        return "sibling" if sms > HEAD_MIN_SMS else "event"

    def _prepare_head(self, side, rows: Sequence[int], polled=None) -> None:
        """MR8's ``PreparedKdaHead`` over this layer's packed BF16 weight and FP32 state,
        with ``KdaHeadVerify`` for one request x q verify rows (``rows``), its F_A waits
        bounded (``kernels/ssm/rubin_ull_kda_head_fp32.py``); its output is
        ``output`` (read by the o-projection after its PDL wait) or, with ``polled``,
        this layer's polled buffer (``POLLED_OUT``). Its F_A / beta buffer and counter
        pair are this layer's own; the counter is re-armed by its last reader, so a
        layer runs at most once per step."""
        if not rows:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the fused KDA head before graph capture")
        from ..kernels.ssm.rubin_ull_kda_head_fp32 import PreparedKdaHeadFp32

        weight = self.weights.inputs.packed.weight
        device, dim = weight.device, self.core.dim
        fa_out = torch.empty(
            (max(rows), weight.shape[0] - 4 * dim), dtype=torch.bfloat16, device=device
        )
        counter = torch.zeros(2, dtype=torch.int32, device=device)
        verify = tuple((r, self.tokens) for r in rows if r == self.tokens)
        prepared = PreparedKdaHeadFp32(
            weight, self.core, (), side, counter, fa_out, verify=verify,
            order=self.head_order(device),
        )  # fmt: skip
        if polled is not None:
            own, other = polled
            if own.shape != (self.tokens, dim) or other.shape != own.shape:
                raise ValueError("Polled KDA head buffers must be [q, D_local]")
            # PreparedKdaHead keys its polled verify kernel on ``polled`` (it filters the
            # constructor argument by its q = 1 batches, which this layer does not run).
            prepared.polled = tuple(r for r, _ in verify)
            self.head_polled = polled
        self.head = prepared.prepare_caches(self.caches)
        self.head_rows = frozenset(r for r, _ in verify)

    def _prepare_multi(self, rows: Sequence[int]) -> None:
        """MR8's ``KdaHeadMulti`` (``kernels/ssm/rubin_ull_kda_head_multi_fp32.py``) for two
        requests x q8 verify rows (``rows``: ``MULTI_ROWS`` among this layer's rows) over this
        layer's packed BF16 weight and FP32 state, in the 8-row head's launch order
        (``head_order``), on the step's F_A / beta side stream. Its F_A / beta buffer and counter
        pair are its own, re-armed by its last reader; its output is ``output``, read by the
        o-projection after its PDL wait (no polled buffer)."""
        rows = [r for r in rows if r == 2 * self.tokens]
        if not rows or self.multi is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the fused KDA multi head before graph capture")
        from ..kernels.ssm.rubin_ull_kda_head_multi_fp32 import PreparedKdaHeadMultiFp32

        weight = self.weights.inputs.packed.weight
        device, dim = weight.device, self.core.dim
        fa_out = torch.empty(
            (max(rows), weight.shape[0] - 4 * dim), dtype=torch.bfloat16, device=device
        )
        counter = torch.zeros(2, dtype=torch.int32, device=device)
        verify = tuple((r, self.tokens) for r in rows)
        prepared = PreparedKdaHeadMultiFp32(
            weight, self.core, (), self._multi_side, counter, fa_out, verify=verify,
            order=self.head_order(device),
        )  # fmt: skip
        self.multi = prepared.prepare_caches(self.caches)
        self.multi_rows = frozenset(rows)

    def _prepare_projection(self, rows: Sequence[int]) -> None:
        """MR8's split-K BF16 in-projection with K3 ULL's per-bin tactics for ``rows``.

        Compiled code is cached per tactic, so layers share it; each layer holds
        only its weight's TMA descriptors. Eager only (before graph capture).
        """
        known = set(self.projection.kernels) if self.projection is not None else set()
        wanted = known | set(rows)
        if wanted == known:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the KDA in-projection before graph capture")
        from ..kernels.vendor.vllm_mr8 import module

        providers = module("runtime.providers")
        weight = self.weights.inputs.packed.weight
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        policies = providers.projection_policies(
            16, weight.shape[0], sorted(wanted), sms, self.tokens
        )
        missing = sorted(wanted - set(policies))
        if missing:
            raise ValueError(f"No MR8 KDA in-projection tactic for rows {missing}")
        self.projection = providers.PreparedProjection(weight, policies).prepare()
        if self.projected.shape[0] < max(wanted):
            self.projected = self.projected.new_empty((max(wanted), self.projected.shape[1]))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """BF16 [M,pitch] packed operand rows for MR8 KDA."""
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device != self.output.device:
            raise TypeError("KDA input must be BF16 rows on the projection device")
        if x.shape[0] not in self.projection.kernels:
            self._prepare_projection((x.shape[0],))
        return prepared_projection(self.projection, x, self.projected[: x.shape[0]])

    def output_partial(self, rows: torch.Tensor) -> torch.Tensor:
        """This rank's BF16 [M,H] o-projection partial of the core rows (solo layer).

        The chained layers instead give the core rows to MR8's fused o-projection
        + all-reduce + AttnRes (``PreparedOproj``).
        """
        return bf16_linear(rows, self.weights.output.weight)

    def _grow_output(self, rows: int) -> None:
        if self.output.shape[0] >= rows:
            return
        if self.shared_output:
            raise ValueError("Caller-owned KDA core rows are smaller than the requested rows")
        self.output = self.output.new_empty((rows, self.output.shape[1]))

    def prepare_rows(self, rows: Sequence[int]) -> None:
        """Incrementally prepare wider verify shapes using the same native weights.

        Call during eager warmup before graph capture. No weights are repacked and
        existing compiled shapes remain valid when output scratch grows.
        """
        missing = sorted(set(rows) - set(self.rows))
        if not missing:
            return
        if any(r <= 0 or r % self.tokens for r in missing):
            raise ValueError("KDA verify rows must divide the configured token width")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare KDA row shapes before graph capture")
        known = self.core.policies
        added = {(r // self.tokens, self.tokens): (self.weights.inputs.pitch, 0) for r in missing}
        self.core.policies = added
        try:
            self.core.prepare_caches(self.caches)
        finally:
            self.core.policies = known | added
        self.rows = tuple(sorted(set(self.rows) | set(missing)))
        self._grow_output(max(missing))
        self._prepare_projection(missing)
        if self._multi_side is not None:
            self._prepare_multi([r for r in MULTI_ROWS if r in missing])
        if self._fold_on:
            self._prepare_fold([r for r in FOLD_ROWS if r in missing])

    def prepare_decode(self, rows: Sequence[int]) -> None:
        """Prepare MR8 q1 kernels over compact state/conv scratch, sharing weights."""
        if not rows or any(r <= 0 for r in rows):
            raise ValueError("KDA decode rows must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare q1 KDA kernels before graph capture")
        if self.decode_core is None:
            from ..kernels.vendor.vllm_mr8 import module

            prepared = module("cutedsl.kda.prepared")
            core = prepared.PreparedKda.__new__(prepared.PreparedKda)
            for name in (
                "heads",
                "dim",
                "has_bias",
                "weights",
                "device",
                "scalars",
                "conv_dim_first",
                "readers",
            ):
                setattr(core, name, getattr(self.core, name))
            core.policies, core.kernels = {}, {}
            core.prepared, core.prepared_kernels = frozenset(), {}
            self.decode_core = core
        core = self.decode_core
        grow = max(rows) > self.decode_capacity
        if grow:
            capacity, heads = max(rows), self.weights.inputs.local_heads
            device = self.output.device
            self.decode_caches = (
                torch.empty(
                    (capacity + 1, 3 * heads * 128, 3), dtype=torch.bfloat16, device=device
                ),
                torch.empty((capacity + 1, heads, 128, 128), dtype=torch.float32, device=device),
            )
            self.decode_capacity = capacity
            core.kernels.clear()
        added = {r: (self.weights.inputs.pitch, 0) for r in rows if r not in core.kernels}
        policies = core.policies | added
        core.policies = policies if grow else added
        try:
            core.prepare_caches(self.decode_caches)
        finally:
            core.policies = policies
        self._grow_output(max(rows))
        self._prepare_projection(rows)

    def run_decode(
        self, x: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor
    ) -> torch.Tensor:
        """q1 KDA layer: the o-projection partial of ``attend_decode``."""
        return self.output_partial(self.attend_decode(x, slots, accepted))

    def attend_decode(
        self, x: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor
    ) -> torch.Tensor:
        """q1 in-projection + KDA core rows, preserving checkpoint-zero publication.

        ``slots`` covers all checkpoint positions, even though ``x`` has one row
        per request. The caller records zero accepted drafts after the complete
        plain-decode step; the selected prior history is read before each layer
        publishes its new checkpoint zero. Padding never writes the flat cache.
        """
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device != self.output.device:
            raise ValueError("q1 KDA requires native BF16 rows on the projection device")
        rows = x.shape[0]
        if self.decode_core is None or rows not in self.decode_core.kernels:
            self.prepare_decode((rows,))
        if slots.ndim != 2 or slots.shape[0] != rows:
            raise ValueError("q1 checkpoint indices must cover each request's position bank")
        scratch = tuple(t[: rows + 1] for t in self.decode_caches)
        kda_decode_checkpoint(self.caches, scratch, slots, accepted, scatter=False)
        dense_slots = torch.arange(1, rows + 1, dtype=torch.int32, device=x.device)
        dense_slots = torch.where(slots[:, 0] > 0, dense_slots, 0)
        projected = self.project(x)
        out = kda_decode(
            self.decode_core, projected, self.output[:rows], self.decode_caches, dense_slots
        )
        kda_decode_checkpoint(self.caches, scratch, slots, accepted, scatter=True)
        return out

    def run(self, x: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
        """q8 verify KDA layer: the o-projection partial of ``attend``."""
        return self.output_partial(self.attend(x, slots, accepted))

    def attend(self, x: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
        """In-projection, then MR8 verify recurrence (``head_rows`` / ``multi_rows``: both in
        the fused head; ``fold_rows``: the folded pair); returns the BF16 core rows.

        ``x`` is BF16 [M,H]; ``slots`` int32 [M/q,q] and ``accepted`` int32
        [>=M/q]. No acceptance or state tensor is read back to the host.
        """
        if x.ndim != 2:
            raise ValueError("KDA inputs must be BF16 rows")
        if x.shape[0] not in self.rows:
            self.prepare_rows((x.shape[0],))
        requests = x.shape[0] // self.tokens
        if slots.dtype != torch.int32 or slots.shape != (requests, self.tokens):
            raise ValueError("KDA checkpoint indices must be int32 [requests,q]")
        if accepted.dtype != torch.int32 or accepted.ndim != 1 or accepted.numel() < requests:
            raise ValueError("KDA accepted counts must be int32 [>=requests]")
        if slots.device != x.device or accepted.device != x.device:
            raise ValueError("KDA checkpoint metadata must share the activation device")
        if x.shape[0] in self.head_rows:
            self.head_launches += 1
            if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.output.device:
                raise TypeError("The fused KDA head needs contiguous BF16 rows on its device")
            out, rearm = self.output[: x.shape[0]], None
            if self.head_polled is not None:
                out, rearm = self.head_polled
            return kda_head_verify_fp32(self.head, x, out, self.caches, slots, accepted, rearm)
        if x.shape[0] in self.multi_rows:
            self.head_launches += 1
            if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.output.device:
                raise TypeError("The fused KDA multi head needs contiguous BF16 rows on its device")
            out = self.output[: x.shape[0]]
            return kda_head_multi_fp32(self.multi, x, out, self.caches, slots, accepted)
        if x.shape[0] in self.fold_rows:
            self.fold_launches += 1
            if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.output.device:
                raise TypeError("The folded KDA pair needs contiguous BF16 rows on its device")
            out = self.output[: x.shape[0]]
            return kda_fold_fp32(self.fold, x, out, self.caches, slots, accepted)
        projected = self.project(x)
        return kda_verify(
            self.core, projected, self.output[: x.shape[0]], self.caches, slots, accepted
        )


# ---- dsa_inproj ----------------------------------------
PITCH, HIDDEN = 3040, 7168
# Row counts this engine serves (the rest keep ``bf16_projection``).
ROWS = range(3, 17)
_LOGGED: set[str] = set()


def _warp():
    from ..kernels.vendor.vllm_main import module

    return module("cutedsl.gemm.warp_splitk")


def smem_limits(device: torch.device) -> tuple[int, int]:
    """(per-CTA opt-in, per-SM) shared memory bytes of ``device``."""
    props = torch.cuda.get_device_properties(device)
    cta = props.shared_memory_per_block_optin
    return cta, getattr(props, "shared_memory_per_multiprocessor", cta + 1024)


def warp_tactic(rows: int, sms: int, cta_smem: int, sm_smem: int):
    """vLLM main ``decode.warp_tactic`` for the [3040, 7168] bank, with the ring depth bounded by
    ``cta_smem`` per CTA and ``sm_smem`` per SM (each resident CTA also takes the driver's 1 KB)
    instead of sm_100's constants (and by the vendored kernel's own sm_100 check); None if
    even a 2-stage ring does not fit.

    Output tile 16 where its ``PITCH // 16`` CTAs fit one per SM (Rubin: 190 of 212 SMs), else
    vLLM's 32 (GB300: 95). The CTAs fill their weight rings before the PDL wait; the bank's
    remaining tiles stream after it, so twice the CTAs (each with half the rows and the deeper
    ring that fits) halve the bytes per SM left after the wait. An output's K split and
    accumulation order do not depend on the tile width, so neither do its bits."""
    w = _warp()
    if PITCH % 16 == 0 and PITCH // 16 <= sms:
        output = 16
    else:
        output = 32 if PITCH % 32 == 0 and PITCH // 32 <= 2 * sms else 16
    token = 8 if rows <= 8 else 16 if rows <= 16 else 32
    k_tile = 256 if HIDDEN % 256 == 0 and HIDDEN // 256 != 2 else 128
    resident = -(-((PITCH // output) * -(-rows // token)) // sms)
    cta_smem = min(cta_smem, w._SMEM_CAPACITY)
    best = None
    for stages in range(2, min(w._MAX_STAGES, HIDDEN // k_tile) + 1):
        t = w.WarpSplitKTactic(output, token, k_tile, stages, 2)
        need = w._smem_bytes(t)
        if need <= cta_smem and resident * (need + w._CTA_RESERVED_SMEM) <= sm_smem:
            best = t
    return best


#: The deepest warp split-K ring the kernel runs: one warp's lanes initialize the stage barriers.
#: vLLM's tactic check stops at its policy cap (16 stages, where a cold read saturates); the
#: in-projection's weight loaders fill the ring before the PDL wait, so a deeper ring (as far as
#: shared memory allows) moves weight bytes ahead of the wait.
WARP_RING_STAGES_MAX = 32
#: Rubin (``_rubin_rows``) in-projection tactics by row count that replace ``warp_tactic``'s pick;
#: empty: every row count takes the rule.
_DSA_INPROJ_TACTICS_RUBIN: dict = {}


def _check_ring_tactic(tactic, n: int, k: int) -> None:
    """vLLM's tactic check with the ring depth bounded by ``WARP_RING_STAGES_MAX`` and the K tiles
    instead of the policy cap."""
    w = _warp()
    t = tactic
    capped = w.WarpSplitKTactic(
        t.output_tile, t.token_tile, t.k_tile, min(t.stages, w._MAX_STAGES), t.b_loader_warps
    )
    w.check_tactic(capped, n, k)
    if not 2 <= tactic.stages <= min(WARP_RING_STAGES_MAX, k // tactic.k_tile):
        raise ValueError(
            f"invalid ring depth {tactic.stages} for K={k} in tiles of {tactic.k_tile}"
        )
    if w._smem_bytes(tactic) > w._SMEM_CAPACITY:
        raise ValueError(f"{tactic} needs {w._smem_bytes(tactic)} B of shared memory")


@lru_cache(maxsize=None)
def _compile_ring_warp_splitk(device_index: int, n: int, k: int, tactic):
    """vLLM's ``compile_warp_splitk`` (dynamic M up to 32, PDL on) under ``_check_ring_tactic``."""
    w = _warp()
    _check_ring_tactic(tactic, n, k)
    with torch.cuda.device(device_index):
        shapes = (((n, k), False), ((w._MAX_M, k), True), ((n,), False), ((w._MAX_M, n), True))
        args = tuple(
            w._view(torch.empty(s, device="cuda", dtype=torch.bfloat16), dynamic_m=d)
            for s, d in shapes
        )
        stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(w.CpAsyncWarpSplitKKernel(tactic, True, False), *args, stream)


class _RingWarpSplitK:
    """``PreparedWarpSplitK`` whose tactics may take rings past vLLM's policy cap
    (``_check_ring_tactic``); the same kernel class and launch."""

    def __init__(self, weight: torch.Tensor, tactics: dict) -> None:
        w = _warp()
        n, k = weight.shape
        self.weight, self.tactics = weight, dict(tactics)
        self.codes = {
            m: _compile_ring_warp_splitk(weight.device.index, n, k, t)
            for m, t in self.tactics.items()
        }
        self.weight_view = w._view(weight)

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        w = _warp()
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self.codes[x.shape[0]](
            self.weight_view,
            w._view(x, dynamic_m=True),
            w._view(out[0]),
            w._view(out, dynamic_m=True),
            stream,
        )
        return out


def parse_warp_tactic(spec: str):
    """``WarpSplitKTactic`` of "output_tile:token_tile:k_tile:stages:b_loader_warps"."""
    return _warp().WarpSplitKTactic(*(int(p) for p in spec.split(":")))


def dsa_inproj_tactics(device: torch.device) -> dict:
    """{rows: tactic} that replace ``warp_tactic`` for the in-projection on ``device``: the model
    config's ``dsa_inproj_tactic`` at the row counts of its token tile (3-8 for 8), else the Rubin
    table on Rubin, else none."""
    spec = k31_features().dsa_inproj_tactic
    if spec:
        t = parse_warp_tactic(spec)
        _check_ring_tactic(t, PITCH, HIDDEN)
        low = 0 if t.token_tile == 8 else 8 if t.token_tile == 16 else 16
        return {r: t for r in ROWS if low < r <= t.token_tile}
    if device.type == "cuda" and _rubin_rows(device):
        return dict(_DSA_INPROJ_TACTICS_RUBIN)
    return {}


class DsaInProjection:
    """vLLM main's warp split-K of the packed DSA bank for the prepared row counts: ``warp_tactic``
    or, per row count, ``tactics`` (``dsa_inproj_tactics``)."""

    def __init__(self, weight: torch.Tensor, tactics: Optional[dict] = None) -> None:
        if (
            weight.dtype != torch.bfloat16
            or weight.shape != (PITCH, HIDDEN)
            or not weight.is_contiguous()
            or weight.data_ptr() % 32
        ):
            raise ValueError("The DSA in-projection needs the packed contiguous BF16 [3040, 7168]")
        self.weight = weight
        self.tactics = dict(tactics or {})
        self.rows: tuple[int, ...] = ()
        self.engine = None

    def prepare(self, rows: Iterable[int]) -> None:
        """Compile the engines of ``rows`` (those in ``ROWS``) outside graph capture."""
        wanted = tuple(sorted(set(self.rows) | {r for r in rows if r in ROWS}))
        if wanted == self.rows or not self.weight.is_cuda:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the DSA in-projection before graph capture")
        props = torch.cuda.get_device_properties(self.weight.device)
        cta, sm = smem_limits(self.weight.device)
        tactics = {
            r: self.tactics.get(r) or warp_tactic(r, props.multi_processor_count, cta, sm)
            for r in wanted
        }
        missing = [r for r, t in tactics.items() if t is None]
        if missing:
            raise RuntimeError(
                f"K31_DSA_INPROJ_PREFLIGHT: no warp split-K ring of >= 2 stages fits rows {missing} "
                f"(sm_{props.major}{props.minor}, {props.multi_processor_count} SMs, "
                f"{cta} B per CTA, {sm} B per SM)"
            )
        w = _warp()
        deep = any(t.stages > w._MAX_STAGES for t in tactics.values())
        engine = _RingWarpSplitK if deep else w.PreparedWarpSplitK
        self.engine, self.rows = engine(self.weight, tactics), wanted
        for r in wanted:
            if r in self.tactics and _rubin_rows(self.weight.device):
                t = self.tactics[r]
                log_rubin_row(
                    "J",
                    f"rows={r} tactic=({t.output_tile}, {t.token_tile}, {t.k_tile}, {t.stages}, "
                    f"{t.b_loader_warps})",
                )
        line = self.preflight()
        if line not in _LOGGED:
            from tensorrt_llm.logger import logger

            _LOGGED.add(line)
            logger.info(line)

    def supports(self, x: torch.Tensor) -> bool:
        return (
            self.engine is not None
            and x.shape[0] in self.rows
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == HIDDEN
            and x.is_contiguous()
            and x.data_ptr() % 32 == 0
            and x.device == self.weight.device
        )

    def ready(self, x: torch.Tensor) -> bool:
        """:meth:`supports`, preparing ``x``'s row count first when outside graph capture."""
        rows = x.shape[0]
        if (
            x.is_cuda
            and rows not in self.rows
            and rows in ROWS
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.prepare((rows,))
        return self.supports(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], PITCH), dtype=x.dtype, device=x.device)
        return warp_splitk_gemm(self.engine, x, out)

    def launch_configs(self) -> dict[int, dict]:
        """Per prepared row count: tactic, grid, block, cluster, dynamic shared memory (the Rubin
        preflight list)."""
        w = _warp()
        configs = {}
        for rows in self.rows:
            t = self.engine.tactics[rows]
            configs[rows] = dict(
                kind="warp",
                tactic=(t.output_tile, t.token_tile, t.k_tile, t.stages, t.b_loader_warps),
                grid=(-(-PITCH // t.output_tile), -(-rows // t.token_tile), 1),
                block=(2 + t.b_loader_warps + 4) * 32,
                cluster=1,
                smem=w._smem_bytes(t),
            )
        return configs

    def preflight(self) -> str:
        """Raise if a prepared kernel needs more dynamic shared memory than this device allows;
        return ``K31_DSA_INPROJ_PREFLIGHT ok ...`` with each row count's launch config."""
        props = torch.cuda.get_device_properties(self.weight.device)
        limit, per_sm = smem_limits(self.weight.device)
        fields = []
        for rows, c in self.launch_configs().items():
            if c["smem"] > limit:
                raise RuntimeError(
                    f"K31 DSA in-projection warp split-K at {rows} rows needs {c['smem']} B of "
                    f"shared memory per CTA, the device allows {limit} B "
                    f"(sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)"
                )
            grid = "x".join(map(str, c["grid"][:2]))
            fields.append(
                f"r{rows}={','.join(map(str, c['tactic']))}:grid{grid}:block{c['block']}"
                f":smem{c['smem']}"
            )
        return (
            f"K31_DSA_INPROJ_PREFLIGHT ok sm={props.major}{props.minor} "
            f"sms={props.multi_processor_count} smem_optin={limit} smem_per_sm={per_sm} "
            + " ".join(fields)
        )


# ---- dsa_qproj ----------------------------------------
INDEXER_ROWS = slice(18432, 22528)  # the 32 x 128 indexer query rows of the packed bank
# The largest SM count of the GB300 table (MR8's ``providers`` uses the same split).
GB300_SMS = 152


# The Rubin rows of the tables below are keyed on SM107 (compute capability 10.7) with more SMs than
# GB300_SMS, so another part with more SMs never takes them by its SM count alone.
RUBIN_CAPABILITY = (10, 7)


@lru_cache(maxsize=None)
def _device_sms(device: torch.device) -> int:
    """The SM count of ``device``."""
    return torch.cuda.get_device_properties(device).multi_processor_count


@lru_cache(maxsize=None)
def _rubin_rows(device: torch.device) -> bool:
    """Whether ``device`` takes the Rubin rows: compute capability ``RUBIN_CAPABILITY`` and more SMs
    than ``GB300_SMS``."""
    return (
        torch.cuda.get_device_capability(device) == RUBIN_CAPABILITY
        and _device_sms(device) > GB300_SMS
    )


#: (item, detail) already logged (``log_rubin_row``).
_RUBIN_ROWS_LOGGED: set = set()


def log_rubin_row(item: str, detail: str) -> None:
    """``K31_RUBIN_ROW item=<item> <detail> rank=<rank>`` once per process and (item, detail): this process
    launches a table row that only ``_rubin_rows`` devices (Rubin) take. The choice is made
    on the host, so a captured graph replays the logged launch."""
    if (item, detail) in _RUBIN_ROWS_LOGGED:
        return
    _RUBIN_ROWS_LOGGED.add((item, detail))
    logger.info(f"K31_RUBIN_ROW item={item} {detail} rank={mpi_rank()}")


# (mma_m, mma_n, split_k, stages[, full_rows]) by the largest row count each serves, per
# device class (``full_rows``: each CTA reads whole q_a rows instead of exchanging sums).
# GB300 (152 SMs), single GPU, CUDA graph, 24 rotated weight copies, us added after the DSA hidden
# projection on the indexer stream, two kernels -> this: rows 1-8 (128, 8, 3, 4), 96 CTAs:
# 5.7 / 10.2 / 7.1 / 7.1 -> 4.9 / 4.9 / 3.7 / 3.5 (rows 1 / 2 / 4 / 8); rows 9-16
# (64, 16, 3, 4), 192 CTAs: 6.0 -> 4.1 (rows 16). Rows 17-32 keep the two kernels (6.7): every
# N32 tactic (8.9+) and two N16 tiles (7.6+) were slower.
GB300_TACTICS = ((8, (128, 8, 3, 4)), (16, (64, 16, 3, 4)))
# Rubin (212 SMs): 192 CTAs (64-row M tiles, K split 3) resident in one wave for both bins.
# Not measured on Rubin; its numerics are checked on GB300 with the same tactics forced, and
# the Rubin preflight times it against the GB300 table's (128, 8, 3, 4).
RUBIN_TACTICS = ((8, (64, 8, 3, 4)), (16, (64, 16, 3, 4)))


def qproj_tactic_for(rows: int, sms: int):
    """The (mma_m, mma_n, split_k, stages) tactic of ``rows`` on a device of ``sms`` SMs,
    or None (the two-kernel path)."""
    table = GB300_TACTICS if sms <= GB300_SMS else RUBIN_TACTICS
    return next((t for bound, t in table if 1 <= rows <= bound), None)


@cache
def _qproj_compile(tactic: tuple, arch: str):
    """Code-only cache: layers share the compiled kernel of a tactic."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from ..kernels.attention.rubin_ull_dsa_qproj import NormSplitKGemm, _splitk

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    full_rows = bool(tactic[4]) if len(tactic) > 4 else False
    kernel = NormSplitKGemm(tactic=_splitk.SplitKTactic(*tactic[:4]), full_rows=full_rows)
    args = (fake(1), fake(1), fake(0), cutlass.Int64(0), cutlass.Int64(0), cutlass.Float32(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


class IndexerQProjection:
    """One DSA layer's fused q_a RMSNorm + indexer query projection, prepared for ``rows``.

    ``weight`` is the layer's [4096, 1536] BF16 indexer row view of the packed query bank
    (no copy); ``norm`` the q_a ``NormWeight``. Construction compiles (cached per tactic)
    and builds the weight's TMA descriptors; call it before graph capture.
    ``tactics`` overrides the per-device table (tests): {rows: tactic}.
    """

    def __init__(self, weight: torch.Tensor, norm, rows: Sequence[int], tactics=None) -> None:
        from ..kernels.vendor.vllm_mr8 import module

        if (
            weight.dtype != torch.bfloat16
            or weight.shape != (4096, 1536)
            or weight.stride() != (1536, 1)
            or weight.data_ptr() % 16
        ):
            raise ValueError("The indexer query rows must be a contiguous BF16 [4096, 1536] view")
        if norm.weight.shape != (1536,) or not norm.weight.is_contiguous():
            raise ValueError("The q_a RMSNorm weight must be a contiguous BF16 [1536] vector")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the indexer query projection before graph capture")
        self.device = weight.device
        self.weight, self.gamma, self.eps = weight, norm.weight, float(norm.eps)
        self.host = module("cutedsl.common.host")
        self.projection = module("cutedsl.gemm.projection")
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        arch = self.host.architecture(self.device)
        if tactics is None:
            tactics = {r: qproj_tactic_for(r, sms) for r in rows}
        self.tactics = {r: tuple(t) for r, t in tactics.items() if t is not None}
        self.weight_view = self.projection.matrix_view(weight)
        self.kernels, self.maps = {}, {}
        with torch.cuda.device(self.device):
            for count, tactic in self.tactics.items():
                self.kernels[count] = _qproj_compile(tactic, arch)
                if tactic[0] not in self.maps:
                    self.maps[tactic[0]] = self.projection.weight_map(weight, tactic[0])

    def supports(self, qa: torch.Tensor) -> bool:
        return (
            qa.shape[0] in self.kernels
            and qa.ndim == 2
            and qa.shape[1] == 1536
            and qa.dtype == torch.bfloat16
            and qa.device == self.device
            and qa.stride(1) == 1
            and (qa.stride(0) * 2) % 16 == 0
            and qa.data_ptr() % 16 == 0
        )

    def run(self, qa: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """[M, 4096] BF16 indexer queries of q_a rows ``qa`` [M, 1536] (any 16 B row pitch),
        on the current stream."""
        import cutlass

        rows = qa.shape[0]
        if not self.supports(qa):
            raise ValueError(f"No indexer query projection for q_a {tuple(qa.shape)}")
        if out is None:
            out = torch.empty((rows, 4096), dtype=torch.bfloat16, device=self.device)
        mma_m = self.tactics[rows][0]
        self.kernels[rows](
            self.weight_view,
            self.projection.matrix_view(qa),
            self.projection.matrix_view(out, output=True),
            cutlass.Int64(self.gamma.data_ptr()),
            cutlass.Int64(self.maps[mma_m].data_ptr()),
            cutlass.Float32(self.eps),
            self.host.stream(self.device),
        )
        return out


# ---- mla_query ----------------------------------------
QUERY_HEADS, RANKS, LOCAL = 96, 16, 6
CAPACITY = 64  # mailbox rows: the largest generation row count of the flat chain
# Rows served by the query kernel; wider steps keep the per-rank 96-head query. GB300, 16
# ranks, query side + FP8 append per DSA layer, max over ranks (us, current -> kernel):
# 1 row 18.1 -> 12.6, 4 rows 18.5 -> 11.1, 8 rows 18.7 -> 13.4, 16 rows 19.2 -> 16.2,
# 24 rows 20.1 -> 16.2, 32 rows 20.2 -> 18.6, 64 rows 23.3 -> 29.9.
MAX_ROWS = 32
# The DSA hidden projection row: q_a 1536 | kv 512 | pe 64 | index k 128 | w 32 [| gate 768]
# (``FlatDsa.hidden_pitch``); the kernel reads its first 2112 columns.
PITCHES = (2272, 3040)
_SENTINEL = 0xFF


#: Stream layout of ``FlatDsa.generation`` with the DSA core. ``r9``: the query side on the
#: caller's stream, IndexFront and the indexer on the indexer stream. ``vllm`` (vLLM's DCP
#: layout, at the rows ``STREAMS_MAX_ROWS`` covers): IndexFront and the indexer on the caller's
#: stream, the query side forked onto the indexer stream after IndexFront, its kernel launched
#: without PDL (so it starts once IndexFront completed). GB300, 16 ranks, 225k context, one
#: request, 4 real DSA layers in one graph, per-layer wall us (rows 8 nsys median; rows 4 / 1
#: event totals over 16 ranks):
#:   rows   r9   vllm after, no PDL   after, PDL   before, PDL   before, no PDL
#:   8     71.1        68.5             69.3          71.4           70.3
#:   4     75.0        70.3             72.0          72.1           71.8
#:   1     70.4        69.6             71.0          70.8           70.7
#: At 8 rows IndexFront runs alone (6.0 -> 4.35 us), the indexer chain is the critical path
#: and the query side ends 11.7 us before it (r9: the core waited 19 us for the indexer).
#: Rows the ``vllm`` layout serves (with the DSA core); wider steps keep ``r9``.
STREAMS_MAX_ROWS = 8


def query_fork(rows: int) -> str | None:
    """Where the ``vllm`` layout forks the query side of ``rows`` rows (``after`` IndexFront),
    or None (the ``r9`` layout)."""
    return "after" if rows <= STREAMS_MAX_ROWS else None


def query_pdl(rows: int) -> bool:
    """Whether the query kernel of ``rows`` rows launches with PDL: under ``r9``; not under
    ``vllm`` (forked after IndexFront, it starts once IndexFront completed)."""
    return query_fork(rows) is None


def query_wait_kernel(rows: int):
    """The gathered-mailbox poll of ``rows`` rows (``kernels/attention/rubin_ull_query_wait``); a
    timed-out poll traps (``kernels/common/rubin_ull_spin_trap``). It releases its dependents only after its polls.
    Speculative decode reads and re-arms the rows without
    ``griddepcontrol.wait`` (``QueryWait`` module doc)."""
    from ..kernels.attention.rubin_ull_query_wait import QueryWait
    from ..kernels.common import rubin_ull_spin_trap as spin_trap

    return QueryWait(rows * QUERY_HEADS * 36, spin_trap.record_address(), release_after_poll=True)


def _tile(rows: int) -> int:
    """Requests per CTA of the tiled kernel (0: MlaQuery's default). 17-32 rows run as two
    16-request tiles (96 CTAs) rather than one 32-request tile (48 CTAs), as K3 runs them
    on Rubin (``query.WIDE_TOKENS``); the grid stays one wave, which the gathered-mailbox
    poll needs (every CTA of every rank must be resident)."""
    return 16 if 17 <= rows <= 32 else 0


class FlatMlaQuery:
    """The q mailbox, the compiled query kernels and every DSA layer's per-rank weight views.

    Construction is collective over ``group`` (the 16 TP/CP ranks) and compiles one kernel
    per row count; run it before graph capture. ``dsa`` maps model layer -> ``FlatDsa``.
    """

    def __init__(
        self,
        group,
        rank: int,
        dsa: AbcMapping,
        rows: Sequence[int],
        device: torch.device,
    ) -> None:
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from ..kernels.vendor.vllm_mr8 import module

        if not 0 <= rank < RANKS or not dsa or len(dsa) % 2:
            # An even count keeps the parity alternating from one step's last DSA layer to
            # the next step's first.
            raise ValueError("The DSA query mailbox needs a rank in 0..15 and an even DSA count")
        rows = tuple(sorted({r for r in rows if 1 <= r <= min(MAX_ROWS, CAPACITY)}))
        if not rows:
            raise ValueError(f"The DSA query kernel needs row counts in 1..{MAX_ROWS}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.device, self.rows = rank, device, rows
        self.group = group
        pitches = {getattr(d, "hidden_pitch", PITCHES[0]) for d in dsa.values()}
        if len(pitches) != 1 or not pitches <= set(PITCHES):
            raise ValueError(f"DSA layers must share one hidden row pitch of {PITCHES}")
        self.pitch = pitches.pop()
        self.host = module("cutedsl.common.host")
        self.query_module = module("cutedsl.mla.query")
        from ..kernels.vendor.vllm_mr8.ops.cute_dsl.latent_moe_tail.primitives import to_cute

        self.to_cute = to_cute
        order = sorted(dsa)
        self.parity = {layer: i % 2 for i, layer in enumerate(order)}
        self.previous = {layer: order[i - 1] for i, layer in enumerate(order) if i}
        self.last = order[-1]

        self.mailbox = symm_mem.empty(
            (2, CAPACITY, QUERY_HEADS, 576), dtype=torch.uint8, device=device
        )
        handle = symm_mem.rendezvous(self.mailbox, group)
        self.multicast = int(handle.multicast_ptr or 0)
        if not self.multicast:
            raise RuntimeError("The DSA query mailbox needs multicast memory")
        self.handle = handle
        self.mailbox.fill_(_SENTINEL)

        local = slice(LOCAL * 192 * rank, LOCAL * 192 * (rank + 1))
        heads = slice(LOCAL * rank, LOCAL * (rank + 1))
        f32 = dict(dtype=torch.float32, device=device)
        self.q_scale_inv = torch.ones(1, **f32)
        self.k_scale_inv = torch.ones(1, **f32)
        # The kernel's KV insert is disabled (every slot -1); its cache operand is never read.
        self.slots = torch.full((CAPACITY,), -1, dtype=torch.int64, device=device)
        self.cache = torch.zeros((1, 64, 576), dtype=torch.uint8, device=device)
        self.local = torch.empty((CAPACITY, LOCAL, 576), dtype=torch.uint8, device=device)
        # BF16 query operands of the append/attention ABI; with a complete FP8 query
        # (``q_rope_applied``) neither kernel reads them.
        self.unused_q = torch.zeros(
            (CAPACITY, QUERY_HEADS, 576), dtype=torch.bfloat16, device=device
        )
        eps = {float(d.weights.q_norm.eps) for d in dsa.values()}
        if len(eps) != 1:
            raise ValueError("DSA layers must share one q_a RMSNorm epsilon")
        import cutlass

        self.eps = cutlass.Float32(eps.pop())
        self.weights = {}
        for layer, owner in dsa.items():
            w = owner.weights
            q_b, k_b = w.query.weight[local], w.k_b[heads]
            if q_b.shape != (LOCAL * 192, 1536) or not (
                q_b.is_contiguous() and k_b.is_contiguous()
            ):
                raise ValueError("DSA q_b rows and W_UK heads must be contiguous per-rank views")
            tensors = (w.q_norm.weight, w.kv_norm.weight, q_b, k_b)
            self.weights[layer] = tensors
        self.views = {
            layer: tuple(self._cute(t) for t in (*tensors, self.q_scale_inv, self.k_scale_inv))
            for layer, tensors in self.weights.items()
        }
        self.cache_view = self._cute(self.cache)
        self.slots_view = self.to_cute(self.slots, 8)
        self.box_views = [self._cute(self.mailbox[p]) for p in (0, 1)]
        self.local_views = {}
        self.kernels = {}
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.sms = sms
        # Publish-only query kernels + QueryWait (module docstring).
        self.waits, self.wait_views = {}, {}
        for count in rows:
            self.kernels[count] = self._compile(count, sms)
            self.local_views[count] = self._cute(self.local[:count])
            self.waits[count] = self._compile_wait(count)
            self.wait_views[count] = [self._cute(self.mailbox[p, :count]) for p in (0, 1)]
        # Poll timeout counters (module docstring); read on the host by ``spin_timeouts``.
        self.spins = torch.zeros(4, dtype=torch.int32, device=device)
        self.spins_view = self.to_cute(self.spins, 4)
        # Diagnostic (features.K31Features.dsa_stats_every; 0 on the served path: nothing registered).
        self._stats_every = k31_features().dsa_stats_every
        self._stats_steps = 0
        if self._stats_every > 0:
            from ..ops.attention.rubin_ull_dsa_metadata import register_step_stats

            register_step_stats(self.maybe_log_stats)
        self.events = (torch.cuda.Event(), torch.cuda.Event())
        self.pending = None  # (layer, event) of a re-arm on the indexer stream
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        # The sparse attention core (``FlatDsaCore``) over this mailbox, for the row counts it
        # serves (``features.K31Features.dsa_core`` "fmha" or an unpacked gate: None, the sparse-MLA path).
        self.core = None
        if self.pitch == PITCHES[1]:
            if core_mode() == "sparse" and any(r <= max_rows_default() for r in rows):
                self.core = FlatDsaCore(
                    mla_query=self,
                    v_b={layer: owner.weights.v_b for layer, owner in dsa.items()},
                    rank=rank,
                    world=RANKS,
                    rows=rows,
                    group=group,
                    device=device,
                )

    def _cute(self, tensor: torch.Tensor):
        return self.to_cute(tensor, 4 if tensor.dtype == torch.float32 else 16)

    def _compile(self, count: int, sms: int):
        """The query kernel of ``count`` rows, publishing into a 16-rank mailbox and
        gathering it (as MR8's ``compile_core`` builds its DCP query)."""
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        host, q = self.host, self.query_module
        any_layer = next(iter(self.weights.values()))
        specs = [*any_layer, self.q_scale_inv, self.k_scale_inv, self.cache]
        qnorm, kvnorm, qb, uk, qinv, kinv, kv = (
            host.fake(
                tuple(t.shape), tuple(t.stride()), t.dtype, 4 if t.dtype == torch.float32 else 16
            )
            for t in specs
        )
        query = host.fake((count, LOCAL, 576), (LOCAL * 576, 576, 1), torch.uint8)
        projected = host.fake((count, 2112), (self.pitch, 1))
        slots = host.fake((count,), (1,), torch.int64, 8)
        mailbox = host.fake((CAPACITY, QUERY_HEADS, 576), None, torch.uint8)
        rubin = sms > q.GB300_SMS
        gate = rubin and count in q.GATED_GATHER.get(RANKS, {})
        kernel = q.MlaQuery(
            LOCAL, count, dcp=RANKS, gate=gate, gather=False, sms=sms,
            tokens=_tile(count),
        )  # fmt: skip
        if not query_pdl(count):
            kernel.options = dict(kernel.options, pdl=False)
        args = (projected, qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv)
        args += (cutlass.Float32(1e-6), query, mailbox, cutlass.Int64(0), cutlass.Int32(0))
        with torch.cuda.device(self.device):
            compiled = cute.compile(kernel, *args, make_fake_stream())
            if count > 1:
                compiled = host.prepare_shared_carveout(compiled, self.device.index, 100)
        return compiled

    def _compile_wait(self, count: int):
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        with torch.cuda.device(self.device):
            view = self._cute(self.mailbox[0, :count])
            spins = self.host.fake((4,), None, torch.int32, 4)
            return cute.compile(query_wait_kernel(count), view, spins, make_fake_stream())

    def wait(self, layer: int, rows: int) -> None:
        """Poll DSA ``layer``'s gathered rows on the current stream (before the sparse attention
        reads them)."""
        p = self.parity[layer]
        dsa_query_wait(
            self.waits[rows],
            self.wait_views[rows][p],
            self.spins_view,
            self.host.stream(self.device),
        )

    def supports(self, hidden: torch.Tensor) -> bool:
        """Whether ``hidden`` (the DSA hidden projection rows) has a compiled kernel."""
        return (
            hidden.shape[0] in self.kernels
            and hidden.dtype == torch.bfloat16
            and hidden.shape[1] == self.pitch
            and hidden.stride() == (self.pitch, 1)
            and hidden.data_ptr() % 16 == 0
            and hidden.device == self.device
        )

    def unused_operands(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """BF16 fused Q [rows,96,576] and q_pe [rows,96,64] for the append/attention ABI."""
        q = self.unused_q[:rows]
        return q, q[..., 512:]

    def fork_prefetch(self, layer: int, rows: int) -> None:
        """Called once per DSA layer and step on the main stream right after the in-projection
        (before :meth:`run`, which may follow on another stream): the point where a model
        runtime forks per-layer side work (the L2 weight prefetch). Nothing here."""

    def run(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """The gathered FP8 query [rows,96,576] (uint8) of DSA ``layer`` from its hidden
        projection rows, on the current stream (complete after :meth:`wait`)."""
        rows = hidden.shape[0]
        if not self.supports(hidden):
            raise ValueError("Unsupported DSA hidden rows for the MLA query kernel")
        main = torch.cuda.current_stream(self.device)
        if self.pending is not None:
            pending, event = self.pending
            if pending != self.previous.get(layer):
                raise RuntimeError("DSA layers must run in model order within a step")
            main.wait_event(event)
            self.pending = None
        p = self.parity[layer]
        dsa_mla_query(self, rows, p, layer, hidden)
        return self.mailbox[p, :rows]

    def spin_timeouts(self) -> int:
        """Mailbox poll timeouts so far over every slot (host read; syncs the device)."""
        return int(self.spins.sum().item())

    def maybe_log_stats(self) -> None:
        """Every ``dsa_stats_every`` host steps (0 = never), log the poll timeouts."""
        if self._stats_every <= 0:
            return
        self._stats_steps += 1
        if self._stats_steps % self._stats_every == 0:
            from tensorrt_llm.logger import logger

            logger.info(
                f"[dsa core] rank {self.rank}: {self._stats_steps} steps, "
                f"{self.spin_timeouts()} spin timeouts"
            )

    def _mc(self, parity: int):
        import cutlass

        return cutlass.Int64(self.multicast + parity * self.mailbox[0].numel())

    def _rank(self):
        import cutlass

        return cutlass.Int32(self.rank)

    def release(self, layer: int, rows: int, side: torch.cuda.Stream | None) -> None:
        """Re-arm this layer's mailbox rows after the sparse attention (current stream) read them.

        On ``side`` (the indexer stream) where a later DSA layer of the step waits for it in
        ``run``; on the current stream for the step's last DSA layer or without a side stream.
        """
        box = self.mailbox[self.parity[layer], :rows]
        if side is None or layer == self.last:
            box.fill_(_SENTINEL)
            self.pending = None
            return
        side.wait_stream(torch.cuda.current_stream(self.device))
        event = self.events[self.parity[layer]]
        with torch.cuda.stream(side):
            box.fill_(_SENTINEL)
            event.record(side)
        self.pending = (layer, event)


# ---- dsa_front ----------------------------------------
HEADS, DIM = 32, 128
_FRONT_BARRIER = 2  # named barrier of the four epilogue warps (1: unused here; 0: the CTA)


def _asm(code, constraints, operands, result=None, *, effects=True, loc=None, ip=None):
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([t.mlir_type for t in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [o.ir_value(loc=loc, ip=ip) for o in operands],
        code,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            t(llvm.extractvalue(t.mlir_type, value, [i], loc=loc, ip=ip))
            for i, t in enumerate(result)
        )
    return result(value)


@dsl_user_op
def _ld_bf16x4(addr, *, loc=None, ip=None):
    """Four BF16 at 8-byte aligned ``addr`` (a tensor no kernel in flight writes) as FP32."""
    return _asm(
        "{ .reg .b32 a, b; ld.global.nc.v2.b32 {a, b}, [$4]; "
        "shl.b32 $0, a, 16; and.b32 $1, a, 0xffff0000; shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,l",
        (Uint64(addr),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_bf16x4_own(addr, *, loc=None, ip=None):
    """Four BF16 this grid stored (coherent load) as FP32."""
    return _asm(
        "{ .reg .b32 a, b; ld.global.v2.b32 {a, b}, [$4]; "
        "shl.b32 $0, a, 16; and.b32 $1, a, 0xffff0000; shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,l",
        (Uint64(addr),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .b16 h; .reg .b32 w; ld.global.nc.b16 h, [$1]; cvt.u32.u16 w, h; shl.b32 $0, w, 16; }",
        "=f,l",
        (Uint64(addr),),
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_max_abs(a, b, c, d, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 x, y, m; .reg .b32 u; abs.f32 x, $1; abs.f32 y, $2; max.f32 m, x, y; "
        "abs.f32 x, $3; abs.f32 y, $4; max.f32 x, x, y; max.f32 m, m, x; "
        "mov.b32 u, m; redux.sync.max.u32 u, u, -1; mov.b32 $0, u; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_sum(x, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 t, s; mov.f32 s, $1; "
        "shfl.sync.bfly.b32 t, s, 16, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 8, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 4, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 2, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 1, 31, -1; add.f32 $0, s, t; }",
        "=f,f",
        (Float32(x),),
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _scale_inv(amax, *, loc=None, ip=None):
    return _asm(
        "{ .reg .f32 a, r; .reg .b32 bits, m, e; .reg .pred p; "
        "max.f32 a, $2, 0f2B8CBCCC; mul.f32 r, a, 0f3B124925; mov.b32 bits, r; "
        "and.b32 m, bits, 0x007FFFFF; and.b32 e, bits, 0x7F800000; setp.ne.u32 p, m, 0; "
        "@p add.u32 e, e, 0x00800000; mov.b32 $0, e; rcp.approx.ftz.f32 $1, $0; }",
        "=f,=f,f",
        (Float32(amax),),
        (Float32, Float32),
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fp8x4(a, b, c, d, inv, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .f32 x, y, z, w; .reg .b16 lo, hi; mul.f32 x, $1, $5; mul.f32 y, $2, $5; "
        "mul.f32 z, $3, $5; mul.f32 w, $4, $5; cvt.rn.satfinite.e4m3x2.f32 lo, y, x; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, w, z; mov.b32 $0, {lo, hi}; }",
        "=r,f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d), Float32(inv)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ln_bf16(x, mean, rstd, w, b, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 t; .reg .b16 h; .reg .b32 u; sub.f32 t, $1, $2; mul.f32 t, t, $3; "
        "fma.rn.f32 t, t, $4, $5; cvt.rn.bf16.f32 h, t; cvt.u32.u16 u, h; shl.b32 u, u, 16; "
        "mov.b32 $0, u; }",
        "=f,f,f,f,f,f",
        (Float32(x), Float32(mean), Float32(rstd), Float32(w), Float32(b)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rstd(var, eps, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 t; add.f32 t, $1, $2; rsqrt.approx.f32 $0, t; }",
        "=f,f,f",
        (Float32(var), Float32(eps)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_u32(addr, value, *, loc=None, ip=None) -> None:
    _asm("st.global.b32 [$0], $1;", "l,r", (Uint64(addr), Uint32(value)), loc=loc, ip=ip)


@dsl_user_op
def _st_u32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.b32 [$1], $2; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_f32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.f32 [$1], $2; }",
        "r,l,f",
        (Uint32(cond), Uint64(addr), Float32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bits(x, *, loc=None, ip=None) -> Uint32:
    return _asm("mov.b32 $0, $1;", "=r,f", (Float32(x),), Uint32, effects=False, loc=loc, ip=ip)


@dsl_user_op
def _cache_offset(flat, inner, s0, *, loc=None, ip=None) -> Uint64:
    return _asm(
        "{ .reg .u64 q, r; div.u64 q, $1, $2; rem.u64 r, $1, $2; mad.lo.u64 $0, q, $3, r; }",
        "=l,l,l,l",
        (Uint64(flat), Uint64(inner), Uint64(s0)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_f32x4(addr, *, loc=None, ip=None):
    return _asm(
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        (Uint64(addr),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mul2(w, scale, ws, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 t; mul.rn.f32 t, $1, $2; mul.rn.f32 $0, t, $3; }",
        "=f,f,f,f",
        (Float32(w), Float32(scale), Float32(ws)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _add4(a, b, c, d, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 s, t; add.rn.f32 s, $1, $2; add.rn.f32 t, $3, $4; add.rn.f32 $0, s, t; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sq4(a, b, c, d, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 x, y, s, t; mul.rn.f32 x, $1, $1; mul.rn.f32 y, $2, $2; add.rn.f32 s, x, y; "
        "mul.rn.f32 x, $3, $3; mul.rn.f32 y, $4, $4; add.rn.f32 t, x, y; add.rn.f32 $0, s, t; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_s64_if(n, addr, *, loc=None, ip=None) -> Uint64:
    return _asm(
        "{ .reg .pred p; setp.ne.u64 p, $1, 0; mov.b64 $0, -1; @p ld.global.nc.b64 $0, [$2]; }",
        "=l,l,l",
        (Uint64(n), Uint64(addr)),
        Uint64,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _is_zero(x, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.eq.u32 p, $1, 0; selp.u32 $0, 1, 0, p; }",
        "=r,r",
        (Uint32(x),),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _write_ok(tok, n_write, slot, slot_s, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p, q, r; setp.lt.s64 p, $1, $2; setp.ge.s64 q, $3, 0; setp.ge.s64 r, $4, 0; "
        "and.pred p, p, q; and.pred p, p, r; selp.u32 $0, 1, 0, p; }",
        "=r,l,l,l,l",
        (Uint64(tok), Uint64(n_write), Uint64(slot), Uint64(slot_s)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


class IndexFrontGemm(NormSplitKGemm):
    """``NormSplitKGemm`` (mma_m 128: one indexer head per CTA tile) + the indexer prologue."""

    def __init__(self, *, tactic, k: int = 1536, full_rows: bool = False):
        super().__init__(tactic=tactic, k=k, full_rows=full_rows)
        if self.cta_m != DIM or self.cta_n > HEADS:
            raise ValueError("IndexFrontGemm needs 128-row weight tiles and at most 32 tokens/tile")

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, gamma: cutlass.Int64,
        a_map: cutlass.Int64, eps: cutlass.Float32, q_in: cutlass.Int64, q_stride: cutlass.Int64, k: cutlass.Int64, k_stride: cutlass.Int64,  # noqa: E501
        w: cutlass.Int64, w_stride: cutlass.Int64, ln_w: cutlass.Int64, ln_b: cutlass.Int64,
        q_out: cutlass.Int64, w_out: cutlass.Int64, cache: cutlass.Int64, slot_fp8: cutlass.Int64,
        slot_scale: cutlass.Int64, n_write: cutlass.Int64, page: cutlass.Int64, s0: cutlass.Int64,
        key_eps: cutlass.Float32, weight_scale: cutlass.Float32,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            a,
            b,
            c,
            gamma,
            a_map,
            eps,
            q_in,
            q_stride,
            k,
            k_stride,
            w,
            w_stride,
            ln_w,
            ln_b,
            q_out,
            w_out,
            cache,
            slot_fp8,
            slot_scale,
            n_write,
            page,
            s0,
            key_eps,
            weight_scale,
        ).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.fit),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, gamma: cutlass.Int64,
        a_map: cutlass.Int64, eps: cutlass.Float32, q_in: cutlass.Int64, q_stride: cutlass.Int64, k: cutlass.Int64, k_stride: cutlass.Int64,  # noqa: E501
        w: cutlass.Int64, w_stride: cutlass.Int64, ln_w: cutlass.Int64, ln_b: cutlass.Int64,
        q_out: cutlass.Int64, w_out: cutlass.Int64, cache: cutlass.Int64, slot_fp8: cutlass.Int64,
        slot_scale: cutlass.Int64, n_write: cutlass.Int64, page: cutlass.Int64, s0: cutlass.Int64,
        key_eps: cutlass.Float32, weight_scale: cutlass.Float32,
    ):  # fmt: skip
        """NormSplitKGemm.kernel, with warp 1 inserting a key and the epilogue quantizing.

        Kernel body after vLLM e3ddbdbd60 cutedsl/gemm/splitk.py SplitKGemm (through rubin_ull_dsa_qproj.py).
        """
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        bar_full = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator
        bar_mma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _splitk._smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _splitk._smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog
        partials, bar_norm = sA, bar_mma_epilog
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _splitk._smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _splitk._smem(cutlass.Int64, 1, 8).iterator
        if cutlass.const_expr(self.exchanges()):
            partials = _splitk._smem(cutlass.Float32, self.split_k * self.cta_n, 16)
            bar_norm = _splitk._smem(cutlass.Int64, 1, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, self.full_arrivals())
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)
                if cutlass.const_expr(self.exchanges()):
                    cute.arch.mbarrier_init(bar_norm, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cutlass.Int32(self.k_tiles)
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            row0 = bidx * self.cta_m
            self.weight_warp(bar_full, bar_empty, sA, a_map, row0, k_tile_start, k_tile_count)
        elif warp_idx == 1:
            # The key of token n_idx * mma_n + bidx (the owner CTA of head bidx).
            rows = cute.size(mB, mode=[0])
            token = n_idx * self.cta_n + bidx
            if split_rank == _splitk.OWNER_RANK:
                if bidx < self.cta_n:
                    if token < rows:
                        self.insert_key(
                            Uint64(Uint32(token)),
                            k,
                            k_stride,
                            ln_w,
                            ln_b,
                            cache,
                            slot_fp8,
                            slot_scale,
                            n_write,
                            page,
                            s0,
                            key_eps,
                        )
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            tid = cute.arch.thread_idx()[0] - 128
            self.norm_rows(mB, gamma, eps, sB, bar_full, partials, bar_norm, split_rank, n_idx, tid)
            tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
            self.epilog_warp(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile, tid,
                mC.element_type, utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce,
                split_rank,
            )  # fmt: skip
            if split_rank == _splitk.OWNER_RANK:
                # The owner stored the head's BF16 rows of its tokens; quantize them.
                cute.arch.barrier(barrier_id=_FRONT_BARRIER, number_of_threads=128)
                rows = cute.size(mB, mode=[0])
                self.quantize_queries(
                    tid,
                    bidx,
                    n_idx,
                    rows,
                    q_in,
                    q_stride,
                    w,
                    w_stride,
                    q_out,
                    w_out,
                    weight_scale,
                )

    @cute.experimental.jit
    def quantize_queries(
        self, tid: cutlass.Int32, head: cutlass.Int32, n_idx: cutlass.Int32,
        rows: cutlass.Int32, q_in: cutlass.Int64, q_stride: cutlass.Int64, w: cutlass.Int64,
        w_stride: cutlass.Int64, q_out: cutlass.Int64, w_out: cutlass.Int64,
        weight_scale: cutlass.Float32,
    ):  # fmt: skip
        """The prologue's query half for head ``head`` of this tile's tokens (warp ``w`` of the
        four: tokens w, w + 4, ..; lane l: columns 4l .. 4l+3)."""
        q_in, q_stride, w, w_stride = Uint64(q_in), Uint64(q_stride), Uint64(w), Uint64(w_stride)
        q_out, w_out = Uint64(q_out), Uint64(w_out)
        lane = Uint32(tid) & Uint32(31)
        warp = Int32(tid) // 32
        col = Uint64(lane * Uint32(4))
        is_lane0 = _is_zero(lane)
        h = Uint64(Uint32(head))
        for r in cutlass.range(warp, self.cta_n, 4, unroll=1):
            token = n_idx * self.cta_n + r
            if token < rows:
                tok = Uint64(Uint32(token))
                x0, x1, x2, x3 = _ld_bf16x4_own(
                    q_in + (tok * q_stride + h * Uint64(DIM) + col) * Uint64(2)
                )
                wh = _ld_bf16(w + (tok * w_stride + h) * Uint64(2))
                scale, inv = _scale_inv(_warp_max_abs(x0, x1, x2, x3))
                q_row = q_out + tok * Uint64(HEADS * DIM)
                _st_u32(q_row + h * Uint64(DIM) + col, _fp8x4(x0, x1, x2, x3, inv))
                w_row = w_out + tok * Uint64(HEADS * 4)
                _st_f32_if(is_lane0, w_row + h * Uint64(4), _mul2(wh, scale, Float32(weight_scale)))

    @cute.experimental.jit
    def insert_key(
        self, tok: Uint64, k: cutlass.Int64, k_stride: cutlass.Int64, ln_w: cutlass.Int64,
        ln_b: cutlass.Int64, cache: cutlass.Int64, slot_fp8: cutlass.Int64,
        slot_scale: cutlass.Int64, n_write: cutlass.Int64, page: cutlass.Int64,
        s0: cutlass.Int64, key_eps: cutlass.Float32,
    ):  # fmt: skip
        """The prologue's key half (its warp 0) for token ``tok``, by this warp."""
        k, k_stride, ln_w, ln_b = Uint64(k), Uint64(k_stride), Uint64(ln_w), Uint64(ln_b)
        cache, slot_fp8, slot_scale = Uint64(cache), Uint64(slot_fp8), Uint64(slot_scale)
        n_write, page, s0 = Uint64(n_write), Uint64(page), Uint64(s0)
        lane = Uint32(cute.arch.thread_idx()[0]) & Uint32(31)
        col = Uint64(lane * Uint32(4))
        is_lane0 = _is_zero(lane)
        cute.arch.griddepcontrol_wait()
        k0, k1, k2, k3 = _ld_bf16x4(k + (tok * k_stride + col) * Uint64(2))
        g0, g1, g2, g3 = _ld_f32x4(ln_w + col * Uint64(4))
        b0, b1, b2, b3 = _ld_f32x4(ln_b + col * Uint64(4))
        slot = _ld_s64_if(n_write, slot_fp8 + tok * Uint64(8))
        slot_s = _ld_s64_if(n_write, slot_scale + tok * Uint64(8))
        mean = _warp_sum(_add4(k0, k1, k2, k3)) * Float32(1.0 / DIM)
        c0, c1, c2, c3 = k0 - mean, k1 - mean, k2 - mean, k3 - mean
        var = _warp_sum(_sq4(c0, c1, c2, c3)) * Float32(1.0 / DIM)
        rstd = _rstd(var, Float32(key_eps))
        v0 = _ln_bf16(k0, mean, rstd, g0, b0)
        v1 = _ln_bf16(k1, mean, rstd, g1, b1)
        v2 = _ln_bf16(k2, mean, rstd, g2, b2)
        v3 = _ln_bf16(k3, mean, rstd, g3, b3)
        scale, inv = _scale_inv(_warp_max_abs(v0, v1, v2, v3))
        codes = _fp8x4(v0, v1, v2, v3, inv)
        write = _write_ok(tok, n_write, slot, slot_s)
        _st_u32_if(write, cache + _cache_offset(slot, page, s0) + col, codes)
        _st_u32_if(write & is_lane0, cache + _cache_offset(slot_s, page, s0), _bits(scale))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

# (mma_m, mma_n, split_k, stages) by the largest row count each serves: 128-row weight tiles (one
# head per CTA) at the indexer query projection's K split. GB300, layer-3 weights over 24 copies,
# CUDA graph, us, the two launches (qproj + CuTe prologue) -> this, each call after a 1-element
# kernel / back to back: rows 1 7.0 -> 5.8 / 6.1 -> 3.9, 4 7.2 -> 5.9 / 6.1 -> 3.9,
# 8 7.2 -> 6.3 / 6.2 -> 4.1, 16 8.0 -> 7.6 / 6.6 -> 5.9; FP8 queries, FP32 head weights and
# cache bytes bitwise the two launches (rows 16 too, against the (64, 16, 3, 4) projection).
# Rows 17-32 keep the two-kernel path: (128, 32, 4, 3) was 2-3 us slower than its cuBLAS query
# rows + prologue and not bitwise.
TACTICS = ((8, (128, 8, 3, 4)), (16, (128, 16, 3, 4)))
# Rubin (``_rubin_rows``): rows 1-8 at K split 4 (128 CTAs, three K tiles each); rows 9-16
# keep TACTICS' row. One Rubin GPU (the indexer rows after the in-projection's warp split-K,
# PDL-chained, cold weights; us from the in-projection's end): 3.81 -> 3.46. Not bitwise TACTICS'
# split 3 (another K split of the indexer queries); full_rows was slower at every split.
# Rows 17-32 take the 16-row tactic over two 16-token tiles (grid y = ceil(rows / 16) x split) in place
# of the three-kernel path (RMSNorm + cuBLAS query rows + CuTe prologue): one Rubin GPU (PDL,
# cold weights, each call behind its producer, 24 DSA layers a step) 24 rows 7.52 -> 6.77 us a call,
# 32 rows 7.49 -> 7.25. Not bitwise the three kernels (another K order of the query rows).
RUBIN_TACTICS_FRONT = ((8, (128, 8, 4, 3)), (16, (128, 16, 3, 4)), (32, (128, 16, 3, 4)))


def front_table(device: torch.device) -> tuple:
    """The device-specific (bound, tactic) table for ``IndexFront``."""
    return RUBIN_TACTICS_FRONT if _rubin_rows(device) else TACTICS


def tactic_for(rows: int, table=TACTICS):
    return next((t for bound, t in table if 1 <= rows <= bound), None)


@_cache
def _compile(tactic: tuple, arch: str):
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = IndexFrontGemm(tactic=_splitk.SplitKTactic(*tactic[:4]))
    args = (fake(1), fake(1), fake(0), Int64(0), Int64(0), Float32(0))
    args += tuple(Int64(0) for _ in range(16)) + (Float32(0), Float32(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


def _f32(x: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", x))[0]


class IndexFront:
    """One DSA layer's indexer front (q_a RMSNorm + query rows + prologue), prepared for ``rows``.

    ``weight``: the [4096, 1536] BF16 indexer row view of the packed query bank; ``norm``: the q_a
    ``NormWeight``; ``index_norm`` / ``index_bias``: the FP32 [128] key LayerNorm; ``index_eps``.
    Construction compiles (cached per tactic) and builds the weight's TMA descriptors; run it
    before graph capture. ``tactics`` overrides the table (tests): {rows: tactic}.
    """

    def __init__(self, weight, norm, index_norm, index_bias, index_eps: float,
                 rows: Sequence[int], tactics=None) -> None:  # fmt: skip
        from ..kernels.vendor.vllm_mr8 import module

        if (
            weight.dtype != torch.bfloat16
            or weight.shape != (HEADS * DIM, 1536)
            or weight.stride() != (1536, 1)
            or weight.data_ptr() % 16
        ):
            raise ValueError("The indexer query rows must be a contiguous BF16 [4096, 1536] view")
        if norm.weight.shape != (1536,) or not norm.weight.is_contiguous():
            raise ValueError("The q_a RMSNorm weight must be a contiguous BF16 [1536] vector")
        for t in (index_norm, index_bias):
            if t.dtype != torch.float32 or t.shape != (DIM,) or not t.is_contiguous():
                raise ValueError("The indexer key LayerNorm needs contiguous FP32 [128] constants")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the indexer front before graph capture")
        self.device = weight.device
        self.weight, self.gamma, self.eps = weight, norm.weight, float(norm.eps)
        self.index_norm, self.index_bias, self.index_eps = index_norm, index_bias, float(index_eps)
        self.host = module("cutedsl.common.host")
        self.projection = module("cutedsl.gemm.projection")
        arch = self.host.architecture(self.device)
        table = TACTICS
        if tactics is None:
            table = front_table(self.device)
            tactics = {r: tactic_for(r, table) for r in rows}
        self.table = table
        self.tactics = {r: tuple(t) for r, t in tactics.items() if t is not None}
        self.weight_view = self.projection.matrix_view(weight)
        self.kernels, self.maps = {}, {}
        with torch.cuda.device(self.device):
            for count, tactic in self.tactics.items():
                self.kernels[count] = _compile(tactic, arch)
                if tactic[0] not in self.maps:
                    self.maps[tactic[0]] = self.projection.weight_map(weight, tactic[0])

    def supports(self, qa: torch.Tensor, ik: torch.Tensor, iw: torch.Tensor) -> bool:
        rows = qa.shape[0]
        return (
            rows in self.kernels
            and qa.shape == (rows, 1536)
            and ik.shape == (rows, DIM)
            and iw.shape == (rows, HEADS)
            and all(t.dtype == torch.bfloat16 and t.device == self.device for t in (qa, ik, iw))
            and all(t.stride(1) == 1 for t in (qa, ik, iw))
            and (qa.stride(0) * 2) % 16 == 0
            and qa.data_ptr() % 16 == 0
            and ik.data_ptr() % 8 == 0
            and (ik.stride(0) * 2) % 8 == 0
        )

    def run(self, qa, ik, iw, weight_scale: float, cache=None, slot_fp8=None, slot_scale=None,
            n_write: int = 0, q_rows=None):  # fmt: skip
        """FP8 queries [M, 32, 128] and FP32 head weights [M, 32] of q_a ``qa``, index keys ``ik``
        and head weights ``iw`` (BF16 rows of the hidden projection); the first ``n_write`` keys
        go to ``cache`` at the int64 slot mappings. ``q_rows``: optional [M, 4096] BF16 buffer
        for the query rows (else allocated)."""
        rows = qa.shape[0]
        if not self.supports(qa, ik, iw):
            raise ValueError(f"No indexer front for q_a {tuple(qa.shape)}")
        if q_rows is None:
            q_rows = torch.empty((rows, HEADS * DIM), dtype=torch.bfloat16, device=self.device)
        q_fp8 = torch.empty((rows, HEADS, DIM), dtype=torch.float8_e4m3fn, device=self.device)
        index_w = torch.empty((rows, HEADS), dtype=torch.float32, device=self.device)
        if cache is not None and n_write > 0:
            c = cache.view(torch.uint8)
            dims, strides = c.shape, c.stride()
            if not (
                c.dim() == 4 and strides[3] == 1 and strides[2] == dims[3]
                and strides[1] == dims[2] * dims[3] and strides[0] % 4 == 0 and dims[3] % 4 == 0
                and slot_fp8.dtype == slot_scale.dtype == torch.int64
            ):  # fmt: skip
                raise ValueError("The indexer K cache must be the pool's contiguous page view")
            cache_args = (c.data_ptr(), slot_fp8.data_ptr(), slot_scale.data_ptr(), int(n_write),
                          dims[1] * dims[2] * dims[3], strides[0])  # fmt: skip
        else:
            cache_args = (0, 0, 0, 0, 1, 0)
        mma_m = self.tactics[rows][0]
        if self.table is not TACTICS and self.tactics[rows] != tactic_for(rows):
            log_rubin_row("C", f"rows={rows} tactic={self.tactics[rows]}")
        self.kernels[rows](
            self.weight_view,
            self.projection.matrix_view(qa),
            self.projection.matrix_view(q_rows, output=True),
            Int64(self.gamma.data_ptr()),
            Int64(self.maps[mma_m].data_ptr()),
            Float32(self.eps),
            Int64(q_rows.data_ptr()),
            Int64(q_rows.stride(0)),
            Int64(ik.data_ptr()),
            Int64(ik.stride(0)),
            Int64(iw.data_ptr()),
            Int64(iw.stride(0)),
            Int64(self.index_norm.data_ptr()),
            Int64(self.index_bias.data_ptr()),
            Int64(q_fp8.data_ptr()),
            Int64(index_w.data_ptr()),
            *(Int64(v) for v in cache_args),
            Float32(_f32(self.index_eps)),
            Float32(_f32(weight_scale)),
            self.host.stream(self.device),
        )
        return q_fp8, index_w


# ---- dsa_index ----------------------------------------
if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )


def gather_topk_scores(scores: torch.Tensor, group: list[int]) -> torch.Tensor:
    """Native MPI/NCCL gather of FP32 [rows,k] scores into [cp,rows,k]."""
    if scores.dtype != torch.float32 or scores.ndim != 2:
        raise ValueError("Helix Top-K gather requires FP32 [rows,k] scores")
    if not group or len(set(group)) != len(group):
        raise ValueError("Helix CP group must be nonempty with unique ranks")
    if len(group) == 1:
        return scores.unsqueeze(0)
    source = scores.contiguous().view(1, -1)
    gathered = torch.ops.trtllm.allgather(source, None, group)
    return gathered.view(len(group), *scores.shape)


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


_TRITON_TOPK_REMAP_MIN_ROWS = 256
_TRITON_TOPK_REMAP_BLOCK = 1024


def triton_convert_req_index_to_global_index(*args, **kwargs):
    """The family's Triton Top-K remap (-1 for a negative index, a page past the table or a
    padding table entry); imported only for its wide-row branch."""
    from ..kernels.attention.rubin_dsa_kernels import (
        triton_convert_req_index_to_global_index as remap,
    )

    return remap(*args, **kwargs)


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

    attn_metadata._ensure_pool_view_cached(layer_idx)
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


def _helix_zero_kv_mask(
    attn_metadata: AttentionMetadata,
    num_tokens: int,
    *,
    seq_start: int = 0,
    num_seqs: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return a per-token bool mask marking tokens with zero local KV on this CP rank.

    These tokens belong to sequences for which this rank owns no KV blocks. Since
    kv_lens_cuda is per-sequence, it is expanded to per-token using the
    seq_lens_cuda query lengths, which stays correct when a sequence spans
    multiple tokens (e.g. speculative decoding). The buffers are static and
    output_size is passed to repeat_interleave, so the mask is CUDA-graph safe.
    Returns None when a length buffer is unavailable.

    Args:
        attn_metadata: Attention metadata holding the length buffers.
        num_tokens: Number of tokens covered by partial_o; used as the static
            output_size of the per-token expansion.
        seq_start: Index of the first sequence covered by partial_o. Defaults to
            0 for generation-only callers; MLA passes num_contexts to select the
            generation slice.
        num_seqs: Number of sequences covered by partial_o. Defaults to all
            sequences after seq_start.
    """
    kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
    seq_lens = getattr(attn_metadata, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    if num_seqs is None:
        num_seqs = attn_metadata.num_seqs - seq_start
    seq_slice = slice(seq_start, seq_start + num_seqs)
    per_seq_mask = kv_lens[seq_slice] == 0
    return torch.repeat_interleave(per_seq_mask, seq_lens[seq_slice], output_size=num_tokens)


def helix_gen_zero_kv_mask(
    attn_metadata: AttentionMetadata, num_tokens: int
) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    Used by the Helix generation combine (``_helix_post_process``) to make those rows a
    no-op. It only depends on the step's metadata, so callers may compute it ahead of the
    attention (see ``MLA.helix_zero_kv_mask_fn``).
    """
    helix_kv_bounds = getattr(attn_metadata, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(attn_metadata, "_helix_spec_tokens_valid", False):
        # Speculative verify groups: KV ownership is per-TOKEN. A rank
        # owning only the tail page of a group has zero visible KV for
        # the group's leading tokens while its per-sequence kv_len is
        # nonzero, so the per-sequence mask below misses those rows.
        # Their decode rows are fully masked with a finite sentinel,
        # making partial_o an average over (possibly uninitialized)
        # pool values; the combine multiplies by corr = 0 and
        # 0 * NaN would poison the token on every CP rank — sanitize
        # by the per-token bound instead.
        return helix_kv_bounds[:num_tokens] == 0
    # MLA processes only the generation token slice, so build the mask
    # from [num_contexts, num_seqs). Skip this expansion when the more
    # precise per-token bounds above are valid.
    return _helix_zero_kv_mask(
        attn_metadata,
        num_tokens,
        seq_start=attn_metadata.num_contexts,
        num_seqs=attn_metadata.num_generations,
    )


# ---- dsa_topk ----------------------------------------
class TopKImplementation(str, Enum):
    """Top-K implementations grouped by backend and algorithm."""

    TORCH = "torch"
    CUDA_RADIX = "cuda_radix"
    CUTE_DSL_RADIX = "cute_dsl_radix"
    CUTE_DSL_GVR = "cute_dsl_gvr"


_GVR_IMPLEMENTATIONS = {
    TopKImplementation.CUTE_DSL_GVR,
}
_MAX_RADIX_BLOCKS_PER_ROW = 10
# One 16-byte vector per copy, matching the Blackwell prefill op's contract.
_CUTE_DSL_PREFILL_COPY_BITS = 128


class FlatDsaTopK:
    """Select Top-K indices for sparse prefill and decode paths.

    GVR decode state is owned by the caller so it can be shared with the
    request metadata and retain a stable address across CUDA Graph replays.
    Native CUDA scratch is supplied by the caller so its lifetime follows the
    caller's eager or CUDA-graph metadata.
    """

    def __init__(
        self,
        top_k: int,
        *,
        prefill_implementation: TopKImplementation | None = None,
        decode_implementation: TopKImplementation | None = None,
        compress_ratio: int = 1,
        gvr_self_sampling: bool = True,
        gvr_deterministic: bool = False,
        gvr_sorting: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.prefill_implementation = TopKImplementation(
            prefill_implementation or TopKImplementation.CUDA_RADIX
        )
        self.decode_implementation = TopKImplementation(
            decode_implementation or TopKImplementation.CUDA_RADIX
        )
        self.compress_ratio = compress_ratio
        # Second-level GVR dispatch for CUTE_DSL_GVR: True selects the
        # hint-free self-sampling engine, False the temporal-hint engine.
        self.gvr_self_sampling = gvr_self_sampling
        # self-sampling GVR only: deterministic selection / order, optional ascending rows
        self.gvr_deterministic = gvr_deterministic
        self.gvr_sorting = gvr_sorting
        # forwarded only when enabled so the default path's calls are unchanged
        self._gvr_det_kwargs = (
            {"deterministic": True, "sorting": gvr_sorting} if gvr_deterministic else {}
        )
        # emission-assisted GVR (opt-in via prepare_gvr_emission): the module
        # owns the closed-loop emission state; only reachable on the temporal
        # (gvr_self_sampling=False) V1 path.
        self._gvr_emission_state = None
        self._gvr_emission_route = None
        self._gvr_emission_armed = False

    @property
    def needs_gvr_prior(self) -> bool:
        """Return whether decode consumes previous-step Top-K indices."""
        return (
            self.decode_implementation == TopKImplementation.CUTE_DSL_GVR
            and not self.gvr_self_sampling
        )

    def forward(
        self,
        scores: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        is_prefill: bool,
        row_starts: torch.Tensor | None = None,
        row_ends: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        scan_lengths: torch.Tensor | None = None,
        row_kv_lens: torch.Tensor | None = None,
        next_n: int = 1,
        max_seq_len: int | None = None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None = None,
        radix_aux_indices: torch.Tensor | None = None,
        radix_aux_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write prefill or decode Top-K indices into ``output_indices``.

        Args:
            scores: Top-K input scores with shape ``[num_rows, num_columns]``.
            output_indices: Int32 output with shape ``[num_rows, top_k]``.
            is_prefill: Whether to run the prefill implementation.
            row_starts: Per-row inclusive starts for prefill.
            row_ends: Per-row exclusive ends for prefill.
            sequence_lengths: Per-request logical KV lengths for decode.
            scan_lengths: Per-request score-column lengths for decode.
            row_kv_lens: Optional int32 logical KV lengths in uncompressed
                token units, with shape ``[num_rows]`` on ``scores.device``,
                for ragged decode. The kernel divides by ``compress_ratio``.
                When present, dispatches the CUDA radix kernel;
                uniform ``next_n`` implementations cannot represent this map.
            next_n: Number of decode rows per request.
            max_seq_len: Maximum decode score width used for GVR kernel tuning.
            gvr_ext_kwargs: GVR-only keyword arguments. ``gvr_prior_indices``
                is required by the temporal GVR path (``CUTE_DSL_GVR`` with
                ``gvr_self_sampling=False``). It is
                caller-owned int32 previous selection with shape
                ``[num_requests, top_k]`` on ``scores.device``. The
                self-sampling engine does not consume this state.
                ``gvr_row_order`` is an optional int32 request ordering with
                shape ``[num_requests]`` on the same device.
            radix_aux_indices: Caller-owned int32 workspace for native CUDA
                Radix split work.
            radix_aux_logits: Caller-owned float32 workspace for native CUDA
                Radix split work.

        Returns:
            ``output_indices`` after the selected implementation writes it.
        """
        if is_prefill:
            assert row_starts is not None and row_ends is not None
            return self._forward_prefill(scores, row_starts, row_ends, output_indices)

        assert sequence_lengths is not None and scan_lengths is not None
        return self._forward_decode(
            scores,
            sequence_lengths,
            scan_lengths,
            row_kv_lens,
            output_indices,
            next_n,
            max_seq_len,
            gvr_ext_kwargs,
            radix_aux_indices,
            radix_aux_logits,
        )

    def _forward_prefill(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefill_implementation == TopKImplementation.TORCH:
            return self._forward_prefill_torch(
                scores,
                row_starts,
                row_ends,
                output_indices,
            )
        if self.prefill_implementation == TopKImplementation.CUTE_DSL_GVR:
            # hint-free k derives from the output width; pin it to the module's k
            assert output_indices.shape[1] == self.top_k
            if not self.gvr_self_sampling:
                # the temporal (hint) GVR engine has no prefill form
                logger.warning_once(
                    "temporal GVR has no prefill engine; using the CUDA radix prefill Top-K.",
                    key="gvr_temporal_prefill_radix",
                )
            elif scores.shape[1] <= self.top_k:
                # every row is short (nv <= k): the exact radix path emits the
                # identity/-1 answer without reading logits — cheaper than a
                # zero-work self-sampling launch. Deliberate, no warning.
                pass
            elif self._selfsampling_prefill_ok(scores):
                from tensorrt_llm._torch.kernels.blackwell.top_k import (
                    selfsampling_topk_prefill_ready,
                    selfsampling_topk_run_prefill,
                )

                if self._prefill_capturing(scores) and not selfsampling_topk_prefill_ready(
                    scores,
                    output_indices,
                    **({"deterministic": True} if self.gvr_deterministic else {}),
                ):
                    # the engine never JIT-compiles under capture; an engine
                    # missed by warmup takes the exact radix path in the graph
                    logger.warning_once(
                        "self-sampling GVR prefill engine is not compiled for this "
                        "shape and cannot JIT under CUDA graph capture; using the "
                        "CUDA radix prefill Top-K.",
                        key="selfsampling_topk_prefill_capture_radix",
                    )
                else:
                    logger.info_once(
                        "self-sampling GVR prefill top-K engaged "
                        f"(K={self.top_k}, cr={self.compress_ratio}, hint-free).",
                        key="selfsampling_topk_prefill_engaged",
                    )
                    # ks/ke are already in compressed column units; run_prefill
                    # writes the local (column - ks) frame with -1 pad and no
                    # host reads (envelope from scores.shape[1]).
                    selfsampling_topk_run_prefill(
                        scores, row_starts, row_ends, output_indices, **self._gvr_det_kwargs
                    )
                    return output_indices
            else:
                # engine hardware-format gate missed (e.g. a non-fp4 layer with
                # an odd DeepGEMM width, or a bf16 producer): exact radix.
                logger.warning_once(
                    "self-sampling GVR prefill is selected but the scores do "
                    "not satisfy the engine's hardware-format gate "
                    f"(dtype={scores.dtype}, strides={tuple(scores.stride())}); "
                    "falling back to the CUDA radix prefill Top-K.",
                    key="selfsampling_topk_prefill_fallthrough",
                )
        elif self.prefill_implementation == TopKImplementation.CUTE_DSL_RADIX:
            # Keep the op's reread policy default; only its copy width is tuned.
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                scores,
                row_starts,
                row_ends,
                output_indices,
                self.top_k,
                _CUTE_DSL_PREFILL_COPY_BITS,
            )
            return output_indices
        elif self.prefill_implementation != TopKImplementation.CUDA_RADIX:
            raise NotImplementedError(
                f"{self.prefill_implementation.value} does not support prefill Top-K"
            )
        torch.ops.trtllm.indexer_topk_prefill(
            scores,
            row_starts,
            row_ends,
            output_indices,
            self.top_k,
        )
        return output_indices

    def _selfsampling_prefill_ok(self, scores: torch.Tensor) -> bool:
        """Engine format gate: fp32 row-major scores with a float4-aligned row
        stride and a 16B base (the DeepGEMM prefill logits arena)."""
        return (
            scores.dtype == torch.float32
            and scores.stride(1) == 1
            and scores.stride(0) % 4 == 0
            and scores.data_ptr() % 16 == 0
        )

    @staticmethod
    def _prefill_capturing(scores: torch.Tensor) -> bool:
        return scores.is_cuda and torch.cuda.is_current_stream_capturing()

    def _forward_decode(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        row_kv_lens: torch.Tensor | None,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None,
        radix_aux_indices: torch.Tensor | None,
        radix_aux_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if row_kv_lens is not None:
            if (
                self.decode_implementation != TopKImplementation.CUDA_RADIX
                and not torch.compiler.is_dynamo_compiling()
            ):
                logger.info_once(
                    "Ragged per-row KV lengths require CUDA radix decode Top-K; "
                    f"using it instead of {self.decode_implementation.value}.",
                    key="ragged_decode_radix_fallback",
                )
            return self._forward_decode_radix(
                scores,
                sequence_lengths,
                scan_lengths,
                output_indices,
                next_n,
                radix_aux_indices,
                radix_aux_logits,
                row_kv_lens=row_kv_lens,
            )

        if self.decode_implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        if self.decode_implementation in _GVR_IMPLEMENTATIONS:
            return self._forward_decode_gvr(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                max_seq_len=max_seq_len,
                radix_aux_indices=radix_aux_indices,
                radix_aux_logits=radix_aux_logits,
                **(gvr_ext_kwargs or {}),
            )

        return self._forward_decode_radix(
            scores,
            sequence_lengths,
            scan_lengths,
            output_indices,
            next_n,
            radix_aux_indices,
            radix_aux_logits,
        )

    def _forward_decode_radix(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        radix_aux_indices: torch.Tensor | None,
        radix_aux_logits: torch.Tensor | None,
        row_kv_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_cute_dsl = (
            row_kv_lens is None
            and self.decode_implementation == TopKImplementation.CUTE_DSL_RADIX
            and not (self.compress_ratio > 1 and next_n > 1)
        )
        if use_cute_dsl:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                scores,
                scan_lengths,
                output_indices,
                self.top_k,
                next_n,
            )
            return output_indices

        radix_indices, radix_values = self._get_radix_workspace(
            scores, radix_aux_indices, radix_aux_logits
        )
        torch.ops.trtllm.indexer_topk_decode(
            scores,
            sequence_lengths,
            output_indices,
            next_n,
            self.top_k,
            compress_ratio=self.compress_ratio,
            radix_aux_indices=radix_indices,
            radix_aux_logits=radix_values,
            row_kv_lens=row_kv_lens,
        )
        return output_indices

    def _get_radix_workspace(
        self,
        scores: torch.Tensor,
        radix_aux_indices: torch.Tensor | None,
        radix_aux_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if scores.dtype != torch.float32:
            # The C++ bf16/fp16 entry has no split-work tier or aux-buffer
            # arguments and rejects widths that would require split work.
            return None, None

        if radix_aux_indices is None or radix_aux_logits is None:
            raise ValueError(
                "Native CUDA Radix TopK requires radix_aux_indices and radix_aux_logits"
            )
        num_rows = scores.shape[0]
        return (
            radix_aux_indices[:num_rows, :_MAX_RADIX_BLOCKS_PER_ROW, : self.top_k],
            radix_aux_logits[:num_rows, :_MAX_RADIX_BLOCKS_PER_ROW, : self.top_k],
        )

    def _forward_decode_gvr(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        radix_aux_indices: torch.Tensor | None,
        radix_aux_logits: torch.Tensor | None,
        gvr_prior_indices: torch.Tensor | None = None,
        gvr_row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decode_implementation == TopKImplementation.CUTE_DSL_GVR and self.gvr_self_sampling:
            assert max_seq_len is not None
            if (
                # engine hardware-format gate (falls through otherwise):
                # fp32 row-major scores with a float4-aligned row stride and
                # a 16B-aligned base (the DSL paged-MQA arena view — column-
                # sliced from a 256-aligned buffer — satisfies this; odd
                # max_seq_len DeepGEMM layouts do not). Single-row batches
                # derive their row window from shape[1] (arena last-row
                # safety), so that width must satisfy the same float4 rule —
                # otherwise run_varlen raises instead of falling through.
                scores.dtype == torch.float32
                and scores.stride(1) == 1
                and scores.stride(0) % 4 == 0
                and scores.data_ptr() % 16 == 0
                and (scores.shape[0] > 1 or scores.shape[1] % 4 == 0)
            ):
                # hint-free k derives from the output width; pin it to the module's k
                assert output_indices.shape[1] == self.top_k
                from tensorrt_llm._torch.kernels.blackwell.top_k import selfsampling_topk_run_varlen

                logger.info_once(
                    "self-sampling GVR top-K engaged "
                    f"(K={self.top_k}, cr={self.compress_ratio}, "
                    f"next_n={next_n}, hint-free).",
                    key="selfsampling_topk_engaged",
                )
                # Self-sampling GVR varlen engine:
                # one launch for the batch; per-row n from device kv_lens,
                # capture-stable tuning from the max-seq-len engine constant
                # (no host reads — CUDA-graph safe). The module receives
                # max_seq_len in compressed index space; run_varlen's value
                # is in KV-token space like sequence_lengths, so multiply it
                # back by the compression ratio.
                selfsampling_topk_run_varlen(
                    scores,
                    sequence_lengths,
                    output_indices,
                    next_n=next_n,
                    compress_ratio=self.compress_ratio,
                    max_seq_len=max_seq_len * self.compress_ratio,
                    **self._gvr_det_kwargs,
                )
                return output_indices
            logger.warning_once(
                "self-sampling GVR is selected but the decode scores do not "
                "satisfy the engine's hardware-format gate "
                f"(dtype={scores.dtype}, strides={tuple(scores.stride())}); "
                "falling back to the CUDA insertion/radix Top-K path.",
                key="selfsampling_topk_fallthrough",
            )
            radix_indices, radix_values = self._get_radix_workspace(
                scores, radix_aux_indices, radix_aux_logits
            )
            torch.ops.trtllm.indexer_topk_decode(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                self.top_k,
                compress_ratio=self.compress_ratio,
                radix_aux_indices=radix_indices,
                radix_aux_logits=radix_values,
            )
            return output_indices

        assert gvr_prior_indices is not None
        assert max_seq_len is not None
        # V1 temporal (DSL). Emission-assisted candidates (opt-in) are only
        # armed on this hint-first path; the self-sampling V2 path above never
        # arms them.
        emission_kwargs: dict = {}
        if self._gvr_emission_armed:
            state = self._gvr_emission_state
            num_rows = scores.shape[0]
            emission_kwargs = state.topk_ext_kwargs(
                self._gvr_emission_route,
                num_rows,
                state.block_max[:num_rows] if state.block_max is not None else None,
            )
            self._gvr_emission_armed = False
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            scores,
            gvr_prior_indices,
            sequence_lengths,
            output_indices,
            self.top_k,
            next_n=next_n,
            compress_ratio=self.compress_ratio,
            max_seq_len=max_seq_len,
            order_row=gvr_row_order,
            **emission_kwargs,
        )
        return output_indices

    def prepare_gvr_emission(
        self,
        batch: int,
        n_comp: int,
        num_sms: int,
        gvr_prior_indices: torch.Tensor,
    ) -> dict:
        """Plan the emission-assisted GVR tier for this decode step.

        Returns the emission kwargs for the paged-MQA scoring op (empty
        when the planner declines this step); the matching consume-side
        kwargs are injected into the next GVR Top-K call automatically.
        Host arithmetic on engine-static shapes plus capturable device
        ops, so a captured graph bakes the tier and replays refresh the
        state buffers in place.

        Args:
            batch: Number of decode requests this step.
            n_comp: Engine-static compressed maximum sequence length.
            num_sms: Device SM count.
            gvr_prior_indices: Caller-owned previous-selection state;
                defines the emission state's row capacity and device.
        """
        # the emission/xstate writes are undeclared mutations (see the op's
        # schema note), so the tier is eager / CUDA-graph only
        if torch.compiler.is_dynamo_compiling():
            return {}
        from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_emission import (
            LIST_EMIT_MIN_N,
            GvrEmissionState,
        )

        if self._gvr_emission_state is None:
            self._gvr_emission_state = GvrEmissionState(
                max_rows=gvr_prior_indices.shape[0],
                top_k=self.top_k,
                device=gvr_prior_indices.device,
                enable_list_tier=n_comp >= LIST_EMIT_MIN_N,
                own_prior=False,
            )
        state = self._gvr_emission_state
        emit_tier, self._gvr_emission_route = state.plan(
            batch, n_comp, num_sms, compress_ratio=max(self.compress_ratio, 1)
        )
        self._gvr_emission_armed = self._gvr_emission_route.tier != "none"
        if emit_tier in ("counts", "list", "rungs"):
            state.update_seed_rows(batch, emit_tier)
        kwargs: dict = {}
        if emit_tier in ("counts", "list"):
            kwargs = state.indexer_emit_kwargs(emit_tier, batch)
        if self._gvr_emission_route.attach_block_max or emit_tier in ("counts", "list"):
            kwargs["block_max_out"] = state.ensure_block_max(n_comp)[:batch]
        return kwargs

    def reset_gvr_emission_rows(self, rows: slice) -> None:
        """Cold-start the emission closed-loop state for reused request
        slots (prefill-to-decode handoff): a zeroed xstate reads as
        invalid and routes those rows to the stock path in-kernel."""
        if self._gvr_emission_state is not None:
            self._gvr_emission_state.xstate[rows].zero_()

    def update_gvr_prior_from_prefill(
        self,
        output_indices: torch.Tensor,
        request_lengths: torch.Tensor,
        gvr_prior_indices: torch.Tensor | None,
        *,
        request_offset: int = 0,
    ) -> None:
        """Update GVR prior indices from each prefill request's last row.

        Args:
            output_indices: Int32 prefill selections with shape
                ``[num_prefill_rows, top_k]``.
            request_lengths: Per-request prefill row counts on
                ``output_indices.device``; a host tensor here makes the row
                gather a synchronous host-to-device copy.
            gvr_prior_indices: Int32 caller-owned state on
                ``output_indices.device`` with shape ``[capacity, top_k]``.
                The slice starting at ``request_offset`` is updated in place.
            request_offset: First request row to update in the prior state.
        """
        if not self.needs_gvr_prior:
            return
        assert gvr_prior_indices is not None
        last_rows = (torch.cumsum(request_lengths, dim=0) - 1).to(dtype=torch.long)
        num_requests = request_lengths.shape[0]
        gvr_prior_indices[request_offset : request_offset + num_requests].copy_(
            output_indices[last_rows]
        )

    def _forward_prefill_torch(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        columns = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        valid = (columns >= row_starts.unsqueeze(1)) & (columns < row_ends.unsqueeze(1))
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected - row_starts.unsqueeze(1)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices

    def _forward_decode_torch(
        self,
        scores: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor:
        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        positions = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        row_indices = torch.arange(scores.shape[0], device=scores.device) // next_n
        next_n_offsets = torch.arange(scores.shape[0], device=scores.device) % next_n
        row_ends = scan_lengths[row_indices] - next_n + next_n_offsets + 1
        valid = positions < row_ends.unsqueeze(1)
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices

    __call__ = forward


# ---- dsa_core ----------------------------------------
DSA_HEADS, TOPK, ROW, LATENT, V_DIM = 96, 2048, 576, 512, 128
GATE_COLUMN, DSA_PITCH = 2272, 3040  # the packed [hidden 2272 | gate 768] projection row
_DSA_LOGGED: set = set()


def core_mode() -> str:
    """``sparse`` (served): this core where it supports the rows; ``fmha``: the sparse-MLA path everywhere
    (``features.K31Features.dsa_core``)."""
    return k31_features().dsa_core


def max_rows_default() -> int:
    """Largest row count served by this core (8 served); wider steps keep the sparse-MLA path
    (``features.K31Features.dsa_core_max_rows``)."""
    return k31_features().dsa_core_max_rows


def check_enabled() -> bool:
    """Diagnostic, off on the served path: before each core launch, check on the device that every
    row's kept count stays within its leading pool rows (``kernels/attention/rubin_ull_dsa_core_check.py``; prints the
    row and fails a device assert). ``features.K31Features.dsa_core_check``."""
    return k31_features().dsa_core_check


def _dsa_f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def softmax_scale_log2(q_scaling: float = 1.0, head_dim: int = 192) -> float:
    """``mla_bmm1_scale[1]`` of ``mla_rope_generation`` (FP32 ops, FP8 q / KV dequant scales 1):
    ``1.f / (q_scaling * sqrtf(head_dim)) * kLog2e``, exactly (each step correctly rounded)."""
    host = _dsa_f32(1.0 / _dsa_f32(_dsa_f32(q_scaling) * _dsa_f32(float(head_dim) ** 0.5)))
    return _dsa_f32(host * _dsa_f32(1.4426950408889634074))


def log_core(core: str, rows: int, rank: int, value: str = "dcp") -> None:
    """``K31_DSA_CORE core=<core> ... rows=<rows>`` once per process and (core, rows, value).
    ``value``: what merges the decode's partials, "dcp" (DcpValue) or "oproj" (the fused value
    o-projection, ``FlatDsaOproj.bind_value``); the line names it for the sparse core, and for
    the speculative core only when it is "oproj"."""
    if (core, rows, value) in _DSA_LOGGED:
        return
    _DSA_LOGGED.add((core, rows, value))
    from tensorrt_llm.logger import logger

    tail = f" value={value}" if core == "sparse" or value != "dcp" else ""
    logger.info(f"K31_DSA_CORE core={core}{tail} rows={rows} rank={rank}")


def _aligned(sizes) -> int:
    """Bytes of consecutive ``(size, alignment)`` allocations from offset 0 (SmemAllocator)."""
    offset = 0
    for size, align in sizes:
        offset = -(-offset // align) * align + size
    return offset


def sparse_smem(tiles: int, bulk: bool) -> int:
    """SparseDcpDecode's dynamic shared memory: its allocations in kernel order (kvs, qs, ps,
    red, inbox, stats, [staged], bar, inbound) with their alignments."""
    ch, pitch, p_pitch, cols, split, chunk, qk_warps = 16 * tiles, 592, 80, 128, 4, 64, 8
    sizes = [
        (chunk * pitch, 128),
        (ch * pitch, 128),
        (ch * p_pitch, 128),
        (2 * qk_warps * ch * 4, 16),
        (split * ch * cols * 4, 16),
        (split * 2 * ch * 4, 16),
    ]
    if bulk:
        sizes.append(((split - 1) * ch * cols * 4, 128))
    return _aligned(sizes + [(64, 8), (64, 8)])


def value_smem(rows_per_cta: int, world: int) -> int:
    """DcpValue's (pushed) dynamic shared memory: the BF16 [16, 544] latent tile and the
    [R or 16, world] FP32 source weights."""
    return _aligned([(16 * 544 * 2, 128), ((rows_per_cta or 16) * world * 4, 16)])


def max_active_clusters(compiled, device_index: int, grid, block: int, smem: int, cluster: int):
    """``cuOccupancyMaxActiveClusters`` of a compiled CuTe kernel at this launch shape (an int),
    or the error text if the query itself fails."""
    try:
        import cuda.bindings.driver as cuda

        from ..kernels.vendor.vllm_main import module

        _, function = module("cutedsl.common.host").resolve_kernel(compiled, device_index)
        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid
        config.blockDimX, config.blockDimY, config.blockDimZ = block, 1, 1
        config.sharedMemBytes = smem
        attr = cuda.CUlaunchAttribute()
        attr.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = cluster, 1, 1
        config.attrs = [attr]
        config.numAttrs = 1
        status, count = cuda.cuOccupancyMaxActiveClusters(function, config)
        if status != cuda.CUresult.CUDA_SUCCESS:
            return f"error:{status}"
        return int(count)
    except Exception as error:  # noqa: BLE001 - reported in the preflight line
        return f"error:{type(error).__name__}:{str(error)[:80]}".replace(" ", "_")


def preflight(kernels, world: int, device: torch.device, rank: int = 0, log: bool = True) -> str:
    """Load-time check of the core's kernels on this device (``kernels``: rows -> FlatDsaCore's
    tuple). Raises (naming kernel, rows, bytes needed vs available, cc, SM count) when a kernel
    needs more dynamic shared memory than ``shared_memory_per_block_optin`` or its 4-CTA cluster
    cannot be resident (``cuOccupancyMaxActiveClusters`` == 0); a failing occupancy query is
    reported in the line, not fatal. Logs and returns ``K31_DSA_CORE_PREFLIGHT ok ...``."""
    props = torch.cuda.get_device_properties(device)
    cc = f"{props.major}{props.minor}"
    sms, smem_limit = props.multi_processor_count, props.shared_memory_per_block_optin
    index = device.index if device.index is not None else torch.cuda.current_device()
    where = f"(sm_{cc}, {sms} SMs, shared_memory_per_block_optin {smem_limit} B)"
    fields = []
    for rows, (decode, value, (tiles, bulk), value_grid, per_cta) in sorted(kernels.items()):
        need = sparse_smem(tiles, bulk)
        if need > smem_limit:
            raise RuntimeError(
                f"K31 DSA core: SparseDcpDecode at {rows} rows (tiles {tiles}, bulk {bulk}) needs "
                f"{need} B of shared memory per CTA, the device allows {smem_limit} B {where}"
            )
        grid = (DSA_HEADS // (16 * tiles) * 4, rows, 1)
        clusters = max_active_clusters(decode, index, grid, 512, need, 4)
        if clusters == 0:
            raise RuntimeError(
                f"K31 DSA core: SparseDcpDecode at {rows} rows cannot launch a 4-CTA cluster of "
                f"512 threads with {need} B of shared memory each {where}"
            )
        vneed = value_smem(per_cta, world)
        if vneed > smem_limit:
            raise RuntimeError(
                f"K31 DSA core: DcpValue at {rows} rows needs {vneed} B of shared memory per CTA, "
                f"the device allows {smem_limit} B {where}"
            )
        fields.append(f"sparse{rows}={need} clusters{rows}={clusters} value{rows}={vneed}")
    line = f"K31_DSA_CORE_PREFLIGHT ok sm={cc} sms={sms} smem_optin={smem_limit} " + " ".join(
        fields
    )
    line += f" query_wait=0 rank={rank}"
    if log:
        from tensorrt_llm.logger import logger

        logger.info(line)
    return line


class FlatDsaCore:
    """SparseDcpDecode (push) + DcpValue (pushed) for the flat DSA layers, shared by every layer.

    ``mla_query``: the ``FlatMlaQuery`` (q mailbox [2, CAP, 96, 576], ``parity``, ``rows``,
    ``spins``/``spins_view``, ``spin_timeouts``, ``maybe_log_stats``, ``group``); ``v_b``: model
    layer -> BF16 [96, 128, 512]; ``rows``: candidate row counts (served: those <= ``max_rows``).
    Construction is collective over ``group`` (the DcpMailbox peer rendezvous) and compiles both
    kernels per served row count: call it outside graph capture.
    """

    def __init__(
        self,
        *,
        mla_query,
        v_b: AbcMapping[int, torch.Tensor],
        rank: int,
        world: int,
        rows: Sequence[int],
        group=None,
        max_rows: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        import torch.distributed as dist

        from ..kernels.vendor.vllm_main import module

        if world <= 0 or DSA_HEADS % world or not 0 <= rank < world:
            raise ValueError("The DSA core needs a rank of a group dividing the 96 heads")
        self.mla_query, self.rank, self.world = mla_query, rank, world
        self.local = DSA_HEADS // world
        self.max_rows = max_rows_default() if max_rows is None else max_rows
        mailbox = mla_query.mailbox
        if (
            mailbox.dtype != torch.uint8
            or mailbox.shape[0] != 2
            or mailbox.shape[2:] != (DSA_HEADS, ROW)
        ):
            raise ValueError("The DSA core reads a [2, CAP, 96, 576] uint8 q mailbox")
        served = sorted({r for r in rows if 1 <= r <= min(self.max_rows, mailbox.shape[1])})
        if not served:
            raise ValueError(f"No row count of {tuple(rows)} is within 1..{self.max_rows}")
        self.rows = tuple(served)
        self.cap = cap = max(served)
        self.device = device = device or mailbox.device
        self.group = group if group is not None else mla_query.group
        self.sparse_module = sparse = module("cutedsl.dsa.attn_sparse")
        self.dcp = dcp = module("cutedsl.mla.dcp")
        self.host = host = module("cutedsl.common.host")
        self.to_cute = host.to_cute
        import cutlass

        self.cutlass = cutlass
        self.scale_log2 = softmax_scale_log2()
        self.out_scale = 1.0  # FlatSparseMla.kv_scale, the bmm2 scale
        # One DcpMailbox ("bf16" partial rows [2, world, cap, local, 512] + LSE): peer stores.
        self.mail = dcp.DcpMailbox(self.group, device, world, cap, self.local, "bf16")
        # Non-push outputs of SparseDcpDecode (its push launch takes them as unused operands).
        self.latent = torch.zeros((cap, DSA_HEADS, LATENT), dtype=torch.bfloat16, device=device)
        self.lse = torch.zeros((cap, DSA_HEADS), dtype=torch.float32, device=device)
        # DcpValue's per-request totals: every row is served (no padded-row zeroing; the sparse-MLA
        # path's Helix tail gates every row as well).
        self.totals = torch.ones(cap, dtype=torch.int32, device=device)
        heads = slice(self.local * rank, self.local * (rank + 1))
        self.w_uv = {}
        for layer, weight in v_b.items():
            if weight.dtype != torch.bfloat16 or weight.shape != (DSA_HEADS, V_DIM, LATENT):
                raise ValueError("The DSA core reads W_UV from BF16 v_b [96, 128, 512]")
            if not weight.is_contiguous():
                raise ValueError("The DSA core reads W_UV in place from a contiguous v_b")
            self.w_uv[layer] = weight.transpose(1, 2)[heads]  # [local, 512, 128], latent stride 1
        uv = next(iter(self.w_uv.values()))
        uv_layout = (tuple(uv.shape), tuple(uv.stride()), uv.dtype)
        self.uv_views = {layer: self.to_cute(w, 16) for layer, w in self.w_uv.items()}
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.sms = sms
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.kernels = {}
        push = (self.local, world)
        from ..kernels.attention import rubin_ull_dsa_spec_decode as spec_decode
        from ..kernels.attention.rubin_ull_dsa_index_exchange import block_pairs
        from ..kernels.common import rubin_ull_spin_trap as spin_trap

        # Prepare speculative DSA decode where the row count and topology are supported.
        self.spec = {}

        with torch.cuda.device(device):
            for r in served:
                tiles, bulk = sparse.sparse_layout(r, DSA_HEADS, push)
                decode = sparse.compile_sparse(r, DSA_HEADS, DSA_HEADS, TOPK, cap, index, push=push)
                value = dcp.compile_value(
                    self.local,
                    world,
                    r,
                    cap,
                    uv_layout,
                    DSA_PITCH,
                    index,
                    sms,
                    1,
                    dsa_heads=DSA_HEADS,
                    dsa_bf16=False,
                    lse_base_e=False,
                    partials="bf16",
                    pushed=True,
                    record=spin_trap.record_address(),
                )
                plan = dcp.DcpValue(
                    self.local, world, r, sms, 1, 36, DSA_HEADS, 1.0, True, "bf16", False, True
                )
                self.kernels[r] = (decode, value, (tiles, bulk), plan.grid, plan.rows_per_cta)
                if spec_decode.spec_supported_rows(r, DSA_HEADS, DSA_HEADS, push, world):
                    self.spec[r] = spec_decode.compile_spec(
                        r, DSA_HEADS, DSA_HEADS, TOPK, cap, index, push, world,
                        spec_decode.CANDIDATES, block_pairs(world, spec_decode.CANDIDATES, TOPK),
                    )  # fmt: skip
        self.q_views = [self.to_cute(mailbox[p, :cap]) for p in (0, 1)]
        self.box_views = [
            (self.to_cute(self.mail.partial[p]), self.to_cute(self.mail.lse[p], 4)) for p in (0, 1)
        ]
        self.peer_views = (self.to_cute(self.mail.peers, 8), self.to_cute(self.mail.lse_peers, 8))
        # Fails loudly here (not mid-step) where this device cannot run the kernels.
        self.preflight = preflight(self.kernels, world, device, rank)
        self.check = None
        if check_enabled():
            from ..kernels.attention.rubin_ull_dsa_core_check import check_core_rows

            self.check = check_core_rows
            with torch.cuda.device(device):  # compiled here, not under graph capture
                empty = torch.full((cap, TOPK), -1, dtype=torch.int32, device=device)
                check_core_rows(
                    empty, torch.zeros(cap, dtype=torch.int32, device=device), 1, rank, -1
                )
        torch.cuda.synchronize(device)
        if self.group is not None and world > 1:
            dist.barrier(group=self.group)

    def supports(self, rows: int) -> bool:
        return rows in self.kernels

    def spec_ready(self, rows: int) -> bool:
        """Whether the speculative decode serves this row count."""
        return rows in self.spec

    def launch_table(self) -> list[dict]:
        """Per served row count: both kernels' grid / block / cluster (the Rubin preflight list)."""
        sparse = self.sparse_module
        table = []
        for r, (_, _, (tiles, bulk), value_grid, per_cta) in sorted(self.kernels.items()):
            heads = sparse.HEADS * tiles
            table.append(
                dict(
                    rows=r,
                    sparse_grid=(DSA_HEADS // heads * sparse.SPLIT, r, 1),
                    sparse_block=sparse.THREADS,
                    sparse_cluster=(sparse.SPLIT, 1, 1),
                    sparse_tiles=tiles,
                    sparse_bulk=bulk,
                    value_grid=value_grid,
                    value_block=512,
                    value_rows_per_cta=per_cta,
                )
            )
        return table

    def run(
        self,
        layer_idx: int,
        rows: int,
        kv_rows_u8: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        gate: torch.Tensor,
        out: torch.Tensor | None = None,
        local: torch.Tensor | None = None,
        spec=None,
        dcp_value: bool = True,
    ) -> torch.Tensor | None:
        """Gated attention rows [rows, local*128] BF16 of DSA ``layer_idx`` (into ``out``).

        ``kv_rows_u8`` [N, 576] uint8 (the FP8 pool's flat rows), ``indices`` int32 [rows, 2048]
        physical rows (kept prefix, -1 tail), ``valid`` int32 [rows] kept counts, ``gate`` BF16
        [rows, local*128] (unit column stride; the packed projection's pitch). Runs on the
        current stream after ``mla_query.wait`` and the indexer's join; re-arms the q mailbox.
        ``local``: the request-local Top-K ``indices`` came from (only read by the
        ``dsa_core_check`` diagnostic's report). ``spec`` (``(merge workspace, published pairs,
        row lengths, KvRows)``): the key-shard merge grid of this step published its thresholds
        and took no epoch ticket, so the speculative decode (``spec_ready``) must run.
        ``dcp_value`` False (``FlatDsaOproj.bind_value``): only the decode runs and nothing is
        returned; the fused value o-projection merges the pushed partials.
        """
        if rows not in self.kernels:
            raise ValueError(f"The DSA core serves rows {self.rows}, not {rows}")
        width = self.local * V_DIM
        if (
            kv_rows_u8.dtype != torch.uint8
            or kv_rows_u8.dim() != 2
            or kv_rows_u8.shape[1] != ROW
            or kv_rows_u8.stride() != (ROW, 1)
            or kv_rows_u8.data_ptr() % 16
        ):
            raise ValueError("The DSA core reads the KV pool as contiguous uint8 [N, 576] rows")
        if (
            indices.dtype != torch.int32
            or indices.shape != (rows, TOPK)
            or not indices.is_contiguous()
        ):
            raise ValueError("The DSA core needs contiguous int32 [rows, 2048] physical indices")
        if valid.dtype != torch.int32 or valid.shape != (rows,) or valid.stride() != (1,):
            raise ValueError("The DSA core needs int32 [rows] kept counts")
        if (
            gate.dtype != torch.bfloat16
            or gate.shape != (rows, width)
            or gate.stride() != (DSA_PITCH, 1)
            or gate.data_ptr() % 16
        ):
            raise ValueError(
                f"The DSA core needs a BF16 [rows, {width}] gate view of pitch {DSA_PITCH}"
            )
        if not dcp_value:
            out = None
        elif out is None:
            out = torch.empty((rows, width), dtype=torch.bfloat16, device=gate.device)
        elif out.dtype != torch.bfloat16 or out.shape != (rows, width) or not out.is_contiguous():
            raise ValueError(f"The DSA core writes contiguous BF16 [rows, {width}]")
        for t in (kv_rows_u8, indices, valid, gate, *(() if out is None else (out,))):
            if t.device != self.device:
                raise ValueError("The DSA core's operands must be on its device")
        tail = "dcp" if dcp_value else "oproj"
        log_core("sparse", rows, self.rank, tail)
        if self.check is not None:
            self.check(indices, valid, kv_rows_u8.shape[0], self.rank, layer_idx, local)
        parity = self.mla_query.parity[layer_idx]
        decode, value = self.kernels[rows][:2]
        stream = self.host.stream(self.device)
        latent, lse = self.latent[:rows], self.lse[:rows]
        if spec is not None:
            if rows not in self.spec:
                raise RuntimeError(
                    "The merge published for a row count without a speculative decode"
                )
            from ..kernels.attention.rubin_ull_dsa_spec_decode import run_spec

            merge, pub, row_lens, remap = spec
            log_core("spec", rows, self.rank, tail)
            run_spec(
                self.spec[rows], self.mla_query.mailbox[parity, : self.cap], kv_rows_u8, indices,
                valid, latent, lse, self.scale_log2, self.out_scale, (self.mail, parity, self.rank),
                merge, pub, row_lens, remap,
            )  # fmt: skip
        else:
            # SparseDcpDecode, pushing into the owners' DcpMailbox (vendored run_sparse's
            # arguments).
            dsa_sparse_dcp_decode(
                self, decode, parity, kv_rows_u8, indices, valid, latent, lse, stream
            )
        if not dcp_value:
            # DsaOprojFused merges the pushed partials; it waits (griddepcontrol.wait) before
            # every read of this layer's buffers (kernels/attention/rubin_ull_dsa_oproj_fused.py).
            return None
        # Early start: the DSA o-projection (FlatDsaOproj's consumer) reads prefix/blocks
        # before its own griddepcontrol.wait -- safe only because DcpValue releases its dependents
        # after its wait, so every kernel back to the layer's AttnRes ring has completed.
        dsa_dcp_value(self, value, parity, layer_idx, valid, gate, out, latent, lse, stream)
        return out

    def spin_timeouts(self) -> int:
        """Mailbox poll timeouts so far (QueryWait + DcpValue; host read, syncs)."""
        return self.mla_query.spin_timeouts()

    def maybe_log_stats(self) -> None:
        """The per-step ``[dsa core] rank r: n steps, z spin timeouts`` line (``FlatMlaQuery``'s,
        which the DSA metadata's ``prepare`` already calls every step)."""
        self.mla_query.maybe_log_stats()


# ---- dsa_attention ----------------------------------------
if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


@dataclass(frozen=True)
class MlaGenerationQuery:
    """Per-step query scratch, shared by append and sparse attention on one stream."""

    fused_q: torch.Tensor
    q_pe: torch.Tensor
    cu_q_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    fmha_scheduler_counter: torch.Tensor
    mla_bmm1_scale: torch.Tensor
    mla_bmm2_scale: torch.Tensor
    quant_q_buffer: torch.Tensor


def identity_rope_table(max_positions: int, device: torch.device) -> torch.Tensor:
    """One NoPE table, matching the baseline's duplicated/interleaved FP32 ABI.

    Allocate once during model setup and share across all 24 attention owners.
    The capacity follows the runtime window; no resizing occurs inside a graph.
    """
    if max_positions <= 0:
        raise ValueError("The identity table needs a positive runtime position ceiling")
    table = torch.empty((1, max_positions * 64 * 2), dtype=torch.float32, device=device)
    flat = table.view(-1)
    flat[0::2] = 1.0
    flat[1::2] = 0.0
    if table.is_cuda:
        torch.cuda.synchronize(table.device)
    return table


#: The trtllm-gen sparse MLA runs one CTA per row when the host max KV length is at most this.
FMHA_SINGLE_CTA_HOST_KV = 256


def single_cta_fmha_view(view):
    """``view`` with host KV lengths clamped to ``FMHA_SINGLE_CTA_HOST_KV``.

    The host lengths are upper bounds the op plans with (``max_past_kv_length``); the fmha reads
    each row's device length, the kept count, whatever the host max. Above 256 it splits a row over
    ceil(max / 256) CTAs and merges their partials in global memory, which at a Helix rank's
    ~128 kept entries per row is all overhead (16 / 24 rows: ~14 us per layer split, ~8 single).
    """
    host = view.host_past_key_value_lengths
    return replace(view, host_past_key_value_lengths=host.clamp(max=FMHA_SINGLE_CTA_HOST_KV))


class FlatSparseMla:
    """No shared MLA/backend instance; only explicit native operations and IO state."""

    def __init__(
        self, *, layer_idx: int, rotary_cos_sin: torch.Tensor, predicted_tokens_per_seq: int = 1
    ) -> None:
        if not 0 <= layer_idx < 93 or predicted_tokens_per_seq <= 0:
            raise ValueError("Sparse MLA needs a target layer and positive token prediction width")
        if (
            rotary_cos_sin.dtype != torch.float32
            or rotary_cos_sin.ndim != 2
            or rotary_cos_sin.shape[0] != 1
            or rotary_cos_sin.numel() == 0
            or rotary_cos_sin.numel() % 128
            or not rotary_cos_sin.is_contiguous()
        ):
            raise ValueError("Sparse MLA needs the native contiguous FP32 identity table")
        self.layer_idx = layer_idx
        self.rotary_cos_sin = rotary_cos_sin
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.num_heads, self.num_kv_heads, self.head_dim = 96, 1, 576
        self.quant_mode = int(QuantMode.FP8_KV_CACHE)
        self.q_scaling = 1.0
        self.q_lora_rank, self.kv_lora_rank = 1536, 512
        self.qk_nope_head_dim, self.qk_rope_head_dim = 128, 64
        # Native attention emits latents; BF16 V up-projection happens later.
        self.v_head_dim, self.rope_append = 512, True
        self.rotary_inv_freq = None
        self.position_embedding_type = int(PositionEmbeddingType.rope_gpt_neox)
        self.rope_dim, self.rope_base = 64, 10000.0
        self.rope_scale_type, self.rope_scale = int(RotaryScalingType.none), 1.0
        self.rope_short_m_scale = self.rope_long_m_scale = 1.0
        self.rope_max_positions = self.rope_original_max_positions = rotary_cos_sin.numel() // 128
        self.is_mla_enable, self.attention_chunk_size = True, 0
        self.skip_correction_threshold, self.uses_spcompress = 0.0, False
        self.local_layer_idx = None
        self.skip_softmax_stat = torch.zeros(2, dtype=torch.uint32, device=rotary_cos_sin.device)
        self.kv_scale = torch.ones(1, dtype=torch.float32, device=rotary_cos_sin.device)

    def get_local_layer_idx(self, metadata) -> int:
        return metadata.kv_cache_manager.layer_offsets[self.layer_idx]

    def append(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent: torch.Tensor,
        metadata,
        quant_q: torch.Tensor | None = None,
    ) -> MlaGenerationQuery:
        """Append native FP8 latent KV and emit quantized Q/scheduler buffers.

        Inputs are BF16 fused Q [rows,96,576], original PE [rows,96,64],
        and normalized latent+PE [rows,576]. The owner retains no weight mirror.
        ``quant_q``: the complete FP8 query [rows,96,576] (uint8), which becomes the
        attention's ``quant_q_buffer``; the kernel then skips its query blocks
        (``q_rope_applied``) and reads neither ``fused_q`` nor ``q_pe``.
        """
        rows = fused_q.shape[0]
        if quant_q is not None and (
            quant_q.dtype != torch.uint8
            or quant_q.shape != (rows, 96, 576)
            or not quant_q.is_contiguous()
            or quant_q.device != fused_q.device
        ):
            raise ValueError("A complete FP8 query must be contiguous uint8 [rows,96,576]")
        for tensor, shape in (
            (fused_q, (rows, 96, 576)),
            (q_pe, (rows, 96, 64)),
            (latent, (rows, 576)),
        ):
            if tensor.dtype != torch.bfloat16 or tensor.shape != shape:
                raise ValueError("Sparse MLA append operands must match native BF16 geometry")
            if tensor.device != self.rotary_cos_sin.device:
                raise ValueError("Sparse MLA operands and identity table must share one device")
        if metadata.num_contexts or rows != metadata.num_tokens:
            raise ValueError("Flat sparse MLA append expects generation-only metadata")
        if metadata.max_seq_len > self.rotary_cos_sin.numel() // 128:
            raise ValueError("Prepare a larger identity table before graph capture")
        if metadata.kv_cache_manager is None:
            raise ValueError("Sparse MLA append requires the FP8 cache manager")
        if metadata.kv_cache_manager.get_kv_cache_dtype(self.layer_idx) != DataType.FP8:
            raise ValueError("Sparse MLA native quant mode must match FP8 latent cache pages")
        n = max(metadata.num_seqs, rows) + 1 if metadata.is_ragged_verify else metadata.num_seqs + 1
        device = fused_q.device
        buffers = MlaGenerationQuery(
            fused_q,
            q_pe,
            torch.empty(n, dtype=torch.int32, device=device),
            torch.empty(n, dtype=torch.int32, device=device),
            torch.empty(1, dtype=torch.uint32, device=device),
            torch.empty(2, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.float32, device=device),
            torch.empty((rows, 96, 576), dtype=torch.uint8, device=device)
            if quant_q is None
            else quant_q,
        )
        helix = [metadata.helix_position_offsets, metadata.helix_is_inactive_rank]
        if metadata._helix_spec_tokens_valid:
            helix.append(metadata.helix_local_slots)
        with metadata.presented_token_major(metadata.token_major_gen_view()):
            self._mla_rope_generation_impl(
                metadata,
                fused_q,
                q_pe,
                latent,
                buffers.cu_q_seqlens,
                buffers.cu_kv_seqlens,
                buffers.fmha_scheduler_counter,
                buffers.mla_bmm1_scale,
                buffers.mla_bmm2_scale,
                buffers.quant_q_buffer,
                None,
                helix,
                None,
                1e-6,
                False,
                False,
                False,
                False,
                None,
                quant_q is not None,
            )
        return buffers

    def attend(
        self, query: MlaGenerationQuery, latent: torch.Tensor, indices: torch.Tensor, metadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Existing token-sparse native MLA; return BF16 latents and FP32 stats.

        Physical pool indices [rows,2048] are from this layer's remap. The
        metadata presentation uses kept counts so -1 padding is never read.
        """
        from tensorrt_llm._torch.attention.backends.interface import (
            AttentionForwardArgs,
            AttentionInputType,
        )
        from tensorrt_llm._torch.attention.backends.sparse.params import SparseRuntimeParams

        rows = query.fused_q.shape[0]
        if indices.dtype != torch.int32 or indices.shape != (rows, 2048):
            raise ValueError("Sparse MLA requires physical int32 [rows,2048] indices")
        if metadata.num_contexts or rows != metadata.num_tokens:
            raise ValueError("Flat sparse MLA attention expects generation-only metadata")
        if indices.device != query.fused_q.device:
            raise ValueError("Sparse MLA indices and queries must share one device")
        self.local_layer_idx = self.get_local_layer_idx(metadata)
        output = torch.empty((rows, 96 * 512), dtype=torch.bfloat16, device=query.fused_q.device)
        stats = torch.empty((rows, 96, 2), dtype=torch.float32, device=query.fused_q.device)
        forward_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            output=output,
            softmax_stats_tensor=stats,
            latent_cache=latent,
            q_pe=query.q_pe,
            cu_q_seqlens=query.cu_q_seqlens,
            cu_kv_seqlens=query.cu_kv_seqlens,
            fmha_scheduler_counter=query.fmha_scheduler_counter,
            mla_bmm1_scale=query.mla_bmm1_scale,
            mla_bmm2_scale=query.mla_bmm2_scale,
            quant_q_buffer=query.quant_q_buffer,
            kv_scale_orig_quant=self.kv_scale,
            kv_scale_quant_orig=self.kv_scale,
            attention_window_size=int(metadata.max_seq_len),
            is_fused_qkv=False,
            update_kv_cache=True,
            sparse_runtime_params=SparseRuntimeParams(
                sparse_attn_indices=indices,
                sparse_attn_indices_block_size=1,
            ),
        )
        view = metadata.token_major_gen_view(token_sparse=True)
        if view is not None:
            view = single_cta_fmha_view(view)
        with metadata.presented_token_major(view):
            self._attention_impl(query.fused_q.view(rows, 96 * 576), metadata, forward_args)
        return output, stats

    def _mla_rope_generation_impl(
        self,
        metadata: TrtllmAttentionMetadata,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        kv_norm_weight: Optional[torch.Tensor],
        kv_norm_eps: float,
        precomputed_cu_seqlens: bool,
        precomputed_fmha_scheduler: bool,
        kv_only: bool,
        kv_done_elsewhere: bool,
        quant_scale_qkv: Optional[torch.Tensor],
        q_rope_applied: bool,
    ) -> None:
        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            self.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,  # sequence_length
            metadata.kv_lens_runtime,  # host_past_key_value_lengths
            metadata.prompt_lens_cpu_runtime,  # host_context_lengths,
            metadata.num_contexts,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            getattr(self, "_nvfp4_mla_kv_scale_orig_quant", None),
            out_scale,
            metadata.block_ids_per_seq,
            helix_tensor_params,
            self.predicted_tokens_per_seq,
            self.get_local_layer_idx(metadata),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            getattr(metadata.kv_cache_manager, "mla_kv_cache_residual_dim", 0),
            metadata.kv_cache_manager.tokens_per_block,
            int(metadata.max_seq_len),  # attention_window_size
            metadata.beam_width,
            self.quant_mode,
            self.q_scaling,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.rope_append,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
            q_rope_applied,
        )

    def _attention_impl(self, q: torch.Tensor, metadata, forward_args) -> None:
        """The existing fallback FMHA native call, without a shared backend instance."""
        attn, k, v = self, None, None
        # The catalog entry's call; it omits two binding arguments this call left at the binding's
        # defaults (sage_attn_smooth_k False, uses_spcompress unset = False).
        thop_attention(
            q=q,
            k=k,
            v=v,
            output=forward_args.output,
            output_sf=forward_args.output_sf,
            workspace_=metadata.effective_workspace,
            # --- Per-step batch state (TrtllmAttentionMetadata) ---
            sequence_length=metadata.kv_lens_cuda_runtime,
            host_past_key_value_lengths=metadata.kv_lens_runtime,
            host_total_kv_lens=metadata.host_total_kv_lens,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            host_context_lengths=metadata.prompt_lens_cpu_runtime,
            host_request_types=metadata.host_request_types_runtime,
            max_context_q_len_override=metadata.max_context_q_len_override,
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            cache_indirection=metadata.cache_indirection,
            block_ids_per_seq=metadata.block_ids_per_seq,
            tokens_per_block=metadata.tokens_per_block,
            max_num_requests=metadata.max_num_requests,
            max_num_sequences=metadata.max_num_sequences,
            beam_width=metadata.effective_beam_width,
            use_paged_context_fmha=metadata.use_paged_context_fmha,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            spec_decoding_generation_lengths=metadata.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=metadata.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=metadata.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=metadata.max_total_draft_tokens,
            force_prepare_spec_dec_tree_mask=metadata.force_prepare_spec_dec_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=metadata.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=metadata.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=metadata.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=metadata.flash_mla_num_splits,
            num_contexts=metadata.num_contexts,
            num_ctx_tokens=metadata.num_ctx_tokens,
            max_context_length=metadata.max_context_length,
            max_seq_len=metadata.max_seq_len,
            trtllm_gen_jit_warmup=metadata.trtllm_gen_jit_warmup,
            is_cross=metadata.is_cross,
            # --- Per-call (AttentionForwardArgs) ---
            out_scale=forward_args.out_scale,
            kv_scale_orig_quant=forward_args.kv_scale_orig_quant,
            kv_scale_quant_orig=forward_args.kv_scale_quant_orig,
            latent_cache=forward_args.latent_cache,
            q_pe=forward_args.q_pe,
            attention_sinks=forward_args.attention_sinks,
            mask_type=forward_args.mask_type,
            attention_input_type=int(forward_args.attention_input_type),
            attention_window_size=forward_args.attention_window_size,
            chunked_prefill_buffer_batch_size=forward_args.chunked_prefill_buffer_batch_size,
            mrope_rotary_cos_sin=forward_args.mrope_rotary_cos_sin,
            mrope_position_deltas=forward_args.mrope_position_deltas,
            softmax_stats_tensor=forward_args.softmax_stats_tensor,
            cu_q_seqlens=forward_args.cu_q_seqlens,
            cu_kv_seqlens=forward_args.cu_kv_seqlens,
            fmha_scheduler_counter=forward_args.fmha_scheduler_counter,
            mla_bmm1_scale=forward_args.mla_bmm1_scale,
            mla_bmm2_scale=forward_args.mla_bmm2_scale,
            quant_q_buffer=forward_args.quant_q_buffer,
            quant_scale_qkv=forward_args.quant_scale_qkv,
            dsv4_inv_rope_cos_sin_cache=forward_args.dsv4_inv_rope_cos_sin_cache,
            enable_dsv4_epilogue_fusion=forward_args.enable_dsv4_epilogue_fusion,
            kv_norm_weight=forward_args.kv_norm_weight,
            kv_norm_eps=forward_args.kv_norm_eps,
            sage_attn_num_elts_per_blk_q=forward_args.sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=forward_args.sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=forward_args.sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=forward_args.sage_attn_qk_int8,
            is_fused_qkv=forward_args.is_fused_qkv,
            update_kv_cache=forward_args.update_kv_cache,
            cross_kv=forward_args.cross_kv,
            relative_attention_bias=forward_args.relative_attention_bias,
            relative_attention_max_distance=forward_args.relative_attention_max_distance,
            # --- Module config (TrtllmAttention) ---
            rotary_inv_freq=attn.rotary_inv_freq,
            rotary_cos_sin=attn.rotary_cos_sin,
            predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
            local_layer_idx=attn.local_layer_idx,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            quant_mode=attn.quant_mode,
            q_scaling=attn.q_scaling,
            position_embedding_type=attn.position_embedding_type,
            rope_dim=attn.rope_dim,
            rope_base=attn.rope_base,
            rope_scale_type=attn.rope_scale_type,
            rope_scale=attn.rope_scale,
            rope_short_m_scale=attn.rope_short_m_scale,
            rope_long_m_scale=attn.rope_long_m_scale,
            rope_max_positions=attn.rope_max_positions,
            rope_original_max_positions=attn.rope_original_max_positions,
            is_mla_enable=attn.is_mla_enable,
            q_lora_rank=attn.q_lora_rank,
            kv_lora_rank=attn.kv_lora_rank,
            qk_nope_head_dim=attn.qk_nope_head_dim,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            v_head_dim=attn.v_head_dim,
            rope_append=attn.rope_append,
            attention_chunk_size=attn.attention_chunk_size,
            skip_softmax_stat=attn.skip_softmax_stat,
            skip_correction_threshold=attn.skip_correction_threshold,
            # --- Sparse runtime parameters ---
            sparse_kv_indices=forward_args.sparse_runtime_params.sparse_kv_indices,
            sparse_kv_offsets=forward_args.sparse_runtime_params.sparse_kv_offsets,
            sparse_attn_indices=forward_args.sparse_runtime_params.sparse_attn_indices,
            sparse_attn_offsets=forward_args.sparse_runtime_params.sparse_attn_offsets,
            sparse_attn_indices_block_size=(
                forward_args.sparse_runtime_params.sparse_attn_indices_block_size
            ),
            sparse_attn_kv_lens=forward_args.sparse_runtime_params.sparse_attn_kv_lens,
            aux_kv_cache_pool_ptr=forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr,
            skip_softmax_threshold_scale_factor_prefill=(
                forward_args.sparse_runtime_params.threshold_scale_factor_prefill
            ),
            skip_softmax_threshold_scale_factor_decode=(
                forward_args.sparse_runtime_params.threshold_scale_factor_decode
            ),
        )


# ---- dsa ----------------------------------------
HIDDEN_ROWS = 2272  # q_a 1536 | kv 512 | pe 64 | index k 128 | index w 32
PACKED_ROWS = HIDDEN_ROWS + 768  # ... | this rank's 6 heads of g_proj rows

# Low-M plans of the packed [hidden | gate] bank, (bound, tactic) as KIMI_K31_DECODE_PLAN.
# GB300, layer-3 weights rotated over 8 copies, CUDA graph, each call after a 1-element kernel,
# us, [2272] + [768] planned GEMMs + sigmoid kernel -> this GEMM (bitwise vs the two GEMMs'
# columns): rows 1 10.6 -> 8.0 (both), 2 12.0 -> 8.4 (both), 3/4 16.5/16.8 -> 10.0 (hidden),
# 8 14.4 -> 10.1 (both), 16 13.4 -> 10.4 (neither: K split 2 for 4; split 4 is 12.6, both),
# 24 15.7 -> 11.1 (hidden), 32 15.7 -> 10.9 (neither: K split 2 for 4; split 4 is 13.8, both).
# Not bitwise means another FP32 summation order of the same BF16 products.
ULL_DECODE_PLAN = {
    (PACKED_ROWS, 7168): (
        (2, ("direct", 128, 2)),
        (8, ("splitk", 128, 8, 4, 6)),
        (16, ("splitk", 64, 16, 2, 11)),
        (32, ("splitk", 64, 32, 2, 9)),
    ),
}


@lru_cache(maxsize=None)
def _sm100_family(device: torch.device) -> bool:
    from tensorrt_llm._utils import is_sm_100f

    major, minor = torch.cuda.get_device_capability(device)
    return is_sm_100f(major * 10 + minor)


def _planned_bf16_projection(x: torch.Tensor, weight: Bf16Weight) -> torch.Tensor | None:
    """The existing K3.1 packed-projection plan, with explicit BF16 kernel calls.

    The plan is data only; no shared Linear or GEMM dispatch module is used.
    Return None outside the plan so the caller preserves its original fallback.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != weight.weight.shape[1]:
        raise ValueError("DSA projections require BF16 [M,K] input matching the native weight")
    w = weight.weight
    choice = None
    if x.is_cuda and _sm100_family(x.device):
        from ..kernels.gemm.rubin_ull_low_m_gemm_plans import KIMI_K31_DECODE_PLAN

        policies = ULL_DECODE_PLAN.get(tuple(w.shape)) or KIMI_K31_DECODE_PLAN.get(
            tuple(w.shape), ()
        )
        choice = next((c for bound, c in policies if x.shape[0] <= bound), None)
    if (
        choice is None
        or choice[0] == "default"
        or x.shape[0] < 1
        or not x.is_contiguous()
        or not w.is_contiguous()
        or x.data_ptr() % 32
        or w.data_ptr() % 32
    ):
        return None
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
    return bf16_low_m_gemm(x, w.detach(), out, choice, get_env_enable_pdl())


def bf16_projection(x: torch.Tensor, weight: Bf16Weight) -> torch.Tensor:
    """Existing low-M plan, otherwise cuBLAS using the same native BF16 bank."""
    result = _planned_bf16_projection(x, weight)
    return bf16_linear(x, weight.weight) if result is None else result


# The vendored MR8 split-K adds evict_first to the weight TMA. It uses the same MMAs,
# K split and reduction order as the FlashInfer tactic, so the outputs are bitwise identical.
DSA_INPROJ_ROWS = range(17, 25)
DSA_INPROJ_TACTIC = (64, 24, 2, 10)


class DsaInProjVendored:
    """The packed DSA in-projection at 17–24 rows on SM100-family devices.

    The caller uses ``bf16_projection`` for unsupported rows. :meth:`prepare` compiles the
    tactic and builds the weight's TMA descriptor;
    call it outside graph capture."""

    def __init__(self, weight: torch.Tensor) -> None:
        if (
            weight.dtype != torch.bfloat16
            or weight.shape != (PACKED_ROWS, 7168)
            or not weight.is_contiguous()
        ):
            raise ValueError("The DSA in-projection needs the packed contiguous BF16 [3040, 7168]")
        self.weight = weight
        self.projection = None
        self.ran: set[int] = set()

    def prepare(self) -> None:
        device = self.weight.device
        if self.projection is not None or device.type != "cuda" or not _sm100_family(device):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the vendored DSA in-projection before graph capture")
        from tensorrt_llm._utils import mpi_rank

        from ..kernels.vendor.vllm_mr8 import module

        projection = module("cutedsl.gemm.projection")
        policies = dict.fromkeys(DSA_INPROJ_ROWS, DSA_INPROJ_TACTIC)
        self.projection = projection.PreparedProjection(self.weight, policies).prepare()
        line = (
            f"K31_DSA_INPROJ_VENDORED feature=dsa_inproj24 rank={mpi_rank()} "
            f"rows={DSA_INPROJ_ROWS[0]}-{DSA_INPROJ_ROWS[-1]} "
            f"tactic={','.join(map(str, DSA_INPROJ_TACTIC))}"
        )
        if line not in _LOGGED:
            _LOGGED.add(line)
            logger.info(line)

    def supports(self, x: torch.Tensor) -> bool:
        return (
            self.projection is not None
            and x.shape[0] in self.projection.kernels
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == 7168
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and x.device == self.weight.device
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        if rows not in self.ran:
            # Host side, once per row count (also when captured): the served path took this engine.
            from tensorrt_llm._utils import mpi_rank

            self.ran.add(rows)
            line = f"K31_DSA_INPROJ_VENDORED_RUN rank={mpi_rank()} rows={rows}"
            if line not in _LOGGED:
                _LOGGED.add(line)
                logger.info(line)
        out = torch.empty((rows, PACKED_ROWS), dtype=x.dtype, device=x.device)
        return self.projection.run(x, out)


def sigmoid_gate_into(x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out = x * gate.sigmoid()`` in one kernel, bitwise the two PyTorch BF16 kernels.

    The K3 output-gate kernel of ``sigmoid_gate_mul``, at any row count and into
    ``out``; all three tensors are contiguous BF16 of one shape on one GPU.
    """
    from ..kernels.attention.rubin_ull_mla_output_gate import _BLOCK, _sigmoid_gate_mul_kernel

    if (
        any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x, gate, out))
        or not x.shape == gate.shape == out.shape
        or not x.is_cuda
    ):
        raise ValueError("The DSA output gate requires contiguous BF16 rows of one shape")
    return dsa_output_gate(_sigmoid_gate_mul_kernel, _BLOCK, x, gate, out)


@dataclass(frozen=True)
class DsaWeights:
    """Single packed BF16 banks; indexer LayerNorm is promoted losslessly to FP32."""

    hidden: Bf16Weight
    query: Bf16Weight
    q_norm: NormWeight
    kv_norm: NormWeight
    index_norm: torch.Tensor
    index_bias: torch.Tensor
    index_eps: float
    k_b: torch.Tensor
    v_b: torch.Tensor
    gate: Bf16Weight
    output: Bf16Weight
    # [hidden | gate] as one contiguous bank whose row views ``hidden`` and ``gate`` are,
    # or None (two separate banks).
    packed: Bf16Weight | None = None

    def __post_init__(self) -> None:
        if self.packed is not None:
            bank = self.packed.weight
            if (
                bank.shape != (PACKED_ROWS, 7168)
                or not bank.is_contiguous()
                or self.hidden.weight.data_ptr() != bank.data_ptr()
                or self.gate.weight.data_ptr() != bank[HIDDEN_ROWS:].data_ptr()
            ):
                raise ValueError("The packed DSA bank must be [hidden | gate] with both as views")
        for name, shape in (
            ("hidden", (2272, 7168)),
            ("query", (22528, 1536)),
            ("gate", (768, 7168)),
            ("output", (7168, 768)),
        ):
            value = getattr(self, name).weight
            if value.shape != shape or not value.is_contiguous():
                raise ValueError(f"Native CP16 DSA {name} requires contiguous {shape}")
        if self.q_norm.weight.shape != (1536,) or self.kv_norm.weight.shape != (512,):
            raise ValueError("DSA normalization widths must match checkpoint latent ranks")
        for value in (self.k_b, self.v_b):
            if (
                value.dtype != torch.bfloat16
                or value.shape != (96, 128, 512)
                or not value.is_contiguous()
            ):
                raise ValueError("Helix V-before-exchange retains all 96 native BF16 K/V heads")
        for value in (self.index_norm, self.index_bias):
            if value.dtype != torch.float32 or value.shape != (128,):
                raise ValueError(
                    "Indexer LayerNorm requires losslessly promoted FP32 [128] constants"
                )
        if not isfinite(self.index_eps) or self.index_eps <= 0:
            raise ValueError("Indexer LayerNorm epsilon must be positive")
        device = self.hidden.weight.device
        tensors = (
            self.query.weight,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.index_norm,
            self.index_bias,
            self.k_b,
            self.v_b,
            self.gate.weight,
            self.output.weight,
        )
        if any(t.device != device for t in tensors):
            raise ValueError("All DSA native weights must share one device")


_INDEXER_STREAMS: dict = {}
_STREAMS_LOGGED: set = set()


def log_streams(fork, rows: int, rank: int) -> None:
    """``K31_DSA_STREAMS layout=<vllm|r9> ... rows=<rows> rank=<rank>`` once per process and
    (layout, fork, rows): the stream layout ``FlatDsa.generation`` runs for these rows."""
    key = (fork, rows)
    if key in _STREAMS_LOGGED:
        return
    _STREAMS_LOGGED.add(key)
    from tensorrt_llm.logger import logger

    if fork is None:
        logger.info(f"K31_DSA_STREAMS layout=r9 rows={rows} rank={rank}")
    else:
        pdl = int(query_pdl(rows))
        logger.info(f"K31_DSA_STREAMS layout=vllm fork={fork} pdl={pdl} rows={rows} rank={rank}")


def indexer_stream(device: torch.device) -> torch.cuda.Stream | None:
    """One side stream per device for the DSA indexer (shared by the 24 layers, which run in order).

    None (everything on the caller's stream) off CUDA.
    """
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _INDEXER_STREAMS.get(index)
    if stream is None:
        stream = _INDEXER_STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


class FlatDsa:
    """Packed projections and indexer cache write, without shared attention modules.

    The attention owner supplies metadata to the existing append, Top-K and
    sparse MLA kernels. This stage adds no weight quantization: only the
    indexer Q/K operands use the existing user-selected FP8 representation.
    """

    def __init__(
        self, weights: DsaWeights, *, layer_idx: int, deterministic_topk: bool = False
    ) -> None:
        if not 0 <= layer_idx < 93:
            raise ValueError("DSA layer index must identify a target layer")
        self.weights, self.layer_idx = weights, layer_idx
        device = weights.hidden.weight.device
        self.indexer_stream = indexer_stream(device)
        # The gate rows in the hidden projection's GEMM and the sigmoid in the Helix tail.
        self.packed_gate = weights.packed is not None
        # The packed in-projection on vLLM main's warp split-K GEMM for the rows it serves
        # (``DsaInProjection``), or None (``bf16_projection``).
        self.in_proj = None
        # The packed ``DsaInProjVendored`` path for eligible rows.
        self.in_proj_vendored = None
        if self.packed_gate:
            w = weights.packed.weight
            self.in_proj = DsaInProjection(w, dsa_inproj_tactics(w.device))
            self.in_proj_vendored = DsaInProjVendored(w)
        # One launch for the indexer keys and queries (:meth:`index_prologue`).
        self.fused_keys = True
        # Key-shard logits + local Top-L + pack as CuTe DSL kernels (``kernels/attention/rubin_ull_dsa_index*``).
        self.cute_indexer = True
        if self.fused_keys:
            # The fused kernel's RoPE operands; the K3.1 indexer is position-free (rope_dim 0),
            # so they are read only for their address. One position row per token.
            self._key_positions = torch.zeros(4096, dtype=torch.int32, device=device)
            self._key_cos_sin = torch.zeros((1, 2), dtype=torch.float32, device=device)
        # The model's ``FlatMlaQuery`` (per-rank query heads, multicast gather), or None
        # for the full 96-head query on every rank (see the ``mla_query`` setter).
        self._mla_query = None
        # With the MLA query: its ``FlatDsaCore`` (the sparse decode + DcpValue
        # in place of the sparse MLA, re-arm, combine mask and gated Helix tail), or None.
        self.dsa_core = None
        # With the MLA query: the indexer's q_a RMSNorm + query rows as one launch, prepared
        # for the query kernel's rows.
        self.index_qproj = None
        # With the MLA query and the CuTe prologue: that launch and the prologue as one
        # (``IndexFrontGemm``).
        self.index_front = None
        self.top_k = FlatDsaTopK(
            2048,
            prefill_implementation=TopKImplementation.CUTE_DSL_GVR,
            decode_implementation=TopKImplementation.CUTE_DSL_GVR,
            gvr_self_sampling=True,
            gvr_deterministic=deterministic_topk,
        )

    @property
    def hidden_pitch(self) -> int:
        """Row pitch of the hidden projection rows ``generation`` produces."""
        return PACKED_ROWS if self.packed_gate else HIDDEN_ROWS

    @property
    def mla_query(self):
        return self._mla_query

    @mla_query.setter
    def mla_query(self, value) -> None:
        """Attach the model's MLA query (before graph capture); prepares the fused indexer
        query projection for its row counts."""
        self._mla_query = value
        self.dsa_core = getattr(value, "core", None)
        self.index_qproj = None
        self.index_front = None
        if value is not None and self.in_proj is not None:
            self.in_proj.prepare(value.rows)
        if value is not None and self.in_proj_vendored is not None:
            self.in_proj_vendored.prepare()
        if value is not None:
            self.index_qproj = IndexerQProjection(
                self.weights.query.weight[INDEXER_ROWS], self.weights.q_norm, value.rows
            )
        if value is not None and self.fused_keys and self.cute_indexer:
            self.index_front = IndexFront(
                self.weights.query.weight[INDEXER_ROWS],
                self.weights.q_norm,
                self.weights.index_norm,
                self.weights.index_bias,
                self.weights.index_eps,
                [r for r in value.rows if self.fused_indexer_rows(r)],
            )

    def front_rows(self, qa: torch.Tensor, ik: torch.Tensor, iw: torch.Tensor) -> bool:
        """Whether :meth:`index_front` serves these rows."""
        front = self.index_front
        return front is not None and front.supports(qa, ik, iw)

    def run_index_front(self, qa, ik, iw, metadata) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`index_query_rows` + :meth:`index_prologue` in one launch (``IndexFrontGemm``):
        the same FP8 queries, FP32 head weights and indexer K cache bytes."""
        manager = metadata.kv_cache_manager
        write = manager is not None and metadata.slot_mapping_fp8 is not None
        if write and manager.quant_block_size != 128:
            raise ValueError("Flat DSA requires FP8 indexer K with 128-element blocks")
        # Early release: IndexFront triggers its dependents before its own wait, so the scorer (and
        # layer 3's torch op) may start before it completes -- safe only while the scorer waits in
        # every CTA (griddepcontrol.wait) before its first read.
        return dsa_index_front(
            self.index_front,
            qa,
            ik,
            iw,
            128**-0.5 * 32**-0.5,
            manager.get_indexer_k_cache_buffers(self.layer_idx) if write else None,
            metadata.slot_mapping_fp8 if write else None,
            metadata.slot_mapping_scale if write else None,
            qa.shape[0] if write else 0,
        )

    def index_query_rows(self, qa: torch.Tensor) -> torch.Tensor:
        """The indexer's [M,4096] query rows from the q_a rows: RMSNorm + projection, fused
        where prepared."""
        fused = self.index_qproj
        if fused is not None and fused.supports(qa):
            return fused.run(qa)
        q_norm = self.weights.q_norm
        normed = flashinfer_rmsnorm(qa, q_norm.weight, q_norm.eps)
        return bf16_linear(normed, self.weights.query.weight[INDEXER_ROWS])

    def index_keys(self, ik: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 indexer keys and scales. The existing indexer is position-free: split its two
        nonempty halves only for the fused FP8 cat/quantize operator's ABI."""
        normed = F.layer_norm(
            ik.float(),
            (128,),
            self.weights.index_norm,
            self.weights.index_bias,
            self.weights.index_eps,
        ).to(ik.dtype)
        ik_pe, ik_nope = normed.split((64, 64), dim=-1)
        return torch.ops.trtllm.fused_cat_fp8(ik_pe, ik_nope, True)

    def index_queries(
        self, iq: torch.Tensor, iw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 indexer queries [rows,32,128] and their FP32 head weights, scaled by the query scales."""
        iq = iq.contiguous().view(-1, 32, 128)
        iq_pe, iq_nope = iq.split((64, 64), dim=-1)
        q_fp8, q_scale = torch.ops.trtllm.fused_cat_fp8(iq_pe, iq_nope, True)
        q_scale = q_scale.view(-1, 32, 1)
        iw = iw * q_scale.squeeze(-1) * (128**-0.5 * 32**-0.5)
        return q_fp8.view(-1, 32, 128), iw

    def project_heads(self, qr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the existing packed-Q plan and its >=128-row split fallback."""
        weight = self.weights.query
        projected = _planned_bf16_projection(qr, weight)
        if projected is None and qr.shape[0] >= 128:
            return bf16_linear(qr, weight.weight[:18432]), bf16_linear(qr, weight.weight[18432:])
        if projected is None:
            projected = bf16_linear(qr, weight.weight)
        q, index_q = projected.split((18432, 4096), dim=-1)
        return q, index_q

    def fused_indexer_rows(self, rows: int) -> bool:
        """Whether :meth:`index_prologue` serves a step of ``rows`` tokens."""
        return self.fused_keys and rows <= self._key_positions.numel()

    def index_prologue(
        self, ik: torch.Tensor, iq: torch.Tensor, iw: torch.Tensor, metadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`index_keys` + :meth:`update_index_cache` + :meth:`index_queries` in one launch.

        GLM-5.2's fused indexer prologue with RoPE off (the K3.1 indexer is position-free):
        FP32 LayerNorm of the BF16 keys and BF16 rounding, ``fused_cat_fp8``'s UE8M0 scale,
        ``rcp.approx.ftz`` reciprocal and E4M3 conversion for keys and queries, the scatter
        kernel's cache layout and slot rule, and ``weights * q_scale * 128^-0.5 * 32^-0.5``
        in the same operation order. Returns FP8 queries [rows,32,128] and FP32 weights.
        """
        from ..kernels.attention.rubin_ull_glm_indexer_prologue import launch_indexer_prologue

        manager = metadata.kv_cache_manager
        write = manager is not None and metadata.slot_mapping_fp8 is not None
        if write and manager.quant_block_size != 128:
            raise ValueError("Flat DSA requires FP8 indexer K with 128-element blocks")
        rows = ik.shape[0]
        if getattr(self, "cute_indexer", False):
            # The same arithmetic as one CuTe DSL kernel (kernels/attention/rubin_ull_dsa_index_prologue.py).
            q_fp8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device=ik.device)
            index_w = torch.empty((rows, 32), dtype=torch.float32, device=ik.device)
            dsa_indexer_prologue(
                indexer_prologue(ik.device),
                ik,
                iq.view(rows, 32, 128),
                iw,
                self.weights.index_norm,
                self.weights.index_bias,
                self.weights.index_eps,
                128**-0.5 * 32**-0.5,
                q_fp8,
                index_w,
                manager.get_indexer_k_cache_buffers(self.layer_idx) if write else None,
                metadata.slot_mapping_fp8 if write else None,
                metadata.slot_mapping_scale if write else None,
                rows if write else 0,
            )
            return q_fp8, index_w
        q_fp8, _, _, index_w, _ = launch_indexer_prologue(
            ik,
            iw,
            iq.view(rows, 32, 128),
            self._key_positions[:rows],
            self._key_cos_sin,
            self.weights.index_norm,
            self.weights.index_bias,
            self.weights.index_eps,
            128**-0.5 * 32**-0.5,
            0,
            True,
            True,
            manager.get_indexer_k_cache_buffers(self.layer_idx) if write else None,
            metadata.slot_mapping_fp8 if write else None,
            metadata.slot_mapping_scale if write else None,
            rows if write else 0,
            False,
            False,
            False,
        )
        return q_fp8, index_w

    def update_index_cache(self, k: torch.Tensor, scale: torch.Tensor, metadata) -> None:
        manager = metadata.kv_cache_manager
        if manager is None or metadata.slot_mapping_fp8 is None:
            return
        if manager.quant_block_size != 128:
            raise ValueError("Flat DSA requires FP8 indexer K with 128-element blocks")
        if scale.element_size() == 1:
            scale = scale.view(torch.int32)
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k,
            scale,
            manager.get_indexer_k_cache_buffers(self.layer_idx),
            metadata.slot_mapping_fp8,
            metadata.slot_mapping_scale,
            k.shape[0],
        )

    def key_shard_ready(self, metadata, rows: int, requests: int) -> bool:
        """Existing FP8 self-sampling key-shard geometry, with 32 indexer heads.

        The selected native path uses compression ratio one and the existing
        CuTe paged logits operator. Wider/ragged steps need the legacy selector.
        """
        next_n = metadata.key_shard_next_n
        return (
            metadata.key_shard_world > 1
            and next_n > 0
            and rows == requests * next_n
            and next_n * 32 <= 256
        )

    def key_shard_decode(
        self,
        metadata,
        q: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        requests: int,
        kv_rows: KvRows | None = None,
    ) -> tuple[bool, torch.Tensor]:
        """Existing FP8 local logits → local Top-K → exact Helix merge.

        Q is FP8 [rows,32,128], weights are FP32 [rows,32] and output is
        int32 [rows,2048]. Metadata owns peer-visible logits and merge buffers;
        no scratch or communication state is retained in the weight container.
        With ``kv_rows`` the LL exchange merge writes ``output`` as KV pool rows where it
        runs. Returns (whether ``output`` holds KV pool rows, the kept count of each row).
        """
        rows = output.shape[0]
        if not self.key_shard_ready(metadata, rows, requests):
            raise ValueError("DSA key-shard requires uniform rows and at most eight rows/request")
        if q.dtype != torch.float8_e4m3fn or q.shape != (rows, 32, 128):
            raise ValueError("DSA key-shard requires native FP8 [rows,32,128] queries")
        if weights.dtype != torch.float32 or weights.shape != (rows, 32):
            raise ValueError("DSA key-shard requires FP32 scaled indexer weights")
        if output.dtype != torch.int32 or output.shape != (rows, 2048):
            raise ValueError("DSA key-shard requires int32 [rows,2048] output")
        next_n = metadata.key_shard_next_n
        local_len = metadata.key_shard_max_local_len
        merge = metadata.key_shard_merge
        block_table, ctx_lens, sched, row_lens = metadata.helix_key_shard_inputs(rows)
        logits = merge.local_logits[:rows]
        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        counts = metadata.helix_sparse_row_lens_out(rows)
        bounds = metadata.helix_spec_row_kv_lens(rows) if counts is not None else None
        mask = (
            torch.empty((rows,), dtype=torch.bool, device=output.device)
            if bounds is not None
            else None
        )
        # Early read: the local_topk merge reads the epoch (merge.state[0]), row_lens and its own
        # LL slots before its griddepcontrol.wait -- safe only because every select CTA releases
        # after its own wait and the previous DSA layer's merge (the epoch's only writer) has
        # completed by then; a stale epoch would settle silently on the other parity block's
        # pairs from two layers back (stale Top-K, no timeout). test_rubin_ull_dsa_key_shard_select's
        # epoch contract.
        runner = select_pack_for(merge) if self.cute_indexer else None
        if runner is not None:
            runner.want_publish = bool(getattr(metadata, "dsa_spec_request", False)) and (
                kv_rows is not None
            )
        launched = self.cute_indexer and key_shard_local_select(
            merge,
            q.view(requests, next_n, 32, 128),
            k_cache,
            weights,
            ctx_lens[:requests],
            block_table,
            sched,
            local_len,
            row_lens,
            rows,
            owned=(output, counts, bounds, mask),
            kv_rows=kv_rows,
        )
        if not launched:
            torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits_into(
                q.view(requests, next_n, 32, 128),
                k_cache,
                weights,
                ctx_lens[:requests],
                block_table,
                sched,
                local_len,
                logits,
                kv_evict_first=True,
            )
            local_rows = metadata.key_shard_local_topk[:rows]
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                logits[:, :local_len],
                row_lens,
                local_rows,
                merge.local_k,
                1,
            )
            merge.pack(local_rows, row_lens)
        if runner is not None:
            if runner.published:
                metadata.dsa_spec_args = (merge, runner.pub, row_lens, kv_rows)
            runner.want_publish = False
        if launched not in (MERGED, DSA_INDEX_ROWS):
            # The CuTe select/pack pushes the exact local Top-L by key (score desc, index asc).
            merge.merge_owned(
                output, counts, bounds=bounds, mask_out=mask, ties_exact=launched == PACKED
            )
        if not self.fused_indexer_rows(rows):
            prologue = "chain"
        elif self.index_front is not None and rows in self.index_front.kernels:
            prologue = "front"
        else:
            prologue = "cute" if self.cute_indexer else "triton"
        log_indexer_paths(
            prologue,
            "cute" if launched else "fallback",
            "cute" if launched in (MERGED, DSA_INDEX_ROWS) else "triton",
            rows,
            merge.world,
            "direct" if launched == DSA_INDEX_ROWS else "remap",
        )
        metadata.mark_helix_sparse_row_lens_set(combine_mask=mask)
        if counts is None:
            counts = merge._counts_scratch(rows)[:rows]
        return launched == DSA_INDEX_ROWS, counts

    def legacy_topk(
        self,
        logits: torch.Tensor,
        metadata,
        output: torch.Tensor,
        next_n: int,
    ) -> None:
        """Existing non-key-shard selector and Helix global-candidate merge.

        The caller supplies logits from the existing FP8 paged-MQA kernel.
        Ragged/Helix-bounded rows select the existing CUDA radix path; uniform
        rows preserve the GVR dispatch and its native exact-radix format fallback.
        """
        rows = output.shape[0]
        requests, contexts = metadata.num_generations, metadata.num_contexts
        lengths = metadata.kv_lens_cuda_runtime[contexts : contexts + requests]
        scan_lengths = metadata.gen_indexer_kv_lens_cuda_runtime
        if scan_lengths is None:
            raise ValueError("DSA decode requires prepared indexer scan lengths")
        helix_bounds = metadata.helix_spec_row_kv_lens(rows)
        row_bounds = helix_bounds if helix_bounds is not None else metadata.ragged_row_kv_lens(rows)
        self.top_k(
            logits,
            output,
            is_prefill=False,
            sequence_lengths=lengths,
            scan_lengths=scan_lengths,
            row_kv_lens=row_bounds,
            next_n=1 if metadata.is_ragged_verify or helix_bounds is not None else next_n,
            max_seq_len=metadata.get_indexer_max_seq_len(),
            radix_aux_indices=metadata.radix_aux_indices,
            radix_aux_logits=metadata.radix_aux_logits,
        )
        if metadata.enable_helix and metadata.mapping.cp_size > 1:
            if helix_bounds is not None:
                output.masked_fill_((helix_bounds == 0).unsqueeze(1), -1)
            mapping = metadata.mapping
            merge_helix_topk(
                logits,
                output,
                mapping.cp_rank,
                lambda scores: gather_topk_scores(scores, list(mapping.cp_group)),
            )
            metadata.set_helix_sparse_row_lens(output)

    def legacy_logits(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        """Existing FP8 paged logits dispatch for uniform or expanded GEN rows.

        The self-sampling profile uses the existing CuTe kernel for unbounded
        uniform rows. Helix-bounded/ragged rows retain DeepGEMM's prepared
        per-row schedules. All inputs use the same FP8 indexer cache bank.
        """
        from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits

        rows, requests = q.shape[0], metadata.num_generations
        if requests <= 0 or q.shape != (rows, 32, 128):
            raise ValueError("DSA GEN logits require queries [rows,32,128] and active requests")
        if q.dtype != torch.float8_e4m3fn or weights.dtype != torch.float32:
            raise ValueError("DSA GEN logits require native FP8 queries and FP32 weights")
        if weights.shape != (rows, 32):
            raise ValueError("DSA scaled indexer weights must cover every query/head")
        next_n = rows // requests
        ragged = metadata.is_ragged_verify
        if ragged:
            if rows != sum(metadata.ragged_verify_lens):
                raise ValueError("DSA ragged rows must match prepared request lengths")
        elif rows != requests * next_n:
            raise ValueError("DSA uniform rows must divide active requests")
        strided = not ragged and (not metadata.use_expanded_buffers_for_mtp or next_n == 1)
        if strided:
            query = q.view(requests, next_n, 32, 128)
            lengths = metadata.kv_lens_cuda_2d[:requests, :next_n].contiguous()
            tables = metadata.indexer_k_cache_block_offsets[
                metadata.num_contexts : metadata.num_contexts + requests
            ]
            schedule = (
                metadata.scheduler_metadata_buffer
                if next_n == 1
                else metadata.scheduler_metadata_buffer_full_next_n
            )
        else:
            query = q.view(rows, 1, 32, 128)
            lengths = metadata.kv_lens_expanded_cuda[:rows].view(-1, 1)
            tables = metadata.block_table_expanded[:rows]
            schedule = metadata.scheduler_metadata_buffer_expanded
        cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        maximum = metadata.get_indexer_max_seq_len()
        if ragged or metadata.helix_spec_row_kv_lens(rows) is not None:
            return fp8_paged_mqa_logits(query, cache, weights, lengths, tables, schedule, maximum)
        lengths = metadata.gen_indexer_kv_lens_cuda_runtime
        if lengths is None:
            raise ValueError("DSA CuTe logits require prepared indexer lengths")
        single_atom = 1 < next_n <= 8
        atom_split = (
            metadata.dsl_expand_factor > 1
            and next_n == metadata.dsl_expand_factor * metadata.dsl_atom
            and not single_atom
        )
        kernel_split = (
            metadata.dsl_kernel_atoms > 1
            and next_n == metadata.dsl_kernel_next_n
            and not single_atom
        )
        extra = {}
        if kernel_split:
            extra = dict(num_next_n_atoms=metadata.dsl_kernel_atoms, atom=metadata.dsl_atom)
        schedule = metadata.scheduler_metadata_buffer
        if atom_split:
            batch = requests * metadata.dsl_expand_factor
            query = query.reshape(batch, metadata.dsl_atom, 32, 128)
            lengths = metadata.kv_lens_expanded_cuda[:batch]
            tables = metadata.block_table_expanded[:batch]
            schedule = metadata.scheduler_metadata_buffer_expanded
        elif kernel_split:
            schedule = metadata.scheduler_metadata_buffer_expanded
        if single_atom:
            extra["kv_evict_first"] = True
        return torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            query,
            cache,
            weights,
            lengths,
            tables,
            schedule,
            maximum,
            **extra,
        )

    def decode_indexer(
        self, index_q: torch.Tensor, index_w: torch.Tensor, metadata
    ) -> torch.Tensor:
        """Generation-only Top-K using metadata-owned graph-stable output buffers. The caller
        has written this step's indexer keys (:meth:`update_index_cache`)."""
        rows = index_q.shape[0]
        requests = metadata.num_generations
        if metadata.num_contexts or rows != metadata.num_tokens or requests <= 0:
            raise ValueError("Flat DSA indexer expects a generation-only metadata view")
        output = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (rows, 2048),
            cache_name="indexer_topk_out_buffer_gen",
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )
        self.select_rows(index_q, index_w, metadata, output)
        return output

    def select_rows(self, index_q, index_w, metadata, output, kv_rows=None):
        """:meth:`decode_indexer` into ``output``; with ``kv_rows`` as KV pool rows where the
        key-shard exchange runs. Returns (whether ``output`` holds KV pool rows, the key-shard
        kept counts or None)."""
        rows, requests = index_q.shape[0], metadata.num_generations
        if metadata.skip_indexer_for_gen_reqs:
            output.copy_(metadata.topk_indices_buffer[:rows])
        elif self.key_shard_ready(metadata, rows, requests):
            return self.key_shard_decode(
                metadata, index_q, index_w, output, requests, kv_rows=kv_rows
            )
        else:
            logits = self.legacy_logits(index_q, index_w, metadata)
            self.legacy_topk(logits, metadata, output, rows // requests)
        return False, None

    def kv_rows(self, metadata) -> KvRows | None:
        """This layer's remap arguments (``remap_topk``), or None where the kernels' 64-token
        page arithmetic does not apply."""
        layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        metadata._ensure_pool_view_cached(layer)
        if metadata._cached_tokens_per_block != PAGE_TOKENS:
            return None
        page_scale, layer_offset = metadata.kv_cache_manager.get_primary_pool_page_index_params(
            layer
        )
        return KvRows(
            metadata._cached_block_table_gen,
            metadata._cached_req_idx_gen,
            page_scale * PAGE_TOKENS,
            layer_offset * PAGE_TOKENS,
        )

    def indexer_rows(
        self, index_q: torch.Tensor, index_w: torch.Tensor, metadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The generation rows' sparse attention operands: (rows int32 [rows, 2048], valid int32
        [rows]). ``rows[r, :valid[r]]`` are KV pool rows of ``pool.view(-1, 576)`` (the pool
        view of ``transform_local_topk_and_prepare_pool_view``), bitwise :meth:`remap_topk` of
        :meth:`decode_indexer`, ``-1`` after them. The caller has written this step's indexer
        keys (:meth:`update_index_cache`).

        Where the key-shard LL exchange runs, its merge writes the KV pool rows itself; the other
        paths remap their local indices. ``valid``: the key-shard kept counts (the Helix sparse
        row lengths, ``metadata.helix_sparse_row_lens_out``, with the combine mask set as
        :meth:`key_shard_decode` does), else the count of non-negative entries. Everything runs
        on the current stream."""
        rows = index_q.shape[0]
        requests = metadata.num_generations
        if metadata.num_contexts or rows != metadata.num_tokens or requests <= 0:
            raise ValueError("Flat DSA indexer expects a generation-only metadata view")
        kv_rows = None
        if (
            self.cute_indexer
            and not metadata.skip_indexer_for_gen_reqs
            and self.key_shard_ready(metadata, rows, requests)
        ):
            kv_rows = self.kv_rows(metadata)
        if kv_rows is not None:
            # One buffer per layer: a layer's rows stay valid while a later layer's indexer runs.
            output = metadata.get_empty(
                metadata.cuda_graph_buffers,
                (rows, 2048),
                cache_name=f"k31_flat_indexer_rows_gen_{self.layer_idx}",
                dtype=torch.int32,
                capture_graph=metadata.is_cuda_graph,
            )
        else:
            output = metadata.get_empty(
                metadata.cuda_graph_buffers,
                (rows, 2048),
                cache_name="indexer_topk_out_buffer_gen",
                dtype=torch.int32,
                capture_graph=metadata.is_cuda_graph,
            )
        direct, counts = self.select_rows(index_q, index_w, metadata, output, kv_rows)
        rows_out = output if direct else self.remap_topk(output, metadata)
        valid = counts
        if valid is None:
            # Legacy Helix merges set the row lengths (set_helix_sparse_row_lens); else count.
            valid = metadata.helix_sparse_row_lens_out(rows)
            if valid is None or not getattr(metadata, "_helix_rows_set", False):
                valid = (rows_out >= 0).sum(dim=1, dtype=torch.int32)
        return rows_out, valid

    def remap_topk(self, local: torch.Tensor, metadata) -> torch.Tensor:
        """Existing per-layer local-token → physical-pool mapping for FP8 KV.

        Every K3.1 DSA layer is a full indexer layer (frequency one), so grouped
        leader/shared-layer fan-out is inert. Preserve the native per-layer
        C++/Triton dispatch and metadata's pool/row ownership.
        """
        return self.remap_topk_rows(local, metadata)[0]

    def remap_topk_rows(self, local: torch.Tensor, metadata) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`remap_topk`'s physical indices and the FP8 pool they index as uint8 [N,576]
        rows (the pool view of the same remap: row = block * page_scale * 64 + layer_offset * 64
        + token % 64, at byte row * 576)."""
        layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        indices, pool = transform_local_topk_and_prepare_pool_view(local, metadata, layer, True)
        return indices, pool.view(-1, pool.shape[-1]).view(torch.uint8)

    def project_query(self, q: torch.Tensor) -> torch.Tensor:
        """Absorb K into the query's NoPE half: fused Q [rows,96,576] with columns 512: unset.

        ``q`` [rows,18432] may be a row-strided view. The FP8 append reads only the NoPE
        columns of the fused Q and takes the PE half from ``q_pe`` (its own strides).
        """
        q = q.view(-1, 96, 192)
        out = torch.empty((q.shape[0], 96, 576), dtype=q.dtype, device=q.device)
        bmm_out(q[..., :128].transpose(0, 1), self.weights.k_b, out[..., :512].transpose(0, 1))
        return out

    def project_v(self, partial: torch.Tensor) -> torch.Tensor:
        """The existing BF16 V projection before the CP exchange."""
        partial = partial.view(-1, 96, 512)
        out = torch.empty((partial.shape[0], 96, 128), dtype=partial.dtype, device=partial.device)
        bmm_out(partial.transpose(0, 1), self.weights.v_b.transpose(1, 2), out.transpose(0, 1))
        return out.reshape(partial.shape[0], -1)

    def project_output(
        self, x: torch.Tensor, combined: torch.Tensor, gate=None, gated: bool = False
    ) -> torch.Tensor:
        """Native BF16 gate/output partial; the caller owns the TP16 reduction.

        ``gate``: the gate projection rows (else projected from ``x``); ``gated``: ``combined``
        is already gated."""
        if not gated:
            if gate is None:
                gate = bf16_projection(x, self.weights.gate)
            gated_rows = sigmoid_gate_mul(combined, gate)
            if gated_rows is None:
                gated_rows = combined * gate.sigmoid()
            combined = gated_rows
        return bf16_linear(combined, self.weights.output.weight)

    def gate_rows(
        self, x: torch.Tensor, combined: torch.Tensor, out: torch.Tensor, gate=None
    ) -> torch.Tensor:
        """``combined * sigmoid(gate(x))`` into ``out``, bitwise ``project_output``'s gating
        (``gate``: the gate projection rows, else projected from ``x``)."""
        gate = bf16_projection(x, self.weights.gate) if gate is None else gate.contiguous()
        return sigmoid_gate_into(combined, gate, out)

    def exchange(
        self, partial: torch.Tensor, stats: torch.Tensor, mask, helix, gate=None, out=None
    ) -> tuple[torch.Tensor, bool]:
        """V projection, Helix exchange and combine of the latent partial [M,96*512].

        The fused K3 Helix tail where the transport provides it; otherwise the BF16
        BMM before the exchange and the transport's combine. With ``gate`` the fused tail
        also applies the sigmoid gate (into ``out`` if given). Returns the rows and whether
        they are gated.
        """
        if gate is not None:
            gated = helix.fused_tail(partial, stats, self.weights.v_b, mask, gate=gate, out=out)
            if gated is not None:
                return gated, True
        combined = helix.fused_tail(partial, stats, self.weights.v_b, mask)
        if combined is None:
            combined = helix(self.project_v(partial), stats, mask)
        return combined, False

    def _query_side(self, mq, projected, kv, pe, metadata, mla, rows):
        """The MLA query side on the current stream: the query kernel's publish, the kv RMSNorm
        + latent row, the FP8 append and the poll of the peers' query heads. Returns the
        append's query and the latent."""
        quant_q = mq.run(self.layer_idx, projected)
        kv = flashinfer_rmsnorm(kv, self.weights.kv_norm.weight, self.weights.kv_norm.eps)
        latent = torch.cat((kv, pe), dim=-1)
        fused_q, q_pe = mq.unused_operands(rows)
        query = mla.append(fused_q, q_pe, latent, metadata, quant_q=quant_q)
        # The peers' query heads, polled after the append (publish-only query kernel).
        mq.wait(self.layer_idx, rows)
        return query, latent

    def _indexer(self, index_q, index_w, metadata, core):
        """The indexer's Top-K on the current stream as physical indices (:meth:`indexer_rows`:
        written by the key-shard exchange itself where it runs, else remapped), for ``core`` the
        FP8 pool rows they index (else None), and None for the request-local Top-K. With
        ``core`` serving the speculative decode at these rows, the key-shard merge is asked to
        publish (``metadata.dsa_spec_request``); ``metadata.dsa_spec_args`` holds what it published
        (``FlatDsaCore.run``'s ``spec``) or None."""
        metadata.dsa_spec_request = core is not None and core.spec_ready(index_q.shape[0])
        metadata.dsa_spec_args = None
        indices, _ = self.indexer_rows(index_q, index_w, metadata)
        metadata.dsa_spec_request = False
        if core is not None:
            pool = metadata._cached_pool_view
            return indices, pool.view(-1, pool.shape[-1]).view(torch.uint8), None
        return indices, None, None

    def _core(
        self, x, core, rows, kv_rows, indices, valid, gate, output, reduce, local=None, spec=None
    ):
        """SparseDcpDecode (re-arms the q mailbox) + DcpValue: gated rows for the o-projection
        (``local``: the Top-K the indices came from, for the ``dsa_core_check`` diagnostic).
        Where ``output`` fuses the value merge at these rows (``value_rows``, ``bind_value``):
        the decode, then the fused value o-projection."""
        value_rows = getattr(output, "value_rows", None)
        if value_rows is not None and value_rows(rows):
            core.run(
                self.layer_idx, rows, kv_rows, indices, valid, gate, None, local=local, spec=spec,
                dcp_value=False,
            )  # fmt: skip
            return output(x, None, gate=gate, value=True)
        rows_out = getattr(output, "rows", None)
        target = rows_out(rows) if rows_out is not None else None
        gated = core.run(
            self.layer_idx, rows, kv_rows, indices, valid, gate, target, local=local, spec=spec
        )
        if output is not None:
            return output(x, gated, gate=gate, gated=True)
        return reduce(self.project_output(x, gated, gate=gate, gated=True))

    def generation(
        self, x: torch.Tensor, metadata, *, mla, helix, reduce, output=None
    ) -> torch.Tensor:
        """Explicit DSA GEN layer using one native weight set and existing kernels.

        ``mla`` owns the private native attention IO; ``helix`` and ``reduce``
        own their native collective workspaces. Model/cache adapters prepare the
        metadata before entering this same path for eager or graph execution.
        ``output(x, combined)``, if given, replaces the gated o-projection partial
        and its ``reduce`` (the fused o-projection + all-reduce + AttnRes).
        """
        if self.packed_gate:
            # One GEMM for the hidden rows and the gate; the tail applies the gate.
            if self.in_proj is not None and self.in_proj.ready(x):
                projected = self.in_proj(x)
            elif self.in_proj_vendored is not None and self.in_proj_vendored.supports(x):
                projected = self.in_proj_vendored(x)
            else:
                projected = bf16_projection(x, self.weights.packed)
            hidden, gate = projected.split((HIDDEN_ROWS, PACKED_ROWS - HIDDEN_ROWS), dim=-1)
        else:
            projected = hidden = bf16_projection(x, self.weights.hidden)
            gate = None
        qa, kv, pe, ik, iw = hidden.split((1536, 512, 64, 128, 32), dim=-1)
        mq = self.mla_query
        if mq is not None and not mq.supports(projected):
            mq = None
        rows = x.shape[0]
        if mq is not None:
            mq.fork_prefetch(self.layer_idx, rows)
        # The sparse attention core (``FlatDsaCore``) where it serves these rows (needs the MLA query, the
        # packed gate and the key-shard kept counts); else the sparse-MLA path (sparse MLA, re-arm, mask,
        # Helix tail).
        core = self.dsa_core if mq is not None and gate is not None else None
        kept = getattr(metadata, "helix_sparse_row_lens_out", None) if core is not None else None
        valid = kept(rows) if kept is not None else None
        if core is None or valid is None or not core.supports(rows):
            core = None
        side = self.indexer_stream
        main = torch.cuda.current_stream(x.device) if side is not None else None
        on_side = (lambda: torch.cuda.stream(side)) if side is not None else contextlib.nullcontext
        fused = self.fused_indexer_rows(x.shape[0])
        front = mq is not None and fused and self.front_rows(qa, ik, iw)
        fork = query_fork(rows) if front and core is not None and side is not None else None
        log_streams(fork, rows, getattr(mq, "rank", -1))
        if fork is not None:
            # vLLM's layout: IndexFront and the indexer on this stream, the query side forked
            # onto the side stream (after or before IndexFront), joined before the core.
            if fork == "after":
                index_q, index_w = self.run_index_front(qa, ik, iw, metadata)
            side.wait_stream(main)
            with on_side():
                self._query_side(mq, projected, kv, pe, metadata, mla, rows)
            if fork == "before":
                index_q, index_w = self.run_index_front(qa, ik, iw, metadata)
            indices, kv_rows, local = self._indexer(index_q, index_w, metadata, core)
            main.wait_stream(side)
            spec = getattr(metadata, "dsa_spec_args", None)
            return self._core(
                x, core, rows, kv_rows, indices, valid, gate, output, reduce, local, spec
            )
        # The indexer (keys after the hidden projection, then its Q rows, logits and Top-K after
        # the Q norm) runs on its own stream beside the query path; attention joins both.
        if side is not None:
            side.wait_stream(main)
        if not fused:
            with on_side():
                k_fp8, k_scale = self.index_keys(ik)
                self.update_index_cache(k_fp8, k_scale, metadata)
        if mq is not None:
            # The query kernel normalizes q_a itself and leaves the normalized rows unexported:
            # the indexer's 4096 Q rows take their own norm on the indexer stream (with the
            # front, its query rows, FP8 queries, head weights and keys in one launch).
            with on_side():
                if front:
                    index_q, index_w = self.run_index_front(qa, ik, iw, metadata)
                else:
                    iq = self.index_query_rows(qa)
            query, latent = self._query_side(mq, projected, kv, pe, metadata, mla, rows)
        else:
            qr = flashinfer_rmsnorm(qa, self.weights.q_norm.weight, self.weights.q_norm.eps)
            if side is None:
                q, iq = self.project_heads(qr)
            else:
                # Split Q projection: the indexer's 4096 rows on the side stream right after
                # the norm, the 18432 attention rows here (as the >=128-row path of
                # project_heads).
                side.wait_stream(main)
                with on_side():
                    iq = bf16_linear(qr, self.weights.query.weight[18432:])
                q = bf16_linear(qr, self.weights.query.weight[:18432])
            kv = flashinfer_rmsnorm(kv, self.weights.kv_norm.weight, self.weights.kv_norm.eps)
        with on_side():
            if front:
                pass
            elif fused:
                index_q, index_w = self.index_prologue(ik, iq, iw, metadata)
            else:
                index_q, index_w = self.index_queries(iq, iw)
            indices, kv_rows, local = self._indexer(index_q, index_w, metadata, core)
        if mq is None:
            latent = torch.cat((kv, pe), dim=-1)
            fused_q = self.project_query(q)
            q_pe = q.view(-1, 96, 192)[..., 128:]
            query = mla.append(fused_q, q_pe, latent, metadata)
        if side is not None:
            main.wait_stream(side)
        if core is not None:
            spec = getattr(metadata, "dsa_spec_args", None)
            return self._core(
                x, core, rows, kv_rows, indices, valid, gate, output, reduce, local, spec
            )

        log_core("fmha", rows, getattr(mq, "rank", -1))
        partial, stats = mla.attend(query, latent, indices, metadata)
        if mq is not None:
            mq.release(self.layer_idx, x.shape[0], side)
        mask = metadata.helix_zero_or_empty_rows(x.shape[0])
        if mask is None:
            mask = helix_gen_zero_kv_mask(metadata, x.shape[0])
            empty = metadata.helix_sparse_empty_rows(x.shape[0])
            if empty is not None:
                mask = empty if mask is None else mask | empty
        if gate is None:
            combined, _ = self.exchange(partial, stats, mask, helix)
            if output is not None:
                return output(x, combined)
            return reduce(self.project_output(x, combined))
        rows_out = getattr(output, "rows", None)
        target = rows_out(x.shape[0]) if rows_out is not None else None
        combined, gated = self.exchange(partial, stats, mask, helix, gate=gate, out=target)
        if output is not None:
            return output(x, combined, gate=gate, gated=gated)
        return reduce(self.project_output(x, combined, gate=gate, gated=gated))


# ---- nv_moe ----------------------------------------
_TARGET = "kimi_k3_nvfp4__sm_107__helix16_moetp4ep4"


def runtime():
    """``kernels/moe/rubin_ull_experts_nv_experts_runtime`` of the helix16 target, loaded by path
    after :func:`install` (its ``nv_experts`` puts the same vendored tree on ``sys.path``); importing
    the target package would import the whole model."""
    install()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_experts_nv_experts_runtime as nv_experts_runtime,  # noqa: E501
    )

    return nv_experts_runtime


def image_rows(batch: int) -> bool:
    """Rows on ``RubinNvExpertsKernel`` (the latent image); above: the grouped kernel (plan)."""
    return batch <= 8


def polled_rows(batch: int) -> bool:
    """Rows whose experts poll the route mailbox: the image holds every token row only at 8 (the
    routing kernel's image count reaches the experts' ``2 * 8 - 1`` only then)."""
    return batch == 8


#: The published one-shot MoE tail at 16 rows multiplies the RMS reciprocal after the up
#: projection. The RMS squares exchange overlaps the MMA.
TAIL_LATE_RMS_ROWS = (16,)


def tail_late_rms(tail, rows: Sequence[int]) -> list:
    """Recompile the prepared one-shot published mma.sync tails for ``rows`` with late RMS.

    Keep the option in the recorded compile arguments for later ``use_heads`` recompiles.
    Return the row counts recompiled."""
    from ..kernels.vendor.vllm_mr8 import module

    compile_tail = module("cutedsl.moe.tail_runtime").compile_tail
    weights, done = tail.tail_weights, []
    for b in rows:
        if b not in weights.compiles:
            continue
        args, options = weights.compiles[b]
        if args[3] != "publish" or options.get("umma"):
            continue
        options = dict(options, late_rms=True)
        weights.compiles[b] = (args, options)
        weights.codes[b] = (compile_tail(*args, **options), *weights.codes[b][1:])
        done.append(b)
    return done


def direct_columns(weights) -> torch.Tensor:
    """w2 row order (output column of each row, :func:`weights.repack_w2`) under which the
    non-published rows (publish False) come out in column order: ``d[p^-1[d]]``, ``d`` / ``p`` =
    ``weights.w2_row_order(False / True)`` (the 8-row check reads the direct rows of the
    ``d`` packing at columns ``d`` as columns ``p``)."""
    d, p = weights.w2_row_order(False), weights.w2_row_order(True)
    inverse = torch.empty_like(p)
    inverse[p] = torch.arange(p.numel())
    return d[inverse[d]]


def direct_output_columns(weights) -> torch.Tensor:
    """Gather direct output columns from the single publish-ordered weight bank.

    ``direct_columns`` is the old direct bank's canonical column per packed row.
    That epilogue places row r at column ``direct_columns[r]``. A publish-ordered
    bank instead has canonical column ``p[r]`` there, hence gather at
    ``direct_columns[inverse(p)]``. Only activations are permuted.
    """
    p = weights.w2_row_order(True)
    return direct_columns(weights)[p.argsort()]


class NvFront:
    """``SharedFront(nv=...)`` of one NVFP4 layer."""

    def __init__(self, gsf: float, *, wide_routing: bool):
        nv = runtime().nv
        self.writers = (nv.image_chunk_nv, nv.plan_chunk_nv)
        self.gsf = float(gsf)
        self.wide_routing = wide_routing

    @staticmethod
    def image(batch: int) -> bool:
        return image_rows(batch)

    @staticmethod
    def polled(batch: int) -> bool:
        return polled_rows(batch)


class NvScratch:
    """The NVFP4 layers' shared experts workspace, route mailbox and plan buffer (one device)."""

    def __init__(self, device):
        rt = runtime()
        self.device = torch.device(device)
        self.slots = rt.slots_for(self.device, polled=True)
        buffers = rt.scratch(self.device, self.slots)
        self.workspace = buffers["workspace"]
        self.routes = rt.route_mailbox(self.workspace, self.slots)
        # SharedFront compiles its routing for vLLM's plan size; the NVFP4 image / plan are smaller
        # and start at the same base.
        vllm_plan = module("cutedsl.moe.experts").PLAN_BYTES
        self.plan = torch.zeros(max(vllm_plan, rt.nv.PLAN_BYTES), dtype=torch.uint8, device=device)
        self.image = self.plan[: rt.nv.IMAGE_BYTES]
        self.rows = self.plan[: rt.nv.PLAN_BYTES]

    def rearm(self):
        """The workspace as built (0xFF, hidden images 0x88): after a failed or aborted call."""
        runtime().arm(self.workspace, self.slots)


class NvExperts:
    """One layer's NVFP4 routed experts behind ``PersistentExperts``' interface (module docstring).

    ``packed``: ``weights.pack_routed_experts_tp16(..., publish=True)``; ``scratch``: the shared
    :class:`NvScratch`. ``prepare`` compiles outside CUDA graph captures; ``run`` only launches."""

    def __init__(
        self, packed: Dict[str, torch.Tensor], scratch: NvScratch, *, beta, linear_beta, q=1,
        rows: Optional[Sequence[int]] = None,
    ):  # fmt: skip
        import cutlass

        self.rt = rt = runtime()
        self.er = module("cutedsl.moe.experts_runtime")
        self.packed, self.scratch, self.q = packed, scratch, q
        self.device = packed["w13"].device
        self.tp, self.experts = 16, packed["w13"].shape[0]
        self.grid, self.grids = self.er.grids(self.tp, self.device)
        self.dies = ()
        self.slots = scratch.slots
        self.x_gsf = float(packed["x_gsf"].reshape(-1)[0])
        self.situ = (beta * linear_beta, 1 / beta, 1 / linear_beta)
        self.scalars = tuple(map(cutlass.Float32, self.situ))
        self.views = tuple(rt.to_cute(packed[k]) for k in ("alphas", "w13_sf", "w2_sf"))
        self.weights = (packed["w13"], packed["w13_sf"], packed["w2"], packed["w2_sf"])
        from . import weights as w

        # Direct-row output gather; no second resident w2 bank. None: the direct rows already land in
        # column order (the publish and direct w2 row orders coincide), so ``run`` gathers nothing.
        index = direct_output_columns(w)
        identity = torch.equal(index, torch.arange(index.numel(), dtype=index.dtype))
        self.direct_index = None if identity else index.to(self.device)
        self.rubin: set = set()  # rows on RubinNvExpertsKernel (``plan_coverage``: "image8")
        self.prepared: Dict[int, dict] = {}
        # ``rows``: the row counts compiled at ``prepare``; others (SpecializedDecode prepares every
        # tuned count up to 64) compile at their first ``run``, outside CUDA graph captures.
        self.rows = None if rows is None else set(rows)
        self.deferred: Dict[int, tuple] = {}

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def _w2(self, mode):
        return self.packed["w2"], self.views[2]

    def prepare(self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False):
        """``PersistentExperts.prepare``: ``buffers`` (vLLM's scratch) is not used, the NVFP4
        layers share :class:`NvScratch`; ``planned``: the routing kernel writes the image / plan
        (else ``run`` writes it from x first). Row counts outside ``rows`` are compiled at their
        first ``run``; two-shot ones raise there."""
        if self.rows is not None and batch not in self.rows:
            self.deferred[batch] = (batch, buffers, out, publish, planned, twoshot)
            return
        self._prepare(batch, out, publish, planned, twoshot)

    def _prepare(self, batch, out, publish, planned, twoshot):
        import cutlass

        rt, host = self.rt, module("cutedsl.common.host")
        if not 1 <= batch <= 64:
            raise ValueError(f"NVFP4 routed experts: rows {batch} (1-64 only)")
        if twoshot and image_rows(batch):
            raise NotImplementedError(f"NVFP4 routed experts, rows {batch}: no two-shot publish")
        mode = "twoshot" if twoshot else "push" if len(publish) > 3 else bool(publish)
        grid, dev = self.grid_for(batch), self.device.index
        poll = False
        if image_rows(batch):
            poll = bool(planned) and polled_rows(batch)
            _, code = rt.compile_experts(
                self.experts, grid, self.slots, dev, mode, poll, True, True
            )
            plan, writer = self.scratch.image, rt.write_image
            self.rubin.add(batch)
        else:
            _, code = rt.compile_grouped(self.experts, batch, grid, self.slots, dev, mode, True)
            plan, writer = self.scratch.rows, rt.write_plan
        host.check_residency(code, dev, grid, 384, "Kimi K3 NVFP4 experts")
        w2, w2_sf = self._w2(mode)
        addresses = (cutlass.Int64(self.packed["w13"].data_ptr()), cutlass.Int64(w2.data_ptr()))
        views = (
            rt.to_cute(plan),
            self.views[0],
            self.views[1],
            w2_sf,
            rt.to_cute(self.scratch.workspace),
        )
        # Direct rows land in packed-column order in ``raw``; ``run`` gathers them into ``out``.
        raw = None if publish or self.direct_index is None else torch.empty_like(out)
        launch = NvExpertsLaunch(
            code=code, views=views, output=rt.to_cute(out if raw is None else raw),
            box=tuple(publish), addresses=addresses, scalars=self.scalars, out=out, raw=raw,
            direct_index=self.direct_index, rows=batch, polled=poll, mode=mode, plan=plan,
            packed=self.packed, workspace=self.scratch.workspace, routes=self.scratch.routes,
            situ=self.situ,
        )  # fmt: skip
        self.prepared[batch] = dict(launch=launch, writer=None if planned else (writer, plan))

    def group(self, ids):
        """No pregrouped unit table (the NVFP4 engines group in the kernel)."""

    def run(self, x, ids, weights=None):
        b = x.shape[0]
        if b not in self.prepared and b in self.deferred:
            batch, _, out, publish, planned, twoshot = self.deferred.pop(b)
            self._prepare(batch, out, publish, planned, twoshot)
        p = self.prepared[b]
        if p["writer"] is not None:
            write, plan = p["writer"]
            write(x, plan, self.x_gsf)
        stream = module("cutedsl.common.host").stream(self.device)
        return nv_routed_experts(p["launch"], ids, weights, b, stream)


# ---- moe ----------------------------------------
@dataclass(frozen=True)
class MoeWeights:
    """One resident representation per projection, without shared engine modules.

    Front packs the rank's 56 router rows, eight zero padding rows, and 224
    latent-down rows. Up retains only this rank's 448 output rows. Native routed
    weights are the losslessly packed NVFP4 bank owned by the checkpoint loader.
    ``shared`` is the exact BF16 dequantization of the MXFP8 shard, run by MR8's own BF16
    shared-expert kernels as on K3 ULL.
    """

    front: torch.Tensor
    bias: torch.Tensor
    norm: torch.Tensor
    up: torch.Tensor
    shared: Bf16SharedExpertWeights
    routed: dict[str, torch.Tensor]
    scale: float
    renormalize: bool
    eps: float

    def __post_init__(self) -> None:
        for name, shape, dtype in (
            ("front", (288, 7168), torch.bfloat16),
            ("bias", (896,), torch.float32),
            ("norm", (3584,), torch.bfloat16),
            ("up", (448, 3584), torch.bfloat16),
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"MoE {name} requires contiguous {dtype} {shape}")
            if value.device != self.shared.device:
                raise ValueError("MoE native weights must share one device")
        if not isfinite(self.scale) or not isfinite(self.eps) or self.eps <= 0:
            raise ValueError("MoE requires finite model scaling and positive norm epsilon")


def prepare_moe(
    weights: MoeWeights,
    *,
    state,
    front_box,
    tail_box,
    nv_scratch,
    rank: int,
    ordinal: int,
    count: int,
    q: int,
):
    """Build a provider directly; no FakeBackbone, module rewrite, or weight mirror.

    The decode owner shares scratch and mailboxes and calls every ordinal in
    order. The three-parity tail requires a layer count other than 1 modulo 3.
    Preparation and first-use compilation must finish before graph capture.
    """
    if state.tp != 16 or not 0 <= rank < 16 or front_box.rank != rank:
        raise ValueError("Native MoE requires matching TP16 scratch and mailbox rank")
    if count <= 0 or count % 3 == 1 or not 0 <= ordinal < count:
        raise ValueError("MoE tail requires valid ordinal and layer count != 1 modulo 3")
    if not state.batches or any(not 1 <= b <= 64 for b in state.batches):
        raise ValueError("Native prepared MoE supports 1-64 rows; wider decode is not wired")
    if weights.front.device != state.device or nv_scratch.device != state.device:
        raise ValueError("MoE scratch and weights must share one device")

    from ..kernels.vendor.vllm_mr8 import module

    # Metadata only: these objects own no weights or executable shared modules.
    metadata = SimpleNamespace(
        routed_scaling_factor=weights.scale,
        moe_renormalize=weights.renormalize,
        gate=SimpleNamespace(e_score_correction_bias=weights.bias),
    )
    if not isinstance(weights.shared, Bf16SharedExpertWeights):
        raise TypeError("MoE shared experts must be the BF16 bank (Bf16SharedExpertWeights)")
    # MR8's BF16 shared path reads the two weight tensors by vLLM's attribute
    # names; the fused shared-expert kernel (B9-64) uses the scratch's counters,
    # as K3 ULL's SpecializedDecode passes them.
    metadata.shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=weights.shared.gate_up.weight),
        down_proj=SimpleNamespace(weight=weights.shared.down.weight),
    )
    shared = {"counters": state.shared}
    sms = torch.cuda.get_device_properties(state.device).multi_processor_count
    front = module("cutedsl.moe.front").SharedFront(
        metadata,
        weights.front,
        front_box,
        state.front,
        state.batches,
        4.0,
        25.0,
        plan=nv_scratch.plan,
        q=q,
        routes=nv_scratch.routes,
        nv=NvFront(float(weights.routed["x_gsf"].reshape(-1)[0]), wide_routing=sms <= 152),
        **shared,
    )
    experts = NvExperts(
        weights.routed, nv_scratch, beta=4.0, linear_beta=25.0, q=q, rows=state.batches
    )
    contract = SimpleNamespace(tp_size=16, rms_eps=weights.eps)
    tail = module("cutedsl.moe.tail_runtime").PreparedTail(
        SimpleNamespace(contract=contract, rank=rank),
        weights.norm,
        weights.up,
        state.batches,
        (tail_box, ordinal, count),
        local_up=True,
    )
    late = tail_late_rms(tail, TAIL_LATE_RMS_ROWS)
    if ordinal == 0:
        logger.info(f"K31_MOE_TAIL16_LATE_RMS rank={rank} rows={late}")
    moe = module("cutedsl.moe.prepared").PreparedMoE(front, experts, tail, state)
    front.route_launches = route_launches(
        front, state.front, front_box.buffer, weights, nv_scratch.plan, nv_scratch.routes
    )
    return moe


def route_launches(front, storage, mailbox, weights, plan, routes) -> dict[int, MoeRouteLaunch]:
    """``SharedFront.route``'s routing launch per row count (``ops.moe.rubin_ull_moe_route``): the kernel
    and the arguments ``SharedFront`` prepared, with the tensors they view. ``storage``: the
    front scratch ``SharedFront`` was built on; ``mailbox``: its front mailbox buffer;
    ``plan`` / ``routes``: the NVFP4 experts' plan and route mailbox it was given."""
    launches = {}
    for b, (_, _, code, _, args, extra, _) in front.plans.items():
        views = tuple(t[:b] for t in storage)
        signal = bool(front.route_opts[b].get("signal"))
        image = front.route_keys[b][6] == "image"
        launches[b] = MoeRouteLaunch(
            code=code, args=args, extra=extra, rows=b, mailbox=mailbox, bias=weights.bias,
            ids=views[2], weights=views[3], latent=views[6], plan=plan, image=image,
            routes=routes if signal else None,
            scale=weights.scale, renorm=weights.renormalize,
            gsf=float(weights.routed["x_gsf"].reshape(-1)[0]),
        )  # fmt: skip
    return launches


# ---- chain ----------------------------------------
# The largest SM count on which MR8's prefetch keeps its non-``WIDE`` tables (GB300). MR8's L2
# weight prefetch runs with the device's tables (``WIDE`` above this, i.e. Rubin):
# ``TLLM_K31_FLAT_L2_PREFETCH=auto``, the only mode a served configuration ran.
NARROW_SMS = 152


def _resident(value) -> tuple:
    """The tensors one projection's kernel reads: a BF16 weight, a W8A16 buffer (codes and
    scales in one) or native MXFP8 codes, then scales."""
    if isinstance(value, torch.Tensor):
        return (value,)
    buffer = getattr(value, "codes_and_scales", None)
    if isinstance(buffer, torch.Tensor):
        return (buffer,)
    codes = getattr(value, "codes", None)
    if isinstance(codes, torch.Tensor):
        return (codes, value.scales)
    weight = getattr(value, "weight", None)
    if weight is None or weight is value:
        raise TypeError(f"No resident tensor to prefetch in {type(value).__name__}")
    return _resident(weight)


class _HookedMlaQuery(FlatMlaQuery):
    """``FlatMlaQuery`` calling ``hook(layer, rows)`` from ``fork_prefetch``, which
    ``FlatDsa.generation`` calls once per DSA layer and step on the main stream right after the
    DSA hidden (in-)projection in both stream layouts (``run`` may follow on the query side
    stream): where the chain forks the ``attention_moe`` L2 prefetch."""

    hook = None

    def fork_prefetch(self, layer, rows):
        if self.hook is not None:
            self.hook(layer, rows)


class FlatChainRuntime:
    """All 69 KDA layers and all 92 TP16 MoE layers in one ordered owner.

    With BF16 (exactly dequantized) projections, every chained KDA layer runs MR8's
    split-K in-projection, KDA core and fused o-projection/all-reduce/AttnRes, and
    every MoE layer MR8's BF16 shared expert, as on K3 ULL. The DSA layers share one
    ``FlatMlaQuery``: each rank projects its 6 query heads and the ranks gather the FP8
    query by multicast.

    MR8's L2 weight prefetch (``runtime.prefetch.PrefetchController``) runs on its own
    side stream, joined in ``finish``.
    """

    # MR8's PrefetchController, or None (``_prepare_prefetch``).
    prefetch = None
    # The DSA layer whose ``attention_moe`` prefetch was forked after its in-projection.
    _dsa_forked = None
    # MR8's ``KdaSide`` (F_A / beta of the fused KDA heads), joined in ``finish``; None
    # without fused heads (``head_rows``).
    kda_side = None
    # Chained KDA layer -> (own, previous) MR8 ``POLLED_OUT`` buffers of the fused head; the
    # rows whose fused o-projection polls the head's output (``_prepare_polled``).
    head_polled: dict = {}
    polled_rows: frozenset = frozenset()
    # Rows whose KDA layers run the fused head (``FlatKda.head_rows``), set at construction.
    head_rows: frozenset = frozenset()
    _step_heads = 0
    # Layer -> the AttnRes call that also finishes an all-reduce an earlier layer left
    # to it (``FlatDense0``: L0's dense MLP -> L1); reset at L0 every step.
    deferred_before: dict = {}
    # Layer -> the rows its AttnRes stores the tap into instead of ``tap`` (93: the final
    # AttnRes), bound every step (``bind_captures``).
    captures: dict = {}

    def __init__(
        self,
        model_weights,
        *,
        loader,
        mapping,
        group,
        dsa,
        mla,
        helix,
        reduce: Callable[[torch.Tensor], torch.Tensor],
        gate_lower_bound: float,
        kda_norm_eps: float,
        moe_scale: float,
        moe_renormalize: bool,
        moe_norm_eps: float,
        rows: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64),
        tokens: int = 8,
        state_indices: Callable | None = None,
    ) -> None:
        if mapping.tp_size != 16 or not 0 <= mapping.tp_rank < 16:
            raise ValueError("The native chain requires plain TP16 mapping")
        if tokens != 8 or not rows or any(not 1 <= r <= 64 for r in rows):
            raise ValueError("Native speculative chain requires q8 and prepared rows in 1..64")
        if len(dsa) != 24 or set(dsa) != set(mla) or 0 in dsa:
            raise ValueError("The chain requires all 24 private DSA owners and solo KDA L0")
        self.weights, self.mapping, self.group = model_weights, mapping, group
        self.dsa, self.mla, self.helix = dsa, mla, helix
        self.reduce = reduce
        self.tokens, self.rows = tokens, tuple(sorted(set(rows)))
        self.gate_lower_bound, self.kda_norm_eps = gate_lower_bound, kda_norm_eps
        self.state_indices = state_indices or self._state_indices
        self.block_size = model_weights.attn_res_block_size
        self.device = model_weights.embedding.weight.device
        self.kda_weights = {
            i: loader.kda(f"language_model.model.layers.{i}.self_attn", hidden=7168)
            for i in range(93)
            if i not in dsa
        }
        self.moe_weights = {
            i: loader.moe(i, scale=moe_scale, renormalize=moe_renormalize, eps=moe_norm_eps)
            for i in range(1, 93)
        }
        self.kda = {}
        self._manager = None
        self._slots = self._accepted = None
        # The embedding rows and checkpoint indices in one launch (ops.embedding.rubin_ull_begin_fold).
        # Always on: TLLM_K31_FLAT_BEGIN_FOLD=0 is not carried over.
        self.begin_fold = True
        self._begin_fold_logged = False
        self._step_fold_logged: set = set()
        self._step_tokens = 0
        self._prefix = None
        self._next = None
        self._final = None
        self._next_moe = 1
        self._count = 0
        self._step_plain_slots = None

        from ..kernels.vendor.vllm_mr8 import module

        owner = (group, self.device, 16, mapping.tp_rank)
        self.moe_state = module("cutedsl.moe.prepared").MoEScratch(
            16, self.device, self.rows, owner=owner
        )
        self.front_box = module("cutedsl.oproj.prepared").OprojMailbox(
            *owner, (1, 64, 3584 + 2 * 896)
        )
        self.tail_box = module("cutedsl.moe.tail_runtime").MoeTailBox(*owner)
        self.nv_scratch = NvScratch(self.device)
        self.moe = {
            i: prepare_moe(
                w,
                state=self.moe_state,
                front_box=self.front_box,
                tail_box=self.tail_box,
                nv_scratch=self.nv_scratch,
                rank=mapping.tp_rank,
                ordinal=i - 1,
                count=92,
                q=tokens,
            )
            for i, w in self.moe_weights.items()
        }
        prepared = module("cutedsl.attn_res.prepared").PreparedAttnRes
        self.head_rows = self._head_rows()
        if self.head_rows:
            self.kda_side = module("cutedsl.kda.head").KdaSide(self.device)
            self._prepare_polled()
            # MR8's ``use_heads``: the MoE tail before a fused-head layer releases that
            # layer's AttnRes early at these rows (Rubin's WaveMoeTail release tables;
            # outputs bitwise identical, no change elsewhere).
            for i, moe in self.moe.items():
                if i + 1 in self.kda_weights:
                    moe.tail.tail_weights.use_heads(sorted(self.head_rows))
        # MR8's carve-out for the AttnRes before a fused-head layer: a head CTA (which asks
        # only for the shared memory it uses) is then placed beside an AttnRes CTA.
        head_carveout = module("runtime.providers").HEAD_BEFORE_CARVEOUT
        self.before_plain, self.before_ring, self.post = {}, {}, {}
        for i, weights in enumerate(model_weights.layers):
            nb = (i + self.block_size - 1) // self.block_size
            boundary = i % self.block_size == 0
            write = nb if boundary else -1
            w = weights.attention_res
            values = (w.selection_norm.weight, w.projection.weight, w.output_norm.weight)
            eps = (w.selection_norm.eps, w.output_norm.eps)
            carveout = head_carveout if self.head_rows and i in self.kda_weights else None
            plain = prepared(values, (nb, False, write, False), eps, carveout=carveout)
            plain.bind_aux(mix=True)
            self.before_plain[i] = plain
            if i >= 2:
                ring = prepared(values, (nb, True, write, False), eps, carveout=carveout)
                ring.bind_ring(self.tail_box.out.buffer)
                ring.bind_aux(mix=True)
                self.before_ring[i] = ring
            post = weights.mlp_res
            values = (post.selection_norm.weight, post.projection.weight, post.output_norm.weight)
            eps = (post.selection_norm.eps, post.output_norm.eps)
            self.post[i] = prepared(values, (nb + int(boundary), not boundary, -1, False), eps)
        final = model_weights.output
        values = (final.selection_norm.weight, final.projection.weight, final.output_norm.weight)
        eps = (final.selection_norm.eps, final.output_norm.eps)
        blocks = (93 + self.block_size - 1) // self.block_size
        self.final_ring = prepared(values, (blocks, True, -1, True), eps)
        self.final_ring.bind_ring(self.tail_box.out.buffer)
        self.final_ring.bind_aux(mix=True)
        capacity = max(self.rows)
        self.normed = torch.empty((capacity, 7168), dtype=torch.bfloat16, device=self.device)
        self.post_normed = torch.empty_like(self.normed)
        self.tap = torch.empty_like(self.normed)
        kinds = {type(w.inputs) for w in self.kda_weights.values()}
        if kinds != {Bf16KdaInput}:
            raise TypeError(f"KDA layers must hold the packed BF16 in-projection, not {kinds}")
        # The BF16 o-projection that MR8's fused o-projection consumes.
        self.local = None
        self.oproj = {}
        self._prepare_oproj(owner, capacity)
        self.dsa_oproj = FlatDsaOproj.build(
            owner,
            dsa,
            model_weights.layers,
            block_size=self.block_size,
            rows=self.rows,
            capacity=capacity,
        )
        self.mla_query = _HookedMlaQuery(group, mapping.tp_rank, dsa, self.rows, self.device)
        for owner in dsa.values():
            owner.mla_query = self.mla_query
        # The DSA layers' fused value o-projection (DsaOprojFused) at the rows it serves.
        core = self.mla_query.core
        if self.dsa_oproj is not None and core is not None:
            self.dsa_oproj.bind_value(core)
        self._prepare_prefetch()

    def _head_rows(self) -> frozenset:
        """Rows whose KDA layers run MR8's fused head: ``HEAD_ROWS`` among the
        prepared rows when every KDA layer supports it (``FlatKda.head_supported``)."""

        rows = frozenset(r for r in HEAD_ROWS if r in self.rows and r % self.tokens == 0)
        supported = all(FlatKda.head_supported(w, self.tokens) for w in self.kda_weights.values())
        return rows if supported else frozenset()

    def _prepare_polled(self) -> None:
        """MR8's ``POLLED_OUT`` (compute capability 10.7, ``polled_out_batches``) for the
        fused-head rows of the chained KDA layers:
        one [q, D_local] BF16 buffer per chained KDA layer in model order, every element
        armed to ``OUT_SENT`` once here; head(k) writes buffer k and re-arms buffer k - 1
        (mod n) after its PDL wait, and only o-proj(k) reads buffer k (see ``POLLED_OUT`` in
        ``cutedsl/kda/head.py`` for why that is safe across the kernels between them)."""
        from ..kernels.vendor.vllm_mr8 import module

        self.head_polled, self.polled_rows = {}, frozenset()
        batches = module("cutedsl.kda.head").polled_out_batches(16, self.device)
        rows = frozenset(r for r in self.head_rows if r in batches and r == self.tokens)
        chained = sorted(i for i in self.kda_weights if i != 0)
        if not rows or len(chained) < 2:
            return
        dim = next(iter(self.kda_weights.values())).inputs.local_heads * 128
        buffers = torch.empty(
            (len(chained), self.tokens, dim), dtype=torch.bfloat16, device=self.device
        )
        buffers.view(torch.int16).fill_(-1)  # OUT_SENT
        n = len(chained)
        self.head_polled = {i: (buffers[k], buffers[(k - 1) % n]) for k, i in enumerate(chained)}
        self.polled_rows = rows

    def _prepare_prefetch(self) -> None:
        """MR8's L2 weight prefetch over this chain's layers, as MR8's decode runtime builds it
        (device tables: ``WIDE`` above ``NARROW_SMS`` SMs).

        Points (``runtime.prefetch``), forked where MR8's providers fork them:
        - ``attention_moe``: the MoE front rows + shared gate/up/down (DSA layers: MR8's MLA
          variant, lean = front + down). DSA: after the hidden in-projection (the MLA query's
          launch; after the attention core without the MLA query). KDA: after the KDA core,
          MR8's KdaHead position;
        - ``oproj_down``: the front rows after the fused o-projection;
        - ``next_attention``: the next layer's attention weights after the routed experts launch.
        MR8 skips ``next_attention`` before KdaHead and before a tensor-memory MLA in-projection;
        here at the fused-head rows (``head_rows``) before every KDA layer (``BEFORE_HEAD``);
        the DSA in-projection is a FlashInfer GEMM, so nothing is skipped before it.
        ``finish`` joins the side stream into the step.
        """
        from tensorrt_llm.logger import logger

        from ..kernels.vendor.vllm_mr8 import module

        self.prefetch = None
        mla_query = getattr(self, "mla_query", None)
        if isinstance(mla_query, _HookedMlaQuery):
            mla_query.hook = None
        for moe in self.moe.values():
            moe.prefetch = None
        mode = "auto"  # the TLLM_K31_FLAT_L2_PREFETCH mode, kept in the log line
        if not self.moe:
            logger.info(
                f"K31_FLAT_L2_PREFETCH rank={self.mapping.tp_rank} mode={mode} off "
                "(needs BF16 projections and BF16 shared experts)"
            )
            return
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        wide = sms > NARROW_SMS
        layers = []
        for i in sorted(set(self.kda_weights) | set(self.dsa)):
            if i in self.dsa:
                attention = SimpleNamespace(prefetch_weights=self._dsa_weights(i))
            else:
                w = self.kda_weights[i]
                weights = (
                    *_resident(getattr(w.inputs, "packed", w.inputs)),
                    w.f_b,
                    *_resident(w.output),
                )
                # ``in_proj_qkvgfab`` marks a KDA layer for MR8's ``attention_moe`` table.
                attention = SimpleNamespace(in_proj_qkvgfab=None, prefetch_weights=weights)
            mlp = None
            if i in self.moe:
                shared = self.moe_weights[i].shared
                mlp = SimpleNamespace(
                    shared_experts=SimpleNamespace(
                        gate_up_proj=SimpleNamespace(weight=shared.gate_up.weight),
                        down_proj=SimpleNamespace(weight=shared.down.weight),
                    )
                )
            layers.append(
                SimpleNamespace(
                    layer_idx=i, is_moe_layer=i in self.moe, self_attn=attention, mlp=mlp
                )
            )
        packed = {i: self.moe_weights[i].front for i in self.moe}
        runtime = SimpleNamespace(device=self.device, tp=16)
        prefetch = module("runtime.prefetch").PrefetchController(runtime, layers, packed, wide=wide)
        for i, moe in self.moe.items():  # PreparedMoE.finish forks next_attention
            moe.prefetch, moe.layer_idx = prefetch, i
        # MR8's BEFORE_HEAD: no next-attention prefetch before a fused-head KDA layer, which
        # streams its own in-projection tiles before its PDL wait.
        before_head = module("runtime.prefetch").BEFORE_HEAD
        points = len(prefetch.tables)
        for i in self.moe:
            if self.head_rows and i + 1 in self.kda_weights:
                prefetch.skip_before(i, sorted(self.head_rows), before_head)
        if isinstance(mla_query, _HookedMlaQuery):
            mla_query.hook = self._dsa_in_projected
        self.prefetch = prefetch
        hooked = isinstance(mla_query, _HookedMlaQuery)
        logger.info(
            f"K31_FLAT_L2_PREFETCH rank={self.mapping.tp_rank} mode={mode} wide={int(wide)} "
            f"sms={sms} dsa_fork={'in_proj' if hooked else 'output'} "
            f"head_rows={sorted(self.head_rows)} skipped_before_head={points - len(prefetch.tables)} "
            + self.prefetch_summary()
        )

    def _dsa_weights(self, layer: int) -> tuple:
        """A DSA layer's attention weights in consumption order: the hidden in-projection (with
        the gate rows where one packed bank holds both), the query rows (this rank's MLA-query
        heads and W_UK, or all 96 heads), the indexer query rows, W_UV (Helix tail), the output
        gate (unless packed) and the o-projection."""
        w = self.dsa[layer].weights
        packed = getattr(w, "packed", None)
        index_rows = w.query.weight[18432:]
        mla_query = getattr(self, "mla_query", None)
        if mla_query is not None and layer in mla_query.weights:
            _, _, q_b, k_b = mla_query.weights[layer]
        else:
            q_b, k_b = w.query.weight[:18432], w.k_b
        if packed is not None:
            return (packed.weight, q_b, k_b, index_rows, w.v_b, w.output.weight)
        return (w.hidden.weight, q_b, k_b, index_rows, w.v_b, w.gate.weight, w.output.weight)

    def prefetch_summary(self, batches: Sequence[int] = (1, 8)) -> str:
        """``<point>@<layer kind>[-><next kind>]:b<rows>=<MiB>`` of every enabled point
        (several sizes joined by ``|``)."""
        if self.prefetch is None:
            return "off"
        kinds = {i: "dsa" if i in self.dsa else "kda" for i in (*self.kda_weights, *self.dsa)}
        seen = {}
        for (point, layer, batch), ranges in self.prefetch.tables.items():
            if batch not in batches:
                continue
            kind = kinds[layer]
            if point == "next_attention":
                kind += "->" + kinds.get(layer + 1, "?")
            nbytes = int(ranges.table[:, 1].sum().item()) if ranges.table.numel() else 0
            seen.setdefault((batch, point, kind), set()).add(nbytes)
        parts = [
            f"{point}@{kind}:b{batch}=" + "|".join(f"{n / 2**20:.2f}" for n in sorted(sizes))
            for (batch, point, kind), sizes in sorted(seen.items())
        ]
        return " ".join(parts) or "no points at these rows"

    def _fork(self, point: str, layer: int, rows: int) -> None:
        if self.prefetch is not None:
            self.prefetch.fork(point, layer, rows)

    def _dsa_in_projected(self, layer: int, rows: int) -> None:
        """``attention_moe`` after the DSA hidden in-projection (``_HookedMlaQuery.fork_prefetch``)."""
        if self.prefetch is not None and layer in self.dsa:
            self._dsa_forked = layer
            self.prefetch.fork("attention_moe", layer, rows)

    def _prefetching_output(self, layer: int, output):
        """The DSA output stage with ``attention_moe`` (unless forked after the in-projection)
        before it and ``oproj_down`` after the fused o-projection."""

        def run(x, combined, *args, **kwargs):
            if self._dsa_forked != layer:
                self._fork("attention_moe", layer, x.shape[0])
            out = output(x, combined, *args, **kwargs)
            self._fork("oproj_down", layer, x.shape[0])
            return out

        run.__dict__.update(output.__dict__)  # the stage's own attributes (e.g. ``rows``)
        return run

    def _prepare_oproj(self, owner, capacity: int) -> None:
        """K3 ULL's fused o-projection + TP16 all-reduce + post-attention AttnRes per chained
        KDA layer (MR8 ``PreparedOproj``), on the exactly dequantized BF16 o-projection.

        Parity alternates over chained KDA layers in model order, so consecutive calls never
        share a Lamport slot set. All KDA core rows go through one ``local`` buffer, the
        fused producer's mapped x. Solo L0 keeps the explicit partial + reduce.
        """
        from ..kernels.vendor.vllm_mr8 import module

        oproj = module("cutedsl.oproj.prepared")
        twoshot = module("cutedsl.common.twoshot")
        dim = next(iter(self.kda_weights.values())).inputs.local_heads * 128
        self.local = torch.empty((capacity, dim), dtype=torch.bfloat16, device=self.device)
        self.oproj_box = oproj.OprojMailbox(*owner)
        self.twoshot = twoshot.TwoShot(*owner)
        chained = sorted(i for i in self.kda_weights if i != 0)
        if len(chained) % 2:
            raise ValueError(
                f"The fused KDA o-projection needs an even number of chained KDA layers, got "
                f"{len(chained)}: OprojMailbox / TwoShot slots alternate on the layer ordinal's "
                "parity, so the last layer of a step and the first layer of the next must "
                "differ in parity to never share a Lamport slot set"
            )
        for ordinal, i in enumerate(chained):
            post = self.weights.layers[i].mlp_res
            boundary = i % self.block_size == 0

            def norm(weight):
                return SimpleNamespace(weight=weight.weight, variance_epsilon=weight.eps)

            decoder = SimpleNamespace(
                layer_idx=ordinal,  # the mailbox / two-shot parity only
                self_attn=SimpleNamespace(
                    o_proj=SimpleNamespace(weight=self.kda_weights[i].output.weight)
                ),
                mlp_res_norm=norm(post.selection_norm),
                mlp_res_proj=SimpleNamespace(weight=post.projection.weight),
                post_attention_layernorm=norm(post.output_norm),
                prev_valid_blocks=(i + self.block_size - 1) // self.block_size,
                is_block_write_layer=boundary,
            )
            self.oproj[i] = oproj.PreparedOproj(
                decoder,
                self.oproj_box,
                self.rows,
                False,
                twoshot=self.twoshot,
                x_buffer=self.local,
                head_polled=tuple(sorted(self.polled_rows)),
            )
        carve_oproj_consumers(
            "kda", self.oproj, self.oproj_box.tp, int(self.oproj_box.rank), self.device
        )
        missing = [
            (i, r)
            for i, po in self.oproj.items()
            for r in self.polled_rows
            if r not in po.head_polled
        ]
        if missing:  # every polled head needs its o-projection's polling code
            from tensorrt_llm.logger import logger

            logger.warning(
                f"K31 flat KDA head: no polled o-projection for {missing}; POLLED_OUT off"
            )
            self.head_polled, self.polled_rows = {}, frozenset()

    @staticmethod
    def _state_indices(metadata):
        mamba = metadata.mamba_metadata
        aligned = getattr(mamba, "generation_state_indices", None)
        if aligned is not None:
            return aligned
        return mamba.state_indices[metadata.num_contexts : metadata.seq_lens.shape[0]]

    def _begin(self, metadata, embed=None) -> torch.Tensor | None:
        rows = metadata.num_tokens
        if rows not in self.rows or metadata.num_contexts:
            raise ValueError("Warm up a prepared generation-only native chain row count")
        manager = metadata.kv_cache_manager
        if not getattr(manager, "uses_kda_checkpoints", False):
            raise ValueError("Flat chain requires the manager's native checkpoint role")
        physical = self.state_indices(metadata)
        if physical.numel() <= 0 or rows % physical.numel():
            raise ValueError("KDA requests must uniformly cover the generation rows")
        width = rows // physical.numel()
        if width not in (1, self.tokens):
            raise ValueError("The native chain supports plain q1 or speculative q8 steps")
        if manager is not self._manager:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Bind KDA checkpoint cache views before graph capture")
            verify = tuple(r for r in self.rows if r % self.tokens == 0)
            extra = dict(projection_rows=self.rows, output=self.local)
            if self.kda_side is not None:
                extra.update(head_side=self.kda_side)
            self.kda = {
                i: FlatKda(
                    w,
                    *manager.mamba_layer_cache(i).kda_ckpt_kernel_caches,
                    rows=verify,
                    tokens=self.tokens,
                    gate_lower_bound=self.gate_lower_bound,
                    norm_eps=self.kda_norm_eps,
                    head_polled=self.head_polled.get(i),
                    **extra,
                )
                for i, w in self.kda_weights.items()
            }
            self._manager = manager
            self._audit_caches(manager)
        embedded, begun = None, None
        run_held = getattr(metadata, "run_held_step_begin", None)
        if run_held is not None:

            def launch(prep, meta_args, meta_kwargs):
                nonlocal begun
                if embed is None:
                    prep_and_metadata(prep, meta_args, meta_kwargs)
                else:
                    begun = step_begin(
                        prep, meta_args, meta_kwargs, manager._kda_replay, physical, *embed
                    )

            held = run_held(launch)
            if rows not in self._step_fold_logged and not torch.cuda.is_current_stream_capturing():
                from tensorrt_llm.logger import logger

                self._step_fold_logged.add(rows)
                logger.info(
                    f"K31_GLUE_PFOLD rank={self.mapping.tp_rank} rows={rows} held={int(held)} "
                    f"folded={int(begun is not None)}"
                )
        if begun is not None:
            self._slots, self._accepted, embedded = begun
        elif embed is None:
            self._slots, self._accepted = manager._kda_replay.checkpoint_indices(physical)
        else:
            self._slots, self._accepted, embedded = begin_fold(
                manager._kda_replay, physical, *embed
            )
        if (
            self._slots.dtype != torch.int32
            or self._slots.shape != (physical.numel(), self.tokens)
            or self._accepted.dtype != torch.int32
            or self._accepted.shape != (physical.numel(),)
        ):
            raise ValueError(
                "The manager must provide q8 position indices and one accepted count per request"
            )
        if width == 1:
            for core in self.kda.values():
                if core.decode_core is None or rows not in core.decode_core.kernels:
                    core.prepare_decode((rows,))
        self._step_plain_slots = physical if width == 1 else None
        self._step_heads = sum(k.head_launches for k in self.kda.values())
        self._step_folds = sum(k.fold_launches for k in self.kda.values())
        self._step_tokens = width
        self._next, self._final = None, None
        self._next_moe, self._count = 1, 0
        return embedded

    def _audit_caches(self, manager) -> None:
        """Log the runtime cache precision this rank actually binds (once per manager)."""
        from tensorrt_llm.logger import logger

        state = {str(core.caches[1].dtype) for core in self.kda.values()}
        conv = {str(core.caches[0].dtype) for core in self.kda.values()}
        if state != {"torch.float32"}:
            raise RuntimeError(f"Flat KDA recurrent checkpoints must be FP32, got {state}")
        pools = getattr(manager, "indexer_k_cache_pool_per_layer", None) or {}
        index = {str(getattr(p, "dtype", type(p).__name__)) for p in pools.values()}
        logger.info(
            f"K31_ULL_FLAT_CACHE_AUDIT rank={self.mapping.tp_rank} "
            f"latent_kv={getattr(manager, 'dtype', 'unknown')} indexer_k_pools={sorted(index)} "
            f"kda_state={sorted(state)} kda_conv={sorted(conv)} kda_layers={len(self.kda)} "
            f"kda_head_rows={sorted(self.head_rows)} "
            f"kda_head_layers={sum(bool(k.head_rows) for k in self.kda.values())}"
        )
        multi = [k.multi for k in self.kda.values() if k.multi_rows]
        if multi:
            logger.info(
                f"K31_KDA_HEAD_MULTI16 rank={self.mapping.tp_rank} layers={len(multi)} "
                f"order={'/'.join(sorted({h.order for h in multi}))} "
                f"rows={sorted(set().union(*(k.multi_rows for k in self.kda.values())))}"
            )
        folds = [k for k in self.kda.values() if k.fold_rows]
        if folds:
            logger.info(
                f"K31_KDA_FOLD rank={self.mapping.tp_rank} layers={len(folds)} "
                f"rows={sorted(set().union(*(k.fold_rows for k in folds)))} "
                f"tactic={folds[0].fold.tactic}"
            )
        heads = [k.head for k in self.kda.values() if k.head_rows]
        if heads:
            # One launch order beside SignalGemm for the device's fused heads.
            orders = sorted({h.order for h in heads})
            logger.info(
                f"K31_KDA_HEAD_ORDER {'/'.join(orders)} rank={self.mapping.tp_rank} "
                f"heads={len(heads)} requested=sibling"
            )

    def begin_step(self, metadata) -> None:
        """Bind graph-stable checkpoint indices before the first model layer."""
        self._begin(metadata)

    def begin_step_embedding(
        self, metadata, ids: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor | None:
        """``begin_step`` that also gathers the target's embedding rows of ``ids`` from
        ``table`` in the same launch, and returns them (``ops.embedding.rubin_ull_begin_fold``).

        Returns None, having done nothing, for ids and a table that :func:`embedding` would
        not take the row copy for; the caller then gathers the rows and calls ``begin_step``.
        """
        folds = self.begin_fold and folds_embedding(ids, table)
        if not self._begin_fold_logged:
            from tensorrt_llm.logger import logger

            self._begin_fold_logged = True
            logger.info(
                f"K31_FLAT_BEGIN_FOLD rank={self.mapping.tp_rank} folded={int(folds)} "
                f"switch={int(self.begin_fold)}"
            )
        if not folds:
            return None
        return self._begin(metadata, (ids, table))

    def before(self, layer, prefix, blocks, count, capture=None):
        if layer == 0:
            self._next, self._final = None, None
            self._next_moe, self._count = 1, 0
            self.deferred_before = {}
        if self._next is not None:
            expected, normed, saved, count, boundary = self._next
            if expected != layer:
                raise RuntimeError("Native MoE tail must feed the next layer in model order")
            self._next = None
            self._prefix = saved
        else:
            expected = (layer + self.block_size - 1) // self.block_size
            if count != expected:
                raise ValueError("AttnRes snapshot count disagrees with the layer boundary")
            self._prefix = prefix
            rows = prefix.shape[0]
            normed = self.normed[:rows]
            deferred = self.deferred_before.pop(layer, None)
            if deferred is not None:
                deferred(prefix, blocks.permute(1, 0, 2), normed, aux=self._tap(layer, rows))
            else:
                cluster_attn_res(
                    self.before_plain[layer],
                    prefix,
                    prefix,
                    blocks.permute(1, 0, 2),
                    normed,
                    self._tap(layer, rows),
                )
            boundary = layer % self.block_size == 0
            count += int(boundary)
        if capture is not None:
            capture(self.tap[: normed.shape[0]])
        self._count = count
        return normed, None if boundary else self._prefix, count

    def bind_captures(self, rows: dict) -> None:
        """This step's captured taps: layer -> the rows of the drafter's capture buffer that
        the AttnRes before ``layer`` (93: the final one) stores its mixture into, as its aux
        output, in place of ``tap`` (``FlatModel._capture_rows``)."""
        self.captures = rows

    def _tap(self, layer: int, rows: int) -> torch.Tensor:
        """The aux rows of the AttnRes before ``layer``: its capture rows, else ``tap``."""
        rows_out = self.captures.get(layer)
        return self.tap[:rows] if rows_out is None else rows_out[:rows]

    def _attend(self, layer, x, metadata):
        if self._slots is None:
            self._begin(metadata)
        core = self.kda[layer]
        if self._step_tokens == 1:
            return core.attend_decode(x, self._slots, self._accepted)
        return core.attend(x, self._slots, self._accepted)

    def run_kda(self, layer, x, metadata):
        """Reduced o-projection of a KDA layer whose residual glue is the caller's (L0)."""
        return self.reduce(self.kda[layer].output_partial(self._attend(layer, x, metadata)))

    def run_dsa(self, layer, x, metadata):
        return self.dsa[layer].generation(
            x, metadata, mla=self.mla[layer], helix=self.helix, reduce=self.reduce
        )

    def fuses_dsa(self, layer) -> bool:
        return self.dsa_oproj is not None and layer in self.dsa_oproj

    def run_dsa_layer(self, layer, prefix, blocks, count, metadata, capture=None):
        """DSA attention whose o-projection, all-reduce and post AttnRes are MR8's fused one."""
        normed, _, count = self.before(layer, prefix, blocks, count, capture)
        prefix, out = self._prefix, self.post_normed[: normed.shape[0]]
        output = self.dsa_oproj.output(layer, self.dsa[layer], prefix, blocks.permute(1, 0, 2), out)
        if self.prefetch is not None:
            self._dsa_forked = None
            output = self._prefetching_output(layer, output)
        self.dsa[layer].generation(
            normed,
            metadata,
            mla=self.mla[layer],
            helix=self.helix,
            reduce=self.reduce,
            output=output,
        )
        return self.run_moe(layer, out, prefix, blocks, count, metadata)

    def run_kda_layer(self, layer, prefix, blocks, count, metadata, capture=None):
        normed, _, count = self.before(layer, prefix, blocks, count, capture)
        prefix = self._prefix
        out = self.post_normed[: normed.shape[0]]
        if layer in self.oproj:
            rows = self._attend(layer, normed, metadata)
            self._fork("attention_moe", layer, normed.shape[0])
            head = normed.shape[0] in self.kda[layer].head_rows and self._step_tokens > 1
            polled = head and self.kda[layer].head_polled is not None
            self.oproj[layer].run(
                rows, prefix, blocks.permute(1, 0, 2), out, after_head=head, polled=polled
            )
            self._fork("oproj_down", layer, normed.shape[0])
        else:
            delta = self.run_kda(layer, normed, metadata)
            cluster_attn_res(self.post[layer], delta, prefix, blocks.permute(1, 0, 2), out, None)
        return self.run_moe(layer, out, prefix, blocks, count, metadata)

    def run_moe(self, layer, x, prefix, blocks, count, metadata):
        if layer != self._next_moe:
            raise RuntimeError("The shared native MoE runtime must execute every ordinal 1..92")
        self._next_moe += 1
        self._prefix = prefix
        rows, view = x.shape[0], blocks.permute(1, 0, 2)
        target = self.normed[:rows]

        def consume(out):
            if layer == 92:
                cluster_attn_res_ring(self.final_ring, prefix, view, out, self._tap(93, rows))
                self._final = out
            else:
                nxt = layer + 1
                cluster_attn_res_ring(
                    self.before_ring[nxt], prefix, view, out, self._tap(nxt, rows)
                )
                boundary = nxt % self.block_size == 0
                next_count = count + int(boundary)
                self._next = (nxt, out, prefix, next_count, boundary)
                self._count = next_count
            return out

        self.moe[layer].run(x, target, consume=consume)
        return prefix, self._count

    def finish(self, prefix, blocks, count, capture=None):
        if self._next_moe != 93 or self._final is None:
            raise RuntimeError("The final native tail/AttnRes consumer did not run")
        if self.prefetch is not None:  # the prefetch side stream rejoins inside the step
            self.prefetch.join()
        if self.kda_side is not None:  # likewise the fused heads' F_A / beta stream
            self.kda_side.join()
        if capture is not None:
            capture(self.tap[: prefix.shape[0]])
        if self._step_plain_slots is not None:
            # A plain decode publishes checkpoint zero for every KDA layer.
            # Do not reset this per layer: later layers still read the old count.
            self._manager._kda_replay.prev_num_accepted_tokens.index_fill_(
                0, self._step_plain_slots.long(), 0
            )
        rows = prefix.shape[0]
        logged = self.__dict__.setdefault("_head_logged", {})
        if self.head_rows and logged.get(rows, 0) < 2:
            # The first two steps per row count (warmup / capture): fused-head launches.
            from tensorrt_llm.logger import logger

            logged[rows] = logged.get(rows, 0) + 1
            heads = sum(k.head_launches for k in self.kda.values()) - self._step_heads
            logger.info(
                f"K31_FLAT_KDA_HEAD rank={self.mapping.tp_rank} rows={rows} q={self._step_tokens} "
                f"head_launches={heads} kda_layers={len(self.kda)}"
            )
        fold_logged = self.__dict__.setdefault("_fold_logged", {})
        if (
            rows in FOLD_ROWS
            and fold_logged.get(rows, 0) < 2
            and any(k.fold_rows for k in self.kda.values())
        ):
            # The first two steps per folded row count (warmup / capture): folded-pair launches.
            from tensorrt_llm.logger import logger

            fold_logged[rows] = fold_logged.get(rows, 0) + 1
            folds = sum(k.fold_launches for k in self.kda.values()) - getattr(
                self, "_step_folds", 0
            )
            logger.info(
                f"K31_FLAT_KDA_FOLD rank={self.mapping.tp_rank} rows={rows} q={self._step_tokens} "
                f"fold_launches={folds} kda_layers={len(self.kda)}"
            )
        self._slots = self._accepted = None
        return self._final


# ---- dense0 ----------------------------------------
#: Ranks the DENSE0 split covers (the plain TP16 group).
WORLD = 16
#: Hidden size the kernels serve.
DENSE0_HIDDEN = 7168
#: Largest step the split-K kernels serve.
DENSE0_MAX_ROWS = 32
#: Rows of the L1 mailbox slots (MR8's mailbox capacity).
MAILBOX_ROWS = 64
#: Widest step whose finish moves to L1 (module docstring).
FINISH_L1_MAX_ROWS = 8


def _kernels():
    from ..kernels.gemm import rubin_ull_k3_dense0 as k3_dense0

    return k3_dense0


def _dense0_sm100_family(device: torch.device) -> bool:
    from tensorrt_llm._utils import is_sm_100f

    major, minor = torch.cuda.get_device_capability(device)
    return is_sm_100f(major * 10 + minor)


class FlatDense0:
    """L0's dense MLP + all-reduce + residual add on one rank of the TP16 split.

    ``weights``: this rank's ``[2 I_s, H]`` (gate rows then up rows) / ``[H, I_s]`` slice of
    the BF16 checkpoint MLP; ``reduce``: a TP16 ``FlatAllReduce`` used by this owner
    only. Construction is collective (none here beyond ``reduce``) and outside graph capture:
    it builds the kernels' weight layout and the workspace's device view and compiles the
    kernels of every row count in ``rows``.
    """

    def __init__(
        self,
        weights: DenseWeights,
        reduce,
        *,
        beta: float,
        linear_beta: Optional[float],
        rows: TypingSequence[int] = (),
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FlatDense0 must be built outside CUDA graph capture")
        if getattr(reduce, "world", 0) != WORLD or reduce.dtype != torch.bfloat16:
            raise ValueError("FlatDense0 needs a BF16 TP16 all-reduce")
        if reduce.hidden != DENSE0_HIDDEN:
            raise ValueError("FlatDense0 serves the native hidden width")
        gate_up, down = weights.gate_up.weight, weights.down.weight
        if gate_up.shape[1] != DENSE0_HIDDEN or down.shape[0] != DENSE0_HIDDEN:
            raise ValueError("Dense slice does not match the native hidden width")
        self.weights, self.reduce = weights, reduce
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        self.inter = down.shape[1]
        self.threshold = int(reduce.threshold)
        self.view = None
        self.l1 = None
        self.chain = None
        self.stats: Dict[str, int] = {"fitpush": 0, "fallback": 0, "finish_l1": 0}
        self._logged: set = set()
        k = _kernels()
        if not _dense0_sm100_family(gate_up.device):
            return
        with torch.no_grad():
            # slice_weights(index 0 of 1): the fused 64-row gate/up tiles, K-padded down.
            self.gate_up, self.down = k.slice_weights(gate_up, down, 0, 1)
        padded = self.down.shape[1]
        self.hidden = torch.zeros(
            DENSE0_MAX_ROWS, padded, dtype=torch.bfloat16, device=gate_up.device
        )
        self.grid_out = torch.empty(
            DENSE0_MAX_ROWS, 2 * self.inter, dtype=torch.bfloat16, device=gate_up.device
        )
        from ..kernels.comm.rubin_ull_k3_lamport_ar import LamportWorkspace

        self.view = LamportWorkspace.from_mnnvl(
            {
                "handle": reduce._handle,
                "uc_buffer": reduce._buffer,
                "buffer_flags": reduce._flags,
                "buffer_size_bytes": int(reduce.capacity),
            },
            reduce.world,
            reduce.mapping.tp_rank,
        )
        self._rows = sorted({r for r in rows if 0 < r <= DENSE0_MAX_ROWS})
        self._precompile(self._rows)

    def bind_l1(self, chain) -> bool:
        """Leave the all-reduce finish and residual add to L1's AttnRes in ``chain`` (a
        ``FlatChainRuntime``: ``before_plain[1]``, ``weights``, ``deferred_before``, ``group``,
        ``device``, ``mapping``). Collective (the mailbox), outside graph capture. Returns
        whether it is bound (not without the kernels)."""
        if self.view is None:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind the L1 finish outside CUDA graph capture")
        from ..kernels.vendor.vllm_mr8 import module

        plain = chain.before_plain[1]
        nb, has_delta, write, final = plain.key
        if has_delta or final:
            raise ValueError("L1's AttnRes must be the plain (no delta) pre-attention site")
        res = chain.weights.layers[1].attention_res
        weights = (res.selection_norm.weight, res.projection.weight, res.output_norm.weight)
        eps = (res.selection_norm.eps, res.output_norm.eps)
        rank = chain.mapping.tp_rank
        mailbox = module("cutedsl.oproj.prepared").OprojMailbox(
            chain.group, chain.device, WORLD, rank, (WORLD, MAILBOX_ROWS, DENSE0_HIDDEN)
        )
        ring = module("cutedsl.attn_res.prepared").PreparedAttnRes(
            weights, (nb, True, write, False), eps, carveout=plain.carveout
        )
        ring.bind_ring(mailbox.buffer, sanitize=True)
        if plain.aux is not None:
            ring.bind_aux(mix=plain.aux_mix)
        from types import SimpleNamespace

        device = self.gate_up.device
        # The push kernel's one-shot address is mc + (tok * world + rank) * row bytes in the
        # buffer that ``flags`` selects: world 1, buffer 0 and rank -> rank x mailbox rows give
        # slot ``rank`` row ``tok`` of the mailbox.
        self._mail = SimpleNamespace(
            world=1,
            rank=rank * MAILBOX_ROWS,
            mc_base=mailbox.mc,
            peer_ptrs=torch.zeros(1, dtype=torch.int64, device=device),
            flags=torch.zeros(4, dtype=torch.int32, device=device),
        )
        self._mail_out = torch.empty(
            DENSE0_MAX_ROWS, DENSE0_HIDDEN, dtype=torch.bfloat16, device=device
        )
        self.mailbox, self.l1, self.chain = mailbox, ring, chain
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        k = _kernels()
        pdl = get_env_enable_pdl()
        rows = [r for r in self._rows if r <= FINISH_L1_MAX_ROWS and self._plan(r)]
        for gu, down, one_shot in {self._plan(r) for r in rows}:
            if not one_shot:
                k._compiled_push(torch.bfloat16, down, pdl, 1, True, False, True)
        return True

    # ------------------------------------------------------------------ checks
    def _one_shot(self, rows: int) -> bool:
        """The transport of the C++ all-reduce the call replaces (its threshold)."""
        return rows * DENSE0_HIDDEN * WORLD * 2 <= self.threshold

    def _plan(self, rows: int) -> Optional[Tuple[object, object, bool]]:
        """``(gate/up tactic, down tactic, one_shot)`` for ``rows`` rows, or None."""
        if self.view is None or not 1 <= rows <= DENSE0_MAX_ROWS or rows > self.reduce.max_rows:
            return None
        from ..kernels.comm.rubin_ull_k3_lamport_ar import required_bytes

        k = _kernels()
        gu, down = k.default_tactics(rows)
        try:
            k.validate_tactics(rows, self.inter, gu, down)
        except ValueError:
            return None
        one_shot = self._one_shot(rows)
        if required_bytes(rows, DENSE0_HIDDEN, WORLD, one_shot) > self.view.bytes_per_buffer:
            return None
        return gu, down, one_shot

    def _precompile(self, rows: TypingSequence[int]) -> None:
        """Compile the kernels every row count in ``rows`` launches (no launch, no collective)."""
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ..kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

        k = _kernels()
        pdl = get_env_enable_pdl()
        compiled = set()
        for r in rows:
            plan = self._plan(r)
            if plan is None:
                continue
            gu, down, one_shot = plan
            key = (gu, down, one_shot)
            if key in compiled:
                continue
            k.ensure_cute_runtime()
            k._compiled_situ(torch.bfloat16, gu, pdl, self.beta, self.linear_beta)
            k._compiled_push(torch.bfloat16, down, pdl, WORLD, one_shot, False, True)
            k3_lamport_ar._compiled_finish(
                WORLD, DENSE0_HIDDEN, one_shot, pdl, False, one_shot, True, False
            )
            compiled.add(key)

    def _log(self, path: str, rows: int, finish: str = "l0") -> None:
        if (path, rows, finish) in self._logged:
            return
        self._logged.add((path, rows, finish))
        from tensorrt_llm.logger import logger

        logger.info(
            f"K31_ULL_L0_DENSE path={path} finish={finish} rows={rows} split=tp16 "
            f"inter={self.inter} rank={self.reduce.mapping.tp_rank}"
        )

    # ------------------------------------------------------------------ issue
    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """The rank's BF16 TP16 partial through ``F.linear`` + the Triton SiTU (fallback)."""

        return dense_partial(x, self.weights, beta=self.beta, linear_beta=self.linear_beta)

    def __call__(self, x: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """``prefix + all_reduce_tp16(dense_slice(x))`` for BF16 ``x`` / ``prefix`` ``[M, H]``.

        With the L1 finish bound (:meth:`bind_l1`) and the two-shot transport, returns
        ``prefix`` itself: L1's AttnRes adds the reduced rows to it in place."""
        if x.dtype != torch.bfloat16 or prefix.dtype != torch.bfloat16:
            raise TypeError("L0 dense requires BF16 activations")
        rows = x.shape[0]
        plan = None if torch.compiler.is_compiling() else self._plan(rows)
        if plan is None:
            self.stats["fallback"] += 1
            self._log("fallback", rows)
            return prefix + self.reduce(self.partial(x))
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ..kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar
        from ..ops.comm.rubin_ull_push_allreduce import LamportPushTarget
        from ..ops.gemm.rubin_ull_dense0_down_push import dense0_down_push
        from ..ops.gemm.rubin_ull_dense0_situ_gate_up import dense0_situ_gate_up

        gu, down, one_shot = plan
        pdl = get_env_enable_pdl()
        if not x.is_contiguous():
            x = x.contiguous()
        hidden = self.hidden[:rows]
        deferred = self.l1 is not None and not one_shot and rows <= FINISH_L1_MAX_ROWS
        dense0_situ_gate_up(
            x,
            self.gate_up,
            hidden,
            self.grid_out[:rows],
            gu,
            self.beta,
            self.linear_beta,
            pdl,
        )
        if deferred:
            from types import SimpleNamespace

            mail = SimpleNamespace(
                workspace=self._mail, one_shot=True, output=self._mail_out[:rows], pushed=False
            )
            dense0_down_push(hidden, self.down, mail, down, pdl)
            self.chain.deferred_before[1] = self.l1.run_ring
            self.stats["fitpush"] += 1
            self.stats["finish_l1"] += 1
            self._log("fitpush", rows, "l1")
            return prefix
        out = torch.empty_like(prefix)
        target = LamportPushTarget(self.view, one_shot, one_shot, out)
        dense0_down_push(hidden, self.down, target, down, pdl)
        k3_lamport_ar.run_lamport_finish(
            self.view, out, one_shot, use_pdl=pdl, cpp_one_shot=one_shot
        )
        self.stats["fitpush"] += 1
        self._log("fitpush", rows)
        return prefix + out


__all__ = ["FlatDense0", "WORLD"]


# ---- model ----------------------------------------
@dataclass(frozen=True)
class FlatLayerWeights:
    """The two attention-residual sites; L0 additionally owns native dense weights."""

    attention_res: AttnResWeights
    mlp_res: AttnResWeights
    dense: DenseWeights | None = None


@dataclass(frozen=True)
class FlatModelWeights:
    """Target embedding, glue and output head, with no shared compute modules."""

    embedding: Bf16Weight
    layers: tuple[FlatLayerWeights, ...]
    output: AttnResWeights
    lm_head: Bf16Weight
    vocab_size: int
    vocab_start: int
    attn_res_block_size: int
    dense_beta: float
    dense_linear_beta: float | None

    def __post_init__(self) -> None:
        hidden = self.embedding.weight.shape[1]
        if self.vocab_size <= 0 or self.vocab_size % 16:
            raise ValueError(
                "Target vocabulary must contain a positive TP16-aligned number of rows"
            )
        if len(self.layers) != 93 or hidden != 7168 or self.attn_res_block_size <= 0:
            raise ValueError(
                "Flat K3.1 needs all 93 native layers and a positive snapshot block size"
            )
        if self.embedding.weight.shape[0] != self.vocab_size:
            raise ValueError("Target embedding must retain its complete native vocabulary")
        if self.lm_head.weight.shape != (self.vocab_size // 16, hidden):
            raise ValueError("Target LM head must be the rank's native TP16 vocabulary shard")
        if self.vocab_start % (self.vocab_size // 16):
            raise ValueError("Target vocabulary shard must align with TP16")
        if not 0 <= self.vocab_start < self.vocab_size:
            raise ValueError("Target vocabulary start is outside the checkpoint vocabulary")
        if self.layers[0].dense is None or any(
            layer.dense is not None for layer in self.layers[1:]
        ):
            raise ValueError("The native K3.1 target has a dense MLP at L0 only")


Capture = Callable[[torch.Tensor], None]


class FlatDecodeBackend(Protocol):
    """Native kernel providers owned by the model, including K3 ULL full-layer chains.

    Full-layer KDA consumers own their attention-residual transitions. The
    remaining calls consume already-normalized rows. MoE returns the updated
    prefix so its native tail can fuse the residual update.
    """

    def begin_step(self, metadata) -> None: ...

    def run_kda_layer(
        self,
        layer: int,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        count: int,
        metadata,
        capture: Capture | None,
    ) -> tuple[torch.Tensor, int]: ...

    def before(
        self,
        layer: int,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        count: int,
        capture: Capture | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]: ...

    def run_kda(self, layer: int, x: torch.Tensor, metadata) -> torch.Tensor: ...

    def run_dsa(self, layer: int, x: torch.Tensor, metadata) -> torch.Tensor: ...

    def fuses_dsa(self, layer: int) -> bool: ...

    def run_dsa_layer(
        self,
        layer: int,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        count: int,
        metadata,
        capture: Capture | None,
    ) -> tuple[torch.Tensor, int]: ...

    def run_moe(
        self,
        layer: int,
        x: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        count: int,
        metadata,
    ) -> tuple[torch.Tensor, int]: ...

    def finish(
        self, prefix: torch.Tensor, blocks: torch.Tensor, count: int, capture: Capture | None = None
    ) -> torch.Tensor: ...


class FlatLanguageHead:
    """Native BF16 target head, shared by target scoring and the draft facade."""

    def __init__(
        self, weight: Bf16Weight, *, group: Sequence[int], vocab_size: int, vocab_start: int
    ) -> None:
        self.weight = weight
        self.group = tuple(group)
        self.vocab_size, self.vocab_start = vocab_size, vocab_start
        # The executor validates request token ids against the head's full vocabulary.
        self.num_embeddings = vocab_size
        if len(self.group) != 16 or len(set(self.group)) != 16:
            raise ValueError("The flat target head requires the plain TP16 group")
        if weight.weight.shape[0] * 16 != vocab_size:
            raise ValueError("Target language-head shard does not cover one TP16 vocabulary part")

    def local_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """BF16 GEMM rounded before the baseline's FP32 logits conversion."""
        if hidden.dtype != torch.bfloat16 or hidden.shape[-1] != self.weight.weight.shape[1]:
            raise ValueError("The native target head consumes BF16 hidden rows")
        return bf16_linear(hidden, self.weight.weight).float()

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return full-vocabulary FP32 logits through the existing native gather."""
        if hidden.dtype != torch.bfloat16 or hidden.shape[-1] != self.weight.weight.shape[1]:
            raise ValueError("The native target head consumes BF16 hidden rows")
        return self.gather(bf16_linear(hidden, self.weight.weight))

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        """The FP32 full-vocabulary logits of every rank's BF16 shard ``local [..., shard]``."""
        shape = local.shape
        logits = gather_columns(local.reshape(-1, shape[-1]), self.group)
        return logits.reshape(*shape[:-1], self.vocab_size).float()


def gather_columns(local: torch.Tensor, group: Sequence[int]) -> torch.Tensor:
    """``[rows, len(group) * width]``: every rank's ``local [rows, width]`` side by side in group
    order (one all-gather of the flattened shards, then a copy into rows)."""
    rows, width = local.shape
    gathered = allgather(local.contiguous().view(1, -1), None, list(group))
    return gathered.view(len(group), rows, width).transpose(0, 1).reshape(rows, -1)


#: The draft-token agreement's all-reduce row, in FP32 columns (the drafts take the first K). The
#: MNNVL kernels clear the Lamport buffer of an earlier call with the threads of the current one
#: (lamportUtils.cuh, clearDirtyLamportBuf): a row of K = 7 drafts is one block of two threads, which
#: cleared the drafter's earlier all-reduce buffers in about 260 us a step (GB300); a 2048-column
#: row is a block of 512.
_DRAFT_AGREEMENT_WIDTH = 2048


class ZeroPaddedReduce:
    """A copy across a group as one all-reduce of a zero-padded buffer.

    A writing rank puts ``local [rows, columns]`` at its ``offset`` in a ``[rows, width]``
    buffer, a rank without an offset writes nothing, and all ranks all-reduce the buffer: every
    column is zero on every rank but the one that wrote it, and a sum of one value and zeros is
    that value. Every rank at its own offset makes an all-gather; one writing
    rank, a broadcast. Bitwise the copy, except that a -0.0 arrives as +0.0 (the MNNVL
    all-reduce's Lamport sentinel). The ``[max_rows, width]`` buffer persists and only this
    rank's columns are ever written, so the others stay zero without a fill: one copy kernel and
    one MNNVL all-reduce in place of NCCL's ring all-gather (about 20-35 us at the drafter's sizes
    on GB300). ``all_reduce`` is the group's engine ``AllReduce``; ``build`` returns None unless
    it runs on MNNVL in the buffer's dtype and the shared MNNVL workspace already holds every
    call up to ``max_rows`` (growing it would move buffers other owners hold views of).
    """

    def __init__(
        self, all_reduce: Any, offset: Optional[int], columns: int, buffer: torch.Tensor
    ) -> None:
        self.all_reduce = all_reduce
        self.offset = offset
        self.columns = columns
        self.buffer = buffer

    @classmethod
    def build(
        cls,
        all_reduce: Any,
        offset: Optional[int],
        columns: int,
        width: int,
        max_rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional["ZeroPaddedReduce"]:
        mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
        if mnnvl is None or mnnvl.dtype != dtype:
            return None
        # The kernels move 16-byte vectors along a row.
        vector = 16 // dtype.itemsize
        width = -(-width // vector) * vector
        if offset is not None and not 0 <= offset <= offset + columns <= width:
            raise ValueError(f"{columns} columns at {offset} outside a row of {width}")
        world = mnnvl.mapping.tp_size
        need = max(
            mnnvl.get_required_workspace_size(rows, width, world, dtype)
            for rows in range(1, max_rows + 1)
        )
        workspace = get_or_scale_allreduce_mnnvl_workspace(mnnvl.mapping, dtype)
        if need > workspace["buffer_size_bytes"]:
            return None
        return cls(
            all_reduce, offset, columns, torch.zeros(max_rows, width, dtype=dtype, device=device)
        )

    def fits(self, rows: int, columns: int) -> bool:
        """Whether a ``[rows, columns]`` copy goes through this buffer; the same answer on every
        rank of the group."""
        return 0 < rows <= self.buffer.shape[0] and columns == self.columns

    def own_columns(self, rows: int) -> torch.Tensor:
        """This rank's ``[rows, columns]`` of the buffer (a strided view) for a producer to write in
        place of :meth:`stage`; only a writing rank (one with an offset) has columns."""
        if self.offset is None:
            raise ValueError("a rank without an offset writes no columns")
        return self.buffer[:rows, self.offset : self.offset + self.columns]

    def stage(self, local: torch.Tensor, rows: int) -> None:
        """This rank's ``local [rows, columns]`` into its columns of the buffer (no exchange)."""
        if self.offset is not None:
            self.buffer[:rows, self.offset : self.offset + local.shape[1]].copy_(local)

    def reduce(self, rows: int) -> torch.Tensor:
        """The all-reduced ``[rows, width]`` buffer, as staged."""
        return self.all_reduce(self.buffer[:rows])

    def __call__(self, local: torch.Tensor, rows: int) -> torch.Tensor:
        """The all-reduced ``[rows, width]`` buffer after this rank's ``local [rows, columns]``."""
        self.stage(local, rows)
        return self.reduce(rows)


class FlatVocabParallelHead(VocabParallelHead):
    """The engine's vocab-parallel token selection over the native head.

    The engine's ``VocabParallelHead`` takes a module ``LMHead``; this one takes
    ``FlatLanguageHead``, whose BF16 weight is this rank's vocabulary shard (rows from
    ``vocab_start``) of the plain TP16 group ``mapping``. The pair exchange is the base class's:
    an exact float32 all-reduce over that group (MNNVL where it builds, else NCCL). ``local_logits``
    are the BF16 shard logits the gathered path converts to FP32, so the selections equal
    ``torch.argmax`` / the sampler over ``FlatLanguageHead``'s gathered logits.
    """

    def __init__(self, head: FlatLanguageHead, mapping: Any) -> None:
        shard = head.weight.weight.shape[0]
        if mapping.tp_size != 16 or head.vocab_start != mapping.tp_rank * shard:
            raise ValueError("the native head's shard must be this TP16 rank's vocabulary part")
        self.lm_head = head
        self.vocab_size = head.vocab_size
        self.shard_width = shard
        self.col_base = head.vocab_start
        self._exchange = None
        if MNNVLAllReduce.is_mnnvl(mapping, torch.float32):
            candidate = tp_pair_exchange(mapping, AllReduceStrategy.AUTO)
            if candidate.all_reduce.mnnvl_allreduce is not None:
                self._exchange = candidate
        if self._exchange is None:
            self._exchange = tp_pair_exchange(mapping, AllReduceStrategy.NCCL)

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return head_linear(hidden_states, self.lm_head.weight.weight)

    def gather_logits(self, local: torch.Tensor) -> torch.Tensor:
        return self.lm_head.gather(local)

    def selection_pairs(self, local: torch.Tensor) -> tuple:
        """Every rank's ``(value, global index)`` argmax pairs of shard logits ``local [rows,
        shard]``, exchanged and not yet reduced: ``(pairs, row stride, slot stride, slots)``,
        the operand of ``select_accept``. The local kernel and the exchange of
        ``vocab_parallel_argmax`` (whose last kernel, the pair reduction, ``select_accept``
        does in its launch)."""
        rows, n_cols = local.shape
        assert local.stride(-1) == 1
        layout = self._exchange.layout(rows, n_cols, local.device)
        if rows > 0:
            _local_argmax_kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                n_cols,
                self.col_base,
                layout.chunk,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
        pairs, slots = self._exchange.exchange(layout)
        return pairs, pairs.stride(0), 2, slots


class FlatModel:
    """K3.1 generation-only model with explicit glue and optimized-layer providers.

    No baseline model is built or rewritten. Weight owners and providers are
    constructed once; eager, graph, warmup and larger rows use this same loop.
    """

    def __init__(
        self,
        weights: FlatModelWeights,
        backend: FlatDecodeBackend,
        *,
        dsa_layers: Sequence[int],
        group: Sequence[int],
        dense0: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        # ``dense0(normed, prefix)``: L0's dense MLP, all-reduce and residual add in one
        # provider (``FlatDense0``).
        self.weights, self.backend, self.dense0 = weights, backend, dense0
        self.dsa_layers = frozenset(dsa_layers)
        if len(self.dsa_layers) != 24 or any(not 0 <= i < 93 for i in self.dsa_layers):
            raise ValueError("Flat K3.1 requires the checkpoint's 24 zero-based DSA layers")
        if 0 in self.dsa_layers:
            raise ValueError("L0 must use the native solo KDA path")
        self.num_attn_res_snapshots = (
            93 + weights.attn_res_block_size - 1
        ) // weights.attn_res_block_size
        self.lm_head = FlatLanguageHead(
            weights.lm_head,
            group=group,
            vocab_size=weights.vocab_size,
            vocab_start=weights.vocab_start,
        )

    @staticmethod
    def _capture_rows(spec_metadata, rows: int) -> dict[int, torch.Tensor]:
        """Layer -> the rows of DFlash's capture buffer that the tap of the layer before it
        goes to (93: layer 92's, the final AttnRes), when the buffer takes the tap as is
        (``DFlashSpecMetadata.maybe_capture_hidden_states`` without a projection, a BF16
        buffer 7168 wide per layer); empty otherwise. The AttnRes that mixes the tap then
        stores it there itself (its aux rows take any 16-byte row stride) instead of
        ``maybe_capture_hidden_states``' slice copy: a graph memcpy node, which also cuts
        the PDL chain to the next kernel (+5 us on each captured KDA layer on Rubin)."""
        if (
            not isinstance(spec_metadata, DFlashSpecMetadata)
            or type(spec_metadata).maybe_capture_hidden_states
            is not DFlashSpecMetadata.maybe_capture_hidden_states
            or spec_metadata._capture_projection is not None
            or spec_metadata.captured_hidden_states is None
            or spec_metadata.captured_hidden_states.dtype != torch.bfloat16
            or spec_metadata.hidden_size != 7168
        ):
            return {}
        buffer = spec_metadata.captured_hidden_states
        return {
            layer + 1: buffer[:rows, i * 7168 : (i + 1) * 7168]
            for layer, i in spec_metadata._layer_to_idx.items()
            if 0 <= layer <= 92
        }

    def _capture(self, spec_metadata, layer: int) -> Capture | None:
        if spec_metadata is None or layer < 0:
            return None
        selected = getattr(spec_metadata, "_capture_layer_set", None)
        if selected is not None and layer not in selected:
            return None
        return lambda value: spec_metadata.maybe_capture_hidden_states(layer, value, None)

    @staticmethod
    def _attention_input(
        prefix: torch.Tensor, blocks: torch.Tensor, weights: AttnResWeights, capture: Capture | None
    ) -> torch.Tensor:
        if capture is not None:
            mixed = attn_res(prefix, blocks, weights) if blocks.shape[0] else prefix
            capture(mixed)
            return rms_norm(mixed, weights.output_norm)
        if blocks.shape[0]:
            return attn_res_and_norm(prefix, blocks, weights)
        return rms_norm(prefix, weights.output_norm)

    def __call__(
        self,
        metadata,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_metadata=None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one target input: ids or native embeddings")
        if metadata.num_contexts:
            raise ValueError("The flat generation target does not serve context requests")
        if getattr(metadata, "is_ragged_verify", False):
            raise ValueError(
                "The flat target requires uniform q8 verification or q1 generation rows"
            )
        prefix, begun = inputs_embeds, False
        if prefix is None:
            # A backend that gathers the rows with its step begin returns them; None otherwise.
            begin_embedding = getattr(self.backend, "begin_step_embedding", None)
            if begin_embedding is not None:
                prefix = begin_embedding(metadata, input_ids, self.weights.embedding.weight)
                begun = prefix is not None
            if prefix is None:
                prefix = embedding(input_ids, self.weights.embedding)
        if prefix.dtype != torch.bfloat16 or prefix.ndim != 2 or prefix.shape[1] != 7168:
            raise ValueError("Target hidden rows must be BF16 [rows,7168]")
        if prefix.shape[0] != metadata.num_tokens:
            raise ValueError("Flat target rows must match generation metadata")
        blocks = prefix.new_empty((self.num_attn_res_snapshots, *prefix.shape))
        if not begun:
            self.backend.begin_step(metadata)
        # Taps the backend's AttnRes stores straight into the capture buffer: no callable.
        bind = getattr(self.backend, "bind_captures", None)
        direct = self._capture_rows(spec_metadata, prefix.shape[0]) if bind is not None else {}
        if bind is not None:
            bind(direct)
        count = 0
        for i, weights in enumerate(self.weights.layers):
            capture = None if i in direct else self._capture(spec_metadata, i - 1)
            if i in self.dsa_layers and self.backend.fuses_dsa(i):
                prefix, count = self.backend.run_dsa_layer(
                    i, prefix, blocks, count, metadata, capture
                )
                continue
            if i and i not in self.dsa_layers:
                prefix, count = self.backend.run_kda_layer(
                    i, prefix, blocks, count, metadata, capture
                )
                continue
            normed, prefix, count = self.backend.before(i, prefix, blocks, count, capture)
            delta = (
                self.backend.run_dsa(i, normed, metadata)
                if i in self.dsa_layers
                else self.backend.run_kda(i, normed, metadata)
            )
            prefix = delta if prefix is None else prefix + delta
            normed = attn_res_and_norm(prefix, blocks[:count], weights.mlp_res)
            if i:
                prefix, count = self.backend.run_moe(i, normed, prefix, blocks, count, metadata)
            else:
                prefix = self.dense0(normed, prefix)
        final = None if 93 in direct else self._capture(spec_metadata, 92)
        return self.backend.finish(prefix, blocks, count, final)


# ---- spec ----------------------------------------
class FlatLogitsProcessor:
    """The baseline row selection and FP32 logits conversion without a module."""

    def forward(
        self,
        hidden: torch.Tensor,
        head: FlatLanguageHead,
        metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not return_context_logits:
            if metadata is None:
                hidden = hidden[-1]
            else:
                last = torch.cumsum(metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden = hidden[last]
        return head(hidden).float()

    __call__ = forward


class _DraftEmbedding:
    """A non-owning callable over the target's embedding table (``FlatModelWeights.embedding``)."""

    def __init__(self, core: "DraftCore") -> None:
        self._core = weakref.ref(core)

    @property
    def table(self) -> torch.Tensor:
        """The ``[vocab, hidden]`` table, the full vocabulary on every rank."""
        return self._core().w["embed"]

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.table)

    forward = __call__


# ---- DFlash2: the drafter container, its block step and the speculative worker -------------------
#
# The drafter's weights are ``DraftCore.w["d_*"]`` (weights.py's ``FlatCheckpointLoader.dflash2``,
# BF16), beside ``w["embed"]``, the target's embedding table the drafter embeds its block with.
# ``DraftModel`` holds no parameter of its own: it reads them through the core and answers what the
# engine's worker, the KV cache creator and the shell read off a drafter. ``draft_step`` is the block
# decode of the five drafter layers, straight-line, every kernel a family or catalog entry.
#
# Helix CP16: the history (one FP8 latent of 576 values per token and drafter layer) is page-sharded
# over the 16 helix ranks of the unified draft cache, as the target's KV is: a rank owns the
# positions of every 16th page. Each rank computes all 96 query heads over the keys it owns (the
# window entry writes each row's partial and its log-sum-exp), then one exchange over the target's
# helix mapping merges the ranks and leaves each rank its 6 heads. ``o_proj`` and the MLP are
# row-parallel over the 16 ranks repurposed as TP (the plain TP16 group of the target's dense MLP),
# one all-reduce each. ``fc`` is output-row-parallel over the same group: a rank projects its
# sixteenth of the drafter's hidden width (448 rows) and one all-gather assembles the row. The
# block's latents go into a persistent scratch the window entry reads, never into the pool. The
# target and drafter logits are the target head's (``FlatLanguageHead``): this rank's vocabulary
# shard where the worker selects vocab-parallel (``FlatVocabParallelHead``), else gathered.
# --------------------------------------------------------------------------------------------------

#: The drafter this target runs: the DFlash2 MLA drafter at a fixed draft length (``max_draft_len``),
#: verified as a block of ``max_draft_len + 1`` rows per request (``FlatChainRuntime``'s q8).
DRAFT_LEN = 7
# The drafter history's storage: FP8 E4M3 at a unit scale, a target constant.
_DRAFT_HISTORY_DTYPE = torch.float8_e4m3fn
# speculative_config.context_kv_dtype may leave it to the target ("auto"; None where the config has
# no such field) or name it ("fp8"); routing and the worker refuse any other value.
_DRAFT_HISTORY_CONFIGS = (None, "auto", "fp8")
# Block verification (not token-wise rejection sampling) for a sampled batch.
_DRAFT_BLOCK_VERIFY = True
# The engine's dflash_forward keywords, as the worker passes them.
_DRAFT_FORWARD_ARGS = (
    "noise_embedding",
    "query_positions",
    "num_ctx_per_req",
    "ctx_k_cache",
    "ctx_v_cache",
    "ctx_cache_batch_idx",
    "ctx_kv_cache",
    "ctx_page_table",
    "block_placement",
)
#: First-GEN ring fills (headtp) checked against the CP ranks' owner pages when the
#: ``ring_fill_check`` diagnostic is on; every fill logs one K31_RING_FILL line (a gate counts them
#: against the requests that reached generation).
_RING_FILL_CHECKS = 8
#: Forced-acceptance calls between two forced-count lines.
_FORCED_PERIOD = 200
#: Transferred requests between two K31_CTX_FIRST_DRAFTS lines (the first request logs too).
_CTX_FIRST_DRAFT_PERIOD = 200
#: Steps between two K31_RS_AGREE readouts of the sampled path's agreement totals.
_RS_AGREE_PERIOD = 200
#: Steps after the first K31_RS_AGREE_HASH line (step=first) to the late one (step=late).
_RS_HASH_LATE_STEPS = 10 * _RS_AGREE_PERIOD
# The digest's 64-bit mixing multipliers (splitmix64's), as signed int64.
_RS_HASH_MUL = (-7046029254386353131, -4658895280553007687, -7723592293110705685)


def _rs_digest(*tensors: torch.Tensor) -> torch.Tensor:
    """A device int64 digest of the tensors' bytes: every 32-bit word, tagged with its position,
    mixed and summed (wrapping), so equal bytes give equal digests on every rank in any reduction
    order and one changed word changes the digest. Eager; a few small launches."""
    words = torch.cat([t.contiguous().view(torch.int32).reshape(-1) for t in tensors])
    x = (words.to(torch.int64) & 0xFFFFFFFF) | (
        torch.arange(words.numel(), dtype=torch.int64, device=words.device) << 32
    )
    x = x * _RS_HASH_MUL[0]
    x = x ^ ((x >> 30) & 0x3FFFFFFFF)
    x = x * _RS_HASH_MUL[1]
    x = x ^ ((x >> 27) & 0x1FFFFFFFFF)
    return (x * _RS_HASH_MUL[2]).sum()


# select_accept holds every verify row in one program's registers: at most this many requests of
# each kind (contexts, generations) take it.
_SELECT_ACCEPT_MAX_REQUESTS = 16
# The selector's candidates per slot that select_candidate_tokens' walk takes (the op's TOP_K).
_SELECTOR_TOP_K = 32
# The drafter's attention window in positions, the query's own included (sliding_window).
_DRAFT_WINDOW = DRAFT_SHAPE["sliding_window"]
# Body chunks of the window entry: one, as on helix8.
_DRAFT_WINDOW_SPLITS = 1
# The window entry's launch tiles at one request's 8 rows x 96 heads: BLOCK_DV 64 splits the tail's
# 512 latent columns over 8 CTAs (96 in all, 24 at the default 256), bitwise the default tiles.
_DRAFT_WINDOW_TILES = WindowTiles(block_m=64, block_dv=64, num_warps=8, num_stages=1)
# The ranked headtp partial's (BC, BK, warps, splits, maxnreg) on Rubin (``_rubin_rows``; GB300: the ops'
# RANKED_TILES) by requests in the step; other counts take ``_DRAFT_RANKED_TILES_RUBIN_REST``. BK and
# maxnreg do not change the partials' bits; the splits change the FP32 order of each group's W_UV sum.
# One request: two splits of the latent columns over CTAs, 192 CTAs on 212 SMs, 140 registers and no
# spill at BK 128 under the serving Triton / ptxas for sm_107a (one split: 255 and 16 spill stores).
# Other request counts keep one split: on Rubin, two requests' split tiles (128, 64, 8, 2, 128) timed
# slower than one split.
_DRAFT_RANKED_TILES_RUBIN = {1: (128, 128, 8, 2, None)}
_DRAFT_RANKED_TILES_RUBIN_REST = (128, 128, 8, 1, None)
# AllReduceFusionOp.NONE: the drafter's o_proj and MLP sums.
_DRAFT_AR_NONE = 0
# Low-M plans of the drafter's projections at one request's block, (bound, tactic) as ULL_DECODE_PLAN;
# larger row counts keep cuBLAS. GB300, 8 rows, weights rotated over 8 copies, CUDA graph, PDL on, us,
# cuBLAS -> plan: conv_proj and gate_up 7.8 -> 5.1, q_a 8.8 -> 4.7, kv_a 4.5 -> 3.2 (bitwise),
# q_b 10.0 -> 9.4 (bitwise), o_proj 2.7 -> 2.3 (bitwise), down 3.3 -> 2.7 (bitwise), the history's
# stacked kv_a 9.7 -> 7.5. The split-K rows sum the same BF16 products in another FP32 order.
# fc's 448 output rows per rank keep cuBLAS: 10.5 against 12.2 for the best plan row (whole fc 96.4).
_DRAFT_GEMM_PLAN = {
    (1792, 7168): ((8, ("splitk", 64, 8, 4, 12)),),  # attention / MLP conv_proj; gate_up / 16
    (1536, 7168): ((8, ("splitk", 64, 8, 4, 12)),),  # q_a
    (576, 7168): ((8, ("splitk", 64, 8, 4, 12)),),  # kv_a
    # [q_a | kv_a] of one layer: q_a's and kv_a's tactic, so each output column sums the same
    # split-K partials as the two GEMMs did (bitwise them), in one launch: 4.66 + 3.21 -> 5.97.
    (2112, 7168): ((8, ("splitk", 64, 8, 4, 12)),),
    (18432, 1536): ((8, ("splitk", 128, 8, 1, 6)),),  # q_b, 96 heads
    # q_b, 6 heads (the headtp layout); on Rubin: 2.41 against 6.48 for the 96 heads.
    (1152, 1536): ((8, ("splitk", 64, 8, 4, 12)),),
    # The headtp layout's absorbed query (``_absorbed_query_weight``): 6 heads x [W_UK^T q_b nope | q_b rope].
    (3456, 1536): ((8, ("splitk", 64, 8, 2, 6)),),
    (7168, 768): ((8, ("tgv", 18)),),  # o_proj, 6 heads
    (7168, 896): ((8, ("tgv", 18)),),  # down / 16
    (2880, 7168): ((8, ("splitk", 64, 8, 2, 12)),),  # the five layers' kv_a (history)
}


# Rubin rows of the drafter's projections (``_rubin_rows``) at one request's block: vLLM
# main's warp split-K (``_warp``; output tile 16, K tile 256, the deepest ring, two B-loader warps),
# (output_tile, token_tile, k_tile, stages, b_loader_warps) by weight shape. One Rubin GPU, PDL-chained,
# cold weights, us a call: conv_proj / gate_up 3.70 -> 3.63, [q_a | kv_a] 3.92 ->
# 3.82, q_b (6 heads) 1.94 -> 1.73; o_proj kept its tgv row (no gain). Not bitwise the plan's
# split-K (another K split). Each weight's engine is prepared on its first call outside graph capture.
_DRAFT_WARP_ROWS = 8
_DRAFT_WARP_RUBIN = {
    (1792, 7168): (16, 8, 256, 16, 2),
    (2112, 7168): (16, 8, 256, 16, 2),
    (1152, 1536): (16, 8, 256, 6, 2),
    (3456, 1536): (16, 8, 256, 6, 2),
}
#: (weight data_ptr, shape) -> the weight's prepared warp split-K engine (``_draft_warp_engine``).
_DRAFT_WARP_ENGINES: dict = {}


def _draft_warp_engine(weight: torch.Tensor):
    """The prepared warp split-K engine of ``weight`` (a ``_DRAFT_WARP_RUBIN`` shape), or None while
    a graph is being captured and it was not prepared yet."""
    key = (weight.data_ptr(), tuple(weight.shape))
    engine = _DRAFT_WARP_ENGINES.get(key)
    if engine is None and not torch.cuda.is_current_stream_capturing():
        w = _warp()
        tactic = w.WarpSplitKTactic(*_DRAFT_WARP_RUBIN[tuple(weight.shape)])
        engine = _DRAFT_WARP_ENGINES[key] = w.PreparedWarpSplitK(weight, {_DRAFT_WARP_ROWS: tactic})
    return engine


# The drafter's side-chain projections, off the drafter block's chain (the step's context overlap):
# the five layers' stacked history kv_a [2880, 7168] (``_DRAFT_GEMM_PLAN``'s split-K) and fc's rows
# of this rank [448, 43008] (cuBLAS), up to one request's verify block. Rubin rows (``_rubin_rows``)
# by weight shape replace that choice; a choice is "cublas", a ``bf16_low_m_gemm`` split-K tactic,
# or ``("proj", m_tile, n_tile, split_k, stages)``: MR8's split-K projection (``PreparedProjection``:
# split K by any divisor of K / 128, the weight TMA before the PDL wait). The model config field
# (``_DRAFT_SIDE_FIELDS``) forces one choice. Empty: every device keeps its choice, bit for bit.
_DRAFT_SIDE_ROWS = 8
_DRAFT_SIDE_GEMM_RUBIN: dict = {}
_DRAFT_SIDE_FIELDS = {(2880, 7168): "draft_kv_stack_gemm", (448, 43008): "draft_fc_gemm"}
#: (weight data_ptr, shape, tactic) -> the weight's prepared MR8 projection (``_draft_side_gemm``).
_DRAFT_PROJECTIONS: dict = {}


def draft_side_choice(x: torch.Tensor, weight: torch.Tensor) -> Optional[tuple]:
    """The kernel choice of a side-chain projection ``x @ weight.T`` (``_DRAFT_SIDE_FIELDS``
    shapes): the model config's field, else the Rubin row; None keeps ``draft_linear``'s choice."""
    field = _DRAFT_SIDE_FIELDS.get(tuple(weight.shape))
    if field is None or not (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= _DRAFT_SIDE_ROWS
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.data_ptr() % 32
        and not weight.data_ptr() % 32
        and _sm100_family(x.device)
    ):
        return None
    spec = getattr(k31_features(), field)
    if spec:
        return ("cublas",) if spec == "cublas" else parse_gemm_choice(spec)
    if _rubin_rows(x.device):
        return _DRAFT_SIDE_GEMM_RUBIN.get(tuple(weight.shape))
    return None


def _draft_side_gemm(x: torch.Tensor, weight: torch.Tensor, choice: tuple) -> torch.Tensor | None:
    """``x @ weight.T`` on ``choice``'s kernel; None when an MR8 projection is not prepared yet and
    a graph is being captured."""
    if choice[0] == "cublas":
        return rubin_linear(x, weight, False)
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    if choice[0] == "proj":
        key = (weight.data_ptr(), tuple(weight.shape), choice)
        projection = _DRAFT_PROJECTIONS.get(key)
        if projection is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            providers = module("runtime.providers")
            policies = {rows: choice[1:] for rows in range(1, _DRAFT_SIDE_ROWS + 1)}
            projection = providers.PreparedProjection(weight, policies).prepare()
            _DRAFT_PROJECTIONS[key] = projection
        return prepared_projection(projection, x, out)
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return bf16_low_m_gemm(x, weight, out, choice, get_env_enable_pdl())


# Rubin rows (``_rubin_rows``) of the drafter's projections at three and four requests' blocks
# (24 and 32 rows at DL7), which otherwise run cuBLAS: {rows: choice} by weight shape. A choice is
# ``_HEAD_GEMM_ROWS``'s (``("warp", output_tile, token_tile, k_tile, stages, b_loader_warps)`` or a
# ``bf16_low_m_gemm`` tactic) or ``("proj", m_tile, n_tile, split_k, stages)``, MR8's split-K
# projection. Only the row counts of ``_DRAFT_WIDE_ROWS`` are looked up, so one request's 8 rows and
# two requests' 16 rows run the kernels they ran. Shapes without a row keep cuBLAS.
# One Rubin GPU (PDL, cold weights, each call behind its producer, us a call cuBLAS -> row):
# conv_proj / gate_up 24 rows 6.07 -> 5.66 (x15), 32 rows 6.15 -> 5.75; [q_a | kv_a]
# 24 rows 8.06 -> 5.90 (x5), 32 rows 8.10 -> 5.93; 112 / 132 CTAs. The absorbed query,
# o_proj, down and the history's kv_a had no row that was clearly faster over a step.
_DRAFT_WIDE_ROWS = (24, 32)
_DRAFT_WIDE_RUBIN: dict = {
    (1792, 7168): {24: ("splitk", 64, 32, 4, 8), 32: ("splitk", 64, 32, 4, 8)},
    (2112, 7168): {24: ("splitk", 64, 32, 4, 8), 32: ("splitk", 64, 32, 4, 8)},
}
#: (weight data_ptr, shape, rows, choice) -> the weight's prepared MR8 projection (``_draft_wide_gemm``).
_DRAFT_WIDE_PROJECTIONS: dict = {}


def _draft_wide_choice(x: torch.Tensor, weight: torch.Tensor) -> Optional[tuple]:
    """The ``_DRAFT_WIDE_RUBIN`` choice of ``x @ weight.T`` on this device, or None."""
    rows = x.shape[0]
    if rows not in _DRAFT_WIDE_ROWS or not (x.is_cuda and x.ndim == 2):
        return None
    choice = _DRAFT_WIDE_RUBIN.get(tuple(weight.shape), {}).get(rows)
    if choice is None or not (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.data_ptr() % 32
        and not weight.data_ptr() % 32
        and _rubin_rows(x.device)
    ):
        return None
    return choice


def _draft_wide_gemm(x: torch.Tensor, weight: torch.Tensor, choice: tuple) -> torch.Tensor | None:
    """``x @ weight.T`` on ``choice``'s kernel; None when its engine or projection is not prepared
    yet and a graph is being captured."""
    if choice[0] != "proj":
        return _head_gemm(x, weight, choice)
    rows = x.shape[0]
    key = (weight.data_ptr(), tuple(weight.shape), rows, choice)
    projection = _DRAFT_WIDE_PROJECTIONS.get(key)
    if projection is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        providers = module("runtime.providers")
        projection = providers.PreparedProjection(weight, {rows: choice[1:]}).prepare()
        _DRAFT_WIDE_PROJECTIONS[key] = projection
    out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    return prepared_projection(projection, x, out)


def draft_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for the drafter's BF16 projections: a side-chain projection's
    ``draft_side_choice``, else the ``_DRAFT_WIDE_RUBIN`` row, else ``_DRAFT_WARP_RUBIN`` on Rubin,
    else ``_DRAFT_GEMM_PLAN``, else cuBLAS."""
    side = draft_side_choice(x, weight)
    if side is not None:
        if _rubin_rows(x.device):
            log_rubin_row(
                "D", f"shape={weight.shape[0]}x{weight.shape[1]} rows={x.shape[0]} {side}"
            )
        out = _draft_side_gemm(x, weight, side)
        if out is not None:
            return out
    wide = _draft_wide_choice(x, weight)
    if wide is not None:
        out = _draft_wide_gemm(x, weight, wide)
        if out is not None:
            log_rubin_row(
                "W", f"shape={weight.shape[0]}x{weight.shape[1]} rows={x.shape[0]} {wide}"
            )
            return out
    if (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.shape[0] == _DRAFT_WARP_ROWS
        and tuple(weight.shape) in _DRAFT_WARP_RUBIN
        and _rubin_rows(x.device)
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.data_ptr() % 32
        and not weight.data_ptr() % 32
    ):
        engine = _draft_warp_engine(weight)
        if engine is not None:
            n, k = weight.shape
            log_rubin_row("A", f"shape={n}x{k} tactic={_DRAFT_WARP_RUBIN[(n, k)]}")
            out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
            return warp_splitk_gemm(engine, x, out)
    if x.is_cuda and x.ndim == 2 and x.dtype == torch.bfloat16 and _sm100_family(x.device):
        policies = _DRAFT_GEMM_PLAN.get(tuple(weight.shape), ())
        choice = next((c for bound, c in policies if x.shape[0] <= bound), None)
        if (
            choice is not None
            and x.shape[0] >= 1
            and x.is_contiguous()
            and weight.is_contiguous()
            and not x.data_ptr() % 32
            and not weight.data_ptr() % 32
        ):
            from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

            out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
            return bf16_low_m_gemm(x, weight, out, choice, get_env_enable_pdl())
    return rubin_linear(x, weight, False)


#: (weight data_ptr, shape) -> the conv kernel projection's prepared fold engine (``_draft_fold_engine``).
_DRAFT_FOLD_ENGINES: dict = {}


def _draft_fold_engine(weight: torch.Tensor):
    """The prepared ``PreparedConvProj`` of the conv kernel projection ``weight`` (the Rubin rows' warp
    split-K tactic of its shape), or None while a graph is being captured and it was not prepared yet."""
    key = (weight.data_ptr(), tuple(weight.shape))
    engine = _DRAFT_FOLD_ENGINES.get(key)
    if engine is None and not torch.cuda.is_current_stream_capturing():
        from ..kernels.spec.rubin_ull_dflash2_conv_proj_fold import PreparedConvProj

        tactic = _warp().WarpSplitKTactic(*_DRAFT_WARP_RUBIN[tuple(weight.shape)])
        engine = _DRAFT_FOLD_ENGINES[key] = PreparedConvProj(weight, tactic)
    return engine


def draft_conv_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    base: torch.Tensor,
    block: int,
    d: "_DraftRt",
    pdl: bool,
    site: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A conv's kernel projection and its side-0 convolution of ``x``: ``(coef [rows, 2, taps, groups], conv)``.

    An 8-row block on an sm_100 family device can use one launch
    (``dflash2_conv_proj_fold``: the convolution in the GEMM's epilogue); else ``draft_linear`` then
    ``dflash2_grouped_conv``. The fold is bitwise the Rubin rows' warp split-K GEMM and convolution."""
    rows = x.shape[0]
    taps, groups = d.taps, d.groups
    if (
        rows == _DRAFT_WARP_ROWS
        and taps == 2
        and d.group == 16
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not x.data_ptr() % 32
        and weight.is_contiguous()
        and not weight.data_ptr() % 32
        and base.is_contiguous()
        and not base.data_ptr() % 32
        and tuple(weight.shape) in _DRAFT_WARP_RUBIN
        and _sm100_family(x.device)
    ):
        engine = _draft_fold_engine(weight)
        if engine is not None:
            _log_draft_path(d, site, rows, "conv_fold")
            coef, conv = dflash2_conv_proj_fold(engine, x, base, block)
            return coef.view(rows, 2, taps, groups), conv
    coef = draft_linear(x, weight).view(rows, 2, taps, groups)
    return coef, dflash2_grouped_conv(x, coef[:, 0], base, block, d.group, pdl)


# The vocabulary shard GEMM ``[rows, 7168] x [10240, 7168]^T`` of ``FlatVocabParallelHead.local_logits``:
# the target head at 8 rows a request (verify) and the drafter's block at 7 rows a request. cuBLAS
# runs it as a GEMM plus a split-K reduce kernel, without PDL. These rows run one PDL kernel instead,
# (largest row count served, choice) by device class; a choice is a ``bf16_low_m_gemm`` tactic or
# ``("warp", output_tile, token_tile, k_tile, stages, b_loader_warps)`` (vLLM main's warp split-K).
# Row counts above the last bound keep cuBLAS. Not bitwise cuBLAS: each kernel sums the same BF16
# products in its own FP32 order and rounds once.
# GB300: warp split-K, 32-row output tiles (320 CTAs, up to 5 / 3 resident per SM by shared memory at
# 8 / 16 rows, one wave on 152 SMs). Single GPU, cold weights (8 rotated copies), CUDA graph, PDL, each
# call behind a producer kernel, us a call: 8 rows cuBLAS 26.76 -> 23.38, 16 rows 27.38 -> 23.84.
# Rubin (``_rubin_rows``): up to 8 rows, warp split-K with 32-row output tiles and 8 stages (320 CTAs,
# 2 resident per SM, one wave on 212 SMs; two waves on GB300's 152): faster than cuBLAS over a step at
# 8 rows on one Rubin GPU (the same bench, both heads a step). Rubin 9-32 rows (two to four
# requests; the drafter's block falls in the same tier as the target's verify rows, so a warp row also
# runs the drafter's head + selector-gate bank): with the same bench (target head + drafter head and
# gate), 9-16 rows warp (32, 16, 256, 4, 2) and 17-32 rows warp (32, 32, 128, 6, 2) were faster than
# cuBLAS over a step (320 CTAs for the head, 328 for the bank, one resident per SM).
_HEAD_GEMM_ROWS: tuple = (
    (8, ("warp", 32, 8, 128, 6, 2)),
    (16, ("warp", 32, 16, 128, 4, 2)),
)
_HEAD_GEMM_ROWS_RUBIN: tuple = (
    (8, ("warp", 32, 8, 128, 8, 2)),
    (16, ("warp", 32, 16, 256, 4, 2)),
    (32, ("warp", 32, 32, 128, 6, 2)),
)
#: (weight data_ptr, shape, tactic) -> the weight's prepared warp split-K engine (``_head_gemm``).
_HEAD_WARP_ENGINES: dict = {}


def _head_gemm_choice(x: torch.Tensor, weight: torch.Tensor):
    """The ``_HEAD_GEMM_ROWS`` / ``_HEAD_GEMM_ROWS_RUBIN`` choice of ``x @ weight.T``, or None for
    cuBLAS."""
    if not (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.shape[0] >= 1
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.data_ptr() % 32
        and not weight.data_ptr() % 32
        and _sm100_family(x.device)
    ):
        return None
    rows = _HEAD_GEMM_ROWS_RUBIN if _rubin_rows(x.device) else _HEAD_GEMM_ROWS
    return next((c for bound, c in rows if x.shape[0] <= bound), None)


def _head_gemm(x: torch.Tensor, weight: torch.Tensor, choice: tuple) -> torch.Tensor | None:
    """``x @ weight.T`` on ``choice``'s kernel with PDL as the engine sets it; None when a warp
    split-K engine is not prepared yet and a graph is being captured."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    n = weight.shape[0]
    out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    if choice[0] != "warp":
        return bf16_low_m_gemm(x, weight, out, choice, get_env_enable_pdl())
    key = (weight.data_ptr(), tuple(weight.shape), choice)
    engine = _HEAD_WARP_ENGINES.get(key)
    if engine is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        w = _warp()
        tactic = w.WarpSplitKTactic(*choice[1:])
        # One kernel per tactic (dynamic M); the engine keys it by every row count up to 32.
        engine = w.PreparedWarpSplitK(weight, {m: tactic for m in range(1, 33)})
        _HEAD_WARP_ENGINES[key] = engine
    return warp_splitk_gemm(engine, x, out)


def head_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for the vocabulary shard (``FlatVocabParallelHead.local_logits``): the
    ``_HEAD_GEMM_ROWS`` kernel of the device class and row count, else ``bf16_linear``."""
    choice = _head_gemm_choice(x, weight)
    if choice is not None:
        if _rubin_rows(x.device):
            log_rubin_row(
                "H", f"shape={weight.shape[0]}x{weight.shape[1]} rows={x.shape[0]} {choice}"
            )
        out = _head_gemm(x, weight, choice)
        if out is not None:
            return out
    return bf16_linear(x, weight)


def draft_head_selector_bank(head: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
    """``[shard + rank, H]``: this rank's lm_head vocabulary shard ``head [shard, H]`` (rows
    ``tp_rank * shard`` on, as the loader cut them), then the selector's hidden projection
    ``selector [rank, H]`` (whole on every rank). ``bank[:shard]`` replaces the head weight and
    ``bank[shard:]`` the projection: the same values, one allocation."""
    if head.dtype != selector.dtype or head.ndim != 2 or head.shape[1] != selector.shape[1]:
        raise ValueError(
            f"lm_head shard {tuple(head.shape)} {head.dtype} and selector projection "
            f"{tuple(selector.shape)} {selector.dtype} do not stack"
        )
    return torch.cat((head, selector))


def draft_head_selector_gemm(x: torch.Tensor, bank: torch.Tensor) -> Optional[torch.Tensor]:
    """``x @ bank.T`` (``draft_head_selector_bank``) on the head table's kernel for ``x``'s rows:
    ``[rows, shard + rank]``, the shard logits bitwise ``head_linear(x, bank[:shard])`` (a warp
    split-K output's K order does not depend on N) and the selector's gate after them. None where
    the table has no row (the caller runs the head and the gate separately) or a graph is being
    captured before the bank's engine is prepared."""
    choice = _head_gemm_choice(x, bank)
    if choice is None or choice[0] != "warp":
        return None
    if _rubin_rows(x.device):
        log_rubin_row(
            "H", f"shape={bank.shape[0]}x{bank.shape[1]} rows={x.shape[0]} {choice} +selector"
        )
    return _head_gemm(x, bank, choice)


# The DFlash2 selector's hidden projection ``[rows, 7168] x [256, 7168]^T`` (the gate of
# ``select_candidate_tokens``, 7 rows a request): cuBLAS runs it as a GEMM plus a split-K reduce
# kernel, without PDL. (largest row count served, choice) by device class, a choice as
# ``_HEAD_GEMM_ROWS``'s (a ``bf16_low_m_gemm`` tactic or ``("warp", ...)``), launched with PDL as
# the engine sets it. Row counts above the last bound keep cuBLAS. An empty table keeps cuBLAS on
# that device class; the model config's ``selector_gate`` forces one choice. Rubin's one-request
# row (7 rows) is the direct kernel with one 128-column slice a CTA: weights load before the PDL
# wait and there is no reduce kernel; its output equals cuBLAS bit for bit at this shape.
_SELECTOR_GATE_ROWS: tuple = ()
_SELECTOR_GATE_ROWS_RUBIN: tuple = ((7, ("direct", 128, 1)),)


def parse_gemm_choice(spec: str) -> Optional[tuple]:
    """``"cublas"`` -> None; ``"<kind>:<p>:..."`` -> ``(kind, p, ...)``."""
    if spec == "cublas":
        return None
    kind, *params = spec.split(":")
    return (kind, *(int(p) for p in params))


def selector_gate_choice(x: torch.Tensor, weight: torch.Tensor) -> Optional[tuple]:
    """The kernel choice of the selector gate ``x @ weight.T`` (``x`` [rows, H]), or None for
    cuBLAS: the model config's ``selector_gate``, else the device's ``_SELECTOR_GATE_ROWS`` row."""
    if not (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= 32
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.data_ptr() % 32
        and not weight.data_ptr() % 32
        and _sm100_family(x.device)
    ):
        return None
    spec = k31_features().selector_gate
    if spec:
        return parse_gemm_choice(spec)
    rows = _SELECTOR_GATE_ROWS_RUBIN if _rubin_rows(x.device) else _SELECTOR_GATE_ROWS
    return next((c for bound, c in rows if x.shape[0] <= bound), None)


def selector_gate_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for the selector gate (``x`` [..., H]): the
    ``selector_gate_choice`` kernel on the rows, else cuBLAS."""
    rows = x.reshape(-1, x.shape[-1])
    choice = selector_gate_choice(rows, weight)
    if choice is not None:
        if _rubin_rows(x.device):
            log_rubin_row(
                "SEL", f"shape={weight.shape[0]}x{weight.shape[1]} rows={rows.shape[0]} {choice}"
            )
        out = _head_gemm(rows, weight, choice)
        if out is not None:
            return out.view(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight)


# The row-parallel down at up to one request's block pushes its partials into the Lamport slots of
# its all-reduce (bf16_gemm_push), and rubin_lamport_finish completes the call on the transport the
# C++ all-reduce would take: bitwise ``_DRAFT_GEMM_PLAN``'s product + the C++ all-reduce (split_k 1:
# bitwise cuBLAS at this shape). Two kernels in place of GEMM + two-shot all-reduce. GB300 in the
# served graph (us per call, four ranks): down 10.9-11.4 (TGV 18 + two-shot) -> 8.1-9.0
# (push + finish); o_proj 9.5-10.1 -> 9.4-10.2, no gain (the push GEMM at K 768 is 5.5 against
# TGV's 3.7), so o_proj keeps GEMM + all-reduce.
_DRAFT_PUSH_ROWS = 8
_DRAFT_PUSH_TACTIC = ("splitk", 128, 8, 1, 6)
# Rubin (``_rubin_rows``): 64-row tiles, 112 CTAs in place of 56 on 212 SMs, the same K order
# (split_k 1), so the same products bit for bit. One Rubin GPU (the plain split-K kernel at
# [7168, 896], PDL-chained, cold weights): 3.03 -> 2.51 us a call, bitwise.
_DRAFT_PUSH_TACTIC_RUBIN = ("splitk", 64, 8, 1, 12)
_DRAFT_PUSH_SHAPES = frozenset({(7168, 896)})  # down / 16
# CTAs per row of the drafter's conv + add + RMSNorm: the 7168-wide row over a 7-CTA cluster (one
# 8-wide vector per thread, the sum of squares reduced in DSMEM). The residual is bitwise the
# one-CTA kernel's; the norm sums in another order.
_DRAFT_NORM_CTAS = 7


def _draft_push_fits(x: torch.Tensor, weight: torch.Tensor, d: "_DraftRt") -> bool:
    """The push GEMM + finish serve ``x @ weight.T``'s all-reduce (``draft_reduce``)."""
    return (
        d.push is not None
        and tuple(weight.shape) in _DRAFT_PUSH_SHAPES
        and x.shape[0] <= _DRAFT_PUSH_ROWS
        and x.is_contiguous()
    )


def draft_reduce(
    x: torch.Tensor, weight: torch.Tensor, d: "_DraftRt", all_reduce: Any, act: bool = False
) -> torch.Tensor:
    """``all_reduce_tp16(x @ weight.T)``: pushed and finished on ``d.push`` for the
    ``_DRAFT_PUSH_SHAPES`` weights, else GEMM + all-reduce. With ``act`` ``x`` is a gate / up
    output and the product is of ``silu_and_mul(x)``: the push GEMM computes it in its activation
    load (bit for bit), else a separate ``flashinfer_silu_and_mul``."""
    rows = x.shape[0]
    push = d.push
    if _draft_push_fits(x, weight, d):
        one_shot, _ = push.transport(rows)
        out = x.new_empty(rows, weight.shape[0])
        rubin = _rubin_rows(x.device)
        tactic = _DRAFT_PUSH_TACTIC_RUBIN if rubin else _DRAFT_PUSH_TACTIC
        if rubin:
            log_rubin_row("B", f"shape={weight.shape[0]}x{weight.shape[1]} tactic={tactic}")
        bf16_gemm_push(x, weight, out, push.view, one_shot, tactic, push.pdl, act)
        return rubin_lamport_finish(push.view, out, one_shot, push.pdl, False, None, False)
    if act:
        x = flashinfer_silu_and_mul(x)
    return allreduce_module(draft_linear(x, weight), all_reduce, _DRAFT_AR_NONE, None, None)


#: (site, rows, path) already logged (``_log_draft_path``).
_DRAFT_PATHS_LOGGED: set = set()


def _log_draft_path(d: "_DraftRt", site: str, rows: int, path: str) -> None:
    """``K31_DRAFT_PATH site=<site> rows=<rows> path=<path> rank=<r>``: the path a drafter site
    takes, once per process and (site, rows, path), on helix rank 0 and the last helix rank (the
    path is decided on the host, so a captured graph replays the logged one)."""
    rank, last = d.mapping.rank, d.mapping.world_size - 1
    if rank not in (0, last) or (site, rows, path) in _DRAFT_PATHS_LOGGED:
        return
    _DRAFT_PATHS_LOGGED.add((site, rows, path))
    logger.info(f"K31_DRAFT_PATH site={site} rows={rows} path={path} rank={rank}")


def draft_reduce_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.Tensor,
    eps: float,
    block: int,
    d: "_DraftRt",
    all_reduce: Any,
    pdl: bool,
    site: str,
    early: bool = True,
    act: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += Conv(all_reduce_tp16(x @ weight.T); base_kernel, delta)`` (in place) and
    ``(RMSNorm(residual) * norm, residual)``: ``draft_reduce``, then ``dflash2_conv_add_rmsnorm``
    (each row over ``_DRAFT_NORM_CTAS`` CTAs). With ``act`` ``x`` is a gate / up output and
    the product is of ``silu_and_mul(x)`` (``draft_reduce``). ``site`` names the call in
    ``K31_DRAFT_PATH``."""
    rows = x.shape[0]
    path = "separate"
    if act:
        path = "act_fold" if _draft_push_fits(x, weight, d) else "act_separate"
    path += f" ctas={_DRAFT_NORM_CTAS}"
    _log_draft_path(d, site, rows, path)
    a = draft_reduce(x, weight, d, all_reduce, act)
    return dflash2_conv_add_rmsnorm(
        a, delta, base_kernel, residual, norm, eps, block, d.group, pdl, early, _DRAFT_NORM_CTAS
    )


class _DraftLayerW(NamedTuple):
    """One drafter layer at this rank's share: views of ``DraftCore.w["d_l{j}_*"]``."""

    in_norm: torch.Tensor  # [H]
    post_norm: torch.Tensor  # [H]
    q_kv_a: torch.Tensor  # [q_lora + kv_lora + rope, H]: q_a rows, then this layer's kv_a rows
    q_a: torch.Tensor  # [q_lora, H]: rows of q_kv_a
    q_a_norm: torch.Tensor  # [q_lora]
    q_b: torch.Tensor  # [heads * (nope + rope), q_lora]
    kv_a: torch.Tensor  # [kv_lora + rope, H]: rows of d_kv_a
    kv_a_norm: torch.Tensor  # [kv_lora]: a row of d_kv_a_norm
    k_b: torch.Tensor  # [heads, nope, kv_lora]: W_UK
    v_b_t: torch.Tensor  # [heads, kv_lora, v]: W_UV, transposed view
    o: torch.Tensor  # [H, o heads * v]
    gate_up: torch.Tensor  # [2 * I / tp, H]: gate rows, then up rows
    down: torch.Tensor  # [H, I / tp]
    attn_conv_base: torch.Tensor  # [2, taps, H]: side 0 the input's, side 1 the output's
    attn_conv_proj: torch.Tensor  # [2 * taps * groups, H]
    mlp_conv_base: torch.Tensor
    mlp_conv_proj: torch.Tensor


class _DraftW(NamedTuple):
    """The drafter's weights (``DW``)."""

    fc: torch.Tensor  # [H, captures * H]
    layers: tuple  # _DraftLayerW per layer
    final_norm: torch.Tensor


class _DraftStep(NamedTuple):
    """One block step's drafter state (``st``), from the worker's drafter inputs."""

    block: int  # queries per request: max_draft_len + 1
    rows: int  # requests * block
    num_ctx: torch.Tensor  # [B] int64: each request's history length (the block's RoPE base)
    page_table: torch.Tensor  # the draft page table, rows named by table_rows (int32)
    table_rows: torch.Tensor  # [B]: each request's row of page_table
    page_size: int
    positions: torch.Tensor  # [rows]: the queries' positions
    pools: Any  # per drafter layer, this rank's history pool [pages, 1, 1, page, 576]
    # The headtp layout's ring inputs (None otherwise): the step's verify latents [B * width,
    # layers, 576] and their history positions [B * width] (the generation prepare's
    # precompute_context_kv), and the requests' batch rows [B] (their slots' index).
    latents: Any = None
    latent_positions: Any = None
    batch_rows: Any = None
    # headtp with the context overlap (the worker's _ContextOverlap): the step's ring targets,
    # computed in the generation prepare, and the event after which the context stream has stored
    # the verify latents into the ring; layer 0's block RoPE waits for it. None: draft_step
    # computes the targets and stores the latents itself.
    ring_targets: Any = None
    join: Any = None


class _DraftRt(NamedTuple):
    """The drafter's runtime (``rt.draft``), built at the worker's first prepare, before capture."""

    freqs: torch.Tensor  # [positions, rope / 2] complex64: the drafter's YaRN table
    softmax_scale: float
    eps: float
    kv_scale: float  # the FP8 history's dequant scale
    taps: int  # the block convolutions' taps
    groups: int  # channel groups of a convolution's per-position kernel
    group: int  # channels per group
    mapping: Any  # the target's helix mapping: the history's page owners, the exchange's group
    cp_size: int  # its CP ranks
    cp_rank: int  # this rank's place among them
    plan: Any  # the window entry's plan buffers, for up to max_requests requests
    workspace: Any  # the window entry's body partials
    block_kv: torch.Tensor  # [max rows, 1, 576]: the block latents, in the history's dtype
    block_slots: torch.Tensor  # [max rows] int64: block_kv's row of each query
    block_offsets: torch.Tensor  # [max rows] int64: zeros (one latent per scratch row)
    stats: torch.Tensor  # [max rows, heads, 2] FP32: each row's (max, sum) over its local keys
    push: Any  # the TP16 all-reduce's Lamport push workspace (draft_reduce), None without MNNVL
    # The helix_mla layout's packed window and MLA decode workspace for max rows; None: the
    # helix layout's Triton window entry.
    helix_mla: Any = None
    # The headtp layout (plan, workspace, block_kv, block_slots, block_offsets and stats are None
    # there): this rank's window ring (HeadTpRing), its decode's split buffers for max requests,
    # and the worker's batch-row -> context-slot table [max requests] the ring is keyed by.
    ring: Any = None
    headtp: Any = None
    slot_table: Any = None
    # The headtp layout's absorbed query weights, one per layer (``_absorbed_query_weight``).
    q_abs: Any = None
    # The headtp layout's folded query weights (``fold_query_weight``): layers 1.. run the q_a norm,
    # the query projection, its RoPE and the
    # block latent store as one launch (``dflash2_query_fold``); layer 0 holds None (its block store waits for the
    # context overlap's join). None: no layer folds.
    q_fold: Any = None


def _absorbed_query_weight(q_b: torch.Tensor, k_b: torch.Tensor) -> torch.Tensor:
    """``[heads * (kv_lora + rope), q_lora]`` BF16: per head ``W_UK_h^T @ q_b_nope_h`` (FP64, rounded once
    to BF16), then ``q_b_rope_h``, so ``qa @ weight^T`` viewed ``[rows, heads, kv_lora + rope]`` is the
    absorbed query with the rope columns not yet rotated. ``q_b`` is ``[heads * (nope + rope), q_lora]``,
    ``k_b`` (W_UK) ``[heads, nope, kv_lora]``."""
    heads, nope, kv_lora = k_b.shape
    q = q_b.view(heads, -1, q_b.shape[-1])
    latent = torch.matmul(k_b.double().transpose(1, 2), q[:, :nope].double()).to(q_b.dtype)
    return torch.cat((latent, q[:, nope:]), dim=1).reshape(-1, q_b.shape[-1]).contiguous()


#: The query fold's rows: one request's block (the warp split-K token tile).
_QUERY_FOLD_ROWS = 8
#: The query fold's warp split-K tactic over the folded weight [6 heads x 576, q_lora 1536]: the Rubin row of the
#: absorbed query (``_DRAFT_WARP_RUBIN``), on every sm_100 family device.
_QUERY_FOLD_TACTIC = (16, 8, 256, 6, 2)
#: (weight data_ptr, shape) -> the layer's prepared query fold (``_draft_query_fold``).
_DRAFT_QUERY_FOLD_ENGINES: dict = {}


def _draft_query_fold(
    d: "_DraftRt", j: int, L: "_DraftLayerW", q_kv: torch.Tensor, st: "_DraftStep", step: Any
) -> Any:
    """Layer ``j``'s prepared ``PreparedQueryFold`` when the query fold serves this call (``d.q_fold``, the headtp
    ring, at most ``_QUERY_FOLD_ROWS`` packed rows, an FP8 ring), else None. The engine is prepared on the
    first eligible call outside graph capture; while a capture finds none, the call keeps the unfused path."""
    weights = d.q_fold
    if weights is None or weights[j] is None or step is None:
        return None
    pool = d.ring.pools[j][:, 0, 0]
    rows = q_kv.shape[0]
    if not (
        rows <= _QUERY_FOLD_ROWS
        and q_kv.dtype == torch.bfloat16
        and q_kv.is_contiguous()
        and not q_kv.data_ptr() % 32
        and q_kv.shape[1] == L.q_a.shape[0] + L.kv_a.shape[0]
        and pool.dtype == torch.float8_e4m3fn
        and pool.is_contiguous()
        and not pool.data_ptr() % 32
        and st.positions.is_contiguous()
        and step.block_pages.is_contiguous()
        and step.block_offsets.is_contiguous()
        and step.block_pages.dtype == step.block_offsets.dtype
    ):
        return None
    weight = weights[j]
    key = (weight.data_ptr(), tuple(weight.shape))
    engine = _DRAFT_QUERY_FOLD_ENGINES.get(key)
    if engine is None and not torch.cuda.is_current_stream_capturing():
        from ..kernels.spec.rubin_ull_dflash2_query_fold import PreparedQueryFold

        tactic = _warp().WarpSplitKTactic(*_QUERY_FOLD_TACTIC)
        engine = _DRAFT_QUERY_FOLD_ENGINES[key] = PreparedQueryFold(
            weight, L.kv_a_norm, d.eps, tactic
        )
    return engine


class _DraftCoreRt(NamedTuple):
    """What the drafter reads off its core's runtime (``core._rt``)."""

    all_reduce: Any  # the plain TP16 group's engine AllReduce (allreduce_module's)
    draft: Optional[_DraftRt]  # set by DraftModel.bind_runtime


class DraftCore:
    """The drafter's weight table and runtime, held by the target (``FlatSpecTarget.draft_core``).

    ``w``: the drafter's ``d_*`` tensors at this rank's share (weights.py) and ``embed``, the
    target's embedding table (the same storage). ``model_config`` is the target's (its
    ``spec_config``), ``cp_mapping`` the target's helix mapping, ``_rt`` the TP16 all-reduce and the
    drafter's runtime. Not a module: the shell registers the tensors' storages itself.
    """

    def __init__(self, w: dict, model_config: Any, cp_mapping: Any, all_reduce: Any) -> None:
        self.w = w
        self.model_config = model_config
        self.cp_mapping = cp_mapping
        self._rt = _DraftCoreRt(all_reduce, None)


class _Weight(NamedTuple):
    """An object with ``.weight``: what the engine reads off a drafter's ``fc`` or ``hidden_norm``."""

    weight: torch.Tensor


#: ``DraftCore.w`` key for the bank containing the vocabulary shard and selector projection.
DRAFT_HEAD_SELECTOR_BANK = "d_head_sel"


class _CandidateSelector(NamedTuple):
    """The DFlash2 candidate selector's tables, as the engine reads them."""

    predecessor_codebook: torch.Tensor  # [vocab, rank]
    successor_codebook: torch.Tensor  # [vocab, rank]
    hidden_projection_weight: torch.Tensor  # [rank, H]
    top_k: int


class _DraftBackbone(NamedTuple):
    """``draft_model.draft_model_full.model``: the embedding the worker embeds the block with."""

    embed_tokens: Any


class _DraftFull(NamedTuple):
    """``draft_model.draft_model_full``: the backbone and the LM head the drafter shares."""

    model: _DraftBackbone
    lm_head: Any


def _yarn_mscale(scale: float, mscale: float) -> float:
    """YaRN's attention scale factor."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _draft_rope(cfg: Any) -> dict:
    """The drafter's YaRN parameters (``rope_parameters``, else ``rope_scaling``)."""
    scaling = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_type = scaling.get("rope_type") or scaling.get("type")
    assert rope_type == "yarn", f"the DFlash2 drafter's RoPE is YaRN, got {rope_type!r}"
    return dict(
        base=float(scaling.get("rope_theta") or cfg.rope_theta),
        scaling_factor=float(scaling["factor"]),
        original_max_position_embeddings=int(scaling["original_max_position_embeddings"]),
        beta_fast=float(scaling.get("beta_fast", 32.0)),
        beta_slow=float(scaling.get("beta_slow", 1.0)),
        mscale=float(scaling.get("mscale", 1.0)),
        mscale_all_dim=float(scaling.get("mscale_all_dim", 0.0)),
    )


def _draft_freqs(rope: dict, positions: int, dim: int, device: Any) -> torch.Tensor:
    """``[positions, dim / 2]`` complex64: the YaRN rotation as adjacent (re, im) pairs.

    The engine's sinusoid table with its frequency half duplicated, read as complex pairs; mscale
    rides in the modulus.
    """
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=positions, dim=dim, duplicate_data=True, **rope
    )
    table = torch.from_numpy(packed).reshape(positions, dim, 2)
    cos = table[..., 0].contiguous().to(device)
    sin = table[..., 1].contiguous().to(device)
    return torch.complex(cos[..., : dim // 2], sin[..., : dim // 2]).contiguous()


def _draft_views(w: dict, layers: int, nope: int, kv_lora: int) -> _DraftW:
    """The drafter's views over ``DraftCore.w`` (weights.py lays out every row as read here)."""
    kv_a, kv_a_norm = w["d_kv_a"], w["d_kv_a_norm"]
    latent = kv_a.shape[0] // layers
    views = []
    for j in range(layers):
        kv_b = w[f"d_l{j}_kv_b"]  # [every head's k_nope rows ; every head's v rows]
        heads = kv_b.shape[0] // (2 * nope)
        views.append(
            _DraftLayerW(
                in_norm=w[f"d_l{j}_in_norm"],
                post_norm=w[f"d_l{j}_post_norm"],
                q_kv_a=w[f"d_l{j}_q_kv_a"],
                q_a=w[f"d_l{j}_q_kv_a"][: w[f"d_l{j}_q_a_norm"].shape[0]],
                q_a_norm=w[f"d_l{j}_q_a_norm"],
                q_b=w[f"d_l{j}_q_b"],
                kv_a=kv_a[j * latent : (j + 1) * latent],
                kv_a_norm=kv_a_norm[j],
                k_b=kv_b[: heads * nope].view(heads, nope, kv_lora),
                v_b_t=kv_b[heads * nope :].view(heads, -1, kv_lora).transpose(1, 2),
                o=w[f"d_l{j}_o"],
                gate_up=w[f"d_l{j}_gate_up"],
                down=w[f"d_l{j}_down"],
                attn_conv_base=w[f"d_l{j}_attn_conv_base"],
                attn_conv_proj=w[f"d_l{j}_attn_conv_proj"],
                mlp_conv_base=w[f"d_l{j}_mlp_conv_base"],
                mlp_conv_proj=w[f"d_l{j}_mlp_conv_proj"],
            )
        )
    return _DraftW(w["d_fc"], tuple(views), w["d_final_norm"])


def draft_config_for(model_config: Any) -> Any:
    """The DFlash2 drafter checkpoint's ``ModelConfig``, as the engine's KV cache creator reads it.

    A copy of the target's config over the drafter's ``config.json``: the mapping repurposed to the
    plain TP16 group, an FP8 KV-cache quantization (the history's dtype comes from the container,
    ``context_kv_fp8``) and no speculative config. The drafter's architecture is weights.py's
    (``DRAFT_SHAPE``, ``DRAFT_DFLASH``); another one raises here.
    """
    spec_config = model_config.spec_config
    mode = spec_config.spec_dec_mode
    if not (mode.use_one_engine() and mode.is_dflash()):
        raise ValueError(
            "This target drafts with one-engine DFlash (the DFlash2 MLA drafter) only."
        )
    # The engine's config loader (what ``ModelConfig.from_pretrained`` reads an external drafter
    # with): the checkpoint's ``model_type`` class, so transformers 5's RoPE standardization finds
    # ``max_position_embeddings``; a bare ``PretrainedConfig.from_dict`` fails there.
    cfg = load_pretrained_config(spec_config.speculative_model, trust_remote_code=True)
    for key, want in DRAFT_SHAPE.items():
        got = getattr(cfg, key, None)
        assert got == want, f"the DFlash2 drafter's {key} is {got}, this target runs {want}"
    dflash = getattr(cfg, "dflash_config", None) or {}
    for key, want in DRAFT_DFLASH.items():
        assert dflash.get(key) == want, f"dflash_config.{key} {dflash.get(key)} is not {want}"
    layer_types = getattr(cfg, "layer_types", None) or []
    assert layer_types and all(t == "sliding_attention" for t in layer_types), layer_types
    assert getattr(cfg, "is_causal", True), "the DFlash2 drafter's block attention is causal"
    draft_config = copy.copy(model_config)
    draft_config._frozen = False
    draft_config.pretrained_config = cfg
    draft_config.mapping = model_config.mapping.repurpose_helix_cp_to_tp()
    draft_config.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    draft_config.quant_config_dict = None
    draft_config.spec_config = None
    draft_config._frozen = True
    return draft_config


class DraftModel(nn.Module):
    """The DFlash2 drafter the engine's worker drives (Helix CP16: 96 heads, page-sharded history).

    No parameter or submodule of its own: every weight is ``DraftCore.w["d_*"]``, read through the
    core (held weakly). ``model_config`` is the drafter's ``ModelConfig`` (``draft_config_for``);
    the speculative config is the core's. ``lm_head`` and ``logits_processor`` are the target's
    (``FlatLanguageHead``, ``FlatLogitsProcessor``): the drafter's logits are the target head's
    (its vocabulary shard under the worker's vocab-parallel selection). This target is a
    generation server, so the drafter is whole (``context_kv_only`` refused); the history-only
    methods serve the engine's context-server contract.
    """

    # The engine's unified draft cache holds the history, in aggregated runs too.
    supports_unified_draft_cache = True
    unified_draft_cache_when_aggregated = True
    # One latent per token and layer (no V half), in pages the block decode reads itself.
    _kv_factor = 1
    _num_kv_heads = 1
    _paged_ctx_cache = True
    _uses_worker_attention_backend = False
    _uses_lilicorr = False
    lilicorr = None
    is_dflash2 = True
    dflash_attention_backend = "VANILLA"
    mask_token_embedding = None

    def __init__(
        self,
        core: DraftCore,
        model_config: Any,
        lm_head: Any,
        logits_processor: Any,
    ) -> None:
        super().__init__()
        spec_config = core.model_config.spec_config
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self._core = weakref.ref(core)
        # The target's head, held outside the module tree.
        self._head = (lm_head, logits_processor)
        cfg = self.config
        dflash = cfg.dflash_config
        self.context_kv_only = bool(getattr(spec_config, "context_kv_only", False))
        history = getattr(spec_config, "context_kv_dtype", None)
        assert history in _DRAFT_HISTORY_CONFIGS, (
            f"this target keeps the drafter history in {_DRAFT_HISTORY_DTYPE}: "
            f"speculative_config.context_kv_dtype {history!r} names another storage"
        )
        # The engine sizes the history pool from this flag.
        self.context_kv_fp8 = _DRAFT_HISTORY_DTYPE == torch.float8_e4m3fn
        # The runtime block (the checkpoint's trained block_size is not the engine's).
        self.block_size = spec_config.max_draft_len + 1
        self.mask_token_id = dflash["mask_token_id"]
        self.context_cache_window = int(cfg.sliding_window)
        self._num_attn_layers = cfg.num_hidden_layers
        # The head-TP runtime attends with each rank's loaded q_b / kv_b heads over the
        # replicated window ring.
        self._num_heads = cfg.num_attention_heads
        # headtp: the generation prepare's verify latents and their positions
        # (precompute_context_kv), stored into the ring by the next dflash_forward.
        self._step_latents: Optional[tuple] = None
        # headtp, the worker's context overlap for the next dflash_forward (_ContextOverlap: the
        # step's ring targets and the context stream's join event); None: the serial store.
        self._context_overlap: Any = None
        # The history is page-sharded over the target's own helix mapping, so the KV cache creator
        # shards the draft pages as the target's and the exchange reuses the target's group.
        self._helix_mapping = core.cp_mapping
        assert self._helix_mapping.has_cp_helix(), "the drafter shards its history by helix page"
        self._head_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        self._rope = _draft_rope(cfg)
        mscale = _yarn_mscale(self._rope["scaling_factor"], self._rope["mscale_all_dim"])
        self.softmax_scale = (
            mscale * mscale / math.sqrt(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        )
        self._selector_top_k = int(dflash["selector_top_k"])
        self._check_head_selector_bank()
        # Written by the worker before any table is sized: the positions it serves.
        self._runtime_position_ceiling = None
        self._dw: Optional[_DraftW] = None
        self._freqs: Optional[torch.Tensor] = None
        self._rt: Any = None
        # fc's row as one all-reduce of this rank's columns (bind_runtime; None: the all-gather).
        self._fc_gather: Optional[ZeroPaddedReduce] = None
        logger.info(
            f"DFlash2 drafter: {self._num_attn_layers} layers, {self._num_heads} heads per rank, "
            f"block {self.block_size}, window {self.context_cache_window}, FP8 history, "
            f"{'context KV only' if self.context_kv_only else 'full'}"
        )

    # -- what the engine reads -------------------------------------------------

    def _w(self, key: str) -> torch.Tensor:
        return self._core().w[key]

    @property
    def lm_head(self) -> Any:
        return self._head[0]

    @property
    def logits_processor(self) -> Any:
        return self._head[1]

    @property
    def fc(self) -> _Weight:
        return _Weight(self._w("d_fc"))

    @property
    def hidden_norm(self) -> _Weight:
        return _Weight(self._w("d_hidden_norm"))

    @property
    def has_candidate_selector(self) -> bool:
        return not self.context_kv_only

    @property
    def candidate_selector(self) -> Optional[_CandidateSelector]:
        if self.context_kv_only:
            return None
        return _CandidateSelector(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            self._selector_top_k,
        )

    @property
    def head_selector_bank(self) -> Optional[torch.Tensor]:
        """``[shard + rank, H]``: the head shard and the selector projection in one weight
        (``draft_head_selector_bank``), or None when they are loaded apart."""
        if self.context_kv_only:
            return None
        return self._core().w.get(DRAFT_HEAD_SELECTOR_BANK)

    def _check_head_selector_bank(self) -> None:
        """A bank must hold the head's weight and the selector projection as its two row blocks
        (the same storage, in that order): the head GEMM over it then gives the head's logits."""
        bank = self._core().w.get(DRAFT_HEAD_SELECTOR_BANK)
        if bank is None:
            return
        head = self.lm_head.weight.weight
        selector = self._w("d_sel_hidden_proj")
        shard = head.shape[0]
        if not (
            bank.is_contiguous()
            and bank.shape == (shard + selector.shape[0], head.shape[1])
            and _same_view(head, bank[:shard])
            and _same_view(selector, bank[shard:])
        ):
            raise RuntimeError(
                "the drafter head bank is not [the head's shard ; the selector projection] in one "
                "storage"
            )
        logger.info(
            f"K31_DRAFT_HEAD_SELECTOR bank={bank.shape[0]}x{bank.shape[1]} shard={shard} "
            f"selector_rows={selector.shape[0]} (one head GEMM gives the logits and the gate)"
        )

    @property
    def draft_model_full(self) -> _DraftFull:
        return _DraftFull(_DraftBackbone(_DraftEmbedding(self._core())), self.lm_head)

    @property
    def model(self) -> _DraftBackbone:
        return self.draft_model_full.model

    def _build_fused_kv_buffers(self) -> None:
        """Nothing to build: weights.py stacks every layer's latent projection (``d_kv_a``)."""

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to take: the embedding and the LM head are the target's, read through the core."""
        del target_model

    def bind_runtime(
        self,
        max_rows: int,
        page_size: int,
        slots: Optional[int] = None,
        slot_table: Optional[torch.Tensor] = None,
    ) -> None:
        """Read the weight views and build the drafter's runtime, before any capture.

        The worker calls it once its history buffers are bound (and again when the KV cache
        manager is replaced); ``_runtime_position_ceiling`` sizes the RoPE table, ``max_rows``
        (requests x block) and the history's ``page_size`` the window entry's buffers. The headtp
        layout's ring holds ``slots`` context slots (the worker's, its dummy slot included),
        addressed through the worker's ``slot_table`` (batch row -> slot).
        """
        core = self._core()
        cfg = self.config
        self._dw = _draft_views(
            core.w, cfg.num_hidden_layers, cfg.qk_nope_head_dim, cfg.kv_lora_rank
        )
        positions = self._runtime_position_ceiling or cfg.max_position_embeddings
        device = self._dw.fc.device
        draft = self._headtp_runtime(max_rows, page_size, slots, slot_table, positions, device)
        self._finish_bind(draft, max_rows, device)

    def _headtp_runtime(
        self,
        max_rows: int,
        page_size: int,
        slots: Optional[int],
        slot_table: Optional[torch.Tensor],
        positions: int,
        device: Any,
    ) -> "_DraftRt":
        """The headtp layout's runtime: the ring, the decode's buffers, no helix window state."""
        core = self._core()
        cfg = self.config
        dflash = cfg.dflash_config
        if slots is None or slot_table is None:
            raise RuntimeError("the headtp drafter's ring needs the worker's slots and slot table")
        layer = self._dw.layers[0]
        heads = layer.k_b.shape[0]
        if heads * layer.v_b_t.shape[-1] != layer.o.shape[1]:
            raise RuntimeError(
                f"headtp: {heads} q_b / kv_b heads, o_proj takes {layer.o.shape[1]} columns of "
                f"{layer.v_b_t.shape[-1]} per head; weights.py loaded another layout"
            )
        block = self.block_size
        ring = build_ring(
            cfg.num_hidden_layers,
            slots,
            _DRAFT_WINDOW,
            block,
            block,
            page_size,
            self._head_dim,
            _DRAFT_HISTORY_DTYPE,
            device,
        )
        logger.info(
            f"K31_DRAFT_ATTENTION layout=headtp heads={heads} ring_rows={ring.rows} slots={slots} "
            f"ring_mib={ring.data.numel() * ring.data.element_size() / 2**20:.1f} "
            "merge=ranked "
            "(replicated FP8 window ring, shared-key decode, no exchange)"
        )
        # Layer 0's block store waits for the context overlap's join, which the side chain reaches after the
        # layer's query projection would: it keeps the unfused path.
        fold = _sm100_family(torch.device(device))
        folds = tuple(fold and j > 0 for j in range(len(self._dw.layers)))
        logger.info(
            f"K31_DRAFT_QUERY_FOLD layers={[j for j, f in enumerate(folds) if f]} "
            "(q_a norm + absorbed query + RoPE + block store, one launch "
            f"at up to {_QUERY_FOLD_ROWS} rows)"
        )
        return _DraftRt(
            freqs=_draft_freqs(self._rope, int(positions), cfg.qk_rope_head_dim, device),
            softmax_scale=self.softmax_scale,
            eps=float(cfg.rms_norm_eps),
            kv_scale=DRAFT_FP8_KV_SCALE,
            taps=int(dflash["conv_kernel_size"]),
            groups=cfg.hidden_size // int(dflash["conv_group_size"]),
            group=int(dflash["conv_group_size"]),
            mapping=self._helix_mapping,
            cp_size=self._helix_mapping.cp_size,
            cp_rank=self._helix_mapping.cp_rank,
            plan=None,
            workspace=None,
            block_kv=None,
            block_slots=None,
            block_offsets=None,
            stats=None,
            # Collective (sizes the shared MNNVL workspace): every rank binds at the same point.
            push=rubin_lamport_push_workspace(
                core._rt.all_reduce, _DRAFT_PUSH_ROWS, cfg.hidden_size
            ),
            ring=ring,
            headtp=headtp_ranked_workspace(max_rows // block, block, heads, device),
            slot_table=slot_table,
            q_abs=tuple(_absorbed_query_weight(L.q_b, L.k_b) for L in self._dw.layers),
            q_fold=(
                tuple(
                    fold_query_weight(L.q_b, L.k_b, L.q_a_norm) if folds[j] else None
                    for j, L in enumerate(self._dw.layers)
                )
                if any(folds)
                else None
            ),
        )

    def _finish_bind(self, draft: "_DraftRt", max_rows: int, device: Any) -> None:
        core = self._core()
        cfg = self.config
        logger.info(
            "K31_DRAFT_AR_NORM separate (o_proj / down: all-reduce, then conv + add + norm)"
        )
        # Also an attribute: the engine counts a context server's drafter tables off the module.
        self._freqs = draft.freqs
        self._rt = core._rt._replace(draft=draft)
        self._fc_gather = None
        local_rows = self._dw.fc.shape[0]
        all_reduce = core._rt.all_reduce
        group = tuple(getattr(getattr(all_reduce, "mapping", None), "tp_group", ()))
        if local_rows != cfg.hidden_size and group == self.lm_head.group:
            rank = group.index(all_reduce.mapping.rank)
            self._fc_gather = ZeroPaddedReduce.build(
                all_reduce,
                rank * local_rows,
                local_rows,
                cfg.hidden_size,
                max_rows,
                self._dw.fc.dtype,
                device,
            )

    # -- the history -----------------------------------------------------------

    def _fc_rows(self, local: torch.Tensor) -> torch.Tensor:
        """``fc``'s full output row from this rank's: ``d_fc`` holds the rank's sixteenth of the
        output rows (weights.py), gathered over the plain TP16 group (the target head's, in whose
        order the shares lie): a zero-padded all-reduce on MNNVL up to the runtime's rows, else the
        all-gather. A whole ``d_fc`` is returned as is."""
        if local.shape[-1] == self.config.hidden_size:
            return local
        rows = local.shape[0]
        gather = self._fc_gather
        if gather is not None and gather.fits(rows, local.shape[-1]):
            return gather(local, rows)
        return gather_columns(local, self.lm_head.group)

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(captures))``: the target's captured features in the drafter's space.

        Each output element is one dot product over all captures, on this rank's rows of ``fc``.
        """
        fc = self._w("d_fc")
        return flashinfer_rmsnorm(
            self._fc_rows(draft_linear(hidden_states.to(fc.dtype).contiguous(), fc)),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    def can_stage_target_hidden(self, rows: int) -> bool:
        """Whether ``fc`` of ``rows`` verify rows goes through the zero-padded all-reduce (the
        context overlap's exchange); else the step keeps the serial chain."""
        fc = self._w("d_fc")
        gather = self._fc_gather
        return (
            fc.shape[0] != self.config.hidden_size
            and gather is not None
            and gather.fits(rows, fc.shape[0])
        )

    def stage_target_hidden(self, hidden_states: torch.Tensor) -> None:
        """The context overlap's first half: this rank's ``fc`` rows of ``hidden_states`` (the
        step's captured verify rows) into the gather's zero-padded buffer, no exchange."""
        fc = self._w("d_fc")
        self._fc_gather.stage(
            draft_linear(hidden_states.to(fc.dtype).contiguous(), fc), hidden_states.shape[0]
        )

    def finish_target_hidden(self, rows: int) -> torch.Tensor:
        """The context overlap's second half: ``hidden_norm`` of the staged ``fc`` row all-reduced
        over the group; ``project_target_hidden`` of the staged rows."""
        return flashinfer_rmsnorm(
            self._fc_gather.reduce(rows),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    def precompute_context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor):
        """Every layer's history latent: ``([N, layers, 1, 576], None)``.

        One projection over the stacked ``kv_a`` rows, then [RMSNorm | RoPE] of all layers in one
        launch.
        """
        kv_a = self._w("d_kv_a")
        latent = draft_linear(projected_hidden.to(kv_a.dtype).contiguous(), kv_a)
        latent = latent.view(projected_hidden.shape[0], self._num_attn_layers, self._head_dim)
        out = dflash2_mla_ctx_norm_rope(
            latent,
            self._w("d_kv_a_norm"),
            float(self.config.rms_norm_eps),
            self._rt.draft.freqs,
            positions.view(-1),
        )
        if self._rt is not None and self._rt.draft.ring is not None:
            # The generation prepare's call is the last before dflash_forward (a prefill's comes
            # first in the step): the ring's verify store reads these.
            self._step_latents = (out, positions.view(-1))
        return out.unsqueeze(2), None

    def capture_projection_blocks(self, num_capture_layers: int) -> list[torch.Tensor]:
        """``fc``'s ``[out, hidden]`` column block per captured layer (views of ``d_fc``: this rank's
        output rows)."""
        fc = self._w("d_fc")
        width, rest = divmod(fc.shape[1], num_capture_layers)
        assert not rest, f"fc's {fc.shape[1]} inputs do not split over {num_capture_layers} layers"
        return [fc[:, i * width : (i + 1) * width] for i in range(num_capture_layers)]

    def finish_capture_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """``hidden_norm`` of the FP32 sum of the blocks' products, rounded once to fc's dtype (and
        gathered over the group, as ``project_target_hidden``)."""
        return flashinfer_rmsnorm(
            self._fc_rows(projected.to(self._w("d_fc").dtype)),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    # -- the block -------------------------------------------------------------

    def select_candidate_path(
        self, block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
    ):
        """The selector's greedy walk: ``block_logits`` rewritten in place to the walked path."""
        return k31_dflash2_selector_greedy_path(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_logits,
        )

    def select_candidate_tokens(
        self,
        candidate_ids,
        unary_logits,
        block_hidden_states,
        anchor_token_ids,
        block_dtype,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The greedy walk's tokens, int32 ``[n, K]``: the argmax of the block
        ``select_candidate_path`` writes into a ``-inf`` block of ``block_dtype``, without the
        block. ``gate``: the selector's gate from the head GEMM over ``head_selector_bank``
        (``[n, K, rank]``, read in place); None: ``selector_gate_linear`` computes it."""
        return k31_dflash2_selector_greedy_tokens(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_dtype,
            gate_linear=selector_gate_linear,
            gate=gate,
        )

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
        block_placement=None,
    ):
        """The block decode of ``[B, block, hidden]`` noise embeddings; ``[B * block, hidden]``."""
        del ctx_k_cache, ctx_v_cache, block_placement
        batch, block, hidden = noise_embedding.shape
        ring = None
        overlap, self._context_overlap = self._context_overlap, None
        if self._rt.draft.ring is not None and overlap is not None:
            ring = dict(ring_targets=overlap.step, join=overlap.join)
        elif self._rt.draft.ring is not None:
            latents, latent_positions = self._step_latents
            if latents.shape[0] != batch * self._rt.draft.ring.width:
                raise RuntimeError(
                    f"headtp: {latents.shape[0]} verify latents for {batch} requests; the ring's "
                    "store needs the generation prepare's"
                )
            ring = dict(
                zip(
                    ("latents", "latent_positions", "batch_rows"),
                    (latents, latent_positions, ctx_cache_batch_idx[:batch]),
                )
            )
        st = _DraftStep(
            block=block,
            rows=batch * block,
            num_ctx=num_ctx_per_req[:batch].long(),
            page_table=ctx_page_table,
            table_rows=ctx_cache_batch_idx,
            page_size=ctx_kv_cache[0].shape[-2],
            positions=query_positions.reshape(-1),
            pools=ctx_kv_cache,
            **(ring or {}),
        )
        return draft_step(self._dw, noise_embedding.reshape(batch * block, hidden), st, self._rt)


def _draft_window_plan(st: _DraftStep, d: _DraftRt, block: int) -> Any:
    """Helix layout, once per step: the window's body, head and block keys this rank owns."""
    return rubin_dflash2_window_plan(
        st.num_ctx,
        st.page_table,
        st.table_rows,
        block,
        _DRAFT_WINDOW,
        st.page_size,
        d.cp_size,
        d.cp_rank,
        _DRAFT_WINDOW_SPLITS,
        d.plan,
    )


def _draft_attention_helix(
    L: "_DraftLayerW",
    j: int,
    qa: torch.Tensor,
    q_kv: torch.Tensor,
    plan: Any,
    st: _DraftStep,
    d: _DraftRt,
    block_kv: torch.Tensor,
    stats: torch.Tensor,
    eps: float,
    mla: Any = None,
) -> torch.Tensor:
    """Helix layout: every head over this rank's keys (partial and log-sum-exp), W_UV, then the
    helix exchange to this rank's heads; ``[rows, heads * v]``. With ``mla`` (the helix_mla
    layout's buffers) the window is packed to BF16 and attended by the MLA decode, its empty rows
    masked in the exchange."""
    rows = qa.shape[0]
    heads, nope, kv_lora = L.k_b.shape
    latent = L.kv_a.shape[0]
    q = draft_linear(qa, L.q_b)
    q = q.view(rows, heads, -1)
    query = q.new_empty(rows, heads, latent)
    bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
    dflash2_mla_block_rope(
        q,
        query,
        q_kv[:, L.q_a.shape[0] :],
        L.kv_a_norm,
        eps,
        d.freqs,
        st.positions,
        block_kv,
        d.block_slots[:rows],
        d.block_offsets[:rows],
    )
    empty = None
    if mla is not None:
        ctx, stats, empty = rubin_ull_dflash2_helix_window(
            query,
            st.pools[j][:, 0, 0],
            st.page_table,
            st.table_rows,
            st.num_ctx,
            block_kv,
            st.block,
            d.softmax_scale,
            d.cp_size,
            d.cp_rank,
            mla,
        )
    else:
        ctx = rubin_dflash2_window_triton(
            query,
            st.pools[j][:, 0, 0],
            block_kv[:, 0],
            plan,
            d.softmax_scale,
            kv_lora,
            d.kv_scale,
            d.workspace,
            stats,
            tiles=_DRAFT_WINDOW_TILES,
        )
    v_dim = L.v_b_t.shape[-1]
    partial = ctx.new_empty(rows, heads, v_dim)
    bmm_out(ctx.transpose(0, 1), L.v_b_t, partial.transpose(0, 1))
    return rubin_dflash2_helix_combine(partial, stats, d.mapping, heads, v_dim, zero_kv_mask=empty)


def _ranked_tiles(device: torch.device, requests: int) -> tuple:
    """The ranked partial's (BC, BK, warps, splits, maxnreg) on ``device`` for ``requests`` requests."""
    if not _rubin_rows(device):
        return RANKED_TILES
    return _DRAFT_RANKED_TILES_RUBIN.get(requests, _DRAFT_RANKED_TILES_RUBIN_REST)


def _draft_attention_headtp(
    L: "_DraftLayerW",
    j: int,
    qa: Optional[torch.Tensor],
    q_kv: torch.Tensor,
    st: _DraftStep,
    d: _DraftRt,
    step: Any,
    eps: float,
    *,
    join: Any = None,
    fold: Any = None,
) -> torch.Tensor:
    """headtp layout: this rank's heads (``L.q_b`` / ``L.k_b`` / ``L.v_b_t`` hold only theirs) over
    every key of the window, read from the ring; ``[rows, heads * v]``, the o_proj input. The block
    RoPE writes the block's latents into the ring at ``step``'s block targets before the decode
    reads them; no exchange. ``join`` (layer 0 under the context overlap): the event after the
    context stream's verify store, waited for right before the block RoPE: the block overwrites
    the rejected verify rows, so the verify store must land first (not only before the decode). ``fold``
    (``_draft_query_fold``): one launch takes ``q_kv`` to the query and the block store (``qa`` is unused)."""
    rows = q_kv.shape[0]
    heads, nope, kv_lora = L.k_b.shape
    latent = L.kv_a.shape[0]
    pool = d.ring.pools[j][:, 0, 0]  # [slots * pages, page, 576]
    q_abs = getattr(d, "q_abs", None)
    if fold is not None:
        if join is not None:
            raise ValueError("the query fold stores the block before the context overlap's join")
        query, _ = dflash2_query_fold(
            fold, q_kv, d.freqs, st.positions, pool, step.block_pages, step.block_offsets
        )
    elif q_abs is not None:
        # The absorbed query in one GEMM; the block RoPE rotates its rope columns in place (q's rope
        # columns are query's: each element is read, then written, by one thread).
        query = draft_linear(qa, q_abs[j]).view(rows, heads, latent)
        q = query[..., kv_lora - nope :]
    else:
        q = draft_linear(qa, L.q_b).view(rows, heads, -1)
        query = q.new_empty(rows, heads, latent)
        bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
    if fold is None:
        if join is not None:
            torch.cuda.current_stream(q.device).wait_event(join)
        dflash2_mla_block_rope(
            q,
            query,
            q_kv[:, L.q_a.shape[0] :],
            L.kv_a_norm,
            eps,
            d.freqs,
            st.positions,
            pool,
            step.block_pages,
            step.block_offsets,
        )
    # helix_mla's rounding order (the drafter's acceptance length depends on it): per helix page
    # group a normalized BF16 partial times W_UV, rounded to BF16, and an FP32 log-sum-exp merge
    # rounded once.
    tiles = _ranked_tiles(query.device, rows // st.block)
    if _rubin_rows(query.device):
        log_rubin_row("D", f"rows={query.shape[0]} tiles={tiles}")
    part, lse = headtp_decode_ranked(
        query,
        pool.view(-1, latent),
        step.slots,
        st.num_ctx,
        d.ring.rows,
        d.softmax_scale,
        d.kv_scale,
        _DRAFT_WINDOW,
        L.v_b_t,
        d.headtp,
        tiles,
    )
    return headtp_ranked_lse_merge(part, lse, st.block)


def draft_step(
    DW: _DraftW, hidden_streams: torch.Tensor, st: _DraftStep, rt: _DraftCoreRt
) -> torch.Tensor:
    """One DFlash2 block over the five drafter layers; returns ``[B * block, hidden]``.

    ``hidden_streams`` is the noise block's embedding. Once per step the window plan of this rank's
    keys. Per layer: input norm, the attention conv's input side, [q_a | kv_a], the attention
    (``_draft_attention_helix``) to this rank's 6 heads, ``o_proj`` and its all-reduce, the conv's
    output side with the residual add and the next norm; then the MLP between the MLP conv's two
    sides. The last layer's output side ends in the final norm. The headtp layout instead stores the
    step's verify latents into the ring once, and attends per layer with ``_draft_attention_headtp``.
    """
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    d = rt.draft
    eps = d.eps
    rows, block = st.rows, st.block
    step = mla = plan = block_kv = stats = None
    if d.ring is not None and st.join is not None:
        # headtp with the context overlap: the targets came with the step, and the context stream
        # stores the verify latents; layer 0's block RoPE joins it.
        step = st.ring_targets
    elif d.ring is not None:
        # headtp: the step's ring targets, then the verify latents into every layer's ring.
        step = ring_step(d.ring, d.slot_table, st.batch_rows, st.latent_positions, st.num_ctx)
        ring_store(d.ring, st.latents, step.verify_pages, step.verify_offsets)
    else:
        mla = d.helix_mla if d.helix_mla is not None and d.helix_mla.rows == rows else None
        plan = _draft_window_plan(st, d, block) if mla is None else None
        block_kv = d.block_kv[:rows]
        stats = d.stats[:rows]

    h = hidden_streams
    residual = h.clone()  # the conv + add + norm below updates it in place
    # The convolutions and norms launch with PDL when the GEMMs do, and release the next kernel
    # right after their wait: every reader of their output (the projections' GEMMs, the next
    # convolution or norm) waits before reading it. The final norm keeps the release at its end,
    # as its readers are the engine's.
    pdl = get_env_enable_pdl()
    x = dflash2_rmsnorm(h, DW.layers[0].in_norm, eps, pdl)
    for j, L in enumerate(DW.layers):
        coef, x = draft_conv_proj(
            x, L.attn_conv_proj, L.attn_conv_base[0], block, d, pdl, "attn_conv"
        )
        q_kv = draft_linear(x, L.q_kv_a)  # [rows, q_lora | kv_lora + rope]
        fold = _draft_query_fold(d, j, L, q_kv, st, step)
        if fold is not None:
            # The q_a norm, the query, its RoPE and the block store in one launch (no qa).
            _log_draft_path(d, "query", rows, "query_fold")
            attn = _draft_attention_headtp(L, j, None, q_kv, st, d, step, eps, fold=fold)
        elif step is not None:
            qa = dflash2_rmsnorm(q_kv[:, : L.q_a.shape[0]], L.q_a_norm, eps, pdl)
            _log_draft_path(d, "attention", rows, "headtp")
            join = st.join if j == 0 else None
            attn = _draft_attention_headtp(L, j, qa, q_kv, st, d, step, eps, join=join)
        else:
            qa = dflash2_rmsnorm(q_kv[:, : L.q_a.shape[0]], L.q_a_norm, eps, pdl)
            _log_draft_path(d, "attention", rows, "helix" if mla is None else "helix_mla")
            attn = _draft_attention_helix(L, j, qa, q_kv, plan, st, d, block_kv, stats, eps, mla)
        x, residual = draft_reduce_norm(
            attn,
            L.o,
            coef[:, 1],
            L.attn_conv_base[1],
            residual,
            L.post_norm,
            eps,
            block,
            d,
            rt.all_reduce,
            pdl,
            "o_proj",
        )
        coef, x = draft_conv_proj(x, L.mlp_conv_proj, L.mlp_conv_base[0], block, d, pdl, "mlp_conv")
        last = j + 1 == len(DW.layers)
        gate_up = draft_linear(x, L.gate_up)
        x, residual = draft_reduce_norm(
            gate_up,
            L.down,
            coef[:, 1],
            L.mlp_conv_base[1],
            residual,
            DW.final_norm if last else DW.layers[j + 1].in_norm,
            eps,
            block,
            d,
            rt.all_reduce,
            pdl,
            "down_last" if last else "down",
            not last,
            act=True,
        )
    return x


_CONTEXT_STREAMS: dict = {}


def context_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device for the headtp drafter's context chain (_ContextOverlap)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _CONTEXT_STREAMS.get(index)
    if stream is None:
        stream = _CONTEXT_STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


_SEQ_LENS_STREAMS: dict = {}


def seq_lens_copy_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device for the fused select-accept's sequence-lengths copy
    (``DFlashWorker._prepare_attn_metadata_off_path``)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SEQ_LENS_STREAMS.get(index)
    if stream is None:
        stream = _SEQ_LENS_STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


class _ContextOverlap(NamedTuple):
    """The headtp drafter's context chain on its own stream, one generation step.

    The chain is the step's verify rows into the drafter's history: ``fc`` (this rank's rows), the
    zero-padded all-reduce, ``hidden_norm``, ``kv_a``, [RMSNorm | RoPE], the ring's verify store
    and the owner pages' store. None of it depends on the acceptance except the owner store's
    mask and the RoPE positions, which the generation prepare's kernel writes. So:

    * at the forward's start (the captures and the target head are queued), the context stream
      forks and stages ``fc`` (``DraftModel.stage_target_hidden``: no exchange);
    * the generation prepare (``_prepare_paged_latent_gen_inputs``) runs its kernel and the ring
      targets on this stream, then forks the rest; this stream goes on with the noise embedding
      and the drafter's layer 0;
    * layer 0's block RoPE joins (``_draft_attention_headtp``): it overwrites the rejected verify
      rows, so the verify store lands first.

    The all-reduce is the engine's on the group's MNNVL workspace, which is keyed by the mapping
    only: the pair exchange (and the sampled path's pairs gather) use the same Lamport buffers.
    Every one of them is queued on this stream before the second fork, and nothing between the
    fork and the join on this stream all-reduces, so the users stay in one order.
    """

    stream: torch.cuda.Stream
    rows: int  # the step's verify rows: requests * width
    step: Any = None  # the ring targets (RingStep), set at the second fork
    join: Any = None  # the context stream's event after both stores


class DFlashWorker(_EngineDFlashWorker):
    """The engine's DFlash worker with this target's DFlash2 differences.

    Every method not defined here is the engine's: the managed draft cache and its history, the
    paged history stores, the fused generation prepare. Defined here: the unified-cache check and
    the drafter's runtime (``_lazy_init_ctx_buffers``), the target-metadata hook, the
    vocab-parallel selection over the native head (``bind_target_lm_head``), acceptance and
    sampling through the family's spec entries, the forced-acceptance count, the DFlash2 sampled
    path (selector walk, block verification), ``_forward_impl``, and the step shell's pieces: the
    managed draft cache's uploads staged into the step's input commit (through the engine's
    preparation, whose methods the family rules keep inherited) and the fused select-accept and
    rewind.
    """

    # prepare_managed_draft_cache stages its uploads into the step's input commit when the engine
    # runs it inside the commit step (the decoder runner).
    stages_managed_draft_cache = True
    # The selector stores global candidate probabilities for block verification.
    supports_cp_rejection_sampling = True
    # The sampled path's state, all None until its first (eager) step makes the proposal store:
    # the step's proposal rows for the agreement (_sample_block_path sets them, the agreement takes
    # them), the agreement's device totals [rows whose local proposal differed from rank 0's,
    # rows], the pending readout of those totals and the store's digest (_report_rs_agreement),
    # and the K31_RS_AGREE_HASH lines' state (the step=first line's step, step=late logged).
    _rs_agree_rows: Optional[Tuple[torch.Tensor, int, Optional[int]]] = None
    _rs_agree_count: Optional[torch.Tensor] = None
    _rs_agree_pending: Optional[tuple] = None
    _rs_agree_steps = 0
    _rs_hash_first_step: Optional[int] = None
    _rs_hash_late_logged = False
    # The one-hop greedy tail's runtime on the pair exchange's Lamport workspace, or None
    # when the layout cannot use it. The flags record publication of rank 0's drafts and
    # consumption by the next-token kernel so _agree_draft_tokens can skip that all-reduce.
    _rank_topk_walk: Optional[RankTopKWalk] = None
    _walk_published = False
    _agreed_by_rewind = False

    def __init__(self, spec_config: Any, mapping: Any, use_separate_draft_kv_cache: bool = False):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        history = getattr(spec_config, "context_kv_dtype", None)
        if history not in _DRAFT_HISTORY_CONFIGS:
            raise ValueError(
                f"this target keeps the drafter history in {_DRAFT_HISTORY_DTYPE}: "
                f"speculative_config.context_kv_dtype {history!r} names another storage; leave "
                "it 'auto' or set 'fp8'"
            )
        if self.force_num_accepted_tokens != 0.0 and self._force_accept_rng_pool is None:
            raise RuntimeError(
                "the forced-acceptance draws must exist before any CUDA graph capture"
            )
        # The sampled path's proposals by sequence slot, made on its first (eager) step.
        self._dflash2_proposals: Optional[DFlash2ProposalStore] = None
        # The headtp drafter's ring (None in the helix layouts), its runtime, and the history
        # binding each request's ring window was filled from (_fill_rings).
        self._ring: Any = None
        self._ring_draft_rt: Any = None
        self._ring_bindings: dict = {}
        self._ring_fills = 0  # first-GEN fills this process made (K31_RING_FILL lines)
        # headtp: this step's context overlap between the forward's start and the generation
        # prepare (_begin_context_overlap); None: the serial chain in the prepare.
        self._overlap: Optional[_ContextOverlap] = None
        # Forced acceptance: device totals (generation rows, accepted tokens) and Python calls.
        self._forced: Optional[torch.Tensor] = None
        self._forced_calls = 0
        # The drafter's top-k pairs all-gather and the draft-token agreement as zero-padded MNNVL
        # all-reduces over the pair exchange's group (_bind_zero_padded_copies; None: the engine's
        # all-gathers).
        self._pairs_gather: Optional[ZeroPaddedReduce] = None
        self._draft_agreement: Optional[ZeroPaddedReduce] = None
        # The one-hop tail's runtime is bound with the copies (_bind_zero_padded_copies); its
        # K31_DRAFT_PATH line is logged once a phase.
        self._rank_topk_walk = None
        self._tail_path_logged: set = set()
        # ``arange(max requests)``, bound with the history: the generation prepare's batch rows.
        self._batch_rows: Optional[torch.Tensor] = None

    def _draft_reads_target_metadata(self) -> bool:
        """The drafter reads no target attention metadata: the draft step skips its refresh."""
        return False

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        """Bind the history to the unified draft cache, then the drafter's runtime.

        Only the unified draft cache of KV cache manager V2 serves this target: anything else
        raises (routing refuses those deployments first).
        """
        if not getattr(draft_kv_cache_manager, "has_standalone_draft_cache", False):
            raise RuntimeError(
                "this target's DFlash2 drafter keeps its history in the unified draft cache of KV "
                "cache manager V2; the step's KV cache manager has none"
            )
        fresh = not (self._ctx_buf_inited and self._ctx_kv_manager is draft_kv_cache_manager)
        super()._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        if fresh:
            draft_model.bind_runtime(
                spec_metadata.max_num_requests * self._compute_block_size,
                self._ctx_page_size,
                slots=self._ctx_len.shape[0],
                slot_table=self._batch_to_slot_buffer.tensor,
            )
            # headtp: a new ring holds no window; every resident request is filled again.
            self._ring_draft_rt = draft_model._rt.draft
            self._ring = self._ring_draft_rt.ring
            self._ring_bindings = {}
            # The drafter shares the target's head object (FlatSpecTarget checks it).
            self.bind_target_lm_head(draft_model.lm_head, None)
            self._bind_zero_padded_copies(draft_model, spec_metadata.max_num_requests)
            self._batch_rows = torch.arange(
                spec_metadata.max_num_requests, dtype=torch.long, device=self._ctx_len.device
            )

    def _begin_context_overlap(
        self, draft_model, spec_metadata, attn_metadata, total_target_tokens: int
    ) -> Optional[_ContextOverlap]:
        """The context overlap's first fork (``_ContextOverlap``): on a headtp generation-only
        step that the fused generation prepare serves, the context stream waits for this stream
        (the captures and the target head) and stages ``fc``. None, with nothing queued, on any
        other step."""
        width = self._draft_tokens_per_req
        rows = attn_metadata.num_seqs * width
        if (
            getattr(self, "_ring", None) is None
            or attn_metadata.num_contexts
            or attn_metadata.num_seqs == 0
            or spec_metadata.runtime_draft_len == 0
            or draft_model.context_kv_only
            or not self._fused_gen_prepare_applies(spec_metadata)
            or not draft_model.can_stage_target_hidden(rows)
            or total_target_tokens < rows
        ):
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)[:rows]
        main = torch.cuda.current_stream(captured.device)
        side = context_stream(captured.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            draft_model.stage_target_hidden(captured)
        _log_draft_path(draft_model._rt.draft, "context", rows, "overlap")
        return _ContextOverlap(side, rows)

    def _end_context_overlap(self, draft_model) -> None:
        """A context overlap the step forked and did not join (its prepare or its drafter did not
        run): this stream waits for the context stream, so a captured graph rejoins it."""
        pending = getattr(self, "_overlap", None) or getattr(draft_model, "_context_overlap", None)
        self._overlap = None
        if getattr(draft_model, "_context_overlap", None) is not None:
            draft_model._context_overlap = None
        if pending is not None:
            torch.cuda.current_stream().wait_stream(pending.stream)

    def _prepare_paged_latent_gen_inputs(
        self,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens,
        mask_token_id,
        trained_mask_embedding,
    ) -> dict:
        """The engine's generation prepare; under the context overlap (``_ContextOverlap``) its
        context chain goes to the context stream after the prepare's kernel and the ring targets,
        and the drafter's layer 0 joins it. Same inputs and side effects otherwise."""
        overlap, self._overlap = getattr(self, "_overlap", None), None
        if overlap is None:
            return super()._prepare_paged_latent_gen_inputs(
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                total_target_tokens,
                mask_token_id,
                trained_mask_embedding,
            )
        num_gens = attn_metadata.num_seqs
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        if attn_metadata.num_contexts or num_gens * width != overlap.rows:
            raise RuntimeError(
                f"context overlap: forked for {overlap.rows} verify rows, the prepare has "
                f"{attn_metadata.num_contexts} context and {num_gens} generation requests"
            )
        shard = self._helix_shard()
        prep = dflash_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            self._batch_to_slot_buffer.tensor[:num_gens],
            self._ctx_len,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            0,
            self._dummy_slot,
            self._max_ctx,
            self._ctx_page_size,
            block_size,
            width,
            position_offset=self._ctx_position_offset if self._has_unified_draft_cache() else None,
            cp_size=1 if shard is None else shard.cp_size,
            cp_rank=0 if shard is None else shard.cp_rank,
        )
        attn_metadata._seq_lens[:num_gens] = width
        if self._batch_rows is not None and num_gens <= self._batch_rows.shape[0]:
            batch_rows = self._batch_rows[:num_gens]
        else:
            batch_rows = torch.arange(num_gens, dtype=torch.long, device="cuda")
        d = draft_model._rt.draft
        step = ring_step(d.ring, d.slot_table, batch_rows, prep.ctx_positions, prep.num_ctx.long())
        # The second fork: after the prepare's kernel (positions, owner targets and mask), the
        # ring targets and every all-reduce of the step so far on this stream.
        main = torch.cuda.current_stream(prep.ctx_positions.device)
        side = overlap.stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            projected = draft_model.finish_target_hidden(overlap.rows)
            latent_dtype = self._ctx_kv_buf[0].dtype
            if latent_dtype == torch.float8_e4m3fn:
                latent_dtype = projected.dtype  # project in the weights' dtype; the stores cast
            latent, _ = draft_model.precompute_context_kv(
                projected.to(latent_dtype), prep.ctx_positions
            )
            ring_store(d.ring, latent[:, :, 0], step.verify_pages, step.verify_offsets)
            store_context_latents(
                latent[:, :, 0],
                self._ctx_kv_buf,
                self._ctx_kv_ptrs,
                prep.store_pages,
                prep.store_offsets,
                prep.store_mask,
            )
            join = torch.cuda.Event()
            join.record(side)
        draft_model._step_latents = None
        # The prepare's outputs and the targets stay referenced until the join: the inputs below
        # and the drafter's step hold them.
        draft_model._context_overlap = overlap._replace(step=step, join=join)

        if trained_mask_embedding is not None:
            raise ValueError("this drafter's mask row is the embedding table's (no trained row)")
        # ``dflash_noise_block_embedding`` over the plain table, in one copy kernel.
        noise_embedding = k31_dflash2_noise_block(
            draft_model.draft_model_full.model.embed_tokens.table,
            prep.bonus,
            mask_token_id,
            block_size,
        )
        return {
            "lilicorr_anchor": None,
            "noise_embedding": noise_embedding,
            "query_positions": prep.query_positions,
            "num_ctx_per_req": prep.num_ctx,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            "ctx_cache_batch_idx": batch_rows,
            "first_prev_tokens": prep.bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": DraftBlockPlacement(
                prep.blk_pages, prep.blk_offsets, prep.blk_seq_lens, prep.blk_kv_bounds
            ),
            "_context_keep": (prep, step),
        }

    def _bind_zero_padded_copies(self, draft_model, max_requests: int) -> None:
        """The drafter's two TP16 copies as zero-padded all-reduces on the pair exchange's FP32
        MNNVL all-reduce (``ZeroPaddedReduce``), when the selection is vocab-parallel: the top-k
        ``(global id, value)`` pairs (each rank its 2 x top_k columns, in the worker mapping's rank
        order, as ``allgather_module``) and the draft-token agreement (rank 0's tokens; ids below
        2**24 are exact in FP32). Every rank binds at the same point."""
        self._pairs_gather = self._draft_agreement = None
        self._rank_topk_walk = None
        if self._vocab_parallel is None or draft_model.context_kv_only:
            return
        exchange = self._vocab_parallel.head._exchange
        if (exchange.tp_size, exchange.tp_rank) != (self.mapping.tp_size, self.mapping.tp_rank):
            return
        self._bind_rank_topk_walk(exchange)
        device = self._ctx_len.device
        k, rank = self.max_draft_len, exchange.tp_rank
        pair_width = 2 * draft_model.candidate_selector.top_k
        self._pairs_gather = ZeroPaddedReduce.build(
            exchange.all_reduce,
            rank * pair_width,
            pair_width,
            exchange.tp_size * pair_width,
            max_requests * k,
            torch.float32,
            device,
        )
        agreement = self._draft_agreement_mapping()
        if (
            self.agree_draft_tokens_across_tp
            and agreement is not None
            and (agreement.tp_size, agreement.tp_rank) == (exchange.tp_size, rank)
            and draft_model.lm_head.vocab_size < 2**24
        ):
            self._draft_agreement = ZeroPaddedReduce.build(
                exchange.all_reduce,
                0 if rank == 0 else None,
                k,
                _DRAFT_AGREEMENT_WIDTH,
                max_requests,
                torch.float32,
                device,
            )

    def _bind_rank_topk_walk(self, exchange) -> None:
        """Bind the greedy tail's runtime on the pair exchange's existing FP32 MNNVL workspace.
        The exchange must span this
        worker's TP group in its rank order and the shard is the TP16 head's. Every rank binds at
        the same point and takes the same decision (the workspace size and the shapes agree)."""
        mnnvl = getattr(exchange.all_reduce, "mnnvl_allreduce", None)
        head = self._vocab_parallel.head
        if (
            mnnvl is None
            or mnnvl.dtype != torch.float32
            or (mnnvl.mapping.tp_size, mnnvl.mapping.tp_rank)
            != (exchange.tp_size, exchange.tp_rank)
            or head.col_base != exchange.tp_rank * head.shard_width
        ):
            return
        workspace = get_or_scale_allreduce_mnnvl_workspace(mnnvl.mapping, torch.float32)
        slots, shard = self.max_draft_len, head.shard_width
        if not RankTopKWalk.serves(
            exchange.tp_size, slots, shard, int(workspace["buffer_size_bytes"])
        ):
            return
        self._rank_topk_walk = RankTopKWalk(
            PairExchange.from_mnnvl(workspace, exchange.tp_size, exchange.tp_rank),
            slots,
            shard,
            self._ctx_len.device,
        )
        logger.info(
            f"K31_DRAFT_RANK_TOPK_WALK bound world={exchange.tp_size} rank={exchange.tp_rank} "
            f"slots={slots} shard={shard} agree_multicast=True"
        )

    def _rank_topk_walk_tokens(self, draft_model, gen_logits, gate, anchor, num_gens: int):
        """The greedy walk's tokens ``[1, K]`` from the one-hop tail (``RankTopKWalk``), or None
        for the engine path: one request's block, the head bank's logits and gate read in place,
        and kernels compiled before any graph capture (an eager step or a warmup compiles them)."""
        rt = self._rank_topk_walk
        if rt is None or gate is None or num_gens != 1:
            return None
        logits = gen_logits.reshape(-1, gen_logits.shape[-1])
        if (
            logits.shape != (rt.slots, rt.shard)
            or gate.shape != (rt.slots, 256)
            or logits.dtype != torch.bfloat16
            or logits.stride(1) != 1
            or gate.stride(0) != logits.stride(0)
        ):
            return None
        pred = draft_model._w("d_sel_pred")
        key = (logits.stride(0), logits.dtype, True, pred.dtype, anchor.dtype)
        if not rt.is_ready(*key):
            if torch.cuda.is_current_stream_capturing():
                return None
            rt.prepare(*key)
        tokens = rt.tokens(
            logits,
            gate,
            pred,
            draft_model._w("d_sel_succ"),
            anchor.reshape(1),
            logits.dtype,
            publish=True,
        )
        self._walk_published = True
        phase = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        if phase not in self._tail_path_logged:
            self._tail_path_logged.add(phase)
            logger.info(
                f"K31_DRAFT_PATH site=tail rows={rt.slots} path=rank_topk_walk phase={phase} "
                f"rank={rt.exchange.rank} publish=1"
            )
        return tokens

    def _agree_draft_tokens(self, outputs) -> None:
        """The engine's agreement (every rank takes rank 0's ``next_draft_tokens`` and the draft
        columns of ``next_new_tokens``) as one broadcast through ``_draft_agreement``; the
        engine's all-gather when that is unbound or the shapes are not its own. A sampled step
        (``_sample_block_path``) agrees on rank 0's proposals as well (``_agree_proposals``).
        Nothing where this step's next-token kernel already took rank 0's drafts from the walk's
        publication (``k31_draft_agreed_rewind``)."""
        proposals, self._rs_agree_rows = self._rs_agree_rows, None
        agreed, self._agreed_by_rewind = self._agreed_by_rewind, False
        if agreed:
            return
        agreement = self._draft_agreement
        drafts = outputs.get("next_draft_tokens") if isinstance(outputs, dict) else None
        new_tokens = outputs.get("next_new_tokens") if isinstance(outputs, dict) else None
        if proposals is not None and drafts is not None and drafts.dim() == 2:
            self._agree_proposals(outputs, drafts, new_tokens, proposals)
            return
        if (
            agreement is None
            or drafts is None
            or drafts.dim() != 2
            or not agreement.fits(*drafts.shape)
            or (
                new_tokens is not None
                and tuple(new_tokens.shape) != (drafts.shape[0], drafts.shape[1] + 1)
            )
        ):
            super()._agree_draft_tokens(outputs)
            return
        rows, width = drafts.shape
        reduced = agreement(drafts, rows)
        if new_tokens is None:
            outputs["next_draft_tokens"] = reduced[:, :width].to(drafts.dtype)
            return
        # The agreed drafts and the next step's tokens with them, in one launch.
        agreed, next_new = k31_draft_agreed_tokens(reduced, width, drafts.dtype, new_tokens)
        outputs["next_draft_tokens"] = agreed
        outputs["next_new_tokens"] = next_new

    def _agree_proposals(self, outputs, drafts, new_tokens, proposals) -> None:
        """A sampled step's agreement: every rank takes rank 0's drafts and rank 0's proposals for
        them (candidate ids and ``q``), which block verification reads at the next step.

        Each rank's walk is a pure function of inputs that agree bitwise, so the proposals are
        expected to agree already; a rank whose walk split anyway would hold its own drafts in the
        store, load the point mass for rank 0's drafts and accept a different count. One FP32 row
        per request carries ``[drafts K | candidate ids K x top_k | q K x top_k]`` (7 + 224 + 224
        = 455 columns of the agreement's 2048 at K = 7; ids below 2**24 and q are exact in FP32,
        a sum of one value and zeros), through the agreement's MNNVL broadcast (rank 0 writes,
        then clears the proposal columns, so a greedy step's row is as before) or, when that is
        unbound or the shapes are not its own, an all-gather of every rank's row. The result is
        written back to every rank's store; ``_rs_agree_count`` adds the rows (the CUDA-graph
        padding slot's excluded) whose local tokens, ids or ``q`` differed from rank 0's.
        """
        slots, num_contexts, dummy_slot = proposals
        store = self._dflash2_proposals
        rows, k = drafts.shape
        if new_tokens is not None and tuple(new_tokens.shape) != (rows, k + 1):
            raise ValueError(
                f"next_new_tokens {tuple(new_tokens.shape)} is not [rows, 1 + draft_len] for "
                f"drafts {tuple(drafts.shape)}"
            )
        gens = rows - num_contexts
        ids_at = k
        q_at = ids_at + k * store.top_k
        width = q_at + k * store.top_k
        agreement = self._draft_agreement
        if (
            agreement is not None
            and agreement.fits(rows, k)
            and agreement.offset in (0, None)
            and width <= agreement.buffer.shape[1]
        ):
            buffer = agreement.buffer
            if agreement.offset is not None:
                buffer[:rows, :k].copy_(drafts)
                buffer[num_contexts:rows, ids_at:q_at].copy_(
                    store.candidate_ids[slots, :k].view(gens, -1)
                )
                buffer[num_contexts:rows, q_at:width].copy_(store.q[slots, :k].view(gens, -1))
            reduced = agreement.reduce(rows)
            if agreement.offset is not None:
                buffer[num_contexts:rows, ids_at:width].zero_()
        else:
            packed = drafts.new_zeros((rows, width), dtype=torch.float32)
            packed[:, :k] = drafts
            packed[num_contexts:, ids_at:q_at] = store.candidate_ids[slots, :k].view(gens, -1)
            packed[num_contexts:, q_at:width] = store.q[slots, :k].view(gens, -1)
            mapping = self._draft_agreement_mapping()
            reduced = packed if mapping is None else allgather_module(packed, mapping, 0, None)
        agreed = reduced[:rows, :k].to(drafts.dtype)
        ids = reduced[num_contexts:rows, ids_at:q_at].to(torch.int64).view(gens, k, store.top_k)
        q = reduced[num_contexts:rows, q_at:width].reshape(gens, k, store.top_k)
        differs = (
            (drafts[num_contexts:] != agreed[num_contexts:]).any(-1)
            | (store.candidate_ids[slots, :k] != ids).flatten(1).any(-1)
            | (store.q[slots, :k] != q).flatten(1).any(-1)
        )
        real = slots != dummy_slot if dummy_slot is not None else torch.ones_like(differs)
        self._rs_agree_count[0].add_((differs & real).sum())
        self._rs_agree_count[1].add_(real.sum())
        store.candidate_ids[slots, :k] = ids
        store.q[slots, :k] = q
        store.tokens[slots, :k] = agreed[num_contexts:].long()
        outputs["next_draft_tokens"] = agreed
        if new_tokens is not None:
            outputs["next_new_tokens"] = torch.cat(
                [new_tokens[:, :1], agreed.to(new_tokens.dtype)], dim=1
            )

    def _report_rs_agreement(self) -> None:
        """Every ``_RS_AGREE_PERIOD`` steps once the sampled path has run, one K31_RS_AGREE line of
        the agreement's device totals (``_agree_proposals``), read without a host
        synchronization: the totals are copied to pinned memory on the stream and logged at a
        later step whose check finds the copy complete. Host-side, outside graph capture; the
        engine calls it at every step's draft-table refresh, CUDA-graph replays included.

        Each readout also carries ``_rs_digest`` of the proposal store (candidate ids, ``q`` and
        tokens of every slot but the padding scratch row, launched eagerly here, after every
        step already enqueued): a store an agreement wrote holds the received row bit for bit,
        so the 16 ranks' digests at one step are equal exactly when every agreement so far
        delivered them the same bytes. Until the first readout with an agreed row, a readout
        is taken every step; that readout logs ``K31_RS_AGREE_HASH ... step=first``, and the
        first readout ``_RS_HASH_LATE_STEPS`` steps later ``step=late``, once each."""
        count = self._rs_agree_count
        if count is None or torch.cuda.is_current_stream_capturing():
            return
        rank = getattr(self.mapping, "tp_rank", 0)
        pending = self._rs_agree_pending
        if pending is not None and pending[0].query():
            _, host, steps = pending
            mismatched, rows, digest = host.tolist()
            if steps % _RS_AGREE_PERIOD == 1:
                logger.info(
                    f"K31_RS_AGREE rank={rank} mismatched={mismatched} rows={rows} steps={steps}"
                )
            first = self._rs_hash_first_step
            if (first is None and rows > 0) or (
                first is not None
                and not self._rs_hash_late_logged
                and steps >= first + _RS_HASH_LATE_STEPS
            ):
                logger.info(
                    f"K31_RS_AGREE_HASH rank={rank} step={'first' if first is None else 'late'} "
                    f"h={digest & 0xFFFFFFFFFFFFFFFF:016x} steps={steps} rows={rows}"
                )
                if first is None:
                    self._rs_hash_first_step = steps
                else:
                    self._rs_hash_late_logged = True
            self._rs_agree_pending = pending = None
        self._rs_agree_steps += 1
        if pending is None and (
            self._rs_agree_steps % _RS_AGREE_PERIOD == 1 or self._rs_hash_first_step is None
        ):
            host = self.__dict__.get("_rs_agree_host")
            if host is None:
                host = self._rs_agree_host = torch.empty(
                    3, dtype=torch.int64, pin_memory=prefer_pinned()
                )
            store = self._dflash2_proposals
            scratch = store.num_rows - 1
            digest = _rs_digest(
                store.candidate_ids[:scratch], store.q[:scratch], store.tokens[:scratch]
            )
            host.copy_(torch.cat((count, digest.view(1))), non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self._rs_agree_pending = (event, host, self._rs_agree_steps)

    # -- the managed draft cache's uploads, staged ------------------------------
    # Inside the step's open input commit each host-to-device upload of the preparation is staged
    # and written by the commit kernel (ops/attention/input_staging.py); outside one each is the
    # engine's direct copy. The two scatters first issue the last snapshot's gathers, which read
    # the state they write (``_issue_pending_history``: nothing when none is pending), staged or not.

    def _write_slot_values(self, target: torch.Tensor, values: dict[int, int]) -> None:
        if not values:
            return
        self._issue_pending_history()
        if stage_scatter(target, *zip(*values.items())):
            return
        super()._write_slot_values(target, values)

    def _write_ctx_len(self, updates: dict[int, int]) -> None:
        if not updates:
            return
        self._issue_pending_history()
        if stage_scatter(self._ctx_len, *zip(*updates.items())):
            for slot, value in updates.items():  # the host mirror, as the engine's
                self._ctx_len_host[slot] = value
            return
        super()._write_ctx_len(updates)

    def _rebuild_batch_to_slot(self, request_ids: list[int]) -> None:
        commit = active_input_commit()
        if commit is None or not commit.is_open:
            super()._rebuild_batch_to_slot(request_ids)
            return
        host = torch.tensor(
            [self._req_to_slot.get(rid, self._dummy_slot) for rid in request_ids], dtype=torch.long
        )
        buffer = self._batch_to_slot_buffer
        # DeviceCopyBuffer.update, staged: nothing when the buffer already holds the values.
        if not buffer.holds(host):
            stage_or_copy(buffer.tensor[: host.numel()], host)
            buffer.record_written(host)

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """The engine's refresh (:meth:`_refresh_draft_tables`), then the headtp ring's first-GEN
        fill of the requests newly bound to a history (``_fill_rings``). Once the sampled path
        has run, its agreement totals are reported from here (``_report_rs_agreement``)."""
        self._report_rs_agreement()
        refreshed = self._refresh_draft_tables(attn_metadata, num_seqs, request_ids)
        if request_ids is not None and num_seqs > 0:
            self._fill_rings(request_ids[:num_seqs])
        return refreshed

    def _fill_rings(self, request_ids: list[int]) -> None:
        """headtp: the context window of every request whose history binding is new since the last
        fill (the engine's ``_managed_cache_bindings``, set by its history preparation just
        before) into its slot's ring, from the CP ranks' owner pages: all of a step's new requests
        in one ``ring_fill_batch`` (one pack launch, one all-gather, one scatter launch; outside
        capture, no host read of the device; every rank sees the same requests in the same order).
        """
        ring = getattr(self, "_ring", None)  # a worker stub built without __init__ has none
        if ring is None:
            return
        bound, live = self._ring_bindings, {}
        d = self._ring_draft_rt
        group = list(d.mapping.cp_group)

        def all_gather(x: torch.Tensor) -> torch.Tensor:
            return allgather(x.contiguous().view(1, -1), None, group).view(len(group), *x.shape)

        fresh = []
        for rid in request_ids:
            cache = self._managed_cache_bindings.get(rid)
            if cache is None:
                continue
            live[rid] = cache
            if bound.get(rid) is not cache and rid not in fresh:
                fresh.append(rid)
        self._ring_bindings = live
        if not fresh:
            return
        slots = [self._req_to_slot[rid] for rid in fresh]
        lengths = [int(self._ctx_len_host[slot]) for slot in slots]
        tables = self._ctx_kv_manager.get_draft_block_table(fresh)
        meta = torch.empty(
            (len(fresh), 2 + tables.shape[1]), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        meta[:, 0] = torch.tensor(slots, dtype=torch.int32)
        meta[:, 1] = torch.tensor(lengths, dtype=torch.int32)
        meta[:, 2:] = tables
        meta = meta.to(self._ctx_len.device, non_blocking=True)
        ring_fill_batch(
            ring,
            meta,
            self._ctx_kv_buf,
            self._ctx_kv_ptrs,
            self._ctx_page_size,
            d.cp_size,
            d.cp_rank,
            all_gather,
        )
        for rid, slot, length, table in zip(fresh, slots, lengths, tables):
            self._ring_fills = getattr(self, "_ring_fills", 0) + 1
            n = self._ring_fills
            if n <= _RING_FILL_CHECKS and k31_features().ring_fill_check:
                # The ring's window against the owner pages of every rank (one more all-gather).
                args = (self._ctx_kv_buf, table, length, self._ctx_page_size, d.cp_size, d.cp_rank)
                sums = all_gather(window_checksums(ring, slot, *args)).view(len(group), 2)
                want, got = int(sums[:, 0].sum()), int(sums[d.cp_rank, 1])
                ok = want == got
                logger.info(
                    f"K31_RING_FILL rank={d.cp_rank} fills={n} request={rid} length={length} "
                    f"check={'ok' if ok else 'MISMATCH'} want={want} got={got}"
                )
                if not ok:
                    raise RuntimeError(
                        f"headtp ring fill of request {rid}: window checksum {got} != {want}"
                    )
            else:
                logger.info(
                    f"K31_RING_FILL rank={d.cp_rank} fills={n} request={rid} length={length}"
                )

    def _refresh_draft_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """The engine's; the host-built table of the helix draft pool and its block counts are
        staged. A device decode reads the target's block offsets, which the commit writes: the
        uploads staged so far are written first."""
        commit = active_input_commit()
        if commit is None or not commit.is_open:
            return super()._refresh_ctx_block_tables(attn_metadata, num_seqs, request_ids)
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        if not self._has_unified_draft_cache() or self._device_draft_tables:
            flush_input_commit()
            return super()._refresh_ctx_block_tables(attn_metadata, num_seqs, request_ids)
        manager = self._ctx_kv_manager
        rows = request_ids[:num_seqs]
        stage_or_copy(
            self._ctx_block_tables[:num_seqs], maybe_pin_memory(manager.get_draft_block_table(rows))
        )
        stage_or_copy(
            self._ctx_block_counts[:num_seqs],
            torch.tensor(
                [manager.get_draft_num_blocks(rid) for rid in rows],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            ),
        )
        return True

    def bind_target_lm_head(self, lm_head, logits_processor) -> None:
        """Vocab-parallel selection over the native head (``FlatVocabParallelHead``).

        The engine's binding takes a module ``LMHead`` only. Without a guided decoder or
        occurrence penalties (both edit the gathered logits), the target tokens of all-greedy
        batches and the drafter's top-k candidates come from this rank's vocabulary shard;
        sampled batches keep the gathered logits. Collective (the pair exchange's all-reduce):
        every rank binds at the same point, outside graph capture (the first prepare of the
        drafter's runtime, ``_lazy_init_ctx_buffers``).
        """
        del logits_processor
        if (
            self.guided_decoder is not None
            or self.spec_config.enable_penalty
            or not isinstance(lm_head, FlatLanguageHead)
        ):
            self._vocab_parallel = None
            return
        if self._vocab_parallel is not None and self._vocab_parallel.head.lm_head is lm_head:
            return
        self._vocab_parallel = VocabParallelSelection(
            head=FlatVocabParallelHead(lm_head, self.mapping), samples=False
        )

    def target_head_rows(self, hidden_states, attn_metadata, spec_metadata) -> torch.Tensor:
        """Read the final hidden rows in place when the gather is an identity.

        The engine lays out each generation request's gather ids as its own token rows in batch
        order, so in a step without context requests (all this target serves) they are
        ``arange(num_tokens)``: a gather of exactly ``hidden_states.shape[0]`` ids copies every row
        in place. The rows are then handed on uncopied when they are contiguous and 32-byte
        aligned, as the gather's output is (the head's kernel choice reads both). Outside a graph
        capture, the first step of each row count checks the ids on the device and raises if they
        are not the identity.
        """
        ids = spec_metadata.gather_ids
        rows = hidden_states.shape[0]
        if (
            attn_metadata.num_contexts
            or ids is None
            or ids.dim() != 1
            or ids.shape[0] != rows
            or not hidden_states.is_contiguous()
            or hidden_states.data_ptr() % 32
        ):
            return super().target_head_rows(hidden_states, attn_metadata, spec_metadata)
        checked = self.__dict__.setdefault("_h1b_checked", set())
        if rows not in checked and not torch.cuda.is_current_stream_capturing():
            arange = torch.arange(rows, dtype=ids.dtype, device=ids.device)
            if not torch.equal(ids, arange):
                raise RuntimeError(
                    f"a generation-only step's {rows} gather ids are not "
                    f"arange({rows}): {ids.tolist()}"
                )
            checked.add(rows)
            logger.info(
                f"K31_GLUE_H1B rank={self.mapping.rank} rows={rows} the target head reads the "
                "final hidden rows in place"
            )
        return hidden_states

    def _target_tokens_from_shard(self, spec_metadata) -> bool:
        """Whether a step's target tokens come from this rank's vocab shard: an all-greedy
        batch without a guided decoder."""
        return (
            self._vocab_parallel is not None
            and self.guided_decoder is None
            and spec_metadata.is_all_greedy_sample
            and not dflash2_sampled_step(self, spec_metadata)
        )

    def _fused_select_accept(self, logits, attn_metadata, spec_metadata):
        """``(accepted_tokens, num_accepted_tokens)`` with one ``select_accept`` launch doing
        what the unfused steps do, or None for them.

        On the exchanged pairs of this rank's shard (an all-greedy step whose target tokens come
        from the shard), the launch reduces the pairs (``vocab_parallel_argmax``'s last kernel),
        accepts strictly (``strict_accept``), forces the count (the engine's forced acceptance:
        same pool, counter and warmup skip, only the rows ``SpecMetadata.real_draft_rows`` marks),
        writes the KDA replay record
        (``update_mamba_states`` where that is all it writes, :meth:`_kda_replay_operands`) and
        the drafter's KV-length prep (``_prepare_kv_for_draft_forward``: generation rows + 1,
        the rewind ``1 - accepted`` returned). ``_prepare_attn_metadata_for_dflash`` runs
        first (``_prepare_attn_metadata_off_path``): it only saves metadata the launch does not
        read. Integer and FP32-compare work:
        the values of the unfused steps.
        """
        if not isinstance(logits, VocabShardedLogits) or spec_metadata.is_ragged_verify:
            return None
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        if max(num_contexts, num_gens) > _SELECT_ACCEPT_MAX_REQUESTS:
            return self._log_select_accept(f"batch {num_contexts}/{num_gens}")
        if num_gens > 0 and self._can_use_rejection_sampling(spec_metadata):
            return self._log_select_accept("rejection sampling")
        local = logits.local
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, local.device)
        draft_len = draft_tokens.shape[-1]
        if (
            draft_tokens.dim() != 2
            or draft_tokens.shape[0] != num_gens
            or (num_gens > 0 and draft_tokens.stride(-1) != 1)
            or local.dim() != 2
            or local.stride(-1) != 1
            or local.shape[0] != num_contexts + num_gens * (draft_len + 1)
        ):
            return self._log_select_accept("layout")
        kda_replay = None
        if num_gens > 0:
            why, kda_replay = self._kda_replay_operands(attn_metadata)
            if why is not None:
                return self._log_select_accept(f"kda {why}")
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        seq_lens_copied = self._prepare_attn_metadata_off_path(attn_metadata, spec_metadata)
        count = self._forced_totals(spec_metadata, local.device)
        force = self._force_accept_kernel_args(draft_len, spec_metadata, local.device)
        accepted, num_accepted, rewind = select_accept(
            logits.head.selection_pairs(local),
            draft_tokens,
            num_contexts,
            batch_size,
            force,
            kda_replay=kda_replay,
            kv_lens=kv_lens,
            count=count,
            real_rows=None if force is None else self._real_draft_rows(spec_metadata, num_gens),
        )
        if seq_lens_copied is not None:
            # The saved metadata's device sequence lengths are read after this point only.
            torch.cuda.current_stream(local.device).wait_event(seq_lens_copied)
        if self.force_num_accepted_tokens != 0.0:
            self._count_forced(num_accepted, num_contexts, spec_metadata, counted=count is not None)
        if kv_lens is not None:
            self._kv_rewind_amount = rewind
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
            attn_metadata.update_for_spec_dec(refresh_device_metadata=False)
        self._log_select_accept(None, kv_prep=kv_lens is not None, kda=kda_replay is not None)
        return accepted, num_accepted

    def _prepare_attn_metadata_off_path(
        self, attn_metadata, spec_metadata
    ) -> Optional[torch.cuda.Event]:
        """``_prepare_attn_metadata_for_dflash`` for the fused select-accept.

        With ``features.glue_h1a`` the save's copy of the device sequence lengths
        (``_seq_lens_cuda``) is made on a side stream forked here, after the target head, and the
        returned event is what the main stream waits on after ``select_accept`` (the copy's first
        reader is the drafter's prepare, which writes its generation rows). The host lengths and,
        on a CUDA-graph warmup step, ``kv_lens_cuda`` are saved as the engine saves them. Nothing
        writes the original lengths during the step, so the copy holds the same bytes as the
        engine's ``clone()`` on the main stream, where it is a 4-byte memcpy node between the
        target head and ``_local_argmax``. Returns None, having run the engine's save, with the
        switch off.
        """
        if not k31_features().glue_h1a:
            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            return None
        warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        original = attn_metadata._seq_lens_cuda
        attn_metadata.prepare_for_spec_dec("_seq_lens", *(("kv_lens_cuda",) if warmup else ()))
        copy = torch.empty_like(original)
        main = torch.cuda.current_stream(original.device)
        side = seq_lens_copy_stream(original.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            copy.copy_(original)
        done = torch.cuda.Event()
        done.record(side)
        attn_metadata._saved_tensors["_seq_lens_cuda"] = original
        attn_metadata._seq_lens_cuda = copy
        if not self.__dict__.get("_h1a_logged") and not torch.cuda.is_current_stream_capturing():
            self._h1a_logged = True
            logger.info(
                f"K31_GLUE_H1A rank={self.mapping.rank} the saved device sequence lengths are "
                "copied on a side stream (features.glue_h1a)"
            )
        return done

    def _log_select_accept(self, reason: Optional[str], kv_prep: bool = False, kda: bool = False):
        """One INFO line per path on its first call outside graph capture (the served evidence of
        which acceptance ran); returns None (the unfused path) for a reason."""
        key = reason or f"fused kv_prep={int(kv_prep)} kda_replay={int(kda)}"
        logged = self.__dict__.setdefault("_select_accept_logged", set())
        if key not in logged and not torch.cuda.is_current_stream_capturing():
            logged.add(key)
            fused = "fused=0 reason=" if reason else "fused=1 "
            logger.info(f"K31_FLAT_SELECT_ACCEPT {fused}{key}")
        return None

    @staticmethod
    def _kda_replay_operands(attn_metadata):
        """``(why_unfused, kda_replay)`` for ``select_accept``.

        ``why_unfused`` is None when ``update_mamba_states(attn_metadata, num_accepted,
        state_indices=mamba_metadata.state_indices)`` is ``KimiK3HybridCacheManager``'s and
        writes nothing but ``KDAReplayState.record_acceptance``; ``kda_replay`` is then
        ``(prev_num_accepted_tokens, state_indices, dummy_mask)``, or None when that call records
        nothing (a context-only KDA manager, or no local KDA layers).
        """
        mgr = attn_metadata.kv_cache_manager
        if (
            not isinstance(mgr, KimiK3HybridCacheManager)
            or type(mgr).update_mamba_states is not KimiK3HybridCacheManager.update_mamba_states
        ):
            return "manager", None
        if mgr.kda_context_only or mgr.local_num_mamba_layers == 0:
            return None, None
        replay = mgr._speculative_state
        # A subclass (the DSA manager's ``KimiK31KDAReplayState``) qualifies when it records the
        # acceptance with the base methods.
        if (
            not isinstance(replay, KDAReplayState)
            or type(replay).update is not KDAReplayState.update
            or type(replay).record_acceptance is not KDAReplayState.record_acceptance
        ):
            return "replay_class", None
        if replay.prev_num_accepted_tokens is None:
            return "replay_unbound", None
        if mgr._dummy_request_mask is None:
            return "dummy_mask", None
        state_indices = attn_metadata.mamba_metadata.state_indices
        if state_indices is None:
            state_indices = mgr.get_state_indices()
        dummy_mask = mgr._dummy_request_mask
        if (
            replay.prev_num_accepted_tokens.dim() != 1
            or replay.prev_num_accepted_tokens.dtype != torch.int32
            or state_indices.stride(-1) != 1
            or dummy_mask.stride(-1) != 1
        ):
            return "layout", None
        return None, (replay.prev_num_accepted_tokens, state_indices, dummy_mask)

    def _force_accept_kernel_args(
        self, runtime_draft_len: int, spec_metadata, device: torch.device
    ) -> Optional[tuple]:
        """``select_accept``'s ``force`` operand reproducing ``_apply_force_accepted_tokens``:
        None when it changes nothing, ``(base_total,)`` for a fixed count, else the fractional
        draw's pool, counter and strides."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return None
        int_part = int(self.force_num_accepted_tokens)
        frac_part = self.force_num_accepted_tokens - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if frac_part > 0.0 and base_total < max_total:
            self._ensure_force_accept_rng_state(device)
            return (
                base_total,
                frac_part,
                max_total,
                self._force_accept_rng_pool,
                self._force_accept_rng_counter,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            )
        return (base_total,)

    def _fused_kv_rewind_operands(
        self, attn_metadata, spec_metadata, batch_size, accepted_tokens, next_draft_tokens
    ):
        """``(kv_lens, rewind, num_contexts)`` when the pending rewind after the draft forward
        runs inside the next-new-tokens kernel (``draft_next_new_tokens_rewind``), else None
        (``_apply_kv_rewind_after_draft`` and ``_prepare_next_new_tokens``). A warmup restores
        ``kv_lens`` wholesale and rewinds nothing, so it takes the unfused path."""
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return None
        amount = self._kv_rewind_amount
        if (
            not self._kv_rewind_pending
            or amount is None
            or not hasattr(attn_metadata, "kv_lens_cuda")
            or self._kv_rewind_bs != batch_size
            or not amount.is_contiguous()
            or next_draft_tokens.shape[0] != batch_size
            or not next_draft_tokens.is_contiguous()
            or accepted_tokens.stride(-1) != 1
        ):
            return None
        return attn_metadata.kv_lens_cuda, amount, self._kv_rewind_nc

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        assert spec_metadata is None or not spec_metadata.is_ragged_verify, (
            "routing refuses ragged verification"
        )
        runtime_draft_len = draft_tokens.shape[-1]
        if isinstance(logits, torch.Tensor) and logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        accepted_tokens, num_accepted_tokens = strict_accept(
            target_tokens, draft_tokens, num_contexts, batch_size
        )
        return accepted_tokens, self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )

    def _apply_force_accepted_tokens(
        self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata=None
    ):
        """The engine's forced acceptance; counts what it forced (both verify paths call it)."""
        out = super()._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )
        if self.force_num_accepted_tokens != 0.0:
            self._count_forced(out, num_contexts, spec_metadata)
        return out

    def _forced_totals(self, spec_metadata, device: torch.device) -> Optional[torch.Tensor]:
        """The forced-acceptance totals for ``select_accept`` to add to (int64 ``[generation
        rows, accepted tokens]``), or None: no forced acceptance, or a graph warmup (the engine
        forces nothing there)."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if self._forced is None:
            self._forced = torch.zeros(2, dtype=torch.int64, device=device)
        capturing = torch.cuda.is_current_stream_capturing()
        if spec_metadata is not None and spec_metadata.is_cuda_graph and not capturing:
            return None
        return self._forced

    def _count_forced(
        self, out: torch.Tensor, num_contexts: int, spec_metadata, counted: bool = False
    ) -> None:
        """Add this call's forced rows to the device totals (unless ``select_accept`` already
        added them: ``counted``); log them every 200 calls.

        Rows that verified placeholder drafts kept their own count and are left out. The totals
        are device adds, so graph replays count; the line is written from Python calls outside
        capture (eager steps).
        """
        if self._forced is None:
            self._forced = torch.zeros(2, dtype=torch.int64, device=out.device)
        capturing = out.is_cuda and torch.cuda.is_current_stream_capturing()
        if spec_metadata is not None and spec_metadata.is_cuda_graph and not capturing:
            return  # graph warmup: the engine forced nothing
        if not counted:
            gen = out[num_contexts:]
            rows = self._real_draft_rows(spec_metadata, gen.shape[0])
            if rows is None:
                self._forced[0].add_(gen.shape[0])
                self._forced[1].add_(gen.sum())
            else:
                self._forced[0].add_(rows.sum())
                self._forced[1].add_(torch.where(rows, gen, 0).sum())
        self._forced_calls += 1
        if self._forced_calls % _FORCED_PERIOD == 0 and not capturing:
            logger.info(self.forced_line())

    def forced_line(self) -> str:
        """The forced-acceptance line: setting, calls, generation rows, accepted tokens, mean."""
        rows, accepted = self._forced.tolist() if self._forced is not None else (0, 0)
        return (
            f"Kimi K3.1 Rubin forced acceptance: {_TARGET_NAME} "
            f"value={self.force_num_accepted_tokens} calls={self._forced_calls} rows={rows} "
            f"accepted={accepted} mean={accepted / max(rows, 1):.4f}"
        )

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        if isinstance(logits, VocabShardedLogits):
            # Only all-greedy batches keep the shard (_target_tokens_from_shard).
            exchange = logits.head._exchange
            return vocab_parallel_argmax(
                logits.local, exchange.all_reduce, exchange.tp_size, exchange.tp_rank
            )
        if spec_metadata.is_all_greedy_sample:
            return torch.argmax(logits, dim=-1)
        num_tokens = logits.shape[0]
        seed, offset = self._rng_state_per_token(spec_metadata, num_tokens)
        return self._sample_from_logits(
            spec_metadata.advanced_sampling_mode,
            logits,
            spec_metadata.temperatures[:num_tokens],
            spec_metadata.top_ks[:num_tokens],
            spec_metadata.top_ps[:num_tokens],
            spec_metadata.min_ps[:num_tokens],
            seed=seed,
            offset=offset,
        )

    @staticmethod
    def _sample_from_logits(
        advanced_sampling_mode,
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        *,
        seed=None,
        offset=None,
    ):
        if advanced_sampling_mode.is_fused:
            return fused_sample_from_logits(
                logits, temperatures, top_ks, top_ps, min_ps, seed, offset
            )
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(
            advanced_sampling_mode, top_ks, top_ps
        )
        return flashinfer_sample_from_logits(logits, temperatures, top_ks, top_ps, seed, offset)

    @staticmethod
    def _compute_probs_from_logits(
        advanced_sampling_mode, logits, temperatures, top_ks, top_ps, min_ps
    ):
        if advanced_sampling_mode.is_fused:
            return fused_compute_probs_from_logits(logits, temperatures, top_ks, top_ps, min_ps)
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(
            advanced_sampling_mode, top_ks, top_ps
        )
        return flashinfer_compute_probs_from_logits(logits, temperatures, top_ks, top_ps)

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        # A DFlash2 sampling batch verifies the selector's block by block verification; every
        # other batch takes the engine's acceptance.
        if dflash2_sampled_step(self, spec_metadata):
            return self._verify_sampled_block(logits, attn_metadata, spec_metadata)
        return SpecWorkerBase.sample_and_accept_draft_tokens(
            self, logits, attn_metadata, spec_metadata
        )

    # -- the DFlash2 sampled path ----------------------------------------------

    def _proposal_store(self, spec_metadata) -> DFlash2ProposalStore:
        """The selector's proposals by sequence slot; made on the first sampled (eager) step."""
        store = self._dflash2_proposals
        if store is None:
            assert not torch.cuda.is_current_stream_capturing(), (
                "the DFlash2 proposal store must exist before any CUDA graph capture"
            )
            draft_probs = getattr(spec_metadata, "draft_probs", None)
            rows = (
                int(draft_probs.shape[0])
                if draft_probs is not None
                else int(spec_metadata.num_seq_slots or spec_metadata.max_num_requests) + 1
            )
            store = DFlash2ProposalStore(
                rows,
                self.max_draft_len,
                self._draft_model_ref().candidate_selector.top_k,
                spec_metadata.batch_slot_ids.device,
            )
            self._dflash2_proposals = store
            self._rs_agree_count = torch.zeros(
                2, dtype=torch.int64, device=spec_metadata.batch_slot_ids.device
            )
        return store

    def _sample_block_path(
        self, draft_model, gen_logits, block_hidden, anchor, spec_metadata, num_contexts, batch_size
    ):
        """A sampling batch's draft tokens.

        The selector's lattice over each slot's global top-k, one sampled walk (greedy rows the
        argmax), the proposals stored by sequence slot; the agreement after the step makes every
        rank's store rank 0's (``_agree_proposals``).
        """
        selector = draft_model.candidate_selector
        candidate_ids, unary_logits, _ = self._dflash2_global_top_k(
            gen_logits,
            spec_metadata,
            top_k=selector.top_k,
            full_vocab=selector.predecessor_codebook.shape[0],
            block=False,
        )
        seed, offset = self._rng_state_per_request(
            spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_draft_slot(0)
        )
        store = self._proposal_store(spec_metadata)
        slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
        dummy_slot = getattr(spec_metadata, "dummy_slot_row", 0)
        self._rs_agree_rows = (slot_ids, num_contexts, dummy_slot if dummy_slot > 0 else None)
        return k31_dflash2_selector_draft(
            selector.predecessor_codebook,
            selector.successor_codebook,
            candidate_ids,
            unary_logits,
            rubin_linear(block_hidden, selector.hidden_projection_weight, False),
            anchor,
            seed,
            offset,
            spec_metadata.request_temperatures[num_contexts:batch_size],
            GREEDY_TEMPERATURE_THRESHOLD,
            store.candidate_ids,
            store.q,
            store.tokens,
            slot_ids,
        )

    def _verify_sampled_block(self, logits, attn_metadata, spec_metadata):
        """A sampling batch's acceptance.

        Context rows sample their first token; generation rows verify the selector's block (block
        verification; greedy rows the strict rule). Returns ``(accepted_tokens [batch, K + 1],
        num_accepted_tokens [batch])``, int32.
        """
        assert not spec_metadata.is_ragged_verify, "routing refuses ragged verification"
        assert isinstance(logits, torch.Tensor), "sampled verification reads gathered logits"
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        device = logits.device
        self._log_rs_path(spec_metadata, num_gens)
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, device)
        logits = self._reshape_logits_for_accept(logits, num_contexts, num_gens, spec_metadata)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        logits = self._apply_occurrence_penalties(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )
        K = draft_tokens.shape[1]
        store = self._proposal_store(spec_metadata)
        slot_ids = spec_metadata.batch_slot_ids
        accepted_tokens = torch.zeros((batch_size, K + 1), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int32, device=device)
        if num_contexts > 0:
            accepted_tokens[:num_contexts, 0] = self._sample_tokens_for_batch(
                logits[:num_contexts], spec_metadata, num_contexts, num_contexts
            ).to(torch.int32)
            store.invalidate(slot_ids[:num_contexts])
        if num_gens > 0:
            start, end = num_contexts, num_contexts + num_gens * (K + 1)
            gen_logits = self._zero_padding_rows(
                logits[start:end],
                spec_metadata,
                num_contexts,
                batch_size,
                K + 1,
                packed_verify=True,
            )
            vocab = gen_logits.shape[-1]
            target_probs = self._compute_probs_from_logits(
                spec_metadata.advanced_sampling_mode,
                gen_logits,
                spec_metadata.temperatures[start:end],
                spec_metadata.top_ks[start:end],
                spec_metadata.top_ps[start:end],
                spec_metadata.min_ps[start:end],
            ).view(num_gens, K + 1, vocab)
            candidate_ids, q = k31_dflash2_proposal_load(
                store.candidate_ids,
                store.q,
                store.tokens,
                slot_ids[num_contexts:batch_size],
                draft_tokens,
            )
            real = self._real_draft_rows(spec_metadata, num_gens)
            if real is not None:
                # A row verifying drafts the drafter did not make (SpecMetadata.real_draft_rows)
                # takes the point mass on them, even when its slot's stored walk matches them.
                point_ids = draft_tokens.to(candidate_ids.dtype)[..., None].expand_as(candidate_ids)
                lane = torch.arange(q.shape[-1], device=device)
                point_q = (lane == 0).to(q.dtype).expand_as(q)
                candidate_ids = torch.where(real[:, None, None], candidate_ids, point_ids)
                q = torch.where(real[:, None, None], q, point_q)
            seed, offset = self._rng_state_per_request(
                spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_accept_slot
            )
            tau, bonus = dflash2_block_verify_seeded(
                target_probs, draft_tokens, candidate_ids, q, seed, offset, _DRAFT_BLOCK_VERIFY
            )
            greedy = (
                spec_metadata.request_temperatures[num_contexts:batch_size]
                <= GREEDY_TEMPERATURE_THRESHOLD
            )
            greedy_tau, greedy_bonus = dflash2_greedy_verify(
                gen_logits.view(num_gens, K + 1, vocab), draft_tokens
            )
            tau = torch.where(greedy, greedy_tau, tau)
            bonus = torch.where(greedy, greedy_bonus, bonus)
            # Drafts before tau, the bonus at tau; later positions keep drafts (emitted only under
            # a forced acceptance count).
            tokens = torch.cat((draft_tokens.long(), draft_tokens[:, -1:].long()), dim=1)
            tokens = tokens.scatter(1, tau.unsqueeze(1), bonus.unsqueeze(1))
            accepted_tokens[num_contexts:] = tokens.to(torch.int32)
            num_accepted_tokens[num_contexts:] = (tau + 1).to(torch.int32)
        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, K, spec_metadata=spec_metadata
        )
        return self._commit_occurrence_counts(
            (accepted_tokens, num_accepted_tokens), batch_size, spec_metadata
        )

    def _log_rs_path(self, spec_metadata, num_gens: int) -> None:
        """One INFO line per phase (``warmup``: a CUDA-graph warmup, whose path the captured
        graph replays; ``eager``) the first time the sampled path verifies outside graph capture:
        the served evidence that block verification runs, on every rank."""
        if torch.cuda.is_current_stream_capturing():
            return
        phase = "warmup" if spec_metadata.is_cuda_graph else "eager"
        logged = self.__dict__.setdefault("_rs_path_logged", set())
        if phase in logged:
            return
        logged.add(phase)
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        logger.info(
            f"K31_RS_PATH block_verify={int(_DRAFT_BLOCK_VERIFY)} phase={phase} "
            f"rank={getattr(self.mapping, 'tp_rank', 0)} gens={num_gens} "
            f"K={spec_metadata.runtime_draft_len} mode={getattr(mode, 'value', mode)}"
        )

    def _selector_tokens_apply(self, draft_model, spec_metadata) -> bool:
        """Whether the greedy draft tokens come from the selector walk directly
        (``select_candidate_tokens``) instead of the engine's ``sample_draft_tokens`` over the
        block: all-greedy drafting, no d2t, the 32-candidate selector, and no draft-confidence
        record (it reads the block's step)."""
        selector = getattr(draft_model, "candidate_selector", None)
        return (
            not spec_metadata.wants_advanced_draft_sampling
            and self._d2t is None
            and selector is not None
            and selector.top_k == _SELECTOR_TOP_K
            and (self._accept_stats is None or self._accept_stats.confidence_provider is None)
        )

    def _draft_shard_logits(self, draft_model, hidden: torch.Tensor, walk_tokens: bool):
        """This rank's vocabulary-shard logits of the drafter's block rows ``hidden [rows, H]``,
        and the selector's gate ``[rows, rank]`` when the greedy walk takes it from the same
        GEMM (``walk_tokens`` and a ``head_selector_bank``): views of one ``[rows, shard +
        rank]`` output, the logits its leading columns. Otherwise the head alone and None."""
        head = self._vocab_parallel.head
        bank = draft_model.head_selector_bank if walk_tokens else None
        if bank is not None:
            out = draft_head_selector_gemm(hidden, bank)
            if out is not None:
                shard = head.shard_width
                return out[:, :shard], out[:, shard:]
        return head.local_logits(hidden), None

    def _dflash2_global_top_k(
        self, gen_logits, spec_metadata, top_k: int, full_vocab: int, block: bool = True
    ):
        """Per-position global top-k candidates of the draft logits: the gathered vocabulary's
        (scored in place), or this rank's shard's merged over the TP16 group (helix8's), with
        ``-inf`` full-vocabulary block logits for the selector to write (None without
        ``block``)."""
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            unary_logits, candidate_ids = k31_dflash2_shard_top32(gen_logits, top_k, 0, False)
            return candidate_ids, unary_logits, gen_logits
        head = self._vocab_parallel.head
        assert shard * 16 == full_vocab, f"draft logits of width {shard} are no TP16 shard"
        n, slots = gen_logits.shape[:2]
        rows, width = n * slots, 2 * top_k
        pairs = self._pairs_gather
        if pairs is not None and pairs.offset is not None and pairs.fits(rows, width):
            # The pairs land in this rank's columns of the exchange buffer: no staging copy.
            out = pairs.own_columns(rows)
            k31_dflash2_shard_top32(gen_logits, top_k, head.col_base, True, out=out)
            gathered = pairs.reduce(rows).view(n, slots, -1)
        else:
            combined = k31_dflash2_shard_top32(gen_logits, top_k, head.col_base, True)
            gathered = allgather_module(combined, self.mapping, -1, None)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
        if not block:
            return candidate_ids, unary_logits, None
        block_logits = gen_logits.new_full((*gen_logits.shape[:-1], full_vocab), float("-inf"))
        return candidate_ids, unary_logits, block_logits

    def _block_gather_ids(self, draft_model, num_gens, block_size, num_draft_tokens, num_rows):
        """Block-output rows whose hidden states give the K draft logits (slots 1 .. K)."""
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        return ids.clamp(max=num_rows - 1)

    def _block_rows(self, draft_model, hidden_states_out, num_gens: int, num_draft_tokens: int):
        """``hidden_states_out[_block_gather_ids(...)]``: slots 1 .. K of each request's block.
        With the engine's slot convention and whole blocks those rows are a strided view (one
        request: a contiguous slice, no launch); otherwise the index gather."""
        block = self._compute_block_size
        rows, hidden = hidden_states_out.shape
        if (
            type(self)._draft_slot_ids is _EngineDFlashWorker._draft_slot_ids
            and rows == num_gens * block
            and num_draft_tokens + 1 <= block
        ):
            view = hidden_states_out.view(num_gens, block, hidden)[:, 1 : num_draft_tokens + 1]
            return view.reshape(num_gens * num_draft_tokens, hidden)
        return hidden_states_out[
            self._block_gather_ids(draft_model, num_gens, block, num_draft_tokens, rows)
        ]

    # -- the step --------------------------------------------------------------

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        # Set again by this step's sampled draft (_sample_block_path) only.
        self._rs_agree_rows = None

        raw_logits = logits
        logits_vocab_shard = False
        K = spec_metadata.runtime_draft_len
        if isinstance(logits, VocabShardedLogits):
            if K > 0 and self._target_tokens_from_shard(spec_metadata):
                # The target tokens come from the shard; the engine gathers the shard only for a
                # logits post-processor.
                raw_logits = logits.local
                logits_vocab_shard = True
            else:
                logits = raw_logits = logits.gather()
        if K == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        if not self._ctx_buf_inited or self._ctx_kv_manager is not draft_kv_cache_manager:
            raise RuntimeError("the unified draft cache must be prepared before the forward")

        # Snapshots for the rollback after a warmup or a failed forward; never inside a capture.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            self._saved_ctx_len = self._ctx_len.clone()
            self._saved_ctx_position_offset = self._ctx_position_offset.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._overlap = self._begin_context_overlap(
            draft_model, spec_metadata, attn_metadata, input_ids.shape[0]
        )
        self._execute_guided_decoder_if_present(logits)

        # Selection, acceptance and the bookkeeping after it in one launch where it applies.
        fused = self._fused_select_accept(logits, attn_metadata, spec_metadata)
        if fused is not None:
            accepted_tokens, num_accepted_tokens = fused
        else:
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                logits, attn_metadata, spec_metadata
            )

        if (
            self._accept_stats is not None
            and num_gens > 0
            and not spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
            and spec_metadata.request_ids is not None
        ):
            self._accept_stats.on_accept(
                spec_metadata.request_ids[num_contexts:batch_size],
                num_accepted_tokens[num_contexts:batch_size].tolist(),
            )

        if fused is None:
            # The KDA states follow the accepted tokens (the manager's replay record).
            if num_gens > 0:
                attn_metadata.kv_cache_manager.update_mamba_states(
                    attn_metadata=attn_metadata,
                    num_accepted_tokens=num_accepted_tokens,
                    state_indices=attn_metadata.mamba_metadata.state_indices,
                )
            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            self._prepare_kv_for_draft_forward(
                attn_metadata, num_accepted_tokens, num_contexts, batch_size
            )

        # The drafter's RoPE is 1-D: the first (temporal) row of an mrope position tensor.
        position_ids = position_ids[0, 0] if position_ids.ndim == 3 else position_ids.squeeze(0)
        total_target_tokens = input_ids.shape[0]

        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        drafts = not draft_model.context_kv_only
        if not drafts and num_gens > 0:
            self._check_no_real_generation(spec_metadata, num_contexts, batch_size)

        vocab_size = None
        gen_draft_tokens = None
        if num_gens > 0 and drafts:
            inputs = self.prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                total_target_tokens=total_target_tokens,
            )
            with self.draft_kv_cache_context(attn_metadata, None):
                hidden_states_out = draft_model.dflash_forward(
                    **{name: inputs[name] for name in _DRAFT_FORWARD_ARGS}
                )
                gen_hidden_states = self._block_rows(draft_model, hidden_states_out, num_gens, K)
                block_hidden = gen_hidden_states.reshape(num_gens, K, -1)
                sampled = dflash2_sampled_step(self, spec_metadata)
                walk_tokens = not sampled and self._selector_tokens_apply(
                    draft_model, spec_metadata
                )
                selector_gate = None
                if self._vocab_parallel is not None:
                    # This rank's vocabulary shard: the top-k candidates merge over the group.
                    gen_logits, selector_gate = self._draft_shard_logits(
                        draft_model, gen_hidden_states, walk_tokens
                    )
                else:
                    gen_logits = draft_model.logits_processor(
                        gen_hidden_states, draft_model.lm_head, attn_metadata, True
                    )
                gen_logits = gen_logits.reshape(num_gens, K, gen_logits.shape[-1])
                if sampled:
                    gen_draft_tokens = self._sample_block_path(
                        draft_model,
                        gen_logits,
                        block_hidden,
                        inputs["first_prev_tokens"],
                        spec_metadata,
                        num_contexts,
                        batch_size,
                    )
                elif (
                    walk_tokens
                    and (
                        tail_tokens := self._rank_topk_walk_tokens(
                            draft_model,
                            gen_logits,
                            selector_gate,
                            inputs["first_prev_tokens"].long(),
                            num_gens,
                        )
                    )
                    is not None
                ):
                    # One request's block: the one-hop tail.
                    gen_draft_tokens = tail_tokens
                elif walk_tokens:
                    # The greedy walk's tokens straight from the top-k candidates: the argmax of
                    # the block below, without writing or reading the full-vocabulary block.
                    selector = draft_model.candidate_selector
                    candidate_ids, unary_logits, _ = self._dflash2_global_top_k(
                        gen_logits,
                        spec_metadata,
                        top_k=selector.top_k,
                        full_vocab=selector.predecessor_codebook.shape[0],
                        block=False,
                    )
                    gen_draft_tokens = draft_model.select_candidate_tokens(
                        candidate_ids,
                        unary_logits,
                        block_hidden,
                        inputs["first_prev_tokens"].long(),
                        gen_logits.dtype,
                        gate=(
                            None
                            if selector_gate is None
                            else selector_gate.reshape(num_gens, K, selector_gate.shape[-1])
                        ),
                    )
                else:
                    # One coherent path through the block instead of per-position picks.
                    gen_logits = self._apply_dflash2_selector(
                        draft_model,
                        gen_logits,
                        block_hidden,
                        inputs["first_prev_tokens"],
                        spec_metadata,
                    )
                vocab_size = draft_model.lm_head.vocab_size
                if gen_draft_tokens is None:
                    gen_draft_tokens = self.sample_draft_tokens(
                        gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                    )
                if (
                    self._accept_stats is not None
                    and self._accept_stats.confidence_provider is not None
                    and not spec_metadata.is_cuda_graph
                    and not torch.cuda.is_current_stream_capturing()
                    and spec_metadata.request_ids is not None
                    and self._d2t is None
                ):
                    self._accept_stats.record_draft_confidence(
                        spec_metadata.request_ids[num_contexts:batch_size],
                        draft_model,
                        block_hidden,
                        inputs["first_prev_tokens"],
                        gen_draft_tokens,
                    )
        else:
            # A context-KV-only drafter, or no generation rows: placeholder drafts.
            gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
        self._end_context_overlap(draft_model)

        # Context requests draft nothing yet; a one-hot draft-prob row keeps them a legal
        # distribution when they turn into generation requests.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, vocab_size)

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        rewind = self._fused_kv_rewind_operands(
            attn_metadata, spec_metadata, batch_size, accepted_tokens, next_draft_tokens
        )
        published, self._walk_published = self._walk_published, False
        if (
            rewind is not None
            and published
            and num_contexts == 0
            and tuple(next_draft_tokens.shape) == (1, self._rank_topk_walk.slots)
        ):
            # The agreement rides in the next-new-tokens kernel too: rank 0's drafts, published
            # by its walk, polled on every rank.
            self._kv_rewind_pending = False
            kv_lens, amount, nc = rewind
            next_draft_tokens, next_new_tokens = k31_draft_agreed_rewind(
                self._rank_topk_walk.exchange,
                self._rank_topk_walk.slots,
                accepted_tokens[:batch_size],
                num_accepted_tokens[:batch_size],
                kv_lens[nc:],
                amount,
            )
            self._agreed_by_rewind = True
        elif rewind is not None:
            # The rewind rides in the next-new-tokens kernel.
            self._kv_rewind_pending = False
            next_new_tokens = draft_next_new_tokens_rewind(
                accepted_tokens[:batch_size],
                next_draft_tokens,
                num_accepted_tokens[:batch_size],
                *rewind,
            )
        else:
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)
        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        if rewind is None:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )

        # A warmup leaves the history lengths as it found them; real steps keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        return {
            # The gathered FP32 target logits, or this rank's vocab shard when the target tokens
            # came from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }


class FlatSpecTarget:
    """One-engine target IO: the flat model, the target head and the DFlash2 drafter and worker.

    The engine's worker keeps request, cache and acceptance control; the drafter computes on
    ``draft_core``'s weights and the target's head.
    """

    def __init__(
        self, model: FlatModel, *, draft_core: DraftCore, draft: DraftModel, worker: DFlashWorker
    ) -> None:
        self.model = model
        self.lm_head = model.lm_head
        self.logits_processor = draft.logits_processor
        self.draft_core, self.draft_model, self.spec_worker = draft_core, draft, worker
        if draft.lm_head is not self.lm_head:
            raise ValueError("The drafter must share the target head by identity")
        worker.set_draft_model(draft)
        worker.set_target_model(self)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ):
        """One engine step: the flat model, the target logits and the DFlash2 worker."""
        if spec_metadata is None:
            raise ValueError("The DFlash2 target forward requires engine speculative metadata")
        hidden = self.model(
            attn_metadata,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
        )
        rows = self.spec_worker.target_head_rows(hidden, attn_metadata, spec_metadata)
        logits = self.spec_worker.target_logits(
            rows,
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        original = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        return self.spec_worker(
            input_ids=original,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    __call__ = forward


# ---- entry ----------------------------------------
@dataclass(frozen=True)
class RuntimeLimits:
    """Native workspace admission, derived from the existing engine configuration."""

    max_batch_size: int
    max_num_tokens: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_batch_size <= 8:
            raise ValueError("The native flat runtime supports one through eight GEN requests")
        if self.max_num_tokens < 8 * self.max_batch_size:
            raise ValueError(
                "The engine token budget must cover the configured DL7 verification batch"
            )

    @property
    def max_rows(self) -> int:
        return 8 * self.max_batch_size

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    *range(1, self.max_batch_size + 1),
                    *(8 * b for b in range(1, self.max_batch_size + 1)),
                }
            )
        )


def module_loading_mode() -> str:
    """The CUDA driver's module loading mode in this process (``cuModuleGetLoadingMode``):
    ``LAZY`` or ``EAGER``. Lazy loading defers a kernel's module load to its first launch, which
    waits for resident kernels; the polling kernels of this target rely on every module they wait
    behind being loaded first."""
    from cuda.bindings import driver

    err, mode = driver.cuModuleGetLoadingMode()
    if err != driver.CUresult.CUDA_SUCCESS:
        return f"unknown ({err.name})"
    names = {
        driver.CUmoduleLoadingMode.CU_MODULE_LAZY_LOADING: "LAZY",
        driver.CUmoduleLoadingMode.CU_MODULE_EAGER_LOADING: "EAGER",
    }
    return names.get(mode, f"unknown ({mode})")


def validate_flat_config(model_config, *, native: bool = True) -> None:
    """Reject a configuration whose selected computation differs from this owner.

    ``native`` checks the checkpoint's ModelOpt recipe; it needs the source
    (multimodal) pretrained config, which carries ``quantization_config``.
    """
    from tensorrt_llm._torch.decode_switches import active

    from ..ops.common.rubin_ull_features import validate_native_config

    if native:
        validate_native_config(model_config)
    mapping = model_config.mapping
    if (
        mapping.world_size,
        mapping.tp_size,
        mapping.cp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    ) != (16, 1, 16, 1, 16, 1):
        raise ValueError("Flat ULL requires Helix attention TP1/CP16 and MoE TP16/EP1")
    if not mapping.has_cp_helix() or mapping.enable_attention_dp:
        raise ValueError("Flat ULL requires replicated token rows over the true Helix CP mapping")
    if model_config.mm_encoder_only:
        raise ValueError("The private flat entry is the complete generation text target")
    if model_config.use_cute_dsl_bf16_bmm or model_config.use_cute_dsl_bf16_gemm:
        raise ValueError(
            "Flat ULL currently preserves the native BF16 bmm_out/cuBLAS configuration"
        )
    if model_config.locality_domain_policy.enabled:
        raise ValueError(
            "Alternate locality-domain BF16 projections are not qualified for flat ULL"
        )
    if model_config.lm_head_output_dtype not in (None, torch.bfloat16):
        raise ValueError("Flat ULL keeps BF16 head GEMM rounding before FP32 logits")
    sparse = model_config.sparse_attention_config
    if (
        getattr(sparse, "algorithm", None) != "dsa"
        or getattr(sparse, "indexer_k_dtype", "fp8") != "fp8"
        or not sparse.enable_heuristic_topk
        or not sparse.use_self_sampling_topk
        or not active().dsa_helix_key_shard
    ):
        raise ValueError("Flat ULL selects the FP8 Helix key-shard/self-sampling DSA profile")
    spec = model_config.spec_config
    if (
        spec is None
        or not spec.spec_dec_mode.is_dflash()
        or not spec.spec_dec_mode.use_one_engine()
        or getattr(spec, "spec_dec_variant", None) != 2
    ):
        raise ValueError(
            "Flat ULL requires one-engine DFlash with the DFlash2 MLA drafter (spec_dec_variant 2)"
        )
    if spec.max_draft_len != DRAFT_LEN:
        raise ValueError(
            f"Flat ULL verifies q{DRAFT_LEN + 1} blocks: max_draft_len must be {DRAFT_LEN}, "
            f"got {spec.max_draft_len}"
        )
    if spec.attention_backend not in ("AUTO", DraftModel.dflash_attention_backend):
        raise ValueError(
            "The DFlash2 drafter pages its own window attention: speculative_config."
            f"attention_backend must be AUTO or {DraftModel.dflash_attention_backend}"
        )
    if getattr(spec, "context_kv_dtype", None) not in _DRAFT_HISTORY_CONFIGS:
        raise ValueError(
            f"The drafter history is FP8: speculative_config.context_kv_dtype "
            f"{spec.context_kv_dtype!r} must be 'auto' or 'fp8'"
        )
    if spec.context_kv_only:
        raise ValueError("The flat generation entry needs a complete drafter")
    if (
        spec.max_concurrency is not None
        or spec.acceptance_rate_window_size is not None
        or spec.acceptance_rate_threshold is not None
        or (spec.draft_len_schedule is not None and not spec._translated_from_max_concurrency)
    ):
        raise ValueError(
            "Helix cannot disable speculative decoding dynamically for in-flight requests"
        )


def native_process_group(mapping):
    """Create the native providers' control-plane ProcessGroup under MPI once.

    The engine already selected the current device. All ranks agree on the
    rendezvous before initialization; no shared MoE instance is needed to
    bootstrap the group as a side effect of its constructor.
    """
    import torch.distributed as dist

    from tensorrt_llm._utils import mpi_comm

    if dist.is_initialized():
        if dist.get_world_size() != 16 or dist.get_rank() != mapping.rank:
            raise ValueError("Native flat providers need a rank-matched TP16 process group")
        return dist.group.WORLD
    comm = mpi_comm()
    if comm.Get_size() != 16 or comm.Get_rank() != mapping.rank:
        raise ValueError(
            "Flat ULL bootstrap requires the generation MPI world to contain its 16 ranks"
        )
    rendezvous = None
    if mapping.rank == 0:
        try:
            host = socket.gethostbyname(socket.gethostname())
            if host.startswith("127."):
                raise ValueError("NCCL rendezvous must be reachable across generation nodes")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            rendezvous = (host, port, None)
        except (OSError, ValueError) as error:
            rendezvous = (None, None, str(error))
    host, port, error = comm.bcast(rendezvous, root=0)
    if error is not None:
        raise RuntimeError(f"Flat ULL native rendezvous failed: {error}")
    timeout = timedelta(minutes=5)
    store = dist.TCPStore(host, port, 16, mapping.rank == 0, timeout=timeout)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        store=store,
        rank=mapping.rank,
        world_size=16,
        timeout=timeout,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    return dist.group.WORLD


def _bank_head_selector(weights: FlatModelWeights, draft_weights: dict) -> tuple:
    """``weights`` and ``draft_weights`` with the lm_head shard and the selector projection as the
    two row blocks of one ``draft_head_selector_bank`` (``DRAFT_HEAD_SELECTOR_BANK`` in the
    drafter's weights). The lm_head is vocabulary-parallel over the plain TP16 group (this rank's
    rows ``vocab_start`` on) and the selector projection is whole on every rank, so every rank's
    bank is its own shard then the same 256 rows; the separate tensors are released."""
    head = weights.lm_head.weight
    bank = draft_head_selector_bank(head, draft_weights["d_sel_hidden_proj"])
    shard = head.shape[0]
    draft_weights = {
        **draft_weights,
        "d_sel_hidden_proj": bank[shard:],
        DRAFT_HEAD_SELECTOR_BANK: bank,
    }
    return replace(weights, lm_head=Bf16Weight(bank[:shard])), draft_weights


def build_flat_target(
    loader: FlatCheckpointLoader,
    draft_loader: FlatCheckpointLoader,
    model_config,
    draft_config,
    *,
    limits: RuntimeLimits,
) -> FlatSpecTarget:
    """Load and construct the complete native target directly, without a rewrite."""
    from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache

    # The engine entry validated the native recipe on the source config before
    # selecting the text config; recheck everything else here.
    validate_flat_config(model_config, native=False)
    max_rows = limits.max_rows
    mapping = model_config.mapping
    plain = mapping.repurpose_helix_cp_to_tp()
    config = getattr(model_config.pretrained_config, "text_config", model_config.pretrained_config)
    import sys

    flat_dense0 = sys.modules[__name__]

    weights = loader.model(config)
    if plain.tp_rank != mapping.cp_rank:
        raise ValueError("The repurposed TP16 rank must be the Helix CP rank")
    # The DFlash2 drafter's weights at this rank's share (read before the target's runtimes take
    # the head weight, which the bank replaces).
    draft_weights = draft_loader.dflash2(
        draft_config.pretrained_config,
        cp_rank=mapping.cp_rank,
        head_parallel=True,
    )
    weights, draft_weights = _bank_head_selector(weights, draft_weights)
    dsa_layers = tuple(i - 1 for i in config.dsa_config["dsa_layers"])
    group = native_process_group(mapping)
    reduce = FlatAllReduce(plain, dtype=torch.bfloat16, max_rows=max_rows, hidden=7168)
    # L0's dense MLP all-reduce (FlatDense0's own TP16 workspace).
    dense_reduce = FlatAllReduce(plain, dtype=torch.bfloat16, max_rows=max_rows, hidden=7168)
    helix = FlatHelix(mapping.cp_group, mapping)
    maximum = model_config.max_seq_len
    if maximum is None or maximum <= 0:
        raise ValueError(
            "Flat ULL must know its runtime window before allocating the shared NoPE table"
        )
    # The executor widens its attention max_seq_len for speculative decoding: up to
    # 2 * (tokens_per_gen_step - 1) for the overlap scheduler and the draft step, plus the
    # one-engine extra KV tokens (py_executor_creator). Cover the widest value.
    from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

    spec = model_config.spec_config
    if spec is not None:
        maximum += 2 * (spec.tokens_per_gen_step - 1) + get_num_extra_kv_tokens(spec)
    table = identity_rope_table(maximum, loader.device)
    dsa = {
        i: FlatDsa(
            loader.dsa(i, config),
            layer_idx=i,
            deterministic_topk=model_config.sparse_attention_config.gvr_deterministic_topk,
        )
        for i in dsa_layers
    }
    mla = {i: FlatSparseMla(layer_idx=i, rotary_cos_sin=table) for i in dsa_layers}
    from tensorrt_llm._utils import mpi_rank

    # Once per rank: the rows > 8 sparse MLA's host KV lengths are clamped (one CTA per row).
    logger.info(f"K31_DSA_FMHA_SINGLE_CTA rank={mpi_rank()} cap={FMHA_SINGLE_CTA_HOST_KV}")
    rows = limits.rows
    backend = FlatChainRuntime(
        weights,
        loader=loader,
        mapping=plain,
        group=group,
        dsa=dsa,
        mla=mla,
        helix=helix,
        reduce=reduce,
        gate_lower_bound=config.linear_attn_config["gate_lower_bound"],
        kda_norm_eps=config.rms_norm_eps,
        moe_scale=config.routed_scaling_factor,
        moe_renormalize=config.moe_renormalize,
        moe_norm_eps=config.rms_norm_eps,
        rows=rows,
        tokens=8,
    )
    dense0 = flat_dense0.FlatDense0(
        weights.layers[0].dense,
        dense_reduce,
        beta=weights.dense_beta,
        linear_beta=weights.dense_linear_beta,
        rows=rows,
    )
    dense0.bind_l1(backend)
    model = FlatModel(weights, backend, dsa_layers=dsa_layers, group=plain.tp_group, dense0=dense0)
    model.model_config, model.dtype = model_config, torch.bfloat16
    # The DFlash2 drafter: its weights beside the target's embedding table, its o_proj and MLP
    # sums over the plain TP16 group (the engine AllReduce), its history paged over the helix
    # mapping.
    draft_core = DraftCore(
        {"embed": weights.embedding.weight, **draft_weights},
        model_config,
        mapping,
        rubin_all_reduce(plain, model_config.allreduce_strategy, 16),
    )
    draft = DraftModel(draft_core, draft_config, model.lm_head, FlatLogitsProcessor())
    worker = DFlashWorker(
        model_config.spec_config,
        plain.mapping_for_full_rows(),
        should_use_separate_draft_kv_cache(model_config.spec_config),
    )
    return FlatSpecTarget(model, draft_core=draft_core, draft=draft, worker=worker)


def _same_view(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Exact view identity from storage metadata, including during meta construction."""
    return (
        actual.untyped_storage()._cdata == expected.untyped_storage()._cdata
        and actual.storage_offset() == expected.storage_offset()
        and actual.shape == expected.shape
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
        and actual.device == expected.device
    )


def _weight_tensors(value) -> Iterator[torch.Tensor]:
    """Walk the private weight-container schema, never a compute module tree."""
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _weight_tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for tensor in value.values():
            yield from _weight_tensors(tensor)
    elif isinstance(value, (tuple, list)):
        for tensor in value:
            yield from _weight_tensors(tensor)


def _audit_banks(value, counts: dict, path: str = "") -> None:
    """Count the native containers and leaf dtypes of the private weight schema."""
    from .weights import DequantizedMxfp8

    if isinstance(value, DequantizedMxfp8):
        counts["bf16_from_mxfp8"] = counts.get("bf16_from_mxfp8", 0) + 1
    elif isinstance(value, Bf16Weight):
        counts["bf16"] = counts.get("bf16", 0) + 1
    elif isinstance(value, torch.Tensor):
        key = "routed_" + str(value.dtype).removeprefix("torch.") if ".routed" in path else None
        key = key or "tensor_" + str(value.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + 1
        return
    if is_dataclass(value):
        for field in fields(value):
            _audit_banks(getattr(value, field.name), counts, f"{path}.{field.name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _audit_banks(item, counts, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for item in value:
            _audit_banks(item, counts, path)


def audit_flat_precision(banks) -> str:
    """The loaded representation, by container type and leaf dtype.

    Container constructors already reject a non-native dtype or layout (E4M3
    codes with UE8M0 scales, BF16 exclusions); this records what was loaded.
    Routed experts must hold packed NVFP4 bytes, never a BF16/FP16 dequantized bank.

    Each of the checkpoint's MXFP8 projections (KDA q/k/v/g/f_a/b/o, shared
    gate/up/down) is held in one form: native MXFP8 (E4M3 codes + UE8M0 scales,
    dynamic MXFP8 activations) or BF16 dequantized exactly from those codes and
    scales at load (codes and scales freed, BF16 activations). The modes are
    ``proj=native_mxfp8`` (all native), ``proj=bf16_from_mxfp8`` (all BF16) and
    ``proj=kda_in_mxfp8`` (the KDA in-projections native, the rest BF16).
    """
    counts: dict = {}
    _audit_banks(banks, counts)
    routed = {k for k in counts if k.startswith("routed_")}
    if not routed or routed & {"routed_bfloat16", "routed_float16"}:
        raise RuntimeError(f"Routed experts are not native NVFP4: {sorted(routed)}")
    native, dequantized = counts.get("mxfp8", 0), counts.get("bf16_from_mxfp8", 0)
    if not native and not dequantized:
        raise RuntimeError("No MXFP8 checkpoint projection was loaded")
    if native and dequantized:
        if native != counts.get("kda_in_mxfp8", 0):
            raise RuntimeError(
                f"Only KDA in-projections may stay native beside BF16 projections: {counts}"
            )
        mode = "proj=kda_in_mxfp8"
    else:
        mode = "proj=native_mxfp8" if native else "proj=bf16_from_mxfp8"
    return mode + " " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _lazy_checkpoint_type() -> type:
    """``_LazyCheckpoint``, built on first use: its base comes from the checkpoint package, which
    imports every checkpoint loader (as ``HfWeightLoader`` did, inside the load)."""
    global _LAZY_CHECKPOINT
    if _LAZY_CHECKPOINT is None:
        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict

        class _LazyCheckpoint(ConsumableWeightsDict):
            """Name -> lazy safetensors slice of a checkpoint's shards, holding their file handles
            until cleared (``HfWeightLoader._load_lazy_safetensors``, ``_LazySafetensorsWeights``)."""

            def __init__(self, checkpoint_dir: str) -> None:
                weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
                if not weight_files:
                    raise RuntimeError(f"No safetensors files in {checkpoint_dir}.")
                # The sharded files when both sharded and consolidated ones are present.
                sharded = [x for x in weight_files if "consolidated" not in os.path.split(x)[1]]
                if sharded:
                    weight_files = sharded
                weights = {}
                self._handles = []
                for file_name in weight_files:
                    handle = safetensors.safe_open(file_name, framework="pt", device="cpu")
                    self._handles.append(handle)
                    for name in handle.keys():
                        weights[name] = handle.get_slice(name)
                logger.info(
                    f"Lazily opened {len(weight_files)} safetensors files "
                    f"({len(weights)} tensors) from {checkpoint_dir}"
                )
                super().__init__(weights)
                self.checkpoint_dir = checkpoint_dir

            def clear(self) -> None:
                super().clear()
                self._handles.clear()

        _LAZY_CHECKPOINT = _LazyCheckpoint
    return _LAZY_CHECKPOINT


_LAZY_CHECKPOINT = None


@contextmanager
def _lazy_checkpoint(path: str):
    weights = _lazy_checkpoint_type()(path)
    try:
        yield weights
    finally:
        weights.clear()


class PrefillTarget:
    """Steps with context rows (the generation server's context warmups): the flat target, whose
    model splits the step's rows by the metadata itself."""

    def __init__(self, model: "ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1"):
        self.model = model

    def forward(self, attn_metadata, *args, **kwargs):
        return self.model._target(attn_metadata, *args, **kwargs)


class DecodeTarget:
    """Generation-only steps (the served path): the flat target."""

    def __init__(self, model: "ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1"):
        self.model = model

    def forward(self, attn_metadata, *args, **kwargs):
        return self.model._target(attn_metadata, *args, **kwargs)


class ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1(DecoderModelForCausalLM):
    """Kimi K3.1 NVFP4 on sm_107, the ULL generation deployment (``features`` profile ``ull16``).

    DecoderRunner model protocol around private native target/spec owners.

    The base IO protocol is retained, but its constructor (which creates a
    shared LMHead) is intentionally not called. The parameter registry below
    consists solely of same-storage aliases for engine device/introspection IO.
    """

    mamba_metadata_cls = KimiK3MambaMetadata
    # The K3 ULL decode-step shell (pyexecutor/ull_shell.py): batched input commit, step prep
    # folded onto the execution stream, pinned executor thread, 200 us GIL switch interval, the
    # decode-step head in one kernel; the first 200 input commits are verified against the
    # copy-by-copy uploads.
    ull_step_shell = StepShell(
        step_prep=True,
        step_fold=True,
        pin_executor=True,
        gil_switch_us=200,
        verify_commits=200,
        spec_head_fusion=True,
    )
    _context_warmups_supported = True
    # ``load_weights`` reads the DFlash2 drafter checkpoint into the drafter's core.
    loads_draft_weights_internally = True
    use_fx_for_pcg_fallback = False

    def attention_backend_for(self, backend_name: str, sparse_params) -> type:
        """The engine's TRTLLM DSA backend with this target's metadata (``RubinUllDSAMetadata``)."""
        from ..ops.attention.rubin_ull_dsa_metadata import attention_backend_for

        return attention_backend_for(backend_name, sparse_params)

    def __init__(self, model_config, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        # The engine's limits (ModelLoader records llm_args' values): the workspaces' row counts.
        self.runtime_limits = RuntimeLimits(
            max_batch_size=model_config.extra_attrs["max_batch_size"],
            max_num_tokens=model_config.extra_attrs["max_num_tokens"],
        )
        validate_flat_config(model_config)
        # The diagnostics and levers the deployment's model_kwargs set (features.MODEL_CONFIG_FIELDS),
        # read off the source config before the text config replaces it; none on the served path.
        from ..ops.common.rubin_ull_features import apply_model_config

        apply_model_config(model_config.pretrained_config)
        self._checkpoint_dir = str(model_config.pretrained_config._name_or_path)
        self._source_pretrained_config = model_config.pretrained_config
        text_config = getattr(
            model_config.pretrained_config, "text_config", model_config.pretrained_config
        )
        if text_config is not model_config.pretrained_config:
            model_config._frozen = False
            model_config.pretrained_config = text_config
            model_config._frozen = True
        self.model_config = model_config
        self.mapping = model_config.mapping
        logger.info(f"K31_MODULE_LOADING rank={self.mapping.rank} mode={module_loading_mode()}")
        self.pp_rank, self.pp_size = self.mapping.pp_rank, self.mapping.pp_size
        self.spec_config = model_config.spec_config
        self.has_custom_lm_head = False
        self.layer_idx = -1
        self.prologue, self.epilogue = [], []
        self._target = None
        self._prefill, self._decode = PrefillTarget(self), DecodeTarget(self)
        self._weights_transformed = False
        self._native_weights = nn.ParameterDict()
        self._native_aliases = []
        self.register_buffer(
            "_device_anchor", torch.empty(0, dtype=torch.bfloat16), persistent=False
        )
        self._resident_weight_bytes = 0
        self.model_config.extra_attrs["kda_checkpoints"] = True
        self.model_config.extra_attrs["draft_kv_cache_quant_algo_override"] = "FP8"
        # The drafter's config, which the KV cache creator reads off the model (draft_config).
        self.draft_config = draft_config_for(model_config)

    def __post_init__(self) -> None:
        """Private loading owns recipe selection and the final native representation."""

    def __pp_init__(self) -> None:
        if self.pp_size != 1:
            raise ValueError("Flat ULL has no pipeline partition")

    @staticmethod
    def open_checkpoint_weights(checkpoint_dir: str, mapping=None):
        return _lazy_checkpoint(checkpoint_dir)

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config=None) -> bool:
        return True

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None) -> str:
        return "PYTHON"

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return {"kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 64}}

    @property
    def vocab_size_padded(self) -> int:
        config = getattr(self.config, "text_config", self.config)
        return config.vocab_size

    def load_weights(self, weights: dict) -> None:
        if self._target is not None:
            raise RuntimeError("Flat ULL loads its one native weight set once")
        checkpoint_dir = Path(self._checkpoint_dir)
        source = json.loads((checkpoint_dir / "config.json").read_text())
        recipe = source["quantization_config"]["quantized_layers"]
        device = torch.device("cuda", torch.cuda.current_device())
        loader = FlatCheckpointLoader(weights, recipe, device=device, rank=self.mapping.rank)
        with _lazy_checkpoint(self.spec_config.speculative_model) as draft_weights:
            draft_loader = FlatCheckpointLoader(
                draft_weights, {}, device=device, rank=self.mapping.rank
            )
            target = build_flat_target(
                loader,
                draft_loader,
                self.model_config,
                self.draft_config,
                limits=self.runtime_limits,
            )
            torch.cuda.synchronize(device)
        self._target = target
        self.model, self.lm_head = target.model, target.lm_head
        self.draft_model, self.spec_worker = target.draft_model, target.spec_worker
        self.logits_processor = target.logits_processor
        self.use_separate_draft_kv_cache = self.spec_worker.use_separate_draft_kv_cache
        self.epilogue = [self.spec_worker]
        banks = (
            self.model.weights,
            self.model.backend.kda_weights,
            self.model.backend.moe_weights,
            tuple(dsa.weights for dsa in self.model.backend.dsa.values()),
            target.draft_core.w,
        )
        storages = set()
        for tensor in _weight_tensors(banks):
            storage = tensor.untyped_storage()
            key = storage._cdata
            if key in storages:
                continue
            storages.add(key)
            self._resident_weight_bytes += storage.nbytes()
            alias = nn.Parameter(tensor, requires_grad=False)
            self._native_weights[f"bank_{len(storages):05d}"] = alias
            self._native_aliases.append((alias, tensor))
        self._assert_native_aliases()
        from tensorrt_llm.logger import logger

        logger.info(
            f"K31_ULL_NATIVE_PRECISION_OK rank={self.mapping.rank} flat "
            f"{audit_flat_precision(banks)} resident_gib={self._resident_weight_bytes / 2**30:.2f}"
        )
        self._weights_transformed = True

    def _assert_native_aliases(self) -> None:
        for alias, source in self._native_aliases:
            if not _same_view(alias, source):
                raise RuntimeError(
                    "Engine parameter IO changed a native weight bank's storage or dtype"
                )

    def _apply(self, fn, recurse: bool = True):
        if self._target is None:
            return super()._apply(fn, recurse=recurse)
        checked = set()
        for alias, _ in self._native_aliases:
            key = (alias.dtype, alias.device)
            if key in checked:
                continue
            checked.add(key)
            probe = torch.empty(0, dtype=alias.dtype, device=alias.device)
            if fn(probe) is not probe:
                raise RuntimeError(
                    "Loaded flat native weights do not support dtype/device/storage conversion"
                )
        self._assert_native_aliases()
        return self

    def setup_aliases(self) -> None:
        if self._target is not None and self.draft_model.lm_head is not self.lm_head:
            raise RuntimeError("The private draft must retain the target's single head object")
        self._assert_native_aliases()

    def transform_weights(self) -> None:
        self.setup_aliases()

    def cache_derived_state(self) -> None:
        self.setup_aliases()

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Attach the grammar decoder to the DFlash target-token verification."""
        return self.spec_worker.set_guided_decoder(guided_decoder)

    def prepare_disagg_generation_request(self, request: Any) -> None:
        """A transferred request's first step (the executor calls it once the transfer completes):
        the drafts its context server sent count as drafts the drafter made, and its sequence
        slot's stored proposals are dropped.

        The context server's first block (tep8, ``DFlashWorker._draft_context_rows``) is the
        drafter's greedy walk over the same history. It is real (``py_has_real_drafts``: forced
        acceptance applies to it) when it is a full block:
        ``max_draft_len`` ids, not all zero (a context server that drafts nothing sends none, or
        zeros from one that predates the empty send). The executor already set the flag to False
        for a parallel-draft mode, and placeholders for an empty list.

        The proposal store keeps a slot's last walk until the next walk or a context step
        invalidates it; a transferred request has no context step here, so its slot's row is
        invalidated now. A sampled first step then verifies the sent block as a point mass (the
        exact law of a deterministic proposal) instead of a previous request's proposals.
        """
        worker = self.spec_worker
        invalidate_dflash2_proposal_slot(worker, request.py_seq_slot)
        params = getattr(request, "context_phase_params", None)
        drafts = list(getattr(params, "draft_tokens", None) or ())
        real = len(drafts) == worker.max_draft_len and any(drafts) and min(drafts) >= 0
        request.py_has_real_drafts = bool(real)
        counts = worker.__dict__.setdefault("_ctx_first_draft_counts", [0, 0])
        counts[0] += 1
        counts[1] += int(real)
        if counts[0] == 1 or counts[0] % _CTX_FIRST_DRAFT_PERIOD == 0:
            logger.info(
                f"K31_CTX_FIRST_DRAFTS rank={self.mapping.rank} requests={counts[0]} "
                f"real={counts[1]} sent={len(drafts)} enabled=1"
            )

    def validate_request(self, request: Any) -> None:
        """Refuse, before it is scheduled, media on a request that is not generation-only: this
        target builds no vision tower, so its context step cannot encode images. A generation-only
        request arrives with its context server's KV (the images encoded there, by the dep16 or
        tep8 target) and prompt ids whose media placeholders are ordinary in-vocabulary ids; it
        carries no media. A refused request fails alone, where the forward would fail the batch."""
        if getattr(request, "py_multimodal_data", None) and not getattr(
            request, "is_generation_only_request", False
        ):
            raise ValueError(
                f"Image input is not supported: {_TARGET_NAME} runs no vision tower (a "
                "disaggregated generation server: its context server encodes the images)"
            )

    def forward(self, attn_metadata, *args, **kwargs):
        if self._target is None:
            raise RuntimeError("Load the native flat checkpoint before model execution")
        target = self._prefill if attn_metadata.num_contexts > 0 else self._decode
        return target.forward(attn_metadata, *args, **kwargs)

    @classmethod
    def select_kv_cache_manager_cls(
        cls, model_config, kv_cache_config, is_disagg=False, cache_transceiver_config=None
    ):
        """The hybrid KDA + MLA cache manager with the DSA indexer K-cache for a DSA model with KDA
        layers, else None; it needs the KV cache manager V2 and, disaggregated, the Python NIXL
        transceiver."""
        from tensorrt_llm._torch.pyexecutor._util import _resolve_disagg_transceiver_route
        from tensorrt_llm._torch.pyexecutor.config_utils import is_hybrid_linear
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
            use_py_mamba_cache_manager,
        )

        sparse_config = model_config.sparse_attention_config
        if getattr(sparse_config, "algorithm", None) != "dsa":
            return None
        if (
            not is_hybrid_linear(model_config.pretrained_config)
            or model_config.get_num_mamba_layers() == 0
        ):
            return None
        if kv_cache_config.use_kv_cache_manager_v2 is not True:
            raise ValueError(
                "Kimi K3.1 DSA requires use_kv_cache_manager_v2=True "
                "(the indexer K-cache is a V2 extra buffer)."
            )
        if is_disagg:
            # Only the Python NIXL transceiver moves KDA state and the INDEX_KEY buffers.
            backend, runtime = _resolve_disagg_transceiver_route(cache_transceiver_config)
            if runtime != "PYTHON" or backend != "NIXL":
                raise ValueError(
                    "Kimi K3.1 disaggregated serving requires cache_transceiver_config "
                    "backend='NIXL' with transceiver_runtime='PYTHON' (got "
                    f"backend={backend!r}, transceiver_runtime={runtime!r})."
                )
        if kv_cache_config.enable_block_reuse and kv_cache_config.enable_kv_pool_rebalance:
            raise ValueError(
                "V2 Mamba block reuse is not compatible with enable_kv_pool_rebalance because the "
                "rebalancer does not yet model retained recurrent-state snapshots."
            )
        if use_py_mamba_cache_manager():
            raise ValueError(
                "Kimi K3.1 DSA supports only its V2 cache manager; unset TRTLLM_USE_PY_MAMBA."
            )
        from ..ops.ssm.rubin_ull_kda_dsa_cache_manager import KimiK3DSAHybridCacheManager

        return KimiK3DSAHybridCacheManager
