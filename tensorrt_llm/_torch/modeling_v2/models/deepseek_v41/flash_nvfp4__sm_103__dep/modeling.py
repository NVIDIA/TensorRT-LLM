# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-v4.1-flash-nvfp4 / sm_103 / dep (the generation role at 8 or 32 ranks).

DeepSeek-V4.1-Flash, modelopt MIXED_PRECISION export: 40 layers, hidden 5120,
64 query heads of dim 512 (one shared KV head), q-LoRA 1280, 384 routed experts
at top-6 plus one shared expert (intermediate 2304). Routed experts are NVFP4
(group 16); attention, shared experts and the engram projections are block-32
FP8 with UE8M0 scales; the head and the router are bf16.

The residual stream is `hc_mult = 4` parallel copies (Hyper-Connections). Each
sub-layer reads a collapse of the copies and writes back through a doubly
stochastic mix; the pre-mix a sub-layer computes is consumed by the *next*
one, so the loop carries `pre_mix` across layers exactly as the checkpoint's
reference does.

`dep` is `tensor_parallel_size` = `moe_expert_parallel_size` = world size (8 or 32) with
`enable_attention_dp: true` (the generation role: embedded DSpark speculation, `enable_lm_head_tp_in_adp`,
CUDA graphs; deployed at batch 320 on 8 ranks and batch 25 on 32): requests are split across the ranks, the whole
attention block and the shared expert are replicated, and only the experts (`384 / world` per rank) and the engram
tables are sharded. The head is replicated
(the engine's `LMHead` under attention DP) and the DSpark worker verifies from TP shards of it.

Composition. Attention (`attention.py`), the routed experts, the shared expert, the Hyper-Connection mixes,
the clamped SwiGLU and the engram gate are built from entries: the family's own (`models/deepseek_v41/ops/
{attention,residual,activation,engram}`: `csa2_*`, `hc_*`, `engram_gate`, `clamped_swiglu`, `engram_lookback_hash`
(decode-sized n-gram hash), `engram_adp_*` (the engine's attention-DP peer-memory engram exchange)) and the shared
catalog's (`gemm/cublas_mm`, `norm/flashinfer_rmsnorm`, `quantization/fp4_quantize`, `moe/fp4_block_scale_moe_runner`,
`comm/allgather`, `comm/reducescatter`), with the engine's `CSA2CacheManager` as the allocator of the cache pools.
This file owns the trunk: the HC loop, routing, the engram branch (hash, table lookup), weight routing and the
registration shell. Not an entry: the plain-torch engram n-gram hash of wide steps, the table lookup, the token
embedding, the engram's fallback all-reduce/all-gather (the engine's), and the inherited `DecoderModel` /
`DecoderModelForCausalLM` shells (lm_head, logits gather, speculative worker).

Two execution paths, chosen per step by `ModelingV2Core._select_target`: `DecodeTarget` (generation-only steps of
at most `_PAGED_MAX_ROWS` rows with equal request lengths: the indexer reads lengths on the device, so the step is
CUDA-graph safe) and `PrefillTarget` (everything else: exact span per request, one host read each).

MoE backends. `moe_config.backend: MEGAMOE_CUTEDSL` runs the routed experts (and, with `megamoe_fuse_shared_expert`,
the FP8 shared expert) through one fused MegaMoE launch per layer (`megamoe.py`, weights packed by `weights.py`:
`load_megamoe_experts` / `load_megamoe_shared`); every other backend runs the TRTLLM fp4 runner (`_Moe`) as before.
The MegaMoE expert placement starts from the engine's default (or the config's `initial_global_assignments`); with
`load_balancer.warmup_placement` the engine's warmup placement controller re-places it from the served traffic, as for
the engine's own MoE layers (`ops/moe/megamoe_placement`).
The online balancer (`layer_updates_per_iter > 0`) is refused.

Bounded decoder replay (context steps; stock's `TRTLLM_V41_DECODER_BOUNDED_REPLAY`, on by default; the policy and the
plan are stock's own, `ops/attention/csa2_decoder_replay`). Layer 20, the last unpooled KV owner, publishes the
compressed KV and index keys of every row of a context chunk; from there to the last layer only the rows of the chunk
inside the final 140 tokens of its prompt run (queries, index selection, attention, experts), with the window of their
first rows truncated at the window start, and every generation row of a mixed step. The executor plans the rows
(`prepare_adp_inputs`, before the token-count exchange) and the experts' collectives use every rank's decoder counts
from that layer on. Approximate across stacked SWA layers, as stock is. Requests that return context logits (or other
per-row outputs), CUDA-graph steps and per-request prefill graphs run every row, as in stock.

Engram look-back and speculative decoding:

* engram hashes read the n-gram look-back from a device-side ring of recent token ids per request
  (graph-safe, correct under the overlap scheduler and speculative rollback); an engine hook
  seeds it for requests new to the engine;
* the speculative (DSpark) draft model and worker are the engine's (`SpecDecOneEngineForCausalLM` builds them
  from `speculative_config`; the draft stages live in the same checkpoint, `mtp.*`); this target only captures
  the residual stream the draft reads.
"""

import collections
import contextlib
import dataclasses
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.host_staging import copy_host_to_device
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.a2a_experts import (
    a2a_experts,
    build_a2a_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
    csa2_complete_encoder_replay,
    csa2_encoder_recovery,
    csa2_encoder_replay_tokens,
    csa2_replay_plan,
    csa2_replay_policy,
    csa2_step_replay_plan,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    SMALL_M_ROWS,
    csa2_mxfp8_agree_tactics,
    csa2_mxfp8_gemm_prewarm,
    csa2_mxfp8_gemm_tune_rows,
    csa2_mxfp8_graph_rows,
    csa2_mxfp8_linear,
    csa2_mxfp8_tune_engine_gemm,
    mxfp8_row_scales,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.comm.engine_comm import (
    engine_allgather,
    engine_allreduce,
    engine_allreduce_create,
    engine_alltoallv_tokens,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.embedding.token_embedding import (
    token_embedding,
    token_embedding_create,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_adp_exchange import (
    engram_adp_exchange_create,
    engram_adp_exchange_hashes,
    engram_adp_push,
    engram_adp_receive,
    engram_adp_takes,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_gate_fused import (
    engram_gate_fused,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_history_slots import (
    EngramHistorySlots,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookback_hash import (
    MAX_ROWS as ENGRAM_HASH_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookback_hash import (
    engram_lookback_hash,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
    engram_table_mxfp4,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route_nvfp4_quant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import MoeA2AWorkspace
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.engram_gate import engram_gate
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_allreduce_post_collapse_norm import (
    hc_allreduce_post_collapse_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_boundary import hc_boundary
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_collapse import hc_collapse
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lagged_coefficients import (
    SMALL_MAX_ROWS as _ANY_COEFF_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lagged_coefficients import (
    hc_lagged_coefficients,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lowm import (
    hc_lamport_workspace,
    hc_split_enabled,
    moe_unfinalized_finalize,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mega_boundary import (
    hc_mega_boundary,
    hc_mega_boundary_prewarm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_collapse_norm import (
    hc_post_collapse_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (
    hc_pre_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.step_hooks import step_hooks
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from ..common import Fp8Rows, KvPools, opt_into_small_m_gemm, text_model_config
from ..engram_hash import ENGRAM_RING, EngramHash, engram_buckets
from ..image_input import refuse_image_request, refuse_image_step
from ..limits import dep_refused
from ..loaders import load_head
from ..shared_expert import SharedExpertFp8
from ..swa_tables import engine_swa_tables
from . import weights as _weights
from .attention import StepCtx, _Attention
from .megamoe import _MegaMoe, attach_build, prime, resolve_megamoe_config

# The GPU architecture this target IS; routing will not send another one here.
_SM = (10, 3)

_BLOCK = 32  # FP8 block edge of the checkpoint's dense weights
_HC_POST_MULT = 2.0  # post-mix scale of the Hyper-Connections (`post = 2 * sigmoid(.)`)
# Boundaries with a post update of at least `hc_mega_boundary.MIN_ROWS` rows run DeepGEMM's one-kernel `mega_mhc`, as
# the engine's fused lagged boundary does.
# The engram tables stay in pinned host memory that the lookup kernel reads through UVA, as the engine's
# `ShardedFp8MultiHeadEmbedding` does; processes of one node that pass the same `share_key` map one copy.
_RING = ENGRAM_RING  # positions of history per slot (the family's ring)
_MAX_REQS = 4096
# Rows per rank the engram attention-DP exchange's peer buffers hold (the engine's `max_num_tokens`, up to this; a
# forward with more rows on some rank keeps the all-gather / all-reduce path).
_ENGRAM_EXCHANGE_MAX_ROWS = 2048
# query rows of an eager generation step on the paged path (the fused indexer has no row cap; a CUDA-graph step takes
# it at any width)
_PAGED_MAX_ROWS = 4096
_EAGER_ROWS_CACHED = 8  # eager (non-graph) step shapes whose `_uniform_rows` constants are kept


class _Engram(nn.Module):
    """Adds a gated n-gram lookup into the residual stream (layers 1 and 14)."""

    # The table is built in `load`, after the loader's post-init check: the module tells it that the pinned host
    # table is filled in place, which only the regular load_format=AUTO checkpoint load does.
    _requires_regular_checkpoint_loading = True
    # Engine hook read by the decode-graph warm-up (`warmup.py`): a model holding a module with this flag tunes and
    # dispatches its MXFP8 `Linear`s to FlashInfer inside decode CUDA graphs. The built-in model sets it on its Engram
    # projection (`EngramFp8Projection`); this target's own GEMMs are entries, so the flag's effect here is on the
    # engine's DSpark drafter, whose MXFP8 projections then run their tuned FlashInfer tactics as in the built-in model
    # instead of the untuned fallback.
    _use_flashinfer_mxfp8_decode_graph_default = True

    def __init__(self, cfg, mapping, hash_index: int):
        super().__init__()
        self.mapping = mapping
        self.hash_index = hash_index
        self.hc = cfg.hc_mult
        self.dim = cfg.hidden_size
        self.head_dim = cfg.engram_head_dim
        self.cfg = cfg  # the checkpoint's quantization config picks the table format at load (`engram_table_mxfp4`)
        self.rows = cfg.engram_num_embeddings[hash_index]
        self.eps = cfg.rms_norm_eps
        self.cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.bucket_sizes = engram_buckets(cfg)[1][hash_index]
        # Under attention DP the heads are sharded over the exchange group: the tensor-parallel group, or (as the
        # engine's `DeepseekV41Engram`) the largest rank group whose size divides the head count (dep32: 24 heads,
        # groups of 8). One table copy per group.
        tp = mapping.tp_size
        self.group_start, self.group_size = 0, tp
        if tp > 1 and mapping.enable_attention_dp and self.cols % tp:
            divisors = [d for d in range(2, tp) if tp % d == 0 and self.cols % d == 0]
            self.group_size = max(divisors, default=tp)
            self.group_start = mapping.tp_rank - mapping.tp_rank % self.group_size
        # global ranks of the exchange group (the members of the attention-DP all-to-all)
        self.group_ranks = (
            list(mapping.tp_group[self.group_start : self.group_start + self.group_size])
            if tp > 1
            else [0]
        )
        # The peer-memory exchange of attention DP
        # (`ModelingV2Core._engram_exchange_setup`) and this layer's push event.
        self.exchange = None
        self.push_event = None
        # The table stays FP8 as stored, in pinned host memory (see `load`).
        self.table = None
        self.wkv = None
        self.q_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.gate_w = None  # q_weight * k_weight in fp32 (see `load`)
        self.allreduce = engine_allreduce_create(mapping)

    def load(self, weights, prefix: str, device, share_key: Optional[str]) -> None:
        emb = weights[prefix + "embed.weight"]
        sc = weights[prefix + "embed.scale"]
        # stock's MXFP4 Engram tables: E2M1 pairs, D / 2 bytes a row
        mxfp4 = engram_table_mxfp4(self.cfg, prefix)
        self.table = ShardedFp8MultiHeadEmbedding(
            list_of_N=list(self.bucket_sizes),
            D=self.head_dim,
            block_size=_BLOCK,
            dtype=torch.bfloat16,
            tp_size=self.group_size,
            tp_rank=self.mapping.tp_rank - self.group_start,
            share_key=share_key,
            mxfp4=mxfp4,
        )
        self.table.to(
            device
        )  # the offsets buffer follows the model to the GPU; the host tables stay put
        self.table.load_weights(
            {
                "weight": Fp8Rows(emb, torch.uint8 if mxfp4 else torch.float8_e4m3fn),
                "scale": Fp8Rows(sc, torch.float8_e8m0fnu),
            }
        )
        # the k | v projection on the checkpoint's E4M3 bytes (an MXFP8 GEMM, as the engine's engram projection runs)
        w = weights[prefix + "wkv.weight"].to(device)
        assert w.dtype == torch.float8_e4m3fn, "the engram wkv is E4M3 in the checkpoint"
        self.wkv = w.contiguous()
        self.wkv_s = mxfp8_row_scales(weights[prefix + "wkv.scale"].to(device), tuple(w.shape))
        self.wkv_sw = mxfp8_swizzled_scales(self.wkv_s)
        self.wkv_alpha = torch.ones(1, dtype=torch.float32, device=device)
        self.q_weight.data.copy_(weights[prefix + "q_weight"])
        self.k_weight.data.copy_(weights[prefix + "k_weight"])
        # the fp32 product of the two gate norms the fused gate reads (the kernel's precomputed-product form)
        self.gate_w = (self.q_weight.float() * self.k_weight.float()).contiguous()

    def _lookup(self, ids: torch.Tensor, rows: Optional[slice] = None) -> torch.Tensor:
        """ids [m, cols] bucket-local -> [m, cols, head_dim] bf16, partial:
        what this rank's table shard does not hold is 0.

        The table's UVA kernel reads the pinned host rows and dequantizes them in one pass. Under head
        sharding it returns only this rank's heads, which are placed at their columns (only in `rows`, the
        tokens of this rank's exchange group, when given).
        """
        out = self.table(ids)
        if out.shape[1] == self.cols:
            return out
        full = out.new_zeros(ids.shape[0], self.cols, self.head_dim)
        rows = slice(None) if rows is None else rows
        full[rows, self.table.head_start : self.table.head_start + out.shape[1]] = out[rows]
        return full

    def precompute(self, ids: torch.Tensor, all_rank_num_tokens, stream: torch.cuda.Stream):
        """The engine's attention-DP `DeepseekV41Engram.precompute`, launched at the start of the forward: the hash rows
        of the exchange group gathered on the forward's stream (it orders the model's collectives), this rank's heads of
        them looked up on `stream` with half the SMs (`background`), beside the first layers. -> `(rows, counts, event)`
        for `forward(pre=...)`, or None when the group has no tokens. Head-sharded tables only."""
        sizes = all_rank_num_tokens
        g0 = self.group_start
        counts = [int(c) for c in sizes[g0 : g0 + self.group_size]]
        if not sum(counts):
            return None
        gathered = engine_allgather(ids, self.mapping, 0, sizes)
        group_ids = gathered.narrow(0, sum(sizes[:g0]), sum(counts))
        caller = torch.cuda.current_stream()
        stream.wait_stream(caller)
        group_ids.record_stream(stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            local = self.table(
                group_ids, background=True
            )  # [group tokens, this rank's heads, head_dim]
            event.record()
        local.record_stream(caller)
        return local, counts, event

    def forward(
        self, x: torch.Tensor, ids: torch.Tensor, all_rank_num_tokens, exchanged=None, pre=None
    ) -> torch.Tensor:
        """x [n, hc, d]; ids [n, cols] -> x + gate * value. `exchanged`: this forward's round of the attention-DP
        exchange (hashes exchanged and this rank's rows pushed by `ModelingV2Core._engram_exchange_launch`), or None.
        `pre`: this layer's `precompute` of the forward (the all-gather path's lookup, already launched), or None."""
        sizes = all_rank_num_tokens
        if pre is not None:
            # the lookup ran beside the first layers (`precompute`); here each member gets its tokens' rows of every
            # head shard, as the engine's `_local_token_embeddings`
            local, counts, event = pre
            torch.cuda.current_stream().wait_event(event)
            shards = engine_alltoallv_tokens(
                local.reshape(local.shape[0], -1).contiguous(),
                counts,
                self.group_ranks,
                self.mapping.tp_rank - self.group_start,
            )
            emb = torch.cat(shards.unbind(), dim=-1)  # [n, cols * head_dim]
        elif exchanged is not None:
            # This rank's push of the layer (side stream) before its receive, as the exchange protocol requires.
            torch.cuda.current_stream().wait_event(self.push_event)
            emb = engram_adp_receive(
                self.exchange, exchanged, self.hash_index
            )  # [n, cols * head_dim]
        elif (
            self.mapping.tp_size > 1 and self.mapping.enable_attention_dp and self.table.shard_heads
        ):
            # Ranks hold different tokens (a step past the exchange's rows): as the engine's attention-DP Engram
            # (`DeepseekV41Engram._local_token_embeddings`), gather the hash rows, look up this rank's heads for the
            # tokens of its exchange group only, and send every member its own tokens' rows of each head shard
            # (all-to-all within the group; source-major is head order). No all-reduce of every rank's rows: at large
            # steps that overran the engine all-reduce's symmetric buffer (illegal address at max_num_tokens 16384).
            g0, g1 = self.group_start, self.group_start + self.group_size
            counts = [int(c) for c in sizes[g0:g1]]
            if not sum(counts):
                emb = ids.new_empty((0, self.cols * self.head_dim), dtype=torch.bfloat16)
            else:
                gathered = engine_allgather(ids, self.mapping, 0, sizes)
                group_ids = gathered.narrow(0, sum(sizes[:g0]), sum(counts))
                local = self.table(group_ids)  # [group tokens, this rank's heads, head_dim]
                shards = engine_alltoallv_tokens(
                    local.reshape(local.shape[0], -1).contiguous(),
                    counts,
                    self.group_ranks,
                    self.mapping.tp_rank - g0,
                )
                emb = torch.cat(shards.unbind(), dim=-1)  # [n, cols * head_dim]
        elif self.mapping.tp_size > 1 and self.mapping.enable_attention_dp:
            # Row-sharded tables: gather ids, look up the local rows for the tokens of this rank's exchange group,
            # sum the shards, keep this rank's rows.
            gathered = engine_allgather(ids, self.mapping, 0, sizes)
            g0, g1 = self.group_start, self.group_start + self.group_size
            group_rows = (
                slice(sum(sizes[:g0]), sum(sizes[:g1]))
                if self.group_size < self.mapping.tp_size
                else None
            )
            emb = engine_allreduce(self.allreduce, self._lookup(gathered, group_rows))
            first = sum(sizes[: self.mapping.tp_rank])
            emb = emb[first : first + ids.shape[0]]
        elif self.mapping.tp_size > 1:
            emb = engine_allreduce(self.allreduce, self._lookup(ids))
        else:
            emb = self._lookup(ids)
        kv = csa2_mxfp8_linear(emb.flatten(1), self.wkv, self.wkv_s, self.wkv_sw, self.wkv_alpha)
        if x.shape[0] and self.gate_w is not None:  # one launch over the WKV output, as stock
            return engram_gate_fused(
                x.contiguous(), kv, self.q_weight, self.k_weight, self.gate_w, self.eps
            )
        key, value = torch.split(kv, [self.hc * self.dim, self.dim], dim=-1)
        key = key.unflatten(-1, (self.hc, self.dim))
        return engram_gate(x, key, value, self.q_weight, self.k_weight, self.eps, None)


_SF_VEC = 16
_SF_LINEAR = False  # the MoE runner reads the activation scales row-major, not swizzled
_MOE_MAX_T = 8192  # the runner's certified token column
_ROUTING_METHOD_INERT = 1  # routing is ours; the runner is handed ids and weights
_ACT_TYPE_SWIGLU = 0


def _moe_chunk_sizes(total: int) -> list:
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


class _Moe(nn.Module):
    """Routed experts: sqrtsoftplus routing (bias picks, raw scores weigh) and the NVFP4 grouped
    GEMMs of `fp4_block_scale_moe_runner`.

    Experts are split `moe_ep_size` ways (a contiguous window each) and each expert's intermediate
    width `moe_tp_size` ways, so every rank holds a partial sum. With attention DP the ranks hold
    different tokens: gather them, run, reduce-scatter. Without it every rank holds the same
    tokens: run, then an all-reduce (a reduce-scatter and an all-gather over padded rows).
    """

    def __init__(self, cfg, mapping):
        super().__init__()
        self.cfg, self.topk = cfg, cfg.num_experts_per_tok
        self.adp = bool(mapping.enable_attention_dp)
        self.group = list(range(mapping.world_size))
        self.tp, self.tp_rank = mapping.moe_tp_size, mapping.moe_tp_rank
        self.local = cfg.n_routed_experts // mapping.moe_ep_size
        self.offset = mapping.moe_ep_rank * self.local
        self.inter_t = _weights.padded_inter(cfg.moe_intermediate_size, self.tp)
        h, e, i = cfg.hidden_size, self.local, self.inter_t
        f8, u8, f32 = torch.float8_e4m3fn, torch.uint8, torch.float32

        def buf(*shape, dtype):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.gate_weight = buf(cfg.n_routed_experts, h, dtype=torch.bfloat16)
        self.gate_bias = buf(cfg.n_routed_experts, dtype=f32)
        self.fc1_w = buf(e, 2 * i, h // 2, dtype=u8)
        self.fc1_s = buf(e, 2 * i, h // _SF_VEC, dtype=f8)
        self.fc2_w = buf(e, h, i // 2, dtype=u8)
        self.fc2_s = buf(e, h, i // _SF_VEC, dtype=f8)
        self.o1, self.o1_gate, self.o2, self.clamp = (buf(e, dtype=f32) for _ in range(4))
        self.g1 = buf(1, dtype=f32)
        # The layers' shared all-to-all workspace (`ModelingV2Core._build_a2a`), or None: the gather / scatter path.
        self.a2a: Optional[MoeA2AWorkspace] = None

    def _route_quant(self, x: torch.Tensor):
        """Routing and the NVFP4 quantization of `x`'s rows in one launch (the built-in TRTLLM MoE's fused route):
        (ids, bf16 weights, x_fp4, row-major x_sf)."""
        logits = cublas_mm(x, self.gate_weight.t(), None, torch.float32)
        return dsv4_route_nvfp4_quant(
            logits, self.gate_bias, x, self.g1, self.topk, self.cfg.routed_scaling_factor
        )

    def _run_received(self, ids, w, xq, xsf) -> torch.Tensor:
        """This rank's experts over the rows the all-to-all brought it (`[ep * rows, ...]`, dead rows' ids -1),
        finalized: its partial sum of each row, in runner-sized chunks."""
        parts = []
        start = 0
        for size in _moe_chunk_sizes(ids.shape[0]):
            sl = slice(start, start + size)
            start += size
            parts.append(
                fp4_block_scale_moe_runner(
                    None, None, xq[sl], xsf[sl].reshape(-1).view(torch.float8_e4m3fn),
                    self.fc1_w, self.fc1_s, None, None, None, self.clamp,
                    self.fc2_w, self.fc2_s, None,
                    self.o1, self.o1_gate, self.o2,
                    self.cfg.n_routed_experts, self.topk, None, None,
                    self.inter_t, self.offset, self.local, None,
                    _ROUTING_METHOD_INERT, True, _ACT_TYPE_SWIGLU, w[sl], ids[sl],
                )[0]
            )  # fmt: skip
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _route(self, x: torch.Tensor):
        logits = cublas_mm(x, self.gate_weight.t(), None, torch.float32)
        scores = torch.sqrt(F.softplus(logits))
        ids = torch.topk(scores + self.gate_bias, self.topk, dim=-1)[1]
        w = torch.gather(scores, 1, ids)
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-20)  # norm_topk_prob
        return (w * self.cfg.routed_scaling_factor).to(torch.bfloat16), ids.to(torch.int32)

    def _experts(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for chunk in torch.split(x, _moe_chunk_sizes(x.shape[0]), dim=0):
            topk_w, topk_ids = self._route(chunk)
            xq, xsf = fp4_quantize(chunk, self.g1, _SF_VEC, False, _SF_LINEAR)
            parts.append(
                fp4_block_scale_moe_runner(
                    None, None, xq, xsf.view(torch.float8_e4m3fn),
                    self.fc1_w, self.fc1_s, None, None, None, self.clamp,
                    self.fc2_w, self.fc2_s, None,
                    self.o1, self.o1_gate, self.o2,
                    self.cfg.n_routed_experts, self.topk, None, None,
                    self.inter_t, self.offset, self.local, None,
                    _ROUTING_METHOD_INERT, True, _ACT_TYPE_SWIGLU, topk_w, topk_ids,
                )[0]
            )  # fmt: skip
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def forward(self, x: torch.Tensor, all_rank_num_tokens) -> torch.Tensor:
        n, ws = x.shape[0], len(self.group)
        if self.adp and self.a2a is not None:
            top = max(int(c) for c in all_rank_num_tokens)
            if top == 0:  # every rank sees the same counts: no rank joins a collective
                return x.new_empty(0, x.shape[-1])
            if top <= self.a2a.max_rows:  # as the built-in TRTLLM MoE: the one-sided all-to-all
                return a2a_experts(
                    self.a2a, x, all_rank_num_tokens, self.cfg.n_routed_experts, self.topk,
                    self._route_quant, self._run_received,
                )  # fmt: skip
        if self.adp:
            rows = max(int(c) for c in all_rank_num_tokens)
            xp = x if rows == n else F.pad(x, [0, 0, 0, rows - n])
            routed = cat_reducescatter(
                self._experts(cat_allgather(xp.contiguous(), None, self.group)), None, self.group
            )
            return routed if rows == n else torch.split(routed, [n, rows - n], dim=0)[0]
        rows = -(-n // ws) * ws
        xp = x if rows == n else F.pad(x, [0, 0, 0, rows - n])
        part = self._experts(xp.contiguous())
        out = cat_allgather(cat_reducescatter(part, None, self.group), None, self.group)
        return out if rows == n else torch.split(out, [n, rows - n], dim=0)[0]


@dataclass
class _HcState:
    """The residual stream between layers.

    `pre_mix` is the previous sub-layer's `pre`, which the next boundary collapses with. A *deferred* state
    (`post` is set) has not applied the last FFN's post update yet: `x_prev` is that FFN's output, a
    tensor-parallel partial sum (a `MoeUnfinalized` when the next boundary also does the top-k combine) until
    `reduced_x_prev` all-reduces it with `allreduce`, and the next layer's boundary kernel folds the update in,
    so the stream is never materialized in between. `coeff_event` marks coefficients still being computed on
    the side stream.
    """

    residual: torch.Tensor  # [n, hc, d] bf16
    pre_mix: torch.Tensor  # [n, hc] fp32
    post: Optional[torch.Tensor] = None  # [n, hc] fp32
    comb: Optional[torch.Tensor] = None  # [n, hc, hc] fp32
    x_prev: Optional[torch.Tensor] = None  # [n, d] bf16 or MoeUnfinalized
    allreduce: Optional[Callable] = None  # set while x_prev is still a partial sum
    coeff_event: Optional[torch.cuda.Event] = None

    @property
    def deferred(self) -> bool:
        return self.post is not None

    def wait_coefficients(self) -> None:
        if self.coeff_event is not None:
            torch.cuda.current_stream().wait_event(self.coeff_event)
            self.coeff_event = None

    def reduced_x_prev(self) -> torch.Tensor:
        if self.allreduce is not None:
            self.x_prev = self.allreduce(self.x_prev)  # a layer's `_reduce` / `_reduce_unfinalized`
            self.allreduce = None
        return self.x_prev


class _Layer(nn.Module):
    def __init__(
        self, model_config: ModelConfig, layer_idx: int, aux_stream_dict, engram_slot, mega_cfg=None
    ):
        super().__init__()
        cfg = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping
        self.dim = cfg.hidden_size
        self.hc = cfg.hc_mult
        self.sinkhorn = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.topk = cfg.num_experts_per_tok

        self.attn = _Attention(cfg, layer_idx, self.mapping)
        sparse = getattr(model_config, "sparse_attention_config", None)
        self.attn.heuristic_topk = bool(getattr(sparse, "enable_heuristic_topk", False))
        self.attn.prefill_self_sampling = bool(getattr(sparse, "use_self_sampling_topk", True))
        self.attn_norm = nn.Parameter(
            torch.ones(self.dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.ffn_norm = nn.Parameter(
            torch.ones(self.dim, dtype=torch.bfloat16), requires_grad=False
        )
        mix = (2 + self.hc) * self.hc
        for n in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{n}_fn",
                nn.Parameter(torch.empty(mix, self.hc * self.dim), requires_grad=False),
            )
            setattr(self, f"hc_{n}_base", nn.Parameter(torch.empty(mix), requires_grad=False))
            setattr(self, f"hc_{n}_scale", nn.Parameter(torch.empty(3), requires_grad=False))
            setattr(
                self, f"hc_{n}_bw", None
            )  # `hc_boundary.prepare_weight(fn)`, built after the weights are loaded
        self.moe = (
            _Moe(cfg, self.mapping) if mega_cfg is None else _MegaMoe(cfg, mega_cfg, layer_idx)
        )
        # Fused into the MegaMoE launch, the shared expert has no module (and no bf16 copy of its weights).
        fused = mega_cfg is not None and mega_cfg.fused_shared
        self.shared = (
            None
            if fused
            else SharedExpertFp8(self.dim, cfg.moe_intermediate_size, cfg.swiglu_limit)
        )
        self.engram = _Engram(cfg, self.mapping, engram_slot) if engram_slot is not None else None
        # Attention data parallel: every rank owns its own tokens, so there is no tensor-parallel sum to take.
        self.lamport = None
        self._coeff_stream = (aux_stream_dict or {}).get(AuxStreamType.MhcCoefficients)
        # the attention's generation-step side work (index weights /
        # compressed rows, or the query projection), as the engine
        self.attn.side_stream = (aux_stream_dict or {}).get(AuxStreamType.Attention)
        self.attn.query_stream = (aux_stream_dict or {}).get(
            AuxStreamType.MlaIndexer
        )  # index-source q_b, next to the index q_b
        self.attn.weights_stream = (aux_stream_dict or {}).get(
            AuxStreamType.MlaIndexerAux
        )  # index weights, next to the compressor
        self.attn.gen_fmha_graph_rows = max(
            csa2_mxfp8_graph_rows(model_config), default=0
        )  # the gen_fmha staging tile's rows
        self._events = {
            k: (torch.cuda.Event(), torch.cuda.Event()) for k in ("attn", "ffn", "shared")
        }
        # the shared expert beside the routed experts (the built-in MoE's shared-expert overlap): its own stream
        self._shared_stream = (aux_stream_dict or {}).get(AuxStreamType.MoeShared)
        self._coeff_state = {}  # per boundary: the one-launch coefficient kernels' counters (`post_load`)
        self.allreduce = None

    def post_load(self, lamport) -> None:
        if hc_boundary.supported(self.dim):
            for n in ("attn", "ffn"):
                setattr(self, f"hc_{n}_bw", hc_boundary.prepare_weight(getattr(self, f"hc_{n}_fn")))
                self._coeff_state[n] = hc_lagged_coefficients.prepare_state(self.hc_attn_fn.device)
            torch.cuda.current_stream().synchronize()
        self.lamport = (
            lamport  # None: the split boundaries reduce with the engine's all-reduce instead
        )

    def _reduce(self, x: torch.Tensor):
        return x if self.allreduce is None else engine_allreduce(self.allreduce, x)

    def _reduce_unfinalized(self, parts) -> torch.Tensor:
        return self._reduce(moe_unfinalized_finalize(parts))

    def _ffn(self, x: torch.Tensor, all_rank_num_tokens, unfinalized: bool):
        # The experts are sharded over the ranks' tokens (MegaMoE / all-to-all inside `moe`), so the routed
        # output is already complete; the shared expert is replicated and meets it here once. Fused (MegaMoE),
        # the routed output already includes it.
        assert not unfinalized
        if self.shared is None:
            return self.moe(x, all_rank_num_tokens)
        if self._shared_stream is None or not do_multi_stream() or x.shape[0] == 0:
            routed = self.moe(x, all_rank_num_tokens)
            return torch.add(routed, self.shared(x)).to(x.dtype)  # bf16: fp32 sum, one rounding
        # Under a multi-stream (CUDA-graph) forward the shared expert runs on its own stream beside the routed
        # experts' route, dispatch, GEMMs and combine, as the built-in MoE overlaps it. `x` stays referenced until the
        # join below, so the side stream's reads of it need no `record_stream`.
        fork, done = self._events["shared"]
        fork.record()
        with torch.cuda.stream(self._shared_stream):
            fork.wait()
            shared = self.shared(x)
            done.record()
        routed = self.moe(x, all_rank_num_tokens)
        torch.cuda.current_stream().wait_event(done)
        # one launch: torch's bf16 add sums in fp32 and rounds once, the bits of the fp32 add cast back
        return torch.add(routed, shared).to(x.dtype)

    def _split(self, n: int, ffn: bool = False, post_update: bool = True) -> bool:
        """Decode-sized rows under a multi-stream (CUDA-graph) forward: the boundary's critical half runs in one
        launch and its coefficients on the side stream, overlapping the sub-layer (the built-in `_mhc_split`, #22360).
        An attention entry with a post update splits up to 1023 rows, the others up to 4096 (`hc_split_enabled`);
        past 64 rows only where the one-launch coefficient kernel runs (`hc_lagged_coefficients.ready`)."""
        if not (self._coeff_stream is not None and do_multi_stream()):
            return False
        if not hc_split_enabled(n, self.dim, self.hc, mega_mhc=post_update and not ffn):
            return False
        which = "ffn" if ffn else "attn"
        if n <= _ANY_COEFF_MAX_ROWS:
            return True  # an uncompiled small-batch kernel leaves these rows to the two-launch coefficients
        fn, state = getattr(self, f"hc_{which}_fn"), self._coeff_state.get(which)
        return state is not None and hc_lagged_coefficients.ready(
            n, self.dim, self.sinkhorn, state, fn
        )

    def _coefficients(self, which: str, residual: torch.Tensor):
        """The boundary's own (pre, post, comb) on the side stream -> (pre, post, comb, event to wait on): the
        built-in model's one-launch coefficient kernels by row count (`hc_lagged_coefficients`)."""
        args = (getattr(self, f"hc_{which}_fn"), getattr(self, f"hc_{which}_bw"), getattr(self, f"hc_{which}_scale"),
                getattr(self, f"hc_{which}_base"), self.hc, self.sinkhorn, self.norm_eps, self.hc_eps,
                _HC_POST_MULT)  # fmt: skip
        fork, done = self._events[which]
        fork.record()
        with torch.cuda.stream(self._coeff_stream):
            fork.wait()
            pre, post, comb = hc_lagged_coefficients(residual, self._coeff_state[which], *args)
            done.record()
        return pre, post, comb, done

    def _boundary(self, which, x_prev, residual, post_prev, comb_prev, pre_mix, mxfp8=None):
        """One lagged boundary of the attention (`which = "attn"`) or FFN sub-layer.

        -> (residual with the previous post update applied, own pre, own post, own comb, normalized input).
        The single-pass kernel when this many rows are in its range (and compiled, under graph capture),
        else the post mapping and the two-launch pre-mapping. With `mxfp8` given (True or False) a sixth element: the
        normalized input in MXFP8 from the same launch when True and the single-pass kernel ran, else None (the
        attention then quantizes it itself). Last element: the event `comb` is pending on, or None. Under a multi-stream
        (CUDA-graph) forward the single-pass boundary leaves the Sinkhorn iterations of `comb` to the coefficient stream
        (the engine's deferred boundary), overlapping the sub-layer that follows; the next boundary waits for it."""
        norm = self.attn_norm if which == "attn" else self.ffn_norm
        fn, bw = getattr(self, f"hc_{which}_fn"), getattr(self, f"hc_{which}_bw")
        scale, base = getattr(self, f"hc_{which}_scale"), getattr(self, f"hc_{which}_base")
        n = residual.shape[0]
        args = (self.hc, self.sinkhorn, self.norm_eps, self.norm_eps, self.hc_eps, _HC_POST_MULT)
        with_post, mx = x_prev is not None, bool(mxfp8)
        # A multi-stream forward (graph warm-up included) defers Sinkhorn
        # and compiles only that variant, so it is checked first.
        defer = (
            bw is not None
            and self._coeff_stream is not None
            and do_multi_stream()
            and hc_boundary.available(
                n, self.dim, with_post, self.sinkhorn, deferred=True, mxfp8=mx
            )
        )
        if with_post and hc_mega_boundary.available(n, self.dim):
            # 1024+ rows (dep8's full decode steps of 256+ requests): DeepGEMM's one-launch boundary, as the engine's
            # fused lagged boundary runs it from 1024 rows (`mhc_fused_lagged` -> `mega_mhc`), not the post mapping and
            # the two-launch pre-mapping. The same launch writes the q_a |
            # kv GEMM's MXFP8 input (stock's `prequantized`).
            out = hc_mega_boundary(
                x_prev,
                residual,
                post_prev,
                comb_prev,
                pre_mix,
                fn,
                scale,
                base,
                norm,
                *args,
                mxfp8=mx,
            )
            return out + ((None,) if mxfp8 is False else ()) + (None,)
        if bw is not None and (
            defer or hc_boundary.available(n, self.dim, with_post, self.sinkhorn, mxfp8=mx)
        ):
            side = (self._coeff_stream, *self._events[which]) if defer else None
            out = hc_boundary(
                x_prev,
                residual,
                post_prev,
                comb_prev,
                pre_mix,
                fn,
                bw,
                scale,
                base,
                norm,
                *args,
                mxfp8=mx,
                sinkhorn_side=side,
            )
            out = out + (None,) if mxfp8 is False else out
            return out + (side[2] if defer else None,)
        if x_prev is not None:
            residual = hc_post_mapping(x_prev, residual, post_prev, comb_prev)
        pre, post, comb, h = hc_pre_mapping(residual, pre_mix, fn, scale, base, norm, *args)
        return (residual, pre, post, comb, h) + ((None,) if mxfp8 is not None else ()) + (None,)

    def _split_boundary(
        self,
        which,
        x_prev,
        partial,
        residual,
        post_prev,
        comb_prev,
        pre_mix,
        mxfp8=False,
        residual_after_wait=False,
    ):
        """Critical half of a split boundary -> (residual, normalized input, its MXFP8 copy or None). `partial`:
        `x_prev` still has to be summed over the group, which the kernel does in the same launch when the workspace is
        there. `mxfp8`: the same launch also writes the copy the q_a | kv GEMM reads past the low-M rows, as the
        engine's split boundary does (`takes_mxfp8_input_for`), instead of a separate quantize in the GEMM.
        `residual_after_wait`: the kernel reads `residual` after its dependency wait (a residual no joined side-stream
        work released)."""
        norm = self.attn_norm if which == "attn" else self.ffn_norm
        if partial and x_prev is not None and self.lamport is not None:
            out = hc_allreduce_post_collapse_norm(
                x_prev, residual, post_prev, comb_prev, pre_mix, norm, self.norm_eps, self.lamport
            )
            return (*out, None)
        if x_prev is None:
            post_prev = comb_prev = None
        out = hc_post_collapse_norm(
            x_prev,
            residual,
            post_prev,
            comb_prev,
            pre_mix,
            norm,
            self.norm_eps,
            mxfp8=mxfp8,
            residual_after_wait=residual_after_wait and x_prev is not None,
        )
        return out if mxfp8 else (*out, None)

    def publish_global(self, residual: torch.Tensor, pre_mix: torch.Tensor, ctx) -> None:
        """Bounded-replay boundary (the last unpooled KV owner, stock's `prepare_ced_global_kv`): publish the compressed
        KV and index-key rows of every row of the step from the resolved stream, normalized by the layer's own attention
        boundary, before the rows are narrowed to the final window."""
        h = self._boundary("attn", None, residual, None, None, pre_mix)[4]
        self.attn.publish_global(h, ctx)

    def idle(self, counts) -> None:
        """A rank with no row in the decoder half still joins the experts' collectives of its peers (stock skips the
        attention and the mHC of an empty decoder input)."""
        self.moe(self.attn_norm.new_empty(0, self.dim), counts)

    def fused_close(self, st: _HcState, norm_weight: torch.Tensor) -> Optional[torch.Tensor]:
        """`rmsnorm(collapse(post_update(last FFN output)))` of the whole stream close in one launch (with the last
        all-reduce), or None where the split boundary does not apply."""
        if not (
            st.deferred
            and st.allreduce is not None
            and self.lamport is not None
            and self._split(st.residual.shape[0])
        ):
            return None
        st.wait_coefficients()
        return hc_allreduce_post_collapse_norm(
            st.x_prev,
            st.residual,
            st.post,
            st.comb,
            st.pre_mix,
            norm_weight,
            self.norm_eps,
            self.lamport,
        )[1]

    def forward(
        self,
        st: _HcState,
        ctx,
        attn_metadata,
        engram_ids,
        spec_metadata=None,
        engram_round=None,
        engram_pre=None,
        capture_rows=None,
    ) -> _HcState:
        """`st` -> the state after this layer (deferred when the next boundary can take the FFN's post update).
        `capture_rows`: in the decoder half of a bounded replay, the step rows the state's rows are (the draft's
        captures go to those rows)."""
        st.wait_coefficients()
        residual, pre_mix = st.residual, st.pre_mix
        deferred = st.deferred
        if self.engram is not None:
            if deferred:  # the engram reads and rewrites the materialized stream
                residual = hc_post_mapping(st.reduced_x_prev(), residual, st.post, st.comb)
                deferred = False
            residual = self.engram(
                residual,
                engram_ids[:, self.engram.hash_index],
                attn_metadata.all_rank_num_tokens,
                engram_round,
                None if engram_pre is None else engram_pre.get(self.engram.hash_index),
            )
        split = self._split(residual.shape[0], post_update=deferred)
        ffn_split = self._split(residual.shape[0], ffn=True)
        lamport = self.lamport if split else None
        a_event = None
        if split:
            partial = deferred and st.allreduce is not None
            x_prev = (
                (st.x_prev if partial and lamport is not None else st.reduced_x_prev())
                if deferred
                else None
            )
            residual, h, h_mx = self._split_boundary(
                "attn",
                x_prev,
                partial,
                residual,
                st.post,
                st.comb,
                pre_mix,
                mxfp8=residual.shape[0] > SMALL_M_ROWS,
            )
            a_pre, a_post, a_comb, a_event = self._coefficients("attn", residual)
        else:
            x_prev = st.reduced_x_prev() if deferred else None
            # past the low-M kernel's rows the q_a | kv GEMM reads MXFP8: the
            # boundary writes that copy too (no separate quantize)
            residual, a_pre, a_post, a_comb, h, h_mx, a_event = self._boundary(
                "attn",
                x_prev,
                residual,
                st.post,
                st.comb,
                pre_mix,
                mxfp8=residual.shape[0] > SMALL_M_ROWS,
            )
        capture_done = None
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            # The draft reads the residual stream entering its target layers (the hc copies, flattened). As the engine's
            # layer does, it is captured on a side stream beside the attention (multi-stream forwards; else right here)
            # and joined before the FFN boundary.
            on_side = getattr(spec_metadata, "capture_on_side_stream", None)
            rows = {} if capture_rows is None else {"row_indices": capture_rows}
            if on_side is not None:
                capture_done = on_side(self.layer_idx, residual.flatten(1), **rows)
            else:
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, residual.flatten(1), **rows
                )
        x_attn = self.attn(h, ctx, h_mx)  # this rank's partial sum
        if capture_done is not None:
            torch.cuda.current_stream().wait_event(capture_done)
        f_event = None
        if ffn_split:
            if a_event is not None:
                torch.cuda.current_stream().wait_event(a_event)
            if lamport is None:
                x_attn = self._reduce(x_attn)
            # After a single-pass attention entry (no coefficients joined from the side stream) the kernel reads the
            # residual after its dependency wait, as the built-in split FFN boundary does.
            residual, h, _ = self._split_boundary(
                "ffn", x_attn, lamport is not None, residual, a_post, a_comb, a_pre,
                residual_after_wait=not split,
            )  # fmt: skip
            f_pre, f_post, f_comb, f_event = self._coefficients("ffn", residual)
        else:
            x_attn = self._reduce(x_attn)
            if (
                a_event is not None
            ):  # the attention boundary's comb is finished on the coefficient stream
                torch.cuda.current_stream().wait_event(a_event)
            residual, f_pre, f_post, f_comb, h, f_event = self._boundary(
                "ffn", x_attn, residual, a_post, a_comb, a_pre
            )
        unfinalized = lamport is not None
        x = self._ffn(h, attn_metadata.all_rank_num_tokens, unfinalized)
        # Leave the partial sum and the post update to the next boundary (or the stream close).
        return _HcState(
            residual, f_pre, f_post, f_comb, x,
            self._reduce_unfinalized if unfinalized else (self._reduce if self.allreduce is not None else None),
            f_event,
        )  # fmt: skip


def _step_positions(attn_metadata, position_ids, n: int) -> torch.Tensor:
    """The step's token positions `[n]` int32. A generation step (no context rows) reads the CSA2 metadata's live query
    positions (`csa2_positions`: prepared from the cached lengths, refreshed on the device by `on_update_kv_lens`, which
    the engine runs before the model), as stock's CSA2 layers do, so the cache rows and compressor state it writes are
    those stock writes; the engine's `position_ids` can differ from them (the warm-up's max-length dummy request carries
    position 0 at a cache length of max_seq_len - 1). A step with context rows, whose CSA2 prepare the metadata may
    leave to stock's first CSA2 layer, and metadata without CSA2 positions (unit tests) read `position_ids`."""
    live = getattr(attn_metadata, "csa2_positions", None)
    if isinstance(live, torch.Tensor) and getattr(attn_metadata, "num_contexts", 1) == 0:
        assert live.dtype == torch.int32 and live.numel() >= n, (
            "the CSA2 metadata's positions cover the step's rows"
        )
        return live[:n]
    return position_ids.reshape(-1).to(torch.int32).contiguous()


class ModelingV2Core(DecoderModel):
    # Whether the engine may capture a context step as a whole-body prefill CUDA graph (`prefill_cuda_graph_backend`):
    # never. Stock declines it under attention DP with a decoder replay plan (`DeepseekV41Model`); this target's
    # context path (`PrefillTarget`) runs eagerly in every layout and asserts it is not captured.
    supports_prefill_cuda_graph = False

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        csa2_mxfp8_gemm_tune_rows(
            csa2_mxfp8_graph_rows(model_config)
        )  # CUDA-graph step rows: MXFP8 GEMM tactics tuned at warm-up
        # The DSpark draft's MXFP8 LM head (engine code) runs `[graph batch x block, hidden] x [vocab, hidden]^T`: its
        # rows, tuned at warm-up as stock's engine tunes them (`csa2_mxfp8_tune_engine_gemm`), else it runs the slower
        # fallback tactic.
        spec = getattr(model_config, "spec_config", None)
        block = (
            int(getattr(spec, "block_size", None) or getattr(spec, "max_draft_len", 0) or 0)
            if spec is not None
            else 0
        )
        graphs = getattr(model_config, "cuda_graph_batch_sizes", None) or ()
        self._head_tune_rows = tuple(sorted({int(b) * block for b in graphs})) if block else ()
        cfg = model_config.pretrained_config
        assert torch.cuda.get_device_capability() == _SM, (
            f"this target is sm_{_SM[0]}{_SM[1]}; got {torch.cuda.get_device_capability()}"
        )
        self.cfg = cfg
        self.mapping = model_config.mapping
        self.hc = cfg.hc_mult
        self.norm_eps = cfg.rms_norm_eps
        # The checkpoint directory (engram's tokenizer, the node-shared table key), as stock reads it: the path the
        # config was loaded from (`ModelConfig.from_pretrained` fills `_name_or_path` from its checkpoint dir; the
        # composite's travels on the text config, see `text_model_config`).
        self.checkpoint_dir = getattr(cfg, "_v41_checkpoint_dir", None) or getattr(
            cfg, "_name_or_path", None
        )
        assert self.checkpoint_dir and os.path.isdir(self.checkpoint_dir), (
            "the config carries no checkpoint directory (`_name_or_path`): load it with ModelConfig.from_pretrained"
        )
        self.aux_stream_dict = {k: torch.cuda.Stream() for k in AuxStreamType}
        # stock's `DeepseekV4Model.embed_tokens` (`use_masked_embedding_kernel=True`)
        self.embed_tokens = token_embedding_create(
            cfg.vocab_size, cfg.hidden_size, torch.bfloat16, masked_kernel=True
        )
        slots = {lid: i for i, lid in enumerate(cfg.engram_layer_ids)}
        # None unless `moe_config.backend` is MEGAMOE_CUTEDSL (then every layer's MoE is the MegaMoE one).
        self.mega_cfg = resolve_megamoe_config(model_config, cfg)
        self.mega_build = None
        self.layers = nn.ModuleList(
            [
                _Layer(model_config, i, self.aux_stream_dict, slots.get(i), self.mega_cfg)
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.engram_hash: Optional[EngramHash] = None
        self.engram_exchange = (
            None  # the attention-DP peer-memory exchange (`_engram_exchange_setup`), or None
        )
        self._engram_stream: Optional[torch.cuda.Stream] = None  # the exchange's pushes
        # Engram look-back state (see `EngramHash.lookback`): a ring of compressed ids per request row (`hist`), the
        # step's request -> row map (`req_slot`) and which request owns which row, as stock owns them: a request keeps
        # its row until the engine releases it (`release_request_state`); the ring holds every request the rank can
        # (`EngramHistorySlots`, sized at the first step). Its last row is the dummy that padded (CUDA-graph) and
        # unknown requests write into.
        self.engram_slots: Optional[EngramHistorySlots] = None
        self._prev_reqs = 0
        self._prev_ids = None  # the request ids `on_step` last placed in `req_slot`, in row order
        self._recovery = None  # this step's Encoder recovery `(hist_end, floor)` per request (`csa2_encoder_recovery`)
        self._recovery_logged = False
        self._staging = {}  # pinned host staging of `on_step`'s uploads (`copy_host_to_device`)
        # (rows, rows per request) -> `_uniform_rows`: the CUDA-graph shapes (bounded by the graph batch sizes, read
        # by the captured graphs) kept, the other eager shapes the most recent `_EAGER_ROWS_CACHED` only
        self._rows_cache: dict = {}
        self._eager_rows_cache: "collections.OrderedDict" = collections.OrderedDict()
        self._fwd_stream: Optional[torch.cuda.Stream] = (
            None  # the stream the forward runs on (set by the first forward)
        )
        self._prefill = PrefillTarget()
        self._decode = DecodeTarget()
        # Bounded decoder replay (stock's policy, `csa2_replay_policy`). The executor reads these names to decide
        # whether the cache manager's private decoder pages and the replay plan apply to this model.
        self.decoder_replay_split, self.decoder_replay_window = csa2_replay_policy(
            cfg, model_config, captures=True
        )
        self.ced_kv_precompute = self.decoder_replay_split is not None

    # ---- weights: see weights.py ---------------------------------------------------
    def load(self, weights: dict) -> None:
        if self.mega_cfg is not None:
            attach_build(
                self
            )  # collective: every bucket's symmetric regions, before any weight and any graph capture
        _weights.load(self, weights)
        if self.mega_cfg is not None:
            prime(self)  # collective: compile and fence every bucket's kernel with zero rows

    def _build_a2a(self) -> None:
        """Every backend but MegaMoE routes its experts over the NVLink one-sided all-to-all, as the built-in model's
        TRTLLM MoE (`a2a_experts`): one workspace for every layer, sized for the most rows a rank can carry into a
        layer (the step's token budget). Collective: every rank builds it once, in `post_load`, in the same order."""
        if self.mega_cfg is not None:
            return
        c = self.cfg
        a2a = build_a2a_workspace(
            self.mapping, int(self.model_config.max_num_tokens), c.hidden_size, c.num_experts_per_tok,
            c.n_routed_experts,
        )  # fmt: skip
        if a2a is None:
            return
        for layer in self.layers:
            layer.moe.a2a = a2a
        logger.info(
            f"DeepSeek-V4.1 {self.mapping.world_size}-rank experts on the NVLink all-to-all: {a2a.max_rows} rows per "
            f"rank, {a2a.workspace.shape[1] / 2**20:.0f} MiB per rank, sentinel transport up to "
            f"{a2a.sentinel_max_rows} rows per rank"
        )

    def post_load(self) -> None:
        self._build_a2a()
        # No tensor-parallel group to reduce over under attention DP: `hc_lamport_workspace` answers None.
        lamport = hc_lamport_workspace(self.mapping, self.model_config.allreduce_strategy)
        for layer in self.layers:
            layer.post_load(lamport)
            if layer.engram is not None:
                layer.engram.table.post_load_weights()  # an importer of a node-shared table waits for its owner
        logger.info(
            "modeling_v2 deepseek_v41 engram config: "
            + ", ".join(
                f"{k}={getattr(self.cfg, k, '<missing>')!r}"
                for k in (
                    "engram_layer_ids",
                    "engram_vocab_size",
                    "engram_num_embeddings",
                    "engram_max_ngram_size",
                    "engram_n_heads",
                    "engram_head_dim",
                    "engram_pad_token_id",
                    "engram_compressed_vocab_size",
                )
            )  # fmt: skip
        )
        self.engram_hash = EngramHash(self.cfg, self.checkpoint_dir, torch.device("cuda"))
        self._engram_exchange_setup()
        dev = torch.device("cuda")
        c = self.cfg
        rs = c.rope_scaling
        n_pos = c.max_position_embeddings
        # the engine's [positions, rope dims] fp32 layout only (`csa2_stock_rope_table`): the (cos, sin) pair tables
        # are converted once and not kept
        self.rope = {
            0: csa2_stock_rope_table(
                csa2_rope_table(
                    c.qk_rope_head_dim,
                    n_pos,
                    0,
                    c.rope_theta,
                    rs["factor"],
                    rs["beta_fast"],
                    rs["beta_slow"],
                    dev,
                )
            ),
            1: csa2_stock_rope_table(
                csa2_rope_table(
                    c.qk_rope_head_dim,
                    n_pos,
                    rs["original_max_position_embeddings"],
                    c.compress_rope_theta,
                    rs["factor"],
                    rs["beta_fast"],
                    rs["beta_slow"],
                    dev,
                )
            ),
        }
        self.engram_slots = EngramHistorySlots(_RING, self.engram_hash.pad_id, _MAX_REQS, dev)
        self._prev_ids = None  # a fresh buffer: the next `on_step` writes every row

    def _engram_exchange_setup(self) -> None:
        """Under attention DP, the engine's peer-memory engram exchange (`engram_adp_exchange_create`) in place of the
        all-gather of the hash rows + lookup + all-reduce of a zero-filled `[all ranks' tokens, cols, head_dim]` buffer.
        Collective over the tensor-parallel group; built once (`post_load` may run again). Its peer buffers hold
        `max_num_tokens` rows per rank (stock sizes them only up to 256 rows and keeps NCCL above that, which would
        leave the dep8 c7000 engine (1280) on the all-gather path; the rows only size the buffers: 1280 rows of 24
        heads x 256 are ~79 MiB); a forward with more rows on some rank keeps the all-gather path."""
        if self.engram_exchange is not None:
            return
        m = self.mapping
        engrams = [layer.engram for layer in self.layers if layer.engram is not None]
        if not (m.enable_attention_dp and m.tp_size > 1) or not engrams:
            return
        capacity = int(getattr(self.model_config, "max_num_tokens", 0) or 0)
        reason = None
        if not 0 < capacity <= _ENGRAM_EXCHANGE_MAX_ROWS:
            reason = f"max_num_tokens {capacity} outside (0, {_ENGRAM_EXCHANGE_MAX_ROWS}]"
        elif m.has_pp() or m.cp_size > 1:
            reason = "pipeline or context parallelism is enabled"
        elif len(engrams) > 2 or [e.hash_index for e in engrams] != list(range(len(engrams))):
            reason = f"{len(engrams)} engram layers (supported: 1 or 2, in hash order)"
        elif not all(
            e.table.shard_heads and e.group_start == engrams[0].group_start for e in engrams
        ):
            reason = "the engram tables are not head shards of one exchange group"
        if reason is not None:
            logger.info(f"modeling_v2 deepseek_v41 engram ADP exchange off: {reason}")
            return
        e0 = engrams[0]
        exchange = engram_adp_exchange_create(
            m, e0.group_start, e0.group_size, capacity, e0.cols, e0.head_dim, len(engrams)
        )
        if exchange is None:
            logger.info(
                "modeling_v2 deepseek_v41 engram ADP exchange off: no peer memory (IPC or MNNVL)"
            )
            return
        self.engram_exchange = exchange
        self._engram_stream = torch.cuda.Stream()
        for e in engrams:
            e.exchange = exchange
            e.push_event = torch.cuda.Event()
        logger.info(
            f"modeling_v2 deepseek_v41 engram ADP exchange: groups of {e0.group_size} of {m.tp_size} ranks, "
            f"{capacity} rows per rank, {len(engrams)} layers"
        )

    def _engram_exchange_launch(self, engram_ids: torch.Tensor, attn_metadata):
        """This forward's round of the engram exchange, or None (no exchange, or a rank past its capacity: every
        rank decides alike from the counts they all hold). The hashes of every engram layer go out in one launch on
        the forward's stream; each layer's push (this rank's heads of every group token, into the requesters'
        buffers) runs on the engram stream and records the layer's event, which the layer waits on before receiving
        (as stock pushes at the start of the forward and receives at the layer)."""
        ex = self.engram_exchange
        counts = attn_metadata.all_rank_num_tokens
        if not engram_adp_takes(ex, counts):
            return None
        engrams = [layer.engram for layer in self.layers if layer.engram is not None]
        e0 = engrams[0]
        hashes = [engram_ids[:, e.hash_index] for e in engrams]
        round_ = engram_adp_exchange_hashes(
            ex, hashes, list(counts[e0.group_start : e0.group_start + e0.group_size])
        )
        caller, stream = torch.cuda.current_stream(), self._engram_stream
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            for e in engrams:
                engram_adp_push(ex, round_, e.hash_index, e.table)
                e.push_event.record()
        round_.gathered.record_stream(stream)
        round_.counts.record_stream(stream)
        return round_

    def _engram_precompute(self, engram_ids: torch.Tensor, attn_metadata):
        """Without this forward's exchange round, under attention DP with head-sharded tables: every engram layer's
        lookup launched at the start of the forward (`_Engram.precompute`), as the engine's `_precompute_engram` does
        (its precompute start layer is 0 outside context overlap). -> {hash index: precompute} or None."""
        m = self.mapping
        engrams = [layer.engram for layer in self.layers if layer.engram is not None]
        if (
            not engrams
            or not (m.enable_attention_dp and m.tp_size > 1)
            or not all(e.table.shard_heads for e in engrams)
        ):
            return None
        stream = self.aux_stream_dict[AuxStreamType.EngramPrecompute]
        return {
            e.hash_index: e.precompute(
                engram_ids[:, e.hash_index], attn_metadata.all_rank_num_tokens, stream
            )
            for e in engrams
        }

    @property
    def hist(self) -> torch.Tensor:
        return self.engram_slots.hist

    @property
    def req_slot(self) -> torch.Tensor:
        return self.engram_slots.req_slot

    def release_request_state(self, request_id: int) -> None:
        """Engine hook (completion, cancellation): the request's look-back row is free for the next request. The next
        `on_step` re-places every row: a request released and rescheduled in the same position gets a fresh row and its
        look-back seeded, as a new request does."""
        if self.engram_slots is not None:
            self.engram_slots.release(request_id)
        self._prev_ids = None

    def on_step(self, scheduled_requests, kvm) -> None:
        """Engine hook, outside graph capture: give every request of the step a history slot.

        A request new to this engine is seeded with its `max_n - 1` committed tokens before the
        step's first position. That is exact: it has never run here, so nothing is in flight and
        the host list is complete. Later steps maintain the ring on the device themselves.
        """
        reqs = list(scheduled_requests.context_requests) + list(
            scheduled_requests.generation_requests
        )
        # every request the rank can hold has a row (stock's history never drops an active request's tokens)
        self.engram_slots.reserve(int(getattr(kvm, "max_batch_size", 0) or 0))
        ids = [r.py_request_id for r in reqs]
        if ids == self._prev_ids:
            return  # the same requests in the same rows: their slots are on the device already
        self._prev_ids = ids
        live = set(ids)
        look = self.cfg.engram_max_ngram_size - 1
        slots, seed_slot, seed_pos, seed_tok = [], [], [], []
        for r in reqs:
            rid = r.py_request_id
            s, new = self.engram_slots.acquire(rid, live)
            if new:
                if getattr(r, "is_context_init_state", False):
                    # an Encoder replay recomputes its historical rows first: the step starts at the replay start
                    p0 = int(
                        getattr(r, "context_current_position", 0)
                    ) - csa2_encoder_replay_tokens(r)
                else:
                    p0 = r.get_num_tokens(0) - 1
                # only the look-back tokens, as the engine's `prepare_disagg_generation_request` reads them
                # (`get_tokens_range`): `get_tokens` marshals the request's whole history, O(prompt) per new request
                lo = max(0, p0 - look)
                toks = r.get_tokens_range(0, lo, p0) if p0 > lo else []
                for k, pos in enumerate(range(lo, p0)):
                    seed_slot.append(s)
                    seed_pos.append(pos % _RING)
                    seed_tok.append(int(toks[k]))
            slots.append(s)
        dev = self.hist.device
        # One upload through reused pinned staging (the engine's `copy_host_to_device`, as its engram history seeds and
        # rows are uploaded), not a fresh pinned tensor per copy: the rows' slots, then a churn step's seeds (flat ring
        # index, token id).
        n, k = max(len(slots), self._prev_reqs), len(seed_slot)
        host = np.full(n + 2 * k, self.engram_slots.dummy, dtype=np.int64)
        host[: len(slots)] = slots
        if k:
            host[n : n + k] = np.asarray(seed_slot, dtype=np.int64) * _RING + np.asarray(
                seed_pos, dtype=np.int64
            )
            host[n + k :] = seed_tok
        buf = torch.empty(n + 2 * k, dtype=torch.int64, device=dev)
        copy_host_to_device(self._staging, ("on_step", dev), buf, torch.from_numpy(host))
        self.req_slot[:n].copy_(buf[:n])
        if k:
            self.hist.view(-1).index_copy_(
                0, buf[n : n + k], self.engram_hash.token_map[buf[n + k :]].to(self.hist.dtype)
            )
        self._prev_reqs = len(slots)

    # ---- forward -------------------------------------------------------------------
    def _select_target(self, step: "_Step", attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        A step of nothing but generation requests (no context rows) is the decode specialization, as
        the engine's backend serves every generation-only step: its indexer reads lengths on the
        device, so it can be captured in a CUDA graph, and an eager step whose requests carry
        different row counts (e.g. a draft-less first step after admission) stays on it too. Steps
        with context rows (chunks, a mixed batch) are the general case and take the exact span of
        each request (one host read per request)."""
        # A step on the engine's CUDA-graph metadata (a generation graph's warm-up runs and its capture) is decode at
        # any width: the general path is never captured, and the graph's warm-up must run the forward its capture
        # records.
        graph = bool(getattr(attn_metadata, "is_cuda_graph", False))
        decode = getattr(attn_metadata, "num_contexts", 1) == 0 and (
            step.n <= _PAGED_MAX_ROWS or graph
        )
        return self._decode if decode else self._prefill

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if lora_params is not None:  # as the built-in model's attention: no LoRA
            raise NotImplementedError("the DeepSeek-V4.1 modeling_v2 targets do not implement LoRA")
        # Encoder recovery (stock's CED replay, as the dep2 target): with block reuse, a context request whose Encoder
        # pages were not kept recomputes its last Encoder rows before its chunk (`EncoderReplay`). Its historical rows
        # rebuild the Encoder working state only: they write no GLOBAL / INDEX rows (the replay's Global claim holds
        # them), and the request's Encoder SWA reads start at its recovery floor. The plan is consumed only after the
        # forward succeeds, as stock's `DeepseekV41ForCausalLM.forward` does.
        requests = kwargs.get("context_requests")
        self._recovery = csa2_encoder_recovery(requests, len(attn_metadata.seq_lens))
        if self._recovery is not None and not self._recovery_logged and any(self._recovery[0]):
            self._recovery_logged = True
            hist = sum(csa2_encoder_replay_tokens(r) for r in requests)
            logger.info(
                f"DeepSeek-V4.1 dep Encoder recovery engaged: {hist} historical rows recomputed in this step"
            )
        try:
            out = self._forward(
                attn_metadata, input_ids, position_ids, inputs_embeds, lora_params, **kwargs
            )
        finally:
            self._recovery = None
        if requests:
            csa2_complete_encoder_replay(requests)
        return out

    def _forward(
        self, attn_metadata, input_ids, position_ids, inputs_embeds, lora_params, **kwargs
    ) -> torch.Tensor:
        assert input_ids is not None and inputs_embeds is None
        csa2_mxfp8_gemm_prewarm(
            input_ids.device
        )  # untuned-row GEMM kernels compile here (warm-up), not in a serving step
        # and every split-K form of the one-kernel HC boundary (DeepGEMM JIT, ~3.5 s each)
        hc_mega_boundary_prewarm(input_ids.device, self.cfg.hidden_size)
        csa2_mxfp8_tune_engine_gemm(
            input_ids.device, self.cfg.vocab_size, self.cfg.hidden_size, self._head_tune_rows
        )
        csa2_mxfp8_agree_tactics()  # after the tuning pass, before the graphs: every rank on the majority's tactics
        n = input_ids.shape[0]
        positions = _step_positions(attn_metadata, position_ids, n)
        seq_lens = attn_metadata.seq_lens
        seq_lens = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        nreq = len(seq_lens)
        dev = input_ids.device
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            self._fwd_stream = torch.cuda.current_stream()
        uniform = len(set(seq_lens)) == 1
        if uniform:  # uniform (generation, CUDA graph): no host data baked in
            L = seq_lens[0]
            graph = capturing or bool(getattr(attn_metadata, "is_cuda_graph", False))
            tok_req, idx_in = self._uniform_rows(n, L, dev, capturing, graph)
            cs = positions - idx_in
            ce = cs + L
            bounds = [(r * L, (r + 1) * L) for r in range(nreq)]
        else:
            total = int(sum(seq_lens))
            idx = np.concatenate(
                [np.repeat(np.arange(nreq), seq_lens), np.full(n - total, nreq - 1)]
            ).astype(np.int64)
            lens = np.concatenate(
                [np.repeat(seq_lens, seq_lens), np.ones(n - total, dtype=np.int64)]
            )
            starts = np.concatenate([[0], np.cumsum(seq_lens)])
            # pinned, asynchronous uploads: a mixed-length generation step reads nothing back from the device
            tok_req = maybe_pin_memory(torch.from_numpy(idx)).to(dev, non_blocking=True)
            first = maybe_pin_memory(torch.from_numpy(starts[:-1].astype(np.int64))).to(
                dev, non_blocking=True
            )
            cs = positions[first][tok_req]
            ce = cs + maybe_pin_memory(torch.from_numpy(lens)).to(dev, non_blocking=True).to(
                positions.dtype
            )
            bounds = [(int(starts[r]), int(starts[r + 1])) for r in range(nreq)]
        step = _Step(
            n=n, nreq=nreq, positions=positions, tok_req=tok_req, cs=cs, ce=ce, bounds=bounds,
            uniform=uniform, capturing=capturing,
        )  # fmt: skip
        return self._select_target(step, attn_metadata).forward(
            self, step, attn_metadata, input_ids, **kwargs
        )

    def _uniform_rows(self, n: int, L: int, dev, capturing: bool, graph: bool = True):
        """(request row `[n]` int64, index within the request `[n]` int32) of a step of `n // L` requests of `L` rows:
        constants of the shape, built once outside capture (the warm-up's eager step of the same shape) instead of
        five launches per step; a shape first seen under capture builds them inline. The shapes of CUDA-graph steps
        (`graph`) are kept for their graphs; other eager shapes (any context chunk of equal-length requests) only the
        most recent `_EAGER_ROWS_CACHED`, so the cache does not grow with the shapes served."""
        key = (n, L)
        rows = self._rows_cache.get(key)
        if rows is None and not graph:
            rows = self._eager_rows_cache.get(key)
            if rows is not None:
                self._eager_rows_cache.move_to_end(key)
        if rows is None:
            r = torch.arange(n, device=dev)
            rows = (r // L, (r % L).to(torch.int32))
            if graph and not capturing:
                self._rows_cache[key] = rows
            elif not capturing:
                self._eager_rows_cache[key] = rows
                while len(self._eager_rows_cache) > _EAGER_ROWS_CACHED:
                    self._eager_rows_cache.popitem(last=False)
        return rows

    def _run_layers(
        self, step: "_Step", ctx: StepCtx, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        """Embedding, the 40 Hyper-Connection layers and the final norm; `ctx` fixes how the
        indexer reads its keys (exact span per request, or device-side lengths)."""
        positions, n = step.positions, step.n
        eh = self.engram_hash
        if (
            n <= ENGRAM_HASH_MAX_ROWS
        ):  # two launches (decode sizes) instead of ~40 single-block int64 ones
            engram_ids = engram_lookback_hash(
                input_ids.contiguous(),
                positions,
                step.tok_req,
                self.req_slot,
                eh.token_map,
                self.hist,
                eh.mults,
                eh.primes,
                eh.pad_id,
            )
        else:
            engram_ids = eh.lookback(
                input_ids, positions, self.req_slot[step.tok_req].long(), self.hist
            )
        engram_round = self._engram_exchange_launch(engram_ids, attn_metadata)
        engram_pre = (
            self._engram_precompute(engram_ids, attn_metadata) if engram_round is None else None
        )
        h = token_embedding(self.embed_tokens, input_ids)
        x = h.unsqueeze(1).repeat(1, self.hc, 1)
        pre_mix = x.new_zeros(n, self.hc, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        spec_metadata = kwargs.get("spec_metadata")
        if hasattr(spec_metadata, "context_capture_lens"):
            spec_metadata.context_capture_lens = None
        plan = csa2_step_replay_plan(
            self,
            attn_metadata,
            bool(kwargs.get("all_token_states_required", True)),
            kwargs.get("context_requests"),
        )
        st = _HcState(x, pre_mix)
        rows = None  # the step rows the decoder half runs, when the replay compacts them
        saved_counts = attn_metadata.all_rank_num_tokens
        try:
            for li, layer in enumerate(self.layers):
                if plan is not None and li == self.decoder_replay_split:
                    st, ctx, rows = self._enter_decoder(layer, st, ctx, plan, attn_metadata)
                    if (
                        rows is not None and rows.numel() == 0
                    ):  # no row of this rank runs the decoder half
                        counts = attn_metadata.all_rank_num_tokens
                        if sum(
                            int(c) for c in counts
                        ):  # (no rank has one: every rank skips it, as stock)
                            for idle in self.layers[li:]:
                                idle.idle(counts)
                        out = h.new_zeros(n, h.shape[-1])
                        break
                st = layer(
                    st,
                    ctx,
                    attn_metadata,
                    engram_ids,
                    spec_metadata,
                    engram_round,
                    engram_pre,
                    rows,
                )
            else:
                out = self._close(st)
        finally:
            attn_metadata.all_rank_num_tokens = saved_counts
        if plan is not None and hasattr(spec_metadata, "context_capture_lens"):
            spec_metadata.context_capture_lens = tuple(
                int(v) for v in plan.replay_seq_lens[: attn_metadata.num_contexts]
            )
        if rows is None or rows.numel() == 0:
            return out
        return out.new_zeros(n, out.shape[-1]).index_copy_(0, rows, out)

    def _close(self, st: _HcState) -> torch.Tensor:
        """The stream's collapse and the final norm."""
        closed = self.layers[-1].fused_close(st, self.norm)
        if closed is not None:
            return closed
        st.wait_coefficients()
        x = st.residual
        if st.deferred:
            x = hc_post_mapping(st.reduced_x_prev(), x, st.post, st.comb)
        # one launch (the engine's collapse, as stock closes its stream at decode sizes under attention DP)
        h = hc_collapse(x, st.pre_mix) if x.is_contiguous() else hc_pre(x, st.pre_mix)
        return flashinfer_rmsnorm(h, self.norm, self.norm_eps)

    def _enter_decoder(self, layer, st: _HcState, ctx: StepCtx, plan, attn_metadata):
        """The bounded-replay boundary, stock's `_enter_bounded_replay`: every rank adopts the decoder row counts (the
        experts' collectives); a rank whose rows or window floors change resolves the stream, publishes the owner
        layer's compressed rows for every row, keeps only the plan's rows (the last `window` rows of each prompt chunk,
        every generation row) and applies the plan's per-request SWA floors. -> (state, ctx, kept rows or None)"""
        if plan.replay_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = plan.replay_all_rank_num_tokens
        if not plan.updates_local_metadata:
            return st, ctx, None
        st.wait_coefficients()
        x = st.residual
        if st.deferred:
            x = hc_post_mapping(st.reduced_x_prev(), x, st.post, st.comb)
        pre_mix, dev = st.pre_mix, x.device
        layer.publish_global(x, pre_mix, ctx)
        floors = maybe_pin_memory(torch.tensor(plan.swa_floors, dtype=torch.int32)).to(
            dev, non_blocking=True
        )
        if (
            not plan.replays_local_tokens
        ):  # every row is kept; the floors still hide window rows of earlier chunks
            dctx = dataclasses.replace(ctx, published=True, floor=floors[ctx.rid])
            return _HcState(x, pre_mix), dctx, None
        lens = [int(v) for v in plan.replay_seq_lens.tolist()]
        rows = plan.rows.to(dev, non_blocking=True).to(torch.int64)
        x, pre_mix = x.index_select(0, rows), pre_mix.index_select(0, rows)
        if rows.numel() == 0:
            return _HcState(x, pre_mix), ctx, rows
        pos, rid, ce = (
            ctx.pos.index_select(0, rows),
            ctx.rid.index_select(0, rows),
            ctx.ce.index_select(0, rows),
        )
        begin = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        first = maybe_pin_memory(torch.from_numpy(np.repeat(begin, lens))).to(
            dev, non_blocking=True
        )
        segs = [
            (int(begin[r]), int(begin[r]) + lens[r], r, span)
            for _, _, r, span in ctx.segs
            if lens[r] > 0
        ]
        nctx = -1 if ctx.nctx < 0 else int(sum(lens[: attn_metadata.num_contexts]))
        dctx = dataclasses.replace(
            ctx, pos=pos, rid=rid, cs=pos.index_select(0, first), ce=ce, segs=segs, topk=None, cands=None, nctx=nctx,
            published=True, floor=floors[rid],
        )  # fmt: skip
        return _HcState(x, pre_mix), dctx, rows

    def _step_ctx(self, step: "_Step", attn_metadata, segs, paged: bool) -> StepCtx:
        swa = engine_swa_tables(attn_metadata, self.cfg.num_hidden_layers)
        return StepCtx(
            pos=step.positions, rid=step.tok_req, cs=step.cs, ce=step.ce, segs=segs,
            kvm=KvPools(attn_metadata.kv_cache_manager, swa), rope=self.rope,
            nreq=step.nreq, paged=paged, uniform=step.uniform,
            rows=None if step.uniform or not paged else tuple(b - a for a, b in step.bounds),
            nctx=-1 if paged else int(getattr(attn_metadata, "num_ctx_tokens", -1)),
        )  # fmt: skip


@dataclass
class _Step:
    """The step's token -> request bookkeeping, built once by `ModelingV2Core.forward`."""

    n: int
    nreq: int
    positions: torch.Tensor  # [n] int32 absolute positions
    tok_req: torch.Tensor  # [n] request row of each token
    cs: torch.Tensor  # [n] first position of the token's request in this step
    ce: torch.Tensor  # [n] one past its last position
    bounds: Sequence  # (first token, one past the last) per request
    uniform: bool  # every request has the same number of tokens
    capturing: bool  # inside CUDA-graph capture


def _context_spans(step: "_Step", attn_metadata) -> list:
    """One past each request's last position. A step of context requests only takes them from the host's cached lengths
    (its positions are the engine's `position_ids`, cached + offset; dep2's form), so no device read drains the stream
    before the layers. A step with generation rows reads its positions back: those come from the CSA2 metadata's live
    positions, which the engine corrects on the device (`on_update_kv_lens`), so the host's lengths can lag them."""
    cached = getattr(
        getattr(attn_metadata, "kv_cache_params", None), "num_cached_tokens_per_seq", None
    )
    nreq = len(step.bounds)
    if (
        cached is not None
        and len(cached) >= nreq
        and getattr(attn_metadata, "num_contexts", -1) == nreq
    ):
        return [int(cached[r]) + (b - a) for r, (a, b) in enumerate(step.bounds)]
    return (torch.stack([step.positions[b - 1] for _, b in step.bounds]).long() + 1).tolist()


class PrefillTarget:
    """The general case: context rows (chunks), a mixed batch, or any step too wide for the paged indexer."""

    def forward(
        self, core: ModelingV2Core, step: _Step, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        assert not step.capturing, "only generation steps are captured"
        spans = _context_spans(step, attn_metadata)
        segs = [(a, b, r, int(spans[r])) for r, (a, b) in enumerate(step.bounds)]
        ctx = core._step_ctx(step, attn_metadata, segs=segs, paged=False)
        if (
            core._recovery is not None
        ):  # Encoder recovery: SWA reads from the recovery floor, history unpublished
            hist_end, floor = (
                maybe_pin_memory(torch.tensor(v, dtype=torch.int32)).to(
                    step.positions.device, non_blocking=True
                )
                for v in core._recovery
            )
            ctx = dataclasses.replace(ctx, floor=floor[ctx.rid], hist_end=hist_end)
        return core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)


class DecodeTarget:
    """Generation-only steps: lengths stay on the device (CUDA-graph safe), paged indexer kernel."""

    def forward(
        self, core: ModelingV2Core, step: _Step, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        ctx = core._step_ctx(step, attn_metadata, segs=[], paged=True)
        return core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)


class ModelingV2DeepseekV41FlashNvfp4Sm103Dep(
    SpecDecOneEngineForCausalLM[ModelingV2Core, PretrainedConfig]
):
    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The built-in model's step-loop hooks (`step_hooks`), the speculative set when DSpark runs."""
        return step_hooks(getattr(self, "spec_config", None))

    def __init__(self, model_config: ModelConfig):
        model_config = text_model_config(model_config)
        # what routing declines (`limits.dep_refused`), for a construction routing did not see: `auto` then falls back
        refused = dep_refused(model_config)
        if refused:
            raise NoTarget(f"the deepseek_v41 dep target does not implement: {'; '.join(refused)}")
        cfg = model_config.pretrained_config
        assert cfg is not None
        # The inherited shell builds the DSpark draft model and its worker from
        # `speculative_config`; the draft stages live in the same checkpoint (`mtp.*`).
        super().__init__(
            ModelingV2Core(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        opt_into_small_m_gemm(self)

    def forward(self, *args, return_context_logits: bool = False, **kwargs) -> torch.Tensor:
        """Tell the core whether anything downstream reads non-final rows (stock's `DeepseekV41ForCausalLM.forward`):
        context logits, or the explicit full-context demand of a DSpark gather. The engine's forward consumes
        `return_context_logits` itself, so the core gets it as `all_token_states_required`; a bounded replay is refused
        for such a step."""
        required = bool(return_context_logits)
        spec_metadata = kwargs.get("spec_metadata")
        if (
            isinstance(spec_metadata, DSparkSpecMetadata)
            and spec_metadata.gather_ids is not None
            and spec_metadata._requires_all_context_logits is not None
        ):
            required = spec_metadata._requires_all_context_logits
        kwargs["all_token_states_required"] = required
        refuse_image_step(kwargs.get("multimodal_params"))
        return super().forward(*args, return_context_logits=return_context_logits, **kwargs)

    def validate_request(self, request) -> None:
        """Engine hook at admission: this target serves text only, so a context request with image input is refused
        (`image_input.refuse_image_request`)."""
        refuse_image_request(request)

    def prepare_adp_inputs(
        self, attn_metadata, *, all_token_states_required: bool, requests=None
    ) -> None:
        """Engine hook, before the attention-DP count exchange: fix which rows of each context chunk run the decoder
        half, so the exchange carries the decoder row counts of every rank (the experts' collectives use them)."""
        attn_metadata.decoder_replay_plan = csa2_replay_plan(
            self.model, attn_metadata, all_token_states_required, requests
        )

    def release_request_state(self, request_id: int) -> None:
        """Engine hook (the executor's request release: completion, cancellation, a warm-up dummy), as stock's: the
        request's Engram look-back row is freed for the next request."""
        self.model.release_request_state(request_id)

    def prepare_request_inputs(
        self, scheduled_requests, attn_metadata, promoted_context_request_ids
    ) -> None:
        """Engine hook, called outside graph capture before the step's inputs are built."""
        # The engine may call this hook on another stream than the forward runs on. The page tables, slot map and
        # history ring are read by the forward (or a graph replay) of the previous step, so every device write
        # here goes on the forward's own stream: stream order keeps it after that step and before this one.
        stream = self.model._fwd_stream
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            self.model.on_step(scheduled_requests, attn_metadata.kv_cache_manager)

    def load_weights(self, weights, *args, **kwargs) -> None:
        self.model.load(weights)
        load_head(self, weights)

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load()
