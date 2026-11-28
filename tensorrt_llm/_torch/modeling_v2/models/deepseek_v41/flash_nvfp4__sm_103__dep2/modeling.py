# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-v4.1-flash-nvfp4 / sm_103 / dep2 (the context role).

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

`dep2` is `tensor_parallel_size: 2`, `moe_expert_parallel_size: 2` and
`enable_attention_dp: true`: requests are split across the two ranks, the whole
attention block is replicated, and only the experts and the engram tables are
sharded.

Composition. Attention (`attention.py`), the routed experts, the shared expert, the Hyper-Connection mixes and the
engram gate are built from entries: the family's own (`models/deepseek_v41/ops/{attention,residual,moe}`: `csa2_*`,
`hc_*` (`hc_mega_boundary` for context-sized boundaries), `engram_gate`, `dsv4_route_nvfp4_quant`, `cute_dsl_moe_nvfp4`,
`moe_a2a`) and the shared catalog's (`gemm/cublas_mm`, `norm/flashinfer_rmsnorm`, `comm/allgather`,
`comm/reducescatter`), with the engine's `CSA2CacheManager` as the allocator of the cache pools. Under attention
DP the experts exchange their rows over the NVLink all-to-all (`moe_a2a`, as stock's context role); the engram rows
and the non-DP layouts use the NCCL collectives.
This file owns the trunk: the HC loop, routing, the engram branch (hash, table lookup), weight routing and the
registration shell. Not an entry: the engram n-gram hash and table lookup, the token embedding, the engram's
all-reduce/all-gather (the engine's), and the inherited `DecoderModel` / `DecoderModelForCausalLM` shells (lm_head,
logits gather, speculative worker).

Two execution paths, chosen per step by `ModelingV2Core._select_target`: `DecodeTarget` (generation-only steps of
at most `_PAGED_MAX_ROWS` rows with equal request lengths: the indexer reads lengths on the device, so the step is
CUDA-graph safe) and `PrefillTarget` (everything else: exact span per request, one host read each).

Bounded decoder replay (context steps; stock's `TRTLLM_V41_DECODER_BOUNDED_REPLAY`, on by default). Layer 20, the last
unpooled KV owner, publishes the compressed KV and index keys of every row of a context chunk; from there to the last
layer only the rows of the chunk inside the final 140 tokens of its prompt run (queries, attention, experts), with the
window of their first rows truncated at the window start. The executor plans the rows (`prepare_adp_inputs`, the same
plan stock uses), exchanges every rank's row count, and the experts' collectives use the decoder counts. Approximate
across stacked SWA layers, as stock is; it requires the private decoder SWA pages the cache manager allocates when
prefix reuse is on. Generation steps and requests that return context logits run every row.

Engram look-back and speculative decoding:

* engram hashes read the n-gram look-back from a device-side ring of recent token ids per request
  (graph-safe, correct under the overlap scheduler and speculative rollback); an engine hook
  seeds it for requests new to the engine;
* the speculative (DSpark) draft path is not built here (the generation targets inherit the engine's).
"""

import contextlib
import dataclasses
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
    csa2_complete_encoder_replay,
    csa2_encoder_recovery,
    csa2_encoder_replay_tokens,
    csa2_replay_plan,
    csa2_replay_policy,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    csa2_mxfp8_gemm_prewarm,
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
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.embedding.token_embedding import (
    token_embedding,
    token_embedding_create,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_history_slots import (
    EngramHistorySlots,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_packed import (
    engram_lookup_packed,
    engram_packed_projection,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
    engram_table_mxfp4,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_moe_nvfp4 import (
    cute_dsl_moe_nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route,
    dsv4_route_nvfp4_quant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import (
    INVALID_EXPERT,
    MoeA2AWorkspace,
    moe_a2a_combine,
    moe_a2a_combine_payload,
    moe_a2a_dispatch,
    moe_a2a_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (
    MIN_ROWS as FFN_QUANT_MIN_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (
    nvfp4_fp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.engram_gate import (
    engram_gate,
    engram_gate_kv,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mega_boundary import (
    hc_mega_boundary,
    hc_mega_boundary_prewarm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (
    hc_pre_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.step_hooks import step_hooks
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from ..common import Fp8Rows, KvPools, text_model_config
from ..engram_hash import ENGRAM_RING, EngramHash, engram_buckets
from ..image_input import refuse_image_request, refuse_image_step
from ..limits import dep2_refused
from ..loaders import dequant_block32, load_head
from ..shared_expert import SharedExpertFp8
from ..swa_tables import engine_swa_tables
from . import weights as _weights
from .attention import StepCtx, _Attention
from .prefill_graphs import DecoderPassGraph, EncoderPassGraph, PassGraphPool, forks_in_capture

# The GPU architecture this target IS; routing will not send another one here.
_SM = (10, 3)

_BLOCK = 32  # FP8 block edge of the checkpoint's dense weights
_HC_POST_MULT = 2.0  # post-mix scale of the Hyper-Connections (`post = 2 * sigmoid(.)`)
# Hyper-Connection boundaries of at least `hc_mega_boundary.MIN_ROWS` rows (a sub-layer's post update, the next one's
# mixes, collapse and norm) run as DeepGEMM's one-kernel `mega_mhc`, as stock's context role does; the FFN's post update
# is carried into the next layer's attention boundary.
# The two engram tables are ~92 GiB per rank at EP2, which does not fit beside the experts in 276 GiB.
# They stay in pinned host memory that the lookup kernel reads through UVA, as the engine's
# `ShardedFp8MultiHeadEmbedding` does; processes of one node that pass the same `share_key` (the dep2 servers
# sharing a node hold the same rows) map one copy.
_RING = ENGRAM_RING  # positions of history per slot (the family's ring)
_MAX_REQS = 4096
# Breakable two-pass prefill graphs on the engine's breakable CUDA graphs (`prefill_graphs.py`, stock's
# `deepseekv41_decoder_graph.py`): attention and engram as eager breaks over the live rows. On, as in stock, when the
# engine runs breakable prefill graphs (`prefill_cuda_graph_backend: breakable`); every other context step runs
# eagerly, as stock's does.
_PAGED_MAX_ROWS = (
    4096  # query rows of an eager generation step on the paged path (a CUDA-graph step: any)
)


class _Engram(nn.Module):
    """Adds a gated n-gram lookup into the residual stream (layers 1 and 14)."""

    # The table is built in `load`, after the loader's post-init check: the module tells it that the pinned host
    # table is filled in place, which only the regular load_format=AUTO checkpoint load does.
    _requires_regular_checkpoint_loading = True

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
        # The table stays FP8 as stored, in pinned host memory (see `load`).
        self.table = None
        self.wkv = None
        self.q_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.allreduce = engine_allreduce_create(mapping)
        # Plain NCCL for the breakable pass graphs: their all-reduce runs inside a CUDA graph this model captures
        # itself, where the engine's NCCL symmetric windows must be preallocated by the engine's own capture, which
        # this one is not.
        self.allreduce_plain = engine_allreduce_create(mapping, AllReduceStrategy.NCCL)
        self.qk_weight = None

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
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
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
        w, sc = weights[prefix + "wkv.weight"].to(device), weights[prefix + "wkv.scale"].to(device)
        self._wkv_block_scale = (
            sc  # the bf16 projection of the unpacked paths is built from it on first use
        )
        # stock's packed path reads the checkpoint's E4M3 WKV with its
        # 32-row scale tiles expanded per row (bytes unchanged)
        sc = sc.view(torch.uint8) if sc.dtype != torch.uint8 else sc
        self.wkv_q = w.view(torch.float8_e4m3fn).contiguous()
        self.wkv_s = sc.repeat_interleave(32, 0)[: w.shape[0]].contiguous()
        self.wkv_sw = mxfp8_swizzled_scales(self.wkv_s)
        self.wkv_alpha = torch.ones(1, dtype=torch.float32, device=device)
        self.q_weight.data.copy_(weights[prefix + "q_weight"])
        self.k_weight.data.copy_(weights[prefix + "k_weight"])

    def _lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [m, cols] bucket-local -> [m, cols, head_dim] bf16, partial:
        what this rank's table shard does not hold is 0.

        The table's UVA kernel reads the pinned host rows and dequantizes them in one pass. Under head
        sharding it returns only this rank's heads, which are placed at their columns.
        """
        out = self.table(ids)
        if out.shape[1] == self.cols:
            return out
        full = out.new_zeros(ids.shape[0], self.cols, self.head_dim)
        full[:, self.table.head_start : self.table.head_start + out.shape[1]] = out
        return full

    def forward(
        self, x: torch.Tensor, ids: torch.Tensor, all_rank_num_tokens, plain: bool = False
    ) -> torch.Tensor:
        """x [n, hc, d]; ids [n, cols] -> x + gate * value."""
        sizes = all_rank_num_tokens
        allreduce = self.allreduce_plain if plain else self.allreduce
        if self.mapping.tp_size > 1 and self.mapping.enable_attention_dp and self.table.shard_heads:
            # Stock's packed path: look up this rank's heads of every rank's tokens as packed MXFP8 rows, gather the
            # shards' rows, and project this rank's tokens straight from them (no bf16 embedding, no all-reduce)
            gathered = engine_allgather(ids, self.mapping, 0, sizes)
            weight, scale = self.table.storage()
            rows = engram_lookup_packed(
                weight.view(torch.uint8), scale.view(torch.uint8), gathered, self.table.offsets,
                head_start=self.table.head_start, local_heads=self.table.num_local_heads,
                block_size=self.table.block_size, num_ctas=_ENGRAM_CTAS, mxfp4=self.table.mxfp4,
            )  # fmt: skip
            ws = self.mapping.tp_size
            shards = engine_allgather(rows, self.mapping, 0).view(ws, rows.shape[0], rows.shape[1])
            first = sum(sizes[: self.mapping.tp_rank])
            kv = engram_packed_projection(
                shards[:, first : first + ids.shape[0]],
                self.wkv_q,
                self.wkv_s,
                self.wkv_sw,
                self.wkv_alpha,
            )
            return self._gate(x, kv)
        if self.mapping.tp_size > 1 and self.mapping.enable_attention_dp:
            # Ranks hold different tokens: gather ids, look up the local shard,
            # sum the shards, keep this rank's rows.
            gathered = engine_allgather(ids, self.mapping, 0, sizes)
            emb = engine_allreduce(allreduce, self._lookup(gathered))
            first = sum(sizes[: self.mapping.tp_rank])
            emb = emb[first : first + ids.shape[0]]
        elif self.mapping.tp_size > 1:
            emb = engine_allreduce(allreduce, self._lookup(ids))
        else:
            emb = self._lookup(ids)
        if self.wkv is None:  # (first use is an eager step; the packed path above never needs it)
            self.wkv = dequant_block32(self.wkv_q, self._wkv_block_scale)
        return self._gate(x, cublas_mm(emb.flatten(-2), self.wkv.t()))

    def _gate(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """x + gate * value from the fused k | v projection (stock's gate kernel when the stream is contiguous)."""
        if x.is_contiguous() and kv.is_contiguous():
            if (
                self.qk_weight is None
            ):  # the cached q_weight * k_weight (first use is an eager step)
                self.qk_weight = (self.q_weight.float() * self.k_weight.float()).contiguous()
            return engram_gate_kv(x, kv, self.q_weight, self.k_weight, self.qk_weight, self.eps)
        key, value = torch.split(kv, [self.hc * self.dim, self.dim], dim=-1)
        return engram_gate(
            x,
            key.unflatten(-1, (self.hc, self.dim)),
            value,
            self.q_weight,
            self.k_weight,
            self.eps,
            None,
        )


_ENGRAM_CTAS = 132  # CTAs of the packed table gather

_SF_VEC = 16
# Expert tile rows of the CuTe-DSL grouped GEMMs (stock autotunes 128 / 256 per token bucket): 256-row 2-CTA tiles from
# this many rows of the call (the rows this rank's experts receive), where they are the faster of the two tuned tiles
# at this model's width (H 5120 / I 2304, 192 of 384 experts, top-6).
_MOE_TILE256_MIN_ROWS = 12288
# Stock's `_EAGER_SHARED_EXPERT_OVERLAP_MIN_TOKENS`: the shared expert overlaps the all-to-all routed experts from this
# many rows of the rank's own step.
_SHARED_OVERLAP_MIN_ROWS = 12288
# The routed experts' kernels are stock's context role's CuTe-DSL grouped GEMMs (`moe_config.backend: CUTEDSL`), one
# call for any row count. Under attention DP with the experts split only across ranks they exchange their rows over
# stock's NVLink one-sided all-to-all (`moe_a2a`: dispatch the quantized rows to the ranks holding their experts, write
# the partial sums into the workspace's combine region, combine them back with the native combine); other layouts
# all-gather the quantized rows and reduce-scatter the partial sums.


class _Moe(nn.Module):
    """Routed experts: sqrtsoftplus routing (bias picks, raw scores weigh) and the NVFP4 CuTe-DSL grouped GEMMs
    (`cute_dsl_moe_nvfp4`).

    Experts are split `moe_ep_size` ways (a contiguous window each) and each expert's intermediate
    width `moe_tp_size` ways, so every rank holds a partial sum. With attention DP the ranks hold
    different tokens: gather them, run, reduce-scatter. Without it every rank holds the same
    tokens: run, then an all-reduce (a reduce-scatter and an all-gather over padded rows).
    """

    def __init__(self, cfg, mapping, aux_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.cfg, self.topk = cfg, cfg.num_experts_per_tok
        # Stock's eager shared-expert overlap (`DeepseekV4MoE`, `_fork_after_routing`): on the all-to-all path, steps of
        # at least `_SHARED_OVERLAP_MIN_ROWS` rows run the shared expert on this side stream from the end of routing,
        # beside the dispatch, the experts and the combine (which leave SMs idle), and add it after the combine.
        self.aux_stream = aux_stream
        self._routed_event, self._shared_event = torch.cuda.Event(), torch.cuda.Event()
        self._forked = None
        self.adp = bool(mapping.enable_attention_dp)
        self.a2a: Optional[MoeA2AWorkspace] = (
            None  # the target's shared all-to-all workspace (`ModelingV2Core`)
        )
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
        self.o1_gate = buf(e, dtype=f32)
        self.g1 = buf(1, dtype=f32)
        # The routing weights' precision is the experts' (stock's `routing_scales_dtype`): fp32 for the CuTe-DSL grouped
        # GEMMs (stock's `try_fused_route` routes in fp32).
        self.weights_dtype = torch.float32
        # FC2's dequant scalars and the FC1 output's NVFP4 global scale (one for every expert)
        self.fc2_alpha = buf(e, dtype=f32)
        self.fc2_global = buf(1, dtype=f32)

    def _route_quant(self, x: torch.Tensor, pre=None):
        """Routing and the NVFP4 quantization of `x`'s rows, one launch: (ids, weights, x_fp4, x_sf). `pre`: `x`
        already quantized (x_fp4, x_sf), as the FFN boundary's `nvfp4_fp8_quantize` hands it; routing alone then."""
        logits = cublas_mm(x, self.gate_weight.t(), None, torch.float32)
        scale, wdt = self.cfg.routed_scaling_factor, self.weights_dtype
        if pre is not None:
            return (*dsv4_route(logits, self.gate_bias, self.topk, scale, x.shape[1], wdt), *pre)
        return dsv4_route_nvfp4_quant(logits, self.gate_bias, x, self.g1, self.topk, scale, wdt)

    def _run_experts(self, ids, w, xq, xsf, out=None) -> torch.Tensor:
        """The routed experts over routed, quantized rows (their own rank's partial sum). `out`: the buffer to write
        into, e.g. the all-to-all's combine region."""
        return cute_dsl_moe_nvfp4(
            xq, xsf, ids, w, self.fc1_w, self.fc1_s, self.fc2_w, self.fc2_s, self.o1_gate, self.fc2_alpha,
            self.fc2_global, self.cfg.swiglu_limit, self.cfg.n_routed_experts, self.offset, out,
            256 if ids.shape[0] >= _MOE_TILE256_MIN_ROWS else 128,
        )  # fmt: skip

    def _experts(self, x: torch.Tensor) -> torch.Tensor:
        return self._run_experts(*self._route_quant(x))

    def _fork_shared(self, x: torch.Tensor, shared):
        """Start `shared(x)` on the side stream after what the current stream has issued (the routing)."""
        main = torch.cuda.current_stream()
        self._routed_event.record(main)
        with torch.cuda.stream(self.aux_stream):
            self._routed_event.wait()
            # No `x.record_stream(aux)` (stock does not record it either): `x` stays referenced until the main stream
            # has waited for the side stream (`_join_shared`, before the MoE returns), so its block is reused only by
            # main-stream work ordered after the shared expert. A recorded stream would hold the block back until the
            # allocator sees the side stream's event complete; with the host ahead of the GPU that is about half a
            # step of 330 MB inputs per layer, which the allocator then maps anew (serving-time growth the warm-up
            # memory profile does not see).
            out = shared(x)
            self._shared_event.record()
        # No `out.record_stream(main)` either: the side stream next runs only in the next fork, after waiting for an
        # event the main stream records after its add of `out`, so `out`'s block is reused only once the add is done.
        # A recorded stream defers each layer's free until the allocator sees the main stream pass the add; with the
        # host ahead of the GPU the side stream's pool would hold many layers' outputs at once, memory the warm-up
        # profile does not see.
        self._forked = out
        return out

    def _join_shared(self, out: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """`out`, after the side stream produced it when it is the forked shared output."""
        if out is not None and out is self._forked:
            self._shared_event.wait(torch.cuda.current_stream())
            self._forked = None
        return out

    def _forward_a2a(
        self,
        x: torch.Tensor,
        counts,
        addend: Optional[torch.Tensor],
        shared=None,
        pre=None,
        unrouted: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention DP over the all-to-all: route and quantize this rank's rows, dispatch them to the ranks that hold
        their experts, run this rank's experts over what it received straight into the workspace's combine region,
        combine the partial sums back. A rank without rows dispatches one row routed nowhere (the kernels take no empty
        side), as stock does."""
        a, n, h = self.a2a, x.shape[0], self.cfg.hidden_size
        rows = max(max(1, int(c)) for c in counts)
        if n:
            ids, w, xq, xsf = self._route_quant(x, pre)
            if (
                unrouted is not None
            ):  # a pass graph's padding rows go to no expert (stock's `unrouted_rows`)
                ids = torch.where(unrouted, INVALID_EXPERT, ids)
            if (
                shared is not None
            ):  # (a caller passes `shared` only when the step forks: see `forward`)
                addend = self._fork_shared(x, shared)
        else:
            ids = torch.full((1, self.topk), INVALID_EXPERT, dtype=torch.int32, device=x.device)
            w = torch.zeros(1, self.topk, dtype=self.weights_dtype, device=x.device)
            xq = torch.zeros(1, h // 2, dtype=torch.uint8, device=x.device)
            xsf = torch.zeros(1, h // _SF_VEC, dtype=torch.uint8, device=x.device)
        payloads = [t.contiguous() for t in (xq.view(torch.uint8), xsf.view(torch.uint8), ids, w)]
        (rq, rsf, rids, rw), offset = moe_a2a_dispatch(
            payloads[2],
            payloads,
            2,
            a.workspace,
            a.metainfo,
            rows,
            a.ep_rank,
            a.ep_size,
            self.cfg.n_routed_experts,
            a.sentinel_max_rows,
        )
        region = moe_a2a_combine_payload(a.workspace, a.metainfo, a.ep_rank, a.ep_size, rows, h)
        partial = self._run_experts(rids, rw, rq, rsf, region)
        out = moe_a2a_combine(
            partial,
            max(n, 1),
            a.workspace,
            a.metainfo,
            rows,
            a.ep_rank,
            a.ep_size,
            self.topk,
            offset,
            a.sentinel_max_rows,
            True,
        )
        out = out if n else out[:0]
        return out if addend is None else out + self._join_shared(addend)

    def forward(
        self,
        x: torch.Tensor,
        all_rank_num_tokens,
        addend: Optional[torch.Tensor] = None,
        shared=None,
        pre=None,
        unrouted: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The routed experts' output for `x`'s rows, plus the shared expert's output: `addend` when given, else
        `shared(x)` (a callable), which the all-to-all path overlaps with the routed experts on large steps. `pre`:
        `x` already quantized to NVFP4 (x_fp4, x_sf), as the FFN boundary hands it."""
        if self.adp and self.a2a is not None:
            top = max(int(c) for c in all_rank_num_tokens)
            if top == 0 and shared is not None:
                addend, shared = shared(x), None
            if top == 0:
                out = x.new_empty(0, x.shape[-1])
                return out if addend is None else out + addend
            if (
                top <= self.a2a.max_rows
            ):  # every rank sees the same counts, so every rank takes the same path
                fork = (
                    shared is not None
                    and self.aux_stream is not None
                    and x.shape[0] >= _SHARED_OVERLAP_MIN_ROWS
                    and (not torch.cuda.is_current_stream_capturing() or forks_in_capture())
                )
                if shared is not None and not fork:
                    addend, shared = shared(x), None
                return self._forward_a2a(x, all_rank_num_tokens, addend, shared, pre, unrouted)
        if shared is not None:
            addend = shared(x)
        routed = self._forward_collectives(x, all_rank_num_tokens, pre)
        return routed if addend is None else routed + addend

    def _forward_collectives(self, x: torch.Tensor, all_rank_num_tokens, pre=None) -> torch.Tensor:
        n, ws = x.shape[0], len(self.group)
        if self.adp:
            # Each rank routes and quantizes its own rows once; the quantized rows travel (a quarter of the bytes of
            # bf16 rows) and every rank runs its experts over all of them.
            rows = max(int(c) for c in all_rank_num_tokens)
            xp = x if rows == n else F.pad(x, [0, 0, 0, rows - n])
            rq = self._route_quant(xp, pre if rows == n else None)
            gathered = [cat_allgather(t.contiguous(), None, self.group) for t in rq]
            routed = cat_reducescatter(self._run_experts(*gathered), None, self.group)
            return routed if rows == n else torch.split(routed, [n, rows - n], dim=0)[0]
        rows = -(-n // ws) * ws
        xp = x if rows == n else F.pad(x, [0, 0, 0, rows - n])
        part = self._experts(xp.contiguous())
        out = cat_allgather(cat_reducescatter(part, None, self.group), None, self.group)
        return out if rows == n else torch.split(out, [n, rows - n], dim=0)[0]


class _Layer(nn.Module):
    def __init__(self, model_config: ModelConfig, layer_idx: int, aux_stream_dict, engram_slot):
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
        self.moe = _Moe(cfg, self.mapping, aux_stream_dict[AuxStreamType.MoeShared])
        self.shared = SharedExpertFp8(self.dim, cfg.moe_intermediate_size, cfg.swiglu_limit)
        self.engram = _Engram(cfg, self.mapping, engram_slot) if engram_slot is not None else None

    def _ffn(
        self, x: torch.Tensor, all_rank_num_tokens, unrouted: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # The shared expert is replicated and the routed partial sums are already reduced, so the
        # two meet here exactly once.
        if x.shape[0] >= FFN_QUANT_MIN_ROWS and nvfp4_fp8_quantize.ready(x.shape[1]):
            # Stock's FFN boundary: one pass quantizes the input for both consumers (NVFP4 for the routed experts,
            # whose launch then only routes; FP8 1x128 for the shared expert's first GEMM), from FFN_QUANT_MIN_ROWS rows
            fp4, sf, a, a_sf = nvfp4_fp8_quantize(x, self.moe.g1)

            def shared(
                _x,
            ):  # may run on the MoE's side stream: `a` / `a_sf` stay referenced by this closure until
                return self.shared.forward_fp8(
                    a, a_sf
                )  # the MoE joined that stream (see `_Moe._fork_shared`)

            return self.moe(x, all_rank_num_tokens, shared=shared, pre=(fp4, sf), unrouted=unrouted)
        return self.moe(x, all_rank_num_tokens, shared=self.shared, unrouted=unrouted)

    def _boundary(self, which: str, h, residual, post, comb, pre_mix):
        """The previous sub-layer's post update (`h`, `post`, `comb` into `residual`) and this sub-layer's mixes,
        collapse and norm: one `mega_mhc` launch for context-sized steps, else `hc_post_mapping` + `hc_pre_mapping`.
        -> (residual, pre, post, comb, normalized input)."""
        fn, scale, base = (getattr(self, f"hc_{which}_{k}") for k in ("fn", "scale", "base"))
        norm = self.attn_norm if which == "attn" else self.ffn_norm
        args = (self.hc, self.sinkhorn, self.norm_eps, self.norm_eps, self.hc_eps, _HC_POST_MULT)
        if hc_mega_boundary.available(residual.shape[0], self.dim):
            return hc_mega_boundary(h, residual, post, comb, pre_mix, fn, scale, base, norm, *args)
        x = hc_post_mapping(h, residual, post, comb)
        return (x, *hc_pre_mapping(x, pre_mix, fn, scale, base, norm, *args))

    def publish_global(self, x, pre_mix, ctx) -> None:
        """Bounded-replay boundary (the last unpooled KV owner): publish the compressed KV and index-key rows
        of every row of the step, before the rows are narrowed to the final window."""
        h = hc_pre_mapping(
            x, pre_mix, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.attn_norm,
            self.hc, self.sinkhorn, self.norm_eps, self.norm_eps, self.hc_eps, _HC_POST_MULT,
        )[3]  # fmt: skip
        self.attn.publish_global(h, ctx)

    def idle(self, counts) -> None:
        """A rank with no rows in the decoder half still joins the experts' collectives of its peers."""
        self.moe(self.attn_norm.new_empty(0, self.dim), counts)

    def _engram_live(self, x: torch.Tensor, live) -> torch.Tensor:
        """The engram inside a pass graph: an eager break over this rank's live rows of `x`, in place (with the live
        per-rank counts its collectives exchange); nothing during the graphs' warm runs and captures."""

        def apply(states: torch.Tensor) -> None:
            if live.counts is None:
                return
            rows = live.rows
            states[:rows].copy_(
                self.engram(
                    states[:rows], live.engram_ids[:rows, self.engram.hash_index], live.counts
                )
            )

        eager_on_graph(apply)(x)
        return x

    def _attn_live(self, h: torch.Tensor, live) -> torch.Tensor:
        """Attention inside a pass graph: an eager break over this rank's live rows with the step's context (stock's
        `_forward_live_rows`); padding rows come out zero."""
        out = torch.empty_like(h)

        def attend(states: torch.Tensor, dest: torch.Tensor) -> None:
            rows = live.rows
            if rows:  # the output projection writes the live rows in place (stock's `out=destination[:rows]`)
                got = self.attn(states[:rows], live.ctx, out=dest[:rows])
                if got.data_ptr() != dest.data_ptr():
                    dest[:rows].copy_(got)
            if rows < dest.shape[0]:
                dest[rows:].zero_()

        eager_on_graph(attend)(h, out)
        return out

    def forward(
        self,
        x,
        pre_mix,
        ctx,
        attn_metadata,
        engram_ids,
        counts=None,
        enc_counts=None,
        pending=None,
        defer: bool = False,
        live=None,
        unrouted: Optional[torch.Tensor] = None,
    ):
        """x [n, hc, d] bf16; pre_mix [n, hc] fp32 -> (x, next pre_mix). `counts`: rows of every attention-DP
        rank at this layer (the decoder half of a bounded replay has fewer than the encoder half).

        `pending` is the previous layer's FFN post update not yet applied to `x` (`(h, post, comb)`, see `defer`):
        this layer's attention boundary applies it. With `defer` this layer's own FFN post update is returned
        instead of applied, as `(x, next pre_mix, pending)`; `resolve_hc` applies one.

        `live` (a pass graph's `PassLive`): attention and the engram run in eager breaks over the live rows, and
        the routed experts skip the `unrouted` (padding) rows."""
        counts = attn_metadata.all_rank_num_tokens if counts is None else counts
        if x.shape[0] == 0 and self.engram is None:
            self.idle(counts)
            return (x, pre_mix, None) if defer else (x, pre_mix)
        if self.engram is not None:
            x = resolve_hc(x, pending)  # the engram adds into the materialized residual
            pending = None
            if live is not None:
                x = self._engram_live(x, live)
            else:
                enc = attn_metadata.all_rank_num_tokens if enc_counts is None else enc_counts
                x = self.engram(
                    x, engram_ids[:, self.engram.hash_index], enc, plain=enc_counts is not None
                )
        if pending is not None:
            x, a_pre, a_post, a_comb, h = self._boundary(
                "attn", *pending[:1], x, *pending[1:], pre_mix
            )
        else:
            a_pre, a_post, a_comb, h = hc_pre_mapping(
                x, pre_mix, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.attn_norm,
                self.hc, self.sinkhorn, self.norm_eps, self.norm_eps, self.hc_eps, _HC_POST_MULT,
            )  # fmt: skip
        h = self.attn(h, ctx) if live is None else self._attn_live(h, live)
        x, f_pre, f_post, f_comb, h = self._boundary("ffn", h, x, a_post, a_comb, a_pre)
        h = self._ffn(h, counts, unrouted)
        if defer:
            return x, f_pre, (h, f_post, f_comb)
        return hc_post_mapping(h, x, f_post, f_comb), f_pre


def resolve_hc(x: torch.Tensor, pending) -> torch.Tensor:
    """Apply a deferred FFN post update (`_Layer.forward(defer=True)`) to the residual `x`."""
    return x if pending is None else hc_post_mapping(pending[0], x, pending[1], pending[2])


def _h2d(a, dev) -> torch.Tensor:
    """A host array (or CPU tensor) on `dev` through pinned memory, without waiting for the stream."""
    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
    if t.is_cuda:
        return t
    return maybe_pin_memory(t).to(dev, non_blocking=True)


def _replay_policy(cfg, model_config: ModelConfig):
    """`(first decoder layer, window)` of the bounded decoder replay, or `(None, 0)`: stock's policy
    (`csa2_replay_policy`, the same switches and window the engine's cache manager follows). This target writes no
    speculative captures, and exchanges the decoder row counts of the experts' collectives through attention DP."""
    if not model_config.mapping.enable_attention_dp:
        logger.warning(
            "DeepSeek-V4.1 decoder bounded replay is off for this target: the row counts of the experts' "
            "collectives are exchanged through attention DP; full prefill."
        )
        return None, 0
    return csa2_replay_policy(cfg, model_config, captures=False)


class ModelingV2Core(DecoderModel):
    # Whether the engine may capture a context step as a whole-body prefill CUDA graph (`prefill_cuda_graph_backend`):
    # never. Stock declines it under attention DP with a decoder replay plan (`DeepseekV41Model`); this target's
    # context path (`PrefillTarget`) runs eagerly in every layout and asserts it is not captured.
    # Its own breakable two-pass prefill graphs (`_forward_bcg`) it captures itself.
    supports_prefill_cuda_graph = False

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
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
        self.layers = nn.ModuleList(
            [
                _Layer(model_config, i, self.aux_stream_dict, slots.get(i))
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self._a2a_rows = int(model_config.max_num_tokens)
        self.a2a: Optional[MoeA2AWorkspace] = None  # built in `post_load` (`_build_a2a`)
        # Bounded decoder replay (see `_replay_policy`). The executor reads these two names to decide whether the
        # cache manager's PRIVATE decoder pages and the replay recovery apply to this model.
        self.decoder_replay_split, self.decoder_replay_window = _replay_policy(cfg, model_config)
        self.ced_kv_precompute = self.decoder_replay_split is not None
        self._replay_reuse_warned = False
        self._replay_logged = False
        # The breakable pass graphs (`_forward_bcg`): when the engine runs breakable prefill graphs
        # (`prefill_cuda_graph_backend: breakable`, as stock's), on the context role (two attention-DP ranks) with a
        # bounded replay; every other context step runs eagerly.
        self._bcg = bool(model_config.extra_attrs.get("breakable_prefill_cuda_graph", False))
        self._prefill_graph = (
            self._bcg and self.decoder_replay_split is not None and self.mapping.world_size == 2
        )
        # this step's Encoder recovery `(hist_end, floor)` per request row (`csa2_encoder_recovery`)
        self._recovery = None
        self._recovery_logged = False
        self._enc_graph: Optional[EncoderPassGraph] = (
            None  # breakable pass graphs (`_forward_bcg`), built on first use
        )
        self._dec_graph: Optional[DecoderPassGraph] = None
        self.engram_hash: Optional[EngramHash] = None
        # Engram look-back state (see `EngramHash.lookback`): a ring of compressed ids per request row (`hist`), the
        # step's request -> row map (`req_slot`) and which request owns which row, as stock owns them: a request keeps
        # its row until the engine releases it (`release_request_state`); the ring holds every request the rank can
        # (`EngramHistorySlots`, sized at the first step). Its last row is the dummy that padded (CUDA-graph) and
        # unknown requests write into.
        self.engram_slots: Optional[EngramHistorySlots] = None
        self._prev_reqs = 0
        self._fwd_stream: Optional[torch.cuda.Stream] = (
            None  # the stream the forward runs on (set by the first forward)
        )
        # this step's SWA page tables: the engine's (`engine_swa_tables`), bound by every forward
        self._swa_tables: Optional[torch.Tensor] = None
        self._prefill = PrefillTarget()
        self._decode = DecodeTarget()

    def _build_a2a(self) -> Optional[MoeA2AWorkspace]:
        """The experts' all-to-all workspace, one for every layer (they run one after another): attention DP with the
        experts split only across ranks (the CuTe-DSL experts take the -1 ids of dead rows). Sized for the most
        rows a rank can carry into a layer: the step's token budget (a pass graph's bucket stays within it).
        Collective: every rank builds it once, in `post_load` (after the meta-device construction), in the same
        order."""
        m = self.mapping
        if not (m.enable_attention_dp and m.moe_tp_size == 1 and m.moe_ep_size == m.world_size > 1):
            return None
        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

        if not MnnvlMemory.supports_mnnvl():
            # As the engine's strategy selection (`NVLinkOneSided.is_platform_supported`): without MNNVL the experts
            # run over the all-gather / reduce-scatter (`_Moe.forward` with no workspace).
            logger.info(
                "DeepSeek-V4.1 dep2: NVLink one-sided all-to-all not available on this platform (no MNNVL); "
                "the experts run over all-gather / reduce-scatter"
            )
            return None
        h, k = self.cfg.hidden_size, self.cfg.num_experts_per_tok
        rows = self._a2a_rows
        a2a = moe_a2a_workspace(
            m,
            rows,
            k,
            [h // 2, h // _SF_VEC, 4 * k, 4 * k],
            2 * h,
            h,  # fp32 routing weights
            self.cfg.n_routed_experts,
        )
        for layer in self.layers:
            layer.moe.a2a = a2a
        logger.info(
            f"DeepSeek-V4.1 dep2 experts on the NVLink all-to-all: {rows} rows per rank, "
            f"{a2a.workspace.shape[1] / 2**20:.0f} MiB per rank, sentinel transport up to "
            f"{a2a.sentinel_max_rows} rows per rank"
        )
        return a2a

    # ---- weights: see weights.py ---------------------------------------------------
    def load(self, weights: dict) -> None:
        _weights.load(self, weights)

    def post_load(self) -> None:
        for layer in self.layers:
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
        if self.a2a is None:
            self.a2a = self._build_a2a()
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

    @property
    def hist(self) -> torch.Tensor:
        return self.engram_slots.hist

    @property
    def req_slot(self) -> torch.Tensor:
        return self.engram_slots.req_slot

    def release_request_state(self, request_id: int) -> None:
        """Engine hook (completion, cancellation): the request's look-back row is free for the next request."""
        if self.engram_slots is not None:
            self.engram_slots.release(request_id)

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
        live = {r.py_request_id for r in reqs}
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
        if seed_slot:

            def t(v):
                return maybe_pin_memory(torch.tensor(v, dtype=torch.int64)).to(
                    dev, non_blocking=True
                )

            self.hist[t(seed_slot), t(seed_pos)] = self.engram_hash.token_map[t(seed_tok)].to(
                self.hist.dtype
            )
        n = max(len(slots), self._prev_reqs)
        pad = np.full(n, self.engram_slots.dummy, dtype=np.int32)
        pad[: len(slots)] = slots
        self.req_slot[:n].copy_(maybe_pin_memory(torch.from_numpy(pad)), non_blocking=True)
        self._prev_reqs = len(slots)

    # ---- forward -------------------------------------------------------------------
    def _select_target(self, step: "_Step", attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        A step of nothing but short uniform generation requests (no context rows) is the decode
        specialization: its indexer reads lengths on the device, so it can be captured in a CUDA
        graph. Every other step -- context rows, chunks, a mixed batch -- is the general case and
        takes the exact span of each request (one host read per request)."""
        # A step on the engine's CUDA-graph metadata (a generation graph's warm-up runs and its capture) is decode at
        # any width: the general path is never captured, and the graph's warm-up must run the forward its capture
        # records.
        graph = bool(getattr(attn_metadata, "is_cuda_graph", False))
        decode = (
            step.uniform
            and getattr(attn_metadata, "num_contexts", 1) == 0
            and (step.n <= _PAGED_MAX_ROWS or graph)
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
        # Encoder recovery (stock's CED replay): a resumed request whose Encoder pages were not kept recomputes its last
        # Encoder rows before its chunk (`EncoderReplay`). Its historical rows rebuild the Encoder working state only:
        # they write no GLOBAL / compressor rows (the replay's Global claim holds them), and every request's Encoder SWA
        # reads start at its recovery floor. The plan is consumed only after the forward succeeds, as stock does.
        requests = kwargs.pop("context_requests", None)
        nreq = len(attn_metadata.seq_lens)
        self._recovery = csa2_encoder_recovery(requests, nreq)
        if self._recovery is not None and not self._recovery_logged and any(self._recovery[0]):
            self._recovery_logged = True
            hist = sum(csa2_encoder_replay_tokens(r) for r in requests)
            logger.info(
                f"DeepSeek-V4.1 dep2 Encoder recovery engaged: {hist} historical rows recomputed in this step"
            )
        try:
            out = self._forward(attn_metadata, input_ids, position_ids, inputs_embeds, **kwargs)
        finally:
            self._recovery = None
        if requests:
            csa2_complete_encoder_replay(requests)
        return out

    def _forward(
        self, attn_metadata, input_ids, position_ids, inputs_embeds, **kwargs
    ) -> torch.Tensor:
        assert input_ids is not None and inputs_embeds is None
        csa2_mxfp8_gemm_prewarm(
            input_ids.device
        )  # untuned-row GEMM kernels compile here (warm-up), not in a serving step
        hc_mega_boundary_prewarm(
            input_ids.device, self.cfg.hidden_size
        )  # every split-K form of the one-kernel HC
        # boundary (DeepGEMM JIT, ~3.5 s each)
        n = input_ids.shape[0]
        seq_lens = attn_metadata.seq_lens
        seq_lens = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        nreq = len(seq_lens)
        self._swa_tables = engine_swa_tables(attn_metadata, self.cfg.num_hidden_layers)
        positions = position_ids.reshape(-1).to(torch.int32).contiguous()
        dev = input_ids.device
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            self._fwd_stream = torch.cuda.current_stream()
        uniform = len(set(seq_lens)) == 1
        if uniform:  # uniform (generation, CUDA graph): no host data baked in
            L = seq_lens[0]
            tok_req = torch.arange(n, device=dev) // L
            idx_in = torch.arange(n, device=dev) % L
            cs = positions - idx_in.to(positions.dtype)
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
            tok_req = _h2d(idx, dev)
            first = _h2d(starts[:-1].astype(np.int64), dev)
            cs = positions[first][tok_req]
            ce = cs + _h2d(lens, dev).to(positions.dtype)
            bounds = [(int(starts[r]), int(starts[r + 1])) for r in range(nreq)]
        step = _Step(
            n=n, nreq=nreq, positions=positions, tok_req=tok_req, cs=cs, ce=ce, bounds=bounds,
            uniform=uniform, capturing=capturing,
        )  # fmt: skip
        return self._select_target(step, attn_metadata).forward(
            self, step, attn_metadata, input_ids, **kwargs
        )

    def _run_layers(
        self,
        step: "_Step",
        ctx: StepCtx,
        attn_metadata,
        input_ids,
        stop: Optional[int] = None,
        **kwargs,
    ):
        """Embedding, the 40 Hyper-Connection layers and the final norm; `ctx` fixes how the
        indexer reads its keys (exact span per request, or device-side lengths).

        With a bounded replay plan (`prepare_adp_inputs`) the layers from `decoder_replay_split` on run over the
        plan's rows only; the output is scattered back to the step's row layout (rows that did not run are zero)."""
        positions, n = step.positions, step.n
        slot = self.req_slot[step.tok_req].long()
        engram_ids = self.engram_hash.lookback(input_ids, positions, slot, self.hist)
        h = token_embedding(self.embed_tokens, input_ids)
        x = torch.stack(
            [h] * self.hc, dim=1
        )  # as stock (one batched cat; `repeat` runs a strided copy ~2.5x slower)
        pre_mix = x.new_zeros(n, self.hc, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        plan = (
            None
            if self.decoder_replay_split is None
            else getattr(attn_metadata, "decoder_replay_plan", None)
        )
        counts, rows = attn_metadata.all_rank_num_tokens, None
        enc_counts = None
        pending = None  # the previous layer's FFN post update, applied by the next boundary (`_Layer.forward`)
        for li, layer in enumerate(self.layers):
            if (
                li == stop
            ):  # the encoder pass of the two-pass graphs: the residual state entering layer `stop`
                return resolve_hc(x, pending), pre_mix
            if plan is not None and li == self.decoder_replay_split:
                if plan.replay_all_rank_num_tokens is None:
                    raise ValueError(
                        "exchange the attention-DP token counts after prepare_adp_inputs()"
                    )
                counts = plan.replay_all_rank_num_tokens
                x, pending = resolve_hc(x, pending), None
                if sum(counts) == 0:
                    # No rank has a row in the decoder half (every rank's chunk ends before its prompt's final window),
                    # but the boundary owner's compressed rows of every row are still published, as stock's boundary
                    # (`prepare_ced_global_kv`) does unconditionally: later chunks, later turns and the generation
                    # worker read them.
                    if x.shape[0]:
                        layer.publish_global(x, pre_mix, ctx)
                    return h.new_zeros(n, h.shape[-1])
                x, pre_mix, ctx, rows = self._enter_decoder(layer, x, pre_mix, ctx, plan)
            x, pre_mix, pending = layer(
                x, pre_mix, ctx, attn_metadata, engram_ids, counts, enc_counts, pending, defer=True
            )
        x = resolve_hc(x, pending)
        h = hc_pre(x, pre_mix) if x.shape[0] else x.new_empty(0, h.shape[-1])
        h = flashinfer_rmsnorm(h, self.norm, self.norm_eps) if h.shape[0] else h
        if rows is None:
            return h
        return h.new_zeros(n, h.shape[-1]).index_copy_(0, rows, h)

    def _enter_decoder(self, layer, x, pre_mix, ctx: StepCtx, plan):
        """The bounded-replay boundary: publish the owner layer's compressed rows for every row, then keep only the
        plan's rows (the last `window` rows of each prompt chunk, every generation row) and apply the plan's
        per-request SWA floors, so the window of the first kept rows is truncated as in stock."""
        dev = x.device
        floors = _h2d(np.asarray(plan.swa_floors, dtype=np.int32), dev)
        if (
            not plan.replays_local_tokens
        ):  # every row is kept; the floors still hide window rows of earlier chunks
            return x, pre_mix, dataclasses.replace(ctx, floor=floors[ctx.rid]), None
        lens = [int(v) for v in plan.replay_seq_lens.tolist()]
        rows = _h2d(plan.rows, dev).to(torch.int64)
        layer.publish_global(x, pre_mix, ctx)
        x, pre_mix = x.index_select(0, rows), pre_mix.index_select(0, rows)
        if rows.numel() == 0:
            return x, pre_mix, ctx, rows
        pos, rid, ce = (
            ctx.pos.index_select(0, rows),
            ctx.rid.index_select(0, rows),
            ctx.ce.index_select(0, rows),
        )
        begin = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        first = _h2d(np.repeat(begin, lens), dev)
        segs = [
            (int(begin[r]), int(begin[r]) + lens[r], r, span)
            for _, _, r, span in ctx.segs
            if lens[r] > 0
        ]
        dctx = StepCtx(
            pos=pos, rid=rid, cs=pos.index_select(0, first), ce=ce, segs=segs, kvm=ctx.kvm, rope=ctx.rope,
            nreq=ctx.nreq, paged=False, cache=ctx.cache, published=True, floor=floors[rid],
        )  # fmt: skip
        return x, pre_mix, dctx, rows

    def make_replay_plan(self, attn_metadata, all_token_states_required: bool, requests=None):
        """The step's bounded-replay plan (stock's `_make_bounded_replay_plan`, `csa2_replay_plan`): which rows of each
        context chunk run the decoder half. None when the model or step cannot replay."""
        plan = csa2_replay_plan(self, attn_metadata, all_token_states_required, requests)
        if plan is not None and plan.replays_local_tokens and not self._replay_logged:
            self._replay_logged = True
            logger.info(
                f"DeepSeek-V4.1 decoder bounded replay engaged: {plan.num_replay_tokens} of {plan.num_encoder_tokens} "
                f"rows run from layer {self.decoder_replay_split}."
            )
        return plan

    # ---- breakable two-pass prefill graphs (`prefill_graphs.py`) -------------------------------------------------
    def _forward_bcg(
        self, step: "_Step", ctx: StepCtx, attn_metadata, input_ids
    ) -> Optional[torch.Tensor]:
        """A step with a decoder replay plan, from the breakable pass graphs: the encoder pass at the token bucket of
        the largest rank, the bounded-replay boundary eagerly over this rank's rows, the decoder pass at the row bucket
        of the largest rank's replayed rows. None (run it eagerly) when a pass has no bucket: decided from the exchanged
        counts alone, so every attention-DP rank decides alike, whichever target its own rows take (a peer with only
        generation rows replays the graphs too, or the experts' collectives would run at different shapes)."""
        if step.capturing or not self._prefill_graph:
            return None
        plan = getattr(attn_metadata, "decoder_replay_plan", None)
        counts = [int(c) for c in (attn_metadata.all_rank_num_tokens or ())]
        if plan is None or plan.replay_all_rank_num_tokens is None or not counts:
            return None
        dcounts = [int(c) for c in plan.replay_all_rank_num_tokens]
        if self._enc_graph is None:
            pool = PassGraphPool()
            self._enc_graph = EncoderPassGraph(self._encoder_body, pool, len(counts))
            self._dec_graph = DecoderPassGraph(
                self._decoder_body, pool, len(counts), self.decoder_replay_window
            )
        max_tokens = int(getattr(attn_metadata, "max_num_tokens", None) or self._a2a_rows)
        max_reqs = int(getattr(attn_metadata, "max_num_requests", None) or _MAX_REQS)
        enc = self._enc_graph.select(counts, max_tokens)
        dec = self._dec_graph.select(dcounts, max_tokens, max_reqs)
        if enc is None or dec is None:
            return None
        n = step.n
        slot = self.req_slot[step.tok_req].long()
        engram_ids = self.engram_hash.lookback(input_ids, step.positions, slot, self.hist)
        x, pre_mix = self._enc_graph.replay(
            enc, (token_embedding(self.embed_tokens, input_ids),), n, ctx, engram_ids, counts
        )
        x, pre_mix, dctx, rows = self._enter_decoder(
            self.layers[self.decoder_replay_split], x, pre_mix, ctx, plan
        )
        (h,) = self._dec_graph.replay(dec, (x, pre_mix), x.shape[0], dctx, None, dcounts)
        return h.clone() if rows is None else h.new_zeros(n, h.shape[-1]).index_copy_(0, rows, h)

    def _encoder_body(self, h: torch.Tensor, live, counts, padding):
        """The encoder pass graph's body: embeddings -> the residual stream entering `decoder_replay_split`."""
        x = torch.stack([h] * self.hc, dim=1)
        pre_mix = x.new_zeros(h.shape[0], self.hc, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        pending = None
        for layer in self.layers[: self.decoder_replay_split]:
            x, pre_mix, pending = layer(
                x,
                pre_mix,
                None,
                None,
                None,
                counts,
                counts,
                pending,
                defer=True,
                live=live,
                unrouted=padding,
            )
        return resolve_hc(x, pending), pre_mix

    def _decoder_body(self, x: torch.Tensor, pre_mix: torch.Tensor, live, counts, padding):
        """The decoder pass graph's body: the replayed layers, the final collapse and norm."""
        pending = None
        for layer in self.layers[self.decoder_replay_split :]:
            x, pre_mix, pending = layer(
                x,
                pre_mix,
                None,
                None,
                None,
                counts,
                counts,
                pending,
                defer=True,
                live=live,
                unrouted=padding,
            )
        x = resolve_hc(x, pending)
        return (flashinfer_rmsnorm(hc_pre(x, pre_mix), self.norm, self.norm_eps),)

    def _step_ctx(self, step: "_Step", attn_metadata, segs, paged: bool) -> StepCtx:
        return StepCtx(
            pos=step.positions, rid=step.tok_req, cs=step.cs, ce=step.ce, segs=segs,
            kvm=KvPools(attn_metadata.kv_cache_manager, self._swa_tables), rope=self.rope,
            nreq=step.nreq, paged=paged,
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


class PrefillTarget:
    """The general case: context rows (chunks), a mixed batch, or any step too wide for the paged indexer."""

    def forward(
        self, core: ModelingV2Core, step: _Step, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        assert not step.capturing, "only generation steps are captured"
        # one past each request's last position, from the host's cached lengths (positions are cached + offset); a
        # device read here would drain the stream before every context step
        cached = getattr(
            getattr(attn_metadata, "kv_cache_params", None), "num_cached_tokens_per_seq", None
        )
        if cached is not None and len(cached) >= len(step.bounds):
            spans = [int(cached[r]) + (b - a) for r, (a, b) in enumerate(step.bounds)]
        else:
            spans = (
                torch.stack([step.positions[b - 1] for _, b in step.bounds]).long() + 1
            ).tolist()
        segs = [(a, b, r, int(spans[r])) for r, (a, b) in enumerate(step.bounds)]
        ctx = core._step_ctx(step, attn_metadata, segs=segs, paged=False)
        if (
            core._recovery is not None
        ):  # Encoder recovery: SWA reads from the recovery floor, historical rows unpublished
            hist_end, floor = (
                _h2d(np.asarray(v, dtype=np.int32), step.positions.device) for v in core._recovery
            )
            ctx = dataclasses.replace(
                ctx, floor=floor[ctx.rid], publish=ctx.pos >= hist_end[ctx.rid]
            )
        out = core._forward_bcg(step, ctx, attn_metadata, input_ids) if core._bcg else None
        return (
            out
            if out is not None
            else core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)
        )


class DecodeTarget:
    """Short uniform generation steps: lengths stay on the device (CUDA-graph safe), paged indexer kernel."""

    def forward(
        self, core: ModelingV2Core, step: _Step, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        ctx = core._step_ctx(step, attn_metadata, segs=[], paged=True)
        # An attention-DP peer of a context step (an idle rank's dummy request included) replays the same pass graphs
        # on its generation rows: the experts' collectives run at the
        # bucket's shape on every rank (stock's identity plan)
        out = core._forward_bcg(step, ctx, attn_metadata, input_ids) if core._bcg else None
        return (
            out
            if out is not None
            else core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)
        )


class ModelingV2DeepseekV41FlashNvfp4Sm103Dep2(
    DecoderModelForCausalLM[ModelingV2Core, PretrainedConfig]
):
    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The built-in model's step-loop hooks without speculation (`step_hooks`)."""
        return step_hooks(None)

    def __init__(self, model_config: ModelConfig):
        model_config = text_model_config(model_config)
        # what routing declines (`limits.dep2_refused`), for a construction routing did not see: `auto` then falls back
        refused = dep2_refused(model_config)
        if refused:
            raise NoTarget(f"the deepseek_v41 dep2 target does not implement: {'; '.join(refused)}")
        cfg = model_config.pretrained_config
        assert cfg is not None
        super().__init__(
            ModelingV2Core(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        assert getattr(model_config, "spec_config", None) is None, (
            "the ctx target has no speculative path"
        )

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        """The engine's `DecoderModelForCausalLM.forward`, plus the step's context requests handed to the core, as
        stock's `DeepseekV41ForCausalLM.forward` takes them: the engine passes `context_requests` (attention DP, a
        private decoder cache, or an Encoder replay in the step), and the core reads their Encoder recovery plans
        (`py_ced_replay`) and consumes them after the forward succeeds. The engine's base forward drops every extra
        keyword, so without this the core never saw a recovery and a replay was never consumed: the request's next
        chunk then failed the metadata's Global-claim check."""
        refuse_image_step(kwargs.get("multimodal_params"))
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
            context_requests=kwargs.get("context_requests"),
        )
        return self.logits_processor.forward(
            output, self.lm_head, attn_metadata, return_context_logits
        )

    def validate_request(self, request) -> None:
        """Engine hook at admission: this target serves text only, so a context request with image input is refused
        (`image_input.refuse_image_request`)."""
        refuse_image_request(request)

    def prepare_adp_inputs(
        self, attn_metadata, *, all_token_states_required: bool, requests=None
    ) -> None:
        """Engine hook, before the attention-DP count exchange: fix which rows of each context chunk run the decoder
        half, so the exchange carries the decoder row counts of every rank (the experts' collectives use them)."""
        attn_metadata.decoder_replay_plan = self.model.make_replay_plan(
            attn_metadata, all_token_states_required, requests
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
