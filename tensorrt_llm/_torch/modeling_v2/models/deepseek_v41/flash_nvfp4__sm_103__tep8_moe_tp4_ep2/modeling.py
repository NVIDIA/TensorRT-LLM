# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-v4.1-flash-nvfp4 / sm_103 / tep8, MoE tp4 x ep2 (the generation role).

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

`tep8_moe_tp4_ep2` is `tensor_parallel_size: 8`, `moe_expert_parallel_size: 2` (so the
experts are 4-way tensor x 2-way expert parallel) and `enable_attention_dp: false`: every rank sees
the same tokens, attention heads and output groups are split 8 ways inside the CSA2 module, the
experts and the engram tables are sharded, and each rank runs its share of the shared expert (`_SharedExpert`: a
TP piece of its intermediate width, or a shard scaled by `shards / tp`) and adds it to its routed partial sum before
the layer's one all-reduce, which sums both.

Composition. Attention (`attention.py`), the routed experts, the shared expert, the Hyper-Connection mixes,
the clamped SwiGLU and the engram gate are built from entries: the family's own (`models/deepseek_v41/ops/
{attention,residual,moe,engram}`: `csa2_*`, `hc_*`, `dsv4_router_logits`, `dsv4_route_nvfp4_quant`,
`nvfp4_fp8_quantize`, `shared_expert_fp8_deep_gemm(_prequant)`, `engram_gate_fused` (stock's Engram gate),
`engram_lookback_hash`), the target's (`ops/`: `shared_expert_lowm`, `engram_ll_allgather`, `dsv4_route_tables`,
`csa2_q_norm`, `csa2_logits_schedule`) and the shared catalog's (`gemm/cublas_mm`, `norm/flashinfer_rmsnorm`,
`moe/fp4_block_scale_moe_runner`, `comm/allgather`, `comm/reducescatter`), with the engine's `CSA2CacheManager` as the
allocator of the cache pools. This file owns the trunk: the HC loop, routing, the engram branch (hash, table lookup),
weight routing and the registration shell. Not an entry: the engram n-gram hash and table lookup, the token embedding,
the engram's all-reduce/all-gather (the engine's), and the inherited `DecoderModel` / `DecoderModelForCausalLM` shells
(lm_head, logits gather, speculative worker).

Two execution paths, chosen per step by `ModelingV2Core._select_target`: `DecodeTarget` (generation-only steps of
at most `_PAGED_MAX_ROWS` rows with equal request lengths: the indexer reads lengths on the device, so the step is
CUDA-graph safe) and `PrefillTarget` (everything else: exact span per request, one host read each).

Bounded decoder replay (context steps; stock's `TRTLLM_V41_DECODER_BOUNDED_REPLAY`, on by default; the policy and the
plan are stock's own, `ops/attention/csa2_decoder_replay`). Layer 20, the last unpooled KV owner, publishes the
compressed KV and index keys of every row of a context chunk; from there to the last layer only the rows of the chunk
inside the final 140 tokens of its prompt run (queries, index selection, attention, experts), with the window of their
first rows truncated at the window start, and every generation row of a mixed step. Without attention DP the plan is
made in the forward, as stock's (`_plan_bounded_replay`); the engine's `prepare_adp_inputs` hook makes it ahead of a
breakable prefill graph's padding choice. Approximate across stacked SWA layers, as stock is. Requests that return
context logits (or other per-row outputs) and CUDA-graph steps run every row, as in stock.

Engram look-back and speculative decoding:

* engram hashes read the n-gram look-back from a device-side ring of recent token ids per request
  (graph-safe, correct under the overlap scheduler and speculative rollback); an engine hook
  seeds it for requests new to the engine;
* the speculative (DSpark) draft model and worker are the engine's (`SpecDecOneEngineForCausalLM` builds them
  from `speculative_config`; the draft stages live in the same checkpoint, `mtp.*`); this target only captures
  the residual stream the draft reads.
"""

import contextlib
import dataclasses
import math
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
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
    csa2_complete_encoder_replay,
    csa2_encoder_recovery,
    csa2_encoder_replay_enabled,
    csa2_encoder_replay_tokens,
    csa2_replay_plan,
    csa2_replay_policy,
    csa2_step_replay_plan,
    csa2_uses_private_decoder_cache,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    SMALL_M_ROWS,
    csa2_mxfp8_gemm_prewarm,
    csa2_mxfp8_gemm_tune_rows,
    csa2_mxfp8_graph_rows,
    csa2_mxfp8_linear,
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
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.embedding.token_embedding import (
    token_embedding,
    token_embedding_create,
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
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_router_logits import (
    dsv4_router_logits,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (
    MIN_ROWS as FP8_1X128_MIN_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (
    nvfp4_fp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (
    deep_gemm_weight_scale,
    shared_expert_fp8_deep_gemm,
    shared_expert_fp8_deep_gemm_prequant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_allreduce_post_collapse_norm import (
    hc_allreduce_post_collapse_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_boundary import hc_boundary
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_coefficients import (
    hc_coefficients,
    hc_coefficients_fma,
    hc_fma_fits,
    hc_fma_state,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lowm import (
    hc_lamport_workspace,
    hc_split_enabled,
    moe_unfinalized,
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
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.context import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType, decode_stream_priority
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from ..common import Fp8Rows, KvPools, opt_into_small_m_gemm, text_model_config
from ..engram_hash import ENGRAM_RING, EngramHash, engram_buckets
from ..image_input import refuse_image_request, refuse_image_step
from ..limits import tep8_refused
from ..loaders import load_head
from ..swa_tables import engine_swa_tables
from . import weights as _weights
from .attention import StepCtx, _Attention
from .ops.engram.engram_ll_allgather import (
    engram_ll_allgather,
    ll_allgather_takes,
    ll_allgather_workspace,
)
from .ops.engram.engram_ll_allgather_pair import (
    ENGRAM_SIDE_MAX_PROGRAMS,
    engram_ll_allgather_pair,
    ll_allgather_pair_takes,
)
from .ops.moe.dsv4_route_tables import (
    ROUTING_TABLE_TILES,
    dsv4_route_nvfp4_quant_tables,
    dsv4_route_tables,
    dynamic_ep_applies,
    fp4_moe_runner_tables,
    routing_table_max_ctas,
    routing_tables_fit,
    table_permutation,
)
from .ops.moe.shared_expert_lowm import shared_expert_lowm, shared_expert_lowm_serves

# The GPU architecture this target IS; routing will not send another one here.
_SM = (10, 3)

_BLOCK = 32  # FP8 block edge of the checkpoint's dense weights
_SHARED_TILE = (
    128  # the shared expert's per-rank width is padded to this (the MXFP8 GEMMs' tile edge)
)
_HC_POST_MULT = 2.0  # post-mix scale of the Hyper-Connections (`post = 2 * sigmoid(.)`)
# The engram tables stay in pinned host memory that the lookup kernel reads through UVA, as the engine's
# `ShardedFp8MultiHeadEmbedding` does; processes of one node that pass the same `share_key` map one copy.
# The loop index before which both Engram layers' WKV shards are projected: the built-in sm_103 TP4 / TP8 decode
# schedule (`_engram_projection_schedule = {0: (1, 14)}`: one gather of both lookups, both projections, one gather of
# both WKV shards); the lookups start with the step.
_ENGRAM_PROJECT_AT = 0
_RING = ENGRAM_RING  # positions of history per slot (the family's ring)
_MAX_REQS = 4096
_UNIFORM_ROWS, _UNIFORM_MAX_L = (
    4096,
    16,
)  # uniform steps whose request rows / in-request offsets are load-time constants
# query rows of an eager generation step on the paged path (the fused indexer has no row cap; a CUDA-graph step takes
# it at any width)
_PAGED_MAX_ROWS = 4096


class _Engram(nn.Module):
    """Adds a gated n-gram lookup into the residual stream (layers 1 and 14)."""

    # The table is built in `load`, after the loader's post-init check: the module tells it that the pinned host
    # table is filled in place, which only the regular load_format=AUTO checkpoint load does.
    _requires_regular_checkpoint_loading = True

    # Engine hook read by the decode-graph warmup (`warmup.py`): a model holding a module with this flag tunes and
    # dispatches its MXFP8 `Linear`s to FlashInfer inside decode CUDA graphs. Stock sets it on its Engram WKV
    # (`EngramFp8Projection`); this target's own GEMMs are entries, so the flag's effect here is on the engine's DSpark
    # drafter, whose MXFP8 projection then runs the tuned FlashInfer GEMM as in stock.
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
        # The table stays FP8 as stored, in pinned host memory (see `load`).
        self.table = None
        self.wkv = None
        self.wkv_q = self.wkv_cols = self.alpha = self.gate_w = None  # see `load`
        self.q_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(self.hc, self.dim), requires_grad=False)
        self.allreduce = engine_allreduce_create(mapping)
        self.events = (
            torch.cuda.Event(),
            torch.cuda.Event(),
        )  # (looked up, projected) of the precomputed path

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
        # the k | v projection on the checkpoint's E4M3 bytes (an MXFP8 GEMM, as the engine's engram projection runs)
        w, s = weights[prefix + "wkv.weight"].to(device), weights[prefix + "wkv.scale"].to(device)
        assert w.dtype == torch.float8_e4m3fn, "the engram wkv is E4M3 in the checkpoint"
        self.wkv = w.contiguous()
        self.wkv_s = mxfp8_row_scales(s, tuple(w.shape))
        self.wkv_sw = mxfp8_swizzled_scales(self.wkv_s)
        self.wkv_alpha = torch.ones(1, dtype=torch.float32, device=device)
        # This rank's contiguous `out / tp` WKV rows in the checkpoint's own MXFP8 (E4M3 bytes, the 32x32 UE8M0 tiles
        # expanded to per-row 1x32 scales): the decode path projects only these columns and gathers the shards.
        out_f, in_f = w.shape
        if out_f % self.mapping.tp_size == 0 and (out_f // self.mapping.tp_size) % 128 == 0:
            rows = out_f // self.mapping.tp_size
            r0 = self.mapping.tp_rank * rows
            scales = self.wkv_s
            cut = rows // 2 // 128 * 128  # whole 128-row tiles: 3200 -> 1536 + 1664
            self.wkv_q = []  # the shard in two parts: (E4M3 [part, in], scales, swizzled scales)
            for a, b in ((r0, r0 + cut), (r0 + cut, r0 + rows)):
                ws = scales[a:b].contiguous()
                self.wkv_q.append(
                    (w.view(torch.float8_e4m3fn)[a:b].contiguous(), ws, mxfp8_swizzled_scales(ws))
                )
            self.wkv_cols = rows
            self.alpha = torch.ones(1, dtype=torch.float32, device=device)
        self.q_weight.data.copy_(weights[prefix + "q_weight"])
        self.k_weight.data.copy_(weights[prefix + "k_weight"])
        self.gate_w = (self.q_weight.float() * self.k_weight.float()).contiguous()

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

    # ---- decode sizes: the engine's precomputed path (lookup and projection ahead of the layer, peer-memory gathers)
    def precomputes(self, n: int, gather) -> bool:
        """Whether a step of `n` rows takes the precomputed path: head-sharded table, this rank's WKV column shard, and
        both shards fit the peer-memory gather. Reads only shapes, so every rank of the group decides alike."""
        return (
            self.wkv_q is not None
            and not self.mapping.enable_attention_dp
            and self.table.shard_heads
            and ll_allgather_takes(
                gather, n, self.table.num_local_heads * self.head_dim, torch.bfloat16
            )
            and ll_allgather_takes(gather, n, self.wkv_cols, torch.bfloat16)
        )

    def lookup_async(
        self,
        ids: torch.Tensor,
        stream: torch.cuda.Stream,
        done: torch.cuda.Event,
        ordered: bool = False,
    ) -> torch.Tensor:
        """This rank's heads of the table rows `[n, local_heads * head_dim]`, looked up on `stream` (half the SMs, as
        the engine's background lookup), `done` recorded after. `ordered`: `ids` were produced on `stream` (no wait for
        this stream's work)."""
        if not ordered:
            stream.wait_stream(torch.cuda.current_stream())
        ids.record_stream(
            stream
        )  # read on `stream` after this step's code drops it: keep the allocator off it until then
        with torch.cuda.stream(stream):
            emb = self.table(ids, background=True).flatten(1)
            done.record()
        emb.record_stream(torch.cuda.current_stream())
        return emb

    def project_async(
        self, emb: torch.Tensor, stream: torch.cuda.Stream, done: torch.cuda.Event
    ) -> torch.Tensor:
        """This rank's WKV columns `[n, out / tp]` of the gathered lookup `emb` `[n, heads * head_dim]`, projected on
        `stream` (the Engram stream) after the current stream's work so far; `done` recorded after (the built-in
        `project_combined`)."""
        cur = torch.cuda.current_stream()
        stream.wait_stream(cur)
        emb.record_stream(stream)
        with torch.cuda.stream(stream):
            # the shard in two parts: each fits the low-M kernel's weight cap (exact bf16-activation products)
            kv = torch.cat(
                [csa2_mxfp8_linear(emb, w, ws, wsw, self.alpha) for w, ws, wsw in self.wkv_q], dim=1
            )
            done.record()
        kv.record_stream(cur)
        return kv

    def forward_precomputed(
        self, x: torch.Tensor, kv: tuple, gather, projected: torch.cuda.Event
    ) -> torch.Tensor:
        """x [n, hc, d] -> x + gate * value, from the projected WKV `kv = (tensor, gathered)`: the whole `[n, out]` WKV
        when the pair gather took both shards on the Engram stream, else this rank's `[n, out / tp]` shard, gathered
        here on the model stream (the built-in `_EngramKvPairShard` / `_EngramKvShard`)."""
        torch.cuda.current_stream().wait_event(projected)
        kv, gathered = kv
        if not gathered:
            kv = engram_ll_allgather(gather, kv)
        return engram_gate_fused(
            x.contiguous(), kv, self.q_weight, self.k_weight, self.gate_w, self.eps
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor, all_rank_num_tokens) -> torch.Tensor:
        """x [n, hc, d]; ids [n, cols] -> x + gate * value."""
        sizes = all_rank_num_tokens
        if self.mapping.tp_size > 1 and self.mapping.enable_attention_dp:
            # Ranks hold different tokens: gather ids, look up the local shard,
            # sum the shards, keep this rank's rows.
            gathered = engine_allgather(ids, self.mapping, 0, sizes)
            emb = engine_allreduce(self.allreduce, self._lookup(gathered))
            first = sum(sizes[: self.mapping.tp_rank])
            emb = emb[first : first + ids.shape[0]]
        elif self.mapping.tp_size > 1:
            emb = engine_allreduce(self.allreduce, self._lookup(ids))
        else:
            emb = self._lookup(ids)
        kv = csa2_mxfp8_linear(emb.flatten(-2), self.wkv, self.wkv_s, self.wkv_sw, self.wkv_alpha)
        # stock's gate at every step size (`engram_gate` with the cached norm product, the residual added in fp32), as
        # the precomputed path: the family's two-norm Triton gate reduces in another order (bf16 ulps on a few rows)
        return engram_gate_fused(
            x.contiguous(), kv, self.q_weight, self.k_weight, self.gate_w, self.eps
        )


class _SharedExpert(nn.Module):
    """One SwiGLU FFN every token goes through, as stock's `DeepseekV4MoE` runs its shared expert, on stock's bytes (the
    checkpoint's E4M3 requantized to 128x128 tiles, `weights.load_shared_expert`):

    * bf16 steps of at most 64 rows: stock's piece path (`full_tp_shared_expert_forward`, `shared_expert_lowm`): this
      rank's `inter / tp` rows of the intermediate width (`w13` = gate rows then up rows, `w2` by columns) with per-row
      1x32 UE8M0 scales, zero-padded to `part_p` (a multiple of 128: 288 -> 384 at tp 8; a padded gate and up give
      `silu(0) * 0 = 0`, which the zero `w2` columns ignore). Its output is this rank's share of the all-reduce's sum.
    * every other step: stock's `GatedMLP` on this rank's shard of `shards = gcd(inter / 128, tp)` (stock's
      `_compute_shared_expert_tp_size`: 2 at tp 8, so `inter / 2` rows, rank `% shards`) with 128x128 FP32 block
      scales, on its SM100 DeepGEMM path (`shared_expert_fp8_deep_gemm`; from 1024 rows on the FFN boundary's FP8
      1x128 copy, `shared_expert_fp8_deep_gemm_prequant`), scaled by `shards / tp` (stock's `shared_output_scale`)
      before the all-reduce sums the `tp / shards` replicas.
    """

    def __init__(self, dim: int, inter: int, limit: float, tp: int = 1):
        super().__init__()
        self.limit = limit
        assert inter % tp == 0, (inter, tp)
        self.part = inter // tp
        self.part_p = -(-self.part // _SHARED_TILE) * _SHARED_TILE
        f8, u8, f32, pp = torch.float8_e4m3fn, torch.uint8, torch.float32, self.part_p
        self.w13 = nn.Parameter(torch.empty(2 * pp, dim, dtype=f8), requires_grad=False)
        self.w13_s = nn.Parameter(torch.empty(2 * pp, dim // _BLOCK, dtype=u8), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(dim, pp, dtype=f8), requires_grad=False)
        self.w2_s = nn.Parameter(torch.empty(dim, pp // _BLOCK, dtype=u8), requires_grad=False)
        self.shards = math.gcd(inter // _SHARED_TILE, tp)
        self.shard_scale = self.shards / tp
        si, t = inter // self.shards, _SHARED_TILE
        self.g13 = nn.Parameter(torch.empty(2 * si, dim, dtype=f8), requires_grad=False)
        self.g13_s = nn.Parameter(
            torch.empty(2 * si // t, dim // t, dtype=f32), requires_grad=False
        )
        self.g2 = nn.Parameter(torch.empty(dim, si, dtype=f8), requires_grad=False)
        self.g2_s = nn.Parameter(torch.empty(dim // t, si // t, dtype=f32), requires_grad=False)
        self.g13_sf = self.g2_sf = (
            None  # the shard's scales in DeepGEMM's layout, built by `post_load`
        )

    def post_load(self) -> None:
        self.g13_sf = deep_gemm_weight_scale(self.g13_s, *self.g13.shape)
        self.g2_sf = deep_gemm_weight_scale(self.g2_s, *self.g2.shape)

    def forward(self, x: torch.Tensor, fp8_1x128=None) -> torch.Tensor:
        """`fp8_1x128`: `x` as stock's FFN boundary quantized it for the shared expert from `FP8_1X128_MIN_ROWS` rows
        (`(E4M3 [n, H], packed UE8M0)`, `nvfp4_fp8_quantize`), which stock's `GatedMLP` then takes instead of `x`."""
        if fp8_1x128 is not None:
            y = shared_expert_fp8_deep_gemm_prequant(
                *fp8_1x128, self.g13, self.g13_sf, self.g2, self.g2_sf, self.limit
            )
        elif shared_expert_lowm_serves(x):
            # stock's piece path (`full_tp_shared_expert_forward`) for every
            # bf16 step of at most 64 rows, eager or graph
            return shared_expert_lowm(x, self.w13, self.w13_s, self.w2, self.w2_s, self.limit)
        else:
            y = shared_expert_fp8_deep_gemm(
                x, self.g13, self.g13_sf, self.g13_s, self.g2, self.g2_sf, self.g2_s, self.limit
            )
        if (
            self.shard_scale != 1
        ):  # stock: `shared_output *= self.shared_output_scale` (a power of two: exact)
            y *= self.shard_scale
        return y


_SF_VEC = 16
_MOE_MAX_T = 8192  # the runner's certified token column
# The routing the runner is told, as the engine's DeepSeek-V4 MoE tells it (`DeepSeekV4MoeRoutingMethod`: type
# DeepSeekV3, the kernels having no V4 type; the gate's n_group = topk_group = 1 when the checkpoint has none; its
# routed scaling): the ids and weights are ours either way, but the runner's own routing (tactics off the tables) and
# the autotuner's dummy routing follow it -- with the Renormalize type the dummy spread six picks per row over all
# experts and the tuner timed, and chose tactics for, a load the decode steps never see.
_ROUTING_METHOD, _N_GROUP, _TOPK_GROUP = 2, 1, 1
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
    tokens: run and hand back the partial sum; the layer reduces it together with the shared expert's.

    With `dynamic` (the built-in model's dynamic expert parallelism,
    `ConfigurableMoE._dynamic_expert_parallelism_applies`;
    `dynamic_ep_applies`) every rank holds its MoE-TP shard of all the experts (`held` = every expert from 0), and a
    decode call on routing tables runs this EP rank's home-first share of the experts it uses; every other call runs the
    static window through views of the weights.
    """

    def __init__(self, cfg, mapping, dynamic: bool = False):
        super().__init__()
        self.cfg, self.topk = cfg, cfg.num_experts_per_tok
        self.adp = bool(mapping.enable_attention_dp)
        self.group = list(range(mapping.world_size))
        self.tp, self.tp_rank = mapping.moe_tp_size, mapping.moe_tp_rank
        self.local = cfg.n_routed_experts // mapping.moe_ep_size
        self.offset = mapping.moe_ep_rank * self.local
        self.dynamic_rank = mapping.moe_ep_rank if dynamic else None
        # The experts whose weights this rank holds: its static window, or with the dynamic split all of them.
        self.held_offset, self.held = (
            (0, cfg.n_routed_experts) if dynamic else (self.offset, self.local)
        )
        self.inter_t = _weights.padded_inter(cfg.moe_intermediate_size, self.tp)
        h, e, i = cfg.hidden_size, self.held, self.inter_t
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
        self._table_tile: dict = {}  # rows -> the routing-table tile the runner's tactic ran on (see `_run_on_tables`)

    def boundary_quantizes(self, n: int) -> bool:
        """Whether a step of `n` rows routes on the tables from rows the FFN boundary quantized (stock's
        `try_fused_route` after the boundary's NVFP4 copy): the routing launch then only routes."""
        return not self.adp and routing_tables_fit(n, self.local, self.cfg.n_routed_experts)

    def _window(self, t: torch.Tensor) -> torch.Tensor:
        """The static EP window's slice of a per-expert operand (a view; the operand itself when that is all it
        holds)."""
        return (
            t
            if self.held == self.local
            else t.narrow(0, self.offset - self.held_offset, self.local)
        )

    def _run_on_tables(self, chunk: torch.Tensor, finalize: bool, xq=None):
        """Decode sizes (as stock's TRTLLM-Gen NVFP4 MoE): the routing launch also writes the runner's routing tables
        for this rank's expert window, so the runner skips its routing kernel. The tables cover the tile the runner's
        tactic used for this many rows once it ran on one, else every tile (`_table_tile`, target state)."""
        n = chunk.shape[0]
        tile = self._table_tile.get(n)
        tiles = list(ROUTING_TABLE_TILES) if tile is None else [tile]
        logits = dsv4_router_logits(chunk, self.gate_weight)
        if (
            xq is not None
        ):  # the FFN boundary wrote the NVFP4 rows (`self.g1`): the launch only routes
            topk_ids, topk_w, tables = dsv4_route_tables(
                logits,
                self.gate_bias,
                self.topk,
                self.cfg.routed_scaling_factor,
                chunk.shape[1],
                self.offset,
                self.local,
                tiles,
                self.dynamic_rank,
            )
            xq, xsf = xq
        else:
            topk_ids, topk_w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(
                logits,
                self.gate_bias,
                chunk,
                self.g1,
                self.topk,
                self.cfg.routed_scaling_factor,
                self.offset,
                self.local,
                tiles,
                self.dynamic_rank,
            )
        experts, dyn = self.cfg.n_routed_experts, self.dynamic_rank
        out = fp4_moe_runner_tables(
            xq, xsf.reshape(-1).view(torch.float8_e4m3fn), self.fc1_w, self.fc1_s, self.clamp, self.fc2_w, self.fc2_s,
            self.o1, self.o1_gate, self.o2, self.cfg.n_routed_experts, self.topk, self.inter_t, self.offset, self.local,
            _ROUTING_METHOD, finalize, _ACT_TYPE_SWIGLU, topk_w, topk_ids, tables, tiles,
            _N_GROUP, _TOPK_GROUP, self.cfg.routed_scaling_factor,
            cute_fc2=not finalize,  # as the built-in decode MoE: FC1-only TRTLLM-Gen run plus the CuTe DSL FC2
            # the expert GEMMs' grids bounded by the tables, and the dynamic split over every held expert
            routing_max_ctas=routing_table_max_ctas(n, self.topk, experts, dyn),
            dynamic_experts=experts if dyn is not None else 0,
        )  # fmt: skip
        if finalize:
            return out, topk_w
        perm, used = table_permutation(out, tables, tiles, n, self.topk)
        if (
            not torch.cuda.is_current_stream_capturing()
        ):  # a capture replays the tile the warm-up runs settled on
            if used is None:
                self._table_tile.pop(n, None)
            else:
                self._table_tile[n] = used
        return [out[0], out[1], perm], topk_w

    def _run(self, chunk: torch.Tensor, finalize: bool, xq=None):
        if not self.adp and routing_tables_fit(
            chunk.shape[0], self.local, self.cfg.n_routed_experts
        ):
            return self._run_on_tables(chunk, finalize, xq)
        # Routing and the NVFP4 quantization of the rows in one launch (row-major scales, as the runner reads them).
        logits = dsv4_router_logits(chunk, self.gate_weight)
        topk_ids, topk_w, xq, xsf = dsv4_route_nvfp4_quant(
            logits, self.gate_bias, chunk, self.g1, self.topk, self.cfg.routed_scaling_factor
        )
        w = self._window
        out = fp4_block_scale_moe_runner(
            None, None, xq, xsf.reshape(-1).view(torch.float8_e4m3fn),
            w(self.fc1_w), w(self.fc1_s), None, None, None, w(self.clamp),
            w(self.fc2_w), w(self.fc2_s), None,
            w(self.o1), w(self.o1_gate), w(self.o2),
            self.cfg.n_routed_experts, self.topk, _N_GROUP, _TOPK_GROUP,
            self.inter_t, self.offset, self.local, self.cfg.routed_scaling_factor,
            _ROUTING_METHOD, finalize, _ACT_TYPE_SWIGLU, topk_w, topk_ids,
        )  # fmt: skip
        return out, topk_w

    def _experts(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            self._run(chunk, True)[0][0]
            for chunk in torch.split(x, _moe_chunk_sizes(x.shape[0]), dim=0)
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def experts_unfinalized(self, x: torch.Tensor, xq=None):
        """-> (expert rows `[P, h]`, routing weights `[n, topk]`, `[n, topk]` row of each (token, slot), -1 if
        the expert is on another rank): the routed output before the top-k combine, which the next boundary
        kernel finalizes in the same launch as the all-reduce. One chunk only."""
        assert x.shape[0] <= _MOE_MAX_T
        out, topk_w = self._run(x, False, xq)
        return out[0], topk_w, out[2]

    def forward(self, x: torch.Tensor, all_rank_num_tokens) -> torch.Tensor:
        n = x.shape[0]
        if self.adp:
            rows = max(int(c) for c in all_rank_num_tokens)
            xp = x if rows == n else F.pad(x, [0, 0, 0, rows - n])
            routed = cat_reducescatter(
                self._experts(cat_allgather(xp.contiguous(), None, self.group)), None, self.group
            )
            return routed if rows == n else torch.split(routed, [n, rows - n], dim=0)[0]
        return self._experts(
            x.contiguous()
        )  # this rank's partial sum: the layer's all-reduce completes it


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
            setattr(
                self, f"hc_{n}_bw", None
            )  # `hc_boundary.prepare_weight(fn)`, built after the weights are loaded
        spec = getattr(model_config, "spec_config", None)
        # the built-in model's dynamic expert parallelism (it opts in with `moe_dynamic_ep`): batch-1 workers at MoE EP
        # 2
        self.moe = _Moe(
            cfg,
            self.mapping,
            dynamic_ep_applies(
                getattr(model_config, "max_batch_size", None),
                spec.tokens_per_gen_step if spec is not None else 1,
                bool(self.mapping.enable_attention_dp),
                self.mapping.moe_ep_size,
                self.mapping.moe_tp_size,
            ),
        )
        self.tp = self.mapping.tp_size
        assert not self.mapping.enable_attention_dp, "tep8 keeps the same tokens on every rank"
        self.shared = _SharedExpert(self.dim, cfg.moe_intermediate_size, cfg.swiglu_limit, self.tp)
        self.engram = _Engram(cfg, self.mapping, engram_slot) if engram_slot is not None else None
        # Attention and the FFN hand back partial sums over the tensor-parallel group; this layer reduces
        # them with the engine's all-reduce (multi-node NVLink one-/two-shot kernels, not NCCL).
        self.attn.reduce_output = False
        self.attn.side_stream = (aux_stream_dict or {}).get(AuxStreamType.Attention)
        self.attn.weights_stream = (aux_stream_dict or {}).get("csa2_index_weights")
        self.lamport = None
        self._coeff_stream = (aux_stream_dict or {}).get(AuxStreamType.MhcCoefficients)
        self._fma = None  # {"attn" | "ffn": (weight, ticket)} of the one-launch coefficients, built by `post_load`
        self._events = {
            k: (torch.cuda.Event(), torch.cuda.Event()) for k in ("attn", "ffn", "shared")
        }
        self._shared_stream = (aux_stream_dict or {}).get(AuxStreamType.MoeShared)
        self.allreduce = (
            engine_allreduce_create(self.mapping, model_config.allreduce_strategy, torch.bfloat16)
            if self.tp > 1
            else None
        )

    def post_load(self, lamport) -> None:
        self.shared.post_load()
        if hc_boundary.supported(self.dim):
            for n in ("attn", "ffn"):
                setattr(self, f"hc_{n}_bw", hc_boundary.prepare_weight(getattr(self, f"hc_{n}_fn")))
            torch.cuda.current_stream().synchronize()
        self.lamport = (
            lamport  # None: the split boundaries reduce with the engine's all-reduce instead
        )
        # Under attention TP every rank computes the same residual stream (replicated), so boundaries of up to 8 rows
        # take their coefficients in one CUDA-core launch (the built-in `mHC._fma_coefficients_ready`): its TF32-rounded
        # weight and an arrival ticket per boundary.
        self._fma = None
        if not self.mapping.enable_attention_dp and hc_fma_fits(1, self.dim, self.hc):
            self._fma = {
                n: hc_fma_state(getattr(self, f"hc_{n}_fn"), self.dim, self.sinkhorn)
                for n in ("attn", "ffn")
            }
            torch.cuda.current_stream().synchronize()

    def _reduce(self, x: torch.Tensor):
        return x if self.allreduce is None else engine_allreduce(self.allreduce, x)

    def _reduce_unfinalized(self, parts) -> torch.Tensor:
        return self._reduce(moe_unfinalized_finalize(parts))

    def _shared_overlapped(self, x: torch.Tensor):
        """-> (the shared expert's output, event to wait on before reading it). Under a multi-stream (CUDA-graph)
        forward it runs on the side stream, overlapping the routed experts; else inline (event None). From
        `FP8_1X128_MIN_ROWS` rows the shared expert reads the FP8 1x128 copy of `x` stock's FFN boundary makes for it
        (`nvfp4_fp8_quantize`, as stock's `fused_hc_lagged` with `fp8_1x128`; the routed experts quantize their NVFP4
        rows themselves, bitwise the same as that launch's)."""
        fp8 = None
        if (
            x.shape[0] >= FP8_1X128_MIN_ROWS
            and x.shape[1] % 512 == 0
            and nvfp4_fp8_quantize.ready(x.shape[1])
        ):
            fp8 = nvfp4_fp8_quantize(x.contiguous(), self.moe.g1)[2:]
        if self._shared_stream is None or not do_multi_stream():
            return self.shared(x, fp8), None
        fork, done = self._events["shared"]
        fork.record()
        with torch.cuda.stream(self._shared_stream):
            fork.wait()
            out = self.shared(x, fp8)
            done.record()
        return out, done

    def _ffn(self, x: torch.Tensor, all_rank_num_tokens, unfinalized: bool, xq=None):
        """This rank's partial sum of the routed and shared experts (both are split over the tensor-parallel
        group), reduced by the next boundary; with `unfinalized` the routed rows are left uncombined."""
        shared, done = self._shared_overlapped(x)
        if unfinalized:
            fc2, scales, perm = self.moe.experts_unfinalized(x, xq)
            routed = None
        else:
            routed = self.moe(x, all_rank_num_tokens)
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
        if unfinalized:
            n = x.shape[0]
            return moe_unfinalized(shared, fc2, scales.view(n, -1), perm.view(n, -1))
        return torch.add(routed, shared)

    def _split(self, n: int) -> bool:
        """Decode-sized rows under a multi-stream (CUDA-graph) forward: the boundary's critical half runs in one
        launch and its coefficients on the side stream, overlapping the sub-layer."""
        return (
            self._coeff_stream is not None
            and do_multi_stream()
            and hc_split_enabled(n, self.dim, self.hc)
        )

    def _coefficients(self, which: str, residual: torch.Tensor, release_after_wait: bool = False):
        """The boundary's own (pre, post, comb) on the side stream -> (pre, post, comb, event to wait on). Up to 8 rows
        in one CUDA-core launch (the built-in model's), which with `release_after_wait` lets its dependents launch from
        its dependency wait on; else the mixer GEMM and bigfuse."""
        scale, base = getattr(self, f"hc_{which}_scale"), getattr(self, f"hc_{which}_base")
        tail = (self.hc, self.sinkhorn, self.norm_eps, self.hc_eps, _HC_POST_MULT)
        fma = self._fma is not None and hc_fma_fits(residual.shape[0], self.dim, self.hc)
        fork, done = self._events[which]
        fork.record()
        with torch.cuda.stream(self._coeff_stream):
            fork.wait()
            if fma:
                weight, counter = self._fma[which]
                pre, post, comb = hc_coefficients_fma(
                    residual,
                    weight,
                    counter,
                    scale,
                    base,
                    *tail,
                    release_after_wait=release_after_wait,
                )
            else:
                fn = getattr(self, f"hc_{which}_fn")
                pre, post, comb = hc_coefficients(residual, fn, scale, base, *tail)
            done.record()
        return pre, post, comb, done

    def _boundary(self, which, x_prev, residual, post_prev, comb_prev, pre_mix, mxfp8=None):
        """One lagged boundary of the attention (`which = "attn"`) or FFN sub-layer.

        -> (residual with the previous post update applied, own pre, own post, own comb, normalized input).
        The single-pass kernel when this many rows are in its range (and compiled, under graph capture), DeepGEMM's
        one-kernel boundary from 1024 rows with a post update (`hc_mega_boundary`), else the post mapping and the
        two-launch pre-mapping. With `mxfp8` given (True or False) a sixth element: the
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
            # 1024+ rows: DeepGEMM's one-launch boundary, as stock's fused lagged boundary runs it past its single-pass
            # kernel's 1023 rows (`fused_hc_lagged` -> `mhc_fused_hc` -> `mega_mhc`), not the post mapping and the
            # two-launch pre-mapping. The same launch writes the q_a | kv GEMM's MXFP8 input (stock's `prequantized`).
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
        self, which, x_prev, partial, residual, post_prev, comb_prev, pre_mix, nvfp4_scale=None
    ):
        """Critical half of a split boundary -> (residual, normalized input). `partial`: `x_prev` still has to be
        summed over the group, which the kernel does in the same launch when the workspace is there. With `nvfp4_scale`
        (the FFN boundary ahead of the routed experts, as stock's) a third element: the normalized input's NVFP4 copy
        `(fp4, scales)` from the same launch, or None where that launch does not run."""
        norm = self.attn_norm if which == "attn" else self.ffn_norm
        if x_prev is None:
            return hc_post_collapse_norm(None, residual, None, None, pre_mix, norm, self.norm_eps)
        if partial and self.lamport is not None:
            return hc_allreduce_post_collapse_norm(
                x_prev,
                residual,
                post_prev,
                comb_prev,
                pre_mix,
                norm,
                self.norm_eps,
                self.lamport,
                nvfp4_scale=nvfp4_scale,
            )
        out = hc_post_collapse_norm(
            x_prev, residual, post_prev, comb_prev, pre_mix, norm, self.norm_eps
        )
        return out if nvfp4_scale is None else (*out, None)

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

    def publish_global(self, residual: torch.Tensor, pre_mix: torch.Tensor, ctx) -> None:
        """Bounded-replay boundary (the last unpooled KV owner, stock's `prepare_ced_global_kv`): publish the compressed
        KV and index-key rows of every row of the step from the resolved stream, normalized by the layer's own attention
        boundary, before the rows are narrowed to the final window."""
        h = self._boundary("attn", None, residual, None, None, pre_mix)[4]
        self.attn.publish_global(h, ctx)

    def idle(self, counts) -> None:
        """A rank with no row in the decoder half still joins the experts' collectives of its attention-DP peers (stock
        skips the attention and the mHC of an empty decoder input)."""
        self.moe(self.attn_norm.new_empty(0, self.dim), counts)

    def forward(
        self,
        st: _HcState,
        ctx,
        attn_metadata,
        engram: "_EngramStep",
        spec_metadata=None,
        capture_rows=None,
    ) -> _HcState:
        """`st` -> the state after this layer (deferred when the next boundary can take the FFN's post update).
        `capture_rows`: in the decoder half of a bounded replay, the step rows the state's rows are (the draft's
        captures go to those rows)."""
        st.wait_coefficients()
        residual, pre_mix = st.residual, st.pre_mix
        split = self._split(residual.shape[0])
        lamport = self.lamport if split else None
        deferred = st.deferred
        if self.engram is not None:
            if deferred:  # the engram reads and rewrites the materialized stream
                if lamport is not None and st.allreduce is not None:
                    # all-reduce + post update in one launch (the collapse it also returns is not needed here)
                    residual = hc_allreduce_post_collapse_norm(
                        st.x_prev,
                        residual,
                        st.post,
                        st.comb,
                        pre_mix,
                        self.attn_norm,
                        self.norm_eps,
                        lamport,
                    )[0]
                else:
                    residual = hc_post_mapping(st.reduced_x_prev(), residual, st.post, st.comb)
                deferred = False
            kv_shard = engram.kv.get(self.layer_idx)
            if (
                kv_shard is not None
            ):  # looked up and projected ahead of this layer (`ModelingV2Core._run_layers`)
                residual = self.engram.forward_precomputed(
                    residual, kv_shard, engram.gather, self.engram.events[1]
                )
            else:
                residual = self.engram(
                    residual,
                    engram.ids[:, self.engram.hash_index],
                    attn_metadata.all_rank_num_tokens,
                )
        a_event = None
        if split:
            partial = deferred and st.allreduce is not None
            x_prev = (
                (st.x_prev if partial and lamport is not None else st.reduced_x_prev())
                if deferred
                else None
            )
            residual, h = self._split_boundary(
                "attn", x_prev, partial, residual, st.post, st.comb, pre_mix
            )
            # The split FFN boundary reads these after its dependency wait: their launch releases it from its own wait.
            a_pre, a_post, a_comb, a_event = self._coefficients(
                "attn", residual, release_after_wait=True
            )
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
        x_attn = self.attn(h, ctx, None if split else h_mx)  # this rank's partial sum
        if capture_done is not None:
            torch.cuda.current_stream().wait_event(capture_done)
        f_event = None
        if split:
            torch.cuda.current_stream().wait_event(a_event)
            if lamport is None:
                x_attn = self._reduce(x_attn)
            # the routed experts' NVFP4 rows come from the boundary's launch where the routing then only routes (stock)
            scale = (
                self.moe.g1
                if lamport is not None and self.moe.boundary_quantizes(residual.shape[0])
                else None
            )
            residual, h, *xq = self._split_boundary(
                "ffn", x_attn, lamport is not None, residual, a_post, a_comb, a_pre, scale
            )
            xq = xq[0] if xq else None
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
        x = self._ffn(h, attn_metadata.all_rank_num_tokens, unfinalized, xq if split else None)
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
    # The decode prologue's mark (`mark_prologue_start`): none until the engine marks a captured decode step.
    _prologue_pending = False
    _prologue_start: Optional[torch.cuda.Event] = None
    _prologue_event: Optional[torch.cuda.Event] = None

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        csa2_mxfp8_gemm_tune_rows(
            csa2_mxfp8_graph_rows(model_config)
        )  # CUDA-graph step rows: MXFP8 GEMM tactics tuned at warm-up
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
        # The shared expert's stream runs at the decode graphs' priority, as stock's DeepseekV4ForCausalLM creates it
        # (the next boundary waits for its output); the other auxiliary streams stay at the default priority, as there
        self.aux_stream_dict[AuxStreamType.MoeShared] = torch.cuda.Stream(
            priority=decode_stream_priority(model_config.max_num_tokens)
        )
        self.aux_stream_dict["csa2_index_weights"] = (
            torch.cuda.Stream()
        )  # the engine's second CSA2 side stream
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
        self.engram_hash: Optional[EngramHash] = None
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
        self._fwd_stream: Optional[torch.cuda.Stream] = (
            None  # the stream the forward runs on (set by the first forward)
        )
        self._step_inputs = (
            torch.cuda.Event()
        )  # the step's inputs (ids, positions) are ready: the Engram stream forks here
        self.ll_gather = None  # see `post_load`
        # A captured decode step's mark (`mark_prologue_start`, as the built-in DeepseekV41Model): the n-gram hash with
        # the Engram lookups, and the embedding with the stacked residual on `_prologue_stream`, wait only for it --
        # beside the engine's remaining input corrections. The model stream joins the prologue stream before layer 0:
        # it runs at the decode priority.
        self._prologue_stream = torch.cuda.Stream(
            priority=decode_stream_priority(model_config.max_num_tokens)
        )
        self._prologue_event = torch.cuda.Event()
        self._prologue_done = torch.cuda.Event()
        self._prologue_pending = False
        self._prologue_start: Optional[torch.cuda.Event] = None
        self._pre_mix0: Optional[torch.Tensor] = (
            None  # [_UNIFORM_ROWS, hc] fp32, column 0 ones: layer 0's `pre_mix`
        )
        # {rows per request: (request row, offset in the request)} of up to `_UNIFORM_ROWS` rows
        self._uniform_rows = None
        self._prefill = PrefillTarget()
        self._decode = DecodeTarget()
        # Bounded decoder replay (stock's policy, `csa2_replay_policy`). The executor reads these names to decide
        # whether the cache manager's private decoder pages and the replay plan apply to this model.
        self.decoder_replay_split, self.decoder_replay_window = csa2_replay_policy(
            cfg, model_config, captures=True
        )
        self.ced_kv_precompute = self.decoder_replay_split is not None
        self.encoder_replay_enabled = csa2_encoder_replay_enabled(
            model_config.max_num_tokens, self.decoder_replay_window, self.ced_kv_precompute
        )

    # ---- weights: see weights.py ---------------------------------------------------
    def load(self, weights: dict) -> None:
        _weights.load(self, weights)

    def post_load(self) -> None:
        # The Lamport workspace of the TP group (one-shot all-reduce inside the boundary kernels): looked up now, not
        # during a CUDA-graph capture; collective over the group.
        lamport = hc_lamport_workspace(self.mapping, self.model_config.allreduce_strategy)
        # The peer-memory all-gather of the Engram shards at decode sizes
        # (None: NCCL / all-reduce path only); collective.
        self.ll_gather = ll_allgather_workspace(self.mapping) if self.mapping.tp_size > 1 else None
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
        iota = torch.arange(_UNIFORM_ROWS, device=dev)
        self._uniform_rows = {
            L: (iota // L, (iota % L).to(torch.int32)) for L in range(1, _UNIFORM_MAX_L + 1)
        }
        self._pre_mix0 = torch.zeros(_UNIFORM_ROWS, self.hc, dtype=torch.float32, device=dev)
        self._pre_mix0[:, 0] = 1.0  # read only: every boundary writes its `pre` to a new tensor

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

    def mark_prologue_start(self, attn_metadata) -> None:
        """Engine hook (`DecoderRunner`), called behind a step's first input correction (the position correction):
        under the capture of a decode-only, attention-TP step, record where its Engram hash and lookups, embedding and
        stacked residual start (the built-in `DeepseekV41Model.mark_prologue_start`). They read only graph inputs (input
        ids, corrected position ids) and state the previous step left (the look-back ring, the request slots `on_step`
        wrote before the replay)."""
        self._prologue_pending = False
        if (
            attn_metadata is not None
            and attn_metadata.num_contexts == 0
            and torch.cuda.is_current_stream_capturing()
            and not is_in_breakable_cuda_graph()
        ):
            self._prologue_event.record()
            self._prologue_pending = True

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        self._prologue_start = self._prologue_event if self._prologue_pending else None
        self._prologue_pending = False
        try:
            return self._forward_step(
                attn_metadata, input_ids, position_ids, inputs_embeds, lora_params, **kwargs
            )
        finally:
            self._prologue_start = None

    def _forward_step(
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
        # Encoder recovery (stock's CED replay, as the dep2 / dep targets): with block reuse, a context request
        # whose Encoder pages were not kept recomputes its last Encoder rows before its chunk (`EncoderReplay`). Its
        # historical rows rebuild the Encoder working state only: they write no GLOBAL / INDEX rows (the replay's Global
        # claim holds them), and the request's Encoder SWA reads start at its recovery floor (`_step_ctx`). The plan is
        # consumed only after the forward succeeds, as stock's `DeepseekV41ForCausalLM.forward` does.
        requests = kwargs.get("context_requests")
        self._recovery = csa2_encoder_recovery(requests, len(attn_metadata.seq_lens))
        if self._recovery is not None and not self._recovery_logged and any(self._recovery[0]):
            self._recovery_logged = True
            hist = sum(csa2_encoder_replay_tokens(r) for r in requests)
            logger.info(
                f"DeepSeek-V4.1 tep8 Encoder recovery engaged: {hist} historical rows recomputed in this step"
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
        n = input_ids.shape[0]
        positions = _step_positions(attn_metadata, position_ids, n)
        self._step_inputs.record()  # ahead of the step's bookkeeping launches (the hash needs none of them)
        seq_lens = attn_metadata.seq_lens
        seq_lens = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        nreq = len(seq_lens)
        dev = input_ids.device
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            self._fwd_stream = torch.cuda.current_stream()
        uniform = len(set(seq_lens)) == 1
        const_rows = False
        if uniform:  # uniform (generation, CUDA graph): no host data baked in
            L = seq_lens[0]
            if n <= _UNIFORM_ROWS and L <= _UNIFORM_MAX_L and self._uniform_rows is not None:
                # constants of the step's shape, built at load: views, no launch in the step (the engine prepares its
                # per-token bookkeeping outside the captured graph)
                tok_req, idx_in = (t[:n] for t in self._uniform_rows[L])
                const_rows = True
            else:
                tok_req = torch.arange(n, device=dev) // L
                idx_in = (torch.arange(n, device=dev) % L).to(positions.dtype)
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
        # A marked step (`mark_prologue_start`) hashes from the corrected position ids, as the built-in hash provider
        # does, at the mark: the CSA2 metadata's live positions (`positions`) are refreshed after it. Both hold the same
        # positions on every request row; CUDA-graph padding rows hash into the dummy slot either way.
        hash_positions = positions
        if (
            self._prologue_start is not None
            and const_rows
            and position_ids is not None
            and position_ids.dtype == torch.int32
        ):
            hash_positions = position_ids.reshape(-1)[:n]
        step = _Step(
            n=n, nreq=nreq, positions=positions, tok_req=tok_req, cs=cs, ce=ce, bounds=bounds,
            uniform=uniform, capturing=capturing, const_rows=const_rows, hash_positions=hash_positions,
        )  # fmt: skip
        return self._select_target(step, attn_metadata).forward(
            self, step, attn_metadata, input_ids, **kwargs
        )

    def _run_layers(
        self, step: "_Step", ctx: StepCtx, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        """Embedding, the 40 Hyper-Connection layers and the final norm; `ctx` fixes how the
        indexer reads its keys (exact span per request, or device-side lengths)."""
        n = step.n
        x, pre_mix, prologue_done = self._prologue(step, input_ids)
        engram = _EngramStep(None, {}, self.ll_gather)
        projections = self._engram_launch(engram, step, input_ids)
        if engram.ids is None:  # the Engram layers look up in place: hashed here
            engram.ids = self._engram_hash(step, input_ids)
        if (
            prologue_done is not None
        ):  # join the prologue stream (the built-in `_PrologueFork.join`)
            cur = torch.cuda.current_stream()
            cur.wait_event(prologue_done)
            x.record_stream(cur)
            # The join's edges must land on a plain launch: a PDL kernel (layer 0's boundary) would make them
            # programmatic.
            x.new_zeros(1)
        spec_metadata = kwargs.get("spec_metadata")
        if hasattr(spec_metadata, "context_capture_lens"):
            spec_metadata.context_capture_lens = None
        requests = kwargs.get("context_requests")
        if not (
            self.encoder_replay_enabled
            or self.mapping.enable_attention_dp
            or csa2_uses_private_decoder_cache(self, attn_metadata.kv_cache_manager)
        ):
            requests = None  # (stock's forward hands its planner no requests then)
        plan = csa2_step_replay_plan(
            self, attn_metadata, bool(kwargs.get("all_token_states_required", True)), requests
        )
        st = _HcState(x, pre_mix)
        rows = None  # the step rows the decoder half runs, when the replay compacts them
        saved_counts = attn_metadata.all_rank_num_tokens
        try:
            for i, layer in enumerate(self.layers):
                if plan is not None and i == self.decoder_replay_split:
                    st, ctx, rows = self._enter_decoder(layer, st, ctx, plan, attn_metadata)
                    if (
                        rows is not None and rows.numel() == 0
                    ):  # no row of this rank runs the decoder half
                        counts = attn_metadata.all_rank_num_tokens
                        if counts is not None and sum(
                            int(c) for c in counts
                        ):  # (attention DP: a peer has rows)
                            for idle in self.layers[i:]:
                                idle.idle(counts)
                        out = x.new_zeros(n, x.shape[-1])  # (x: the embedded rows, `[n, hc, dim]`)
                        break
                for project in projections.get(i, ()):
                    project()
                st = layer(st, ctx, attn_metadata, engram, spec_metadata, rows)
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
        h = hc_pre(x, st.pre_mix)
        return flashinfer_rmsnorm(h, self.norm, self.norm_eps)

    def _enter_decoder(self, layer, st: _HcState, ctx: StepCtx, plan, attn_metadata):
        """The bounded-replay boundary, stock's `_enter_bounded_replay`: the decoder row counts (under attention DP,
        every rank's); where the rows or window floors change, the stream is resolved, the owner layer's compressed rows
        of every row are published, only the plan's rows are kept (the last `window` rows of each prompt chunk, every
        generation row) and the plan's per-request SWA floors apply. -> (state, ctx, kept rows or None)"""
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
        floors = torch.zeros(ctx.nreq, dtype=torch.int64)
        floors[: len(plan.swa_floors)] = torch.tensor(plan.swa_floors, dtype=torch.int64)
        floors = maybe_pin_memory(floors).to(dev, non_blocking=True)
        if (
            not plan.replays_local_tokens
        ):  # every row is kept; the floors still hide window rows of earlier chunks
            return (
                _HcState(x, pre_mix),
                dataclasses.replace(ctx, published=True, floor=floors),
                None,
            )
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
        dctx = dataclasses.replace(
            ctx, pos=pos, rid=rid, cs=pos.index_select(0, first), ce=ce, segs=segs, topk=None, cands=None,
            published=True, floor=floors,
        )  # fmt: skip
        return _HcState(x, pre_mix), dctx, rows

    def _prologue(self, step: "_Step", input_ids: torch.Tensor):
        """The step's embedding and stacked residual `[n, hc, d]`, and layer 0's `pre_mix` `[n, hc]`. A marked uniform
        step (`hash_positions` is not `positions`) launches the embedding and the stack on the prologue stream from the
        mark (the built-in `_PrologueFork`) and returns the done event the model stream joins before layer 0;
        `pre_mix` is a load-time constant at decode sizes."""
        n = step.n
        pre_mix = self._pre_mix0[:n] if self._pre_mix0 is not None and n <= _UNIFORM_ROWS else None
        if pre_mix is None:
            pre_mix = torch.zeros(n, self.hc, dtype=torch.float32, device=input_ids.device)
            pre_mix[:, 0] = 1.0
        if step.hash_positions is step.positions:
            h = token_embedding(self.embed_tokens, input_ids)
            return h.unsqueeze(1).repeat(1, self.hc, 1), pre_mix, None
        stream = self._prologue_stream
        stream.wait_event(self._prologue_start)
        with torch.cuda.stream(stream):
            h = token_embedding(self.embed_tokens, input_ids)
            x = h.unsqueeze(1).repeat(1, self.hc, 1)
            self._prologue_done.record()
        return x, pre_mix, self._prologue_done

    def _engram_hash(self, step: "_Step", input_ids: torch.Tensor) -> torch.Tensor:
        """-> the step's bucket-local hash ids `[n, engram layers, cols]` (and the look-back ring updated)."""
        eh = self.engram_hash
        if step.n <= ENGRAM_HASH_MAX_ROWS:  # one launch (decode sizes)
            return engram_lookback_hash(
                input_ids.contiguous(),
                step.hash_positions,
                step.tok_req,
                self.req_slot,
                eh.token_map,
                self.hist,
                eh.mults,
                eh.primes,
                eh.pad_id,
                True,  # layer-major: each Engram layer's ids contiguous for its lookup
            )
        return eh.lookback(
            input_ids, step.hash_positions, self.req_slot[step.tok_req].long(), self.hist
        )

    def _engram_launch(self, engram: "_EngramStep", step: "_Step", input_ids: torch.Tensor) -> dict:
        """Decode sizes, as the engine's decode graph: the n-gram hash and both Engram layers' table lookups run now on
        the Engram stream (the engine hashes ahead of the graph; here the hash leaves the main stream's chain to the
        embedding); each projection (gather the head shards, project this rank's WKV columns) is issued ahead of its
        layer (`_ENGRAM_PROJECT_AT`: both layers' at once, as the built-in sm_103 schedule) and fills `engram.kv`
        (layer -> `(tensor, gathered)`). -> {loop index: [issue the projections]}; empty where a
        step is too wide for the peer-memory gathers (the Engram layers then look up and project in place)."""
        layers = [layer for layer in self.layers if layer.engram is not None]
        if not layers or not all(
            layer.engram.precomputes(step.n, engram.gather) for layer in layers
        ):
            return {}
        stream = (
            self.aux_stream_dict[AuxStreamType.EngramPrecompute]
            if do_multi_stream()
            else torch.cuda.current_stream()
        )
        if step.hash_positions is not step.positions:  # a marked step: fork at the mark
            stream.wait_event(self._prologue_start)
        elif (
            step.const_rows
        ):  # the request rows are load-time constants: fork right after the step's inputs
            stream.wait_event(self._step_inputs)
        else:  # the request rows were uploaded since
            stream.wait_stream(torch.cuda.current_stream())
        for t in (input_ids, step.hash_positions, step.tok_req):
            t.record_stream(stream)
        with torch.cuda.stream(stream):
            engram.ids = self._engram_hash(step, input_ids)
            ids = [engram.ids[:, layer.engram.hash_index].contiguous() for layer in layers]
        looked = []
        for layer, layer_ids in zip(layers, ids):
            e = layer.engram
            looked.append(
                (layer.layer_idx, e, e.lookup_async(layer_ids, stream, e.events[0], ordered=True))
            )
        n = step.n

        def project():
            # As the built-in sm_103 decode graph (`_precompute_engram_kv_pair`): one peer-memory gather of both
            # head-shard lookups on the model stream, both WKV projections on the Engram stream, then one gather of
            # both WKV shards on the Engram stream with its grid capped (it spins beside the model stream's
            # all-reduce boundaries, which need free SMs on every rank); where a pair does not fit one shard slot,
            # each layer gathers its own (its WKV shard on the model stream, at its layer).
            cur = torch.cuda.current_stream()
            for _, e, _ in looked:
                cur.wait_event(e.events[0])
            embs = [emb for _, _, emb in looked]
            gather = engram.gather
            if len(embs) == 2 and ll_allgather_pair_takes(
                gather, n, embs[0].shape[1], embs[0].dtype
            ):
                gathered = engram_ll_allgather_pair(gather, embs[0], embs[1])
            else:
                gathered = [engram_ll_allgather(gather, emb) for emb in embs]
            shards = [
                e.project_async(g, stream, e.events[1]) for (_, e, _), g in zip(looked, gathered)
            ]
            if len(shards) == 2 and ll_allgather_pair_takes(
                gather, n, shards[0].shape[1], shards[0].dtype
            ):
                with torch.cuda.stream(stream):  # behind both projections
                    whole = engram_ll_allgather_pair(
                        gather, shards[0], shards[1], max_programs=ENGRAM_SIDE_MAX_PROGRAMS
                    )
                    for _, e, _ in looked:
                        e.events[1].record()
                for (idx, _, _), kv in zip(looked, whole):
                    kv.record_stream(cur)
                    engram.kv[idx] = (kv, True)
            else:
                for (idx, _, _), kv in zip(looked, shards):
                    engram.kv[idx] = (kv, False)

        return {_ENGRAM_PROJECT_AT: [project]}

    def _step_ctx(self, step: "_Step", attn_metadata, segs, paged: bool) -> StepCtx:
        # a context step's Encoder recovery: per request its Global end and the floor of its SWA reads
        hist_end = floor = None
        if not paged and self._recovery is not None:
            hist_end, floor = (
                maybe_pin_memory(torch.tensor(v, dtype=torch.int64)).to(
                    step.positions.device, non_blocking=True
                )
                for v in self._recovery
            )
        swa = engine_swa_tables(attn_metadata, self.cfg.num_hidden_layers)
        return StepCtx(
            pos=step.positions, rid=step.tok_req, cs=step.cs, ce=step.ce, segs=segs,
            kvm=KvPools(attn_metadata.kv_cache_manager, swa), rope=self.rope,
            nreq=step.nreq, paged=paged, uniform=step.uniform,
            rows=None if step.uniform or not paged else tuple(b - a for a, b in step.bounds),
            floor=floor, hist_end=hist_end,
        )  # fmt: skip


@dataclass
class _EngramStep:
    """The step's Engram inputs: `ids [n, engram layers, cols]` bucket-local hash ids, the WKV shards projected ahead
    of their layers (`kv`: layer index -> `[n, out / tp]`), and the peer-memory gather of the group (or None)."""

    ids: torch.Tensor
    kv: dict
    gather: object


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
    const_rows: bool = (
        False  # `tok_req` is a view of load-time constants (no launch of this step produced it)
    )
    # [n] int32 positions the Engram hash reads: `positions`, or a marked step's corrected position ids
    hash_positions: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.hash_positions is None:
            self.hash_positions = self.positions


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
        return core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)


class DecodeTarget:
    """Generation-only steps: lengths stay on the device (CUDA-graph safe), paged indexer kernel."""

    def forward(
        self, core: ModelingV2Core, step: _Step, attn_metadata, input_ids, **kwargs
    ) -> torch.Tensor:
        ctx = core._step_ctx(step, attn_metadata, segs=[], paged=True)
        return core._run_layers(step, ctx, attn_metadata, input_ids, **kwargs)


class ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2(
    SpecDecOneEngineForCausalLM[ModelingV2Core, PretrainedConfig]
):
    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The built-in model's step-loop hooks (`step_hooks`), the speculative set when DSpark runs."""
        return step_hooks(getattr(self, "spec_config", None))

    def __init__(self, model_config: ModelConfig):
        model_config = text_model_config(model_config)
        # what routing declines (`limits.tep8_refused`), for a construction routing did not see: `auto` then falls back
        refused = tep8_refused(model_config)
        if refused:
            raise NoTarget(f"the deepseek_v41 tep8 target does not implement: {'; '.join(refused)}")
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
        """Engine hook, before the attention-DP token-count exchange (or a breakable prefill graph's padding choice):
        fix which rows of each context chunk run the decoder half (stock's `prepare_adp_inputs`)."""
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
