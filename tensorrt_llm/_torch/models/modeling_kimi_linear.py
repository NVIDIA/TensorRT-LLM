# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiLinearForCausalLM — Kimi K3 text model, PyTorch backend.

Runtime integration of the Kimi K3 hybrid architecture for the standard
TRT-LLM PyTorch-backend flow (``LLM(model=<ckpt>) -> generate``):

* 93 decoder layers: 69 KDA (Kimi Delta Attention, linear attention) layers
  and 24 MLA (absorbed-MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 uses a dense SiTU MLP; layers 1..92 use the 896-expert latent MoE
  (top-16, sigmoid + e_score_correction_bias routing, MXFP4 routed experts,
  2 shared experts, latent down/norm/up projections).
* The attention-residual ("attn_res") scheme from the HF reference
  ``modeling_kimi.py`` is applied per token: snapshot mixing before
  ``input_layernorm`` / ``post_attention_layernorm`` and at the model output,
  with a new snapshot appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching
-------
KDA states use ``KimiK3HybridCacheManager`` with the ``KvCacheConfig``
default ``use_kv_cache_manager_v2=True`` (selected in ``pyexecutor/_util.py``).
Setting this field to ``"auto"`` falls back to V1 for
``KimiLinearForCausalLM``, which declares no V2 model preference.
Per layer, the state consists of a short-conv slot of
``[3 * num_heads * head_dim, W]`` bf16 (the full FLA ``ShortConvolution``
cache window, sections ``[q | k | v]``) and a delta-rule recurrent slot of
``[num_heads, head_dim, head_dim]`` fp32 (``[H, V, K]``, the
``state_v_first`` FLA layout). MLA layers use the paged-KV side with
``num_kv_heads=1`` and ``head_dim = kv_lora_rank + qk_rope_head_dim`` (576),
SELFKONLY, exactly like DeepSeek MLA.

MLA prefill routing
-------------------
The in-tree ``KimiK3MLAAttention`` routes prefill through the normal
unabsorbed MLA context FMHA. It consumes the executor's original mixed-batch
metadata, letting the shared MLA implementation dispatch context and cached
generation work in one forward call.

Parallelism
-----------
The routed-expert bank supports a configurable MoE TP x EP split
(``moe_tp_size * moe_ep_size == mapping.tp_size``); the default is EP-only
(``moe_ep_size == mapping.tp_size``), the historical K3 layout. Under EP each
MoE layer holds a contiguous ``num_experts / moe_ep_size`` slice of the MXFP4
expert bank (whole experts); under MoE TP each rank holds ALL experts, with
w1/w3 column-sharded and w2 row-sharded along the intermediate dim
(``intermediate / moe_tp_size`` per rank; group-32 MXFP4 packed bytes and
scales sliced consistently by the stock TRTLLM-Gen quant-method loaders).
The split is EP-only unless the user sets ``moe_tensor_parallel_size`` /
``moe_expert_parallel_size`` explicitly. Routing is computed replicated; the
routed partial sums — EP partials of whole experts, or TP partials over the
intermediate shards — are all-reduced in the latent space (before
``routed_expert_norm`` / ``routed_expert_up_proj``, which are nonlinear/linear
layers applied to the full sum). When attention DP is off, the shared experts
use standard MLP TP over the model TP group: gate/up are column-sharded and down
is row-sharded. The shared down projection reduces on the auxiliary stream
while the routed expert chain runs on the main stream. After the streams join,
the routed latent partial is reduced before its norm/up projection. With
``TRTLLM_K3_MERGED_MOE_AR=1`` the shared expert leaves its partial unreduced and
one all-reduce over ``[latent partial | shared partial]`` serves both.
Fused-communication routed backends already return a complete routed result and
need no outer reduction. Under attention DP the shared experts stay replicated.
``lm_head`` uses the stock ``LMHead`` (vocab-sharded + gather), so logits are
identical on all ranks.

Speculative decoding: SA (suffix automaton, one-engine, draft-weight-free);
the KDA/MLA runtimes implement multi-token verification with deferred
state promotion.

Chunked prefill is supported: continuation chunks feed the previous KDA
conv/recurrent state back into the FLA kernels (``use_initial_states``)
and the MLA prefill path natively attends over the cached latent prefix
(``kv_len = cached + q_len``). KV-cache block reuse is supported as an
opt-in via ``kv_cache_config.enable_block_reuse=true`` on the default V2
manager. Configuring ``mamba_state_config.periodic_snapshot_interval``
adds periodic KDA state snapshots; reuse uses FORCE_CHUNK context chunking.

Not supported: pipeline parallelism, draft-head spec-dec modes
(MTP/Eagle — no draft-head checkpoint exists). SA speculative decoding
is validated only without block reuse.
"""

from __future__ import annotations

import copy
import functools
import gc
import json
import math
import os
import threading
import time
import weakref
from contextlib import ExitStack
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from safetensors import safe_open
from torch import nn

from ..._utils import get_sm_version, is_sm_100f
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention.attention import _helix_cp_pad
from ..attention.backends import AttentionMetadata
from ..attention.backends.sparse.dsa.params import disagg_role
from ..configs.kimi_k3 import KIMI_K3_COMPOSITE_MODEL_TYPES
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    allgather,
    reducescatter,
)
from ..distributed.ops import (
    LamportPending,
    LamportResult,
    bf16_oproj_ar_finish_enabled,
    configure_k31_oproj_all_reduce,
)
from ..distributed.symm_mem_allgather import SymmetricMemoryAllGather
from ..kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV, get_kimi_k3_fp8_kv_layer_ids
from ..model_config import ModelConfig
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_k3_expert_placement import expert_placement_for_layer, permute_router_to_slots
from ..modules.kimi_k3_l2_prefetch import POINTS as L2PF_POINTS
from ..modules.kimi_k3_l2_prefetch import L2PrefetchConfig, L2PrefetchController
from ..modules.kimi_k3_moe_tail import (
    KimiK3LatentMoETail,
    UnfinalizedRoutedOutput,
    fused_tail_unsupported_reason,
)
from ..modules.kimi_k3_router_input import router_input_fp32
from ..modules.kimi_kda import KimiKDALinearAttention
from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..modules.linear import (
    Fp8BlockScaleLowMGemmRange,
    TensorParallelMode,
    fp8_block_scale_ue8m0_gemm,
    load_weight_shard,
)
from ..modules.linear import Linear as TrtllmLinear
from ..modules.low_m_gemm import planned_low_m_linear
from ..modules.low_m_gemm_plans import kimi_k31_decode_plan
from ..modules.multi_stream_utils import do_multi_stream, maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..moe.fused_moe import ConfigurableMoE, SiTuActivation, TRTLLMGenFusedMoE, create_moe
from ..moe.fused_moe.interface import MoESchedulerKind
from ..moe.fused_moe.moe_scheduler import MOE_GRAPH_PADDING_ROWS_ATTR
from ..moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from ..pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ..utils import (
    AuxStreamType,
    Fp4QuantizedTensor,
    MxFp8QuantizedTensor,
    get_model_extra_attrs,
    get_per_request_prefill_cuda_graph_flag,
)
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, run_concurrently

# The decode switches below marked "(K3.1 TEP gen)" / "(K3.1 DEP gen)" are set by ``_configure_k31_decode`` when the
# model is built: on for Kimi K3.1 generation servers with attention TP / attention DP, off for every other model,
# the context servers and aggregated serving.
# Fused MoE residual update (K3.1 DEP gen): the MoE block takes the layer's residual and returns
# ``residual + bf16(routed + shared)`` (``_fused_moe_residual_add``), or a ``_DeferredMoEResidual`` for the next layer's
# mix where the PDL attn_res kernel in mode "2" defers it. The same two BF16 roundings: bitwise.
_K3_FUSE_MOE_RESIDUAL = False

# MoE block with a TP-sharded shared expert and an outer routed reduction: one all-reduce over
# [routed latent partial | shared-expert partial] replaces the latent all-reduce (+RMSNorm) and
# the shared expert's own all-reduce (see ``KimiK3MoERuntime._merged_reduce``).
_K3_MERGED_MOE_AR = os.environ.get("TRTLLM_K3_MERGED_MOE_AR", "0") == "1"
# Largest token count that takes the merged all-reduce (0: no limit). At a few hundred tokens
# the reduction is bandwidth-bound, so merging saves little, and it loses the overlap of the
# shared expert's own all-reduce with the routed experts when that chain is the longer one.
_K3_MERGED_MOE_AR_MAX_TOKENS = int(os.environ.get("TRTLLM_K3_MERGED_MOE_AR_MAX_TOKENS", "128"))

# Zero residual operand of the merged MoE all-reduce, one per device, shared by every layer.
# It only grows outside CUDA graph capture, and a replaced buffer is kept alive because
# graphs captured earlier still read it.
_K3_ZERO_RESIDUALS: Dict[torch.device, List[torch.Tensor]] = {}


def _k3_zero_residual(numel: int, device: torch.device) -> Optional[torch.Tensor]:
    """``numel`` BF16 zeros, or ``None`` if that needs a new buffer during graph capture."""
    buffers = _K3_ZERO_RESIDUALS.setdefault(device, [])
    if not buffers or buffers[-1].numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            return None
        size = max(numel, 2 * buffers[-1].numel()) if buffers else numel
        buffers.append(torch.zeros(size, dtype=torch.bfloat16, device=device))
    return buffers[-1][:numel]


# Packed MoE front (K3.1 TEP gen; BF16 router / latent_down / shared gate_up): the three
# projections of the MoE input -- router logits (FP32), latent_down and the shared expert's gate_up -- run as one
# low-M GEMM over their row-packed weights (kernels/blackwell/tgv_multi_out_gemm: one launch, each row block
# stored as its own contiguous output) instead of three GEMMs that compete for the SMs, up to
# TRTLLM_KIMI_K31_PACKED_MOE_FRONT_MAX_M tokens. Other GEMM kernels than the separate projections' (FP32
# accumulation in another order), so the logits and activations can differ in the last bit.
_K31_PACKED_MOE_FRONT = False
_K31_PACKED_MOE_FRONT_MAX_M = int(os.environ.get("TRTLLM_KIMI_K31_PACKED_MOE_FRONT_MAX_M", "288"))
# Below _K31_PACKED_MOE_FRONT_MIN_M tokens (the fused MoE tail's 192 + 1) the separate projections run, with
# latent_down (+ its MXFP8 quantization) on the latent stream beside the router GEMM and the routing. In GB300
# generation steps (TEP16 MoE TP4 x EP4 and TEP8 MoE TP4 x EP2, DSpark) the latent stream is as fast as or faster
# than the packed front at most batch sizes below this bound, and the packed front is faster above it.
_K31_PACKED_MOE_FRONT_MIN_M = 193
# (largest M, TGV tactic) of the packed front: [896 | 3584 | 768 or 1536] x 7168 measured on GB300 (cold
# weights, CUDA graphs, PDL; mean of back-to-back calls and calls behind a producer). Above 128 rows tactic 24
# runs several token tiles; with the routing sort at TEP16 (56 local experts) the packed front still saves 3.1 /
# 0.9 / 5.1 us per MoE layer at 144 / 224 / 288 rows (front + routing + experts in one CUDA graph, GB300).
_K31_PACKED_MOE_FRONT_TACTICS = ((8, 18), (32, 21), (64, 23), (128, 24), (512, 24))
# Sharded MoE front (K3.1 TEP gen; BF16 router / latent_down, routed experts on the
# TRTLLM-Gen MXFP4 x MXFP8 runner): MoE forwards of 1 .. TRTLLM_KIMI_K31_SHARDED_MOE_FRONT_MAX_M (default and at most
# 32) rows run the front as one kernel per 16 rows (moe/kernels/blackwell/k31_moe_front.py) over the routed
# all-reduce's TP group (GB300 TEP16 microbench per MoE layer, front + routing: 21.3 -> 9.1 us at 8 rows, 22.5 ->
# 9.5 at 16, 26.4 -> 17.1 at 32):
# every rank computes its 3584 / W latent_down columns and 896 / W router logits (FP32 accumulation, BF16 weights),
# multicasts them into an MNNVL workspace, and per token one CTA takes the top-16 of the gathered logits and one
# the MXFP8 quantization of the gathered (BF16-rounded) latent, with the arithmetic of
# kimi_k3_noaux_tc_mxfp8_quant; with KIMI_K31_ROUTER_SORT's runner tile the last launch also writes the runner's
# routing. Each rank reads 1 / W of the two weights instead of all of them; the routed experts take the top-16 and
# the MXFP8 input. Other GEMM summation order than the replicated projections, so logits and latent can differ in
# the last bit.
_K31_SHARDED_MOE_FRONT = False
_K31_SHARDED_MOE_FRONT_MAX_M = int(os.environ.get("TRTLLM_KIMI_K31_SHARDED_MOE_FRONT_MAX_M", "32"))
# Shared-expert part of the sharded MoE front (K3.1 TEP gen, forwards of at most 16 rows): the sharded front also
# computes this rank's shared-expert gate_up rows and their SiTU (its GEMV clusters stream the TP shard of the BF16
# gate_up weight beside the head), so only the shared down projection runs on the shared-expert stream. Without it the
# shared gate_up GEMM, its split-K reduction and the SiTU kernel run there while FC1 holds the SMs, and the MoE tail
# waits for them. Other summation order and SiTU arithmetic than the separate kernels (not bitwise).
_K31_SHARDED_MOE_FRONT_SHARED = False
_K31_SHARDED_MOE_FRONT_SHARED_MAX_M = 16
# Routing sort of the sharded MoE front (K3.1 TEP gen at TP 32): the front also writes the runner's routing; else it
# hands over only the top-16 and the runner computes its routing in a one-CTA launch of its own. With the fused routing
# FC1 becomes resident (PDL) while the front still sorts and holds the SMs the shared expert's gate_up needs, so the
# shared output and the MoE tail come later (TEP16 B1 nsys: tail +5 us per MoE layer, and a shorter step without
# it); TEP32 gains from it.
_K31_SHARDED_MOE_FRONT_SORT = False
# KIMI_K31_ROUTER_SORT (Kimi K3.1 only, routed experts on the TRTLLM-Gen MXFP4 x MXFP8 runner without attention DP or
# a dispatch): the CuTe routing tail (moe/kernels/blackwell/k3_route_tail) replaces the decode routing chain --
# noaux_tc top-16 (deepseek_v3_topk, or kimi_k3_noaux_tc_mxfp8_quant at <= 64 rows), the BF16 weight cast and the
# runner's own routing kernel (routingIndicesClusterKernel) -- with one launch that also writes the runner's
# permutation for the tile of its tuned tactic; the runner then skips its routing. Same experts, order and BF16
# weights (at <= 64 rows rounded once from the FP64 expression like the fused op, above it the cast of noaux_tc's
# FP32 weights), and the separate MXFP8 quantization equals the fused op's, so the MoE output is bitwise unchanged.
# "1" on, "0" off; KIMI_K31_ROUTER_SORT_MAX_M bounds the row count (cluster kernel <= 256, multi-CTA above).
_K31_ROUTER_SORT = os.environ.get("KIMI_K31_ROUTER_SORT", "1")
if _K31_ROUTER_SORT not in ("0", "1"):
    raise ValueError(f"KIMI_K31_ROUTER_SORT must be 0 or 1, got {_K31_ROUTER_SORT!r}")
_K31_ROUTER_SORT_MAX_M = int(os.environ.get("KIMI_K31_ROUTER_SORT_MAX_M", "2048"))
# From _K31_ROUTER_SORT_WIDE_FROM rows on (every row count) the routing uses the multi-CTA kernel instead of the
# one-cluster kernel where k3_route_tail.wide_sort_fits; it has no DSMEM broadcast ahead of a cluster barrier. Same
# routing, bitwise.
_K31_ROUTER_SORT_WIDE_FROM = 1
# KIMI_K31_ROUTER_SORT_QUANT ("1" default with the sort): up to KIMI_K31_ROUTER_SORT_QUANT_MAX_M rows (128: above
# them the eight CTAs of the routing cluster quantize slower than the separate kernel) the same launch also
# MXFP8-quantizes the routed input (bitwise mxfp8_quantize), so the separate quantization kernel does not run.
_K31_ROUTER_SORT_QUANT = os.environ.get("KIMI_K31_ROUTER_SORT_QUANT", "1") == "1"
_K31_ROUTER_SORT_QUANT_MAX_M = int(os.environ.get("KIMI_K31_ROUTER_SORT_QUANT_MAX_M", "128"))
# KIMI_K31_LATENT_STREAM_QUANT ("1" default): when latent_down runs on its own stream (the latent_down stream)
# and the routing sort serves the forward, the latent stream also MXFP8-quantizes the routed input (the backend's
# own quantize_input), so the router -> routing tail chain on the main stream and the latent_down -> quantization
# chain run side by side and FC1 joins both. Same quantization, bitwise.
_K31_LATENT_STREAM_QUANT = os.environ.get("KIMI_K31_LATENT_STREAM_QUANT", "1") == "1"
# Largest row count served by kimi_k3_noaux_tc_mxfp8_quant (BF16 weights rounded from FP64).
_K31_FUSED_ROUTE_QUANT_MAX_M = 64
# KIMI_K31_ROUTER_TAIL_DEP (Kimi K3.1 only, "1" on, "0" off; unset: "0" on attention-DP context servers and "1"
# elsewhere, resolved by ``_configure_k31_decode``): with attention DP and a token dispatch (DEP gens)
# the CuTe routing tail computes the top-16 of each rank's rows above 64 (the fused noaux + MXFP8 quant op keeps
# the smaller batches) instead of noaux_tc_op (deepseek_v3_topk) + the BF16 weight cast: same experts, order and
# weights (the cast of noaux_tc's FP32 weights), so bitwise. GB300 us per launch: 112 rows 4.5 vs 6.8, 288 rows
# 4.8 vs 10.9, 448 rows 5.0 vs 11.8. Only for backends that read BF16 routing scales (TRTLLM-Gen): the others take
# noaux_tc's FP32 weights, which the BF16 tail would not reproduce.
_K31_ROUTER_TAIL_DEP = os.environ.get("KIMI_K31_ROUTER_TAIL_DEP", "1") == "1"


def _k31_takes_bf16_routing_scales(backend) -> bool:
    """Whether a routed-expert backend reads BF16 routing scales (``MoEInputRequirement.routing_scales_dtype``):
    there the scheduler casts noaux_tc's FP32 weights to BF16, which is what the routing tail's BF16 weights are."""
    requirement = getattr(backend, "input_requirement", None)
    return getattr(requirement, "routing_scales_dtype", None) == torch.bfloat16


# Small decode batches route with one fused CuTe launch: router GEMV + noaux_tc top-16
# ("topk"), plus the CuteDSL MoE's moe_sort metadata of this rank ("sort").
_K3_FUSED_ROUTER = os.environ.get("TRTLLM_K3_FUSED_ROUTER", "0")
if _K3_FUSED_ROUTER not in ("0", "topk", "sort"):
    raise ValueError(f"TRTLLM_K3_FUSED_ROUTER must be 0, topk or sort, got {_K3_FUSED_ROUTER!r}")

# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): "1" replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing);
# "gemv" also computes the logits of M <= 32 tokens with a 40-CTA GEMV that runs beside the
# latent down-projection, which then quantizes the routed input right away.
_K3_ROUTE_TAIL = os.environ.get("TRTLLM_K3_ROUTE_TAIL", "0")
if _K3_ROUTE_TAIL not in ("0", "1", "gemv"):
    raise ValueError(f"TRTLLM_K3_ROUTE_TAIL must be 0, 1 or gemv, got {_K3_ROUTE_TAIL!r}")
if _K3_ROUTE_TAIL != "0" and _K3_FUSED_ROUTER != "0":
    raise ValueError("TRTLLM_K3_ROUTE_TAIL and TRTLLM_K3_FUSED_ROUTER are mutually exclusive")
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = int(os.environ.get("TRTLLM_K3_ROUTE_TAIL_MAX_M", "2048"))
# The routing tail shortens the routing chain enough that the persistent routed grid (FC12, or the
# CuteDSL FC1) can start while the shared expert's GEMMs or all-reduce are still pending; these
# then wait until the grid ends (it holds every SM, and FC12 also fills each SM's register file),
# and the main stream joins the shared all-reduce right after it. Where the grid has work for
# every SM for its whole run -- from TRTLLM_K3_ROUTE_TAIL_ORDER_MIN_ROWS expected routed rows per
# rank, tokens x top-k x local experts / experts (EP8: 2 rows per token, EP16: 1) -- the routed
# experts of a tail-routed forward therefore also wait for the shared expert's GEMMs, which fork
# after the router GEMM ("router") or with the latent chain ("fork"), and the latent chain
# quantizes the routed input itself, so the quantization waits for neither. The shared all-reduce
# follows its GEMMs on the shared stream. Same kernels and inputs, so bit-exact. "0" keeps the
# routed experts independent of the shared expert.
_K3_ROUTE_TAIL_ORDER = os.environ.get("TRTLLM_K3_ROUTE_TAIL_ORDER", "router")
if _K3_ROUTE_TAIL_ORDER not in ("0", "router", "fork"):
    raise ValueError(
        f"TRTLLM_K3_ROUTE_TAIL_ORDER must be 0, router or fork, got {_K3_ROUTE_TAIL_ORDER!r}"
    )
_K3_ROUTE_TAIL_ORDER_MIN_ROWS = int(os.environ.get("TRTLLM_K3_ROUTE_TAIL_ORDER_MIN_ROWS", "80"))
# With TRTLLM_K3_ROUTE_TAIL=1: "umma" computes the router logits of up to
# TRTLLM_K3_ROUTER_GEMM_MAX_M tokens as split-K partials of the 35-CTA tcgen05 router GEMM
# (moe/kernels/blackwell/k3_router_umma.py) instead of the cuBLAS GEMM (about 112 CTAs, which
# slow the latent down-projection running beside it) + split-K reduction; the routing tail sums
# the partials. Same stream order as the cuBLAS GEMM (the shared expert forks after it); the
# logits differ from cuBLAS at FP32 rounding level. The default limit of 64 tokens keeps 128-token
# steps (b16) on cuBLAS: there the 35-CTA GEMM and the tail on its partials end later than the
# cuBLAS chain and the ranks' shared-expert chains drift apart (gen-only b16: a longer step).
_K3_ROUTER_GEMM = os.environ.get("TRTLLM_K3_ROUTER_GEMM", "cublas")
if _K3_ROUTER_GEMM not in ("cublas", "umma"):
    raise ValueError(f"TRTLLM_K3_ROUTER_GEMM must be cublas or umma, got {_K3_ROUTER_GEMM!r}")
_K3_ROUTER_GEMM_MAX_M = int(os.environ.get("TRTLLM_K3_ROUTER_GEMM_MAX_M", "64"))
# With TRTLLM_K3_ROUTE_TAIL=gemv: "1" launches only the router GEMV (k3_route_tail.ROUTER_GEMV)
# before the shared expert forks and runs the routing tail on its split-K partials when the routed
# experts ask for the routing, so the shared expert starts right after the GEMV.
_K3_ROUTE_GEMV_DEFER_TAIL = os.environ.get("TRTLLM_K3_ROUTE_GEMV_DEFER_TAIL", "0") == "1"

# Shared expert ahead of the routed FC1 for token counts in [TRTLLM_K3_SHARED_FIRST_MIN_M,
# TRTLLM_K3_SHARED_FIRST_MAX_M]. The persistent routed grids (FC1, FC2) hold every SM while they
# run, and FC1 also fills each SM's register file, so shared-expert kernels that are still
# pending when FC1 starts wait for it (and the down GEMM for FC2 too). With this on, the shared
# gate/up + down GEMMs run first: they fork with the latent chain ("1") or after the router GEMM
# ("router"), FC1 waits for them, and the shared all-reduce waits for FC1, so it runs beside FC2
# (whose CTAs leave room for it). Same kernels and inputs, so bit-exact; needs the latent-chain
# stream and the CuteDSL backend.
_K3_SHARED_FIRST = os.environ.get("TRTLLM_K3_SHARED_FIRST", "0")
if _K3_SHARED_FIRST not in ("0", "1", "router"):
    raise ValueError(f"TRTLLM_K3_SHARED_FIRST must be 0, 1 or router, got {_K3_SHARED_FIRST!r}")
_K3_SHARED_FIRST_MIN_M = int(os.environ.get("TRTLLM_K3_SHARED_FIRST_MIN_M", "257"))
_K3_SHARED_FIRST_MAX_M = int(os.environ.get("TRTLLM_K3_SHARED_FIRST_MAX_M", "2048"))
# The shared expert's all-reduce runs beside the persistent routed GEMMs, which leave no room for
# the per-SM programs of the pipelined NVLS kernel (TRTLLM_NVLS_AR_V2=1), so it keeps the one-shot
# NVLS kernel unless TRTLLM_K3_SHARED_AR_NVLS_V2=1.
_K3_SHARED_AR_NVLS_V2 = os.environ.get("TRTLLM_K3_SHARED_AR_NVLS_V2", "0") == "1"
# Small-batch MoE front end (b1-b4 decode), each bit-identical to the path it replaces:
#   TRTLLM_K3_ROUTER_EXACT=1: router logits of M <= 16 tokens from k3_router_exact (bit-identical
#     to dsv3_router_gemm_op; its weight is copied into shared memory before griddepcontrol.wait).
#     =kda: only in the MoE blocks of KDA layers. The faster router also starts the route tail
#     earlier, inside the burst of latent_down and the shared gate_up; after an MLA attention block
#     the route tail then stretches by ~3 us (b2 gen-only, GB300) and the routing chain gets longer
#     than with dsv3_router_gemm_op, while after a KDA attention block the front gets shorter.
_K3_ROUTER_EXACT_MODE = os.environ.get("TRTLLM_K3_ROUTER_EXACT", "0")
if _K3_ROUTER_EXACT_MODE not in ("0", "1", "kda"):
    raise ValueError(f"TRTLLM_K3_ROUTER_EXACT must be 0, 1 or kda, got {_K3_ROUTER_EXACT_MODE!r}")
# TRTLLM_K3_MOE_L2_PREFETCH: comma list of what the latent-down stream prefetches into L2 once the
# routed FC2 is done (the latent all-reduce, latent_up and the next attention block leave HBM mostly
# idle; k3_l2_prefetch, hints only): "lu" this layer's latent_up weight, "front" the next MoE layer's
# router, latent_down and shared-expert weights. Up to TRTLLM_K3_MOE_L2_PREFETCH_MAX_M tokens. The
# prefetch stays on the latent-down stream, which the next MoE layer joins after its latent_down.
_K3_MOE_L2_PREFETCH = frozenset(
    v for v in os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH", "").split(",") if v and v != "0"
)
if not _K3_MOE_L2_PREFETCH <= {"lu", "front"}:
    raise ValueError(f"TRTLLM_K3_MOE_L2_PREFETCH must list lu / front, got {_K3_MOE_L2_PREFETCH}")
_K3_MOE_L2_PREFETCH_MAX_M = int(os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH_MAX_M", "64"))
# TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA=1: where the next decoder layer is an MLA layer, the MoE block
# also prefetches that layer's q_b projection weight and scale (28 MB) into L2 on the latent-down
# stream once its routed FC2 is done, under the same conditions as TRTLLM_K3_MOE_L2_PREFETCH (up to
# TRTLLM_K3_MOE_L2_PREFETCH_MAX_M tokens, a later MoE layer joins the stream). At decode batch
# sizes most q_b CTAs become resident only when the q_a GEMM's CTAs leave and then stream their
# weight slice from HBM; from L2 that part is shorter. Hints only: bit-exact.
_K3_MOE_L2_PREFETCH_NEXT_MLA = os.environ.get("TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA", "0") == "1"
# MoE runtimes of one model by layer index (keyed by the model's text config), for the prefetch of the
# next MoE layer's weights.
_K3_MOE_LAYERS: Dict[int, "weakref.WeakValueDictionary[int, KimiK3MoERuntime]"] = {}
# MLA runtimes of one model by layer index (keyed like _K3_MOE_LAYERS), for the prefetch of the
# next MLA layer's q_b weight (TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA).
_K3_MLA_LAYERS: Dict[int, "weakref.WeakValueDictionary[int, KimiMLARuntime]"] = {}
# TRTLLM_K3_LATENT_AR_OWN_WS=1: for up to TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed latent
# all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
# one it shares with the shared expert's all-reduce. It then no longer has to follow the shared-expert
# chain: the main stream issues it and latent_up right after the routed FC2 and joins the shared
# stream after latent_up. Applies where latent_up does not add the shared output itself (no fused
# latent_up + residual variant); the output is bit-identical.
_K3_LATENT_AR_OWN_WS = os.environ.get("TRTLLM_K3_LATENT_AR_OWN_WS", "0") == "1"
_K3_LATENT_AR_OWN_WS_MAX_M = int(os.environ.get("TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M", "32"))
# TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES=<bytes> (c240; with TRTLLM_K3_LATENT_AR_OWN_WS; 0: the global
# TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES): the one-shot limit (rows x latent x ranks x 2 B) of the routed
# latent all-reduce on its own workspace, which is sized for it. The other MNNVL all-reduces keep the
# global limit; e.g. 6291456 runs the 96-row latent all-reduce of 8 ranks (5.5 MB) one-shot.
_K3_LATENT_AR_ONESHOT_MAX_BYTES = int(os.environ.get("TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES", "0"))
# TRTLLM_K3_LATENT_AR_CUTE (with TRTLLM_K3_LATENT_AR_OWN_WS): the one-shot sizes of that all-reduce
# run k3_latent_ar_norm, bit-identical to the C++ one-shot kernel but triggering its programmatic
# dependents (latent_up) right after its griddepcontrol.wait instead of after the cross-rank
# reduction ("1"), or after the reduction like the C++ kernel ("late").
_K3_LATENT_AR_CUTE = os.environ.get("TRTLLM_K3_LATENT_AR_CUTE", "0")
if _K3_LATENT_AR_CUTE not in ("0", "1", "late"):
    raise ValueError(f"TRTLLM_K3_LATENT_AR_CUTE must be 0, 1 or late, got {_K3_LATENT_AR_CUTE!r}")
# TRTLLM_K3_LATENT_AR_ROW=1 (with TRTLLM_K3_LATENT_AR_CUTE): the one-shot sizes of the latent
# all-reduce run k3_latent_ar_norm's row kernel (one 512-thread CTA per row instead of a cluster of
# four; the rank's own slot is summed from registers instead of being polled back from the
# multicast loopback; slots that arrived are not reloaded): the same result, bit for bit.
_K3_LATENT_AR_ROW = os.environ.get("TRTLLM_K3_LATENT_AR_ROW", "0") == "1"
# TRTLLM_K3_LATENT_AR_SIDE_CLEAR (with TRTLLM_K3_LATENT_AR_CUTE): the latent-down stream re-arms the
# Lamport buffer the latent all-reduce used (k3_latent_ar_norm.run_k3_lamport_clear) and marks the
# workspace state clean, so the next layer's all-reduce skips the clear (the M x ranks x 7 KB of
# sentinel stores) on its critical path. Only where a later MoE layer exists. The data is unchanged.
#   "1":  once latent_up is issued; the next layer's latent_down follows the clear on that stream
#         and is joined into the main stream before its routed GEMMs. A kernel right before
#         latent_down on that stream turns latent_down's programmatic edge from the attention block
#         into a full one, so latent_down loses its early launch (weight streaming before its
#         griddepcontrol.wait).
#   "ld": in the next MoE layer, right after its latent_down on that stream (which keeps the
#         backend's output memset of the previous layer as its stream predecessor, as without the
#         clear); the clear then reaches the main stream through that layer's output memset, which
#         the CuteDSL backend forks onto the same stream after the routing and joins before FC2,
#         i.e. before the layer's all-reduce. Used only where the backend does so; elsewhere "1".
_K3_LATENT_AR_SIDE_CLEAR = os.environ.get("TRTLLM_K3_LATENT_AR_SIDE_CLEAR", "0")
if _K3_LATENT_AR_SIDE_CLEAR not in ("0", "1", "ld"):
    raise ValueError(
        f"TRTLLM_K3_LATENT_AR_SIDE_CLEAR must be 0, 1 or ld, got {_K3_LATENT_AR_SIDE_CLEAR!r}"
    )
# The routed latent all-reduce's own MNNVL workspaces, by all-reduce mapping.
_K3_LATENT_AR_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


def _k3_latent_ar_workspace(
    mapping: Mapping, dtype: torch.dtype, hidden_size: int, max_tokens: int
) -> Dict[str, Any]:
    """MNNVL Lamport workspace of the routed latent all-reduce alone (three buffers and the state
    flags, laid out as ``get_or_scale_allreduce_mnnvl_workspace`` lays out the shared one), sized
    for up to ``max_tokens`` rows of ``hidden_size``. Collective over the all-reduce group: every
    rank builds it at the same point of model construction."""
    workspace = _K3_LATENT_AR_WORKSPACES.get(mapping)
    if workspace is not None:
        if workspace["max_tokens"] < max_tokens:
            raise ValueError(
                f"Kimi K3 latent all-reduce workspace serves {workspace['max_tokens']} tokens, "
                f"{max_tokens} requested"
            )
        return workspace
    from ...bindings.internal.runtime import McastGPUBuffer
    from ..distributed.ops import MNNVLAllReduce, get_or_scale_allreduce_mnnvl_workspace

    # The shared workspace's communicator (split by the all-reduce group) is the same group.
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)["comm"]
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    if _K3_LATENT_AR_ONESHOT_MAX_BYTES:
        # One-shot rows under the latent all-reduce's own limit: every rank's rows in one buffer.
        row_bytes = hidden_size * mapping.tp_size * torch.tensor([], dtype=dtype).element_size()
        one_shot_rows = min(max_tokens, _K3_LATENT_AR_ONESHOT_MAX_BYTES // row_bytes)
        need = max(need, one_shot_rows * row_bytes)
    buffer_bytes = -(-need // (1 << 20)) << 20
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle,
        comm.py2f(),
    )
    # Lamport-clean (-0.0) before any rank writes into it.
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    # [current index, dirty index, bytes per buffer, dirty stages, bytes to clear x4, arrivals]
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    workspace = {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        # The multicast view of the same buffers (stores reach every rank's copy).
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        # Counter of the side-stream Lamport clear (TRTLLM_K3_LATENT_AR_SIDE_CLEAR).
        "clear_ctr": torch.zeros(1, dtype=torch.int32, device=buffer.device),
        "max_tokens": max_tokens,
        # Bytes of one of the three Lamport buffers.
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }
    _K3_LATENT_AR_WORKSPACES[mapping] = workspace
    return workspace


# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)

# Kimi K3.1 routed-expert tactic at verify widths under MoE TP4 (TRTLLM-Gen
# MXFP4 x MXFP8 runner, 768-wide local intermediate): (lo, hi, tile_n, config)
# over the autotuner's power-of-two token buckets. [16, 50] (token tile 16) is
# the fastest of the runner's tactics on K3.1 verify routing at 28 / 56 / 112
# rows for both 448 and 224 local experts (TP4xEP2 / TP4xEP4), 2.3-3.5 us per
# MoE layer below the token-tile-32 tactic the autotuner ranks first under its
# dummy routing, and its output is bitwise-equal to that tactic's.
# KIMI_K31_MOE_TACTIC_PIN=0 leaves the choice to the autotuner. The 128-token bucket (128..255 rows: TEP16
# TP4xEP4 at B24+) is pinned too: [16, 50] 73.09 vs the autotuned token-tile-32 tactic 74.53 us per layer at 144
# rows over 224 local experts (GB300, realistic verify routing). The 256- and 512-token buckets (256..1023 rows: TEP16
# B48+ and Helix16 gens under MoE TP4 x EP4) take [32, 34] instead of the autotuned token-tile-128 tactic: 94.1 /
# 101.6 / 118.3 / 131.3 vs 105.0 / 112.1 / 126.4 / 137.7 us per layer at 288 / 384 / 576 / 768 rows, bitwise-equal.
_K31_TP4_DECODE_MOE_TACTIC_PINS = ((16, 128, 16, 50), (256, 512, 32, 34))
# Without MoE TP or with MoE TP2 (EP8 / EP16 / TP2xEP4, 3072- / 1536-wide local intermediate) the autotuner's pick
# is the fastest tactic in the 16- and 32-token buckets but not in the 64-token bucket (64..127 rows: 112 rows =
# TEP8 B16 / ~96 rows = TEP16 B16), where it profiles 64 dummy rows and picks token tile 32: [16, 54] is faster at
# the real rows with bitwise-equal output (tactic sweep on realistic verify routing, GB300, us per MoE layer): EP8
# 112 rows 114.17 -> 111.50, EP16 96 rows 63.12 -> 62.25, TP2xEP4 112 rows 118.86 -> 115.87. Tensor-parallel
# attention only (measured there; attention-DP gens keep the autotuner). KIMI_K31_MOE_EP_TACTIC_PIN=0 drops it.
_K31_EP_DECODE_MOE_TACTIC_PINS = ((64, 64, 16, 54),)
# Without MoE TP the 256-token bucket (256..511 rows: TEP16 / Helix16 gens at B48-B85) also takes [64, 11] instead of
# the autotuned token-tile-128 tactic: 100.2 / 106.6 vs 104.9 / 111.6 us per layer at 288 / 384 rows over 56 local
# experts, bitwise-equal (from 512 rows the autotuner's pick is the fastest).
_K31_EP1_DECODE_MOE_TACTIC_PINS = _K31_EP_DECODE_MOE_TACTIC_PINS + ((256, 256, 64, 11),)


# Every decode bucket of the deployed K3.1 gen layouts, keyed by (MoE TP size, local experts): without a pin the
# tactic is the autotuner's pick on dummy routing, made per process, and in some processes it lands on a much slower
# one (token tile 256 runs 2-3x slower than tile 128 at 256..2048 rows); with lockstep collectives one slow rank
# stalls every rank. Each pin is the tactic with the smallest worst-case slowdown over the measured row counts of its
# bucket (us per MoE layer, GB300, 92 layers of realistic verify routing in one CUDA graph; output bitwise-equal to
# the runner's default tactic):
# - EP8 (TEP8, 112 x 3072): tile 8 at 4..15 rows, [16, 54] for 16..127 (e.g. 108.9 vs autotuned 111.5 at 112).
# - EP16 (TEP16 / Helix16 EP16, and DEP16 gens: 56 x 3072): tile 8 at 4..15, [16, 54] 16..127, [64, 11] 128..511
#   (144 / 240 / 288 rows 81.9 / 96.7 / 101.5 vs autotuned 83.9 / 101.8 / 106.3), tile 128 from 512 rows.
# - DEP32 (28 x 3072; the runner sees all 32 ranks' rows): [128, 18] / [128, 19] for 256..8191 rows (tile 256:
#   150-212 vs 60-155).
# - MoE TP4 x EP2 (TEP8, 448 x 768): tile 8 at 4..15, [16, 50] 16..255.
# - MoE TP4 x EP4 (TEP16 / Helix16, 224 x 768): tile 8 at 4..15, [16, 50] 16..255, [32, 34] 256..511, [64, 10]
#   512..2047 (576 / 1152 / 1792 rows 120.4 / 148.8 / 175.1 vs autotuned 125.0 / 153.2 / 177.8).
# - MoE TP4 x EP8 (TEP32, 112 x 768): [8, 128] at 4..31 rows (16 rows 18.67 vs autotuned 19.27 and [16, 50]), [16, 48]
#   at 32..63 (32 rows 25.46 vs 25.80); [16, 50] 64..255 and [32, 34] 256..511 as the TP4 defaults below (at 512 rows
#   over 112 experts the runner lists only larger token tiles).
# Layouts without a table keep the pins above.
_K31_LAYOUT_DECODE_MOE_TACTIC_PINS = {
    (1, 112): ((4, 4, 8, 132), (8, 8, 8, 140), (16, 64, 16, 54)),
    (1, 56): (
        (4, 4, 8, 140),
        (8, 8, 8, 132),
        (16, 64, 16, 54),
        (128, 256, 64, 11),
        (512, 512, 128, 18),
        (1024, 1024, 128, 19),
    ),
    (1, 28): ((256, 2048, 128, 18), (4096, 4096, 128, 19)),
    (4, 448): ((4, 4, 8, 140), (8, 8, 8, 128), (16, 128, 16, 50)),
    (4, 224): ((4, 8, 8, 128), (16, 128, 16, 50), (256, 256, 32, 34), (512, 1024, 64, 10)),
    (4, 112): ((4, 16, 8, 128), (32, 32, 16, 48), (64, 128, 16, 50), (256, 256, 32, 34)),
}
# The layouts whose tables were measured with attention-DP row counts too (DEP16 / DEP32 gens).
_K31_ATTENTION_DP_PIN_LAYOUTS = frozenset({(1, 56), (1, 28)})
# Row-range tactics inside the buckets, per layout: the bucket's tactic has to be one the runner lists at the bucket's
# own row count (its token-tile window, from the mean rows per local expert as if every row stayed local), but on
# verify routing (most rows routed to other ranks' experts) smaller tiles stay faster further up, so the MXFP4 ops
# run these (bitwise-equal) tactics for the real row count instead (us per MoE layer, same benchmark as the pins):
# - EP16 / DEP16 (56 x 3072): [16, 54] to 229 rows (156 / 224 rows 73.7 / 86.5 vs [64, 11] 83.9 / 93.1), [32, 18] to
#   895 (320 / 576 rows 98.0 / 119.2 vs 103.6 / 126.1), [64, 11] to 2047 (1536 rows 159.9 vs 164.5), [128, 19] to
#   8191 (DEP16 at 16+ requests per rank: 2688 / 4096 rows 193.4 / 222.4 vs [64, 11] 196.9 / 240.4, token tile 32
#   248.9 / 322.9).
# - DEP32 (28 x 3072): [32, 18] for 256..767 rows (256 / 512 rows 52.2 / 64.1 vs [128, 18] 60.4 / 70.9), [64, 11]
#   768..1279, [128, 18] to 4095 and [128, 19] to 8191 rows. The runner sees all 32 ranks' rows (32 x the busiest rank's
#   batch), but under attention DP the autotuner buckets the per-rank token count, so the bucket pins above do not
#   reach these row counts and without a row tactic the per-rank tuning picks e.g. token tile 32 at 1568 rows
#   (7 requests per rank, DSpark6): 105.4 vs [128, 18] 90.4 us at 1536 rows.
# - MoE TP4 x EP4 (224 x 768): [32, 34] for 512..735 rows (520 / 640 rows 114.3 / 123.4 vs [64, 10] 117.3 / 124.6).
# KIMI_K31_MOE_ROW_TACTICS=0 keeps the bucket's tactic.
_K31_LAYOUT_DECODE_MOE_ROW_TACTICS = {
    (1, 56): ((128, 229, 16, 54), (230, 895, 32, 18), (896, 2047, 64, 11), (2048, 8191, 128, 19)),
    (1, 28): (
        (256, 767, 32, 18),
        (768, 1279, 64, 11),
        (1280, 4095, 128, 18),
        (4096, 8191, 128, 19),
    ),
    (4, 224): ((512, 735, 32, 34),),
}


def _k31_moe_tactic_pins(
    cfg, moe_tp_size: int, attention_dp: bool = False, local_num_experts: Optional[int] = None
):
    if os.environ.get("KIMI_K31_MOE_TACTIC_PIN", "1") == "0":
        return None
    if getattr(cfg, "dsa_config", None) is None:
        return None
    ep_pins_on = os.environ.get("KIMI_K31_MOE_EP_TACTIC_PIN", "1") == "1"
    layout = (moe_tp_size, local_num_experts)
    table = _K31_LAYOUT_DECODE_MOE_TACTIC_PINS.get(layout)
    if table is not None and (moe_tp_size == 4 or ep_pins_on):
        if not attention_dp or layout in _K31_ATTENTION_DP_PIN_LAYOUTS:
            return table
        return None
    if moe_tp_size == 4:
        return _K31_TP4_DECODE_MOE_TACTIC_PINS
    if moe_tp_size in (1, 2) and not attention_dp and ep_pins_on:
        return (
            _K31_EP1_DECODE_MOE_TACTIC_PINS if moe_tp_size == 1 else _K31_EP_DECODE_MOE_TACTIC_PINS
        )
    return None


def _k31_moe_row_tactics(
    cfg, moe_tp_size: int, attention_dp: bool = False, local_num_experts: Optional[int] = None
):
    """Row-range tactics of the layout, where its per-layout pin table applies (else ``()``)."""
    if os.environ.get("KIMI_K31_MOE_ROW_TACTICS", "1") == "0":
        return ()
    layout = (moe_tp_size, local_num_experts)
    pins = _k31_moe_tactic_pins(cfg, moe_tp_size, attention_dp, local_num_experts)
    if pins is None or pins is not _K31_LAYOUT_DECODE_MOE_TACTIC_PINS.get(layout):
        return ()
    return _K31_LAYOUT_DECODE_MOE_ROW_TACTICS.get(layout, ())


# Fused latent-MoE tail (K3.1 TEP gen) for TP MoE layers
# (modules/kimi_k3_moe_tail, up to its per-TP dispatch limit): routed AllReduce +
# RMSNorm + shared ReduceScatter in one kernel, then a 1/TP-row up-projection whose
# epilogue adds the shared shard and multicasts the all-gather; BF16 throughout (BF16
# weights, BF16 Lamport collectives with FP32 rank-order sums). The up-projection's
# per-shard accumulation order differs from the full GEMM. Per call it excludes the
# MoE residual deferral of the PDL attn_res kernel's mode "2" and the other tails: a call that
# takes it returns the dense residual update. Where it runs, it leaves its output in its
# multicast mailbox and the next layer's pre-attention mix reads it from there
# (trtllm::k3_attn_res_mailbox_add_rmsnorm: poll, re-arm, BF16 add to the residual, mix,
# RMSNorm) instead of the tail's copy kernel, the separate residual add and the mix on
# the sum; the same bits. Layers whose mix cannot read it (DSpark taps, the last layer,
# shapes the PDL kernel does not serve) run the copy kernel and the add.
_K31_FUSED_MOE_TAIL = False


# latent_down stream (K3.1 TEP gen, TRTLLM-Gen routed experts): the latent
# down-projection runs on the MoeOutputMemset aux stream (as the CuteDSL backends do), so it
# runs beside the router GEMM instead of after it; the routed experts join that stream before
# their first read of the latent (ConfigurableMoE ``input_ready_event``). At decode batch sizes
# both GEMMs are latency-bound weight streams (router 12.8 MB, latent_down 51 MB per rank), and
# running them back to back on the main stream puts both latencies on the MoE front. Same
# kernels, same inputs: bitwise.
_K31_LATENT_DOWN_STREAM = False
# TLLM_TRTLLMGEN_FORCE_SEPARATED_ROUTING for the Kimi K3.1 routing (K3.1 TEP gen): the top-k is
# computed outside the TRTLLM-Gen kernel (``_KimiK3RoutingMethod.requires_separated_routing``),
# which the router sort and the sharded MoE front hand their routing over through.
_K31_SEPARATED_ROUTING = False
# TRTLLM_K3_L2PF_V2 (K3.1 TEP gen): the decode L2 weight prefetch at every point
# (modules/kimi_k3_l2_prefetch.py).
_K31_L2PF = False


def _k31_latent_down_stream_requested(moe_backend: str) -> bool:
    return moe_backend == "TRTLLM" and _K31_LATENT_DOWN_STREAM


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...llmapi.llm_args import DecodingBaseConfig

# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0), but the chunked-context path indexes the table by
# absolute position, so it must cover max_position_embeddings (~512MB per
# backend for the 1M-position checkpoint); a smaller table is read out of
# bounds. KIMI_K3_MLA_MAX_POSITIONS overrides the size for short-context
# deployments.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"
_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES = (
    ".self_attn.mixer.k_b_proj_trans",
    ".self_attn.mixer.v_b_proj",
    ".self_attn.mixer.k_b_proj_trans_scale",
    ".self_attn.mixer.v_b_proj_scale",
    ".self_attn.mixer.k_b_proj_trans_dequant",
    ".self_attn.mixer.v_b_proj_dequant",
)

# BF16 shared/latent MLP weights are optionally quantized once after loading.
# Checkpoint-native attention FP8 is independent of this lossy conversion.
_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"


class KimiK3MoEGate(nn.Module):
    """Kimi K3 gate weights and routing method for ``ConfigurableMoE``."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        assert self.moe_router_activation_func in ("sigmoid", "softmax"), (
            "K3 MoE gate supports sigmoid or softmax scoring only"
        )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            return torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_2d.contiguous(),
                self.weight.t(),
                bias=None,
                out_dtype=torch.float32,
            )
        return torch.nn.functional.linear(
            router_input_fp32(hidden_2d),
            self.weight.type(torch.float32),
            None,
        )

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        """Return the shared DeepSeek-V3 router used by ``ConfigurableMoE``."""
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 ConfigurableMoE routing requires sigmoid scores.")
        if not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 ConfigurableMoE routing requires top-k weight renormalization."
            )
        return _KimiK3RoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )


class _KimiK3RoutingMethod(DeepSeekV3MoeRoutingMethod):
    """DeepSeek-V3 routing that can hand out top-k results computed ahead of time.

    ``KimiK3MoERuntime`` stores the fused router's ``(topk_ids, topk_weights)`` in
    ``precomputed`` right before calling the routed experts; the next ``apply``
    returns and clears them instead of routing ``logits`` again. Alternatively it
    stores a ``deferred`` callable that the next ``apply`` runs (once) on ``logits``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.deferred: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        # KIMI_K31_ROUTER_SORT: ``(runner routing tensors, tile, topk_ids)`` for the precomputed
        # top-k; the TRTLLM-Gen MXFP4 backend takes and clears it.
        self.presorted: Optional[Tuple[List[torch.Tensor], int, torch.Tensor]] = None

    @property
    def requires_separated_routing(self) -> bool:
        """Kimi K3.1 TEP generation servers route outside the TRTLLM-Gen kernel
        (``_K31_SEPARATED_ROUTING``)."""
        return _K31_SEPARATED_ROUTING or super().requires_separated_routing

    def apply(
        self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.precomputed is not None:
            result, self.precomputed = self.precomputed, None
        elif self.deferred is not None:
            route, self.deferred = self.deferred, None
            result = route(logits)
        else:
            result = super().apply(logits, input_ids)
        return result


class KimiK3RMSNorm(nn.Module):
    """RMSNorm matching the Kimi checkpoint implementation's rounding."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.to(torch.float32)
        variance = hidden_states_float.pow(2).mean(-1, keepdim=True)
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states_float.to(input_dtype)


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite K3-family config."""
    if getattr(pretrained_config, "model_type", None) in KIMI_K3_COMPOSITE_MODEL_TYPES or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


def _router_exact_for_layer(cfg, layer_idx: int) -> bool:
    """Whether ``TRTLLM_K3_ROUTER_EXACT`` serves the router of this layer's MoE block."""
    if _K3_ROUTER_EXACT_MODE == "kda":
        return _is_kda_layer(cfg, layer_idx)
    return _K3_ROUTER_EXACT_MODE == "1"


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
"""Which residual-stream value the DFlash/DSpark hidden-state tap captures.

``1`` (default) captures the pre-norm attn_res mixture -- the value the next
consumer actually reads. ``0`` captures the raw running prefix sum instead.

Both conventions exist in the wild and a drafter distilled against one scores
lower on the other with nothing raised, so this is a property of the DRAFTER
checkpoint, not a performance knob. SGLang (and therefore RadixArk/Kimi-K3-DSpark)
uses the mixture: ``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py
aggregate_stream``. vLLM implements both and defaults to the prefix
(``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``, ``models/kimi_k3/nvidia/model.py
_capture_aux_hidden_stream``), which is what a TorchSpec-distilled drafter may
have been trained against. Measured cost of getting it wrong on K3 + RadixArk:
AR 71.4% -> 66.9%.

Per-checkpoint measurements on K3, GSM8K AL, n=200, TEP8:

===================  ==================  ============  ======
drafter              stream (default 1)  prefix (0)    delta
===================  ==================  ============  ======
RadixArk (GQA)       71.8%               --            --
Inferact (MLA)       65.7%               66.6%         +0.9pt
===================  ==================  ============  ======

So the default is right for RadixArk. For Inferact the prefix convention
matches its vLLM/TorchSpec lineage and measures better, but +0.9pt at n=200 is
inside this harness's noise band (it treats RadixArk's own 71-73% spread as
noise), so this is a direction, not a settled requirement -- unlike the 4.5pt
RadixArk case above, which was unambiguous. Both Inferact acc_len values (5.60
and 5.66) sit on its model card's 5.64, so neither convention is grossly wrong
for it.

Deriving this from checkpoint metadata is not possible today: neither published
drafter's config records which capture convention it was distilled against."""

_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"

KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to disable the in-tree fused Torch op
``trtllm::attn_res_fwd`` (Blackwell only). Default: fused with fallback."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"
"""Largest per-rank token count for which the fused epilogue is taken.

Accepts any integer >= 1. The default follows ``KIMI_K3_ATTN_RES_TOPOLOGY``:
1 when the persistent port is off (the pre-existing decode-only gate) and 32
when it is on. Setting this explicitly overrides both.

Resolved again with the topology when the model is built (before any CUDA-graph
capture), so capture and replay cannot disagree about it.
"""

KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"
"""Which fused attention-residual kernel topology to use.

Accepted values:

  ``per_token``   one CTA (N<=4) or one 8-CTA cluster (N>=5) per token; the
                  persistent kernel is never used. This is the default, i.e.
                  the feature is off and behaviour is unchanged, except on
                  Kimi K3.1 attention-DP context servers, whose default is
                  ``split`` (``_configure_k31_decode``).
  ``persistent``  one CTA per SM looping over tokens, used at every shape the
                  persistent kernel implements (H == 7168 and 2 <= N <= 9).
  ``split``       persistent above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``
                  tokens, per-token at or below it.
  ``1``           the only accepted on-value; resolves to ``split``.

Off by default because the persistent kernel measures as no-regression on
NVFP4 but as a 2.98x context-phase regression on W4A8_MXFP4_MXFP8, through a
downstream interaction with the TRTLLM-gen MoE rather than through this kernel.

Read at import and again when the model is built (``_configure_k31_decode``,
before any CUDA-graph capture), so capture and replay cannot disagree about it.
"""

_ATTN_RES_TOPOLOGIES = ("per_token", "persistent", "split")

# Turning the port on should not require knowing which of the three topologies
# is the right one: "1" selects the measured policy (``split``), so enabling the
# feature and choosing the policy are one action through one variable.
_ATTN_RES_TOPOLOGY_ON = "1"


def _read_attn_res_topology(default: str = "per_token") -> str:
    """``KIMI_K3_ATTN_RES_TOPOLOGY``, else ``default`` (``per_token``: the persistent kernel is opt-in).

    ``1`` is the only accepted on-value and resolves to ``split``. The named
    topologies stay available for measurement: ``persistent`` uses the
    persistent kernel at every shape it implements, ``per_token`` at none.
    """
    raw = os.environ.get(KIMI_K3_ATTN_RES_TOPOLOGY_ENV, default)
    if raw == _ATTN_RES_TOPOLOGY_ON:
        return "split"
    if raw not in _ATTN_RES_TOPOLOGIES:
        # Loudly, for the same reason as the token ceiling below: a mistyped A/B
        # arm that silently fell back to the default would measure one side
        # twice and report no difference.
        raise ValueError(
            f"{KIMI_K3_ATTN_RES_TOPOLOGY_ENV} must be one of "
            f"{_ATTN_RES_TOPOLOGIES} or {_ATTN_RES_TOPOLOGY_ON!r} "
            f"(which means 'split'), got {raw!r}"
        )
    return raw


def _read_fused_attn_res_max_tokens() -> int:
    """Resolved after the topology, because its default follows it.

    With the persistent port off -- the default -- the ceiling is 1, the
    pre-existing gate: the fused epilogue is taken at the single-token decode
    shape and nowhere else. Enabling the port raises it to 32, the top of the
    measured range, so that "off" keeps meaning "unchanged".
    """
    default = "1" if _ATTN_RES_TOPOLOGY == "per_token" else "32"
    raw = os.environ.get(KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV, default)
    try:
        value = int(raw)
    except ValueError:
        # Failing loudly matters more than usual here: a mistyped A/B arm that
        # silently fell back to the default would measure the candidate twice
        # and report no difference.
        raise ValueError(
            f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be >= 1, got {value}")
    return value


_ATTN_RES_TOPOLOGY = _read_attn_res_topology()
_FUSED_ATTN_RES_MAX_TOKENS = _read_fused_attn_res_max_tokens()


def _persistent_attn_res_applicable(M: int, H: int, N: int) -> bool:
    """Shape gate for the persistent kernel: H == 7168 and 2 <= N <= 9.

    No token ceiling: the persistent grid is sized by the SM count, not by the
    token count, so prefill is the case it exists for.
    """
    del M  # deliberately unused; see above
    return H == 7168 and 2 <= N <= 9


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Pick between the two fused kernels for this call site.

    ``persistent`` takes the persistent kernel at every shape it implements;
    ``split`` takes it only above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens,
    which stands in for the prefill/decode boundary. Shapes the persistent
    kernel does not implement fall through to the caller's existing gate and
    land on the unfused path.
    """
    if not _persistent_attn_res_applicable(M, H, N):
        return False
    if _ATTN_RES_TOPOLOGY == "persistent":
        return True
    return _ATTN_RES_TOPOLOGY == "split" and M > _FUSED_ATTN_RES_MAX_TOKENS


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    proj_w = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_w = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix, block):
        t = prefix.shape[0]
        output, _rsigma, _probs, _logits = attn_res_op(
            prefix.reshape(t, 1, H).contiguous(),
            block.reshape(K, t, 1, H).contiguous(),
            proj_w,
            norm_w,
            float(norm.eps),
        )
        return output.reshape(t, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op takes at most _ATTN_RES_OP_MAX_TOKENS tokens per call and is
    # per-token, so longer prefill chunks run in slices; the fp32 reference
    # below would otherwise cost seconds per iteration at 24k+ tokens.
    out = torch.empty_like(prefix_sum)
    for lo in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        hi = min(M, lo + _ATTN_RES_OP_MAX_TOKENS)
        out[lo:hi] = run(prefix_sum[lo:hi], block_residual[:, lo:hi])
    return out


def _rms_norm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return float(norm.variance_epsilon)


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether the fused path was actually reached, once per shape.

    ``_FUSED_ATTN_RES_ENABLED`` only says the feature is switched on, not that
    the shape gate let the call through, and a rejected call looks exactly like
    a disabled one in the logs. Emitted at debug level: it is once per distinct
    shape, not once per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


# PDL attn_res kernel (_K3_ATTN_RES_PDL_MODE "1" K3.1 TEP gen, "2" K3.1 DEP gen; "0" off): up to
# _K3_ATTN_RES_PDL_MAX_M tokens (min(max_num_tokens, 192 TEP / 256 DEP)), the fused attn_res(+add)
# +RMSNorm runs as trtllm::k3_attn_res_(add_)rmsnorm (kernels/blackwell/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements (TRTLLM_K3_ATTN_RES_PDL_EXACT=1 selects a slower, bit-identical variant).
# Mode "2" also leaves the MoE residual update residual + bf16(routed + shared)
# of the BF16-add path (kimi_k3_moe_residual) to the next layer's pre-attention mix, which
# computes it bit-identically (trtllm::k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.
_K3_ATTN_RES_PDL_MODE = "0"
_K3_ATTN_RES_PDL = False
_K3_ATTN_RES_PDL_MAX_M = 0
# A_mid's in-place read (K3.1 TEP gen, with the pushed all-reduces): where the mix after attention (A_mid) adds
# the attention output to the running prefix sum, the attention block finishes its pushed o_proj
# all-reduce in place (two-shot transport, no gather) and A_mid reads the result straight from the
# MNNVL Lamport workspace (trtllm::k3_attn_res_add_rmsnorm_lamport): the same bits, one kernel
# boundary and one copy of the rows fewer.
_K3_AR_PUSH_AMID = False
# A_mid's fused finish (K3.1 TEP gen, with A_mid's in-place read): where A_mid reads the o_proj
# all-reduce from the Lamport workspace, A_mid itself finishes the pushed call (k3_attn_res
# ``lamport_finish``: one-shot transport up to TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot
# above, where each row's owner reduces and multicasts it) and advances the workspace: no finish
# kernel or C++ all-reduce runs in between. Where o_proj's GEMM has no push epilogue a copy kernel
# pushes its output. The sums have the order and rounding of the finish kernel, so A_mid produces
# the same bits.
_K3_AR_FINISH_FUSED = False
# A_mid's early poll (K3.1 TEP gen, with A_mid's fused finish): A_mid finishes the pushed o_proj
# all-reduce before its griddepcontrol.wait (k3_attn_res ``early_poll``): it becomes resident when
# the o_proj push GEMM passes its own wait, polls every rank's slots while o_proj runs, and only
# its stores wait for the o_proj grid, so the o_proj -> A_mid boundary leaves the critical path
# (the post-attention collective sandwich as two PDL grids). Same bits.
_K3_ULL_OPROJ_SANDWICH = False


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

        return _fused_moe_residual_add(self.routed, self.shared, self.residual)


class _DeferredTailResidual(NamedTuple):
    """A MoE layer's ``bf16(residual + moe_output)`` whose MoE output the fused latent-MoE tail
    left in the first ``rows`` rows of its mailbox (the mailbox fold): the next
    layer's pre-attention mix reads it from there, or ``materialize`` runs the tail's copy kernel
    and the BF16 add (consuming the mailbox either way, exactly once)."""

    residual: torch.Tensor
    tail: KimiK3LatentMoETail
    mailbox: torch.Tensor

    def materialize(self) -> torch.Tensor:
        return self.residual + self.tail.copy_out(self.mailbox, self.residual.shape[0])


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return _K3_ATTN_RES_PDL_MODE == "2" and num_tokens <= _K3_ATTN_RES_PDL_MAX_M


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, (_DeferredMoEResidual, _DeferredTailResidual)):
        return hidden_states.materialize()
    return hidden_states


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
):
    """``trtllm::k3_attn_res_(add_)rmsnorm`` when enabled and applicable, else ``None``."""
    if not _K3_ATTN_RES_PDL or prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    from ..kernels.blackwell import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    eps = float(norm.eps)
    out_eps = _rms_norm_eps(output_norm)
    if addend is None:
        return torch.ops.trtllm.k3_attn_res_rmsnorm(
            prefix_sum, block_residual, *weights, eps, out_eps
        )
    return torch.ops.trtllm.k3_attn_res_add_rmsnorm(
        prefix_sum, addend, block_residual, *weights, eps, out_eps
    )


def _k3_attn_res_lamport_applies(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> bool:
    """Whether A_mid on ``prefix_sum`` can read its addend from a Lamport workspace
    (A_mid's in-place read): the vectorized PDL attn_res kernel serves the ADD_PREFIX mix."""
    if (
        not _K3_AR_PUSH_AMID
        or not _K3_ATTN_RES_PDL
        or torch.compiler.is_compiling()
        or prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M
    ):
        return False
    from ..kernels.blackwell import k3_attn_res

    if k3_attn_res._EXACT:
        return False
    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    return k3_attn_res.can_use_k3_attn_res(prefix_sum, (prefix_sum,), block_residual, *weights)


def _k3_attn_res_pdl_lamport(
    prefix_sum: torch.Tensor,
    addend: Union[LamportResult, LamportPending],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum + all-reduce result, normed mix)`` with the result read from the Lamport
    workspace (``trtllm::k3_attn_res_add_rmsnorm_lamport``; for a ``LamportPending`` the mix also
    finishes the all-reduce, A_mid's fused finish), or ``None`` if that does not apply."""
    if tuple(addend.shape) != tuple(prefix_sum.shape) or not _k3_attn_res_lamport_applies(
        prefix_sum, block_residual, proj, norm, output_norm
    ):
        return None
    M, H = prefix_sum.shape
    if isinstance(addend, LamportPending):
        _note_attn_res_fusion(
            "add+attn_res+norm/pdl/lamport_finish",
            True,
            M,
            H,
            int(block_residual.shape[0]) + 1,
        )
        ws = addend.workspace
        return torch.ops.trtllm.k3_attn_res_lamport_add_rmsnorm(
            prefix_sum,
            ws.uc_local,
            ws.flags.data_ptr(),
            block_residual,
            proj.weight.reshape(-1),
            norm.weight,
            output_norm.weight,
            float(norm.eps),
            _rms_norm_eps(output_norm),
            *addend.finish_args,
            False,
            _K3_ULL_OPROJ_SANDWICH,
        )
    _note_attn_res_fusion(
        "add+attn_res+norm/pdl/lamport", True, M, H, int(block_residual.shape[0]) + 1
    )
    return torch.ops.trtllm.k3_attn_res_add_rmsnorm_lamport(
        prefix_sum,
        addend.workspace.uc_local,
        addend.workspace.flags.data_ptr(),
        block_residual,
        proj.weight.reshape(-1),
        norm.weight,
        output_norm.weight,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``."""
    residual, routed, shared = deferred
    if residual.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    from ..kernels.blackwell import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    return torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm(
        residual,
        routed,
        shared,
        block_residual,
        *weights,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )


def _k3_attn_res_pdl_tail_mailbox(
    deferred: _DeferredTailResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum, normed mix)`` of a fused-tail residual update left in the tail's mailbox
    (the mailbox fold), where the mix on the materialized sum would run the
    vectorized PDL kernel (``_k3_attn_res_pdl``), else ``None``: the kernel adds the mailbox rows
    as they arrive, with the bits of the copy kernel + BF16 add + that mix."""
    residual = deferred.residual
    M, H = residual.shape
    if (
        not _FUSED_ATTN_RES_ENABLED
        or not _K3_ATTN_RES_PDL
        or M > _K3_ATTN_RES_PDL_MAX_M
        or torch.compiler.is_compiling()
    ):
        return None
    from ..kernels.blackwell import k3_attn_res

    if k3_attn_res._EXACT:
        return None
    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (residual,), block_residual, *weights):
        return None
    _note_attn_res_fusion(
        "tail_mailbox+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1
    )
    return torch.ops.trtllm.k3_attn_res_mailbox_add_rmsnorm(
        residual,
        deferred.mailbox,
        block_residual,
        *weights,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # The fused path is taken for M <= _FUSED_ATTN_RES_MAX_TOKENS, H == 7168 and
    # N <= 12, which is the measured window; larger token counts have not been
    # measured and fall back to the unfused add + attn_res_fwd + RMSNorm path.
    pdl_out = _k3_attn_res_pdl(prefix_sum, None, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        _, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_rmsnorm_op = torch.ops.trtllm.attn_res_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = attn_res_rmsnorm_op(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or addend.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not addend.is_cuda
        or not block_residual.is_cuda
        or prefix_sum.shape != addend.shape
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Same measured window as _apply_attn_res_rmsnorm_fused above.
    pdl_out = _k3_attn_res_pdl(prefix_sum, addend, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("add+attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        updated_prefix_sum, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_add_rmsnorm_op = torch.ops.trtllm.attn_res_add_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = attn_res_add_rmsnorm_op(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch directly to the in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
        # The fp32 reference below is orders of magnitude slower at prefill
        # sizes; say so once per shape rather than fall back silently.
        _note_attn_res_fusion(
            "attn_res",
            False,
            prefix_sum.shape[0],
            prefix_sum.shape[1],
            int(block_residual.shape[0]) + 1,
        )
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
        if fused is not None:
            return fused
    return output_norm(_apply_attn_res(prefix_sum, block_residual, proj, norm))


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    ``addend`` may be a ``LamportResult`` (A_mid's in-place read): the attention block's
    all-reduce result left in the MNNVL workspace, read in place where the PDL attn_res kernel
    serves the mix and gathered into a tensor otherwise; or a ``LamportPending``
    (A_mid's fused finish), which the PDL kernel finishes or the finish kernel turns into
    a tensor."""
    if isinstance(addend, (LamportResult, LamportPending)):
        fused = _k3_attn_res_pdl_lamport(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
        addend = addend.materialize()
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


# ---------------------------------------------------------------------------
# Dense / shared-expert MLP: fused [gate | up] layout (``GatedMLP``).
#
# The HF checkpoint stores separate ``gate_proj`` / ``up_proj`` tensors;
# ``load_weights`` row-concatenates them into ``gate_up_proj`` (see
# ``_gate_up_ckpt_keys``), replacing two GEMMs + torch.cat with one GEMM.
# ---------------------------------------------------------------------------


_GATE_UP_FUSED_SUFFIX = ".gate_up_proj.weight"


def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
    """Checkpoint ``(gate_proj, up_proj)`` keys whose row-concat loads the
    fused ``gate_up_proj`` parameter named by ``fused_key``."""
    return (
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".gate_proj.weight"),
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".up_proj.weight"),
    )


def _shard_head_major_param(
    name: str,
    src: torch.Tensor,
    param: torch.nn.Parameter,
    *,
    kda_tp_size: int,
    kda_tp_rank: int,
    model_tp_rank: int,
) -> torch.Tensor:
    """TP-shard a KDA ``linear_attn`` or shared-expert / dense-MLP
    ``down_proj`` checkpoint tensor down to this rank's local slice.

    Returns ``src`` unchanged when it already matches ``param`` (replicated or
    single-rank tensors) and for every other name — the caller then routes
    MLA head-shards and the shape-match copy. The actual slicing is delegated
    to :func:`load_weight_shard` so the framework owns the ceil-divide
    semantics.

    - ``.linear_attn.`` names are KDA head-major tensors (``linear_attn``
      exists only on KDA layers): ``o_proj`` shards its input columns (ROW),
      every other projection its output rows (COLUMN), by ``kda_tp_size``.
    - a shared-expert / dense-MLP ``down_proj`` is ROW-sharded on its input
      columns; the fused ``gate_up_proj`` is row-concatenated and returns
      before this helper, so ``down_proj`` is the only half that reaches here.
      The TP factor comes from the checkpoint-vs-param shapes; a subgroup
      smaller than model TP repeats, so the shard index is ``model_tp_rank``
      modulo the parameter's shard count.
    """
    if src.shape == param.shape:
        return src
    if ".linear_attn." in name:
        mode = (
            TensorParallelMode.ROW if name.endswith(".o_proj.weight") else TensorParallelMode.COLUMN
        )
        return load_weight_shard(src, kda_tp_size, kda_tp_rank, mode, device=param.device)
    if name.endswith(".down_proj.weight") and (".shared_experts." in name or ".mlp." in name):
        assert src.shape[1] % param.shape[1] == 0, (
            f"{name}: checkpoint input dim {src.shape[1]} is not "
            f"divisible by param input dim {param.shape[1]}"
        )
        tp = src.shape[1] // param.shape[1]
        return load_weight_shard(
            src, tp, model_tp_rank % tp, TensorParallelMode.ROW, device=param.device
        )
    return src


def _helix_cp_v_b_shard(
    v_weight: torch.Tensor,
    *,
    num_heads_tp_cp: int,
    cp_rank: int,
) -> torch.Tensor:
    """Select this CP rank's ``v_b_proj`` head chunk from a tp-local KV-B split.

    Under Helix context-parallel (cp_size > 1) ``v_b_proj`` holds only this
    rank's 1/cp post-all-to-all head chunk, while ``kv_b_proj`` and
    ``k_b_proj_trans`` keep every tp-local head. When cp_size == 1
    (``num_heads_tp_cp`` equals the full tp-local head count) the tensor is
    returned unchanged. ``v_weight`` is ``[num_heads_tp, v_head_dim,
    kv_lora_rank]``; only its leading head axis is sliced.
    """
    if num_heads_tp_cp != v_weight.shape[0]:
        lo = cp_rank * num_heads_tp_cp
        v_weight = v_weight[lo : lo + num_heads_tp_cp]
    return v_weight


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _low_m_gemm_scale_from_mxfp8(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from the SM107 CuTe MXFP8 scale."""
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. On both, the low-M GEMM is a candidate of forward like in the
    generic FP8 block-scale linear (``Fp8BlockScaleLowMGemmRange``); it reads a
    DeepGEMM-packed scale, which is ``weight_scale`` on SM100/SM103 and a copy
    repacked from the MXFP8 scale on SM107.
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Rows the low-M GEMM serves and the DeepGEMM-packed scale it reads; set at post-load.
        self._low_m = Fp8BlockScaleLowMGemmRange()
        self._low_m_weight_scale: Optional[torch.Tensor] = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether a producer may hand this GEMM an FP8 activation in CuTe layout."""
        return _rubin_mxfp8_is_available()

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, backend-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from ...deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear._prepare_weights(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_weights(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        # Both GEMMs consume UE8M0 scales rather than the checkpoint's FP32 grid.
        weight_fp8, weight_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            weight_scale = transform_k128_scales_to_cutedsl_mxfp8_layout(
                weight_scale, mn=weight_fp8.shape[0], k=weight_fp8.shape[1]
            )
            return weight_fp8, weight_scale
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        weight_scale = transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[0],
            k=weight_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )
        return weight_fp8, weight_scale

    @classmethod
    def from_linear(cls, linear: nn.Linear | TrtllmLinear) -> "_Fp8BlockScaleWeightReadLinear":
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self.weight, self.weight_scale = self._prepare_weights(self.weight, self.weight_scale)
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self._prepare_low_m_gemm()

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode).
        """
        n, k = self.out_features, self.in_features
        self._low_m = Fp8BlockScaleLowMGemmRange.for_weight(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_weight_scale = self.weight_scale
        elif self._low_m.max_m and self._fused_projection is None:
            self._low_m_weight_scale = _low_m_gemm_scale_from_mxfp8(self.weight_scale, n, k)
        else:
            self._low_m, self._low_m_weight_scale = Fp8BlockScaleLowMGemmRange(), None

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, x.shape[-1])
        if not _rubin_mxfp8_is_available():
            out = fp8_block_scale_ue8m0_gemm(
                x_flat, self.weight, self.weight_scale, self._low_m, output_dtype=x.dtype
            )
        elif self._low_m.serves(x_flat):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale
            )
        else:
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                self.weight,
                act_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        return out.reshape(out_shape)

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an FP8 activation a fused producer already quantized.

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 scales in the CuTe K32 R128c4
                layout that ``fp8_quantize_1x128_cutedsl_ue8m0`` produces.
        """
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
            activation.reshape(-1, activation.shape[-1]),
            self.weight,
            activation_scale,
            self.weight_scale,
            output_dtype=torch.bfloat16,
            fine_grained_m=True,
        )
        return out.reshape(activation.shape[:-1] + (self.out_features,))


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    setattr(parent, attr, _Fp8BlockScaleWeightReadLinear.from_linear(child))
    child.weight.data = child.weight.data.new_empty(0)
    return 1


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. Attention
    follows checkpoint quantization separately. Routed experts and the dense
    layer-0 MLP are not converted. Returns the number of projections converted.
    """
    count = 0

    for layer in model.layers:
        if not _has_weights(layer):
            continue
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            # GatedMLP fuses gate and up into gate_up_proj; keep the split
            # names too so either MLP layout converts. The fused gate_up read
            # only pays off when attention DP re-reads it per rank per step;
            # under TP the bf16 GEMM overlaps on the aux stream and the FP8
            # quantize+GEMM would serialize onto the critical path.
            shared_attrs = (
                ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                if include_fused_gate_up
                else ("gate_proj", "up_proj", "down_proj")
            )
            for attr in shared_attrs:
                child = getattr(shared, attr, None)
                if isinstance(child, TrtllmLinear) and child.tp_size != 1:
                    continue
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, TrtllmLinear)
                )
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            count += _swap_linear_to_fp8_weight_read(moe, attr)

    # Return the freed BF16 blocks to the driver so the raw (non-caching-
    # allocator) allocations made during executor creation succeed on the
    # tight DEP16 memory headroom.
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


# ---------------------------------------------------------------------------
# Latent MoE block using the unified ConfigurableMoE stack.
# ---------------------------------------------------------------------------

# Routed-expert key spellings that ModelOpt emits for Kimi K3. The NVFP4
# checkpoint (``nvidia/Kimi-K3-NVFP4``) lists every prefix x module-name
# combination in ``quantized_layers``, so a lookup over this product finds it
# without needing the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The subset of the above that can be a real module path. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would
# widen what matches instead of just missing, as it does in the dict lookup.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ---------------------------------------------------------------------------
# Routed-expert checkpoint layouts.
#
# K3 streams routed experts one at a time (see ``load_weights``), so the tensor
# names, the per-expert loader and the finalization it needs are all decided by
# the checkpoint's routed-expert quantization rather than by the MoE backend.
# Three call sites need that decision — the expected-key plan, the loader, and
# the file grouping — so it lives in one spec instead of three conditionals.
# ---------------------------------------------------------------------------


# Loader signature: ``(backend, base, ckpt_expert_idx, slot_expert_idx, local_slot_id,
# get_tensor, get_optional_tensor)``. ``ckpt_expert_idx`` names the checkpoint tensors;
# ``slot_expert_idx`` is the routed-expert id the slot serves (the same id unless
# TRTLLM_K3_EXPERT_PLACEMENT relabels it).
def _load_packed_mxfp4_expert(
    backend,
    base,
    ckpt_expert_idx,
    slot_expert_idx,
    local_slot_id,
    get_tensor,
    get_optional_tensor,
) -> None:
    del get_optional_tensor  # this layout stores no optional per-expert tensor
    prefix = f"{base}.{ckpt_expert_idx}"
    backend.quant_method.load_packed_mxfp4_expert(
        backend,
        global_expert_id=slot_expert_idx,
        local_slot_id=local_slot_id,
        w1_weight=get_tensor(f"{prefix}.w1.weight_packed"),
        w1_weight_scale=get_tensor(f"{prefix}.w1.weight_scale"),
        w2_weight=get_tensor(f"{prefix}.w2.weight_packed"),
        w2_weight_scale=get_tensor(f"{prefix}.w2.weight_scale"),
        w3_weight=get_tensor(f"{prefix}.w3.weight_packed"),
        w3_weight_scale=get_tensor(f"{prefix}.w3.weight_scale"),
    )


def _load_nvfp4_expert(
    backend,
    base,
    ckpt_expert_idx,
    slot_expert_idx,
    local_slot_id,
    get_tensor,
    get_optional_tensor,
) -> None:
    # Only an NVFP4_AWQ / NVFP4_ARC calibration writes a pre-quant scale; a
    # plain NVFP4 checkpoint has none and the pair stays None, which leaves
    # ``fc31_act_scale`` unset exactly as before. Dropping it for a checkpoint
    # that HAS one would be silently wrong -- the activation would be quantized
    # unsmoothed against an input scale calibrated on the smoothed one.
    pre_quant_scales = {
        f"{w}_pre_quant_scale": get_optional_tensor(f"{base}.{ckpt_expert_idx}.{w}.pre_quant_scale")
        for w in ("w1", "w3")
    }
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=slot_expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{ckpt_expert_idx}.{w}.{kind}")
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
        **pre_quant_scales,
    )


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # NVFP4 defers cat/pad/interleave and the alpha computation to
    # ``process_weights_after_loading``; the MXFP4 loaders write through.
    needs_layer_finalize: bool
    # Whole per-expert key suffixes the loader passes through WHEN PRESENT, for
    # a calibration artifact only some checkpoints of this layout carry. Spelled
    # per weight rather than per kind, because they are not stored for every
    # ``w``, and kept out of ``kinds`` because demanding them would reject the
    # checkpoints that have none.
    optional_suffixes: Tuple[str, ...] = ()


_K3_NVFP4_EXPERT_CKPT_SPEC = _K3ExpertCkptSpec(
    kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
    loader=_load_nvfp4_expert,
    loaded_slots_attr="_streamed_expert_slots",
    needs_layer_finalize=True,
    optional_suffixes=("w1.pre_quant_scale", "w3.pre_quant_scale"),
)

_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: _K3ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
        needs_layer_finalize=False,
    ),
    QuantAlgo.NVFP4: _K3_NVFP4_EXPERT_CKPT_SPEC,
    # The two NVFP4 calibration recipes store plain NVFP4 expert weights plus
    # the per-channel pre-quant scale above, so they share one layout; the MoE
    # quantization method folds them onto the NVFP4 pipeline as well.
    QuantAlgo.NVFP4_AWQ: _K3_NVFP4_EXPERT_CKPT_SPEC,
    QuantAlgo.NVFP4_ARC: _K3_NVFP4_EXPERT_CKPT_SPEC,
}


def _takes_prequantized_nvfp4_input(backend: Any) -> bool:
    """Whether the routed experts' backend takes an unswizzled NVFP4 ``Fp4QuantizedTensor``
    as its input in place of quantizing the BF16 rows itself (the CuteDSL NVFP4 leaves,
    fused FC12 included, whose ``quantize_input`` is ``quantize_input_nvfp4``)."""
    from ..moe.fused_moe.cutedsl.kernel_inputs import quantize_input_nvfp4

    return getattr(type(backend), "quantize_input", None) is quantize_input_nvfp4


def _k3_expert_ckpt_spec(quant_algo: Optional[QuantAlgo]) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    return spec


def _is_helix_a2a(model_config: ModelConfig) -> bool:
    """Helix + attention DP: CP ranks repurposed to TP, MoE on token slices."""
    mapping = model_config.mapping
    return mapping.enable_attention_dp and mapping.helix_repurposed_cp_size > 1


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

    # TRTLLM_K3_LATENT_AR_OWN_WS: the routed latent all-reduce's own MNNVL workspace (``None``: it
    # shares the one of the shared expert's all-reduce).
    _latent_ar_ws: Optional[Dict[str, Any]] = None
    # Packed MoE front: (packed [router | latent_down | shared gate_up] weight, row counts), and
    # the shared gate_up output of the current forward (``None``: off / not produced by the packed GEMM).
    _packed_front: Optional[Tuple[torch.Tensor, Tuple[int, int, int]]] = None
    _packed_shared_gate_up: Optional[torch.Tensor] = None
    # TRTLLM_K3_LATENT_AR_SIDE_CLEAR: recorded once latent_up is issued, and the stream that then
    # re-arms the latent all-reduce's Lamport buffer (the latent-down stream; ``None``: no side clear).
    moe_lu_event: Optional[torch.cuda.Event] = None
    latent_ar_clear_stream: Optional[torch.cuda.Stream] = None
    # TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld: the previous MoE layer's ``moe_lu_event`` when this layer's
    # forward is to re-arm the workspace after its latent_down (set by that layer, taken by this one).
    _latent_ar_clear_pending: Optional[torch.cuda.Event] = None
    # Whether the routed backend joins the latent-down stream into the main stream before FC2
    # (``None``: not checked yet).
    _latent_stream_joined_before_fc2: Optional[bool] = None
    # TRTLLM_K3_ROUTER_EXACT: whether this layer's router logits come from k3_router_exact (set
    # per layer in ``__init__``; off unless the switch selects the layer).
    _router_exact = False
    # TRTLLM_K3_L2PF_V2: the model's decode L2 prefetch controller (``None``: off).
    _l2pf: Optional[L2PrefetchController] = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self._router_exact = _router_exact_for_layer(cfg, layer_idx)
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay fp32; with attention-DP off the gate GEMM runs
        # bf16xbf16 with fp32 accumulate/output (checkpoint stores the gate
        # weight in bf16; saves a per-layer input cast + fp32 splitK pair on
        # the bs1 decode path). Under attention-DP the legacy upcast-to-fp32
        # GEMM is kept: the bf16-input min-latency GEMM's different reduction
        # order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # 3-run bisect on 62b20dd868), and the bs1-latency win is irrelevant
        # at DEP batch sizes. helix_a2a ranks share requests and keep bf16.
        # KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else model_config.mapping.tp_group_shares_requests
        )
        self.gate = KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config.quant_algo)
        routed_moe_kwargs = dict(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            # Kimi owns the latent reduction so it can order that collective
            # after the shared expert's auxiliary-stream reduction.
            reduce_results=False,
            # Lets a FUSED_COMM backend return its globally combined routed
            # output; ``_reduce_routed_output`` then skips that reduction.
            accepts_global_routed_output=True,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
            activation=SiTuActivation(
                gate_softcap=situ_beta,
                linear_softcap=situ_linear_beta,
            ),
            # A request that silently degraded to CUTLASS would be benchmarked
            # as if it were the backend that was asked for, and the decline is
            # easy to trigger: MegaMoE has its own token / top-k limits and is
            # EP-only, and CuteDSL declines on activation shape, SM version and
            # the CuTe DSL dependency. Measured 2026-09-08: a CUTEDSL request
            # was turned down on every one of the 92 MoE layers, on all 16
            # ranks, and still produced correct text and a zero exit -- the
            # only trace was a warning line per layer. Fail in the resolver
            # instead, which reports the rejection trail.
            #
            # CUTLASS is absent on purpose: it is the fallback target, so
            # "degraded to CUTLASS" is not a thing that can happen to it.
            allow_backend_degradation=routed_moe_model_config.moe_backend
            not in ("MEGAMOE_DEEPGEMM", "MEGAMOE_CUTEDSL", "CUTEDSL", "CUTEDSL_FC12"),
        )
        self._check_trtllm_situ_quant(
            routed_moe_model_config.moe_backend, routed_quant_config.quant_algo
        )
        self.routed_experts = create_moe(**routed_moe_kwargs)
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
            )
        if self.routed_experts.layer_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "dynamic EPLB or replicated expert slots."
            )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank
        # TRTLLM_K3_EXPERT_PLACEMENT: routed-expert id (slot) s holds checkpoint expert
        # expert_of_slot[s]; ``load_weights`` loads the slots accordingly and permutes the router
        # into slot order. ``ckpt_expert_ids`` names the checkpoint expert of each local slot.
        self.expert_of_slot = expert_placement_for_layer(
            layer_idx, self.num_experts, routed_moe_model_config.mapping.moe_ep_size
        )
        self.ckpt_expert_ids = (
            self.local_expert_ids
            if self.expert_of_slot is None
            else tuple(self.expert_of_slot[slot] for slot in self.local_expert_ids)
        )
        moe_mapping = routed_moe_model_config.mapping
        pin_layout = (
            cfg,
            moe_mapping.moe_tp_size,
            model_config.mapping.enable_attention_dp,
            self.num_experts // moe_mapping.moe_ep_size,
        )
        tactic_pins = _k31_moe_tactic_pins(*pin_layout)
        row_tactics = _k31_moe_row_tactics(*pin_layout)
        pin_tactics = getattr(self.routed_experts.backend, "pin_decode_tactics", None)
        if tactic_pins is not None and pin_tactics is not None:
            if row_tactics:
                pin_tactics(tactic_pins, row_tactics=row_tactics)
            else:
                pin_tactics(tactic_pins)

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        self._shared_intermediate = shared_intermediate
        attention_dp = model_config.mapping.enable_attention_dp
        shared_model_config = copy.copy(model_config)
        shared_model_config.quant_config = QuantConfig()
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        # Fused-comm MegaMoE runs its own cross-rank exchange inside the
        # kernel. Order the shared-expert all-reduce after the routed MoE
        # instead of running the two collectives concurrently (startup
        # deadlocks). CuteDSL is the validated scope, not a correctness
        # dependency: DeepGEMM rejects TP without attention DP.
        self._defer_shared_allreduce = (
            use_shared_tp
            and routed_moe_model_config.moe_backend == "MEGAMOE_CUTEDSL"
            and self.routed_experts.backend.scheduler_kind == MoESchedulerKind.FUSED_COMM
        )
        self._reduce_routed_output = (
            use_shared_tp and not self.routed_experts.returns_global_routed_output
        )
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
            )
        # TRTLLM_K3_MERGED_MOE_AR: the shared expert hands back its row-parallel partial and the
        # routed reduction sums it together with the latent partial (``_merged_reduce``). The
        # shared partial is reduced as rows of the latent width, so the hidden size must be a
        # multiple of it.
        self._merge_moe_allreduce = (
            _K3_MERGED_MOE_AR
            and self._reduce_routed_output
            and cfg.hidden_size % self.moe_hidden_size == 0
        )
        self.shared_experts = GatedMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=shared_intermediate,
            bias=False,
            activation=SituAndMul(
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                use_fused_activation=True,
            ),
            dtype=dtype,
            config=shared_model_config,
            overridden_tp_size=1 if attention_dp else None,
            reduce_output=use_shared_tp,
            layer_idx=layer_idx,
            is_shared_expert=True,
        )
        shared_ar = getattr(self.shared_experts.down_proj, "all_reduce", None)
        if not _K3_SHARED_AR_NVLS_V2 and getattr(shared_ar, "nvls_allreduce", None) is not None:
            shared_ar.nvls_allreduce.version = 1
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        self.moe_routed_event = torch.cuda.Event() if self._defer_shared_allreduce else None
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        # The latent_down stream (K3.1 TEP gen) does the same for the TRTLLM-Gen backend.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if routed_moe_model_config.moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            or _k31_latent_down_stream_requested(routed_moe_model_config.moe_backend)
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        # TRTLLM_K3_SHARED_FIRST: shared GEMMs (and latent chain) done; routed FC1 done.
        self.moe_shared_gemm_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_fc1_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # Stock fused RMSNorm (flashinfer kernel; the no-flashinfer
        # fallback is the same fp32-variance eager math as KimiK3RMSNorm).
        self.routed_expert_norm = RMSNorm(
            hidden_size=self.moe_hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        # TRTLLM_K3_MOE_L2_PREFETCH: the MoE layers of this model by index, and the event after the
        # routed FC2 that starts this layer's prefetch on the latent-down stream.
        self._moe_group = _K3_MOE_LAYERS.setdefault(id(cfg), weakref.WeakValueDictionary())
        self._moe_group[layer_idx] = self
        self._layers_key = id(cfg)
        self._l2_ranges_cache: Dict[str, List[Tuple[int, int]]] = {}
        self.moe_fc2_event = torch.cuda.Event() if has_latent_stream else None
        if _K3_LATENT_AR_SIDE_CLEAR != "0" and has_latent_stream:
            self.moe_lu_event = torch.cuda.Event()
            self.latent_ar_clear_stream = self.latent_down_stream
        if _K3_LATENT_AR_OWN_WS and self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )
        self._decode_gemm_plan = kimi_k31_decode_plan(cfg)
        # Sharded MoE front: the head workspace of the routed all-reduce's TP group (collective:
        # every rank builds it here; one workspace serves every layer).
        self._k31f_ws = None
        self._k31f_weight = None
        self._k31f_shared = None
        self._k31f_shared_act = None
        if (
            _K31_SHARDED_MOE_FRONT
            and self._reduce_routed_output
            and getattr(cfg, "dsa_config", None) is not None
            and self._decode_gemm_plan is not None
        ):
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size in (4, 8, 16, 32):
                from ..moe.kernels.blackwell import k31_moe_front

                self._k31f_ws = k31_moe_front.head_workspace(mnnvl.mapping)
        # KIMI_K31_ROUTER_SORT: Kimi K3.1 only (the layers of a K3 config keep their routing).
        self._k31_router_sort_wanted = (
            _K31_ROUTER_SORT == "1" and getattr(cfg, "dsa_config", None) is not None
        )
        self._k31_router_tail_dep_wanted = (
            _K31_ROUTER_TAIL_DEP and getattr(cfg, "dsa_config", None) is not None
        )
        # The fused tail replaces exactly the direct MoE-TP tail below: routed
        # partials reduced here (not by a fused-comm backend) and the shared
        # all-reduce not deferred behind one.
        self._fused_tail_mapping = (
            self.routed_experts.all_reduce.mapping
            if self._reduce_routed_output and not self._defer_shared_allreduce
            else None
        )
        self._fused_tail_wanted = self._fused_tail_mapping is not None and _K31_FUSED_MOE_TAIL
        if self._fused_tail_wanted:
            reason = fused_tail_unsupported_reason(
                self._fused_tail_mapping,
                cfg.hidden_size,
                self.moe_hidden_size,
                torch.device("cuda", torch.cuda.current_device()),
            )
            if reason is not None:
                logger.info_once(
                    f"Kimi K3 fused latent-MoE tail disabled: {reason}",
                    key="kimi_k3_fused_moe_tail_disabled",
                )
                self._fused_tail_wanted = False
        self._fused_tail: Optional[KimiK3LatentMoETail] = None

    def _select_fused_tail(self, hidden_states: torch.Tensor) -> Optional[KimiK3LatentMoETail]:
        """The fused tail op for this call, or ``None`` for the unfused tail.

        The op (shared by every layer) is built on the first eager forward;
        that is collective over the TP group, which runs every forward in
        lockstep, so every rank takes the same branch here.
        """
        if not self._fused_tail_wanted or torch.compiler.is_compiling():
            return None
        up = self.routed_expert_up_proj
        # The FP8 weight-read conversion replaces the BF16 up-projection.
        if not isinstance(up, nn.Linear) or up.weight.dtype != torch.bfloat16:
            return None
        if self._fused_tail is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._fused_tail = KimiK3LatentMoETail.get(
                self._fused_tail_mapping,
                hidden_size=self.hidden_size,
                latent_size=self.moe_hidden_size,
                rms_eps=float(self.routed_expert_norm.variance_epsilon),
            )
        if not self._fused_tail.can_run(hidden_states.shape[0]):
            return None
        return self._fused_tail

    def _next_moe_layer(self) -> Optional["KimiK3MoERuntime"]:
        """The MoE runtime of the next MoE layer of this model, if any."""
        later = [idx for idx in self._moe_group.keys() if idx > self.layer_idx]
        return self._moe_group.get(min(later)) if later else None

    def l2_prefetch_ranges(self, what: str) -> List[Tuple[int, int]]:
        """``(address, bytes)`` of the weights a decode MoE block streams from HBM: ``"lu"`` the
        latent_up weight and scale; ``"front"`` the router, latent_down and shared-expert weights
        and scales (the front end of the block)."""
        ranges = self._l2_ranges_cache.get(what)
        if ranges is None:
            from ..kernels.blackwell.k3_l2_prefetch import MAX_RANGES, tensor_range

            if what == "lu":
                modules = [self.routed_expert_up_proj]
                tensors = []
            else:
                modules = [
                    self.routed_expert_down_proj,
                    self.shared_experts.gate_up_proj,
                    self.shared_experts.down_proj,
                ]
                tensors = [self.gate.weight]
            for module in modules:
                tensors += [getattr(module, "weight", None), getattr(module, "weight_scale", None)]
            ranges = []
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.numel():
                    try:
                        ranges.append(tensor_range(tensor))
                    except ValueError:
                        continue
            ranges = ranges[:MAX_RANGES]
            self._l2_ranges_cache[what] = ranges
        return ranges

    def _issue_l2_prefetch(self, num_tokens: int) -> None:
        """``TRTLLM_K3_MOE_L2_PREFETCH`` / ``TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA``: once the routed
        FC2 is done, prefetch into L2 on the latent-down stream. Only when a later MoE layer exists:
        its latent_down follows on this stream and is joined into the main stream, so the prefetch
        needs no join of its own (a join in front of a programmatic dependent would cost it its
        early launch)."""
        if (
            not (_K3_MOE_L2_PREFETCH or _K3_MOE_L2_PREFETCH_NEXT_MLA)
            or num_tokens > _K3_MOE_L2_PREFETCH_MAX_M
            or self.latent_down_stream is None
            or self.moe_fc2_event is None
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return
        nxt = self._next_moe_layer()
        if nxt is None or nxt.latent_down_stream is not self.latent_down_stream:
            return
        mla_ranges = []
        if _K3_MOE_L2_PREFETCH_NEXT_MLA:
            next_mla = _K3_MLA_LAYERS.get(getattr(self, "_layers_key", None), {}).get(
                self.layer_idx + 1
            )
            if next_mla is not None:
                mla_ranges = next_mla.q_b_l2_prefetch_ranges()
        if not _K3_MOE_L2_PREFETCH and not mla_ranges:
            return
        from ..kernels.blackwell.k3_l2_prefetch import run_k3_l2_prefetch

        self.moe_fc2_event.record()
        with torch.cuda.stream(self.latent_down_stream):
            self.moe_fc2_event.wait()
            if "lu" in _K3_MOE_L2_PREFETCH:
                run_k3_l2_prefetch(self.l2_prefetch_ranges("lu"))
            if "front" in _K3_MOE_L2_PREFETCH:
                run_k3_l2_prefetch(nxt.l2_prefetch_ranges("front"))
            if mla_ranges:
                run_k3_l2_prefetch(mla_ranges)

    def _issue_latent_ar_clear(self, num_tokens: int) -> None:
        """``TRTLLM_K3_LATENT_AR_SIDE_CLEAR``: once latent_up is issued, have the Lamport buffer of
        the latent all-reduce that just ran re-armed on the latent-down stream
        (``run_k3_lamport_clear``), which also marks the workspace state clean, so the next latent
        all-reduce on the workspace skips its own clear. Only when a later MoE layer exists: its
        latent_down runs on this stream and is joined into the main stream before its routed GEMMs
        (and with them that layer's all-reduce). The next call on the workspace writes into the
        cleared buffer only after its push of the call after it (three buffers).

        "1": the clear is issued now, ahead of the next layer's latent_down on that stream. "ld":
        the next layer issues it right after its latent_down (``_issue_pending_latent_ar_clear``),
        where that layer's output memset carries it into the main stream before FC2."""
        workspace = self._latent_ar_ws
        stream = self.latent_ar_clear_stream
        if (
            self.moe_lu_event is None
            or stream is None
            or workspace is None
            or num_tokens > workspace["max_tokens"]
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return
        nxt = self._next_moe_layer()
        if nxt is None or nxt.latent_down_stream is not stream:
            return
        self.moe_lu_event.record()
        if (
            _K3_LATENT_AR_SIDE_CLEAR == "ld"
            and nxt._latent_ar_ws is workspace
            and nxt._latent_stream_joins_before_fc2()
        ):
            nxt._latent_ar_clear_pending = self.moe_lu_event
            return
        with torch.cuda.stream(stream):
            self._run_latent_ar_clear(self.moe_lu_event)

    def _run_latent_ar_clear(self, after: torch.cuda.Event) -> None:
        """On the current stream: wait for ``after`` (a latent_up of this workspace), then re-arm
        the workspace's dirty Lamport buffer and mark the state clean."""
        from ..kernels.blackwell.k3_latent_ar_norm import run_k3_lamport_clear

        workspace = self._latent_ar_ws
        after.wait()
        run_k3_lamport_clear(
            workspace["buffer"].data_ptr(), workspace["flags"], workspace["clear_ctr"]
        )

    def _latent_stream_joins_before_fc2(self) -> bool:
        """Whether the routed backend forks its output memset onto the latent-down stream after the
        routing and joins it into the main stream before FC2 (``CuteDslFusedMoE`` NVFP4 with the
        fused finalize and its MoeOutputMemset aux stream): then work issued on that stream after
        latent_down completes before this layer's all-reduce."""
        if self._latent_stream_joined_before_fc2 is None:
            from ..moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE

            backend = getattr(self.routed_experts, "backend", None)
            joined = (
                isinstance(backend, CuteDslFusedMoE)
                and bool(getattr(backend, "has_nvfp4", False))
                and bool(getattr(backend, "use_fused_finalize", False))
                and backend._has_moe_output_memset_aux_stream()
                and backend._moe_output_memset_run_stream() is self.latent_down_stream
            )
            self._latent_stream_joined_before_fc2 = joined
            logger.info_once(
                "TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld: "
                + (
                    "clears issued after the next layer's latent_down"
                    if joined
                    else "no output-memset join before FC2 on the latent-down stream, "
                    "clears issued after latent_up as with 1"
                ),
                key=f"k3_latent_ar_side_clear_ld_{joined}",
            )
        return self._latent_stream_joined_before_fc2

    def _issue_pending_latent_ar_clear(self, pending: Optional[torch.cuda.Event]) -> None:
        """``TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld``, on the latent-down stream right after this layer's
        latent_down: the clear the previous MoE layer left to this one (``pending``: its
        ``moe_lu_event``). It reaches the main stream through this layer's output memset (see
        ``_latent_stream_joins_before_fc2``), i.e. before this layer's all-reduce."""
        if pending is not None:
            self._run_latent_ar_clear(pending)

    def _join_expert_l2_prefetch(self) -> None:
        """``TRTLLM_K3_L2PF_V2``: in the last MoE layer, join the decode prefetch streams into the
        current (main) stream. Called right where the main stream joins the shared-expert stream;
        the last layer's prefetches were all issued in its attention block."""
        if self._l2pf is not None and self._l2pf.pending and self._next_moe_layer() is None:
            self._l2pf.join()

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits; ``TRTLLM_K3_ROUTER_EXACT`` serves ``M <= 16`` BF16 tokens with
        ``k3_router_exact`` (bit-identical to the ``dsv3_router_gemm_op`` it replaces) in the
        layers it selects."""
        if self._router_exact and not torch.compiler.is_compiling():
            from ..moe.kernels.blackwell import k3_router_exact

            x = hidden_states.reshape(-1, self.gate.gating_dim)
            if k3_router_exact.can_run(x, self.gate.weight):
                return torch.ops.trtllm.k3_router_logits(x, self.gate.weight)
        return self.gate.compute_logits(hidden_states)

    def _shared_gemms_can_lead(self) -> bool:
        """Whether this forward can issue the shared expert's GEMMs ahead of the routed experts
        (``_shared_gemms_first``) and its all-reduce separately."""
        return (
            self.shared_expert_stream is not None
            and self.moe_shared_gemm_event is not None
            and not self._defer_shared_allreduce
            and not self._merge_moe_allreduce
            and not getattr(self.shared_experts.down_proj, "use_fused_gemm_allreduce", False)
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )

    def _shared_first_applies(self, num_tokens: int) -> bool:
        """Whether ``TRTLLM_K3_SHARED_FIRST`` orders this forward's shared expert before FC1."""
        return (
            _K3_SHARED_FIRST != "0"
            and _K3_SHARED_FIRST_MIN_M <= num_tokens <= _K3_SHARED_FIRST_MAX_M
            and hasattr(self.routed_experts.backend, "post_fc1_event")
            and self._shared_gemms_can_lead()
        )

    def _tail_waits_for_shared(self, num_tokens: int, num_local: int) -> bool:
        """Whether ``TRTLLM_K3_ROUTE_TAIL_ORDER`` runs the routed experts of a forward that the
        routing tail routes (``num_local`` experts on this rank) after the shared expert's GEMMs,
        with the routed input quantized on the latent chain."""
        rows = num_tokens * self.top_k * num_local // self.num_experts
        if _K3_ROUTE_TAIL_ORDER == "0" or rows < _K3_ROUTE_TAIL_ORDER_MIN_ROWS:
            return False
        backend = self.routed_experts.backend
        return (
            getattr(backend, "has_nvfp4", False)
            and getattr(backend, "fc31_input_scale", None) is not None
            and self._shared_gemms_can_lead()
        )

    def _shared_gemms_first(
        self, hidden_states: torch.Tensor, fork_event: torch.cuda.Event
    ) -> torch.Tensor:
        """``TRTLLM_K3_SHARED_FIRST`` / ``TRTLLM_K3_ROUTE_TAIL_ORDER``: issue the shared expert's
        GEMMs (without its all-reduce) on the shared stream after ``fork_event``;
        ``moe_shared_gemm_event`` then marks them and the latent chain done, which is what the
        routed experts wait for. Returns the partial."""
        with torch.cuda.stream(self.shared_expert_stream):
            fork_event.wait()
            partial = self.shared_experts(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
            self.moe_latent_ready_event.wait()
            self.moe_shared_gemm_event.record()
        return partial

    def _shared_needs_all_reduce(self) -> bool:
        """Whether the shared expert's output is a row-parallel partial to all-reduce."""
        return bool(getattr(self.shared_experts.down_proj, "reduce_output", False))

    def _shared_all_reduce(self) -> nn.Module:
        """The shared expert's row-parallel reduction."""
        return self.shared_experts.down_proj.all_reduce

    def post_load_weights(self) -> None:
        self.prepare_packed_front()
        self.prepare_k31_sharded_front()

    def cache_derived_state(self) -> None:
        self.prepare_packed_front()
        self.prepare_k31_sharded_front()

    def _k31_sharded_front_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the sharded MoE front, or ``None``."""
        if self._k31f_ws is None:
            return (
                "no MNNVL routed all-reduce over a TP group of 4, 8, 16 or 32 ranks (Kimi K3.1 TEP)"
            )
        moe = self.routed_experts
        if moe.use_dp or moe.comm is not None:
            return "attention DP or a dispatching MoE"
        if (
            not isinstance(moe.routing_method, _KimiK3RoutingMethod)
            or moe.apply_router_weight_on_input
        ):
            return "the routing is not the Kimi-K3 routing method"
        if not getattr(moe.backend, "accepts_mxfp8_input", False):
            return "the routed experts do not take an MXFP8 input"
        if (
            _K3_FUSED_ROUTER != "0"
            or _K3_ROUTE_TAIL != "0"
            or self._router_exact
            or self._merge_moe_allreduce
        ):
            return "another opt-in MoE front / tail path is on"
        down = self.routed_expert_down_proj
        if (
            self.gate.weight.dtype != torch.bfloat16
            or getattr(down, "weight", None) is None
            or down.weight.dtype != torch.bfloat16
            or getattr(down, "bias", None) is not None
            or getattr(down, "has_any_quant", False)
        ):
            return "the router or latent_down is not a plain BF16 projection"
        from ..moe.kernels.blackwell import k31_moe_front

        world = self._k31f_ws["world"]
        if (
            tuple(self.gate.weight.shape) != (k31_moe_front.NUM_EXPERTS, self.hidden_size)
            or tuple(down.weight.shape) != (k31_moe_front.HIDDEN_SIZE, self.hidden_size)
            or self.top_k != k31_moe_front.TOP_K
        ):
            return "not the 896-expert top-16 latent MoE of 3584 latent columns"
        if not k31_moe_front.weight_supported(world, 0, self.hidden_size, self.gate.weight.device):
            return f"the front kernel does not serve TP {world} / hidden {self.hidden_size} on this GPU"
        return None

    def prepare_k31_sharded_front(self) -> bool:
        """Sharded MoE front: copy this rank's head rows (its latent_down and router slices) into
        the front's weight (128-row tiles; 5.5 MB per layer at TP16). Runs post-load, before CUDA graph capture."""
        self._k31f_weight = None
        if not _K31_SHARDED_MOE_FRONT or self._k31f_ws is None:
            return False
        reason = self._k31_sharded_front_unsupported()
        if reason is not None:
            logger.info_once(
                f"Kimi K3.1 sharded MoE front: MoE layers keep the replicated front: {reason}",
                key=f"k31_sharded_front_off_{reason}",
            )
            return False
        from ..moe.kernels.blackwell import k31_moe_front

        ws = self._k31f_ws
        shared = self._k31_sharded_front_shared_operands()
        self._k31f_shared = shared
        with torch.no_grad():
            self._k31f_weight = k31_moe_front.front_weight(
                k31_moe_front.head_weight(
                    self.routed_expert_down_proj.weight, self.gate.weight, ws["rank"], ws["world"]
                ),
                shared[0] if shared is not None else None,
            )
        if shared is not None:
            logger.info_once(
                f"Kimi K3.1 sharded MoE front (shared expert): the front computes the shared expert's gate_up + SiTU "
                f"({shared[1]} columns per rank) up to {_K31_SHARDED_MOE_FRONT_SHARED_MAX_M} rows",
                key="k31_sharded_front_shared_on",
            )
        logger.info_once(
            f"Kimi K3.1 sharded MoE front: sharded MoE front over {ws['world']} ranks up to "
            f"{min(_K31_SHARDED_MOE_FRONT_MAX_M, k31_moe_front.MAX_ROWS)} rows",
            key="k31_sharded_front_on",
        )
        return True

    def _k31_sharded_front_shared_operands(
        self,
    ) -> Optional[Tuple[torch.Tensor, int, float, float]]:
        """``(gate_up weight, shared columns, SiTU beta, SiTU linear beta)`` of this rank's shared expert if the front
        computes its gate_up + SiTU (the sharded front's shared-expert part), else ``None``."""
        if not _K31_SHARDED_MOE_FRONT_SHARED:
            return None
        mlp = self.shared_experts
        gate_up = getattr(mlp, "gate_up_proj", None)
        act = getattr(mlp, "activation", None)
        weight = getattr(gate_up, "weight", None)
        reason = None
        if not isinstance(mlp, GatedMLP) or weight is None or not isinstance(act, SituAndMul):
            reason = "the shared expert is not a gated SiTU MLP with a fused gate_up"
        elif (
            weight.dtype != torch.bfloat16
            or getattr(gate_up, "bias", None) is not None
            or getattr(gate_up, "has_any_quant", False)
            or weight.dim() != 2
            or weight.shape[1] != self.hidden_size
            or weight.shape[0] % 128
        ):
            reason = "the shared gate_up is not a plain BF16 projection of whole 128-row tiles"
        elif act.linear_beta is None:
            reason = "the SiTU has no linear beta"
        elif (
            getattr(mlp, "split_gate_up", False)
            or mlp._can_fuse_gate_up_swiglu()
            or mlp._can_fuse_swiglu_fp8_quant()
        ):
            reason = "the shared expert fuses its gate_up GEMM"
        if reason is not None:
            logger.info_once(
                f"Kimi K3.1 sharded MoE front (shared expert): the shared expert keeps its own gate_up: {reason}",
                key=f"k31_sharded_front_shared_off_{reason}",
            )
            return None
        return weight, weight.shape[0] // 2, float(act.beta), float(act.linear_beta)

    def _k31_sharded_front_shared_cols(self, num_tokens: int) -> int:
        """Shared columns the front computes for a ``num_tokens``-row forward (0: the shared expert runs its own
        gate_up)."""
        shared = getattr(self, "_k31f_shared", None)
        if shared is None or num_tokens > _K31_SHARDED_MOE_FRONT_SHARED_MAX_M:
            return 0
        return shared[1]

    def k31_sharded_front_weight(self, num_tokens: int) -> Optional[torch.Tensor]:
        """The rows of the sharded MoE front's weight a ``num_tokens``-row forward reads (the head, plus the shared
        gate_up rows where the front computes them), or ``None`` if the front does not serve those forwards."""
        if getattr(self, "_k31f_weight", None) is None:
            return None
        from ..moe.kernels.blackwell import k31_moe_front

        if not 1 <= num_tokens <= min(_K31_SHARDED_MOE_FRONT_MAX_M, k31_moe_front.MAX_ROWS):
            return None
        if self._k31_sharded_front_shared_cols(num_tokens):
            return self._k31f_weight
        world = self._k31f_ws["world"]
        return self._k31f_weight[: k31_moe_front.head_tiles(world) * k31_moe_front.CTA_M]

    def k31_sharded_front_reads_shared_gate_up(self, num_tokens: int) -> bool:
        """Whether a ``num_tokens``-row forward's sharded front computes the shared expert's gate_up."""
        return (
            self.k31_sharded_front_weight(num_tokens) is not None
            and self._k31_sharded_front_shared_cols(num_tokens) > 0
        )

    def _k31_sharded_front_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this forward runs the sharded MoE front."""
        if self._k31f_weight is None or torch.compiler.is_compiling():
            return False
        from ..moe.kernels.blackwell import k31_moe_front

        m = hidden_states.shape[0]
        if (
            hidden_states.dim() != 2
            or hidden_states.dtype != torch.bfloat16
            or not 1 <= m <= min(_K31_SHARDED_MOE_FRONT_MAX_M, k31_moe_front.MAX_ROWS)
        ):
            return False
        ws = self._k31f_ws
        # The kernel compiles on its first call, which must not fall into a CUDA graph capture.
        shared_cols = self._k31_sharded_front_shared_cols(m)
        shared = self._k31f_shared if shared_cols else None
        return not torch.cuda.is_current_stream_capturing() or k31_moe_front.is_compiled(
            m, self.hidden_size, ws["world"], shared_cols, hidden_states.device,
            gate_cap=shared[2] if shared else 1.0, linear_cap=shared[3] if shared else 1.0,
            do_sort=self._k31_sharded_front_sort(m) is not None,
        )  # fmt: skip

    def _k31_sharded_front_sort(self, num_tokens: int) -> Optional[Tuple[int, int, int, int]]:
        """``(tile, local_expert_offset, local_num_experts, num_experts)`` if the sharded front also writes the
        TRTLLM-Gen MXFP4 runner's routing (``KIMI_K31_ROUTER_SORT``'s tables, for finalized and unfinalized
        routed outputs alike), else ``None`` (the runner routes from the top-16)."""
        from ..moe.kernels.blackwell import k31_moe_front

        if not _K31_SHARDED_MOE_FRONT_SORT:
            return None
        targets = [self._k31_runner_sort_target(num_tokens, f) for f in (True, False)]
        if targets[0] is None or targets[0][:4] != (targets[1] or (None,))[:4]:
            return None
        tile, slot_start, num_local, num_slots, _ = targets[0]
        if not k31_moe_front.runner_sort_fits(num_tokens, num_local, tile):
            return None
        return tile, slot_start, num_local, num_slots

    def _k31_sharded_front(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, MxFp8QuantizedTensor]:
        """Run the sharded MoE front: hands the top-16 (and, where it applies, the runner's routing) to the routing
        method and returns ``(router_logits placeholder, MXFP8 routed input)``."""
        from ..moe.kernels.blackwell import k31_moe_front

        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        sort = self._k31_sharded_front_sort(x.shape[0])
        shared_cols = self._k31_sharded_front_shared_cols(x.shape[0])
        caps = self._k31f_shared[2:] if shared_cols else (1.0, 1.0)
        topk_ids, topk_weights, quantized, scales, shared_act, prerouted = (
            k31_moe_front.run_k31_moe_front(
                x,
                self._k31f_weight,
                self.gate.e_score_correction_bias,
                float(self.gate.routed_scaling_factor),
                self._k31f_ws,
                shared_cols=shared_cols,
                gate_cap=caps[0],
                linear_cap=caps[1],
                sort=sort,
            )
        )
        # Sharded front's shared-expert part: the shared expert continues from this activation.
        self._k31f_shared_act = shared_act
        routing = self.routed_experts.routing_method
        routing.precomputed = (topk_ids, topk_weights)
        if prerouted is not None:
            routing.presorted = (prerouted, sort[0], topk_ids)
        # The routing method returns the precomputed top-k; the logits only carry the shape.
        logits = torch.empty(
            (x.shape[0], self.gate.num_experts), dtype=torch.float32, device=x.device
        )
        return logits, MxFp8QuantizedTensor(quantized, scales)

    def _packed_front_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the packed MoE front, or ``None``."""
        if self._decode_gemm_plan is None:
            return "not a Kimi K3.1 MoE layer on SM100 / SM103"
        if self.routed_experts.use_dp:
            return "attention DP"
        if (
            _K3_FUSED_ROUTER != "0"
            or _K3_ROUTE_TAIL != "0"
            or self._router_exact
            or self._merge_moe_allreduce
        ):
            return "another opt-in MoE front / tail path is on"

        def plain_bf16(linear) -> bool:
            return (isinstance(linear, nn.Linear) or type(linear).__name__ == "Linear") and (
                getattr(linear, "bias", None) is None
                and linear.weight.dtype == torch.bfloat16
                and not getattr(linear, "has_any_quant", False)
                and getattr(linear, "lora", None) is None
            )

        mlp = self.shared_experts
        if self.gate.weight.dtype != torch.bfloat16:
            return "the router weight is not BF16"
        if not plain_bf16(self.routed_expert_down_proj) or not plain_bf16(mlp.gate_up_proj):
            return "latent_down or the shared gate_up is not a plain BF16 projection"
        if (
            mlp.split_gate_up
            or mlp._can_fuse_gate_up_swiglu()
            or mlp._can_fuse_swiglu_fp8_quant()
            or getattr(mlp, "fuse_bf16_gate_up_swiglu", False)
        ):
            return "the shared expert fuses its gate_up GEMM"
        from ..kernels.blackwell import tgv_multi_out_gemm

        if not tgv_multi_out_gemm.IS_AVAILABLE:
            return "the TGV CuTe DSL kernel is not available"
        cta_m = min(tgv_multi_out_gemm.tactic_cta_m(t) for _, t in _K31_PACKED_MOE_FRONT_TACTICS)
        rows = (
            self.gate.weight.shape[0],
            self.routed_expert_down_proj.weight.shape[0],
            mlp.gate_up_proj.weight.shape[0],
        )
        if (
            any(r % cta_m for r in rows)
            or len({w.shape[1] for w in self._packed_front_weights()}) != 1
        ):
            return "the projections do not tile into the packed GEMM"
        return None

    def _packed_front_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.gate.weight,
            self.routed_expert_down_proj.weight,
            self.shared_experts.gate_up_proj.weight,
        )

    def prepare_packed_front(self) -> bool:
        """Packed MoE front: pack the router, latent_down and shared gate_up weights into one
        row-major tensor; the three modules' weights become its row views (no memory added). Runs post-load,
        before CUDA graph capture."""
        self._packed_front = None
        self._packed_shared_gate_up = None
        if not _K31_PACKED_MOE_FRONT:
            return False
        reason = self._packed_front_unsupported()
        if reason is not None:
            logger.info_once(
                f"Kimi K3.1 packed MoE front: MoE layers keep the separate front GEMMs: {reason}",
                key=f"k31_packed_front_off_{reason}",
            )
            return False
        weights = self._packed_front_weights()
        rows = tuple(int(w.shape[0]) for w in weights)
        base = weights[0]
        offsets = (0, rows[0], rows[0] + rows[1])
        already = all(
            w.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
            and w.storage_offset() == base.storage_offset() + off * w.shape[1]
            for w, off in zip(weights, offsets)
        )
        if already:
            packed = torch.as_strided(
                base, (sum(rows), base.shape[1]), base.stride(), base.storage_offset()
            )
        else:
            with torch.no_grad():
                packed = torch.cat([w.data for w in weights], dim=0)
                for w, off, r in zip(weights, offsets, rows):
                    w.data = packed[off : off + r]
        self._packed_front = (packed, rows)
        logger.info_once(
            f"Kimi K3.1 packed MoE front: router + latent_down + shared gate_up as one GEMM {rows} x "
            f"{packed.shape[1]}",
            key="k31_packed_front_on",
        )
        return True

    def _packed_front_tactic(self, hidden_states: torch.Tensor) -> Optional[int]:
        """The packed MoE front's TGV tactic for this forward, or ``None`` to keep the separate GEMMs."""
        if getattr(self, "_packed_front", None) is None or torch.compiler.is_compiling():
            return None
        m = hidden_states.shape[0]
        if (
            hidden_states.dim() != 2
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or not max(1, _K31_PACKED_MOE_FRONT_MIN_M) <= m <= _K31_PACKED_MOE_FRONT_MAX_M
        ):
            return None
        return next((t for bound, t in _K31_PACKED_MOE_FRONT_TACTICS if m <= bound), None)

    def _packed_front_forward(
        self, hidden_states: torch.Tensor, tactic: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """(FP32 router logits, latent_down output, shared gate_up output) from the packed GEMM, or ``None``."""
        from ..flashinfer_utils import get_env_enable_pdl
        from ..kernels.blackwell import tgv_multi_out_gemm

        packed, rows = self._packed_front
        outs = tgv_multi_out_gemm.multi_out_linear(
            hidden_states,
            packed,
            rows,
            (torch.float32, hidden_states.dtype, hidden_states.dtype),
            tactic,
            get_env_enable_pdl(),
        )
        return None if outs is None else tuple(outs)

    def _merged_reduce_buffers(
        self, hidden_states: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(buffer, zero_residual)`` of the merged MoE all-reduce, or ``None`` to keep two.

        ``buffer`` is flat: the routed latent partial ``[M, moe_hidden_size]`` followed by the
        shared-expert partial ``[M, hidden_size]``, both contiguous.
        """
        num_tokens = hidden_states.shape[0]
        if (
            not self._merge_moe_allreduce
            or num_tokens == 0
            or 0 < _K3_MERGED_MOE_AR_MAX_TOKENS < num_tokens
            or hidden_states.dtype != torch.bfloat16
            or torch.compiler.is_compiling()
        ):
            return None
        numel = num_tokens * (self.moe_hidden_size + self.hidden_size)
        zero_residual = _k3_zero_residual(numel, hidden_states.device)
        if zero_residual is None:
            return None
        return torch.empty_like(zero_residual), zero_residual

    def _merged_reduce(
        self, buffer: torch.Tensor, zero_residual: torch.Tensor, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``[routed latent partial | shared partial]`` in one collective.

        The buffer is viewed as rows of the latent width (a shared row spans
        ``hidden_size / moe_hidden_size`` of them) and reduced with the residual-RMSNorm fusion
        against a zero residual: the residual output is the plain sum of every row, and the norm
        output of the first ``num_tokens`` rows is ``routed_expert_norm`` of the reduced latent.
        Returns ``(normed latent [M, moe_hidden_size], reduced shared [M, hidden_size])``.
        """
        latent = self.moe_hidden_size
        rows = buffer.numel() // latent
        normed, reduced = self.routed_experts.all_reduce(
            buffer.view(rows, latent),
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=zero_residual.view(rows, latent),
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        return normed[:num_tokens], reduced[num_tokens:].view(num_tokens, self.hidden_size)

    def _fused_route(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Route with the fused CuTe router when enabled and applicable.

        Hands the top-k to the routed experts' routing method and, in ``sort``
        mode, attaches this rank's ``moe_sort`` outputs to the top-k ids (read by
        the CuteDSL backends). Returns the FP32 router logits, or ``None`` when
        the caller must route the regular way.
        """
        if _K3_FUSED_ROUTER == "0" or torch.compiler.is_compiling():
            return None
        routing = self.routed_experts.routing_method
        backend = self.routed_experts.backend
        if not isinstance(routing, _KimiK3RoutingMethod) or self.routed_experts.use_dp:
            return None
        from ..moe.kernels.blackwell import k3_fused_router

        bias = self.gate.e_score_correction_bias
        if not k3_fused_router.can_use_k3_fused_router(hidden_states, self.gate.weight, bias):
            return None
        if torch.cuda.is_current_stream_capturing() and not k3_fused_router.workspace_ready(
            hidden_states.device
        ):
            return None
        scale = float(self.gate.routed_scaling_factor)
        slot_start = getattr(backend, "slot_start", None)
        num_local = getattr(backend, "expert_size_per_partition", None)
        if (
            _K3_FUSED_ROUTER == "sort"
            and slot_start is not None
            and num_local is not None
            and getattr(backend, "num_slots", None) == self.gate.num_experts
        ):
            (topk_ids, topk_weights, *sort_outputs, logits) = torch.ops.trtllm.k3_fused_router_sort(
                hidden_states,
                self.gate.weight,
                bias,
                scale,
                slot_start,
                num_local,
                128,
                True,
            )
            topk_ids._k3_presorted = ((slot_start, num_local, 128), tuple(sort_outputs))
        else:
            topk_ids, topk_weights, logits = torch.ops.trtllm.k3_fused_router_topk(
                hidden_states, self.gate.weight, bias, scale, True
            )
        routing.precomputed = (topk_ids, topk_weights)
        return logits

    def _route_tail_target(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """``(local_expert_offset, local_num_experts)`` if the CuTe routing tail serves this
        forward (``TRTLLM_K3_ROUTE_TAIL``), else ``None``."""
        if _K3_ROUTE_TAIL == "0" or torch.compiler.is_compiling():
            return None
        routing = self.routed_experts.routing_method
        backend = self.routed_experts.backend
        if not isinstance(routing, _KimiK3RoutingMethod) or self.routed_experts.use_dp:
            return None
        slot_start = getattr(backend, "slot_start", None)
        num_local = getattr(backend, "expert_size_per_partition", None)
        num_slots = self.gate.num_experts
        if (
            slot_start is None
            or num_local is None
            or getattr(backend, "num_slots", None) != num_slots
        ):
            return None
        from ..moe.kernels.blackwell import k3_route_tail

        device = self.gate.e_score_correction_bias.device
        if not k3_route_tail.supports_device(device):
            return None
        if not 1 <= num_tokens <= min(_K3_ROUTE_TAIL_MAX_M, k3_route_tail.WIDE_MAX_M):
            return None
        if num_tokens > k3_route_tail.MAX_M:
            if not k3_route_tail.wide_sort_fits(num_tokens, num_local):
                return None
            # The multi-CTA kernel's workspace must exist before graph capture.
            if not torch.cuda.is_current_stream_capturing():
                k3_route_tail.wide_workspace(device)
            elif not k3_route_tail.wide_workspace_ready(device):
                return None
        return slot_start, num_local

    def _can_gemv_route(self, hidden_states: torch.Tensor) -> bool:
        """Whether ``TRTLLM_K3_ROUTE_TAIL=gemv`` computes this forward's router logits."""
        from ..moe.kernels.blackwell import k3_route_tail

        backend = self.routed_experts.backend
        return (
            _K3_ROUTE_TAIL == "gemv"
            and hidden_states.shape[0] <= k3_route_tail.GEMV_ROUTE_MAX_M
            and getattr(backend, "has_nvfp4", False)
            and getattr(backend, "fc31_input_scale", None) is not None
            and k3_route_tail.can_use_k3_route(hidden_states, self.gate.weight)
        )

    def _attach_route(
        self, outs: Tuple[torch.Tensor, ...], target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(topk_ids, topk_weights)`` of a routing-tail result; the ids carry the
        ``moe_sort`` outputs for the CuteDSL backends."""
        topk_ids = outs[0]
        topk_ids._k3_presorted = ((target[0], target[1], 128), tuple(outs[2:8]))
        return topk_ids, outs[1]

    def _tail_route(
        self, logits: torch.Tensor, target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = torch.ops.trtllm.k3_route_tail(
            logits,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            target[0],
            target[1],
            128,
        )
        return self._attach_route(outs, target)

    def _k31_runner_sort_target(
        self, num_tokens: int, do_finalize: bool
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """``(tile, local_expert_offset, local_num_experts, num_experts, wide_from)`` if the CuTe routing tail
        routes this forward for the TRTLLM-Gen MXFP4 runner (``KIMI_K31_ROUTER_SORT``), else
        ``None``. Needs the tile of the runner's tactic for ``num_tokens`` rows, known from the
        first (eager or autotuning) forward with that row count on."""
        if not self._k31_router_sort_wanted or torch.compiler.is_compiling():
            return None
        if not 1 <= num_tokens <= _K31_ROUTER_SORT_MAX_M:
            return None
        moe = self.routed_experts
        backend = moe.backend
        if (
            moe.use_dp
            or moe.comm is not None
            or moe.apply_router_weight_on_input
            or not isinstance(moe.routing_method, _KimiK3RoutingMethod)
            or not hasattr(backend, "_take_prerouted")
            or not getattr(getattr(backend, "op_backend", None), "supports_prerouted", False)
            # The scheduler must take the top-k from the routing method (separated routing) and
            # run the forward as one chunk.
            or not backend._routes_outside_the_kernel()
            or num_tokens > getattr(moe, "moe_max_num_tokens", num_tokens)
        ):
            return None
        slot_start = getattr(backend, "slot_start", None)
        num_local = getattr(backend, "expert_size_per_partition", None)
        num_slots = getattr(backend, "num_slots", None)
        if slot_start is None or num_local is None or num_slots != self.gate.num_experts:
            return None
        from ..custom_ops.trtllm_gen_custom_ops import (
            mxfp4_prerouted_tile,
            mxfp4_runner_has_prerouted,
        )
        from ..moe.kernels.blackwell import k3_route_tail

        if not mxfp4_runner_has_prerouted(finalized=do_finalize):
            return None
        device = self.gate.e_score_correction_bias.device
        if not k3_route_tail.supports_device(device):
            return None
        # The multi-CTA kernel above 256 rows (or from _K31_ROUTER_SORT_WIDE_FROM where its tables fit), else
        # the one-cluster kernel.
        wide = num_tokens > k3_route_tail.MAX_M or (
            num_tokens >= _K31_ROUTER_SORT_WIDE_FROM
            and k3_route_tail.wide_sort_fits(num_tokens, num_local)
        )
        if not (
            k3_route_tail.wide_sort_fits(num_tokens, num_local)
            if wide
            else k3_route_tail.runner_sort_fits(num_tokens, num_local)
        ):
            return None
        if wide:
            # The multi-CTA kernel's workspace must exist before graph capture.
            if not torch.cuda.is_current_stream_capturing():
                k3_route_tail.wide_workspace(device)
            elif not k3_route_tail.wide_workspace_ready(device):
                return None
        tile = mxfp4_prerouted_tile(num_tokens, num_local, self.top_k)
        if tile is None:
            return None
        return tile, slot_start, num_local, num_slots, 1 if wide else k3_route_tail.MAX_M + 1

    def _k31_runner_route(
        self,
        logits: torch.Tensor,
        target: Tuple[int, int, int, int, int],
        quant_input: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Optional[MxFp8QuantizedTensor]]:
        """Route ``logits`` with the CuTe tail in the TRTLLM-Gen MXFP4 runner's layout and hand
        the top-k and the runner's routing to the routed experts (``KIMI_K31_ROUTER_SORT``).
        Returns ``(routed, quantized)``: ``routed`` is ``False`` (nothing handed over) if the logits
        are not the tail's input; ``quantized`` is the MXFP8 routed input when the launch also
        quantized ``quant_input`` (``KIMI_K31_ROUTER_SORT_QUANT``)."""
        if (
            logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.shape[1] != self.gate.num_experts
            or not logits.is_contiguous()
            or logits.data_ptr() % 16
        ):
            return False, None
        from ..moe.kernels.blackwell import k3_route_tail

        tile, slot_start, num_local, num_slots, wide_from = target
        m = logits.shape[0]
        if quant_input is not None and not (
            _K31_ROUTER_SORT_QUANT
            and m <= min(k3_route_tail.MAX_M, _K31_ROUTER_SORT_QUANT_MAX_M)
            and quant_input.dtype == torch.bfloat16
            and quant_input.dim() == 2
            and quant_input.shape[0] == m
            and quant_input.shape[1] % 32 == 0
            and quant_input.is_contiguous()
            and quant_input.data_ptr() % 16 == 0
        ):
            quant_input = None
        topk_ids, topk_weights, prerouted, quantized = k3_route_tail.run_k3_route_runner(
            logits,
            m,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            slot_start,
            num_local,
            num_slots,
            tile,
            # Where the fused noaux + MXFP8 quant op would route (one FP64 -> BF16 rounding).
            m <= _K31_FUSED_ROUTE_QUANT_MAX_M,
            quant_input=quant_input,
            wide_from=wide_from,
        )
        routing = self.routed_experts.routing_method
        routing.precomputed = (topk_ids, topk_weights)
        routing.presorted = (prerouted, tile, topk_ids)
        logger.info_once(
            "KIMI_K31_ROUTER_SORT: the routing tail writes the MXFP4 runner's routing"
            f"{' and the MXFP8 routed input' if quantized is not None else ''} (first: {m} rows, tile {tile})",
            key=f"k31_router_sort_{quantized is not None}",
        )
        if quantized is None:
            return True, None
        return True, MxFp8QuantizedTensor(quantized[0], quantized[1])

    def _k31_latent_stream_quant_applies(self, latent: torch.Tensor) -> bool:
        """``KIMI_K31_LATENT_STREAM_QUANT``: whether the latent stream quantizes this forward's routed input. Only
        where the routing sort will route it (in either finalize mode), so the top-k is the tail's (at <= 64 rows the
        fused noaux + quant op's weights) and the result stays bitwise."""
        if not (
            _K31_LATENT_STREAM_QUANT
            and isinstance(latent, torch.Tensor)
            and latent.dim() == 2
            and latent.dtype == torch.bfloat16
        ):
            return False
        n = latent.shape[0]
        return (
            getattr(self.routed_experts.backend, "accepts_mxfp8_input", False)
            and self._k31_runner_sort_target(n, True) is not None
            and self._k31_runner_sort_target(n, False) is not None
        )

    def _k31_dep_route(self, logits: torch.Tensor, all_rank_num_tokens) -> bool:
        """``KIMI_K31_ROUTER_TAIL_DEP``: hand the routing tail's top-16 of this rank's rows, with BF16 weights (the cast
        of noaux_tc's), to the routing method of a dispatching (attention-DP) MoE whose backend reads BF16 routing
        scales. ``False`` where it does not apply."""
        moe = self.routed_experts
        if not self._k31_router_tail_dep_wanted or torch.compiler.is_compiling():
            return False
        if moe.comm is None or not _k31_takes_bf16_routing_scales(moe.backend):
            return False
        m = logits.shape[0]
        rank_tokens = list(all_rank_num_tokens) if all_rank_num_tokens is not None else [m]
        one_chunk = moe.calculate_num_chunks(rank_tokens) == 1
        if (
            not moe.use_dp
            or moe.apply_router_weight_on_input
            or not isinstance(moe.routing_method, _KimiK3RoutingMethod)
            or not _K31_FUSED_ROUTE_QUANT_MAX_M < m <= _K31_ROUTER_SORT_MAX_M
            or logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.shape[1] != self.gate.num_experts
            or not logits.is_contiguous()
            or logits.data_ptr() % 16
        ):
            return False
        # The scheduler must route this forward as one chunk with the routing method's top-k.
        if not one_chunk:
            return False
        from ..moe.kernels.blackwell import k3_route_tail

        device = logits.device
        if not k3_route_tail.supports_device(device):
            return False
        topk_ids = torch.empty((m, self.top_k), dtype=torch.int32, device=device)
        topk_weights = torch.empty((m, self.top_k), dtype=torch.bfloat16, device=device)
        # The multi-CTA kernel from _K31_ROUTER_SORT_WIDE_FROM rows on (top-k only it needs no workspace).
        wide = m > k3_route_tail.MAX_M or m >= _K31_ROUTER_SORT_WIDE_FROM
        run = k3_route_tail.run_k3_route_wide if wide else k3_route_tail.run_k3_route_tail
        run(
            logits[:m],
            m,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            topk_ids,
            topk_weights,
            None,
            0,
            self.gate.num_experts,
        )
        moe.routing_method.precomputed = (topk_ids, topk_weights)
        logger.info_once(
            f"KIMI_K31_ROUTER_TAIL_DEP: the routing tail computes the top-k (first: {m} rows)",
            key="k31_router_tail_dep",
        )
        return True

    def _can_umma_route(self, hidden_states: torch.Tensor) -> bool:
        """Whether ``TRTLLM_K3_ROUTER_GEMM=umma`` computes this forward's router logits."""
        if _K3_ROUTER_GEMM != "umma" or not 1 <= hidden_states.shape[0] <= _K3_ROUTER_GEMM_MAX_M:
            return False
        from ..moe.kernels.blackwell import k3_router_umma

        return k3_router_umma.can_run(hidden_states, self.gate.weight)

    def _umma_router_partials(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 split-K partial router logits ``[S, M, 896]`` of the UMMA router GEMM."""
        from ..moe.kernels.blackwell import k3_router_umma

        partials = torch.empty(
            (k3_router_umma.DEFAULT_SPLITS, hidden_states.shape[0], self.gate.num_experts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        k3_router_umma.run_k3_router_umma(hidden_states, self.gate.weight, partials)
        return partials

    def _gemv_router_partials(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Split-K partial router logits ``[S, M, 896]`` of ``k3_route``'s router GEMM (the
        ``mma.sync`` GEMV or, with ``TRTLLM_K3_ROUTER_GEMV=umma``, the UMMA GEMM)."""
        from ..moe.kernels.blackwell import k3_route_tail

        if k3_route_tail.ROUTER_GEMV == "umma":
            return self._umma_router_partials(hidden_states)
        partials = torch.empty(
            (k3_route_tail.DEFAULT_GEMV_SPLITS, hidden_states.shape[0], self.gate.num_experts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        k3_route_tail.run_k3_router_gemv(hidden_states, self.gate.weight, partials)
        return partials

    def _tail_route_partials(
        self, partials: torch.Tensor, logits: torch.Tensor, target: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Routing tail on a router GEMM's split-K partials (``logits`` is their first split, the
        placeholder the routed experts were handed)."""
        return self._tail_route(partials, target)

    def _gemv_route(self, hidden_states: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
        """Route with the router GEMV + tail; hands the result to the routing method and
        returns the FP32 logits."""
        outs = torch.ops.trtllm.k3_route(
            hidden_states,
            self.gate.weight,
            self.gate.e_score_correction_bias,
            float(self.gate.routed_scaling_factor),
            target[0],
            target[1],
            128,
        )
        self.routed_experts.routing_method.precomputed = self._attach_route(outs, target)
        return outs[8]

    def _quantize_routed_input(self, x: torch.Tensor) -> Fp4QuantizedTensor:
        """NVFP4-quantize the routed input like the CuteDSL backend's ``quantize_input``."""
        backend = self.routed_experts.backend
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x, backend.fc31_input_scale, backend.scaling_vector_size, False, False
        )
        return Fp4QuantizedTensor(x_fp4, x_sf, is_sf_swizzled=False)

    def _own_workspace_latent_tail(
        self,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
        residual: Optional[torch.Tensor],
        defer_residual: bool,
    ):
        """``TRTLLM_K3_LATENT_AR_OWN_WS``: the routed latent all-reduce + RMSNorm on its own MNNVL
        workspace and latent_up, issued on the main stream before it joins the shared-expert
        stream, whose output only the residual update reads. Returns what ``forward`` returns."""
        workspace = self._latent_ar_ws
        gamma = self.routed_expert_norm.weight
        eps = float(self.routed_expert_norm.variance_epsilon)
        latent = routed_out.reshape(-1, routed_out.shape[-1]).contiguous()
        up = self.routed_expert_up_proj
        routed_latent = None
        if _K3_LATENT_AR_CUTE != "0":
            from ..distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES
            from ..kernels.blackwell import k3_latent_ar_norm

            world_size = workspace["world_size"]
            one_shot = latent.numel() * world_size * latent.element_size() <= (
                _K3_LATENT_AR_ONESHOT_MAX_BYTES or _MNNVL_ONE_SHOT_THRESHOLD_BYTES
            )
            row = _K3_LATENT_AR_ROW and k3_latent_ar_norm.can_run_row(
                latent, gamma, world_size, workspace["buffer_bytes"]
            )
            if one_shot and (row or k3_latent_ar_norm.can_run(latent, gamma, world_size)):
                routed_latent = torch.empty_like(latent)
                launch = (
                    k3_latent_ar_norm.run_k3_latent_ar_norm_row
                    if row
                    else k3_latent_ar_norm.run_k3_latent_ar_norm
                )
                launch(
                    latent,
                    gamma,
                    eps,
                    routed_latent,
                    uc_addr=workspace["buffer"].data_ptr(),
                    mc_addr=workspace["mc"].data_ptr(),
                    flags=workspace["flags"],
                    rank=workspace["rank"],
                    world_size=world_size,
                    early_trigger=_K3_LATENT_AR_CUTE == "1",
                )
        if routed_latent is None:
            (routed_latent,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                latent,
                gamma,
                None,
                eps,
                workspace["buffer"],
                workspace["flags"],
                True,
                None,
                int(AllReduceFusionOp.RMS_NORM),
            )
        routed_latent = routed_latent.view(routed_out.shape)
        routed_out = self._routed_projection(routed_latent, up)
        self._issue_latent_ar_clear(latent.shape[0])
        self._join_expert_l2_prefetch()
        self.moe_shared_event.wait()
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        return routed_out + shared_out

    def _routed_projection(
        self, hidden_states: torch.Tensor, projection: nn.Module
    ) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        if self._decode_gemm_plan is not None:
            out = planned_low_m_linear(self._decode_gemm_plan, hidden_states, projection.weight)
            if out is not None:
                return out
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    def _can_fuse_router_down(self, hidden_states: torch.Tensor) -> bool:
        """Whether one fused GEMM can produce the router logits and the latent
        down-projection (trtllm::dsv3_router_latent_gemm_op).

        Scoped to the validated configuration (TP MEGAMOE_CUTEDSL with fused
        communication on SM107); the fusion itself does not depend on it.
        """
        if (
            torch.compiler.is_compiling()
            or _K3_DISABLE_MIN_LATENCY_LATENT_PROJ
            or not hidden_states.is_cuda
            or self.routed_experts.use_dp
            or self.routed_experts.parallel_size <= 1
            or self.routed_experts.moe_backend != "MEGAMOE_CUTEDSL"
            or self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
            or get_sm_version() != 107
        ):
            return False
        down = self.routed_expert_down_proj
        if not isinstance(down, nn.Linear) or down.bias is not None:
            return False
        if (
            hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 16
            or hidden_states.shape[1] != 7168
            or self.gate.weight.shape != (896, 7168)
            or down.weight.shape != (3584, 7168)
        ):
            return False
        # Inspect the loaded weights: FP8 conversion can replace the down module.
        return all(
            tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.device == hidden_states.device
            and tensor.stride() == (7168, 1)
            and tensor.data_ptr() % 16 == 0
            for tensor in (hidden_states, self.gate.weight, down.weight)
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        ``nvidia/Kimi-K3-NVFP4`` declares the routed experts per layer as
        ``NVFP4`` with ``group_size=16``; the original ``moonshotai/Kimi-K3``
        declares nothing per layer and keeps the historical
        ``W4A8_MXFP4_MXFP8`` default. Reading the checkpoint instead of
        hardcoding is what lets one code path serve both.

        An exclusion outranks the per-layer entry and the default below:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second. It is
        matched as a pattern, so it is asked only about real module names.
        """
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            logger.debug(
                "Kimi K3 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        # Logged once per layer: the routed-expert format decides
                        # which MoE backends can serve this checkpoint at all.
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _check_trtllm_situ_quant(moe_backend: str, quant_algo: Optional[QuantAlgo]) -> None:
        """Reject a routed-expert format trtllm-gen has no fused SiTu cubin for.

        trtllm-gen has fused SiTu FC1 cubins for two input formats and no
        standalone SiTu activation kernel, so anything else has to die here
        rather than in a cubin lookup deep inside the runner. Checked against
        the resolved backend, not the K3 architecture branch, because the
        generic FP8_BLOCK_SCALES fallback in ``resolve_moe_backend`` can also
        land on TRTLLM.

        The admitted set is read off the backend rather than restated here,
        because restating it is what broke. This guard was written in #17865
        when MXFP4 was the only fused SiTu drop; #17940 then added the NVFP4
        (group-16 ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*``) cubins and updated
        ``TRTLLMGenFusedMoE``'s set without touching this copy. For the week
        in between, an NVFP4 K3 checkpoint could not start at all -- and not
        only when TRTLLM was asked for by name, because
        ``ModelConfig.resolve_moe_backend`` sends every K3 architecture to
        TRTLLM, so the default AUTO configuration hit this raise too. The unit
        tests did not catch it: they call ``create_moe`` directly and never
        reach this guard, so the kernel path stayed green while the model path
        was closed.

        A staticmethod, not an inline block, so that the invariant is
        reachable from a test without constructing the whole runtime.
        """
        situ_supported = TRTLLMGenFusedMoE.situ_supported_quant_algos()
        if moe_backend != "TRTLLM" or quant_algo in situ_supported:
            return
        supported = ", ".join(sorted(algo.name for algo in situ_supported))
        raise ValueError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, which the "
            "TRTLLM (trtllm-gen) MoE backend cannot serve: fused SiTu cubins "
            f"exist only for {supported}. Set moe_config.backend to CUTLASS "
            "or MEGAMOE_CUTEDSL."
        )

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        # Every backend here declares ``ActivationType.SiTu`` in its
        # ``activation_support``; the list is not a preference order. CUTEDSL
        # joined once its act-fusion kernel grew the SiTU epilogue, and
        # CUTEDSL_FC12 (Rubin/SM107) once its fused FC1+FC2 kernel did.
        supported_backends = {
            "CUTLASS",
            "TRTLLM",
            "CUTEDSL",
            "CUTEDSL_FC12",
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
        }
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                "Kimi K3 SiTU routed experts only support the CUTLASS, TRTLLM, "
                "CUTEDSL, CUTEDSL_FC12, MEGAMOE_DEEPGEMM, and MEGAMOE_CUTEDSL backends; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            # DWDP does not hook the load path at all: setup_dwdp() scans the
            # model *after* every weight is loaded, reads w3_w1_weight /
            # w2_weight and their block scales straight off the backend
            # (dwdp/setup.py::collect_moe_params), copies them into fabric
            # memory, frees the originals, and rebinds param.data to a
            # composite VA view.  K3's packed-checkpoint streaming produces
            # exactly those tensors, so the two are compatible -- but only on
            # the NVFP4 CuteDSL leaves that declare supports_dwdp (CUTEDSL,
            # CUTEDSL_FC12).
            #
            # MEGAMOE_CUTEDSL is excluded deliberately rather than by
            # oversight: its process_weights_after_loading packs into the mega
            # buffers and shrinks the raw source params back to placeholders
            # (see maybe_finalize_layer), which has never been checked against
            # DWDP's copy-then-free of those same originals.
            if model_config.moe_backend not in ("CUTEDSL", "CUTEDSL_FC12"):
                raise NotImplementedError(
                    "Kimi K3 with DWDP requires moe_config.backend CUTEDSL or "
                    f"CUTEDSL_FC12; got {model_config.moe_backend!r}."
                )

        moe_tp, moe_ep = KimiK3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        # MegaMoE uses this value as global DP SymmBuffer capacity, then
        # divides it by EP size for the per-rank allocation. Other backends
        # keep the user-configured value as their MoE chunking bound.
        # Preserve an explicitly larger capacity.
        if routed_model_config.moe_backend in {
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
        }:
            # helix_a2a ranks route ceil(max_num_tokens / tp) rows each.
            rank_max_num_tokens = routed_model_config.max_num_tokens
            if _is_helix_a2a(model_config):
                rank_max_num_tokens = -(-rank_max_num_tokens // routed_mapping.tp_size)
            default_moe_max_num_tokens = rank_max_num_tokens * routed_mapping.dp_size
            configured_moe_max_num_tokens = int(routed_model_config.moe_max_num_tokens or 0)
            if configured_moe_max_num_tokens < default_moe_max_num_tokens:
                logger.info_once(
                    "Kimi K3 MegaMoE raises moe_max_num_tokens from "
                    f"{configured_moe_max_num_tokens} to {default_moe_max_num_tokens} "
                    "because the global DP SymmBuffer requires capacity for "
                    "max_num_tokens * dp_size.",
                    key=(
                        "kimi_k3_megamoe_capacity_override_"
                        f"{configured_moe_max_num_tokens}_{default_moe_max_num_tokens}"
                    ),
                )
            routed_model_config.moe_max_num_tokens = max(
                configured_moe_max_num_tokens,
                default_moe_max_num_tokens,
            )
        routed_model_config._frozen = True
        return routed_model_config

    def _issue_router(
        self,
        hidden_states: torch.Tensor,
        gemv_route: bool,
        route_target: Optional[Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Issue the router GEMM on the current stream; returns ``(router_logits,
        router_partials)`` (the split-K partials of the router GEMMs whose routing tail sums
        them, else ``None``; ``router_logits`` is ``None`` with the fused router)."""
        # With the fused router, routing runs inside the routed branch, so the shared experts
        # fork before it instead of waiting for the router.
        if gemv_route and _K3_ROUTE_GEMV_DEFER_TAIL:
            router_partials = self._gemv_router_partials(hidden_states)
            return router_partials[0], router_partials
        if gemv_route:
            return self._gemv_route(hidden_states, route_target), None
        if route_target is not None and self._can_umma_route(hidden_states):
            router_partials = self._umma_router_partials(hidden_states)
            return router_partials[0], router_partials
        if _K3_FUSED_ROUTER != "0":
            return None, None
        return self._router_logits(hidden_states), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
        tail_mailbox_residual: Optional[torch.Tensor] = None,
        defer_residual_to: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        ``tail_mailbox_residual`` (the mailbox fold, without ``residual``): where
        the fused latent-MoE tail runs, it returns a ``_DeferredTailResidual`` on that residual
        instead of the MoE output; the other paths ignore it (the caller adds).
        ``defer_residual_to`` (PDL attn_res kernel in mode "2", without ``residual``): where the
        routed and shared outputs would be summed into the MoE output, a ``_DeferredMoEResidual``
        on that residual instead (the next layer's mix computes ``residual + bf16(routed +
        shared)``, the caller's two BF16 adds).

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead where that update would be the separate BF16 add.
        """
        identity = hidden_states
        # TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld: issued after this layer's latent_down, else dropped (the
        # all-reduce then clears the buffer itself).
        pending_clear = self._latent_ar_clear_pending
        self._latent_ar_clear_pending = None
        projected_input = None
        early_routed_in = None
        shared_first = False
        shared_partial = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        packed_tactic = self._packed_front_tactic(hidden_states)
        k31_front_routed = False
        self._packed_shared_gate_up = None
        gemv_route = False
        tail_order = False
        router_partials = None
        merged = self._merged_reduce_buffers(hidden_states)
        routed_slot = shared_slot = None
        if merged is not None:
            split = hidden_states.shape[0] * self.moe_hidden_size
            routed_slot = merged[0][:split].view(hidden_states.shape[0], self.moe_hidden_size)
            shared_slot = merged[0][split:].view(hidden_states.shape[0], self.hidden_size)
        if (
            route_target is None
            and merged is None
            and pending_clear is None
            and self._k31_sharded_front_applies(hidden_states)
        ):
            # Sharded MoE front: top-16 + the MXFP8 routed input from the sharded front.
            router_logits, projected_input = self._k31_sharded_front(hidden_states)
            k31_front_routed = True
        elif self._can_fuse_router_down(hidden_states):
            router_logits, projected_input = torch.ops.trtllm.dsv3_router_latent_gemm_op(
                hidden_states, self.gate.weight, self.routed_expert_down_proj.weight
            )
        elif (
            packed_tactic is not None
            and route_target is None
            and merged is None
            and pending_clear is None
            and (packed := self._packed_front_forward(hidden_states, packed_tactic)) is not None
        ):
            # Packed MoE front: router logits, the routed latent and the shared gate_up
            # from one GEMM; the shared expert continues from its gate_up output (_shared_output).
            router_logits, projected_input, self._packed_shared_gate_up = packed
        else:
            if (
                self.latent_down_stream is not None
                and do_multi_stream()
                and not torch.compiler.is_compiling()
            ):
                # Forked ahead of the router GEMM: the latent chain needs only
                # hidden_states.
                gemv_route = route_target is not None and self._can_gemv_route(hidden_states)
                shared_first = self._shared_first_applies(hidden_states.shape[0])
                tail_order = (
                    route_target is not None
                    and not gemv_route
                    and not shared_first
                    and self._tail_waits_for_shared(hidden_states.shape[0], route_target[1])
                )
                self.moe_latent_fork_event.record()
                with torch.cuda.stream(self.latent_down_stream):
                    self.moe_latent_fork_event.wait()
                    early_routed_in = self._routed_projection(
                        hidden_states, self.routed_expert_down_proj
                    )
                    if gemv_route or tail_order:
                        # Quantized here, the routed input does not wait for the routing.
                        early_routed_in = self._quantize_routed_input(early_routed_in)
                    elif self._k31_latent_stream_quant_applies(early_routed_in):
                        # KIMI_K31_LATENT_STREAM_QUANT: MXFP8 beside the routing tail.
                        x_q, x_sf = self.routed_experts.backend.quantize_input(early_routed_in)
                        early_routed_in = MxFp8QuantizedTensor(x_q, x_sf)
                    self.moe_latent_ready_event.record()
                    self._issue_pending_latent_ar_clear(pending_clear)
                # No record_stream (under graph capture it holds every layer's
                # blocks until capture ends): both tensors are read on main only
                # after the join, and later work on this stream forks from main.
                if (shared_first and _K3_SHARED_FIRST == "1") or (
                    tail_order and _K3_ROUTE_TAIL_ORDER == "fork"
                ):
                    shared_partial = self._shared_gemms_first(identity, self.moe_latent_fork_event)
            router_logits, router_partials = self._issue_router(
                hidden_states, gemv_route, route_target
            )
            if (shared_first and _K3_SHARED_FIRST == "router") or (
                tail_order and _K3_ROUTE_TAIL_ORDER == "router"
            ):
                self.moe_main_event.record()
                shared_partial = self._shared_gemms_first(identity, self.moe_main_event)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None
        # Our fused latent-MoE tail serves only the plain replicated-tail call (no merged reduction,
        # no early shared-expert fork of the other tails).
        fused_tail = (
            self._select_fused_tail(hidden_states)
            if merged is None and shared_partial is None
            else None
        )
        fold_finalize = (
            fused_tail is not None
            and fused_tail.folds_finalize
            and self.routed_experts.backend.supports_unfinalized_output()
        )
        overlap_shared_allreduce = (
            self._defer_shared_allreduce
            and do_multi_stream()
            and self.shared_expert_stream is not None
            and not torch.compiler.is_compiling()
        )
        num_tokens = hidden_states.shape[0]

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ=1 restores nn.Linear
            # (A/B escape hatch). When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            logits = router_logits
            if logits is None:
                logits = self._fused_route(hidden_states)
                if logits is None:
                    logits = self.gate.compute_logits(hidden_states)
            moe_kwargs = {}
            if projected_input is not None:
                routed_in = projected_input
            elif early_routed_in is not None:
                routed_in = early_routed_in
                # ConfigurableMoE joins the event before its first read of x.
                moe_kwargs = {
                    "input_ready_event": (
                        self.moe_latent_ready_event
                        if shared_partial is None
                        else self.moe_shared_gemm_event
                    )
                }
            else:
                routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, (Fp4QuantizedTensor, MxFp8QuantizedTensor)):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            if merged is not None:
                moe_kwargs["moe_output"] = routed_slot
            if route_target is not None and (router_partials is not None or not gemv_route):
                # Routed by the CuTe tail when the backend asks for the routing (after it
                # forks the input quantization, unless the latent chain quantized it); with the
                # UMMA router GEMM the tail sums its split-K partials.
                if router_partials is not None:
                    self.routed_experts.routing_method.deferred = functools.partial(
                        self._tail_route_partials, router_partials, target=route_target
                    )
                else:
                    self.routed_experts.routing_method.deferred = functools.partial(
                        self._tail_route, target=route_target
                    )
            unfinalized = fold_finalize
            if unfinalized:
                # The fused tail does the top-k weighted sum while it loads the
                # routed row, so the routed experts skip their finalize kernel.
                moe_kwargs["do_finalize"] = False
            if (
                route_target is None
                and not gemv_route
                and merged is None
                and _K3_FUSED_ROUTER == "0"
                and logits is not None
                and not isinstance(routed_in, Fp4QuantizedTensor)
                and not k31_front_routed
            ):
                # KIMI_K31_ROUTER_SORT: top-k + the runner's routing from the CuTe tail (and, for an
                # input produced on this stream, its MXFP8 quantization).
                sort_target = self._k31_runner_sort_target(num_tokens, not unfinalized)
                if sort_target is not None:
                    fuse_quant = (
                        isinstance(routed_in, torch.Tensor)
                        and "input_ready_event" not in moe_kwargs
                        and getattr(self.routed_experts.backend, "accepts_mxfp8_input", False)
                    )
                    _, quantized = self._k31_runner_route(
                        logits, sort_target, routed_in if fuse_quant else None
                    )
                    if quantized is not None:
                        routed_in = quantized
                        moe_kwargs["output_dtype"] = hidden_states.dtype
                else:
                    self._k31_dep_route(logits, all_rank_num_tokens)
            y = self.routed_experts(
                routed_in,
                logits,
                all_rank_num_tokens=all_rank_num_tokens,
                **moe_kwargs,
            )
            if early_routed_in is not None:
                self._issue_l2_prefetch(num_tokens)
            if self._l2pf is not None:
                # TRTLLM_K3_L2PF_V2: the next layer's attention weights, once FC2 is done.
                self._l2pf.fork("next_attention", self.layer_idx, num_tokens)
            if route_target is not None:
                self.routed_experts.routing_method.deferred = None
            if overlap_shared_allreduce:
                self.moe_routed_event.record()
            if merged is not None:
                if y.data_ptr() != routed_slot.data_ptr():
                    routed_slot.copy_(y)
                return routed_slot
            if unfinalized:
                gemm2_permuted, expert_weights, expanded_idx = y
                return UnfinalizedRoutedOutput(
                    gemm2_permuted,
                    expert_weights.contiguous(),
                    expanded_idx.view(expert_weights.shape[0], -1),
                )
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            """The shared expert's (reduced) output."""
            final_params = (
                AllReduceParams(enable_allreduce=False)
                if self._defer_shared_allreduce or merged is not None or fused_tail is not None
                else None
            )
            packed_gate_up, self._packed_shared_gate_up = self._packed_shared_gate_up, None
            front_act, self._k31f_shared_act = self._k31f_shared_act, None
            mlp = self.shared_experts
            if front_act is not None:
                # Sharded front's shared-expert part: the front computed gate_up + SiTU.
                return _shared_reduce(
                    mlp.down_proj(
                        front_act, all_reduce_params=final_params, layer_idx=mlp.layer_idx
                    )
                )
            if packed_gate_up is not None:
                # GatedMLP.forward's unfused path from the packed front's gate_up output.
                return _shared_reduce(
                    mlp.down_proj(
                        mlp._apply_activation(packed_gate_up),
                        all_reduce_params=final_params,
                        layer_idx=mlp.layer_idx,
                    )
                )
            return _shared_reduce(
                self.shared_experts(identity, final_all_reduce_params=final_params)
            )

        def _shared_reduce(shared_out: torch.Tensor) -> torch.Tensor:
            """Reduce the shared expert's down-projection output (or hand it to the merged
            reduction)."""
            if fused_tail is not None:
                # The fused tail reduce-scatters the partial itself.
                return shared_out
            if merged is not None:
                # The partial joins the routed reduction (``_merged_reduce``).
                if shared_out.data_ptr() != shared_slot.data_ptr():
                    shared_slot.copy_(shared_out)
                return shared_slot
            if overlap_shared_allreduce:
                # Wait for all routed chunks, but overlap the norm/up tail.
                self.moe_routed_event.wait()
                shared_out = self._shared_all_reduce()(shared_out)
            return shared_out

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. The shared GatedMLP includes its output all-reduce on
        # the auxiliary stream (deferred past the routed MoE for fused-comm
        # MegaMoE). The final join includes the shared all-reduce, so it
        # precedes the routed all-reduce: concurrent collectives on different
        # streams can corrupt SYMM_MEM all-reduce state.
        if shared_partial is not None:
            backend = self.routed_experts.backend
            if shared_first:
                backend.post_fc1_event = self.moe_fc1_event
            routed_out = _routed_output()
            fc1_recorded = shared_first and backend.post_fc1_event is None
            if shared_first:
                backend.post_fc1_event = None
            with torch.cuda.stream(self.shared_expert_stream):
                if fc1_recorded:
                    self.moe_fc1_event.wait()
                shared_out = shared_partial
                if self._shared_needs_all_reduce():
                    shared_out = self._shared_all_reduce()(shared_partial)
                self.moe_shared_event.record()
            self._join_expert_l2_prefetch()
            self.moe_shared_event.wait()
        elif (
            self._latent_ar_ws is not None
            and fused_tail is None
            and merged is None
            and moe_all_reduce is not None
            and not self._defer_shared_allreduce
            and num_tokens <= self._latent_ar_ws["max_tokens"]
            and self.shared_expert_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        ):
            # The fork / join of maybe_execute_in_parallel, with the join left open until it is
            # known whether the latent all-reduce runs on its own workspace.
            self.moe_main_event.record()
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_out = _shared_output()
                self.moe_shared_event.record()
            if routed_out.dtype == torch.bfloat16:
                return self._own_workspace_latent_tail(
                    routed_out, shared_out, residual, defer_residual
                )
            self._join_expert_l2_prefetch()
            self.moe_shared_event.wait()
        else:
            routed_out, shared_out = maybe_execute_in_parallel(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
                disable_on_compile=True,
            )
            self._join_expert_l2_prefetch()
        if fused_tail is not None:
            # Routed AllReduce + RMSNorm, shared ReduceScatter, 1/TP-row up-projection + shared
            # add, multicast all-gather. The residual update is done here (no deferral).
            if residual is None and tail_mailbox_residual is not None:
                # The next layer's mix reads the mailbox.
                mailbox = fused_tail.forward_to_mailbox(
                    routed_out,
                    shared_out,
                    self.routed_expert_norm.weight,
                    self.routed_expert_up_proj.weight,
                )
                return _DeferredTailResidual(tail_mailbox_residual, fused_tail, mailbox)
            moe_out = fused_tail(
                routed_out,
                shared_out,
                self.routed_expert_norm.weight,
                self.routed_expert_up_proj.weight,
            )
            return moe_out if residual is None else residual + moe_out
        if self._defer_shared_allreduce and not overlap_shared_allreduce:
            shared_out = self._shared_all_reduce()(shared_out)
        if merged is not None or self._reduce_routed_output:
            if merged is not None:
                routed_latent, shared_out = self._merged_reduce(*merged, num_tokens)
            else:
                # RMSNorm folded into the routed all-reduce (kARRMSNorm): the
                # MNNVL kernel normalizes straight out of the lamport buffer, so
                # the reduced row never round-trips through HBM for a norm kernel.
                routed_latent = moe_all_reduce(
                    routed_out,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RMS_NORM,
                        norm_weight=self.routed_expert_norm.weight,
                        eps=float(self.routed_expert_norm.variance_epsilon),
                    ),
                )
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            from ..modules.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        if defer_residual_to is not None:
            return _DeferredMoEResidual(defer_residual_to, routed_out, shared_out)
        return routed_out + shared_out

    @property
    def supports_seq_parallel(self) -> bool:
        """Direct MoE-TP whose routed and shared partials this module reduces itself (see ``forward_seq_parallel``)."""
        return self._reduce_routed_output and not self._defer_shared_allreduce

    def forward_seq_parallel(
        self, hidden_states: torch.Tensor, sp: _KimiSeqParallel
    ) -> torch.Tensor:
        """Sequence-parallel MoE block: ``hidden_states`` and the result are this rank's ``[sp.rows, hidden]`` rows.

        The router and the latent down-projection run on the local rows; one grouped all-gather brings every rank
        the full normed input (for the TP-sharded shared expert), latent and router logits (for its routed experts),
        and one grouped reduce-scatter sums the routed latent and shared partials back onto the local rows, where
        the latent norm and up-projection run.
        """
        router_logits = self.gate.compute_logits(hidden_states)
        latent = self._routed_projection(hidden_states, self.routed_expert_down_proj)
        hidden_full, latent_full, logits_full = sp.gather([hidden_states, latent, router_logits])
        routed_partial = self.routed_experts(latent_full, logits_full, all_rank_num_tokens=None)
        shared_partial = self.shared_experts(
            hidden_full, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
        )
        routed_latent, shared_out = sp.reduce_scatter([routed_partial, shared_partial])
        routed_out = self._routed_projection(
            self.routed_expert_norm(routed_latent), self.routed_expert_up_proj
        )
        return routed_out + shared_out


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# helix_a2a: token-sliced residual stream around head-TP attention.
# ---------------------------------------------------------------------------

# Helix + enable_attention_dp: every MoE layer keeps its residual stream
# (prefix sum, attn-res bank, norms, MoE) on this rank's ceil(T / cp) rows, the
# split the executor's helix all_rank_num_tokens assumes. KDA and MLA stay
# head-TP over the CP group on all T rows: an all-gather feeds them and their
# o_proj partials are reduce-scattered. The MoE runs the attention-DP
# all-to-all with a replicated shared expert. The all-gather is symm_mem
# MULTIMEM (NCCL fallback) and the reduce-scatter is AllReduce + own rows: at
# 16 ranks, T=512 they take 19.4 / 28.1 us against NCCL's 40.0 / 38.4 us.


def _multimem_allgather(mapping: Mapping, max_bytes: int) -> SymmetricMemoryAllGather:
    """SymmetricMemoryAllGather admitted for this GPU and TP size.

    The upstream allow-list stops at SM 10.0 and 8 ranks; MULTIMEM all-gather
    also runs at 16 and 32 ranks on Rubin. The class's rendezvous probe still
    disables it when MULTIMEM is not usable.
    """
    major, minor = torch.cuda.get_device_capability()
    capability = f"{major}.{minor}"
    cls = type(
        "_HelixA2AMultimemAllGather",
        (SymmetricMemoryAllGather,),
        {
            "_WORLD_SIZES_MULTIMEM": {capability: [mapping.tp_size]},
            "_MAX_SIZES": {capability: {mapping.tp_size: max_bytes}},
        },
    )
    # Reuse WORLD when it is the TP group (helix GEN: world == tp * cp).
    group = (
        torch.distributed.group.WORLD
        if torch.distributed.is_initialized()
        and torch.distributed.get_world_size() == mapping.tp_size
        else None
    )
    return cls(mapping, dtype=torch.bfloat16, group=group)


class KimiK3HelixA2A(nn.Module):
    """Moves rows between the full T-row batch and this rank's chunk.

    Row ownership and zero padding follow the upstream helix + attention-DP
    path (``_helix_cp_pad``): rank r owns rows ``[r * chunk, (r + 1) * chunk)``.
    """

    def __init__(
        self,
        mapping: Mapping,
        hidden_size: int,
        max_num_tokens: int,
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            raise ValueError(
                "Kimi K3 helix_a2a needs the head-TP (non-attention-DP) mapping; got "
                f"tp_size={mapping.tp_size}, enable_attention_dp={mapping.enable_attention_dp}."
            )
        self.mapping = mapping
        self.world_size = mapping.tp_size
        self.rank = mapping.tp_rank
        padded_rows = -(-max_num_tokens // self.world_size) * self.world_size
        # 1 MiB granularity; can_use_symm_mem rejects out_bytes >= max.
        symm = _multimem_allgather(mapping, ((padded_rows * hidden_size * 2 >> 20) + 1) << 20)
        if symm.disabled:
            logger.warning(
                "Kimi K3 helix_a2a: symm_mem MULTIMEM all-gather unavailable at "
                f"tp_size={self.world_size}; using NCCL all-gather."
            )
        self._symm_allgather = None if symm.disabled else symm
        self._allreduce = AllReduce(
            mapping=mapping, strategy=allreduce_strategy, dtype=torch.bfloat16
        )
        logger.info_once(
            f"Kimi K3 helix_a2a: token-sliced MoE layers over tp_size={self.world_size}, "
            f"all-gather={'nccl' if symm.disabled else 'symm'}",
            key="kimi_k3_helix_a2a",
        )

    def all_rank_num_tokens(self, num_tokens: int) -> List[int]:
        return [-(-num_tokens // self.world_size)] * self.world_size

    def scatter_rows(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """This rank's chunk of a rank-replicated tensor (rows on dim -2)."""
        x, chunk = _helix_cp_pad(x, num_tokens, self.world_size)
        return x.narrow(-2, self.rank * chunk, chunk).contiguous()

    def all_gather(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[chunk, H]`` per rank -> ``[num_tokens, H]`` on every rank."""
        gathered = None
        if self._symm_allgather is not None:
            gathered = self._symm_allgather(x, dim=0)
        if gathered is None:
            gathered = allgather(x, self.mapping, dim=0)
        return gathered[:num_tokens]

    def reduce_scatter(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Row-parallel partial sums ``[num_tokens, H]`` -> this rank's ``[chunk, H]``."""
        return self.scatter_rows(self._allreduce(x), num_tokens)


# Sequence-parallel residual stream for head-sharded tensor-parallel prefill (on for the Kimi K3.1 context servers
# of a disaggregated deployment, see ``KimiLinearModel._seq_parallel``).
#
# Under TP without attention DP every rank otherwise runs the attention-residual mixing, the norms, the residual adds,
# the MoE router and the latent down / up projections on all rows of the batch. A forward carrying at least
# _SEQ_PARALLEL_MIN_TOKENS tokens keeps the residual stream on ``1 / tp`` of the rows per rank instead: each layer
# all-gathers the normed input of its column-parallel projections (the MoE all-gathers the latent and the router
# logits instead of recomputing them) and reduce-scatters the row-parallel outputs it used to all-reduce. Not bitwise
# (other reduction order). Eager forwards only; CUDA-graph and breakable-graph forwards keep the replicated path.
# Below a few thousand tokens the four NCCL collectives per layer cost more than the replicated row-wise work they
# remove.
_SEQ_PARALLEL_MIN_TOKENS = 8192


KIMI_K3_TRIM_GRAPH_PADDING_ENV = "TLLM_KIMI_K3_TRIM_GRAPH_PADDING"
"""``0`` keeps prefill-CUDA-graph padding rows in the attention-DP MoE dispatch (A/B escape hatch; default ``1``).

Read at import, so capture and replay cannot disagree about it.
"""

_TRIM_GRAPH_PADDING_ROWS = os.environ.get(KIMI_K3_TRIM_GRAPH_PADDING_ENV, "1") == "1"


def _fill_graph_padding_rows(rows: torch.Tensor, num_tokens: int) -> None:
    """``rows[i]`` = whether row ``i`` of the step is padding (``i >= num_tokens``)."""
    rows[:num_tokens].fill_(False)
    rows[num_tokens:].fill_(True)


def _mark_graph_padding_rows(rows: torch.Tensor) -> None:
    """``_fill_graph_padding_rows`` for the live batch; an eager break of a breakable prefill graph."""
    metadata = get_model_extra_attrs()["attention_metadata"]()
    _fill_graph_padding_rows(rows, metadata.num_tokens)


_maybe_bcg_mark_graph_padding_rows = eager_on_graph(_mark_graph_padding_rows)


class _KimiSeqParallel:
    """Row split of one forward's tokens over the TP group (rank ``r`` owns rows ``[r * rows, (r + 1) * rows)``).

    The split is padded to ``rows * tp_size`` rows; the padding rows of the last ranks start at zero and stay zero
    through every layer (zero rows normalize, route and project to zero), so they never reach a real row.
    """

    def __init__(self, mapping: Mapping, num_tokens: int) -> None:
        self.mapping = mapping
        self.num_tokens = num_tokens
        self.rows = -(-num_tokens // mapping.tp_size)
        self.padded_tokens = self.rows * mapping.tp_size
        self.row_begin = min(mapping.tp_rank * self.rows, num_tokens)
        self.row_end = min(self.row_begin + self.rows, num_tokens)

    def shard(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a replicated ``[num_tokens, ...]`` tensor, zero-padded to ``rows``."""
        if self.row_end - self.row_begin == self.rows:
            return full[self.row_begin : self.row_end]
        local = full.new_zeros((self.rows,) + tuple(full.shape[1:]))
        local[: self.row_end - self.row_begin].copy_(full[self.row_begin : self.row_end])
        return local

    def gather(self, locals_: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-gather ``[rows, ...]`` tensors into ``[padded_tokens, ...]`` (one grouped NCCL call)."""
        return allgather(locals_, self.mapping, dim=0)

    def reduce_scatter(self, partials: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sum ``[num_tokens or padded_tokens, ...]`` TP partials and keep this rank's ``[rows, ...]``."""
        padded = []
        for x in partials:
            if x.shape[0] != self.padded_tokens:
                pad = x.new_zeros((self.padded_tokens,) + tuple(x.shape[1:]))
                pad[: x.shape[0]].copy_(x)
                x = pad
            padded.append(x)
        return reducescatter(padded, self.mapping, dim=0)


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()

        from ..modules.kimi_k3_mla import KimiK3MLAAttention

        max_positions = int(
            os.environ.get(
                _KIMI_K3_MLA_MAX_POSITIONS_ENV,
                cfg.max_position_embeddings,
            )
        )
        self.layer_idx = layer_idx
        # KimiK3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials. reduce_output=False
        # leaves that reduction to the caller (helix_a2a reduce-scatter).
        mapping = model_config.mapping.mapping_for_full_rows()
        reduce_output = reduce_output and not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            AllReduce(
                mapping=mapping,
                strategy=model_config.allreduce_strategy,
                dtype=torch.bfloat16,
            )
            if reduce_output
            else None
        )
        # Set by the decoder layer for one forward (A_mid's in-place read): a pushed o_proj
        # all-reduce may be finished in place and returned as a LamportResult.
        self.ar_result_in_place = False
        # Set by the decoder layer for one forward (A_mid's fused finish): a pushed o_proj
        # all-reduce may be returned unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False
        # TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA: found by the previous layer's MoE block.
        _K3_MLA_LAYERS.setdefault(id(cfg), weakref.WeakValueDictionary())[layer_idx] = self
        self._q_b_l2_ranges: Tuple[Tuple[int, ...], List[Tuple[int, int]]] = ((), [])
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        fp8_kv_layer_ids = get_kimi_k3_fp8_kv_layer_ids(
            cfg,
            nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4,
            sparse_dsa=model_config.sparse_attention_config is not None
            and getattr(cfg, "dsa_config", None) is not None,
        )
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires the model-wide "
                    "KV cache dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        if getattr(cfg, "dsa_config", None) is not None and not cfg.is_dsa_layer(layer_idx):
            # Kimi K3.1 DSA covers only its dsa_layers; other MLA layers stay dense.
            attention_config.sparse_attention_config = None
        attention_config._frozen = model_config._frozen
        self.mixer = KimiK3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
        )

    def q_b_l2_prefetch_ranges(self) -> List[Tuple[int, int]]:
        """``(address, bytes)`` of the q_b projection's weight and scale
        (TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA); recomputed when the tensors move."""
        q_b_proj = getattr(self.mixer, "q_b_proj", None)
        tensors = [
            t
            for t in (getattr(q_b_proj, "weight", None), getattr(q_b_proj, "weight_scale", None))
            if isinstance(t, torch.Tensor) and t.is_cuda and t.numel()
        ]
        key = tuple(t.data_ptr() for t in tensors)
        if self._q_b_l2_ranges[0] != key:
            from ..kernels.blackwell.k3_l2_prefetch import tensor_range

            ranges = []
            for tensor in tensors:
                try:
                    ranges.append(tensor_range(tensor))
                except ValueError:
                    continue
            self._q_b_l2_ranges = (key, ranges)
        return self._q_b_l2_ranges[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        # BF16 o_proj's deferred finish: the BF16 o_proj hands its all-reduce to A_mid
        # (deferred only).
        bf16_finish = self.ar_finish_deferred and bf16_oproj_ar_finish_enabled(
            hidden_states.shape[0], *o_proj.weight.shape
        )
        ar_push = None
        if self._o_allreduce is not None and bf16_finish:
            ar_push = self._o_allreduce.lamport_push_target(
                hidden_states.shape[0],
                o_proj.out_features,
                in_place=self.ar_result_in_place,
                deferred=self.ar_finish_deferred,
            )
            if ar_push is not None and not ar_push.deferred:
                ar_push = None
            else:
                # BF16 o_proj push GEMM: the mixer's output projection may push it.
                self.mixer.bf16_ar_push = ar_push
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        self.mixer.bf16_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # A_mid's in-place read: the result may stay in the workspace for A_mid
            # (with A_mid's fused finish, A_mid may also finish it).
            out_shape = out.shape
            out = self._o_allreduce.finish_lamport_push(
                ar_push, in_place=self.ar_result_in_place, deferred=self.ar_finish_deferred
            )
            if not isinstance(out, (LamportResult, LamportPending)):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # A_mid's fused finish: o_proj ran without the push epilogue; A_mid finishes.
            self._o_allreduce.lamport_push_tensor(ar_push, out.reshape(ar_push.output.shape))
            out = self._o_allreduce.finish_lamport_push(ar_push, deferred=True)
        elif self._o_allreduce is not None and reduce_output:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


@nccl_window_tensor_scoped_module
class KimiLinearDecoderLayer(nn.Module):
    # TRTLLM_K3_L2PF_V2: the model's decode L2 prefetch controller (``None``: off).
    _l2pf: Optional[L2PrefetchController] = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.tp_size = model_config.mapping.tp_size
        dtype = torch.bfloat16
        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )
        # helix_a2a: an MoE layer keeps its residual stream and MoE on this
        # rank's token rows, attention on all rows (see KimiK3HelixA2A).
        self.token_sliced = self.is_moe and _is_helix_a2a(model_config)

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = KimiKDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping.mapping_for_full_rows(),
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
                reduce_output=not self.token_sliced,
            )
        else:
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                # CP original stashed by _setup_helix_mappings; None outside helix.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
                reduce_output=not self.token_sliced,
            )

        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            # Dense layers run on every row (helix_a2a slices only MoE layers).
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            if mlp_mapping.enable_attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, mlp_mapping.tp_size)
                if self.mlp_tp_size > mlp_mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(self.mlp_tp_size, mlp_mapping.gpus_per_node)
            mlp_model_config = copy.copy(model_config)
            mlp_model_config._frozen = False
            mlp_model_config.mapping = mlp_mapping
            mlp_model_config.quant_config = QuantConfig()
            mlp_model_config._frozen = model_config._frozen
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3. Attention DP replicates the
            # MLP because ranks own different tokens; otherwise the subgroup
            # is block-aligned and stays within one node.
            self.mlp = GatedMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                bias=False,
                activation=SituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                    use_fused_activation=True,
                ),
                dtype=dtype,
                config=mlp_model_config,
                overridden_tp_size=self.mlp_tp_size,
                reduce_output=self.mlp_tp_size > 1,
                layer_idx=layer_idx,
            )

        # Stock fused RMSNorm for the plain (whole-tensor) norms; numerics
        # are drop-in for KimiK3RMSNorm (fp32 variance, weight applied
        # after downcast, use_gemma=False).
        self.input_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3). The res norms stay
        # KimiK3RMSNorm: they are consumed field-wise (.weight/.eps) by
        # _apply_attn_res and the fused attn_res op, never called as
        # modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.mlp_res_norm = KimiK3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    def _capture_hidden_states(self, capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``."""
        capture[0].maybe_capture_hidden_states(capture[1], value, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
        helix_a2a: Optional[KimiK3HelixA2A] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows. The prefix sum (input and output) can be a
        ``_DeferredMoEResidual`` (PDL attn_res kernel in mode "2").

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.

        On a ``token_sliced`` layer M is this rank's chunk of the batch and
        ``helix_a2a`` moves rows between that chunk and the full batch.
        """
        sliced = helix_a2a if self.token_sliced else None
        num_tokens = attn_metadata.num_tokens if sliced is not None else None
        if self.token_sliced and helix_a2a is None:
            raise RuntimeError(
                f"Kimi K3 layer {self.layer_idx} is token-sliced but got no helix_a2a"
            )

        def _full_rows(t: torch.Tensor) -> torch.Tensor:
            return t if sliced is None else sliced.all_gather(t, num_tokens)

        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        if isinstance(hidden_states, _DeferredTailResidual):
            # The previous layer's fused-tail output, read from its
            # mailbox by this mix where the PDL kernel serves it (the DSpark tap path and other
            # shapes materialize it first).
            if capture is None and num_snapshots > 0:
                fused = _k3_attn_res_pdl_tail_mailbox(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                )
                if fused is not None:
                    hidden_states, normed = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        elif isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                )
                if fused is not None:
                    hidden_states, normed = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None:
            # The mixture tap needs the PRE-norm value, so the two steps split on captured layers.
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            # A property of the DRAFTER checkpoint, not a knob: a mismatch only lowers
            # acceptance, silently. hidden_states is the pre-norm attn_res mixture;
            # prefix_only wants the running prefix, already in hand as prefix_sum.
            tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
            self._capture_hidden_states(capture, _full_rows(tapped))
            hidden_states = self.input_layernorm(hidden_states)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # Attention is head-TP over every row: KDA rows are gathered in and its
        # row-sharded o_proj partials reduce-scattered back out here; helix MLA
        # does both itself (_helix_cp_allgather_input / _helix_cp_output_projection).
        # A_mid's in-place read: when A_mid below adds the attention output to the prefix sum,
        # the attention block may hand back its all-reduce result in place (a LamportResult).
        # A_mid must read it before the next all-reduce on the workspace re-arms that buffer:
        # the shared expert, whose down-projection all-reduce is the next one, forks after A_mid,
        # and its pushed finish kernel can only become resident once its producer has passed its
        # own griddepcontrol.wait (every grid before, A_mid included, has then completed).
        # (Not with KDA row slicing: its reduce-scatter needs the gathered tensor.)
        attn = self.linear_attn if self.is_kda else self.self_attn
        attn.ar_result_in_place = (
            prefix_sum is not None
            and not (self.is_kda and sliced is not None)
            and _k3_attn_res_lamport_applies(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        )
        # A_mid's fused finish: the same A_mid also finishes a one-shot pushed all-reduce
        # (a LamportPending); like the in-place result it lies on every path to the next
        # all-reduce on the workspace.
        attn.ar_finish_deferred = attn.ar_result_in_place and _K3_AR_FINISH_FUSED
        if self.is_kda:
            hidden_states = self.linear_attn(_full_rows(hidden_states), attn_metadata)
            if sliced is not None:
                hidden_states = sliced.reduce_scatter(hidden_states, num_tokens)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)
        attn.ar_result_in_place = False
        attn.ar_finish_deferred = False

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self._l2pf is not None and self.is_moe:
            # TRTLLM_K3_L2PF_V2: this layer's MoE front, once the attention output is mixed in.
            self._l2pf.fork("oproj_down", self.layer_idx, hidden_states.shape[0])
        if self.is_moe and _K3_FUSE_MOE_RESIDUAL and not torch.compiler.is_compiling():
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                getattr(attn_metadata, "all_rank_num_tokens", None),
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
            )
            return prefix_sum, num_snapshots
        if self.is_moe:
            # The fused tail may leave the residual update to the next layer's mix (a
            # _DeferredTailResidual); every other path returns the MoE output.
            mailbox_kwargs = (
                {"tail_mailbox_residual": prefix_sum}
                if sliced is None and not torch.compiler.is_compiling()
                else {}
            )
            if (
                mailbox_kwargs
                and _defer_moe_residual(prefix_sum.shape[0])
                and prefix_sum.shape == hidden_states.shape
            ):
                # PDL attn_res kernel in mode "2": rows the fused tail does not serve (above its
                # capacity) leave their MoE residual update to the next layer's mix as well.
                mailbox_kwargs["defer_residual_to"] = prefix_sum
            hidden_states = self.block_sparse_moe(
                hidden_states,
                sliced.all_rank_num_tokens(num_tokens)
                if sliced is not None
                else getattr(attn_metadata, "all_rank_num_tokens", None),
                **mailbox_kwargs,
            )
            if isinstance(hidden_states, (_DeferredTailResidual, _DeferredMoEResidual)):
                return hidden_states, num_snapshots
        else:
            hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    @property
    def supports_seq_parallel(self) -> bool:
        """Every row-parallel output of this layer can be handed back unreduced for a reduce-scatter."""
        if self.is_kda:
            mixer_ok = self.linear_attn._o_allreduce is not None
        else:
            mixer_ok = self.self_attn._o_allreduce is not None
        if self.is_moe:
            return mixer_ok and self.block_sparse_moe.supports_seq_parallel
        return mixer_ok

    def forward_seq_parallel(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        sp: _KimiSeqParallel,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """``forward`` with the residual stream, ``block_residual`` and every row-wise op on this rank's
        ``[sp.rows, hidden]`` rows (see ``_SEQ_PARALLEL_MIN_TOKENS``); the attention mixer and the MLP see all
        rows through an all-gather and return partials that are reduce-scattered.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]

        if capture is not None:
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
            capture[0].maybe_capture_hidden_states(
                capture[1], sp.gather([tapped])[0][: sp.num_tokens], None
            )
            hidden_states = self.input_layernorm(hidden_states)
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        full = sp.gather([hidden_states])[0][: sp.num_tokens]
        if self.is_kda:
            partial = self.linear_attn(full, attn_metadata, reduce_output=False)
        else:
            partial = self.self_attn(full, attn_metadata, reduce_output=False)
        hidden_states = sp.reduce_scatter([partial])[0]

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            hidden_states = self.block_sparse_moe.forward_seq_parallel(hidden_states, sp)
        elif self.mlp_tp_size == self.tp_size:
            full = sp.gather([hidden_states])[0]
            partial = self.mlp(
                full, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
            hidden_states = sp.reduce_scatter([partial])[0]
        else:
            # The dense MLP is sharded over a subgroup of the TP ranks (and replicated across subgroups), so
            # its partials do not sum over the whole TP group: run it on all rows and keep this rank's.
            full = self.mlp(sp.gather([hidden_states])[0])
            begin = sp.mapping.tp_rank * sp.rows
            hidden_states = full[begin : begin + sp.rows]

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


# ---------------------------------------------------------------------------
# TRTLLM_K3_L2PF_V2: decode L2 weight prefetch (modules/kimi_k3_l2_prefetch.py).
# ---------------------------------------------------------------------------


def _k3_l2pf_linear_tensors(module: Optional[nn.Module]) -> List[torch.Tensor]:
    """What a projection's GEMM reads: its weight scale (if any), then its weight."""
    if module is None:
        return []
    names = ("weight_scale", "weight")
    tensors = (getattr(module, name, None) for name in names)
    return [t for t in tensors if isinstance(t, torch.Tensor)]


def _k3_l2pf_kda_tensors(attn: KimiKDALinearAttention) -> List[torch.Tensor]:
    """A KDA layer's attention weights in the order its fused verify path reads them: QKVG,
    ``[f_a | b]``, f_b (inside the verify kernel), o_proj."""
    tensors = [attn._qkvg_proj_weight] + _k3_l2pf_linear_tensors(attn.qkvg_proj)
    tensors += [attn._bfa_proj_weight, attn.f_b_proj.weight]
    tensors += _k3_l2pf_linear_tensors(attn.o_proj)
    return [t for t in tensors if isinstance(t, torch.Tensor)]


def _k3_l2pf_mla_tensors(mixer: nn.Module, skip_q_b: bool) -> List[torch.Tensor]:
    """An MLA layer's attention weights in the order its generation path reads them:
    ``[q_a | kv_a]`` (or q_a, then kv_a), q_b (unless ``skip_q_b``), the q absorption W_UK, then
    after the decode core the V up-projection W_UV (all heads under Helix), the output gate and
    o_proj."""
    tensors = []
    if not getattr(mixer, "fuse_qkv_a_proj", True):
        tensors += _k3_l2pf_linear_tensors(getattr(mixer, "q_a_proj", None))
    tensors += _k3_l2pf_linear_tensors(getattr(mixer, "kv_a_proj_with_mqa", None))
    if not skip_q_b:
        tensors += _k3_l2pf_linear_tensors(getattr(mixer, "q_b_proj", None))
    # The BMMs read the BF16 copies where they exist, else the FP8 weights and their scales.
    w_uk = getattr(mixer, "k_b_proj_trans_dequant", None)
    if not isinstance(w_uk, torch.Tensor):
        w_uk = (
            getattr(mixer, "k_b_proj_trans_scale", None),
            getattr(mixer, "k_b_proj_trans", None),
        )
    w_uv = getattr(mixer, "helix_v_b_proj_full", None)
    if not isinstance(w_uv, torch.Tensor):
        w_uv = getattr(mixer, "v_b_proj_dequant", None)
    if not isinstance(w_uv, torch.Tensor):
        w_uv = (getattr(mixer, "v_b_proj_scale", None), getattr(mixer, "v_b_proj", None))
    for weights in (w_uk, w_uv):
        tensors += list(weights) if isinstance(weights, tuple) else [weights]
    tensors += _k3_l2pf_linear_tensors(getattr(mixer, "g_proj", None))
    tensors += _k3_l2pf_linear_tensors(getattr(mixer, "o_proj", None))
    return [t for t in tensors if isinstance(t, torch.Tensor)]


def _k3_l2pf_front_covered(moe: KimiK3MoERuntime, num_tokens: int) -> bool:
    """Whether ``TRTLLM_K3_MOE_L2_PREFETCH=front`` already prefetches this MoE layer's front and
    shared-expert weights for ``num_tokens`` tokens (from the MoE layer before it)."""
    if (
        "front" not in _K3_MOE_L2_PREFETCH
        or num_tokens > _K3_MOE_L2_PREFETCH_MAX_M
        or moe.latent_down_stream is None
    ):
        return False
    earlier = [idx for idx in moe._moe_group.keys() if idx < moe.layer_idx]
    prev = moe._moe_group.get(max(earlier)) if earlier else None
    return prev is not None and prev.latent_down_stream is moe.latent_down_stream


def _k3_l2pf_q_b_covered(prev: Optional[KimiLinearDecoderLayer], num_tokens: int) -> bool:
    """Whether ``TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA`` already prefetches the q_b weight of the MLA
    layer after ``prev`` for ``num_tokens`` tokens (from ``prev``'s MoE block)."""
    if (
        not _K3_MOE_L2_PREFETCH_NEXT_MLA
        or num_tokens > _K3_MOE_L2_PREFETCH_MAX_M
        or prev is None
        or not prev.is_moe
    ):
        return False
    moe = prev.block_sparse_moe
    nxt = moe._next_moe_layer()
    return (
        moe.latent_down_stream is not None
        and nxt is not None
        and nxt.latent_down_stream is moe.latent_down_stream
    )


def _k3_l2pf_tensors(
    model_ref: "weakref.ref[KimiLinearModel]", point: str, layer_idx: int, num_tokens: int
) -> List[torch.Tensor]:
    """The weights ``TRTLLM_K3_L2PF_V2`` prefetches at ``point`` of layer ``layer_idx`` for
    ``num_tokens`` tokens, in the order their kernels read them, without those an older prefetch
    switch already covers there: ``attention_moe`` the layer's MoE front (router, latent_down) and
    shared expert, ``oproj_down`` its MoE front, ``next_attention`` the next layer's attention. Where the sharded MoE
    front serves the rows, the MoE front is its weight (this rank's head rows)."""
    model = model_ref()
    if model is None:
        return []
    layers = {layer.layer_idx: layer for layer in model.layers}
    if point == "next_attention":
        nxt = layers.get(layer_idx + 1)
        if nxt is None:
            return []
        if nxt.is_kda:
            return _k3_l2pf_kda_tensors(nxt.linear_attn)
        skip_q_b = _k3_l2pf_q_b_covered(layers.get(layer_idx), num_tokens)
        return _k3_l2pf_mla_tensors(nxt.self_attn.mixer, skip_q_b)
    layer = layers.get(layer_idx)
    if layer is None or not layer.is_moe:
        return []
    moe = layer.block_sparse_moe
    if _k3_l2pf_front_covered(moe, num_tokens):
        return []
    sharded = moe.k31_sharded_front_weight(num_tokens)
    if sharded is not None:
        # The sharded MoE front serves these rows: the front reads its own copy of this rank's head
        # rows, not the replicated router and latent_down.
        tensors = [sharded]
    else:
        tensors = [moe.gate.weight] + _k3_l2pf_linear_tensors(moe.routed_expert_down_proj)
    if point == "attention_moe":
        # The shared gate_up rows are part of the front's weight where the front computes them.
        names = ("gate_up_proj", "down_proj")
        if sharded is not None and moe.k31_sharded_front_reads_shared_gate_up(num_tokens):
            names = ("down_proj",)
        for name in names:
            tensors += _k3_l2pf_linear_tensors(getattr(moe.shared_experts, name, None))
    return tensors


def _k3_l2pf_controller(model: "KimiLinearModel") -> Optional[L2PrefetchController]:
    """``TRTLLM_K3_L2PF_V2`` (``_K31_L2PF``): the model's decode L2 prefetch controller, wired into its
    decoder layers, MoE blocks and the attention blocks of MoE layers; ``None`` when the switch is off."""
    if not _K31_L2PF:
        return None
    config = L2PrefetchConfig.from_env(points=frozenset(L2PF_POINTS))
    controller = L2PrefetchController(
        config, functools.partial(_k3_l2pf_tensors, weakref.ref(model))
    )
    for layer in model.layers:
        layer._l2pf = controller
        if layer.is_moe:
            layer.block_sparse_moe._l2pf = controller
            attn = layer.linear_attn if layer.is_kda else layer.self_attn.mixer
            attn.l2pf_attention_hook = functools.partial(
                controller.fork, "attention_moe", layer.layer_idx
            )
    logger.info(
        f"TRTLLM_K3_L2PF_V2: decode L2 prefetch at {sorted(config.points)}, CTAs {config.ctas}"
    )
    return controller


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiLinearModel(DecoderModel):
    # TRTLLM_K3_L2PF_V2: the decode L2 weight prefetch controller (``None``: off).
    _l2pf: Optional[L2PrefetchController] = None

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16
        if getattr(cfg, "dsa_config", None) is not None:
            # Kimi K3.1: the MLA output gate is one kernel at every batch size (same bits).
            from ..modules.kimi_k3_mla import output_gate

            output_gate.enable_small_batches()
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
                # The attention-residual kernel picks its tuned CTAs per token (same bits).
                from ..kernels.blackwell import k3_attn_res

                k3_attn_res.enable_tuned_cluster()

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # Rows are rank-replicated up to the first token-sliced layer (the dense
        # layers stay on all rows), and sliced from there to the end.
        sliced = [layer.token_sliced for layer in self.layers]
        self.helix_a2a_first_layer = sliced.index(True) if any(sliced) else len(sliced)
        if not all(sliced[self.helix_a2a_first_layer :]):
            raise ValueError(
                "Kimi K3 helix_a2a needs every layer after the first MoE layer to be MoE"
            )
        # After the layers: MegaMoE bootstraps torch.distributed under MPI,
        # which the symm_mem all-gather needs.
        self.helix_a2a = (
            KimiK3HelixA2A(
                model_config.mapping.mapping_for_full_rows(),
                cfg.hidden_size,
                model_config.max_num_tokens,
                model_config.allreduce_strategy,
            )
            if any(sliced)
            else None
        )
        self._l2pf = _k3_l2pf_controller(self)
        self.norm = RMSNorm(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # KimiK3RMSNorm (not RMSNorm): consumed field-wise (.weight/.eps)
        # by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        mapping = model_config.mapping
        self._seq_parallel_mapping = mapping
        self._trim_graph_padding_rows = mapping.enable_attention_dp and _TRIM_GRAPH_PADDING_ROWS
        self._graph_padding_rows: Optional[torch.Tensor] = None
        # Kimi K3.1 context servers (aggregated serving keeps the replicated residual stream).
        seq_parallel_wanted = (
            getattr(cfg, "dsa_config", None) is not None and disagg_role() == "context"
        )
        self._seq_parallel = (
            seq_parallel_wanted
            and mapping.tp_size > 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
            and getattr(model_config, "_helix_mapping_with_cp", None) is None
            and all(layer.supports_seq_parallel for layer in self.layers)
        )
        if seq_parallel_wanted:
            logger.info_once(
                f"Kimi K3 sequence-parallel prefill: {'on' if self._seq_parallel else 'unsupported here, off'} "
                f"(tp={mapping.tp_size}, attention_dp={mapping.enable_attention_dp}, "
                f"min_tokens={_SEQ_PARALLEL_MIN_TOKENS})",
                key="kimi_k3_seq_parallel",
            )

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = attn_metadata.num_tokens if self.helix_a2a is not None else None
        if self.helix_a2a is not None and hidden_states.shape[0] != num_tokens:
            # Rows are split by the metadata token count on every rank.
            raise ValueError(
                f"Kimi K3 helix_a2a does not support padded batches (got "
                f"{hidden_states.shape[0]} rows, metadata says {num_tokens} tokens)."
            )
        if self._l2pf is not None:
            # TRTLLM_K3_L2PF_V2: the layers' prefetch forks act in generation-only steps.
            self._l2pf.begin_step(attn_metadata.num_tokens, attn_metadata.num_contexts == 0)
        # A step a prefill CUDA graph covers (its replay, its warmups and an eager run of a
        # bucket padded for one) keeps the replicated path, so warmup and capture see one path.
        if (
            self._seq_parallel
            and self.helix_a2a is None
            and hidden_states.shape[0] >= _SEQ_PARALLEL_MIN_TOKENS
            and not attn_metadata.is_cuda_graph
            and not is_in_breakable_cuda_graph()
            and attn_metadata.padded_num_tokens is None
            and not get_per_request_prefill_cuda_graph_flag()
            and not torch.compiler.is_compiling()
        ):
            return self._forward_seq_parallel(
                attn_metadata,
                hidden_states,
                _KimiSeqParallel(self._seq_parallel_mapping, hidden_states.shape[0]),
                spec_metadata,
            )

        extra_attrs = get_model_extra_attrs()
        graph_padding_rows = (
            self._mark_graph_padding_rows(hidden_states.shape[0], attn_metadata)
            if extra_attrs is not None
            else None
        )
        if graph_padding_rows is None:
            return self._forward_layers(attn_metadata, hidden_states, spec_metadata, num_tokens)
        extra_attrs[MOE_GRAPH_PADDING_ROWS_ATTR] = graph_padding_rows
        try:
            return self._forward_layers(attn_metadata, hidden_states, spec_metadata, num_tokens)
        finally:
            extra_attrs.pop(MOE_GRAPH_PADDING_ROWS_ATTR, None)

    def _mark_graph_padding_rows(
        self, num_rows: int, attn_metadata: AttentionMetadata
    ) -> Optional[torch.Tensor]:
        """``[num_rows]`` bool mask of this step's prefill-graph padding rows, for the MoE dispatch to skip.

        Attention DP only: there the routed MoE ships each row to its experts' ranks, so a padding row costs
        all-to-all bytes and expert GEMM rows (and every padding row routes alike, onto the same experts). The mask
        lives in one persistent buffer that an eager break refreshes from the live batch on every graph replay;
        ``None`` when the step is not a prefill-graph step.
        """
        if not (self._trim_graph_padding_rows and get_per_request_prefill_cuda_graph_flag()):
            return None
        buffer = self._graph_padding_rows
        if buffer is None or buffer.shape[0] < num_rows:
            if torch.cuda.is_current_stream_capturing():
                # The warmups of the largest bucket run before any capture and size the buffer.
                logger.warning_once(
                    f"Kimi K3 prefill graph: no {num_rows}-row padding mask at capture; "
                    "the MoE dispatch keeps the padding rows",
                    key="kimi_k3_graph_padding_rows",
                )
                return None
            buffer = torch.empty(num_rows, dtype=torch.bool, device="cuda")
            self._graph_padding_rows = buffer
        rows = buffer[:num_rows]
        if is_in_breakable_cuda_graph():
            _maybe_bcg_mark_graph_padding_rows(rows)
        else:
            _fill_graph_padding_rows(rows, attn_metadata.num_tokens)
        return rows

    def _forward_layers(
        self,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
        spec_metadata,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """``forward`` past the embedding on the replicated residual stream (``num_tokens``: the
        helix_a2a row split's token count, None without helix_a2a)."""
        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # DFlash/DSpark hidden-state capture. The drafter is distilled on
            # the aggregated stream value -- the pre-norm softmax mixture its
            # next consumer sees -- not on the raw prefix sum a layer returns,
            # which is SGLang's fallback for models without the
            # attention-residual scheme. Capturing the prefix sum costs 4.5pt
            # of draft acceptance on K3 + RadixArk DSpark (AR 66.9% -> 71.4%).
            # The tap fires inside layer i+1, which computes that tensor
            # anyway; see its forward docstring. Ground truth: SGLang
            # kimi_k3.py:2697 _dspark_capture_stream, attn_residual.py:313
            # aggregate_stream_torch.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            if i == self.helix_a2a_first_layer:
                # Rows are rank-replicated up to here: keep this rank's chunk.
                hidden_states = _materialize_prefix_sum(hidden_states)
                hidden_states = self.helix_a2a.scatter_rows(hidden_states, num_tokens)
                block_residual = self.helix_a2a.scatter_rows(block_residual, num_tokens)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
                helix_a2a=self.helix_a2a,
            )
        if self._l2pf is not None:
            # TRTLLM_K3_L2PF_V2: the last MoE layer normally joined the prefetch streams already.
            self._l2pf.join()
        hidden_states = _materialize_prefix_sum(hidden_states)
        sliced = self.helix_a2a is not None

        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                if sliced:
                    tail = self.helix_a2a.all_gather(tail, num_tokens)
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        hidden_states = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        if sliced:
            hidden_states = self.helix_a2a.all_gather(hidden_states, num_tokens)
        return hidden_states

    def _forward_seq_parallel(
        self,
        attn_metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor,
        sp: _KimiSeqParallel,
        spec_metadata=None,
    ) -> torch.Tensor:
        """``forward`` with the residual stream on this rank's rows (``_SEQ_PARALLEL_MIN_TOKENS``); the final
        norm output is all-gathered back to ``[num_tokens, hidden]``."""
        hidden_states = sp.shard(inputs_embeds)
        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots, sp.rows, hidden_states.shape[1]
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer.forward_seq_parallel(
                hidden_states, block_residual, num_snapshots, attn_metadata, sp, capture=capture
            )

        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(
                    last.layer_idx, sp.gather([tail])[0][: sp.num_tokens], None
                )

        out = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        out = sp.gather([out])[0][: sp.num_tokens]
        return out


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


_FP8_BLOCK_SCALE_SUFFIX = "_scale"
_FP8_BLOCK = 128


def _fp8_block_scale_key(weight_key: str) -> str:
    return weight_key + _FP8_BLOCK_SCALE_SUFFIX


def _checkpoint_fp8_pair(ckpt_key: str, src: torch.Tensor, weights):
    """``(fp8 weight, fp32 128x128 block scale)`` if this tensor is FP8_PB_WO.

    ``None`` for an ordinary (BF16) checkpoint tensor. Raises when an FP8
    tensor has no companion scale rather than letting the caller fall through
    to a cast that would silently drop it.
    """
    if src.dtype != torch.float8_e4m3fn:
        return None
    scale_key = _fp8_block_scale_key(ckpt_key)
    if scale_key not in weights and scale_key + "_inv" in weights:
        scale_key += "_inv"
    if scale_key not in weights:
        raise KeyError(
            f"Kimi K3: {ckpt_key} is FP8 E4M3 but has no {scale_key} or {scale_key}_inv; refusing "
            "to load it as if it were unquantized."
        )
    scale = _materialize(weights[scale_key]).float()
    # The checkpoint stores the block scale 4-D as
    # [ceil(N/128), 1, ceil(K/128), 1]. Normalize here, once, so every
    # consumer sees the plain 2-D [n_blocks_m, n_blocks_k] that both the
    # dequantization below and deep_gemm's transform_sf_into_required_layout
    # expect -- the latter asserts on rank and gave an unhelpful
    # ``assert sf.dim() == ...`` when handed the raw 4-D tensor.
    if scale.dim() == 4:
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    return src, scale


def _dequantize_fp8_block_scaled(ckpt_key: str, src: torch.Tensor, weights) -> torch.Tensor:
    """Apply checkpoint scales for projections explicitly configured as BF16.

    Checkpoint-native FP8 attention bypasses this loader. This handles an
    excluded projection or another unquantized trunk parameter whose source
    tensor is quantized; casting the codes alone would discard its scales.
    """
    pair = _checkpoint_fp8_pair(ckpt_key, src, weights)
    if pair is None:
        return src
    src, scale = pair
    out = src.to(torch.float32)
    if scale.numel() == 1:
        return (out * scale.reshape(())).to(torch.bfloat16)
    expanded = scale.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    if expanded.shape[0] < out.shape[0] or expanded.shape[1] < out.shape[1]:
        raise ValueError(
            f"Kimi K3: {_fp8_block_scale_key(ckpt_key)} covers "
            f"{tuple(expanded.shape)} but {ckpt_key} is {tuple(out.shape)}."
        )
    return (out * expanded[: out.shape[0], : out.shape[1]]).to(torch.bfloat16)


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    # ``[:]`` is how a lazy slice is realized, but it is invalid on a 0-dim
    # entry. The NVFP4 checkpoint stores weight_scale_2 / input_scale as
    # scalars, so realize those entries with ``[()]`` instead.
    get_shape = getattr(value, "get_shape", None)
    if get_shape is not None and len(get_shape()) == 0:
        return value[()]
    return value[:]


_SAFETENSORS_DTYPES = {
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}


def read_safetensors_slices(
    path: str, names: Iterable[str], get_buffer: Callable[[int], torch.Tensor]
) -> Optional[Dict[str, torch.Tensor]]:
    """Read the named tensors of one safetensors file with a few large preadv calls.

    The tensors' byte ranges are merged across gaps under 1 MiB and read into the
    uint8 tensor ``get_buffer(nbytes)`` returns (at least ``nbytes`` long). The result
    maps every name found in the file to a view of that buffer, valid until the buffer
    is reused. Each range starts at a buffer offset congruent to its file offset mod
    64, so the views keep the file's alignment. Returns None when a requested tensor
    has a dtype this reader does not map or the file ends early.
    """
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    data_start = 8 + header_len
    entries = []
    for name in names:
        meta = header.get(name)
        if meta is None:
            continue
        dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            return None
        start, end = meta["data_offsets"]
        entries.append((start, end, name, dtype, meta["shape"]))
    entries.sort()
    ranges = []
    for start, end, *_ in entries:
        if ranges and start - ranges[-1][1] < (1 << 20):
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    placements = []
    buf_size = 0
    for start, end in ranges:
        buf_off = (buf_size + 63) // 64 * 64 + start % 64
        placements.append(buf_off)
        buf_size = buf_off + end - start
    buf = get_buffer(buf_size)
    view = memoryview(buf.numpy())
    fd = os.open(path, os.O_RDONLY)
    try:
        for (start, end), buf_off in zip(ranges, placements):
            done = 0
            while done < end - start:
                got = os.preadv(
                    fd,
                    [view[buf_off + done : buf_off + end - start]],
                    data_start + start + done,
                )
                if got <= 0:
                    return None
                done += got
    finally:
        os.close(fd)
    tensors = {}
    range_idx = 0
    for start, end, name, dtype, shape in entries:
        while end > ranges[range_idx][1]:
            range_idx += 1
        buf_off = placements[range_idx] + start - ranges[range_idx][0]
        tensors[name] = buf[buf_off : buf_off + end - start].view(dtype).view(shape)
    return tensors


def _configure_k31_decode(cfg, model_config: ModelConfig) -> None:
    """Set this process's Kimi K3.1 decode switches when the model is built.

    A Kimi K3.1 (DSA) generation server of a disaggregated deployment runs the decode fusions:

    - attention TP: the latent_down stream, the fused MoE tail, separated routing, the packed and
      sharded MoE fronts (with the front's shared-expert part; its routing sort at TP 32), the decode
      L2 prefetch, the PDL attn_res kernel (mode 1), the o_proj all-reduce chain (pushed all-reduces,
      the BF16 push GEMM, A_mid's in-place read, fused finish and early poll) and, at TP <= 16, the
      attn_res kernel's early release up to 8 rows;
    - attention DP: the PDL attn_res kernel that defers the MoE residual (mode 2) and the MoE block's
      fused residual update.

    Both cap the PDL kernel at min(max_num_tokens, 192 with attention TP / 256 with attention DP)
    rows. Kimi K3, the context servers and aggregated serving run none of them.

    A Kimi K3.1 attention-DP context server routes its MoE with noaux_tc instead of the routing tail
    (``KIMI_K31_ROUTER_TAIL_DEP``) and takes the ``split`` attn_res topology (``KIMI_K3_ATTN_RES_TOPOLOGY``),
    unless the env sets them.
    """
    global _K31_LATENT_DOWN_STREAM, _K31_FUSED_MOE_TAIL, _K31_SEPARATED_ROUTING, _K31_L2PF
    global _K31_PACKED_MOE_FRONT, _K31_SHARDED_MOE_FRONT, _K31_SHARDED_MOE_FRONT_SHARED
    global \
        _K31_SHARDED_MOE_FRONT_SORT, \
        _K3_AR_PUSH_AMID, \
        _K3_AR_FINISH_FUSED, \
        _K3_ULL_OPROJ_SANDWICH
    global _K3_ATTN_RES_PDL_MODE, _K3_ATTN_RES_PDL, _K3_ATTN_RES_PDL_MAX_M, _K3_FUSE_MOE_RESIDUAL
    global _K31_ROUTER_TAIL_DEP, _ATTN_RES_TOPOLOGY, _FUSED_ATTN_RES_MAX_TOKENS
    mapping = model_config.mapping
    k31 = getattr(cfg, "dsa_config", None) is not None
    k31_gen = k31 and disagg_role() == "generation"
    tep = k31_gen and not mapping.enable_attention_dp
    dep = k31_gen and mapping.enable_attention_dp
    _K31_LATENT_DOWN_STREAM = _K31_FUSED_MOE_TAIL = _K31_SEPARATED_ROUTING = _K31_L2PF = tep
    _K31_PACKED_MOE_FRONT = _K31_SHARDED_MOE_FRONT = _K31_SHARDED_MOE_FRONT_SHARED = tep
    _K31_SHARDED_MOE_FRONT_SORT = tep and mapping.tp_size == 32
    _K3_AR_PUSH_AMID = _K3_AR_FINISH_FUSED = _K3_ULL_OPROJ_SANDWICH = tep
    _K3_ATTN_RES_PDL_MODE = "1" if tep else "2" if dep else "0"
    _K3_ATTN_RES_PDL = k31_gen
    _K3_ATTN_RES_PDL_MAX_M = min(model_config.max_num_tokens, 256 if dep else 192) if k31_gen else 0
    _K3_FUSE_MOE_RESIDUAL = dep
    # Kimi K3.1 attention-DP context servers: noaux_tc routing and the split attn_res topology unless the env sets them.
    dep_ctx = k31 and mapping.enable_attention_dp and disagg_role() == "context"
    _K31_ROUTER_TAIL_DEP = (
        os.environ.get("KIMI_K31_ROUTER_TAIL_DEP", "0" if dep_ctx else "1") == "1"
    )
    _ATTN_RES_TOPOLOGY = _read_attn_res_topology("split" if dep_ctx else "per_token")
    _FUSED_ATTN_RES_MAX_TOKENS = _read_fused_attn_res_max_tokens()
    configure_k31_oproj_all_reduce(tep)
    if tep and mapping.tp_size <= 16:
        from ..kernels.blackwell import k3_attn_res

        # Early trigger: release the dependent grid right after the
        # attn_res kernel's griddepcontrol.wait for calls of up to 8 rows.
        k3_attn_res._EARLY_TRIGGER_MAX_M = 8
    if k31_gen:
        logger.info(
            f"Kimi K3.1 {'attention-TP' if tep else 'attention-DP'} generation server: decode "
            f"fusions on (PDL attn_res mode {_K3_ATTN_RES_PDL_MODE} up to "
            f"{_K3_ATTN_RES_PDL_MAX_M} rows)"
        )


@register_auto_model("KimiLinearForCausalLM")
class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE).

    Serves text-only ``kimi_linear`` checkpoints directly, and is reused as the
    text backbone by the multimodal ``KimiK3ForConditionalGeneration`` wrapper
    (``modeling_kimi_k3_vl``). The composite ``KimiK3ForConditionalGeneration``
    architecture is registered by that wrapper, not here."""

    mamba_metadata_cls = KimiK3MambaMetadata

    def __init__(self, model_config: ModelConfig):
        mlp_fp8 = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "0")
        if mlp_fp8 not in ("0", "1"):
            raise ValueError(
                f"{_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV} must be 0 or 1; got {mlp_fp8!r}"
            )
        self._fp8_weight_read_moe_mlp = mlp_fp8 == "1"
        cfg = _get_text_config(model_config.pretrained_config)
        _configure_k31_decode(cfg, model_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        # Supported spec-dec modes:
        # - SA (suffix automaton): one-engine in-forward drafting, no draft
        #   weights; the KDA/MLA verify paths below implement multi-token
        #   verification for it.
        # - DFlash: external-drafter parallel drafting; the drafter is a
        #   separate dense checkpoint (K2.7-Code-DFlash schema) consumed by
        #   the generic DFlashForCausalLM wrapper, and the target only has
        #   to expose per-layer hidden states via maybe_capture_hidden_states
        #   (see KimiLinearModel.forward).
        # - DSpark: the same external-drafter flow with the Markov and
        #   confidence heads enabled (RadixArk/Kimi-K3-DSpark and friends).
        #   The target side is identical -- the capture in
        #   KimiLinearModel.forward is unconditional -- so this gate is the
        #   only place the mode has to be admitted.
        # Modes needing draft heads (MTP/Eagle) are blocked until a
        # draft-head checkpoint exists.
        assert (
            spec_config is None
            or spec_config.spec_dec_mode.is_sa()
            or spec_config.spec_dec_mode.is_dflash()
            or spec_config.spec_dec_mode.is_dspark()
        ), "Kimi K3 supports speculative decoding only with SA, DFlash or DSpark"
        super().__init__(
            KimiLinearModel(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def _setup_helix_mappings(
        self,
        model_config: ModelConfig,
        cfg: "PretrainedConfig",
        spec_config: Optional["DecodingBaseConfig"],
    ) -> None:
        """Validate helix preconditions and stage the dual-mapping swap.

        DeepseekV3 pattern: the MLA layers keep the CP original; everything
        else is built against the repurposed mapping (CP ranks become TP
        ranks). Sets ``mapping_with_cp`` (restored after construction) and
        ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
        None outside helix.
        """
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        if not model_config.mapping.has_cp_helix():
            return
        # Helix + attention DP = helix_a2a: KDA keeps head-TP state across the
        # CP group (every CP rank holds the same requests) and only the MoE
        # layers run attention-DP on ceil(T / cp) token rows each.
        if model_config.mapping.enable_attention_dp and model_config.mapping.tp_size != 1:
            raise NotImplementedError(
                "Kimi K3 helix with enable_attention_dp requires tp_size=1 "
                f"(got tp_size={model_config.mapping.tp_size}): KDA and MLA are "
                "head-TP over the CP group, so there is no request-level DP to add."
            )
        if spec_config is not None:
            # Helix supports only the standalone DSpark drafter (verified on
            # the V2 superblock ledger); reject everything else loudly rather
            # than let an unsupported spec mode run silently wrong.
            if not spec_config.spec_dec_mode.is_dspark():
                raise ValueError(
                    "Kimi K3 helix supports speculative decoding only with "
                    f"DSpark (standalone drafter); got "
                    f"{spec_config.decoding_type!r}."
                )
            # The SpeculationGate acceptance-rate trip permanently disables
            # speculation mid-flight while enable_spec_decode stays True;
            # in-flight helix requests then fall into the plain generation
            # loop whose position math (total_input_len_cp +
            # py_decoding_iter - 1) is stale once any draft token was
            # accepted -> silently wrong RoPE positions and KV slots. Reject
            # the trip wires until that loop is helix-group aware.
            if (
                spec_config.acceptance_rate_window_size is not None
                or spec_config.acceptance_rate_threshold is not None
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "acceptance-rate gate (acceptance_rate_window_size / "
                    "acceptance_rate_threshold): dynamically disabling "
                    "speculation mid-flight leaves helix requests on a "
                    "single-token position formula."
                )
            # max_concurrency is the same trip wire by another name: the
            # drafter re-evaluates should_use_spec_decode on every scheduling
            # iteration and flips enable_spec_decode off as soon as the active
            # batch exceeds the cap. In-flight helix requests then take the
            # plain generation loop, whose position formula counts ITERATIONS
            # (total_input_len_cp + py_decoding_iter - 1) rather than
            # committed tokens, so it is stale by however many draft tokens
            # were accepted -- a wrong RoPE position, and across a ledger page
            # boundary a KV write to the wrong CP rank. Mirror the drafter's
            # own "unset" test (Drafter.should_use_spec_decode returns True
            # when max_concurrency is None) so an unset value is not rejected.
            if spec_config.max_concurrency is not None:
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "concurrency cutoff (max_concurrency): disabling "
                    "speculation above the cap leaves in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration, which accepted draft "
                    "tokens break."
                )
            # draft_len_schedule is the user-facing alternative to
            # max_concurrency (llm_args rejects setting both) and reaches the
            # same end by a route that does not go through
            # should_use_spec_decode at all: py_executor turns speculation off
            # directly once the schedule yields draft_len 0 for the active
            # batch size. Guarding only max_concurrency would leave this door
            # open. Skip the schedule that llm_args synthesized from
            # max_concurrency, so a config that set only that field raises the
            # message above naming the field the user actually wrote.
            if (
                spec_config.draft_len_schedule is not None
                and not spec_config._translated_from_max_concurrency
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the dynamic draft-length "
                    "schedule (draft_len_schedule): a batch size past the "
                    "last entry drops the draft length to 0 and turns "
                    "speculation off mid-run, leaving in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration."
                )
        cp = model_config.mapping.cp_size
        repurposed_tp = model_config.mapping.tp_size * cp
        if cfg.num_attention_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
                f"to divide the MLA head count ({cfg.num_attention_heads})."
            )
        kda_heads = cfg.linear_attn_config["num_heads"]
        if kda_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
                f"divide the KDA head count ({kda_heads})."
            )
        # MoE splits apply to the repurposed tp*cp group (helix
        # moe_world_size = tp*cp); default EP-only. The Mapping constructor
        # skips its product check when both sizes are 1, so validate here.
        moe_ep = repurposed_tp
        if model_config.mapping.moe_tp_ep_user_specified:
            moe_tp = model_config.mapping.moe_tp_size
            moe_ep = model_config.mapping.moe_ep_size
            if moe_tp * moe_ep != repurposed_tp:
                raise ValueError(
                    f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                    f"moe_expert_parallel_size ({moe_ep}) must equal "
                    f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                    "repurposed tp*cp group."
                )
        if cfg.num_experts and cfg.num_experts % moe_ep != 0:
            raise ValueError(
                f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
                f"divide the routed expert count ({cfg.num_experts}): each "
                "EP rank of the repurposed tp*cp group holds whole experts."
            )
        self.mapping_with_cp = copy.deepcopy(model_config.mapping)
        repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
        # repurpose passes resolved moe sizes, which the Mapping constructor
        # mistakes for user-specified values; restore the flag.
        repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
        # load_weights shard selection must use this tp_rank; the restored
        # CP original's tp_rank is 0 on every rank.
        self._repurposed_tp_mapping = repurposed
        model_config._frozen = False
        model_config.mapping = repurposed
        # Side-channel for the MLA layers (avoids threading a kwarg through
        # every intermediate signature).
        model_config._helix_mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Keep block reuse opt-in because its combination with SA
        #   speculative decoding is not validated. Both settings use V2
        #   with the default use_kv_cache_manager_v2=True. The "auto"
        #   setting follows the model preference and can resolve to V1.
        # - tokens_per_block=64: keep the validated K3 page size. Earlier
        #   releases needed it to avoid a legacy MLA fallback that could not
        #   serve attention-DP's 96 query heads. The current FlashInfer
        #   TRTLLM-Gen backend declines H=96 at either page size, allowing
        #   the updated legacy TRTLLM-Gen autotuner to select a Q16 kernel.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """Kimi K3.1 DSA keeps its indexer K-cache in a KV cache manager V2 buffer.

        Without a ``dsa_config`` (Kimi K3) the V1 hybrid managers stay available.
        """
        if pretrained_config is None:
            return False
        return getattr(_get_text_config(pretrained_config), "dsa_config", None) is not None

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    # ------------------------------------------------------------------
    # Weight loading (streams the 1.5TB checkpoint; only the rank-local
    # expert slice of each MoE layer is kept: whole experts under MoE EP,
    # the intra-expert intermediate shard of ALL experts under MoE TP —
    # in the TP case every expert tensor is read and sliced, so expect a
    # correspondingly longer load).
    # ------------------------------------------------------------------

    def _attention_fp8_linears(self) -> list[tuple[str, TrtllmLinear, nn.Module]]:
        """Collect checkpoint-backed FP8 attention projections for weight loading.

        The loader uses this list to enumerate required weight/scale keys, exclude
        these parameters from ordinary trunk loading, and load FP8 codes and scales
        together. Derived QKVG fusion is excluded because it is built after loading.

        Returns:
            Tuples of runtime module path, linear, and owning attention module.
        """
        projections = []
        for index, layer in enumerate(self.model.layers):
            if not _has_weights(layer):
                continue
            if layer.is_kda:
                attention = layer.linear_attn
                scope = f"model.layers.{index}.linear_attn"
            else:
                attention = layer.self_attn.mixer
                scope = f"model.layers.{index}.self_attn.mixer"
            for attr, module in attention.named_children():
                if attr == "qkvg_proj":
                    # Decode fusion is derived from separately loaded checkpoint projections.
                    continue
                if isinstance(module, TrtllmLinear) and module.has_fp8_block_scales:
                    projections.append((f"{scope}.{attr}", module, attention))
        return projections

    def _load_attention_fp8(self, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        """Load checkpoint FP8 codes/scales without a BF16 projection intermediate."""
        projections = self._attention_fp8_linears()
        for name, linear, attention in projections:
            # Both attention implementations use self_attn keys in the checkpoint.
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            attr = name.rsplit(".", 1)[1]
            if attr == "kv_a_proj_with_mqa" and attention.fuse_qkv_a_proj:
                # The runtime fuses Q-A/KV-A, but the checkpoint stores them separately.
                # Construction checks that the fusion boundary aligns with FP8 blocks.
                q_key = key.replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                pairs = []
                for part_key, rows in (
                    (q_key, attention.q_lora_rank),
                    (key, attention.kv_lora_rank + attention.qk_rope_head_dim),
                ):
                    part = _materialize(weights[part_key])
                    pair = _checkpoint_fp8_pair(part_key, part, weights)
                    if pair is None:
                        raise ValueError(
                            f"{part_key}: quant config declares FP8 but tensor is not E4M3"
                        )
                    codes, scales = pair
                    expected = (rows, attention.hidden_size)
                    if tuple(codes.shape) != expected:
                        raise ValueError(
                            f"{part_key}: checkpoint shape {tuple(codes.shape)} != {expected}"
                        )
                    if tuple(scales.shape) != tuple(math.ceil(d / _FP8_BLOCK) for d in expected):
                        raise ValueError(f"{part_key}: scale shape does not cover {expected}")
                    pairs.append(pair)
                linear.load_weights(
                    [
                        {
                            "weight": torch.cat([pair[0] for pair in pairs], dim=0),
                            "weight_scale": torch.cat([pair[1] for pair in pairs], dim=0),
                        }
                    ]
                )
                continue
            source = weights[key]
            shape = (
                tuple(source.shape)
                if isinstance(source, torch.Tensor)
                else tuple(source.get_shape())
            )
            local_shape = (linear.out_features, linear.in_features)
            scale_slice = None
            if isinstance(attention, KimiKDALinearAttention):
                # KDA linears already have rank-local dimensions. Slice the lazy source
                # before materializing it, and apply the same shard to its block scales.
                split_dim = 1 if attr == "o_proj" else 0
                tp_size = attention._kda_tp_size
                tp_rank = attention._kda_tp_rank
                expected_shape = list(local_shape)
                expected_shape[split_dim] *= tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                if tp_size > 1:
                    width = local_shape[split_dim]
                    start, end = tp_rank * width, (tp_rank + 1) * width
                    if start % _FP8_BLOCK or end % _FP8_BLOCK:
                        raise ValueError(f"{key}: TP slice [{start}:{end}] is not block-aligned")
                    indices = [slice(None), slice(None)]
                    indices[split_dim] = slice(start, end)
                    source = source[tuple(indices)]
                    scale_indices = [slice(None), slice(None)]
                    scale_indices[split_dim] = slice(
                        start // _FP8_BLOCK, math.ceil(end / _FP8_BLOCK)
                    )
                    scale_slice = tuple(scale_indices)
            pair = _checkpoint_fp8_pair(key, _materialize(source), weights)
            if pair is None:
                raise ValueError(
                    f"{key}: quant config declares FP8 but checkpoint tensor is not E4M3"
                )
            weight, scale = pair
            full_scale_shape = tuple(math.ceil(dim / _FP8_BLOCK) for dim in shape)
            if tuple(scale.shape) != full_scale_shape:
                raise ValueError(f"{key}: scale shape {tuple(scale.shape)} != {full_scale_shape}")
            if scale_slice is not None:
                scale = scale[scale_slice]
            if isinstance(attention, KimiKDALinearAttention):
                if tuple(scale.shape) != tuple(math.ceil(dim / _FP8_BLOCK) for dim in local_shape):
                    raise ValueError(f"{key}: scale shard does not cover local shape {local_shape}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
            elif attr == "kv_b_proj":
                # KV-B also supplies decode absorption weights; reorder codes and scales
                # together and retain raw FP8 copies for the absorption kernels.
                self._load_mla_fp8_kv_b(attention, weight, scale)
            else:
                # Other MLA projections delegate TP sharding to the common Linear loader.
                expected_shape = list(local_shape)
                if linear.tp_mode is not None:
                    split_dim = 0 if linear.tp_mode == TensorParallelMode.COLUMN else 1
                    expected_shape[split_dim] *= linear.tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
        return len(projections)

    @staticmethod
    def _load_mla_fp8_kv_b(attention: nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Reorder checkpoint KV-B heads and their scale grid together."""
        linear = attention.kv_b_proj
        h, n, v, k = (
            attention.num_heads_tp,
            attention.qk_nope_head_dim,
            attention.v_head_dim,
            attention.kv_lora_rank,
        )
        if any(dim % _FP8_BLOCK for dim in (n, v, k)):
            raise ValueError(
                "Kimi MLA FP8 absorption requires block-aligned head and latent dimensions"
            )
        expected_shape = (h * linear.tp_size * (n + v), k)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"Kimi MLA KV-B shape {tuple(weight.shape)} != {expected_shape}")
        device = linear.weight.device
        local = linear.load_shard(weight, device=device).reshape(h, n + v, k)
        local_scale = linear.load_shard(scale, scale_span=_FP8_BLOCK, device=device).reshape(
            h, (n + v) // _FP8_BLOCK, k // _FP8_BLOCK
        )
        key_weight, value_weight = local.split((n, v), dim=1)
        key_scale, value_scale = local_scale.split((n // _FP8_BLOCK, v // _FP8_BLOCK), dim=1)
        linear.weight.data.copy_(
            torch.cat((key_weight.reshape(h * n, k), value_weight.reshape(h * v, k)))
        )
        linear.weight_scale.data.copy_(
            torch.cat((key_scale.flatten(0, 1), value_scale.flatten(0, 1)))
        )
        attention.k_b_proj_trans.data.copy_(key_weight.transpose(1, 2))
        attention.k_b_proj_trans_scale.data.copy_(key_scale.transpose(1, 2))
        value_weight = _helix_cp_v_b_shard(
            value_weight,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        value_scale = _helix_cp_v_b_shard(
            value_scale,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        # Absorption must keep raw checkpoint codes independent of Linear's
        # backend-specific weight transformations.
        attention.v_b_proj = nn.Parameter(value_weight.clone().contiguous(), requires_grad=False)
        attention.v_b_proj_scale.data.copy_(value_scale)
        for target, codes, scales in (
            (
                attention.k_b_proj_trans_dequant,
                key_weight.transpose(1, 2),
                key_scale.transpose(1, 2),
            ),
            (attention.v_b_proj_dequant, value_weight, value_scale),
        ):
            if target is not None:
                expanded = scales.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                    _FP8_BLOCK, -1
                )
                target.data.copy_((codes.float() * expanded).to(target.dtype))
        linear.process_weights_after_loading()

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs."""
        fp8_prefixes = tuple(name + "." for name, _, _ in self._attention_fp8_linears())
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.startswith(fp8_prefixes)
            and ".linear_attn.qkvg_proj." not in name
            and not name.endswith(_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES)
        }

    def checkpoint_name_plan(
        self, prefix: str
    ) -> Tuple[Dict[str, str], Set[str], List[Tuple[int, KimiK3MoERuntime, str]]]:
        """Return ``(name_map, expected_keys, expert_jobs)``.

        ``name_map`` maps every model parameter name to its checkpoint key
        (for fused ``gate_up_proj`` parameters the mapped key is virtual;
        the two real per-half keys come from ``_gate_up_ckpt_keys``);
        ``expected_keys`` additionally covers the rank-local per-expert MXFP4
        tensors; ``expert_jobs`` lists ``(layer_idx, moe_module, key_base)``
        for backend-owned expert slots. Exposed separately so the weight-name
        mapping can be dry-run without touching any tensor data.
        """
        params = self._trunk_parameters()
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            # ConfigurableMoE's backend owns already-packed runtime weights,
            # generated zero biases, and SiTU constants. They do not have
            # one-to-one checkpoint parameter names.
            if ".routed_experts.backend." in name:
                continue
            if name == "lm_head.weight":
                ckpt_key = prefix + "lm_head.weight"
            else:
                if ".linear_attn." in name:
                    ckpt_key = prefix + name.replace(".linear_attn.", ".self_attn.")
                else:
                    # MLA retains its runtime/mixer hierarchy; the checkpoint
                    # has no intermediate ``mixer`` scope. A K3.1 DSA indexer
                    # lives on the MQA backend but sits directly under
                    # ``self_attn`` in the checkpoint.
                    ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.").replace(
                        ".self_attn.mqa.indexer.", ".self_attn.indexer."
                    )
            name_map[name] = ckpt_key
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Fused [gate | up] MLP layout (dense mlp / shared_experts):
                # the checkpoint stores two separate tensors.
                expected_keys.update(_gate_up_ckpt_keys(ckpt_key))
            else:
                expected_keys.add(ckpt_key)

        # Backend-owned expert slots (per-expert checkpoint tensors; the
        # rank-local id range — an EP slice of whole experts, or ALL experts
        # when the routed MoE is TP-sharded (moe_ep=1 -> ids 0..num_experts)).
        expert_jobs = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, "is_moe", False) or not _has_weights(layer):
                continue
            moe = layer.block_sparse_moe
            base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
            # Required suffixes only: ``optional_suffixes`` are per-checkpoint
            # (an AWQ pre-quant scale), and demanding them here would reject
            # every checkpoint of the same layout that does not carry them.
            # Unmatched expert keys only warn, and this validator already
            # exempts expert keys from that, so the present ones stay quiet.
            kinds = moe.expert_ckpt_spec.kinds
            for expert_idx in moe.ckpt_expert_ids:
                for w in ("w1", "w2", "w3"):
                    for kind in kinds:
                        expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
            expert_jobs.append((layer_idx, moe, base))
        for name, _, _ in self._attention_fp8_linears():
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            expected_keys.update((key, _fp8_block_scale_key(key)))
        for index, layer in enumerate(self.model.layers):
            if getattr(layer, "is_kda", True) or not _has_weights(layer):
                continue
            attention = layer.self_attn.mixer
            if getattr(attention, "fuse_qkv_a_proj", False):
                key = f"{prefix}model.layers.{index}.self_attn.q_a_proj.weight"
                expected_keys.add(key)
                if attention.kv_a_proj_with_mqa.has_fp8_block_scales:
                    expected_keys.add(_fp8_block_scale_key(key))
        return name_map, expected_keys, expert_jobs

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys, expert_jobs = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        num_params += self._load_attention_fp8(weights, prefix)
        self._load_expert_slices(weights, expert_jobs)
        self._finalize_weight_load(num_params, len(expert_jobs))
        device = next(self.parameters()).device
        if device.type == "cuda":
            # Lazy source mappings are load-scoped; finish nonblocking H2D
            # work before the caller can release the weights container.
            torch.cuda.synchronize(device)
        self._permute_routers_to_expert_placement()

    def _permute_routers_to_expert_placement(self) -> None:
        """Reorder the router of every layer with a static expert placement into slot order.

        Runs once, after the checkpoint values are in place; see ``TRTLLM_K3_EXPERT_PLACEMENT``.
        """
        num_permuted = 0
        for layer in self.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if not isinstance(moe, KimiK3MoERuntime) or moe.expert_of_slot is None:
                continue
            permute_router_to_slots(
                moe.gate.weight.data, moe.gate.e_score_correction_bias.data, moe.expert_of_slot
            )
            num_permuted += 1
        if num_permuted:
            logger.info(
                f"Kimi K3: routers of {num_permuted} MoE layers permuted to the expert placement"
            )

    def _validate_checkpoint_keys(
        self, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
    ) -> None:
        """Key-set validation (both directions): every expected key must be
        present; unmatched checkpoint keys (beyond the expected leftovers)
        only warn."""
        ckpt_keys = set(weights.keys())
        relevant_ckpt_keys = {
            k
            for k in ckpt_keys
            if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
        }
        missing = sorted(
            key
            for key in expected_keys - ckpt_keys
            if not (key.endswith(".weight_scale") and key + "_inv" in ckpt_keys)
        )
        if missing:
            raise KeyError(
                f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
                f"keys are missing, e.g. {missing[:10]}"
            )
        unexpected = relevant_ckpt_keys - expected_keys
        # Non-local experts and (in layer-truncated debug mode) extra layers
        # are expected leftovers.
        surprising = sorted(
            k
            for k in unexpected
            if ".block_sparse_moe.experts." not in k and not k.startswith(f"{prefix}model.layers.")
        )
        if surprising:
            logger.warning(
                f"Kimi K3 load_weights: {len(surprising)} unmatched "
                f"checkpoint keys, e.g. {surprising[:10]}"
            )

    def _load_trunk_params(
        self,
        weights: Dict[str, torch.Tensor],
        params: Dict[str, torch.nn.Parameter],
        name_map: Dict[str, str],
    ) -> int:
        """Load every non-expert trunk parameter concurrently (with the
        per-parameter TP-shard / pad / fuse conversions) and return the
        number of parameters loaded."""
        # The checkpoint stores every MLA KV-B head as interleaved [K | V]
        # rows. Runtime keeps one DeepSeek-style [all K | all V] parameter
        # instead, so context can project directly into the FMHA layout and
        # absorbed decode can take zero-copy K/V views.
        mla_mixers = [
            layer.self_attn.mixer
            for layer in self.model.layers
            if not getattr(layer, "is_kda", True) and _has_weights(layer)
        ]
        mla_kv_b_mixers = {id(mixer.kv_b_proj.weight): mixer for mixer in mla_mixers}
        mla_fused_a_mixers = {
            id(mixer.kv_a_proj_with_mqa.weight): mixer
            for mixer in mla_mixers
            if getattr(mixer, "fuse_qkv_a_proj", False)
        }
        mla_head_shard_linears = {}
        for mixer in mla_mixers:
            mla_head_shard_linears[id(mixer.q_b_proj.weight)] = mixer.q_b_proj
            mla_head_shard_linears[id(mixer.o_proj.weight)] = mixer.o_proj
            g_proj = getattr(mixer, "g_proj", None)
            if g_proj is not None:
                mla_head_shard_linears[id(g_proj.weight)] = g_proj

        device = next(self.parameters()).device

        # MLP TP shard index. A dense MLP whose intermediate size does not
        # divide model TP uses a smaller repeated TP subgroup, so its local
        # shard rank is model tp_rank modulo the parameter's shard count.
        # Under helix the modules were sharded against the repurposed
        # mapping; the restored CP original's tp_rank is 0 on every rank.
        model_tp_rank = (
            self._repurposed_tp_mapping.tp_rank
            if self._repurposed_tp_mapping is not None
            else self.model_config.mapping.tp_rank
        )
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                kda_tp_size = layer.linear_attn._kda_tp_size
                kda_tp_rank = layer.linear_attn._kda_tp_rank
                break

        COL = TensorParallelMode.COLUMN  # gate/up column shard (output rows)

        def load_param(name: str, param: torch.nn.Parameter) -> None:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Row-concat the checkpoint's separate gate_proj / up_proj
                # tensors into the fused [gate | up] parameter.
                gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
                inter = param.shape[0] // 2
                # Materialize + FP8-block-dequant each half (this branch reads
                # its own sources, so it needs the same FP8 handling as the
                # single-tensor path below), then column(TP)-shard and
                # row-concat. TP factor from the checkpoint-vs-param shapes; a
                # subgroup smaller than model TP repeats, so the shard index is
                # model tp_rank modulo the half's shard count.
                gate_full = _dequantize_fp8_block_scaled(
                    gate_key, _materialize(weights[gate_key]), weights
                )
                tp = gate_full.shape[0] // inter
                # load_weight_shard returns the whole tensor when tp <= 1, so
                # the rank needs no tp > 1 guard.
                rk = model_tp_rank % tp
                gate = load_weight_shard(gate_full, tp, rk, COL, device=param.device)
                up = load_weight_shard(
                    _dequantize_fp8_block_scaled(up_key, _materialize(weights[up_key]), weights),
                    tp,
                    rk,
                    COL,
                    device=param.device,
                )
                if gate.shape != (inter, param.shape[1]) or up.shape != gate.shape:
                    raise ValueError(
                        f"{name}: checkpoint gate/up shapes "
                        f"{tuple(gate.shape)} / {tuple(up.shape)} do not "
                        f"concat to param shape {tuple(param.shape)}"
                    )
                param.data[:inter].copy_(gate.to(param.dtype))
                param.data[inter:].copy_(up.to(param.dtype))
                return
            src = _materialize(weights[name_map[name]])
            src = _dequantize_fp8_block_scaled(name_map[name], src, weights)
            fused_a = mla_fused_a_mixers.get(id(param))
            if fused_a is not None:
                q_key = name_map[name].replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                q_weight = _dequantize_fp8_block_scaled(
                    q_key, _materialize(weights[q_key]), weights
                )
                if q_weight.shape != (fused_a.q_lora_rank, fused_a.hidden_size) or src.shape != (
                    fused_a.kv_lora_rank + fused_a.qk_rope_head_dim,
                    fused_a.hidden_size,
                ):
                    raise ValueError(
                        f"{name}: checkpoint Q-A/KV-A shapes do not match MLA dimensions"
                    )
                src = torch.cat((q_weight, src), dim=0)
            if name == "lm_head.weight":
                # LMHead is vocab-sharded (TP column) + gathered; its
                # load_weights shards the full checkpoint tensor.
                self.lm_head.load_weights(weights=[{"weight": src}])
                return
            mla_mixer = mla_kv_b_mixers.get(id(param))
            if mla_mixer is not None:
                h = mla_mixer.num_heads_tp
                n = mla_mixer.qk_nope_head_dim
                v = mla_mixer.v_head_dim
                kv = mla_mixer.kv_lora_rank
                local = mla_mixer.kv_b_proj.load_shard(src, device=param.device).view(h, n + v, kv)
                k_weight, v_weight = local.split([n, v], dim=1)
                param.data.copy_(
                    torch.cat(
                        [
                            k_weight.reshape(h * n, kv),
                            v_weight.reshape(h * v, kv),
                        ],
                        dim=0,
                    ).to(param.dtype)
                )
                mla_mixer.k_b_proj_trans.data.copy_(k_weight.transpose(1, 2))
                # Share the loaded BF16 V rows with context GEMM instead of keeping a
                # separate copy. Wrap the view to retain v_b_proj's parameter registration.
                v_weight = _helix_cp_v_b_shard(
                    param[h * n :].view(h, v, kv),
                    num_heads_tp_cp=mla_mixer.num_heads_tp_cp,
                    cp_rank=mla_mixer.mapping.cp_rank,
                )
                mla_mixer.v_b_proj = nn.Parameter(v_weight, requires_grad=False)
                return
            if name.endswith(".A_log") and src.numel() != param.numel():
                # The checkpoint pads A_log from [num_heads] to [head_dim]
                # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
                # head-shard TP the param holds this rank's head range
                # instead of the full [num_heads].
                assert src.numel() > param.numel(), (name, src.shape)
                if kda_tp_size > 1:
                    lo = kda_tp_rank * param.numel()
                    src = src[lo : lo + param.numel()]
                else:
                    tail = src[param.numel() :]
                    if tail.abs().max().item() != 0.0:
                        raise ValueError(
                            f"{name}: expected zero padding beyond "
                            f"{param.numel()} entries, got nonzero tail"
                        )
                    src = src[: param.numel()]
            # KDA ``linear_attn`` (head-major) and shared-expert / dense-MLP
            # ``down_proj`` shards are shape-derived; replicated tensors and
            # every other name pass through unchanged. MLA head-shards and the
            # shape-match copy are handled by the block below.
            src = _shard_head_major_param(
                name,
                src,
                param,
                kda_tp_size=kda_tp_size,
                kda_tp_rank=kda_tp_rank,
                model_tp_rank=model_tp_rank,
            )

            if src.shape != param.shape:
                # MLA q_b/g/o head-shard: delegate to the same Linear modules
                # that own their COLUMN/ROW sharding policy (#17684 removed the
                # 96->128 head padding). KV-B is handled above; KDA and
                # shared-expert/MLP shards were resolved in the pre-block above.
                mla_sharded_linear = mla_head_shard_linears.get(id(param))
                if mla_sharded_linear is not None:
                    shard = mla_sharded_linear.load_shard(src, device=param.device)
                    if shard.shape != param.shape:
                        raise ValueError(
                            f"{name}: MLA shard shape {tuple(shard.shape)} does "
                            f"not match param shape {tuple(param.shape)}"
                        )
                    param.data.copy_(shard.to(param.dtype))
                    return
                raise ValueError(
                    f"{name}: shard/pad result {tuple(src.shape)} != param "
                    f"shape {tuple(param.shape)}"
                )
            param.data.copy_(src.to(param.dtype))

        param_jobs = [(name, params[name]) for name in name_map]
        run_concurrently(load_param, param_jobs, num_workers=8)

        logger.info(
            f"Kimi K3: loaded {len(mla_mixers)} MLA KV-B projections in grouped runtime layout"
        )
        return len(param_jobs)

    def _load_expert_slices(
        self,
        weights: Dict[str, torch.Tensor],
        expert_jobs: List[Tuple[int, KimiK3MoERuntime, str]],
    ) -> None:
        """Load the rank-local MXFP4 expert slices of every MoE layer into
        the backend expert slots, then verify every slot was filled."""
        device = next(self.parameters()).device
        # Startup accounting, logged once below: checkpoint reads (bytes and
        # thread-summed time), the backend's per-expert loader (reads, H2D
        # copy, shuffle, scale interleave; thread-summed) and layer
        # finalization. TLLM_K3_EXPERT_LOAD_THREADS sets the loader threads.
        load_threads = int(os.environ.get("TLLM_K3_EXPERT_LOAD_THREADS", "4"))
        load_stats = {
            "read_s": 0.0,
            "read_bytes": 0,
            "expert_s": 0.0,
            "finalize_s": 0.0,
            "experts": 0,
        }
        stats_lock = threading.Lock()
        load_start = time.perf_counter()

        def timed_read(get):
            def read(key):
                start = time.perf_counter()
                tensor = get(key)
                elapsed = time.perf_counter() - start
                nbytes = 0 if tensor is None else tensor.numel() * tensor.element_size()
                with stats_lock:
                    load_stats["read_s"] += elapsed
                    load_stats["read_bytes"] += nbytes
                return tensor

            return read

        # Layouts whose per-expert loader only stages its input must have the
        # staging containers created before any thread runs, and must be
        # finalized once the layer's last slot lands: nothing else on this path
        # calls process_weights_after_loading. Which thread completes a layer is
        # a race, so the completion test and the "already finalized" bookkeeping
        # are one critical section.
        finalize_lock = threading.Lock()
        finalized_backends = set()
        prepared_backends = set()

        def ensure_prepared(moe: KimiK3MoERuntime):
            """Prepare a layer's streaming state on its FIRST expert, not up front.

            Preparing every layer before the load starts is what OOM-ed the
            MegaMoE CuteDSL backend: it keeps its raw NVFP4 source params as
            0-element placeholders and rematerializes them at full shape here,
            so preparing all 92 layers held 92 layers of raw weights at once
            instead of the handful actually being filled. Cutlass did not care
            because its parameters are allocated either way.

            Bounded lazily instead. The paired shrink already happens per layer
            in process_weights_after_loading, so the live set is whatever is
            genuinely in flight -- measured at 1 layer per shard file for this
            checkpoint (its rank-local experts are 1:1 with files), so ~4 with
            4 loader threads. Correctness does not depend on that layout
            though; a checkpoint that split a layer across files would only
            raise the peak, not break this.

            The membership add happens AFTER preparing, so the lock-free fast
            path can only ever be stale in the safe direction.
            """
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            if id(backend) in prepared_backends:
                return
            with finalize_lock:
                if id(backend) in prepared_backends:
                    return
                backend.quant_method.prepare_streaming_expert_load(backend)
                prepared_backends.add(id(backend))

        def maybe_finalize_layer(moe: KimiK3MoERuntime):
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            with finalize_lock:
                loaded = len(getattr(backend, spec.loaded_slots_attr, ()))
                if loaded != backend.expert_size_per_partition:
                    return
                if id(backend) in finalized_backends:
                    return
                finalized_backends.add(id(backend))
            # Computes the alphas, interleaves the w2 scales, and (MegaMoE)
            # packs into the mega buffers and shrinks the raw source params
            # back to placeholders. Paired with ensure_prepared above, that
            # pairing is what bounds the per-layer footprint. The per-expert
            # drain inside load_streaming_nvfp4_expert separately bounds the
            # Cutlass w3_w1 staging, which is per expert rather than per layer.
            backend.process_weights_after_loading()

        def load_expert(
            moe: KimiK3MoERuntime,
            base: str,
            local_slot_id: int,
            expert_idx: int,
            get_tensor,
            get_optional_tensor,
        ):
            """Load checkpoint expert ``expert_idx`` into local slot ``local_slot_id``."""
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            start = time.perf_counter()
            moe.expert_ckpt_spec.loader(
                backend,
                base,
                expert_idx,
                moe.local_expert_ids[local_slot_id],
                local_slot_id,
                timed_read(get_tensor),
                timed_read(get_optional_tensor),
            )
            loaded = time.perf_counter()
            maybe_finalize_layer(moe)
            with stats_lock:
                load_stats["expert_s"] += loaded - start
                load_stats["finalize_s"] += time.perf_counter() - loaded
                load_stats["experts"] += 1

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                load_expert(
                    moe,
                    base,
                    local_slot_id,
                    expert_idx,
                    lambda key: _materialize(weights[key]),
                    lambda key: _materialize(weights[key]) if key in weights else None,
                )

        # ---- backend expert slots: file-grouped streaming ----
        # The shared lazy ``weights`` dict keeps every shard mmapped for the
        # whole load, so pages it touches cannot be dropped until the load
        # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
        # DISTINCT pages per rank — with 4 ranks/node that overruns the job
        # cgroup and OOM-kills the step (observed repeatedly on GB300
        # trays). Instead, group the rank-local expert tensors by shard file
        # and stream each file through a short-lived handle:
        # open -> copy -> close (unmap) -> fadvise(DONTNEED).
        # The lazy loader records the directory it opened; prefer it over
        # ``_name_or_path``, which transformers no longer populates (it is
        # empty on transformers 5.x, which silently sent the whole load down
        # the fallback below and OOM-killed the step).
        ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
            self.model_config.pretrained_config, "_name_or_path", None
        )
        index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
        checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")

        def checkpoint_key(key: str) -> str:
            return f"{checkpoint_prefix}{key}"

        if expert_jobs and ckpt_dir and os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            per_file: Dict[str, list] = {}
            per_file_keys: Dict[str, list] = {}
            split_file_jobs = []
            for layer_idx, moe, base in expert_jobs:
                del layer_idx
                spec = moe.expert_ckpt_spec
                for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                    keys = [
                        f"{base}.{expert_idx}.{w}.{kind}"
                        for w in ("w1", "w2", "w3")
                        for kind in spec.kinds
                    ]
                    # An optional tensor this checkpoint does store has to be
                    # counted here: it decides which files the job opens, and
                    # ``get_optional_tensor`` below can only read a file that
                    # ``files`` named.
                    keys += [
                        key
                        for key in (
                            f"{base}.{expert_idx}.{suffix}" for suffix in spec.optional_suffixes
                        )
                        if checkpoint_key(key) in weight_map
                    ]
                    files = {weight_map[checkpoint_key(key)] for key in keys}
                    job = (moe, base, local_slot_id, expert_idx)
                    if len(files) == 1:
                        file_name = files.pop()
                        per_file.setdefault(file_name, []).append(job)
                        per_file_keys.setdefault(file_name, []).extend(keys)
                    else:
                        split_file_jobs.append((job, files))

            def drop_file_pages(file_name: str):
                path = os.path.join(ckpt_dir, file_name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass

            # Kimi K3.1: read this rank's byte ranges of a shard
            # (merged across gaps under 1 MiB) with large preadv calls into a
            # per-thread pinned host buffer and serve its expert tensors as
            # views of it. Tensors are stored sorted by name, so experts
            # 0..N/ep-1 (rank 0) are spread over the whole shard (experts 0, 1,
            # 10-19, 100-...) while every other rank's slice is one compact
            # region. Copying them to the GPU straight out of the safetensors
            # mapping faults the file in page by page, which is random access
            # for rank 0: its loads took 3x as long as the other ranks' on a
            # GB300 TEP8 worker. Any problem (an unknown dtype, a short read)
            # falls back to the mapping.
            pread_experts = (
                getattr(_get_text_config(self.model_config.pretrained_config), "dsa_config", None)
                is not None
            )
            pread_buffers = threading.local()

            def pread_buffer(nbytes: int) -> torch.Tensor:
                buf = getattr(pread_buffers, "buf", None)
                if buf is None or buf.numel() < nbytes:
                    pread_buffers.buf = None
                    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=device.type == "cuda")
                    pread_buffers.buf = buf
                return buf

            def read_file_tensors(file_name: str) -> Optional[Dict[str, torch.Tensor]]:
                return read_safetensors_slices(
                    os.path.join(ckpt_dir, file_name),
                    [checkpoint_key(key) for key in per_file_keys[file_name]],
                    pread_buffer,
                )

            def load_expert_file(file_name: str, jobs: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                if pread_experts:
                    read_start = time.perf_counter()
                    try:
                        tensors = read_file_tensors(file_name)
                    except (OSError, ValueError, KeyError, RuntimeError):
                        tensors = None
                    if tensors is not None:
                        # Bytes are counted when the loaders take the views.
                        with stats_lock:
                            load_stats["read_s"] += time.perf_counter() - read_start
                        for moe, base, local_slot_id, expert_idx in jobs:
                            load_expert(
                                moe,
                                base,
                                local_slot_id,
                                expert_idx,
                                lambda key: tensors[checkpoint_key(key)],
                                lambda key: tensors.get(checkpoint_key(key)),
                            )
                        if device.type == "cuda":
                            # Copies out of the buffer may still be in flight;
                            # the next shard overwrites it.
                            torch.cuda.current_stream(device).synchronize()
                        drop_file_pages(file_name)
                        return
                path = os.path.join(ckpt_dir, file_name)
                with safe_open(path, framework="pt", device="cpu") as fh:
                    # A job lands here only when every key it needs -- optional
                    # ones included, see the grouping above -- is in this file,
                    # so presence in this file is the whole question.
                    file_keys = set(fh.keys())
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(
                            moe,
                            base,
                            local_slot_id,
                            expert_idx,
                            lambda key: fh.get_tensor(checkpoint_key(key)),
                            lambda key: (
                                fh.get_tensor(checkpoint_key(key))
                                if checkpoint_key(key) in file_keys
                                else None
                            ),
                        )
                # Handle closed -> pages unmapped -> the drop takes effect.
                drop_file_pages(file_name)

            def load_split_file_expert(job, files):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                with ExitStack() as stack:
                    handles = {
                        file_name: stack.enter_context(
                            safe_open(
                                os.path.join(ckpt_dir, file_name), framework="pt", device="cpu"
                            )
                        )
                        for file_name in files
                    }

                    def get_tensor(key):
                        source_key = checkpoint_key(key)
                        return handles[weight_map[source_key]].get_tensor(source_key)

                    def get_optional_tensor(key):
                        source_key = checkpoint_key(key)
                        # Absent from the index means the checkpoint does not
                        # store it; present means the grouping above put its
                        # file in ``handles``.
                        if source_key not in weight_map:
                            return None
                        return handles[weight_map[source_key]].get_tensor(source_key)

                    load_expert(*job, get_tensor, get_optional_tensor)
                for file_name in files:
                    drop_file_pages(file_name)

            run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=load_threads)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=load_threads)
        else:
            # Falling back is a silent loss of the whole point of the block
            # above: the shared lazy dict keeps every shard mapped, which is
            # the OOM this streaming path exists to avoid. Say so.
            if expert_jobs:
                logger.warning(
                    f"Kimi K3: no safetensors index at '{index_path}', so routed "
                    "experts are loaded from the shared lazy weight dict instead "
                    "of being streamed per shard file. Every shard stays mapped "
                    "for the whole load, which OOM-kills the step at DEP8 scale."
                )
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=load_threads)

        for _, moe, _ in expert_jobs:
            spec = moe.expert_ckpt_spec
            backend = moe.routed_experts.backend
            loaded_slots = getattr(backend, spec.loaded_slots_attr, set())
            expected_slots = set(range(backend.expert_size_per_partition))
            if loaded_slots != expected_slots:
                missing_slots = sorted(expected_slots - loaded_slots)
                raise RuntimeError(
                    "Kimi K3 streaming expert loading did not fill all backend "
                    f"slots; missing {missing_slots[:10]}."
                )
            if (
                spec.needs_layer_finalize
                and expected_slots
                and id(backend) not in finalized_backends
            ):
                # Unreachable via load_expert (the last slot finalizes), so
                # reaching it means the two bookkeeping paths disagree. Guarded
                # on expected_slots because a layer that owns no local slots is
                # never prepared and so is legitimately never finalized.
                raise RuntimeError(
                    "Kimi K3 streaming expert loading filled every slot but "
                    "never finalized the layer."
                )
            backend._weights_transformed = False
        wall = time.perf_counter() - load_start
        logger.info(
            f"Kimi K3: expert slices: {load_stats['experts']} experts, "
            f"{load_stats['read_bytes'] / 2**30:.1f} GiB read in {load_stats['read_s']:.1f}s, "
            f"per-expert load {load_stats['expert_s']:.1f}s (incl. reads), "
            f"finalize {load_stats['finalize_s']:.1f}s (thread-summed), wall {wall:.1f}s, "
            f"{load_threads} threads"
        )

    def _finalize_weight_load(self, num_params: int, num_moe_layers: int) -> None:
        """Finalize checkpoint-loaded attention and optional BF16 MLP conversion."""
        for layer in self.model.layers:
            if not _has_weights(layer):
                continue
            if getattr(layer, "is_kda", False):
                attention = layer.linear_attn
                attention.finalize_decode_weights()
                attention._build_mtp_conv_weights()
            else:
                attention = layer.self_attn.mixer
            for name, linear in attention.named_children():
                if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
                    _swap_linear_to_fp8_weight_read(attention, name, (TrtllmLinear,))
            fused = getattr(attention, "qkvg_proj", None)
            if isinstance(fused, _Fp8BlockScaleWeightReadLinear):
                offset = 0
                for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                    projection = getattr(attention, name)
                    projection._fused_projection = (weakref.ref(fused), offset)
                    offset += projection.out_features
                    # Repoint to the fused rows now rather than in post_load:
                    # deferring kept every layer's standalone codes alongside
                    # its fused copy, ~22.6 GiB at peak across K3's 69 KDA
                    # layers, which overflows a DEP8 context rank.
                    projection.transform_weights()
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
        )
        if self._fp8_weight_read_moe_mlp and is_sm_100f():
            converted = _convert_moe_mlps_to_fp8_weight_read(
                self.model, include_fused_gate_up=self.model_config.mapping.enable_attention_dp
            )
            logger.info(f"Kimi K3: quantized {converted} BF16 shared/latent MLP projections to FP8")
