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
the routed latent partial is reduced before its norm/up projection.
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
import weakref
from contextlib import ExitStack, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
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

from tensorrt_llm._torch import kimi_k3_cache_policy as _kimi_k3_cache_policy
from tensorrt_llm._torch.attention.attention import _helix_cp_pad
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    allgather,
)
from tensorrt_llm._torch.distributed.symm_mem_allgather import SymmetricMemoryAllGather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, run_concurrently
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (
    Fp8BlockScaleLowMGemmRange,
    TensorParallelMode,
    load_weight_shard,
)
from tensorrt_llm._torch.modules.linear import Linear as TrtllmLinear
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.situ import SituAndMul
from tensorrt_llm._torch.moe.fused_moe import (
    ConfigurableMoE,
    SiTuActivation,
    TRTLLMGenFusedMoE,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from tensorrt_llm._torch.speculative.spec_head_kernels import (
    embedding_gather,
    spec_head_fusion_enabled,
)
from tensorrt_llm._torch.utils import AuxStreamType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from .comm.lamport import (
    LamportPending,
    LamportResult,
    finish_lamport_push,
    lamport_push_target,
    lamport_push_tensor,
)
from .comm.lamport import mnnvl_workspace_mpi_comm as _k3_mnnvl_mpi_comm
from .features import active as _features
from .kda import KimiKDALinearAttention
from .kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from .moe.kimi_k3_expert_placement import get_routing_recorder
from .spec.shell import install_k3_dspark

_F = _features()

_K3_FUSE_MOE_RESIDUAL = _F.fuse_moe_residual

# FP8 weight-read projections with few rows (per-shape, per-architecture limit, see
# k3_fp8w_lowm_gemm.max_m_for) run the CuTe low-M GEMM: on SM100/SM103 instead of DeepGEMM
# swap-AB, on SM107 instead of the CuTe MXFP8 GEMM (with a DeepGEMM-packed copy of the weight
# scale, built at post-load, see _Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm).
_K3_FP8W_LOWM_GEMM = _F.fp8w_lowm_gemm
# With the low-M GEMM on, also serve projections marked ``fp8w_lowm_partial_n_tile`` (the MLA
# fused q_a/kv_a projection, 2112 rows) although N is not a multiple of 128.
_K3_FP8W_LOWM_FUSED_A = _F.fp8w_lowm_fused_a

# Helix MLA generation applies the V up-projection of all heads before the CP
# all-to-all (see ``MLA._helix_project_v_partials``), which needs every head's
# BF16 V weight on each rank.
_K3_HELIX_V_UP_BEFORE_A2A = _F.helix_v_up_before_a2a

# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): "1" replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing);
# "gemv" also computes the logits of M <= 32 tokens with a 40-CTA GEMV that runs beside the
# latent down-projection, which then quantizes the routed input right away.
_K3_ROUTE_TAIL = _F.route_tail
if _K3_ROUTE_TAIL not in ("0", "1", "gemv"):
    raise ValueError(f"TRTLLM_K3_ROUTE_TAIL must be 0, 1 or gemv, got {_K3_ROUTE_TAIL!r}")
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = 2048
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
_K3_ROUTE_TAIL_ORDER = _F.route_tail_order
if _K3_ROUTE_TAIL_ORDER not in ("0", "router", "fork"):
    raise ValueError(
        f"TRTLLM_K3_ROUTE_TAIL_ORDER must be 0, router or fork, got {_K3_ROUTE_TAIL_ORDER!r}"
    )
_K3_ROUTE_TAIL_ORDER_MIN_ROWS = 80

# Fused latent projection (kernels/k3_latent_fp8_gemm.py), for token counts
# in [TRTLLM_K3_LD_FUSED_MIN_M, TRTLLM_K3_LD_FUSED_MAX_M]:
#   TRTLLM_K3_LD_FUSED: latent_down + the routed experts' NVFP4 input quantization in one chain
#     (trtllm::k3_latent_down_fp4), bit-exact with fp8_swap_ab_gemm followed by fp4_quantize.
# "1" fuses only where the unfused projection would run fp8_swap_ab_gemm; "2" also replaces the
# low-M kernel (TRTLLM_K3_FP8W_LOWM_GEMM), whose 1x32 activation scales then become the 1x128
# scales of the larger-M path. On SM107 the op runs the low-M kernel's split-K GEMM with the
# fused epilogue up to k3_latent_fp8_gemm.lowm_route_max_m tokens (in either mode; bit-exact
# with that GEMM at the same tactic followed by fp4_quantize) and the persistent kernel above;
# both read a DeepGEMM-packed copy of the weight scale built at post-load
# (_Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm).
_K3_LD_FUSED = _F.ld_fused
if _K3_LD_FUSED not in ("0", "1", "2"):
    raise ValueError(f"TRTLLM_K3_LD_FUSED must be 0, 1 or 2, got {_K3_LD_FUSED!r}")
_K3_LD_FUSED_MIN_M = 1
_K3_LD_FUSED_MAX_M = 4096
# Up to this M the low-M kernel's cluster split-K beats the fused chains (which use one CTA per
# 128-row x 128..256-column tile) even in mode "2".
_K3_LOWM_OVER_FUSED_MAX_M = 128

# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = 0
# TRTLLM_K3_ROUTER_FIRST=1: for up to TRTLLM_K3_ROUTER_FIRST_MAX_M tokens, the router GEMM is issued
# (captured) on the main stream before latent_down is issued on the latent stream. Both still
# depend only on the MoE input (the latent fork event is recorded before the router), so the graph
# has the same edges; only the node creation order changes, which decides which of the two the
# graph launches first once the MoE input's producer releases its dependents (PDL). At decode
# batch sizes latent_down's low-M GEMM is 112 CTAs x ~225 KB of shared memory, and with up to 16
# tokens the router GEMM (dsv3 router MMA kernel) is 112 CTAs whose registers fill most of an SM,
# so no SM holds one of each: launched second, the router gets only the SMs latent_down leaves
# and runs in several waves, and the routing (which gates the routed FC1 together with
# latent_down) ends late. Launched first, the router takes its SMs and latent_down starts on the
# rest. nsys node tracing turns latent_down's cross-stream programmatic edge into a full edge (it
# then waits for the MoE input's producer to complete), so such traces show the router first in
# either order.
_K3_ROUTER_FIRST = _F.router_first
_K3_ROUTER_FIRST_MAX_M = 64
# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = 0
# SM-count target of the shared expert's cuBLAS GEMMs (0: off). FC12's persistent grid needs
# whole SMs, so shared-expert GEMMs still resident when routing finishes delay its launch; with
# TRTLLM_K3_SHARED_SM=N they run on N SMs (torch SM carveout, captured into CUDA graphs) and
# TRTLLM_CUTEDSL_SM_RESERVE=N / TRTLLM_K3_FC12_SM_RESERVE=N keep those SMs free of FC12.
_K3_SHARED_SM = _F.shared_sm


@contextmanager
def _k3_shared_sm_scope():
    """Run the enclosed cuBLAS GEMMs on ``TRTLLM_K3_SHARED_SM`` SMs (no-op when 0)."""
    if _K3_SHARED_SM <= 0:
        yield
        return
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    prev = torch._C._get_sm_carveout_experimental()
    torch._C._set_sm_carveout_experimental(max(0, num_sms - _K3_SHARED_SM))
    try:
        yield
    finally:
        torch._C._set_sm_carveout_experimental(prev)


# TRTLLM_K3_SHARED_TAIL_PRIO=<p> (e.g. -1; unset: off): for TRTLLM_K3_SHARED_TAIL_MIN_M ..
# TRTLLM_K3_SHARED_TAIL_MAX_M tokens, where the routed latent all-reduce runs on its own workspace
# (TRTLLM_K3_LATENT_AR_OWN_WS), the shared expert's activation, down projection and all-reduce run
# on a second stream of CUDA priority <p>, forked from the shared-expert stream after gate_up, and
# the main stream joins that stream instead. At decode batch sizes the routed FC1 / FC2 kernels
# are persistent grids whose pending CTAs (the PDL-launched FC2 waits for SMs while FC1 runs)
# would otherwise be served before the shared expert's small kernels launched after them; with
# the higher priority those kernels take the SMs that FC1 / FC2 leave and the first ones FC1's
# CTAs free. Same kernels and inputs: bit-exact.
_K3_SHARED_TAIL_PRIO = ""
_K3_SHARED_TAIL_MIN_M = 1
_K3_SHARED_TAIL_MAX_M = 64
# The shared-expert tail streams, by (device index, priority); every MoE layer uses the same one.
_K3_SHARED_TAIL_STREAMS: Dict[Tuple[int, int], torch.cuda.Stream] = {}


def _k3_shared_tail_stream() -> Optional[torch.cuda.Stream]:
    """The ``TRTLLM_K3_SHARED_TAIL_PRIO`` stream of the current device, or ``None`` if off."""
    if not _K3_SHARED_TAIL_PRIO or not torch.cuda.is_available():
        return None
    key = (torch.cuda.current_device(), int(_K3_SHARED_TAIL_PRIO))
    stream = _K3_SHARED_TAIL_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(priority=key[1])
        _K3_SHARED_TAIL_STREAMS[key] = stream
    return stream


# TRTLLM_K3_FC12_PREZERO=1 (CUTEDSL_FC12 routed experts, in forwards that fork the latent chain,
# i.e. multi-stream decode steps): the routed output and the fused FC12 kernel's three atomic
# counters are allocated where the latent chain forks and zeroed by one fill kernel on a stream of
# their own, which the main stream joins right before FC12. FC12's op then issues neither its
# counter fill (a 1-CTA kernel) nor its output memset (a memset node) between the routing and the
# kernel: FC12 directly follows the routing kernel, and its programmatic dependent launch overlaps
# FC12's prologue with the end of the routing tail. Same values in the same kernels: bit-exact.
# FC12's persistent grid then also takes the SMs earlier, before shared-expert kernels issued
# around the same time on their stream (see TRTLLM_K3_ROUTE_TAIL_ORDER); TRTLLM_K3_FC12_SM_RESERVE
# (custom_ops/cute_dsl_custom_ops.py) leaves SMs for them.
_K3_FC12_PREZERO = _F.fc12_prezero
# The TRTLLM_K3_FC12_PREZERO zeroing streams, by device index; every MoE layer uses the same one.
_K3_FC12_PREZERO_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _k3_fc12_prezero_stream() -> torch.cuda.Stream:
    """The ``TRTLLM_K3_FC12_PREZERO`` zeroing stream of the current device."""
    device = torch.cuda.current_device()
    stream = _K3_FC12_PREZERO_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream()
        _K3_FC12_PREZERO_STREAMS[device] = stream
    return stream


# Small-batch MoE front end (b1-b4 decode), bit-identical to the path it replaces:
#   TRTLLM_K3_ROUTER_EXACT=1: router logits of M <= 16 tokens from k3_router_exact (bit-identical
#     to dsv3_router_gemm_op; its weight is copied into shared memory before griddepcontrol.wait).
#     =kda: only in the MoE blocks of KDA layers. The faster router also starts the route tail
#     earlier, inside the burst of latent_down and the shared gate_up; after an MLA attention block
#     the route tail then stretches by ~3 us (b2 gen-only, GB300) and the routing chain gets longer
#     than with dsv3_router_gemm_op, while after a KDA attention block the front gets shorter.
_K3_ROUTER_EXACT_MODE = _F.router_exact
if _K3_ROUTER_EXACT_MODE not in ("0", "1", "kda"):
    raise ValueError(f"TRTLLM_K3_ROUTER_EXACT must be 0, 1 or kda, got {_K3_ROUTER_EXACT_MODE!r}")
# TRTLLM_K3_LATENT_AR_OWN_WS=1: for up to TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed latent
# all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
# one it shares with the shared expert's all-reduce. It then no longer has to follow the shared-expert
# chain: the main stream issues it and latent_up right after the routed FC2 and joins the shared
# stream after latent_up. Applies where latent_up does not add the shared output itself (no fused
# latent_up + residual variant); the output is bit-identical.
_K3_LATENT_AR_OWN_WS = _F.latent_ar_own_ws
_K3_LATENT_AR_OWN_WS_MAX_M = _F.latent_ar_own_ws_max_m
# TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES=<bytes> (with TRTLLM_K3_LATENT_AR_OWN_WS; 0: the global
# MNNVL one-shot threshold, TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES on this tree): the one-shot limit
# (rows x latent x ranks x 2 B) of the routed latent all-reduce on its own workspace, which is
# sized for it. The other MNNVL all-reduces keep the global limit; e.g. 6291456 runs the 96-row
# latent all-reduce of 8 ranks (5.5 MB) one-shot.
_K3_LATENT_AR_ONESHOT_MAX_BYTES = _F.latent_ar_oneshot_max_bytes
# TRTLLM_K3_LATENT_AR_CUTE (with TRTLLM_K3_LATENT_AR_OWN_WS): the one-shot sizes of that all-reduce
# run k3_latent_ar_norm, bit-identical to the C++ one-shot kernel but triggering its programmatic
# dependents (latent_up) right after its griddepcontrol.wait instead of after the cross-rank
# reduction ("1"), or after the reduction like the C++ kernel ("late").
_K3_LATENT_AR_CUTE = _F.latent_ar_cute
if _K3_LATENT_AR_CUTE not in ("0", "1", "late"):
    raise ValueError(f"TRTLLM_K3_LATENT_AR_CUTE must be 0, 1 or late, got {_K3_LATENT_AR_CUTE!r}")
# TRTLLM_K3_LU_MID_MXFP8=1 (with TRTLLM_K3_LATENT_AR_CUTE and TRTLLM_K3_LATENT_AR_ROW): where
# latent_up runs the mid-M path of trtllm::k3_fp8w_lowm_gemm (above the low-M cut-over, e.g. the
# 96-row helix8 b16 mtp5 verify step), the row kernel also stores the MXFP8 form of the normalized
# latent in that GEMM's scale layout and latent_up runs the mid-M GEMM on it
# (k3_fp8w_lowm_gemm.run_mid_prequantized) instead of quantizing the BF16 rows in a kernel of its
# own: bitwise the same projection, one kernel less after the all-reduce.
_K3_LU_MID_MXFP8 = _F.lu_mid_mxfp8
# TRTLLM_K3_LATENT_AR_ROW=1 (with TRTLLM_K3_LATENT_AR_CUTE): the one-shot sizes of the latent
# all-reduce run k3_latent_ar_norm's row kernel (one 512-thread CTA per row instead of a cluster of
# four; the rank's own slot is summed from registers instead of being polled back from the
# multicast loopback; slots that arrived are not reloaded): the same result, bit for bit.
_K3_LATENT_AR_ROW = _F.latent_ar_row
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
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    # The shared workspace's communicator (split by the all-reduce group) is the same group.
    comm = _k3_mnnvl_mpi_comm(get_or_scale_allreduce_mnnvl_workspace(mapping, dtype))
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

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

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
            hidden_2d.type(torch.float32),
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

    ``KimiK3MoERuntime`` stores the ``(topk_ids, topk_weights)`` of a router that also routes
    (the router GEMV + routing tail, the ULL front) in ``precomputed`` right before calling the
    routed experts; the next ``apply`` returns and clears them instead of routing ``logits``
    again. Alternatively it stores a ``deferred`` callable that the next ``apply`` runs (once)
    on ``logits``.
    ``observer``, when set, is called with every top-k id tensor ``apply`` returns.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.deferred: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self.observer: Optional[Callable[[torch.Tensor], None]] = None

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
        if self.observer is not None and not torch.compiler.is_compiling():
            self.observer(result[0])
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
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
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

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"
"""Which fused attention-residual kernel topology to use.

Accepted values:

  ``per_token``   one CTA (N<=4) or one 8-CTA cluster (N>=5) per token; the
                  persistent kernel is never used. This is the default, i.e.
                  the feature is off and behaviour is unchanged.
  ``persistent``  one CTA per SM looping over tokens, used at every shape the
                  persistent kernel implements (H == 7168 and 2 <= N <= 9).
  ``split``       persistent above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``
                  tokens, per-token at or below it.
  ``1``           the only accepted on-value; resolves to ``split``.

Off by default because the persistent kernel measures as no-regression on
NVFP4 but as a 2.98x context-phase regression on W4A8_MXFP4_MXFP8, through a
downstream interaction with the TRTLLM-gen MoE rather than through this kernel.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

_ATTN_RES_TOPOLOGIES = ("per_token", "persistent", "split")

# Turning the port on should not require knowing which of the three topologies
# is the right one: "1" selects the measured policy (``split``), so enabling the
# feature and choosing the policy are one action through one variable.
_ATTN_RES_TOPOLOGY_ON = "1"


def _read_attn_res_topology() -> str:
    """Default ``per_token``: the persistent kernel is opt-in.

    ``1`` is the only accepted on-value and resolves to ``split``. The named
    topologies stay available for measurement: ``persistent`` uses the
    persistent kernel at every shape it implements, ``per_token`` at none.
    """
    raw = os.environ.get(KIMI_K3_ATTN_RES_TOPOLOGY_ENV, "per_token")
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


# TRTLLM_K3_ATTN_RES_PDL=1: up to TRTLLM_K3_ATTN_RES_PDL_MAX_M tokens, the fused attn_res(+add)
# +RMSNorm runs as trtllm::k3_attn_res_(add_)rmsnorm (kernels/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements.
# TRTLLM_K3_ATTN_RES_PDL=2 also leaves the MoE residual update residual + bf16(routed + shared)
# of the BF16-add path (kimi_k3_moe_residual) to the next layer's pre-attention mix, which
# computes it bit-identically (trtllm::k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.
_K3_ATTN_RES_PDL_MODE = _F.attn_res_pdl
if _K3_ATTN_RES_PDL_MODE not in ("0", "1", "2"):
    raise ValueError(f"TRTLLM_K3_ATTN_RES_PDL must be 0, 1 or 2, got {_K3_ATTN_RES_PDL_MODE!r}")
_K3_ATTN_RES_PDL = _K3_ATTN_RES_PDL_MODE != "0"
_K3_ATTN_RES_PDL_MAX_M = _F.attn_res_pdl_max_m
# TRTLLM_K3_ATTN_RES_MXFP8=1: the pre-attention mix that folds in the previous MoE residual
# (trtllm::k3_attn_res_moe_add_rmsnorm_mx) also stores the MXFP8 form of its output, and the
# attention's first projection (KDA fused QKVG, MLA fused q_a / kv_a) runs the low-M GEMM on it
# (k3_fp8w_lowm_mxfp8) instead of quantizing the BF16 rows itself: bitwise the same projection,
# generation-only batches where the projection runs the low-M kernel.
_K3_ATTN_RES_MXFP8 = _F.attn_res_mxfp8
# TRTLLM_K3_AR_PUSH_AMID=1 (with TRTLLM_K3_AR_PUSH=1): where the mix after attention (A_mid) adds
# the attention output to the running prefix sum, the attention block finishes its pushed o_proj
# all-reduce in place (two-shot transport, no gather) and A_mid reads the result straight from the
# MNNVL Lamport workspace (trtllm::k3_attn_res_add_rmsnorm_lamport): the same bits, one kernel
# boundary and one copy of the rows fewer.
_K3_AR_PUSH_AMID = _F.ar_push_amid
# TRTLLM_K3_AR_FINISH_FUSED=1 (with TRTLLM_K3_AR_PUSH_AMID): where A_mid reads the o_proj
# all-reduce from the Lamport workspace, A_mid itself finishes the pushed call (k3_attn_res
# ``lamport_finish``: one-shot transport up to TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot
# above, where each row's owner reduces and multicasts it) and advances the workspace: no finish
# kernel or C++ all-reduce runs in between. Where o_proj's GEMM has no push epilogue (the MXFP8
# GEMM on a prequantized KDA output) a copy kernel pushes its output. The sums have the order and
# rounding of the finish kernel, so A_mid produces the same bits.
_K3_AR_FINISH_FUSED = _F.ar_finish_fused
# TRTLLM_K3_ULL_OPROJ_SANDWICH=1 (with TRTLLM_K3_AR_FINISH_FUSED): A_mid finishes the pushed o_proj
# all-reduce before its griddepcontrol.wait (k3_attn_res ``early_poll``): it becomes resident when
# the o_proj push GEMM passes its own wait, polls every rank's slots while o_proj runs, and only
# its stores wait for the o_proj grid, so the o_proj -> A_mid boundary leaves the critical path
# (the post-attention collective sandwich as two PDL grids). Same bits.
_K3_ULL_OPROJ_SANDWICH = _F.ull_oproj_sandwich
# TRTLLM_K3_ULL_TAIL_SANDWICH=1 (with TRTLLM_K3_RP_TAIL): the next layer's pre-attention mix
# finishes the row-parallel MoE tail's pushed all-reduce the same way (the pre-attention sandwich:
# tail GEMM + push, then the early-polling mix). Same bits.
_K3_ULL_TAIL_SANDWICH = _F.ull_tail_sandwich
# TRTLLM_K3_ULL2_COLL_CP_TAIL=1 (with TRTLLM_K3_RP_TAIL, BF16 shared expert): the MoE tail of a
# generation-only forward splits latent_up by output columns instead of by latent columns. After
# the latent all-reduce + RMSNorm (k3_latent_ar_norm's row kernel with its MXFP8 output, as
# TRTLLM_K3_ULL_TAIL_SANDWICH) every rank computes its whole 128-row blocks of latent_up's output
# over the full latent (the MXFP8 low-M GEMM with the replicated projection's tactic) and multicasts
# them into the all-gather rows of the next call on a Lamport workspace of the tail's own
# (k3_lamport_ar "ag"); the shared expert's BF16 partial is pushed into the same call's slots on
# the shared stream as soon as it is ready. The next layer's pre-attention mix finishes the call
# (k3_attn_res ``finish_ag``): bf16(latent_up + bf16(all-reduced shared)), the replicated tail's
# bf16(routed + shared), so the result is bitwise the replicated tail (latent_up on every rank +
# the shared expert's MNNVL all-reduce) while each rank reads 1 / W of latent_up as the
# row-parallel tail does. Compared with the row-parallel tail, the critical path after the latent
# all-reduce carries 1 / W of the bytes (W x M x 7168 x 2 B of slots land on every rank, but the
# shared expert's land while the experts still run) and no two-shot step above the one-shot size.
# The latent all-reduce runs one-shot on a workspace of the tail's own, up to the C++ one-shot size
# (the sizes where the C++ all-reduce is one-shot too, so the sums keep its order); larger sizes keep
# the row-parallel tail. The mix polls the call before its griddepcontrol.wait (``early_poll``).
_K3_ULL2_COLL_CP_TAIL = _F.ull2_coll_cp_tail
# TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE=clear (with TRTLLM_K3_ULL2_COLL_CP_TAIL): the tail's latent
# all-reduce row kernel reads its workspace's rotation state, counts itself and re-arms the
# previous call's buffer before its griddepcontrol.wait (k3_latent_ar_norm ``early_state``: the
# previous call on that workspace, the previous MoE layer's, completed long before the routed
# experts launched). Same bits.
_K3_ULL2_COLL_AR_EARLY_STATE = _F.ull2_coll_ar_early_state
if _K3_ULL2_COLL_AR_EARLY_STATE not in ("0", "clear"):
    raise ValueError(
        f"TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE must be 0 or clear, got {_K3_ULL2_COLL_AR_EARLY_STATE!r}"
    )
# TRTLLM_K3_ULL_HYGIENE=1 (decode launch hygiene; the same bits): in CUDA-graph decode steps
# (multi-stream on)
# - the DSpark hidden-state taps (``maybe_capture_hidden_states``, a strided copy into the capture
#   buffer) run on a side stream forked after their source and joined at the end of the model
#   forward, instead of between the mix and the input norm of the tapped layer;
# - at an attention-residual block boundary served by the fused column-parallel / row-parallel tail
#   mix (``_k3_attn_res_pdl_rp_tail``), the mix writes the updated prefix sum straight into the
#   snapshot slot of the bank, so the snapshot copy (a memcpy graph node, which also cuts the PDL
#   edge to the next layer's first GEMM) is gone.
_K3_ULL_HYGIENE = _F.ull_hygiene
# TRTLLM_K3_ULL_TAIL1=1 (with TRTLLM_K3_ULL2_COLL_CP_TAIL and the ULL routed experts): the MoE tail
# of a generation-only forward of at most 8 rows is one kernel (k3_moe_tail1). The routed experts
# push every rank's latent partial rows into the slots of the latent workspace (k3_moe_ull's red
# combine); k3_moe_tail1 polls them, sums them in the latent all-reduce's order, applies the RMSNorm
# after the products, multiplies by this rank's HIDDEN / W rows of latent_up and multicasts them
# into the column-parallel tail call's all-gather rows, which the next pre-attention mix finishes as
# before (``finish_ag``). It replaces the latent all-reduce row kernel and the column-parallel GEMM.
# Not bitwise (BF16 activation instead of MXFP8, the RMS scale applied to the FP32 products).
_K3_ULL_TAIL1 = _F.ull_tail1
# TRTLLM_K3_ULL_SHARED_PERSIST=N (with TRTLLM_K3_ULL2_COLL_CP_TAIL): on the column-parallel tail
# path of at most 8 rows, the shared expert (gate/up, SiTU, down and the push of its partial into
# the tail call's slots) is one persistent kernel on N CTAs (k3_shared_persist) on the shared stream,
# beside the routed experts, which then run on the SMs left (TRTLLM_K3_ULL_EXPERTS_CTAS = SMs - N).
# The next pre-attention mix reads the slots as before. Not bitwise (BF16 GEMM products in another
# summation order, tanh / sigmoid with ex2.approx): GSM8K before adoption. Needs the BF16 GatedMLP
# shard with SiTU on the tail's TP group; the weights are copied once into the kernel's stream
# order (16.5 MB per MoE layer and rank at TP16).
_K3_ULL_SHARED_PERSIST = _F.ull_shared_persist
# The shared expert's part of the forward was pushed by the persistent kernel (no partial).
_SHARED_PUSHED = object()


class _SideCopies:
    """``TRTLLM_K3_ULL_HYGIENE``: copies that nothing reads before the end of the forward run on a
    side stream. Each is forked after its source; the sources stay referenced until
    :meth:`join` makes the current stream wait for every pending copy, so the caching allocator
    cannot hand their memory to a later kernel while a copy may still read it."""

    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream
        self._pending: List[Tuple[torch.Tensor, torch.cuda.Event]] = []

    @staticmethod
    def active() -> bool:
        """Side copies only where the model forks side streams (CUDA graph capture)."""
        return do_multi_stream() and not torch.compiler.is_compiling()

    def run(self, fn: Callable[[], None], source: torch.Tensor) -> None:
        """Run ``fn`` (a copy reading ``source``) on the side stream."""
        fork = torch.cuda.Event()
        fork.record()
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(fork)
            fn()
            done.record()
        self._pending.append((source, done))

    def join(self) -> None:
        current = torch.cuda.current_stream()
        for _, done in self._pending:
            current.wait_event(done)
        self._pending.clear()


# The column-parallel tail's Lamport workspaces, by (use, all-reduce mapping), and the multicast
# handles that own their memory.
_K3_CP_TAIL_WORKSPACES: Dict[Tuple[str, Mapping], Any] = {}
_K3_CP_TAIL_HANDLES: Dict[Tuple[str, Mapping], Any] = {}


def _k3_cp_tail_workspace(mapping: Mapping, hidden_size: int, max_tokens: int, use: str = "tail"):
    """``k3_lamport_ar.LamportWorkspace`` of the column-parallel MoE tail
    (``TRTLLM_K3_ULL2_COLL_CP_TAIL``) and the rotation state. ``use="tail"``: three Lamport buffers
    of ``max_tokens x (W + 1)`` BF16 rows of ``hidden_size`` (the one-shot slots plus the
    all-gather rows); ``"latent"``: ``max_tokens x W`` one-shot rows.
    Collective over the all-reduce group: every rank builds it at the same point of model
    construction. ``_K3_CP_TAIL_HANDLES[(use, mapping)]`` holds the multicast buffer."""
    workspace = _K3_CP_TAIL_WORKSPACES.get((use, mapping))
    if workspace is not None:
        return workspace
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    from .kernels.k3_lamport_ar import LamportWorkspace, required_bytes

    comm = _k3_mnnvl_mpi_comm(get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16))
    need = required_bytes(max_tokens, hidden_size, mapping.tp_size, True, ag=use == "tail")
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
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    peers = [
        handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(mapping.tp_size)
    ]
    workspace = LamportWorkspace(
        world=mapping.tp_size,
        rank=mapping.tp_rank,
        uc_local=buffer.data_ptr(),
        mc_base=handle.get_mc_buffer((1,), torch.uint8, 0).data_ptr(),
        peer_ptrs=torch.tensor(peers, dtype=torch.int64, device=buffer.device),
        flags=flags,
        bytes_per_buffer=buffer_bytes,
    )
    _K3_CP_TAIL_WORKSPACES[(use, mapping)] = workspace
    _K3_CP_TAIL_HANDLES[(use, mapping)] = handle
    return workspace


# TRTLLM_K3_MID_AR_PUSH=1 (with TRTLLM_K3_AR_PUSH=1): above the low-M range the o_proj push
# also runs, with the mid-M GEMM's push epilogue (k3_fp8w_lowm_gemm.mid_ar_push_tactic) instead of
# the BF16 output plus a copy kernel: on BF16 rows its quantization kernel + push GEMM (the bits of
# the mid-M GEMM), on the KDA verify's MXFP8 operand the push GEMM alone (up to 128 rows, where
# the R128c4 scales are the mid-M layout; the mid-M GEMM instead of the CuTe MXFP8 GEMM, so the
# projection's FP32 accumulation order can differ there).
_K3_MID_AR_PUSH = _F.mid_ar_push
# TRTLLM_K3_MID_AR_PUSH_MAX_M=<rows> (with TRTLLM_K3_MID_AR_PUSH): forward on BF16 rows
# pushes with the mid-M GEMM up to this many rows also above the per-shape gate of the low-M GEMM
# (k3_fp8w_lowm_gemm.max_m_for), where it would otherwise run the CuTe MXFP8 GEMM and a copy kernel
# (0: off). Other numerics than that GEMM above the gate (the mid-M GEMM's accumulation order).
_K3_MID_AR_PUSH_MAX_M = 0
# TRTLLM_K3_LU_MID_MAX_M=<rows> (with TRTLLM_K3_LU_MID_MXFP8): latent_up takes the latent
# all-reduce row kernel's MXFP8 operand and runs the mid-M GEMM on it up to this many rows also above the
# per-shape gate (k3_fp8w_lowm_gemm.max_m_for(7168, 3584) = 96), where forward would otherwise quantize the
# latent in a kernel of its own and run the CuTe MXFP8 GEMM (0: off). Other numerics than that path above
# the gate (1x32 instead of 1x128 activation scales, the mid-M GEMM's accumulation order).
_K3_LU_MID_MAX_M = 0
# TRTLLM_K3_MID_MXFP8_IN=1 (with TRTLLM_K3_ATTN_RES_MXFP8=1): the attention's first
# projection (KDA fused QKVG, MLA fused q_a / kv_a) also takes the pre-attention mix's MXFP8 form
# above the low-M range, up to 128 rows, where it runs the mid-M GEMM: the mix stores the scales
# in that GEMM's layout (k3_attn_res ``mx_atom``) and the GEMM runs on them
# (``run_mid_prequantized``) without its quantization kernel; bitwise the BF16 path.
_K3_MID_MXFP8_IN = _F.mid_mxfp8_in
# TRTLLM_K3_RP_TAIL=1: row-parallel MoE tail for generation-only batches of up to
# TRTLLM_K3_RP_TAIL_MAX_M tokens (the low-M GEMM's range), where the routed latent is all-reduced
# by the model (MoE TP/EP over the all-reduce group, TP-sharded shared expert) and the layer's
# residual update is fused (TRTLLM_K3_FUSE_MOE_RESIDUAL). After the latent all-reduce + RMSNorm,
# every rank runs one low-M FP8 GEMM over its whole 128-column blocks of the normalized latent
# with the matching latent_up columns (FP8 codes and block scales read in place) and pushes its
# partial [M, hidden] plus its shared-expert partial into the next all-reduce on the latent
# all-reduce's MNNVL workspace: the shared expert keeps its BF16 down GEMM (row-parallel shard, no
# all-reduce) on its stream, and the tail GEMM's push epilogue adds that BF16 partial to its BF16
# output. The next layer's pre-attention mix finishes that
# all-reduce (one-shot transport up to TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot above)
# and adds it to the residual (``_DeferredRpTail``). This replaces
# the shared expert's all-reduce and the replicated latent_up.
# Not bitwise: the latent_up sum is split over the ranks.
_K3_RP_TAIL = _F.rp_tail
_K3_RP_TAIL_MAX_M = _F.rp_tail_max_m
# TRTLLM_K3_ULL_FRONT=1: MoE forwards of 1 .. TRTLLM_K3_ULL_FRONT_MAX_M (<= 32) rows run the MoE
# front of kernels/k3_ull_front.py on the routed all-reduce's TP group, two kernels
# on the main stream: each rank computes the router logits of every expert (k3_router_exact's
# arithmetic) and the latent_down columns of its 128-column tiles (the low-M GEMM of
# k3_latent_down_fp4) into an MNNVL workspace (the latent is multicast to every rank); a routing-tail
# cluster polls the logits and writes the top-16 and this rank's moe_sort tables, and gather CTAs
# write the NVFP4 latent. This replaces the router GEMM, the routing tail, the replicated latent_down
# + quantization and the latent stream's fork / join. The shared expert forks at the front's start.
# Routing is bitwise (per 16-row tile of the router); the latent is bitwise up to 16 rows (above,
# k3_latent_down_fp4 runs the mid-M kernel, split 4, and the front the low-M kernel, split 4).
# With CUTEDSL_FC12 the route kernel also zeroes the routed output and FC12's counters (as
# TRTLLM_K3_FC12_PREZERO does on a side stream), so FC12 follows it without a fill kernel and a
# memset.
_K3_ULL_FRONT = _F.ull_front
_K3_ULL_FRONT_MAX_M = 32
# TRTLLM_K3_ULL_FRONT_EARLY=1 (with TRTLLM_K3_ULL_FRONT and the A_mid that finishes the pushed
# o_proj all-reduce, TRTLLM_K3_AR_FINISH_FUSED): A_mid releases the front's head at kernel entry
# (k3_attn_res ``entry_trigger``) where the head follows it, so the head's CTAs become resident
# while A_mid polls the o_proj all-reduce and stream the router and latent_down weights into
# shared memory before their griddepcontrol.wait; after the wait only the activation loads, the
# MMAs and the pushes remain. Before its wait the head reads only weights and writes nothing, and
# it releases the route kernel only after its wait. The early-polling A_mid otherwise releases it
# after its stores. Same bits.
# Forwards of up to _K3_ULL_FRONT_FORK_AT_HEAD_MAX_M rows then fork the shared expert after the
# head (between the head and the route kernel) instead of at the front's start: under CUDA graph
# capture the shared GEMMs become programmatic dependents of the kernel the fork follows, so a fork
# at the start lets the entry-releasing A_mid release them with the head, and their CTAs take the
# SMs the head's CTAs need while A_mid polls (on helix16, forking at the head gave the shorter step
# at the smaller row counts; at 24 / 32 rows the start stays ahead, where the larger gate_up
# competes with the routed experts instead).
_K3_ULL_FRONT_EARLY = _F.ull_front_early
_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M = 16


def _ull_shared_fork(lead: bool, rows: int) -> Optional[str]:
    """Where the ULL front forks the shared expert's GEMMs on ``rows`` rows (``None``: no fork, ``lead`` is
    ``_shared_gemms_can_lead()``).

    * ``"start"``: ``moe_main_event`` is recorded and the shared GEMMs are captured before the front's head.
    * ``"head"`` (``TRTLLM_K3_ULL_FRONT_EARLY``, up to ``_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M`` rows): the event is
      recorded after the head, so the shared GEMMs depend on the head.
    * ``"after_head"`` (``TRTLLM_K3_ULL_FRONT_EARLY``, more rows): the event is recorded before the head, so the
      shared GEMMs depend on what precedes the head (the entry-releasing A_mid), as with the fork at the start.

    With ``TRTLLM_K3_ULL_FRONT_EARLY`` the shared GEMMs themselves are issued after the front
    (``KimiK3MoERuntime._ull_fork_shared``).
    """
    if not lead:
        return None
    if not _K3_ULL_FRONT_EARLY:
        return "start"
    return "head" if rows <= _K3_ULL_FRONT_FORK_AT_HEAD_MAX_M else "after_head"


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        from .moe.kimi_k3_moe_residual import _fused_moe_residual_add

        return _fused_moe_residual_add(self.routed, self.shared, self.residual)


class _DeferredRpTail(NamedTuple):
    """A MoE layer's ``bf16(residual + all-reduce of the ranks' row-parallel tail partials)``
    (``TRTLLM_K3_RP_TAIL``): ``result`` is the pushed all-reduce, left for the next layer's
    pre-attention mix to finish (``LamportPending``; a ``LamportResult`` finished in place is read
    the same way)."""

    residual: torch.Tensor
    result: Any

    def materialize(self) -> torch.Tensor:
        # A BF16 add rounds the FP32 sum to nearest, as the fused mix does.
        return self.residual + self.result.materialize().view(self.residual.shape)


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return _K3_ATTN_RES_PDL_MODE == "2" and num_tokens <= _K3_ATTN_RES_PDL_MAX_M


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, (_DeferredMoEResidual, _DeferredRpTail)):
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
    from .kernels import k3_attn_res

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
    (``TRTLLM_K3_AR_PUSH_AMID``): the vectorized PDL attn_res kernel serves the ADD_PREFIX mix."""
    if (
        not _K3_AR_PUSH_AMID
        or not _K3_ATTN_RES_PDL
        or torch.compiler.is_compiling()
        or prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M
    ):
        return False
    from .kernels import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    return k3_attn_res.can_use_k3_attn_res(prefix_sum, (prefix_sum,), block_residual, *weights)


def _k3_attn_res_pdl_lamport(
    prefix_sum: torch.Tensor,
    addend: Union[LamportResult, LamportPending],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    entry_trigger: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum + all-reduce result, normed mix)`` with the result read from the Lamport
    workspace (``trtllm::k3_attn_res_add_rmsnorm_lamport``; for a ``LamportPending`` the mix also
    finishes the all-reduce, ``TRTLLM_K3_AR_FINISH_FUSED``), or ``None`` if that does not apply.
    ``entry_trigger`` (finishing mix only): release the dependent at kernel entry
    (``TRTLLM_K3_ULL_FRONT_EARLY``: the caller checked that the next launch is the front's head)."""
    if tuple(addend.shape) != tuple(prefix_sum.shape) or not _k3_attn_res_lamport_applies(
        prefix_sum, block_residual, proj, norm, output_norm
    ):
        return None
    M, H = prefix_sum.shape
    if isinstance(addend, LamportPending):
        _note_attn_res_fusion(
            "add+attn_res+norm/pdl/lamport_finish" + ("/entry" if entry_trigger else ""),
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
            entry_trigger,
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


def _k3_attn_res_pdl_rp_tail(
    deferred: _DeferredRpTail,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    mxfp8: bool = False,
    entry_trigger: bool = False,
    prefix_out: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred row-parallel MoE tail (``TRTLLM_K3_RP_TAIL``):
    the mix reads the all-reduce from the Lamport workspace, finishing it where it is still
    pending, and adds it to the residual with a round-to-nearest BF16 add; ``None`` if the
    vectorized PDL kernel does not serve the shape. With ``mxfp8``, ``(prefix_sum, normed mix, xq,
    xs)``. ``entry_trigger`` (with ``mxfp8``): the kernel releases its dependent at entry (a KDA
    QKVG projection that stages its weights, as in ``_k3_attn_res_pdl_moe_add``). ``prefix_out``
    (``TRTLLM_K3_ULL_HYGIENE``, a contiguous buffer of the residual's shape outside
    ``block_residual``): the kernel stores ``prefix_sum`` there (the same bits)."""
    residual, result = deferred
    if tuple(result.shape) != tuple(residual.shape) or torch.compiler.is_compiling():
        return None
    from .kernels import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (residual,), block_residual, *weights):
        return None
    M, H = residual.shape
    pending = isinstance(result, LamportPending)
    _note_attn_res_fusion(
        "rp_tail+attn_res+norm/pdl/" + ("lamport_finish" if pending else "lamport"),
        True,
        M,
        H,
        int(block_residual.shape[0]) + 1,
    )
    ws = result.workspace
    op = (
        torch.ops.trtllm.k3_attn_res_lamport_add_rmsnorm_mx
        if mxfp8
        else torch.ops.trtllm.k3_attn_res_lamport_add_rmsnorm
    )
    # TRTLLM_K3_ULL2_COLL_CP_TAIL: the call carries the column-parallel tail's all-gather rows.
    finish_ag = pending and result.finish_ag
    if prefix_out is not None:
        op = functools.partial(k3_attn_res.k3_attn_res_lamport_add_rmsnorm_into, prefix_out, mxfp8)
    return op(
        residual,
        ws.uc_local,
        ws.flags.data_ptr(),
        block_residual,
        *weights,
        float(norm.eps),
        _rms_norm_eps(output_norm),
        *(result.finish_args if pending else (0, 0, False, 0, 0)),
        True,
        pending and (_K3_ULL_TAIL_SANDWICH or finish_ag),
        entry_trigger and mxfp8,
        finish_ag,
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    mxfp8: bool = False,
    entry_trigger: bool = False,
    mx_atom: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``; with
    ``mxfp8``, ``(prefix_sum, normed mix, xq, xs)`` (the MXFP8 form of the mix).
    ``entry_trigger`` (with ``mxfp8``): the kernel releases its dependent at entry
    (``k3_attn_res.run_k3_attn_res``), for a KDA QKVG projection that stages its weights.
    ``mx_atom`` (with ``mxfp8``, ``TRTLLM_K3_MID_MXFP8_IN``): ``xs`` in the mid-M GEMM's layout."""
    residual, routed, shared = deferred
    if residual.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    from .kernels import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    if mxfp8:
        return torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm_mx(
            residual,
            routed,
            shared,
            block_residual,
            *weights,
            float(norm.eps),
            _rms_norm_eps(output_norm),
            entry_trigger,
            mx_atom,
        )
    return torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm(
        residual,
        routed,
        shared,
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
    entry_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    ``addend`` may be a ``LamportResult`` (``TRTLLM_K3_AR_PUSH_AMID``): the attention block's
    all-reduce result left in the MNNVL workspace, read in place where the PDL attn_res kernel
    serves the mix and gathered into a tensor otherwise; or a ``LamportPending``
    (``TRTLLM_K3_AR_FINISH_FUSED``), which the PDL kernel finishes or the finish kernel turns into
    a tensor. ``entry_trigger``: see :func:`_k3_attn_res_pdl_lamport`."""
    if isinstance(addend, (LamportResult, LamportPending)):
        fused = _k3_attn_res_pdl_lamport(
            prefix_sum, addend, block_residual, proj, norm, output_norm, entry_trigger
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
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _low_m_gemm_scale_from_mxfp8(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from the SM107 CuTe MXFP8 scale."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


def _k3_lowm_gemm_can_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
) -> bool:
    """Whether ``trtllm::k3_fp8w_lowm_gemm`` serves this projection (registers the op)."""
    from .kernels import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm.can_run(
        x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile
    )


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

    The Rubin agentic kernels sit in front of that dispatch: the K3 low-M GEMM
    (``TRTLLM_K3_FP8W_LOWM_GEMM``, ``kernels/k3_fp8w_lowm_gemm.py``) with its gated,
    sigmoid, MXFP8-input and all-reduce-push variants. With ``TRTLLM_K3_FP8W_LOWM_GEMM``
    set the K3 low-M GEMM owns the low-M rows and the generic low-M GEMM is not
    prepared; with it unset the generic one serves them as on main.
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
        # Set by a caller for the next forward only: a contiguous BF16 ``[M, out_features]``
        # tensor that the low-M GEMM writes its result into. Other paths return their own
        # tensor, so the caller compares the result with the buffer it handed in.
        self.next_output: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only (every forward consumes and clears it): the
        # ``LamportPushTarget`` (``comm.lamport.lamport_push_target``) of the all-reduce that
        # follows this projection (TRTLLM_K3_AR_PUSH). Where the low-M GEMM without split-K
        # serves ``forward`` / ``forward_gated``, it stores its output into that all-reduce's
        # Lamport buffers instead, marks the target pushed and returns the all-reduce's
        # output tensor, which ``comm.lamport.finish_lamport_push`` fills.
        self.next_ar_push = None
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # Whether the fused latent projection (TRTLLM_K3_LD_FUSED) may replace this one (set when
        # it is swapped in).
        self.k3_latent_fused = False
        # SM107 with TRTLLM_K3_FP8W_LOWM_GEMM or ``k3_latent_fused``: the weight scale in the
        # DeepGEMM-packed UE8M0 layout of the low-M GEMM and the fused latent projections
        # (``weight_scale`` holds the CuTe MXFP8 layout there).
        self.lowm_weight_scale: Optional[torch.Tensor] = None

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

    @property
    def packed_weight_scale(self) -> Optional[torch.Tensor]:
        """The weight scale in DeepGEMM's packed UE8M0 layout, as the low-M GEMM and the fused
        latent projections read it: ``weight_scale`` on SM100/SM103, ``lowm_weight_scale`` on
        SM107 (``None`` unless :meth:`prepare_lowm_gemm` built it)."""
        if _rubin_mxfp8_is_available():
            return self.lowm_weight_scale
        return self.weight_scale

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer that can emit this GEMM's scale layout should hand ``num_rows``
        rows to :meth:`forward_prequantized` rather than BF16 rows to :meth:`forward`.

        False where :meth:`forward` serves BF16 rows with the low-M GEMM (activation quantized
        inside the kernel) faster than the CuTe MXFP8 GEMM runs on the prequantized operand.
        """
        if not self.supports_prequantized_input:
            return False
        if self.lowm_weight_scale is None:
            return True
        from .kernels import k3_fp8w_lowm_gemm

        return num_rows > k3_fp8w_lowm_gemm.bf16_over_prequantized_max_m(
            self.out_features, self.in_features
        )

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
        from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear._prepare_weights(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_weights(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm.quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

        # Both GEMMs consume UE8M0 scales rather than the checkpoint's FP32 grid.
        weight_fp8, weight_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        return weight_fp8, _Fp8BlockScaleWeightReadLinear._prepare_scale_layout(
            weight_scale, weight_fp8.shape[0], weight_fp8.shape[1]
        )

    @staticmethod
    def _prepare_scale_layout(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
        from tensorrt_llm.quantization.utils.fp8_utils import (
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            return transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        return transform_sf_into_required_layout(
            weight_scale,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear | TrtllmLinear) -> "_Fp8BlockScaleWeightReadLinear":
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            result.fp8w_lowm_partial_n_tile = getattr(linear, "fp8w_lowm_partial_n_tile", False)
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                if self.weight_scale.dtype != torch.float32 or self.weight_scale.ndim != 2:
                    raise ValueError("KDA fused FP8 views require a raw FP32 scale grid")
                # Codes and raw scale grids alias the fused projection. Only its
                # owner may resmooth them in place; each view packs the updated
                # scales without modifying the shared codes again.
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
                self.weight_scale = self._prepare_scale_layout(
                    self.weight_scale, self.out_features, self.in_features
                )
            else:
                self.weight, self.weight_scale = self._prepare_weights(
                    self.weight, self.weight_scale
                )
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.prepare_lowm_gemm()
        self._prepare_low_m_gemm()

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode). Not prepared under ``TRTLLM_K3_FP8W_LOWM_GEMM``: the K3 low-M GEMM
        owns those rows then (see the class docstring).
        """
        if _K3_FP8W_LOWM_GEMM:
            self._low_m, self._low_m_weight_scale = Fp8BlockScaleLowMGemmRange(), None
            return
        n, k = self.out_features, self.in_features
        self._low_m = Fp8BlockScaleLowMGemmRange.for_weight(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_weight_scale = self.weight_scale
        elif self._low_m.max_m and self._fused_projection is None:
            self._low_m_weight_scale = _low_m_gemm_scale_from_mxfp8(self.weight_scale, n, k)
        else:
            self._low_m, self._low_m_weight_scale = Fp8BlockScaleLowMGemmRange(), None

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 if TRTLLM_K3_FP8W_LOWM_GEMM is set or a fused
        latent projection may replace this one (``k3_latent_fused``).

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not ((_K3_FP8W_LOWM_GEMM or self.k3_latent_fused) and _rubin_mxfp8_is_available()):
            return False
        if self._fused_projection is not None:
            # A row view of the KDA fused q/k/v/g weight; the fused module serves forward.
            return False
        self.transform_weights()
        from .kernels import k3_fp8w_lowm_gemm

        n, k = self.weight.shape
        if n % k3_fp8w_lowm_gemm.N_ALIGN or k % 128:
            return False
        self.lowm_weight_scale = k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout(
            self.weight_scale, n, k
        )
        return True

    def _lowm_gemm_applies(self, x_flat: torch.Tensor, weight_scale: torch.Tensor) -> bool:
        """Whether ``trtllm::k3_fp8w_lowm_gemm`` serves ``x_flat`` with ``weight_scale``."""
        return x_flat.dtype == torch.bfloat16 and _k3_lowm_gemm_can_run(
            x_flat,
            self.weight,
            weight_scale,
            allow_partial_n_tile=_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile,
        )

    def _run_lowm_gemm(
        self,
        x_flat: torch.Tensor,
        weight_scale: torch.Tensor,
        out_buffer: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The low-M GEMM, into ``out_buffer`` when the caller handed in a usable one."""
        if (
            out_buffer is not None
            and out_buffer.dtype == torch.bfloat16
            and out_buffer.shape == (x_flat.shape[0], self.out_features)
            and out_buffer.is_contiguous()
        ):
            from .kernels.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm_out

            return k3_fp8w_lowm_gemm_out(x_flat, self.weight, weight_scale, out_buffer)
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(x_flat, self.weight, weight_scale)

    def _take_next_output(self, num_rows: int) -> Optional[torch.Tensor]:
        """Clear :attr:`next_output`; return it if it is a contiguous, 16-byte aligned BF16
        ``[num_rows, out_features]`` tensor."""
        out, self.next_output = self.next_output, None
        if (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (num_rows, self.out_features)
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return None
        return out

    def _take_next_ar_push(self, x_flat: torch.Tensor, gated: bool = False):
        """Clear :attr:`next_ar_push`; return it if the low-M GEMM can push this forward's
        ``[M, out_features]`` output into it (low-M kernel without split-K)."""
        push, self.next_ar_push = self.next_ar_push, None
        if (
            push is None
            or x_flat.dtype != torch.bfloat16
            or tuple(push.output.shape) != (x_flat.shape[0], self.out_features)
        ):
            return None
        from .kernels import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.ar_push_tactic(
            x_flat.shape[0], self.out_features, self.in_features, gated=gated
        )
        if tactic is None and _K3_MID_AR_PUSH and not gated:
            tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(
                x_flat.shape[0], self.out_features, self.in_features
            )
        return push if tactic is not None else None

    def mid_push_pdl_ready(self, num_rows: int) -> bool:
        """``TRTLLM_K3_KDA_MX_PDL``: whether :meth:`forward_prequantized` on the producer's
        ``[num_rows, in_features]`` R128c4 operand will push with the mid-M GEMM (the conditions
        of :meth:`_mid_prequantized_push_tactic` that do not depend on the tensors), which reads
        nothing the producer writes before its ``griddepcontrol.wait``: the producer may then
        release it early."""
        push = self.next_ar_push
        if not _K3_MID_AR_PUSH or push is None or self.lowm_weight_scale is None:
            return False
        if torch.compiler.is_compiling() or num_rows > 128:
            return False
        if tuple(push.output.shape) != (num_rows, self.out_features):
            return False
        from .kernels import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(num_rows, self.out_features, self.in_features)
        return tactic is not None and tactic.mma_n <= 128

    def _mid_prequantized_push_tactic(self, activation: torch.Tensor, scale: torch.Tensor, push):
        """``TRTLLM_K3_MID_AR_PUSH``: the mid-M tactic :meth:`forward_prequantized` pushes the KDA
        verify's MXFP8 operand with, or None. Up to 128 rows (one scale-factor block), where the
        R128c4 scales of the operand are the mid-M GEMM's ``[K / 128][1][512 B]`` layout."""
        if not _K3_MID_AR_PUSH or push is None or self.lowm_weight_scale is None:
            return None
        if torch.compiler.is_compiling() or activation.dim() != 2 or not activation.is_contiguous():
            return None
        m, k = activation.shape
        if k != self.in_features or tuple(push.output.shape) != (m, self.out_features):
            return None
        from .kernels import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(m, self.out_features, k)
        if tactic is None or tactic.mma_n > 128:
            return None
        if scale.numel() != (k // 128) * 512 or not scale.is_contiguous():
            return None
        return tactic

    def _run_ar_push(self, x_flat: torch.Tensor, push, gate=None) -> torch.Tensor:
        """The low-M GEMM with the all-reduce push epilogue; returns the all-reduce output."""
        from .kernels.k3_fp8w_lowm_gemm import run_k3_fp8w_lowm_gemm_ar_push

        run_k3_fp8w_lowm_gemm_ar_push(
            x_flat, self.weight, self.packed_weight_scale, push.workspace, push.one_shot, gate=gate
        )
        push.pushed = True
        return push.output

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
        out_buffer, self.next_output = self.next_output, None
        ar_push = self._take_next_ar_push(x_flat)

        if (
            ar_push is not None
            and _K3_MID_AR_PUSH
            and x_flat.shape[0] <= _K3_MID_AR_PUSH_MAX_M
            and self.packed_weight_scale is not None
            and not self._lowm_gemm_applies(x_flat, self.packed_weight_scale)
        ):
            # TRTLLM_K3_MID_AR_PUSH_MAX_M: above the per-shape gate, where
            # forward would run the CuTe MXFP8 GEMM followed by the copy kernel, push with the
            # mid-M GEMM (faster at 120 / 144 rows in our measurements).
            out = self._run_ar_push(x_flat, ar_push)
        elif (
            _K3_FP8W_LOWM_GEMM
            and self.lowm_weight_scale is not None
            and self._lowm_gemm_applies(x_flat, self.lowm_weight_scale)
        ):
            # SM107 (only there is lowm_weight_scale set): up to the per-shape M gate.
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            else:
                out = self._run_lowm_gemm(x_flat, self.lowm_weight_scale, out_buffer)
        elif _rubin_mxfp8_is_available() and self._low_m.serves(x_flat):
            # Main's low-M GEMM (prepared only without TRTLLM_K3_FP8W_LOWM_GEMM).
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale
            )
        elif _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                self.weight,
                act_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        elif _K3_FP8W_LOWM_GEMM and self._lowm_gemm_applies(x_flat, self.weight_scale):
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            else:
                out = self._run_lowm_gemm(x_flat, self.weight_scale, out_buffer)
        elif x.dtype is torch.bfloat16 and self._low_m.serves(x_flat):
            # SM100/SM103: main's fp8_block_scale_ue8m0_gemm low-M branch.
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self.weight_scale
            )
        else:
            out = torch.ops.trtllm.fp8_swap_ab_gemm(
                x_flat,
                self.weight,
                self.weight_scale,
                output_dtype=x.dtype,
                disable_ue8m0_cast=True,
            )
        return out.reshape(out_shape)

    def _lowm_forward_scale(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """The weight scale :meth:`forward` runs ``trtllm::k3_fp8w_lowm_gemm`` with on the BF16
        rows ``x_flat`` (``lowm_weight_scale`` on SM107, ``weight_scale`` on SM100/SM103), or
        None where :meth:`forward` runs another GEMM."""
        if (
            not _K3_FP8W_LOWM_GEMM
            or torch.compiler.is_compiling()
            or x_flat.dtype != torch.bfloat16
        ):
            return None
        if self.lowm_weight_scale is not None:
            scale = self.lowm_weight_scale
        elif _rubin_mxfp8_is_available():
            return None
        else:
            scale = self.weight_scale
        return scale if self._lowm_gemm_applies(x_flat, scale) else None

    def _runs_low_m_kernel(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows would run the low-M kernel (the
        input-independent part of its dispatch: the low-M GEMM on, not the Rubin CuTe MXFP8
        path, ``num_rows`` within the low-M kernel's range). On SM107 forward runs it on
        ``lowm_weight_scale`` up to the per-shape gate (``max_m_for``)."""
        if not _K3_FP8W_LOWM_GEMM or torch.compiler.is_compiling():
            return False
        if _rubin_mxfp8_is_available() and self.lowm_weight_scale is None:
            return False
        if self.weight.shape[0] % 128 and not (
            _K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile
        ):
            return False
        from .kernels import k3_fp8w_lowm_gemm

        limit = k3_fp8w_lowm_gemm.low_m_max_for(*self.weight.shape)
        if self.lowm_weight_scale is not None:
            limit = min(limit, k3_fp8w_lowm_gemm.max_m_for(*self.weight.shape))
        return 1 <= num_rows <= limit

    def takes_mxfp8_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_mxfp8` the MXFP8 form of ``num_rows``
        rows: where :meth:`forward` would run the low-M kernel, whose in-kernel quantization the
        MXFP8 GEMM replaces (the attention's first projection also asks
        :meth:`takes_mid_mxfp8_input`)."""
        return self._runs_low_m_kernel(num_rows)

    def takes_mid_mxfp8_input(self, num_rows: int) -> bool:
        """``TRTLLM_K3_MID_MXFP8_IN``: whether :meth:`forward` on ``num_rows`` BF16 rows
        runs the mid-M path of ``trtllm::k3_fp8w_lowm_gemm`` (above the low-M cut-over, up to the
        per-shape gate) with one 128-row scale block, so that a producer may hand
        :meth:`forward_mxfp8` the rows' MXFP8 form with the scales in that GEMM's layout
        (``k3_attn_res`` ``mx_atom``): the GEMM then runs without its quantization kernel."""
        return _K3_MID_MXFP8_IN and self.mid_operand_tactic(num_rows) is not None

    def mid_operand_tactic(self, num_rows: int):
        """The mid-M tactic :meth:`forward` runs on ``num_rows`` BF16 rows (SM107 low-M GEMM
        switch on, above the low-M cut-over, up to the per-shape gate) when it has one 128-row
        scale block, else None. A producer may then store the rows' MXFP8 form in that GEMM's
        layout (``mid_prequant_plan``) for :meth:`forward_mxfp8` / :meth:`forward_mid_mxfp8`."""
        if not _K3_FP8W_LOWM_GEMM or torch.compiler.is_compiling():
            return None
        if self.lowm_weight_scale is None:
            return None
        n, k = self.weight.shape
        if n % 128 and not (_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile):
            return None
        from .kernels import k3_fp8w_lowm_gemm

        if (
            not k3_fp8w_lowm_gemm.low_m_max_for(n, k)
            < num_rows
            <= min(128, k3_fp8w_lowm_gemm.max_m_for(n, k))
        ):
            return None
        plan = k3_fp8w_lowm_gemm.mid_prequant_plan(num_rows, n, k)
        return plan[0] if plan is not None and plan[1] == 1 else None

    def forward_mid_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """``TRTLLM_K3_MID_OPROJ_PC``: :meth:`forward` on rows whose MXFP8 form a producer
        stored with the scales in the mid-M GEMM's layout (``xs`` 1-D, see
        :meth:`mid_operand_tactic`; ``x`` only gives the output shape): the mid-M GEMM without
        its quantization kernel, with the all-reduce push epilogue where :attr:`next_ar_push`
        takes it. Bitwise :meth:`forward` on the rows the producer quantized."""
        from .kernels import k3_fp8w_lowm_gemm

        m = xq.shape[0]
        tactic = self.mid_operand_tactic(m)
        if tactic is None or xs.dim() != 1:
            raise RuntimeError("forward_mid_mxfp8 needs the mid-M operand layout")
        x_flat = x.reshape(-1, x.shape[-1])
        push = self._take_next_ar_push(x_flat)
        self.next_output = None
        if push is not None:
            k3_fp8w_lowm_gemm.run_mid_prequantized_ar_push(
                xq.reshape(m, -1),
                xs,
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
        out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_lowm_gemm.run_mid_prequantized(
            xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic
        )
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def _staged_above_cutover(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` serves ``num_rows`` rows above the
        low-M kernel's cut-over with the staged MXFP8 low-M GEMM: :meth:`forward` runs the mid-M
        path of ``trtllm::k3_fp8w_lowm_gemm`` there, whose bits the staged kernel reproduces
        (``k3_fp8w_lowm_mxfp8.staged_matches_mid_m``)."""
        if (
            not _K3_FP8W_LOWM_GEMM
            or torch.compiler.is_compiling()
            or self.lowm_weight_scale is None
        ):
            return False
        from .kernels import k3_fp8w_lowm_gemm, k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if num_rows > k3_fp8w_lowm_gemm.max_m_for(n, k):
            return False
        return k3_fp8w_lowm_mxfp8.staged_matches_mid_m(num_rows, n, k)

    def takes_staged_mxfp8_input(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` runs the staged MXFP8 low-M GEMM on
        the MXFP8 form of ``num_rows`` rows (bitwise :meth:`forward`): where :meth:`forward` runs
        the low-M kernel, and above its cut-over where it runs the mid-M path with the same
        split-K (up to 64 rows)."""
        from .kernels import k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if self._runs_low_m_kernel(num_rows):
            # TRTLLM_K3_KDA_QKVG_STAGE_RING: also where the shared-memory ring alone holds the
            # CTA's whole weight slice.
            return k3_fp8w_lowm_mxfp8.weights_on_chip(num_rows, n, k)
        return self._staged_above_cutover(num_rows)

    def forward_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, stage_weights: bool = False
    ) -> torch.Tensor:
        """:meth:`forward` on the BF16 rows ``x``, bit for bit, given their MXFP8 form ``(xq,
        xs)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, e.g. from
        ``trtllm::k3_attn_res_*_mx``): the low-M GEMM on the prequantized operand where
        :meth:`forward` would run the low-M kernel, else :meth:`forward` on ``x``.
        ``stage_weights``: the low-M GEMM keeps its whole weight slice on chip before its
        ``griddepcontrol.wait`` (``k3_fp8w_lowm_mxfp8.staged_mx_tactic``, the same bits), also
        above the low-M cut-over where :meth:`takes_staged_mxfp8_input` says so."""
        from .kernels import k3_fp8w_lowm_mxfp8

        x_flat = x.reshape(-1, x.shape[-1])
        if xs.dim() == 1 and self.takes_mid_mxfp8_input(x_flat.shape[0]):
            # TRTLLM_K3_MID_MXFP8_IN: the producer stored the scales in the mid-M GEMM's layout.
            from .kernels import k3_fp8w_lowm_gemm

            m = x_flat.shape[0]
            tactic, _ = k3_fp8w_lowm_gemm.mid_prequant_plan(m, self.out_features, self.in_features)
            self.next_ar_push = None
            out = self._take_next_output(m)
            if out is None:
                out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x.device)
            k3_fp8w_lowm_gemm.run_mid_prequantized(
                xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic
            )
            return out.reshape(x.shape[:-1] + (self.out_features,))
        allow_partial = _K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile
        # The scale forward's low-M branch uses (lowm_weight_scale on SM107).
        w_scale = self.packed_weight_scale
        above_cutover = (
            stage_weights
            and not self._runs_low_m_kernel(x_flat.shape[0])
            and self._staged_above_cutover(x_flat.shape[0])
        )
        if not (
            x.dtype == torch.bfloat16
            and (above_cutover or self._runs_low_m_kernel(x_flat.shape[0]))
            and xq.shape[0] == x_flat.shape[0]
            and _k3_lowm_gemm_can_run(
                x_flat, self.weight, w_scale, allow_partial_n_tile=allow_partial
            )
            and k3_fp8w_lowm_mxfp8.can_run_mxfp8(
                xq,
                xs,
                self.weight,
                w_scale,
                allow_partial_n_tile=allow_partial,
                above_cutover=above_cutover,
            )
        ):
            return self.forward(x)
        out = self._take_next_output(x_flat.shape[0])
        # The MXFP8 GEMM has no all-reduce push epilogue: the caller runs the all-reduce.
        self.next_ar_push = None
        if out is None:
            out = torch.empty(
                (x_flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        stage_weights = stage_weights and k3_fp8w_lowm_mxfp8.weights_on_chip(
            x_flat.shape[0], self.out_features, self.in_features
        )
        k3_fp8w_lowm_mxfp8.run_k3_fp8w_lowm_gemm_mxfp8(
            xq, xs, self.weight, w_scale, out, stage_weights=stage_weights
        )
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def gated_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated` serves ``x`` and ``gate`` with one gated low-M GEMM.

        That is where :meth:`forward` would run the low-M GEMM on ``x * gate`` (same kernel
        family and split), so the gated launch is bitwise equal to the multiply plus
        :meth:`forward`.
        """
        if x.dim() != 2 or not x.is_contiguous():
            return False
        scale = self._lowm_forward_scale(x)
        if scale is None:
            return False
        from .kernels import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.can_run_gated(
            x,
            gate,
            self.weight,
            scale,
            allow_partial_n_tile=_K3_FP8W_LOWM_FUSED_A and self.fp8w_lowm_partial_n_tile,
        )

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one low-M GEMM launch with a sigmoid
        epilogue where :meth:`forward` would run the low-M GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        scale = self._lowm_forward_scale(x_flat)
        if scale is not None:
            from .kernels import k3_fp8w_lowm_gemm

            if x_flat.shape[0] <= k3_fp8w_lowm_gemm.low_m_max_for(
                self.out_features, self.in_features
            ):
                self.next_output = None
                self.next_ar_push = None
                out = k3_fp8w_lowm_gemm.k3_fp8w_lowm_gemm_sigmoid(x_flat, self.weight, scale)
                return out.reshape(x.shape[:-1] + (self.out_features,))
        return self.forward(x).sigmoid()

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)`` in one gated low-M GEMM launch (where :meth:`gated_applies`).

        ``x`` and ``gate`` are BF16 ``[M, in_features]``; the gate may be a row-strided view.
        """
        from .kernels.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm_gated

        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            return self._run_ar_push(x, ar_push, gate=gate)
        return k3_fp8w_lowm_gemm_gated(
            x, gate, self.weight, self.packed_weight_scale, out=self._take_next_output(x.shape[0])
        )

    def gated_mxfp8_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated_mxfp8` runs the MXFP8 low-M GEMM on the producer's MXFP8
        form of ``bf16(x * gate)`` (``TRTLLM_K3_OPROJ_PC``): where :meth:`forward_gated` runs
        the gated low-M GEMM, whose tiling the MXFP8 kernel shares (same bits)."""
        if not self.gated_applies(x, gate):
            return False
        from .kernels import k3_fp8w_lowm_mxfp8

        m, k = x.shape
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(m, self.out_features, k)
        except ValueError:
            return False
        return k % 32 == 0

    def forward_gated_mxfp8(
        self, x: torch.Tensor, gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`forward_gated` ``(x, gate)``, bit for bit, given the MXFP8 form ``(xq, xs)`` of
        ``bf16(x * gate)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, the gated
        low-M GEMM's own quantization): the MXFP8 low-M GEMM, with the all-reduce push epilogue
        where :attr:`next_ar_push` is set, so the activation path after its
        ``griddepcontrol.wait`` is two TMA loads instead of the BF16 loads, the gate multiply and
        the quantization."""
        from .kernels import k3_fp8w_lowm_mxfp8

        w_scale = self.packed_weight_scale
        if w_scale is None or not k3_fp8w_lowm_mxfp8.can_run_mxfp8(xq, xs, self.weight, w_scale):
            return self.forward_gated(x, gate)
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            k3_fp8w_lowm_mxfp8.run_k3_fp8w_lowm_gemm_mxfp8_ar_push(
                xq, xs, self.weight, w_scale, ar_push.workspace, ar_push.one_shot
            )
            ar_push.pushed = True
            return ar_push.output
        out = self._take_next_output(x.shape[0])
        if out is None:
            out = torch.empty(
                (x.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        k3_fp8w_lowm_mxfp8.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, self.weight, w_scale, out)
        return out

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
        push, self.next_ar_push = self.next_ar_push, None
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        rows = activation.reshape(-1, activation.shape[-1])
        tactic = self._mid_prequantized_push_tactic(rows, activation_scale, push)
        if tactic is not None:
            # TRTLLM_K3_MID_AR_PUSH: the mid-M GEMM pushes into the o_proj all-reduce.
            from .kernels import k3_fp8w_lowm_gemm

            k3_fp8w_lowm_gemm.run_mid_prequantized_ar_push(
                rows,
                activation_scale.reshape(-1),
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
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
    converted = _Fp8BlockScaleWeightReadLinear.from_linear(child)
    converted.k3_latent_fused = attr == "routed_expert_down_proj" and _K3_LD_FUSED != "0"
    setattr(parent, attr, converted)
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


def _load_packed_mxfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    del get_optional_tensor  # this layout stores no optional per-expert tensor
    backend.quant_method.load_packed_mxfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        w1_weight=get_tensor(f"{base}.{expert_idx}.w1.weight_packed"),
        w1_weight_scale=get_tensor(f"{base}.{expert_idx}.w1.weight_scale"),
        w2_weight=get_tensor(f"{base}.{expert_idx}.w2.weight_packed"),
        w2_weight_scale=get_tensor(f"{base}.{expert_idx}.w2.weight_scale"),
        w3_weight=get_tensor(f"{base}.{expert_idx}.w3.weight_packed"),
        w3_weight_scale=get_tensor(f"{base}.{expert_idx}.w3.weight_scale"),
    )


def _load_nvfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    # Only an NVFP4_AWQ / NVFP4_ARC calibration writes a pre-quant scale; a
    # plain NVFP4 checkpoint has none and the pair stays None, which leaves
    # ``fc31_act_scale`` unset exactly as before. Dropping it for a checkpoint
    # that HAS one would be silently wrong -- the activation would be quantized
    # unsmoothed against an input scale calibrated on the smoothed one.
    pre_quant_scales = {
        f"{w}_pre_quant_scale": get_optional_tensor(f"{base}.{expert_idx}.{w}.pre_quant_scale")
        for w in ("w1", "w3")
    }
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
        **pre_quant_scales,
    )


def _k3_fc12_hooks(backend) -> bool:
    """Whether a routed-expert backend has the FC12 leaf hooks: the pre-zeroed output and
    counters (``new_zeroed_output`` / ``zeroed_output``,
    TRTLLM_K3_FC12_PREZERO and the ULL front), the ULL experts' token-indexed tables
    (``ull_moeblk``) and slot push (``ull_slot_push``, TRTLLM_K3_ULL_TAIL1).

    ``CuteDslFc12FusedMoE`` (``moe/fused_moe/cutedsl/fused_fc12.py``) has them; on a backend
    without them those switches stay inert (it zeroes its own output and builds its own tables).
    """
    return callable(getattr(backend, "new_zeroed_output", None))


def _quantizes_input_with_fp4_quantize(backend) -> bool:
    """Whether a routed-expert backend NVFP4-quantizes a BF16 input with ``fp4_quantize`` itself
    and takes an unswizzled ``Fp4QuantizedTensor`` of the same bytes in its place: the CuteDSL
    NVFP4 leaves (grouped GEMM and FC12), which bind ``quantize_input_nvfp4`` as
    ``quantize_input``, without a pre-quant activation scale."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.kernel_inputs import quantize_input_nvfp4

    return (
        getattr(type(backend), "quantize_input", None) is quantize_input_nvfp4
        and getattr(backend, "fc31_act_scale", None) is None
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


class _RpTailOperands(NamedTuple):
    """One rank's operands of the row-parallel MoE tail GEMM (``TRTLLM_K3_RP_TAIL``): K is this
    rank's ``k1`` latent columns from column ``k0`` (``k1 / 128`` whole blocks of latent_up, whose
    FP8 codes the GEMM reads in place); the GEMM's push epilogue adds the shared expert's BF16
    partial."""

    k0: int
    k1: int
    # DeepGEMM-packed UE8M0 scale of the [hidden, k1] tail weight.
    scale: torch.Tensor


class _CpTailOperands(NamedTuple):
    """One rank's operands of the column-parallel MoE tail (``TRTLLM_K3_ULL2_COLL_CP_TAIL``): the
    ``n`` output rows of latent_up from row ``c0`` (whole 128-row blocks; their FP8 codes read in
    place) and the DeepGEMM-packed UE8M0 scale of that ``[n, latent]`` slice."""

    c0: int
    n: int
    weight: torch.Tensor
    scale: torch.Tensor


def _rp_tail_latent_slice(latent_size: int, world: int, rank: int) -> Tuple[int, int]:
    """``(first column, columns)`` of ``rank``'s 128-column blocks of the latent: whole blocks,
    as even as possible (28 blocks over 8 ranks: 4 on ranks 0-3, 3 on ranks 4-7)."""
    blocks = latent_size // 128
    base, extra = divmod(blocks, world)
    first = rank * base + min(rank, extra)
    return first * 128, (base + (rank < extra)) * 128


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
    # TRTLLM_K3_ULL2_COLL_CP_TAIL: the column-parallel tail's workspace and this rank's operands.
    _cp_ws: Optional[Any] = None
    _cp_latent_ws: Optional[Any] = None
    _cp_tail: Optional[_CpTailOperands] = None
    # TRTLLM_K3_ULL_TAIL1: latent_up's UE8M0 block exponents [HIDDEN / 128, LATENT / 128] (None:
    # off).
    _tail1_wexp: Optional[torch.Tensor] = None
    # TRTLLM_K3_ULL_SHARED_PERSIST: the shared expert's weights in k3_shared_persist's stream order,
    # its exchange workspace and SiTU's (beta, linear beta) (None: off).
    _persist_frags: Optional[torch.Tensor] = None
    _persist_ws: Optional[Any] = None
    _persist_act: Optional[Tuple[float, Optional[float]]] = None
    # TRTLLM_K3_ROUTER_EXACT: whether this layer's router logits come from k3_router_exact (set
    # per layer in ``__init__``; off unless the switch selects the layer).
    _router_exact = False
    # TRTLLM_K3_FC12_PREZERO: the stream that zeroes the routed output and FC12's counters while
    # the routing runs, and the event recorded there after the zeroing (``None``: off).
    fc12_prezero_stream: Optional[torch.cuda.Stream] = None
    moe_prezero_event: Optional[torch.cuda.Event] = None
    # TRTLLM_K3_RP_TAIL: this rank's tail GEMM operands, built after loading (``None``: off or not
    # applicable to this layer).
    _rp_tail: Optional[_RpTailOperands] = None
    # TRTLLM_K3_ULL_FRONT: the front's MNNVL workspace of the routed all-reduce group (built at
    # construction) and this layer's front operands on this rank (built after loading).
    _ull_ws = None
    _ull_plan = None

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
        # 3-run bisect), and the bs1-latency win is irrelevant
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
        routing_recorder = get_routing_recorder(
            rank=model_config.mapping.rank,
            num_layers=cfg.num_hidden_layers,
            num_experts=self.num_experts,
            first_layer=cfg.first_k_dense_replace,
        )
        if routing_recorder is not None and isinstance(
            self.routed_experts.routing_method, _KimiK3RoutingMethod
        ):
            self.routed_experts.routing_method.observer = functools.partial(
                routing_recorder.record, layer_idx
            )

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
        if _K3_SHARED_SM > 0:
            shared_forward = self.shared_experts.forward

            def _shared_forward_confined(*args, **kwargs):
                with _k3_shared_sm_scope():
                    return shared_forward(*args, **kwargs)

            self.shared_experts.forward = _shared_forward_confined
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # TRTLLM_K3_SHARED_TAIL_PRIO: the shared expert's activation / down / all-reduce stream and
        # the event (after gate_up) it forks from.
        self.shared_tail_stream = _k3_shared_tail_stream()
        self.moe_shared_act_event = (
            torch.cuda.Event() if self.shared_tail_stream is not None else None
        )
        self.moe_routed_event = torch.cuda.Event() if self._defer_shared_allreduce else None
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if routed_moe_model_config.moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        if (
            _K3_FC12_PREZERO
            and has_latent_stream
            and routed_moe_model_config.moe_backend == "CUTEDSL_FC12"
        ):
            self.fc12_prezero_stream = _k3_fc12_prezero_stream()
            self.moe_prezero_event = torch.cuda.Event()
        # TRTLLM_K3_ROUTE_TAIL_ORDER: shared GEMMs (and latent chain) done.
        self.moe_shared_gemm_event = torch.cuda.Event() if has_latent_stream else None
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
        if _K3_LATENT_AR_OWN_WS and self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )
        # TRTLLM_K3_ULL_FRONT: recorded once the front zeroed FC12's output and counters.
        self.moe_ull_zero_event = torch.cuda.Event() if _K3_ULL_FRONT else None
        if _K3_ULL_FRONT and self._reduce_routed_output:
            from .kernels import k3_ull_front

            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and k3_ull_front.supports_world(mnnvl.mapping.tp_size):
                # Collective over the all-reduce group: every rank builds it here.
                self._ull_ws = k3_ull_front.ull_front_workspace(mnnvl.mapping)
        if _K3_ULL2_COLL_CP_TAIL and _K3_RP_TAIL and self._reduce_routed_output:
            # TRTLLM_K3_ULL2_COLL_CP_TAIL: the column-parallel tail's workspaces (collective).
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                world = mnnvl.mapping.tp_size
                row_bytes = self.moe_hidden_size * world * 2
                rows = min(_K3_RP_TAIL_MAX_M, self._cp_latent_oneshot_bytes() // row_bytes)
                if rows >= 1:
                    self._cp_ws = _k3_cp_tail_workspace(
                        mnnvl.mapping, self.hidden_size, _K3_RP_TAIL_MAX_M
                    )
                    self._cp_latent_ws = _k3_cp_tail_workspace(
                        mnnvl.mapping, self.moe_hidden_size, rows, use="latent"
                    )

    def _lu_mid_prequant_plan(self, latent: torch.Tensor):
        """``TRTLLM_K3_LU_MID_MXFP8``: ``(tactic, sf_blocks)`` of the mid-M path that
        latent_up's :meth:`_Fp8BlockScaleWeightReadLinear.forward` runs on the BF16 rows ``latent``
        (SM107: ``trtllm::k3_fp8w_lowm_gemm`` on ``lowm_weight_scale`` above the low-M cut-over),
        or ``None`` where it runs another GEMM."""
        up = self.routed_expert_up_proj
        if (
            not _K3_FP8W_LOWM_GEMM
            or not isinstance(up, _Fp8BlockScaleWeightReadLinear)
            or up.lowm_weight_scale is None
            or up.next_ar_push is not None
            or torch.compiler.is_compiling()
        ):
            return None
        if not up._lowm_gemm_applies(latent, up.lowm_weight_scale):
            # TRTLLM_K3_LU_MID_MAX_M: above the per-shape gate (only M fails the check; one row
            # stands in for the rest of it), up to that many rows.
            if latent.shape[0] > _K3_LU_MID_MAX_M or not up._lowm_gemm_applies(
                latent[:1], up.lowm_weight_scale
            ):
                return None
        from .kernels import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.mid_prequant_plan(latent.shape[0], up.out_features, up.in_features)

    def _latent_up_mid_prequantized(
        self, routed_latent: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, tactic
    ) -> torch.Tensor:
        """latent_up on the MXFP8 form ``(xq, xs)`` of ``routed_latent`` in the mid-M GEMM's
        layout (``TRTLLM_K3_LU_MID_MXFP8``): what :meth:`_routed_projection` computes, bit for
        bit, without its quantization kernel."""
        from .kernels import k3_fp8w_lowm_gemm

        up = self.routed_expert_up_proj
        m = xq.shape[0]
        out = up._take_next_output(m)
        if out is None:
            out = torch.empty((m, up.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_lowm_gemm.run_mid_prequantized(xq, xs, up.weight, up.lowm_weight_scale, out, tactic)
        return out.reshape(routed_latent.shape[:-1] + (up.out_features,))

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits; ``TRTLLM_K3_ROUTER_EXACT`` serves ``M <= 16`` BF16 tokens with
        ``k3_router_exact`` (bit-identical to the ``dsv3_router_gemm_op`` it replaces) in the
        layers it selects."""
        if self._router_exact and not torch.compiler.is_compiling():
            from .kernels import k3_router_exact

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
            and not getattr(self.shared_experts.down_proj, "use_fused_gemm_allreduce", False)
            and do_multi_stream()
            and not torch.compiler.is_compiling()
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
        """``TRTLLM_K3_ROUTE_TAIL_ORDER``: issue the shared expert's GEMMs (without its
        all-reduce) on the shared stream after ``fork_event``; ``moe_shared_gemm_event`` then marks
        them and the latent chain done, which is what the routed experts wait for. Returns the
        partial."""
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
        self.prepare_rp_tail()
        self.prepare_ull_front()

    def cache_derived_state(self) -> None:
        self.prepare_rp_tail()
        self.prepare_ull_front()

    def _ull_front_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the sharded MoE front (``TRTLLM_K3_ULL_FRONT``), or
        ``None``."""
        if self._ull_ws is None:
            return "no MNNVL routed all-reduce over a supported TP group"
        moe = self.routed_experts
        backend = moe.backend
        if not isinstance(moe.routing_method, _KimiK3RoutingMethod) or moe.use_dp:
            return "the routing is not the Kimi-K3 routing method of a TP group"
        if (
            getattr(backend, "slot_start", None) is None
            or getattr(backend, "expert_size_per_partition", None) is None
            or getattr(backend, "num_slots", None) != self.gate.num_experts
        ):
            return "the routed experts have no static contiguous expert slice"
        if (
            moe.comm is not None
            or moe.apply_router_weight_on_input
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "scaling_vector_size", None) != 16
            or getattr(backend, "fc31_input_scale", None) is None
        ):
            return "the routed experts do not take an NVFP4 input with a global scale"
        down = self.routed_expert_down_proj
        if (
            not isinstance(down, _Fp8BlockScaleWeightReadLinear)
            or down._fused_projection is not None
        ):
            return "latent_down is not an FP8 weight-read projection"
        if self.gate.weight.dtype != torch.bfloat16:
            return "the router weight is not BF16"
        return None

    def prepare_ull_front(self) -> bool:
        """``TRTLLM_K3_ULL_FRONT``: build this layer's front operands on this rank (views of the
        router and latent_down weights, the packed scale of the rank's latent_down tiles). Rebuilt
        on every call (post-load), before CUDA graph capture."""
        self._ull_plan = None
        if not _K3_ULL_FRONT:
            return False
        reason = self._ull_front_unsupported()
        if reason is not None:
            logger.info_once(
                f"TRTLLM_K3_ULL_FRONT: MoE layers keep the replicated front: {reason}",
                key=f"k3_ull_front_off_{reason}",
            )
            return False
        from .kernels import k3_ull_front

        down = self.routed_expert_down_proj
        down.transform_weights()
        backend = self.routed_experts.backend
        ws = self._ull_ws
        with torch.no_grad():
            self._ull_plan = k3_ull_front.make_plan(
                down.weight,
                down.weight_scale,
                self.gate.weight,
                self.gate.e_score_correction_bias,
                float(self.gate.routed_scaling_factor),
                backend.fc31_input_scale,
                ws.world,
                ws.rank,
                backend.slot_start,
                backend.expert_size_per_partition,
            )
        logger.info_once(
            f"TRTLLM_K3_ULL_FRONT: sharded MoE front over {ws.world} ranks, routed experts "
            f"[{backend.slot_start}, {backend.slot_start + backend.expert_size_per_partition}) "
            f"on rank {ws.rank}",
            key="k3_ull_front_on",
        )
        return True

    def _ull_front_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this forward runs the sharded MoE front."""
        plan = self._ull_plan
        if plan is None or torch.compiler.is_compiling():
            return False
        if not 1 <= hidden_states.shape[0] <= min(_K3_ULL_FRONT_MAX_M, 32):
            return False
        from .kernels import k3_ull_front

        x = hidden_states.reshape(-1, self.hidden_size)
        if not k3_ull_front.can_run(x, plan):
            return False
        # The kernels compile on their first call, which must not fall into a CUDA graph capture.
        return not torch.cuda.is_current_stream_capturing() or k3_ull_front.is_compiled(
            x.shape[0], plan, self._ull_moeblk(x.shape[0])
        )

    def _ull_moeblk(self, m: int) -> bool:
        """``TRTLLM_K3_ULL_MOEBLK``: whether the routed experts of this ``m``-row forward run the
        ULL experts kernel with the token-indexed layout, so the front's route kernel only
        publishes the top-k and the experts kernel builds its tables itself."""
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE

        from .moe.k3_ull_experts import ull_moeblk_applies

        backend = getattr(self.routed_experts, "backend", None)
        return (
            isinstance(backend, CuteDslFc12FusedMoE)
            and _k3_fc12_hooks(backend)
            and ull_moeblk_applies(backend, m)
        )

    def ull_front_releases_early(self, like: torch.Tensor) -> bool:
        """``TRTLLM_K3_ULL_FRONT_EARLY``: whether the mix that produces this forward's input may
        release the front's head at kernel entry, i.e. whether the next forward on rows shaped like
        ``like`` (a contiguous BF16 ``[M, hidden]`` tensor, e.g. the prefix sum the mix updates)
        runs the sharded front, whose head is then the next launch on the stream. Conservative:
        ``False`` where ``like`` cannot stand for the mix's output."""
        if not _K3_ULL_FRONT_EARLY or like.dim() != 2 or not like.is_contiguous():
            return False
        return self._ull_front_applies(like)

    def _ull_front(self, hidden_states: torch.Tensor, after_head=None):
        """Run the sharded MoE front: hands the top-k (with this rank's ``moe_sort`` tables) to the
        routing method and returns ``(router_logits placeholder, NVFP4 routed input, zeroed FC12
        output or None)``; ``after_head`` runs between the front's two launches."""
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE

        from .kernels import k3_ull_front

        plan = self._ull_plan
        x = hidden_states.reshape(-1, self.hidden_size)
        if not x.is_contiguous():
            x = x.contiguous()
        moe = self.routed_experts
        backend = moe.backend
        # The global scale can be replaced after the plan was built (calibration, reload).
        plan.global_sf = backend.fc31_input_scale.reshape(-1)[:1]
        zeroed = None
        if isinstance(backend, CuteDslFc12FusedMoE) and _k3_fc12_hooks(backend):
            zeroed = backend.new_zeroed_output(
                x.shape[0], self.top_k, x.device, self.moe_ull_zero_event
            )
        # TRTLLM_K3_ULL_MOEBLK: the route kernel publishes each token's top-k into the experts
        # kernel's route word (no moe_sort tables); the experts build their tables from it.
        moeblk = self._ull_moeblk(x.shape[0])
        publish = 0
        if moeblk:
            from .kernels.k3_moe_ull import CTR_ROUTE
            from .moe.k3_ull_experts import K3UllExpertsRunner

            counters = getattr(backend, "_ull_counters", None)
            if counters is None:
                counters = K3UllExpertsRunner.new_counters(x.device)
                backend._ull_counters = counters
            publish = counters.data_ptr() + CTR_ROUTE * 4
        routing, codes, scales = k3_ull_front.run_k3_ull_front(
            x,
            plan,
            self._ull_ws,
            zero=zeroed.buffer if zeroed is not None else None,
            after_head=after_head,
            publish=publish,
        )
        if moeblk:
            backend.ull_moeblk = (routing[0], plan.local_offset)
        if zeroed is not None:
            # Zeroed by the route kernel on this stream.
            zeroed.ready.record()
        target = (plan.local_offset, plan.local_num)
        moe.routing_method.precomputed = self._attach_route(routing, target)
        # The routing method returns the precomputed top-k; the logits only carry the shape.
        logits = torch.empty(
            (x.shape[0], self.gate.num_experts), dtype=torch.float32, device=x.device
        )
        return logits, Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False), zeroed

    def _ull_fork_shared(
        self, identity: torch.Tensor, persist: bool, rows: int
    ) -> Tuple[Optional[Callable[[], None]], Optional[Callable[[], None]], List[torch.Tensor]]:
        """Fork the shared expert's GEMMs for a ULL front on ``rows`` rows where ``_ull_shared_fork`` puts it.

        Returns ``(after_head, after_front, forked)``: the callback the front runs between its head and route
        launches, the one the caller runs once the front is issued (``None``: nothing to run) and the list that
        holds the shared partial once the GEMMs are issued.

        With ``TRTLLM_K3_ULL_FRONT_EARLY`` the shared GEMMs are captured after the whole front, whichever kernel
        they depend on: each early-releasing kernel's first dependent in the CUDA graph then is the one its
        trigger is meant for (A_mid's the head, the head's the route kernel), and the executor keeps that one on
        the releasing kernel's stream. A shared gate_up captured before it would take that place whenever no
        other edge pins it to the shared stream, and the intended dependent would wait for the releasing kernel
        to finish.
        """
        forked: List[torch.Tensor] = []

        def run_shared():
            forked.append(self._ull_shared_first(identity, persist))

        fork = _ull_shared_fork(self._shared_gemms_can_lead(), rows)
        if fork == "start":
            self.moe_main_event.record()
            run_shared()
            return None, None, forked
        if fork == "head":
            return self.moe_main_event.record, run_shared, forked
        if fork == "after_head":
            self.moe_main_event.record()
            return None, run_shared, forked
        return None, None, forked

    def _ull_shared_first(self, identity: torch.Tensor, persist: bool = False) -> torch.Tensor:
        """The shared expert's GEMMs on the shared stream from ``moe_main_event`` (recorded before
        the sharded front), without its all-reduce; ``moe_shared_gemm_event`` marks them done.
        ``persist`` (``TRTLLM_K3_ULL_SHARED_PERSIST``): the persistent kernel instead, which
        pushes the partial itself (returns ``_SHARED_PUSHED``)."""
        with torch.cuda.stream(self.shared_expert_stream):
            self.moe_main_event.wait()
            if persist:
                self._run_shared_persist(identity)
                partial = _SHARED_PUSHED
            else:
                partial = self.shared_experts(
                    identity, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
                )
            self.moe_shared_gemm_event.record()
        return partial

    def _rp_tail_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the row-parallel MoE tail, or ``None``."""
        if not self._reduce_routed_output or self._defer_shared_allreduce:
            return "the routed latent is not all-reduced by the model"
        moe_all_reduce = self.routed_experts.all_reduce
        if getattr(moe_all_reduce, "mnnvl_allreduce", None) is None:
            return "the routed all-reduce is not the MNNVL all-reduce"
        up = self.routed_expert_up_proj
        if not isinstance(up, _Fp8BlockScaleWeightReadLinear) or up._fused_projection is not None:
            return (
                "latent_up is not an FP8 weight-read projection (KIMI_K3_FP8_WEIGHT_READ_MOE_MLP)"
            )
        mlp = self.shared_experts
        if (
            not isinstance(mlp, GatedMLP)
            or mlp._can_fuse_gate_up_swiglu_fp4out()
            or mlp._can_fuse_gate_up_swiglu()
            or mlp._can_fuse_swiglu_fp8_quant()
        ):
            return "the shared expert does not run the unfused gate_up -> activation -> down path"
        if not self._shared_needs_all_reduce():
            return "the shared expert is not row-parallel"
        down = mlp.down_proj
        if getattr(down, "bias", None) is not None:
            return "the shared down projection has a bias"
        world = moe_all_reduce.mapping.tp_size
        shard = down.weight.shape[1]
        if shard * world != self._shared_intermediate:
            return f"the shared down shard ({shard}) is not a 1/{world} shard"
        if (
            tuple(up.weight.shape) != (self.hidden_size, self.moe_hidden_size)
            or self.moe_hidden_size % 128
            or self.hidden_size % 128
            or self.moe_hidden_size // 128 < world
        ):
            return "the latent is not split into whole 128-column blocks over the ranks"
        if tuple(down.weight.shape) != (self.hidden_size, shard):
            return "unexpected shared down weight shape"
        return None

    def prepare_rp_tail(self) -> bool:
        """``TRTLLM_K3_RP_TAIL``: build this rank's tail operands (``_RpTailOperands``): the packed
        UE8M0 scale of this rank's latent_up blocks (whose FP8 codes stay where they are). Rebuilt
        on every call (post-load), before CUDA graph capture."""
        self._rp_tail = None
        if not _K3_RP_TAIL:
            return False
        reason = self._rp_tail_unsupported()
        if reason is not None:
            logger.info_once(
                f"TRTLLM_K3_RP_TAIL: MoE layers keep the replicated tail: {reason}",
                key=f"k3_rp_tail_off_{reason}",
            )
            return False
        from .kernels import k3_fp8w_lowm_gemm

        up = self.routed_expert_up_proj
        up.transform_weights()
        down = self.shared_experts.down_proj
        mapping = self.routed_experts.all_reduce.mapping
        k0, k1 = _rp_tail_latent_slice(self.moe_hidden_size, mapping.tp_size, mapping.tp_rank)
        hidden, shard = down.weight.shape
        with torch.no_grad():
            exponents = k3_fp8w_lowm_gemm.block_exponents_of(
                up.weight_scale, hidden, self.moe_hidden_size
            )[:, k0 // 128 : (k0 + k1) // 128]
            scale = k3_fp8w_lowm_gemm.pack_ue8m0_exponents(exponents.contiguous(), hidden, k1)
        self._rp_tail = _RpTailOperands(k0, k1, scale)
        self._cp_tail = None
        if self._cp_ws is not None:
            # TRTLLM_K3_ULL2_COLL_CP_TAIL: this rank's whole 128-row blocks of latent_up's output.
            c0, n = _rp_tail_latent_slice(hidden, mapping.tp_size, mapping.tp_rank)
            with torch.no_grad():
                cp_exps = k3_fp8w_lowm_gemm.block_exponents_of(
                    up.weight_scale, hidden, self.moe_hidden_size
                )[c0 // 128 : (c0 + n) // 128]
                cp_scale = k3_fp8w_lowm_gemm.pack_ue8m0_exponents(
                    cp_exps.contiguous(), n, self.moe_hidden_size
                )
            self._cp_tail = _CpTailOperands(c0, n, up.weight[c0 : c0 + n], cp_scale)
            logger.info_once(
                f"TRTLLM_K3_ULL2_COLL_CP_TAIL: column-parallel MoE tail, latent_up rows "
                f"[{c0}, {c0 + n}) (rank {mapping.tp_rank} of {mapping.tp_size})",
                key=f"k3_cp_tail_on_{n}",
            )
            self._prepare_tail1(up, hidden)
            self._prepare_shared_persist()
        logger.info_once(
            f"TRTLLM_K3_RP_TAIL: row-parallel MoE tail, K = latent [{k0}, {k0 + k1}), BF16 shared "
            f"partial added (rank {mapping.tp_rank} of {mapping.tp_size})",
            key=f"k3_rp_tail_on_{k1}_{shard}",
        )
        return True

    def _prepare_tail1(self, up, hidden: int) -> None:
        """``TRTLLM_K3_ULL_TAIL1``: latent_up's block exponents for k3_moe_tail1 (the FP8 codes
        are read in place)."""
        self._tail1_wexp = None
        if not _K3_ULL_TAIL1 or self._cp_latent_ws is None:
            return
        from .kernels import k3_fp8w_lowm_gemm, k3_moe_tail1

        weight = up.weight
        if (
            weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (k3_moe_tail1.HIDDEN, k3_moe_tail1.LATENT)
            or hidden != k3_moe_tail1.HIDDEN
            or self.moe_hidden_size != k3_moe_tail1.LATENT
            or not weight.is_contiguous()
        ):
            logger.info_once(
                "TRTLLM_K3_ULL_TAIL1: latent_up is not the [7168, 3584] FP8 weight, tail unchanged",
                key="k3_tail1_off_weight",
            )
            return
        with torch.no_grad():
            wexp = k3_fp8w_lowm_gemm.block_exponents_of(
                up.weight_scale, hidden, self.moe_hidden_size
            ).contiguous()
        if int(wexp.min().item()) < 1 or int(wexp.max().item()) > 254:
            logger.info_once(
                "TRTLLM_K3_ULL_TAIL1: latent_up block scale outside the kernel's range, tail "
                "unchanged",
                key="k3_tail1_off_scale",
            )
            return
        self._tail1_wexp = wexp
        world, rank = self._cp_latent_ws.world, self._cp_latent_ws.rank
        c0, n = k3_moe_tail1.rank_rows(world, rank)
        logger.info_once(
            f"TRTLLM_K3_ULL_TAIL1: one-kernel MoE tail up to {k3_moe_tail1.MAX_M} rows, latent_up "
            f"rows [{c0}, {c0 + n}) (rank {rank} of {world})",
            key=f"k3_tail1_on_{n}",
        )

    def _prepare_shared_persist(self) -> None:
        """``TRTLLM_K3_ULL_SHARED_PERSIST``: this rank's shared expert weights in
        k3_shared_persist's stream order and its workspace."""
        self._persist_frags = None
        if _K3_ULL_SHARED_PERSIST <= 0 or self._cp_ws is None:
            return
        from tensorrt_llm._torch.modules.situ import SituAndMul

        from .kernels import k3_shared_persist

        ws = self._cp_ws
        mlp = self.shared_experts
        sms = _K3_ULL_SHARED_PERSIST
        gate_up = getattr(getattr(mlp, "gate_up_proj", None), "weight", None)
        down = getattr(getattr(mlp, "down_proj", None), "weight", None)
        inter = k3_shared_persist.INTER // ws.world
        reason = None
        if not isinstance(getattr(mlp, "activation", None), SituAndMul):
            reason = "the shared expert's activation is not SiTU"
        elif (
            gate_up is None
            or down is None
            or gate_up.dtype != torch.bfloat16
            or down.dtype != torch.bfloat16
            or tuple(gate_up.shape) != (2 * inter, k3_shared_persist.HIDDEN)
            or tuple(down.shape) != (k3_shared_persist.HIDDEN, inter)
        ):
            reason = "the shared expert is not the BF16 [2 I, 7168] / [7168, I] shard"
        elif mlp.down_proj.tp_size != ws.world or mlp.down_proj.tp_rank != ws.rank:
            reason = "the shared expert's TP group is not the tail's"
        elif not k3_shared_persist.can_run(k3_shared_persist.MAX_M, ws.world, sms):
            reason = f"{sms} CTAs do not split the {inter}-column shard"
        if reason is not None:
            logger.info_once(
                f"TRTLLM_K3_ULL_SHARED_PERSIST: shared expert unchanged: {reason}",
                key=f"k3_persist_off_{reason}",
            )
            return
        with torch.no_grad():
            self._persist_frags = k3_shared_persist.persist_fragments(
                gate_up.data.contiguous(), down.data.contiguous(), sms
            )
        self._persist_ws = k3_shared_persist.PersistWorkspace(ws.world, gate_up.device)
        self._persist_act = (float(mlp.activation.beta), mlp.activation.linear_beta)
        logger.info_once(
            f"TRTLLM_K3_ULL_SHARED_PERSIST: the shared expert on {sms} CTAs beside the routed "
            f"experts up to {k3_shared_persist.MAX_M} rows (rank {ws.rank} of {ws.world})",
            key=f"k3_persist_on_{sms}",
        )

    def _persist_applies(self, num_tokens: int, target) -> bool:
        """Whether this forward's shared expert is the persistent kernel
        (``TRTLLM_K3_ULL_SHARED_PERSIST``; the column-parallel tail)."""
        if (
            self._persist_frags is None
            or target is None
            or self.shared_expert_stream is None
            or not do_multi_stream()
            or torch.compiler.is_compiling()
            or not _K3_ULL2_COLL_CP_TAIL
            or not self._cp_tail_applies(num_tokens, target)
        ):
            return False
        from .kernels import k3_shared_persist

        return k3_shared_persist.can_run(num_tokens, self._cp_ws.world, _K3_ULL_SHARED_PERSIST)

    def _run_shared_persist(self, identity: torch.Tensor) -> None:
        """The persistent shared expert of ``identity`` on the current (shared) stream, pushing
        into the tail call's slots."""
        from .kernels import k3_shared_persist

        ws = self._cp_ws
        x = identity if identity.is_contiguous() else identity.contiguous()
        beta, linear_beta = self._persist_act
        k3_shared_persist.run_k3_shared_persist(
            x=x,
            frags=self._persist_frags,
            world=ws.world,
            rank=ws.rank,
            sms=_K3_ULL_SHARED_PERSIST,
            ws=self._persist_ws,
            out_base=ws.mc_base,
            out_flags=ws.flags,
            beta=beta,
            linear_beta=linear_beta,
        )

    def _tail1_applies(self, num_tokens: int) -> bool:
        """Whether this forward's MoE tail runs k3_moe_tail1 (``TRTLLM_K3_ULL_TAIL1``; the caller
        still needs the routed experts to have pushed, ``ull_slot_push_used``)."""
        if self._tail1_wexp is None or self._cp_latent_ws is None or torch.compiler.is_compiling():
            return False
        if not _k3_fc12_hooks(getattr(self.routed_experts, "backend", None)):
            return False
        from .kernels import k3_moe_tail1

        return k3_moe_tail1.can_run(num_tokens, self._cp_latent_ws.world)

    def _rp_tail_target(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], rp_tail_ok: bool
    ):
        """The all-reduce push target of this forward's row-parallel MoE tail
        (``TRTLLM_K3_RP_TAIL``), or ``None`` where the forward keeps the replicated tail: a
        generation-only batch of at most ``TRTLLM_K3_RP_TAIL_MAX_M`` BF16 tokens with the fused
        residual update, served by the low-M GEMM's push epilogue. The consumer finishes the
        call: one-shot transport up to TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot
        above."""
        rp = self._rp_tail
        num_tokens = hidden_states.shape[0]
        if (
            rp is None
            or not rp_tail_ok
            or residual is None
            or hidden_states.dtype != torch.bfloat16
            or not 1 <= num_tokens <= _K3_RP_TAIL_MAX_M
            or torch.compiler.is_compiling()
        ):
            return None
        from .kernels import k3_fp8w_lowm_gemm

        if k3_fp8w_lowm_gemm.split_source_tactic(num_tokens, self.hidden_size, rp.k1) is None:
            return None
        return lamport_push_target(
            self.routed_experts.all_reduce,
            num_tokens,
            self.hidden_size,
            deferred=True,
            force=True,
        )

    def _rp_shared_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The shared expert's part of the row-parallel tail on this rank: its row-parallel BF16
        down-projection partial without the all-reduce."""
        return self.shared_experts(
            hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
        )

    def _rp_tail_mx_applies(self, num_tokens: int) -> bool:
        """Whether the row-parallel tail GEMM can run as the MXFP8 low-M GEMM on the latent
        all-reduce's MXFP8 output (``TRTLLM_K3_ULL_TAIL_SANDWICH``; the tail GEMM has one source):
        the same tactic as the split-source BF16 GEMM, so the same bits."""
        from .kernels import k3_fp8w_lowm_mxfp8

        if num_tokens > k3_fp8w_lowm_mxfp8.MAX_M:
            return False
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(num_tokens, self.hidden_size, self._rp_tail.k1)
        except ValueError:
            return False
        return True

    def _rp_tail_latent_ar_early(
        self, routed_out: torch.Tensor, target, mxfp8: bool = False
    ) -> Optional[Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """``TRTLLM_K3_ULL_TAIL_SANDWICH``: the routed latent all-reduce + RMSNorm of the
        row-parallel tail by ``k3_latent_ar_norm`` (its row kernel where it serves the size, else
        the cluster kernel) on the MNNVL workspace the tail GEMM pushes into: bit-identical to the
        C++ one-shot kernel, but it releases the tail GEMM right after its own
        ``griddepcontrol.wait``, so the GEMM streams its weight slice while the all-reduce waits
        for the other ranks. Returns ``(latent, (xq, xs) with mxfp8 else None)``, or ``None``
        where it does not apply (the C++ kernel's two-shot sizes). ``mxfp8``: the kernel also
        stores the MXFP8 form of the latent, bitwise the low-M GEMM's own quantization of it."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        from .kernels import k3_latent_ar_norm

        ws = target.workspace
        latent = routed_out.reshape(-1, self.moe_hidden_size)
        if not latent.is_contiguous():
            latent = latent.contiguous()
        gamma = self.routed_expert_norm.weight
        if latent.numel() * ws.world * latent.element_size() > _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
            return None
        row = k3_latent_ar_norm.can_run_row(latent, gamma, ws.world, ws.bytes_per_buffer)
        if not (row or k3_latent_ar_norm.can_run(latent, gamma, ws.world)):
            return None
        launch = (
            k3_latent_ar_norm.run_k3_latent_ar_norm_row
            if row
            else k3_latent_ar_norm.run_k3_latent_ar_norm
        )
        out = torch.empty_like(latent)
        m, k = latent.shape
        latent_mx = None
        if mxfp8:
            latent_mx = (
                torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                torch.empty((m, k // 32), dtype=torch.uint8, device=latent.device),
            )
        launch(
            latent,
            gamma,
            float(self.routed_expert_norm.variance_epsilon),
            out,
            uc_addr=ws.uc_local,
            mc_addr=ws.mc_base,
            flags=ws.flags,
            rank=ws.rank,
            world_size=ws.world,
            early_trigger=True,
            mxfp8_out=latent_mx,
        )
        return out, latent_mx

    @staticmethod
    def _cp_latent_oneshot_bytes() -> int:
        """Largest one-shot latent all-reduce of the column-parallel tail (bytes of slots): the C++
        one-shot size."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        return _MNNVL_ONE_SHOT_THRESHOLD_BYTES

    def _cp_tail_applies(self, num_tokens: int, target) -> bool:
        """Whether this forward's MoE tail runs column-parallel (``TRTLLM_K3_ULL2_COLL_CP_TAIL``):
        the latent all-reduce runs one-shot by ``k3_latent_ar_norm``'s row kernel on the tail's
        latent workspace (its MXFP8 output feeds the tail GEMM), the replicated projection's MXFP8
        tactic serves ``num_tokens`` and the tail's workspaces hold them."""
        cp = self._cp_tail
        lws = self._cp_latent_ws
        if cp is None or lws is None or num_tokens > _K3_RP_TAIL_MAX_M:
            return False
        from .kernels import k3_fp8w_lowm_mxfp8, k3_latent_ar_norm

        if (
            lws.world not in k3_latent_ar_norm.ROW_WORLD_SIZES
            or num_tokens > k3_latent_ar_norm.ROW_MAX_M
            or self.moe_hidden_size != k3_latent_ar_norm.HIDDEN
        ):
            return False
        nbytes = num_tokens * self.moe_hidden_size * lws.world * 2
        if nbytes > self._cp_latent_oneshot_bytes() or nbytes > lws.bytes_per_buffer:
            return False
        if num_tokens > k3_fp8w_lowm_mxfp8.MAX_M:
            return False
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(num_tokens, self.hidden_size, self.moe_hidden_size)
        except ValueError:
            return False
        return True

    def _cp_tail_forward(
        self,
        routed_output: Callable[[], torch.Tensor],
        identity: torch.Tensor,
        shared_part: Optional[torch.Tensor],
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """``TRTLLM_K3_ULL2_COLL_CP_TAIL``: the routed experts, the shared expert's BF16 partial
        pushed into the tail's call on the shared stream, the latent all-reduce + RMSNorm (MXFP8
        output), then this rank's latent_up output columns multicast into the call's all-gather
        rows. Returns a ``_DeferredRpTail`` whose pending call the next pre-attention mix finishes
        (``finish_ag``), or the updated residual."""
        from .kernels import k3_lamport_ar, k3_latent_ar_norm
        from .kernels.k3_fp8w_lowm_mxfp8 import (
            default_mx_tactic,
            run_k3_fp8w_lowm_gemm_mxfp8_ag_push,
        )

        cp = self._cp_tail
        cp_ws = self._cp_ws
        num_tokens = identity.shape[0]
        # TRTLLM_K3_ULL_TAIL1: the latent push is offered to the routed backend for this call and
        # taken only where k3_moe_ull serves it (ull_slot_push_used), else the tail below is
        # unchanged.
        backend = getattr(self.routed_experts, "backend", None)
        tail1 = self._tail1_applies(num_tokens)
        if tail1:
            lws = self._cp_latent_ws
            backend.ull_slot_push = (lws.mc_base, lws.flags, lws.rank, lws.world)
            backend.ull_slot_push_used = False
            inner_routed = routed_output

            def routed_output():
                try:
                    return inner_routed()
                finally:
                    backend.ull_slot_push = None

        def push_shared(part: torch.Tensor) -> None:
            part = part.reshape(num_tokens, -1)
            if not part.is_contiguous():
                part = part.contiguous()
            k3_lamport_ar.run_lamport_push(cp_ws, part, True)

        side = (
            self.shared_expert_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )
        if shared_part is not None:
            # The shared expert ran ahead on the shared stream (moe_shared_gemm_event); the
            # persistent kernel pushed its partial itself.
            routed_out = routed_output()
            if side:
                with torch.cuda.stream(self.shared_expert_stream):
                    self.moe_shared_gemm_event.wait()
                    if shared_part is not _SHARED_PUSHED:
                        push_shared(shared_part)
                    self.moe_shared_event.record()
            else:
                self.moe_shared_gemm_event.wait()
                push_shared(shared_part)
        elif side:
            self.moe_main_event.record()
            persist = self._persist_applies(num_tokens, target)
            routed_out = routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                if persist:
                    self._run_shared_persist(identity)
                else:
                    push_shared(self._rp_shared_output(identity))
                self.moe_shared_event.record()
        else:
            routed_out = routed_output()
            push_shared(self._rp_shared_output(identity))
        if tail1 and backend.ull_slot_push_used:
            # The whole tail in one kernel (the experts pushed their latent rows), into the call's
            # all-gather rows.
            from .kernels import k3_moe_tail1

            backend.ull_slot_push_used = False
            lws = self._cp_latent_ws
            c0, _ = k3_moe_tail1.rank_rows(lws.world, lws.rank)
            k3_moe_tail1.run_k3_moe_tail1(
                m=num_tokens,
                world=lws.world,
                lat_uc=lws.uc_local,
                lat_flags=lws.flags,
                weight=self.routed_expert_up_proj.weight,
                wexp=self._tail1_wexp,
                gamma=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
                ag_base=cp_ws.mc_base,
                ag_flags=cp_ws.flags,
                c0=c0,
            )
            return self._cp_tail_finish(cp_ws, num_tokens, residual, side, defer_residual)
        if tail1:
            backend.ull_slot_push_used = False
        # The latent all-reduce + RMSNorm on the tail's latent workspace (one-shot, checked by
        # _cp_tail_applies), releasing the tail GEMM at once, with its MXFP8 output.
        latent = routed_out.reshape(-1, self.moe_hidden_size)
        if not latent.is_contiguous():
            latent = latent.contiguous()
        ws = self._cp_latent_ws
        latent_mx = (
            torch.empty(latent.shape, dtype=torch.float8_e4m3fn, device=latent.device),
            torch.empty(
                (num_tokens, self.moe_hidden_size // 32), dtype=torch.uint8, device=latent.device
            ),
        )
        k3_latent_ar_norm.run_k3_latent_ar_norm_row(
            latent,
            self.routed_expert_norm.weight,
            float(self.routed_expert_norm.variance_epsilon),
            torch.empty_like(latent),
            uc_addr=ws.uc_local,
            mc_addr=ws.mc_base,
            flags=ws.flags,
            rank=ws.rank,
            world_size=ws.world,
            early_trigger=True,
            mxfp8_out=latent_mx,
            early_state=_K3_ULL2_COLL_AR_EARLY_STATE == "clear",
        )
        # The replicated projection's tactic: every 128-row output tile is computed as there.
        run_k3_fp8w_lowm_gemm_mxfp8_ag_push(
            *latent_mx,
            cp.weight,
            cp.scale,
            cp_ws,
            cp.c0,
            self.hidden_size,
            tactic=default_mx_tactic(num_tokens, self.hidden_size, self.moe_hidden_size),
        )
        return self._cp_tail_finish(cp_ws, num_tokens, residual, side, defer_residual)

    def _cp_tail_finish(
        self, cp_ws, num_tokens: int, residual: torch.Tensor, side: bool, defer_residual: bool
    ):
        """The column-parallel tail call (all-gather rows pushed, the shared expert's slots pushed
        on the shared stream, joined here) as a ``_DeferredRpTail`` the next pre-attention mix
        finishes, or the updated residual."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        from .comm.lamport import LamportPushTarget

        if side:
            self.moe_shared_event.wait()
        # The call reproduces the shared expert's C++ all-reduce (its transport by size).
        cpp_one_shot = (
            num_tokens * self.hidden_size * cp_ws.world * 2 <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        )
        cp_target = LamportPushTarget(
            cp_ws, True, cpp_one_shot, torch.empty_like(residual.reshape(num_tokens, -1))
        )
        cp_target.pushed = True
        cp_target.deferred = True
        cp_target.ag = True
        deferred = _DeferredRpTail(residual.reshape(num_tokens, -1), LamportPending(cp_target))
        if defer_residual:
            return deferred
        return deferred.materialize().view(residual.shape)

    def _rp_tail_forward(
        self,
        routed_output: Callable[[], torch.Tensor],
        identity: torch.Tensor,
        shared_part: Optional[torch.Tensor],
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """``TRTLLM_K3_RP_TAIL``: the routed experts (``routed_output``) and the shared expert's
        part (``_rp_shared_output``; ``shared_part`` if it already ran ahead of the routed experts),
        then the latent all-reduce + RMSNorm, then this rank's tail GEMM pushing its partial into
        ``target``. Returns a ``_DeferredRpTail`` (``defer_residual``) or the updated residual."""
        if _K3_ULL2_COLL_CP_TAIL and self._cp_tail_applies(
            residual.numel() // self.hidden_size, target
        ):
            return self._cp_tail_forward(
                routed_output, identity, shared_part, residual, target, defer_residual
            )
        if shared_part is not None:
            # The shared expert ran ahead on the shared stream and the routed experts waited for
            # it (moe_shared_gemm_event).
            routed_out = routed_output()
            self.moe_shared_gemm_event.wait()
        elif (
            self.shared_expert_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        ):
            self.moe_main_event.record()
            routed_out = routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_part = self._rp_shared_output(identity)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        else:
            routed_out = routed_output()
            shared_part = self._rp_shared_output(identity)
        moe_all_reduce = self.routed_experts.all_reduce
        routed_latent = None
        latent_mx = None
        if _K3_ULL_TAIL_SANDWICH:
            num_rows = routed_out.numel() // self.moe_hidden_size
            early = self._rp_tail_latent_ar_early(
                routed_out, target, mxfp8=self._rp_tail_mx_applies(num_rows)
            )
            if early is not None:
                routed_latent, latent_mx = early
        if routed_latent is None:
            routed_latent = moe_all_reduce(
                routed_out,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.routed_expert_norm.weight,
                    eps=float(self.routed_expert_norm.variance_epsilon),
                ),
            )
        latent = routed_latent.reshape(-1, self.moe_hidden_size)
        num_tokens = latent.shape[0]
        shared_part = shared_part.reshape(num_tokens, -1)
        if not shared_part.is_contiguous():
            shared_part = shared_part.contiguous()
        rp = self._rp_tail
        lo, hi = rp.k0, rp.k0 + rp.k1
        from .kernels.k3_fp8w_lowm_gemm import run_k3_fp8w_lowm_gemm_split_ar_push

        if latent_mx is not None:
            # TRTLLM_K3_ULL_TAIL_SANDWICH: the MXFP8 GEMM on the all-reduce's MXFP8 output (its
            # slice read in place), no quantization in the tail GEMM; the same bits.
            from .kernels.k3_fp8w_lowm_mxfp8 import run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push

            run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push(
                *latent_mx,
                self.routed_expert_up_proj.weight,
                rp.scale,
                lo,
                rp.k1,
                target.workspace,
                target.one_shot,
                addend=shared_part,
            )
        else:
            run_k3_fp8w_lowm_gemm_split_ar_push(
                latent[:, lo:hi],
                self.routed_expert_up_proj.weight[:, lo:hi],
                rp.scale,
                target.workspace,
                target.one_shot,
                addend=shared_part,
            )
        target.pushed = True
        result = finish_lamport_push(moe_all_reduce, target, deferred=True)
        deferred = _DeferredRpTail(residual.reshape(num_tokens, -1), result)
        if defer_residual:
            return deferred
        return deferred.materialize().view(residual.shape)

    def _new_fc12_prezero(self, hidden_states: torch.Tensor):
        """``TRTLLM_K3_FC12_PREZERO``: the routed output and FC12 counters of this forward
        (``Fc12ZeroedOutput``), allocated on the current stream and not zeroed yet, or ``None``
        where FC12 zeroes them itself: switch off, no NVFP4 FC12 backend or external MoE
        communication (the output lives in its workspace)."""
        num_tokens = hidden_states.shape[0]
        if self.fc12_prezero_stream is None or num_tokens == 0:
            return None
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE

        backend = getattr(self.routed_experts, "backend", None)
        if (
            not isinstance(backend, CuteDslFc12FusedMoE)
            or not backend.has_nvfp4
            or getattr(self.routed_experts, "comm", None) is not None
        ):
            return None
        if not _k3_fc12_hooks(backend):
            logger.warning_once(
                "TRTLLM_K3_FC12_PREZERO is inert: the FC12 backend has no pre-zeroed output hook "
                "on this tree.",
                key="k3_fc12_prezero_no_hook",
            )
            return None
        return backend.new_zeroed_output(
            num_tokens, self.top_k, hidden_states.device, self.moe_prezero_event
        )

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
        if (
            slot_start is None
            or num_local is None
            or getattr(backend, "num_slots", None) != self.gate.num_experts
        ):
            return None
        from .kernels import k3_route_tail

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
        from .kernels import k3_route_tail

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

    @staticmethod
    def _fused_projection_applies(mode: str, x: torch.Tensor, projection: nn.Module) -> bool:
        """Whether a fused latent projection (mode "1"/"2") replaces ``projection(x)``."""
        if mode == "0" or torch.compiler.is_compiling():
            return False
        if not isinstance(projection, _Fp8BlockScaleWeightReadLinear):
            return False
        packed_scale = projection.packed_weight_scale
        if packed_scale is None or x.dtype != torch.bfloat16:
            return False
        if not _K3_LD_FUSED_MIN_M <= x.shape[0] <= _K3_LD_FUSED_MAX_M:
            return False
        from .kernels import k3_latent_fp8_gemm

        # Where the fused op itself runs the low-M GEMM (SM107), it only adds the epilogue.
        if x.shape[0] <= k3_latent_fp8_gemm.lowm_route_max_m(*projection.weight.shape):
            return True
        if (
            (mode == "1" or x.shape[0] <= _K3_LOWM_OVER_FUSED_MAX_M)
            and _K3_FP8W_LOWM_GEMM
            and _k3_lowm_gemm_can_run(x, projection.weight, packed_scale)
        ):
            return False
        return True

    def _fused_latent_input(self, hidden_states: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """NVFP4 routed-expert input straight from the hidden states, or ``None``.

        Runs the latent_down projection with the experts' input quantization fused in
        (``trtllm::k3_latent_down_fp4``) when enabled and the routed experts would quantize
        the projection with ``fp4_quantize`` themselves (CuteDSL NVFP4, no dispatch).
        """
        down = self.routed_expert_down_proj
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not self._fused_projection_applies(_K3_LD_FUSED, x, down):
            return None
        moe = self.routed_experts
        backend = moe.backend
        if (
            moe.comm is not None
            or moe.apply_router_weight_on_input
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "scaling_vector_size", None) != 16
            or not _quantizes_input_with_fp4_quantize(backend)
        ):
            return None
        max_tokens = getattr(moe, "moe_max_num_tokens", None)
        if max_tokens is not None and x.shape[0] > max_tokens:
            return None
        from .kernels import k3_latent_fp8_gemm

        global_sf = getattr(backend, "fc31_input_scale", None)
        w_scale = down.packed_weight_scale
        if not k3_latent_fp8_gemm.can_run(x, down.weight, w_scale, global_sf):
            return None
        codes, scales = torch.ops.trtllm.k3_latent_down_fp4(x, down.weight, w_scale, global_sf)
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)

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
        routed_latent = None
        latent_mxfp8 = None
        mid_plan = None
        if _K3_LATENT_AR_CUTE != "0":
            from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

            from .kernels import k3_latent_ar_norm

            world_size = workspace["world_size"]
            one_shot = latent.numel() * world_size * latent.element_size() <= (
                _K3_LATENT_AR_ONESHOT_MAX_BYTES or _MNNVL_ONE_SHOT_THRESHOLD_BYTES
            )
            row = _K3_LATENT_AR_ROW and k3_latent_ar_norm.can_run_row(
                latent, gamma, world_size, workspace["buffer_bytes"]
            )
            if one_shot and (row or k3_latent_ar_norm.can_run(latent, gamma, world_size)):
                routed_latent = torch.empty_like(latent)
                m, k = latent.shape
                if row and _K3_LU_MID_MXFP8:
                    # The MXFP8 operand of latent_up's mid-M GEMM, in its scale layout.
                    mid_plan = self._lu_mid_prequant_plan(latent)
                    if mid_plan is not None:
                        latent_mxfp8 = (
                            torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                            torch.empty(
                                (k // 128) * mid_plan[1] * 512,
                                dtype=torch.uint8,
                                device=latent.device,
                            ),
                        )
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
                    mxfp8_out=latent_mxfp8,
                    **({"sf_blocks": mid_plan[1]} if mid_plan is not None else {}),
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
        if mid_plan is not None:
            routed_out = self._latent_up_mid_prequantized(routed_latent, *latent_mxfp8, mid_plan[0])
        else:
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        self.moe_shared_event.wait()
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            from .moe.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        return routed_out + shared_out

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
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

    def _shared_tail_applies(self, num_tokens: int) -> bool:
        """Whether ``TRTLLM_K3_SHARED_TAIL_PRIO`` splits this forward's shared expert (the
        unfused gate_up -> activation -> down GatedMLP path only)."""
        mlp = self.shared_experts
        return (
            self.shared_tail_stream is not None
            and _K3_SHARED_TAIL_MIN_M <= num_tokens <= _K3_SHARED_TAIL_MAX_M
            and isinstance(mlp, GatedMLP)
            and not mlp._can_fuse_gate_up_swiglu_fp4out()
            and not mlp._can_fuse_gate_up_swiglu()
            and not mlp._can_fuse_swiglu_fp8_quant()
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )

    def _issue_router(
        self,
        hidden_states: torch.Tensor,
        gemv_route: bool,
        route_target: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        """Issue the router GEMM on the current stream; returns the router logits."""
        if gemv_route:
            return self._gemv_route(hidden_states, route_target)
        return self._router_logits(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
        rp_tail_ok: bool = False,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead where that update would be the separate BF16 add, or a
        ``_DeferredRpTail`` where the row-parallel tail runs (``TRTLLM_K3_RP_TAIL``; the caller
        allows it with ``rp_tail_ok`` for generation-only batches).
        """
        identity = hidden_states
        rp_target = self._rp_tail_target(hidden_states, residual, rp_tail_ok)
        # TRTLLM_K3_ULL_SHARED_PERSIST: the shared expert as the persistent kernel (forked with the
        # ULL front's shared fork, else in the column-parallel tail).
        persist = self._persist_applies(hidden_states.shape[0], rp_target)
        projected_input = None
        early_routed_in = None
        shared_partial = None
        prezero = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        gemv_route = False
        tail_order = False
        if self._ull_front_applies(hidden_states):
            # TRTLLM_K3_ULL_FRONT: the shared expert forks where _ull_fork_shared puts it; the sharded
            # front routes (route_target is the front's) and produces the NVFP4 routed input on this stream.
            route_target = None
            after_head, after_front, forked = self._ull_fork_shared(
                identity, persist, hidden_states.shape[0]
            )
            router_logits, projected_input, prezero = self._ull_front(
                hidden_states, after_head=after_head
            )
            if after_front is not None:
                after_front()
            if forked:
                shared_partial = forked[0]
        elif self._can_fuse_router_down(hidden_states):
            router_logits, projected_input = torch.ops.trtllm.dsv3_router_latent_gemm_op(
                hidden_states, self.gate.weight, self.routed_expert_down_proj.weight
            )
        else:
            router_issued = False
            if (
                self.latent_down_stream is not None
                and do_multi_stream()
                and not torch.compiler.is_compiling()
            ):
                # Forked ahead of the router GEMM: the latent chain needs only
                # hidden_states.
                gemv_route = route_target is not None and self._can_gemv_route(hidden_states)
                tail_order = (
                    route_target is not None
                    and not gemv_route
                    and self._tail_waits_for_shared(hidden_states.shape[0], route_target[1])
                )
                # Allocated before the fork: the zeroing stream then follows every earlier user
                # of the memory on this stream.
                prezero = self._new_fc12_prezero(hidden_states)
                self.moe_latent_fork_event.record()
                if prezero is not None:
                    with torch.cuda.stream(self.fc12_prezero_stream):
                        self.moe_latent_fork_event.wait()
                        prezero.buffer.zero_()
                        prezero.ready.record()
                if _K3_ROUTER_FIRST and hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                    # Router before latent_down in creation order (see _K3_ROUTER_FIRST).
                    router_logits = self._issue_router(hidden_states, gemv_route, route_target)
                    router_issued = True
                with torch.cuda.stream(self.latent_down_stream):
                    self.moe_latent_fork_event.wait()
                    early_routed_in = self._fused_latent_input(hidden_states)
                    if early_routed_in is None:
                        early_routed_in = self._routed_projection(
                            hidden_states, self.routed_expert_down_proj
                        )
                        if gemv_route or tail_order:
                            # Quantized here, the routed input does not wait for the routing.
                            early_routed_in = self._quantize_routed_input(early_routed_in)
                    self.moe_latent_ready_event.record()
                # No record_stream (under graph capture it holds every layer's
                # blocks until capture ends): both tensors are read on main only
                # after the join, and later work on this stream forks from main.
                if tail_order and _K3_ROUTE_TAIL_ORDER == "fork":
                    shared_partial = self._shared_gemms_first(identity, self.moe_latent_fork_event)
            if not router_issued:
                router_logits = self._issue_router(hidden_states, gemv_route, route_target)
            if tail_order and _K3_ROUTE_TAIL_ORDER == "router":
                self.moe_main_event.record()
                shared_partial = self._shared_gemms_first(identity, self.moe_main_event)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None
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
                routed_in = self._fused_latent_input(hidden_states)
                if routed_in is None:
                    routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, Fp4QuantizedTensor):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            if route_target is not None and not gemv_route:
                # Routed by the CuTe tail when the backend asks for the routing (after it
                # forks the input quantization, unless the latent chain quantized it).
                self.routed_experts.routing_method.deferred = functools.partial(
                    self._tail_route, target=route_target
                )
            if prezero is not None:
                # TRTLLM_K3_FC12_PREZERO: FC12 runs into the output zeroed beside the routing.
                self.routed_experts.backend.zeroed_output = prezero
            y = self.routed_experts(
                routed_in,
                logits,
                all_rank_num_tokens=all_rank_num_tokens,
                **moe_kwargs,
            )
            if prezero is not None:
                self.routed_experts.backend.drop_zeroed_output()
            if route_target is not None:
                self.routed_experts.routing_method.deferred = None
            if overlap_shared_allreduce:
                self.moe_routed_event.record()
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output(tail_stream: Optional[torch.cuda.Stream] = None):
            """The shared expert's (reduced) output. With ``tail_stream``, gate_up runs on the
            current stream and everything after it on ``tail_stream``, which then records
            ``moe_shared_event``."""
            final_params = (
                AllReduceParams(enable_allreduce=False) if self._defer_shared_allreduce else None
            )
            if tail_stream is None:
                return _shared_reduce(
                    self.shared_experts(identity, final_all_reduce_params=final_params)
                )
            # GatedMLP.forward's unfused path, split after gate_up (see _shared_tail_applies).
            mlp = self.shared_experts
            gate_up = mlp.gate_up_proj(identity)
            self.moe_shared_act_event.record()
            with torch.cuda.stream(tail_stream):
                self.moe_shared_act_event.wait()
                partial = mlp.down_proj(
                    mlp._apply_activation(gate_up),
                    all_reduce_params=final_params,
                    layer_idx=mlp.layer_idx,
                )
                shared_out = _shared_reduce(partial)
                self.moe_shared_event.record()
            return shared_out

        def _shared_reduce(shared_out: torch.Tensor) -> torch.Tensor:
            """Reduce the shared expert's down-projection output."""
            if overlap_shared_allreduce:
                # Wait for all routed chunks, but overlap the norm/up tail.
                self.moe_routed_event.wait()
                shared_out = self._shared_all_reduce()(shared_out)
            return shared_out

        if rp_target is not None:
            return self._rp_tail_forward(
                _routed_output, identity, shared_partial, residual, rp_target, defer_residual
            )
        # The final join includes the shared all-reduce, keeping it ordered before any outer
        # routed reduction or the next layer's collectives.
        if shared_partial is not None:
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                shared_out = shared_partial
                if self._shared_needs_all_reduce():
                    shared_out = self._shared_all_reduce()(shared_partial)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        elif (
            self._latent_ar_ws is not None
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
            tail_stream = self.shared_tail_stream if self._shared_tail_applies(num_tokens) else None
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_out = _shared_output(tail_stream)
                if tail_stream is None:
                    self.moe_shared_event.record()
            if routed_out.dtype == torch.bfloat16:
                return self._own_workspace_latent_tail(
                    routed_out, shared_out, residual, defer_residual
                )
            self.moe_shared_event.wait()
        else:
            # Shared experts depend only on the block input, so overlap their GEMMs
            # with the routed dispatch/expert/combine chain. Multi-stream engages
            # only under CUDA graphs; otherwise both branches run in order on the
            # default stream. The shared GatedMLP includes its output all-reduce on
            # the auxiliary stream (deferred past the routed MoE for fused-comm
            # MegaMoE). The final join includes the shared all-reduce, so it
            # precedes the routed all-reduce: concurrent collectives on different
            # streams can corrupt SYMM_MEM all-reduce state.
            routed_out, shared_out = maybe_execute_in_parallel(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
                disable_on_compile=True,
            )
        if self._defer_shared_allreduce and not overlap_shared_allreduce:
            shared_out = self._shared_all_reduce()(shared_out)
        if self._reduce_routed_output:
            # RMSNorm folded into the routed all-reduce (RMS_NORM fusion): the
            # MNNVL kernel normalizes straight out of the lamport buffer, so
            # the reduced row never round-trips through HBM for a norm kernel.
            # Strategies without the fusion fall back inside AllReduce.
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
            from .moe.kimi_k3_moe_residual import _fused_moe_residual_add

            return _fused_moe_residual_add(routed_out, shared_out, residual)
        return routed_out + shared_out


_KIMI_K3_FP8_KV_LAYERS_ENV = "TLLM_KIMI_K3_FP8_KV_LAYERS"


def _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo: Optional[QuantAlgo]) -> frozenset:
    """MLA layers whose KV cache falls back to FP8 under NVFP4 KV (``TLLM_KIMI_K3_FP8_KV_LAYERS``).

    Where the cache policy provides the FP8 selection, the NVFP4 default is the last two MLA
    layers and the FP4 MLA cache manager allocates those layers as FP8
    (``kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids``, read by the manager and here).
    Engine requirement: main's cache policy has no FP8 selection (only
    ``TLLM_KIMI_K3_BF16_KV_LAYERS``) and its cache manager allocates every MLA layer in the
    model-wide dtype, so switching a layer to FP8 here alone would mismatch the cache. The
    override is therefore taken only when the engine's cache policy provides the selector;
    until then every layer keeps the model-wide KV dtype and a warning names the gap.
    """
    selector = getattr(_kimi_k3_cache_policy, "get_kimi_k3_fp8_kv_layer_ids", None)
    if selector is not None:
        return selector(cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4)
    requested = os.environ.get(_KIMI_K3_FP8_KV_LAYERS_ENV)
    if (requested is None and kv_quant_algo == QuantAlgo.NVFP4) or (requested or "").strip():
        logger.warning_once(
            f"Kimi K3: FP8 KV-cache fallback layers ({_KIMI_K3_FP8_KV_LAYERS_ENV}="
            f"{requested if requested is not None else '<unset: last two MLA layers>'}) are not "
            "applied: the KV cache manager has no per-layer FP8 allocation on this tree. Every "
            "MLA layer uses the model-wide KV cache dtype.",
            key="kimi_k3_fp8_kv_layers_unsupported",
        )
    return frozenset()


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

        from .mla import KimiK3MLAAttention

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
        # Set by the decoder layer for one forward (TRTLLM_K3_AR_PUSH_AMID): a pushed o_proj
        # all-reduce may be finished in place and returned as a LamportResult.
        self.ar_result_in_place = False
        # Set by the decoder layer for one forward (TRTLLM_K3_AR_FINISH_FUSED): a pushed o_proj
        # all-reduce may be returned unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False
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
        fp8_kv_layer_ids = _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo)
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    f"{_KIMI_K3_FP8_KV_LAYERS_ENV} requires the model-wide KV cache dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
            for projection_config in attention_config.quant_config_dict.values():
                projection_config.kv_cache_quant_algo = kv_quant_algo
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
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

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        # TRTLLM_K3_AR_PUSH: o_proj stores its partial straight into the all-reduce's Lamport
        # buffers where the low-M GEMM serves it.
        ar_push = None
        if self._o_allreduce is not None and hasattr(o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                hidden_states.shape[0],
                o_proj.out_features,
                in_place=self.ar_result_in_place,
                deferred=self.ar_finish_deferred,
            )
            o_proj.next_ar_push = ar_push
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        if ar_push is not None:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # TRTLLM_K3_AR_PUSH_AMID: the result may stay in the workspace for A_mid
            # (TRTLLM_K3_AR_FINISH_FUSED: A_mid may also finish it).
            out_shape = out.shape
            out = finish_lamport_push(
                self._o_allreduce,
                ar_push,
                in_place=self.ar_result_in_place,
                deferred=self.ar_finish_deferred,
            )
            if not isinstance(out, (LamportResult, LamportPending)):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # TRTLLM_K3_AR_FINISH_FUSED: o_proj ran without the push epilogue; A_mid finishes.
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


@nccl_window_tensor_scoped_module
class KimiLinearDecoderLayer(nn.Module):
    # TRTLLM_K3_ULL_HYGIENE: the model's side-stream copies (``None``: off).
    _side_copies: Optional[_SideCopies] = None

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

    def _attn_takes_mxfp8(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Whether the pre-attention mix should also emit its MXFP8 form for the attention mixer
        (``TRTLLM_K3_ATTN_RES_MXFP8``; generation-only batches): the KDA fused QKVG projection
        (``qkvg_takes_mxfp8``) or the MLA fused q_a / kv_a projection (``q_a_takes_mxfp8``)."""
        if not _K3_ATTN_RES_MXFP8 or attn_metadata.num_contexts != 0:
            return False
        if self.token_sliced:
            # helix_a2a: the mix runs on this rank's rows, the attention on all of them.
            return False
        if self.is_kda:
            return self.linear_attn.qkvg_takes_mxfp8(num_rows)
        q_a_takes_mxfp8 = getattr(self.self_attn.mixer, "q_a_takes_mxfp8", None)
        return q_a_takes_mxfp8 is not None and q_a_takes_mxfp8(num_rows)

    def _attn_mx_atom(self, num_rows: int) -> bool:
        """``TRTLLM_K3_MID_MXFP8_IN``: whether that MXFP8 form goes to the mid-M GEMM, i.e.
        the mix stores its scales in that GEMM's layout (k3_attn_res ``mx_atom``)."""
        if not _K3_MID_MXFP8_IN:
            return False
        if self.is_kda:
            mx_atom = getattr(self.linear_attn, "qkvg_mx_atom", None)
        else:
            mx_atom = getattr(self.self_attn.mixer, "q_a_mx_atom", None)
        return mx_atom is not None and bool(mx_atom(num_rows))

    def _snapshot_slot(
        self, block_residual: torch.Tensor, num_snapshots: int
    ) -> Optional[torch.Tensor]:
        """``TRTLLM_K3_ULL_HYGIENE``: on a block-boundary layer, the bank slot this layer's
        snapshot goes to, for the fused pre-attention mix to store the prefix sum into (no copy)."""
        if (
            not _K3_ULL_HYGIENE
            or self.layer_idx % self.attn_res_block_size != 0
            or num_snapshots >= block_residual.shape[0]
        ):
            return None
        return block_residual[num_snapshots]

    def _capture_hidden_states(self, capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``; on the model's side stream with
        ``TRTLLM_K3_ULL_HYGIENE`` (joined at the end of the model forward, before the drafter)."""
        side = self._side_copies
        if side is None or not side.active():
            capture[0].maybe_capture_hidden_states(capture[1], value, None)
            return
        side.run(lambda: capture[0].maybe_capture_hidden_states(capture[1], value, None), value)

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
        ``_DeferredMoEResidual`` (``TRTLLM_K3_ATTN_RES_PDL=2``).

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

        all_rank_num_tokens = (
            sliced.all_rank_num_tokens(num_tokens)
            if sliced is not None
            else getattr(attn_metadata, "all_rank_num_tokens", None)
        )
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        attn_input_mxfp8 = None
        if isinstance(hidden_states, _DeferredRpTail):
            # The previous layer's row-parallel MoE tail (TRTLLM_K3_RP_TAIL): this mix finishes
            # its all-reduce where it can; the DSpark tap path materializes it first.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_rp_tail(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # TRTLLM_K3_ULL_TAIL_SANDWICH with TRTLLM_K3_KDA_QKVG_STAGE: release the
                    # staging QKVG GEMM at kernel entry, as the MoE-residual mix does.
                    entry_trigger=(_K3_ULL_TAIL_SANDWICH or _K3_ULL2_COLL_CP_TAIL)
                    and mxfp8
                    and self.is_kda
                    and self.linear_attn.qkvg_stages_weights(rows),
                    prefix_out=self._snapshot_slot(block_residual, num_snapshots),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        elif isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # TRTLLM_K3_KDA_QKVG_STAGE: release the staging QKVG GEMM at kernel entry.
                    entry_trigger=mxfp8
                    and self.is_kda
                    and self.linear_attn.qkvg_stages_weights(rows),
                    mx_atom=mxfp8 and self._attn_mx_atom(rows),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None and not _AUX_ATTN_RES_STREAM_ENABLED:
            # prefix_only wants the incoming running prefix, which is already in
            # hand -- no pre-norm intermediate is needed, so the tap is a plain
            # read and this layer keeps the fused attn_res+RMSNorm path like the
            # other 88. ``prefix_sum`` already has the previous layer's MLP
            # output folded in, i.e. the quantity vLLM builds as
            # ``prefix_sum + pending_mlp_out`` on its deferred-add path.
            self._capture_hidden_states(capture, _full_rows(prefix_sum))
            capture = None

        if capture is not None:
            # The mixture tap needs the PRE-norm value (SGLang's
            # aggregate_stream is a "softmax mixture, no output norm"), which
            # the fused attn-res + RMSNorm kernel does not expose. Keep the two
            # steps split on captured layers only and fuse everywhere else.
            mixed = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if num_snapshots > 0
                else hidden_states
            )
            self._capture_hidden_states(capture, _full_rows(mixed))
            hidden_states = self.input_layernorm(mixed)
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
            snapshot = block_residual[num_snapshots]
            if prefix_sum.data_ptr() != snapshot.data_ptr():
                # (TRTLLM_K3_ULL_HYGIENE: the fused mix above may have stored it there already.)
                snapshot.copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # TRTLLM_K3_AR_PUSH_AMID: when A_mid below adds the attention output to the prefix sum,
        # the attention block may hand back its all-reduce result in place (a LamportResult).
        # A_mid must read it before the next all-reduce on the workspace re-arms that buffer:
        # the shared expert, whose down-projection all-reduce is the next one, forks after A_mid,
        # and its pushed finish kernel can only become resident once its producer has passed its
        # own griddepcontrol.wait (every grid before, A_mid included, has then completed).
        # A token-sliced layer reduce-scatters the attention output itself (no all-reduce).
        attn = self.linear_attn if self.is_kda else self.self_attn
        attn.ar_result_in_place = (
            sliced is None
            and prefix_sum is not None
            and _k3_attn_res_lamport_applies(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        )
        # TRTLLM_K3_AR_FINISH_FUSED: the same A_mid also finishes a one-shot pushed all-reduce
        # (a LamportPending); like the in-place result it lies on every path to the next
        # all-reduce on the workspace.
        attn.ar_finish_deferred = attn.ar_result_in_place and _K3_AR_FINISH_FUSED
        # Attention is head-TP over every row: KDA rows are gathered in and its
        # row-sharded o_proj partials reduce-scattered back out here; helix MLA
        # does both itself (_helix_cp_allgather_input / _helix_cp_output_projection).
        # The MXFP8 form of the mix covers this rank's rows only, so a token-sliced
        # layer never produces one (``_attn_takes_mxfp8``).
        if self.is_kda:
            if attn_input_mxfp8:
                self.linear_attn.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.linear_attn(_full_rows(hidden_states), attn_metadata)
            if sliced is not None:
                hidden_states = sliced.reduce_scatter(hidden_states, num_tokens)
        else:
            if attn_input_mxfp8:
                self.self_attn.mixer.next_input_mxfp8 = tuple(attn_input_mxfp8)
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
            # TRTLLM_K3_ULL_FRONT_EARLY: the MoE front's head is the next launch after A_mid.
            releases_early = (
                getattr(self.block_sparse_moe, "ull_front_releases_early", None)
                if self.is_moe
                else None
            )
            front_early = releases_early is not None and releases_early(prefix_sum)
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
                entry_trigger=front_early,
            )
        if self.is_moe and _K3_FUSE_MOE_RESIDUAL and not torch.compiler.is_compiling():
            # TRTLLM_K3_RP_TAIL: the row-parallel tail serves generation-only batches.
            rp_kwargs = (
                {"rp_tail_ok": getattr(attn_metadata, "num_contexts", 1) == 0}
                if _K3_RP_TAIL
                else {}
            )
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                all_rank_num_tokens,
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
                **rp_kwargs,
            )
            return prefix_sum, num_snapshots
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states, all_rank_num_tokens)
        else:
            hidden_states = self.mlp(hidden_states)

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
# Model.
# ---------------------------------------------------------------------------


# TRTLLM_K3_ULL4_MLA_PLAN_EARLY=1: in a generation-only step (multi-stream) the model writes the
# FP4 MLA step metadata (generation lengths + packed split plan, device page table) on a side
# stream at the step start; the first FP4 MLA layer joins it instead of launching those kernels on
# its critical path. The same kernels with the same inputs: bitwise.
_K3_ULL4_MLA_PLAN_EARLY = os.environ.get("TRTLLM_K3_ULL4_MLA_PLAN_EARLY", "0") == "1"


class KimiLinearModel(DecoderModel):
    # TRTLLM_K3_ULL_HYGIENE: side-stream copies of the decode step (``None``: off).
    _side_copies: Optional[_SideCopies] = None

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream(priority=_K3_SHARED_STREAM_PRIORITY)]
        aux_stream_list += [torch.cuda.Stream() for _ in range(2)]
        aux_stream_list.append(torch.cuda.Stream(priority=_K3_LATENT_STREAM_PRIORITY))
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
        # TRTLLM_K3_ULL_HYGIENE: one side stream for the DSpark taps of every layer.
        self._side_copies = _SideCopies(torch.cuda.Stream()) if _K3_ULL_HYGIENE else None
        for layer in self.layers:
            layer._side_copies = self._side_copies
        # TRTLLM_K3_ULL4_MLA_PLAN_EARLY: the side stream and events of the step-metadata prelaunch.
        self._mla_plan_stream = torch.cuda.Stream() if _K3_ULL4_MLA_PLAN_EARLY else None
        self._mla_plan_events = (
            (torch.cuda.Event(), torch.cuda.Event()) if _K3_ULL4_MLA_PLAN_EARLY else None
        )
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

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    def _prelaunch_mla_step_metadata(self, attn_metadata: AttentionMetadata) -> None:
        """TRTLLM_K3_ULL4_MLA_PLAN_EARLY: fork the step's FP4 MLA metadata kernels to a side stream
        here; the first FP4 MLA layer joins it (``step_metadata_ready``)."""
        if not do_multi_stream():
            return
        mixer = None
        for layer in self.layers:
            if layer.is_kda:
                continue
            candidate = getattr(layer.self_attn, "mixer", None)
            mqa = getattr(candidate, "mqa", None)
            if getattr(mqa, "uses_fp4_mla_attention", False) and hasattr(
                candidate, "prelaunch_step_metadata"
            ):
                mixer = candidate
                break
        if mixer is None:
            return
        fork, done = self._mla_plan_events
        fork.record()
        with torch.cuda.stream(self._mla_plan_stream):
            fork.wait()
            mixer.prelaunch_step_metadata(attn_metadata)
            done.record()
        mixer.step_metadata_ready = done

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
            weight = self.embed_tokens.weight
            if spec_head_fusion_enabled() and input_ids.dim() == 1 and weight.is_contiguous():
                # Same rows as nn.Embedding, copied with 16-byte accesses.
                inputs_embeds = embedding_gather(input_ids, weight)
            else:
                inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = attn_metadata.num_tokens if self.helix_a2a is not None else None
        if self.helix_a2a is not None and hidden_states.shape[0] != num_tokens:
            # Rows are split by the metadata token count on every rank.
            raise ValueError(
                f"Kimi K3 helix_a2a does not support padded batches (got "
                f"{hidden_states.shape[0]} rows, metadata says {num_tokens} tokens)."
            )
        if self._mla_plan_stream is not None and attn_metadata.num_contexts == 0:
            self._prelaunch_mla_step_metadata(attn_metadata)

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
        sliced = self.helix_a2a is not None
        hidden_states = _materialize_prefix_sum(hidden_states)

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

        output = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        if sliced:
            output = self.helix_a2a.all_gather(output, num_tokens)
        if self._side_copies is not None:
            # TRTLLM_K3_ULL_HYGIENE: the DSpark taps land before anything after the model (the
            # drafter) reads the capture buffer.
            self._side_copies.join()
        return output


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


class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE) of the Rubin agentic package.

    The text backbone of the package's ``KimiK3ForConditionalGeneration`` (``wrapper.py``).
    Unlike the built-in ``models/modeling_kimi_linear.py`` it is registered nowhere: the
    modeling_v2 targets reach it through the wrapper, and ``modeling_v2=off`` keeps the
    built-in class.

    Speculative decoding: main's one-engine shell (``SpecDecOneEngineForCausalLM``) builds the
    drafter and the worker; :meth:`_install_private_spec` is the single place where the
    package's own DSpark drafter / worker replace them."""

    mamba_metadata_cls = KimiK3MambaMetadata

    def __init__(self, model_config: ModelConfig):
        mlp_fp8 = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "0")
        if mlp_fp8 not in ("0", "1"):
            raise ValueError(
                f"{_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV} must be 0 or 1; got {mlp_fp8!r}"
            )
        self._fp8_weight_read_moe_mlp = mlp_fp8 == "1"
        cfg = _get_text_config(model_config.pretrained_config)
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
        # The drafter and worker of main's shell are built (under the repurposed helix mapping,
        # which is still active here); the package's own replace them at this one point.
        self._install_private_spec(model_config)

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def _install_private_spec(self, model_config: ModelConfig) -> None:
        """HOOK POINT (k3_rubin/spec): swap in the package's own DSpark drafter and worker.

        Called once from ``__init__``, right after ``SpecDecOneEngineForCausalLM.__init__`` and
        before the helix CP mapping is restored, i.e. with the same mapping main's drafter was
        built under. At that point main's shell has set ``self.spec_config``,
        ``self.draft_config``, ``self.draft_model`` (from the draft-builder registry, or None for a
        separate draft engine), ``self.spec_worker`` (``mapping_for_full_rows()`` mapping,
        ``set_target_model(self)`` done), ``self.use_separate_draft_kv_cache`` and ``self.epilogue``.

        A package-internal implementation must replace ``draft_model`` / ``spec_worker`` consistently:
        swap them in ``self.epilogue`` (main appends both there), call ``set_target_model(self)`` on the
        new worker, keep ``self.lm_head`` a stock ``LMHead`` and ``self.logits_processor`` exactly a
        ``LogitsProcessor`` (main's vocab-parallel selection checks the type), and keep the
        attribute contract main's ``DFlashSpecMetadata.prepare()`` reads from the worker
        (``_req_to_slot``, ``_free_slots``, ``_assign_slot``, ``_batch_to_slot_buffer``).

        One-engine DSpark with a standalone MLA drafter gets the package's drafter
        (``spec/modeling_dspark_mla.py``) and worker (``spec/dspark.py`` on ``spec/dflash.py``)
        through :func:`~.spec.shell.install_k3_dspark`, which keeps that contract; every other
        configuration keeps main's.
        """
        install_k3_dspark(self, model_config)

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
        if _K3_HELIX_V_UP_BEFORE_A2A and attention.mapping.has_cp_helix():
            expanded = value_scale.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                _FP8_BLOCK, -1
            )
            # main's MLA owns the V-up-before-exchange flag; bind through it (its
            # transform_weights rebuilds the same weight from kv_b_proj by default).
            attention._bind_helix_v_up_weight((value_weight.float() * expanded).to(torch.bfloat16))
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
                    # has no intermediate ``mixer`` scope.
                    ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.")
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
            for expert_idx in moe.local_expert_ids:
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
                if _K3_HELIX_V_UP_BEFORE_A2A and mla_mixer.mapping.has_cp_helix():
                    mla_mixer._bind_helix_v_up_weight(param[h * n :].view(h, v, kv))
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
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            moe.expert_ckpt_spec.loader(
                backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
            )
            maybe_finalize_layer(moe)

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
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
            split_file_jobs = []
            for layer_idx, moe, base in expert_jobs:
                del layer_idx
                spec = moe.expert_ckpt_spec
                for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
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
                        per_file.setdefault(files.pop(), []).append(job)
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

            def load_expert_file(file_name: str, jobs: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
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

            run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=4)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=4)
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
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=4)

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
                self.model,
                include_fused_gate_up=self.model_config.mapping.enable_attention_dp,
            )
            logger.info(f"Kimi K3: quantized {converted} BF16 shared/latent MLP projections to FP8")
