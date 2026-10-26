# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: step5-preview-nvfp4 / sm_103 / unified -- one forward for every
parallel layout the routing module admits (tp4, tep4, dep4, dep2).

Step-5-Preview, ModelOpt NVFP4 export: 92 decoder layers, hidden 4096, an fp32 residual
stream with zero-centred fp32 RMSNorms (`x / rms(x) * (1 + w)`). Attention is GQA, 64
query heads over 4 KV heads of 192, with a per-head QK norm and a per-head sigmoid output
gate. Layers repeat [sliding x 3, full]: the 69 sliding layers attend over the newest 512
keys with full RoPE at theta 1e4; the 23 full layers (3, 7, ..., 91) rotate the leading
64 dims at theta 1e7 and attend sparsely -- an indexer scores 8-token regions and each
query attends its top-512 regions plus its own. Layers 0-2 and 91 run a dense clamped
SwiGLU; 3-90 run 352 routed experts (top-8, sigmoid router with a selection-only bias,
weights renormalized and scaled by 3.0) beside a shared expert, all with the SwiGLU
clamped after SiLU at 7.0. The routed experts of 3-87 are NVFP4 (W4A4), the rest bf16.

The weight shapes come from the engine's `Mapping` (weights.py); the layout is settled
once, in `post_load_weights`, rather than by branching in the forward:

* **Attention TP** (attention DP off): each rank runs its share of the heads over every
  token -- 16 query heads, one KV head, one indexer provider group at TP4 -- and its
  slice of each dense MLP and shared expert. o_proj and the dense MLP leave bf16 partial
  sums completed by a plain `comm/allreduce`; a MoE layer's routed and shared partials
  are added in fp32 and completed by one fp32 allreduce, which runs under NCCL (the
  entry refuses a 4-byte sum on the IPC workspace). The routed experts are split by
  moe_tp x moe_ep inside the same group.

* **Attention DP**: each rank runs every head and the whole shared expert over its own
  requests; the routed experts are a round trip -- pad to the group's largest token
  count, `comm/allgather`, route and run this rank's experts over the gathered set,
  `comm/reducescatter` the fp32 partials back.

The residual stream stays fp32: each residual add and the norm after it is one
`cuda_tile_rms_norm_fuse_residual` call on the widened delta, the normed fp32 output
rounded once to bf16. The final norm reads the residual rounded to bf16 first, as the
reference does.

**Heads are padded.** trtllm-gen has no head-192 FMHA on sm_103 and the sparse kernels
take power-of-two heads, so q, k and v are written into a zero-padded 256-wide packed
buffer after the QK norm, rotated there (`flashinfer_apply_rope_with_cos_sin_cache` at
head 256 -- its in-place head-192 rotary-192 path is defective), and attended at head
256: `q_scaling` sqrt(192/256) on `thop_attention` restores the 192^-0.5 softmax scale,
which the sparse kernels take directly. The pools the cache manager allocates are 256
wide; the pad columns of every output come back zero and are dropped before o_proj.

**Sparse layers** keep, per KV block, the indexer's per-token proxy key and logits
(`[k | z]` bf16) and one e4m3 summary per completed 8-token region, all on the
`Step5CacheManager`'s index page. A step stores its tokens' K/V (`qsa_paged_kv_store`)
and `[k | z]` rows, summarizes every region one of its tokens completes
(`csa_region_compress`), scores each query against the summaries of the regions before
its own (`csa_index_scores`), selects the top 512 (`flashinfer_top_k_ragged_transform`,
ties to the lower region) and attends the selected regions' tokens plus its own region's
(`qsa_paged_sparse_gqa`, one call per KV group). Each selection row holds the current
region's 8 positions first, then the regions' (`-1` past the query and in unused slots).

**Images** (vision.py). A prefill step whose requests carry images runs the vision tower
over their tiles in the shell, eagerly, before the step: its rows replace the embeddings
of the step's `<im_patch>` tokens (id 128001, in vocabulary, so the shell publishes it as
`multimodal_token_ids`), and the targets read the spliced embedding as `inputs_embeds`.
A step without images, every decode step among them, runs as it would without them.
"""

import math
import os
from typing import Callable, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.fused_sigmoid_mul_inplace import (
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_apply_rope_with_cos_sin_cache import (
    flashinfer_apply_rope_with_cos_sin_cache,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_top_k_ragged_transform import (
    TopKTieBreak,
    flashinfer_top_k_ragged_transform,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_kv_store import (
    qsa_paged_kv_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_kv_store_nvfp4 import (
    qsa_paged_kv_store_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (
    qsa_paged_sparse_gqa_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    NCCL,
    NONE,
    allreduce,
    allreduce_group_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.cuda_tile_rms_norm import (
    cuda_tile_rms_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.cuda_tile_rms_norm_fuse_residual import (
    cuda_tile_rms_norm_fuse_residual,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.quantization.mode import QuantAlgo

from . import weights as _weights
from .ops.activation.clamped_swiglu import clamped_swiglu
from .ops.attention.csa_index_scores import csa_index_scores
from .ops.attention.csa_region_compress import csa_region_compress
from .vision import VisionTower

# Per-call constants of the sliding layers' attention call: packed-QKV causal GQA, RoPE
# applied outside (in-kernel RoPE off; the rope_* values are inert placeholders), no
# MLA, no sparse argument group, no speculation. `quant_mode`, the KV scales,
# `q_scaling` and the window are passed beside these.
_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# The norms' kernel path: the persistent cuTile kernel, zero-centred weights.
_NORM_PERSISTENT = True
_NORM_GATHER = True
_ZERO_CENTRED = True

_SF_VEC = 16
_SF_SWIZZLED = True
# `fused_moe`'s own activation enum: 5 is Swiglu (not the trtllm-gen runner's enum).
_FUSED_MOE_ACT_SWIGLU = 5
# The router is one group: `noaux_tc_op` with a single group selects over all experts.
_ROUTER_GROUPS = 1

# The MoE entries' certified row counts stop at 8192. Under attention DP the gathered
# set reaches world * max_num_tokens, so the expert call is chunked; every token is
# independent through routing, quantization and the expert GEMMs, so the chunks
# re-joined are the whole call.
_MOE_MAX_T = 8192  # capacity: fixed: the certification bound of fused_moe / fp4_quantize

# Region top-k selection: exact set, ties to the lower region id (the reference's
# `torch.topk` over (-score, id)); ids emitted ascending so a step's selection, and with
# it the sparse attention's summation order, is the same run to run.
_TOPK_TIE_BREAK = TopKTieBreak.SMALL
_TOPK_DETERMINISTIC = True
_TOPK_GRAPH_SAFE = False


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    """llama3 frequency scaling of `inv_freq`, as the checkpoint's `rope_scaling` block
    defines it. At this checkpoint's factor 1.0 it is the identity up to fp32 rounding."""
    factor = float(scaling["factor"])
    low = float(scaling["low_freq_factor"])
    high = float(scaling["high_freq_factor"])
    original = float(scaling["original_max_position_embeddings"])
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > original / low, inv_freq / factor, inv_freq)
    smooth = (original / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    in_band = (wavelen >= original / high) & (wavelen <= original / low)
    return torch.where(in_band, smoothed, scaled)


def _rope_table(theta: float, span: int, scaling, rows: int, device) -> torch.Tensor:
    """`[rows, span]` fp32 `[cos | sin]` table of a NeoX rotation over `span` dims.

    Built as the reference builds its tables -- frequencies and angles in fp32 -- and
    rounded through bf16, the precision the reference stores and rotates with, so the
    rotation matches its arithmetic rather than an fp32 table's."""
    inv = 1.0 / (theta ** (torch.arange(0, span, 2, device=device, dtype=torch.float32) / span))
    if scaling:
        inv = _llama3_inv_freq(inv, scaling)
    angles = torch.outer(torch.arange(rows, device=device, dtype=torch.float32), inv)
    table = torch.cat([angles.cos(), angles.sin()], dim=1)
    return table.to(torch.bfloat16).float().contiguous()


def _rope_key(theta: float, span: int, scaling) -> tuple:
    """A rope table's identity: layers that rotate alike share one table."""
    return (float(theta), int(span), repr(scaling))


class _Bound(NamedTuple):
    """The views a step reads from one KV cache manager.

    `pool_row[i]` is layer i's row of the pool mapping (`layer_offsets`), what the
    attention op indexes. `slot_pool` / `slot_scale` turn the device block offsets of
    the full-attention life cycle into the slot ids the views take. Per sparse layer:
    `kv` the K and V pool views (HND, 256 wide) and, under NVFP4, their block-scale
    views; `kz` the `[k | z]` plane, `summary` the region summaries."""

    cm: object
    pool_row: dict
    slot_pool: int
    slot_scale: int
    tokens_per_block: int
    regions_per_block: int
    kv: dict
    kz: dict
    summary: dict


class _Layer(NamedTuple):
    """One decoder layer's weights at this rank's share, as the views the ops read: GEMM
    operands are `.t()` views; `norm_next` is the next layer's input norm (None on the
    last layer, which runs the final norm instead). Fields of another layer kind are
    None."""

    index: int
    sliding: bool
    rope_key: tuple
    qkvg: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: torch.Tensor
    norm2: torch.Tensor
    norm_next: torch.Tensor | None
    idx: torch.Tensor | None
    idx_qn: torch.Tensor | None
    idx_kn_w: torch.Tensor | None
    idx_kn_b: torch.Tensor | None
    idx_rope_key: tuple | None
    gu: torch.Tensor | None
    dn: torch.Tensor | None
    limit: float
    router: torch.Tensor | None
    router_bias: torch.Tensor | None
    sgu: torch.Tensor | None
    sdn: torch.Tensor | None
    shared_limit: float
    fc1: torch.Tensor | None
    fc2: torch.Tensor | None
    gs_in: torch.Tensor | None
    quant_scales: list
    expert_limits: torch.Tensor | None


def _layer_views(core: "Step5PreviewModelingV2") -> list[_Layer]:
    """Every layer's `_Layer`, read off the realized weights and the scalars
    `post_load_weights` derived."""
    d = core.dims
    w = core.w
    cfg = core.text_config
    layers = []
    for i in range(d.num_layers):
        sparse = i in d.sparse_layers
        moe = i in d.moe_layers
        nvfp4 = i in d.nvfp4_layers
        experts: dict = dict(fc1=None, fc2=None, gs_in=None, quant_scales=[], expert_limits=None)
        if moe:
            experts["expert_limits"] = core._expert_limits[i]
            if nvfp4:
                gs_in, fc1_global, g2, fc2_global = core._expert_scales[i]
                experts.update(
                    fc1=w[f"l{i}_fc1_w"].view(torch.int64),
                    fc2=w[f"l{i}_fc2_w"].view(torch.int64),
                    gs_in=gs_in,
                    quant_scales=[
                        gs_in,
                        w[f"l{i}_fc1_s"].view(torch.int32),
                        fc1_global,
                        g2,
                        w[f"l{i}_fc2_s"].view(torch.int32),
                        fc2_global,
                    ],
                )
            else:
                experts.update(fc1=w[f"l{i}_fc1"], fc2=w[f"l{i}_fc2"])
        layers.append(
            _Layer(
                index=i,
                sliding=d.sliding[i],
                rope_key=core._layer_rope[i],
                qkvg=w[f"l{i}_qkvg"].t(),
                q_norm=w[f"l{i}_q_norm"],
                k_norm=w[f"l{i}_k_norm"],
                o=w[f"l{i}_o"].t(),
                norm2=w[f"l{i}_norm2"],
                norm_next=w[f"l{i + 1}_norm1"] if i + 1 < d.num_layers else None,
                idx=w[f"l{i}_idx"].t() if sparse else None,
                idx_qn=w[f"l{i}_idx_qn"] if sparse else None,
                idx_kn_w=w[f"l{i}_idx_kn_w"] if sparse else None,
                idx_kn_b=w[f"l{i}_idx_kn_b"] if sparse else None,
                idx_rope_key=core._index_rope[i] if sparse else None,
                gu=None if moe else w[f"l{i}_gu"].t(),
                dn=None if moe else w[f"l{i}_dn"].t(),
                limit=float(cfg.swiglu_limit_for(i)),
                router=w[f"l{i}_router"].t() if moe else None,
                router_bias=w[f"l{i}_router_bias"] if moe else None,
                sgu=w[f"l{i}_sgu"].t() if moe else None,
                sdn=w[f"l{i}_sdn"].t() if moe else None,
                shared_limit=float(cfg.swiglu_limit_shared_for(i)),
                **experts,
            )
        )
    return layers


class Step5PreviewModelingV2(DecoderModel):
    """The core: the weights, the derived operands and the layout-dependent collectives;
    builds the prefill and decode targets and dispatches each step to one."""

    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        super().__init__(model_config)
        self.composite_config = composite_config
        self.text_config = composite_config.text_config
        cfg = self.text_config
        m = model_config.mapping
        sparse = cfg.sparse_config

        kv_algo = model_config.quant_config.kv_cache_quant_algo
        self.quant_mode = {
            None: 0,
            QuantAlgo.FP8: QM_FP8_KV_CACHE,
            QuantAlgo.NVFP4: QM_NVFP4_KV_CACHE,
        }[kv_algo]

        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)
        d = self.dims

        self.eps = float(cfg.rms_norm_eps)
        self.window = int(cfg.sliding_window)
        # Heads are zero-padded to the power of two the attention kernels take.
        self.pad_dim = 1 << (d.head_dim - 1).bit_length()
        self.q_scaling = math.sqrt(d.head_dim / self.pad_dim)
        self.softmax_scale = d.head_dim**-0.5
        self.region = int(sparse["region_block_size"])
        self.region_topk = int(sparse["topk"])
        # The indexer's per-head weights are prescaled by heads-per-group^-0.5.
        self.idx_groups = d.kv_heads
        self.idx_heads_per_group = d.idx_heads // d.kv_heads
        self.idx_weight_scale = float(self.idx_heads_per_group) ** -0.5
        self.idx_rope_dim = int(sparse["sparse_indexer_rope_dim"])
        self.topk = int(cfg.moe_top_k)
        self.routed_scale = float(cfg.moe_router_scaling_factor)
        # fused_moe's FC2 + finalize fusion adds each token's top-k expert outputs with
        # atomics, in an order that changes run to run; unfused, a separate finalize
        # reduces them in a fixed order. The engine asks for that with
        # FORCE_DETERMINISTIC=1 or `moe_config.disable_finalize_fusion`. Read here, once,
        # so the forward only passes it on.
        self.moe_fused_finalize = not (
            os.environ.get("FORCE_DETERMINISTIC", "0") == "1"
            or model_config.moe_disable_finalize_fusion
        )

        # Rope geometry per layer: (theta, rotated span, scaling). The span is
        # head_dim * partial_rotary_factor, made even; the indexer rotates its leading
        # `sparse_indexer_rope_dim` dims at the layer's theta and scaling.
        self._layer_rope = {}
        self._index_rope = {}
        self._rope_specs = {}
        for i in range(d.num_layers):
            span = cfg.rotary_dim_for(i)
            span -= span % 2
            theta, scaling = cfg.rope_theta_for(i), cfg.rope_scaling_for(i)
            self._layer_rope[i] = _rope_key(theta, span, scaling)
            self._rope_specs[self._layer_rope[i]] = (theta, span, scaling)
            if i in d.sparse_layers:
                self._index_rope[i] = _rope_key(theta, self.idx_rope_dim, scaling)
                self._rope_specs[self._index_rope[i]] = (theta, self.idx_rope_dim, scaling)
        self._rope: dict = {}
        self._rope_rows = 0

        self.rank = m.rank
        self.dp_size = m.world_size if m.enable_attention_dp else 1
        self.dp_group = list(range(m.world_size))
        # The communicator side of the attention-DP allgather and reduce-scatter: under MPI
        # the plain symbols, with MPI disabled the `_pg` symbols over the TP process group,
        # which spans the DP ranks.
        self.dp_process_group = (
            m.tp_group_pg.boxed() if mpi_disabled() and self.dp_size > 1 else None
        )

        self._bound: _Bound | None = None
        self._expert_scales: dict = {}
        self._expert_limits: dict = {}
        self.kv_scale_orig_quant = None
        self.kv_scale_quant_orig = None
        self._sum_bf16: Callable | None = None
        self._sum_fp32: Callable | None = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
        self.vision: VisionTower | None = None

    def post_load_weights(self) -> None:
        """Derive what the targets read, then build them, now that the weights are real.

        Named for the shell hook that drives it; DecoderModel has no such hook. The
        attention-TP sums are chosen here, once, on every rank in the same order:
        building the IPC workspace is collective over the TP group.
        """
        d = self.dims
        w = self.w
        device = w["embed"].device
        window = slice(d.expert_offset, d.expert_offset + d.local_experts)
        for i in d.moe_layers:
            limit = float(self.text_config.swiglu_limit_for(i))
            self._expert_limits[i] = torch.full(
                (d.local_experts,), limit, dtype=torch.float32, device=device
            )
        for i in d.nvfp4_layers:
            isc1, isc2 = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc2"]
            # One quantization of the hidden states feeds FC1 of every expert on every
            # rank, so its activation scale is one value; ModelOpt calibrated the same
            # value for every expert and for the up and gate halves of FC1, whose weight
            # global scales FC1 also reads as one.
            assert torch.equal(isc1, w[f"l{i}_e_isc1_up"]) and torch.equal(
                w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]
            ), f"layer {i}: the FC1 halves carry different NVFP4 scales"
            isc1_one = isc1.max()
            isc2_one = isc2.max()
            gs_in = (1.0 / isc1_one).reshape(()).contiguous()
            g2 = (1.0 / isc2_one).reshape(()).contiguous()
            fc1_global = (isc1_one * w[f"l{i}_e_ws2_1"][window]).contiguous()
            fc2_global = (isc2_one * w[f"l{i}_e_ws2_2"][window]).contiguous()
            self._expert_scales[i] = (gs_in, fc1_global, g2, fc2_global)
        if self.quant_mode in (QM_FP8_KV_CACHE, QM_NVFP4_KV_CACHE):
            scales = 3 if self.quant_mode == QM_NVFP4_KV_CACHE else 1
            self.kv_scale_quant_orig = torch.ones(scales, dtype=torch.float32, device=device)
            self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        if self.model_config.max_seq_len:
            self._ensure_rope(int(self.model_config.max_seq_len), device)
        self._sum_bf16, self._sum_fp32 = _tp_sums(self)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        self.vision = VisionTower(self)

    def _ensure_rope(self, rows: int, device) -> None:
        """Every distinct rope table at `rows` positions or more. A row depends only on
        its own index, so rebuilding at a larger size changes no existing row."""
        if rows <= self._rope_rows:
            return
        self._rope = {
            key: _rope_table(theta, span, scaling, rows, device)
            for key, (theta, span, scaling) in self._rope_specs.items()
        }
        self._rope_rows = rows

    def _bind(self, md) -> None:
        """Bind to one KV cache manager: take every pool view from it.

        Once per manager, not per model: with capacity estimation the engine builds a
        throwaway manager, warms up, shuts it down and builds the real one on the same
        module, and a view held across the swap dangles silently."""
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        assert not torch.cuda.is_current_stream_capturing(), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        d = self.dims
        pool_row = {i: cm.layer_offsets[i] for i in range(d.num_layers)}
        first = d.sparse_layers[0]
        slot_pool, slot_scale = cm.get_block_offset_slot_scale(first)
        kv, kz, summary = {}, {}, {}
        for i in d.sparse_layers:
            pool = cm.get_kv_slot_buffers(i, kv_layout="HND")
            scales = (
                cm.get_kv_block_scale_slot_buffers(i, kv_layout="HND")
                if self.quant_mode == QM_NVFP4_KV_CACHE
                else None
            )
            kv[i] = (
                pool[:, 0],
                pool[:, 1],
                None if scales is None else scales[:, 0],
                None if scales is None else scales[:, 1],
            )
            kz[i] = cm.get_index_kz_buffer(i)
            summary[i] = cm.get_index_summary_buffer(i)
        self._bound = _Bound(
            cm=cm,
            pool_row=pool_row,
            slot_pool=int(slot_pool),
            slot_scale=int(slot_scale),
            tokens_per_block=int(cm.tokens_per_block),
            regions_per_block=int(cm.tokens_per_block) // self.region,
            kv=kv,
            kz=kz,
            summary=summary,
        )

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: decode iff there are no context rows. One host
        integer read: a decode CUDA graph is a per-capture constant environment."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        md = attn_metadata
        # The rope tables cover the engine's max_seq_len (an unchecked out-of-bounds read
        # otherwise), and the pool views follow the manager; both fixed by the first step
        # on a manager, before any capture.
        self._ensure_rope(md.max_seq_len, self.w["embed"].device)
        if self._bound is None or md.kv_cache_manager is not self._bound.cm:
            self._bind(md)
        return self._select_target(md).forward(
            md, input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def _dp_pad_rows(self, md, num_tokens: int) -> int:
        """Rows this rank pads its block with before the MoE round trip: up to the largest
        per-rank token count in the group, so both collectives run in their uniform form.
        Zero without attention DP."""
        if self.dp_size == 1:
            return 0
        counts = [int(n) for n in md.all_rank_num_tokens]
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.rank] == num_tokens, (
            f"rank {self.rank} holds {num_tokens} rows but all_rank_num_tokens says "
            f"{counts[self.rank]} ({counts}); the collectives would gather the wrong split"
        )
        return max(counts) - num_tokens


def _tp_sums(core: Step5PreviewModelingV2) -> tuple[Callable, Callable]:
    """The bf16 and fp32 completions of attention-TP partial sums, chosen once.

    Under attention TP the o_proj and dense-MLP outputs are bf16 partials, summed by a
    plain allreduce on the IPC workspace; a MoE layer's routed + shared partials are fp32
    and summed under NCCL. Under attention DP each rank's sums are already whole."""
    if core.dims.attn_tp == 1:

        def whole(x: torch.Tensor) -> torch.Tensor:
            return x

        return whole, whole

    mapping = core.model_config.mapping
    bf16_group = allreduce_group_args(mapping)
    fp32_group = allreduce_group_args(mapping, NCCL)

    def sum_bf16(x: torch.Tensor) -> torch.Tensor:
        return allreduce(
            x,
            None,
            None,
            None,
            None,
            op=NONE,
            eps=core.eps,
            trigger_completion_at_end=False,
            **bf16_group,
        )

    def sum_fp32(x: torch.Tensor) -> torch.Tensor:
        return allreduce(
            x,
            None,
            None,
            None,
            None,
            op=NONE,
            eps=core.eps,
            trigger_completion_at_end=False,
            **fp32_group,
        )

    return sum_bf16, sum_fp32


# ------------------------------------------------------------------ layer blocks --


def _project_qkv(core, L: _Layer, x, qkv, positions) -> torch.Tensor:
    """The fused `[q | k | v | g]` projection, the per-head QK norm, and q, k, v written
    into the zero-padded packed buffer `qkv` `[T, (H + 2 Hkv) * 256]`, q and k rotated in
    place there. Returns the gate logits `[T, H]` (a view of the projection)."""
    d = core.dims
    rows = x.shape[0]
    hd, pad = d.head_dim, core.pad_dim
    h, hkv = d.heads, d.kv_heads
    qw, kw = h * hd, hkv * hd
    proj = cublas_mm(x, L.qkvg)
    q = cuda_tile_rms_norm(
        proj[:, :qw].reshape(rows * h, hd),
        L.q_norm,
        core.eps,
        _NORM_PERSISTENT,
        _NORM_GATHER,
        _ZERO_CENTRED,
    )
    k = cuda_tile_rms_norm(
        proj[:, qw : qw + kw].reshape(rows * hkv, hd),
        L.k_norm,
        core.eps,
        _NORM_PERSISTENT,
        _NORM_GATHER,
        _ZERO_CENTRED,
    )
    heads = qkv.view(rows, h + 2 * hkv, pad)
    heads[:, :h, :hd].copy_(q.view(rows, h, hd))
    heads[:, h : h + hkv, :hd].copy_(k.view(rows, hkv, hd))
    heads[:, h + hkv :, :hd].copy_(proj[:, qw + kw : qw + 2 * kw].view(rows, hkv, hd))
    flat = qkv.view(rows, -1)
    flashinfer_apply_rope_with_cos_sin_cache(
        positions,
        flat[:, : h * pad],
        flat[:, h * pad : (h + hkv) * pad],
        pad,
        core._rope[L.rope_key],
        True,
    )
    return proj[:, qw + 2 * kw :]


def _gate_out(core, L: _Layer, attn, gate_logits) -> torch.Tensor:
    """The head-wise sigmoid gate on the padded attention output, the pad columns
    dropped, o_proj, and the TP sum. Returns bf16 `[T, hidden]`."""
    d = core.dims
    rows = attn.shape[0]
    pad = core.pad_dim
    gate = gate_logits.unsqueeze(-1).expand(rows, d.heads, pad).contiguous()
    attn = fused_sigmoid_mul_inplace(attn, gate)
    o_in = attn.view(rows, d.heads, pad)[:, :, : d.head_dim].reshape(rows, d.heads * d.head_dim)
    return core._sum_bf16(cublas_mm(o_in, L.o))


def _add_norm(core, delta: torch.Tensor, residual: torch.Tensor, weight) -> torch.Tensor:
    """`residual += delta` (fp32, in place) and the zero-centred norm of the new residual,
    rounded once to bf16."""
    x, _ = cuda_tile_rms_norm_fuse_residual(
        delta.float(), residual, weight, core.eps, _NORM_PERSISTENT, _NORM_GATHER, _ZERO_CENTRED
    )
    return x.to(core.dims.dtype)


def _final_norm(core, residual: torch.Tensor) -> torch.Tensor:
    """The final norm reads the residual rounded to bf16 (the reference's `norm_input`)."""
    return cuda_tile_rms_norm(
        residual.to(core.dims.dtype),
        core.w["final_norm"],
        core.eps,
        _NORM_PERSISTENT,
        _NORM_GATHER,
        _ZERO_CENTRED,
    )


def _swiglu_mlp(x, gate_up, down, limit: float) -> torch.Tensor:
    """A clamped SwiGLU MLP over this rank's intermediate slice: a partial sum under
    attention TP."""
    return cublas_mm(clamped_swiglu(cublas_mm(x, gate_up), limit), down)


def _experts(core, L: _Layer, x: torch.Tensor) -> torch.Tensor:
    """Route, quantize and run this rank's experts -- its EP window at its moe_tp slice of
    the intermediate -- over `x`, at most `_MOE_MAX_T` rows a call. bf16 partial sum."""
    d = core.dims
    parts = []
    for chunk in torch.split(x, _MOE_MAX_T, 0):
        logits = cublas_mm(chunk, L.router, None, torch.float32)
        weights, ids = noaux_tc_op(
            logits, L.router_bias, _ROUTER_GROUPS, _ROUTER_GROUPS, core.topk, core.routed_scale
        )
        if L.gs_in is not None:
            xq, xsf = fp4_quantize(chunk, L.gs_in, _SF_VEC, False, _SF_SWIZZLED)
            moe_in, moe_kw = xq, dict(input_sf=xsf, swizzled_input_sf=_SF_SWIZZLED)
        else:
            moe_in, moe_kw = chunk, {}
        parts.append(
            fused_moe(
                moe_in,
                ids,
                weights,
                L.fc1,
                None,
                L.fc2,
                None,
                chunk.dtype,
                L.quant_scales,
                swiglu_limit=L.expert_limits,
                tp_size=d.moe_tp,
                tp_rank=d.moe_tp_rank,
                ep_size=d.moe_ep,
                ep_rank=d.moe_ep_rank,
                use_fused_finalize=core.moe_fused_finalize,
                activation_type=_FUSED_MOE_ACT_SWIGLU,
                swiglu_clamp_after_silu=True,
                **moe_kw,
            )[0]
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _moe(core, L: _Layer, x: torch.Tensor, pad_rows: int) -> torch.Tensor:
    """A MoE layer's output, whole: routed + shared added in fp32 and completed across
    ranks (the fp32 TP sum, or the attention-DP round trip). fp32 `[T, hidden]`."""
    shared = _swiglu_mlp(x, L.sgu, L.sdn, L.shared_limit)
    if core.dp_size == 1:
        return core._sum_fp32(_experts(core, L, x).float() + shared.float())
    rows = x.shape[0]
    x_pad = x if not pad_rows else nn.functional.pad(x, [0, 0, 0, pad_rows])
    group, pg = core.dp_group, core.dp_process_group
    gathered = allgather(x_pad, None, group, pg)
    routed = reducescatter(_experts(core, L, gathered).float(), None, group, pg)
    return (routed if not pad_rows else routed[:rows]) + shared.float()


def _index_project(core, L: _Layer, x, positions):
    """The indexer's fused `[q | k | z | w]` projection: q head-normed and k layer-normed,
    both rotated on their leading dims; w widened to fp32 and prescaled. Returns
    `(q [T, Hi, 256], k [T, 256], z [T, 256] view, w [T, Hi] fp32)`."""
    d = core.dims
    rows = x.shape[0]
    p, hi = d.proxy_dim, d.idx_heads
    proj = cublas_mm(x, L.idx)
    q = cuda_tile_rms_norm(
        proj[:, : hi * p].reshape(rows * hi, p),
        L.idx_qn,
        core.eps,
        _NORM_PERSISTENT,
        _NORM_GATHER,
        _ZERO_CENTRED,
    ).view(rows, hi * p)
    k = layer_norm(proj[:, hi * p : hi * p + p].contiguous(), L.idx_kn_w, L.idx_kn_b, core.eps)
    flashinfer_apply_rope_with_cos_sin_cache(positions, q, k, p, core._rope[L.idx_rope_key], True)
    z = proj[:, hi * p + p : hi * p + 2 * p]
    weights = proj[:, hi * p + 2 * p : hi * p + 2 * p + hi].float() * core.idx_weight_scale
    return q.view(rows, hi, p), k, z, weights


def _store_kv(core, kv, qkv, req, positions, table) -> None:
    """Append this step's K/V (the padded heads of `qkv`) to a sparse layer's pool: a
    plain move over a bf16 or fp8 pool (fp8 taken unscaled, as the attention op does at
    unit KV scales), the quantizing store over an NVFP4 one."""
    d = core.dims
    rows = qkv.shape[0]
    heads = qkv.view(rows, d.heads + 2 * d.kv_heads, core.pad_dim)
    k = heads[:, d.heads : d.heads + d.kv_heads]
    v = heads[:, d.heads + d.kv_heads :]
    tpb = core._bound.tokens_per_block
    if kv[2] is None:
        qsa_paged_kv_store(k, v, kv[0], kv[1], req, positions, table, tpb)
        return
    qsa_paged_kv_store_nvfp4(
        k, v, kv[0], kv[1], kv[2], kv[3], req, positions, table, tpb, core.kv_scale_orig_quant
    )


def _sparse_attend(core, kv, qkv, sel, req, table, attn) -> None:
    """Sparse GQA over each KV group's selection, one call per group on the group's
    single-head pool view, written into the padded output `attn` `[T, H * 256]`."""
    d = core.dims
    rows = qkv.shape[0]
    hpg = d.heads // d.kv_heads
    heads = qkv.view(rows, d.heads + 2 * d.kv_heads, core.pad_dim)
    out = attn.view(rows, d.heads, core.pad_dim)
    tpb = core._bound.tokens_per_block
    for g in range(d.kv_heads):
        q = heads[:, g * hpg : (g + 1) * hpg]
        if kv[2] is None:
            res = qsa_paged_sparse_gqa(
                q=q,
                k_cache=kv[0][:, g : g + 1],
                v_cache=kv[1][:, g : g + 1],
                block_table=table,
                selected_tokens=sel[g],
                request_indices=req,
                tokens_per_block=tpb,
                softmax_scale=core.softmax_scale,
            )
        else:
            res = qsa_paged_sparse_gqa_nvfp4(
                q=q,
                k_cache=kv[0][:, g : g + 1],
                v_cache=kv[1][:, g : g + 1],
                k_scale_cache=kv[2][:, g : g + 1],
                v_scale_cache=kv[3][:, g : g + 1],
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                block_table=table,
                selected_tokens=sel[g],
                request_indices=req,
                tokens_per_block=tpb,
                softmax_scale=core.softmax_scale,
            )
        out[:, g * hpg : (g + 1) * hpg].copy_(res)


def _expand_regions(core, ids: torch.Tensor, dst: torch.Tensor) -> None:
    """Selected region ids `[G * n, K]` (`-1` unused) -> their tokens' logical positions,
    written into the region columns `dst` `[G, n, K * region]`."""
    offsets = torch.arange(core.region, device=ids.device, dtype=torch.int32)
    tokens = ids.unsqueeze(-1) * core.region + offsets
    tokens = torch.where(ids.unsqueeze(-1) >= 0, tokens, -1)
    dst.copy_(tokens.view(dst.shape))


def _slot_table(core, md, rows: int) -> torch.Tensor:
    """`[rows, blocks]` int32 slot ids of each request's full-attention blocks, from the
    engine's device block offsets; `-1` past each request's allocated blocks (columns the
    engine did not stage this step)."""
    b = core._bound
    offsets = md.kv_cache_block_offsets[b.slot_pool, :rows, 0]
    table = torch.div(offsets, b.slot_scale, rounding_mode="floor")
    blocks = torch.div(
        md.kv_lens_cuda_runtime[:rows] + (b.tokens_per_block - 1),
        b.tokens_per_block,
        rounding_mode="floor",
    )
    cols = torch.arange(table.shape[1], device=table.device, dtype=blocks.dtype)
    return torch.where(cols[None, :] < blocks[:, None], table, -1).to(torch.int32).contiguous()


def _step_args(md: TrtllmAttentionMetadata) -> dict:
    """The batch state every sliding-layer attention call of one step shares."""
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        workspace_=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        use_paged_context_fmha=md.use_paged_context_fmha,
        beam_width=md.effective_beam_width,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        max_context_q_len_override=md.max_context_q_len_override,
        is_cross=md.is_cross,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        use_spec_decoding=md.use_spec_decoding,
        is_spec_dec_tree=md.is_spec_dec_tree,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=md.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


class _PrefillPlan(NamedTuple):
    """The host-derived index state of one prefill step, shared by every sparse layer.

    `req` the batch row of every token; `table` the slot table; `kz_slot` / `kz_off`
    where each token's `[k | z]` row goes; `comp_slot` / `comp_off` `[R, 8]` where the
    tokens of each region this step completes sit, `sum_slot` / `sum_reg` where its
    summary goes (R may be 0); `tail` `[T, 8]` each token's own region's positions (`-1`
    past it); `lengths` `[T]` its complete history regions; `ctx` per context request
    `(request, first row, rows, columns)` with columns = its last position // 8; `gen` the first
    generation row (the step's last rows, one per generation request)."""

    req: torch.Tensor
    table: torch.Tensor
    kz_slot: torch.Tensor
    kz_off: torch.Tensor
    comp_slot: torch.Tensor | None
    comp_off: torch.Tensor | None
    sum_slot: torch.Tensor | None
    sum_reg: torch.Tensor | None
    tail: torch.Tensor
    lengths: torch.Tensor
    ctx: list
    gen: int
    gen_columns: int


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: Step5PreviewModelingV2) -> None:
        """Take the per-layer weight views this target's forward reads, once."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; each target owns its flag.
        self._contract_pending = step_contract_enabled()
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: calling `step_args` reads every metadata field
        this target consumes, so an upstream rename surfaces here rather than mid-forward."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The batch state the sliding-layer attention calls of one step share."""
        step = _step_args(md)
        step["num_contexts"] = md.num_contexts
        step["num_ctx_tokens"] = md.num_ctx_tokens
        return step

    def _plan(self, md, positions: torch.Tensor) -> _PrefillPlan:
        """The step's sparse-layer index state, from the host metadata."""
        core = self.core
        b = core._bound
        dev = positions.device
        tpb, rpb, region = b.tokens_per_block, b.regions_per_block, core.region
        num_seqs = md.num_seqs
        num_ctx = md.num_contexts
        new = [int(n) for n in md.seq_lens[:num_seqs]]
        total = [int(n) for n in md.kv_lens_runtime[:num_seqs]]
        cached = [t - n for t, n in zip(total, new)]

        req_host = torch.repeat_interleave(
            torch.arange(num_seqs, dtype=torch.int32), torch.tensor(new, dtype=torch.int64)
        )
        req = req_host.to(dev, non_blocking=True)
        table = _slot_table(core, md, num_seqs)
        pos_l = positions.long()
        kz_slot = table[req.long(), pos_l // tpb].long()
        kz_off = pos_l % tpb

        # The regions this step completes: those whose last token is among its new tokens.
        reg_req, reg_id = [], []
        for j in range(num_seqs):
            first, last = cached[j] // region, total[j] // region
            if last > first:
                reg_req.append(torch.full((last - first,), j, dtype=torch.int64))
                reg_id.append(torch.arange(first, last, dtype=torch.int64))
        comp_slot = comp_off = sum_slot = sum_reg = None
        if reg_req:
            rr = torch.cat(reg_req).to(dev, non_blocking=True)
            ri = torch.cat(reg_id).to(dev, non_blocking=True)
            src = ri[:, None] * region + torch.arange(region, device=dev)
            comp_slot = table[rr[:, None], src // tpb].long()
            comp_off = src % tpb
            sum_slot = table[rr, (ri * region) // tpb].long()
            sum_reg = ri % rpb

        base = (positions // region) * region
        tail = base[:, None] + torch.arange(region, device=dev, dtype=positions.dtype)
        tail = torch.where(tail <= positions[:, None], tail, -1).to(torch.int32)
        lengths = (positions // region).to(torch.int32)

        ctx, row = [], 0
        for j in range(num_ctx):
            ctx.append((j, row, new[j], (total[j] - 1) // region))
            row += new[j]
        return _PrefillPlan(
            req=req,
            table=table,
            kz_slot=kz_slot,
            kz_off=kz_off,
            comp_slot=comp_slot,
            comp_off=comp_off,
            sum_slot=sum_slot,
            sum_reg=sum_reg,
            tail=tail,
            lengths=lengths,
            ctx=ctx,
            gen=row,
            gen_columns=-(-md.max_seq_len // region),
        )

    def _sparse_layer(self, L: _Layer, x, qkv, attn, positions, plan: _PrefillPlan, sel):
        """A sparse layer's attention core over the step's rows, into `attn`."""
        core = self.core
        d = core.dims
        b = core._bound
        i = L.index
        rows = x.shape[0]
        kv = b.kv[i]
        _store_kv(core, kv, qkv, plan.req, positions, plan.table)
        q, k, z, w = _index_project(core, L, x, positions)
        kz = b.kz[i]
        kz[plan.kz_slot, plan.kz_off] = torch.stack((k, z), 1)
        if plan.comp_slot is not None:
            region_rows = kz[plan.comp_slot, plan.comp_off]
            b.summary[i][plan.sum_slot, plan.sum_reg] = csa_region_compress(
                region_rows[:, :, 0], region_rows[:, :, 1], torch.float8_e4m3fn
            )
        groups = d.kv_heads
        summary = b.summary[i]
        for j, first, count, columns in plan.ctx:
            dst = sel[:, first : first + count, core.region :]
            if columns == 0:
                dst.fill_(-1)
                continue
            last = first + count
            scores = csa_index_scores(
                q[first:last],
                w[first:last],
                summary,
                plan.table[j : j + 1],
                plan.req[first:last],
                positions[first:last],
                groups,
                core.region,
                columns,
            )
            ids = flashinfer_top_k_ragged_transform(
                scores,
                torch.zeros(groups * count, dtype=torch.int32, device=x.device),
                plan.lengths[first:last].repeat(groups),
                core.region_topk,
                _TOPK_DETERMINISTIC,
                _TOPK_TIE_BREAK,
                _TOPK_GRAPH_SAFE,
            )
            _expand_regions(core, ids, dst)
        if plan.gen < rows:
            gen_rows = rows - plan.gen
            scores = csa_index_scores(
                q[plan.gen :],
                w[plan.gen :],
                summary,
                plan.table,
                plan.req[plan.gen :],
                positions[plan.gen :],
                groups,
                core.region,
                plan.gen_columns,
            )
            ids = flashinfer_top_k_ragged_transform(
                scores,
                torch.zeros(groups * gen_rows, dtype=torch.int32, device=x.device),
                plan.lengths[plan.gen :].repeat(groups),
                core.region_topk,
                _TOPK_DETERMINISTIC,
                _TOPK_TIE_BREAK,
                _TOPK_GRAPH_SAFE,
            )
            _expand_regions(core, ids, sel[:, plan.gen :, core.region :])
        _sparse_attend(core, kv, qkv, sel, plan.req, plan.table, attn)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        d = core.dims
        step = self.step_args(md)

        h = (
            nn.functional.embedding(input_ids, core.w["embed"])
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        dev = h.device
        positions = position_ids.view(-1)
        pad_rows = core._dp_pad_rows(md, rows)
        plan = self._plan(md, positions)

        # One set of attention scratch for every layer: each is fully rewritten by layer
        # i's producer before its consumer reads it, all on one stream. The pad lanes of
        # `qkv` are written zero once and never again.
        qkv = torch.zeros(
            rows, (d.heads + 2 * d.kv_heads) * core.pad_dim, dtype=h.dtype, device=dev
        )
        attn = torch.empty(rows, d.heads * core.pad_dim, dtype=h.dtype, device=dev)
        sel = torch.empty(
            d.kv_heads, rows, core.region * (core.region_topk + 1), dtype=torch.int32, device=dev
        )
        sel[:, :, : core.region].copy_(plan.tail.unsqueeze(0).expand(d.kv_heads, rows, core.region))

        residual = h.float()
        x = cuda_tile_rms_norm(
            residual, self._norm0, core.eps, _NORM_PERSISTENT, _NORM_GATHER, _ZERO_CENTRED
        ).to(d.dtype)
        for L in self._layers:
            gate_logits = _project_qkv(core, L, x, qkv, positions)
            if L.sliding:
                thop_attention(
                    q=qkv,
                    output=attn,
                    local_layer_idx=core._bound.pool_row[L.index],
                    num_heads=d.heads,
                    num_kv_heads=d.kv_heads,
                    head_size=core.pad_dim,
                    quant_mode=core.quant_mode,
                    kv_scale_orig_quant=core.kv_scale_orig_quant,
                    kv_scale_quant_orig=core.kv_scale_quant_orig,
                    q_scaling=core.q_scaling,
                    attention_window_size=core.window,
                    **step,
                    **_CALL_CONSTANTS,
                )
            else:
                self._sparse_layer(L, x, qkv, attn, positions, plan, sel)
            o = _gate_out(core, L, attn, gate_logits)
            x = _add_norm(core, o, residual, L.norm2)
            if L.router is None:
                mlp = core._sum_bf16(_swiglu_mlp(x, L.gu, L.dn, L.limit)).float()
            else:
                mlp = _moe(core, L, x, pad_rows)
            if L.norm_next is None:
                residual.add_(mlp)
            else:
                x = _add_norm(core, mlp, residual, L.norm_next)
        return _final_norm(core, residual)


class DecodeTarget:
    """The specialization: routed to only when there are no context rows, one row per
    request."""

    def __init__(self, core: Step5PreviewModelingV2) -> None:
        """Take the per-layer weight views this target's forward reads, once."""
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: calling `step_args` reads every metadata field
        this target consumes, so an upstream rename surfaces here rather than mid-forward."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The batch state the sliding-layer attention calls of one step share. States the
        two context counts instead of reading them back: this target runs only when there
        are none."""
        step = _step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

    def _sparse_layer(self, L: _Layer, x, qkv, attn, positions, req, table, idx, sel):
        """A sparse layer's attention core over one row per request, into `attn`. Every
        row writes the summary of its own region from that region's tokens so far: the
        summary is complete (and read) only once the row holding the region's last token
        writes it, and a region not yet complete is never read."""
        core = self.core
        d = core.dims
        b = core._bound
        i = L.index
        rows = x.shape[0]
        kv = b.kv[i]
        kz_slot, kz_off, comp_slot, comp_off, sum_reg, lengths, columns = idx
        _store_kv(core, kv, qkv, req, positions, table)
        q, k, z, w = _index_project(core, L, x, positions)
        kz = b.kz[i]
        kz[kz_slot, kz_off] = torch.stack((k, z), 1)
        region_rows = kz[comp_slot, comp_off]
        b.summary[i][kz_slot, sum_reg] = csa_region_compress(
            region_rows[:, :, 0], region_rows[:, :, 1], torch.float8_e4m3fn
        )
        groups = d.kv_heads
        scores = csa_index_scores(
            q, w, b.summary[i], table, req, positions, groups, core.region, columns
        )
        ids = flashinfer_top_k_ragged_transform(
            scores,
            torch.zeros(groups * rows, dtype=torch.int32, device=x.device),
            lengths,
            core.region_topk,
            _TOPK_DETERMINISTIC,
            _TOPK_TIE_BREAK,
            _TOPK_GRAPH_SAFE,
        )
        _expand_regions(core, ids, sel[:, :, core.region :])
        _sparse_attend(core, kv, qkv, sel, req, table, attn)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        d = core.dims
        step = self.step_args(md)

        h = (
            nn.functional.embedding(input_ids, core.w["embed"])
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        dev = h.device
        positions = position_ids.view(-1)
        pad_rows = core._dp_pad_rows(md, rows)

        # The sparse layers' index state, the same for every one of them.
        b = core._bound
        tpb, rpb, region = b.tokens_per_block, b.regions_per_block, core.region
        req = torch.arange(rows, dtype=torch.int32, device=dev)
        table = _slot_table(core, md, rows)
        pos_l = positions.long()
        kz_slot = table[req.long(), pos_l // tpb].long()
        kz_off = pos_l % tpb
        src = (pos_l[:, None] - (region - 1) + torch.arange(region, device=dev)).clamp_min(0)
        comp_slot = table[req.long()[:, None], src // tpb].long()
        comp_off = src % tpb
        sum_reg = (pos_l // region) % rpb
        lengths = (positions // region).to(torch.int32).repeat(d.kv_heads)
        columns = -(-md.max_seq_len // region)
        idx = (kz_slot, kz_off, comp_slot, comp_off, sum_reg, lengths, columns)

        qkv = torch.zeros(
            rows, (d.heads + 2 * d.kv_heads) * core.pad_dim, dtype=h.dtype, device=dev
        )
        attn = torch.empty(rows, d.heads * core.pad_dim, dtype=h.dtype, device=dev)
        sel = torch.empty(
            d.kv_heads, rows, region * (core.region_topk + 1), dtype=torch.int32, device=dev
        )
        base = (positions // region) * region
        tail = base[:, None] + torch.arange(region, device=dev, dtype=positions.dtype)
        tail = torch.where(tail <= positions[:, None], tail, -1).to(torch.int32)
        sel[:, :, :region].copy_(tail.unsqueeze(0).expand(d.kv_heads, rows, region))

        residual = h.float()
        x = cuda_tile_rms_norm(
            residual, self._norm0, core.eps, _NORM_PERSISTENT, _NORM_GATHER, _ZERO_CENTRED
        ).to(d.dtype)
        for L in self._layers:
            gate_logits = _project_qkv(core, L, x, qkv, positions)
            if L.sliding:
                thop_attention(
                    q=qkv,
                    output=attn,
                    local_layer_idx=b.pool_row[L.index],
                    num_heads=d.heads,
                    num_kv_heads=d.kv_heads,
                    head_size=core.pad_dim,
                    quant_mode=core.quant_mode,
                    kv_scale_orig_quant=core.kv_scale_orig_quant,
                    kv_scale_quant_orig=core.kv_scale_quant_orig,
                    q_scaling=core.q_scaling,
                    attention_window_size=core.window,
                    **step,
                    **_CALL_CONSTANTS,
                )
            else:
                self._sparse_layer(L, x, qkv, attn, positions, req, table, idx, sel)
            o = _gate_out(core, L, attn, gate_logits)
            x = _add_norm(core, o, residual, L.norm2)
            if L.router is None:
                mlp = core._sum_bf16(_swiglu_mlp(x, L.gu, L.dn, L.limit)).float()
            else:
                mlp = _moe(core, L, x, pad_rows)
            if L.norm_next is None:
                residual.add_(mlp)
            else:
                x = _add_norm(core, mlp, residual, L.norm_next)
        return _final_norm(core, residual)


#: The keys under `multimodal_data["image"]` the input processor emits.
_IMAGE_KEYS = ("pixel_values", "patch_pixel_values", "num_patches")


class ModelingV2Step5PreviewNvfp4Sm103Unified(
    DecoderModelForCausalLM[Step5PreviewModelingV2, PretrainedConfig], MultimodalModelMixin
):
    """The shell: lm_head, the logits, and the images.

    `MultimodalModelMixin` is what makes the engine profile the vision tower
    (`encode_multimodal_inputs` over the input processor's `get_dummy_mm_data`) before it
    sizes the KV cache, and what its executor reads `multimodal_token_ids` and
    `multimodal_data_device_paths` off. The mixin's forward is not used: this shell's
    runs the tower itself and hands the targets the spliced embedding. Encoder-output
    caching and item scheduling stay off (the mixin's defaults)."""

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """The sparse layers' indexer state is a KV cache manager V2 buffer."""
        return "V2"

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config=None) -> bool:
        return True

    def __init__(self, model_config: ModelConfig):
        # Adopt the composite's text config as the model's config, in place: everything
        # that sizes the pools and the step reads `model_config.pretrained_config`, and
        # the composite carries the decoder's fields only under `text_config`. The core
        # keeps the composite.
        composite = model_config.pretrained_config
        text = composite.text_config
        model_config.pretrained_config = text
        super().__init__(
            Step5PreviewModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        self._image_token_id = int(composite.image_token_id)
        self._mm_token_ids: torch.Tensor | None = None

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()

    # ----------------------------------------------------------- multimodal ---

    @property
    def dtype(self) -> torch.dtype:
        """The activation dtype, in which the engine's profiler builds image tiles."""
        return self.model.dims.dtype

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """`<im_patch>`, the placeholder whose embeddings the vision rows replace. It is
        in vocabulary (128001 < 128896), so the executor's default `>= vocab_size` would
        select nothing. Built on first read (`__init__` runs under meta init)."""
        if self._mm_token_ids is None:
            self._mm_token_ids = torch.tensor([self._image_token_id], dtype=torch.int32)
        return self._mm_token_ids

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """The tensors under `multimodal_data` the executor moves to the device: the
        pixels. `num_patches` and the engine's bookkeeping stay on the host."""
        return ["image.pixel_values", "image.patch_pixel_values"]

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Guided decoding is not wired into this target."""
        return False

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Admission check of one request's image payload, so a malformed one fails that
        request instead of the step it would be batched into."""
        image = mm_data.get("image")
        if image is None:
            return
        d = self.model.dims
        overview, crop = d.v_grid * d.v_patch, d.v_crop_grid * d.v_patch
        pixels = image.get("pixel_values")
        counts = image.get("num_patches")
        crops = image.get("patch_pixel_values")
        if pixels is None or counts is None:
            raise ValueError(f"the image payload carries {sorted(image)}; it needs {_IMAGE_KEYS}")
        if tuple(pixels.shape[1:]) != (d.v_channels, overview, overview):
            raise ValueError(
                f"pixel_values is {tuple(pixels.shape)}; overviews are "
                f"[images, {d.v_channels}, {overview}, {overview}]"
            )
        counts = [int(n) for n in counts]
        if len(counts) != pixels.shape[0] or min(counts, default=0) < 0:
            raise ValueError(
                f"num_patches {counts} must hold one crop count per overview ({pixels.shape[0]})"
            )
        held = 0 if crops is None else crops.shape[0]
        if held != sum(counts) or (
            crops is not None and tuple(crops.shape[1:]) != (d.v_channels, crop, crop)
        ):
            raise ValueError(
                f"patch_pixel_values is {None if crops is None else tuple(crops.shape)}; "
                f"num_patches {counts} needs [{sum(counts)}, {d.v_channels}, {crop}, {crop}]"
            )
        cumsum = mm_data.get("multimodal_embed_mask_cumsum")
        if cumsum is not None:
            v = self.model.vision
            rows = len(counts) * v.out_rows[True] + sum(counts) * v.out_rows[False]
            if int(cumsum[-1]) != rows:
                raise ValueError(
                    f"the prompt holds {int(cumsum[-1])} <im_patch> tokens and its images "
                    f"encode to {rows} rows"
                )

    def encode_multimodal_inputs(self, multimodal_params) -> torch.Tensor:
        """The vision rows of every image the params carry, request by request, each
        request's in its prompt order."""
        rows = []
        for params in multimodal_params:
            image = params.multimodal_data["image"]
            rows.append(
                self.model.vision(
                    image["pixel_values"], image.get("patch_pixel_values"), image["num_patches"]
                )
            )
        return rows[0] if len(rows) == 1 else torch.cat(rows, 0)

    def _multimodal_embeds(self, kwargs) -> tuple[torch.Tensor, torch.Tensor] | None:
        """This step's vision rows and the packed-batch rows they replace, or None when it
        carries none.

        `multimodal_params` holds the context requests that carry media, in batch order
        (a subset of the context requests, so never zipped with them). Under chunked
        prefill or a reused prefix, `multimodal_runtime` says which of a request's rows
        fall in this chunk: the tower runs over the request's whole images and the
        chunk's slice is taken; a chunk with none of them skips the tower.
        `mm_token_indices` is the executor's list of the step's `<im_patch>` rows, in the
        same order."""
        rows = []
        for params in kwargs.get("multimodal_params") or ():
            if not (params.multimodal_data or {}).get("image"):
                continue
            runtime = params.multimodal_runtime
            if runtime is not None and runtime.num_mm_tokens_in_chunk == 0:
                continue
            embeds = self.encode_multimodal_inputs([params])
            if runtime is not None:
                total = int(runtime.total_embeds_in_request)
                assert total == embeds.shape[0], (
                    f"the prompt holds {total} <im_patch> tokens and its images encode to "
                    f"{embeds.shape[0]} rows"
                )
                start = int(runtime.num_cached_mm_tokens)
                embeds = embeds[start : start + int(runtime.num_mm_tokens_in_chunk)]
            rows.append(embeds)
        if not rows:
            return None
        vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
        mm_token_indices = kwargs.get("mm_token_indices")
        assert mm_token_indices is not None and mm_token_indices.shape[0] == vision.shape[0], (
            f"the step's <im_patch> rows "
            f"({None if mm_token_indices is None else mm_token_indices.shape[0]}) and its "
            f"vision rows ({vision.shape[0]}) disagree"
        )
        return vision, mm_token_indices

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """One engine step. The base forward drops `multimodal_params`, so the vision
        rows are spliced here: the token embedding with the step's `<im_patch>` rows
        replaced becomes the targets' `inputs_embeds`."""
        vision = self._multimodal_embeds(kwargs)
        if vision is not None:
            rows, mm_token_indices = vision
            inputs_embeds = nn.functional.embedding(input_ids, self.model.w["embed"])
            inputs_embeds.index_copy_(0, mm_token_indices, rows)
            input_ids = None
        return super().forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            lora_params,
            **kwargs,
        )
