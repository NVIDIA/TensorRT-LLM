# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / tp8ep2 -- self-contained modeling code.

A 108-block hybrid stack (48 Mamba2, 12 attention, 48 LatentMoE blocks, the order
read from the checkpoint's `layers_block_type`), every block pre-normed with its
own RMSNorm and added back to a bf16 residual stream. Every weight and every
activation is bf16 (the router's logits and the SSM state are fp32):

* Mamba2: `in_proj` -> depthwise causal conv + SiLU over x|B|C -> SSD scan
  (chunked on context rows, one recurrence step on generation rows) -> gated
  grouped RMSNorm (gate before the norm) -> `out_proj`.
* attention: GQA with no positional encoding at all, bf16 KV cache.
* LatentMoE: an fp32 sigmoid router with a selection-only correction bias over
  the full hidden state; the 512 relu2 experts run in a 2048-wide latent space
  between two projections; a relu2 shared expert runs on the full hidden state
  beside them.

tp8ep2: every block's output is a per-rank partial sum (mamba and attention by
head, the shared expert by its intermediate split, the routed experts by expert
window and intermediate split) completed by one all-reduce over the 8 ranks. The
routed experts are split two ways at once, as the engine's mapping lays out
moe_ep 2 x moe_tp 4: rank r runs the experts of window r % 2 (256 of the 512),
each on its quarter of the intermediate width (slice r // 2 of four 1280-wide
slices of 5120). Against tep8's eight 64-expert windows, a step's routed work is
spread over two windows, so the busiest rank's excess over the others -- what
they wait for at the MoE block's all-reduce -- shrinks; against splitting every
expert eight ways, a 1280-wide slice keeps the grouped GEMMs of a prefill step
efficient.

Each target is self-contained: it holds the per-layer weight views it reads,
passes every argument to every op at the call site, and shares no code with the
other targets.
"""

from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_fusion_allreduce import (  # noqa: E501
    FUSION_RESIDUAL_RMS_NORM,
    mnnvl_fusion_allreduce,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.flashinfer_direct_bf16_gemm import (  # noqa: E501
    flashinfer_direct_bf16_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import (
    ACTIVATION_RELU2,
    fused_moe,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_grouped import (
    rms_norm_gated_grouped,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_fwd import (
    causal_conv1d_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.flashinfer_selective_state_update import (  # noqa: E501
    flashinfer_selective_state_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    mamba_chunk_scan_combined,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.selective_state_update import (
    selective_state_update,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_allreduce_workspace,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger

from . import weights as _weights

# The certification is per GPU architecture: this code was gated on sm_100 only.
_SM = (10, 0)

# An empty slot in the recurrent-state pools (padding rows of a captured batch).
_PAD_SLOT_ID = -1
# Decode steps of up to this many rows run the K = 8192 projections (mamba in_proj, qkv,
# shared-expert up, fc1_latent) through FlashInfer's register-prefetch kernel instead of
# cublas_mm: 1.3-2.3x faster at one row on those shapes, still ahead at two, not from four
# rows on (see flashinfer_direct_bf16_gemm).
_LOWM_MAX_ROWS = 2
# The decode step runs FlashInfer's kernel from this many rows on and the Triton kernel below:
# at this rank's shapes FlashInfer's is the faster from 32 rows (1.17x there, 1.36x at 256, in
# CUDA-graph replay) and the slower below (0.87x at 8 rows, 0.73x at one). A mixed step's
# generation rows stay on the Triton kernel: FlashInfer's reads x, B and C through TMA, which
# needs them 128-byte aligned, and those rows start past the context tokens' in_proj rows
# (8768 bytes each), only 64-byte aligned after an odd number of them.
_FI_SSU_MIN_ROWS = 32
# Within one node AUTO picks the one-shot / two-shot IPC kernels for decode-sized payloads
# and NCCL for large prefill payloads (it falls back to NCCL past the workspace). Between
# nodes the IPC kernels have no peers to reach -- the workspace is never opened. On an NVL72
# system (aarch64, fabric handles) the group uses the MNNVL kernels, which fuse the residual
# add and the next block's RMSNorm into the all-reduce; anywhere else it runs NCCL. The user's
# `allreduce_strategy` decides whether MNNVL is a candidate at all, as it does for the engine's
# own AllReduce: AUTO and MNNVL take it where it applies, NCCL (or any other strategy) opts out.
_ALLREDUCE_STRATEGY = AllReduceStrategy.AUTO
# What a group runs where the IPC kernels have no peers (it spans nodes), or when asked for.
_ALLREDUCE_STRATEGY_NCCL = AllReduceStrategy.NCCL
_MNNVL_STRATEGIES = (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL)
# The MNNVL kernels record a buffer's size in a uint32, and the engine grows the workspace in
# 8 MiB steps: the largest call it can honour is the last multiple of 8 MiB below 2**32, and a
# call needing more is not taken.
_MNNVL_MAX_WORKSPACE_BYTES = 2**32 - 8 * 1024 * 1024
# The checkpoint's time_step_limit (0, inf): softplus(dt + dt_bias) is not clamped.
_DT_LIMIT = (0.0, float("inf"))

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
    q_scaling=1.0,
    # bf16 KV cache: no quantization of Q, K, V or the cache, so no scale tensors.
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    # No positional encoding: in-kernel RoPE is disabled; the rope_* values below are
    # inert placeholders.
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


class _Mamba(NamedTuple):
    """One Mamba2 block's weights, as the views the ops read."""

    in_w: torch.Tensor
    conv_w: torch.Tensor
    conv_b: torch.Tensor
    A: torch.Tensor
    A_hd: torch.Tensor
    D: torch.Tensor
    D_hd: torch.Tensor
    D_hd16: torch.Tensor
    dt_bias: torch.Tensor
    dt_bias_hd: torch.Tensor
    dt_bias_hd16: torch.Tensor
    gnorm: torch.Tensor
    out_w: torch.Tensor


class _Attention(NamedTuple):
    qkv: torch.Tensor
    o: torch.Tensor


class _Moe(NamedTuple):
    gate: torch.Tensor
    gate_bias: torch.Tensor
    fc1_lat: torch.Tensor
    fc2_lat: torch.Tensor
    sh_up: torch.Tensor
    sh_down: torch.Tensor
    fc1_w: torch.Tensor
    fc2_w: torch.Tensor


class _Bound(NamedTuple):
    """Everything taken off one KV cache manager: the attention layers' pool rows and the
    mamba layers' conv and SSM state pools."""

    cm: object
    pool_row: dict
    conv: dict
    ssm: dict


def _layer_views(core) -> list:
    """Per block, the views its forward reads, and the *next* block's input norm (the
    final norm after the last block) as the second element."""
    w = core.w
    d = core.dims
    hd = d.head_dim_ssm
    st = d.ssm_state
    views = []
    for i, kind in enumerate(d.kinds):
        if kind == "mamba":
            A = w[f"l{i}_A"]
            D = w[f"l{i}_D"]
            dtb = w[f"l{i}_dt_bias"]
            L = _Mamba(
                in_w=w[f"l{i}_in_proj_w"].t(),
                conv_w=w[f"l{i}_conv_w"],
                conv_b=w[f"l{i}_conv_b"],
                A=A,
                # The decode step reads per-(head, dim, state) A and per-(head, dim) D and
                # dt_bias; broadcast views, no copies.
                A_hd=A[:, None, None].expand(d.heads_r, hd, st),
                D=D,
                D_hd=D[:, None].expand(d.heads_r, hd),
                # FlashInfer's decode kernel takes D and dt_bias only in dt's dtype: the
                # checkpoint's bf16 values, which the fp32 copies hold exactly.
                D_hd16=D.to(torch.bfloat16)[:, None].expand(d.heads_r, hd),
                dt_bias=dtb,
                dt_bias_hd=dtb[:, None].expand(d.heads_r, hd),
                dt_bias_hd16=dtb.to(torch.bfloat16)[:, None].expand(d.heads_r, hd),
                gnorm=w[f"l{i}_gnorm"],
                out_w=w[f"l{i}_out_proj_w"].t(),
            )
        elif kind == "attention":
            L = _Attention(qkv=w[f"l{i}_qkv_w"].t(), o=w[f"l{i}_o_w"].t())
        else:
            assert kind == "moe", kind
            L = _Moe(
                gate=w[f"l{i}_gate_w"].t(),
                gate_bias=w[f"l{i}_gate_b"],
                fc1_lat=w[f"l{i}_fc1_lat_w"].t(),
                fc2_lat=w[f"l{i}_fc2_lat_w"].t(),
                sh_up=w[f"l{i}_sh_up_w"].t(),
                sh_down=w[f"l{i}_sh_down_w"].t(),
                fc1_w=w[f"l{i}_ex_fc1_w"],
                fc2_w=w[f"l{i}_ex_fc2_w"],
            )
        nxt = w[f"l{i + 1}_norm"] if i + 1 < d.num_layers else w["final_norm"]
        views.append((L, nxt))
    return views


class NemotronUltraBf16Tp8ep2ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)
        mapping = model_config.mapping
        self.tp_group = list(mapping.tp_group)
        self._bound: "_Bound | None" = None
        self._binds = 0
        # Set by post_load_weights, once the weights are real.
        self.mnnvl = False
        self.ar_strategy = _ALLREDUCE_STRATEGY
        self.ar_workspace = None
        self.side_stream: "torch.cuda.Stream | None" = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Construct the two targets, now that the weights are real.

        Called twice by the engine (the shell's hook, then the loader's module
        walk), so it rebuilds rather than accumulates. Must never run from
        `__init__`, where the containers are still meta.
        """
        # The IPC workspace the custom allreduce kernels use, allocated once outside any
        # capture (the same tensor the engine's own AllReduce modules hold). A group that
        # spans nodes has no IPC peers: it uses the MNNVL kernels where the system has them
        # (below) and NCCL, which needs no workspace, for the calls MNNVL does not take.
        mapping = self.model_config.mapping
        dtype = self.model_config.torch_dtype
        requested = self.model_config.allreduce_strategy
        self.mnnvl = requested in _MNNVL_STRATEGIES and MNNVLAllReduce.is_mnnvl(
            mapping, dtype, explicitly_requested=requested == AllReduceStrategy.MNNVL
        )
        if mapping.is_multi_node() or requested == AllReduceStrategy.NCCL:
            self.ar_strategy, self.ar_workspace = _ALLREDUCE_STRATEGY_NCCL, None
        else:
            self.ar_strategy = _ALLREDUCE_STRATEGY
            self.ar_workspace = get_allreduce_workspace(mapping)
        if self.mnnvl:
            # Created here, on every rank at the same point and outside any graph capture:
            # the setup is a collective, and a workspace cannot be created or grown while a
            # capture is recording. Later calls only ever grow it, eagerly (see mnnvl_buffers).
            try:
                get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)
            except RuntimeError as exc:
                # The setup ends in a fence that every rank leaves together, so a failure on
                # one rank is a failure on all of them and they all fall back to NCCL.
                logger.warning(f"MNNVL all-reduce workspace setup failed, using NCCL: {exc}")
                self.mnnvl = False
        if mapping.rank == 0:
            logger.info(
                f"nemotron3_ultra_bf16__sm_100__tp8ep2: all-reduce over {mapping.tp_size} ranks via "
                + (
                    "MNNVL, fused with the residual add and RMSNorm"
                    if self.mnnvl
                    else self.ar_strategy.name
                )
            )
        # Whether kernels that take the choice launch as programmatic dependents.
        self.pdl = get_env_enable_pdl()
        # The shared expert runs on this stream beside the routed experts.
        self.side_stream = torch.cuda.Stream()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def mnnvl_buffers(self, num_tokens: int) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """The MNNVL workspace's `(comm_buffer, buffer_flags)` for a call over `num_tokens` rows,
        or None when this group does not use MNNVL (or the call is past what the kernels size).

        Called once per forward, outside the layer loop. The workspace grows here, eagerly, when
        a call needs more than it holds. It never has to grow during a CUDA graph capture: the
        engine runs every shape it captures once eagerly first (the runner's warmup step), so a
        captured batch's growth has happened by the time the capture starts. A growth that fails
        (an allocation on any rank) raises on every rank, as in the engine's own AllReduce: only
        the first, load-time creation falls back to NCCL.
        """
        if not self.mnnvl:
            return None
        mapping = self.model_config.mapping
        dtype = self.model_config.torch_dtype
        need = MNNVLAllReduce.get_required_workspace_size(
            num_tokens, self.dims.hidden, mapping.tp_size, dtype
        )
        if need > _MNNVL_MAX_WORKSPACE_BYTES:
            return None
        workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype, buffer_size_bytes=need)
        return workspace["uc_buffer"].view(dtype).view(3, -1), workspace["buffer_flags"]

    def _bind_cache_manager(self, md) -> None:
        """Take every pool view this model reads from one KV cache manager.

        Called once per manager rather than once per model: with capacity
        estimation the engine builds a throwaway manager, warms up on it and then
        builds the real one under the same module, and a view held across the swap
        would dangle silently.
        """
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        self._binds += 1
        d = self.dims
        for name in ("get_conv_states", "get_ssm_states", "layer_offsets"):
            assert hasattr(cm, name), (
                f"{type(cm).__name__} has no {name}: the runtime allocated no hybrid "
                "recurrent-state pool for this model"
            )
        # thop_attention addresses the KV pool by the layer's pool row; counting
        # attention layers instead would alias layers with finite, plausible output.
        pool_row = {i: cm.layer_offsets[i] for i in d.attn_layers}
        conv = {i: cm.get_conv_states(i) for i in d.mamba_layers}
        ssm = {i: cm.get_ssm_states(i) for i in d.mamba_layers}
        c0, s0 = conv[d.mamba_layers[0]], ssm[d.mamba_layers[0]]
        assert c0.shape[1:] == (d.conv_r, d.conv_kernel - 1), (
            f"conv state pool {tuple(c0.shape)}: expected [slots, {d.conv_r}, {d.conv_kernel - 1}]"
        )
        assert s0.shape[1:] == (d.heads_r, d.head_dim_ssm, d.ssm_state), (
            f"ssm state pool {tuple(s0.shape)}: expected "
            f"[slots, {d.heads_r}, {d.head_dim_ssm}, {d.ssm_state}]"
        )
        # The pools hold what the runtime was configured with, not what this target was gated on:
        # `kv_cache_config.dtype` or `mamba_ssm_cache_dtype` can change them, and the kernels
        # would read and write bf16 KV (no scales, quant_mode 0) and an fp32 SSM state regardless.
        assert getattr(cm, "dtype", DataType.BF16) == DataType.BF16, (
            f"KV cache dtype {cm.dtype}: this target reads and writes a bf16 KV cache"
        )
        assert c0.dtype == torch.bfloat16 and s0.dtype == torch.float32, (
            f"conv state {c0.dtype} / ssm state {s0.dtype}: this target needs bf16 / fp32"
        )
        # The pool depth follows max_batch_size, and with it the largest slot offset the state
        # kernels index: past 2^31 elements from about 171 slots of this fp32 state on.
        if self.model_config.mapping.rank == 0:
            logger.info(
                f"nemotron3_ultra_bf16__sm_100__tp8ep2: recurrent-state pools of {s0.shape[0]} "
                f"slots, ssm slot stride {s0.stride(0)} elements ({s0.dtype}), conv slot stride "
                f"{c0.stride(0)} ({c0.dtype})"
            )
        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        self._bound = _Bound(cm, pool_row, conv, ssm)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """One host-side integer read: a mixed batch goes to prefill, so prefill is the
        general case and decode the specialization that may assume no context rows."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(self, attn_metadata: AttentionMetadata, *args, **kwargs) -> torch.Tensor:
        """Bind the step's cache manager if it changed, then route the step."""
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata)
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    """The general case: context rows, and possibly generation rows after them."""

    def __init__(self, core: NemotronUltraBf16Tp8ep2ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        self._eps = cfg.layer_norm_epsilon
        self._norm0 = core.w["l0_norm"]
        self._layers = _layer_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes
        must exist (they are private trtllm surface). Calling `step_args` and reading
        the mamba fields is the check."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
            mm = md.mamba_metadata
            for name in ("state_indices", "has_initial_states", "query_start_loc", "seq_idx"):
                getattr(mm, name)
            mm.use_initial_states, mm.chunk_size  # noqa: B018 — read for the AttributeError
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
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
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
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

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        d = core.dims
        md = attn_metadata
        eps = self._eps
        bound = core._bound
        group = core.tp_group

        self._check_step_contract(md)
        step = self.step_args(md)
        window = md.max_seq_len

        # Host-side batch split: context rows first, then single-token generation rows.
        nc = md.num_contexts
        nct = md.num_ctx_tokens
        ns = md.num_seqs
        mm = md.mamba_metadata
        si = mm.state_indices
        qsl = mm.query_start_loc
        hi = mm.has_initial_states
        use_init = mm.use_initial_states
        seq_idx = mm.seq_idx
        chunk_indices = mm.chunk_indices
        chunk_offsets = mm.chunk_offsets
        chunk = mm.chunk_size
        si_ctx = si[:nc]
        si_gen = si[nc:ns]
        has_gen = ns > nc

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        attn_out = torch.empty([num_tokens, d.q_width_r], dtype=h.dtype, device=h.device)
        mnnvl = core.mnnvl_buffers(num_tokens)

        din, conv_c = d.d_inner_r, d.conv_r
        heads, hd, gn = d.heads_r, d.head_dim_ssm, d.gn_r
        groups, st = d.groups_r, d.ssm_state

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, (L, norm_next) in enumerate(self._layers):
            if isinstance(L, _Mamba):
                conv_state = bound.conv[i]
                ssm_state = bound.ssm[i]
                zxbcdt = cublas_mm(x, L.in_w)
                z = zxbcdt[:, :din]
                xbc = zxbcdt[:, din : din + conv_c]
                dt = zxbcdt[:, din + conv_c :]
                y = torch.empty([num_tokens, heads, hd], dtype=h.dtype, device=h.device)

                # Context rows: varlen conv (seeding from and writing back the conv
                # state), then the chunked scan seeded from the SSM state where the
                # sequence has one, with each sequence's final state written back.
                # Channel-last views on both sides: the result is token-major
                # [nct, x|B|C], so x, B and C are column views of it.
                xbc_ctx = causal_conv1d_fwd(
                    xbc[:nct].t(),
                    L.conv_w,
                    L.conv_b,
                    conv_state,
                    qsl[: nc + 1],
                    si_ctx,
                    hi[:nc],
                    True,
                    _PAD_SLOT_ID,
                    torch.empty([nct, conv_c], dtype=h.dtype, device=h.device).t(),
                ).t()
                xs = xbc_ctx[:, :din].reshape(1, nct, heads, hd)
                Bm = xbc_ctx[:, din : din + gn].reshape(1, nct, groups, st)
                Cm = xbc_ctx[:, din + gn :].reshape(1, nct, groups, st)
                # A fresh row's slot still holds the previous request's state, which
                # the scan cannot tell apart from a carried one: zero it here.
                init = None
                if use_init:
                    init = ssm_state[si_ctx].to(torch.float32)
                    init.masked_fill_(~hi[:nc].view(nc, 1, 1, 1), 0)
                final = mamba_chunk_scan_combined(
                    xs,
                    dt[:nct].unsqueeze(0),
                    L.A,
                    Bm,
                    Cm,
                    chunk,
                    L.D,
                    None,
                    L.dt_bias,
                    init,
                    seq_idx,
                    chunk_indices,
                    chunk_offsets,
                    qsl[: nc + 1],
                    True,
                    _DT_LIMIT,
                    y[:nct].view(1, nct, heads, hd),
                    False,
                    True,
                    torch.float32,
                )
                ssm_state[si_ctx] = final.to(ssm_state.dtype)

                # Generation rows: one conv step and one recurrence step each.
                if has_gen:
                    # Activates the generation rows of xBC in place inside zxbcdt.
                    xbc_gen = causal_conv1d_update(
                        xbc[nct:], conv_state, L.conv_w, L.conv_b, True, None, si_gen, _PAD_SLOT_ID
                    )
                    ng = num_tokens - nct
                    selective_state_update(
                        ssm_state,
                        xbc_gen[:, :din].reshape(ng, heads, hd),
                        dt[nct:][:, :, None].expand(ng, heads, hd),
                        L.A_hd,
                        xbc_gen[:, din : din + gn].reshape(ng, groups, st),
                        xbc_gen[:, din + gn :].reshape(ng, groups, st),
                        D=L.D_hd,
                        z=None,
                        dt_bias=L.dt_bias_hd,
                        dt_softplus=True,
                        state_batch_indices=si_gen,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y[nct:],
                    )

                yn = rms_norm_gated_grouped(
                    y.reshape(num_tokens, din), z, L.gnorm, eps, d.norm_group
                )
                out = cublas_mm(yn, L.out_w)
            elif isinstance(L, _Attention):
                qkv = cublas_mm(x, L.qkv)
                attn_out = thop_attention(
                    q=qkv,
                    output=attn_out,
                    local_layer_idx=bound.pool_row[i],
                    num_heads=d.q_heads_r,
                    num_kv_heads=1,
                    head_size=d.head_dim,
                    attention_window_size=window,
                    **step,
                    **_CALL_CONSTANTS,
                )
                out = cublas_mm(attn_out, L.o)
            else:
                # The shared expert reads only x: run it on the side stream while the
                # routed experts run on the main one, joined before the sum.
                side = core.side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    u = cublas_mm(x, L.sh_up)
                    a = torch.square(torch.relu(u))
                    shared = cublas_mm(a, L.sh_down)
                # Router: fp32 logits over the full hidden state, selection on sigmoid +
                # correction bias, weights from the unbiased sigmoid, renormalized and scaled.
                logits = cublas_mm(x, L.gate, out_dtype=torch.float32)
                topk_w, topk_ids = noaux_tc_op(logits, L.gate_bias, 1, 1, d.top_k, d.routed_scale)
                # Routed experts in the latent space: every rank routes the same tokens over
                # all 512 experts and runs the ones in its 256-expert window, each on its
                # quarter of the intermediate width (FC1 rows, FC2 columns), so each rank's
                # output is a partial sum over its window and its slice of the width.
                xl = cublas_mm(x, L.fc1_lat)
                routed = fused_moe(
                    xl,
                    topk_ids,
                    topk_w,
                    L.fc1_w,
                    None,
                    L.fc2_w,
                    None,
                    torch.bfloat16,
                    [],
                    tp_size=d.moe_tp,
                    tp_rank=d.moe_tp_rank,
                    ep_size=d.ep_size,
                    ep_rank=d.ep_rank,
                    activation_type=ACTIVATION_RELU2,
                )[0]
                out = cublas_mm(routed, L.fc2_lat)
                main.wait_stream(side)
                out = out + shared
            if mnnvl is None:
                out = allreduce(out, group, core.ar_strategy, core.ar_workspace)
                x, residual = flashinfer_fused_add_rmsnorm(out, residual, norm_next, eps)
            else:
                # One op: the sum over the group, the residual add and the next block's
                # RMSNorm (a single kernel up to 1 MiB of payload, two kernels above).
                x, residual = mnnvl_fusion_allreduce(
                    out,
                    norm_next,
                    residual,
                    eps,
                    mnnvl[0],
                    mnnvl[1],
                    True,
                    None,
                    FUSION_RESIDUAL_RMS_NORM,
                )
        return x


class DecodeTarget:
    """The specialization: routed to only when there are no context rows."""

    def __init__(self, core: NemotronUltraBf16Tp8ep2ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        self._eps = cfg.layer_norm_epsilon
        self._norm0 = core.w["l0_norm"]
        self._layers = _layer_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes
        must exist (they are private trtllm surface)."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
            md.mamba_metadata.state_indices  # noqa: B018 — read for the AttributeError
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
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
            num_contexts=0,
            num_ctx_tokens=0,
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

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        d = core.dims
        md = attn_metadata
        eps = self._eps
        bound = core._bound
        group = core.tp_group

        self._check_step_contract(md)
        step = self.step_args(md)
        window = md.max_seq_len

        # Every row is one generation token of its own sequence.
        ns = md.num_seqs
        si = md.mamba_metadata.state_indices[:ns]

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        attn_out = torch.empty([num_tokens, d.q_width_r], dtype=h.dtype, device=h.device)
        mnnvl = core.mnnvl_buffers(num_tokens)

        din, conv_c = d.d_inner_r, d.conv_r
        heads, hd, gn = d.heads_r, d.head_dim_ssm, d.gn_r
        groups, st = d.groups_r, d.ssm_state

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, (L, norm_next) in enumerate(self._layers):
            if isinstance(L, _Mamba):
                conv_state = bound.conv[i]
                ssm_state = bound.ssm[i]
                if num_tokens <= _LOWM_MAX_ROWS:
                    zxbcdt = flashinfer_direct_bf16_gemm(x, L.in_w, core.pdl)
                else:
                    zxbcdt = cublas_mm(x, L.in_w)
                z = zxbcdt[:, :din]
                xbc = zxbcdt[:, din : din + conv_c]
                dt = zxbcdt[:, din + conv_c :]
                # Activates xBC in place inside zxbcdt.
                xbc = causal_conv1d_update(
                    xbc, conv_state, L.conv_w, L.conv_b, True, None, si, _PAD_SLOT_ID
                )
                y = torch.empty([num_tokens, heads, hd], dtype=h.dtype, device=h.device)
                if num_tokens >= _FI_SSU_MIN_ROWS:
                    flashinfer_selective_state_update(
                        ssm_state,
                        xbc[:, :din].reshape(num_tokens, heads, hd),
                        dt[:, :, None].expand(num_tokens, heads, hd),
                        L.A_hd,
                        xbc[:, din : din + gn].reshape(num_tokens, groups, st),
                        xbc[:, din + gn :].reshape(num_tokens, groups, st),
                        D=L.D_hd16,
                        z=None,
                        dt_bias=L.dt_bias_hd16,
                        dt_softplus=True,
                        state_batch_indices=si,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y,
                    )
                else:
                    selective_state_update(
                        ssm_state,
                        xbc[:, :din].reshape(num_tokens, heads, hd),
                        dt[:, :, None].expand(num_tokens, heads, hd),
                        L.A_hd,
                        xbc[:, din : din + gn].reshape(num_tokens, groups, st),
                        xbc[:, din + gn :].reshape(num_tokens, groups, st),
                        D=L.D_hd,
                        z=None,
                        dt_bias=L.dt_bias_hd,
                        dt_softplus=True,
                        state_batch_indices=si,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y,
                    )
                yn = rms_norm_gated_grouped(
                    y.reshape(num_tokens, din), z, L.gnorm, eps, d.norm_group
                )
                out = cublas_mm(yn, L.out_w)
            elif isinstance(L, _Attention):
                if num_tokens <= _LOWM_MAX_ROWS:
                    qkv = flashinfer_direct_bf16_gemm(x, L.qkv, core.pdl)
                else:
                    qkv = cublas_mm(x, L.qkv)
                attn_out = thop_attention(
                    q=qkv,
                    output=attn_out,
                    local_layer_idx=bound.pool_row[i],
                    num_heads=d.q_heads_r,
                    num_kv_heads=1,
                    head_size=d.head_dim,
                    attention_window_size=window,
                    **step,
                    **_CALL_CONSTANTS,
                )
                out = cublas_mm(attn_out, L.o)
            else:
                # The shared expert reads only x: run it on the side stream while the
                # routed experts run on the main one, joined before the sum.
                side = core.side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    if num_tokens <= _LOWM_MAX_ROWS:
                        u = flashinfer_direct_bf16_gemm(x, L.sh_up, core.pdl)
                    else:
                        u = cublas_mm(x, L.sh_up)
                    a = torch.square(torch.relu(u))
                    shared = cublas_mm(a, L.sh_down)
                logits = cublas_mm(x, L.gate, out_dtype=torch.float32)
                topk_w, topk_ids = noaux_tc_op(logits, L.gate_bias, 1, 1, d.top_k, d.routed_scale)
                if num_tokens <= _LOWM_MAX_ROWS:
                    xl = flashinfer_direct_bf16_gemm(x, L.fc1_lat, core.pdl)
                else:
                    xl = cublas_mm(x, L.fc1_lat)
                routed = fused_moe(
                    xl,
                    topk_ids,
                    topk_w,
                    L.fc1_w,
                    None,
                    L.fc2_w,
                    None,
                    torch.bfloat16,
                    [],
                    tp_size=d.moe_tp,
                    tp_rank=d.moe_tp_rank,
                    ep_size=d.ep_size,
                    ep_rank=d.ep_rank,
                    activation_type=ACTIVATION_RELU2,
                )[0]
                out = cublas_mm(routed, L.fc2_lat)
                main.wait_stream(side)
                out = out + shared
            if mnnvl is None:
                out = allreduce(out, group, core.ar_strategy, core.ar_workspace)
                x, residual = flashinfer_fused_add_rmsnorm(out, residual, norm_next, eps)
            else:
                # One op: the sum over the group, the residual add and the next block's
                # RMSNorm (a single kernel up to 1 MiB of payload, two kernels above).
                x, residual = mnnvl_fusion_allreduce(
                    out,
                    norm_next,
                    residual,
                    eps,
                    mnnvl[0],
                    mnnvl[1],
                    True,
                    None,
                    FUSION_RESIDUAL_RMS_NORM,
                )
        return x


class ModelingV2Nemotron3UltraBf16Sm100Tp8ep2(
    DecoderModelForCausalLM[NemotronUltraBf16Tp8ep2ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cap = torch.cuda.get_device_capability()
        assert cap == _SM, (
            f"this target is certified on sm_{_SM[0]}{_SM[1]} only; the device is "
            f"sm_{cap[0]}{cap[1]}"
        )
        cfg = model_config.pretrained_config
        super().__init__(
            NemotronUltraBf16Tp8ep2ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
