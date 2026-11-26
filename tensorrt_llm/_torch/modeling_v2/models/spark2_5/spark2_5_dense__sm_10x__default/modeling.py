# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Spark-X2.5/X2.6 dense BF16 checkpoints / sm_10x / tp1.

Per decoder layer:

    x -> rmsnorm -> [q_k_v GEMM -> per-layer-type RoPE -> paged FMHA
         -> headwise sigmoid gate -> out GEMM] -> +residual, rmsnorm
      -> [gate_up GEMM -> erf-GELU-and-mul -> down GEMM] -> +residual, rmsnorm

Sliding layers rotate the whole head with theta 1e4; full layers rotate the
first `partial_rotary_factor` slice with theta 5e6 (`rope_parameters` in the
checkpoint config). The gate reads the same normalized hidden states as the
QKV projection, per the HF reference. The tied embedding is declared by the
core so the shell can tie lm_head to it; logits stay the shell's.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import (
    disable_v2_by_default,
    step_contract_enabled,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

from ..ops.activation.gelu_erf_and_mul import gelu_erf_and_mul
from ..ops.attention.headwise_output_gate import headwise_output_gate
from . import weights as _weights

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
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    # In-kernel RoPE is disabled here; the rope_* values below are inert
    # placeholders, not this model's rope config.
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


class _TiedEmbedding(nn.Module):
    """Embedding parameter with the attributes the shell's tie check reads.

    The shell compares `lm_head.tp_mode == embed_tokens.tp_mode`; lm_head
    carries `TensorParallelMode.COLUMN`, a str enum whose value is "column",
    so the plain string compares equal without importing a V1 module.
    """

    def __init__(self, weight: nn.Parameter, tp_size: int):
        super().__init__()
        self.weight = weight
        self.tp_mode = "column"
        self.tp_size = tp_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class _LayerWeights(NamedTuple):
    """One decoder layer's weights as the ops read them: `.t()` for the GEMM
    operands, the layer's RoPE constants, and the *next* layer's input norm
    (the final norm on the last layer) as `norm_next`."""

    qkv: torch.Tensor
    g: torch.Tensor
    o: torch.Tensor
    norm2: torch.Tensor
    fc: torch.Tensor
    down: torch.Tensor
    norm_next: torch.Tensor
    theta: float
    rotary_dim: int


class Spark2_5DenseModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        m = model_config.mapping

        # The layout facts routing does not read. Each one changes which rank
        # holds which rows or which tokens, so running through would compute
        # a plausible wrong answer rather than fail.
        assert m.world_size == 1 and m.tp_size == 1, "the target computes no collective"
        assert not m.enable_attention_dp, "attention DP is not implemented"
        assert m.pp_size == 1 and m.cp_size == 1, "pipeline/context parallelism not implemented"
        assert not getattr(cfg, "attention_bias", False) and not getattr(cfg, "mlp_bias", False), (
            "biased projections are not implemented"
        )
        assert cfg.gate_attn_act_mode == "sigmoid", (
            f"gate activation {cfg.gate_attn_act_mode!r} is not implemented"
        )
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo is None, f"unsupported KV algo {kv_algo}; bf16 pool only"

        layer_types = list(cfg.layer_types)
        self.sliding = [t == "sliding_attention" for t in layer_types]
        self.window = cfg.sliding_window

        # Per-layer-type RoPE, read once: sliding layers rotate the whole
        # head, full layers the first partial_rotary_factor slice.
        self.theta = {
            kind: float(cfg.get_rope_theta(kind))
            for kind in ("sliding_attention", "full_attention")
        }
        self.rotary_dim = {
            kind: int(cfg.head_dim * cfg.get_partial_rotary_factor(kind))
            for kind in ("sliding_attention", "full_attention")
        }
        self._layer_kind = layer_types

        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)

        self.embed_tokens = _TiedEmbedding(
            nn.Parameter(
                torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=model_config.torch_dtype),
                requires_grad=False,
            ),
            tp_size=m.tp_size,
        )

        self._step: "_Step | None" = None

    def post_load_weights(self) -> None:
        """Construct the step, now that the weights are real.

        Must only be called after meta init is over, never from `__init__`,
        where the shell's containers are still meta. Named for the shell hook
        that drives it; not an override -- DecoderModel has no such method.
        """
        self._step = _Step(self)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else."""
        assert self._step is not None, "post_load_weights has not run"
        return self._step.forward(attn_metadata, *args, **kwargs)


class _Step:
    """The one forward: context rows, generation rows, or a mix."""

    def __init__(self, core: Spark2_5DenseModelingV2) -> None:
        """Take the views and constants the forward reads, once, off the real
        weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here
        # rather than per forward.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype

        self._norm0 = w["l0_norm1"]
        self._layers = [
            _LayerWeights(
                qkv=w[f"l{i}_qkv"].t(),
                g=w[f"l{i}_g"].t(),
                o=w[f"l{i}_o"].t(),
                norm2=w[f"l{i}_norm2"],
                fc=w[f"l{i}_fc"].t(),
                down=w[f"l{i}_down"].t(),
                norm_next=(
                    w[f"l{i + 1}_norm1"] if i + 1 < cfg.num_hidden_layers else w["final_norm"]
                ),
                theta=core.theta[core._layer_kind[i]],
                rotary_dim=core.rotary_dim[core._layer_kind[i]],
            )
            for i in range(cfg.num_hidden_layers)
        ]
        # fused_qk_norm_rope requires valid q/k norm weight tensors even when
        # is_qk_norm=False, where the values are unused.
        self._no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE
        asks for it: the metadata fields this step consumes must exist (they
        are private trtllm surface). Calling `step_args` is the check."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch
        state, once per forward."""
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
            workspace=md.effective_workspace,
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
            spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
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
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        cfg = core.model_config.pretrained_config
        d = core.dims
        eps = cfg.rms_norm_eps

        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        # Full-attention layers take the no-window value; sliding layers take
        # the checkpoint's window. Both are host-derived per-capture constants.
        full_window = attn_metadata.max_seq_len
        no_qk_norm = self._no_qk_norm

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = F.embedding(input_ids, core.embed_tokens.weight)
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        attn_out = torch.empty([num_tokens, d.q_width], dtype=h.dtype, device=h.device)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            qkv = cublas_mm(x, L.qkv)
            qkv = fused_qk_norm_rope(
                qkv,
                num_heads_q=d.heads_q,
                num_heads_k=d.heads_kv,
                num_heads_v=d.heads_kv,
                head_dim=d.head_dim,
                rotary_dim=L.rotary_dim,
                eps=eps,
                q_weight=no_qk_norm,
                k_weight=no_qk_norm,
                base=L.theta,
                is_neox=True,
                position_ids=pos,
                factor=1.0,
                low=0.0,
                high=0.0,
                attention_factor=1.0,
                is_qk_norm=False,
            )
            # The gate reads the same normalized hidden states as the QKV
            # projection, per the HF reference.
            gate = cublas_mm(x, L.g)
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=i,
                num_heads=d.heads_q,
                num_kv_heads=d.heads_kv,
                head_size=d.head_dim,
                attention_sinks=None,
                attention_window_size=(core.window if core.sliding[i] else full_window),
                **step,
                **_CALL_CONSTANTS,
            )
            attn_out = headwise_output_gate(attn_out, gate, d.heads_q, d.head_dim)
            o = cublas_mm(attn_out, L.o)
            o, residual = flashinfer_fused_add_rmsnorm(o, residual, L.norm2, eps)
            fc = cublas_mm(o, L.fc)
            mlp = cublas_mm(gelu_erf_and_mul(fc), L.down)
            x, residual = flashinfer_fused_add_rmsnorm(mlp, residual, L.norm_next, eps)
        return x


@disable_v2_by_default
class ModelingV2Spark25DenseSm10xDefault(
    DecoderModelForCausalLM[Spark2_5DenseModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        # The Spark checkpoints declare "dtype" rather than "torch_dtype" on
        # older transformers; without this, lm_head would be sized fp32 while
        # every other tensor is bf16.
        if cfg.torch_dtype is None:
            cfg.torch_dtype = model_config.torch_dtype
        super().__init__(
            Spark2_5DenseModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
