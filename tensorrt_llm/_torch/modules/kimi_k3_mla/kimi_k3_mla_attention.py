# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, gated
output projection, and the fused FP4 decode query of Helix generation on SM107.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Optional

import torch

from ...._utils import get_sm_version
from ....functional import PositionEmbeddingType
from ....mapping import Mapping
from ....quantization import QuantAlgo
from ...attention.backends import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
)
from ...attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ...attention.mla import MLA, _helix_gen_zero_kv_mask
from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ...distributed import AllReduceParams
from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from ...model_config import ModelConfig
from ...utils import AuxStreamType
from ..linear import Linear, TensorParallelMode
from ..multi_stream_utils import do_multi_stream
from ..projection_capabilities import GatedEpilogueProjection, PrequantizedInputProjection
from ..rms_norm import RMSNorm
from .helix_tail import K3HelixTail


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite a rotary cos/sin table with identity values in place.

    Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    Setting cos=1 and sin=0 per position makes the rotation the
    identity — a mathematical no-op — which preserves K3's NoPE
    semantics without patching the backend.
    """
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Install an identity rotary cos/sin table on ``backend``.

    The C++ MLA rope kernels (``mla_rope_generation`` and the context
    preprocess) read this table and apply the rotation; identity values
    make that a copy, preserving K3's NoPE.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. ``_ensure_rope_table_size`` is replaced
    with an identity-preserving resize: the table may GROW (so the
    fused rope-generation op can never index out of bounds for long
    sequences) but its values are always rewritten to identity right
    after a regeneration, so the real sinusoids never leak in.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    _write_identity_rope_values(cos_sin)

    orig_resize = backend._ensure_rope_table_size  # bound method

    def _identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        orig_resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = _identity_preserving_resize


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(MLA):
    """Kimi K3 MLA implemented as a thin specialization of :class:`MLA`.

    K3 keeps the standard dense MLA attention/cache flow and only changes the
    checkpoint projection topology, positional encoding, KV-B runtime layout,
    and gated output projection.
    """

    # Whether generation-only Helix steps of up to k3_mla_q_fused.MAX_TOKENS tokens take
    # the fused FP4 decode query (_forward_generation_fused_query); set in
    # cache_derived_state, as are the two flags below.
    _fused_decode_query = False
    # The fused query kernel applies the q_a RMSNorm itself.
    _fused_decode_query_norm = False
    # The KV half writes the latent row (kv_a RMSNorm + k_pe) in one launch.
    _fused_latent_row = False
    # (q_b weight scale, its block exponents or None), cached on first use.
    _q_b_block_exponents: Optional[tuple] = None
    # Largest generation batch whose Helix tail runs as one kernel (K3HelixTail); 0: never.
    _helix_tail_max_tokens: int = 0
    # Set for one generation-only forward of a fused-tail batch: the output gate, forked when the
    # forward starts (beside q_a / q_b) and joined before the tail, and whether it is still to join.
    _helix_tail_gate: Optional[torch.Tensor] = None
    _helix_tail_gate_pending: bool = False
    # Set for one such forward that takes the fused decode query, which forks the gate behind
    # its KV half on the aux stream instead (_forward_generation_fused_query).
    _helix_tail_gate_behind_kv: bool = False
    # Whether o_proj takes a prequantized input (PrequantizedInputProjection); set in
    # cache_derived_state. The tail then hands it the MXFP8 form of the gated rows, set for one
    # forward in _helix_tail_operand.
    _o_proj_takes_prequantized: bool = False
    _helix_tail_operand: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
        projection_configs = model_config.quant_config_dict or {}
        q_a_config = projection_configs.get("q_a_proj", model_config.quant_config)
        kv_a_config = projection_configs.get("kv_a_proj_with_mqa", model_config.quant_config)
        q_a_algo = q_a_config.quant_algo if q_a_config else None
        kv_a_algo = kv_a_config.quant_algo if kv_a_config else None
        fuse_qkv_a_proj = q_a_algo == kv_a_algo and (
            q_a_algo is None or (q_a_algo == QuantAlgo.FP8_BLOCK_SCALES and q_lora_rank % 128 == 0)
        )
        # Resolve each Linear's recipe before allocating weights, as in the
        # model's layerwise quantization post-init. Shared MLA stays generic.
        deferred_config = copy.copy(model_config)
        deferred_config._frozen = False
        deferred_config.skip_create_weights_in_init = True
        deferred_config._frozen = model_config._frozen
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=False,
            config=deferred_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            reduce_output=False,
            fuse_qkv_a_proj=fuse_qkv_a_proj,
            rms_norm_eps=rms_norm_eps,
        )
        # Keep the base MLA registration enabled so breakable CUDA graphs use
        # the shared custom op. The output gate is a base hook and runs on both
        # the registered and eager paths before the row-parallel o_proj.

        self.use_output_gate = use_output_gate
        if use_output_gate:
            if self.mapping.has_cp_helix():
                # The gate GEMM runs beside the Helix all-to-all (_output_gate), or beside the
                # q_a / q_b GEMMs of a fused-tail batch (forward).
                self.helix_exchange_event = torch.cuda.Event()
                self._gate_ready = torch.cuda.Event()
                self._tail_gate_fork = torch.cuda.Event()

        if use_output_gate:
            # The gate must match o_proj's input sharding (under helix the
            # post-all-to-all 1/cp head chunk); outside helix this equals
            # q_b_proj's head sharding, replicated under attention-DP.
            self.g_proj = Linear(
                hidden_size,
                num_heads * v_head_dim,
                bias=False,
                dtype=dtype,
                mapping=self.o_proj.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=(model_config.quant_config_dict or {}).get(
                    "g_proj", model_config.quant_config
                ),
                skip_create_weights_in_init=True,
                allreduce_strategy=model_config.allreduce_strategy,
                force_dynamic_quantization=model_config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

        for name, module in self.named_children():
            if isinstance(module, Linear):
                module.quant_config = projection_configs.get(name, model_config.quant_config)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
            for module in self.children():
                if isinstance(module, Linear):
                    module.create_weights()

        # K3 is NoPE. The base MLA backends still require real RoPE tables, so
        # retain their expected shape and replace every rotation with identity.
        assert isinstance(self.mha, TrtllmAttention)
        assert isinstance(self.mqa, TrtllmAttention)
        _install_identity_rope_table(self.mha)
        _install_identity_rope_table(self.mqa)
        self.rotary_emb = None
        self.apply_rotary_emb = False
        # Whether g_proj and o_proj fuse the gate into their GEMMs
        # (GatedEpilogueProjection) in place of _project_output; set once the
        # loaded projections are known (cache_derived_state). Not under Helix +
        # attention DP, whose output projection reduce-scatters.
        self._fused_output_gate = False
        # Whether the fused q_a / kv_a projection can consume a prequantized
        # input (PrequantizedInputProjection); set in cache_derived_state.
        self._qkv_a_takes_prequantized = False

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self._fused_output_gate = (
            self.use_output_gate
            and not (self.mapping.has_cp_helix() and self.mapping.enable_attention_dp)
            and isinstance(self.g_proj, GatedEpilogueProjection)
            and isinstance(self.o_proj, GatedEpilogueProjection)
        )
        self._qkv_a_takes_prequantized = self.fuse_qkv_a_proj and isinstance(
            self.kv_a_proj_with_mqa, PrequantizedInputProjection
        )
        self._fused_decode_query = self._supports_fused_decode_query()
        self._fused_decode_query_norm = (
            self._fused_decode_query
            and self._resolve_qa_fused_scale() is None
            and self._runs_flashinfer_cute_rmsnorm(self.q_a_layernorm)
        )
        self._fused_latent_row = self._fused_decode_query and self._runs_flashinfer_cute_rmsnorm(
            self.kv_a_layernorm
        )
        self._q_b_block_exponents = None

        self._o_proj_takes_prequantized = isinstance(self.o_proj, PrequantizedInputProjection)
        self._helix_tail_max_tokens = 0
        if self._helix_v_up_before_exchange and self.helix_v_b_proj_full.is_contiguous():
            self._helix_tail_max_tokens = K3HelixTail.max_tokens(
                self.mapping, self.num_heads_tp, self.kv_lora_rank, self.v_head_dim
            )

    @staticmethod
    def _runs_flashinfer_cute_rmsnorm(norm: torch.nn.Module) -> bool:
        """Whether ``norm`` of BF16 rows runs FlashInfer's CuTe DSL RMSNorm, the arithmetic
        the fused K3 MLA kernels repeat bit for bit."""
        if not (
            IS_FLASHINFER_AVAILABLE
            and type(norm) is RMSNorm
            and not (norm.use_cuda_tile or norm.use_gemma or norm.is_nvfp4)
            and not norm.return_hp_output
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.is_contiguous()
        ):
            return False
        try:
            from flashinfer.norm import _use_cuda_norm
        except ImportError:
            return False
        return not _use_cuda_norm(norm.weight.device)

    def _absorption_weight_bf16(self) -> Optional[torch.Tensor]:
        """The BF16 W_UK ``[H, kv_lora_rank, qk_nope_head_dim]`` the absorbed generation BMM
        reads (the module weight, or its dequantized copy)."""
        if self.k_b_proj_trans.dtype == torch.bfloat16:
            return self.k_b_proj_trans
        return self.k_b_proj_trans_dequant

    def _supports_fused_decode_query(self) -> bool:
        """Whether Helix generation of this FP4-KV layer can take the fused decode query: the
        SM107 kernel's head geometry, the V up-projection before the all-to-all, and FP8 q_b
        with a BF16 W_UK that the default generation path reads too."""
        if not (
            IS_CUTLASS_DSL_AVAILABLE
            and get_sm_version() == 107
            and self.mapping.has_cp_helix()
            and self._helix_v_up_before_exchange
            and self.mqa.uses_fp4_mla_attention
            and self.sparse_attn_hooks is None
            and self.fuse_qkv_a_proj
            and not self.is_lite
            and not self.apply_rotary_emb
            and not self.llama_4_scaling
        ):
            return False
        from ...kernels.blackwell import k3_mla_q_fused

        w_uk = self._absorption_weight_bf16()
        return (
            (
                self.num_heads_tp,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            )
            == (
                k3_mla_q_fused.NUM_HEADS,
                k3_mla_q_fused.Q_LORA_RANK,
                k3_mla_q_fused.KV_LORA_RANK,
                k3_mla_q_fused.NOPE_DIM,
                k3_mla_q_fused.ROPE_DIM,
            )
            and w_uk is not None
            and k3_mla_q_fused.weights_supported(self.q_b_proj.weight, w_uk)
        )

    def _q_b_exponents(self) -> Optional[torch.Tensor]:
        """Block exponents of the q_b weight (``k3_mla_q_fused.block_exponents``), cached per
        weight scale; None for a scale layout the kernel does not read, or while capturing a
        CUDA graph before the first eager step built them."""
        scale = self.q_b_proj.weight_scale
        cached = self._q_b_block_exponents
        if cached is not None and cached[0] is scale:
            return cached[1]
        if torch.cuda.is_current_stream_capturing():
            return None
        from ...kernels.blackwell import k3_mla_q_fused

        try:
            exponents = k3_mla_q_fused.block_exponents(scale, *self.q_b_proj.weight.shape)
        except ValueError:
            exponents = None
        self._q_b_block_exponents = (scale, exponents)
        return exponents

    def _takes_fused_decode_query(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache_gen: Optional[torch.Tensor],
    ) -> bool:
        if not self._fused_decode_query or latent_cache_gen is not None:
            return False
        from ...kernels.blackwell import k3_mla_q_fused

        num_tokens = attn_metadata.num_tokens
        return (
            type(hidden_states) is torch.Tensor
            and attn_metadata.num_contexts == 0
            and 0 < num_tokens <= k3_mla_q_fused.MAX_TOKENS
            and num_tokens % attn_metadata.num_generations == 0
            and attn_metadata.helix_position_offsets is not None
            and attn_metadata.helix_is_inactive_rank is not None
            and self._q_b_exponents() is not None
        )

    def _forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
        prequantized_input: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if self._takes_fused_decode_query(hidden_states, attn_metadata, latent_cache_gen):
            self._forward_generation_fused_query(
                hidden_states, attn_metadata, output, prequantized_input
            )
            return
        super()._forward_impl(
            position_ids,
            hidden_states,
            attn_metadata,
            output,
            latent_cache_gen=latent_cache_gen,
            prequantized_input=prequantized_input,
        )

    def _forward_generation_fused_query(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        prequantized_input: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Absorbed Helix generation of an FP4-KV layer with the fused decode query.

        After the fused q_a / kv_a projection the KV half (the latent row, then the FP4
        generation cache update without its query rows) runs on the aux stream, and the main
        stream runs ``k3_mla_q_fused`` (q_a RMSNorm, q_b, W_UK, RoPE and the FP4 query
        quantization), which writes the query rows into the Q buffers the cache update records
        on the metadata. The decode joins both; the BF16 query it is handed provides shapes.
        """
        from ...attention.backends.fp4_mla.cache_update import _prepare_fp4_mla_q_buffers
        from ...attention.backends.fp4_mla.config import _fp4_mla_packed_rows_per_seq
        from ...attention.backends.fp4_mla.layout import _get_fp4_mla_q_global_scale
        from ...kernels.blackwell import k3_mla_q_fused

        md = attn_metadata
        num_tokens = md.num_tokens
        heads, lora, rope = self.num_heads_tp, self.kv_lora_rank, self.qk_rope_head_dim
        hidden_states = hidden_states[:num_tokens]
        device = hidden_states.device
        gen_len = num_tokens // md.num_generations
        # Create the Q buffers (zero-filled, once) on this stream, which writes them.
        _prepare_fp4_mla_q_buffers(
            md, num_tokens, heads, device, rows_per_seq=_fp4_mla_packed_rows_per_seq(heads, gen_len)
        )
        q, compressed_kv, k_pe = self._project_qkv_a(hidden_states, prequantized_input).split(
            [self.q_lora_rank, lora, rope], -1
        )
        kv_stream = self.aux_stream if do_multi_stream() else None
        kv_fork, kv_ready = self.ln_events
        if kv_stream is not None:
            kv_fork.record()
            kv_stream.wait_event(kv_fork)
        with torch.cuda.stream(kv_stream) if kv_stream is not None else nullcontext():
            if self._fused_latent_row:
                latent_cache = k3_mla_q_fused.k3_mla_latent_row(
                    compressed_kv,
                    k_pe,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                    use_pdl=get_env_enable_pdl(),
                )
            else:
                latent_cache = torch.concat([self.kv_a_layernorm(compressed_kv), k_pe], dim=-1)
            self.mqa.fp4_mla_kv_append_generation(latent_cache, md, heads)
            if kv_stream is not None:
                kv_ready.record()
            if self._helix_tail_gate_behind_kv:
                # The fused Helix tail's gate, beside the fused query; the decode joins only the
                # KV half, the tail joins the gate.
                self._helix_tail_gate = self._gate(hidden_states)
                if kv_stream is not None:
                    self._gate_ready.record()
                    self._helix_tail_gate_pending = True
        norm = None
        if self._fused_decode_query_norm and k3_mla_q_fused.input_supported(q):
            norm = (self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon)
        else:
            q = self._q_a_layernorm_maybe_fused(q)
        state = md.fp4_mla_state
        k3_mla_q_fused.k3_mla_q_fused(
            q,
            self.q_b_proj.weight,
            self._q_b_exponents(),
            self._absorption_weight_bf16(),
            _get_fp4_mla_q_global_scale(md, device),
            self.mqa.rotary_cos_sin,
            md.helix_position_offsets,
            state.prequantized_q,
            state.prequantized_q_sf.view(torch.uint8),
            gen_len=gen_len,
            rows_per_seq=state.prequantized_q_rows_per_seq or gen_len * heads,
            use_pdl=get_env_enable_pdl(),
            norm=norm,
        )
        if kv_stream is not None:
            torch.cuda.current_stream().wait_event(kv_ready)
            if not torch.cuda.is_current_stream_capturing():
                latent_cache.record_stream(torch.cuda.current_stream())
        fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
        num_seqs = md.num_seqs
        attn_out = self._attn_forward_gen(
            self.mqa,
            fused_q.view([num_tokens, heads * (lora + rope)]),
            None,
            None,
            None,
            md,
            helix_v_up=True,
            helix_output=output[:num_tokens],
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,
            q_pe=fused_q[..., lora:],
            cu_q_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
            cu_kv_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
            fmha_scheduler_counter=torch.empty(1, dtype=torch.uint32, device=device),
            mla_bmm1_scale=torch.empty(2, dtype=torch.float32, device=device),
            mla_bmm2_scale=torch.empty(1, dtype=torch.float32, device=device),
            quant_q_buffer=torch.empty(
                num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
            ),
        )
        output[:num_tokens].view(attn_out.shape).copy_(attn_out)

    def takes_prequantized_input(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Generation-only batches whose fused q_a / kv_a projection takes the MXFP8 form of
        all ``num_rows`` rows (E4M3 and UE8M0 1x32 R128c4 scales, quantized with the low-M
        GEMM's arithmetic), which projects bitwise like the rows."""
        return (
            self._qkv_a_takes_prequantized
            and attn_metadata.num_contexts == 0
            and num_rows == attn_metadata.num_tokens
            and self.kv_a_proj_with_mqa.takes_prequantized_input(num_rows)
        )

    def _project_qkv_a(
        self,
        hidden_states: torch.Tensor,
        prequantized_input: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if prequantized_input is None:
            return super()._project_qkv_a(hidden_states, None)
        assert prequantized_input[0].shape[0] == hidden_states.shape[0]
        return self.kv_a_proj_with_mqa.forward_prequantized(*prequantized_input)

    def _gate_on_aux_stream(self) -> bool:
        return (
            self.helix_exchange_event is not None
            and self.aux_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )

    def _gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused_output_gate:
            return self.g_proj.forward_sigmoid(hidden_states)
        return self.g_proj(hidden_states).sigmoid()

    def _fork_gate(self, hidden_states: torch.Tensor, fork: torch.cuda.Event) -> torch.Tensor:
        """The gate on the aux stream from ``fork``; :meth:`_join_gate` joins it."""
        # Multi-stream follows maybe_execute_in_parallel: any later aux-stream
        # work first waits for the calling stream, which waits for the gate.
        self.aux_stream.wait_event(fork)
        with torch.cuda.stream(self.aux_stream):
            result = self._gate(hidden_states)
            self._gate_ready.record()
        return result

    def _join_gate(self, result: torch.Tensor) -> torch.Tensor:
        torch.cuda.current_stream().wait_event(self._gate_ready)
        if not torch.cuda.is_current_stream_capturing():
            result.record_stream(torch.cuda.current_stream())
        return result

    def _output_gate(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """``sigmoid(g_proj(hidden_states))``, the gate on o_proj's input.

        Under Helix CP with generation tokens it runs on the aux stream from
        the moment the all-to-all is enqueued, on the SMs the exchange leaves
        idle; it only depends on the module input.
        """
        if attn_metadata is None or not (
            attn_metadata.num_generations > 0 and self._gate_on_aux_stream()
        ):
            return self._gate(hidden_states)
        return self._join_gate(self._fork_gate(hidden_states, self.helix_exchange_event))

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
        prequantized_input: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if (
            self.use_output_gate
            and 1 <= attn_metadata.num_tokens <= self._helix_tail_max_tokens
            and attn_metadata.num_contexts == 0
            and not torch.compiler.is_compiling()
        ):
            # A fused-tail batch: the decode can hold every SM, so the gate GEMM runs beside the
            # q_a / q_b GEMMs rather than after the decode.
            if self._takes_fused_decode_query(hidden_states, attn_metadata, latent_cache_gen):
                self._helix_tail_gate_behind_kv = True
            elif self._gate_on_aux_stream():
                self._tail_gate_fork.record()
                self._helix_tail_gate = self._fork_gate(hidden_states, self._tail_gate_fork)
                self._helix_tail_gate_pending = True
            else:
                self._helix_tail_gate = self._gate(hidden_states)
        try:
            return super().forward(
                position_ids,
                hidden_states,
                attn_metadata,
                all_reduce_params=all_reduce_params,
                latent_cache_gen=latent_cache_gen,
                prequantized_input=prequantized_input,
            )
        finally:
            self._helix_tail_gate = None
            self._helix_tail_gate_pending = False
            self._helix_tail_gate_behind_kv = False
            self._helix_tail_operand = None

    def _join_tail_gate(self) -> None:
        if self._helix_tail_gate_pending:
            self._join_gate(self._helix_tail_gate)
            self._helix_tail_gate_pending = False

    def _attn_forward_gen(
        self,
        attn_backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        helix_v_up: bool = False,
        helix_output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """``MLA._attn_forward_gen``; a Helix batch of at most ``_helix_tail_max_tokens`` tokens
        with the V up-projection before the exchange runs the decode with its split reduction
        left to one tail kernel (fold, V up-projection, exchange, combine; K3HelixTail) instead
        of the reduction, the BMM, the all-to-all and the combine. The output gate of such a
        batch (forked in :meth:`forward`) is joined before the tail, so o_proj only depends on
        the tail; where o_proj takes a prequantized input, the tail also stores the MXFP8 form of
        the gated rows for it."""
        num_tokens = q.shape[0]
        if not (
            helix_v_up
            and 1 <= num_tokens <= self._helix_tail_max_tokens
            and not torch.compiler.is_compiling()
        ):
            return super()._attn_forward_gen(
                attn_backend,
                q,
                k,
                v,
                position_ids,
                attn_metadata,
                helix_v_up=helix_v_up,
                helix_output=helix_output,
                **kwargs,
            )
        softmax_stats = torch.empty(
            (num_tokens, self.num_heads_tp, 2), device=q.device, dtype=torch.float32
        )
        forward_args = AttentionForwardArgs(
            softmax_stats_tensor=softmax_stats, defer_split_reduce=True, **kwargs
        )
        partial_o = attn_backend.forward(q, k, v, attn_metadata, forward_args=forward_args)
        assert partial_o.shape[-1] == self.num_heads_tp * self.kv_lora_rank
        zero_kv_mask = _helix_gen_zero_kv_mask(attn_metadata, num_tokens)
        gate = self._helix_tail_gate
        self._join_tail_gate()
        if gate is None and self.helix_exchange_event is not None:
            # The gate of this forward is forked at the exchange (_output_gate).
            self.helix_exchange_event.record()
        width = self.num_heads_tp_cp * self.v_head_dim
        mx_out = None
        if (
            gate is not None
            and self._o_proj_takes_prequantized
            and self.o_proj.takes_prequantized_input(num_tokens)
        ):
            xq = torch.empty((num_tokens, width), dtype=torch.float8_e4m3fn, device=q.device)
            xs = torch.empty(
                ((num_tokens + 127) // 128 * 128 * width // 32,), dtype=torch.uint8, device=q.device
            )
            mx_out = (gate, xq, xs)
            self._helix_tail_operand = (xq, xs)
        out = helix_output
        if out is None:
            out = partial_o.new_empty((num_tokens, width))
        return K3HelixTail.get(self.mapping)(
            partial_o,
            softmax_stats,
            self.helix_v_b_proj_full,
            zero_kv_mask,
            fold=forward_args.split_partials,
            out=out.view(num_tokens, width),
            mx_out=mx_out,
        )

    def _gate_and_project_output(
        self,
        hidden_states: torch.Tensor,
        attn_output: list[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        # g_proj matches o_proj's input sharding, so the gate composes with the
        # helix-CP output projection. The fused projection runs only where the
        # output projection needs no reduction (cache_derived_state).
        if not self.use_output_gate:
            return super()._gate_and_project_output(
                hidden_states, attn_output, position_ids, attn_metadata, all_reduce_params
            )
        if self._helix_tail_operand is not None:
            return self.o_proj.forward_prequantized(*self._helix_tail_operand)
        gate = self._helix_tail_gate
        if gate is None:
            gate = self._output_gate(hidden_states, attn_metadata)
        self._join_tail_gate()
        if self._fused_output_gate:
            return self.o_proj.forward_gated(attn_output[0], gate)
        attn_output[0] = attn_output[0] * gate
        return super()._gate_and_project_output(
            hidden_states, attn_output, position_ids, attn_metadata, all_reduce_params
        )
