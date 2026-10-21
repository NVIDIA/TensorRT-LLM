# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, and
gated output projection.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from ....functional import PositionEmbeddingType
from ....mapping import Mapping
from ....quantization import QuantAlgo
from ...attention.backends import AttentionMetadata, TrtllmAttention
from ...attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ...attention.mla import MLA
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...utils import AuxStreamType
from ..linear import Linear, TensorParallelMode
from ..multi_stream_utils import do_multi_stream
from ..projection_capabilities import GatedEpilogueProjection, PrequantizedInputProjection


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
                # The gate GEMM runs beside the Helix all-to-all (_output_gate).
                self.helix_exchange_event = torch.cuda.Event()
                self._gate_ready = torch.cuda.Event()

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

    def takes_prequantized_input(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Generation-only batches whose fused q_a / kv_a projection runs the low-M kernel on
        all ``num_rows`` rows: there the MXFP8 form of the rows (E4M3 and UE8M0 1x32 R128c4
        scales, quantized with the low-M kernel's arithmetic) projects bitwise like the rows."""
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

    def _output_gate(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """``sigmoid(g_proj(hidden_states))``, the gate on o_proj's input.

        Under Helix CP with generation tokens it runs on the aux stream from
        the moment the all-to-all is enqueued, on the SMs the exchange leaves
        idle; it only depends on the module input.
        """

        def gate() -> torch.Tensor:
            if self._fused_output_gate:
                return self.g_proj.forward_sigmoid(hidden_states)
            return self.g_proj(hidden_states).sigmoid()

        if not (
            self.helix_exchange_event is not None
            and attn_metadata.num_generations > 0
            and self.aux_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        ):
            return gate()
        # Multi-stream follows maybe_execute_in_parallel: any later aux-stream
        # work first waits for the calling stream, which waits for the gate.
        self.aux_stream.wait_event(self.helix_exchange_event)
        with torch.cuda.stream(self.aux_stream):
            result = gate()
            self._gate_ready.record()
        torch.cuda.current_stream().wait_event(self._gate_ready)
        if not torch.cuda.is_current_stream_capturing():
            result.record_stream(torch.cuda.current_stream())
        return result

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
        gate = self._output_gate(hidden_states, attn_metadata)
        if self._fused_output_gate:
            return self.o_proj.forward_gated(attn_output[0], gate)
        attn_output[0] = attn_output[0] * gate
        return super()._gate_and_project_output(
            hidden_states, attn_output, position_ids, attn_metadata, all_reduce_params
        )
