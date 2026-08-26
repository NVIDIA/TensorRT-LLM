# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorRT-LLM text implementation for Qwen3.8-Flash-Next checkpoints."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

if TYPE_CHECKING:
    from ..speculative.interface import SpecWorkerBase

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce, AllReduceParams, allgather
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..modules.mamba.mamba2_metadata import Mamba2Metadata
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.qwen4_exp_hyper_connection import Qwen4ExpHyperConnection
from ..modules.qwen4_exp_ple import PLEMetadata, Qwen4ExpPLE
from ..modules.rms_norm import RMSNorm
from ..pyexecutor.config_utils import get_qwen3_hybrid_layer_types, get_qwen4_exp_ple_layer_mask
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, create_lm_head_tp_mapping
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_qwen3_5 import _normalize_qwen35_exclude_modules
from .modeling_qwen3_next import Qwen3NextSparseMoeBlock
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
)
from .modeling_qwen4_exp_attention import Qwen4ExpAttention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (
    DecoderModel,
    register_auto_model,
    register_vision_encoder,
    remove_weights,
)


def _qwen4_exp_tp_output_reduction_enabled(mapping) -> bool:
    """Return whether decoder outputs require a tensor-parallel reduction."""
    return mapping.tp_size > 1 and not mapping.enable_attention_dp


class Qwen4ExpGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Gated DeltaNet mixer with the Qwen4-Exp **sigmoid** output gate.

    Reuses the production ``Qwen3NextGatedDeltaNet`` kernel path (causal conv1d,
    ``chunk_gated_delta_rule`` prefill, carried-state decode, in-proj / out-proj)
    unchanged, and only replaces the gated output RMSNorm: Qwen4-Exp sets
    ``output_gate_type="sigmoid"`` so the gate is ``sigmoid(z)`` rather than the
    ``silu(z)`` baked into ``rms_norm_gated_token_major``. It is implemented as
    native torch (``rmsnorm(x) * weight * sigmoid(z)``,
    ``norm_before_gate=True``, fp32 accumulation) — parity-first and CUDA-graph
    capturable; the fused kernel is a perf variant reserved for a later stage.
    """

    def _postprocess_gdn_output(
        self,
        attn_out: torch.Tensor,
        z: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        head_v = self.head_v_dim
        x = attn_out.reshape(-1, head_v).float()
        gate = z.reshape(-1, head_v).float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.norm.eps)
        out = x_norm * self.norm.weight.float() * torch.sigmoid(gate)
        out = out.to(attn_out.dtype).reshape(-1, self.value_dim_per_tp)
        return self.out_proj(out, all_reduce_params=all_reduce_params)


class Qwen4ExpDecoderLayer(DecoderLayer):
    """A single Qwen4-Exp decoder layer (linear GDN **or** full QSA attention).

    The mixer is selected by ``layer_type``; everything else — the two
    Hyper-Connection blocks, the routed+shared MoE, and (at the PLE layer) the
    n-gram short-conv side path — is shared. The residual bundle is threaded as
    ``[num_tokens, hc_count * hidden_size]`` and the block's tensor-parallel
    all-reduce is placed **between the mixer and ``combine``** (blueprint), so the
    replicated Hyper-Connection weights see the reduced block output.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
        layer_type: str,
        is_ple_layer: bool,
        ple_layer_index: int = 0,
    ):
        super().__init__()
        config = model_config.pretrained_config
        dtype = config.torch_dtype
        self.model_config = model_config
        self.mapping = model_config.mapping
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_dim = self.hc_count * self.hidden_size

        if layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(model_config, aux_stream, layer_idx)
        elif layer_type == "full_attention":
            # reduce_output=False: the layer's attn_tp_all_reduce (below) owns the
            # tensor-parallel reduction before the Hyper-Connection combine.
            self.self_attn = Qwen4ExpAttention(
                model_config, layer_idx=layer_idx, reduce_output=False
            )
        else:
            raise ValueError(f"Unsupported Qwen4-Exp layer type: {layer_type!r}")

        # 512 routed experts (top-10 renormalize, silu) + sigmoid-gated shared
        # expert, via create_moe (CUTLASS) — contract C5.
        self.mlp = Qwen3NextSparseMoeBlock(model_config, aux_stream, layer_idx=layer_idx)

        self.attn_hyper_connection = Qwen4ExpHyperConnection.from_config(
            config, use_mix=True, use_combine=True, dtype=dtype
        )
        self.mlp_hyper_connection = Qwen4ExpHyperConnection.from_config(
            config, use_mix=True, use_combine=True, dtype=dtype
        )

        self.ple: Optional[Qwen4ExpPLE] = None
        if is_ple_layer:
            self.ple = Qwen4ExpPLE(
                config, dtype=dtype, ple_layer_index=ple_layer_index, layer_id=layer_idx
            )

        self.enable_tp_output_reduction = _qwen4_exp_tp_output_reduction_enabled(self.mapping)
        self.attn_allreduce = (
            AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy, dtype=dtype
            )
            if self.enable_tp_output_reduction
            else None
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Optional[Mamba2Metadata] = None,
        ple_state: Optional[tuple] = None,
        spec_metadata=None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Expand the 2560-wide embedding into the 4-stream residual bundle at the
        # first layer; later layers already receive the 10240-wide bundle.
        if hidden_states.shape[-1] != self.hc_dim:
            assert hidden_states.shape[-1] == self.hidden_size, (
                f"Qwen4-Exp layer {self.layer_idx} expected a {self.hidden_size} "
                f"embedding or {self.hc_dim} bundle, got {hidden_states.shape[-1]}"
            )
            hidden_states = torch.cat([hidden_states] * self.hc_count, dim=-1)

        # PLE n-gram short-conv side path, injected into the bundle before mix.
        if self.ple is not None and ple_state is not None:
            ple_meta, conv_state, ngram_context = ple_state
            hidden_states = hidden_states + self.ple(
                hidden_states, ple_meta, conv_state, ngram_context
            )

        # Attention block: mix -> mixer -> attn_tp_all_reduce -> combine.
        mixed, residual = self.attn_hyper_connection.mix(hidden_states)
        no_reduce = AllReduceParams(enable_allreduce=False)
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(
                mixed,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
                all_reduce_params=no_reduce,
            )
        else:
            attn_out = self.self_attn(
                position_ids=position_ids,
                hidden_states=mixed,
                attn_metadata=attn_metadata,
                all_reduce_params=no_reduce,
                lora_params=lora_params,
            )
        if self.attn_allreduce is not None:
            attn_out = self.attn_allreduce(attn_out)
        hidden_states = self.attn_hyper_connection.combine(attn_out, residual)

        # MoE block: mix -> MoE(routed 512 + shared) -> combine. The MoE owns its
        # own tensor-parallel reduction of the combined routed+shared output.
        mixed, residual = self.mlp_hyper_connection.mix(hidden_states)
        moe_out = self.mlp(
            mixed,
            attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=self.enable_tp_output_reduction),
            lora_params=lora_params,
        )
        hidden_states = self.mlp_hyper_connection.combine(moe_out, residual)
        return hidden_states


class Qwen4ExpModel(DecoderModel):
    """Qwen4-Exp hybrid text decoder stack (contracts C1-C6).

    Carries the 4-stream Hyper-Connection residual bundle across a hybrid stack
    of Gated-DeltaNet linear layers and QSA full-attention layers and injects the
    PLE side path at its configured layer. The widened bundle is returned so a
    recurrent MTP layer can consume the target state; the logits processor owns
    the final ``hyper_connection_mixer.mix`` before the untied ``lm_head``.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        dtype = config.torch_dtype

        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }
        self.aux_stream = self.aux_stream_dict[AuxStreamType.Attention]
        self.preload_weight_modules = []
        if model_config.moe_backend == "TRTLLM":
            self.preload_weight_modules = ["experts", "routing_method", "all_reduce"]

        # VocabParallelEmbedding 248320x2560 (contract C6).
        if model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        else:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        layer_types = get_qwen3_hybrid_layer_types(config)
        ple_layer_mask = get_qwen4_exp_ple_layer_mask(config)
        # PLE embedding tables are indexed by a per-PLE-layer index (0-based over
        # the sorted ple_layer_ids), matching the checkpoint hashing seed.
        ple_layer_ids_sorted = sorted(set(getattr(config, "ple_layer_ids", None) or []))
        ple_layer_index_of = {(abs_id - 1): idx for idx, abs_id in enumerate(ple_layer_ids_sorted)}
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    model_config,
                    layer_idx,
                    self.aux_stream,
                    layer_types[layer_idx],
                    is_ple_layer=ple_layer_mask[layer_idx],
                    ple_layer_index=ple_layer_index_of.get(layer_idx, 0),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.num_hidden_layers = config.num_hidden_layers
        self.hc_dim = config.hc_count * config.hidden_size
        self.ple_layer_mask = ple_layer_mask
        self.has_ple = any(ple_layer_mask)
        self.eos_token_id = int(getattr(config, "eos_token_id", 0) or 0)

        # Fallback per-slot PLE recurrent state (short-conv state + n-gram
        # context) for harnesses WITHOUT a cache manager. At checkpoint scale the
        # authoritative pools are owned by **KVCacheManagerV2** and read via
        # ``kv_cache_manager.ple_layer_cache``; these model-owned
        # tensors are only the no-cache-manager fallback (unit tests). Lazily
        # allocated to the cache slot count and carried in place across the
        # prefill->decode boundary, mirroring the GDN conv/ssm per-slot pools and
        # indexed by the SAME mamba ``state_indices`` (see ``_prepare_ple_state``).
        # Shapes match ``config_utils.extract_qwen4_exp_ple_cache_params``
        # (conv_state_shape / ngram_context_len).
        self._ple_conv_state: Optional[torch.Tensor] = None
        self._ple_ngram_context: Optional[torch.Tensor] = None

        # The final Hyper-Connection mixer IS the last norm (use_combine=False);
        # there is no separate final RMSNorm (contracts C1/C6).
        self.hyper_connection_mixer = Qwen4ExpHyperConnection.from_config(
            config, use_mix=True, use_combine=False, dtype=dtype
        )

    def __pp_init__(self) -> None:
        """Assign the checkpoint-defined final mixer to the last PP stage."""
        super().__pp_init__()
        if self.model_config.mapping.has_pp():
            if not self.model_config.mapping.is_last_pp_rank():
                remove_weights(self.hyper_connection_mixer)
            self.has_ple = any(
                layer.ple is not None and not getattr(layer.ple, "_weights_removed", False)
                for layer in self.layers[: self.num_hidden_layers]
            )

    def _ensure_ple_pools(self, ple_module, num_slots: int, device: torch.device) -> tuple:
        """Lazily allocate / grow the persistent per-slot PLE recurrent-state
        pools so they cover ``num_slots`` cache slots, preserving existing state
        on growth. Returns ``(conv_state, ngram_context)``.

        conv_state: ``[num_slots, conv_channels, short_conv_state_len]`` (init 0);
        ngram_context: ``[num_slots, ngram_size - 1]`` int64 (init eos). Both are
        indexed in place by the mamba ``state_indices`` (mamba-style), so the PLE
        short conv + n-gram context carry across the prefill->decode boundary.
        """
        conv = self._ple_conv_state
        ctx = self._ple_ngram_context
        have = 0 if conv is None else conv.shape[0]
        if conv is not None and have >= num_slots and conv.device == device:
            return conv, ctx
        new_conv = torch.zeros(
            (num_slots, *ple_module.conv_state_shape), device=device, dtype=self.dtype
        )
        new_ctx = torch.full(
            (num_slots, ple_module.ngram_context_len),
            self.eos_token_id,
            device=device,
            dtype=torch.long,
        )
        # Preserve carried state for slots that already existed (growth path).
        if conv is not None and have:
            keep = min(have, num_slots)
            new_conv[:keep] = conv[:keep].to(device=device)
            new_ctx[:keep] = ctx[:keep].to(device=device)
        self._ple_conv_state = new_conv
        self._ple_ngram_context = new_ctx
        return new_conv, new_ctx

    def _prepare_ple_state(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor,
        mamba_metadata: Optional[Mamba2Metadata],
        spec_metadata: Optional[SpecMetadata],
    ) -> Optional[tuple]:
        """Build the PLE per-forward metadata + advance its recurrent-state pools.

        The n-gram context and short-conv state persist across the prefill->decode
        boundary, indexed by the SAME per-sequence mamba ``state_indices`` as the
        GDN conv/ssm state, and are advanced in place by :meth:`Qwen4ExpPLE.forward`.
        At checkpoint scale the pools are owned by **KVCacheManagerV2** and read via
        ``attn_metadata.kv_cache_manager.ple_layer_cache``; a
        harness without a cache manager falls back to persistent model-owned pools
        (``_ple_conv_state`` / ``_ple_ngram_context``). Both paths are **host-sync
        free** on the decode path (no ``int(state_indices.max().item())``) so the
        overlap scheduler and CUDA-graph capture see no host-device sync.

        A newly-starting sequence's slot is reset to the fresh state (conv 0,
        n-gram context eos) before the PLE module reads it — exactly as the mamba
        cache treats a reused conv/ssm slot as empty via ``has_initial_states`` —
        so a reused slot never leaks a prior sequence's state (risk R4).
        """
        if not self.has_ple:
            return None
        ple_layer_idx = next(i for i, m in enumerate(self.ple_layer_mask) if m)
        ple_module = self.layers[ple_layer_idx].ple

        input_ids = input_ids.reshape(-1)
        device = input_ids.device
        num_contexts = attn_metadata.num_contexts
        num_seq = attn_metadata.num_seqs
        use_spec_decoding = (
            spec_metadata is not None and getattr(spec_metadata, "runtime_draft_len", 0) > 0
        )
        is_decode = num_contexts == 0 and not use_spec_decoding

        # ``num_seq`` (one new token per sequence on decode) is derived from the
        # device-side, static-shape ``input_ids`` on the decode path rather than
        # copying ``attn_metadata.seq_lens`` host->device: that pageable H2D copy
        # runs *inside* the captured model forward and would both host-sync (breaking
        # the overlap-scheduler no-host-sync contract) and break CUDA-graph capture
        # path. ``PLEMetadata.build`` ignores ``seq_lens`` on decode (it
        # reconstructs the layout from ``input_ids``), so no decode information is
        # lost. The per-sequence chunk lengths are only needed for prefill, which is
        # never graph-captured, so the seq_lens H2D there is harmless.
        uniform_row_width = None
        if is_decode:
            input_ids = input_ids[:num_seq]
            seq_lens = None
        elif use_spec_decoding and num_contexts == 0:
            # A target verification row contains one golden token followed by
            # ``runtime_draft_len`` proposals. Use this static layout during
            # CUDA graph capture instead of reading lengths back to the host.
            uniform_row_width = int(spec_metadata.runtime_draft_len) + 1
            semantic_tokens = num_seq * uniform_row_width
            input_ids = input_ids[:semantic_tokens]
            seq_lens = torch.full((num_seq,), uniform_row_width, device=device, dtype=torch.long)
        else:
            sequence_lengths = [int(length) for length in attn_metadata.seq_lens[:num_seq]]
            semantic_tokens = sum(sequence_lengths)
            input_ids = input_ids[:semantic_tokens]
            seq_lens = torch.tensor(sequence_lengths, device=device, dtype=torch.long)

        state_indices = None
        if (
            mamba_metadata is not None
            and getattr(mamba_metadata, "state_indices", None) is not None
        ):
            state_indices = mamba_metadata.state_indices[:num_seq].to(
                device=device, dtype=torch.long
            )
        if state_indices is None:
            state_indices = torch.arange(num_seq, device=device, dtype=torch.long)

        if seq_lens is None:
            # Decode placeholder: ``PLEMetadata.build`` never reads ``seq_lens`` on
            # the decode path (each sequence contributes exactly one token). A
            # device-side ones tensor keeps the call host-sync free.
            seq_lens = state_indices.new_ones(num_seq)

        ple_meta = PLEMetadata.build(
            input_ids,
            seq_lens,
            state_indices,
            is_decode=is_decode,
            eos_token_id=self.eos_token_id,
            physical_tokens=attn_metadata.num_tokens,
            num_contexts=num_contexts,
            use_spec_decoding=use_spec_decoding,
            uniform_row_width=uniform_row_width,
        )

        conv_state, ngram_context = self._resolve_ple_pools(
            attn_metadata, ple_module, ple_layer_idx, num_seq, device
        )

        # Reset the slots of sequences STARTING this forward (fresh prefill, no
        # prior recurrent state) — device-indexed, no host sync; a no-op on the
        # decode path (num_contexts == 0). Context requests are the first
        # ``num_contexts`` sequences; honour ``has_initial_states`` when the mamba
        # metadata exposes it (chunked-prefill continuations keep their state),
        # else reset every context slot (no chunked prefill here).
        if num_contexts > 0:
            ctx_slots = state_indices[:num_contexts]
            has_init = getattr(mamba_metadata, "has_initial_states", None)
            if has_init is not None:
                has_init = has_init[:num_contexts].to(device=device).bool()
                fresh = ctx_slots[~has_init]
            else:
                fresh = ctx_slots
            if fresh.numel():
                conv_state[fresh] = 0
                ngram_context[fresh] = self.eos_token_id

        # Surface the resolved pools on metadata so graph capture and the
        # decoder layer observe the same tensors.
        # The cache-manager branch below is checked first each forward, so this
        # write never shadows the authoritative KVCacheManagerV2-owned pools.
        attn_metadata.qwen4_exp_ple_state = {ple_layer_idx: (conv_state, ngram_context)}
        return ple_meta, conv_state, ngram_context

    def _resolve_ple_pools(
        self,
        attn_metadata: AttentionMetadata,
        ple_module,
        ple_layer_idx: int,
        num_seq: int,
        device: torch.device,
    ) -> tuple:
        """Resolve the PLE (short-conv-state, n-gram-context) pools for a forward.

        Priority (all host-sync-free):
          1. **KVCacheManagerV2-owned pools** (checkpoint-scale runtime) —
             exposed via ``attn_metadata.kv_cache_manager.ple_layer_cache``.
          2. **Explicitly-provided pools** — ``attn_metadata.qwen4_exp_ple_state``
             set by a unit-test harness that owns its own state tensors.
          3. **Persistent model-owned pools** — no cache manager present; sized to
             the cache slot capacity exposed by ``attn_metadata`` (never
             ``int(state_indices.max().item())``, so no host-device sync).
        """
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is not None and hasattr(kv_cache_manager, "ple_layer_cache"):
            pools = kv_cache_manager.ple_layer_cache(ple_layer_idx)
            if pools is not None:
                conv_state, ngram_context = pools
                self._assert_ple_pool_shape(conv_state, ngram_context, ple_module)
                return conv_state, ngram_context

        provided = getattr(attn_metadata, "qwen4_exp_ple_state", None)
        if isinstance(provided, dict) and ple_layer_idx in provided:
            conv_state, ngram_context = provided[ple_layer_idx]
            self._assert_ple_pool_shape(conv_state, ngram_context, ple_module)
            return conv_state, ngram_context

        max_slots = int(getattr(attn_metadata, "max_num_requests", 0) or 0)
        needed = max(max_slots, num_seq, 1)
        return self._ensure_ple_pools(ple_module, needed, device)

    @staticmethod
    def _assert_ple_pool_shape(
        conv_state: torch.Tensor, ngram_context: torch.Tensor, ple_module
    ) -> None:
        """Static shape guard (no host sync) so a config/module drift between the
        cache-manager pool and the PLE module is a loud error, not silent-wrong."""
        expected_conv = tuple(ple_module.conv_state_shape)
        if tuple(conv_state.shape[1:]) != expected_conv:
            raise ValueError(
                f"PLE short-conv pool per-slot shape {tuple(conv_state.shape[1:])}"
                f" != module.conv_state_shape {expected_conv}"
            )
        if ngram_context.shape[1] != ple_module.ngram_context_len:
            raise ValueError(
                f"PLE n-gram-context width {ngram_context.shape[1]} != "
                f"module.ngram_context_len {ple_module.ngram_context_len}"
            )
        if conv_state.shape[0] != ngram_context.shape[0]:
            raise ValueError(
                f"PLE pool slot counts differ: conv {conv_state.shape[0]} vs "
                f"n-gram {ngram_context.shape[0]}"
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the "
                "same time, and must specify either one"
            )

        # Refresh mamba metadata for the batch size, mirroring Qwen3NextModel.
        mamba_metadata = getattr(attn_metadata, "mamba_metadata", None)
        if (
            mamba_metadata is not None
            and getattr(mamba_metadata, "max_batch_size", None) != attn_metadata.max_num_requests
        ):
            attn_metadata.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests, chunk_size=128
            )
            mamba_metadata = attn_metadata.mamba_metadata

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        mapping = self.model_config.mapping
        if mapping.has_pp() and not mapping.is_first_pp_rank():
            # The preceding stage sends the widened Hyper-Connection bundle.
            hidden_states = inputs_embeds.new_empty(inputs_embeds.shape[0], self.hc_dim)
        else:
            hidden_states = inputs_embeds

        ple_state = (
            self._prepare_ple_state(attn_metadata, input_ids, mamba_metadata, spec_metadata)
            if input_ids is not None
            else None
        )

        for layer_idx, decoder_layer in enumerate(self.layers[: self.num_hidden_layers]):
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                mamba_metadata=mamba_metadata,
                ple_state=ple_state if self.ple_layer_mask[layer_idx] else None,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        # Preserve the widened bundle for both the next PP stage and the
        # optional recurrent MTP layer. Qwen4ExpLogitsProcessor owns the final
        # Hyper-Connection collapse on the last PP stage.
        return hidden_states


class Qwen4ExpLogitsProcessor(nn.Module):
    """Collapse the target residual bundle before the language head."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        hyper_connection_mixer: Qwen4ExpHyperConnection,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.hc_count = model_config.pretrained_config.hc_count
        self.hidden_size = model_config.pretrained_config.hidden_size
        # The target model remains the sole owner of checkpoint parameters.
        object.__setattr__(self, "_hyper_connection_mixer", hyper_connection_mixer)

    def _collapse(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                "Qwen4-Exp logits processor received an invalid residual width: "
                f"expected {expected}, got {hidden_states.shape[-1]}"
            )
        return self._hyper_connection_mixer.mix(hidden_states)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not self.model_config.mapping.is_last_pp_rank():
            return lm_head(hidden_states).float()
        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]
        return lm_head(self._collapse(hidden_states)).float()


class Qwen4ExpMTPHead(Qwen4ExpLogitsProcessor):
    """Draft language head with its own checkpoint-defined final mixer."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        config = model_config.pretrained_config
        hyper_connection_mixer = Qwen4ExpHyperConnection.from_config(
            config, use_mix=True, use_combine=False, dtype=config.torch_dtype
        )
        super().__init__(model_config, hyper_connection_mixer)
        self.hyper_connection_mixer = hyper_connection_mixer
        object.__setattr__(self, "_hyper_connection_mixer", self.hyper_connection_mixer)
        self.mapping_lm_head_tp = None

    def _prepare_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool,
    ) -> torch.Tensor:
        """Select request rows and collapse the Hyper-Connection streams."""
        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)
        return self._collapse(hidden_states)

    def forward_local_full_vocab(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        """Produce local full-vocabulary logits for advanced MTP sampling.

        ADP rejection sampling cannot use the row-stacked, vocabulary-sharded
        LM-head-TP fast path.  The dense LM-head weight is replicated in ADP,
        but Qwen4-Exp must still apply its checkpoint-defined final
        Hyper-Connection mixer before that local projection.
        """
        hidden_states = self._prepare_hidden_states(
            hidden_states, attn_metadata, return_context_logits
        )
        return lm_head(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        hidden_states = self._prepare_hidden_states(
            hidden_states, attn_metadata, return_context_logits
        )
        mapping = self.model_config.mapping
        enable_lm_head_tp_in_adp = mapping.enable_attention_dp and mapping.enable_lm_head_tp_in_adp
        if enable_lm_head_tp_in_adp:
            self.mapping_lm_head_tp = create_lm_head_tp_mapping(mapping, hidden_states.shape[0])
            hidden_states = allgather(hidden_states, self.mapping_lm_head_tp, dim=0)

        if not mapping.enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = False
        logits = lm_head(
            hidden_states,
            mapping_lm_head_tp=self.mapping_lm_head_tp,
            is_spec_decoding_head=True,
        )
        if not mapping.enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = True
        return logits


class Qwen4ExpMTP(Qwen4ExpDecoderLayer):
    """One checkpoint MTP layer, replayed recurrently for each draft token."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict[AuxStreamType.Attention],
            "full_attention",
            False,
        )
        config = model_config.pretrained_config
        self.pre_fc_norm_embedding = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            hidden_size=config.hc_count * config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        linear_kwargs = {
            "bias": False,
            "dtype": config.torch_dtype,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
        }
        if model_config.mapping.enable_attention_dp:
            self.fc_embedding = Linear(config.hidden_size, config.hidden_size, **linear_kwargs)
            self.fc_hidden = Linear(config.hidden_size, config.hidden_size, **linear_kwargs)
        else:
            self.fc_embedding = Linear(
                config.hidden_size,
                config.hidden_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                **linear_kwargs,
            )
            self.fc_hidden = Linear(
                config.hidden_size,
                config.hidden_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                **linear_kwargs,
            )
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {
            event_type: torch.cuda.Event() for event_type in (EventType.Main, EventType.MoeShared)
        }
        self.shared_head = Qwen4ExpMTPHead(model_config)

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        config = self.model_config.pretrained_config

        def norm_embeds() -> torch.Tensor:
            return self.pre_fc_norm_embedding(embed_tokens(input_ids))

        def norm_hidden() -> torch.Tensor:
            return self.pre_fc_norm_hidden(hidden_states)

        inputs_embeds, hidden_states = maybe_execute_in_parallel(
            norm_embeds,
            norm_hidden,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
            disable_on_compile=True,
        )
        hidden_states = hidden_states.unflatten(-1, (config.hc_count, config.hidden_size))

        mapping = self.model_config.mapping
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            inputs_embeds = torch.chunk(inputs_embeds, mapping.tp_size, dim=-1)[
                mapping.tp_rank
            ].contiguous()
            hidden_states = torch.chunk(hidden_states, mapping.tp_size, dim=-1)[
                mapping.tp_rank
            ].contiguous()

        hidden_states = self.fc_hidden(hidden_states)
        hidden_states = hidden_states + self.fc_embedding(inputs_embeds).unsqueeze(-2)

        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens
        try:
            return super().forward(
                position_ids=position_ids,
                hidden_states=hidden_states.flatten(start_dim=-2),
                attn_metadata=attn_metadata,
                mamba_metadata=attn_metadata.mamba_metadata,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        finally:
            attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens


@register_auto_model("Qwen4ExpForCausalLM")
class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen4ExpModel, PretrainedConfig]):
    """Qwen4-Exp hybrid text core (arch ``Qwen4ExpForCausalLM``).

    Assembles the :class:`Qwen4ExpModel` decoder stack under an **untied**
    ``lm_head`` (``tie_word_embeddings=False``). Selects **KVCacheManagerV2** for
    the hybrid recurrent-state layout and supports the checkpoint's single
    recurrent one-model MTP layer.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        _normalize_qwen35_exclude_modules(model_config)
        spec_config = getattr(model_config, "spec_config", None)
        if spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model():
            # The checkpoint contains one trained layer. MTP3/5/7 replay the
            # same layer recurrently rather than expecting distinct weights.
            model_config.pretrained_config.num_nextn_predict_layers = 1
        SpecDecOneEngineForCausalLM.__init__(
            self,
            Qwen4ExpModel(model_config),
            model_config,
        )
        self.logits_processor = Qwen4ExpLogitsProcessor(
            model_config, self.model.hyper_connection_mixer
        )
        self.preload_weight_modules = self.model.preload_weight_modules
        if spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model():
            self.model.layers.extend(self.draft_model.mtp_layers)
            self._register_auxiliary_speculative_state_handlers(
                self.spec_worker,
                self.model.layers[: self.model.num_hidden_layers],
            )

    @staticmethod
    def _register_auxiliary_speculative_state_handlers(
        spec_worker: "SpecWorkerBase", layers: Sequence[DecoderLayer]
    ) -> None:
        """Register target recurrent side state that follows MTP acceptance."""
        for layer in layers:
            if layer.ple is not None:
                spec_worker.register_auxiliary_state_handler(layer.ple)
            attention = getattr(layer, "self_attn", None)
            indexer = getattr(attention, "indexer", None)
            if indexer is not None and hasattr(indexer, "commit_speculative_states"):
                spec_worker.register_auxiliary_state_handler(indexer)

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Return conservative defaults for hybrid state and QSA."""
        return {
            "kv_cache_config": {"enable_block_reuse": False},
            "moe_config": {"disable_finalize_fusion": True},
            "sparse_attention_config": {"algorithm": "qsa"},
            "cuda_graph_config": None,
        }

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: object | None = None
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 for the hybrid recurrent-state layout."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: object | None = None
    ) -> Literal["PYTHON"]:
        """Use the Python transceiver for hybrid-state transfers."""
        return "PYTHON"

    def load_weights(
        self,
        weights: dict,
        weight_mapper: BaseWeightMapper,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ):
        """Load a Qwen3.8-Flash-Next checkpoint through its text mapper.

        ``Qwen4ExpHfWeightMapper.preprocess_weights`` rewrites the composite
        checkpoint's ``model.language_model.*`` text tensors into this model's
        ``model.*`` tree, including the Gated-DeltaNet, MoE, PLE, and
        Hyper-Connection transformations.
        """
        new_weights = weight_mapper.preprocess_weights(
            weights, allow_partial_loading=allow_partial_loading
        )
        # ``new_weights`` aliases every source tensor ``preprocess_weights`` did
        # not rewrite; releasing the source dict makes those aliases the last
        # reference so the loader can free weights module by module (mirrors
        # Qwen3Next). The PLE n-gram shards were already streamed into the model
        # during preprocessing, so dropping them here frees ~100 GB promptly.
        if hasattr(weights, "clear"):
            weights.clear()
        super().load_weights(
            new_weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )


_QWEN4_EXP_VL_PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)


@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen4ExpForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen4_exp",
    placeholder_metadata=_QWEN4_EXP_VL_PLACEHOLDER_METADATA,
)
class Qwen4ExpForConditionalGeneration(Qwen3VLModelBase):
    """Composite model using the production Qwen3-VL vision stack."""

    supports_encoder_cache = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return Qwen4ExpForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: object | None = None
    ) -> Literal["V2"]:
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: object | None = None
    ) -> Literal["PYTHON"]:
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs) -> None:
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get("disable_fuse_rope", False)
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    @property
    def embedding_dim(self) -> int:
        # PP ranks without the token embedding still need the encoder-output
        # capacity calculation during engine initialization.
        text_config = self.llm.model_config.pretrained_config
        return text_config.hidden_size * (self.deepstack_num_level + 1)

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.llm.model_config.pretrained_config.torch_dtype

    def load_weights(
        self,
        weights: Dict[str, torch.Tensor],
        weight_mapper: BaseWeightMapper,
        allow_partial_loading: bool = False,
    ) -> None:
        from .checkpoints.hf.qwen4_exp_weight_mapper import Qwen4ExpHfWeightMapper

        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights, allow_partial_loading=allow_partial_loading)
        if not isinstance(weight_mapper, Qwen4ExpHfWeightMapper):
            raise TypeError(
                "Qwen4ExpForConditionalGeneration requires "
                f"Qwen4ExpHfWeightMapper, got {type(weight_mapper).__name__}"
            )
        if weight_mapper.model is not self.llm:
            weight_mapper.init_model_and_config(self.llm, self.llm.model_config)
        self.llm.load_weights(
            weights,
            weight_mapper,
            allow_partial_loading=allow_partial_loading,
        )
