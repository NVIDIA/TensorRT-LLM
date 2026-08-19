# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import inspect
from dataclasses import replace
from typing import Dict, Generic, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from tensorrt_llm.logger import logger

from ...functional import PositionEmbeddingType
from ..attention.attention import Attention
from ..attention.mla import MLA
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig, TConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import moe_load_balancer_set_repeated_for_next_layer
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from ..speculative import (SpecMetadata, get_spec_worker,
                           should_use_separate_draft_kv_cache)
from ..speculative.interface import SpeculativeDecodingMode
from ..utils import AuxStreamType
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM, TModel,
                             get_model_architecture,
                             get_registered_draft_model_builder,
                             register_auto_model, register_draft_model)

_SPECULATIVE_POSITION_HEADROOM = "_speculative_position_headroom"


def _ensure_draft_vocab_size(config: PretrainedConfig) -> None:
    if hasattr(config,
               "draft_vocab_size") and config.draft_vocab_size is not None:
        return

    logger.warning(
        "Missing 'draft_vocab_size' in pretrained config; defaulting to 'vocab_size'. "
        "Set 'draft_vocab_size' explicitly if the draft head uses a different vocabulary."
    )
    config.draft_vocab_size = config.vocab_size


def _slice_spec_position_ids(position_ids: Optional[torch.Tensor],
                             num_tokens: int) -> Optional[torch.Tensor]:
    """Slice speculative position IDs along the token dimension."""
    if position_ids is None:
        return None
    return position_ids[..., :num_tokens]


# ---------------------------------------------------------------------------
# DSpark draft-network heads, shared by both drafters that implement DSpark:
# the DFlash path (modeling_dflash.py, standalone Kimi K3 style drafters) and
# the DeepSeek-V4-Pro path (modeling_dspark.py, mtp.* stages inside the target
# checkpoint). DFlash is the degenerate case -- DSpark with the Markov and
# confidence heads switched off -- so the math below is the *only* copy; the
# two drafters differ solely in how they store the weights and whether their
# draft lm_head is TP vocab-sharded.
#
# Ported from DeepSeek's DeepSpec reference implementation
# (https://github.com/deepseek-ai/DeepSpec, MIT License).
# ---------------------------------------------------------------------------


def greedy_or_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Argmax for temperature<=0, else temperature-scaled multinomial.

    Args:
        logits: ``[..., vocab]``.
    Returns:
        token ids with the trailing vocab dim reduced.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(probs.shape[:-1])


def dspark_markov_step_bias(prev_tokens: torch.Tensor, markov_w1: torch.Tensor,
                            markov_w2: torch.Tensor) -> torch.Tensor:
    """Vanilla Markov head logit bias for one intra-block draft step.

    Reference: DeepSpec ``VanillaMarkov`` (deepspec/modeling/dspark/
    markov_head.py): ``bias = markov_w2(markov_w1(prev_token))`` where
    markov_w1 is nn.Embedding(vocab, rank) and markov_w2 is
    nn.Linear(rank, vocab, bias=False). With both weights stored
    [vocab, rank] this is ``markov_w1[prev] @ markov_w2.T``.

    Args:
        prev_tokens: [B] long, previous token per request (draft vocab).
        markov_w1: [vocab, rank].
        markov_w2: [vocab_or_shard, rank] (rows may be a TP vocab shard).
    Returns:
        [B, vocab_or_shard] bias in the markov weights' dtype.
    """
    return F.linear(F.embedding(prev_tokens, markov_w1), markov_w2)


def dspark_markov_chain(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    step_bias_fn,
    *,
    hidden_states: Optional[torch.Tensor] = None,
    next_token_fn=None,
    cast_bias_to_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The intra-block Markov refinement loop, shared by every DSpark head.

    Reference: DeepSpec ``VanillaMarkov.sample_block_tokens``. For step i,
    ``logits_i += bias(prev_i)`` with ``prev_0`` = the anchor token (the last
    accepted token, block slot 0) and ``prev_{i>0}`` = the token drawn from
    step i-1's *biased* logits. llama.cpp PR #25173 implements the same chain.

    The chain being sequential is load-bearing and is NOT derivable from a
    checkpoint: published DSpark drafters ship only the training-time
    ``apply_block_logits``, which biases the whole block in one teacher-forced
    pass. SGLang ``srt/models/dspark.py:34-64 run_markov_block`` settles it --
    it steps the same way, feeding back the token drawn from the biased logits.

    Two callers drive this, and a change here has to satisfy both: the embedded
    flavour chains it inside the model, which owns its sampler, while the
    standalone flavour is driven from the worker through ``next_token_fn`` so
    the chain can advance on a TP-gathered global argmax the model cannot
    compute on its own.

    Args:
        base_logits: [B, K, vocab_or_shard] shared-lm_head logits.
        first_prev_tokens: [B] long, anchor token ids (draft vocab).
        step_bias_fn: ``(prev_tokens [B], step_hidden or None) -> bias``. A
            closure, so a stateful head (RNN) can carry its recurrent state
            across positions without a second loop.
        hidden_states: [B, K, d] fed to ``step_bias_fn`` one position at a
            time; None for the memoryless heads.
        next_token_fn: ``([B, vocab_or_shard]) -> [B]`` token ids in the FULL
            draft vocab; defaults to a plain argmax. Drafters whose draft
            logits are TP vocab-sharded pass a shard-aware argmax here.
        cast_bias_to_logits: cast the bias down to ``base_logits.dtype``
            before adding. The DFlash drafter does; the V4-Pro drafter does
            not (its Markov weights and its logits already agree), and adding
            the cast there would silently narrow its accumulation dtype.
    Returns:
        sampled_tokens [B, K], corrected_logits [B, K, vocab_or_shard].
        Greedy per-position argmax of the corrected logits reproduces the
        reference sampled chain exactly.
    """
    batch_size, block_size = base_logits.shape[:2]
    if block_size == 0:
        empty = torch.empty(batch_size,
                            0,
                            dtype=torch.long,
                            device=base_logits.device)
        return empty, base_logits
    sampled, corrected = [], []
    prev = first_prev_tokens.long()
    for k in range(block_size):
        step_hidden = None if hidden_states is None else hidden_states[:, k]
        bias = step_bias_fn(prev, step_hidden)
        if cast_bias_to_logits:
            bias = bias.to(base_logits.dtype)
        step_logits = base_logits[:, k] + bias
        corrected.append(step_logits.unsqueeze(1))
        if next_token_fn is None:
            prev = torch.argmax(step_logits, dim=-1)
        else:
            prev = next_token_fn(step_logits).long()
        sampled.append(prev)
    return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


def dspark_markov_chain_logits(base_logits: torch.Tensor,
                               first_prev_tokens: torch.Tensor,
                               markov_w1: torch.Tensor,
                               markov_w2: torch.Tensor,
                               argmax_fn=None) -> torch.Tensor:
    """Raw-tensor entry to :func:`dspark_markov_chain`, corrected logits only.

    For drafters that keep the Markov head as plain checkpoint tensors rather
    than a :class:`VanillaMarkov` module (the DFlash path). ``markov_w2`` may
    already be sliced down to this rank's TP vocab shard, in which case
    ``argmax_fn`` must map a shard-local row back to a full-vocab token id.
    """

    def _step_bias(prev_tokens, step_hidden):
        del step_hidden
        return dspark_markov_step_bias(prev_tokens, markov_w1, markov_w2)

    _, corrected = dspark_markov_chain(base_logits,
                                       first_prev_tokens,
                                       _step_bias,
                                       next_token_fn=argmax_fn,
                                       cast_bias_to_logits=True)
    return corrected


class VanillaMarkov(nn.Module):
    """Low-rank token-bigram logit bias: ``bias = W2(W1[token])``."""

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}.")
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank,
                                   self.vocab_size,
                                   bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.long(), self.markov_w1.weight)

    def project_bias(self,
                     latent_states: torch.Tensor,
                     *,
                     vocab_slice: Optional[slice] = None) -> torch.Tensor:
        w2 = self.markov_w2.weight
        if vocab_slice is not None:
            w2 = w2[vocab_slice]
        return F.linear(latent_states, w2)

    def compute_step_bias(self,
                          token_ids: torch.Tensor,
                          hidden_states: Optional[torch.Tensor],
                          *,
                          vocab_slice: Optional[slice] = None) -> torch.Tensor:
        del hidden_states
        w2 = self.markov_w2.weight
        if vocab_slice is not None:
            w2 = w2[vocab_slice]
        return dspark_markov_step_bias(token_ids.long(), self.markov_w1.weight,
                                       w2)

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
        vocab_slice: Optional[slice] = None,
        next_token_fn=None,
        cast_bias_to_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive block sampling with the (memoryless) Markov bias.

        Args:
            base_logits: ``[batch, block_size, vocab]`` from backbone+lm_head.
            first_prev_token_ids: ``[batch]`` token preceding the 1st position.
            hidden_states: ``[batch, block_size, d]`` (unused by vanilla).
            vocab_slice / next_token_fn / cast_bias_to_logits: see
                :func:`dspark_markov_chain`; only the TP vocab-sharded DFlash
                drafter sets them.
        Returns:
            sampled_tokens ``[batch, block_size]``,
            corrected_logits ``[batch, block_size, vocab]``.
        """

        def _step_bias(prev_tokens, step_hidden):
            return self.compute_step_bias(prev_tokens,
                                          step_hidden,
                                          vocab_slice=vocab_slice)

        def _sample_step(step_logits):
            return greedy_or_sample(step_logits, temperature)

        return dspark_markov_chain(
            base_logits,
            first_prev_token_ids,
            _step_bias,
            hidden_states=hidden_states,
            next_token_fn=_sample_step
            if next_token_fn is None else next_token_fn,
            cast_bias_to_logits=cast_bias_to_logits,
        )


class GatedMarkovHead(VanillaMarkov):
    """Markov bias gated by a sigmoid of [hidden, prev_embedding]."""

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(self,
                          token_ids: torch.Tensor,
                          hidden_states: Optional[torch.Tensor],
                          *,
                          vocab_slice: Optional[slice] = None) -> torch.Tensor:
        assert hidden_states is not None
        prev_emb = self.get_prev_embeddings(token_ids)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([hidden_states, prev_emb],
                                     dim=-1))).to(dtype=prev_emb.dtype)
        return self.project_bias(gate * prev_emb, vocab_slice=vocab_slice)


class RNNHead(VanillaMarkov):
    """GRU-style head carrying recurrent state across block positions."""

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        # [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size,
                                    3 * markov_rank)

    def _rnn_step(self,
                  state,
                  prev_embeddings,
                  hidden_states,
                  *,
                  vocab_slice: Optional[slice] = None):
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(cand_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(out_raw), vocab_slice=vocab_slice)
        return new_state, bias

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
        vocab_slice: Optional[slice] = None,
        next_token_fn=None,
        cast_bias_to_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None
        state = torch.zeros(base_logits.shape[0],
                            self.markov_rank,
                            device=base_logits.device,
                            dtype=hidden_states.dtype)

        def _step_bias(prev_tokens, step_hidden):
            nonlocal state
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state,
                                         prev_emb,
                                         step_hidden,
                                         vocab_slice=vocab_slice)
            return bias

        def _sample_step(step_logits):
            return greedy_or_sample(step_logits, temperature)

        return dspark_markov_chain(
            base_logits,
            first_prev_token_ids,
            _step_bias,
            hidden_states=hidden_states,
            next_token_fn=_sample_step
            if next_token_fn is None else next_token_fn,
            cast_bias_to_logits=cast_bias_to_logits,
        )


def build_markov_head(*, markov_head_type: str, vocab_size: int,
                      markov_rank: int,
                      hidden_size: int) -> Optional[nn.Module]:
    """Factory mirroring DeepSpec ``build_markov_head``; None if rank==0."""
    if int(markov_rank) <= 0:
        return None
    kind = str(markov_head_type).lower()
    if kind == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if kind == "gated":
        return GatedMarkovHead(vocab_size=vocab_size,
                               markov_rank=markov_rank,
                               hidden_size=hidden_size)
    if kind == "rnn":
        return RNNHead(vocab_size=vocab_size,
                       markov_rank=markov_rank,
                       hidden_size=hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance-confidence predictor (DeepSpec
    AcceptRatePredictor).

    Input features are the backbone hidden state, optionally concatenated with
    the Markov head's previous-token embedding. Output is a single logit per
    position.
    """

    def __init__(self,
                 *,
                 hidden_size: int,
                 markov_rank: int = 0,
                 with_markov: bool = False):
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if with_markov else 0)
        # The checkpoint stores ``proj`` as a bias-free bf16 weight, but the
        # confidence score is computed in fp32 (mirrors the DeepSpec reference
        # ``Linear(input_dim, 1, dtype=torch.float32)`` with the fp32 matmul).
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self,
                hidden_states: torch.Tensor,
                prev_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.with_markov:
            assert prev_embeddings is not None
            features = torch.cat(
                [hidden_states,
                 prev_embeddings.to(hidden_states.dtype)],
                dim=-1)
        else:
            features = hidden_states
        # fp32 matmul for a stable confidence score (mirrors the reference).
        return self.proj(features.float()).squeeze(-1)


def confident_prefix_length(confidence_logits: torch.Tensor, *, block_size: int,
                            threshold: float) -> int:
    """First position k where ``sigmoid(confidence_k) < threshold``.

    Returns ``block_size`` when threshold<=0 (no truncation) or all positions
    are confident. Assumes batch size 1 (functional-first scope).
    """
    if threshold <= 0.0:
        return int(block_size)
    below = confidence_logits.sigmoid() < threshold
    if not bool(below[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below[0], as_tuple=False)[0].item())


class Eagle3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        next_layer_regular: bool = False,
    ):
        config = model_config.pretrained_config
        self._next_layer_regular = next_layer_regular
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        tp_size = model_config.mapping.tp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1
        # Override the QKV projection. The number of input features
        # is twice as big for EAGLE3 draft models.
        if not self._next_layer_regular:
            qkv_shard_indices_mapping = {
                "q": (0, self.q_size),
                "k": (self.q_size, self.kv_size),
                "v": (self.q_size + self.kv_size, self.kv_size),
            }
            self.qkv_proj = Linear(
                2 * self.hidden_size,
                tp_size * self.q_size + 2 * tp_size * self.kv_size,
                bias=config.attention_bias,
                dtype=config.torch_dtype,
                mapping=self.qkv_proj.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR),
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
            )


class Eagle3MLAttention(MLA):
    """
    MLA (Multi-head Latent Attention) for Eagle3 draft model (e.g., DeepSeekV3).
    The first layer takes concatenated [embeds, hidden_states] as input (2x hidden_size),
    while subsequent layers take regular hidden_states (1x hidden_size).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        next_layer_regular: bool = False,
    ):
        config = model_config.pretrained_config
        self._next_layer_regular = next_layer_regular

        predicted_tokens_per_seq = (model_config.spec_config.tokens_per_gen_step
                                    if model_config.spec_config is not None else
                                    1)

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            v_head_dim=config.v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn,
                rope=RopeParams.from_config(config),
                is_neox=False,
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            aux_stream_dict=aux_stream_dict,
        )

        # Override the kv_a_proj_with_mqa projection for first layer.
        # The number of input features is twice as big for EAGLE3 draft models.
        if not self._next_layer_regular:
            quant_config = model_config.get_quant_config()
            # For Eagle3, first layer takes [embeds, hidden_states] concatenated
            self.kv_a_proj_with_mqa = Linear(
                2 * config.hidden_size,  # Double input size
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )


class Eagle3DecoderLayer(DecoderLayer):
    """
    Unified decoder layer for Eagle3 speculative decoding.
    Supports both standard attention (Llama-style) and MLA (DeepSeekV3-style).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int = 0,
        is_first_layer: bool = True,
        use_mla: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config
        eagle_config = config.eagle_config if hasattr(config,
                                                      "eagle_config") else {}
        self.layer_idx = layer_idx
        self._next_layer_regular = (eagle_config.get("next_layer_regular", True)
                                    and not is_first_layer) or eagle_config.get(
                                        "eh_proj_before_attn", False)

        # Select attention type based on config
        if use_mla:
            self.self_attn = Eagle3MLAttention(
                model_config,
                layer_idx,
                aux_stream_dict={AuxStreamType.Attention: aux_stream},
                next_layer_regular=self._next_layer_regular,
            )
        else:
            self.self_attn = Eagle3Attention(model_config, layer_idx,
                                             self._next_layer_regular)

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=inter_size,
            bias=getattr(config, "mlp_bias", False),
            dtype=config.torch_dtype,
            config=model_config,
            overridden_tp_size=1
            if model_config.mapping.enable_attention_dp else None,
        )

        if not self._next_layer_regular:
            self.input_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

        self.hidden_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        if not self._next_layer_regular:
            embeds = self.input_layernorm(embeds)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
        # PyExecutor will extract these from the draft model engine's spec metadata.
        # They will be passed to the draft model engine on the next iteration.
        # TODO: can we support multiple model outputs instead?
        spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states,
                                                  residual)
        return hidden_states, residual


class Eagle3DraftModel(DecoderModel):
    """
    Unified Eagle3 draft model supporting both standard attention (Llama-style)
    and MLA attention (DeepSeekV3-style).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
        use_mla: bool = False,
    ) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        eagle_config = config.eagle_config if hasattr(config,
                                                      "eagle_config") else {}
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping
        self.num_layers = model_config.pretrained_config.num_hidden_layers
        self._eh_proj_before_attn = eagle_config.get("eh_proj_before_attn",
                                                     False)
        self._norm_before_fc = eagle_config.get("norm_before_fc", False)
        self._use_mla = use_mla

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self._return_hidden_post_norm = eagle_config.get(
            "return_hidden_post_norm", False) or getattr(
                config, "norm_output", False)

        # Create auxiliary CUDA stream for MLA operations (only needed for MLA)
        self.aux_stream = torch.cuda.Stream() if use_mla else None

        if self.spec_config.num_capture_layers > 1:
            self.fc = Linear(
                self.hidden_size_in * self.spec_config.num_capture_layers,
                config.hidden_size,
                bias=getattr(config, "bias", False),
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
            )
        if self._norm_before_fc:
            self.input_norm = RMSNorm(
                hidden_size=self.hidden_size_in *
                self.spec_config.num_capture_layers,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
        else:
            self.input_norm = None

        self._use_fc_norm = getattr(config, "fc_norm", False)
        if self._use_fc_norm:
            self.fc_norm = nn.ModuleList([
                RMSNorm(
                    hidden_size=self.hidden_size_in,
                    eps=config.rms_norm_eps,
                    dtype=config.torch_dtype,
                ) for _ in range(self.spec_config.num_capture_layers)
            ])
        else:
            self.fc_norm = None

        if self.num_layers > 1:
            self.midlayer = nn.ModuleList([
                Eagle3DecoderLayer(
                    model_config,
                    start_layer_idx + i,
                    is_first_layer=(i == 0),
                    use_mla=use_mla,
                    aux_stream=self.aux_stream,
                ) for i in range(self.num_layers)
            ])
        else:
            self.midlayer = Eagle3DecoderLayer(
                model_config,
                start_layer_idx,
                use_mla=use_mla,
                aux_stream=self.aux_stream,
            )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if (config.draft_vocab_size is not None
                and config.vocab_size != config.draft_vocab_size):
            self.d2t = nn.Parameter(
                torch.empty((config.draft_vocab_size, ), dtype=torch.int32),
                requires_grad=False,
            )

        if self._eh_proj_before_attn:
            self.enorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.eh_proj = nn.Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=eagle_config.get("eh_proj_bias", False),
                dtype=config.torch_dtype,
            )

        if self.hidden_size_in != config.hidden_size:
            if model_config.mapping.enable_attention_dp:
                self.embed_tokens = Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                )
            else:
                self.embed_tokens = Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                    mapping=model_config.mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                )
        else:
            # Shared with target model.
            self.embed_tokens = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # When ``all_rank_num_tokens`` is supplied the caller wants this draft
        # forward to run with a different attention-DP token distribution
        # (e.g. the worker's per-step value); restore the original on exit so
        # the next call sees the same attn_metadata it had on entry.
        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                assert self.embed_tokens is not None
                inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)

            assert hidden_states is not None
            # NOTE: If hidden states from the target model have to be concatenated,
            # ideally, we expect that to happen outside the model definition. This
            # helps us avoid data-dependent control flow and gives us better CUDA
            # graph coverage.
            if self._eh_proj_before_attn:
                input_embeds = self.enorm(inputs_embeds)
                hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
                hidden_states = self.eh_proj(hidden_states)

            residual = None
            if self.num_layers > 1:
                for layer in self.midlayer:
                    if residual is not None:
                        hidden_states = hidden_states + residual
                    hidden_states, residual = layer(
                        position_ids=position_ids,
                        embeds=inputs_embeds,
                        hidden_states=hidden_states,
                        attn_metadata=attn_metadata,
                        spec_metadata=spec_metadata,
                    )
            else:
                hidden_states, residual = self.midlayer(
                    position_ids=position_ids,
                    embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                )

            hidden_states, hidden_states_to_save = self.norm(
                hidden_states, residual)
            if self._return_hidden_post_norm:
                return hidden_states, hidden_states
            return hidden_states, hidden_states_to_save
        finally:
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens


# We use Llama3 as the base architecture for EAGLE3 draft layers
@register_auto_model("EAGLE3LlamaForCausalLM")
@register_auto_model("Eagle3DeepSeekV3ForCausalLM")
class Eagle3ForCausalLM(DecoderModelForCausalLM[Eagle3DraftModel,
                                                PretrainedConfig]):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
    ):
        config = model_config.pretrained_config
        _ensure_draft_vocab_size(config)

        # Determine if we should use MLA attention based on config
        # MLA is used for DeepSeekV3-style models that have kv_lora_rank
        config = model_config.pretrained_config
        self._use_mla = hasattr(config, 'kv_lora_rank') and config.kv_lora_rank

        draft_model = Eagle3DraftModel(
            model_config,
            start_layer_idx,
            use_mla=self._use_mla,
        )

        super().__init__(
            draft_model,
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.draft_vocab_size,
        )
        self.load_lm_head_from_target = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.apply_eagle3_fc(spec_metadata.get_hidden_states())
        output, _ = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            hidden_states=hidden_states,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        # Remap weight names: some Eagle3 checkpoints use "layers.X.*" naming convention
        # while the model expects "midlayer.*" naming. Handle both formats.
        import re
        remapped_weights = {}
        # Access num_layers from the inner draft model (self.model is Eagle3DraftModel)
        num_layers = self.model.num_layers
        for k, v in weights.items():
            new_k = k
            # For single-layer models: "layers.0.*" -> "midlayer.*"
            # For multi-layer models: "layers.X.*" -> "midlayer.X.*"
            if num_layers == 1:
                # Single layer: layers.0.foo -> midlayer.foo
                new_k = re.sub(r'^layers\.0\.', 'midlayer.', new_k)
            else:
                # Multi-layer: layers.X.foo -> midlayer.X.foo
                new_k = re.sub(r'^layers\.(\d+)\.', r'midlayer.\1.', new_k)
            remapped_weights[new_k] = v

        new_weights = {}
        for k, v in remapped_weights.items():
            if 'lm_head' not in k:
                new_k = "model." + k
            else:
                self.load_lm_head_from_target = False
                new_k = k
            new_weights[new_k] = v

        if self._use_mla:
            # Use DeepseekV3WeightLoader for proper MLA weight handling
            from .modeling_deepseekv3 import DeepseekV3WeightLoader
            weight_loader = DeepseekV3WeightLoader(self, is_draft_model=False)
            if self.load_lm_head_from_target:
                weight_loader.load_weights(new_weights,
                                           skip_modules=['lm_head'])
            else:
                weight_loader.load_weights(new_weights)
        else:
            if self.load_lm_head_from_target:
                super().load_weights(weights=new_weights,
                                     weight_mapper=weight_mapper,
                                     skip_modules=['lm_head'])
            else:
                super().load_weights(weights=new_weights,
                                     weight_mapper=weight_mapper)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Hack for eagle3. We might need to run a matmul to reduce
        the dimensionality of the hidden states on the first pass
        through the draft model. Shape dependent control flow will
        not work with CUDA graphs. So we have hoisted this logic out
        of the forward pass - the pyexecutor will call this function
        before running forward when applicable.
        """
        hidden_states = hidden_states.to(self.model.dtype)

        expected_hidden_size = self.model.hidden_size
        if hidden_states.shape[-1] != expected_hidden_size:
            if self.model.fc_norm is not None:
                chunks = hidden_states.chunk(len(self.model.fc_norm), dim=-1)
                hidden_states = torch.cat([
                    norm(chunk)
                    for norm, chunk in zip(self.model.fc_norm, chunks)
                ],
                                          dim=-1)
            elif self.model._norm_before_fc:
                hidden_states = self.model.input_norm(hidden_states)
            hidden_states = self.model.fc(hidden_states)

        return hidden_states


class MistralLarge3DraftModel(DecoderModel):

    def __init__(
        self,
        model_config: ModelConfig,
        start_layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__(model_config)

        from .modeling_deepseekv3 import DeepseekV3DecoderLayer
        config = model_config.pretrained_config
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping
        self.num_layers = model_config.pretrained_config.num_hidden_layers

        self.fc = Linear(
            self.hidden_size * 2,
            config.hidden_size,
            bias=getattr(config, "bias", False),
            dtype=config.torch_dtype,
            quant_config=model_config.get_quant_config(),
        )
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(model_config, start_layer_idx,
                                   aux_stream_dict)
        ])

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)
        self.embed_tokens = None

    def post_load_weights(self):
        self.layers[0].next_layer_layernorm = self.norm

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        hidden_states: torch.Tensor | None = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None
            inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)

        assert hidden_states is not None

        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            # NOTE: If hidden states from the target model have to be concatenated,
            # we expect that to happen outside the model definition. This helps us
            # avoid data-dependent control flow and gives us better CUDA graph
            # coverage.
            residual = None
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            hidden_states, residual = self.layers[0](
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=None,
                spec_metadata=spec_metadata)
        finally:
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens

        return hidden_states, hidden_states


# We use MistralLarge3 as the base architecture for EAGLE3 draft layers
# NOTE: Class name says "Eagle" not "Eagle3" to match checkpoint naming (e.g., "Mistral-Large-3-675B-Instruct-2512-Eagle")
@register_auto_model("MistralLarge3EagleForCausalLM")
class MistralLarge3EagleForCausalLM(DecoderModelForCausalLM):

    def __init__(
        self,
        model_config: ModelConfig,
        start_layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        draft_vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(MistralLarge3DraftModel(model_config, start_layer_idx,
                                                 aux_stream_dict),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=draft_vocab_size)
        self.load_lm_head_from_target = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: SpecMetadata | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = spec_metadata.get_hidden_states()
        output, _ = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            hidden_states=hidden_states,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict, *args, **kwargs):
        from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import \
            MistralLarge3WeightMapper
        params_map = kwargs.get("params_map")
        weight_mapper = MistralLarge3WeightMapper()
        if params_map is None:
            params_map = weight_mapper.mistral_llm_mapping

        llm_weights = weight_mapper.rename_by_params_map(weights=weights,
                                                         params_map=params_map)
        from .modeling_deepseekv3 import DeepseekV3WeightLoader
        weight_loader = DeepseekV3WeightLoader(self, is_draft_model=False)
        weight_loader.load_weights(llm_weights, skip_modules=['lm_head'])

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.model.dtype)
        return hidden_states


class PARDForCausalLM(nn.Module):
    """Draft model wrapper for PARD (Parallel Draft) speculative decoding.

    See PARDWorker for the full algorithm description.
    """

    def __init__(self, draft_config):
        super().__init__()
        DraftModelClass, _ = get_model_architecture(
            draft_config.pretrained_config)

        # Remove spec_config to prevent recursive spec-dec initialization
        draft_config_no_spec = replace(draft_config,
                                       spec_config=None,
                                       lm_head_gather_output=False)

        # Weights will be loaded later by ModelLoader.load_draft_weights()
        self.draft_model_full = DraftModelClass(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head

        # Required by weight mappers
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config

        # Fall back: pard_token -> mask_token_id -> vocab_size
        pretrained_config = draft_config.pretrained_config
        self.mask_token_id = getattr(
            pretrained_config, 'pard_token',
            getattr(pretrained_config, 'mask_token_id',
                    pretrained_config.vocab_size))
        logger.info(
            f"PARD draft model initialized with mask_token_id: {self.mask_token_id}"
        )

        self.logits_processor = None  # Set by caller after construction

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the PARD draft model."""
        self.draft_model_full.load_weights(weights=weights,
                                           weight_mapper=weight_mapper,
                                           **kwargs)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_out = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        return hidden_states_out, hidden_states_out


class MTPForCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
        lm_head: nn.Module = None,
        model: nn.Module = None,
    ):
        super().__init__()
        # Import here to avoid circular import
        model_type = model_config.pretrained_config.model_type
        mtp_layer = None
        match model_type:
            case "glm4_moe":
                from .modeling_glm import Glm4MTP
                mtp_layer = Glm4MTP
            case "deepseek_v3" | "deepseek_v32" | "glm_moe_dsa":
                from .modeling_deepseekv3 import DeepseekV3MTP
                mtp_layer = DeepseekV3MTP
            case "exaone_moe":
                from .modeling_exaone_moe import ExaoneMoeMTP
                mtp_layer = ExaoneMoeMTP
            case "nemotron_h" | "nemotron_h_puzzle":
                from .modeling_nemotron_h import NemotronHMTP
                mtp_layer = NemotronHMTP
            case "qwen3_next" | "qwen3_5_text" | "qwen3_5_moe_text":
                from .modeling_qwen3_next import Qwen3NextMTP
                mtp_layer = Qwen3NextMTP
            case "qwen4_exp_text":
                from .modeling_qwen4_exp import Qwen4ExpMTP
                mtp_layer = Qwen4ExpMTP
            case "step3p7" | "step3p5":
                from .modeling_step3p7 import Step3p7MTP
                mtp_layer = Step3p7MTP
            case "deepseek_v4":
                from .modeling_deepseekv4 import DeepseekV4MTP
                mtp_layer = DeepseekV4MTP
            case _:
                raise ValueError(
                    f"Model type {model_type} not supported for MTP")

        spec_dec_mode = model_config.spec_config.spec_dec_mode
        assert spec_dec_mode.is_mtp_one_model()
        checkpoint_mtp_num_layers = model_config.pretrained_config.num_nextn_predict_layers
        if spec_dec_mode.is_mtp_eagle_one_model():
            mtp_num_layers = 1
            mtp_repeat_count = model_config.spec_config.max_draft_len
        else:
            mtp_num_layers = min(model_config.spec_config.max_draft_len,
                                 checkpoint_mtp_num_layers)
            mtp_repeat_count = 1

        moe_load_balancer_set_repeated_for_next_layer(mtp_repeat_count)

        self.mtp_layers = nn.ModuleList([
            mtp_layer(model_config, layer_idx + start_layer_idx,
                      model.aux_stream_dict)
            for layer_idx in range(mtp_num_layers)
        ])
        self.lm_head = lm_head
        self.embed_tokens = model.embed_tokens


class MTPDraftModel(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        # Import here to avoid circular import
        model_type = model_config.pretrained_config.model_type
        if model_type == "glm4_moe":
            from .modeling_glm import Glm4MTP
            mtp_layer = Glm4MTP(model_config,
                                layer_idx,
                                aux_stream_dict,
                                is_separate_draft_engine=True)
        elif model_type in ["deepseek_v3", "deepseek_v32", "glm_moe_dsa"]:
            from .modeling_deepseekv3 import DeepseekV3MTP
            mtp_layer = DeepseekV3MTP(model_config,
                                      layer_idx,
                                      aux_stream_dict,
                                      is_separate_draft_engine=True)
        elif model_type in ["exaone_moe"]:
            from .modeling_exaone_moe import ExaoneMoeMTP
            mtp_layer = ExaoneMoeMTP(model_config, layer_idx, aux_stream_dict)
        else:
            raise ValueError(
                f"MTPDraftModel does not support model_type: {model_type}")
        setattr(self, f"layers.{layer_idx}", mtp_layer)
        self.layers = mtp_layer
        self.layer_idx = layer_idx
        self.config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.config.torch_dtype,
        )

    def __repr__(self):
        """Custom string representation to display layer index"""
        return f"(layers): ({self.layer_idx}): {repr(self.layers)}"

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layers(
            input_ids,
            position_ids,
            hidden_states,
            embed_tokens=self.embed_tokens,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=all_rank_num_tokens,
            spec_metadata=spec_metadata,
        )

        return hidden_states


@register_auto_model("MTPDraftModelForCausalLM")
class MTPDraftModelForCausalLM(DecoderModelForCausalLM[MTPDraftModel,
                                                       PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        self.model_config = model_config
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }
        super().__init__(
            MTPDraftModel(self.model_config,
                          self.model_config.pretrained_config.num_hidden_layers,
                          self.aux_stream_dict),
            config=self.model_config,
            hidden_size=self.model_config.pretrained_config.hidden_size,
            vocab_size=self.model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):
        # Import here to avoid circular import
        model_type = self.model_config.pretrained_config.model_type
        match model_type:
            case "glm4_moe":
                from .modeling_glm import Glm4WeightLoader
                weight_loader = Glm4WeightLoader(self, is_draft_model=True)
            case "deepseek_v3" | "deepseek_v32" | "glm_moe_dsa":
                from .modeling_deepseekv3 import DeepseekV3WeightLoader
                weight_loader = DeepseekV3WeightLoader(self,
                                                       is_draft_model=True)
            case "exaone_moe":
                raise ValueError(
                    f"Model type {model_type} not supported for MTP for two engine mode. Please use one engine mode instead."
                )
            case _:
                raise ValueError(
                    f"Model type {model_type} not supported for MTP")
        weight_loader.load_weights(weights)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: torch.IntTensor = None,
                position_ids: torch.IntTensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                return_context_logits: bool = False,
                spec_metadata: Optional[SpecMetadata] = None,
                hidden_states: torch.Tensor = None,
                **kwargs) -> torch.Tensor:

        hidden_states = spec_metadata.get_hidden_states()
        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            spec_metadata=spec_metadata,
            **kwargs)
        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )


def external_drafter_config_kwargs(model_config, spec_config) -> dict:
    """`ModelConfig.from_pretrained` kwargs for a one-model external drafter.

    The drafter is a separate checkpoint, so it gets its own `ModelConfig`; the
    kwargs below are the execution-layout properties it must inherit from the
    target engine it runs inside.

    `moe_load_balancer` is propagated for DSpark ONLY. DSpark's draft stages are
    full DeepSeek-V4 blocks sharing the target's expert topology and layer-index
    namespace (`layer_idx = num_hidden_layers + stage_id`), so they can register
    into the target's EPLB manager. Other external drafters (PARD, DFlash,
    draft-target) are independent checkpoints whose expert topology and layer
    numbering need not match the target's, and whose EPLB configs would therefore
    be keyed against a different namespace -- do not generalize this without
    designing a per-drafter EPLB config domain and layer identity first.
    """
    kwargs = dict(
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=model_config.moe_backend,
        mapping=model_config.mapping,
        spec_config=None,  # Avoid recursive spec-dec
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )
    # Only the embedded DSpark draft shares the target's EPLB namespace (its
    # stages are target decoder blocks registered into the target's balancer).
    # A standalone DSpark drafter is an independent checkpoint, so it falls
    # under the "other external drafters" rule above.
    if (spec_config.spec_dec_mode.is_dspark()
            and spec_config.draft_is_embedded_in_target):
        kwargs["moe_load_balancer"] = model_config.moe_load_balancer
    return kwargs


@register_draft_model(SpeculativeDecodingMode.EAGLE3_ONE_MODEL)
def _build_eagle3_one_model_draft(model_config, draft_config, lm_head, model):
    """Build the EAGLE3 one-model drafter for the configured draft arch."""
    eagle3_model_arch = model_config.spec_config.eagle3_model_arch
    if eagle3_model_arch == "llama3":
        # Eagle3ForCausalLM handles both Llama3 and DeepSeekV3 architectures
        return Eagle3ForCausalLM(
            draft_config, model_config.pretrained_config.num_hidden_layers)
    elif eagle3_model_arch == "mistral_large3":
        return MistralLarge3EagleForCausalLM(
            draft_config, model_config.pretrained_config.num_hidden_layers,
            model.aux_stream_dict)
    else:
        raise ValueError(
            f"Unsupported eagle3 model architecture: {eagle3_model_arch}")


@register_draft_model(SpeculativeDecodingMode.MTP)
@register_draft_model(SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL)
def _build_mtp_one_model_draft(model_config, draft_config, lm_head, model):
    """Build the one-model MTP drafter (vanilla MTP and MTP-Eagle share it)."""
    return MTPForCausalLM(model_config,
                          model_config.pretrained_config.num_hidden_layers,
                          lm_head, model)


@register_draft_model(SpeculativeDecodingMode.MTP_EAGLE)
def _build_mtp_eagle_draft(model_config, draft_config, lm_head, model):
    """Build the two-model MTP-Eagle drafter."""
    return MTPDraftModelForCausalLM(model_config)


@register_draft_model(SpeculativeDecodingMode.PARD)
def _build_pard_draft(model_config, draft_config, lm_head, model):
    """Build the PARD drafter."""
    return PARDForCausalLM(draft_config)


@register_draft_model(SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL)
def _build_draft_target_one_model_draft(model_config, draft_config, lm_head,
                                        model):
    """Build the one-model draft-target drafter from its own checkpoint."""
    # Keep the draft LM head vocab-sharded so greedy draft sampling uses the
    # lighter TP gather (see SpecWorkerBase.greedy_sample_draft_with_tp_gather).
    was_frozen = draft_config._frozen
    draft_config._frozen = False
    draft_config.lm_head_gather_output = False
    draft_config._frozen = was_frozen
    return AutoModelForCausalLM.from_config(draft_config)


def get_draft_model(model_config, draft_config, lm_head, model):
    """Construct the draft model for the configured speculative-decoding mode.

    Dispatch is registry-based: each mode's builder lives next to the draft
    model it constructs and registers itself via ``@register_draft_model``, so
    this function never imports a concrete draft implementation (which is what
    used to force a lazy import for DSpark, whose provider imports back into
    this module through modeling_deepseekv4).

    Args:
        model_config: the target engine's ``ModelConfig``, carrying spec_config.
        draft_config: the drafter's own ``ModelConfig``, or None when the mode
            builds its draft from the target config alone.
        lm_head: the target's LM head, shared by the one-model MTP drafter.
        model: the target model, for drafters reusing its aux streams.

    Returns:
        The draft ``nn.Module`` for this mode.
    """
    assert getattr(model_config, 'spec_config', None) is not None
    spec_config = model_config.spec_config
    spec_dec_mode = spec_config.spec_dec_mode
    # An external draft model is loaded straight from its own checkpoint, so it
    # has no mode-specific builder to register: this stays an explicit pre-check
    # ahead of the registry lookup rather than becoming a registry key.
    #
    # No mode guard is needed. `uses_external_draft_model` already implies
    # `is_mtp_one_model()` (llm_args), which is disjoint from every other mode,
    # so this branch cannot divert a drafter that a builder would have claimed.
    # Pinned by test_draft_model_registry.py::
    # test_external_draft_model_bypasses_the_registry.
    if spec_config.uses_external_draft_model:
        if draft_config is None:
            raise ValueError(
                "MTP speculative decoding with an external draft model requires "
                "its model config.")
        return AutoModelForCausalLM.from_config(draft_config)
    builder = get_registered_draft_model_builder(spec_dec_mode)
    if builder is None:
        raise NotImplementedError(
            f"get_draft_model does not support speculative decoding mode {spec_dec_mode}."
        )
    return builder(model_config, draft_config, lm_head, model)


class SpecDecOneEngineForCausalLM(DecoderModelForCausalLM[TModel, TConfig],
                                  Generic[TModel, TConfig]):

    def __init__(self,
                 model: TModel,
                 model_config: ModelConfig[TConfig],
                 hidden_size: int | None = None,
                 vocab_size: int | None = None) -> None:
        # Composite configs (e.g. vision-language wrappers) may not expose
        # hidden_size/vocab_size at the top level; callers can pass the
        # text-config values explicitly.
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        if vocab_size is None:
            vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(model,
                         config=model_config,
                         hidden_size=hidden_size,
                         vocab_size=vocab_size)
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        spec_config = getattr(model_config, 'spec_config', None)
        self.spec_config = spec_config
        if spec_config and spec_config.spec_dec_mode.use_one_engine():
            # Only create draft_model for modes MTP, Eagle3 (not SA)
            if not spec_config.spec_dec_mode.is_sa():
                if spec_config.spec_dec_mode.is_eagle3_one_model():
                    if spec_config.eagle3_model_arch == "mistral_large3":
                        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
                            MistralConfigLoader
                        self.draft_config = MistralConfigLoader().load(
                            spec_config.speculative_model,
                            mapping=model_config.mapping,
                            moe_backend=model_config.moe_backend,
                            moe_max_num_tokens=model_config.moe_max_num_tokens,
                            max_num_tokens=model_config.max_num_tokens,
                            moe_load_balancer=model_config.moe_load_balancer,
                            skip_create_weights_in_init=True,
                        )
                    elif spec_config.eagle3_model_arch == "llama3":
                        self.draft_config = ModelConfig.from_pretrained(
                            model_config.spec_config.speculative_model,
                            trust_remote_code=True,
                            attn_backend=model_config.attn_backend,
                            moe_backend=model_config.moe_backend,
                            mapping=model_config.mapping,
                            spec_config=model_config.spec_config,
                            max_num_tokens=model_config.max_num_tokens,
                            moe_max_num_tokens=model_config.moe_max_num_tokens)
                    else:
                        raise ValueError(
                            f"Unsupported eagle3 model architecture for draft model: {spec_config.eagle3_model_arch}"
                        )
                    self.draft_config.quant_config.kv_cache_quant_algo = \
                    model_config.quant_config.kv_cache_quant_algo
                    self.draft_config.extra_attrs = model_config.extra_attrs

                elif spec_config.uses_external_draft_model:
                    self.draft_config = ModelConfig.from_pretrained(
                        spec_config.speculative_model,
                        trust_remote_code=True,
                        attn_backend=model_config.attn_backend,
                        moe_backend=model_config.moe_backend,
                        mapping=model_config.mapping,
                        spec_config=None,
                        max_num_tokens=model_config.max_num_tokens,
                        moe_max_num_tokens=model_config.moe_max_num_tokens)
                    self.draft_config.quant_config.kv_cache_quant_algo = \
                        model_config.quant_config.kv_cache_quant_algo
                    self.draft_config.extra_attrs = dict(
                        model_config.extra_attrs)
                    self.draft_config.extra_attrs[
                        _SPECULATIVE_POSITION_HEADROOM] = (
                            2 * spec_config.tokens_per_gen_step)

                elif spec_config.spec_dec_mode.is_external_drafter():
                    self.draft_config = ModelConfig.from_pretrained(
                        model_config.spec_config.speculative_model,
                        **external_drafter_config_kwargs(
                            model_config, spec_config))
                    self.draft_config.quant_config.kv_cache_quant_algo = \
                        model_config.quant_config.kv_cache_quant_algo
                    self.draft_config.extra_attrs = model_config.extra_attrs

                self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(
                    spec_config)

                self.draft_model = get_draft_model(model_config,
                                                   self.draft_config,
                                                   self.lm_head, self.model)
                if self.draft_model is not None:
                    self.epilogue.append(self.draft_model)
                if (spec_config.spec_dec_mode.is_parallel_draft()
                    ) and self.draft_model is not None:
                    self.draft_model.logits_processor = self.logits_processor

            # spec_worker is created for all one-engine modes (MTP, Eagle3, SA)
            self.spec_worker = get_spec_worker(
                model_config.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache)
            if self.spec_worker is not None:
                # Cache the static draft->target vocab map now that the draft
                # model is loaded, so workers read self._d2t instead of probing
                # draft_model.model.d2t on every forward.
                self.spec_worker.set_draft_model(self.draft_model)
                self.epilogue.append(self.spec_worker)
        self.layer_idx = -1

    def setup_aliases(self) -> None:
        if (self.draft_model is not None
                and getattr(self.draft_model, "shares_target_kv_cache", False)):
            self.draft_model.load_weights_from_target_model(self)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[:attn_metadata.num_tokens]

        if self.spec_worker is not None:
            # get logits
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )

            # VLM wrappers (e.g. Qwen3VLModelBase) replace input_ids with
            # fused inputs_embeds; fall back to the pre-fusion token IDs
            # they forward via `orig_input_ids` so MTP / Eagle drafters
            # can still access the prompt tokens.
            spec_input_ids = input_ids if input_ids is not None else kwargs.get(
                "orig_input_ids")
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_input_ids is not None:
                    # Slice along the first dimension
                    spec_input_ids = spec_input_ids[:attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens)

            # get accepted tokens and next draft tokens
            return self.spec_worker(input_ids=spec_input_ids,
                                    position_ids=spec_position_ids,
                                    hidden_states=hidden_states,
                                    logits=logits,
                                    attn_metadata=attn_metadata,
                                    spec_metadata=spec_metadata,
                                    draft_model=self.draft_model,
                                    resource_manager=resource_manager)
        else:
            logits = self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )

        return logits

    def mtp_head_module_names(self) -> List[str]:
        """Names of the MTP heads under every alias they are reachable by.

        One-model MTP registers the same head objects twice: under
        ``draft_model.mtp_layers.{h}`` and, after the target model extends its
        layer list, under ``model.layers.{num_hidden_layers + h}``. A load that
        wants to leave the heads untouched has to exclude both aliases.
        """
        mtp_layers = getattr(self.draft_model, "mtp_layers", None)
        if not mtp_layers:
            return []
        head_ids = {id(layer) for layer in mtp_layers}
        return [
            name for name, module in self.named_modules(remove_duplicate=False)
            if name and id(module) in head_ids
        ]

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional[BaseWeightMapper] = None,
                     params_map: Optional[Dict[str, str]] = None,
                     allow_partial_loading: bool = False):
        from tensorrt_llm._torch.speculative.utils import (
            filter_mtp_checkpoint_weights, uses_mtp_head_checkpoint)

        skip_modules = ["draft_model"]
        if uses_mtp_head_checkpoint(self.spec_config):
            # The heads come from speculative_model in a second pass
            # (load_draft_weights), so exclude them here. They must be
            # *skipped* rather than tolerated via allow_partial_loading:
            # partial loading suppresses process_weights_after_loading() on
            # every quantized Linear/MoE it touches, which would leave the
            # target model's quant scales (NVFP4 alphas, MoE input scales)
            # uninitialized.
            weights = filter_mtp_checkpoint_weights(weights)
            skip_modules.extend(self.mtp_head_module_names())
        super().load_weights(weights=weights,
                             weight_mapper=weight_mapper,
                             skip_modules=skip_modules,
                             params_map=params_map,
                             allow_partial_loading=allow_partial_loading)

    def load_draft_weights(self,
                           weights: Dict,
                           weight_mapper: Optional[BaseWeightMapper] = None):
        from tensorrt_llm._torch.models.modeling_utils import \
            _load_weights_impl_v2
        from tensorrt_llm._torch.speculative.utils import (
            remap_preprocessed_mtp_weights_for_draft_model,
            select_mtp_checkpoint_weights,
            skip_modules_for_separate_mtp_checkpoint, uses_mtp_head_checkpoint)

        if uses_mtp_head_checkpoint(self.spec_config):
            # Load MTP heads into draft_model only, and verify every non-shared
            # MTP parameter has a matching tensor. The previous parent-model
            # load used allow_partial_loading=True, which silently left MTP
            # modules at random init when keys did not bind.
            n_total = len(weights)
            weights = select_mtp_checkpoint_weights(weights)
            if not weights:
                raise ValueError(
                    "speculative_model was set for MTP but no 'mtp.*' weights "
                    f"were found in {self.spec_config.speculative_model!r}. "
                    "Expected keys like 'mtp.layers.0.*'.")
            n_dropped = n_total - len(weights)
            if n_dropped:
                logger.warning(
                    "Ignoring %d non-mtp.* tensors from speculative_model while "
                    "loading MTP heads (kept %d mtp.* tensors).", n_dropped,
                    len(weights))
            if weight_mapper is None:
                raise ValueError(
                    "weight_mapper is required to load separate MTP heads")
            weights = weight_mapper.preprocess_weights(weights)
            num_hidden_layers = self.config.num_hidden_layers
            num_mtp_layers = len(self.draft_model.mtp_layers)
            weights = remap_preprocessed_mtp_weights_for_draft_model(
                weights,
                num_hidden_layers=num_hidden_layers,
                num_mtp_layers=num_mtp_layers,
            )

            # Skip optional modules (e.g. shared_head) only when absent from
            # this checkpoint; architectures that ship those tensors still load
            # them under allow_partial_loading=False.
            _load_weights_impl_v2(
                self.draft_model,
                weights,
                weight_mapper,
                skip_modules=skip_modules_for_separate_mtp_checkpoint(weights),
                allow_partial_loading=False,
            )
            return

        args = inspect.getfullargspec(self.draft_model.load_weights).args
        if "weight_mapper" in args:
            self.draft_model.load_weights(weights=weights,
                                          weight_mapper=weight_mapper)
        else:
            self.draft_model.load_weights(weights=weights)

        if self.spec_config and (
                not self.spec_config.spec_dec_mode.is_external_drafter()
                or self.spec_config.spec_dec_mode.is_dflash()
                or self.spec_config.spec_dec_mode.is_dspark()):
            self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
