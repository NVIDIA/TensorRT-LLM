# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared BF16 native DSpark draft model for Mistral Large 4 targets."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.activation.tanh_softcap import (
    tanh_softcap,
)

from .common import _rope_tables


class DraftModel(nn.Module):
    """This specialization's BF16 DSpark backbone and recurrent head.

    Runtime owns accepted-token bookkeeping and the paged context pool. The
    draft attends the last trained window of captured target features plus
    its whole non-causal query block. All parameters live in the core table.
    """

    _uses_worker_attention_backend = False
    _paged_ctx_cache = True
    _kv_factor = 1
    _dspark_shift_label = True
    has_markov_head = True

    def __init__(self, shell, draft_config, *, cublas_op=None):
        super().__init__()
        # Keep references without registering the parent as a child module.
        object.__setattr__(self, "_shell", shell)
        self._cublas_op = cublas_op
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        core = shell.model
        spec = shell.spec_config
        native = spec._draft_model_config
        self.block_size = spec.block_size
        self.mask_token_id = spec.mask_token_id
        self.target_layer_ids = tuple(spec.target_layer_ids)
        self.window = int(native["ragged_attention"].split(",")[0])
        self._num_attn_layers = int(native["n_layers"])
        self._num_heads = core.heads
        self._num_kv_heads = 1
        self._head_dim = core.lat_dim
        self.dflash_attention_backend = "VANILLA"
        self._runtime_position_ceiling = None
        self._rope = None
        self._layers = None

    @property
    def core(self):
        return self._shell.model

    @property
    def lm_head(self):
        return self._shell.lm_head

    @property
    def markov_w2(self):
        return self.core.w["draft_markov_output"]

    def logits_processor(self, hidden, lm_head, attn_metadata, gather):
        logits = self._shell.logits_processor(hidden, lm_head, attn_metadata, gather)
        # The shared-head logits are soft-capped before the RNN bias is added.
        return tanh_softcap(logits, self._shell.softcap)

    def apply_draft_logit_transform(self, logits):
        """Return ``logits`` unchanged: :meth:`logits_processor` already soft-caps them.

        The worker calls this on the block logits after ``logits_processor``.
        This drafter declares no ``logit_multiplier`` or ``logit_softcap``, so the
        cap is applied once, in ``logits_processor``.
        """
        return logits

    def _build_fused_kv_buffers(self):
        # The worker calls this before sizing its cache. Take views once after
        # materialization; context projection is explicit per layer here.
        if self._layers is not None:
            return
        w, core = self.core.w, self.core
        self._layers = []
        for i in range(self._num_attn_layers):
            prefix = f"l{i}_draft_"
            kv_b = w[prefix + "wkv_b"].view(core.heads, core.nope + core.v_dim, core.kv_lora)
            self._layers.append(
                SimpleNamespace(
                    attn_norm=w[prefix + "attn_norm"],
                    q=w[prefix + "wq"].t(),
                    kv_a=w[prefix + "wkv_a"].t(),
                    kv_norm=w[prefix + "kv_a_norm"],
                    k_b=kv_b[:, : core.nope].contiguous(),
                    v_b=kv_b[:, core.nope :].contiguous(),
                    out=w[prefix + "wo"].t(),
                    ffn_norm=w[prefix + "ffn_norm"],
                    gate=w[prefix + "w1"].t(),
                    up=w[prefix + "w3"].t(),
                    down=w[prefix + "w2"].t(),
                    next_norm=(
                        w[f"l{i + 1}_draft_attn_norm"]
                        if i + 1 < self._num_attn_layers
                        else w["draft_final_norm"]
                    ),
                )
            )

    def _ensure_rope(self):
        cap = self._runtime_position_ceiling or self.core.rope_rows
        if self._rope is None or self._rope.rows != cap:
            w = self.core.w["draft_final_norm"]
            # alpha=0 removes LeRoPE; all draft layers use plain YaRN.
            self._rope = _rope_tables(
                self.core.model_config.pretrained_config,
                torch.zeros(self.core.rope_dim // 2),
                cap,
                w.dtype,
                w.device,
            )

    def _rotate(self, x, positions):
        table = self._rope.indexer[positions.long()]
        cos, sin = table.chunk(2, dim=-1)
        # One head axis for Q, none for the shared latent K.
        if x.ndim == positions.ndim + 2:
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
            .flatten(-2)
            .to(x.dtype)
        )

    def embed_draft_block(self, anchors, block_size):
        # The route pins the trained block width; no mask token is looked up
        # in the vocabulary (its sentinel is deliberately out of range).
        weight = self.core.w["embed"]
        valid = (anchors >= 0) & (anchors < weight.shape[0])
        embedded = nn.functional.embedding(anchors.clamp(0, weight.shape[0] - 1), weight)
        embedded = embedded.masked_fill(~valid.unsqueeze(-1), 0)
        masks = self.core.w["draft_mask"].unsqueeze(0).expand(anchors.shape[0], -1, -1)
        return torch.cat((embedded.unsqueeze(1), masks), dim=1)

    def _mm(self, *args):
        return (self._cublas_op or cublas_mm)(*args)

    def project_target_hidden(self, hidden):
        w = self.core.w
        projected = self._mm(hidden, w["draft_target_projection"].t(), None, None)
        return flashinfer_rmsnorm(projected, w["draft_target_norm"], self.core.eps)

    def precompute_context_kv(self, projected_hidden, positions):
        self._build_fused_kv_buffers()
        self._ensure_rope()
        core = self.core
        latent = []
        for layer in self._layers:
            kv = self._mm(projected_hidden, layer.kv_a, None, None)
            content = flashinfer_rmsnorm(
                kv[:, : core.kv_lora].contiguous(), layer.kv_norm, core.eps
            )
            rope = self._rotate(kv[:, core.kv_lora :], positions.reshape(-1))
            latent.append(torch.cat((content, rope), dim=-1))
        return torch.stack(latent, dim=1).unsqueeze(2), None

    def dflash_forward_with_metadata(self, *, attn_metadata, **kwargs):
        # The phase belongs to this engine step, including mixed batches.
        return self.core._select_target(attn_metadata).dspark_forward(self, **kwargs)

    def apply_markov_chain_logits(self, base_logits, first_prev_tokens, *, hidden_states):
        """The recurrent Markov bias over full-vocab ``[B, K, vocab]`` block logits.

        Each position's greedy token (the argmax of its biased logits) is the
        next position's previous token; ``hidden_states`` is the ``[B, K,
        hidden]`` block output the logits were projected from.
        """
        w = self.core.w
        output = w["draft_markov_output"]
        previous = first_prev_tokens.long()
        state = None
        refined = []
        for i in range(base_logits.shape[1]):
            weight = w["draft_markov_embedding"]
            valid = (previous >= 0) & (previous < weight.shape[0])
            embedding = nn.functional.embedding(previous.clamp(0, weight.shape[0] - 1), weight)
            embedding = embedding.masked_fill(~valid.unsqueeze(-1), 0)
            if state is None:
                state = torch.zeros_like(embedding)
            joint = self._mm(
                torch.cat((state, embedding, hidden_states[:, i]), dim=-1),
                w["draft_joint"].t(),
                w["draft_joint_bias"],
                None,
            )
            gate, candidate, emitted = joint.chunk(3, dim=-1)
            gate = torch.sigmoid(gate)
            state = gate * state + (1 - gate) * torch.tanh(candidate)
            bias = self._mm(torch.tanh(emitted), output.t(), None, None)
            logits = base_logits[:, i] + bias
            refined.append(logits)
            previous = logits.argmax(dim=-1)
        return torch.stack(refined, dim=1)
