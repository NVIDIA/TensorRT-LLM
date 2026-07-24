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
"""Qwen3-based DSpark draft model (DeepSeek DeepSpec ``Qwen3DSparkModel``).

Unlike the DeepSeek-V4 DSpark drafter (``modeling_dspark.py``) — whose draft
weights live in the target checkpoint's ``mtp.*`` namespace and whose stages
are full V4 blocks (MLA + MoE + mHC) — the DeepSpec-released Qwen3 drafters
(e.g. ``deepseek-ai/dspark_qwen3_8b_block7``) are separate dense bf16
checkpoints with a flat namespace:

  - ``fc`` + ``hidden_norm``: project the concatenated captured target-layer
    hidden states (``hidden_size * num_capture_layers``) into the draft hidden
    size — the layer-invariant *context stream* (``main_x``).
  - ``layers.{0..n-1}``: standard Qwen3 GQA decoder layers (q/k per-head
    RMSNorm, RoPE, gated-SiLU MLP), except attention keys/values come from
    BOTH the projected context (one row per committed token) and the draft
    block itself, and the ``block_size`` draft queries attend bidirectionally
    within the block.
  - ``norm`` + (target-shared) ``lm_head`` -> :func:`dspark_propose` (Markov
    head refinement; the confidence head is inert scaffolding, as in the V4
    path).

The worker-facing protocol (``write_context_windows`` /
``write_context_windows_batched`` / ``forward_batched`` / ``_attn_params`` /
``num_stages`` / ``run_moe_lockstep_noop``) matches :class:`DSparkDraftModel`,
so :class:`DSparkWorker` and :class:`DSparkSpecMetadata` drive both drafters
unchanged. The worker-owned rolling buffer here holds per-layer context K/V
(``[max_batch, num_layers, window, 2 * num_kv_heads * head_dim]``, K then V,
K stored RoPE'd/k-normed). Frame convention: the worker passes window frames
``f = absolute_position + 1``; this model stores the context row of the token
at absolute position ``p`` at ring index ``p % window`` with RoPE phase ``p``,
matching the DeepSpec reference (its draft ``DynamicCache`` holds the context
K/V of positions ``0..start-1`` in order; the ring is an O(1)-memory window
over the most recent ``window`` positions — acceptance-rate only, standard
target verification keeps outputs correct regardless).

Reference: https://github.com/deepseek-ai/DeepSpec
(``deepspec/modeling/dspark/qwen3/modeling.py`` and
``deepspec/eval/dspark/draft_ops.py``).
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger

from .dspark.draft import build_draft_input_ids, dspark_propose
from .dspark.heads import build_markov_head

# Ring-window length for the per-layer context K/V cache (per request slot).
# The DeepSpec reference attends over the FULL committed context; a ring
# window bounds the worker buffer at
# ``max_batch * num_layers * window * 2 * kv_dim`` bytes while remaining
# exactly full-context for sequences up to ``window`` tokens.
_DEFAULT_CTX_WINDOW = 2048
_CTX_WINDOW_ENV = "TRTLLM_DSPARK_QWEN3_CTX_WINDOW"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF-style rotary embedding on ``[..., num_heads, head_dim]``.

    ``cos``/``sin`` are ``[..., head_dim]`` (one row per position) and are
    broadcast over the heads axis.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return x * cos + _rotate_half(x) * sin


class _RMSNorm(nn.Module):
    """Qwen3RMSNorm: fp32 normalize, cast back, then scale by the bf16 weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dt)


class _Attention(nn.Module):
    """Weight container for one draft layer's GQA attention (names match ckpt)."""

    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.q_norm = _RMSNorm(head_dim, eps)
        self.k_norm = _RMSNorm(head_dim, eps)


class _MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        eps: float,
    ):
        super().__init__()
        self.self_attn = _Attention(hidden, n_heads, n_kv_heads, head_dim, eps)
        self.mlp = _MLP(hidden, intermediate)
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)


class _ConfidenceHead(nn.Module):
    """DeepSpec ``AcceptRatePredictor`` (fp32 linear WITH bias for the Qwen3
    drafters). Inert scaffolding — the worker always passes
    ``confidence_threshold=0.0`` — kept for checkpoint completeness."""

    def __init__(self, input_dim: int, with_markov: bool):
        super().__init__()
        self.with_markov = bool(with_markov)
        self.proj = nn.Linear(int(input_dim), 1, bias=True, dtype=torch.float32)

    def forward(
        self, hidden_states: torch.Tensor, prev_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.with_markov:
            assert prev_embeddings is not None
            features = torch.cat([hidden_states, prev_embeddings.to(hidden_states.dtype)], dim=-1)
        else:
            features = hidden_states
        return self.proj(features.float()).squeeze(-1)


class Qwen3DSparkDraftModel(nn.Module):
    """The dense Qwen3 DSpark draft backbone + heads (DeepSpec reference math).

    Shares ``embed_tokens`` / ``lm_head`` with the target model (the checkpoint
    carries frozen copies of both; they are identical to the target's, so the
    shared modules are used and the copies skipped at load).
    """

    def __init__(self, model_config, block_size: Optional[int] = None):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config

        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or self.hidden_size // self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.softmax_scale = self.head_dim**-0.5
        eps = float(config.rms_norm_eps)
        num_layers = int(config.num_hidden_layers)
        # Worker-facing stage count (== draft layer count for this drafter).
        self.num_stages = num_layers

        spec_cfg = getattr(model_config, "spec_config", None)
        self.block_size = int(
            block_size if block_size is not None else getattr(config, "block_size", 7)
        )
        mask_token_id = getattr(spec_cfg, "mask_token_id", None)
        self.noise_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else getattr(config, "mask_token_id", config.vocab_size)
        )

        target_layer_ids = list(getattr(config, "target_layer_ids", []) or [])
        self.num_capture_layers = len(target_layer_ids)
        assert self.num_capture_layers > 0, (
            "Qwen3 DSpark drafter config must provide target_layer_ids"
        )

        # Plain-RoPE parameters. transformers>=5 nests rope_theta under
        # rope_parameters; older versions keep the flat attribute.
        rope_params = getattr(config, "rope_parameters", None) or {}
        self._rope_theta = float(
            rope_params.get("rope_theta", None) or getattr(config, "rope_theta", 1000000.0)
        )
        max_pos = int(getattr(config, "max_position_embeddings", 40960))
        self._freqs_cap = max_pos + self.block_size + 2
        self._rope_cache: Dict[str, tuple] = {}

        # Ring-window length for the worker-owned context K/V buffer.
        window = int(os.environ.get(_CTX_WINDOW_ENV, _DEFAULT_CTX_WINDOW))
        window = max(self.block_size + 2, min(window, max_pos))
        # Worker allocation contract (DSparkWorker._lazy_init):
        # kv_windows = [max_batch, num_stages, window_size, head_dim], where
        # "head_dim" is the per-position row width — here K||V flattened.
        self._attn_params = dict(
            window_size=window,
            head_dim=2 * self.kv_dim,
        )

        # Build the weight modules on the meta device; load_weights() assigns
        # the real cuda tensors (torch load_state_dict(assign=True)).
        with torch.device("meta"):
            self.fc = nn.Linear(
                self.num_capture_layers * self.hidden_size, self.hidden_size, bias=False
            )
            self.hidden_norm = _RMSNorm(self.hidden_size, eps)
            self.layers = nn.ModuleList(
                [
                    _DecoderLayer(
                        self.hidden_size,
                        int(config.intermediate_size),
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        eps,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = _RMSNorm(self.hidden_size, eps)
            self.markov_head = build_markov_head(
                markov_head_type=str(getattr(config, "markov_head_type", "vanilla")),
                vocab_size=int(config.vocab_size),
                markov_rank=int(getattr(config, "markov_rank", 0)),
                hidden_size=self.hidden_size,
            )
            self.confidence_head = None
            if bool(getattr(config, "enable_confidence_head", False)):
                with_markov = bool(getattr(config, "confidence_head_with_markov", False))
                input_dim = self.hidden_size + (int(config.markov_rank) if with_markov else 0)
                self.confidence_head = _ConfidenceHead(input_dim, with_markov)

        # Shared with target; wired by the spec wrapper after construction.
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

    # ------------------------------------------------------------------ RoPE

    def _cos_sin_tables(self, device: torch.device):
        key = str(device)
        cached = self._rope_cache.get(key)
        if cached is None:
            inv_freq = 1.0 / (
                self._rope_theta
                ** (
                    torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                    / self.head_dim
                )
            )
            t = torch.arange(self._freqs_cap, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cached = (emb.cos(), emb.sin())
            self._rope_cache[key] = cached
        return cached

    def _gather_cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        cos_t, sin_t = self._cos_sin_tables(positions.device)
        # Clamp for graph-safety: masked-out entries may carry arbitrary
        # (already clamped by the worker) positions.
        p = positions.long().clamp(min=0, max=self._freqs_cap - 1)
        return cos_t[p].to(dtype), sin_t[p].to(dtype)

    # ---------------------------------------------------------- context K/V

    def _project_ctx(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(captured))`` — the layer-invariant context stream."""
        return self.hidden_norm(self.fc(main_hidden))

    def _ctx_kv(
        self, layer: _DecoderLayer, main_x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-layer context K/V rows ``[..., 2*kv_dim]`` (K RoPE'd/normed || V)."""
        a = layer.self_attn
        shp = main_x.shape[:-1]
        k = a.k_proj(main_x).view(*shp, self.num_kv_heads, self.head_dim)
        k = a.k_norm(k)
        cos, sin = self._gather_cos_sin(positions, k.dtype)
        k = _apply_rope(k, cos, sin)
        v = a.v_proj(main_x)
        return torch.cat([k.reshape(*shp, self.kv_dim), v], dim=-1)

    @torch.inference_mode()
    def write_context_windows(
        self, main_hidden: torch.Tensor, positions: torch.Tensor, stage_windows: torch.Tensor
    ) -> None:
        """Seed one request's ring windows from captured context (prefill path).

        Args:
            main_hidden: ``[M, num_capture * hidden]`` captured target hiddens.
            positions: ``[M]`` window frames (= absolute position + 1, the
                worker's generation-path convention).
            stage_windows: ``[num_layers, window, 2*kv_dim]``, updated in place.
        """
        M = int(main_hidden.shape[0])
        if M == 0:
            return
        win = int(self._attn_params["window_size"])
        p = positions.to(main_hidden.device).long() - 1
        cols = p % win
        main_x = self._project_ctx(main_hidden)
        for li, layer in enumerate(self.layers):
            kv = self._ctx_kv(layer, main_x, p)
            stage_windows[li, cols] = kv.to(stage_windows.dtype)

    def write_context_windows_batched(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        mask: torch.Tensor,
        kv_windows: torch.Tensor,
    ) -> None:
        """CUDA-graph-safe masked back-fill of interim accepted tokens.

        Args:
            main_hidden: ``[G, M, num_capture * hidden]``.
            positions: ``[G, M]`` window frames (masked entries arbitrary).
            slots: ``[G]`` request rows into ``kv_windows``.
            mask: ``[G, M]`` bool validity.
            kv_windows: ``[N, num_layers, window, 2*kv_dim]``, in place.
        """
        G, M = positions.shape
        if G == 0 or M == 0:
            return
        win = int(self._attn_params["window_size"])
        p = positions.long() - 1
        cols = p % win
        rows = slots.long()[:, None].expand(-1, M)
        mask3 = mask.unsqueeze(-1)
        main_x = self._project_ctx(main_hidden)
        for li, layer in enumerate(self.layers):
            kv = self._ctx_kv(layer, main_x, p)  # [G, M, 2*kv_dim]
            win_l = kv_windows[:, li]  # [N, win, 2*kv_dim] view
            cur = win_l[rows, cols]
            win_l[rows, cols] = torch.where(mask3, kv.to(win_l.dtype), cur)

    # ------------------------------------------------------------- backbone

    def forward_batched(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        kv_windows: torch.Tensor,
        slots: torch.Tensor,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> tuple:
        """CUDA-graph-safe batched block draft (all gen requests at once).

        Mirrors DeepSpec ``forward_dspark_draft_block`` + ``build_dspark_proposal``:
        write the newest committed token's context K/V (position ``start_pos-1``),
        run the ``block_size`` draft queries ``[bonus, mask, ...]`` at positions
        ``start_pos + [0..block)`` through the Qwen3 layers attending to the ring
        context + the block (bidirectional), then Markov-refine the block logits.

        Args:
            main_hidden: ``[G, num_capture * hidden]`` captured hidden of the
                newest committed context token per request.
            bonus_token_ids: ``[G]`` last accepted token per request.
            start_pos: ``[G]`` absolute decode position of the bonus token.
            kv_windows: ``[N, num_layers, window, 2*kv_dim]`` persistent buffer.
            slots: ``[G]`` request rows into ``kv_windows``.
        Returns:
            ``(draft_tokens [G, block], num_proposed [G])`` and, with
            ``return_logits``, the corrected block logits ``[G, block, vocab]``.
        """
        del all_rank_num_tokens  # dense drafter: no cross-rank MoE lockstep
        G = int(main_hidden.shape[0])
        B = self.block_size
        win = int(self._attn_params["window_size"])
        device = main_hidden.device
        start_pos = start_pos.long()

        main_x = self._project_ctx(main_hidden)  # [G, hidden]
        p0 = start_pos - 1
        cols0 = p0 % win

        draft_ids = build_draft_input_ids(
            bonus_token_ids, block_size=B, noise_token_id=self.noise_token_id
        )
        h = self.embed_tokens(draft_ids)  # [G, B, hidden]

        pos_q = start_pos.unsqueeze(1) + torch.arange(B, device=device)  # [G, B]
        cos_q, sin_q = self._gather_cos_sin(pos_q, h.dtype)

        # Bool attention mask [G, 1, B, win + B]: ring rows valid iff their
        # index < min(start_pos, win) (ring holds the last `win` positions);
        # the block attends to itself bidirectionally.
        ctx_valid = (
            torch.arange(win, device=device)[None, :] < start_pos.clamp(max=win)[:, None]
        )  # [G, win]
        attn_mask = torch.cat(
            [
                ctx_valid[:, None, None, :].expand(G, 1, B, win),
                torch.ones(G, 1, B, B, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )

        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            win_l = kv_windows[:, li]  # [N, win, 2*kv_dim] view
            # Write the newest committed token's context row, then read the ring.
            kv0 = self._ctx_kv(layer, main_x, p0)  # [G, 2*kv_dim]
            win_l[slots, cols0] = kv0.to(win_l.dtype)
            ctx = win_l[slots]  # [G, win, 2*kv_dim]
            k_ctx = ctx[..., : self.kv_dim].view(G, win, self.num_kv_heads, self.head_dim)
            v_ctx = ctx[..., self.kv_dim :].view(G, win, self.num_kv_heads, self.head_dim)

            residual = h
            x = layer.input_layernorm(h)
            q = a.q_norm(a.q_proj(x).view(G, B, self.num_heads, self.head_dim))
            q = _apply_rope(q, cos_q, sin_q)
            k_blk = a.k_norm(a.k_proj(x).view(G, B, self.num_kv_heads, self.head_dim))
            k_blk = _apply_rope(k_blk, cos_q, sin_q)
            v_blk = a.v_proj(x).view(G, B, self.num_kv_heads, self.head_dim)

            k = torch.cat([k_ctx.to(h.dtype), k_blk], dim=1).transpose(1, 2)
            v = torch.cat([v_ctx.to(h.dtype), v_blk], dim=1).transpose(1, 2)
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k, v, attn_mask=attn_mask, scale=self.softmax_scale
            )
            o = o.transpose(1, 2).reshape(G, B, self.num_heads * self.head_dim)
            h = residual + a.o_proj(o)

            residual = h
            h = residual + layer.mlp(layer.post_attention_layernorm(h))

        h = self.norm(h)
        base_logits = self.lm_head(h)
        return dspark_propose(
            base_logits,
            bonus_token_ids=bonus_token_ids,
            block_hidden=h,
            markov_head=self.markov_head,
            confidence_head=self.confidence_head,
            block_size=B,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )

    def forward(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos,
        *,
        kv_windows: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        """Eager/single-shot convenience wrapper over :meth:`forward_batched`."""
        T = int(main_hidden.shape[0])
        device = main_hidden.device
        if not torch.is_tensor(start_pos):
            start_pos = torch.full((T,), int(start_pos), dtype=torch.long, device=device)
        if kv_windows is None:
            kv_windows = torch.zeros(
                (
                    T,
                    self.num_stages,
                    self._attn_params["window_size"],
                    self._attn_params["head_dim"],
                ),
                dtype=torch.bfloat16,
                device=device,
            )
        slots = torch.arange(T, device=device)
        return self.forward_batched(
            main_hidden, bonus_token_ids, start_pos, kv_windows=kv_windows, slots=slots, **kwargs
        )

    def run_moe_lockstep_noop(self, all_rank_num_tokens, device) -> None:
        """Dense drafter: no cross-rank MoE barrier to keep in lockstep."""
        return None

    # --------------------------------------------------------------- loading

    def load_weights(self, weights: Dict) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = {}
        for k, v in weights.items():
            # embed_tokens / lm_head are frozen copies of the target's; the
            # shared target modules are used instead (load_weights_from_target_model).
            if k.startswith(("embed_tokens.", "lm_head.")):
                continue
            t = v.to(device)
            t = t.float() if k.startswith("confidence_head.") else t.to(torch.bfloat16)
            sd[k] = t
        self.load_state_dict(sd, strict=True, assign=True)
        logger.info(
            f"[DSpark-Qwen3] loaded {len(sd)} draft params "
            f"({self.num_stages} layers, block_size={self.block_size}, "
            f"ctx_window={self._attn_params['window_size']})"
        )


class Qwen3DSparkForCausalLM(nn.Module):
    """One-engine draft wrapper for the Qwen3 DSpark drafter.

    Mirrors :class:`DSparkForCausalLM`: created by ``get_draft_model``,
    appended to the target's epilogue, and driven by ``DSparkWorker`` through
    the shared draft-model protocol. ``embed_tokens`` / ``lm_head`` are shared
    with the target model (the drafter checkpoint's copies are identical
    frozen snapshots and are skipped at load).
    """

    def __init__(self, draft_config, block_size: Optional[int] = None):
        super().__init__()
        self.dspark_model = Qwen3DSparkDraftModel(draft_config, block_size=block_size)
        # Generic handles expected by the loader / weight mappers.
        self.model = self.dspark_model
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        # Worker-facing interface.
        self.num_stages = self.dspark_model.num_stages
        self._attn_params = self.dspark_model._attn_params
        self.lm_head = None  # shared from the target
        self.logits_processor = None  # set by the caller after construction

    @property
    def block_size(self):
        return self.dspark_model.block_size

    @property
    def embed_tokens(self):
        return self.dspark_model.embed_tokens

    def forward(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        return self.dspark_model.forward(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def forward_batched(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        return self.dspark_model.forward_batched(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def run_moe_lockstep_noop(self, all_rank_num_tokens, device):
        return self.dspark_model.run_moe_lockstep_noop(all_rank_num_tokens, device)

    def write_context_windows(self, main_hidden, positions, stage_windows):
        return self.dspark_model.write_context_windows(main_hidden, positions, stage_windows)

    def write_context_windows_batched(self, main_hidden, positions, slots, mask, kv_windows):
        return self.dspark_model.write_context_windows_batched(
            main_hidden, positions, slots, mask, kv_windows
        )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load the flat-namespace Qwen3 DSpark drafter checkpoint.

        ``weight_mapper`` is accepted for loader-interface parity but unused —
        the checkpoint names map 1:1 onto the module tree.
        """
        self.dspark_model.load_weights(weights)

    def load_weights_from_target_model(self, target_model):
        """Share the target's embed_tokens / lm_head (identical frozen copies)."""
        if self.dspark_model.embed_tokens is None:
            self.dspark_model.embed_tokens = target_model.model.embed_tokens
        if self.lm_head is None:
            self.lm_head = target_model.lm_head
            self.dspark_model.lm_head = target_model.lm_head


__all__ = ["Qwen3DSparkDraftModel", "Qwen3DSparkForCausalLM"]
