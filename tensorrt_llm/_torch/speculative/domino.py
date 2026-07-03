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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .dflash import DFlashSpecMetadata, DFlashWorker

if TYPE_CHECKING:
    from ...llmapi.llm_args import DominoDecodingConfig


@dataclass
class DominoSpecMetadata(DFlashSpecMetadata):
    """Metadata for Domino speculative decoding.

    Identical to DFlashSpecMetadata in structure — Domino reuses DFlash's
    layer-wise hidden-state capture and per-slot context bookkeeping. The
    distinct type lets dispatch and instrumentation distinguish the two
    algorithms while sharing all behavior.
    """

    pass


class DominoWorker(DFlashWorker):
    """Worker for Domino speculative decoding.

    Domino extends DFlash with a lightweight causal correction "Domino head"
    applied during draft sampling: for the first ``pure_draft_prefix_len``
    suffix positions the base draft logits are used unchanged; for the
    remaining positions the model walks a small GRU over the embeddings of
    already-committed prefix tokens to produce a causal state ``s_i``, and a
    2-layer MLP turns ``[z_i, s_i]`` into a per-vocab bias added to the base
    logits before argmax.

    Reference (training-side counterpart): SpecForge ``OnlineDominoModel``.
    The cross-attention forward, K/V pool buffer, prefill capture, attention
    metadata management, and accept-path are all inherited unchanged from
    :class:`DFlashWorker`.
    """

    def __init__(
        self,
        spec_config: "DominoDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        # Per-shape caches populated lazily on first draft-sample call.
        self._graph_cache: dict = {}
        self._compiled_loop_cache: dict = {}
        self._compile_kwargs_cache: Optional[dict] = None
        self._prewarm_done = False
        logger.info(
            f"DominoWorker initialized (use_separate_draft_kv_cache={use_separate_draft_kv_cache})"
        )

    def _sample_gen_draft_tokens(
        self,
        hidden_states_out: torch.Tensor,
        draft_model: nn.Module,
        attn_metadata: AttentionMetadata,
        num_gens: int,
        K: int,
        inputs: dict,
    ) -> torch.Tensor:
        """Domino draft-token sampling with GRU-based causal correction.

        Runs base-logits projection once (depends on attn_metadata, stays
        outside the captured graph), then the K-step Domino loop through a
        per-shape CUDA Graph replay (or eager if an outer graph is capturing).
        """
        suffix_hidden = self._gather_mask_position_hidden(
            hidden_states_out=hidden_states_out,
            num_gens=num_gens,
            K=K,
        )
        base_logits = self._project_logits(
            suffix_hidden, draft_model, attn_metadata, num_gens, K
        )  # [N, K, vocab]

        d2t = getattr(draft_model.model, "d2t", None)
        prefix_len = min(max(0, int(self.spec_config.pure_draft_prefix_len or 0)), K)

        prefix_gru = getattr(draft_model, "prefix_gru", None)
        embed_proj = getattr(draft_model, "embed_proj", None)

        # Fall back to plain DFlash-style argmax when the draft model lacks
        # Domino modules (config mismatch).
        if prefix_gru is None or embed_proj is None:
            gen_draft_tokens = torch.argmax(base_logits, dim=-1, keepdim=False).long()
            if d2t is not None:
                gen_draft_tokens = d2t[gen_draft_tokens] + gen_draft_tokens
            return gen_draft_tokens.type(torch.int32)

        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        bonus_token_ids = inputs["bonus_token_ids"].long()
        z = suffix_hidden.reshape(num_gens, K, suffix_hidden.shape[-1])

        if self._compile_kwargs_cache is None:
            self._compile_kwargs_cache = self._build_compile_kwargs(draft_model)
        compile_kwargs = self._compile_kwargs_cache

        # If an outer CUDA graph is currently capturing, run eager so our
        # kernels stream into that capture. Nesting graph.replay() inside an
        # active capture is illegal.
        if torch.cuda.is_current_stream_capturing():
            return self._sample_eager(
                z=z,
                base_logits=base_logits,
                bonus_token_ids=bonus_token_ids,
                prefix_len=prefix_len,
                K=K,
                num_gens=num_gens,
                embed_tokens=embed_tokens,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                d2t=d2t,
                compile_kwargs=compile_kwargs,
            )

        # Lazy prewarm on first steady-state call — draft_model / embed
        # weights are only available now, not at __init__ time.
        if not self._prewarm_done:
            self._prewarm_graph_cache(
                K=K,
                prefix_len=prefix_len,
                z_dtype=z.dtype,
                base_logits_dtype=base_logits.dtype,
                hidden_size=z.shape[-1],
                vocab_size=base_logits.shape[-1],
                embed_tokens=embed_tokens,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                d2t=d2t,
                compile_kwargs=compile_kwargs,
            )
            self._prewarm_done = True

        return self._sample_graphed(
            z=z,
            base_logits=base_logits,
            bonus_token_ids=bonus_token_ids,
            prefix_len=prefix_len,
            K=K,
            num_gens=num_gens,
            embed_tokens=embed_tokens,
            prefix_gru=prefix_gru,
            embed_proj=embed_proj,
            d2t=d2t,
            compile_kwargs=compile_kwargs,
        )

    @staticmethod
    def _build_compile_kwargs(draft_model: nn.Module) -> Optional[dict]:
        """Collect raw weight tensors for the compiled-loop path.

        Returns None when the transposed GRU buffers are missing (the loop
        then uses the eager cuDNN GRU + nn.Sequential fallback).
        """
        w_ih_t = getattr(draft_model, "prefix_gru_w_ih_t", None)
        w_hh_t = getattr(draft_model, "prefix_gru_w_hh_t", None)
        proj = getattr(draft_model, "embed_proj", None)
        if w_ih_t is None or w_hh_t is None or proj is None:
            return None
        return {
            "w_ih_t": w_ih_t,
            "w_hh_t": w_hh_t,
            "ep0_w": proj[0].weight,
            "ep2_w": proj[2].weight,
            "gru_hidden_dim": int(w_hh_t.shape[0]),
        }

    def _prewarm_graph_cache(
        self,
        *,
        K: int,
        prefix_len: int,
        z_dtype: torch.dtype,
        base_logits_dtype: torch.dtype,
        hidden_size: int,
        vocab_size: int,
        embed_tokens: nn.Module,
        prefix_gru: nn.GRU,
        embed_proj: nn.Module,
        d2t: Optional[torch.Tensor],
        compile_kwargs: Optional[dict] = None,
    ) -> None:
        """Pre-capture the Domino loop graph for every batch size 1..max.

        Each per-shape capture costs ~30-65 ms; batch size drops as requests
        complete (e.g. 8→4→2→1), so lazy capture would hit the hot path.
        Prewarm concentrates that one-time cost at startup.
        """
        max_batch = self._batch_to_slot.shape[0] if self._batch_to_slot is not None else 0
        if max_batch <= 0:
            return
        prewarmed = []
        for n in range(1, max_batch + 1):
            key = (n, K, prefix_len, d2t is not None)
            if key in self._graph_cache:
                continue
            try:
                self._get_or_capture_graph(
                    num_gens=n,
                    K=K,
                    prefix_len=prefix_len,
                    z_shape=(n, K, hidden_size),
                    z_dtype=z_dtype,
                    base_logits_shape=(n, K, vocab_size),
                    base_logits_dtype=base_logits_dtype,
                    embed_tokens=embed_tokens,
                    prefix_gru=prefix_gru,
                    embed_proj=embed_proj,
                    d2t=d2t,
                    compile_kwargs=compile_kwargs,
                )
                prewarmed.append(n)
            except Exception as exc:
                logger.warning(
                    "DominoWorker: prewarm capture failed for num_gens=%d (%s); "
                    "will fall back to lazy capture.",
                    n,
                    exc,
                )
                self._graph_cache[key] = "eager"
        if prewarmed:
            logger.info(
                f"DominoWorker: prewarm captured graphs for num_gens={prewarmed} "
                f"K={K} prefix_len={prefix_len}"
            )

    @staticmethod
    def _loop_body(
        z: torch.Tensor,  # [N, K, H]
        base_logits: torch.Tensor,  # [N, K, V]
        out_tokens: torch.Tensor,  # [N, K], int64 (writes positions [prefix_len:K])
        prefix_embeds: torch.Tensor,  # [N, 1+pref, H]
        embed_weight: torch.Tensor,  # [V, H]
        w_ih_t: torch.Tensor,  # [H, 3G]  (transposed for hand GRU)
        w_hh_t: torch.Tensor,  # [G, 3G]
        ep0_w: torch.Tensor,  # [EMB, H+G]  (embed_proj[0].weight, NOT transposed)
        ep2_w: torch.Tensor,  # [V, EMB]    (embed_proj[2].weight, NOT transposed)
        d2t: Optional[torch.Tensor],
        prefix_len: int,
        K: int,
        num_gens: int,
        gru_hidden_dim: int,
    ) -> None:
        """Inductor-friendly K-step loop body.

        Same math as the cuDNN GRU + nn.Sequential (cat-then-Linear) path in
        `_run_domino_loop`, but expressed as raw tensor ops so torch.compile
        can lower the whole loop into Triton kernels. The hand-rolled GRU
        (over cuDNN _VF.gru) lets Inductor fuse gate-split/activation and
        accumulate gates in fp32 — token-exact against cuDNN in bf16.
        cat-then-Linear (over split-W1) preserves cuDNN's reduction order;
        split-W1 accumulates z/h projections separately and drifts ~12%.
        """
        G_dim = gru_hidden_dim

        h = torch.zeros(num_gens, G_dim, device=z.device, dtype=z.dtype)
        for k in range(prefix_embeds.shape[1]):
            x = prefix_embeds[:, k, :]
            gi = x @ w_ih_t
            gh = h @ w_hh_t
            r = torch.sigmoid(gi[:, :G_dim] + gh[:, :G_dim])
            zg = torch.sigmoid(gi[:, G_dim : 2 * G_dim] + gh[:, G_dim : 2 * G_dim])
            n = torch.tanh(gi[:, 2 * G_dim :] + r * gh[:, 2 * G_dim :])
            h = (1 - zg) * n + zg * h

        ep0_t = ep0_w.t()
        ep2_t = ep2_w.t()
        for i in range(prefix_len, K):
            cat_i = torch.cat([z[:, i, :], h], dim=-1)
            inner = torch.nn.functional.silu(cat_i @ ep0_t)
            logits_i = base_logits[:, i, :] + inner @ ep2_t
            tok_i = torch.argmax(logits_i, dim=-1).long()
            if d2t is not None:
                tok_i = d2t[tok_i] + tok_i
            out_tokens[:, i] = tok_i
            if i + 1 < K:
                x = embed_weight[tok_i]
                gi = x @ w_ih_t
                gh = h @ w_hh_t
                r = torch.sigmoid(gi[:, :G_dim] + gh[:, :G_dim])
                zg = torch.sigmoid(gi[:, G_dim : 2 * G_dim] + gh[:, G_dim : 2 * G_dim])
                n = torch.tanh(gi[:, 2 * G_dim :] + r * gh[:, 2 * G_dim :])
                h = (1 - zg) * n + zg * h

    def _compiled_loop(
        self,
        *,
        num_gens: int,
        K: int,
        prefix_len: int,
        has_d2t: bool,
    ):
        """Get-or-compile the K-step loop body for a given shape.

        Returns None under TP>1 (compiled body indexes embed_tokens.weight
        directly, unsafe against a sharded embedding); the caller falls back
        to the eager body.
        """
        if self.mapping.tp_size > 1:
            return None
        key = (num_gens, K, prefix_len, has_d2t)
        cached = self._compiled_loop_cache.get(key)
        if cached is not None:
            return cached
        # mode="default" (not "reduce-overhead"): "reduce-overhead" wraps the
        # compiled function in Inductor's own cudagraph, which collides with
        # our outer manual capture in _get_or_capture_graph. Default just
        # emits Triton kernels that stream into the active capture cleanly.
        compiled = torch.compile(
            DominoWorker._loop_body,
            mode="default",
            dynamic=False,
            fullgraph=True,
        )
        self._compiled_loop_cache[key] = compiled
        return compiled

    @staticmethod
    def _run_domino_loop(
        z: torch.Tensor,  # [N, K, H]
        base_logits: torch.Tensor,  # [N, K, V]
        bonus_token_ids: torch.Tensor,  # [N], int64
        out_tokens: torch.Tensor,  # [N, K], int64 (written in place)
        prefix_len: int,
        K: int,
        num_gens: int,
        embed_tokens: nn.Module,
        prefix_gru: nn.GRU,
        embed_proj: nn.Module,
        d2t: Optional[torch.Tensor],
        compiled_loop=None,
        compile_kwargs: Optional[dict] = None,
    ) -> None:
        """Run the full Domino sampling loop, writing into out_tokens.

        Pure tensor-graph (no Python control on tensor values), safe to capture
        with torch.cuda.graph; shapes are baked in via Python ints. Uses the
        Inductor-compiled body when ``compiled_loop`` + ``compile_kwargs`` are
        provided, else the eager cuDNN GRU + nn.Sequential path.
        """
        if prefix_len > 0:
            prefix_argmax = torch.argmax(base_logits[:, :prefix_len, :], dim=-1).long()
            if d2t is not None:
                prefix_argmax = d2t[prefix_argmax] + prefix_argmax
            out_tokens[:, :prefix_len] = prefix_argmax

        # GRU prime: input = [bonus, prefix_drafts...], length = 1 + prefix_len.
        prefix_with_bonus = torch.cat(
            [bonus_token_ids.unsqueeze(1), out_tokens[:, :prefix_len]],
            dim=1,
        )
        prefix_embeds = embed_tokens(prefix_with_bonus)  # [N, 1+pref, H]

        if compiled_loop is not None and compile_kwargs:
            compiled_loop(
                z,
                base_logits,
                out_tokens,
                prefix_embeds,
                embed_tokens.weight,
                compile_kwargs["w_ih_t"],
                compile_kwargs["w_hh_t"],
                compile_kwargs["ep0_w"],
                compile_kwargs["ep2_w"],
                d2t,
                prefix_len,
                K,
                num_gens,
                compile_kwargs["gru_hidden_dim"],
            )
            return

        _, gru_hidden = prefix_gru(prefix_embeds)  # [1, N, gru_hidden]

        for i in range(prefix_len, K):
            z_i = z[:, i : i + 1, :]
            s_i = gru_hidden.transpose(0, 1)
            bias_i = embed_proj(torch.cat([z_i, s_i], dim=-1))
            logits_i = base_logits[:, i : i + 1, :] + bias_i
            tok_i = torch.argmax(logits_i.squeeze(1), dim=-1).long()
            if d2t is not None:
                tok_i = d2t[tok_i] + tok_i
            out_tokens[:, i] = tok_i
            if i + 1 < K:
                new_embed = embed_tokens(tok_i.unsqueeze(1))
                _, gru_hidden = prefix_gru(new_embed, gru_hidden)

    def _sample_eager(
        self,
        *,
        z: torch.Tensor,
        base_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        prefix_len: int,
        K: int,
        num_gens: int,
        embed_tokens: nn.Module,
        prefix_gru: nn.GRU,
        embed_proj: nn.Module,
        d2t: Optional[torch.Tensor],
        compile_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        out_tokens = torch.empty((num_gens, K), dtype=torch.long, device=base_logits.device)
        compiled_loop = (
            self._compiled_loop(
                num_gens=num_gens, K=K, prefix_len=prefix_len, has_d2t=d2t is not None
            )
            if compile_kwargs
            else None
        )
        self._run_domino_loop(
            z=z,
            base_logits=base_logits,
            bonus_token_ids=bonus_token_ids,
            out_tokens=out_tokens,
            prefix_len=prefix_len,
            K=K,
            num_gens=num_gens,
            embed_tokens=embed_tokens,
            prefix_gru=prefix_gru,
            embed_proj=embed_proj,
            d2t=d2t,
            compiled_loop=compiled_loop,
            compile_kwargs=compile_kwargs,
        )
        return out_tokens.type(torch.int32)

    def _get_or_capture_graph(
        self,
        *,
        num_gens: int,
        K: int,
        prefix_len: int,
        z_shape: Tuple[int, int, int],
        z_dtype: torch.dtype,
        base_logits_shape: Tuple[int, int, int],
        base_logits_dtype: torch.dtype,
        embed_tokens: nn.Module,
        prefix_gru: nn.GRU,
        embed_proj: nn.Module,
        d2t: Optional[torch.Tensor],
        compile_kwargs: Optional[dict] = None,
    ):
        """Return a cached graph entry, capturing it on first miss.

        The d2t presence (not value) participates in the cache key — its
        absence eliminates an indexing kernel from the captured loop.
        """
        key = (num_gens, K, prefix_len, d2t is not None)
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached

        device = torch.device("cuda")
        z_buf = torch.empty(z_shape, dtype=z_dtype, device=device)
        base_logits_buf = torch.empty(base_logits_shape, dtype=base_logits_dtype, device=device)
        # Zero-init: bonus_buf flows into embed_tokens(...) during warmup; an
        # uninitialized index above vocab_size would cause a CUDA-side OOB
        # read that can corrupt the side-stream allocator pool.
        bonus_buf = torch.zeros((num_gens,), dtype=torch.long, device=device)
        tokens_buf = torch.zeros((num_gens, K), dtype=torch.long, device=device)

        # Warm-up on a side stream before capture so cuDNN picks an algorithm,
        # Inductor specializes its compiled kernels, and any lazy allocators
        # settle before capture.
        compiled_loop = (
            self._compiled_loop(
                num_gens=num_gens, K=K, prefix_len=prefix_len, has_d2t=d2t is not None
            )
            if compile_kwargs
            else None
        )
        run_kwargs = dict(
            z=z_buf,
            base_logits=base_logits_buf,
            bonus_token_ids=bonus_buf,
            out_tokens=tokens_buf,
            prefix_len=prefix_len,
            K=K,
            num_gens=num_gens,
            embed_tokens=embed_tokens,
            prefix_gru=prefix_gru,
            embed_proj=embed_proj,
            d2t=d2t,
            compiled_loop=compiled_loop,
            compile_kwargs=compile_kwargs,
        )
        # 4 warmups (vs 2) when compiled body is in use so Inductor finishes
        # specialization + autotune before capture.
        n_warmup = 4 if compiled_loop is not None else 2
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(n_warmup):
                self._run_domino_loop(**run_kwargs)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_domino_loop(**run_kwargs)

        entry = {
            "graph": graph,
            "z_buf": z_buf,
            "base_logits_buf": base_logits_buf,
            "bonus_buf": bonus_buf,
            "tokens_buf": tokens_buf,
        }
        self._graph_cache[key] = entry
        logger.info(
            f"DominoWorker: captured CUDA graph for num_gens={num_gens} "
            f"K={K} prefix_len={prefix_len} d2t={d2t is not None}"
        )
        return entry

    def _sample_graphed(
        self,
        *,
        z: torch.Tensor,
        base_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        prefix_len: int,
        K: int,
        num_gens: int,
        embed_tokens: nn.Module,
        prefix_gru: nn.GRU,
        embed_proj: nn.Module,
        d2t: Optional[torch.Tensor],
        compile_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        # base_logits may be a non-contiguous reshape view depending on lm_head
        # output layout; copy_ into the static buffer requires contiguous.
        z_c = z.contiguous()
        base_logits_c = base_logits.contiguous()
        bonus_c = bonus_token_ids.contiguous()

        entry = None
        try:
            entry = self._get_or_capture_graph(
                num_gens=num_gens,
                K=K,
                prefix_len=prefix_len,
                z_shape=tuple(z_c.shape),
                z_dtype=z_c.dtype,
                base_logits_shape=tuple(base_logits_c.shape),
                base_logits_dtype=base_logits_c.dtype,
                embed_tokens=embed_tokens,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                d2t=d2t,
                compile_kwargs=compile_kwargs,
            )
        except Exception as exc:
            # cuDNN GRU sometimes refuses capture in older builds — permanently
            # fall back to eager for this shape.
            logger.warning(
                "DominoWorker: CUDA graph capture failed (%s); "
                "falling back to eager Domino loop for this shape.",
                exc,
            )
            self._graph_cache[(num_gens, K, prefix_len, d2t is not None)] = "eager"

        if entry is None or entry == "eager":
            return self._sample_eager(
                z=z_c,
                base_logits=base_logits_c,
                bonus_token_ids=bonus_c,
                prefix_len=prefix_len,
                K=K,
                num_gens=num_gens,
                embed_tokens=embed_tokens,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                d2t=d2t,
                compile_kwargs=compile_kwargs,
            )

        entry["z_buf"].copy_(z_c)
        entry["base_logits_buf"].copy_(base_logits_c)
        entry["bonus_buf"].copy_(bonus_c)
        entry["graph"].replay()
        # tokens_buf is int64 and mutated in place by replay; `.to(int32)` on a
        # different dtype always allocates, giving the caller its own storage.
        return entry["tokens_buf"].to(torch.int32)
