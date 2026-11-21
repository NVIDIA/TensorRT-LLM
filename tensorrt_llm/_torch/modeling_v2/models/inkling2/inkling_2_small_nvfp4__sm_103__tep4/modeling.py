# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small, NVFP4/MXFP8 experts / sm_103 / attention TP4 + MoE EP4: token ids to logits.

One engine step through the 48 decoder layers of the text model. Per layer, with `P` the
partial residual block (bf16 between layers) and `blocks` the normalized completed blocks:

    attention inputs   mix(blocks, P) through the input norm(s): one input on a sliding layer,
                       q, k and v inputs on a global layer (layer 0 reads the embedding block)
    projections        q, r (16 per head), gate, sink, k and v; a global layer's k and v also
                       give their convolution's per-token taps
    k, v               short convolution (dynamic on a global layer), then the prologue: q and r
                       scaled and clipped (times tau on a global layer), k clipped, v normalized
    attention          paged, relative bias from r, sink and output gate (window 1024 on a
                       sliding layer); output projection; all-reduce over TP; short convolution
    residual           a block boundary (layer % 6 == 0) closes P into blocks and restarts the
                       partial block from the attention output; otherwise P' = P + a
    MoE                mix(blocks, P') through the MLP norm; fp32 sigmoid router, top-8 of 384;
                       this rank's 96 experts (NVFP4 or MXFP8); all-reduce over EP; short conv
    output             P'' = P' + m, rounded to bf16

then the final mix over the 8 blocks and the last partial block through the final norm. The
shell's vocab-parallel lm_head GEMM writes bf16 logits, which the shell upcasts to fp32 (as the
Hugging Face implementation does); they are soft-capped in fp32 (30 tanh(x / 30)) and cut to
the tokenizer's 200058 entries (the padded rows are set to -inf).

The KV cache is the engine's `KVCacheManagerV2` (BF16, 2 KV heads of 128 per rank, a 1024-token
window on the 40 sliding layers) extended with each request's short-convolution windows
(`Inkling2KVCacheManager`, `cache.py`). The forward runs eagerly: no CUDA graphs, no chunked
prefill, no KV cache reuse, no speculative decoding; routing admits only that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.comm.allreduce import allreduce, allreduce_group_args
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.logger import logger

from ..config import ExpertFormat, InklingDims, read_hf_quant_config
from ..kernels.paged_attention import block_tokens, tile_plan
from ..ops.attention.inkling2_attention_prologue import inkling2_attention_prologue, log_scaling
from ..ops.attention.inkling2_paged_attention import SM_SCALE, inkling2_paged_attention
from ..ops.conv.inkling2_short_conv import inkling2_dynamic_short_conv, inkling2_short_conv
from ..ops.moe.inkling2_mxfp8_experts import inkling2_mxfp8_experts
from ..ops.moe.inkling2_nvfp4_experts import fc2_input_scale, inkling2_nvfp4_experts
from ..ops.moe.inkling2_router import inkling2_router
from ..ops.residual.inkling2_residual_mix import (
    inkling2_residual_mix,
    normalize_block,
    rmsnorm_to_bf16,
)
from . import weights as _weights
from .cache import Inkling2KVCacheManager

_NO_FUSION = int(AllReduceFusionOp.NONE)


def text_config_of(pretrained_config: PretrainedConfig) -> PretrainedConfig:
    return getattr(pretrained_config, "text_config", None) or pretrained_config


@dataclass
class _Step:
    """Per-step views of the attention metadata, shared by every layer."""

    num_tokens: int
    query_start: torch.Tensor  # [R + 1] int32
    kv_len: torch.Tensor  # [R] int32, cached + new
    token_request: torch.Tensor  # [T] int32
    has_state: torch.Tensor  # [R] int32: the request has convolution history
    conv_slots: torch.Tensor  # [R] int32
    tile_request: torch.Tensor
    tile_start: torch.Tensor
    block_t: int
    page_tables: dict  # layer -> [R, W] int32
    tau: torch.Tensor  # [T] fp32


class Inkling2SmallCore(DecoderModel):
    """The 48-layer text decoder on one rank of attention TP4 + MoE EP4."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        text = text_config_of(model_config.pretrained_config)
        checkpoint_dir = getattr(model_config.pretrained_config, "_name_or_path", None)
        hf_quant_config = read_hf_quant_config(checkpoint_dir)
        if hf_quant_config is None:
            # Without it every layer would read as BF16 experts, which this target does not serve.
            raise ValueError(
                f"Inkling2: found no hf_quant_config.json in {checkpoint_dir!r}, the directory the "
                "config was loaded from (`_name_or_path`); this target reads each layer's expert "
                "format from that file, so load the config from the checkpoint directory"
            )
        self.dims = InklingDims.from_config(text, hf_quant_config)
        self.shares = _weights.shares_of(self.dims, model_config.mapping)
        self.eps = self.dims.rms_norm_eps
        self.w = _weights.declare(self.shares)
        self._layers: Optional[list] = None
        self._allreduce: Optional[dict] = None
        self._fc2_scale: Optional[torch.Tensor] = None
        self._shared_tables: Optional[bool] = None

    def post_load_weights(self) -> None:
        """Per-layer weight views and the TP communicator, once the weights are real."""
        self._layers = [_weights.layer_view(self.w, layer) for layer in range(self.dims.num_layers)]
        self._allreduce = allreduce_group_args(self.model_config.mapping)
        self._fc2_scale = fc2_input_scale(self.w["embed"].device)

    # ------------------------------------------------------------------------------- step plan

    def _plan(self, md: AttentionMetadata, position_ids: torch.Tensor, num_tokens: int) -> _Step:
        manager: Inkling2KVCacheManager = md.kv_cache_manager
        num_seqs = md.num_seqs
        lengths = [int(n) for n in md.seq_lens[:num_seqs].tolist()]
        cached = [int(c) for c in md.kv_cache_params.num_cached_tokens_per_seq[:num_seqs]]
        request_ids = list(md.request_ids[:num_seqs])
        assert sum(lengths) == num_tokens, (lengths, num_tokens)
        bounds = [0]
        for n in lengths:
            bounds.append(bounds[-1] + n)
        device = position_ids.device

        def i32(values):
            return torch.tensor(values, dtype=torch.int32).to(device, non_blocking=True)

        block_t = block_tokens(max(lengths))
        tile_request, tile_start = tile_plan(lengths, block_t, device)
        kv_lens = [c + n for c, n in zip(cached, lengths)]
        blocks = [-(-length // manager.tokens_per_block) for length in kv_lens]
        page_tables = self._page_tables(manager, request_ids, blocks, cached, device)
        return _Step(
            num_tokens=num_tokens,
            query_start=i32(bounds),
            kv_len=i32(kv_lens),
            token_request=i32([r for r, n in enumerate(lengths) for _ in range(n)]),
            has_state=i32([int(c > 0) for c in cached]),
            conv_slots=manager.conv_slots(request_ids),
            tile_request=tile_request,
            tile_start=tile_start,
            block_t=block_t,
            page_tables=page_tables,
            tau=log_scaling(
                position_ids.reshape(-1), self.dims.log_scaling_alpha, self.dims.log_scaling_n_floor
            ),
        )

    def _page_tables(self, manager, request_ids, blocks, cached, device) -> dict:
        """Each layer's page table `[R, W]` int32 (-1 past a request's pages).

        A layer's indices address its own `get_buffers` view. The manager gives the layers of one
        window the same indices; the first step checks that once, after which one table per window
        is built and shared. Every table is checked to have a live page for each block a request
        reads this step (see `_check_live_pages`).
        """

        def build(layer):
            rows = manager.get_batch_cache_indices(
                request_ids, layer_idx=layer, num_blocks_per_seq=blocks
            )
            # As wide as the longest read, so a row the manager returns short shows as -1.
            table = torch.full((len(rows), max(blocks)), -1, dtype=torch.int32)
            for i, r in enumerate(rows):
                table[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
            return table

        layers = range(self.dims.num_layers)
        window = {layer: self.dims.is_global(layer) for layer in layers}
        if self._shared_tables is None:
            tables = {layer: build(layer) for layer in layers}
            first = {}
            for layer in layers:
                first.setdefault(window[layer], layer)
            self._shared_tables = all(
                torch.equal(tables[layer], tables[first[window[layer]]]) for layer in layers
            )
            if not self._shared_tables:
                logger.warning(
                    "Inkling2: layers of one window have different page tables; building one per layer"
                )
        elif self._shared_tables:
            per_window = {}
            for layer in layers:
                if window[layer] not in per_window:
                    per_window[window[layer]] = build(layer)
            tables = {layer: per_window[window[layer]] for layer in layers}
        else:
            tables = {layer: build(layer) for layer in layers}
        # One check and one upload per distinct table: the layers of a window share theirs.
        tokens_per_block = manager.tokens_per_block
        read_end = torch.tensor(blocks)
        sliding_start = torch.tensor(
            [max(0, c - self.dims.sliding_window + 1) // tokens_per_block for c in cached]
        )
        uploaded = {}
        for layer, t in tables.items():
            if id(t) not in uploaded:
                start = torch.zeros_like(read_end) if window[layer] else sliding_start
                self._check_live_pages(t, start, read_end, layer, request_ids)
                uploaded[id(t)] = t.to(device, non_blocking=True)
        return {layer: uploaded[id(t)] for layer, t in tables.items()}

    @staticmethod
    def _check_live_pages(table, start, end, layer, request_ids) -> None:
        """Fail if a block a request reads this step has no page (-1) in `table`.

        Request `r` reads blocks `[start[r], end[r])`: from the window of its first new token on a
        sliding layer (from block 0 on a global one) to its last token. `KVCacheManagerV2` keeps
        a sliding layer's pages from `history + 1 - window` on, which covers that range, so a hole
        there is a broken retention contract, reported here instead of read by the kernels.
        """
        blocks = torch.arange(table.shape[1])
        read = (blocks[None, :] >= start[:, None]) & (blocks[None, :] < end[:, None])
        holes = (table < 0) & read
        if bool(holes.any()):
            row, block = (int(i) for i in holes.nonzero()[0])
            raise RuntimeError(
                f"Inkling2: layer {layer}: request {request_ids[row]} reads KV cache block "
                f"{block}, which has no page; the KV cache manager did not keep the window "
                "the attention kernels read"
            )

    # ------------------------------------------------------------------------------- forward

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_config.mapping.tp_size == 1:
            return x
        return allreduce(
            x,
            None,
            None,
            None,
            None,
            op=_NO_FUSION,
            eps=self.eps,
            trigger_completion_at_end=True,
            **self._allreduce,
        )

    def _conv(self, x, taps, state, step: _Step):
        return inkling2_short_conv(
            x, taps, state, step.conv_slots, step.query_start, step.has_state, step.token_request
        )

    def _attention(
        self, layer: int, inputs: torch.Tensor, step: _Step, md: AttentionMetadata
    ) -> torch.Tensor:
        d, s, w = self.dims, self.shares, self._layers[layer]
        manager: Inkling2KVCacheManager = md.kv_cache_manager
        state = manager.conv_state(layer)
        t = step.num_tokens
        if d.is_global(layer):
            yq = cublas_mm(inputs[0], w.w_q.t())
            q, r, gate, sink = yq.split([s.q_width, s.r_width, s.q_width, s.heads], dim=1)
            yk = cublas_mm(inputs[1], w.w_k.t())
            yv = cublas_mm(inputs[2], w.w_v.t())
            k_lin, k_taps = yk.split([s.kv_width, s.kv_heads * 4], dim=1)
            v_lin, v_taps = yv.split([s.kv_width, s.kv_heads * 4], dim=1)
            conv_args = (
                step.conv_slots,
                step.query_start,
                step.has_state,
                step.token_request,
                d.head_dim,
            )
            k = inkling2_dynamic_short_conv(
                k_lin.contiguous(), w.k_conv, k_taps.contiguous(), state["k"], *conv_args
            )
            v = inkling2_dynamic_short_conv(
                v_lin.contiguous(), w.v_conv, v_taps.contiguous(), state["v"], *conv_args
            )
            tau, window = step.tau, 0
        else:
            y = cublas_mm(inputs[0], w.w_in.t())
            q, r, gate, sink, k_lin, v_lin = y.split(
                [s.q_width, s.r_width, s.q_width, s.heads, s.kv_width, s.kv_width], dim=1
            )
            k = self._conv(k_lin.contiguous(), w.k_conv, state["k"], step)
            v = self._conv(v_lin.contiguous(), w.v_conv, state["v"], step)
            tau, window = None, d.sliding_window
        q, k, v, r = inkling2_attention_prologue(
            q.reshape(t, s.heads, d.head_dim),
            k.view(t, s.kv_heads, d.head_dim),
            v.view(t, s.kv_heads, d.head_dim),
            r.reshape(t, s.heads, d.d_rel),
            w.q_scale,
            w.r_scale,
            tau,
            d.qk_clip_threshold,
            d.rel_clip_threshold,
            d.rms_norm_eps,
        )
        # The bias profiles mix 16 values per head; TF32 would round them, so a true fp32 product.
        with true_fp32_matmul():
            rel_logits = torch.matmul(r.float(), w.rel_proj.float())  # [T, heads, rel_extent]
        o = inkling2_paged_attention(
            q,
            k,
            v,
            manager.get_buffers(layer, kv_layout="HND"),
            step.page_tables[layer],
            step.query_start,
            step.kv_len,
            step.token_request,
            step.tile_request,
            step.tile_start,
            step.block_t,
            rel_logits,
            sink.contiguous(),
            gate.reshape(t, s.heads, d.head_dim),
            window,
            SM_SCALE,
        )
        a = self._all_reduce(cublas_mm(o.view(t, s.q_width), w.wo.t()))
        return self._conv(a, w.attn_conv, state["attn"], step)

    def _moe(self, layer: int, x: torch.Tensor, step: _Step, md: AttentionMetadata) -> torch.Tensor:
        d, s, w = self.dims, self.shares, self._layers[layer]
        ids, weights = inkling2_router(x, w.router, w.router_bias, d.top_k, d.route_scale)
        if d.expert_formats[layer] is ExpertFormat.NVFP4:
            m = inkling2_nvfp4_experts(
                x, ids, weights, w.w13, w.w13_sf, w.w13_alpha, w.w2, w.w2_sf, w.w2_alpha, self._fc2_scale,
                d.num_experts, s.expert_offset, d.gate_softcap, d.up_softcap,
            )  # fmt: skip
        else:
            m = inkling2_mxfp8_experts(
                x,
                ids,
                weights,
                w.w13,
                w.w13_s,
                w.w2,
                w.w2_s,
                s.expert_offset,
                d.gate_softcap,
                d.up_softcap,
            )
        m = self._all_reduce(m)
        return self._conv(m, w.mlp_conv, md.kv_cache_manager.conv_state(layer)["mlp"], step)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert inputs_embeds is None, "the text-only target embeds token ids itself"
        d, w = self.dims, self.w
        ids = input_ids.reshape(-1)
        num_tokens = ids.shape[0]
        step = self._plan(attn_metadata, position_ids, num_tokens)

        partial = rmsnorm_to_bf16(
            torch.nn.functional.embedding(ids, w.embed), w.embed_norm, self.eps
        )
        blocks = torch.empty(
            d.num_block_states, num_tokens, d.hidden_size, dtype=torch.float32, device=ids.device
        )
        num_blocks = 0
        for layer in range(d.num_layers):
            lw = self._layers[layer]
            norms = (lw.q_norm, lw.k_norm, lw.v_norm) if d.is_global(layer) else (lw.attn_norm,)
            if layer == 0:
                inputs = [rmsnorm_to_bf16(partial, n, self.eps) for n in norms]
            else:
                inputs = inkling2_residual_mix(
                    blocks[:num_blocks], partial, lw.attn_query, norms, self.eps
                )
            a = self._attention(layer, inputs, step, attn_metadata)
            if d.is_block_boundary(layer):
                blocks[num_blocks] = normalize_block(partial)
                num_blocks += 1
                residual = a.float()
            else:
                residual = partial.float() + a.float()
            x = inkling2_residual_mix(
                blocks[:num_blocks], residual, lw.mlp_query, (lw.mlp_norm,), self.eps
            )[0]
            m = self._moe(layer, x, step, attn_metadata)
            partial = (residual + m.float()).to(torch.bfloat16)
        assert num_blocks == d.num_block_states
        return inkling2_residual_mix(blocks, partial, w.final_query, (w.final_norm,), self.eps)[0]


class ModelingV2Inkling2SmallNvfp4Sm103Tep4(
    DecoderModelForCausalLM[Inkling2SmallCore, PretrainedConfig]
):
    """Inkling-2-Small on GB300 with attention TP4 and MoE EP4: the engine-facing shell."""

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        text = text_config_of(cfg)
        # lm_head is built in the config's dtype, which `ModelConfig.from_pretrained` takes from
        # the text config when config.json declares none at the top level.
        if getattr(cfg, "torch_dtype", None) is None:
            raise ValueError(
                "Inkling2: the config declares no dtype (`torch_dtype`, or `dtype` in its text "
                "config) to build lm_head in"
            )
        super().__init__(
            Inkling2SmallCore(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        dims = self.model.dims
        self.logit_softcap = dims.logit_softcap
        self.unpadded_vocab_size = dims.unpadded_vocab_size

    #: The engine builds the KV cache manager with the short-convolution windows.
    runtime_hooks = RuntimeHooks(kv_cache_manager_cls=Inkling2KVCacheManager)

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights, self.model.shares)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()

    def forward(self, attn_metadata: AttentionMetadata, *args, **kwargs) -> torch.Tensor:
        logits = super().forward(attn_metadata, *args, **kwargs)
        logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        if logits.shape[-1] > self.unpadded_vocab_size:
            logits[..., self.unpadded_vocab_size :] = float("-inf")
        return logits


__all__ = ["Inkling2SmallCore", "ModelingV2Inkling2SmallNvfp4Sm103Tep4"]
