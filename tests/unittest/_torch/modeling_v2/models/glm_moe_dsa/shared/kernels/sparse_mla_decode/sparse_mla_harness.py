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
"""Native DSA generation harness and FP64 oracles for the sparse_mla_decode tests.

The harness drives the ``DSATrtllmAttention`` generation path exactly as
``forward_absorption_generation`` does: ``mla_rope_generation`` (RoPE, FP8 query,
latent append, device scales) followed by ``TrtllmAttention.forward``. Each layer and its
indexer are adopted as the target's ``GlmDsaBackend`` / ``GlmIndexer`` and every generation
forward is a decode step, so the predictor runs the target's (with no fast path selected, the
built-in one). The
indexer is either replaced by a fixed local top-k (``forward``) or runs
unmodified on given indexer projections (``forward_native_indexer``). The FMHA
library list of the layer decides whether the native library or sparse_mla_decode computes
the attention.
"""

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from unittest.mock import Mock

import torch

import tensorrt_llm
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
    GlmDsaBackend,
    GlmIndexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E501
    SparseMlaDecodeFmha,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSABackendForwardArgs, DSACacheManager
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_HEADS = 8
KV_LORA_RANK = 512
ROPE_DIM = 64
NOPE_DIM = 192
V_HEAD_DIM = 256
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
Q_LORA_RANK = 2048
HIDDEN_SIZE = 6144
TOP_K = 2048
TOKENS_PER_BLOCK = 64
E4M3_MAX = 448.0


@dataclass
class _GlmRopeConfig:
    """Rotary settings of GLM-5.2 (``rope_theta`` 8e6, no scaling)."""

    max_position_embeddings: int
    hidden_size: int = 6144
    num_attention_heads: int = 64
    qk_rope_head_dim: int = ROPE_DIM
    rope_theta: float = 8.0e6
    rope_scaling: Optional[dict] = None
    model_type: str = "deepseek_v32"


@dataclass
class StepBuffers:
    """Per-step tensors produced by the native ``mla_rope_generation``."""

    fused_q: torch.Tensor
    q_pe: torch.Tensor
    latent: torch.Tensor
    cu_q_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    scheduler_counter: torch.Tensor
    bmm1_scale: torch.Tensor
    bmm2_scale: torch.Tensor
    quant_q: torch.Tensor


class DsaGenHarness:
    """One generation step of native DSA layers over an FP8 latent pool.

    Args:
        ctx_lens: Cached tokens per sequence before this step.
        seq_len_q: Query rows per sequence (1 + draft tokens).
        num_layers: Attention layers sharing the pool.
        index_topk: DSA top-k of the sparse config.
        kv_dtype: KV cache dtype (FP8 for the sparse_mla_decode domain).
        seed: Seed of the random pool contents.
        latent_std: Standard deviation of the random latent rows.
        fill_random: Fill the whole pool with random rows (instead of running a context step).
        defer_generation: Leave room for ``run_context`` before ``start_generation``.
        sparse_overrides: Extra ``DeepSeekSparseAttentionConfig`` fields (indexer kernels).
        mtp_draft_len: Draft length announced to the generation metadata, as the
            speculative-decoding runner does (``update_spec_dec_param``); 0 leaves it unset.
        ragged_verify_lens: Query rows per sequence of a ragged-verification step (one
            entry per sequence; replaces ``seq_len_q``). The generation metadata then
            enables ragged verification, and ``TrtllmAttention`` presents one row per
            query token to the RoPE and FMHA calls.
    """

    def __init__(
        self,
        ctx_lens: Sequence[int],
        seq_len_q: int,
        num_layers: int = 1,
        index_topk: int = TOP_K,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        seed: int = 0,
        latent_std: float = 1.0,
        fill_random: bool = True,
        defer_generation: bool = False,
        sparse_overrides: Optional[dict] = None,
        mtp_draft_len: int = 0,
        ragged_verify_lens: Optional[Sequence[int]] = None,
    ):
        self.ctx_lens = [int(x) for x in ctx_lens]
        self.num_seqs = len(self.ctx_lens)
        self.ragged = ragged_verify_lens is not None
        if self.ragged:
            self.verify_lens = [int(x) for x in ragged_verify_lens]
            assert len(self.verify_lens) == self.num_seqs and min(self.verify_lens) >= 1
            self.seq_len_q = max(self.verify_lens)
        else:
            self.seq_len_q = int(seq_len_q)
            self.verify_lens = [self.seq_len_q] * self.num_seqs
        self.num_rows = sum(self.verify_lens)
        # Sequence and position within the sequence's query rows, per generation row.
        self.row_seq = [s for s, n in enumerate(self.verify_lens) for _ in range(n)]
        self.row_pos = [p for n in self.verify_lens for p in range(n)]
        self.device = torch.device("cuda")
        self.mtp_draft_len = int(mtp_draft_len)
        self.sparse_config = DeepSeekSparseAttentionConfig(
            index_n_heads=32,
            index_head_dim=128,
            index_topk=index_topk,
            skip_indexer_for_short_seqs=False,
            **(sparse_overrides or {}),
        )
        self.attn_cls = get_attention_backend("TRTLLM", self.sparse_config)
        max_seq_len = max(self.ctx_lens) + 2 * self.seq_len_q
        max_positions = 1 << max(16, math.ceil(math.log2(max_seq_len + 1)))
        rope_config = _GlmRopeConfig(max_position_embeddings=max_positions)
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn, rope=RopeParams.from_config(rope_config), is_neox=False
        )
        mla_params = MLAParams(
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            qk_nope_head_dim=NOPE_DIM,
            v_head_dim=V_HEAD_DIM,
            rope_append=True,
            predicted_tokens_per_seq=1,
            hidden_size=HIDDEN_SIZE,
        )
        quant_config = (
            QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)
            if kv_dtype == torch.float8_e4m3fn
            else None
        )
        self.layers = [
            self.attn_cls(
                layer_idx=i,
                num_heads=NUM_HEADS,
                head_dim=HEAD_DIM,
                num_kv_heads=1,
                quant_config=quant_config,
                q_scaling=1.0,
                pos_embd_params=pos_embd_params,
                mla_params=mla_params,
                sparse_attention_config=self.sparse_config,
                dtype=torch.bfloat16,
            )
            for i in range(num_layers)
        ]
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        model_config = ModelConfig(
            mapping=self.mapping,
            sparse_attention_config=self.sparse_config,
            pretrained_config=_PretrainedStub(),
        )
        blocks_per_seq = (max_seq_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
        self.kv_cache_manager = DSACacheManager(
            KvCacheConfig(
                max_tokens=blocks_per_seq * TOKENS_PER_BLOCK * self.num_seqs,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=self.num_seqs,
            mapping=self.mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(kv_dtype)),
            sparse_attn_config=self.sparse_config,
            model_config=model_config,
        )
        for layer in self.layers:
            GlmDsaBackend.adopt(layer)
            if layer.indexer is not None:
                GlmIndexer.adopt(layer.indexer)
        self.request_ids = list(range(self.num_seqs))
        self.kv_cache_manager.add_dummy_requests(self.request_ids, self.ctx_lens)
        if fill_random:
            self.fill_pool(seed, latent_std)
        self.metadata = None
        if not defer_generation:
            self.start_generation()

    def run_context(
        self,
        layer_idx: int,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent: torch.Tensor,
        topk: Optional[torch.Tensor],
        indexer_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        chunk_tokens: Optional[int] = None,
        attention: bool = True,
    ) -> None:
        """Native context step: RoPE and append the context latents of every sequence.

        ``topk`` replaces the context-phase indexer selection (the context output is not
        used). ``indexer_k`` = ``(k_fp8, k_scale)`` of the context tokens, when given, is
        written to the indexer K cache with the native ``_update_k_cache``, as
        ``_forward_dsa_attn`` does before the attention. With ``chunk_tokens`` (one
        sequence only) the context runs as chunked prefill: one context step per chunk,
        each with the previous chunks as cached tokens. With ``attention=False`` only the
        indexer K cache is written (the latent pool keeps its contents).
        """
        if chunk_tokens is None or chunk_tokens >= max(self.ctx_lens):
            chunks = [(None, None)]
        else:
            assert self.num_seqs == 1, "chunked context steps support one sequence"
            ctx = self.ctx_lens[0]
            chunks = [(s, min(s + chunk_tokens, ctx)) for s in range(0, ctx, chunk_tokens)]
        layer = self.layers[layer_idx]
        for start, end in chunks:
            if start is None:
                seq_lens, cached, rows = self.ctx_lens, [0] * self.num_seqs, slice(None)
            else:
                seq_lens, cached, rows = [end - start], [start], slice(start, end)
            ctx_metadata = self.attn_cls.Metadata(
                seq_lens=torch.tensor(seq_lens, dtype=torch.int),
                request_ids=self.request_ids,
                max_num_requests=self.num_seqs,
                num_contexts=self.num_seqs,
                prompt_lens=self.ctx_lens,
                max_num_tokens=sum(seq_lens),
                kv_cache_manager=self.kv_cache_manager,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
                mapping=self.mapping,
                sparse_attention_config=self.sparse_config,
            )
            ctx_metadata.prepare()
            if indexer_k is not None:
                layer.indexer._update_k_cache(indexer_k[0][rows], indexer_k[1][rows], ctx_metadata)
            if not attention:
                torch.cuda.synchronize()
                continue
            layer.indexer.forward_from_projected = Mock(return_value=topk[rows])
            layer.forward(
                fused_q[rows].clone(),
                None,
                None,
                ctx_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent[rows],
                q_pe=q_pe[rows],
                sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=[]),
            )
            torch.cuda.synchronize()

    def start_generation(self) -> None:
        """Allocate this step's rows and build the generation metadata."""
        for request_id, rows in zip(self.request_ids, self.verify_lens):
            for _ in range(rows):
                self.kv_cache_manager.impl.add_token(request_id)
        # The ragged row buffers are allocated at construction; the per-step widths are
        # installed before ``prepare`` (they are not a constructor argument).
        ragged_kwargs = dict(enable_ragged_verification=True) if self.ragged else {}
        self.metadata = self.attn_cls.Metadata(
            seq_lens=torch.tensor(self.verify_lens, dtype=torch.int),
            request_ids=self.request_ids,
            max_num_requests=self.num_seqs,
            num_contexts=0,
            prompt_lens=self.ctx_lens,
            max_num_tokens=self.num_rows,
            kv_cache_manager=self.kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=list(self.ctx_lens)
            ),
            mapping=self.mapping,
            sparse_attention_config=self.sparse_config,
            **ragged_kwargs,
        )
        if self.ragged:
            # Ragged verification sizes its row buffers from the draft ceiling and needs the
            # dense spec-decoding metadata disabled.
            self.metadata.update_spec_dec_param(
                batch_size=self.num_seqs,
                is_spec_decoding_enabled=False,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=self.seq_len_q - 1,
                max_total_draft_tokens=self.seq_len_q - 1,
            )
            self.metadata.ragged_verify_lens = list(self.verify_lens)
        elif self.mtp_draft_len > 0:
            self.metadata.update_spec_dec_param(
                batch_size=self.num_seqs,
                is_spec_decoding_enabled=True,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=self.mtp_draft_len,
                max_total_draft_tokens=self.mtp_draft_len,
            )
        self.metadata.prepare()

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()

    @property
    def cu_len(self) -> int:
        """Entries of the cumulative-length buffers: one per presented row plus one.

        ``forward_absorption_generation`` sizes them the same way: per sequence, and
        per query token under the ragged-verification presentation.
        """
        return (self.num_rows if self.ragged else self.num_seqs) + 1

    # ----- pool access -----------------------------------------------------------------
    def pool(self) -> torch.Tensor:
        """Whole primary pool as an e4m3 tensor."""
        pool = self.kv_cache_manager.get_unique_primary_pool()
        if pool.dtype != torch.float8_e4m3fn and pool.element_size() == 1:
            pool = pool.view(torch.float8_e4m3fn)
        return pool

    def pool_rows(self) -> torch.Tensor:
        """``[num_pool_rows, 576]`` view addressed by global sparse indices."""
        return self.pool().reshape(-1, HEAD_DIM)

    def pool_base_pointer(self, layer_idx: int = 0) -> int:
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E501
            pool_base_pointer,
        )

        return pool_base_pointer(
            self.metadata, self.layers[layer_idx].get_local_layer_idx(self.metadata)
        )

    def fill_pool(self, seed: int, latent_std: float) -> None:
        """Fill every pool row with random finite e4m3 latent/rope values."""
        pool = self.pool()
        gen = torch.Generator(device=self.device).manual_seed(seed)
        flat = pool.view(-1)
        chunk = 1 << 26
        for start in range(0, flat.numel(), chunk):
            end = min(start + chunk, flat.numel())
            values = torch.randn(end - start, device=self.device, generator=gen) * latent_std
            flat[start:end] = values.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    # ----- one step ---------------------------------------------------------------------
    def make_step(self, layer_idx: int, seed: int, q_std: float = 1.0) -> StepBuffers:
        """Random absorbed query and new latents, then the native ``mla_rope_generation``."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        rows = self.num_rows
        fused_q = (
            torch.randn(rows, NUM_HEADS * HEAD_DIM, device=self.device, generator=gen) * q_std
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(rows, NUM_HEADS, ROPE_DIM, device=self.device, generator=gen) * q_std
        ).to(torch.bfloat16)
        latent = torch.randn(rows, HEAD_DIM, device=self.device, generator=gen).to(torch.bfloat16)
        step = StepBuffers(
            fused_q=fused_q,
            q_pe=q_pe,
            latent=latent,
            cu_q_seqlens=torch.empty(self.cu_len, dtype=torch.int32, device=self.device),
            cu_kv_seqlens=torch.empty(self.cu_len, dtype=torch.int32, device=self.device),
            scheduler_counter=torch.empty(1, dtype=torch.uint32, device=self.device),
            bmm1_scale=torch.empty(2, dtype=torch.float32, device=self.device),
            bmm2_scale=torch.empty(1, dtype=torch.float32, device=self.device),
            quant_q=torch.empty(rows, NUM_HEADS * HEAD_DIM, dtype=torch.uint8, device=self.device),
        )
        self.rope_and_quantize(layer_idx, step)
        return step

    def make_step_from(
        self, layer_idx: int, fused_q: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor
    ) -> StepBuffers:
        """Step buffers from given absorbed queries and latents, then ``mla_rope_generation``."""
        step = StepBuffers(
            fused_q=fused_q.clone(),
            q_pe=q_pe.clone(),
            latent=latent.clone(),
            cu_q_seqlens=torch.empty(self.cu_len, dtype=torch.int32, device=self.device),
            cu_kv_seqlens=torch.empty(self.cu_len, dtype=torch.int32, device=self.device),
            scheduler_counter=torch.empty(1, dtype=torch.uint32, device=self.device),
            bmm1_scale=torch.empty(2, dtype=torch.float32, device=self.device),
            bmm2_scale=torch.empty(1, dtype=torch.float32, device=self.device),
            quant_q=torch.empty(
                self.num_rows, NUM_HEADS * HEAD_DIM, dtype=torch.uint8, device=self.device
            ),
        )
        self.rope_and_quantize(layer_idx, step)
        return step

    def rope_and_quantize(self, layer_idx: int, step: StepBuffers) -> None:
        self.layers[layer_idx].mla_rope_generation(
            step.fused_q,
            step.q_pe,
            step.latent,
            self.metadata,
            step.cu_q_seqlens,
            step.cu_kv_seqlens,
            step.scheduler_counter,
            step.bmm1_scale,
            step.bmm2_scale,
            step.quant_q,
        )

    def select_library(self, layer_idx: int, use_sparse_mla: bool) -> None:
        """Put sparse_mla_decode first in (or remove it from) the layer's FMHA library list."""
        layer = self.layers[layer_idx]
        manager = layer._fmha_manager
        libs = [lib for lib in manager.fmha_libs if not isinstance(lib, SparseMlaDecodeFmha)]
        if use_sparse_mla:
            libs.insert(0, SparseMlaDecodeFmha(layer))
        manager.fmha_libs = libs
        manager._cache.clear()

    def selected_library(self, layer_idx: int) -> List[str]:
        return [type(lib).__name__ for lib in self.layers[layer_idx]._fmha_manager._cache.values()]

    def forward(self, layer_idx: int, step: StepBuffers, topk_local: torch.Tensor) -> torch.Tensor:
        """``TrtllmAttention.forward`` for the generation rows; returns ``[R, 8*512]`` BF16."""
        layer = self.layers[layer_idx]
        layer.indexer.forward_from_projected = Mock(return_value=topk_local)
        return self._forward_generation(layer_idx, step, [])

    def update_indexer_cache(
        self, layer_idx: int, k_fp8: torch.Tensor, k_scale: torch.Tensor
    ) -> None:
        """Native indexer K-cache append of this step's rows (``_update_k_cache``)."""
        self.layers[layer_idx].indexer._update_k_cache(k_fp8, k_scale, self.metadata)

    def forward_native_indexer(
        self, layer_idx: int, step: StepBuffers, indexer_intermediates: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``TrtllmAttention.forward`` with the unmodified native indexer.

        ``indexer_intermediates`` is the native ``pre_indexer_proj`` output of this step's
        rows. Returns the attention output and a copy of the local top-k the native
        ``Indexer.forward_from_projected`` produced (and the native remap consumed).
        """
        indexer = self.layers[layer_idx].indexer
        indexer.__dict__.pop("forward_from_projected", None)
        native_select = indexer.forward_from_projected
        captured = []

        def capture(*args, **kwargs):
            topk = native_select(*args, **kwargs)
            captured.append(topk.clone())
            return topk

        # Also keep the decode scores and lengths the native top-k module received.
        top_k_module = indexer.top_k
        top_k_module.__dict__.pop("forward", None)
        native_top_k = top_k_module.forward
        self.last_topk_inputs = None

        def capture_top_k(scores, *args, **kwargs):
            lengths = kwargs.get("sequence_lengths")
            if isinstance(lengths, torch.Tensor):
                self.last_topk_inputs = {
                    "scores": scores.clone(),
                    "sequence_lengths": lengths.clone(),
                    "next_n": kwargs.get("next_n"),
                }
            return native_top_k(scores, *args, **kwargs)

        indexer.forward_from_projected = capture
        top_k_module.forward = capture_top_k
        try:
            out = self._forward_generation(layer_idx, step, list(indexer_intermediates))
        finally:
            indexer.__dict__.pop("forward_from_projected", None)
            top_k_module.__dict__.pop("forward", None)
        assert len(captured) == 1, len(captured)
        return out, captured[0]

    def _forward_generation(
        self, layer_idx: int, step: StepBuffers, indexer_intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        with dispatch.step(dispatch.Phase.DECODE):
            return layer.forward(
                step.fused_q,
                None,
                None,
                self.metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=step.latent,
                q_pe=step.q_pe,
                cu_q_seqlens=step.cu_q_seqlens,
                cu_kv_seqlens=step.cu_kv_seqlens,
                fmha_scheduler_counter=step.scheduler_counter,
                mla_bmm1_scale=step.bmm1_scale,
                mla_bmm2_scale=step.bmm2_scale,
                quant_q_buffer=step.quant_q,
                sparse_backend_args=DSABackendForwardArgs(
                    indexer_intermediates=indexer_intermediates
                ),
            )

    @contextlib.contextmanager
    def capture_sparse_mla_inputs(self) -> Iterator[List[dict]]:
        """Record the global indices, KV lengths and row split of every sparse_mla_decode call (eager only)."""
        calls: List[dict] = []
        original = SparseMlaDecodeFmha.forward

        def forward(fmha, q, k, v, metadata, forward_args):
            gen_lens = metadata.kv_lens_cuda_runtime[metadata.num_contexts :]
            calls.append(
                {
                    "indices": forward_args.sparse_runtime_params.sparse_attn_indices.clone(),
                    "seq_lens_kv": gen_lens.clone(),
                    "seq_len_q": q.shape[0] // gen_lens.shape[0],
                    "ragged": metadata.token_major_gen_view() is not None,
                }
            )
            return original(fmha, q, k, v, metadata, forward_args)

        SparseMlaDecodeFmha.forward = forward
        try:
            yield calls
        finally:
            SparseMlaDecodeFmha.forward = original

    def global_indices(self, layer_idx: int, topk_local: torch.Tensor) -> torch.Tensor:
        """Native local -> global pool-row conversion of a local top-k."""
        layer = self.layers[layer_idx]
        global_idx, _ = transform_local_topk_and_prepare_pool_view(
            topk_local, self.metadata, layer.get_local_layer_idx(self.metadata), is_generation=True
        )
        return global_idx

    def grouped_global_indices(self, layer_idx: int, topk_local: torch.Tensor) -> torch.Tensor:
        """The same conversion through the native grouped (cross-layer fan-out) remap op."""
        from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
            transform_local_topk_and_prepare_pool_view_grouped,
        )

        local_layer = self.layers[layer_idx].get_local_layer_idx(self.metadata)
        page_index_scale, layer_offset = (
            self.metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer)
        )
        layer_ids = torch.tensor([int(layer_offset)], dtype=torch.int32, device=self.device)
        grouped = transform_local_topk_and_prepare_pool_view_grouped(
            topk_local, self.metadata, layer_ids, int(page_index_scale), is_generation=True
        )
        return grouped[0]

    # ----- top-k builders ---------------------------------------------------------------
    def causal_len(self, row: int) -> int:
        """Tokens visible to generation row ``row`` (its own position included)."""
        return self.ctx_lens[self.row_seq[row]] + self.row_pos[row] + 1

    def list_limits(self) -> List[int]:
        """Usable index-list positions per row (native rule: the causal length)."""
        return [self.causal_len(r) for r in range(self.num_rows)]

    def random_topk(self, seed: int, overlap: float = 0.0, recent: int = 0) -> torch.Tensor:
        """Per-row distinct local token indices; ``-1`` fills rows with fewer tokens.

        Args:
            overlap: Fraction of each MTP row's selection copied from the previous row.
            recent: Number of most recent tokens every row always selects.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        topk = torch.full((self.num_rows, TOP_K), -1, dtype=torch.int32)
        prev = None
        for row in range(self.num_rows):
            visible = self.causal_len(row)
            if visible <= TOP_K:
                sel = torch.arange(visible, dtype=torch.int64)
                sel = sel[torch.randperm(visible, generator=gen)]
            else:
                must = torch.arange(visible - min(recent, TOP_K), visible, dtype=torch.int64)
                chosen = set(must.tolist())
                if prev is not None and overlap > 0 and self.row_pos[row] != 0:
                    keep = prev[torch.randperm(prev.numel(), generator=gen)[: int(overlap * TOP_K)]]
                    chosen.update(x for x in keep.tolist() if x < visible)
                while len(chosen) < TOP_K:
                    draw = torch.randint(0, visible, (2 * TOP_K,), generator=gen).tolist()
                    for x in draw:
                        chosen.add(x)
                        if len(chosen) == TOP_K:
                            break
                sel = torch.tensor(sorted(chosen), dtype=torch.int64)
                sel = sel[torch.randperm(sel.numel(), generator=gen)]
            topk[row, : sel.numel()] = sel.to(torch.int32)
            prev = sel
        return topk.to(self.device)


class _PretrainedStub:
    rms_norm_eps = 1e-5


def negative_inside_causal(
    indices: torch.Tensor, seq_lens_kv: torch.Tensor, seq_len_q: int
) -> torch.Tensor:
    """Per row, the negative entries at index-list positions below the row's causal length.

    This is the input class on which sparse_mla_decode deliberately differs from the native kernel
    (sparse_mla_decode skips the entry, the native kernel reads the pool row ``-1`` addresses).
    """
    rows = indices.shape[0]
    row = torch.arange(rows, device=indices.device)
    seq = row // seq_len_q
    usable = seq_lens_kv.to(indices.device).long()[seq] - (seq_len_q - 1 - row % seq_len_q)
    positions = torch.arange(indices.shape[1], device=indices.device)
    inside = positions[None, :] < usable[:, None]
    return ((indices < 0) & inside).sum(dim=1)


# ----- rope_kv_append's top-k bridge as the producer of sparse_mla_decode's indices
# ------------------------------------


def rope_append_selected() -> bool:
    """Whether this process selects rope_kv_append (whose fused bridge then replaces the native remap)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    return "rope_kv_append" in dispatch.SELECTOR_PATHS and dispatch.is_enabled("rope_kv_append")


@contextlib.contextmanager
def capture_rope_append_bridge() -> Iterator[List[dict]]:
    """Record every generation call of rope_kv_append's fused top-k bridge (eager only).

    The DSA predictor looks up rope_kv_append's ``remap_topk_to_global`` entry on every call, so
    replacing the registered entry sees each call. Engagement is counted here, per call:
    ``dispatch.log_capability`` keeps only the first decision per key, and the first
    bridge decision of a process is usually a context batch, which the bridge declines.

    Each record holds the layer, the local top-k the bridge remapped (cloned before the
    call), the call's metadata, and the global indices it returned (the returned tensor).
    Yields an empty list when rope_kv_append is not selected.
    """
    calls: List[dict] = []
    if not rope_append_selected():
        yield calls
        return
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    original = dispatch.get_impl("rope_kv_append", "remap_topk_to_global")

    def bridge(backend, topk, metadata, packet):
        if metadata.num_contexts != 0:
            return original(backend, topk, metadata, packet)
        source = topk if packet.source is None else packet.source
        entry = {
            "layer": backend.layer_idx,
            "local_layer": backend.get_local_layer_idx(metadata),
            "topk": source.clone(),
            "metadata": metadata,
        }
        entry["global"] = original(backend, topk, metadata, packet)
        calls.append(entry)
        return entry["global"]

    dispatch._REGISTRY["rope_kv_append"]["remap_topk_to_global"] = bridge
    try:
        yield calls
    finally:
        dispatch._REGISTRY["rope_kv_append"]["remap_topk_to_global"] = original


def rope_append_bridge_report(call: dict, seq_lens_kv: torch.Tensor, seq_len_q: int) -> dict:
    """Producer contract of one recorded bridge call (after the step completed).

    ``equals_native_remap``: the bridge output equals the native local-to-global remap
    of the same local top-k bit for bit. ``negative_below_causal``: per row, ``-1``
    entries the bridge emitted at index-list positions below the row's causal length
    (``seq_lens_kv`` = KV lengths of the generation sequences, this step included).
    """
    native, _ = transform_local_topk_and_prepare_pool_view(
        call["topk"], call["metadata"], call["local_layer"], is_generation=True
    )
    out = call["global"]
    return {
        "layer": call["layer"],
        "equals_native_remap": bool(out.shape == native.shape and torch.equal(out, native)),
        "negative_below_causal": negative_inside_causal(out, seq_lens_kv, seq_len_q).tolist(),
        "input_negative_below_causal": negative_inside_causal(
            call["topk"], seq_lens_kv, seq_len_q
        ).tolist(),
        "negative_total": int((out < 0).sum().item()),
    }


# ----- CUPTI kernel records ------------------------------------------------------------------

SPARSE_MLA_KERNEL = "sparseMlaDecodeKernel"


def is_native_sparse_fmha(name: str) -> bool:
    """The TRT-LLM-Gen sparse MLA generation kernel (``fmhaSm10x…Sparse…``)."""
    return "fmhaSm1" in name and "Sparse" in name


def kernel_trace(fn: Callable[[], None]) -> List[dict]:
    """CUPTI records of the kernels ``fn`` enqueues (graph replays included), by start time.

    Uses ``torch.profiler``'s CUDA activity tracing; each record has the kernel name, the
    stream and the start / end time in microseconds.
    """
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    finally:
        os.unlink(path)
    kernels = [
        {
            "name": e["name"],
            "stream": e.get("args", {}).get("stream"),
            "start": float(e["ts"]),
            "end": float(e["ts"]) + float(e.get("dur", 0.0)),
        }
        for e in events
        if e.get("cat") == "kernel"
    ]
    return sorted(kernels, key=lambda k: k["start"])


def attention_neighbours(kernels: List[dict]) -> List[dict]:
    """For every sparse attention kernel (sparse_mla_decode or native): its neighbours on its stream.

    ``successor_early``: the next kernel of the stream started before the attention
    kernel ended (it was launched early as a programmatic dependent), by
    ``successor_lead_us``; ``started_before_predecessor_end``: the attention kernel itself
    started early, by ``predecessor_lead_us``.
    """
    by_stream: Dict[object, List[dict]] = {}
    for k in kernels:
        by_stream.setdefault(k["stream"], []).append(k)
    out = []
    for stream_kernels in by_stream.values():
        for i, k in enumerate(stream_kernels):
            is_sparse_mla = SPARSE_MLA_KERNEL in k["name"]
            if not (is_sparse_mla or is_native_sparse_fmha(k["name"])):
                continue
            prev = stream_kernels[i - 1] if i > 0 else None
            nxt = stream_kernels[i + 1] if i + 1 < len(stream_kernels) else None
            out.append(
                {
                    "kernel": "sparse_mla_decode" if is_sparse_mla else "native",
                    "start": k["start"],
                    "duration_us": k["end"] - k["start"],
                    "predecessor": prev["name"][:96] if prev else None,
                    "started_before_predecessor_end": bool(prev and k["start"] < prev["end"]),
                    "predecessor_lead_us": (prev["end"] - k["start"]) if prev else None,
                    "successor": nxt["name"][:96] if nxt else None,
                    "successor_early": bool(nxt and nxt["start"] < k["end"]),
                    "successor_lead_us": (k["end"] - nxt["start"]) if nxt else None,
                }
            )
    return sorted(out, key=lambda a: a["start"])


# ----- FP64 oracles --------------------------------------------------------------------------


def e4m3_round(scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Probability rounding to e4m3 after multiplying by ``scale``."""

    def fn(p: torch.Tensor) -> torch.Tensor:
        q = (p * scale).clamp(max=E4M3_MAX).to(torch.float32).to(torch.float8_e4m3fn)
        return q.to(torch.float64) / scale

    return fn


def fp16_round(p: torch.Tensor) -> torch.Tensor:
    return p.to(torch.float16).to(torch.float64)


def bf16_round(p: torch.Tensor) -> torch.Tensor:
    return p.to(torch.bfloat16).to(torch.float64)


def oracle_attention(
    q_bytes: torch.Tensor,
    pool_rows: torch.Tensor,
    global_idx: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    *,
    list_limit: Optional[Sequence[int]] = None,
    prob_round: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    split: Optional[int] = None,
    step: Optional[int] = None,
) -> torch.Tensor:
    """FP64 sparse MLA reference over the e4m3 query and pool; returns ``[R, 8, 512]``.

    Without ``prob_round`` this is the exact softmax. With it, the probabilities fed to
    P*V are rounded the way an online-softmax kernel would round them: the index list
    is cut into ``split``-position chunks merged at the end, each processed in
    ``step``-position steps against the running maximum; row sums stay unrounded.
    """
    rows = q_bytes.shape[0]
    q = q_bytes.reshape(rows, NUM_HEADS, HEAD_DIM).view(torch.float8_e4m3fn).to(torch.float64)
    pool_bytes = pool_rows.view(torch.uint8)

    def gather(sel: torch.Tensor) -> torch.Tensor:
        return pool_bytes[sel].view(torch.float8_e4m3fn).to(torch.float64)

    out = torch.zeros(rows, NUM_HEADS, KV_LORA_RANK, dtype=torch.float64, device=q.device)
    positions = torch.arange(global_idx.shape[1], device=global_idx.device)
    for r in range(rows):
        idx = global_idx[r].to(torch.int64)
        valid = idx >= 0
        if list_limit is not None:
            valid &= positions < list_limit[r]
        if not bool(valid.any()):
            continue
        if prob_round is None:
            k = gather(idx[valid])
            s = (q[r] @ k.T) * bmm1_scale
            p = torch.softmax(s, dim=-1)
            out[r] = p @ k[:, :KV_LORA_RANK]
            continue
        chunk = split or global_idx.shape[1]
        stride = step or chunk
        maxes, sums, accs = [], [], []
        for a in range(0, global_idx.shape[1], chunk):
            m = torch.full((NUM_HEADS,), -math.inf, dtype=torch.float64, device=q.device)
            row_sum = torch.zeros(NUM_HEADS, dtype=torch.float64, device=q.device)
            acc = torch.zeros(NUM_HEADS, KV_LORA_RANK, dtype=torch.float64, device=q.device)
            for b in range(a, min(a + chunk, global_idx.shape[1]), stride):
                sel = valid[b : b + stride]
                if not bool(sel.any()):
                    continue
                k = gather(idx[b : b + stride][sel])
                s = (q[r] @ k.T) * bmm1_scale
                m_new = torch.maximum(m, s.max(dim=-1).values)
                alpha = torch.exp(m - m_new)
                p = torch.exp(s - m_new[:, None])
                row_sum = row_sum * alpha + p.sum(dim=-1)
                acc = acc * alpha[:, None] + prob_round(p) @ k[:, :KV_LORA_RANK]
                m = m_new
            maxes.append(m)
            sums.append(row_sum)
            accs.append(acc)
        m_all = torch.stack(maxes)
        global_max = m_all.max(dim=0).values
        weights = torch.where(
            torch.isinf(m_all), torch.zeros_like(m_all), torch.exp(m_all - global_max)
        )
        total = (weights * torch.stack(sums)).sum(dim=0)
        out[r] = (weights[:, :, None] * torch.stack(accs)).sum(dim=0) / total[:, None]
    return out * bmm2_scale


def error_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """Normalized RMS and max-abs error of ``candidate`` against ``reference``."""
    c = candidate.reshape(reference.shape).to(torch.float64)
    diff = c - reference
    ref_rms = reference.pow(2).mean().sqrt().item()
    return {
        "nrms": diff.pow(2).mean().sqrt().item() / max(ref_rms, 1e-30),
        "max_abs": diff.abs().max().item(),
        "ref_rms": ref_rms,
        "nonfinite": int((~torch.isfinite(c)).sum().item()),
    }


# ----- native-versus-sparse_mla_decode comparison
# -----------------------------------------------------------


def record(filename: str, case: str, payload: dict) -> None:
    """Print one JSON result line and append it to ``$GLM_MOE_DSA_SPARSE_MLA_DECODE_RESULTS/<filename>``."""
    import json
    import os

    line = json.dumps({"case": case, **payload}, sort_keys=True)
    print(line)
    results_dir = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_RESULTS")
    if results_dir:
        with open(os.path.join(results_dir, filename), "a") as f:
            f.write(line + "\n")


def run_both(harness: DsaGenHarness, step: StepBuffers, topk: torch.Tensor):
    """Native and sparse_mla_decode outputs for one step plus the FP64 oracle with the native rules."""
    harness.select_library(0, use_sparse_mla=False)
    native = harness.forward(0, step, topk).clone()
    assert "SparseMlaDecodeFmha" not in harness.selected_library(0)
    harness.select_library(0, use_sparse_mla=True)
    sparse_out = harness.forward(0, step, topk).clone()
    assert harness.selected_library(0) == ["SparseMlaDecodeFmha"], harness.selected_library(0)
    global_idx = harness.global_indices(0, topk)
    return native, sparse_out, global_idx, oracle_for(harness, step, global_idx)


def usable_rows(harness: DsaGenHarness, global_idx: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(global_idx.shape[1], device=global_idx.device)
    limits = torch.tensor(harness.list_limits(), device=global_idx.device)[:, None]
    return ((global_idx >= 0) & (positions[None, :] < limits)).any(dim=1)


def oracle_for(harness: DsaGenHarness, step: StepBuffers, global_idx: torch.Tensor) -> torch.Tensor:
    """FP64 oracle with the native rules for this step's query and the current pool."""
    return oracle_attention(
        step.quant_q,
        harness.pool_rows(),
        global_idx,
        float(step.bmm1_scale[0]),
        float(step.bmm2_scale[0]),
        list_limit=harness.list_limits(),
    )


def compare(
    harness: DsaGenHarness,
    step: StepBuffers,
    topk: torch.Tensor,
    case: str,
    filename: str = "sparse_mla_correctness.jsonl",
    **extra,
) -> dict:
    """Check sparse_mla_decode against native and the oracle with the frozen error budget."""
    native, sparse_out, global_idx, oracle = run_both(harness, step, topk)
    return check_budget(harness, native, sparse_out, global_idx, oracle, case, filename, **extra)


def check_budget(
    harness: DsaGenHarness,
    native: torch.Tensor,
    sparse_out: torch.Tensor,
    global_idx: torch.Tensor,
    oracle: torch.Tensor,
    case: str,
    filename: str = "sparse_mla_correctness.jsonl",
    **extra,
) -> dict:
    """The frozen error budget for one native and one sparse_mla_decode output of the same inputs."""
    native_err = error_stats(native.double(), oracle)
    sparse_mla_err = error_stats(sparse_out.double(), oracle)
    sparse_mla_vs_native = error_stats(sparse_out.double(), native.double().reshape(oracle.shape))
    record(
        filename,
        case,
        {
            "native_vs_oracle": native_err,
            "sparse_mla_vs_oracle": sparse_mla_err,
            "sparse_mla_vs_native": sparse_mla_vs_native,
            **extra,
        },
    )
    native_rows = native.reshape(harness.num_rows, -1)
    sparse_mla_rows = sparse_out.reshape(harness.num_rows, -1)
    live = usable_rows(harness, global_idx)
    if bool((~live).any()):
        torch.testing.assert_close(
            sparse_mla_rows[~live], native_rows[~live], rtol=0, atol=0, equal_nan=True
        )
    assert sparse_mla_err["nonfinite"] == native_err["nonfinite"]
    assert sparse_mla_err["nrms"] <= 1.25 * native_err["nrms"] + 2e-4, (sparse_mla_err, native_err)
    assert sparse_mla_err["max_abs"] <= 2.0 * native_err["max_abs"] + 1e-3, (
        sparse_mla_err,
        native_err,
    )
    oracle_rows = oracle.reshape(harness.num_rows, -1)
    assert torch.allclose(sparse_mla_rows[live].double(), oracle_rows[live], atol=0.12, rtol=0.01)
    return {
        "native": native,
        "sparse_mla_decode": sparse_out,
        "oracle": oracle,
        "global_idx": global_idx,
    }
