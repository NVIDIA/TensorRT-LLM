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
"""Inkling attention backend: KV write, page tables, prefill/decode dispatch."""

import torch

from ..trtllm import TrtllmAttention
from .kernels import (
    build_page_table,
    inkling_decode_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)
from .metadata import InklingAttentionMetadata


class InklingTritonAttention(TrtllmAttention):
    """Runs Inkling's Triton attention over the paged KV cache.

    Entry point is :meth:`forward_inkling`, not the inherited
    ``AttentionBackend.forward``: Inkling carries a per-(query, head,
    relative-distance) additive bias (``rel_logits``) that has no home in the
    shared ``AttentionForwardArgs``, so the backend exposes a name-explicit
    method instead of widening a dataclass shared by every other model. This
    mirrors MiniMax-M3's ``forward_sparse`` (``sparse/minimax_m3/``).

    Subclassing ``TrtllmAttention`` rather than ``AttentionBackend`` is
    deliberate: :class:`InklingAttentionMetadata` extends
    ``TrtllmAttentionMetadata`` to reuse its ``prepare()`` (page tables, seq
    lens), the model engine gates attention warmup on that same subclass check,
    and the base ``Attention`` module calls ``support_fused_qkv()`` /
    ``update_quant_config()`` on whatever backend it built.

    Three per-layer scalars the moved compute reads (``sm_scale``,
    ``rel_extent``, ``window_left``) are assigned by ``InklingAttention``
    after construction, because ``create_attention()`` has a fixed kwarg list
    with no passthrough. ``layer_idx`` comes from ``create_attention`` and is
    the *global* layer index, which is what ``KVCacheManagerV2`` expects (it
    maps through ``layer_offsets`` itself).
    """

    Metadata = InklingAttentionMetadata

    def forward_inkling(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel_logits: torch.Tensor,
        attn_metadata: InklingAttentionMetadata,
        *,
        allow_mixed: bool = False,
    ) -> torch.Tensor:
        """Dispatch prefill / decode over the paged cache, supporting mixed
        context+generation batches.

        The runtime packs context requests first (each with its full new-token
        span) then one-token generation requests (``seq_lens == 1``). We slice
        the packed q/k/v/rel_logits + per-request metadata at that boundary and
        run the context slice through the prefill kernel and the generation
        slice through the paged-decode kernel, concatenating the outputs. Pure
        context (``num_contexts == num_seqs``) and pure generation
        (``num_contexts == 0``) fall out as the single-slice cases.
        """
        # KVCacheManagerV2 takes the global layer index and maps it through
        # ``layer_offsets`` itself.
        cache_layer = self.layer_idx
        kv = attn_metadata.kv_cache_manager.get_buffers(cache_layer, kv_layout="HND")
        # kv: [num_pages, 2, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = kv[:, 0], kv[:, 1]
        page_size = kv.shape[3]
        mgr = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids
        num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        num_seqs = len(seq_lens)
        ctx_tokens = sum(seq_lens[:num_contexts])

        # A mixed context+generation batch needs the per-request short-conv state
        # pool: the stateless path would convolve across the context/generation
        # boundary. Refuse it unless the pool path is active.
        if 0 < num_contexts < num_seqs and not allow_mixed:
            raise NotImplementedError(
                "InklingAttention: mixed context+generation batch needs the "
                "short-conv state pool (pass conv_cache/conv_rt); the stateless "
                "short-conv path cannot mix a batch"
            )

        outs = []
        if num_contexts > 0:
            outs.append(
                self._run_context(
                    q[:ctx_tokens],
                    k[:ctx_tokens],
                    v[:ctx_tokens],
                    rel_logits[:ctx_tokens],
                    seq_lens[:num_contexts],
                    num_cached[:num_contexts],
                    request_ids[:num_contexts],
                    mgr,
                    cache_layer,
                    k_cache,
                    v_cache,
                    page_size,
                )
            )
        if num_contexts < num_seqs:
            outs.append(
                self._run_generation(
                    q[ctx_tokens:],
                    k[ctx_tokens:],
                    v[ctx_tokens:],
                    rel_logits[ctx_tokens:],
                    num_cached[num_contexts:],
                    request_ids[num_contexts:],
                    mgr,
                    cache_layer,
                    k_cache,
                    v_cache,
                    page_size,
                    attn_metadata,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _run_context(
        self,
        q,
        k,
        v,
        rel_logits,
        seq_lens,
        num_cached,
        request_ids,
        mgr,
        cache_layer,
        k_cache,
        v_cache,
        page_size,
    ):
        device = q.device
        # Persist new K/V to the paged cache for later generation reuse.
        block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
        off = 0
        for i, sl in enumerate(seq_lens):
            write_kv_cache_hnd(
                k_cache,
                v_cache,
                k[off : off + sl],
                v[off : off + sl],
                block_ids[i],
                int(num_cached[i]),
                page_size,
            )
            off += sl
        cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
        max_seqlen = max(seq_lens)
        # NOTE: this attends only to the tokens of THIS call. The write above
        # honours ``num_cached``, but ``inkling_prefill_attention`` takes no
        # paged-KV argument, so a context request carrying cached history
        # (chunked prefill, or a reused prefix) would silently drop all of it.
        # Both are refused up front by
        # ``reject_unsupported_inkling_kv_cache_features``; adding either one
        # means giving Inkling a chunked-context prefill path that reads the
        # pages back while carrying rel_logits and the sliding window across the
        # boundary. ``num_cached`` is non-zero here only in that unsupported
        # case, which is why the write path already accounts for it.
        return inkling_prefill_attention(
            q, k, v, cu, max_seqlen, self.sm_scale, rel_logits, self.rel_extent, self.window_left
        )

    def _run_generation(
        self,
        q,
        k,
        v,
        rel_logits,
        num_cached,
        request_ids,
        mgr,
        cache_layer,
        k_cache,
        v_cache,
        page_size,
        attn_metadata,
    ):
        device = q.device
        # --- Runtime CUDA-graph-safe path. ---------------------------------
        # ``InklingAttentionMetadata.prepare()`` published this batch's decode
        # metadata into stable GPU buffers, so the captured forward does zero
        # host->device copy: it reads ``ink_seq_lens`` / ``ink_page_table`` and
        # persists the new K/V with an in-graph scatter whose (page, offset)
        # indices are derived on-GPU. Padding rows carry their own dummy request
        # slots, so the scatter never corrupts a real request's page.
        num_req = q.shape[0]
        if getattr(attn_metadata, "ink_num_gen", 0) == num_req:
            sl = attn_metadata.ink_seq_lens[:num_req]
            pt = attn_metadata.ink_page_table[cache_layer][:num_req]
            pos = (sl - 1).long()  # write slot = total_kv_len - 1 = num_cached
            page_row = torch.div(pos, page_size, rounding_mode="floor")
            offs = pos - page_row * page_size
            pages = pt.gather(1, page_row.unsqueeze(1)).squeeze(1).long()
            # Paired advanced indices select one (page, slot) per request ->
            # [num_req, num_kv_heads, head_dim], matching the new k/v.
            k_cache[pages, :, offs, :] = k.to(k_cache.dtype)
            v_cache[pages, :, offs, :] = v.to(v_cache.dtype)
            return inkling_decode_attention(
                q,
                k_cache,
                v_cache,
                sl,
                pt,
                page_size,
                self.sm_scale,
                rel_logits,
                self.rel_extent,
                self.window_left,
            )
        # Eager fallback (never captured): the decode metadata was not published,
        # so build it here from the host block table, like the context path. This
        # path is illegal under CUDA graph, and the usual cause is an explicit
        # ``attn_backend`` override beating the model default -- say so.
        if getattr(attn_metadata, "is_cuda_graph", False):
            raise RuntimeError(
                "Inkling decode metadata was not published for a CUDA-graph "
                f"batch (ink_num_gen="
                f"{getattr(attn_metadata, 'ink_num_gen', None)}, expected "
                f"{num_req}); attn_metadata is "
                f"{type(attn_metadata).__name__}, not InklingAttentionMetadata. "
                "Inkling requires attn_backend='INKLING'; remove any "
                "attn_backend override from --extra_llm_api_options / "
                "LLM(attn_backend=...) and let the model default apply."
            )
        num_req = len(request_ids)
        block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
        for i in range(num_req):
            write_kv_cache_hnd(
                k_cache,
                v_cache,
                k[i : i + 1],
                v[i : i + 1],
                block_ids[i],
                int(num_cached[i]),
                page_size,
            )
        total = [int(num_cached[i]) + 1 for i in range(num_req)]
        decode_seq_lens = torch.tensor(total, dtype=torch.int32, device=device)
        max_pages = max(len(b) for b in block_ids)
        decode_page_table = build_page_table(block_ids, max_pages, device)
        return inkling_decode_attention(
            q,
            k_cache,
            v_cache,
            decode_seq_lens,
            decode_page_table,
            page_size,
            self.sm_scale,
            rel_logits,
            self.rel_extent,
            self.window_left,
        )
