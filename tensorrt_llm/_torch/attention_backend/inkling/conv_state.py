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
"""Per-request short-conv state for Inkling: the pool, the per-forward split,
and the helper that runs one short convolution through them.

This is runtime state, not model weights: it has the lifetime of the paged KV
cache, is owned by :class:`~.cache_manager.InklingHybridCacheManager`, and is
published each step from :meth:`~.metadata.InklingAttentionMetadata.prepare`.
It therefore lives beside the cache manager and the metadata rather than in
``models/modeling_inkling.py``, which owns only the weight-bearing modules.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

import torch

from ...._utils import prefer_pinned

# Per-request short-conv state of one decoder layer, carried across decode steps.
# Each field is a ``[num_req, channels, sconv_kernel_size - 1]`` window of the
# previous pre-conv inputs (oldest first): ``k``/``v`` for the attention k/v
# convs (TP-sharded), ``attn``/``mlp`` for the residual-stream convs (replicated).
InklingConvState = namedtuple("InklingConvState", ["k", "v", "attn", "mlp"])


class InklingConvStateCache:
    """Runtime-owned per-request short-conv state pool for the whole decoder.

    Carries the four causal short-convs of every decoder layer per request
    across decode steps, with the same lifetime as the paged KV cache.

    Per layer it allocates the four :class:`InklingConvState` buffers, each
    ``[max_batch, channels, kernel_size - 1]``. The k/v conv channels follow the
    fused-qkv k/v split (TP-sharded); the residual-stream convs are replicated.

    All buffers, including the ``[max_batch]`` int32 ``state_indices``, keep
    stable device addresses and are mutated in place, so a captured CUDA graph
    replays cleanly (the Mamba2Metadata stable-pointer pattern).
    """

    def __init__(
        self,
        pretrained_config,
        tp_size: int,
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Takes the pretrained config + tp_size rather than a ``ModelConfig`` so
        # the KV cache manager can build the pool from what it already has.
        # Accept either the text config or the top-level multimodal one.
        config = getattr(pretrained_config, "text_config", pretrained_config)
        kwin = config.sconv_kernel_size - 1
        self.max_batch_size = max_batch_size
        self.kwin = kwin

        def buf(channels):
            return torch.zeros(max_batch_size, channels, kwin, device=device, dtype=dtype)

        self._layers: List[InklingConvState] = []
        for i in range(config.num_hidden_layers):
            kv_dim = (config.layer_num_kv_heads(i) * config.layer_head_dim(i)) // tp_size
            hidden = config.hidden_size
            self._layers.append(
                InklingConvState(k=buf(kv_dim), v=buf(kv_dim), attn=buf(hidden), mlp=buf(hidden))
            )
        # Stable per-request slot-index buffer, refreshed in place per forward
        # from input preparation (see :meth:`write_state_indices`) so a captured
        # decode graph aliases it and every replay sees the current batch.
        self.state_indices = torch.arange(max_batch_size, dtype=torch.int32, device=device)
        # Pinned host staging for that write: one async H2D copy per forward,
        # legal under graph capture. Kept in lock-step size with ``state_indices``.
        self.state_indices_cpu = torch.zeros(
            max_batch_size, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._slot_of = {}
        self._free = list(range(max_batch_size - 1, -1, -1))

    def layer_state(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers for ``layer_idx`` (pool views)."""
        return self._layers[layer_idx]

    def slots_for(self, request_ids: List[int]) -> List[int]:
        """Map request ids to their (stable) pool rows, allocating new ones.

        Fresh requests get a zero-initialised slot; existing requests keep their
        row so their carried short-conv windows persist across decode steps.

        If a single forward presents more *fresh* requests than the pool has
        free rows, the pool grows to fit (see :meth:`_grow`). Steady-state
        serving is bounded by ``max_batch_size`` (+1 CUDA-graph pad row) and
        never triggers growth, but the one-time KV-cache estimation forward can
        exceed it: that dummy batch is sized to saturate ``max_num_tokens`` (and
        is replicated ``x tp_size`` under attention DP), independent of
        ``max_batch_size``. Growing there (instead of ``IndexError`` on an empty
        free list) lets estimation profile memory correctly, and because growth
        only happens in that eager estimation/warmup window the buffers a later
        CUDA graph captures are the final, pointer-stable ones.
        """
        num_new = sum(1 for r in request_ids if r not in self._slot_of)
        if num_new > len(self._free):
            self._grow(num_new - len(self._free))
        slots = []
        for r in request_ids:
            if r not in self._slot_of:
                slot = self._free.pop()
                self._slot_of[r] = slot
                for st in self._layers:
                    for t in st:
                        t[slot].zero_()
            slots.append(self._slot_of[r])
        return slots

    def _grow(self, extra: int):
        """Append ``extra`` fresh (zeroed) rows to every per-request buffer.

        Reallocates each layer's four short-conv state tensors and the shared
        ``state_indices`` scratch to ``max_batch_size + extra`` rows, copying the
        existing rows forward so any in-flight request keeps its carried window,
        and returns the new rows to the free list. Called only from
        :meth:`slots_for` when a batch needs more rows than the pool owns; see
        there for why that happens (KV-cache estimation / attention-DP), and why
        it is safe w.r.t. CUDA-graph pointer stability.
        """
        old = self.max_batch_size
        new = old + extra
        for i, st in enumerate(self._layers):
            grown = []
            for t in st:
                buf = torch.zeros(new, t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
                buf[:old].copy_(t)
                grown.append(buf)
            self._layers[i] = InklingConvState(*grown)
        self.state_indices = torch.arange(new, dtype=torch.int32, device=self.state_indices.device)
        # Keep the pinned host-staging buffer sized in lock-step, else the eager
        # H2D write in write_state_indices would index past its end.
        self.state_indices_cpu = torch.zeros(new, dtype=torch.int32, pin_memory=prefer_pinned())
        # New rows old..new-1 join the free list, popped ascending like __init__.
        self._free = list(range(new - 1, old - 1, -1)) + self._free
        self.max_batch_size = new

    def write_state_indices(self, request_ids: List[int], is_graph: bool) -> List[int]:
        """Resolve ``request_ids`` to pool rows and publish them into the stable
        ``state_indices`` CUDA buffer -- the eager, pre-capture slot write.

        Returns the resolved slots in packed batch order (contexts first). A
        captured decode graph aliases ``state_indices``, so this must run every
        forward from eager input-prep, not inside ``model.forward``.

        ``is_graph`` guards pool-pointer stability: growth reallocates
        ``state_indices`` and would strand the captured pointer, so it may only
        happen while eager. The pool is sized above any graph batch, so this is
        a loud check on an otherwise silent decode corruption.
        """
        before = self.state_indices.data_ptr()
        slots = self.slots_for(request_ids)
        if is_graph and self.state_indices.data_ptr() != before:
            raise RuntimeError(
                "Inkling short-conv pool grew during CUDA graph capture/replay; "
                "the pool must be sized to the max graph batch up front (a grown "
                "pool strands the captured state_indices pointer)."
            )
        n = len(slots)
        self.state_indices_cpu[:n].copy_(torch.tensor(slots, dtype=torch.int32))
        self.state_indices[:n].copy_(self.state_indices_cpu[:n], non_blocking=True)
        return slots

    def free(self, request_ids: List[int]):
        for r in request_ids:
            slot = self._slot_of.pop(r, None)
            if slot is not None:
                self._free.append(slot)


@dataclass
class InklingConvRuntime:
    """Per-forward short-conv plumbing for the pool path (all layers share it).

    Splits the packed ``[context tokens | one-token generation]`` batch at the
    context boundary so each of the four short-convs seeds the pool for context
    requests (varlen ``causal_conv1d_fn``) and updates it in place for generation
    requests (``causal_conv1d_update``), exactly like the paged attention split
    in :meth:`InklingAttention._attention`.
    """

    num_ctx_tokens: int
    ctx_indices: Optional[torch.Tensor]  # int32 pool slots, context requests
    gen_indices: Optional[torch.Tensor]  # int32 pool slots, generation requests
    query_start_loc: Optional[torch.Tensor]  # int32 [n_ctx+1] varlen offsets
    has_initial_state: Optional[torch.Tensor]  # bool [n_ctx]

    @classmethod
    def build(cls, attn_metadata, cache: InklingConvStateCache) -> "InklingConvRuntime":
        """Publish this batch's pool rows, then build the context/generation split.

        The split mirrors the attention split: context requests first (each with
        its full new-token span), then one-token generation requests. Called from
        ``InklingAttentionMetadata.prepare()``, so the host->device slot write
        lands outside the captured ``model.forward``.
        """
        is_graph = bool(getattr(attn_metadata, "is_cuda_graph", False))
        slots = cache.write_state_indices(list(attn_metadata.request_ids), is_graph)
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        state_indices = cache.state_indices
        device = state_indices.device
        num_ctx_tokens = sum(seq_lens[:num_contexts])
        ctx_indices = state_indices[:num_contexts] if num_contexts else None
        gen_indices = (
            state_indices[num_contexts : len(slots)] if num_contexts < len(slots) else None
        )
        query_start_loc = has_initial_state = None
        if num_contexts:
            cu = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.tensor(seq_lens[:num_contexts], dtype=torch.int32, device=device).cumsum(
                0
            )
            query_start_loc = cu
            # Fresh prefill carries no prior conv window. This is correct only
            # because the two features that would leave a context request with a
            # prior window -- KV block reuse and chunked prefill -- are refused
            # up front by ``reject_unsupported_inkling_kv_cache_features``.
            #
            # Do NOT "fix" this line on its own. Deriving has_initial_state from
            # ``num_cached_tokens_per_seq`` (the ``Mamba2Metadata`` pattern) is
            # necessary but NOT sufficient: ``_run_context`` attends only to the
            # tokens of its own call, so a request carrying cached history would
            # still lose that history in attention and stay silently wrong. See
            # ``reject_unsupported_inkling_kv_cache_features``.
            has_initial_state = torch.zeros(num_contexts, dtype=torch.bool, device=device)
        return cls(
            num_ctx_tokens=num_ctx_tokens,
            ctx_indices=ctx_indices,
            gen_indices=gen_indices,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
        )


def apply_short_conv(
    sconv,  # models.modeling_inkling.InklingShortConv (duck-typed to avoid a cycle)
    x: torch.Tensor,
    pool_buf: Optional[torch.Tensor],
    rt: Optional[InklingConvRuntime],
) -> torch.Tensor:
    """Run one short-conv over a (possibly mixed) batch through the state pool.

    ``rt is None`` -> stateless full-sequence causal conv (no pool registered).
    Otherwise the context slice seeds ``pool_buf`` (varlen prefill) and the
    generation slice updates it in place at ``rt.gen_indices`` (decode), then the
    two outputs are concatenated in packed order. ``pool_buf`` is this conv's
    ``[max_batch, channels, kernel-1]`` state buffer from
    :class:`InklingConvStateCache`.
    """
    if rt is None:
        return sconv(x)
    parts = []
    nctx = rt.num_ctx_tokens
    if nctx > 0:
        parts.append(
            sconv.forward(
                x[:nctx],
                conv_state=pool_buf,
                cache_indices=rt.ctx_indices,
                query_start_loc=rt.query_start_loc,
                has_initial_state=rt.has_initial_state,
                is_decode=False,
            )
        )
    if x.shape[0] > nctx:
        parts.append(
            sconv.forward(
                x[nctx:], conv_state=pool_buf, cache_indices=rt.gen_indices, is_decode=True
            )
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
