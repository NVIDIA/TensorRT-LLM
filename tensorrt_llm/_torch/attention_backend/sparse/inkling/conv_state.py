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
cache, is owned by :class:`~.cache_manager.InklingHybridCacheManager`, and its
rows are published each step from
:meth:`~.metadata.InklingAttentionMetadata.prepare`, before the captured region.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

from ....._utils import prefer_pinned

# Per-request short-conv state of one decoder layer, carried across decode steps.
# Each field is a ``[num_req, channels, sconv_kernel_size - 1]`` window of the
# previous pre-conv inputs (oldest first): ``k``/``v`` for the attention k/v
# convs (TP-sharded), ``attn``/``mlp`` for the residual-stream convs (replicated).
InklingConvState = namedtuple("InklingConvState", ["k", "v", "attn", "mlp"])


class InklingRole:
    """V2 pool roles for the four short convolutions of a decoder layer.

    Four roles despite only two widths: k/v and attn/mlp merely happen to pair up
    in size today, and V2 coalesces same-sized buffers on its own.
    """

    CONV_K = DataRole("inkling_conv_k")
    CONV_V = DataRole("inkling_conv_v")
    CONV_ATTN = DataRole("inkling_conv_attn")
    CONV_MLP = DataRole("inkling_conv_mlp")


# Order matches InklingConvState's fields so the two can be zipped.
CONV_ROLES = (InklingRole.CONV_K, InklingRole.CONV_V, InklingRole.CONV_ATTN, InklingRole.CONV_MLP)


class InklingConvStateCache:
    """Runtime-owned per-request short-conv state pool for the whole decoder.

    Per layer it holds the four :class:`InklingConvState` buffers, each
    ``[num_slots, channels, kernel_size - 1]``; the k/v conv channels follow the
    fused-qkv k/v split (TP-sharded), the residual-stream convs are replicated.
    The buffers come from the ``allocate`` callback, which
    :class:`InklingHybridCacheManager` wires to V2 SSM-layer pool memory: this
    class owns the request-to-row mapping, V2 owns the bytes.

    The row count is fixed at construction and every buffer keeps a stable device
    address, mutated in place, so a captured graph replays cleanly.
    """

    @staticmethod
    def reserved_slot_count(*, reserve_attention_dp_slot: bool) -> int:
        """Rows sitting above the per-request ones: one shared by every CUDA-graph
        padding sentinel, plus one for the attention-DP idle dummy when enabled.

        Static because the cache manager needs the same count to declare
        V2's min-slots floor, and the two must agree exactly.
        """
        return 1 + int(reserve_attention_dp_slot)

    def __init__(
        self,
        pretrained_config,
        tp_size: int,
        num_request_slots: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        reserve_attention_dp_slot: bool = False,
        max_draft_len: int = 0,
        allocate: Optional[Callable[[int, object, List[int]], torch.Tensor]] = None,
    ):
        # Accept either the text config or the top-level multimodal one.
        config = getattr(pretrained_config, "text_config", pretrained_config)
        kwin = config.sconv_kernel_size - 1
        if num_request_slots <= 0:
            raise ValueError(f"num_request_slots must be > 0, got {num_request_slots}")
        self.num_request_slots = num_request_slots
        # Reserved rows sit above the real ones so a real slot id is always a
        # valid index into the first ``num_request_slots`` rows.
        self._padding_slot = num_request_slots
        self._attention_dp_dummy_slot = num_request_slots + 1 if reserve_attention_dp_slot else None
        self._max_draft_len = max(0, int(max_draft_len))
        num_slots = num_request_slots + self.reserved_slot_count(
            reserve_attention_dp_slot=reserve_attention_dp_slot
        )
        self.num_slots = num_slots
        self.kwin = kwin

        # ``allocate`` hands back V2 pool memory when the cache manager built us;
        # standalone callers fall back to a private allocation of the same shapes.
        # Pool memory arrives uninitialized, which is safe only because
        # ``slots_for`` zeroes a row when it first hands it out.
        if allocate is None:

            def allocate(_layer_idx, _role, state_shape):
                return torch.zeros(num_slots, *state_shape, device=device, dtype=dtype)

        self._layers: List[InklingConvState] = []
        for i in range(config.num_hidden_layers):
            kv_dim = (config.layer_num_kv_heads(i) * config.layer_head_dim(i)) // tp_size
            hidden = config.hidden_size
            bufs = [
                allocate(i, role, [channels, kwin])
                for role, channels in zip(CONV_ROLES, (kv_dim, kv_dim, hidden, hidden))
            ]
            for buf in bufs:
                if buf.shape[0] < num_slots:
                    raise RuntimeError(
                        f"Inkling conv pool layer {i}: allocator returned "
                        f"{buf.shape[0]} slots but the pool needs {num_slots} "
                        f"({num_request_slots} request + reserved). The V2 SSM "
                        "layer was sized from a different slot count than "
                        "InklingConvStateCache assumes."
                    )
            self._layers.append(InklingConvState(*(b[:num_slots] for b in bufs)))
        # Refreshed in place per forward so a captured graph aliases it and every
        # replay sees the current batch. Indexed by batch position, not by slot.
        self.state_indices = torch.arange(num_slots, dtype=torch.int32, device=device)
        # Pinned host staging for that write: one async H2D copy per forward.
        self.state_indices_cpu = torch.zeros(
            num_slots, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._slot_of = {}
        self._free = list(range(num_request_slots - 1, -1, -1))

    def conv_state_bytes(self) -> int:
        """Total device bytes this pool holds. Reported by the cache manager."""
        return sum(t.numel() * t.element_size() for st in self._layers for t in st)

    def layer_state(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers for ``layer_idx`` (pool views)."""
        return self._layers[layer_idx]

    def _reserved_slot_for(self, request_id: int) -> Optional[int]:
        """The reserved row ``request_id`` aliases, or None for a real request.

        ``cuda_graph_runner`` caches one dummy id per runtime draft length, so the
        whole descending range maps here, not just the exact sentinel.
        """
        from ....pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        from ....pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID

        if CUDA_GRAPH_DUMMY_REQUEST_ID - self._max_draft_len <= request_id:
            return self._padding_slot
        if request_id == ATTENTION_DP_DUMMY_REQUEST_ID:
            return self._attention_dp_dummy_slot
        return None

    def slots_for(self, request_ids: List[int]) -> List[int]:
        """Map request ids to their (stable) pool rows, allocating new ones.

        Fresh requests get a zero-initialised row; existing ones keep theirs so
        their conv windows persist across decode steps. Padding sentinels and the
        attention-DP dummy alias their reserved rows. Raises on exhaustion rather
        than handing out a row another request is still carrying state in.
        """
        slots = []
        for r in request_ids:
            slot = self._slot_of.get(r)
            if slot is None:
                slot = self._reserved_slot_for(r)
                if slot is None:
                    if not self._free:
                        raise RuntimeError(
                            "Inkling short-conv pool is out of rows "
                            f"({self.num_request_slots} request rows, all live). "
                            "Every resident sequence needs one row; a row is "
                            "returned only by the cache manager's free_resources."
                        )
                    slot = self._free.pop()
                self._slot_of[r] = slot
                for st in self._layers:
                    for t in st:
                        t[slot].zero_()
            slots.append(slot)
        return slots

    def write_state_indices(self, request_ids: List[int]) -> List[int]:
        """Resolve ``request_ids`` to pool rows and publish them into the stable
        ``state_indices`` CUDA buffer, in packed batch order (contexts first).

        A captured decode graph aliases ``state_indices``, so this must run every
        forward from eager input-prep -- it is reached from the manager's
        ``get_state_indices``, which the base ``prepare()`` calls.
        """
        slots = self.slots_for(request_ids)
        n = len(slots)
        if n > self.num_slots:
            raise RuntimeError(
                f"Inkling short-conv batch has {n} rows but the pool publishes "
                f"only {self.num_slots} state indices; the pool is sized to the "
                "padded scheduler batch, so this signals a batch-shape mismatch."
            )
        self.state_indices_cpu[:n].copy_(torch.tensor(slots, dtype=torch.int32))
        self.state_indices[:n].copy_(self.state_indices_cpu[:n], non_blocking=True)
        return slots

    def free(self, request_ids: List[int]):
        for r in request_ids:
            slot = self._slot_of.pop(r, None)
            # Reserved rows are permanent: returning one to the free list would
            # hand a real request a row the next padding batch overwrites.
            if slot is not None and slot < self.num_request_slots:
                self._free.append(slot)


@dataclass
class InklingConvRuntime:
    """Per-forward short-conv plumbing for the pool path (all layers share it).

    Splits the packed ``[context tokens | one-token generation]`` batch at the
    context boundary so each short-conv seeds the pool for context requests and
    updates it in place for generation requests, like the paged attention split.
    """

    num_ctx_tokens: int
    ctx_indices: Optional[torch.Tensor]  # int32 pool slots, context requests
    gen_indices: Optional[torch.Tensor]  # int32 pool slots, generation requests
    query_start_loc: Optional[torch.Tensor]  # int32 [n_ctx+1] varlen offsets
    has_initial_state: Optional[torch.Tensor]  # bool [n_ctx]

    @classmethod
    def from_metadata(cls, attn_metadata) -> Optional["InklingConvRuntime"]:
        """Build this forward's split, or ``None`` for the stateless conv path.

        Called from ``model.forward``, so the pool rows come back as a view of the
        buffer ``prepare()`` already refreshed. The varlen offsets below do
        allocate, but only for a batch carrying context requests, and a captured
        batch never does -- decode graphs are generation-only.
        """
        mgr = getattr(attn_metadata, "kv_cache_manager", None)
        cache = getattr(mgr, "conv_state_cache", None)
        if cache is None or attn_metadata.request_ids is None:
            return None
        num_contexts = attn_metadata.num_contexts
        batch_size = len(attn_metadata.request_ids)
        state_indices = cache.state_indices
        device = state_indices.device
        query_start_loc = has_initial_state = None
        if num_contexts:
            seq_lens = attn_metadata.seq_lens.tolist()[:num_contexts]
            cu = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
            query_start_loc = cu
            # Fresh prefill carries no prior conv window -- correct only because
            # KV block reuse and chunked prefill are refused by
            # reject_unsupported_inkling_kv_cache_features. Deriving this from
            # num_cached_tokens_per_seq is not enough on its own: _run_context
            # attends only to its own call's tokens.
            has_initial_state = torch.zeros(num_contexts, dtype=torch.bool, device=device)
        return cls(
            num_ctx_tokens=sum(attn_metadata.seq_lens.tolist()[:num_contexts]),
            ctx_indices=state_indices[:num_contexts] if num_contexts else None,
            gen_indices=(
                state_indices[num_contexts:batch_size] if num_contexts < batch_size else None
            ),
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
    Otherwise the context slice seeds ``pool_buf`` and the generation slice
    updates it in place, then the two outputs are concatenated in packed order.
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
