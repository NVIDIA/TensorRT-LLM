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
"""Inkling text decoder for the TensorRT-LLM PyTorch backend.

The multimodal towers and their input processor live in
``modeling_inkling_multimodal.py``; see ``configs/inkling.py`` for the config
classes and ``checkpoints/hf/inkling_weight_mapper.py`` for the HF -> TRT weight
mapping. MTP is not implemented (its weights are accounted as unused).

Architecture summary:
  * RoPE-free attention with per-head q/k RMSNorm and score scale ``1/head_dim``.
  * Learned relative-position bias (``RelLogitsProj``), added pre-softmax as a
    ``score_mod`` inside the Inkling Triton attention kernels (prefill + paged
    decode); see ``attention_backend/inkling/``.
  * Hybrid layers: 55 local sliding-window (win=512, 16 kv-heads) + 11 global
    full-causal (8 kv-heads). Global layers apply log-scaling tau (a no-op below
    128k tokens, still implemented for correctness).
  * Four causal short convolutions per layer (k, v inside attention before the
    k/q norm; one post-attention and one post-MLP on the residual stream).
  * Sigmoid-gated MoE, top-6 of 256 routed experts with an additive selection
    bias, log-sigmoid renorm over the selected-routed *plus* two shared logits,
    scaled by ``route_scale * global_scale``. Layers 0/1 are dense MLP.
  * Routed experts for layers 3..65 are NVFP4; layer-2 experts and everything
    else are bf16.
  * muP: divide hidden states by ``logits_mup_width_multiplier`` before the head;
    slice logits to ``unpadded_vocab_size``. ``embed_norm`` folds onto embeddings.
"""

import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.inkling import (
    build_page_table,
    inkling_decode_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    filter_weights,
    register_auto_model,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.fused_moe import (
    BaseMoeRoutingMethod,
    RoutingMethodType,
    create_moe,
)
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

# Protocol only, to avoid an import cycle back through pyexecutor.
from tensorrt_llm._utils import prefer_pinned

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from ..configs.inkling import InklingConfig, InklingTextConfig
from .modeling_inkling_multimodal import (
    DEFAULT_AUDIO_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
    InklingAudioModel,
    InklingInputProcessor,
    InklingVisionModel,
)
from .modeling_multimodal_utils import (
    filter_mm_token_from_input_ids,
    find_input_mm_embeds,
    fuse_input_embeds,
)

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


def _apply_sconv(
    sconv: "InklingShortConv",
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


def _module_excluded_from_quant(model_config: ModelConfig, name: str) -> bool:
    """True if ``name`` (or an ancestor) is bf16, not NVFP4.

    This plain-NVFP4 checkpoint lists its bf16 modules in
    ``hf_quant_config.json`` ``quantization.exclude_modules`` (read into
    ``quant_config.exclude_modules`` by ``from_pretrained``) rather than in
    ``per_layer_quant_configs`` (only populated for MIXED_PRECISION checkpoints).
    ``QuantConfig.is_module_excluded_from_quantization`` walks the dotted
    ancestry, so a listed ``model.llm.layers.5.attn`` covers the qkv/o
    projections under it. Used to build attention (all ``.attn`` excluded) and
    layer-2 routed experts (``.mlp.experts`` excluded) as bf16.
    """
    qc = model_config.quant_config
    return (
        qc is not None
        and qc.exclude_modules is not None
        and qc.is_module_excluded_from_quantization(name)
    )


# ----------------------------------------------------------------------------
# Routing method
# ----------------------------------------------------------------------------
class InklingMoeRoutingMethod(BaseMoeRoutingMethod):
    """Sigmoid gate + additive-bias top-k selection + log-sigmoid renorm.

    The renorm denominator spans the selected routed logits *and* the shared
    logits together (``shared_expert_sink``), so this cannot be expressed by the
    stock sigmoid/MiniMax routing methods. ``apply`` returns only the routed
    ``(topk_ids, topk_weights)`` needed by the fused MoE; the shared gammas come
    from the same joint renorm and are recomputed in :class:`InklingMoE` for the
    shared-expert branch (see :func:`inkling_joint_renorm`).
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        n_shared_experts: int,
        callable_gate_bias,
        callable_global_scale,
        route_scale: float,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_shared_experts = n_shared_experts
        self._callable_gate_bias = callable_gate_bias
        self._callable_global_scale = callable_global_scale
        self.route_scale = route_scale

    def apply(
        self, router_logits: torch.Tensor, input_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # router_logits: [num_tokens, num_experts + n_shared] in fp32.
        routed_w, topk_idx, _ = inkling_joint_renorm(
            router_logits.float(),
            gate_bias=self._callable_gate_bias(),
            global_scale=self._callable_global_scale(),
            route_scale=self.route_scale,
            top_k=self.top_k,
            num_routed=self.num_experts,
            n_shared=self.n_shared_experts,
        )
        return topk_idx.to(torch.int32), routed_w.to(torch.float32)

    @property
    def routing_method_type(self):
        # CUTLASS computes the dispatch torch-side via :meth:`apply`, so the
        # kernel needs no routing enum of its own.
        return RoutingMethodType.Unspecified


def inkling_joint_renorm(
    router_logits: torch.Tensor,
    gate_bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    top_k: int,
    num_routed: int,
    n_shared: int,
):
    """Exact Inkling router math (fp32). Mirrors HF ``InklingTopkRouter``.

    Returns ``(routed_weights [T, top_k], topk_idx [T, top_k], shared_gammas
    [T, n_shared])``. Selection uses ``sigmoid(routed) + bias``; the weights are
    a softmax over ``logsigmoid`` of the selected-routed-plus-shared *logits*,
    scaled by ``route_scale * global_scale``.
    """
    routed_logits = router_logits[..., :num_routed]
    shared_logits = router_logits[..., num_routed : num_routed + n_shared]

    scores = routed_logits.sigmoid()
    scores_for_choice = scores + gate_bias
    topk_idx = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False)[1]

    topk_logits = torch.cat([routed_logits.gather(-1, topk_idx), shared_logits], dim=-1)
    topk_log_probs = torch.nn.functional.logsigmoid(topk_logits)
    weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))
    weights = weights * route_scale * global_scale

    routed_weights = weights[..., :top_k].contiguous()
    shared_gammas = weights[..., top_k : top_k + n_shared].contiguous()
    return routed_weights, topk_idx, shared_gammas


# ----------------------------------------------------------------------------
# Short convolution (four per layer)
# ----------------------------------------------------------------------------
class InklingShortConv(nn.Module):
    """Causal depthwise short convolution (kernel 4) with an internal residual.

    The weight matches the checkpoint layout ``[channels, 1, kernel]``. At
    prefill this runs :func:`causal_conv1d_fn`; at cached decode it runs
    :func:`causal_conv1d_update` against the per-request conv state carried by
    the state cache manager. ``conv_state`` (and the runtime metadata that
    selects the per-request slot) is threaded in by the caller; when it is
    ``None`` the module falls back to a self-contained causal convolution over
    the provided sequence.

    TP sharding (``tp_shard=True``): the k/v short convs act on the per-rank
    slice of the fused qkv projection, so their channels are sharded by kv-head
    like that projection and :meth:`load_weights` slices the rank's block out of
    the full checkpoint weight. The residual-stream convs are replicated.
    """

    def __init__(self, channels: int, kernel_size: int, mapping=None, tp_shard: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.tp_size = mapping.tp_size if (mapping is not None and tp_shard) else 1
        self.tp_rank = mapping.tp_rank if (mapping is not None and tp_shard) else 0
        assert channels % self.tp_size == 0, (channels, self.tp_size)
        self.channels_full = channels
        # Local (this rank's) channel count -- what the forward actually sees.
        self.channels = channels // self.tp_size
        # Depthwise conv weight, one filter per (local) channel: [channels,1,kernel].
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        self.register_parameter("bias", None)

    def load_weights(self, weights, allow_partial_loading: bool = False):
        """Copy the (full) checkpoint conv weight, slicing this rank's channels.

        Replicated convs run with ``tp_size == 1`` and copy the full tensor; the
        sharded k/v convs take the rank's contiguous, kv-head-aligned block.
        """
        w = weights[0]["weight"]
        if self.tp_size > 1:
            w = w.chunk(self.tp_size, dim=0)[self.tp_rank]
        self.weight.data.copy_(w[:])

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        cache_indices: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        """x: [num_tokens, channels]; internal residual ``y = conv(x) + x``.

        The stateless (no-cache) branch runs the conv in fp32 (per the source);
        the fused cached branches run in the input dtype (the ``causal_conv1d``
        ops require ``weight.dtype == x.dtype``, so the fp32 conv Parameter is
        cast to ``x.dtype`` and ``conv_state`` -- the bf16 state pool -- matches).
        Output is cast back to the input dtype. ``conv_state`` is updated in place
        by the fused ops.
        """
        in_dtype = x.dtype
        residual = x
        # Fused ops need weight and state in the input dtype (bf16); the fp32
        # conv Parameter is cast here (the stateless branch below uses fp32).
        w = self.weight.squeeze(1).to(x.dtype)  # [channels, kernel]
        if conv_state is not None and is_decode:
            # Cached decode. ``causal_conv1d_update`` writes in place into its
            # ``x`` argument, so pass a copy -- otherwise it clobbers
            # ``residual`` and the internal residual becomes conv(x) + conv(x).
            y = causal_conv1d_update(
                x.clone(),
                conv_state,
                w,
                self.bias,
                activation=None,
                conv_state_indices=cache_indices,
            )
        elif conv_state is not None:
            # Prefill with cache: varlen [channels, total_tokens].
            xt = x.transpose(0, 1).contiguous()
            y = causal_conv1d_fn(
                xt,
                w,
                self.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                conv_states=conv_state,
                activation=None,
            )
            y = y.transpose(0, 1).contiguous()
        else:
            # No cache: self-contained causal depthwise conv over the sequence.
            xt = x.float().transpose(0, 1).unsqueeze(0)  # [1, channels, T]
            y = torch.nn.functional.conv1d(
                xt,
                self.weight.float(),
                bias=None,
                padding=self.kernel_size - 1,
                groups=self.channels,
            )
            y = y[..., : x.shape[0]].squeeze(0).transpose(0, 1)
        return (y.to(in_dtype) + residual).to(in_dtype)


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------
class InklingAttention(QKNormRoPEAttention):
    """RoPE-free attention with per-head q/k RMSNorm, k/v short-conv, and a
    learned relative-position bias applied as a Triton ``score_mod``.

    Reuses :class:`QKNormRoPEAttention` for the fused qkv/o projections and
    per-head q/k RMSNorm (``skip_rope=True`` gives qk-norm without RoPE), and
    owns the extra ``r`` projection, the k/v short convolutions, and the
    relative-logit projection. The attention *compute* runs through the Inkling
    Triton path (``attention_backend/inkling/``) rather than the base backend:
    Inkling's learned relative bias is a per-(query, head, relative-distance)
    additive ``score_mod``, which no fused, CUDA-graph-safe TensorRT-LLM backend
    exposes. The bias is precomputed torch-side into a ``rel_logits``
    ``[num_query_tokens, local_heads, rel_extent]`` tensor and gathered+added by
    the Triton kernels. Local layers apply the sliding window natively in the
    kernel; global layers fold the log-scaling ``tau`` into ``rel_logits``.

    KV read/write goes through ``KVCacheManagerV2`` in the HND paged layout.
    ``self.attn`` (the base backend) is built but unused -- only its
    runtime-assigned ``local_layer_idx`` is read here.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig], layer_idx: int):
        config = model_config.pretrained_config
        self.is_local = config.is_local_layer(layer_idx)
        head_dim = config.layer_head_dim(layer_idx)
        num_heads = config.layer_num_heads(layer_idx)
        num_kv_heads = config.layer_num_kv_heads(layer_idx)
        self.attention_window_size = config.layer_window(layer_idx)
        self.d_rel = config.d_rel
        self.rel_extent = config.sliding_window_size if self.is_local else config.rel_extent
        self.log_scaling_n_floor = None if self.is_local else config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha

        # Attention is bf16, not NVFP4 -- the checkpoint excludes
        # ``model.llm.layers.{i}.attn``. Hand the base a shallow ModelConfig copy
        # with an empty quant_config so qkv_proj/o_proj are built unquantized.
        # (``ModelConfig.__setattr__`` whitelists ``quant_config`` for exactly
        # this per-module override.)
        attn_model_config = model_config
        if _module_excluded_from_quant(model_config, f"model.llm.layers.{layer_idx}.attn"):
            from tensorrt_llm.models.modeling_utils import QuantConfig

            attn_model_config = copy.copy(model_config)
            attn_model_config.quant_config = QuantConfig()

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            # No RoPE: forward runs the Inkling Triton attention directly, so
            # this keeps the base from building an unused RotaryEmbedding.
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=attn_model_config,
            # q/k are per-head RMS-normalized, so the score scale is 1/head_dim.
            # The backend uses 1/(sqrt(head_dim) * q_scaling).
            q_scaling=float(head_dim) ** 0.5,
            skip_rope=True,
            fuse_qk_norm_rope=False,
            is_qk_norm=True,
        )
        # head_dim differs from hidden_size // num_heads, so the base Attention
        # must have picked it up from config.head_dim (there is no kwarg for it).
        assert self.head_dim == head_dim, (self.head_dim, head_dim)

        # Applied directly by the Triton kernels. The sliding window uses an
        # inclusive radius: query p attends to keys [p - (w - 1), p].
        self.sm_scale = 1.0 / float(head_dim)
        self.window_left = (self.attention_window_size - 1) if self.is_local else -1

        # Attention-scoped TP. Under attention DP every rank runs the full head
        # set over its own requests, so the base built qkv_proj / o_proj with
        # tp_size=1. The Inkling-only tensors below (r_proj, k/v sconv) hang off
        # the same head/kv-head split and must follow the attention TP, not the
        # global one -- otherwise they silently mismatch qkv_proj's shape.
        tp_size = 1 if model_config.mapping.enable_attention_dp else model_config.mapping.tp_size
        # Cross-check against the base so a change to how modules/attention.py
        # scopes attention TP fails here at load rather than silently.
        assert self.num_heads == num_heads // tp_size, (
            f"attention TP disagrees with the base Attention: base kept "
            f"{self.num_heads} of {num_heads} heads, this rule expects "
            f"{num_heads // tp_size} (enable_attention_dp="
            f"{model_config.mapping.enable_attention_dp}, "
            f"mapping.tp_size={model_config.mapping.tp_size})"
        )
        # r projection: per-head relative states (num_heads * d_rel), sharded by
        # head like q. Output is not gathered (consumed locally to build bias).
        # Under attention DP it is replicated: no mapping / no TP mode, matching
        # how DeepSeek-V3 builds its non-expert Linears under ADP.
        self.r_proj = Linear(
            config.hidden_size,
            num_heads * self.d_rel,
            bias=False,
            dtype=config.torch_dtype,
            mapping=None if model_config.mapping.enable_attention_dp else model_config.mapping,
            tensor_parallel_mode=(
                None if model_config.mapping.enable_attention_dp else TensorParallelMode.COLUMN
            ),
            gather_output=False,
        )
        # Learned relative-logit profiles, replicated across TP ranks. The profile
        # length is per-layer (local layers store only the sliding-window extent),
        # so this must use ``self.rel_extent``, not ``config.rel_extent``.
        self.rel_logits_proj = nn.Parameter(torch.empty(self.d_rel, self.rel_extent))
        # k/v short convs act on the k/v stream of the fused qkv projection, so
        # they are sharded by kv-head like it. Pass the full channel count and
        # let InklingShortConv slice this rank's block at load.
        full_kv_dim = num_kv_heads * head_dim
        sconv_tp_shard = not model_config.mapping.enable_attention_dp
        self.k_sconv = InklingShortConv(
            full_kv_dim,
            config.sconv_kernel_size,
            mapping=model_config.mapping,
            tp_shard=sconv_tp_shard,
        )
        self.v_sconv = InklingShortConv(
            full_kv_dim,
            config.sconv_kernel_size,
            mapping=model_config.mapping,
            tp_shard=sconv_tp_shard,
        )
        self.local_num_heads = num_heads // tp_size

    def _project(self, hidden_states, conv_pool_kv=None, conv_rt=None):
        """Fused qkv projection -> split -> k/v short-conv -> per-head qk RMSNorm.

        Returns ``(q, k, v)`` shaped ``[T, local_heads, head_dim]`` /
        ``[T, local_kv_heads, head_dim]``. With ``conv_pool_kv=(pool_k, pool_v)``
        + ``conv_rt`` the k/v short-convs run through the runtime state pool
        (seed for context tokens, in-place update at the per-request slots for
        generation tokens; fused ops, CUDA-graph safe, mixed-batch capable);
        without them they run the stateless full-sequence causal conv.
        """
        D = self.head_dim
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv, None, None)
        # k/v short convolution before the q/k norm (source order).
        if conv_pool_kv is not None:
            pool_k, pool_v = conv_pool_kv
            k = _apply_sconv(self.k_sconv, k, pool_k, conv_rt)
            v = _apply_sconv(self.v_sconv, v, pool_v, conv_rt)
        else:
            k = self.k_sconv(k)
            v = self.v_sconv(v)
        q, k = self.apply_qk_norm(q, k)
        nh = self.q_size // D
        nkv = self.kv_size // D
        return (
            q.view(num_tokens, nh, D),
            k.view(num_tokens, nkv, D),
            v.view(num_tokens, nkv, D),
        )

    def _build_rel_logits(
        self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Contiguous relative-bias aux tensor ``[T, local_heads, rel_extent]``.

        ``rel_logits[t, h, e] = sum_d r[t, h, d] * proj[d, e]`` (fp32), mirroring
        the reference ``InklingRelativeLogits``. Global layers fold in the
        per-query-token log-scaling ``tau``. The Triton kernels index this by
        ``clamp(q_pos - k_pos, 0, rel_extent - 1)`` and zero it outside the range.
        """
        r = self.r_proj(hidden_states).view(-1, self.local_num_heads, self.d_rel)
        rel = torch.einsum(
            "thd,de->the", r.float(), self.rel_logits_proj.float()
        )  # [T, H, rel_extent]
        if self.log_scaling_n_floor is not None and position_ids is not None:
            pos = position_ids.reshape(-1).float()
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                ((pos + 1.0) / self.log_scaling_n_floor).clamp(min=1.0)
            )
            rel = rel * tau[:, None, None]
        return rel.contiguous()

    def _attention(self, q, k, v, rel_logits, attn_metadata, *, allow_mixed=False):
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
        # ``layer_offsets`` itself. ``self.attn.local_layer_idx`` is only primed
        # inside the base attention forward, which Inkling bypasses.
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

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        conv_pool_kv=None,
        conv_rt=None,
        **kwargs,
    ):
        """Inkling attention through the Triton score_mod path.

        ``conv_pool_kv=(pool_k, pool_v)`` + ``conv_rt`` drive the k/v short-convs
        through the runtime state pool (seed on context, in-place update at the
        per-request slots on generation, mixed-batch capable, CUDA-graph safe);
        without them the short-convs run stateless over the whole sequence.
        """
        num_tokens = hidden_states.shape[0]
        # The pre-attention RMSNorm can emit fp32 while the attention/r
        # projections are bf16, so cast once here.
        hidden_states = hidden_states.to(self.qkv_proj.weight.dtype)
        q, k, v = self._project(hidden_states, conv_pool_kv, conv_rt)
        rel_logits = self._build_rel_logits(hidden_states, position_ids)
        attn_out = self._attention(
            q, k, v, rel_logits, attn_metadata, allow_mixed=conv_rt is not None
        )
        attn_out = attn_out.reshape(num_tokens, self.q_size)
        return self.o_proj(attn_out)


# ----------------------------------------------------------------------------
# Dense MLP (layers 0, 1) and MoE (layers 2..65)
# ----------------------------------------------------------------------------
class InklingDenseMLP(nn.Module):
    """SwiGLU MLP with a learned scalar ``global_scale`` (layers 0, 1).

    Fused gate+up (``w13_dn``) column-parallel, down (``w2_md``) row-parallel.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        super().__init__()
        config = model_config.pretrained_config
        inter = config.dense_intermediate_size
        # Under attention DP the dense MLP goes data-parallel too: each rank
        # holds the full weight and runs it over its own tokens. Keeping the
        # column/row split would be a correctness bug -- the row-parallel
        # down_proj would all-reduce partials belonging to different requests.
        # Mirrors DeepSeek-V3's ``_compute_mlp_tp_size``.
        dp = model_config.mapping.enable_attention_dp
        mlp_mapping = None if dp else model_config.mapping
        self.gate_up_proj = Linear(
            config.hidden_size,
            2 * inter,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mlp_mapping,
            tensor_parallel_mode=None if dp else TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
            ),
        )
        self.down_proj = Linear(
            inter,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mlp_mapping,
            tensor_parallel_mode=None if dp else TensorParallelMode.ROW,
        )
        self.global_scale = nn.Parameter(torch.ones(1))
        self.act_fn = torch.nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        # ``global_scale`` is fp32 and promotes the output; cast back so the
        # residual stream stays in the input dtype.
        out = self.down_proj(self.act_fn(gate) * up) * self.global_scale
        return out.to(x.dtype)


class InklingGate(nn.Module):
    """fp32 router: logits over 256 routed + 2 shared experts, plus the additive
    selection bias and the learned global scale. Feeds
    :class:`InklingMoeRoutingMethod`.
    """

    def __init__(self, config: InklingTextConfig):
        super().__init__()
        self.num_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        n_total = self.num_routed + self.n_shared
        self.weight = nn.Parameter(torch.empty(n_total, config.hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(self.num_routed, dtype=torch.float32))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.float(), self.weight)

    @property
    def routing_method(self) -> InklingMoeRoutingMethod:
        return InklingMoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_routed,
            n_shared_experts=self.n_shared,
            callable_gate_bias=lambda: self.bias,
            callable_global_scale=lambda: self.global_scale,
            route_scale=self.route_scale,
        )


class InklingSharedExperts(nn.Module):
    """Two shared SwiGLU experts, each weighted by a per-token gamma and summed.

    Reference: HF ``InklingSharedExperts`` (batched 2-expert SwiGLU, fp32 sum).
    """

    def __init__(self, config: InklingTextConfig):
        super().__init__()
        self.n_shared = config.n_shared_experts
        inter = config.intermediate_size
        hidden = config.hidden_size
        # [n_shared, 2*inter, hidden] fused gate+up; [n_shared, hidden, inter] down.
        # Created in the model dtype: these run as raw bmms against the bf16
        # hidden stream, and the checkpoint stores them bf16 (not quantized).
        self.shared_w13 = nn.Parameter(
            torch.empty(self.n_shared, 2 * inter, hidden, dtype=config.torch_dtype)
        )
        self.shared_w2 = nn.Parameter(
            torch.empty(self.n_shared, hidden, inter, dtype=config.torch_dtype)
        )
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        # hidden_states: [T, hidden]; gammas: [T, n_shared] fp32. Both bmms stay
        # in the activation dtype and the gamma is applied in fp32 after the
        # (linear) down projection, where it commutes.
        x = hidden_states.unsqueeze(0).expand(self.n_shared, -1, -1)
        gate_up = torch.bmm(x, self.shared_w13.transpose(1, 2))
        # ``shared_w13`` loads raw, with gate/up interleaved along its 2*inter
        # output dim, so gate = even channels and up = odd. A contiguous
        # chunk(2) here would pair the wrong channels.
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        activated = self.act_fn(gate) * up
        out = torch.bmm(activated, self.shared_w2.transpose(1, 2))  # [S, T, hidden]
        out = out.float() * gammas.transpose(0, 1).unsqueeze(-1).float()
        return out.sum(dim=0).to(hidden_states.dtype)


class InklingMoE(nn.Module):
    """Router + routed experts (fused MoE) + two shared experts.

    Routed experts run through :func:`create_moe` (NVFP4 for layers 3..65, bf16
    for layer 2 via a per-layer quant override). Shared experts and the router
    stay bf16/fp32. The routed output already reduces over the top-6 experts; the
    gamma-weighted shared output is added on top (source ``h + shared``).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.gate = InklingGate(config)
        self.num_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale

        experts_quant_config = self._experts_quant_config(model_config, layer_idx)
        # reduce_results=True: under TP each rank holds a shard of the experts and
        # produces only a partial routed sum, so it must be all-reduced before the
        # replicated shared-expert output is added on top.
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_routed,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=True,
            model_config=model_config,
            override_quant_config=experts_quant_config,
            layer_idx=layer_idx,
        )
        self.shared_experts = InklingSharedExperts(config)

    @staticmethod
    def _experts_quant_config(model_config: ModelConfig, layer_idx: int):
        """Per-layer expert quant: NVFP4 unless the checkpoint excludes it.

        The checkpoint lists its bf16 modules in ``hf_quant_config.json``
        ``quantization.exclude_modules``, which is the authoritative per-layer
        signal here (``per_layer_quant_configs`` is only populated for
        MIXED_PRECISION checkpoints). Excluded expert modules get an empty
        ``QuantConfig`` so ``create_moe`` builds an unquantized bf16 MoE.
        """
        if _module_excluded_from_quant(model_config, f"model.llm.layers.{layer_idx}.mlp.experts"):
            from tensorrt_llm.models.modeling_utils import QuantConfig

            return QuantConfig()
        return model_config.quant_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Routed + shared experts.

        ``all_rank_num_tokens`` is the per-rank token count this step, taken from
        ``attn_metadata``; ``FusedMoE`` needs it to pad and gather across ranks
        under DP. ``None`` is the non-DP case.
        """
        router_logits = self.gate(hidden_states)  # [T, 258] fp32
        routed = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )
        _, _, shared_gammas = inkling_joint_renorm(
            router_logits,
            gate_bias=self.gate.bias,
            global_scale=self.gate.global_scale,
            route_scale=self.route_scale,
            top_k=self.top_k,
            num_routed=self.num_routed,
            n_shared=self.n_shared,
        )
        shared = self.shared_experts(hidden_states, shared_gammas)
        # fp32 scales in the routed/shared paths can promote the sum; keep the
        # residual-stream dtype.
        return (routed + shared).to(hidden_states.dtype)


# ----------------------------------------------------------------------------
# Decoder layer / model / causal LM
# ----------------------------------------------------------------------------
class InklingDecoderLayer(nn.Module):
    """Pre-norm attention + MLP, each followed by a short-conv with an internal
    residual, then the residual add (HF ``InklingDecoderLayer`` order).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.attn = InklingAttention(model_config, layer_idx)
        self.attn_sconv = InklingShortConv(config.hidden_size, config.sconv_kernel_size)
        self.mlp_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        if config.is_dense_layer(layer_idx):
            self.mlp = InklingDenseMLP(model_config)
        else:
            self.mlp = InklingMoE(model_config, layer_idx)
        self.mlp_sconv = InklingShortConv(config.hidden_size, config.sconv_kernel_size)

    def _run_mlp(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]],
    ) -> torch.Tensor:
        """Dense layers 0/1 take only the activations; MoE layers also take the
        per-rank token counts the fused kernel needs to gather across ranks."""
        if isinstance(self.mlp, InklingMoE):
            return self.mlp(hidden_states, all_rank_num_tokens=all_rank_num_tokens)
        return self.mlp(hidden_states)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        conv_state: Optional[InklingConvState] = None,
        conv_rt: Optional[InklingConvRuntime] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pre-norm attention + MLP, each followed by a short-conv (internal
        residual), then the residual add.

        With ``conv_rt`` given, ``conv_state`` holds this layer's four
        ``[max_batch, C, K-1]`` pool buffers
        (:meth:`InklingConvStateCache.layer_state`) and each short-conv seeds the
        pool for context tokens and updates it in place at the per-request slots
        for generation tokens (fused ops, mixed-batch + CUDA-graph safe).
        Without it the short-convs run stateless over the whole sequence.
        """
        if conv_rt is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attn(position_ids, hidden_states, attn_metadata)
            hidden_states = self.attn_sconv(hidden_states)  # internal residual
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
            hidden_states = self._run_mlp(hidden_states, all_rank_num_tokens)
            hidden_states = self.mlp_sconv(hidden_states)  # internal residual
            return residual + hidden_states

        # --- Runtime state-pool path (prefill-seed / decode / mixed). ---
        residual = hidden_states
        h = self.attn_norm(hidden_states)
        h = self.attn(
            position_ids,
            h,
            attn_metadata,
            conv_pool_kv=(conv_state.k, conv_state.v),
            conv_rt=conv_rt,
            **kwargs,
        )
        h = residual + _apply_sconv(self.attn_sconv, h, conv_state.attn, conv_rt)

        residual = h
        hm = self._run_mlp(self.mlp_norm(h), all_rank_num_tokens)
        return residual + _apply_sconv(self.mlp_sconv, hm, conv_state.mlp, conv_rt)


class InklingModel(DecoderModel):
    """The Inkling text decoder stack. ``embed_norm`` folds onto the token
    embeddings before the layers (``use_embed_norm``)."""

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.embed_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.layers = nn.ModuleList(
            [InklingDecoderLayer(model_config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_embeds_prenormed: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Decoder stack. The runtime short-conv state pool and the
        context/generation split come from ``attn_metadata``; each layer reads
        its own four state buffers. A metadata without them (no conv-capable
        cache manager) keeps the stateless conv.

        ``inputs_embeds_prenormed``: on the multimodal path the wrapper has
        already applied ``embed_norm`` to the text embeddings and scattered the
        raw tower rows in afterwards, so the fused stream must not be re-normed
        here. Text-only callers pass raw ``inputs_embeds`` and keep the norm."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        conv_cache = getattr(attn_metadata, "ink_conv_cache", None)
        conv_rt = getattr(attn_metadata, "ink_conv_rt", None)
        # Per-rank token counts for this step, set on attn_metadata only under
        # attention DP; FusedMoE reads it to pad and gather across ranks.
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        hidden_states = inputs_embeds if inputs_embeds_prenormed else self.embed_norm(inputs_embeds)
        for i, layer in enumerate(self.layers):
            layer_state = conv_cache.layer_state(i) if conv_cache is not None else None
            hidden_states = layer(
                position_ids,
                hidden_states,
                attn_metadata,
                conv_state=layer_state,
                conv_rt=conv_rt,
                all_rank_num_tokens=all_rank_num_tokens,
            )
        return self.norm(hidden_states)


class InklingForCausalLM(DecoderModelForCausalLM[InklingModel, InklingTextConfig]):
    """Text CausalLM: muP logit scaling + unpadded-vocab slice.

    ``embed`` and ``unembed`` are separate checkpoint tensors (never tied). The
    ``LMHead`` is built at the unpadded vocab size so its forward slices off the
    padding automatically; hidden states are divided by
    ``logits_mup_width_multiplier`` before the head (accuracy-critical).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        config = model_config.pretrained_config
        self.mup_multiplier = float(config.logits_mup_width_multiplier)
        super().__init__(
            InklingModel(model_config),
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.unpadded_vocab_size,
        )
        self._assert_inkling_attn_backend(model_config)
        self._assert_inkling_moe_parallel(model_config)
        self._apply_allreduce_strategy()

    @staticmethod
    def _assert_inkling_attn_backend(model_config) -> None:
        """Fail at load if the Inkling attention backend was overridden.

        ``get_model_defaults`` selects ``attn_backend='INKLING'`` because
        ``InklingAttentionMetadata`` is what publishes the decode seq_lens and
        page table into CUDA-graph-stable buffers. Model defaults are a
        deep-merge in which an explicit user value wins, so an
        ``attn_backend: TRTLLM`` left in ``--extra_llm_api_options`` -- a very
        easy thing to carry over from another model's serve config -- silently
        removes that publish. The run then dies deep in CUDA-graph capture with
        "Cannot copy between CPU and CUDA tensors", which names neither Inkling
        nor the setting responsible.
        """
        backend = getattr(model_config, "attn_backend", None)
        if backend is not None and str(backend).upper() != "INKLING":
            raise ValueError(
                f"Inkling requires attn_backend='INKLING' (got {backend!r}). "
                "The Triton decode kernel reads its per-step seq_lens and page "
                "table from InklingAttentionMetadata, which only the INKLING "
                "backend supplies. Remove the attn_backend override from "
                "--extra_llm_api_options / LLM(attn_backend=...) so the model "
                "default applies."
            )

    @staticmethod
    def _assert_inkling_moe_parallel(model_config) -> None:
        """Reject an expert-parallel layout the MoE backend cannot serve.

        Inkling's routed experts go through the generic ``create_moe`` factory,
        so expert parallelism needs no Inkling-specific code: ``Mapping``
        derives ``moe_ep_size`` / ``moe_tp_size``, ``FusedMoE`` slices the 256
        experts with ``_compute_ep_partition``, and CutlassFusedMoE remaps the
        NVFP4 per-expert scales onto the local slice. What it does NOT have is
        a check that the requested split is one the backend supports.

        ``FusedMoE._supports_non_divisible_ep`` is opt-in and the CUTLASS
        backend -- the only routed-expert backend Inkling ships -- does not opt
        in, so a non-divisible ``moe_expert_parallel_size`` fails somewhere
        inside expert-slot bookkeeping rather than at load. 256 divides evenly
        by every power of two, so this only bites on values like 3, 5 or 6.

        Note this deliberately does NOT constrain moe_tp_size: with
        ``moe_ep_size = 1`` (the default) the experts are TP-sharded, which is
        what every Inkling accuracy run to date measured.
        """
        mapping = getattr(model_config, "mapping", None)
        if mapping is None:
            return
        ep_size = getattr(mapping, "moe_ep_size", 1) or 1
        if ep_size <= 1:
            return
        config = model_config.pretrained_config
        config = getattr(config, "text_config", config)
        num_experts = getattr(config, "n_routed_experts", None)
        if num_experts is None:
            return
        # Report "more ranks than experts" first: it subsumes non-divisibility,
        # and "pick a divisor" is not useful advice in that case.
        if num_experts < ep_size:
            raise ValueError(
                f"moe_expert_parallel_size={ep_size} exceeds Inkling's "
                f"{num_experts} routed experts; ranks with zero experts are "
                f"not supported by any MoE backend."
            )
        if num_experts % ep_size != 0:
            raise ValueError(
                f"Inkling has {num_experts} routed experts, which "
                f"moe_expert_parallel_size={ep_size} does not divide evenly. "
                f"The CUTLASS MoE backend does not opt into non-divisible "
                f"expert parallelism, so the uneven split would fail inside "
                f"expert-slot bookkeeping instead of here. Pick a divisor of "
                f"{num_experts}."
            )
        # Pure expert parallelism (moe_tp_size 1) segfaults during CUDA-graph
        # capture; the same layout is correct with graphs disabled. Root cause
        # not yet found, so reject only that combination rather than the whole
        # layout, and point at the configuration that works.
        moe_tp_size = getattr(mapping, "moe_tp_size", None)
        use_cuda_graph = getattr(model_config, "use_cuda_graph", False)
        if moe_tp_size is not None and moe_tp_size < 2 and use_cuda_graph:
            raise ValueError(
                f"moe_expert_parallel_size={ep_size} leaves moe_tp_size="
                f"{moe_tp_size}, which segfaults during CUDA-graph capture "
                f"for Inkling. The same layout runs correctly with CUDA "
                f"graphs disabled (cuda_graph_config=None), reproducing the "
                f"TP-only accuracy per item. Either disable CUDA graphs or "
                f"use moe_expert_parallel_size <= {max(1, ep_size // 2)}."
            )

    def _apply_allreduce_strategy(self) -> None:
        """Keep Inkling's all-reduces off the NCCL_SYMMETRIC tactic.

        Under CUDA-graph capture a symmetric all-reduce corrupts the run when its
        send buffer is unregistered while its recv buffer is a registered NCCL
        window; Inkling's decode message size hits that case exactly and decode
        collapses to a repeated token.

        The affected all-reduces are built by generic modules (attention
        ``o_proj``, MoE ``down_proj``), so rebuilding them here keeps the
        mitigation model-local. Pinning ONESHOT also drops the NCCL window
        requirement. The cost is giving up the symmetric tactic everywhere,
        eager included.
        """
        for mod in self.modules():
            old = getattr(mod, "all_reduce", None)
            # ``None`` means the module reduces nothing; adding an AllReduce here
            # would add a collective rather than remove a tactic.
            if not isinstance(old, AllReduce):
                continue
            # Carry the module's own mapping and dtype over so the rebuild
            # differs from the original in strategy alone.
            mod.all_reduce = AllReduce(
                mapping=old.mapping,
                strategy=AllReduceStrategy.ONESHOT,
                dtype=getattr(mod, "dtype", None),
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        inputs_embeds_prenormed: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # The short-conv state pool is owned by InklingHybridCacheManager and
        # reaches the decoder through attn_metadata, published outside the
        # captured region -- no conv kwargs and no ResourceManager lookup here.
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            inputs_embeds_prenormed=inputs_embeds_prenormed,
        )
        hidden_states = hidden_states / self.mup_multiplier
        return self.logits_processor.forward(
            hidden_states, self.lm_head, attn_metadata, return_context_logits
        )


def _encode_inkling_image_embeds(
    visual: InklingVisionModel, multimodal_params: list
) -> List[torch.Tensor]:
    """Run the hMLP vision tower over the context requests' patch features.

    Reads ``multimodal_data['image']['vision_patches_bthwc']`` (the tensor the
    :class:`InklingInputProcessor` attaches) from each context
    ``MultimodalParams``, concatenates them, and runs the tower on the tower's
    device/dtype. Returns a single-element list ``[feats]`` with ``feats`` of
    shape ``(sum_patches, decoder_dmodel)`` -- the same shape
    ``get_multimodal_embeddings`` returns and ``find_input_mm_embeds`` slices --
    or ``[]`` when no context request carries image features."""
    patches = []
    for param in multimodal_params:
        data = getattr(param, "multimodal_data", None) or {}
        image = data.get("image") or {}
        vp = image.get("vision_patches_bthwc")
        if vp is not None:
            patches.append(vp)
    if not patches:
        return []
    p = next(visual.parameters())
    x = torch.cat([vp.to(device=p.device, dtype=p.dtype) for vp in patches], dim=0)
    return [visual(x)]


def _encode_inkling_audio_embeds(
    audio_tower: InklingAudioModel, multimodal_params: list
) -> List[torch.Tensor]:
    """Run the dMel audio tower over the context requests' audio features.

    Reads ``multimodal_data['audio']['dmel_bins']`` (the tensor the
    :class:`InklingInputProcessor` attaches) from each context
    ``MultimodalParams``, concatenates them, and runs the tower. Returns a
    single-element list ``[feats]`` with ``feats`` of shape
    ``(sum_frames, decoder_dmodel)`` -- the same shape contract as the image
    encoder -- or ``[]`` when no context request carries audio features."""
    frames = []
    for param in multimodal_params:
        data = getattr(param, "multimodal_data", None) or {}
        audio = data.get("audio") or {}
        db = audio.get("dmel_bins")
        if db is not None:
            frames.append(db)
    if not frames:
        return []
    dev = audio_tower.encoder.weight.device
    # dMel bins are integer codebook indices; keep them integral (the tower casts
    # to long internally), only moving them onto the tower's device.
    x = torch.cat([f.to(device=dev) for f in frames], dim=0)
    return [audio_tower(x)]


@register_auto_model("InklingForConditionalGeneration")
@register_input_processor(
    InklingInputProcessor,
    model_type="inkling_mm_model",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        # Image and audio are distinct modalities; video has no separate token or
        # tower -- its frames are served as ordinary ``<image>`` placeholders.
        placeholder_map={"image": "<image>", "audio": "<audio>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.OPENAI,
    ),
)
class InklingForConditionalGeneration(InklingForCausalLM):
    """Registered entry point for the multimodal ``inkling_mm_model`` checkpoint.

    Text-only requests route straight to the text :class:`InklingForCausalLM`
    over the ``text_config`` sub-config and consume only ``model.llm.*`` weights.
    Image requests are preprocessed by the registered
    :class:`InklingInputProcessor`, which expands the ``<image>`` placeholder to
    one token per vision patch and attaches the ``vision_patches_bthwc``
    features; the hMLP vision tower (:class:`InklingVisionModel`) is built as a
    replicated bf16 submodule ``self.visual`` and its per-patch outputs are fused
    into ``inputs_embeds`` at the placeholder positions before the text decoder
    (OOV-safe ``fuse_input_embeds``). Audio requests flow through the identical
    fusion path: the processor expands the ``<audio>`` placeholder to one token
    per dMel frame and the dMel audio tower (:class:`InklingAudioModel`,
    replicated bf16 ``self.audio_tower``) emits one row per frame. Video is
    multi-frame images through the vision tower. MTP is not implemented. See
    ``checkpoints/hf/inkling_weight_mapper.py`` for the HF -> TRT name mapping
    and consumed/deferred accounting.
    """

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        # ``use_kv_cache_manager_v2``: Inkling's per-layer KV-head split (local
        # layers carry more KV heads than global ones) needs V2's per-layer
        # geometry; V1's unified pool would mis-size the per-layer KV bytes.
        # ``_util`` already forces the V2 class for Inkling -- declaring it here
        # keeps the resolved flag (and its readers) agreeing with that.
        #
        # ``enable_block_reuse``: the short-conv window is per-request state
        # outside the KV cache and every context request is seeded empty, so a
        # reused prefix would convolve against padding -- wrong outputs, not just
        # a cache miss. ``MixedMambaHybridCacheManager`` has the same limitation.
        # An explicit user setting still wins the deep-merge.
        #
        # ``attn_backend``: ``InklingAttentionMetadata`` publishes the decode
        # seq_lens and page table into fixed-pointer GPU buffers before
        # CUDA-graph capture. ``InklingTritonAttention`` changes nothing but the
        # metadata class -- the compute lives in ``InklingAttention.forward``.
        return {
            "attn_backend": "INKLING",
            "kv_cache_config": {
                "use_kv_cache_manager_v2": True,
                "enable_block_reuse": False,
            },
        }

    def __init__(self, model_config: ModelConfig[InklingConfig]):
        text_model_config = _text_sub_model_config(model_config)
        super().__init__(text_model_config)
        self._top_model_config = model_config
        # The hMLP vision tower is a replicated bf16 submodule: excluded from
        # NVFP4 and not TP-sharded, since every rank runs the identical tower
        # over identical patches. ``None`` for a text-only checkpoint.
        vision_config = getattr(model_config.pretrained_config, "vision_config", None)
        if vision_config is not None and getattr(vision_config, "decoder_dmodel", None):
            self.visual = InklingVisionModel(vision_config).to(torch.bfloat16)
        else:
            self.visual = None
        # The dMel audio tower follows the same rules as the vision tower, with
        # one row per dMel frame. ``None`` when the config has no ``audio_config``.
        audio_config = getattr(model_config.pretrained_config, "audio_config", None)
        if audio_config is not None and getattr(audio_config, "decoder_dmodel", None):
            self.audio_tower = InklingAudioModel(audio_config).to(torch.bfloat16)
        else:
            self.audio_tower = None
        # The media placeholder ids the Inkling chat template emits. They must be
        # in-vocab, since the executor rejects out-of-range request token ids.
        # Surfaced to the model engine's ``_prepare_multimodal_indices`` so it can
        # locate the media rows; the audio id is registered only when the audio
        # tower exists, leaving vision/text-only checkpoints unchanged.
        self.image_token_id = int(
            getattr(model_config.pretrained_config, "image_token_id", DEFAULT_IMAGE_TOKEN_ID)
        )
        self.audio_token_id = int(
            getattr(model_config.pretrained_config, "audio_token_id", DEFAULT_AUDIO_TOKEN_ID)
        )
        mm_ids = [self.image_token_id]
        if self.audio_tower is not None:
            mm_ids.append(self.audio_token_id)
        self._mm_token_ids = torch.tensor(mm_ids, dtype=torch.int32)

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    def _resolve_mm_indices(self, input_ids, kwargs):
        """Executor-precomputed text/mm indices if shipped, else compute them
        from the placeholder ids via ``isin`` (a host sync; eager/warmup only)."""
        ti = kwargs.get("text_token_indices")
        mi = kwargs.get("mm_token_indices")
        if ti is not None and mi is not None:
            return ti, mi
        return filter_mm_token_from_input_ids(
            input_ids,
            vocab_size=self.model.embed_tokens.num_embeddings,
            mm_token_ids=self._mm_token_ids.to(input_ids.device),
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Fuse media embeddings into the text embedding stream, then run the
        text decoder. Only context (prefill) requests carry media features;
        decode steps have ``num_contexts == 0`` and pass straight through. A
        text-only request never touches the towers."""
        inputs_embeds_prenormed = False
        if inputs_embeds is None and (self.visual is not None or self.audio_tower is not None):
            multimodal_params = kwargs.get("multimodal_params", []) or []
            num_ctx = attn_metadata.num_contexts
            ctx_params = multimodal_params[:num_ctx]
            if ctx_params:
                # Keep the replicated tower(s) on the decoder's device (one-time
                # move; the full-model loader may leave them on CPU).
                dev = self.model.embed_tokens.weight.device
                if self.visual is not None and next(self.visual.parameters()).device != dev:
                    self.visual = self.visual.to(dev)
                if (
                    self.audio_tower is not None
                    and next(self.audio_tower.parameters()).device != dev
                ):
                    self.audio_tower = self.audio_tower.to(dev)
                # Per-modality tower rows, read from the attached media data.
                vis_raw = (
                    _encode_inkling_image_embeds(self.visual, ctx_params)
                    if self.visual is not None
                    else []
                )
                aud_raw = (
                    _encode_inkling_audio_embeds(self.audio_tower, ctx_params)
                    if self.audio_tower is not None
                    else []
                )
                if vis_raw and not aud_raw:
                    # Vision-only fusion: the executor's precomputed indices
                    # already select exactly the image rows.
                    mm_embeds = find_input_mm_embeds(vis_raw, ctx_params)
                    if mm_embeds:
                        ti, mi = self._resolve_mm_indices(input_ids, kwargs)
                        # Embed text positions through ``embed_norm``; the vision
                        # rows keep the tower's own final norm and skip it.
                        input_ids, inputs_embeds = fuse_input_embeds(
                            self._embed_tokens_with_norm,
                            input_ids,
                            mm_embeds,
                            mm_token_ids=None,  # explicit indices: placeholder never embedded
                            text_token_indices=ti,
                            mm_token_indices=mi,
                        )
                        inputs_embeds_prenormed = True
                elif aud_raw and not vis_raw:
                    # Audio-only fusion: same structure as vision, with the audio
                    # tower rows and placeholder positions.
                    mm_embeds = find_input_mm_embeds(aud_raw, ctx_params)
                    if mm_embeds:
                        ti, mi = self._resolve_mm_indices(input_ids, kwargs)
                        input_ids, inputs_embeds = fuse_input_embeds(
                            self._embed_tokens_with_norm,
                            input_ids,
                            mm_embeds,
                            mm_token_ids=None,
                            text_token_indices=ti,
                            mm_token_indices=mi,
                        )
                        inputs_embeds_prenormed = True
                elif vis_raw and aud_raw:
                    # Mixed image+audio: the modalities' placeholder positions may
                    # interleave, so a single index tensor cannot align them --
                    # scatter each modality separately.
                    input_ids, inputs_embeds = self._fuse_media_embeds(
                        input_ids,
                        [
                            (self.image_token_id, vis_raw),
                            (self.audio_token_id, aud_raw),
                        ],
                    )
                    inputs_embeds_prenormed = True
        return super().forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            inputs_embeds_prenormed=inputs_embeds_prenormed,
            **kwargs,
        )

    def _embed_tokens_with_norm(self, ids: torch.IntTensor) -> torch.Tensor:
        """Text embedder that folds ``embed_norm`` onto the token embedding (the
        text tokens are normed while the scattered tower rows keep their own
        norm). Used only by the multimodal fusion path; ``fuse_input_embeds``
        calls it on the text-position ids and scatters the RAW rows in after."""
        return self.model.embed_norm(self.model.embed_tokens(ids))

    def _fuse_media_embeds(
        self,
        input_ids: torch.IntTensor,
        modality_groups: List[Tuple[int, List[torch.Tensor]]],
    ) -> Tuple[torch.IntTensor, torch.Tensor]:
        """Scatter more than one media modality into ``inputs_embeds`` (mixed
        image+audio requests). Text positions are embedded through
        ``embed_norm``; each ``(token_id, embeds)`` group's RAW tower rows
        overwrite exactly that modality's placeholder positions. This generalizes
        the OOV branch of ``fuse_input_embeds`` for the case where a single
        combined index tensor cannot align two disjoint row groups. Fails loudly
        on any per-modality placeholder/feature-row count mismatch."""
        device = input_ids.device
        media_ids = torch.tensor(
            [tid for tid, _ in modality_groups], device=device, dtype=input_ids.dtype
        )
        is_media = torch.isin(input_ids, media_ids)
        text_idx = torch.nonzero(~is_media, as_tuple=True)[0]
        rows_per_group = [
            (tid, torch.cat(embeds, dim=0) if len(embeds) > 1 else embeds[0])
            for tid, embeds in modality_groups
            if embeds
        ]
        hidden_dim = int(rows_per_group[0][1].shape[-1])
        text_embed = self._embed_tokens_with_norm(input_ids[text_idx])
        out = torch.empty(
            input_ids.shape[0], hidden_dim, device=text_embed.device, dtype=text_embed.dtype
        )
        out[text_idx, :] = text_embed
        for tid, rows in rows_per_group:
            pos = torch.nonzero(input_ids == tid, as_tuple=True)[0]
            if int(pos.numel()) != int(rows.shape[0]):
                raise ValueError(
                    f"Inkling media fusion: {int(pos.numel())} placeholder "
                    f"position(s) for token {tid} but {int(rows.shape[0])} feature "
                    f"row(s); counts must match."
                )
            out[pos, :] = rows.to(dtype=out.dtype, device=out.device)
        return input_ids, out

    def load_weights(self, weights: dict, weight_mapper=None):
        # Load the bf16 vision + audio towers first -- the ``model.visual.*`` /
        # ``model.audio.*`` keys the text loader drops -- so any post-load
        # completeness check sees them populated.
        if self.visual is not None:
            visual_weights = {k: v for k, v in weights.items() if k.startswith("model.visual.")}
            self.visual.load_weights(visual_weights)
        if self.audio_tower is not None:
            audio_weights = {k: v for k, v in weights.items() if k.startswith("model.audio.")}
            self.audio_tower.load_weights(audio_weights)
        from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
            InklingHfWeightMapper,
        )

        if weight_mapper is None:
            weight_mapper = InklingHfWeightMapper()
            weight_mapper.init_model_and_config(self, self.model_config)
        # Keep only the text tower, drop mtp, then remap the checkpoint keys to
        # the TRT module tree. This must run here (like modeling_nemotron_h):
        # the base _load_weights_impl_v2 assumes already-mapped names.
        text_weights = filter_weights("model.llm", weights)
        text_weights = weight_mapper.preprocess_weights(text_weights)
        super().load_weights(text_weights, weight_mapper=weight_mapper)


def _text_sub_model_config(
    model_config: ModelConfig[InklingConfig],
) -> ModelConfig[InklingTextConfig]:
    """Build a text-only ``ModelConfig`` from the multimodal one, preserving the
    mapping / quant config so NVFP4 expert loading and TP sharding are intact."""
    text_config = model_config.pretrained_config.text_config
    text_model_config = copy.copy(model_config)
    text_model_config.pretrained_config = text_config
    return text_model_config
