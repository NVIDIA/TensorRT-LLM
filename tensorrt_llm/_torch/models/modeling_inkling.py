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
    decode); see ``attention_backend/sparse/inkling/``.
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
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.inkling import (
    InklingConvRuntime,
    InklingConvState,
    apply_short_conv,
    inkling_forward_args,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import InklingHfWeightMapper
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    MetaInitException,
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
from tensorrt_llm.logger import logger

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


def _assert_draft_chain_loaded(mtp_layers) -> None:
    """Refuse a draft chain whose weights never arrived.

    A trained projection is not all zeros. A block that was built but never
    loaded is: it keeps whatever ``to_empty``/init left, which is zeros for the
    matrices and ones for the norms. That state costs nothing at load and
    everything at runtime -- the block returns zeros, its logits are flat, the
    drafter proposes token 0 forever, the target rejects every draft, and
    speculative decoding burns a whole extra forward per step for no speedup,
    with no error and correct output. It took a probe on the drafter's own
    proposals to see it.

    So the invariant is checked where it is cheap: once, at load.

    The chain's shape is logged alongside it. How many blocks get built and
    which depth each one carries is decided by two config readers that
    disagreed once already -- the top-level multimodal config carries no
    ``num_nextn_predict_layers``, the depth fell back to 1, and MTP resolved to
    EAGLE mode. By this point the chain exists, and this is the only place its
    real shape is visible.
    """
    # An f-string, not %-args: this logger concatenates its arguments rather
    # than interpolating them, so the lazy form prints the format string.
    depths = [getattr(b, "depth", None) for b in mtp_layers]
    logger.info(f"MTP: draft chain loaded with {len(mtp_layers)} block(s), depths {depths}")
    for depth, block in enumerate(mtp_layers):
        for name, param in block.named_parameters():
            # Norms legitimately load as all-ones; a weight MATRIX does not
            # legitimately load as all-zeros.
            if param.dim() < 2:
                continue
            if not torch.any(param != 0):
                raise RuntimeError(
                    f"Inkling MTP depth {depth}: '{name}' is all zeros after "
                    f"loading, i.e. the draft chain's weights never reached the "
                    f"module. The chain would run, propose nothing the target "
                    f"accepts, and cost a forward per step."
                )


def _unquantized_like(quant_config):
    """The target's quant config with weight quantization switched off.

    The MTP chain is NOT quantized in either shipped checkpoint. The evidence is
    direct: under ``model.mtp`` the only scale tensor is the dense MLP's
    ``global_scale`` (which BF16 dense layers carry too) -- there is no
    ``weight_scale``, ``weight_scale_2`` or ``input_scale`` anywhere in the
    chain. And ``hf_quant_config.json``'s ``exclude_modules`` names only
    ``model.llm.*`` entries, so nothing in it can mark the chain as excluded:
    the chain is outside the quantized subtree entirely rather than carved out
    of it.

    Building the draft blocks NVFP4 anyway produces two failures a long way from
    the cause: a strict load reporting missing ``input_proj.weight_scale`` and
    friends, and -- once loading is fixed -- ``fp4_quantize only supports
    fp16/bf16/e4m3`` from the quantize op, which reads as a dtype problem at the
    activation rather than as "this module should never have been quantized".

    ``kv_cache_quant_algo`` is preserved: the draft KV cache follows the
    target's KV quantization regardless of how the chain's weights are stored.
    """
    if quant_config is None:
        return None
    from tensorrt_llm.models.modeling_utils import QuantConfig

    # A fresh instance, not a copy with quant_algo cleared: ``quant_mode`` and
    # ``layer_quant_mode`` are cached_property, so a copy keeps the NVFP4 mode
    # that was already computed and every Linear still builds quantized while
    # the algo field reads None. Nothing about that is visible until the
    # quantize op rejects the activation.
    return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)


def _module_excluded_from_quant(model_config: ModelConfig, name: str) -> bool:
    """True if ``name`` (or an ancestor) is bf16, not NVFP4.

    The checkpoint lists its bf16 modules in ``quant_config.exclude_modules``, and
    the lookup walks the dotted ancestry, so a listed ``...layers.5.attn`` covers
    the projections under it.
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

    The renorm denominator spans the selected routed logits and the shared logits
    together, which the stock routing methods cannot express. ``apply`` returns
    only the routed pair the fused MoE needs; :class:`InklingMoE` recomputes the
    shared gammas from the same joint renorm.
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

    Prefill runs :func:`causal_conv1d_fn`, cached decode
    :func:`causal_conv1d_update` against the per-request conv state; with no
    ``conv_state`` the module falls back to a self-contained causal convolution.

    Under ``tp_shard`` the channels follow the fused qkv projection's kv-head
    sharding and :meth:`load_weights` slices the rank's block out of the
    checkpoint weight. The residual-stream convs are replicated.
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

        The stateless branch runs in fp32 (per the source); the fused cached
        branches run in the input dtype, since the ``causal_conv1d`` ops require
        ``weight.dtype == x.dtype``. ``conv_state`` is updated in place.
        """
        in_dtype = x.dtype
        residual = x
        w = self.weight.squeeze(1).to(x.dtype)  # [channels, kernel]
        if conv_state is not None and is_decode:
            # causal_conv1d_update writes in place into its x argument, so pass a
            # copy -- otherwise the internal residual becomes conv(x) + conv(x).
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
    relative-logit projection. The learned relative bias is a per-(query, head,
    relative-distance) additive ``score_mod`` that no fused backend exposes, so it
    is precomputed here into ``rel_logits`` and gathered+added by the Triton
    kernels. Local layers apply the sliding window natively in the kernel; global
    layers fold the log-scaling ``tau`` into ``rel_logits``.

    The KV write, page-table construction and prefill/decode dispatch run in
    :meth:`InklingTritonAttention.forward`, reached through the standard backend
    contract. ``forward`` overrides the base because the k/v short-convs must run
    between ``split_qkv`` and ``apply_qk_norm``, and ``rel_logits`` is projected
    from ``hidden_states``, which ``forward_impl`` never receives.
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

        # Attention is bf16, not NVFP4: hand the base a shallow ModelConfig copy
        # with an empty quant_config so qkv_proj/o_proj are built unquantized.
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
        # These three differ between local and global layers, and
        # create_attention() has no passthrough for model-specific kwargs.
        self.attn.sm_scale = self.sm_scale
        self.attn.rel_extent = self.rel_extent
        self.attn.window_left = self.window_left

        # Attention-scoped TP: the Inkling-only tensors below (r_proj, k/v sconv)
        # hang off the same head/kv-head split as qkv_proj, so they must follow
        # the attention TP rather than the global one.
        tp_size = 1 if model_config.mapping.enable_attention_dp else model_config.mapping.tp_size
        assert self.num_heads == num_heads // tp_size, (
            f"attention TP disagrees with the base Attention: base kept "
            f"{self.num_heads} of {num_heads} heads, this rule expects "
            f"{num_heads // tp_size} (enable_attention_dp="
            f"{model_config.mapping.enable_attention_dp}, "
            f"mapping.tp_size={model_config.mapping.tp_size})"
        )
        # r projection: per-head relative states, sharded by head like q and not
        # gathered (consumed locally to build the bias). Replicated under ADP.
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
        # k/v short convs act on the k/v stream of the fused qkv projection and
        # are sharded by kv-head like it; InklingShortConv slices at load.
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

    def _project(self, hidden_states, conv_pool_kv=None, conv_rt=None, conv_capture_kv=None):
        """Fused qkv projection -> split -> k/v short-conv -> per-head qk RMSNorm.

        Returns ``(q, k, v)`` shaped ``[T, local_heads, head_dim]`` /
        ``[T, local_kv_heads, head_dim]``. With ``conv_pool_kv`` + ``conv_rt`` the
        k/v short-convs run through the runtime state pool; without them they run
        the stateless full-sequence causal conv. ``conv_capture_kv=(cap_k, cap_v)``
        is the verify-step capture the post-verify commit replays from.
        """
        D = self.head_dim
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv, None, None)
        # k/v short convolution before the q/k norm (source order).
        if conv_pool_kv is not None:
            pool_k, pool_v = conv_pool_kv
            cap_k, cap_v = conv_capture_kv if conv_capture_kv is not None else (None, None)
            k = apply_short_conv(self.k_sconv, k, pool_k, conv_rt, cap_k)
            v = apply_short_conv(self.v_sconv, v, pool_v, conv_rt, cap_v)
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

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        conv_pool_kv=None,
        conv_capture_kv=None,
        conv_rt=None,
        **kwargs,
    ):
        """Inkling attention through the Triton score_mod path.

        ``conv_pool_kv`` + ``conv_rt`` drive the k/v short-convs through the
        runtime state pool; without them they run stateless over the sequence.
        ``conv_capture_kv`` carries the verify-step captures for the post-verify
        conv-window commit.
        """
        num_tokens = hidden_states.shape[0]
        # The pre-attention RMSNorm can emit fp32 while the attention/r
        # projections are bf16, so cast once here.
        hidden_states = hidden_states.to(self.qkv_proj.weight.dtype)
        q, k, v = self._project(hidden_states, conv_pool_kv, conv_rt, conv_capture_kv)
        rel_logits = self._build_rel_logits(hidden_states, position_ids)
        # Standard backend contract; rel_logits and the mixed-batch certificate
        # ride AttentionForwardArgs.sparse_backend_args (see inkling/params.py).
        attn_out = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=inkling_forward_args(rel_logits, allow_mixed=conv_rt is not None),
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
        # Under attention DP the dense MLP goes data-parallel too: keeping the
        # column/row split would all-reduce partials from different requests.
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
        # [n_shared, 2*inter, hidden] fused gate+up; [n_shared, hidden, inter]
        # down. Model dtype: these run as raw bmms and load unquantized.
        self.shared_w13 = nn.Parameter(
            torch.empty(self.n_shared, 2 * inter, hidden, dtype=config.torch_dtype)
        )
        self.shared_w2 = nn.Parameter(
            torch.empty(self.n_shared, hidden, inter, dtype=config.torch_dtype)
        )
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        # hidden_states: [T, hidden]; gammas: [T, n_shared] fp32, applied after
        # the (linear) down projection where it commutes.
        x = hidden_states.unsqueeze(0).expand(self.n_shared, -1, -1)
        gate_up = torch.bmm(x, self.shared_w13.transpose(1, 2))
        # shared_w13 loads raw with gate/up interleaved along its output dim, so
        # gate = even channels and up = odd; chunk(2) would pair the wrong ones.
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
        """Per-layer expert quant: NVFP4 unless the checkpoint excludes it, in
        which case an empty ``QuantConfig`` gives ``create_moe`` a bf16 MoE."""
        if _module_excluded_from_quant(model_config, f"model.llm.layers.{layer_idx}.mlp.experts"):
            from tensorrt_llm.models.modeling_utils import QuantConfig

            return QuantConfig()
        return model_config.quant_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Routed + shared experts. ``all_rank_num_tokens`` is this step's
        per-rank token count, which ``FusedMoE`` needs to gather under DP."""
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
        conv_capture: Optional[InklingConvState] = None,
        conv_rt: Optional[InklingConvRuntime] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pre-norm attention + MLP, each followed by a short-conv (internal
        residual), then the residual add.

        With ``conv_rt`` given, ``conv_state`` holds this layer's four pool
        buffers and each short-conv runs through them; without it they run
        stateless over the whole sequence. ``conv_capture`` holds the matching
        verify-step captures (all-None outside a verify step).
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
        caps = (
            conv_capture if conv_capture is not None else InklingConvState(None, None, None, None)
        )
        h = self.attn(
            position_ids,
            h,
            attn_metadata,
            conv_pool_kv=(conv_state.k, conv_state.v),
            conv_capture_kv=(caps.k, caps.v),
            conv_rt=conv_rt,
            **kwargs,
        )
        h = residual + apply_short_conv(self.attn_sconv, h, conv_state.attn, conv_rt, caps.attn)

        residual = h
        hm = self._run_mlp(self.mlp_norm(h), all_rank_num_tokens)
        return residual + apply_short_conv(self.mlp_sconv, hm, conv_state.mlp, conv_rt, caps.mlp)


def _mtp_num_depths(config: InklingTextConfig) -> int:
    """How many depths the draft chain has.

    Inkling declares this on ``mtp_config`` rather than at the top level of the
    text config, which is where the framework's MTPForCausalLM looks
    (``pretrained_config.num_nextn_predict_layers``), so it is mirrored there --
    see InklingForCausalLM -- and read back through one accessor here.
    """
    n = getattr(config, "num_nextn_predict_layers", None)
    if n:
        return int(n)
    # The ids name WHICH depths are banded, not how many exist
    # (``is_mtp_local_depth`` treats them as a membership set), so the count
    # comes from the largest index. The shipped small checkpoint declares 8
    # depths as [0, 2, 4, 5, 6, 7]: six ids, last one 7, and only ``max + 1``
    # gets back to 8.
    ids = getattr(config, "mtp_local_layer_ids", None) or ()
    return (max(ids) + 1) if ids else 1


def _mtp_depth_from_global_index(config: InklingTextConfig, global_layer_idx: int) -> int:
    """Recover a chain depth from the layer index the framework builds with.

    ``MTPForCausalLM`` passes ``depth + start_layer_idx`` where start_layer_idx
    is the trunk's layer count, so the trunk is SUBTRACTED back off. A modulo
    over the depth count gives the same answer only when
    ``num_hidden_layers % num_depths == 0``; on the shipped small checkpoint
    that is 42 % 8 = 2, which shifted every block onto another depth's geometry
    while it carried its own depth's weights.

    Raises rather than clamping: an index outside the chain means the caller and
    this config disagree about how tall the trunk is, and every downstream
    symptom of that (wrong window, wrong KV-head count, a conv pool sized from a
    different depth) is silent.
    """
    depth = global_layer_idx - config.num_hidden_layers
    num_depths = _mtp_num_depths(config)
    if not 0 <= depth < num_depths:
        raise ValueError(
            f"Inkling MTP block built at global layer index {global_layer_idx}, "
            f"i.e. depth {depth} of a {num_depths}-deep chain sitting above "
            f"{config.num_hidden_layers} trunk layers. MTPForCausalLM passes the "
            "trunk's layer count as start_layer_idx, so this means the two "
            "disagree about the trunk."
        )
    return depth


class InklingMTPHead(nn.Module):
    """Per-depth head: optional chain post-norm, then the shared LM head.

    Mirrors ``DeepseekV3MTPHead``. The norm exists only when the checkpoint
    declares ``chain_hidden_post_norm`` -- both shipped Inkling releases set it
    False and ship no ``chain_norm`` tensor, so building it unconditionally
    would create a parameter the loader then has to explain away.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig], use_norm: bool):
        super().__init__()
        config = model_config.pretrained_config
        self.norm = (
            RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            if use_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """One logit row per SEQUENCE, not per token.

        ``MTPWorker`` samples one draft token per sequence from what this
        returns and then writes it back at ``last_tokens_idx``, so a row per
        token makes that assignment a shape mismatch -- [draft_len] into [1].
        Gathering here rather than in the worker also avoids running the
        vocab-sized projection over every token of the batch, which is the
        reason DeepSeek's MTP head does the same thing.
        """
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if not return_context_logits and attn_metadata is not None:
            last_tokens = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden_states = hidden_states[last_tokens]
        return lm_head(hidden_states)


class InklingMTPBlock(nn.Module):
    """One depth of the next-N draft chain.

    Structure follows SGLang's ``InklingMTPLayer``: the previous depth's hidden
    state and this depth's token embedding are each normalized, concatenated and
    projected back to hidden width, then run through what is otherwise an
    ordinary decoder layer.

    The decoder layer is reused unchanged. It asks its config which layers are
    dense and which are banded, so ``mtp_block_config`` hands it a config where
    those ordinary questions return the draft answers -- rather than teaching
    the layer a second notion of what layer it is.
    """

    def __init__(
        self,
        model_config: ModelConfig[InklingTextConfig],
        depth: int,
        aux_stream_dict: Optional[dict] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        # MTPForCausalLM passes the target's layer count as start_layer_idx, so
        # the index arrives offset by the trunk depth; the chain's own geometry
        # is indexed from 0.
        # The offset index is the GLOBAL layer index, which is what the draft KV
        # cache manager keys its layer offsets by; the chain's own geometry is
        # indexed from 0. Both are needed, so keep both rather than folding one
        # away: passing the chain depth as the layer index asks the draft
        # manager for layer 2 when the buffers live at trunk+2, a KeyError deep
        # in the first draft forward.
        global_layer_idx = depth
        depth = _mtp_depth_from_global_index(config, global_layer_idx)
        # Accepted for the framework's uniform constructor signature. Inkling's
        # draft blocks are dense, so there is no MoE/shared-expert overlap to
        # schedule on a second stream.
        del aux_stream_dict
        self.depth = depth
        self.dtype = config.torch_dtype
        self.embed_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.hidden_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        # Concatenation of two hidden-width tensors, projected back to hidden.
        self.input_proj = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            # The chain is BF16; see _unquantized_like.
            quant_config=None,
        )
        block_model_config = copy.copy(model_config)
        block_model_config.pretrained_config = config.mtp_block_config(depth, global_layer_idx)
        block_model_config.quant_config = _unquantized_like(model_config.quant_config)
        self.transformer_block = InklingDecoderLayer(block_model_config, global_layer_idx)
        # MTPWorker calls shared_head(hidden, lm_head, attn_metadata) per depth.
        # ``chain_hidden_post_norm`` is False in both shipped checkpoints, which
        # ships no chain_norm weight -- so the norm is built only when the
        # checkpoint declares it, and the head is otherwise a straight LM-head
        # application.
        self.shared_head = InklingMTPHead(
            model_config, use_norm=bool(getattr(config, "chain_hidden_post_norm", False))
        )

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Fold the previous depth's hidden state into this depth's embedding.

        The signature is the one MTPWorker calls with -- it passes the target
        model's ``embed_tokens`` in and the rest as ``**draft_inputs`` -- so the
        embedding lookup happens here rather than in the caller.
        """
        inputs_embeds = embed_tokens(input_ids)
        combined = torch.cat(
            (self.hidden_norm(hidden_states), self.embed_norm(inputs_embeds)), dim=-1
        )
        # RMSNorm can emit fp32 while the NVFP4 quantize op accepts only
        # fp16/bf16/e4m3 and refuses fp32 outright, so this boundary needs a
        # cast to the compute dtype. Three candidates were tried on the cluster
        # and only the last is that dtype: ``input_proj.weight.dtype`` is a
        # quantized Linear's PACKED storage type; ``config.torch_dtype`` is what
        # the config declares rather than what the model runs in; and the
        # incoming ``hidden_states`` are whatever the spec worker hands over,
        # which is not guaranteed either. The norm's own weight is a real
        # parameter of the model, built at the compute dtype.
        combined = combined.to(self.embed_norm.weight.dtype)
        # The chain's own short-conv state. Without it the block takes the
        # decoder layer's stateless branch, which is wrong twice over: a
        # stateless conv runs across the context/generation boundary of a packed
        # batch (the trunk raises NotImplementedError for exactly that), and the
        # chain would carry no conv history between steps at all. Neither fails.
        #
        # It comes from the manager in play (``mgr.prepare_conv_runtime``) rather
        # than from the metadata's published runtime: that split was published
        # once, during prepare(), from the TARGET manager, while the draft
        # forward runs inside the draft KV cache context with the manager
        # swapped underneath.
        #
        # KNOWN GAP, and deliberately left: the chain's windows are never rolled
        # back. ``MTPWorker`` calls ``commit_conv_state_after_verify`` once, on
        # ``attn_metadata.kv_cache_manager``, before entering the draft context
        # -- so the commit only ever reaches the TARGET's pool. The chain
        # advances its own windows over every drafted token and keeps the ones
        # the next step rejects.
        #
        # This is an acceptance-rate defect, not a correctness one, and the
        # asymmetry is the whole reason the target-side commit exists: a wrong
        # window in the TARGET corrupts the logits that decide acceptance, so
        # the committed tokens leave the greedy trajectory; a wrong window in
        # the CHAIN only produces a worse draft, which the target then rejects.
        # Output stays exactly right and the cost is a forward per step.
        conv_state = conv_capture = conv_rt = None
        mgr = getattr(attn_metadata, "kv_cache_manager", None)
        prepare = getattr(mgr, "prepare_conv_runtime", None)
        if prepare is not None:
            cache, conv_rt = prepare(attn_metadata)
            layer_idx = self.transformer_block.layer_idx
            conv_state = cache.layer_state(layer_idx)
            conv_capture = cache.layer_capture(layer_idx)
        return self.transformer_block(
            position_ids=position_ids,
            hidden_states=self.input_proj(combined),
            attn_metadata=attn_metadata,
            conv_state=conv_state,
            conv_capture=conv_capture,
            conv_rt=conv_rt,
            **kwargs,
        )


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
        # MTPForCausalLM reads this when it builds the draft chain. Inkling's
        # blocks are dense and schedule nothing on a second stream, so an empty
        # mapping is the honest value rather than a fabricated stream.
        self.aux_stream_dict: dict = {}

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_embeds_prenormed: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Decoder stack. The short-conv pool comes from the cache manager and its
        per-forward split from the state metadata the base ``prepare()`` built; a
        metadata without one keeps the stateless conv.

        ``inputs_embeds_prenormed`` is set on the multimodal path, where the
        wrapper has already applied ``embed_norm`` to the text embeddings before
        scattering the raw tower rows in."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Views only, no copy: the slot write happened in prepare(), outside the
        # captured region. conv_rt is the gate for the pool path.
        conv_rt = InklingConvRuntime.from_metadata(attn_metadata)
        conv_cache = (
            getattr(attn_metadata.kv_cache_manager, "conv_state_cache", None)
            if conv_rt is not None
            else None
        )
        # Per-rank token counts for this step, set on attn_metadata only under
        # attention DP; FusedMoE reads it to pad and gather across ranks.
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        hidden_states = inputs_embeds if inputs_embeds_prenormed else self.embed_norm(inputs_embeds)
        for i, layer in enumerate(self.layers):
            layer_state = conv_cache.layer_state(i) if conv_cache is not None else None
            layer_capture = conv_cache.layer_capture(i) if conv_cache is not None else None
            hidden_states = layer(
                position_ids,
                hidden_states,
                attn_metadata,
                conv_state=layer_state,
                conv_capture=layer_capture,
                conv_rt=conv_rt,
                all_rank_num_tokens=all_rank_num_tokens,
            )
        return self.norm(hidden_states)


class InklingForCausalLM(SpecDecOneEngineForCausalLM[InklingModel, InklingTextConfig]):
    """Text CausalLM: muP logit scaling + unpadded-vocab slice.

    ``embed`` and ``unembed`` are separate checkpoint tensors (never tied). The
    ``LMHead`` is built at the unpadded vocab size so its forward slices off the
    padding automatically; hidden states are divided by
    ``logits_mup_width_multiplier`` before the head (accuracy-critical).

    The base is ``SpecDecOneEngineForCausalLM`` rather than the plain
    ``DecoderModelForCausalLM`` because one-engine speculative decoding is not
    something a model opts into piecemeal: that base is what builds the draft
    model, creates the spec worker, and -- the part that actually bit -- routes
    the forward through the worker so logits are taken at
    ``spec_metadata.gather_ids`` instead of over every token. Without it the
    trunk returns a flat [tokens, vocab] where the sampler expects one entry per
    verified position, which surfaces as an IndexError in HandleLogits naming
    neither speculation nor Inkling. With no ``spec_config`` the base is
    behaviourally the plain decoder it replaces.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        config = model_config.pretrained_config
        self.mup_multiplier = float(config.logits_mup_width_multiplier)
        # ``model_config`` positionally: the one-engine base takes it as
        # ``model_config`` (it needs the spec_config off it) and forwards it to
        # the decoder base as ``config``.
        super().__init__(
            InklingModel(model_config),
            model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.unpadded_vocab_size,
        )
        self._assert_inkling_attn_backend(model_config)
        self._assert_inkling_moe_parallel(model_config)
        self._assert_inkling_spec_conv_state(model_config)
        self._apply_allreduce_strategy()

    @staticmethod
    def _assert_inkling_spec_conv_state(model_config) -> None:
        """Check the conv pool can roll back before allowing speculative decoding.

        The short-conv state is a sliding window of past INPUTS, mutated in
        place. A verify step advances it over every drafted token while only a
        prefix is accepted, so without a commit the window is left holding
        tokens the model never emitted -- and nothing about that is detectable:
        right shape, right dtype, a perfectly valid window. The KV cache is safe
        here by construction, being position-indexed, which leaves the conv as
        the one piece of state needing an explicit commit and the one no smoke
        test would catch.

        That commit now exists (``InklingConvStateCache.commit_after_verify``,
        driven from ``MTPWorker`` once acceptance is known). What is checked here
        is the precondition it depends on: the capture buffers are sized from
        ``max_draft_len``, so a chain deeper than they allow would commit the
        wrong window rather than fail.
        """
        spec_config = getattr(model_config, "spec_config", None)
        if spec_config is None:
            return
        # The chain's blocks are addressed by global layer index, which only the
        # separate draft KV cache manager provides. Without it every draft
        # forward dies on a bare KeyError several minutes in, so the condition is
        # checked where it can still be explained.
        from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache

        if not should_use_separate_draft_kv_cache(spec_config):
            raise ValueError(
                "Inkling speculative decoding requires a separate draft KV cache: "
                "the draft chain's layers are not the target's, so they cannot "
                "share its cache, and they are addressed by the global layer "
                "index the separate manager is keyed by."
            )
        text_config = getattr(
            model_config.pretrained_config, "text_config", model_config.pretrained_config
        )
        if getattr(text_config, "num_nextn_predict_layers", None) is None:
            # ``MTPForCausalLM`` reads this as a bare attribute, so its absence
            # is an AttributeError from inside framework code rather than a
            # statement about the checkpoint. It is absent exactly when the
            # checkpoint carries no ``mtp_config`` -- i.e. ships no draft chain
            # -- which is worth saying in those words.
            raise ValueError(
                "This Inkling checkpoint declares no MTP chain (no mtp_config, "
                "so no draft depths to build). Speculative decoding needs a "
                "checkpoint that ships one; run without speculative_config."
            )
        if not spec_config.spec_dec_mode.is_mtp_vanilla():
            # Inkling's chain is vanilla MTP: every depth has its own weights
            # and its own banded/global attention geometry. The EAGLE-style
            # modes build ONE block and replay it, which is a different model.
            #
            # This is reachable by configuration rather than by asking for it:
            # ``spec_dec_mode`` resolves to EAGLE whenever the spec config's
            # ``num_nextn_predict_layers`` comes out as 1, and that field is
            # filled from the TOP-LEVEL pretrained config. Nothing downstream
            # complains -- the model side reads the text config and believes it
            # has N depths while the framework has decided there is one.
            raise ValueError(
                f"Inkling MTP needs vanilla MTP, got {spec_config.spec_dec_mode}. "
                "Its draft depths have distinct weights and attention geometry, "
                "so a single replayed block is a different model. This usually "
                "means the chain depth did not reach the speculative config; "
                "set use_mtp_vanilla=True on MTPDecodingConfig if the checkpoint "
                "genuinely declares one depth."
            )
        if getattr(getattr(model_config, "mapping", None), "enable_attention_dp", False):
            # Inkling MTP requires a separate draft KV cache; attention DP
            # refuses to provide one. Both sides say so in as many words:
            # _util._should_create_separate_draft_kv_cache returns False under
            # attention DP ("separate draft KV cache is not supported"), and the
            # guard above this one requires it. Incompatible by construction.
            #
            # So this is not a patchable mismatch. The conv pool's sizing, its
            # layer offset and the draft-context manager swap all assume the
            # chain has its own manager. Supporting the combination needs either
            # the framework to allow a separate draft cache under attention DP,
            # or Inkling's global-layer addressing to work inside the target
            # manager. Refused until one of those is chosen.
            raise ValueError(
                "Inkling MTP does not support attention DP. MTP needs a "
                "separate draft KV cache and attention DP does not provide "
                "one, so the draft chain's layers fold into the target "
                "manager, which is sized and addressed for the trunk alone. "
                "Set enable_attention_dp=False when enabling MTP on Inkling."
            )
        if getattr(spec_config, "use_relaxed_acceptance_for_thinking", False):
            # Relaxed acceptance is LOSSY by design -- it takes a draft that
            # matches any of the target's top-K instead of its top-1 -- and it
            # buys that back by applying only inside the thinking phase, which
            # it locates with ``begin_thinking_phase_token`` /
            # ``end_thinking_phase_token``. Those default to 128798/128799,
            # DeepSeek-R1's ``<think>``/``</think>``. In Inkling's vocabulary
            # they are not special tokens at all; its thinking run is opened by
            # ``<|content_thinking|>`` (200008) and closed by the NEXT channel
            # marker rather than by a matching end token, so the paired-token
            # shape cannot express it however the ids are set.
            #
            # Left alone this does not raise. The phase is simply never entered
            # where it should be, and is entered wherever those two ordinary
            # token ids happen to fall -- relaxing acceptance outside the
            # thinking text, which is the one place the mode is not meant to be
            # lossy. Nothing in the output says so.
            raise ValueError(
                "Inkling MTP does not support relaxed acceptance for thinking. "
                "It is gated on begin/end_thinking_phase_token, a paired "
                "<think>/</think> shape; Inkling opens thinking with "
                "<|content_thinking|> and ends it by switching channel, so "
                "there is no end token to name. Enabling it would relax "
                "acceptance -- which is lossy -- outside the thinking phase. "
                "Set use_relaxed_acceptance_for_thinking=False."
            )
        if getattr(model_config, "use_cuda_graph", False):
            # The verify step walks the drafted positions one at a time, writing
            # KV and re-attending per position, which is not capturable; the
            # backend raises when it sees a captured batch. That raise lands
            # inside warmup, minutes in, from a stack that names neither
            # speculation nor the graph setting -- so refuse the combination
            # here, where both are still in hand.
            raise ValueError(
                "Inkling speculative decoding cannot run with CUDA graphs: the "
                "verify step walks the drafted positions one at a time and "
                "cannot be captured. Set cuda_graph_config=None (or remove it "
                "from --extra_llm_api_options) when enabling MTP on Inkling."
            )

    @staticmethod
    def _assert_inkling_attn_backend(model_config) -> None:
        """Fail at load if the attention backend family was overridden.

        Inkling's backend is registered in ``sparse/registry.py`` under the
        ``TRTLLM`` family only; the ``VANILLA`` / ``FLASHINFER`` registries do not
        know the ``"inkling"`` algorithm and fail with a message naming neither
        Inkling nor the setting responsible.
        """
        backend = getattr(model_config, "attn_backend", None)
        if backend is not None and str(backend).upper() != "TRTLLM":
            raise ValueError(
                f"Inkling requires the TRTLLM attention backend family (got "
                f"{backend!r}). Its backend is registered in "
                "attention_backend/sparse/registry.py under that family only, "
                "and its Triton decode kernel reads per-step seq_lens and page "
                "tables off TrtllmAttentionMetadata. Remove the attn_backend "
                "override from --extra_llm_api_options / LLM(attn_backend=...)."
            )

    @staticmethod
    def _assert_inkling_moe_parallel(model_config) -> None:
        """Reject an expert-parallel layout the MoE backend cannot serve.

        The CUTLASS backend -- the only routed-expert backend Inkling ships --
        does not opt into ``_supports_non_divisible_ep``, so a non-divisible
        ``moe_expert_parallel_size`` fails inside expert-slot bookkeeping rather
        than at load.
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

    def _apply_allreduce_strategy(self) -> None:
        """Keep Inkling's all-reduces off the NCCL_SYMMETRIC tactic.

        Under CUDA-graph capture a symmetric all-reduce corrupts the run when its
        send buffer is unregistered while its recv buffer is a registered NCCL
        window; Inkling's decode message size hits that case and decode collapses
        to a repeated token. The affected all-reduces are built by generic
        modules, so rebuilding them here keeps the mitigation model-local.
        """
        for mod in self.modules():
            old = getattr(mod, "all_reduce", None)
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
        spec_metadata=None,
        resource_manager=None,
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
            **kwargs,
        )
        # Padded rows are scratch the batch was rounded up to; the spec worker
        # indexes by request and would read them as real tokens. The one-engine
        # base trims here for the same reason -- this override has to repeat it
        # rather than inherit it.
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        # muP: accuracy-critical, and it applies to the lm_head input only. The
        # draft chain is handed the UNDIVIDED hidden states below, matching the
        # SGLang reference, because the division belongs to the head rather than
        # to the residual stream the chain continues.
        head_input = hidden_states / self.mup_multiplier
        if self.spec_worker is not None:
            logits = self.logits_processor.forward(
                head_input[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )
            # On a multimodal request ``fuse_input_embeds`` returns input_ids as
            # None -- the token stream became an embedding stream -- and the
            # worker subscripts what it is given (``input_ids[:num_ctx_tokens]``
            # in prepare_drafter_inputs). The wrapper forwards the pre-fusion ids
            # under ``orig_input_ids`` for exactly this, the same key
            # Qwen3-VL and Gemma4-MM use.
            spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_input_ids is not None:
                    spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
                if spec_position_ids is not None:
                    spec_position_ids = spec_position_ids[..., : attn_metadata.num_tokens]
            return self.spec_worker(
                input_ids=spec_input_ids,
                position_ids=spec_position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )
        return self.logits_processor.forward(
            head_input, self.lm_head, attn_metadata, return_context_logits
        )

    def load_weights(self, weights: dict, weight_mapper=None):
        """Load the trunk, then the draft chain if one was built.

        The text-only path goes through here; the multimodal subclass overrides
        load_weights for the towers and calls _load_mtp_weights itself. Without
        this override the base implementation runs and the draft blocks stay at
        their initial values -- speculative decoding would then produce garbage
        drafts that the target rejects, i.e. a silent speed regression rather
        than an error.
        """
        super().load_weights(weights, weight_mapper=weight_mapper)
        self._load_mtp_weights(weights, weight_mapper)

    def _load_mtp_weights(self, weights: dict, weight_mapper) -> None:
        """Load the draft chain, if one was built, through the generic loader.

        Only the depths that exist are loaded: the runtime caps the chain at
        ``min(max_draft_len, checkpoint depths)``, so a server asking for 3
        draft tokens builds 3 blocks out of the checkpoint's 8. Loading all 8
        into 3 modules would fail; skipping the extras is correct, and they are
        reported so a silent shortfall is visible.

        ``load_state_dict`` cannot do this job. The checkpoint carries the raw
        per-projection names and full-width tensors while the block has fused
        ``qkv_proj``/``gate_up_proj``, NVFP4 scale tensors and TP-sharded
        widths; fusion, scales and sharding are all the loader's work. The
        weight mapper has already renamed the chain to ``mtp_layers.<d>....``,
        so the same ``_load_weights_impl`` that loads the trunk applies here.
        """
        from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl

        draft_model = getattr(self, "draft_model", None)
        mtp_layers = getattr(draft_model, "mtp_layers", None)
        if not mtp_layers:
            return
        built = len(mtp_layers)
        # The chain arrives under its CHECKPOINT names (``model.mtp.layers.N.``)
        # and the loader walks the MODULE tree (``mtp_layers.N.``); the mapper
        # is what turns one into the other. Looking for the mapped names in the
        # raw dict finds nothing, and "nothing" is a silent success here: every
        # draft block keeps its initial values, the drafter proposes token 0 on
        # every step, the target rejects all of it, and speculative decoding
        # runs at a loss with no error anywhere. Measured before this was fixed:
        # every weight matrix of the built blocks read absmean 0.0, with the
        # norms at their init 1.0.
        mtp_weights = weight_mapper.preprocess_weights(
            {k: v for k, v in weights.items() if k.startswith("model.mtp.")}
        )
        available = {
            int(k.split("mtp_layers.")[1].split(".")[0])
            for k in mtp_weights
            if k.startswith("mtp_layers.")
        }
        if not available:
            logger.warning(
                "MTP chain built but the checkpoint carries no draft weights; "
                "the draft blocks stay at their initial values."
            )
            return
        # Hand the loader only the depths that were built: it walks the module
        # tree, so extra depths would simply go unclaimed, but reporting the
        # shortfall is what keeps "capped by max_draft_len" from looking like a
        # loading bug later.
        depth_weights = {
            k: v
            for k, v in mtp_weights.items()
            if k.startswith("mtp_layers.") and int(k.split("mtp_layers.")[1].split(".")[0]) < built
        }
        # ``_load_weights_impl`` reads ``model.model_config`` for the quant and
        # mapping config it needs to fuse and shard. ``MTPForCausalLM`` keeps
        # only ``mtp_layers``/``lm_head``/``embed_tokens``, so it is handed the
        # config the chain was built from -- the same object, not a copy, since
        # the blocks were constructed with it.
        if not hasattr(draft_model, "model_config"):
            draft_model.model_config = self.model_config
        if not hasattr(draft_model, "config"):
            # The loader reads ``config.num_key_value_heads`` for ONE purpose:
            # duplicating KV weights when the head count is below tp_size. The
            # chain's depths disagree on that count (banded 16, global 8), so a
            # single value cannot be right for all of them -- it is safe here
            # only because both counts exceed the tensor-parallel sizes Inkling
            # runs at. If Inkling is ever run at TP > 8 without attention DP,
            # this needs the per-depth accessor instead.
            # ``self.model_config`` is already the TEXT sub-config here (the
            # causal LM is constructed from it), so no further descent.
            draft_model.config = self.model_config.pretrained_config
        _load_weights_impl(draft_model, depth_weights)
        _assert_draft_chain_loaded(mtp_layers)
        if len(available) > built:
            logger.info(
                f"MTP: built {built} of the checkpoint's {len(available)} draft depths "
                f"(capped by max_draft_len); the remainder are not loaded."
            )


def _encode_inkling_image_embeds(
    visual: InklingVisionModel, multimodal_params: list
) -> List[torch.Tensor]:
    """Run the hMLP vision tower over the context requests' patch features.

    Returns ``[feats]`` of shape ``(sum_patches, decoder_dmodel)`` -- the shape
    ``find_input_mm_embeds`` slices -- or ``[]`` when no context request carries
    image features."""
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

    Returns ``[feats]`` of shape ``(sum_frames, decoder_dmodel)``, the same shape
    contract as the image encoder, or ``[]`` when there is no audio."""
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


def _has_meta_tensors(module: nn.Module) -> bool:
    return any(getattr(p, "is_meta", False) for p in module.parameters()) or any(
        getattr(b, "is_meta", False) for b in module.buffers()
    )


def _build_replicated_bf16_tower(tower_cls, tower_config):
    """Build a replicated bf16 media tower, or defer it under ``MetaInitMode``.

    Returns ``(tower, deferred_config)``: exactly one is ``None``. A non-``None``
    ``deferred_config`` means ``load_weights`` must rebuild the tower before
    loading into it.

    The deferral is what keeps meta init working: ``MetaInitMode`` permits only
    random-init ops on meta parameters, so the tower's ``.to(torch.bfloat16)``
    raises ``MetaInitException``. The loader wraps that mode around the entire
    ``from_config`` call, so one tower raising there silently costs the whole text
    stack its meta init. ``modeling_kimi_k3_vl`` defers its encoder for the same
    reason.
    """
    if tower_config is None or not getattr(tower_config, "decoder_dmodel", None):
        return None, None
    try:
        tower = tower_cls(tower_config)
    except MetaInitException:
        return None, tower_config
    # Constructing may succeed while leaving meta parameters behind; the
    # `.to(bfloat16)` below is what would raise on them, so check first.
    if _has_meta_tensors(tower):
        return None, tower_config
    return tower.to(torch.bfloat16), None


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

    Text-only requests route straight to :class:`InklingForCausalLM` over the
    ``text_config`` sub-config. Media requests are preprocessed by
    :class:`InklingInputProcessor`, which expands each placeholder to one token
    per vision patch / dMel frame; the replicated bf16 towers (``self.visual``,
    ``self.audio_tower``) emit one row per position and those rows are fused into
    ``inputs_embeds`` before the text decoder. Video is multi-frame images
    through the vision tower. MTP is not implemented.
    """

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        # use_kv_cache_manager_v2: the per-layer KV-head split needs V2's
        # per-layer geometry; V1's unified pool would mis-size the KV bytes.
        # enable_block_reuse: the short-conv window is per-request state outside
        # the KV cache and every context request is seeded empty, so a reused
        # prefix would convolve against padding.
        return {
            "kv_cache_config": {
                "use_kv_cache_manager_v2": True,
                "enable_block_reuse": False,
            },
        }

    def __init__(self, model_config: ModelConfig[InklingConfig]):
        text_model_config = _text_sub_model_config(model_config)
        super().__init__(text_model_config)
        self._top_model_config = model_config
        # Both towers are replicated bf16 submodules -- every rank runs the
        # identical tower over identical inputs -- and are built through
        # _build_replicated_bf16_tower, which returns None under MetaInitMode and
        # records the config for load_weights to rebuild from.
        vision_config = getattr(model_config.pretrained_config, "vision_config", None)
        self.visual, self._deferred_vision_config = _build_replicated_bf16_tower(
            InklingVisionModel, vision_config
        )
        audio_config = getattr(model_config.pretrained_config, "audio_config", None)
        self.audio_tower, self._deferred_audio_config = _build_replicated_bf16_tower(
            InklingAudioModel, audio_config
        )
        # The media placeholder ids the chat template emits, surfaced to the model
        # engine's _prepare_multimodal_indices so it can locate the media rows.
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
        # Kept before fusion: ``fuse_input_embeds`` returns input_ids as None,
        # and the MTP worker needs the token stream. Passed down under
        # ``orig_input_ids`` (Qwen3-VL / Gemma4-MM use the same key) rather than
        # threaded as a named argument, so the non-speculative path is untouched.
        orig_input_ids = input_ids
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
        kwargs.setdefault("orig_input_ids", orig_input_ids)
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

    def _materialize_deferred_towers(self):
        """Rebuild any tower that ``__init__`` skipped under ``MetaInitMode``.

        Runs before the towers are loaded into. By this point the meta-init
        context is long gone, so construction takes its normal path.
        """
        if self._deferred_vision_config is not None:
            self.visual = InklingVisionModel(self._deferred_vision_config).to(torch.bfloat16)
            self._deferred_vision_config = None
        if self._deferred_audio_config is not None:
            self.audio_tower = InklingAudioModel(self._deferred_audio_config).to(torch.bfloat16)
            self._deferred_audio_config = None

    def load_weights(self, weights: dict, weight_mapper=None):
        self._materialize_deferred_towers()
        # Load the bf16 vision + audio towers first -- the ``model.visual.*`` /
        # ``model.audio.*`` keys the text loader drops -- so any post-load
        # completeness check sees them populated.
        if self.visual is not None:
            visual_weights = {k: v for k, v in weights.items() if k.startswith("model.visual.")}
            self.visual.load_weights(visual_weights)
        if self.audio_tower is not None:
            audio_weights = {k: v for k, v in weights.items() if k.startswith("model.audio.")}
            self.audio_tower.load_weights(audio_weights)
        if weight_mapper is None:
            weight_mapper = InklingHfWeightMapper()
            weight_mapper.init_model_and_config(self, self.model_config)
        # Keep only the text tower, drop mtp, then remap the checkpoint keys to
        # the TRT module tree. This must run here (like modeling_nemotron_h):
        # the base _load_weights_impl_v2 assumes already-mapped names.
        text_weights = filter_weights("model.llm", weights)
        text_weights = weight_mapper.preprocess_weights(text_weights)
        # Named base rather than ``super()``: InklingForCausalLM.load_weights
        # would run _load_mtp_weights over ``text_weights``, which has had the
        # ``model.mtp.*`` keys filtered out -- reporting a missing draft chain
        # that is in fact right there in ``weights``. The chain is loaded from
        # the full dict just below instead.
        #
        # Not DecoderModelForCausalLM either: the one-engine base's own
        # load_weights is what passes ``skip_modules=["draft_model"]``, without
        # which the generic loader tries to bind the chain from trunk weights.
        SpecDecOneEngineForCausalLM.load_weights(self, text_weights, weight_mapper=weight_mapper)
        self._load_mtp_weights(weights, weight_mapper)


def _text_sub_model_config(
    model_config: ModelConfig[InklingConfig],
) -> ModelConfig[InklingTextConfig]:
    """Build a text-only ``ModelConfig`` from the multimodal one, preserving the
    mapping / quant config so NVFP4 expert loading and TP sharding are intact."""
    text_config = model_config.pretrained_config.text_config
    text_model_config = copy.copy(model_config)
    text_model_config.pretrained_config = text_config
    return text_model_config
