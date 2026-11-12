# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: GLM-5.3-Flash (NVFP4) / sm_100 / default parallel layout, MTP fused into the forward.

GLM-5.3-Flash's language model: 45 layers, hidden 4096. Most layers are Kimi Delta Attention (KDA), a gated
linear-attention recurrence with per-request state; the others are pool-compressed sparse MLA (DSA). Four
hyper-connection residual streams (mHC) mix every sublayer's input and output. The first three MLPs are dense, the
others are 288-expert MoE (top-8, one shared expert). One MTP layer (``model.layers.45``) drafts for one-model MTP.
The routed experts and most linears are NVFP4 (ModelOpt); the modules the checkpoint excludes stay bf16.

**Baseline.** This target's forward is the built-in GLM-5.3-Flash text model (``modeling_glm5_next.py``), copied
here so it can be specialized in place: every step runs the generic path below, on stock modules and ops that have
no catalog entries yet. ``UNCERTIFIED_GENERIC_CALLS`` names the stock code it runs.

**MTP runs in the model.** This target serves one-model MTP-Eagle only. The causal LM builds its own drafter (the
checkpoint's MTP layer with the target's LM head and embedding) and runs the whole speculative step in its forward
(``Glm5NextMtpStep``): the verify logits, strict acceptance, the KDA state commit and the draft chain. No speculative
worker is built; the engine keeps what lies outside the forward (input packing, the speculative sampler, the KV
rewind). It computes what the stock MTP-Eagle worker computes for greedy and plain sampling; every other speculative
feature, and running without speculation, is served by the sibling split target
``glm_5_3_flash_nvfp4__sm_100__default``. Routing declines them here, and the constructor checks again
(``_check_speculation``). Routing also sends DWDP deployments to the split target, whose stock drafter declares the
MTP layer to DWDP as a repeated layer; the constructor does not see DWDP.

``default`` is the built-in model's parallel coverage: the weight shapes are computed from the engine's ``Mapping``
(tensor-parallel or expert-parallel MoE, attention DP), without pipeline or context parallelism.

**Text only.** The checkpoint is the vision-language wrapper. This target builds and loads no vision tower: the
weight mapper's audit allowlists the ``model.visual.*`` namespace. A request that carries multimodal input is
refused at admission (``validate_multimodal_request_data``), so a deployment that keeps the multimodal encoder
serves text requests and fails image and video requests alone.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
    Glm5NextMamba2Metadata,
    GlmKpoolBackendForwardArgs,
    GlmKpoolSparseParams,
    glm_kpool_cache_row_dim,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.glm5_next_weight_mapper import glm5_next_is_quantized
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer_set_repeated_for_next_layer
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm._torch.pyexecutor.sampler.ops import custom as fused_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_tail_kernels import strict_accept
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import async_tensor_h2d
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

from ..routing import fused_speculation_refused
from . import weights as _weights

# The GPU architecture this target IS. Routing will not send another one here, but a direct instantiation could,
# and the certification is per arch.
_SM = (10, 0)

#: Stock code the generic path runs outside the catalog, declared so it is not consumed silently: every
#: tensorrt_llm import of this module that computes (this target's unit tests check the list both ways, with
#: ``_glm5_next_generic_calls.py``). An entry leaves the list when a catalog entry replaces it.
UNCERTIFIED_GENERIC_CALLS = (
    # The causal LM's and the text model's base classes and the checkpoint's weight mapper.
    "tensorrt_llm._torch.models.modeling_utils.DecoderModelForCausalLM",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    "tensorrt_llm._torch.modules.decoder_layer.DecoderLayer",
    "tensorrt_llm._torch.models.checkpoints.base_weight_mapper.BaseWeightMapper",
    "tensorrt_llm._torch.models.checkpoints.hf.glm5_next_weight_mapper.glm5_next_is_quantized",
    "tensorrt_llm._torch.pyexecutor.config_utils.unwrap_glm5_next_text_config",
    "tensorrt_llm._utils.async_tensor_h2d",
    # KDA: the stock mixer and its recurrent-state metadata.
    "tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer.KimiKDALinearAttention",
    "tensorrt_llm._torch.modules.mamba.mamba2_metadata.Mamba2Metadata",
    "tensorrt_llm._torch.attention.backends.sparse.glm_kpool.Glm5NextMamba2Metadata",
    # Sparse MLA: the pool-compressed sparse attention backend and its indexer top-k.
    "tensorrt_llm._torch.attention.backends.interface.AttentionForwardArgs",
    "tensorrt_llm._torch.attention.backends.interface.AttentionInputType",
    "tensorrt_llm._torch.attention.backends.sparse.glm_kpool.GlmKpoolBackendForwardArgs",
    "tensorrt_llm._torch.attention.backends.sparse.glm_kpool.GlmKpoolSparseParams",
    "tensorrt_llm._torch.attention.backends.sparse.glm_kpool.glm_kpool_cache_row_dim",
    "tensorrt_llm._torch.attention.backends.utils.create_attention",
    "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE",
    "tensorrt_llm._torch.modules.top_k.TopK",
    "tensorrt_llm._torch.modules.top_k.TopKImplementation",
    # MoE: the router gate and the fused-MoE backend.
    "tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate",
    "tensorrt_llm._torch.moe.fused_moe.activation.SwigluActivation",
    "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe",
    # Hyper-connections, the dense MLPs, the linears, norms, embedding and collectives.
    "tensorrt_llm._torch.modules.mhc.hyper_connection.mHC",
    "tensorrt_llm._torch.modules.gated_mlp.GatedMLP",
    "tensorrt_llm._torch.modules.linear.Linear",
    "tensorrt_llm._torch.modules.linear.TensorParallelMode",
    "tensorrt_llm._torch.modules.rms_norm.RMSNorm",
    "tensorrt_llm._torch.modules.layer_norm.LayerNorm",
    "tensorrt_llm._torch.modules.embedding.Embedding",
    "tensorrt_llm._torch.distributed.AllReduce",
    "tensorrt_llm._torch.modules.multi_stream_utils.maybe_execute_in_parallel",
    # The in-model MTP step: target-token sampling, strict acceptance, the draft argmax's TP exchange, the
    # prompt-lookahead sentinel and the MoE load balancer's count of the MTP layer.
    "tensorrt_llm._torch.pyexecutor.sampler.ops.custom",
    "tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer",
    "tensorrt_llm._torch.speculative.spec_tail_kernels.strict_accept",
    "tensorrt_llm._torch.distributed.ops.allgather",
    "tensorrt_llm._torch.speculative.interface.INVALID_PROMPT_LOOKAHEAD_TOKEN",
    "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer_set_repeated_for_next_layer",
)


# ---------------------------------------------------------------------------
# Literal schedule vocabulary
# ---------------------------------------------------------------------------

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


@dataclass(frozen=True)
class Glm5NextSchedule:
    """The two literal per-layer dispatch lists, validated against each other."""

    attention: tuple[str, ...]
    mlp: tuple[str, ...]

    @property
    def num_layers(self) -> int:
        return len(self.attention)

    def attention_indices(self, kind: str) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.attention) if t == kind)

    def mlp_indices(self, kind: str) -> tuple[int, ...]:
        return tuple[int, ...](i for i, t in enumerate(self.mlp) if t == kind)


def resolve_glm5_next_schedule(config: PretrainedConfig) -> Glm5NextSchedule:
    """Validate the explicit attention and MLP schedules.

    Cross-check optional redundant fields; do not infer layer types from a cadence.
    """
    text = unwrap_glm5_next_text_config(config)
    num_layers = int(text.num_hidden_layers)

    attention = tuple(text.layer_types)
    mlp = tuple(text.mlp_layer_types)

    for name, values, allowed in (
        ("layer_types", attention, {LINEAR_ATTENTION, SPARSE_ATTENTION}),
        ("mlp_layer_types", mlp, {DENSE_MLP, SPARSE_MLP}),
    ):
        if len(values) != num_layers:
            raise ValueError(
                f"glm5_next {name} has {len(values)} entries but num_hidden_layers={num_layers}"
            )
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"glm5_next {name} contains unsupported entries {unknown}")

    schedule = Glm5NextSchedule(attention=attention, mlp=mlp)

    # Optional redundant schedule fields must agree with the explicit lists.
    linear_attn_config = getattr(text, "linear_attn_config", None) or {}
    kda_layers = linear_attn_config.get("kda_layers")
    full_attn_layers = linear_attn_config.get("full_attn_layers")
    if kda_layers is not None:
        if tuple(kda_layers) != schedule.attention_indices(LINEAR_ATTENTION):
            raise ValueError("glm5_next linear_attn_config.kda_layers disagrees with layer_types")
    if full_attn_layers is not None:
        if tuple(full_attn_layers) != schedule.attention_indices(SPARSE_ATTENTION):
            raise ValueError(
                "glm5_next linear_attn_config.full_attn_layers disagrees with layer_types"
            )

    # Validate the dense prefix without using it to infer the MLP schedule.
    first_k_dense = getattr(text, "first_k_dense_replace", None)
    if first_k_dense is not None:
        expected_dense = tuple(range(int(first_k_dense)))
        if schedule.mlp_indices(DENSE_MLP) != expected_dense:
            raise ValueError(
                f"glm5_next mlp_layer_types dense entries "
                f"{schedule.mlp_indices(DENSE_MLP)} disagree with "
                f"first_k_dense_replace={first_k_dense}"
            )

    return schedule


def glm5_next_tp_reduces(mapping: Mapping | None) -> bool:
    """Whether a TP branch output is a partial that needs one all-reduce.

    Under attention data parallelism every rank runs its own batch through
    replicated attention / dense weights, so nothing is reduced there (the
    fused MoE does its own dispatch/combine from ``all_rank_num_tokens``).
    """
    return mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp


def glm5_next_attention_mapping(mapping: Mapping | None) -> Mapping | None:
    """The Mapping the attention projections shard over: the model's, or a
    TP=1 view of it under attention DP (heads replicated per rank) -- the
    same remap :class:`~tensorrt_llm._torch.attention.mla.MLA` applies."""
    if mapping is None or not mapping.enable_attention_dp:
        return mapping
    return Mapping(
        world_size=mapping.pp_size * mapping.tp_size,
        tp_size=1,
        pp_size=mapping.pp_size * mapping.tp_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        enable_attention_dp=True,
    )


class Glm5NextAllReduce(nn.Module):
    """Use the configured TP collective for small messages and NCCL for large ones.

    The 512-token ONESHOT/NCCL crossover is tuned for TP4 BF16 messages;
    it is independent of the C++ workspace-capacity fallback. Other explicitly
    selected strategies apply at every size. Dispatch is fixed at graph capture.
    """

    # TP4/BF16 tuning uses fused collectives up to 512 tokens, then NCCL.
    SMALL_MAX_TOKENS = 512

    def __init__(
        self,
        mapping: Mapping,
        dtype: torch.dtype = torch.bfloat16,
        strategy: AllReduceStrategy = AllReduceStrategy.ONESHOT,
    ) -> None:
        super().__init__()
        from tensorrt_llm._torch.distributed import AllReduce

        self.small = AllReduce(mapping=mapping, strategy=strategy, dtype=dtype)
        self.large = (
            self.small
            if strategy != AllReduceStrategy.ONESHOT
            else AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return (self.small if x.shape[0] <= self.SMALL_MAX_TOKENS else self.large)(x)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _normalize_glm5_next_top_config(config: PretrainedConfig) -> None:
    """Expose text-config fields required by runtime capacity planning and MTP.

    Keep existing top-level values when the config is already flattened.
    """
    text = unwrap_glm5_next_text_config(config)
    if getattr(config, "num_hidden_layers", None) is None:
        config.num_hidden_layers = int(text.num_hidden_layers)
    if getattr(config, "torch_dtype", None) is None:
        config.torch_dtype = getattr(text, "torch_dtype", None) or torch.bfloat16
    # The engine reads the checkpoint's next-n layer count from the top-level
    # composite config (for example when it resolves the speculative config);
    # the field lives on ``text_config``.
    if getattr(config, "num_nextn_predict_layers", None) is None:
        config.num_nextn_predict_layers = int(getattr(text, "num_nextn_predict_layers", 0) or 0)


@dataclass(frozen=True)
class Glm5NextContextRows:
    """Device-side row schedule of the packed context tokens of one forward.

    Built once per forward from the host ``cu_seqlens``/``cached_lens`` and
    shared by every sparse layer, so the prefill path issues one kernel per
    stage for all context requests instead of a Python loop per request.
    ``positions[i]`` is packed token ``i``'s cache position, ``request_ids[i]``
    its executor request index (the batch's block-table row);
    ``final_positions``/``final_request_ids`` list the pool-final positions
    this forward completes (one per pool, so pool-key refreshes never race).
    """

    positions: torch.Tensor
    request_ids: torch.Tensor
    final_positions: torch.Tensor
    final_request_ids: torch.Tensor

    @classmethod
    def build(
        cls,
        cu_seqlens: Sequence[int],
        cached_lens: Sequence[int],
        kpool: int,
        device: torch.device,
    ) -> Glm5NextContextRows:
        positions: list[int] = []
        request_ids: list[int] = []
        finals: list[int] = []
        final_ids: list[int] = []
        for i in range(len(cu_seqlens) - 1):
            length = int(cu_seqlens[i + 1]) - int(cu_seqlens[i])
            cached = int(cached_lens[i])
            positions.extend(range(cached, cached + length))
            request_ids.extend([i] * length)
            first_final = cached + (-(cached + 1)) % kpool
            for pos in range(first_final, cached + length, kpool):
                finals.append(pos)
                final_ids.append(i)
        # Pinned non-blocking copies do not wait for the GPU work already queued.
        return cls(
            positions=async_tensor_h2d(positions, torch.long, device),
            request_ids=async_tensor_h2d(request_ids, torch.int32, device),
            final_positions=async_tensor_h2d(finals, torch.long, device),
            final_request_ids=async_tensor_h2d(final_ids, torch.int32, device),
        )


@dataclass(frozen=True)
class Glm5NextGenerationRows:
    """Device-side rows of the generation requests of one forward.

    Built once per forward from the live ``kv_lens`` and shared by every sparse
    layer, so each derived length tensor is one kernel per forward rather than
    per layer. ``positions`` is ``[B]`` for plain decode and ``[B, T]`` while a
    target verifies ``T - 1`` drafts; ``visible[i]`` is row ``i``'s visible
    prefix length and ``num_candidates[i]`` its complete-pool count (int32).
    Verification rows also get ``request_ids[i]``, row ``i``'s block-table row
    (int32, counted from ``first_table``, the batch row of the first generation
    request), so kernels index tables instead of copying them.
    """

    positions: torch.Tensor
    visible: torch.Tensor
    num_candidates: torch.Tensor
    request_ids: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        kv_lens: torch.Tensor,
        tokens_per_request: int,
        kpool: int,
        first_table: int,
    ) -> Glm5NextGenerationRows:
        request_ids = None
        if tokens_per_request <= 1:
            positions = kv_lens - 1
            visible = kv_lens
        else:
            device = kv_lens.device
            steps = torch.arange(tokens_per_request, device=device)
            positions = (kv_lens - tokens_per_request).unsqueeze(1) + steps
            visible = (positions + 1).reshape(-1)
            tables = torch.arange(
                first_table, first_table + kv_lens.shape[0], dtype=torch.int32, device=device
            )
            request_ids = tables.repeat_interleave(tokens_per_request)
        return cls(
            positions=positions,
            visible=visible,
            num_candidates=(visible // kpool).to(torch.int32),
            request_ids=request_ids,
        )


@dataclass
class Glm5NextRuntimeContext:
    """Per-forward schedule shared by all layers of a context-first packed batch.

    Sparse backends derive cache state from metadata; KDA consumes it directly.
    """

    num_contexts: int
    num_ctx_tokens: int
    num_generations: int
    ctx_cu_seqlens: list[int]
    cached_lens: list[int]
    #: Per-request visible lengths (cached + this step's tokens) as a device
    #: tensor. The decode path consumes ONLY device values so that captured
    #: CUDA graphs replay against prepare()-refreshed buffers rather than
    #: Python ints baked in at capture time.
    kv_lens: torch.Tensor
    #: The engine's prepared, typed attention metadata this context was
    #: derived from; the sparse layers' attention backend consumes it as the
    #: single source of cache state.
    metadata: AttentionMetadata
    #: Tokens per generation request in this step: ``1`` for plain decode,
    #: ``1 + runtime_draft_len`` while a speculative-decoding target verifies
    #: its draft tokens (the executor packs every generation request to the
    #: same width, so this is one host int, fixed at CUDA-graph capture).
    gen_tokens_per_request: int = 1
    #: Context-row schedules by pool size, built on first use and shared by
    #: all sparse layers of this forward (prefill only, never captured).
    ctx_rows_cache: dict[int, Glm5NextContextRows] = field(default_factory=dict)
    #: Generation-row lengths by pool size, built on first use and shared by
    #: all sparse layers of this forward (inside a captured graph, too).
    gen_rows_cache: dict[int, Glm5NextGenerationRows] = field(default_factory=dict)

    def context_rows(self, kpool: int, device: torch.device) -> Glm5NextContextRows:
        rows = self.ctx_rows_cache.get(kpool)
        if rows is None:
            rows = Glm5NextContextRows.build(
                self.ctx_cu_seqlens, self.cached_lens[: self.num_contexts], kpool, device
            )
            self.ctx_rows_cache[kpool] = rows
        return rows

    def generation_rows(self, kpool: int) -> Glm5NextGenerationRows:
        rows = self.gen_rows_cache.get(kpool)
        if rows is None:
            rows = Glm5NextGenerationRows.build(
                self.kv_lens[self.num_contexts :],
                self.gen_tokens_per_request,
                kpool,
                first_table=self.num_contexts,
            )
            self.gen_rows_cache[kpool] = rows
        return rows


def _glm5_gen_tokens_per_request(attn_metadata: AttentionMetadata, num_generations: int) -> int:
    """Tokens per generation request, from the metadata's host token counts.

    Plain decode has one; a speculative-decoding target verifying drafts has
    ``1 + runtime_draft_len``, uniformly across the generation rows (the
    executor pads every drafted request to the same width). Derived from
    host ints only, so it is a fixed shape parameter inside CUDA graphs.
    """
    if num_generations <= 0:
        return 1
    num_tokens = attn_metadata.num_tokens
    gen_tokens = int(num_tokens) - int(attn_metadata.num_ctx_tokens)
    if gen_tokens <= 0 or gen_tokens % num_generations:
        raise ValueError(
            f"glm5_next: {gen_tokens} generation tokens do not split evenly over "
            f"{num_generations} generation requests"
        )
    return gen_tokens // num_generations


def build_glm5_next_runtime_context(attn_metadata: AttentionMetadata) -> Glm5NextRuntimeContext:
    """Derive per-forward schedules from prepared GLM metadata.

    Prefill uses host schedules prepared before forward. Decode and MTP use
    kv_lens_cuda, which receives overlap corrections and draft rewinds on device.
    No cache tables or lengths are reconstructed inside forward.
    """
    if attn_metadata.kv_cache_manager is None:
        raise ValueError("glm5_next requires a kv cache manager; got None")
    mamba_metadata = attn_metadata.mamba_metadata
    assert isinstance(mamba_metadata, Glm5NextMamba2Metadata), (
        "glm5_next requires Glm5NextMamba2Metadata"
    )
    if mamba_metadata.glm_block_tables is None:
        raise RuntimeError(
            "glm5_next requires prepared glm_block_tables; call attn_metadata.prepare() "
            "with Glm5NextMamba2Metadata before eager execution or CUDA graph capture"
        )
    live_lengths = attn_metadata.kv_lens_cuda
    if live_lengths is None:
        raise ValueError("glm5_next requires prepared attn_metadata.kv_lens_cuda")
    batch = int(attn_metadata.seq_lens.shape[0])
    num_contexts = int(attn_metadata.num_contexts)
    num_generations = batch - num_contexts
    return Glm5NextRuntimeContext(
        num_contexts=num_contexts,
        num_ctx_tokens=int(attn_metadata.num_ctx_tokens),
        num_generations=num_generations,
        ctx_cu_seqlens=mamba_metadata.glm_ctx_cu_seqlens,
        cached_lens=mamba_metadata.glm_cached_lens_host,
        kv_lens=live_lengths[:batch],
        metadata=attn_metadata,
        gen_tokens_per_request=_glm5_gen_tokens_per_request(attn_metadata, num_generations),
    )


def glm5_next_text_quant_config(model_config: ModelConfig) -> ModelConfig:
    """Copy ``model_config`` with checkpoint quantization exclusions renamed for the text runtime.

    The checkpoint names text modules ``model.language_model.*``; the runtime omits the
    ``language_model`` level. Which modules stay unquantized, including the MTP layer, is
    taken from the checkpoint's ``exclude_modules`` as is.
    """
    quant = model_config.quant_config
    if quant is None or quant.exclude_modules is None:
        return model_config
    model_config = copy.copy(model_config)
    model_config.quant_config = copy.deepcopy(quant)
    model_config.quant_config.exclude_modules = [
        name.replace("model.language_model.", "model.", 1) for name in quant.exclude_modules
    ]
    return model_config


class Glm5NextForCausalLM(DecoderModelForCausalLM):
    """GLM-5.3-Flash decoder with its one-model MTP drafter.

    The whole speculative step runs in this model's forward
    (``Glm5NextMtpStep``): the verify logits, the acceptance, the KDA state
    commit and the draft chain. No speculative worker is built. The weight
    mapper routes text and MTP tensors and excludes the vision namespace.
    """

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # Avoid AUTO autotuning on TP decode; honor explicit user overrides.
        return {"allreduce_strategy": "ONESHOT"}

    @property
    def mamba_metadata_cls(self) -> type[Mamba2Metadata]:
        """Metadata with paged tables refreshed before CUDA graph replay."""
        return Glm5NextMamba2Metadata

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _check_speculation(model_config.spec_config)
        if (
            model_config.mapping.enable_attention_dp
            and model_config.mapping.enable_lm_head_tp_in_adp
        ):
            raise NotImplementedError(
                "glm5_next MTP with attention DP and a tensor-parallel LM head is not supported"
            )
        text_config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        model_config = glm5_next_text_quant_config(model_config)
        # tie_word_embeddings is false on this checkpoint, so lm_head is a real
        # weight rather than a view of the embedding; it is asserted, not assumed.
        if bool(getattr(text_config, "tie_word_embeddings", False)):
            raise ValueError(
                "glm5_next was validated with untied output embeddings; a tied "
                "checkpoint would need lm_head to alias embed_tokens"
            )
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        spec_config = model_config.spec_config
        super().__init__(
            Glm5NextModel(model_config),
            config=model_config,
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
        )
        # ``config`` is a base-class property (the top-level composite config);
        # the narrowed decoder contract lives here.
        self.text_config = text_config
        self.schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        # The engine reads these off a speculative model: the drafter (for the
        # loader's draft-weight paths, None-safe), a separate draft checkpoint's
        # config (none: the MTP layer is in this checkpoint), the worker (none:
        # this model runs the step itself) and whether the drafter has its own
        # KV cache (it shares the target's).
        self.spec_config = spec_config
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        # One-model MTP: the checkpoint's one MTP layer (layer_idx 45), run
        # ``max_draft_len`` times per step. It is aliased onto
        # ``model.layers[45:]`` so the checkpoint's ``model.layers.45.*`` keys
        # are placed by the same exact loader, the same projection swap, and
        # the same per-owner materialization.
        # The MoE load balancer counts the MTP layer once per draft step.
        moe_load_balancer_set_repeated_for_next_layer(spec_config.max_draft_len)
        mtp_layer = Glm5NextMTP(model_config, self.schedule.num_layers, self.model.aux_stream_dict)
        self.draft_model = Glm5NextDraftModel(mtp_layer, self.lm_head, self.model.embed_tokens)
        self.epilogue.append(self.draft_model)
        self.model.layers.append(mtp_layer)
        self.mtp_layers: tuple[nn.Module, ...] = (mtp_layer,)
        self.mtp_step = Glm5NextMtpStep(model_config.mapping, int(text_config.vocab_size))
        # Derived from the ModelConfig, never a constructor flag: the runtime's
        # AutoModelForCausalLM.from_config calls cls(model_config) and nothing
        # else, so this is the only place the decision can live.
        self.quantized = glm5_next_is_quantized(model_config)
        # Log the resolved backends from each worker process.
        attn_backends = sorted(
            {
                type(layer.self_attn.attn_backend).__name__
                for layer in self.model.layers
                if isinstance(layer.self_attn, Glm5NextSparseAttention)
            }
        )
        moe_backends = sorted(
            {
                layer.mlp.moe_backend_name
                for layer in self.model.layers
                if isinstance(layer.mlp, Glm5NextMoE)
            }
        )
        logger.info(
            f"glm5_next runtime stack: sparse_attention={attn_backends}, "
            f"moe_backend={moe_backends}, quantized={self.quantized}, "
            f"kv_cache_manager=V2 (Glm5NextCacheManager), "
            f"mtp_layers={len(self.mtp_layers)}"
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: Any = None,
        resource_manager: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """The decoder, then the whole speculative step.

        Returns the dict the speculative sampler reads (see ``Glm5NextMtpStep.__call__``). The verify logits are
        always the gathered rows, so ``return_context_logits`` does not apply, and the step holds no resources of
        its own.
        """
        del return_context_logits, resource_manager
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        if input_ids is None:
            input_ids = kwargs.get("orig_input_ids")
        if attn_metadata.padded_num_tokens is not None:
            input_ids = input_ids[: attn_metadata.num_tokens]
            position_ids = position_ids[..., : attn_metadata.num_tokens]
        logits = self.logits_processor.forward(
            hidden_states[spec_metadata.gather_ids], self.lm_head, attn_metadata, True
        )
        return self.mtp_step(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
        )

    def set_guided_decoder(self, guided_decoder: Any) -> bool:
        """Guided decoding is not supported: the speculative step runs no guided decoder."""
        del guided_decoder
        return False

    @property
    def num_hidden_layers(self) -> int:
        return self.schedule.num_layers

    def apply_quant_config_exclude_modules(self) -> None:
        """Base-class exclusion by runtime name, extended to the fused Linears.

        A fused projection (``FUSED_LINEARS`` on its owner) is spelled in the
        checkpoint -- and in ``modules_to_not_convert`` -- as its separate
        source tensors, so the verdict is taken from the sources: all excluded
        keeps the fused module BF16, none excluded keeps it quantized, a mix
        is not representable and is rejected.
        """
        super().apply_quant_config_exclude_modules()
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return
        new_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for name, module in self.named_modules():
            for fused_attr, sources in getattr(type(module), "FUSED_LINEARS", ()):
                fused = module.get_submodule(fused_attr)
                if getattr(fused, "quant_config", None) is None:
                    continue
                prefix = f"{name}." if name else ""
                verdicts = [
                    quant_config.is_module_excluded_from_quantization(prefix + src)
                    for src in sources
                ]
                if all(verdicts):
                    fused.quant_config = new_config
                    fused._weights_created = False
                elif any(verdicts):
                    raise ValueError(
                        f"glm5_next {prefix}{fused_attr}: fused sources disagree on "
                        f"quantization ({dict(zip(sources, verdicts))}); fusing tensors "
                        "with different quantization is not representable"
                    )

    def infer_max_seq_len(self) -> int:
        """Use the text config's context limit directly; this model has no RoPE scaling."""
        return int(self.text_config.max_position_embeddings)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Use V2 for the shared latent-KV, pool-indexer and recurrent-state cache."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None) -> str:
        """Use Python NIXL to transfer the hybrid cache, including recurrent state."""
        return "PYTHON"


# ---------------------------------------------------------------------------
# KDA linear attention
# ---------------------------------------------------------------------------


class Glm5NextLinearAttention(KimiKDALinearAttention):
    """Configure the shared KDA mixer for GLM's low-rank output gate and FP32 gates.

    Recurrence, convolution and speculative replay use shared KDA kernels.
    The loader shards local-width projections by head range; A_log and dt_bias
    retain the checkpoint's FP32 precision.
    """

    #: Checkpoint tensors sharded by this rank's head *channel* range on dim 0.
    _CHANNEL_ROW_SHARDED = frozenset(
        (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "f_b_proj.weight",
            "g_b_proj.weight",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "dt_bias",
        )
    )
    #: Checkpoint tensors sharded by this rank's head range on dim 0.
    _HEAD_ROW_SHARDED = frozenset(("b_proj.weight", "A_log"))
    #: Replicated on every rank.
    _REPLICATED = frozenset(("f_a_proj.weight", "g_a_proj.weight", "o_norm.weight"))

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Mapping | None = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.ONESHOT,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        del dtype  # the shared mixer is bf16 (fp32 gate parameters and pools)
        # The checkpoint's ``linear_attn_config`` does not name the gate rank:
        # GLM-5.3-Flash's output gate is always the low-rank ``g_a``/``g_b``
        # pair, which the shared mixer reads from the same key Kimi K3 uses.
        config = copy.copy(config)
        config.linear_attn_config = dict(config.linear_attn_config)
        config.linear_attn_config.setdefault("use_full_rank_gate", False)
        linear = config.linear_attn_config
        # The mixer's own row-parallel o_proj AllReduce (fixed strategy; the
        # size-aware Glm5NextAllReduce used elsewhere is a perf option, not a
        # correctness requirement, and AUTO's autotuner raced at TP4 decode).
        super().__init__(
            config,
            layer_idx,
            mapping=mapping,
            allreduce_strategy=allreduce_strategy,
            aux_stream=aux_stream,
        )
        # Preserve checkpoint FP32 gate parameters. MetaInitMode rejects detach
        # on meta tensors, so create empty FP32 parameters on that branch.
        for name in ("A_log", "dt_bias"):
            param = getattr(self, name)
            data = (
                torch.empty(param.shape, dtype=torch.float32, device=param.device)
                if param.is_meta
                else param.detach().float()
            )
            setattr(self, name, nn.Parameter(data, requires_grad=False))
        self.tp_size = self._kda_tp_size
        self.tp_rank = self._kda_tp_rank
        self.total_num_heads = int(linear["num_heads"])

    # -- tensor-parallel ownership (consumed by the exact-placement loader) --

    def kda_head_range(self) -> tuple[int, int]:
        """This rank's contiguous ``[start, end)`` on the head axis."""
        return self.tp_rank * self.num_heads, (self.tp_rank + 1) * self.num_heads

    def kda_channel_range(self) -> tuple[int, int]:
        """This rank's contiguous ``[start, end)`` on the ``heads * head_dim`` axis."""
        start, end = self.kda_head_range()
        return start * self.head_dim, end * self.head_dim

    def shard_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Slice one full-width checkpoint tensor (module-relative ``name``)
        down to this rank's head range."""
        if self.tp_size == 1 or name in self._REPLICATED:
            return tensor
        if name in self._CHANNEL_ROW_SHARDED:
            start, end = self.kda_channel_range()
            return tensor[start:end]
        if name in self._HEAD_ROW_SHARDED:
            start, end = self.kda_head_range()
            return tensor[start:end]
        if name == "o_proj.weight":
            start, end = self.kda_channel_range()
            return tensor[:, start:end]
        raise ValueError(f"glm5_next KDA has no tensor-parallel ownership for {name!r}")

    def finalize_weights(self) -> None:
        """Post-load fused decode projections and kernel constants, as the
        Kimi K3 loader does (the fused shards become the parameters' storage,
        so nothing is duplicated)."""
        self.finalize_decode_weights()
        self._build_mtp_conv_weights()


def _glm5_linear_kwargs(
    model_config: ModelConfig | None, mapping: Mapping | None
) -> dict[str, Any]:
    """Build BF16 or block-FP8 Linear arguments with attention TP/DP mapping.

    The base model applies quantization exclusions by runtime module name.
    Block-FP8 projections use the CuTe DSL GEMM path (disable_deep_gemm=True).
    """
    return {
        "bias": False,
        "dtype": torch.bfloat16,
        "mapping": glm5_next_attention_mapping(
            model_config.mapping if model_config is not None else mapping
        ),
        "quant_config": model_config.get_quant_config() if model_config is not None else None,
        "skip_create_weights_in_init": (
            model_config.skip_create_weights_in_init if model_config is not None else False
        ),
        "allreduce_strategy": (
            model_config.allreduce_strategy
            if model_config is not None
            else AllReduceStrategy.ONESHOT
        ),
        "disable_deep_gemm": True,
    }


class Glm5NextIndexer(nn.Module):
    """Select complete key pools and append the incomplete causal tail.

    The backend caches [key | compression gate | pooled key] per token, storing
    pooled keys at each pool's first row. Replicated projections and scoring heads
    produce the same selection on each TP rank without a score all-reduce.
    Packed requests use visible lengths instead of HF's left-padding mask.
    """

    #: ``wk``, the pool-compress gate and ``weights_proj`` all read the same
    #: hidden state and are BF16: the runtime holds them as one fused Linear
    #: (``[k | gate | head weights]`` rows, in this order); the loader
    #: concatenates the checkpoint's three tensors into it.
    FUSED_LINEARS = (("wk_gate_wp", ("wk", "index_kpool_compress_gate", "weights_proj")),)

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Mapping | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        del dtype  # bf16 module (the checkpoint publishes every indexer tensor in bf16)
        if model_config is not None:
            mapping = model_config.mapping
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_n_heads = int(config.index_n_heads)
        self.tp_size = mapping.tp_size if mapping is not None else 1
        self.tp_rank = mapping.tp_rank if mapping is not None else 0
        if self.total_n_heads % self.tp_size:
            raise ValueError(
                f"glm5_next indexer has {self.total_n_heads} scoring heads, not "
                f"divisible by tp_size {self.tp_size}"
            )
        self.n_heads = self.total_n_heads
        self.head_dim = int(config.index_head_dim)
        self.index_topk = int(config.index_topk)
        self.index_kpool = int(config.index_kpool)
        self.always_select_tail = bool(config.index_kpool_always_select_tail)
        self.softmax_scale = self.head_dim**-0.5
        self.head_mix_scale = self.total_n_heads**-0.5
        self.select_k = self.index_topk // self.index_kpool
        self.output_width = self.index_topk + (
            self.index_kpool - 1 if self.always_select_tail else 0
        )

        # Replicated projections (no tensor_parallel_mode).
        lin_kwargs = _glm5_linear_kwargs(model_config, mapping)
        self.wq_b = Linear(
            int(config.q_lora_rank), self.total_n_heads * self.head_dim, **lin_kwargs
        )
        self.wk_gate_wp = Linear(
            self.hidden_size, 2 * self.head_dim + self.total_n_heads, **lin_kwargs
        )
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6, dtype=torch.bfloat16)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.bfloat16)
        )
        # Reuse DSA's TopK, bounded by each request's candidate count.
        # Torch handles the FP32 fallback without caller-owned CUDA radix scratch.
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
        from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

        self.pool_top_k = TopK(
            self.select_k,
            prefill_implementation=TopKImplementation.TORCH,
            decode_implementation=(
                TopKImplementation.CUTE_DSL_RADIX
                if IS_CUTLASS_DSL_AVAILABLE
                else TopKImplementation.TORCH
            ),
        )

    @property
    def cache_state_dim(self) -> int:
        """Width of a cached [key | compression gate | pooled key] row."""
        return glm_kpool_cache_row_dim(self.head_dim)

    def project_state(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(packed [k(head_dim) | gate(head_dim)], head weights [n_heads])``
        from the one fused input GEMM."""
        out = self.wk_gate_wp(hidden_states)
        hd = self.head_dim
        packed = torch.cat([self.k_norm(out[:, :hd]), out[:, hd : 2 * hd]], dim=-1)
        return packed, out[:, 2 * hd :]


class Glm5NextSparseAttention(nn.Module):
    """NoPE sparse MLA with a pool-compressed indexer.

    Owns projections, sparse selection and MLA weight absorption. The TRTLLM
    sparse backend owns paged-cache access and attention, receiving metadata and
    selected rows through AttentionForwardArgs. BF16 kv_b_proj weights are
    reassociated directly, without dequantization.

    TP shards query heads and output projections; latent and indexer state stay
    replicated. The output projection returns a partial for one explicit
    reduction. Attention DP instead keeps all heads local with no TP reduction.
    """

    #: The two low-rank input projections are both block-FP8 and replicated:
    #: one fused Linear (``[q_a | kv_a]`` rows), one activation quantization
    #: -- the DeepSeek-V3 ``fuse_qkv_a_proj`` layout under the same name.
    FUSED_LINEARS = (("kv_a_proj_with_mqa", ("q_a_proj", "kv_a_proj_with_mqa")),)

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: str = "TRTLLM",
        mapping: Mapping | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if model_config is not None:
            mapping = model_config.mapping
        # Heads are sharded over TP, or replicated per rank under attention DP.
        attn_mapping = glm5_next_attention_mapping(mapping)
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.tp_size = attn_mapping.tp_size if attn_mapping is not None else 1
        self.tp_rank = attn_mapping.tp_rank if attn_mapping is not None else 0
        if self.total_num_heads % self.tp_size:
            raise ValueError(
                f"glm5_next sparse MLA has {self.total_num_heads} heads, not divisible "
                f"by tp_size {self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "glm5_next text attention is fully NoPE; a non-zero qk_rope_head_dim "
                f"({self.qk_rope_head_dim}) is not supported"
            )
        self.scaling = self.qk_head_dim**-0.5
        eps = float(config.rms_norm_eps)

        lin_kwargs = _glm5_linear_kwargs(model_config, mapping)
        # Low-rank latents replicated, per-head maps column-sharded, output
        # row-sharded and returned as a partial (the module runs the branch's
        # one reduction through Glm5NextAllReduce).
        self.kv_a_proj_with_mqa = Linear(
            self.hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            **lin_kwargs,
        )
        self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank, eps=eps, dtype=dtype)
        self.q_b_proj = Linear(
            self.q_lora_rank,
            self.total_num_heads * self.qk_head_dim,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            **lin_kwargs,
        )
        self.kv_a_layernorm = RMSNorm(hidden_size=self.kv_lora_rank, eps=eps, dtype=dtype)
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            **lin_kwargs,
        )
        self.o_proj = Linear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=False,
            **lin_kwargs,
        )
        self.tp_all_reduce = (
            Glm5NextAllReduce(
                mapping,
                strategy=model_config.allreduce_strategy
                if model_config is not None
                else AllReduceStrategy.ONESHOT,
            )
            if glm5_next_tp_reduces(mapping)
            else None
        )
        self.indexer = Glm5NextIndexer(
            config, layer_idx, mapping=mapping, model_config=model_config
        )

        # The production sparse-MLA backend, selected through the standard
        # dispatch (`get_attention_backend(attn_backend, sparse_params)` via
        # `create_attention`). It consumes absorbed latent-space queries, so
        # its head_dim is kv_lora_rank, and the latent cache is MQA-style
        # (one KV head). AttentionBackend is not an nn.Module: this is a
        # plain attribute, invisible to state_dict/loading.
        self.attn_backend = create_attention(
            attn_backend,
            layer_idx,
            num_heads=self.num_heads,
            head_dim=self.kv_lora_rank,
            num_kv_heads=1,
            dtype=dtype,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=0,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
            rope_append=False,
            sparse_params=GlmKpoolSparseParams(
                index_topk=self.indexer.index_topk,
                index_kpool=self.indexer.index_kpool,
                index_always_select_tail=self.indexer.always_select_tail,
                index_head_dim=self.indexer.head_dim,
            ),
        )

    def project_inputs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q_resid [T, q_lora], latent [T, kv_lora])`` from the fused GEMM."""
        qa_kva = self.kv_a_proj_with_mqa(hidden_states)
        q_resid = self.q_a_layernorm(qa_kva[:, : self.q_lora_rank])
        latent = self.kv_a_layernorm(qa_kva[:, self.q_lora_rank :])
        return q_resid, latent

    def _project_attention_and_indexer(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q_resid, latent, packed [k | gate], index_weights)`` -- the two
        fused input GEMMs (attention and indexer)."""
        q_resid, latent = self.project_inputs(hidden_states)
        packed, weights = self.indexer.project_state(hidden_states)
        return q_resid, latent, packed, weights

    def absorbed_kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head absorbed views of ``kv_b_proj``: ``(w_k, w_v_t)``.

        ``w_k`` is ``[H, qk_nope, kv_lora]`` (queries -> latent space) and
        ``w_v_t`` is ``[H, kv_lora, v_head]`` (latent attention output -> V
        space). Score and value absorption are exact reassociations of the
        unabsorbed math: ``q . (W_k @ c) == (W_k^T @ q) . c`` and
        ``sum_j p_j (W_v @ c_j) == W_v @ sum_j p_j c_j``. Contiguous copies
        are cached after weight loading (keyed on the weight's data pointer
        and version) so the per-step bmm never re-materializes them; the
        cache is a plain tuple attribute, invisible to ``state_dict``.
        """
        weight = self.kv_b_proj.weight
        key = (weight.data_ptr(), weight._version)
        cached = self.__dict__.get("_absorbed_kv_b_cache")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        if weight.dtype != torch.bfloat16:
            raise ValueError(
                "glm5_next sparse MLA absorption expects the BF16-excluded "
                f"kv_b_proj weight, got dtype {weight.dtype}"
            )
        per_head = weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_k = per_head[:, : self.qk_nope_head_dim, :].contiguous()
        w_v_t = per_head[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        self._absorbed_kv_b_cache = (key, w_k, w_v_t)
        return w_k, w_v_t

    def absorb_query(self, query: torch.Tensor) -> torch.Tensor:
        """``[T, H, qk_nope] -> [T, H, kv_lora]`` (fully NoPE: q is all nope)."""
        w_k, _ = self.absorbed_kv_b()
        return torch.bmm(query.transpose(0, 1), w_k).transpose(0, 1)

    def project_output_latent(self, out_latent: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """Flat latent attention output -> V space -> ``o_proj``. ``[T, hidden]``.

        ``out_latent`` is the backend's base-contract result
        ``[T, num_heads * kv_lora]``; the per-head view for the absorbed V
        projection is this module's concern, not the backend boundary's.
        """
        _, w_v_t = self.absorbed_kv_b()
        tokens = out_latent.shape[0]
        per_head = out_latent.view(tokens, self.num_heads, self.kv_lora_rank)
        # Write the per-head products straight into [T, H, v_head] for o_proj.
        out = per_head.new_empty(tokens, self.num_heads, self.v_head_dim)
        torch.bmm(per_head.transpose(0, 1), w_v_t, out=out.transpose(0, 1))
        out = self.o_proj(out.view(tokens, -1))
        if not reduce or self.tp_all_reduce is None:
            return out
        return self.tp_all_reduce(out)

    def _select_and_attend_paged(
        self,
        q_resid: torch.Tensor,
        query: torch.Tensor,
        index_weights: torch.Tensor,
        visible: torch.Tensor,
        metadata: AttentionMetadata,
        input_type: AttentionInputType,
        rows_per_request: int = 1,
        reduce: bool = True,
        request_ids: torch.Tensor | None = None,
        max_visible: int | None = None,
        num_candidates: torch.Tensor | None = None,
        context_spans: Sequence[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Score pools, select members and tail, then attend to paged latent rows.

        visible gives each query's prefix length. Packed context uses request_ids;
        verification uses rows_per_request and request_ids. Single-token decode
        reads live lengths from metadata. All device work uses fixed buffer shapes
        for graph replay.
        max_visible is an optional host bound on visible that limits scoring work;
        num_candidates, if given, is visible // index_kpool as int32. Packed context
        passes context_spans, each request's (rows, visible bound), so scoring
        runs in calls of bounded memory.
        """
        indexer = self.indexer
        rows = q_resid.shape[0]
        q_index = indexer.wq_b(q_resid).view(rows, indexer.n_heads, indexer.head_dim)
        # Generation rows read their visible length from the metadata unless
        # each request contributes several rows (context or verification).
        plain_decode = request_ids is None and rows_per_request == 1
        kv_lens = None if plain_decode else visible
        num_cand = (
            (visible // indexer.index_kpool).to(torch.int32)
            if num_candidates is None
            else num_candidates
        )
        selected = torch.empty(rows, indexer.select_k, dtype=torch.int32, device=q_index.device)
        # Each call scores and selects its own rows.
        if context_spans is None:
            calls = [(slice(None), max_visible, None)]
        else:
            calls = [
                (slice(start, end), max_kv_len, (first, last))
                for start, end, first, last, max_kv_len in self.attn_backend.plan_context_scores(
                    metadata, context_spans
                )
            ]
        for part, max_kv_len, table_range in calls:
            scores = self.attn_backend.score_pools(
                q_index[part],
                index_weights[part],
                metadata,
                q_scale=indexer.softmax_scale,
                w_scale=indexer.head_mix_scale,
                kv_lens=None if kv_lens is None else kv_lens[part],
                rows_per_request=rows_per_request,
                request_ids=None if request_ids is None else request_ids[part],
                max_kv_len=max_kv_len,
                table_range=table_range,
            )
            indexer.pool_top_k(
                scores,
                selected[part],
                is_prefill=False,
                sequence_lengths=num_cand[part],
                scan_lengths=num_cand[part],
            )
            # Release this call's scores before the next call allocates its own.
            del scores
        topk_rows = self.attn_backend.expand_selection(
            selected,
            metadata,
            kv_lens=kv_lens,
            request_ids=request_ids,
        )
        out_latent = self.attn_backend.forward(
            self.absorb_query(query),
            None,
            None,
            metadata,
            AttentionForwardArgs(
                attention_input_type=input_type,
                sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=topk_rows),
            ),
        )
        return self.project_output_latent(out_latent, reduce)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Sequence[int],
        cached_lens: Sequence[int],
        metadata: AttentionMetadata,
        reduce: bool = True,
        ctx_rows_fn: Callable[[int, torch.device], Glm5NextContextRows] | None = None,
    ) -> torch.Tensor:
        """Process all packed context rows, including continuation chunks.

        cached_lens counts tokens before this chunk. Each query attends to its full
        visible prefix. Row schedules identify request block tables and completed
        pools; ctx_rows_fn caches these schedules across layers of one forward.
        """
        kpool = self.indexer.index_kpool
        device = hidden_states.device
        if ctx_rows_fn is not None:
            rows = ctx_rows_fn(kpool, device)
        else:
            rows = Glm5NextContextRows.build(cu_seqlens, cached_lens, kpool, device)
        q_resid, latent, packed, index_weights = self._project_attention_and_indexer(hidden_states)
        self.attn_backend.append_paged_state(
            latent, packed, rows.positions, metadata, request_ids=rows.request_ids
        )
        # Pool keys for every pool this forward completes (one program per
        # pool: only pool-final positions). A pool left incomplete at a chunk
        # end is refreshed by whichever later write completes it.
        if rows.final_positions.shape[0]:
            self.attn_backend.update_pool_keys(
                rows.final_positions,
                self.indexer.index_kpool_compress_ape,
                metadata,
                request_ids=rows.final_request_ids,
            )
        query = self.q_b_proj(q_resid).view(-1, self.num_heads, self.qk_head_dim)
        spans = [
            (
                int(cu_seqlens[i + 1]) - int(cu_seqlens[i]),
                int(cached_lens[i]) + int(cu_seqlens[i + 1]) - int(cu_seqlens[i]),
            )
            for i in range(len(cu_seqlens) - 1)
        ]
        max_visible = max(visible for _, visible in spans)
        return self._select_and_attend_paged(
            q_resid,
            query,
            index_weights,
            rows.positions + 1,
            metadata,
            AttentionInputType.context_only,
            reduce=reduce,
            request_ids=rows.request_ids,
            max_visible=max_visible,
            context_spans=spans,
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_lens: torch.Tensor,
        metadata: AttentionMetadata,
        reduce: bool = True,
        runtime_ctx: Glm5NextRuntimeContext | None = None,
    ) -> torch.Tensor:
        """Decode one token per request using device-resident visible lengths.

        kv_lens includes the new token, whose position is kv_lens - 1. Shapes depend
        only on buffer geometry; metadata preparation refreshes device tensors before
        CUDA graph replay. This path must not synchronize lengths to the host.

        The backend masks invalid cache positions. Every query is covered by either
        the incomplete tail pool or its final complete pool.
        """
        batch = hidden_states.shape[0]
        indexer = self.indexer
        rows = (
            Glm5NextGenerationRows.build(
                kv_lens, 1, indexer.index_kpool, first_table=int(metadata.num_contexts)
            )
            if runtime_ctx is None
            else runtime_ctx.generation_rows(indexer.index_kpool)
        )
        positions = rows.positions  # [B]
        q_resid, latent, packed, index_weights = self._project_attention_and_indexer(hidden_states)
        self.attn_backend.append_paged_state(
            latent.unsqueeze(1), packed.unsqueeze(1), positions.unsqueeze(1), metadata
        )
        # Fused indexer path: refresh the pool the new token belongs to, then
        # score / top-k / expand over the cached pool keys.
        self.attn_backend.update_pool_keys(positions, indexer.index_kpool_compress_ape, metadata)
        query = self.q_b_proj(q_resid).view(batch, self.num_heads, self.qk_head_dim)
        return self._select_and_attend_paged(
            q_resid,
            query,
            index_weights,
            kv_lens,
            metadata,
            AttentionInputType.generation_only,
            reduce=reduce,
            num_candidates=rows.num_candidates,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        runtime_ctx: Glm5NextRuntimeContext | None = None,
    ) -> torch.Tensor:
        """Process a packed batch, splitting phases only inside attention.

        Reuse the model's per-forward schedules and row cache across layers.
        Reduce the complete output once, including context/verify mixtures.
        """
        ctx = (
            runtime_ctx
            if runtime_ctx is not None
            else build_glm5_next_runtime_context(attn_metadata)
        )
        parts = []
        if ctx.num_contexts:
            parts.append(
                self.forward_prefill(
                    hidden_states[: ctx.num_ctx_tokens],
                    cu_seqlens=ctx.ctx_cu_seqlens,
                    cached_lens=ctx.cached_lens[: ctx.num_contexts],
                    metadata=ctx.metadata,
                    ctx_rows_fn=ctx.context_rows,
                    reduce=False,
                )
            )
        if ctx.num_generations:
            generation = hidden_states[ctx.num_ctx_tokens :]
            lengths = ctx.kv_lens[ctx.num_contexts :]
            if ctx.gen_tokens_per_request == 1:
                out = self.forward_decode(
                    generation,
                    kv_lens=lengths,
                    metadata=ctx.metadata,
                    reduce=False,
                    runtime_ctx=ctx,
                )
            else:
                out = self.forward_verify(
                    generation,
                    kv_lens=lengths,
                    metadata=ctx.metadata,
                    tokens_per_request=ctx.gen_tokens_per_request,
                    reduce=False,
                    runtime_ctx=ctx,
                )
            parts.append(out)
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out if self.tp_all_reduce is None else self.tp_all_reduce(out)

    def forward_verify(
        self,
        hidden_states: torch.Tensor,
        kv_lens: torch.Tensor,
        metadata: AttentionMetadata,
        tokens_per_request: int,
        reduce: bool = True,
        runtime_ctx: Glm5NextRuntimeContext | None = None,
    ) -> torch.Tensor:
        """Verify packed [batch * tokens_per_request, hidden] generation rows.

        Request tokens occupy positions kv_lens - T through kv_lens - 1. Each query
        uses its own visible prefix, excluding later drafts. Rejected draft entries
        are overwritten after lengths rewind. Shapes depend only on buffer geometry
        and T, allowing CUDA graph replay with refreshed lengths and block tables.
        """
        tokens_per_request = int(tokens_per_request)
        if tokens_per_request <= 1:
            return self.forward_decode(
                hidden_states, kv_lens, metadata, reduce=reduce, runtime_ctx=runtime_ctx
            )
        num_tokens = hidden_states.shape[0]
        batch = num_tokens // tokens_per_request
        if batch * tokens_per_request != num_tokens or kv_lens.shape[0] != batch:
            raise ValueError(
                f"glm5_next sparse verify: {num_tokens} tokens for {kv_lens.shape[0]} requests "
                f"at {tokens_per_request} tokens per request"
            )
        kpool = self.indexer.index_kpool
        rows = (
            Glm5NextGenerationRows.build(
                kv_lens, tokens_per_request, kpool, first_table=int(metadata.num_contexts)
            )
            if runtime_ctx is None
            else runtime_ctx.generation_rows(kpool)
        )
        positions = rows.positions  # [B, T]
        q_resid, latent, packed, index_weights = self._project_attention_and_indexer(hidden_states)
        self.attn_backend.append_paged_state(
            latent.view(batch, tokens_per_request, -1),
            packed.view(batch, tokens_per_request, -1),
            positions,
            metadata,
        )
        # Keep the cached pool keys current, in position order, so a pool
        # spanning several drafts ends up with all of its visible members.
        self.attn_backend.update_pool_keys(
            positions, self.indexer.index_kpool_compress_ape, metadata
        )
        query = self.q_b_proj(q_resid).view(num_tokens, self.num_heads, self.qk_head_dim)
        # Row i of request b sees its own prefix: visible = position + 1.
        return self._select_and_attend_paged(
            q_resid,
            query,
            index_weights,
            rows.visible,
            metadata,
            AttentionInputType.generation_only,
            rows_per_request=tokens_per_request,
            reduce=reduce,
            request_ids=rows.request_ids,
            num_candidates=rows.num_candidates,
        )


class Glm5NextGate(DeepseekV3Gate):
    """DeepSeek noaux_tc routing with FP32 logits and correction bias.

    The checkpoint router weight is BF16, so the base BF16 GEMM with FP32 output
    forms the same products as FP32 math. Preserve small inter-expert score
    differences when adding the correction bias; rounding it to BF16 can change
    expert selection.
    """

    def __init__(self, config: PretrainedConfig, moe_backend: str = "CUTLASS") -> None:
        if str(getattr(config, "scoring_func", "sigmoid")) != "sigmoid":
            raise ValueError(f"glm5_next expects sigmoid scoring, got {config.scoring_func!r}")
        super().__init__(
            int(config.hidden_size),
            int(config.n_routed_experts),
            top_k=int(config.num_experts_per_tok),
            n_group=int(getattr(config, "n_group", 1) or 1),
            topk_group=int(getattr(config, "topk_group", 1) or 1),
            routed_scaling_factor=float(config.routed_scaling_factor),
            dtype=torch.bfloat16,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=moe_backend,
        )
        self.norm_topk_prob = bool(config.norm_topk_prob)
        # FP32 regardless of the MoE backend (the base class picks bf16 for
        # TRTLLM); the fused routing kernels consume the FP32 tensor directly.
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(int(config.n_routed_experts), dtype=torch.float32), requires_grad=False
        )


class Glm5NextMoE(nn.Module):
    """Fused routed experts with a shared GatedMLP expert.

    Uses FP32 DeepSeek routing and clamped SwiGLU. Under TP, routed and shared
    expert partials are summed before one reduction. Under attention DP, fused
    MoE owns dispatch/combine and the shared expert is replicated.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        model_config: ModelConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.num_experts = int(config.n_routed_experts)
        self.swiglu_limit = float(config.swiglu_limit)
        self.gate = Glm5NextGate(config, moe_backend=str(model_config.moe_backend or "CUTLASS"))
        inter = self.moe_intermediate_size
        self.moe_backend_name: str | None = None
        # The fused routing kernel always normalizes the gathered top-8
        # weights.
        if not self.gate.norm_topk_prob:
            raise ValueError(
                "glm5_next production MoE requires norm_topk_prob=True "
                "(the DeepSeek noaux_tc routing kernel always normalizes)"
            )
        from tensorrt_llm._torch.moe.fused_moe.activation import SwigluActivation
        from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe

        # The gate's own routing method (DeepSeek noaux_tc; the bias is
        # fetched per call so it follows the parameter through
        # materialization onto its final device).
        routing = self.gate.routing_method
        # Routed experts use the checkpoint's quantization; excluded layers
        # (including the NVFP4 checkpoint's BF16 MTP layer) stay unquantized.
        quant = model_config.get_quant_config()
        experts_quant = (
            QuantConfig(quant_algo=quant.quant_algo, group_size=quant.group_size)
            if quant is not None
            and quant.quant_algo is not None
            and not quant.is_module_excluded_from_quantization(
                f"model.layers.{layer_idx}.mlp.experts"
            )
            else QuantConfig()
        )
        self.experts = create_moe(
            routing_method=routing,
            num_experts=self.num_experts,
            hidden_size=hidden,
            intermediate_size=inter,
            dtype=dtype,
            reduce_results=False,
            model_config=model_config,
            override_quant_config=experts_quant,
            layer_idx=layer_idx,
            activation=SwigluActivation(clamp=self.swiglu_limit),
        )
        backend = getattr(self.experts, "backend", self.experts)
        self.moe_backend_name = type(backend).__name__
        # Four-rank composition: the fused layer runs
        # with reduce_results=False, so its output is a rank partial --
        # a K-dim partial per expert in the TP4 layout (moe_tp_size=4),
        # a local-expert partial sum in the TP4/EP4 layout (moe_ep_size=4)
        # -- and this module owns the exactly-one reduction that combines
        # it together with the TP-sharded shared-expert partial (the
        # DeepSeek-V3 composition).
        self.mapping = model_config.mapping
        self.use_dp = bool(self.mapping.enable_attention_dp)
        self.shared_expert_stream = None if self.use_dp else aux_stream
        self.shared_expert_start = torch.cuda.Event() if self.shared_expert_stream else None
        self.shared_expert_done = torch.cuda.Event() if self.shared_expert_stream else None
        # Reduce routed and TP-sharded shared-expert partials once. Under
        # attention DP the fused layer combines across ranks itself and the
        # shared expert is replicated, so there is nothing to reduce here.
        self.moe_all_reduce = (
            Glm5NextAllReduce(self.mapping, strategy=model_config.allreduce_strategy)
            if glm5_next_tp_reduces(self.mapping)
            else None
        )
        # Shared expert: TP-sharded over ``model_config.mapping`` with
        # ``reduce_output=False``, so its output is a rank partial summed with
        # the routed partial before this module's single all-reduce
        # (replicated, ``overridden_tp_size=1``, under attention DP).
        self.shared_experts = GatedMLP(
            hidden_size=hidden,
            intermediate_size=self.moe_intermediate_size * int(config.n_shared_experts),
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if self.use_dp else None,
            reduce_output=False,
            layer_idx=layer_idx,
            is_shared_expert=True,
            swiglu_limit=self.swiglu_limit,
            # Same GEMM the rest of the model's block-FP8 projections run
            # (fp8_quantize_1x128 + the CuTe DSL Blackwell kernel).
            disable_deep_gemm=True,
        )

    def forward(
        self, x: torch.Tensor, all_rank_num_tokens: list[int] | None = None
    ) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden_size)
        # The fused layer routes internally from the FP32 logits and runs
        # FC1 -> clamped SwiGLU -> FC2 in one backend call; one code path
        # serves prefill and decode with no host-dependent branching, so
        # decode stays CUDA-graph-capturable. ``all_rank_num_tokens`` drives
        # the fused layer's dispatch/combine under attention DP.
        # Small TP batches leave enough GPU capacity to overlap the shared
        # expert with routing and the routed experts. The helper limits stream
        # switching to CUDA graphs and joins before the single reduction.
        routed, shared = maybe_execute_in_parallel(
            lambda: self.experts(
                flat,
                self.gate(flat),
                all_rank_num_tokens=all_rank_num_tokens if self.use_dp else None,
            ),
            lambda: self.shared_experts(flat),
            self.shared_expert_start,
            self.shared_expert_done,
            self.shared_expert_stream if flat.shape[0] <= 8 else None,
            disable_on_compile=True,
        )
        # Routed and shared are both rank partials (a K-dim partial per expert
        # in the TP4 layout, the local-expert partial sum in the TP4/EP4
        # layout). Sum them, then exactly one reduction covers the whole MoE
        # branch -- the DeepSeek-V3 order.
        mixed = routed + shared
        if self.moe_all_reduce is not None:
            mixed = self.moe_all_reduce(mixed)
        return mixed.view_as(x)


# ---------------------------------------------------------------------------
# Hyper-connected decoder
# ---------------------------------------------------------------------------


def glm5_next_hyper_connection(
    config: PretrainedConfig, dtype: torch.dtype = torch.bfloat16
) -> mHC:
    """Configure shared mHC with GLM's normalization and mixing semantics.

    Post weights are 2 * sigmoid(...). Pass the full hc_sinkhorn_iters count:
    the kernel includes the initial column normalization. Input normalization uses
    rms_norm_eps; mixing and Sinkhorn use hc_eps.
    """
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    return mHC(
        mult=int(config.hc_mult),
        hidden_size=int(config.hidden_size),
        sinkhorn_iters=int(config.hc_sinkhorn_iters),
        dtype=dtype,
        eps=float(config.hc_eps),
        norm_eps=float(config.rms_norm_eps),
        sinkhorn_eps=float(config.hc_eps),
        post_mult_value=2.0,
    )


def glm5_next_expand_streams(embeds: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[tokens, hidden]`` -> ``[tokens, hc_mult, hidden]``.

    The streams start as exact copies of the embedding and only diverge once the
    first hyper-connection mixes them. ``contiguous()`` is required, not merely
    tidy: an expanded view aliases a single row, and ``post_mapping`` writes each
    stream separately.
    """
    return embeds.unsqueeze(-2).expand(-1, hc_mult, -1).contiguous()


def glm5_next_hyper_head(hidden_streams: torch.Tensor) -> torch.Tensor:
    """Collapse the ``hc_mult`` streams with an **unweighted** mean.

    Deliberately not ``modules.mhc.HCHead``: that head is the DeepSeek-V4
    variant and carries learned ``fn``/``base``/``scale`` weights. GLM-5.3-Flash
    has no such parameters in the checkpoint and the source head is a plain
    mean, so using the weighted head would require inventing weights.
    """
    return hidden_streams.mean(dim=-2)


class Glm5NextDecoderLayer(DecoderLayer):
    """Wrap attention and FFN in separate mHC residual connections.

    Each connection collapses [tokens, hc_mult, hidden] streams for the sublayer
    and mixes its output back into the streams. Attention and MLP types come from
    the explicit per-layer schedules.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        schedule: Glm5NextSchedule,
        model_config: ModelConfig,
        dtype: torch.dtype = torch.bfloat16,
        aux_stream: torch.cuda.Stream | None = None,
        attention_aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = schedule.attention[layer_idx]
        self.mlp_type = schedule.mlp[layer_idx]
        eps = float(config.rms_norm_eps)

        mapping = model_config.mapping
        if self.attention_type == LINEAR_ATTENTION:
            self.self_attn: nn.Module = Glm5NextLinearAttention(
                config,
                layer_idx,
                dtype=dtype,
                mapping=mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=attention_aux_stream,
            )
        else:
            self.self_attn = Glm5NextSparseAttention(
                config,
                layer_idx,
                dtype=dtype,
                attn_backend=model_config.attn_backend,
                model_config=model_config,
            )
        self.mlp = (
            Glm5NextMoE(config, model_config, layer_idx, dtype=dtype, aux_stream=aux_stream)
            if self.mlp_type == SPARSE_MLP
            else GatedMLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.intermediate_size),
                bias=False,
                dtype=dtype,
                config=model_config,
                # Replicated under attention DP (the MLP follows the attention
                # layout, as in DeepSeek-V3).
                overridden_tp_size=1 if mapping.enable_attention_dp else None,
                # The layer owns the branch's single reduction (size-aware
                # strategy), as it does for attention and the MoE combine.
                reduce_output=False,
                layer_idx=layer_idx,
                swiglu_limit=float(config.swiglu_limit),
                disable_deep_gemm=True,
            )
        )
        self.mlp_all_reduce = (
            Glm5NextAllReduce(mapping, strategy=model_config.allreduce_strategy)
            if self.mlp_type != SPARSE_MLP and glm5_next_tp_reduces(mapping)
            else None
        )
        self.input_layernorm = RMSNorm(hidden_size=int(config.hidden_size), eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=eps, dtype=dtype
        )
        self.hc_attn = glm5_next_hyper_connection(config, dtype=dtype)
        self.hc_ffn = glm5_next_hyper_connection(config, dtype=dtype)

    def forward(
        self,
        position_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        runtime_ctx: Glm5NextRuntimeContext | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run attention and one full-batch FFN between the mHC connections."""
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        metadata = runtime_ctx.metadata
        residual = hidden_states
        post, comb, collapsed = self.hc_attn.pre_mapping(hidden_states)
        normed = self.input_layernorm(collapsed)
        if self.attention_type == LINEAR_ATTENTION:
            attn_out = self.self_attn(normed, metadata)
        else:
            attn_out = self.self_attn(normed, metadata, runtime_ctx=runtime_ctx)
        hidden_states = self.hc_attn.post_mapping(attn_out, residual, post, comb)

        residual = hidden_states
        post, comb, collapsed = self.hc_ffn.pre_mapping(hidden_states)
        # ADP ranks may have different phase mixes; every rank calls MoE once.
        mlp_out = self.run_mlp(
            self.post_attention_layernorm(collapsed), metadata.all_rank_num_tokens
        )
        return self.hc_ffn.post_mapping(mlp_out, residual, post, comb)

    def run_mlp(self, x: torch.Tensor, all_rank_num_tokens: list[int] | None) -> torch.Tensor:
        """The FFN branch: the MoE (fed the attention-DP token counts), or
        the shared dense ``GatedMLP`` followed by this layer's single TP
        reduction of its partial."""
        if self.mlp_type == SPARSE_MLP:
            return self.mlp(x, all_rank_num_tokens)
        out = self.mlp(x)
        return out if self.mlp_all_reduce is None else self.mlp_all_reduce(out)


class Glm5NextModel(DecoderModel):
    """Carry [tokens, hc_mult, hidden] streams through the decoder layers.

    Expand embeddings into streams, then collapse with an unweighted mean and
    final RMSNorm.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        # PP transports contiguous [tokens, hc_mult, hidden] streams. The base
        # class prunes remote layers, which load_weights skips on this rank.
        self.config = config
        self.schedule = schedule
        self.hc_mult = int(config.hc_mult)
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

        # Decoder and draft layers execute serially and share this side stream.
        self.aux_stream_dict = {}
        if torch.cuda.is_available():
            # KDA overlaps its small gate projections with the QKV projection.
            self.aux_stream_dict[AuxStreamType.Attention] = torch.cuda.Stream()
            if not model_config.mapping.enable_attention_dp:
                self.aux_stream_dict[AuxStreamType.MoeShared] = torch.cuda.Stream()

        self.embed_tokens = Embedding(int(config.vocab_size), int(config.hidden_size), dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(
                    config,
                    i,
                    schedule,
                    model_config,
                    dtype=dtype,
                    aux_stream=self.aux_stream_dict.get(AuxStreamType.MoeShared),
                    attention_aux_stream=self.aux_stream_dict.get(AuxStreamType.Attention),
                )
                for i in range(schedule.num_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=float(config.rms_norm_eps), dtype=dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_ctx: Glm5NextRuntimeContext | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """The runtime forward over the executor's packed token batch.

        ``runtime_ctx`` is normally derived from ``attn_metadata`` here; the
        parameter lets a caller (or a parity test) inject a context built by
        another route. The stream axis opens once, is carried through all 45
        layers, and closes in :meth:`collapse_streams`.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        # Only the decoder stack: under one-model MTP the speculative base
        # class appends the draft layer(s) to ``self.layers`` (so the
        # checkpoint's ``model.layers.45.*`` names resolve), and those run in
        # the speculative worker's draft loop, not here.
        # Every batch phase takes the fused hyper-connection loop, so the
        # engine's warmup / KV-cache sizing pass measures the same activation
        # peak that mixed context+generation iterations reach in production.
        if self._fused_hc_ok():
            return self.collapse_streams(
                self._forward_single_phase_fused_hc(inputs_embeds, runtime_ctx)
            )
        streams = self.expand_streams(inputs_embeds)
        for layer_idx in range(self.schedule.num_layers):
            streams = self.layers[layer_idx](
                position_ids=position_ids,
                hidden_states=streams,
                attn_metadata=attn_metadata,
                runtime_ctx=runtime_ctx,
            )
        return self.collapse_streams(streams)

    def _fused_hc_ok(self) -> bool:
        """Whether every decoder layer is local (no PP pruning) for the fused loop."""
        cached = self.__dict__.get("_fused_hc_ok_cache")
        if cached is None:
            cached = all(
                not getattr(self.layers[i], "_weights_removed", False)
                for i in range(self.schedule.num_layers)
            )
            self._fused_hc_ok_cache = cached
        return cached

    def _forward_single_phase_fused_hc(
        self, embeds: torch.Tensor, runtime_ctx: Glm5NextRuntimeContext
    ) -> torch.Tensor:
        """Single-phase forward with fused hyper-connection boundaries.

        Every boundary between two sublayers is ``post_mapping`` of the
        previous site, ``pre_mapping`` of the next, and the next sublayer's
        RMSNorm; ``mHC.fused_hc`` (the in-tree DeepSeek-V4 boundary op) runs
        the three as one kernel, so a layer costs 2 fused boundaries instead
        of 4 mappings + 2 norms. The math matches the per-layer forward; the
        first pre-mapping and last post-mapping stay unfused. Pipeline-parallel
        ranks keep the per-layer path (see :meth:`_fused_hc_ok`).

        Only ``residual`` holds the streams and each sublayer output is dropped
        once its boundary consumed it, so a long prefill keeps one copy of the
        [tokens, hc_mult, hidden] streams alive, as the per-layer loop does.
        """
        num_layers = self.schedule.num_layers
        first = self.layers[0]
        residual = self.expand_streams(embeds)
        post, comb, x = first.hc_attn.pre_mapping(residual)
        x = first.input_layernorm(x)
        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            if layer.attention_type == LINEAR_ATTENTION:
                attn_out = layer.self_attn(x, runtime_ctx.metadata)
            else:
                attn_out = layer.self_attn(x, runtime_ctx.metadata, runtime_ctx=runtime_ctx)
            residual, post, comb, x = layer.hc_ffn.fused_hc(
                attn_out,
                residual,
                post,
                comb,
                norm_weight=layer.post_attention_layernorm.weight,
                norm_eps=layer.post_attention_layernorm.variance_epsilon,
            )
            del attn_out
            mlp_out = layer.run_mlp(x, runtime_ctx.metadata.all_rank_num_tokens)
            if layer_idx + 1 < num_layers:
                nxt = self.layers[layer_idx + 1]
                residual, post, comb, x = nxt.hc_attn.fused_hc(
                    mlp_out,
                    residual,
                    post,
                    comb,
                    norm_weight=nxt.input_layernorm.weight,
                    norm_eps=nxt.input_layernorm.variance_epsilon,
                )
            else:
                residual = layer.hc_ffn.post_mapping(mlp_out, residual, post, comb)
            del mlp_out
        return residual

    def expand_streams(self, embeds: torch.Tensor) -> torch.Tensor:
        """Open the four-stream axis at the embedding."""
        return glm5_next_expand_streams(embeds, self.hc_mult)

    def collapse_streams(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Unweighted stream mean followed by the final RMS norm."""
        return self.norm(glm5_next_hyper_head(hidden_streams))


# ---------------------------------------------------------------------------
# Multi-token prediction (one-model MTP speculative decoding)
# ---------------------------------------------------------------------------


class Glm5NextMTPHead(nn.Module):
    """Normalize MTP hidden states and project draft logits through the target head.

    Glm5NextMTP applies norm before this forward. The speculative sampler consumes
    vocab-sharded logits; the checkpoint has no separate MTP head weight.
    """

    def __init__(
        self, config: PretrainedConfig, model_config: ModelConfig, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.norm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=float(config.rms_norm_eps), dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """Draft logits of every row of ``hidden_states``; the caller picks the drafting rows."""
        # The draft sampler consumes vocab-sharded logits. Preserve the target
        # head's setting even if projection fails.
        gather_output = lm_head.gather_output
        lm_head.gather_output = False
        try:
            return lm_head(hidden_states, is_spec_decoding_head=True)
        finally:
            lm_head.gather_output = gather_output


class Glm5NextMTP(nn.Module):
    """Native next-token prediction layer with ordinary residual connections.

    Normalizes and concatenates token embeddings and target hidden states before
    eh_proj. Reuses the main stack's sparse attention and MoE, with its own
    latent/indexer pages, followed by shared_head.norm. Unlike the main decoder,
    the MTP layer has no mHC weights.

    The first draft step shares the target's packed context/verification layout.
    Later steps read live metadata lengths after the speculative worker rewinds
    them. The target supplies post-final-norm hidden states.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Any = None,
    ) -> None:
        super().__init__()
        config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        num_nextn = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        if not (schedule.num_layers <= layer_idx < schedule.num_layers + num_nextn):
            raise ValueError(
                f"glm5_next MTP layer index {layer_idx} is outside the checkpoint's appended "
                f"range [{schedule.num_layers}, {schedule.num_layers + num_nextn})"
            )
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        hidden = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.model_config = model_config
        self.layer_idx = layer_idx
        self.attention_type = SPARSE_ATTENTION
        self.mlp_type = SPARSE_MLP

        self.enorm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        self.hnorm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        # Row-parallel (DeepSeek-V3 MTP ownership): each rank consumes its own
        # input chunk, one in-Linear reduction; replicated under attention DP.
        self.eh_proj = Linear(
            2 * hidden,
            hidden,
            tensor_parallel_mode=(
                None if model_config.mapping.enable_attention_dp else TensorParallelMode.ROW
            ),
            **_glm5_linear_kwargs(model_config, None),
        )
        self.input_layernorm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        self.self_attn = Glm5NextSparseAttention(
            config,
            layer_idx,
            dtype=dtype,
            attn_backend=model_config.attn_backend,
            model_config=model_config,
        )
        self.mlp = Glm5NextMoE(
            config,
            model_config,
            layer_idx,
            dtype=dtype,
            aux_stream=aux_stream_dict.get(AuxStreamType.MoeShared) if aux_stream_dict else None,
        )
        self.shared_head = Glm5NextMTPHead(config, model_config, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        embed_tokens: nn.Module,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        del position_ids  # NoPE
        if all_rank_num_tokens is None:
            all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        inputs_embeds = self.enorm(embed_tokens(input_ids))
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        mapping = self.model_config.mapping
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            # Row-parallel eh_proj: each rank consumes its own input chunk.
            hidden_states = torch.chunk(hidden_states, mapping.tp_size, dim=-1)[mapping.tp_rank]
        hidden_states = self.eh_proj(hidden_states)

        runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        residual = hidden_states
        attn_in = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(attn_in, attn_metadata, runtime_ctx=runtime_ctx)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states), all_rank_num_tokens)
        hidden_states = residual + hidden_states

        return self.shared_head.norm(hidden_states)


# ---------------------------------------------------------------------------
# One-model MTP, run inside the model's forward
# ---------------------------------------------------------------------------


def _check_speculation(spec_config: Any) -> None:
    """Reject at construction the speculative configurations the in-model MTP step does not compute.

    Routing declines the same configurations (:func:`fused_speculation_refused`) and sends them to the split target
    ``glm_5_3_flash_nvfp4__sm_100__default``; this check covers direct construction.
    """
    refused = fused_speculation_refused(spec_config)
    if refused:
        raise ValueError(
            "the GLM-5.3-Flash MTP-fused modeling_v2 target runs linear one-model MTP-Eagle with strict acceptance "
            f"inside its forward; it does not run {', '.join(refused)}"
        )


class Glm5NextDraftModel(nn.Module):
    """The drafter: the checkpoint's one MTP layer, with the target's LM head and embedding (shared, not copied)."""

    def __init__(self, mtp_layer: nn.Module, lm_head: nn.Module, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.mtp_layers = nn.ModuleList([mtp_layer])
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens


class Glm5NextMtpStep:
    """One engine step of one-model MTP-Eagle, after the target decoder ran.

    1. Accept: sample the target token of every verify row (argmax for an all-greedy batch, else each row's own
       temperature / top-k / top-p / min-p), then strict prefix acceptance of the previous step's drafts.
    2. Commit the KDA recurrent state: the cache manager records each generation request's accepted count, which
       the next verify replays (``kda_mtp_decode``).
    3. Draft ``K = runtime_draft_len`` tokens: the MTP layer on (accepted token, target hidden state) rows, its
       shared head on the vocabulary shard, and the global argmax across the TP group; step 0 runs on the verify
       layout, steps 1.. on one row per request.
    4. Restore the attention metadata the chain rewrote (CUDA-graph warmup replays this forward on the same
       metadata) and return what the speculative sampler reads.

    Every shape is fixed by host integers of the step (batch size, number of contexts, K), so the step captures in
    one CUDA graph with the target.
    """

    def __init__(self, mapping: Mapping, vocab_size: int) -> None:
        self.mapping = mapping
        self.vocab_size = vocab_size
        self._saved: dict[str, Any] = {}

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Any,
        draft_model: Glm5NextDraftModel,
    ) -> dict[str, torch.Tensor]:
        """Returns ``logits`` (the verify logits, as given), ``new_tokens [B, K+1]`` (accepted tokens; a context
        request defines column 0 only), ``new_tokens_lens [B]`` (1 + accepted drafts), ``next_draft_tokens [B, K]``
        and ``next_new_tokens [B, K+1]`` (each request's last accepted token, then its drafts: the next step's input
        under the overlap scheduler)."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if spec_metadata.is_ragged_verify:
            raise ValueError(
                "the GLM-5.3-Flash modeling_v2 target verifies a uniform draft width only"
            )
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        draft_len = spec_metadata.runtime_draft_len
        target_tokens = self._sample_target(logits, spec_metadata)
        if draft_len == 0:
            return self._without_drafts(logits, target_tokens, batch_size)

        if num_gens > 0:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, draft_len)
        else:
            draft_tokens = torch.empty((0, draft_len), dtype=torch.int32, device=logits.device)
        accepted_tokens, num_accepted_tokens = strict_accept(
            target_tokens, draft_tokens, num_contexts, batch_size
        )

        if num_gens > 0:
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        try:
            # Saved inside the ``try``: a save that fails partway, such as an out-of-memory copy during the engine's
            # warmup, must not leave the lengths saved, or every later step fails in ``prepare_for_spec_dec``.
            self._save_metadata(attn_metadata, batch_size)
            next_draft_tokens = self._draft(
                input_ids,
                position_ids,
                hidden_states,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
            )
        finally:
            self._restore_metadata(attn_metadata)
            if original_all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        last_accepted = accepted_tokens[
            spec_metadata.batch_indices_cuda[:batch_size], num_accepted_tokens - 1
        ]
        next_new_tokens = torch.cat([last_accepted.unsqueeze(1), next_draft_tokens], dim=1)
        attn_metadata.use_spec_decoding = True
        return {
            "logits": logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    # -- acceptance ---------------------------------------------------------

    @staticmethod
    def _sample_target(logits: torch.Tensor, spec_metadata: Any) -> torch.Tensor:
        """The target token of every logits row: the argmax, or a sample from that row's sampling parameters."""
        if spec_metadata.is_all_greedy_sample:
            return torch.argmax(logits, dim=-1)
        rows = logits.shape[0]
        temperatures = spec_metadata.temperatures[:rows]
        top_ks = spec_metadata.top_ks[:rows]
        top_ps = spec_metadata.top_ps[:rows]
        min_ps = spec_metadata.min_ps[:rows]
        seed, offset = spec_metadata.seeds[:rows], spec_metadata.offsets[:rows]
        mode = spec_metadata.advanced_sampling_mode
        if mode.is_fused:
            return fused_sampling.fused_sample_from_logits(
                logits, temperatures, top_ks, top_ps, min_ps, seed=seed, offset=offset
            )
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(mode, top_ks, top_ps)
        return flashinfer_sampling.sample_from_logits_op(
            logits, temperatures, top_ks, top_ps, seed=seed, offset=offset
        )

    @staticmethod
    def _without_drafts(
        logits: torch.Tensor, target_tokens: torch.Tensor, batch_size: int
    ) -> dict[str, torch.Tensor]:
        """A step with no drafts (runtime draft length 0): the target token alone."""
        tokens = target_tokens[:batch_size].to(torch.int32).unsqueeze(1)
        return {
            "logits": logits,
            "new_tokens": tokens,
            "new_tokens_lens": torch.ones(batch_size, dtype=torch.int32, device=logits.device),
            "next_draft_tokens": torch.zeros(
                (batch_size, 0), dtype=torch.int32, device=logits.device
            ),
            "next_new_tokens": tokens.clone(),
        }

    def _draft(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Any,
        draft_model: Glm5NextDraftModel,
    ) -> torch.Tensor:
        """``next_draft_tokens [B, K]``. Rewrites the attention metadata's lengths as it goes; the caller restores."""
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        draft_len = spec_metadata.runtime_draft_len
        mtp_layer = draft_model.mtp_layers[0]

        # Step 0 runs on the target's own rows: a context request's prompt shifted left by one, its last row the
        # token that follows the chunk; a generation request's K+1 accepted-token columns.
        context_ids = self._context_input_ids(
            input_ids, num_ctx_tokens, spec_metadata, accepted_tokens, num_contexts
        )
        gen_ids = accepted_tokens[num_contexts:, : draft_len + 1].flatten()
        step_ids = torch.cat([context_ids, gen_ids], dim=0)
        step_positions = position_ids.squeeze(0)
        step_hidden = hidden_states
        # The row each request drafts from on step 0: a context request's last token, a generation request's last
        # accepted one.
        last_rows = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        gen_rows = (
            spec_metadata.batch_indices_cuda[:num_gens].long() * (draft_len + 1)
            + num_accepted_tokens[num_contexts:]
            - 1
            + num_ctx_tokens
        )
        step0_rows = torch.cat([last_rows[:num_contexts], gen_rows], dim=0)

        drafts = []
        for step in range(draft_len):
            all_rank_num_tokens = (
                spec_metadata.all_rank_num_tokens
                if step == 0
                else spec_metadata.subseq_all_rank_num_tokens
            )
            out = mtp_layer(
                embed_tokens=draft_model.embed_tokens,
                all_rank_num_tokens=all_rank_num_tokens,
                input_ids=step_ids,
                position_ids=step_positions,
                hidden_states=step_hidden,
                attn_metadata=attn_metadata,
            )
            rows = step0_rows if step == 0 else slice(0, batch_size)
            request_hidden = out[rows]
            draft_logits = mtp_layer.shared_head(request_hidden, draft_model.lm_head)
            tokens = self._draft_argmax(draft_logits)
            drafts.append(tokens)
            if step + 1 < draft_len:
                step_ids = tokens
                step_hidden = request_hidden
                step_positions = (
                    step_positions[rows] if step_positions.dim() == 1 else step_positions[..., rows]
                ) + 1
            self._advance_lengths(
                attn_metadata, step, num_contexts, batch_size, draft_len, num_accepted_tokens
            )
        return torch.stack(drafts, dim=1)

    @staticmethod
    def _context_input_ids(
        input_ids: torch.Tensor,
        num_ctx_tokens: int,
        spec_metadata: Any,
        accepted_tokens: torch.Tensor,
        num_contexts: int,
    ) -> torch.Tensor:
        """A context chunk's tokens shifted left by one; each chunk's last row takes the prompt's next token when
        the chunk is not the prompt's last (``context_prompt_lookahead_tokens``), else the accepted target token."""
        if num_ctx_tokens == 0:
            return torch.empty(0, dtype=torch.int32, device=accepted_tokens.device)
        prompt = input_ids[:num_ctx_tokens]
        shifted = torch.empty_like(prompt, dtype=torch.int32)
        shifted[:-1].copy_(prompt[1:])
        tail = accepted_tokens[:num_contexts, 0]
        lookahead = spec_metadata.context_prompt_lookahead_tokens
        if lookahead is not None:
            lookahead = lookahead[:num_contexts]
            tail = torch.where(lookahead != INVALID_PROMPT_LOOKAHEAD_TOKEN, lookahead, tail)
        shifted[spec_metadata.gather_ids[:num_contexts]] = tail
        return shifted

    def _draft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Each row's draft token: the argmax over the full vocabulary. Under tensor parallelism (without attention
        DP) the head yields this rank's vocabulary shard, so each rank contributes its (global index, max) pair and
        every rank picks the same global argmax."""
        mapping = self.mapping
        if (
            mapping.tp_size > 1
            and not mapping.enable_attention_dp
            and logits.shape[-1] < self.vocab_size
        ):
            values, index = torch.max(logits, dim=-1, keepdim=True)
            global_index = index.to(torch.int32) + mapping.tp_rank * logits.shape[-1]
            pairs = torch.stack([global_index.float(), values.float()], dim=-1).flatten(-2)
            gathered = allgather(pairs, mapping, dim=-1)
            best = torch.argmax(gathered[..., 1::2], dim=-1, keepdim=True)
            return torch.gather(gathered[..., 0::2], -1, best).squeeze(-1).to(torch.int32)
        return torch.argmax(logits, dim=-1).to(torch.int32)

    @staticmethod
    def _advance_lengths(
        attn_metadata: AttentionMetadata,
        step: int,
        num_contexts: int,
        batch_size: int,
        draft_len: int,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        """Lengths for the next draft step. After step 0 every request feeds one row: a generation request's KV
        length drops its rejected drafts and a context request's grows by its first draft, then each step adds one."""
        if step == 0:
            attn_metadata._seq_lens[:batch_size].fill_(1)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
            attn_metadata.on_update()
            has_kv_cache = attn_metadata.kv_cache_manager is not None
            if has_kv_cache and hasattr(attn_metadata, "host_request_types"):
                attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
            if has_kv_cache:
                attn_metadata.num_contexts = 0
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                draft_len - num_accepted_tokens[num_contexts:]
            )
            attn_metadata.kv_lens_cuda[:num_contexts] += 1
            if (
                has_kv_cache
                and num_contexts > 0
                and getattr(attn_metadata, "enable_flash_mla", False)
            ):
                attn_metadata.block_ids_per_seq[:batch_size].copy_(
                    attn_metadata.kv_block_ids_per_seq[:batch_size], non_blocking=True
                )
            attn_metadata.update_for_spec_dec()
            attn_metadata.use_spec_decoding = False
        else:
            attn_metadata.kv_lens_cuda[:batch_size] += 1
            attn_metadata.update_for_spec_dec()

    # -- attention metadata save / restore ----------------------------------

    def _save_metadata(self, attn_metadata: AttentionMetadata, batch_size: int) -> None:
        """Save what the draft chain overwrites: the sequence lengths (``prepare_for_spec_dec``) and the target's
        speculative masks, position offsets and generation lengths."""
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        saved = {}
        if attn_metadata.spec_decoding_packed_mask is not None:
            saved["packed_mask"] = attn_metadata.spec_decoding_packed_mask[:batch_size].clone()
        if attn_metadata.spec_decoding_position_offsets is not None:
            saved["position_offsets"] = attn_metadata.spec_decoding_position_offsets.clone()
        if attn_metadata.spec_decoding_generation_lengths is not None:
            saved["generation_lengths"] = attn_metadata.spec_decoding_generation_lengths[
                :batch_size
            ].clone()
        self._saved = saved

    def _restore_metadata(self, attn_metadata: AttentionMetadata) -> None:
        """Undo :meth:`_save_metadata`, also after a save that failed partway: each field only if it was saved."""
        if attn_metadata.has_spec_dec_saved_state:
            attn_metadata.restore_from_spec_dec()
            attn_metadata.on_update()
        saved, self._saved = self._saved, {}
        if "packed_mask" in saved:
            mask = saved["packed_mask"]
            attn_metadata.spec_decoding_packed_mask[: mask.shape[0]].copy_(mask)
        if "position_offsets" in saved:
            attn_metadata.spec_decoding_position_offsets.copy_(saved["position_offsets"])
        if "generation_lengths" in saved:
            lengths = saved["generation_lengths"]
            attn_metadata.spec_decoding_generation_lengths[: lengths.shape[0]].copy_(lengths)


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


def _check_construction(model_config: ModelConfig) -> None:
    """Fail at construction on what this target asserts rather than adapts: the GPU it was written for and the
    NVFP4 checkpoint. Routing checks both; a direct instantiation would not."""
    capability = torch.cuda.get_device_capability()
    if capability != _SM:
        raise ValueError(
            f"this target is written for sm{_SM[0]}{_SM[1]}; the device is sm{capability[0]}{capability[1]}"
        )
    quant_config = model_config.quant_config
    quant_algo = None if quant_config is None else quant_config.quant_algo
    if quant_algo != QuantAlgo.NVFP4:
        raise ValueError(
            f"this target loads the NVFP4 GLM-5.3-Flash checkpoint; the checkpoint is {quant_algo}"
        )


class ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused(Glm5NextForCausalLM):
    """The class ``route`` returns: this module's GLM-5.3-Flash text model behind its checks.

    The causal LM, the decoder, the MTP layer and the speculative step are this module's; the engine hooks (model
    defaults, the V2 hybrid cache manager, the KDA metadata class) come with them.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _check_construction(model_config)
        super().__init__(model_config)

    def load_weights(self, weights: Any, weight_mapper: BaseWeightMapper | None = None) -> None:
        _weights.load(self, weights, weight_mapper)

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Refuse, at admission, a request that carries multimodal input: this target runs no vision tower.

        The executor calls this only for a request with multimodal data. Refusing here fails that request alone;
        the forward would otherwise serve its text without the image or video.
        """
        del mm_data
        raise ValueError(
            "the GLM-5.3-Flash modeling_v2 target serves text only and runs no vision tower; the request carries "
            "multimodal input"
        )
