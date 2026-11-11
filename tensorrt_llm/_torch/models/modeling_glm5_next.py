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
"""GLM-5.3-Flash text decoder and one-model MTP support.

The decoder combines KDA recurrent attention, pool-compressed sparse MLA,
clamped-SwiGLU experts, and hyper-connections. The composite checkpoint's
``text_config`` defines the layer schedule; vision weights are not loaded.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn
from transformers import PretrainedConfig

from ..._utils import async_tensor_h2d
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantConfig
from ...quantization.mode import QuantAlgo
from ..attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from ..attention.backends.sparse.glm_kpool import (
    Glm5NextMamba2Metadata,
    GlmKpoolBackendForwardArgs,
    GlmKpoolSparseParams,
    glm_kpool_cache_row_dim,
)
from ..attention.backends.utils import create_attention
from ..distributed import AllReduceStrategy
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..pyexecutor.config_utils import unwrap_glm5_next_text_config
from ..utils import AuxStreamType
from .checkpoints.hf.glm5_next_weight_mapper import (
    Glm5NextHfWeightMapper,
    glm5_next_is_quantized,
    glm5_next_weight_owner,
)
from .modeling_deepseekv3 import DeepseekV3Gate
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model

if TYPE_CHECKING:
    from ..modules.mamba.mamba2_metadata import Mamba2Metadata
    from ..modules.mhc.hyper_connection import mHC
    from .checkpoints.base_weight_mapper import BaseWeightMapper


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
        from ..distributed import AllReduce

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
    # The one-model MTP drafter (``MTPForCausalLM``) reads the checkpoint's
    # next-n layer count from the config it is handed, which is this top-level
    # composite one; the field lives on ``text_config``.
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


@register_auto_model("Glm5NextForCausalLM")
class Glm5NextForCausalLM(SpecDecOneEngineForCausalLM):
    """GLM-5.3-Flash decoder with an optional one-model MTP drafter.

    The speculative base class manages draft/verify execution. The weight mapper
    routes text and MTP tensors and excludes the vision namespace.
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
        if (
            model_config.mapping.enable_attention_dp
            and model_config.mapping.enable_lm_head_tp_in_adp
            and model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
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
        super().__init__(
            Glm5NextModel(model_config),
            model_config,
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
        )
        # ``config`` is a base-class property (the top-level composite config);
        # the narrowed decoder contract lives here.
        self.text_config = text_config
        self.schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        # One-model MTP: the base class built ``draft_model.mtp_layers`` (one
        # Glm5NextMTP at layer_idx 45) and the speculative worker. Alias the
        # draft layer(s) onto ``model.layers[45:]`` so the checkpoint's
        # ``model.layers.45.*`` keys are placed by the same exact loader, the
        # same projection swap, and the same per-owner materialization.
        self.mtp_layers: tuple[nn.Module, ...] = ()
        spec_config = model_config.spec_config
        if spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model():
            mtp_layers = tuple(self.draft_model.mtp_layers)
            if len(mtp_layers) != 1:
                raise ValueError(
                    f"glm5_next builds exactly one MTP layer (num_nextn_predict_layers=1); "
                    f"the drafter constructed {len(mtp_layers)}"
                )
            self.model.layers.extend(mtp_layers)
            self.mtp_layers = mtp_layers
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

    # -- whole-model materialization --------------------------------------

    def load_weights(
        self,
        weights: Any,
        weight_mapper: BaseWeightMapper | None = None,
    ) -> None:
        """Load checkpoint tensors owner by owner into the allocated model.

        The registered GLM mapper resolves keys and destination owners. FP8 payloads
        and FP32 block scales retain their checkpoint representation; module loaders
        handle sharding and fused projections. Unresolved or unfilled tensors raise.

        Args:
            weights: Mapping of checkpoint keys to tensors or lazy safetensors slices.
            weight_mapper: Initialized Glm5NextHfWeightMapper, or None to create one.
        """
        if weight_mapper is None:
            weight_mapper = Glm5NextHfWeightMapper()
            weight_mapper.init_model_and_config(self, self.model_config)
        elif not isinstance(weight_mapper, Glm5NextHfWeightMapper):
            raise ValueError(
                "glm5_next needs its registered Glm5NextHfWeightMapper; got "
                f"{type(weight_mapper).__name__}"
            )
        if not self.quantized:
            raise ValueError(
                "glm5_next whole-model loading requires a block-FP8 or NVFP4 build "
                "(quantized=True). Dequantizing all 288 experts of 42 routed "
                "layers to bf16 would double the resident model to ~656 GB and "
                "exceed the supported loading footprint."
            )
        # Decoder stack plus the aliased MTP draft layer(s), if any: owners
        # ``45..`` are the draft layers, placed from the checkpoint's own
        # ``layers.45.*`` keys instead of being allowlisted away.
        num_layers = self.schedule.num_layers + len(self.mtp_layers)
        audit = weight_mapper.audit(list(weights.keys()))

        # Checkpoint tensors are grouped by owner (a decoder layer or a named
        # top-level part) and placed by that owner's modules.
        targets: dict[Any, tuple[nn.Module, str]] = {
            **{i: (self.model.layers[i], f"model.layers.{i}.") for i in range(num_layers)},
            "embed": (self.model.embed_tokens, "model.embed_tokens."),
            "norm": (self.model.norm, "model.norm."),
            "head": (self.lm_head, "lm_head."),
        }
        # Owners pruned by pipeline parallelism (`__pp_init__` cleared their
        # parameters): their checkpoint keys belong to another rank.
        remote = {
            owner
            for owner, (module, _) in targets.items()
            if getattr(module, "_weights_removed", False)
        }

        by_owner: dict[Any, list[tuple[str, str]]] = {}
        for dest, key in audit.destinations.items():
            owner = glm5_next_weight_owner(dest, num_layers)
            if owner is None:
                raise ValueError(f"glm5_next has no destination owner for {key!r} -> {dest!r}")
            if owner in remote:
                continue
            by_owner.setdefault(owner, []).append((key, dest))
        if audit.unresolved:
            raise ValueError(f"glm5_next cannot place {sorted(audit.unresolved)[:5]}")

        device = torch.device("cuda", torch.cuda.current_device())
        for owner, (module, prefix) in targets.items():
            if owner in remote:
                continue
            self._fill_module(module, owner, prefix, by_owner.get(owner, []), weights, device)
        # The shared KDA mixer's post-load kernel constants (built from the
        # just-loaded local shards; the same step the Kimi K3 loader runs).
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, Glm5NextLinearAttention) and not getattr(
                layer, "_weights_removed", False
            ):
                attn.finalize_weights()
        logger.info(f"glm5_next loaded {len(targets) - len(remote)} owners")

    def _fill_module(
        self,
        module: nn.Module,
        owner: Any,
        prefix: str,
        entries: Sequence[tuple[str, str]],
        weights: Any,
        device: torch.device,
    ) -> None:
        """Place one materialized owner's tensors.

        * A TensorRT-LLM ``Linear`` destination receives its checkpoint tensors
          through the module's own ``load_weights`` (which owns the TP slicing
          and the block-scale layout). Projections the runtime keeps fused
          (``GatedMLP.gate_up_proj``, the sparse layers' ``[q_a | kv_a]`` and
          the indexer's ``[wk | gate | weights_proj]``) collect their separate
          checkpoint tensors first (:attr:`FUSED_LINEARS` on the owning module).
        * Production routed experts go to the fused MoE layer's loader,
          filtered to ``initial_local_expert_ids`` except for the scalar
          ``input_scale``, which is passed for every expert.
        * The KDA layer's tensors (the shared mixer's local-width modules) are
          sliced to this rank's head range by the module itself.
        * Everything else (norms, HC, router, embeddings) is an exact-shape
          replicated copy.
        """
        params = dict(module.named_parameters())
        params.update(dict(module.named_buffers()))
        named_modules = dict(module.named_modules())
        kda = named_modules.get("self_attn")
        if not isinstance(kda, Glm5NextLinearAttention):
            kda = None
        filled: set[str] = set()

        # source module path -> (fused Linear path, position in the concatenation)
        fused_of: dict[str, tuple[str, int]] = {}
        for mod_name, sub in named_modules.items():
            base = f"{mod_name}." if mod_name else ""
            for fused_attr, sources in getattr(type(sub), "FUSED_LINEARS", ()):
                for pos, src in enumerate(sources):
                    fused_of[base + src] = (base + fused_attr, pos)
            if isinstance(sub, GatedMLP):
                fused_of[base + "gate_proj"] = (base + "gate_up_proj", 0)
                fused_of[base + "up_proj"] = (base + "gate_up_proj", 1)
        linear_groups: dict[str, dict[str, Any]] = {}
        fused_groups: dict[str, dict[int, dict[str, Any]]] = {}

        experts = getattr(getattr(module, "mlp", None), "experts", None)
        local_expert_ids = set(experts.initial_local_expert_ids) if experts is not None else set()
        experts_quant = getattr(experts, "quant_config", None)
        quant_algo = experts_quant.quant_algo if experts_quant is not None else None
        # The loader reduces every expert's scalar input_scale into the
        # layer-wide activation scale, which must be identical on every rank
        # because tokens are quantized before dispatch.
        collect_all_expert_input_scales = quant_algo == QuantAlgo.NVFP4
        moe_weights: dict[str, torch.Tensor] = {}
        proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

        def materialize(t: Any) -> torch.Tensor:
            # Ellipsis supports both scalar calibration scales and matrix weights.
            return t if torch.is_tensor(t) else t[...]

        for key, dest in sorted(entries):
            local = dest.removeprefix(prefix)
            lazy = weights[key]

            moe = _FUSED_MOE_RE.match(dest)
            if moe is not None:
                expert_id = int(moe.group("expert"))
                tensor = moe.group("tensor")
                # Other EP ranks' weights are never read.
                if expert_id in local_expert_ids or (
                    collect_all_expert_input_scales and tensor == "input_scale"
                ):
                    moe_weights[f"{expert_id}.{proj_to_w[moe.group('proj')]}.{tensor}"] = (
                        materialize(lazy)
                    )
                continue

            if local.endswith(".weight_scale_inv"):
                mod_path, label = local[: -len(".weight_scale_inv")], "weight_scale_inv"
            else:
                mod_path, _, label = local.rpartition(".")
            if mod_path in fused_of:
                fused_path, pos = fused_of[mod_path]
                fused_groups.setdefault(fused_path, {}).setdefault(pos, {})[label] = materialize(
                    lazy
                )
                continue
            if local in fused_of:  # a bare parameter (no ``.weight``) feeding a fused Linear
                fused_path, pos = fused_of[local]
                fused_groups.setdefault(fused_path, {}).setdefault(pos, {})["weight"] = materialize(
                    lazy
                )
                continue
            if isinstance(named_modules.get(mod_path), Linear):
                linear_groups.setdefault(mod_path, {})[label] = materialize(lazy)
                continue

            tensor = materialize(lazy)
            if kda is not None and local.startswith("self_attn."):
                tensor = kda.shard_checkpoint_tensor(local[len("self_attn.") :], tensor)
            target = params.get(local)
            if target is None:
                raise KeyError(f"glm5_next has no parameter for {key!r} (destination {dest!r})")
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"glm5_next {dest!r}: checkpoint shape {tuple(tensor.shape)} does not "
                    f"match parameter shape {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor.to(device=device, dtype=target.dtype))
            filled.add(local)

        def mark_linear(mod_path: str, labels: Iterable[str] = ()) -> None:
            # ``labels`` are extra tensors the Linear's load_weights consumed
            # (e.g. the router gate's e_score_correction_bias).
            linear = named_modules[mod_path]
            param_names = ["weight", "weight_scale", "bias", *labels]
            if (
                linear.quant_config is not None
                and linear.quant_config.quant_algo == QuantAlgo.NVFP4
            ):
                # The shared loader fills the calibrated scales and derives
                # alpha; the KV scales keep their construction values.
                param_names += [
                    "input_scale",
                    "inv_input_scale",
                    "weight_scale_2",
                    "alpha",
                    "kv_scales",
                    "inv_kv_scales",
                ]
            for param_name in param_names:
                # mod_path is "" when the owner *is* the Linear (the LMHead).
                name = f"{mod_path}.{param_name}" if mod_path else param_name
                if name in params:
                    filled.add(name)

        def load_linear(
            mod_path: str, groups: list[dict[str, torch.Tensor]], labels: Iterable[str] = ()
        ) -> None:
            linear = named_modules[mod_path]
            if (
                linear.quant_config is not None
                and linear.quant_config.quant_algo == QuantAlgo.NVFP4
            ):
                required = {"weight", "weight_scale", "weight_scale_2", "input_scale"}
                for part, group in enumerate(groups):
                    missing = required - group.keys()
                    if missing:
                        raise ValueError(
                            f"glm5_next owner {owner!r}: NVFP4 projection {mod_path!r} "
                            f"part {part} is missing checkpoint tensors {sorted(missing)}"
                        )
            linear.load_weights(groups)
            mark_linear(mod_path, labels)

        for mod_path, group in linear_groups.items():
            load_linear(mod_path, [group], group)
        for fused_path, parts in fused_groups.items():
            dest_mod = named_modules[fused_path]
            groups = [parts[i] for i in sorted(parts)]
            if isinstance(
                dest_mod, Linear
            ) and dest_mod.weights_loading_config.weight_mode.name == ("FUSED_GATE_UP_LINEAR"):
                # GatedMLP: the Linear shards each half over this rank's
                # intermediate range and stacks them [gate; up] itself.
                load_linear(fused_path, groups)
            else:
                # Replicated fusions: one row-concatenated tensor (and block
                # scales), the order being the module's declared source order.
                fused_quant = getattr(dest_mod, "quant_config", None)
                if fused_quant is not None and fused_quant.quant_algo == QuantAlgo.NVFP4:
                    # Each NVFP4 source has its own per-tensor scales; merging them is not
                    # implemented.
                    raise ValueError(
                        f"glm5_next owner {owner!r}: cannot fuse NVFP4 projections into "
                        f"{fused_path!r}: each source has its own per-tensor scales "
                        "(weight_scale_2, input_scale); exclude the sources from quantization"
                    )
                merged = {"weight": torch.cat([g["weight"] for g in groups], dim=0)}
                if all("weight_scale_inv" in g for g in groups):
                    merged["weight_scale_inv"] = torch.cat(
                        [g["weight_scale_inv"] for g in groups], dim=0
                    )
                load_linear(fused_path, [merged])

        if experts is not None:
            tensors_per_expert = {
                QuantAlgo.FP8_BLOCK_SCALES: 6,
                QuantAlgo.NVFP4: 12,
            }.get(quant_algo, 3)
            expected = tensors_per_expert * len(local_expert_ids)
            if collect_all_expert_input_scales:
                # One input_scale per projection of every non-local expert.
                expected += len(proj_to_w) * (experts.num_experts - len(local_expert_ids))
            if len(moe_weights) != expected:
                raise ValueError(
                    f"glm5_next owner {owner!r}: fused MoE collected {len(moe_weights)} expert "
                    f"tensors, expected {expected} (local experts {len(local_expert_ids)})"
                )
            experts.load_weights([moe_weights])
            if hasattr(experts, "post_load_weights"):
                experts.post_load_weights()
            filled.update(n for n in params if n.startswith("mlp.experts."))

        # FP8_BLOCK_SCALES' static activation scales: this checkpoint is
        # activation_scheme='dynamic', so they are never read; zero them rather
        # than leave uninitialized memory behind.
        unused = {
            n for n in params if n not in filled and n.endswith(("input_scale", "inv_input_scale"))
        }
        with torch.no_grad():
            for name in unused:
                params[name].zero_()
        unfilled = sorted(set(params) - filled - unused)
        if unfilled:
            raise ValueError(f"glm5_next owner {owner!r}: no checkpoint tensor reached {unfilled}")


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
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
        from ..modules.top_k import TopK, TopKImplementation

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
        from ..moe.fused_moe.activation import SwigluActivation
        from ..moe.fused_moe.create_moe import create_moe

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
    from ..modules.mhc.hyper_connection import mHC

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

    @staticmethod
    def get_last_token_states(hidden_states: torch.Tensor, attn_metadata) -> torch.Tensor:
        last_tokens = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        return hidden_states[last_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = self.get_last_token_states(hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

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
        is_separate_draft_engine: bool = False,
    ) -> None:
        super().__init__()
        if is_separate_draft_engine:
            raise NotImplementedError(
                "glm5_next MTP runs one-model speculative decoding only (MTP / MTP_EAGLE_ONE_MODEL)"
            )
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
        spec_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_ids, kwargs  # NoPE
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

        hidden_states = self.shared_head.norm(hidden_states)
        if spec_metadata is not None:
            # Two-model-path hook kept for parity with the DeepSeek-V3 MTP layer.
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)
        return hidden_states


# ---------------------------------------------------------------------------
# Whole-model materialization
# ---------------------------------------------------------------------------

#: Routed-expert destinations: consumed by the fused MoE layer's loader rather
#: than placed by name.
_FUSED_MOE_RE = re.compile(
    r"^(?P<layer>model\.layers\.\d+\.mlp)\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<tensor>weight|weight_scale_inv|weight_scale|weight_scale_2|input_scale)$"
)
