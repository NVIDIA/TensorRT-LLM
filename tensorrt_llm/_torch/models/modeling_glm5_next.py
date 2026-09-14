# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn
from transformers import PretrainedConfig

from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantConfig
from ..attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from ..attention.backends.sparse.glm_kpool import (
    Glm5NextCacheManager,
    Glm5NextMamba2Metadata,
    GlmKpoolBackendForwardArgs,
    GlmKpoolSparseParams,
)
from ..attention.backends.utils import create_attention
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..pyexecutor.config_utils import unwrap_glm5_next_text_config
from .checkpoints.hf.glm5_next_weight_mapper import (
    Disposition,
    Glm5NextHfWeightMapper,
    Glm5NextWeightAudit,
    glm5_next_is_quantized,
)
from .modeling_deepseekv3 import DeepseekV3Gate
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model

if TYPE_CHECKING:
    from ..distributed import AllReduceStrategy
    from ..modules.mamba.mamba2_metadata import Mamba2Metadata
    from ..modules.mhc.hyper_connection import mHC
    from ..pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
    from .checkpoints.base_weight_mapper import BaseWeightMapper


# ---------------------------------------------------------------------------
# Literal schedule vocabulary
# ---------------------------------------------------------------------------

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


def get_glm5_next_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text-decoder config, accepting either nesting level.

    The runtime resolves the model from the top-level ``Glm5NextConfig``, but
    every decoder contract (schedules, ranks, MoE, HC) lives on
    ``text_config``. Callers may already hold the inner config, so this is
    idempotent.
    """
    return unwrap_glm5_next_text_config(config)


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
    """Read and cross-validate the literal dispatch lists.

    Raises on any disagreement between the three redundant encodings. A model
    whose attention schedule is inferred from a cadence, or whose MLP schedule
    is inferred from ``first_k_dense_replace``, would silently place the wrong
    module (and therefore the wrong cache descriptor) at some layer; the config
    states both lists explicitly, so there is no reason to guess.
    """
    text = get_glm5_next_text_config(config)
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

    # Third, redundant encoding of the attention schedule. It is not used for
    # dispatch, but a disagreement means the checkpoint is not the variant this
    # bring-up was validated against.
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

    # first_k_dense_replace is asserted against the literal list, not used to
    # build it.
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


def glm5_next_allreduce_strategy() -> AllReduceStrategy:
    """The ``AllReduceStrategy`` for every TP collective this model owns.

    Default ``ONESHOT``: the fused one-shot Lamport kernel (15-20 us for the
    decode-sized [tokens, hidden] messages here, vs 20-75 us for NCCL's LL
    ring at c=8..64; ``MIN_LATENCY`` switches to the two-shot kernel from a
    few hundred tokens, which measured 38 us vs 19 us one-shot for the
    256-token speculative-verify messages). Messages above
    :attr:`Glm5NextAllReduce.SMALL_MAX_TOKENS` go to NCCL regardless. The
    bring-up pinned ``NCCL`` because the ``AUTO`` *autotuner* raced at TP4
    decode; a fixed strategy never enters the autotuner, so the race does not
    apply. ``TLLM_GLM5_ALLREDUCE`` selects another strategy by name
    (``NCCL``, ``MIN_LATENCY``, ``TWOSHOT``, ``AUTO``, ``NCCL_SYMMETRIC``) for
    A/B measurement and as the escape hatch.
    """
    from ..distributed import AllReduceStrategy

    name = os.environ.get("TLLM_GLM5_ALLREDUCE", "ONESHOT").strip().upper()
    try:
        return AllReduceStrategy[name]
    except KeyError as exc:
        raise ValueError(
            f"TLLM_GLM5_ALLREDUCE={name!r} is not an AllReduceStrategy name: "
            f"{[m.name for m in AllReduceStrategy]}"
        ) from exc


def glm5_next_tp_reduces(mapping: Mapping | None) -> bool:
    """Whether a TP branch output is a partial that needs one all-reduce.

    Under attention data parallelism every rank runs its own batch through
    replicated attention / dense weights, so nothing is reduced there (the
    fused MoE does its own dispatch/combine from ``all_rank_num_tokens``).
    """
    return (
        mapping is not None
        and int(getattr(mapping, "tp_size", 1) or 1) > 1
        and not bool(getattr(mapping, "enable_attention_dp", False))
    )


def glm5_next_attention_mapping(mapping: Mapping | None) -> Mapping | None:
    """The Mapping the attention projections shard over: the model's, or a
    TP=1 view of it under attention DP (heads replicated per rank) -- the
    same remap :class:`~tensorrt_llm._torch.attention.mla.MLA` applies."""
    if mapping is None or not getattr(mapping, "enable_attention_dp", False):
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
    """TP all-reduce whose strategy follows the message size at call time.

    The fused one-shot Lamport kernel (``ONESHOT``) is 15-20 us for
    decode-sized ``[tokens, hidden]`` messages where NCCL's LL ring takes
    20-75 us, but from ~1K tokens up the fused kernels are slower than NCCL
    (56 vs 46 us at 1K tokens; 408 vs 355 us two-shot vs NCCL at 8K). One
    ``AllReduce`` per strategy, chosen by the token count -- a Python int, so a
    captured decode graph has a fixed choice. Parameter-free.
    """

    #: Messages with more tokens than this go to NCCL. Measured on this node
    #: (TP4, hidden 4096, bf16): the fused kernels win up to ~256 tokens
    #: (15-19 us vs 20-27 us NCCL), tie around 512, and lose from 1024 tokens
    #: (56 vs 46 us) upward.
    SMALL_MAX_TOKENS = 512

    def __init__(self, mapping: Mapping, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        from ..distributed import AllReduce, AllReduceStrategy

        strategy = glm5_next_allreduce_strategy()
        self.small = AllReduce(mapping=mapping, strategy=strategy, dtype=dtype)
        self.large = (
            self.small
            if strategy == AllReduceStrategy.NCCL
            else AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return (self.small if x.shape[0] <= self.SMALL_MAX_TOKENS else self.large)(x)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _normalize_glm5_next_top_config(config: PretrainedConfig) -> None:
    """Give the composite multimodal config the fields the runtime reads.

    The checkpoint's top-level ``Glm5NextConfig`` carries no
    ``num_hidden_layers`` or ``torch_dtype`` -- both live on ``text_config`` --
    but ``DecoderModel``/``DecoderModelForCausalLM`` and executor capacity
    planning read them from the config they are handed. Copy them up once,
    from the text config, rather than teaching every runtime consumer about
    the wrapper. Values already present are left alone.
    """
    text = get_glm5_next_text_config(config)
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
        as_dev = lambda vals, dtype: torch.tensor(vals, dtype=dtype, device=device)  # noqa: E731
        return cls(
            positions=as_dev(positions, torch.long),
            request_ids=as_dev(request_ids, torch.int32),
            final_positions=as_dev(finals, torch.long),
            final_request_ids=as_dev(final_ids, torch.int32),
        )


@dataclass
class Glm5NextRuntimeContext:
    """Per-forward schedule arguments derived once from AttentionMetadata.

    The sparse layers take *schedule* values (request boundaries, positions)
    plus the prepared ``metadata`` itself -- their attention backend derives
    every cache pool, block table, and visible length from that metadata. The
    KDA layers consume the prepared metadata directly (the shared Kimi KDA
    mixer reads its pools and slot ids from it). This object is built
    once per model forward; requests are packed context-first, matching the
    executor's batch layout.
    """

    manager: Any
    num_contexts: int
    num_ctx_tokens: int
    num_generations: int
    ctx_cu_seqlens: list[int]
    cached_lens: list[int]
    state_indices: torch.Tensor
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

    def context_rows(self, kpool: int, device: torch.device) -> Glm5NextContextRows:
        rows = self.ctx_rows_cache.get(kpool)
        if rows is None:
            rows = Glm5NextContextRows.build(
                self.ctx_cu_seqlens, self.cached_lens[: self.num_contexts], kpool, device
            )
            self.ctx_rows_cache[kpool] = rows
        return rows

    @property
    def gen_phase(self) -> str:
        """Which attention entry point the generation rows take.

        ``"decode"`` is the single-token path; ``"verify"`` is the
        multi-token speculative verification path (the fused Kimi KDA replay
        kernel: the state is committed after the golden token and the drafts
        are cached for replay once the sampler has decided the accepted count).
        """
        return "decode" if self.gen_tokens_per_request == 1 else "verify"

    def mixed_kwargs(self, layer_idx: int) -> dict[str, Any]:
        """Arguments of the sparse layers' ``forward_mixed`` for a
        context+generation batch.

        The attention module splits the packed tokens at ``num_ctx_tokens``
        and runs its context rows through the prefill kernels and its
        generation rows through the decode kernels; everything outside
        attention runs once over the whole batch (the vLLM layout). The KDA
        layers do this split inside the shared mixer from the metadata.
        """
        return {
            "num_ctx_tokens": self.num_ctx_tokens,
            "prefill": self.sparse_kwargs(layer_idx, "prefill"),
            "decode": self.sparse_kwargs(layer_idx, "decode"),
        }

    def sparse_kwargs(self, layer_idx: int, phase: str) -> dict[str, Any]:
        # Schedule only: the backend (keyed by its own layer_idx) derives the
        # slot-indexed latent/indexer views, block tables, and lengths from
        # the prepared metadata itself.
        del layer_idx
        kwargs: dict[str, Any] = {"metadata": self.metadata}
        if phase == "prefill":
            kwargs.update(
                cached_lens=self.cached_lens[: self.num_contexts],
                cu_seqlens=self.ctx_cu_seqlens,
                ctx_rows_fn=self.context_rows,
            )
        else:
            kwargs.update(kv_lens=self.kv_lens[self.num_contexts :])
            if phase == "verify":
                kwargs.update(tokens_per_request=self.gen_tokens_per_request)
        return kwargs


def _glm5_gen_tokens_per_request(attn_metadata: AttentionMetadata, num_generations: int) -> int:
    """Tokens per generation request, from the metadata's host token counts.

    Plain decode has one; a speculative-decoding target verifying drafts has
    ``1 + runtime_draft_len``, uniformly across the generation rows (the
    executor pads every drafted request to the same width). Derived from
    host ints only, so it is a fixed shape parameter inside CUDA graphs.
    """
    if num_generations <= 0:
        return 1
    num_tokens = getattr(attn_metadata, "num_tokens", None)
    if num_tokens is None:
        # Harness carriers without the runtime's cached token count: the
        # host seq_lens carry the same information (never a captured path).
        num_tokens = int(attn_metadata.seq_lens.sum())
    gen_tokens = int(num_tokens) - int(attn_metadata.num_ctx_tokens)
    if gen_tokens <= 0 or gen_tokens % num_generations:
        raise ValueError(
            f"glm5_next: {gen_tokens} generation tokens do not split evenly over "
            f"{num_generations} generation requests"
        )
    return gen_tokens // num_generations


def glm5_next_visible_lens(attn_metadata: AttentionMetadata, batch: int) -> torch.Tensor | None:
    """Per-request visible lengths (``cached + this step's tokens``) as int64.

    Prefers the attention metadata's own ``kv_lens_cuda`` over the
    ``Glm5NextMamba2Metadata`` copy. Both hold the same values after
    ``prepare()``, but only ``kv_lens_cuda`` receives the engine's in-graph
    corrections: under the overlap scheduler with speculative decoding the
    host prepares generation requests as if every draft of the previous step
    had been accepted, and ``_preprocess_inputs`` subtracts the rejected
    count on device (``previous_kv_lens_offsets_cuda``) right before the
    forward; the speculative worker likewise rewinds it between draft steps.
    Reading the host-derived copy there positions the new latent/indexer rows
    past the real prefix and attends stale page contents -- observed as
    non-deterministic MTP output under config E. Returns ``None`` when the
    metadata carries no ``kv_lens_cuda`` (harness carriers), so callers fall
    back to the GLM buffer. The int32 -> int64 cast is a device op with no
    host sync, so it is legal inside CUDA-graph capture.
    """
    live = getattr(attn_metadata, "kv_lens_cuda", None)
    if live is None:
        return None
    return live[:batch].to(torch.long)


def build_glm5_next_runtime_context(
    attn_metadata: AttentionMetadata,
    *,
    kv_lens_source: str = "glm",
) -> Glm5NextRuntimeContext:
    """Derive the per-forward cache arguments from prepared metadata.

    Requires ``attn_metadata.prepare()`` to have run: that is what attaches
    ``mamba_metadata`` (the manager is a ``BaseMambaCacheManager``) and fills
    its batch-ordered ``state_indices``. ``cached_lens`` follows the runtime's
    own convention -- tokens already in the cache, excluding the ones in this
    step -- which is exactly what ``forward_prefill``/``forward_decode`` seed
    and position from.

    Visible lengths come from :func:`glm5_next_visible_lens` (the metadata's
    device-corrected ``kv_lens_cuda``) whenever the metadata carries it, for
    both the target and the MTP draft layer; the ``Glm5NextMamba2Metadata``
    copy is the fallback for harness carriers. ``kv_lens_source`` is kept for
    call-site documentation (``"metadata"`` marks the draft layer, whose
    lengths the speculative worker rewinds in place between draft steps) and
    to reject unknown values; it no longer changes the source when
    ``kv_lens_cuda`` is present.
    """
    manager = attn_metadata.kv_cache_manager
    if manager is None:
        raise ValueError("glm5_next requires a kv cache manager; got None")
    mamba_metadata = attn_metadata.mamba_metadata
    if mamba_metadata is None or mamba_metadata is False:
        raise ValueError(
            "glm5_next requires mamba_metadata; call attn_metadata.prepare() "
            "with the Glm5NextCacheManager attached"
        )
    if kv_lens_source not in ("glm", "metadata"):
        raise ValueError(f"glm5_next: unknown kv_lens_source {kv_lens_source!r}")
    batch = int(attn_metadata.seq_lens.shape[0])
    num_contexts = int(attn_metadata.num_contexts)
    num_generations = batch - num_contexts
    gen_tokens_per_request = _glm5_gen_tokens_per_request(attn_metadata, num_generations)

    if getattr(mamba_metadata, "glm_block_tables", None) is not None:
        # Persistent path: every tensor below is a prepare()-refreshed buffer
        # slice, so this function does no allocation, no H2D, and no host
        # sync -- it is safe to run inside CUDA graph capture, and replays
        # read the refreshed values at the same addresses.
        kv_lens = glm5_next_visible_lens(attn_metadata, batch)
        if kv_lens is None:
            if kv_lens_source == "metadata":
                raise ValueError(
                    "glm5_next draft layer needs attn_metadata.kv_lens_cuda (the "
                    "TRTLLM metadata family); the attached metadata has none"
                )
            kv_lens = mamba_metadata.glm_kv_lens[:batch]
        return Glm5NextRuntimeContext(
            manager=manager,
            num_contexts=num_contexts,
            num_ctx_tokens=int(attn_metadata.num_ctx_tokens),
            num_generations=num_generations,
            ctx_cu_seqlens=mamba_metadata.glm_ctx_cu_seqlens,
            cached_lens=mamba_metadata.glm_cached_lens_host,
            state_indices=mamba_metadata.state_indices[:batch],
            kv_lens=kv_lens,
            metadata=attn_metadata,
            gen_tokens_per_request=gen_tokens_per_request,
        )

    # Legacy eager construction, kept for harnesses whose fake managers do
    # not attach the GLM metadata buffers. It allocates and copies, so it
    # must never run inside a captured region. (The sparse backend applies
    # the same rule to its own metadata-derived block tables.)
    if getattr(attn_metadata, "is_cuda_graph", False):
        raise RuntimeError(
            "glm5_next CUDA-graph execution requires the Glm5NextCacheManager's "
            "Glm5NextMamba2Metadata (persistent prepare()-refreshed buffers); "
            "the attached mamba_metadata has no glm_block_tables"
        )
    lens = attn_metadata.seq_lens.tolist()
    kv_params = attn_metadata.kv_cache_params
    if kv_params is None or kv_params.num_cached_tokens_per_seq is None:
        raise ValueError("glm5_next requires kv_cache_params.num_cached_tokens_per_seq")
    cached_lens = [int(n) for n in kv_params.num_cached_tokens_per_seq[:batch]]

    ctx_cu = [0]
    for length in lens[:num_contexts]:
        ctx_cu.append(ctx_cu[-1] + int(length))

    device = torch.device("cuda", torch.cuda.current_device())
    kv_lens = torch.as_tensor(
        [c + n for c, n in zip(cached_lens, lens)], dtype=torch.long, device=device
    )

    live = glm5_next_visible_lens(attn_metadata, batch)
    if live is not None:
        kv_lens = live

    return Glm5NextRuntimeContext(
        manager=manager,
        num_contexts=num_contexts,
        num_ctx_tokens=int(attn_metadata.num_ctx_tokens),
        num_generations=num_generations,
        ctx_cu_seqlens=ctx_cu,
        cached_lens=cached_lens,
        state_indices=mamba_metadata.state_indices[:batch],
        kv_lens=kv_lens,
        metadata=attn_metadata,
        gen_tokens_per_request=gen_tokens_per_request,
    )


@register_auto_model("Glm5NextForCausalLM")
class Glm5NextForCausalLM(SpecDecOneEngineForCausalLM):
    """GLM-5.3-Flash text decoder with an optional one-model MTP drafter.

    The speculative base class owns logits processing and the draft/verify
    lifecycle. This class narrows the composite config and loads each rank's
    checkpoint shard, including the optional appended MTP layer. Vision weights
    are excluded explicitly by :func:`audit_glm5_next_checkpoint`.
    """

    @property
    def mamba_metadata_cls(self) -> type[Mamba2Metadata]:
        """Metadata with paged tables refreshed before CUDA graph replay."""
        return glm5_next_mamba_metadata_cls()

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        text_config = get_glm5_next_text_config(model_config.pretrained_config)
        if model_config.mapping.enable_attention_dp and model_config.spec_config is not None:
            raise ValueError(
                "glm5_next does not support attention DP with speculative decoding; "
                "set enable_attention_dp=False when enabling MTP."
            )
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
        spec_config = getattr(model_config, "spec_config", None)
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
        # One provenance line per rank: engine-scale runs (LLM API / serving)
        # spawn MPI workers whose model objects the driver cannot introspect,
        # so the resolved production stack is published through the worker log.
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
        """Max sequence length the runtime sizes KV/mamba caches for.

        The executor calls this during capacity planning. GLM-5.3-Flash declares
        ``max_position_embeddings=1048576`` and is fully NoPE (no rope-factor
        scaling), so the value is the text config's directly.
        """
        return int(self.text_config.max_position_embeddings)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Opt this model into ``KVCacheManagerV2``.

        The hybrid latent-KV + pool-indexer + recurrent/conv state is owned by a
        single ``Glm5NextCacheManager`` (a ``MambaHybridCacheManagerV2`` subclass,
        :func:`glm5_next_cache_manager_cls`); V1 cannot express it. This is the
        ``"auto"`` -> V2 resolution hook the runtime consults.
        """
        return "V2"

    def attention_type(self, layer_idx: int) -> str:
        """The literal attention module type for ``layer_idx``."""
        return self.schedule.attention[layer_idx]

    def mlp_type(self, layer_idx: int) -> str:
        """The literal feed-forward module type for ``layer_idx``."""
        return self.schedule.mlp[layer_idx]

    def audit_checkpoint(self, keys: Iterable[str]) -> Glm5NextWeightAudit:
        """Resolve every checkpoint key against this model's destinations."""
        mapper = Glm5NextHfWeightMapper()
        mapper.init_model_and_config(self, self.model_config)
        return mapper.audit(keys)

    # -- whole-model materialization --------------------------------------

    def load_weights(
        self,
        weights: Any,
        weight_mapper: BaseWeightMapper | None = None,
        *,
        device_map: dict[Any, Any] | None = None,
    ) -> None:
        """Materialize and fill the whole text model from a raw checkpoint.

        ``weight_mapper`` is the :class:`Glm5NextHfWeightMapper` registered for
        this architecture (the runtime's ``ModelLoader`` hands over the
        initialized instance; harnesses may omit it and one is created here).
        It owns every checkpoint-key decision -- the audit, the key remap and
        the destination owner -- while this method owns the materialization:
        exact-shape placement one owner at a time, then the decode fusions.
        A generic HF mapper is rejected: its module-name rules cannot place
        this checkpoint.

        ``weights`` is any mapping from checkpoint key to the tensor **as
        stored** -- e4m3 payloads and their FP32 block scales are copied
        verbatim, never dequantized, because excluded modules are published in
        BF16 and quantized ones in e4m3 with a scale, with no overlap. That
        makes the load a 1:1 placement and keeps the resident model at the
        checkpoint's own 328 GB.

        ``device_map`` maps each owner -- a layer index, or ``"embed"``,
        ``"norm"``, ``"head"`` -- to a device. The model is expected to have
        been constructed on ``meta``: each owner is materialized directly onto
        its target device and filled immediately, so peak memory is one layer
        above the final footprint rather than a second full copy.

        A checkpoint tensor that finds no parameter, and a parameter that
        receives no tensor, are both errors: either one leaves a model that
        still runs and still looks plausible.
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
                "glm5_next whole-model loading requires the block-FP8 build "
                "(quantized=True). Dequantizing all 288 experts of 42 routed "
                "layers to bf16 would double the resident model to ~656 GB and "
                "move it four times further from the source's own arithmetic."
            )
        # Decoder stack plus the aliased MTP draft layer(s), if any: owners
        # ``45..`` are the draft layers, placed from the checkpoint's own
        # ``layers.45.*`` keys instead of being allowlisted away.
        num_layers = self.schedule.num_layers + len(self.mtp_layers)
        audit = weight_mapper.audit(list(weights.keys()))

        # Each owner is materialized and filled on its own, so peak memory is
        # one layer above the final footprint rather than a second full copy.
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
        for key, disposition in audit.disposition.items():
            if disposition == Disposition.IGNORED:
                continue
            dest = weight_mapper.destination(key)
            owner = weight_mapper.owner(dest, num_layers) if dest else None
            if owner is None:
                raise ValueError(f"glm5_next has no destination owner for {key!r} -> {dest!r}")
            if owner in remote:
                continue
            by_owner.setdefault(owner, []).append((key, dest))
        if audit.unresolved:
            raise ValueError(f"glm5_next cannot place {sorted(audit.unresolved)[:5]}")

        default_device = torch.device("cuda", torch.cuda.current_device())
        for owner, (module, prefix) in targets.items():
            if owner in remote:
                continue
            device = torch.device((device_map or {}).get(owner, default_device))
            module.to_empty(device=device)
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
          filtered to ``initial_local_expert_ids``.
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
        moe_weights: dict[str, torch.Tensor] = {}
        proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

        def materialize(t: Any) -> torch.Tensor:
            # Lazy safetensors slice: indexing materializes only this tensor.
            return t if torch.is_tensor(t) else t[:]

        for key, dest in sorted(entries):
            local = dest.removeprefix(prefix)
            lazy = weights[key]

            moe = _FUSED_MOE_RE.match(dest.removesuffix("_scale_inv").removesuffix(".weight"))
            if moe is not None:
                expert_id = int(moe.group("expert"))
                if expert_id in local_expert_ids:  # other EP ranks' bytes are never read
                    suffix = ".weight_scale_inv" if dest.endswith("_scale_inv") else ".weight"
                    moe_weights[f"{expert_id}.{proj_to_w[moe.group('proj')]}{suffix}"] = (
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

        def mark_linear(mod_path: str) -> None:
            for param_name in ("weight", "weight_scale", "bias"):
                # mod_path is "" when the owner *is* the Linear (the LMHead).
                name = f"{mod_path}.{param_name}" if mod_path else param_name
                if name in params:
                    filled.add(name)

        for mod_path, group in linear_groups.items():
            named_modules[mod_path].load_weights([group])
            mark_linear(mod_path)
        for fused_path, parts in fused_groups.items():
            dest_mod = named_modules[fused_path]
            groups = [parts[i] for i in sorted(parts)]
            if (
                isinstance(dest_mod, Linear)
                and getattr(dest_mod.weights_loading_config, "weight_mode", None) is not None
                and dest_mod.weights_loading_config.weight_mode.name == ("FUSED_GATE_UP_LINEAR")
            ):
                # GatedMLP: the Linear shards each half over this rank's
                # intermediate range and stacks them [gate; up] itself.
                dest_mod.load_weights(groups)
            else:
                # Replicated fusions: one row-concatenated tensor (and block
                # scales), the order being the module's declared source order.
                merged = {"weight": torch.cat([g["weight"] for g in groups], dim=0)}
                if all("weight_scale_inv" in g for g in groups):
                    merged["weight_scale_inv"] = torch.cat(
                        [g["weight_scale_inv"] for g in groups], dim=0
                    )
                dest_mod.load_weights([merged])
            mark_linear(fused_path)

        if experts is not None:
            expected = 6 * len(local_expert_ids)
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
        # than leave to_empty garbage behind.
        unused = {n for n in params if n.endswith(("input_scale", "inv_input_scale"))}
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
    """GLM-5.3-Flash KDA layer: the shared Kimi KDA mixer, configured.

    The recurrence, convolution, projections, mixed context+generation
    batches and the speculative-verify replay kernel are all the shared
    module's (``trtllm::kda_prefill`` / ``kda_decode`` / ``kda_mtp_decode``
    with the FLA fallbacks), reading the ``MambaHybridCacheManagerV2`` pools
    the way Kimi K3 does. What GLM-5.3-Flash configures differently:

    * the **low-rank output gate** ``g_b_proj(g_a_proj(x))`` (the checkpoint
      has no full-rank ``g_proj``), selected through the mixer's own
      ``use_full_rank_gate`` config key and served by its fused
      ``[f_a | g_a | b]`` / ``[f_b; g_b]`` decode projections;
    * ``A_log`` / ``dt_bias`` are **published in fp32** and kept so.

    The exact-placement loader shards this rank's head range itself
    (:meth:`shard_checkpoint_tensor`), as the projections are the mixer's own
    local-width ``nn.Linear`` modules rather than Mapping-aware ``Linear``.
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
    ) -> None:
        del dtype  # the shared mixer is bf16 (fp32 gate parameters and pools)
        # The checkpoint's ``linear_attn_config`` does not name the gate rank:
        # GLM-5.3-Flash's output gate is always the low-rank ``g_a``/``g_b``
        # pair, which the shared mixer reads from the same key Kimi K3 uses.
        config.linear_attn_config.setdefault("use_full_rank_gate", False)
        linear = dict(config.linear_attn_config)
        # The mixer's own row-parallel o_proj AllReduce (fixed strategy; the
        # size-aware Glm5NextAllReduce used elsewhere is a perf option, not a
        # correctness requirement, and AUTO's autotuner raced at TP4 decode).
        super().__init__(
            config, layer_idx, mapping=mapping, allreduce_strategy=glm5_next_allreduce_strategy()
        )
        # The checkpoint publishes ``A_log`` / ``dt_bias`` in fp32 (the mixer
        # stores them in bf16 by default; the kernels consume fp32 copies
        # either way). Rebuilt without ``.detach``/``.data``, which MetaInitMode
        # rejects on meta tensors.
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
        self.qkv_dim = self.proj_size
        self.total_qkv_dim = self.total_num_heads * self.head_dim

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
    """Construction arguments shared by every Mapping-aware ``Linear`` here.

    The quant config is the checkpoint's (``FP8_BLOCK_SCALES`` with its
    published ``modules_to_not_convert``); the base class's
    ``apply_quant_config_exclude_modules`` then flips the excluded modules to
    bf16 by *runtime module name* before weights are created, so no
    model-specific plan is needed. ``disable_deep_gemm`` pins the block-FP8
    GEMM to ``fp8_quantize_1x128`` + the CuTe DSL Blackwell kernel (the same
    arithmetic the routed experts run; DeepGEMM is not available here).
    Direct (harness) construction without a ``model_config`` yields plain
    bf16 modules over ``mapping``.
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
        "allreduce_strategy": glm5_next_allreduce_strategy(),
        "disable_deep_gemm": True,
    }


class Glm5NextIndexer(nn.Module):
    """Pool-compressed DSA indexer (``Glm5NextTextIndexer``).

    Stock DeepSeek-V3.2 DSA selects ``index_topk`` individual keys. This one
    selects ``index_topk / index_kpool`` *compressed pools*, expands each back
    into its member positions, and always appends the incomplete tail, so the
    two are not interchangeable even though both end up with ~2048 positions.

    The backend caches each token's key and compression gate, plus the pooled
    key at each pool's first row. It updates the affected pool when tokens are
    appended; only complete pools are scored, and the current tail is selected
    separately so future tokens cannot influence a query.

    Unlike the HF module there is no left padding here: TensorRT-LLM stores
    exactly one request's tokens per cache slot starting at position 0, so
    ``first_key`` is always 0 and the packed validity channel HF carries for
    padded batches is replaced by the request's own ``kv_len``.

    Tensor parallelism: the whole indexer is replicated (the vLLM/SGLang
    ownership) -- every rank runs all 32 scoring heads on the replicated
    pool-key path (``wk``, ``k_norm``, the APE and compress gate) and selects
    identical pool/tail/-1 indices by identical compute, with no collective.
    The two scoring GEMMs are tiny (1536->4096, 4096->32), far cheaper than
    the fp32 ``[tokens, pools]`` score all-reduce the sharded-head design
    needed before top-k.

    Decode and prefill score cached *pool keys* (the trailing
    ``head_dim`` columns of each pool's first-member cache row, maintained by
    the backend's ``update_pool_keys``) through the fused paged kernels.
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
        self.tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(mapping, "tp_rank", 0) or 0)
        if self.total_n_heads % self.tp_size:
            raise ValueError(
                f"glm5_next indexer has {self.total_n_heads} scoring heads, not "
                f"divisible by tp_size {self.tp_size}"
            )
        # Replicated scoring heads (vLLM/SGLang ownership): every rank runs
        # all 32 heads and selects identical indices by identical compute,
        # so no score collective is needed before top-k.
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
        # Decode top-k over the fused pool scores: the DSA indexer's TopK
        # module. The CuTe-DSL radix kernel is bounded by each request's
        # candidate count (1-6 us here vs 20-30 us for the CUDA radix kernel
        # and ~40 us for torch.topk over the capacity-wide row); identical
        # selections measured. Falls back to the CUDA radix kernel where the
        # CUTLASS DSL is unavailable.
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
        from ..modules.top_k import TopK, TopKImplementation

        self.pool_top_k = TopK(
            self.select_k,
            prefill_implementation=TopKImplementation.TORCH,
            decode_implementation=(
                TopKImplementation.CUTE_DSL_RADIX
                if IS_CUTLASS_DSL_AVAILABLE
                else TopKImplementation.CUDA_RADIX
            ),
        )

    @property
    def cache_state_dim(self) -> int:
        """Width of one indexer cache row: ``[k | gate | pool key]``.

        The trailing ``head_dim`` columns hold, on a pool's first-member row,
        the pool's compressed key -- maintained incrementally by the backend
        so decode scores cached pool keys instead of rebuilding every pool
        from the packed state each step.
        """
        return 3 * self.head_dim

    @property
    def packed_state_dim(self) -> int:
        """Width of the cached per-token state: ``[k | gate]``."""
        return 2 * self.head_dim

    def project_state(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(packed [k(head_dim) | gate(head_dim)], head weights [n_heads])``
        from the one fused input GEMM."""
        out = self.wk_gate_wp(hidden_states)
        hd = self.head_dim
        packed = torch.cat([self.k_norm(out[:, :hd]), out[:, hd : 2 * hd]], dim=-1)
        return packed, out[:, 2 * hd :]

    def packed_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-token ``[k(head_dim) | gate(head_dim)]`` written to the cache."""
        return self.project_state(hidden_states)[0]


class Glm5NextSparseAttention(nn.Module):
    """Fully NoPE sparse MLA with a pool-compressed indexer.

    ``qk_rope_head_dim`` is 0 on this checkpoint and ``mla_use_nope`` is true:
    there is no text rotary call and no rotary cache. ``indexer_rope_interleave``
    is present in the config but vestigial for the text path, so no rotary
    branch is created for it -- long-range position sensitivity comes from the
    causal KDA layers and the indexer's learned pool APE.

    Layering follows the attention developer guide. This module owns the
    module math only: low-rank q/kv projections and norms, the pool indexer's
    scoring/selection (model-layer sparse prediction, as in MiniMax-M3), the
    absorbed-MLA query/value reassociation, and ``o_proj``. Everything below
    that -- the paged latent/indexer cache path and the sparse-MLA core --
    belongs to ``self.attn_backend``, a
    :class:`~tensorrt_llm._torch.attention.backends.sparse.glm_kpool.GlmKpoolSparseAttention`:
    a ``TrtllmAttention`` subclass (the fully-NoPE branch of the TRTLLM sparse
    family) constructed through the standard ``create_attention(...)``
    dispatch on the configured backend slot (``ModelConfig.attn_backend``,
    default TRTLLM) with ``SparseParams(algorithm="glm_kpool")``. Its typed
    metadata family is ``TrtllmAttentionMetadata`` (``attn_backend.Metadata``),
    the class the engine constructs for this model.

    Cache ownership: one latent ``kv_lora_rank``-wide entry per token (the
    pre-``kv_b_proj`` latent, not expanded K/V) plus the indexer's packed
    ``[k | gate]`` state, both held by the one hybrid ``KVCacheManagerV2`` and
    read/written only by the backend, which derives every pool, block table,
    and visible length from the prepared attention metadata it is handed --
    this module passes schedule values and the metadata, never raw pools. The
    pool-expanded selection travels to the backend inside the standard
    ``AttentionForwardArgs.sparse_backend_args`` carrier.

    ``kv_b_proj`` is on this checkpoint's ``modules_to_not_convert`` list
    (BF16), so absorption is a reassociation of the same BF16 weights, not a
    dequantization.

    Tensor parallelism: with ``tp_size > 1`` this module owns
    ``num_heads = 64 // tp_size`` local query heads and the matching rows of
    the column-sharded ``q_b_proj``/``kv_b_proj`` (so the absorbed per-head
    views are local by construction), while the low-rank latents
    (``q_a``/``kv_a``) and both norms stay replicated and the row-sharded
    ``o_proj`` returns a partial that this module reduces once through
    :class:`Glm5NextAllReduce` -- the DeepSeek-V3 MLA ownership with a
    message-size-aware collective. The latent cache and the indexer's packed state stay
    *complete* (512- and 256-wide) on every rank: ``num_kv_heads == 1`` is
    never divided, all ranks compute identical latent/packed rows from the
    replicated projections, and each rank's backend reads its own full copy.
    The backend is constructed with the local head count, exactly as MLA's
    per-rank ``num_heads // tp_size``. At ``tp_size == 1`` construction and
    math are byte-identical to the pre-TP module.
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
        self.tp_size = int(getattr(attn_mapping, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(attn_mapping, "tp_rank", 0) or 0)
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
                f"({self.qk_rope_head_dim}) would need a rotary path this bring-up does not have"
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
        self.tp_all_reduce = Glm5NextAllReduce(mapping) if glm5_next_tp_reduces(mapping) else None
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
            sparse_params=GlmKpoolSparseParams(
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                q_lora_rank=self.q_lora_rank,
                v_head_dim=self.v_head_dim,
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

    def _decode_projections(
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
        out = torch.bmm(per_head.transpose(0, 1), w_v_t)  # [H, T, v_head]
        out = self.o_proj(out.transpose(0, 1).reshape(tokens, -1))
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
        request_index: int | None = None,
        rows_per_request: int = 1,
        reduce: bool = True,
        request_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused indexer selection over the paged cache, then sparse attention.

        Score the cached pool keys, top-k, expand + tail + row translation,
        attend. ``visible[i]`` is query row ``i``'s visible length (its own
        position + 1); ``request_index`` selects one context request's block
        table, ``request_ids`` (``[rows]`` int32) the block table of each
        packed context row (``None`` for both: the generation rows). Every step is a fixed-shape
        kernel with work proportional to the visible length, not the buffer
        capacity, so decode replays it inside CUDA graphs and prefill no
        longer materializes ``[tokens, pools, head_dim]`` gathers.
        """
        indexer = self.indexer
        rows = q_resid.shape[0]
        q_index = indexer.wq_b(q_resid).view(rows, indexer.n_heads, indexer.head_dim)
        # Generation rows read their visible length from the metadata unless
        # each request contributes several rows (context or verification).
        plain_decode = request_index is None and request_ids is None and rows_per_request == 1
        kv_lens = None if plain_decode else visible
        scores = self.attn_backend.score_pools(
            q_index,
            index_weights,
            metadata,
            q_scale=indexer.softmax_scale,
            w_scale=indexer.head_mix_scale,
            request_index=request_index,
            kv_lens=kv_lens,
            rows_per_request=rows_per_request,
            request_ids=request_ids,
        )
        num_cand = (visible // indexer.index_kpool).to(torch.int32)
        selected = torch.empty(rows, indexer.select_k, dtype=torch.int32, device=scores.device)
        indexer.pool_top_k(
            scores,
            selected,
            is_prefill=False,
            sequence_lengths=num_cand,
            scan_lengths=num_cand,
        )
        topk_rows = self.attn_backend.expand_selection(
            selected,
            metadata,
            request_index=request_index,
            kv_lens=kv_lens,
            rows_per_request=rows_per_request,
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
        """Context phase, including continuation chunks -- all requests at once.

        ``cached_lens[i]`` is how many tokens of request ``i`` are already in
        the cache, so a chunk is scored against the whole visible prefix rather
        than only against its own tokens. Scoring a chunk in isolation is the
        classic chunked-prefill bug here: it still passes a one-shot test.
        Only schedule values are passed here; every pool write and read goes
        through the backend's metadata-derived cache path.

        The packed context tokens of every request go through each stage in
        one launch (projections, cache write, pool-key refresh, scoring,
        top-k, expansion, attention): the kernels address each row's block
        table through :class:`Glm5NextContextRows.request_ids`. A per-request
        Python loop here was the dominant cost of mixed context+generation
        iterations (measured 50% GPU idle with ~17 short prompts in flight).
        ``ctx_rows_fn`` (the runtime context's cache) shares the schedule
        across the sparse layers of one forward.
        """
        kpool = self.indexer.index_kpool
        device = hidden_states.device
        if ctx_rows_fn is not None:
            rows = ctx_rows_fn(kpool, device)
        else:
            rows = Glm5NextContextRows.build(cu_seqlens, cached_lens, kpool, device)
        q_resid, latent, packed, index_weights = self._decode_projections(hidden_states)
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
        return self._select_and_attend_paged(
            q_resid,
            query,
            index_weights,
            rows.positions + 1,
            metadata,
            AttentionInputType.context_only,
            reduce=reduce,
            request_ids=rows.request_ids,
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_lens: torch.Tensor,
        metadata: AttentionMetadata,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Generation phase: one token per request, fully batched.

        CUDA-graph contract: every shape here is a function of the *buffer*
        geometry (the metadata's block-table width times tokens_per_block),
        never of the current lengths, and every request-dependent value
        (``kv_lens`` and the metadata's block tables) is a device tensor
        refreshed by metadata ``prepare()`` outside the captured region. No
        ``.item()``/``.tolist()``, no per-request Python loop, no
        host->device copy, and no data-dependent branch runs on this path, so
        a captured decode graph replays correctly as lengths grow and slots
        are reused.

        ``kv_lens[i]`` is the request's visible length *including* the token
        being decoded, so the new token's position is ``kv_lens[i] - 1``; it
        must be the same prepare()-refreshed lengths the metadata carries,
        sliced to the generation rows. Positions at or beyond a request's
        ``kv_lens`` gather page-0 garbage in the indexer prefix; the backend
        masks them by replacement and the indexer's own validity masks exclude
        them from pools, selection, and the tail. The attention core never
        gathers the latent at all: the backend reads the paged pool directly
        through its storage row view derived from the metadata, and only
        selected (valid) positions are translated into row ids -- sentinels
        stay ``-1``. There is no empty-row assertion here: it would force a
        host sync, which is illegal under CUDA-graph capture -- and a decode
        query is always covered by construction, either by the
        always-selected tail (``visible % kpool != 0``) or by the final
        complete pool, whose last member *is* the query position
        (``visible % kpool == 0``).
        """
        batch = hidden_states.shape[0]
        positions = kv_lens - 1  # [B]
        indexer = self.indexer
        q_resid, latent, packed, index_weights = self._decode_projections(hidden_states)
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
        )

    def forward_mixed(
        self,
        hidden_states: torch.Tensor,
        *,
        num_ctx_tokens: int,
        prefill: dict[str, Any],
        decode: dict[str, Any],
    ) -> torch.Tensor:
        """Context rows through the prefill path, generation rows through the
        fused decode path, one TP reduction over the concatenated output."""
        ctx = self.forward_prefill(hidden_states[:num_ctx_tokens], reduce=False, **prefill)
        gen = self.forward_decode(hidden_states[num_ctx_tokens:], reduce=False, **decode)
        out = torch.cat([ctx, gen], dim=0)
        return out if self.tp_all_reduce is None else self.tp_all_reduce(out)

    def forward_verify(
        self,
        hidden_states: torch.Tensor,
        kv_lens: torch.Tensor,
        metadata: AttentionMetadata,
        tokens_per_request: int,
    ) -> torch.Tensor:
        """Generation phase with ``tokens_per_request`` tokens per request.

        The speculative-decoding target scores the golden token plus the
        drafts in one pass; the MTP draft layer's first step sees the same
        packed layout. Request ``i``'s tokens sit at cache positions
        ``kv_lens[i] - T .. kv_lens[i] - 1`` (``kv_lens`` already counts them,
        exactly as in :meth:`forward_decode`), so the latent/indexer rows are
        appended there and every query is scored against its *own* visible
        prefix: pool scoring masks pools whose last member lies beyond the
        query's position, and selection expansion builds the tail from the query's
        own position, so token ``j`` never sees tokens ``j+1..``. Rows written
        for drafts that are later rejected are simply overwritten by the next
        step, which re-appends at the rewound ``kv_lens`` -- the same
        positional-cache convention MLA relies on.

        Every shape is a function of ``tokens_per_request`` and the buffer
        geometry, with no host sync, so a captured verification graph replays
        against refreshed lengths and tables.
        """
        tokens_per_request = int(tokens_per_request)
        if tokens_per_request <= 1:
            return self.forward_decode(hidden_states, kv_lens, metadata)
        num_tokens = hidden_states.shape[0]
        batch = num_tokens // tokens_per_request
        if batch * tokens_per_request != num_tokens or kv_lens.shape[0] != batch:
            raise ValueError(
                f"glm5_next sparse verify: {num_tokens} tokens for {kv_lens.shape[0]} requests "
                f"at {tokens_per_request} tokens per request"
            )
        device = hidden_states.device
        steps = torch.arange(tokens_per_request, device=device)
        positions = (kv_lens - tokens_per_request).unsqueeze(1) + steps  # [B, T]
        q_resid, latent, packed, index_weights = self._decode_projections(hidden_states)
        self.attn_backend.append_paged_state(
            latent.view(batch, tokens_per_request, -1),
            packed.view(batch, tokens_per_request, -1),
            positions,
            metadata,
        )
        # Keep the cached pool keys current, in position order, so a pool
        # spanning several drafts ends up with all of its visible members.
        for step in range(tokens_per_request):
            self.attn_backend.update_pool_keys(
                positions[:, step].contiguous(), self.indexer.index_kpool_compress_ape, metadata
            )
        query = self.q_b_proj(q_resid).view(num_tokens, self.num_heads, self.qk_head_dim)
        # Row i of request b sees its own prefix: visible = position + 1.
        return self._select_and_attend_paged(
            q_resid,
            query,
            index_weights,
            (positions + 1).reshape(-1),
            metadata,
            AttentionInputType.generation_only,
            rows_per_request=tokens_per_request,
        )


# ---------------------------------------------------------------------------
# Heterogeneous request state
# ---------------------------------------------------------------------------


def glm5_next_mamba_metadata_cls() -> type[Mamba2Metadata]:
    """Return the model's ``Mamba2Metadata`` subclass (see the sparse backend's ``cache_manager``)."""
    return Glm5NextMamba2Metadata


def glm5_next_cache_manager_cls() -> type[MambaHybridCacheManagerV2]:
    """Return the model's ``KVCacheManagerV2`` subclass (see the sparse backend's ``cache_manager``)."""
    return Glm5NextCacheManager


class Glm5NextGate(DeepseekV3Gate):
    """The shared DeepSeek noaux_tc gate, kept FP32 end to end.

    Same routing math and weights as :class:`DeepseekV3Gate` (selection on
    bias-corrected sigmoid scores, weights gathered from the uncorrected
    scores, normalization before ``routed_scaling_factor``); the two
    overrides are the parameter dtypes and the logits GEMM. The correction
    bias sits around magnitude ~10 while the sigmoid scores it corrects are
    O(1e-2), so ranking turns on inter-expert gaps of 4e-5 - 6e-4; bf16
    resolution at that magnitude is ~1e-2, three orders of magnitude too
    coarse -- it silently changes the top-8 while every aggregate check still
    passes. The router weight is FP32 in the checkpoint, so the logits are an
    FP32 ``F.linear`` (the shared bf16 router GEMM does not apply).
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
            dtype=torch.float32,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=moe_backend,
        )
        self.hidden_size = int(config.hidden_size)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        # FP32 regardless of the MoE backend (the base class picks bf16 for
        # TRTLLM); the fused routing kernels consume the FP32 tensor directly.
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(int(config.n_routed_experts), dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, self.hidden_size)
        return torch.nn.functional.linear(flat.float(), self.weight)


class Glm5NextMoE(nn.Module):
    """Routed experts plus one always-active shared expert.

    The routed experts are a fused-MoE layer built through ``create_moe`` --
    the one selection entry point -- with the DeepSeek noaux_tc routing method
    (:class:`Glm5NextGate`, the shared DeepSeek gate kept FP32) and the
    DSV4-style uniform ``swiglu_limit_scalar``. On this checkpoint (FP8 block
    scales, SM100) the resolver lands on ``TRTLLMGenFusedMoE`` whose
    ``trtllm::fp8_block_scale_moe_runner`` consumes the clamp limit as
    ``gemm1_clamp_limit``; the ``AUTO`` ``moe_backend`` default resolves to
    ``TRTLLM`` for exactly this quant/SM pair in
    ``ModelConfig.resolve_moe_backend``. The shared expert is the shared
    ``GatedMLP`` (DeepSeek-V3 composition): TP-sharded with
    ``reduce_output=False``, its partial summed with the routed partial before
    this module's single all-reduce.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        model_config: ModelConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
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
        # to_empty/materialization onto its final device).
        routing = self.gate.routing_method
        # The experts are uniformly block-FP8 on the published checkpoint;
        # the exclusion patterns concern *other* modules (applied by name by
        # the base class), so the layer-scoped config the fused backend sees
        # carries only the algorithm and block size. A bf16 build (no quant
        # config) gets bf16 experts.
        quant = model_config.get_quant_config()
        experts_quant = (
            QuantConfig(quant_algo=quant.quant_algo, group_size=quant.group_size)
            if quant is not None and quant.quant_algo is not None
            else None
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
        # Reduce routed and TP-sharded shared-expert partials once. Under
        # attention DP the fused layer combines across ranks itself and the
        # shared expert is replicated, so there is nothing to reduce here.
        self.moe_all_reduce = (
            Glm5NextAllReduce(self.mapping) if glm5_next_tp_reduces(self.mapping) else None
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
        routed = self.experts(
            flat,
            self.gate(flat),
            all_rank_num_tokens=all_rank_num_tokens if self.use_dp else None,
        )
        # Routed and shared are both rank partials (a K-dim partial per expert
        # in the TP4 layout, the local-expert partial sum in the TP4/EP4
        # layout). Sum them, then exactly one reduction covers the whole MoE
        # branch -- the DeepSeek-V3 order.
        mixed = routed + self.shared_experts(flat)
        if self.moe_all_reduce is not None:
            mixed = self.moe_all_reduce(mixed)
        return mixed.view_as(x)


# ---------------------------------------------------------------------------
# Hyper-connected decoder
# ---------------------------------------------------------------------------


def glm5_next_hyper_connection(
    config: PretrainedConfig, dtype: torch.dtype = torch.bfloat16
) -> mHC:
    """Build the shared ``mHC`` for one hyper-connection site.

    This reuses TensorRT-LLM's existing manifold-constrained hyper-connection
    rather than adding a model-local one. Two settings are model-specific and
    were pinned by measurement against the source module, not by assumption:

    * ``post_mult_value=2.0`` -- the source computes ``2 * sigmoid(...)`` for the
      block-output placement weights. Leaving the default 1.0 halves them and
      shows up as ``max_abs`` 0.96 on ``post`` against a [0.26, 1.92] range.
    * ``sinkhorn_iters`` is the config's own ``hc_sinkhorn_iters`` (20), not
      ``iters - 1``. The source runs one initial column normalization plus
      ``iters - 1`` row/column rounds, which is what this kernel's ``iters``
      counts; passing 19 leaves a measurable 1.9e-5 residual on ``comb`` versus
      1.3e-6 at 20.

    ``norm_eps`` is the decoder's ``rms_norm_eps`` because the source's
    hyper-connection input norm is constructed with it, while ``eps`` and
    ``sinkhorn_eps`` are the separate ``hc_eps``.
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
    """One decoder layer: two hyper-connection sites wrapping attention and FFN.

    The residual path is *not* an ordinary add. Each site collapses the four
    streams into one sequence with the learned ``pre`` weights, runs the
    sublayer, then writes the result back across the streams as
    ``post * out + comb^T @ residual``. Both module choices come from the two
    literal per-layer lists, never from a cadence or ``first_k_dense_replace``.

    Two entry points share one implementation: the runtime ``forward``
    receives ``AttentionMetadata`` (plus the once-per-forward
    :class:`Glm5NextRuntimeContext`) and derives this layer's cache
    arguments; ``forward_direct`` takes them explicitly and is what the
    component tests call. The runtime path is
    a thin argument-derivation shim over the direct path, so parity between
    them is an argument-sourcing check, not a second implementation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        schedule: Glm5NextSchedule,
        model_config: ModelConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = schedule.attention[layer_idx]
        self.mlp_type = schedule.mlp[layer_idx]
        eps = float(config.rms_norm_eps)

        mapping = model_config.mapping
        if self.attention_type == LINEAR_ATTENTION:
            self.self_attn: nn.Module = Glm5NextLinearAttention(
                config, layer_idx, dtype=dtype, mapping=mapping
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
            Glm5NextMoE(config, model_config, layer_idx, dtype=dtype)
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
            Glm5NextAllReduce(mapping)
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
        """Runtime entry: derive this layer's cache arguments from metadata.

        The executor packs context requests first, then generation requests;
        the two phases run through :meth:`forward_direct` separately, which is
        exact because every non-attention operation here is token-local.
        ``position_ids`` is unused: the text path is fully NoPE and the
        indexer derives its positions from the cached lengths.
        """
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        all_rank_num_tokens = getattr(runtime_ctx.metadata, "all_rank_num_tokens", None)
        if self.attention_type == LINEAR_ATTENTION:
            # The shared KDA mixer splits context / generation rows (prefill,
            # decode, verify) itself from the prepared metadata.
            return self.forward_direct(
                hidden_states,
                all_rank_num_tokens=all_rank_num_tokens,
                attn_metadata=runtime_ctx.metadata,
            )
        derive = runtime_ctx.sparse_kwargs
        if runtime_ctx.num_contexts > 0 and runtime_ctx.num_generations > 0:
            if runtime_ctx.gen_phase == "decode":
                # Mixed context+generation batch: one pass over all tokens
                # (hyper-connections, norms, MoE, the o_proj reduction), with
                # the attention module splitting the rows internally between
                # its prefill and decode kernels -- the vLLM layout. A second
                # small-batch pass per layer would cost a whole decode step
                # per mixed iteration.
                return self.forward_direct(
                    hidden_states,
                    phase="mixed",
                    all_rank_num_tokens=all_rank_num_tokens,
                    **runtime_ctx.mixed_kwargs(self.layer_idx),
                )
            # Speculative verification rows run the replay verify kernel; they
            # keep the split path (prefill rows, then verify rows).
            parts = [
                self.forward_direct(
                    hidden_states[: runtime_ctx.num_ctx_tokens],
                    phase="prefill",
                    all_rank_num_tokens=all_rank_num_tokens,
                    **derive(self.layer_idx, "prefill"),
                ),
                self.forward_direct(
                    hidden_states[runtime_ctx.num_ctx_tokens :],
                    phase="verify",
                    all_rank_num_tokens=all_rank_num_tokens,
                    **derive(self.layer_idx, "verify"),
                ),
            ]
            return torch.cat(parts, dim=0)
        if runtime_ctx.num_contexts > 0:
            return self.forward_direct(
                hidden_states,
                phase="prefill",
                all_rank_num_tokens=all_rank_num_tokens,
                **derive(self.layer_idx, "prefill"),
            )
        # "decode" (one token per request) or "verify" (golden + drafts per
        # request while a speculative-decoding target scores them).
        phase = runtime_ctx.gen_phase
        return self.forward_direct(
            hidden_states,
            phase=phase,
            all_rank_num_tokens=all_rank_num_tokens,
            **derive(self.layer_idx, phase),
        )

    def forward_direct(
        self,
        hidden_streams: torch.Tensor,
        phase: str = "prefill",
        all_rank_num_tokens: list[int] | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """``hidden_streams`` is ``[num_tokens, hc_mult, hidden]``.

        A KDA layer takes ``attn_metadata`` and runs the shared mixer over all
        rows; a sparse layer's ``attn_kwargs`` are forwarded verbatim to its
        ``forward_<phase>``. ``all_rank_num_tokens`` (attention DP) reaches the
        fused MoE.
        """
        residual = hidden_streams
        post, comb, collapsed = self.hc_attn.pre_mapping(hidden_streams)
        normed = self.input_layernorm(collapsed)
        if self.attention_type == LINEAR_ATTENTION:
            attn_out = self.self_attn(normed, attn_kwargs["attn_metadata"])
        else:
            attn_out = getattr(self.self_attn, f"forward_{phase}")(normed, **attn_kwargs)
        hidden_streams = self.hc_attn.post_mapping(attn_out, residual, post, comb)

        residual = hidden_streams
        post, comb, collapsed = self.hc_ffn.pre_mapping(hidden_streams)
        mlp_out = self.run_mlp(self.post_attention_layernorm(collapsed), all_rank_num_tokens)
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
    """Embedding, the 45 hyper-connected decoder layers, and the final readout.

    A ``DecoderModel`` whose hidden state between layers is the four-stream
    tensor ``[num_tokens, hc_mult, hidden]`` rather than ``[num_tokens,
    hidden]``: the stream axis opens at the embedding and closes in
    :meth:`collapse_streams` (unweighted mean plus the final norm), which is
    this model's equivalent of the base class's trailing ``self.norm``.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        super().__init__(model_config)
        config = get_glm5_next_text_config(model_config.pretrained_config)
        schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        # Pipeline parallelism rides the base machinery wholesale: the
        # inter-layer activation is the four-stream tensor [tokens, hc_mult,
        # hidden], and both `forward_after_recv`/`forward_before_send` and
        # `pp_recv/send` are shape-agnostic — the recv buffer on a non-first
        # rank is exactly `expand_streams(embed_tokens.skip_forward(...))`,
        # which is a real contiguous [tokens, hc_mult, hidden] tensor.
        # `__pp_init__` prunes non-local layers; `load_weights` skips
        # pruned owners (see `skipped_remote`); the hybrid cache manager
        # slices its layer masks per rank from `mapping`.
        self.config = config
        self.schedule = schedule
        self.hc_mult = int(config.hc_mult)
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

        self.embed_tokens = Embedding(int(config.vocab_size), int(config.hidden_size), dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(config, i, schedule, model_config, dtype=dtype)
                for i in range(schedule.num_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=float(config.rms_norm_eps), dtype=dtype
        )
        # Read by the one-model MTP drafter factory (``MTPForCausalLM`` passes
        # ``model.aux_stream_dict`` to every MTP layer). This model runs its
        # branches on the main stream, so the draft layer receives an empty map.
        self.aux_stream_dict: dict[Any, Any] = {}

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
        streams = self.expand_streams(inputs_embeds)
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        # Only the decoder stack: under one-model MTP the speculative base
        # class appends the draft layer(s) to ``self.layers`` (so the
        # checkpoint's ``model.layers.45.*`` names resolve), and those run in
        # the speculative worker's draft loop, not here.
        # Generation-only batches take the fused hyper-connection loop. Prefill
        # (and mixed) batches keep the per-layer path so the engine's warmup /
        # KV-cache sizing pass measures the same activation peak that mixed
        # context+generation iterations reach in production (a lower
        # pure-prefill peak over-allocates the KV cache and OOMs later).
        if (
            runtime_ctx.num_contexts == 0
            and runtime_ctx.num_generations > 0
            and self._fused_hc_ok()
        ):
            return self.collapse_streams(self._forward_single_phase_fused_hc(streams, runtime_ctx))
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
        self, streams: torch.Tensor, runtime_ctx: Glm5NextRuntimeContext
    ) -> torch.Tensor:
        """Single-phase forward with fused hyper-connection boundaries.

        Every boundary between two sublayers is ``post_mapping`` of the
        previous site, ``pre_mapping`` of the next, and the next sublayer's
        RMSNorm; ``mHC.fused_hc`` (the in-tree DeepSeek-V4 boundary op) runs
        the three as one kernel, so a layer costs 2 fused boundaries instead
        of 4 mappings + 2 norms. Same math as :meth:`Glm5NextDecoderLayer.
        forward_direct` chained over the stack; the first pre-mapping and the
        last post-mapping stay unfused. Only generation-only batches use it
        (see :meth:`forward`); the phase is still derived here so the loop
        stays valid for a pure-context batch.
        """
        phase = "prefill" if runtime_ctx.num_generations == 0 else runtime_ctx.gen_phase
        num_layers = self.schedule.num_layers
        first = self.layers[0]
        residual = streams
        post, comb, x = first.hc_attn.pre_mapping(streams)
        x = first.input_layernorm(x)
        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            if layer.attention_type == LINEAR_ATTENTION:
                attn_out = layer.self_attn(x, runtime_ctx.metadata)
            else:
                attn_out = getattr(layer.self_attn, f"forward_{phase}")(
                    x, **runtime_ctx.sparse_kwargs(layer_idx, phase)
                )
            residual, post, comb, x = layer.hc_ffn.fused_hc(
                attn_out,
                residual,
                post,
                comb,
                norm_weight=layer.post_attention_layernorm.weight,
                norm_eps=layer.post_attention_layernorm.variance_epsilon,
            )
            mlp_out = layer.run_mlp(x, getattr(runtime_ctx.metadata, "all_rank_num_tokens", None))
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
    """The MTP layer's ``shared_head``: its final norm plus the draft logits.

    ``norm`` is applied by :class:`Glm5NextMTP` at the end of its forward (the
    checkpoint's ``shared_head.norm``); ``forward`` here turns the resulting
    hidden states into logits through the *target's* ``lm_head`` -- the
    checkpoint publishes no separate MTP head weight -- the way the
    speculative worker calls it: ``shared_head(hidden, lm_head, attn_metadata,
    return_context_logits)``. The LM-head TP handling mirrors the DeepSeek-V3
    MTP head exactly, because the worker's greedy draft sampler recovers the
    global argmax from vocab-sharded logits (``is_spec_decoding_head``).
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
    """GLM-5.3-Flash's one next-token-prediction layer (``layers.45``).

    Structure, verified against the checkpoint's own keys and the vLLM GLM-5
    port (``glm5next/nvidia/mtp.py``) since the HF reference implements no
    MTP: ``enorm(embed(next_token)) ++ hnorm(target_hidden) -> eh_proj`` into
    a **plain-residual** decoder block -- the MTP layer has no
    hyper-connection weights, unlike the 45 main layers -- made of the same
    sparse-MLA + k-pool indexer attention and 288+1-expert MoE as the main
    sparse layers, closed by ``shared_head.norm``. The attention and MoE are
    the *same classes* as the main stack (with ``layer_idx = 45``), so the
    draft layer owns its own latent/indexer pages in the one hybrid cache
    manager (the executor appends one attention layer for it) and shares the
    projection-swap, TP-shard, and exact-placement loading contracts.

    Constructed by :class:`~.modeling_speculative.MTPForCausalLM` through
    the model-type dispatch and called by the one-model speculative worker
    as ``mtp_layer(input_ids, position_ids, hidden_states, embed_tokens,
    attn_metadata, all_rank_num_tokens, spec_metadata)``. ``hidden_states``
    are the target's post-final-norm states for the same packed tokens (the
    convention the DeepSeek-V3/Qwen3-Next/vLLM MTP paths all use), and the
    first draft step's token schedule is *identical* to the target
    verification pass (contexts: prompt shifted by one; generation:
    ``1 + runtime_draft_len`` accepted/padded tokens at the same positions),
    so it reuses the target's runtime-context derivation, reading the
    metadata's live ``kv_lens_cuda`` so later draft steps' rewinds are seen.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Any = None,
        is_separate_draft_engine: bool = False,
    ) -> None:
        super().__init__()
        del aux_stream_dict  # single-stream model; accepted for the factory's call shape
        if is_separate_draft_engine:
            raise NotImplementedError(
                "glm5_next MTP runs one-model speculative decoding only (MTP / MTP_EAGLE_ONE_MODEL)"
            )
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        config = get_glm5_next_text_config(model_config.pretrained_config)
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
        self.mlp = Glm5NextMoE(config, model_config, layer_idx, dtype=dtype)
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

        runtime_ctx = build_glm5_next_runtime_context(attn_metadata, kv_lens_source="metadata")
        residual = hidden_states
        attn_in = self.input_layernorm(hidden_states)
        parts = []
        if runtime_ctx.num_contexts > 0:
            parts.append(
                self.self_attn.forward_prefill(
                    attn_in[: runtime_ctx.num_ctx_tokens],
                    **runtime_ctx.sparse_kwargs(self.layer_idx, "prefill"),
                )
            )
        if runtime_ctx.num_generations > 0:
            phase = runtime_ctx.gen_phase
            parts.append(
                getattr(self.self_attn, f"forward_{phase}")(
                    attn_in[runtime_ctx.num_ctx_tokens :],
                    **runtime_ctx.sparse_kwargs(self.layer_idx, phase),
                )
            )
        attn_out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
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
    r"\.(?P<proj>gate_proj|up_proj|down_proj)$"
)
