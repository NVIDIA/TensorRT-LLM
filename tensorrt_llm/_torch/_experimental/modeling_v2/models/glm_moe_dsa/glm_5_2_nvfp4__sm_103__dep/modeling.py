# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: glm-5.2-nvfp4 / sm_103 / dep.

GLM-5.2, modelopt NVFP4 export: 78 layers, hidden 6144, 64 query heads, MLA with
DSA sparse attention (index top-k 2048), layers 0-2 dense and 3-77 MoE (256
routed experts at top-8 plus a shared expert). Layer 78 is the checkpoint's bf16
MTP module, driven by the one-model MTP-Eagle loop. `dep` is attention data
parallelism over eight or sixteen ranks (each rank runs the whole attention on its
own requests) and the experts expert-parallel over the same ranks.

``route`` claims three deployments by the facts it reads: the rank count, whether
the LM head is tensor-parallel in attention DP, and the draft tokens per request.
Two more facts fix what the fast paths can serve there, the KV-cache dtype and the
MoE backend, and only the model config carries them, so this target checks them
first, at construction (``CELLS``, ``target_paths``). Under ``require`` a
deployment outside the cells raises; under ``auto`` it serves no fast path and
runs the built-in forward.

Under attention DP the fast paths that take this target's shapes are rank-local:
the attention's rope, KV append and top-k bridge with an FP8 KV cache
(``rope_kv_append``), the rope of the full-indexer layers' query front
(``indexer_layer_front``), the draft loop (``draft_loop``), the decode step's
prologue (``decode_prologue``) and the host page retention policy
(``host_page_retention``). The plan of a cell serves those and the bases they
extend; the bases decline every call under attention DP. The other fast paths of
the family take tensor-parallel attention, so this target serves none of them: its
decoder layers, MoE modules and draft layer stay the built-in classes, which run
the built-in model's attention-DP communication on every step.

The shell is the built-in ``GlmMoeDsaForCausalLM``: the same module tree, the
same weights and the same engine preferences (KV cache manager, KV transceiver,
sparse-attention defaults), so a built-in context server and this target's
generation server exchange the same KV and indexer-K layout. Construction runs,
in order:

1. the cell check (``target_paths``), from the model config alone, so every rank
   decides alike before any collective;
2. the rank-uniform check of the knobs (``dispatch.check_rank_uniform``), before
   every collective the construction below makes;
3. the dispatch plan (``dispatch.resolve``) of the paths the cell serves, before
   the built-in model is built, so a knob value no rank can parse fails early;
4. the plan is installed (``dispatch.install``): from here on the fast paths read
   it;
5. the built-in model, then the core: its text model adopted as
   ``GlmMoeDsaDepModelingV2``, whose forward hands each step to
   ``PrefillTarget`` or ``DecodeTarget``;
6. ``adopt_model``, which re-classes the attention modules of every layer
   (``shared.attention``) when the plan runs an attention path, and the
   speculative worker (``shared.spec_worker``) when it runs the draft loop;
7. the rank-uniform check of the attached fast-path state
   (``dispatch.check_rank_uniform_adopted``, ``shared.layers.adopted_state``).

The step-loop hooks the target declares to the engine (``runtime_hooks``,
``shared.runtime.make_runtime_hooks``) are built later, on the engine's first
read, after the model is loaded.

The rank-uniform check of step 2 also covers the shared switches the kernels
read and the modeling_v2 mode (``dispatch.rank_uniform_inputs``).

Every step runs under the phase of the target that takes it
(``dispatch.step``): the shell's forward enters it around the built-in forward,
so the model, the speculative worker and its draft loop all run in it. A
prefill step runs the built-in decoder forward and the built-in draft loop; a
decode step runs ``DecodeTarget``'s loop over the built-in layers, the attention
modules' decode composition and the worker's draft loop. Each rank picks its
phase from its own requests; the layers' communication is the built-in one in
both phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    ModelingV2Mode,
    resolved_mode,
)
from tensorrt_llm._torch.attention.attention import maybe_allgather_for_helix_cp
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model, GlmMoeDsaForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..shared import dispatch, runtime
from ..shared.attention import GlmMLA
from ..shared.layers import adopted_state
from ..shared.spec_worker import GlmMtpWorker
from . import weights as _weights

if TYPE_CHECKING:
    from ..shared.kernels.decode_prologue import DecodePrologueHook


# The paths that engage under attention DP with an FP8 KV cache: the attention's rope, KV append
# and top-k bridge, the full-indexer layers' query-front rope, the draft loop, the decode
# prologue and the host page retention policy.
_FP8_KV_ENGAGED = frozenset(
    {
        "rope_kv_append",
        "indexer_layer_front",
        "draft_loop",
        "decode_prologue",
        "host_page_retention",
    }
)
# The bases these paths extend. Under attention DP their kernels decline every call: the query
# fronts on every layer, the top-k selection on every step, and the post-attention hop is never
# built (the built-in layers run no post-attention all-reduce under attention DP).
_FP8_KV_BASES = frozenset(
    {
        "absorbed_query",
        "draft_absorbed_query",
        "indexer_topk",
        "cross_rank_topk",
        "post_attention_hop",
    }
)
# With an NVFP4 KV cache the attention paths are outside the shapes their kernels take; the
# draft loop, the decode prologue and the host page retention policy engage.
_NVFP4_KV_ENGAGED = frozenset({"draft_loop", "decode_prologue", "host_page_retention"})
_NVFP4_KV_BASES = frozenset({"post_attention_hop"})
_NO_LAYERS: frozenset = frozenset()

#: What this target serves with an FP8 KV cache. The query fronts decline attention DP on every
#: layer, which is outside their shapes here.
FP8_KV_PATHS = dispatch.TargetPaths(
    served=_FP8_KV_ENGAGED | _FP8_KV_BASES,
    layer_cells={
        **dispatch.LAYER_CELLS,
        "absorbed_query": {"layer": _NO_LAYERS},
        "draft_absorbed_query": {"layer": _NO_LAYERS},
        "indexer_layer_front": {"target_layer": _NO_LAYERS, "draft_layer": _NO_LAYERS},
    },
)
#: What this target serves with an NVFP4 KV cache.
NVFP4_KV_PATHS = dispatch.TargetPaths(served=_NVFP4_KV_ENGAGED | _NVFP4_KV_BASES)
#: What a deployment outside the cells serves under ``auto``: no fast path.
NO_PATHS = dispatch.TargetPaths(served=frozenset())


@dataclass(frozen=True)
class Cell:
    """One deployment this target serves: the facts ``route`` does not read, and the plan."""

    kv_cache: QuantAlgo
    moe_backend: str
    paths: dispatch.TargetPaths


#: The deployments this target serves, per (ranks, LM head tensor-parallel in attention DP,
#: draft tokens per request), the facts ``route`` claims them by.
CELLS: Mapping[Tuple[int, bool, int], Cell] = {
    (16, False, 3): Cell(QuantAlgo.FP8, "TRTLLM", FP8_KV_PATHS),
    (8, False, 3): Cell(QuantAlgo.FP8, "CUTEDSL", FP8_KV_PATHS),
    (8, True, 5): Cell(QuantAlgo.NVFP4, "CUTEDSL", NVFP4_KV_PATHS),
}


def target_paths(model_config: ModelConfig, mode: ModelingV2Mode) -> dispatch.TargetPaths:
    """What this target serves on ``model_config``'s deployment: its cell's paths, or none.

    Reads only the model config, so every rank decides alike.

    Raises:
        ValueError: under ``require``, if the deployment is not one of ``CELLS`` (its KV-cache
            dtype or MoE backend is not its cell's).
    """
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    key = (
        mapping.world_size,
        bool(mapping.enable_lm_head_tp_in_adp),
        getattr(spec_config, "max_draft_len", None),
    )
    kv_cache = model_config.quant_config.kv_cache_quant_algo
    cell = CELLS.get(key)
    if cell is not None and (kv_cache, model_config.moe_backend) == (
        cell.kv_cache,
        cell.moe_backend,
    ):
        return cell.paths
    served = "; ".join(
        f"{ranks} ranks, LM head tensor-parallel in attention DP {lm_head_tp}, {draft} draft "
        f"tokens: KV cache {cell.kv_cache.value}, MoE backend {cell.moe_backend}"
        for (ranks, lm_head_tp, draft), cell in CELLS.items()
    )
    words = (
        f"{key[0]} ranks, LM head tensor-parallel in attention DP {key[1]}, {key[2]} draft "
        f"tokens, KV cache {kv_cache}, MoE backend {model_config.moe_backend} is not a "
        f"deployment this target serves ({served})"
    )
    if mode is ModelingV2Mode.REQUIRE:
        raise ValueError(f"modeling_v2='require': {words}")
    logger.warning(f"[glm_moe_dsa] dispatch: {words}; no fast path runs")
    return NO_PATHS


def adopt_model(model: "ModelingV2Glm52Nvfp4Sm103Dep", plan: dispatch.Plan) -> None:
    """Re-class the attention modules of every layer and the speculative worker, as far as
    ``plan`` runs their fast paths.

    The attention modules of the decoder layers and of the draft layer, in index order, when
    the plan runs the rope and KV append (the attention paths that engage under attention DP
    extend it); then the speculative worker, which the built-in model builds after the draft
    layer, when the plan runs the draft loop. The layers, their MoE modules and the draft
    layer stay the built-in classes. The fast paths read the installed plan, which is ``plan``.

    Raises:
        TypeError: if an attention module or the speculative worker is not exactly the
            built-in class.
    """
    if dispatch.is_enabled("rope_kv_append"):
        for layer in model.model.layers:
            GlmMLA.adopt(layer.self_attn)
    if model.spec_worker is not None and dispatch.is_enabled("draft_loop"):
        GlmMtpWorker.adopt(model.spec_worker)


class GlmMoeDsaDepModelingV2(DeepseekV3Model):
    """The text model: the built-in ``DeepseekV3Model``, whose forward picks a target per step.

    Never constructed directly: ``adopt`` re-classes the built-in model the shell
    has already built, so the module tree, the parameters and their names stay
    the built-in model's. The class adds methods and plain attributes only.
    """

    _prefill: "PrefillTarget"
    _decode: "DecodeTarget"
    # The decode prologue the step loop runs, set when the target's runtime hooks are built;
    # the decode target takes the embedding rows of its launch.
    decode_prologue: Optional["DecodePrologueHook"] = None

    @classmethod
    def adopt(cls, model: DeepseekV3Model) -> "GlmMoeDsaDepModelingV2":
        """Re-class the built-in text model as this core and build its two targets.

        Raises:
            TypeError: if ``model`` is not exactly the built-in ``DeepseekV3Model``.
        """
        if type(model) is not DeepseekV3Model:
            raise TypeError(
                f"{cls.__name__} adopts the built-in DeepseekV3Model, got {type(model).__name__}"
            )
        model.__class__ = cls
        model._prefill = PrefillTarget(model)
        model._decode = DecodeTarget(model)
        return model

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment. A mixed batch -- context and generation
        rows in one step -- goes to prefill, so prefill is the general case and
        decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them.

    Runs the built-in decoder forward.
    """

    phase = dispatch.Phase.PREFILL

    def __init__(self, core: GlmMoeDsaDepModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        return DeepseekV3Model.forward(
            self.core,
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Runs this target's decoder loop.
    """

    phase = dispatch.Phase.DECODE

    def __init__(self, core: GlmMoeDsaDepModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The decoder layers of the text model on a decode step.

        The built-in loop with the decode prologue's embedding. Each layer is the built-in
        decoder layer, called as a module, so its forward hooks run; its attention runs the
        attention modules' decode composition under the step's phase. The draft layer is not
        run here; the last decoder layer applies the final norm.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        prologue = self.core.decode_prologue
        if inputs_embeds is None and prologue is not None:
            # The rows the decode prologue launch embedded (None unless that launch ran for
            # exactly these ids and this table).
            inputs_embeds = prologue.take_embedding(
                attn_metadata, input_ids, self.core.embed_tokens
            )
        if inputs_embeds is None:
            inputs_embeds = self.core.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for idx, decoder_layer in enumerate(self.core.layers[: self.core.num_hidden_layers]):
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        hidden_states = maybe_allgather_for_helix_cp(
            hidden_states, attn_metadata, self.core.mapping_with_cp
        )
        return hidden_states


class ModelingV2Glm52Nvfp4Sm103Dep(GlmMoeDsaForCausalLM):
    """GLM-5.2-NVFP4 on sm_103 at dep: the built-in causal LM with this target's core and dispatch."""

    def __init__(self, model_config: ModelConfig):
        world_size = model_config.mapping.world_size
        mode = resolved_mode(type(self))
        paths = target_paths(model_config, mode)
        knobs = dispatch.read_knobs()
        dispatch.check_rank_uniform(dispatch.rank_uniform_inputs(knobs, mode), world_size)
        plan = dispatch.resolve(mode, dict(knobs), paths)
        dispatch.install(plan)
        super().__init__(model_config)
        GlmMoeDsaDepModelingV2.adopt(self.model)
        self.plan = plan
        adopt_model(self, self.plan)
        dispatch.check_rank_uniform_adopted(adopted_state(self), world_size)

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The step-loop hooks this target declares, built on the engine's first read.

        Built after the model is loaded: a hook may hold pinned host memory, which model
        construction (under meta-device initialization) must not allocate.

        Raises:
            RuntimeError: if building the hooks raises ``AttributeError``, which
                ``runtime_hooks_of`` would otherwise read as "no hooks declared".
        """
        hooks = self.__dict__.get("_runtime_hooks")
        if hooks is None:
            try:
                hooks = runtime.make_runtime_hooks(self)
            except AttributeError as exc:
                raise RuntimeError(
                    f"building the step-loop hooks of {type(self).__name__} failed: {exc}"
                ) from exc
            self._runtime_hooks = hooks
        return hooks

    def forward(
        self,
        attn_metadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ):
        """The built-in forward under the phase of the target that takes this step.

        The scope covers the text model, the speculative worker and its draft loop.
        """
        with dispatch.step(self.model._select_target(attn_metadata).phase):
            return super().forward(
                attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                spec_metadata=spec_metadata,
                resource_manager=resource_manager,
                **kwargs,
            )

    def load_weights(self, weights):
        _weights.load(self, weights)
