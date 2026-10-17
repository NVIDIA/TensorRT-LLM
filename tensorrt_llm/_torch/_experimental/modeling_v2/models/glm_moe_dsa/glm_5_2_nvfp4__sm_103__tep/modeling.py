# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: glm-5.2-nvfp4 / sm_103 / tep.

GLM-5.2, modelopt NVFP4 export: 78 layers, hidden 6144, 64 query heads, MLA with
DSA sparse attention (index top-k 2048), layers 0-2 dense and 3-77 MoE (256
routed experts at top-8 plus a shared expert). Layer 78 is the checkpoint's bf16
MTP module, driven by the one-model MTP-Eagle loop at five draft tokens. `tep`
is the attention heads tensor-parallel over four or eight ranks and the experts
expert-parallel over the same ranks: each rank holds whole experts and runs the
built-in expert-parallel MoE on them.

The fast paths this target serves depend on its tensor-parallel size
(``TARGET_PATHS``): at eight ranks the attention runs at the same shapes as on
the tp8 target, so it serves every fast path of the family except the routed-MoE
ones, which take only tensor-parallel experts; at four ranks it serves the paths
whose kernels take four ranks (the indexer, the top-k selection, the router GEMM,
the dense MLP, the draft loop and the step-loop paths) and the bases they extend.
The layers run the family's composition (``shared.layers``), in which the block of
a path this target does not serve never runs.

The shell is the built-in ``GlmMoeDsaForCausalLM``: the same module tree, the
same weights and the same engine preferences (KV cache manager, KV transceiver,
sparse-attention defaults), so a built-in context server and this target's
generation server exchange the same KV and indexer-K layout. Construction runs,
in order:

1. the rank-uniform check of the knobs (``dispatch.check_rank_uniform``), first,
   so it runs before every collective the construction below makes;
2. the dispatch plan (``dispatch.resolve``) of the paths this target serves at its
   tensor-parallel size, before the built-in model is built, so a knob value no
   rank can parse fails early;
3. the plan is installed (``dispatch.install``): from here on the fast paths read
   it;
4. the built-in model, then the core: its text model adopted as
   ``GlmMoeDsaTepModelingV2``, whose forward hands each step to
   ``PrefillTarget`` or ``DecodeTarget``;
5. ``adopt_model`` (``shared.layers``), which re-classes the decoder layers,
   the draft layer, their attention modules (``shared.attention``) and their MoE
   modules as the family's layer classes and attaches the fast-path objects the
   plan enables to them, then re-classes the speculative worker
   (``shared.spec_worker``). The attention backends build their FMHA libraries
   after construction, in ``create_weights``, each with the libraries adoption
   declared on it first;
6. the rank-uniform check of that state (``dispatch.check_rank_uniform_adopted``).

The step-loop hooks the target declares to the engine (``runtime_hooks``,
``shared.runtime.make_runtime_hooks``) are built later, on the engine's first
read, after the model is loaded.

The rank-uniform check of step 1 also covers the shared switches the kernels
read and the modeling_v2 mode (``dispatch.rank_uniform_inputs``).

Every step runs under the phase of the target that takes it
(``dispatch.step``): the shell's forward enters it around the built-in forward,
so the model, the speculative worker and its draft loop all run in it. A
prefill step runs the built-in decoder forward, the built-in layers and the
built-in draft loop; a decode step runs ``DecodeTarget``'s loop, the layers'
decode composition and the worker's draft loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import resolved_mode
from tensorrt_llm._torch.attention.attention import maybe_allgather_for_helix_cp
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model, GlmMoeDsaForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative import SpecMetadata

from ..shared import dispatch, runtime
from ..shared.layers import adopt_model, adopted_state
from . import weights as _weights

if TYPE_CHECKING:
    from ..shared.kernels.decode_prologue import DecodePrologueHook


# The routed-MoE paths take tensor-parallel experts (every expert on every rank); with the experts
# expert-parallel their kernels decline every call, so this target serves none of them.
_ROUTED_MOE_PATHS = frozenset(
    {
        "routed_moe",
        "routed_moe_into_hop",
        "hop_routing",
        "shared_expert_prefetch",
        "draft_routed_moe",
        "cross_layer_expert_prefetch",
    }
)
# At four ranks the attention operands, the attention output and the post-MoE hop are outside
# the kernels' shapes (sixteen local heads, a four-rank output projection, a single-node group);
# these paths take four ranks.
_TEP4_PATHS = frozenset(
    {
        "indexer_layer_front",
        "rope_kv_append",
        "indexer_topk",
        "cross_rank_topk",
        "cross_rank_topk_cluster_merge",
        "cross_rank_topk_global_emit",
        "router_gemm",
        "dense_mlp",
        "draft_front",
        "draft_tail",
        "draft_loop",
        "decode_prologue",
        "deferred_input_prep",
        "sampler_store",
        "execution_stream_loop",
        "host_page_retention",
    }
)
# The bases these paths extend at four ranks. Their kernels decline every layer there: the query
# fronts on their layer cells, the post-attention hop on its native all-reduce, which a
# single-node group runs without multi-node NVLink.
_TEP4_BASES = frozenset({"absorbed_query", "draft_absorbed_query", "post_attention_hop"})
_NO_LAYERS: frozenset = frozenset()

#: What this target serves at each tensor-parallel size.
TARGET_PATHS = {
    4: dispatch.TargetPaths(
        served=_TEP4_PATHS | _TEP4_BASES,
        layer_cells={
            **dispatch.LAYER_CELLS,
            "absorbed_query": {"layer": _NO_LAYERS},
            "draft_absorbed_query": {"layer": _NO_LAYERS},
            "indexer_layer_front": {"target_layer": _NO_LAYERS, "draft_layer": _NO_LAYERS},
        },
        uncovered={"post_attention_hop": ("does not use MNNVL",)},
    ),
    8: dispatch.TargetPaths(served=frozenset(dispatch.FAST_PATH_NAMES) - _ROUTED_MOE_PATHS),
}


class GlmMoeDsaTepModelingV2(DeepseekV3Model):
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
    def adopt(cls, model: DeepseekV3Model) -> "GlmMoeDsaTepModelingV2":
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

    def __init__(self, core: GlmMoeDsaTepModelingV2) -> None:
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

    def __init__(self, core: GlmMoeDsaTepModelingV2) -> None:
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

        The built-in loop with the decode prologue's embedding.
        Each layer is called as a module, so its forward hooks run, and runs its decode
        composition under the step's phase. The draft layer is not run here; the last decoder
        layer applies the final norm.
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


class ModelingV2Glm52Nvfp4Sm103Tep(GlmMoeDsaForCausalLM):
    """GLM-5.2-NVFP4 on sm_103 at tep: the built-in causal LM with this target's core and dispatch."""

    def __init__(self, model_config: ModelConfig):
        world_size = model_config.mapping.world_size
        knobs = dispatch.read_knobs()
        mode = resolved_mode(type(self))
        dispatch.check_rank_uniform(dispatch.rank_uniform_inputs(knobs, mode), world_size)
        plan = dispatch.resolve(mode, dict(knobs), TARGET_PATHS[model_config.mapping.tp_size])
        dispatch.install(plan)
        super().__init__(model_config)
        GlmMoeDsaTepModelingV2.adopt(self.model)
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
