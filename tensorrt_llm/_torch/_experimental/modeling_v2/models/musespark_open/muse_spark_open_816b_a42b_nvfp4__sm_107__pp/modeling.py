# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: muse-spark-open-816b-a42b-nvfp4 / sm_107 / pp4, one pipeline stage of a context server.

Muse Spark Open, NVFP4 export: 62 decoder layers, hidden 8192, 128 query and 16 key-value
heads of width 64; every fourth layer counted from the last is a full-attention NoPE layer
and every other layer a 2048-token sliding-window RoPE layer; the MoE routes each token to 8
of 256 experts at half the residual width. ``pp4`` is four pipeline stages of one rank each
(tensor-parallel and expert-parallel size 1), on the four GPUs of one node: every stage runs
all heads and all experts of a contiguous slice of the layers.

What it serves: the context server of a disaggregated deployment (``MUSESPARK_OPEN_LEVER__ROLE
=ctx``), without speculative decoding or with a DFlash drafter on the paged TRTLLM attention
backend. A pipeline does not pay a generation step: the expert bytes per decoded token would
be four times a TEP4 rank's. So the composition (``TARGET_PATHS``) is the context-step paths
only, and an aggregated or generation server routed here has no composition: it runs the
built-in stage forward under ``auto`` and raises under ``require``. There is no decode,
sequence-parallel, hop, transport or publish path in this target, and no draft step.

With a DFlash drafter (``dispatch.DrafterContext``) the stages write the drafter's prompt
context into the draft layers of the shared paged KV cache, which the cache transceiver
sends to the generation server with the target layers (``kernels/dflash_context``):

* the built-in model is constructed inside the built-in model's
  ``pp_dflash_context_server()`` scope, which is what lets it admit the drafter on a
  pipeline stage;
* a context step's stage boundary also carries the captures of the drafter's target layers
  to the last stage (``carry.CaptureCarry``, run by ``PrefillTarget`` only), which owns the
  draft layers and assembles the step's capture buffer in layer order;
* the last stage's DFlash worker stores the drafter context on context steps and never runs
  a draft step, also not on the engine's own steps with generation rows (``worker``);
* once the weights are loaded, the last stage keeps of the drafter only what the context
  write reads (``slim``);
* when the KV cache manager keeps only a window of the drafter context (the step's
  ``SpecMetadata.draft_context_keep_from``), the shell's forward plans the step's kept rows
  on every stage before the stage forward or its graph replay (``CaptureCarry.plan_step``),
  and only those rows' captures cross the boundaries and are stored (``carry``, ``window``,
  ``worker``).

The shell is the built-in ``MuseSparkOpenOmniForConditionalGeneration`` with its text model
built as pipeline stages (``MuseSparkOpenModel.__pp_init__``: the other stages' layers
stripped, the embedding on the first stage, the final norm and the LM head on the last; the
stage boundary carries the FP32 residual and the next layer's BF16 pre-QKV norm output).
Construction decides the plan (``dispatch.compose``) and adopts the text model as
``MuseSparkOpenPpModelingV2``, whose forward hands a step with context rows to
``PrefillTarget`` and any other step to ``DecodeTarget``; both run the built-in stage forward
(``MuseSparkOpenModel._forward_pipeline_stage``), ``PrefillTarget`` with the planned paths:

* ``context_prep`` (``kernels/context_prep``): each stage layer's attention is re-classed as
  the TEP target's ``MuseSparkOpenTepAttention``, and every ``PrefillTarget`` layer call
  carries the ``preattn_prep`` op, which runs the context step's pre-attention prep in one
  launch per layer at 128 / 16 heads, with the bits of the built-in prep; padded rows of a
  replayed prefill graph get a zero attention output. ``context_prep_rows_vec`` selects the
  op's vectorized rows kernel.
* ``prefill_cutedsl_experts`` (``kernels/prefill_cutedsl_experts``, after the load): every
  NVFP4 MoE layer of the stage runs its 4096..8192-row expert calls over all 256 experts on
  the TEP target's Rubin CuTe DSL op.
* ``context_fp8_dense`` (``kernels/context_fp8_dense``, after the load; lossy, on only by its
  knob): the stage's QKV, o_proj, pool and unpool calls of ``MIN_ROWS`` rows or more run the
  Rubin MXFP8 GEMM, at TP1 tactics for the shapes the TEP table does not list.
* ``fmk_pure_context_fmha`` (``kernels/fmk_pure_context_fmha``, at construction; numeric,
  on only by its knob): the full-attention layers of the stage
  run the context FMHA of pure context steps on the TEP target's multi-instance paged
  context FMHA, built and planned at the stage's 128 query / 16 KV heads. Without it the
  stage's context FMHA is the built-in trtllm-gen one.
* ``context_router_input`` (``kernels/context_router_input``, at construction; bitwise): on a
  ``PrefillTarget`` forward, each layer's attention-side residual join also writes the
  router's FP32 input for context calls of ``stage_moe.MIN_ROWS`` rows or more, and the
  router GEMM reads it in place of the built-in BF16 -> FP32 copy of the norm output (the TEP
  target's ``context_sp_router_input`` join, which acts only inside its sequence-parallel
  forward).
* ``context_finalize_bf16`` (``kernels/context_finalize_bf16``, at construction; bitwise):
  on those calls the MoE finalize's sparse-slot kernel stores BF16 itself, in place of an
  FP32 store and a cast (the TEP target's sequence-parallel finalize).

``PrefillTarget`` with a drafter runs its stage forward through ``CaptureCarry`` (the
built-in per-layer calls with the enlarged boundary); ``DecodeTarget`` never does.

Both re-class the layers' MoE with ``kernels/stage_moe.StageContextMoE``, which runs the
built-in TP1 MoE call with those swaps while ``PrefillTarget``'s forward runs
(``StageStep.active``); every other call is the built-in MoE call. Every other op of a
context step -- the residual joins, the output norm-gate, the routing, the pre-expert norm
and quantization, the router GEMM -- is the built-in model's fused kernel, the same one a
TEP4 context step runs per row (the TEP4 target's decode-step kernels never run on a
context step).

The weights are the stage's slice of the checkpoint, loaded by the built-in loader
(``weights.py``).
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    resolved_facts,
    resolved_mode,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)
from tensorrt_llm.logger import logger

from . import dispatch as _dispatch
from . import weights as _weights
from ._tep import module as _tep_module

#: The fast paths a stage installs when the shell is constructed, and after the load.
CONSTRUCT_PATHS = frozenset(
    {
        "context_prep",
        "context_prep_rows_vec",
        "fmk_pure_context_fmha",
        "context_router_input",
        "context_finalize_bf16",
    }
)
LOAD_PATHS = frozenset({"context_fp8_dense", "prefill_cutedsl_experts"})


class MuseSparkOpenPpModelingV2(MuseSparkOpenModel):
    """The text model of one stage: the built-in ``MuseSparkOpenModel`` built as a pipeline
    stage, whose forward picks a target per step.

    Never constructed directly: ``adopt`` re-classes the built-in text model the shell has
    already built, so the module tree, the parameters and their names stay the built-in
    model's. The class adds methods and plain attributes only.
    """

    _prefill: "PrefillTarget"
    _decode: "DecodeTarget"
    _plan: "_dispatch.Plan"
    #: Set while ``PrefillTarget`` runs a forward; None without a stage MoE swap.
    _stage_step: Optional[Any] = None
    #: The context-step stage forward that carries the DFlash captures
    #: (``kernels.dflash_context.carry.CaptureCarry``); None without a drafter.
    _capture_carry: Optional[Any] = None

    @classmethod
    def adopt(
        cls,
        model: MuseSparkOpenModel,
        plan: "_dispatch.Plan" = _dispatch.NO_PATHS,
        capture_carry: Optional[Any] = None,
        compute_capability: Optional[Tuple[int, int]] = None,
    ) -> "MuseSparkOpenPpModelingV2":
        """Re-class the built-in stage text model as this core and build its two targets.

        ``capture_carry`` is the stage's ``CaptureCarry`` with a DFlash drafter, else None.
        ``compute_capability`` is the device's, which ``fmk_pure_context_fmha`` builds for.

        Raises:
            TypeError: if ``model`` is not exactly the built-in ``MuseSparkOpenModel``.
            ValueError: if ``model`` is not a pipeline stage, or a fast path of the plan
                cannot run on it.
        """
        if type(model) is not MuseSparkOpenModel:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenModel, got {type(model).__name__}"
            )
        if getattr(model, "_pp_stage", None) is None:
            raise ValueError(
                f"{_dispatch.TAG} the text model was not built as a pipeline stage "
                "(MuseSparkOpenModel.__pp_init__ set no _pp_stage)"
            )
        unknown = sorted(plan.paths - CONSTRUCT_PATHS - LOAD_PATHS)
        if unknown:
            raise ValueError(f"{_dispatch.TAG} no installer for the planned fast paths {unknown}")
        mapping = model.model_config.mapping
        if plan.enabled("context_prep"):
            from .kernels.context_prep import install as _context_prep

            _context_prep.install(model, mapping, plan)
        if plan.enabled("fmk_pure_context_fmha"):
            # Agreed over the stages; a refusal leaves every stage's pure context steps on
            # the built-in context FMHA (and raises for the lever set on by its knob).
            from .kernels.fmk_pure_context_fmha import install as _fmk_pure_context

            try:
                _fmk_pure_context.install(model, mapping, compute_capability)
            except ValueError as refusal:
                plan.installed("fmk_pure_context_fmha", False, str(refusal))
            else:
                plan.installed("fmk_pure_context_fmha", True)
        step = None
        if plan.enabled("context_router_input") or plan.enabled("context_finalize_bf16"):
            from .kernels import stage_moe as _stage_moe

            step = _stage_moe.StageStep()
        if plan.enabled("context_router_input"):
            from .kernels.context_router_input import install as _router_input

            _router_input.install(model, mapping, step)
            plan.installed("context_router_input", True)
        if plan.enabled("context_finalize_bf16"):
            from .kernels.context_finalize_bf16 import install as _finalize_bf16

            _finalize_bf16.install(model, mapping, step)
            plan.installed("context_finalize_bf16", True)
        model.__class__ = cls
        model._stage_step = step
        model._capture_carry = capture_carry
        model._plan = plan
        model._prefill = PrefillTarget(model)
        model._decode = DecodeTarget(model)
        return model

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: one host-side integer read and nothing else."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def _forward_pipeline_stage(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor],
        position_ids: Optional[torch.IntTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        **kwargs,
    ) -> torch.Tensor:
        """The stage forward: ``CaptureCarry``'s while ``PrefillTarget`` activated it, else the
        built-in one (``DecodeTarget``, and every step without a drafter)."""
        carry = self._capture_carry
        if carry is not None and carry.active:
            return carry.forward_stage(
                self, attn_metadata, input_ids, position_ids, inputs_embeds, **kwargs
            )
        return MuseSparkOpenModel._forward_pipeline_stage(
            self, attn_metadata, input_ids, position_ids, inputs_embeds, **kwargs
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class PrefillTarget:
    """A step with context rows: the built-in stage forward, with ``context_prep`` when planned.

    The stage receives the boundary tensors, runs its layers and sends them on, or (the last
    stage) applies the final norm (``MuseSparkOpenModel._forward_pipeline_stage``). With
    ``context_prep`` every layer call carries the ``preattn_prep`` op, which the adopted
    attention runs in place of its built-in pre-attention prep. With a stage MoE swap
    (``context_router_input``, ``context_finalize_bf16``) the forward runs with
    ``StageStep.active`` set, which the re-classed joins and MoE read. With a DFlash drafter
    the forward runs with ``CaptureCarry.active`` set, so the stage boundary also carries the
    drafter's captures to the last stage.
    """

    #: The op every layer call carries as ``context_prep``; None without the path.
    context_prep = None

    def __init__(self, core: MuseSparkOpenPpModelingV2) -> None:
        self.core = core
        if core._plan.enabled("context_prep"):
            self.context_prep = _tep_module("ops.attention.preattn_prep").preattn_prep

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.context_prep is not None:
            kwargs = dict(kwargs, context_prep=self.context_prep)
        step = self.core._stage_step
        carry = self.core._capture_carry
        with contextlib.ExitStack() as scope:
            if step is not None:
                scope.enter_context(_active(step))
            if carry is not None:
                scope.enter_context(carry.activated())
            return MuseSparkOpenModel.forward(
                self.core,
                attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )


@contextlib.contextmanager
def _active(step: Any):
    """``step.active`` for the duration."""
    step.active = True
    try:
        yield
    finally:
        step.active = False


class DecodeTarget:
    """A step without context rows: the built-in stage forward, with no fast path.

    A context server serves no generation step; the steps that reach this target are the
    engine's own (warm-up, the capture of a decode graph). This target has no decode path,
    so it runs exactly the built-in stage forward.
    """

    def __init__(self, core: MuseSparkOpenPpModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return MuseSparkOpenModel.forward(
            self.core,
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


#: The context-step paths a stage serves at sm107: the prep op, the MXFP8 dense copies, the
#: CuTe DSL prefill experts, the multi-instance context FMHA at the stage's head counts, and
#: the two stage MoE swaps (the router's FP32 input from the join, the BF16 finalize).
_SM107_PP4_CTX_PATHS = frozenset(
    {
        "context_prep",
        "context_prep_rows_vec",
        "context_fp8_dense",
        "prefill_cutedsl_experts",
        "fmk_pure_context_fmha",
        "context_router_input",
        "context_finalize_bf16",
    }
)

#: The compositions, per (SM, tensor-parallel size) (``dispatch.CompositionKey``), each with
#: the pipeline topology it is written for, on a context server without speculative decoding
#: (``dispatch.composition``); every other deployment runs the built-in stage forward only.
TARGET_PATHS = {
    (107, 1): _dispatch.TargetPaths(served=_SM107_PP4_CTX_PATHS, topology=_dispatch.ONE_NODE_PP4),
}


def device_compute_capability() -> Optional[Tuple[int, int]]:
    """This rank's compute capability, the fact ``route`` routes on; None without a CUDA device."""
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


class ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Pp(MuseSparkOpenOmniForConditionalGeneration):
    """Muse-Spark-Open-816B-A42B-NVFP4 on sm_107 at pp4: one stage of a context server."""

    #: The fast paths this stage is built with (``dispatch.compose``).
    _plan: "_dispatch.Plan" = _dispatch.NO_PATHS
    #: Whether ``load_weights`` installed the plan's load paths.
    _loaded_installed: bool = False
    #: The DFlash drafter context the stages write (``dispatch.DrafterContext``); None
    #: without a drafter.
    _drafter: Optional["_dispatch.DrafterContext"] = None

    def __init__(self, model_config: ModelConfig, *args, **kwargs):
        """Build the built-in model as pipeline stages, decide the plan, and adopt the text model.

        The composition decision is agreed over the stages first (one collective every
        stage enters), so a stage whose environment, device or drafter differs cannot leave
        the others waiting in a later collective. With an agreed DFlash drafter the built-in
        model is constructed in its ``pp_dflash_context_server()`` scope, and the stage's
        drafter context is installed (``kernels.dflash_context.install``, collective) before
        the text model is adopted.

        Raises:
            ValueError: under ``require`` without a composition; under any mode with a lever
                error or a missing precondition (``dispatch.compose``), or a drafter context
                some stage cannot install.
        """
        mode = resolved_mode(type(self))
        capability = device_compute_capability()
        snapshot = _dispatch.lever_environment()
        decision = _dispatch.agreed_composition(model_config, TARGET_PATHS, capability, snapshot)
        _dispatch.require_served(
            model_config, mode, TARGET_PATHS, capability, snapshot, decision=decision
        )
        with _builtin_scope(decision.drafter):
            super().__init__(model_config, *args, **kwargs)
        plan = _dispatch.compose(
            self.llm.model,
            model_config,
            mode,
            TARGET_PATHS,
            capability,
            facts=resolved_facts(type(self)),
            snapshot=snapshot,
            decision=decision,
        )
        self._plan = plan
        self._drafter = decision.drafter
        # The built-in constructor returns early when it is re-entered; adopt once.
        if not isinstance(self.llm.model, MuseSparkOpenPpModelingV2):
            carry = None
            if decision.drafter is not None:
                from .kernels.dflash_context import install as _dflash_context

                carry = _dflash_context.install(self, model_config, decision.drafter)
                if plan.rank0 or carry.last:
                    logger.info(
                        f"{_dispatch.TAG} "
                        + _dflash_context.describe(
                            carry.route,
                            carry.stage.stage,
                            getattr(model_config, "max_num_tokens", None),
                            carry,
                        )
                    )
            MuseSparkOpenPpModelingV2.adopt(self.llm.model, plan, carry, capability)

    def forward(self, attn_metadata, *args, **kwargs):
        """The built-in forward; with a drafter, the stage's ``CaptureCarry`` first plans the
        step's drafter context window.

        Eager on every step: a replayed prefill graph patches only the text model's forward,
        so this runs around each replay, and the graph's eager breaks read the plan.
        """
        carry = getattr(self.llm.model, "_capture_carry", None)
        if carry is not None:
            spec_metadata = kwargs.get("spec_metadata", args[4] if len(args) > 4 else None)
            carry.plan_step(attn_metadata, spec_metadata)
        return super().forward(attn_metadata, *args, **kwargs)

    def load_weights(self, weights, weight_mapper=None):
        """The built-in load of the stage's slice, then the plan's load paths.

        Expert tensors are final once the built-in loader returns, so this is the first point
        ``prefill_cutedsl_experts`` (which compiles over them) and ``context_fp8_dense`` (which
        copies the projections) can run; both are collective over the stages, and every
        stage reaches them before any CUDA graph is captured. Under ``NO_PATHS`` this is the
        built-in load and nothing else.
        """
        result = _weights.load(self, weights, weight_mapper)
        plan = self._plan
        if plan and not self._loaded_installed:
            _weights.install_loaded_features(
                self.llm.model, self.model_config.mapping, plan, device_compute_capability()
            )
            self._loaded_installed = True
        return result

    def post_load_weights(self) -> None:
        """Report the installed lever table once, after every load path; with a DFlash drafter,
        reduce the last stage's drafter to the context write's tensors first.

        The loader walks this module before its submodules, and the drafter's weights are
        final once every load path ran (the drafter's projections are unquantized), so the
        reduced drafter is built from the loaded weights, and the walk then skips the drafter
        modules whose weights were removed. ``LoadFormat.DUMMY`` never calls
        ``load_weights``: a planned load path is then not installed, which is reported (and
        raises for a lever its knob set ``on``) instead of silently missing.
        """
        self._slim_drafter()
        plan = self._plan
        if not plan or self.__dict__.get("_table_logged"):
            return
        self._table_logged = True
        if not self._loaded_installed:
            for name in sorted(plan.paths & LOAD_PATHS):
                plan.installed(name, False, "the loader never called load_weights (a DUMMY load)")
        logger.info(
            f"{_dispatch.TAG} stage {plan.stage} lever table json (installed): {plan.report()}"
        )

    def cache_derived_state(self) -> None:
        """``post_load_weights`` for the loader paths whose walk calls only this stage."""
        self.post_load_weights()

    def _slim_drafter(self) -> None:
        """Once, on the last stage of a context server with a DFlash drafter:
        ``dflash_context.install.slim_last_stage``."""
        if self._drafter is None or self.__dict__.get("_drafter_slimmed"):
            return
        self._drafter_slimmed = True
        if not self.model_config.mapping.is_last_pp_rank():
            return
        from .kernels.dflash_context import install as _dflash_context

        released = _dflash_context.slim_last_stage(self)
        if released is not None:
            logger.info(
                f"{_dispatch.TAG} last stage: the drafter keeps fc, hidden_norm, the fused "
                f"context K/V and the RoPE table; {released / (1 << 30):.2f} GiB of draft-step "
                "weights released"
            )


@contextlib.contextmanager
def _builtin_scope(drafter: Optional["_dispatch.DrafterContext"]):
    """The scope the built-in model is constructed in: the built-in model's
    ``pp_dflash_context_server()`` with a drafter, nothing without one.

    The built-in model admits the drafter on a pipeline stage only with its own opt-in on
    (``modeling_musespark_open.pipeline_parallel_dflash_context``); without it, it refuses
    speculative decoding under pipeline parallelism. A ``ValueError`` from the construction
    is logged with that hint and re-raised as it is.
    """
    if drafter is None:
        yield
        return
    from tensorrt_llm._torch.models.modeling_musespark_open import pp_dflash_context_server

    with pp_dflash_context_server():
        try:
            yield
        except ValueError:
            logger.error(
                f"{_dispatch.TAG} the built-in decoder refused this DFlash context server "
                "stage; it admits the drafter on a pipeline stage only when its opt-in for "
                "pipelined DFlash context servers is on on every context rank (see "
                "modeling_musespark_open.pipeline_parallel_dflash_context). Removing the "
                "speculative_config instead serves the context server without the drafter "
                "context."
            )
            raise
