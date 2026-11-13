# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 generation worker on sm_107 with attention DP over every rank and the experts split over all of them
(DEP4 / DEP8 / DEP16), on the MegaMoE CuteDSL fused-communication MoE."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    disable_v2_by_default,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dflash_merge as _dflash_merge,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_glue_install as _dsg,
)
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3VLForConditionalGeneration,
    get_moe_layer_ids,
    get_text_config,
)
from tensorrt_llm._torch.modules import logits_processor as _logits_processor
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as _sampler_flashinfer
from tensorrt_llm._torch.speculative import dspark as _spec_dspark
from tensorrt_llm._torch.speculative import interface as _spec_interface
from tensorrt_llm.logger import logger

from .. import runtime_hooks_compat
from ..msa import engine as _msa_engine
from ..ops import add_rmsnorm as _native_norm
from ..ops import native_swiglu as _swiglu
from ..ops import step_hooks as _step_hooks
from ..ops.moe.megamoe_local import local_impl as _megamoe_local
from ..runtime_hooks_compat import MsaRuntimePolicy, RuntimeHooks
from . import corun_band as _cb
from . import drafter_mxfp8 as _mxd
from . import gumbel_draft as _gumbel
from . import idx_pdl as _ipdl
from . import idxk_producer as _ik
from . import lc2_decode as _lc2
from . import logits_widen as _widen
from . import markov_glue as _glue
from . import merge_mxfp8 as _mx
from . import norm_mxfp8 as _nm
from . import pad_fill as _pad_fill
from . import pad_mask as _pad
from . import proj_pin as _pp
from . import rej_mc_fi as _rej
from . import router_operand as _ro
from . import se_side as _ss
from . import sfp_mc_fi as _sfp
from . import shared_mxfp8 as _sm
from . import softmax_mc as _softmax
from . import tail as _tail
from . import topp_det as _topp
from .engagement import (
    ARM_IDENTITIES,
    arm_identity,
    deployment_mismatch,
    identity_lines,
    moe_engagement,
)
from .m31_switches import m31_disabled, m31_enabled

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig

# The GPU architecture this target is written for (SM107).
_SM = (10, 7)

# The kernel the SM107 MegaMoE op builds per launch bucket (cute_dsl_megamoe_custom_op._construct_megamoe_kernel); a
# rejected gen-phase specialization builds the generic kernel and is logged at debug level only.
_KERNEL = (
    f"RubinInferenceGenphaseMegaMoE up to {_megamoe_op._SM107_GENPHASE_MAX_TOKENS_PER_RANK} tokens per rank "
    f"(generic-kernel buckets {sorted(_megamoe_op._SM107_GENERIC_KERNEL_BUCKETS)}), RubinInferenceMegaMoE otherwise"
)


def dep_runtime_hooks() -> Optional[RuntimeHooks]:
    """The target's step-loop hooks: the sm_107 TP targets' hooks wherever this target's path takes them, each under
    the TP targets' kill switch, in four groups, plus the late-compile detector (it only logs), the cross-rank
    tactic merge, the tactic persistence, the int32 prompt tokens and the disaggregated draft window. They are built
    for this engine (``runtime_hooks_compat``): a requested hook the engine does not have is dropped and logged.

    * Tactic merge (``TRTLLM_M31_DISABLE_DEP_TACTIC_PIN=1`` keeps the per-rank choice): every attention-DP rank tunes
      its GEMMs on its own, and rank-local profiling noise otherwise leaves the ranks serving different tactics for
      one shape (CUTLASS on some ranks, FlashInfer on others); the slowest rank then sets every MoE collective. The
      merge keeps the fastest tuned tactic per shape on every rank. The sm_107 MXFP8 GEMM backends write the same
      bits, so the merge changes timing only.
    * Tactic persistence (``TRTLLM_M31_DISABLE_DEP_TACTIC_BUNDLE=1`` keeps it off): a tuning run
      (``TLLM_AUTOTUNER_CACHE_PATH`` set) also saves the FlashInfer MXFP8 tactics of the warmup and decode-graph passes
      into that cache, so a restart reuses them for every pass.
    * Step loop (``DEP_STEP_HOOKS``), in two halves:

      - Ordering and timing: the overlap loop runs on the execution stream, the sampler copies its store to the host
        on a side stream, the iteration log takes one timing event per boundary, and a finished KV receive commits
        its prompt blocks for reuse after the step's forward is launched instead of before it (DEP_DEFER_REUSE_COMMIT;
        the request itself never reads the reuse tree). Only the streams, the timing events and when the commit's host
        work runs change.
      - Input path (``DEP_STEP_INPUTS``): a decode step's small input copies go out as one pack, scattered in parallel
        by the decode graph's prologue from a window uploaded on a side stream; uploads whose values the device already
        holds are skipped and a changed packed table stages only its changed span; the MSA KV lengths are derived once
        per speculative step; the spec sampler's Philox seeds / offsets and slot ids ride the pack, and the
        accepted-draft counts are not uploaded (nothing on this target's path reads
        ``spec_metadata.num_accepted_draft_tokens``); the drafter's batch map and block table are copied only on change
        (a few changed page indices patched from the host). The same values are meant to reach the device, but these
        items change which copies carry them.

      Item switches as on the TP targets: HOSTTAIL (its six items), SKIP_ACCEPTED_UPLOAD, BATCH_CHANGED_ONLY,
      HOST_D2H_SIDE, SMALL_H2D, PHILOX_PACK, KV_LENS_ONCE, PACK_PSCATTER, PACK_IN_GRAPH, PACK_EARLY_COPY,
      H2D_CHANGED_SPAN, BLOCKOFF_GUARD, BLOCKOFF_PATCH. SLOT_IDS_CHANGED (the spec sampler's slot ids re-uploaded only
      on change) stays opt-in (``TRTLLM_M31_ENABLE_SLOT_IDS_CHANGED=1``, with the input path), as on the TP8 target.
    * Native step launches (``DEP_STEP_NATIVE``, inside the step loop; ``TRTLLM_M31_DISABLE_DEP_STEP_NATIVE=1`` restores
      the engine's own lines, as the group switch does): the sm_107 TP targets' four step-glue launches, from the
      family module ``ops.step_hooks`` (CUDA C++ ops, no Triton), each writing what the line it replaces writes:
      the decode graph's step-input prologue in one launch (``DEP_STEP_PROLOGUE``; with PACK_EARLY_COPY, and the step
      epoch), the speculative step correction in one launch (``DEP_SPEC_OFFSETS``), the MSA KV-length correction in one
      launch instead of ~23 (``DEP_MSA_KV_UPDATE``) and the engine's previous-batch input gathers in one launch
      (``STEP_GATHER_FUSED``). With the input path the step-input pack never launches a Triton kernel
      (``pack_cpp_scatter``, no switch): without the one-launch prologue the pack is issued before the replay by the
      native serial scatter.
    * MSA metadata (``DEP_MSA_POLICY``, inside the step loop's input path): block-table copies with unchanged inputs
      skipped, pool views built once, block-id tables copied from the page-index views, and the sub-page expansion
      done inside the forward. The tables a forward reads are the same.
    * Kernels (``DEP_KERNEL_HOOKS``): FlashInfer's CuTe DSL MXFP8 GEMM above 32 rows at its pinned tactics (bitwise
      the default tactic's output) and the sampler tail without its no-op bookkeeping launches (TAIL_BOOKKEEP, bitwise).
    * Fixes (``DEP_FIX_HOOKS``): the MSA cache's E4M3 writes saturate as the CUDA FP8 writers do (bitwise for every
      finite in-range value), the MSA KV pools are zeroed at allocation and priced at their real footprint, and the
      standalone drafter's pool takes its sizing / resize fixes and its per-step reserve with a guard page.
    * Prompt tokens: every request's prompt reaches
      LlmRequest as the executor request's int32 array instead of a list. The shared context-only scope is extended to
      every request type, since this generation worker's requests are generation-only. The tokens are the same.
    * Draft window (``DEP_DISAGG_DRAFT_KV``): the drafter's prompt window travels with the target's KV, so this
      worker's drafter starts from the prompt instead of an empty window. That changes the drafts and so the
      acceptance length, not the target's distribution. A request whose context worker ships no window starts from an
      empty one.

    Opt-in: SLOT_IDS_CHANGED. Never armed: the short-sequence identity selection and the greedy-only Philox skip.
    ``TRTLLM_M31_DISABLE_<GROUP>=1`` turns a group off; ``TRTLLM_M31_DISABLE_DEP_HOOKS=1`` keeps the built-in model's
    hooks (None)."""
    if m31_disabled("DEP_HOOKS"):
        return None
    step = not m31_disabled("DEP_STEP_HOOKS")
    inputs = step and not m31_disabled("DEP_STEP_INPUTS")
    host_tail = step and not m31_disabled("HOSTTAIL")
    spec_inputs = inputs and host_tail
    parallel_scatter = inputs and not m31_disabled("PACK_PSCATTER")
    in_graph = parallel_scatter and not m31_disabled("PACK_IN_GRAPH")
    draft_offsets = inputs and not m31_disabled("BLOCKOFF_GUARD")
    msa = inputs and not m31_disabled("DEP_MSA_POLICY")
    native = step and not m31_disabled("DEP_STEP_NATIVE") and _step_hooks.ops_available()
    early_copy = in_graph and not m31_disabled("PACK_EARLY_COPY")
    prologue = native and early_copy and not m31_disabled("DEP_STEP_PROLOGUE")
    offsets = native and not m31_disabled("DEP_SPEC_OFFSETS")
    kv_update = native and not m31_disabled("DEP_MSA_KV_UPDATE")
    fold = native and not m31_disabled("STEP_GATHER_FUSED")
    state = {True: "on", False: "off"}
    logger.info(
        f"[M3.1 DEP] native step launches: prologue {state[prologue]}, spec offsets {state[offsets]}, "
        f"MSA kv update {state[kv_update]}, STEP_GATHER_FUSED {state[fold]}; step-input pack without Triton "
        f"{state[inputs] if inputs else 'off (no pack)'}"
    )
    kernels = not m31_disabled("DEP_KERNEL_HOOKS")
    fixes = not m31_disabled("DEP_FIX_HOOKS")
    persist_tactics = not m31_disabled("DEP_TACTIC_BUNDLE")
    hooks = dict(
        sparse_attention=_msa_engine.sparse_attention_classes(),
        late_compile_detect=True,
        merge_tuned_tactics_across_ranks=not m31_disabled("DEP_TACTIC_PIN"),
        pack_step_inputs=inputs,
        pack_parallel_scatter=parallel_scatter,
        pack_in_graph=in_graph,
        pack_early_copy=early_copy,
        pack_step_epoch=prologue,
        pack_prologue_op=_step_hooks.step_prologue if prologue else None,
        pack_cpp_scatter=inputs,
        skip_unchanged_h2d=inputs,
        h2d_changed_span=inputs and not m31_disabled("H2D_CHANGED_SPAN"),
        kv_lens_update_once=spec_inputs and not m31_disabled("KV_LENS_ONCE"),
        spec_offsets_op=_step_hooks.spec_offsets if offsets else None,
        spec_step_fold=_step_hooks.M3DSparkStepFold() if fold else None,
        loop_on_execution_stream=step,
        pack_spec_rng_uploads=spec_inputs and not m31_disabled("PHILOX_PACK"),
        spec_small_h2d=spec_inputs and not m31_disabled("SMALL_H2D"),
        skip_accepted_draft_upload=spec_inputs and not m31_disabled("SKIP_ACCEPTED_UPLOAD"),
        spec_batch_map_changed_only=spec_inputs and not m31_disabled("BATCH_CHANGED_ONLY"),
        spec_slot_ids_changed_only=inputs and m31_enabled("SLOT_IDS_CHANGED"),
        sampler_d2h_side_stream=host_tail and not m31_disabled("HOST_D2H_SIDE"),
        skip_unchanged_draft_block_offsets=draft_offsets,
        patch_unchanged_draft_block_offsets=draft_offsets and not m31_disabled("BLOCKOFF_PATCH"),
        iteration_timing_one_event=step,
        mxfp8_pinned_tactics=kernels,
        spec_tail_bookkeep=kernels and not m31_disabled("TAIL_BOOKKEEP"),
        fp8_saturating_casts=fixes,
        msa_kv_cache_fixes=fixes,
        draft_pool_fixes=fixes,
        draft_kv_reserve_guard=fixes,
        disagg_draft_kv=not m31_disabled("DEP_DISAGG_DRAFT_KV"),
        defer_gen_reuse_commit=step and not m31_disabled("DEP_DEFER_REUSE_COMMIT"),
        context_token_array=True,
        token_array_all_requests=True,
        persist_flashinfer_mxfp8_tactics=persist_tactics,
    )
    msa_policy = dict(
        skip_unchanged_block_offsets=msa,
        reuse_pool_views=msa,
        block_ids_from_views=msa,
        forward_staging=msa,
        kv_update=_step_hooks.msa_kv_update if kv_update else None,
    )
    return runtime_hooks_compat.build(hooks, msa=msa_policy, target=__name__)


def armed_hooks(hooks: Optional[RuntimeHooks]) -> List[str]:
    """The names of the hooks ``hooks`` sets away from their defaults (``msa.<field>`` for the MSA policy)."""
    if hooks is None:
        return []
    names = [
        f.name for f in fields(hooks) if f.name != "msa" and getattr(hooks, f.name) != f.default
    ]
    default_msa = MsaRuntimePolicy()
    names += [
        f"msa.{f.name}"
        for f in fields(hooks.msa)
        if getattr(hooks.msa, f.name) != getattr(default_msa, f.name)
    ]
    return names


# The engine's override registries each item installs through, by item; a registry the runtime does not have is None,
# and its item installs nothing (the stock op runs).
_OVERRIDE_SETTERS = {
    "SOFTMAX_MC": (getattr(_sampler_flashinfer, "set_softmax_override", None),),
    "TOPP_DET": (getattr(_sampler_flashinfer, "set_top_p_renorm_override", None),),
    "DEP_PAD_FILL": (getattr(_spec_interface, "set_padding_rows_fill_override", None),),
    "DEP_REJ_MC_FI": (
        getattr(_spec_interface, "set_chain_speculative_sampling_override", None),
        getattr(_spec_interface, "set_chain_speculative_sampling_slot_override", None),
    ),
    "DEP_SFP_MC_FI": (getattr(_sampler_flashinfer, "set_sampling_from_probs_override", None),),
    "DEP_MARKOV_GLUE": (getattr(_spec_dspark, "set_ancestral_glue_override", None),),
    "DEP_GUMBEL_DRAFT": (getattr(_spec_dspark, "set_ancestral_block_override", None),),
    "DEP_LOGITS_WIDEN": (getattr(_logits_processor, "set_logits_widen_override", None),),
}


def _setters(item: str):
    """The registry setters of ``item``, or None when the runtime lacks any of them."""
    setters = _OVERRIDE_SETTERS[item]
    return None if any(setter is None for setter in setters) else setters


def _install_sampler_overrides() -> None:
    """Install ``softmax_mc.softmax_mc`` and ``topp_det.top_p_renorm_det`` as the sampler's softmax and top-p renorm
    overrides, each when its switch is on and its op is built, ``pad_fill.pad_rows_fill`` as its padding-row fill when
    DEP_PAD_FILL is on, ``rej_mc_fi``'s two forms as the rejection sampler's chain overrides when DEP_REJ_MC_FI is
    on and its ops are built, ``sfp_mc_fi.sampling_from_probs_mc_fi`` as the sampler's sampling-from-probs
    override when DEP_SFP_MC_FI is on and its op is built, ``markov_glue.MarkovGlue`` as the DSpark ancestral
    loop's glue when DEP_MARKOV_GLUE is on, ``gumbel_draft.draw_block`` as the DSpark ancestral block's sampler when
    DEP_GUMBEL_DRAFT is on and its Triton module imports (the loop, with the overrides above, is then its fallback), and
    ``logits_widen.widen`` as the logits processor's widening when DEP_LOGITS_WIDEN is on, and log the eight
    engagements once per rank. An item whose override registry this runtime does not have installs nothing and logs
    that instead."""
    setters = {item: _setters(item) for item in _OVERRIDE_SETTERS}
    if setters["SOFTMAX_MC"] and _softmax.SOFTMAX_MC and _softmax.mc_op() is not None:
        setters["SOFTMAX_MC"][0](_softmax.softmax_mc)
    if setters["TOPP_DET"] and _topp.TOPP_DET and _topp.topp_det_op() is not None:
        setters["TOPP_DET"][0](_topp.top_p_renorm_det)
    if setters["DEP_PAD_FILL"] and _pad_fill.DEP_PAD_FILL:
        setters["DEP_PAD_FILL"][0](_pad_fill.pad_rows_fill)
    if (
        setters["DEP_REJ_MC_FI"]
        and _rej.DEP_REJ_MC_FI
        and _rej.mc_fi_op() is not None
        and _rej.mc_fi_slots_op() is not None
    ):
        setters["DEP_REJ_MC_FI"][0](_rej.chain_spec_mc_fi)
        setters["DEP_REJ_MC_FI"][1](_rej.chain_spec_mc_fi_slots)
    if setters["DEP_SFP_MC_FI"] and _sfp.DEP_SFP_MC_FI and _sfp.sfp_mc_fi_op() is not None:
        setters["DEP_SFP_MC_FI"][0](_sfp.sampling_from_probs_mc_fi)
    if setters["DEP_MARKOV_GLUE"] and _glue.DEP_MARKOV_GLUE:
        setters["DEP_MARKOV_GLUE"][0](_glue.MarkovGlue())
    if (
        setters["DEP_GUMBEL_DRAFT"]
        and _gumbel.DEP_GUMBEL_DRAFT
        and _gumbel.block_sampler() is not None
    ):
        setters["DEP_GUMBEL_DRAFT"][0](_gumbel.draw_block)
    if setters["DEP_LOGITS_WIDEN"] and _widen.DEP_LOGITS_WIDEN:
        setters["DEP_LOGITS_WIDEN"][0](_widen.widen)
    for item, module in (
        ("SOFTMAX_MC", _softmax),
        ("TOPP_DET", _topp),
        ("DEP_PAD_FILL", _pad_fill),
        ("DEP_REJ_MC_FI", _rej),
        ("DEP_SFP_MC_FI", _sfp),
        ("DEP_MARKOV_GLUE", _glue),
        ("DEP_GUMBEL_DRAFT", _gumbel),
        ("DEP_LOGITS_WIDEN", _widen),
    ):
        if setters[item] is None:
            logger.info(
                f"[M3.1 DEP] {item}: not taken: its override registry is not available on this runtime; "
                "the stock op runs"
            )
        else:
            module.log_engagement()


@disable_v2_by_default
class ModelingV2MiniMaxM31FlashNvfp4Sm107Dep(MiniMaxM3VLForConditionalGeneration):
    """MiniMax-M3.1 DEP generation worker, sm_107: the built-in model, checked and identified, with its own kernels.

    The layers, weights, forward, samplers and speculative path are the built-in class's, with these replacements: the
    sparse layers' generation decode attention runs LC2 (``lc2_decode``: within the stock Triton decode's
    float64-reference band, not bitwise), on the calls LC2 takes the block-owner decode OWN-T (DEP_OWNT, likewise not
    bitwise), and either hands its o_proj
    the output's MXFP8 form (``merge_mxfp8``, bitwise the o_proj's own quantize), the spec sampler's softmax and
    top-p renorm run SOFTMAX_MC (``softmax_mc``) and TOPP_DET (``topp_det``), both bitwise FlashInfer's, its
    CUDA-graph padding rows are filled in place with only
    those rows written (DEP_PAD_FILL, ``pad_fill``, bitwise the stock fills), its rejection step runs REJ_MC_FI on the
    slot-indexed draft rows (``rej_mc_fi``, bitwise FlashInfer's chain sampling), its draft draws run SFP_MC_FI
    (``sfp_mc_fi``, bitwise FlashInfer's sampling from probabilities), the DSpark ancestral draft loop's
    per-position gather, widening add and proposal store run in one launch each (DEP_MARKOV_GLUE, ``markov_glue``,
    bitwise the stock lines), the logits processor widens the vocabulary slice of the lm_head output in one
    vectorized launch (DEP_LOGITS_WIDEN, ``logits_widen``, bitwise ``.float()``), on the MegaMoE local backend the MoE
    layers' top-k reduce runs in the shared-expert add and, where that norm is FlashInfer's, the boundary add + RMSNorm
    (DEP_TAIL, ``tail``, bitwise the stock tail), each layer boundary's add + RMSNorm hands the next qkv_proj its MXFP8
    form (DEP_NORM_MXFP8, ``norm_mxfp8``, bitwise the stock norm and quantize), each MoE layer's post-attention
    add + RMSNorm writes its gate's replicated operand (DEP_ROUTER_OPERAND, ``router_operand``, bitwise the stock norm
    and ``torch.cat``) and its shared expert's gate_up_proj MXFP8 operand, the shared expert's SwiGLU writes its
    down_proj's (DEP_SHARED_MXFP8, ``shared_mxfp8``, bitwise the stock norm, SwiGLU and quantizes), at 8-way expert
    parallelism the MegaMoE launches of the buckets above 32 up to 64 rows per rank keep the shared expert beside them
    on the aux stream (DEP_CORUN_BAND, ``corun_band``, the same outputs), the buckets above 64 up to 128 run it there
    on a 28-SM green-context side stream beside the 184-CTA routed launch where every rank can split its device
    (DEP_SE_SIDE, ``se_side``, the same outputs), and
    qkv_proj / o_proj run the CuTe DSL MXFP8 GEMM at its pinned tuned tactic (DEP_PROJ_PIN,
    ``proj_pin``, bitwise the default tactic). With TRTLLM_M31_DISABLE_DEP_LC2=1 the outputs are bitwise the stock DEP
    deployment's when both run the same autotuner tactics. Construction refuses a deployment other than DEP on
    MEGAMOE_CUTEDSL, and any of
    the config's MoE layers (``get_moe_layer_ids``) whose implementation identity is not the arm's exact one
    (``engagement.ARM_IDENTITIES``; the resolver's fallback is CutlassFusedMoE), then logs each rank's MoE identity.

    Built only for a deployment that sets ``modeling_v2`` itself (``@disable_v2_by_default``).
    """

    runtime_hooks = dep_runtime_hooks()

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        assert torch.cuda.get_device_capability() == _SM, (
            f"this target is built for sm_{_SM[0]}{_SM[1]}, the device is sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        mapping = model_config.mapping
        mismatch = deployment_mismatch(mapping, model_config.moe_backend)
        if mismatch is not None:
            raise NoTarget(
                f"the MiniMax-M3.1 DEP target does not serve this deployment: {mismatch}"
            )
        super().__init__(model_config)
        # The family's MSA backend in place of main's when the engine builds its step metadata (msa.engine).
        for layer in self.model.layers:
            _msa_engine.use_family_backend(layer.self_attn)
        if getattr(self.model, "disagg_remote_tail_replay", False):
            raise NoTarget(
                "the MiniMax-M3.1 DEP target does not serve the attention-DP remote tail replay"
            )
        _, moe_layer_ids = get_moe_layer_ids(get_text_config(model_config.pretrained_config))
        identities = moe_engagement(
            self.model.layers,
            moe_layer_ids,
            MoESchedulerKind.FUSED_COMM,
            # A runtime whose MoE resolution cannot select the MegaMoE local implementation builds the stock one
            # whatever the switch says.
            arm_identity() if _megamoe_local.is_registered() else ARM_IDENTITIES[False],
        )
        for line in identity_lines(mapping.rank, mapping.world_size, identities, _KERNEL):
            logger.info(line)
        self._rank = mapping.rank
        self._hidden_size = get_text_config(model_config.pretrained_config).hidden_size
        self._lc2_layers = 0
        self._merge_layers = 0
        self._norm_sites = 0
        self._router_sites = 0
        self._shared_gate_up = 0
        self._shared_down = 0
        self._corun_band_layers = 0
        self._shared_plan_layers = 0
        self._se_side = (0, None, None)
        self._proj_pins = 0
        self._idxk_producer_nvfp4 = False
        self._idx_pdl = False
        self._native_swiglu = 0
        self._mxdraft = (0, 0)
        self._plain_norms = 0
        self._dspark_glue = {}
        self._pad_mask = _pad.PadMask(mapping.rank)
        self._pad_layers = 0
        _install_sampler_overrides()
        self._dflash_merge = _dflash_merge.install()

    def cache_derived_state(self) -> None:
        """The built-in model's derived state, then, once the weights are real and before the executor sizes its KV
        pool, LC2's decode hooks with their o_proj hand-off, the tail, the boundary norms' qkv_proj hand-off, the
        post-attention norms' gate operand, the horizontal producer's direct NVFP4 index-K pages, and the workspaces of
        LC2's split, OWN-T, the three hand-offs, both sampler overrides, the rejection override and the draw override on
        this rank's device. The hooks go on here, not in ``__init__``: with deferred weight creation a layer's MSA
        decode library exists only once ``create_weights`` has built its FMHA manager, and a gate's split BF16 weight
        only once its weights load."""
        super().cache_derived_state()
        # NATIVE_ADD_RMSNORM's plain-norm wrappers come off first so the hand-off installs below see the stock forwards
        # (they decline a norm whose forward another install replaced); they go back on after them.
        _native_norm.uninstall_plain(self.model.modules())
        self._lc2_layers = _lc2.install(self.model.layers)
        self._merge_layers = _mx.install(self.model.layers, _lc2.serves) if self._lc2_layers else 0
        tail_a, tail_b = _tail.install(self.model.layers, self.model_config)
        # The shared expert's placement by the MegaMoE backend's answers, on every MoE block the tail did not take too.
        self._shared_plan_layers = _tail.install_shared_plan(self.model.layers)
        self._norm_sites = _nm.install(self.model.layers)
        self._router_sites = _ro.install(self.model.layers)
        self._shared_gate_up, self._shared_down = _sm.install(self.model.layers, _ro.is_site)
        # DEP_SE_SIDE: a collective across ranks; where it engages the co-run band widens to its own.
        self._se_side = _ss.install(self.model.layers, _tail.serves)
        self._corun_band_layers = _cb.install(
            self.model.layers, _ss.CORUN_BAND if self._se_side[0] else _cb.DEFAULT_BAND
        )
        self._proj_pins = _pp.install(self.model.layers, self.model_config)
        self._idxk_producer_nvfp4 = _ik.install()
        self._idx_pdl = _ipdl.install()
        # NATIVE_SWIGLU: the dense layers' MLPs and the drafter's MLPs (the MoE layers' shared experts are
        # DEP_SHARED_MXFP8's).
        draft = getattr(self, "draft_model", None)
        self._native_swiglu = _swiglu.install(
            [getattr(layer, "mlp", None) for layer in self.model.layers]
            + (list(draft.modules()) if draft is not None else [])
        )
        # DEP_DRAFTER_MXFP8: the drafter's qkv / o / gate_up / down projections as MXFP8 GEMMs on the checkpoint's own
        # MXFP8 weights (vLLM's DEP drafter layout; not bitwise the BF16 GEMMs, gated on natural AL). After
        # NATIVE_SWIGLU, so the rows it does not take keep that activation. TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1 keeps
        # them BF16.
        self._mxdraft = _mxd.install(draft)
        # NATIVE_ADD_RMSNORM: the target model's remaining FlashInfer fused add + RMSNorm calls (the norms no
        # hand-off install took; tail.py's own check, so DEP_TAIL's form (b) still finds FlashInfer's kernel form).
        self._plain_norms = _native_norm.install_plain(
            self.model.modules(), _tail.flashinfer_norm_decline_reason
        )
        # DEP_DSPARK_GLUE: the TP targets' DSpark drafter-glue set on the in-tree DSpark worker
        # (ops/dspark/dspark_glue_install; CUDA C++ ops, bitwise, the stock lines for any layout an op does not
        # take). The lean drafter glue is on at every draft length here: DEP has no drafter all-reduce (what keeps the
        # TP targets' lean glue to short drafts), and in bound mode with shift_label and block_size == K its sites are
        # bitwise (dflash.py keeps the stock lines for any other layout).
        # TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1 turns the lean glue off; TRTLLM_M31_DISABLE_DEP_DSPARK_GLUE=1 installs
        # nothing.
        worker = getattr(self, "spec_worker", None)
        self._dspark_glue = (
            {}
            if m31_disabled("DEP_DSPARK_GLUE")
            else _dsg.install_dspark_glue(
                worker,
                draft,
                draft_len=int(
                    getattr(getattr(worker, "spec_config", None), "max_draft_len", 0) or 0
                ),
                device=torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else None,
                s10_any_draft_len=True,
                disabled=m31_disabled,
            )
        )
        armed = armed_hooks(self.runtime_hooks)
        hooks = f"{len(armed)} armed" if self.runtime_hooks is not None else "built-in"
        handoff = (
            f"on {self._merge_layers} layers"
            if self._merge_layers
            else (
                "off (TRTLLM_M31_DISABLE_DEP_MERGE_MXFP8=1)"
                if not _mx.DEP_MERGE_MXFP8
                else "not taken"
            )
        )
        lc2 = (
            f"on {self._lc2_layers} layers (OWN-T {_lc2.own_t_status()}; "
            f"split PDL {'on' if _lc2.DEP_LC2_PDL else 'off'}; "
            f"MXFP8 o_proj hand-off {handoff})"
            if self._lc2_layers
            else ("off (TRTLLM_M31_DISABLE_DEP_LC2=1)" if not _lc2.DEP_LC2 else "not taken")
        )
        tail = (
            f"form (b) reduce + shared add + add-RMSNorm on {tail_b} layers, "
            f"form (a) reduce + shared add on {tail_a} layers"
            if tail_a or tail_b
            else ("off (TRTLLM_M31_DISABLE_DEP_TAIL=1)" if not _tail.DEP_TAIL else "not installed")
        ) + f"; shared-expert placement by the MegaMoE backend on {self._shared_plan_layers} layers"
        norm_handoff = (
            f"on {self._norm_sites} layer boundaries"
            if self._norm_sites
            else (
                "off (TRTLLM_M31_DISABLE_DEP_NORM_MXFP8=1)"
                if not _nm.DEP_NORM_MXFP8
                else "not installed"
            )
        )
        router_operand = (
            f"on {self._router_sites} MoE layers"
            if self._router_sites
            else (
                "off (TRTLLM_M31_DISABLE_DEP_ROUTER_OPERAND=1)"
                if not _ro.DEP_ROUTER_OPERAND
                else "not installed"
            )
        )
        shared = (
            f"gate_up_proj on {self._shared_gate_up} MoE layers, down_proj on {self._shared_down} MoE layers"
            if self._shared_gate_up or self._shared_down
            else (
                "off (TRTLLM_M31_DISABLE_DEP_SHARED_MXFP8=1)"
                if not _sm.DEP_SHARED_MXFP8
                else "not installed"
            )
        )
        band = _ss.CORUN_BAND if self._se_side[0] else _cb.DEFAULT_BAND
        corun_band = (
            f"{band[0]}:{band[1]} on {self._corun_band_layers} MoE layers"
            if self._corun_band_layers
            else f"not installed (EP {_cb.EP_SIZE} only; an explicit or disabled band stays as set)"
        )
        proj_pin = (
            f"on {self._proj_pins} projections"
            if self._proj_pins
            else (
                "off (TRTLLM_M31_DISABLE_DEP_PROJ_PIN=1)"
                if not _pp.DEP_PROJ_PIN
                else "not installed"
            )
        )
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: hooks {hooks}; DEP_LC2 decode {lc2}; DEP_TAIL {tail}; "
            f"DEP_NORM_MXFP8 qkv_proj hand-off {norm_handoff}; "
            f"DEP_ROUTER_OPERAND gate operand from the post-attention norm {router_operand}; "
            f"DEP_SHARED_MXFP8 shared expert MXFP8 hand-offs {shared}; "
            f"DEP_CORUN_BAND shared-expert co-run band {corun_band}; "
            f"DEP_PROJ_PIN qkv_proj / o_proj pinned CuTe DSL tactics {proj_pin}; "
            f"DEP_IDXK_PRODUCER_NVFP4 horizontal producer NVFP4 index-K pages "
            f"{_ik.status(self._idxk_producer_nvfp4)}; "
            f"DEP_IDX_PDL producer early trigger + reader pre-wait prefetch {_ipdl.status(self._idx_pdl)}; "
            "REJ_MC not installed; shared sm_107 opt-ins not armed"
        )
        logger.info(f"[M3.1 DEP] rank {self._rank}: armed hooks {' '.join(armed) or 'none'}")
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: DEP_SE_SIDE shared expert on a green side stream "
            f"{_ss.engagement(*self._se_side)}"
        )
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: NATIVE_SWIGLU dense + drafter MLPs "
            f"{_swiglu.engagement(self._native_swiglu)}"
        )
        glue = (
            "off (TRTLLM_M31_DISABLE_DEP_DSPARK_GLUE=1)"
            if m31_disabled("DEP_DSPARK_GLUE")
            else _dsg.engagement(self._dspark_glue)
        )
        logger.info(f"[M3.1 DEP] rank {self._rank}: DEP_DSPARK_GLUE drafter glue {glue}")
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: NATIVE_DFLASH_MERGE drafter non-causal block merge "
            f"{'on' if getattr(self, '_dflash_merge', False) else 'not installed'}"
        )
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: DEP_DRAFTER_MXFP8 drafter projections {_mxd.engagement(self._mxdraft)}"
        )
        plain = (
            f"CUDA plain add + RMSNorm on {self._plain_norms} norms"
            if self._plain_norms
            else (
                "off (TRTLLM_M31_DISABLE_NATIVE_ADD_RMSNORM=1) or the op is not built"
                if _native_norm.add_rmsnorm_op() is None
                else "no FlashInfer norm left"
            )
        )
        logger.info(
            f"[M3.1 DEP] rank {self._rank}: NATIVE_ADD_RMSNORM plain boundary / layer norms {plain}"
        )
        self._pad_layers, left_out = _pad.install(self.model.layers, self._pad_mask)
        skip = _pad.arm_lc2(self._pad_mask, self._pad_layers, self._lc2_layers)
        if not _pad.PAD_MASK:
            pad = "off (TRTLLM_M31_DISABLE_MEGAMOE_PAD_MASK=1)"
        else:
            pad = f"on {self._pad_layers} MoE layers" if self._pad_layers else "not taken"
            pad += f" ({left_out})" if left_out else ""
            if self._pad_layers:
                pad += (
                    f", LC2 skip on {skip} layers"
                    if skip
                    else (
                        ", LC2 skip off (TRTLLM_M31_DISABLE_DEP_LC2_PAD_SKIP=1)"
                        if not _pad.LC2_PAD_SKIP
                        else ", LC2 skip off (no LC2 layer)"
                    )
                )
        logger.info(f"[M3.1 DEP] rank {self._rank}: PAD_MASK {pad}")
        device = next(self.parameters()).device
        if device.type != "cuda":
            return
        with torch.cuda.device(device):
            if self._lc2_layers:
                _lc2.reserve_workspace(device)
                _lc2.reserve_own_t_workspace(device)
            if self._merge_layers:
                _mx.reserve(device)
            if self._norm_sites:
                _nm.reserve(device, self._hidden_size)
            if self._router_sites:
                _ro.reserve(device, self._hidden_size)
            if self._shared_gate_up or self._shared_down:
                _sm.reserve(device)
            if _softmax.SOFTMAX_MC:
                _softmax.prebuild(device)
            if _topp.TOPP_DET:
                _topp.prebuild(device)
            if _rej.DEP_REJ_MC_FI:
                _rej.prebuild(device)
            if _sfp.DEP_SFP_MC_FI:
                _sfp.prebuild(device)

    def forward(
        self,
        attn_metadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """The built-in forward, with the step's pad-row masks built first when a MoE layer takes them (``pad_mask``):
        the MoE row mask and, when armed, the LC2 decode's KV lengths with the padding requests' set to 0."""
        if self._pad_layers:
            tokens = input_ids if input_ids is not None else inputs_embeds
            self._pad_mask.begin(attn_metadata, kwargs.get("spec_metadata"), tokens.shape[0])
        try:
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                **kwargs,
            )
        finally:
            self._pad_mask.end()
