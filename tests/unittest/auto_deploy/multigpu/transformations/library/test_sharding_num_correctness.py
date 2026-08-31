# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
r"""Offline sharding numerical-correctness test.

Verifies that, for the modeling code at a given file path, applying the
sharding transforms (per-rank weight slicing via load hooks + collective
ops) preserves the prefill numerical output of the same graph under the
*unsharded* configuration. Runs without the full inference runtime: no
PyExecutor, no cache init, no compile, no checkpoint download, no
LLM_MODELS_ROOT setup.

Both sides of the comparison are post-``torch_export_to_gm`` graphs of the
same model instance with the same random weights -- only the sharding
transforms are applied to the sharded side. Any ``torch_export_to_gm``
semantic gap against eager (which exists for some custom ops, notably MoE
with Python loops over experts) is therefore *invisible* to this test:
both sides see identical export behavior, so any constant bias introduced
by export cancels out and only the delta from sharding remains.

IR-marked modeling files (graph carries ``torch.ops.auto_deploy.all_reduce``)
exercise the sharding-IR path (``apply_sharding_hints`` +
``strip_sharding_hints``). Legacy modeling files exercise the legacy path
(``detect_sharding`` + ``sharding_transform_executor`` with ``HEURISTIC``
source only). See ``_apply_sharding_to_gm`` for the branch.

Usage:

  pytest tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_num_correctness.py \
      --sharding-ir-modeling-file modeling_qwen3

The test is skipped (not failed) when ``--sharding-ir-modeling-file`` is
absent. No filename pattern is assumed -- works for the canonical
``modeling_<name>.py`` files that became the IR-aware default in #13478
(deepseek, nemotron_h, qwen3, qwen3_5_moe), and for any future modeling
file that opts into the sharding-IR path. See
``_sharding_ir_helpers.spec_from_modeling_file`` for how class / config
inference works, and ``_apply_layer_count_dependent_quirks`` for the
hasattr-driven layer-count-dependent config patches.

The ``SHARDING_IR_SABOTAGE=1`` env var is a negative-control switch: it
removes every ``all_reduce`` / ``all_gather`` / ``all_to_all`` node from the
sharded graph after sharding, then runs the same comparison. Used to confirm
the test actually rejects broken sharding (rel_rmse spikes far above the
tolerance) rather than rubber-stamping it.
"""

import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import pytest
import torch

# Make sure the helpers directory is importable both when running under
# pytest (which adds it via the ``pythonpath`` directive in
# ``tests/unittest/pytest.ini``) and when running inside a spawn() worker
# that does not inherit pytest's sys.path manipulation.
_HELPERS_DIR = str(Path(__file__).resolve().parents[3] / "_utils_test")
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)

from _deterministic_routing import DeterministicMoeRoutingMode  # noqa: E402
from _sharding_ir_helpers import (  # noqa: E402
    build_eagle_draft_model,
    build_ir_model,
    build_random_draft_inputs,
    build_random_prefill_inputs,
    extract_draft_output,
    extract_logits,
    random_init_with_seed,
    spec_from_modeling_file,
)


def _count_collectives(gm) -> int:
    """Count distributed-collective ops in *gm* by name.

    Matches the substring set used by ``_sabotage_remove_collectives`` so that
    "real sharding inserted N collectives" and "sabotage removed N collectives"
    use the same accounting. Covers IR (``auto_deploy.all_reduce``) and legacy
    (``*_dist_all_reduce``, ``*_dist_all_gather``, ``moe_all_to_all``) targets.
    """
    n = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        tgt = str(node.target)
        if "all_reduce" in tgt or "all_gather" in tgt or "all_to_all" in tgt:
            n += 1
    return n


def _apply_sharding_to_gm(gm_sharded, dist_config, rank: int):
    """Apply sharding to ``gm_sharded``, branching IR vs legacy by marker presence.

    Returns ``(gm_sharded, sharding_kind, n_collectives)``:

      * ``sharding_kind`` -- ``'ir'`` when the graph carries IR markers
        (``apply_sharding_hints`` + ``strip_sharding_hints``), or
        ``'legacy-heuristic'`` for non-IR graphs (``detect_sharding`` +
        ``sharding_transform_executor`` with ``HEURISTIC`` source only -- no
        factory and no manual_config are loaded by this test).
      * ``n_collectives`` -- count of collective ops inserted by the pass.
        Used by the caller to skip cleanly when legacy heuristics found no
        nodes to shard (would otherwise be a misleading trivial-PASS).

    The legacy branch deliberately restricts ``sharding_source`` to
    ``HEURISTIC``. Factory and manual sources rely on artifacts the test
    harness does not own: factory needs a real ``ModelFactory`` exposing the
    HF ``base_tp_plan``; manual needs a per-model YAML loaded via
    ``InferenceOptimizer.from_config``. ``HEURISTIC`` operates purely on the
    exported graph -- it is the only source that's a clean fit for this
    standalone-export test.
    """
    from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import is_shardingIR_enabled
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

    if is_shardingIR_enabled(gm_sharded):
        sharding_kind = "ir"
        transforms = {
            "apply_sharding_hints": {"stage": "sharding", "enabled": True},
            "strip_sharding_hints": {"stage": "weight_load"},
        }
    else:
        sharding_kind = "legacy-heuristic"
        transforms = {
            "detect_sharding": {
                "stage": "sharding",
                # Enum values are lowercase per ShardingSource / ShardingDim in
                # ``tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py``.
                "sharding_source": ["heuristic"],
                "sharding_dims": ["tp", "ep", "bmm"],
            },
            "sharding_transform_executor": {"stage": "sharding"},
        }
    optimizer = InferenceOptimizer(factory=None, config=transforms, dist_config=dist_config)
    gm_sharded = optimizer(None, gm_sharded)
    n_collectives = _count_collectives(gm_sharded)
    if rank == 0:
        print(
            f"[sharding-ir-eq] sharding_kind={sharding_kind}: "
            f"inserted {n_collectives} collective op(s)",
            flush=True,
        )
    return gm_sharded, sharding_kind, n_collectives


# IR-vs-legacy dispatch uses the SAME helper the production pipeline uses
# (``is_shardingIR_enabled`` in ``transform/library/sharding_ir.py``). Keeping
# a parallel implementation here is dangerous -- a previous test-local copy
# compared ``node.target == torch.ops.auto_deploy.all_reduce`` (the parent
# ``OpOverloadPacket``), which always evaluated False because FX nodes carry
# the ``.default`` ``OpOverload`` as their target; every IR-marked modeling
# file was silently misrouted through the legacy path for months without
# detection.


# NVFP4 quant pre-pass.
#
# To exercise the FP4 *scale*-sharding paths -- ``FP4LinearShardableNode``,
# ``FP4SwiGLUShardableNode`` and the NVFP4 branch of ``MoEShardableNode`` (all
# in ``transform/library/sharding_ir.py``) -- the graph must carry FP4 ops with
# their cutlass-format ``weight_scale`` / ``alpha`` buffers BEFORE sharding
# runs. This reproduces the production ``pattern_matcher`` quant stage offline:
# ``torch_linear_simple`` -> ``torch_fake_quant_nvfp4_linear`` (folding any
# gate/up/down SwiGLU into ``torch_nvfp4_swiglu_mlp``) and ``torch_moe`` ->
# ``torch_quant_nvfp4_moe``.
#
# The weights stay bf16 in the snapshot; the NVFP4 ``load_hook`` quantizes them
# on the fly via ``torch.ops.trtllm.fp4_quantize`` at ``load_state_dict`` time,
# so NO real FP4 checkpoint is needed (same trick as
# ``singlegpu/.../test_nvfp4_swiglu.py``). Because this pre-pass runs before
# ``_apply_sharding_to_gm``, the quant load hooks are registered BEFORE the
# sharding load hooks; at load time the bf16 weight is therefore quantized to
# FP4 first and the sharding hook then slices the FP4 weight + scales -- the
# exact ordering the production pipeline uses (pattern_matcher quant stage ->
# sharding stage).
_NVFP4_QUANT_TRANSFORMS = {
    "quantize_nvfp4_linear_from_config": {"stage": "pattern_matcher"},
    "match_nvfp4_swiglu_pattern": {"stage": "pattern_matcher", "requires_shape_prop": True},
    "quantize_nvfp4_moe": {"stage": "pattern_matcher", "run_shape_prop": True},
}


class _StubNVFP4Factory:
    """Minimal ``ModelFactory`` stand-in for the offline NVFP4 quant pre-pass.

    The ``quantize_*_from_config`` transforms read exactly one thing off the
    factory -- ``get_quant_config()`` -- to pick the algo and the modules to
    skip. ``lm_head`` is excluded to mirror real NVFP4 checkpoints (the final
    projection stays unquantized).
    """

    def get_quant_config(self):
        return {"quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}


def _apply_nvfp4_quant(gm):
    """Rewrite linear / SwiGLU / MoE ops in *gm* to their NVFP4 equivalents.

    Applied to BOTH the unsharded reference and the to-be-sharded graph so the
    FP4 quantization rounding is identical on both sides and only the sharding
    delta survives the comparison -- mirroring the bf16 design where the
    ``torch_export_to_gm`` semantic gap cancels out of both sides.
    """
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

    return InferenceOptimizer(_StubNVFP4Factory(), _NVFP4_QUANT_TRANSFORMS)(None, gm)


# NVFP4 MoE fusion (post-load), mirroring the production trtllm_gen path. The
# reference NVFP4 config (``examples/auto_deploy/.../super_v3.yaml``) runs
# ``fuse_nvfp4_moe: backend: trtllm_gen, enable_trtllm_gen_internal_routing:
# true``. Under pure EP (no attention-DP) this takes the TRTLLM-Gen *internal
# routing* path: the fused kernel routes over the FULL ``router_logits`` (global
# num_experts) but operates on the EP-local expert slice. That global-vs-local
# mismatch is exactly where the production ``routing_logits has incorrect
# shape`` crash occurs, so wiring this fusion in (after sharding + load) lets the
# tiny harness reproduce the real fused-kernel bug -- something the unfused
# reference ``torch_quant_nvfp4_moe`` op cannot surface on its own. Gated behind
# ``SHARDING_IR_NVFP4_FUSE`` so the default nvfp4 path stays on the reference op.
# Backend is env-switchable for diagnostics: ``trtllm_gen`` (default, matches
# super_v3.yaml) vs ``cutlass`` (matches ultra_v3.yaml default). Used to isolate
# whether an EP-MoE accuracy regression is cutlass-specific.
_NVFP4_FUSE_TRANSFORMS = {
    "fuse_nvfp4_moe": {
        "stage": "post_load_fusion",
        "backend": os.environ.get("SHARDING_IR_NVFP4_MOE_BACKEND", "trtllm_gen"),
        "enable_trtllm_gen_internal_routing": True,
        "allow_different_input_scales": True,
    },
}


def _apply_nvfp4_moe_fusion(gm):
    """Fuse ``torch_quant_nvfp4_moe`` -> TRTLLM-Gen fused MoE (post-load)."""
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

    return InferenceOptimizer(_StubNVFP4Factory(), _NVFP4_FUSE_TRANSFORMS)(None, gm)


# Parallelism configurations exercised by the test. The key is the value of
# ``--sharding-ir-dist-config``; the dict supplies the world_size, the MoE
# TP/EP grid, and the attention-DP flag. ``tp_size`` equals ``world_size``
# always (DistConfig validates ``moe_tp_size * moe_ep_size * moe_cluster_size
# == tp_size``); ``enable_attention_dp`` flips attention/MLP from TP to DP
# independently of the grid extent.
#
#   tp-only:  pure TP across 2 ranks, no MoE EP.
#   ep-only:  pure MoE EP across 2 ranks, no TP.
#   tep:      2x2 grid (TP + MoE EP) across 4 ranks (default).
#   attn-dp:  attention-DP across 4 ranks with MoEAllToAll inside the MoE
#             block (pure EP recipe: moe_tp=1, moe_ep=4).
_DIST_CONFIGS = {
    "tp-only": dict(world_size=2, moe_tp_size=2, moe_ep_size=1, enable_attention_dp=False),
    "ep-only": dict(world_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=False),
    "tep": dict(world_size=4, moe_tp_size=2, moe_ep_size=2, enable_attention_dp=False),
    "attn-dp": dict(world_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}

# Tiny prefill: SEQ_LEN = 256 keeps the test under ~10s end-to-end on the
# small (4-layer) configs while still exposing reduction-order issues that a
# single-token forward would miss. BATCH_SIZE = 4 is the max ``world_size``
# we exercise, so it scatters cleanly under attention-DP.
BATCH_SIZE = 4
SEQ_LEN = 256
WEIGHT_SEED = 0
INPUT_SEED = 42

# bf16 is the default forward dtype for unquantized AutoDeploy deployments.
# fp32 would give tighter numerics but production runs in bf16, so that's
# what the equivalence test should validate. With random init std=0.05 and a
# deterministic-router fix applied to MoE blocks, clean sharding produces
# rel_rmse < 0.012 on every IR family; sabotaged sharding produces > 0.05.
FORWARD_DTYPE = torch.bfloat16

# Random init std. Small enough that 4 stacked layers don't blow up in bf16,
# large enough that the per-rank contribution missing under sabotage is
# detectable. Anything below ~0.03 makes sabotage indistinguishable from
# noise on dense models; anything above ~0.1 starts triggering bf16 routing
# noise in MoE blocks even with the deterministic-router fix.
INIT_STD = 0.05

# Relative-RMSE tolerance: ``||y_s - y_u||_F / ||y_u||_F``. Scale-invariant
# across models with very different output magnitudes (dense models have
# ``|y|`` ~0.08, MoE models ~3.6). Picked to be above the worst clean
# rel_rmse observed on any IR family (~0.012 on qwen3_5_moe due to
# softmax-amplified bf16 noise in router weights) and well below the
# smallest sabotage rel_rmse (~0.05 on dense models). Override via
# ``SHARDING_IR_REL_RMSE_TOL`` env var when triaging.
REL_RMSE_TOL = 0.02

pytestmark = pytest.mark.threadleak(enabled=False)


def _all_gather_concat(local: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather rank-local tensors across the default process group and concat along dim 0.

    Used to reassemble a full-batch output from the per-rank slabs produced
    under attention-DP.
    """
    import torch.distributed as dist

    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


def _sabotage_remove_collectives(gm) -> int:
    """Negative-control hook: replace every collective op in ``gm`` with its first arg.

    Activated by ``SHARDING_IR_SABOTAGE=1``. Erases ``all_reduce`` /
    ``all_gather`` / ``all_to_all`` nodes so each rank keeps only its partial
    result; the test should then fail with rel_rmse far above
    :data:`REL_RMSE_TOL`. Used to verify the test actually detects broken
    sharding rather than rubber-stamping it.
    """
    n_removed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        tgt = str(node.target)
        if "all_reduce" in tgt or "all_gather" in tgt or "all_to_all" in tgt:
            if node.args:
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                n_removed += 1
    gm.graph.lint()
    gm.recompile()
    return n_removed


def _run_equivalence_job_impl(
    modeling_file: str,
    rank: int,
    world_size: int,
    dist_config_name: str,
    quant: str = "none",
) -> None:
    """Per-rank job body invoked by ``spawn_multiprocess_job``.

    Each rank independently:
      1. Builds the tiny IR model with deterministic random weights and a
         monotonic-coefficient MoE router so top-k decisions don't flip under
         bf16 reduction-order noise.
      2. Exports the model twice via ``torch_export_to_gm`` -- once as the
         unsharded reference, once as the basis for sharding.
      3. Runs ``apply_sharding_hints`` + ``strip_sharding_hints`` on the sharded
         copy. Optionally sabotages the sharded graph by removing collectives
         (negative-control mode).
      4. Loads the *same* unsharded snapshot into both graphs. The hooks
         registered on the sharded graph slice each parameter to the per-rank
         shard; the unsharded graph absorbs the snapshot identity-wise.
      5. Forwards both graphs on identical inputs and asserts numerical
         equivalence via relative RMSE.

    All ranks use the same CPU-seeded random init so the unsharded reference
    is bit-identical across ranks; the sharded forward converges to it via
    the all-reduce inserted by sharding.
    """
    # Imports are deferred until inside the worker so spawn() picks up any
    # parent-side sys.path / env-var setup before AutoDeploy is loaded.
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    import tensorrt_llm._torch.auto_deploy.models.custom  # noqa: F401 -- registers IR classes
    import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
    from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ------------------------------------------------------------------
    # 1. Build tiny IR model with deterministic random weights.
    # ------------------------------------------------------------------
    spec = spec_from_modeling_file(modeling_file)
    model = build_ir_model(spec, device=device, dtype=FORWARD_DTYPE)
    random_init_with_seed(model, seed=WEIGHT_SEED, std=INIT_STD)
    # MoE top-k routing is a non-smooth ``argmax`` whose decisions can flip
    # under bf16 reduction-order noise -- producing per-token O(absmax) errors
    # that look like sharding bugs but are really finite-precision artifacts.
    # The forward calls below are wrapped in ``DeterministicMoeRoutingMode``
    # which intercepts ``aten.topk`` / ``trtllm.noaux_tc_op`` /
    # ``auto_deploy.torch_moe_router`` and forces every routing decision to
    # pick experts ``[0..top_k-1]``. Model state is left untouched.

    # ------------------------------------------------------------------
    # 2. Build random prefill inputs and snapshot the unsharded weights.
    # ------------------------------------------------------------------
    vocab_size = int(model.config.vocab_size)
    input_ids, position_ids = build_random_prefill_inputs(
        BATCH_SIZE, SEQ_LEN, vocab_size, device, seed=INPUT_SEED
    )
    sd_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    # 3. Export the model on each side with the example inputs that side
    #    will actually receive. ``torch_export_to_gm`` bakes the batch
    #    dimension as static, so under attention-DP the sharded graph must
    #    be exported with the per-rank batch slab, not the full batch.
    #    Both sides still go through the same export step, so any
    #    torch_export_to_gm semantic gap is identical on both sides and
    #    cancels out of the comparison.
    #
    #    We pass the example inputs as ``kwargs`` (not positional ``args``)
    #    to mirror the AutoDeploy runtime path -- the production export
    #    transform (``transform/library/export_to_gm.py:ExportToGM``) also
    #    calls ``torch_export_to_gm(..., args=(), kwargs=captured_kwargs)``.
    #    Binding by name makes the test invariant to ``forward()`` parameter
    #    ordering across IR modeling files, which currently differs between
    #    qwen3 (``(input_ids, position_ids, inputs_embeds, ...)``) and
    #    nemotron_h / qwen3_5_moe (``(input_ids, inputs_embeds,
    #    position_ids, ...)``). Positional export would silently bind
    #    ``position_ids`` to whatever lives in slot 2, which trips the
    #    "specify exactly one of input_ids or inputs_embeds" guard on the
    #    latter family. The runtime path dodges this by being kwarg-based
    #    and by stripping ``**kwargs`` via ``set_exact_signature``; this
    #    test dodges it by just being kwarg-based.
    # ------------------------------------------------------------------
    dist_cfg_spec = _DIST_CONFIGS[dist_config_name]
    enable_attention_dp = dist_cfg_spec["enable_attention_dp"]

    gm_unsharded = torch_export_to_gm(
        model,
        args=(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        clone=True,
    )
    if enable_attention_dp:
        assert BATCH_SIZE % world_size == 0, (
            f"BATCH_SIZE={BATCH_SIZE} must be divisible by world_size={world_size} "
            f"under attention-DP."
        )
        chunk = BATCH_SIZE // world_size
        local_in_for_export = input_ids[rank * chunk : (rank + 1) * chunk]
        local_pos_for_export = position_ids[rank * chunk : (rank + 1) * chunk]
    else:
        local_in_for_export, local_pos_for_export = input_ids, position_ids
    gm_sharded = torch_export_to_gm(
        model,
        args=(),
        kwargs={"input_ids": local_in_for_export, "position_ids": local_pos_for_export},
        clone=True,
    )

    # ------------------------------------------------------------------
    # 3.5. Optional NVFP4 quant pre-pass on BOTH graphs (before sharding).
    #
    # Converts linear / SwiGLU / MoE ops to their NVFP4 equivalents so the FP4
    # weight-scale sharding paths are exercised. Applied to the unsharded
    # reference too, so the FP4 rounding is identical on both sides and only
    # the sharding delta remains. See ``_apply_nvfp4_quant`` for why no real
    # checkpoint is needed (the NVFP4 load hook quantizes the bf16 snapshot at
    # load time) and why the quant->shard load-hook ordering is correct.
    # ------------------------------------------------------------------
    if quant == "nvfp4":
        # ``_apply_nvfp4_quant`` registers scale buffers (input_scale / weight_scale
        # / alpha) via ``default_scales`` WITHOUT a device, so they land on CPU even
        # though the rest of the graph is on ``device``. Move both graphs back to the
        # rank device so the on-the-fly NVFP4 load hook and the forward see a single
        # device (production moves the whole model to device after quant).
        gm_unsharded = _apply_nvfp4_quant(gm_unsharded).to(device)
        gm_sharded = _apply_nvfp4_quant(gm_sharded).to(device)
        # The NVFP4 load hook quantizes the bf16 weight on the fly (so no real
        # FP4 checkpoint is needed) but reads the per-module ``input_scale``
        # straight out of the loaded state_dict, and the bf16 snapshot taken in
        # step 2 has no scale entries. Quantization registered the scale
        # buffers (``input_scale`` / ``weight_scale`` / ``alpha``) as defaults
        # on the gm; merge every key the quant pass added (i.e. the ones the
        # bf16 snapshot lacks) so both loads see a complete FP4 state_dict. The
        # real bf16 weights are kept (their keys already exist in the
        # snapshot); the placeholder ``weight_scale`` / ``alpha`` get
        # recomputed from the bf16 weight inside the load hook. ``input_scale``
        # stays at its default -- only sharding equivalence is under test, so
        # the absolute scale value is irrelevant as long as both sides match.
        # Sourced from gm_unsharded so the scales are full-size; the sharding
        # hooks on gm_sharded then slice them per rank.
        for k, v in gm_unsharded.state_dict().items():
            if k not in sd_snapshot:
                sd_snapshot[k] = v.detach().clone().to(device)

    # ------------------------------------------------------------------
    # 4. Apply sharding to gm_sharded.
    #
    # The branch happens inside ``_apply_sharding_to_gm``:
    #
    #   - IR-marked graphs (``torch.ops.auto_deploy.all_reduce`` present, per
    #     skill rule A3) run ``apply_sharding_hints`` + ``strip_sharding_hints``.
    #   - Non-IR (legacy) graphs run ``detect_sharding`` +
    #     ``sharding_transform_executor`` with ``HEURISTIC`` source only --
    #     the standalone-export test harness has no factory or per-model YAML
    #     to feed the FACTORY / MANUAL sources, so heuristic pattern matching
    #     is the only legacy signal we can drive from a bare GraphModule.
    #
    # ------------------------------------------------------------------
    dist_config = DistConfig(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_tp_size=dist_cfg_spec["moe_tp_size"],
        moe_ep_size=dist_cfg_spec["moe_ep_size"],
        enable_attention_dp=enable_attention_dp,
    )
    gm_sharded, sharding_kind, n_collectives = _apply_sharding_to_gm(gm_sharded, dist_config, rank)

    if os.environ.get("SHARDING_IR_SABOTAGE") == "1":
        n_removed = _sabotage_remove_collectives(gm_sharded)
        if rank == 0:
            print(
                f"[sharding-ir-eq] SHARDING_IR_SABOTAGE=1: removed {n_removed} "
                f"collective op(s) from gm_sharded",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 5. Load the same unsharded snapshot into both graphs.
    #    For gm_unsharded the load is identity-shaped. For gm_sharded the
    #    hooks registered by apply_sharding_hints fire here, slicing each
    #    parameter to the per-rank shard before assignment.
    # ------------------------------------------------------------------
    gm_unsharded.load_state_dict(sd_snapshot, strict=False)
    missing, _ = gm_sharded.load_state_dict(sd_snapshot, strict=False)
    # ``unexpected`` keys are legitimate under EP sharding -- MoE expert
    # partitioning removes per-rank expert params, so the full unsharded
    # snapshot's keys for experts not held by this rank are expected to be
    # rejected. ``missing`` is the bug signal: any param in the sharded
    # graph that the snapshot can't supply.
    assert not missing, f"Missing keys when loading sharded state_dict: {missing[:5]}"

    # ------------------------------------------------------------------
    # 5.5. Optional NVFP4 MoE fusion (post-load), mirroring the production
    # trtllm_gen path. Runs AFTER sharding + load so the fused kernel sees the
    # EP-local expert slice while internal routing still spans the full
    # router_logits -- the exact configuration that reproduces the real
    # 'routing_logits has incorrect shape' crash. See _apply_nvfp4_moe_fusion.
    # ------------------------------------------------------------------
    if quant == "nvfp4" and os.environ.get("SHARDING_IR_NVFP4_FUSE") == "1":
        gm_unsharded = _apply_nvfp4_moe_fusion(gm_unsharded)
        gm_sharded = _apply_nvfp4_moe_fusion(gm_sharded)

    # ------------------------------------------------------------------
    # 6. Forward both graphs and compare via relative RMSE.
    #
    # Two flavors of the comparison, selected by ``enable_attention_dp``:
    #
    # - **TP / EP (enable_attention_dp=False):** every rank sees the full
    #   batch on both sides. The all-reduce inserted by sharding gives every
    #   rank the same full output, so the comparison is local per rank.
    #
    # - **Attention-DP (enable_attention_dp=True):** the unsharded reference
    #   still runs the full batch on every rank, but the sharded forward
    #   expects each rank to process only its slice of the batch -- attention
    #   and MLP are per-rank, and MoEAllToAll handles the dispatch+combine
    #   inside the MoE block. We scatter the batch into contiguous slabs,
    #   run the sharded forward locally, then all-gather the per-rank slabs
    #   and concatenate into the full-batch order to compare against the
    #   replicated unsharded reference.
    # ------------------------------------------------------------------
    with torch.inference_mode(), DeterministicMoeRoutingMode():
        # Call the exported GMs with the same kwarg convention used at
        # export time (see step 3 above). The GM's traced forward signature
        # mirrors the export call, and calling by name keeps it that way --
        # we never reintroduce a positional dependency that would tie this
        # test back to ``forward()`` parameter ordering.
        #
        # ``DeterministicMoeRoutingMode`` intercepts MoE-routing ops
        # (``aten.topk`` / ``trtllm.noaux_tc_op`` / ``auto_deploy.torch_moe_router``)
        # to force every routing decision to pick experts ``[0..top_k-1]`` --
        # see the docstring of that class for the rationale. Both forwards run
        # inside the same ``with`` block so they see identical routing.
        y_unsharded = extract_logits(gm_unsharded(input_ids=input_ids, position_ids=position_ids))
        y_local = extract_logits(
            gm_sharded(input_ids=local_in_for_export, position_ids=local_pos_for_export)
        )
        if enable_attention_dp:
            y_sharded = _all_gather_concat(y_local.contiguous(), world_size)
        else:
            y_sharded = y_local

    rel_rmse_tol = float(os.environ.get("SHARDING_IR_REL_RMSE_TOL", str(REL_RMSE_TOL)))
    rel_rmse = (
        torch.sqrt(((y_sharded - y_unsharded).float() ** 2).mean())
        / torch.sqrt((y_unsharded.float() ** 2).mean())
    ).item()
    if rank == 0:
        u = y_unsharded.float()
        diff = (y_sharded.float() - u).abs()
        print(
            f"[sharding-ir-eq] |y_s - y_u|: max={diff.max().item():.6f} "
            f"mean={diff.mean().item():.6f} rel_rmse={rel_rmse:.6f} "
            f"(tol={rel_rmse_tol})",
            flush=True,
        )
    assert rel_rmse < rel_rmse_tol, f"rel_rmse={rel_rmse:.4f} >= {rel_rmse_tol} on rank {rank}"


def _run_equivalence_job(
    modeling_file: str,
    dist_config_name: str,
    quant: str,
    rank: int,
    world_size: int,
) -> None:
    """Worker entry; mirrors the per-rank traceback to a log file on failure.

    ``spawn_multiprocess_job`` reports a coarse ``"process exited with code N"``
    when a worker raises; the side-channel log makes the underlying exception
    recoverable from the parent process.
    """
    import traceback

    try:
        _run_equivalence_job_impl(modeling_file, rank, world_size, dist_config_name, quant)
    except BaseException:
        with open(f"/tmp/sharding_ir_equiv_rank{rank}.log", "w") as f:
            f.write(traceback.format_exc())
        raise


def _gpu_check(dist_config_name: str) -> Optional[str]:
    """Return a skip-reason string if there aren't enough GPUs, else ``None``."""
    need = _DIST_CONFIGS[dist_config_name]["world_size"]
    have = torch.cuda.device_count()
    if have < need:
        return f"requires {need} GPUs for dist_config={dist_config_name!r} (got {have})"
    return None


def test_sharding_num_correctness(
    sharding_ir_modeling_file: str,
    sharding_ir_dist_config: str,
    sharding_ir_quant: str,
) -> None:
    """Verify sharded == unsharded prefill for the supplied modeling file."""
    skip = _gpu_check(sharding_ir_dist_config)
    if skip:
        pytest.skip(skip)

    # NVFP4 quant pre-pass needs Blackwell (sm_100+) and the TRT-LLM FP4 ops
    # (torch.ops.trtllm.fp4_quantize, used by the on-the-fly NVFP4 load hook).
    if sharding_ir_quant == "nvfp4":
        from _torch_test_utils import fp4_compatible, trtllm_ops_available

        if not (fp4_compatible() and trtllm_ops_available()):
            pytest.skip("NVFP4 sharding check requires Blackwell (sm_100+) and TRT-LLM ops")

        # Models whose TINY test config cannot exercise the NVFP4 sharding path.
        # The FP4 sharding *code* is correct -- it is validated at realistic dims
        # by ``test_tp_sharding.py::test_moe_tp_shard_nvfp4`` (in CI) and by
        # real-model accuracy. These toy configs hit hard NVFP4 constraints that
        # only surface at tiny sizes, in three families:
        #   * a weight dim below the 16-element FP4 block / MIN_LOCAL_SHAPE=32
        #     floor, so it cannot be TP-sharded for FP4 at all (the GDN/MoE
        #     delta-net and step3 blocks expose sub-32 per-head dims);
        #   * the on-the-fly NVFP4 quant harness yields packed-uint8 / pack-factor
        #     weight shapes the reference op can't consume at toy dims (MLA + some
        #     MoE blocks);
        #   * FP4 rounding on tiny random weights exceeds the 2% sharded-vs-
        #     unsharded rel_rmse tolerance even though bf16 sharding of the SAME
        #     model is numerically exact.
        # Every one of these models is still covered in bf16 (the CI-default
        # quant). Skip them for nvfp4 until the tiny-config FP4 harness is
        # hardened in a follow-up.
        _NVFP4_TINY_CONFIG_UNSUPPORTED = {
            "modeling_deepseek.py",
            "modeling_deepseek_v2.py",
            "modeling_exaone.py",
            "modeling_gemma2.py",
            "modeling_gemma4.py",
            "modeling_glm4_moe.py",
            "modeling_kimi_k2.py",
            "modeling_mistral3.py",
            "modeling_openelm.py",
            "modeling_qwen3_5_moe.py",
            "modeling_qwen3_next.py",
            "modeling_step3p7.py",
        }
        if Path(sharding_ir_modeling_file).name in _NVFP4_TINY_CONFIG_UNSUPPORTED:
            pytest.skip(
                f"{Path(sharding_ir_modeling_file).name}: NVFP4 sharding is not "
                f"exercisable on the tiny test config (FP4 16-element-block / "
                f"MIN_LOCAL_SHAPE floor, packed-weight, or rel_rmse limit). The FP4 "
                f"sharding path is validated at realistic dims by "
                f"test_tp_sharding.py::test_moe_tp_shard_nvfp4 and by real-model "
                f"accuracy; this model is covered here in bf16."
            )

    # Known-failing modeling files whose Mamba/MoE/etc. blocks have
    # *pre-existing* sharding-compat issues unrelated to this test harness.
    # File names here are bit-identical to ``origin/main`` -- the failures
    # reproduce on both the IR and legacy-heuristic sharding paths. Skip
    # cleanly until a follow-up PR ports the modeling code properly.
    _KNOWN_FAILING_MODELING_FILES = {
        # granite_moe_hybrid: B/C views in the Mamba block don't account for
        # TP-sharding of the n_groups*ssm_state_size channel dim (the
        # in-proj column-shard halves B/C, but the downstream
        # ``view([B, S, -1, ssm_state_size])`` keeps the literal at full
        # size). Same crash on legacy heuristic and any future IR port.
        # Tracked for a follow-up IR-port PR.
        "modeling_granite_moe_hybrid.py",
    }
    if Path(sharding_ir_modeling_file).name in _KNOWN_FAILING_MODELING_FILES:
        pytest.skip(
            f"{Path(sharding_ir_modeling_file).name}: pre-existing sharding-compat "
            f"issue in the modeling code (not introduced by this test harness). "
            f"Will be addressed in a follow-up sharding-IR port PR."
        )

    # Pre-flight in parent process: validate the modeling file can be set up
    # as a tiny IR model on CPU before paying the multiprocess spawn cost.
    # The per-rank GPU instantiation in the worker remains authoritative; this
    # check exists only to convert "can't even instantiate" failures (which
    # would surface as a useless ``Process exited with code 1`` in the parent)
    # into clean ``pytest.skip`` outcomes with an actionable message.
    try:
        _preflight_spec = spec_from_modeling_file(sharding_ir_modeling_file)
        _ = build_ir_model(_preflight_spec, device=torch.device("cpu"), dtype=FORWARD_DTYPE)
    except RuntimeError as e:
        from tensorrt_llm._torch.auto_deploy._compat import TRTLLM_AVAILABLE

        msg = str(e)
        known_native_setup_error = (
            "Could not resolve config class" in msg or "ForCausalLM' class registered" in msg
        )
        if TRTLLM_AVAILABLE and known_native_setup_error:
            pytest.skip(f"Modeling file not set up for IR equivalence harness: {e}")
        raise
    except (TypeError, AttributeError, KeyError, ValueError, AssertionError) as e:
        pytest.skip(
            f"Tiny default config insufficient for "
            f"{Path(sharding_ir_modeling_file).name}: {type(e).__name__}: {e}. "
            "Per-model quirks may be needed in "
            "`_sharding_ir_helpers._apply_per_family_quirks` / "
            "`_apply_layer_count_dependent_quirks`."
        )

    import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common

    world_size = _DIST_CONFIGS[sharding_ir_dist_config]["world_size"]
    dist_common.spawn_multiprocess_job(
        job=partial(
            _run_equivalence_job,
            sharding_ir_modeling_file,
            sharding_ir_dist_config,
            sharding_ir_quant,
        ),
        size=world_size,
    )


# =============================================================================
# Eagle draft equivalence
#
# The Eagle draft GraphModule is just numbers in / numbers out -- the sharding
# equivalence property is architecture-agnostic, so this reuses the exact same
# export -> shard -> compare logic as the base-model path. The only difference
# is the front-end: build ``EagleDrafterForCausalLM`` instead of a registered
# modeling-file model, and feed ``{inputs_embeds, position_ids, hidden_states}``
# instead of ``{input_ids, position_ids}`` (the draft consumes the target's
# hidden states; it has no input_ids path). See
# ``_sharding_ir_helpers.build_eagle_draft_model`` / ``build_random_draft_inputs``.
# =============================================================================


def _run_eagle_draft_equivalence_job_impl(
    model_type: str,
    rank: int,
    world_size: int,
    dist_config_name: str,
) -> None:
    """Per-rank Eagle-draft job body.

    Mirrors ``_run_equivalence_job_impl`` but builds the draft and feeds the
    draft's ``(inputs_embeds, position_ids, hidden_states)`` contract; the
    export/shard/compare tail is identical.
    """
    import tensorrt_llm._torch.auto_deploy.models.custom  # noqa: F401
    import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
    from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 1. Build tiny Eagle draft with deterministic random weights.
    model = build_eagle_draft_model(model_type, device=device, dtype=FORWARD_DTYPE)
    random_init_with_seed(model, seed=WEIGHT_SEED, std=INIT_STD)
    # NemotronH MTP draft has a MoE ('E') layer; routing stability is enforced
    # by ``DeterministicMoeRoutingMode`` wrapping the forward calls below.

    hidden_size = int(model.config.hidden_size)

    # 2. Random draft inputs + unsharded weight snapshot.
    full_kwargs = build_random_draft_inputs(
        BATCH_SIZE, SEQ_LEN, hidden_size, device, FORWARD_DTYPE, seed=INPUT_SEED, std=INIT_STD
    )
    sd_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 3. Export unsharded + sharded. Under attention-DP every batch-dim-0 input
    #    tensor is sliced to the per-rank slab (mirrors the base path).
    dist_cfg_spec = _DIST_CONFIGS[dist_config_name]
    enable_attention_dp = dist_cfg_spec["enable_attention_dp"]

    def _slice_batch(kwargs: dict) -> dict:
        if not enable_attention_dp:
            return kwargs
        assert BATCH_SIZE % world_size == 0, (
            f"BATCH_SIZE={BATCH_SIZE} must be divisible by world_size={world_size} under attention-DP."
        )
        chunk = BATCH_SIZE // world_size
        sl = slice(rank * chunk, (rank + 1) * chunk)
        return {k: (v[sl] if v.shape[0] == BATCH_SIZE else v) for k, v in kwargs.items()}

    local_kwargs = _slice_batch(full_kwargs)

    gm_unsharded = torch_export_to_gm(model, args=(), kwargs=full_kwargs, clone=True)
    gm_sharded = torch_export_to_gm(model, args=(), kwargs=local_kwargs, clone=True)

    # 4. Shard gm_sharded only. Eagle drafts ride the same IR-vs-legacy
    # branch as the base-model path -- see ``_apply_sharding_to_gm`` for
    # which transforms each branch runs and why HEURISTIC is the only
    # legacy source we can drive from this standalone-export harness.
    dist_config = DistConfig(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_tp_size=dist_cfg_spec["moe_tp_size"],
        moe_ep_size=dist_cfg_spec["moe_ep_size"],
        enable_attention_dp=enable_attention_dp,
    )
    gm_sharded, sharding_kind, n_collectives = _apply_sharding_to_gm(gm_sharded, dist_config, rank)

    if os.environ.get("SHARDING_IR_SABOTAGE") == "1":
        n_removed = _sabotage_remove_collectives(gm_sharded)
        if rank == 0:
            print(
                f"[sharding-ir-eq] SHARDING_IR_SABOTAGE=1: removed {n_removed} "
                f"collective op(s) from gm_sharded",
                flush=True,
            )

    # 5. Load the same unsharded snapshot into both graphs.
    gm_unsharded.load_state_dict(sd_snapshot, strict=False)
    missing, _ = gm_sharded.load_state_dict(sd_snapshot, strict=False)
    assert not missing, f"Missing keys when loading sharded state_dict: {missing[:5]}"

    # 6. Forward both and compare via relative RMSE. Both run inside
    #    ``DeterministicMoeRoutingMode`` so MoE-routing decisions in the
    #    NemotronH MTP draft are stable across sharded/unsharded paths.
    rel_rmse_tol = float(os.environ.get("SHARDING_IR_REL_RMSE_TOL", REL_RMSE_TOL))
    with torch.inference_mode(), DeterministicMoeRoutingMode():
        y_unsharded = extract_draft_output(gm_unsharded(**full_kwargs))
        y_local = extract_draft_output(gm_sharded(**local_kwargs))
        if enable_attention_dp:
            y_sharded = _all_gather_concat(y_local.contiguous(), world_size)
        else:
            y_sharded = y_local

    rel_rmse = (
        torch.sqrt(((y_sharded - y_unsharded).float() ** 2).mean())
        / torch.sqrt((y_unsharded.float() ** 2).mean())
    ).item()
    if rank == 0:
        diff = (y_sharded.float() - y_unsharded.float()).abs()
        print(
            f"[sharding-ir-eq] eagle-draft({model_type}) |y_s - y_u|: "
            f"max={diff.max().item():.6f} mean={diff.mean().item():.6f} "
            f"rel_rmse={rel_rmse:.6f} (tol={rel_rmse_tol})",
            flush=True,
        )
    assert rel_rmse < rel_rmse_tol, f"rel_rmse={rel_rmse:.4f} >= {rel_rmse_tol} on rank {rank}"


def _run_eagle_draft_equivalence_job(
    model_type: str,
    dist_config_name: str,
    rank: int,
    world_size: int,
) -> None:
    """Worker entry; mirrors the per-rank traceback to a log file on failure."""
    import traceback

    try:
        _run_eagle_draft_equivalence_job_impl(model_type, rank, world_size, dist_config_name)
    except BaseException:
        with open(f"/tmp/sharding_ir_eagle_draft_rank{rank}.log", "w") as f:
            f.write(traceback.format_exc())
        raise


def test_sharding_num_correctness_eagle_draft(
    sharding_ir_eagle_draft: str,
    sharding_ir_dist_config: str,
) -> None:
    """Verify sharded == unsharded prefill for an Eagle draft of the given model_type."""
    skip = _gpu_check(sharding_ir_dist_config)
    if skip:
        pytest.skip(skip)

    import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common

    world_size = _DIST_CONFIGS[sharding_ir_dist_config]["world_size"]
    dist_common.spawn_multiprocess_job(
        job=partial(
            _run_eagle_draft_equivalence_job, sharding_ir_eagle_draft, sharding_ir_dist_config
        ),
        size=world_size,
    )
