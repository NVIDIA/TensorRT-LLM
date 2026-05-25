# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
r"""Offline sharding-IR equivalence test.

Verifies that, for the sharding-IR modeling code at a given file path,
applying the IR sharding transforms (``apply_sharding_hints`` +
``strip_sharding_hints`` + per-rank weight slicing via load hooks) preserves
the prefill numerical output of the same graph under the *unsharded*
configuration. Runs without the full inference runtime: no PyExecutor, no
cache init, no compile, no checkpoint download, no LLM_MODELS_ROOT setup.

Both sides of the comparison are post-``torch_export_to_gm`` graphs of the
same model instance with the same random weights -- only ``apply_sharding_hints``
+ ``strip_sharding_hints`` are applied to the sharded side. Any
``torch_export_to_gm`` semantic gap against eager (which exists for some
custom ops, notably MoE with Python loops over experts) is therefore
*invisible* to this test: both sides see identical export behavior, so any
constant bias introduced by export cancels out and only the delta from
sharding remains.

Usage:

  pytest tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_ir_equivalence.py \
      --sharding-ir-modeling-file tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3.py

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

# Make sure the helpers directory is importable both when running under
# pytest (which adds it via the ``pythonpath`` directive in
# ``tests/unittest/pytest.ini``) and when running inside a spawn() worker
# that does not inherit pytest's sys.path manipulation.
_HELPERS_DIR = str(Path(__file__).resolve().parents[3] / "_utils_test")
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)


def _scrub_unbuilt_tensorrt_llm_from_sys_path() -> None:
    """Drop sys.path entries that contain an *uncompiled* tensorrt_llm tree.

    The pytest setup at ``tests/unittest/conftest.py`` and
    ``tests/unittest/utils/cpp_paths.py`` adds the repo root to sys.path. When
    the test is invoked from a git worktree that is *separate* from the build
    root, that worktree's ``tensorrt_llm/`` is a source-only checkout with no
    compiled C++ extensions, and spawn workers pick it up before the editable
    install finder, breaking the runtime imports. This scrub is a no-op in
    CI / single-tree setups (where every sys.path entry containing
    ``tensorrt_llm`` also has the .so), so it's safe to always apply.

    Runs at module-import time so it takes effect *before* a spawn worker
    unpickles this module's symbols and triggers ``import tensorrt_llm``.
    """
    so_relpath = os.path.join("tensorrt_llm", "runtime", "kv_cache_manager_v2", "rawref")
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            real = os.path.realpath(entry)
        except OSError:
            continue
        init_py = os.path.join(real, "tensorrt_llm", "__init__.py")
        if not os.path.isfile(init_py):
            continue
        rawref_dir = os.path.join(real, so_relpath)
        if os.path.isdir(rawref_dir) and any(f.endswith(".so") for f in os.listdir(rawref_dir)):
            continue
        sys.path.remove(entry)


_scrub_unbuilt_tensorrt_llm_from_sys_path()

import pytest  # noqa: E402  -- after sys.path scrub to win over uncompiled tensorrt_llm
import torch  # noqa: E402
from _sharding_ir_helpers import (  # noqa: E402
    build_ir_model,
    build_random_prefill_inputs,
    extract_logits,
    fix_moe_routers_deterministic,
    random_init_with_seed,
    spec_from_modeling_file,
)

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
    # parent-side sys.path / env-var setup before tensorrt_llm is loaded.
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    import tensorrt_llm._torch.auto_deploy.models.custom  # noqa: F401 -- registers IR classes
    import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
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
    # Override the router weights / biases (and Qwen3.5-MoE's bias-less
    # router's forward) so top-k always picks experts ``[0..top_k-1]``
    # regardless of input. Routing decisions then become rock-stable across
    # ranks and precisions; only true sharding bugs cause output drift.
    n_routers_fixed = fix_moe_routers_deterministic(model)
    if rank == 0 and n_routers_fixed > 0:
        print(
            f"[sharding-ir-eq] fixed {n_routers_fixed} MoE router(s) to deterministic top-k",
            flush=True,
        )

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
    # 4. Apply the sharding transforms only to gm_sharded.
    # ------------------------------------------------------------------
    dist_config = DistConfig(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_tp_size=dist_cfg_spec["moe_tp_size"],
        moe_ep_size=dist_cfg_spec["moe_ep_size"],
        enable_attention_dp=enable_attention_dp,
    )
    sharded_transforms = {
        "apply_sharding_hints": {"stage": "sharding", "enabled": True},
        "strip_sharding_hints": {"stage": "weight_load"},
    }
    optimizer = InferenceOptimizer(factory=None, config=sharded_transforms, dist_config=dist_config)
    gm_sharded = optimizer(None, gm_sharded)

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
    with torch.inference_mode():
        # Call the exported GMs with the same kwarg convention used at
        # export time (see step 3 above). The GM's traced forward signature
        # mirrors the export call, and calling by name keeps it that way --
        # we never reintroduce a positional dependency that would tie this
        # test back to ``forward()`` parameter ordering.
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
        _run_equivalence_job_impl(modeling_file, rank, world_size, dist_config_name)
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


def test_sharding_ir_equivalence(
    sharding_ir_modeling_file: str,
    sharding_ir_dist_config: str,
) -> None:
    """Verify sharded == unsharded prefill for the supplied modeling file."""
    skip = _gpu_check(sharding_ir_dist_config)
    if skip:
        pytest.skip(skip)

    import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common

    world_size = _DIST_CONFIGS[sharding_ir_dist_config]["world_size"]
    dist_common.spawn_multiprocess_job(
        job=partial(_run_equivalence_job, sharding_ir_modeling_file, sharding_ir_dist_config),
        size=world_size,
    )
