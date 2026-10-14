# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step-loop hooks this target opts into (``RuntimeHooks``), measured on GB300 AgentX decode (tp4 and tp8).

* ``pack_step_inputs``: a decode step's small input copies go out as one upload and one scatter at the graph
  replay instead of a chain of copies the next graph waits on (~30 us of idle GPU at one request).
* ``skip_unchanged_h2d``: steady decode re-uploads slot indices, query lengths and page tables that rarely change.
* ``loop_on_execution_stream``: the overlap loop's own stream work and the forward on one stream, so step N+1's
  inputs wait on graph N in program order rather than through two cross-stream event hops.
* ``merge_tuned_tactics_across_ranks``: tensor-parallel peers capture the same GEMM tactics (rank-local profiling
  noise otherwise splits them, and the slowest rank sets every step).
* ``skip_greedy_rng_upload``: all-greedy speculative batches never read the Philox seeds.
* ``spec_step_fold``: the spec sampler's slot stores and the engine's next-step input gathers in one launch each.
* MSA: identity selections for short sequences (and short/long graph families), the one-launch KV-length update,
  unchanged block-table copies skipped, pool views built once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy, RuntimeHooks

from .kernels import msa_kv_update
from .spec_fold import M3SpecStepFold

if TYPE_CHECKING:
    from .modeling import ModelingV2MiniMaxM3Nvfp4Sm103Tp8


def make_runtime_hooks(
    model: "ModelingV2MiniMaxM3Nvfp4Sm103Tp8",
) -> Tuple[RuntimeHooks, Optional[M3SpecStepFold]]:
    """This target's hooks, and the speculative step fold they carry (None without speculation)."""
    spec_fold = M3SpecStepFold() if model.spec_config is not None else None
    hooks = RuntimeHooks(
        pack_step_inputs=True,
        skip_unchanged_h2d=True,
        loop_on_execution_stream=True,
        merge_tuned_tactics_across_ranks=True,
        skip_greedy_rng_upload=True,
        spec_step_fold=spec_fold,
        msa=MsaRuntimePolicy(
            short_sequence_identity=True,
            kv_update=msa_kv_update.msa_kv_update,
            skip_unchanged_block_offsets=True,
            reuse_pool_views=True,
        ),
    )
    return hooks, spec_fold
