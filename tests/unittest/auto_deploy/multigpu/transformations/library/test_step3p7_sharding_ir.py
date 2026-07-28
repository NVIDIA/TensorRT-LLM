# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharding numerical-correctness test for the StepFun Step-3.7-Flash AutoDeploy custom model.

This is a model-specific, directly-runnable wrapper around the generic sharding numerical-correctness
harness in ``test_sharding_num_correctness.py``. Unlike the generic test (which is skipped unless
``--sharding-ir-modeling-file`` is supplied on the command line), this file pins the modeling file
to ``modeling_step3p7.py`` and parametrizes over every parallelism configuration, so it runs with a
plain::

    pytest tests/unittest/auto_deploy/multigpu/transformations/library/test_step3p7_sharding_ir.py

For each config it builds a tiny (4-layer, hidden_size=64) Step-3.7-Flash instance, exports it,
applies ``apply_sharding_hints`` to one copy, and asserts the sharded prefill matches the unsharded
reference within a relative-RMSE tolerance. It needs no PyExecutor / compile / checkpoint download
and skips automatically when fewer GPUs are available than a config requires.

Tolerance: Step-3.7-Flash's correct-sharding rel_rmse is ~0.05 — higher than the generic default
(0.02, calibrated on plain dense/MoE models) because the head-wise attention gate and the MoE
``routed_scaling_factor=3.0`` amplify the bf16 ``all_reduce`` summation-order rounding. That this is
genuine finite-precision noise and not a sharding bug is confirmed by the harness's sabotage control
(removing the collectives drives rel_rmse to ~0.9, i.e. ~17x). We therefore set the per-model
tolerance below.
"""

import sys
from functools import partial
from pathlib import Path

import pytest

# The generic harness and helpers live alongside this file / under _utils_test.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_sharding_num_correctness import (  # noqa: E402
    _DIST_CONFIGS,
    _gpu_check,
    _run_equivalence_job,
)

# A bare module name lets the harness resolve the model from either bundled
# AutoDeploy or the generated standalone package.
_MODELING_FILE = "modeling_step3p7"

# Per-model relative-RMSE tolerance (see module docstring for the rationale).
STEP3P7_REL_RMSE_TOL = 0.08

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.mark.parametrize("dist_config", list(_DIST_CONFIGS))
def test_step3p7_sharding_num_correctness(dist_config: str, monkeypatch) -> None:
    """Sharded == unsharded prefill for Step-3.7-Flash under each parallelism config."""
    skip = _gpu_check(dist_config)
    if skip:
        pytest.skip(skip)

    # The per-rank worker reads the tolerance from this env var (spawned workers inherit it).
    monkeypatch.setenv("SHARDING_IR_REL_RMSE_TOL", str(STEP3P7_REL_RMSE_TOL))

    import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common

    world_size = _DIST_CONFIGS[dist_config]["world_size"]
    dist_common.spawn_multiprocess_job(
        job=partial(_run_equivalence_job, _MODELING_FILE, dist_config, "none"),
        size=world_size,
    )
