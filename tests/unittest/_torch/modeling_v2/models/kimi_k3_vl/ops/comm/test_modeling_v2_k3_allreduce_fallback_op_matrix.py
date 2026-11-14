# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for ``K3AllReduce`` against the built-in ``AllReduce`` off MNNVL.

The checks are ``_k3_allreduce_fallback_op_matrix.py`` beside this file, its own W-rank launcher (both modules
share the process's autotuner cache and NCCL window pool, so one job, not independent cases); see ``_rank_job``
for why that is left intact.
"""

from pathlib import Path

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "the all-reduce requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("the Kimi K3 targets run on sm_100 and sm_103", allow_module_level=True)

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


def _mnnvl_available() -> bool:
    """The condition the modules take MNNVL on with the MNNVL strategy named (the weaker of the two strategies the
    check builds): where it holds, the check's premise does not."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.mnnvl_workspace import (
        mnnvl_supported,
    )
    from tensorrt_llm.mapping import Mapping

    world = _rank_job.WORLD_SIZE
    mapping = Mapping(world_size=world, rank=0, gpus_per_node=world, tp_size=world)
    return mnnvl_supported(mapping, torch.bfloat16, explicitly_requested=True)


if _mnnvl_available():
    pytest.skip(
        "the check is for a TP group without MNNVL (x86 B200 / B300 NVLink); this host supports MNNVL, so the "
        "modules take it (the MNNVL path is _k3_allreduce_op_matrix.py)",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_k3_allreduce_fallback_op_matrix() -> None:
    _rank_job.run("k3_allreduce_fallback", directory=Path(__file__).resolve().parent)
