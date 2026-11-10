# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the FP8 target's collective matrices: each `_<entry>_op_matrix.py`
beside this file is its own launcher, which `_rank_job` runs in a fresh interpreter on `world`
sm_103 GPUs of one node."""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

_HERE = Path(__file__).resolve().parent

#: The group sizes the target runs each entry at (each launcher's `CERTIFIED_WORLDS`, restated
#: so that collecting this module on a machine without GPUs loads no launcher).
_WORLDS = {
    "mnnvl_allreduce__tp_sum": (2, 4),
    "ce_exchange__sp_pair": (2,),
    "lamport_allgather__index_split": (2, 4),
    "hc_ar_inject_norm": (2, 4),
    "moe_a2a__dp_dispatch_combine": (2, 4),
}
_CASES = [(entry, world) for entry, worlds in _WORLDS.items() for world in worlds]


def _sm103_devices(world: int) -> int:
    """How many of the devices a `world`-rank job would claim are sm_103."""
    if not torch.cuda.is_available():
        return 0
    claimed = range(min(torch.cuda.device_count(), world))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


# Each case spawns its own ranks over the visible devices, so xdist workers would share GPUs.
@pytest.mark.no_xdist
@pytest.mark.parametrize("entry, world", _CASES, ids=[f"{e}-world{w}" for e, w in _CASES])
def test_op_matrix(entry: str, world: int) -> None:
    if _sm103_devices(world) < world:
        pytest.skip(f"the {entry} matrix at world {world} runs on {world} sm_103 GPUs in one node")
    _rank_job.run(entry, world_size=world, directory=_HERE)
