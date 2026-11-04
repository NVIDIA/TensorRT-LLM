# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the qwen4_exp FP8 target's collective certification matrices.

Each matrix is `_<entry>_op_matrix.py` beside this file, its own launcher, for one of the
target's collective entries (`ops/comm/`):

- `allreduce__tp_sum`: the `tp_sum` call at TP 2 (tp2, tep2) and 4 (tp4, tep4), every check
  over torch.distributed (the `allreduce_pg` symbol) and then under mpirun (the plain
  `allreduce` symbol a served engine calls);
- `allgather__dp_gather`: the attention-DP gather at each group size the target serves (the
  matrix's own `CERTIFIED_WORLDS`, dep2 and dep4), over torch.distributed (the `_pg` symbol)
  and then under mpirun (the MPI-session symbol the target calls when it serves);
- `reducescatter__dp_scatter`: the scatter half of the attention-DP round trip at dep2 and
  dep4, under mpirun, because the call it certifies is the MPI-session symbol a served engine
  makes.

`_rank_job` (in the shared comm test directory) starts a matrix in a fresh interpreter over
the first `world` visible devices and turns its exit code into an assertion. A job needs
`world` sm_103 GPUs in one node, and skips on fewer.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

_HERE = Path(__file__).resolve().parent

#: The group sizes the target runs each entry at. The gather's are read off its launcher,
#: which keys its cells by them, without importing tensorrt_llm.
_WORLDS = {
    "allreduce__tp_sum": (2, 4),
    "allgather__dp_gather": _rank_job._load_launcher_constants(
        _HERE / "_allgather__dp_gather_op_matrix.py"
    ).CERTIFIED_WORLDS,
    "reducescatter__dp_scatter": (2, 4),
}
_CASES = [(entry, world) for entry, worlds in _WORLDS.items() for world in worlds]


def _sm103_devices(world: int) -> int:
    """How many of the devices a `world`-rank job would claim are sm_103."""
    if not torch.cuda.is_available():
        return 0
    claimed = range(min(torch.cuda.device_count(), world))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


# Each case spawns its own ranks over the visible devices. Under xdist several workers would
# fight for the same GPUs, so this must run alone, like the shared collective entries.
@pytest.mark.no_xdist
@pytest.mark.parametrize("entry, world", _CASES, ids=[f"{e}-world{w}" for e, w in _CASES])
def test_op_matrix(entry: str, world: int) -> None:
    if _sm103_devices(world) < world:
        pytest.skip(f"the {entry} matrix at world {world} runs on {world} sm_103 GPUs in one node")
    _rank_job.run(entry, world_size=world, directory=_HERE)
