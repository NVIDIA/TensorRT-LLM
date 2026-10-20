# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the plain-sum op the Nemotron-3 Ultra TP
targets call, cell by cell.

The matrix itself is ``_allreduce_op_matrix.py`` beside this file, which is its
own launcher; see ``_rank_job`` for why that is left intact. Unlike its two
siblings it is rank-count agnostic: it runs at eight ranks where the node has
eight devices -- the tep8 target's own world size -- and at four otherwise.
"""

import _rank_job
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import Allreduce

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_ARCH = "sm_{}{}".format(*torch.cuda.get_device_capability())
_CERTIFIED = sorted(arch.value for arch in Allreduce.ARCHS)


# Each case spawns its own rank processes over the visible devices. Under xdist
# several workers would fight for the same GPUs, so this must run alone -- the
# same reason the other collective entries in this directory carry the marker.
@pytest.mark.no_xdist
@pytest.mark.skipif(
    _ARCH not in _CERTIFIED,
    reason=(
        f"allreduce is certified on {', '.join(_CERTIFIED)}, not {_ARCH}: its AUTO cells name "
        f"the kernel the op's strategy table picks, which nobody has measured here. Certify an "
        f"architecture by running _allreduce_op_matrix.py on it, then add it to ARCHS"
    ),
)
def test_allreduce_op_matrix() -> None:
    _rank_job.run("allreduce", world_sizes=(8, 4))
