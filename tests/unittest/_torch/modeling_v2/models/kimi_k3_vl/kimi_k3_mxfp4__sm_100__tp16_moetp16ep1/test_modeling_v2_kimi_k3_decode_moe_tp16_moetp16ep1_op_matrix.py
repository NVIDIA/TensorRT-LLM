# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the MoE decode path of the Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp16ep1``
(``decode_moe.py``: every expert on each rank, the routed experts pushing into the latent exchange).

The checks are ``_kimi_k3_decode_moe_tp16_moetp16ep1_op_matrix.py`` beside this file, its own W-rank launcher (the
layers share the path's collective state, so one job, not independent cases); see ``_rank_job`` for why that is left
intact.
"""

from pathlib import Path
from typing import TYPE_CHECKING

__extra_import_path__ = ["../../../comm"]
if TYPE_CHECKING:
    from _torch.modeling_v2.comm import _rank_job
else:
    import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "the Kimi K3 target's MoE decode path requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    # Runs on sm_100 and sm_103, the architectures the K3 decode kernels are certified on; others skip.
    pytest.skip("K3 decode-path kernels require SM100 or SM103", allow_module_level=True)

if torch.cuda.device_count() < _rank_job.WORLD_SIZE:
    # One rank per device: fewer visible devices cannot host the check's world size.
    pytest.skip(
        f"the check runs {_rank_job.WORLD_SIZE} ranks, one per device; "
        f"{torch.cuda.device_count()} visible",
        allow_module_level=True,
    )


# The case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_kimi_k3_decode_moe_tp16_moetp16ep1_op_matrix() -> None:
    _rank_job.run("kimi_k3_decode_moe_tp16_moetp16ep1", directory=Path(__file__).resolve().parent)
