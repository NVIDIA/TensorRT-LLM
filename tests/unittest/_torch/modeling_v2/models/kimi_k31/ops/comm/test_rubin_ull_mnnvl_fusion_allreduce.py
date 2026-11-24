# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's mnnvl_fusion_allreduce entry: its 4-rank matrix
(``_mnnvl_fusion_allreduce_matrix.py``, run as its own job; see ``_k31_ull_comm_ranks``)."""

import pytest
import torch

__extra_import_path__ = ["."]
import _k31_ull_comm_ranks as ranks  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or len(ranks.visible_devices()) < ranks.WORLD_SIZE,
    reason=f"the MNNVL all-reduce matrix needs {ranks.WORLD_SIZE} SM10x GPUs on one NVLink domain",
)


# The matrix spawns 4 ranks over the visible devices; xdist workers would fight over them.
@pytest.mark.no_xdist
def test_certified_cells_on_one_tray() -> None:
    ranks.run("mnnvl_fusion_allreduce")
