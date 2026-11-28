# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the megamoe_workspace entry (`megamoe_token_buckets`) and its helpers: the bucket ladder, the
shared-workspace probe and the local buffers. The symmetric rendezvous itself needs 4 ranks and is exercised by
`test_modeling_v2_cute_dsl_megamoe_nvfp4.py`."""

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_workspace import (  # noqa: E402
    megamoe_pick_bucket,
    megamoe_shared_workspace_bytes,
    megamoe_token_buckets,
    megamoe_topk_idx_buffer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize(
    "cell", megamoe_token_buckets.CELLS, ids=[c.why[:44] for c in megamoe_token_buckets.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(megamoe_token_buckets)
    args = (cell.spec["max_num_tokens"], 103)
    assert megamoe_token_buckets(*args) == list(cell.spec["expected"])
    assert megamoe_token_buckets.reference(*args) == list(cell.spec["expected"])


def test_the_ladder_matches_the_engines():
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl

    for n in (1, 32, 150, 256, 300, 1000, 1280, 4096, 5000):
        assert megamoe_token_buckets(n, 103) == MegaMoECuteDsl._resolve_maxt_buckets(n)


def test_pick_bucket_is_the_smallest_fit_and_raises_past_the_top():
    ladder = [256, 1024, 1280]
    assert [megamoe_pick_bucket(ladder, n) for n in (0, 1, 256, 257, 1024, 1025, 1280)] == [
        256, 256, 256, 1024, 1024, 1280, 1280,
    ]  # fmt: skip
    with pytest.raises(RuntimeError):
        megamoe_pick_bucket(ladder, 1281)


def test_topk_idx_buffer_is_idle_rows():
    buf = megamoe_topk_idx_buffer(150, 6, "cuda")
    assert buf.shape == (150, 6) and buf.dtype == torch.int64 and (buf == -1).all()


def test_shared_workspace_probe_depends_on_the_launch_problem():
    """A workspace sized for the wrong problem is written out of bounds: the probe must see every
    argument that changes the kernel."""
    kw = dict(
        world_size=4, num_topk=6, num_experts_per_rank=34, hidden_size=5120,
        intermediate_size=2304, gate_up_clamp=10.0,
    )  # fmt: skip
    small, _ = megamoe_shared_workspace_bytes(max_tokens_per_rank=256, **kw)
    large, _ = megamoe_shared_workspace_bytes(max_tokens_per_rank=1024, **kw)
    lowp, _ = megamoe_shared_workspace_bytes(
        max_tokens_per_rank=256, combine_format="32e4m3xe8m0", **kw
    )
    assert small > 0 and large > small, "the workspace grows with the bucket"
    assert lowp != small, "the combine wire format changes the workspace"


def test_rejections():
    megamoe_token_buckets.is_valid(150, 103)
    with pytest.raises(AssertionError, match="negative"):
        megamoe_token_buckets.is_valid(-1, 103)
    with pytest.raises(AssertionError, match="sm_103"):
        megamoe_token_buckets.is_valid(150, 100)
