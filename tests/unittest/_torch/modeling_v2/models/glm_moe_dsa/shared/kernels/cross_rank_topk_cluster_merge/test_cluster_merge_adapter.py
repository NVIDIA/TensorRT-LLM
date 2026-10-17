# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selector and call site of cross_rank_topk_cluster_merge.

- ``cross_rank_topk_cluster_merge`` runs only with ``cross_rank_topk`` (disabling ``cross_rank_topk``
  turns it off; disabling it keeps ``cross_rank_topk``) and is recorded as a
  cross_rank_topk-identical ``rank-consistent`` variant.
- The cross_rank_topk region of the indexer's decode top-k (``region.cross_rank_topk_decode``,
  which ``GlmIndexer._fill_decode_topk`` asks first) tries cross_rank_topk_cluster_merge
  first when ``cross_rank_topk_cluster_merge`` is selected and falls back to cross_rank_topk when
  cross_rank_topk_cluster_merge declines; without ``cross_rank_topk_cluster_merge`` it calls
  cross_rank_topk only (composition F unchanged), and without ``cross_rank_topk`` nothing.
- The cross_rank_topk_cluster_merge package registers its entry points and declines a device that
  cannot keep every row's cluster resident, and a group size its GPU tests
  do not cover.
"""

import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch


def _plan(disabled=()):
    environ = {dispatch.DISABLE_ENV: ",".join(disabled)} if disabled else {}
    return dispatch.resolve(dispatch.ModelingV2Mode.AUTO, environ)


def test_selector_closure():
    assert dispatch.REQUIRED_FEATURES["cross_rank_topk_cluster_merge"] == {"cross_rank_topk"}
    assert _plan().enabled("cross_rank_topk_cluster_merge")
    assert not _plan(["cross_rank_topk"]).enabled("cross_rank_topk_cluster_merge")
    record = dispatch.feature_record(
        ["cross_rank_topk", "cross_rank_topk_cluster_merge"], environ={}
    )
    assert (
        "cross_rank_topk_cluster_merge[rank-consistent]" in record
        and "cross_rank_topk[rank-consistent]" in record
    )


def test_disabling_cluster_merge_keeps_cross_rank_topk():
    plan = _plan(["cross_rank_topk_cluster_merge"])
    assert plan.enabled("cross_rank_topk")
    assert not plan.enabled("cross_rank_topk_cluster_merge")


class _Calls:
    def __init__(self, cluster_merge_result):
        self.cluster_merge_result = cluster_merge_result
        self.log = []

    def cross_rank_topk_cluster_merge(self, *args):
        self.log.append("cross_rank_topk_cluster_merge")
        return self.cluster_merge_result

    def indexer_topk(self, *args):
        self.log.append("indexer_topk")
        return True


@pytest.mark.parametrize(
    "features, cluster_merge_result, expected, calls",
    [
        (
            frozenset({"cross_rank_topk", "cross_rank_topk_cluster_merge"}),
            True,
            True,
            ["cross_rank_topk_cluster_merge"],
        ),
        (
            frozenset({"cross_rank_topk", "cross_rank_topk_cluster_merge"}),
            False,
            True,
            ["cross_rank_topk_cluster_merge", "indexer_topk"],
        ),
        (frozenset({"cross_rank_topk"}), True, True, ["indexer_topk"]),
        (frozenset(), True, False, []),
    ],
)
def test_indexer_hook(monkeypatch, features, cluster_merge_result, expected, calls):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        region,
    )

    rec = _Calls(cluster_merge_result)
    monkeypatch.setattr(dispatch, "enabled_features", lambda: features)
    impls = {
        ("cross_rank_topk_cluster_merge", "cp_indexer_decode"): rec.cross_rank_topk_cluster_merge,
        ("indexer_topk", "cp_indexer_decode"): rec.indexer_topk,
    }
    monkeypatch.setattr(dispatch, "get_impl", lambda fid, name: impls[(fid, name)])
    got = region.cross_rank_topk_decode(None, None, None, None, 0, 6, None)
    assert got is expected
    assert rec.log == calls


def test_package_registration_and_cluster_reason():
    impl = dispatch.get_impl("cross_rank_topk_cluster_merge", "cp_indexer_decode")
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_cluster_merge import (  # noqa: E501
        cp_workspace,
        try_cp_indexer_decode,
    )

    assert impl is try_cp_indexer_decode
    assert dispatch.get_impl("cross_rank_topk_cluster_merge", "cp_decode_topk") is not None
    info = [0] * 12
    info[cp_workspace.INFO_MAX_ROWS] = 8
    info[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] = 16
    assert cp_workspace.cluster_reason(info) == ""
    info[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] = 7
    assert "7 co-resident" in cp_workspace.cluster_reason(info)
    assert "not built" in cp_workspace.cluster_reason(None)


def test_world_size_domain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_cluster_merge import (  # noqa: E501
        cp_indexer_decode,
    )

    assert cp_indexer_decode.SUPPORTED_WORLD_SIZES == (4, 8)
    for world in (4, 8):
        assert cp_indexer_decode.world_size_reason(world) == ""
    for world in (1, 2, 3, 5, 6, 7, 16):
        assert (
            "not in cross_rank_topk_cluster_merge's verified sizes"
            in cp_indexer_decode.world_size_reason(world)
        )


def test_op_info_when_built():
    import torch

    import tensorrt_llm  # noqa: F401  (the trtllm op namespace)
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_cluster_merge import (  # noqa: E501
        cp_workspace,
    )

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    library = os.environ.get("GLM_MOE_DSA_CROSS_RANK_TOPK_CLUSTER_MERGE_JIT_LIB", "")
    if not hasattr(
        torch.ops.trtllm, "glm_moe_dsa_cross_rank_topk_cluster_merge_workspace_info"
    ) and os.path.exists(library):
        # A JIT build of the op on an image without it.
        torch.ops.load_library(library)
    info = cp_workspace.op_info(8)
    if info is None:
        pytest.skip(
            "glm_moe_dsa_cross_rank_topk_cluster_merge is not built into this image (no development build given)"
        )
    assert info[cp_workspace.INFO_STATE_WORDS] == cp_workspace.EPOCH_BASE + 64
    assert info[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] >= info[cp_workspace.INFO_MAX_ROWS]
    # Cached per group size and device; callers get their own copy.
    again = cp_workspace.op_info(8)
    assert again == info and again is not info
    info4 = cp_workspace.op_info(4)
    assert info4[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] >= info4[cp_workspace.INFO_MAX_ROWS]
