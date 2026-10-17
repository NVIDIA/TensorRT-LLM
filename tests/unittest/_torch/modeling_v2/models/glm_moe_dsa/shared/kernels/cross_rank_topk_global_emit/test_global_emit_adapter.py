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
"""Selector, call sites and bridge plan of cross_rank_topk_global_emit.

- ``cross_rank_topk_global_emit`` runs only with ``rope_kv_append``, ``cross_rank_topk``
  and ``cross_rank_topk_cluster_merge`` and is recorded as a ``rank-consistent`` variant; disabling it
  keeps the paths it extends.
- The cross_rank_topk region of the indexer's decode top-k (``region.cross_rank_topk_decode``)
  tries cross_rank_topk_global_emit first (with the call's rope_kv_append packet, which
  ``GlmIndexer._fill_decode_topk`` passes) when ``cross_rank_topk_global_emit`` is selected,
  then cross_rank_topk_cluster_merge, then cross_rank_topk; without ``cross_rank_topk_global_emit``
  it is B5's hook.
- The backend's packet and remap (``GlmDsaBackend.sparse_attn_predict``) come from
  cross_rank_topk_global_emit when it is selected.
- ``bridge.predict_copy`` / ``predict_remap`` reproduce the call sites' rules
  (rope_kv_append's stash, the backend's shared copy, rope_kv_append's grouping), and
  ``remap_topk_to_global`` returns the planned pool rows, falls back to rope_kv_append's
  remap without a plan, and raises on any difference from the plan.
- The package registers its entry points; group sizes and op info.
- deferred_input_prep's tables list cross_rank_topk_global_emit's package directories and its side
  selector.
"""

import inspect
import os
import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch


def _plan(disabled=()):
    environ = {dispatch.DISABLE_ENV: ",".join(disabled)} if disabled else {}
    return dispatch.resolve(dispatch.ModelingV2Mode.AUTO, environ)


def test_selector_closure():
    assert dispatch.REQUIRED_FEATURES["cross_rank_topk_global_emit"] == {
        "rope_kv_append",
        "cross_rank_topk",
        "cross_rank_topk_cluster_merge",
    }
    assert _plan().enabled("cross_rank_topk_global_emit")
    for base in ("rope_kv_append", "cross_rank_topk", "cross_rank_topk_cluster_merge"):
        assert not _plan([base]).enabled("cross_rank_topk_global_emit"), base
    record = dispatch.feature_record(
        [
            "rope_kv_append",
            "cross_rank_topk",
            "cross_rank_topk_cluster_merge",
            "cross_rank_topk_global_emit",
        ],
        environ={},
    )
    assert (
        "cross_rank_topk_global_emit[rank-consistent]" in record
        and "cross_rank_topk_cluster_merge[rank-consistent]" in record
    )


def test_disabling_global_emit_keeps_cluster_merge():
    plan = _plan(["cross_rank_topk_global_emit"])
    assert not plan.enabled("cross_rank_topk_global_emit")
    for base in ("rope_kv_append", "cross_rank_topk", "cross_rank_topk_cluster_merge"):
        assert plan.enabled(base), base


class _Calls:
    def __init__(self, global_emit_result, cluster_merge_result):
        self.global_emit_result = global_emit_result
        self.cluster_merge_result = cluster_merge_result
        self.log = []
        self.packets = []

    def cross_rank_topk_global_emit(self, *args):
        self.log.append("cross_rank_topk_global_emit")
        self.packets.append(args[-1])
        assert len(args) == 8
        return self.global_emit_result

    def cross_rank_topk_cluster_merge(self, *args):
        self.log.append("cross_rank_topk_cluster_merge")
        assert len(args) == 7
        return self.cluster_merge_result

    def indexer_topk(self, *args):
        self.log.append("indexer_topk")
        return True


@pytest.mark.parametrize(
    "features, global_emit_result, cluster_merge_result, calls",
    [
        (
            frozenset(
                {
                    "rope_kv_append",
                    "cross_rank_topk",
                    "cross_rank_topk_cluster_merge",
                    "cross_rank_topk_global_emit",
                }
            ),
            True,
            True,
            ["cross_rank_topk_global_emit"],
        ),
        (
            frozenset(
                {
                    "rope_kv_append",
                    "cross_rank_topk",
                    "cross_rank_topk_cluster_merge",
                    "cross_rank_topk_global_emit",
                }
            ),
            False,
            True,
            ["cross_rank_topk_global_emit", "cross_rank_topk_cluster_merge"],
        ),
        (
            frozenset(
                {
                    "rope_kv_append",
                    "cross_rank_topk",
                    "cross_rank_topk_cluster_merge",
                    "cross_rank_topk_global_emit",
                }
            ),
            False,
            False,
            ["cross_rank_topk_global_emit", "cross_rank_topk_cluster_merge", "indexer_topk"],
        ),
        (
            frozenset({"rope_kv_append", "cross_rank_topk", "cross_rank_topk_cluster_merge"}),
            True,
            True,
            ["cross_rank_topk_cluster_merge"],
        ),
        (frozenset({"rope_kv_append", "cross_rank_topk"}), True, True, ["indexer_topk"]),
        (frozenset(), True, True, []),
    ],
)
def test_indexer_hook(monkeypatch, features, global_emit_result, cluster_merge_result, calls):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        region,
    )

    rec = _Calls(global_emit_result, cluster_merge_result)
    monkeypatch.setattr(dispatch, "enabled_features", lambda: features)
    impls = {
        ("cross_rank_topk_global_emit", "cp_indexer_decode"): rec.cross_rank_topk_global_emit,
        ("cross_rank_topk_cluster_merge", "cp_indexer_decode"): rec.cross_rank_topk_cluster_merge,
        ("indexer_topk", "cp_indexer_decode"): rec.indexer_topk,
    }
    monkeypatch.setattr(dispatch, "get_impl", lambda fid, name: impls[(fid, name)])
    packet = object()
    got = region.cross_rank_topk_decode(None, None, None, None, 0, 6, None, packet)
    # Each tried region but the last declined; the last one served the call.
    assert got is bool(calls)
    assert rec.log == calls
    assert all(p is packet for p in rec.packets)


def test_indexer_call_site_passes_the_packet():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        GlmIndexer,
    )

    src = inspect.getsource(GlmIndexer._fill_decode_topk)
    call = src[src.index("region.cross_rank_topk_decode(") :]
    call = call[: call.index(")\n")]
    assert "self._topk_bridge" in call


def test_backend_uses_global_emit_packet_and_remap_only_when_selected():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        GlmDsaBackend,
    )

    src = inspect.getsource(GlmDsaBackend.sparse_attn_predict)
    # The formatter wraps long statements: compare with the whitespace collapsed.
    src = re.sub(r"(?<=[(\[])\s+|\s+(?=[)\]])", "", " ".join(src.split()))
    assert (
        'bridge_path = ("cross_rank_topk_global_emit" if '
        'dispatch.is_enabled("cross_rank_topk_global_emit") else "rope_kv_append")' in src
    )
    assert 'get_impl(bridge_path, "new_bridge_packet")' in src
    assert 'bridge_path, "remap_topk_to_global"' in src
    # Every other rope_kv_append call site of the backend is rope_kv_append's own.
    assert src.count('get_impl("rope_kv_append", "defer_shared_copy")') == 1


def test_package_registration():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        cross_rank_topk_global_emit,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
        cp_indexer_decode,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        ops as rope_ops,
    )

    assert (
        dispatch.get_impl("cross_rank_topk_global_emit", "cp_indexer_decode")
        is cp_indexer_decode.try_cp_indexer_decode
    )
    assert (
        dispatch.get_impl("cross_rank_topk_global_emit", "new_bridge_packet")
        is bridge.new_bridge_packet
    )
    assert (
        dispatch.get_impl("cross_rank_topk_global_emit", "remap_topk_to_global")
        is bridge.remap_topk_to_global
    )
    assert issubclass(cross_rank_topk_global_emit.GlobalEmitBridgePacket, rope_ops.BridgePacket)
    assert cross_rank_topk_global_emit.GlobalEmitBridgePacket().plan is None


def test_world_size_domain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_indexer_decode,
    )

    assert cp_indexer_decode.SUPPORTED_WORLD_SIZES == (4, 8)
    for world in (4, 8):
        assert cp_indexer_decode.world_size_reason(world) == ""
    for world in (1, 2, 3, 5, 6, 7, 16):
        assert (
            "not in cross_rank_topk_global_emit's verified sizes"
            in cp_indexer_decode.world_size_reason(world)
        )


# ---- Bridge plan ------------------------------------------------------------------------------------------------


class _Kvm:
    def __init__(self, tokens_per_block=64):
        self.tokens_per_block = tokens_per_block
        self.layer_offsets = {layer: layer for layer in range(100)}

    def get_primary_pool_page_index_params(self, local_layer):
        return 79, local_layer


def _struct(group_ids):
    """Group struct of rope_kv_append's grouping: layers 0, 1 singletons, then groups of len(group_ids)."""
    n = 2 + 2 * len(group_ids)
    leader_of = [0, 1] + [2] * len(group_ids) + [2 + len(group_ids)] * len(group_ids)
    slot_of = [0, 0] + list(range(len(group_ids))) * 2
    return {
        "leader_of": leader_of[:n],
        "slot_of": slot_of[:n],
        "group_active": {0: False, 1: False, 2: True, 2 + len(group_ids): True},
        "group_scale": {2: 79, 2 + len(group_ids): 79},
        "group_layer_ids": {
            2: torch.tensor(group_ids, dtype=torch.int32),
            2 + len(group_ids): torch.tensor(group_ids, dtype=torch.int32),
        },
    }


def _metadata(*, draft=False, shared=True, accepted=torch.int32, group=4, rows=6, struct=None):
    st = _struct(list(range(group))) if struct is None else struct
    return SimpleNamespace(
        num_contexts=0,
        num_generations=1,
        num_ctx_tokens=0,
        num_tokens=rows,
        in_mtp_draft_loop=draft,
        shared_topk_indices=torch.zeros(8, 2048, dtype=torch.int32) if shared else None,
        mtp_num_accepted=None if accepted is None else torch.tensor([3, 1], dtype=accepted),
        kv_cache_manager=_Kvm(),
        _ensure_group_remap_struct=lambda: st,
        _group_remap_batched={},
    )


def _indexer(share=True, layer=2):
    return SimpleNamespace(
        mtp_index_share=share, layer_idx=layer, top_k=SimpleNamespace(top_k=2048)
    )


def test_predict_copy_rules():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )

    # Target layer: the backend's shared copy of the generation rows.
    md = _metadata()
    c = bridge.predict_copy(_indexer(), md, 6)
    assert (c.dst is md.shared_topk_indices, c.row_start, c.rows, c.mode) == (True, 0, 6, 0)
    assert bridge.predict_copy(_indexer(), _metadata(shared=False), 6) is None
    # MTP draft step 0 with index sharing: rope_kv_append's stash (mode 1, the request's accepted
    # count).
    md = _metadata(draft=True)
    c = bridge.predict_copy(_indexer(), md, 6)
    assert (c.dst is md.shared_topk_indices, c.row_start, c.rows, c.mode, c.next_n) == (
        True,
        0,
        1,
        1,
        6,
    )
    assert c.num_accepted.dtype == torch.int32 and c.num_accepted.tolist() == [3]
    c = bridge.predict_copy(_indexer(), _metadata(draft=True, accepted=None), 6)
    assert c.mode == 1 and c.num_accepted is None
    c = bridge.predict_copy(_indexer(), _metadata(draft=True, accepted=torch.int64), 6)
    assert c.num_accepted.dtype == torch.int64
    # The stash is not deferred (rope_kv_append declines): the native stash runs and the backend
    # keeps the rows.
    assert bridge.predict_copy(_indexer(), _metadata(draft=True, shared=False), 6) is None
    assert bridge.predict_copy(_indexer(), _metadata(draft=True, accepted=torch.float32), 6) is None
    # A draft step without index sharing takes the backend's shared copy.
    c = bridge.predict_copy(_indexer(share=False), _metadata(draft=True), 6)
    assert c.mode == 0 and c.rows == 6


def test_predict_remap_rules(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    monkeypatch.setattr(dsa_backend, "_GROUP_REMAP", True)
    md = _metadata(group=4)
    r = bridge.predict_remap(md, 2)
    assert (r.grouped, r.leader, r.slot, r.page_index_scale) == (True, 2, 0, 79)
    assert r.layer_ids is md._ensure_group_remap_struct()["group_layer_ids"][2]
    r = bridge.predict_remap(md, 1)
    assert (r.grouped, r.page_index_scale, r.layer_offset) == (False, 79, 1)
    assert bridge.predict_remap(_metadata(group=9), 2) is None
    r = bridge.predict_remap(_metadata(draft=True), 2)
    assert not r.grouped and r.layer_offset == 2
    r = bridge.predict_remap(_metadata(struct={}), 2)
    assert not r.grouped
    monkeypatch.setattr(dsa_backend, "_GROUP_REMAP", False)
    assert not bridge.predict_remap(_metadata(), 2).grouped


def _backend(layer=2, indexer=True):
    return SimpleNamespace(
        indexer=object() if indexer else None, get_local_layer_idx=lambda metadata: layer
    )


def _planned(md, layer=2, grouped=True):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )

    ids = torch.zeros(6, 2048, dtype=torch.int32)
    remap = bridge.predict_remap(md, layer)
    copy = bridge.predict_copy(_indexer(layer=layer), md, 6)
    pool = (
        torch.arange(4 * 6 * 2048, dtype=torch.int32).view(4, 6, 2048)
        if remap.grouped
        else torch.ones(6, 2048, dtype=torch.int32)
    )
    packet = bridge.GlobalEmitBridgePacket()
    packet.plan = bridge.GlobalEmitPlan(ids=ids, remap=remap, copy=copy, pool=pool)
    return packet, ids, pool, copy


def test_remap_uses_the_plan_once(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        ops as rope_ops,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    monkeypatch.setattr(dsa_backend, "_GROUP_REMAP", True)
    rope_append_calls = []
    monkeypatch.setattr(
        rope_ops, "remap_topk_to_global", lambda *a: rope_append_calls.append(a) or "rope_kv_append"
    )
    md = _metadata()
    packet, ids, pool, copy = _planned(md, layer=3)
    rope_ops.defer_shared_copy(packet, md.shared_topk_indices, 0, ids)
    out = bridge.remap_topk_to_global(_backend(layer=3), ids, md, packet)
    assert out.data_ptr() == pool[1].data_ptr() and md._group_remap_batched[2] is pool
    assert packet.plan is None and not rope_append_calls
    # A packet without a plan (draft reuse, declines) is rope_kv_append's remap.
    assert (
        bridge.remap_topk_to_global(_backend(layer=3), ids, md, packet) == "rope_kv_append"
        and len(rope_append_calls) == 1
    )
    # Per-layer plan.
    packet, ids, pool, copy = _planned(md, layer=1)
    rope_ops.defer_shared_copy(packet, md.shared_topk_indices, 0, ids)
    assert bridge.remap_topk_to_global(_backend(layer=1), ids, md, packet) is pool


def test_remap_rejects_differences(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        ops as rope_ops,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    monkeypatch.setattr(dsa_backend, "_GROUP_REMAP", True)

    def check(mutate, match):
        md = _metadata()
        packet, ids, pool, copy = _planned(md, layer=2)
        rope_ops.defer_shared_copy(packet, md.shared_topk_indices, 0, ids)
        args = mutate(md, packet, ids)
        with pytest.raises(bridge.GlobalEmitContractError, match=match):
            bridge.remap_topk_to_global(*args)

    check(lambda md, p, ids: (_backend(), ids, md, setattr(p, "copy", None) or p), "copy")
    check(lambda md, p, ids: (_backend(), ids, md, setattr(p.copy, "rows", 5) or p), "copy")
    check(lambda md, p, ids: (_backend(), ids, md, setattr(p, "source", ids) or p), "source")
    check(
        lambda md, p, ids: (_backend(), ids.clone(), md, p),
        "rows cross_rank_topk_global_emit wrote",
    )
    check(lambda md, p, ids: (_backend(layer=3), ids, md, p), "local layer")
    check(lambda md, p, ids: (_backend(indexer=False), ids, md, p), "no indexer")

    def regroup(md, p, ids):
        md._ensure_group_remap_struct = lambda: {}
        return _backend(), ids, md, p

    check(regroup, "remap decision")


def test_copy_reason_matches_the_bridge_domain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        ops as rope_ops,
    )

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    dst = torch.zeros(8, 2048, dtype=torch.int32, device="cuda")
    assert bridge.copy_reason(None, 6, 2048) == ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 0, 6), 6, 2048) == ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 3, 6), 6, 2048) != ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 0, 7), 6, 2048) != ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst.cpu(), 0, 6), 6, 2048) != ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst[:, :100], 0, 6), 6, 2048) != ""
    acc = torch.tensor([2], dtype=torch.int32, device="cuda")
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 0, 1, 1, 6, acc), 6, 2048) == ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 0, 2, 1, 6, None), 6, 2048) != ""
    assert bridge.copy_reason(rope_ops.BridgeCopy(dst, 0, 1, 1, 6, acc.float()), 6, 2048) != ""


def test_bridge_reason_declines_before_device_work(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge,
        cp_indexer_decode,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        ops as rope_ops,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    monkeypatch.setattr(dsa_backend, "_GROUP_REMAP", True)
    reason, _, _ = cp_indexer_decode.bridge_reason(
        _indexer(), _metadata(), rope_ops.BridgePacket(), 6
    )
    assert "no cross_rank_topk_global_emit bridge packet" in reason
    reason, _, _ = cp_indexer_decode.bridge_reason(_indexer(), _metadata(), None, 6)
    assert "no cross_rank_topk_global_emit bridge packet" in reason
    md = _metadata()
    md.kv_cache_manager.tokens_per_block = 32
    reason, _, _ = cp_indexer_decode.bridge_reason(
        _indexer(), md, bridge.GlobalEmitBridgePacket(), 6
    )
    assert "block size" in reason
    reason, _, _ = cp_indexer_decode.bridge_reason(
        _indexer(), _metadata(group=9), bridge.GlobalEmitBridgePacket(), 6
    )
    assert "group" in reason
    used = bridge.GlobalEmitBridgePacket()
    used.copy = rope_ops.BridgeCopy(torch.zeros(1, 1), 0, 1)
    reason, _, _ = cp_indexer_decode.bridge_reason(_indexer(), _metadata(), used, 6)
    assert "already holds" in reason


def test_op_info_when_built():
    import tensorrt_llm  # noqa: F401  (the trtllm op namespace)
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_workspace,
    )

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    library = os.environ.get("GLM_MOE_DSA_CROSS_RANK_TOPK_GLOBAL_EMIT_JIT_LIB", "")
    if not hasattr(
        torch.ops.trtllm, "glm_moe_dsa_cross_rank_topk_global_emit_workspace_info"
    ) and os.path.exists(library):
        # A JIT build of the op on an image without it.
        torch.ops.load_library(library)
    info = cp_workspace.op_info(8)
    if info is None:
        pytest.skip(
            "glm_moe_dsa_cross_rank_topk_global_emit is not built into this image (no development build given)"
        )
    assert info[cp_workspace.INFO_STATE_WORDS] == cp_workspace.EPOCH_BASE + 64
    assert info[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] >= info[cp_workspace.INFO_MAX_ROWS]
    assert info[cp_workspace.INFO_MAX_GROUP_SIZE] == 8
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk.cross_rank_logits_runner import (  # noqa: E501
        local_capacity,
    )

    for world in (4, 8):
        winfo = cp_workspace.op_info(world)
        assert winfo[cp_workspace.INFO_MAX_LOCAL_CAPACITY] >= local_capacity((1 << 20) + 14, world)
        assert winfo[cp_workspace.INFO_MAX_ACTIVE_CLUSTERS] >= winfo[cp_workspace.INFO_MAX_ROWS]
