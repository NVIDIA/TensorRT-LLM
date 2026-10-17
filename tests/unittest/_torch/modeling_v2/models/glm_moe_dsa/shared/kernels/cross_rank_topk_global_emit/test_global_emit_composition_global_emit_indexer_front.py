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
"""cross_rank_topk_global_emit and indexer_layer_front composed in the cross_rank_topk CP indexer
region (host-side checks).

cross_rank_topk_global_emit forks the region's top-k (its emission also writes rope_kv_append's
bridge outputs) and indexer_layer_front forks the region's CP logits launch (entry wait, PDL
launch). With both selected, cross_rank_topk_global_emit serves the region and takes
indexer_layer_front's logits launcher; the indexer hook tries cross_rank_topk_global_emit, then
indexer_layer_front (cross_rank_topk_cluster_merge's or cross_rank_topk's top-k), then
cross_rank_topk_cluster_merge, then cross_rank_topk. These tests mock every device step: they check
which entry serves a call, which logits launcher cross_rank_topk_global_emit uses, and the
capability lines; the numerics are covered by the GPU tests of each package.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch


class _Calls:
    def __init__(self, results):
        self.results = results
        self.log = []

    def make(self, name, nargs):
        def entry(*args):
            self.log.append(name)
            assert len(args) == nargs, (name, len(args))
            return self.results[name]

        return entry


_ALL = {
    "cross_rank_topk_global_emit": True,
    "indexer_layer_front": True,
    "cross_rank_topk_cluster_merge": True,
    "indexer_topk": True,
}


@pytest.mark.parametrize(
    "features, results, calls",
    [
        # F4: cross_rank_topk_global_emit serves (with indexer_layer_front's logits).
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
            },
            _ALL,
            ["cross_rank_topk_global_emit"],
        ),
        # cross_rank_topk_global_emit declines: indexer_layer_front's region
        # (cross_rank_topk_cluster_merge's top-k) serves.
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
            },
            {**_ALL, "cross_rank_topk_global_emit": False},
            ["cross_rank_topk_global_emit", "indexer_layer_front"],
        ),
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
            },
            {**_ALL, "cross_rank_topk_global_emit": False, "indexer_layer_front": False},
            ["cross_rank_topk_global_emit", "indexer_layer_front", "cross_rank_topk_cluster_merge"],
        ),
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
            },
            {
                "cross_rank_topk_global_emit": False,
                "indexer_layer_front": False,
                "cross_rank_topk_cluster_merge": False,
                "indexer_topk": True,
            },
            [
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
                "cross_rank_topk_cluster_merge",
                "indexer_topk",
            ],
        ),
        # F4 - cross_rank_topk_global_emit: indexer_layer_front first.
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "indexer_layer_front",
            },
            _ALL,
            ["indexer_layer_front"],
        ),
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "indexer_layer_front",
            },
            {**_ALL, "indexer_layer_front": False},
            ["indexer_layer_front", "cross_rank_topk_cluster_merge"],
        ),
        # Without indexer_layer_front: cross_rank_topk_global_emit first, then
        # cross_rank_topk_cluster_merge.
        (
            {
                "rope_kv_append",
                "cross_rank_topk",
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
            },
            {**_ALL, "cross_rank_topk_global_emit": False},
            ["cross_rank_topk_global_emit", "cross_rank_topk_cluster_merge"],
        ),
        # cross_rank_topk_cluster_merge only.
        (
            {"rope_kv_append", "cross_rank_topk", "cross_rank_topk_cluster_merge"},
            _ALL,
            ["cross_rank_topk_cluster_merge"],
        ),
        (set(), _ALL, []),
    ],
)
def test_indexer_hook_order(monkeypatch, features, results, calls):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        region,
    )

    rec = _Calls(results)
    impls = {
        ("cross_rank_topk_global_emit", "cp_indexer_decode"): rec.make(
            "cross_rank_topk_global_emit", 8
        ),
        ("indexer_layer_front", "cp_indexer_decode"): rec.make("indexer_layer_front", 7),
        ("cross_rank_topk_cluster_merge", "cp_indexer_decode"): rec.make(
            "cross_rank_topk_cluster_merge", 7
        ),
        ("indexer_topk", "cp_indexer_decode"): rec.make("indexer_topk", 7),
    }
    monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset(features))
    monkeypatch.setattr(dispatch, "get_impl", lambda fid, name: impls[(fid, name)])
    got = region.cross_rank_topk_decode(None, None, None, None, 0, 6, None, object())
    assert rec.log == calls
    assert got is bool(calls) and (not calls or results[calls[-1]])


def _region_inputs(num_heads=64, head_dim=128, next_n=6):
    top_k = SimpleNamespace(
        needs_gvr_prior=False,
        decode_implementation=SimpleNamespace(value="default"),
        gvr_self_sampling=False,
        top_k=2048,
        compress_ratio=1,
    )
    indexer = SimpleNamespace(
        top_k=top_k,
        use_cute_dsl_paged_mqa_logits=False,
        use_fp4=False,
        use_gvr_emission=False,
        n_heads=num_heads,
        head_dim=head_dim,
        layer_idx=3,
    )
    kvm = SimpleNamespace(
        get_indexer_k_cache_buffers=lambda layer: torch.zeros(
            4, 64, 1, head_dim + 4, dtype=torch.uint8
        ),
        tokens_per_block=64,
        layer_offsets={3: 3},
    )
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=1,
        seq_lens=torch.tensor([next_n], dtype=torch.int32),
        kv_cache_manager=kvm,
        indexer_k_cache_block_offsets=torch.zeros(1, 16, dtype=torch.int32),
        mapping=SimpleNamespace(tp_size=8, enable_attention_dp=False),
        get_indexer_max_seq_len=lambda: 32768,
        is_ragged_verify=False,
        use_expanded_buffers_for_mtp=False,
        in_mtp_draft_loop=False,
        gen_indexer_kv_lens_cuda_runtime=torch.tensor([32768], dtype=torch.int32),
        req_idx_per_token=torch.zeros(next_n, dtype=torch.int32),
        num_ctx_tokens=0,
        num_tokens=next_n,
        block_table=torch.zeros(1, 512, dtype=torch.int32),
        num_seqs=1,
    )
    q_fp8 = torch.zeros(next_n, num_heads, head_dim)
    weights = torch.zeros(next_n, num_heads)
    out = torch.zeros(next_n, 2048, dtype=torch.int32)
    return indexer, metadata, q_fp8, weights, out


@pytest.mark.parametrize("with_indexer_front", [True, False])
def test_global_emit_takes_the_logits_launcher_of_indexer_front(monkeypatch, with_indexer_front):
    """cross_rank_topk_global_emit's region launches indexer_layer_front's logits kernel exactly
    when indexer_layer_front is selected."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        bridge as global_emit_bridge,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_indexer_decode as global_emit_decode,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_workspace as global_emit_ws,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        cp_logits_runner as indexer_front_runner,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        cross_rank_logits_runner as indexer_topk_runner,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        cross_rank_policy,
    )

    features = {
        "absorbed_query",
        "indexer_topk",
        "draft_absorbed_query",
        "rope_kv_append",
        "cross_rank_topk",
        "cross_rank_topk_cluster_merge",
        "cross_rank_topk_global_emit",
    } | ({"indexer_layer_front"} if with_indexer_front else set())
    monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset(features))
    launched = []

    def runner(owner):
        def ensure_compiled(*args, **kwargs):
            launched.append((owner, "compile"))

        def paged_mqa_logits_cp(
            q, k_cache, weights, kv_lens, block_table, max_seq_len, rank, world
        ):
            launched.append((owner, "logits"))
            assert (rank, world) == (2, 8)
            return torch.zeros(q.shape[1], 16)

        return ensure_compiled, paged_mqa_logits_cp

    for owner, mod in (
        ("indexer_layer_front", indexer_front_runner),
        ("indexer_topk", indexer_topk_runner),
    ):
        ensure, logits = runner(owner)
        monkeypatch.setattr(mod, "ensure_compiled", ensure)
        monkeypatch.setattr(mod, "paged_mqa_logits_cp", logits)
    monkeypatch.setattr(
        cross_rank_policy,
        "decide_cp_uniform",
        lambda **kw: SimpleNamespace(eligible=True, reason=""),
    )
    monkeypatch.setattr(cross_rank_policy, "rank_local_reason", lambda **kw: "")
    monkeypatch.setattr(global_emit_ws, "op_info", lambda world: (0,) * 16)
    monkeypatch.setattr(global_emit_ws, "cluster_reason", lambda info: "")
    workspace = SimpleNamespace(rank=2, world=8, state=None, mc_ptr=0, uc_ptr=0)

    def get_workspace(mapping, prepare=None, local_reason=""):
        prepare()
        return workspace

    monkeypatch.setattr(global_emit_ws, "get_workspace", get_workspace)
    remap = SimpleNamespace(page_index_scale=1, grouped=False, layer_ids=None, layer_offset=3)
    monkeypatch.setattr(global_emit_decode, "bridge_reason", lambda *a: ("", remap, None))
    topk_calls = []

    def topk(local_logits, kv_lens, out, ws, next_n, **kw):
        topk_calls.append((tuple(local_logits.shape), next_n, kw["layer_id"]))
        return torch.zeros(next_n, 2048, dtype=torch.int32)

    monkeypatch.setattr(global_emit_decode, "cp_topk_from_local_logits", topk)
    logged = {}
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", logged)

    indexer, metadata, q_fp8, weights, out = _region_inputs()
    packet = SimpleNamespace(plan=None)
    assert global_emit_decode.try_cp_indexer_decode(
        indexer, metadata, q_fp8, weights, 0, 6, out, packet
    )
    owner = "indexer_layer_front" if with_indexer_front else "indexer_topk"
    assert launched == [(owner, "compile"), (owner, "logits")]
    assert topk_calls == [((6, 16), 6, 3)]
    assert isinstance(packet.plan, global_emit_bridge.GlobalEmitPlan) and packet.plan.ids is out
    assert logged.get(("cross_rank_topk_global_emit", "cp_indexer_decode")) is True
    assert (
        ("indexer_layer_front", "cp_indexer_decode_global_emit") in logged
    ) is with_indexer_front
    if with_indexer_front:
        assert logged[("indexer_layer_front", "cp_indexer_decode_global_emit")] is True


def test_global_emit_decline_is_logged_for_both(monkeypatch):
    """A cross_rank_topk_global_emit decline with indexer_layer_front selected is visible on both capability keys."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_indexer_decode as global_emit_decode,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.cross_rank_topk_global_emit import (  # noqa: E501
        cp_workspace as global_emit_ws,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        cross_rank_policy,
    )

    features = {
        "absorbed_query",
        "indexer_topk",
        "draft_absorbed_query",
        "rope_kv_append",
        "cross_rank_topk",
        "cross_rank_topk_cluster_merge",
        "cross_rank_topk_global_emit",
        "indexer_layer_front",
    }
    monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset(features))
    monkeypatch.setattr(
        cross_rank_policy,
        "decide_cp_uniform",
        lambda **kw: SimpleNamespace(eligible=False, reason="why"),
    )
    monkeypatch.setattr(cross_rank_policy, "rank_local_reason", lambda **kw: "")
    monkeypatch.setattr(global_emit_ws, "op_info", lambda world: (0,) * 16)
    monkeypatch.setattr(global_emit_ws, "cluster_reason", lambda info: "")
    logged = {}
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", logged)
    indexer, metadata, q_fp8, weights, out = _region_inputs()
    packet = SimpleNamespace(plan=None)
    assert not global_emit_decode.try_cp_indexer_decode(
        indexer, metadata, q_fp8, weights, 0, 6, out, packet
    )
    assert packet.plan is None
    assert logged == {
        ("cross_rank_topk_global_emit", "cp_indexer_decode"): False,
        ("indexer_layer_front", "cp_indexer_decode_global_emit"): False,
    }


def test_mla_rope_line():
    """The rope_kv_append rope branch logs indexer_layer_front/rope and picks indexer_layer_front's
    rope only when its op is loaded."""
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        attention_forward,
    )

    src = inspect.getsource(attention_forward.forward_absorption_generation)
    assert '"indexer_layer_front", "rope_available"' in src
    assert '"indexer_layer_front",\n' in src and '"rope",' in src
    assert 'rope_owner = "rope_kv_append"' in src
