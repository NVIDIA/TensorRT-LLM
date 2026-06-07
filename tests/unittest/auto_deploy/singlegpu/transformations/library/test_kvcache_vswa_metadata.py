# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-shape tests for the VSWA per-group metadata wiring in the kvcache transform.

The kvcache transform builds a per-window page table layout. Phase-1 of the
Gemma-4 enablement (PR #13745) only routed *standard* metadata (cache_loc,
cu_num_pages, last_page_len) per group; the *extra*-metadata op (the host-side
prepare that produces e.g. update_paged_kv_cache write positions) was built
once from group-0 inputs and reused for every layer. With seq_len_with_cache
added to the swappable set (Phase 2), the extra-metadata op MUST also run per
group, otherwise non-zero-group layers silently consume group-0's
seq_len_with_cache (window-uncapped) → wrong write positions under eviction.

These tests are graph-shape only — no GPU, no kernel execution.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
    InsertCachedAttention,
    InsertCachedAttentionConfig,
)


class _TwoWindowModule(torch.nn.Module):
    """A tiny two-layer model with one SWA layer and one full-attention layer.

    Layer 0 is SWA (sliding_window=256), layer 1 is full attention.  This
    forces the transform to create two window groups.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], 2, 4)
        swa_layer = torch.ops.auto_deploy.torch_attention(
            qkv,
            qkv,
            qkv,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            1.0,  # scale
            None,  # sinks
            256,  # sliding_window  <-- SWA group
            None,  # logit_cap
            "bsnd",
            0,  # layer_idx
        )
        full_layer = torch.ops.auto_deploy.torch_attention(
            qkv,
            qkv,
            qkv,
            None,
            0.0,
            True,
            1.0,
            None,
            None,  # sliding_window=None  <-- full-attention group
            None,
            "bsnd",
            1,
        )
        return swa_layer + full_layer


def _run_transform(backend: str):
    module = _TwoWindowModule().eval()
    gm = torch_export_to_gm(module, (torch.randn(1, 4, 8),))
    cm = CachedSequenceInterface(
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=512,
        device="cpu",
    )
    transform = InsertCachedAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend=backend)
    )
    gm, info = transform._apply(gm, cm, factory=None, shared_config=SharedConfig())
    return gm, info, cm


def _placeholder_names(gm) -> list:
    return [node.target for node in gm.graph.nodes if node.op == "placeholder"]


def test_vswa_registers_per_group_seq_len_with_cache_placeholders():
    """Group 1 must have its own seq_len_with_cache_g1{_host} placeholders.

    This is required so the SWA group's window-capped value reaches the
    kernel and the prepare-extra-metadata op.
    """
    gm, info, cm = _run_transform(backend="triton")
    assert info.num_matches == 2, "expected 2 cached-attention insertions"

    names = _placeholder_names(gm)
    # Group 0 keeps the unsuffixed name; group 1 must have its own _g1 variant.
    assert "seq_len_with_cache_host" in names
    assert "seq_len_with_cache_g1_host" in names, (
        "seq_len_with_cache must be in vswa_swappable_bases — without per-group "
        "wiring, SWA layers silently consume group-0's window-uncapped value."
    )


def test_vswa_extra_metadata_is_per_group():
    """Prepare-extra-metadata must be invoked once per window group.

    The Triton backend's prepare-extra op consumes seq_len_with_cache. With
    seq_len_with_cache routed per-group, the transform MUST invoke the
    extra-op once per group; otherwise non-zero-group layers route through
    group-0's prepare-extra output and the swappable seq_len_with_cache_g1
    placeholder is dead.

    This is the regression guard against the pre-fix transform behavior
    described in the autodeploy-kvcache-vswa-meta-nodes-extra backlog.
    """
    gm, info, cm = _run_transform(backend="triton")

    prep_meta_op = torch.ops.auto_deploy.triton_prepare_metadata.default
    prep_calls = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == prep_meta_op
    ]
    assert len(prep_calls) == 2, (
        f"expected one prepare-extra call per group (2 groups), got {len(prep_calls)}"
    )

    # The two calls must consume DIFFERENT seq_len_with_cache placeholders:
    # one from group 0 (unsuffixed), one from group 1 (suffixed).
    swc_inputs_by_call = []
    for call in prep_calls:
        swc_input = None
        for arg in call.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.op == "placeholder"
                and "seq_len_with_cache" in str(arg.target)
            ):
                swc_input = str(arg.target)
                break
        assert swc_input is not None, (
            f"prepare-extra call {call.format_node()} did not consume any "
            f"seq_len_with_cache placeholder"
        )
        swc_inputs_by_call.append(swc_input)

    assert len(set(swc_inputs_by_call)) == 2, (
        f"both prepare-extra calls consume the same seq_len_with_cache "
        f"input ({swc_inputs_by_call}); per-group routing failed"
    )


def test_vswa_each_layer_routes_to_its_groups_extra_metadata():
    """Each layer's cached-attn must wire to its own group's prepare-extra outputs.

    The cached-attn call for group 1 (the SWA layer) must consume the
    group-1 prepare-extra outputs, not group 0's.
    """
    gm, info, cm = _run_transform(backend="triton")

    cached_op = torch.ops.auto_deploy.triton_mha_with_cache.default
    cached_calls = [
        node for node in gm.graph.nodes if node.op == "call_function" and node.target == cached_op
    ]
    assert len(cached_calls) == 2

    # For each cached-attn call, find which seq_len_with_cache placeholder its
    # extra-metadata args ultimately depend on.  We do a simple recursive walk
    # through args (bounded by graph size) since the prepare-extra outputs are
    # getitem nodes of a call_function whose args include the placeholder.
    def find_swc_dep(node, visited=None):
        if visited is None:
            visited = set()
        if id(node) in visited:
            return None
        visited.add(id(node))
        if isinstance(node, torch.fx.Node):
            if node.op == "placeholder" and "seq_len_with_cache" in str(node.target):
                return str(node.target)
            if node.op in ("call_function", "call_method"):
                for arg in list(node.args) + list(node.kwargs.values()):
                    result = find_swc_dep(arg, visited)
                    if result is not None:
                        return result
        return None

    deps_per_call = []
    for call in cached_calls:
        # The first three args are q/k/v; standard metadata follows.  Walk all
        # args until we find a seq_len_with_cache placeholder dependency.
        dep = None
        for arg in call.args:
            dep = find_swc_dep(arg)
            if dep is not None:
                break
        assert dep is not None, "no seq_len_with_cache dependency found"
        deps_per_call.append(dep)

    # The two cached-attn calls (one per layer = one per group) must each route
    # to a distinct seq_len_with_cache placeholder.
    assert len(set(deps_per_call)) == 2, (
        f"both cached-attn calls depend on the same seq_len_with_cache "
        f"placeholder ({deps_per_call}); per-group wiring failed"
    )


@pytest.mark.parametrize(
    "backend, expected_cyclic",
    [("triton", False), ("trtllm", True)],
)
def test_vswa_sets_kernel_handles_cyclic_swa(backend, expected_cyclic):
    """The transform records the backend's cyclic-SWA capability on the interface.

    trtllm's kernel masks the sliding window internally (cyclic), so the
    executor must pass full block tables + global lengths; triton must not.
    """
    gm, info, cm = _run_transform(backend=backend)
    assert info.num_matches == 2
    assert cm.kernel_handles_cyclic_swa is expected_cyclic
    # Both backends still register two window groups regardless of cyclic-ness.
    assert len(cm.kv_group_windows) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
