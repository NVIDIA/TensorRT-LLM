# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page tables of CSA2 context-only workers (``context_swa_layer_limit``) against full workers,
built by the layout adapter of the real ``CSA2CacheManager`` over stand-in storage. Under bounded
replay on generation the two register both ways, and every generation buffer of a model layer the
context worker holds is written unless a remote-tail request leaves it to the generation worker; a
generation worker without replay, which would leave it unwritten, is refused."""

from types import SimpleNamespace

import pytest
from utils.page_layout_oracle import assert_layouts_reshard

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout import layout_mapping
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.adapter import adapter_for
from tensorrt_llm.bindings import DataType

NUM_LAYERS = 40


def _layout() -> CSA2Layout:
    return CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer_id=20,
    )


def _storage(layers) -> SimpleNamespace:
    """The runtime storage for layer configs ``layers``: a layer group per attention window, in
    order of first appearance, each in a pool group of its own whose pool ``i`` holds the group's
    ``i``-th buffer size, its buffers back to back."""
    windows: dict = {}
    for layer in layers:
        windows.setdefault(getattr(layer, "window_size", None), []).append(int(layer.layer_id))
    grouping = list(windows.values())
    descs = []
    for g, layer_ids in enumerate(grouping):
        by_size: dict = {}
        for layer_id in layer_ids:
            for buffer in layers[layer_id].buffers:
                by_size.setdefault(int(buffer.size), []).append(
                    SimpleNamespace(layer_id=layer_id, role=buffer.role)
                )
        coalesced = [
            SimpleNamespace(single_buffer_size=size, buffer_ids=ids)
            for size, ids in by_size.items()
        ]
        descs.append(
            SimpleNamespace(
                pool_group_index=g,
                num_slots=4,
                pools=[
                    SimpleNamespace(
                        pool_index=i,
                        base_address=((g + 1) << 32) + (i << 24),
                        slot_bytes=c.single_buffer_size * len(c.buffer_ids),
                    )
                    for i, c in enumerate(coalesced)
                ],
                slot_desc=SimpleNamespace(
                    variants=[SimpleNamespace(layer_group_id=g, coalesced_buffers=coalesced)]
                ),
            )
        )
    return SimpleNamespace(
        layer_grouping=grouping,
        pool_group_descs=descs,
        get_life_cycle_pool_group_indices=lambda: list(range(len(grouping))),
    )


def _manager(limit: int | None, replay: bool = True) -> CSA2CacheManager:
    """A TP1 stage of all 40 layers; a context-only worker exists only under bounded replay on
    generation, which peers share."""
    manager = object.__new__(CSA2CacheManager)
    manager.context_swa_layer_limit = limit
    manager.bounded_replay_on_generation = replay
    manager.layout = _layout()
    manager.pp_layers = list(range(NUM_LAYERS))
    manager.tokens_per_block = 128
    manager.max_draft_len = manager.reuse_match_backoff = 0
    manager._decoder_replay_window = 0
    manager._encoder_replay_enabled = False
    config = manager._build_cache_config(SimpleNamespace(swa_scratch_reuse=None))
    manager.kv_cache_manager_py_config = config
    manager.impl = _storage(list(config.layers))
    manager.mapping = SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False)
    manager.num_kv_heads = 1
    manager.num_kv_heads_per_layer = [1] * NUM_LAYERS
    manager._has_cp_helix = False
    manager._helix_cp_rank, manager._helix_cp_size = 0, 1
    manager.dtype = DataType.FP8
    manager._model_attention_windows = {}
    manager._layer_group_life_cycles = None
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [2, 20, 21, 39])
def test_a_context_only_worker_pairs_with_a_replaying_generation_worker(limit):
    context, generation = _manager(limit), _manager(None)
    ctx = adapter_for(context).page_table().page_layout
    gen = adapter_for(generation).page_table().page_layout
    layout_mapping.check_compatible(ctx, gen)
    layout_mapping.check_compatible(gen, ctx)
    held = {model for group in ctx.layer_groups for model in group.model_layers}
    model_of = {
        layer: model
        for group in gen.layer_groups
        for layer, model in zip(group.layers, group.model_layers)
    }
    excluded = adapter_for(generation).untransferred_buffers(
        SimpleNamespace(py_csa2_remote_tail_split=limit)
    )
    written = {(r.dst.layer, r.dst.role) for r in layout_mapping.map_buffers(ctx, gen)}
    wanted = {(b.layer, b.role) for b in gen.buffers if b.transferred and model_of[b.layer] in held}
    assert not wanted - written - excluded, sorted(wanted - written - excluded)
    assert_layouts_reshard([ctx], [gen])


@pytest.mark.cpu_only
def test_a_context_only_worker_is_refused_by_a_generation_worker_without_replay():
    ctx = adapter_for(_manager(20)).page_table().page_layout
    gen = adapter_for(_manager(None, replay=False)).page_table().page_layout
    with pytest.raises(ValueError, match="transferred by one rank only"):
        layout_mapping.check_compatible(ctx, gen)
