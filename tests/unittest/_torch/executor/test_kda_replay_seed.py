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
"""The batched KDA replay seed against the per-layer loop it replaces.

The reference is the loop of ``KDAHybridCacheManagerV2.seed_kda_replay_caches_for_disagg_gen``
(clear every replay buffer of the seeded slots, then copy the committed conv window into
the Q/K/V replay windows), run on an identical copy of the buffers. The batched seed must
leave every byte of every buffer equal to the loop's: seeded rows, other slots, and the
alignment padding of the beta cache.
"""

from typing import Dict, List, Sequence, Tuple

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kda_replay_seed import (
    CONV_BUFFER_NAMES,
    ReplaySeedPlan,
    batched_replay_seed_enabled,
    build_replay_seed_plan,
    seed_replay_caches,
    warmup_replay_seed_kernels,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

Layers = List[Dict[str, torch.Tensor]]
# The pool storage behind each view, so padding and untouched slots are compared too.
Storage = List[Dict[str, torch.Tensor]]


def _random(shape: Sequence[int], dtype: torch.dtype, gen: torch.Generator, device) -> torch.Tensor:
    values = torch.randn(*shape, generator=gen, dtype=torch.float32)
    return values.to(dtype).to(device)


def _make_replay_state(
    *,
    num_layers: int,
    num_slots: int,
    num_heads: int,
    head_dim: int,
    committed: int,
    num_spec: int,
    aligned_heads: int,
    seed: int,
    device: str,
) -> Tuple[Layers, Storage, List[torch.Tensor]]:
    """Per-layer replay views as ``_allocate_pool_replay_buffers`` builds them.

    Conv windows are ``[slots, window, dim]`` storage used transposed as ``[slots, dim,
    window]``; the beta cache keeps its CuTe row alignment and exposes the first
    ``num_heads`` columns.
    """
    gen = torch.Generator().manual_seed(seed)
    dim = num_heads * head_dim
    layers: Layers = []
    storage: Storage = []
    conv_states: List[torch.Tensor] = []
    for _ in range(num_layers):
        views: Dict[str, torch.Tensor] = {}
        bases: Dict[str, torch.Tensor] = {}
        for name in CONV_BUFFER_NAMES:
            base = _random((num_slots, committed + num_spec, dim), torch.bfloat16, gen, device)
            bases[name] = base
            views[name] = base.transpose(-1, -2)
        for name, dtype in (
            ("kda_k_cache", torch.bfloat16),
            ("kda_g_cache", torch.float32),
            ("kda_v_cache", torch.bfloat16),
        ):
            base = _random((num_slots, num_spec, dim), dtype, gen, device)
            bases[name] = base
            views[name] = base
        beta = _random((num_slots, num_spec, aligned_heads), torch.bfloat16, gen, device)
        bases["kda_beta_cache"] = beta
        views["kda_beta_cache"] = beta[..., :num_heads]
        layers.append(views)
        storage.append(bases)
        conv_states.append(_random((num_slots, 3 * dim, committed), torch.bfloat16, gen, device))
    return layers, storage, conv_states


def _loop_seed(
    layers: Layers,
    conv_states: List[torch.Tensor],
    slots: torch.Tensor,
    section_offsets: List[int],
    committed: int,
) -> None:
    """``seed_kda_replay_caches_for_disagg_gen``'s per-layer fallback, verbatim."""
    for layer, conv_state in zip(layers, conv_states):
        for replay_buffer in layer.values():
            replay_buffer.index_fill_(0, slots, 0)
        selected_conv = conv_state.index_select(0, slots)
        for section_idx, name in enumerate(CONV_BUFFER_NAMES):
            replay_layer = layer[name]
            section = selected_conv[
                :, section_offsets[section_idx] : section_offsets[section_idx + 1], :
            ]
            replay_layer[:, :, :committed].index_copy_(0, slots, section.to(replay_layer.dtype))


def _assert_storage_equal(got: Storage, expected: Storage) -> None:
    for layer, (got_layer, expected_layer) in enumerate(zip(got, expected)):
        for name, expected_buffer in expected_layer.items():
            torch.testing.assert_close(
                got_layer[name], expected_buffer, rtol=0, atol=0, msg=f"layer {layer} {name}"
            )


# (layers, slots in the pool, heads, head dim, committed window, draft length, aligned heads)
SHAPES = [
    pytest.param(3, 6, 4, 32, 3, 3, 8, id="tiny"),
    pytest.param(4, 9, 8, 128, 3, 5, 8, id="tp8-like"),  # D = 1024, draft 5
    pytest.param(3, 5, 64, 128, 3, 3, 64, id="dep16-like"),  # D = 8192, draft 3
    pytest.param(2, 5, 12, 128, 3, 3, 16, id="ragged-tile"),  # D = 1536: partial last tile
]
SLOT_SETS = [[2], [0, 3], [4, 1, 2]]


@requires_gpu
@pytest.mark.parametrize(
    "num_layers,num_slots,num_heads,head_dim,committed,num_spec,aligned", SHAPES
)
@pytest.mark.parametrize("slot_list", SLOT_SETS, ids=lambda s: f"slots{len(s)}")
def test_batched_seed_matches_the_layer_loop(
    num_layers, num_slots, num_heads, head_dim, committed, num_spec, aligned, slot_list
) -> None:
    kwargs = dict(
        num_layers=num_layers,
        num_slots=num_slots,
        num_heads=num_heads,
        head_dim=head_dim,
        committed=committed,
        num_spec=num_spec,
        aligned_heads=aligned,
        seed=1234,
        device="cuda",
    )
    layers, storage, conv_states = _make_replay_state(**kwargs)
    ref_layers, ref_storage, ref_conv = _make_replay_state(**kwargs)
    dim = num_heads * head_dim
    offsets = [0, dim, 2 * dim, 3 * dim]
    slots = torch.tensor(sorted(slot_list), dtype=torch.long, device="cuda")

    plan = build_replay_seed_plan(layers, conv_states, offsets, committed)
    assert plan is not None
    seed_replay_caches(plan, slots)
    _loop_seed(ref_layers, ref_conv, slots, offsets, committed)

    _assert_storage_equal(storage, ref_storage)
    for got, expected in zip(conv_states, ref_conv):  # the transferred state is only read
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@requires_gpu
def test_batched_seed_is_stream_ordered_without_host_sync() -> None:
    kwargs = dict(
        num_layers=3,
        num_slots=4,
        num_heads=8,
        head_dim=128,
        committed=3,
        num_spec=5,
        aligned_heads=8,
        seed=7,
        device="cuda",
    )
    layers, storage, conv_states = _make_replay_state(**kwargs)
    ref_layers, ref_storage, ref_conv = _make_replay_state(**kwargs)
    offsets = [0, 1024, 2048, 3072]
    plan = build_replay_seed_plan(layers, conv_states, offsets, 3)
    assert plan is not None
    slots = torch.tensor([1, 3], dtype=torch.long, device="cuda")

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        seed_replay_caches(plan, slots)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    _loop_seed(ref_layers, ref_conv, slots, offsets, 3)
    _assert_storage_equal(storage, ref_storage)


@requires_gpu
def test_warmup_compiles_without_touching_any_buffer() -> None:
    kwargs = dict(
        num_layers=2,
        num_slots=3,
        num_heads=4,
        head_dim=64,
        committed=3,
        num_spec=3,
        aligned_heads=8,
        seed=11,
        device="cuda",
    )
    layers, storage, conv_states = _make_replay_state(**kwargs)
    before, _, _ = _make_replay_state(**kwargs)
    plan = build_replay_seed_plan(layers, conv_states, [0, 256, 512, 768], 3)
    assert plan is not None

    warmup_replay_seed_kernels(plan)

    for got, expected in zip(layers, before):
        for name in got:
            torch.testing.assert_close(got[name], expected[name], rtol=0, atol=0)


@requires_gpu
def test_live_admissions_are_counted_and_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    # The plan's construction line only says the batched path exists; these lines come from the
    # admissions that were really seeded (warm-up compiles without seeding and does not count).
    import types

    from tensorrt_llm._torch.pyexecutor import kda_replay_seed as seed_module

    logged: List[str] = []
    monkeypatch.setattr(seed_module, "logger", types.SimpleNamespace(info=logged.append))
    monkeypatch.setattr(seed_module, "_seeded_admissions", 0)
    monkeypatch.setattr(seed_module, "_seeded_slots", 0)
    monkeypatch.setattr(seed_module, "_next_log_admissions", 1)
    layers, _, conv_states = _make_replay_state(
        num_layers=2,
        num_slots=3,
        num_heads=4,
        head_dim=64,
        committed=3,
        num_spec=3,
        aligned_heads=8,
        seed=5,
        device="cuda",
    )
    plan = build_replay_seed_plan(layers, conv_states, [0, 256, 512, 768], 3)
    assert plan is not None
    warmup_replay_seed_kernels(plan)
    no_slots = torch.zeros(0, dtype=torch.long, device="cuda")
    seed_replay_caches(plan, no_slots)  # nothing to seed: not an admission
    assert not logged
    slots = torch.tensor([2, 0], dtype=torch.long, device="cuda")
    for _ in range(12):
        seed_replay_caches(plan, slots)
    torch.cuda.synchronize()
    assert logged == [
        "KDA replay seed: 1 admissions (2 slots) seeded by the batched plan",
        "KDA replay seed: 10 admissions (20 slots) seeded by the batched plan",
    ]


@requires_gpu
def test_seed_of_no_slots_is_a_no_op() -> None:
    layers, storage, conv_states = _make_replay_state(
        num_layers=2,
        num_slots=3,
        num_heads=4,
        head_dim=64,
        committed=3,
        num_spec=3,
        aligned_heads=8,
        seed=5,
        device="cuda",
    )
    before, before_storage, _ = _make_replay_state(
        num_layers=2,
        num_slots=3,
        num_heads=4,
        head_dim=64,
        committed=3,
        num_spec=3,
        aligned_heads=8,
        seed=5,
        device="cuda",
    )
    plan = build_replay_seed_plan(layers, conv_states, [0, 256, 512, 768], 3)
    assert plan is not None
    seed_replay_caches(plan, torch.empty(0, dtype=torch.long, device="cuda"))
    _assert_storage_equal(storage, before_storage)


def _cpu_state(**overrides) -> Tuple[Layers, List[torch.Tensor], List[int], int]:
    kwargs = dict(
        num_layers=3,
        num_slots=4,
        num_heads=4,
        head_dim=32,
        committed=3,
        num_spec=3,
        aligned_heads=8,
        seed=3,
        device="cpu",
    )
    kwargs.update(overrides)
    layers, _, conv_states = _make_replay_state(**kwargs)
    dim = kwargs["num_heads"] * kwargs["head_dim"]
    return layers, conv_states, [0, dim, 2 * dim, 3 * dim], kwargs["committed"]


@pytest.mark.cpu_only
def test_plan_records_one_address_per_layer_and_buffer() -> None:
    layers, conv_states, offsets, committed = _cpu_state()

    plan = build_replay_seed_plan(layers, conv_states, offsets, committed)

    assert isinstance(plan, ReplaySeedPlan)
    assert plan.num_layers == 3
    assert [buf.ptrs.tolist() for buf in plan.buffers] == [
        [layer[name].data_ptr() for layer in layers] for name in layers[0]
    ]
    assert plan.conv_ptrs.tolist() == [state.data_ptr() for state in conv_states]
    # The conv windows read the three conv-state sections; the histories read none.
    assert [buf.conv_channel_offset for buf in plan.buffers] == [
        0,
        128,
        256,
        None,
        None,
        None,
        None,
    ]


def _non_uniform_dtype(layers, conv_states, offsets):
    layers[1]["kda_k_cache"] = layers[1]["kda_k_cache"].to(torch.float32)


def _non_uniform_stride(layers, conv_states, offsets):
    layers[2]["kda_v_cache"] = (
        layers[2]["kda_v_cache"].transpose(-1, -2).contiguous().transpose(-1, -2)
    )


def _missing_buffer(layers, conv_states, offsets):
    del layers[1]["kda_conv_k"]


def _conv_state_layout_differs(layers, conv_states, offsets):
    conv_states[1] = conv_states[1].transpose(-1, -2).contiguous().transpose(-1, -2)


def _unequal_sections(layers, conv_states, offsets):
    offsets[2] = offsets[1] + 64


def _channel_dim_not_contiguous(layers, conv_states, offsets):
    for layer in layers:
        layer["kda_conv_q"] = layer["kda_conv_q"].contiguous()  # [slots, dim, window] row-major


def _unsupported_dtype(layers, conv_states, offsets):
    for layer in layers:
        layer["kda_g_cache"] = layer["kda_g_cache"].to(torch.float64)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mutate",
    [
        _non_uniform_dtype,
        _non_uniform_stride,
        _missing_buffer,
        _conv_state_layout_differs,
        _unequal_sections,
        _channel_dim_not_contiguous,
        _unsupported_dtype,
    ],
)
def test_plan_falls_back_for_layouts_the_kernel_cannot_express(mutate) -> None:
    layers, conv_states, offsets, committed = _cpu_state()
    mutate(layers, conv_states, offsets)

    assert build_replay_seed_plan(layers, conv_states, offsets, committed) is None


@pytest.mark.cpu_only
def test_plan_requires_layers() -> None:
    assert build_replay_seed_plan([], [], [0, 1, 2, 3], 3) is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True)],
    ids=["unset", "empty", "0", "1", "1-padded"],
)
def test_batched_seed_is_opt_in(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv("TRTLLM_KDA_BATCHED_REPLAY_SEED", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_KDA_BATCHED_REPLAY_SEED", value)

    assert batched_replay_seed_enabled() is expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["2", "-1", "true", "on", "auto", "01"])
def test_batched_seed_switch_rejects_other_values(monkeypatch, value) -> None:
    monkeypatch.setenv("TRTLLM_KDA_BATCHED_REPLAY_SEED", value)

    with pytest.raises(ValueError, match="TRTLLM_KDA_BATCHED_REPLAY_SEED"):
        batched_replay_seed_enabled()
