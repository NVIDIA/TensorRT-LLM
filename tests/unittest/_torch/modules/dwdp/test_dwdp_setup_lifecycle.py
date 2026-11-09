# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Failure rollback tests for the DWDP setup ownership chain."""

import gc
import pickle
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp import page_pool, weight_buffer
from tensorrt_llm._torch.modules.dwdp import setup as dwdp_setup_module
from tensorrt_llm._torch.modules.dwdp.page_pool import PagePool
from tensorrt_llm._torch.modules.dwdp.specs import MnnvlHandleSet, WeightSpec
from tensorrt_llm._torch.modules.dwdp.weight_buffer import PrefetchRingMemoryError, WeightBuffer
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager

pytestmark = pytest.mark.cpu_only


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    return model


def _mapping():
    return SimpleNamespace(dwdp_enabled=True, dwdp_rank=0, dwdp_size=2)


def _setup_mocks(stack: ExitStack) -> tuple[MagicMock, MagicMock, MagicMock]:
    transport = MagicMock()
    transport.get_handle_set.return_value = MagicMock()
    transport.get_peer_views.return_value = {}
    weight_buffer = MagicMock()
    weight_manager = MagicMock()

    stack.enter_context(
        patch.object(dwdp_setup_module.torch.cuda, "mem_get_info", return_value=(1, 2))
    )
    stack.enter_context(
        patch.object(
            dwdp_setup_module,
            "collect_moe_params",
            return_value=({}, ["gate_up_proj", "down_proj"]),
        )
    )
    stack.enter_context(patch.object(dwdp_setup_module, "_get_model_num_experts", return_value=4))
    stack.enter_context(
        patch.object(dwdp_setup_module, "build_weight_specs", return_value=MagicMock())
    )
    stack.enter_context(patch.object(dwdp_setup_module, "_agree_on_partition"))
    stack.enter_context(
        patch.object(dwdp_setup_module, "get_allocation_granularity", return_value=2 << 20)
    )
    stack.enter_context(
        patch.object(dwdp_setup_module.DWDPTransport, "create", return_value=transport)
    )
    stack.enter_context(
        patch.object(dwdp_setup_module.WeightBuffer, "create", return_value=weight_buffer)
    )
    stack.enter_context(patch.object(dwdp_setup_module, "fill_edge_bytes"))
    stack.enter_context(
        patch.object(dwdp_setup_module, "DWDPWeightManager", return_value=weight_manager)
    )
    stack.enter_context(
        patch.object(dwdp_setup_module, "_exchange_expert_scalars", side_effect=_vote)
    )
    stack.enter_context(patch.object(dwdp_setup_module, "fixup_moe_backends"))
    return transport, weight_buffer, weight_manager


def _vote(*_args: object, error: Exception | None = None, **_kwargs: object) -> dict:
    """Stands in for ``_exchange_expert_scalars``: a rank that failed raises its own error there."""
    if error is not None:
        raise error
    return {}


def _run_setup(**kwargs: object) -> DWDPWeightManager | None:
    return dwdp_setup_module.setup_dwdp(
        model=_model(),
        mapping=_mapping(),
        device_id=0,
        comm=MagicMock(),
        layer_indices=[3],
        num_experts_per_worker=2,
        num_prefetch_experts=2,
        num_prefetch_slots=2,
        **kwargs,
    )


def test_weight_buffer_creation_failure_releases_transport():
    with ExitStack() as stack:
        transport, _weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.WeightBuffer.create.side_effect = RuntimeError("buffer failed")

        with pytest.raises(RuntimeError, match="buffer failed"):
            _run_setup()

    transport.release.assert_called_once_with()


def test_edge_fill_failure_releases_buffer_and_transport():
    with ExitStack() as stack:
        transport, weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.fill_edge_bytes.side_effect = RuntimeError("fill failed")

        with pytest.raises(RuntimeError, match="fill failed"):
            _run_setup()

    weight_buffer.release.assert_called_once_with()
    transport.release.assert_called_once_with()


def test_keyboard_interrupt_after_transport_allocation_still_rolls_back():
    """Ctrl-C is not an ``Exception``; rollback must still run for it."""
    with ExitStack() as stack:
        transport, _weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.WeightBuffer.create.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            _run_setup()

    transport.release.assert_called_once_with()


def test_rollback_release_failure_preserves_the_original_error():
    """A raise from release() must not replace the error that broke setup."""
    with ExitStack() as stack:
        transport, weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.fill_edge_bytes.side_effect = RuntimeError("fill failed")
        weight_buffer.release.side_effect = RuntimeError("release failed")

        # "fill failed", not "release failed".
        with pytest.raises(RuntimeError, match="fill failed"):
            _run_setup()

    # The buffer release raised, yet the transport was still released.
    weight_buffer.release.assert_called_once_with()
    transport.release.assert_called_once_with()


def test_backend_fixup_failure_releases_owned_manager():
    with ExitStack() as stack:
        transport, _weight_buffer, weight_manager = _setup_mocks(stack)
        dwdp_setup_module.fixup_moe_backends.side_effect = RuntimeError("fixup failed")

        with pytest.raises(RuntimeError, match="fixup failed"):
            _run_setup()

    assert weight_manager._transport is transport
    weight_manager.release.assert_called_once_with()


def test_an_expert_weight_in_a_dense_slot_is_refused_before_the_transport() -> None:
    """The transport would free a reserved storage (a committed dense slot) under its owner."""
    slot = torch.zeros(64, dtype=torch.uint8)
    params = {
        (3, "w3_w1_weight"): slot[16:32],
        (3, "w2_weight"): torch.zeros(16, dtype=torch.uint8),
    }
    for reserved_storages, refused in (({slot.untyped_storage().data_ptr()}, True), ({1}, False)):
        with ExitStack() as stack:
            _setup_mocks(stack)
            dwdp_setup_module.collect_moe_params.return_value = (params, ["w3_w1_weight"])
            if refused:
                with pytest.raises(
                    RuntimeError,
                    match=r"Expert weights in reserved storage, which the transport would free: "
                    r"layers\.3\.w3_w1_weight\. They are views of the committed dense DWDP slots",
                ):
                    _run_setup(reserved_storages=frozenset(reserved_storages))
                dwdp_setup_module.DWDPTransport.create.assert_not_called()
            else:
                _run_setup(reserved_storages=frozenset(reserved_storages))
                dwdp_setup_module.DWDPTransport.create.assert_called_once()


GRANULARITY, PAGE, TOTAL = 2 << 20, 16 << 20, 256 << 30


@pytest.mark.parametrize("num_slots", [2, 3])
def test_a_ring_larger_than_free_memory_is_refused_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch, num_slots: int
) -> None:
    """One MoE layer per slot, of two 1 MiB-per-expert weights, 4 of 16 experts local: 2 pages
    per slot."""
    shapes = {"w1": (1024, 1024), "w2": (512, 2048)}
    specs = {
        layer: {
            name: WeightSpec(
                16, chunk_shape=(4, *shape), full_shape=(16, *shape), dtype=torch.uint8
            )
            for name, shape in shapes.items()
        }
        for layer in range(3, 3 + num_slots)
    }
    keys = [(layer, name) for layer in specs for name in shapes]
    handles = MnnvlHandleSet(
        handles={key: index for index, key in enumerate(keys, 1)},
        sizes=dict.fromkeys(keys, 2 * GRANULARITY),  # the 4 local experts' pages
    )
    needed, calls = 2 * num_slots * PAGE, []
    monkeypatch.setattr(weight_buffer, "get_allocation_granularity", lambda *_a, **_k: GRANULARITY)
    monkeypatch.setattr(weight_buffer.torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        weight_buffer.torch.cuda,
        "mem_get_info",
        lambda device_id: calls.append(("mem_get_info", device_id)) or (needed - 1, TOTAL),
    )
    create = MagicMock()
    monkeypatch.setattr(weight_buffer.PagePool, "create", create)
    monkeypatch.setattr(weight_buffer.WeightBuffer, "_setup_layer", MagicMock())

    with pytest.raises(PrefetchRingMemoryError) as raised:
        WeightBuffer.create(
            layer_weight_specs=specs,
            handles=handles,
            local_start=0,
            local_end=4,
            dwdp_size=4,
            device_id=0,
            num_slots=num_slots,
        )

    create.assert_not_called()
    assert calls == ["empty", ("mem_get_info", 0)]
    error = raised.value
    assert isinstance(error, RuntimeError)
    numbers = (error.needed_bytes, error.free_bytes, error.total_bytes, error.device_id)
    assert numbers == (needed, needed - 1, TOTAL, 0)
    assert (error.slot_pages, error.page_size) == ((2,) * num_slots, PAGE)
    message = str(error)
    pages = " + ".join(["2"] * num_slots)
    gib = needed / (1 << 30)
    assert message.startswith(
        f"[DWDP Setup] The expert prefetch ring does not fit on GPU 0: its {num_slots} slots need "
        f"{gib:.2f} GiB ({pages} pages of 16 MiB, the remote experts of {num_slots} MoE layers), "
        f"but {gib:.2f} GiB of 256.00 GiB is free."
    )
    assert "use a larger dwdp_size" in message
    assert "dense DWDP (dwdp_config.enable_dense)" in message
    # Fewer slots is a remedy only above the two the prefetch needs.
    assert ("fewer prefetch slots (DwdpConfig.num_prefetch_slots)" in message) == (num_slots > 2)
    # The executor sends a setup error to the proxy pickled: the error survives it.
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is PrefetchRingMemoryError and str(copy) == message
    assert (copy.needed_bytes, copy.slot_pages, copy.page_size) == (needed, (2,) * num_slots, PAGE)


def test_a_page_pool_allocation_failure_releases_every_handle_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slot 1 page 2 fails: the 5 handles made so far are released once, ``__del__`` adds none."""
    created, released = [], []

    def create_local_handle(size: int, device_id: int) -> int:
        assert (size, device_id) == (PAGE, 0)
        if len(created) == 5:
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
        created.append(1000 + len(created))
        return created[-1]

    monkeypatch.setattr(page_pool, "create_local_handle", create_local_handle)
    monkeypatch.setattr(page_pool, "release_handle", released.append)
    error = None
    try:
        PagePool([3 * PAGE, 3 * PAGE], 0, granularity=GRANULARITY, page_size=PAGE)
    except RuntimeError as exc:
        error = exc
    assert str(error) == "Failed to allocate page 2 in slot 1: CUDA_ERROR_OUT_OF_MEMORY"
    assert str(error.__cause__) == "CUDA_ERROR_OUT_OF_MEMORY"
    # Drop the traceback, the last reference to the half-built pool: its __del__ runs now.
    del error
    gc.collect()
    assert created == [1000, 1001, 1002, 1003, 1004]
    assert sorted(released) == created
