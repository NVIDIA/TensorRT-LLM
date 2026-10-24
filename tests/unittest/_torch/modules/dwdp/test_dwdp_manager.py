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
"""Unit tests for the DwdpManager facade.

DwdpManager is a thin lifecycle layer that:
- creates a DWDP MPI sub-communicator
- owns the Single Source of Truth for registered MoE layer indices
- forwards runtime calls to the DWDPWeightManager produced by setup_dwdp
- owns the dense DWDP runtime: sets it up after setup_dwdp, primes it together
  with the expert prefetch and releases it before the expert teardown

These tests mock out COMM_WORLD / dwdp_global_rank / setup_dwdp /
setup_dense_dwdp so the full lifecycle can be exercised on a single CPU without
MPI / CUDA / MNNVL. Dense DWDP is off unless a test turns it on. The cleanup
tests with real dense bindings repoint the toy Kimi-K3 layers of
``test_dense_layout.py``.
"""

import contextlib
import gc
import os
import unittest
import weakref
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
from test_dense_layout import COVERED, ToyModel
from test_dense_repoint import allocated_ring, setup_bindings

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp.dense.config import DENSE_ENV
from tensorrt_llm._torch.modules.dwdp.dense.repoint import read_binding
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    ROW_ALIGN,
    DenseProtocolError,
    DenseRepointError,
    DenseSetupError,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.pyexecutor.dwdp import (
    LEGACY_PREFETCH_ENV,
    WRAP_PREFETCH_ENV,
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FAC = "tensorrt_llm._torch.pyexecutor.dwdp"
_DENSE_SETUP = "tensorrt_llm._torch.modules.dwdp.dense.setup"


def _make_config(dwdp_size: int = 2, num_groups: int = 1) -> DwdpConfig:
    return DwdpConfig(
        dwdp_size=dwdp_size,
        num_groups=num_groups,
        num_experts_per_worker=4,
        num_prefetch_experts=4,
    )


def _make_mapping(dwdp_size: int = 2, dwdp_rank: int = 0) -> Mapping:
    return Mapping(
        world_size=dwdp_size,
        rank=dwdp_rank,
        tp_size=dwdp_size,
        dwdp_size=dwdp_size,
        dwdp_rank=dwdp_rank,
    )


def _configure_mock_comm_world(mock_comm_world, world_size: int = 1024):
    """Configure a class-decorator-patched COMM_WORLD mock with sensible defaults.

    Tests that construct a real DwdpManager need both ``Create_group`` (returns
    the dwdp sub-comm) and ``Get_size`` (used by num_groups validation) to be
    set; raw MagicMock returns from ``Get_size`` would fail integer comparison.
    """
    sub_comm = MagicMock()
    mock_comm_world.Create_group.return_value = sub_comm
    mock_comm_world.group.Incl.return_value = MagicMock()
    mock_comm_world.Get_size.return_value = world_size
    return sub_comm


def _start(test: unittest.TestCase, patcher):
    """Start ``patcher`` for the rest of ``test`` and return its mock."""
    mock = patcher.start()
    test.addCleanup(patcher.stop)
    return mock


def _set_dense_env(test: unittest.TestCase, value: str | None) -> None:
    """Set ``TRTLLM_DWDP_DENSE`` (None: unset) for the rest of ``test``.

    ``TRTLLM_DWDP_LEGACY_PREFETCH`` and ``TRTLLM_DWDP_WRAP_PREFETCH`` are unset too, so an
    ambient value cannot change the protocol.
    """
    _start(test, patch.dict(os.environ))
    os.environ.pop(DENSE_ENV, None)
    os.environ.pop(LEGACY_PREFETCH_ENV, None)
    os.environ.pop(WRAP_PREFETCH_ENV, None)
    if value is not None:
        os.environ[DENSE_ENV] = value


@patch("tensorrt_llm._torch.pyexecutor.dwdp.setup_dwdp")
@patch("tensorrt_llm._torch.pyexecutor.dwdp.dwdp_global_rank", return_value=0)
@patch("tensorrt_llm._torch.pyexecutor.dwdp.COMM_WORLD")
class TestDwdpManagerLifecycle(unittest.TestCase):
    def setUp(self):
        # Ensure no leftover global singleton from a prior test.
        set_global_dwdp_manager(None)
        # Dense DWDP off: a model without dense roots, TRTLLM_DWDP_DENSE unset.
        _set_dense_env(self, None)
        _start(self, patch(f"{_FAC}.dense_roots_available", return_value=False))
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp"))

    def tearDown(self):
        set_global_dwdp_manager(None)
        self.setup_dense.assert_not_called()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_init_rejects_existing_singleton_before_creating_comm(
        self, mock_comm_world, _rank, _setup
    ):
        set_global_dwdp_manager(MagicMock())

        with self.assertRaisesRegex(RuntimeError, "already registered globally"):
            DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

        mock_comm_world.Create_group.assert_not_called()

    def test_init_stores_fields(self, mock_comm_world, _rank, _setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024

        mgr = DwdpManager(
            config=_make_config(dwdp_size=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=0),
        )
        self.assertEqual(mgr.dwdp_size, 2)
        self.assertEqual(mgr.dwdp_rank, 0)
        self.assertEqual(mgr._registered_layers, [])
        self.assertIs(mgr.dwdp_group, sub_comm)
        self.assertIsNone(mgr._weight_manager)
        # start_expert_id / end_expert_id derive from
        # num_prefetch_experts (stride) and num_experts_per_worker (size).
        # rank=0: start=0, end=4.
        self.assertEqual(mgr.start_expert_id, 0)
        self.assertEqual(mgr.end_expert_id, 4)

    def test_init_expert_range_uniform(self, mock_comm_world, _rank, _setup):
        """Uniform partition (size == stride): each rank's range is a clean slice."""
        _configure_mock_comm_world(mock_comm_world)
        # dwdp_rank=1 with stride=4 -> start=4, end=4+4=8.
        mgr = DwdpManager(
            config=_make_config(dwdp_size=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=1),
        )
        self.assertEqual(mgr.start_expert_id, 4)
        self.assertEqual(mgr.end_expert_id, 8)

    def test_init_expert_range_redundancy(self, mock_comm_world, _rank, _setup):
        """Redundancy partition (size > stride): adjacent ranges overlap."""
        _configure_mock_comm_world(mock_comm_world)
        redundant_config = DwdpConfig(
            dwdp_size=2,
            num_groups=1,
            num_experts_per_worker=6,  # storage size
            num_prefetch_experts=4,  # stride < size -> 2-expert overlap
        )
        # rank=0: [0, 6). rank=1 would be [4, 10) -> 2-expert overlap [4, 6).
        mgr = DwdpManager(
            config=redundant_config,
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=0),
        )
        self.assertEqual(mgr.start_expert_id, 0)
        self.assertEqual(mgr.end_expert_id, 6)
        # Stored config knobs match what's plumbed into setup_dwdp.
        self.assertEqual(mgr.num_experts_per_worker, 6)
        self.assertEqual(mgr.num_prefetch_experts, 4)

    # ------------------------------------------------------------------
    # num_groups topology validation (was a dead schema field through 1fbc0d49)
    # ------------------------------------------------------------------

    def test_init_accepts_multi_group_topology(self, mock_comm_world, _rank, _setup):
        # num_groups=2, dwdp_size=2 -> 4 CTX ranks total. World >= 4 is fine.
        _configure_mock_comm_world(mock_comm_world, world_size=8)
        mgr = DwdpManager(
            config=DwdpConfig(
                dwdp_size=2,
                num_groups=2,  # valid: this rank (0) is in group 0
                num_experts_per_worker=4,
                num_prefetch_experts=4,
            ),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        self.assertEqual(mgr.num_groups, 2)

    # ------------------------------------------------------------------
    # Global singleton + enter/exit
    # ------------------------------------------------------------------

    def test_enter_registers_global(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        self.assertIsNone(get_global_dwdp_manager())
        mgr.__enter__()
        self.assertIs(get_global_dwdp_manager(), mgr)
        mgr.__exit__(None, None, None)
        self.assertIsNone(get_global_dwdp_manager())

    def test_cleanup_frees_comm_and_idempotent(self, mock_comm_world, _rank, _setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.cleanup()
        sub_comm.Free.assert_called_once()
        self.assertIsNone(mgr.dwdp_group)
        # Idempotent — second call must not raise
        mgr.cleanup()
        sub_comm.Free.assert_called_once()  # still only one Free

    def test_exit_clears_global_even_when_cleanup_fails(self, mock_comm_world, _rank, _setup):
        _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.__enter__()

        with patch.object(mgr, "cleanup", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                mgr.__exit__(None, None, None)

        self.assertIsNone(get_global_dwdp_manager())

    def test_cleanup_frees_comm_when_weight_release_fails(self, mock_comm_world, _rank, _setup):
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr._weight_manager = MagicMock()
        mgr._weight_manager.release.side_effect = RuntimeError("release failed")

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            mgr.cleanup()

        sub_comm.Free.assert_called_once_with()
        self.assertIsNone(mgr.dwdp_group)

    def test_cleanup_logs_a_failing_free_after_a_release_error(
        self, mock_comm_world, _rank, _setup
    ):
        """The release error is raised; the Free failure after it is logged through ``error``."""
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        sub_comm.Free.side_effect = RuntimeError("free failed")
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr._weight_manager = MagicMock()
        mgr._weight_manager.release.side_effect = RuntimeError("release failed")

        # Only ``error`` is replaced: any other logger method the path calls must exist.
        with patch(f"{_FAC}.logger.error") as log_error:
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                mgr.cleanup()

        sub_comm.Free.assert_called_once_with()
        (message,), _ = log_error.call_args
        self.assertIn("Failed to free DWDP group", message)
        self.assertIn("free failed", message)
        self.assertIsNone(mgr.dwdp_group)
        self.assertIsNone(mgr._weight_manager)

    def test_cleanup_releases_weight_manager_once(self, mock_comm_world, _rank, _setup):
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        weight_manager = MagicMock()
        mgr._weight_manager = weight_manager

        mgr.cleanup()
        mgr.cleanup()

        weight_manager.release.assert_called_once_with()
        sub_comm.Free.assert_called_once_with()

    # ------------------------------------------------------------------
    # add_layer (SSOT)
    # ------------------------------------------------------------------

    def test_add_layer_appends_to_registered(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.add_layer(3)
        mgr.add_layer(5)
        mgr.add_layer(7)
        self.assertEqual(mgr._registered_layers, [3, 5, 7])

    def test_add_layer_duplicate_is_idempotent(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.add_layer(3)
        mgr.add_layer(3)
        self.assertEqual(mgr._registered_layers, [3])

    # ------------------------------------------------------------------
    # setup(model) forwards to setup_dwdp with layer_indices SSOT
    # ------------------------------------------------------------------

    def test_setup_forwards_layer_indices(self, mock_comm_world, _rank, mock_setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_weight_manager = MagicMock()
        mock_setup.return_value = fake_weight_manager

        mapping = _make_mapping()
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=mapping,
        )
        # Registration order should be sorted before passing to setup_dwdp
        mgr.add_layer(7)
        mgr.add_layer(3)
        mgr.add_layer(5)

        fake_model = MagicMock()
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            result = mgr.setup(fake_model)

        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        self.assertIs(kwargs["model"], fake_model)
        self.assertIs(kwargs["mapping"], mapping)
        self.assertEqual(kwargs["device_id"], 0)
        self.assertIs(kwargs["comm"], sub_comm)
        self.assertEqual(kwargs["layer_indices"], [3, 5, 7])  # sorted SSOT
        # Config-driven expert range fields are forwarded from DwdpConfig.
        self.assertEqual(kwargs["num_experts_per_worker"], 4)
        self.assertEqual(kwargs["num_prefetch_experts"], 4)
        self.assertIs(result, fake_weight_manager)
        self.assertIs(mgr._weight_manager, fake_weight_manager)

    # ------------------------------------------------------------------
    # Runtime forwards require setup()
    # ------------------------------------------------------------------

    def test_prefetch_first_layers_forwards(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.first_moe_layer.return_value = 3
        fake_wm.next_moe_layer.return_value = 5
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        mgr.prefetch_first_layers()
        # First MoE layer 3 + ping-pong depth -> also prefetch next (5)
        self.assertEqual(
            fake_wm.prefetch_layer.call_args_list,
            [
                unittest.mock.call(3),
                unittest.mock.call(5),
            ],
        )

    def test_record_compute_and_prefetch_next_schedules_next_next(
        self, mock_comm_world, _rank, mock_setup
    ):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.side_effect = lambda layer_idx: {7: 9, 9: 11}.get(layer_idx)
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.record_compute_and_prefetch_next(7)
        fake_wm.record_compute.assert_called_once_with(7)
        self.assertEqual(
            fake_wm.next_moe_layer.call_args_list,
            [unittest.mock.call(7), unittest.mock.call(9)],
        )
        fake_wm.prefetch_layer.assert_called_once_with(11)

    def test_record_compute_and_prefetch_next_last_layer_is_noop(
        self, mock_comm_world, _rank, mock_setup
    ):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.return_value = None
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.record_compute_and_prefetch_next(60)
        fake_wm.record_compute.assert_called_once_with(60)
        fake_wm.prefetch_layer.assert_not_called()

    def test_prefetch_pipeline_schedules_each_layer_once(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.first_moe_layer.return_value = 3
        fake_wm.next_moe_layer.side_effect = lambda layer_idx: {
            3: 5,
            5: 7,
            7: 9,
            9: None,
        }.get(layer_idx)
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual(
            fake_wm.prefetch_layer.call_args_list,
            [
                unittest.mock.call(3),
                unittest.mock.call(5),
                unittest.mock.call(7),
                unittest.mock.call(9),
            ],
        )

    def _wrap_manager(self, mock_comm_world, mock_setup, layers, wrap="1"):
        """Manager over ``layers`` (MoE order) with slot = MoE position % 2, after setup().

        ``wrap`` is the ``TRTLLM_DWDP_WRAP_PREFETCH`` value (None: unset).
        """
        if wrap is not None:
            os.environ[WRAP_PREFETCH_ENV] = wrap
        _configure_mock_comm_world(mock_comm_world)
        fake_wm = MagicMock()
        fake_wm.first_moe_layer.return_value = layers[0]
        fake_wm.next_moe_layer.side_effect = dict(zip(layers, layers[1:])).get
        fake_wm.slot_index.side_effect = {layer: pos % 2 for pos, layer in enumerate(layers)}.get
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        return mgr, fake_wm

    def test_wrap_prefetch_primes_next_forward_into_released_slots(
        self, mock_comm_world, _rank, mock_setup
    ):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7, 9])

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.record_compute_and_prefetch_next(layer_idx)

        # 7 releases slot 0 -> first MoE layer (3); 9 releases slot 1 -> second (5).
        self.assertEqual(
            [c.args[0] for c in fake_wm.prefetch_layer.call_args_list], [3, 5, 7, 9, 3, 5]
        )
        self.assertEqual([c.args[0] for c in fake_wm.record_compute.call_args_list], [3, 5, 7, 9])

    def test_wrap_prefetch_skipped_for_odd_moe_layer_count(
        self, mock_comm_world, _rank, mock_setup
    ):
        # 3 MoE layers: 5 frees slot 1 but the first layer uses slot 0, which still holds 7.
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7])

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual([c.args[0] for c in fake_wm.prefetch_layer.call_args_list], [3, 5, 7])

    def test_wrap_prefetch_never_targets_the_layer_itself(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5])

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual([c.args[0] for c in fake_wm.prefetch_layer.call_args_list], [3, 5])

    def test_wrap_prefetch_off_by_env(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7, 9], wrap="0")

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual([c.args[0] for c in fake_wm.prefetch_layer.call_args_list], [3, 5, 7, 9])

    def test_wrap_prefetch_is_off_by_default(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7, 9], wrap=None)

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual([c.args[0] for c in fake_wm.prefetch_layer.call_args_list], [3, 5, 7, 9])
        fake_wm.slot_index.assert_not_called()

    @staticmethod
    def _slot_calls(fake_wm):
        return [
            (c[0], c.args[0])
            for c in fake_wm.method_calls
            if c[0] in ("record_compute", "prefetch_layer")
        ]

    def test_priming_after_an_aborted_forward_releases_both_slots_first(
        self, mock_comm_world, _rank, mock_setup
    ):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7, 9], wrap=None)
        mgr.prefetch_first_layers()
        mgr.wait_and_bind(MagicMock(), 3)
        mgr.record_compute_and_prefetch_next(3)
        # The forward stops after binding layer 5: its kernel never released slot 1.
        mgr.wait_and_bind(MagicMock(), 5)
        fake_wm.reset_mock()

        mgr.prefetch_first_layers()

        self.assertEqual(
            self._slot_calls(fake_wm),
            [
                ("record_compute", 3),
                ("record_compute", 5),
                ("prefetch_layer", 3),
                ("prefetch_layer", 5),
            ],
        )
        fake_wm.reset_mock()
        mgr.prefetch_first_layers()  # the barrier is one-shot
        self.assertEqual(self._slot_calls(fake_wm), [("prefetch_layer", 3), ("prefetch_layer", 5)])

    def test_priming_after_a_complete_forward_adds_no_barrier(
        self, mock_comm_world, _rank, mock_setup
    ):
        mgr, fake_wm = self._wrap_manager(mock_comm_world, mock_setup, [3, 5, 7, 9], wrap=None)
        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.wait_and_bind(MagicMock(), layer_idx)
            mgr.record_compute_and_prefetch_next(layer_idx)
        fake_wm.reset_mock()

        mgr.prefetch_first_layers()

        self.assertEqual(self._slot_calls(fake_wm), [("prefetch_layer", 3), ("prefetch_layer", 5)])

    def test_wait_and_bind_forwards(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        backend = MagicMock()
        mgr.wait_and_bind(backend, 5)
        fake_wm.wait_and_bind.assert_called_once_with(backend, 5)
        fake_wm.record_other_slot_consumed.assert_not_called()

    # ------------------------------------------------------------------
    # TRTLLM_DWDP_LEGACY_PREFETCH (one-layer protocol, A/B switch)
    # ------------------------------------------------------------------

    def test_legacy_prefetch_releases_other_slot_and_prefetches_next(
        self, mock_comm_world, _rank, mock_setup
    ):
        _configure_mock_comm_world(mock_comm_world)
        os.environ[LEGACY_PREFETCH_ENV] = "1"
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.side_effect = lambda layer_idx: {7: 9, 9: None}.get(layer_idx)
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        backend = MagicMock()
        mgr.wait_and_bind(backend, 7)
        fake_wm.wait_and_bind.assert_called_once_with(backend, 7)
        fake_wm.record_other_slot_consumed.assert_called_once_with(7)

        mgr.record_compute_and_prefetch_next(7)
        fake_wm.prefetch_layer.assert_called_once_with(9)
        fake_wm.record_compute.assert_not_called()

        # The last MoE layer records its own slot for the next forward's priming.
        mgr.record_compute_and_prefetch_next(9)
        fake_wm.prefetch_layer.assert_called_once_with(9)
        fake_wm.record_compute.assert_called_once_with(9)

    def test_legacy_prefetch_refused_with_dense_before_creating_comm(
        self, mock_comm_world, _rank, _setup
    ):
        os.environ[LEGACY_PREFETCH_ENV] = "1"
        os.environ[DENSE_ENV] = "1"

        with self.assertRaisesRegex(RuntimeError, LEGACY_PREFETCH_ENV):
            DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

        mock_comm_world.Create_group.assert_not_called()


# ----------------------------------------------------------------------
# Dense DWDP wiring
# ----------------------------------------------------------------------


class _FakeWeightManager:
    """The DWDPWeightManager calls DwdpManager makes, appended to a shared call log."""

    def __init__(self, log: list[tuple], layers: list[int]) -> None:
        self.log = log
        self.layers = list(layers)

    def first_moe_layer(self) -> int:
        return self.layers[0]

    def next_moe_layer(self, layer: int) -> int | None:
        return next((later for later in self.layers if later > layer), None)

    def prefetch_layer(self, layer: int) -> None:
        self.log.append(("E", layer))

    def record_compute(self, layer: int) -> None:
        self.log.append(("record_compute", layer))

    def wait_and_bind(self, backend: object, layer: int) -> None:
        self.log.append(("wait_and_bind", layer))

    def release(self) -> None:
        self.log.append(("wm.release",))


class _FakeDense:
    """The DenseDWDP calls DwdpManager makes, appended to the shared call log.

    The flags follow the runtime: ``war_barrier`` clears ``primed`` and
    ``forward_open``, ``mark_primed`` sets ``primed``.
    """

    def __init__(
        self,
        log: list[tuple],
        *,
        primed: bool = False,
        forward_open: bool = False,
        release_error: BaseException | None = None,
        eager_error: BaseException | None = None,
    ) -> None:
        self.log = log
        self.primed = primed
        self.forward_open = forward_open
        self.release_error = release_error
        self.eager_error = eager_error
        self.eager_checks: list[str] = []

    def require_eager(self, hook: str) -> None:
        self.eager_checks.append(hook)
        if self.eager_error is not None:
            raise self.eager_error

    def prefetch(self, layer: int) -> None:
        self.log.append(("L", layer))

    def mark_primed(self) -> None:
        self.log.append(("mark_primed",))
        self.primed = True

    def war_barrier(self) -> None:
        self.log.append(("war_barrier",))
        self.primed = False
        self.forward_open = False

    def release(self) -> None:
        self.log.append(("dense.release",))
        if self.release_error is not None:
            raise self.release_error


class _DenseModel(nn.Module):
    """A model the fake dense setup marks the way the real one does."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()


class _DenseManagerTest(unittest.TestCase):
    """A DwdpManager over fakes: one call log across dense, weight manager and teardown."""

    LAYERS = [3, 5, 7]
    #: Global rank 3 is DWDP rank 1 of group 1 and drives CUDA device 2: three different
    #: numbers, so passing one in place of another shows.
    GLOBAL_RANK = 3
    DWDP_RANK = 1
    DEVICE_ID = 2

    def setUp(self):
        set_global_dwdp_manager(None)
        self.addCleanup(set_global_dwdp_manager, None)
        self.log: list[tuple] = []
        sub_comm = _configure_mock_comm_world(_start(self, patch(f"{_FAC}.COMM_WORLD")))
        sub_comm.Free.side_effect = lambda: self.log.append(("group.Free",))
        self.sub_comm = sub_comm
        _start(self, patch(f"{_FAC}.dwdp_global_rank", return_value=self.GLOBAL_RANK))
        _start(self, patch(f"{_FAC}.torch.cuda.current_device", return_value=self.DEVICE_ID))
        self.weight_manager = _FakeWeightManager(self.log, self.LAYERS)
        self.setup_dwdp = _start(
            self, patch(f"{_FAC}.setup_dwdp", return_value=self.weight_manager)
        )
        self.teardown = _start(
            self,
            patch(
                f"{_FAC}.teardown_dwdp",
                side_effect=lambda model, expected_manager: self.log.append(("teardown_dwdp",)),
            ),
        )
        self.roots_available = _start(
            self, patch(f"{_FAC}.dense_roots_available", return_value=True)
        )
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp", return_value=None))
        _set_dense_env(self, None)
        self.model = _DenseModel()
        self.manager = DwdpManager(
            config=_make_config(num_groups=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_rank=self.DWDP_RANK),
        )
        for layer in reversed(self.LAYERS):
            self.manager.add_layer(layer)

    def setup_with_dense(self, dense: object) -> None:
        """Run ``DwdpManager.setup`` with a dense setup that commits ``dense``."""

        def commit(model, *_args, **_kwargs):
            model._dense_dwdp_active = True
            model.model._dense_dwdp_active = True
            return dense

        self.setup_dense.side_effect = commit
        self.manager.setup(self.model)


class TestDwdpManagerDenseSetup(_DenseManagerTest):
    def test_setup_runs_dense_setup_when_roots_are_available_but_not_requested(self):
        """Collective #1 runs on every rank of a model with roots, requested or not."""
        seen = {}

        def dense_setup(model, weight_manager, comm, **kwargs):
            seen["model_ref"] = self.manager._model_ref()
            seen["weight_manager"] = self.manager._weight_manager
            return None

        self.setup_dense.side_effect = dense_setup

        result = self.manager.setup(self.model)

        self.assertEqual(self.setup_dwdp.call_args.kwargs["device_id"], self.DEVICE_ID)
        self.roots_available.assert_called_once_with(self.model)
        self.setup_dense.assert_called_once_with(
            self.model,
            self.weight_manager,
            self.sub_comm,
            dwdp_rank=self.DWDP_RANK,
            dwdp_size=2,
            device_id=self.DEVICE_ID,
            moe_layers=[3, 5, 7],
        )
        # It runs after the expert setup and after the model reference is kept.
        self.assertIs(seen["model_ref"], self.model)
        self.assertIs(seen["weight_manager"], self.weight_manager)
        self.assertIs(result, self.weight_manager)
        self.assertIsNone(self.manager.dense)

    def _request_dense_without_roots(self, **switches: str) -> None:
        """A model without roots; ``TRTLLM_DWDP_DENSE=1`` plus ``switches`` (name suffixes)."""
        self.roots_available.return_value = False
        _set_dense_env(self, "1")
        for name in [name for name in os.environ if name.startswith(f"{DENSE_ENV}_")]:
            del os.environ[name]
        for suffix, value in switches.items():
            os.environ[f"{DENSE_ENV}_{suffix}"] = value

    def test_a_rank_requesting_dense_without_roots_raises_alone_and_at_once(self):
        """No rank makes a dense collective on a model without roots, so none waits for one.

        A requesting rank refuses by itself right after the expert setup: it raises unless its
        own ``TRTLLM_DWDP_DENSE_REQUIRED`` is off, and switches it cannot read count as required.
        """
        cases = ({}, {"REQUIRED": "1"}, {"REQUIRED": "maybe"}, {"REQUIRED": "0", "SEGMENTS": "atn"})
        for switches in cases:
            with self.subTest(**switches):
                self._request_dense_without_roots(**switches)
                with self.assertRaisesRegex(
                    DenseSetupError,
                    "TRTLLM_DWDP_DENSE asks for dense DWDP, but the decoder layers of _DenseModel "
                    "do not provide dense_dwdp_roots",
                ):
                    self.manager.setup(self.model)
                self.assertIsNone(self.manager.dense)
        self.assertEqual(self.setup_dwdp.call_count, len(cases))
        self.setup_dense.assert_not_called()
        self.sub_comm.allgather.assert_not_called()

        self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp",), ("group.Free",)])

    def test_a_rank_requesting_dense_without_roots_with_required_off_keeps_serving(self):
        self._request_dense_without_roots(REQUIRED="0")
        log_error = _start(self, patch(f"{_DENSE_SETUP}.logger.error"))

        result = self.manager.setup(self.model)

        self.assertIs(result, self.weight_manager)
        self.assertIsNone(self.manager.dense)
        self.setup_dense.assert_not_called()
        self.sub_comm.allgather.assert_not_called()
        (message,), _ = log_error.call_args
        self.assertIn("do not provide dense_dwdp_roots", message)
        self.assertIn("TRTLLM_DWDP_DENSE_REQUIRED=0, so dense DWDP is off on this rank", message)
        self.manager.prefetch_first_layers()
        self.assertEqual(self.log, [("E", 3), ("E", 5)])

    def test_setup_skips_dense_without_roots_or_request(self):
        """DeepSeek path: no dense setup, no collective, expert-only priming."""
        self.roots_available.return_value = False
        for value in (None, "0"):
            with self.subTest(TRTLLM_DWDP_DENSE=value):
                _set_dense_env(self, value)
                self.manager.setup(self.model)
                self.setup_dense.assert_not_called()
                self.assertIsNone(self.manager.dense)

        self.manager.prefetch_first_layers()

        self.assertEqual(self.log, [("E", 3), ("E", 5)])

    def test_dense_setup_error_propagates_and_cleanup_still_tears_down(self):
        self.setup_dense.side_effect = DenseSetupError("dense setup abandoned")

        with self.assertRaisesRegex(DenseSetupError, "dense setup abandoned"):
            self.manager.setup(self.model)
        self.assertIsNone(self.manager.dense)

        self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp",), ("group.Free",)])


class TestDwdpManagerRepeatedSetup(_DenseManagerTest):
    """``setup`` once dense DWDP is committed, after ``cleanup`` and on a model it already moved."""

    def test_a_second_setup_keeps_the_committed_runtime_without_any_collective(self):
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)

        for _ in range(2):
            self.assertIs(self.manager.setup(self.model), self.weight_manager)

        self.setup_dwdp.assert_called_once()
        self.setup_dense.assert_called_once()
        self.sub_comm.allgather.assert_not_called()
        self.assertIs(self.manager.dense, dense)
        self.assertEqual(self.log, [])
        with self.assertRaises(AttributeError):
            self.manager.dense = None

    def test_setup_for_another_model_is_refused_while_dense_is_committed(self):
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)

        with self.assertRaisesRegex(RuntimeError, "already committed for another model"):
            self.manager.setup(_DenseModel())

        self.setup_dwdp.assert_called_once()
        self.setup_dense.assert_called_once()
        self.assertIs(self.manager.dense, dense)
        self.assertIs(self.manager._model_ref(), self.model)
        self.assertEqual(self.log, [])

    def test_setup_after_cleanup_is_refused(self):
        self.setup_with_dense(_FakeDense(self.log))
        self.manager.cleanup()

        with self.assertRaisesRegex(RuntimeError, r"setup\(\) called after cleanup\(\)"):
            self.manager.setup(self.model)

        self.setup_dwdp.assert_called_once()
        self.setup_dense.assert_called_once()

    def test_a_model_whose_dense_weights_moved_is_refused_before_any_setup_work(self):
        """Without its runtime (released) the modules hold zero-size tensors: no expert setup."""
        for marked in (self.model, self.model.model):
            with self.subTest(marked=type(marked).__name__):
                marked._dense_dwdp_active = True
                with self.assertRaisesRegex(
                    RuntimeError, "already moved this model's dense weights"
                ):
                    self.manager.setup(self.model)
                del marked._dense_dwdp_active

        self.setup_dwdp.assert_not_called()
        self.setup_dense.assert_not_called()
        self.assertIsNone(self.manager._weight_manager)


class TestDwdpManagerDensePriming(_DenseManagerTest):
    def test_prefetch_first_layers_primes_dense_before_experts(self):
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)

        self.manager.prefetch_first_layers()

        self.assertEqual(self.log, [("L", 3), ("E", 3), ("L", 5), ("E", 5), ("mark_primed",)])
        self.assertTrue(dense.primed)

    def test_prefetch_first_layers_leaves_primed_or_open_dense_alone(self):
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)
        for primed, forward_open in ((True, False), (False, True)):
            with self.subTest(primed=primed, forward_open=forward_open):
                dense.primed, dense.forward_open = primed, forward_open

                self.manager.prefetch_first_layers()

                self.assertEqual(self.log, [])
                self.assertEqual((dense.primed, dense.forward_open), (primed, forward_open))

    def test_ensure_primed_primes_once(self):
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)

        self.manager.ensure_primed()
        self.manager.ensure_primed()
        self.manager.prefetch_first_layers()

        self.assertEqual(self.log, [("L", 3), ("E", 3), ("L", 5), ("E", 5), ("mark_primed",)])

    def test_ensure_primed_after_abort_records_barriers_before_repriming(self):
        dense = _FakeDense(self.log, forward_open=True)
        self.setup_with_dense(dense)

        self.manager.ensure_primed()

        self.assertEqual(
            self.log,
            [
                ("war_barrier",),
                ("record_compute", 3),
                ("record_compute", 5),
                ("L", 3),
                ("E", 3),
                ("L", 5),
                ("E", 5),
                ("mark_primed",),
            ],
        )
        self.assertTrue(dense.primed)
        self.assertFalse(dense.forward_open)

    def test_single_moe_layer_primes_and_recovers_one_layer(self):
        self.weight_manager.layers = [3]
        dense = _FakeDense(self.log)
        self.setup_with_dense(dense)

        self.manager.prefetch_first_layers()
        self.assertEqual(self.log, [("L", 3), ("E", 3), ("mark_primed",)])

        self.log.clear()
        dense.primed, dense.forward_open = False, True
        self.manager.ensure_primed()
        self.assertEqual(
            self.log,
            [("war_barrier",), ("record_compute", 3), ("L", 3), ("E", 3), ("mark_primed",)],
        )

    def test_ensure_primed_checks_eager_execution_before_anything_else(self):
        dense = _FakeDense(self.log, forward_open=True)
        self.setup_with_dense(dense)

        self.manager.ensure_primed()

        self.assertEqual(dense.eager_checks, ["DwdpManager.ensure_primed (H0)"])
        self.assertEqual(self.log[0], ("war_barrier",))

    def test_ensure_primed_refused_records_and_enqueues_nothing(self):
        self.setup_with_dense(_FakeDense(self.log))
        for primed, forward_open in ((False, True), (False, False), (True, False)):
            with self.subTest(primed=primed, forward_open=forward_open):
                self.log.clear()
                dense = _FakeDense(
                    self.log,
                    primed=primed,
                    forward_open=forward_open,
                    eager_error=DenseProtocolError("capturing a CUDA graph"),
                )
                self.manager._dense = dense

                with self.assertRaisesRegex(DenseProtocolError, "capturing a CUDA graph"):
                    self.manager.ensure_primed()

                self.assertEqual(self.log, [])
                self.assertEqual((dense.primed, dense.forward_open), (primed, forward_open))

    def test_ensure_primed_requires_dense(self):
        self.roots_available.return_value = False
        self.manager.setup(self.model)

        with self.assertRaisesRegex(RuntimeError, "dense DWDP"):
            self.manager.ensure_primed()


class TestDwdpManagerDenseCleanup(_DenseManagerTest):
    def test_cleanup_releases_dense_before_expert_teardown(self):
        self.setup_with_dense(_FakeDense(self.log))

        self.manager.cleanup()
        self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("teardown_dwdp",), ("group.Free",)])
        self.assertIsNone(self.manager.dense)
        # The modules hold zero-size tensors from now on: reloads stay refused.
        self.assertTrue(self.model._dense_dwdp_active)
        self.assertTrue(self.model.model._dense_dwdp_active)

    def test_failing_dense_release_still_tears_down_experts_and_reraises(self):
        self.setup_with_dense(_FakeDense(self.log, release_error=RuntimeError("detach failed")))

        with self.assertRaisesRegex(RuntimeError, "detach failed"):
            self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("teardown_dwdp",), ("group.Free",)])
        self.assertIsNone(self.manager.dense)
        self.assertIsNone(self.manager._weight_manager)
        self.assertIsNone(self.manager.dwdp_group)

    def test_dense_release_error_wins_over_a_later_teardown_error(self):
        self.setup_with_dense(_FakeDense(self.log, release_error=RuntimeError("detach failed")))
        self.teardown.side_effect = RuntimeError("unmap failed")

        with self.assertRaisesRegex(RuntimeError, "detach failed"):
            self.manager.cleanup()

        self.teardown.assert_called_once_with(self.model, expected_manager=self.weight_manager)
        self.assertEqual(self.log, [("dense.release",), ("group.Free",)])

    def test_keyboard_interrupt_in_dense_release_still_tears_down_experts(self):
        """The expert transport owns MNNVL handles that outlive the process."""
        self.setup_with_dense(_FakeDense(self.log, release_error=KeyboardInterrupt()))

        with self.assertRaises(KeyboardInterrupt):
            self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("teardown_dwdp",), ("group.Free",)])

    def test_failing_group_free_after_a_dense_release_error_is_logged(self):
        """The dense release error is raised; the Free failure after it is logged through ``error``."""
        self.setup_with_dense(_FakeDense(self.log, release_error=RuntimeError("detach failed")))
        self.sub_comm.Free.side_effect = RuntimeError("free failed")
        # Only ``error`` is replaced: any other logger method the path calls must exist.
        log_error = _start(self, patch(f"{_FAC}.logger.error"))

        with self.assertRaisesRegex(RuntimeError, "detach failed"):
            self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("teardown_dwdp",)])
        self.sub_comm.Free.assert_called_once_with()
        (message,), _ = log_error.call_args
        self.assertIn("Failed to free DWDP group", message)
        self.assertIn("free failed", message)
        self.assertIsNone(self.manager.dense)
        self.assertIsNone(self.manager._weight_manager)
        self.assertIsNone(self.manager.dwdp_group)

    def test_repoint_error_with_the_model_alive_is_raised(self):
        self.setup_with_dense(
            _FakeDense(self.log, release_error=DenseRepointError("an owner lost its binding"))
        )

        with self.assertRaisesRegex(DenseRepointError, "an owner lost its binding"):
            self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("teardown_dwdp",), ("group.Free",)])

    def test_cleanup_without_model_releases_dense_then_weight_manager(self):
        self.setup_with_dense(_FakeDense(self.log))
        self.manager._model_ref = weakref.ref(nn.Module())  # the model is gone

        self.manager.cleanup()

        self.assertEqual(self.log, [("dense.release",), ("wm.release",), ("group.Free",)])

    def _check_cleanup_without_model_raises(self, error: Exception) -> None:
        self.setup_with_dense(_FakeDense(self.log, release_error=error))
        self.manager._model_ref = weakref.ref(nn.Module())  # the model is gone
        warning = _start(self, patch(f"{_FAC}.logger.warning"))

        with self.assertRaises(type(error)) as raised:
            self.manager.cleanup()

        self.assertIs(raised.exception, error)
        self.assertEqual(self.log, [("dense.release",), ("wm.release",), ("group.Free",)])
        warning.assert_not_called()

    def test_cleanup_without_model_raises_a_repoint_error(self):
        """A collected model excuses no repoint error: the release skips its bindings itself."""
        self._check_cleanup_without_model_raises(DenseRepointError("a live owner lost a binding"))

    def test_cleanup_without_model_raises_other_dense_release_errors(self):
        self._check_cleanup_without_model_raises(RuntimeError("synchronize failed"))


# ----------------------------------------------------------------------
# DwdpManager driving the real dense DWDP runtime
# ----------------------------------------------------------------------


class _Stream:
    def __init__(self, name: str) -> None:
        self.name = name

    def wait_event(self, event: object) -> None:
        pass


class _Event:
    def __init__(self, ops: "_RecordingOps") -> None:
        self.ops = ops

    def record(self, stream: _Stream | None = None) -> None:
        stream = self.ops.current if stream is None else stream
        self.ops.log.append(("record", stream.name))


class _RecordingOps:
    """CPU ``DeviceOps`` of the runtime: event records go to the shared call log."""

    def __init__(self, log: list[tuple]) -> None:
        self.log = log
        self.compute = _Stream("compute")
        self.copy = _Stream("copy")
        self.current = self.compute
        self.capturing = False
        self.multi_stream = False

    def new_event(self) -> _Event:
        return _Event(self)

    def current_stream(self) -> _Stream:
        return self.current

    @contextlib.contextmanager
    def stream(self, stream: _Stream):
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def synchronize(self) -> None:
        pass

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def do_multi_stream(self) -> bool:
        return self.multi_stream

    @contextlib.contextmanager
    def nvtx_range(self, message: str):
        yield


class TestDwdpManagerWithDenseRuntime(_DenseManagerTest):
    """``_prime_all``, ``ensure_primed`` and ``prefetch_first_layers`` against ``DenseDWDP``."""

    LAYERS = [3, 5, 7, 9]

    def setUp(self):
        super().setUp()
        ops = _RecordingOps(self.log)
        self.ops = ops
        slots = {
            kind: allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2), "cpu")
            for kind in (SegmentKind.ATTN, SegmentKind.MOE)
        }
        self.dense = DenseDWDP(
            self.LAYERS, (SegmentKind.ATTN, SegmentKind.MOE), slots, ops.copy, ops=ops
        )
        self.dense.set_launcher(lambda layer: self.log.append(("L", layer)))
        self.dense.commit(None, fill_byte=0)
        self.setup_with_dense(self.dense)
        self.addCleanup(self.dense.release)
        self.log.clear()

    def run_forward(self, *, stop_after_attention: int | None = None) -> None:
        """The model and ConfigurableMoE calls of one forward, from H0 to H3 of the last layer."""
        self.manager.ensure_primed()
        for layer in self.LAYERS:
            self.dense.wait_layer(layer)
            self.dense.attention_done(layer)
            if layer == stop_after_attention:
                return
            self.manager.wait_and_bind(MagicMock(), layer)
            self.manager.record_compute_and_prefetch_next(layer)
            self.dense.layer_done(layer)

    def launches(self) -> list[tuple]:
        return [entry for entry in self.log if entry[0] in ("L", "E")]

    def test_executor_priming_satisfies_mark_primed(self):
        self.manager.prefetch_first_layers()

        self.assertEqual(self.launches(), [("L", 3), ("E", 3), ("L", 5), ("E", 5)])
        self.assertTrue(self.dense.primed)
        for kind in (SegmentKind.ATTN, SegmentKind.MOE):
            self.assertEqual(self.dense.slot_tag(kind, self.dense.slot_index(kind, 3)), 3)
            self.assertEqual(self.dense.slot_tag(kind, self.dense.slot_index(kind, 5)), 5)

        self.log.clear()
        self.manager.prefetch_first_layers()
        self.manager.ensure_primed()
        self.assertEqual(self.log, [])

    def test_forwards_without_executor_priming_prime_in_the_model(self):
        """Warmup-like forwards: H0 primes each one; the hooks never demand-fetch."""
        for _ in range(2):
            self.run_forward()
            self.assertFalse(self.dense.forward_open)
            self.assertFalse(self.dense.primed)

        self.assertEqual(
            [entry for entry in self.launches() if entry in (("L", 3), ("L", 5))],
            [("L", 3), ("L", 5)] * 2,
        )
        self.assertEqual((self.dense.demand_fetches, self.dense.war_fences), (0, 0))

    def test_aborted_forward_is_recovered_by_the_model(self):
        self.manager.prefetch_first_layers()
        self.run_forward(stop_after_attention=5)
        self.assertTrue(self.dense.forward_open)

        self.log.clear()
        self.manager.prefetch_first_layers()  # the executor loop leaves it to the model
        self.assertEqual(self.log, [])

        self.manager.ensure_primed()

        dense_consume_events = sum(RING_DEPTH[kind] for kind in self.dense.segments)
        self.assertEqual(
            self.log[:dense_consume_events], [("record", "compute")] * dense_consume_events
        )
        self.assertEqual(
            [entry for entry in self.log if entry[0] != "record"],
            [
                ("record_compute", 3),
                ("record_compute", 5),
                ("L", 3),
                ("E", 3),
                ("L", 5),
                ("E", 5),
            ],
        )
        self.assertTrue(self.dense.primed)
        self.assertFalse(self.dense.forward_open)

        self.run_forward()
        self.assertEqual((self.dense.demand_fetches, self.dense.war_fences), (0, 0))

    def test_a_refused_capture_after_an_abort_leaves_the_barrier_to_the_next_forward(self):
        """H0 raises inside a CUDA graph capture before the barrier or the priming."""
        self.manager.prefetch_first_layers()
        self.run_forward(stop_after_attention=5)
        self.assertTrue(self.dense.forward_open)
        tags = [
            self.dense.slot_tag(kind, index)
            for kind in self.dense.segments
            for index in range(RING_DEPTH[kind])
        ]
        for flag, message in (("capturing", "capturing a CUDA graph"), ("multi_stream", "multi")):
            with self.subTest(flag=flag):
                self.log.clear()
                setattr(self.ops, flag, True)
                with self.assertRaisesRegex(DenseProtocolError, message):
                    self.manager.ensure_primed()
                setattr(self.ops, flag, False)
                self.assertEqual(self.log, [])
                self.assertTrue(self.dense.forward_open)
                self.assertFalse(self.dense.primed)
                self.assertEqual(
                    [
                        self.dense.slot_tag(kind, index)
                        for kind in self.dense.segments
                        for index in range(RING_DEPTH[kind])
                    ],
                    tags,
                )

        self.manager.ensure_primed()

        self.assertEqual(
            [entry for entry in self.log if entry[0] != "record"],
            [
                ("record_compute", 3),
                ("record_compute", 5),
                ("L", 3),
                ("E", 3),
                ("L", 5),
                ("E", 5),
            ],
        )
        self.run_forward()
        self.assertEqual((self.dense.demand_fetches, self.dense.war_fences), (0, 0))


class TestDwdpManagerCleanupOfCommittedBindings(_DenseManagerTest):
    """``cleanup`` against a committed ``DenseDWDP`` whose ``DenseBindings`` repointed a toy model."""

    LAYERS = list(COVERED)

    def setUp(self):
        super().setUp()
        model = ToyModel()
        layout, slots, bindings = setup_bindings(model, allocate=allocated_ring)
        bindings.repoint()
        bindings.commit(strict=False)
        ops = _RecordingOps([])
        self.dense = DenseDWDP(layout.layers, layout.segment_kinds, slots, ops.copy, ops=ops)
        self.dense.set_launcher(lambda layer: None)
        self.dense.commit(bindings, fill_byte=0)
        self.addCleanup(self.dense.release)
        self.bindings = [binding for anchor in layout.anchors() for binding in anchor.bindings]
        self.slot_refs = [
            weakref.ref(slot.storage.untyped_storage()) for ring in slots.values() for slot in ring
        ]
        del slots
        self.setup_dense.return_value = self.dense
        self.manager.setup(model)
        # The mocks keep their call arguments; after this only ``self.toy_model`` holds the model.
        for mock in (self.setup_dwdp, self.roots_available, self.setup_dense):
            mock.reset_mock()
        self.toy_model = model
        self.warning = _start(self, patch(f"{_FAC}.logger.warning"))
        self.error = _start(self, patch(f"{_FAC}.logger.error"))

    def collect_the_model(self) -> None:
        """Drop the model, as ``PyExecutor.shutdown`` does before it exits the manager."""
        model_ref = weakref.ref(self.toy_model)
        del self.toy_model
        gc.collect()
        self.assertIsNone(model_ref())

    def assert_slots_freed(self) -> None:
        self.assertIsNone(self.manager.dense)
        self.assertEqual(dict(self.dense.slots), {})
        gc.collect()
        self.assertEqual([ref for ref in self.slot_refs if ref() is not None], [])

    def test_cleanup_detaches_every_binding_before_the_expert_teardown(self):
        def teardown(model, expected_manager):
            sizes = {read_binding(binding).numel() for binding in self.bindings}
            self.log.append(("teardown_dwdp", sizes))

        self.teardown.side_effect = teardown

        self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp", {0}), ("group.Free",)])
        self.assert_slots_freed()
        self.warning.assert_not_called()
        self.error.assert_not_called()

    def test_cleanup_after_the_model_was_collected_detaches_cleanly(self):
        """The bindings of the collected modules count as detached: nothing raises or warns."""
        self.collect_the_model()
        self.assertTrue(all(binding.owner_ref() is None for binding in self.bindings))

        self.manager.cleanup()

        self.assertEqual(self.log, [("wm.release",), ("group.Free",)])
        self.assert_slots_freed()
        self.warning.assert_not_called()
        self.error.assert_not_called()

    def test_cleanup_raises_for_a_live_module_that_lost_its_binding(self):
        """A module that survives the model and no longer holds its binding is a real failure.

        Its other bindings are detached and the collected modules' bindings are skipped; the
        expert teardown and the group free still run before the error is raised.
        """
        survivor = self.toy_model.layers[2].linear_attn
        del survivor._bfa_proj_weight
        self.collect_the_model()

        with self.assertRaisesRegex(
            DenseRepointError,
            r"^detach could not reach every binding: layers\.2\.linear_attn\._bfa_proj_weight: "
            r"ToyKDA at linear_attn no longer holds [^;]*$",
        ):
            self.manager.cleanup()

        self.assertEqual(self.log, [("wm.release",), ("group.Free",)])
        survivor_bindings = [
            binding for binding in self.bindings if binding.owner_ref() is not None
        ]
        self.assertTrue(survivor_bindings)
        for binding in survivor_bindings:
            if binding.name != "_bfa_proj_weight":
                self.assertEqual(read_binding(binding).numel(), 0, binding.qualified_name)
        self.assert_slots_freed()
        self.warning.assert_not_called()
        self.error.assert_called_once()

    def test_cleanup_with_the_model_alive_raises_a_lost_binding(self):
        self.toy_model.layers[1].block_sparse_moe.gate._parameters.pop("weight")

        with self.assertRaisesRegex(
            DenseRepointError,
            r"layers\.1\.block_sparse_moe\.gate\.weight: ToyGate at block_sparse_moe\.gate no "
            r"longer holds",
        ):
            self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp",), ("group.Free",)])
        self.assert_slots_freed()


if __name__ == "__main__":
    unittest.main()
