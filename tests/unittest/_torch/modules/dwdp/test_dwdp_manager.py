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

These tests mock out COMM_WORLD / dwdp_global_rank / setup_dwdp so the full
lifecycle can be exercised on a single CPU without MPI / CUDA / MNNVL.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _make_config(dwdp_size: int = 2) -> DwdpConfig:
    return DwdpConfig(
        dwdp_size=dwdp_size,
        num_groups=1,
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


@patch("tensorrt_llm._torch.pyexecutor.dwdp.setup_dwdp")
@patch("tensorrt_llm._torch.pyexecutor.dwdp.dwdp_global_rank", return_value=0)
@patch("tensorrt_llm._torch.pyexecutor.dwdp.COMM_WORLD")
class TestDwdpManagerLifecycle(unittest.TestCase):
    def setUp(self):
        # Ensure no leftover global singleton from a prior test.
        set_global_dwdp_manager(None)

    def tearDown(self):
        set_global_dwdp_manager(None)

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

    def test_enter_reserves_the_gather_sms(self, mock_comm_world, _rank, _setup):
        """The gather's resident SMs reach the persistent grids through set_sm_reserve, from
        __enter__ (before the model is built) to __exit__."""
        _configure_mock_comm_world(mock_comm_world)
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.fabric_gather.resident_sms", return_value=30
        ):
            mgr = DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )
        self.assertEqual(mgr.resident_sms, 30)
        self.assertEqual(sm_reserve(), 0)
        mgr.__enter__()
        self.assertEqual(sm_reserve(), 30)
        mgr.__exit__(None, None, None)
        self.assertEqual(sm_reserve(), 0)

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
        self.assertEqual(kwargs["num_prefetch_slots"], 2)
        self.assertIs(result, fake_weight_manager)
        self.assertIs(mgr._weight_manager, fake_weight_manager)
        self.assertEqual(kwargs["repeated_layer_indices"], [])

    def test_setup_forwards_repeated_dwdp_layers(self, mock_comm_world, _rank, _setup):
        """Repeated layers reach setup_dwdp; layers without a DWDP MoE are dropped."""
        mgr, _ = self._set_up_manager(
            mock_comm_world,
            _setup,
            before_setup=lambda m: (
                m.add_layer(3),
                m.add_layer(61),
                m.add_repeated_layer(61),
                m.add_repeated_layer(61),
                m.add_repeated_layer(62),
            ),
        )
        _, kwargs = _setup.call_args
        self.assertEqual(kwargs["layer_indices"], [3, 61])
        self.assertEqual(kwargs["repeated_layer_indices"], [61])

    # ------------------------------------------------------------------
    # Runtime forwards require setup()
    # ------------------------------------------------------------------

    def _set_up_manager(self, mock_comm_world, mock_setup, before_setup=None):
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
        if before_setup is not None:
            before_setup(mgr)
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        return mgr, fake_wm

    def test_start_forward_forwards(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._set_up_manager(mock_comm_world, mock_setup)
        mgr.start_forward()
        fake_wm.start_forward.assert_called_once_with()

    def test_finish_layer_forwards(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._set_up_manager(mock_comm_world, mock_setup)
        mgr.finish_layer(7)
        fake_wm.finish_layer.assert_called_once_with(7)

    def test_invalidate_prefetch_forwards(self, mock_comm_world, _rank, mock_setup):
        mgr, fake_wm = self._set_up_manager(mock_comm_world, mock_setup)
        mgr.invalidate_prefetch()
        fake_wm.invalidate_prefetch.assert_called_once_with()

    def test_invalidate_prefetch_before_setup_is_noop(self, mock_comm_world, _rank, _setup):
        _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.invalidate_prefetch()
        with self.assertRaises(RuntimeError):
            mgr.start_forward()

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


if __name__ == "__main__":
    unittest.main()
