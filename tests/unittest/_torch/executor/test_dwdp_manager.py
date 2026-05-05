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

These tests mock out COMM_WORLD / global_mpi_rank / setup_dwdp so the full
lifecycle can be exercised on a single CPU without MPI / CUDA / MNNVL.
"""
import unittest
from unittest.mock import MagicMock, patch

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping


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


def _make_mock_comm_world():
    """Build a mpi4py-style COMM_WORLD mock that yields a mock sub-comm."""
    comm_world = MagicMock()
    sub_comm = MagicMock()
    comm_world.Create_group.return_value = sub_comm
    comm_world.group.Incl.return_value = MagicMock()  # a group handle
    comm_world.Get_size.return_value = 1024
    return comm_world, sub_comm


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
@patch("tensorrt_llm._torch.pyexecutor.dwdp.global_mpi_rank", return_value=0)
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

    def test_init_rejects_non_mpi_dist(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        with self.assertRaises(RuntimeError):
            DwdpManager(
                config=_make_config(),
                dist=object(),  # not an MPIDist
                mapping=_make_mapping(),
            )

    def test_init_rejects_non_dwdp_mapping(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        non_dwdp_mapping = Mapping(world_size=1, rank=0, tp_size=1)
        with self.assertRaises(RuntimeError):
            DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=non_dwdp_mapping,
            )

    # ------------------------------------------------------------------
    # num_groups topology validation (was a dead schema field through 1fbc0d49)
    # ------------------------------------------------------------------

    def test_init_rejects_non_positive_num_groups(self, mock_comm_world, _rank, _setup):
        _configure_mock_comm_world(mock_comm_world)
        bad_config = DwdpConfig(
            dwdp_size=2,
            num_groups=0,  # invalid
            num_experts_per_worker=4,
            num_prefetch_experts=4,
        )
        with self.assertRaisesRegex(ValueError, "num_groups must be positive"):
            DwdpManager(
                config=bad_config,
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

    def test_init_rejects_group_id_exceeding_num_groups(
            self, mock_comm_world, mock_rank, _setup):
        # rank=4, dwdp_size=2 -> group_id=2; with num_groups=2 group_id must be < 2.
        # Override the class-level rank mock to test out-of-range group_id.
        mock_rank.return_value = 4
        _configure_mock_comm_world(mock_comm_world)
        with self.assertRaisesRegex(ValueError, "group_id=2"):
            DwdpManager(
                config=DwdpConfig(
                    dwdp_size=2,
                    num_groups=2,  # only allows group_ids 0, 1
                    num_experts_per_worker=4,
                    num_prefetch_experts=4,
                ),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

    def test_init_rejects_world_smaller_than_topology(self, mock_comm_world, _rank, _setup):
        # num_groups=4, dwdp_size=2 declares 8 CTX workers, but world has only 2.
        _configure_mock_comm_world(mock_comm_world, world_size=2)
        with self.assertRaisesRegex(ValueError, "MPI world size is only 2"):
            DwdpManager(
                config=DwdpConfig(
                    dwdp_size=2,
                    num_groups=4,
                    num_experts_per_worker=4,
                    num_prefetch_experts=4,
                ),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

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

    def test_duplicate_enter_raises(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr1 = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr1.__enter__()
        mgr2 = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with self.assertRaises(RuntimeError):
            mgr2.__enter__()
        mgr1.__exit__(None, None, None)

    def test_cleanup_frees_comm_and_idempotent(
            self, mock_comm_world, _rank, _setup):
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

    # ------------------------------------------------------------------
    # add_layer (SSOT)
    # ------------------------------------------------------------------

    def test_add_layer_appends_to_registered(
            self, mock_comm_world, _rank, _setup):
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

    def test_add_layer_duplicate_is_idempotent(
            self, mock_comm_world, _rank, _setup):
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

    def test_setup_forwards_layer_indices(
            self, mock_comm_world, _rank, mock_setup):
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

    def test_runtime_before_setup_raises(
            self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with self.assertRaises(RuntimeError):
            mgr.prefetch_first_layers()
        with self.assertRaises(RuntimeError):
            mgr.wait_and_bind(MagicMock(), 3)
        with self.assertRaises(RuntimeError):
            mgr.record_compute_and_prefetch_next(3)

    def test_prefetch_first_layers_forwards(
            self, mock_comm_world, _rank, mock_setup):
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
        self.assertEqual(fake_wm.prefetch_layer.call_args_list, [
            unittest.mock.call(3),
            unittest.mock.call(5),
        ])

    def test_record_compute_and_prefetch_next_schedules_next(
            self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.return_value = 9
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
        fake_wm.next_moe_layer.assert_called_once_with(7)
        fake_wm.prefetch_layer.assert_called_once_with(9)

    def test_record_compute_and_prefetch_next_last_layer_is_noop(
            self, mock_comm_world, _rank, mock_setup):
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
        fake_wm.prefetch_layer.assert_not_called()

    def test_wait_and_bind_forwards(
            self, mock_comm_world, _rank, mock_setup):
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
