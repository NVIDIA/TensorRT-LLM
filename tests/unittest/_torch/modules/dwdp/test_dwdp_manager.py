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
- resolves the expert partition each MoE backend is sized from
- owns the Single Source of Truth for registered MoE layer indices
- forwards runtime calls to the DWDPWeightManager produced by setup_dwdp
- owns the attached DenseDwdpManager: sets dense DWDP up before setup_dwdp

Also here: ConfigurableMoE's decision, once its backend exists, whether DWDP manages a MoE layer;
a layer it does not manage keeps every expert.

These tests mock out COMM_WORLD / dwdp_global_rank / setup_dwdp so the full
lifecycle can be exercised on a single CPU without MPI / CUDA / MNNVL.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    H0_HOOK,
    DenseDwdpManager,
    get_global_dense_dwdp_manager,
    set_global_dense_dwdp_manager,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, DenseSetupError
from tensorrt_llm._torch.modules.dwdp.specs import ExpertPartition
from tensorrt_llm._torch.modules.dwdp.weight_buffer import PrefetchRingMemoryError
from tensorrt_llm._torch.moe.fused_moe import configurable_moe
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.interface import MoE
from tensorrt_llm._torch.moe.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    _AttachedTransfer,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FAC = "tensorrt_llm._torch.pyexecutor.dwdp"


def _make_config(dwdp_size: int = 2, num_groups: int = 1, enable_dense: bool = False) -> DwdpConfig:
    return DwdpConfig(
        enable_dense=enable_dense,
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


def _start(test: unittest.TestCase, patcher: object) -> MagicMock:
    """Start ``patcher`` for the rest of ``test`` and return its mock."""
    mock = patcher.start()
    test.addCleanup(patcher.stop)
    return mock


def _clear_global_managers() -> None:
    """Unregister both managers: a leftover one fails the next ``__init__``."""
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


@patch("tensorrt_llm._torch.pyexecutor.dwdp.setup_dwdp")
@patch("tensorrt_llm._torch.pyexecutor.dwdp.dwdp_global_rank", return_value=0)
@patch("tensorrt_llm._torch.pyexecutor.dwdp.COMM_WORLD")
class TestDwdpManagerLifecycle(unittest.TestCase):
    def setUp(self):
        # Ensure no leftover global singleton from a prior test.
        _clear_global_managers()
        # Dense DWDP off: a model without dense roots, enable_dense unset.
        _start(self, patch(f"{_FAC}.dense_roots_available", return_value=False))
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp"))

    def tearDown(self):
        _clear_global_managers()
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

    def test_init_stores_fields(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
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
        # rank=0 of a uniform 4/4 partition of 8 experts: [0, 4).
        self.assertEqual(mgr.local_expert_range(8), (0, 4))

    def test_init_expert_range_uniform(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        """Uniform partition (size == stride): each rank's range is a clean slice."""
        _configure_mock_comm_world(mock_comm_world)
        # dwdp_rank=1 with stride=4 -> start=4, end=4+4=8.
        mgr = DwdpManager(
            config=_make_config(dwdp_size=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=1),
        )
        self.assertEqual(mgr.local_expert_range(8), (4, 8))

    def test_init_expert_range_redundancy(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        """Redundancy partition (size > stride): adjacent ranges overlap."""
        _configure_mock_comm_world(mock_comm_world)
        redundant_config = DwdpConfig(
            dwdp_size=2,
            num_groups=1,
            num_experts_per_worker=6,  # storage size
            num_prefetch_experts=4,  # stride < size -> 2-expert overlap
        )
        # rank=0: [0, 6). rank=1: [4, 10) -> 2-expert overlap [4, 6).
        mgr = DwdpManager(
            config=redundant_config,
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=0),
        )
        self.assertEqual(mgr.expert_partition(10).peer_ranges, [(0, 6), (4, 10)])
        # Stored config knobs match what's plumbed into setup_dwdp.
        self.assertEqual(mgr.num_experts_per_worker, 6)
        self.assertEqual(mgr.num_prefetch_experts, 4)

    # ------------------------------------------------------------------
    # Expert partition: DwdpConfig -> the range each MoE backend is sized from
    # ------------------------------------------------------------------

    def _k3_manager(
        self, mock_comm_world: MagicMock, size: int, stride: int, dwdp_rank: int
    ) -> DwdpManager:
        """DWDP rank ``dwdp_rank`` of 12 with the given partition fields."""
        _configure_mock_comm_world(mock_comm_world)
        config = DwdpConfig(dwdp_size=12, num_experts_per_worker=size, num_prefetch_experts=stride)
        mapping = _make_mapping(dwdp_size=12, dwdp_rank=dwdp_rank)
        return DwdpManager(config=config, dist=MagicMock(spec=MPIDist), mapping=mapping)

    def test_config_refuses_a_half_set_or_negative_partition(self, *_mocks: MagicMock) -> None:
        for size, stride in ((0, 74), (82, 0), (-1, -1), (-1, 4)):
            with self.subTest(size=size, stride=stride), self.assertRaises(ValueError):
                DwdpConfig(dwdp_size=12, num_experts_per_worker=size, num_prefetch_experts=stride)

    def test_balanced_partition_is_resolved_per_expert_count(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        mgr = self._k3_manager(mock_comm_world, 0, 0, dwdp_rank=8)
        self.assertEqual(mgr.local_expert_range(896), (600, 674))  # 8 x 75 + 4 x 74
        self.assertIs(mgr.expert_partition(896), mgr.expert_partition(896))
        self.assertEqual(mgr.local_expert_range(72), (48, 54))

    def test_moe_backend_is_sized_from_the_rank_range(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        """``MoE._init_dwdp_expert_layout`` sizes a backend with exactly its rank's rows."""
        # balanced, capped (the last rank stores 71), redundant
        cases = (((0, 0), 8, (600, 674)), ((75, 75), 11, (825, 896)), ((82, 74), 11, (814, 896)))
        for (size, stride), rank, (start, end) in cases:
            with self.subTest(size=size, stride=stride):
                set_global_dwdp_manager(self._k3_manager(mock_comm_world, size, stride, rank))
                moe = SimpleNamespace(num_experts=896, layer_load_balancer=None)
                MoE._init_dwdp_expert_layout(moe)
                set_global_dwdp_manager(None)

                self.assertEqual(
                    (moe.num_slots, moe.expert_size_per_partition, moe.slot_start, moe.slot_end),
                    (896, end - start, start, end),
                )
                self.assertEqual(moe.initial_local_expert_ids, list(range(start, end)))
                self.assertEqual(moe.initial_global_assignments, list(range(896)))
        set_global_dwdp_manager(self._k3_manager(mock_comm_world, 82, 74, dwdp_rank=0))
        with self.assertRaises(ValueError):  # 82/74 does not fit 900 experts
            MoE._init_dwdp_expert_layout(SimpleNamespace(num_experts=900, layer_load_balancer=None))

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
        self.assertIsNone(get_global_dense_dwdp_manager())
        mgr.__enter__()
        self.assertIs(get_global_dwdp_manager(), mgr)
        self.assertIs(get_global_dense_dwdp_manager(), mgr.dense_manager)
        self.assertIs(mgr.dense_manager.mode, DenseMode.ATTACHED)
        mgr.__exit__(None, None, None)
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIsNone(get_global_dense_dwdp_manager())

    def test_enter_frees_the_group_when_the_dense_manager_cannot_register(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        """Neither stays registered; a failing Free is logged and the registration error raised."""
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        sub_comm.Free.side_effect = RuntimeError("free failed")
        mgr = DwdpManager(
            config=_make_config(), dist=MagicMock(spec=MPIDist), mapping=_make_mapping()
        )
        stray = DenseDwdpManager(DenseMode.ATTACHED).__enter__()

        with patch(f"{_FAC}.logger.error") as log_error:
            with self.assertRaisesRegex(
                RuntimeError, "^DenseDwdpManager already registered globally$"
            ):
                mgr.__enter__()

        sub_comm.Free.assert_called_once_with()
        self.assertIn("free failed", log_error.call_args.args[0])
        self.assertIsNone(mgr.dwdp_group)
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIs(get_global_dense_dwdp_manager(), stray)
        with self.assertRaisesRegex(RuntimeError, r"setup\(\) called after cleanup\(\)"):
            mgr.setup(MagicMock())

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

    def test_copy_mode_selects_the_expert_transfer_through_the_config(
        self, mock_comm_world: MagicMock, _rank: MagicMock, mock_setup: MagicMock
    ) -> None:
        """copy_mode reaches setup_dwdp and decides the gather's SM reserve; it overrides
        TRTLLM_DWDP_PREFETCH_GATHER, with one warning when the two disagree."""
        _configure_mock_comm_world(mock_comm_world)
        cases = [
            # (copy_mode, TRTLLM_DWDP_PREFETCH_GATHER, gather, warned)
            ("gather", None, True, False),
            ("batched", None, False, False),
            ("batched", "1", False, True),
            ("gather", "0", True, True),
            ("gather", "1", True, False),
            (None, "1", True, False),
        ]
        for copy_mode, env, gather, warned in cases:
            with (
                self.subTest(copy_mode=copy_mode, env=env),
                patch.dict(os.environ),
                patch(f"{_FAC}.fabric_gather.gather_op", return_value=object()),
                patch(f"{_FAC}.fabric_gather.resident_sms", return_value=0) as resident_sms,
                patch(f"{_FAC}.logger.warning") as warning,
                patch(f"{_FAC}.torch.cuda.current_device", return_value=0),
            ):
                os.environ.pop("TRTLLM_DWDP_PREFETCH_GATHER", None)
                if env is not None:
                    os.environ["TRTLLM_DWDP_PREFETCH_GATHER"] = env
                config = _make_config().model_copy(update={"copy_mode": copy_mode})
                mgr = DwdpManager(
                    config=config, dist=MagicMock(spec=MPIDist), mapping=_make_mapping()
                )
                resident_sms.assert_called_once_with(gather=gather)
                overrides = [
                    call.args[0]
                    for call in warning.call_args_list
                    if "overrides TRTLLM_DWDP_PREFETCH_GATHER" in call.args[0]
                ]
                self.assertEqual(len(overrides), int(warned))
                mgr.setup(MagicMock())
                self.assertEqual(mock_setup.call_args.kwargs["copy_mode"], copy_mode)
                mgr.cleanup()

    def test_copy_mode_gather_without_its_op_fails_before_the_group(
        self, mock_comm_world: MagicMock, _rank: MagicMock, _setup: MagicMock
    ) -> None:
        _configure_mock_comm_world(mock_comm_world)
        config = _make_config().model_copy(update={"copy_mode": "gather"})
        with patch(f"{_FAC}.fabric_gather.gather_op", return_value=None):
            with self.assertRaisesRegex(
                ValueError, "copy_mode='gather' needs trtllm::dwdp_fabric_gather"
            ):
                DwdpManager(config=config, dist=MagicMock(spec=MPIDist), mapping=_make_mapping())
        mock_comm_world.Create_group.assert_not_called()

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
        self.assertIsNone(get_global_dense_dwdp_manager())

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
        # Dense DWDP is off: the weight manager makes its own copy stream, nothing is reserved.
        self.assertIsNone(kwargs["copy_stream"])
        self.assertEqual(kwargs["reserved_storages"], frozenset())

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


class _FakeWeightManager:
    """The DWDPWeightManager calls DwdpManager and its dense manager make, logged."""

    def __init__(self, log: list[tuple]) -> None:
        self.log = log
        # What DwdpManager.setup compares with the transfer dense DWDP ran on.
        self.copy_stream, self.use_gather, self.use_batched_prefetch = None, False, False

    def start_forward(self) -> None:
        self.log.append(("start_forward",))


class _FakeDense:
    """The DenseDWDP calls the attached dense manager makes, logged; ``slots`` are its rings."""

    def __init__(self, log: list[tuple], layers: list[int], slots: dict | None = None) -> None:
        self.log, self.layers, self.slots = log, tuple(layers), slots or {}
        self.primed = self.forward_open = False
        self.eager_checks: list[str] = []

    def require_eager(self, hook: str) -> None:
        self.eager_checks.append(hook)

    def prefetch(self, layer: int) -> None:
        self.log.append(("L", layer))

    def mark_primed(self) -> None:
        self.log.append(("mark_primed",))
        self.primed = True

    def war_barrier(self) -> None:
        self.log.append(("war_barrier",))
        self.primed = self.forward_open = False

    def release(self) -> None:
        self.log.append(("dense.release",))


class _DenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()


class TestDwdpManagerDenseSetup(unittest.TestCase):
    """A DwdpManager over fakes, not entered unless a test enters it; its dense manager is real."""

    LAYERS = [3, 5, 7]
    #: Global rank 3 is DWDP rank 1 of group 1 on device 2: passing one for another shows.
    GLOBAL_RANK, DWDP_RANK, DEVICE_ID = 3, 1, 2

    def setUp(self) -> None:
        _clear_global_managers()
        self.addCleanup(_clear_global_managers)
        self.log: list[tuple] = []
        self.sub_comm = _configure_mock_comm_world(_start(self, patch(f"{_FAC}.COMM_WORLD")))
        self.sub_comm.Free.side_effect = lambda: self.log.append(("group.Free",))
        _start(self, patch(f"{_FAC}.dwdp_global_rank", return_value=self.GLOBAL_RANK))
        _start(self, patch(f"{_FAC}.torch.cuda.current_device", return_value=self.DEVICE_ID))
        self.weight_manager = _FakeWeightManager(self.log)
        self.setup_dwdp = _start(
            self, patch(f"{_FAC}.setup_dwdp", return_value=self.weight_manager)
        )
        teardown = patch(
            f"{_FAC}.teardown_dwdp",
            side_effect=lambda *_a, **_k: self.log.append(("teardown_dwdp",)),
        )
        _start(self, teardown)
        self.roots_available = _start(
            self, patch(f"{_FAC}.dense_roots_available", return_value=True)
        )
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp", return_value=None))
        self.model = _DenseModel()
        self.manager = self.new_manager()

    def new_manager(self, enable_dense: bool = False) -> DwdpManager:
        """A DwdpManager of DWDP rank ``DWDP_RANK`` with ``LAYERS`` registered in reverse order."""
        mapping = _make_mapping(dwdp_rank=self.DWDP_RANK)
        manager = DwdpManager(
            config=_make_config(num_groups=2, enable_dense=enable_dense),
            dist=MagicMock(spec=MPIDist),
            mapping=mapping,
        )
        for layer in reversed(self.LAYERS):
            manager.add_layer(layer)
        return manager

    def test_dense_setup_runs_first_and_hands_its_stream_and_slots_to_the_expert_setup(
        self,
    ) -> None:
        """On every rank of a model with roots, requested or not, before the expert setup.

        The expert setup gets the copy stream dense setup made (none if it made none) and the
        committed slots.  If it then fails (the ring does not fit), the runtime no manager took
        over is released and the error propagates as raised.  Dense setup gets this rank's
        ``enable_dense``.
        """
        _start(self, patch.dict(os.environ))
        os.environ.pop("TRTLLM_DWDP_PREFETCH_GATHER", None)
        stream = object()
        _start(self, patch(f"{_FAC}.new_expert_copy_stream", return_value=stream))
        slots = {"ring": [SimpleNamespace(storage=torch.zeros(1)) for _ in range(2)]}
        dense, seen = _FakeDense(self.log, self.LAYERS, slots), []

        def dense_setup(
            model: nn.Module, transfer: object, comm: object, **kwargs: object
        ) -> _FakeDense:
            seen.append((type(transfer), self.setup_dwdp.call_count, self.manager._model_ref))
            self.assertIs(transfer.copy_stream, stream)  # the first read, in P2, creates it
            return dense

        self.setup_dense.side_effect = dense_setup
        self.weight_manager.copy_stream = stream
        self.assertIs(self.manager.setup(self.model), self.weight_manager)

        self.setup_dense.assert_called_once_with(
            self.model,
            unittest.mock.ANY,
            self.sub_comm,
            dwdp_rank=self.DWDP_RANK,
            dwdp_size=2,
            device_id=self.DEVICE_ID,
            moe_layers=[3, 5, 7],
            config=DenseConfig(enabled=False),
        )
        self.assertEqual(seen, [(_AttachedTransfer, 0, None)])
        kwargs = self.setup_dwdp.call_args.kwargs
        self.assertEqual((kwargs["device_id"], kwargs["copy_stream"]), (self.DEVICE_ID, stream))
        self.assertEqual(kwargs["num_prefetch_slots"], 2)
        self.assertEqual(kwargs["repeated_layer_indices"], [])
        storages = {slot.storage.untyped_storage().data_ptr() for slot in slots["ring"]}
        self.assertEqual(kwargs["reserved_storages"], storages)
        self.assertIs(self.manager.dense, dense)

        self.setup_dense.side_effect = None  # dense DWDP off on every rank: no stream made
        self.new_manager().setup(_DenseModel())
        self.assertIsNone(self.setup_dwdp.call_args.kwargs["copy_stream"])

        self.setup_dense.side_effect = lambda *_args, **_kwargs: _FakeDense(self.log, self.LAYERS)
        error = self.setup_dwdp.side_effect = PrefetchRingMemoryError("the ring does not fit")
        manager = self.new_manager()
        with self.assertRaises(PrefetchRingMemoryError) as raised:
            manager.setup(_DenseModel())

        self.assertIs(raised.exception, error)
        self.assertEqual(self.log, [("dense.release",)])
        self.assertIsNone(manager.dense)

    def test_an_expert_setup_on_another_copy_stream_or_transfer_is_refused(self) -> None:
        """The dense and the expert copies must share one copy stream and one transfer.

        A weight manager on another copy stream than dense setup made, or with another
        ``use_batched_prefetch``, is refused after the expert setup; the runtime no manager took
        over is released.
        """
        _start(self, patch.dict(os.environ))
        os.environ.pop("TRTLLM_DWDP_PREFETCH_GATHER", None)
        stream = object()
        _start(self, patch(f"{_FAC}.new_expert_copy_stream", return_value=stream))

        def dense_setup(
            model: nn.Module, transfer: object, comm: object, **kwargs: object
        ) -> _FakeDense:
            self.assertIs(transfer.copy_stream, stream)  # the first read, in P2, creates it
            return _FakeDense(self.log, self.LAYERS)

        self.setup_dense.side_effect = dense_setup
        for copy_stream, batched in ((object(), False), (stream, True)):
            with self.subTest(other_stream=copy_stream is not stream, batched=batched):
                self.log.clear()
                self.weight_manager.copy_stream = copy_stream
                self.weight_manager.use_batched_prefetch = batched
                manager = self.new_manager()
                with self.assertRaisesRegex(RuntimeError, "must share one copy stream"):
                    manager.setup(_DenseModel())

                self.assertEqual(self.log, [("dense.release",)])
                self.assertIsNone(manager.dense)

    def test_a_model_without_roots_makes_no_dense_collective(self) -> None:
        """A rank that asks for dense DWDP raises alone, after the expert setup.

        A rank that does not ask serves expert DWDP alone.
        """
        self.roots_available.return_value = False
        for enable_dense in (False, True):
            with self.subTest(enable_dense=enable_dense):
                manager = self.new_manager(enable_dense=enable_dense)
                if enable_dense:
                    with self.assertRaisesRegex(
                        DenseSetupError,
                        "_DenseModel do not provide dense_dwdp_roots.*set "
                        "dwdp_config.enable_dense=False for this model",
                    ):
                        manager.setup(self.model)
                else:
                    self.assertIs(manager.setup(self.model), self.weight_manager)
                self.assertIsNone(manager.dense)
                manager.cleanup()
        self.assertEqual(self.setup_dwdp.call_count, 2)
        self.setup_dense.assert_not_called()
        self.sub_comm.allgather.assert_not_called()

    def test_setup_priming_abort_recovery_and_teardown_keep_their_calls(self) -> None:
        """The attached lifecycle against one call log.

        The dense priming of the first two layers before every expert start; after an aborted
        forward the executor's start leaves the recovery to H0: the WAR barrier, the dense
        priming, then the expert start; dense released first.
        """
        dense = _FakeDense(self.log, self.LAYERS)
        self.setup_dense.return_value = dense

        with self.manager:
            self.manager.setup(self.model)
            self.assertIs(self.manager.dense_manager.dense, dense)
            self.manager.start_forward()  # the executor loop, before the forward
            self.manager.ensure_primed()  # H0 of that forward: already primed
            dense.primed, dense.forward_open = False, True  # the forward aborts midway
            self.manager.start_forward()  # skipped: H0 recovers on the execution stream
            self.manager.dense_manager.ensure_primed()  # H0 as the model calls it
            self.manager.start_forward()  # dense primed again: the expert start only

        priming = [("L", 3), ("L", 5), ("mark_primed",)]
        start = [("start_forward",)]
        recovery = [("war_barrier",)] + priming + start
        teardown = [("dense.release",), ("teardown_dwdp",), ("group.Free",)]
        self.assertEqual(self.log, priming + start + start + recovery + start + teardown)
        self.assertEqual(dense.eager_checks, [H0_HOOK, H0_HOOK])
        self.assertIsNone(get_global_dense_dwdp_manager())

    def test_more_than_two_expert_slots_are_logged_when_dense_dwdp_commits(self) -> None:
        """With more than two expert prefetch slots each expert copy runs ahead of its layer's
        dense copy on the shared copy stream; setup says so at INFO."""
        info = _start(self, patch(f"{_FAC}.logger.info"))
        for slots, committed in ((2, True), (3, False), (3, True)):
            with self.subTest(slots=slots, committed=committed):
                self.setup_dense.return_value = MagicMock() if committed else None
                manager = self.new_manager()
                manager.config = manager.config.model_copy(update={"num_prefetch_slots": slots})
                info.reset_mock()
                manager.setup(_DenseModel())
                messages = [call.args[0] for call in info.call_args_list]
                logged = [
                    message
                    for message in messages
                    if f"{slots} expert prefetch slots: each expert copy runs ahead of its "
                    "layer's dense copy" in message
                ]
                self.assertEqual(len(logged), int(slots > 2 and committed), messages)


class _Backend:
    """A MoE backend as ``ConfigurableMoE._decide_dwdp`` reads it."""

    def __init__(self, *, supports_dwdp: bool, uses_locality_domain: bool, nvfp4: bool) -> None:
        self.capabilities = SimpleNamespace(supports_dwdp=supports_dwdp)
        self.uses_locality_domain = uses_locality_domain
        quant_mode = SimpleNamespace(has_nvfp4=lambda: nvfp4)
        self.quant_config = SimpleNamespace(layer_quant_mode=quant_mode)


class _Layer:
    """The ConfigurableMoE state ``_decide_dwdp`` reads and writes, with ``MoE.__init__``'s
    layout under ``moe_ep_size`` 1: every expert."""

    _decide_dwdp = ConfigurableMoE._decide_dwdp
    _init_dwdp_expert_layout = MoE._init_dwdp_expert_layout

    NUM_EXPERTS, LAYER_IDX = 896, 5

    def __init__(
        self, *, supports_dwdp: bool = True, uses_locality_domain: bool = False, nvfp4: bool = True
    ) -> None:
        self.backend = _Backend(
            supports_dwdp=supports_dwdp, uses_locality_domain=uses_locality_domain, nvfp4=nvfp4
        )
        self.model_config = SimpleNamespace(quant_config=None)
        self.layer_idx, self.num_experts = self.LAYER_IDX, self.NUM_EXPERTS
        self.layer_load_balancer = None
        self.num_slots = self.expert_size_per_partition = self.slot_end = self.NUM_EXPERTS
        self.slot_start = 0
        self.initial_local_expert_ids = list(range(self.NUM_EXPERTS))
        self.initial_global_assignments = list(range(self.NUM_EXPERTS))

    def layout(self) -> tuple[int, int, int, int]:
        """``(num_slots, expert_size_per_partition, slot_start, slot_end)``."""
        return self.num_slots, self.expert_size_per_partition, self.slot_start, self.slot_end


class _MinimalMoE(MoE):
    """A MoE layer without kernels, to run ``MoE.__init__`` alone."""

    @classmethod
    def can_implement(cls, p: object, d: object) -> None:
        raise NotImplementedError

    def quantize_input(self, x: torch.Tensor, **kwargs: object) -> None:
        raise NotImplementedError

    def run_moe(self, ctx: object, *, workspace: object = None) -> None:
        raise NotImplementedError

    def forward_impl(self, x: torch.Tensor, router_logits: torch.Tensor, **kwargs: object) -> None:
        raise NotImplementedError


class TestConfigurableMoEDwdpDecision(unittest.TestCase):
    """Whether DWDP manages a MoE layer is decided once its backend exists.

    A layer it manages is sized to this rank's range of the expert partition and registered; any
    other layer keeps every expert, with one warning naming the layer and the reason.
    """

    def setUp(self) -> None:
        _clear_global_managers()
        self.addCleanup(_clear_global_managers)
        # DWDP rank 8 of a balanced DWDP12: 8 x 75 + 4 x 74 experts.
        self.manager = MagicMock(dwdp_rank=8)
        self.manager.expert_partition.return_value = ExpertPartition.balanced(
            num_experts=_Layer.NUM_EXPERTS, dwdp_size=12
        )
        self.warning = _start(self, patch.object(configurable_moe.logger, "warning"))

    def test_an_eligible_layer_is_sized_to_its_range_and_registered(self) -> None:
        set_global_dwdp_manager(self.manager)
        layer = _Layer()

        layer._decide_dwdp()

        self.assertTrue(layer.enable_dwdp)
        self.assertEqual(layer.layout(), (896, 74, 600, 674))
        self.assertEqual(layer.initial_local_expert_ids, list(range(600, 674)))
        self.manager.add_layer.assert_called_once_with(layer_idx=_Layer.LAYER_IDX)
        self.warning.assert_not_called()

    def test_a_layer_dwdp_cannot_manage_keeps_every_expert(self) -> None:
        set_global_dwdp_manager(self.manager)
        cases = {
            "does not support DWDP": dict(supports_dwdp=False),
            "uses locality domains": dict(uses_locality_domain=True),
            "is not NVFP4-quantized": dict(nvfp4=False),  # e.g. a BF16 MTP layer
        }
        for reason, backend in cases.items():
            with self.subTest(reason):
                self.manager.reset_mock()
                self.warning.reset_mock()
                layer = _Layer(**backend)

                layer._decide_dwdp()

                self.assertFalse(layer.enable_dwdp)
                self.assertEqual(layer.layout(), (896, 896, 0, 896))
                self.manager.add_layer.assert_not_called()
                self.manager.expert_partition.assert_not_called()
                self.warning.assert_called_once()
                message = self.warning.call_args.args[0]
                self.assertIn("MoE layer 5 (_Backend) keeps all 896 experts on every rank", message)
                self.assertIn(reason, message)

    def test_without_a_dwdp_manager_nothing_changes(self) -> None:
        layer = _Layer()

        layer._decide_dwdp()

        self.assertFalse(layer.enable_dwdp)
        self.assertEqual(layer.layout(), (896, 896, 0, 896))
        self.warning.assert_not_called()

    def test_moe_init_alone_keeps_every_expert(self) -> None:
        """Only ConfigurableMoE's decision sizes a layer: a backend built by ``MoE.__init__``,
        wrapped or not, keeps the layout of ``moe_ep_size`` 1."""
        set_global_dwdp_manager(self.manager)

        moe = _MinimalMoE(
            routing_method=DefaultMoeRoutingMethod(top_k=2),
            num_experts=8,
            hidden_size=16,
            intermediate_size=32,
            init_load_balancer=False,
        )

        self.assertEqual((moe.expert_size_per_partition, moe.slot_start, moe.slot_end), (8, 0, 8))
        self.manager.expert_partition.assert_not_called()


if __name__ == "__main__":
    unittest.main()
