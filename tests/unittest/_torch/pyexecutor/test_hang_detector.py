# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest


@pytest.mark.cpu_only
class TestAllRanksCrashedProbe:
    def test_single_rank_is_never_symmetric(self) -> None:
        from tensorrt_llm._torch.pyexecutor import hang_detector

        assert hang_detector.all_ranks_crashed(1) is False

    def test_mpi_disabled_falls_back_to_the_kill(self) -> None:
        from tensorrt_llm._torch.pyexecutor import hang_detector

        with mock.patch.object(hang_detector, "mpi_disabled", return_value=True):
            assert hang_detector.all_ranks_crashed(4) is False

    def test_probe_errors_fall_back_to_the_kill(self) -> None:
        from mpi4py import MPI

        from tensorrt_llm._torch.pyexecutor import hang_detector

        with (
            mock.patch.object(hang_detector, "mpi_disabled", return_value=False),
            mock.patch.object(MPI, "Is_initialized", return_value=True),
            mock.patch.object(MPI, "Query_thread", return_value=MPI.THREAD_MULTIPLE),
            mock.patch.object(hang_detector, "ENABLE_MULTI_DEVICE", True),
            mock.patch.object(
                hang_detector, "mpi_comm", side_effect=RuntimeError("no comm")
            ) as mpi_comm,
        ):
            assert hang_detector.all_ranks_crashed(4) is False
            mpi_comm.assert_called_once_with()

    def test_disabled_hard_kill_skips_the_barrier(self) -> None:
        # With the kill disabled (grace is None) no kill follows a False
        # verdict, so posting Ibarrier would leak it into a surviving world.
        # The probe must return False without posting the barrier.
        from mpi4py import MPI

        from tensorrt_llm._torch.pyexecutor import hang_detector

        comm = mock.Mock()
        comm.Get_size.return_value = 4
        with (
            mock.patch.object(hang_detector, "mpi_disabled", return_value=False),
            mock.patch.object(MPI, "Is_initialized", return_value=True),
            mock.patch.object(MPI, "Query_thread", return_value=MPI.THREAD_MULTIPLE),
            mock.patch.object(hang_detector, "ENABLE_MULTI_DEVICE", True),
            mock.patch.object(hang_detector, "mpi_comm", return_value=comm),
            mock.patch.object(hang_detector, "_rank_crash_kill_grace", return_value=None),
        ):
            assert hang_detector.all_ranks_crashed(4) is False
            comm.Ibarrier.assert_not_called()
