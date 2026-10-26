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
"""PyExecutor construction tests for DWDP: manager ownership and the dense DWDP launch gate."""

import re
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.modules.dwdp import vmm
from tensorrt_llm._torch.modules.dwdp.dense.gate import validate_dense_gate
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager
from tensorrt_llm.llmapi.llm_args import DwdpConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_HANDLE_TYPE = vmm.cuda.CUmemAllocationHandleType


@contextmanager
def _restore_global_dwdp_manager():
    """Keep the process-global manager from leaking into later tests."""
    previous = get_global_dwdp_manager()
    try:
        yield
    finally:
        set_global_dwdp_manager(previous)


def test_create_py_executor_rolls_back_new_dwdp_manager_on_error():
    with _restore_global_dwdp_manager():
        set_global_dwdp_manager(None)
        manager = MagicMock()
        manager.__exit__.side_effect = lambda *_args: set_global_dwdp_manager(None)

        def fail_after_registration(**_kwargs):
            set_global_dwdp_manager(manager)
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_after_registration,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        manager.__exit__.assert_called_once_with(None, None, None)
        assert get_global_dwdp_manager() is None


def test_create_py_executor_leaves_a_pre_existing_dwdp_manager_alone():
    """The wrapper only tears down a manager it created, not one it inherited."""
    with _restore_global_dwdp_manager():
        existing = MagicMock()
        set_global_dwdp_manager(existing)

        def fail_without_registering(**_kwargs):
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_without_registering,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        existing.__exit__.assert_not_called()
        assert get_global_dwdp_manager() is existing


# ----------------------------------------------------------------------
# Dense DWDP launch gate
# ----------------------------------------------------------------------


@pytest.fixture
def fabric_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this host share FABRIC handles, whatever its architecture."""
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC)


def _dense_launch(**overrides: object) -> TorchLlmArgs:
    """A dense DWDP context-worker launch, with ``overrides`` applied without validation."""
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        dwdp_config=DwdpConfig(
            enable_dense=True,
            dwdp_size=16,
            num_groups=1,
            num_experts_per_worker=56,
            num_prefetch_experts=56,
        ),
    )
    return args.model_copy(update=overrides)


def _mapping(tp_size: int = 1, pp_size: int = 1, cp_size: int = 1) -> Mapping:
    return Mapping(
        world_size=tp_size * pp_size * cp_size,
        rank=0,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
    )


def test_dense_gate_accepts_a_dense_dwdp_context_worker(fabric_host: None) -> None:
    validate_dense_gate(_dense_launch(), _mapping())


def test_dense_gate_requires_a_dwdp_group(fabric_host: None) -> None:
    dwdp_config = DwdpConfig(enable_dense=True, dwdp_size=1)
    with pytest.raises(ValueError, match=re.escape("dwdp_size > 1, but dwdp_size=1")):
        validate_dense_gate(_dense_launch(dwdp_config=dwdp_config), _mapping())


@pytest.mark.parametrize(
    "dwdp_config, gated",
    [(None, False), (DwdpConfig(dwdp_size=16), False), (DwdpConfig(enable_dense=True), True)],
    ids=["no-dwdp", "expert-only", "dense"],
)
def test_the_creator_runs_the_gate_only_for_dense_dwdp(
    dwdp_config: DwdpConfig | None, gated: bool
) -> None:
    """``_check_dense_dwdp_launch`` gates a launch with ``dwdp_config.enable_dense`` only."""
    llm_args = _dense_launch(dwdp_config=dwdp_config)
    with patch.object(py_executor_creator, "validate_dense_gate") as gate:
        py_executor_creator._check_dense_dwdp_launch(llm_args, _mapping())
    assert gate.called is gated
