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
"""CPU tests of the dense DWDP launch gate (``gate.validate_dense_gate``).

``TorchLlmArgs.validate_dwdp_launch`` refuses what the arguments alone rule out
(``test_dense_config.py``); the gate checks what dense DWDP needs from the host and returns the
mode it runs in, standalone on an attention-DP instance; it refuses dense DWDP with expert DWDP.
Every launch starts with MPI on, FABRIC peer handles and a registered gather op; a launch changes
only what it checks.
"""

import os
import re
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.modules.dwdp import fabric_gather, vmm
from tensorrt_llm._torch.modules.dwdp.dense import gate
from tensorrt_llm._torch.modules.dwdp.dense.gate import validate_dense_gate
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm.llmapi.llm_args import DwdpConfig, TorchLlmArgs

pytestmark = pytest.mark.cpu_only

STANDALONE = DenseMode.STANDALONE
_HANDLE_TYPE = vmm.cuda.CUmemAllocationHandleType


def _llm_args(dwdp_config: object) -> TorchLlmArgs:
    """Arguments with ``dwdp_config``, applied unvalidated: the gate reads only that."""
    return TorchLlmArgs(model="/tmp/dummy_model", skip_tokenizer_init=True).model_copy(
        update={"dwdp_config": dwdp_config}
    )


class _Launch(NamedTuple):
    """A standalone dense DWDP launch on a host that ``lacks`` FABRIC handles (``fabric``),
    ``mpi`` or the ``gather_op``, with ``dwdp`` fields changed, and the gate's verdict.

    With ``error`` the gate must reject the launch with it in its message, else accept it with
    ``warning`` in its only warning, or with no warning.
    """

    error: str = ""
    warning: str = ""
    dwdp: dict = {}
    lacks: str = ""

    def check(self, monkeypatch: pytest.MonkeyPatch) -> DenseMode:
        handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC
        if self.lacks == "fabric":
            handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        monkeypatch.setattr(vmm, "peer_handle_type", lambda: handle_type)
        monkeypatch.setattr(gate, "mpi_disabled", lambda: self.lacks == "mpi")
        gather_op = None if self.lacks == "gather_op" else object()
        monkeypatch.setattr(fabric_gather, "gather_op", lambda: gather_op)
        dwdp_config = DwdpConfig(enable_experts=False, enable_dense=True, **self.dwdp)
        return validate_dense_gate(_llm_args(dwdp_config))


_GATHER = (
    "Setup applies TRTLLM_CUTEDSL_SM_RESERVE (the gather's block count), "
    "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER=clc_dynamic and "
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH=1, each unless it is set, so that the persistent dense GEMM "
    "grids do not run a second wave on the SMs the gather holds and cooperative routing does not "
    "wait for them; TRTLLM_DWDP_GATHER_KERNEL=tma picks the TMA gather kernel"
)

LAUNCHES = {
    "standalone": _Launch(),
    "peer_handle_type": _Launch(
        "requires FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC), but this host shares "
        "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR handles",
        lacks="fabric",
    ),
    # The standalone dense group is an MPI sub-communicator.
    "mpi": _Launch("requires the MPI backend", lacks="mpi"),
    "batched_copy": _Launch(dwdp={"copy_mode": "batched"}),
    "per_slice_copy": _Launch(dwdp={"copy_mode": "per_slice"}),
    # A gather launch passes with one warning, which names the compute-side defaults setup
    # applies for the gather and the setting that picks its kernel.
    "gather_copy": _Launch(warning=_GATHER, dwdp={"copy_mode": "gather"}),
    "gather_without_op": _Launch(
        "requires trtllm::dwdp_fabric_gather for dwdp_config.copy_mode='gather', but this "
        "build does not register it.",
        dwdp={"copy_mode": "gather"},
        lacks="gather_op",
    ),
}


@pytest.mark.parametrize("launch", LAUNCHES.values(), ids=LAUNCHES)
def test_dense_gate(monkeypatch: pytest.MonkeyPatch, launch: _Launch) -> None:
    warnings = []
    monkeypatch.setattr(gate, "logger", SimpleNamespace(warning=warnings.append))
    if launch.error:
        with pytest.raises(ValueError, match=re.escape(launch.error)) as error:
            launch.check(monkeypatch)
        assert str(error.value).startswith("Dense DWDP (dwdp_config.enable_dense) ")
        assert str(error.value).endswith(
            "Change the launch configuration or set dwdp_config.enable_dense=False."
        )
    else:
        assert launch.check(monkeypatch) is STANDALONE
    assert len(warnings) == bool(launch.warning) and all(launch.warning in w for w in warnings)


def test_the_gate_refuses_dense_dwdp_with_expert_dwdp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense DWDP is not attached to expert DWDP; arguments that bypassed the DwdpConfig checks
    (applied unvalidated) are refused by the gate too."""
    handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: handle_type)
    config = DwdpConfig(dwdp_size=16).model_copy(update={"enable_dense": True})
    with pytest.raises(ValueError, match=re.escape("but dwdp_config.enable_experts=True")):
        validate_dense_gate(_llm_args(config))


def test_the_gather_warning_names_each_compute_side_default() -> None:
    """The gather warning names every setting that setup applies for the gather."""
    saved = dict(os.environ)
    try:
        for name in (
            "TRTLLM_CUTEDSL_SM_RESERVE",
            "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER",
            "TLLM_ROUTING_DISABLE_COOP_LAUNCH",
        ):
            os.environ.pop(name, None)
        before = dict(os.environ)
        fabric_gather.apply_compute_side_defaults()
        applied = {name: value for name, value in os.environ.items() if before.get(name) != value}
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert len(applied) == 3, applied
    reserve = applied.pop("TRTLLM_CUTEDSL_SM_RESERVE")
    assert reserve.isdigit() and "TRTLLM_CUTEDSL_SM_RESERVE (the gather's block count)" in _GATHER
    for name, value in applied.items():
        assert f"{name}={value}" in _GATHER


@pytest.mark.parametrize(
    "dwdp_config, gated",
    [
        (None, False),
        (DwdpConfig(dwdp_size=16), False),
        (DwdpConfig(enable_experts=False, enable_dense=True), True),
    ],
    ids=["no-dwdp", "expert-only", "standalone"],
)
def test_the_creator_runs_the_gate_only_for_dense_dwdp(
    dwdp_config: DwdpConfig | None, gated: bool
) -> None:
    """``_check_dense_dwdp_launch`` gates a launch with ``dwdp_config.enable_dense`` only and
    returns the gate's mode."""
    with patch.object(
        py_executor_creator, "validate_dense_gate", return_value=STANDALONE
    ) as gate_call:
        mode = py_executor_creator._check_dense_dwdp_launch(_llm_args(dwdp_config))
    assert gate_call.called is gated
    assert mode is (STANDALONE if gated else None)
