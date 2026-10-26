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
"""CPU tests of the dense DWDP launch gate (``gate.validate_dense_gate``) in both modes.

The gate returns the mode the launch runs dense DWDP in: attached, with expert DWDP
(``dwdp_config.enable_experts`` with ``dwdp_size > 1``, ``tp_size=1``, the overlap scheduler off),
or standalone on an attention-DP instance (``enable_experts=False``, ``enable_attention_dp`` and
``tp_size > 1``).  It rejects a launch that fits neither, or that breaks a requirement, with a
message naming it.  Every launch starts with MPI on, FABRIC peer handles and a registered gather
op; a launch changes only what it checks.
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
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    DwdpConfig,
    MTPDecodingConfig,
    PrefillCudaGraphBackend,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

ATTACHED = DenseMode.ATTACHED
STANDALONE = DenseMode.STANDALONE
_HANDLE_TYPE = vmm.cuda.CUmemAllocationHandleType


def _dwdp_config(mode: DenseMode, **overrides: object) -> DwdpConfig:
    """The ``dwdp_config`` of a dense DWDP launch in ``mode``."""
    if mode is ATTACHED:
        fields = dict(
            dwdp_size=16, num_groups=1, num_experts_per_worker=56, num_prefetch_experts=56
        )
    else:
        fields = dict(enable_experts=False)
    return DwdpConfig(enable_dense=True, **{**fields, **overrides})


class _Launch(NamedTuple):
    """The accepted context-worker launch of ``mode``, changed as given, and the gate's verdict.

    ``args`` are ``llm_args`` fields (applied unvalidated), ``dwdp`` ``DwdpConfig`` fields,
    ``mapping`` ``Mapping`` arguments and ``lacks`` what the host lacks: ``fabric`` handles,
    ``mpi`` or the ``gather_op``.  With ``error`` the gate must reject the launch with it in its
    message, else accept it with ``warning`` in its only warning, or with no warning.
    """

    mode: DenseMode
    error: str = ""
    warning: str = ""
    args: dict = {}
    dwdp: dict = {}
    mapping: dict = {}
    lacks: str = ""

    def check(self, monkeypatch: pytest.MonkeyPatch) -> DenseMode:
        handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC
        if self.lacks == "fabric":
            handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        monkeypatch.setattr(vmm, "peer_handle_type", lambda: handle_type)
        monkeypatch.setattr(gate, "mpi_disabled", lambda: self.lacks == "mpi")
        gather_op = None if self.lacks == "gather_op" else object()
        monkeypatch.setattr(fabric_gather, "gather_op", lambda: gather_op)
        llm_args = TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            cuda_graph_config=None,
            disable_overlap_scheduler=True,
            dwdp_config=_dwdp_config(self.mode, **self.dwdp),
        ).model_copy(update=self.args)
        # Attached: tp 1.  Standalone: a 16-rank attention-DP context instance, experts parallel
        # across it.
        fields = dict(tp_size=1, pp_size=1, cp_size=1, enable_attention_dp=False)
        if self.mode is STANDALONE:
            fields.update(tp_size=16, enable_attention_dp=True)
        fields.update(self.mapping)
        if fields["enable_attention_dp"] and fields["cp_size"] == 1:
            fields["moe_ep_size"] = fields["tp_size"]  # CP has no MoE EP
        world_size = fields["tp_size"] * fields["pp_size"] * fields["cp_size"]
        return validate_dense_gate(llm_args, Mapping(world_size=world_size, rank=0, **fields))


_GATHER = (
    "Setup applies TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER=clc_dynamic and "
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH=1, each unless it is set, so that the persistent dense GEMM "
    "grids do not run a second wave on the SMs the gather holds and cooperative routing does not "
    "wait for them; TRTLLM_DWDP_GATHER_KERNEL picks the gather kernel"
)

LAUNCHES = {
    "attached": _Launch(ATTACHED),
    "standalone": _Launch(STANDALONE),
    # Without a DWDP group of two or more ranks, only an attention-DP group of them will do.
    "dwdp_size_1": _Launch(
        ATTACHED, "but this launch has expert DWDP with dwdp_size=1.", dwdp={"dwdp_size": 1}
    ),
    "standalone_without_attention_dp": _Launch(
        STANDALONE,
        "but this launch has no expert DWDP and enable_attention_dp=False.",
        mapping={"enable_attention_dp": False},
    ),
    "standalone_tp1": _Launch(
        STANDALONE, "but this launch has no expert DWDP and tp_size=1.", mapping={"tp_size": 1}
    ),
    # copy_mode picks the standalone transfer; with expert DWDP it sets the expert transfer,
    # which DwdpManager checks, and the gate says nothing.
    "standalone-batched_copy": _Launch(STANDALONE, dwdp={"copy_mode": "batched"}),
    # A standalone gather launch passes with one warning, which names the compute-side defaults
    # setup applies for the gather and the variable that picks its kernel.
    "standalone-gather_copy": _Launch(STANDALONE, warning=_GATHER, dwdp={"copy_mode": "gather"}),
    "standalone-gather_without_op": _Launch(
        STANDALONE,
        "requires trtllm::dwdp_fabric_gather for dwdp_config.copy_mode='gather', but this "
        "build does not register it.",
        dwdp={"copy_mode": "gather"},
        lacks="gather_op",
    ),
    "attached-gather_copy": _Launch(ATTACHED, dwdp={"copy_mode": "gather"}, lacks="gather_op"),
    # One launch per requirement: attached also needs tp_size 1 and the overlap scheduler off;
    # those of both modes alternate between the modes; then those of standalone alone.
    "attached-tp_size": _Launch(
        ATTACHED, "this launch has expert DWDP with tp_size=2", mapping={"tp_size": 2}
    ),
    "attached-overlap_scheduler": _Launch(
        ATTACHED,
        "this launch has expert DWDP with disable_overlap_scheduler=False",
        args={"disable_overlap_scheduler": False},
    ),
    **{
        f"{mode.value}-{name}": _Launch(
            mode, f"requires {name}=None, but {name} is set", args={name: value}
        )
        for mode, (name, value) in zip(
            [ATTACHED, STANDALONE] * 3,
            {
                "cuda_graph_config": CudaGraphConfig(),
                "torch_compile_config": TorchCompileConfig(),
                "sleep_config": SleepConfig(),
                "lora_config": LoraConfig(),
                "speculative_config": MTPDecodingConfig(max_draft_len=1),
            }.items(),
        )
    },
    "standalone-prefill_cuda_graph_backend": _Launch(
        STANDALONE,
        "requires prefill_cuda_graph_backend='disabled', but prefill_cuda_graph_backend=",
        args={"prefill_cuda_graph_backend": PrefillCudaGraphBackend.BREAKABLE},
    ),
    "attached-enable_locality_domains": _Launch(
        ATTACHED,
        "requires enable_locality_domains=False, but enable_locality_domains=True",
        args={"enable_locality_domains": True},
    ),
    "standalone-pp_size": _Launch(
        STANDALONE, "requires pp_size=1, but pp_size=2", mapping={"pp_size": 2}
    ),
    "attached-cp_size": _Launch(
        ATTACHED, "requires cp_size=1, but cp_size=2", mapping={"cp_size": 2}
    ),
    "standalone-peer_handle_type": _Launch(
        STANDALONE,
        "requires FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC), but this host shares "
        "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR handles",
        lacks="fabric",
    ),
    "standalone-mpi": _Launch(STANDALONE, "requires the MPI backend", lacks="mpi"),
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
        assert launch.check(monkeypatch) is launch.mode
    assert len(warnings) == bool(launch.warning) and all(launch.warning in w for w in warnings)


def test_the_gather_warning_names_each_compute_side_default() -> None:
    """The standalone gather warning names every setting that setup applies for the gather."""
    saved = dict(os.environ)
    try:
        for name in ("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", "TLLM_ROUTING_DISABLE_COOP_LAUNCH"):
            os.environ.pop(name, None)
        before = dict(os.environ)
        fabric_gather.apply_compute_side_defaults()
        applied = {name: value for name, value in os.environ.items() if before.get(name) != value}
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert len(applied) == 2, applied
    for name, value in applied.items():
        assert f"{name}={value}" in _GATHER


@pytest.mark.parametrize(
    "dwdp_config, gated",
    [
        (None, False),
        (DwdpConfig(dwdp_size=16), False),
        (_dwdp_config(ATTACHED), True),
        (_dwdp_config(STANDALONE), True),
    ],
    ids=["no-dwdp", "expert-only", "attached", "standalone"],
)
def test_the_creator_runs_the_gate_only_for_dense_dwdp(
    dwdp_config: DwdpConfig | None, gated: bool
) -> None:
    """``_check_dense_dwdp_launch`` gates a launch with ``dwdp_config.enable_dense`` only and
    returns the gate's mode."""
    llm_args = TorchLlmArgs(model="/tmp/dummy_model", skip_tokenizer_init=True).model_copy(
        update={"dwdp_config": dwdp_config}
    )
    mapping = Mapping(world_size=1, rank=0)
    with patch.object(
        py_executor_creator, "validate_dense_gate", return_value=ATTACHED
    ) as gate_call:
        mode = py_executor_creator._check_dense_dwdp_launch(llm_args, mapping)
    assert gate_call.called is gated
    assert mode is (ATTACHED if gated else None)
