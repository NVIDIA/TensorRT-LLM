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
"""Tests for the model-load checks and the MoE layer binding of Wide-EP fault tolerance."""

import contextlib
import os
import re
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.distributed.membership import HANG_TIMEOUT_S, host_weights_prefix
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import SYNC_PROTOCOL_ENV
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import DeploymentHostWeights
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import SERVING_TIMEOUT_ENV
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
from tensorrt_llm._torch.pyexecutor import rank_failure_model_load
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.pyexecutor.rank_failure_model_load import check_supported, prepare_model
from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.llmapi.llm_args import LoadFormat

pytestmark = pytest.mark.cpu_only

WORLD = 4


# The startup checks, before the model loads.

_SERVING_TIMEOUT_S = int(2 * HANG_TIMEOUT_S)


@pytest.mark.parametrize(
    "env, lacks, fix",
    [
        ({}, None, None),
        ({SYNC_PROTOCOL_ENV: "auto"}, None, None),
        ({SYNC_PROTOCOL_ENV: "fence"}, None, None),
        # Deprecated and ignored.
        ({"TLLM_FAULT_TOLERANCE_MODE": "1"}, None, None),
        ({SERVING_TIMEOUT_ENV: str(_SERVING_TIMEOUT_S)}, None, None),
        (
            {SYNC_PROTOCOL_ENV: "sentinel"},
            None,
            f"the fence path ({SYNC_PROTOCOL_ENV} unset, auto or fence)",
        ),
        (
            {SERVING_TIMEOUT_ENV: str(_SERVING_TIMEOUT_S - 1)},
            None,
            f"not {_SERVING_TIMEOUT_S - 1} s), because a wait on a hung rank",
        ),
        ({}, "ep", f"(moe_expert_parallel_size={WORLD}, not {WORLD // 2})"),
        ({}, "kv_v2", "(kv_cache_config.use_kv_cache_manager_v2=True, not False)"),
        # "auto" resolves only as the model loads.
        ({}, "kv_auto", "(kv_cache_config.use_kv_cache_manager_v2=True, not 'auto')"),
        # The executor creator turns it off after the arguments' validation in some modes.
        ({}, "overlap", "the overlap scheduler, which speculative decoding without overlap"),
        ({}, "architecture", "a model architecture in ['DeepseekV3ForCausalLM']"),
        ({}, "no_moe", "requires a model with MoE layers"),
    ],
)
def test_startup_rejects_unsupported_configuration(monkeypatch, env, lacks, fix):
    """Check that each unsupported setup is refused with its reason and how to turn it off."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    llm_args = SimpleNamespace(
        attention_dp_config=None,
        disable_overlap_scheduler=lacks == "overlap",
        kv_cache_config=SimpleNamespace(
            use_kv_cache_manager_v2={"kv_v2": False, "kv_auto": "auto"}.get(lacks, True)
        ),
    )
    mapping = SimpleNamespace(moe_ep_size=WORLD // 2 if lacks == "ep" else WORLD, world_size=WORLD)
    check = partial(check_supported, llm_args, mapping)
    if fix is None:
        check()
        return
    model = torch.nn.Module()  # A model without MoE layers.
    architecture = "Qwen3MoeForCausalLM" if lacks == "architecture" else "DeepseekV3ForCausalLM"
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=[architecture]),
    )
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        (partial(prepare_model, model) if lacks in ("architecture", "no_moe") else check)()
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


def test_model_load_checks_support_first(monkeypatch):
    """Check that the model load refuses an unsupported setup before it builds the model."""
    monkeypatch.setenv(SYNC_PROTOCOL_ENV, "sentinel")
    monkeypatch.setattr(
        rank_failure_model_load,
        "create_moe_load_balancer",
        lambda config, mapping: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        model_loader_module.AutoModelForCausalLM,
        "from_config",
        Mock(side_effect=AssertionError("the model was built before the support check")),
    )
    llm_args = SimpleNamespace(
        enable_wide_ep_fault_tolerance=True,
        load_format=LoadFormat.AUTO,
        disable_overlap_scheduler=False,
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True),
    )
    mapping = SimpleNamespace(moe_ep_size=WORLD, world_size=WORLD)
    loader = ModelLoader(llm_args, mapping, None, None, max_num_tokens=128, max_seq_len=128)
    loader._load_and_validate_config = Mock(return_value=SimpleNamespace(pretrained_config=None))
    with pytest.raises(ValueError, match=re.escape(f"the fence path ({SYNC_PROTOCOL_ENV} unset")):
        loader.load("/ckpt", Mock())


# Host weights. Every node keeps the weights of every expert in /dev/shm.

_EXPERTS = 8
_LOCAL_RANKS = 2


class _MoeWeights(MoEEplbWeightLayoutMixin, torch.nn.Module):
    """MoE layer weights of 3 local slots of 128 KiB and 4 bytes, plus one weight not per slot."""

    expert_size_per_partition = 3

    def __init__(self, sharer):
        super().__init__()
        self.layer_load_balancer = SimpleNamespace(host_tensor_sharer=sharer)
        self.w3_w1_weight = torch.nn.Parameter(torch.empty(3, 64, 1024, dtype=torch.float16))
        self.fc31_alpha = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.fc31_input_scale = torch.nn.Parameter(torch.empty((), dtype=torch.float32))


# Two weights per expert aligned to 256 bytes, plus one fingerprint block per local rank segment.
_NODE_BYTES = _EXPERTS * (128 * 1024 + 256) + _LOCAL_RANKS * 256


@pytest.mark.parametrize("shortfall", [0, 1])
def test_startup_check_needs_dev_shm_room(monkeypatch, shortfall):
    """Check the room for host weights in /dev/shm and the leftovers the error names."""
    sharer = DeploymentHostWeights(
        0,
        _EXPERTS,
        Mock(Get_rank=Mock(return_value=0), Get_size=Mock(return_value=_LOCAL_RANKS)),
        replacement=False,
    )
    sharer.set_shared_memory_base_name(f"wide_ep_ft_test_{os.getpid()}_{time.monotonic_ns()}")
    model = torch.nn.ModuleList([_MoeWeights(sharer)])
    free = _NODE_BYTES - shortfall
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=free, f_frsize=1))
    if not shortfall:
        rank_failure_model_load._check_host_weights_fit(model)
        return
    leftover = f"{host_weights_prefix()}{os.getpid()}_{time.monotonic_ns()}_l0_lr0_all"
    with open(os.path.join("/dev/shm", leftover), "wb") as segment:
        segment.write(bytes(1 << 20))
    try:
        with pytest.raises(ValueError, match=f"needs {_NODE_BYTES} bytes more") as error:
            rank_failure_model_load._check_host_weights_fit(model)
    finally:
        os.unlink(os.path.join("/dev/shm", leftover))
    message = str(error.value)
    assert f"{leftover} (0.0 GiB)" in message
    assert TURN_FAULT_TOLERANCE_OFF in message


def test_leftover_unlinked_during_scan_is_skipped(monkeypatch):
    """Check that host weights unlinked while /dev/shm is scanned are not named as leftovers."""

    def unlinked():
        raise FileNotFoundError

    entry = SimpleNamespace(name=f"{host_weights_prefix()}unlinked", stat=unlinked)
    monkeypatch.setattr(os, "scandir", lambda path: contextlib.nullcontext([entry]))
    assert rank_failure_model_load._leftover_entries() == ""
