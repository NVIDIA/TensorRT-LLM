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
"""The shared-path opt-ins the MiniMax-M3.1 sm_107 targets take: routing arms each of them, process-wide, before it
returns an sm_107 target (``arm_sm107_shared_opt_ins``), every one is off until armed, and the matching RuntimeHooks
fields default to off (so a model that declares none, GB300's included, keeps the stock path). CPU only."""

import ast
import dataclasses
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch import cute_aot
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    cache_manager as msa_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.m3ax_msa_prewarm import (
    hook as msa_prewarm_hook,
)
from tensorrt_llm._torch.pyexecutor import llm_request, perf_metrics_manager, py_executor
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_ops
from tensorrt_llm._torch.speculative import dflash, interface

pytestmark = pytest.mark.cpu_only

SHARED_OPT_INS = (
    "draft_kv_reserve_guard",
    "kv_gpu_prefetch",
    "context_token_array",
    "msa_prefill_fork",
    "proxy_ms_reuse",
    "context_onehot_index_fill",
    "dflash_host_ctx_alloc",
)
ARM_CALLS = (
    "arm",  # cute_aot
    "arm_draft_kv_reserve_guard",
    "arm_kv_gpu_prefetch",
    "arm_context_token_array",
    "arm_prefill_fork",
    "arm_proxy_ms_reuse",
    "arm_sm107_msa_paths",
    "arm_onehot_index_fill",
    "arm_sm107_spec_paths",
    "arm_host_ctx_alloc",
    "arm_gpu_timestamps",
)
# (module, flag, value once armed) for every process-wide flag the helper sets.
FLAGS = (
    (cute_aot, "_armed", True),
    (kv_cache_manager_v2, "_DRAFT_KV_RESERVE", True),
    (kv_cache_manager_v2, "_DRAFT_KV_GUARD", True),
    (py_executor, "_KV_GPU_PREFETCH", True),
    (llm_request, "_L1_TOKEN_ARRAY", True),
    (msa_prefill, "_PREFILL_FORK_ARMED", True),
    (msa_indexer, "_PROXY_MS_REUSE", True),
    (msa_indexer, "PREFILL_PROXY_SPLIT", False),
    (msa_indexer, "PREFILL_HOST_TRIMS", False),
    (msa_indexer, "PREFILL_SELECT_TILE", False),
    (msa_prewarm_hook, "ENABLED", False),
    (msa_cache_manager, "NVFP4_POOLLESS", False),
    (interface, "_ONEHOT_INDEX_FILL", True),
    (interface, "_ONEHOT_INDEX_FILL_ARMED", True),
    (interface, "_CHEAP_PADDING_ROWS", False),
    (flashinfer_ops, "TOP_P_RENORM_DETERMINISTIC", False),
    (dflash, "_HOST_CTX_ALLOC_DISABLED", False),
    (dflash, "_HOST_CTX_ALLOC_ARMED", True),
    (perf_metrics_manager, "_GPU_TIMESTAMPS", True),
)
KILL_SWITCHES = (
    "TRTLLM_M31_DISABLE_DRAFT_KV_RESERVE",
    "TRTLLM_M31_DISABLE_DRAFT_KV_GUARD",
    "TRTLLM_M31_DISABLE_L1_TOKEN_ARRAY",
    "TRTLLM_M31_DISABLE_PROXY_MS_REUSE",
    "TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL",
    "TRTLLM_DFLASH_DISABLE_HOST_CTX_ALLOC",
    "TRTLLM_M31_DISABLE_GPU_TIMESTAMPS",
)


def _routing_tree() -> ast.Module:
    path = Path(_family.__file__).resolve().parent / "routing.py"
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _called_names(fn: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None))
    return names


def test_helper_arms_every_shared_opt_in():
    called = _called_names(_function(_routing_tree(), "arm_sm107_shared_opt_ins"))
    assert set(ARM_CALLS) <= called, f"missing arm calls: {set(ARM_CALLS) - called}"


def test_route_arms_before_returning_an_sm107_target():
    assert "arm_sm107_shared_opt_ins" in _called_names(_function(_routing_tree(), "route"))


def test_helper_sets_every_flag(monkeypatch):
    for env in KILL_SWITCHES:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(py_executor, "_KV_GPU_PREFETCH_KILLED", False)
    # CPU only: stand in for the native stamp op, which GPU timestamps need to arm.
    monkeypatch.setattr(perf_metrics_manager, "_native_stamp_op", lambda: object())
    for module, flag, armed in FLAGS:
        monkeypatch.setattr(module, flag, not armed)
    routing.arm_sm107_shared_opt_ins()
    for module, flag, armed in FLAGS:
        assert getattr(module, flag) is armed, f"{module.__name__}.{flag}"


def test_shared_opt_ins_are_off_by_default():
    defaults = {f.name: f.default for f in dataclasses.fields(RuntimeHooks)}
    assert all(defaults[field] is False for field in SHARED_OPT_INS)
