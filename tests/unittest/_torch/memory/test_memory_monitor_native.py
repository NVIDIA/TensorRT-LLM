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
"""Memory that TensorRT-LLM's C++ code allocates outside PyTorch.

The end-to-end cases drive the real allocation sites -- the MoE GEMM
profiler and the expert load balancer's host-accessible allocator -- and
read the C++ ledger back through ``trtllm::get_native_memory_usage``.
"""

import gc

import pytest
import torch

from tensorrt_llm._torch.memory.monitor.providers.native import LABELS, NativeMemoryProvider
from tensorrt_llm._torch.memory.monitor.schema import Category
from tensorrt_llm._torch.memory.monitor.summary import GROUP_OF, Group

MB = 1 << 20


def _ledger():
    labels, counts, sizes = torch.ops.trtllm.get_native_memory_usage(torch.cuda.current_device())
    return {label: (count, size) for label, count, size in zip(labels, counts, sizes)}


def _with_fake_ledger(monkeypatch, rows):
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    sizes = [r[2] for r in rows]
    monkeypatch.setattr(
        NativeMemoryProvider, "_op", staticmethod(lambda: lambda _device: (labels, counts, sizes))
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


class TestLabels:
    def test_known_labels_land_in_their_groups(self, monkeypatch):
        _with_fake_ledger(
            monkeypatch,
            [
                ("nccl_communicators", 2, 900 * MB),
                ("moe_gemm_profiler_workspace", 1, 64 * MB),
                ("eplb_device", 4, 512 * MB),
            ],
        )
        entries = {e.owner: e for e in NativeMemoryProvider().collect()}
        assert GROUP_OF[entries["NCCL communicators"].category] is Group.COMMUNICATION
        assert GROUP_OF[entries["MoE GEMM profiler workspace"].category] is Group.WORKSPACE
        # Expert weight slots moved out of PyTorch are still weights.
        assert GROUP_OF[entries["MoE load balancer, host-accessible"].category] is Group.WEIGHTS

    def test_unknown_label_is_reported_under_its_own_name(self, monkeypatch):
        _with_fake_ledger(monkeypatch, [("new_cpp_buffer", 1, 32 * MB)])
        (entry,) = NativeMemoryProvider().collect()
        assert entry.owner == "new_cpp_buffer" and entry.size_bytes == 32 * MB

    def test_released_labels_are_not_listed(self, monkeypatch):
        _with_fake_ledger(monkeypatch, [("moe_gemm_profiler_workspace", 0, 0)])
        assert NativeMemoryProvider().collect() == []

    def test_negative_measurement_is_unknown_not_zero(self, monkeypatch):
        # Memory freed elsewhere on the device during creation.
        _with_fake_ledger(monkeypatch, [("nccl_communicators", 1, -4 * MB)])
        (entry,) = NativeMemoryProvider().collect()
        assert entry.size_bytes is None

    def test_memory_that_may_not_be_device_usage_is_not_summed(self, monkeypatch):
        _with_fake_ledger(monkeypatch, [("eplb_managed", 1, 64 * MB)])
        (entry,) = NativeMemoryProvider().collect()
        assert entry.counts_toward_total is False

    def test_provider_is_on_by_default(self):
        from tensorrt_llm._torch.memory.monitor.providers import default_providers

        assert any(isinstance(p, NativeMemoryProvider) for p in default_providers())

    def test_every_label_has_a_group(self):
        for label in LABELS.values():
            assert label.category in GROUP_OF and GROUP_OF[label.category] is not None


def _has_ledger():
    try:
        torch.ops.trtllm.get_native_memory_usage
    except (AttributeError, RuntimeError):
        return False
    return True


needs_ledger = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_ledger(), reason="needs a GPU and the C++ ledger"
)


@needs_ledger
def test_moe_gemm_profiler_workspace_is_recorded_while_profiling(monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    label = "moe_gemm_profiler_workspace"
    MoERunner.runner_dict.clear()
    gc.collect()
    before = _ledger().get(label, (0, 0))
    during = []
    profile = MoERunner.forward

    def recording_forward(self, inputs, gemm_idx=0, tactic=-1, do_preparation=False):
        out = profile(self, inputs, gemm_idx=gemm_idx, tactic=tactic, do_preparation=do_preparation)
        if do_preparation:
            during.append(_ledger().get(label, (0, 0)))
        return out

    monkeypatch.setattr(MoERunner, "forward", recording_forward)
    tokens, hidden, inter, experts, top_k = 16, 256, 512, 4, 2
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    w3_w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.02
    scores, ids = torch.topk(torch.randn(tokens, experts, device="cuda"), k=top_k, dim=-1)
    try:
        AutoTuner.get().clear_cache()
        with autotune():
            torch.ops.trtllm.fused_moe(
                input=x,
                token_selected_experts=ids.to(torch.int32),
                token_final_scales=torch.softmax(scores, dim=-1).float(),
                fc1_expert_weights=w3_w1,
                fc1_expert_biases=None,
                fc2_expert_weights=w2,
                fc2_expert_biases=None,
                output_dtype=torch.bfloat16,
                quant_scales=[],
            )
        # An OOM while tuning would find the workspace held and named ...
        assert during and all(c == before[0] + 1 and s > before[1] for c, s in during)
        # ... and the real MoE run that follows tuning has given it back.
        assert _ledger().get(label, (0, 0)) == before
    finally:
        MoERunner.runner_dict.clear()
        gc.collect()


@needs_ledger
def test_eplb_host_accessible_memory_is_recorded(monkeypatch):
    import tensorrt_llm.bindings.internal.runtime as tbr

    # Read when the allocator first initialises; lets any machine run this.
    monkeypatch.setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1")
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    balancer = tbr.MoeLoadBalancer(0, 1, 0)
    if not tbr.is_host_accessible_device_memory_supported():
        pytest.skip("host-accessible device memory unsupported here")
    before = {k: v for k, v in _ledger().items() if k.startswith("eplb_")}
    weight = torch.ones(16 * MB, dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.migrate_to_host_accessible(weight)
    after = {k: v for k, v in _ledger().items() if k.startswith("eplb_")}
    (label,) = [k for k in after if after[k] != before.get(k, (0, 0))]
    assert after[label][1] - before.get(label, (0, 0))[1] == 16 * MB
    assert LABELS[label].category is Category.EPLB
    from tensorrt_llm._torch.memory.monitor import capture, tagging

    # The migrated tensor stays a parameter of its module; claiming the
    # module must not count it as weights a second time.
    tagging.clear_claims()
    module = torch.nn.Module()
    module.slot = torch.nn.Parameter(weight, requires_grad=False)
    tagging.claim_module_memory(module, "WEIGHTS")
    try:
        entries = capture(tag="unit").frame.entries
    finally:
        tagging.clear_claims()
    assert sum(e.size_bytes for e in entries if e.category is Category.EPLB) >= 16 * MB
    assert sum(e.size_bytes or 0 for e in entries if e.category is Category.WEIGHTS) < 16 * MB
    del module, weight
    gc.collect()
    assert _ledger()[label] == before.get(label, (0, 0))
    del balancer
