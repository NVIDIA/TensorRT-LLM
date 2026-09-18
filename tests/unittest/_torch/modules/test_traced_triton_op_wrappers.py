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
"""Custom-op wrappers for raw Triton launchers on the torch.compile path.

A raw @triton.jit launcher traced by dynamo becomes a triton_kernel_wrapper
HOP whose functionalization clones the written buffer on every call (plus an
AOT copy-back); those DtoD copies were baked into decode CUDA graphs (~470
copies per replay on Qwen3.5). fused_sigmoid_gate_mul_add and
rms_norm_gated_token_major therefore dispatch to opaque custom ops when the
engine compiles the model. These tests pin: bitwise numerics parity between
the raw and custom-op paths, the absence of triton HOPs in the AOT graph, and
that remove_copy_for_mutates_args rewrites the mutable op back to a direct
in-place call (requires the inplace_map entry — omitting it makes the
following allreduce consume a functionalization clone, a correctness bug).
"""

import pytest
import torch

from tensorrt_llm._torch.compilation.remove_copy_pass import remove_copy_for_mutates_args
from tensorrt_llm._torch.moe.fused_shared_expert import fused_sigmoid_gate_mul_add
from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major
from tensorrt_llm._torch.utils import set_torch_compiling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS, HIDDEN = 96, 512
HEADS, HEAD_DIM = 4, 128


@pytest.fixture
def compiling_flag():
    """Restore the engine-level compiling flag around each test."""
    yield
    set_torch_compiling(False)
    torch._dynamo.reset()


def _sigmoid_inputs():
    torch.manual_seed(0)
    final = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(TOKENS, 1, dtype=torch.bfloat16, device="cuda")
    shared = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    return final, gate, shared


def test_fused_sigmoid_custom_op_matches_raw_launcher(compiling_flag):
    final, gate, shared = _sigmoid_inputs()

    set_torch_compiling(False)
    ref_inplace = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)
    out_buf = torch.empty_like(final)
    ref_outplace = fused_sigmoid_gate_mul_add(final, gate, shared, output=out_buf)

    set_torch_compiling(True)
    got_inplace = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)
    out_buf2 = torch.empty_like(final)
    got_outplace = fused_sigmoid_gate_mul_add(final, gate, shared, output=out_buf2)

    assert torch.equal(ref_inplace, got_inplace)
    assert torch.equal(ref_outplace, got_outplace)
    # The explicit-output form must write the caller's buffer (the following
    # allreduce consumes it by identity, e.g. a symmetric-heap allocation).
    assert got_outplace is out_buf2


def test_rms_norm_gated_custom_op_matches_raw_launcher(compiling_flag):
    torch.manual_seed(0)
    x = torch.randn(TOKENS * HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda")
    fp8_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    set_torch_compiling(False)
    ref = rms_norm_gated_token_major(x, z, weight, 1e-6)
    ref_fp8 = rms_norm_gated_token_major(x, z, weight, 1e-6, fp8_scale=fp8_scale)

    set_torch_compiling(True)
    got = rms_norm_gated_token_major(x, z, weight, 1e-6)
    got_fp8 = rms_norm_gated_token_major(x, z, weight, 1e-6, fp8_scale=fp8_scale)

    assert torch.equal(ref, got)
    assert got_fp8.dtype == torch.float8_e4m3fn
    assert torch.equal(ref_fp8.view(torch.uint8), got_fp8.view(torch.uint8))


def _capture_aot_forward(fn, example_inputs):
    from torch._functorch.aot_autograd import aot_module_simplified

    captured = []

    def backend(gm, inputs):
        def fw_compiler(fw_gm, fw_inputs):
            captured.append(fw_gm)
            return fw_gm

        return aot_module_simplified(gm, inputs, fw_compiler=fw_compiler)

    prev_v2 = torch._inductor.config.enable_auto_functionalized_v2
    torch._inductor.config.enable_auto_functionalized_v2 = False  # matches Backend
    try:
        torch.compile(fn, fullgraph=True, backend=backend)(*example_inputs)
    finally:
        torch._inductor.config.enable_auto_functionalized_v2 = prev_v2
    assert captured, "AOT forward graph was not captured"
    return captured[-1]


def _node_targets(gm):
    return [str(node.target) for node in gm.graph.nodes if node.op == "call_function"]


def test_compiled_graph_is_hop_free_and_remove_copy_restores_inplace(compiling_flag):
    set_torch_compiling(True)
    final, gate, shared = _sigmoid_inputs()

    def fn(final, gate, shared):
        # Mutate an intermediate (as production does: allocate_output / MoE
        # output), not a graph input.
        h = final * 1.0
        return fused_sigmoid_gate_mul_add(h, gate, shared)

    fw_gm = _capture_aot_forward(fn, (final, gate, shared))
    targets = _node_targets(fw_gm)
    assert not any("triton_kernel_wrapper" in t for t in targets), targets
    assert any("auto_functionalized" in t for t in targets), targets

    remove_copy_for_mutates_args(fw_gm.graph)
    targets = _node_targets(fw_gm)
    assert not any("auto_functionalized" in t for t in targets), targets
    assert any("fused_sigmoid_gate_mul_add_" in t for t in targets), targets


def test_compiled_rms_norm_is_single_opaque_call(compiling_flag):
    set_torch_compiling(True)
    torch.manual_seed(0)
    x = torch.randn(TOKENS * HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda")

    def fn(x, z, weight):
        return rms_norm_gated_token_major(x, z, weight, 1e-6)

    fw_gm = _capture_aot_forward(fn, (x, z, weight))
    targets = _node_targets(fw_gm)
    assert not any("triton_kernel_wrapper" in t for t in targets), targets
    assert any("rms_norm_gated_token_major" in t for t in targets), targets
