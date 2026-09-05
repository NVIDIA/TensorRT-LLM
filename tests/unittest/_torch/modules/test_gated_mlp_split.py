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
"""Tests for GatedMLP's opt-in split gate/up topology.

Statically quantized checkpoints calibrate a separate weight scale per
projection. Fusing gate and up into one Linear forces a single scale on the pair
and re-quantizes the other onto it, discarding calibration. `split_gate_up=True`
keeps them separate so each loads its checkpoint tensor and scale unchanged.

The flag is opt-in and every existing caller relies on the fused default, so
these tests pin the default's topology and parameter names as much as they pin
the new path.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

HIDDEN, INTERMEDIATE, TOKENS = 512, 1024, 256
INPUT_SCALE, WEIGHT_SCALE = 1e-2, 1.1e-3

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make(split, *, quant_algo=QuantAlgo.FP8, activation=F.silu, force_dynamic=False):
    quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else QuantConfig()
    config = ModelConfig(quant_config=quant_config)
    config.force_dynamic_quantization = force_dynamic
    return GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        bias=False,
        activation=activation,
        dtype=torch.bfloat16,
        config=config,
        split_gate_up=split,
    ).cuda()


def _capture(fn):
    """Capture fn() after warming up on a side stream.

    Capture is sensitive to allocator state left by earlier tests, so the
    documented warmup protocol is used rather than a single default-stream call.
    """
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return out


def _fp8(rows, cols):
    return (torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.05).to(
        torch.float8_e4m3fn
    )


def _set_scales(linear, weight_scale=WEIGHT_SCALE, input_scale=INPUT_SCALE):
    linear.weight_scale.data.fill_(weight_scale)
    linear.input_scale.data.fill_(input_scale)
    linear.inv_input_scale.data.fill_(1.0 / input_scale)


def _set_weights(linear):
    """Give the projection finite weights.

    Linear allocates FP8 weights uninitialized, so a test that skips this reads
    whatever the caching allocator hands back -- NaN often enough to matter. Any
    torch.equal assertion then fails regardless of the behaviour under test,
    because torch.equal is False for NaN even when both sides are bit-identical.
    """
    linear.weight.data.copy_(_fp8(*linear.weight.shape))


@requires_cuda
def test_default_topology_is_unchanged():
    """The default must stay byte-for-byte what existing callers already build."""
    mlp = _make(False)

    assert mlp.split_gate_up is False
    assert isinstance(mlp.gate_up_proj, torch.nn.Module)
    assert mlp.gate_up_proj.out_features == INTERMEDIATE * 2
    assert not hasattr(mlp, "gate_proj")
    assert not hasattr(mlp, "up_proj")

    names = set(mlp.state_dict())
    assert any(n.startswith("gate_up_proj.") for n in names)
    assert not any(n.startswith("gate_proj.") or n.startswith("up_proj.") for n in names)


@requires_cuda
def test_split_topology_builds_separate_projections():
    mlp = _make(True)

    assert mlp.gate_up_proj is None
    assert mlp.gate_proj.out_features == INTERMEDIATE
    assert mlp.up_proj.out_features == INTERMEDIATE

    names = set(mlp.state_dict())
    assert any(n.startswith("gate_proj.") for n in names)
    assert any(n.startswith("up_proj.") for n in names)
    assert not any(n.startswith("gate_up_proj.") for n in names)


@requires_cuda
def test_split_projections_load_vanilla_weights_directly():
    """Each split Linear owns one checkpoint tensor -- no fused shard mapping."""
    from tensorrt_llm._torch.modules.linear import WeightMode

    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj):
        assert linear.weights_loading_config.weight_mode == WeightMode.VANILLA
        assert getattr(linear, "fused_weight_shard_indices_mapping", None) is None

    weight = _fp8(INTERMEDIATE, HIDDEN)
    mlp.gate_proj.load_weights(
        [
            {
                "weight": weight,
                "weight_scale": torch.tensor(WEIGHT_SCALE),
                "input_scale": torch.tensor(INPUT_SCALE),
            }
        ]
    )
    # Loaded exactly: no requantization onto a shared scale.
    assert torch.equal(mlp.gate_proj.weight.data, weight)
    assert mlp.gate_proj.weight_scale.item() == pytest.approx(WEIGHT_SCALE)


@requires_cuda
def test_split_matches_fused_when_scales_agree():
    """With one scale for both, fusion requantizes nothing, so results must match.

    This isolates the topology change from the scale effect: any difference here
    is a bug in the split path rather than the calibration it preserves.
    """
    gate_w, up_w, down_w = (
        _fp8(INTERMEDIATE, HIDDEN),
        _fp8(INTERMEDIATE, HIDDEN),
        _fp8(HIDDEN, INTERMEDIATE),
    )

    fused, split = _make(False), _make(True)
    fused.gate_up_proj.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))
    _set_scales(fused.gate_up_proj)
    split.gate_proj.weight.data.copy_(gate_w)
    split.up_proj.weight.data.copy_(up_w)
    _set_scales(split.gate_proj)
    _set_scales(split.up_proj)
    for mlp in (fused, split):
        mlp.down_proj.weight.data.copy_(down_w)
        _set_scales(mlp.down_proj)
    split.post_load_weights()

    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    assert torch.equal(fused(x), split(x))


@requires_cuda
def test_shared_input_is_quantized_once():
    """Both projections must receive the *same* already-quantized activation.

    Two Linears would otherwise each quantize the same values independently --
    which would also yield FP8 at both inputs, so dtype alone proves nothing.
    Storage identity is the discriminating check.
    """
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    seen = {}

    def record(name):
        def hook(_module, inputs):
            seen[name] = (inputs[0].dtype, inputs[0].data_ptr())

        return hook

    mlp.gate_proj.register_forward_pre_hook(record("gate"))
    mlp.up_proj.register_forward_pre_hook(record("up"))
    mlp(torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02)

    assert seen["gate"][0] == torch.float8_e4m3fn
    assert seen["up"][0] == torch.float8_e4m3fn
    # Same storage, not merely the same dtype: two independent quantizations of
    # the same values would also both be FP8, so identity is what proves the
    # activation was quantized once and shared.
    assert seen["gate"][1] == seen["up"][1]


@requires_cuda
def test_fp8_output_feeds_down_proj():
    """SwiGLU emits FP8 for an FP8 down_proj, so down_proj skips its own quant."""
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    seen = {}
    mlp.down_proj.register_forward_pre_hook(
        lambda _m, inputs: seen.__setitem__("dtype", inputs[0].dtype)
    )
    mlp(torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02)

    assert seen["dtype"] == torch.float8_e4m3fn


@requires_cuda
def test_post_load_weights_rejects_mismatched_input_scales():
    """The shared quantization is only valid if both scales agree.

    Checked once here rather than in forward: reading the tensors on the hot path
    would synchronize the device and make the graph data-dependent, which breaks
    fullgraph compilation.
    """
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    mlp.up_proj.input_scale.data.fill_(INPUT_SCALE * 2)
    with pytest.raises(ValueError, match="same calibrated input_scale"):
        mlp.post_load_weights()


@requires_cuda
def test_post_load_weights_is_noop_for_fused():
    _make(False).post_load_weights()


@requires_cuda
def test_post_load_weights_ignores_scales_for_bf16():
    mlp = _make(True, quant_algo=None)
    mlp.post_load_weights()  # must not raise


@requires_cuda
def test_torch_compile_fullgraph():
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    eager = mlp(x)
    compiled = torch.compile(lambda t: mlp(t), fullgraph=True)(x)

    assert compiled.dtype == eager.dtype
    assert torch.equal(compiled, eager)


@requires_cuda
def test_cuda_graph_capture():
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02

    captured = _capture(lambda: mlp(x))
    assert torch.equal(captured, mlp(x))


@requires_cuda
def test_lora_is_rejected_on_split_path():
    """forward_lora fuses gate/up LoRA into a projection the split path lacks.

    Failing here beats dereferencing gate_up_proj=None with a bare TypeError.
    """
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02

    with pytest.raises(NotImplementedError, match="LoRA is not supported"):
        mlp(x, lora_params={"any": "value"})


@requires_cuda
def test_non_swiglu_activation_is_rejected_on_split_path():
    mlp = _make(True, activation=F.gelu)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02

    with pytest.raises(NotImplementedError, match="requires SwiGLU"):
        mlp(x)


@requires_cuda
def test_split_rejects_forced_dynamic_quantization():
    """Dynamic quantization must use the fused topology.

    The activation emits FP8 with down_proj's calibrated scale, which would make
    down_proj skip the dynamic quantization it was configured for. Rejected at
    construction rather than silently downgraded.
    """
    with pytest.raises(ValueError, match="force_dynamic_quantization"):
        _make(True, force_dynamic=True)


@requires_cuda
def test_fused_topology_still_allows_forced_dynamic():
    """The rejection must be scoped to the split path only."""
    mlp = _make(False, force_dynamic=True)
    assert mlp.gate_up_proj is not None


@requires_cuda
def test_bf16_split_runs_without_quantization():
    """split_gate_up must not assume a quantized checkpoint."""
    mlp = _make(True, quant_algo=None)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02

    assert mlp._can_share_gate_up_quantization(x) is False
    out = mlp(x)
    assert out.shape == (TOKENS, HIDDEN)
    assert out.dtype == torch.bfloat16


@requires_cuda
@pytest.mark.parametrize(
    "shape", [(2, 8, HIDDEN), (4, TOKENS, HIDDEN)], ids=["small_rank3", "model_rank3"]
)
def test_rank3_activations(shape):
    """Models carry [batch, seq, hidden]; rank-2 tests alone would miss this.

    The op previously asserted rank 2 while its fake accepted rank 3, so tracing
    succeeded and execution failed.
    """
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    out = mlp(x)
    assert out.shape == (*shape[:-1], HIDDEN)

    # Same values through a rank-2 view must give the same result.
    flat = mlp(x.reshape(-1, shape[-1]))
    assert torch.equal(out.reshape(-1, HIDDEN), flat)


@requires_cuda
def test_rank3_compile_and_cuda_graph():
    mlp = _make(True)
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        _set_weights(linear)
        _set_scales(linear)
    mlp.post_load_weights()

    x = torch.randn(2, 8, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    eager = mlp(x)
    assert torch.equal(torch.compile(lambda t: mlp(t), fullgraph=True)(x), eager)

    captured = _capture(lambda: mlp(x))
    assert torch.equal(captured, mlp(x))
