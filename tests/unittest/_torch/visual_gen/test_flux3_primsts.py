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

"""FP8 and NVFP4 PrimsTS parity against FP32 math and native FLUX.3 blocks."""

import copy
from functools import partial
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch
import torch.nn.functional as F
from flux3_test_utils import assert_parity
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils._python_dispatch import TorchDispatchMode

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as ops
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as flux3
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import Flux3FP8RowwiseLinearMethod
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3ModeBlock,
    Flux3SingleStreamBlock,
)
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_rowwise
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.fixture(autouse=True)
def reset_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _require_fp8_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("PrimsTS FP8 tests target SM100/SM103/SM107")
    assert primsts._has_fp8_prims_ts(), "Vendored PrimsTS FP8 GEMMs are unavailable"


_assert_fp8_parity = partial(assert_parity, zero_atol=1e-3)


def test_swiglu_packing_preserves_value_gate_pairs():
    """L1 T1: paired FP8 weights/scales reproduce silu(gate)*value."""
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.float8_e4m3fn)
    scale = torch.arange(1, 7, dtype=torch.float32)
    packed, packed_scale = ops.pack_swiglu_weight(weight, scale)
    x = torch.tensor([[0.1, -0.2, 0.3, -0.1]])
    gate, value = ((x @ weight.float().T) * scale).chunk(2, dim=-1)
    projected = ((x @ packed.float().T) * packed_scale).reshape(1, 3, 2)
    torch.testing.assert_close(projected[..., 0] * F.silu(projected[..., 1]), F.silu(gate) * value)


def test_fp8_fake_output_contracts():
    tokens = 17
    a = torch.empty(tokens, 512, device="meta", dtype=torch.float8_e4m3fn)
    w = torch.empty(1536, 512, device="meta", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty(tokens, device="meta")
    w_scale = torch.empty(1536, device="meta")
    norm = torch.empty(128, device="meta", dtype=torch.bfloat16)
    cos_sin = torch.empty(tokens, 128, device="meta")
    positions = torch.empty(tokens, device="meta", dtype=torch.int64)
    outputs = (
        (ops.fp8_linear(a, w, a_scale, w_scale), 1536),
        (ops.fp8_linear_swiglu(a, w, a_scale, w_scale), 768),
        (ops.fp8_qkv_qknorm_rope(a, w, a_scale, w_scale, norm, norm, cos_sin, positions), 1536),
    )
    for output, width in outputs:
        assert output.shape == (tokens, width)
        assert output.dtype == torch.bfloat16 and output.is_contiguous()


def _qkv_as_packed(out, activation, weight, qkv_scale=False):
    """Undo the Q, K and V planes when the config the op ran with writes them."""
    q, k, v = ops.qkv_planes(out, activation, weight, 128, qkv_scale=qkv_scale)
    return torch.cat((q, k, v), dim=-1)


@pytest.mark.parametrize("epilogue", ["linear", "swiglu", "qkv"])
def test_fp8_ops_math_and_registration(epilogue):
    """L1 T1: PrimsTS versus FP32 math, including BF16 QKV epilogue rounding."""
    _require_fp8_gpu()
    torch.manual_seed(21)
    n = 1536 if epilogue.startswith("qkv") else 1024
    a = torch.randint(-2, 3, (33, 512), device="cuda").to(torch.float8_e4m3fn)
    w = torch.randint(-2, 3, (n, 512), device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.rand(33, device="cuda") * 0.02 + 0.01
    w_scale = torch.rand(n, device="cuda") * 0.02 + 0.01
    args = (a, w, a_scale, w_scale)
    reference = (a.float() @ w.float().T) * a_scale[:, None] * w_scale
    if epilogue == "linear":
        op = ops.fp8_linear
    elif epilogue == "swiglu":
        op = ops.fp8_linear_swiglu
        paired = reference.reshape(33, n // 2, 2)
        reference = paired[..., 0] * F.silu(paired[..., 1])
    else:
        op = ops.fp8_qkv_qknorm_rope
        norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        # Identity RoPE isolates the fixed-epsilon QK normalization contract.
        cos_sin = torch.cat(
            (torch.ones(33, 64, device="cuda"), torch.zeros(33, 64, device="cuda")), dim=-1
        )
        positions = torch.arange(33, device="cuda", dtype=torch.int64)
        args += (norm, norm, cos_sin, positions)
        q, k, v = reference.bfloat16().reshape(33, 3, 4, 128).unbind(dim=1)
        reference = torch.cat(
            (
                F.rms_norm(q, (128,), norm, 1e-6).flatten(1),
                F.rms_norm(k, (128,), norm, 1e-6).flatten(1),
                v.flatten(1),
            ),
            dim=-1,
        )
    with torch.no_grad():
        snapshots = [tensor.view(torch.uint8).clone() for tensor in args]
        out = op(*args)
        _assert_fp8_parity(
            _qkv_as_packed(out, a, w) if epilogue == "qkv" else out, reference.bfloat16()
        )
        # Torch's schema checker uses unsupported FP8 allclose; compare bytes instead.
        for tensor, snapshot in zip(args[: len(snapshots)], snapshots, strict=True):
            assert torch.equal(tensor.view(torch.uint8), snapshot)
        torch.library.opcheck(
            op,
            args,
            test_utils=(
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ),
        )


def _make_fp8_block(block_type, eps=1e-6, mapping=None, hidden_size=512):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN),
        mapping=mapping or Mapping(),
    )
    block = (
        block_type(hidden_size, hidden_size // 128, config, mlp_ratio=2.0, eps=eps).cuda().eval()
    )
    with torch.no_grad():
        for parameter in block.parameters():
            if parameter.dtype != torch.float8_e4m3fn:
                parameter.fill_(1)
        for layer in block.modules():
            if isinstance(layer, Linear) and type(layer.quant_method) is FP8RowwiseLinearMethod:
                weight = torch.randn(layer.weight.shape, device="cuda", dtype=torch.bfloat16) * 0.02
                data, scales = quantize_fp8_rowwise(weight)
                layer.weight.copy_(data)
                layer.weight_scale.copy_(scales)
    return block


@pytest.mark.parametrize("block_type", [Flux3ModeBlock, Flux3SingleStreamBlock])
@pytest.mark.parametrize("execution", ["eager", "compile", "graph", "compile_graph"])
@pytest.mark.parametrize("package_recipe", [False, True])
def test_fp8_block_dispatch_parity(block_type, execution, package_recipe):
    """L2 T1: native versus PrimsTS with identical weights, including changed-input replay."""
    _require_fp8_gpu()
    torch.manual_seed(24)
    block = _make_fp8_block(block_type)
    if package_recipe:
        for layer in block.modules():
            if isinstance(layer, Linear) and type(layer.quant_method) is FP8RowwiseLinearMethod:
                layer.quant_method = Flux3FP8RowwiseLinearMethod()
    x = torch.randn(1, 66, 512, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(1, 66, 1, 64, device="cuda").repeat_interleave(2, dim=-1)
    freqs = (angles.cos(), angles.sin())
    kwargs = {"sample_bounds": ((0, 29), (29, 66))}
    if block_type is Flux3SingleStreamBlock:
        kwargs.update(pag_strength=0.3, pag_samples=(0,))
    with torch.inference_mode():
        reference = block(x, freqs, mod, **kwargs)
        changed_x = x * 0.5
        changed_reference = block(changed_x, freqs, mod, **kwargs)
        primsts.prepare_fp8_block(block)
        assert block.attn._primsts_fp8_qkv_rope
        assert block.ff.gate_up_proj._flux3_primsts_fp8.swiglu
        for layer in (block.attn.qkv_proj, block.attn_out, block.ff.down_proj):
            # MLP-down consumes the intermediate width, not the block width.
            operand = x.new_empty((*x.shape[:-1], layer.weight.shape[1]))
            assert layer._flux3_primsts_fp8.can_run(operand)
        run = block
        if "compile" in execution:
            backend = CompileCounterWithBackend("inductor")
            # Match production block compilation, keeping external GEMMs opaque.
            run = torch.compile(block, backend=backend, fullgraph=False)
        actual = run(x, freqs, mod, **kwargs)
        _assert_fp8_parity(actual, reference)
        if "compile" in execution:
            targets = {str(node.target) for gm in backend.graphs for node in gm.graph.nodes}
            for name in ("linear", "swiglu", "qkv_rope"):
                assert f"trtllm.flux3_primsts_fp8_{name}.default" in targets, targets
        if "graph" in execution:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(x, freqs, mod, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run(x, freqs, mod, **kwargs)
            x.copy_(changed_x)
            graph.replay()
            _assert_fp8_parity(captured, changed_reference)


def test_fp8_preparation_refresh_and_epsilon_fallback():
    _require_fp8_gpu()
    block = _make_fp8_block(Flux3ModeBlock, eps=1e-4)
    with torch.inference_mode():
        primsts.prepare_fp8_block(block)
        assert not block.attn._primsts_fp8_qkv_rope
        before = block.ff.gate_up_proj._flux3_primsts_fp8.weight.clone()
        block.ff.gate_up_proj.weight.zero_()
        primsts.prepare_fp8_block(block)
        after = block.ff.gate_up_proj._flux3_primsts_fp8.weight
        assert torch.count_nonzero(before.float()) > 0
        assert torch.count_nonzero(after.float()) == 0


@pytest.mark.parametrize("arch", [100, 103, 107])
@pytest.mark.parametrize("m", [20000, 40960, 65536])
def test_gemm_tactic_policy(arch, m):
    """Match SGLang's pinned tactics across GPUs, outputs, and large row counts."""
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import gemm_config

    config = gemm_config(arch, "fp8_e4m3", "linear", m, 3072, 3072)
    assert config.cluster_shape == (4, 4, 1) and config.use_tma_store
    fallback = gemm_config(arch, "fp8_e4m3", "linear", m, 768, 3072)
    assert fallback.cluster_shape == (2, 2, 1) and not fallback.use_tma_store
    assert gemm_config(arch, "fp8_e4m3", "linear", m, 3072, 3072, tune=True) is None
    packed = gemm_config(
        arch, "nvfp4_e2m1", "swiglu", 1024, 18432, 3072, output_format="nvfp4_e2m1"
    )
    assert packed.cluster_shape == (4, 4, 1) and packed.ab_stages == 5
    assert packed.tmem_overlap and not packed.use_tma_store
    qkv = gemm_config(arch, "nvfp4_e2m1", "qkv_qknorm_rope", m, 9216, 3072, head_dim=128)
    assert qkv.tmem_overlap and not qkv.use_tma_store and qkv.nvfp4_mma_k == 64


@pytest.mark.parametrize("arch", [100, 103])
@pytest.mark.parametrize("output_format", ["bf16", "nvfp4_e2m1"])
def test_nvfp4_swiglu_overlap_tactics(arch, output_format):
    """Offer SwiGLU overlap without allowing unsupported stores or tile shapes."""
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm.tactics import legal_tactics, tactic_is_legal

    tactics = legal_tactics(arch, "nvfp4_e2m1", output_format, "swiglu")
    assert any(tactic[4] for tactic in tactics)
    assert any(not tactic[4] for tactic in tactics)
    assert all(not (tactic[4] and tactic[9]) for tactic in tactics)
    overlap = (2, 2, 256, 256, True, 5, 64, True, 8, False)
    assert tactic_is_legal(arch, "nvfp4_e2m1", output_format, "swiglu", overlap)
    for index, value in ((2, 128), (8, 4), (9, True)):
        invalid = list(overlap)
        invalid[index] = value
        assert not tactic_is_legal(arch, "nvfp4_e2m1", output_format, "swiglu", tuple(invalid))
    assert not tactic_is_legal(arch, "nvfp4_e2m1", output_format, "qkv_qknorm_rope", overlap)


@pytest.mark.parametrize("arch", [100, 103, 107])
def test_nvfp4_qkv_overlap_tactics(arch):
    """QKV overlap retires one subtile and therefore requires MMA-K=64/direct stores."""
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm.tactics import legal_tactics, tactic_is_legal

    tactics = legal_tactics(arch, "nvfp4_e2m1", "bf16", "qkv_qknorm_rope")
    assert any(tactic[4] for tactic in tactics)
    assert any(not tactic[4] for tactic in tactics)
    overlap = (2, 2, 256, 256, True, 4, 64, True, 8, False)
    assert tactic_is_legal(arch, "nvfp4_e2m1", "bf16", "qkv_qknorm_rope", overlap)
    for index, value in ((2, 128), (3, 512), (6, 96), (8, 4), (9, True)):
        invalid = list(overlap)
        invalid[index] = value
        assert not tactic_is_legal(arch, "nvfp4_e2m1", "bf16", "qkv_qknorm_rope", tuple(invalid))


def test_prims_ts_warmup_before_capture(monkeypatch):
    """Tuning must finish with capture disabled before the graph warmup pass."""
    from contextlib import contextmanager

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_gemm_config
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import Flux3Pipeline

    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    state = {"capture": True, "tuning": False}
    calls = []

    @contextmanager
    def capture_off():
        state["capture"] = False
        try:
            yield
        finally:
            state["capture"] = True

    @contextmanager
    def autotune(device, rank, *, tune=True):
        assert device == "cuda:0" and rank == 0
        state["tuning"] = tune
        try:
            yield
        finally:
            state["tuning"] = False

    monkeypatch.setattr(flux3_gemm_config, "autotune_context", autotune)
    pipeline = SimpleNamespace(
        device="cuda:0",
        rank=0,
        pipeline_config=SimpleNamespace(cuda_graph=SimpleNamespace(enable=True)),
        disallow_cuda_graph_capture=capture_off,
        _run_flux3_warmup=lambda *args: calls.append((state.copy(), args)),
    )
    Flux3Pipeline._run_warmup(pipeline, 1024, 1024, 1, 2)
    assert calls == [
        ({"capture": False, "tuning": True}, (1024, 1024, 1, 2)),
        ({"capture": True, "tuning": False}, (1024, 1024, 1, 2)),
    ]


def test_prims_ts_source_digest(monkeypatch, tmp_path) -> None:
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import _gemm_source_digest
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    kernel = tmp_path / "kernel.py"
    monkeypatch.setattr(gemm, "__file__", str(tmp_path / "__init__.py"))
    _gemm_source_digest.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="kernel.py"):
            _gemm_source_digest()
        kernel.write_text("first version")
        first = _gemm_source_digest()
        kernel.write_text("second version")
        _gemm_source_digest.cache_clear()
        assert _gemm_source_digest() != first
    finally:
        _gemm_source_digest.cache_clear()


def test_prims_ts_tuning_disabled(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context

    monkeypatch.delenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", raising=False)
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", "invalid")
    with autotune_context("cpu", 0), autotune_context("cpu", 0, tune=False):
        pass


@pytest.mark.parametrize("raw, expected", [(None, None), ("33, 128,33", (33, 128))])
def test_prims_ts_tuning_options(monkeypatch, tmp_path, raw, expected):
    from contextlib import nullcontext

    import flashinfer

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (
        _gemm_source_digest,
        autotune_context,
    )

    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", str(tmp_path))
    if raw is None:
        monkeypatch.delenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", raw)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA B200")
    calls = []

    def tune(mode, **kwargs):
        calls.append((mode, kwargs))
        return nullcontext()

    monkeypatch.setattr(flashinfer, "autotune", tune)
    with autotune_context("cuda:0", 2):
        pass
    with autotune_context("cuda:0", 2, tune=False):
        pass
    assert calls == [
        (
            True,
            dict(
                cache=str(
                    tmp_path
                    / "NVIDIA_B200"
                    / "rank2"
                    / f"gemm-cold-l2-{_gemm_source_digest()}.json"
                ),
                tuning_buckets=expected,
                round_up=True,
            ),
        ),
        (False, dict(tuning_buckets=expected, round_up=True)),
    ]


@pytest.mark.parametrize("raw", ["", "0", "-1", "33,", "abc", "1.5"])
def test_prims_ts_invalid_tuning_buckets(monkeypatch, tmp_path, raw):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context

    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", raw)
    with pytest.raises(ValueError, match="positive comma-separated integers"):
        autotune_context("cuda:0", 0)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("precision", ["fp8", "nvfp4"])
def test_prims_ts_autotune_dispatch(monkeypatch, tmp_path, precision):
    """L1 T1: unpinned dispatch profiles real tactics and preserves GEMM output."""
    from flashinfer.autotuner import AutoTuner

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm.runner import PrimsTsGemmRunner

    _require_fp8_gpu()
    torch.manual_seed(48)
    a = torch.randn(33, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    if precision == "fp8":
        args = (
            a.to(torch.float8_e4m3fn),
            w.to(torch.float8_e4m3fn),
            torch.ones(33, device="cuda"),
            torch.ones(1024, device="cuda"),
        )
        op = ops.fp8_linear
        op_name = "prims_ts_fp8_linear"
    else:
        inverse = torch.tensor([10.0], device="cuda")
        aq, asc = ops.nvfp4_quantize(a, inverse)
        wq, wsc = ops.nvfp4_quantize(w, inverse)
        args = (aq, wq, asc, wsc, 0.1, 0.1)
        op = ops.nvfp4_linear
        op_name = "prims_ts_fp4_linear"
    reference = op(*args)
    original = PrimsTsGemmRunner.get_valid_tactics
    measure = AutoTuner._profile_single_kernel
    profiled = []
    measurements = []

    def candidates(self, inputs, profile):
        profiled.append(self.op_name)
        # Two tactics are enough to exercise profiling and cache reuse.
        return original(self, inputs, profile)[:2]

    def measure_kernel(self, *args, **kwargs):
        measurements.append(True)
        return measure(self, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", measure_kernel)
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())
    monkeypatch.setattr(PrimsTsGemmRunner, "get_valid_tactics", candidates)
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", "33")
    with autotune_context(a.device, 0):
        actual = op(*args)
    assert op_name in profiled and measurements
    _assert_fp8_parity(actual, reference)
    assert list(tmp_path.glob("*/rank0/gemm-cold-l2-*.json"))
    measurements.clear()
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())
    with autotune_context(a.device, 0):
        _assert_fp8_parity(op(*args), reference)
    assert not measurements, "Reloading the saved cache must not profile these shapes again"
    tuner = AutoTuner.get()
    search = tuner.search_cache
    hits = []

    def search_cache(*args, **kwargs):
        result = search(*args, **kwargs)
        hits.append(result[0])
        return result

    monkeypatch.setattr(tuner, "search_cache", search_cache)
    with autotune_context(a.device, 0, tune=False):
        _assert_fp8_parity(op(*args), reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(*args)
    assert hits and all(hits), "Inference must reuse a tuned tactic instead of the fallback"
    graph.replay()
    _assert_fp8_parity(captured, reference)


class _CountQuant(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in (
            "tensorrt_llm::quantize_e4m3_activation",
            "trtllm::tunable_fp8_per_token_quant",
        ):
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_cls", [flux3.Flux3ModeBlock, flux3.Flux3SingleStreamBlock])
@pytest.mark.parametrize(
    "compiled,fused_qknorm,tunable,fused_output",
    [
        (False, True, False, False),
        (False, False, True, False),
        (True, True, True, False),
        (False, True, True, True),
    ],
)
def test_shared_fp8_block(
    block_cls, compiled, fused_qknorm, tunable, fused_output, monkeypatch
) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FLUX.3 fused QK norm test requires Blackwell")
    torch.manual_seed(123)
    monkeypatch.setattr(FP8RowwiseLinearMethod, "use_tunable_quantize", tunable)
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN),
        dynamic_weight_quant=fused_output,
    )
    block = block_cls(256, 2, config).cuda().eval()
    block.attn.fuse_qk_norm_rope = fused_qknorm
    with torch.no_grad():
        for param in block.parameters():
            if param.dtype != torch.float8_e4m3fn:
                param.copy_(torch.randn_like(param) * 0.05)
        for layer in block.modules():
            if isinstance(layer, Linear) and isinstance(layer.quant_method, FP8RowwiseLinearMethod):
                weight = torch.randn(layer.weight.shape, device="cuda", dtype=torch.bfloat16) * 0.05
                data, scales = quantize_fp8_rowwise(weight)
                layer.weight.copy_(data)
                layer.weight_scale.copy_(scales)
    x = torch.randn(2, 19, 256, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(2, 19, 1, 128, device="cuda")
    freqs = (angles.cos(), angles.sin())
    kwargs = {"pag_strength": 0.5} if block_cls is flux3.Flux3SingleStreamBlock else {}
    with torch.inference_mode():
        with (
            patch.object(flux3, "_flux3_shared_fp8_input", return_value=None),
            _CountQuant() as baseline,
        ):
            expected = block(x, freqs, mod, **kwargs)
        with _CountQuant() as optimized:
            actual = block(x, freqs, mod, **kwargs)
        assert optimized.calls == baseline.calls - 1, (optimized.calls, baseline.calls)
        assert optimized.calls > 0
        if compiled:
            reference_backend = CompileCounterWithBackend("inductor")
            with patch.object(flux3, "_flux3_shared_fp8_input", return_value=None):
                expected = torch.compile(block, backend=reference_backend)(x, freqs, mod, **kwargs)
            assert reference_backend.frame_count > 0
            backend = CompileCounterWithBackend("inductor")
            actual = torch.compile(block, backend=backend)(x, freqs, mod, **kwargs)
            assert backend.frame_count > 0
    a, b = actual.float().flatten(), expected.float().flatten()
    delta = a - b
    relative_l2 = (delta.norm() / b.norm()).item()
    cosine = F.cosine_similarity(a, b, dim=0).item()
    metrics = dict(relative_l2=relative_l2, cosine=cosine, max_abs=delta.abs().max().item())
    print(block_cls.__name__, compiled, metrics)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert relative_l2 <= 1e-2 and cosine >= 0.9999, metrics
    # A quantization-excluded MLP must not receive an FP8 carrier.
    unquantized = Linear(256, 512, dtype=torch.bfloat16).cuda()
    assert flux3._flux3_shared_fp8_input(block.attn, unquantized, x) is None


def _require_nvfp4_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("PrimsTS NVFP4 targets SM100/SM103/SM107")
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    assert callable(gemm.fp4_qkv_qknorm_rope)


def _scale_indices(rows, cols, device):
    """Independent byte-offset oracle for the documented 128x4 scale layout."""
    row = torch.arange(rows, device=device)[:, None]
    col = torch.arange(cols, device=device)[None, :]
    return (
        col % 4
        + col // 4 * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + row // 128 * (128 * ((cols + 3) // 4 * 4))
    )


def _swizzle(scales):
    rows, cols = scales.shape
    output = torch.zeros(
        ops.nvfp4_scale_size(rows, cols * 16), device=scales.device, dtype=torch.uint8
    )
    output[_scale_indices(rows, cols, scales.device)] = scales.view(torch.uint8)
    return output


def _decode(data, scales, global_scale):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(1).long()
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    rows, k = values.shape
    scale = (
        scales[_scale_indices(rows, k // 16, data.device)]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    return values * scale.repeat_interleave(16, dim=1) * global_scale


_assert_nvfp4_parity = partial(assert_parity, zero_atol=1e-3)


@pytest.mark.parametrize("interleaved", [False, True])
def test_nvfp4_swiglu_packing(interleaved):
    """L1 T1: both native layouts preserve each row's bytes and scale association."""
    weight = torch.arange(256 * 128).reshape(256, 128).to(torch.uint8)
    scales = torch.arange(256 * 16).reshape(256, 16).to(torch.uint8)
    # Byte wraparound would otherwise make distinct rows identical.
    weight[:, 0] = torch.arange(256).to(torch.uint8)
    scales[:, 0] = weight[:, 0]
    native_weight, native_scales = weight, scales
    if interleaved:

        def native(t):
            return torch.cat((t[128:], t[:128])).reshape(2, 2, 64, -1).transpose(0, 1).reshape_as(t)

        native_weight, native_scales = native(weight), native(scales)
    packed, sf = ops.pack_nvfp4_swiglu_weight(
        native_weight, native_scales, interleaved_64=interleaved
    )
    torch.testing.assert_close(packed[0::2], weight[128:])
    torch.testing.assert_close(packed[1::2], weight[:128])
    torch.testing.assert_close(sf[0::2], scales[128:])
    torch.testing.assert_close(sf[1::2], scales[:128])


@pytest.mark.parametrize("tokens", [33, 129])
def test_nvfp4_fake_contract_and_fullgraph(tokens):
    """Meta tracing retains opaque external ops, including padded scale outputs."""
    from torch._dynamo.testing import CompileCounterWithBackend

    x = torch.empty(tokens, 512, device="meta", dtype=torch.bfloat16)
    inv = torch.empty(1, device="meta")
    weight = torch.empty(1536, 256, device="meta", dtype=torch.uint8)
    sf = torch.empty(ops.nvfp4_scale_size(1536, 512), device="meta", dtype=torch.uint8)
    norm = torch.empty(128, device="meta", dtype=torch.bfloat16)
    rope = torch.empty(tokens, 128, device="meta")
    pos = torch.empty(tokens, device="meta", dtype=torch.int64)

    def run(x):
        a, a_sf = ops.nvfp4_quantize(x, inv)
        args = (a, weight, a_sf, sf, 0.1, 0.2)
        return (
            ops.nvfp4_linear(*args),
            ops.nvfp4_linear_swiglu(*args),
            ops.nvfp4_qkv_qknorm_rope(*args, norm, norm, rope, pos),
            *ops.nvfp4_swiglu_packed(*args, inv),
        )

    counter = CompileCounterWithBackend("eager")
    outputs = torch.compile(run, backend=counter, fullgraph=True)(x)
    assert [out.shape for out in outputs] == [
        (tokens, 1536),
        (tokens, 768),
        (tokens, 1536),
        (tokens, 384),
        (ops.nvfp4_scale_size(tokens, 768),),
    ]
    assert all(out.dtype == torch.bfloat16 for out in outputs[:3])
    assert all(out.dtype == torch.uint8 for out in outputs[3:])
    targets = {str(node.target) for graph in counter.graphs for node in graph.graph.nodes}
    for name in ("quantize", "linear", "swiglu", "qkv_rope", "swiglu_packed"):
        assert f"trtllm.flux3_primsts_nvfp4_{name}.default" in targets


@pytest.mark.parametrize(
    "epilogue,k,n",
    [
        ("linear", 512, 1024),
        ("swiglu", 512, 1024),
        ("qkv", 512, 1536),
        ("qkv_scales", 512, 1536),
        # FLUX.3 TP4: six local heads, with QKV width not divisible by 512.
        ("qkv", 3072, 2304),
        ("qkv_scales", 3072, 2304),
    ],
)
def test_nvfp4_kernel_math_and_registration(epilogue, k, n):
    """L1 T1: scale-aware GEMM plus the real epilogue, against a decoded oracle."""
    _require_nvfp4_gpu()
    torch.manual_seed(31)
    m = 33
    a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    w = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    a_sf = _swizzle((torch.rand(m, k // 16, device="cuda") * 0.25).to(torch.float8_e4m3fn))
    w_sf = _swizzle((torch.rand(n, k // 16, device="cuda") * 0.25).to(torch.float8_e4m3fn))
    args = (a, w, a_sf, w_sf, 0.5, 0.25)
    reference = _decode(a, a_sf, 0.5) @ _decode(w, w_sf, 0.25).T
    if epilogue == "linear":
        op = ops.nvfp4_linear
    elif epilogue == "swiglu":
        op = ops.nvfp4_linear_swiglu
        reference = reference[:, 0::2] * F.silu(reference[:, 1::2])
    else:
        op = ops.nvfp4_qkv_qknorm_rope
        norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(m, 64, device="cuda")
        rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
        positions = torch.arange(m, device="cuda", dtype=torch.int64)
        args += (norm, norm, rope, positions)
        if epilogue == "qkv_scales":
            qkv_scale = torch.tensor([0.0625, 0.25, 0.375], device="cuda")
            args += (qkv_scale,)
            reference *= (qkv_scale / (0.5 * 0.25)).repeat_interleave(n // 3)
        qkv = reference.bfloat16().float().reshape(m, 3, n // (3 * 128), 128)
        for part in (0, 1):
            value = F.rms_norm(qkv[:, part], (128,), norm.float(), 1e-6)
            even, odd = value[..., 0::2], value[..., 1::2]
            cos, sin = angles.cos()[:, None], angles.sin()[:, None]
            qkv[:, part] = torch.stack(
                (even * cos - odd * sin, odd * cos + even * sin), dim=-1
            ).flatten(-2)
        reference = qkv.reshape(m, n)
    with torch.no_grad():
        out = op(*args)
        _assert_nvfp4_parity(
            _qkv_as_packed(out, a, w, epilogue == "qkv_scales")
            if epilogue.startswith("qkv")
            else out,
            reference.bfloat16(),
        )
        torch.library.opcheck(op, args)
        if epilogue == "qkv_scales":
            from tensorrt_llm._torch.visual_gen.prims_ts import gemm

            prepare_args = (w, w_sf, 0.5, 0.25, "qkv_qknorm_rope")
            prepared = ops.prepare_nvfp4_gemm(
                *prepare_args, q_norm=norm, k_norm=norm, qkv_scale=qkv_scale
            )
            assert (
                ops.prepare_nvfp4_gemm(*prepare_args, q_norm=norm, k_norm=norm, qkv_scale=qkv_scale)
                is prepared
            )
            qkv_scale.mul_(2)
            assert (
                ops.prepare_nvfp4_gemm(*prepare_args, q_norm=norm, k_norm=norm, qkv_scale=qkv_scale)
                is not prepared
            )
            expected = gemm.fp4_qkv_qknorm_rope(
                a,
                a_sf,
                0.5,
                w,
                w_sf,
                0.25,
                norm,
                norm,
                rope,
                positions,
                num_q_heads=n // (3 * 128),
                num_kv_heads=n // (3 * 128),
                head_dim=128,
                qkv_scale=qkv_scale,
            )
            _assert_nvfp4_parity(_qkv_as_packed(op(*args), a, w, True), expected)


def test_nvfp4_quantize_and_packed_swiglu():
    """L1 T1: adapter preserves FlashInfer quantization and prepared FP4-output behavior."""
    _require_nvfp4_gpu()
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as reference_quantize

    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    torch.manual_seed(32)
    x = torch.randn(33, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    inverse = torch.tensor([100.0], device="cuda")
    a, a_sf = ops.nvfp4_quantize(x, inverse)
    reference_a, reference_sf = reference_quantize(
        x,
        inverse,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    torch.testing.assert_close(a, reference_a.view(torch.uint8))
    # Padding bytes are unspecified; compare logical decoded elements instead.
    _assert_nvfp4_parity(
        _decode(a, a_sf, 0.01), _decode(reference_a, reference_sf.view(torch.uint8).flatten(), 0.01)
    )
    packed_w, w_sf = ops.nvfp4_quantize(w, inverse)
    args = (a, packed_w, a_sf, w_sf, 0.01, 0.01, inverse)
    actual, actual_sf = ops.nvfp4_swiglu_packed(*args)
    prepared = gemm.prepare_fp4_linear_swiglu(
        packed_w,
        w_sf,
        0.01,
        0.01,
        out_dtype=torch.uint8,
        output_quant_scale=inverse,
    )
    reference, reference_sf = prepared(a, a_sf)
    _assert_nvfp4_parity(_decode(actual, actual_sf, 0.01), _decode(reference, reference_sf, 0.01))
    # AOT opcheck includes unspecified padding; compare decoded values below.
    registration_checks = ("test_schema", "test_autograd_registration", "test_faketensor")
    torch.library.opcheck(ops.nvfp4_quantize, (x, inverse), test_utils=registration_checks)
    torch.library.opcheck(ops.nvfp4_swiglu_packed, args, test_utils=registration_checks)
    compiled_quant = torch.compile(ops.nvfp4_quantize, backend="inductor", fullgraph=True)
    compiled_swiglu = torch.compile(ops.nvfp4_swiglu_packed, backend="inductor", fullgraph=True)
    compiled_a, compiled_sf = compiled_quant(x, inverse)
    _assert_nvfp4_parity(_decode(compiled_a, compiled_sf, 0.01), _decode(a, a_sf, 0.01))
    compiled_output, compiled_sf = compiled_swiglu(*args)
    _assert_nvfp4_parity(
        _decode(compiled_output, compiled_sf, 0.01), _decode(actual, actual_sf, 0.01)
    )


@pytest.mark.parametrize("mma_k", [64, 96])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.uint8])
def test_nvfp4_swiglu_overlap_matches_plain(mma_k, out_dtype):
    """L1 T1: overlapped SwiGLU matches feature-off, including decoded FP4 output."""
    _require_nvfp4_gpu()
    if mma_k == 96 and torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("MMA-K=96 requires SM103")
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    torch.manual_seed(6255)
    # Multiple accumulator windows and an incomplete final block of scale rows.
    x = torch.randn(4097, 3072, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    inverse = torch.tensor([100.0], device="cuda")
    a, a_sf = ops.nvfp4_quantize(x, inverse)
    b, b_sf = ops.nvfp4_quantize(w, inverse)
    kwargs = dict(out_dtype=out_dtype, mma_k=mma_k)
    if out_dtype == torch.uint8:
        kwargs["output_quant_scale"] = inverse
    plain = gemm.prepare_fp4_linear_swiglu(b, b_sf, 0.01, 0.01, **kwargs)
    overlap = gemm.prepare_fp4_linear_swiglu(b, b_sf, 0.01, 0.01, tmem_overlap=True, **kwargs)
    assert not plain.config.tmem_overlap and overlap.config.tmem_overlap
    assert overlap._module.acc_stages == 2
    expected, actual = plain(a, a_sf), overlap(a, a_sf)
    if out_dtype == torch.uint8:
        # Decode only real rows; padding in the final scale block is unwritten.
        expected = _decode(*expected, 0.01)
        actual = _decode(*actual, 0.01)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert_parity(actual, expected)


@pytest.mark.parametrize("scaled_qkv", [False, True])
def test_nvfp4_qkv_overlap_matches_plain(scaled_qkv):
    """L1 T1: overlapped QKV output matches feature-off through CUDA graph replay."""
    _require_nvfp4_gpu()
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    torch.manual_seed(6286)
    m, k, heads = 4097, 3072, 4
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3 * heads * 128, k, device="cuda", dtype=torch.bfloat16) * 0.1
    inverse = torch.tensor([100.0], device="cuda")
    a, a_sf = ops.nvfp4_quantize(x, inverse)
    b, b_sf = ops.nvfp4_quantize(w, inverse)
    q_norm = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    k_norm = torch.randn_like(q_norm)
    angles = torch.randn(m, 64, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(m - 1, -1, -1, device="cuda", dtype=torch.int64)
    kwargs = dict(num_q_heads=heads, num_kv_heads=heads, head_dim=128, is_neox=False)
    if scaled_qkv:
        kwargs["qkv_scale"] = torch.tensor([0.0001, 0.0002, 0.0003], device="cuda")
    plain = gemm.prepare_fp4_qkv_qknorm_rope(b, b_sf, 0.01, 0.01, q_norm, k_norm, **kwargs)
    overlap = gemm.prepare_fp4_qkv_qknorm_rope(
        b, b_sf, 0.01, 0.01, q_norm, k_norm, tmem_overlap=True, **kwargs
    )
    assert not plain.config.tmem_overlap and overlap.config.tmem_overlap
    assert overlap._module.acc_stages == 2
    args = dict(cos_sin=rope, positions=positions)
    expected, actual = plain(a, a_sf, **args), overlap(a, a_sf, **args)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert_parity(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = overlap(a, a_sf, **args)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=1e-3, atol=1e-3)
    assert_parity(captured, expected)


@pytest.mark.parametrize("precision", ["fp8", "nvfp4"])
def test_qkv_cached_positions(precision):
    """L1 T1: cached identity positions match explicit indices in eager and graph replay."""
    _require_nvfp4_gpu()
    torch.manual_seed(27)
    m, k, n = 33, 512, 1536
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    if precision == "nvfp4":
        inverse = torch.tensor([100.0], device="cuda")
        a, a_sf = ops.nvfp4_quantize(x, inverse)
        b, b_sf = ops.nvfp4_quantize(w, inverse)
        op = ops.nvfp4_qkv_qknorm_rope
        args = (a, b, a_sf, b_sf, 0.01, 0.01)
    else:
        op = ops.fp8_qkv_qknorm_rope
        args = (
            x.to(torch.float8_e4m3fn),
            w.to(torch.float8_e4m3fn),
            torch.ones(m, device="cuda"),
            torch.ones(n, device="cuda"),
        )
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(m, 64, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    args += (norm, norm, rope)
    expected = op(*args, torch.arange(m, device="cuda", dtype=torch.int64))
    assert_parity(op(*args, None), expected)
    hits = ops._token_positions.cache_info().hits
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(*args, None)
    assert ops._token_positions.cache_info().hits > hits
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=1e-3, atol=1e-3)
    assert_parity(captured, expected)
    torch.library.opcheck(
        op,
        (*args, None),
        test_utils=("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def _make_nvfp4_block(block_type, mapping=None, hidden_size=512):
    from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        mapping=mapping or Mapping(),
    )
    block = block_type(hidden_size, hidden_size // 128, config, mlp_ratio=2.0).cuda().eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(1)
        for layer in list(block.modules()):
            if not isinstance(layer, Linear) or type(layer.quant_method) is not NVFP4LinearMethod:
                continue
            inverse = torch.tensor([100.0], device="cuda")
            w = (
                torch.randn(
                    layer.weight.shape[0],
                    layer.weight.shape[1] * 2,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.02
            )
            if layer is block.ff.gate_up_proj and layer.use_cute_dsl_blockscaling_mm:
                n = w.shape[0]
                w = (
                    torch.cat((w[n // 2 :], w[: n // 2]))
                    .reshape(2, n // 128, 64, -1)
                    .transpose(0, 1)
                    .reshape_as(w)
                    .contiguous()
                )
            weight, scales = ops.nvfp4_quantize(w, inverse)
            layer.weight.copy_(weight)
            layer.weight_scale.copy_(scales)
            layer.input_scale.fill_(100)
            layer.inv_input_scale.fill_(100 / 6)
            layer.weight_scale_2.fill_(0.01)
            layer.alpha.fill_(0.0001)
    return block


@pytest.mark.parametrize("kind", ["mode", "single"])
@pytest.mark.parametrize("execution", ["eager", "compile", "graph", "compile_graph"])
def test_nvfp4_block_parity(kind, execution):
    """L2 T1: feature-off block versus all PrimsTS paths; changed-input replay."""
    _require_nvfp4_gpu()
    from torch._dynamo.testing import CompileCounterWithBackend

    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        Flux3ModeBlock,
        Flux3SingleStreamBlock,
    )

    torch.manual_seed(34)
    block = _make_nvfp4_block(Flux3ModeBlock if kind == "mode" else Flux3SingleStreamBlock)
    with torch.no_grad():
        for layer, scale in ((block.attn.to_k, 0.02), (block.attn.to_v, 0.03)):
            layer.weight_scale_2.fill_(scale)
            layer.alpha.fill_(0.01 * scale)
    x = torch.randn(1, 66, 512, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(1, 66, 1, 64, device="cuda").repeat_interleave(2, dim=-1)
    freqs = angles.cos(), angles.sin()
    kwargs = {"sample_bounds": ((0, 29), (29, 66))}
    if kind == "single":
        kwargs.update(pag_strength=0.3, pag_samples=(0,))
    with torch.inference_mode():
        expected = block(x, freqs, mod, **kwargs)
        changed = x * 0.5
        changed_expected = block(changed, freqs, mod, **kwargs)
        primsts.prepare_nvfp4_block(block)
        assert block.attn._flux3_primsts_nvfp4_qkv is not None
        assert block.ff.gate_up_proj._flux3_primsts_nvfp4.swiglu
        assert primsts.shared_nvfp4_input(block.attn, block.ff.gate_up_proj, x) is not None
        run = block
        if "compile" in execution:
            counter = CompileCounterWithBackend("inductor")
            # Match production block compilation, keeping external GEMMs opaque.
            run = torch.compile(block, backend=counter, fullgraph=False)
        _assert_nvfp4_parity(run(x, freqs, mod, **kwargs), expected)
        if "compile" in execution:
            targets = {str(node.target) for gm in counter.graphs for node in gm.graph.nodes}
            for name in ("quantize", "linear", "swiglu", "qkv"):
                assert f"trtllm.flux3_nvfp4_{name}_prepared.default" in targets, sorted(targets)
        if "graph" in execution:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(x, freqs, mod, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run(x, freqs, mod, **kwargs)
            x.copy_(changed)
            graph.replay()
            _assert_nvfp4_parity(actual, changed_expected)


def test_nvfp4_missing_scale_matches_unit_scale_policy(monkeypatch):
    """Missing-scale adapters match a block with explicit unit input scales."""
    _require_nvfp4_gpu()
    from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3ModeBlock

    torch.manual_seed(38)
    block = _make_nvfp4_block(Flux3ModeBlock)
    reference = copy.deepcopy(block)
    layers = [
        layer
        for layer in block.modules()
        if isinstance(layer, Linear) and type(layer.quant_method) is NVFP4LinearMethod
    ]
    with torch.inference_mode():
        for layer in reference.modules():
            if isinstance(layer, Linear) and type(layer.quant_method) is NVFP4LinearMethod:
                layer.input_scale.fill_(1)
                layer.inv_input_scale.fill_(1 / 6)
                layer.alpha.copy_(layer.weight_scale_2)
                layer.scalar_alpha = float(layer.alpha.item())
        for layer in layers:
            layer.force_dynamic_quantization = True
            layer._flux3_nvfp4_missing_input_scale = True
        x = torch.randn(1, 33, 512, device="cuda", dtype=torch.bfloat16)
        mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
        angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, dim=-1)
        freqs = angles.cos(), angles.sin()
        native_dynamic = block(x, freqs, mod)
        expected = reference(x, freqs, mod)
        primsts.prepare_nvfp4_block(block)
        assert block.attn._flux3_primsts_nvfp4_qkv is not None
        assert all(layer._flux3_primsts_nvfp4.input_global_scale == 1 for layer in layers)
        _assert_nvfp4_parity(block(x, freqs, mod), expected)
        for layer in layers:
            assert layer.force_dynamic_quantization
            torch.testing.assert_close(layer.input_scale, torch.full_like(layer.input_scale, 100))
        monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS", "0")
        primsts.prepare_nvfp4_block(block)
        assert all(layer._flux3_primsts_nvfp4 is None for layer in layers)
        _assert_nvfp4_parity(block(x, freqs, mod), native_dynamic)


def test_nvfp4_scale_mismatch_refresh_and_dynamic_fallback():
    """L2 T1: unequal weight scales match native attention; refreshed state stays valid."""
    _require_nvfp4_gpu()
    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3ModeBlock

    block = _make_nvfp4_block(Flux3ModeBlock)
    with torch.inference_mode():
        block.attn.to_k.weight_scale_2.fill_(0.02)
        block.attn.to_k.alpha.fill_(0.0002)
        x = torch.randn(1, 33, 512, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, dim=-1)
        freqs = angles.cos(), angles.sin()
        expected = block.attn.attn_pre_out(x, freqs)
        primsts.prepare_nvfp4_block(block)
        assert block.attn._flux3_primsts_nvfp4_qkv is not None
        assert all(
            layer._flux3_primsts_nvfp4 is not None
            for layer in (block.attn.to_q, block.attn.to_k, block.attn.to_v)
        )
        _assert_nvfp4_parity(block.attn.attn_pre_out(x, freqs), expected)
        shared = primsts.shared_nvfp4_input(block.attn, block.ff.gate_up_proj, x)
        assert shared is not None
        # Cached activations are valid only for the same global scale.
        k_adapter = block.attn.to_k._flux3_primsts_nvfp4
        k_adapter.input_global_scale *= 2
        k_adapter.inverse_scale.mul_(0.5)
        # Scale reconciliation is opaque to Dynamo and happens inside the GEMM op.
        _assert_nvfp4_parity(k_adapter(shared), k_adapter(x))
        before = block.ff.gate_up_proj._flux3_primsts_nvfp4.weight.clone()
        block.ff.gate_up_proj.weight.zero_()
        primsts.prepare_nvfp4_block(block)
        assert before.count_nonzero() > 0
        assert block.ff.gate_up_proj._flux3_primsts_nvfp4.weight.count_nonzero() == 0
        block.attn.to_q.force_dynamic_quantization = True
        primsts.prepare_nvfp4_block(block)
        assert block.attn.to_q._flux3_primsts_nvfp4 is None


@pytest.mark.parametrize("kind", ["mode", "single"])
def test_nvfp4_shared_activation_avoids_duplicate_quantization(kind):
    """L2 T1: same PrimsTS block with reuse off/on; one fewer quantizer call."""
    _require_nvfp4_gpu()
    from torch.utils._python_dispatch import TorchDispatchMode

    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as flux3

    class CountQuant(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func._schema.name == "trtllm::flux3_nvfp4_quantize_prepared":
                self.calls += 1
            return func(*args, **(kwargs or {}))

    torch.manual_seed(35)
    block = _make_nvfp4_block(
        flux3.Flux3ModeBlock if kind == "mode" else flux3.Flux3SingleStreamBlock
    )
    x = torch.randn(1, 33, 512, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, dim=-1)
    freqs = angles.cos(), angles.sin()
    with torch.inference_mode():
        primsts.prepare_nvfp4_block(block)
        with patch.object(flux3, "_flux3_shared_fp8_input", return_value=None), CountQuant() as off:
            expected = block(x, freqs, mod)
        with CountQuant() as on:
            actual = block(x, freqs, mod)
        assert (off.calls, on.calls) == (3, 2)
        _assert_nvfp4_parity(actual, expected)


def test_nvfp4_prepared_launch_reuse_and_refresh():
    """Prepared GEMM matches a direct vendored GEMM after weight and scale updates."""
    _require_nvfp4_gpu()
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    torch.manual_seed(36)
    x = torch.randn(33, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16) * 0.02
    inverse = torch.tensor([100.0], device="cuda")
    a, a_sf = ops.nvfp4_quantize(x, inverse)
    weight, scale = ops.nvfp4_quantize(w, inverse)
    with (
        torch.no_grad(),
        patch.object(gemm, "prepare_fp4_linear", wraps=gemm.prepare_fp4_linear) as prepare,
    ):
        for _ in range(2):
            actual = ops.nvfp4_linear(a, weight, a_sf, scale, 0.01, 0.01)
        assert prepare.call_count == 1
        expected = gemm.fp4_linear(a, a_sf, 0.01, weight, scale, 0.01)
        _assert_nvfp4_parity(actual, expected)
        weight.zero_()
        actual = ops.nvfp4_linear(a, weight, a_sf, scale, 0.01, 0.01)
        assert prepare.call_count == 2
        _assert_nvfp4_parity(actual, torch.zeros_like(actual))
        ops.nvfp4_linear(a, weight, a_sf, scale, 0.01, 0.02)
        assert prepare.call_count == 3


@pytest.mark.parametrize("include_bf16", [False, True])
def test_nvfp4_coverage_fails_closed_and_native_escape(monkeypatch, include_bf16):
    """Missing fused QKV must fail model preparation rather than conceal fallback."""
    _require_nvfp4_gpu()
    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3ModeBlock

    block = _make_nvfp4_block(Flux3ModeBlock)
    blocks = [("image.0", block)]
    if include_bf16:
        bf16 = (
            Flux3ModeBlock(
                512, 4, DiffusionModelConfig(pretrained_config=SimpleNamespace()), mlp_ratio=2.0
            )
            .cuda()
            .eval()
        )
        blocks.append(("image.1", bf16))
    monkeypatch.delenv("TRTLLM_FLUX3_PRIMS_TS", raising=False)
    with torch.inference_mode():
        primsts.prepare_nvfp4_block(block)
        primsts.verify_nvfp4_coverage(blocks)
        block.attn.to_k.input_scale.fill_(50)
        primsts.prepare_nvfp4_block(block)
        with pytest.raises(RuntimeError, match=r"image\.0\.attn.*QKV"):
            primsts.verify_nvfp4_coverage(blocks)
        monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS", "0")
        primsts.prepare_nvfp4_block(block)
        primsts.verify_nvfp4_coverage(blocks)
        assert block.ff.gate_up_proj._flux3_primsts_nvfp4 is None


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
@pytest.mark.parametrize("block_type", [Flux3ModeBlock, Flux3SingleStreamBlock])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_local_block_dispatch_parity(recipe, block_type, rank):
    """T1 on real TP-local shapes; collective integration needs the multi-GPU suite."""
    _require_fp8_gpu() if recipe == "fp8" else _require_nvfp4_gpu()
    torch.manual_seed(42 + rank)
    make = _make_fp8_block if recipe == "fp8" else _make_nvfp4_block
    prepare = primsts.prepare_fp8_block if recipe == "fp8" else primsts.prepare_nvfp4_block
    compare = _assert_fp8_parity if recipe == "fp8" else _assert_nvfp4_parity
    # Only suppress the collective: weights and head dimensions remain genuinely sharded.
    with (
        patch("tensorrt_llm.mapping.mpi_disabled", return_value=False),
        patch.object(Mapping, "tp_group_pg", new_callable=PropertyMock, return_value=None),
    ):
        # Use arithmetic rank metadata even when collected with torch.distributed tests.
        mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        block = make(block_type, mapping=mapping, hidden_size=1024)
    # This test validates TP-local math without an initialized process group.
    block._allreduce = torch.nn.Identity()
    assert block.attn.tp_size == 2
    assert block.attn.local_q_dim == 512
    assert block.attn_out.weight.shape[1] == (256 if recipe == "nvfp4" else 512)
    x = torch.randn(1, 33, 1024, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, -1)
    freqs = angles.cos(), angles.sin()
    with torch.inference_mode():
        expected = block(x, freqs, mod)
        prepare(block)
        if recipe == "fp8":
            assert block.attn._primsts_fp8_qkv_rope
        else:
            assert block.attn._flux3_primsts_nvfp4_qkv is not None
        assert getattr(block.ff.gate_up_proj, f"_flux3_primsts_{recipe}").swiglu
        compare(block(x, freqs, mod), expected)
        eligible = primsts._eligible_fp8 if recipe == "fp8" else primsts._eligible_nvfp4
        assert eligible(block.attn_out)
        block.attn_out.reduce_output = True
        assert not eligible(block.attn_out), "An adapter must not skip a required all-reduce"
        projection = block.attn.qkv_proj if recipe == "fp8" else block.attn.to_q
        assert eligible(projection)
        projection.gather_output = True
        assert not eligible(projection), "An adapter must not skip a required all-gather"


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (10, 7), (12, 0), (12, 1)])
def test_primsts_architecture_dispatch(recipe, capability, monkeypatch):
    """Keep FP8 and NVFP4 architecture eligibility distinct."""
    _require_fp8_gpu() if recipe == "fp8" else _require_nvfp4_gpu()
    make = _make_fp8_block if recipe == "fp8" else _make_nvfp4_block
    block = make(Flux3ModeBlock)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    eligible = primsts._eligible_fp8 if recipe == "fp8" else primsts._eligible_nvfp4
    supported = ((10, 0), (10, 3), (10, 7))
    if recipe == "nvfp4":
        supported += ((12, 0), (12, 1))
    assert eligible(block.attn_out) == (capability in supported)


@pytest.mark.parametrize("block_type", [Flux3ModeBlock, Flux3SingleStreamBlock])
@pytest.mark.parametrize("sm", [100, 103, 107, 120])
def test_native_nvfp4_swiglu_architecture_dispatch(block_type, sm, monkeypatch):
    """Rubin must not select the SM100/SM103-only CuteDSL SwiGLU kernel."""
    _require_nvfp4_gpu()
    monkeypatch.setattr(flux3, "get_sm_version", lambda: sm)
    block = _make_nvfp4_block(block_type)
    assert block.ff.use_cute_dsl_blockscaling_mm == (sm in (100, 103))
    if sm not in (100, 103):
        assert not block.ff._can_fuse_gate_up_swiglu()
        assert not block.ff._can_fuse_gate_up_swiglu_fp4out()
