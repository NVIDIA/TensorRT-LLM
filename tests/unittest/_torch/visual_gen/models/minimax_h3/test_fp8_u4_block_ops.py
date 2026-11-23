# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_103_107__u4 import (
    block_ops,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_103_107__u4.pipeline import (
    Fp8UlyssesAttention,
    Fp8UlyssesBlock,
)


class _HarnessBlock(block_ops.Fp8UlyssesMlpMixin, Fp8UlyssesBlock):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(
        self,
        hidden: torch.Tensor,
        normalized: torch.Tensor,
        mod: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_mlp_residual(hidden, normalized, mod, indices)


@pytest.mark.cpu_only
@pytest.mark.parametrize("precision,compiling", [(None, True), ("bf16", False), ("fp8", True)])
def test_forward_falls_back_to_stock(monkeypatch, precision, compiling):
    block = _HarnessBlock()
    assert block.attention_cls is Fp8UlyssesAttention
    block._u4_ffn_precision = precision
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    result = torch.zeros((1, 4, 3), dtype=torch.bfloat16)
    observed = []

    def stock(self, hidden_states, norm_hidden_states, mod, adaln_indices):
        observed.append((hidden_states, norm_hidden_states, mod, adaln_indices))
        return result

    monkeypatch.setattr(block_ops.MiniMaxH3TransformerBlock, "_forward_mlp_residual", stock)
    args = (
        torch.empty_like(result),
        torch.empty_like(result),
        torch.empty(1, 18, dtype=torch.bfloat16),
        torch.zeros(4, dtype=torch.int64),
    )
    with torch.inference_mode():
        assert block(*args) is result
    assert len(observed) == 1 and all(x is y for x, y in zip(observed[0], args, strict=True))


@pytest.mark.cpu_only
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_initialize_returns_before_loading_kernels_on_non_sm107(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 3))
    transformer = SimpleNamespace(
        proj_out=SimpleNamespace(weight=SimpleNamespace(device=torch.device(device)))
    )
    block_ops.initialize_block_ops(transformer)


def _require_sm107():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires a CUDA device with SM107")


class _Projection(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("input_scale", torch.full((), 0.01, device="cuda"))
        self.register_buffer("weight_scale", torch.full((), 0.02, device="cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == torch.bfloat16:
            return F.linear(x, self.weight)
        if x.dtype == torch.bfloat16:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.input_scale)
        return torch.ops.trtllm.cublas_scaled_mm(
            x,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=torch.bfloat16,
        )


class _FFN(torch.nn.Module):
    swiglu_limit = None
    swiglu_alpha = None
    swiglu_beta = None

    def __init__(self, precision: str, generator: torch.Generator) -> None:
        super().__init__()
        if precision == "bf16":
            dtype = torch.bfloat16
            scale = 0.01
        else:
            dtype = torch.float8_e4m3fn
            scale = 0.5
        gate_w = torch.randn(
            (28672, 5376), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        down_w = torch.randn(
            (5376, 14336), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        self.gate_up_proj = _Projection(gate_w.mul_(scale).to(dtype))
        self.down_proj = _Projection(down_w.mul_(scale).to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.gate_up_proj(x)
        activation = (
            swiglu(projected)
            if projected.dtype == self.down_proj.weight.dtype
            else swiglu(
                projected,
                quant_scale=self.down_proj.input_scale,
                quant_type=self.down_proj.weight.dtype,
            )
        )
        return self.down_proj(activation)


@pytest.mark.parametrize("precision,rows,padding", [("fp8", 9543, 3), ("bf16", 9455, 0)])
def test_compiled_ffn_matches_stock_bits_and_checks_shape_fallback(
    monkeypatch, precision, rows, padding
):
    _require_sm107()
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1.kernels import (
        gate_up_swiglu,
        projection_residual,
    )

    generator = torch.Generator(device="cuda").manual_seed(7423)
    block = _HarnessBlock().cuda().eval()
    block.ff = _FFN(precision, generator)
    # The loader and full real-shard dtype census are tested by the Rhea harness.
    device = block.ff.gate_up_proj.weight.device
    device_index = device.index
    bf16 = precision == "bf16"
    gate_up_swiglu._get_compiled(device_index, bf16=bf16)
    projection_residual.compile_kernel(
        device_index,
        5376,
        14336,
        projection_residual.DOWN_GATE,
        torch.bfloat16,
        torch.int64,
        bf16=bf16,
    )
    if bf16:
        block._u4_bf16_unit = torch.ones(1, dtype=torch.float32, device=device)
    else:
        block._u4_packed_gate_up = gate_up_swiglu.pack_weight(block.ff.gate_up_proj.weight)

    def data(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator).mul_(
            0.125
        )

    hidden, normalized = data((1, rows, 5376)), data((1, rows, 5376))
    modulation = data((2, 32256))
    # Rank 3 starts at 3 * rows in the shared Ulysses index tensor.
    full_indices = torch.arange(4 * rows, dtype=torch.int64, device=device) % 2
    indices = full_indices[3 * rows :]
    assert indices.data_ptr() % 16 == 8
    if padding:
        indices[-padding:] = 0
    compiled = torch.compile(block, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        block._u4_ffn_precision = None
        stock = compiled(hidden, normalized, modulation, indices)
        short_stock = compiled(hidden[:, :5], normalized[:, :5], modulation, indices[:5])
        block._u4_ffn_precision = precision
        fused = compiled(hidden, normalized, modulation, indices)
        short = compiled(hidden[:, :5], normalized[:, :5], modulation, indices[:5])
    assert torch.equal(fused.view(torch.int16), stock.view(torch.int16))
    assert torch.equal(short.view(torch.int16), short_stock.view(torch.int16))

    counts = {"gate": 0, "down": 0}
    for name, cache in (
        ("gate", gate_up_swiglu._compiled),
        ("down", projection_residual._compiled),
    ):
        for key, launch in list(cache.items()):

            def count(*args, name=name, launch=launch):
                counts[name] += 1
                return launch(*args)

            monkeypatch.setitem(cache, key, count)
    with torch.inference_mode():
        compiled(hidden, normalized, modulation, indices)
        assert counts == {"gate": 1, "down": 1}
        compiled(hidden[:, :5], normalized[:, :5], modulation, indices[:5])
        assert counts == {"gate": 1, "down": 1}
