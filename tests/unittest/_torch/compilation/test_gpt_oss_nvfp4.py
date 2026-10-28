# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the GPT-OSS norm boundary through TRT-LLM's actual compiler."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.models.modeling_gpt_oss import TransformerBlock
from tensorrt_llm._torch.modules.fused_ops import rmsnorm_fp4_quant as fused_norm
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import (
    _add_rmsnorm_fp4_quant_out,
    add_rmsnorm_fp4_quant_available,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import piecewise_cuda_graph, torch_compiling


class _QuantParams(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([100.0], device="cuda"))

    def get_nvfp4_input_quant_params(self) -> tuple[torch.Tensor, int]:
        return self.scale, 3072


class _MlpInput(torch.nn.Module):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        block = TransformerBlock.__new__(TransformerBlock)
        torch.nn.Module.__init__(block)
        block.enable_fused_mlp_nvfp4_input = enabled
        block.post_attention_layernorm = RMSNorm(
            hidden_size=2880, eps=1e-5, dtype=torch.bfloat16
        ).cuda()
        block.mlp = torch.nn.Module()
        block.mlp.experts = _QuantParams()
        self.block = block

    def forward(
        self, inputs_embeds: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.block._try_fused_mlp_nvfp4_input(inputs_embeds, residual, None)
        if fused is not None:
            carrier, residual = fused
            return (
                carrier.fp4_tensor,
                carrier.scaling_factor,
                carrier.unquantized_hidden_states,
                residual,
            )
        normalized, residual = self.block.post_attention_layernorm(inputs_embeds, residual)
        fp4, sf = torch.ops.trtllm.fp4_quantize(
            F.pad(normalized, (0, 192)), self.block.mlp.experts.scale, 16, False, False
        )
        return fp4, sf.view(-1, 192), normalized, residual


@pytest.mark.skipif(
    not torch.cuda.is_available() or not add_rmsnorm_fp4_quant_available(),
    reason="requires CUDA and FlashInfer add RMSNorm FP4 quantization",
)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("direct_padding", [True, False])
@pytest.mark.parametrize(
    "piecewise, enable_inductor", [(False, False), (True, False), (False, True)]
)
@torch.inference_mode()
def test_gpt_oss_nvfp4_compiled_prefill(
    enabled: bool,
    piecewise: bool,
    direct_padding: bool,
    enable_inductor: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    if direct_padding and not fused_norm._ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING:
        pytest.skip("requires FlashInfer direct padded output support")
    monkeypatch.setattr(fused_norm, "_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", direct_padding)
    # PiecewiseRunner normally receives these attributes from ModelEngine.
    from tensorrt_llm._torch.utils import _model_extra_attrs

    monkeypatch.setattr(
        _model_extra_attrs, "attrs", {"global_stream": torch.cuda.current_stream()}, raising=False
    )
    torch.manual_seed(42)
    model = _MlpInput(enabled)
    backend = Backend(
        enable_inductor=enable_inductor,
        enable_piecewise_cuda_graph=piecewise,
        capture_num_tokens=[128, 256, 512, 8192],
    )
    graphs = []

    def record_backend(gm: torch.fx.GraphModule, examples: list[torch.Tensor]):
        graphs.append(gm)
        return backend(gm, examples)

    compiled = torch.compile(model, backend=record_backend, fullgraph=True)
    # Token padding is done by ModelEngine, not by PiecewiseRunner itself.
    cases = [(127, 128), (128, 128), (129, 256), (255, 256), (256, 256), (257, 512), (7700, 8192)]
    if not piecewise:
        cases = [(n, n) for n in (7, 127, 128, 129, 255, 256, 257, 1024, 7700)]
    x_storage = torch.empty((8192, 2880), device="cuda", dtype=torch.bfloat16)
    residual_storage = torch.empty_like(x_storage)
    try:
        for live, physical in cases:
            x = x_storage[:physical]
            residual = residual_storage[:physical]
            torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
            torch._dynamo.mark_dynamic(residual, 0, min=2, max=8192)
            # Three warmups, capture, then repeated changed-data replay.
            for iteration in range(7):
                source_x = torch.randn_like(x)
                source_residual = torch.randn_like(residual)
                source_x[live:] = 0
                source_residual[live:] = 0
                with torch_compiling(False):
                    expected = model(source_x.clone(), source_residual.clone())
                x.copy_(source_x)
                residual.copy_(source_residual)
                with (
                    torch_compiling(True),
                    piecewise_cuda_graph(piecewise),
                    capture_piecewise_cuda_graph(piecewise),
                ):
                    actual = compiled(x, residual)
                torch.cuda.synchronize()
                if iteration == 0:
                    targets = [str(node.target) for gm in graphs for node in gm.graph.nodes]
                    assert any("add_rmsnorm_fp4_quant_out" in name for name in targets) is enabled
                for out, ref in zip(actual, expected, strict=True):
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
                assert actual[0].dtype == actual[1].dtype == torch.uint8
                assert actual[0].shape == (physical, 1536)
                assert actual[1].shape == (physical, 192)
                assert actual[2].shape == (physical, 2880)
                if enabled:
                    assert actual[3].data_ptr() == residual.data_ptr()
                assert not actual[0][:, 1440:].count_nonzero().item()
                assert not actual[1][:, 180:].count_nonzero().item()
        # Dynamic token sizes must share the traced boundary (no silent graph breaks).
        assert len(graphs) == 1
        if piecewise:
            runners = list(backend._piecewise_runners)
            assert runners
            assert all(
                r.entries[n].cuda_graph is not None for r in runners for n in (128, 256, 512, 8192)
            )
    finally:
        backend.clear_piecewise_cuda_graphs()
        torch._dynamo.reset()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not add_rmsnorm_fp4_quant_available(),
    reason="requires CUDA and FlashInfer add RMSNorm FP4 quantization",
)
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_add_rmsnorm_fp4_quant_out_opcheck(num_tokens: int) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    direct_padding = fused_norm._ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING
    width = 3072 if direct_padding else 2880
    x = torch.randn((num_tokens, 2880), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(2880, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([100.0], device="cuda")
    fp4 = torch.empty((num_tokens, width // 2), device="cuda", dtype=torch.uint8)
    sf = torch.empty((num_tokens, width // 16), device="cuda", dtype=torch.uint8)
    y_out = torch.empty_like(x)
    torch.library.opcheck(
        _add_rmsnorm_fp4_quant_out,
        (x, residual, weight, scale, fp4, sf, y_out, 1e-5, direct_padding),
    )
