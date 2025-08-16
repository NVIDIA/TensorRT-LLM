import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax /
                                                             448.0).view(
                                                                 x_view.size(0),
                                                                 x_view.size(2))


def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, 128), align(n, 128)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2))


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def calc_woq_tolerence(x: torch.Tensor, weight_dtype: torch.dtype):
    # align with woq_assert_near_eq function in tests/unittest/trt/quantization/_utils.py
    if weight_dtype == torch.int8:
        bits_in_type = 8
    elif weight_dtype == torch.quint4x2:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))
    max_val = torch.max(abs(x)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding

    return atol


def reference_moe_torch(x: torch.Tensor, selected_experts: torch.Tensor,
                        final_scales: torch.Tensor, num_experts: int,
                        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    # cast back to the input dtype
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1_weight = weights[f"{expert_id}.w1.weight"]
        w2_weight = weights[f"{expert_id}.w2.weight"]
        w3_weight = weights[f"{expert_id}.w3.weight"]
        expert_inputs = x[batch_idx]
        output = (F.silu(expert_inputs @ w1_weight.t()) *
                  (expert_inputs @ w3_weight.t())) @ w2_weight.t()
        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)


def reference_block_scale_moe_torch(
        x: torch.Tensor, selected_experts: torch.Tensor,
        final_scales: torch.Tensor, num_experts: int,
        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1 = weights[f"{expert_id}.w1.weight"]
        w2 = weights[f"{expert_id}.w2.weight"]
        w3 = weights[f"{expert_id}.w3.weight"]

        w1_fp8, w1_scale = per_block_cast_to_fp8(w1)
        w2_fp8, w2_scale = per_block_cast_to_fp8(w2)
        w3_fp8, w3_scale = per_block_cast_to_fp8(w3)

        x_fp8, x_scale = per_token_cast_to_fp8(x[batch_idx])

        def block_scale_gemm(mat_a: torch.Tensor, mat_scale_a: torch.Tensor,
                             mat_b: torch.Tensor, mat_scale_b: torch.Tensor):
            shape_m, shape_k = mat_a.shape
            shape_n = mat_b.shape[0]
            result = torch.zeros((shape_m, shape_n), dtype=torch.float32).cuda()

            for m in range(shape_m):
                for n in range(shape_n):
                    for k in range(0, shape_k, 128):
                        scale_factor = mat_scale_a[m, k //
                                                   128] * mat_scale_b[n // 128,
                                                                      k // 128]
                        tile_a = mat_a[m, k:k + 128]
                        tile_b = mat_b[n, k:k + 128]
                        tile_d = torch.dot(tile_a.float(), tile_b.float())
                        result[
                            m,
                            n] += scale_factor.cuda() * tile_d.cuda().float()

            result_bf16 = result.bfloat16()

            return result_bf16

        # gemm1
        fc3_output = block_scale_gemm(x_fp8, x_scale, w1_fp8, w1_scale)
        gate_output = F.silu(fc3_output)
        fc1_output = block_scale_gemm(x_fp8, x_scale, w3_fp8, w3_scale)
        act_output = gate_output * fc1_output
        # gemm2
        act_fp8, act_scale = per_token_cast_to_fp8(act_output)
        output = block_scale_gemm(act_fp8, act_scale, w2_fp8, w2_scale)

        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata,
    ) -> None:
        self.batch_size = batch_size
        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size, ),
                                    device=device,
                                    dtype=torch.int32)
        self.position_ids = torch.zeros((1, batch_size),
                                        device=device,
                                        dtype=torch.int32)

        self.attn_metadata = attn_metadata
        self._output = None
        self._graph = None

    def __del__(self):
        self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
        }

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        for _ in range(2):
            forward_fn(inputs)
        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)
        set_graph_capturing(False)
        # Mark weak ref here. The output tensor should be freed properly.
        from tensorrt_llm._torch.utils import make_weak_ref
        self._output = make_weak_ref(output)
        return self._graph.pool()

    def needs_capture(self) -> bool:
        return self._output is None

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        assert "input_ids" in inputs
        assert "position_ids" in inputs
        assert "attn_metadata" in inputs

        attn_metadata = inputs["attn_metadata"]
        assert attn_metadata is self.attn_metadata, (
            "attn_metadata does not match the attn_metadata instance that was used to "
            "capture this graph.")

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        self.position_ids[:, :seqlen].copy_(position_ids)

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
