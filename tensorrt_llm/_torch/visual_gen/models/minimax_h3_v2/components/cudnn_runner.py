# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN dense self-attention plans for H3's projected NHD tensors."""

import math
import threading
from functools import lru_cache
from typing import Any

import cudnn
import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

_handle_lock = threading.Lock()


@lru_cache(maxsize=None)
def _device_handle(device: torch.device):
    with torch.cuda.device(device):
        return cudnn.create_handle()


@lru_cache(maxsize=1)
def _library_version() -> int:
    version = cudnn.backend_version()
    torch_version = torch.backends.cudnn.version()
    if version != torch_version:
        logger.critical(
            f"PyTorch cuDNN version {torch_version} differs from frontend version {version}."
        )
    return version


def _round_up(size: int, multiple: int) -> int:
    return (size + multiple - 1) // multiple * multiple


def _quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data, descale = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
    return data.transpose(1, 2), descale.float().reshape(1, 1, 1, 1)


def _quantize_mxfp8_qk(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block Q/K along the head dimension with scale tiles aligned per head."""
    b, h, s, d = x.shape
    padded_s = _round_up(s, 128)
    if s != padded_s:
        padded = x.new_zeros(b, h, padded_s, d)
        padded[:, :, :s] = x
    else:
        padded = x.contiguous()
    data, factors = torch.ops.trtllm.mxfp8_quantize(padded.reshape(-1, d), True, 32)
    return data.view(b, h, padded_s, d)[:, :, :s], factors.view(
        b, h, padded_s, _round_up(d // 32, 4)
    )


def _quantize_mxfp8_v(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block V along the sequence dimension contracted by the second matmul."""
    b, h, s, d = x.shape
    padded_s, padded_d = _round_up(s, 128), _round_up(d, 128)
    transposed = x.new_zeros(b, h, padded_d, padded_s)
    transposed[:, :, :d, :s] = x.transpose(2, 3)
    data, factors = torch.ops.trtllm.mxfp8_quantize(transposed.reshape(-1, padded_s), True, 32)
    data = data.view(b, h, padded_d, padded_s)[:, :, :d, :s].transpose(2, 3).contiguous()
    factors = factors.view(b, h, padded_d, _round_up(padded_s // 32, 4)).transpose(2, 3)
    return data, factors


class CuDNNSdpaRunner:
    """Build and execute dense cuDNN attention using model-owned plan and scale caches."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype | None,
        quant_attention_config: QuantAttentionConfig | None,
        attention_metadata_state: dict[str, Any],
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)
        self.dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        self.quant_dtype = (
            None if quant_attention_config is None else quant_attention_config.qk_dtype
        )
        if quant_attention_config is not None and (
            self.quant_dtype != quant_attention_config.v_dtype
            or self.quant_dtype not in ("fp8", "mxfp8")
        ):
            raise ValueError("H3 cuDNN requires matching FP8 or MXFP8 attention formats for Q/K/V.")
        version = _library_version()
        if version < 90100 or (self.quant_dtype == "mxfp8" and version < 92100):
            raise RuntimeError("H3 cuDNN SDPA requires cuDNN 9.1 or later, or 9.21 for MXFP8.")
        if self.quant_dtype and head_dim > 128:
            raise ValueError("H3 quantized cuDNN attention requires head_dim <= 128.")
        if self.quant_dtype == "mxfp8" and head_dim % 32:
            raise ValueError("H3 cuDNN MXFP8 requires head_dim to be a multiple of 32.")
        self._graphs = attention_metadata_state.setdefault("h3_cudnn_graphs", {})
        self._scales = attention_metadata_state.setdefault("h3_cudnn_softmax_scales", {})

    @torch.compiler.disable
    def _softmax_scales(self, device: torch.device):
        with _handle_lock:
            if device not in self._scales:
                scale = torch.full((1, 1, 1, 1), 448.0, dtype=torch.float32, device=device)
                self._scales[device] = (scale, scale.reciprocal(), torch.ones_like(scale))
            return self._scales[device]

    def _build(self, buffers: dict[str, torch.Tensor], output: torch.Tensor):
        out_dtype = (
            cudnn.data_type.HALF if self.dtype == torch.float16 else cudnn.data_type.BFLOAT16
        )
        io_dtype = cudnn.data_type.FP8_E4M3 if self.quant_dtype else out_dtype
        f32 = cudnn.data_type.FLOAT
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=f32,
            compute_data_type=f32,
            name=f"h3_sdpa_{self.quant_dtype or 'dense'}",
        )
        inputs = {}
        for name, tensor in buffers.items():
            kwargs = {}
            if name in ("q", "k", "v"):
                data_type = io_dtype
            elif self.quant_dtype == "mxfp8":
                data_type = cudnn.data_type.FP8_E8M0
                kwargs["reordering_type"] = cudnn.tensor_reordering.F8_128x4
            else:
                data_type = f32
            inputs[name] = graph.tensor(
                name=name,
                dim=list(tensor.shape),
                stride=list(tensor.stride()),
                data_type=data_type,
                **kwargs,
            )
        sdpa_options = dict(attn_scale=self.scale, use_causal_mask=False, generate_stats=False)
        if self.quant_dtype is None:
            out, _ = graph.sdpa(**inputs, **sdpa_options)
            amax = ()
        elif self.quant_dtype == "fp8":
            out, _, amax_s, amax_o = graph.sdpa_fp8(**inputs, **sdpa_options)
            amax = (amax_s, amax_o)
        else:
            out, _, amax_o = graph.sdpa_mxfp8(**inputs, **sdpa_options)
            amax = (amax_o,)
        out.set_output(True).set_dim(list(output.shape)).set_stride(
            list(output.stride())
        ).set_data_type(out_dtype)
        amax = tuple(tensor for tensor in amax if tensor is not None)
        for tensor in amax:
            tensor.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
                f32
            )
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        return graph, graph.get_workspace_size(), inputs, out, amax

    @torch.compiler.disable
    def _execute(self, buffers: dict[str, torch.Tensor], output: torch.Tensor) -> None:
        device = output.device
        key = (
            device,
            self.dtype,
            self.quant_dtype,
            self.scale,
            tuple((name, x.shape, x.stride()) for name, x in buffers.items()),
        )
        with torch.cuda.device(device), _handle_lock:
            if key not in self._graphs:
                if self.quant_dtype and get_sm_version() not in (100, 103, 107):
                    raise RuntimeError("cuDNN quantized attention requires a Blackwell-class GPU.")
                self._graphs[key] = self._build(buffers, output)
            graph, workspace_size, inputs, out, amax = self._graphs[key]
            bindings = {desc: buffers[name] for name, desc in inputs.items()}
            bindings[out] = output
            for desc in amax:
                bindings[desc] = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=device)
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
            handle = _device_handle(device)
            cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
            graph.execute(bindings, workspace, handle=handle)

    def run_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if (
            q.ndim != 4
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[2:] != (self.num_heads, self.head_dim)
        ):
            raise ValueError(
                "H3 cuDNN attention requires matching [batch, rows, configured heads, head_dim] tensors."
            )
        q, k, v = q.to(self.dtype), k.to(self.dtype), v.to(self.dtype)
        if self.quant_dtype is None:
            buffers = {
                name: x.contiguous().transpose(1, 2) for name, x in (("q", q), ("k", k), ("v", v))
            }
        else:
            if self.quant_dtype == "fp8":
                q_data, dq = _quantize_fp8(q)
                k_data, dk = _quantize_fp8(k)
                v_data, dv = _quantize_fp8(v)
            else:
                q_data, dq = _quantize_mxfp8_qk(q.transpose(1, 2))
                k_data, dk = _quantize_mxfp8_qk(k.transpose(1, 2))
                v_data, dv = _quantize_mxfp8_v(v.transpose(1, 2))
            buffers = dict(q=q_data, k=k_data, v=v_data, descale_q=dq, descale_k=dk, descale_v=dv)
            if self.quant_dtype == "fp8":
                scale_s, descale_s, scale_o = self._softmax_scales(q.device)
                buffers.update(descale_s=descale_s, scale_s=scale_s, scale_o=scale_o)
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
        self._execute(buffers, output.transpose(1, 2))
        return output
