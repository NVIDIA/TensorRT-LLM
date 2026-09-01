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
"""
cuDNN SDPA backend for visual generation models.

Three recipes, selected by ``quant_attention_config`` (see
``tensorrt_llm.visual_gen.args.AttentionConfig``):

============  =========================================  ==================
Recipe        ``quant_attention_config``                 cuDNN node
============  =========================================  ==================
``no_quant``  ``None``                                   ``sdpa``
``fp8``       ``qk_dtype='fp8'``, ``v_dtype='fp8'``      ``sdpa_fp8``
``mxfp8``     ``qk_dtype='mxfp8'``, ``v_dtype='mxfp8'``  ``sdpa_mxfp8``
============  =========================================  ==================

Layout: HND ``[B, H, S, D]``.
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple

import cudnn
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_up(x: int, multiple: int) -> int:
    return _ceil_div(x, multiple) * multiple


def _row_major_stride(*dims: int) -> list:
    """Row-major (contiguous) strides for ``dims``."""
    acc = 1
    strides = []
    for dim in reversed(dims):
        strides.append(acc)
        acc *= dim
    return list(reversed(strides))


def _torch_to_cudnn_dtype(dtype: torch.dtype) -> Any:
    mapping = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    }
    if dtype not in mapping:
        raise ValueError(f"No cudnn.data_type mapping for torch dtype {dtype}.")
    return mapping[dtype]


# ============================================================================
# Quantization helpers
# ============================================================================


def _quantize_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 e4m3 with a single scale.

    The fused amax+quantize op computes the scale and the FP8 data in one pass and
    returns the descale on device, so the recipe stays free of host synchronization.

    Returns:
        x_q: FP8 tensor with the same shape as ``x``.
        descale: ``[1, 1, 1, 1]`` float32 device tensor with ``x ~= x_q * descale``.
    """
    # The op requires contiguous input.
    x_q, descale = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
    return x_q, descale.float().reshape(1, 1, 1, 1)


def _quantize_mxfp8_qk(x_bhsd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a Q or K tensor to MXFP8, blocking along the head dimension.

    cuDNN wants ``descale_q``/``descale_k`` as ``[B, H, S_padded, D_scale]`` with
    ``S_padded`` a multiple of 128, ``D_scale = ceil(D / 32)`` padded to a multiple
    of 4, ``stride[3] == 1`` and ``F8_128x4`` reordering, which is what
    ``torch.ops.trtllm.mxfp8_quantize(..., is_sf_swizzled_layout=True)`` emits for a
    ``[B * H * S_padded, D]`` matrix. S is padded to a multiple of 128 so that the
    128-row scale-factor tiles align with ``(b, h)`` boundaries and the flat
    scale-factor buffer can be viewed as ``[B, H, S_padded, D_scale]``.

    The quantized data is returned as a view into the S-padded buffer; cuDNN accepts
    a strided Q/K as long as the head-dim stride is 1.

    Returns:
        x_q: ``[B, H, S, D]`` float8_e4m3fn view into a ``[B, H, S_padded, D]`` buffer.
        x_sf: ``[B, H, S_padded, D_scale]`` uint8 E8M0 scale factors.
    """
    if x_bhsd.dim() != 4:
        raise ValueError(f"_quantize_mxfp8_qk expects [B, H, S, D]; got {tuple(x_bhsd.shape)}.")
    b, h, s, d = x_bhsd.shape
    if d % 32 != 0:
        raise ValueError(f"head_dim={d} must be a multiple of the MXFP8 block size 32.")

    s_pad = _pad_up(s, 128)
    if s_pad != s:
        x_padded = x_bhsd.new_zeros(b, h, s_pad, d)
        x_padded[:, :, :s, :] = x_bhsd
    else:
        x_padded = x_bhsd.contiguous()

    x_q_2d, x_sf_1d = torch.ops.trtllm.mxfp8_quantize(x_padded.reshape(b * h * s_pad, d), True, 32)
    d_scale = _pad_up(d // 32, 4)
    x_q = x_q_2d.view(b, h, s_pad, d)[:, :, :s, :]
    return x_q, x_sf_1d.view(b, h, s_pad, d_scale)


def _quantize_mxfp8_v(x_bhsd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a V tensor to MXFP8, blocking along the *sequence* dimension.

    The second attention GEMM contracts over S, so V's scale factors block along S
    rather than D. cuDNN wants ``descale_v`` as ``[B, H, S_scale, D_padded]`` with
    ``S_scale = ceil(S_kv / 32)`` padded to a multiple of 4, ``stride[2] == 1`` and
    ``F8_128x4`` reordering. Quantizing the transposed tensor
    ``[B, H, D_padded, S_padded]`` produces that buffer; the returned view carries
    the transposed (S-scale contiguous) strides cuDNN asks for.

    Returns:
        x_q: ``[B, H, S, D]`` float8_e4m3fn (unpadded).
        x_sf: ``[B, H, S_scale, D_padded]`` uint8 E8M0 scale factors, ``stride[2] == 1``.
    """
    if x_bhsd.dim() != 4:
        raise ValueError(f"_quantize_mxfp8_v expects [B, H, S, D]; got {tuple(x_bhsd.shape)}.")
    b, h, s, d = x_bhsd.shape

    s_pad = _pad_up(s, 128)
    d_pad = _pad_up(d, 128)
    # [B, H, D_padded, S_padded]. Zero padding does not affect the per-block amax.
    x_t = x_bhsd.new_zeros(b, h, d_pad, s_pad)
    x_t[:, :, :d, :s] = x_bhsd.transpose(2, 3)

    x_q_2d, x_sf_1d = torch.ops.trtllm.mxfp8_quantize(x_t.reshape(b * h * d_pad, s_pad), True, 32)
    s_scale = _pad_up(s_pad // 32, 4)
    x_q = x_q_2d.view(b, h, d_pad, s_pad)[:, :, :d, :s].permute(0, 1, 3, 2).contiguous()
    # [B, H, D_padded, S_scale] -> [B, H, S_scale, D_padded] (stride[2] == 1).
    x_sf = x_sf_1d.view(b, h, d_pad, s_scale).permute(0, 1, 3, 2)
    return x_q, x_sf


# ============================================================================
# Graph geometry
# ============================================================================


@dataclass
class _CuDNNGraphBundle:
    """A built cuDNN graph plus the tensor handles needed to bind buffers."""

    graph: Any
    workspace_size: int
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _CuDNNProblemShape:
    """Problem geometry shared by all three recipes.

    The Q/K/V strides are part of the geometry, and of the graph cache key; the
    MXFP8 recipe passes Q/K as views into sequence-padded buffers.
    """

    b: int
    h_q: int
    h_kv: int
    s_q: int
    s_kv: int
    d_qk: int
    d_v: int
    q_strides: Tuple[int, ...]
    k_strides: Tuple[int, ...]
    v_strides: Tuple[int, ...]


# ============================================================================
# VisualGen AttentionBackend class
# ============================================================================


class CuDNNAttention(AttentionBackend):
    """cuDNN SDPA backend for visual generation.

    Runs unquantized (bf16/fp16), per-tensor FP8, or block-scaled MXFP8 attention
    through cuDNN's fused ``sdpa`` / ``sdpa_fp8`` / ``sdpa_mxfp8`` nodes. The recipe
    comes from ``quant_attention_config``; ``None`` means unquantized.

    The compiled-graph cache is process-wide and shared by every instance.
    cuDNN handles and cached graphs are isolated by CUDA device.
    """

    _cudnn_lib_version = None
    _cudnn_handles: ClassVar[Dict[int, Any]] = {}
    _graph_cache: ClassVar[Dict[Tuple, _CuDNNGraphBundle]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype or torch.bfloat16
        self.quant_attention_config = quant_attention_config
        self.recipe = self.resolve_recipe(quant_attention_config)
        self.check_library_feature(self.recipe)
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.HND

    @staticmethod
    def resolve_recipe(quant_attention_config: Optional[QuantAttentionConfig]) -> str:
        """Map the validated public recipe onto a cuDNN SDPA node."""
        if quant_attention_config is None:
            return "no_quant"
        qk_dtype, v_dtype = quant_attention_config.qk_dtype, quant_attention_config.v_dtype
        if qk_dtype != v_dtype:
            raise ValueError(
                f"cuDNN backend requires qk_dtype == v_dtype; got qk_dtype={qk_dtype!r}, "
                f"v_dtype={v_dtype!r}. cuDNN's FP8 and MXFP8 SDPA nodes quantize both GEMMs "
                "with the same element format."
            )
        if qk_dtype in ("fp8", "mxfp8"):
            return qk_dtype
        raise ValueError(
            f"cuDNN backend does not support qk_dtype={qk_dtype!r}; supported recipes are "
            "unquantized (quant_attention_config=None), 'fp8' and 'mxfp8'."
        )

    # ------------------------------------------------------------------
    # cuDNN handle and compiled-graph cache
    # ------------------------------------------------------------------

    @classmethod
    def _get_lib_version(cls):
        if cls._cudnn_lib_version is None:
            cls._cudnn_lib_version = cudnn.backend_version()
            torch_cudnn_lib_version = torch.backends.cudnn.version()
            if cls._cudnn_lib_version != torch_cudnn_lib_version:
                logger.critical(
                    "PyTorch and cuDNN Frontend loaded different cuDNN backends: "
                    f"PyTorch:v{torch_cudnn_lib_version} != cuDNN-FE:v{cls._cudnn_lib_version}. "
                )
        return cls._cudnn_lib_version

    @classmethod
    def _get_handle(cls, device: torch.device) -> Any:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        with cls._cache_lock:
            handle = cls._cudnn_handles.get(device_index)
            if handle is None:
                with torch.cuda.device(device_index):
                    handle = cudnn.create_handle()
                cls._cudnn_handles[device_index] = handle
            return handle

    @classmethod
    def check_hardware_compatibility(cls, device: torch.device, recipe: str = "no_quant") -> None:
        compute_capability = torch.cuda.get_device_capability(device)
        gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
        if gpu_arch not in ("sm_100a", "sm_103a") and recipe != "no_quant":
            raise ImportError("cuDNN quantized attention requires NVIDIA Blackwell-class GPU.")

    @classmethod
    def check_library_feature(cls, recipe: str = "no_quant") -> None:
        # Check if the cuDNN library supports the requested functionality.
        if cls._get_lib_version() < 90100:
            raise ImportError("cuDNN attention backend requires cuDNN library v9.1.0 or later.")
        if cls._get_lib_version() < 92100 and recipe == "mxfp8":
            raise ImportError("cuDNN MXFP8 attention requires cuDNN library v9.21.0 or later.")

    @classmethod
    def clear_graph_cache(cls) -> None:
        """Drop every compiled cuDNN graph (used by tests)."""
        with cls._cache_lock:
            cls._graph_cache.clear()

    @staticmethod
    def _build_graph(
        recipe: str,
        shape: _CuDNNProblemShape,
        is_causal: bool,
        sm_scale: float,
        out_dtype: torch.dtype,
        with_lse: bool,
    ) -> _CuDNNGraphBundle:
        s = shape
        out_cudnn_dtype = _torch_to_cudnn_dtype(out_dtype)
        fp8 = cudnn.data_type.FP8_E4M3
        e8m0 = cudnn.data_type.FP8_E8M0
        f32 = cudnn.data_type.FLOAT

        io_dtype = fp8 if recipe in ("fp8", "mxfp8") else out_cudnn_dtype
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=f32,
            compute_data_type=f32,
            name=f"visual_gen_sdpa_{recipe}",
        )

        def _tensor(name: str, dims: Tuple[int, ...], dtype: Any, **kwargs) -> Any:
            return graph.tensor(
                name=name,
                dim=list(dims),
                stride=_row_major_stride(*dims),
                data_type=dtype,
                **kwargs,
            )

        q_t = graph.tensor(
            name="q", dim=[s.b, s.h_q, s.s_q, s.d_qk], stride=list(s.q_strides), data_type=io_dtype
        )
        k_t = graph.tensor(
            name="k",
            dim=[s.b, s.h_kv, s.s_kv, s.d_qk],
            stride=list(s.k_strides),
            data_type=io_dtype,
        )
        v_t = graph.tensor(
            name="v", dim=[s.b, s.h_kv, s.s_kv, s.d_v], stride=list(s.v_strides), data_type=io_dtype
        )
        inputs: Dict[str, Any] = {"q": q_t, "k": k_t, "v": v_t}
        amax_s_t = None

        if recipe == "no_quant":
            o_t, stats_t = graph.sdpa(
                q=q_t,
                k=k_t,
                v=v_t,
                attn_scale=sm_scale,
                use_causal_mask=is_causal,
                generate_stats=with_lse,
            )
            amax_o_t = None
        elif recipe == "fp8":
            # Per-tensor descales for Q/K/V plus the FP8-quantized softmax output S.
            for name in ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"):
                inputs[name] = _tensor(name, (1, 1, 1, 1), f32)
            o_t, stats_t, amax_s_t, amax_o_t = graph.sdpa_fp8(
                q=q_t,
                k=k_t,
                v=v_t,
                descale_q=inputs["descale_q"],
                descale_k=inputs["descale_k"],
                descale_v=inputs["descale_v"],
                descale_s=inputs["descale_s"],
                scale_s=inputs["scale_s"],
                scale_o=inputs["scale_o"],
                attn_scale=sm_scale,
                use_causal_mask=is_causal,
                generate_stats=with_lse,
            )
        elif recipe == "mxfp8":
            s_q_pad = _pad_up(s.s_q, 128)
            s_kv_pad = _pad_up(s.s_kv, 128)
            qk_d_scale = _pad_up(_ceil_div(s.d_qk, 32), 4)
            v_s_scale = _pad_up(s_kv_pad // 32, 4)
            v_d_pad = _pad_up(s.d_v, 128)
            reorder = {"reordering_type": cudnn.tensor_reordering.F8_128x4}

            inputs["descale_q"] = _tensor(
                "descale_q", (s.b, s.h_q, s_q_pad, qk_d_scale), e8m0, **reorder
            )
            inputs["descale_k"] = _tensor(
                "descale_k", (s.b, s.h_kv, s_kv_pad, qk_d_scale), e8m0, **reorder
            )
            # descale_v blocks along S, so its S-scale dimension is the contiguous one.
            inputs["descale_v"] = graph.tensor(
                name="descale_v",
                dim=[s.b, s.h_kv, v_s_scale, v_d_pad],
                stride=[s.h_kv * v_s_scale * v_d_pad, v_s_scale * v_d_pad, 1, v_s_scale],
                data_type=e8m0,
                **reorder,
            )
            o_t, stats_t, amax_o_t = graph.sdpa_mxfp8(
                q=q_t,
                k=k_t,
                v=v_t,
                descale_q=inputs["descale_q"],
                descale_k=inputs["descale_k"],
                descale_v=inputs["descale_v"],
                attn_scale=sm_scale,
                use_causal_mask=is_causal,
                generate_stats=with_lse,
            )
        else:
            raise ValueError(f"Unknown cuDNN SDPA recipe {recipe!r}.")

        out_dims = (s.b, s.h_q, s.s_q, s.d_v)
        o_t.set_output(True).set_dim(list(out_dims)).set_stride(
            _row_major_stride(*out_dims)
        ).set_data_type(out_cudnn_dtype)
        outputs: Dict[str, Any] = {"o": o_t}

        if with_lse:
            stats_dims = (s.b, s.h_q, s.s_q, 1)
            stats_t.set_output(True).set_dim(list(stats_dims)).set_stride(
                _row_major_stride(*stats_dims)
            ).set_data_type(f32)
            outputs["stats"] = stats_t
        # The quantized nodes emit amax(S) / amax(O) for calibration; inference binds
        # scratch buffers for them.
        for name, tensor in (("amax_s", amax_s_t), ("amax_o", amax_o_t)):
            if tensor is not None:
                tensor.set_output(True).set_dim([1, 1, 1, 1]).set_stride(
                    [1, 1, 1, 1]
                ).set_data_type(f32)
                outputs[name] = tensor

        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        return _CuDNNGraphBundle(
            graph=graph, workspace_size=graph.get_workspace_size(), inputs=inputs, outputs=outputs
        )

    @classmethod
    @torch.compiler.disable
    def _get_or_build_graph(
        cls,
        recipe: str,
        shape: _CuDNNProblemShape,
        is_causal: bool,
        sm_scale: float,
        out_dtype: torch.dtype,
        with_lse: bool,
        device: torch.device,
    ) -> _CuDNNGraphBundle:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        key = (device_index, recipe, shape, is_causal, sm_scale, out_dtype, with_lse)
        with cls._cache_lock:
            bundle = cls._graph_cache.get(key)
            if bundle is None:
                cls.check_hardware_compatibility(device, recipe)
                logger.debug(
                    f"[CuDNNAttention] building graph on cuda:{device_index}: "
                    f"recipe={recipe} {shape} causal={is_causal}"
                )
                with torch.cuda.device(device_index):
                    bundle = cls._build_graph(
                        recipe, shape, is_causal, sm_scale, out_dtype, with_lse
                    )
                cls._graph_cache[key] = bundle
            return bundle

    @classmethod
    @torch.compiler.disable
    def _execute_graph(
        cls,
        bundle: _CuDNNGraphBundle,
        tensor_map: Dict[Any, torch.Tensor],
        device: torch.device,
    ) -> None:
        with torch.cuda.device(device):
            handle = cls._get_handle(device)
            cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
            workspace = torch.empty(bundle.workspace_size, dtype=torch.uint8, device=device)
            bundle.graph.execute(tensor_map, workspace, handle=handle)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _validate_inputs(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.dim() != 4:
                raise ValueError(
                    f"cuDNN backend expects a 4D [B, H, S, D] {name}; got {tuple(tensor.shape)}."
                )
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}.")
        if q.shape[0] != k.shape[0]:
            raise ValueError(f"Batch size mismatch: q={q.shape[0]} vs k={k.shape[0]}.")
        if q.shape[3] != k.shape[3]:
            raise ValueError(f"Q/K head_dim mismatch: {q.shape[3]} vs {k.shape[3]}.")
        if q.shape[3] != self.head_dim:
            raise ValueError(
                f"cuDNN backend was configured with head_dim={self.head_dim}, "
                f"but received head_dim={q.shape[3]}."
            )
        if q.shape[1] != self.num_heads:
            raise ValueError(
                f"cuDNN backend was configured with num_heads={self.num_heads}, "
                f"but received num_heads={q.shape[1]}."
            )
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                f"num_heads={q.shape[1]} must be a multiple of num_kv_heads={k.shape[1]} for GQA."
            )
        # cuDNN's FP8 and MXFP8 SDPA engines support head_dim <= 128.
        if self.recipe != "no_quant" and max(q.shape[3], v.shape[3]) > 128:
            raise ValueError(
                f"cuDNN quantized SDPA supports head_dim <= 128; got qk={q.shape[3]}, "
                f"v={v.shape[3]}. Drop quant_attention_config to run unquantized."
            )
        if self.recipe == "mxfp8" and q.shape[3] % 32 != 0:
            raise ValueError(
                f"cuDNN MXFP8 requires head_dim to be a multiple of 32; got {q.shape[3]}."
            )

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        with_lse: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._validate_inputs(q, k, v)
        b, h_q, s_q, d_qk = q.shape
        _, h_kv, s_kv, d_v = v.shape
        device = q.device
        out_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q, k, v = q.to(out_dtype), k.to(out_dtype), v.to(out_dtype)

        buffers: Dict[str, torch.Tensor] = {}
        if self.recipe == "no_quant":
            buffers.update(q=q.contiguous(), k=k.contiguous(), v=v.contiguous())
        elif self.recipe == "fp8":
            q_q, descale_q = _quantize_fp8(q)
            k_q, descale_k = _quantize_fp8(k)
            v_q, descale_v = _quantize_fp8(v)
            # Scale the softmax output, which lies in [0, 1], to the FP8 range for the
            # second GEMM.
            scale_s = torch.full((1, 1, 1, 1), 448.0, dtype=torch.float32, device=device)
            buffers.update(
                q=q_q,
                k=k_q,
                v=v_q,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_s=scale_s.reciprocal(),
                scale_s=scale_s,
                scale_o=torch.ones(1, 1, 1, 1, dtype=torch.float32, device=device),
            )
        else:  # mxfp8
            q_q, descale_q = _quantize_mxfp8_qk(q)
            k_q, descale_k = _quantize_mxfp8_qk(k)
            v_q, descale_v = _quantize_mxfp8_v(v)
            buffers.update(
                q=q_q, k=k_q, v=v_q, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v
            )

        shape = _CuDNNProblemShape(
            b=b,
            h_q=h_q,
            h_kv=h_kv,
            s_q=s_q,
            s_kv=s_kv,
            d_qk=d_qk,
            d_v=d_v,
            q_strides=tuple(buffers["q"].stride()),
            k_strides=tuple(buffers["k"].stride()),
            v_strides=tuple(buffers["v"].stride()),
        )
        bundle = self._get_or_build_graph(
            self.recipe,
            shape,
            is_causal=is_causal,
            sm_scale=self.scale,
            out_dtype=out_dtype,
            with_lse=with_lse,
            device=device,
        )

        output = torch.empty(b, h_q, s_q, d_v, dtype=out_dtype, device=device)
        tensor_map = {bundle.inputs[name]: tensor for name, tensor in buffers.items()}
        tensor_map[bundle.outputs["o"]] = output

        stats: Optional[torch.Tensor] = None
        if with_lse:
            stats = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device=device)
            tensor_map[bundle.outputs["stats"]] = stats
        for amax_name in ("amax_s", "amax_o"):
            if amax_name in bundle.outputs:
                tensor_map[bundle.outputs[amax_name]] = torch.empty(
                    1, 1, 1, 1, dtype=torch.float32, device=device
                )

        self._execute_graph(bundle, tensor_map, device)

        # cuDNN returns stats as [B, H, S, 1]; other backends expose LSE as [B, S, H].
        lse = None if stats is None else stats.squeeze(-1).transpose(1, 2).contiguous()
        return output, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention.

        Args:
            q: Query tensor ``[B, H, S_q, D]``.
            k: Key tensor ``[B, H_kv, S_kv, D]``.
            v: Value tensor ``[B, H_kv, S_kv, D_v]``.
            attention_mask: ``CAUSAL`` or ``FULL``.
            key_padding_mask: Not supported by this backend.

        Returns:
            Output tensor ``[B, H, S_q, D_v]``.
        """
        output, _ = self._run(
            q, k, v, is_causal=self._resolve_mask(attention_mask, key_padding_mask), with_lse=False
        )
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as :meth:`forward`, additionally returning the softmax log-sum-exp.

        Returns:
            output: ``[B, H, S_q, D_v]``
            lse: ``[B, S_q, H]`` float32
        """
        output, lse = self._run(
            q, k, v, is_causal=self._resolve_mask(attention_mask, key_padding_mask), with_lse=True
        )
        assert lse is not None, "cuDNN graph was built with stats but returned none."
        return output, lse

    @staticmethod
    def _resolve_mask(
        attention_mask: PredefinedAttentionMask, key_padding_mask: Optional[torch.Tensor]
    ) -> bool:
        if key_padding_mask is not None:
            raise NotImplementedError(
                "cuDNN backend does not support key_padding_mask; use the VANILLA backend."
            )
        return attention_mask == PredefinedAttentionMask.CAUSAL

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout
