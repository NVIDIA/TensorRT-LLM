# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense FlashInfer prefill over VisualGen NHD tensors."""

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

from . import _flashinfer_runtime
from ._flashinfer_runtime import FlashInferPrefillRunner


def _validate_inputs(
    runner: FlashInferPrefillRunner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("FlashInfer attention expects 4D NHD Q/K/V tensors.")
    if q.device.type != "cuda":
        raise RuntimeError("FlashInfer attention requires CUDA tensors.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("FlashInfer attention requires Q/K/V on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("FlashInfer attention requires Q/K/V with the same dtype.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashInfer attention expects FP16 or BF16 Q/K/V tensors.")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("FlashInfer attention requires aligned Q/K/V batch dimensions.")
    if q.shape[2:] != (runner.num_heads, runner.head_dim):
        raise ValueError(
            "Invalid query shape for FlashInfer attention: "
            f"expected [..., {runner.num_heads}, {runner.head_dim}], got {tuple(q.shape)}."
        )
    expected_kv = (runner.num_kv_heads, runner.head_dim)
    if k.shape[2:] != expected_kv or v.shape[2:] != expected_kv:
        raise ValueError(
            "Invalid key/value shape for FlashInfer attention: "
            f"expected [..., {expected_kv[0]}, {expected_kv[1]}], "
            f"got K={tuple(k.shape)}, V={tuple(v.shape)}."
        )
    if runner.num_heads % runner.num_kv_heads != 0:
        raise ValueError("FlashInfer attention requires num_heads divisible by num_kv_heads.")
    if key_padding_mask is not None and key_padding_mask.shape != (
        q.shape[0],
        k.shape[1],
    ):
        raise ValueError(
            "key_padding_mask must have shape "
            f"{(q.shape[0], k.shape[1])}, got {tuple(key_padding_mask.shape)}."
        )


class FlashInferPrefill(OpWrapper):
    """FP16/BF16 prefill over [batch, rows, heads, dim] queries, keys and values.

    qk_dtype and v_dtype select dynamic quantization; None uses the input precision.
    Padding masks use True for visible keys. Returns output in the query layout
    and FP32 natural-log LSE in [batch, query heads, query rows].
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single BF16 self-attention", dict(dtype=torch.bfloat16, batch=1, q=47, kv=47)),
        Cell(
            "batched padded self-attention",
            dict(dtype=torch.bfloat16, batch=2, q=47, kv=47, padding=True),
        ),
        Cell(
            "cross-attention with independent padding masks",
            dict(dtype=torch.bfloat16, batch=2, q=47, kv=53, padding=True),
        ),
        Cell(
            "causal grouped-query attention",
            dict(dtype=torch.bfloat16, batch=2, q=47, kv=47, kv_heads=2, causal=True),
        ),
        Cell("FP16 input and output", dict(dtype=torch.float16, batch=2, q=47, kv=47)),
        Cell(
            "dynamic global scale with native NVFP4 blocks",
            dict(
                dtype=torch.bfloat16,
                batch=1,
                q=129,
                kv=129,
                heads=2,
                dim=128,
                recipe="nvfp4",
            ),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        runner: FlashInferPrefillRunner,
        qk_dtype: str | None = None,
        v_dtype: str | None = None,
        is_causal: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_inputs(runner, q, k, v, key_padding_mask)
        if is_causal and q.shape[1] != k.shape[1]:
            raise ValueError("FlashInfer causal attention requires equal Q and K/V lengths.")
        if is_causal and key_padding_mask is not None:
            raise ValueError("FlashInfer attention does not combine causal and padding masks.")

        if (qk_dtype is None) != (v_dtype is None):
            raise ValueError("FlashInfer quantization requires both qk_dtype and v_dtype.")
        if qk_dtype is None:
            # FlashInfer's single-prefill FA2/FA3 heuristics regress on SM10x.
            if q.shape[0] == 1 and torch.cuda.get_device_capability(q.device)[0] != 10:
                return _flashinfer_runtime.run_single_prefill(
                    runner,
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    key_padding_mask=key_padding_mask,
                )
            return _flashinfer_runtime.run_batch_prefill(
                runner,
                q,
                k,
                v,
                is_causal=is_causal,
                key_padding_mask=key_padding_mask,
            )
        if key_padding_mask is not None:
            raise ValueError("FlashInfer quantized attention does not support key padding masks.")

        capability = torch.cuda.get_device_capability(q.device)
        if capability in ((10, 0), (10, 3)) and qk_dtype in ("mxfp8", "nvfp4") and v_dtype == "fp8":
            return _flashinfer_runtime.run_blockscaled_sm10x(
                runner,
                q,
                k,
                v,
                qk_mode=qk_dtype,
                is_causal=is_causal,
            )
        if capability[0] == 12 and qk_dtype == "nvfp4" and v_dtype == "nvfp4":
            return _flashinfer_runtime.run_nvfp4_sm12x(runner, q, k, v, is_causal=is_causal)
        raise RuntimeError(
            "Unsupported FlashInfer quantized attention recipe for "
            f"SM{capability[0]}{capability[1]}: qk_dtype={qk_dtype!r}, "
            f"v_dtype={v_dtype!r}. Use qk_dtype='mxfp8' or 'nvfp4' with "
            "v_dtype='fp8' on SM100/SM103, or qk_dtype='nvfp4' with "
            "v_dtype='nvfp4' on SM120/SM121."
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        runner: FlashInferPrefillRunner,
        qk_dtype: str | None = None,
        v_dtype: str | None = None,
        is_causal: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ideal attention in FP32 before rounding the output to the query dtype."""
        query = q.transpose(1, 2).float()
        repeats = q.size(2) // k.size(2)
        key = k.transpose(1, 2).float().repeat_interleave(repeats, dim=1)
        value = v.transpose(1, 2).float().repeat_interleave(repeats, dim=1)
        with true_fp32_matmul():
            scores = (query @ key.transpose(-2, -1)) * runner.scale
            if is_causal:
                future = torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool).triu(1)
                scores.masked_fill_(future, -torch.inf)
            if key_padding_mask is not None:
                scores.masked_fill_(
                    ~key_padding_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :],
                    -torch.inf,
                )
            output = scores.softmax(dim=-1).nan_to_num(nan=0.0) @ value
        return output.transpose(1, 2).to(q.dtype), scores.logsumexp(dim=-1)


flashinfer_prefill = FlashInferPrefill()
