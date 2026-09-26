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

import torch


def convert_gptq_weights(
    weights: dict[str, torch.Tensor], group_size: int, checkpoint_format: str = "gptq"
) -> dict[str, torch.Tensor]:
    """Convert a GPTQ linear layer to the grouped INT4 Linear checkpoint layout.

    GPTQ packs eight unsigned weights along K into each int32, and packs zero
    points along N. Linear shards [N / 2, K] signed INT4 weights and [N, K / group]
    scales/additive zero offsets. Conversion precedes TP slicing and KV-head
    replication; the source tensors are left unchanged.
    """
    if checkpoint_format not in ("gptq", "gptq_v2"):
        raise ValueError(f"Unsupported GPTQ checkpoint_format: {checkpoint_format}")
    if group_size not in (64, 128):
        raise ValueError("GPTQ requires group_size=64 or 128")
    missing = {"qweight", "qzeros", "scales"} - weights.keys()
    if missing:
        raise ValueError(f"Missing GPTQ tensors: {sorted(missing)}")
    # HF loaders may provide lazy safetensors slices.
    qweight, qzeros, scales = (weights[key][:] for key in ("qweight", "qzeros", "scales"))
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("GPTQ qweight and qzeros must use int32 packing")
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("GPTQ qweight, qzeros and scales must be matrices")
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    if k % group_size or n % 8:
        raise ValueError("GPTQ dimensions must align with group_size and int32 packing")
    if scales.shape != (k // group_size, n) or qzeros.shape != (k // group_size, n // 8):
        raise ValueError("GPTQ scales or qzeros shape does not match qweight")
    if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("GPTQ scales must be floating point")
    if "g_idx" in weights:
        g_idx = weights["g_idx"][:]
        expected = torch.arange(k, device=g_idx.device) // group_size
        if g_idx.shape != (k,) or not torch.equal(g_idx, expected):
            raise ValueError("GPTQ activation-order/non-contiguous g_idx is not supported")

    # Conversion unpacks the full matrix before TP slicing. Dense temporary
    # tensors coexist with the packed inputs, which can increase peak host
    # memory, especially with concurrent loaders or TP ranks.
    # TP sharding does not reduce these full-matrix conversion temporaries.
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    unsigned = ((qweight[:, None, :] >> shifts[None, :, None]) & 15).to(torch.int8)
    # Recenter unsigned [0, 15] to signed [-8, 7], then pack along N.
    signed = (unsigned.reshape(k, n) - 8) & 15
    packed = signed[:, 0::2] | (signed[:, 1::2] << 4)
    zeros = ((qzeros[:, :, None] >> shifts.to(qzeros.device)) & 15).reshape(-1, n)
    if checkpoint_format == "gptq":
        # GPTQ v1 stores (zero_point - 1) in each nibble.
        zeros = (zeros + 1) & 15
    offsets = (8 - zeros).to(scales.dtype) * scales
    result = {
        "weight": packed.T.contiguous(),
        "weight_scale": scales.T.contiguous(),
        "weight_zero": offsets.T.contiguous(),
    }
    if "bias" in weights:
        result["bias"] = weights["bias"]
    return result
