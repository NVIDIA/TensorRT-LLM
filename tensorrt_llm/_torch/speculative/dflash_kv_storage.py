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
"""Storage dtype of a DFlash drafter's KV layers in the target's KV cache manager.

By default the drafter's KV layers keep the drafter's weight dtype (BF16), even
when the target's KV is FP8. ``TRTLLM_DFLASH_FP8_DRAFT_KV=1`` (default off) stores
them as FP8 E4M3 instead; it requires the draft layers to share the target's
manager and the target's KV to be FP8. Both servers of a disaggregated pair must
set it: the transfer views of the two sides then agree on their bytes per KV
head.

Only the KV cache creator reads the knob; it sets the draft layers' dtype.
Everything downstream keys on the dtype of the pool it is handed:

* the drafter K/V are still computed in the activation dtype (BF16)
  (:func:`kv_compute_dtype`) and stored with one saturating cast
  (:func:`to_kv_storage`): round to nearest even, values past the E4M3 range
  clamped to +-448, NaN kept, at a per-tensor scale of 1.0 (the convention of
  the target's FP8 KV);
* readers widen E4M3 to the activation dtype on load. Every E4M3 value is
  exact in BF16, so the attention sees exactly the stored values.

This module imports nothing from the package, so any of them can import it.
"""

import os
from typing import Mapping, Optional

import torch

FP8_DRAFT_KV_ENV = "TRTLLM_DFLASH_FP8_DRAFT_KV"

#: Largest finite magnitude of ``torch.float8_e4m3fn``.
E4M3_MAX = 448.0

#: KV storage dtypes the drafter never computes in: its K/V are computed in the
#: activation dtype and cast on store.
STORAGE_ONLY_DTYPES = (torch.float8_e4m3fn,)


def fp8_draft_kv_requested(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``TRTLLM_DFLASH_FP8_DRAFT_KV`` is ``"1"`` after stripping whitespace.

    ``environ`` defaults to ``os.environ``; it is read on every call.
    """
    env = os.environ if environ is None else environ
    return env.get(FP8_DRAFT_KV_ENV, "0").strip() == "1"


def is_storage_only_dtype(dtype: torch.dtype) -> bool:
    """Whether a KV pool of ``dtype`` stores values computed in another dtype."""
    return dtype in STORAGE_ONLY_DTYPES


def kv_compute_dtype(cache_dtype: torch.dtype, activation_dtype: torch.dtype) -> torch.dtype:
    """The dtype the drafter K/V are computed in before they are stored in ``cache_dtype``.

    The cache dtype itself, except for a storage-only cache dtype (FP8), whose
    K/V are computed in ``activation_dtype``.
    """
    if is_storage_only_dtype(cache_dtype):
        return activation_dtype
    return cache_dtype


def to_kv_storage(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x`` as a KV pool of ``dtype`` stores it.

    Returns ``x`` itself when it already has ``dtype``. For
    ``torch.float8_e4m3fn`` it clamps to +-``E4M3_MAX`` (infinities included;
    NaN stays NaN), then rounds to nearest even, so the saturation does not
    depend on how a conversion treats out-of-range values.

    Raises:
        ValueError: for any other pair of dtypes.
    """
    if x.dtype == dtype:
        return x
    if dtype == torch.float8_e4m3fn:
        return x.clamp(-E4M3_MAX, E4M3_MAX).to(dtype)
    raise ValueError(f"a DFlash draft KV pool of {dtype} does not store {x.dtype} K/V")
