# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi K3.1 ssm/ entry tests share: pools laid out like the V2 hybrid manager's per-slot coalesced ones
(contiguous within a slot, padded slot stride), K3.1's KDA constants and the comparison bands.

Not a test file: the entry tests import it by name from their own directory.
"""

from __future__ import annotations

import math
import zlib

import pytest
import torch

HEAD_DIM = 128  # K == V
CONV_WIDTH = 4
COMMITTED = CONV_WIDTH - 1
LOWER_BOUND = -5.0
SCALE = HEAD_DIM**-0.5
EPS = 1e-5

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)


def cell_seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def generator(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(cell_seed(cell))


def padded_slots(
    num_slots: int, row_shape: tuple[int, ...], dtype: torch.dtype, pad_bytes: int = 48
) -> torch.Tensor:
    """`[num_slots, *row_shape]`, contiguous within each slot, with `pad_bytes` of padding between slots
    (a multiple of 16 bytes, as the V2 manager's per-slot coalesced pools keep 16-byte alignment)."""
    row = math.prod(row_shape)
    pad = pad_bytes // torch.tensor([], dtype=dtype).element_size()
    storage = torch.zeros(num_slots, row + pad, dtype=dtype, device="cuda")
    return storage[:, :row].view(num_slots, *row_shape)


def transposed_padded_slots(
    num_slots: int, rows: int, cols: int, dtype: torch.dtype, pad_bytes: int = 48
) -> torch.Tensor:
    """`[num_slots, cols, rows]` view of `[num_slots, rows, cols]` padded-slot storage: the dim-contiguous
    replay conv caches (`stride(1) == 1`)."""
    return padded_slots(num_slots, (rows, cols), dtype, pad_bytes).transpose(1, 2)


def replay_caches(num_slots: int, heads: int, num_spec: int, gen: torch.Generator) -> dict:
    """One layer's fused-verify replay caches with the target's dtypes and layouts, random finite contents:
    raw conv inputs in `cs_*` (bf16 [slots, D, 3 + num_spec]), normalized keys and SiLU values in k / v
    (bf16), log-decays in [LOWER_BOUND, 0) in g (fp32) and sigmoid gains in beta (bf16 rows padded to 16
    bytes)."""
    dim = heads * HEAD_DIM
    caches = {
        name: transposed_padded_slots(num_slots, COMMITTED + num_spec, dim, torch.bfloat16)
        for name in ("cs_q", "cs_k", "cs_v")
    }
    for name in ("cs_q", "cs_k", "cs_v"):
        caches[name].copy_(torch.randn(caches[name].shape, generator=gen, device="cuda") * 0.5)
    keys = torch.randn(num_slots, num_spec, heads, HEAD_DIM, generator=gen, device="cuda")
    keys = torch.nn.functional.normalize(keys, dim=-1).reshape(num_slots, num_spec, dim)
    caches["k_cache"] = padded_slots(num_slots, (num_spec, dim), torch.bfloat16)
    caches["k_cache"].copy_(keys)
    caches["v_cache"] = padded_slots(num_slots, (num_spec, dim), torch.bfloat16)
    caches["v_cache"].copy_(torch.randn(num_slots, num_spec, dim, generator=gen, device="cuda"))
    caches["g_cache"] = padded_slots(num_slots, (num_spec, dim), torch.float32)
    caches["g_cache"].copy_(
        LOWER_BOUND * torch.rand(num_slots, num_spec, dim, generator=gen, device="cuda")
    )
    padded_heads = (heads + 7) // 8 * 8
    beta = padded_slots(num_slots, (num_spec, padded_heads), torch.bfloat16)[..., :heads]
    beta.copy_(torch.rand(num_slots, num_spec, heads, generator=gen, device="cuda"))
    caches["beta_cache"] = beta
    return caches


def clone_strided(t: torch.Tensor) -> torch.Tensor:
    """A copy with the same strides, so the same layout constraints hold for the reference run."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def assert_scaled_close(got: torch.Tensor, ref: torch.Tensor, atol: float, what: str) -> None:
    """Within `atol` of `ref`'s largest magnitude (kernel accumulation order and approximate exp / rcp)."""
    scale = ref.float().abs().max().clamp_min(1e-6)
    torch.testing.assert_close(
        got.float() / scale, ref.float() / scale, rtol=0.0, atol=atol, msg=lambda m: f"{what}: {m}"
    )


def bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def bare(cls, **attributes):
    """An instance of ``cls`` without its constructor, carrying the given attributes (a cache
    manager without V2 pools, say)."""
    obj = object.__new__(cls)
    for name, value in attributes.items():
        setattr(obj, name, value)
    return obj
