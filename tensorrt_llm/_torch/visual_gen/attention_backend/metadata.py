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
"""Attention metadata helpers for VisualGen.

VisualGen reuses the shared-core ``AttentionMetadata`` rather than defining its
own: it already models no-KV-cache operation, mixed Q/KV lengths, and several
metadata objects held side by side.

An *attention site* is one place a model attends, and gets its own metadata
object -- e.g. WAN has ``self``, ``cross_text`` and ``cross_image``.
"""

from typing import Sequence, Type

import torch

from tensorrt_llm.mapping import Mapping

from ...attention_backend.interface import AttentionMetadata, AttentionRuntimeFeatures

__all__ = [
    "make_diffusion_attn_metadata",
    "create_diffusion_attn_metadata",
    "prepare_diffusion_attn_metadata",
]

SeqLens = int | Sequence[int] | torch.Tensor


@torch.compiler.disable
def _seqlens_as_s32(value: SeqLens, batch_size: int) -> torch.Tensor:
    """Normalize a per-batch sequence length spec to an int32 CPU tensor."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().flatten().to(device="cpu", dtype=torch.int32)
    elif isinstance(value, int):
        tensor = torch.full((batch_size,), value, dtype=torch.int32)
    else:
        tensor = torch.tensor(list(value), dtype=torch.int32)

    if tensor.shape[0] != batch_size:
        raise ValueError(f"sequence lengths have batch {tensor.shape[0]} but {batch_size=}")
    return tensor


@torch.compiler.disable
def _cu_seqlens(seq_lens: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
    cu = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32)
    torch.cumsum(seq_lens, dim=0, dtype=torch.int32, out=cu[1:])
    device = getattr(attn_metadata, "seq_lens_cuda", None)
    if device is not None:
        cu = cu.to(device=device.device, non_blocking=True)
    return cu


@torch.compiler.disable
def _batch_changed(attn_metadata: AttentionMetadata, batch_size: int) -> bool:
    current = attn_metadata.seq_lens
    return current is None or current.shape[0] != batch_size


@torch.compiler.disable
def _is_already_prepared(
    attn_metadata: AttentionMetadata,
    batch_size: int,
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor | None,
) -> bool:
    """Whether the site already carries exactly this shape."""
    if attn_metadata.num_contexts != batch_size:
        return False
    if _batch_changed(attn_metadata, batch_size):
        return False
    if not torch.equal(attn_metadata.seq_lens, q_lens):
        return False
    if (kv_lens is not None) != attn_metadata.is_cross:
        return False
    if kv_lens is not None and not torch.equal(attn_metadata.seq_lens_kv, kv_lens):
        return False
    return True


@torch.compiler.disable
def create_diffusion_attn_metadata(
    metadata_cls: Type[AttentionMetadata],
    *,
    max_batch_size: int,
    max_seq_len: int,
    mapping: Mapping | None = None,
) -> AttentionMetadata:
    """Allocate one no-KV-cache attention metadata site. Can't be dynamo-traced.

    Args:
        metadata_cls: Concrete metadata type, from the backend's ``Metadata``.
        max_batch_size: Upper bound on sequences per forward.
        max_seq_len: Upper bound on the longest sequence, Q or KV.
        mapping: Defaults to single-rank; diffusion parallelism lives in the
            ``parallel.py`` wrappers, not the kernel.
    """
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    metadata = metadata_cls(
        max_num_requests=max_batch_size,
        max_num_tokens=max_batch_size * max_seq_len,
        max_num_sequences=max_batch_size,
        kv_cache_manager=None,  # Diffusion attention runs without a KV cache.
        mapping=mapping if mapping is not None else Mapping(),
        runtime_features=AttentionRuntimeFeatures(),
        # Makes the `seq_lens` setter copy in place rather than reallocating
        # `seq_lens_cuda` on every assignment.
        is_cuda_graph=True,
    )
    # `max_seq_len` is a property with a manual setter on TrtllmAttentionMetadata
    # (required for the no-cache path) and a plain attribute elsewhere.
    metadata.max_seq_len = max_seq_len
    return metadata


@torch.compiler.disable
def prepare_diffusion_attn_metadata(
    attn_metadata: AttentionMetadata,
    *,
    batch_size: int,
    q_seq_lens: SeqLens,
    kv_seq_lens: SeqLens | None = None,
) -> AttentionMetadata:
    """Populate and prepare one site in place, then return it.

    Every sequence is a context request and there is no KV cache, mirroring
    ``_prepare_qwen_vl_vision_attn_metadata``. Can't be dynamo-traced.

    Args:
        attn_metadata: A site from :func:`create_diffusion_attn_metadata`.
        batch_size: Number of sequences in this forward.
        q_seq_lens: Query length per sequence.
        kv_seq_lens: Key/value length per sequence; ``None`` for self-attention.
    """
    q_lens = _seqlens_as_s32(q_seq_lens, batch_size)

    capacity = getattr(attn_metadata, "max_num_requests", None)
    if capacity is not None and batch_size > capacity:
        raise ValueError(
            f"batch_size={batch_size} exceeds the site's allocated capacity "
            f"({capacity}). Size the site for its actual batch in "
            f"create_diffusion_attn_metadata()."
        )

    kv_lens = None if kv_seq_lens is None else _seqlens_as_s32(kv_seq_lens, batch_size)

    if _is_already_prepared(attn_metadata, batch_size, q_lens, kv_lens):
        # Skipping leaves the device-side length buffers untouched between
        # CUDA graph replays.
        return attn_metadata

    # The in-place `seq_lens` copy needs matching shapes; drop the stale
    # buffers on a batch change so the setter reallocates instead of raising.
    if _batch_changed(attn_metadata, batch_size):
        attn_metadata._seq_lens_cuda = None
        attn_metadata._seq_lens_kv_cuda = None

    # Order matters: `num_contexts` feeds `context_lens`, which the no-cache
    # branch of TrtllmAttentionMetadata.prepare() uses to derive `prompt_lens`.
    attn_metadata.num_contexts = batch_size
    attn_metadata.request_ids = list(range(batch_size))
    attn_metadata.seq_lens = q_lens
    # Always assign: None restores the getter's identity fall-back to
    # `seq_lens`, which is what makes `is_cross` False again.
    attn_metadata.seq_lens_kv = kv_lens

    attn_metadata.cu_q_seqlens = _cu_seqlens(attn_metadata.seq_lens, attn_metadata)
    attn_metadata.cu_kv_seqlens = (
        attn_metadata.cu_q_seqlens
        if kv_lens is None
        else _cu_seqlens(attn_metadata.seq_lens_kv, attn_metadata)
    )

    # Full `prepare()`, not `prepare_encoder_only()`: the latter binds KV
    # lengths to the query lengths, wrong for a cross site.
    attn_metadata.prepare()
    return attn_metadata


@torch.compiler.disable
def make_diffusion_attn_metadata(
    metadata_cls: Type[AttentionMetadata],
    *,
    batch_size: int,
    q_seq_lens: SeqLens,
    kv_seq_lens: SeqLens | None = None,
) -> AttentionMetadata:
    """Allocate and prepare one attention site in a single call.

    The entry point a model's ``create_attn_metadata()`` uses, one call per
    site. Sizes the site from the lengths it is given, so the caller only states
    the shape the attention module will be called with.

    Can't be dynamo-traced; see :func:`prepare_diffusion_attn_metadata`.

    Args:
        metadata_cls: Concrete metadata type, normally the model's
            ``attn_backend_metadata_cls``.
        batch_size: Number of sequences in this forward.
        q_seq_lens: Query length, either shared by the batch or one per sequence.
        kv_seq_lens: Key/value length; ``None`` for self-attention.
    """
    q_lens = _seqlens_as_s32(q_seq_lens, batch_size)
    max_seq_len = int(q_lens.max())
    if kv_seq_lens is not None:
        max_seq_len = max(max_seq_len, int(_seqlens_as_s32(kv_seq_lens, batch_size).max()))

    return prepare_diffusion_attn_metadata(
        create_diffusion_attn_metadata(
            metadata_cls, max_batch_size=batch_size, max_seq_len=max_seq_len
        ),
        batch_size=batch_size,
        q_seq_lens=q_seq_lens,
        kv_seq_lens=kv_seq_lens,
    )
