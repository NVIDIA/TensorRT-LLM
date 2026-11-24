# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 ULL target (``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``): the KDA + DSA hybrid
cache manager, the KDA replay / checkpoint state and the cache policy, in four sections (the FP8-KV
layer policy, the KDA checkpoint layout, the flat KDA cache manager, the KDA + DSA cache manager).
"""

from __future__ import annotations

import itertools
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    _get_indexer_k_cache_bytes_per_token,
    _get_indexer_k_cache_size_per_token,
    derive_indexer_k_cache_layer_mask,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    Role,
    _BasePageTableMaterializer,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaRole,
    MambaStateLayout,
    _get_local_mamba_cache_layout,
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.step_upload_kernels import (
    block_offsets_triton_enabled,
    copy_batch_block_offsets_from_host,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
    exact_div,
)

from ...kernels.ssm.rubin_ull_kda_checkpoint_seed import seed_checkpoints, seed_checkpoints_applies
from ..common.rubin_ull_features import active as k31_features
from .draft_window_cache import DraftWindowCacheMixin, set_draft_window, window_draft_cost
from .rubin_kda_dsa_cache_manager import DRAFT_WINDOW_ATTR, draft_cache_option

_kda_cache_manager = sys.modules[__name__]


# ---- the FP8-KV layer cache policy ----------------------------------------
#: The feature field that selects the layers.
FIELD = "fp8_kv_layers"


def _get_kimi_k31_text_config(pretrained_config: object) -> object:
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is not None:
        return text_config
    return pretrained_config


def get_kimi_k31_fp8_kv_layer_ids(
    pretrained_config: object, *, nvfp4_enabled: bool = False
) -> frozenset[int]:
    """Return zero-based Kimi K3.1 MLA layers that use FP8 KV cache.

    With ``fp8_kv_layers`` None, an NVFP4 KV cache keeps the last two MLA layers of the global
    layer schedule in FP8 and any other KV cache none. ``()`` selects none; explicit layer ids
    select those layers, and the model call site rejects them without an NVFP4 KV cache.

    Raises:
        TypeError: ``fp8_kv_layers`` is neither None nor a tuple of ints.
        ValueError: An explicit layer id is out of range or not an MLA layer.
    """
    config = _get_kimi_k31_text_config(pretrained_config)
    if getattr(config, "dsa_config", None) is None:
        return frozenset()

    selected = k31_features().fp8_kv_layers
    if selected is not None:
        if not isinstance(selected, tuple) or any(type(layer) is not int for layer in selected):
            raise TypeError(
                f"Kimi K3.1 features: {FIELD} must be None or a tuple of zero-based layer IDs, "
                f"got {selected!r}."
            )
        if not selected:
            return frozenset()
    elif not nvfp4_enabled:
        return frozenset()

    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or getattr(
        pretrained_config, "architectures", None
    )
    is_kimi_k3 = model_type in ("kimi_k3", "kimi_k31", "kimi_k3_text", "kimi_linear") or any(
        "KimiK3" in architecture or "KimiLinear" in architecture
        for architecture in (architectures or ())
    )
    if not is_kimi_k3:
        return frozenset()

    linear_attn_config = getattr(config, "linear_attn_config", None)
    if linear_attn_config is None:
        raise ValueError(f"Kimi K3.1 features: {FIELD} requires Kimi K3's linear_attn_config.")
    full_attn_layers = linear_attn_config.get("full_attn_layers", ())
    mla_layer_ids = {int(layer_number) - 1 for layer_number in full_attn_layers}
    if selected is None:
        layer_ids = frozenset(sorted(mla_layer_ids)[-2:])
    else:
        layer_ids = frozenset(selected)

    num_hidden_layers = int(config.num_hidden_layers)
    out_of_range = sorted(
        layer_id for layer_id in layer_ids if not 0 <= layer_id < num_hidden_layers
    )
    if out_of_range:
        raise ValueError(
            f"Kimi K3.1 features: {FIELD} contains layer IDs outside "
            f"[0, {num_hidden_layers}): {out_of_range}."
        )

    non_mla_layers = sorted(layer_ids - mla_layer_ids)
    if non_mla_layers:
        raise ValueError(
            f"Kimi K3.1 features: {FIELD} can select only MLA layers; "
            f"these layers use KDA instead: {non_mla_layers}."
        )

    return layer_ids


__all__ = [
    "FIELD",
    "get_kimi_k31_fp8_kv_layer_ids",
]


# ---- the KDA checkpoint layout ----------------------------------------
KDA_CHECKPOINT_ROLE = "kda_ckpt"
_KDA_CKPT_INDEX_BYTES = 16
_KDA_CKPT_HEADER_BYTES = 256


@dataclass(frozen=True)
class KDACheckpointLayout:
    """One KDA layer's checkpoint-mode payload of a state slot (vLLM MR8's spec cache semantics).

    Byte layout of the slot's ``kda_ckpt`` role (one KVCM buffer, so every view below shares
    the slot stride ``P``)::

        [256 B unused][num_checkpoints x state][conv page][pad to 256 B]

    Checkpoint ``t`` holds the recurrent state after verify position ``t``; the conv page
    ``[conv_dim, conv_window + num_spec]`` holds the last ``conv_window`` committed raw inputs
    from column ``accepted - 1`` on. vLLM's kernels index one state tensor and one conv tensor
    with int32 ``slots[b, t]`` (conv: ``slots[b, 0]``) and treat ``slot <= 0`` as padding; with
    both flat views at a row stride of ``U`` bytes (:func:`kda_checkpoint_views`: the largest
    power of two up to 256 dividing ``P``, the buffer address and the state size),
    ``slots[b, t] = 256 / U + s * P / U + t * state_bytes / U`` addresses checkpoint ``t`` and the
    conv page of slot ``s``. The leading bytes keep every real index positive without a view
    before the buffer; the 256-byte header and padding let ``U`` reach 256 where the slot's other
    roles allow it, so a pool of ``slots * P`` bytes needs ``slots * P / U`` rows (int32 up to
    512 GiB at ``U`` = 256; 32 GiB at 16 bytes).
    """

    num_checkpoints: int
    state_shape: tuple[int, int, int]
    state_dtype: torch.dtype
    conv_dim: int
    conv_width: int
    conv_dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.state_dtype != torch.float32 or self.conv_dtype != torch.bfloat16:
            raise TypeError("ULL checkpoints require FP32 KDA state and BF16 raw conv inputs")
        if self.num_checkpoints <= 0 or self.conv_dim <= 0 or self.conv_width <= 0:
            raise ValueError("KDA checkpoint dimensions must be positive")
        if len(self.state_shape) != 3 or any(dim <= 0 for dim in self.state_shape):
            raise ValueError("KDA recurrent state requires positive [heads,V,K] dimensions")

    @property
    def state_bytes(self) -> int:
        return self.state_dtype.itemsize * math.prod(self.state_shape)

    @property
    def conv_offset(self) -> int:
        return _KDA_CKPT_HEADER_BYTES + self.num_checkpoints * self.state_bytes

    @property
    def nbytes(self) -> int:
        conv_bytes = self.conv_dtype.itemsize * self.conv_dim * self.conv_width
        end = self.conv_offset + conv_bytes
        return -(-end // _KDA_CKPT_HEADER_BYTES) * _KDA_CKPT_HEADER_BYTES


@dataclass(frozen=True)
class KDACheckpointViews:
    """Views of one layer's ``kda_ckpt`` buffer (``raw``: ``[slots, nbytes]`` uint8)."""

    ckpt: torch.Tensor
    vconv: torch.Tensor
    state_flat: torch.Tensor
    conv_flat: torch.Tensor
    slot_units: int
    state_units: int
    header_units: int = 1


def _kda_checkpoint_unit(slot_bytes: int, address: int, state_bytes: int) -> int:
    """The row stride of the flat kernel views: the largest power of two from 16 up to the header
    size that divides the slot stride, the buffer address and the state size."""
    unit = _KDA_CKPT_HEADER_BYTES
    while unit > _KDA_CKPT_INDEX_BYTES and (
        slot_bytes % unit or address % unit or state_bytes % unit
    ):
        unit //= 2
    return unit


def kda_checkpoint_views(raw: torch.Tensor, layout: KDACheckpointLayout) -> KDACheckpointViews:
    """The typed per-slot views and vLLM's flat kernel views of a ``kda_ckpt`` buffer (row stride:
    :func:`_kda_checkpoint_unit`)."""
    if raw.dtype != torch.uint8 or raw.dim() != 2 or raw.stride(1) != 1:
        raise ValueError("kda_ckpt buffer must be a [slots, bytes] uint8 view")
    slots, slot_bytes = raw.shape[0], raw.stride(0)
    unit = _kda_checkpoint_unit(slot_bytes, raw.data_ptr(), layout.state_bytes)
    if raw.shape[1] < layout.nbytes or slot_bytes % unit or raw.data_ptr() % unit:
        raise ValueError(
            f"kda_ckpt buffer [{raw.shape[0]}, {raw.shape[1]}] (slot stride {slot_bytes} B) "
            f"cannot hold {layout.nbytes} B per slot at a {unit}-byte alignment"
        )
    if layout.state_bytes % unit:
        raise ValueError(f"KDA checkpoint states must be a multiple of {unit} bytes")
    n, (h, v, k) = layout.num_checkpoints, layout.state_shape
    state_isz, conv_isz = layout.state_dtype.itemsize, layout.conv_dtype.itemsize
    states = raw[:, _KDA_CKPT_HEADER_BYTES : layout.conv_offset].view(layout.state_dtype)
    ckpt = states.unflatten(-1, (n, h, v, k))
    conv_bytes = layout.conv_dim * layout.conv_width * conv_isz
    page = raw[:, layout.conv_offset : layout.conv_offset + conv_bytes].view(layout.conv_dtype)
    vconv = page.unflatten(-1, (layout.conv_dim, layout.conv_width))
    slot_units, state_units = slot_bytes // unit, layout.state_bytes // unit
    header = _KDA_CKPT_HEADER_BYTES // unit
    rows = (slots - 1) * slot_units + header + (n - 1) * state_units + 1
    if rows > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"{slots} KDA checkpoint slots of {slot_bytes} B at a {unit}-byte row stride exceed "
            "vLLM's int32 slot indices"
        )
    # Row r of a flat view starts at byte r * unit of the buffer: slot s's checkpoint t at
    # header + s * slot_units + t * state_units, its conv page at header + s * slot_units.
    typed = raw.view(layout.state_dtype)
    state_flat = typed.as_strided(
        (rows, h, v, k), (unit // state_isz, v * k, k, 1), typed.storage_offset()
    )
    # vLLM reads a request's conv page only at its first index (``slots[b, 0]``, the slot's row), so
    # the conv view ends at the last slot's row; a row past it would overrun the buffer.
    conv_rows = (slots - 1) * slot_units + header + 1
    typed = raw.view(layout.conv_dtype)
    conv_flat = typed.as_strided(
        (conv_rows, layout.conv_dim, layout.conv_width),
        (unit // conv_isz, layout.conv_width, 1),
        typed.storage_offset() + (layout.conv_offset - _KDA_CKPT_HEADER_BYTES) // conv_isz,
    )
    return KDACheckpointViews(ckpt, vconv, state_flat, conv_flat, slot_units, state_units, header)


def seed_kda_checkpoints(
    ckpt: torch.Tensor,
    vconv: torch.Tensor,
    ssm_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Checkpoint 0 and the committed conv window of ``slots`` from the live pools."""
    if slots.numel() == 0:
        return
    window = conv_pool.shape[-1]
    ckpt[slots, 0] = ssm_pool.index_select(0, slots).to(ckpt.dtype)
    vconv[slots, :, :window] = conv_pool.index_select(0, slots).to(vconv.dtype)


def commit_kda_checkpoints(
    ckpt: torch.Tensor,
    vconv: torch.Tensor,
    ssm_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    slots: torch.Tensor,
    accepted_drafts: torch.Tensor,
) -> None:
    """The live pools of ``slots`` from their current checkpoints (``accepted_drafts[slot]``)."""
    if slots.numel() == 0:
        return
    window = conv_pool.shape[-1]
    position = accepted_drafts.index_select(0, slots).long()
    ssm_pool.index_copy_(0, slots, ckpt[slots, position].to(ssm_pool.dtype))
    columns = position[:, None, None] + torch.arange(window, device=slots.device)
    pages = vconv.index_select(0, slots)
    columns = columns.expand(-1, pages.shape[1], -1)
    conv_pool.index_copy_(0, slots, pages.gather(2, columns).to(conv_pool.dtype))


def checkpoint_indices(
    physical_slots: torch.Tensor,
    accepted_drafts: torch.Tensor,
    views: KDACheckpointViews,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MR8 int32 position indices and accepted-token counts for physical slots.

    ``accepted_drafts`` is the graph-stable per-physical-slot counter owned by
    the speculative cache manager. It excludes the target token; MR8 includes
    it. Device operations only, including when called during graph capture.
    """
    indices = physical_slots.long()
    positions = torch.arange(views.ckpt.shape[1], device=indices.device)
    rows = views.header_units + indices[:, None] * views.slot_units
    rows = rows + positions * views.state_units
    accepted = accepted_drafts.index_select(0, indices) + 1
    return rows.to(torch.int32), accepted.to(torch.int32)


# ---- the flat KDA cache manager ----------------------------------------
if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


_MAMBA_SSM_SEED_MASK = (1 << 62) - 1
_MAMBA_SSM_UINT64_MASK = (1 << 64) - 1
_MAMBA_SSM_SEED_BASE = 0x6A09E667F3BCC908
_MAMBA_SSM_SEED_MIX_COUNTER = 0x2545F4914F6CDD1D
_MAMBA_SSM_SEED_MIX_SLOT = 0x1B873593CC9E2D51
_MAMBA_SSM_SEED_MIX_RANK = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """Return the SplitMix64 finalizer for ``value``."""
    value = (value + 0x9E3779B97F4A7C15) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 31
    return value & _MAMBA_SSM_UINT64_MASK


def compute_deterministic_mamba_seed(counter: int, slot: int, rank_offset: int) -> int:
    """Return a reproducible, nonzero int64 Philox seed."""
    folded = (
        _MAMBA_SSM_SEED_BASE
        + counter * _MAMBA_SSM_SEED_MIX_COUNTER
        + slot * _MAMBA_SSM_SEED_MIX_SLOT
        + rank_offset * _MAMBA_SSM_SEED_MIX_RANK
    ) & _MAMBA_SSM_UINT64_MASK
    seed = _splitmix64(folded) & _MAMBA_SSM_SEED_MASK
    return seed or 1


def allocate_mamba_seed_buffer(
    cache_size: int, rank_offset: int, device: torch.device
) -> torch.Tensor:
    """Allocate the graph-stable per-slot Philox seed buffer."""
    seeds = [compute_deterministic_mamba_seed(0, slot, rank_offset) for slot in range(cache_size)]
    return torch.tensor(seeds, dtype=torch.int64, device=device)


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayerCache(MambaLayerCache):
    """Per-layer KDA replay tensors consumed by fused multi-token verify."""

    prev_num_accepted_tokens: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    kda_conv_q: torch.Tensor | None = None
    kda_conv_k: torch.Tensor | None = None
    kda_conv_v: torch.Tensor | None = None
    kda_k_cache: torch.Tensor | None = None
    kda_g_cache: torch.Tensor | None = None
    kda_v_cache: torch.Tensor | None = None
    kda_beta_cache: torch.Tensor | None = None
    # [slots, 3, num_spec, heads, head_dim]: fp32 vn, beta * k, and decay.
    kda_state_tok: torch.Tensor | None = None
    # Checkpoint mode (see ``KDACheckpointLayout``): [slots, 1 + num_spec, H, V, K] recurrent
    # states, one per verify position, in the pool's state dtype.
    kda_ckpt: torch.Tensor | None = None
    # [slots, conv_dim, W - 1 + num_spec] raw convolution inputs, dimension-first.
    kda_vconv: torch.Tensor | None = None
    # The same storage as the flat ``[conv, state]`` caches vLLM MR8's KDA kernels index with
    # ``KDAReplayState.checkpoint_indices``.
    kda_ckpt_kernel_caches: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def has_kda_replay_caches(self) -> bool:
        return self.kda_conv_q is not None

    @property
    def has_kda_checkpoints(self) -> bool:
        return self.kda_ckpt is not None

    def commit_conv_window(self, slot_indices: torch.Tensor, conv_pool: torch.Tensor) -> None:
        """Seed the replay convolution windows from the live pool.

        In checkpoint mode, seed checkpoint 0 and the committed convolution window from the live
        pools instead: a prefill or a plain decode step advanced them, and the next verify starts
        from position 0 (the step's acceptance records no drafts).
        """
        if self.has_kda_checkpoints:
            seed_kda_checkpoints(
                self.kda_ckpt, self.kda_vconv, self.temporal, conv_pool, slot_indices.long()
            )
            return
        from ...kernels.ssm.rubin_ull_kda_kernels import copy_kda_replay_conv_window

        copy_kda_replay_conv_window(
            conv_pool,
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            slot_indices,
        )

    def load_checkpoints(self, slot_indices: torch.Tensor) -> None:
        """Write the slots' current checkpoints (``prev_num_accepted_tokens``) to the live pools,
        for a step that reads the pools (plain decode)."""
        commit_kda_checkpoints(
            self.kda_ckpt,
            self.kda_vconv,
            self.temporal,
            self.conv,
            slot_indices.long(),
            self.prev_num_accepted_tokens,
        )


_KDA_CKPT_ROLE = "kda_ckpt"
# Smallest row stride of the flat kernel views (vLLM's 16-byte state alignment).
_KDA_CKPT_INDEX_BYTES = 16
# Leading unused bytes of a slot's payload and its size granule: the largest row stride.
_KDA_CKPT_HEADER_BYTES = 256


_KDA_BETA_CACHE_ALIGNMENT_BYTES = 16
# KVCM copies 16-byte payloads; the replay kernel reads only the first int32.
_KDA_REPLAY_COUNT_BYTES = 16
_KDA_REPLAY_COUNT_ROLE = DataRole("prev_num_accepted_tokens")
_KDA_REPLAY_CONV_ROLES = ("kda_conv_q", "kda_conv_k", "kda_conv_v")
_KDA_REPLAY_DRAFT_ROLES = ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache")
_KDA_DRAFT_RECORDS_ROLE = "kda_state_tok"


def _kda_replay_layout(
    num_spec: int, num_heads: int, head_dim: int, conv_window: int, draft_records: bool = False
) -> dict[str, tuple[torch.dtype, list[int]]]:
    """Per-slot physical shapes of one layer's KDA replay caches.

    Conv caches hold raw bf16 projection inputs (exact in bf16); k / v / beta
    are bf16 and the gate stays fp32 because it feeds the cumulative-gate
    exponents of the replayed chunk. Beta rows keep the 16-byte CuTe
    alignment by padding the head dimension.

    With ``draft_records``, the records of every draft of the last verify
    round (``kda_state_tok``: per draft, the row innovations vn, then beta * k
    and the decay, each fp32 [H, K] with V = K = ``head_dim``) take the place
    of the k / g / v / beta caches.
    """
    dim = num_heads * head_dim
    layout = {
        "kda_conv_q": (torch.bfloat16, [conv_window + num_spec, dim]),
        "kda_conv_k": (torch.bfloat16, [conv_window + num_spec, dim]),
        "kda_conv_v": (torch.bfloat16, [conv_window + num_spec, dim]),
    }
    if draft_records:
        layout[_KDA_DRAFT_RECORDS_ROLE] = (torch.float32, [3, num_spec, num_heads, head_dim])
        return layout
    heads_per_alignment = _KDA_BETA_CACHE_ALIGNMENT_BYTES // torch.bfloat16.itemsize
    aligned_heads = (
        (num_heads + heads_per_alignment - 1) // heads_per_alignment
    ) * heads_per_alignment
    layout.update(
        {
            "kda_k_cache": (torch.bfloat16, [num_spec, dim]),
            "kda_g_cache": (torch.float32, [num_spec, dim]),
            "kda_v_cache": (torch.bfloat16, [num_spec, dim]),
            "kda_beta_cache": (torch.bfloat16, [num_spec, aligned_heads]),
        }
    )
    return layout


def _kda_checkpoint_layout(
    num_spec: int,
    num_heads: int,
    head_dim: int,
    conv_window: int,
    state_dtype: torch.dtype,
    conv_dtype: torch.dtype,
) -> KDACheckpointLayout:
    """Per-slot checkpoint-mode payload of one layer: ``1 + num_spec`` states, one conv page."""
    return KDACheckpointLayout(
        num_checkpoints=1 + num_spec,
        state_shape=(num_heads, head_dim, head_dim),
        state_dtype=state_dtype,
        conv_dim=3 * num_heads * head_dim,
        conv_width=conv_window + num_spec,
        conv_dtype=conv_dtype,
    )


def _kda_replay_bytes_per_slot(
    layout: dict[str, tuple[torch.dtype, list[int]]], num_local_layers: int
) -> int:
    """Replay bytes one recurrent-state slot holds across local layers."""
    if not layout or num_local_layers == 0:
        return 0
    layer_bytes = sum(dtype.itemsize * math.prod(shape) for dtype, shape in layout.values())
    return num_local_layers * layer_bytes + _KDA_REPLAY_COUNT_BYTES


def _host_slots_to_device(slots: list[int], device: torch.device) -> torch.Tensor:
    """Copy host slot IDs without synchronizing queued GPU work.

    Each copy uses fresh pinned storage so later host writes cannot race it.
    """
    return torch.tensor(
        slots, dtype=torch.long, pin_memory=device.type == "cuda" and prefer_pinned()
    ).to(device, non_blocking=True)


def _kda_beta_cache_padded_heads(num_heads: int) -> int:
    heads_per_alignment = _KDA_BETA_CACHE_ALIGNMENT_BYTES // torch.bfloat16.itemsize
    return (num_heads + heads_per_alignment - 1) // heads_per_alignment * heads_per_alignment


def _allocate_kda_beta_cache(shape: tuple[int, ...], device: torch.device | None) -> torch.Tensor:
    """Keep logical head count while aligning each physical row for CuTe."""
    return torch.zeros(
        *shape[:-1],
        _kda_beta_cache_padded_heads(shape[-1]),
        dtype=torch.bfloat16,
        device=device,
    )[..., : shape[-1]]


def allocate_kda_replay_fields(
    *,
    num_local_layers: int,
    cache_size: int,
    num_speculative_tokens: int,
    conv_dim: int,
    conv_kernel_size: int,
    num_heads: int,
    device: torch.device,
    draft_records: bool = False,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Allocate the complete KDA fused-verify replay payload."""
    if conv_dim % 3 != 0:
        raise ValueError("KDA replay caches require q/k/v convolution sections of equal width")
    section_dim = conv_dim // 3
    extended_window = conv_kernel_size - 1 + num_speculative_tokens

    def allocate_conv_cache() -> torch.Tensor:
        return torch.zeros(
            num_local_layers,
            cache_size,
            extended_window,
            section_dim,
            dtype=torch.bfloat16,
            device=device,
        ).transpose(-1, -2)

    fields = {
        "prev_num_accepted_tokens": torch.zeros(cache_size, dtype=torch.int32, device=device),
        "kda_conv_q": allocate_conv_cache(),
        "kda_conv_k": allocate_conv_cache(),
        "kda_conv_v": allocate_conv_cache(),
    }
    if draft_records:
        fields["kda_state_tok"] = torch.zeros(
            num_local_layers,
            cache_size,
            3,
            num_speculative_tokens,
            num_heads,
            section_dim // num_heads,
            dtype=torch.float32,
            device=device,
        )
        return fields, [
            value for name, value in fields.items() if name != "prev_num_accepted_tokens"
        ]
    fields.update(
        {
            "kda_k_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            "kda_g_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.float32,
                device=device,
            ),
            "kda_v_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            "kda_beta_cache": _allocate_kda_beta_cache(
                (num_local_layers, cache_size, num_speculative_tokens, num_heads), device
            ),
        }
    )
    scratch = [
        fields["kda_conv_q"],
        fields["kda_conv_k"],
        fields["kda_conv_v"],
        fields["kda_k_cache"],
        fields["kda_g_cache"],
        fields["kda_v_cache"],
        fields["kda_beta_cache"],
    ]
    return fields, scratch


@dataclass(frozen=True, kw_only=True)
class KDAIntermediateLayerCache(MambaLayerCache):
    """Generic intermediate-state view used during speculative decoding."""

    intermediate_conv_window: torch.Tensor | None = None
    intermediate_ssm: torch.Tensor | None = None


class KDAIntermediateState:
    """Own conventional speculative intermediate states and promotion."""

    def __init__(self) -> None:
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_ssm: torch.Tensor | None = None
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_ssm: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow persistent per-layer views until shutdown; allocate owned scratch."""
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._stacked_ssm = _stack_state_views(ssm_states)
        self._stacked_conv = _stack_state_views(conv_states)
        if not context.mamba_pp_layers:
            return

        device = ssm_states[0].device

        if context.spec_config is not None:
            tokens_per_step = context.spec_config.tokens_per_gen_step
            common_shape = [
                len(context.mamba_pp_layers),
                context.max_batch_size,
                tokens_per_step,
            ]
            self.intermediate_ssm = torch.zeros(
                common_shape + list(context.ssm_state_shape),
                dtype=context.ssm_state_dtype,
                device=device,
            )
            self.intermediate_conv = torch.zeros(
                common_shape + list(context.conv_state_shape),
                dtype=context.conv_state_dtype,
                device=device,
            )
            self.intermediate_indices = torch.arange(
                context.max_batch_size, dtype=torch.int32, device=device
            )

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields: dict[str, torch.Tensor | None] = {}
        if self.intermediate_conv is not None:
            fields["intermediate_conv_window"] = self.intermediate_conv[layer_offset]
        if self.intermediate_ssm is not None:
            fields["intermediate_ssm"] = self.intermediate_ssm[layer_offset]
        return fields

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        """Plain intermediate buffers have no request-persistent bookkeeping."""

    def update(self, batch: MambaAcceptanceBatch) -> None:
        if self.intermediate_indices is None:
            return
        _promote_intermediate_states(
            self._ssm_states, self.intermediate_ssm, batch, self._stacked_ssm
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def shutdown(self) -> None:
        self._ssm_states = ()
        self._conv_states = ()
        self._stacked_ssm = None
        self._stacked_conv = None
        self.intermediate_ssm = None
        self.intermediate_conv = None
        self.intermediate_indices = None

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache:
        return KDAIntermediateLayerCache(
            conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset)
        )


class KDAReplayState:
    """Own all V2 state used by the KDA fused multi-token verifier.

    With ``checkpoints``, the slots keep vLLM MR8's spec cache semantics instead of the replay
    contract: one recurrent-state checkpoint per verify position and one extended conv page
    (``KDACheckpointLayout``, role ``kda_ckpt``), read at ``prev_num_accepted_tokens`` (the
    accepted drafts of the last verify, recorded as for replay). Checkpoint 0 and the conv page
    are seeded from the live pools after a prefill, a plain decode or a transfer, and the
    current checkpoint is committed back to them before a suspension or a block-reuse release.
    """

    @property
    def intermediate_ssm(self) -> torch.Tensor | None:
        return None

    @property
    def intermediate_conv(self) -> torch.Tensor | None:
        return None

    def __init__(
        self,
        num_speculative_tokens: int,
        *,
        stochastic_rounding: bool = False,
        draft_records: bool = False,
        checkpoints: bool = False,
    ) -> None:
        if checkpoints and (draft_records or stochastic_rounding):
            raise ValueError(
                "KDA checkpoints exclude draft records and stochastic rounding of the state"
            )
        self._stochastic_rounding = stochastic_rounding
        self.draft_records = draft_records
        self.checkpoints = checkpoints
        self.checkpoint_layout: KDACheckpointLayout | None = None
        # The flat kernel views' slot and state strides and header in rows (checkpoint mode).
        self.checkpoint_slot_units = 0
        self.checkpoint_state_units = 0
        self.checkpoint_header_units = 1
        self.pool_layers: list[dict[str, torch.Tensor]] | None = None
        self.num_speculative_tokens = num_speculative_tokens
        self.suspended_requests: set[int] = set()
        self.acceptance_pending: set[int] = set()
        self.unseeded_requests: set[int] = set()
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._committed_window = 0
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.kda_conv_q: torch.Tensor | None = None
        self.kda_conv_k: torch.Tensor | None = None
        self.kda_conv_v: torch.Tensor | None = None
        self.kda_k_cache: torch.Tensor | None = None
        self.kda_g_cache: torch.Tensor | None = None
        self.kda_v_cache: torch.Tensor | None = None
        self.kda_beta_cache: torch.Tensor | None = None
        self.kda_state_tok: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self._conv_section_dims: tuple[int, ...] = ()
        # Batched seeding plan of seed_transferred_slots: None until first use,
        # False when the per-layer loop is used.
        self._batched_seed_plan: object | None = None

    def validate(self, context: MambaStateLayout) -> None:
        if context.spec_config is None:
            raise ValueError("KDA replay caches require speculative decoding")
        expected_num_spec = context.spec_config.tokens_per_gen_step - 1
        if self.num_speculative_tokens != expected_num_spec:
            raise ValueError(
                "KDA replay cache width "
                f"({self.num_speculative_tokens}) must match the draft length "
                f"({expected_num_spec})"
            )
        if not context.mamba_pp_layers:
            return
        if context.conv_state_layout != "q_k_v":
            raise ValueError("KDA replay requires conv_state_layout='q_k_v'")
        if len(context.conv_section_dims) != 3 or len(set(context.conv_section_dims)) != 1:
            raise ValueError(
                "KDA replay caches require equal [Q | K | V] convolution-state sections"
            )

    def pool_layout(self, context: MambaStateLayout) -> dict[str, tuple[torch.dtype, list[int]]]:
        if not context.mamba_pp_layers:
            return {}
        if len(context.conv_section_dims) != 3 or len(set(context.conv_section_dims)) != 1:
            raise ValueError(
                "KDA replay caches require equal [Q | K | V] convolution-state sections"
            )
        if (self.draft_records or self.checkpoints) and (
            context.ssm_state_shape[1] != context.ssm_state_shape[-1]
        ):
            raise ValueError(
                "KDA draft records and checkpoints require equal value and key head dimensions"
            )
        if self.checkpoints:
            layout = self._checkpoint_layout(context)
            return {_KDA_CKPT_ROLE: (torch.uint8, [layout.nbytes])}
        return _kda_replay_layout(
            self.num_speculative_tokens,
            context.ssm_state_shape[0],
            context.ssm_state_shape[1],
            context.conv_state_shape[1],
            draft_records=self.draft_records,
        )

    def _checkpoint_layout(self, context: MambaStateLayout) -> KDACheckpointLayout:
        return _kda_checkpoint_layout(
            self.num_speculative_tokens,
            context.ssm_state_shape[0],
            context.ssm_state_shape[1],
            context.conv_state_shape[1],
            context.ssm_state_dtype,
            context.conv_state_dtype,
        )

    def pool_buffer_configs(self, context: MambaStateLayout, layer_id: int) -> list[BufferConfig]:
        layout = self.pool_layout(context)
        buffers = [
            BufferConfig(role=DataRole(name), size=dtype.itemsize * math.prod(shape))
            for name, (dtype, shape) in layout.items()
        ]
        if layout and layer_id == context.mamba_pp_layers[0]:
            buffers.append(BufferConfig(role=_KDA_REPLAY_COUNT_ROLE, size=self.count_bytes))
        return buffers

    @property
    def count_bytes(self) -> int:
        """Bytes of the slot's accepted-count buffer. Checkpoint mode pads it to the checkpoint
        payload's granule: where the manager coalesces it with the payloads, the slot stride stays
        a multiple of the flat views' largest row stride."""
        return _KDA_CKPT_HEADER_BYTES if self.checkpoints else _KDA_REPLAY_COUNT_BYTES

    def disagg_ignored_roles(self, context: MambaStateLayout) -> frozenset[DataRole]:
        roles = {DataRole(name) for name in self.pool_layout(context)}
        if roles:
            roles.add(_KDA_REPLAY_COUNT_ROLE)
        return frozenset(roles)

    def _layer_bytes_per_slot(self, section_dim: int, committed_window: int, num_heads: int) -> int:
        extended_window = committed_window + self.num_speculative_tokens
        if self.draft_records:
            return (
                3 * section_dim * extended_window * torch.bfloat16.itemsize
                + 3 * self.num_speculative_tokens * section_dim * torch.float32.itemsize
            )
        bf16_elements = (
            3 * section_dim * extended_window
            + self.num_speculative_tokens * 2 * section_dim
            + self.num_speculative_tokens * _kda_beta_cache_padded_heads(num_heads)
        )
        gate_elements = self.num_speculative_tokens * section_dim
        return bf16_elements * torch.bfloat16.itemsize + gate_elements * torch.float32.itemsize

    def bytes_per_slot(self, context: MambaStateLayout, layer_id: int) -> int:
        if not context.mamba_pp_layers:
            return 0
        shared_bytes = 0
        if layer_id == context.mamba_pp_layers[0]:
            shared_bytes = 4
            if self._stochastic_rounding:
                shared_bytes += 8
        if self.checkpoints:
            return self._checkpoint_layout(context).nbytes + shared_bytes
        return (
            self._layer_bytes_per_slot(
                context.conv_section_dims[0],
                context.conv_state_shape[1],
                context.ssm_state_shape[0],
            )
            + shared_bytes
        )

    @classmethod
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        """Estimate rank-local replay storage using the runtime replay gate and geometry."""
        spec_config = kwargs.get("spec_config")
        num_spec = kwargs.get("kda_replay_num_spec")
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(spec_config, manager_supports_replay=True)
        if num_spec is None:
            return 0
        params, local_layers, _ = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        if not local_layers:
            return 0
        tp_size = _mamba_effective_tp_size(mapping)
        quant_config = model_config.quant_config
        stochastic_rounding = kwargs.get(
            "mamba_ssm_stochastic_rounding",
            getattr(quant_config, "mamba_ssm_stochastic_rounding", False)
            and params.mamba_ssm_cache_dtype == torch.float16,
        )
        if kwargs.get("kda_checkpoints", model_uses_kda_checkpoints(model_config)):
            layout = _kda_checkpoint_layout(
                num_spec,
                params.num_heads // tp_size,
                params.head_dim,
                params.conv_kernel - 1,
                params.mamba_ssm_cache_dtype or params.dtype,
                params.dtype,
            )
            return _kda_replay_bytes_per_slot(
                {_KDA_CKPT_ROLE: (torch.uint8, [layout.nbytes])}, local_layers
            )
        state = cls(num_spec, draft_records=kwargs.get("kda_draft_records", False))
        return _kda_replay_bytes_per_slot(
            _kda_replay_layout(
                num_spec,
                params.num_heads // tp_size,
                params.head_dim,
                params.conv_kernel - 1,
                draft_records=state.draft_records,
            ),
            local_layers,
        ) + (8 if stochastic_rounding else 0)

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow persistent views and bind bookkeeping; standalone use allocates replay storage."""
        self._seed_rank_offset = (
            context.mapping.tp_rank * 1_000_003
            + context.mapping.pp_rank * 1_000_033
            + context.mapping.rank * 1_009
        )
        self._conv_section_dims = context.conv_section_dims
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._batched_seed_plan = None
        self._committed_window = context.conv_state_shape[1] if conv_states else 0
        states = ssm_states
        if not states:
            return

        cache_size = context.slot_capacity
        if cache_size is None:
            raise RuntimeError("KDA replay requires the allocated V2 slot capacity")
        device = states[0].device
        if self.pool_layers is None:
            if self.checkpoints:
                raise RuntimeError("KDA checkpoints live in the V2 manager's pools")
            fields, _ = allocate_kda_replay_fields(
                num_local_layers=len(context.mamba_pp_layers),
                cache_size=cache_size,
                num_speculative_tokens=self.num_speculative_tokens,
                conv_dim=context.conv_state_shape[0],
                conv_kernel_size=context.conv_state_shape[1] + 1,
                num_heads=context.ssm_state_shape[0],
                device=device,
                draft_records=self.draft_records,
            )
            for name, value in fields.items():
                setattr(self, name, value)
        self.intermediate_indices = torch.arange(
            context.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        if self._stochastic_rounding:
            self.rand_seed = allocate_mamba_seed_buffer(
                cache_size,
                self._seed_rank_offset,
                device,
            )
        logger.info(f"Mamba Cache (kda-replay) is allocated for {cache_size} state slots")

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        if self.pool_layers is not None:
            return {
                **self.pool_layers[layer_offset],
                "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            }
        return {
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "kda_conv_q": None if self.kda_conv_q is None else self.kda_conv_q[layer_offset],
            "kda_conv_k": None if self.kda_conv_k is None else self.kda_conv_k[layer_offset],
            "kda_conv_v": None if self.kda_conv_v is None else self.kda_conv_v[layer_offset],
            "kda_k_cache": None if self.kda_k_cache is None else self.kda_k_cache[layer_offset],
            "kda_g_cache": None if self.kda_g_cache is None else self.kda_g_cache[layer_offset],
            "kda_v_cache": None if self.kda_v_cache is None else self.kda_v_cache[layer_offset],
            "kda_beta_cache": (
                None if self.kda_beta_cache is None else self.kda_beta_cache[layer_offset]
            ),
            "kda_state_tok": None
            if self.kda_state_tok is None
            else self.kda_state_tok[layer_offset],
        }

    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return KDAReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self._layer_cache_fields(layer_offset),
        )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None or slots.numel() == 0:
            return
        slots = _host_slots_to_device(host_slots, self.prev_num_accepted_tokens.device)
        self.prev_num_accepted_tokens.index_fill_(0, slots, 0)
        if self.rand_seed is not None:
            self._seed_request_counter += 1
            seeds = [
                compute_deterministic_mamba_seed(
                    self._seed_request_counter,
                    slot,
                    self._seed_rank_offset,
                )
                for slot in host_slots
            ]
            # Fresh pinned staging and a non-blocking upload, as in _host_slots_to_device: a
            # pageable torch.tensor(..., device=) would wait for the decode step in flight.
            seeds = torch.tensor(
                seeds,
                dtype=torch.int64,
                pin_memory=self.rand_seed.is_cuda and prefer_pinned(),
            )
            self.rand_seed.index_copy_(0, slots, seeds.to(self.rand_seed.device, non_blocking=True))

    def relocate_slots(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        if self.pool_layers is not None and self.rand_seed is None:
            return
        moves = [(old, new) for old, new in zip(old_slots, new_slots) if old >= 0 and old != new]
        if not moves:
            return
        device = self.prev_num_accepted_tokens.device
        source_slots = _host_slots_to_device([old for old, _ in moves], device)
        destination_slots = _host_slots_to_device([new for _, new in moves], device)
        if self.pool_layers is None:
            for replay_buffer in self._layer_replay_buffers():
                replay_buffer.index_copy_(
                    1,
                    destination_slots,
                    replay_buffer.index_select(1, source_slots),
                )
            self.prev_num_accepted_tokens.index_copy_(
                0,
                destination_slots,
                self.prev_num_accepted_tokens.index_select(0, source_slots),
            )
        if self.rand_seed is not None:
            self.rand_seed.index_copy_(
                0,
                destination_slots,
                self.rand_seed.index_select(0, source_slots),
            )

    def update(self, batch: MambaAcceptanceBatch) -> None:
        self.record_acceptance(
            batch.destination_state_indices,
            batch.num_accepted_tokens - 1,
            batch.is_dummy_request,
        )

    @torch.inference_mode()
    def seed_transferred_slots(self, state_indices: torch.Tensor) -> None:
        """Seed completed-transfer slots from borrowed persistent conv views."""
        if self.prev_num_accepted_tokens is None or state_indices.numel() == 0:
            return
        committed_window = self._committed_window
        section_offsets = [0]
        for section_dim in self._conv_section_dims:
            section_offsets.append(section_offsets[-1] + section_dim)

        self.prev_num_accepted_tokens.index_fill_(0, state_indices, 0)
        plan = self._get_batched_seed_plan()
        if plan is not None:
            kernels, plan = plan
            kernels.seed_kda_replay_layers(plan, state_indices)
            return
        if self.checkpoints:
            slots = state_indices.long()
            for layer, (ssm_states, conv_states) in enumerate(
                zip(self._ssm_states, self._conv_states)
            ):
                views = self.pool_layers[layer]
                seed_kda_checkpoints(
                    views["kda_ckpt"], views["kda_vconv"], ssm_states, conv_states, slots
                )
            return
        for layer_offset, conv_state in enumerate(self._conv_states):
            replay = self._layer_cache_fields(layer_offset)
            for name in (*_KDA_REPLAY_CONV_ROLES, *_KDA_REPLAY_DRAFT_ROLES):
                buffer = replay.get(name)
                if buffer is not None:
                    buffer.index_fill_(0, state_indices, 0)
            selected_conv = conv_state.index_select(0, state_indices)
            for section_idx, name in enumerate(_KDA_REPLAY_CONV_ROLES):
                buffer = replay[name]
                section = selected_conv[
                    :, section_offsets[section_idx] : section_offsets[section_idx + 1], :
                ]
                buffer[:, :, :committed_window].index_copy_(
                    0, state_indices, section.to(buffer.dtype)
                )

    def _get_batched_seed_plan(self) -> tuple[object, object] | None:
        """The K3 package's seed kernels and their plan over the bound views, if usable."""
        if self._batched_seed_plan is None:
            self._batched_seed_plan = self._plan_batched_seed() or False
        return self._batched_seed_plan or None

    def _plan_batched_seed(self) -> tuple[object, object] | None:
        """None: this state seeds transferred slots layer by layer; subclasses may plan a batched
        seed (of the replay caches, or in checkpoint mode of the checkpoints)."""
        return None

    def record_acceptance(
        self,
        state_indices: torch.Tensor,
        num_accepted_draft_tokens: torch.Tensor,
        is_dummy_request: torch.Tensor | None,
    ) -> None:
        """Record accepted drafts, preserving dummy rows; each input has shape [batch]."""
        if self.prev_num_accepted_tokens is None:
            raise RuntimeError("KDA replay acceptance requires replay bookkeeping")
        slots = state_indices.to(torch.long)
        accepted = num_accepted_draft_tokens.to(torch.int32).clamp(min=0)
        if is_dummy_request is not None:
            current = self.prev_num_accepted_tokens[slots]
            accepted = torch.where(is_dummy_request, current, accepted)
        self.prev_num_accepted_tokens[slots] = accepted

    @staticmethod
    @contextmanager
    def on_cache_stream(stream):
        """Order replay commits and seeding against V2 page migration."""
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    def commit_pending_drafts(self, slot: int) -> None:
        """Fold accepted drafts and the replay conv window into the persistent states."""
        from ...kernels.ssm.rubin_ull_kda_kernels import commit_kda_pending_drafts

        assert self.prev_num_accepted_tokens is not None
        slots = torch.full(
            (1,), slot, dtype=torch.int32, device=self.prev_num_accepted_tokens.device
        )
        if self.checkpoints:
            self.commit_checkpoints(slots.long())
            self.prev_num_accepted_tokens[slot].zero_()
            return
        num_accepted = self.prev_num_accepted_tokens.index_select(0, slots)
        for layer, (ssm_states, conv_states) in enumerate(zip(self._ssm_states, self._conv_states)):
            replay = self._layer_cache_fields(layer)
            commit_kda_pending_drafts(
                ssm_states,
                conv_states,
                replay["kda_conv_q"],
                replay["kda_conv_k"],
                replay["kda_conv_v"],
                replay.get("kda_k_cache"),
                replay.get("kda_g_cache"),
                replay.get("kda_v_cache"),
                replay.get("kda_beta_cache"),
                slots,
                num_accepted,
                state_tok=replay.get(_KDA_DRAFT_RECORDS_ROLE),
            )
        # zero_() on the element view stays stream-ordered; assigning a Python scalar
        # through the index stages it on the host and waits for the step in flight.
        self.prev_num_accepted_tokens[slot].zero_()

    def commit_checkpoints(self, slots: torch.Tensor) -> None:
        """Write the current checkpoint of ``slots`` (int64) in every layer to the live pools;
        the recorded acceptance is kept."""
        for layer, (ssm_states, conv_states) in enumerate(zip(self._ssm_states, self._conv_states)):
            views = self.pool_layers[layer]
            commit_kda_checkpoints(
                views["kda_ckpt"],
                views["kda_vconv"],
                ssm_states,
                conv_states,
                slots,
                self.prev_num_accepted_tokens,
            )

    def checkpoint_indices(
        self,
        state_indices: torch.Tensor,
        slots: torch.Tensor | None = None,
        accepted: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """vLLM's ``(spec_state_indices [B, 1 + num_spec], num_accepted_tokens [B])`` int32 for
        the flat kernel views (``KDAReplayLayerCache.kda_ckpt_kernel_caches``) of the requests in
        ``state_indices`` [B]. Device ops only (CUDA-graph safe); ``slots`` / ``accepted`` are
        written in place when given."""
        if not self.checkpoints or not self.checkpoint_slot_units:
            raise RuntimeError("KDA checkpoint indices need a bound checkpoint-mode state")
        base = state_indices.long()
        positions = torch.arange(1 + self.num_speculative_tokens, device=base.device)
        rows = self.checkpoint_header_units + base[:, None] * self.checkpoint_slot_units
        rows = (rows + positions * self.checkpoint_state_units).to(torch.int32)
        counts = (self.prev_num_accepted_tokens.index_select(0, base) + 1).to(torch.int32)
        if slots is None:
            slots = rows
        else:
            slots.copy_(rows)
        if accepted is None:
            accepted = counts
        else:
            accepted.copy_(counts)
        return slots, accepted

    def _layer_replay_buffers(self) -> tuple[torch.Tensor, ...]:
        buffers = (
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            self.kda_k_cache,
            self.kda_g_cache,
            self.kda_v_cache,
            self.kda_beta_cache,
            self.kda_state_tok,
        )
        if any(buffer is None for buffer in buffers[:3]):
            raise RuntimeError("KDA replay buffers are not bound")
        return tuple(buffer for buffer in buffers if buffer is not None)

    def shutdown(self) -> None:
        self.pool_layers = None
        self._ssm_states = ()
        self._conv_states = ()
        self.suspended_requests.clear()
        self.acceptance_pending.clear()
        self.unseeded_requests.clear()
        self.prev_num_accepted_tokens = None
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_k_cache = None
        self.kda_g_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None
        self.kda_state_tok = None
        self.intermediate_indices = None
        self.rand_seed = None
        self._batched_seed_plan = None


def _bind_checkpoint_views(
    state: KDAReplayState, manager: MambaHybridCacheManagerV2, raw: torch.Tensor
) -> dict[str, object]:
    layout = state._checkpoint_layout(manager._state_layout)
    views = kda_checkpoint_views(raw, layout)
    units = (views.slot_units, views.state_units, views.header_units)
    if state.checkpoint_slot_units and units != (
        state.checkpoint_slot_units,
        state.checkpoint_state_units,
        state.checkpoint_header_units,
    ):
        raise RuntimeError("KDA checkpoint buffers of the local layers differ in slot stride")
    state.checkpoint_layout = layout
    (
        state.checkpoint_slot_units,
        state.checkpoint_state_units,
        state.checkpoint_header_units,
    ) = units
    return {
        "kda_ckpt": views.ckpt,
        "kda_vconv": views.vconv,
        "kda_ckpt_kernel_caches": (views.conv_flat, views.state_flat),
    }


def bind_kda_replay_pool(state: KDAReplayState, manager: MambaHybridCacheManagerV2) -> None:
    """Bind model-owned KDA replay roles to views of the manager's native pools."""
    layout = state.pool_layout(manager._state_layout)
    if not layout:
        return
    required = {
        (manager.layer_offsets[layer_id], str(role))
        for layer_id in manager.mamba_pp_layers
        for role in (MambaRole.SSM_STATE, MambaRole.CONV_STATE, *layout)
    }
    required.add((manager.layer_offsets[manager.mamba_pp_layers[0]], str(_KDA_REPLAY_COUNT_ROLE)))
    if not any(
        required.issubset(
            {
                (int(buffer_id.layer_id), str(buffer_id.role))
                for coalesced in variant.coalesced_buffers
                for buffer_id in coalesced.buffer_ids
            }
        )
        for group in manager.impl.pool_group_descs
        for variant in group.slot_desc.variants
    ):
        raise RuntimeError(
            "KDA replay and recurrent-state buffers must share one KVCM "
            "pool group and lifecycle to reuse the same logical slot index"
        )

    state.pool_layers = []
    state._batched_seed_plan = None
    for layer_id in manager.mamba_pp_layers:
        layer = {}
        for name, (dtype, shape) in layout.items():
            view = manager._get_state_buffer(
                manager.layer_offsets[layer_id], DataRole(name), dtype, shape
            )
            if name == _KDA_CKPT_ROLE:
                if view.shape[0] != manager._state_layout.slot_capacity:
                    raise RuntimeError(f"KDA buffer {name} has a different slot capacity")
                layer.update(_bind_checkpoint_views(state, manager, view))
                continue
            if name in _KDA_REPLAY_CONV_ROLES:
                view = view.transpose(-1, -2)
            elif name == "kda_beta_cache":
                view = view[..., : manager.ssm_state_shape[0]]
            if view.shape[0] != manager._state_layout.slot_capacity:
                raise RuntimeError(f"KDA replay buffer {name} has a different slot capacity")
            layer[name] = view
        state.pool_layers.append(layer)
    state.prev_num_accepted_tokens = manager._get_state_buffer(
        manager.layer_offsets[manager.mamba_pp_layers[0]],
        _KDA_REPLAY_COUNT_ROLE,
        torch.int32,
        # The view's shape is the whole buffer: the manager derives the slot stride from it.
        [state.count_bytes // torch.int32.itemsize],
    )[..., 0]


def model_uses_kda_checkpoints(model_config) -> bool:
    """Whether the model asks for KDA checkpoint mode (``extra_attrs["kda_checkpoints"]``)."""
    extra_attrs = getattr(model_config, "extra_attrs", None)
    return bool(extra_attrs.get(KDA_CHECKPOINTS_ATTR, False)) if extra_attrs else False


#: ``ModelConfig.extra_attrs`` key a model sets to keep its KDA states in checkpoint mode.
KDA_CHECKPOINTS_ATTR = "kda_checkpoints"


def select_kda_replay_state(
    *,
    spec_config: object | None,
    manager_cls: type,
) -> KDAReplayState | None:
    """Apply KDA fused-verify and manager gates at the feature owner."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
        MixedMambaHybridCacheManager,
    )

    manager_supports_replay = issubclass(
        manager_cls,
        (MambaHybridCacheManagerV2, MixedMambaHybridCacheManager),
    )
    num_spec = get_kda_replay_num_spec(
        spec_config,
        manager_supports_replay=manager_supports_replay,
    )
    if num_spec is None:
        return None
    return KDAReplayState(num_spec)


def get_kda_replay_num_spec(
    spec_config: object | None, *, manager_supports_replay: bool
) -> int | None:
    """Return the fused-verify replay width when its complete gate passes."""
    if spec_config is None or not manager_supports_replay:
        return None
    from ...kernels.ssm.rubin_ull_kda_kernels import kda_mtp_replay_num_spec

    return kda_mtp_replay_num_spec(spec_config.tokens_per_gen_step - 1)


class KimiK3HybridCacheManager(MambaHybridCacheManagerV2):
    """The ULL target's own KDA/MLA replay and checkpoint lifecycle over the V2 pools."""

    # This target serves generation only (routing admits no other role for ull16), so its managers
    # never run context-only; the engine and the drafter read the flag.
    kda_context_only = False

    @override
    def __init__(
        self,
        *args,
        kda_replay_num_spec: int | None = None,
        kda_draft_records: bool = False,
        kda_checkpoints: bool = False,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        if use_replay_state_update:
            raise ValueError("KDA replay and Mamba2 replay are mutually exclusive")
        self._mamba_ssm_stochastic_rounding = kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self._requested_num_spec = kda_replay_num_spec
        self._requested_draft_records = kda_draft_records
        self._requested_checkpoints = kda_checkpoints
        self._kda_replay: KDAReplayState | None = None
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_spec_state(self) -> KDAIntermediateState | KDAReplayState | None:
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        if num_spec is not None:
            # Checkpoint mode is passed only when asked for, so the replay state is built exactly as
            # before for every other model.
            extra = {"checkpoints": True} if getattr(self, "_requested_checkpoints", False) else {}
            self._kda_replay = KDAReplayState(
                num_spec,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
                draft_records=self._requested_draft_records,
                **extra,
            )
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return KDAIntermediateState()

    @property
    def use_kda_replay_update(self) -> bool:
        return self._kda_replay is not None

    @property
    def uses_kda_checkpoints(self) -> bool:
        return self._kda_replay is not None and self._kda_replay.checkpoints

    @property
    def keeps_kda_draft_records(self) -> bool:
        return self.kda_state_tok is not None

    @property
    def kda_state_tok(self) -> list[torch.Tensor] | None:
        replay = self._kda_replay
        if replay is None or not replay.pool_layers:
            return None
        if _KDA_DRAFT_RECORDS_ROLE not in replay.pool_layers[0]:
            return None
        return [layer[_KDA_DRAFT_RECORDS_ROLE] for layer in replay.pool_layers]

    def _kda_replay_layout(self) -> dict[str, tuple[torch.dtype, list[int]]]:
        if self._kda_replay is None:
            return {}
        return self._kda_replay.pool_layout(self._state_layout)

    @override
    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        buffers = super()._get_recurrent_buffer_configs(global_layer_id)
        if self._kda_replay is not None:
            buffers.extend(
                self._kda_replay.pool_buffer_configs(self._state_layout, global_layer_id)
            )
        return buffers

    @override
    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        roles = super().get_disagg_ignored_roles()
        if self._kda_replay is not None:
            roles |= self._kda_replay.disagg_ignored_roles(self._state_layout)
        return roles

    @override
    def _setup_model_state(self) -> None:
        if self._kda_replay is not None:
            bind_kda_replay_pool(self._kda_replay, self)
        super()._setup_model_state()

    @override
    def _build_cache_config(self, config):
        if (
            self._kda_replay is not None
            and self.local_num_mamba_layers > 0
            and self.kv_cache_config.enable_kv_pool_rebalance
        ):
            raise ValueError(
                "KDA replay views are bound once to the SSM pool's slot count, "
                "and a rebalance can grow that pool past them: set "
                "kv_cache_config.enable_kv_pool_rebalance=False."
            )
        return super()._build_cache_config(config)

    @override
    def suspend_request(self, request) -> None:
        replay = self._kda_replay
        request_id = request.py_request_id
        cache = self.kv_cache_map.get(request_id)
        slot = self._request_id_to_state_index.get(request_id, -1)
        if replay is not None and slot >= 0 and cache is not None and cache.is_active:
            if request_id in replay.acceptance_pending:
                raise RuntimeError(
                    f"Request {request_id} is being suspended before update_resources "
                    "recorded the acceptance of its last verify, so its pending KDA drafts are unknown"
                )
            if request_id in replay.unseeded_requests:
                replay.unseeded_requests.remove(request_id)
            else:
                with replay.on_cache_stream(self._stream):
                    replay.commit_pending_drafts(slot)
            del self._request_id_to_state_index[request_id]
            replay.suspended_requests.add(request_id)
        super().suspend_request(request)

    @override
    def _setup_state_indices(self, requests, num_contexts: int = 0) -> None:
        super()._setup_state_indices(requests, num_contexts)
        replay = self._kda_replay
        if replay is None or self.local_num_mamba_layers == 0:
            return
        replay.suspended_requests.difference_update(
            request.py_request_id for request in requests[:num_contexts]
        )
        slots = []
        for request in requests[num_contexts:]:
            if request.py_request_id in replay.suspended_requests:
                replay.suspended_requests.remove(request.py_request_id)
                slots.append(self._request_id_to_state_index[request.py_request_id])
        if slots:
            with replay.on_cache_stream(self._stream):
                replay.seed_transferred_slots(
                    _host_slots_to_device(slots, replay.prev_num_accepted_tokens.device)
                )

    @override
    def prepare_resources(self, scheduled_batch) -> None:
        super().prepare_resources(scheduled_batch)
        replay = self._kda_replay
        if replay is None or self.local_num_mamba_layers == 0:
            return
        replay.unseeded_requests.update(
            request.py_request_id
            for request in scheduled_batch.context_requests
            if request.is_disagg_generation_init_state
        )
        if getattr(self.spec_config, "decoding_type", None) == "NGram":
            replay.acceptance_pending.update(
                request.py_request_id
                for request in scheduled_batch.generation_requests
                if not request.is_dummy
            )

    @override
    def revert_allocate_generation(self, request) -> None:
        if self._kda_replay is not None:
            self._kda_replay.acceptance_pending.discard(request.py_request_id)
        super().revert_allocate_generation(request)

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        self._commit_checkpoints_for_reuse(request)
        if self._kda_replay is not None:
            self._kda_replay.suspended_requests.discard(request.py_request_id)
            self._kda_replay.acceptance_pending.discard(request.py_request_id)
            self._kda_replay.unseeded_requests.discard(request.py_request_id)
        super().free_resources(request, pin_on_release)

    def _commit_checkpoints_for_reuse(self, request) -> None:
        """With block reuse, the released slot's state is offered to later requests: commit its
        current checkpoint to the live pools first (the base release commits the blocks)."""
        replay = self._kda_replay
        if (
            replay is None
            or not replay.checkpoints
            or not self.kv_cache_config.enable_block_reuse
            or self.local_num_mamba_layers == 0
            or request.py_request_id in replay.unseeded_requests
            or request.py_request_id in replay.suspended_requests
        ):
            return
        cache = self.kv_cache_map.get(request.py_request_id)
        slot = self._request_id_to_state_index.get(request.py_request_id, -1)
        if cache is None or not cache.is_active or slot < 0:
            return
        slots = _host_slots_to_device([slot], replay.prev_num_accepted_tokens.device)
        with replay.on_cache_stream(self._stream):
            replay.commit_checkpoints(slots)

    @override
    def _extra_scratch_bytes_per_slot(self) -> int:
        if self._kda_replay is None or not self._kda_replay._stochastic_rounding:
            return 0
        return torch.int64.itemsize

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)

    @override
    def _on_state_slots_relocated(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self._kda_replay is None:
            return
        self._kda_replay.relocate_slots(old_slots, new_slots)
        fresh_slots = [new for old, new in zip(old_slots, new_slots) if old < 0]
        if fresh_slots:
            slots = _host_slots_to_device(fresh_slots, self.cuda_state_indices.device)
            self._kda_replay.reset_slots(slots, fresh_slots)

    @override
    def update_mamba_states(
        self, attn_metadata, num_accepted_tokens, state_indices=None, accepted_leaf_positions=None
    ) -> None:
        super().update_mamba_states(
            attn_metadata, num_accepted_tokens, state_indices, accepted_leaf_positions
        )

    @override
    def update_resources(
        self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None
    ) -> None:
        super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)
        if (
            self.local_num_mamba_layers
            and self._kda_replay is not None
            and getattr(self.spec_config, "decoding_type", None) == "NGram"
        ):
            self._record_replay_request_acceptance(scheduled_batch)

    def _record_replay_request_acceptance(self, scheduled_batch: object) -> None:
        replay = self._kda_replay
        if replay.prev_num_accepted_tokens is None:
            return
        generation_requests = scheduled_batch.generation_requests
        replay.acceptance_pending.difference_update(
            request.py_request_id for request in generation_requests
        )
        drafted_requests = [
            request
            for request in generation_requests
            if request.py_draft_tokens is not None and len(request.py_draft_tokens) > 0
        ]
        if not drafted_requests:
            return
        if len(drafted_requests) != len(generation_requests):
            raise RuntimeError(
                "Mixed drafted/undrafted generation batch is not supported "
                "for KDA replay bookkeeping"
            )
        state_index_map = self._request_id_to_state_index
        dummy_map = self._request_id_to_is_dummy
        active_requests = [
            request for request in generation_requests if request.py_request_id in state_index_map
        ]
        if not active_requests:
            return
        device = replay.prev_num_accepted_tokens.device
        state_indices = torch.tensor(
            [state_index_map[request.py_request_id] for request in active_requests],
            dtype=torch.int32,
            device=device,
        )
        accepted_drafts = torch.tensor(
            [request.py_num_accepted_draft_tokens for request in active_requests],
            dtype=torch.int32,
            device=device,
        )
        is_dummy_request = torch.tensor(
            [dummy_map.get(request.py_request_id, False) for request in active_requests],
            dtype=torch.bool,
            device=device,
        )
        replay.record_acceptance(state_indices, accepted_drafts, is_dummy_request)

    @override
    def on_state_transfer_complete(self, request_ids: list[int]) -> None:
        replay = self._kda_replay
        if replay is None or replay.prev_num_accepted_tokens is None:
            return
        replay.unseeded_requests.difference_update(request_ids)
        slots = sorted(
            {
                self._request_id_to_state_index[request_id]
                for request_id in request_ids
                if request_id in self._request_id_to_state_index
            }
        )
        if slots:
            replay.seed_transferred_slots(
                _host_slots_to_device(slots, replay.prev_num_accepted_tokens.device)
            )

    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: list[int]) -> None:
        """Compatibility entry point; the executor uses the generic transfer hook."""
        self.on_state_transfer_complete(request_ids)


def resolve_kimi_ssm_cache_dtype(config) -> torch.dtype:
    """Default KDA state to FP32; BF16 requires an explicit cache configuration.

    BF16 prefill, decode and sequential verify stage state through FP32 and
    round the committed state back to BF16. Fused MTP verify reads and writes
    BF16 state directly, rounding the committed state to nearest.
    Checkpoint-declared state dtypes do not override this default.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import resolve_ssm_cache_dtype

    declared = resolve_ssm_cache_dtype(config)
    if declared is not None and declared != torch.float32:
        logger.info(
            "Kimi K3: the checkpoint declares "
            f"mamba_ssm_cache_dtype={declared}; keeping the fp32 "
            "recurrent-state pool (kv_cache_config.mamba_ssm_cache_dtype "
            "opts in to bfloat16)."
        )
    return torch.float32


def validate_kimi_state_cache_dtype(
    mamba_ssm_cache_dtype: torch.dtype, mamba_ssm_stochastic_rounding: bool = False
) -> None:
    """Reject cache dtypes and rounding modes unsupported by the KDA kernels."""
    if mamba_ssm_cache_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Kimi K3 KDA recurrent-state cache supports float32 (default) or "
            f"bfloat16; got mamba_ssm_cache_dtype={mamba_ssm_cache_dtype}."
        )
    if mamba_ssm_stochastic_rounding and mamba_ssm_cache_dtype != torch.float32:
        raise ValueError(
            "Kimi K3 KDA kernels round the committed recurrent state to "
            "nearest; mamba_ssm_stochastic_rounding is not supported with a "
            f"{mamba_ssm_cache_dtype} state cache."
        )


def get_kimi_cache_params(config, *, spec_config=None, quant_config=None):
    """KDA uses equal Q/K/V sections and defaults to FP32 state alongside MLA."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        build_mamba_kv_cache_params,
        get_kimi_linear_layer_masks,
        unwrap_kimi_text_config,
    )

    linear = unwrap_kimi_text_config(config).linear_attn_config
    attention, recurrent = get_kimi_linear_layer_masks(config)
    params = build_mamba_kv_cache_params(
        config,
        state_size=linear["head_dim"],
        conv_kernel=linear["short_conv_kernel_size"],
        num_heads=linear["num_heads"],
        n_groups=linear["num_heads"],
        head_dim=linear["head_dim"],
        mamba_mask=recurrent,
        target_full_attn_mask=attention,
        spec_config=spec_config,
        quant_config=quant_config,
    )
    return params


__all__ = [
    "KDAIntermediateLayerCache",
    "KDAIntermediateState",
    "KDAReplayState",
    "KDAReplayLayerCache",
    "KDA_CHECKPOINTS_ATTR",
    "KDACheckpointLayout",
    "KimiK3HybridCacheManager",
    "allocate_kda_replay_fields",
    "commit_kda_checkpoints",
    "kda_checkpoint_views",
    "model_uses_kda_checkpoints",
    "seed_kda_checkpoints",
    "bind_kda_replay_pool",
    "get_kda_replay_num_spec",
    "get_kimi_cache_params",
    "resolve_kimi_ssm_cache_dtype",
    "select_kda_replay_state",
    "validate_kimi_state_cache_dtype",
]


# ---- the KDA + DSA cache manager ----------------------------------------
if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


# Replay buffers a transferred slot is seeded or cleared in, as the shared per-layer loop does.
_SEEDED_ROLES = (
    *_kda_cache_manager._KDA_REPLAY_CONV_ROLES,
    *_kda_cache_manager._KDA_REPLAY_DRAFT_ROLES,
)


@dataclass(frozen=True)
class _StackedReplaySeed:
    """``[layers, slots, ...]`` views of the local KDA layers' conv states and seeded replay
    buffers."""

    conv: torch.Tensor
    replay: dict[str, torch.Tensor]
    # Bounds of the [Q | K | V] conv-state sections: 0, D, 2D, 3D.
    section_bounds: tuple[int, ...]
    committed_window: int


def _seed_stacked_layers(plan: _StackedReplaySeed, state_indices: torch.Tensor) -> None:
    """Write what the shared per-layer loop writes, in all layers at once: one launch per replay
    buffer kind and conv section, whatever the layer count."""
    for buffer in plan.replay.values():
        buffer.index_fill_(1, state_indices, 0)
    selected_conv = plan.conv.index_select(1, state_indices)
    bounds = plan.section_bounds
    for section, name in enumerate(_kda_cache_manager._KDA_REPLAY_CONV_ROLES):
        buffer = plan.replay[name]
        buffer[..., : plan.committed_window].index_copy_(
            1,
            state_indices,
            selected_conv[:, :, bounds[section] : bounds[section + 1], :].to(buffer.dtype),
        )


# The shared seed_transferred_slots hands a plan to seed_kda_replay_layers of the object the
# planner returns with it.
_STACKED_SEEDER = SimpleNamespace(seed_kda_replay_layers=_seed_stacked_layers)


@dataclass(frozen=True)
class _StackedCheckpointSeed:
    """``[layers, slots, ...]`` views of the local KDA layers' live pools and of what a checkpoint
    seed writes: checkpoint 0 and the committed window of the conv page."""

    ssm: torch.Tensor
    conv: torch.Tensor
    ckpt0: torch.Tensor
    vconv_window: torch.Tensor
    # Whether the one-launch seed covers these views (seed_checkpoints_applies).
    one_launch: bool = False


def _seed_stacked_checkpoints(plan: _StackedCheckpointSeed, state_indices: torch.Tensor) -> None:
    """Write what ``seed_kda_checkpoints`` writes layer by layer, in all layers at once: one launch
    (``kernels/ssm/rubin_ull_kda_checkpoint_seed.py``) where its views apply, else one gather and
    one scatter per pool kind, whatever the layer count."""
    slots = state_indices.long()
    if plan.one_launch:
        seed_checkpoints(plan.ssm, plan.ckpt0, plan.conv, plan.vconv_window, slots)
        return
    plan.ckpt0.index_copy_(1, slots, plan.ssm.index_select(1, slots).to(plan.ckpt0.dtype))
    plan.vconv_window.index_copy_(
        1, slots, plan.conv.index_select(1, slots).to(plan.vconv_window.dtype)
    )


_STACKED_CHECKPOINT_SEEDER = SimpleNamespace(seed_kda_replay_layers=_seed_stacked_checkpoints)


class KimiK31KDAReplayState(KDAReplayState):
    """The shared KDA fused-verify replay state, seeding transferred slots from stacked views.

    The shared state seeds the replay caches of transferred requests in all layers at once with
    the Kimi K3 package's seed kernels. Kimi K3.1 instead stacks each kind of per-layer pool view
    into one ``[layers, slots, ...]`` view, as the V2 pools space a kind's per-layer views evenly,
    and seeds all local KDA layers with one plain copy per buffer kind and conv section. In
    checkpoint mode it stacks the live pools, checkpoint 0 and the conv pages' committed windows
    the same way and seeds them with one gather and one scatter per pool kind. Views that do not
    stack (CPU tensors, unevenly spaced layers) keep the shared state's per-layer loop of the same
    copies.
    """

    @override
    def _plan_batched_seed(
        self,
    ) -> tuple[SimpleNamespace, _StackedReplaySeed | _StackedCheckpointSeed] | None:
        """Stacked views of every seeded buffer kind, or None when one of them does not stack.

        Reads only the views' pointers, shapes and strides. The shared ``_get_batched_seed_plan``
        keeps the result until the views are bound again or released.
        """
        if self.checkpoints:
            return self._plan_checkpoint_seed()
        conv_states = list(self._conv_states)
        stacked_conv = _stack_state_views(conv_states)
        stacked_replay = {}
        if stacked_conv is not None:
            layers = [self._layer_cache_fields(layer) for layer in range(len(conv_states))]
            for name in _SEEDED_ROLES:
                if layers[0].get(name) is not None:
                    stacked_replay[name] = _stack_state_views([layer[name] for layer in layers])
        if stacked_conv is None or any(view is None for view in stacked_replay.values()):
            if conv_states and conv_states[0].is_cuda:
                logger.warning_once(
                    f"The KDA conv states or replay buffers of the {len(conv_states)} local KDA "
                    "layers are not evenly spaced in their pools; transferred requests are seeded "
                    "layer by layer",
                    key="k31_ull_kda_replay_seed_per_layer",
                )
            return None
        logger.info_once(
            f"KDA replay caches of transferred requests are seeded in all {len(conv_states)} "
            "local KDA layers at once",
            key="k31_ull_kda_replay_seed_stacked",
        )
        section_bounds = tuple(itertools.accumulate(self._conv_section_dims, initial=0))
        return _STACKED_SEEDER, _StackedReplaySeed(
            stacked_conv, stacked_replay, section_bounds, self._committed_window
        )

    def _plan_checkpoint_seed(self) -> tuple[SimpleNamespace, _StackedCheckpointSeed] | None:
        """Stacked views of the live pools, checkpoint 0 and the conv pages' committed windows, or
        None when one of them does not stack."""
        ssm_states, conv_states = list(self._ssm_states), list(self._conv_states)
        layers = self.pool_layers or []
        stacked = None
        if conv_states and len(layers) == len(conv_states) == len(ssm_states):
            window = conv_states[0].shape[-1]
            stacked = (
                _stack_state_views(ssm_states),
                _stack_state_views(conv_states),
                _stack_state_views([layer["kda_ckpt"][:, 0] for layer in layers]),
                _stack_state_views([layer["kda_vconv"][..., :window] for layer in layers]),
            )
        if stacked is None or any(view is None for view in stacked):
            if conv_states and conv_states[0].is_cuda:
                logger.warning_once(
                    f"The KDA pools or checkpoints of the {len(conv_states)} local KDA layers are "
                    "not evenly spaced in their pools; transferred requests are seeded layer by "
                    "layer",
                    key="k31_ull_kda_checkpoint_seed_per_layer",
                )
            return None
        ssm, conv, ckpt0, vconv_window = stacked
        one_launch = seed_checkpoints_applies(ssm, ckpt0, conv, vconv_window)
        logger.info_once(
            f"KDA checkpoints of transferred requests are seeded in all {len(conv_states)} local "
            f"KDA layers at once ({'one launch' if one_launch else 'torch copies'})",
            key="k31_ull_kda_checkpoint_seed_stacked",
        )
        return _STACKED_CHECKPOINT_SEEDER, _StackedCheckpointSeed(
            ssm, conv, ckpt0, vconv_window, one_launch=one_launch
        )


class KimiK3DSAHybridCacheManager(DraftWindowCacheMixin, KimiK3HybridCacheManager):
    """Kimi K3 KDA/MLA cache plus a DSA indexer K-cache (Kimi K3.1).

    Kimi K3.1 adds a DeepSeek sparse-attention indexer to its MLA layers. The
    indexer keys live in one extra ``Role.INDEX_KEY`` buffer per indexer layer,
    so they share allocation, reuse and release with the latent KV pages. The
    DSA accessors mirror :class:`DSACacheManagerV2`, except that local layer
    indices also cover the KDA layers, which own neither a K page nor an
    indexer buffer.

    A unified standalone DFlash2 drafter keeps only its sliding window of its
    history (the checkpoint's ``sliding_window``, ``set_draft_window``) unless
    ``kimi_k31_draft_window`` turns that off, as the Rubin context targets'
    manager does; a context server and this generation server then hand over
    histories of one window. The drafter numbers its history by position and
    reads only that window, so the stored window holds every position it reads.

    With an NVFP4 cache, the FP8 KV layers of ``get_kimi_k31_fp8_kv_layer_ids``
    (by default the last two MLA layers) keep their latent KV as E4M3 pages.
    They keep their indexer buffer and the DSA lifecycle, so they share slot
    ids with the NVFP4 layers, while V2 puts their larger K pages in a second
    physical pool of the DSA layer group. The page tables stay one per
    lifecycle: one virtual attention-op pool, after every lifecycle pool,
    addresses that pool from the DSA slot ids.
    """

    # Read by ``is_dsa_cache_manager``: the DSA backend and metadata may use
    # this manager's indexer K-cache and primary-pool page mapping.
    provides_dsa_indexer_k_cache = True

    @classmethod
    def extra_init_kwargs(cls, model_config) -> dict:
        """Constructor arguments the engine adds for this manager (``_create_kv_cache_manager``):
        the DSA config, the text config and the ``kimi_k31_draft_window`` option (default on)."""
        pretrained_config = model_config.pretrained_config
        return {
            "sparse_attention_config": model_config.sparse_attention_config,
            "kda_checkpoints": bool((model_config.extra_attrs or {}).get("kda_checkpoints", False)),
            "pretrained_config": getattr(pretrained_config, "text_config", None)
            or pretrained_config,
            "draft_window": draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True),
        }

    @property
    def supports_mixed_latent_kv(self) -> bool:
        """Whether this rank stores some MLA layers' latent KV as FP8 pages beside NVFP4 ones.

        The DSA backend reads it: for such a manager it checks once per layer that the
        layer's KV quant mode matches ``get_kv_cache_dtype(layer)``.
        """
        return bool(getattr(self, "_fp8_latent_local_layers", None))

    @override
    def __init__(
        self,
        *args,
        sparse_attention_config: "SparseAttentionConfig",
        pretrained_config=None,
        draft_window: bool = False,
        **kwargs,
    ) -> None:
        """Build the hybrid manager with an indexer buffer per DSA layer.

        Args:
            sparse_attention_config: DSA sparse-attention config of the model.
            pretrained_config: Text config; resolves indexer sharing per layer.
            draft_window: Keep only the DFlash2 drafter's sliding window of its history in a
                unified draft cache (``set_draft_window``); the engine passes the
                ``kimi_k31_draft_window`` option (default on).
            **kwargs: ``KimiK3HybridCacheManager`` arguments; ``layer_mask``
                marks the MLA layers.
        """
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for the Kimi DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("The Kimi DSA cache requires DSA sparse parameters")
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is None:
            raise ValueError("The Kimi DSA cache needs layer_mask to place the indexer buffers")
        if draft_window:
            set_draft_window(kwargs)

        # Read by _extra_buffers_per_layer during the base construction.
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        self.use_fp4 = self.indexer_k_dtype == "fp4"
        self.use_fp8_ds_mla = False
        self.mla_kv_cache_residual_dim = 0
        self._fp8_mla_global_layer_ids = get_kimi_k31_fp8_kv_layer_ids(
            pretrained_config, nvfp4_enabled=kwargs.get("dtype") == DataType.NVFP4
        )
        if self._fp8_mla_global_layer_ids:
            if kwargs.get("dtype") != DataType.NVFP4:
                raise ValueError(
                    "Kimi K3.1 features: fp8_kv_layers requires the NVFP4 KV cache storage."
                )
            logger.info(
                "Kimi K3 DSA layers using FP8 KV-cache fallback: "
                f"{sorted(self._fp8_mla_global_layer_ids)}."
            )
        self._primary_pools: dict[int, torch.Tensor] = {}
        # get_primary_pool by local layer, so a per-layer call skips the binding lookups.
        self._primary_pool_by_layer: dict[int, torch.Tensor] = {}
        # Set by _prepare_page_table_tensor when this rank holds FP8 latent layers.
        self._fp8_latent_op_pool: Optional[int] = None
        self._fp8_latent_source_pool: Optional[int] = None
        self._fp8_latent_index_scales: Optional[torch.Tensor] = None
        self._fp8_latent_kv_offsets: Optional[torch.Tensor] = None
        self._fp8_latent_materializer: Optional[_BasePageTableMaterializer] = None
        full_indexer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, len(layer_mask)
        )
        self._indexer_layer_ids = frozenset(
            layer_idx
            for layer_idx, (is_attention, has_indexer) in enumerate(
                zip(layer_mask, full_indexer_mask)
            )
            if is_attention and has_indexer
        )
        self._unique_primary_pool: Optional[torch.Tensor] = None

        super().__init__(*args, **kwargs)

        # A unified standalone drafter (DSpark) registers its layers as model layers; they own
        # neither a coalesced MLA K page nor an indexer buffer, so the DSA views skip them.
        self._attention_local_layers = [
            local_layer_idx
            for local_layer_idx in range(self.num_local_layers)
            if not self._is_standalone_draft_layer(local_layer_idx)
            and not self._is_local_mamba_layer(local_layer_idx)
        ]
        if not self._attention_local_layers:
            raise ValueError("DSA requires at least one local MLA layer")
        self._primary_pool_page_index_params = {}
        for local_layer_idx in self._attention_local_layers:
            converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            if self._get_layer_dtype_by_local_id(local_layer_idx) == DataType.NVFP4:
                scale_converter = self.impl.get_page_index_converter(
                    local_layer_idx, Role.KEY_BLOCK_SCALE
                )
                if (converter.scale, converter.layer_offset, converter.expansion) != (
                    scale_converter.scale,
                    scale_converter.layer_offset,
                    scale_converter.expansion,
                ):
                    raise ValueError("NVFP4 DSA data and scale pages must use the same indices.")
            self._primary_pool_page_index_params[local_layer_idx] = (
                int(converter.scale),
                int(converter.layer_offset),
            )
        indexer_local_layers = [
            local_layer_idx
            for local_layer_idx, has_indexer in enumerate(self.indexer_k_cache_local_layer_mask)
            if has_indexer
        ]
        indexer_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.INDEX_KEY)
            for local_layer_idx in indexer_local_layers
        ]
        if any(int(converter.expansion) != 1 for converter in indexer_converters):
            raise ValueError("DSA indexer K-cache does not support page expansion")
        # Each cache view starts at its SHARED base address, so the converter's
        # layer offset is already represented by the base pointer.
        self.indexer_k_cache_page_scale = (
            int(indexer_converters[0].scale) if indexer_converters else 1
        )
        if any(
            int(converter.scale) != self.indexer_k_cache_page_scale
            for converter in indexer_converters[1:]
        ):
            raise ValueError("DSA requires a uniform shared INDEX_KEY page mapping across layers")
        self._attention_pool_id = self.layer_to_pool_mapping_dict[self._attention_local_layers[0]]
        if any(
            self.layer_to_pool_mapping_dict[layer] != self._attention_pool_id
            for layer in self._attention_local_layers
        ):
            raise ValueError("Kimi DSA MLA layers must share one request-slot lifecycle.")
        self._num_attention_slots = self._count_attention_slots()
        self.indexer_k_cache_pool_per_layer = {
            local_layer_idx: self._get_indexer_k_cache_pool_data(local_layer_idx)
            for local_layer_idx in indexer_local_layers
        }
        # The MLA layers whose latent KV lives in the manager-dtype pool, and the FP8 ones.
        self._fp8_latent_local_layers = self._local_fp8_latent_layers()
        fp8_latent_local_layers = frozenset(self._fp8_latent_local_layers)
        self._nvfp4_latent_local_layers = [
            local_layer_idx
            for local_layer_idx in self._attention_local_layers
            if local_layer_idx not in fp8_latent_local_layers
        ]
        if self._fp8_latent_local_layers:
            self._validate_mixed_latent_layout()
        # Startup logs show which cache manager serves the model.
        logger.info_once(
            f"Kimi K3.1 cache manager: {type(self).__name__} (DSA indexer K-cache on "
            f"{len(indexer_local_layers)} of {len(self._attention_local_layers)} local MLA layers, "
            f"indexer_k_dtype={self.indexer_k_dtype})",
            key="k31_ull_dsa_cache_manager",
        )
        self._log_kv_cache_pools()

    def _pool_group_kinds(self) -> dict[int, str]:
        """What each GPU pool group holds, by pool-group index: ``drafter`` (only the unified
        drafter's history layers), ``main`` (the target's MLA latent KV and index keys) or
        ``state`` (the KDA recurrent state)."""
        attention_layers = frozenset(self._attention_local_layers)
        kinds = {}
        for pool_group in self.impl.pool_group_descs:
            layers = {
                int(buffer_id.layer_id)
                for variant in pool_group.slot_desc.variants
                for coalesced in variant.coalesced_buffers
                for buffer_id in coalesced.buffer_ids
            }
            if layers and all(self._is_standalone_draft_layer(layer) for layer in layers):
                kind = "drafter"
            elif layers & attention_layers:
                kind = "main"
            else:
                kind = "state"
            kinds[int(pool_group.pool_group_index)] = kind
        return kinds

    def _log_kv_cache_pools(self) -> None:
        """Log this rank's GPU pool groups on one line, once, when the cache that serves is built.

        The line has the format of the Rubin manager's ``Kimi K3.1 KV cache pools`` line (the slots,
        slot bytes and bytes of each pool group, in pool-group order; a slot of the paged pools is
        one page) with what the group holds appended to its segment (``pool=main``, ``drafter`` or
        ``state``). A cache built for the cache size estimate (``is_estimating_kv_cache``) logs
        nothing.
        """
        if self.is_estimating_kv_cache:
            return
        try:
            stats = self._get_storage_statistics(GPU_LEVEL)
            kinds = self._pool_group_kinds()
        except (AttributeError, RuntimeError) as error:
            # A log line only: a cache that cannot report its pools still serves.
            logger.warning(f"Kimi K3.1 KV cache pools unavailable: {error}")
            return
        entries = []
        for pool_group_id, stat in enumerate(stats):
            slot_sizes = [int(size) for size in self._stats_slot_sizes(stat)]
            num_slots = int(stat.total)
            entries.append(
                f"pool_group_id={pool_group_id}, num_slots={num_slots}, slot_size={slot_sizes}, "
                f"bytes={num_slots * sum(slot_sizes)}, pool={kinds.get(pool_group_id, 'unknown')}"
            )
        logger.info(
            f"Kimi K3.1 KV cache pools (ULL generation, rank {self.mapping.rank}): "
            f"{'; '.join(entries)}"
        )

    @override
    def _initialize_spec_state(self) -> KDAIntermediateState | KDAReplayState | None:
        """The shared manager's speculative state, with the K3.1 replay state."""
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        if num_spec is not None:
            self._kda_replay = KimiK31KDAReplayState(
                num_spec,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
                draft_records=self._requested_draft_records,
                checkpoints=self._requested_checkpoints,
            )
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return KDAIntermediateState()

    @override
    def _setup_model_state(self) -> None:
        super()._setup_model_state()
        if self._kda_replay is not None:
            # Plan the seeding once the views are bound, so that the startup log tells whether
            # transferred requests are seeded in all layers at once or layer by layer.
            self._kda_replay._get_batched_seed_plan()

    @override
    def _align_fp4_mla_snapshot_points(self, request) -> None:
        """Round the recurrent snapshot points of an NVFP4 cache down to whole cache blocks.

        A reused context resumes at a recurrent-state snapshot. On a block boundary it resumes on
        whole NVFP4 latent pages, with their block scales and indexer keys, and never copies part
        of a page. Points inside the first block are dropped. The shared manager rounds only for a
        cache with an FP4 MLA layout policy, which this manager does not have.
        """
        if self.dtype != DataType.NVFP4:
            super()._align_fp4_mla_snapshot_points(request)
            return
        request.expect_snapshot_points = sorted(
            {
                point // self.tokens_per_block * self.tokens_per_block
                for point in request.expect_snapshot_points
                if point >= self.tokens_per_block
            }
        )

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        """With an NVFP4 cache, move each context request's history to its context position.

        The history of an NVFP4 cache follows every completed context chunk, also while a later
        snapshot point is pending; the shared manager does so only for a cache with an FP4 MLA
        layout policy. The snapshot points sit on block boundaries (above), so a commit at one of
        them ends at the history.

        The shared method then runs for every request. For the requests handled here it must do
        nothing more: their history already stands at the context position.
        """
        if self.dtype == DataType.NVFP4:
            for request in scheduled_batch.context_requests:
                kv_cache = self.kv_cache_map.get(request.py_request_id)
                if kv_cache is None or not kv_cache.is_active:
                    continue
                position = request.context_current_position
                if kv_cache.history_length < position and not kv_cache.resize(None, position):
                    raise ValueError(
                        "Failed to resize history length of V2 Mamba cache for request "
                        f"{request.py_request_id} to {position} tokens at context update"
                    )
        super().update_context_resources(scheduled_batch)

    def _is_fp8_latent_layer(self, local_layer_idx: int) -> bool:
        """Whether a local layer stores its latent KV as FP8 pages.

        Standalone draft layers keep their own dtype: their global ids follow this
        rank's last layer, so on an earlier pipeline stage they can repeat the id of
        a selected model layer.
        """
        return (
            bool(self._fp8_mla_global_layer_ids)
            and local_layer_idx < len(self.pp_layers)
            and local_layer_idx not in self.cp_as_tp_layer_ids
            and self.pp_layers[local_layer_idx] in self._fp8_mla_global_layer_ids
        )

    def _local_fp8_latent_layers(self) -> list[int]:
        """This rank's FP8 latent layers as local indices, in local order.

        Read from ``pp_layers``, so the base construction may call it before
        ``_attention_local_layers`` exists.
        """
        if not self._fp8_mla_global_layer_ids:
            return []
        return [
            local_layer_idx
            for local_layer_idx in range(len(self.pp_layers))
            if self._is_fp8_latent_layer(local_layer_idx)
        ]

    @override
    def _kv_pool_mapping_offset(
        self, layer_id: LayerId, layer_group_id: int, key_base_addr: int
    ) -> int:
        # An FP8 latent page lies in another physical pool than the NVFP4 pages that the
        # lifecycle's pointer row addresses. _prepare_page_table_tensor maps the layer to
        # the virtual attention-op pool instead; this 0 is a placeholder.
        if self._is_fp8_latent_layer(int(layer_id)):
            return 0
        return super()._kv_pool_mapping_offset(layer_id, layer_group_id, key_base_addr)

    @override
    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        """Keep one page table per lifecycle and add an attention-op pool for FP8 latent pages.

        The FP8 and NVFP4 K pages of the DSA layer group lie in two physical pools with
        different page strides, for which the base would give every local layer a page
        table of its own (``num_attention_op_pools`` = local layers): a host table that
        is expanded into one plane per layer every step, and planes whose ids no longer
        are the layer group ids the DSpark drafter decodes its block table from. Instead
        the DSA lifecycle's pointer row addresses the NVFP4 pool, and one virtual
        attention-op pool addresses the FP8 pool from the same slot ids, which
        :meth:`copy_batch_block_offsets` scales by the FP8 page scale. The virtual pool
        comes after every lifecycle pool, and ``index_scales``, ``kv_offset`` and the host
        block offsets keep one entry per lifecycle, so every other layer keeps its plane.
        """
        fp8_local_layers = self._local_fp8_latent_layers()
        if not fp8_local_layers:
            super()._prepare_page_table_tensor(index_mapper_capacity)
            return
        if self.enable_swa_scratch_reuse:
            raise NotImplementedError(
                "FP8 KV layers of the Kimi DSA cache do not support SWA scratch reuse."
            )
        fp8_layers = frozenset(fp8_local_layers)
        for pool_id, layer_ids in enumerate(self.impl.layer_grouping):
            role = self._get_pool_roles(pool_id)[0]
            page_strides = {
                self.impl.get_page_stride(layer_id, role)
                for layer_id in layer_ids
                if int(layer_id) not in fp8_layers
            }
            if len(page_strides) > 1:
                raise NotImplementedError(
                    f"Layer group {pool_id} holds {role} pages of strides {sorted(page_strides)} "
                    "besides the FP8 latent pages; one page table per lifecycle cannot "
                    "address them."
                )
        dsa_pool_id = int(self.impl.get_layer_group_id(LayerId(fp8_local_layers[0])))
        nvfp4_layers = [
            int(layer_id)
            for layer_id in self.impl.layer_grouping[dsa_pool_id]
            if int(layer_id) not in fp8_layers
        ]
        if not nvfp4_layers:
            raise NotImplementedError(
                "FP8 KV layers of the Kimi DSA cache need an NVFP4 DSA layer on every rank."
            )
        # The DSA lifecycle's row addresses the NVFP4 pool. Pin its representative to the
        # first NVFP4 layer: the descriptor order is not part of the V2 contract.
        self._pool_layer_ids_by_role[(dsa_pool_id, Role.KEY)] = LayerId(min(nvfp4_layers))
        self._use_per_layer_page_tables = False
        # The NVFP4 layers make the rows [primary, secondary] x [data, block scale].
        pool_pointers, pool_mapping = self._build_pool_mapping_tensors()
        op_pool_id = self.num_pools
        fp8_base_address = int(
            self.impl.get_mem_pool_base_address(
                LayerId(fp8_local_layers[0]), Role.KEY, PageIndexMode.PER_LAYER
            )
        )
        pointer_rows = pool_pointers.tolist()
        # E4M3 pages carry no block scales.
        pointer_rows.append([[fp8_base_address, 0], [0, 0]])
        mapping_rows = pool_mapping.tolist()
        for local_layer_idx in fp8_local_layers:
            converter = self.impl.get_page_index_converter(LayerId(local_layer_idx), Role.KEY)
            mapping_rows[local_layer_idx] = [op_pool_id, int(converter.layer_offset)]
        pinned = prefer_pinned()
        self.kv_cache_pool_pointers = torch.tensor(
            pointer_rows, dtype=pool_pointers.dtype, device="cpu", pin_memory=pinned
        )
        self.kv_cache_pool_mapping = torch.tensor(
            mapping_rows, dtype=pool_mapping.dtype, device="cpu", pin_memory=pinned
        )
        self.num_attention_op_pools = op_pool_id + 1

        # The lifecycle conversion and host table, as the base builds them; the virtual
        # pool has its own conversion below.
        self.index_scales = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=pinned, device="cpu"
        )
        self.kv_offset = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=pinned, device="cpu"
        )
        for pool_id in range(self.num_pools):
            role_a, role_b = self._get_pool_roles(pool_id)
            layer_id = self._pool_layer_ids_by_role[(pool_id, role_a)]
            self.index_scales[pool_id] = self.impl.get_page_index_scale(layer_id, role_a)
            if role_b is not None:
                self.kv_offset[pool_id] = exact_div(
                    self.impl.get_mem_pool_base_address(layer_id, role_b, PageIndexMode.SHARED)
                    - self.impl.get_mem_pool_base_address(layer_id, role_a, PageIndexMode.SHARED),
                    self.impl.get_page_stride(layer_id, role_a),
                )
            else:
                self.kv_offset[pool_id] = 0
        self._index_scale_ints = self.index_scales.tolist()
        self.host_kv_cache_block_offsets = torch.zeros(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=pinned,
            device="cpu",
        )
        self._prepare_page_table_materializer(index_mapper_capacity)

        # The virtual pool's plane holds the DSA slot ids times the FP8 page scale.
        fp8_converter = self.impl.get_page_index_converter(LayerId(fp8_local_layers[0]), Role.KEY)
        self._fp8_latent_op_pool = op_pool_id
        self._fp8_latent_source_pool = dsa_pool_id
        self._fp8_latent_index_scales = torch.tensor(
            [int(fp8_converter.scale)], dtype=torch.int32, device="cpu", pin_memory=pinned
        )
        self._fp8_latent_kv_offsets = torch.zeros(
            1, dtype=torch.int32, device="cpu", pin_memory=pinned
        )
        self._fp8_latent_materializer = (
            _BasePageTableMaterializer(
                self.host_kv_cache_block_offsets[dsa_pool_id : dsa_pool_id + 1],
                self._stream,
                self._fp8_latent_index_scales,
                self._fp8_latent_kv_offsets,
                output_row_capacity=index_mapper_capacity * self.max_copy_beam_width,
            )
            if self._page_table_materializer.uses_device_expansion
            else None
        )

    @override
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """Copy the batch's block offsets of every attention-op pool to the device.

        Args:
            dst_tensor: ``[attention_op_pools, sequences, 2, blocks]`` int32 device
                block offsets.
            request_ids: Requests of the batch, in batch order.
            beam_width: Beam width of the batch.
            num_contexts: Number of context requests, which lead the batch.
            num_seqs: Number of sequences of the batch.
            max_blocks: Accepted for signature parity with the base.
        """
        op_pool_id = self._fp8_latent_op_pool
        if op_pool_id is None:
            super().copy_batch_block_offsets(
                dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=max_blocks
            )
            return
        if dst_tensor.shape[0] <= op_pool_id:
            raise ValueError(
                f"The block offsets hold {dst_tensor.shape[0]} attention-op pools, but the FP8 "
                f"latent pages need {self.num_attention_op_pools}."
            )
        # The base conversion owns the lifecycle planes.
        super().copy_batch_block_offsets(
            dst_tensor[: self.num_pools],
            request_ids,
            beam_width,
            num_contexts,
            num_seqs,
            max_blocks=max_blocks,
        )
        # The FP8 plane: the DSA lifecycle's slot ids times the FP8 page scale.
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        fp8_plane = dst_tensor[op_pool_id : op_pool_id + 1]
        if self._fp8_latent_materializer is not None:
            self._fp8_latent_materializer.copy_block_offsets_to(fp8_plane, copy_idx)
            return
        source_pool_id = self._fp8_latent_source_pool
        source = self.host_kv_cache_block_offsets[source_pool_id : source_pool_id + 1]
        # As the base does for the lifecycle planes (TRTLLM_KV_BLOCK_OFFSETS_TRITON, beam 1).
        if (
            beam_width == 1
            and block_offsets_triton_enabled()
            and copy_batch_block_offsets_from_host(
                source,
                fp8_plane,
                copy_idx,
                self._fp8_latent_index_scales,
                self._fp8_latent_kv_offsets,
                self._stream,
            )
        ):
            return
        copy_batch_block_offsets_to_device(
            source,
            fp8_plane,
            copy_idx,
            self._fp8_latent_index_scales,
            self._fp8_latent_kv_offsets,
            self._stream.cuda_stream,
        )

    @override
    def materialize_block_offsets_snapshot(self, dst_tensor: torch.Tensor, *args, **kwargs) -> None:
        if self._fp8_latent_op_pool is not None:
            raise NotImplementedError(
                "Block-offset snapshots do not cover the FP8 latent plane of the Kimi DSA cache."
            )
        super().materialize_block_offsets_snapshot(dst_tensor, *args, **kwargs)

    @property
    def indexer_k_cache_local_layer_mask(self) -> list[bool]:
        """Indexer ownership of each local layer; KDA layers own none."""
        return [layer_idx in self._indexer_layer_ids for layer_idx in self.pp_layers]

    @override
    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per local indexer layer."""
        return {
            self.layer_offsets[layer_idx]: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self._indexer_bytes_per_token() * tokens_per_block,
                )
            ]
            for layer_idx in sorted(self._indexer_layer_ids)
            if layer_idx in self.layer_offsets
        }

    def _indexer_bytes_per_token(self) -> int:
        return _get_indexer_k_cache_bytes_per_token(
            self.index_head_dim, self.quant_block_size, self.indexer_k_dtype == "fp4"
        )

    @override
    def _get_layer_dtype_by_local_id(self, local_layer_id: int) -> DataType:
        if self._is_fp8_latent_layer(local_layer_id):
            return DataType.FP8
        return super()._get_layer_dtype_by_local_id(local_layer_id)

    def get_kv_cache_dtype(self, layer_idx: int | None = None) -> DataType:
        """Return the storage dtype for a global MLA layer."""
        if layer_idx is None:
            return self.dtype
        return self._get_layer_dtype_by_local_id(self.layer_offsets[layer_idx])

    @override
    def _get_buffer_roles_for_layer(self, local_layer_idx: int) -> list[DataRole]:
        if self._is_fp8_latent_layer(local_layer_idx):
            return [Role.KEY]
        return super()._get_buffer_roles_for_layer(local_layer_idx)

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        index_bytes = (
            self._indexer_bytes_per_token()
            if local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._indexer_layer_ids
            else 0
        )
        if data_role == Role.INDEX_KEY:
            return index_bytes
        if self._is_fp8_latent_layer(local_layer_idx):
            cache_bytes = (
                self.num_kv_heads_per_layer[local_layer_idx]
                * self.head_dim_per_layer[local_layer_idx]
                if data_role in (Role.KEY, Role.ALL)
                else 0
            )
        else:
            cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    @override
    def _attention_cache_bytes_per_token(self) -> int:
        if self._draft_layer_sizes:
            # The base sums the data roles of the target layers (K, NVFP4 block scales) and the
            # standalone draft layers with their page expansion. The indexer K is a role of its
            # own on the MLA layers, which the base leaves out, so it is added here.
            return super()._attention_cache_bytes_per_token() + sum(
                self.get_layer_bytes_per_token(local_layer_idx, Role.INDEX_KEY)
                for local_layer_idx in range(self.num_local_layers)
            )
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @classmethod
    @override
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        tokens_per_block: int = 32,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ) -> tuple[int, int]:
        slope, fixed_cost = super().get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        if draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True):
            slope, fixed_cost = window_draft_cost(
                slope,
                fixed_cost,
                tokens_per_block=tokens_per_block,
                max_batch_size=max_batch_size,
                mapping=mapping,
                **kwargs,
            )
        index_bytes_per_token = _get_indexer_k_cache_size_per_token(model_config, mapping)
        pretrained_config = model_config.pretrained_config
        config = getattr(pretrained_config, "text_config", None) or pretrained_config
        quant_config = model_config.quant_config
        fp8_layers = get_kimi_k31_fp8_kv_layer_ids(
            config,
            nvfp4_enabled=quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(),
        )
        local_fp8_layers = fp8_layers.intersection(mapping.pp_layers(config.num_hidden_layers))
        head_dim = int(config.kv_lora_rank + config.qk_rope_head_dim)
        # The base slope sizes every MLA layer as a latent-only NVFP4 page (half a byte per value
        # plus one E4M3 scale per 16 values): a DSA config does not take the FP4 MLA formula, which
        # would size FP8 layers by itself. An FP8 KV layer stores one byte per latent value.
        fallback_bytes_per_token = len(local_fp8_layers) * (
            head_dim - head_dim // 2 - head_dim // 16
        )
        extra_bytes_per_token = index_bytes_per_token + fallback_bytes_per_token
        state_config = kv_cache_config.mamba_state_config
        seq_limit = max_seq_len if max_seq_len is not None else float("inf")
        has_snapshots = kv_cache_config.enable_block_reuse and (
            0 < state_config.periodic_snapshot_interval <= seq_limit
            or any(
                offset <= seq_limit
                for offset in state_config.additional_snapshot_offsets_from_start
            )
            or any(
                offset < seq_limit for offset in state_config.additional_snapshot_offsets_from_end
            )
        )
        # V2 reserves one partial attention page per resident lineage when
        # snapshots are reachable; include the indexer section of those pages.
        if has_snapshots:
            fixed_cost += (
                max_batch_size * mapping.pp_size * tokens_per_block * extra_bytes_per_token
            )
        return slope + extra_bytes_per_token, fixed_cost

    def get_primary_pool_page_index_params(self, local_layer_idx: int) -> tuple[int, int]:
        """Return the V2 page scale and layer offset of an MLA layer's K page."""
        return self._primary_pool_page_index_params[local_layer_idx]

    def _count_attention_slots(self, local_layer_idx: Optional[int] = None) -> int:
        """Physical slot count of an MLA K pool (not its V2 page bound).

        Counts the K pool of ``local_layer_idx``, by default the first MLA layer's.
        """
        if local_layer_idx is None:
            local_layer_idx = self._attention_local_layers[0]
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.KEY)
        expansion = int(converter.expansion)
        assert page_upper % expansion == 0
        num_pages_with_offset = page_upper // expansion + int(converter.layer_offset)
        scale = int(converter.scale)
        assert num_pages_with_offset % scale == 0
        return num_pages_with_offset // scale

    def _get_indexer_k_cache_pool_data(self, local_layer_idx: int) -> torch.Tensor:
        """Return a contiguous shared-page view for one indexer K-cache."""
        address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.INDEX_KEY, PageIndexMode.SHARED
        )
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.INDEX_KEY)
        flat_page_size = self.tokens_per_block * self._indexer_bytes_per_token()
        return convert_to_torch_tensor(
            TensorWrapper(address, torch.uint8, [page_upper, flat_page_size])
        )

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return the page-indexed indexer K-cache view for a global layer."""
        local_layer_idx = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer.get(local_layer_idx)
        assert pool is not None, f"Layer {layer_idx} owns no DSA indexer K-cache."
        return pool.view(pool.shape[0], self.tokens_per_block, 1, self._indexer_bytes_per_token())

    def get_pool_block_indices(
        self,
        num_seqs: int,
        *,
        request_ids: Optional[list[int]] = None,
        num_contexts: int = 0,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Read the MLA pool's V2 stable slots in current-batch order."""
        if request_ids is None:
            raise ValueError("KimiK3DSAHybridCacheManager requires request_ids to map slots")
        if len(request_ids) != num_seqs:
            raise ValueError(f"Expected {num_seqs} request IDs, got {len(request_ids)}")
        copy_idx = self.index_mapper.get_copy_index(list(request_ids), num_contexts, beam_width)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        num_cols = self._batch_block_columns(request_ids)
        block_indices = self.host_kv_cache_block_offsets[
            self._attention_pool_id, copy_idx, 0, :num_cols
        ]
        return block_indices.clamp(min=0, max=self._num_attention_slots - 1).to(torch.int32)

    def _batch_block_columns(self, request_ids: list[int]) -> int:
        """Block-table columns that can hold a page of this batch.

        A request's pages cover its KV capacity (every token any kernel reads, draft slots
        included), so columns past the largest capacity hold no page of the batch. Reading only
        those keeps the per-step host gather proportional to the batch's context instead of
        ``max_seq_len``.
        """
        width = self.host_kv_cache_block_offsets.shape[-1]
        capacity = 0
        for request_id in request_ids:
            kv_cache = self.kv_cache_map.get(request_id)
            if kv_cache is None:
                return width
            capacity = max(capacity, int(kv_cache.capacity))
        return min(width, max(1, -(-capacity // self.tokens_per_block)))

    def get_primary_pool(self, local_layer_idx: int) -> torch.Tensor:
        """Return this layer's physical K pool as ``[slots, pages, 1, elements]``.

        Different storage dtypes have distinct physical pools in the same
        lifecycle. Slot IDs remain shared with the indexer; page scales and
        offsets describe the layers coalesced in each physical pool.
        """
        pool = self._primary_pool_by_layer.get(local_layer_idx)
        if pool is not None:
            return pool
        if self.kv_factor != 1:
            raise ValueError("DSA requires a SELFKONLY KV cache")
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        if int(converter.expansion) != 1:
            raise ValueError("DSA MLA K-cache does not support page expansion")
        base_address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.KEY, PageIndexMode.PER_LAYER
        )
        if base_address in self._primary_pools:
            pool = self._primary_pools[base_address]
            self._primary_pool_by_layer[local_layer_idx] = pool
            return pool
        head_dim = self.head_dim_per_layer[local_layer_idx]
        num_heads = self.num_kv_heads_per_layer[local_layer_idx]
        dtype = self._get_layer_dtype_by_local_id(local_layer_idx)
        # NVFP4 codes are packed two per byte; their E4M3 scales live in a
        # separate role. DSA gathers both through the manager's pool pointers.
        packed = dtype == DataType.NVFP4
        elements_per_layer = self.tokens_per_block * num_heads * head_dim // (2 if packed else 1)
        shape = [self._num_attention_slots, int(converter.scale), 1, elements_per_layer]
        pool = convert_to_torch_tensor(
            TensorWrapper(base_address, torch.int8 if packed else dtype, shape)
        )
        self._primary_pools[base_address] = pool
        self._primary_pool_by_layer[local_layer_idx] = pool
        return pool

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the primary pool when all MLA layers share one storage layout."""
        pools = [self.get_primary_pool(layer) for layer in self._attention_local_layers]
        if any(pool.data_ptr() != pools[0].data_ptr() for pool in pools[1:]):
            raise ValueError("Mixed Kimi DSA storage requires a layer-specific primary pool.")
        return pools[0]

    def _get_page_index_params(self, local_layer_idx: int, role: DataRole) -> tuple[int, int, int]:
        """Return the (scale, expansion, layer offset) of a buffer's page-index converter."""
        converter = self.impl.get_page_index_converter(local_layer_idx, role)
        return int(converter.scale), int(converter.expansion), int(converter.layer_offset)

    def _validate_mixed_latent_layout(self) -> None:
        """Check the NVFP4 / FP8 latent page layout once, before the first forward.

        The attention ops, the DSA gathers, the indexer and the DSpark drafter address
        pages through the tables that :meth:`_prepare_page_table_tensor` built for this
        layout; under any other V2 layout they would read or write the wrong pages.

        Raises:
            RuntimeError: V2 laid the pages out differently, or a page table does not
                address them as the attention ops expect.
        """
        nvfp4_layers = self._nvfp4_latent_local_layers
        fp8_layers = self._fp8_latent_local_layers
        dsa_pool_id = int(self._attention_pool_id)
        op_pool_id = self.num_pools

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise RuntimeError(f"Kimi DSA FP8 KV layer layout: {message}.")

        require(bool(nvfp4_layers), "no NVFP4 MLA layer on this rank")
        # Each latent pool holds one page per layer in every slot, in local layer order.
        for ordinal, local_layer_idx in enumerate(nvfp4_layers):
            layer_idx = self.pp_layers[local_layer_idx]
            key = self._get_page_index_params(local_layer_idx, Role.KEY)
            expected = (len(nvfp4_layers), 1, ordinal)
            require(
                key == expected,
                f"NVFP4 layer {layer_idx} has K page index (scale, expansion, offset) "
                f"{key}, expected {expected}",
            )
            block_scale = self._get_page_index_params(local_layer_idx, Role.KEY_BLOCK_SCALE)
            require(
                block_scale == key,
                f"NVFP4 layer {layer_idx} has block-scale page index {block_scale}, "
                f"but K page index {key}",
            )
        for ordinal, local_layer_idx in enumerate(fp8_layers):
            layer_idx = self.pp_layers[local_layer_idx]
            key = self._get_page_index_params(local_layer_idx, Role.KEY)
            expected = (len(fp8_layers), 1, ordinal)
            require(
                key == expected,
                f"FP8 layer {layer_idx} has K page index (scale, expansion, offset) "
                f"{key}, expected {expected}",
            )
            elements = (
                self.tokens_per_block
                * self.num_kv_heads_per_layer[local_layer_idx]
                * self.head_dim_per_layer[local_layer_idx]
            )
            stride = int(self.impl.get_page_stride(local_layer_idx, Role.KEY))
            require(
                stride == elements,
                f"FP8 layer {layer_idx} has K pages of {stride} B, expected {elements}",
            )
        # Both latent pools share the DSA slot ids, which get_pool_block_indices bounds.
        nvfp4_slots = self._count_attention_slots(nvfp4_layers[0])
        fp8_slots = self._count_attention_slots(fp8_layers[0])
        require(
            nvfp4_slots == fp8_slots == self._num_attention_slots,
            f"the NVFP4 latent pool has {nvfp4_slots} slots and the FP8 latent pool "
            f"{fp8_slots}, expected {self._num_attention_slots}",
        )

        # The DSA lifecycle's row addresses the NVFP4 pool, the virtual attention-op pool
        # after every lifecycle pool addresses the FP8 pool.
        pointers = self.kv_cache_pool_pointers.tolist()
        mapping = self.kv_cache_pool_mapping.tolist()
        require(
            self._fp8_latent_op_pool == op_pool_id
            and self.num_attention_op_pools == len(pointers) == op_pool_id + 1
            and not self._use_per_layer_page_tables,
            f"{len(pointers)} pool pointer rows and {self.num_attention_op_pools} "
            f"attention-op pools, expected {op_pool_id + 1} with the FP8 pool last",
        )
        nvfp4_row = [
            [
                int(
                    self.impl.get_mem_pool_base_address(
                        nvfp4_layers[0], Role.KEY, PageIndexMode.SHARED
                    )
                ),
                int(
                    self.impl.get_mem_pool_base_address(
                        nvfp4_layers[0], Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                    )
                ),
            ],
            [0, 0],
        ]
        require(
            pointers[dsa_pool_id] == nvfp4_row and 0 not in nvfp4_row[0],
            f"DSA pool row {pointers[dsa_pool_id]}, expected the NVFP4 pool {nvfp4_row}",
        )
        fp8_row = [
            [
                int(
                    self.impl.get_mem_pool_base_address(
                        fp8_layers[0], Role.KEY, PageIndexMode.PER_LAYER
                    )
                ),
                0,
            ],
            [0, 0],
        ]
        require(
            pointers[op_pool_id] == fp8_row and fp8_row[0][0] != 0,
            f"FP8 pool row {pointers[op_pool_id]}, expected {fp8_row}",
        )
        for pool_id, local_layers in ((dsa_pool_id, nvfp4_layers), (op_pool_id, fp8_layers)):
            for ordinal, local_layer_idx in enumerate(local_layers):
                require(
                    mapping[local_layer_idx] == [pool_id, ordinal],
                    f"layer {self.pp_layers[local_layer_idx]} maps to "
                    f"{mapping[local_layer_idx]}, expected {[pool_id, ordinal]}",
                )
        require(
            self._index_scale_ints[dsa_pool_id] == len(nvfp4_layers),
            f"DSA pool page scale {self._index_scale_ints[dsa_pool_id]}, expected "
            f"{len(nvfp4_layers)}",
        )

        # Every other layer keeps its layer group's plane, which the conversion tables
        # cover one entry per lifecycle; DSpark decodes its draft table from that plane
        # with the draft layers' page scale.
        require(
            self.index_scales.numel()
            == self.kv_offset.numel()
            == self.host_kv_cache_block_offsets.shape[0]
            == op_pool_id,
            "the page-index conversion must hold one entry per lifecycle pool",
        )
        fp8_layer_set = frozenset(fp8_layers)
        for local_layer_idx, (pool_id, _) in enumerate(mapping):
            if local_layer_idx not in fp8_layer_set:
                layer_group_id = int(self.layer_to_pool_mapping_dict[local_layer_idx])
                require(
                    pool_id == layer_group_id,
                    f"layer {self.pp_layers[local_layer_idx]} maps to attention-op pool "
                    f"{pool_id}, not to its layer group {layer_group_id}",
                )
        for layer_idx in self.draft_layer_ids:
            local_layer_idx = self.layer_offsets[layer_idx]
            pool_id = mapping[local_layer_idx][0]
            page_scale = int(self.impl.get_page_index_scale(local_layer_idx, Role.KEY))
            require(
                self._index_scale_ints[pool_id] == page_scale,
                f"draft layer {layer_idx} has page scale {page_scale}, its pool "
                f"{self._index_scale_ints[pool_id]}",
            )

        logger.info(
            f"Kimi DSA latent KV: NVFP4 layers "
            f"{[self.pp_layers[layer] for layer in nvfp4_layers]} (page scale "
            f"{len(nvfp4_layers)}), FP8 layers {[self.pp_layers[layer] for layer in fp8_layers]} "
            f"(page scale {len(fp8_layers)}, attention-op pool {op_pool_id} of "
            f"{self.num_attention_op_pools})"
        )

    @override
    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = {}
        self._unique_primary_pool = None
        self._primary_pools.clear()
        self._primary_pool_by_layer.clear()
        super().shutdown()
