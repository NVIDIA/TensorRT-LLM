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
"""DeepSeek-V4.1 KV-cache layout: which layer owns which pool and who reads it.

A DeepSeek-V4.1 model can run on the DeepSeek-V4 cache manager and attention
metadata in a separate layout mode. ``ModelConfig.from_pretrained`` gives a
DeepSeek-V4.1 checkpoint the CSA2 sparse config, which the in-tree DeepSeek-V4.1
model runs on; a model built for this layout mode installs the ``deepseek_v4``
sparse config instead (``use_deepseek_v41_layout_mode``). With that config, the
mode is selected by the checkpoint's own config
(``architectures[0] == "DeepseekV41ForCausalLM"`` or a ``deepseek_v41``
``model_type``), never by a user option, and it reads the per-layer roles from
that config rather than from ``DeepSeekV4SparseAttentionConfig.compress_ratios``
(whose validator maps ratio 0 to 1, erasing the difference between a
window-only layer and a ratio-1 compressed layer).

Per backbone layer ``L`` with ``r = compress_ratios[L]``:

- ``r == 0``: window-only. Owns an SWA pool and nothing else.
- ``r > 0``: attends the window plus the compressed KV of ``kv_source(L)``, the
  latest KV-source layer at or before ``L``. Only KV-source layers own
  ``COMPRESS`` (main KV) and ``INDEXER_COMPRESS`` (index K) pools; the other
  compressed layers are readers and own no compressed pool. KV sources with
  ``r > 1`` also own the compressor pending pair (``COMPRESSOR_KV`` /
  ``COMPRESSOR_SCORE``); ratio 1 has no pending state.
- ``index_source(L)``: the latest index-source layer at or before ``L``, whose
  top-k selection ``L`` consumes. An index source that is not a KV source (a
  "Reindex" layer) scores its own queries over the index K of its KV source.
- The Engram token history (``ENGRAM_TOKEN``, one int32 compressed token id
  per position) is owned by the first Engram layer and shared by all of them.

The reader-to-owner maps live here and not in the cache manager's
``(layer, role) -> manager layer`` table, whose disaggregated-transfer encoding
assumes one owner per entry.
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

from .params import DeepseekV4AttentionType

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

DEEPSEEK_V41_ARCHITECTURES = ("DeepseekV41ForCausalLM",)
DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
DEEPSEEK_V41_SUPPORTED_RATIOS = (0, 1, 2)

# Storage of the SWA and main-KV pools; index K is native MXFP4 in both modes.
# "bf16" stores the reference's fake-quantized values in bf16 containers
# (1024 B per entry). "native" stores the same values as uint8 entries of codes
# followed by their block scales: SWA E4M3 [512] + UE8M0 [16] = 528 B, main KV
# packed E2M1 [256] + E4M3 [32] = 288 B (no global scale).
DEEPSEEK_V41_KV_STORAGE_ENV = "TRTLLM_DSV41_KV_STORAGE"
DEEPSEEK_V41_KV_STORAGE_BF16 = "bf16"
DEEPSEEK_V41_KV_STORAGE_NATIVE = "native"
DEEPSEEK_V41_KV_STORAGES = (DEEPSEEK_V41_KV_STORAGE_BF16, DEEPSEEK_V41_KV_STORAGE_NATIVE)
# Storage used when TRTLLM_DSV41_KV_STORAGE is unset.
DEEPSEEK_V41_DEFAULT_KV_STORAGE = DEEPSEEK_V41_KV_STORAGE_NATIVE

# Encoder SWA bounded replay (tech report 3.2.2): the prefix cache keeps only global KV
# (main KV + index K); a hit of ``reused`` tokens re-forwards the last
# ``min(window, reused)`` cached tokens as replay rows, which rebuild the sliding windows.
DEEPSEEK_V41_ENCODER_REPLAY_ENV = "TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY"
# Replay mode used when TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY is unset. The cache
# manager resolves an unset switch off where replay is not supported (disaggregated
# serving, a KV connector, a separately reused draft pool, a token
# budget of at most one window, or a backend without the rebuild flag); an explicit
# 1 there raises instead.
DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY = True
# Decoder SWA bounded replay (tech report 3.2.2): the layers after the last KV-source
# layer run, in a request's final context chunk, over its last ``window`` prompt
# positions only, so their sliding-window KV holds that tail and nothing below it. The
# model applies it; the cache manager resolves the switch against the engine (unset:
# on, attention TP or DP, except off when block reuse is on while encoder replay is
# off, since a prefix-cache hit inside a tail would then keep window
# KV no forward of the request wrote).
DEEPSEEK_V41_DECODER_REPLAY_ENV = "TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY"
_TRUE_STRINGS = ("1", "true", "on", "yes")
_FALSE_STRINGS = ("0", "false", "off", "no")

# Roles a DeepSeek-V4.1 layout allocates. INDEXER_COMPRESSOR_* (V4's ratio-4
# overlap indexer compressor) never exists in V4.1.
DEEPSEEK_V41_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.SWA,
    DeepseekV4AttentionType.COMPRESSOR_KV,
    DeepseekV4AttentionType.COMPRESSOR_SCORE,
    DeepseekV4AttentionType.ENGRAM_TOKEN,
)
DEEPSEEK_V41_NON_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.COMPRESS,
    DeepseekV4AttentionType.INDEXER_COMPRESS,
)
DEEPSEEK_V41_ATTENTION = DEEPSEEK_V41_SLIDING_ATTENTION + DEEPSEEK_V41_NON_SLIDING_ATTENTION
# Width of the per-layer sliding block table in V4.1 mode, indexed by
# ``attn_type.value`` exactly as in V4. Columns of roles V4.1 never allocates
# stay BAD_PAGE_INDEX.
DEEPSEEK_V41_SLIDING_TABLE_COLUMNS = max(t.value for t in DEEPSEEK_V41_SLIDING_ATTENTION) + 1


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(config: Any) -> Any:
    text_config = _cfg_get(config, "text_config")
    return text_config if text_config is not None else config


def is_deepseek_v41_config(config: Any) -> bool:
    """Whether ``config`` (composite or text sub-config) is a DeepSeek-V4.1 checkpoint."""
    if config is None:
        return False
    architectures = _cfg_get(config, "architectures") or ()
    if len(architectures) > 0 and architectures[0] in DEEPSEEK_V41_ARCHITECTURES:
        return True
    if _cfg_get(config, "model_type") in DEEPSEEK_V41_MODEL_TYPES:
        return True
    return _cfg_get(_cfg_get(config, "text_config"), "model_type") in DEEPSEEK_V41_MODEL_TYPES


def get_deepseek_v41_kv_storage() -> str:
    storage = os.environ.get(DEEPSEEK_V41_KV_STORAGE_ENV, DEEPSEEK_V41_DEFAULT_KV_STORAGE)
    storage = storage.strip().lower() or DEEPSEEK_V41_DEFAULT_KV_STORAGE
    if storage not in DEEPSEEK_V41_KV_STORAGES:
        raise ValueError(
            f"{DEEPSEEK_V41_KV_STORAGE_ENV}={storage!r} is not one of {DEEPSEEK_V41_KV_STORAGES}."
        )
    return storage


def _get_switch_setting(name: str) -> Optional[bool]:
    """The explicit value of boolean switch ``name``, or None when unset."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in _TRUE_STRINGS:
        return True
    if value in _FALSE_STRINGS:
        return False
    raise ValueError(f"{name}={value!r} is not one of {_TRUE_STRINGS + _FALSE_STRINGS}.")


def get_deepseek_v41_encoder_replay_setting() -> Optional[bool]:
    """The explicit value of ``TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY``, or None when unset."""
    return _get_switch_setting(DEEPSEEK_V41_ENCODER_REPLAY_ENV)


def get_deepseek_v41_decoder_replay_setting() -> Optional[bool]:
    """The explicit value of ``TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY``, or None when unset."""
    return _get_switch_setting(DEEPSEEK_V41_DECODER_REPLAY_ENV)


def get_deepseek_v41_encoder_replay() -> bool:
    """The switch's value, ``DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY`` when unset (before the
    cache manager resolves an unset switch against the engine configuration)."""
    setting = get_deepseek_v41_encoder_replay_setting()
    return DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY if setting is None else setting


def deepseek_v41_native_entry_split(
    attn_type: DeepseekV4AttentionType, head_dim: int
) -> Tuple[int, int]:
    """(code bytes, scale bytes) of one native SWA or main-KV entry.

    An entry is one contiguous uint8 row: the codes, then their block scales.
    SWA: E4M3 codes, one UE8M0 byte per 32 values. Main KV: E2M1 codes packed
    two per byte (even element in the low nibble), one E4M3 byte per 16 values.
    """
    if attn_type == DeepseekV4AttentionType.SWA:
        return head_dim, head_dim // 32
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return head_dim // 2, head_dim // 16
    raise ValueError(f"DeepSeek-V4.1 stores {attn_type.name} in one format only.")


def deepseek_v41_token_bytes(
    attn_type: DeepseekV4AttentionType,
    head_dim: int,
    index_head_dim: int,
    kv_storage: str = DEEPSEEK_V41_KV_STORAGE_BF16,
) -> int:
    """Bytes one cache entry of ``attn_type`` occupies in the V4.1 layout."""
    if attn_type in (DeepseekV4AttentionType.SWA, DeepseekV4AttentionType.COMPRESS):
        if kv_storage == DEEPSEEK_V41_KV_STORAGE_NATIVE:
            return sum(deepseek_v41_native_entry_split(attn_type, head_dim))
        return head_dim * 2
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        # MXFP4: packed E2M1 codes plus one UE8M0 byte per 32 values, in every storage mode.
        return index_head_dim // 2 + index_head_dim // 32
    if attn_type in (
        DeepseekV4AttentionType.COMPRESSOR_KV,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
    ):
        return head_dim * 4
    if attn_type == DeepseekV4AttentionType.ENGRAM_TOKEN:
        return 4
    raise ValueError(f"DeepSeek-V4.1 has no cache role {attn_type.name}.")


@dataclass(frozen=True)
class DeepseekV41Layout:
    """Per-layer cache roles of a DeepSeek-V4.1 backbone."""

    # One entry per backbone layer (DSpark stage entries of the checkpoint's
    # list are dropped): 0 = window only, 1 or 2 = window plus compressed KV.
    compress_ratios: Tuple[int, ...]
    kv_source_layer_ids: Tuple[int, ...]
    index_source_layer_ids: Tuple[int, ...]
    candidate_source_layer_id: Optional[int]
    engram_layer_ids: Tuple[int, ...]
    window_size: int
    head_dim: int
    index_head_dim: int
    index_topk: int
    # Positions of Engram history an n-gram hash reads before a run start.
    engram_history_len: int
    kv_storage: str = DEEPSEEK_V41_DEFAULT_KV_STORAGE
    # Encoder SWA bounded replay: windowed roles stay out of the prefix cache and a
    # reuse hit re-forwards the last ``window_size`` cached tokens.
    encoder_replay: bool = DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY

    def __post_init__(self) -> None:
        num_layers = len(self.compress_ratios)
        bad = sorted(set(self.compress_ratios) - set(DEEPSEEK_V41_SUPPORTED_RATIOS))
        if bad:
            raise ValueError(
                f"DeepSeek-V4.1 compress_ratios must be in {DEEPSEEK_V41_SUPPORTED_RATIOS}, got {bad}."
            )
        for name, ids in (
            ("kv_source_layer_ids", self.kv_source_layer_ids),
            ("index_source_layer_ids", self.index_source_layer_ids),
            ("engram_layer_ids", self.engram_layer_ids),
        ):
            if list(ids) != sorted(set(ids)) or any(not 0 <= i < num_layers for i in ids):
                raise ValueError(
                    f"DeepSeek-V4.1 {name} must be sorted, unique backbone layer ids in "
                    f"[0, {num_layers}), got {list(ids)}."
                )
        for layer in self.kv_source_layer_ids:
            if self.compress_ratios[layer] == 0:
                raise ValueError(f"KV-source layer {layer} has compress ratio 0.")
        for layer in self.index_source_layer_ids:
            if self.compress_ratios[layer] == 0:
                raise ValueError(f"Index-source layer {layer} has compress ratio 0.")
        for layer, ratio in enumerate(self.compress_ratios):
            if ratio == 0:
                continue
            kv_source = self.kv_source(layer)
            if kv_source is None or self.compress_ratios[kv_source] != ratio:
                raise ValueError(
                    f"Layer {layer} (ratio {ratio}) has no KV-source layer of the same ratio "
                    f"at or before it (kv_source_layer_ids={list(self.kv_source_layer_ids)})."
                )
            index_source = self.index_source(layer)
            if index_source is None or self.kv_source(index_source) != kv_source:
                raise ValueError(
                    f"Layer {layer} reads the compressed KV of layer {kv_source} but its index "
                    f"source {index_source} selects over a different pool."
                )
        if self.candidate_source_layer_id is not None and (
            self.candidate_source_layer_id not in self.kv_source_layer_ids
            or self.candidate_source_layer_id not in self.index_source_layer_ids
        ):
            raise ValueError(
                f"candidate_source_layer_id {self.candidate_source_layer_id} must be both a "
                "KV source and an index source."
            )
        if self.kv_storage not in DEEPSEEK_V41_KV_STORAGES:
            raise ValueError(f"Unknown DeepSeek-V4.1 KV storage {self.kv_storage!r}.")

    @classmethod
    def from_config(
        cls,
        config: Any,
        kv_storage: Optional[str] = None,
        encoder_replay: Optional[bool] = None,
    ) -> "DeepseekV41Layout":
        """Build the layout from a DeepSeek-V4.1 composite config or its text sub-config."""
        text = _text_config(config)

        def required(name: str) -> Any:
            value = _cfg_get(text, name)
            if value is None:
                raise ValueError(f"DeepSeek-V4.1 config is missing {name!r}.")
            return value

        num_layers = int(required("num_hidden_layers"))
        ratios = [int(r) for r in required("compress_ratios")]
        if len(ratios) < num_layers:
            raise ValueError(
                f"compress_ratios has {len(ratios)} entries for {num_layers} backbone layers."
            )
        head_dim = _cfg_get(text, "head_dim")
        if head_dim is None:
            head_dim = int(required("kv_lora_rank")) + int(required("qk_rope_head_dim"))
        candidate = _cfg_get(text, "candidate_source_layer_id")
        return cls(
            compress_ratios=tuple(ratios[:num_layers]),
            kv_source_layer_ids=tuple(int(i) for i in required("kv_source_layer_ids")),
            index_source_layer_ids=tuple(int(i) for i in required("index_source_layer_ids")),
            candidate_source_layer_id=None if candidate is None else int(candidate),
            engram_layer_ids=tuple(int(i) for i in (_cfg_get(text, "engram_layer_ids") or ())),
            window_size=int(required("sliding_window")),
            head_dim=int(head_dim),
            index_head_dim=int(_cfg_get(text, "index_head_dim", 128)),
            index_topk=int(_cfg_get(text, "index_topk", 512)),
            engram_history_len=max(0, int(_cfg_get(text, "engram_max_ngram_size", 4)) - 1),
            kv_storage=get_deepseek_v41_kv_storage() if kv_storage is None else kv_storage,
            encoder_replay=(
                get_deepseek_v41_encoder_replay() if encoder_replay is None else encoder_replay
            ),
        )

    @property
    def num_layers(self) -> int:
        return len(self.compress_ratios)

    @property
    def compressed_ratios(self) -> Tuple[int, ...]:
        """Ratios that have a compressed stream (a COMPRESS pool per KV source)."""
        return tuple(sorted({self.compress_ratios[i] for i in self.kv_source_layer_ids}))

    @property
    def layer_ratios(self) -> Tuple[int, ...]:
        """Every ratio a backbone layer has, window-only (0) included."""
        return tuple(sorted(set(self.compress_ratios)))

    @property
    def replay_window(self) -> int:
        """Most cached tokens a reuse hit re-forwards (``n_win``, the sliding window)."""
        return self.window_size

    @property
    def reuse_alignment(self) -> int:
        """A reuse boundary under encoder replay is a multiple of every compress ratio, so
        no compressed group straddles it and no compressor state is needed there."""
        return math.lcm(*self.compressed_ratios) if self.compressed_ratios else 1

    @property
    def engram_token_owner(self) -> Optional[int]:
        return self.engram_layer_ids[0] if self.engram_layer_ids else None

    @staticmethod
    def _latest_at_or_before(ids: Tuple[int, ...], layer: int) -> Optional[int]:
        pos = bisect.bisect_right(ids, layer)
        return ids[pos - 1] if pos > 0 else None

    def kv_source(self, layer: int) -> Optional[int]:
        """Owner of the compressed KV and index K ``layer`` reads; None for window-only layers."""
        if self.compress_ratios[layer] == 0:
            return None
        return self._latest_at_or_before(self.kv_source_layer_ids, layer)

    def index_source(self, layer: int) -> Optional[int]:
        """Layer whose top-k selection ``layer`` consumes; None for window-only layers."""
        if self.compress_ratios[layer] == 0:
            return None
        return self._latest_at_or_before(self.index_source_layer_ids, layer)

    def is_kv_source(self, layer: int) -> bool:
        return layer in self.kv_source_layer_ids

    def has_attention(self, layer: int, attn_type: DeepseekV4AttentionType) -> bool:
        """Whether ``layer`` owns a pool of ``attn_type`` (owner-aware, not ratio-based)."""
        if not 0 <= layer < self.num_layers:
            return False
        if attn_type == DeepseekV4AttentionType.SWA:
            return True
        if attn_type in (
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ):
            return self.is_kv_source(layer)
        if attn_type in (
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
        ):
            return self.is_kv_source(layer) and self.compress_ratios[layer] > 1
        if attn_type == DeepseekV4AttentionType.ENGRAM_TOKEN:
            return layer == self.engram_token_owner
        return False

    def owner(self, layer: int, attn_type: DeepseekV4AttentionType) -> Optional[int]:
        """The layer whose pool of ``attn_type`` ``layer`` uses, or None if it uses none."""
        if attn_type == DeepseekV4AttentionType.SWA:
            return layer
        if attn_type in (
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ):
            return self.kv_source(layer)
        if attn_type == DeepseekV4AttentionType.ENGRAM_TOKEN:
            return self.engram_token_owner if layer in self.engram_layer_ids else None
        return layer if self.has_attention(layer, attn_type) else None

    def attention_types(self, layer: int) -> Tuple[DeepseekV4AttentionType, ...]:
        return tuple(t for t in DEEPSEEK_V41_ATTENTION if self.has_attention(layer, t))

    def window(self, attn_type: DeepseekV4AttentionType, max_draft_len: int = 0) -> Optional[int]:
        """Sliding-window size in tokens of a windowed role; None for persistent roles."""
        if attn_type == DeepseekV4AttentionType.SWA:
            base = self.window_size
        elif attn_type in (
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
        ):
            # One pending group of the largest ratio, indexed by absolute position.
            base = max(self.compressed_ratios, default=1)
        elif attn_type == DeepseekV4AttentionType.ENGRAM_TOKEN:
            # The manager keeps positions [h + 1 - window, h] live, h being the
            # first position of the step, and the hash of h reads the
            # engram_history_len positions before it. A replay step starts
            # replay_window positions before the reuse boundary h, and its
            # first rows read the history before that.
            base = self.engram_history_len + 1
            if self.encoder_replay:
                base += self.replay_window
        else:
            return None
        return base + max_draft_len

    def stage_dependency_error(self, layers: Iterable[int]) -> Optional[str]:
        """Why a pipeline stage holding ``layers`` cannot run on its own pools, or None.

        Every cross-layer read of a V4.1 backbone must stay on the stage: a compressed
        layer reads the main KV and index K of its KV source and the top-k of its index
        source; an index source of the candidate source's ratio after it applies
        that layer's candidate-block mask; an Engram layer hashes over the token history owned by
        the first Engram layer. A pool, or a per-step product, of another stage is not
        reachable from this one.
        """
        local = set(layers)
        cand = self.candidate_source_layer_id
        for layer in sorted(local):
            if self.compress_ratios[layer]:
                for what, source in (
                    ("KV source", self.kv_source(layer)),
                    ("index source", self.index_source(layer)),
                ):
                    if source not in local:
                        return (
                            f"layer {layer} reads its {what} layer {source}, held by another stage"
                        )
                if (
                    cand is not None
                    and layer > cand
                    and layer in self.index_source_layer_ids
                    and self.compress_ratios[layer] == self.compress_ratios[cand]
                    and cand not in local
                ):
                    return (
                        f"layer {layer} applies the candidate-block mask of layer {cand}, held by "
                        "another stage"
                    )
            if layer in self.engram_layer_ids and self.engram_token_owner not in local:
                return (
                    f"Engram layer {layer} reads the token history of layer "
                    f"{self.engram_token_owner}, held by another stage"
                )
        return None

    def pipeline_boundaries(self) -> Tuple[int, ...]:
        """Layers ``c`` at which a two-stage split ``[0, c) | [c, num_layers)`` is valid."""
        return tuple(
            c
            for c in range(1, self.num_layers)
            if self.stage_dependency_error(range(c)) is None
            and self.stage_dependency_error(range(c, self.num_layers)) is None
        )

    def persistent_bytes_per_token(self, layers: Iterable[int]) -> int:
        """Global (non-windowed) cache bytes per token of ``layers``: main KV plus index K."""
        total = 0
        for layer in layers:
            if not self.is_kv_source(layer):
                continue
            ratio = self.compress_ratios[layer]
            for attn_type in DEEPSEEK_V41_NON_SLIDING_ATTENTION:
                total += (
                    deepseek_v41_token_bytes(
                        attn_type, self.head_dim, self.index_head_dim, self.kv_storage
                    )
                    // ratio
                )
        return total


def get_deepseek_v41_layout(config: Any) -> Optional[DeepseekV41Layout]:
    """The V4.1 layout of ``config``, or None when it is not a DeepSeek-V4.1 checkpoint."""
    if not is_deepseek_v41_config(config):
        return None
    return DeepseekV41Layout.from_config(config)


def deepseek_v41_sparse_attention_config(
    pretrained_config: Any,
) -> "DeepSeekV4SparseAttentionConfig":
    """The ``deepseek_v4`` sparse config that runs a DeepSeek-V4.1 checkpoint in the layout mode.

    The ``deepseek_v4`` algorithm selects ``DeepseekV4CacheManager`` and
    ``DeepseekV4TrtllmAttentionMetadata``, which enter the V4.1 layout mode from
    ``pretrained_config``. The per-layer layout is not expressible in this config:
    ratio 0 (window only) and ratio 1 (window plus the global KV of the ratio-1
    source layer) are different roles, and ``DeepSeekV4SparseAttentionConfig``
    normalizes 0 to 1. The manager and the metadata therefore read the layout from
    ``pretrained_config``; ``compress_ratios`` here is a placeholder of the
    checkpoint's length in which every layer has ratio 1. The window and the indexer
    geometry come from the checkpoint; the other indexer settings take the values
    ``ModelConfig.from_pretrained`` gives a DeepSeek-V4 checkpoint loaded without a
    user sparse config, with the index K stored as MXFP4.

    Args:
        pretrained_config: DeepSeek-V4.1 composite config or its text sub-config.
    """
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    text = _text_config(pretrained_config)
    checkpoint_ratios = _cfg_get(text, "compress_ratios")
    if not checkpoint_ratios:
        raise ValueError("DeepSeek-V4.1 requires per-layer `compress_ratios` in the checkpoint.")
    window_size = _cfg_get(text, "sliding_window")
    if window_size is None:
        raise ValueError("DeepSeek-V4.1 config is missing 'sliding_window'.")
    return DeepSeekV4SparseAttentionConfig(
        compress_ratios=[1] * len(checkpoint_ratios),
        window_size=int(window_size),
        index_n_heads=_cfg_get(text, "index_n_heads"),
        index_head_dim=int(_cfg_get(text, "index_head_dim", 128)),
        index_topk=int(_cfg_get(text, "index_topk", 512)),
        indexer_max_chunk_size=None,
        skip_indexer_for_short_seqs=True,
        use_cute_dsl_topk=False,
        use_cute_dsl_paged_mqa_logits=False,
        q_split_threshold=8192,
        indexer_rope_interleave=False,
        enable_heuristic_topk=False,
        use_self_sampling_topk=True,
        use_gvr_emission=False,
        indexer_k_dtype="fp4",
    )


def use_deepseek_v41_layout_mode(model_config: "ModelConfig") -> None:
    """Make the engine run ``model_config``'s DeepSeek-V4.1 checkpoint in the layout mode.

    Replaces ``model_config.sparse_attention_config`` with
    ``deepseek_v41_sparse_attention_config(model_config.pretrained_config)``. The
    engine reads the sparse config off the built model's ``model_config`` to pick
    the KV cache manager and the attention metadata, so a model calls this from its
    constructor. ``ModelConfig.from_pretrained`` has given the checkpoint the CSA2
    config by then, the only one it accepts for DeepSeek-V4.1; a CSA2 config that
    carries user-set fields is refused rather than silently dropped.

    Args:
        model_config: The (frozen) ``ModelConfig`` of a DeepSeek-V4.1 checkpoint.
    """
    pretrained_config = model_config.pretrained_config
    if not is_deepseek_v41_config(pretrained_config):
        raise ValueError("use_deepseek_v41_layout_mode() needs a DeepSeek-V4.1 checkpoint config.")
    current = model_config.sparse_attention_config
    if current is not None:
        algorithm = getattr(current, "algorithm", None)
        if algorithm == "deepseek_v4":
            return
        user_fields = set(getattr(current, "model_fields_set", ()))
        if algorithm != "csa2" or user_fields:
            raise ValueError(
                "The DeepSeek-V4.1 layout mode runs on the deepseek_v4 sparse runtime and "
                "takes its settings from the checkpoint; got sparse_attention_config "
                f"{algorithm!r} with user-set fields {sorted(user_fields)}. Leave "
                "sparse_attention_config unset."
            )
    config = deepseek_v41_sparse_attention_config(pretrained_config)
    frozen = model_config._frozen
    model_config._frozen = False
    try:
        model_config.sparse_attention_config = config
    finally:
        model_config._frozen = frozen
