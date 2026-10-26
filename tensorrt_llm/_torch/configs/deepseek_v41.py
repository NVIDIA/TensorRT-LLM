# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-tree config for DeepSeek-V4.1 ("deepseek_v41") checkpoints.

DeepSeek-V4.1 is **not** a config-compatible variant of DeepSeek-V4. Reusing
:class:`~tensorrt_llm._torch.configs.deepseekv4.DeepseekV4Config` for it builds a
silently wrong network, so this module ships a separate config family:

* :class:`DeepseekV41TextConfig` (``deepseek_v41_text``) — the language model.
* :class:`DeepseekV41VisionConfig` (``deepseek_v41_vision``) — the ViT + aligner.
* :class:`DeepseekV41Config` (``deepseek_v41``) — the composite the checkpoint
  ships, nesting the two above plus a **top-level** ``quantization_config``.

The two load-bearing pieces of behaviour live here rather than in the model or the
attention backend, because both need them before a single weight is read:

``build_layer_descriptors``
    The authoritative per-layer classification. V4 encoded a *layer role* in
    ``compress_ratios`` (``{0, 4, 128}``, alternating); V4.1 encodes a *pooling
    factor* over contiguous bands (``{0, 1, 2}``). The value sets share only
    ``0``, so every V4-era ``ratio == 4`` / ``ratio > 1`` test misclassifies V4.1.
    Every backend decision (RoPE selection, indexer enable, compressor dtype, KV
    cache allocation) must read the descriptor instead of comparing a ratio to a
    magic number. See :mod:`tensorrt_llm._torch.configs.deepseek_v41` docstrings
    on :class:`DeepseekV41LayerDescriptor` for the reference citations.

``parse_quantization_layout`` / ``assert_weight_layout``
    V4.1 stores **three** different quantized layouts, and ``quantization_config``
    advertises only one of them (``weight_block_size: [32, 32]``, correct for just
    6.9 GiB of the 475.2 GiB checkpoint). Worse, the routed experts are MXFP4
    packed two elements per ``int8``, so deriving their block naively off the file
    shapes yields 16 — exactly NVFP4's block size — and a loader that believes it
    mis-dequantizes 464 GiB without raising. Deriving the block per tensor,
    accounting for packing, and asserting it per role turns that into a load-time
    failure.

Reference: ``/code/llm-models/DeepSeek-V4.1-Flash/inference/{model,kernel,convert}.py``.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "DEEPSEEK_V41_ARCHITECTURE",
    "DeepseekV41Config",
    "DeepseekV41LayerDescriptor",
    "DeepseekV41QuantLayout",
    "DeepseekV41QuantRole",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "assert_weight_layout",
    "build_layer_descriptors",
    "derive_block",
    "layout_for_role",
    "parse_quantization_layout",
    "quant_role_for_weight_key",
]

DEEPSEEK_V41_ARCHITECTURE = "DeepseekV41ForCausalLM"


# ---------------------------------------------------------------------------
# Per-layer descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepseekV41LayerDescriptor:
    """Everything the runtime needs to know about one layer's attention topology.

    Built once, from the **raw** ``compress_ratios`` list, and never re-derived.
    Field semantics, with the reference lines that define them
    (``inference/model.py`` of the DeepSeek-V4.1-Flash release):

    ``has_long_range``
        ``ratio != 0``. The reference gates the indexed long-range path on plain
        truthiness (``if self.compress_ratio:``, model.py:680). Ratio 1 is
        emphatically *not* "no compression": ``compress_len = (start_pos +
        seqlen) // 1`` is the full sequence, so ratio-1 layers select
        ``index_topk`` positions from all history, unpooled.
    ``pools_kv`` / ``pool_factor``
        ``ratio > 1`` (model.py:447) — the narrower "actually pools" condition.
    ``rope_theta`` / ``yarn_enabled``
        Selected by the *same* truthiness test (model.py:680-696): non-zero takes
        YaRN with ``compress_rope_theta``; zero disables YaRN outright
        (``original_seq_len = 0``) and falls back to the base ``rope_theta``. A
        misclassified layer is therefore evaluated under the wrong RoPE base as
        well as losing its reach — two bugs, not one.
    ``is_kv_source`` / ``owns_compressor`` / ``owns_indexer_wk``
        ``compress_ratio > 0`` does **not** mean the layer compresses its own KV:
        only ``kv_source_layer_ids`` do, and the rest read that same cache
        (model.py:618). Allocating a compressed-KV pool per compressing layer
        over-allocates by ~10x and does not match the reference numerically.
    ``is_index_source``
        ``index_source_layer_ids`` — eight layers against only four
        ``kv_source_layer_ids``. The extra four contribute *queries* against keys
        produced upstream and own no ``indexer.wk``.
    ``compressor_wkv_dtype``
        The reference runs a pooling compressor's ``wkv``/``wgate`` in fp32
        (``x.float()`` at model.py:446) but a ratio-1 compressor in bf16 with no
        gate and no pooling state. The CSA2 module keeps the checkpoint's bf16
        weights (fused ``[wkv | wgate]`` for a pooling compressor) and emits bf16
        rows that the pooling kernel promotes to fp32.
    """

    layer_idx: int
    kind: str  # "decoder" | "mtp"
    compress_ratio: int
    has_long_range: bool
    pools_kv: bool
    pool_factor: int
    rope_theta: float
    yarn_enabled: bool
    window_size: int
    is_kv_source: bool
    is_index_source: bool
    owns_compressor: bool
    owns_indexer_wk: bool
    has_engram: bool
    compressor_wkv_dtype: Optional[str]
    # Which source layer this layer reads its compressed KV / top-k indices from.
    # Equal to ``layer_idx`` on a source layer itself, and ``None`` for a layer
    # with no long-range path (pure sliding window) or before the first source.
    kv_source_layer_idx: Optional[int]
    index_source_layer_idx: Optional[int]
    # Layer 20 publishes the two-level candidate mask; later index sources mask
    # their own scores with it before their own top-k.
    is_candidate_source: bool
    consumes_candidates: bool

    def to_dump_dict(self) -> Dict[str, Any]:
        """Serialize for ``gates/layer_plan.py --diff``.

        The differ compares only the fields it knows about and ignores extras, so
        emitting the full descriptor is both a gate input and a debug artifact.
        """
        return asdict(self)


def build_layer_descriptors(
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    rope_theta: float,
    compress_rope_theta: float,
    window_size: int,
    kv_source_layer_ids: Sequence[int],
    index_source_layer_ids: Sequence[int],
    engram_layer_ids: Sequence[int] = (),
    candidate_source_layer_id: Optional[int] = None,
) -> Tuple[DeepseekV41LayerDescriptor, ...]:
    """Classify every ratio entry from the **raw** config list.

    ``compress_ratios`` must be the untouched checkpoint list. In particular it
    must *not* have been through V4's ``ratio if ratio > 0 else 1`` normalization:
    that rewrite plus a ``ratio > 1`` classifier happens to cancel on ratio-0
    layers, so correcting only one of the two turns a long-context-only bug into a
    short-context regression. Classify from the raw list and the problem does not
    arise.
    """
    if not compress_ratios:
        raise ValueError(
            "DeepSeek-V4.1 requires the per-layer `compress_ratios` list from the "
            "checkpoint config; got an empty list. Refusing to synthesize a "
            "default, which would silently change attention semantics."
        )

    kv_src = set(int(i) for i in kv_source_layer_ids)
    idx_src = set(int(i) for i in index_source_layer_ids)
    engram = set(int(i) for i in engram_layer_ids)

    unknown_kv = sorted(i for i in kv_src if not 0 <= i < len(compress_ratios))
    unknown_idx = sorted(i for i in idx_src if not 0 <= i < len(compress_ratios))
    if unknown_kv or unknown_idx:
        raise ValueError(
            f"kv_source_layer_ids={unknown_kv} / index_source_layer_ids="
            f"{unknown_idx} reference layers outside the "
            f"{len(compress_ratios)}-entry compress_ratios list."
        )

    # A KV source must itself have a long-range path, otherwise it would own a
    # compressor it never uses. Likewise every index source. This is an invariant
    # of the released checkpoint; assert it rather than silently producing a
    # descriptor that cannot be right.
    for i in sorted(kv_src | idx_src):
        if compress_ratios[i] == 0:
            raise ValueError(
                f"layer {i} is listed as a kv/index source but has compress_ratio "
                f"0 (pure sliding window). The config is inconsistent: a source "
                f"layer must have a long-range path."
            )

    descriptors: List[DeepseekV41LayerDescriptor] = []
    last_kv_source: Optional[int] = None
    last_index_source: Optional[int] = None

    for idx, raw_ratio in enumerate(compress_ratios):
        ratio = int(raw_ratio)
        if ratio < 0:
            raise ValueError(f"compress_ratios[{idx}] = {ratio} must be >= 0")

        has_long_range = ratio != 0  # model.py:680 — truthiness, NOT `> 1`
        pools_kv = ratio > 1  # model.py:447 — the narrower pooling condition

        is_kv_source = idx in kv_src
        is_index_source = idx in idx_src
        if is_kv_source:
            last_kv_source = idx
        if is_index_source:
            last_index_source = idx

        if is_kv_source:
            compressor_wkv_dtype = "fp32" if pools_kv else "bf16"
        else:
            compressor_wkv_dtype = None

        is_candidate_source = candidate_source_layer_id is not None and idx == int(
            candidate_source_layer_id
        )
        # Every index source *after* the candidate source masks its scores with
        # the published candidate mask before its own top-k.
        consumes_candidates = (
            is_index_source
            and candidate_source_layer_id is not None
            and idx > int(candidate_source_layer_id)
        )

        descriptors.append(
            DeepseekV41LayerDescriptor(
                layer_idx=idx,
                kind="mtp" if idx >= num_hidden_layers else "decoder",
                compress_ratio=ratio,
                has_long_range=has_long_range,
                pools_kv=pools_kv,
                pool_factor=ratio if pools_kv else 1,
                rope_theta=compress_rope_theta if has_long_range else rope_theta,
                yarn_enabled=has_long_range,
                window_size=window_size,
                is_kv_source=is_kv_source,
                is_index_source=is_index_source,
                owns_compressor=is_kv_source,
                owns_indexer_wk=is_kv_source,
                has_engram=idx in engram,
                compressor_wkv_dtype=compressor_wkv_dtype,
                kv_source_layer_idx=last_kv_source if has_long_range else None,
                index_source_layer_idx=last_index_source if has_long_range else None,
                is_candidate_source=is_candidate_source,
                consumes_candidates=consumes_candidates,
            )
        )

    return tuple(descriptors)


# ---------------------------------------------------------------------------
# Quantization layout
# ---------------------------------------------------------------------------


class DeepseekV41QuantRole:
    """Roles a quantized V4.1 tensor can play, each with its own block layout."""

    EXPERT = "expert"
    ENGRAM_EMBED = "engram_embed"
    DENSE = "dense"


@dataclass(frozen=True)
class DeepseekV41QuantLayout:
    """How one role's tensors are laid out on disk.

    ``element_block`` and ``stored_block`` are deliberately separate, because for
    the routed experts they differ and conflating them is the trap this class
    exists to close. The experts are OCP MXFP4: 32 **elements** share one e8m0
    exponent, but two 4-bit elements are packed per byte and safetensors reports
    the packed extent, so ``weight.shape[-1] // scale.shape[-1]`` on disk yields
    **16**, not 32.

    Sixteen is precisely NVFP4's block size. A loader that derives the block from
    the file and takes it at face value therefore reads MXFP4 as NVFP4, and since
    both formats pack two nibbles per byte, nothing about the shapes objects. The
    result is a silently mis-dequantized 464 GiB of expert weights.
    """

    # Element-level interpretation of the packed bits.
    weight_dtype: str
    # The dtype safetensors actually reports, which for MXFP4 is a container
    # (``int8``) rather than the element type.
    storage_dtype: str
    scale_dtype: str
    # Elements sharing one scale, as (dim0, dim1).
    element_block: Tuple[int, int]
    # Elements packed into one ``storage_dtype`` container along ``packed_dim``.
    pack_factor: int = 1
    packed_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pack_factor < 1:
            raise ValueError(f"pack_factor must be >= 1, got {self.pack_factor}")
        if (self.pack_factor > 1) != (self.packed_dim is not None):
            raise ValueError(
                "pack_factor > 1 requires packed_dim and vice versa; got "
                f"pack_factor={self.pack_factor}, packed_dim={self.packed_dim}"
            )
        if self.packed_dim is not None:
            extent = self.element_block[self.packed_dim]
            if extent % self.pack_factor:
                raise ValueError(
                    f"element_block{self.element_block} is not divisible by "
                    f"pack_factor={self.pack_factor} on dim {self.packed_dim}"
                )

    @property
    def stored_block(self) -> Tuple[int, int]:
        """What ``weight.shape[i] // scale.shape[i]`` must equal for this role.

        Equal to ``element_block`` for the unpacked roles, and ``element_block``
        with the packed dimension divided by ``pack_factor`` for MXFP4.
        """
        if self.packed_dim is None:
            return self.element_block
        block = list(self.element_block)
        block[self.packed_dim] //= self.pack_factor
        return tuple(block)

    def logical_shape(self, stored_shape: Sequence[int]) -> Tuple[int, ...]:
        """Element-count shape of a tensor whose on-disk shape is ``stored_shape``."""
        shape = [int(d) for d in stored_shape]
        if self.packed_dim is not None:
            shape[self.packed_dim] *= self.pack_factor
        return tuple(shape)


# Role -> layout, derived from the safetensors headers of DeepSeek-V4.1-Flash
# rather than from ``quantization_config`` (which advertises [32, 32] for
# everything and is right only for DENSE). Cross-checked against the checkpoint by
# ``<workspace>/gates/quant_layout.py``.
_EXPECTED_LAYOUT: Dict[str, DeepseekV41QuantLayout] = {
    # Routed experts: OCP MXFP4 — e2m1 elements, two per int8 byte, 32-element
    # blocks along K only, e8m0 shared exponent, no per-tensor global scale.
    # 47232 of the checkpoint's 96085 tensors.
    DeepseekV41QuantRole.EXPERT: DeepseekV41QuantLayout(
        weight_dtype="mxfp4_e2m1",
        storage_dtype="int8",
        scale_dtype="float8_e8m0",
        element_block=(1, 32),
        pack_factor=2,
        packed_dim=1,
    ),
    # Engram tables: fp8 e4m3 with one e8m0 scale per 32 columns, unpacked.
    DeepseekV41QuantRole.ENGRAM_EMBED: DeepseekV41QuantLayout(
        weight_dtype="float8_e4m3fn",
        storage_dtype="float8_e4m3fn",
        scale_dtype="float8_e8m0",
        element_block=(1, 32),
    ),
    # Everything else quantized: fp8 e4m3 blockwise 32x32 (V4 was 128x128).
    DeepseekV41QuantRole.DENSE: DeepseekV41QuantLayout(
        weight_dtype="float8_e4m3fn",
        storage_dtype="float8_e4m3fn",
        scale_dtype="float8_e8m0",
        element_block=(32, 32),
    ),
}


# Only the *routed* experts are MXFP4 — verified against the checkpoint: the
# int8-packed tensors are exactly `{layers,mtp}.N.ffn.experts.M.w{1,2,3}.weight` and
# nothing else. `ffn.shared_experts.*` is plain fp8 e4m3 32x32 despite living next
# door, so the expert pattern requires `experts.` to be preceded by a `.` and
# followed by an index — `shared_experts.` therefore falls through to DENSE.
#
# `mlp` is accepted alongside the checkpoint's `ffn` because TensorRT-LLM's own
# module naming uses `mlp`, and this classifier is applied to both sides of the
# load. `<workspace>/gates/quant_layout.py` uses the same alternation.
_EXPERT_RE = re.compile(r"(?:^|\.)(?:ffn|mlp)\.experts\.\d+\.")
_ENGRAM_EMBED_RE = re.compile(r"\.engram\.embed(?:\.|$)")


def quant_role_for_weight_key(key: str) -> str:
    """Classify a checkpoint weight key into a quantization role.

    ``key`` is a tensor name with the trailing ``.weight`` / ``.scale`` still
    attached or already stripped; only the path matters. Accepts both the
    checkpoint's names and TensorRT-LLM's.
    """
    if _EXPERT_RE.search(key):
        return DeepseekV41QuantRole.EXPERT
    if _ENGRAM_EMBED_RE.search(key):
        return DeepseekV41QuantRole.ENGRAM_EMBED
    return DeepseekV41QuantRole.DENSE


def parse_quantization_layout(
    quantization_config: Optional[Dict[str, Any]],
) -> Dict[str, DeepseekV41QuantLayout]:
    """Turn the top-level ``quantization_config`` into a per-role layout table.

    ``quantization_config`` selects the *format family* (fp8 weights with ue8m0
    scales, fp4 routed experts). It is deliberately **not** trusted for the block
    size: ``weight_block_size`` is ``[32, 32]``, which is correct for the dense
    tensors only, while 464.5 of the checkpoint's 475.2 GiB use a 1x32 element
    block.
    """
    cfg = dict(quantization_config or {})
    layout = dict(_EXPECTED_LAYOUT)

    quant_method = cfg.get("quant_method")
    if quant_method not in (None, "fp8"):
        raise ValueError(
            f"DeepSeek-V4.1 expects quantization_config.quant_method 'fp8', got "
            f"{quant_method!r}. Refusing to guess a layout."
        )

    scale_fmt = cfg.get("scale_fmt")
    if scale_fmt not in (None, "ue8m0"):
        raise ValueError(
            f"DeepSeek-V4.1 expects quantization_config.scale_fmt 'ue8m0', got "
            f"{scale_fmt!r}. ue8m0 scales were power-of-two rounded at "
            f"quantization time; another format needs its own dequant path."
        )

    expert_dtype = cfg.get("expert_dtype", "fp4")
    # Recorded rather than rediscovered below: the ``weight_block_size`` override
    # has to follow this aliasing, and an ``is``-identity rescan to find it would
    # break the moment an entry is copied or ``replace()``d anywhere in between.
    expert_aliases_dense = expert_dtype == "fp8"
    if expert_aliases_dense:
        # ``inference/convert.py --expert-dtype fp8`` widens the MXFP4 experts to
        # fp8 e4m3 with 32x32 e8m0 scales at exactly zero error. Supported as a
        # diagnostic oracle; it doubles routed-expert bytes so it is not the
        # shipping configuration.
        layout[DeepseekV41QuantRole.EXPERT] = layout[DeepseekV41QuantRole.DENSE]
    elif expert_dtype != "fp4":
        raise ValueError(
            f"DeepSeek-V4.1 expects quantization_config.expert_dtype 'fp4' (MXFP4) "
            f"or 'fp8' (losslessly widened), got {expert_dtype!r}."
        )

    # ``weight_block_size`` is only authoritative for the dense role. Honour an
    # explicit override there so a future checkpoint that really does use another
    # dense block is not silently mis-read, but never let it touch the other two.
    block = cfg.get("weight_block_size")
    if block is not None:
        if len(block) != 2:
            raise ValueError(f"weight_block_size must have 2 entries, got {block!r}")
        dense = layout[DeepseekV41QuantRole.DENSE]
        overridden = DeepseekV41QuantLayout(
            weight_dtype=dense.weight_dtype,
            storage_dtype=dense.storage_dtype,
            scale_dtype=dense.scale_dtype,
            element_block=(int(block[0]), int(block[1])),
        )
        layout[DeepseekV41QuantRole.DENSE] = overridden
        if expert_aliases_dense:
            layout[DeepseekV41QuantRole.EXPERT] = overridden

    return layout


_DECODER_BOUNDED_REPLAY_ENV = "TRTLLM_V41_DECODER_BOUNDED_REPLAY"


def encoder_replay_enabled() -> bool:
    """Allow independent Encoder state eviction and bounded recovery on a miss."""
    enabled = os.environ.get("TRTLLM_V41_ENCODER_REPLAY", "0") not in ("0", "", "false", "False")
    if enabled and not decoder_bounded_replay_enabled():
        raise ValueError("TRTLLM_V41_ENCODER_REPLAY requires TRTLLM_V41_DECODER_BOUNDED_REPLAY=1")
    return enabled


def decoder_bounded_replay_enabled() -> bool:
    """Enable approximate bounded replay by default; 0 selects full decoder prefill.

    Decoder queries retain one SWA window, enlarged for embedded DSpark captures,
    while global KV is produced from the full encoder input. Ordinary text
    requests execute the tail of every context chunk.
    Model defaults enable chunked prefill and disable prefix block reuse and SWA
    scratch reuse; explicit user settings take precedence. Unsupported topologies
    and scratch reuse retain full prefill. Full-context consumers retain all rows.
    """
    return os.environ.get(_DECODER_BOUNDED_REPLAY_ENV, "1") not in ("0", "", "false", "False")


def disagg_context_decoder_skipping_enabled(bounded_replay_on_generation: bool) -> bool:
    """Omit decoder weights only on explicitly designated remote-tail sources.

    Unspecified roles and generation workers always load the complete model.
    """
    return (
        bounded_replay_on_generation
        and decoder_bounded_replay_enabled()
        and os.environ.get("TRTLLM_DISAGG_ROLE") == "context"
    )


def layout_for_role(
    role: str,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> DeepseekV41QuantLayout:
    """Layout for ``role``, optionally under a specific ``quantization_config``."""
    # ``parse_quantization_layout(None)`` already starts from ``_EXPECTED_LAYOUT``
    # and returns it unchanged, so there is one path to the default, not two.
    layout = parse_quantization_layout(quantization_config)
    try:
        return layout[role]
    except KeyError as exc:
        raise ValueError(f"Unknown DeepSeek-V4.1 quant role: {role!r}") from exc


def derive_block(weight_shape: Sequence[int], scale_shape: Sequence[int]) -> Tuple[int, ...]:
    """Block extent implied by a weight/scale shape pair, per dimension.

    This is the derivation the load-time check is specified against: the block is
    read off the *files*, never taken from ``quantization_config``. Raises when the
    shapes are not compatible at all, which is itself a corruption signal.

    ``weight_shape`` is the shape as *stored*, so the result is comparable against
    :attr:`DeepseekV41QuantLayout.stored_block` — for MXFP4 experts that is 16, not
    32, because two e2m1 elements share a byte. ``<workspace>/gates/quant_layout.py``
    performs the same derivation but unpacks first, so its blocks are comparable
    against :attr:`~DeepseekV41QuantLayout.element_block` (32). Both conventions
    describe the same layout; do not compare a value from one against the other.
    """
    if len(weight_shape) != len(scale_shape):
        raise ValueError(
            f"weight shape {tuple(weight_shape)} and scale shape "
            f"{tuple(scale_shape)} have different ranks"
        )
    block = []
    for dim, (w, s) in enumerate(zip(weight_shape, scale_shape)):
        if s <= 0 or w % s:
            raise ValueError(
                f"weight shape {tuple(weight_shape)} is not an integer multiple of "
                f"scale shape {tuple(scale_shape)} on dim {dim}"
            )
        block.append(int(w) // int(s))
    return tuple(block)


# safetensors spells dtypes in its own header vocabulary, torch in another. The
# caller of `assert_weight_layout` is a weight loader walking those headers, so
# normalizing here removes a mapping table from every call site — and removes the
# failure mode where a caller's mapping is wrong and the dtype half of the check
# silently passes on a name it never actually compared.
_DTYPE_ALIASES = {
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FN": "float8_e4m3fn",
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "FLOAT8_E8M0FNU": "float8_e8m0fnu",
    "I8": "int8",
    "U8": "uint8",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}

# Cache bytes per *lane* of a compressed KV entry, including the per-block scales
# that share the row. Keyed by the layout name
# ``DeepseekV41TextConfig.bytes_per_compressed_entry`` takes; the packed
# ``fp8_ds_mla`` footer layout has no per-lane rate and is handled there.
_LATENT_BYTES_PER_LANE = {
    "bf16": 2.0,
    "fp8": 1.0,
    # Tech report §2.4.4: e2m1 data plus one e4m3 scale per 16-lane block, i.e.
    # NVFP4 without the second-level global scale, as stored by CSA2.
    "fp4": 0.5 + 1.0 / 16.0,
}
_INDEX_K_BYTES_PER_LANE = {
    # e4m3 data plus one fp32 scale per 128-lane block.
    "fp8": 1.0 + 4.0 / 128.0,
    # mxfp4: e2m1 data plus one e8m0 scale per 32-lane block.
    "fp4": 0.5 + 1.0 / 32.0,
}


def _normalize_dtype(name: str) -> str:
    """Map a safetensors / torch / ``str(torch.dtype)`` spelling to one vocabulary."""
    text = str(name)
    if text.startswith("torch."):
        text = text[len("torch.") :]
    return _DTYPE_ALIASES.get(text.upper(), text)


def assert_weight_layout(
    key: str,
    weight_shape: Sequence[int],
    scale_shape: Sequence[int],
    storage_dtype: Optional[str] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> DeepseekV41QuantLayout:
    """Fail loudly when a quantized tensor's on-disk layout is not what its role wants.

    Called once per quantized tensor at load time. Raises :class:`ValueError` — a
    warning would be worse than nothing here, because reading MXFP4 as NVFP4
    produces plausible-looking weights and fluent-but-wrong text, so the failure
    has to be non-negotiable. Returns the matched layout so the caller can use it
    to dequantize without looking it up twice.

    ``weight_shape`` is the shape as stored (packed), and ``storage_dtype`` may be a
    safetensors header dtype (``"I8"``), a torch dtype name (``"int8"``), or
    ``str(tensor.dtype)`` (``"torch.int8"``) — all three are accepted.
    """
    role = quant_role_for_weight_key(key)
    layout = layout_for_role(role, quantization_config)

    if storage_dtype is not None:
        seen = _normalize_dtype(storage_dtype)
        if seen != _normalize_dtype(layout.storage_dtype):
            raise ValueError(
                f"{key}: role {role!r} expects on-disk dtype "
                f"{layout.storage_dtype!r} (elements {layout.weight_dtype!r}) but the "
                f"checkpoint stores {storage_dtype!r}."
            )

    derived = derive_block(weight_shape, scale_shape)
    if derived != layout.stored_block:
        detail = ""
        if layout.pack_factor > 1:
            detail = (
                f" Elements per scale are {layout.element_block}; on disk that is "
                f"{layout.stored_block} because {layout.pack_factor} "
                f"{layout.weight_dtype} elements share one {layout.storage_dtype}."
            )
        raise ValueError(
            f"{key}: role {role!r} expects on-disk block "
            f"{layout.stored_block} but weight {tuple(weight_shape)} / scale "
            f"{tuple(scale_shape)} derives {derived}.{detail}"
        )
    return layout


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class DeepseekV41VisionConfig(PretrainedConfig):
    """ViT + aligner sub-config (``inference/vision.py``).

    The tower is bidirectional (unmasked SDPA), ``PatchEmbed`` is a
    ``nn.Linear(3 * patch_size**2 -> hidden_size)`` rather than a ``Conv2d``, and
    the aligner does a ``downsample_ratio``-squared pixel unshuffle.
    """

    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: Optional[float] = None,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        # The vision tower keeps eps 1e-6; only the text stack moved to 1e-20.
        self.rms_norm_eps = rms_norm_eps
        super().__init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_dim(self) -> int:
        """2D RoPE half-width: ``hidden / heads / 2`` = 32 for the release."""
        return self.head_dim // 2

    @property
    def patch_input_dim(self) -> int:
        """``PatchEmbed`` input width: ``3 * patch_size ** 2`` = 588."""
        return 3 * self.patch_size**2


class DeepseekV41TextConfig(PretrainedConfig):
    """Language-model sub-config for DeepSeek-V4.1.

    Every default here is the released DeepSeek-V4.1-Flash value. That matters
    more than usual: several keys V4 defaults are **absent** from the V4.1 config,
    and absence is an instruction rather than an omission.

    * No ``intermediate_size`` and no ``first_k_dense_replace`` — all layers are
      MoE, there is no dense FFN prefix.
    * No ``num_hash_layers`` — the V4 hash-routing path is off, not defaulted to 3.
    * No ``n_group`` / ``topk_group`` — there is no node-limited routing, so the
      ``noaux_tc`` grouped-topk path degrades to plain top-k over all 384 experts.

    A config class that supplied V4 defaults for these would build the wrong
    network without raising, which is why they are ``None``/absent here.
    """

    model_type = "deepseek_v41_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        # 1e-20, down from V4's 1e-6, and load-bearing: it appears in every
        # RMSNorm. Note the MoE top-k renormalization epsilon is an independent
        # literal 1e-20 that must NOT be plumbed from here (see the router).
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        # MoE
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        gate_temp: float = 1.0,
        # Sparse attention topology
        sliding_window: int = 128,
        compress_ratios: Optional[List[int]] = None,
        compress_rope_theta: float = 160000,
        kv_source_layer_ids: Optional[List[int]] = None,
        index_source_layer_ids: Optional[List[int]] = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: Optional[int] = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        # mHC (hyper-connections)
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Engram
        engram_layer_ids: Optional[List[int]] = None,
        engram_num_embeddings: Optional[List[int]] = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        # DSpark / MTP
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: Optional[List[int]] = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        if rope_scaling is None:
            rope_scaling = {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
        else:
            rope_scaling = dict(rope_scaling)
        # transformers validates these as floats and warns loudly on the ints the
        # checkpoint ships. Normalize rather than let a warning train obscure real
        # load diagnostics.
        for _key in ("factor", "beta_fast", "beta_slow"):
            if _key in rope_scaling and rope_scaling[_key] is not None:
                rope_scaling[_key] = float(rope_scaling[_key])
        self.rope_scaling = rope_scaling

        # MoE. `n_group`/`topk_group` are intentionally absent: V4.1 has no
        # node-limited routing, so noaux_tc degrades to plain top-k over all
        # `n_routed_experts`.
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        # Named `gate_temp` to match the reference gate, which divides the router
        # logits by it (`inference/model.py:811`). Spelling it `gate_temperature`
        # here would send a checkpoint/override that says `gate_temp` into
        # `**kwargs` while every read still saw the 1.0 default.
        self.gate_temp = gate_temp
        # `n_group` / `topk_group` are absent from the V4.1 checkpoint on purpose
        # (see the class docstring), but the shared DeepSeek-V4 modeling code
        # reads them unconditionally when it builds the router. 1/1 is the
        # identity setting -- one group holding all `n_routed_experts`, and that
        # one group selected -- so it reproduces plain global top-k, which is
        # what the reference gate does (`indices = (scores + bias).topk(topk)`,
        # model.py:818). They are written here rather than defaulted as
        # constructor arguments so that a checkpoint which *does* declare group
        # routing cannot silently land on the degenerate setting.
        self.n_group = 1
        self.topk_group = 1
        # V4's hash-routed prefix is off, not defaulted to 3: the checkpoint has
        # no `tid2eid` table and every `ffn.gate` ships a real `bias`, so all 40
        # layers route by top-k.
        self.n_hash_layers = 0

        # Sparse attention topology.
        self.sliding_window = sliding_window
        self.compress_ratios = list(compress_ratios) if compress_ratios is not None else None
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = (
            list(kv_source_layer_ids) if kv_source_layer_ids is not None else []
        )
        self.index_source_layer_ids = (
            list(index_source_layer_ids) if index_source_layer_ids is not None else []
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size

        # mHC.
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # Engram. `engram_num_embeddings` is per layer and the two tables have
        # DIFFERENT sizes, so they are not shared; do not size either from
        # `engram_vocab_size` or `engram_compressed_vocab_size`.
        self.engram_layer_ids = list(engram_layer_ids) if engram_layer_ids is not None else []
        self.engram_num_embeddings = (
            list(engram_num_embeddings) if engram_num_embeddings is not None else []
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        # Two different meanings share this name. The checkpoint's
        # `engram_vocab_size` is a *scalar*: the per-bucket target size that the
        # prime search starts from (`current = args.engram_vocab_size - 1`,
        # engram.py:108). TRT-LLM's `EngramConfig.engram_vocab_size` is a *list*
        # indexed by n-gram size (`self.vocab_size_per_ngram[ngram - 2]`), because
        # its `NgramHashMapping` allows a different target per n-gram order. The
        # reference uses the same target for every order, so the faithful
        # translation is the scalar broadcast across all `max_ngram_size - 1`
        # orders. Verified end-to-end: with 16_000_000 broadcast over 3 orders x 8
        # heads and a `seen` set shared across layers, the resulting prime sums are
        # 384006168 and 384016682 -- exactly this checkpoint's
        # `engram_num_embeddings`, so the derived bucket layout is bit-identical
        # to the one the tables were trained with.
        if isinstance(engram_vocab_size, (list, tuple)):
            distinct = set(int(v) for v in engram_vocab_size)
            if len(distinct) > 1:
                raise ValueError(
                    "DeepSeek-V4.1 derives every n-gram bucket from one scalar "
                    f"engram_vocab_size; got per-order values {list(engram_vocab_size)}. "
                    "A non-uniform list would shift the prime search and rehash "
                    "every table."
                )
            self.engram_bucket_vocab_size = distinct.pop() if distinct else 0
        else:
            self.engram_bucket_vocab_size = int(engram_vocab_size)
        self.engram_vocab_size = [self.engram_bucket_vocab_size] * max(engram_max_ngram_size - 1, 1)
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        # Take pad from the config, never from the tokenizer: the tokenizer
        # reports pad_token_id 1 while the config (and Engram) use 2. Engram
        # substitutes pad_id for blocked n-gram positions, so the wrong sentinel
        # is hashed into every truncated bucket with no error.
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        # Aliases for the names the shared DeepSeek-V4 modeling code and
        # `EngramConfig` read. V4.1's checkpoint spells the geometry per head
        # (`engram_n_heads` x `engram_head_dim`); `EngramConfig` wants the
        # concatenated width per n-gram and divides it back out
        # (`embed_dim_per_head = n_embed_per_ngram // n_head_per_ngram`), so the
        # product is the faithful translation: 8 x 256 = 2048, giving
        # `(max_ngram_size - 1) * 2048 = 6144`, which is exactly the reference
        # `wkv` input width (`n_hash_cols * head_dim`, model.py:344-345).
        self.has_engram = bool(self.engram_layer_ids)
        self.engram_n_head_per_ngram = engram_n_heads
        self.engram_n_embed_per_ngram = engram_n_heads * engram_head_dim
        self.engram_pad_id = engram_pad_token_id
        # The reference derives every hash multiplier from a per-layer RNG seeded
        # by the layer id alone (`compute_hash_multipliers`, engram.py:64-86), so
        # there is no extra global seed to carry; 0 keeps TRT-LLM's generator on
        # its documented default.
        self.engram_seed = 0
        # V4.1's Engram is `embed -> wkv -> gate` with no depthwise convolution
        # (model.py:335-365), unlike the `ShortConv` the TRT-LLM V4 Engram builds.
        # This value therefore sizes a module V4.1 never runs; it is set so
        # `EngramConfig` construction does not have to special-case V4.1, and
        # `DeepseekV41Engram` skips the conv rather than configuring it away.
        self.engram_kernel_size = 4

        # DSpark / MTP.
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            list(dspark_target_layer_ids) if dspark_target_layer_ids is not None else []
        )
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # -- derived geometry ---------------------------------------------------

    @property
    def qk_nope_head_dim(self) -> int:
        """``head_dim - qk_rope_head_dim`` = 448 for the release."""
        return self.head_dim - self.qk_rope_head_dim

    @property
    def kv_lora_rank(self) -> int:
        """The *no-position* part of the shared latent = 448 for the release.

        The reference projects one ``head_dim``-wide latent per token
        (``Attention.wkv = Linear(dim, head_dim)``, model.py:643) and rotates
        only its trailing ``qk_rope_head_dim`` lanes. TRT-LLM's MLA splits that
        same tensor into ``kv_lora_rank`` un-rotated lanes plus
        ``qk_rope_head_dim`` rotated ones and reassembles them by concatenation,
        so ``kv_lora_rank`` is the latent width *minus* the rope lanes -- not the
        latent width. Returning ``head_dim`` here would make the
        ``kv_lora_rank + qk_rope_head_dim`` cache entry 64 lanes too wide and
        misalign every compressed-KV read. Same value V4 pins explicitly (448).
        """
        return self.head_dim - self.qk_rope_head_dim

    @property
    def v_head_dim(self) -> int:
        """Attention output width per head, = the full latent (``head_dim``).

        MLA absorbs V into the same latent it reads K from, so the value width
        is the whole ``head_dim`` while the key's un-rotated part is
        ``kv_lora_rank``. V4 spells this out as ``v_head_dim=512`` next to
        ``kv_lora_rank=448``; V4.1's config ships only ``head_dim``, so derive
        it rather than letting the V4 default leak in.
        """
        return self.head_dim

    def engram_num_embeddings_for_layer(self, layer_idx: int) -> int:
        """Row count of the Engram table hosted by ``layer_idx``.

        The two tables differ in size (384006168 vs 384016682), so this is
        looked up by position in ``engram_layer_ids`` rather than shared.
        """
        try:
            pos = self.engram_layer_ids.index(layer_idx)
        except ValueError as exc:
            raise ValueError(
                f"layer {layer_idx} does not host an Engram table; "
                f"engram_layer_ids = {self.engram_layer_ids}"
            ) from exc
        if pos >= len(self.engram_num_embeddings):
            raise ValueError(
                f"engram_num_embeddings has {len(self.engram_num_embeddings)} "
                f"entries but engram_layer_ids has {len(self.engram_layer_ids)}; "
                f"they must agree."
            )
        return int(self.engram_num_embeddings[pos])

    # -- the descriptor -----------------------------------------------------

    @functools.cached_property
    def layer_descriptors(self) -> Tuple[DeepseekV41LayerDescriptor, ...]:
        """Per-layer attention topology, built once and cached.

        Cached on the instance rather than recomputed so every consumer sees the
        same objects and there is exactly one place the classification happens.
        """
        return build_layer_descriptors(
            compress_ratios=self.compress_ratios or [],
            num_hidden_layers=self.num_hidden_layers,
            rope_theta=self.rope_theta,
            compress_rope_theta=self.compress_rope_theta,
            window_size=self.sliding_window,
            kv_source_layer_ids=self.kv_source_layer_ids,
            index_source_layer_ids=self.index_source_layer_ids,
            engram_layer_ids=self.engram_layer_ids,
            candidate_source_layer_id=self.candidate_source_layer_id,
        )

    def layer_descriptor(self, layer_idx: int) -> DeepseekV41LayerDescriptor:
        return self.layer_descriptors[layer_idx]

    @property
    def quantization_layout(self) -> Dict[str, DeepseekV41QuantLayout]:
        """Per-role ``(weight dtype, scale dtype, block)`` for this checkpoint.

        Lives on the text config because that is what the weight loader holds: a
        flat V4.1 text checkpoint has no composite above it. The composite forwards
        this like every other public text field.
        """
        return parse_quantization_layout(getattr(self, "quantization_config", None))

    def layer_plan_dump(self) -> Dict[str, Any]:
        """Build the JSON payload ``gates/layer_plan.py --diff`` consumes."""
        descriptors = self.layer_descriptors
        return {
            "source": "tensorrt_llm._torch.configs.deepseek_v41.DeepseekV41TextConfig",
            "num_hidden_layers": self.num_hidden_layers,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "num_ratio_entries": len(descriptors),
            "rope_theta": self.rope_theta,
            "compress_rope_theta": self.compress_rope_theta,
            "sliding_window": self.sliding_window,
            "kv_source_layer_ids": sorted(self.kv_source_layer_ids),
            "index_source_layer_ids": sorted(self.index_source_layer_ids),
            "engram_layer_ids": sorted(self.engram_layer_ids),
            "layers": [d.to_dump_dict() for d in descriptors],
        }

    def write_layer_plan_dump(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.layer_plan_dump(), indent=2))
        return out

    def bytes_per_compressed_entry(
        self,
        kv_dtype: str = "bf16",
        indexer_k_dtype: str = "fp8",
    ) -> float:
        """Bytes one KV-source layer stores per *emitted* latent.

        Two tensors are cached per emitted position, under deliberately different
        quantization recipes (the reference uses three different recipes in this
        area; the third, the 128-slot window KV, is fp8 and sized per sequence
        rather than per token, so it is not part of this figure):

        * the compressed KV latent, sized over the **whole** ``head_dim`` width.
          ``head_dim`` and not ``kv_lora_rank``: the cache stores the rotated
          rope lanes alongside the un-rotated ones, and ``kv_lora_rank`` is only
          the un-rotated part (see :attr:`kv_lora_rank`), so sizing from it
          would under-provision every compressed entry by 64 lanes;
        * the indexer key, sized over ``index_head_dim`` with its per-block
          scales folded into the same row (which is how the allocator lays it
          out in the CSA2 GLOBAL record).

        The historical BF16/FP8 defaults give ``1024 + 132 = 1156`` bytes.
        Pass ``kv_dtype="fp4", indexer_k_dtype="fp4"`` for CSA2's actual
        NVFP4 main/MXFP4 index record: ``(256 + 32) + (64 + 4) = 356`` bytes.
        These arguments only select an analytical estimate, not the runtime
        cache encoding or allocation.
        """
        if kv_dtype == "fp8_ds_mla":
            # A packed layout with its scales in a per-row footer, so it has no
            # per-lane rate to quote; take the width the kernels are built for.
            from ..attention.backends.sparse.deepseek_v4 import footer_scale_kv

            if self.head_dim != footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE:
                raise ValueError(
                    f"footer-scale KV is built for head_dim "
                    f"{footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE}, "
                    f"not {self.head_dim}"
                )
            latent_bytes = float(footer_scale_kv.TOKEN_BYTES)
        else:
            try:
                latent_bytes = self.head_dim * _LATENT_BYTES_PER_LANE[kv_dtype]
            except KeyError:
                raise ValueError(
                    f"Unsupported kv_dtype {kv_dtype!r}; expected one of "
                    f"{sorted((*_LATENT_BYTES_PER_LANE, 'fp8_ds_mla'))}."
                ) from None
        try:
            index_k_bytes = self.index_head_dim * _INDEX_K_BYTES_PER_LANE[indexer_k_dtype]
        except KeyError:
            raise ValueError(
                f"Unsupported indexer_k_dtype {indexer_k_dtype!r}; expected one of "
                f"{sorted(_INDEX_K_BYTES_PER_LANE)}."
            ) from None
        return latent_bytes + index_k_bytes

    def kv_bytes_per_token(
        self,
        kv_dtype: str = "bf16",
        indexer_k_dtype: str = "fp8",
    ) -> float:
        """Compressed-KV cost per token, summed over the KV **source** layers.

        Only ``kv_source_layer_ids`` own a compressed cache, and their pooling
        factors differ (2:1 vs unpooled), so the per-layer factors have to be used
        rather than a single model-wide ratio. V4 pooled 128:1; V4.1's band of
        eighteen layers pools 2:1, i.e. 64x more latents, so any capacity
        heuristic tuned against V4 under-provisions V4.1 by close to two orders of
        magnitude on those layers.

        The dtype arguments are forwarded to
        :meth:`bytes_per_compressed_entry`; for the defaults this is
        ``3 * 578 + 1156 = 2890`` B/token — three ratio-2 source layers (2, 8,
        14) emitting every other position, plus the one ratio-1 source layer
        (20) emitting at every position.
        """
        per_entry = self.bytes_per_compressed_entry(kv_dtype, indexer_k_dtype)
        return sum(per_entry / d.pool_factor for d in self.layer_descriptors if d.is_kv_source)


class DeepseekV41Config(PretrainedConfig):
    """Composite config the DeepSeek-V4.1 checkpoint ships.

    Nests :class:`DeepseekV41TextConfig` as ``text_config`` and
    :class:`DeepseekV41VisionConfig` as ``vision_config``, with a **top-level**
    ``quantization_config`` — unlike every DeepSeek config before it, where the
    language-model fields were flat. Sub-configs arrive as nested dicts from
    ``AutoConfig.from_pretrained`` and are rebuilt with the in-tree classes here,
    so no ``trust_remote_code`` is needed for the config.
    """

    model_type = "deepseek_v41"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": DeepseekV41TextConfig,
        "vision_config": DeepseekV41VisionConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[Dict[str, Any], DeepseekV41TextConfig]] = None,
        vision_config: Optional[Union[Dict[str, Any], DeepseekV41VisionConfig]] = None,
        image_token_id: int = 129264,
        **kwargs,
    ):
        # `PretrainedConfig.__init__` runs *first*, deliberately. It is what moves
        # the top-level `quantization_config` out of `kwargs` onto `self`, and its
        # `__post_init__` probes public attributes (`rope_parameters`) that the
        # total forwarding below would otherwise answer out of `text_config` --
        # standardizing the sub-config's RoPE dict through the composite as a side
        # effect. Leaving `text_config` unset until the base class is finished keeps
        # the two configs' RoPE state separate.
        super().__init__(**kwargs)

        # V4.1 carries `quantization_config` at the **top level only**; every
        # DeepSeek config before it was flat, so the language-model fields and the
        # quantization block lived together. A sub-config built from `text_config`
        # verbatim therefore has no quantization at all, and all 475 GiB then load
        # as unquantized bf16 with nothing raising. Propagate it here rather than at
        # the loader call site so every construction path -- `AutoConfig`,
        # `from_dict`, `_CONFIG_REGISTRY`, direct construction in tests -- gets it.
        quantization_config = getattr(self, "quantization_config", None)

        # `sub_configs` is the transformers-declared name for these classes; read it
        # rather than hard-coding them a second time.
        text_cls = self.sub_configs["text_config"]
        vision_cls = self.sub_configs["vision_config"]

        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_dict = dict(text_config)
            if quantization_config is not None:
                text_dict.setdefault("quantization_config", quantization_config)
            text_config = text_cls(**text_dict)
        elif (
            quantization_config is not None
            and getattr(text_config, "quantization_config", None) is None
        ):
            text_config.quantization_config = quantization_config

        if vision_config is None:
            vision_config = vision_cls()
        elif isinstance(vision_config, dict):
            vision_config = vision_cls(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id

        if quantization_config is not None and not getattr(
            self.text_config, "quantization_config", None
        ):
            raise RuntimeError(
                "DeepSeek-V4.1: the top-level quantization_config did not survive into "
                "text_config, so the weights would load as unquantized bf16. Got "
                f"{quantization_config!r} at the top level and "
                f"{getattr(self.text_config, 'quantization_config', None)!r} on text_config."
            )

    # -- pass-through so callers can treat this like a text config -------------
    #
    # Generic TensorRT-LLM machinery (KV-cache sizing, the `ModelConfig` helpers,
    # cpp config conversion) reads language-model fields flat off
    # `pretrained_config`. Forwarding keeps that code working against the nested
    # layout without making every access site conditional.
    #
    # Forwarding is **total** for public names rather than an allowlist. The
    # allowlist this replaced covered 30 of the 59 fields the release
    # `text_config` carries, and the failure mode is silent: `getattr(config,
    # "swiglu_limit", None)` answers `None` for a name that was merely forgotten,
    # which disables the asymmetric SwiGLU clamp with nothing raising. An
    # allowlist also rots the next time the checkpoint grows a field.
    #
    # Leading-underscore names are excluded: those are framework-private state
    # (`_attn_implementation_internal`, `_name_or_path`, `_commit_hash`) that each
    # config owns separately, and every checkpoint field is public.
    #
    # This is delegation, not a copy, and it deliberately reduces
    # `_mirror_text_subconfig_attrs` (`_torch/model_config.py`) to a no-op for
    # V4.1: that helper copies the sub-config's public fields onto the parent, so
    # the 59 fields would exist twice -- diverging under later mutation and
    # serializing at both levels through `to_dict()`. It also only runs inside
    # `ModelConfig.from_pretrained`, where a bare `AutoConfig.from_pretrained`
    # (what the config tests and `gates/layer_plan.py` use) never reaches it.

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails, so this never shadows a
        # real attribute on the composite config.
        if not name.startswith("_") and name not in ("text_config", "vision_config"):
            text_config = self.__dict__.get("text_config")
            if text_config is not None:
                return getattr(text_config, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def tokenizer_name_or_path(self) -> str:
        """Where Engram's ``CompressedTokenizer`` should load the tokenizer from.

        Engram hashes *compressed* token ids and derives every hash multiplier
        from the compressed vocab size, so the wrong tokenizer does not degrade
        quality -- it rehashes both tables into rows that were never trained. The
        reference guards this with an assert (``vocab_size ==
        args.engram_compressed_vocab_size``, ``inference/engram.py:145``).
        TRT-LLM's shared V4 path defaults to ``deepseek-ai/DeepSeek-V3``, which
        would hit the network *and* compress differently, so answer with the
        directory this config was loaded from.

        Defined on the composite rather than on ``text_config`` because that is
        where the load path is recorded (it is not copied into sub-configs) and
        because the shared V4 code reads this off ``pretrained_config``, i.e. off
        the composite. Note that transformers 5 does *not* stamp
        ``_name_or_path`` itself when the config JSON omits it --
        ``ModelConfig.from_pretrained`` fills it in from ``checkpoint_dir``, so
        that is the only entry point guaranteed to satisfy this property.
        Verified: the release
        tokenizer through TRT-LLM's ``CompressedTokenizer`` yields exactly
        ``engram_compressed_vocab_size`` (99092) ids and maps
        ``engram_pad_token_id`` 2 to 2, so the derived multipliers match the
        reference's.
        """
        path = self.__dict__.get("_name_or_path")
        if not path:
            raise RuntimeError(
                "DeepSeek-V4.1 Engram needs the checkpoint's own tokenizer, but this "
                "config carries no `_name_or_path` to load it from. Load the config "
                "with `from_pretrained` (which sets it) rather than constructing it "
                "inline, or assign `_name_or_path` explicitly."
            )
        return str(path)

    def layer_plan_dump(self) -> Dict[str, Any]:
        return self.text_config.layer_plan_dump()

    def write_layer_plan_dump(self, path: Union[str, Path]) -> Path:
        return self.text_config.write_layer_plan_dump(path)
