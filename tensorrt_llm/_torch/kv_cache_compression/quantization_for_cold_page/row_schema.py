# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row schemas for cold-page quantization: what each hot KV row means.

A :class:`LayerSchema` names the element spans of every hot buffer in one KVCM
layer. A span is ``content`` (NoPE-like values) or ``position`` (RoPE-like
values); a buffer without spans is opaque and is always copied byte-for-byte.
A :class:`ColdPagePolicy` maps span kinds to precisions and turns the resolved
spans into the single kernel row shape ``[lossless prefix][quantized run]
[lossless suffix]``.

Schemas come from an ordered table of resolvers. Each resolver sees the whole
layer list of one KVCM, claims the layers it understands, and never sees a
``model_type``: DeepSeek-V4 is recognized by its buffer roles, MLA by its
key-only layout and latent geometry, and everything else is treated as GQA.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

SpanKind = Literal["content", "position"]
Precision = Literal["quantized", "lossless"]
PositionPolicy = Literal["auto", "quantized", "lossless"]
Family = Literal["deepseek_v4", "mla", "gqa"]

SPAN_ALIGNMENT_ELEMENTS = 16
"""Every span length is a multiple of this, so a quantized run always starts on an NVFP4 scale group."""

_DEFAULT_POSITION_PRECISION: Mapping[str, Precision] = {
    "deepseek_v4": "lossless",
    "mla": "quantized",
    "gqa": "quantized",
}
"""Shipped behavior before the switch existed; ``rope_precision='auto'`` keeps it."""

# HF per-layer-type ``rope_parameters`` keys (Gemma3/Gemma4 style). Their rows
# differ per layer type, which the single-run row shape does not describe.
_PER_LAYER_TYPE_ROPE_KEYS = frozenset({"full_attention", "sliding_attention", "chunked_attention"})

_DEEPSEEK_V4_PREFIX = "deepseek_v4_"
_DEEPSEEK_V4_SWA = f"{_DEEPSEEK_V4_PREFIX}swa"
_DEEPSEEK_V4_COMPRESS = f"{_DEEPSEEK_V4_PREFIX}compress"
_DEEPSEEK_V4_INDEXER_COMPRESS = f"{_DEEPSEEK_V4_PREFIX}indexer_compress"
_DEEPSEEK_V4_CSA_ROLES = frozenset({_DEEPSEEK_V4_COMPRESS, _DEEPSEEK_V4_INDEXER_COMPRESS})
_DEEPSEEK_V4_HCA_ROLES = frozenset({_DEEPSEEK_V4_COMPRESS})
_DEEPSEEK_V4_ROLES = frozenset(
    {
        _DEEPSEEK_V4_SWA,
        _DEEPSEEK_V4_COMPRESS,
        _DEEPSEEK_V4_INDEXER_COMPRESS,
        f"{_DEEPSEEK_V4_PREFIX}compressor_kv",
        f"{_DEEPSEEK_V4_PREFIX}compressor_score",
        f"{_DEEPSEEK_V4_PREFIX}indexer_compressor_kv",
        f"{_DEEPSEEK_V4_PREFIX}indexer_compressor_score",
    }
)
_DEEPSEEK_V4_NOPE_DIM = 448
_DEEPSEEK_V4_ROW_STRIDE = 512
_DEEPSEEK_V4_FOOTER_SCALE_ROW_BYTES = 584


@dataclass(frozen=True)
class RowSpan:
    """One contiguous element range of a hot row."""

    kind: SpanKind
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class BufferSchema:
    """One hot buffer of a layer. No spans means opaque: copied byte-for-byte."""

    role: str
    row_stride_elements: int | None = None
    spans: tuple[RowSpan, ...] = ()
    scale_key: Literal["k", "v"] | None = None

    @property
    def is_opaque(self) -> bool:
        return not self.spans


@dataclass(frozen=True)
class LayerSchema:
    layer_id: int
    family: Family
    model_layer: int | None
    """Model layer for ModelOpt scale lookup; None for draft KVCMs and opaque layers."""
    num_kv_heads: int
    tokens_per_page: int
    buffers: tuple[BufferSchema, ...]

    @property
    def rows(self) -> int:
        return self.num_kv_heads * self.tokens_per_page


class _Exclude:
    """Resolver verdict: KVCM's default lossless codec owns this layer."""

    def __repr__(self) -> str:
        return "EXCLUDE"


EXCLUDE = _Exclude()


@dataclass(frozen=True)
class RowGeometry:
    """Kernel row shape: one quantized run with optional lossless prefix and suffix."""

    quantized_run_start_elements: int
    quantized_run_elements: int
    raw_row_stride_elements: int

    @property
    def lossless_prefix_elements(self) -> int:
        return self.quantized_run_start_elements

    @property
    def lossless_suffix_elements(self) -> int:
        return (
            self.raw_row_stride_elements
            - self.quantized_run_start_elements
            - self.quantized_run_elements
        )


@dataclass(frozen=True)
class ColdPagePolicy:
    """Which precision each span kind gets."""

    position: PositionPolicy = "auto"
    default_position_precision: Mapping[str, Precision] = field(
        default_factory=lambda: dict(_DEFAULT_POSITION_PRECISION)
    )

    def position_precision(self, family: str) -> Precision:
        if self.position == "auto":
            return self.default_position_precision[family]
        return self.position

    def span_precision(self, kind: SpanKind, family: str) -> Precision:
        return "quantized" if kind == "content" else self.position_precision(family)

    def row_geometry(self, buffer: BufferSchema, family: str, *, where: str) -> RowGeometry:
        """Resolve, merge, and validate the spans of one buffer into the kernel row shape."""

        stride = buffer.row_stride_elements
        if stride is None or stride <= 0 or not buffer.spans:
            raise ValueError(f"{where}: a quantized buffer needs a positive row stride and spans")
        spans = sorted(buffer.spans, key=lambda span: span.start)
        cursor = 0
        for span in spans:
            if span.start != cursor or span.length <= 0 or span.length % SPAN_ALIGNMENT_ELEMENTS:
                raise ValueError(
                    f"{where}: spans {spans} must tile the {stride}-element row in order "
                    f"with lengths that are multiples of {SPAN_ALIGNMENT_ELEMENTS}"
                )
            cursor = span.end
        if cursor != stride:
            raise ValueError(f"{where}: spans {spans} cover {cursor} of {stride} row elements")

        # Merge adjacent spans that resolve to the same precision.
        runs: list[tuple[Precision, int, int]] = []
        for span in spans:
            precision = self.span_precision(span.kind, family)
            if runs and runs[-1][0] == precision:
                previous = runs[-1]
                runs[-1] = (precision, previous[1], previous[2] + span.length)
            else:
                runs.append((precision, span.start, span.length))

        quantized = [run for run in runs if run[0] == "quantized"]
        if not quantized:
            raise ValueError(
                f"{where}: no quantized span is left after applying rope_precision="
                f"{self.position_precision(family)!r}; the whole row is position-encoded, "
                "so 'lossless' would leave this buffer uncompressed. Use 'quantized' or 'auto'."
            )
        if len(quantized) > 1:
            raise ValueError(
                f"{where}: {len(quantized)} separate quantized runs {quantized}; the cold-page "
                "kernel supports one lossless prefix, one quantized run, and one lossless suffix"
            )
        _, start, length = quantized[0]
        return RowGeometry(
            quantized_run_start_elements=start,
            quantized_run_elements=length,
            raw_row_stride_elements=stride,
        )


@dataclass(frozen=True)
class ResolverContext:
    """Everything a resolver may read. Geometry arrays are indexed lazily so a
    KVCM layer id that the cache-manager wrapper never registered fails loudly."""

    pretrained_config: object
    policy: ColdPagePolicy
    pp_layers: Sequence[int]
    tokens_per_block: int
    runtime_type: int
    is_draft: bool
    num_kv_heads_per_layer: Sequence[int]
    head_dim_per_layer: Sequence[int]

    @property
    def element_bytes(self) -> int:
        return 1 if self.runtime_type == 2 else 2

    def head_dim(self, layer_id: int) -> int:
        return _indexed(self.head_dim_per_layer, "head_dim", layer_id)

    def num_kv_heads(self, layer_id: int) -> int:
        return _indexed(self.num_kv_heads_per_layer, "num_kv_heads", layer_id)

    def model_layer(self, layer_id: int) -> int | None:
        if self.is_draft:
            return None
        return _indexed(self.pp_layers, "model layer", layer_id)


def _indexed(values: Sequence[int], name: str, layer_id: int) -> int:
    if not 0 <= layer_id < len(values):
        raise ValueError(
            f"KVCM layer {layer_id} has no registered {name}; the cache manager "
            f"registered {len(values)} entries"
        )
    return int(values[layer_id])


def text_config(pretrained_config: object) -> object:
    """Return the text sub-config of a composite (VLM) config, or the config itself."""

    getter = getattr(pretrained_config, "get_text_config", None)
    if callable(getter):
        text = getter()
        if text is not None:
            return text
    text = getattr(pretrained_config, "text_config", None)
    return text if text is not None else pretrained_config


def rotary_dim(text: object, head_dim: int) -> int | None:
    """Number of leading K elements that carry RoPE, or None when the model
    declares per-layer-type RoPE that a single row shape cannot describe."""

    rope_parameters = getattr(text, "rope_parameters", None)
    if hasattr(rope_parameters, "to_dict"):
        rope_parameters = rope_parameters.to_dict()
    if (
        isinstance(rope_parameters, Mapping)
        and rope_parameters
        and set(map(str, rope_parameters)) <= _PER_LAYER_TYPE_ROPE_KEYS
    ):
        return None

    explicit = getattr(text, "rotary_dim", None)
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        value = explicit
    else:
        factor = getattr(text, "partial_rotary_factor", None)
        if factor is None:
            factor = getattr(text, "rotary_pct", None)
        if factor is None and isinstance(rope_parameters, Mapping):
            factor = rope_parameters.get("partial_rotary_factor")
        value = head_dim if factor is None else int(round(head_dim * float(factor)))
    if not 0 <= value <= head_dim or value % SPAN_ALIGNMENT_ELEMENTS:
        raise ValueError(
            f"RoPE covers {value} of {head_dim} head elements; NVFP4 cold pages need a "
            f"multiple of {SPAN_ALIGNMENT_ELEMENTS} within the row"
        )
    return value


Resolution = dict[int, "LayerSchema | _Exclude"]
Resolver = Callable[[Sequence["AttentionLayerConfig"], ResolverContext], "Resolution | None"]


def resolve_deepseek_v4(
    layers: Sequence["AttentionLayerConfig"], ctx: ResolverContext
) -> Resolution | None:
    """Claim every layer that carries a ``deepseek_v4_`` role.

    CSA layers quantize the 448-element NoPE prefix of each 512-element
    compressed row and keep the 64-element RoPE suffix as a ``position`` span.
    An HCA layer colocated with a CSA cache is an opaque layer. SWA, compressor
    state, and HCA without a CSA cache are excluded, so KVCM's lossless codec
    handles them as before.
    """

    roles_by_layer: dict[int, set[str]] = {}
    for layer in layers:
        layer_id = int(layer.layer_id)
        buffer_roles = {str(buffer.role) for buffer in layer.buffers}
        deepseek_v4_roles = {role for role in buffer_roles if role.startswith(_DEEPSEEK_V4_PREFIX)}
        if deepseek_v4_roles:
            if deepseek_v4_roles != buffer_roles or not buffer_roles <= _DEEPSEEK_V4_ROLES:
                raise NotImplementedError(
                    f"Unsupported DeepSeek-V4 cold-page roles: {sorted(buffer_roles)}"
                )
            roles_by_layer[layer_id] = buffer_roles
    if not roles_by_layer:
        return None

    has_csa_cache = any(roles == _DEEPSEEK_V4_CSA_ROLES for roles in roles_by_layer.values())
    model_layers: dict[int, int] = {}
    if has_csa_cache:
        model_layer = None
        num_model_layers = 0
        for layer in layers:
            layer_id = int(layer.layer_id)
            roles = roles_by_layer.get(layer_id)
            if roles is None:
                continue
            if _DEEPSEEK_V4_SWA in roles:
                if num_model_layers == len(ctx.pp_layers):
                    raise ValueError(
                        "DeepSeek-V4 KVCM layout has more model-layer anchors than pp_layers"
                    )
                model_layer = int(ctx.pp_layers[num_model_layers])
                num_model_layers += 1
            elif model_layer is None:
                raise ValueError(
                    "DeepSeek-V4 KVCM layout must begin each model layer with an SWA Page"
                )
            model_layers[layer_id] = model_layer
        if num_model_layers != len(ctx.pp_layers):
            raise ValueError(
                "DeepSeek-V4 KVCM layout and pp_layers have different model-layer counts"
            )

    schemas: Resolution = {}
    for layer in layers:
        layer_id = int(layer.layer_id)
        buffer_roles = roles_by_layer.get(layer_id)
        if buffer_roles is None:
            continue
        if buffer_roles != _DEEPSEEK_V4_CSA_ROLES:
            if has_csa_cache and buffer_roles == _DEEPSEEK_V4_HCA_ROLES:
                schemas[layer_id] = LayerSchema(
                    layer_id=layer_id,
                    family="deepseek_v4",
                    model_layer=None,
                    num_kv_heads=0,
                    tokens_per_page=0,
                    buffers=tuple(BufferSchema(role=str(buffer.role)) for buffer in layer.buffers),
                )
            else:
                schemas[layer_id] = EXCLUDE
            continue

        tokens_per_page = int(ctx.tokens_per_block)
        if tokens_per_page % 4 != 0:
            raise ValueError("DeepSeek-V4 CSA Page geometry must divide tokens_per_block by 4")
        tokens_per_page //= 4

        raw_bytes = tokens_per_page * _DEEPSEEK_V4_ROW_STRIDE * ctx.element_bytes
        configured_bytes = next(
            int(buffer.size)
            for buffer in layer.buffers
            if str(buffer.role) == _DEEPSEEK_V4_COMPRESS
        )
        footer_bytes = tokens_per_page * _DEEPSEEK_V4_FOOTER_SCALE_ROW_BYTES
        if ctx.runtime_type == 2 and configured_bytes == footer_bytes:
            raise NotImplementedError(
                "NVFP4 cold-page compression does not support DeepSeek-V4 "
                "fp8_ds_mla footer-scale Pages"
            )
        if configured_bytes != raw_bytes:
            raise ValueError(
                f"DeepSeek-V4 {_DEEPSEEK_V4_COMPRESS} buffer has "
                f"{configured_bytes} bytes; expected {raw_bytes} for an ordinary runtime Page"
            )

        buffers = []
        for buffer in layer.buffers:
            role = str(buffer.role)
            if role == _DEEPSEEK_V4_COMPRESS:
                buffers.append(
                    BufferSchema(
                        role=role,
                        row_stride_elements=_DEEPSEEK_V4_ROW_STRIDE,
                        spans=(
                            RowSpan("content", 0, _DEEPSEEK_V4_NOPE_DIM),
                            RowSpan(
                                "position",
                                _DEEPSEEK_V4_NOPE_DIM,
                                _DEEPSEEK_V4_ROW_STRIDE - _DEEPSEEK_V4_NOPE_DIM,
                            ),
                        ),
                        scale_key="k",
                    )
                )
            else:
                buffers.append(BufferSchema(role=role))
        schemas[layer_id] = LayerSchema(
            layer_id=layer_id,
            family="deepseek_v4",
            model_layer=None if ctx.is_draft else model_layers[layer_id],
            num_kv_heads=1,
            tokens_per_page=tokens_per_page,
            buffers=tuple(buffers),
        )
    return schemas


def resolve_mla(
    layers: Sequence["AttentionLayerConfig"], ctx: ResolverContext
) -> Resolution | None:
    """Claim every key-only layer as an MLA latent row: ``kv_lora_rank`` content
    elements followed by ``qk_rope_head_dim`` position elements. Any other
    key-only geometry (for example the packed ``fp8_ds_mla`` layout) is rejected
    rather than passed on as GQA."""

    text = text_config(ctx.pretrained_config)
    schemas: Resolution = {}
    for layer in layers:
        roles = [str(buffer.role) for buffer in layer.buffers]
        if "key" not in roles or "value" in roles:
            continue
        layer_id = int(layer.layer_id)
        head_dim = ctx.head_dim(layer_id)
        kv_lora_rank = getattr(text, "kv_lora_rank", None)
        rope_dim = getattr(text, "qk_rope_head_dim", None)
        if (
            not isinstance(kv_lora_rank, int)
            or not isinstance(rope_dim, int)
            or kv_lora_rank + rope_dim != head_dim
        ):
            raise NotImplementedError(
                f"NVFP4 cold-page compression: key-only layer {layer_id} has head_dim {head_dim}, "
                f"which is not the MLA latent geometry kv_lora_rank + qk_rope_head_dim "
                f"({kv_lora_rank} + {rope_dim}); packed layouts such as "
                "kv_cache_config.dtype='fp8_ds_mla' are not supported"
            )
        spans = [RowSpan("content", 0, kv_lora_rank)]
        if rope_dim:
            spans.append(RowSpan("position", kv_lora_rank, rope_dim))
        buffers = [BufferSchema(role="key", row_stride_elements=head_dim, spans=tuple(spans))]
        buffers.extend(BufferSchema(role=role) for role in roles if role != "key")
        schemas[layer_id] = LayerSchema(
            layer_id=layer_id,
            family="mla",
            model_layer=ctx.model_layer(layer_id),
            num_kv_heads=ctx.num_kv_heads(layer_id),
            tokens_per_page=int(ctx.tokens_per_block),
            buffers=tuple(buffers),
        )
    return schemas or None


def resolve_gqa(layers: Sequence["AttentionLayerConfig"], ctx: ResolverContext) -> Resolution:
    """Claim every remaining layer. K rows carry RoPE on their first
    ``rotary_dim`` elements (``partial_rotary_factor`` and friends; the whole
    row when the model rotates every dimension); V rows are all content."""

    text = text_config(ctx.pretrained_config)
    schemas: Resolution = {}
    for layer in layers:
        roles = [str(buffer.role) for buffer in layer.buffers]
        if "key" not in roles:
            raise NotImplementedError(
                "NVFP4 cold-page compression requires an Attention key buffer"
            )
        layer_id = int(layer.layer_id)
        head_dim = ctx.head_dim(layer_id)
        if head_dim <= 0 or head_dim % SPAN_ALIGNMENT_ELEMENTS != 0:
            raise ValueError(f"NVFP4 cold pages require head_dim divisible by 16, got {head_dim}")
        rotary = rotary_dim(text, head_dim)
        if rotary is None:
            if ctx.policy.position_precision("gqa") == "lossless":
                raise NotImplementedError(
                    f"NVFP4 cold-page compression: layer {layer_id} uses per-layer-type "
                    "rope_parameters, whose RoPE layout differs per layer; rope_precision="
                    "'lossless' is not supported for it. Use 'quantized' or 'auto'."
                )
            key_spans: tuple[RowSpan, ...] = (RowSpan("content", 0, head_dim),)
        else:
            key_spans = tuple(
                span
                for span in (
                    RowSpan("position", 0, rotary),
                    RowSpan("content", rotary, head_dim - rotary),
                )
                if span.length
            )
        buffers = [
            BufferSchema(role="key", row_stride_elements=head_dim, spans=key_spans, scale_key="k")
        ]
        if "value" in roles:
            buffers.append(
                BufferSchema(
                    role="value",
                    row_stride_elements=head_dim,
                    spans=(RowSpan("content", 0, head_dim),),
                    scale_key="v",
                )
            )
        buffers.extend(BufferSchema(role=role) for role in roles if role not in ("key", "value"))
        schemas[layer_id] = LayerSchema(
            layer_id=layer_id,
            family="gqa",
            model_layer=ctx.model_layer(layer_id),
            num_kv_heads=ctx.num_kv_heads(layer_id),
            tokens_per_page=int(ctx.tokens_per_block),
            buffers=tuple(buffers),
        )
    return schemas


RESOLVERS: tuple[Resolver, ...] = (resolve_deepseek_v4, resolve_mla, resolve_gqa)


def resolve_layer_schemas(
    layers: Sequence["AttentionLayerConfig"], ctx: ResolverContext
) -> Resolution:
    """Run the resolver table: the first resolver that claims a layer decides it."""

    remaining = list(layers)
    resolution: Resolution = {}
    for resolver in RESOLVERS:
        if not remaining:
            break
        claimed = resolver(remaining, ctx)
        if not claimed:
            continue
        resolution.update(claimed)
        remaining = [layer for layer in remaining if int(layer.layer_id) not in claimed]
    if remaining:
        unclaimed = [int(layer.layer_id) for layer in remaining]
        raise RuntimeError(f"No cold-page resolver claimed layers {unclaimed}")
    return resolution
