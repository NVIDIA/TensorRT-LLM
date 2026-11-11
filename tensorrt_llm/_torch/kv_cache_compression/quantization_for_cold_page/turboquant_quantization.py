# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TurboQuant cold-page layout, metadata, and Triton kernel dispatch."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch

from tensorrt_llm.logger import logger

from ...pyexecutor.resource_manager import DataType
from . import turboquant_math as tq_math
from .quantization_for_cold_page import (
    _DEEPSEEK_V4_COMPRESS,
    _DEEPSEEK_V4_CSA_ROLES,
    _DEEPSEEK_V4_HCA_ROLES,
    _DEEPSEEK_V4_NOPE_DIM,
    _DEEPSEEK_V4_ROW_STRIDE,
    _SKIP_ROPE_QUANTIZATION_MODEL_TYPES,
    ColdPageQuantizationCompression,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

_RUNTIME_TYPES = {DataType.HALF: 0, DataType.BF16: 1, DataType.FP8: 2}
_TORCH_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
_ALIGNMENT = 16
_MAX_PIECES = 4
_MAX_INSTANCE_WIDTH = 512
_INSTANCE_WIDTH = 256
_BIT_PLANES = {1: (1, 0), 2: (2, 0), 3: (2, 1), 4: (4, 0)}
_MODES = {"mse": 0, "prod": 1, "fractional": 2}
# A 64-bit stand-in for the KVCM addresses when kernels are compiled ahead of use.
_ADDRESS = 1 << 40


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _next_power_of_two(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


def _power_of_two_pieces(width: int) -> tuple[int, ...]:
    """Split ``width`` into descending powers of two, e.g. 576 -> (512, 64)."""
    return tuple(1 << bit for bit in reversed(range(width.bit_length())) if width >> bit & 1)


@dataclass(frozen=True)
class _TurboQuantBufferLayout:
    """One hot buffer; a zero ``quantized_range_elements`` stores the buffer unchanged."""

    role: str
    quantized_range_start: int = 0
    quantized_range_elements: int = 0


@dataclass(frozen=True)
class _TurboQuantLayerLayout:
    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    raw_row_stride_elements: int
    buffers: tuple[_TurboQuantBufferLayout, ...]


@dataclass(frozen=True)
class _TurboQuantGeometry:
    """Compile-time shape of the cold section of one quantized buffer.

    Buffers with equal geometry share one kernel launch.
    """

    runtime_type: int
    mode: str
    group_bits: tuple[int, ...]
    width: int
    stride: int
    start: int
    rows: int
    tokens_per_page: int

    @property
    def blocked(self) -> int:
        """How a row wider than 512 is split into independent TurboQuant instances.

        Decoding accumulates every output value of an instance on chip (at most 512 fp32
        columns of tensor memory, shared with the staged operand). 1 (prod): the first 512 values
        and the rest. 2 (mse): pieces of 256 values and the rest, decoded on tensor cores.
        """
        if self.mode == "fractional" or self.width <= _MAX_INSTANCE_WIDTH:
            return 0
        return 1 if self.mode == "prod" else 2

    @property
    def pieces(self) -> tuple[int, ...]:
        """Power-of-two widths in which the kernels load a row (and, if blocked == 2, encode it)."""
        if self.blocked == 2:
            full, rest = divmod(self.width, _INSTANCE_WIDTH)
            return (_INSTANCE_WIDTH,) * full + ((rest,) if rest else ())
        return _power_of_two_pieces(self.width)

    @property
    def groups(self) -> tuple[tuple[int, int], ...]:
        """(width, bits) of every coded group of a row, in coordinate order."""
        if self.mode == "fractional":
            return tuple((self.width // 2, bits) for bits in self.group_bits)
        if self.blocked == 1:
            first = _power_of_two_pieces(self.width)[0]
            return ((first, self.group_bits[0]), (self.width - first, self.group_bits[0]))
        # One set of bit planes; with blocked == 2 the instances own consecutive chunks of it.
        return ((self.width, self.group_bits[0]),)

    @property
    def instances(self) -> tuple[int, ...]:
        """Width of every TurboQuant instance of a row (each has its own scale)."""
        if self.blocked == 2:
            return self.pieces
        return tuple(width for width, _ in self.groups)

    @property
    def element_bytes(self) -> int:
        return 1 if self.runtime_type == 2 else 2

    @property
    def raw_elements(self) -> int:
        return self.stride - self.width

    def regions(self) -> dict[str, int]:
        """Byte offsets of every region of the section, and its total size under "end"."""
        offsets: dict[str, int] = {}
        cursor = 0
        for group, (width, bits) in zip("AB", self.groups):
            for plane, plane_bits in enumerate(_BIT_PLANES[bits]):
                offsets[f"{group}{plane}"] = cursor
                cursor += _align(self.rows * width * plane_bits // 8)
        if self.mode == "prod":
            offsets["qjl"] = cursor
            cursor += _align(self.rows * self.width // 8)
        # BF16 scalars per row and instance: the scale, and for prod the residual norm.
        offsets["scale"] = cursor
        scalars = len(self.instances) * (2 if self.mode == "prod" else 1)
        cursor += _align(self.rows * 2 * scalars)
        if self.mode == "fractional":
            offsets["mask"] = cursor
            cursor += _align(self.rows // self.tokens_per_page * self.width // 8)
        offsets["raw"] = cursor
        cursor += _align(self.rows * self.raw_elements * self.element_bytes)
        offsets["end"] = cursor
        return offsets

    def kernel_constants(self, encode: bool) -> dict[str, Any]:
        """Compile-time arguments of the encode or decode kernel for this geometry."""
        import triton.language as tl

        pieces = self.pieces
        chunk = min(min(width & -width for width in self.instances), 128)
        pages_per_tile, rows_pad = 1, self.rows
        if self.mode == "fractional":
            block_m = max(16, _next_power_of_two(self.tokens_per_page))
            tile_rows = self.tokens_per_page
        else:
            # Tensor-core tiles of 64+ rows accumulate in tensor memory: 512 fp32 columns shared
            # by the accumulators and the staged operand. Decoding accumulates a whole instance
            # and prod encoding its whole QJL projection; wider ones use 32-row tiles, whose
            # accumulators live in registers.
            accumulated = max(self.instances)
            if encode and self.mode == "prod":
                # The QJL projection accumulates the whole instance next to the rotated chunk.
                block_m = 64 if accumulated <= 128 else 32
            elif encode and self.width == 128:
                # Small tiles keep more programs in flight; encoding is bound by the host link.
                block_m = 32
            elif encode or accumulated <= 128:
                block_m = 128 if self.width <= 128 else 64
            elif accumulated <= 448:
                block_m = 64
            else:
                block_m = 32
            tile_rows = block_m
            if self.rows < block_m:
                rows_pad = _next_power_of_two(self.rows)
                pages_per_tile = block_m // rows_pad
        regions = self.regions()
        groups = self.groups
        planes_a = _BIT_PLANES[groups[0][1]]
        planes_b = _BIT_PLANES[groups[-1][1]] if len(groups) > 1 else (0, 0)
        codebooks = {}
        if self.blocked == 2:  # instance 0 and those of its width use CB0, the last one CB1
            coded = ((pieces[0], groups[0][1]), (pieces[-1], groups[0][1]))
        else:
            coded = (groups[0], groups[-1])
        for name, (width, bits) in zip("01", coded):
            # Positive half of the symmetric codebook: thresholds T1..T7, centroids C0..C7.
            centroids, thresholds = tq_math.lloyd_max_codebook(width, bits)
            half = 1 << (bits - 1)
            positive_thresholds = [float(t) for t in thresholds[half:]]
            positive_centroids = [float(c) for c in centroids[half:]]
            codebooks[f"BITS{name}"] = bits
            codebooks[f"CB{name}"] = tuple(
                positive_thresholds
                + [0.0] * (7 - len(positive_thresholds))
                + positive_centroids
                + [0.0] * (8 - len(positive_centroids))
            )
        return dict(
            ELEM=(tl.float16, tl.bfloat16, tl.float8e4nv)[self.runtime_type],
            N=self.width,
            STRIDE=self.stride,
            QS=self.start,
            P0=pieces[0],
            P1=pieces[1] if len(pieces) > 1 else 0,
            P2=pieces[2] if len(pieces) > 2 else 0,
            P3=pieces[3] if len(pieces) > 3 else 0,
            C=chunk,
            ROWS=self.rows,
            BLOCK_M=block_m,
            TILE_ROWS=tile_rows,
            PAGES_PER_TILE=pages_per_tile,
            ROWS_PAD=rows_pad,
            MODE=_MODES[self.mode],
            BLOCKED=self.blocked,
            **codebooks,
            OFF_A0=regions["A0"],
            W_A0=planes_a[0],
            OFF_A1=regions.get("A1", 0),
            W_A1=planes_a[1],
            OFF_B0=regions.get("B0", 0),
            W_B0=planes_b[0],
            OFF_B1=regions.get("B1", 0),
            W_B1=planes_b[1],
            OFF_QJL=regions.get("qjl", 0),
            OFF_SCALE=regions["scale"],
            OFF_MASK=regions.get("mask", 0),
            OFF_RAW=regions["raw"],
            RAW=self.raw_elements,
            RAW_BLOCK=_next_power_of_two(max(self.raw_elements, 1)),
            num_warps=4 if block_m * self.width <= 4096 and self.mode != "prod" else 8,
            # Wide rows keep their pieces resident; pipelining their matrix tiles would not fit.
            num_stages=1 if self.width > 256 else 2,
        )

    @staticmethod
    def grid(constants: dict[str, Any], num_pages: int) -> tuple[int, int]:
        """(programs along the Page axis, tiles per buffer) for a call with ``num_pages``."""
        pages_per_tile = constants["PAGES_PER_TILE"]
        tiles = -(-constants["ROWS"] // constants["TILE_ROWS"]) if pages_per_tile == 1 else 1
        return -(-num_pages // pages_per_tile), tiles


@dataclass
class _TurboQuantLaunch:
    """One kernel launch per call: buffers of one geometry, or every losslessly copied buffer."""

    num_buffers: int
    table: torch.Tensor
    encode_args: tuple
    decode_args: tuple
    encode_constants: dict[str, Any]
    decode_constants: dict[str, Any]
    hot_base: int = 0
    """Address all hot offsets of the table are relative to (the lowest hot buffer)."""
    encode_kernel: Any = None
    """Kernels compiled at configure time and launched without the JIT dispatcher."""
    decode_kernels: tuple = ()
    copy: bool = False


@dataclass(frozen=True)
class _TurboQuantColdPageMetadata:
    """Launch metadata of one KVCM lifecycle; the tensors live on the KVCM's device."""

    launches: tuple[_TurboQuantLaunch, ...]
    cold_page_bytes: int


@dataclass
class _TurboQuantColdPageCodecState:
    """TurboQuant state owned by one target, draft, or retry codec."""

    layer_layouts: dict[int, _TurboQuantLayerLayout]
    layer_ids: tuple[int, ...]
    runtime_type: int
    lifecycle_metadata: tuple[_TurboQuantColdPageMetadata, ...] = field(init=False)


class TurboQuantColdPageQuantizationCompression(ColdPageQuantizationCompression):
    """TurboQuant cold pages (arXiv 2504.19874): rotation, Lloyd-Max codebooks, optional QJL."""

    _FORMAT_NAME = "TurboQuant"
    _RANGE_ALIGNMENT = 16
    _RANGE_ALIGNMENT_UNIT = "TurboQuant rotation pieces"

    def __init__(
        self,
        config: "ColdPageQuantizationCompressionConfig",
        *,
        pretrained_config: "PretrainedConfig",
    ) -> None:
        super().__init__(config, pretrained_config=pretrained_config)
        self._skip_rope_quantization = bool(config.skip_rope_quantization)
        model_type = getattr(pretrained_config, "model_type", None)
        architectures = getattr(pretrained_config, "architectures", None) or []
        if architectures and architectures[0] == "GlmMoeDsaForCausalLM":
            model_type = "glm_moe_dsa"
        if self._skip_rope_quantization and model_type not in _SKIP_ROPE_QUANTIZATION_MODEL_TYPES:
            logger.warning(
                "skip_rope_quantization is supported for model types "
                f"{sorted(_SKIP_ROPE_QUANTIZATION_MODEL_TYPES)} only; ignoring it for "
                f"{model_type!r} and TurboQuant-encoding whole K and V vectors."
            )
            self._skip_rope_quantization = False
        bits = float(config.turboquant_bits)
        self._mode = config.turboquant_mode
        self._seed = int(config.turboquant_seed)
        if bits != int(bits):
            if self._mode != "mse":
                raise ValueError("Fractional TurboQuant bit widths support only mode 'mse'")
            # Outlier channels get the next bit width up, regular channels the one below.
            self._mode = "fractional"
            self._group_bits = (math.ceil(bits), math.floor(bits))
        elif self._mode == "prod":
            self._group_bits = (int(bits) - 1,)
        else:
            self._group_bits = (int(bits),)
        self._parameters: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> _TurboQuantColdPageCodecState:
        from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

        runtime_type = _RUNTIME_TYPES.get(runtime_dtype)
        attention_layers = [
            layer for layer in cache_config.layers if isinstance(layer, AttentionLayerConfig)
        ]
        if attention_layers and runtime_type is None:
            raise RuntimeError(
                "TurboQuant cold-page compression supports FP16, BF16, or FP8 "
                f"Attention KV, not {runtime_dtype}"
            )
        runtime_type = runtime_type if runtime_type is not None else 0

        # The codec holds only the target model's config, so it cannot locate RoPE in a draft
        # model's K vectors: draft KVCMs always quantize whole vectors.
        skip_rope = self._skip_rope_quantization
        if skip_rope and is_draft:
            logger.warning(
                "skip_rope_quantization: draft-model K and V vectors are TurboQuant-encoded in full."
            )
            skip_rope = False

        layouts: dict[int, _TurboQuantLayerLayout] = {}
        deepseek_v4_layers: set[int] = set()
        if getattr(self.pretrained_config, "model_type", None) == "deepseek_v4":
            deepseek_v4_layouts = self._build_deepseek_v4_layer_layouts(
                cache_config,
                attention_layers=attention_layers,
                pp_layers=pp_layers,
                runtime_type=runtime_type,
            )
            deepseek_v4_layers = set(deepseek_v4_layouts)
            layouts.update(
                {lid: layout for lid, layout in deepseek_v4_layouts.items() if layout is not None}
            )

        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            if layer_id in deepseek_v4_layers:
                continue
            buffer_roles = {str(buffer.role) for buffer in layer.buffers}
            if "key" not in buffer_roles:
                raise NotImplementedError(
                    "TurboQuant cold-page compression requires an Attention key buffer"
                )
            compressed_roles = ("key", "value") if "value" in buffer_roles else ("key",)
            head_dim = int(head_dim_per_layer[layer_id])
            key_range = self._calculate_quantized_range(
                head_dim,
                skip_rope=skip_rope,
                key_only=compressed_roles == ("key",),
                buffer_name=f"cold-page layer {layer_id} key",
            )
            buffers = [
                _TurboQuantBufferLayout(
                    role=role,
                    quantized_range_start=key_range[0] if role == "key" else 0,
                    quantized_range_elements=key_range[1] if role == "key" else head_dim,
                )
                for role in compressed_roles
            ]
            for buffer in buffers:
                self._check_quantized_width(buffer.quantized_range_elements, layer_id, buffer.role)
            buffers += [
                _TurboQuantBufferLayout(role=str(buffer.role))
                for buffer in layer.buffers
                if str(buffer.role) not in compressed_roles
            ]
            layouts[layer_id] = _TurboQuantLayerLayout(
                layer_id=layer_id,
                num_kv_heads=int(num_kv_heads_per_layer[layer_id]),
                tokens_per_page=int(cache_config.tokens_per_block),
                raw_row_stride_elements=head_dim,
                buffers=tuple(buffers),
            )

        return _TurboQuantColdPageCodecState(
            layer_layouts=layouts,
            layer_ids=tuple(sorted(layouts)),
            runtime_type=runtime_type,
        )

    def _check_quantized_width(self, width: int, layer_id: int, role: str) -> None:
        unit = 32 if self._mode == "fractional" else 16
        if width <= 0 or width % unit:
            raise ValueError(
                f"TurboQuant cold pages ({self._mode}) need a quantized width divisible by {unit}; "
                f"layer {layer_id} {role} has {width}"
            )
        if width >= 2 * _MAX_INSTANCE_WIDTH:
            raise NotImplementedError(
                f"TurboQuant cold pages support up to {2 * _MAX_INSTANCE_WIDTH - 16} quantized "
                f"elements per vector; layer {layer_id} {role} has {width}"
            )
        rest = width % _INSTANCE_WIDTH
        if self._mode == "mse" and width > _MAX_INSTANCE_WIDTH and rest & (rest - 1):
            raise NotImplementedError(
                f"TurboQuant cold pages split vectors wider than {_MAX_INSTANCE_WIDTH} into "
                f"{_INSTANCE_WIDTH}-value instances plus a power-of-two rest; layer {layer_id} "
                f"{role} has {width} quantized elements"
            )
        if len(_power_of_two_pieces(width // 16)) > _MAX_PIECES:
            raise NotImplementedError(
                f"TurboQuant cold pages split each vector into at most {_MAX_PIECES} power-of-two "
                f"pieces; layer {layer_id} {role} has {width} quantized elements"
            )

    def _build_deepseek_v4_layer_layouts(
        self,
        cache_config: object,
        *,
        attention_layers: Sequence["AttentionLayerConfig"],
        pp_layers: Sequence[int],
        runtime_type: int,
    ) -> dict[int, _TurboQuantLayerLayout | None]:
        """CSA rows keep 64 RoPE values and the indexer cache unchanged; None selects fallback."""

        roles_by_layer, has_csa_cache, _ = self._deepseek_v4_model_layers(
            attention_layers, pp_layers
        )
        layouts: dict[int, _TurboQuantLayerLayout | None] = {}
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            buffer_roles = roles_by_layer.get(layer_id)
            if buffer_roles is None:
                continue
            if buffer_roles != _DEEPSEEK_V4_CSA_ROLES:
                layouts[layer_id] = None
                if has_csa_cache and buffer_roles == _DEEPSEEK_V4_HCA_ROLES:
                    # Co-located with the CSA cache in one lifecycle: owned, but stored as is.
                    layouts[layer_id] = _TurboQuantLayerLayout(
                        layer_id=layer_id,
                        num_kv_heads=0,
                        tokens_per_page=0,
                        raw_row_stride_elements=0,
                        buffers=tuple(
                            _TurboQuantBufferLayout(role=str(buffer.role))
                            for buffer in layer.buffers
                        ),
                    )
                continue
            tokens_per_page = self._deepseek_v4_csa_tokens_per_page(
                cache_config, layer, runtime_type=runtime_type
            )
            layouts[layer_id] = _TurboQuantLayerLayout(
                layer_id=layer_id,
                num_kv_heads=1,
                tokens_per_page=tokens_per_page,
                raw_row_stride_elements=_DEEPSEEK_V4_ROW_STRIDE,
                buffers=tuple(
                    _TurboQuantBufferLayout(
                        role=str(buffer.role), quantized_range_elements=_DEEPSEEK_V4_NOPE_DIM
                    )
                    if str(buffer.role) == _DEEPSEEK_V4_COMPRESS
                    else _TurboQuantBufferLayout(role=str(buffer.role))
                    for buffer in layer.buffers
                ),
            )
        return layouts

    def _page_index_location(self, native: Any) -> object:
        return native.ColdPageIndexLocation.DEVICE

    def build_lifecycle_metadata(
        self, codec_state: _TurboQuantColdPageCodecState, lifecycle: object
    ) -> _TurboQuantColdPageMetadata:
        element_bytes = 1 if codec_state.runtime_type == 2 else 2
        quantized: dict[_TurboQuantGeometry, list[tuple[int, int, int]]] = {}
        copied: list[tuple[int, int, int, int]] = []
        cursor = 0
        for layer_id, hot_buffers in lifecycle.layers.items():
            layout = codec_state.layer_layouts[int(layer_id)]
            if set(hot_buffers) != {buffer.role for buffer in layout.buffers}:
                raise ValueError(f"Cold-page layer {layer_id} roles do not match its KVCM layout")
            rows = layout.num_kv_heads * layout.tokens_per_page
            for buffer in layout.buffers:
                hot = hot_buffers[buffer.role]
                raw_base, raw_slot_bytes = int(hot.raw_base), int(hot.raw_slot_bytes)
                raw_bytes = int(hot.raw_bytes)
                if raw_base <= 0 or raw_bytes <= 0 or raw_bytes > raw_slot_bytes:
                    raise ValueError("Cold-page hot buffer has invalid address or size")
                if not buffer.quantized_range_elements:
                    copied.append((raw_base, raw_slot_bytes, raw_bytes, cursor))
                    cursor += _align(raw_bytes)
                    continue
                if raw_bytes != rows * layout.raw_row_stride_elements * element_bytes:
                    raise ValueError("Hot buffer size does not match TurboQuant geometry")
                if raw_base % _ALIGNMENT or raw_slot_bytes % _ALIGNMENT:
                    raise ValueError(
                        "TurboQuant hot address and Slot stride must be 16-byte aligned"
                    )
                geometry = _TurboQuantGeometry(
                    runtime_type=codec_state.runtime_type,
                    mode=self._mode,
                    group_bits=self._group_bits,
                    width=buffer.quantized_range_elements,
                    stride=layout.raw_row_stride_elements,
                    start=buffer.quantized_range_start,
                    rows=rows,
                    tokens_per_page=layout.tokens_per_page,
                )
                quantized.setdefault(geometry, []).append((raw_base, raw_slot_bytes, cursor))
                cursor += geometry.regions()["end"]
        cold_page_bytes = _align(cursor)
        if cold_page_bytes == 0:
            raise ValueError("TurboQuant cold-page lifecycle has no buffers")

        device = torch.device("cuda", torch.cuda.current_device())
        launches = [
            self._quantized_launch(geometry, entries, device)
            for geometry, entries in quantized.items()
        ]
        if copied:
            launches.append(self._copy_launch(copied, device))
        return _TurboQuantColdPageMetadata(
            launches=tuple(launches), cold_page_bytes=cold_page_bytes
        )

    def _group_parameters(self, geometry: _TurboQuantGeometry, device: torch.device) -> tuple:
        """Device matrices and codebooks of a geometry, shared by every KVCM of this provider."""

        key = (geometry.width, device)
        if key in self._parameters:
            return self._parameters[key]
        widths = list(geometry.instances)
        if self._mode == "fractional":
            # Channel c of the outlier/regular split reads row pos[c] of B^T, B = diag(R_hi, R_lo).
            rotation = torch.block_diag(
                tq_math.haar_rotation(self._seed, widths[0], tq_math.ROTATION),
                tq_math.haar_rotation(self._seed, widths[1], tq_math.REGULAR_ROTATION),
            )
            encode = decode = rotation.T
        else:
            rotation = torch.block_diag(*(tq_math.haar_rotation(self._seed, w) for w in widths))
            encode, decode = rotation.T, rotation
        if self._mode == "prod":
            projection = torch.block_diag(*(tq_math.qjl_projection(self._seed, w) for w in widths))
            qjl_encode, qjl_decode = projection.T, projection @ rotation
        else:
            qjl_encode, qjl_decode = encode, decode
        parameters = (
            encode.to(device=device, dtype=torch.float16).contiguous(),
            decode.to(device=device, dtype=torch.bfloat16).contiguous(),
            qjl_encode.to(device=device, dtype=torch.float16).contiguous(),
            qjl_decode.to(device=device, dtype=torch.bfloat16).contiguous(),
        )
        self._parameters[key] = parameters
        return parameters

    def _quantized_launch(
        self,
        geometry: _TurboQuantGeometry,
        entries: list[tuple[int, int, int]],
        device: torch.device,
    ) -> _TurboQuantLaunch:
        from . import turboquant_kernels as kernels

        encode_w, decode_w, encode_s, decode_s = self._group_parameters(geometry, device)
        element_bytes = geometry.element_bytes
        hot_base = min(base for base, _, _ in entries)
        table = torch.tensor(
            [
                ((base - hot_base) // element_bytes, slot_bytes // element_bytes, cold_offset)
                for base, slot_bytes, cold_offset in entries
            ],
            dtype=torch.int64,
            device=device,
        )
        launch = _TurboQuantLaunch(
            num_buffers=len(entries),
            table=table,
            encode_args=(table, encode_w, encode_s),
            decode_args=(table, decode_w, decode_s),
            encode_constants=dict(geometry.kernel_constants(encode=True), PASS=-1),
            decode_constants=geometry.kernel_constants(encode=False),
            hot_base=hot_base,
        )
        # Compile now: configure() runs at KVCM construction, not on migration. The pointer
        # arguments are compiled for 16-byte aligned addresses, which every launch checks.
        pages = torch.zeros(2, dtype=torch.int32, device=device)
        cold = torch.empty(_ALIGNMENT, dtype=torch.uint8, device=device)
        hot = torch.empty(_ALIGNMENT, dtype=_TORCH_DTYPES[geometry.runtime_type], device=device)
        sample = (pages, cold, 1 << 20, 1 << 20, hot)
        launch.encode_kernel = kernels._tq_encode_kernel.warmup(
            *sample, *launch.encode_args, grid=(1, 1, 1), **launch.encode_constants
        )
        launch.decode_kernels = tuple(
            kernels._tq_decode_kernel.warmup(
                *sample,
                *launch.decode_args,
                grid=(1, 1, 1),
                **dict(launch.decode_constants, PASS=decode_pass),
            )
            for decode_pass in (range(len(geometry.instances)) if geometry.blocked else (-1,))
        )
        # Load the modules now, so that an unsupported geometry fails here, not in a migration.
        for kernel in (launch.encode_kernel, *launch.decode_kernels):
            kernel._init_handles()
        return launch

    def _copy_launch(
        self, entries: list[tuple[int, int, int, int]], device: torch.device
    ) -> _TurboQuantLaunch:
        import triton.language as tl

        from . import turboquant_kernels as kernels

        word_bytes = 4 if all(value % 4 == 0 for entry in entries for value in entry) else 1
        rows = [(base, slot, raw // word_bytes, offset) for base, slot, raw, offset in entries]
        table = torch.tensor(rows, dtype=torch.int64, device=device)
        block = 2048
        constants = dict(
            UNIT=tl.int32 if word_bytes == 4 else tl.uint8,
            BLOCK=block,
            ROWS=-(-max(row[2] for row in rows) // block),
            num_warps=4,
        )
        launch = _TurboQuantLaunch(
            num_buffers=len(entries),
            table=table,
            encode_args=(table,),
            decode_args=(table,),
            encode_constants=constants,
            decode_constants=constants,
            copy=True,
        )
        launch.encode_kernel, decode_kernel = (
            kernels._copy_kernel.warmup(
                _ADDRESS,
                _ADDRESS,
                1 << 20,
                table,
                grid=(1, 1, 1),
                ENCODE=encode,
                UNIT=constants["UNIT"],
                BLOCK=block,
                num_warps=4,
            )
            for encode in (True, False)
        )
        launch.decode_kernels = (decode_kernel,)
        return launch

    def _launch(
        self,
        codec_state: _TurboQuantColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
        encode: bool,
    ) -> None:
        if num_pages == 0:
            return
        if cold_base % _ALIGNMENT:
            raise ValueError("TurboQuant cold pages need a 16-byte aligned cold base address")
        metadata = codec_state.lifecycle_metadata[lifecycle_index]
        for launch in metadata.launches:
            kernels = (launch.encode_kernel,) if encode else launch.decode_kernels
            if launch.copy:
                grid = (num_pages, launch.num_buffers, launch.encode_constants["ROWS"])
                kernels[0][grid](
                    page_indices,
                    cold_base,
                    metadata.cold_page_bytes,
                    *launch.encode_args,
                    stream=stream,
                )
                continue
            constants = launch.encode_constants if encode else launch.decode_constants
            page_programs, tiles = _TurboQuantGeometry.grid(constants, num_pages)
            args = launch.encode_args if encode else launch.decode_args
            for kernel in kernels:
                kernel[(page_programs, launch.num_buffers, tiles)](
                    page_indices,
                    cold_base,
                    metadata.cold_page_bytes,
                    num_pages,
                    launch.hot_base,
                    *args,
                    stream=stream,
                )

    def encode_cold_pages(
        self,
        codec_state: _TurboQuantColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._launch(
            codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream, encode=True
        )

    def decode_cold_pages(
        self,
        codec_state: _TurboQuantColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._launch(
            codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream, encode=False
        )
