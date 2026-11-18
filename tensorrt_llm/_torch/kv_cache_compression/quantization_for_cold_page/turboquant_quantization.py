# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TurboQuant cold-page layout, metadata, and Triton kernel dispatch."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from tensorrt_llm.math_utils import pad_up

from ...pyexecutor.resource_manager import DataType
from . import turboquant_math as tq_math
from .quantization_for_cold_page import (
    _DEEPSEEK_V4_COMPRESS,
    _DEEPSEEK_V4_CSA_ROLES,
    _DEEPSEEK_V4_HCA_ROLES,
    _DEEPSEEK_V4_NOPE_DIM,
    _DEEPSEEK_V4_ROPE_DIM,
    _DEEPSEEK_V4_ROW_STRIDE,
    _RUNTIME_TYPES,
    ColdPageQuantizationCompression,
    _ColdPageBufferLayout,
    _ColdPageCodecState,
    _ColdPageLayerLayout,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

_TORCH_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
_ALIGNMENT = 16
_MAX_PIECES = 4
_MAX_INSTANCE_WIDTH = 512
_INSTANCE_WIDTH = 256
_BIT_PLANES = {1: (1, 0), 2: (2, 0), 3: (2, 1), 4: (4, 0)}
_LUT_STRIDE = 16  # fp32 entries per codebook in the kernels' centroid table (LUT_STRIDE)
_MODES = {"mse": 0, "fractional": 1}
# A 64-bit stand-in for the KVCM addresses when kernels are compiled ahead of use.
_ADDRESS = 1 << 40


def _centroid_table(constants: dict[str, Any], device: torch.device) -> torch.Tensor:
    """The kernels' centroid table: signed codebooks 0 and 1 in code order, fp32.

    Built from the positive centroids of the CB constants and their negations, so a lookup returns
    exactly the value the symmetric codebook gives a code.
    """
    table = torch.zeros(2, _LUT_STRIDE, dtype=torch.float32)
    for row, name in enumerate("01"):
        half = 1 << (constants[f"BITS{name}"] - 1)
        positive = torch.tensor(constants[f"CB{name}"][7 : 7 + half], dtype=torch.float32)
        table[row, : 2 * half] = torch.cat((-positive.flip(0), positive))
    return table.to(device)


def _power_of_two_pieces(width: int) -> tuple[int, ...]:
    """Split ``width`` into descending powers of two, e.g. 576 -> (512, 64)."""
    return tuple(1 << bit for bit in reversed(range(width.bit_length())) if width >> bit & 1)


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
    def blocked(self) -> bool:
        """Whether an mse row wider than 512 is split into independent TurboQuant instances.

        Decoding accumulates every output value of an instance on chip (at most 512 fp32
        columns of tensor memory, shared with the staged operand), so such a row is coded as
        pieces of 256 values and the rest, decoded on tensor cores.
        """
        return self.mode == "mse" and self.width > _MAX_INSTANCE_WIDTH

    @property
    def pieces(self) -> tuple[int, ...]:
        """Power-of-two widths in which the kernels load a row (and, if blocked, encode it)."""
        if self.blocked:
            full, rest = divmod(self.width, _INSTANCE_WIDTH)
            return (_INSTANCE_WIDTH,) * full + ((rest,) if rest else ())
        return _power_of_two_pieces(self.width)

    @property
    def groups(self) -> tuple[tuple[int, int], ...]:
        """(width, bits) of every coded group of a row, in coordinate order."""
        if self.mode == "fractional":
            return tuple((self.width // 2, bits) for bits in self.group_bits)
        # One set of bit planes; blocked instances own consecutive chunks of it.
        return ((self.width, self.group_bits[0]),)

    @property
    def instances(self) -> tuple[int, ...]:
        """Width of every TurboQuant instance of a row (each has its own scale)."""
        if self.blocked:
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
                cursor += pad_up(self.rows * width * plane_bits // 8, _ALIGNMENT)
        # One BF16 scale per row and instance.
        offsets["scale"] = cursor
        cursor += pad_up(self.rows * 2 * len(self.instances), _ALIGNMENT)
        if self.mode == "fractional":
            offsets["mask"] = cursor
            cursor += pad_up(self.rows // self.tokens_per_page * self.width // 8, _ALIGNMENT)
        offsets["raw"] = cursor
        cursor += pad_up(self.rows * self.raw_elements * self.element_bytes, _ALIGNMENT)
        offsets["end"] = cursor
        return offsets

    def kernel_constants(
        self, encode: bool, capability: tuple[int, int] = (0, 0)
    ) -> dict[str, Any]:
        """Compile-time arguments of the encode or decode kernel for this geometry.

        ``capability`` is the GPU's compute capability; the launch shapes measured on SM100-family
        GPUs apply only there. Decode constants carry ``HALVES``: whether a single-instance row
        accumulates its output columns in two halves (see ``decode_passes``).
        """
        import triton
        import triton.language as tl

        pieces = self.pieces
        chunk = min(min(width & -width for width in self.instances), 128)
        pages_per_tile, rows_pad = 1, self.rows
        launch: dict[str, int] = {}
        if self.mode == "fractional":
            # The tokens of one KV head share the channel split: an encode tile is one head.
            block_m = max(16, triton.next_power_of_2(self.tokens_per_page))
            tile_rows = self.tokens_per_page
            # Encoding stages the head's widest piece (fp16) and a [piece, C] chunk of the
            # rotation in shared memory (227 KiB per block on GB300): keep them within 192 KiB.
            # C also lays out the codes, so encode and decode derive it from the head alone.
            while chunk > 16 and 2 * max(pieces) * (block_m + chunk) > 192 * 1024:
                chunk //= 2
            if not encode and self.width > 448 and block_m > 32:
                # Decoding accumulates every output value of the row, and tensor-core tiles of 64+
                # rows accumulate in tensor memory (512 fp32 columns, shared with the staged
                # operand): rows wider than 448 do not fit (on GB300 the decode faults with a
                # misaligned address). Such heads decode in tiles of at most 32 rows, whose
                # accumulators live in registers; a tile is a power of two dividing the head.
                tile_rows = min(32, self.tokens_per_page & -self.tokens_per_page)
                block_m = max(16, tile_rows)
        else:
            # Tensor-core tiles of 64+ rows accumulate in tensor memory: 512 fp32 columns shared
            # by the accumulators and the staged operand. Decoding accumulates a whole instance;
            # wider ones use 32-row tiles, whose accumulators live in registers.
            accumulated = max(self.instances)
            if encode and self.width == 128:
                # Small tiles keep more programs in flight; encoding is bound by the host link.
                block_m = 32
            elif encode or accumulated <= 128:
                block_m = 128 if self.width <= 128 else 64
            elif accumulated <= 448:
                block_m = 64
            else:
                block_m = 32
            launch = self._wide_mse_launch(encode, accumulated, capability)
            block_m = launch.pop("block_m", block_m)
            tile_rows = block_m
            if self.rows < block_m:
                rows_pad = triton.next_power_of_2(self.rows)
                pages_per_tile = block_m // rows_pad
        regions = self.regions()
        groups = self.groups
        planes_a = _BIT_PLANES[groups[0][1]]
        planes_b = _BIT_PLANES[groups[-1][1]] if len(groups) > 1 else (0, 0)
        codebooks = {}
        if self.blocked:  # instance 0 and those of its width use CB0, the last one CB1
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
        halves = launch.pop("halves", False)
        constants = dict(
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
            OFF_SCALE=regions["scale"],
            OFF_MASK=regions.get("mask", 0),
            OFF_RAW=regions["raw"],
            RAW=self.raw_elements,
            RAW_BLOCK=triton.next_power_of_2(max(self.raw_elements, 1)),
            **{
                "num_warps": 4 if block_m * self.width <= 4096 else 8,
                # Wide rows keep their pieces resident; pipelining their matrix tiles would not fit.
                "num_stages": 1 if self.width > 256 else 2,
                **launch,
            },
        )
        if not encode:
            constants["HALVES"] = halves
            # Rows per KV head, for the fractional channel mask of a decode tile.
            constants["HEAD_ROWS"] = self.tokens_per_page if self.mode == "fractional" else 1
        return constants

    def decode_passes(self, halves: bool) -> tuple[tuple[int, bool], ...]:
        """(PASS, HALVES) of every decode launch of a row.

        Blocked rows decode one instance per pass (PASS), every other row in one pass (-1). With
        ``halves``, a row that is a single mse instance accumulates its output columns in two
        halves; each chunk of its codes is still read once.
        """
        if self.blocked:
            return tuple((decode_pass, False) for decode_pass in range(len(self.instances)))
        single = self.mode == "mse" and len(self.pieces) == 1
        return ((-1, bool(halves) and single),)

    def _wide_mse_launch(
        self, encode: bool, accumulated: int, capability: tuple[int, int]
    ) -> dict[str, int]:
        """Tiles, warps, stages, register cap and decode column halves of mse rows wider than 256.

        Measured on GB300 (SM103) and used on SM100-family GPUs only; other GPUs keep the generic
        shapes. MLA latents (576 values, three instances) and DeepSeek-V4 CSA rows (448) encode
        fastest with 64-row tiles of 8 warps capped at 128 registers, so that two programs share
        an SM; blocked rows also pipeline their matrix tiles. Decoding takes 128-row tiles; a
        single instance wider than 448 (512 values) accumulates its output columns in two halves
        of 256, whose accumulators fit tensor memory.
        """
        if self.mode != "mse" or self.width <= 256 or capability[0] != 10:
            return {}
        if encode:
            if self.blocked:
                return dict(block_m=64, num_warps=8, num_stages=2, maxnreg=128)
            if accumulated <= 448:
                return dict(block_m=64, num_warps=8, num_stages=1, maxnreg=128)
            return {}
        if self.blocked:
            return dict(block_m=128, num_warps=4, num_stages=1)
        if accumulated <= 448:
            return dict(block_m=128, num_warps=8, num_stages=1)
        if len(self.pieces) == 1:
            return dict(block_m=128, num_warps=8, num_stages=1, halves=True)
        return dict(block_m=32, num_warps=8, num_stages=2)

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


class TurboQuantColdPageQuantizationCompression(ColdPageQuantizationCompression):
    """TurboQuant cold pages (arXiv 2504.19874): rotation and Lloyd-Max codebooks (mse)."""

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
        bits = float(config.turboquant_bits)
        self._seed = int(config.turboquant_seed)
        if bits != int(bits):
            # Outlier channels get the next bit width up, regular channels the one below.
            self._mode = "fractional"
            self._group_bits = (math.ceil(bits), math.floor(bits))
        else:
            self._mode = "mse"
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
    ) -> _ColdPageCodecState:
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

        skip_rope = self._skip_rope(is_draft)

        layouts: dict[int, _ColdPageLayerLayout] = {}
        deepseek_v4_layers: set[int] = set()
        if getattr(self.pretrained_config, "model_type", None) == "deepseek_v4":
            deepseek_v4_layouts = self._build_deepseek_v4_layer_layouts(
                cache_config,
                attention_layers=attention_layers,
                pp_layers=pp_layers,
                runtime_type=runtime_type,
                skip_rope=skip_rope,
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
                _ColdPageBufferLayout(
                    role=role,
                    quantized_range_start=key_range[0] if role == "key" else 0,
                    quantized_range_elements=key_range[1] if role == "key" else head_dim,
                )
                for role in compressed_roles
            ]
            for buffer in buffers:
                self._check_quantized_width(buffer.quantized_range_elements, layer_id, buffer.role)
            buffers += [
                _ColdPageBufferLayout(role=str(buffer.role))
                for buffer in layer.buffers
                if str(buffer.role) not in compressed_roles
            ]
            layouts[layer_id] = _ColdPageLayerLayout(
                layer_id=layer_id,
                num_kv_heads=int(num_kv_heads_per_layer[layer_id]),
                tokens_per_page=int(cache_config.tokens_per_block),
                raw_row_stride_elements=head_dim,
                buffers=tuple(buffers),
            )

        return _ColdPageCodecState(
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
                f"TurboQuant cold pages support fewer than {2 * _MAX_INSTANCE_WIDTH} quantized "
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
        skip_rope: bool,
    ) -> dict[int, _ColdPageLayerLayout | None]:
        """CSA rows: the 448 NoPE values, and the 64 RoPE values unless ``skip_rope``, are encoded;
        the indexer cache is kept unchanged. None selects the lossless fallback."""

        roles_by_layer, has_csa_cache, _ = self._deepseek_v4_model_layers(
            attention_layers, pp_layers
        )
        layouts: dict[int, _ColdPageLayerLayout | None] = {}
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            buffer_roles = roles_by_layer.get(layer_id)
            if buffer_roles is None:
                continue
            if buffer_roles != _DEEPSEEK_V4_CSA_ROLES:
                layouts[layer_id] = None
                if has_csa_cache and buffer_roles == _DEEPSEEK_V4_HCA_ROLES:
                    # Co-located with the CSA cache in one lifecycle: owned, but stored as is.
                    layouts[layer_id] = _ColdPageLayerLayout(
                        layer_id=layer_id,
                        num_kv_heads=0,
                        tokens_per_page=0,
                        raw_row_stride_elements=0,
                        buffers=tuple(
                            _ColdPageBufferLayout(role=str(buffer.role)) for buffer in layer.buffers
                        ),
                    )
                continue
            tokens_per_page = self._deepseek_v4_csa_tokens_per_page(
                cache_config, layer, runtime_type=runtime_type
            )
            # A compressed row is 448 NoPE elements followed by 64 RoPE elements.
            range_start, range_elements = self._calculate_quantized_range(
                _DEEPSEEK_V4_ROW_STRIDE,
                skip_rope=skip_rope,
                rope=(_DEEPSEEK_V4_NOPE_DIM, _DEEPSEEK_V4_ROPE_DIM),
                buffer_name=f"cold-page layer {layer_id} {_DEEPSEEK_V4_COMPRESS}",
            )
            self._check_quantized_width(range_elements, layer_id, _DEEPSEEK_V4_COMPRESS)
            layouts[layer_id] = _ColdPageLayerLayout(
                layer_id=layer_id,
                num_kv_heads=1,
                tokens_per_page=tokens_per_page,
                raw_row_stride_elements=_DEEPSEEK_V4_ROW_STRIDE,
                buffers=tuple(
                    _ColdPageBufferLayout(
                        role=str(buffer.role),
                        quantized_range_start=range_start,
                        quantized_range_elements=range_elements,
                    )
                    if str(buffer.role) == _DEEPSEEK_V4_COMPRESS
                    else _ColdPageBufferLayout(role=str(buffer.role))
                    for buffer in layer.buffers
                ),
            )
        return layouts

    def _page_index_location(self, native: Any) -> object:
        return native.ColdPageIndexLocation.DEVICE

    def build_lifecycle_metadata(
        self, codec_state: _ColdPageCodecState, lifecycle: object
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
                    cursor += pad_up(raw_bytes, _ALIGNMENT)
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
        cold_page_bytes = pad_up(cursor, _ALIGNMENT)
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
        parameters = (
            encode.to(device=device, dtype=torch.float16).contiguous(),
            decode.to(device=device, dtype=torch.bfloat16).contiguous(),
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

        encode_w, decode_w = self._group_parameters(geometry, device)
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
        capability = torch.cuda.get_device_capability(device)
        encode_constants = dict(geometry.kernel_constants(True, capability), PASS=-1)
        decode_constants = geometry.kernel_constants(False, capability)
        passes = geometry.decode_passes(decode_constants.pop("HALVES"))
        centroids = _centroid_table(encode_constants, device)
        launch = _TurboQuantLaunch(
            num_buffers=len(entries),
            table=table,
            encode_args=(table, encode_w, centroids),
            decode_args=(table, decode_w, centroids),
            encode_constants=encode_constants,
            decode_constants=decode_constants,
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
                **dict(launch.decode_constants, PASS=decode_pass, HALVES=halves),
            )
            for decode_pass, halves in passes
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
        # As for the quantized launches: a module that cannot load fails here, not in a migration.
        for kernel in (launch.encode_kernel, decode_kernel):
            kernel._init_handles()
        return launch

    def _launch(
        self,
        codec_state: _ColdPageCodecState,
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
        codec_state: _ColdPageCodecState,
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
        codec_state: _ColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._launch(
            codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream, encode=False
        )
