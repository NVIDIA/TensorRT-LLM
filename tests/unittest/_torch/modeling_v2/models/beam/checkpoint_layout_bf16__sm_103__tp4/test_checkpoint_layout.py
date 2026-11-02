# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated-data tests for the Beam checkpoint layout component: bf16 / sm_103 / tp4.

No model files. The layout tests use the architecture's configuration values only (no
tensors). The loader tests write small safetensors directories with this file's own writer
and compare every loaded byte with an expected value derived independently here: E4M3
decoded from its bit fields, the product with the BF16 row scale formed exactly, and the
BF16 rounding done on the float32 bit pattern (round to nearest, ties to even). Every rank
of the four-rank layout is exercised, with tiles small enough to cross row seams.

Four-rank execution (collectives, real checkpoint loading, KV cache) needs four GPUs and a
checkpoint and is not part of this file.

Invocation::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE \\
        python -m pytest tests/unittest/_torch/modeling_v2/models/beam/checkpoint_layout_bf16__sm_103__tp4
"""

from __future__ import annotations

import json
import math
import os
import struct

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    layout_bf16 as layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    loader_bf16 as loader,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.layout_bf16 import (  # noqa: E501
    BF16,
    F32,
    FP8,
    TensorSpec,
)

NUM_LAYERS = 52
FULL = {5, 11, 17, 23, 29, 35, 41, 47, 51}


def model_config() -> dict:
    """The architecture's configuration, as its config.json states it (values only)."""
    cfg = dict(layout.SUPPORTED_CONFIG)
    cfg["layer_types"] = [
        "full_attention" if i in FULL else "sliding_attention" for i in range(NUM_LAYERS)
    ]
    cfg["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (NUM_LAYERS - 3)
    cfg["rope_parameters"] = {
        "full_attention": None,
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    return cfg


# ------------------------------------------------------------------ independent expected values


def e4m3(code: int) -> float:
    """E4M3 (fn) value of one byte from its bit fields: bias 7, subnormals, no infinities."""
    sign = -1.0 if code & 0x80 else 1.0
    exp, man = (code >> 3) & 0xF, code & 0x7
    if exp == 0xF and man == 0x7:
        return math.nan
    if exp == 0:
        return sign * man / 8.0 * 2.0**-6
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def bf16_bits_of_float(x: float) -> int:
    """BF16 bits of the float32 value nearest ``x`` (exact here), rounded to nearest even."""
    (u,) = struct.unpack("<I", struct.pack("<f", x))
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def write_checkpoint(path: str, shards: dict) -> None:
    """``shards``: file name -> list of (name, dtype, shape, raw bytes). Writes index + shards."""
    os.makedirs(path, exist_ok=True)
    weight_map = {}
    for filename, entries in shards.items():
        header, payload, offset = {}, b"", 0
        for name, dtype, shape, raw in entries:
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            payload += raw
            offset += len(raw)
            weight_map[name] = filename
        blob = json.dumps(header).encode()
        blob += b" " * (-len(blob) % 8)
        with open(os.path.join(path, filename), "wb") as f:
            f.write(struct.pack("<Q", len(blob)) + blob + payload)
    with open(os.path.join(path, loader.INDEX_FILE), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


FP8_CODES = [c for c in range(256) if c not in (0x7F, 0xFF)]


def fp8_matrix(rows: int, cols: int, start: int) -> list:
    """Codes cycling through every finite E4M3 byte (signs, zeros, subnormals, 448)."""
    return [
        [FP8_CODES[(start + r * cols + c) % len(FP8_CODES)] for c in range(cols)]
        for r in range(rows)
    ]


def row_scales(rows: int, start: int) -> list:
    """Distinct, non-uniform BF16 per-row scales (bits), as a quantizer's amax/448 would be."""
    return [
        bf16_bits_of_float(2.0 ** (-(r % 5) - 3) * (1.0 + (start + 3 * r) % 7 / 8.0))
        for r in range(rows)
    ]


def fp8_entries(name: str, codes: list, scales: list) -> list:
    rows, cols = len(codes), len(codes[0])
    raw = bytes(c for row in codes for c in row)
    sraw = b"".join(struct.pack("<H", s) for s in scales)
    return [(name, FP8, (rows, cols), raw), (name + "_scale", BF16, (rows, 1), sraw)]


def expected_fp8(codes: list, scales: list) -> list:
    """BF16 bits of bf16(fp32(e4m3(q)) * fp32(scale[row])), independently of torch."""
    return [
        [bf16_bits_of_float(e4m3(q) * bf16_value(scales[r])) for q in row]
        for r, row in enumerate(codes)
    ]


# A generated layout: row-split and column-split FP8, replicated BF16, row-split F32, on two shards.
SPECS = {
    "a.rows.weight": TensorSpec("a.rows.weight", (8, 12), FP8, "gen", 0, None, 0),
    "b.cols.weight": TensorSpec("b.cols.weight", (12, 16), FP8, "gen", 0, None, 1),
    "c.norm.weight": TensorSpec("c.norm.weight", (10,), BF16, "gen", 0, None, None),
    "d.router.weight": TensorSpec("d.router.weight", (8, 4), F32, "gen", 0, None, 0),
}
A_CODES, A_SCALES = fp8_matrix(8, 12, 0), row_scales(8, 0)
B_CODES, B_SCALES = fp8_matrix(12, 16, 97), row_scales(12, 5)
C_BITS = [
    bf16_bits_of_float(v)
    for v in (-0.0, 1.5, -2.25, 3.0e-40, 448.0, 1.0 / 3, 7.0, -1e5, 0.0, 65504.0)
]
D_VALUES = [((-1) ** i) * (i + 0.5) / 3.0 for i in range(32)]


def generated_shards() -> dict:
    return {
        "model-00001-of-00002.safetensors": fp8_entries("a.rows.weight", A_CODES, A_SCALES)
        + [("c.norm.weight", BF16, (10,), b"".join(struct.pack("<H", b) for b in C_BITS))],
        "model-00002-of-00002.safetensors": fp8_entries("b.cols.weight", B_CODES, B_SCALES)
        + [("d.router.weight", F32, (8, 4), struct.pack("<32f", *D_VALUES))],
    }


@pytest.fixture
def checkpoint(tmp_path) -> str:
    path = str(tmp_path / "ckpt")
    write_checkpoint(path, generated_shards())
    return path


def _bits16(t: torch.Tensor) -> list:
    return t.contiguous().view(torch.int16).cpu().to(torch.int32).bitwise_and(0xFFFF).tolist()


# ------------------------------------------------------------------ layout


def test_layout_counts_intervals_and_banks() -> None:
    specs = layout.tensor_specs(model_config())
    assert len(specs) == 3 + 3 * 15 + 49 * 782
    fp8 = [s for s in specs.values() if s.stored_dtype == FP8]
    assert len(fp8) == 52 * 5 + 3 * 3 + 49 * 768
    layout.check_partition(specs)
    want = {
        "model.embed_tokens.weight": (0, 37984),
        "lm_head.weight": (0, 37984),
        "model.layers.0.self_attn.q_proj.weight": (0, 1536),
        "model.layers.0.self_attn.gate_proj.weight": (0, 1536),
        "model.layers.0.self_attn.k_proj.weight": (0, 256),
        "model.layers.0.self_attn.v_proj.weight": (0, 256),
        "model.layers.0.self_attn.o_proj.weight": (1, 1536),
        "model.layers.0.self_attn.gate_bias": (0, 12),
        "model.layers.0.mlp.gate_proj.weight": (0, 5120),
        "model.layers.0.mlp.down_proj.weight": (1, 5120),
        "model.layers.51.mlp.experts.255.up_proj.weight": (0, 640),
        "model.layers.51.mlp.experts.255.down_proj.weight": (1, 640),
    }
    for name, (dim, size) in want.items():
        spec = specs[name]
        got = [spec.interval(r) for r in range(4)]
        assert got == [(dim, r * size, (r + 1) * size) for r in range(4)], name
    for name in (
        "model.norm.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.self_attn.q_norm.weight",
    ):
        assert all(specs[name].interval(r) is None for r in range(4))
    router = specs["model.layers.3.mlp.gate.e_score_correction_bias"]
    assert router.stored_dtype == F32 and router.effective_dtype == F32
    expert = specs["model.layers.7.mlp.experts.200.down_proj.weight"]
    assert expert.bank == "layers.7.experts.down" and expert.expert == 200
    assert expert.local_shape(2) == (5120, 640) and expert.scale_name.endswith(
        "down_proj.weight_scale"
    )
    assert (
        specs["lm_head.weight"].stored_dtype == BF16 and specs["lm_head.weight"].scale_name is None
    )


def test_gqa_pairing_and_ledger_rows() -> None:
    specs = layout.tensor_specs(model_config(), layers=(5,))
    q, k = (
        specs["model.layers.5.self_attn.q_proj.weight"],
        specs["model.layers.5.self_attn.k_proj.weight"],
    )
    for rank in range(4):
        _, q0, q1 = q.interval(rank)
        _, k0, k1 = k.interval(rank)
        q_heads = range(q0 // 128, q1 // 128)
        kv_heads = set(range(k0 // 128, k1 // 128))
        assert {h // 6 for h in q_heads} == kv_heads == {2 * rank, 2 * rank + 1}
    rows = {row["name"]: row for row in layout.ownership_ledger(specs)}
    assert len(rows) == len(specs)
    o = rows["model.layers.5.self_attn.o_proj.weight"]["ranks"]
    assert [(r["dim"], r["start"], r["stop"]) for r in o] == [
        (1, 1536 * i, 1536 * (i + 1)) for i in range(4)
    ]
    bank = rows["model.layers.5.mlp.experts.9.gate_proj.weight"]["ranks"][3]
    assert (bank["bank"], bank["slot"], bank["start"], bank["stop"]) == (
        "layers.5.experts.gate",
        9,
        1920,
        2560,
    )
    assert all(r["replicated"] for r in rows["model.layers.5.mlp.gate.weight"]["ranks"])


@pytest.mark.parametrize(
    "change",
    [
        {"num_experts": 255},
        {"routed_scaling_factor": 2.0},
        {"tie_word_embeddings": True},
        {"mlp_layer_types": ["sparse"] + ["dense"] * 2 + ["sparse"] * 49},
        {"layer_types": ["sliding_attention"] * 51},
        {
            "rope_parameters": {
                "full_attention": None,
                "sliding_attention": {"rope_theta": 1e6, "rope_type": "default"},
            }
        },
        {"num_key_value_heads": 4},
    ],
)
def test_inconsistent_configuration_is_rejected(change) -> None:
    cfg = model_config()
    cfg.update(change)
    with pytest.raises(ValueError):
        layout.tensor_specs(cfg)


@pytest.mark.parametrize("rank, world", [(4, 4), (-1, 4), (0, 2), (0, 8), (True, 4), (1.0, 4)])
def test_invalid_rank_or_world_is_rejected(rank, world) -> None:
    with pytest.raises(ValueError):
        layout.check_rank(rank, world)
    spec = layout.tensor_specs(model_config(), layers=(0,))[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    with pytest.raises(ValueError):
        spec.interval(rank, world)


def test_unknown_layers_are_rejected() -> None:
    for layers in ((52,), (-1,), (3, 3), ("3",)):
        with pytest.raises(ValueError):
            layout.tensor_specs(model_config(), layers=layers)


# ------------------------------------------------------------------ catalog validation


def test_catalog_accepts_the_exact_checkpoint(checkpoint) -> None:
    catalog = loader.SafetensorsCatalog(checkpoint)
    catalog.validate(SPECS, complete=True)
    assert set(catalog.entries) == set(SPECS) | {"a.rows.weight_scale", "b.cols.weight_scale"}


def _mutated(entries_by_file: dict, how: str) -> dict:
    shards = {f: list(e) for f, e in entries_by_file.items()}
    first, second = list(shards)
    if how == "missing_scale":
        shards[first] = [e for e in shards[first] if e[0] != "a.rows.weight_scale"]
    elif how == "extra_tensor":
        shards[second].append(("e.unsupported.weight", BF16, (2,), b"\0" * 4))
    elif how == "scale_vector":
        shards[first] = [
            (n, d, (8,) if n == "a.rows.weight_scale" else s, raw) for n, d, s, raw in shards[first]
        ]
    elif how == "wrong_dtype":
        shards[second] = [
            (n, BF16 if n == "d.router.weight" else d, (8, 8) if n == "d.router.weight" else s, raw)
            for n, d, s, raw in shards[second]
        ]
    elif how == "wrong_shape":
        shards[first] = [
            (n, d, (12, 8) if n == "a.rows.weight" else s, raw) for n, d, s, raw in shards[first]
        ]
    elif how == "transposed_scale":
        shards[second] = [
            (n, d, (1, 12) if n.endswith("_scale") else s, raw) for n, d, s, raw in shards[second]
        ]
    return shards


@pytest.mark.parametrize(
    "how",
    [
        "missing_scale",
        "extra_tensor",
        "scale_vector",
        "wrong_dtype",
        "wrong_shape",
        "transposed_scale",
    ],
)
def test_catalog_rejects_malformed_metadata(tmp_path, how) -> None:
    path = str(tmp_path / how)
    write_checkpoint(path, _mutated(generated_shards(), how))
    with pytest.raises(loader.CheckpointFormatError):
        loader.SafetensorsCatalog(path).validate(SPECS, complete=True)


def test_catalog_rejects_index_header_disagreement_and_truncation(checkpoint) -> None:
    index_path = os.path.join(checkpoint, loader.INDEX_FILE)
    index = json.load(open(index_path))
    index["weight_map"]["c.norm.weight"] = "model-00002-of-00002.safetensors"
    json.dump(index, open(index_path, "w"))
    with pytest.raises(loader.CheckpointFormatError):
        loader.SafetensorsCatalog(checkpoint)
    write_checkpoint(checkpoint, generated_shards())
    shard = os.path.join(checkpoint, "model-00002-of-00002.safetensors")
    with open(shard, "rb+") as f:
        f.truncate(os.path.getsize(shard) - 3)
    with pytest.raises(loader.CheckpointFormatError):
        loader.SafetensorsCatalog(checkpoint)


# ------------------------------------------------------------------ streaming reads on CUDA

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@cuda
@pytest.mark.parametrize("tile_bytes", [16, 40, loader.TILE_BYTES])
def test_fp8_slices_match_independent_bytes_on_every_rank(checkpoint, tile_bytes) -> None:
    catalog = loader.SafetensorsCatalog(checkpoint)
    catalog.validate(SPECS)
    reader = loader.TileReader(tile_bytes=tile_bytes, pin=True)
    want_a, want_b = expected_fp8(A_CODES, A_SCALES), expected_fp8(B_CODES, B_SCALES)
    try:
        for rank in range(4):
            a = torch.empty(
                SPECS["a.rows.weight"].local_shape(rank), dtype=torch.bfloat16, device="cuda"
            )
            loader.read_slice_into(catalog, reader, SPECS["a.rows.weight"], rank, a)
            assert _bits16(a) == want_a[2 * rank : 2 * rank + 2]
            b = torch.empty(
                SPECS["b.cols.weight"].local_shape(rank), dtype=torch.bfloat16, device="cuda"
            )
            loader.read_slice_into(catalog, reader, SPECS["b.cols.weight"], rank, b)
            assert _bits16(b) == [row[4 * rank : 4 * rank + 4] for row in want_b]
    finally:
        reader.close()


@cuda
def test_fp8_rule_is_a_row_scale_not_a_cast_or_transpose(checkpoint) -> None:
    """The fixtures distinguish the contract from a plain cast, a reciprocal and a column scale."""
    want = expected_fp8(B_CODES, B_SCALES)
    plain = [[bf16_bits_of_float(e4m3(q)) for q in row] for row in B_CODES]
    recip = [
        [bf16_bits_of_float(e4m3(q) / bf16_value(B_SCALES[r])) for q in row]
        for r, row in enumerate(B_CODES)
    ]
    assert want != plain and want != recip
    catalog = loader.SafetensorsCatalog(checkpoint)
    reader = loader.TileReader(tile_bytes=64)
    try:
        b = torch.empty(SPECS["b.cols.weight"].local_shape(3), dtype=torch.bfloat16, device="cuda")
        loader.read_slice_into(catalog, reader, SPECS["b.cols.weight"], 3, b)
    finally:
        reader.close()
    assert _bits16(b) == [row[12:16] for row in want]


@cuda
def test_bf16_and_f32_tensors_copy_byte_for_byte(checkpoint) -> None:
    catalog = loader.SafetensorsCatalog(checkpoint)
    reader = loader.TileReader(tile_bytes=24)
    try:
        for rank in range(4):
            c = torch.empty(10, dtype=torch.bfloat16, device="cuda")
            loader.read_slice_into(catalog, reader, SPECS["c.norm.weight"], rank, c)
            assert _bits16(c) == C_BITS
            d = torch.empty(2, 4, dtype=torch.float32, device="cuda")
            loader.read_slice_into(catalog, reader, SPECS["d.router.weight"], rank, d)
            want = torch.tensor(D_VALUES, dtype=torch.float32).view(8, 4)[2 * rank : 2 * rank + 2]
            assert torch.equal(d.cpu().view(torch.int32), want.view(torch.int32))
    finally:
        reader.close()


@cuda
def test_wrong_destination_and_short_read_raise(checkpoint) -> None:
    catalog = loader.SafetensorsCatalog(checkpoint)
    reader = loader.TileReader(tile_bytes=32)
    try:
        wrong = torch.empty(2, 12, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError):
            loader.read_slice_into(catalog, reader, SPECS["a.rows.weight"], 0, wrong)
        shard = os.path.join(checkpoint, "model-00002-of-00002.safetensors")
        with open(shard, "rb+") as f:
            f.truncate(os.path.getsize(shard) - 64)
        d = torch.empty(2, 4, dtype=torch.float32, device="cuda")
        with pytest.raises(OSError):
            loader.read_slice_into(catalog, reader, SPECS["d.router.weight"], 3, d)
    finally:
        reader.close()


# ------------------------------------------------------------------ component guards


def _sm103() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


@pytest.mark.skipif(not _sm103(), reason="the component is built for sm_103")
def test_mapping_rejections_and_weight_state_guard() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.component_bf16 import (  # noqa: E501
        BeamCheckpointLayoutBf16Sm103Tp4,
    )
    from tensorrt_llm.mapping import Mapping

    cfg = model_config()
    for bad in (
        dict(world_size=2, rank=0, tp_size=2, moe_tp_size=2),
        dict(world_size=4, rank=0, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    ):
        with pytest.raises(ValueError):
            BeamCheckpointLayoutBf16Sm103Tp4(
                cfg, Mapping(gpus_per_node=4, pp_size=1, **bad), layers=(0,)
            )
    mapping = Mapping(
        world_size=4, rank=1, gpus_per_node=4, tp_size=4, pp_size=1, moe_tp_size=4, moe_ep_size=1
    )
    model = BeamCheckpointLayoutBf16Sm103Tp4(cfg, mapping, layers=(0,))
    assert model.vocab_interval == (37984, 75968)
    assert tuple(model.tensor("model.layers.0.self_attn.k_proj.weight").shape) == (256, 5120)
    state = model.weight_state()
    with pytest.raises(RuntimeError):  # no complete load: unusable
        model.check_weight_state(state)
    with pytest.raises(RuntimeError):
        model.embed(torch.zeros(1, dtype=torch.int64, device="cuda"))
    # White-box: mark usable to exercise the guard without a four-rank load.
    model.usable = True
    model.check_weight_state(state)
    model.tensor("model.layers.0.input_layernorm.weight").add_(0)  # in place: version changes
    with pytest.raises(RuntimeError):
        model.check_weight_state(state)
    state = model.weight_state()
    name = "model__layers__0__self_attn__q_norm__weight"
    old = getattr(model, name)
    setattr(
        model, name, torch.nn.Parameter(old.detach().clone(), requires_grad=False)
    )  # same-shape replacement
    with pytest.raises(RuntimeError):
        model.check_weight_state(state)
