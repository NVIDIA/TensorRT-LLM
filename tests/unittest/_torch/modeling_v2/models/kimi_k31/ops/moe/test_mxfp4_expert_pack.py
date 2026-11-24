# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 MXFP4 routed-expert packing: the two kernel entries over their cells,
and the packer bit for bit against the engine's TRTLLM-Gen MXFP4 quant method at every MoE TP size
K3.1 admits.

The engine method runs on a stand-in module carrying the attributes it reads (the shapes, the
MoE-TP coordinates and the four expert tensors its ``create_weights`` registers).

``TLLM_K31_TEST_CHECKPOINT`` (optional) names a Kimi K3.1 checkpoint directory; its first routed
experts are then packed both ways as well.
"""

import json
import os
import re
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    mxfp4_expert_pack as pack,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod
from tensorrt_llm._utils import is_sm_100f

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or not is_sm_100f():
    pytest.skip("the TRTLLM-Gen MXFP4 layout is SM10x-only", allow_module_level=True)

#: K3.1 routed-expert geometry (latent hidden, unsharded intermediate).
HIDDEN = 3584
INTERMEDIATE = 3072

#: (MoE TP size, ranks) the packer is checked at. 1 / 2 / 4: the perf targets; 8: the general
#: target's last pad-then-shard size; 16 / 32: the shard-then-pad branch (192 / 96 values per rank).
TP_RANKS = (
    (1, (0,)),
    (2, (0, 1)),
    (4, (0, 3)),
    (8, (0, 5, 7)),
    (16, (0, 7, 15)),
    (32, (0, 31)),
)
LAYOUTS = [(tp, rank) for tp, ranks in TP_RANKS for rank in ranks]

shuffle_op = pack.k31_shuffle_matrix
interleave_op = pack.k31_block_scale_interleave


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


@pytest.mark.parametrize("cell", shuffle_op.CELLS, ids=[c.why[:60] for c in shuffle_op.CELLS])
def test_shuffle_matrix_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randint(
        0, 256, (spec["rows"], spec["cols"]), generator=g, dtype=torch.uint8, device="cuda"
    )
    if spec["kind"] == "w31":
        indices = pack._w3_w1_permute_indices(x, num_elts_per_sf=spec["sf"])
    else:
        indices = pack._w2_permute_indices(x, num_elts_per_sf=spec["sf"])
    with validating(shuffle_op):
        out = shuffle_op(x, indices)
    assert torch.equal(out, shuffle_op.reference(x, indices))
    # The permutation moves every row: identity indices would pass the comparison above too.
    assert not torch.equal(indices.cpu(), torch.arange(spec["rows"]))


@pytest.mark.parametrize("cell", interleave_op.CELLS, ids=[c.why[:60] for c in interleave_op.CELLS])
def test_block_scale_interleave_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randint(
        0, 256, (spec["rows"], spec["cols"]), generator=g, dtype=torch.uint8, device="cuda"
    )
    with validating(interleave_op):
        out = interleave_op(x)
    assert torch.equal(out, interleave_op.reference(x))


def _engine_module(num_local_experts: int, tp_size: int, tp_rank: int):
    """The engine method and a stand-in module after ``create_weights`` + ``cache_derived_state``,
    with its expert tensors zeroed on the GPU."""
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = torch.nn.Module()
    module.expert_size_per_partition = num_local_experts
    module.initial_local_expert_ids = list(range(num_local_experts))
    module.intermediate_size = INTERMEDIATE
    module.intermediate_size_per_partition = INTERMEDIATE // tp_size
    module.hidden_size = HIDDEN
    module.tp_size = tp_size
    module.tp_rank = tp_rank
    module.bias = False
    method.create_weights(module)
    method.cache_derived_state(module)
    module.cuda()
    for name in ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale"):
        getattr(module, name).data.zero_()
    return method, module


def _family_tensors(num_local_experts: int, tp_size: int) -> dict:
    shapes = pack.expert_weight_shapes(num_local_experts, INTERMEDIATE, HIDDEN, tp_size)
    return {
        name: torch.zeros(shape, dtype=pack.EXPERT_WEIGHT_DTYPE, device="cuda")
        for name, shape in shapes._asdict().items()
    }


def _random_experts(num_experts: int, seed: int) -> list:
    """Group-32 packed MXFP4 checkpoint experts (uint8, host), K3.1's per-expert shapes."""
    g = torch.Generator().manual_seed(seed)

    def u8(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)

    return [
        dict(
            w1_weight=u8(INTERMEDIATE, HIDDEN // 2),
            w1_weight_scale=u8(INTERMEDIATE, HIDDEN // 32),
            w3_weight=u8(INTERMEDIATE, HIDDEN // 2),
            w3_weight_scale=u8(INTERMEDIATE, HIDDEN // 32),
            w2_weight=u8(HIDDEN, INTERMEDIATE // 2),
            w2_weight_scale=u8(HIDDEN, INTERMEDIATE // 32),
        )
        for _ in range(num_experts)
    ]


def _pack_both(experts: list, tp_size: int, tp_rank: int, intermediate_size: int = INTERMEDIATE):
    """Pack ``experts`` into consecutive slots with the engine method and with the family packer."""
    method, module = _engine_module(len(experts), tp_size, tp_rank)
    family = _family_tensors(len(experts), tp_size)
    loaded_slots = set()
    for slot, tensors in enumerate(experts):
        method.load_packed_mxfp4_expert(
            module, global_expert_id=slot, local_slot_id=slot, **tensors
        )
        pack.load_packed_mxfp4_expert(
            family["w3_w1_weight"],
            family["w3_w1_weight_scale"],
            family["w2_weight"],
            family["w2_weight_scale"],
            slot,
            loaded_slots,
            intermediate_size=intermediate_size,
            moe_tp_size=tp_size,
            moe_tp_rank=tp_rank,
            **tensors,
        )
    torch.cuda.synchronize()
    return method, module, family, loaded_slots


def _assert_equal_tensors(module, family: dict) -> None:
    for name, tensor in family.items():
        engine = getattr(module, name).data
        assert engine.shape == tensor.shape, (name, tuple(engine.shape), tuple(tensor.shape))
        assert engine.dtype == tensor.dtype, (name, engine.dtype, tensor.dtype)
        if not torch.equal(engine, tensor):
            diff = (engine != tensor).nonzero()
            raise AssertionError(
                f"{name}: {diff.shape[0]} bytes differ, first at {diff[0].tolist()} "
                f"(engine {engine[tuple(diff[0])].item()}, family {tensor[tuple(diff[0])].item()})"
            )


@pytest.mark.parametrize("tp_size", [tp for tp, _ in TP_RANKS], ids=lambda n: f"tp{n}")
def test_shapes_and_lean_size_match_engine(tp_size) -> None:
    for tp_rank in dict(TP_RANKS)[tp_size]:
        method, module = _engine_module(3, tp_size, tp_rank)
        shapes = pack.expert_weight_shapes(3, INTERMEDIATE, HIDDEN, tp_size)
        for name, shape in shapes._asdict().items():
            param = getattr(module, name)
            assert tuple(param.shape) == shape, (name, tuple(param.shape), shape)
            assert param.dtype == pack.EXPERT_WEIGHT_DTYPE, (name, param.dtype)
        # bias=False: the engine registers no bias tensors, and the family has none.
        assert module.w3_w1_bias is None and module.w2_bias is None
        assert method.intermediate_size_per_partition_lean == pack.intermediate_size_lean(
            INTERMEDIATE, tp_size, tp_rank
        )
        assert (
            pack.intermediate_size_lean(INTERMEDIATE, tp_size, tp_rank) == INTERMEDIATE // tp_size
        )


@pytest.mark.parametrize("tp_size,tp_rank", LAYOUTS, ids=[f"tp{t}r{r}" for t, r in LAYOUTS])
def test_packer_matches_engine(tp_size, tp_rank) -> None:
    experts = _random_experts(3, seed=_seed(f"{tp_size}/{tp_rank}"))
    _, module, family, loaded_slots = _pack_both(experts, tp_size, tp_rank)
    _assert_equal_tensors(module, family)
    assert loaded_slots == module._packed_mxfp4_loaded_slots == {0, 1, 2}


def test_slot_claims() -> None:
    """A slot loads once; a second load of it raises before writing, as the engine's does."""
    experts = _random_experts(2, seed=11)
    family = _family_tensors(2, 4)
    loaded_slots = set()
    kwargs = dict(intermediate_size=INTERMEDIATE, moe_tp_size=4, moe_tp_rank=1)
    pack.load_packed_mxfp4_expert(*family.values(), 1, loaded_slots, **kwargs, **experts[1])
    before = {name: t.clone() for name, t in family.items()}
    with pytest.raises(ValueError, match="loaded twice"):
        pack.load_packed_mxfp4_expert(*family.values(), 1, loaded_slots, **kwargs, **experts[0])
    for name, tensor in family.items():
        assert torch.equal(tensor, before[name]), name
    with pytest.raises(IndexError):
        pack.load_packed_mxfp4_expert(*family.values(), 2, loaded_slots, **kwargs, **experts[0])
    bad = dict(experts[0], w2_weight_scale=experts[0]["w2_weight_scale"].float())
    with pytest.raises(TypeError):
        pack.load_packed_mxfp4_expert(*family.values(), 0, loaded_slots, **kwargs, **bad)
    assert loaded_slots == {1}


def test_threaded_loads_fill_every_slot_once() -> None:
    """Slots loaded from several threads, as the targets do, give the same bytes as a serial load."""
    experts = _random_experts(6, seed=13)
    tp_size, tp_rank = 2, 1
    serial = _family_tensors(len(experts), tp_size)
    serial_slots = set()
    kwargs = dict(intermediate_size=INTERMEDIATE, moe_tp_size=tp_size, moe_tp_rank=tp_rank)
    for slot, tensors in enumerate(experts):
        pack.load_packed_mxfp4_expert(*serial.values(), slot, serial_slots, **kwargs, **tensors)

    threaded = _family_tensors(len(experts), tp_size)
    threaded_slots = set()
    device = torch.cuda.current_device()

    def worker(slots):
        torch.cuda.set_device(device)
        for slot in slots:
            pack.load_packed_mxfp4_expert(
                *threaded.values(), slot, threaded_slots, **kwargs, **experts[slot]
            )
        torch.cuda.current_stream().synchronize()

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(worker, range(i, 6, 3)) for i in range(3)]:
            future.result()
    torch.cuda.synchronize()
    assert threaded_slots == serial_slots == set(range(6))
    for name in serial:
        assert torch.equal(serial[name], threaded[name]), name


def _checkpoint_experts(ckpt_dir: str, count: int):
    """The first ``count`` routed experts of the first MoE layer of a K3.1 checkpoint."""
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    pattern = re.compile(r"^(.*\.layers\.(\d+)\.block_sparse_moe\.experts)\.0\.w1\.weight_packed$")
    bases = sorted(
        (int(m.group(2)), m.group(1))
        for m in map(pattern.match, weight_map)
        if m is not None and os.path.isfile(os.path.join(ckpt_dir, weight_map[m.group(0)]))
    )
    assert bases, "no routed experts in the checkpoint"
    base = bases[0][1]
    experts = []
    for e in range(count):
        tensors = {}
        for w in ("w1", "w2", "w3"):
            for kind, suffix in (("weight_packed", "weight"), ("weight_scale", "weight_scale")):
                key = f"{base}.{e}.{w}.{kind}"
                with safe_open(
                    os.path.join(ckpt_dir, weight_map[key]), framework="pt", device="cpu"
                ) as fh:
                    tensors[f"{w}_{suffix}"] = fh.get_tensor(key)
        experts.append(tensors)
    return experts


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("TLLM_K31_TEST_CHECKPOINT", "")),
    reason="TLLM_K31_TEST_CHECKPOINT does not name a Kimi K3.1 checkpoint directory",
)
@pytest.mark.parametrize(
    "tp_size,tp_rank", [(1, 0), (4, 2), (16, 9)], ids=["tp1", "tp4r2", "tp16r9"]
)
def test_checkpoint_experts_match_engine(tp_size, tp_rank) -> None:
    experts = _checkpoint_experts(os.environ["TLLM_K31_TEST_CHECKPOINT"], 3)
    assert experts[0]["w1_weight"].shape == (INTERMEDIATE, HIDDEN // 2)
    assert experts[0]["w2_weight_scale"].shape == (HIDDEN, INTERMEDIATE // 32)
    _, module, family, _ = _pack_both(experts, tp_size, tp_rank)
    _assert_equal_tensors(module, family)
