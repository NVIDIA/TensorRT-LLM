# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One real layer's routed experts, width-sliced four or eight ways through the prepared layout, on
one GPU.

At tp4 every rank holds all 128 routed experts of a layer, each cut to its 1536-wide slice of
the 6144-wide intermediate, and at tp8 (eight ranks over two nodes of four) to its 768-wide
slice: `w1`/`w3` by output rows, `w2` by input columns (a strided byte read of the packed codes,
and of the block scales one per 16 inputs), `weight_scale_2` and `input_scale` whole. The NVFP4
MoE runner reads each slice in a layout no checkpoint stores -- FC1's halves interleaved, every
stack's rows block-shuffled, the block scales 128x4-swizzled -- and nothing checks either the
cut or the layout: a wrong one costs hundreds of bf16 ulp and raises nothing. So, on an early, a
middle and a late MoE layer (1, 35, 71) of Mistral-Large-4-1T-A52B-NVFP4 itself, at each world
size W (4 and 8), for each rank r in 0..W-1 in turn:

* the rank's stacks are filled from the layer's stored tensors by the weight table's own manifest
  rows (sources, shards, destination indices, transforms) for the core the target builds on rank
  r, and laid out by the rows' own `layout`, the steps the table's `load` takes; that layout is
  byte for byte the engine's own TRTLLM-Gen preparation of the same slice
  (`trtllmgen_maybe_get_cached_w3_w1_permute_indices`, `shuffle_matrix`,
  `block_scale_interleave`);
* the target's `_mlp` -- the layer's router through `renorm_moe_routing_op`, `fp4_quantize` under
  the layer's static scale, the pre-routed runner over all 128 experts at the core's own
  `moe_inter` with `_moe_scalars` -- runs on bf16 activations at the magnitudes the layer's input
  scale was calibrated for (16 rows through `DecodeTarget`, 1031 and 32768 through
  `PrefillTarget`, the last 131072 permuted rows over the 128 local experts), with the shared
  expert zeroed so the routed partial is all that is measured, and is compared with an fp32
  reference on rank r's slice of the dequantized checkpoint experts, cut here from the whole
  stored tensors (`e2m1 * weight_scale * weight_scale_2`, activations `e2m1 * scale *
  input_scale`, the SwiGLU output requantized to NVFP4 under `1 / input_scale(w2)` as the
  reference implementation does), at the gate the runner's element band certifies a slice of
  width I at: 16 * sqrt(6144 / I) ulp per element of the row's scale (32 at 1536, 45.25 at
  768), 2 ulp relative RMS;

then the W ranks' partials, summed in fp32 in rank order as the closing all-reduce sums them,
are compared with the same reference over the whole experts at the whole-width gate, 16 ulp per
element and 2 ulp RMS. With every token's top-4 skewed onto four experts (124..127, the router's
column 0 biased for them and the activations' element 0 held positive), 32768 tokens put 32768
rows on each of the four; every rank's partial and the W ranks' sum meet the same gates.
Wrong readings run without complaint and must land outside the slice's gate: the stored slice
without the layout, FC1 with its halves swapped, and `w2`'s codes or its block scales taken from
the neighbouring rank's column range; and on the sum, one rank's partial taken twice in place of
its neighbour's.

Skipped where the checkpoint is absent (looked up as `mistral-large-4-nvfp4` or
`Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`)."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)
from tensorrt_llm.mapping import Mapping

DEV = "cuda"
#: The world sizes and each one's slice of every expert's 6144-wide intermediate.
SLICE_WIDTH = {4: 1536, 8: 768}
#: GB300 nodes here hold four devices; the eight-rank world spans two of them.
GPUS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")
#: The expert parameters of one MoE layer, in the table's names.
EXPERT_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "isc1", "isc2", "ws2_13", "ws2_2")
#: The engine bound the target is built for: the runner's tuner bucket and the largest step.
MAX_NUM_TOKENS = 32768
#: The runner's epilogue tile, which fixes its 32-row shuffle (the engine's own constant).
EPILOGUE_TILE_M = 128
#: One bf16 ulp, relative: 2^-8 of the binade top.
ULP = 2.0**-8
#: The largest e2m1 magnitude and the largest finite e4m3 one.
E2M1_MAX, E4M3_MAX = 6.0, 448.0
#: The gates the runner certifies a width slice at: an element of one rank's I-wide partial reads
#: a marginal FC2-input flip up to sqrt(6144 / I) larger against its row's scale than the whole
#: width does -- 2x at 1536, 2.83x at 768 -- so 16 * sqrt(6144 / I) ulp per element there (32 and
#: 45.25) and 16 at the whole width; RMS 2 ulp at both.
ELT_WHOLE, RMS = 16.0, 2.0
ELT_SLICE = {world: ELT_WHOLE * math.sqrt(6144 / width) for world, width in SLICE_WIDTH.items()}
#: A wrong reading must land this far past the RMS gate.
WRONG_RMS = 5 * RMS
#: An early, a middle and a late MoE layer, as the runner's cells spread the layer scalars.
LAYERS = (1, 35, 71)
#: (target, rows): a decode step at the largest captured batch, an eager context step, and the
#: largest step the engine sends (max_num_tokens, unchunked).
STEPS = (("DecodeTarget", 16), ("PrefillTarget", 1031), ("PrefillTarget", MAX_NUM_TOKENS))
#: The rows the wrong readings run at.
WRONG_ROWS = 1031
#: The skew: every token's top-4 on these experts, at the largest step.
SKEW_EXPERTS = (124, 125, 126, 127)
SKEW_ROWS = MAX_NUM_TOKENS

# The reference GEMMs must be true fp32.
torch.backends.cuda.matmul.allow_tf32 = False


def _checkpoint_dir() -> Path | None:
    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        path = root / name
        if (path / "params.json").is_file() and (path / W.SHARD_INDEX).is_file():
            return path
    return None


@pytest.fixture(scope="module")
def checkpoint():
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({llm_models_root()})")
    assert torch.cuda.is_available(), "the routed-expert check needs a GPU"
    weight_map = json.loads((path / W.SHARD_INDEX).read_text())["weight_map"]
    # The configuration the engine adapts from the checkpoint's params.json.
    config = MistralConfigLoader().load_pretrained_config(str(path))
    return SimpleNamespace(path=path, weight_map=weight_map, config=config)


def _core(config, world: int, rank: int) -> M.MistralLarge4Tp4ModelingV2:
    """The target's own core for rank `rank` of `world` (tp4 or tp8), declared on the meta
    device: its geometry is the core's derivation off the mapping, and the weight table reads it
    as `load` does."""
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=world,
        moe_tp_size=world,
        moe_ep_size=1,
    )
    assert (mapping.tp_rank, mapping.moe_tp_rank, mapping.moe_ep_rank) == (rank, rank, 0), mapping
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        max_num_tokens=MAX_NUM_TOKENS,
        max_seq_len=MAX_NUM_TOKENS,
    )
    with torch.device("meta"):
        return M.MistralLarge4Tp4ModelingV2(model_config)


def _read(checkpoint, keys) -> dict[str, torch.Tensor]:
    """The given checkpoint tensors, whole, off the shard files, one file at a time."""
    by_file = collections.defaultdict(list)
    for key in keys:
        by_file[checkpoint.weight_map[key]].append(key)
    out = {}
    for name in sorted(by_file):
        with safe_open(str(checkpoint.path / name), framework="pt", device="cpu") as handle:
            for key in by_file[name]:
                out[key] = handle.get_tensor(key)
    return out


def _layer_keys(core, layer: int) -> list[str]:
    """Every checkpoint tensor the layer's routed experts and router load from, on any rank."""
    manifest = W.MODEL_WEIGHTS.manifest(core)
    keys = {
        k
        for name in EXPERT_PARAMS
        for ckpt, *_ in manifest[f"l{layer}_{name}"]
        for k in W._keys(ckpt)
    }
    return sorted(keys | {f"layers.{layer}.gate.weight"})


def _rank_params(core, stored: dict, layer: int) -> dict[str, torch.Tensor]:
    """Layer `layer`'s expert parameters on the core's rank, as the table's `load` leaves them:
    every manifest source cut by its shard, copied to its index through its transform, then each
    row's `layout` on every expert."""
    d = W.MODEL_WEIGHTS.dims(core)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    rows = {r.name: r for r in W.MODEL_WEIGHTS.WEIGHTS if r.name in EXPERT_PARAMS}
    params = {}
    for name in EXPERT_PARAMS:
        row = rows[name]
        param = torch.empty(row.shape(d), dtype=row.dtype or d.dtype, device=DEV)
        for ckpt, index, transform, shard in manifest[f"l{layer}_{name}"]:
            dst = param if index is None else param[index]
            if isinstance(ckpt, tuple):
                src = transform(tuple(W._materialize(stored[k], shard).to(DEV) for k in ckpt), core)
            else:
                src = W._materialize(stored[ckpt], shard).to(DEV)
                if transform is not None:
                    src = transform(src.contiguous(), core)
            assert tuple(dst.shape) == tuple(src.shape) and dst.dtype == src.dtype, (name, ckpt)
            dst.copy_(src)
        if row.layout is not None:
            for slot in range(param.shape[0]):
                param[slot].copy_(row.layout(param[slot]))
        params[name] = param
    return params


def _stored_slices(stored: dict, layer: int, experts: int, inter: int, rank: int):
    """Rank `rank`'s stored bytes per expert, cut here from the whole tensors: `[w3 ; w1]` rows
    `[inter * rank, + inter)` of the codes and of the block scales, `w2`'s input columns over the
    same range (two per code byte, sixteen per block scale)."""
    rows = slice(inter * rank, inter * (rank + 1))
    code_cols = slice(inter * rank // 2, inter * (rank + 1) // 2)
    scale_cols = slice(inter * rank // 16, inter * (rank + 1) // 16)
    out = collections.defaultdict(list)
    for e in range(experts):
        p = f"layers.{layer}.experts.{e}"
        out["fc1_w"].append(
            torch.cat([stored[f"{p}.w3.weight"][rows], stored[f"{p}.w1.weight"][rows]])
        )
        out["fc1_s"].append(
            torch.cat([stored[f"{p}.w3.weight_scale"][rows], stored[f"{p}.w1.weight_scale"][rows]])
        )
        out["fc2_w"].append(stored[f"{p}.w2.weight"][:, code_cols].contiguous())
        out["fc2_s"].append(stored[f"{p}.w2.weight_scale"][:, scale_cols].contiguous())
    return out


def _engine_layout(stored: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The engine's own TRTLLM-Gen NVFP4 MoE preparation of the stored slices, expert by expert:
    its gated-activation row reorder chained with its row shuffle, `shuffle_matrix`, and
    `block_scale_interleave` on the scales."""
    cache: dict = {}
    sf = torch.uint8
    out = collections.defaultdict(list)
    for w31, s31, w2, s2 in zip(stored["fc1_w"], stored["fc1_s"], stored["fc2_w"], stored["fc2_s"]):
        w31, s31, w2, s2 = (t.to(DEV) for t in (w31, s31, w2, s2))
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(w31, cache, EPILOGUE_TILE_M)
        out["fc1_w"].append(torch.ops.trtllm.shuffle_matrix(w31, perm))
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            s31.view(sf), cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(s31.view(sf), perm)
        out["fc1_s"].append(
            torch.ops.trtllm.block_scale_interleave(shuffled.reshape(s31.shape)).reshape(s31.shape)
        )
        perm = trtllmgen_maybe_get_cached_w2_permute_indices(w2, cache, EPILOGUE_TILE_M)
        out["fc2_w"].append(torch.ops.trtllm.shuffle_matrix(w2, perm))
        perm = trtllmgen_maybe_get_cached_w2_permute_indices(
            s2.view(sf), cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(s2.view(sf), perm)
        out["fc2_s"].append(
            torch.ops.trtllm.block_scale_interleave(shuffled.reshape(s2.shape)).reshape(s2.shape)
        )
    return {name: torch.stack(t) for name, t in out.items()}


def _unpack(codes: torch.Tensor) -> torch.Tensor:
    """`[..., K/2]` packed e2m1 -> `[..., K]` codes, the even element in the low nibble."""
    return torch.stack([codes & 0x0F, codes >> 4], dim=-1).reshape(*codes.shape[:-1], -1)


def _whole_operands(stored: dict, layer: int, experts: int, ws2_13, ws2_2) -> SimpleNamespace:
    """The layer's whole experts as the reference reads them: unpacked e2m1 codes `[E, N, K]` and
    fp32 block scales `[E, N, K/16]` with each expert's `weight_scale_2` multiplied in, for the up
    (`w3`), gate (`w1`) and down (`w2`) projections."""

    def operand(proj, global_scale):
        p = f"layers.{layer}.experts"
        codes = torch.stack(
            [_unpack(stored[f"{p}.{e}.{proj}.weight"].to(DEV)) for e in range(experts)]
        )
        scales = torch.stack(
            [stored[f"{p}.{e}.{proj}.weight_scale"].to(DEV).float() for e in range(experts)]
        )
        return codes, scales * global_scale.view(-1, 1, 1)

    return SimpleNamespace(
        up=operand("w3", ws2_13), gate=operand("w1", ws2_13), down=operand("w2", ws2_2)
    )


def _reference(x_q, x_s, isc1, isc2, ids, weights, ops, inter: int, rank: int | None):
    """The routed half in fp32 over the dequantized checkpoint experts -- rank `rank`'s
    intermediate slice `[inter * rank, + inter)` of each, or the whole experts for None --
    rounded to bf16: the runner entry's `reference`, handed the activation's exact NVFP4 value
    with its global scale undone (`code * scale * input_scale`)."""
    rows = x_q.shape[0]
    act = _unpack(x_q).long()
    act_scale = x_s.view(torch.float8_e4m3fn).float().reshape(rows, -1)
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=DEV,
    )
    hidden = e2m1[act] * act_scale.repeat_interleave(16, dim=1) * isc1.float()
    if rank is None:
        up, gate, down = ops.up, ops.gate, ops.down
    else:
        out_rows = slice(inter * rank, inter * (rank + 1))
        in_scales = slice(inter * rank // 16, inter * (rank + 1) // 16)
        up = (ops.up[0][:, out_rows], ops.up[1][:, out_rows])
        gate = (ops.gate[0][:, out_rows], ops.gate[1][:, out_rows])
        down = (ops.down[0][:, :, out_rows], ops.down[1][:, :, in_scales])
    out = M.fp4_block_scale_moe_runner.reference(
        hidden,
        ids,
        weights,
        up,
        gate,
        down,
        fc2_global_scale=1.0 / float(isc2),
        local_expert_offset=0,
        local_num_experts=up[0].shape[0],
    )
    return out.to(torch.bfloat16)


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP
    return elt, rms


def _layer(d, params, router) -> SimpleNamespace:
    """The `L` a target's `_mlp` reads: a shared expert whose down projection is zero (its
    partial is exactly 0, so the result is the routed partial alone), the router's operand, the
    rank's laid-out slices and the call scalars from `_moe_scalars`."""
    shared = d.shared_inter
    unit_sf = lambda n, k: W._deepgemm_sf(  # noqa: E731
        torch.ones(n // W.BLOCK, k // W.BLOCK, device=DEV), None
    ).t()
    return SimpleNamespace(
        mlp_gu=(torch.randn(2 * shared, d.hidden, device=DEV) * 0.1).to(torch.float8_e4m3fn),
        mlp_gu_s=unit_sf(2 * shared, d.hidden),
        mlp_dn=torch.zeros(d.hidden, shared, device=DEV).to(torch.float8_e4m3fn),
        mlp_dn_s=unit_sf(d.hidden, shared),
        moe=SimpleNamespace(
            router=router.t(),
            fc1_w=params["fc1_w"],
            fc1_s=params["fc1_s"],
            fc2_w=params["fc2_w"],
            fc2_s=params["fc2_s"],
            **vars(
                M._moe_scalars(params["isc1"], params["isc2"], params["ws2_13"], params["ws2_2"])
            ),
        ),
    )


def _target(name: str, core):
    """A `PrefillTarget` or `DecodeTarget` over the core, its per-layer views not taken (the
    meta core holds no weights): `_mlp` reads only the core's geometry and the `L` it is given."""
    target = object.__new__(getattr(M, name))
    target.core = core
    return target


def _run(L, core, x_q, x_s, ids, weights, ops: dict) -> torch.Tensor:
    """The runner as `_mlp` calls it, on the given (possibly wrong) expert operands."""
    s = L.moe
    return M.fp4_block_scale_moe_runner(
        None, None, x_q, x_s.view(torch.float8_e4m3fn), ops["fc1_w"].contiguous(),
        ops["fc1_s"].contiguous(), None, None, None, None, ops["fc2_w"].contiguous(),
        ops["fc2_s"].contiguous(), None, s.out1, s.out1_gate, s.out2, core.num_experts,
        core.topk, None, None, core.moe_inter, core.expert_offset, core.local_experts, None, 1,
        True, 0, topk_weights=weights, topk_ids=ids, tune_max_num_tokens=core.max_num_tokens,
    )[0]  # fmt: skip


@pytest.mark.parametrize("layer", LAYERS, ids=[f"layer{n}" for n in LAYERS])
@pytest.mark.parametrize("world", sorted(SLICE_WIDTH), ids=[f"tp{w}" for w in sorted(SLICE_WIDTH)])
def test_width_slices_of_one_real_layer(checkpoint, world, layer):
    torch.manual_seed(1000 * layer)
    elt_slice = ELT_SLICE[world]
    cores = [_core(checkpoint.config, world, rank) for rank in range(world)]
    geometry = {(c.moe_inter, c.local_experts, c.expert_offset, c.num_experts) for c in cores}
    assert geometry == {(SLICE_WIDTH[world], 128, 0, 128)}, geometry
    inter, experts = cores[0].moe_inter, cores[0].local_experts
    d = W.MODEL_WEIGHTS.dims(cores[0])

    stored = _read(checkpoint, _layer_keys(cores[0], layer))
    router = stored[f"layers.{layer}.gate.weight"].to(DEV)
    params = [_rank_params(core, stored, layer) for core in cores]
    slices = [_stored_slices(stored, layer, experts, inter, rank) for rank in range(world)]

    for rank in range(world):
        # The table's layout is the engine's own preparation of the same slice.
        engine = _engine_layout(slices[rank])
        for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
            mine = params[rank][name].view(torch.uint8)
            assert torch.equal(mine, engine[name].view(torch.uint8)), (rank, name)
        # The per-tensor scalars are whole: every rank holds every expert's.
        for name in ("isc1", "isc2", "ws2_13", "ws2_2"):
            assert torch.equal(params[rank][name], params[0][name]), (rank, name)

    isc1, isc2 = params[0]["isc1"], params[0]["isc2"]
    ops = _whole_operands(stored, layer, experts, params[0]["ws2_13"], params[0]["ws2_2"])
    del stored
    layers = [_layer(d, params[rank], router) for rank in range(world)]
    # Activations at the amax the layer's static input scale was calibrated for (448 * 6 *
    # input_scale): a gaussian whose largest elements reach about it.
    calibrated = E4M3_MAX * E2M1_MAX * float(isc1)
    kept = {}
    for name, rows in STEPS:
        o = (torch.randn(rows, d.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
        # The router and the quantization are replicated: every rank routes every row the same.
        logits = cublas_mm(o, layers[0].moe.router, None, torch.float32)
        ids, weights = renorm_moe_routing_op(logits, cores[0].topk, torch.bfloat16)
        x_q, x_s = M.fp4_quantize(o, layers[0].moe.gscale, 16, False, False)
        partials, refs = [], []
        for rank in range(world):
            out = _target(name, cores[rank])._mlp(o, layers[rank])
            ref = _reference(x_q, x_s, isc1, isc2, ids, weights, ops, inter, rank)
            elt, rms = _deviation(out, ref)
            print(
                f"layer {layer} rank {rank} {name} T={rows}: slice partial {elt:.2f} elt / "
                f"{rms:.3f} rms ulp",
                flush=True,
            )
            assert rms <= RMS, f"rank {rank}: relative RMS {rms:.2f} ulp > {RMS}"
            assert elt <= elt_slice, f"rank {rank}: element deviation {elt:.2f} ulp > {elt_slice}"
            partials.append(out)
            refs.append(ref)
        whole = _reference(x_q, x_s, isc1, isc2, ids, weights, ops, inter, None)
        summed = partials[0].float()
        for partial in partials[1:]:
            summed = summed + partial.float()
        elt, rms = _deviation(summed.to(torch.bfloat16), whole)
        print(
            f"layer {layer} {name} T={rows}: {world} partials summed {elt:.2f} elt / {rms:.3f} "
            "rms ulp against the whole experts",
            flush=True,
        )
        assert rms <= RMS, f"{world}-rank sum: relative RMS {rms:.2f} ulp > {RMS}"
        assert elt <= ELT_WHOLE, f"{world}-rank sum: element deviation {elt:.2f} ulp > {ELT_WHOLE}"
        # One rank's partial taken twice, in place of its neighbour's, is told apart.
        doubled = partials[0].float() * 2
        for partial in partials[2:]:
            doubled = doubled + partial.float()
        _, rms = _deviation(doubled.to(torch.bfloat16), whole)
        print(f"layer {layer} {name} T={rows}: rank 0 summed twice, no rank 1: {rms:.1f} rms ulp")
        assert rms > WRONG_RMS, f"a doubled slice at {rms:.2f} rms ulp is not told apart"
        if rows == WRONG_ROWS:
            kept = dict(x_q=x_q, x_s=x_s, ids=ids, weights=weights, refs=refs)

    # Wrong readings run without complaint and land far away.
    for rank in range(world):
        neighbour = (rank + 1) % world
        as_stored = {name: torch.stack(t).to(DEV) for name, t in slices[rank].items()}
        swapped = torch.cat([as_stored["fc1_w"][:, inter:], as_stored["fc1_w"][:, :inter]], dim=1)
        wrong = {
            "no layout": as_stored,
            "FC1 halves swapped": dict(
                params[rank], fc1_w=torch.stack([W._fc1_codes(t) for t in swapped])
            ),
            f"w2 codes of rank {neighbour}": dict(params[rank], fc2_w=params[neighbour]["fc2_w"]),
            f"w2 block scales of rank {neighbour}": dict(
                params[rank], fc2_s=params[neighbour]["fc2_s"]
            ),
        }
        for what, wrong_ops in wrong.items():
            out = _run(
                layers[rank], cores[rank], kept["x_q"], kept["x_s"], kept["ids"], kept["weights"],
                wrong_ops,
            )  # fmt: skip
            elt, rms = _deviation(out, kept["refs"][rank])
            print(f"layer {layer} rank {rank} {what}: {elt:.1f} elt / {rms:.1f} rms ulp")
            assert rms > WRONG_RMS, f"{what}: {rms:.2f} rms ulp is not told apart from the slice"

    # Skew: every token's top-4 on SKEW_EXPERTS. Column 0 of the router is biased up for those
    # experts and down for the rest, and every token's element 0 is positive, so the bias
    # dominates the logits; the ranking among the four is the router's own.
    o = (torch.randn(SKEW_ROWS, d.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
    o[:, 0] = o[:, 0].abs() + calibrated / 10
    base = cublas_mm(o, router.t(), None, torch.float32)
    bias = 64.0 * float(base.abs().max()) / float(o[:, 0].float().min())
    skewed = router.clone()
    hot = torch.zeros(experts, dtype=torch.bool, device=DEV)
    hot[list(SKEW_EXPERTS)] = True
    skewed[hot, 0] += bias
    skewed[~hot, 0] -= bias
    logits = cublas_mm(o, skewed.t(), None, torch.float32)
    ids, weights = renorm_moe_routing_op(logits, cores[0].topk, torch.bfloat16)
    assert {int(e) for e in ids.unique()} == set(SKEW_EXPERTS), ids.unique()
    x_q, x_s = M.fp4_quantize(o, layers[0].moe.gscale, 16, False, False)
    partials = []
    for rank in range(world):
        out = _target("PrefillTarget", cores[rank])._mlp(o, _layer(d, params[rank], skewed))
        ref = _reference(x_q, x_s, isc1, isc2, ids, weights, ops, inter, rank)
        elt, rms = _deviation(out, ref)
        print(
            f"layer {layer} rank {rank} skew onto {SKEW_EXPERTS} T={SKEW_ROWS}: slice partial "
            f"{elt:.2f} elt / {rms:.3f} rms ulp",
            flush=True,
        )
        assert rms <= RMS, f"skew, rank {rank}: relative RMS {rms:.2f} ulp > {RMS}"
        assert elt <= elt_slice, f"skew, rank {rank}: element deviation {elt:.2f} ulp > {elt_slice}"
        partials.append(out)
    whole = _reference(x_q, x_s, isc1, isc2, ids, weights, ops, inter, None)
    summed = partials[0].float()
    for partial in partials[1:]:
        summed = summed + partial.float()
    elt, rms = _deviation(summed.to(torch.bfloat16), whole)
    print(
        f"layer {layer} skew T={SKEW_ROWS}: {world} partials summed {elt:.2f} elt / {rms:.3f} "
        "rms ulp against the whole experts",
        flush=True,
    )
    assert rms <= RMS, f"skew, {world}-rank sum: relative RMS {rms:.2f} ulp > {RMS}"
    assert elt <= ELT_WHOLE, (
        f"skew, {world}-rank sum: element deviation {elt:.2f} ulp > {ELT_WHOLE}"
    )
    del o, base, logits, x_q, x_s, partials, summed, whole
