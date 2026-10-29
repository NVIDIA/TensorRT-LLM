# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real layers' routed experts through the prepared layout, on one GPU: the dep4 target's expert
windows at four ranks and at eight (dep8).

The NVFP4 MoE runner reads its expert stacks in a layout no checkpoint stores -- FC1's halves
interleaved, every stack's rows block-shuffled, the block scales 128x4-swizzled -- and nothing
checks it: a wrong layout costs hundreds of bf16 ulp and raises nothing. Rank r holds the whole
experts `[Er, Er + E)`, E = 128 / world (32 at four ranks, 16 at eight over two nodes of four),
runs them over every rank's gathered tokens in calls of at most `_EXPERT_CALL_TOKENS[world]`
rows (131072 at four ranks, 32768 at eight), and the reduce-scatter sums the windows' partials.
On an engine with chunked prefill (as the 1M config at max_seq_len 1048576 runs), the calls are
of at most `_EXPERT_CALL_TOKENS_CHUNKED[world]` rows, 32768 at both.

At four ranks, on one MoE layer and one rank's 32-expert window of Mistral-Large-4-1T-A52B-NVFP4
itself (layer 1 rank 0 and layer 45 rank 3):

* the window's stacks are filled from the shard files by the weight table's own manifest rows
  (sources, destination indices, transforms) and laid out by the rows' own `layout`, the steps
  the table's `load` takes (the CPU manifest test checks `load` itself against an independent
  construction of the layout); that layout is byte for byte the engine's own TRTLLM-Gen
  preparation of the same stored bytes (`trtllmgen_maybe_get_cached_w3_w1_permute_indices`,
  `shuffle_matrix`, `block_scale_interleave`);
* each target's `_moe` -- the layer's router through `renorm_moe_routing_op`, `fp4_quantize`
  under the layer's static scale, the pre-routed runner with `_moe_scalars`, in calls of at
  most 131072 tokens, or of at most 32768 as the 1M config runs it -- is run on
  bf16 activations at the magnitudes the layer's input scale was calibrated for, with the
  shared expert zeroed so the routed half is all that is measured, and compared with an fp32
  reference on the window's dequantized checkpoint experts (`e2m1 * weight_scale *
  weight_scale_2`, activations `e2m1 * scale * input_scale`, the SwiGLU output requantized to
  NVFP4 under `1 / input_scale(w2)` as the reference implementation does), at the runner
  entry's gate: 16 ulp per element of the row's scale, 2 ulp relative RMS. One rank alone
  here, so the expert path's gather and reduce-scatter are the identity (the multi-rank round
  trip is `test_modeling_v2_moe_round_trip.py`'s); 138304 rows run it in two calls, as 40000
  do in the 1M config's;
* the runner on the stored bytes without the layout, and on FC1 with its halves swapped,
  lands far outside that gate (measured 260.8 and 22.5 rms ulp at layer 1, 376.9 and 289.2 at
  layer 45, where the right layout lands at 0.70-0.76): the comparison sees a wrong layout;
* the cut into `_EXPERT_CALL_TOKENS_CHUNKED` calls changes no bit: each target's `_moe` over a
  full step's 131072 gathered rows (four ranks' 32768, three of them mostly the zero padding
  of ranks holding few rows) in calls of 32768 equals, bit for bit, one router / routing /
  NVFP4-quantize / runner call over all of them tuned to 131072, the one call per layer of an
  engine without chunked prefill (the 32K serving config); so do the router logits, the
  routing and the quantized activations of the calls against the one call's;
* the padding rows reach no expert: over a step of 8192 / 16 / 3 / 64 rows, each zero-padded
  to 8192, with the gathered padding rows marked by `_padding_rows` (as the targets' forwards
  mark them), each target's `_moe` hands the runner expert id -1 in every padding row's slots
  and the routing's own ids elsewhere, and returns every real row bit for bit as unmarked (the
  padding then routed to experts 0..3, an exact tie at +0.0) and +0.0 in every padding row.

At eight ranks (dep8), where both tables call 32768 rows, on an early, a middle and a late
MoE layer (1, 35 and 71), for every rank r in 0..7 and its 16-expert window:

* the window's stacks are filled and laid out as above, byte for byte the engine's own
  preparation;
* each target's `_moe`, in calls of at most 32768 rows, is run as the one rank of a one-rank
  group: 16 rows through `DecodeTarget`, 1031 and one whole call of 32768 through
  `PrefillTarget`, and 32768 + 7232 through `DecodeTarget` (two calls, each writing its rows of
  one output), each compared at the same gate with the same fp32 reference on the window's
  dequantized checkpoint experts;
* the eight partials, summed in fp32 in rank order, are compared at the same gate with the
  same reference over all 128 experts, built from the checkpoint's expert keys 0..127 directly
  rather than from the table's windows, so a window the table misplaces, doubles or drops is
  seen; leaving any one partial out lands far outside;
* with every token's top-4 skewed into one window (the router's column 0 biased for that
  window's experts, the activations' element 0 held positive), on one call of the chunk's
  size -- 32768 tokens, so 131072 routed rows on one rank -- that window's partial alone meets
  the 128-expert reference and every other window's is exactly zero;
* the runner on the stored bytes without the layout, on FC1 with its halves swapped, and on a
  neighbouring rank's laid-out window under this rank's offset lands far outside the gate.

Skipped where the checkpoint is absent (looked up as `mistral-large-4-nvfp4` or
`Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)
from tensorrt_llm.mapping import Mapping

DEV = "cuda"
#: GB300 nodes here hold four devices: four ranks fill one, eight span two.
GPUS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")
#: The expert parameters of one MoE layer, in the table's names.
EXPERT_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "isc1", "isc2", "ws2_13", "ws2_2")
#: The runner's epilogue tile, which fixes its 32-row shuffle (the engine's own constant).
EPILOGUE_TILE_M = 128
#: One bf16 ulp, relative: 2^-8 of the binade top.
ULP = 2.0**-8
#: The largest e2m1 magnitude and the largest finite e4m3 one.
E2M1_MAX, E4M3_MAX = 6.0, 448.0
#: The token counts each window is driven at, at eight ranks: a decode step, a ragged prefill,
#: one whole expert call, and two calls (the second ragged).
STEPS = (
    ("DecodeTarget", 16),
    ("PrefillTarget", 1031),
    ("PrefillTarget", M._EXPERT_CALL_TOKENS[8]),
    ("DecodeTarget", M._EXPERT_CALL_TOKENS[8] + 7232),
)
#: An early, a middle and a late MoE layer, every window of each at eight ranks.
LAYERS = (1, 35, 71)

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


def _core(config, rank: int, world: int) -> SimpleNamespace:
    """The surface the weight table reads off the core, for rank `rank` of `world`."""
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
        enable_attention_dp=True,
    )
    assert (mapping.tp_rank, mapping.moe_ep_rank, mapping.moe_tp_rank) == (rank, rank, 0)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=config,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
        )
    )


def _read(checkpoint, keys) -> dict[str, torch.Tensor]:
    """The given checkpoint tensors, off the shard files, one file at a time."""
    by_file = collections.defaultdict(list)
    for key in keys:
        by_file[checkpoint.weight_map[key]].append(key)
    out = {}
    for name, names in by_file.items():
        with safe_open(str(checkpoint.path / name), framework="pt", device="cpu") as handle:
            for key in names:
                out[key] = handle.get_tensor(key)
    return out


def _window(checkpoint, layer: int, rank: int, world: int):
    """Layer `layer`'s expert parameters on rank `rank` of `world`, as the table's `load` leaves
    them: every manifest source copied to its index through its transform, then each row's
    `layout` on every expert. Also returns the stored tensors it read and the router."""
    core = _core(checkpoint.config, rank, world)
    d = W.MODEL_WEIGHTS.dims(core)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    rows = {r.name: r for r in W.MODEL_WEIGHTS.WEIGHTS if r.name in EXPERT_PARAMS}
    keys = {
        k
        for name in EXPERT_PARAMS
        for ckpt, *_ in manifest[f"l{layer}_{name}"]
        for k in W._keys(ckpt)
    }
    router_key = f"layers.{layer}.gate.weight"
    stored = _read(checkpoint, sorted(keys | {router_key}))
    params = {}
    for name in EXPERT_PARAMS:
        row = rows[name]
        param = torch.empty(row.shape(d), dtype=row.dtype or d.dtype, device=DEV)
        for ckpt, index, transform in manifest[f"l{layer}_{name}"]:
            dst = param if index is None else param[index]
            if isinstance(ckpt, tuple):
                src = transform(tuple(W._materialize(stored[k]).to(DEV) for k in ckpt), core)
            else:
                src = W._materialize(stored[ckpt]).to(DEV)
                if transform is not None:
                    src = transform(src.contiguous(), core)
            assert tuple(dst.shape) == tuple(src.shape) and dst.dtype == src.dtype, (name, ckpt)
            dst.copy_(src)
        if row.layout is not None:
            for slot in range(param.shape[0]):
                param[slot].copy_(row.layout(param[slot]))
        params[name] = param
    return d, params, stored, stored[router_key].to(DEV)


def _stored_experts(stored: dict, layer: int, experts) -> dict[str, list[torch.Tensor]]:
    """The stored bytes of the given global experts: `[w3 ; w1]` codes and block scales, w2's,
    and each expert's `weight_scale_2` pair."""
    out = collections.defaultdict(list)
    for expert in experts:
        e = f"layers.{layer}.experts.{expert}"
        out["fc1_w"].append(torch.cat([stored[f"{e}.w3.weight"], stored[f"{e}.w1.weight"]]))
        out["fc1_s"].append(
            torch.cat([stored[f"{e}.w3.weight_scale"], stored[f"{e}.w1.weight_scale"]])
        )
        out["fc2_w"].append(stored[f"{e}.w2.weight"])
        out["fc2_s"].append(stored[f"{e}.w2.weight_scale"])
        out["ws2_13"].append(stored[f"{e}.w1.weight_scale_2"])
        out["ws2_2"].append(stored[f"{e}.w2.weight_scale_2"])
    return out


def _engine_layout(stored: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The engine's own TRTLLM-Gen NVFP4 MoE preparation of the stored bytes, expert by expert:
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


def _reference(x_q, x_s, isc1, isc2, ids, weights, stored, first: int, inter: int):
    """The routed half in fp32 over the given experts' dequantized checkpoint bytes, global
    experts `[first, first + len(stored))`: the runner entry's `reference`, handed the stored
    codes (unpacked) and their block scales with each expert's stored `weight_scale_2`
    multiplied in, and the activation's exact NVFP4 value with its global scale undone
    (`code * scale * input_scale`). fp32; the caller rounds."""
    rows = x_q.shape[0]
    act = _unpack(x_q).long()
    act_scale = x_s.view(torch.float8_e4m3fn).float().reshape(rows, -1)
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=DEV,
    )
    hidden = e2m1[act] * act_scale.repeat_interleave(16, dim=1) * isc1.float()
    count = len(stored["fc1_w"])

    def operand(name, half, global_scales):
        codes = torch.stack([_unpack(t.to(DEV))[half] for t in stored[name + "_w"]])
        scales = torch.stack([t.to(DEV).float()[half] for t in stored[name + "_s"]])
        g = torch.stack([s.to(DEV).float() for s in global_scales])
        return codes, scales * g.view(-1, 1, 1)

    up = operand("fc1", slice(0, inter), stored["ws2_13"])
    gate = operand("fc1", slice(inter, 2 * inter), stored["ws2_13"])
    down = operand("fc2", slice(None), stored["ws2_2"])
    return fp4_block_scale_moe_runner.reference(
        hidden,
        ids,
        weights,
        up,
        gate,
        down,
        fc2_global_scale=1.0 / float(isc2),
        local_expert_offset=first,
        local_num_experts=count,
    )


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP
    return elt, rms


def _gate(out: torch.Tensor, ref: torch.Tensor, what: str) -> tuple[float, float]:
    """The runner entry's gate on the rows `ref` is nonzero in: 16 ulp per element of the row's
    scale, 2 ulp relative RMS."""
    live = (ref != 0).any(dim=1)
    assert bool(live.any()), f"{what}: no token routed here, nothing is measured"
    elt, rms = _deviation(out[live], ref[live].to(torch.bfloat16))
    print(f"{what}: {int(live.sum())} tokens, {elt:.2f} elt / {rms:.3f} rms ulp")
    assert rms <= 2.0, f"{what}: relative RMS {rms:.2f} ulp > 2"
    assert elt <= 16.0, f"{what}: element deviation {elt:.2f} ulp > 16"
    return elt, rms


def _layer(d, params, router) -> SimpleNamespace:
    """The `L` a target's `_moe` reads: a shared expert whose down projection is zero (its
    partial is exactly 0, so the result is the routed half alone), the router's operand, the
    laid-out window and the call scalars from `_moe_scalars`."""
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


def _target(cls, d, side, call: int):
    """A `PrefillTarget` or `DecodeTarget` over the surface `_moe` reads off the core, as the one
    rank of a one-rank group, in calls of, and tuning to, `call` tokens: one call's, as the core
    takes it at its world size and chunked prefill flag (`_derive_first_step`)."""
    target = object.__new__(cls)
    target.core = SimpleNamespace(
        topk=4,
        num_experts=d.num_experts,
        moe_inter=d.moe_inter,
        expert_offset=d.expert_offset,
        local_experts=d.local_experts,
        expert_call_tokens=call,
        expert_tune_tokens=call,
        dp_group=[0],
        dp_rank=0,
        dp_pg=None,
        _side_stream=side,
    )
    return target


@pytest.fixture
def one_rank(monkeypatch):
    """The expert path's collectives over a one-rank group: the gather is the rank's own rows,
    the reduce-scatter its own partial sum."""
    monkeypatch.setattr(M, "allgather", lambda x, sizes, group, pg=None: x.clone())
    monkeypatch.setattr(M, "reducescatter", lambda x, sizes, group, pg=None: x.clone())


def _routing(o, router, call: int):
    """Top-4 ids and weights of every row, routed in the target's expert calls of at most `call`
    rows: the router GEMM's rounding depends on its row count."""
    ids, weights = [], []
    for x in torch.split(o, call):
        logits = cublas_mm(x, router.t(), None, torch.float32)
        i, w = renorm_moe_routing_op(logits, 4, torch.bfloat16)
        ids.append(i)
        weights.append(w)
    return torch.cat(ids), torch.cat(weights)


#: (layer, rank) at four ranks: the layers holding the checkpoint's largest and smallest FC2
#: global scale (1 / input_scale(w2): 6898.53 at layer 1, 1.36898 at layer 45), on the first and
#: the last expert window.
CASES = [(1, 0), (45, 3)]
#: The expert call at four ranks: without chunked prefill (the 32K config's), every gathered
#: row of a full step in one call, and with it (the 1M config's).
CALLS_AT_FOUR = [
    pytest.param(M._EXPERT_CALL_TOKENS[4], id="32k"),
    pytest.param(M._EXPERT_CALL_TOKENS_CHUNKED[4], id="1m"),
]


@pytest.mark.parametrize("call", CALLS_AT_FOUR)
@pytest.mark.parametrize("layer, rank", CASES, ids=[f"layer{n}-rank{r}" for n, r in CASES])
def test_routed_experts_of_one_real_window(checkpoint, one_rank, layer, rank, call):
    """At four ranks: one 32-expert window of the layer, in the config's expert calls."""
    world = 4
    torch.manual_seed(1000 * layer + rank)
    side = torch.cuda.Stream()
    d, params, stored_tensors, router = _window(checkpoint, layer, rank, world)
    window = range(d.expert_offset, d.expert_offset + d.local_experts)
    stored = _stored_experts(stored_tensors, layer, window)

    # The table's layout is the engine's own preparation of the same bytes.
    engine = _engine_layout(stored)
    for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
        mine = params[name].view(torch.uint8)
        assert torch.equal(mine, engine[name].view(torch.uint8)), name

    isc1, isc2 = params["isc1"], params["isc2"]
    # Activations at the amax the layer's static input scale was calibrated for (448 * 6 *
    # input_scale): a gaussian whose largest elements reach about it.
    calibrated = E4M3_MAX * E2M1_MAX * float(isc1)
    L = _layer(d, params, router)
    cases = (
        (M.DecodeTarget, 16),
        (M.PrefillTarget, 1031),
        (M.PrefillTarget, 32768),
        # Past one expert call: 131072 + 7232 rows (32768 + 7232 at the 1M config), each call
        # writing its rows of the output.
        (M.DecodeTarget, call + 7232),
    )
    for cls, rows in cases:
        o = (torch.randn(rows, d.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
        out = _target(cls, d, side, call)._moe(o, L, rows)

        # The routing the target's calls see: the router GEMM's rounding depends on its row
        # count, so the reference routes in the same calls.
        ids, weights = _routing(o, router, call)
        local = ids.long() - d.expert_offset
        live = ((local >= 0) & (local < d.local_experts)).any(dim=1)
        assert bool(live.any()), "no token routed into the window: nothing is measured"
        x_q, x_s = M.fp4_quantize(o, L.moe.gscale, 16, False, False)
        ref = _reference(x_q, x_s, isc1, isc2, ids, weights, stored, d.expert_offset, d.moe_inter)
        ref = ref.to(torch.bfloat16)
        assert bool((out[~live] == 0).all()), "a token with no slot in the window must be zero"
        elt, rms = _deviation(out[live], ref[live])
        print(
            f"layer {layer} rank {rank} {cls.__name__} T={rows}: {int(live.sum())} tokens in "
            f"the window, {elt:.2f} elt / {rms:.3f} rms ulp"
        )
        assert rms <= 2.0, f"relative RMS {rms:.2f} ulp > 2"
        assert elt <= 16.0, f"element deviation {elt:.2f} ulp > 16"

    # A wrong layout runs without complaint and lands far away.
    rows = 1031
    o = (torch.randn(rows, d.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
    logits = cublas_mm(o, L.moe.router, None, torch.float32)
    ids, weights = renorm_moe_routing_op(logits, 4, torch.bfloat16)
    x_q, x_s = M.fp4_quantize(o, L.moe.gscale, 16, False, False)
    ref = _reference(x_q, x_s, isc1, isc2, ids, weights, stored, d.expert_offset, d.moe_inter)
    ref = ref.to(torch.bfloat16)
    live = ((ids.long() - d.expert_offset >= 0) & (ids.long() - d.expert_offset < 32)).any(dim=1)
    as_stored = {n: torch.stack(stored[n]).to(DEV) for n in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")}
    inter = d.moe_inter
    swapped = torch.cat([as_stored["fc1_w"][:, inter:], as_stored["fc1_w"][:, :inter]], dim=1)
    swapped_layout = torch.stack([W._fc1_codes(t) for t in swapped])
    wrong = {
        "no layout": as_stored,
        "halves swapped": dict(params, fc1_w=swapped_layout),
    }
    for what, ops in wrong.items():
        s = L.moe
        out = fp4_block_scale_moe_runner(
            None, None, x_q, x_s.view(torch.float8_e4m3fn), ops["fc1_w"].contiguous(),
            ops["fc1_s"].contiguous(), None, None, None, None, ops["fc2_w"].contiguous(),
            ops["fc2_s"].contiguous(), None, s.out1, s.out1_gate, s.out2, d.num_experts, 4, None,
            None, d.moe_inter, d.expert_offset, d.local_experts, None, 1, True, 0,
            topk_weights=weights, topk_ids=ids, tune_max_num_tokens=32768,
        )[0]  # fmt: skip
        _, rms = _deviation(out[live], ref[live])
        print(f"layer {layer} rank {rank} {what}: {rms:.1f} rms ulp")
        assert rms > 5 * 2.0, f"{what}: {rms:.2f} rms ulp is not told apart from the right layout"


@pytest.mark.parametrize("layer", LAYERS, ids=[f"layer{n}" for n in LAYERS])
def test_routed_experts_of_every_window(checkpoint, one_rank, layer):
    """At eight ranks: every 16-expert window of the layer, alone and summed, in the expert
    calls every engine takes there."""
    world, call = 8, M._EXPERT_CALL_TOKENS[8]
    assert call == M._EXPERT_CALL_TOKENS_CHUNKED[8] == 32768
    torch.manual_seed(1000 * layer)
    side = torch.cuda.Stream()
    windows = [_window(checkpoint, layer, rank, world) for rank in range(world)]
    d0, params0, _, router = windows[0]
    inter = d0.moe_inter
    # Every window is whole experts at 16 r, and the eight tile the 128 routed experts.
    assert [w[0].expert_offset for w in windows] == [16 * r for r in range(world)]
    assert {w[0].local_experts for w in windows} == {16}

    # The table's layout is the engine's own preparation of the same bytes, on every window.
    stored_all: dict[str, torch.Tensor] = {}
    for rank, (d, params, stored_tensors, _) in enumerate(windows):
        stored = _stored_experts(stored_tensors, layer, range(16 * rank, 16 * rank + 16))
        engine = _engine_layout(stored)
        for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
            assert torch.equal(params[name].view(torch.uint8), engine[name].view(torch.uint8)), (
                rank,
                name,
            )
        stored_all.update(stored_tensors)
    # Every expert of the layer, by its checkpoint keys: what the 128-expert reference reads.
    full = _stored_experts(stored_all, layer, range(d0.num_experts))
    isc1, isc2 = params0["isc1"], params0["isc2"]
    for _, params, _, _ in windows:
        assert torch.equal(params["isc1"], isc1) and torch.equal(params["isc2"], isc2)

    # Activations at the amax the layer's static input scale was calibrated for (448 * 6 *
    # input_scale): a gaussian whose largest elements reach about it.
    calibrated = E4M3_MAX * E2M1_MAX * float(isc1)
    layers = [_layer(d, params, router) for d, params, _, _ in windows]
    for cls_name, rows in STEPS:
        cls = getattr(M, cls_name)
        o = (torch.randn(rows, d0.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
        ids, weights = _routing(o, router, call)
        # NVFP4 quantization is per row under the layer's static scale: one call or several
        # quantize every row alike.
        x_q, x_s = M.fp4_quantize(o, layers[0].moe.gscale, 16, False, False)
        parts = []
        for rank, ((d, params, stored_tensors, _), L) in enumerate(zip(windows, layers)):
            out = _target(cls, d, side, call)._moe(o, L, rows)
            stored = _stored_experts(stored_tensors, layer, range(16 * rank, 16 * rank + 16))
            ref = _reference(x_q, x_s, isc1, isc2, ids, weights, stored, 16 * rank, inter)
            local = ids.long() - 16 * rank
            outside = ~((local >= 0) & (local < 16)).any(dim=1)
            assert bool((out[outside] == 0).all()), "a token with no slot in the window is zero"
            _gate(out, ref, f"layer {layer} rank {rank} {cls_name} T={rows}")
            parts.append(out)
        # The windows' sum: fp32 in rank order, rounded once.
        summed = sum(p.float() for p in parts).to(torch.bfloat16)
        ref_full = _reference(x_q, x_s, isc1, isc2, ids, weights, full, 0, inter)
        assert bool((ref_full != 0).any(dim=1).all()), "every token routes somewhere"
        _gate(summed, ref_full, f"layer {layer} eight windows summed {cls_name} T={rows}")
        if rows >= 1031:
            for drop in range(world):
                partial = sum(p.float() for i, p in enumerate(parts) if i != drop)
                _, rms = _deviation(partial.to(torch.bfloat16), ref_full.to(torch.bfloat16))
                assert rms > 25.0, f"T={rows} without window {drop}: only {rms:.1f} rms ulp"
        del parts, summed, ref_full, ref

    # Skew: every token's top-4 in one window, on one call of the chunk's size, so that window's
    # rank runs 4 x 32768 routed rows. Column 0 of the router is biased up for that window's
    # experts and down for the rest, and every token's element 0 is positive, so the bias
    # dominates the logits; the ranking inside the window is the router's own.
    rows = call
    for skew in (0, world - 1, 3):
        o = (torch.randn(rows, d0.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
        o[:, 0] = o[:, 0].abs() + calibrated / 10
        base = cublas_mm(o, router.t(), None, torch.float32)
        bias = 64.0 * float(base.abs().max()) / float(o[:, 0].float().min())
        skewed = router.clone()
        inside = torch.zeros(d0.num_experts, dtype=torch.bool, device=DEV)
        inside[16 * skew : 16 * skew + 16] = True
        skewed[inside, 0] += bias
        skewed[~inside, 0] -= bias
        ids, weights = _routing(o, skewed, call)
        assert bool(((ids >= 16 * skew) & (ids < 16 * skew + 16)).all()), "skew missed"
        x_q, x_s = M.fp4_quantize(o, layers[0].moe.gscale, 16, False, False)
        ref_full = _reference(x_q, x_s, isc1, isc2, ids, weights, full, 0, inter)
        parts = []
        for rank, (d, params, _, _) in enumerate(windows):
            L = _layer(d, params, skewed)
            out = _target(M.PrefillTarget, d, side, call)._moe(o, L, rows)
            if rank != skew:
                assert bool((out == 0).all()), f"skew {skew}: window {rank} wrote"
            parts.append(out)
        _gate(parts[skew], ref_full, f"layer {layer} skew into window {skew} T={rows}")
        summed = sum(p.float() for p in parts).to(torch.bfloat16)
        assert torch.equal(summed, parts[skew]), "the other windows' zeros change the sum"
        del parts, summed, ref_full

    # A wrong layout, or a neighbour's window under this rank's offset, runs without complaint
    # and lands far away.
    rows = 1031
    o = (torch.randn(rows, d0.hidden, device=DEV) * (calibrated / 5)).to(torch.bfloat16)
    ids, weights = _routing(o, router, call)
    x_q, x_s = M.fp4_quantize(o, layers[0].moe.gscale, 16, False, False)
    for rank in (0, world - 1):
        d, params, stored_tensors, _ = windows[rank]
        stored = _stored_experts(stored_tensors, layer, range(16 * rank, 16 * rank + 16))
        ref = _reference(x_q, x_s, isc1, isc2, ids, weights, stored, 16 * rank, inter)
        live = (ref != 0).any(dim=1)
        as_stored = {name: torch.stack(stored[name]).to(DEV) for name in ("fc1_w", "fc1_s")}
        as_stored.update({name: torch.stack(stored[name]).to(DEV) for name in ("fc2_w", "fc2_s")})
        swapped = torch.cat([as_stored["fc1_w"][:, inter:], as_stored["fc1_w"][:, :inter]], dim=1)
        swapped_layout = torch.stack([W._fc1_codes(t) for t in swapped])
        neighbour = windows[(rank + 1) % world][1]
        wrong = {
            "no layout": dict(params, **as_stored),
            "halves swapped": dict(params, fc1_w=swapped_layout),
            f"window {(rank + 1) % world}'s experts": dict(neighbour),
        }
        s = layers[rank].moe
        for what, ops in wrong.items():
            out = fp4_block_scale_moe_runner(
                None, None, x_q, x_s.view(torch.float8_e4m3fn), ops["fc1_w"].contiguous(),
                ops["fc1_s"].contiguous(), None, None, None, None, ops["fc2_w"].contiguous(),
                ops["fc2_s"].contiguous(), None, s.out1, s.out1_gate, s.out2, d.num_experts, 4,
                None, None, inter, d.expert_offset, d.local_experts, None, 1, True, 0,
                topk_weights=weights, topk_ids=ids, tune_max_num_tokens=call,
            )[0]  # fmt: skip
            _, rms = _deviation(out[live], ref[live].to(torch.bfloat16))
            print(f"layer {layer} rank {rank} {what}: {rms:.1f} rms ulp")
            assert rms > 5 * 2.0, f"{what}: {rms:.2f} rms ulp is not told apart from the right one"


#: A full step's gathered rows at four ranks, every rank's max_num_tokens: one expert call
#: without chunked prefill (the 32K config).
GATHERED = 4 * 32768
#: Each rank's live rows in that step, the round trip's skew: rank 0 prefills a full chunk,
#: the others hold a decode dummy, sixteen sequences and a part chunk, zero-padded to 32768.
SKEW = (32768, 1, 16, 9001)


def _one_call(o: torch.Tensor, L, d) -> SimpleNamespace:
    """One expert-path call over every row of `o`: router, routing, NVFP4 quantization and the
    runner each once, tuned to all of them. Over a full step's gathered rows at four ranks, the
    32K config's one call per layer."""
    m = L.moe
    logits = cublas_mm(o, m.router, None, torch.float32)
    ids, weights = renorm_moe_routing_op(logits, 4, torch.bfloat16)
    xq, xs = M.fp4_quantize(o, m.gscale, 16, False, False)
    out = fp4_block_scale_moe_runner(
        None, None, xq, xs.view(torch.float8_e4m3fn), m.fc1_w, m.fc1_s, None, None, None, None,
        m.fc2_w, m.fc2_s, None, m.out1, m.out1_gate, m.out2, d.num_experts, 4, None, None,
        d.moe_inter, d.expert_offset, d.local_experts, None, 1, True, 0,
        topk_weights=weights, topk_ids=ids, tune_max_num_tokens=o.shape[0],
    )[0]  # fmt: skip
    return SimpleNamespace(logits=logits, ids=ids, weights=weights, xq=xq, xs=xs, out=out)


def _rows_differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """How many rows of two equally shaped tensors differ in any bit."""
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    if a.dtype.is_floating_point:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return int((a != b).any(dim=1).sum())


@pytest.mark.parametrize("layer, rank", CASES, ids=[f"layer{n}-rank{r}" for n, r in CASES])
def test_expert_calls_give_one_calls_bits(checkpoint, one_rank, layer, rank):
    """At four ranks: cutting a step's gathered rows into the 1M config's
    `_EXPERT_CALL_TOKENS_CHUNKED[4]`-row calls changes no bit of the router logits, the routing,
    the quantized activations or each target's `_moe` output against one call over all of them,
    the 32K config's (`_EXPERT_CALL_TOKENS[4]` rows)."""
    world, call = 4, M._EXPERT_CALL_TOKENS_CHUNKED[4]
    assert GATHERED == M._EXPERT_CALL_TOKENS[4]
    assert GATHERED % call == 0 and GATHERED > call
    torch.manual_seed(7000 + 1000 * layer + rank)
    side = torch.cuda.Stream()
    d, params, _, router = _window(checkpoint, layer, rank, world)
    L = _layer(d, params, router)
    calibrated = E4M3_MAX * E2M1_MAX * float(params["isc1"])
    o = torch.zeros(GATHERED, d.hidden, dtype=torch.bfloat16, device=DEV)
    block = GATHERED // len(SKEW)
    for r, live in enumerate(SKEW):
        rows = torch.randn(live, d.hidden, device=DEV) * (calibrated / 5)
        o[r * block : r * block + live] = rows.to(torch.bfloat16)

    one = _one_call(o, L, d)
    calls = [_one_call(x, L, d) for x in torch.split(o, call)]
    pieces = {
        "router logits": ("logits", one.logits),
        "top-4 ids": ("ids", one.ids),
        "top-4 weights": ("weights", one.weights),
        "NVFP4 codes": ("xq", one.xq),
        "NVFP4 scales": ("xs", one.xs),
    }
    differ = {}
    for what, (field, whole) in pieces.items():
        cut = torch.cat([getattr(c, field) for c in calls])
        differ[what] = _rows_differing(cut, whole)
    for cls in (M.PrefillTarget, M.DecodeTarget):
        target = _target(cls, d, side, call)
        out = target._moe(o, L, GATHERED)
        # The shared expert's down projection is zero, so its partial adds +0.0 to each value.
        shared = target._dense(o, L)
        assert int(shared.ne(0).sum()) == 0
        differ[f"{cls.__name__}._moe"] = _rows_differing(out, torch.add(one.out, shared))
    local = one.ids.long() - d.expert_offset
    live = int(((local >= 0) & (local < d.local_experts)).any(dim=1).sum())
    print(
        f"layer {layer} rank {rank}: {GATHERED} gathered rows in "
        f"{GATHERED // call} calls of {call} against one call, "
        f"{live} rows routed into the window; rows differing {differ}"
    )
    assert live > 0, "no token routed into the window: the runner's bits are not measured"
    assert all(n == 0 for n in differ.values()), differ


#: Each rank's own rows in a step whose gathered rows are mostly padding, at four ranks: rank 0
#: prefills, the others decode beside it, each zero-padded to 8192.
UNEVEN = (8192, 16, 3, 64)


@pytest.mark.parametrize("call", CALLS_AT_FOUR)
@pytest.mark.parametrize("layer, rank", CASES, ids=[f"layer{n}-rank{r}" for n, r in CASES])
def test_padding_rows_reach_no_expert(checkpoint, one_rank, monkeypatch, layer, rank, call):
    """At four ranks: with the gathered padding rows marked by `_padding_rows`, as the targets'
    forwards mark them, each target's `_moe` hands the runner expert id -1 in every slot of
    every padding row and the routing's own ids in every other row, and returns every real row
    bit for bit as without the mark (the padding then routed, an exact tie at +0.0, to experts
    0..3 of rank 0's window), and +0.0 in every padding row."""
    world = 4
    torch.manual_seed(9000 + 1000 * layer + rank)
    side = torch.cuda.Stream()
    d, params, _, router = _window(checkpoint, layer, rank, world)
    L = _layer(d, params, router)
    calibrated = E4M3_MAX * E2M1_MAX * float(params["isc1"])
    seg = max(UNEVEN)
    o = torch.zeros(world * seg, d.hidden, dtype=torch.bfloat16, device=DEV)
    for r, live in enumerate(UNEVEN):
        rows = torch.randn(live, d.hidden, device=DEV) * (calibrated / 5)
        o[r * seg : r * seg + live] = rows.to(torch.bfloat16)
    pad = M._padding_rows(list(UNEVEN), seg, torch.device(DEV))
    assert M._padding_rows([seg] * world, seg, torch.device(DEV)) is None
    assert pad.shape == (world * seg,) and int(pad.sum()) == world * seg - sum(UNEVEN)
    for r, live in enumerate(UNEVEN):
        assert not pad[r * seg : r * seg + live].any() and pad[r * seg + live : (r + 1) * seg].all()

    runner = M.fp4_block_scale_moe_runner
    for cls in (M.PrefillTarget, M.DecodeTarget):
        target = _target(cls, d, side, call)
        out, ids = {}, {}
        for marked in (False, True):
            seen = []

            def recorded(*args, _seen=seen, **kwargs):
                _seen.append(kwargs["topk_ids"].clone())
                return runner(*args, **kwargs)

            monkeypatch.setattr(M, "fp4_block_scale_moe_runner", recorded)
            out[marked] = target._moe(o, L, o.shape[0], pad if marked else None)
            ids[marked] = torch.cat(seen)
        torch.cuda.synchronize()
        real = ~pad
        differ = _rows_differing(out[True][real], out[False][real])
        print(
            f"layer {layer} rank {rank} {cls.__name__}, calls of {call}: {int(pad.sum())} padding "
            f"rows of {o.shape[0]}, unmarked routed to experts {ids[False][pad].unique().tolist()}; "
            f"real rows differing {differ}"
        )
        assert sorted(ids[False][pad].unique().tolist()) == [0, 1, 2, 3]
        assert bool((ids[True][pad] == -1).all())
        assert torch.equal(ids[True][real], ids[False][real])
        assert differ == 0
        assert int(out[True][pad].ne(0).sum()) == 0
