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
"""The DeepSeek-V4.1-Flash-NVFP4 target's vision tower: CUDA-graph replay, per-image assembly.

One GPU, the checkpoint's real vision weights (the target's own weight rows), no engine.

* Replay is BITWISE the same bucket call run eagerly (same staged inputs, same slot metadata), for
  single images from 7 to 9189 patches -- every bucket, both sides of bucket edges, the split-K
  band 2305..2432 -- replayed in shuffled order, twice.
* Replay against the unpadded eager tower (the path with the switch at 0): the only difference
  is the padded row count B vs P. Measured BITWISE equal -- ViT output and span rows -- at every
  image size the input processor can produce (it never makes fewer than ~1469 patches), 31 sizes
  from 1024 to 9189 in 23 buckets; gated bitwise there. A 7-patch call padded to 1024 rows takes
  another cuBLAS algorithm at M = 7 and is gated to a band only.
* Against the golden (golden/vision): graph and eager each within 1.25x the vendor code's own
  distance, and bitwise equal to each other.
* An image's span rows depend on that image alone: the same image inside different requests,
  and a request's rows spliced chunk by chunk by the prefill target's ``_mm_step``, are
  bitwise equal to the whole-prompt splice.
* The per-image tables (``aligner_index_local`` / ``span_source_local``) assemble bitwise the
  rows the request-wide tables give.
* The memory-profiling hook runs the largest image; its peak is reported.

Latency (eager vs replay per image) is printed, not gated.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    modeling as tm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    weights as tw,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.input_processor import (  # noqa: E501
    ImageGrid,
    PreprocessedImage,
    VisionGeometry,
    build_vision_inputs,
)

# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. The golden vision dumps have no staged
# location: MODELING_V2_DSV41_VISION_GOLDEN_DIR. A case that needs one skips when it
# is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_VISION_GOLDEN_DIR", "")
_GOLDEN_CASES = ("mmmu_small", "mmmu_large", "mmmu_multi3", "carrots_corn")
DEV = torch.device("cuda")

# The one GPU architecture the target is written for (its routing module's ``_SM``).
_SM = (10, 3)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != _SM,
    reason=f"the target is certified on sm_{_SM[0]}{_SM[1]}",
)


def _geometry():
    return VisionGeometry(
        patch_size=14, downsample_ratio=3, max_image_tokens=1024, min_pixels=544 * 544,
        max_wh_ratio=None, hidden_size=1024, num_heads=16, num_layers=32, rope_theta=10000.0,
        image_token_id=tm._IMAGE_TOKEN,
    )  # fmt: skip


def _grid(n_vit_h: int, n_vit_w: int) -> ImageGrid:
    g = ImageGrid(n_llm_h=-(-n_vit_h // 3), n_llm_w=-(-n_vit_w // 3), best_height=14 * n_vit_h,
                  best_width=14 * n_vit_w, n_vit_h=n_vit_h, n_vit_w=n_vit_w)  # fmt: skip
    assert g.num_span_tokens <= 1024, (n_vit_h, n_vit_w)
    return g


def _image(grids, seed: int = 0, patches=None) -> dict:
    """A request's ``multimodal_data["image"]`` (on the GPU) for images of these grids."""
    gen = torch.Generator().manual_seed(seed)
    images, starts, pos = [], [], 0
    for k, g in enumerate(grids):
        p = (
            patches[k]
            if patches is not None
            else (torch.rand((g.n_vit_h * g.n_vit_w, 3, 14, 14), generator=gen) * 2 - 1).to(
                torch.bfloat16
            )
        )
        images.append(PreprocessedImage(patches=p, grid=g))
        starts.append(pos + 1)
        pos += 1 + g.num_span_tokens
    data = build_vision_inputs(images, starts, _geometry())
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in data.items()}


def _stats(o: torch.Tensor, g: torch.Tensor) -> dict:
    o, g = o.double().flatten(1), g.double().flatten(1)
    rr = (o - g).norm(dim=1) / g.norm(dim=1).clamp_min(1e-30)
    return {"median": rr.median().item(), "max": rr.max().item(),
            "bitexact": (o == g).sum().item() / o.numel()}  # fmt: skip


@pytest.fixture(scope="module")
def params():
    if not _CKPT_DIR or not os.path.isdir(_CKPT_DIR):
        pytest.skip(f"checkpoint not present: {_CKPT_DIR or 'LLM_MODELS_ROOT unset'}")
    from safetensors import safe_open

    wm = json.load(open(os.path.join(_CKPT_DIR, "model.safetensors.index.json")))["weight_map"]
    # The weight table's vision rows: replicated, not per layer, sourced as they are.
    d = SimpleNamespace(hidden=5120, dtype=torch.bfloat16)
    rows = [e for e in tw.MODEL_WEIGHTS.WEIGHTS if e.name.startswith("vis")]
    handles, out = {}, {}
    for entry in rows:
        ((key, _dst),) = entry.sources(d, None, "")
        keys, srcs, multi = tw._split_source(key)
        ts = []
        for k, src in zip(keys, srcs):
            f = wm[k]
            if f not in handles:
                handles[f] = safe_open(os.path.join(_CKPT_DIR, f), framework="pt", device="cpu")
            ts.append(tw._materialize(handles[f].get_slice(k), src).to(DEV))
        tr = entry.transform
        t = tr(tuple(ts) if multi else ts[0], None) if tr else ts[0]
        shape, dtype = entry.shape(d), entry.dtype or d.dtype
        assert tuple(t.shape) == tuple(shape) and t.dtype == dtype, (entry.name, t.shape, shape)
        out[entry.name] = t.contiguous()
    return out


def _tower(params, graphs: bool, disable_mm_encoder: bool = False):
    core = SimpleNamespace(
        w=params, rank=0, hidden=5120, disable_mm_encoder=disable_mm_encoder, _const={
            "false_b": torch.zeros((1,), dtype=torch.bool, device=DEV),
            "true_b": torch.ones((1,), dtype=torch.bool, device=DEV)},
    )  # fmt: skip
    tower = tm.VisionTower(core)
    core._vision = tower
    saved = os.environ.get(tm._VIT_GRAPH_ENV)
    os.environ[tm._VIT_GRAPH_ENV] = "1" if graphs else "0"
    try:
        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        tower.post_load_weights()
        torch.cuda.synchronize()
        tower.load_seconds = time.perf_counter() - t0
        tower.load_bytes = free0 - torch.cuda.mem_get_info()[0]
    finally:
        if saved is None:
            os.environ.pop(tm._VIT_GRAPH_ENV)
        else:
            os.environ[tm._VIT_GRAPH_ENV] = saved
    return tower


@pytest.fixture(scope="module")
def eager(params):
    t = _tower(params, graphs=False)
    assert t._g is None
    return t


@pytest.fixture(scope="module")
def graph(params):
    t = _tower(params, graphs=True)
    assert set(t._g) == set(tm._VIT_GRAPH_BUCKETS)
    print(
        f"\n[vision] {len(t._g)} graphs captured in {t.load_seconds:.1f} s, "
        f"{t.load_bytes / 2**20:.1f} MiB held"
    )
    # Static buffers + one shared pool for the largest bucket's working set.
    assert t.load_bytes < 2 * 2**30
    return t


# Single-image grids (n_vit_h, n_vit_w), all legal spans (<= 1024 positions): every bucket, both
# sides of bucket edges, exact bucket sizes, the split-K band 2305..2432, 7 and 9189 patches.
_GRIDS = [
    (1, 7), (32, 32), (3, 400), (39, 39), (33, 47), (32, 64), (3, 683), (48, 48), (5, 461),
    (32, 76), (3, 900), (3, 1000), (3, 1100), (3, 1150), (60, 60), (51, 74), (3, 1365), (64, 64),
    (3, 1366), (78, 60), (72, 72), (75, 75), (3, 2000), (3, 2100), (84, 84), (3, 2500), (90, 90),
    (87, 99), (32, 272), (3, 2902), (6, 1530), (3, 3063),
]  # fmt: skip


def _bucket(P: int) -> int:
    return next(b for b in tm._VIT_GRAPH_BUCKETS if b >= P)


@torch.inference_mode()
def test_replay_is_bitwise_the_bucket_call_run_eagerly(graph):
    grids = [_grid(h, w) for h, w in _GRIDS]
    Ps = sorted({g.n_vit_h * g.n_vit_w for g in grids})
    assert {_bucket(P) for P in Ps} == set(tm._VIT_GRAPH_BUCKETS)
    assert {2304 < P <= 2432 for P in Ps} == {True, False}, "the split-K band is covered"
    assert min(Ps) == 7 and max(Ps) == 9189
    order = list(range(len(grids))) * 2
    random.Random(5).shuffle(order)
    first = {}
    for k in order:
        image = _image([grids[k]], seed=k)
        replayed = graph._vit(image, 0)
        P = replayed.shape[0]
        # The same bucket call, eagerly, over the same staged inputs.
        B = graph._stage_graph_inputs(image, image["pixel_values"].reshape(P, -1), 0)
        graph._graph_body(B, graph._g_metas[B])
        eager_bucket = graph._g_bufs["h"][:P].clone()
        assert torch.equal(replayed, eager_bucket), (grids[k], B)
        assert torch.isfinite(replayed).all()
        if k in first:
            assert torch.equal(first[k], replayed), "a second replay of the same image differs"
        first[k] = replayed


@torch.inference_mode()
def test_replay_matches_the_unpadded_eager_tower(eager, graph):
    worst = {"vit": 0.0, "rows": 0.0}
    nbit = 0
    for k, (h, w) in enumerate(_GRIDS):
        image = _image([_grid(h, w)], seed=100 + k)
        P = h * w
        ve, vg = eager._vit(image, 0), graph._vit(image, 0)
        re_, rg = eager.image_rows(image, 0), graph.image_rows(image, 0)
        sv, sr = _stats(vg, ve), _stats(rg, re_)
        bitwise = torch.equal(vg, ve) and torch.equal(rg, re_)
        nbit += bitwise
        print(
            f"[vision] P={P} bucket={_bucket(P)} vit row-rel median {sv['median']:.2e} max "
            f"{sv['max']:.2e} bitexact {sv['bitexact']:.6f}; span rows max {sr['max']:.2e} "
            f"bitexact {sr['bitexact']:.6f}{' (bitwise)' if bitwise else ''}"
        )
        if P >= tm._VIT_GRAPH_BUCKETS[0]:
            # Every size the processor can produce (>= ~1469 patches, min_pixels).
            assert bitwise, P
        else:
            worst["vit"] = max(worst["vit"], sv["max"])
            worst["rows"] = max(worst["rows"], sr["max"])
    print(
        f"[vision] graph == unpadded eager bitwise for {nbit}/{len(_GRIDS)} images; below 1024 "
        f"patches worst {worst}"
    )
    # Inside the target-vs-golden distance (vit ~1.5e-2 median, test_both_modes_match_the_golden).
    assert worst["vit"] < 0.1 and worst["rows"] < 0.05, worst


def _golden(case: str):
    path = os.path.join(_GOLDEN_DIR, case, "vision.pt") if _GOLDEN_DIR else ""
    if not path or not os.path.isfile(path):
        pytest.skip(f"golden not present: {path}")
    g = torch.load(path, map_location="cpu")
    cu = [int(v) for v in g["vit_cu_seqlens"]]
    grids = [_grid(int(h), int(w)) for h, w in g["vit_grid_hw"].tolist()]
    patches = [g["pixel_values"][cu[i] : cu[i + 1]] for i in range(len(grids))]
    return g, _image(grids, patches=patches)


@torch.inference_mode()
@pytest.mark.parametrize("case", _GOLDEN_CASES)
def test_both_modes_match_the_golden(case, eager, graph):
    g, image = _golden(case)
    n = len(image["vit_cu_seqlens"]) - 1
    # The golden's own RoPE rows (its vit_out was computed with them).
    drift = max((image["vit_rope_cos"].cpu() - g["rope_cos"]).abs().max().item(),
                (image["vit_rope_sin"].cpu() - g["rope_sin"]).abs().max().item())  # fmt: skip
    assert drift < 1e-6, drift
    image["vit_rope_cos"] = g["rope_cos"].to(DEV)
    image["vit_rope_sin"] = g["rope_sin"].to(DEV)
    gold_vit = g["vit_out"].to(DEV)
    gold_span = g["span_embeds"].to(DEV)
    # The checkpoint's own modules, run as is, against the golden: the band.
    vendor_vit = _stats(g["vendor_vit_out"].to(DEV), gold_vit)
    vendor_al = _stats(g["vendor_aligner_out"].to(DEV), g["aligner_out"].to(DEV))
    res, outs = {}, {}
    for name, tower in (("eager", eager), ("graph", graph)):
        vit = torch.cat([tower._vit(image, i) for i in range(n)])
        rows = torch.cat([tower.image_rows(image, i) for i in range(n)])
        outs[name] = (vit, rows)
        res[name] = (_stats(vit, gold_vit), _stats(rows, gold_span))
    print(
        f"[vision] {case}: vendor vit median {vendor_vit['median']:.2e} max {vendor_vit['max']:.2e}, "
        f"vendor aligner median {vendor_al['median']:.2e} max {vendor_al['max']:.2e}; "
        + "; ".join(
            f"{k} vit median {v[0]['median']:.2e} max {v[0]['max']:.2e} span median "
            f"{v[1]['median']:.2e} max {v[1]['max']:.2e}"
            for k, v in res.items()
        )
    )
    for name in res:
        vit_s, span_s = res[name]
        assert vit_s["median"] <= 1.25 * vendor_vit["median"] + 1e-3, (name, vit_s, vendor_vit)
        assert span_s["median"] <= 1.25 * vendor_al["median"] + 1e-3, (name, span_s, vendor_al)
    # The golden images are 1517..8787 patches: the replay is the eager path, bit for bit.
    assert torch.equal(outs["graph"][0], outs["eager"][0]) and torch.equal(
        outs["graph"][1], outs["eager"][1]
    )


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_an_image_rows_depend_on_that_image_alone(mode, eager, graph):
    tower = eager if mode == "eager" else graph
    grids = [_grid(39, 39), _grid(51, 74), _grid(3, 683), _grid(33, 47), _grid(64, 64)]
    gen = torch.Generator().manual_seed(9)
    patches = [(torch.rand((g.n_vit_h * g.n_vit_w, 3, 14, 14), generator=gen) * 2 - 1).to(torch.bfloat16)
               for g in grids]  # fmt: skip
    whole = _image(grids, patches=patches)
    rows_whole = [tower.image_rows(whole, i) for i in range(len(grids))]
    # Each image alone, and inside a request with other neighbours, in another order.
    for i in range(len(grids)):
        alone = _image([grids[i]], patches=[patches[i]])
        assert torch.equal(tower.image_rows(alone, 0), rows_whole[i]), i
    perm = [3, 0, 4, 1, 2]
    shuffled = _image([grids[i] for i in perm], patches=[patches[i] for i in perm])
    for j, i in enumerate(perm):
        assert torch.equal(tower.image_rows(shuffled, j), rows_whole[i]), (j, i)


@torch.inference_mode()
def test_local_tables_assemble_the_request_wide_rows(eager):
    """Per-image assembly == the request-wide gather over ``aligner_index`` / ``span_source_index``
    (the per-image ViT outputs concatenated), bitwise."""
    grids = [_grid(39, 39), _grid(5, 461), _grid(51, 74)]
    image = _image(grids, seed=3)
    n = len(grids)
    v = torch.cat([eager._vit(image, i) for i in range(n)])
    padded = torch.cat([v, v.new_zeros(1, v.shape[1])])
    cells = padded[image["aligner_index"]]
    ccu = image["aligner_cu_cells"]
    out = torch.cat([eager._aligner(cells[ccu[i] : ccu[i + 1]]) for i in range(n)])
    ref = torch.cat([out, eager._c["span_tail"]])[image["span_source_index"]]
    ours = torch.cat([eager.image_rows(image, i) for i in range(n)])
    assert torch.equal(ours, ref)


def _mm_param(image, layout, row0, begin, end):
    return SimpleNamespace(
        multimodal_data={"image": image, "token_layout": layout},
        multimodal_runtime=SimpleNamespace(past_seen_token_num=begin, chunk_end_pos=end),
        input_ids_start_offset=row0)  # fmt: skip


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_chunked_splice_equals_the_whole_prompt(mode, eager, graph):
    """``_mm_step`` over 1..4 chunks (a span cut by every boundary) == one chunk, bitwise, and
    only the images a chunk reaches run the tower."""
    tower = eager if mode == "eager" else graph
    grids = [_grid(39, 39), _grid(51, 74), _grid(33, 47)]
    image = _image(grids, seed=21)
    starts, lens = image["span_starts"], image["span_lengths"]
    L = starts[-1] + lens[-1] + 5
    mask = torch.zeros(L, dtype=torch.bool)
    for s, n in zip(starts, lens):
        mask[s : s + n] = True
    layout = {"image_mask": mask.to(DEV), "engram_dead_mask": mask.to(DEV)}
    target = SimpleNamespace(core=SimpleNamespace(_vision=tower, _const=tower._core._const))
    ref = torch.zeros(L, 5120, dtype=torch.bfloat16, device=DEV)
    tm.PrefillTarget._mm_step(target, [_mm_param(image, layout, 0, 0, L)], L, L, DEV, ref)
    for bounds in ([0, 100, L], [0, starts[1] + 7, starts[2] + 3, L], [0, 50, 300, 600, L]):
        emb = torch.zeros(L, 5120, dtype=torch.bfloat16, device=DEV)
        calls = []
        orig = tower.image_rows

        def spy(img, i, _orig=orig):
            calls.append(i)
            return _orig(img, i)

        tower.image_rows = spy
        try:
            for a, b in zip(bounds[:-1], bounds[1:]):
                tm.PrefillTarget._mm_step(target, [_mm_param(image, layout, 0, a, b)], b - a,
                                          b - a, DEV, emb[a:b])  # fmt: skip
        finally:
            del tower.image_rows
        assert torch.equal(emb, ref), bounds
        reached = sum(sum(1 for s, n in zip(starts, lens) if max(s, a) < min(s + n, b))
                      for a, b in zip(bounds[:-1], bounds[1:]))  # fmt: skip
        assert len(calls) == reached, (bounds, calls)
    assert torch.equal(ref[~mask.to(DEV)], torch.zeros_like(ref[~mask.to(DEV)]))


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_profile_largest_image(mode, eager, graph):
    tower = eager if mode == "eager" else graph
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    tower.profile_largest_image(8192)
    peak = torch.cuda.max_memory_allocated() - base
    print(f"[vision] profile_largest_image({mode}): peak {peak / 2**20:.1f} MiB above the model")
    assert torch.cuda.memory_allocated() - base < 2**20


@torch.inference_mode()
def test_disable_mm_encoder_captures_nothing_and_stays_correct(params, eager, graph):
    """``llm_args.disable_mm_encoder`` (a generation-only disaggregated server): no graphs, no
    graph memory; an image that arrives anyway runs eagerly, bitwise as the switch-0 tower. The
    default engine (the fixture ``graph``: context / aggregated) still captures every bucket."""
    free0 = torch.cuda.mem_get_info()[0]
    off = _tower(params, graphs=True, disable_mm_encoder=True)
    assert off._g is None and off._g_bufs is None
    assert off.load_bytes < 64 * 2**20, off.load_bytes
    assert set(graph._g) == set(tm._VIT_GRAPH_BUCKETS)
    image = _image([_grid(51, 74)], seed=33)
    assert torch.equal(off.image_rows(image, 0), eager.image_rows(image, 0))
    print(
        f"[vision] disable_mm_encoder: 0 graphs, {off.load_bytes / 2**20:.1f} MiB at derive "
        f"(graph tower: {graph.load_bytes / 2**20:.1f} MiB); free before {free0 / 2**30:.1f} GiB"
    )


@torch.inference_mode()
def test_print_latency(eager, graph):
    """Per-image tower latency, host included (wall clock to a synchronized end)."""
    for h, w in ((39, 39), (48, 49), (78, 60), (90, 96), (3, 3063)):
        image = _image([_grid(h, w)], seed=7)
        res = {}
        for name, tower in (("eager", eager), ("graph", graph)):
            for _ in range(3):
                tower._vit(image, 0)
            torch.cuda.synchronize()
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                tower._vit(image, 0)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            res[name] = statistics.median(ts)
        print(
            f"[vision] latency P={h * w} bucket={_bucket(h * w)}: eager {res['eager']:.2f} ms, "
            f"graph {res['graph']:.2f} ms, x{res['eager'] / res['graph']:.2f}"
        )
