# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 target's image path, pinned against independent implementations.

* One vision-tower layer's attention -- the table's `[wq ; wk ; wv]` operand, the
  2-D RoPE entry, the no-KV-cache attention call and `wo`, exactly as the tower runs
  them -- on the boot image's patch grid, against transformers' Pixtral vision
  attention fed the same weights converted to its layout. The checkpoint stores each
  head's q/k rows as interleaved (GPT-J) rotary pairs; transformers rotates halves,
  so its q/k projections are the published mistral -> HF permutation of those rows
  (each head's rows `(i, 2j + t)` -> `(i, t * D/2 + j)`). The tower must read the
  checkpoint's rows as published: a control that applies that permutation to the
  tower's own operand (what the stock Pixtral path's weight mapper does before its
  half-split rotary) lands far outside the band.
* The table's q/k/v row reads the three checkpoint tensors in that order and
  concatenates them bit for bit: no permutation on the load path.
* The splice: across the context chunks of a request (chunked prefill, or a resume
  from a reused prefix that ends inside the image) the rows placed at the [IMG]
  positions are, chunk by chunk, the whole image's rows in order; [IMG_BREAK],
  [IMG_END] and text keep their token embeddings; the tower runs once per request.

The boot image is `tests/integration/test_input_files/merlion.png` (1192 x 874) as
the engine's mistral-native processor delivers it: 896 x 1204 pixels, a 64 x 86
patch grid (5504 patch rows), 32 x 43 [IMG] tokens.

The attention tests use the checkpoint's vision layer 0 when MISTRAL_LARGE4_CHECKPOINT_DIR
names the checkpoint, else seeded random weights at the same shapes. They need a GPU;
the splice test runs on CPU.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.engine.runners.common import prepare_multimodal_indices
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData


@pytest.fixture(autouse=True, params=["tep4", "tp4", "dep4"])
def target(request):
    global modeling, weights
    module = (
        "tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4."
        f"mistral_large_4_nvfp4__sm_103__{request.param}"
    )
    modeling = importlib.import_module(module + ".modeling")
    weights = importlib.import_module(module + ".weights")


#: params.json `vision_encoder`: hidden 2048 over 16 heads of 128, rope_theta 1e4,
#: image_size 1540, patch 14.
HIDDEN = 2048
HEADS = 16
HEAD_DIM = 128
THETA = 10000.0
IMAGE_SIZE = 1540
PATCH = 14

#: The boot image's patch grid, rows x cols.
BOOT_GRID = (64, 86)

#: The prompt tokens of an image span (tekken.json special ids).
IMG, IMG_BREAK, IMG_END = 10, 12, 13

_CKPT_ENV = "MISTRAL_LARGE4_CHECKPOINT_DIR"
_LAYER0 = "vision_encoder.transformer.layers.0."

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# --- weights ---------------------------------------------------------------------


def _checkpoint_layer0() -> dict | None:
    """Vision layer 0's attention weights and attention norm, as published, or None."""
    root = os.environ.get(_CKPT_ENV)
    if not root or not Path(root, "consolidated.safetensors.index.json").is_file():
        return None
    from safetensors import safe_open

    index = json.loads(Path(root, "consolidated.safetensors.index.json").read_text())
    names = ("attention.wq", "attention.wk", "attention.wv", "attention.wo", "attention_norm")
    out = {}
    for name in names:
        key = f"{_LAYER0}{name}.weight"
        with safe_open(str(Path(root, index["weight_map"][key])), framework="pt") as f:
            out[name] = f.get_tensor(key)
    return out


def _random_layer0() -> dict:
    gen = torch.Generator().manual_seed(2026)
    std = HIDDEN**-0.5

    def mat():
        return (torch.randn(HIDDEN, HIDDEN, generator=gen) * std).to(torch.bfloat16)

    return {
        "attention.wq": mat(),
        "attention.wk": mat(),
        "attention.wv": mat(),
        "attention.wo": mat(),
        "attention_norm": torch.ones(HIDDEN, dtype=torch.bfloat16),
    }


@pytest.fixture(scope="module")
def layer0():
    found = _checkpoint_layer0()
    return (found, "checkpoint") if found is not None else (_random_layer0(), "random")


def _to_hf_layout(w: torch.Tensor) -> torch.Tensor:
    """The mistral -> HF conversion of a q or k projection: within each head, the
    interleaved rotary pairs `(2j, 2j + 1)` become the halves `(j, j + D/2)`."""
    return w.view(HEADS, HEAD_DIM // 2, 2, HIDDEN).transpose(1, 2).reshape(HEADS * HEAD_DIM, HIDDEN)


def _qkv_row():
    """The table's vision q/k/v row."""
    rows = [w for w in weights.MODEL_WEIGHTS.WEIGHTS if w.tag == "v" and w.name == "qkv"]
    assert len(rows) == 1, rows
    return rows[0]


# --- the two attention implementations --------------------------------------------


def _grid_positions(grids, device) -> torch.Tensor:
    """(row, col) of every patch, each image row-major from (0, 0)."""
    return torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
                ),
                -1,
            ).reshape(-1, 2)
            for h, w in grids
        ]
    )


def _tower_attention(x, wq, wk, wv, wo, grids) -> torch.Tensor:
    """The tower's own attention call, on the operand the table builds."""
    qkv = torch.cat((wq, wk, wv), dim=0)
    block = modeling._VisionBlock(
        attn_norm=None, qkv=qkv.t(), wo=wo.t(), ffn_norm=None, w13=None, w2=None
    )
    seg = [h * w for h, w in grids]
    workspace = torch.empty(0, dtype=torch.int8, device=x.device)
    step = modeling._vision_step(seg, x.device, workspace)
    out = modeling._vision_attention(
        x, block, _grid_positions(grids, x.device), step, HEADS, HEAD_DIM, THETA
    )
    torch.cuda.synchronize()
    return out


def _hf_attention(x, wq, wk, wv, wo, grids, dtype) -> torch.Tensor:
    """transformers' Pixtral vision attention, its rotary embedding, its flat
    position ids and its block mask, on the HF-layout weights, in `dtype`."""
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.modeling_pixtral import (
        PixtralAttention,
        PixtralRotaryEmbedding,
        generate_block_attention_mask,
        position_ids_in_meshgrid,
    )

    cfg = PixtralVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=6144,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        image_size=IMAGE_SIZE,
        patch_size=PATCH,
        rope_parameters={"rope_type": "default", "rope_theta": THETA},
    )
    attn = PixtralAttention(cfg).to(device=x.device, dtype=dtype).eval()
    with torch.no_grad():
        attn.q_proj.weight.copy_(_to_hf_layout(wq))
        attn.k_proj.weight.copy_(_to_hf_layout(wk))
        attn.v_proj.weight.copy_(wv)
        attn.o_proj.weight.copy_(wo)
    rope = PixtralRotaryEmbedding(cfg).to(x.device)
    patch_list = [torch.empty(1, h, w) for h, w in grids]
    position_ids = position_ids_in_meshgrid(patch_list, max_width=IMAGE_SIZE // PATCH).to(x.device)
    hidden = x.to(dtype)[None]
    cos, sin = rope(hidden, position_ids)
    mask = None
    if len(grids) > 1:
        mask = generate_block_attention_mask([h * w for h, w in grids], hidden)
    with torch.no_grad():
        out, _ = attn(hidden, attention_mask=mask, position_embeddings=(cos, sin))
    torch.cuda.synchronize()
    return out[0]


def _errors(out: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """(max |error|, RMS error), relative to the truth's RMS."""
    diff = out.double() - truth.double()
    scale = truth.double().pow(2).mean().sqrt().item()
    return diff.abs().max().item() / scale, diff.pow(2).mean().sqrt().item() / scale


# The band: the tower's error from the fp32 answer may be at most this many times
# transformers' own bf16 error from it (both measures). Measured on sm_103 (GB300)
# with the checkpoint's layer 0: 0.97-0.98 of transformers' bf16 error, at the boot
# grid alone and beside a second image.
_BAND = 1.25
# The permuted-weights control must miss the fp32 answer by more than this many
# times transformers' bf16 RMS error. Measured with the checkpoint's layer 0: 86x
# and 108x.
_CONTROL_MIN = 10.0


@needs_gpu
@pytest.mark.parametrize(
    "grids",
    [(BOOT_GRID,), (BOOT_GRID, (32, 32))],
    ids=["boot_grid", "boot_grid_and_32x32"],
)
def test_vision_attention_matches_transformers_pixtral(layer0, grids):
    w, source = layer0
    dev = torch.device("cuda")
    wq, wk, wv, wo = (w[f"attention.{n}"].to(dev) for n in ("wq", "wk", "wv", "wo"))
    n = sum(h * c for h, c in grids)
    gen = torch.Generator(device=dev).manual_seed(7)
    x = torch.randn(n, HIDDEN, device=dev, generator=gen)
    # The block's attention input is RMS-normalized and scaled by attention_norm.
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    x = (x * w["attention_norm"].to(dev).float()).to(torch.bfloat16)

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        truth = _hf_attention(
            x.float(), *(t.float() for t in (wq, wk, wv, wo)), grids, torch.float32
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    ref_bf16 = _hf_attention(x, wq, wk, wv, wo, grids, torch.bfloat16)
    out = _tower_attention(x, wq, wk, wv, wo, grids)
    assert out.shape == (n, HIDDEN) and out.dtype == torch.bfloat16

    ref_max, ref_rms = _errors(ref_bf16, truth)
    got_max, got_rms = _errors(out, truth)
    print(
        f"[{source} weights, grids {grids}] relative to the fp32 answer: transformers bf16 "
        f"max {ref_max:.3e} rms {ref_rms:.3e}; tower max {got_max:.3e} rms {got_rms:.3e}"
    )
    assert got_max <= _BAND * ref_max, (got_max, ref_max)
    assert got_rms <= _BAND * ref_rms, (got_rms, ref_rms)

    # The control: the stock path's permutation applied to the tower's own q/k rows.
    wrong = _tower_attention(x, _to_hf_layout(wq), _to_hf_layout(wk), wv, wo, grids)
    _, wrong_rms = _errors(wrong, truth)
    print(f"[{source} weights] permuted-q/k control rms {wrong_rms:.3e}")
    assert wrong_rms > _CONTROL_MIN * ref_rms, (wrong_rms, ref_rms)


@pytest.mark.cpu_only
def test_the_qkv_row_reads_the_checkpoint_rows_as_published():
    row = _qkv_row()
    d = SimpleNamespace(v_layers=30, v_hidden=HIDDEN)
    assert row.layer_indices(d) == list(range(30))
    sources = row.sources(d, 7, "vision_encoder.transformer.layers.7")
    prefix = "vision_encoder.transformer.layers.7.attention."
    assert [key for key, _ in sources] == [prefix + name + ".weight" for name in ("wq", "wk", "wv")]
    assert row.transform is None
    assert getattr(row, "shard", None) is None
    gen = torch.Generator().manual_seed(3)
    parts = tuple(torch.randn(HIDDEN, HIDDEN, generator=gen).to(torch.bfloat16) for _ in range(3))
    result = torch.empty(3 * HIDDEN, HIDDEN, dtype=torch.bfloat16)
    for (_, index), part in zip(sources, parts):
        result[index] = part
    assert torch.equal(result, torch.cat(parts, 0))


# --- the splice --------------------------------------------------------------------


class _StubTower:
    """The tower's contract, with known rows: request r's image row k is the vector
    `1000 * (r + 1) + k`, in the text width. Counts its calls per request."""

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden
        self.calls: list[int] = []

    def forward(self, params: list[MultimodalParams]) -> torch.Tensor:
        out = []
        for p in params:
            r = p.multimodal_data["image"]["request"]
            n = p.multimodal_data["image"]["rows"]
            self.calls.append(r)
            out.append(
                (1000.0 * (r + 1) + torch.arange(n, dtype=torch.float32))[:, None].expand(
                    n, self.hidden
                )
            )
        return torch.cat(out).to(torch.bfloat16)


def _prompt(text_before: int, rows: int, cols: int, text_after: int) -> list[int]:
    """BOS, text, an image of `rows` x `cols` [IMG] with [IMG_BREAK] after each row
    but the last, which ends with [IMG_END], then text."""
    ids = [1] + [100 + t for t in range(text_before)]
    for r in range(rows):
        ids += [IMG] * cols + [IMG_END if r == rows - 1 else IMG_BREAK]
    return ids + [200 + t for t in range(text_after)]


@pytest.mark.cpu_only
def test_the_splice_places_each_chunks_image_rows():
    hidden, vocab = 8, 512
    embed = torch.randn(vocab, hidden).to(torch.bfloat16)
    tower = _StubTower(hidden)
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=vocab),
        multimodal_token_ids=torch.tensor([IMG], dtype=torch.int32),
    )
    # Two image requests and a text-only one between them.
    prompts = {0: _prompt(5, 4, 6, 3), 1: [1] + list(range(300, 340)), 2: _prompt(2, 3, 5, 4)}
    image_rows = {0: 4 * 6, 2: 3 * 5}
    data = {
        r: {
            "image": {"request": r, "rows": image_rows[r]},
            "multimodal_embed_mask_cumsum": (torch.tensor(prompts[r]) == IMG).cumsum(
                0, dtype=torch.int64
            ),
        }
        for r in image_rows
    }
    # Each step's window of each request. Request 0 resumes from a reused prefix
    # that ends inside its image (position 9), then chunks; request 2 is chunked
    # from 0, its second chunk wholly inside the image.
    steps = [
        {0: (9, 20), 1: (0, 20), 2: (0, 7)},
        {0: (20, 29), 1: (20, 41), 2: (7, 15)},
        {0: (29, len(prompts[0])), 2: (15, len(prompts[2]))},
    ]
    placed = {r: [] for r in image_rows}
    for windows in steps:
        ids, params = [], []
        spans = {}
        for r, (start, end) in windows.items():
            spans[r] = (len(ids), len(ids) + end - start)
            if r in data:
                runtime = MultimodalRuntimeData(
                    past_seen_token_num=start,
                    chunk_end_pos=end,
                    embed_mask_cumsum=data[r]["multimodal_embed_mask_cumsum"],
                )
                params.append(
                    MultimodalParams(
                        multimodal_data=data[r],
                        multimodal_runtime=runtime,
                        input_ids_start_offset=len(ids),
                    )
                )
            ids += prompts[r][start:end]
        _, mm_idx = prepare_multimodal_indices(ids, model=model)
        input_ids = torch.tensor(ids, dtype=torch.int32)
        rows = modeling._image_rows(tower, params)
        h = modeling._splice_image_rows(embed, input_ids, rows, mm_idx)
        plain = nn.functional.embedding(input_ids, embed)
        is_img = input_ids == IMG
        # Every non-[IMG] position, the specials and text included, keeps its embedding.
        assert torch.equal(h[~is_img], plain[~is_img])
        for r, (lo, hi) in spans.items():
            if r in placed:
                placed[r].append(h[lo:hi][is_img[lo:hi]])

    for r, chunks in placed.items():
        got = torch.cat(chunks)
        start = steps[0][r][0]
        skipped = int((torch.tensor(prompts[r][:start]) == IMG).sum())
        whole = tower.forward(
            [MultimodalParams(multimodal_data={"image": {"request": r, "rows": image_rows[r]}})]
        )
        assert torch.equal(got, whole[skipped:]), f"request {r}: rows out of order"
    # Once per request across all its chunks (the last two calls are this test's own).
    assert sorted(tower.calls[:-2]) == [0, 2]


@pytest.mark.cpu_only
@pytest.mark.parametrize("window", ["before-image", "after-image"])
def test_a_step_without_image_rows_keeps_the_token_path(window):
    """A chunk that covers none of its request's [IMG] tokens splices nothing."""
    tower = _StubTower(4)
    ids = _prompt(3, 2, 2, 6)
    cumsum = (torch.tensor(ids) == IMG).cumsum(0, dtype=torch.int64)
    start, end = (0, ids.index(IMG)) if window == "before-image" else (len(ids) - 6, len(ids))
    param = MultimodalParams(
        multimodal_data={
            "image": {"request": 0, "rows": 4},
            "multimodal_embed_mask_cumsum": cumsum,
        },
        multimodal_runtime=MultimodalRuntimeData(
            past_seen_token_num=start, chunk_end_pos=end, embed_mask_cumsum=cumsum
        ),
    )
    assert modeling._image_rows(tower, [param]) is None
    assert tower.calls == []
    assert "multimodal_embedding" not in param.multimodal_data


@pytest.mark.cpu_only
def test_image_encoding_waits_for_the_first_image_chunk():
    tower = _StubTower(4)
    ids = _prompt(3, 2, 2, 6)
    cumsum = (torch.tensor(ids) == IMG).cumsum(0, dtype=torch.int64)
    data = {"image": {"request": 0, "rows": 4}, "multimodal_embed_mask_cumsum": cumsum}
    image_start = ids.index(IMG)
    for start, end, expected_calls in [(0, image_start, []), (image_start, len(ids), [0])]:
        param = MultimodalParams(
            multimodal_data=data,
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=start, chunk_end_pos=end, embed_mask_cumsum=cumsum
            ),
        )
        rows = modeling._image_rows(tower, [param])
        assert tower.calls == expected_calls
        if expected_calls:
            expected = (1000.0 + torch.arange(4))[:, None].expand(4, 4).to(torch.bfloat16)
            assert torch.equal(rows, expected)
        else:
            assert rows is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("cached_position", [0, 1, 2])
def test_image_prefix_hit_is_skipped_beside_partial_and_fresh_requests(cached_position):
    tower = _StubTower(4)
    ids = _prompt(3, 2, 2, 6)
    cumsum = (torch.tensor(ids) == IMG).cumsum(0, dtype=torch.int64)

    def request(index, start):
        return MultimodalParams(
            multimodal_data={
                "image": {"request": index, "rows": 4},
                "multimodal_embed_mask_cumsum": cumsum,
            },
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=start, chunk_end_pos=len(ids), embed_mask_cumsum=cumsum
            ),
        )

    cached = request(0, len(ids) - 6)
    partial = request(1, ids.index(IMG) + 1)
    fresh = request(2, 0)
    params = [partial, fresh]
    params.insert(cached_position, cached)
    expected = torch.cat([2000.0 + torch.arange(1, 4), 3000.0 + torch.arange(4)])
    expected = expected[:, None].expand(7, 4).to(torch.bfloat16)
    assert torch.equal(modeling._image_rows(tower, params), expected)
    assert tower.calls == [1, 2]
    assert "multimodal_embedding" not in cached.multimodal_data
    # The active requests keep their request-local embeddings for subsequent chunks.
    assert torch.equal(modeling._image_rows(tower, params), expected)
    assert tower.calls == [1, 2]


@pytest.mark.cpu_only
def test_image_rows_without_runtime_keep_the_full_embedding_path():
    tower = _StubTower(4)
    param = MultimodalParams(multimodal_data={"image": {"request": 0, "rows": 4}})
    expected = (1000.0 + torch.arange(4))[:, None].expand(4, 4).to(torch.bfloat16)
    assert torch.equal(modeling._image_rows(tower, [param]), expected)
    assert tower.calls == [0]


@pytest.mark.cpu_only
def test_the_splice_refuses_a_count_mismatch():
    embed = torch.randn(32, 4).to(torch.bfloat16)
    rows = torch.zeros(3, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="image rows"):
        modeling._splice_image_rows(
            embed, torch.tensor([1, IMG, IMG]), rows, torch.tensor([1, 2], dtype=torch.int64)
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sizes",
    [
        None,
        [],
        [[56, 84]],
        [[0, 84], [56, 84]],
        [[112, 84], [56, 84]],
        [[55, 84], [56, 84]],
        [[56], [56, 84]],
    ],
)
def test_vision_requires_true_image_geometry(sizes):
    param = MultimodalParams(
        multimodal_data={"image": {"pixel_values": torch.zeros(2, 3, 56, 84), "image_sizes": sizes}}
    )
    with pytest.raises(ValueError, match="[Ii]mage"):
        modeling.VisionTower._images(SimpleNamespace(patch=14, merge=2), [param])


@pytest.mark.cpu_only
def test_vision_slices_padding_by_each_true_size():
    pixels = torch.rand(2, 3, 112, 84)
    param = MultimodalParams(
        multimodal_data={"image": {"pixel_values": pixels, "image_sizes": [[56, 84], [112, 56]]}}
    )
    images = modeling.VisionTower._images(SimpleNamespace(patch=14, merge=2), [param])
    assert [image[1:] for image in images] == [(4, 6), (8, 4)]
    assert torch.equal(images[0][0], pixels[0, :, :56, :84])
    assert torch.equal(images[1][0], pixels[1, :, :112, :56])


@needs_gpu
def test_full_tower_images_are_independent(target):
    """Full checkpoint vision outputs retain image identity beside other images."""
    from PIL import Image
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import adapt_config_dict
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer
    from tensorrt_llm._torch.models.modeling_mistral import MistralCommonImageProcessor
    from tensorrt_llm.mapping import Mapping

    checkpoint = os.environ.get(_CKPT_ENV)
    if not checkpoint:
        pytest.skip("Requires MISTRAL_LARGE4_CHECKPOINT_DIR for full vision weights")
    root = Path(checkpoint)
    config = adapt_config_dict(json.loads((root / "params.json").read_text()))
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=config,
            torch_dtype=torch.bfloat16,
            mapping=Mapping(world_size=4, tp_size=4, moe_ep_size=4),
        ),
        w={},
    )
    table = weights.MODEL_WEIGHTS
    dimensions = table.dims(core)
    index = json.loads((root / "consolidated.safetensors.index.json").read_text())["weight_map"]
    with torch.inference_mode():
        for entry in table.WEIGHTS:
            if entry.tag != "v" and not entry.name.startswith("vis_"):
                continue
            for key, layer, prefix in table._targets(entry, dimensions):
                destination = torch.empty(
                    entry.shape(dimensions), dtype=entry.dtype or dimensions.dtype, device="cuda"
                )
                for name, selection in entry.sources(dimensions, layer, prefix):
                    with safe_open(str(root / index[name]), framework="pt") as shard:
                        value = shard.get_tensor(name)
                    if entry.transform:
                        value = entry.transform(value, core)
                    destination[selection if selection is not None else ...].copy_(value)
                core.w[key] = destination
        tower = modeling.VisionTower(core)
        processor = MistralCommonImageProcessor(
            MistralTokenizer.from_pretrained(checkpoint), torch.bfloat16, native_chat=True
        )
        red = Image.new("RGB", (224, 168), "red")
        blue = Image.new("RGB", (168, 280), "blue")
        green = Image.new("RGB", (336, 112), "green")

        def encode(images):
            processed = processor(text="", images=images)
            params = [MultimodalParams(multimodal_data={"image": processed})]
            output = tower.forward(params)
            assert output.shape[0] == sum(
                height * width // (PATCH * 2) ** 2
                for height, width in processed["image_sizes"].tolist()
            )
            return output

        singles = [encode([image]) for image in (red, blue, green)]
        for order in ((0, 1), (1, 0), (0, 2), (0, 1, 1)):
            images = (red, blue, green)
            actual = encode([images[index] for index in order])
            expected = torch.cat([singles[index] for index in order])
            relative_rms = (
                actual.float() - expected.float()
            ).square().mean().sqrt() / expected.float().square().mean().sqrt()
            assert relative_rms < 0.005, (order, relative_rms.item())
