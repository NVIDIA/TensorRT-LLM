# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The specialized Kimi K3.1 Rubin targets' vision tower on one GPU, against the generic target's.

dep16's and tep8's flat tower (their ``modeling.py`` vision section over their ``w4_rows``) runs
the generic target's ``K31VisionTower`` call for call on the same entries; that tower is itself
bit-exact to the engine's MoonViT3d (``kimi_k31_pro_mxfp4__sm_10x__default/test_vision.py``).

* On random weights: the image rows of requests of several grids, at the processor's default
  and at the K3.1 checkpoint's image limits, equal the generic tower's bit for bit; a request past
  the encoder budget is the generic tower's encode of each group; the KV-capacity profiling image
  is the generic tower's largest image.
* With ``K31_NVFP4_CKPT`` naming the ModelOpt NVFP4 export: the vision rows the weight table loads
  equal the parameters the generic tower loads; images through the checkpoint's HF processor
  (the LLM API's hashing wrapper) encode to the generic tower's rows; the spliced prompt
  embeddings, with the checkpoint's embedding table, are the generic splice's; and two different
  images of one size behind the same text (equal prompt ids) never share a KV cache block: the
  engine's block-reuse keys, and a live radix tree committed chunk by chunk as a context server
  commits, lend one image's prompt only the text before the other's image.
"""

from __future__ import annotations

import importlib
import os
import time
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
    pytest.skip("the vision entries run on sm_103 / sm_107", allow_module_level=True)

from PIL import Image, ImageDraw  # noqa: E402
from torch import nn  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (  # noqa: E402
    KimiK31VisionConfig,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.input_processor import (  # noqa: E402
    K31InputProcessor,
    moonvit_max_image_grid,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E402
    K31VisionTower,
    build_vision_tower,
    splice_image_rows,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams  # noqa: E402

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_LAYOUTS = ("dep16", "tep8")
_TEXT = 7168
_PLACEHOLDER = 163605
#: K3.1's checkpoint processor limits (``preprocessor_config.json`` ``media_proc_cfg``).
_K31_MEDIA = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "merge_kernel_size": 2,
    "patch_limit_on_one_side": 512,
}
_CKPT = os.environ.get("K31_NVFP4_CKPT")

_NEEDS_CKPT = pytest.mark.skipif(
    not _CKPT or not os.path.isdir(_CKPT), reason="needs K31_NVFP4_CKPT (the ModelOpt NVFP4 export)"
)


def _modules(layout: str):
    target = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__{layout}"
    return importlib.import_module(f"{target}.modeling"), importlib.import_module(
        f"{target}.weights"
    )


def _vision_dims(W):
    return W.Dims(*([0] * len(W.Dims._fields)))._replace(vision=True)


def _flat(layout: str, by_key: dict, media: dict | None):
    """The target's tower view over ``by_key`` (checkpoint name -> tensor) and its processor limits."""
    M, W = _modules(layout)
    d = _vision_dims(W)
    w = {row.name: by_key[row.sources(d, None)[0][0]] for row in W.w4_rows(d)}
    return M, M._vision_view(SimpleNamespace(dims=d, w=w, media_proc_cfg=media))


def _random_tower(seed: int = 0) -> dict:
    """Random tower weights at the export's shapes, under the checkpoint's names, on the GPU."""
    _, W = _modules("dep16")
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for row in W.w4_rows(_vision_dims(W)):
        key = row.sources(None, None)[0][0]
        shape = row.shape(None)
        if key.endswith(
            ("norm0.weight", "norm1.weight", "final_layernorm.weight", "post_norm.weight")
        ):
            value = 1.0 + 0.1 * torch.randn(shape, generator=gen)
        elif key.endswith("pos_emb.weight"):
            value = torch.randn(shape, generator=gen)
        else:
            value = 0.02 * torch.randn(shape, generator=gen)
        out[key] = value.bfloat16().cuda()
    return out


def _generic(by_key: dict, media: dict | None) -> K31VisionTower:
    tower = K31VisionTower(
        KimiK31VisionConfig().to_dict(), _TEXT, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    assert tower.load_weights(by_key)
    if media is not None:
        tower.max_image_grid = tower.largest_image_grid(media)
    return tower


#: The images of each request: the table's own 64x64 grid, resampled grids, the smallest 2x2 grid.
_REQUEST_GRIDS = [[(1, 32, 48), (1, 2, 2)], [(1, 64, 64)], [(1, 20, 34)]]


def _params(grids_per_request, seed: int = 1) -> list[MultimodalParams]:
    gen = torch.Generator().manual_seed(seed)
    out = []
    for grids in grids_per_request:
        patches = sum(t * h * w for t, h, w in grids)
        image = {
            "pixel_values": torch.randn(patches, 3, 14, 14, generator=gen).cuda(),
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64).cuda(),
        }
        out.append(MultimodalParams(multimodal_data={"image": image}))
    return out


@pytest.fixture(scope="module")
def random_tower():
    return _random_tower()


@pytest.mark.parametrize("media", [None, _K31_MEDIA], ids=["default_limits", "k31_limits"])
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_rows_match_the_generic_tower(layout, media, random_tower):
    M, flat = _flat(layout, random_tower, {} if media is None else media)
    tower = _generic(random_tower, media)
    assert flat.max_grid == tower.max_image_grid and flat.capacity == tower.attention_capacity
    params = _params(_REQUEST_GRIDS)
    [expected] = tower.forward(params)
    [rows] = M.vision_forward(flat, params)
    assert rows.shape == expected.shape and rows.dtype == torch.bfloat16
    assert torch.isfinite(rows.float()).all()
    assert torch.equal(rows, expected)


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_a_request_past_the_budget_is_the_generic_encode_of_each_group(layout, random_tower):
    M, flat = _flat(layout, random_tower, {})
    tower = _generic(random_tower, None)
    grids = [(1, 64, 64), (1, 64, 64), (1, 32, 48), (1, 64, 64), (1, 2, 2)]
    budget = 2 * 64 * 64 + 32 * 48
    flat = flat._replace(capacity=(budget, flat.capacity[1]))
    [param] = _params([grids], seed=2)
    [rows] = M.vision_forward(flat, [param])
    pixels = param.multimodal_data["image"]["pixel_values"]
    split = 2 * 4096 + 1536
    expected = torch.cat(
        [tower.encode(pixels[:split], grids[:3]), tower.encode(pixels[split:], grids[3:])]
    )
    assert torch.equal(rows, expected)


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_the_profiling_image_is_the_generic_largest_image(layout, random_tower):
    """The KV-capacity estimate's image: the K3.1 processor's largest grid, the generic tower's
    rows; its peak memory and time are printed."""
    M, flat = _flat(layout, random_tower, _K31_MEDIA)
    estimating = SimpleNamespace(
        num_contexts=1, kv_cache_manager=SimpleNamespace(is_estimating_kv_cache=True)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    rows = M.vision_profiling_rows(flat, estimating, False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - base
    t, h, w = flat.max_grid
    print(
        f"{layout}: largest image {flat.max_grid}, {t * h * w} patches -> {rows.shape[0]} rows, "
        f"peak {peak / 2**30:.2f} GiB, {elapsed:.2f} s"
    )
    assert flat.max_grid == (1, 220, 306) and rows.shape == (16830, _TEXT)
    assert torch.equal(rows, _generic(random_tower, _K31_MEDIA).encode_largest_image())


# --- the checkpoint's own tower and images ---


@pytest.fixture(scope="module")
def checkpoint():
    _, W = _modules("dep16")
    with W.open_checkpoint_weights(_CKPT) as weights:
        yield weights


@pytest.fixture(scope="module")
def composite_config():
    from tensorrt_llm._torch.model_config import ModelConfig

    return ModelConfig.from_pretrained(_CKPT, trust_remote_code=True).pretrained_config


def _loaded(layout: str, checkpoint) -> dict:
    """The vision rows as the target's weight table loads them (checkpoint name -> parameter)."""
    _, W = _modules(layout)
    d = _vision_dims(W)
    rows = W.w4_rows(d)
    with torch.device("cuda"):
        params = W.declare(d, rows)
    consumed = W.fill_table(
        SimpleNamespace(dims=d), checkpoint, rows, {f"w.{k}": v for k, v in params.items()}
    )
    assert consumed == {k for k in checkpoint.keys() if k.startswith(W.CKPT_VISION)}
    torch.cuda.synchronize()
    return {row.sources(d, None)[0][0]: params[row.name] for row in rows}


@pytest.fixture(scope="module")
def generic_checkpoint_tower(checkpoint, composite_config):
    tower = build_vision_tower(composite_config, _CKPT)
    assert tower.load_weights(checkpoint)
    return tower


@_NEEDS_CKPT
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_the_table_loads_the_generic_towers_parameters(
    layout, checkpoint, generic_checkpoint_tower
):
    _, W = _modules(layout)
    loaded = _loaded(layout, checkpoint)
    generic = generic_checkpoint_tower._checkpoint_parameters()
    assert set(loaded) == set(generic) and len(loaded) == 168
    for key, param in loaded.items():
        assert param.dtype == generic[key].dtype == torch.bfloat16, key
        assert torch.equal(param, generic[key]), key
    media = W.media_proc_cfg(checkpoint)
    assert moonvit_max_image_grid(media, (2, 2), 14) == generic_checkpoint_tower.max_image_grid


def _text_image(text: str, size=(640, 200)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((20, size[1] // 3), text, fill="black")
    return image


def _pattern_image(size, seed: int) -> Image.Image:
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8)
    return Image.fromarray(pixels.numpy())


_IMAGE = "<|kimi_image_placeholder|>"
_PROCESSORS: dict = {}


def _processor(composite_config) -> K31InputProcessor:
    """The checkpoint's K3.1 input processor (the one the engine routes by model type, for every target)."""
    if id(composite_config) not in _PROCESSORS:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_CKPT, trust_remote_code=True)
        _PROCESSORS[id(composite_config)] = K31InputProcessor(
            _CKPT, composite_config, tokenizer, trust_remote_code=True
        )
    return _PROCESSORS[id(composite_config)]


def _process(composite_config, images, prompt_text="Read the image.", prompt=None):
    """Token ids and processed media, as the LLM API's hashing wrapper hands them to the executor.
    ``prompt`` (with one image placeholder per image) replaces the default images-then-text prompt."""
    from tensorrt_llm.inputs import create_input_processor_with_hash
    from tensorrt_llm.sampling_params import SamplingParams

    processor = _processor(composite_config)
    if prompt is None:
        prompt = _IMAGE * len(images) + prompt_text
    assert prompt.count(_IMAGE) == len(images)
    token_ids, extra = create_input_processor_with_hash(processor)(
        {"prompt": prompt, "multi_modal_data": {"image": images}}, SamplingParams()
    )
    assert extra.get("multimodal_input") is not None
    return token_ids, extra


def _device_param(extra) -> MultimodalParams:
    image = extra["multimodal_data"]["image"]
    return MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": image["pixel_values"].cuda(),
                "image_grid_thw": image["image_grid_thw"].cuda(),
            }
        }
    )


@pytest.fixture(scope="module")
def processed(composite_config):
    """Three requests: a text image and a pattern image; one large image (downscaled by the
    processor); one tall image."""
    requests = [
        [_text_image("RUBIN 2026 OCR"), _pattern_image((1024, 768), 0)],
        [_pattern_image((3000, 2000), 1)],
        [_pattern_image((224, 1800), 2)],
    ]
    return [_process(composite_config, images) for images in requests]


@_NEEDS_CKPT
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_checkpoint_images_encode_to_the_generic_rows(
    layout, checkpoint, generic_checkpoint_tower, processed
):
    M, flat = _flat(layout, _loaded(layout, checkpoint), _K31_MEDIA)
    params = [_device_param(extra) for _, extra in processed]
    [rows] = M.vision_forward(flat, params)
    [expected] = generic_checkpoint_tower.forward(params)
    placeholders = sum(ids.count(_PLACEHOLDER) for ids, _ in processed)
    assert rows.shape == expected.shape == (placeholders, _TEXT)
    assert torch.isfinite(rows.float()).all()
    assert torch.equal(rows, expected)


@_NEEDS_CKPT
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_the_splice_is_the_generic_splice(layout, checkpoint, processed):
    """The step's prompt embeddings, with the checkpoint's embedding table: the image rows over the
    placeholders, the token embeddings elsewhere, as the generic target splices them."""
    M, flat = _flat(layout, _loaded(layout, checkpoint), _K31_MEDIA)
    embed = checkpoint["language_model.model.embed_tokens.weight"][:].cuda()
    ids, extra = processed[0]
    input_ids = torch.tensor(ids, dtype=torch.int32).cuda()
    indices = torch.nonzero(input_ids == _PLACEHOLDER).flatten()
    param = _device_param(extra)
    embeds = M.image_inputs_embeds(flat, embed, input_ids, [param], indices, 1)
    [rows] = M.vision_forward(flat, [param])
    expected = splice_image_rows(nn.Embedding.from_pretrained(embed), input_ids, [rows], indices)
    assert torch.equal(embeds, expected)
    text = input_ids != _PLACEHOLDER
    assert torch.equal(embeds[text], nn.functional.embedding(input_ids[text], embed))


# --- KV cache reuse across images ---

#: Text around the images: whole 64-token blocks before the first image and after the last one.
_PREFIX = "This page was scanned from a shipping manifest and attached to the request below. " * 12
_SUFFIX = (
    " Transcribe every word in the image exactly, keeping the original line breaks and spacing."
    * 10
)
#: The context servers' ``kv_cache_config.tokens_per_block``.
_TPB = 64


def _reuse_request(ids, extra):
    """The executor's request for a processed prompt, with the multimodal reuse metadata the LLM API sends."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

    mm = extra["multimodal_input"]
    return LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(ids),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        multimodal_hashes=mm.multimodal_hashes,
        multimodal_positions=mm.multimodal_positions,
        multimodal_lengths=mm.multimodal_lengths,
        multimodal_item_run_cu_offsets=mm.multimodal_item_run_cu_offsets,
        multimodal_run_positions=mm.multimodal_run_positions,
        multimodal_run_lengths=mm.multimodal_run_lengths,
    )


def _reuse_keys(composite_config, ids, extra, start=0, end=None) -> list:
    """The radix-tree tokens a context server matches and commits for ``ids[start:end]``:
    ``KVCacheManagerV2._augment_tokens_for_block_reuse`` itself, with the vocabulary size the engine
    resolves from the served config. Nothing of the target is on this path."""
    from tensorrt_llm._torch.pyexecutor.config_utils import resolve_vocab_size
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.vocab_size = resolve_vocab_size(composite_config)
    request = _reuse_request(ids, extra)
    return list(
        KVCacheManagerV2._augment_tokens_for_block_reuse(manager, list(ids), request, start, end)
    )


def _image_runs(ids) -> list[tuple[int, int]]:
    """``(first position, token count)`` of each run of image placeholders."""
    runs = []
    for i, token in enumerate(ids):
        if token != _PLACEHOLDER:
            continue
        if runs and runs[-1][0] + runs[-1][1] == i:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((i, 1))
    return runs


@_NEEDS_CKPT
def test_two_images_of_one_size_never_share_kv_blocks(composite_config):
    """Two different images of one size behind the same text have the same prompt ids; only their
    content hashes differ. The engine's block-reuse keys (the same for every K3.1 target: the input
    processor is routed by model type, and the KV cache manager reads only the request) mark an
    image once, on its first token, with its content digest, and give its later tokens
    ``vocab_size + offset`` (``gen_multimodal_cache_key_tokens``); the radix tree chains each
    block's key through every block before it (``sequence_to_blockchain_keys``). So the keys of
    every block from the one holding the image's first token on differ between the two images, and
    a live radix tree holding one prompt, committed chunk by chunk with a chunk boundary inside the
    image, lends the other prompt only the text before its image, while the same prompt again
    reuses the image. A second image is kept apart the same way, from its own first token on."""
    import itertools

    from tensorrt_llm._torch.pyexecutor.config_utils import resolve_vocab_size
    from tensorrt_llm._torch.pyexecutor.engine.multimodal import is_multimodal
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import _hash_to_digest
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        ReuseScope,
        gen_multimodal_cache_key_tokens,
        sequence_to_blockchain_keys,
    )

    # Any K3.1 target gets multimodal reuse keys: the engine asks the processor (no K3.1 model class
    # is a MultimodalModelMixin), and the key offset is the served config's vocabulary size.
    assert is_multimodal(object(), _processor(composite_config))
    vocab = resolve_vocab_size(composite_config)
    assert vocab == 163840

    def one(text):
        return _process(composite_config, [_text_image(text)], prompt=_PREFIX + _IMAGE + _SUFFIX)

    (a_ids, a), (b_ids, b), (a2_ids, a2) = one("ALPHA 111"), one("BRAVO 222"), one("ALPHA 111")
    assert a_ids == b_ids == a2_ids
    p0, n = (
        a["multimodal_input"].multimodal_positions[0],
        a["multimodal_input"].multimodal_lengths[0],
    )
    assert _image_runs(a_ids) == [(p0, n)]  # the item the engine keys is the placeholder run
    total = len(a_ids)
    digest_a = _hash_to_digest(a["multimodal_input"].multimodal_hashes[0])
    digest_b = _hash_to_digest(b["multimodal_input"].multimodal_hashes[0])
    assert digest_a != digest_b
    assert a2["multimodal_input"].multimodal_hashes == a["multimodal_input"].multimodal_hashes
    # Whole text blocks on both sides of the image, and a chunk boundary inside it.
    assert p0 >= 2 * _TPB and (total - 1) // _TPB * _TPB > p0 + n and n > 37

    # 1. The keys, token by token: the digest on the image's first token, the offsets after it.
    ka, kb, ka2 = (
        _reuse_keys(composite_config, ids, x) for ids, x in ((a_ids, a), (b_ids, b), (a2_ids, a2))
    )
    assert ka == ka2
    assert ka[p0 : p0 + n] == gen_multimodal_cache_key_tokens(vocab, digest_a, n)
    assert kb[p0 : p0 + n] == gen_multimodal_cache_key_tokens(vocab, digest_b, n)
    assert ka[p0] == digest_a and kb[p0] == digest_b
    assert ka[p0 + 1 : p0 + n] == kb[p0 + 1 : p0 + n] == [vocab + k for k in range(1, n)]
    text = [i for i in range(total) if not p0 <= i < p0 + n]
    assert [ka[i] for i in text] == [kb[i] for i in text] == [a_ids[i] for i in text]

    # 2. The block keys the radix tree matches: equal only for the text blocks before the image.
    for tpb in (16, _TPB):

        def chain(keys, tpb=tpb):
            steps = sequence_to_blockchain_keys(tpb, ReuseScope(), keys)
            return [(len(block), key) for block, key in steps if block]

        chain_a, chain_b = chain(ka), chain(kb)
        assert chain_a == chain(ka2)
        ends = list(itertools.accumulate(size for size, _ in chain_a))
        same = [ya == yb for (_, ya), (_, yb) in zip(chain_a, chain_b, strict=True)]
        assert same == [end <= p0 for end in ends], (tpb, p0, ends, same)

    # 3. A live radix tree, committed and matched as a context server does.
    stream = torch.cuda.Stream()
    manager = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=_TPB,
            cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=8192),
                        BufferConfig(role=DataRole("value"), size=8192),
                    ],
                )
            ],
        )
    )

    def commit(scope, ids, keys_of, cuts):
        """One prompt's context: matched with its last token held back, then committed chunk by
        chunk from the reused prefix on, each chunk's own slice of the keys (as the KV cache
        manager commits a context chunk)."""
        kv = manager.create_kv_cache(scope, keys_of(0, len(ids) - 1))
        assert kv.resume(CudaStream(stream.cuda_stream))
        assert kv.resize(len(ids))
        done = kv.num_committed_tokens
        for end in (*(cut for cut in cuts if cut > done), len(ids)):
            kv.commit(keys_of(done, end))
            done = end
        kv.close()
        torch.cuda.synchronize()

    def lent(scope, ids, keys_of):
        return manager.probe_reuse(scope, keys_of(0, len(ids) - 1))

    def engine_keys(ids, extra):
        return lambda start, end: _reuse_keys(composite_config, ids, extra, start, end)

    try:
        scope = ReuseScope()
        commit(scope, a_ids, engine_keys(a_ids, a), cuts=(p0 + 37,))
        lent_b = lent(scope, b_ids, engine_keys(b_ids, b))
        lent_a2 = lent(scope, a2_ids, engine_keys(a2_ids, a2))
        # Control: the raw prompt ids (no digests) would lend BRAVO's prompt ALPHA's image.
        raw = ReuseScope(salt="raw prompt ids")
        commit(raw, a_ids, lambda start, end: list(a_ids[start:end]), cuts=(p0 + 37,))
        lent_raw = lent(raw, b_ids, lambda start, end: list(b_ids[start:end]))
        print(
            f"reuse: image [{p0}, {p0 + n}) of {total} tokens, {_TPB}-token blocks; ALPHA's tree "
            f"lends BRAVO {lent_b}, ALPHA again {lent_a2}; with raw ids it would lend BRAVO {lent_raw}"
        )
        assert p0 // _TPB * _TPB <= lent_b <= p0
        assert lent_a2 >= (total - 1) // _TPB * _TPB
        assert lent_raw >= (total - 1) // _TPB * _TPB

        # Two images (MMMU-Pro sends up to seven): a prompt is kept apart from its first differing image on.
        def two(first, second):
            images = [_text_image(first), _text_image(second)]
            prompt = _PREFIX + _IMAGE + " and the second page: " + _IMAGE + _SUFFIX
            return _process(composite_config, images, prompt=prompt)

        (x_ids, x), (y_ids, y), (z_ids, z) = (
            two("ALPHA 111", "CHARLIE 333"),
            two("ALPHA 111", "DELTA 444"),
            two("BRAVO 222", "CHARLIE 333"),
        )
        assert x_ids == y_ids == z_ids
        mm = x["multimodal_input"]
        [(q0, _), (r0, m)] = _image_runs(x_ids)
        assert [q0, r0] == list(mm.multimodal_positions)
        commit(scope, x_ids, engine_keys(x_ids, x), cuts=(r0 + m // 2,))
        lent_y = lent(scope, y_ids, engine_keys(y_ids, y))
        lent_z = lent(scope, z_ids, engine_keys(z_ids, z))
        print(
            f"reuse: images at {q0} and {r0}; [A, C]'s tree lends [A, D] {lent_y}, [B, C] {lent_z}"
        )
        assert r0 // _TPB * _TPB <= lent_y <= r0
        assert q0 // _TPB * _TPB <= lent_z <= q0
    finally:
        manager.shutdown()
