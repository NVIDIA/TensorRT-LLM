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
"""DeepSeek-V4.1 input processor against the checkpoint's own reference code.

The oracle is the checkpoint itself: ``inference/image_processor.py`` (``prepare_vl_inputs``) for
token ids, per-position types and patches, ``encoding/encoding.py`` for prompt rendering,
``inference/vision.py`` for the ViT rotary table and the aligner's unfold, and
``Transformer.merge_image_embeddings`` in ``inference/model.py`` for the span splice. Every
comparison is exact (``torch.equal``, list equality).

Cases: the checkpoint's own two-image example (carrots/corn), synthetic images from 1x1 to
6000x4000 and 50000x14 in both orientations, every runtime image format, token-id input, and
MMMU validation documents rendered the way ``trtllm-eval mmmu`` renders them (placeholders
before the text, joined by a blank line, through the ``deepseek_v4`` chat renderer); the
single-image ones are also checked against the prompt lengths the external-engine anchor reported.

Two deliberately wrong controls show the comparisons can fail: resizing instead of padding, and
joining the placeholders with ``"\\n"`` instead of ``"\\n\\n"``.

The checkpoint and the MMMU cache are external directories; tests that need one skip, naming it,
when it is absent.
"""

import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.input_processor import (  # noqa: E501
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    TEXT,
    DeepseekV41InputProcessor,
    VisionGeometry,
    plan_image_grid,
    preprocess_image,
    to_rgb_image,
)
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
# The HF datasets cache of MMMU and the external-engine anchor's MMMU records; no default
# location, so the cases that read them skip unless these are set.
_MMMU_DIR = os.environ.get("MODELING_V2_DSV41_MMMU_DIR", "")
_ANCHOR_MMMU_DIR = os.environ.get("MODELING_V2_DSV41_ANCHOR_MMMU_DIR", "")
# 0: the first document of every subject plus every multi-image document; 1: all 900.
_MMMU_ALL = os.environ.get("MODELING_V2_DSV41_MMMU_ALL", "0") == "1"

_IMAGE_TOKEN_ID = 129264
_REFERENCE_MODULES = ("encoding", "image_processor", "vision", "kernel", "engram", "model")


def _require(path: str) -> Path:
    if not os.path.isdir(path):
        pytest.skip(f"external directory not present: {path}")
    return Path(path)


def _load_as(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ckpt() -> Path:
    return _require(_CKPT_DIR)


@pytest.fixture(scope="module")
def ref(ckpt):
    """The checkpoint's reference modules, importable by the names they import each other by."""
    saved = {name: sys.modules.get(name) for name in _REFERENCE_MODULES}
    try:
        ns = types.SimpleNamespace()
        ns.encoding = _load_as("encoding", ckpt / "encoding" / "encoding.py")
        ns.image_processor = _load_as("image_processor", ckpt / "inference" / "image_processor.py")
        ns.vision = _load_as("vision", ckpt / "inference" / "vision.py")
        # model.py imports its GPU kernels and the Engram hash at module scope; the span splice
        # tested here uses neither.
        kernel = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn"):
            setattr(kernel, name, None)
        kernel.sparse_attn = None
        sys.modules["kernel"] = kernel
        engram = types.ModuleType("engram")
        engram.EngramLayout = engram.NgramHashState = object
        sys.modules["engram"] = engram
        ns.model = _load_as("model", ckpt / "inference" / "model.py")
        yield ns
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def ref_args(ckpt):
    """The reference ModelArgs' vision fields, from the checkpoint's inference config."""
    cfg = json.loads((ckpt / "inference" / "config.json").read_text())
    args = types.SimpleNamespace(**cfg)
    args.vision_enabled = cfg["vision_n_layers"] > 0
    return args


@pytest.fixture(scope="module")
def hf_config(ckpt):
    from transformers import AutoConfig

    import tensorrt_llm._torch.configs  # noqa: F401  registers deepseek_v41 with AutoConfig

    return AutoConfig.from_pretrained(str(ckpt))


@pytest.fixture(scope="module")
def tokenizer(ckpt, hf_config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ckpt))


@pytest.fixture(scope="module")
def processor(ckpt, hf_config, tokenizer):
    return DeepseekV41InputProcessor(str(ckpt), hf_config, tokenizer)


@pytest.fixture(scope="module")
def chat_renderer(tokenizer):
    """The renderer ``custom_tokenizer: deepseek_v4`` installs."""
    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer

    return DeepseekV4Tokenizer(tokenizer)


def _ppm_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PPM")
    return buf.getvalue()


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _random_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")


def _run(processor, prompt, images, **sampling):
    inputs = TextPrompt(prompt=prompt)
    if images:
        inputs["multi_modal_data"] = {"image": images}
    return processor(inputs, SamplingParams(**sampling))


def _assert_matches_reference(ours, reference):
    """``ours``: the processor's (ids, extra); ``reference``: prepare_vl_inputs' triple."""
    ids, extra = ours
    ref_tokens, ref_types, ref_images = reference
    assert ids == ref_tokens
    if ref_images is None:
        assert extra is None
        assert all(t == TEXT for t in ref_types)
        return
    image = extra["multimodal_data"]["image"]
    layout = extra["multimodal_data"]["token_layout"]
    assert layout["token_types"].tolist() == ref_types
    assert image["span_starts"] == [img.start for img in ref_images]
    assert image["span_lengths"] == [img.types.numel() for img in ref_images]
    assert image["vit_grid_hw"] == [[img.n_vit_h, img.n_vit_w] for img in ref_images]
    assert torch.equal(image["span_types"], torch.cat([img.types for img in ref_images]))
    assert torch.equal(image["pixel_values"], torch.cat([img.patches for img in ref_images]))
    assert image["pixel_values"].dtype == torch.bfloat16
    # Every image id in the prompt is a span position and every span position is one.
    ids_t = torch.tensor(ids)
    assert torch.equal(layout["image_mask"], ids_t == _IMAGE_TOKEN_ID)
    assert torch.equal(layout["engram_dead_mask"], layout["image_mask"])


def _reference_prompt(ref, num_images: int, text: str) -> str:
    """encoding.py's rendering of one user turn: N image blocks, then the text."""
    blocks = [{"type": "image_url", "image_url": {"url": f"img{i}"}} for i in range(num_images)]
    blocks.append({"type": "text", "text": text})
    return ref.encoding.encode_messages([{"role": "user", "content": blocks}], thinking_mode="chat")


def _string_path_prompt(chat_renderer, num_images: int, text: str, separator: str = "\n\n") -> str:
    """What trtllm-eval builds on the STRING path: placeholders before the text, then the chat
    renderer (``inputs.utils.apply_chat_template``'s DeepSeek branch)."""
    content = separator.join([IMAGE_PLACEHOLDER] * num_images + [text])
    return chat_renderer.apply_chat_template(
        messages=[{"role": "user", "content": content}],
        tools=None,
        thinking=False,
        continue_final_message=False,
    )


# ---------------------------------------------------------------------------------------------
# Geometry


def test_geometry_matches_the_reference_args(hf_config, ref_args):
    geometry = VisionGeometry.from_config(hf_config)
    assert geometry.patch_size == ref_args.vision_patch_size == 14
    assert geometry.downsample_ratio == ref_args.vision_downsample_ratio == 3
    assert geometry.max_image_tokens == ref_args.vision_max_n_token == 1024
    assert geometry.min_pixels == ref_args.vision_min_pixels == 295936
    assert geometry.max_wh_ratio is ref_args.vision_max_wh_ratio is None
    assert geometry.hidden_size == ref_args.vision_dim == 1024
    assert geometry.num_heads == ref_args.vision_n_heads == 16
    assert geometry.num_layers == ref_args.vision_n_layers == 32
    assert geometry.rope_theta == ref_args.vision_rope_theta
    assert geometry.rope_dim == ref_args.vision_dim // ref_args.vision_n_heads // 2 == 32
    assert geometry.image_token_id == ref_args.image_token_id == _IMAGE_TOKEN_ID


def test_grid_plan_matches_the_reference_on_a_size_sweep(hf_config, ref, ref_args):
    geometry = VisionGeometry.from_config(hf_config)
    rng = np.random.default_rng(0)
    sizes = [(int(w), int(h)) for w, h in rng.integers(1, 12000, size=(2000, 2))]
    sizes += [(int(w), int(h)) for w, h in rng.integers(1, 800, size=(2000, 2))]
    sizes += [(1, 1), (544, 544), (543, 545), (14, 14), (42, 42), (1, 100000), (100000, 1)]
    sizes += [(50000, 14), (14, 50000), (768, 384), (1344, 1344), (1387, 1387), (1388, 1388)]
    for width, height in sizes:
        grid = plan_image_grid(width, height, geometry)
        expected = ref.image_processor.plan_image_grid(width, height, ref_args)
        assert (grid.n_llm_h, grid.n_llm_w, grid.best_height, grid.best_width) == expected, (
            width,
            height,
        )
        assert grid.num_span_tokens <= geometry.max_image_tokens


# ---------------------------------------------------------------------------------------------
# The checkpoint's own examples


def _example_cases(ckpt, ref):
    raw = (ckpt / "inference" / "examples" / "example.txt").read_text().rstrip("\n").split("\n\n")
    image_dir = str(ckpt / "inference" / "examples" / "images")
    for prompt in raw:
        prompt = prompt.replace("examples/images", image_dir)
        case = {"messages": [{"role": "user", "content": ref.encoding.parse_tagged_text(prompt)}]}
        yield ref.encoding.encode_case(case, "chat")


def test_checkpoint_examples_match_the_reference(ckpt, ref, ref_args, tokenizer, processor):
    """example.txt: three text prompts and the carrots/corn two-image prompt."""
    num_image_cases = 0
    for prompt, records in _example_cases(ckpt, ref):
        reference = ref.image_processor.prepare_vl_inputs(prompt, records, tokenizer, ref_args)
        images = [Image.open(record["url"]) for record in records]
        ours = _run(processor, prompt, images or None)
        _assert_matches_reference(ours, reference)
        num_image_cases += bool(records)
        if records:
            assert len(records) == 2
            # The two images differ in size, so their grids and spans do.
            grids = ours[1]["multimodal_data"]["image"]["vit_grid_hw"]
            assert grids[0] != grids[1]
    assert num_image_cases == 1


# ---------------------------------------------------------------------------------------------
# Synthetic sizes

_SYNTHETIC_SIZES = [
    (1, 1),
    (5, 7),
    (13, 15),
    (544, 544),
    (545, 543),
    (768, 384),
    (1387, 1386),
    (4000, 3000),
    (6000, 4000),
    (20000, 14),
    (50000, 14),
    (14, 50000),
    (3, 4000),
]


@pytest.mark.parametrize("size", _SYNTHETIC_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_synthetic_image_matches_the_reference(ref, ref_args, tokenizer, processor, size):
    width, height = size
    image = _random_image(width, height, seed=width * 100003 + height)
    prompt = _reference_prompt(ref, 1, "Describe the image.")
    reference = ref.image_processor.prepare_vl_inputs(
        prompt, [{"data": _ppm_bytes(image)}], tokenizer, ref_args
    )
    _assert_matches_reference(_run(processor, prompt, [image]), reference)


def test_every_runtime_image_format_gives_the_same_patches(hf_config, ckpt):
    geometry = VisionGeometry.from_config(hf_config)
    image = Image.open(ckpt / "inference" / "examples" / "images" / "carrots.jpeg")
    rgb = image.convert("RGB")
    expected = preprocess_image(rgb, geometry).patches
    as_float = ToTensor()(rgb)  # the runtime's "pt" format
    as_uint8 = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).contiguous()
    as_numpy = np.asarray(rgb)  # the runtime's "np" format
    for fmt in (image, as_float, as_uint8, as_numpy):
        assert torch.equal(preprocess_image(fmt, geometry).patches, expected)
        assert np.array_equal(np.asarray(to_rgb_image(fmt)), np.asarray(rgb))


def test_three_images_with_text_between_match_the_reference(ref, ref_args, tokenizer, processor):
    images = [
        _random_image(w, h, seed=i) for i, (w, h) in enumerate([(640, 480), (30, 900), (2000, 100)])
    ]
    blocks = [
        {"type": "text", "text": "First:"},
        {"type": "image_url", "image_url": {"url": "a"}},
        {"type": "text", "text": "then"},
        {"type": "image_url", "image_url": {"url": "b"}},
        {"type": "image_url", "image_url": {"url": "c"}},
        {"type": "text", "text": "Compare them."},
    ]
    prompt = ref.encoding.encode_messages(
        [{"role": "user", "content": blocks}], thinking_mode="chat"
    )
    records = [{"data": _ppm_bytes(image)} for image in images]
    reference = ref.image_processor.prepare_vl_inputs(prompt, records, tokenizer, ref_args)
    ours = _run(processor, prompt, images)
    _assert_matches_reference(ours, reference)
    image = ours[1]["multimodal_data"]["image"]
    patches = [h * w for h, w in image["vit_grid_hw"]]
    assert image["vit_cu_seqlens"] == np.cumsum([0] + patches).tolist()
    assert image["llm_grid_hw"] == [[-(-h // 3), -(-w // 3)] for h, w in image["vit_grid_hw"]]


def test_token_id_input_matches_text_input(ckpt, ref, tokenizer, processor):
    prompt, records = list(_example_cases(ckpt, ref))[-1]
    images = [Image.open(record["url"]) for record in records]
    text_ids, text_extra = _run(processor, prompt, images)
    inputs = {"prompt_token_ids": tokenizer.encode(prompt), "multi_modal_data": {"image": images}}
    ids, extra = processor(inputs, SamplingParams())
    assert ids == text_ids
    for key, value in text_extra["multimodal_data"]["image"].items():
        other = extra["multimodal_data"]["image"][key]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    for key, value in text_extra["multimodal_data"]["token_layout"].items():
        assert torch.equal(value, extra["multimodal_data"]["token_layout"][key])


# ---------------------------------------------------------------------------------------------
# The tables the model indexes with


def test_vit_rope_table_matches_the_reference(ref, ref_args, processor):
    images = [
        _random_image(w, h, seed=i) for i, (w, h) in enumerate([(544, 544), (30, 900), (1500, 200)])
    ]
    prompt = _reference_prompt(ref, len(images), "x")
    _, extra = _run(processor, prompt, images)
    image = extra["multimodal_data"]["image"]
    cu = image["vit_cu_seqlens"]
    dim = ref_args.vision_dim // ref_args.vision_n_heads // 2
    for i, (n_h, n_w) in enumerate(image["vit_grid_hw"]):
        cos, sin = ref.vision.get_vision_cos_sin(n_h, n_w, dim, ref_args.vision_rope_theta)
        assert torch.equal(image["vit_rope_cos"][cu[i] : cu[i + 1]], cos.squeeze(1))
        assert torch.equal(image["vit_rope_sin"][cu[i] : cu[i + 1]], sin.squeeze(1))


def test_aligner_index_reproduces_the_reference_unfold(ref, ref_args, processor):
    """One gather over all images == the reference Aligner's w1 input, image by image."""
    sizes = [(14, 14), (70, 28), (544, 544), (768, 384), (2000, 100), (30, 900)]
    images = [_random_image(w, h, seed=i) for i, (w, h) in enumerate(sizes)]
    _, extra = _run(processor, _reference_prompt(ref, len(images), "x"), images)
    image = extra["multimodal_data"]["image"]
    width = 16
    num_patches = image["vit_cu_seqlens"][-1]
    vit_out = torch.randn(num_patches, width, generator=torch.Generator().manual_seed(0))

    gathered = torch.cat([vit_out, vit_out.new_zeros(1, width)])[image["aligner_index"]]
    ours = gathered.transpose(1, 2).reshape(gathered.shape[0], -1)

    args = types.SimpleNamespace(vision_downsample_ratio=3, vision_dim=width, dim=8)
    aligner = ref.vision.Aligner(args)
    captured = []
    aligner.w1.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].clone()))
    cu = image["vit_cu_seqlens"]
    for i, (n_h, n_w) in enumerate(image["vit_grid_hw"]):
        aligner(vit_out[cu[i] : cu[i + 1]], n_h, n_w)
    assert torch.equal(ours, torch.cat(captured))
    n_cells = sum(h * w for h, w in image["llm_grid_hw"])
    assert image["aligner_index"].shape == (n_cells, 9)


def test_span_source_index_reproduces_the_reference_splice(ref, ref_args, processor):
    """The one-gather span assembly == Transformer.merge_image_embeddings."""
    sizes = [(544, 544), (30, 900), (2000, 100)]
    images = [_random_image(w, h, seed=i) for i, (w, h) in enumerate(sizes)]
    prompt = _reference_prompt(ref, len(images), "Compare.")
    records = [{"data": _ppm_bytes(image)} for image in images]
    tokens, _, ref_images = ref.image_processor.prepare_vl_inputs(
        prompt, records, processor.tokenizer, ref_args
    )
    ids, extra = _run(processor, prompt, images)
    image = extra["multimodal_data"]["image"]
    dim = 8
    gen = torch.Generator().manual_seed(1)
    n_cells = [h * w for h, w in image["llm_grid_hw"]]
    aligner_out = torch.randn(sum(n_cells), dim, generator=gen)
    delimiters = torch.randn(3, dim, generator=gen)
    per_image = list(torch.split(aligner_out, n_cells))

    fake = types.SimpleNamespace(
        image_start=delimiters[0], image_end=delimiters[1], image_newline=delimiters[2]
    )
    calls = iter(per_image)
    fake.encode_image = lambda patches, n_h, n_w: next(calls)
    h_ref = torch.zeros(1, len(tokens), dim)
    ref.model.Transformer.merge_image_embeddings(fake, [ref_images], h_ref)

    span_embeds = torch.cat([aligner_out, delimiters])[image["span_source_index"]]
    h_ours = torch.zeros(len(ids), dim)
    h_ours[extra["multimodal_data"]["token_layout"]["image_mask"]] = span_embeds
    assert torch.equal(h_ours, h_ref[0])
    types_ = image["span_types"]
    assert set(types_.unique().tolist()) == {IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END}


def test_image_local_tables_assemble_any_run_of_images_alone(processor):
    """``aligner_index_local`` / ``span_source_local`` rebuild, from the images of a run alone,
    exactly the rows the request-wide ``aligner_index`` / ``span_source_index`` give for them."""
    sizes = [(544, 544), (30, 900), (2000, 100), (768, 384), (14, 14)]
    images = [_random_image(w, h, seed=i) for i, (w, h) in enumerate(sizes)]
    _, extra = _run(processor, "\n\n".join([IMAGE_PLACEHOLDER] * len(images) + ["x"]), images)
    image = extra["multimodal_data"]["image"]
    cu, ccu = image["vit_cu_seqlens"], image["aligner_cu_cells"]
    scu = [0]
    for n in image["span_lengths"]:
        scu.append(scu[-1] + n)
    n_cells = [h * w for h, w in image["llm_grid_hw"]]
    assert ccu == [sum(n_cells[:i]) for i in range(len(n_cells) + 1)]
    assert image["aligner_index_local"].shape == image["aligner_index"].shape
    assert image["span_source_local"].shape == image["span_source_index"].shape
    gen = torch.Generator().manual_seed(2)
    width, dim = 16, 8
    vit_out = torch.randn(cu[-1], width, generator=gen)
    aligner_out = torch.randn(ccu[-1], dim, generator=gen)
    delimiters = torch.randn(3, dim, generator=gen)
    cells_all = torch.cat([vit_out, vit_out.new_zeros(1, width)])[image["aligner_index"]]
    spans_all = torch.cat([aligner_out, delimiters])[image["span_source_index"]]
    for i in range(len(images)):
        vit_i = vit_out[cu[i] : cu[i + 1]]
        idx_i = image["aligner_index_local"][ccu[i] : ccu[i + 1]]
        assert 0 <= int(idx_i.min()) and int(idx_i.max()) <= cu[i + 1] - cu[i]
        cells_i = torch.cat([vit_i, vit_i.new_zeros(1, width)])[idx_i]
        assert torch.equal(cells_i, cells_all[ccu[i] : ccu[i + 1]])
        src_i = image["span_source_local"][scu[i] : scu[i + 1]]
        assert int(src_i.min()) >= 0 and int(src_i.max()) == n_cells[i] + 2
        spans_i = torch.cat([aligner_out[ccu[i] : ccu[i + 1]], delimiters])[src_i]
        assert torch.equal(spans_i, spans_all[scu[i] : scu[i + 1]])


# ---------------------------------------------------------------------------------------------
# Text-only prompts and errors


def test_text_only_prompts_tokenize_as_the_default_processor(ckpt, ref, tokenizer, processor):
    default = DefaultInputProcessor(None, None, tokenizer)
    prompts = [prompt for prompt, records in _example_cases(ckpt, ref) if not records]
    assert len(prompts) == 3
    for prompt in prompts:
        for add_special_tokens in (True, False):
            sp = SamplingParams(add_special_tokens=add_special_tokens)
            ids, extra = processor(TextPrompt(prompt=prompt), sp)
            assert extra is None
            assert ids == default(TextPrompt(prompt=prompt), sp)[0]
        # The checkpoint's tokenizer adds nothing: BOS is text in the rendered prompt.
        assert tokenizer.encode(prompt) == tokenizer.encode(prompt, add_special_tokens=False)


def test_placeholder_and_image_counts_must_agree(ref, processor):
    image = _random_image(64, 64, seed=0)
    with pytest.raises(ValueError, match="Found 2 image placeholder tokens .* but got 1 images"):
        _run(processor, _reference_prompt(ref, 2, "x"), [image])
    with pytest.raises(ValueError, match="Found 1 image placeholder tokens .* but got 0 images"):
        _run(processor, _reference_prompt(ref, 1, "x"), None)
    with pytest.raises(ValueError, match="takes images only"):
        processor(TextPrompt(prompt="x", multi_modal_data={"video": [image]}), SamplingParams())


def test_truncation_never_cuts_an_image_span(ref, processor):
    image = _random_image(544, 544, seed=0)
    prompt = _reference_prompt(ref, 1, "x")
    ids, _ = _run(processor, prompt, [image], truncate_prompt_tokens=8192)
    with pytest.raises(ValueError, match="would cut an image span"):
        _run(processor, prompt, [image], truncate_prompt_tokens=len(ids) - 1)


def test_tokens_per_image_matches_the_expansion(ckpt, hf_config, processor, ref):
    for size in [(1, 1), (544, 544), (768, 384), (50000, 14)]:
        image = _random_image(*size, seed=0)
        _, extra = _run(processor, _reference_prompt(ref, 1, "x"), [image])
        span = extra["multimodal_data"]["image"]["span_lengths"][0]
        rgb = np.asarray(image)
        for fmt in (image, torch.from_numpy(rgb).permute(2, 0, 1), rgb):
            assert processor.get_num_tokens_per_image(image=fmt) == span
    assert processor.get_mm_token_ids().tolist() == [_IMAGE_TOKEN_ID]
    assert processor.get_mm_special_token_ids() is None


# ---------------------------------------------------------------------------------------------
# Deliberately wrong controls


def test_control_resize_instead_of_pad_is_detected(ckpt, ref, ref_args, hf_config):
    """A plain resize to the target size (the max_wh_ratio branch) gives different patches."""
    geometry = VisionGeometry.from_config(hf_config)
    path = ckpt / "inference" / "examples" / "images" / "carrots.jpeg"
    reference = ref.image_processor.load_image({"url": str(path)}, ref_args)[0]
    image = Image.open(path).convert("RGB")
    assert torch.equal(preprocess_image(image, geometry).patches, reference)

    grid = plan_image_grid(image.width, image.height, geometry)
    assert image.size != (grid.best_width, grid.best_height)
    resized = image.resize((grid.best_width, grid.best_height))
    wrong = preprocess_image(resized, geometry).patches
    assert wrong.shape == reference.shape
    assert not torch.equal(wrong, reference)


def test_control_newline_separator_is_detected(ref, ref_args, tokenizer, chat_renderer):
    """Placeholders joined by "\\n" (the registry default) render a prompt encoding.py never does."""
    text = "Which one is larger?"
    for num_images in (1, 2, 3):
        expected = _reference_prompt(ref, num_images, text)
        assert _string_path_prompt(chat_renderer, num_images, text) == expected
        wrong = _string_path_prompt(chat_renderer, num_images, text, separator="\n")
        assert wrong != expected
        assert tokenizer.encode(wrong) != tokenizer.encode(expected)


# ---------------------------------------------------------------------------------------------
# MMMU, rendered as trtllm-eval renders it


def _lm_eval_mmmu_utils():
    spec = importlib.util.find_spec("lm_eval")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("lm_eval is not installed")
    path = Path(spec.submodule_search_locations[0]) / "tasks" / "mmmu" / "utils.py"
    module_spec = importlib.util.spec_from_file_location("_dsv41_lm_eval_mmmu_utils", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _mmmu_validation_docs(mmmu_dir: Path):
    """Every validation row, as the dict lm-eval's MMMU utils read (images decoded to PIL)."""
    import pyarrow as pa

    docs = []
    for arrow in sorted(mmmu_dir.glob("*/0.0.0/*/mmmu-validation.arrow")):
        with pa.memory_map(str(arrow)) as source:
            table = pa.ipc.open_stream(source).read_all()
        for row in table.to_pylist():
            for i in range(1, 8):
                value = row.get(f"image_{i}")
                if value is not None:
                    row[f"image_{i}"] = Image.open(io.BytesIO(value["bytes"]))
            docs.append(row)
    return docs


def _anchor_prompt_tokens(anchor_dir: Path):
    """doc id -> the prompt length the external engine reported for it, when recorded."""
    if not anchor_dir.is_dir():
        return {}
    by_key = {}
    with open(anchor_dir / "samples.jsonl", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            by_key[(sample["task"], str(sample["doc_id"]))] = sample["doc"]["id"]
    tokens = {}
    with open(anchor_dir / "requests.jsonl", encoding="utf-8") as f:
        for line in f:
            request = json.loads(line)
            key = (request["task"], str(request["doc_id"]))
            if key in by_key:
                tokens[by_key[key]] = request["meta"]["usage"]["prompt_tokens"]
    return tokens


def test_mmmu_documents_match_the_reference_and_the_anchor(
    ref, ref_args, tokenizer, processor, chat_renderer
):
    mmmu_dir = _require(_MMMU_DIR)
    utils = _lm_eval_mmmu_utils()
    docs = _mmmu_validation_docs(mmmu_dir)
    assert len(docs) == 900
    if not _MMMU_ALL:
        firsts, seen = [], set()
        for doc in docs:
            subject = doc["id"].rsplit("_", 1)[0]
            if subject not in seen or len(utils.doc_to_image(doc)) > 1:
                firsts.append(doc)
                seen.add(subject)
        docs = firsts
    anchor_tokens = _anchor_prompt_tokens(Path(_ANCHOR_MMMU_DIR))

    checked = anchored = multi_image = 0
    for doc in docs:
        text = utils.doc_to_text(doc)
        num_images = text.count("<image>")
        text = text.replace("<image>", "")
        images = [
            img if img.mode == "RGB" else img.convert("RGB") for img in utils.doc_to_image(doc)
        ]
        assert len(images) == num_images, doc["id"]

        prompt = _string_path_prompt(chat_renderer, num_images, text)
        assert prompt == _reference_prompt(ref, num_images, text), doc["id"]
        records = [{"data": _png_bytes(img)} for img in images]
        reference = ref.image_processor.prepare_vl_inputs(prompt, records, tokenizer, ref_args)
        ours = _run(processor, prompt, images, truncate_prompt_tokens=8192)
        _assert_matches_reference(ours, reference)

        # The external engine puts one span-marker token in front of an image's span, so for a
        # one-image prompt its length is ours plus one. Its multi-image lengths do not decompose
        # that way (between one and N extra tokens), so they are not compared.
        if num_images == 1 and doc["id"] in anchor_tokens:
            assert len(ours[0]) + 1 == anchor_tokens[doc["id"]], doc["id"]
            anchored += 1

        wrong = _string_path_prompt(chat_renderer, num_images, text, separator="\n")
        assert tokenizer.encode(wrong) != tokenizer.encode(prompt)
        checked += 1
        multi_image += num_images > 1
    assert checked >= 30 and multi_image > 0
    if anchor_tokens:
        assert anchored == checked - multi_image
