# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 pipeline tests: CPU cases are marked cpu_only; the rest run end to end on a GPU."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import PIL.Image
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from flux3_test_utils import (
    LATENT_CHANNELS,
    TEXT_DIM,
    FakeTextEncoder,
    diffusers_weights,
    model_config,
    pipeline_manifest,
    tiny_config,
    tiny_vae,
    write_checkpoint,
    write_gpu_checkpoint,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

# Importing the models package registers every pipeline.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (
    MAX_REFERENCES,
    SCHEDULER_SHIFT,
    WARMUP_TEXT_WINDOWS,
    Flux3DistilledPipeline,
    Flux3Pipeline,
    Flux3References,
    Flux3RequestError,
    UnsupportedNumImagesError,
    _manifest_subfolder,
    flux3_scheduler,
    load_reference_images,
    preprocess_references,
    set_flux3_timesteps,
)
from tensorrt_llm._torch.visual_gen.models.flux3.text_encoder import (
    HIDDEN_STATE_LAYERS,
    TEXT_MAX_TOKENS,
    Flux3TextEncoder,
    padded_text_length,
    trim_for_text,
)
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Transformer2DModel
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen import VisualGen
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CompilationConfig,
    CpuOffloadConfig,
    CudaGraphConfig,
    TorchCompileConfig,
    VisualGenArgs,
)
from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams, validate_visual_gen_params

PIPELINES = [
    pytest.param(Flux3Pipeline, id="base"),
    pytest.param(Flux3DistilledPipeline, id="klein"),
]
DOWNLOAD_PATTERNS = [
    "modular_model_index.json",
    "transformer/*",
    "vae/*",
    "text_encoder/*",
    "LICENSE.md",
    "NOTICE.md",
    "README.md",
]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


# =============================================================================
# Helpers
# =============================================================================


def png_bytes(width: int, height: int, seed: int = 0) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), np.uint8)
    buffer = io.BytesIO()
    PIL.Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def config_stub(flow_map: bool = False, **overrides) -> SimpleNamespace:
    """The DiffusionPipelineConfig fields the FLUX.3 pipeline reads, set to supported values."""
    values = dict(
        primary_pretrained_config=SimpleNamespace(flow_map=flow_map, num_attention_heads=24),
        cpu_offload_config=SimpleNamespace(enable=False),
        parallel=SimpleNamespace(parallel_vae_size=1),
        cache=None,
        attention=SimpleNamespace(sparse_attention_config=None, quant_attention_config=None),
        quant_config=SimpleNamespace(quant_algo=None),
        dynamic_weight_quant=False,
        extra_attrs={"variant": None},
        runtime_lora=None,
        visual_gen_mapping=None,
        torch_dtype=torch.bfloat16,
        compilation=SimpleNamespace(resolutions=None, num_frames=None),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeTransformer(torch.nn.Module):
    """Records each call and predicts a constant velocity per packed sample (1, 2, ...)."""

    def __init__(self, flow_map: bool):
        super().__init__()
        self.flow_map = flow_map
        self.content_channels = {"image": LATENT_CHANNELS, "image_cond": LATENT_CHANNELS}
        self.packed_channels = {**self.content_channels, "text": TEXT_DIM}
        self.calls = []

    def prepare_static_inputs(self, *, layout, **inputs):
        prepared = {"_flux3_joint_cos": torch.zeros(1), "_flux3_joint_sin": torch.zeros(1)}
        for name, lengths in layout:
            if any(lengths):
                prepared[f"_flux3_{name}_cos"] = prepared[f"_flux3_{name}_sin"] = torch.zeros(1)
                if not self.flow_map and name != "image":
                    prepared[f"_flux3_{name}_tokens"] = torch.zeros(1)
        return prepared

    def forward(self, *, layout, **inputs):
        self.calls.append({"layout": layout, **inputs})
        num_samples = len(layout[0][1])
        rows = [
            torch.full_like(chunk, i + 1)
            for i, chunk in enumerate(inputs["x_image"].chunk(num_samples))
        ]
        return {"x_image": torch.cat(rows)}


def bare_pipeline(cls=Flux3Pipeline, config=None) -> Flux3Pipeline:
    """A pipeline with no transformer, built without loading anything."""
    pipeline = cls.__new__(cls)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_config = config if config is not None else config_stub(cls.flow_map)
    pipeline._device = torch.device("cpu")
    pipeline._cuda_graph_runners = {}
    pipeline._is_warmup = True
    pipeline._parallel_vae_enabled = False
    pipeline._warmed_up_shapes = set()
    pipeline._warmup_reference_keys = set()
    pipeline.transformer = pipeline.text_encoder = pipeline.vae = None
    return pipeline


def cpu_pipeline(cls=Flux3Pipeline, config=None) -> Flux3Pipeline:
    """A CPU pipeline with a recording fake transformer and text encoder and a tiny real VAE."""
    pipeline = bare_pipeline(cls, config)
    pipeline.transformer = FakeTransformer(cls.flow_map)
    pipeline.text_encoder = FakeTextEncoder()
    pipeline.vae = tiny_vae()
    pipeline.scheduler = flux3_scheduler()
    return pipeline


def make_request(*references: bytes, height=None, width=None, num_images_per_prompt=1):
    params = SimpleNamespace(
        image_reference=[SimpleNamespace(content=r) for r in references] or None,
        height=height,
        width=width,
        num_frames=None,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=None,
        num_inference_steps=2,
        guidance_scale=None,
        seed=0,
    )
    return SimpleNamespace(prompt=["edit this image"], params=params, prepared_inputs={})


# =============================================================================
# Registry, download and defaults
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("manifest", ["model_index.json", "modular_model_index.json"])
@pytest.mark.parametrize(
    "class_name, expected",
    [
        ("Flux3ModularPipeline", "Flux3ModularPipeline"),
        ("Flux3DistilledModularPipeline", "Flux3DistilledModularPipeline"),
        ("Flux3Pipeline", "Flux3ModularPipeline"),
        ("Flux3ImageEditPipeline", "Flux3ModularPipeline"),
        ("FluxPipeline", "FluxPipeline"),
        ("FluxImg2ImgPipeline", "FluxPipeline"),
        ("Flux2Pipeline", "Flux2Pipeline"),
        ("Flux2KleinPipeline", "Flux2Pipeline"),
    ],
)
def test_registry_routes_flux_generations(tmp_path, manifest, class_name, expected):
    (tmp_path / manifest).write_text(json.dumps({"_class_name": class_name}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hf_id, class_name, pipeline_cls",
    [
        ("black-forest-labs/flux-3-image-klein-base", "Flux3ModularPipeline", Flux3Pipeline),
        (
            "black-forest-labs/flux-3-image-klein",
            "Flux3DistilledModularPipeline",
            Flux3DistilledPipeline,
        ),
    ],
)
def test_hub_download_fetches_only_diffusers_components(
    monkeypatch, tmp_path, hf_id, class_name, pipeline_cls
):
    entry = PIPELINE_REGISTRY[class_name]
    assert entry.pipeline_cls is pipeline_cls
    assert entry.hf_ids == [hf_id]
    assert hf_id in VisualGen.supported_models()
    assert VisualGen.pipeline_config(hf_id) == {"variant": None}

    captured = {}

    def download_partial(model, allow_patterns, revision=None):
        captured.update(model=model, allow_patterns=allow_patterns, revision=revision)
        return tmp_path

    def download_full(*args, **kwargs):
        raise AssertionError("FLUX.3 must not download the root BFL weights or variants")

    loader_module = "tensorrt_llm._torch.visual_gen.pipeline_loader"
    monkeypatch.setattr(f"{loader_module}.download_hf_partial", download_partial, raising=False)
    monkeypatch.setattr(f"{loader_module}.download_hf_model", download_full)
    loader = PipelineLoader(VisualGenArgs(model=hf_id), device="cpu")
    assert loader._resolve_checkpoint_dir(hf_id) == str(tmp_path)
    assert captured == {"model": hf_id, "allow_patterns": DOWNLOAD_PATTERNS, "revision": None}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cls, expected",
    [
        pytest.param(
            Flux3Pipeline,
            {"height": 1024, "width": 1024, "num_inference_steps": 50, "guidance_scale": 3.5},
            id="base",
        ),
        pytest.param(
            Flux3DistilledPipeline,
            {"height": 1024, "width": 1024, "num_inference_steps": 16},
            id="klein",
        ),
    ],
)
def test_defaults_follow_diffusers_and_declare_no_seed(cls, expected):
    pipeline = bare_pipeline(cls)
    defaults = pipeline.default_generation_params
    assert defaults == expected
    assert VisualGenParams(**defaults).seed is None
    assert pipeline.extra_param_specs == {}
    validate = dict(
        declared_defaults=defaults,
        extra_param_specs=pipeline.extra_param_specs,
        ref_slot_specs=pipeline.ref_slot_specs,
    )
    validate_visual_gen_params(VisualGenParams(seed=7), **validate)
    with pytest.raises(ValueError, match="Unknown extra_params"):
        validate_visual_gen_params(VisualGenParams(extra_params={"mode": "i2i"}), **validate)
    if cls.flow_map:
        with pytest.raises(ValueError, match="guidance_scale"):
            validate_visual_gen_params(VisualGenParams(guidance_scale=2.0), **validate)


@pytest.mark.cpu_only
def test_reference_slot_accepts_at_most_ten_images():
    pipeline = bare_pipeline()

    def params(count: int) -> VisualGenParams:
        refs = [MediaRef(content=b"image", format="bytes") for _ in range(count)]
        return VisualGenParams(image_reference=refs or None)

    for count in (0, 1, MAX_REFERENCES):
        validate_visual_gen_params(
            params(count),
            declared_defaults=None,
            extra_param_specs={},
            ref_slot_specs=pipeline.ref_slot_specs,
        )
    with pytest.raises(ValueError, match="expected 0..10, got 11"):
        validate_visual_gen_params(
            params(MAX_REFERENCES + 1),
            declared_defaults=None,
            extra_param_specs={},
            ref_slot_specs=pipeline.ref_slot_specs,
        )


class FakeCompiled(torch.nn.Module):
    """Stands in for torch.compile's wrapper and records the arguments it got."""

    def __init__(self, module, kwargs):
        super().__init__()
        self.module, self.kwargs = module, kwargs


@pytest.mark.cpu_only
def test_compile_covers_every_block_list_and_keeps_bf16_rounding(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: FakeCompiled(module, kwargs))
    config = tiny_config(flow_map=True)
    pipeline = bare_pipeline(Flux3DistilledPipeline)
    pipeline.pipeline_config.torch_compile = SimpleNamespace(enable_fullgraph=False)
    pipeline.transformer = Flux3Transformer2DModel(
        model_config(config, skip_create_weights_in_init=True)
    )
    paths = [
        "mode_blocks.image",
        "mode_blocks.image_cond",
        "txt_mode_blocks",
        "single_transformer_blocks",
    ]
    originals = {path: list(pipeline.transformer.get_submodule(path)) for path in paths}

    pipeline.torch_compile()

    assert pipeline._find_transformer_blocks(pipeline.transformer) == paths
    expected = {"dynamic": None, "fullgraph": False, "options": {"emulate_precision_casts": True}}
    for path, blocks in originals.items():
        compiled = list(pipeline.transformer.get_submodule(path))
        assert [block.module for block in compiled] == blocks, path
        assert all(block.kwargs == expected for block in compiled), path


# =============================================================================
# Construction-time rejections
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("cls", PIPELINES)
@pytest.mark.parametrize(
    "overrides, message",
    [
        pytest.param(
            dict(cpu_offload_config=SimpleNamespace(enable=True)), "CPU offloading", id="offload"
        ),
        pytest.param(
            dict(parallel=SimpleNamespace(parallel_vae_size=2)), "parallel VAE", id="parallel_vae"
        ),
        pytest.param(
            dict(cache=SimpleNamespace(cache_backend="teacache")), "teacache", id="teacache"
        ),
        pytest.param(
            dict(cache=SimpleNamespace(cache_backend="cache_dit")), "cache_dit", id="cache_dit"
        ),
        pytest.param(
            dict(
                attention=SimpleNamespace(
                    sparse_attention_config=SimpleNamespace(algorithm="sol"),
                    quant_attention_config=None,
                )
            ),
            "sparse attention",
            id="sparse_attention",
        ),
        pytest.param(
            dict(quant_config=SimpleNamespace(quant_algo="FP8")),
            "on load only with quant_config.quant_algo FP8_PER_CHANNEL_PER_TOKEN, got static FP8",
            id="quant",
        ),
        pytest.param(
            dict(
                attention=SimpleNamespace(
                    sparse_attention_config=None, quant_attention_config=SimpleNamespace()
                )
            ),
            "quantized attention",
            id="quant_attention",
        ),
        pytest.param(dict(runtime_lora=SimpleNamespace(path="lora")), "runtime LoRA", id="lora"),
        pytest.param(
            dict(visual_gen_mapping=VisualGenMapping(world_size=3, rank=0, ulysses_size=3)),
            "ulysses_size must be 1, 2, 4 or 8; got 3",
            id="ulysses3",
        ),
        pytest.param(
            dict(visual_gen_mapping=VisualGenMapping(world_size=12, rank=0, ulysses_size=12)),
            "ulysses_size must be 1, 2, 4 or 8; got 12",
            id="ulysses12",
        ),
        pytest.param(
            dict(visual_gen_mapping=VisualGenMapping(world_size=16, rank=0, ulysses_size=16)),
            "ulysses_size must be 1, 2, 4 or 8; got 16",
            id="ulysses16",
        ),
        pytest.param(
            dict(visual_gen_mapping=VisualGenMapping(world_size=2, rank=0, ring_size=2)),
            "Ring Attention or Attention2D",
            id="ring2",
        ),
        pytest.param(
            dict(
                visual_gen_mapping=VisualGenMapping(
                    world_size=4, rank=0, attn2d_row_size=2, attn2d_col_size=2
                )
            ),
            "Ring Attention or Attention2D",
            id="attn2d",
        ),
    ],
)
def test_unsupported_features_are_rejected_before_loading(cls, overrides, message):
    with pytest.raises(ValueError, match=message):
        cls(config_stub(cls.flow_map, **overrides))


@pytest.mark.cpu_only
def test_klein_rejects_cfg_parallelism_and_base_accepts_it():
    mapping = VisualGenMapping(world_size=2, rank=0, cfg_size=2)
    with pytest.raises(ValueError, match="cfg_size to 1"):
        Flux3DistilledPipeline(config_stub(True, visual_gen_mapping=mapping))
    Flux3Pipeline._check_pipeline_config(config_stub(False, visual_gen_mapping=mapping))


@pytest.mark.cpu_only
@pytest.mark.parametrize("ulysses_size", [2, 4, 8])
def test_sequence_parallel_sizes_that_divide_heads_and_text_windows_load(ulysses_size):
    mapping = VisualGenMapping(world_size=ulysses_size, rank=0, ulysses_size=ulysses_size)
    for cls in (Flux3Pipeline, Flux3DistilledPipeline):
        cls._check_pipeline_config(config_stub(cls.flow_map, visual_gen_mapping=mapping))


def load_pipeline_config(path: Path) -> DiffusionPipelineConfig:
    return DiffusionPipelineConfig.from_pretrained(
        str(path), args=VisualGenArgs(model=str(path)), device="cpu"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("class_name", [None, "Flux3FooPipeline"], ids=["released", "unregistered"])
@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "base"])
def test_auto_pipeline_picks_the_class_from_the_checkpoint(tmp_path, flow_map, class_name):
    config = tiny_config(flow_map)
    write_checkpoint(tmp_path, config, diffusers_weights(config), class_name=class_name)
    with MetaInitMode():
        pipeline = AutoPipeline.from_config(load_pipeline_config(tmp_path), str(tmp_path))
    assert type(pipeline) is (Flux3DistilledPipeline if flow_map else Flux3Pipeline)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "checkpoint_flow_map, class_name, expected",
    [
        pytest.param(
            True, "Flux3ModularPipeline", "Flux3DistilledModularPipeline", id="base_name_klein"
        ),
        pytest.param(
            False, "Flux3DistilledModularPipeline", "Flux3ModularPipeline", id="klein_name_base"
        ),
    ],
)
def test_registered_manifest_names_must_agree_with_flow_map(
    tmp_path, checkpoint_flow_map, class_name, expected
):
    config = tiny_config(checkpoint_flow_map)
    write_checkpoint(tmp_path, config, diffusers_weights(config), class_name=class_name)
    with pytest.raises(ValueError, match=f"set the manifest _class_name to {expected}"):
        with MetaInitMode():
            AutoPipeline.from_config(load_pipeline_config(tmp_path), str(tmp_path))


@pytest.mark.cpu_only
def test_real_config_fields_pass_the_check_and_offload_is_rejected(tmp_path):
    config = tiny_config(flow_map=True)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    Flux3DistilledPipeline._check_pipeline_config(load_pipeline_config(tmp_path))
    offload_config = DiffusionPipelineConfig.from_pretrained(
        str(tmp_path),
        args=VisualGenArgs(model=str(tmp_path), cpu_offload_config=CpuOffloadConfig(enable=True)),
        device="cpu",
    )
    with pytest.raises(ValueError, match="CPU offloading"):
        Flux3DistilledPipeline(offload_config)


# =============================================================================
# Request contract
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_images", [0, 2, 4])
def test_num_images_per_prompt_is_a_client_error_raised_first(num_images):
    pipeline = bare_pipeline()
    pipeline.forward = MagicMock(side_effect=AssertionError("must not generate"))
    req = make_request(png_bytes(64, 64), num_images_per_prompt=num_images)
    with pytest.raises(UnsupportedNumImagesError, match="num_images_per_prompt") as error:
        pipeline.prepare_request(req)
    assert req.prepared_inputs == {} and req.params.height is None
    with pytest.raises(UnsupportedNumImagesError):
        pipeline.infer(req)
    assert isinstance(error.value, Flux3RequestError)
    assert pipeline.classify_request_failure(error.value) == "client"
    assert pipeline.classify_request_failure(ValueError("internal")) is None


@pytest.mark.cpu_only
def test_infer_takes_one_prompt_per_request():
    pipeline = bare_pipeline()
    pipeline.forward = MagicMock(side_effect=AssertionError("must not generate"))
    req = make_request()
    req.prompt = ["a", "b"]
    with pytest.raises(Flux3RequestError, match="one prompt per request, got 2") as error:
        pipeline.infer(req)
    assert pipeline.classify_request_failure(error.value) == "client"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "references, message",
    [
        pytest.param([b"not an image"], "Unable to decode", id="undecodable"),
        pytest.param(
            [png_bytes(32, 32, seed) for seed in range(MAX_REFERENCES + 1)],
            "at most 10 reference images",
            id="too_many",
        ),
        pytest.param([png_bytes(32, 32), png_bytes(10, 40)], "at least 16 pixels", id="tiny"),
    ],
)
def test_bad_references_are_client_errors_raised_while_preparing(references, message):
    pipeline = bare_pipeline()
    req = make_request(*references)
    with pytest.raises(Flux3RequestError, match=message) as error:
        pipeline.prepare_request(req)
    assert req.prepared_inputs == {}
    assert pipeline.classify_request_failure(error.value) == "client"


@pytest.mark.cpu_only
def test_prepare_request_resolves_size_and_infer_reuses_references():
    pipeline = bare_pipeline()
    req = make_request(png_bytes(56, 40))
    pipeline.prepare_request(req)

    references = req.prepared_inputs["references"]
    # A single reference is padded to its 64 x 64 canvas, which sets the output size.
    assert (req.params.height, req.params.width) == (64, 64)
    assert references.shapes == ((64, 64),)
    assert references.crop == (4, 12, 60, 52)
    assert pipeline.request_warmup_cache_key(req) == (64, 64, 1, ((64, 64),))

    pipeline.forward = MagicMock(return_value="output")
    assert pipeline.infer(req) == "output"
    assert pipeline.forward.call_args.args == ("edit this image",)
    assert pipeline.forward.call_args.kwargs["image"] is references
    assert pipeline.forward.call_args.kwargs["height"] == 64


@pytest.mark.cpu_only
def test_explicit_size_drops_the_crop_back():
    pipeline = bare_pipeline()
    req = make_request(png_bytes(56, 40), height=128)
    pipeline.prepare_request(req)
    assert (req.params.height, req.params.width) == (128, 64)
    assert req.prepared_inputs["references"].crop is None


@pytest.mark.cpu_only
def test_warmup_keys_keep_reference_count_and_order():
    pipeline = bare_pipeline()
    text_only = make_request(height=1024, width=1024)
    pipeline.prepare_request(text_only)
    assert pipeline.request_warmup_cache_key(text_only) == (1024, 1024)

    two = make_request(png_bytes(2000, 1500), png_bytes(300, 200), height=512, width=512)
    pipeline.prepare_request(two)
    assert pipeline.request_warmup_cache_key(two) == (512, 512, 2, ((880, 1168), (192, 288)))


@pytest.mark.cpu_only
def test_output_size_defaults():
    pipeline = bare_pipeline()
    assert pipeline._prepare_references(None, None, None) == (None, 1024, 1024)
    assert pipeline._prepare_references(None, 512, None) == (None, 512, 1024)
    references, height, width = pipeline._prepare_references([png_bytes(56, 40)], None, None)
    assert (height, width, references.crop) == (64, 64, (4, 12, 60, 52))
    # Prepared references keep the crop box they were resolved with.
    prepared = Flux3References((torch.zeros(1, 3, 64, 80),), crop=(1, 2, 3, 4))
    assert pipeline._prepare_references(prepared, 64, 80) == (prepared, 64, 80)


# =============================================================================
# Reference preprocessing (Diffusers Flux3ImageProcessor / BFL rule)
# =============================================================================


def to_pil(width: int, height: int, seed: int = 0) -> PIL.Image.Image:
    return load_reference_images(png_bytes(width, height, seed))[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "size, shape, crop",
    [
        # 12 MP photo: fitted to 2341 x 1756 so its 16-px canvas stays within 4 MP.
        pytest.param((4032, 3024), (1760, 2352), (5, 2, 2346, 1758), id="photo_12mp"),
        # Rounding up to the grid overshoots 4 MP, so the scale backs off once.
        pytest.param((2047, 2049), (2048, 2048), (5, 4, 2042, 2043), id="backoff"),
        pytest.param((2048, 2048), (2048, 2048), None, id="exactly_4mp"),
        pytest.param((8000, 100), (64, 4096), (0, 6, 4096, 57), id="side_cap_and_min_side"),
        pytest.param((69, 51), (64, 80), (5, 6, 74, 57), id="small"),
    ],
)
def test_single_reference_is_fitted_and_padded_to_its_canvas(size, shape, crop):
    references = preprocess_references([to_pil(*size)])
    (image,) = references.images
    assert image.shape == (1, 3, *shape)
    assert image.dtype == torch.float32 and image.is_contiguous()
    assert shape[0] % 16 == 0 and shape[1] % 16 == 0
    assert shape[0] * shape[1] <= 4 * 1024 * 1024
    assert references.crop == crop
    if crop is not None:
        left, top, right, bottom = crop
        fitted_aspect = (right - left) / (bottom - top)
        assert fitted_aspect / (size[0] / size[1]) == pytest.approx(1.0, abs=0.01)


@pytest.mark.cpu_only
def test_small_reference_keeps_its_pixels_and_reflects_into_the_padding():
    source = to_pil(69, 51)
    references = preprocess_references([source])
    (image,) = references.images
    left, top, right, bottom = references.crop
    expected = 2.0 * torch.from_numpy(np.asarray(source, np.float32) / 255.0).permute(2, 0, 1) - 1
    torch.testing.assert_close(image[0, :, top:bottom, left:right], expected, rtol=0, atol=0)
    # Reflection mirrors around the edge pixel without repeating it.
    torch.testing.assert_close(image[0, :, top, left - 1], image[0, :, top, left + 1])


@pytest.mark.cpu_only
def test_reference_smaller_than_a_token_is_mirrored():
    references = preprocess_references([to_pil(10, 10)])
    (image,) = references.images
    assert image.shape == (1, 3, 64, 64) and references.crop == (27, 27, 37, 37)
    # Symmetric padding repeats the edge pixel.
    torch.testing.assert_close(image[0, :, 27, 26], image[0, :, 27, 27])


@pytest.mark.cpu_only
def test_multiple_references_are_fitted_under_1mp_and_center_cropped():
    references = preprocess_references([to_pil(2000, 1500), to_pil(300, 200), to_pil(1024, 1024)])
    assert references.shapes == ((880, 1168), (192, 288), (1024, 1024))
    assert references.crop is None
    assert references.num_tokens == 55 * 73 + 12 * 18 + 64 * 64


@pytest.mark.cpu_only
def test_reference_count_and_size_limits():
    preprocess_references([to_pil(32, 32, seed) for seed in range(MAX_REFERENCES)])
    with pytest.raises(ValueError, match="at most 10 reference images"):
        preprocess_references([to_pil(32, 32, seed) for seed in range(MAX_REFERENCES + 1)])
    with pytest.raises(ValueError, match="at least 16 pixels"):
        preprocess_references([to_pil(32, 32), to_pil(10, 40)])
    with pytest.raises(ValueError, match="at least one reference"):
        preprocess_references([])


@pytest.mark.cpu_only
def test_reference_decoding_applies_exif_orientation_and_drops_alpha():
    buffer = io.BytesIO()
    exif = PIL.Image.Exif()
    exif[0x0112] = 6  # Rotated 90 degrees.
    PIL.Image.new("RGB", (40, 20), (200, 10, 10)).save(buffer, format="JPEG", exif=exif)
    rgba = PIL.Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    rotated, opaque = load_reference_images([buffer.getvalue(), rgba])
    assert rotated.size == (20, 40) and rotated.mode == "RGB"
    assert opaque.mode == "RGB" and opaque.getpixel((0, 0)) == (10, 20, 30)
    with pytest.raises(ValueError, match="PIL images or encoded bytes"):
        load_reference_images(["/path/to/image.png"])
    with pytest.raises(ValueError, match="Unable to decode"):
        load_reference_images([b"not an image"])


# =============================================================================
# Scheduler and text encoder
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_steps, second_sigma, last_sigma",
    [(50, 0.99706376, 0.12390487), (16, 0.99047166, 0.31600547)],
)
def test_scheduler_matches_diffusers_flux3_recipe(num_steps, second_sigma, last_sigma):
    scheduler = flux3_scheduler()
    assert scheduler.config.shift == SCHEDULER_SHIFT == 6.93
    timesteps = set_flux3_timesteps(scheduler, num_steps, torch.device("cpu"))
    linear = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    shifted = SCHEDULER_SHIFT * linear / (1 + (SCHEDULER_SHIFT - 1) * linear)
    sigmas = scheduler.sigmas
    assert len(timesteps) == num_steps and len(sigmas) == num_steps + 1
    torch.testing.assert_close(sigmas[:-1], torch.from_numpy(shifted), rtol=1e-6, atol=0)
    assert sigmas[0].item() == pytest.approx(1.0, abs=1e-6) and sigmas[-1].item() == 0.0
    assert sigmas[1].item() == pytest.approx(second_sigma, abs=1e-6)
    assert sigmas[-2].item() == pytest.approx(last_sigma, abs=1e-6)
    torch.testing.assert_close(timesteps, sigmas[:-1] * 1000)
    assert scheduler.begin_index == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_tokens, padded",
    [(1, 80), (8, 80), (80, 80), (81, 160), (8161, 8192), (8192, 8192), (9000, 8192)],
)
def test_text_window_rounds_up_to_80_tokens_capped_at_8192(num_tokens, padded):
    assert padded_text_length(min(num_tokens, TEXT_MAX_TOKENS)) == padded


class FakeTokenizer:
    def __init__(self, num_tokens: int):
        self.num_tokens, self.calls = num_tokens, []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        length = min(self.num_tokens, kwargs["max_length"])
        if kwargs.get("padding") == "max_length":
            length = kwargs["max_length"]
        mask = torch.zeros(1, length, dtype=torch.long)
        mask[:, : min(self.num_tokens, length)] = 1
        return {"input_ids": torch.ones(1, length, dtype=torch.long), "attention_mask": mask}


class FakeProcessor:
    def __init__(self, num_tokens: int):
        self.tokenizer, self.templates = FakeTokenizer(num_tokens), []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return "<templated>"


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_tokens", [8, 81, 9000])
def test_tokenize_uses_the_chat_template_and_right_pads_to_the_window(num_tokens):
    processor = FakeProcessor(num_tokens)
    encoder = Flux3TextEncoder(model=torch.nn.Linear(1, 1), processor=processor)
    input_ids, attention_mask = encoder.tokenize("a cat")
    window = padded_text_length(min(num_tokens, TEXT_MAX_TOKENS))
    assert input_ids.shape == attention_mask.shape == (1, window)
    assert processor.templates == [
        ([{"role": "user", "content": "a cat"}], {"tokenize": False, "add_generation_prompt": True})
    ]
    measure, pad = processor.tokenizer.calls
    assert measure["truncation"] and measure["max_length"] == TEXT_MAX_TOKENS
    assert pad["padding"] == "max_length" and pad["padding_side"] == "right"
    assert pad["max_length"] == window


def tiny_qwen3_vl(num_layers: int = 6):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[0],
            num_position_embeddings=16,
        ),
    )
    return Qwen3VLForConditionalGeneration(config).eval()


@pytest.mark.cpu_only
def test_text_encoder_trim_keeps_selected_hidden_states_and_drops_vision():
    model = tiny_qwen3_vl()
    full = copy.deepcopy(model)
    layers = (2, 4)
    trim_for_text(model, max(layers))

    assert model.model.visual is None
    assert not any("visual" in name for name, _ in model.named_parameters())
    assert len(model.model.language_model.layers) == 4
    assert isinstance(model.model.language_model.norm, torch.nn.Identity)

    encoder = Flux3TextEncoder(model, processor=None, hidden_state_layers=layers)
    input_ids = torch.randint(0, 100, (1, 12), generator=torch.Generator().manual_seed(1))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 9:] = 0
    actual = encoder.encode(input_ids, attention_mask)

    # The reference runs the untrimmed model the way Diffusers calls it.
    position_ids = torch.arange(12)[None].masked_fill(attention_mask == 0, 1)
    with torch.no_grad():
        hidden_states = full(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
    expected = torch.cat([hidden_states[k] for k in layers], dim=-1)
    assert actual.shape == (1, 12, 2 * 64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    encoder.model.to(torch.bfloat16)
    assert encoder.encode(input_ids, attention_mask).dtype == torch.bfloat16


@pytest.mark.cpu_only
def test_text_encoder_trim_limits():
    model = tiny_qwen3_vl()
    trim_for_text(model, 6)
    # Keeping every layer keeps the final norm, which the last hidden state includes.
    assert not isinstance(model.model.language_model.norm, torch.nn.Identity)
    for last_layer in (0, 7):
        with pytest.raises(ValueError, match="not in 1..6"):
            trim_for_text(tiny_qwen3_vl(), last_layer)


@pytest.mark.cpu_only
def test_text_encoder_loads_in_the_requested_dtype_trimmed_and_without_vision(
    tmp_path, monkeypatch
):
    from transformers import AutoProcessor

    num_layers = max(HIDDEN_STATE_LAYERS)
    tiny_qwen3_vl(num_layers + 1).save_pretrained(str(tmp_path / "model"))
    processors = []
    monkeypatch.setattr(
        AutoProcessor, "from_pretrained", lambda path: processors.append(path) or "processor"
    )
    encoder = Flux3TextEncoder.from_pretrained(
        str(tmp_path / "model"), str(tmp_path / "processor"), dtype=torch.bfloat16, device="cpu"
    )
    model = encoder.model
    assert processors == [str(tmp_path / "processor")] and encoder.processor == "processor"
    assert model.model.visual is None and not model.training
    assert len(model.model.language_model.layers) == num_layers
    assert isinstance(model.model.language_model.norm, torch.nn.Identity)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}


@pytest.mark.cpu_only
def test_processor_subfolder_comes_from_the_manifest(tmp_path):
    def subfolder():
        return _manifest_subfolder(str(tmp_path), "tokenizer", "fallback")

    assert subfolder() == "fallback"
    path = tmp_path / "modular_model_index.json"
    # The released manifests keep the Qwen3-VL processor files under text_encoder/.
    path.write_text(json.dumps(pipeline_manifest(flow_map=True)))
    assert subfolder() == "text_encoder"
    path.write_text(json.dumps({"_class_name": "Flux3DistilledModularPipeline"}))
    assert subfolder() == "fallback"


# =============================================================================
# Generation on CPU with a fake transformer
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "guidance_scale, negative_prompt, prompts",
    [
        pytest.param(1.0, None, ["a cat"], id="cfg1"),
        pytest.param(1.0, "blurry", ["a cat"], id="cfg1_ignores_negative"),
        pytest.param(3.5, None, ["a cat", ""], id="cfg3.5"),
        pytest.param(0.5, "blurry", ["a cat", "blurry"], id="cfg0.5"),
    ],
)
def test_base_runs_the_unconditional_sample_only_with_guidance(
    guidance_scale, negative_prompt, prompts
):
    pipeline = cpu_pipeline(Flux3Pipeline)
    output = pipeline.forward(
        "a cat",
        negative_prompt=negative_prompt,
        height=64,
        width=48,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        seed=0,
    )
    assert output.image.shape == (1, 64, 48, 3) and output.image.dtype == torch.uint8
    assert pipeline.text_encoder.tokenized == prompts
    assert len(pipeline.text_encoder.encoded) == len(prompts)
    calls = pipeline.transformer.calls
    assert len(calls) == 2
    for call in calls:
        assert dict(call["layout"])["text"] == (80,) * len(prompts)
        assert "target_timesteps" not in call
        assert "x_text" not in call and "_flux3_text_cos" not in call


@pytest.mark.cpu_only
def test_guidance_combines_conditional_first_as_diffusers_does():
    pipeline = bare_pipeline()
    # Packed samples are (conditional, unconditional).
    prediction = torch.tensor([[1.0], [2.0]])
    torch.testing.assert_close(
        pipeline._guide(prediction, 2, 3.5, cfg_parallel=False), torch.tensor([[-1.5]])
    )
    torch.testing.assert_close(pipeline._guide(prediction[:1], 1, 1.0, False), prediction[:1])


@pytest.mark.cpu_only
@pytest.mark.parametrize("conditional", [True, False], ids=["cfg_rank0", "cfg_rank1"])
def test_cfg_parallel_ranks_split_the_samples_and_gather_conditional_first(
    monkeypatch, conditional
):
    mapping = SimpleNamespace(
        cfg_size=2, seq_size=1, is_cfg_conditional=conditional, cfg_group="cfg_group"
    )
    pipeline = cpu_pipeline(Flux3Pipeline, config_stub(visual_gen_mapping=mapping))
    gathered = []

    def all_gather(outputs, tensor, group=None):
        # CFG rank 0 holds the conditional prediction (3), rank 1 the unconditional (1).
        assert group == "cfg_group" and len(outputs) == 2
        gathered.append(tensor.clone())
        outputs[0].fill_(3.0)
        outputs[1].fill_(1.0)

    monkeypatch.setattr(pipeline_flux3.dist, "all_gather", all_gather)
    kwargs = dict(height=64, width=64, num_inference_steps=2, seed=0)
    pipeline.forward("a cat", negative_prompt="blurry", guidance_scale=3.5, **kwargs)
    assert pipeline.text_encoder.tokenized == ["a cat" if conditional else "blurry"]
    assert all(len(dict(call["layout"])["text"]) == 1 for call in pipeline.transformer.calls)
    # Each rank sends its own single-sample prediction, once per step.
    assert len(gathered) == 2 and all(torch.all(tensor == 1) for tensor in gathered)
    combined = pipeline._guide(torch.zeros(4, 2), 1, 3.5, cfg_parallel=True)
    torch.testing.assert_close(combined, torch.full((4, 2), 1.0 + 3.5 * (3.0 - 1.0)))

    # Without guidance both CFG ranks run the prompt and exchange nothing.
    gathered.clear()
    pipeline.text_encoder.tokenized.clear()
    pipeline.forward("a cat", negative_prompt="blurry", guidance_scale=1.0, **kwargs)
    assert pipeline.text_encoder.tokenized == ["a cat"] and gathered == []


@pytest.mark.cpu_only
def test_klein_samples_one_context_and_targets_the_next_timestep():
    pipeline = cpu_pipeline(Flux3DistilledPipeline)
    pipeline.forward("a cat", height=64, width=64, num_inference_steps=3, seed=0)
    assert pipeline.text_encoder.tokenized == ["a cat"]
    calls = pipeline.transformer.calls
    timesteps = torch.stack([call["x_image_timesteps"] for call in calls])[:, 0]
    targets = torch.stack([call["target_timesteps"] for call in calls])[:, 0]
    torch.testing.assert_close(targets[:-1], timesteps[1:])
    assert targets[-1].item() == 0.0
    assert {"x_text", "x_text_timesteps", "_flux3_text_cos"} <= calls[0].keys()
    assert calls[0]["x_text"].dtype == torch.bfloat16
    assert not any(key.endswith("_ids") for key in calls[0])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cls, prompt, kwargs, message",
    [
        pytest.param(
            Flux3DistilledPipeline,
            "a cat",
            dict(guidance_scale=2.0),
            "omit guidance_scale",
            id="klein_guidance",
        ),
        pytest.param(
            Flux3DistilledPipeline,
            "a cat",
            dict(negative_prompt="blurry"),
            "no negative prompt",
            id="klein_negative",
        ),
        pytest.param(Flux3Pipeline, ["a cat"], {}, "one prompt string", id="prompt_list"),
        pytest.param(
            Flux3Pipeline, "a cat", dict(num_inference_steps=0), "must be positive", id="steps0"
        ),
        pytest.param(Flux3Pipeline, "a cat", dict(height=72), "multiples of", id="size"),
        pytest.param(
            Flux3Pipeline, "a cat", dict(image=b"not an image"), "Unable to decode", id="image"
        ),
    ],
)
def test_request_content_errors_are_client_errors_raised_before_encoding(
    cls, prompt, kwargs, message
):
    pipeline = cpu_pipeline(cls)
    kwargs = {"height": 64, "width": 64, **kwargs}
    with pytest.raises(Flux3RequestError, match=message) as error:
        pipeline.forward(prompt, seed=0, **kwargs)
    assert pipeline.classify_request_failure(error.value) == "client"
    assert pipeline.text_encoder.tokenized == []


@pytest.mark.cpu_only
def test_klein_accepts_guidance_one_and_samples_one_context():
    pipeline = cpu_pipeline(Flux3DistilledPipeline)
    pipeline.forward(
        "a cat", height=64, width=64, num_inference_steps=1, guidance_scale=1.0, seed=0
    )
    assert pipeline.text_encoder.tokenized == ["a cat"]
    assert dict(pipeline.transformer.calls[0]["layout"])["text"] == (80,)


@pytest.mark.cpu_only
def test_edit_is_generated_on_the_reference_canvas_and_cropped_back():
    pipeline = cpu_pipeline(Flux3DistilledPipeline)
    output = pipeline.forward("a cat", num_inference_steps=1, seed=0, image=png_bytes(56, 40))
    assert output.image.shape == (1, 40, 56, 3)
    call = pipeline.transformer.calls[0]
    assert dict(call["layout"])["image_cond"] == (16,)
    assert call["x_image_cond"].dtype == torch.bfloat16


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cls, texts",
    [
        # Klein Base pairs each prompt window with the 80-token default negative prompt.
        pytest.param(Flux3Pipeline, [(80, 80), (160, 80)], id="base"),
        pytest.param(Flux3DistilledPipeline, [(80,), (160,)], id="klein"),
    ],
)
def test_run_warmup_covers_two_text_windows_for_text_to_image_and_a_matching_edit(cls, texts):
    assert WARMUP_TEXT_WINDOWS == (80, 160)

    def run_warmup():
        pipeline = cpu_pipeline(cls)
        pipeline._run_warmup(64, 80, 1, 2)
        return pipeline

    pipeline = run_warmup()
    num_samples = len(texts[0])
    covered = [
        (dict(call["layout"])["image_cond"], dict(call["layout"])["text"])
        for call in pipeline.transformer.calls
    ]
    expected = []
    for image_cond in (0, 20):
        for text in texts:
            expected += [((image_cond,) * num_samples, text)] * 2
    assert covered == expected
    assert pipeline._warmup_reference_keys == {(64, 80, 1, ((64, 80),))}

    # Warmup is deterministic: a second run feeds the transformer the same latents.
    repeat = run_warmup()
    for call, repeated in zip(pipeline.transformer.calls, repeat.transformer.calls):
        torch.testing.assert_close(repeated["x_image"], call["x_image"], rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "seq_size, kwargs, message",
    [
        pytest.param(2, dict(height=1008, width=1008), "3969 target tokens", id="target_sp2"),
        pytest.param(4, dict(height=880, width=1184), "4070 target tokens", id="target_sp4"),
        pytest.param(
            2,
            dict(height=1024, width=1024, image=png_bytes(80, 80)),
            "25 reference tokens",
            id="reference_sp2",
        ),
    ],
)
def test_sequence_parallel_token_counts_are_checked_before_encoding(seq_size, kwargs, message):
    mapping = SimpleNamespace(seq_size=seq_size, cfg_size=1)
    pipeline = cpu_pipeline(Flux3Pipeline, config_stub(visual_gen_mapping=mapping))
    pipeline.vae.encode = MagicMock(side_effect=AssertionError("must not encode"))
    with pytest.raises(Flux3RequestError, match=message) as error:
        pipeline.forward("a cat", seed=0, **kwargs)
    assert f"across {seq_size} ranks" in str(error.value)
    assert pipeline.classify_request_failure(error.value) == "client"
    assert pipeline.text_encoder.tokenized == [] and pipeline.text_encoder.encoded == []


@pytest.mark.cpu_only
def test_sequence_parallel_accepts_even_targets_and_checks_total_reference_tokens():
    mapping = SimpleNamespace(seq_size=2, cfg_size=1)
    pipeline = cpu_pipeline(Flux3Pipeline, config_stub(visual_gen_mapping=mapping))
    for seq_size in (2, 4, 8):
        pipeline.pipeline_config.visual_gen_mapping = SimpleNamespace(seq_size=seq_size)
        pipeline.validate_resolution(1024, 1024, 1)
    pipeline.pipeline_config.visual_gen_mapping = mapping
    # Two 80 x 80 references have 25 tokens each: odd per reference, 50 in total.
    references = [png_bytes(80, 80), png_bytes(80, 80, seed=1)]
    pipeline.forward("a cat", height=64, width=64, num_inference_steps=1, seed=0, image=references)
    assert dict(pipeline.transformer.calls[0]["layout"])["image_cond"] == (50, 50)


@pytest.mark.cpu_only
def test_warmup_plan_skips_sizes_the_sequence_parallel_size_cannot_split():
    config = config_stub(
        visual_gen_mapping=SimpleNamespace(seq_size=2),
        compilation=SimpleNamespace(resolutions=[(1008, 1008), (1024, 1024)], num_frames=None),
    )
    shapes, steps = bare_pipeline(Flux3Pipeline, config).resolve_warmup_plan()
    assert shapes == [(1024, 1024, 1)] and steps == 2


# =============================================================================
# GPU: real tiny transformer and VAE through the pipeline
# =============================================================================


def build_gpu_pipeline(tmp_path: Path, flow_map: bool, **args) -> Flux3Pipeline:
    """Load a synthetic checkpoint through the pipeline; only the text encoder is faked."""
    weights = write_gpu_checkpoint(tmp_path, flow_map)
    visual_gen_args = VisualGenArgs(
        model=str(tmp_path),
        attention_config=AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        **args,
    )
    pipeline_config = DiffusionPipelineConfig.from_pretrained(
        str(tmp_path), args=visual_gen_args, device="cuda"
    )
    cls = Flux3DistilledPipeline if flow_map else Flux3Pipeline
    pipeline = cls(pipeline_config)
    pipeline.transformer.to("cuda")
    pipeline.load_weights({"transformer": {k: v.cuda() for k, v in weights.items()}})
    pipeline.post_load_weights()
    pipeline.load_standard_components(
        str(tmp_path), torch.device("cuda"), skip_components=["text_encoder"]
    )
    pipeline.text_encoder = FakeTextEncoder(device="cuda")
    return pipeline


def record_transformer_calls(pipeline: Flux3Pipeline) -> list:
    calls, forward = [], pipeline.transformer.forward

    def recording_forward(**kwargs):
        calls.append(kwargs)
        return forward(**kwargs)

    pipeline.transformer.forward = recording_forward
    return calls


FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_text_to_image_and_edits_run_end_to_end(tmp_path, flow_map):
    pipeline = build_gpu_pipeline(tmp_path, flow_map)
    assert isinstance(pipeline.vae, AutoencoderKLFlux2) and pipeline.vae.dtype == torch.float32

    def generate(**kwargs):
        return pipeline.forward("a red fox", num_inference_steps=2, seed=0, **kwargs).image

    image = generate(height=64, width=48)
    assert image.shape == (1, 64, 48, 3) and image.dtype == torch.uint8
    torch.testing.assert_close(generate(height=64, width=48), image, rtol=0, atol=0)
    assert not torch.equal(
        pipeline.forward("a red fox", height=64, width=48, num_inference_steps=2, seed=1).image,
        image,
    )

    reference, second = png_bytes(56, 40), png_bytes(64, 64, seed=1)
    assert generate(image=reference).shape == (1, 40, 56, 3)
    assert generate(image=reference, height=64, width=64).shape == (1, 64, 64, 3)
    assert generate(image=[reference, second], height=64, width=48).shape == (1, 64, 48, 3)


@requires_cuda
@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "i2i"])
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_transformer_receives_only_the_tensors_it_reads(tmp_path, flow_map, edit):
    pipeline = build_gpu_pipeline(tmp_path, flow_map)
    calls = record_transformer_calls(pipeline)
    image = png_bytes(64, 64) if edit else None
    pipeline.forward("a red fox", height=64, width=64, num_inference_steps=2, seed=0, image=image)

    modalities = ("image", "image_cond", "text") if edit else ("image", "text")
    if flow_map:
        expected = {"layout", "target_timesteps", "_flux3_joint_cos", "_flux3_joint_sin"}
        for name in modalities:
            expected |= {
                f"x_{name}",
                f"x_{name}_timesteps",
                f"_flux3_{name}_cos",
                f"_flux3_{name}_sin",
            }
    else:
        expected = {
            "layout",
            "x_image",
            "x_image_timesteps",
            "_flux3_image_cos",
            "_flux3_image_sin",
        }
        expected |= {"_flux3_joint_cos", "_flux3_joint_sin"}
        for name in modalities[1:]:
            expected |= {
                f"_flux3_{name}_{field}"
                for field in ("tokens", "single_shift", "single_scale", "single_gate")
            }
    assert len(calls) == 2
    for call in calls:
        assert set(call) == expected
        tensors = [value for value in call.values() if isinstance(value, torch.Tensor)]
        assert all(value.is_cuda for value in tensors)
    if flow_map:
        assert calls[0]["x_text"].dtype == torch.bfloat16


@requires_cuda
def test_base_without_guidance_runs_one_sample(tmp_path):
    pipeline = build_gpu_pipeline(tmp_path, flow_map=False)
    calls = record_transformer_calls(pipeline)
    for guidance_scale, num_samples in ((1.0, 1), (3.5, 2)):
        calls.clear()
        pipeline.forward(
            "a red fox",
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=guidance_scale,
            seed=0,
        )
        assert len(dict(calls[0]["layout"])["text"]) == num_samples


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_warmup_registers_text_to_image_and_edit_keys(tmp_path, flow_map):
    pipeline = build_gpu_pipeline(
        tmp_path, flow_map, compilation_config=CompilationConfig(resolutions=[(64, 80)])
    )
    calls = record_transformer_calls(pipeline)
    pipeline.warmup()
    assert pipeline._warmed_up_shapes == {(64, 80), (64, 80, 1, ((64, 80),))}
    assert len(calls) == 2 * len(WARMUP_TEXT_WINDOWS) * pipeline.default_warmup_steps

    # A request whose reference matches the warmed edit reuses its key.
    req = make_request(png_bytes(80, 64))
    pipeline.prepare_request(req)
    assert pipeline.request_warmup_cache_key(req) in pipeline._warmed_up_shapes
    output = pipeline.infer(req)
    assert output.image.shape == (1, 64, 80, 3)


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_pipeline_loader_compiles_warms_up_and_generates(tmp_path, monkeypatch, flow_map):
    # Two dual-stream layers, so every block list is long enough to be compiled per block.
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2)
    loaded = []

    def from_pretrained(cls, model_path, processor_path, dtype, device):
        loaded.append((model_path, processor_path, dtype, device))
        return FakeTextEncoder(device="cuda")

    monkeypatch.setattr(Flux3TextEncoder, "from_pretrained", classmethod(from_pretrained))
    args = VisualGenArgs(
        model=str(tmp_path),
        attention_config=AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=True, enable_autotune=False),
        compilation_config=CompilationConfig(resolutions=[(64, 80)]),
    )
    pipeline = PipelineLoader(args).load()

    assert type(pipeline) is (Flux3DistilledPipeline if flow_map else Flux3Pipeline)
    text_encoder_dir = str(tmp_path / "text_encoder")
    assert loaded == [(text_encoder_dir, text_encoder_dir, torch.bfloat16, torch.device("cuda"))]
    assert pipeline.vae.dtype == torch.float32
    transformer = pipeline.transformer
    torch.testing.assert_close(
        transformer.context_embedder.weight, weights["context_embedder.weight"].cuda()
    )
    paths = pipeline._find_transformer_blocks(transformer)
    assert len(paths) == 4
    blocks = [block for path in paths for block in transformer.get_submodule(path)]
    assert len(blocks) == 3 * 2 + 2
    assert all(isinstance(block, torch._dynamo.eval_frame.OptimizedModule) for block in blocks)
    assert pipeline._warmed_up_shapes == {(64, 80), (64, 80, 1, ((64, 80),))}

    output = pipeline.forward("a red fox", height=64, width=80, num_inference_steps=2, seed=0)
    assert output.image.shape == (1, 64, 80, 3) and output.image.dtype == torch.uint8


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_cuda_graphs_match_eager_and_warmed_keys_replay(tmp_path, flow_map):
    eager = build_gpu_pipeline(tmp_path, flow_map)
    graphed = build_gpu_pipeline(
        tmp_path,
        flow_map,
        cuda_graph_config=CudaGraphConfig(enable=True),
        compilation_config=CompilationConfig(resolutions=[(64, 64)]),
    )
    runner = graphed._cuda_graph_runners["transformer"]
    try:
        graphed.warmup()
        # One graph per task and warmup text window.
        assert len(runner.graphs) == 2 * len(WARMUP_TEXT_WINDOWS)
        for image in (None, png_bytes(64, 64)):
            kwargs = dict(height=64, width=64, num_inference_steps=2, seed=0, image=image)
            expected = eager.forward("a red fox", **kwargs).image
            actual = graphed.forward("a red fox", **kwargs).image
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(runner.graphs) == 2 * len(WARMUP_TEXT_WINDOWS)
    finally:
        runner.clear()


@requires_cuda
def test_fp32_vae_attention_runs_on_memory_efficient_sdpa(tmp_path, monkeypatch):
    """The FP32 VAE's attention needs no SDPA math fallback when encoding or decoding."""
    pipeline = build_gpu_pipeline(tmp_path, flow_map=True)
    assert pipeline.vae.dtype == torch.float32
    # The released VAE's head size, which memory-efficient SDPA must support.
    pipeline.vae = tiny_vae(mid_channels=512).to("cuda")
    sdpa, vae_calls = torch.nn.functional.scaled_dot_product_attention, []

    def efficient_only_for_fp32(query, *args, **kwargs):
        if query.dtype != torch.float32:
            return sdpa(query, *args, **kwargs)
        vae_calls.append(tuple(query.shape))
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return sdpa(query, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", efficient_only_for_fp32
    )
    output = pipeline.forward(
        "a red fox", height=64, width=64, num_inference_steps=2, seed=0, image=png_bytes(64, 64)
    )
    # One mid-block attention each in the encoder (reference) and the decoder (output).
    assert vae_calls == [(1, 1, 64, 512)] * 2
    assert output.image.shape == (1, 64, 64, 3)
