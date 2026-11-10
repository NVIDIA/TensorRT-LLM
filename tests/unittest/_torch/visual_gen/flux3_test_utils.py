# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Diffusers-format FLUX.3 checkpoints and BFL variants for unit tests."""

import json
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, NVFP4LinearMethod
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import Flux3Pipeline
from tensorrt_llm._torch.visual_gen.models.flux3.text_encoder import HIDDEN_STATE_LAYERS
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    bfl_to_diffusers_module,
    diffusers_to_bfl_key,
    diffusers_to_bfl_module,
    diffusers_to_bfl_state_dict,
)
from tensorrt_llm._torch.visual_gen.models.flux3.variants import (
    NVFP4_GROUP_SIZE,
    VARIANT_QUANT_ALGOS,
    VARIANT_SCALES,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, TorchCompileConfig, VisualGenArgs

# Packed latent channels of the tiny VAE: 4 latent channels in 2 x 2 patches.
LATENT_CHANNELS = 16
TEXT_DIM = 24
# Linears of every BFL block.
BFL_BLOCK_LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")

# transformer/config.json of the released klein checkpoint; klein-base omits flow_map.
KLEIN_CONFIG = {
    "_class_name": "Flux3Transformer2DModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "cond_in_channels": 128,
    "eps": 1e-06,
    "flow_map": True,
    "in_channels": 128,
    "joint_attention_dim": 20480,
    "mlp_ratio": 3.0,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 28,
    "rope_theta": 10000,
    "timestep_channels": 256,
}
KLEIN_BASE_CONFIG = {key: value for key, value in KLEIN_CONFIG.items() if key != "flow_map"}
# Diffusers names of the modules keyed by modality and their TRT-LLM module names.
MODULE_PREFIXES = (
    ("mode_blocks.image", "image_mode_blocks"),
    ("mode_blocks.image_cond", "image_cond_mode_blocks"),
    ("x_embedders.image", "image_embedder"),
    ("x_embedders.image_cond", "image_cond_embedder"),
    *(
        (f"{stage}_stream_modulations.{mode}", f"{mode}_{stage}_modulation")
        for stage in ("early", "single")
        for mode in ("image", "image_cond", "txt")
    ),
)


def tiny_config(flow_map: bool, **overrides: Any) -> dict[str, Any]:
    """A small config with distinct widths, so swapped embedders or projections fail to load."""
    config = {
        "_class_name": "Flux3Transformer2DModel",
        "attention_head_dim": 16,
        "axes_dims_rope": [4, 4, 4, 4],
        "cond_in_channels": 12,
        "eps": 1e-06,
        "in_channels": 8,
        "joint_attention_dim": 24,
        "mlp_ratio": 3.0,
        "num_attention_heads": 2,
        "num_layers": 2,
        "num_single_layers": 3,
        "rope_theta": 10000,
        "timestep_channels": 40,
        **overrides,
    }
    if flow_map:
        config["flow_map"] = True
    return config


def block_names(config: dict[str, Any]) -> list[str]:
    """Diffusers names of every transformer block."""
    return [
        *(
            f"mode_blocks.{mode}.{i}"
            for mode in ("image", "image_cond")
            for i in range(config["num_layers"])
        ),
        *(f"txt_mode_blocks.{i}" for i in range(config["num_layers"])),
        *(f"single_transformer_blocks.{i}" for i in range(config["num_single_layers"])),
    ]


def _swap_prefix(name: str, old: str, new: str) -> Optional[str]:
    if name == old or name.startswith(old + "."):
        return new + name[len(old) :]
    return None


def to_module_name(name: str) -> str:
    """TRT-LLM module or tensor name of a Diffusers FLUX.3 name."""
    for diffusers_prefix, module_prefix in MODULE_PREFIXES:
        renamed = _swap_prefix(name, diffusers_prefix, module_prefix)
        if renamed is not None:
            return renamed
    return name


def to_diffusers_name(name: str) -> str:
    """Diffusers name of a TRT-LLM FLUX.3 module or tensor name."""
    for diffusers_prefix, module_prefix in MODULE_PREFIXES:
        renamed = _swap_prefix(name, module_prefix, diffusers_prefix)
        if renamed is not None:
            return renamed
    return name


def diffusers_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Tensor names and shapes of a BF16 Diffusers FLUX.3 transformer checkpoint."""
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    mlp_hidden = int(hidden * config["mlp_ratio"])
    head_dim = config["attention_head_dim"]
    timestep_channels = config["timestep_channels"]
    shapes = {
        "x_embedders.image.weight": (hidden, config["in_channels"]),
        "x_embedders.image_cond.weight": (hidden, config["cond_in_channels"]),
        "context_embedder.weight": (hidden, config["joint_attention_dim"]),
        "timestep_embedder.linear_1.weight": (hidden, timestep_channels),
        "timestep_embedder.linear_2.weight": (hidden, hidden),
        "norm_out.linear.weight": (2 * hidden, hidden),
        "proj_out.weight": (config["in_channels"], hidden),
    }
    if config.get("flow_map", False):
        shapes["target_timestep_embedder.linear_1.weight"] = (hidden, timestep_channels)
        shapes["target_timestep_embedder.linear_2.weight"] = (hidden, hidden)
    for stage in ("early", "single"):
        for mode in ("image", "image_cond", "txt"):
            shapes[f"{stage}_stream_modulations.{mode}.linear.weight"] = (3 * hidden, hidden)
    for block in block_names(config):
        shapes.update(
            {
                f"{block}.attn.to_q.weight": (hidden, hidden),
                f"{block}.attn.to_k.weight": (hidden, hidden),
                f"{block}.attn.to_v.weight": (hidden, hidden),
                f"{block}.attn.to_out.0.weight": (hidden, hidden),
                f"{block}.attn.norm_q.weight": (head_dim,),
                f"{block}.attn.norm_k.weight": (head_dim,),
                f"{block}.ff.linear_in.weight": (2 * mlp_hidden, hidden),
                f"{block}.ff.linear_out.weight": (hidden, mlp_hidden),
            }
        )
    return shapes


def diffusers_weights(
    config: dict[str, Any], seed: int = 0, unit_scale: bool = False
) -> dict[str, torch.Tensor]:
    """Random BF16 weights; every QK-norm weight is distinct and away from one.

    ``unit_scale`` scales projections by 1 / sqrt(fan_in), so every path changes the output.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for name, shape in diffusers_shapes(config).items():
        value = torch.randn(shape, generator=generator)
        if name.endswith((".norm_q.weight", ".norm_k.weight")):
            value = 1.0 + 0.1 * value
        else:
            value = value * (shape[1] ** -0.5 if unit_scale else 0.02)
        weights[name] = value.to(torch.bfloat16)
    return weights


def model_config(
    config: dict[str, Any],
    quant_config: Optional[QuantConfig] = None,
    backend: str = "VANILLA",
    attention: Optional[AttentionConfig] = None,
    **kwargs: Any,
) -> DiffusionModelConfig:
    """DiffusionModelConfig whose pretrained_config is ``config``, as the loader builds it.

    ``attention`` replaces the default AttentionConfig(backend=backend).
    """
    attention = attention if attention is not None else AttentionConfig(backend=backend)
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**config),
        quant_config=quant_config if quant_config is not None else QuantConfig(),
        attention=attention,
        attention_metadata_state=(
            create_attention_metadata_state() if attention.backend == "TRTLLM" else None
        ),
        **kwargs,
    )


def pipeline_manifest(flow_map: bool, class_name: Optional[str] = None) -> dict[str, Any]:
    """modular_model_index.json of the released checkpoints, without the prompt upsampler."""
    repo = "black-forest-labs/flux-3-image-klein" + ("" if flow_map else "-base")

    def component(library: str, name: str, subfolder: str) -> list:
        spec = {
            "pretrained_model_name_or_path": repo,
            "revision": None,
            "subfolder": subfolder,
            "type_hint": [library, name],
            "variant": None,
        }
        return [library, name, spec]

    default_name = "Flux3DistilledModularPipeline" if flow_map else "Flux3ModularPipeline"
    return {
        "_blocks_class_name": "Flux3DistilledAutoBlocks" if flow_map else "Flux3AutoBlocks",
        "_class_name": class_name or default_name,
        "text_encoder": component(
            "transformers", "Qwen3VLForConditionalGeneration", "text_encoder"
        ),
        "tokenizer": component("transformers", "Qwen3VLProcessor", "text_encoder"),
        "transformer": component("diffusers", "Flux3Transformer2DModel", "transformer"),
        "vae": component("diffusers", "AutoencoderKLFlux2", "vae"),
    }


def write_checkpoint(
    root: Path,
    config: dict[str, Any],
    weights: dict[str, torch.Tensor],
    num_shards: int = 2,
    class_name: Optional[str] = None,
) -> None:
    """Write a modular Diffusers checkpoint whose transformer is sharded like the released one."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    index = pipeline_manifest(config.get("flow_map", False), class_name)
    (root / "modular_model_index.json").write_text(json.dumps(index))
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(json.dumps(config))
    names, weight_map = sorted(weights), {}
    for shard in range(num_shards):
        file_name = f"diffusion_pytorch_model-{shard + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_names = names[shard::num_shards]
        tensors = {name: weights[name].contiguous() for name in shard_names}
        save_file(tensors, str(transformer / file_name))
        weight_map.update(dict.fromkeys(shard_names, file_name))
    index = {"metadata": {}, "weight_map": weight_map}
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))


def assert_parity(
    actual: torch.Tensor, expected: torch.Tensor, relative_l2: float, cosine: float
) -> None:
    """Relative L2 error and cosine similarity bounds, reported together on failure."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    actual, expected = actual.float().flatten(), expected.float().flatten()
    stats = {
        "relative_l2": ((actual - expected).norm() / expected.norm()).item(),
        "cosine": F.cosine_similarity(actual, expected, dim=0).item(),
        "max_abs": (actual - expected).abs().max().item(),
    }
    assert stats["relative_l2"] <= relative_l2 and stats["cosine"] >= cosine, stats


# =============================================================================
# Tiny components for GPU pipelines
# =============================================================================


def tiny_vae(mid_channels: int = 8) -> AutoencoderKLFlux2:
    """A 4-level (8x) Diffusers FLUX.2/3 VAE whose packed latents have LATENT_CHANNELS.

    Its mid-block attention is one head of ``mid_channels``; the released VAE uses 512.
    """
    torch.manual_seed(0)
    return AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, mid_channels),
        layers_per_block=1,
        latent_channels=LATENT_CHANNELS // 4,
        norm_num_groups=4,
    ).eval()


class _FakeProcessor:
    """Chat template and tokenizer of FakeTextEncoder; records each templated prompt."""

    def __init__(self, tokenized: list):
        self.tokenized = tokenized

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.tokenized.append(messages[0]["content"])
        return messages[0]["content"]

    def tokenizer(self, text, return_tensors, truncation, max_length, padding=None, **kwargs):
        length = max_length if padding == "max_length" else min(len(text.split()) + 3, max_length)
        ids = torch.full((1, length), zlib.crc32(text.encode()) % 1000)
        return {"input_ids": ids, "attention_mask": torch.ones(1, length, dtype=torch.long)}


class _FakeQwen3VLModel(torch.nn.Module):
    """Qwen3-VL layer layout whose selected hidden states concatenate to a seeded embedding."""

    def __init__(self, dim: int, encoded: list):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(max(HIDDEN_STATE_LAYERS) + 1)
        )
        self.language_model.norm = torch.nn.Identity()
        self.visual = torch.nn.Identity()
        self.dim, self.encoded = dim, encoded

    def forward(self, input_ids, **kwargs):
        self.encoded.append(int(input_ids[0, 0]))
        generator = torch.Generator().manual_seed(int(input_ids[0, 0]))
        embedding = torch.randn(1, input_ids.shape[1], self.dim, generator=generator)
        chunks = embedding.to(input_ids.device).chunk(len(HIDDEN_STATE_LAYERS), dim=-1)
        # Embeddings, then one entry per layer, as Hugging Face returns them.
        hidden_states = [None] * (len(self.language_model.layers) + 1)
        for layer, chunk in zip(HIDDEN_STATE_LAYERS, chunks):
            hidden_states[layer] = chunk
        return SimpleNamespace(hidden_states=hidden_states)


class FakeTextEncoder(torch.nn.Module):
    """Stand-in Qwen3-VL with its ``processor``: deterministic per-prompt embeddings, where a
    prompt has 3 template tokens plus one per word. Records prompts and encoded token ids.
    """

    def __init__(self, dim: int = TEXT_DIM, device: str = "cpu"):
        super().__init__()
        self.tokenized, self.encoded = [], []
        self.processor = _FakeProcessor(self.tokenized)
        self.model = _FakeQwen3VLModel(dim, self.encoded)
        self.register_buffer("anchor", torch.zeros((), device=device))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype


def use_fake_text_encoder(pipeline: Flux3Pipeline, device: str = "cpu") -> FakeTextEncoder:
    """Give ``pipeline`` a FakeTextEncoder and its processor."""
    pipeline.text_encoder = FakeTextEncoder(device=device)
    pipeline.processor = pipeline.text_encoder.processor
    return pipeline.text_encoder


def patch_text_encoder_load(monkeypatch: pytest.MonkeyPatch, loaded: Optional[list] = None) -> None:
    """Make the pipeline load a FakeTextEncoder and its processor; ``loaded`` records the calls."""
    loaded = [] if loaded is None else loaded
    encoders = []

    class FakeQwen3VL:
        @staticmethod
        def from_pretrained(path, dtype):
            loaded.append(("model", path, dtype))
            encoders.append(FakeTextEncoder())
            return encoders[-1]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path):
            loaded.append(("processor", path))
            return encoders[-1].processor

    monkeypatch.setattr(pipeline_flux3, "Qwen3VLForConditionalGeneration", FakeQwen3VL)
    monkeypatch.setattr(pipeline_flux3, "AutoProcessor", FakeAutoProcessor)


def gpu_transformer_config(flow_map: bool, num_layers: int = 1, **overrides: Any) -> dict:
    """A tiny transformer config that fits the tiny VAE and FakeTextEncoder."""
    return tiny_config(
        flow_map,
        **{
            "attention_head_dim": 32,
            "axes_dims_rope": [8, 8, 8, 8],
            "in_channels": LATENT_CHANNELS,
            "cond_in_channels": LATENT_CHANNELS,
            "joint_attention_dim": TEXT_DIM,
            "num_layers": num_layers,
            "num_single_layers": 2,
            **overrides,
        },
    )


def write_gpu_checkpoint(
    path: Path, flow_map: bool, num_layers: int = 1, **overrides: Any
) -> dict[str, torch.Tensor]:
    """Write a synthetic checkpoint with a tiny VAE unless it exists; returns its weights."""
    config = gpu_transformer_config(flow_map, num_layers, **overrides)
    weights = diffusers_weights(config, unit_scale=True)
    if not (path / "transformer").exists():
        write_checkpoint(path, config, weights)
        tiny_vae().save_pretrained(str(path / "vae"))
    return weights


def load_gpu_pipeline(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attention_config: Optional[AttentionConfig] = None,
    **args,
) -> Flux3Pipeline:
    """Load through PipelineLoader with only the text encoder faked; quantize flags restored."""
    for method in (NVFP4LinearMethod, FP8RowwiseLinearMethod):
        monkeypatch.setattr(method, "use_tunable_quantize", method.use_tunable_quantize)

    patch_text_encoder_load(monkeypatch)
    visual_gen_args = VisualGenArgs(
        model=str(path),
        attention_config=attention_config or AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        **args,
    )
    return PipelineLoader(visual_gen_args).load(skip_warmup=True)


# =============================================================================
# Runtime LoRA adapters
# =============================================================================


def lora_adapter(
    weights: dict[str, torch.Tensor], rank: int = 4, seed: int = 1, std: float = 0.1
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Random BF16 (A [rank, in], B [out, rank]) for every Linear module of Diffusers weights."""
    generator = torch.Generator().manual_seed(seed)
    adapter = {}
    for name, value in sorted(weights.items()):
        if value.dim() == 2:
            out_features, in_features = value.shape
            a = torch.randn(rank, in_features, generator=generator) * std
            b = torch.randn(out_features, rank, generator=generator) * std
            adapter[name[: -len(".weight")]] = (a.to(torch.bfloat16), b.to(torch.bfloat16))
    return adapter


def write_lora(
    path: Path,
    adapter: dict[str, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    bfl: bool = False,
) -> Path:
    """Write a Diffusers/PEFT adapter directory, or a BFL-native one as ComfyUI exports it.

    BFL files use ``diffusion_model.`` names, lora_down/up, per-module alpha, shift-first norm_out.
    """
    from safetensors.torch import save_file

    path.mkdir(parents=True)
    tensors = {}
    for module, (a, b) in adapter.items():
        if not bfl:
            tensors[f"transformer.{module}.lora_A.weight"] = a
            tensors[f"transformer.{module}.lora_B.weight"] = b
            continue
        name = "diffusion_model." + diffusers_to_bfl_module(module)
        if module == "norm_out.linear":
            scale, shift = b.chunk(2)
            b = torch.cat([shift, scale])
        tensors[f"{name}.lora_down.weight"] = a
        tensors[f"{name}.lora_up.weight"] = b
        tensors[f"{name}.alpha"] = torch.tensor(alpha)
    if not bfl:
        rank = next(iter(adapter.values()))[0].shape[0]
        (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": alpha, "r": rank}))
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(path / "adapter_model.safetensors"),
    )
    return path


def merged_weights(
    weights: dict[str, torch.Tensor],
    adapter: dict[str, tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> dict[str, torch.Tensor]:
    """FP32 Diffusers weights with every LoRA delta ``scale * B @ A`` added."""
    merged = {name: value.float() for name, value in weights.items()}
    for module, (a, b) in adapter.items():
        merged[f"{module}.weight"] += scale * (b.float() @ a.float())
    return merged


# =============================================================================
# BFL pre-quantized variants
# =============================================================================


def bfl_variant_excludes(config: dict[str, Any], variant: str) -> list[str]:
    """exclude_modules of BFL's released variants at any depth: the embedders, the output layer
    and the text mode blocks, plus, for nvfp4, the modulations, the first image mode block and the
    last seventh of the single blocks.
    """
    excluded = [
        "txt_in",
        "emb_in.image",
        "emb_in.image_cond",
        "time_in.in_layer",
        "time_in.out_layer",
        "final_layer.image.adaLN_modulation.1",
        "final_layer.image.linear",
    ]
    if config.get("flow_map", False):
        excluded += ["ref_time_in.in_layer", "ref_time_in.out_layer"]
    blocks = [f"txt_mode_blocks.{i}" for i in range(config["num_layers"])]
    if variant == "nvfp4":
        excluded += [
            f"{stage}_stream_modulations.{mode}.lin"
            for stage in ("early", "single")
            for mode in ("image", "image_cond", "txt")
        ]
        num_single = config["num_single_layers"]
        first_excluded = num_single - max(1, num_single // 7)
        blocks += [
            "content_mode_blocks.image.0",
            *(f"single_blocks.{i}" for i in range(first_excluded, num_single)),
        ]
    excluded += [f"{block}.{linear}" for block in blocks for linear in BFL_BLOCK_LINEARS]
    return sorted(excluded)


def bfl_variant_tensor_names(config: dict[str, Any], variant: Optional[str] = None) -> list[str]:
    """Tensor names of a BFL export: BF16 for no variant, else with each quantized Linear's scales."""
    names = [diffusers_to_bfl_key(name) for name in diffusers_shapes(config)]
    if variant is not None:
        excluded = set(bfl_variant_excludes(config, variant))
        modules = [name[: -len(".weight")] for name in names if name.endswith(".weight")]
        names += [
            f"{module}.{suffix}"
            for module in modules
            if module not in excluded
            for suffix in VARIANT_SCALES[variant]
        ]
    return sorted(names)


def bfl_hf_quant_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    """hf_quant_config.json of a BFL variant."""
    quantization = {
        "quant_algo": VARIANT_QUANT_ALGOS[variant],
        "exclude_modules": bfl_variant_excludes(config, variant),
    }
    if variant == "nvfp4":
        quantization["group_size"] = NVFP4_GROUP_SIZE
    return {"producer": {"name": "bfl"}, "quantization": quantization}


def bfl_variant_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    """The architecture fields of a BFL variant's config.json."""
    dit = {
        "hidden": config["num_attention_heads"] * config["attention_head_dim"],
        "heads": config["num_attention_heads"],
        "depth": config["num_layers"],
        "depth_single": config["num_single_layers"],
    }
    # The released klein-base variants omit flow_map.
    if config.get("flow_map", False):
        dit["flow_map"] = True
    return {"format_version": 1, "quantization": variant, "dit": dit}


def write_variant(
    root: Path,
    config: dict[str, Any],
    variant: str,
    tensors: dict[str, torch.Tensor],
    hf_quant_config: Optional[dict[str, Any]] = None,
) -> None:
    """Write ``variants/<variant>/`` with BFL-named ``tensors``."""
    from safetensors.torch import save_file

    directory = root / "variants" / variant
    directory.mkdir(parents=True)
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(directory / "model.safetensors"),
    )
    hf_quant_config = hf_quant_config or bfl_hf_quant_config(config, variant)
    (directory / "hf_quant_config.json").write_text(json.dumps(hf_quant_config))
    (directory / "config.json").write_text(json.dumps(bfl_variant_config(config, variant)))


def placeholder_variant_tensors(config: dict[str, Any], variant: str) -> dict[str, torch.Tensor]:
    """Two-element tensors under every name of a variant, for tests that read names only."""
    return {name: torch.zeros(2) for name in bfl_variant_tensor_names(config, variant)}


def quantize_variant(
    config: dict[str, Any],
    weights: dict[str, torch.Tensor],
    variant: str,
    input_amax: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """BFL-named variant tensors of Diffusers BF16 ``weights``, quantized with TRT-LLM ops on CUDA.

    NVFP4 input scales come from ``input_amax``, the largest input of each Diffusers module.
    """
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_rowwise, quantize_nvfp4

    excluded = {bfl_to_diffusers_module(name) for name in bfl_variant_excludes(config, variant)}
    tensors = {}
    for name, value in weights.items():
        module = name[: -len(".weight")]
        if value.dim() != 2 or module in excluded:
            tensors[name] = value.cpu()
            continue
        if variant == "nvfp4":
            weight, scale, global_scale = quantize_nvfp4(value.cuda())
            # ModelOpt convention: input_scale = amax / (448 * 6).
            input_scale = torch.tensor(input_amax[module] / (448.0 * 6.0), dtype=torch.float32)
            tensors.update(
                {
                    f"{module}.weight_scale_2": global_scale.cpu(),
                    f"{module}.input_scale": input_scale,
                }
            )
        else:
            weight, scale = quantize_fp8_rowwise(value.cuda())
        tensors[name] = weight.cpu()
        tensors[f"{module}.weight_scale"] = scale.cpu()
    return diffusers_to_bfl_state_dict(tensors)
