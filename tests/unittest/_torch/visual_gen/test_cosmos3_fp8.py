# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the statically quantized (ModelOpt FP8) Cosmos3 checkpoints.

These checkpoints ship FP8 ``E4M3`` weights alongside calibrated per-tensor
weight *and* activation scales, so inference must use static activation
quantization. That is distinct from ``TestCosmos3FP8Load`` in
``test_cosmos3_pipeline.py``, which quantizes a BF16 checkpoint dynamically at
load time from a user-supplied quant config.

The FP8 path is expected to work through the existing static-FP8 machinery
without Cosmos3-specific quantization code; these tests pin that contract so a
regression in config resolution, scale loading, or module exclusion is caught.

Config tests need no GPU. Every checkpoint resolves under ``LLM_MODELS_ROOT``
and a missing one fails loudly rather than skipping:

    LLM_MODELS_ROOT=/path/to/llm-models \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_fp8.py -v
"""

import gc
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CompilationConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

pytestmark = [pytest.mark.cosmos3, pytest.mark.usefixtures("disable_cosmos3_guardrails")]


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """TLLM_DISABLE_MPI has to be set before the imports above, so it cannot be
    a fixture -- but leaving it set makes any later module in the same process
    inherit it. Drop it on the way out, as test_cosmos3_pipeline.py does."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# Verbatim ``quantization_config`` shape exported by ModelOpt 0.44.0 into the
# published Cosmos3 FP8 checkpoints' ``transformer/config.json``. Only the keys
# TensorRT-LLM consumes are kept; ``dynamic: false`` on both weights and
# activations is what selects the static path.
MODELOPT_FP8_QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quant_type": "FP8_FP8",
    "quant_algo": "FP8",
    "weight_only": False,
    "config_groups": {
        "group_0": {
            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "targets": ["Linear"],
        }
    },
    "ignore": [
        "proj_in",
        "proj_out",
        "time_embedder*",
        "audio_proj_in",
        "audio_proj_out",
        "action_proj_in",
        "action_proj_out",
        "lm_head",
        "model.visual*",
        "visual*",
    ],
    "producer": {"name": "modelopt", "version": "0.44.0"},
}


# Checkpoint locations under LLM_MODELS_ROOT. Resolution is deliberately
# deferred to get_checkpoint() inside each test: these names are parametrize
# arguments, which are evaluated at collection, and touching the filesystem
# there would turn a missing checkpoint into a collection error.
COSMOS3_NANO_FP8_SUBDIR = "Cosmos3-Nano-FP8/cosmos3-nano-fp8-14072026"
COSMOS3_SUPER_FP8_SUBDIR = "Cosmos3-Super-FP8/cosmos3-super-fp8-14072026"
COSMOS3_NANO_BF16_SUBDIR = "Cosmos3-Nano"

# Runtime TensorRT-LLM ``Linear`` counts per tower. Static FP8 keeps every
# projection separate, so these are exactly the checkpoint's own projection
# counts -- 7 per layer per tower (q, k, v, out, gate, up, down) over 36 Nano
# and 64 Super layers. Under the fused topology GEN QKV and both towers'
# gate/up pairs collapsed, giving 216/144 (Nano) and 384/256 (Super); the
# totals below are what a 1:1 checkpoint mapping looks like. Pinning the split
# catches a tower silently dropping out of quantization *and* catches the
# topology silently reverting to fused.
EXPECTED_FP8_LINEARS = {
    "nano": {"UND": 252, "GEN": 252},
    "super": {"UND": 448, "GEN": 448},
}

# Boundary projections the checkpoint excludes from quantization. These are
# built as native ``nn.Linear`` (never TensorRT-LLM ``Linear``), so they are
# structurally incapable of being quantized -- assert that stays true.
EXPECTED_NATIVE_LINEARS = {
    "vae2llm",
    "llm2vae",
    "audio2llm",
    "llm2audio",
    "time_embedder.mlp.linear_1",
    "time_embedder.mlp.linear_2",
}


def _requires_cuda() -> None:
    """Gate on hardware only.

    A missing checkpoint is a staging failure and raises from get_checkpoint();
    absent CUDA is genuine environment gating and stays a skip.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _tower_of(module_name: str) -> str:
    if module_name.startswith("language_model"):
        return "UND"
    if module_name.startswith("gen_layers"):
        return "GEN"
    return "other"


def _load_transformer(checkpoint_path: str):
    args = VisualGenArgs(
        model=checkpoint_path,
        compilation_config=CompilationConfig(skip_warmup=True),
        torch_compile_config=TorchCompileConfig(enable=False),
        attention_config=AttentionConfig(backend="VANILLA"),
    )
    return PipelineLoader(args).load(skip_warmup=True)


@pytest.fixture
def _cleanup_gpu():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestStaticFp8ConfigResolution:
    """The ModelOpt recipe must resolve to *static* FP8 with no extra plumbing."""

    def test_modelopt_recipe_resolves_to_static_fp8(self):
        quant_config, layer_quant_config, dynamic_weight, dynamic_activation = (
            DiffusionPipelineConfig.load_diffusion_quant_config(MODELOPT_FP8_QUANT_CONFIG)
        )

        assert quant_config.quant_algo == QuantAlgo.FP8
        # ``dynamic: false`` in the checkpoint must not decay into runtime
        # quantization: the calibrated weight/activation scales would be ignored.
        assert dynamic_weight is False
        assert dynamic_activation is False
        assert layer_quant_config is None

    def test_checkpoint_ignore_list_becomes_exclude_modules(self):
        quant_config, _, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
            MODELOPT_FP8_QUANT_CONFIG
        )

        assert quant_config.exclude_modules == MODELOPT_FP8_QUANT_CONFIG["ignore"]
        for excluded in ("proj_in", "proj_out", "lm_head"):
            assert quant_config.is_module_excluded_from_quantization(excluded)

    def test_absent_quantization_config_resolves_to_no_quantization(self):
        quant_config, layer_quant_config, dynamic_weight, dynamic_activation = (
            DiffusionPipelineConfig.load_diffusion_quant_config({})
        )

        assert quant_config.quant_algo is None
        assert layer_quant_config is None
        assert dynamic_weight is False
        assert dynamic_activation is False


@pytest.mark.parametrize(
    "checkpoint_subdir",
    [COSMOS3_NANO_FP8_SUBDIR, COSMOS3_SUPER_FP8_SUBDIR],
    ids=["nano", "super"],
)
def test_checkpoint_config_resolves_to_static_fp8(checkpoint_subdir):
    """The real checkpoint on disk -- not just the recipe dict -- resolves to static FP8."""
    checkpoint_path = get_checkpoint(checkpoint_subdir)

    config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_path, args=VisualGenArgs(model=checkpoint_path)
    )
    transformer_config = config.primary_model_config

    assert transformer_config.quant_config.quant_algo == QuantAlgo.FP8
    assert transformer_config.dynamic_weight_quant is False
    assert transformer_config.force_dynamic_quantization is False
    assert transformer_config.quant_config.exclude_modules is not None


def test_bf16_checkpoint_config_resolves_to_no_quantization():
    """Regression: adding FP8 support must not quantize the BF16 checkpoints."""
    checkpoint_path = get_checkpoint(COSMOS3_NANO_BF16_SUBDIR)

    config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_path, args=VisualGenArgs(model=checkpoint_path)
    )

    assert config.primary_model_config.quant_config.quant_algo is None


def _build_two_layer_transformer(checkpoint_path):
    """Build the transformer from a checkpoint's config, trimmed to two layers.

    Only the topology is under test, so the layer count is cut to keep the build
    cheap. No weights are loaded.
    """
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )

    model_config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_path, args=VisualGenArgs(model=checkpoint_path)
    ).primary_model_config
    model_config.pretrained_config.num_hidden_layers = 2
    return Cosmos3VFMTransformer(model_config=model_config)


@pytest.mark.parametrize(
    "checkpoint_subdir, static_fp8",
    [
        (COSMOS3_NANO_FP8_SUBDIR, True),
        (COSMOS3_NANO_BF16_SUBDIR, False),
    ],
    ids=["fp8_splits", "bf16_stays_fused"],
)
def test_topology_follows_quantization(checkpoint_subdir, static_fp8):
    """Only static FP8 unfuses; BF16 must keep the fused topology untouched.

    The split exists solely to preserve per-projection calibration, which BF16
    does not have. Pinning both directions here means a change to the predicate
    cannot quietly alter the BF16 path -- the one every existing Cosmos3 user is
    on.
    """
    _requires_cuda()
    checkpoint_path = get_checkpoint(checkpoint_subdir)
    label = checkpoint_subdir

    transformer = _build_two_layer_transformer(checkpoint_path)
    try:
        names = set(dict(transformer.named_modules()))
        gen_attn, gen_mlp = "gen_layers.0.cross_attention", "gen_layers.0.mlp"
        und_mlp = "language_model.layers.0.mlp"

        if static_fp8:
            for split in (f"{gen_attn}.to_q", f"{gen_attn}.to_k", f"{gen_attn}.to_v"):
                assert split in names, f"{label}: expected split {split}"
            assert f"{gen_attn}.qkv_proj" not in names, f"{label}: GEN QKV still fused"
            for mlp in (gen_mlp, und_mlp):
                assert f"{mlp}.gate_proj" in names and f"{mlp}.up_proj" in names
                assert f"{mlp}.gate_up_proj" not in names, f"{label}: {mlp} still fused"
        else:
            assert f"{gen_attn}.qkv_proj" in names, f"{label}: GEN QKV unexpectedly split"
            for mlp in (gen_mlp, und_mlp):
                assert f"{mlp}.gate_up_proj" in names, f"{label}: {mlp} unexpectedly split"
                assert f"{mlp}.gate_proj" not in names

        # The UND tower is SEPARATE_QKV in both configurations; only the shared
        # activation quantization is conditional.
        und_attn = transformer.language_model.layers[0].self_attn
        assert und_attn._maybe_share_qkv_quantize is static_fp8
    finally:
        del transformer
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("dynamic_field", ["dynamic_weight_quant", "force_dynamic_quantization"])
def test_dynamic_quantization_stays_fused(dynamic_field):
    """Dynamic FP8 has no calibration to preserve, so it must keep fusing.

    Both dynamic flavors resolve to ``quant_algo == FP8``, so a predicate that
    keyed on the algorithm alone would unfuse them too -- and the split path
    would then quantize activations against a scale that does not exist yet.
    """
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import uses_static_fp8

    _requires_cuda()
    checkpoint_path = get_checkpoint(COSMOS3_NANO_FP8_SUBDIR)

    model_config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_path, args=VisualGenArgs(model=checkpoint_path)
    ).primary_model_config

    assert uses_static_fp8(model_config) is True
    setattr(model_config, dynamic_field, True)
    assert uses_static_fp8(model_config) is False

    model_config.pretrained_config.num_hidden_layers = 2
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )

    transformer = Cosmos3VFMTransformer(model_config=model_config)
    try:
        names = set(dict(transformer.named_modules()))
        assert "gen_layers.0.cross_attention.qkv_proj" in names
        assert "gen_layers.0.mlp.gate_up_proj" in names
        assert transformer.language_model.layers[0].self_attn._maybe_share_qkv_quantize is False
    finally:
        del transformer
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.high_cuda_memory
@pytest.mark.parametrize(
    "checkpoint_subdir, size",
    [
        (COSMOS3_NANO_FP8_SUBDIR, "nano"),
        (COSMOS3_SUPER_FP8_SUBDIR, "super"),
    ],
)
def test_static_fp8_checkpoint_realizes_expected_module_layout(
    checkpoint_subdir, size, _cleanup_gpu
):
    """Load the real checkpoint and pin the realized dtype/scale layout.

    Super is exercised separately from Nano rather than inferred from it: it has
    a different depth and width, and roughly twice the quantized linear count.
    """
    _requires_cuda()
    checkpoint_path = get_checkpoint(checkpoint_subdir)
    label = checkpoint_subdir

    pipeline = _load_transformer(checkpoint_path)
    try:
        transformer = pipeline.transformer

        fp8_by_tower = {"UND": 0, "GEN": 0, "other": 0}
        missing_scales = []
        native_linears = {}

        for name, module in transformer.named_modules():
            if isinstance(module, Linear):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.dtype == torch.float8_e4m3fn:
                    fp8_by_tower[_tower_of(name)] += 1
                    # Both scales must survive loading: ``weight_scale``
                    # dequantizes the GEMM, ``input_scale`` is what makes the
                    # activation path static rather than dynamic.
                    if getattr(module, "weight_scale", None) is None:
                        missing_scales.append(f"{name}.weight_scale")
                    if getattr(module, "input_scale", None) is None:
                        missing_scales.append(f"{name}.input_scale")
            elif isinstance(module, torch.nn.Linear):
                native_linears[name] = module.weight.dtype

        assert not missing_scales, f"{label}: missing FP8 scales: {missing_scales[:10]}"

        expected = EXPECTED_FP8_LINEARS[size]
        assert fp8_by_tower["UND"] == expected["UND"], (
            f"{label}: UND tower FP8 linears {fp8_by_tower['UND']} != {expected['UND']}"
        )
        assert fp8_by_tower["GEN"] == expected["GEN"], (
            f"{label}: GEN tower FP8 linears {fp8_by_tower['GEN']} != {expected['GEN']}"
        )

        assert EXPECTED_NATIVE_LINEARS.issubset(set(native_linears)), (
            f"{label}: expected native boundary linears missing: "
            f"{EXPECTED_NATIVE_LINEARS - set(native_linears)}"
        )
        for boundary in ("vae2llm", "llm2vae", "audio2llm", "llm2audio"):
            assert native_linears[boundary] == torch.bfloat16, (
                f"{label}: {boundary} should stay BF16, got {native_linears[boundary]}"
            )

        # ``post_load_weights`` deliberately promotes the timestep embedder to
        # FP32 for precision; it is excluded from quantization in the checkpoint.
        timestep_dtypes = {p.dtype for p in transformer.time_embedder.parameters()}
        assert timestep_dtypes == {torch.float32}, (
            f"{label}: time_embedder should be FP32, got {timestep_dtypes}"
        )
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# Groups the fused topology merges into one Linear, mapped checkpoint key ->
# runtime module. Fusing keeps max(shard weight_scale) and requantizes the other
# shards onto it, so these are precisely the projections whose calibration the
# split topology exists to preserve. Each entry is the worst shard-scale spread
# in its checkpoint (4.67x for Nano gen QKV, 6.10x for Super gen gate/up), per a
# full sweep of all fused groups -- the case with the most to lose.
FUSED_TOPOLOGY_GROUPS = {
    "nano": {
        "layers.32.self_attn.add_q_proj": "gen_layers.32.cross_attention.to_q",
        "layers.32.self_attn.add_k_proj": "gen_layers.32.cross_attention.to_k",
        "layers.32.self_attn.add_v_proj": "gen_layers.32.cross_attention.to_v",
    },
    "super": {
        "layers.7.mlp_moe_gen.gate_proj": "gen_layers.7.mlp.gate_proj",
        "layers.7.mlp_moe_gen.up_proj": "gen_layers.7.mlp.up_proj",
    },
}


def _load_checkpoint_tensors(
    checkpoint_path, keys, suffixes=("weight", "weight_scale", "input_scale")
):
    import json

    from safetensors.torch import load_file

    transformer_dir = os.path.join(checkpoint_path, "transformer")
    with open(
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
    ) as handle:
        weight_map = json.load(handle)["weight_map"]

    shards, tensors = {}, {}
    for key in keys:
        for suffix in suffixes:
            full_key = f"{key}.{suffix}"
            if full_key not in weight_map:
                continue
            shard = weight_map[full_key]
            if shard not in shards:
                shards[shard] = load_file(os.path.join(transformer_dir, shard))
            tensors[full_key] = shards[shard][full_key]
    return tensors


@pytest.mark.integration
@pytest.mark.high_cuda_memory
@pytest.mark.parametrize(
    "checkpoint_subdir, size",
    [
        (COSMOS3_NANO_FP8_SUBDIR, "nano"),
        (COSMOS3_SUPER_FP8_SUBDIR, "super"),
    ],
)
def test_split_groups_load_exactly(checkpoint_subdir, size, _cleanup_gpu):
    """Every member of a group the fused topology merges must transcribe bit-for-bit.

    Fusing keeps one weight scale per group and re-quantizes the other members
    onto it. Splitting the topology is only worth doing if each projection
    loads its own tensor and its own scale untouched, so this asserts exact
    equality rather than a tolerance -- there is no arithmetic left to drift.

    The group's weight scales are asserted to actually differ first. Were they
    equal, fusing would be lossless and exactness here would hold trivially,
    so the check could not tell the two topologies apart.
    """
    _requires_cuda()
    checkpoint_path = get_checkpoint(checkpoint_subdir)
    label = checkpoint_subdir

    group = FUSED_TOPOLOGY_GROUPS[size]
    tensors = _load_checkpoint_tensors(checkpoint_path, list(group))
    first_key = next(iter(group))
    if f"{first_key}.weight" not in tensors:
        pytest.skip(f"{label}: group {first_key} not present in checkpoint")

    weight_scales = {k: tensors[f"{k}.weight_scale"].float().item() for k in group}
    input_scales = {k: tensors[f"{k}.input_scale"].float().item() for k in group}

    assert len(set(weight_scales.values())) > 1, (
        f"{label}: group {list(group)} has a single weight scale {weight_scales}, so it "
        "cannot distinguish split loading from fused requantization -- pick a "
        "group whose shard scales differ"
    )

    # q/k/v (and gate/up) see the same activation, so ModelOpt calibrates one
    # shared input scale per group. With equal scales the split path quantizes
    # the activation once and hands the same tensor to each projection.
    assert len(set(input_scales.values())) == 1, (
        f"{label}: group {list(group)} has differing input scales {input_scales}"
    )

    pipeline = _load_transformer(checkpoint_path)
    try:
        modules = dict(pipeline.transformer.named_modules())
        for checkpoint_key, runtime_name in group.items():
            parent = runtime_name.rsplit(".", 1)[0]
            siblings = sorted(n for n in modules if n.startswith(parent))[:8]
            assert runtime_name in modules, (
                f"{label}: expected split module {runtime_name}; the topology may "
                f"have reverted to fused (present: {siblings})"
            )
            module = modules[runtime_name]

            assert module.weight_scale.float().item() == pytest.approx(
                weight_scales[checkpoint_key], rel=0, abs=0
            ), f"{label}: {runtime_name} weight_scale was rescaled"
            assert module.input_scale.float().item() == pytest.approx(
                input_scales[checkpoint_key], rel=0, abs=0
            ), f"{label}: {runtime_name} input_scale was rescaled"

            expected = tensors[f"{checkpoint_key}.weight"]
            actual = module.weight.detach().cpu()
            assert actual.dtype == expected.dtype == torch.float8_e4m3fn
            # Compared as bits: FP8 has no exact torch.equal on all platforms and
            # this must catch a single re-rounded element.
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
                f"{label}: {runtime_name} weight differs from the checkpoint; "
                "it was re-quantized rather than loaded directly"
            )
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.high_cuda_memory
def test_static_fp8_scales_match_checkpoint_calibration(_cleanup_gpu):
    """Loaded scales must equal the checkpoint's calibrated values.

    ModelOpt stores ``<module>.weight_scale``/``<module>.input_scale`` next to
    duplicate quantizer-internal tensors (``weight_quantizer._scale``,
    ``*._amax``). Reading the wrong one -- or silently falling back to a
    computed scale -- would still produce plausible images, so compare against
    the raw checkpoint tensors.
    """
    _requires_cuda()
    checkpoint_path = get_checkpoint(COSMOS3_NANO_FP8_SUBDIR)

    # An unfused UND projection: its scales must transcribe exactly, with none
    # of the max-scale rescaling the fused groups undergo.
    checkpoint_key = "layers.0.self_attn.to_q"
    tensors = _load_checkpoint_tensors(checkpoint_path, [checkpoint_key])
    expected_weight_scale = tensors[f"{checkpoint_key}.weight_scale"].float().item()
    expected_input_scale = tensors[f"{checkpoint_key}.input_scale"].float().item()

    pipeline = _load_transformer(checkpoint_path)
    try:
        module = dict(pipeline.transformer.named_modules())[
            "language_model.layers.0.self_attn.to_q"
        ]
        assert module.weight.dtype == torch.float8_e4m3fn
        assert module.weight_scale.float().item() == pytest.approx(expected_weight_scale, rel=1e-6)
        assert module.input_scale.float().item() == pytest.approx(expected_input_scale, rel=1e-6)
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Distilled 4-step sampling with a static-FP8 transformer (synthetic weights)
# =============================================================================
#
# The FP8 4-Step distilled checkpoints combine two features that are otherwise
# only tested apart: the fixed-step SDE sampler (test_cosmos3_distilled.py,
# never with a real transformer) and the static-FP8 W8A8 transformer (above,
# never inside a sampling loop). This section runs the real 4-step
# FlowMatchEuler SDE schedule through a small random-weight static-FP8
# transformer and compares the final latents against the same loop over a BF16
# twin carrying the dequantized weights, so the only difference is activation
# quantization.

# Same small GQA architecture as the multi-GPU parity tests.
_SYNTH_PRETRAINED_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "latent_patch_size": 2,
    "latent_channel": 4,
    "position_embedding_type": "unified_3d_mrope",
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "rope_scaling": {"rope_type": "default", "mrope_section": [12, 10, 10]},
    "rms_norm_eps": 1e-6,
    "vocab_size": 1024,
    "rope_theta": 1_000_000.0,
    "max_position_embeddings": 4096,
    "timestep_scale": 1.0,
    "base_fps": 24.0,
    "unified_3d_mrope_temporal_modality_margin": 100,
    "enable_fps_modulation": True,
}

# The 4-Step checkpoints' scheduler recipe (subset consumed by the sampler).
_DISTILLED_SIGMAS = (1.0, 0.9375, 0.8333333333333334, 0.625)
_DISTILLED_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": True,
    "use_karras_sigmas": False,
    "fixed_step_requires_explicit_sigmas": True,
    "fixed_step_sampler_config": {"sample_type": "sde", "t_list": list(_DISTILLED_SIGMAS)},
}

_SEED_FP8_WEIGHTS = 123
_SEED_LATENTS = 456
_SEED_TEXT = 42
_SEED_SDE = 987
_SYNTH_INPUT_SCALE = 1e-2
_SCALE_SUFFIXES = ("weight_scale", "input_scale", "inv_input_scale", "kv_scales")
_LATENT_SHAPE = (1, 4, 2, 4, 4)  # [B, C, T, H, W]; patch 2 -> 8 gen tokens
_TEXT_LEN = 8
_MAX_TEXT_LEN = 16


def _make_synth_model_config(quant_algo):
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**_SYNTH_PRETRAINED_CONFIG),
        quant_config=QuantConfig(quant_algo=quant_algo) if quant_algo else QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )


def _init_synth_static_fp8(model) -> None:
    """Synthesize a calibrated static-FP8 state (same recipe as the parallel tests)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype == torch.float8_e4m3fn or name.endswith(_SCALE_SUFFIXES):
                continue
            if "norm" in name and name.endswith(".weight"):
                p.fill_(1.0)
            elif p.ndim >= 2:
                std = 0.02 / max(1.0, p.shape[1] ** 0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)
        for _, module in model.named_modules():
            if not (isinstance(module, Linear) and module.weight.dtype == torch.float8_e4m3fn):
                continue
            std = 0.02 / max(1.0, module.weight.shape[1] ** 0.5)
            w = torch.empty(
                module.weight.shape, device=module.weight.device, dtype=torch.float32
            ).uniform_(-std, std)
            weight_scale = w.abs().amax() / 448.0
            module.weight.data.copy_((w / weight_scale).to(torch.float8_e4m3fn))
            module.weight_scale.data.copy_(weight_scale)
            module.input_scale.data.fill_(_SYNTH_INPUT_SCALE)
            module.inv_input_scale.data.fill_(1.0 / _SYNTH_INPUT_SCALE)


def _copy_dequantized_weights(fp8_model, bf16_model) -> None:
    """Load the BF16 twin with the FP8 model's dequantized weights.

    The static-FP8 topology splits GEN q/k/v and both towers' gate/up, while
    BF16 fuses them, so fused destinations concatenate the dequantized splits.
    After this, the two models compute the same function up to activation
    quantization (and BF16 rounding of the dequantized weights).
    """
    fp8_modules = dict(fp8_model.named_modules())
    fp8_params = dict(fp8_model.named_parameters())

    def dequantize(module_name: str) -> torch.Tensor:
        module = fp8_modules[module_name]
        return (module.weight.float() * module.weight_scale.float()).to(torch.bfloat16)

    with torch.no_grad():
        for name, p in bf16_model.named_parameters():
            if name in fp8_params:
                src = fp8_params[name]
                if src.dtype == torch.float8_e4m3fn:
                    p.copy_(dequantize(name.rsplit(".", 1)[0]).to(p.dtype))
                else:
                    p.copy_(src.to(p.dtype))
            elif name.endswith("qkv_proj.weight"):
                prefix = name[: -len("qkv_proj.weight")]
                p.copy_(
                    torch.cat(
                        [dequantize(prefix + part) for part in ("to_q", "to_k", "to_v")], dim=0
                    ).to(p.dtype)
                )
            elif name.endswith("gate_up_proj.weight"):
                prefix = name[: -len("gate_up_proj.weight")]
                p.copy_(
                    torch.cat(
                        [dequantize(prefix + part) for part in ("gate_proj", "up_proj")], dim=0
                    ).to(p.dtype)
                )
            else:
                raise AssertionError(f"No FP8 source for BF16 parameter {name}")


def _build_distilled_test_models(device):
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )

    torch.manual_seed(_SEED_FP8_WEIGHTS)
    fp8_model = Cosmos3VFMTransformer(_make_synth_model_config(QuantAlgo.FP8)).to(device).eval()
    _init_synth_static_fp8(fp8_model)
    fp8_model.post_load_weights()

    cross_attn = fp8_model.gen_layers[0].cross_attention
    assert cross_attn.to_q.weight.dtype == torch.float8_e4m3fn
    assert cross_attn._maybe_share_qkv_quantize is True
    assert fp8_model.gen_layers[0].mlp._maybe_share_gate_up_quantize is True

    bf16_model = Cosmos3VFMTransformer(_make_synth_model_config(None)).to(device).eval()
    _copy_dequantized_weights(fp8_model, bf16_model)
    bf16_model.post_load_weights()
    return fp8_model, bf16_model


def _run_distilled_loop(model, device) -> torch.Tensor:
    """The 4-step fixed-sigma SDE denoise loop the distilled checkpoints run."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import Cosmos3SamplingPolicy

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(_DISTILLED_SCHEDULER_CONFIG)
    policy = Cosmos3SamplingPolicy.from_scheduler(scheduler)
    policy.set_timesteps(scheduler, num_inference_steps=len(_DISTILLED_SIGMAS), device=device)
    step_kwargs = policy.scheduler_step_kwargs(torch.Generator().manual_seed(_SEED_SDE))

    assert torch.allclose(
        scheduler.timesteps.float().cpu(),
        torch.tensor([s * 1000.0 for s in _DISTILLED_SIGMAS]),
        atol=1e-3,
    )

    torch.manual_seed(_SEED_LATENTS)
    latents = torch.randn(_LATENT_SHAPE, device=device, dtype=torch.bfloat16)
    torch.manual_seed(_SEED_TEXT)
    text_ids = torch.randint(1, 1000, (1, _MAX_TEXT_LEN), device=device, dtype=torch.long)
    text_mask = torch.zeros(1, _MAX_TEXT_LEN, device=device, dtype=torch.long)
    text_mask[:, :_TEXT_LEN] = 1
    video_shape = _LATENT_SHAPE[2:]

    for t in scheduler.timesteps:
        raw_timestep = torch.full((1,), float(t), device=device, dtype=torch.float32)
        model.reset_cache()
        with torch.inference_mode():
            velocity = model(
                hidden_states=latents,
                timestep=raw_timestep / scheduler.config.num_train_timesteps,
                raw_timestep=raw_timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=24.0,
            ).video
        assert velocity.shape == latents.shape
        latents = scheduler.step(velocity, t, latents, return_dict=False, **step_kwargs)[0]

    return latents


class TestDistilledSamplingWithStaticFp8:
    """4-step SDE sampling drives a static-FP8 transformer (no checkpoint)."""

    def test_fp8_matches_dequantized_bf16_through_distilled_loop(self, _cleanup_gpu):
        _requires_cuda()
        device = torch.device("cuda")
        fp8_model, bf16_model = _build_distilled_test_models(device)

        fp8_latents = _run_distilled_loop(fp8_model, device).float()
        ref_latents = _run_distilled_loop(bf16_model, device).float()

        assert not torch.isnan(fp8_latents).any()
        assert not torch.isinf(fp8_latents).any()

        error = fp8_latents - ref_latents
        ref_norm = torch.linalg.vector_norm(ref_latents)
        assert ref_norm > 0
        relative_l2 = (torch.linalg.vector_norm(error) / ref_norm).item()
        cosine = torch.nn.functional.cosine_similarity(
            fp8_latents.flatten(), ref_latents.flatten(), dim=0
        ).item()
        print(
            f"[distilled fp8 vs dequantized bf16] relative_l2={relative_l2:.4e}, cosine={cosine:.6f}"
        )

        # The only difference is per-layer activation quantization accumulated
        # over 4 layers x 4 steps, and the residual stream damps it: measured
        # 1.7e-5 on B300. A scale or wiring bug perturbs the attention/MLP
        # contributions by O(1) and lands orders of magnitude above this.
        assert relative_l2 <= 1e-3
        assert cosine >= 0.9999

    def test_fp8_distilled_loop_is_deterministic(self, _cleanup_gpu):
        _requires_cuda()
        device = torch.device("cuda")
        fp8_model, _ = _build_distilled_test_models(device)

        first = _run_distilled_loop(fp8_model, device)
        second = _run_distilled_loop(fp8_model, device)
        assert torch.equal(first, second)
