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

Config tests need no checkpoint or GPU. Load tests require the checkpoints:

    DIFFUSION_MODEL_PATH_COSMOS3_NANO_FP8=/path/to/cosmos3-nano-fp8-14072026 \\
    DIFFUSION_MODEL_PATH_COSMOS3_SUPER_FP8=/path/to/cosmos3-super-fp8-14072026 \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_fp8.py -v
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

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


def _llm_models_root() -> str:
    """Resolve the checkpoint root, without asserting it exists.

    The path constants below call this at module scope, so raising here would
    error the whole module during collection -- including the config tests this
    module documents as needing neither a checkpoint nor a GPU. Returning a
    non-existent path instead lets the per-test ``_skip_if_missing`` guards skip
    only the tests that actually load weights.
    """
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return str(root)


def _checkpoint(env_var: str, *default_parts: str) -> str:
    return os.environ.get(env_var) or os.path.join(_llm_models_root(), *default_parts)


COSMOS3_NANO_FP8_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_COSMOS3_NANO_FP8",
    "Cosmos3-Nano-FP8",
    "cosmos3-nano-fp8-14072026",
)
COSMOS3_SUPER_FP8_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_COSMOS3_SUPER_FP8",
    "Cosmos3-Super-FP8",
    "cosmos3-super-fp8-14072026",
)
COSMOS3_NANO_BF16_PATH = _checkpoint("DIFFUSION_MODEL_PATH_COSMOS3", "Cosmos3-Nano")

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


def _skip_if_missing(path: str, label: str) -> str:
    if not path or not os.path.isdir(path):
        pytest.skip(f"{label} not found: {path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return path


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
    "checkpoint_path, label",
    [
        (COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8"),
        (COSMOS3_SUPER_FP8_PATH, "Cosmos3-Super-FP8"),
    ],
)
def test_checkpoint_config_resolves_to_static_fp8(checkpoint_path, label):
    """The real checkpoint on disk -- not just the recipe dict -- resolves to static FP8."""
    if not os.path.isdir(checkpoint_path):
        pytest.skip(f"{label} not found: {checkpoint_path}")

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
    if not os.path.isdir(COSMOS3_NANO_BF16_PATH):
        pytest.skip(f"Cosmos3-Nano not found: {COSMOS3_NANO_BF16_PATH}")

    config = DiffusionPipelineConfig.from_pretrained(
        COSMOS3_NANO_BF16_PATH, args=VisualGenArgs(model=COSMOS3_NANO_BF16_PATH)
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
    "checkpoint_path, label, static_fp8",
    [
        (COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8", True),
        (COSMOS3_NANO_BF16_PATH, "Cosmos3-Nano", False),
    ],
    ids=["fp8_splits", "bf16_stays_fused"],
)
def test_topology_follows_quantization(checkpoint_path, label, static_fp8):
    """Only static FP8 unfuses; BF16 must keep the fused topology untouched.

    The split exists solely to preserve per-projection calibration, which BF16
    does not have. Pinning both directions here means a change to the predicate
    cannot quietly alter the BF16 path -- the one every existing Cosmos3 user is
    on.
    """
    _skip_if_missing(checkpoint_path, label)

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
        assert und_attn.share_qkv_input_quant is static_fp8
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

    _skip_if_missing(COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8")

    model_config = DiffusionPipelineConfig.from_pretrained(
        COSMOS3_NANO_FP8_PATH, args=VisualGenArgs(model=COSMOS3_NANO_FP8_PATH)
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
        assert transformer.language_model.layers[0].self_attn.share_qkv_input_quant is False
    finally:
        del transformer
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.high_cuda_memory
@pytest.mark.parametrize(
    "checkpoint_path, label, size",
    [
        (COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8", "nano"),
        (COSMOS3_SUPER_FP8_PATH, "Cosmos3-Super-FP8", "super"),
    ],
)
def test_static_fp8_checkpoint_realizes_expected_module_layout(
    checkpoint_path, label, size, _cleanup_gpu
):
    """Load the real checkpoint and pin the realized dtype/scale layout.

    Super is exercised separately from Nano rather than inferred from it: it has
    a different depth and width, and roughly twice the quantized linear count.
    """
    _skip_if_missing(checkpoint_path, label)

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


# Groups TensorRT-LLM used to fuse, mapped checkpoint key -> runtime module.
# Fusion kept max(shard weight_scale) and requantized the other shards onto it,
# so these are precisely the projections whose calibration the split topology
# exists to preserve. Each entry is the worst shard-scale spread in its
# checkpoint (4.67x for Nano gen QKV, 6.10x for Super gen gate/up), per a full
# sweep of all fused groups -- the case with the most to lose.
PREVIOUSLY_FUSED_GROUPS = {
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
    "checkpoint_path, label, size",
    [
        (COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8", "nano"),
        (COSMOS3_SUPER_FP8_PATH, "Cosmos3-Super-FP8", "super"),
    ],
)
def test_previously_fused_groups_now_load_exactly(checkpoint_path, label, size, _cleanup_gpu):
    """Every member of a formerly fused group must transcribe bit-for-bit.

    Fusion kept one weight scale per group and re-quantized the other members
    onto it. Splitting the topology is only worth doing if each projection now
    loads its own tensor and its own scale untouched, so this asserts exact
    equality rather than a tolerance -- there is no arithmetic left to drift.

    The group's weight scales are asserted to actually differ first. Were they
    equal, fusion would have been lossless and exactness here would hold
    trivially, so the check would no longer discriminate between the two
    topologies.
    """
    _skip_if_missing(checkpoint_path, label)

    group = PREVIOUSLY_FUSED_GROUPS[size]
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
    # shared input scale per group. The split path relies on that to quantize
    # the activation once and hand the same tensor to each projection.
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
    _skip_if_missing(COSMOS3_NANO_FP8_PATH, "Cosmos3-Nano-FP8")

    # An unfused UND projection: its scales must transcribe exactly, with none
    # of the max-scale rescaling the fused groups undergo.
    checkpoint_key = "layers.0.self_attn.to_q"
    tensors = _load_checkpoint_tensors(COSMOS3_NANO_FP8_PATH, [checkpoint_key])
    expected_weight_scale = tensors[f"{checkpoint_key}.weight_scale"].float().item()
    expected_input_scale = tensors[f"{checkpoint_key}.input_scale"].float().item()

    pipeline = _load_transformer(COSMOS3_NANO_FP8_PATH)
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
