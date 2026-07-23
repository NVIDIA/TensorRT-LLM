# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Edge (Nemotron-dense backbone) tests.

Unit tests run on a reduced config mirroring the Edge checkpoint's exact key
set: recipe validation, Nemotron norm semantics, the generator-only und
K-norm, native flow schedule parity against cosmos-framework, strict weight
loading, and per-family defaults. Checkpoint-gated tests cover the real
checkpoint (tokenizer, recipe/scheduler wiring, load + forward).

Override checkpoint:
    DIFFUSION_MODEL_PATH_COSMOS3_EDGE=/path/to/Cosmos3-Edge \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_edge.py -v
"""

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import UniPCMultistepScheduler

from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_cosmos3 as pipeline_module
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_EDGE_T2I_PARAMS,
    COSMOS3_EDGE_VIDEO_PARAMS,
    COSMOS3_GENERATION_DEFAULTS,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import Cosmos3SamplingPolicy
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    COSMOS3_EDGE_BACKBONE_TYPE,
    NEMOTRON_DENSE_RECIPE,
    QWEN3_RECIPE,
    Cosmos3VFMTransformer,
    NemotronRMSNorm,
    Qwen3VLTextRMSNorm,
    resolve_arch_recipe,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = [pytest.mark.cosmos3, pytest.mark.usefixtures("disable_cosmos3_guardrails")]

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_edge_checkpoint() -> str:
    """Resolve the Edge checkpoint lazily so unit tests collect and run on
    machines without model storage; only checkpoint-gated tests skip."""
    path = os.environ.get("DIFFUSION_MODEL_PATH_COSMOS3_EDGE")
    if not path:
        root = Path(os.environ.get("LLM_MODELS_ROOT", "/home/scratch.trt_llm_data_ci/llm-models/"))
        if not root.exists():
            root = Path("/scratch/trt_llm_data/llm-models/")
        path = str(root / "Cosmos3-Edge")
    if not os.path.isdir(path):
        pytest.skip(f"Checkpoint not found: {path}")
    return path


def _reduced_edge_config() -> SimpleNamespace:
    # Key set mirrors the Edge checkpoint's transformer/config.json verbatim
    # (including the missing rope_type in rope_scaling); only sizes shrink.
    return SimpleNamespace(
        action_dim=8,
        action_gen=True,
        attention_bias=False,
        attention_dropout=0.0,
        backbone_type=COSMOS3_EDGE_BACKBONE_TYPE,
        base_fps=24,
        enable_fps_modulation=True,
        head_dim=8,
        hidden_act="relu2",
        hidden_size=32,
        intermediate_size=64,
        # Latent geometry stays at the real invariant values (validated for
        # the Edge recipe); only backbone dimensions shrink.
        latent_channel=48,
        latent_patch_size=2,
        num_attention_heads=4,
        num_embodiment_domains=32,
        num_hidden_layers=2,
        num_key_value_heads=2,
        patch_latent_dim=192,
        qk_norm_for_text=False,
        rms_norm_eps=1e-5,
        rope_axes_dim=[2, 1, 1],
        rope_scaling={"mrope_section": [2, 1, 1]},
        rope_theta=100000000,
        sound_dim=None,
        sound_gen=False,
        temporal_compression_factor=4,
        timestep_scale=0.001,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        use_und_k_norm_for_gen=True,
        vocab_size=64,
    )


def _reduced_edge_model_config() -> DiffusionModelConfig:
    model_config = DiffusionModelConfig(pretrained_config=_reduced_edge_config())
    model_config.attention.backend = "VANILLA"
    return model_config


def _reduced_qwen3_config(**overrides) -> SimpleNamespace:
    cfg = _reduced_edge_config()
    cfg.backbone_type = None
    cfg.hidden_act = "silu"
    cfg.qk_norm_for_text = True
    cfg.use_und_k_norm_for_gen = False
    cfg.rms_norm_eps = 1e-6
    cfg.rope_theta = 5000000
    cfg.rope_scaling = {"mrope_section": [2, 1, 1], "rope_type": "default"}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _reduced_qwen3_model_config(**overrides) -> DiffusionModelConfig:
    model_config = DiffusionModelConfig(pretrained_config=_reduced_qwen3_config(**overrides))
    model_config.attention.backend = "VANILLA"
    return model_config


def _synthetic_state_dict(cfg: SimpleNamespace) -> dict:
    """A complete synthetic checkpoint for either recipe (diffusers key layout)."""
    h, d = cfg.hidden_size, cfg.head_dim
    q_dim = cfg.num_attention_heads * d
    kv_dim = cfg.num_key_value_heads * d
    gated = cfg.hidden_act == "silu"
    sd = {
        "embed_tokens.weight": torch.randn(cfg.vocab_size, h),
        "lm_head.weight": torch.randn(cfg.vocab_size, h),
        "norm.weight": torch.ones(h),
        "norm_moe_gen.weight": torch.ones(h),
        "proj_in.weight": torch.randn(h, cfg.patch_latent_dim),
        "proj_in.bias": torch.randn(h),
        "proj_out.weight": torch.randn(cfg.patch_latent_dim, h),
        "proj_out.bias": torch.randn(cfg.patch_latent_dim),
        "time_embedder.linear_1.weight": torch.randn(h, 256),
        "time_embedder.linear_1.bias": torch.randn(h),
        "time_embedder.linear_2.weight": torch.randn(h, h),
        "time_embedder.linear_2.bias": torch.randn(h),
    }
    if getattr(cfg, "sound_gen", False):
        sd.update(
            {
                "audio_proj_in.weight": torch.randn(h, cfg.sound_dim),
                "audio_proj_in.bias": torch.randn(h),
                "audio_proj_out.weight": torch.randn(cfg.sound_dim, h),
                "audio_proj_out.bias": torch.randn(cfg.sound_dim),
                "audio_modality_embed": torch.randn(h),
            }
        )
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}"
        sd.update(
            {
                f"{p}.self_attn.to_q.weight": torch.randn(q_dim, h),
                f"{p}.self_attn.to_k.weight": torch.randn(kv_dim, h),
                f"{p}.self_attn.to_v.weight": torch.randn(kv_dim, h),
                f"{p}.self_attn.to_out.weight": torch.randn(h, q_dim),
                f"{p}.self_attn.add_q_proj.weight": torch.randn(q_dim, h),
                f"{p}.self_attn.add_k_proj.weight": torch.randn(kv_dim, h),
                f"{p}.self_attn.add_v_proj.weight": torch.randn(kv_dim, h),
                f"{p}.self_attn.to_add_out.weight": torch.randn(h, q_dim),
                f"{p}.self_attn.norm_added_q.weight": torch.ones(d),
                f"{p}.self_attn.norm_added_k.weight": torch.ones(d),
                f"{p}.input_layernorm.weight": torch.ones(h),
                f"{p}.input_layernorm_moe_gen.weight": torch.ones(h),
                f"{p}.post_attention_layernorm.weight": torch.ones(h),
                f"{p}.post_attention_layernorm_moe_gen.weight": torch.ones(h),
                f"{p}.mlp.up_proj.weight": torch.randn(cfg.intermediate_size, h),
                f"{p}.mlp.down_proj.weight": torch.randn(h, cfg.intermediate_size),
                f"{p}.mlp_moe_gen.up_proj.weight": torch.randn(cfg.intermediate_size, h),
                f"{p}.mlp_moe_gen.down_proj.weight": torch.randn(h, cfg.intermediate_size),
            }
        )
        if gated:
            sd.update(
                {
                    f"{p}.self_attn.norm_q.weight": torch.ones(d),
                    f"{p}.self_attn.norm_k.weight": torch.ones(d),
                    f"{p}.mlp.gate_proj.weight": torch.randn(cfg.intermediate_size, h),
                    f"{p}.mlp_moe_gen.gate_proj.weight": torch.randn(cfg.intermediate_size, h),
                }
            )
        else:
            sd[f"{p}.self_attn.k_norm_und_for_gen.weight"] = torch.ones(d)
    return sd


def _edge_state_dict(cfg: SimpleNamespace) -> dict:
    return _synthetic_state_dict(cfg)


class TestArchRecipe:
    def test_edge_config_resolves_nemotron_recipe(self):
        recipe = resolve_arch_recipe(_reduced_edge_config())
        assert recipe is NEMOTRON_DENSE_RECIPE

    def test_qwen3_resolves_without_backbone_type(self):
        cfg = SimpleNamespace(hidden_act="silu", qk_norm_for_text=True)
        assert resolve_arch_recipe(cfg) is QWEN3_RECIPE

    def test_unknown_backbone_raises(self):
        cfg = _reduced_edge_config()
        cfg.backbone_type = "cosmos4_hybrid"
        with pytest.raises(ValueError, match="cosmos4_hybrid"):
            resolve_arch_recipe(cfg)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("hidden_act", "silu"),
            ("qk_norm_for_text", True),
            ("use_und_k_norm_for_gen", False),
            ("sound_gen", True),
            ("sound_gen", None),
            ("attention_bias", True),
            ("rms_norm_eps", 1e-6),
        ],
    )
    def test_edge_flag_contradiction_raises(self, key, value):
        cfg = _reduced_edge_config()
        setattr(cfg, key, value)
        with pytest.raises(ValueError, match=key):
            resolve_arch_recipe(cfg)

    @pytest.mark.parametrize("make_config", [_reduced_edge_config, lambda: _reduced_qwen3_config()])
    def test_inconsistent_patch_latent_dim_raises(self, make_config):
        cfg = make_config()
        cfg.patch_latent_dim = cfg.patch_latent_dim + 1
        with pytest.raises(ValueError, match="patch_latent_dim"):
            resolve_arch_recipe(cfg)


class TestRopeAxesResolution:
    """The explicit top-level rope_axes_dim wins over the legacy
    rope_scaling.mrope_section (diffusers precedence); disagreement and
    head_dim mismatches are config errors."""

    def _resolve(self, cfg):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
            resolve_rope_axes_dim,
        )

        return resolve_rope_axes_dim(cfg)

    def test_top_level_only(self):
        cfg = _reduced_edge_config()
        cfg.rope_scaling = {}
        assert self._resolve(cfg) == [2, 1, 1]

    def test_nested_only(self):
        cfg = _reduced_edge_config()
        del cfg.rope_axes_dim
        assert self._resolve(cfg) == [2, 1, 1]

    def test_contradiction_raises(self):
        cfg = _reduced_edge_config()
        cfg.rope_axes_dim = [1, 2, 1]
        with pytest.raises(ValueError, match="contradictory"):
            self._resolve(cfg)

    @pytest.mark.parametrize("axes", [[2, 2, 1], [2, 2], [4]])
    def test_head_dim_mismatch_raises(self, axes):
        cfg = _reduced_edge_config()
        cfg.rope_axes_dim = axes
        cfg.rope_scaling = {}
        with pytest.raises(ValueError, match="head_dim"):
            self._resolve(cfg)

    def test_neither_declared_raises(self):
        cfg = _reduced_edge_config()
        del cfg.rope_axes_dim
        cfg.rope_scaling = {}
        with pytest.raises(ValueError, match="neither"):
            self._resolve(cfg)

    def test_transformer_builds_from_top_level_only(self):
        cfg = _reduced_edge_config()
        cfg.rope_scaling = {}
        model_config = DiffusionModelConfig(pretrained_config=cfg)
        model_config.attention.backend = "VANILLA"
        model = Cosmos3VFMTransformer(model_config)
        assert model.language_model.rotary_emb.mrope_section == [2, 1, 1]

    def test_qwen3_flag_contradiction_raises(self):
        cfg = SimpleNamespace(hidden_act="relu2")
        with pytest.raises(ValueError, match="hidden_act"):
            resolve_arch_recipe(cfg)

    @pytest.mark.parametrize("key,value", [("latent_channel", 4), ("latent_patch_size", 4)])
    def test_edge_latent_geometry_invariants(self, key, value):
        cfg = _reduced_edge_config()
        setattr(cfg, key, value)
        with pytest.raises(ValueError, match=key):
            resolve_arch_recipe(cfg)


class TestNemotronNormSemantics:
    """Pins the one intentional numerics fork: fp32 weight multiply, then
    downcast — vs the Qwen flavor's bf16 multiply after downcast."""

    def test_matches_fp32_weight_multiply_bit_exact(self):
        torch.manual_seed(0)
        x = (torch.randn(256, 8) * 3).bfloat16()
        weight = (torch.randn(8) * 2 + 1.5).bfloat16()

        nemotron = NemotronRMSNorm(hidden_size=8, eps=1e-5)
        qwen = Qwen3VLTextRMSNorm(hidden_size=8, eps=1e-5)
        with torch.no_grad():
            nemotron.weight.copy_(weight)
            qwen.weight.copy_(weight)

        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
        reference = (weight.float() * normed).to(torch.bfloat16)

        assert torch.equal(nemotron(x), reference)
        assert not torch.equal(qwen(x), reference)


class TestEdgeTransformerStructure:
    @pytest.fixture(scope="class")
    def model(self):
        return Cosmos3VFMTransformer(_reduced_edge_model_config())

    def test_recipe_and_flags(self, model):
        assert model.recipe is NEMOTRON_DENSE_RECIPE
        assert model.audio_gen is False
        assert model.has_action_weights is True
        assert model.temporal_compression_factor == 4
        assert model.temporal_compression_factor_declared is True

    def test_und_attention_norms(self, model):
        attn = model.language_model.layers[0].self_attn
        assert attn.norm_q is None
        assert attn.norm_k is None
        assert isinstance(attn.k_norm_und_for_gen, NemotronRMSNorm)

    def test_nemotron_norms_everywhere(self, model):
        und = model.language_model.layers[0]
        gen = model.gen_layers[0]
        for norm in (
            und.input_layernorm,
            und.post_attention_layernorm,
            und.self_attn.k_norm_und_for_gen,
            gen.input_layernorm,
            gen.post_attention_layernorm,
            gen.cross_attention.norm_q,
            gen.cross_attention.norm_k,
            model.norm_moe_gen,
        ):
            assert isinstance(norm, NemotronRMSNorm)
            assert norm.variance_epsilon == 1e-5

    def test_relu2_mlp_no_gate(self, model):
        for layer in (model.language_model.layers[0], model.gen_layers[0]):
            assert isinstance(layer.mlp, MLP)
            assert not hasattr(layer.mlp, "gate_proj")

    def test_rope_defaults_without_rope_type(self, model):
        assert model.language_model.rotary_emb.rope_type == "default"


class TestGeneratorOnlyKNorm:
    """Mirror of diffusers' Edge regression: perturbing k_norm_und_for_gen
    must not change the und causal attention output, only the gen-facing K."""

    def test_k_norm_touches_only_gen_facing_keys(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        model = Cosmos3VFMTransformer(_reduced_edge_model_config()).to(DEVICE).eval()
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0, 0.02)
        model.post_load_weights()

        attn = model.language_model.layers[0].self_attn
        hidden = torch.randn(1, 4, 32, dtype=torch.bfloat16, device=DEVICE)
        cos = torch.ones(1, 4, 1, 8, dtype=torch.bfloat16, device=DEVICE)
        sin = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16, device=DEVICE)

        with torch.inference_mode():
            out_before, k_gen_before, v_before = attn.forward_with_kv(hidden, cos, sin)
            # The rope above is identity (cos=1, sin=0), so the cached gen K
            # must be exactly the Nemotron-normed raw K.
            _, k_raw, _ = attn.get_qkv(hidden)
            k_raw = k_raw.view(1, 4, attn.local_num_key_value_heads, attn.head_dim)
            assert torch.equal(k_gen_before, attn.k_norm_und_for_gen(k_raw))

            attn.k_norm_und_for_gen.weight.fill_(2.0)
            out_after, k_gen_after, v_after = attn.forward_with_kv(hidden, cos, sin)

        assert torch.equal(out_before, out_after)
        assert torch.equal(v_before, v_after)
        assert not torch.equal(k_gen_before, k_gen_after)

    def test_nano_qk_norm_stays_byte_identical(self):
        """Regression pin for the qwen3 recipe: apply_qk_norm must keep
        today's exact ``F.rms_norm`` semantics (a future refactor routing it
        through ``Qwen3VLTextRMSNorm.forward`` would change Nano numerics)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = Cosmos3VFMTransformer(_reduced_qwen3_model_config()).to(DEVICE).eval()
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0, 0.02)
        attn = model.language_model.layers[0].self_attn

        q = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16, device=DEVICE)
        q_normed, k_normed = attn.apply_qk_norm(q, k)
        assert torch.equal(
            q_normed, F.rms_norm(q, (8,), attn.norm_q.weight, attn.norm_q.variance_epsilon)
        )
        assert torch.equal(
            k_normed, F.rms_norm(k, (8,), attn.norm_k.weight, attn.norm_k.variance_epsilon)
        )


# The relevant subset of the Edge checkpoint's scheduler config (values
# verbatim; identical to Nano's).
EDGE_UNIPC_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "num_train_timesteps": 1000,
    "flow_shift": 1.0,
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "use_karras_sigmas": True,
    "sigma_max": 200.0,
    "sigma_min": 0.147,
    "solver_order": 2,
    "solver_type": "bh2",
    "final_sigmas_type": "zero",
    "timestep_spacing": "linspace",
    "lower_order_final": True,
}

# Recorded from cosmos-framework 117c7d2 (`fm_solvers_unipc.py`
# FlowUniPCMultistepScheduler, num_train_timesteps=1000, shift=1.0,
# use_dynamic_shifting=False; set_timesteps(steps, shift=shift)). Each
# trajectory runs the synthetic velocity v = 0.05*x + 0.3*sin(t/1000) - 0.1 in
# float64 from x0 = linspace(-1, 1, 8).reshape(1, 2, 2, 2) through every
# step().
COSMOS_FRAMEWORK_FIXTURES = {
    (3.0, 10): {
        "timesteps": [999, 963, 922, 874, 817, 749, 666, 562, 428, 249],
        "sigmas": [
            0.99966645,
            0.96394110,
            0.92272168,
            0.87463522,
            0.81780970,
            0.74962479,
            0.66629612,
            0.56214833,
            0.42826521,
            0.24979164,
            0.0,
        ],
        "final": [
            -0.995152214121,
            -0.723385865647,
            -0.451619517172,
            -0.179853168698,
            0.091913179777,
            0.363679528251,
            0.635445876726,
            0.907212225200,
        ],
    },
    (10.0, 7): {
        "timesteps": [999, 983, 961, 930, 882, 799, 624],
        "sigmas": [
            0.99989992,
            0.98349357,
            0.96140891,
            0.93008101,
            0.88217115,
            0.79977584,
            0.62472641,
            0.0,
        ],
        "final": [
            -1.040962681673,
            -0.769320359241,
            -0.497678036809,
            -0.226035714377,
            0.045606608054,
            0.317248930486,
            0.588891252918,
            0.860533575350,
        ],
    },
    (5.0, 13): {
        "timesteps": [999, 983, 964, 943, 918, 888, 853, 810, 757, 689, 599, 475, 293],
        "sigmas": [
            0.99979985,
            0.98339677,
            0.96469206,
            0.94316465,
            0.91812354,
            0.88863194,
            0.85338765,
            0.81052577,
            0.75727713,
            0.68934584,
            0.59968787,
            0.47589558,
            0.29389268,
            0.0,
        ],
        "final": [
            -0.998983254680,
            -0.727228149256,
            -0.455473043832,
            -0.183717938407,
            0.088037167017,
            0.359792272441,
            0.631547377865,
            0.903302483290,
        ],
    },
}


class TestNativeFlowSchedule:
    def _native_policy_and_scheduler(self, shift: float):
        scheduler = UniPCMultistepScheduler.from_config(EDGE_UNIPC_CONFIG)
        policy = Cosmos3SamplingPolicy.from_scheduler(scheduler, native_flow_schedule=True)
        return policy, policy.with_flow_shift(scheduler, shift)

    def test_with_flow_shift_disables_karras(self):
        policy, scheduler = self._native_policy_and_scheduler(3.0)
        assert float(scheduler.config.flow_shift) == 3.0
        assert scheduler.config.use_karras_sigmas is False
        # Already matching → same instance.
        assert policy.with_flow_shift(scheduler, 3.0) is scheduler

    def test_karras_rebuilt_even_at_checkpoint_shift(self):
        """flow_shift 1.0 equals the checkpoint value, but the native flow
        schedule still requires the karras grid off."""
        policy, scheduler = self._native_policy_and_scheduler(1.0)
        assert scheduler.config.use_karras_sigmas is False

    def test_non_native_keeps_checkpoint_config(self):
        scheduler = UniPCMultistepScheduler.from_config(EDGE_UNIPC_CONFIG)
        policy = Cosmos3SamplingPolicy.from_scheduler(scheduler, native_flow_schedule=False)
        assert policy.with_flow_shift(scheduler, 1.0) is scheduler
        assert scheduler.config.use_karras_sigmas is True

    @pytest.mark.parametrize("shift,steps", sorted(COSMOS_FRAMEWORK_FIXTURES))
    def test_matches_cosmos_framework_reference(self, shift, steps):
        fixture = COSMOS_FRAMEWORK_FIXTURES[(shift, steps)]
        policy, scheduler = self._native_policy_and_scheduler(shift)
        policy.set_timesteps(scheduler, num_inference_steps=steps, device="cpu")

        assert scheduler.timesteps.tolist() == fixture["timesteps"]
        np.testing.assert_allclose(scheduler.sigmas.tolist(), fixture["sigmas"], atol=1e-6)

        x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(1, 2, 2, 2)
        for t in scheduler.timesteps:
            v = 0.05 * x + 0.3 * torch.sin(t.double() / 1000.0) - 0.1
            x = scheduler.step(v, t, x, return_dict=False)[0]
        np.testing.assert_allclose(x.flatten().tolist(), fixture["final"], atol=1e-9)


class TestStrictLoading:
    def _model(self) -> Cosmos3VFMTransformer:
        return Cosmos3VFMTransformer(_reduced_edge_model_config())

    def test_full_checkpoint_loads(self):
        cfg = _reduced_edge_config()
        sd = _edge_state_dict(cfg)
        sd["layers.0.self_attn.k_norm_und_for_gen.weight"] = torch.full((8,), 0.5)
        model = self._model()
        model.load_weights(sd)
        loaded = model.language_model.layers[0].self_attn.k_norm_und_for_gen.weight
        assert torch.equal(loaded.cpu().float(), torch.full((8,), 0.5))

    @pytest.mark.parametrize(
        "missing_key",
        [
            "layers.0.self_attn.k_norm_und_for_gen.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.1.mlp_moe_gen.down_proj.weight",
            "layers.0.input_layernorm_moe_gen.weight",
            "embed_tokens.weight",
        ],
    )
    def test_missing_weight_raises(self, missing_key):
        sd = _edge_state_dict(_reduced_edge_config())
        del sd[missing_key]
        with pytest.raises(ValueError, match="missing weights"):
            self._model().load_weights(sd)

    def test_partial_fused_qkv_raises(self):
        sd = _edge_state_dict(_reduced_edge_config())
        del sd["layers.0.self_attn.add_v_proj.weight"]
        with pytest.raises(ValueError, match="missing weights"):
            self._model().load_weights(sd)

    def test_intentional_skips_are_logged_with_names(self, monkeypatch):
        """The skip log must name the skipped tensor families (dynamic
        content, not just the logger's static category text)."""
        import tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 as tf_module

        infos = []
        monkeypatch.setattr(tf_module.logger, "info", infos.append)
        cfg = _reduced_edge_config()
        sd = _edge_state_dict(cfg)
        sd.update(
            {
                "action_modality_embed": torch.randn(cfg.hidden_size),
                "action_proj_in.fc.weight": torch.randn(4, 8),
                "action_proj_in.bias.weight": torch.randn(4, 8),
                "action_proj_out.fc.weight": torch.randn(4, 8),
                "action_proj_out.bias.weight": torch.randn(4, 8),
            }
        )
        model = self._model()
        model.load_weights(sd)

        param_names = {name for name, _ in model.named_parameters()}
        assert not any("lm_head" in name for name in param_names)
        assert "language_model.norm.weight" not in param_names
        skip_logs = [m for m in infos if "intentionally unused" in m]
        assert len(skip_logs) == 1
        # The dynamic name list follows the final colon; the static category
        # text also mentions lm_head/norm, so assert the parsed set exactly.
        skipped_families = {name.strip() for name in skip_logs[0].rsplit(": ", 1)[1].split(",")}
        assert skipped_families == {
            "action_modality_embed",
            "action_proj_in",
            "action_proj_out",
            "lm_head",
            "norm",
        }

    def test_model_prefixed_skip_keys_are_intentional(self, monkeypatch):
        """Checkpoints that namespace top-level tensors under "model." must
        have their skip keys recognized (prefix normalization runs before the
        skip check), not warned about as unknown."""
        import tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 as tf_module

        warnings = []
        infos = []
        monkeypatch.setattr(tf_module.logger, "warning", warnings.append)
        monkeypatch.setattr(tf_module.logger, "info", infos.append)
        cfg = _reduced_edge_config()
        sd = _edge_state_dict(cfg)
        sd["model.lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size)
        sd["model.action_modality_embed"] = torch.randn(cfg.hidden_size)
        self._model().load_weights(sd)

        assert not any("unknown checkpoint key" in m for m in warnings)
        skip_logs = [m for m in infos if "intentionally unused" in m]
        assert len(skip_logs) == 1
        skipped_families = {name.strip() for name in skip_logs[0].rsplit(": ", 1)[1].split(",")}
        assert {"lm_head", "action_modality_embed"} <= skipped_families

    def test_unconsumed_mapped_tensor_warns(self, monkeypatch):
        """A checkpoint tensor that remaps to a module the recipe didn't
        construct must warn, not vanish: Edge has no und norm_q, and a
        sound_gen=false model constructs no audio projections."""
        import tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 as tf_module

        warnings = []
        monkeypatch.setattr(tf_module.logger, "warning", warnings.append)
        sd = _edge_state_dict(_reduced_edge_config())
        sd["layers.0.self_attn.norm_q.weight"] = torch.ones(8)
        sd["audio_proj_in.weight"] = torch.randn(32, 4)
        self._model().load_weights(sd)

        matched = [m for m in warnings if "matched no constructed parameter" in m]
        assert len(matched) == 1
        assert "norm_q" in matched[0]
        assert "audio2llm" in matched[0]

    def test_missing_root_parameter_raises(self):
        """Nano-family audio checkpoints carry a root parameter
        (audio_modality_embed); its absence must fail, not stay random."""
        cfg = _reduced_qwen3_config(
            sound_gen=True, sound_dim=4, sound_latent_fps=25, temporal_compression_factor_sound=1
        )
        sd = _synthetic_state_dict(cfg)
        model_config = DiffusionModelConfig(pretrained_config=cfg)
        model_config.attention.backend = "VANILLA"

        Cosmos3VFMTransformer(model_config).load_weights(dict(sd))

        del sd["audio_modality_embed"]
        model_config = DiffusionModelConfig(
            pretrained_config=_reduced_qwen3_config(
                sound_gen=True,
                sound_dim=4,
                sound_latent_fps=25,
                temporal_compression_factor_sound=1,
            )
        )
        model_config.attention.backend = "VANILLA"
        with pytest.raises(ValueError, match="audio_modality_embed"):
            Cosmos3VFMTransformer(model_config).load_weights(sd)

    def test_qwen3_synthetic_checkpoint_loads(self):
        """Nano-loading regression: the gated recipe (gate/up/down + und QK
        norms, no k_norm_und_for_gen) loads cleanly through the same strict
        coverage path."""
        cfg = _reduced_qwen3_config()
        model_config = DiffusionModelConfig(pretrained_config=cfg)
        model_config.attention.backend = "VANILLA"
        model = Cosmos3VFMTransformer(model_config)
        model.load_weights(_synthetic_state_dict(cfg))
        attn = model.language_model.layers[0].self_attn
        assert attn.norm_q is not None
        assert attn.k_norm_und_for_gen is None


def _bare_pipeline(family: str) -> Cosmos3OmniMoTPipeline:
    pipeline = object.__new__(Cosmos3OmniMoTPipeline)
    pipeline.family = family
    pipeline.sampling = Cosmos3SamplingPolicy()
    pipeline.audio_gen = False
    pipeline.action_gen = False
    pipeline.has_action_weights = family == NEMOTRON_DENSE_RECIPE.name
    pipeline.use_native_flow_schedule = family == NEMOTRON_DENSE_RECIPE.name
    return pipeline


class TestEdgeDefaults:
    def test_generation_defaults_matrix(self):
        edge_video = COSMOS3_GENERATION_DEFAULTS[(NEMOTRON_DENSE_RECIPE.name, "video")]
        assert edge_video is COSMOS3_EDGE_VIDEO_PARAMS
        assert (edge_video["height"], edge_video["width"]) == (480, 832)
        assert edge_video["num_frames"] == 121
        assert edge_video["num_inference_steps"] == 50
        assert edge_video["guidance_scale"] == 5.0
        assert edge_video["flow_shift"] == 3.0

        edge_t2i = COSMOS3_GENERATION_DEFAULTS[(NEMOTRON_DENSE_RECIPE.name, "image")]
        assert edge_t2i is COSMOS3_EDGE_T2I_PARAMS
        assert (edge_t2i["height"], edge_t2i["width"]) == (640, 640)
        assert edge_t2i["guidance_scale"] == 4.0
        assert edge_t2i["guidance_interval"] is None

        assert COSMOS3_GENERATION_DEFAULTS[(QWEN3_RECIPE.name, "video")] is COSMOS3_720P_PARAMS

    def test_executor_defaults_are_edge_shaped(self):
        params = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name).default_generation_params
        assert params["num_frames"] == 121
        assert params["max_sequence_length"] == 4096
        assert "flow_shift" not in params
        for unresolved in ("height", "width", "num_inference_steps", "guidance_scale"):
            assert params[unresolved] is None

    def test_warmup_shape_per_family(self):
        edge = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name)
        assert edge.default_warmup_resolutions == [(480, 832)]
        assert edge.default_warmup_num_frames == [121]

        qwen3 = _bare_pipeline(QWEN3_RECIPE.name)
        assert qwen3.default_warmup_resolutions == [(720, 1280)]
        assert qwen3.default_warmup_num_frames == [189]

    def test_hf_id_registered(self):
        assert "nvidia/Cosmos3-Edge" in PIPELINE_REGISTRY["Cosmos3OmniMoTPipeline"].hf_ids

    def test_none_params_resolve_from_edge_tables(self):
        pipeline = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name)
        resolved = pipeline._resolve_generation_params(
            "video",
            height=None,
            width=None,
            num_frames=None,
            num_inference_steps=None,
            guidance_scale=None,
            max_sequence_length=None,
            frame_rate=None,
        )
        assert resolved == {
            "height": 480,
            "width": 832,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "max_sequence_length": 4096,
            "frame_rate": 24.0,
        }
        # Image mode falls back to the video table for fields the image table
        # omits, and explicit values always win.
        image = pipeline._resolve_generation_params(
            "image", height=None, max_sequence_length=None, num_inference_steps=8
        )
        assert image == {"height": 640, "max_sequence_length": 4096, "num_inference_steps": 8}

    def test_sampling_overrides_beat_tables(self):
        pipeline = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name)
        pipeline.sampling = Cosmos3SamplingPolicy(fixed_sigmas=(1.0, 0.5))
        resolved = pipeline._resolve_generation_params(
            "video", num_inference_steps=None, guidance_scale=None, height=None
        )
        assert resolved["num_inference_steps"] == 2
        assert resolved["guidance_scale"] == 1.0
        assert resolved["height"] == 480

    def test_warmup_runs_at_family_guidance(self, monkeypatch):
        pipeline = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name)
        captured = {}

        def fake_forward(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(pipeline, "forward", fake_forward, raising=False)
        pipeline._run_warmup(height=480, width=832, num_frames=121, steps=2)
        assert captured["guidance_scale"] == 5.0
        assert captured["max_sequence_length"] == 4096

    def test_declared_temporal_factor_must_match_vae(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            _validate_temporal_compression,
        )

        declared = SimpleNamespace(
            temporal_compression_factor=4, temporal_compression_factor_declared=True
        )
        _validate_temporal_compression(declared, 4)
        with pytest.raises(ValueError, match="temporal_compression_factor"):
            _validate_temporal_compression(declared, 8)
        # Nano-style configs don't declare it; the VAE value simply wins.
        undeclared = SimpleNamespace(
            temporal_compression_factor=4, temporal_compression_factor_declared=False
        )
        _validate_temporal_compression(undeclared, 8)


class TestSamplingRecipeMatrix:
    """Family + model_index schedule flag + scheduler recipe must be validated
    together: the three facts come from different checkpoint files, and a
    mismatch (e.g. a stale conversion missing use_native_flow_schedule) would
    otherwise sample the wrong trajectory silently."""

    BASE = Cosmos3SamplingPolicy()
    DISTILLED = Cosmos3SamplingPolicy(fixed_sigmas=(1.0, 0.5))

    @pytest.mark.parametrize(
        "family,native,sampling,error",
        [
            (QWEN3_RECIPE.name, False, BASE, None),
            (QWEN3_RECIPE.name, False, DISTILLED, None),
            (NEMOTRON_DENSE_RECIPE.name, True, BASE, None),
            (NEMOTRON_DENSE_RECIPE.name, False, BASE, "use_native_flow_schedule"),
            (QWEN3_RECIPE.name, True, BASE, "use_native_flow_schedule"),
            (NEMOTRON_DENSE_RECIPE.name, True, DISTILLED, "istilled"),
            (NEMOTRON_DENSE_RECIPE.name, False, DISTILLED, "istilled"),
        ],
    )
    def test_startup_matrix(self, family, native, sampling, error):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            _validate_sampling_recipe,
        )

        if error is None:
            _validate_sampling_recipe(family, native, sampling)
        else:
            with pytest.raises(ValueError, match=error):
                _validate_sampling_recipe(family, native, sampling)

    def test_component_loading_rejects_edge_without_native_flag(self, tmp_path):
        """End to end through load_standard_components: a UniPC Edge
        checkpoint whose model_index omits the flag must fail at load."""
        import json

        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        (scheduler_dir / "scheduler_config.json").write_text(
            json.dumps(
                {
                    "_class_name": "UniPCMultistepScheduler",
                    "num_train_timesteps": 1000,
                    "flow_shift": 1.0,
                    "prediction_type": "flow_prediction",
                    "use_flow_sigmas": True,
                    "use_karras_sigmas": True,
                    "solver_order": 2,
                }
            )
        )
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "Cosmos3OmniPipeline"})
        )

        def fresh_pipeline():
            pipeline = _bare_pipeline(NEMOTRON_DENSE_RECIPE.name)
            pipeline.default_use_system_prompt = False
            # Mirror __init__: the flag starts False and only the checkpoint's
            # model_index may turn it on.
            pipeline.use_native_flow_schedule = False
            return pipeline

        skip = ["text_tokenizer", "tokenizer", "vae", "sound_tokenizer"]
        with pytest.raises(ValueError, match="use_native_flow_schedule"):
            fresh_pipeline().load_standard_components(str(tmp_path), torch.device("cpu"), skip)

        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "Cosmos3OmniPipeline", "use_native_flow_schedule": True})
        )
        pipeline = fresh_pipeline()
        pipeline.load_standard_components(str(tmp_path), torch.device("cpu"), skip)
        assert pipeline.sampling.native_flow_schedule is True
        assert pipeline.mode_schedulers["video"].config.use_karras_sigmas is False
        assert float(pipeline.mode_schedulers["video"].config.flow_shift) == 3.0


class TestEnvelopeAdvisory:
    def _warnings(self, monkeypatch):
        records = []
        monkeypatch.setattr(pipeline_module.logger, "warning", records.append)
        return records

    def _advise(self, pipeline, **overrides):
        kwargs = dict(
            is_t2i=False,
            height=480,
            width=832,
            num_frames=121,
            frame_rate=24.0,
            max_sequence_length=4096,
        )
        kwargs.update(overrides)
        pipeline._log_envelope_advisory(**kwargs)

    def test_in_envelope_is_silent(self, monkeypatch):
        records = self._warnings(monkeypatch)
        self._advise(_bare_pipeline(NEMOTRON_DENSE_RECIPE.name))
        assert records == []

    def test_out_of_envelope_logs_once(self, monkeypatch):
        records = self._warnings(monkeypatch)
        self._advise(_bare_pipeline(NEMOTRON_DENSE_RECIPE.name), num_frames=25)
        assert len(records) == 1
        assert "num_frames=25" in records[0]

    def test_family_without_envelope_never_logs(self, monkeypatch):
        records = self._warnings(monkeypatch)
        self._advise(_bare_pipeline(QWEN3_RECIPE.name), num_frames=25, height=13, width=17)
        assert records == []


class TestDiffusersParity:
    """Per-step velocity parity against diffusers main (first release with the
    Edge classes). Runs in a subprocess because diffusers main cannot be
    imported next to the pinned diffusers; gated on DIFFUSERS_MAIN_PATH."""

    def test_per_step_velocity_parity(self):
        import re
        import subprocess
        import sys

        diffusers_main = os.environ.get("DIFFUSERS_MAIN_PATH")
        if not diffusers_main:
            pytest.skip("Set DIFFUSERS_MAIN_PATH to a diffusers checkout with Edge support")
        checkpoint = _require_edge_checkpoint()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        script = Path(__file__).parent / "cosmos3_edge_diffusers_parity.py"
        result = subprocess.run(
            [sys.executable, str(script), checkpoint],
            env={**os.environ, "DIFFUSERS_MAIN_PATH": diffusers_main},
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        rels = [float(m) for m in re.findall(r"rel=([0-9.]+)", result.stdout)]
        assert len(rels) == 2, result.stdout
        # bf16 accumulation band across 28 layers with differing attention
        # backends; observed 0.007 / 0.016 on B200.
        assert all(rel < 0.05 for rel in rels), result.stdout


# Recorded from diffusers main 2919c5096 (`Cosmos3OmniPipeline.tokenize_prompt`
# on the real Edge checkpoint): prompt "A red cube sits on a wooden table.",
# num_frames=93, height=480, width=832, fps=24.0, system prompt off
# (checkpoint default), duration + resolution templates on. Cond branch only:
# the negative branch intentionally mirrors cosmos-framework's keep-metadata
# templates, not diffusers' inverse templates.
DIFFUSERS_COND_TOKEN_GOLDEN = [
    1010,
    10,
    25708,
    1010,
    11,
    1010,
    10,
    3263,
    1010,
    1065,
    4804,
    50061,
    53048,
    1408,
    1261,
    32656,
    4234,
    1046,
    1531,
    7476,
    1395,
    1032,
    1051,
    1046,
    1057,
    12900,
    2730,
    1321,
    1395,
    1307,
    1032,
    1050,
    1052,
    1439,
    8148,
    1046,
    2409,
    7476,
    1395,
    1307,
    1032,
    1052,
    1056,
    1048,
    1120,
    1056,
    1051,
    1050,
    9617,
    1046,
    11,
    1010,
    10,
    1503,
    19464,
    1010,
    12,
    1010,
    11,
    20,
]

# Uncond (CFG) branch pin: TRT-LLM deliberately mirrors cosmos-framework's
# keep-metadata negative-prompt semantics (same duration/resolution templates
# as the positive branch), which diverges from diffusers' inverse templates —
# so this is a self-golden recorded from this code path on the real Edge
# tokenizer (empty negative prompt, num_frames=93, height=480, width=832,
# fps=24.0, system prompt off). Not yet cross-checked against
# cosmos-framework's own tokenization.
UNCOND_TOKEN_GOLDEN = [
    1010,
    10,
    25708,
    1010,
    11,
    1010,
    10,
    3263,
    1010,
    1784,
    7476,
    1395,
    1032,
    1051,
    1046,
    1057,
    12900,
    2730,
    1321,
    1395,
    1307,
    1032,
    1050,
    1052,
    1439,
    8148,
    1046,
    2409,
    7476,
    1395,
    1307,
    1032,
    1052,
    1056,
    1048,
    1120,
    1056,
    1051,
    1050,
    9617,
    1046,
    11,
    1010,
    10,
    1503,
    19464,
    1010,
    12,
    1010,
    11,
    20,
]


class TestEdgeCheckpoint:
    """Gated on the real Cosmos3-Edge checkpoint."""

    def test_tokenizer_specials_and_chat_template(self):
        from transformers import AutoTokenizer

        checkpoint = _require_edge_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="text_tokenizer")
        assert tokenizer.eos_token_id == 11
        assert tokenizer.pad_token_id == 11
        assert tokenizer.convert_tokens_to_ids("<|vision_start|>") == 20
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        assert isinstance(ids, list) and len(ids) > 0

    def test_tokenization_matches_diffusers_golden(self):
        from transformers import AutoTokenizer

        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
            COSMOS3_DURATION_TEMPLATE,
        )

        checkpoint = _require_edge_checkpoint()

        class _CpuPipeline(Cosmos3OmniMoTPipeline):
            device = property(lambda self: torch.device("cpu"))

        pipeline = object.__new__(_CpuPipeline)
        pipeline.tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="text_tokenizer")
        text = pipeline._format_prompt_with_metadata(
            "A red cube sits on a wooden table.",
            height=480,
            width=832,
            num_frames=93,
            frame_rate=24.0,
            duration_template=COSMOS3_DURATION_TEMPLATE,
            resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
        )
        ids, mask = pipeline._tokenize_prompt(text, 128, use_system_prompt=False)
        golden_length = len(DIFFUSERS_COND_TOKEN_GOLDEN)
        assert ids[0, :golden_length].tolist() == DIFFUSERS_COND_TOKEN_GOLDEN
        assert int(mask.sum()) == golden_length

        uncond_text = pipeline._format_prompt_with_metadata(
            "",
            height=480,
            width=832,
            num_frames=93,
            frame_rate=24.0,
            duration_template=COSMOS3_DURATION_TEMPLATE,
            resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
        )
        uncond_ids, uncond_mask = pipeline._tokenize_prompt(
            uncond_text, 128, use_system_prompt=False
        )
        uncond_length = len(UNCOND_TOKEN_GOLDEN)
        assert uncond_ids[0, :uncond_length].tolist() == UNCOND_TOKEN_GOLDEN
        assert int(uncond_mask.sum()) == uncond_length

    def test_model_index_detection(self):
        from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

        checkpoint = _require_edge_checkpoint()
        assert AutoPipeline._detect_from_checkpoint(checkpoint) == "Cosmos3OmniMoTPipeline"

    @pytest.fixture(scope="class")
    def edge_pipeline(self):
        checkpoint = _require_edge_checkpoint()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        args = VisualGenArgs(
            model=checkpoint,
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(
            skip_warmup=True,
            skip_components=[PipelineComponent.SOUND_TOKENIZER],
        )
        yield pipeline
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def test_recipe_and_scheduler_wiring(self, edge_pipeline):
        assert edge_pipeline.family == NEMOTRON_DENSE_RECIPE.name
        assert edge_pipeline.transformer.recipe is NEMOTRON_DENSE_RECIPE
        assert edge_pipeline.use_native_flow_schedule is True
        assert edge_pipeline.sampling.native_flow_schedule is True
        for mode in ("video", "image"):
            scheduler = edge_pipeline.mode_schedulers[mode]
            assert float(scheduler.config.flow_shift) == 3.0
            assert scheduler.config.use_karras_sigmas is False

    def test_load_weights_and_forward(self, edge_pipeline):
        transformer = edge_pipeline.transformer
        channels = transformer.latent_channel_size
        latents = torch.randn(1, channels, 1, 16, 16, dtype=torch.bfloat16, device=DEVICE)
        timestep = torch.tensor([999.0], device=DEVICE)
        text_ids = torch.randint(0, 1000, (1, 32), device=DEVICE)
        text_mask = torch.ones(1, 32, dtype=torch.long, device=DEVICE)

        transformer.reset_cache()
        with torch.inference_mode():
            out = transformer(
                hidden_states=latents,
                timestep=timestep / 1000.0,
                raw_timestep=timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=(1, 16, 16),
                fps=24.0,
            )
        assert out.video.shape == latents.shape
        assert torch.isfinite(out.video.float()).all()

    def test_direct_forward_resolves_edge_defaults(self, edge_pipeline, monkeypatch):
        """A direct forward() call with unset numerics must reach the
        generation path with Edge-table values (denoise and decode stubbed)."""
        captured = {}

        def fake_denoise(**kwargs):
            captured["latents"] = kwargs["latents"]
            captured["guidance_scale"] = kwargs["guidance_scale"]
            captured["scheduler"] = kwargs["scheduler"]
            return kwargs["latents"]

        def fake_decode(latents, decode_fn, **kwargs):
            captured["decoded"] = True
            return torch.zeros(1, 2, 8, 8, 3, dtype=torch.uint8)

        monkeypatch.setattr(edge_pipeline, "denoise", fake_denoise, raising=False)
        monkeypatch.setattr(edge_pipeline, "decode_latents", fake_decode, raising=False)

        edge_pipeline.forward(prompt="warmup-shaped request", seed=0, use_guardrails=False)

        assert captured["guidance_scale"] == 5.0
        # Edge video defaults: 121 frames -> 31 latent frames, 480x832 -> 30x52.
        assert tuple(captured["latents"].shape) == (1, 48, 31, 30, 52)
        assert captured["scheduler"] is edge_pipeline.mode_schedulers["video"]
        assert captured["scheduler"].num_inference_steps == 50
        assert captured["decoded"] is True

    def test_t2v_sanity_generation(self, edge_pipeline):
        out = edge_pipeline.forward(
            prompt="A red ball rolls across a wooden floor.",
            seed=0,
            height=192,
            width=320,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=5.0,
            use_guardrails=False,
        )
        video = out.video
        assert video is not None
        assert tuple(video.shape) == (1, 9, 192, 320, 3)
        assert video.float().std() > 1.0, "generated video is (near-)constant"

    def test_i2v_sanity_generation(self, edge_pipeline):
        from PIL import Image

        image = Image.new("RGB", (320, 192))
        for x in range(320):
            for y in range(0, 192, 4):
                image.putpixel((x, y), (x % 256, (2 * y) % 256, 120))
        out = edge_pipeline.forward(
            prompt="The scene slowly brightens.",
            seed=0,
            image=image,
            height=192,
            width=320,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=5.0,
            use_guardrails=False,
        )
        video = out.video
        assert video is not None
        assert tuple(video.shape) == (1, 9, 192, 320, 3)
        assert video.float().std() > 1.0

    def test_t2i_sanity_generation(self, edge_pipeline):
        out = edge_pipeline.forward(
            prompt="A ceramic teapot on a table.",
            seed=0,
            height=256,
            width=256,
            num_inference_steps=2,
            guidance_scale=4.0,
            use_guardrails=False,
            output_type="image",
        )
        image = out.image
        assert image is not None
        assert tuple(image.shape) == (1, 256, 256, 3)
        assert image.float().std() > 1.0
